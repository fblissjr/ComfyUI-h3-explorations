#!/usr/bin/env python3
"""Two properties of the installed comfy-kitchen build on this card: fused SwiGLU against the unfused chain, and whether token routing repeats.

Written for the day the fork took ports of two open upstream fixes
(Comfy-Org/comfy-kitchen#124 and #242; `docs/wiki/comfy_kitchen.md`, section
"What the fork adds to upstream"). It prints figures for whichever build the
interpreter imports and names that build, so the same run before and after a
rebuild is the comparison. It draws no conclusion.

  SwiGLU  `int8_linear(x, input_act="swiglu")` is meant to equal
          `int8_linear(silu(gate) * up)`. Counted two ways: the INT8 codes of
          the row quantizer (`quantize_int8_rowwise_convrot64`) with every
          16-bit gate pattern of magnitude under 1e4, under four spreads of `up`, and the
          outputs of `int8_linear` at the shape of H3's `fc2`. Zero differing
          means bit-identical. The time of the fused call is printed beside it.
  tokens  `sol_attn(..., token_aug=N)` called again with the same inputs,
          on a case built so every query group has more candidates than the
          budget ("overflow") and on random inputs ("random"). Counted: how
          many of the repeats differ from the first call at all, the largest
          relative difference, the time per call against the call without
          token routing, and at the sizes where dense attention fits, the
          relative error against it.

  widths  which activated row widths the fused quantizer serves at all. The
          CUDA backend folds the activation into the quantizer only for a
          width up to `_CONVROT_FUSED_MAX_K` that is a multiple of 256 and
          fits shared memory; any other width gets the activation applied
          first, in torch, on every build. Printed per width: outputs
          differing between fused and chain, and a hash of the fused output,
          so two builds' lines for one width can be compared by eye. The
          widths are those of the INT8 files this pack loads (`WIDTHS`).

What it cannot see: a render. The inputs are synthetic, so the share of query
groups that overflow on real H3 activations, and with it the real cost of the
retry pass, is not measured here. Times are the best of a few calls on a card
that may be holding a server's cached models.

Needs the GPU and no server. `--small` leaves out the largest shapes, for a
card whose memory is mostly held by something else.

    <comfy-venv-python> bench/probe_kitchen_swiglu_and_token_overflow.py [--small]
"""
import argparse
import importlib.metadata
import time

import torch
import torch.nn.functional as F

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cb

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--small", action="store_true", help="leave out the largest shapes")
SMALL = ap.parse_args().small

print("build:", importlib.metadata.version("comfy_kitchen"), "| torch", torch.__version__, "|",
      torch.cuda.get_device_name(0), "| small" if SMALL else "| full")
print("has tokOverflow slot:", "tokOverflow" in cb._C.sol_attn_plan(1, 4096, 2, token_aug=64))
G = 256

def sync_time(fn, n=5):
    fn(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); best = min(best, time.perf_counter() - t)
    return best * 1e3

# ---- SwiGLU: every 16-bit gate pattern against a spread of up values --------
print("\n[SwiGLU] row quantizer, every 16-bit gate value under 1e4, rows of 256, four spreads of up")
for dtype in (torch.bfloat16, torch.float16):
    bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    gate_all = bits.view(dtype)
    gate_all = gate_all[torch.isfinite(gate_all.float())]
    # keep magnitudes the row absmax can represent without inf in the product
    gate_all = gate_all[gate_all.float().abs() < 1e4]
    n = (gate_all.numel() // 256) * 256
    gate = gate_all[:n].view(-1, 256)
    torch.manual_seed(1)
    tot_neq = tot = rows_bad = 0
    for trial in range(4):
        perm = torch.randperm(n, device="cuda")
        gate_t = gate_all[:n][perm].view(-1, 256)
        up = (torch.randn(gate_t.shape, device="cuda") * (0.5, 2.0, 30.0, 1e-3)[trial]).to(dtype)
        h = torch.cat([gate_t, up], dim=-1).contiguous()
        chain = (F.silu(gate_t) * up).contiguous()
        cq, cs = cb.quantize_int8_rowwise_convrot64(chain, G)
        fq, fs = cb.quantize_int8_rowwise_convrot64(h, G, input_act="swiglu")
        tot_neq += (cq != fq).sum().item(); tot += cq.numel(); rows_bad += (cq != fq).any(-1).sum().item()
    print(f"  {str(dtype):15s}: INT8 codes differing {tot_neq}/{tot}, rows touched {rows_bad}")

print("\n[SwiGLU] H3 fc2 shape, int8_linear fused vs chain output")
torch.manual_seed(0)
for dtype in (torch.bfloat16, torch.float16):
    for m in ((1024, 4096) if SMALL else (1024, 4096, 32700)):
        k, nout = 14336, 5376
        h = torch.randn(m, 2 * k, dtype=dtype, device="cuda") * 2.0
        w = torch.randint(-127, 127, (nout, k), dtype=torch.int8, device="cuda")
        ws = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        gate, up = h.chunk(2, dim=-1)
        ref = ck.int8_linear((F.silu(gate) * up).contiguous(), w, ws, None, dtype, convrot=True, convrot_groupsize=G)
        got = ck.int8_linear(h, w, ws, None, dtype, convrot=True, convrot_groupsize=G, input_act="swiglu")
        neq = (ref != got).sum().item()
        ms = sync_time(lambda: ck.int8_linear(h, w, ws, None, dtype, convrot=True, convrot_groupsize=G, input_act="swiglu"))
        print(f"  {str(dtype):15s} M={m:6d}: outputs differing {neq}/{ref.numel()}, fused call {ms:.2f} ms")
        del h, w, ref, got
        torch.cuda.empty_cache()

# ---- which widths the fused quantizer serves -------------------------------
import hashlib  # noqa: E402

#: activated row width (the linear's in-features) -> the layer in core that has it
WIDTHS = {
    14336: "H3 DiT MLP fc2 (comfy/ldm/minimax/model.py::MLP)",
    8192: "H3 video VAE decoder FeedForward w2 (comfy/ldm/minimax/vae.py::FeedForward)",
    9728: "Qwen3-VL 4B encoder MLP down_proj (comfy/text_encoders/llama.py::MLP)",
    6144: "Qwen3.5 9B upsampler MLP down_proj",
    25600: "Qwen3-VL 32B encoder MLP down_proj",
}
print("\n[widths] bfloat16, int8_linear fused vs chain, and a hash of the fused output; fused limit",
      getattr(cb, "_CONVROT_FUSED_MAX_K", "?"))
for k, where in WIDTHS.items():
    for m in (512, 4096):
        g = torch.Generator(device="cuda").manual_seed(k + m)
        h = (torch.randn(m, 2 * k, device="cuda", generator=g) * 2.0).to(torch.bfloat16)
        w = torch.randint(-127, 127, (256, k), dtype=torch.int8, device="cuda", generator=g)
        ws = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        gate, up = h.chunk(2, dim=-1)
        ref = ck.int8_linear((F.silu(gate) * up).contiguous(), w, ws, None, torch.bfloat16, convrot=True, convrot_groupsize=G)
        got = ck.int8_linear(h, w, ws, None, torch.bfloat16, convrot=True, convrot_groupsize=G, input_act="swiglu")
        digest = hashlib.sha256(got.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:12]
        print(f"  K={k:6d} M={m:5d}: outputs differing {(ref != got).sum().item():8d}/{ref.numel()}, fused output {digest} | {where}")
        del h, w, ref, got
    torch.cuda.empty_cache()

# ---- token routing ------------------------------------------------------------
HD = 128
def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()
def dense(q, k, v):
    return F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2)

def overflow_case(t, h_):
    c = 30.0
    g = torch.Generator(device="cuda").manual_seed(0)
    u = torch.randn(h_, HD, device="cuda", generator=g); u = u / u.norm(dim=-1, keepdim=True)
    sign = torch.where(torch.arange(t, device="cuda") % 2 == 0, 1.0, -1.0)
    q = (c * u)[None].expand(t, h_, HD); k = (c * u)[None] * sign[:, None, None]
    v = torch.randn(t, h_, HD, device="cuda", generator=g)
    return tuple(x[None].to(torch.bfloat16).contiguous() for x in (q, k, v))

def random_case(t, h_):
    g = torch.Generator(device="cuda").manual_seed(3)
    return tuple(torch.randn(1, t, h_, HD, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))

print("\n[token routing] repeatability over 12 repeats, error against dense attention, time per call")
for name, make in (("overflow", overflow_case), ("random", random_case)):
    for t, h_ in (((4096, 2), (16384, 8)) if SMALL else ((4096, 2), (16384, 8), (36445, 56))):
        q, k, v = make(t, h_)
        base_ms = sync_time(lambda: ck.sol_attn(q, k, v, tau=4.0), n=3)
        for aug in (64, 256):
            out = ck.sol_attn(q, k, v, tau=4.0, token_aug=aug)
            diff = [rel(ck.sol_attn(q, k, v, tau=4.0, token_aug=aug), out) for _ in range(12)]
            ms = sync_time(lambda: ck.sol_attn(q, k, v, tau=4.0, token_aug=aug), n=3)
            line = (f"  {name:8s} T={t:6d} H={h_:3d} aug={aug:3d}: {sum(d != 0 for d in diff):2d}/12 repeats differ (max {max(diff):.2e}), "
                    f"{ms:8.2f} ms (no tokens {base_ms:8.2f} ms)")
            if t <= 16384:
                ref = dense(q, k, v)
                line += f", error vs dense {rel(out, ref):.4f} (no tokens {rel(ck.sol_attn(q, k, v, tau=4.0), ref):.4f})"
            print(line)
        del q, k, v
        torch.cuda.empty_cache()
