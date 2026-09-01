#!/usr/bin/env python3
"""Grade the chunked-producer forward before it is given a render.

The node's claim is a memory lever with the direct path's numerics: feed Sol
from chunks of the projection so Q, K, V are never built, applying the same
fused norm-and-rope core applies, and route the way the direct path routes.
Every case below is a comparison against the direct path on identical
weights and inputs; nothing here judges a render.

Claims, i.e. what breaks if a case is deleted:

  agrees_with_the_direct_path
      on an H3-shaped attention module with random weights, the delegate's
      output against `out_proj(sol_attn(q, k, v))` where q, k come from
      core's own fused norm-and-rope on the full projection. Not bitwise --
      the producer thresholds on the PREVIOUS step's K-mean -- so cosine is
      the bar, at the value the kernel-vs-eager check uses, and the distance
      is printed. A wrong chunk layout, a wrong rope table, a wrong scale or
      a doubled rope would all land far below the bar.
  ragged_and_first_call_agree_too
      the same at a length that is not a multiple of the chunk or the block,
      and on the module's FIRST call, where the producer bootstraps its
      statistics by running twice.
  peak_memory_is_lower
      `torch.cuda.max_memory_allocated` over the delegate's forward is below
      the direct path's on the same tensors. This is the lever; a delegate
      that saved nothing would be pure risk.
  taken_through_the_gate_and_recorded
      through Sol's composition wrapper with the delegate published: a call
      the gate takes runs the delegate and not the foreign forward, is
      recorded as `sol_chunked` with counts and `path: chunked_delegate`;
      a call the gate declines runs the foreign forward; a block in
      `dense_blocks` runs the stock forward so Sol's override can route it.
  refuses_non_h3_input
      no rope table, or 3-D input, or a non-CUDA tensor: the stock forward
      runs, the producer is never called.
  counts_match_the_direct_path_closely
      the `sol_chunked` row's counts against the direct path's at the same
      tau: bounded, and the agreement fraction printed.

Needs CUDA and a comfy_kitchen whose `sol_attn_chunked` takes `blk_cnt`;
exits 2 otherwise.

    uv run --active --no-sync python bench/check_sol_chunked.py
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

FAILED: list[str] = []


def check(name, fn):
    try:
        fn()
        print(f"  ok    {name}")
    except AssertionError as exc:
        FAILED.append(name)
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:                          # noqa: BLE001
        FAILED.append(name)
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")


def main() -> int:
    try:
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.backends import cuda as ck_cuda
        from _live_sol import _member, live_sol, sol_observe
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 2
    if "blk_cnt" not in inspect.signature(ck_cuda.sol_attn_chunked).parameters:
        print("SKIP: the installed sol_attn_chunked has no blk_cnt; rebuild from the branch")
        return 2
    node = live_sol()
    obs = sol_observe()
    chunked = _member("sol_chunked_h3")

    torch.manual_seed(0)
    H, D = 4, 128
    hidden = 512

    class Attn(torch.nn.Module):
        """H3's Attention surface: qkv_proj (q | k | v blocks), per-head RMS
        norms, out_proj, heads, head_dim. Random bf16 weights."""
        def __init__(self):
            super().__init__()
            self.heads, self.head_dim = H, D
            self.qkv_proj = torch.nn.Linear(hidden, 3 * H * D, bias=False, dtype=torch.bfloat16, device="cuda")
            self.q_norm = torch.nn.RMSNorm(D, eps=1e-6, dtype=torch.bfloat16, device="cuda")
            self.k_norm = torch.nn.RMSNorm(D, eps=1e-6, dtype=torch.bfloat16, device="cuda")
            self.out_proj = torch.nn.Linear(H * D, hidden, bias=False, dtype=torch.bfloat16, device="cuda")
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(0.5)
                self.q_norm.weight.fill_(1.0).add_(torch.randn_like(self.q_norm.weight) * 0.1)
                self.k_norm.weight.fill_(1.0).add_(torch.randn_like(self.k_norm.weight) * 0.1)
            # loaded ComfyUI weights never require grad, and comfy-kitchen's
            # in-place rope refuses any input that does
            for p in self.parameters():
                p.requires_grad_(False)
            self.stock_calls = 0

        def forward(self, x, rope_freqs=None, transformer_options=None):
            self.stock_calls += 1
            if rope_freqs is None or not torch.is_tensor(x) or x.ndim != 2 or x.device.type != "cuda":
                return torch.zeros(1)          # a stand-in for calls core would handle its own way
            return direct(self, x, rope_freqs)

    def rope_table(t, rot=64):
        angles = torch.randn(t, rot, device="cuda") * 0.5
        half = rot // 2
        ang = angles[:, :half]
        c, s = torch.cos(ang), torch.sin(ang)
        return torch.stack([c, -s, s, c], dim=-1).reshape(1, t, 1, half, 2, 2).to(torch.bfloat16)

    def direct(module, x, rope, tau=1.0, sinks=((0, 0), (0, 0)), blk_cnt=None):
        """Core's forward: full projection, fused norm+rope, then the direct kernel."""
        s = x.shape[0]
        q, k, v = module.qkv_proj(x).split(H * D, dim=-1)
        q = q.contiguous().view(1, s, H, D)
        k = k.contiguous().view(1, s, H, D)
        v = v.contiguous().view(1, s, H, D)
        rot = rope.shape[-3] * 2
        ck.rms_rope_split_half_(q, k, rope, module.q_norm.weight, module.k_norm.weight,
                                epsilon=module.q_norm.eps, rot_dim=rot)
        out = ck.sol_attn(q, k, v, tau=tau, sink_blocks=list(sinks[0]), sink_q=list(sinks[1]), blk_cnt=blk_cnt)
        return module.out_proj(out.view(s, H * D))

    def cosine(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float(a @ b / (a.norm() * b.norm()))

    def settings(**kw):
        base = {"tau": 1.0, "topk_ratio": 0.0, "tail": True, "min_tokens": 64,
                "dense_blocks": [], "sink_conditioning": "off", "n_blocks": 50}
        base.update(kw)
        return base

    def opts(t, block=7, sigma=1.0, **extra):
        o = {"sigmas": torch.tensor([sigma]), "sample_sigmas": torch.tensor([2.0, 1.0, 0.5, 0.0]),
             "sol_block": block, "sol_compose": {"sigma_start": 10.0, "sigma_end": 0.1,
                                                 "min_tokens": 64, "settings": settings()}}
        o.update(extra)
        return o

    BAR = 0.998   # the bar check_solattn_correctness.py holds the kernel to against eager
    tmp = Path(tempfile.mkdtemp(prefix="sol_chunked_"))

    def agrees_with_the_direct_path():
        obs.arm(None)
        m = Attn()
        t = 4096
        x = torch.randn(t, hidden, device="cuda", dtype=torch.bfloat16)
        rope = rope_table(t)
        fwd = chunked.make_chunked_forward(chunk_rows=1024)
        fwd(m, x, rope_freqs=rope, transformer_options=opts(t))       # bootstrap call
        got = fwd(m, x, rope_freqs=rope, transformer_options=opts(t))
        want = direct(m, x, rope)
        c = cosine(got, want)
        assert m.stock_calls == 0, "the delegate ran the stock forward"
        assert c > BAR, f"cos {c:.6f} against the direct path"
        print(f"        cos {c:.6f} vs direct (second call, statistics from the first)")

    def ragged_and_first_call_agree_too():
        obs.arm(None)
        m = Attn()
        t = 4096 + 100                                   # ragged at the chunk and the 64-block
        x = torch.randn(t, hidden, device="cuda", dtype=torch.bfloat16)
        rope = rope_table(t)
        fwd = chunked.make_chunked_forward(chunk_rows=1024)
        got = fwd(m, x, rope_freqs=rope, transformer_options=opts(t))   # FIRST call: bootstrap
        want = direct(m, x, rope)
        c = cosine(got, want)
        assert got.shape == want.shape and c > BAR, f"cos {c:.6f} on the first, ragged call"
        print(f"        cos {c:.6f} vs direct on the first call at T={t}")

    def peak_memory_is_lower():
        obs.arm(None)
        m = Attn()
        t = 32768
        x = torch.randn(t, hidden, device="cuda", dtype=torch.bfloat16)
        rope = rope_table(t)
        fwd = chunked.make_chunked_forward(chunk_rows=4096)
        fwd(m, x, rope_freqs=rope, transformer_options=opts(t))       # bootstrap outside the measurement
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        direct(m, x, rope)
        torch.cuda.synchronize()
        peak_direct = torch.cuda.max_memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fwd(m, x, rope_freqs=rope, transformer_options=opts(t))
        torch.cuda.synchronize()
        peak_chunked = torch.cuda.max_memory_allocated() - base
        print(f"        peak over the call at T={t} H={H}: direct {peak_direct / 2**20:.0f} MiB, "
              f"chunked {peak_chunked / 2**20:.0f} MiB")
        assert peak_chunked < peak_direct, "the chunked path did not reduce the peak"

    def taken_through_the_gate_and_recorded():
        d = tmp / "gate"
        d.mkdir()
        obs.arm(f"dir={d}")
        m = Attn()
        t = 2048
        x = torch.randn(t, hidden, device="cuda", dtype=torch.bfloat16)
        rope = rope_table(t)
        seen = {"foreign": 0}

        def foreign(xx, rope_freqs=None, transformer_options=None):  # noqa: ARG001
            seen["foreign"] += 1
            return xx

        wrapped = node._compose_module_patch(m, foreign)
        delegate = chunked.make_chunked_forward(chunk_rows=1024)
        o = opts(t, block=7, sol_take_forward=delegate)
        out = wrapped(x, rope_freqs=rope, transformer_options=o)          # taken: delegate
        assert seen["foreign"] == 0 and m.stock_calls == 0 and out.shape == (t, hidden)
        assert o.get("h3_attn_route") == "sol_chunked"
        wrapped(x, rope_freqs=rope, transformer_options=opts(t, block=7, sigma=20.0, sol_take_forward=delegate))   # declined
        assert seen["foreign"] == 1
        o_dense = opts(t, block=3, sol_take_forward=delegate)
        o_dense["sol_compose"]["settings"] = settings(dense_blocks=[3])
        wrapped(x, rope_freqs=rope, transformer_options=o_dense)          # dense block: stock forward
        assert m.stock_calls == 1, m.stock_calls
        rows = obs.read_rows(sorted(d.glob("sol_observe_*.jsonl"))[0])
        calls = [r for r in rows if r["kind"] == "call"]
        assert [r["route"] for r in calls] == ["sol_chunked", "composed_patch"], [r["route"] for r in calls]
        r = calls[0]
        assert r["path"] == "chunked_delegate" and r["block"] == 7 and r["H"] == H and r["T"] == t
        assert r["kernel_density"] and r["raw"]["shape"] == [1, H, (t + 63) // 64]
        assert r["peak_alloc_bytes"] and r["peak_alloc_bytes"] > 0
        obs.arm(None)

    def refuses_non_h3_input():
        obs.arm(None)
        m = Attn()
        t = 1024
        x = torch.randn(t, hidden, device="cuda", dtype=torch.bfloat16)
        fwd = chunked.make_chunked_forward(chunk_rows=1024)
        fwd(m, x, rope_freqs=None, transformer_options=opts(t))
        fwd(m, x.unsqueeze(0), rope_freqs=rope_table(t), transformer_options=opts(t))
        fwd(m, x.cpu(), rope_freqs=rope_table(t), transformer_options=opts(t))
        assert m.stock_calls == 3, m.stock_calls

    def counts_match_the_direct_path_closely():
        d = tmp / "counts"
        d.mkdir()
        obs.arm(f"dir={d}")
        m = Attn()
        t = 4096
        x = torch.randn(t, hidden, device="cuda", dtype=torch.bfloat16)
        rope = rope_table(t)
        fwd = chunked.make_chunked_forward(chunk_rows=1024)
        fwd(m, x, rope_freqs=rope, transformer_options=opts(t))
        fwd(m, x, rope_freqs=rope, transformer_options=opts(t))
        jsonl = sorted(d.glob("sol_observe_*.jsonl"))[0]
        rows = [r for r in obs.read_rows(jsonl) if r["kind"] == "call"]
        got = obs.read_raw(jsonl, rows[-1])
        n = (t + 63) // 64
        want = torch.empty(1, H, n, dtype=torch.int32, device="cuda")
        direct(m, x, rope, blk_cnt=want)
        want = want.cpu()
        assert 1 <= int(got.min()) and int(got.max()) <= n
        agree = float((got == want).float().mean())
        print(f"        counts: {agree:.3f} of rows equal the direct path, max |diff| {int((got - want).abs().max())}")
        assert agree > 0.9, agree
        obs.arm(None)

    print("chunked-producer forward against the direct path, installed kernel:")
    print(f"  H={H} D={D} hidden={hidden}\n")
    try:
        # ComfyUI samples under inference mode; comfy-kitchen's in-place rope
        # refuses autograd tensors, and nn.Linear outputs carry grad otherwise.
        with torch.inference_mode():
            for fn in (agrees_with_the_direct_path, ragged_and_first_call_agree_too, peak_memory_is_lower,
                       taken_through_the_gate_and_recorded, refuses_non_h3_input,
                       counts_match_the_direct_path_closely):
                check(fn.__name__, fn)
    finally:
        obs.arm(None)
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
