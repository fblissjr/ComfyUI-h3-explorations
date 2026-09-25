"""Fused-head fidelity across PDD implementations, CPU only.

Record: bench/results/2026-09-25_upstream_pdd_comparison.md. Run with the ComfyUI venv and
CUDA_VISIBLE_DEVICES="" ; set H3_KIJAI_PDD_UPSTREAM_DIR (see below).

Reference: fp64 fusion of alibaba-pai's raw head stack (the vendor's
`pdd_sampling_plan`, i.e. dt-weighted mean over the block of the shifted
32-interval grid), per stream (video shift 12, audio shift 3).

Every implementation is REIMPLEMENTED here from its source; nothing is imported
from coderef/ or from the repo.

Metrics per (impl, partition, stream, block):
  rel_w      ||W - W_ref||_F / ||W_ref||_F
  rel_corr   ||W - W_ref||_F / ||W_ref - W_base||_F   (error against the distilled correction)
  maxabs_w   max |W - W_ref|
  rel_b_corr same as rel_corr for the bias
  rel_out    ||y - y_ref|| / ||y_ref - y_base|| on a fixed random input (2048 x 5376, N(0,1))
Reported as the worst block of the partition.
"""
import json, os, struct, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)
REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

VENDOR = REPO / "coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/MiniMax-H3-FL2VA-Acc-8Step.safetensors"
OURS = COMFY / "models/loras" / h3_config.PDD_FL2VA_LORA
CKPT = COMFY / "models/diffusion_models" / h3_config.MODELS["unet_fl2va"]
# Kijai's reshape-encoded conversion, as symlinked into this install on 2026-08-26
# (HF Kijai/MiniMax-H3-experimental, the pre-2026-08-27T21:25Z upload).
KIJAI_LOCAL = Path(os.environ.get("H3_KIJAI_PDD", COMFY / "models/loras/h3/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors"))
# The current upstream file (HF commit f94b1bcc94, LFS oid 71347e2725...), range-fetched:
# header.json is the safetensors header; part1.bin and part2.bin hold the byte
# ranges of the data section that carry the final_layer tensors (P2OFF below).
KIJAI_UP = Path(os.environ["H3_KIJAI_PDD_UPSTREAM_DIR"])
OUT = Path(os.environ.get("OUT", REPO / "bench/results/2026-09-25_pdd_head_fusion.json"))


N = 32
SHIFT = {"video": 12.0, "audio": 3.0}
VKEY = {"video": "proj_out", "audio": "audio_proj_out"}
CKEY = {"video": "video_out", "audio": "audio_out"}
PARTS = {
    "u8 [4]x8": list(range(0, 33, 4)),
    "u4 [8]x4": list(range(0, 33, 8)),
    "tail6 [8,8,4,4,4,4]": [0, 8, 16, 20, 24, 28, 32],
}


def shifted(sh, s):
    return sh * s / (1 + (sh - 1) * s)


def dt_grid(sh):
    s = torch.linspace(1.0, 0.0, N + 1, dtype=torch.float64)
    return (1.0 - shifted(sh, s)).diff()           # vendor pdd_time_grid(...).diff()


def sigma_grid(sh):                                 # sglang tools/fuse_...::sigma_grid(n+1, shift)
    base = torch.linspace(1.0, 0.0, N + 1, dtype=torch.float64)
    return sh * base / (1 + (sh - 1) * base)


# ---------------------------------------------------------------- inputs
with safe_open(VENDOR, "pt") as f:
    stack = {st: (f.get_tensor(f"{VKEY[st]}.weight"), f.get_tensor(f"{VKEY[st]}.bias")) for st in SHIFT}
with safe_open(CKPT, "pt") as f:
    base = {st: (f.get_tensor(f"final_layer.{CKEY[st]}.weight").float(),
                 f.get_tensor(f"final_layer.{CKEY[st]}.bias").float()) for st in SHIFT}
with safe_open(OURS, "pt") as f:
    ourbank = {st: (f.get_tensor(f"h3_pdd.bank.{st}.weight"), f.get_tensor(f"h3_pdd.bank.{st}.bias")) for st in SHIFT}
    our_base_video = f.get_tensor("h3_pdd.base_video_out")
checks = {
    "our_bank_bitwise_equals_vendor_stack": all(torch.equal(ourbank[st][i], stack[st][i]) for st in SHIFT for i in (0, 1)),
    "vendor_dtype": str(stack["video"][0].dtype),
    "our_base_video_out_equals_ckpt": bool(torch.equal(our_base_video.float(), base["video"][0])),
}


def kijai_decode_local(st):
    """core comfy/weight_adapter/lora.py::calculate_weight on the reshape path:
    pad(base, reshape) + strength*alpha*(up @ down), intermediate fp32, alpha None -> 1."""
    with safe_open(KIJAI_LOCAL, "pt") as f:
        p = f"diffusion_model.final_layer.{CKEY[st]}"
        up, down = f.get_tensor(f"{p}.lora_up.weight").float(), f.get_tensor(f"{p}.lora_down.weight").float()
        bup, bdown = f.get_tensor(f"{p}.bias.lora_up.weight").float(), f.get_tensor(f"{p}.bias.lora_down.weight").float()
        rs, brs = f.get_tensor(f"{p}.reshape_weight").tolist(), f.get_tensor(f"{p}.bias.reshape_weight").tolist()
    return _decode(st, up, down, bup, bdown, rs, brs)


def kijai_decode_upstream(st):
    h = json.loads((KIJAI_UP / "header.json").read_text())
    p1 = (KIJAI_UP / "part1.bin").read_bytes(); p2 = (KIJAI_UP / "part2.bin").read_bytes()
    P2OFF = 1665056824

    def get(key):
        e = h[key]; a, b = e["data_offsets"]
        buf = p1[a:b] if b <= len(p1) else p2[a - P2OFF:b - P2OFF]
        dt = {"F32": torch.float32, "BF16": torch.bfloat16, "I64": torch.int64}[e["dtype"]]
        return torch.frombuffer(bytearray(buf), dtype=dt).reshape(e["shape"])
    p = f"diffusion_model.final_layer.{CKEY[st]}"
    return _decode(st, get(f"{p}.lora_up.weight").float(), get(f"{p}.lora_down.weight").float(),
                   get(f"{p}.bias.lora_up.weight").float(), get(f"{p}.bias.lora_down.weight").float(),
                   get(f"{p}.reshape_weight").tolist(), get(f"{p}.bias.reshape_weight").tolist())


def _decode(st, up, down, bup, bdown, rs, brs):
    bw, bb = base[st]
    W = torch.zeros(rs, dtype=torch.float32); W[:bw.shape[0]] = bw
    W += (up @ down).reshape(W.shape).float()
    B = torch.zeros(brs, dtype=torch.float32); B[:bb.shape[0]] = bb
    B += (bup @ bdown).reshape(B.shape).float()
    out_f = bw.shape[0]
    return W.reshape(N, out_f, -1), B.reshape(N, out_f)


def classify_bank(rows_w, st):
    """Are rows 1..31 absolute heads, or deltas from head 0?"""
    hw = stack[st][0].double()
    r = rows_w.double()
    abs_err = ((r[1:] - hw[1:]).norm() / hw[1:].norm()).item()
    delta_err = ((r[1:] - (hw[1:] - hw[:1])).norm() / (hw[1:] - hw[:1]).norm()).item()
    row0_err = ((r[0] - hw[0]).norm() / hw[0].norm()).item()
    norms = r.flatten(1).norm(dim=1)
    return {"row0_vs_head0_rel": row0_err, "rows1_31_vs_absolute_heads_rel": abs_err,
            "rows1_31_vs_delta_from_head0_rel": delta_err,
            "row_norm_min_over_row0": float(norms[1:].min() / norms[0])}


# ---------------------------------------------------------------- implementations
def ref(st, a, b):
    d = dt_grid(SHIFT[st]); w = d[a:b] / d[a:b].sum()
    W = torch.tensordot(w, stack[st][0][a:b].double(), dims=([0], [0]))
    B = torch.tensordot(w, stack[st][1][a:b].double(), dims=([0], [0]))
    return W, B


def ours(st, a, b):
    """pdd_math.fuse_block + pdd_lora._FusedHeads.get at strength 1.0."""
    bank_w, bank_b = ourbank[st][0].to(torch.float32), ourbank[st][1].to(torch.float32)
    bw, bb = base[st]

    def fuse_block(s):
        d = dt_grid(SHIFT[st]); plan = torch.zeros(N, dtype=torch.float64)
        plan[a:b] = d[a:b] / d[a:b].sum()
        return torch.tensordot(plan[a:b], s[a:b].to(torch.float64), dims=([0], [0])).to(torch.float32)
    w, bias = fuse_block(bank_w), fuse_block(bank_b)
    return bw + 1.0 * (w - bw), bb + 1.0 * (bias - bb)       # fp32 master; input is fp32 so no further cast


def vendor_as_run(st, a, b):
    """minimax_h3_pdd.MiniMaxH3ParallelHead.forward with fp32 parameters (diffusers keeps proj_out fp32)."""
    d = dt_grid(SHIFT[st]); plan = torch.zeros(1, N, dtype=torch.float64)
    plan[0, a:b] = d[a:b] / d[a:b].sum()
    plan = plan.float()                                   # _PDDStepArm.arm(...).float(), then .to(weight.dtype)
    Wp, Bp = stack[st][0].float(), stack[st][1].float()   # bf16 loaded into fp32 parameters
    return torch.einsum("pn,noi->poi", plan, Wp)[0], torch.einsum("pn,no->po", plan, Bp)[0]


def core_formula(rows_w, rows_b, st, a, b):
    """comfy/ldm/minimax/model.py::_pdd_head, weight already fp32 (the fp32 island)."""
    grid = torch.linspace(1.0, 0.0, N + 1, dtype=torch.float64)
    sh = SHIFT[st]
    dt = (1.0 - sh * grid / (1.0 + (sh - 1.0) * grid)).diff()[a:b]
    w = (dt / dt.sum()).to(torch.float32)
    first = max(a, 1)
    W = rows_w[0] + torch.einsum("n,noi->oi", w[first - a:], rows_w[first:b])
    B = rows_b[0] + torch.einsum("n,no->o", w[first - a:], rows_b[first:b])
    return W, B


def t8_encode(bank):
    """T8 pdd_advanced._encode_native_pdd_head_bank: fp32 subtraction, back to bf16."""
    absolute = bank.to(torch.float32)
    return torch.cat((absolute[:1], absolute[1:] - absolute[:1]), dim=0).to(bank.dtype).to(torch.float32)


def exact_delta(bank):
    absolute = bank.to(torch.float32)
    return torch.cat((absolute[:1], absolute[1:] - absolute[:1]), dim=0)


def sglang(st, a, b, plan_bf16=True, store_bf16=True):
    """tools/fuse_minimax_h3_pdd_heads.py::fuse, one uniform block; runtime casts to h.dtype (fp32)."""
    sig = sigma_grid(SHIFT[st]); dsig = (sig[:-1] - sig[1:]).clamp(min=0)
    w = dsig[a:b]; w = w / w.sum()
    if plan_bf16:
        w = w.to(torch.bfloat16)
    out = []
    for heads in stack[st]:
        chunk = heads[a:b].float()
        fused = (chunk * w.float().view(-1, *([1] * (chunk.dim() - 1)))).sum(0)
        out.append(fused.to(torch.bfloat16).float() if store_bf16 else fused)
    return out[0], out[1]


def t8_fallback(st, a, b):
    """T8 pdd_advanced._fuse_head_bank: plan cast to the bank dtype (bf16), bf16 einsum, bf16 result."""
    d = dt_grid(SHIFT[st]); plan = torch.zeros(N, dtype=torch.float64)
    plan[a:b] = d[a:b] / d[a:b].sum()
    plan = plan.to(torch.bfloat16)
    W = torch.einsum("n,noi->oi", plan, stack[st][0]); B = torch.einsum("n,no->o", plan, stack[st][1])
    return W, B        # bf16


def utilscollection(st, a, b):
    """helpers/patcher_helpers.py::fuse_pdd_heads: fp64 sum, stored fp32 (sigma knots with [0]=1,[-1]=0 pinned)."""
    sh = SHIFT[st]
    times = torch.linspace(1.0, 0.0, N + 1, dtype=torch.float64)
    vals = shifted(sh, times); vals[0] = 1.0; vals[-1] = 0.0
    fine = [float(v) for v in vals]
    iv = [fine[i] - fine[i + 1] for i in range(N)]
    span = sum(iv[i] for i in range(a, b))
    W = sum((iv[i] / span) * stack[st][0][i].double() for i in range(a, b)).to(torch.float32)
    B = sum((iv[i] / span) * stack[st][1][i].double() for i in range(a, b)).to(torch.float32)
    return W, B


# ---------------------------------------------------------------- run
H = torch.randn(2048, 5376, dtype=torch.float32)
H64 = H.double()


def metrics(st, a, b, W, B, hidden_bf16=False):
    Wr, Br = ref(st, a, b)
    bw, bb = base[st]
    Wd, Bd = W.double(), B.double()
    corr_w = (Wr - bw.double()).norm(); corr_b = (Br - bb.double()).norm()
    yr = H64 @ Wr.T + Br
    yb = H64 @ bw.double().T + bb.double()
    if hidden_bf16:
        y = F.linear(H.to(torch.bfloat16), W.to(torch.bfloat16), B.to(torch.bfloat16)).double()
    else:
        y = (H64 @ Wd.T + Bd) if W.dtype == torch.float64 else F.linear(H, W.float(), B.float()).double()
    return {"rel_w": float((Wd - Wr).norm() / Wr.norm()),
            "rel_corr": float((Wd - Wr).norm() / corr_w),
            "maxabs_w": float((Wd - Wr).abs().max()),
            "rel_b_corr": float((Bd - Br).norm() / corr_b),
            "rel_out": float((y - yr).norm() / (yr - yb).norm())}


results = {"checks": checks, "bank_encoding": {}, "fidelity": {}}
decoded = {}
for st in SHIFT:
    lw, lb = kijai_decode_local(st)
    uw, ub = kijai_decode_upstream(st)
    decoded[st] = {"kijai_local": (lw, lb), "kijai_upstream": (uw, ub)}
    results["bank_encoding"][st] = {"kijai_local_2026-08-27_1fd13b89": classify_bank(lw, st),
                                    "kijai_upstream_f94b1bcc94": classify_bank(uw, st)}
    # upstream says "base-relative ... strength interpolates toward base": row0 = base + (head0-base)
    results["bank_encoding"][st]["kijai_upstream_row0_vs_head0_with_strength1"] = float(
        (uw[0].double() - stack[st][0][0].double()).norm() / stack[st][0][0].double().norm())

IMPLS = {
    "ours (fp64 fuse, fp32 master, base+1*(w-base))": lambda st, a, b: ours(st, a, b),
    "vendor as run (fp32 params, fp32 plan)": lambda st, a, b: vendor_as_run(st, a, b),
    "UtilsCollection (fp64 -> fp32)": lambda st, a, b: utilscollection(st, a, b),
    "core _pdd_head on exact fp32 deltas (formula control)": lambda st, a, b: core_formula(*[exact_delta(x) for x in stack[st]], st, a, b),
    "core _pdd_head on T8 encoding (bf16 deltas) = T8 native-core route": lambda st, a, b: core_formula(*[t8_encode(x) for x in stack[st]], st, a, b),
    "core _pdd_head on Kijai UPSTREAM file (f94b1bcc94), strength 1": lambda st, a, b: core_formula(*decoded[st]["kijai_upstream"], st, a, b),
    "core _pdd_head on Kijai LOCAL copy (1fd13b89, stale)": lambda st, a, b: core_formula(*decoded[st]["kijai_local"], st, a, b),
    "sglang as shipped (plan bf16, fp32 sum, store bf16)": lambda st, a, b: sglang(st, a, b, True, True),
    "sglang plan-rounding only (store fp32)": lambda st, a, b: sglang(st, a, b, True, False),
    "sglang storage only (fp64 plan, store bf16)": lambda st, a, b: sglang(st, a, b, False, True),
    "T8 fallback (bf16 plan, bf16 einsum, bf16 hidden)": lambda st, a, b: t8_fallback(st, a, b),
}
UNIFORM_ONLY = {"sglang as shipped (plan bf16, fp32 sum, store bf16)", "sglang plan-rounding only (store fp32)",
                "sglang storage only (fp64 plan, store bf16)"}
EIGHT_ONLY = {"T8 fallback (bf16 plan, bf16 einsum, bf16 hidden)"}

for pname, knots in PARTS.items():
    for st in SHIFT:
        for iname, fn in IMPLS.items():
            note = None
            if iname in EIGHT_ONLY and not pname.startswith("u8"):
                note = "not reachable: T8 refuses anything but the official 8-step schedule"
            if iname in UNIFORM_ONLY and pname.startswith("tail6"):
                note = "not expressible: sglang fuse() builds uniform blocks only"
            if iname in UNIFORM_ONLY and pname.startswith("u4"):
                note = "computed, but sglang's builder writes block_size from the file (4), so u4 needs a hand-edited pdd_config"
            if note and "not" in note.split(":")[0]:
                results["fidelity"].setdefault(pname, {}).setdefault(st, {})[iname] = {"note": note}
                continue
            worst = None
            for a, b in zip(knots, knots[1:]):
                W, B = fn(st, a, b)
                m = metrics(st, a, b, W, B, hidden_bf16=(iname in EIGHT_ONLY))
                m["block"] = [a, b]
                if worst is None or m["rel_corr"] > worst["rel_corr"]:
                    worst = m
            if note:
                worst["note"] = note
            results["fidelity"].setdefault(pname, {}).setdefault(st, {})[iname] = worst

out = OUT
out.write_text(json.dumps(results, indent=1))
print(json.dumps(results["checks"]))
print(json.dumps(results["bank_encoding"], indent=1))
for pname, d in results["fidelity"].items():
    print("\n==", pname)
    for st, dd in d.items():
        print("  --", st)
        for iname, m in dd.items():
            if "rel_w" not in m:
                print(f"    {iname:70s} {m['note']}")
                continue
            print(f"    {iname:70s} rel_w {m['rel_w']:.2e} rel_corr {m['rel_corr']:.2e} maxabs {m['maxabs_w']:.2e} "
                  f"rel_b_corr {m['rel_b_corr']:.2e} rel_out {m['rel_out']:.2e} worst-block {m['block']}")
