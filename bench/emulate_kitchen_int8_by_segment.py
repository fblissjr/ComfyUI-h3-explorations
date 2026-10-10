#!/usr/bin/env python3
"""Independent CPU emulation of kitchen-like INT8 attention, per query segment (verifier's own).

Segment table comes from the capture manifest but is cross-checked here against the
sequence length. Variants (all against exact fp32 attention on the same captured q,k,v):
  A_rowK      : K per-row absmax int8, unrotated, nothing else quantised (the record's sim)
  B_grpK      : K per-32-key-group absmax int8 (kitchen k_scale granularity), unrotated
  C_grpK_rot  : B + signed Hadamard-128 on q and k, K mean subtracted before quant
  D_QK_rot    : C + Q per-row int8
  E_full_rot  : D + V per-channel int8 (whole sequence) + P uint8 (numerator only, exact denominator)
  F_full_unrot: E without rotation / shift
  G_PV_only   : Q,K exact; V per-channel int8 + P uint8
Variants E to J (V and P) are UNCALIBRATED: their P emulation gives an error far above kitchen's
measured one, so the block 49 record uses A to D only.
Output: per (cell, variant, segment) rows, sq_err_per_row, sq_out_per_row, rel_err.

CPU only, no card:

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/emulate_kitchen_int8_by_segment.py \\
        --capture <capture dir> --block 49 --step 2 --queries 256 --seed 2 --out data/results/<date>_<name>.json
"""
import argparse, glob, json, math, sys
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--capture", type=Path, required=True)
ap.add_argument("--block", type=int, default=49)
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--queries", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--heads", type=int, default=56)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
torch.set_num_threads(16)
g = torch.Generator().manual_seed(a.seed)

man = json.loads((a.capture / "manifest.json").read_text())
p = sorted(glob.glob(str(a.capture / f"qkv_*_b{a.block}_s{a.step}_*.pt")))
assert len(p) == 1
segs = next(t["segments"] for t in man["captured_tensors"] if t["filename"] == Path(p[0]).name)
d = torch.load(p[0], map_location="cpu", weights_only=True, mmap=True)
S = d["q"].shape[2]
assert segs[0][0] == 0 and segs[-1][1] == S and all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
names = [kind if [x[2] for x in segs].count(kind) == 1 else f"{kind}_{i}" for i, (_, _, kind) in enumerate(segs)]
D = 128
# sample rows per segment (random, seeded)
rows = {}
for n, (s0, s1, _) in zip(names, segs):
    m = min(a.queries, s1 - s0)
    rows[n] = (s0 + torch.randperm(s1 - s0, generator=g)[:m]).sort().values
allrows = torch.cat([rows[n] for n in names])
bounds = {}; o = 0
for n in names:
    bounds[n] = (o, o + len(rows[n])); o += len(rows[n])

# signed Hadamard 128
H = torch.tensor([[1.0]])
while H.shape[0] < D:
    H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H = H / math.sqrt(D)
sgn = (torch.randint(0, 2, (D,), generator=torch.Generator().manual_seed(1234)) * 2 - 1).float()
R = sgn[:, None] * H  # x @ R = (x*sgn) @ H

def q8(x, dim):  # per-vector absmax int8 along dim, dequantised
    sc = x.abs().amax(dim, keepdim=True).clamp_min(1e-8) / 127.0
    return (x / sc).round().clamp(-127, 127) * sc

def q8_group_k(k, gs=32):  # one scale per gs consecutive keys (over all channels)
    n = k.shape[0]; pad = (-n) % gs
    kp = torch.nn.functional.pad(k, (0, 0, 0, pad)).view(-1, gs, k.shape[1])
    sc = kp.abs().amax((1, 2), keepdim=True).clamp_min(1e-8) / 127.0
    return ((kp / sc).round().clamp(-127, 127) * sc).view(-1, k.shape[1])[:n]

def attend(qh, kh, vh, vq=False, pq=False):
    s = (qh @ kh.T) * (D ** -0.5)
    s = s - s.amax(-1, keepdim=True)
    p_ = torch.exp(s)
    num_p = (p_ * 255).round() / 255 if pq else p_
    den = p_.sum(-1, keepdim=True) if pq != 2 else num_p.sum(-1, keepdim=True)   # pq==1: exact denominator, pq==2: denominator of the quantised P
    return (num_p @ vh) / den

VARS = ["A_rowK", "B_grpK", "C_grpK_rot", "D_QK_rot", "E_full_rot", "F_full_unrot", "G_PV_only", "H_P_only_exactden", "I_P_only_qden", "J_V_only"]
acc = {v: {n: [0.0, 0.0, 0.0] for n in names} for v in VARS}  # sum sq_err, sum sq_out, sum rel
for h in range(a.heads):
    q = d["q"][0, h].float(); k = d["k"][0, h].float(); v = d["v"][0, h].float()
    qh = q[allrows]
    vq8 = q8(v, 0)  # per-channel (across all rows) V int8
    ex = attend(qh, k, v)
    # variants
    outs = {}
    outs["A_rowK"] = attend(qh, q8(k, 1), v)
    outs["B_grpK"] = attend(qh, q8_group_k(k), v)
    kr = (k - k.mean(0, keepdim=True)) @ R; qr = qh @ R
    outs["C_grpK_rot"] = attend(qr, q8_group_k(kr), v)
    outs["D_QK_rot"] = attend(q8(qr, 1), q8_group_k(kr), v)
    kq = q8_group_k(kr)
    outs["E_full_rot"] = attend(q8(qr, 1), kq, vq8, pq=2)
    outs["F_full_unrot"] = attend(q8(qh, 1), q8_group_k(k), vq8, pq=2)
    outs["G_PV_only"] = attend(qh, k, vq8, pq=2)
    outs["H_P_only_exactden"] = attend(qh, k, v, pq=1)
    outs["I_P_only_qden"] = attend(qh, k, v, pq=2)
    outs["J_V_only"] = attend(qh, k, vq8)
    for var, o_ in outs.items():
        diff = o_ - ex
        err = diff.pow(2).sum(-1); nrm = ex.pow(2).sum(-1)
        rel = err.sqrt() / nrm.sqrt().clamp_min(1e-8)
        for n, (b0, b1) in bounds.items():
            acc[var][n][0] += float(err[b0:b1].mean()); acc[var][n][1] += float(nrm[b0:b1].mean()); acc[var][n][2] += float(rel[b0:b1].mean())
    print("head", h, flush=True, file=sys.stderr)
nh = a.heads
rec = {"capture": a.capture.name, "block": a.block, "step": a.step, "queries_per_segment": a.queries, "seed": a.seed,
       "heads": nh, "segments": [[s0, s1, k_] for s0, s1, k_ in segs], "S": S, "variants": {}}
for var in VARS:
    rows_ = {n: {"rows": s1 - s0, "sq_err_per_row": acc[var][n][0], "sq_out_per_row": acc[var][n][1], "rel_err": acc[var][n][2] / nh}
             for n, (s0, s1, _) in zip(names, segs)}
    def cell(sel):
        num = sum(rows_[n]["rows"] * rows_[n]["sq_err_per_row"] for n in sel); den = sum(rows_[n]["rows"] * rows_[n]["sq_out_per_row"] for n in sel)
        return math.sqrt(num / den)
    va = [n for n in names if n.startswith(("video", "audio"))]
    rec["variants"][var] = {"segments": rows_, "cell_all_rows": cell(names), "cell_video_audio": cell(va), "cell_video": cell(["video"]),
                            "cell_nonva": cell([n for n in names if n not in va])}
    print(a.capture.name[-8:], f"b{a.block}", var, {k_: round(rec['variants'][var][k_], 4) for k_ in ("cell_all_rows", "cell_video_audio", "cell_video", "cell_nonva")},
          {n: round(rows_[n]["rel_err"], 4) for n in names})
a.out.write_text(json.dumps(rec, indent=1))
