#!/usr/bin/env python3
"""Blend two checkpoints' timestep conditioning at a strength α, exactly. CPU only.

Written 2026-09-27 for the owner's #2 on FastH3 V2: its change lives in the
time embedder (`bench/results/2026-09-26_fasth3_weights.md`), so a strength
dial is a blend of two conditionings, and FastH3 carries the most fine detail
of the distills on 13 of 13 scenes (`2026-09-26_distill_signatures.json`), so
the dial may be a detail control.

In curve form a block's modulation is `W @ table(t) + b`. The blend
`(1 - α) (W_a table_a(t) + b_a) + α (W_b table_b(t) + b_b)` is itself curve
form over a table twice as wide: `[table_a | table_b]`, with weights
`[(1 - α) W_a | α W_b]` and bias `(1 - α) b_a + α b_b`. Core reads the width
from the table (`comfy/model_detection.py`, `time_embed_dim = table.shape[1]`),
so the file loads with no code change. Nothing is refitted. It blends the
post-SiLU time embeddings the projections consume, not the time embedders'
inputs.

Everything outside the conditioning set is `--backbone`'s bytes. The output is
verified by evaluating every block's modulation at all table rows against the
two sources' blend.

    CUDA_VISIBLE_DEVICES= python bench/build_adaln_blend.py --backbone F.safetensors \\
        --cond-a A.safetensors --cond-b B.safetensors --alpha 0.5 --out OUT.safetensors
"""

from __future__ import annotations

import json
import argparse
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_checkpoint_delta import git_commit, header  # noqa: E402
from build_adaln_swap import copy_range, is_conditioning  # noqa: E402

DT = {"F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16}
NAME = {v: k for k, v in DT.items()}


def blended(fa, fb, alpha: float) -> dict:
    out = {}
    ta, tb = fa.get_tensor("adaln_t_table"), fb.get_tensor("adaln_t_table")
    out["adaln_t_table"] = torch.cat([ta, tb], dim=1).to(ta.dtype).contiguous()
    for n in fa.keys():
        if not is_conditioning(n) or n == "adaln_t_table":
            continue
        a, b = fa.get_tensor(n), fb.get_tensor(n)
        if n.endswith(".weight"):
            w = torch.cat([(1 - alpha) * a.double(), alpha * b.double()], dim=1)
        else:
            w = (1 - alpha) * a.double() + alpha * b.double()
        out[n] = w.to(a.dtype).contiguous()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--cond-a", required=True, help="α = 0 end")
    ap.add_argument("--cond-b", required=True, help="α = 1 end")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    hb, bb = header(args.backbone)
    hb.pop("__metadata__", None)
    with safe_open(args.cond_a, "pt") as fa, safe_open(args.cond_b, "pt") as fb:
        new = blended(fa, fb, args.alpha)
    assert set(new) == {n for n in hb if is_conditioning(n)}, "conditioning sets differ"

    meta = {"h3_adaln_blend_backbone": Path(args.backbone).name,
            "h3_adaln_blend_a": Path(args.cond_a).name, "h3_adaln_blend_b": Path(args.cond_b).name,
            "h3_adaln_blend_alpha": repr(args.alpha),
            "h3_adaln_blend_by": f"bench/build_adaln_blend.py @ {git_commit()}"}
    out_hdr: dict = {"__metadata__": meta}
    off = 0
    for name in sorted(hb):
        if name in new:
            t = new[name]
            n, dtype, shape = t.numel() * t.element_size(), NAME[t.dtype], list(t.shape)
        else:
            info = hb[name]
            n, dtype, shape = info["data_offsets"][1] - info["data_offsets"][0], info["dtype"], info["shape"]
        out_hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    hjson = json.dumps(out_hdr, separators=(",", ":")).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    tmp = args.out.with_suffix(args.out.suffix + ".partial")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hjson)))
        w.write(hjson)
        for name in sorted(hb):
            if name in new:
                w.write(new[name].view(torch.uint8).numpy().tobytes() if new[name].dtype == torch.bfloat16
                        else new[name].numpy().tobytes())
            else:
                copy_range(args.backbone, bb, hb[name], w)
    tmp.rename(args.out)

    # Verify: every module's modulation from the output against the blend of
    # the two sources' modulations, at every table row.
    worst = 0.0
    with safe_open(str(args.out), "pt") as fo, safe_open(args.cond_a, "pt") as fa, safe_open(args.cond_b, "pt") as fb:
        to, ta, tb = (f.get_tensor("adaln_t_table").double() for f in (fo, fa, fb))
        for n in sorted(k for k in fo.keys() if k.endswith("adaln_proj.linear.weight")):
            p = n[: -len(".weight")]

            def mod(f, t):
                return t @ f.get_tensor(p + ".weight").double().T + f.get_tensor(p + ".bias").double()
            want = (1 - args.alpha) * mod(fa, ta) + args.alpha * mod(fb, tb)
            got = mod(fo, to)
            worst = max(worst, float(((got - want).norm(dim=1) / want.norm(dim=1)).max()))
    print(json.dumps({"out": args.out.name, "table_width": int(to.shape[1]), "worst_rel_modulation_error": worst,
                      **meta}, indent=1))
    assert worst < 1e-3, f"blend error {worst:.2e} exceeds 1e-3"
    return 0


if __name__ == "__main__":
    sys.exit(main())
