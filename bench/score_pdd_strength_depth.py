#!/usr/bin/env python3
"""Score the PDD strength depth probe: per block, how much does strength move
the output?

Reads the latents `bench/probe_pdd_strength_depth.py` wrote and scores each
arm against **its own block's baseline**, which is that block un-merged and
windowed at strength 1.0.

**Why not a shared baseline.** An un-merged, windowed block carries no delta
outside the window, so every arm at block N differs from a normal render in
that respect regardless of its strength. Scoring against a shared merged render
would fold "this block lost its delta for three steps" into every number and
read it as a strength effect. Scoring within a block cancels it exactly: the
arms share the window, the block and the first three steps, and differ only in
the strength applied at the final evaluation.

The reported quantity per block is therefore **the spread the strength axis
produces at that block** -- how far 0.0, 0.5 and 1.5 land from 1.0. A block
where all three sit near zero is one where the distillation's strength does not
matter at that step; a block with a wide spread is one where it does.

Reuses `probe_block_propagation.py`'s loader and metric rather than restating
them, so the two probes cannot disagree about what rel L2 means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from probe_block_propagation import load_latent, rel_l2  # noqa: E402


def newest(latents: Path, tag: str, stream: str) -> Path | None:
    hits = sorted(latents.glob(f"{tag}_{stream}_*.latent"))
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/mnt/hub/ai/img/output")
    ap.add_argument("--blocks", default="0,1,2,8,16,24,32,40,45,48,49")
    ap.add_argument("--strengths", default="0.0,0.5,1.5")
    ap.add_argument("--arms", default="bench/results/2026-08-30_pdd_strength_depth_arms.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # NOT under `latents/`, unlike `probe_block_propagation.py`, whose
    # `PREFIX` carries the subdirectory in the filename_prefix itself. This
    # probe's prefix does not, so SaveLatent writes beside the other outputs.
    # Kept as-is rather than aligned mid-sweep: changing the prefix while a run
    # is in flight splits one experiment's arms across two directories, which
    # is a worse outcome than an inconsistent path.
    latents = Path(args.output_dir) / "h3_pdd_depth"
    if not latents.is_dir():
        alt = Path(args.output_dir) / "latents" / "h3_pdd_depth"
        if alt.is_dir():
            latents = alt
    blocks = [int(b) for b in args.blocks.split(",")]
    strengths = [float(s) for s in args.strengths.split(",")]

    rows, missing = [], []
    for b in blocks:
        btag = f"b{b:02d}_base"
        base_p = {s: newest(latents, btag, s) for s in ("video", "audio")}
        if not all(base_p.values()):
            missing.append(btag)
            continue
        base = {k: load_latent(v) for k, v in base_p.items()}
        for s in strengths:
            tag = f"b{b:02d}_s{s}"
            p = {st: newest(latents, tag, st) for st in ("video", "audio")}
            if not all(p.values()):
                missing.append(tag)
                continue
            arm = {k: load_latent(v) for k, v in p.items()}
            rows.append({
                "block": b, "strength": s,
                "rel_l2_video": rel_l2(arm["video"], base["video"]),
                "rel_l2_audio": rel_l2(arm["audio"], base["audio"]),
            })

    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    # The block's spread: the largest distance any strength arm reaches from
    # 1.0. One number per block, which is what a depth profile needs.
    spread = {b: max(x["rel_l2_video"] for x in v)
              for b, v in by_block.items() if v}

    out = {
        "measured": "2026-08-30",
        "produced_by": "bench/score_pdd_strength_depth.py",
        "question": ("per block, how far does changing PDD's strength at ONE "
                     "step move the output latent"),
        "arms_record": args.arms,
        "baseline": ("per block: that block un-merged and windowed at strength "
                     "1.0. Not a shared merged render -- see the docstring."),
        "rows": sorted(rows, key=lambda r: (r["block"], r["strength"])),
        "video_spread_by_block": {str(b): spread[b] for b in sorted(spread)},
        "missing": missing,
        "not_established": (
            "one seed, one prompt, one canvas, one step. rel L2 on a latent is "
            "not perceptual, and the different-sample rule means a rendered "
            "pair could not make it so."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print("  block   s=0.0     s=0.5     s=1.5     (rel L2 video vs s=1.0)")
    for b in sorted(by_block):
        g = {r["strength"]: r["rel_l2_video"] for r in by_block[b]}
        cells = "  ".join(f"{g.get(s, float('nan')):.6f}" for s in strengths)
        print(f"    {b:2d}    {cells}")
    if spread:
        lo, hi = min(spread.values()), max(spread.values())
        print(f"\n  spread across blocks: min {lo:.6f}  max {hi:.6f}  "
              f"ratio {hi / lo if lo else float('nan'):.2f}x")
    if missing:
        print(f"\n  MISSING {len(missing)} arm(s): {missing[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
