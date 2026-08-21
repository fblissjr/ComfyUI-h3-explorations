#!/usr/bin/env python3
"""Grade `docs/open_experiments.md` #22: does the AdaLN pruning move the output?

Reads the per-arm capture directories written by `bench/run_pruning_arms.py`
and answers the entry's pre-registered question at its pre-registered
threshold. It does not choose the threshold; the entry did, before the arms ran.

## What is compared

`S(ckpt, input)` -- the relative L2 between the PRUNED and UNPRUNED velocity on
the same fixed input. The velocity is the DiT's own output, captured by the
`final=1` tap, and H3 emits two of them, so video and audio are graded
separately and the video one is the headline.

Two scales sit beside it, and without them a number like 0.004 means nothing:

- `floor`: the same comparison between two runs of the SAME pruned checkpoint,
  which is what the repeat arm is for. Anything at this level is the pipeline's
  own non-determinism, not the pruning.
- `fp8_ref`: fp8_scaled against int8_convrot on the same input -- a
  quantisation-size difference this repo already ships and lives with. If the
  pruning moves the output less than this, it is smaller than a difference
  nobody has ever complained about.

## The decision rule, quoted from the entry rather than invented here

- SURVIVES: `S(ref2va) / S(fl2va) >= 2`, both at least `10 x floor`, and the
  per-depth profile opens late rather than uniformly from block 0.
- REFUTED: the ratio sits in [0.5, 2] with both above the floor, OR both sit
  within `10 x floor` (then the pruning is invisible on both and the ratio is
  noise over noise).
- Anything else, including `S` above `fp8_ref` on either checkpoint, is a
  finding about the pruning itself and reopens `docs/evidence.md` regardless of
  the ratio.

## Self-test (`--self-test`)

Two deliberate violations, and the script refuses to grade if either is missed:
a fabricated pair that is identical must not read as a difference, and a
fabricated pair whose difference is entirely in one block must not be reported
as opening uniformly. A grader that cannot tell those apart cannot tell this
experiment's two outcomes apart either.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BLOCK_RE = re.compile(r"_b(\d+)_s\d+")


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Relative L2 and cosine, float64, on flattened tensors."""
    x = a.to(torch.float64).flatten()
    y = b.to(torch.float64).flatten()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    denom = torch.linalg.vector_norm(y).item()
    diff = torch.linalg.vector_norm(x - y).item()
    nx = torch.linalg.vector_norm(x).item()
    cos = float((x @ y).item() / (nx * denom)) if nx and denom else float("nan")
    return {"rel_l2": diff / denom if denom else float("nan"),
            "cosine": cos,
            "max_abs": float((x - y).abs().max().item())}


def _one(d: Path, pattern: str) -> Path:
    hits = sorted(d.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"{d.name}: expected one {pattern}, found {len(hits)}")
    return hits[0]


def velocity(arm_dir: Path) -> dict:
    return torch.load(_one(arm_dir, "final_*.pt"), map_location="cpu")


def compare_velocity(a: Path, b: Path) -> dict:
    va, vb = velocity(a), velocity(b)
    out = {}
    for stream in sorted(set(va) & set(vb)):
        out[stream] = rel_l2(va[stream], vb[stream])
    return out


def compare_depth(a: Path, b: Path) -> dict:
    """Per-block q/k/v relative L2 -- the profile the decision rule reads."""
    out = {}
    for pa in sorted(a.glob("qkv_*.pt")):
        block = int(BLOCK_RE.search(pa.name).group(1))
        stem = BLOCK_RE.sub(f"_b{block}_s0", pa.name.split("_b")[0])
        hits = sorted(b.glob(f"*_b{block}_s0*.pt"))
        if len(hits) != 1:
            raise SystemExit(f"{b.name}: expected one block-{block} file, "
                             f"found {len(hits)}")
        ta, tb = torch.load(pa, map_location="cpu"), torch.load(hits[0], map_location="cpu")
        out[block] = {k: rel_l2(ta[k], tb[k]) for k in ("q", "k", "v")}
    return out


def opens_late(profile: dict) -> bool:
    """Does the divergence grow with depth rather than being there from 0?

    'Late' is read as the deepest block's q error being at least twice block
    0's. Stated here rather than in the entry, which said 'opens at or after
    the blocks where the reference rows carry the modulation residual' without
    a number; this is the weakest reading that is still testable, and the
    record says so.
    """
    blocks = sorted(profile)
    if len(blocks) < 2:
        return False
    first = profile[blocks[0]]["q"]["rel_l2"]
    last = profile[blocks[-1]]["q"]["rel_l2"]
    return bool(first > 0 and last >= 2 * first)


def self_test() -> None:
    a = torch.randn(64, 32, dtype=torch.float32)
    same = rel_l2(a, a)
    if same["rel_l2"] > 1e-12:
        raise SystemExit("SELF-TEST FAILED: identical tensors read as different")
    b = a.clone()
    b[0, 0] += 1.0
    if rel_l2(a, b)["rel_l2"] <= 0:
        raise SystemExit("SELF-TEST FAILED: a real difference read as zero")
    flat = {0: {"q": {"rel_l2": 0.01}}, 49: {"q": {"rel_l2": 0.01}}}
    late = {0: {"q": {"rel_l2": 0.001}}, 49: {"q": {"rel_l2": 0.01}}}
    if opens_late(flat):
        raise SystemExit("SELF-TEST FAILED: a uniform profile read as opening late")
    if not opens_late(late):
        raise SystemExit("SELF-TEST FAILED: a late-opening profile read as uniform")
    print("self-test ok: identity, a real difference, and both profile shapes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return 0
    if not args.capture_dir or not args.out:
        raise SystemExit("--capture-dir and --out are required unless --self-test")

    cap = Path(args.capture_dir)
    d = lambda name: cap / name

    rec: dict = {
        "date": "2026-08-21",
        "experiment": "docs/open_experiments.md #22",
        "what": "pruned vs unpruned AdaLN at a fixed first-step forward",
        "canvas": "768x768, 124 frames, seed 730451892, 1 sampler step",
        "capture_dir": cap.name,
    }

    rec["floor"] = {
        "arms": ["ref3_ref2va_pruned", "ref3_ref2va_pruned_repeat"],
        "velocity": compare_velocity(d("ref3_ref2va_pruned"),
                                     d("ref3_ref2va_pruned_repeat")),
        "note": "same checkpoint twice, so this is the pipeline's own "
                "non-determinism and the scale everything else is read against",
    }

    rec["S"] = {}
    rec["depth"] = {}
    for ckpt in ("fl2va", "ref2va"):
        for inp in ("t2v", "ref3"):
            pa, pb = d(f"{inp}_{ckpt}_pruned"), d(f"{inp}_{ckpt}_unpruned")
            rec["S"][f"{ckpt}/{inp}"] = compare_velocity(pa, pb)
            if inp == "ref3":
                rec["depth"][ckpt] = compare_depth(pa, pb)

    rec["fp8_ref"] = {
        ckpt: compare_velocity(d(f"ref3_{ckpt}_pruned"), d(f"ref3_{ckpt}_fp8"))
        for ckpt in ("fl2va", "ref2va")
    }

    floor = rec["floor"]["velocity"]["video"]["rel_l2"]
    s_fl = rec["S"]["fl2va/ref3"]["video"]["rel_l2"]
    s_ref = rec["S"]["ref2va/ref3"]["video"]["rel_l2"]
    fp8 = min(rec["fp8_ref"][c]["video"]["rel_l2"] for c in rec["fp8_ref"])
    ratio = s_ref / s_fl if s_fl else float("inf")
    above_floor = s_fl >= 10 * floor and s_ref >= 10 * floor
    late = opens_late(rec["depth"]["ref2va"])

    if s_fl > fp8 or s_ref > fp8:
        verdict, why = "reopens", (
            "S exceeds the int8-vs-fp8 reference on at least one checkpoint, "
            "which the entry says is a finding about the pruning itself "
            "regardless of the ratio")
    elif not above_floor:
        verdict, why = "refuted", (
            "both values sit within 10x the determinism floor, so the pruning "
            "is invisible at the output on both checkpoints and the ratio is "
            "noise over noise")
    elif ratio >= 2 and late:
        verdict, why = "survives", (
            "ref2va moves at least twice as much as fl2va, both are clear of "
            "the floor, and the divergence opens with depth")
    elif 0.5 <= ratio <= 2:
        verdict, why = "refuted", (
            "the ratio sits in [0.5, 2] with both values above the floor")
    else:
        verdict, why = "reopens", (
            "an outcome the entry did not enumerate; read the numbers")

    rec["decision"] = {
        "floor_video_rel_l2": floor,
        "S_fl2va_ref3": s_fl,
        "S_ref2va_ref3": s_ref,
        "ratio_ref2va_over_fl2va": ratio,
        "fp8_reference_min": fp8,
        "both_above_10x_floor": above_floor,
        "profile_opens_late": late,
        "verdict": verdict,
        "why": why,
    }

    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec["decision"], indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
