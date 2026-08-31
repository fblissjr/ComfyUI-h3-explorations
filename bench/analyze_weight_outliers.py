#!/usr/bin/env python3
"""Weight-side outlier structure in the DiT, and what the rotation does to it.

Tier 0 of the quantisation-lever plan: **everything answerable without a card,
a server, or a render.** Two questions, both of which gate a lever that would
otherwise cost card time to evaluate.

1. **Is there headroom in the DiT's int8 build?** The equivalent measurement
   exists for the ENCODER (`2026-08-29_int8_convrot_headroom.json`) and has
   never been run for the DiT -- and on 2026-08-31 the encoder's answer was
   found being carried to a DiT conclusion in `h3_dit_implementations.md`
   section 10.5, which is the two-models crossing `CLAUDE.md` names. So this
   asks it directly: shipped bytes against a deterministic reproduction,
   against `convrot_groupsize` 1024 where the dimension allows it, against no
   rotation at all.

2. **Is there outlier structure left for a wider group or a permutation to
   work on?** The Hadamard's job is to flatten each group before a per-OUTPUT-
   ROW amax sets the step. Two things can still waste levels: a row whose
   overall shape is peaked, and a row whose 256-wide groups disagree with each
   other so one group's outlier sets a step the other groups then waste. The
   second is what a wider rotation (gs 1024) and what a channel permutation
   both attack, and if it is already small then both levers die here for free.

**The groupsize lever is not available everywhere, and the dimensions decide.**
`_build_hadamard` demands a power of 4 dividing `in_features`:

    attn.qkv_proj   5376 = 2^8 * 21   ->  capped at the shipped 256
    mlp.fc1         5376              ->  capped at the shipped 256
    attn.out_proj   7168 = 2^10 * 7   ->  admits 1024
    mlp.fc2        14336 = 2^11 * 7   ->  admits 1024

So the two kinds that can take a wider group are exactly out_proj and fc2.

## What this is NOT

**Stored weights only, and int8_convrot is W8A8.** `int8_linear` rotates the
activation online and quantises it per token, so the error a module carries at
run time is TWO roundings and everything here is the first. A flat result for a
knob whose job is to spread outliers is not evidence the knob is inert -- the
activation is the side with the outliers, and it is `docs/open_experiments.md`
#23 that measures it. This file exists to say which levers are worth putting on
that render, not to decide them.

## The stochastic-rounding arm, which is not a hypothesis

The shipped merge path requantises with stochastic rounding
(`comfy/model_patcher.py:928` passes `seed=string_to_seed(key)` into
`set_weight`). For a value at fractional offset p in a grid cell, round-to-
nearest has squared error `min(p, 1-p)^2` and stochastic rounding has expected
squared error `p(1-p)`; over p uniform that is 1/12 against 1/6. **Exactly 2x
the MSE, sqrt(2) on RMS.** The arm below confirms that on real weights rather
than on the argument, because an analytic factor that does not show up in the
data means the model of the rounding is wrong.

    uv run --active --no-sync python bench/analyze_weight_outliers.py \\
        --base <int8 checkpoint> --reference <release transformer dir> --out <json>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_quant_delta import (  # noqa: E402
    Reference, hf_to_comfy, head_dim, header, marker, stats,
    weight_in_compute_space)
from h3_producer_provenance import producer_provenance  # noqa: E402

sys.path.insert(0, str(_HERE.parents[2]))
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    _build_hadamard, _rotate_weight, dequantize_int8_convrot_weight)

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
#: Powers of 4 only -- `_build_hadamard` raises otherwise, and it is the
#: constraint that decides which kinds this lever is even available on.
CANDIDATE_GROUPSIZES = (64, 256, 1024)


def legal_groupsizes(in_features: int) -> tuple[int, ...]:
    """Which candidate group sizes this input dimension admits.

    Read from the dimension rather than from a table, so a checkpoint with
    different widths answers for itself instead of inheriting H3's answer.
    """
    return tuple(g for g in CANDIDATE_GROUPSIZES if in_features % g == 0)


def quantize(w: np.ndarray, gs: int, seed: int = 0) -> np.ndarray:
    """Rotate / per-output-row amax / round / un-rotate, then dequantise.

    Mirrors `TensorWiseINT8Layout.quantize(convrot=True, per_channel=True)`.
    `seed=0` is round-to-nearest, which is what an offline bake would do;
    `seed>0` is the stochastic branch the run-time merge path takes.
    """
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    h = _build_hadamard(gs, dtype=torch.float32)
    rot = _rotate_weight(t, h, gs)
    row = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    scaled = rot / row
    if seed > 0:
        g = torch.Generator().manual_seed(int(seed))
        noise = torch.rand(scaled.shape, generator=g, dtype=torch.float32)
        q = torch.clamp(torch.floor(scaled + noise), -127, 127).to(torch.int8)
    else:
        q = torch.clamp(torch.round(scaled), -127, 127).to(torch.int8)
    return dequantize_int8_convrot_weight(q, row, gs).numpy()


def no_rotation(w: np.ndarray) -> np.ndarray:
    """Per-output-row int8 with the rotation removed. The control for it."""
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    row = (t.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.clamp(torch.round(t / row), -127, 127).to(torch.int8)
    return (q.to(torch.float32) * row).numpy()


def outlier_profile(w: np.ndarray, gs: int) -> dict:
    """How peaked this weight is, and how much the groups disagree.

    Three numbers, and the third is the one the levers care about:

    `row_peakedness`   median over rows of amax/rms. How much of the int8
                       range a typical row spends on its own tail.
    `row_kurtosis`     median excess kurtosis per row. Same question, less
                       sensitive to a single value.
    `group_disagreement`  median over rows of (max over groups of the group's
                       amax) / (median over groups of the group's amax). This
                       is the WASTE a per-row scale imposes: at 1.0 every
                       group needs the same step and nothing is lost; at 2.0
                       the quiet groups get half the levels they could use.
                       A wider rotation and a channel permutation both attack
                       exactly this, so a value near 1.0 kills both.
    """
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    amax = t.abs().amax(dim=1)
    rms = t.pow(2).mean(dim=1).sqrt().clamp_min(1e-30)
    centred = t - t.mean(dim=1, keepdim=True)
    var = centred.pow(2).mean(dim=1).clamp_min(1e-30)
    kurt = (centred.pow(4).mean(dim=1) / var.pow(2)) - 3.0

    out_f, in_f = t.shape
    groups = t.reshape(out_f, in_f // gs, gs).abs().amax(dim=2)
    gmax = groups.amax(dim=1)
    gmed = groups.median(dim=1).values.clamp_min(1e-30)

    return {
        "row_peakedness": float((amax / rms).median()),
        "row_kurtosis": float(kurt.median()),
        "group_disagreement": float((gmax / gmed).median()),
        "n_groups": in_f // gs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="int8_convrot checkpoint (the shipped bytes)")
    ap.add_argument("--reference", required=True, type=Path,
                    help="release transformer directory (bf16 truth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="0,7,12,24,36,49")
    ap.add_argument("--checkpoint", default="fl2va")
    ap.add_argument("--stochastic-seed", type=int, default=12345)
    args = ap.parse_args()

    blocks = [int(x) for x in args.blocks.split(",")]
    hdr, off = header(args.base)
    ref = Reference(args.reference)
    hd = head_dim(hdr)

    rows = []
    for blk in blocks:
        for kind in KINDS:
            mod = f"blocks.{blk}.{kind}"
            mk = marker(args.base, hdr, off, mod) or {}
            gs = int(mk["convrot_groupsize"])
            w_q = weight_in_compute_space(args.base, hdr, off, mod)
            w_ref = hf_to_comfy(mod + ".weight", ref.get(mod + ".weight"), hd)
            if w_q.shape != w_ref.shape:
                raise SystemExit(
                    f"{mod}: int8 {w_q.shape} against reference {w_ref.shape}")
            in_f = int(w_ref.shape[1])

            row = {
                "block": blk,
                "kind": kind,
                "shape": list(w_ref.shape),
                "shipped_groupsize": gs,
                "legal_groupsizes": list(legal_groupsizes(in_f)),
                # Arm 1: are the shipped bytes reachable by the stock recipe?
                "e_shipped": stats(w_ref, w_q)["rel_delta"],
                "e_no_rotation": stats(w_ref, no_rotation(w_ref))["rel_delta"],
                "by_groupsize": {},
                # Arm 3: the rounding mode, on real weights rather than on the
                # analytic argument. Same group size, same scales, same values.
                "e_deterministic": stats(
                    w_ref, quantize(w_ref, gs, seed=0))["rel_delta"],
                "e_stochastic": stats(
                    w_ref, quantize(w_ref, gs,
                                    seed=args.stochastic_seed))["rel_delta"],
                "outliers": {},
            }
            for g in legal_groupsizes(in_f):
                row["by_groupsize"][str(g)] = stats(
                    w_ref, quantize(w_ref, g, seed=0))["rel_delta"]
                # Structure BEFORE rotation is a property of the weight, not of
                # g; structure AFTER is what the rotation at g achieved.
                t = torch.from_numpy(np.ascontiguousarray(w_ref,
                                                          dtype=np.float32))
                rot = _rotate_weight(
                    t, _build_hadamard(g, dtype=torch.float32), g).numpy()
                row["outliers"][str(g)] = {
                    "unrotated": outlier_profile(w_ref, g),
                    "rotated": outlier_profile(rot, g),
                }
            rows.append(row)
            print(f"  {mod:26s} shipped {row['e_shipped']:.6f}  "
                  f"det {row['e_deterministic']:.6f}  "
                  f"stoch {row['e_stochastic']:.6f}  "
                  f"gs {sorted(row['by_groupsize'])}", flush=True)

    # The producer knows its own shape: blocks x kinds. An output whose shape is
    # the shape of what survived is indistinguishable from a complete one.
    expected = len(blocks) * len(KINDS)
    if len(rows) != expected:
        raise SystemExit(
            f"produced {len(rows)} rows for {len(blocks)} blocks x "
            f"{len(KINDS)} kinds = {expected}")

    def mean(key) -> float:
        # `rows` is non-empty by the shape assertion above, so this cannot be
        # the empty-mean case; asserting rather than returning None keeps the
        # ratios below from silently becoming None-arithmetic.
        return float(np.mean([r[key] for r in rows]))

    wide = [r for r in rows if 1024 in r["legal_groupsizes"]]
    summary = {
        "e_shipped_mean": mean("e_shipped"),
        "e_deterministic_mean": mean("e_deterministic"),
        "e_stochastic_mean": mean("e_stochastic"),
        "e_no_rotation_mean": mean("e_no_rotation"),
        "stochastic_over_deterministic": (
            mean("e_stochastic") / mean("e_deterministic")),
        "rotation_worth": mean("e_no_rotation") / mean("e_shipped"),
        "kinds_admitting_1024": sorted({r["kind"] for r in wide}),
    }
    # PER KIND, not pooled. At block 0 the pooled figure was 0.883 and it is
    # the average of a 21% win on out_proj and nothing at all on fc2 -- one
    # number for two different answers is exactly the shape that hides a
    # finding.
    summary["gs1024_over_gs256_by_kind"] = {
        k: (float(np.mean([r["by_groupsize"]["1024"]
                           for r in wide if r["kind"] == k]))
            / float(np.mean([r["by_groupsize"]["256"]
                             for r in wide if r["kind"] == k])))
        for k in sorted({r["kind"] for r in wide})
    }

    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/analyze_weight_outliers.py",
        "question": ("whether the DiT's int8_convrot build has weight-side "
                     "headroom, and whether any outlier structure remains for "
                     "a wider rotation group or a channel permutation to work "
                     "on"),
        "checkpoint": args.checkpoint,
        "base": Path(args.base).name,
        "reference": str(args.reference.name),
        "path_policy": "logical identifiers only; checkpoints named by file name",
        "is_not": (
            "a runtime measurement. int8_convrot is W8A8 -- `int8_linear` "
            "rotates the activation online and quantises it per TOKEN before "
            "an int8 GEMM whose accumulation is exact -- so this sees ONE of "
            "the two roundings. A flat groupsize result here does not "
            "establish that convrot_groupsize is inert, because the "
            "activation is the side with the outliers. "
            "docs/open_experiments.md #23"),
        "blocks_sampled": blocks,
        "kinds": list(KINDS),
        "shape": {"rows": len(rows), "expected": expected},
        "summary": summary,
        "modules": rows,
        "producer": producer_provenance(__file__),
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"  wrote {Path(args.out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
