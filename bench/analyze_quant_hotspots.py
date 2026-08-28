#!/usr/bin/env python3
"""Where INT8 weight error concentrates: by block, and by module kind.

Reads the stored-weight record `bench/results/2026-08-21_quant_delta_ref2va.json`
and aggregates it two ways. Measures nothing itself -- that file is the
measurement, produced by `bench/analyze_quant_delta.py` against the bf16
release, with its own self-tests (identity, perturbation, rotated-vs-unrotated,
wrong-groupsize).

**Written to answer one question and it came back NO.** `docs/SOLATTN.md` records
that Sol's CUDA kernel disagrees with its eager reference far more at block 49
than anywhere else -- `1 - quant_cos` about 7.5e-3 against 2.1e-4..8.3e-4 through
the trunk, scale-invariant, so not a normalisation effect. The obvious
hypothesis was that block 49's WEIGHTS quantise badly and both effects share a
cause. They do not: stored-weight error is flat across all 50 blocks and block
49 is unremarkable in it.

**These are two different INT8s and the words do not disambiguate them.**
`analyze_sol_error.py::decompose_single` takes q, k, v as inputs and builds all
three references from them, so the checkpoint's error is in all three and
cancels -- its `quant_l2` is the attention kernel's own arithmetic. This file's
numbers are the checkpoint. A claim about one is not a claim about the other,
and that crossing was made and corrected on 2026-08-28.

    uv run --active --no-sync python bench/analyze_quant_hotspots.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "2026-08-21_quant_delta_ref2va.json"
if not SRC.exists():
    SRC = REPO / "bench" / "results" / "2026-08-21_quant_delta_ref2va.json"
OUT = REPO / "bench" / "results" / "2026-08-28_quant_hotspots_ref2va.json"

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def main() -> int:
    src = json.loads(SRC.read_text())
    mods = src["modules"]

    by_block: dict[int, list[float]] = {}
    for m in mods:
        by_block.setdefault(m["block"], []).append(m["int8_vs_bf16"]["rel_delta"])
    block_mean = {b: st.mean(v) for b, v in by_block.items()}
    ranked = sorted(block_mean.items(), key=lambda kv: -kv[1])

    by_kind = {}
    for k in KINDS:
        rows = [m for m in mods if m["kind"] == k]
        by_kind[k] = {
            "int8_vs_bf16_mean": st.mean(m["int8_vs_bf16"]["rel_delta"] for m in rows),
            "int8_vs_bf16_max": max(m["int8_vs_bf16"]["rel_delta"] for m in rows),
            "fp8_vs_bf16_mean": st.mean(m["fp8_vs_bf16"]["rel_delta"] for m in rows),
            "row_rel_p95_median": st.median(m["int8_vs_bf16"]["row_rel_p95"] for m in rows),
            "row_rel_max": max(m["int8_vs_bf16"]["row_rel_max"] for m in rows),
            "n": len(rows),
        }
    base = by_kind["attn.qkv_proj"]["int8_vs_bf16_mean"]
    for k in KINDS:
        by_kind[k]["vs_qkv"] = by_kind[k]["int8_vs_bf16_mean"] / base

    spread = max(block_mean.values()) / min(block_mean.values())
    rank49 = [b for b, _ in ranked].index(49) + 1

    record = {
        "measured": "2026-08-28",
        "produced_by": "bench/analyze_quant_hotspots.py",
        "what": "aggregation of stored-weight int8-vs-bf16 error by block and by "
                "module kind; no new measurement",
        "source": SRC.name,
        "is_not": "Sol's kernel INT8. See this file's docstring -- "
                  "analyze_sol_error.py's quant_l2 is a different quantity and "
                  "the two were crossed once.",
        "by_block": {
            "spread_max_over_min": spread,
            "block_49_rank_of_50": rank49,
            "block_49_mean": block_mean[49],
            "worst_5": [{"block": b, "mean": v} for b, v in ranked[:5]],
            "all": {str(b): v for b, v in sorted(block_mean.items())},
        },
        "by_kind": by_kind,
        "findings": [
            f"Stored-weight INT8 error is FLAT across blocks: {spread:.3f}x from "
            f"best to worst over all 50. Block 49 ranks {rank49}/50. The worst "
            f"blocks are {[b for b,_ in ranked[:5]]}, the shallow end, not the deep "
            f"end. So block 49's Sol-kernel behaviour has no counterpart in its "
            f"weights, and re-baking the checkpoint cannot address it.",
            f"attn.out_proj is the worst module kind under INT8 at "
            f"{by_kind['attn.out_proj']['vs_qkv']:.2f}x qkv_proj on the mean, and "
            f"worse in the tail: row_rel p95 "
            f"{by_kind['attn.out_proj']['row_rel_p95_median']:.5f} against "
            f"{by_kind['attn.qkv_proj']['row_rel_p95_median']:.5f}, row_rel max "
            f"{by_kind['attn.out_proj']['row_rel_max']:.5f} against "
            f"{by_kind['attn.qkv_proj']['row_rel_max']:.5f}.",
            "Under fp8_scaled all four kinds are equal to four decimal places, so "
            "out_proj's excess is a property of the INT8 SCHEME rather than of the "
            "out_proj weights. That is what makes it a candidate for different "
            "treatment in a re-bake; it is also the only lane here the bf16 "
            "release is needed for.",
        ],
        "not_measured": [
            "whether out_proj's excess reaches the output. This is stored-weight "
            "fidelity only -- the same caveat the source record carries.",
            "why the INT8 scheme treats out_proj worse. Outlier structure the "
            "rotation does not spread is the obvious candidate and is untested.",
            "any block-49 activation property. Nothing here looks at activations.",
        ],
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")

    print(f"stored-weight INT8 error, from {SRC.name}")
    print(f"  by block: spread {spread:.3f}x over 50 blocks, "
          f"block 49 ranks {rank49}/50")
    print(f"  worst blocks: {[b for b, _ in ranked[:5]]}")
    print(f"  {'kind':>16} {'int8':>9} {'fp8':>9} {'vs qkv':>8}")
    for k in KINDS:
        d = by_kind[k]
        print(f"  {k:>16} {d['int8_vs_bf16_mean']:>9.5f} "
              f"{d['fp8_vs_bf16_mean']:>9.5f} {d['vs_qkv']:>7.2f}x")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
