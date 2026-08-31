#!/usr/bin/env python3
"""What un-merging N blocks actually recovers, and whether picking N pays.

Reads `bench/results/2026-08-31_pdd_quant_interaction_all_blocks.json` and
aggregates it three ways. Measures nothing itself -- that file is the
measurement, produced by `bench/measure_pdd_quant_interaction.py` against the
bf16 release over all 200 backbone modules.

**Written to settle whether `unmerged_blocks` should take a SUBSET.** The
tooltip and `docs/h3_pdd.md` carried "worst-first by inflation: 49, 7, 24, 16"
from a 7-block sample, which reads as advice to target. Two of those four are
mid-pack over all 50. The answer the curve gives is that there is no hotspot to
target -- block inflation spans 1.131x -- so worst-first is mildly concave
rather than steep, and a subset is a compute optimisation rather than a
fidelity one: the full recovery costs +2.4% FLOPs on the linears and any subset
trades fidelity away to save part of that.

**Inflation and stored error are different rankings, and conflating them was
the sample's other defect.** `attn.qkv_proj` has the HIGHEST inflation under
PDD and the LOWEST error before it; `attn.out_proj` has the highest error
before it and nearly the lowest inflation. "Which kind quantises worst" and
"which kind PDD damages most" are separate questions with opposite answers.

**Stored-weight only, and that is one of two roundings.** int8_convrot is
W8A8; the source record's own caveat field says so, and nothing here is a
runtime cost. See `docs/open_experiments.md` #23.

    uv run --active --no-sync python bench/analyze_pdd_unmerge_curve.py
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import statistics
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SRC = _HERE / "results" / "2026-08-31_pdd_quant_interaction_all_blocks.json"
OUT = _HERE / "results" / "2026-08-31_pdd_unmerge_recovery.json"
STRENGTH = "1.0"


def main() -> int:
    src = json.loads(SRC.read_text())
    mods = src["modules"]
    if not mods:
        raise SystemExit(f"{SRC.name} carries no modules")

    def patched(m):
        return m["by_strength"][STRENGTH]["e_patched"]

    # The producer knows the shape it should have: 50 blocks x 4 kinds. Assert
    # it rather than aggregating whatever survived -- an output whose shape is
    # the shape of what survived is indistinguishable from a complete one.
    blocks = sorted({m["block"] for m in mods})
    kinds = sorted({m["kind"] for m in mods})
    expected = len(blocks) * len(kinds)
    if len(mods) != expected:
        raise SystemExit(
            f"{SRC.name} has {len(mods)} modules for {len(blocks)} blocks x "
            f"{len(kinds)} kinds = {expected}. This aggregation would report a "
            f"ranking over a subset and look complete.")

    by_block = collections.defaultdict(list)
    by_kind = collections.defaultdict(list)
    for m in mods:
        by_block[m["block"]].append(m)
        by_kind[m["kind"]].append(m)

    def mean_err(unmerged):
        """Mean stored-weight error when `unmerged` blocks apply at the call.

        An un-merged module never enters the requantisation, so it carries the
        checkpoint's shipped error at full LoRA strength -- that equality is
        the source record's `e_patched` at strength 0.0, which reproduces
        `e_shipped` to eight digits.
        """
        return statistics.mean(
            m["e_shipped"] if m["block"] in unmerged else patched(m)
            for m in mods)

    all_merged = mean_err(set())
    all_unmerged = mean_err(set(blocks))
    gap = all_merged - all_unmerged

    # Per block: how much un-merging THIS block alone removes from the sum.
    gain = {b: sum(patched(m) - m["e_shipped"] for m in by_block[b])
            for b in blocks}
    order = [b for b, _ in sorted(gain.items(), key=lambda kv: -kv[1])]

    curve = []
    for n in (0, 5, 10, 15, 20, 25, 50):
        if n > len(blocks):
            continue
        e = mean_err(set(order[:n]))
        curve.append({
            "n_blocks": n,
            "mean_error": e,
            "fraction_of_gap_recovered": (all_merged - e) / gap if gap else 0.0,
            # +2.4% FLOPs on an un-merged block's linears, from the shapes.
            # NOT timed -- the tooltip and h3_pdd.md say so and so does this.
            "extra_flops_pct_of_linears": n * 2.4 / len(blocks),
        })

    # The flatness claim, stated as the number that carries it: if targeting
    # paid, the worst-N curve would be steeply concave. Compare against the
    # straight line an arbitrary choice of N would give.
    linear_at = {c["n_blocks"]: c["n_blocks"] / len(blocks) for c in curve}
    # Max over the SAMPLED N, not over all 50 -- the curve is only evaluated
    # at the points above, so this is a lower bound on the true peak.
    targeting_premium = max(
        c["fraction_of_gap_recovered"] - linear_at[c["n_blocks"]]
        for c in curve)

    def infl(m):
        return patched(m) / m["e_shipped"]

    block_infl = {b: statistics.mean(infl(m) for m in by_block[b])
                  for b in blocks}
    ranked = sorted(block_infl.items(), key=lambda kv: -kv[1])

    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/analyze_pdd_unmerge_curve.py",
        "what": ("aggregation of the all-blocks stored-weight record by block, "
                 "by kind, and as a recovery curve; no new measurement"),
        "source": SRC.name,
        "strength": float(STRENGTH),
        "is_not": (
            "a runtime cost. int8_convrot is W8A8 and the source record is a "
            "stored-WEIGHT distance, so this is blind to the activation "
            "rounding -- which un-merging also avoids on the delta, and which "
            "is therefore not counted in any recovery figure here. "
            "docs/open_experiments.md #23"),
        "shape": {"blocks": len(blocks), "kinds": kinds,
                  "modules": len(mods)},
        "means": {
            "all_merged": all_merged,
            "all_unmerged": all_unmerged,
            "recoverable_gap": gap,
            "inflation_all_merged": all_merged / all_unmerged,
        },
        "recovery_curve": curve,
        "targeting_premium_over_arbitrary_choice": targeting_premium,
        "by_block_inflation": {
            "worst_10": [{"block": b, "inflation": v} for b, v in ranked[:10]],
            "quietest_10": [{"block": b, "inflation": v}
                            for b, v in ranked[-10:]],
            "spread_max_over_min": ranked[0][1] / ranked[-1][1],
            "all": {str(b): v for b, v in sorted(block_infl.items())},
        },
        "by_kind": {
            k: {
                "inflation": statistics.mean(infl(m) for m in v),
                "e_shipped_mean": statistics.mean(m["e_shipped"] for m in v),
                "e_patched_mean": statistics.mean(patched(m) for m in v),
            } for k, v in by_kind.items()
        },
        "findings": [
            f"Un-merging every block recovers the whole inflation: "
            f"{all_merged:.6f} merged against {all_unmerged:.6f}, which IS the "
            f"checkpoint's shipped error. The gap is {gap:.6f}, "
            f"{all_merged / all_unmerged:.4f}x.",
            f"THERE IS NO HOT SUBSET, only a mildly concave curve. Block "
            f"inflation spans {ranked[0][1] / ranked[-1][1]:.3f}x across all "
            f"50 -- no hotspot -- so worst-first buys a modest premium over an "
            f"arbitrary choice rather than a big one: worst-5 recovers "
            f"{curve[1]['fraction_of_gap_recovered'] * 100:.1f}% against 10% "
            f"for any 5, and the premium peaks at "
            f"{targeting_premium * 100:.1f} points around the halfway mark "
            f"(worst-25 recovers "
            f"{curve[-2]['fraction_of_gap_recovered'] * 100:.1f}%). So "
            f"targeting is a COMPUTE optimisation, not a fidelity one: the "
            f"full recovery is available at +2.4% FLOPs on the linears, and "
            f"any subset trades fidelity away to save a fraction of that. "
            f"Whoever picks a subset should say which of the two they are "
            f"buying.",
            "The 7-block sample's `worst-first 49, 7, 24, 16` is half right "
            "and reads as advice to target. Blocks 49 and 7 do rank 1st and "
            "5th of 50; blocks 24 and 16 are mid-pack at 17th and 16th. Its "
            "`least worth it: 32 and 40` holds -- they rank 46th and 44th, in "
            "a quiet band running roughly 31-36.",
            f"INFLATION AND STORED ERROR RANK KINDS OPPOSITELY. "
            f"attn.qkv_proj has the highest inflation "
            f"({statistics.mean(infl(m) for m in by_kind['attn.qkv_proj']):.4f}) "
            f"and the lowest error before PDD "
            f"({statistics.mean(m['e_shipped'] for m in by_kind['attn.qkv_proj']):.5f}); "
            f"attn.out_proj has the highest error before PDD "
            f"({statistics.mean(m['e_shipped'] for m in by_kind['attn.out_proj']):.5f}) "
            f"and nearly the lowest inflation "
            f"({statistics.mean(infl(m) for m in by_kind['attn.out_proj']):.4f}). "
            f"'Which kind quantises worst' and 'which kind PDD damages most' "
            f"are different questions and the answers are reversed.",
        ],
        "not_measured": [
            "any runtime or perceptual consequence. See `is_not`.",
            "the +2.4% FLOPs figure, which is arithmetic from the shapes and "
            "has never been timed. Every cost column here inherits that.",
            "ref2va. The source record is fl2va; the partitions have identical "
            "key sets and near-identical stored weights, but that is an "
            "argument rather than this measurement.",
        ],
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")

    print(f"un-merge recovery, from {SRC.name}")
    print(f"  all merged {all_merged:.6f} -> all un-merged {all_unmerged:.6f} "
          f"({all_merged / all_unmerged:.4f}x)")
    for c in curve:
        print(f"  N={c['n_blocks']:2d}  {c['mean_error']:.6f}  "
              f"recovers {c['fraction_of_gap_recovered'] * 100:5.1f}%  "
              f"+{c['extra_flops_pct_of_linears']:.2f}% FLOPs")
    print(f"  targeting premium over an arbitrary N: "
          f"{targeting_premium * 100:.1f} points")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
