#!/usr/bin/env python3
"""What un-merging N blocks actually recovers, and whether picking N pays.

Reads the two full-population 2026-08-31 records and aggregates them. Measures
nothing itself: `pdd_quant_interaction_all_blocks` supplies the deterministic
RTN block curve, while `merge_rounding_regimes` supplies the seeded stochastic
errors the shipped merge actually carries.

**Written to settle whether `unmerged_blocks` should take a SUBSET.** The
tooltip and `docs/h3_pdd.md` carried "worst-first by inflation: 49, 7, 24, 16"
from a 7-block sample, which reads as advice to target. Two of those four are
mid-pack over all 50. The answer the curve gives is that there is no hotspot to
target. On the shipped stochastic path the reachable gain spans 1.836x and
worst-first remains only mildly concave: a subset is a compute optimisation,
not a fidelity one. Selecting all 50 blocks recovers the whole *reachable* gap
at +2.4% arithmetic FLOPs on the linears; `mlp.fc2` remains merged.

**Inflation and stored error are different rankings, and conflating them was
the sample's other defect.** `attn.qkv_proj` has the HIGHEST inflation under
PDD and the LOWEST error before it; `attn.out_proj` has the highest error
before it and nearly the lowest inflation. "Which kind quantises worst" and
"which kind PDD damages most" are separate questions with opposite answers.

**Stored-weight only.** int8_convrot is W8A8; the source records' caveats say
so, and nothing here is a runtime cost. See `docs/open_experiments.md` #23.

    uv run --active --no-sync python bench/analyze_pdd_unmerge_curve.py
"""

from __future__ import annotations

import ast
import collections
import datetime as _dt
import json
import statistics
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SRC = _HERE / "results" / "2026-08-31_pdd_quant_interaction_all_blocks.json"
STOCH_SRC = _HERE / "results" / "2026-08-31_merge_rounding_regimes.json"
PDD_NODE = _HERE.parent / "pdd_lora.py"
OUT = _HERE / "results" / "2026-08-31_pdd_unmerge_recovery.json"
STRENGTH = "1.0"


def runtime_unmerged_kinds() -> tuple[str, ...]:
    """Read the node's literal without importing ComfyUI.

    This analysis once assumed all four kinds remained reachable after
    `mlp.fc2` had been removed from the runtime path. Reading the literal turns
    that mismatch into a red failure instead of maintaining a copied constant.
    """
    tree = ast.parse(PDD_NODE.read_text(), filename=str(PDD_NODE))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "UNMERGED_KINDS"
                        for t in node.targets)):
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and all(isinstance(x, str)
                                                for x in value):
                return value
            break
    raise SystemExit(f"could not read literal UNMERGED_KINDS from {PDD_NODE}")


def main() -> int:
    src = json.loads(SRC.read_text())
    stoch_src = json.loads(STOCH_SRC.read_text())
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

    stoch_mods = stoch_src["modules"]
    stoch_by_key = {(m["block"], m["kind"]): m for m in stoch_mods}
    src_keys = {(m["block"], m["kind"]) for m in mods}
    if len(stoch_mods) != expected or set(stoch_by_key) != src_keys:
        raise SystemExit(
            f"{STOCH_SRC.name} does not carry the same {expected}-module "
            f"population as {SRC.name}")
    shipped_disagreement = max(
        abs(m["e_shipped"]
            - stoch_by_key[(m["block"], m["kind"])]["e_shipped"])
        for m in mods)
    rtn_disagreement = max(
        abs(patched(m)
            - stoch_by_key[(m["block"], m["kind"])]["e_merged_rtn"])
        for m in mods)
    if shipped_disagreement != 0.0 or rtn_disagreement != 0.0:
        raise SystemExit(
            "the two source records do not agree exactly on their shared "
            f"arms (base {shipped_disagreement}, RTN {rtn_disagreement})")
    reachable_kinds = runtime_unmerged_kinds()
    unknown = set(reachable_kinds) - set(kinds)
    if unknown:
        raise SystemExit(f"runtime UNMERGED_KINDS not in records: {unknown}")

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

    # The actual knob reaches only UNMERGED_KINDS, and the merge it replaces
    # uses seeded stochastic rounding. This is the deployable arm; the RTN
    # all-four-kinds curve above remains useful only as the original systematic
    # block-shape analysis.
    def actual_mean(unmerged):
        return statistics.mean(
            m["e_shipped"]
            if m["kind"] in reachable_kinds and m["block"] in unmerged
            else stoch_by_key[(m["block"], m["kind"])]["e_merged_stoch"]
            for m in mods)

    actual_merged = actual_mean(set())
    actual_all = actual_mean(set(blocks))
    actual_gap = actual_merged - actual_all
    actual_gain = {
        b: sum(stoch_by_key[(m["block"], m["kind"])]["e_merged_stoch"]
               - m["e_shipped"]
               for m in by_block[b] if m["kind"] in reachable_kinds)
        for b in blocks
    }
    actual_order = sorted(blocks, key=lambda b: -actual_gain[b])
    actual_curve = []
    for n in (0, 5, 10, 15, 20, 25, 50):
        selected = set(actual_order[:n])
        error = actual_mean(selected)
        actual_curve.append({
            "n_blocks": n,
            "mean_error": error,
            "fraction_of_reachable_gap_recovered": (
                (actual_merged - error) / actual_gap if actual_gap else 0.0),
            "extra_flops_pct_of_linears": n * 2.4 / len(blocks),
        })
    actual_premium = max(
        c["fraction_of_reachable_gap_recovered"]
        - c["n_blocks"] / len(blocks) for c in actual_curve)

    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/analyze_pdd_unmerge_curve.py",
        "what": ("aggregation of the all-blocks stored-weight record by block, "
                 "by kind, and as a recovery curve; no new measurement"),
        "sources": [SRC.name, STOCH_SRC.name],
        "strength": float(STRENGTH),
        "is_not": (
            "a runtime cost. int8_convrot is W8A8 and the source record is a "
            "stored-WEIGHT distance, so this is blind to the activation "
            "rounding -- which un-merging also avoids on the delta, and which "
            "is therefore not counted in any recovery figure here. "
            "docs/open_experiments.md #23"),
        "shape": {"blocks": len(blocks), "kinds": kinds,
                  "modules": len(mods)},
        "source_agreement": {
            "e_shipped_max_abs": shipped_disagreement,
            "e_merged_rtn_max_abs": rtn_disagreement,
            "required": "both exactly zero before the records are joined",
        },
        "legacy_fields_scope": (
            "means, recovery_curve, targeting_premium_over_arbitrary_choice, "
            "by_block_inflation and by_kind are the original deterministic "
            "RTN hypothetical in which all four kinds can be unmerged; use "
            "actual_shipped_path for the deployable node"),
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
        "actual_shipped_path": {
            "rounding": "seeded stochastic",
            "unmerged_kinds_from_runtime": list(reachable_kinds),
            "unreachable_kinds": sorted(set(kinds) - set(reachable_kinds)),
            "all_merged_mean_error": actual_merged,
            "all_blocks_unmerged_mean_error": actual_all,
            "inflation_after_all_reachable_unmerged": actual_all / all_unmerged,
            "fraction_of_shipped_merge_gap_recovered": (
                (actual_merged - actual_all) / (actual_merged - all_unmerged)),
            "recovery_curve": actual_curve,
            "worst_first_blocks": actual_order,
            "gain_spread_max_over_min": (
                max(actual_gain.values()) / min(actual_gain.values())),
            "targeting_premium_over_arbitrary_choice": actual_premium,
        },
        "findings": [
            f"On the actual shipped path, un-merging all reachable kinds in "
            f"all 50 blocks leaves mlp.fc2 merged: {actual_merged:.6f} becomes "
            f"{actual_all:.6f}, or {actual_all / all_unmerged:.4f}x the base "
            f"error. It recovers "
            f"{(actual_merged - actual_all) / (actual_merged - all_unmerged) * 100:.1f}% "
            f"of the stochastic merge gap, not the whole gap.",
            f"The original deterministic RTN, hypothetical four-kind curve "
            f"runs {all_merged:.6f} to {all_unmerged:.6f} "
            f"({all_merged / all_unmerged:.4f}x). It remains for provenance "
            f"and is not the shipped unmerged_blocks result.",
            f"THERE IS NO HOT SUBSET on the deployable arm. Reachable "
            f"stochastic gain spans {max(actual_gain.values()) / min(actual_gain.values()):.3f}x "
            f"across 50 blocks; worst-5 recovers "
            f"{actual_curve[1]['fraction_of_reachable_gap_recovered'] * 100:.1f}% "
            f"of the reachable gap against 10% for an arbitrary 5, and the "
            f"largest sampled targeting premium is {actual_premium * 100:.1f} "
            f"points. Selecting all 50 recovers the reachable gap at +2.4% "
            f"arithmetic FLOPs but still leaves fc2 merged.",
            f"The 7-block list and the old order {order[:10]} describe the "
            f"deterministic four-kind hypothetical. The shipped three-kind "
            f"stochastic order starts {actual_order[:10]}. This is a "
            f"stored-weight compute-allocation order, not a network-sensitivity "
            f"or perceptual-quality ranking.",
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

    print(f"un-merge recovery, from {SRC.name} and {STOCH_SRC.name}")
    print(f"  actual shipped path, kinds {reachable_kinds}: "
          f"{actual_merged:.6f} -> {actual_all:.6f}; remains "
          f"{actual_all / all_unmerged:.4f}x base")
    print(f"  deterministic four-kind reference: {all_merged:.6f} -> "
          f"{all_unmerged:.6f} ({all_merged / all_unmerged:.4f}x)")
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
