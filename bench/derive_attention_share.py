#!/usr/bin/env python3
"""What fraction of an H3 sampler is attention, bounded from an A/B we already ran.

**A bound, not a profile.** Nothing here profiles anything: it re-reads a
rendered ladder and derives what the arms imply. That distinction is the whole
point, because the number gets quoted as though it were measured directly.

## The derivation

The ladder rendered the same scene, seed and length under three attention
configurations that differ ONLY in the attention op. Let `N` be the
non-attention sampler time, which is the same in all three, and `A_x` the
attention time under configuration `x`:

    dense:  N + A_dense = t_dense
    sage:   N + A_sage  = t_sage
    sol:    N + A_sol   = t_sol

Three equations, four unknowns, so the system does not solve. But `A_sol >= 0`
gives `N <= t_sol`, and that is enough for a floor under each share:

    A_dense / t_dense >= (t_dense - t_sol) / t_dense
    A_sage  / t_sage  >= (t_sage  - t_sol) / t_sage

Both are FLOORS. The true shares are higher, because neither sage nor Sol
makes attention free -- `A_sol > 0` strictly, we simply have no arm that
bounds it above.

## Which share answers which question

The dense figure is what supports "attention dominates H3". It is a statement
about a configuration nobody renders in.

The sage figure is the one an Amdahl argument needs: in the configuration that
actually ships, how much is left for an attention kernel change to win. Using
the dense share to rank kernel work overstates the ceiling, which is the error
this record exists to stop. Written 2026-09-08 after a sister project was
about to rank its roadmap on the dense number.

## What this cannot say

**It does not transfer to another clip length, and that is the first thing to
check before quoting it.** Attention is O(S^2) in sequence length while the
projections and the MLP are O(S), so attention's share RISES with the clip and
no single figure is correct for the model. A share measured at a short clip
reads lower than one at the ceiling for a reason that has nothing to do with
the kernel. The geometry every figure was taken at is recorded in the output
beside the figure, from the source record rather than by hand.

Sampler time only. VAE decode is outside it. Bounds, not point estimates. One
seed. And the dense-versus-sparse step split that produces `t_sol` is specific
to the ladder's step count and shift -- `docs/state/open_experiments.md` warns not
to reuse it -- so a different step count or clip length needs its own arms.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = _REPO / "data/results/2026-09-03_ladder_outputs.json"

# The three arms the derivation needs, in the order the algebra uses them.
# `sol` is the reference that bounds N, so it must be the cheapest arm; the
# script asserts that rather than assuming it.
DENSE, SAGE, SOL = "dense", "sage", "sol"


def conditions(record):
    """Geometry and weight format the arms actually ran at.

    Read from the record rather than written here, because the share is
    length-dependent: a reader who does not know the geometry cannot tell
    whether the figure applies to their clip.
    """
    seen = {}
    for arm in record.get("arms", []):
        r = arm.get("rendered") or {}
        for key in ("canvas", "length"):
            if r.get(key) is not None:
                seen.setdefault(key, set()).add(r[key])
        if arm.get("graph"):
            seen.setdefault("graph", set()).add(arm["graph"])
    return {k: (sorted(v)[0] if len(v) == 1 else sorted(v)) for k, v in seen.items()}


def sampler_times(record):
    """{(scene, arm): sampler_s}, warmup rows dropped.

    Warmups are dropped because they share their arm's label and would be
    averaged in as if they were a second sample -- the defect the frontier
    tool had until 2026-09-05.
    """
    out = {}
    for arm in record.get("arms", []):
        if arm.get("warmup") or not arm.get("sampler_s"):
            continue
        scene, _, name = arm["label"].rpartition("_")
        out[(scene, name)] = float(arm["sampler_s"])
    return out


def derive(times):
    """Per-scene floors, plus what was dropped for being incomplete."""
    scenes = sorted({scene for scene, _ in times})
    paired, skipped = [], []
    for scene in scenes:
        t = {arm: times.get((scene, arm)) for arm in (DENSE, SAGE, SOL)}
        if not all(t.values()):
            skipped.append({"scene": scene,
                            "have": sorted(a for a, v in t.items() if v)})
            continue
        if not (t[SOL] <= t[SAGE] <= t[DENSE]):
            raise SystemExit(
                f"{scene}: arms are not ordered sol <= sage <= dense "
                f"({t}). The bound reads N <= the cheapest arm, so an "
                f"out-of-order ladder breaks the derivation rather than "
                f"weakening it.")
        paired.append({
            "scene": scene,
            "sampler_s": {a: round(v, 1) for a, v in t.items()},
            "sage_over_dense": round(t[SAGE] / t[DENSE], 4),
            "sol_over_dense": round(t[SOL] / t[DENSE], 4),
            "attention_floor_of_dense": round(1 - t[SOL] / t[DENSE], 4),
            "attention_floor_of_sage": round((t[SAGE] - t[SOL]) / t[SAGE], 4),
        })
    return paired, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out")
    args = ap.parse_args()

    src = Path(args.source)
    record = json.loads(src.read_text())
    paired, skipped = derive(sampler_times(record))
    if not paired:
        raise SystemExit(f"no scene in {src.name} carries all three arms")

    dense_floor = [p["attention_floor_of_dense"] for p in paired]
    sage_floor = [p["attention_floor_of_sage"] for p in paired]
    out = {
        "what": "a FLOOR under attention's share of H3 sampler time, derived "
                "from three ladder arms that differ only in the attention op",
        "derived": "2026-09-08",
        "produced_by": "bench/derive_attention_share.py",
        "source": str(src.relative_to(_REPO)),
        "conditions": conditions(record),
        "conditions_matter_because": "attention is O(S^2) in sequence length "
            "while the projections and the MLP are O(S), so attention's share "
            "rises with the clip. This figure describes the geometry above and "
            "no other; a shorter clip reads lower for a reason unrelated to the "
            "kernel. The weight format matters for the same reason in reverse: "
            "these are real renders of the shipped graph, so the Linears ran "
            "quantized as they do in production rather than in bf16.",
        "is_not": "a profile. Nothing here times attention directly; the "
                  "floors follow from the arms and from A_sol >= 0.",
        "scenes_paired": len(paired),
        "scenes_skipped": skipped,
        "per_scene": paired,
        "floor_attention_share_of_dense_sampler": {
            "min": min(dense_floor), "median": statistics.median(dense_floor),
            "max": max(dense_floor),
            "reading": "at least this much of a DENSE sampler is attention. "
                       "The figure behind 'attention dominates H3', and a "
                       "statement about a configuration nobody renders in.",
        },
        "floor_attention_share_of_sage_sampler": {
            "min": min(sage_floor), "median": statistics.median(sage_floor),
            "max": max(sage_floor),
            "reading": "at least this much of a SAGE sampler is attention. "
                       "The denominator an Amdahl argument needs, because it "
                       "is the configuration that ships. Ranking kernel work "
                       "on the dense figure overstates the ceiling.",
        },
        "not_established": [
            "an upper bound on either share: no arm makes attention free, so "
            "A_sol > 0 and the true shares are strictly higher than these.",
            "anything about total render time. Sampler only; VAE decode is "
            "outside it.",
            "transfer to another step count or clip length. The sol arm's "
            "dense-versus-sparse step split is specific to the ladder's "
            "schedule, which docs/state/open_experiments.md says not to reuse.",
            "a second seed. One seed, and the per-scene agreement below is "
            "consistency across scenes, not across seeds.",
        ],
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(f"{len(paired)} scene(s) paired, {len(skipped)} skipped")
    for p in paired:
        print(f"  {p['scene']:11s} sage/dense {p['sage_over_dense']:.3f}  "
              f"sol/dense {p['sol_over_dense']:.3f}  "
              f"floor(dense) {p['attention_floor_of_dense']:.1%}  "
              f"floor(sage) {p['attention_floor_of_sage']:.1%}")
    print(f"\nfloor of dense sampler: {min(dense_floor):.1%} to {max(dense_floor):.1%}")
    print(f"floor of sage  sampler: {min(sage_floor):.1%} to {max(sage_floor):.1%}")
    print("\nBoth are floors. Neither sage nor Sol makes attention free.")


if __name__ == "__main__":
    main()
