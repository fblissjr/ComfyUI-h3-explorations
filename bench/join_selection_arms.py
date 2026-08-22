#!/usr/bin/env python3
"""Join the Sol selection arms into one verdict record.

The arms answer a question upstream asked (kijai, 2026-08-22): adaptive tau
against top-k, without and with the SLA LoRA. Six arms, one graph, matched
seeds, alternating -- the difference between arms is the `selection` widget
and which LoRA file loads, and nothing else.

**What this can and cannot say.** Sampler seconds is a clean answer: one
variable per contrast, same graph, same workload, build recorded in the rows.
Which arm looks better is NOT answered here and cannot be, because adaptive
and top-k are a numerical knob and the sampling trajectory diverges completely
between them -- the two clips are different samples, not better and worse
versions of one (CLAUDE.md). The clips are for briefs-met and for watching.

The contrast that carries the actual question is the DIFFERENCE OF
DIFFERENCES: if the SLA LoRA needs the selection it was distilled under, then
top-k should buy it something it does not buy v1.1. If both LoRAs move
together, selection is a kernel property and the distillation did not bake in
a dependence on it.

Warmup rows are recorded and excluded from every statistic, per the runner.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

#: arm -> (lora, selection, nominal density kept). The density column is what
#: turns this from two points into a curve. **It is NOMINAL, not measured**:
#: top-k keeps exactly its fraction by construction, while tau's is upstream's
#: reported figure for that threshold and the real density varies per head and
#: per block. So density-matched here means "asked for about the same
#: fraction", never "kept the same blocks" -- which is the whole difference
#: between the two selections and the reason a speed gap alone cannot be read
#: as one selection being cheaper than the other.
ARM_SPEC = {
    "sla_adaptive": ("sla", "adaptive", 0.16),
    "sla_tau13":    ("sla", "adaptive", 0.07),
    "sla_topk15":   ("sla", "topk",     0.15),
    "sla_topk10":   ("sla", "topk",     0.10),
    "v11_adaptive": ("v11", "adaptive", 0.16),
    "v11_tau13":    ("v11", "adaptive", 0.07),
    "v11_topk15":   ("v11", "topk",     0.15),
    "v11_topk10":   ("v11", "topk",     0.10),
}
ARMS = tuple(ARM_SPEC)


def load(path: pathlib.Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.setdefault(r["label"], []).append(r)
    return rows


def timed(rows: list[dict]) -> list[float]:
    """Sampler seconds from the rows that count: not warmup, no error."""
    return [r["sampler_s"] for r in rows
            if not r.get("warmup") and not r.get("error")
            and r.get("sampler_s") is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"no such rows file: {args.jsonl}", file=sys.stderr)
        return 1
    rows = load(args.jsonl)
    missing = [a for a in ARMS if not timed(rows.get(a, []))]
    if missing:
        print(f"no timed rows for {missing} -- refusing to write a verdict "
              f"that silently covers a subset", file=sys.stderr)
        return 1

    per_arm = {}
    for a in ARMS:
        s = timed(rows[a])
        lora, sel, dens = ARM_SPEC[a]
        per_arm[a] = {
            "lora": lora,
            "selection": sel,
            "nominal_density_kept": dens,
            "sampler_s": [round(x, 2) for x in s],
            "median_s": round(statistics.median(s), 2),
            "runs": len(s),
            # Within-arm spread bounds what a between-arm gap can mean: a
            # contrast smaller than this is noise wearing a label.
            "spread_frac": round((max(s) - min(s)) / statistics.median(s), 4),
            "errors": sum(1 for r in rows[a] if r.get("error")),
        }

    def gap(fast, slow):
        f, s = per_arm[fast]["median_s"], per_arm[slow]["median_s"]
        return round((s - f) / s, 4)

    contrasts = {}
    for lora in ("sla", "v11"):
        base = f"{lora}_adaptive"
        for kp in ("15", "10"):
            contrasts[f"{lora}: topk{kp} vs adaptive"] = {
                "topk_median_s": per_arm[f"{lora}_topk{kp}"]["median_s"],
                "adaptive_median_s": per_arm[base]["median_s"],
                "topk_saves_frac": gap(f"{lora}_topk{kp}", base),
            }

    # Seconds per point of nominal density, within each selection. If the two
    # selections cost the same per block kept, these slopes match and the
    # speed gap is density, not selection. If they diverge, the selection
    # itself has a price. Two points per slope is the minimum that can show
    # anything and is not enough to claim linearity -- read the sign, not the
    # magnitude.
    slopes = {}
    for lora in ("sla", "v11"):
        for sel, pair in (("adaptive", (f"{lora}_adaptive", f"{lora}_tau13")),
                          ("topk", (f"{lora}_topk15", f"{lora}_topk10"))):
            hi, lo = pair
            dd = (per_arm[hi]["nominal_density_kept"]
                  - per_arm[lo]["nominal_density_kept"])
            ds = per_arm[hi]["median_s"] - per_arm[lo]["median_s"]
            slopes[f"{lora}_{sel}"] = {
                "points": {hi: per_arm[hi]["median_s"], lo: per_arm[lo]["median_s"]},
                "density_delta": round(dd, 4),
                "seconds_per_density_point": round(ds / dd, 1) if dd else None,
            }

    verdict = {
        "question": "kijai 2026-08-22: adaptive tau against top-k, without "
                    "and with the SLA LoRA",
        "source_rows": str(args.jsonl),
        "graph_sha256": rows[ARMS[0]][0]["graph_sha256"],
        "per_arm": per_arm,
        "contrasts": contrasts,
        "cost_slope_by_selection": slopes,
        "difference_of_differences": {
            f"topk{kp}": round(contrasts[f"sla: topk{kp} vs adaptive"]["topk_saves_frac"]
                               - contrasts[f"v11: topk{kp} vs adaptive"]["topk_saves_frac"], 4)
            for kp in ("15", "10")
        },
        "what_this_does_not_answer": (
            "which arm looks better. Adaptive and top-k are a numerical knob "
            "and their trajectories diverge completely, so the clips are "
            "different samples rather than better and worse versions of one. "
            "Watch them for briefs met; do not read a quality ranking here. "
            "Nor does it say the two selections kept the SAME blocks at a "
            "matched nominal density -- tau's density varies per head and "
            "block where top-k's does not, and nothing here measures the "
            "realised density of either."),
        "largest_within_arm_spread_frac": max(
            v["spread_frac"] for v in per_arm.values()),
    }

    text = json.dumps(verdict, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
