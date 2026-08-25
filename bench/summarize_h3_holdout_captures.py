#!/usr/bin/env python3
"""Aggregate per-row layer-50 comparisons into one holdout record.

`bench/compare_transformers_comfy_layer50.py --compare` grades one row at a
time: a reference capture against one or more candidate arms, refusing where
the arms answer different questions. A holdout is many rows under more than
one geometry, so this script reads a directory tree of those per-row reports
and writes a single record: per arm and geometry, the distribution of the
relative L2 and cosine at layer 50 (all rows, text rows, vision rows), the
per-row values, and every refusal verbatim. It computes nothing new; a number
here is one the comparator already wrote.

Layout expected:

    <root>/<geometry>/<row_id>.json      one comparator report per row

    python bench/summarize_h3_holdout_captures.py --root DIR \\
        --out bench/results/<date>_v2_holdout_layer50.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from h3_producer_provenance import producer_provenance  # noqa: E402

METRICS = ("relative_l2", "flattened_cosine", "tokenwise_cosine_min")
SLICES = ("all", "text_rows", "vision_rows", "layer0_input")


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "min": ordered[0], "median": statistics.median(ordered),
            "max": ordered[-1], "mean": statistics.fmean(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="directory of <geometry>/<row_id>.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    per_arm: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    refusals: list[dict] = []
    reference = None
    rows_seen: dict = defaultdict(set)
    for geometry_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        geometry = geometry_dir.name
        for report_path in sorted(geometry_dir.glob("*.json")):
            report = json.loads(report_path.read_text())
            row_id = report_path.stem
            rows_seen[geometry].add(row_id)
            ref = report.get("reference_arm", {})
            ref_key = (ref.get("arm"), ref.get("dtype"), ref.get("source", {}).get("logical_name"))
            if reference is None:
                reference = ref_key
            elif reference != ref_key:
                raise SystemExit(f"{report_path}: reference {ref_key} differs from {reference}; "
                                 "one holdout record has one reference")
            for label, entry in report.get("arms", {}).items():
                if "refused" in entry:
                    refusals.append({"geometry": geometry, "row_id": row_id,
                                     "arm": label, "refused": entry["refused"]})
                    continue
                for slice_name in SLICES:
                    block = entry.get(slice_name)
                    if not block:
                        continue
                    for metric in METRICS:
                        if metric in block:
                            per_arm[label][geometry][slice_name].setdefault(metric, {})[row_id] = block[metric]

    summary: dict = {}
    for label, by_geometry in per_arm.items():
        summary[label] = {}
        for geometry, by_slice in by_geometry.items():
            summary[label][geometry] = {}
            for slice_name, by_metric in by_slice.items():
                summary[label][geometry][slice_name] = {
                    metric: {"distribution": _dist(list(values.values())), "per_row": values}
                    for metric, values in by_metric.items()
                }

    record = {
        "purpose": "holdout layer-50 fidelity of W4 arms against the BF16 ComfyUI arm, "
                   "aggregated from per-row comparator reports; computes nothing new",
        "producer": producer_provenance(__file__),
        "comparator_reports_root": root.name,
        "reference": {"arm": reference[0], "dtype": reference[1], "source": reference[2]}
                     if reference else None,
        "rows_per_geometry": {g: sorted(r) for g, r in rows_seen.items()},
        "arms": summary,
        "refusals": refusals,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"reference: {record['reference']}")
    for label, by_geometry in summary.items():
        for geometry, by_slice in by_geometry.items():
            line = f"  {label:<60} {geometry:<10}"
            for slice_name in ("all", "text_rows", "vision_rows"):
                d = by_slice.get(slice_name, {}).get("relative_l2", {}).get("distribution")
                if d and d["n"]:
                    line += f"  {slice_name} relL2 med {d['median']:.4g} max {d['max']:.4g} (n={d['n']})"
            print(line)
    if refusals:
        print(f"  refusals: {len(refusals)}")
        for r in refusals[:20]:
            print(f"    {r['geometry']} {r['row_id']} {r['arm']}: {r['refused']}")
    print(f"wrote {out.name}")
    return 0 if not refusals else 1


if __name__ == "__main__":
    sys.exit(main())
