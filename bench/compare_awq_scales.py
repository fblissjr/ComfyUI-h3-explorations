#!/usr/bin/env python3
"""Compare the AWQ scales two Gate 2B pilot arms found, mapping by mapping.

The kernel-sensitivity control of `canonical/2026-08-25_gate2_arrangement.md`:
expanded-KV efficient attention was accepted as the calibration kernel on the
reading that the layer-49 spread on two fixtures is compounding, and the
record left it **UNKNOWN** whether the AWQ observer -- an aggregate over rows
and positions -- is sensitive at that level. This answers that on the object
the modifier actually produces: the per-channel scales of every smoothed
mapping, saved by `bench/pilot_sequential_feasibility.py --modifier awq` beside
its report, from one arm under grouped-query math and one under expanded-KV
efficient attention, same rows, same storage policy.

It refuses to compare arms that differ on anything but the attention kind:
rows, policy, layers, recipe, and the set of mappings must match, so a
difference can be attributed to the kernel and nothing else.

Run with either virtualenv's python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from h3_producer_provenance import producer_provenance  # noqa: E402


def _load(report: Path) -> tuple[dict, dict, dict[str, torch.Tensor]]:
    r = json.loads(report.read_text())
    steps = [s for s in r["steps"] if s.get("modifier")]
    if len(steps) != 1:
        raise SystemExit(f"{report.name}: expected one modifier-bearing step, found {len(steps)}")
    step = steps[0]
    name = step["modifier"].get("scales_file")
    if not name:
        raise SystemExit(f"{report.name}: its step saved no scales file")
    scales = load_file(str(report.with_name(name)))
    return r, step, scales


def _identity(r: dict, step: dict) -> dict:
    return {
        "rows": step["row_ids"],
        "policy": r["precision_policy"]["policy"],
        "layers": r["model"]["decoder_layers_built"],
        "recipe": r["modifier"]["recipe"],
        "outcome": step["outcome"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="pilot report of the reference arm")
    parser.add_argument("candidate", help="pilot report of the candidate arm")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ref_path = Path(args.reference).expanduser().resolve()
    cand_path = Path(args.candidate).expanduser().resolve()
    ref, ref_step, ref_scales = _load(ref_path)
    cand, cand_step, cand_scales = _load(cand_path)

    ref_id, cand_id = _identity(ref, ref_step), _identity(cand, cand_step)
    differing = sorted(k for k in ref_id if ref_id[k] != cand_id[k])
    ref_kind = ref["attention_kernel"]["kind"]
    cand_kind = cand["attention_kernel"]["kind"]
    report: dict = {
        "comparison": "AWQ scales, mapping by mapping, between two pilot arms",
        "field_under_test": "attention_kernel.kind",
        "reference": {"file": ref_path.name, "attention": ref_kind,
                      "producer": ref["producer"], **ref_id},
        "candidate": {"file": cand_path.name, "attention": cand_kind,
                      "producer": cand["producer"], **cand_id},
        "producer": producer_provenance(__file__),
    }
    if differing:
        report["refused"] = (f"the arms differ on {differing} as well as the kernel, "
                             "so a scale difference could not be attributed")
    elif ref_kind == cand_kind:
        report["refused"] = "both arms used the same attention kind; nothing under test"
    elif set(ref_scales) != set(cand_scales):
        only_ref = sorted(set(ref_scales) - set(cand_scales))
        only_cand = sorted(set(cand_scales) - set(ref_scales))
        report["refused"] = (f"mapping sets differ: only in reference {only_ref[:5]}, "
                             f"only in candidate {only_cand[:5]}")
    if "refused" in report:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"REFUSED: {report['refused']}")
        return 1

    per_mapping = []
    for name in sorted(ref_scales):
        a = ref_scales[name].double()
        b = cand_scales[name].double()
        if a.shape != b.shape:
            per_mapping.append({"smooth": name, "refused": f"shape {tuple(a.shape)} != {tuple(b.shape)}"})
            continue
        diff = a - b
        rel = (diff.abs() / a.abs().clamp_min(1e-12))
        per_mapping.append({
            "smooth": name,
            "channels": int(a.numel()),
            "identical": bool(torch.equal(a, b)),
            "relative_l2": float(diff.norm() / a.norm().clamp_min(1e-12)),
            "max_abs_relative_delta": float(rel.max()),
            "p50_abs_relative_delta": float(rel.median()),
            "channels_over_1pct": int((rel > 0.01).sum()),
            "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)),
        })
    valid = [m for m in per_mapping if "refused" not in m]
    report["mappings"] = per_mapping
    report["summary"] = {
        "mappings": len(per_mapping),
        "identical": sum(1 for m in valid if m["identical"]),
        "max_relative_l2": max((m["relative_l2"] for m in valid), default=None),
        "median_relative_l2": (float(torch.tensor([m["relative_l2"] for m in valid]).median())
                               if valid else None),
        "max_abs_relative_delta_any_channel": max((m["max_abs_relative_delta"] for m in valid),
                                                  default=None),
        "channels_over_1pct_total": sum(m["channels_over_1pct"] for m in valid),
        "channels_total": sum(m["channels"] for m in valid),
        "note": "scales are per input channel of the balance layers; a relative "
                "delta on a scale is a relative change in the smoothing applied "
                "to that channel's weights before quantization",
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    s = report["summary"]
    print(f"{s['mappings']} mappings: {s['identical']} identical, max relative L2 "
          f"{s['max_relative_l2']:.3e}, median {s['median_relative_l2']:.3e}, "
          f"{s['channels_over_1pct_total']}/{s['channels_total']} channels moved over 1%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
