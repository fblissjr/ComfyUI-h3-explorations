#!/usr/bin/env python3
"""Read a pipeline telemetry record into per-node phases.

A record is one prompt's JSONL, written by `pipeline_telemetry.py` in a server
started with `H3_TELEMETRY` (`docs/pipeline_telemetry.md` owns the schema).
This prints one row per node the prompt executed, in order, with what the
card, the bus and the host did during it, and the models dynamic VRAM held
at its start and end. `--json` writes the same as a structured summary.

    python bench/telemetry_report.py RECORD.jsonl [--json OUT.json]

Pure reading: no server, no GPU, no ComfyUI import. Integrals over the
sampler's rows are trapezoids between consecutive samples inside a node's
window; NVML's PCIe figure is a short-window rate, so the byte totals are
estimates at the sampler's resolution, and a node shorter than two samples
has none (`None`, never 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GiB = 1024 ** 3
MiB = 1024 ** 2


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _integral_kb(samples: list[dict], key: str):
    """KB/s rate integrated over time -> bytes, trapezoid rule."""
    pts = [(s["t"], s["gpu"].get(key)) for s in samples if s.get("gpu", {}).get(key) is not None]
    if len(pts) < 2:
        return None
    total = 0.0
    for (t0, a), (t1, b) in zip(pts, pts[1:]):
        total += (a + b) / 2 * (t1 - t0) * 1024
    return total


def _delta(samples: list[dict], section: str, key: str):
    vals = [s[section].get(key) for s in samples if s.get(section, {}).get(key) is not None]
    return (vals[-1] - vals[0]) if len(vals) >= 2 else None


def _max(samples: list[dict], section: str, key: str):
    vals = [s[section].get(key) for s in samples if s.get(section, {}).get(key) is not None]
    return max(vals) if vals else None


def _watts(mw):
    return None if mw is None else round(mw / 1000, 1)


def _models_brief(models: list[dict] | None) -> list[dict]:
    out = []
    for m in models or []:
        v = m.get("vbar") or {}
        blk = v.get("block_resident")
        out.append({
            "class": m.get("class"), "dynamic": m.get("dynamic"),
            "size_mib": _mib(m.get("size")), "on_gpu_mib": _mib(m.get("loaded")),
            "pinned_mib": _mib(m.get("pinned")),
            "blocks_resident": (None if not blk else
                                round(sum(x for x in blk if x is not None) / len(blk), 3)),
        })
    return out


def _mib(x):
    return None if x is None else round(x / MiB)


def phases(rows: list[dict]) -> dict:
    header = next((r for r in rows if r["type"] == "header"), {})
    samples = [r for r in rows if r["type"] == "sample"]
    logs = [r for r in rows if r["type"] == "log"]
    starts = {}
    out_nodes = []
    for r in rows:
        if r["type"] == "node_start":
            starts[r["node"]] = r
        elif r["type"] == "node_end" and r["node"] in starts:
            s = starts.pop(r["node"])
            win = [x for x in samples if s["t"] <= x["t"] <= r["t"]]
            out_nodes.append({
                "node": r["node"], "class_type": r.get("class_type"),
                "start_s": s["t"], "seconds": round(r["t"] - s["t"], 3),
                "samples": len(win),
                "gpu_used_max_mib": _mib(_max(win, "gpu", "used")),
                "aimdo_vram_max_mib": _mib(max((x["aimdo_vram"] for x in win
                                                if x.get("aimdo_vram") is not None), default=None)),
                "pcie_to_gpu_gib": _gib(_integral_kb(win, "pcie_rx_kbs")),
                "pcie_from_gpu_gib": _gib(_integral_kb(win, "pcie_tx_kbs")),
                "util_gpu_mean": _mean(win, "gpu", "util_gpu"),
                "power_w_mean": _watts(_mean(win, "gpu", "power_mw")),
                "proc_anon_delta_mib": _mib(_delta(win, "host", "proc_rssanon")),
                "proc_file_delta_mib": _mib(_delta(win, "host", "proc_rssfile")),
                "page_cache_delta_mib": _mib(_delta(win, "host", "host_cached")),
                "disk_read_mib": _mib(_delta(win, "host", "proc_io_read_bytes")),
                "major_faults": _delta(win, "host", "host_pgmajfault"),
                "cpu_s": _cpu(win),
                "torch_peak_allocated_mib": _mib((r.get("device") or {}).get("torch_peak_allocated")),
                "models_at_start": _models_brief(s.get("models")),
                "models_at_end": _models_brief(r.get("models")),
                "log_events": [x.get("kind") for x in logs if s["t"] <= x["t"] <= r["t"]],
            })
    executed = {n["node"] for n in out_nodes}
    graph = (header.get("graph") or {}).get("nodes") or {}
    return {
        "prompt_id": header.get("prompt_id"), "schema": header.get("schema"),
        "substrate": _redacted(header.get("substrate")),
        "total_s": rows[-1]["t"] if rows else None,
        "nodes": out_nodes,
        "not_observed": sorted(set(graph) - executed, key=lambda k: (len(k), k)),
        "unclosed": sorted(starts),
        "log_counts": _counts(x.get("kind") for x in logs),
    }


def _redacted(sub):
    """The summary is committed; the record is not. Absolute paths in the
    server's argv (output, input and temp directories) say where this machine
    keeps things, so the summary names the flag and drops the path."""
    if not isinstance(sub, dict):
        return sub
    out = dict(sub)
    out["argv"] = ["<path>" if isinstance(a, str) and a.startswith("/") else a
                   for a in sub.get("argv") or []]
    return out


def _gib(x):
    return None if x is None else round(x / GiB, 2)


def _mean(win, section, key):
    vals = [s[section].get(key) for s in win if s.get(section, {}).get(key) is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _cpu(win):
    u = _delta(win, "host", "proc_cpu_user_s")
    s = _delta(win, "host", "proc_cpu_sys_s")
    return None if u is None or s is None else round(u + s, 2)


def _counts(it):
    out: dict = {}
    for k in it:
        out[k] = out.get(k, 0) + 1
    return out


def _fmt(v):
    return "-" if v is None else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path)
    ap.add_argument("--json", type=Path, help="write the structured summary here")
    args = ap.parse_args()
    summary = phases(load(args.record))
    cols = ["node", "class_type", "seconds", "gpu_used_max_mib", "aimdo_vram_max_mib",
            "pcie_to_gpu_gib", "pcie_from_gpu_gib", "util_gpu_mean", "proc_anon_delta_mib",
            "page_cache_delta_mib", "disk_read_mib", "major_faults", "cpu_s"]
    print(f"prompt {summary['prompt_id']}  total {summary['total_s']} s")
    print("  ".join(cols))
    for n in summary["nodes"]:
        print("  ".join(_fmt(n.get(c)) for c in cols))
        for label in ("models_at_start", "models_at_end"):
            for m in n[label]:
                print(f"    {label[7:]:<9} {m['class']:<24} on_gpu {_fmt(m['on_gpu_mib'])}"
                      f"/{_fmt(m['size_mib'])} MiB  pinned {_fmt(m['pinned_mib'])}"
                      f"  blocks_resident {_fmt(m['blocks_resident'])}")
    print(f"not observed (cached, unused, or uncacheable): {summary['not_observed']}")
    if summary["unclosed"]:
        print(f"started but never ended: {summary['unclosed']}")
    print(f"log events: {summary['log_counts']}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
