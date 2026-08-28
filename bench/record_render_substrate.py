#!/usr/bin/env python3
"""What each render RAN UNDER, so a time or memory number can be checked.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Needs a live
server; reads `/history` and allocates nothing. Safe during a render.

**Not a check.** It asserts nothing and grades nothing. It records the
conditions that decide whether a duration or a peak means anything, because on
2026-08-27 four separate readings were built on numbers whose conditions nobody
had recorded:

  - `C_pdd8` at two seeds gave OOM and success on identical graphs. Read as
    nondeterminism for an hour. It was cache position: the first arm after a
    restart paid the whole model load, the second hit cache on the encoder,
    the conditioning and the loader.
  - A fits-or-does-not-fit oracle for `reuse_qkv_memory` was designed on the
    assumption that the outcome was a function of the graph.
  - `D_start0`'s timing compared mid-batch cache-hit arms against a control
    that had paid a full load.
  - A same-seed repeat was queued that would have been a near-total cache hit
    and "succeeded" without executing the thing that failed.

`bench/instrument_render_occupancy.py` could not have caught any of them: it
resolves GPU occupancy WITHIN one render and refuses to start when the card is
busy, so it says nothing about state carried ACROSS renders. That is the gap
this fills, and it is why this exists rather than a flag on that.

The two facts that make it cheap. ComfyUI's cache is keyed by input signature
and persists across prompts in a session (`main.py`'s `RAM_PRESSURE` default),
and it reports what it skipped: every prompt's `execution_cached` message
carries the node list. And `/history` resets when the server restarts, so a
prompt's index in it IS its position in the session.

What this cannot recover post-hoc: host and GPU memory at the time each render
ran. Those need sampling while the queue runs, which is `--sample`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = "http://127.0.0.1:8188"


def _get(path: str):
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=15) as r:
        return json.loads(r.read())


def _label(prompt: dict) -> str | None:
    """The arm, by the output prefix it writes, which is how a human finds it."""
    for node in prompt.values():
        if isinstance(node, dict):
            pre = node.get("inputs", {}).get("filename_prefix")
            if isinstance(pre, str):
                return pre.rsplit("/", 1)[-1]
    return None


def session_rows() -> list[dict]:
    hist = _get("/history")
    rows = []
    for position, (pid, entry) in enumerate(hist.items(), start=1):
        status = entry.get("status", {})
        msgs = status.get("messages", [])
        stamp = {n: p for n, p in msgs if isinstance(p, dict)}
        start = stamp.get("execution_start", {}).get("timestamp")
        end = (stamp.get("execution_success", {}).get("timestamp")
               or stamp.get("execution_error", {}).get("timestamp"))
        cached = stamp.get("execution_cached", {}).get("nodes", [])
        prompt = (entry.get("prompt") or [None, None, {}])[2] or {}
        err = stamp.get("execution_error", {})
        rows.append({
            "position_in_session": position,
            "prompt_id": pid,
            "label": _label(prompt),
            "status": status.get("status_str"),
            "seconds": round((end - start) / 1000, 1) if start and end else None,
            "nodes_total": len(prompt) or None,
            "nodes_cached": len(cached),
            "cache_hit_fraction": (round(len(cached) / len(prompt), 3)
                                   if prompt else None),
            "failed_node": err.get("node_type"),
            "error_class": (err.get("exception_type") or "").rsplit(".", 1)[-1] or None,
        })
    return rows


def sample_substrate() -> dict:
    """Host and device state right now. One sample, no allocation."""
    out = {"sampled_at": time.time()}
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            mem[k] = int(v.split()[0])
        out["host_available_gib"] = round(mem["MemAvailable"] / 1048576, 1)
        out["host_total_gib"] = round(mem["MemTotal"] / 1048576, 1)
        out["swap_total_gib"] = round(mem.get("SwapTotal", 0) / 1048576, 1)
        out["swap_used_gib"] = round(
            (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1048576, 1)
    except Exception as exc:
        out["host_error"] = str(exc)
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        used, total, util = (int(x) for x in q.stdout.strip().split(", "))
        out["gpu_used_mib"], out["gpu_total_mib"], out["gpu_util"] = used, total, util
    except Exception as exc:
        out["gpu_error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write a JSON record here")
    ap.add_argument("--sample", action="store_true",
                    help="also take one host/GPU sample now")
    args = ap.parse_args()

    rows = session_rows()
    print(f"{'#':>3} {'label':34s} {'status':8s} {'sec':>7} {'cached':>10}  failure")
    for r in rows:
        cf = (f"{r['nodes_cached']}/{r['nodes_total']}"
              if r["nodes_total"] else "-")
        fail = f"{r['error_class']} at {r['failed_node']}" if r["error_class"] else ""
        print(f"{r['position_in_session']:>3} {str(r['label'])[:34]:34s} "
              f"{str(r['status']):8s} {str(r['seconds']):>7} {cf:>10}  {fail}")

    first = [r for r in rows if r["position_in_session"] == 1]
    if first:
        print(f"\nposition 1 paid the full load: {first[0]['label']} "
              f"({first[0]['nodes_cached']} nodes cached)")
    print("A duration or a peak is comparable only across rows with similar "
          "position and cache fraction.")

    record = {
        "question": "what conditions did each render in this server session run "
                    "under: position, cache state, outcome",
        "not_a_check": "records; asserts nothing",
        "path_policy": "logical identifiers only",
        "session_rows": rows,
    }
    if args.sample:
        record["substrate_now"] = sample_substrate()
        s = record["substrate_now"]
        print(f"\nnow: host {s.get('host_available_gib')} GiB avail, "
              f"swap {s.get('swap_used_gib')}/{s.get('swap_total_gib')} GiB, "
              f"gpu {s.get('gpu_used_mib')}/{s.get('gpu_total_mib')} MiB")
    if args.out:
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
