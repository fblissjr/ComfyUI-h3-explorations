#!/usr/bin/env python3
"""Controls for `pipeline_telemetry.py` and `bench/telemetry_report.py`.

Run with the ComfyUI venv python; needs no server and no CUDA
(`CUDA_VISIBLE_DEVICES=` is fine). NVML is exercised when the driver library
is present and skipped, and said so, when it is not.

What each case guards:
  inert_unarmed        no spec, or a spec without dir=, arms nothing. The pack
                       imports this module in every server.
  never_caches         the provider's should_cache returns False on both calls,
                       so core schedules no lookup and no store: telemetry can
                       never change what executes.
  boundaries_recorded  a started record gets header, prompt_start, a node's
                       start and end, and prompt_end, in order.
  log_lines_parse      core's staging line yields its numbers, and the handler
                       writes a row. The handler once raised on every record
                       (a keyword collision in emit), which this catches.
  report_arithmetic    two samples one second apart at 1024 KB/s integrate to
                       one MiB; a node with one sample reports None, not 0.
  nvml_reads           the ctypes binding returns memory and PCIe fields.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))
sys.path.insert(1, str(REPO))
sys.path.insert(2, str(REPO / "bench"))

import pipeline_telemetry as pt  # noqa: E402
import telemetry_report as tr  # noqa: E402

results = []


def case(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<22} {detail}")


def main() -> int:
    case("inert_unarmed", not pt.arm("") and not pt.arm("interval_ms=100"),
         "empty spec and a spec without dir= both disarmed")
    spec = pt.parse_spec("dir=/x,interval_ms=100,blocks=0")
    case("spec_parse", spec == {"dir": "/x", "interval_ms": 100, "blocks": False}, str(spec))

    with tempfile.TemporaryDirectory() as tmp:
        rec = pt._Recorder(pt.parse_spec(f"dir={tmp},interval_ms=50"))
        ctx = types.SimpleNamespace(node_id="10", class_type="SamplerCustomAdvanced")
        rec.start("0123456789abcdef")
        a = rec.provider.should_cache(ctx)
        b = rec.provider.should_cache(ctx, object())
        case("never_caches", a is False and b is False, f"lookup {a}, store {b}")
        line = ("Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. "
                "0 patches attached. Force pre-loaded 210 weights: 1175 KB.")
        import logging
        rec.log_handler.emit(logging.LogRecord("comfy", 20, "", 0, line, None, None))
        rec.end("0123456789abcdef")
        files = list(Path(tmp).glob("*.jsonl"))
        rows = tr.load(files[0]) if files else []
        kinds = [r["type"] for r in rows if r["type"] != "sample"]
        want = ["header", "prompt_start", "node_start", "node_end", "log", "prompt_end"]
        case("boundaries_recorded", kinds == want, str(kinds))
        log = next((r for r in rows if r["type"] == "log"), {})
        case("log_lines_parse",
             log.get("kind") == "dynamic_prepared" and log.get("groups", [])[:3] == ["MiniMaxH3", "19995", "0"],
             str(log.get("groups")))

    synth = [
        {"t": 0.0, "seq": 1, "type": "header", "prompt_id": "p", "graph": {"nodes": {"1": "A", "2": "B", "3": "C"}}},
        {"t": 1.0, "seq": 2, "type": "node_start", "node": "1", "class_type": "A", "models": []},
        {"t": 1.0, "seq": 3, "type": "sample", "node": "1", "gpu": {"pcie_rx_kbs": 1024}, "host": {}},
        {"t": 2.0, "seq": 4, "type": "sample", "node": "1", "gpu": {"pcie_rx_kbs": 1024}, "host": {}},
        {"t": 2.0, "seq": 5, "type": "node_end", "node": "1", "class_type": "A", "models": []},
        {"t": 2.1, "seq": 6, "type": "node_start", "node": "2", "class_type": "B", "models": []},
        {"t": 2.2, "seq": 7, "type": "sample", "node": "2", "gpu": {"pcie_rx_kbs": 1024}, "host": {}},
        {"t": 2.3, "seq": 8, "type": "node_end", "node": "2", "class_type": "B", "models": []},
    ]
    s = tr.phases(synth)
    n1 = s["nodes"][0]["pcie_to_gpu_gib"]
    raw = tr._integral_kb([x for x in synth if x["type"] == "sample"][:2], "pcie_rx_kbs")
    n2 = s["nodes"][1]["pcie_to_gpu_gib"]
    case("report_arithmetic", raw == 1024 * 1024 and n2 is None and s["not_observed"] == ["3"],
         f"integral {raw} B (node 1 {n1} GiB), one-sample node {n2}, not observed {s['not_observed']}")

    nv = pt.Nvml(None)
    if nv.lib is None:
        print("  skip  nvml_reads             no libnvidia-ml.so.1")
    else:
        smp = nv.sample()
        case("nvml_reads", "used" in smp and smp.get("pcie_rx_kbs") is not None,
             json.dumps({k: smp.get(k) for k in ("used", "pcie_rx_kbs", "pcie_gen", "pcie_width")}))

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} passed" + (f"; FAILED {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
