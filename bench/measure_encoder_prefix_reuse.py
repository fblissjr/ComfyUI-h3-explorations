#!/usr/bin/env python3
"""Can the encoder's work on a reference be kept across prompt edits, exactly?

    <comfy venv python> bench/measure_encoder_prefix_reuse.py --capture-dir DIR --out OUT.json
        [--qwen-short-edge N] [--host 127.0.0.1:8188]

`data/results/2026-10-03_prompt_edit_conditioning_cost.json` says what a
prompt edit pays at the reference conditioning node, and that nearly all of it
is the text encoder reading the still. The reference comes before the prompt
in a causal encoder, so that work could be kept across prompt edits. This asks
whether keeping it changes the conditioning.

The measurement itself is a bench-only node
(`bench/comfy_capture_nodes/h3_bench_prefix_reuse`, whose docstring lists the
cases) because the shipped encoder needs the server's dynamic VRAM. This file
submits it: the ref2va finish graph's encoder loader, its still and append
node, the graph's prompt and an edit of it. With `--qwen-short-edge` the
encoder gets its own smaller view of the still, for a quick run of the
plumbing; the default is the view the graphs ship.

The server has to be launched with the bench node directory and
`H3_BENCH_CAPTURE_DIR`; when it was not, this prints the command and exits 2.
"""
import argparse
import asyncio
import json
import sys
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from bench_e2e_h3 import run_once  # noqa: E402

GRAPH = "workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json"
NODE = "H3BenchEncoderPrefixReuse"
BENCH_NODES = REPO / "bench" / "comfy_capture_nodes"
EDIT = " The light shifts as a cloud passes."


def has_node(host):
    try:
        with urllib.request.urlopen(f"http://{host}/object_info/{NODE}", timeout=10) as r:
            return NODE in json.loads(r.read())
    except Exception:                                   # noqa: BLE001 -- absent or no server
        return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True,
                    help="the server's H3_BENCH_CAPTURE_DIR; the node writes its record there")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qwen-short-edge", type=int, default=0,
                    help="0: the encoder sees the VAE's copy, as the graphs ship")
    ap.add_argument("--host", default="127.0.0.1:8188")
    args = ap.parse_args()
    capture = Path(args.capture_dir).resolve()

    if not has_node(args.host):
        capture.mkdir(parents=True, exist_ok=True)
        cfg = capture / "bench_nodes.yaml"
        cfg.write_text(f"h3_bench:\n  custom_nodes: {BENCH_NODES}\n")
        print("The server does not have the bench node. Launch it with:\n"
              f"  H3_BENCH_CAPTURE_DIR={capture} ./start.sh default "
              f"--extra-model-paths-config {cfg}")
        return 2

    full = json.loads((REPO / GRAPH).read_text())
    g = {k: full[k] for k in ("2", "15", "50")}
    assert g["50"]["class_type"] == "MiniMaxH3AppendRefImage"
    if args.qwen_short_edge:
        g["50"]["inputs"]["qwen_view"] = "separate"
        g["50"]["inputs"]["qwen_view.qwen_short_edge"] = args.qwen_short_edge
    prompt = full["5"]["inputs"]["prompt"]
    name = f"prefix_reuse_{uuid.uuid4().hex[:8]}"
    g["900"] = {"class_type": NODE, "inputs": {
        "clip": ["2", 0], "references": ["50", 0], "prompt": prompt,
        "edit": prompt.rstrip() + EDIT, "name": name}}

    _total, per_node, err = (await run_once(args.host, g, uuid.uuid4().hex, 1800.0))[:3]
    if err:
        print(f"FAILED: {err}")
        return 1
    measured = json.loads((capture / f"{name}.json").read_text())
    record = {
        "what": "the encoder's reference prefix kept across prompt edits: is the conditioning unchanged",
        "tool": "bench/measure_encoder_prefix_reuse.py",
        "node": "bench/comfy_capture_nodes/h3_bench_prefix_reuse",
        "graph": GRAPH, "encoder": g["2"]["inputs"]["encoder_name"],
        "still": g["15"]["inputs"]["image"], "append": g["50"]["inputs"],
        "edit_appended": EDIT, "node_seconds": round(per_node.get("900", 0.0), 1),
        **measured,
    }
    print(json.dumps(record, indent=1))
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
