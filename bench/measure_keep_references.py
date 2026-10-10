#!/usr/bin/env python3
"""`keep_references` on `MiniMaxH3ReferenceConditioning`, on the real encoder.

    <comfy venv python> bench/measure_keep_references.py --capture-dir DIR --out OUT.json
        [--second IMAGE] [--host 127.0.0.1:8188]

The switch's two functions were held to core's one pass in
`data/results/2026-10-03_reference_split.json`; this runs the node itself.
The conditioning side of the ref2va finish graph, with the conditioning node
twice: as shipped, and with `keep_references` on. A bench node compares what
the two hand on (rows, token tags, the reference latents). Then the prompt is
edited and both run again, so each node's seconds on a prompt edit are read
off the websocket feed; then the same with a second still appended.

No DiT is loaded and nothing is rendered. The server has to be launched with
the bench node directory and `H3_BENCH_CAPTURE_DIR`; when it was not, this
prints the command and exits 2.
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
NODE = "H3BenchCompareConditioning"
BENCH_NODES = REPO / "bench" / "comfy_capture_nodes"
EDITS = ("", " The light shifts as a cloud passes.", " A bell rings twice in the distance.")


def has_node(host):
    try:
        with urllib.request.urlopen(f"http://{host}/object_info/{NODE}", timeout=10) as r:
            return NODE in json.loads(r.read())
    except Exception:                                   # noqa: BLE001 -- absent or no server
        return False


def graph(full, edit, second, name):
    g = {k: json.loads(json.dumps(full[k])) for k in ("2", "3", "4", "15", "50", "27", "5")}
    references = ["50", 0]
    if second:
        g["915"] = {"class_type": "LoadImage", "inputs": {"image": second}}
        # The second still's encoder view is its own and smaller, so the one
        # pass over both fits the card beside the encoder.
        g["951"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": dict(
            g["50"]["inputs"], image=["915", 0], references=["50", 0],
            **{"qwen_view": "separate", "qwen_view.qwen_short_edge": 1024})}
        references = ["951", 0]
    g["5"]["inputs"]["references"] = references
    g["5"]["inputs"]["prompt"] = g["5"]["inputs"]["prompt"].rstrip() + edit
    g["905"] = json.loads(json.dumps(g["5"]))
    g["905"]["inputs"]["keep_references"] = True
    g["900"] = {"class_type": NODE, "inputs": {
        "reference": ["5", 0], "candidate": ["905", 0], "name": name}}
    return g


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--second", default="h3_refs/scene_loft_couch_duo_2752x1536.png")
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
    client, runs = uuid.uuid4().hex, []
    for stills, second in (("one still", None), ("two stills", args.second)):
        for n, edit in enumerate(EDITS):
            name = f"keep_references_{uuid.uuid4().hex[:8]}"
            _total, per_node, err = (await run_once(args.host, graph(full, edit, second, name),
                                                    client, 1800.0))[:3]
            if err:
                print(f"FAILED ({stills}, run {n}): {err}")
                return 1
            row = {"stills": stills, "run": "first" if n == 0 else f"prompt edit {n}",
                   "one_pass_node_s": round(per_node.get("5", 0.0), 2),
                   "keep_references_node_s": round(per_node.get("905", 0.0), 2),
                   "compare": json.loads((capture / f"{name}.json").read_text())}
            runs.append(row)
            print(json.dumps(row), flush=True)
    record = {
        "what": "MiniMaxH3ReferenceConditioning with keep_references on, against the same node as shipped",
        "tool": "bench/measure_keep_references.py",
        "node": "bench/comfy_capture_nodes/h3_bench_prefix_reuse::H3BenchCompareConditioning",
        "graph": GRAPH, "second_still": args.second,
        "state": "no DiT resident; nothing rendered; both nodes in one prompt, so each run's two "
                 "encodes share the card, the one pass first by node order",
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
