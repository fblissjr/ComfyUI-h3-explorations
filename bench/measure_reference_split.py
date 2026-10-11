#!/usr/bin/env python3
"""The two-node reference path against core's one pass, on the real encoder.

    <comfy venv python> bench/measure_reference_split.py --capture-dir DIR --out OUT.json
        [--second IMAGE] [--third IMAGE] [--later-view N] [--host 127.0.0.1:8188]

`reference_encode.py` encodes the references apart from the prompt, each
reference's span chained to the ones before it within one encode (nothing is
kept between encodes: the session store was removed on 2026-10-10). This
submits the bench node `H3BenchSplitReferences`
(`bench/comfy_capture_nodes/h3_bench_prefix_reuse`), which runs that path
beside `clip.encode_from_tokens_scheduled` on:

  one reference       the ref2va finish graph's still at its shipped view,
                      the graph's prompt, then an edit of the prompt on a
                      second encode, whose rows must equal the first's
  two chains          that still followed by a second one, then by a third:
                      the first reference's rows must be the same both
                      times, the later one continued on it

and records, per reference span and for the prompt rows, `torch.equal` and
the distance when they differ, the token tags, and the seconds each step took
(the split path's include moving the kept keys and values to the card).

The later stills get their own smaller encoder view (`--later-view`), kept
from when the chains had to fit a store's budget. The server has to be launched
with the bench node directory and `H3_BENCH_CAPTURE_DIR`; when it was not,
this prints the command and exits 2.
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
NODE = "H3BenchSplitReferences"
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
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--second", default="h3_refs/scene_loft_couch_duo_2752x1536.png")
    ap.add_argument("--third", default="h3_refs/product_soccer_jersey_1600x1600.png")
    ap.add_argument("--later-view", type=int, default=1024)
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
    for load, append, image in (("915", "951", args.second), ("916", "952", args.third)):
        g[load] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g[append] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": dict(
            g["50"]["inputs"], image=[load, 0], references=["50", 0],
            **{"qwen_view": "separate", "qwen_view.qwen_short_edge": args.later_view})}
    prompt = full["5"]["inputs"]["prompt"]
    name = f"split_references_{uuid.uuid4().hex[:8]}"
    g["900"] = {"class_type": NODE, "inputs": {
        "clip": ["2", 0], "references_a": ["50", 0], "references_ab": ["951", 0],
        "references_ac": ["952", 0], "prompt": prompt, "edit": prompt.rstrip() + EDIT,
        "name": name}}

    _total, per_node, err = (await run_once(args.host, g, uuid.uuid4().hex, 2400.0))[:3]
    if err:
        print(f"FAILED: {err}")
        return 1
    measured = json.loads((capture / f"{name}.json").read_text())
    record = {
        "what": "the two-node reference path (reference_encode.py) against core's one pass",
        "tool": "bench/measure_reference_split.py",
        "node": "bench/comfy_capture_nodes/h3_bench_prefix_reuse::H3BenchSplitReferences",
        "graph": GRAPH, "encoder": g["2"]["inputs"]["encoder_name"],
        "stills": {"a": g["15"]["inputs"]["image"], "b": args.second, "c": args.third},
        "views": {"a": "shared with the VAE copy, as shipped", "b_and_c": args.later_view},
        "edit_appended": EDIT, "node_seconds": round(per_node.get("900", 0.0), 1),
        **measured,
    }
    print(json.dumps(record, indent=1))
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
