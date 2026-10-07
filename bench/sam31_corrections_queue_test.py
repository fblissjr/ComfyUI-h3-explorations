#!/usr/bin/env python3
"""The SAM 3.1 Corrections node's last acceptance: on a server's queue, with an H3 render between two reads.

`bench/check_sam31_corrections_on_card.py` shows the node on the real model and the server's patcher class in one
process. What it cannot show is the server itself: the executor's cache, its order of loading, and above all what
happens to the corrections when a video model takes the card and SAM is evicted, which every real render does. This
tool runs that, without putting an unaccepted node on the main server:

    build   make a scratch base directory: every entry of the ComfyUI checkout linked, every installed pack linked
            except this one, and this one COPIED at HEAD with one addition that exists only in the copy: the
            test-only probe node (`bench/sam31_probe_node.py`) in the pack's node list. (For the node's first
            acceptance, before it was registered, the copy registered the node too.) Start a server on it with
            the main server's own launch script and `--port P --base-directory BASE`.
    run     against that server: (1) one graph in which ComfyUI's stock checkpoint loader feeds a chain of probes in
            the order stock, corrected, stock, corrected twice over, corrected, stock; (2) an H3 text-to-video prompt
            that really samples, so the video model takes the card; (3) the chain again; (4) after freeing all models
            the chain a third time; (5) the node run AGAIN, as a second node on the same stock loader output, while
            its first corrected output is still loaded (added 2026-10-07 after a fault on exactly that path was found
            by review: the first acceptance never ran the node while a corrected clip was loaded). Each probe reports, from
            inside the server, the range that reached SAM's first layer, the activation the text encoder ran, the
            patcher class and whether SAM was on the card before its call. Results come back through `/history`.
    render  print the record's table from the json `run` wrote.

PASSES when every stock probe reads 0..1 and the shipped activation, every corrected probe exactly -1..1 and exact
GELU, before and after the H3 prompt, and the H3 prompt ran. A corrected probe that reads 0..1 anywhere is a patch
lost, and the row says which read lost it.

    <python> bench/sam31_corrections_queue_test.py build --base BASE
    <ComfyUI>/start.sh --port 8191 --base-directory BASE          # by hand; its log says whether dynamic VRAM is on
    <python> bench/sam31_corrections_queue_test.py run --server http://127.0.0.1:8191 --json J
    <python> bench/sam31_corrections_queue_test.py render --json J

The H3 prompt is the shipped text-to-video graph cut down (few steps, few frames, nothing saved to the output share):
it has to sample, not to look like anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO / "workflows"))

#: reasoned: the probes of one chain, in the order they run. Both orders of the pair, the corrected one twice running, and
#: the node wired twice in a row.
CHAIN = (("stock", "stock"), ("corrected", "corrected"), ("stock again", "stock"), ("corrected twice over", "twice"),
         ("corrected again", "corrected"), ("stock last", "stock"))
#: reasoned: enough to make the video model take the card and sample; the shipped graph's own values otherwise.
H3_STEPS = 2
H3_LENGTH = 73
H3_GRAPH = "h3_text_to_video_api.json"


def cmd_build(a):
    base = Path(a.base)
    if base.exists() and any(base.iterdir()):
        raise SystemExit(f"{base} is not empty; give a new directory")
    (base / "custom_nodes").mkdir(parents=True)
    for entry in sorted(COMFY.iterdir()):
        if entry.name in ("custom_nodes", "__pycache__", "user", "input", "output", "temp") or entry.name.startswith("."):
            continue
        (base / entry.name).symlink_to(entry)
    for name in ("user", "input"):          # the server writes here; give it its own
        (base / name).mkdir()
    for entry in sorted((COMFY / "custom_nodes").iterdir()):
        if entry.name in (REPO.name, "__pycache__") or entry.name.startswith("."):
            continue
        (base / "custom_nodes" / entry.name).symlink_to(entry)
    copy = base / "custom_nodes" / REPO.name
    copy.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(copy)], input=archive, check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (copy / "sam31_probe_node.py").write_text((REPO / "bench" / "sam31_probe_node.py").read_text())   # the working tree's: it may not be committed yet
    nodes = (copy / "nodes.py").read_text()
    imports = "from .sam31_corrections import MiniMaxH3SAM31Corrections\n"
    listed = "                MiniMaxH3SAM31Corrections]\n"
    if nodes.count(imports) != 1 or nodes.count(listed) != 1:
        raise SystemExit("nodes.py no longer has the two lines this edit hangs on (the Corrections node's import, and its "
                         "entry as the last of the node list); read it and update `cmd_build`")
    nodes = nodes.replace(imports, imports + "from .sam31_probe_node import H3TestSAM31FirstLayer             # SCRATCH COPY ONLY\n")
    nodes = nodes.replace(listed, "                MiniMaxH3SAM31Corrections,\n                H3TestSAM31FirstLayer]   # SCRATCH COPY ONLY\n")
    (copy / "nodes.py").write_text(nodes)
    print(f"built {base}\n  this pack copied at {head}, with the probe registered in the copy only\n"
          f"  start it:  {COMFY / 'start.sh'} --port <port> --base-directory {base}")


def _post(server: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode()
    return json.loads(text) if text.strip() else {}


def _get(server: str, path: str) -> dict:
    with urllib.request.urlopen(server + path, timeout=60) as r:
        return json.loads(r.read().decode())


def _wait(server: str, prompt_id: str, limit: float) -> dict:
    began = time.time()
    while time.time() - began < limit:
        h = _get(server, f"/history/{prompt_id}")
        if prompt_id in h:
            return h[prompt_id]
        time.sleep(2.0)
    raise SystemExit(f"prompt {prompt_id} did not finish in {limit:.0f} s")


def probe_graph(nonce: int) -> dict:
    """Stock loader, the node, the node again on its own output, and the chain of probes wired to run in `CHAIN`'s order."""
    from h3_config import SEGMENTER
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SEGMENTER}},
         "2": {"class_type": "MiniMaxH3SAM31Corrections", "inputs": {"segmenter": ["1", 0], "segmenter_clip": ["1", 1],
                                                                       "correct_image_range": True, "correct_text_activation": True}},
         "3": {"class_type": "MiniMaxH3SAM31Corrections", "inputs": {"segmenter": ["2", 0], "segmenter_clip": ["2", 1],
                                                                       "correct_image_range": True, "correct_text_activation": True}}}
    source = {"stock": "1", "corrected": "2", "twice": "3"}
    previous = None
    for i, (label, which) in enumerate(CHAIN):
        node = str(10 + i)
        inputs = {"segmenter": [source[which], 0], "segmenter_clip": [source[which], 1], "label": label, "nonce": nonce, "detect": True}
        if previous is not None:
            inputs["after"] = [previous, 0]
        g[node] = {"class_type": "H3TestSAM31FirstLayer", "inputs": inputs}
        previous = node
    return g


def rerun_graphs(nonce: int) -> tuple[dict, dict]:
    """Two prompts for the node run AGAIN while its first corrected output is still loaded.

    The text encoder is one module shared by every clone, and while a corrected clip is loaded its patch is what the
    module shows. The first prompt ends on a text-only probe of the corrected pair, so the corrected text encoder is the
    last thing loaded. The second adds a second Corrections node on the same stock loader output and probes ITS output.
    A node that decided what to patch from the live module would attach nothing there.

    The second node has `correct_image_range` OFF. That is not incidental: the executor keys its cache on a node's class
    and inputs, not its id, so a second node with the same inputs is never run; it is handed the first one's output
    (measured 2026-10-07: with identical inputs this step passed on the faulty node). A different input makes it run.
    So its probe must read the stock range and exact GELU.
    """
    from h3_config import SEGMENTER
    base = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SEGMENTER}},
            "2": {"class_type": "MiniMaxH3SAM31Corrections", "inputs": {"segmenter": ["1", 0], "segmenter_clip": ["1", 1],
                                                                          "correct_image_range": True, "correct_text_activation": True}}}
    first = dict(base, **{"30": {"class_type": "H3TestSAM31FirstLayer", "inputs": {"segmenter": ["2", 0], "segmenter_clip": ["2", 1],
                                                                                   "label": "corrected, text only, left loaded", "nonce": nonce,
                                                                                   "detect": False}}})
    second = dict(base, **{"20": {"class_type": "MiniMaxH3SAM31Corrections", "inputs": {"segmenter": ["1", 0], "segmenter_clip": ["1", 1],
                                                                                        "correct_image_range": False, "correct_text_activation": True}},
                           "31": {"class_type": "H3TestSAM31FirstLayer", "inputs": {"segmenter": ["20", 0], "segmenter_clip": ["20", 1],
                                                                                    "label": "the node run again (range off, activation on) while its first output is loaded", "nonce": nonce,
                                                                                    "detect": True}}})
    return first, second


def h3_graph(seed: int) -> dict:
    g = json.loads((REPO / "workflows" / H3_GRAPH).read_text())
    for node in g.values():
        if not isinstance(node, dict):
            continue
        kind, ins = node.get("class_type"), node.get("inputs", {})
        if kind == "BasicScheduler":
            ins["steps"] = H3_STEPS
        elif kind == "MiniMaxH3Resolution":
            ins["length"] = H3_LENGTH
        elif kind == "RandomNoise":
            ins["noise_seed"] = seed
        elif kind == "VHS_VideoCombine":
            ins["save_output"] = False
            ins["filename_prefix"] = "sam31_corrections_queue_test"
    return g


def _probes(entry: dict) -> list[dict]:
    out = []
    for i in range(len(CHAIN)):
        got = entry.get("outputs", {}).get(str(10 + i), {})
        text = next((v[0] if isinstance(v, list) and v else v for k, v in got.items() if k in ("text", "string", "result")), None)
        out.append(json.loads(text) if isinstance(text, str) else {"label": CHAIN[i][0], "missing": True, "raw": got})
    return out


def verdict(probe: dict, which: str) -> str:
    if probe.get("missing") or probe.get("first_layer") is None:
        return "NO READING"
    low, high = probe["first_layer"]
    exact = probe.get("text_ran_exact_gelu")
    if which == "stock":
        ok = abs(low) < 0.02 and abs(high - 1) < 0.02 and exact is False
    else:
        ok = abs(low + 1) < 1e-3 and abs(high - 1) < 1e-3 and exact is True
    return "ok" if ok else "WRONG"


def cmd_run(a):
    server = a.server.rstrip("/")
    info = _get(server, "/object_info/H3TestSAM31FirstLayer")
    if "H3TestSAM31FirstLayer" not in info or "MiniMaxH3SAM31Corrections" not in _get(server, "/object_info/MiniMaxH3SAM31Corrections"):
        raise SystemExit(f"{server} does not serve the node and the probe; this is run against the scratch server `build` describes")
    R = {"server": server, "stats_before": _get(server, "/system_stats").get("devices"), "rounds": []}

    def chain(name: str, nonce: int):
        pid = _post(server, "/prompt", {"prompt": probe_graph(nonce)})["prompt_id"]
        entry = _wait(server, pid, 600)
        probes = _probes(entry)
        for p, (label, which) in zip(probes, CHAIN):
            p["verdict"] = verdict(p, "stock" if which == "stock" else "corrected")
        R["rounds"].append({"round": name, "status": entry.get("status", {}).get("status_str"), "probes": probes})
        for p in probes:
            print(name, json.dumps(p), flush=True)

    chain("before the H3 prompt", 1)
    pid = _post(server, "/prompt", {"prompt": h3_graph(a.seed)})["prompt_id"]
    entry = _wait(server, pid, 1800)
    R["h3_prompt"] = {"graph": H3_GRAPH, "steps": H3_STEPS, "length": H3_LENGTH, "status": entry.get("status", {}).get("status_str"),
                      "outputs": sorted(entry.get("outputs", {}))}
    print("h3", json.dumps(R["h3_prompt"]), flush=True)
    chain("after the H3 prompt", 2)
    _post(server, "/free", {"unload_models": True, "free_memory": True})
    chain("after /free", 3)
    # the node run again while its first corrected output is still loaded
    first, second = rerun_graphs(4)
    probes = []
    for graph, node in ((first, "30"), (second, "31")):
        entry = _wait(server, _post(server, "/prompt", {"prompt": graph})["prompt_id"], 600)
        got = entry.get("outputs", {}).get(node, {})
        text = next((v[0] if isinstance(v, list) and v else v for k, v in got.items() if k in ("text", "string", "result")), None)
        p = json.loads(text) if isinstance(text, str) else {"label": node, "missing": True, "raw": got}
        if p.get("missing"):
            p["verdict"] = "NO READING"
        elif node == "30":      # text only
            p["verdict"] = "ok" if p.get("text_ran_exact_gelu") is True else "WRONG"
        else:                   # the range left alone on purpose, the activation corrected, with a corrected clip loaded before the call
            low, high = p.get("first_layer") or (9, 9)
            p["verdict"] = "ok" if (abs(low) < 0.02 and abs(high - 1) < 0.02 and p.get("text_ran_exact_gelu") is True
                                    and p.get("text_module_showed_exact_before_the_call") is True) else "WRONG"
        probes.append(p)
        print("the node run again", json.dumps(p), flush=True)
    R["rounds"].append({"round": "the node run again while its first output is loaded", "status": entry.get("status", {}).get("status_str"), "probes": probes})
    R["passed"] = (R["h3_prompt"]["status"] == "success"
                   and all(p.get("verdict") == "ok" for r in R["rounds"] for p in r["probes"]))
    Path(a.json).write_text(json.dumps(R, indent=1))
    print("PASSED" if R["passed"] else "FAILED")
    return 0 if R["passed"] else 1


def cmd_render(a):
    D = json.loads(Path(a.json).read_text())
    h = D["h3_prompt"]
    print(f"The H3 prompt between the rounds: `{h['graph']}` at {h['steps']} steps and {h['length']} frames, status {h['status']}. Overall: {'PASSED' if D['passed'] else 'FAILED'}.\n")
    print("| round | probe | patcher class | SAM on the card before the call | first layer saw | text ran exact GELU | verdict |\n|---|---|---|---|---|---|---|")
    for r in D["rounds"]:
        for p in r["probes"]:
            print(f"| {r['round']} | {p.get('label')} | {p.get('patcher')} | {p.get('on_the_card_before_the_call')} | {p.get('first_layer')} | {p.get('text_ran_exact_gelu')} | {p.get('verdict')} |")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.set_defaults(fn=cmd_build)
    s.add_argument("--base", required=True)
    s = sub.add_parser("run")
    s.set_defaults(fn=cmd_run)
    s.add_argument("--server", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("render")
    s.set_defaults(fn=cmd_render)
    s.add_argument("--json", required=True)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
