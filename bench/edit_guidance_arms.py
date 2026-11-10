#!/usr/bin/env python3
"""Build, and with --queue render, the arms that test steering by an edited prompt with no control vector.

What it does: takes one shipped Pika graph as the base, a prompt file and one edit
(`old wording -> new wording`, an exact substring that occurs once), and builds these arms per seed:

    base        the prompt as written
    rewrite     the edited prompt
    guide_sX    both prompts each step; the prediction moved from the original toward the edited by X
                (core `CFGGuider`: positive = edited, negative = original, cfg = X)
    late_sX     the same, with the edit switched on only from --switch-step
    schedule    the original prompt until --switch-step, the edited prompt after (no extrapolation)

The base graph's control-vector node is removed, so the arms read edit guidance alone. The second prompt is a
copy of the base graph's own conditioning node with only the text changed, so both sides share the encoder,
the references and the canvas.

The switch is given in steps. Core's timestep range is a noise level, so the tool converts: it reads the
graph's sigmas and its video shift, and places the boundary between the last original step and the first
edited one (`comfy.model_sampling.time_snr_shift`, inverted).

What it cannot see: whether an arm looks right. It writes each arm's graph, prints both prompts in full,
and with --queue returns each render's `/history` entry status. A person watches the clips.

    <comfy-venv-python> bench/edit_guidance_arms.py --base workflows/h3_text_to_video_pika_control_vector_api.json \
        --prompt-file prompt_bank/t2va_noodle_bar.txt --edit "Live-action, cinematic, -> ..." \
        --name noodle --out data/results/edit_guidance
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONDITIONING_NODES = ("MiniMaxH3Conditioning", "MiniMaxH3ReferenceConditioning")


def _one(graph: dict, class_types: tuple[str, ...] | str) -> tuple[str, dict]:
    wanted = (class_types,) if isinstance(class_types, str) else class_types
    hits = [(k, v) for k, v in graph.items() if isinstance(v, dict) and v.get("class_type") in wanted]
    if len(hits) != 1:
        raise SystemExit(f"the base graph has {len(hits)} node(s) of {wanted}; this tool needs exactly one")
    return hits[0]


def _edit(text: str, spec: str) -> str:
    if "->" not in spec:
        raise SystemExit("--edit must read 'old wording -> new wording'")
    old, new = (part.strip() for part in spec.split("->", 1))
    if not old or not new:
        raise SystemExit("--edit needs wording on both sides of '->'")
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"--edit: the old wording occurs {n} time(s) in the prompt; it must occur exactly once")
    return text.replace(old, new, 1)


def _switch_percent(sigmas: list[float], shift: float, step: int) -> float:
    """The percent whose noise level lies between step-1 and step, on the graph's own schedule."""
    if not 1 <= step < len(sigmas) - 1:
        raise SystemExit(f"--switch-step must be from 1 to {len(sigmas) - 2} on this schedule")
    boundary = math.sqrt(sigmas[step - 1] * sigmas[step])
    t = boundary / (shift - (shift - 1.0) * boundary)  # time_snr_shift inverted
    return round(1.0 - t, 4)


def _base(path: Path, prompt: str, length: int, seed: int, prefix: str, ref_image: str | None) -> dict:
    g = json.loads(path.read_text(encoding="utf-8"))
    cv_id, cv = _one(g, "MiniMaxH3ApplyControlVectors")
    for node in g.values():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and val and val[0] == cv_id:
                node["inputs"][key] = cv["inputs"]["model"]
    del g[cv_id]
    _one(g, CONDITIONING_NODES)[1]["inputs"]["prompt"] = prompt
    _one(g, "MiniMaxH3Resolution")[1]["inputs"]["length"] = length
    _one(g, "RandomNoise")[1]["inputs"]["noise_seed"] = seed
    _one(g, "VHS_VideoCombine")[1]["inputs"]["filename_prefix"] = prefix
    if ref_image is not None:
        _one(g, "LoadImage")[1]["inputs"]["image"] = ref_image
    return g


def _with_second_prompt(g: dict, edited: str) -> tuple[str, str]:
    cond_id, cond = _one(g, CONDITIONING_NODES)
    second = copy.deepcopy(cond)
    second["inputs"]["prompt"] = edited
    g["205"] = second
    return cond_id, "205"


def _ranged(g: dict, at: str, source: str, start: float, end: float) -> None:
    g[at] = {"class_type": "ConditioningSetTimestepRange",
             "inputs": {"conditioning": [source, 0], "start": start, "end": end}}


def _arms(args, prompt: str, edited: str, seed: int) -> tuple[dict[str, dict], dict]:
    def fresh(label: str, text: str = prompt) -> dict:
        return _base(Path(args.base), text, args.length, seed, f"{args.prefix}/{args.name}_{label}_s{seed}",
                     args.ref_image)

    probe = fresh("probe")
    sigmas = [float(x) for x in _one(probe, "ManualSigmas")[1]["inputs"]["sigmas"].split(",")]
    shift = float(_one(probe, "MiniMaxH3SigmaShift")[1]["inputs"]["shift_video"])
    p = _switch_percent(sigmas, shift, args.switch_step)

    arms = {"base": fresh("base"), "rewrite": fresh("rewrite", edited)}

    def guided(label: str, scale: float, late: bool) -> dict:
        g = fresh(label)
        guider_id, guider = _one(g, "BasicGuider")
        first, second = _with_second_prompt(g, edited)
        positive = second
        if late:
            _ranged(g, "206", first, 0.0, p)
            _ranged(g, "207", second, p, 1.0)
            g["208"] = {"class_type": "ConditioningCombine",
                        "inputs": {"conditioning_1": ["206", 0], "conditioning_2": ["207", 0]}}
            positive = "208"
        g[guider_id] = {"class_type": "CFGGuider",
                        "inputs": {"model": guider["inputs"]["model"], "positive": [positive, 0],
                                   "negative": guider["inputs"]["conditioning"], "cfg": scale}}
        return g

    for scale in args.scales:
        arms[f"guide_s{scale:g}"] = guided(f"guide_s{scale:g}", scale, late=False)
    arms[f"late_s{args.late_scale:g}"] = guided(f"late_s{args.late_scale:g}", args.late_scale, late=True)

    g = fresh("schedule")
    guider_id, guider = _one(g, "BasicGuider")
    first, second = _with_second_prompt(g, edited)
    _ranged(g, "206", first, 0.0, p)
    _ranged(g, "207", second, p, 1.0)
    g["208"] = {"class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["206", 0], "conditioning_2": ["207", 0]}}
    guider["inputs"]["conditioning"] = ["208", 0]
    arms["schedule"] = g
    return arms, {"sigmas": sigmas, "shift_video": shift, "switch_step": args.switch_step, "switch_percent": p}


def _http(host: str, path: str, payload: dict | None = None):
    req = urllib.request.Request(f"http://{host}{path}",
                                 data=None if payload is None else json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _render(host: str, graph: dict, timeout_s: float) -> dict:
    prompt_id = _http(host, "/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        entry = _http(host, f"/history/{prompt_id}").get(prompt_id)
        if entry:
            return entry
        time.sleep(3)
    raise SystemExit(f"prompt {prompt_id} did not reach /history in {timeout_s:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="a shipped Pika control-vector graph (API format)")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--edit", required=True, help="'old wording -> new wording'")
    ap.add_argument("--name", required=True, help="short name of the shot, used in file and clip names")
    ap.add_argument("--out", required=True, help="folder for the arm graphs and the record, under data/")
    ap.add_argument("--ref-image", default=None, help="reference image name in the input folder (ref2va base)")
    ap.add_argument("--length", type=int, default=141)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.5])
    ap.add_argument("--late-scale", type=float, default=2.0)
    ap.add_argument("--switch-step", type=int, default=2, help="first step that uses the edited prompt")
    ap.add_argument("--prefix", default="Video/edit_guidance")
    ap.add_argument("--only", nargs="*", default=None, help="arm names to build; default all")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    edited = _edit(prompt, args.edit)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("=== original prompt, in full ===\n" + prompt)
    print("\n=== edited prompt, in full ===\n" + edited + "\n")

    record = {"base": args.base, "prompt_file": args.prompt_file, "edit": args.edit, "length": args.length,
              "name": args.name, "arms": []}
    for seed in args.seeds:
        arms, schedule = _arms(args, prompt, edited, seed)
        record["schedule"] = schedule
        for label, graph in arms.items():
            if args.only and label not in args.only:
                continue
            path = out / f"{args.name}_{label}_s{seed}.json"
            path.write_text(json.dumps(graph, indent=1), encoding="utf-8")
            row = {"arm": label, "seed": seed, "graph": path.name,
                   "clip_prefix": _one(graph, "VHS_VideoCombine")[1]["inputs"]["filename_prefix"]}
            if args.queue:
                t0 = time.time()
                entry = _render(args.host, graph, args.timeout)
                row["status"] = entry.get("status", {}).get("status_str")
                row["seconds"] = round(time.time() - t0, 1)
                print(f"{args.name} {label} seed {seed}: {row['status']} in {row['seconds']}s", flush=True)
            record["arms"].append(row)
            (out / f"{args.name}_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"switch at step {schedule['switch_step']} is percent {schedule['switch_percent']} on this schedule")
    print(f"wrote {len(record['arms'])} arm graph(s) and {args.name}_record.json under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
