#!/usr/bin/env python3
"""Build, and with --queue render, the arms that test steering by an edited prompt and/or control vectors.

What it does: takes one shipped Pika or P-H3-Image graph as the base, a prompt file, and optional `--edit`
clauses (`old wording -> new wording`, including `soundscape:`, `music:`, and `summary:` section prefixes via
`h3_prompt_edit.apply_edits`), and builds these arms per seed:

    base        the prompt as written (control-vector node removed)
    rewrite     the edited prompt (when --edit is given)
    guide_sX    both prompts each step; the prediction moved from the original toward the edited by X
                (core `CFGGuider`: positive = edited, negative = original, cfg = X)
    late_sX     the same, with the edit switched on only from --switch-step
    schedule    the original prompt until --switch-step, the edited prompt after (no extrapolation)

With --node-arms it also builds the same test through the pack's own `MiniMaxH3EditGuider`:

    node_sX              the guider node at scale X from step 0 (`last_step = 1000`); matches guide_sX
    node_sX_lastK        the guider node at scale X stopping after step K (via `--node-last-steps 1000 2`)
    node_late_sX         the guider node from --switch-step; one model pass before it, where late_sX makes two
    node_align_sX        the guider node at scale X with both prompts given one position origin
    node_align_sX_lastK  the aligned guider node stopping after step K

With `--vector-arm LABEL=STEM_OR_FILE:STRENGTH[:TARGET][+STEM2:STRENGTH2[:TARGET2]][,key=val,...]` it also builds arms
that keep the original prompt and configure one or more chained `MiniMaxH3Steer` nodes plus an optional
`MiniMaxH3ReleaseReferences` node (inserting them before `BasicGuider` even when the base graph is a plain Pika
baseline such as `h3_first_last_frame_pika_api.json`).
Supported trailing `key=val` options: `ref_vae_end=F`, `ref_vae_aug=F`, `start=F`, `end=F`, `mode=MODE`,
`allow_recipe_mismatch=true|false`, `task_lora_end=F` (sets `MiniMaxH3LoRABranch.end_percent` on node 72), and
`sol_end=F` (sets `MiniMaxH3Sol.end_percent`). Passing `STEM=none` with `STRENGTH=0` and `ref_vae_end=F` builds a
single-variable unsteered `MiniMaxH3ReleaseReferences` control; passing `none:0,sol_end=F` builds a same-topology
Sol-window control.

Supports `t2va`, `ref2va`, `fl2va` (`--ref-image` + `--ref-image-2`), `P-H3-Image` `t2i`/`edit`, and Pika guide-latent
single-frame image edit (`MiniMaxH3ReferenceToVideo` + `MiniMaxH3GuideLatent`, propagating the guide onto the edited
branch and supporting `--canvas WIDTHxHEIGHT` and `PreviewImage` -> `SaveImage`).

What it cannot see: whether an arm looks right. It writes each arm's graph, prints prompts in full,
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
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h3_prompt_edit import apply_edits  # noqa: E402
from h3_spans import PROMPT_FIELDS  # noqa: E402

CONDITIONING_NODES = (
    "MiniMaxH3Conditioning",
    "MiniMaxH3ReferenceConditioning",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3PH3ImageConditioning",
)
GUIDE_NODES = (
    "MiniMaxH3GuideLatent",
    "MiniMaxH3AddGuide",
)
VECTOR_OPT_KEYS = frozenset(
    {"ref_vae_end", "ref_vae_aug", "start", "end", "mode", "sol_end", "task_lora_end", "allow_recipe_mismatch"}
)


def _all_of(graph: dict, class_types: tuple[str, ...] | str) -> list[tuple[str, dict]]:
    wanted = (class_types,) if isinstance(class_types, str) else class_types
    hits = [(k, v) for k, v in graph.items() if isinstance(v, dict) and v.get("class_type") in wanted]
    return sorted(hits, key=lambda kv: (0, int(kv[0])) if kv[0].isdigit() else (1, kv[0]))


def _one(graph: dict, class_types: tuple[str, ...] | str) -> tuple[str, dict]:
    hits = _all_of(graph, class_types)
    if len(hits) != 1:
        raise SystemExit(f"the base graph has {len(hits)} node(s) of {class_types}; this tool needs exactly one")
    return hits[0]


def _edit(text: str, specs: list[str] | None) -> str | None:
    if not specs:
        return None
    joined = "\n".join(specs)
    try:
        edited, _records = apply_edits(text, joined)
    except ValueError as exc:
        raise SystemExit(f"--edit: {exc}") from exc
    return edited


def _switch_percent(sigmas: list[float], shift: float, step: int) -> float:
    """The percent whose noise level lies between step-1 and step, on the graph's own schedule."""
    if not 1 <= step < len(sigmas) - 1:
        raise SystemExit(f"--switch-step must be from 1 to {len(sigmas) - 2} on this schedule")
    boundary = math.sqrt(sigmas[step - 1] * sigmas[step])
    t = boundary / (shift - (shift - 1.0) * boundary)  # time_snr_shift inverted
    return round(1.0 - t, 4)


def _parse_vector_arm(spec: str) -> tuple[str, list[tuple[str, float, str | None]], dict[str, str]]:
    label, sep, rest = spec.partition("=")
    if not label or not sep or not rest:
        raise SystemExit("--vector-arm must read 'label=file:strength[:mask][+file2:strength2[:mask2]][,key=val,...]'")
    chunks = [c.strip() for c in rest.split(",") if c.strip()]
    if not chunks:
        raise SystemExit(f"--vector-arm {label!r}: missing vector specification")
    chain_raw, opt_chunks = chunks[0], chunks[1:]
    chain: list[tuple[str, float, str | None]] = []
    for item in chain_raw.split("+"):
        parts = item.strip().split(":")
        if len(parts) not in (2, 3) or not parts[0]:
            raise SystemExit(
                f"--vector-arm {label!r}: each '+'"
                "-separated item must read 'file.safetensors:strength[:mask]' or 'none:0'"
            )
        chain.append((parts[0], float(parts[1]), parts[2] if len(parts) == 3 and parts[2] else None))
    opts: dict[str, str] = {}
    for kv in opt_chunks:
        k, eq, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if not eq or k not in VECTOR_OPT_KEYS or not v:
            raise SystemExit(
                f"--vector-arm {label!r}: invalid option {kv!r}; expected key=val with key in {sorted(VECTOR_OPT_KEYS)}"
            )
        opts[k] = v
    return label, chain, opts


def _rewire_consumers(g: dict, old_id: str, new_ref: list, *, skip_ids: set[str] | None = None) -> None:
    skip = skip_ids or set()
    for nid, node in g.items():
        if nid in skip:
            continue
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and val[0] == old_id and val[1] == 0:
                node["inputs"][key] = list(new_ref)


def _set_output_prefix(g: dict, prefix: str) -> None:
    vhs = _all_of(g, "VHS_VideoCombine")
    if vhs:
        vhs[0][1]["inputs"]["filename_prefix"] = prefix
        return
    save_img = _all_of(g, "SaveImage")
    if save_img:
        save_img[0][1]["inputs"]["filename_prefix"] = prefix
        return
    prev_img = _all_of(g, "PreviewImage")
    if prev_img:
        pid, pnode = prev_img[0]
        g[pid] = {
            "class_type": "SaveImage",
            "inputs": {"images": pnode["inputs"]["images"], "filename_prefix": prefix},
        }
        return
    raise SystemExit("the base graph has no VHS_VideoCombine, SaveImage, or PreviewImage output node")


def _get_output_prefix(g: dict) -> str:
    for ctype in ("VHS_VideoCombine", "SaveImage"):
        hits = _all_of(g, ctype)
        if hits:
            return str(hits[0][1]["inputs"]["filename_prefix"])
    raise SystemExit("arm graph has no VHS_VideoCombine or SaveImage output node")


def _parse_canvas(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    w_s, sep, h_s = spec.lower().partition("x")
    if not sep or not w_s.isdigit() or not h_s.isdigit():
        raise SystemExit(f"--canvas must be WIDTHxHEIGHT (e.g. 768x1024), got {spec!r}")
    return int(w_s), int(h_s)


def _resolution_inputs(width: int, height: int, length: int) -> dict:
    wf_dir = str(REPO_ROOT / "workflows")
    comfy_root = str(REPO_ROOT.parent.parent)
    if wf_dir not in sys.path:
        sys.path.insert(0, wf_dir)
    from build_workflows import _resolution_widgets  # noqa: E402

    # Ensure ComfyUI's root precedes REPO_ROOT so `import nodes` inside
    # `comfy_extras.nodes_minimax_h3` resolves to core's `nodes.py`, not this pack's.
    sys.path[:] = [comfy_root] + [p for p in sys.path if p != comfy_root]
    return _resolution_widgets(width, height, length)


def _base(
    path: Path,
    prompt: str,
    length: int,
    seed: int,
    prefix: str,
    ref_image: str | None,
    ref_image_2: str | None = None,
    canvas_wh: tuple[int, int] | None = None,
    vector_chain: list[tuple[str, float, str | None]] | None = None,
    vector_opts: dict[str, str] | None = None,
) -> dict:
    g = json.loads(path.read_text(encoding="utf-8"))
    opts = vector_opts or {}
    if "sol_end" in opts:
        _one(g, "MiniMaxH3Sol")[1]["inputs"]["end_percent"] = float(opts["sol_end"])
    if "task_lora_end" in opts:
        if "72" not in g or g["72"].get("class_type") != "MiniMaxH3LoRABranch":
            raise SystemExit("task_lora_end requires a task LoRA MiniMaxH3LoRABranch at node 72")
        g["72"]["inputs"]["end_percent"] = float(opts["task_lora_end"])

    for old_id, old_node in _all_of(
        g, ("MiniMaxH3Steer", "MiniMaxH3ReleaseReferences", "MiniMaxH3ApplyControlVectors")
    ):
        _rewire_consumers(g, old_id, old_node["inputs"]["model"])
        del g[old_id]
    for helper_id, _ in _all_of(g, ("MiniMaxH3WordVector", "MiniMaxH3LoadControlVector")):
        del g[helper_id]

    def _stem(raw: str) -> str:
        s = re.sub(r"\.safetensors$", "", raw.strip())
        if s in ("", "none"):
            return "none"
        for marker in ("__fl2va", "__ref2va"):
            idx = s.find(marker)
            if idx > 0:
                return s[:idx]
        return s

    has_steer = bool(vector_chain) and not (
        len(vector_chain) == 1
        and _stem(vector_chain[0][0]) == "none"
        and vector_chain[0][1] == 0.0
    )
    has_release = "ref_vae_end" in opts or "ref_vae_aug" in opts

    if has_steer or has_release:
        _guider_id, guider = _one(g, "BasicGuider")
        tail_ref = list(guider["inputs"]["model"])

        if has_steer:
            assert vector_chain is not None
            for step_i, (v_name, v_strength, v_mask) in enumerate(vector_chain):
                node_id = "115" if step_i == 0 else str(214 + step_i)
                steer_inputs: dict = {
                    "model": tail_ref,
                    "vector": _stem(v_name),
                    "strength": v_strength,
                    "mode": opts.get("mode", "add_norm_matched"),
                    "target": "scene_only",
                    "start_percent": float(opts.get("start", 0.0)),
                    "end_percent": float(opts.get("end", 1.0)),
                    "streams": "both",
                    "hold_words": "",
                    "hold_alpha": 1.0,
                    "allow_recipe_mismatch": opts.get("allow_recipe_mismatch", "false").lower()
                    in ("1", "true", "yes"),
                }
                if v_mask:
                    if v_mask in PROMPT_FIELDS:
                        steer_inputs["fields"] = {"__value__": [v_mask]}
                    else:
                        steer_inputs["target"] = "description_only" if v_mask == "description" else v_mask
                g[node_id] = {"class_type": "MiniMaxH3Steer", "inputs": steer_inputs}
                tail_ref = [node_id, 0]

        if has_release:
            g["135"] = {
                "class_type": "MiniMaxH3ReleaseReferences",
                "inputs": {
                    "model": tail_ref,
                    "end_percent": float(opts.get("ref_vae_end", 1.0)),
                    "clean_level": float(opts.get("ref_vae_aug", 0.999)),
                },
            }
            tail_ref = ["135", 0]

        guider["inputs"]["model"] = tail_ref

    _cond_id, cond = _one(g, CONDITIONING_NODES)
    cond["inputs"]["prompt"] = prompt
    res_hits = _all_of(g, "MiniMaxH3Resolution")
    if res_hits:
        if canvas_wh is not None:
            res_hits[0][1]["inputs"] = _resolution_inputs(canvas_wh[0], canvas_wh[1], length)
        else:
            res_hits[0][1]["inputs"]["length"] = length
    else:
        if isinstance(cond["inputs"].get("length"), int):
            cond["inputs"]["length"] = length
        if canvas_wh is not None:
            cond["inputs"]["width"] = canvas_wh[0]
            cond["inputs"]["height"] = canvas_wh[1]
            if "canvas" in cond["inputs"]:
                cond["inputs"]["canvas"] = "explicit"

    _one(g, "RandomNoise")[1]["inputs"]["noise_seed"] = seed
    _set_output_prefix(g, prefix)

    loads = _all_of(g, "LoadImage")
    if ref_image is not None:
        if len(loads) < 1:
            raise SystemExit("--ref-image was given, but the base graph has no LoadImage node")
        loads[0][1]["inputs"]["image"] = ref_image
    if ref_image_2 is not None:
        if len(loads) < 2:
            raise SystemExit("--ref-image-2 was given, but the base graph has fewer than 2 LoadImage nodes")
        loads[1][1]["inputs"]["image"] = ref_image_2
    return g


def _with_second_prompt(g: dict, edited: str) -> tuple[str, str]:
    cond_id, cond = _one(g, CONDITIONING_NODES)
    second = copy.deepcopy(cond)
    second["inputs"]["prompt"] = edited
    g["205"] = second
    for guide_id, guide_node in _all_of(g, GUIDE_NODES):
        if guide_node["inputs"].get("positive") == [cond_id, 0]:
            second_guide = copy.deepcopy(guide_node)
            second_guide["inputs"]["positive"] = ["205", 0]
            if second_guide["inputs"].get("latent") == [cond_id, 1]:
                second_guide["inputs"]["latent"] = ["205", 1]
            g["270"] = second_guide
            return guide_id, "270"
    return cond_id, "205"


def _ranged(g: dict, at: str, source: str, start: float, end: float) -> None:
    g[at] = {"class_type": "ConditioningSetTimestepRange",
             "inputs": {"conditioning": [source, 0], "start": start, "end": end}}


def _effective_length(g: dict) -> int:
    if any(n["inputs"].get("single_frame") for _, n in _all_of(g, GUIDE_NODES)):
        return 1
    res_hits = _all_of(g, "MiniMaxH3Resolution")
    if res_hits:
        return int(res_hits[0][1]["inputs"]["length"])
    _cid, cond = _one(g, CONDITIONING_NODES)
    if isinstance(cond["inputs"].get("length"), int):
        return int(cond["inputs"]["length"])
    return 1


def _arms(args, prompt: str, edited: str | None, seed: int) -> tuple[dict[str, dict], dict]:
    canvas_wh = _parse_canvas(args.canvas)

    def fresh(label: str, text: str = prompt) -> dict:
        return _base(
            Path(args.base),
            text,
            args.length,
            seed,
            f"{args.prefix}/{args.name}_{label}_s{seed}",
            args.ref_image,
            args.ref_image_2,
            canvas_wh=canvas_wh,
        )

    probe = fresh("probe")
    sigmas = [float(x) for x in _one(probe, "ManualSigmas")[1]["inputs"]["sigmas"].split(",")]
    shift = float(_one(probe, "MiniMaxH3SigmaShift")[1]["inputs"]["shift_video"])
    p = _switch_percent(sigmas, shift, args.switch_step)

    arms: dict[str, dict] = {"base": fresh("base")}

    if edited is not None:
        arms["rewrite"] = fresh("rewrite", edited)

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

        if args.node_arms:
            def node(label: str, scale: float, first_step: int, align: bool, last_step: int = 1000) -> dict:
                g = fresh(label)
                guider_id, guider = _one(g, "BasicGuider")
                _first, second = _with_second_prompt(g, edited)
                g[guider_id] = {"class_type": "MiniMaxH3EditGuider",
                                "inputs": {"model": guider["inputs"]["model"],
                                           "original": guider["inputs"]["conditioning"], "edited": [second, 0],
                                           "scale": scale, "first_step": first_step, "last_step": last_step,
                                           "align_positions": align}}
                return g

            low = args.scales[0]
            last_steps = args.node_last_steps or [1000]
            for ls in last_steps:
                suffix = "" if ls == 1000 else f"_last{ls}"
                arms[f"node_s{low:g}{suffix}"] = node(f"node_s{low:g}{suffix}", low, 0, False, last_step=ls)
                arms[f"node_align_s{low:g}{suffix}"] = node(f"node_align_s{low:g}{suffix}", low, 0, True, last_step=ls)
            arms[f"node_late_s{args.late_scale:g}"] = node(
                f"node_late_s{args.late_scale:g}", args.late_scale, args.switch_step, False, last_step=1000
            )

    for spec in args.vector_arm or []:
        label, chain, opts = _parse_vector_arm(spec)
        arms[label] = _base(
            Path(args.base),
            prompt,
            args.length,
            seed,
            f"{args.prefix}/{args.name}_{label}_s{seed}",
            args.ref_image,
            args.ref_image_2,
            canvas_wh=canvas_wh,
            vector_chain=chain,
            vector_opts=opts,
        )
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
    ap.add_argument("--base", required=True, help="a shipped Pika or P-H3-Image graph (API format)")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--edit", action="append", default=None,
                    help="'old wording -> new wording' (repeatable; supports soundscape:, music:, summary: prefixes)")
    ap.add_argument("--name", required=True, help="short name of the shot, used in file and clip names")
    ap.add_argument("--out", required=True, help="folder for the arm graphs and the record, under data/")
    ap.add_argument("--ref-image", default=None, help="first LoadImage input (ref2va ref, fl2va first_frame, or edit image_1)")
    ap.add_argument("--ref-image-2", default=None, help="second LoadImage input (fl2va last_frame or edit image_2)")
    ap.add_argument("--canvas", default=None, help="optional explicit WIDTHxHEIGHT (e.g. 768x1024)")
    ap.add_argument("--length", type=int, default=141)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.5])
    ap.add_argument("--late-scale", type=float, default=2.0)
    ap.add_argument("--switch-step", type=int, default=2, help="first step that uses the edited prompt")
    ap.add_argument("--prefix", default="Video/edit_guidance")
    ap.add_argument("--only", nargs="*", default=None, help="arm names to build; default all")
    ap.add_argument("--node-arms", action="store_true",
                    help="also build the arms that use the pack's MiniMaxH3EditGuider node")
    ap.add_argument("--node-last-steps", type=int, nargs="+", default=[1000],
                    help="last_step value(s) for MiniMaxH3EditGuider node arms (e.g. --node-last-steps 1000 2)")
    ap.add_argument("--vector-arm", action="append", default=None,
                    metavar="LABEL=FILE:STRENGTH[:MASK][+FILE2:...][,key=val,...]",
                    help="build an arm keeping the original prompt with control vector(s) and optional "
                         "ref_vae_end/start/end/sol_end settings; repeatable")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    edited = _edit(prompt, args.edit)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("=== original prompt, in full ===\n" + prompt)
    if edited is not None:
        print("\n=== edited prompt, in full ===\n" + edited + "\n")
    else:
        print("\n=== edited prompt: (none; vector/base arms only) ===\n")

    edit_record: str | list[str] | None = (
        args.edit[0] if args.edit and len(args.edit) == 1 else args.edit
    )
    probe_base = _base(
        Path(args.base),
        prompt,
        args.length,
        args.seeds[0],
        args.prefix,
        args.ref_image,
        args.ref_image_2,
        canvas_wh=_parse_canvas(args.canvas),
    )
    record = {
        "base": args.base,
        "prompt_file": args.prompt_file,
        "edit": edit_record,
        "length": _effective_length(probe_base),
        "name": args.name,
        "arms": [],
    }
    for seed in args.seeds:
        arms, schedule = _arms(args, prompt, edited, seed)
        record["schedule"] = schedule
        for label, graph in arms.items():
            if args.only and label not in args.only:
                continue
            path = out / f"{args.name}_{label}_s{seed}.json"
            path.write_text(json.dumps(graph, indent=1), encoding="utf-8")
            row = {"arm": label, "seed": seed, "graph": path.name,
                   "clip_prefix": _get_output_prefix(graph)}
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
