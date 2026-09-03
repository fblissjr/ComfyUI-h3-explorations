#!/usr/bin/env python3
"""Generate the example graphs that ship with the sidecar weights on the Hub.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Needs no server and
no GPU; writes JSON.

**These are not this repo's graphs and must not go in `workflows/`.** Two
reasons, and the first is enforced:

  * `bench/check_attention_defaults.py` requires Sol-Attn live in every shipped
    video graph, and these deliberately wire NO attention node at all. Landing
    them in `workflows/` would either go red or need an exemption for a graph
    that is not one of this repo's arms.
  * They are for a stranger who has ComfyUI, the weights, and the node pack --
    nothing else. Every shipped graph here wires seven nodes that person does
    not have: this pack's conditioning, preflight, resolution and sage nodes,
    plus `SolAttnMiniMax` and `VHS_VideoCombine` from other packs. A graph that
    fails to load teaches nothing.

So the constraint is **core plus exactly one node**, `MiniMaxH3PDDLoRA`. Core
turns out to carry everything else: `MiniMaxH3ImageToVideo` for conditioning
and the empty latent, and `CreateVideo` -> `SaveVideo` for a muxed file, so not
even VideoHelperSuite is required.

There is deliberately **no `MiniMaxH3SigmaShift`**. Core already carries both
shifts -- `comfy/supported_models.py`'s `MiniMaxH3.sampling_settings` is
`shift: 12.0, audio_shift: 3.0` -- so a node setting them to exactly those
values is a no-op, and an example that wires one teaches the reader it is
load-bearing. (Verified at that source, not quoted: the same node was found
inert in a render comparison on 2026-08-29.)

Generated rather than hand-written for the reason `CLAUDE.md` gives about
`workflows/*.json`: a JSON graph typed by hand drifts from the node schema it
targets and nothing says so. This at least drifts in one place.

    python bench/build_sidecar_examples.py --out <the staged Hub repo>/workflows
    python bench/build_sidecar_examples.py --out <same> --check

## Validated against the live schemas before anything is written

The widget names, combo values and link types below are typed by hand, and
until 2026-09-03 nothing compared them with the nodes they target; the owner
asked whether they were still aligned with the PDD node and the honest answer
was "nobody has looked". So `main` now hands every graph to a subprocess that
loads core ComfyUI (the checkout this pack lives under), its bundled extras,
and the STAGED node bundle beside `--out` -- the node as shipped, not the one
in this tree -- and checks each node's inputs against the class actually
loaded: no unknown input, no required input missing, every literal combo
value among that combo's options (file combos excepted, since those depend on
the reader's folders), and every link's source output type equal to the
input's type. Red means nothing is written. Needs the checkout and the staged
bundle (exit 2 without either); no GPU, no server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "workflows"))

#: Pulled from the generator rather than retyped, so the example scene and the
#: repo's own dialogue arm cannot say different things.
from build_workflows import DIALOGUE_T2V_PROMPT  # noqa: E402

#: The example renders at a trained canvas and a legal frame count. 1344x768 is
#: the shape `docs/h3_resolutions.md` calls trained, and 362 frames is the
#: repo's own dialogue arm -- the prompt's third shot starts at 00:11, so a
#: shorter clip would cut the scene the prompt describes.
WIDTH, HEIGHT, LENGTH, FPS = 1344, 768, 362, 24.0

#: Names as published on the Hub. A stranger's files will be named this only if
#: they downloaded them from there, which is the case this exists for.
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_V = "minimax_h3_video_vae_fp16.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_fl2va_pdd_8step_comfy.safetensors"

SEED = 730451892

#: The ComfyUI checkout this pack lives under, for the validation subprocess.
COMFY = HERE.parent.parent.parent

#: The staged node bundle, beside the workflows folder on the Hub. Overridable
#: with `--node`, but the default is the point: validate against what ships.
BUNDLE_NAME = "comfyui_minimax_h3_pdd"

#: Combo inputs whose options come from the reader's own model folders. A
#: literal there is checked for being a string, not for membership.
FILE_COMBOS = {("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
               ("VAELoader", "vae_name"), ("MiniMaxH3PDDLoRA", "lora_name")}

#: Runs inside the subprocess. `sys.argv`: comfy root, bundle dir, a JSON file
#: of `{graph name: graph}`, a JSON list of file-combo pairs. Prints one JSON
#: object `{graph name: [problem, ...]}`.
VALIDATE_SCRIPT = r"""
import asyncio, importlib.util, json, os, sys
comfy, bundle, graphs_path, file_combos = sys.argv[1:5]
os.chdir(comfy); sys.path.insert(0, comfy)
import nodes
from comfy_api.latest import io
asyncio.run(nodes.init_builtin_extra_nodes())
registry = dict(nodes.NODE_CLASS_MAPPINGS)
name = bundle.rstrip("/").rsplit("/", 1)[-1]
spec = importlib.util.spec_from_file_location(name, bundle + "/__init__.py",
                                              submodule_search_locations=[bundle])
mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
spec.loader.exec_module(mod)
for cls in asyncio.run(asyncio.run(mod.comfy_entrypoint()).get_node_list()):
    registry[cls.define_schema().node_id] = cls
file_combos = {tuple(x) for x in json.loads(file_combos)}

def describe(cls):
    if hasattr(cls, "define_schema"):
        sch = cls.define_schema()
        ins = {}
        for i in sch.inputs:
            # `io.Combo.Input` is its own class, not a subclass of `io.Combo`,
            # so the combo is recognised by its io_type; the first revision
            # used isinstance and checked no combo on any v3 node.
            opts = getattr(i, "options", None) if i.io_type == "COMBO" else None
            ins[i.id] = {"required": not getattr(i, "optional", False),
                         "type": i.io_type,
                         "options": list(opts) if opts is not None else None}
        outs = [o.io_type for o in sch.outputs]
        return ins, outs
    it = cls.INPUT_TYPES(); ins = {}
    for group, req in (("required", True), ("optional", False)):
        for k, v in it.get(group, {}).items():
            t = v[0] if isinstance(v, (tuple, list)) else v
            if isinstance(t, list):
                ins[k] = {"required": req, "type": "COMBO", "options": list(t)}
            else:
                ins[k] = {"required": req, "type": t, "options": None}
    return ins, list(getattr(cls, "RETURN_TYPES", ()))

graphs = json.loads(open(graphs_path).read())
report = {}
for gname, g in graphs.items():
    problems = []
    described = {}
    for nid, n in g.items():
        cls = registry.get(n["class_type"])
        if cls is None:
            problems.append(f"{nid} {n['class_type']}: no such node in core or the bundle")
            continue
        described[nid] = describe(cls)
    for nid, n in g.items():
        if nid not in described:
            continue
        ins, _ = described[nid]; ct = n["class_type"]
        for k, v in n["inputs"].items():
            if k not in ins:
                problems.append(f"{nid} {ct}.{k}: not an input of the loaded node"
                                f" (has {sorted(ins)})")
                continue
            spec = ins[k]
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                src, idx = v
                if src not in described:
                    problems.append(f"{nid} {ct}.{k}: linked to node {src}, which is absent or unknown")
                    continue
                outs = described[src][1]
                if not isinstance(idx, int) or idx >= len(outs):
                    problems.append(f"{nid} {ct}.{k}: linked to output {idx} of {src} "
                                    f"{g[src]['class_type']}, which has {len(outs)} outputs")
                    continue
                if spec["type"] not in ("*", "COMBO") and outs[idx] != "*" and outs[idx] != spec["type"]:
                    problems.append(f"{nid} {ct}.{k} wants {spec['type']} but {src} "
                                    f"{g[src]['class_type']} output {idx} is {outs[idx]}")
            elif spec["options"] is not None:
                if (ct, k) in file_combos:
                    if not isinstance(v, str) or not v:
                        problems.append(f"{nid} {ct}.{k}: file combo needs a filename, got {v!r}")
                elif v not in spec["options"]:
                    problems.append(f"{nid} {ct}.{k}: {v!r} is not among the options "
                                    f"{spec['options'][:8]}{'...' if len(spec['options']) > 8 else ''}")
        for k, spec in ins.items():
            if spec["required"] and k not in n["inputs"]:
                problems.append(f"{nid} {ct}.{k}: required input missing")
    report[gname] = problems
print("VALIDATION " + json.dumps(report))
"""


def validate(graphs: dict, bundle: Path) -> "dict[str, list[str]] | None":
    """Every graph against the live schemas; `None` when it cannot run."""
    if not (COMFY / "nodes.py").exists() or not (COMFY / "comfy_api").is_dir():
        print(f"  SKIP  no ComfyUI checkout at {COMFY}; graphs were not validated")
        return None
    if not (bundle / "__init__.py").exists():
        print(f"  SKIP  no node bundle at {bundle}; build it first "
              f"(bench/build_sidecar_node.py), graphs were not validated")
        return None
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(graphs, f); tmp = f.name
    try:
        proc = subprocess.run([sys.executable, "-c", VALIDATE_SCRIPT, str(COMFY), str(bundle),
                               tmp, json.dumps(sorted(FILE_COMBOS))],
                              capture_output=True, text=True, cwd=str(COMFY))
    finally:
        Path(tmp).unlink(missing_ok=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("VALIDATION ")]
    if proc.returncode != 0 or not lines:
        print("  FAIL  the validator did not run to completion:")
        for line in proc.stderr.strip().splitlines()[-4:]:
            print(f"        {line}")
        return {"<validator>": ["did not run"]}
    return json.loads(lines[-1][len("VALIDATION "):])


def graph(steps: int, sampler: str, head_strength: float = 1.0) -> dict:
    """One example, as a ComfyUI API-format graph.

    `steps` reaches the PDD node and nothing else -- there is no
    `BasicScheduler` here, because the node emits the schedule its own heads
    were fused for and the sampler consumes that. That is the whole point of
    the SIGMAS output and it is why these examples have one fewer node than a
    normal graph rather than one more.
    """
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_V}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_A}},

        # Core's conditioning node: prompt in, positive conditioning and a
        # correctly-shaped empty AV latent out. No reference image, so this is
        # the t2va task on the fl2va partition.
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                         "prompt": DIALOGUE_T2V_PROMPT,
                         "width": WIDTH, "height": HEIGHT, "length": LENGTH}},

        "7": {"class_type": "MiniMaxH3PDDLoRA",
              "inputs": {"model": ["1", 0], "lora_name": LORA,
                         "strength": 1.0, "head_strength": head_strength,
                         "patch_heads": True, "nfe": 0, "steps": steps}},

        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "10": {"class_type": "BasicGuider",
               "inputs": {"model": ["7", 0], "conditioning": ["5", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["10", 0],
                          "sampler": ["9", 0],
                          # the schedule the heads were fused for
                          "sigmas": ["7", 1],
                          "latent_image": ["5", 1]}},

        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "14": {"class_type": "CreateVideo",
               "inputs": {"images": ["12", 0], "fps": FPS, "audio": ["13", 0]}},
        "15": {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": "h3_pdd_dialogue",
                          "format": "auto"}},
    }


EXAMPLES = {
    "t2va_pdd_5step.json": dict(
        steps=5, sampler="euler",
        note="The cheapest count worth running, and the one to reach for "
             "first. Tiles as [8,8,8,4,4] -- the wide blocks are front-loaded, "
             "so the final Euler step still spans 63.2% of the sigma range, "
             "the same as 8 steps. One evaluation more than 4 buys the whole "
             "of that; a sixth buys nothing further."),
    "t2va_pdd_8step.json": dict(
        steps=8, sampler="euler",
        note="The count the LoRA was distilled at, and the reference arm. "
             "Uniform [4,4,4,4,4,4,4,4]. `euler` because every reference "
             "implementation of H3 integrates with deterministic Euler at "
             "eta=0, which also makes this the only arm here that reproduces "
             "on a repeated seed."),
    "t2va_pdd_4step.json": dict(
        steps=4, sampler="euler",
        note="The fast arm, and the one with a known cost. [8,8,8,8] is the "
             "ONLY partition of the 32-point grid into four blocks that is "
             "legal under the trained envelope, so its final Euler step spans "
             "80% of the sigma range rather than 63.2% and that is forced "
             "rather than chosen. Expect coarser motion and rougher audio. "
             "Shipped because it is the first thing anyone tries: better to "
             "know why it looks like that than to conclude the weights are "
             "broken. Use 5 steps instead unless the time matters."),
    "t2va_pdd_8step_heads_off.json": dict(
        steps=8, sampler="euler", head_strength=0.0,
        note="The control arm. `head_strength=0.0` installs no head patches at "
             "all, so the backbone and modulation updates apply against the "
             "checkpoint's own output heads. Use it to see what the "
             "per-interval heads are actually buying."),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate the example graphs that ship with the sidecar weights.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--node", type=Path, default=None,
                    help=f"the staged node bundle to validate against "
                         f"(default: <out>/../{BUNDLE_NAME})")
    ap.add_argument("--check", action="store_true",
                    help="validate and compare with the files in --out instead of writing")
    args = ap.parse_args(argv)
    bundle = args.node or (args.out.parent / BUNDLE_NAME)

    graphs, notes = {}, {}
    for name, spec in EXAMPLES.items():
        spec = dict(spec)
        notes[name] = spec.pop("note")
        graphs[name] = graph(**spec)

    report = validate(graphs, bundle)
    if report is None:
        return 2
    bad = {k: v for k, v in report.items() if v}
    for name, problems in bad.items():
        for pr in problems:
            print(f"  RED   {name}: {pr}")
    if bad:
        print(f"\n{sum(len(v) for v in bad.values())} problem(s) against the loaded "
              f"schemas; nothing written.")
        return 1
    print(f"  ok    {len(graphs)} graph(s) validated against core and {bundle.name}")

    if args.check:
        drift = [name for name, g in graphs.items()
                 if not (args.out / name).exists()
                 or (args.out / name).read_text(encoding="utf-8") != json.dumps(g, indent=2) + "\n"]
        for name in drift:
            print(f"  DRIFT {name}: differs from the generator, or missing")
        if drift:
            print(f"\n{len(drift)} difference(s). Rerun without --check to rebuild, then push.")
            return 1
        print(f"  ok    {len(graphs)} graph(s) match {args.out}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, g in graphs.items():
        (args.out / name).write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
        used = sorted({v["class_type"] for v in g.values()})
        noncore = [c for c in used if c.startswith("MiniMaxH3PDD")]
        print(f"  {name}")
        print(f"     {notes[name]}")
        print(f"     {len(g)} nodes, {len(used)} classes; from the pack: {noncore or 'none'}")
    print(f"\nwrote {len(graphs)} example(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
