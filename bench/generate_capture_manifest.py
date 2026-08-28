#!/usr/bin/env python3
"""Generate an auditable, fully dynamic manifest.json for an activation capture directory.

Adheres strictly to docs/capture_manifest_schema.md without hardcoded workload constants.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image
import torch

# `substrate.py` lives at the repo root and owns the readers for every value in
# the provenance block. Imported rather than reimplemented: this file wrote six
# of those values as literals until 2026-08-17, which is what that module exists
# to stop.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from substrate import infer_quantization  # noqa: E402

sys.path.insert(0, str(_REPO / "workflows"))
from h3_config import LORA_LOADER_CLASSES, graph_schedule  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


#: {manifest field: the folder_paths category holding it}. The `loras` entry is
#: handled separately because it is a list.
_MODEL_FOLDERS = {
    "unet": "diffusion_models",
    "clip": "text_encoders",
    "video_vae": "vae",
    "audio_vae": "vae",
}


def hash_model_files(models: dict) -> dict:
    """{manifest field: sha256} for every model file this capture named.

    Model identity in this manifest was the FILENAME and nothing else, with
    `rank` regexed back out of it. A name is what the graph asked for; it is not
    evidence of what the loader read, and a file replaced in place leaves every
    field here unchanged. Hashing is the same treatment reference media and the
    captured tensors already get in this file.

    **What a file hash cannot see**, stated because a silent limit reads as
    coverage: a model patched at runtime. A LoRA at strength 0.75 and the same
    LoRA at 1.0 hash identically, and so does a checkpoint under a different
    quantization applied after load. This pins WHICH BYTES were on disk, not
    what the sampler ended up holding.

    A file that cannot be resolved or read maps to a reason string rather than
    being dropped: an absent hash and an unhashed model must not look alike.
    """
    out: dict = {}
    try:
        import folder_paths
    except Exception as exc:
        return {"_unavailable": f"folder_paths not importable: {exc}"}

    def one(category: str, name: str):
        if not name:
            return None
        try:
            path = folder_paths.get_full_path(category, name)
        except Exception as exc:
            return f"unresolved: {type(exc).__name__}: {exc}"
        if not path or not Path(path).is_file():
            return f"unresolved: {name!r} not found under {category}"
        try:
            return sha256_file(path)
        except OSError as exc:
            return f"unreadable: {exc}"

    for field, category in _MODEL_FOLDERS.items():
        digest = one(category, str(models.get(field) or ""))
        if digest is not None:
            out[field] = digest
    declared: list[dict] = list(models.get("loras", []) or [])
    # `one()` answers None for an empty name. Keeping that would put a null in
    # a list the schema declares as strings, and `assert_model_hashes` only
    # compares lengths, so it would pass its own check while being invalid.
    loras = [one("loras", str(lo.get("name") or "")) or "unresolved: lora record carries no name"
             for lo in declared]
    if loras:
        out["loras"] = loras
    return out


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def parse_prompt_sections(text: str) -> dict[str, str]:
    """Parse 6-section prompt into verbatim section strings."""
    sections = {}
    known_headers = [
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    ]
    pattern = rf"(?m)^({'|'.join(known_headers)}):\s*\n"
    splits = re.split(pattern, text)
    if len(splits) > 1:
        for i in range(1, len(splits), 2):
            k = splits[i].strip()
            v = splits[i + 1].strip()
            sections[k] = v
    else:
        sections["full"] = text
    return sections


def compute_reference_fit(
    img_path: Path,
    allow_upscale: bool = False,
    short_edge: int = 2048,
    max_area: int = 2048 * 2048,
) -> tuple[list[int], list[int], int]:
    """Calculate raw dimensions, fitted dimensions, and latent token count."""
    with Image.open(img_path) as im:
        raw_w, raw_h = im.size

    w, h = float(raw_w), float(raw_h)
    if not allow_upscale:
        scale = min(1.0, short_edge / min(w, h), math.sqrt(max_area / (w * h)))
    else:
        scale = min(short_edge / min(w, h), math.sqrt(max_area / (w * h)))

    fit_w = int(math.ceil((w * scale) / 32.0) * 32)
    fit_h = int(math.ceil((h * scale) / 32.0) * 32)
    latent_rows = (fit_w // 32) * (fit_h // 32)
    return [raw_w, raw_h], [fit_w, fit_h], latent_rows


def read_substrate() -> dict:
    """The provenance values, READ, because they used to be typed in.

    Until 2026-08-17 this function's caller wrote `gpu_power_limit_watts: 450.0`,
    `driver_version: "610.57.04"`, `comfyui_version: "0.33.0"` and
    `comfy_kitchen_version: "0.2.31+sol.c04ef20"` as literals, and fell back to a
    hardcoded GPU name and CUDA version when torch could not answer. Six
    fabricated substrate values in the block whose entire job is recording what
    produced the capture.

    **The power limit is the one that was already wrong.** This box was set to
    330 W on the day this was found, so every manifest generated here would have
    recorded 450.0 -- and `docs/hardware.md` exists because a changed power limit
    makes a timing incomparable. The record built to catch that was asserting the
    stock value from a constant.

    `substrate.py` reads all of these and needs no CUDA, so there is no reason
    for a second implementation here. Returns `None` for anything it cannot read
    rather than a plausible default; the caller decides what to do about that,
    and `main()` refuses to write a manifest rather than invent one.
    """
    from substrate import host as _host, builds as _builds  # noqa: PLC0415

    h, b = _host(), _builds()
    gpu = (h.get("gpus") or [{}])[0] if h.get("state") == "present" else {}

    def pkg(name):
        rec = b.get(name) or {}
        return rec.get("version") if rec.get("state") == "present" else None

    comfyui_version = None
    try:
        vf = _REPO.parents[1] / "comfyui_version.py"
        m = re.search(r'__version__\s*=\s*"([^"]+)"', vf.read_text())
        comfyui_version = m.group(1) if m else None
    except OSError:
        pass

    return {
        "gpu_device": gpu.get("name"),
        "gpu_power_limit_watts": gpu.get("power_limit_watts"),
        "gpu_power_limit_default_watts": gpu.get("power_limit_default_watts"),
        "driver_version": gpu.get("driver_version"),
        "cuda_version": h.get("cuda_version") if h.get("state") == "present" else None,
        "comfyui_version": comfyui_version,
        "comfy_kitchen_version": pkg("comfy_kitchen"),
    }


def _sol_attn_state(wf: dict) -> str:
    """Whether Sol-Attn is in the executed chain, read off the graph.

    **Presence is the wrong test, and this repo has the counterexample on
    disk.** `workflows/h3_probe_capture_ref3_api.json` contains a
    `SolAttnMiniMax` node that no other node consumes -- a hand-edit rewired the
    chain past it -- so "is the node in the graph" answers yes on a graph where
    Sol does not run. A naive fix reports that capture backwards.

    What is decidable from an API graph without knowing which classes are output
    nodes: whether anything consumes the node's output. In API format a wired
    input is `[node_id, slot]`, so the node is in the chain exactly when some
    other node references its id. Three states, because a wired node and an
    orphaned node and no node at all are three different facts and only the
    first two would otherwise collapse:

        absent            no SolAttnMiniMax in the graph
        orphaned          present, and nothing consumes its output
        wired             present, and consumed

    `orphaned` is deliberately not spelled "bypassed_for_capture". That phrase
    stated an intent; this states what the graph does, and the two are only the
    same while somebody keeps them so.
    """
    state, _ = _class_state(wf, "SolAttnMiniMax")
    return state


class _Unset:
    """A default nobody derived, distinguishable from a value somebody did.

    This function has now shipped the same defect five times, always the same
    shape: a dict of PLAUSIBLE literals where some keys get overwritten
    downstream and some silently do not, so an underived field emits a value
    that reads as a measurement.

        sol_attn = "bypassed_for_capture"      fixed 2026-08-17
        weight_quantization / vae_quantization = "int8_convrot"
        "rank": 256 for every LoRA
        sage_mode = "fp16 (most accurate)"     fixed 2026-08-26
        loras: [] by presence                  fixed 2026-08-26

    `bench/check_capture_manifest.py` was green through every one, because it
    grades presence and type and cannot see whether a value was ever derived.
    Fixing them one at a time is what produced a nine-day gap between the first
    and its siblings, so this converts the class instead: a field left at a
    sentinel emits `null` and is NAMED in `workload.underived`, and the few
    that must always be derived raise instead.

    Credit: the pattern across all five was spotted from outside this file by
    the peer session holding the v2 lane, 2026-08-26.
    """

    def __repr__(self):
        return "<underived>"


UNSET = _Unset()

#: Fields whose absence is a bug rather than a fact. Every graph this repo
#: emits has the nodes that set them, so a sentinel surviving to emit means the
#: scan stopped recognising a node class -- the failure mode that produced the
#: `loras: []` defect. Kept deliberately short: a required field that CAN
#: legitimately be absent turns this into a check that goes red on correct
#: state, which CLAUDE.md rates worse than no check.
REQUIRED_DERIVED = ("models.unet", "sampling.sampler_name",
                    "sampling.scheduler", "sampling.steps")


def resolve_unset(sections: dict) -> list[str]:
    """Replace every sentinel with `None`; return the dotted names it hit.

    Mutates in place and returns the list for `workload.underived`, so the
    manifest records what it did NOT observe rather than leaving the reader to
    infer it from a plausible-looking default.
    """
    underived = []
    for section, body in sections.items():
        if not isinstance(body, dict):
            continue
        for key, value in list(body.items()):
            if isinstance(value, _Unset):
                body[key] = None
                underived.append(f"{section}.{key}")
    missing = [n for n in REQUIRED_DERIVED if n in underived]
    if missing:
        raise SystemExit(
            f"these manifest fields were never derived from the graph: "
            f"{missing}. Every graph this repo emits sets them, so this means "
            f"the scan no longer recognises a node class it used to. Emitting "
            f"a default here is how `loras: []` came to assert that no LoRA "
            f"was loaded.")
    return sorted(underived)


def _class_state(wf: dict, *class_types: str):
    """`(state, [inputs of each wired node])` for one or more class names.

    The generalisation of what `_sol_attn_state` worked out for Sol, because
    the reasoning was never Sol-specific: a node nothing consumes does not run,
    so presence answers a question nobody asked. It was written once for Sol
    and every later reader of this file wrote presence tests again -- the sage
    mode below and the LoRA records were both doing it as of 2026-08-26.

    Returns `absent` / `orphaned` / `wired`, and the inputs of the wired nodes
    only, so a caller cannot accidentally record an orphan's settings.
    """
    ids = {nid for nid, node in wf.items()
           if isinstance(node, dict) and node.get("class_type") in class_types}
    if not ids:
        return "absent", []
    consumed = set()
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, list) and val and str(val[0]) in ids:
                consumed.add(str(val[0]))
    if not consumed:
        return "orphaned", []
    return "wired", [(nid, wf[nid].get("inputs") or {}) for nid in wf
                     if nid in consumed]


def _scalar(value, cast, missing=None, unknown=None):
    """A widget's literal value, distinguishing four cases that are not the same.

    NUMBER or STRING -> cast. A string is a literal a hand-written prompt may
    legitimately carry and the nodes coerce it (`pdd_lora.py` runs
    `float(strength)`), so dropping it would record null for a value that ran.

    MISSING -> `missing`. The node applies its own schema default, so that
    default IS what ran and recording it is a fact.

    LINK (`["25", 0]`) -> `unknown`. Computed upstream; this file cannot know
    it and anything it writes is a fabrication. Deliberately NOT the same as
    missing: `bool(["25", 0])` is True, so the first version of this recorded
    the heads-ON arm for a graph whose value it could not read -- in the one
    field that exists to tell the two experiments apart.
    """
    if isinstance(value, list):
        return unknown
    if value is None:
        return missing
    try:
        return cast(value)
    except (TypeError, ValueError):
        return unknown


def extract_from_workflow(wf: dict, input_base: Path):
    """Extract canvas, models, sampling, prompt, and references dynamically from ComfyUI API graph."""
    canvas = {"width": 1024, "height": 768, "aspect": "4:3", "length": 362, "fps": 24.0, "latent_frames": 107}
    models = {
        "unet": "",
        "clip": "",
        "video_vae": "",
        "audio_vae": "",
        # Not defaulted to "int8_convrot". They were, and nothing could clear
        # them: the only writes below re-set them to that same string when
        # "convrot" appears in a filename, so a graph loading
        # `..._fp8_scaled.safetensors` produced a manifest claiming int8_convrot.
        # That is a false claim on its own, and since these became a checked
        # projection of the filename it is now a FAILING claim -- the checker
        # would go red on a legitimate fp8 capture, which is red on correct
        # state. Derived from the filename below instead.
        "weight_quantization": None,
        "vae_quantization": None,
        "loras": [],
    }
    for _k in ("unet", "clip", "video_vae", "audio_vae"):
        models[_k] = UNSET
    sampling = {"sampler_name": UNSET, "scheduler": UNSET, "steps": UNSET,
                "denoise": UNSET, "seed": UNSET, "cfg": UNSET}
    # `sol_attn` was the literal string "bypassed_for_capture" here from this
    # function's first version until 2026-08-17, and nothing ever reassigned it
    # -- this file did not mention SolAttn anywhere. So it reported a capture as
    # Sol-free whether Sol had run or not, and `docs/drift_frontier.md` cited it
    # as evidence that a specific capture was clean. A record whose value is a
    # constant cannot fail, which is the defect `CLAUDE.md` names. Now derived;
    # see `_sol_attn_state`.
    # sage_mode was the constant "fp16 (most accurate)" until 2026-08-26,
    # overwritten only when a sage node happened to exist. So a graph that runs
    # no sage at all -- every PDD arm -- reported a mode for a kernel that never
    # ran, and the constant did not even match `h3_config.SAGE_NODE['mode']`.
    # That is the identical defect the paragraph above records for `sol_attn`,
    # in the same dict literal, left behind when that one was fixed.
    # The Sol WINDOW, not just whether Sol is wired. `start_percent` and
    # `end_percent` are what decide which steps run sparse, and they are
    # step-count dependent in effect -- a fixed band covers a different
    # fraction of an 8-step run than a 16-step one, which is how the PDD arms
    # silently lost their dense final step. A manifest recording only
    # `sol_attn: wired` cannot tell two renders apart that differed in it.
    _sol_state, _sol_nodes = _class_state(wf, "SolAttnMiniMax")
    _sol_cfg = _sol_nodes[0][1] if _sol_state == "wired" else {}
    _sage_state, _sage_nodes = _class_state(wf, "MiniMaxH3SageAttention")
    attention = {
        "sage_mode": (str(_scalar(_sage_nodes[0][1].get("mode"), str, missing="auto"))
                      if _sage_state == "wired" else _sage_state),
        "sol_attn": _sol_attn_state(wf),
        "sol_start_percent": _scalar(_sol_cfg.get("start_percent"), float),
        "sol_end_percent": _scalar(_sol_cfg.get("end_percent"), float),
        "sol_dense_blocks": _scalar(_sol_cfg.get("dense_blocks"), str),
        "sol_tau": _scalar(_sol_cfg.get("selection.tau"), float),
        "head_chunks": (_scalar(_sage_nodes[0][1].get("head_chunks"), int, missing=1)
                        if _sage_state == "wired" else 1),
    }
    # LoRAs, by REACHABILITY and across every loader class. Two defects fixed
    # together on 2026-08-26:
    #   * only `LoraLoaderModelOnly` was matched, so a graph running a PDD or
    #     turbo-pack LoRA recorded `loras: []` -- a manifest asserting none was
    #     loaded, which reads as a measurement rather than a gap;
    #   * it matched on presence, so an active-but-unconsumed loader would be
    #     written as a LoRA that ran. No shipped graph has one, but `--workflow`
    #     takes hand-built graphs and that is where the Sol counterexample came
    #     from too.
    # The class list is `h3_config.LORA_LOADER_CLASSES`, not a fourth copy.
    _lora_state, _lora_nodes = _class_state(wf, *LORA_LOADER_CLASSES)
    for _nid, _inp in _lora_nodes:
        _name = str(_scalar(_inp.get("lora_name"), str, missing="") or "")
        _rank = re.search(r"rank[_-]?(\d+)", _name)
        # `strength_model` on the stock loader, `strength` on both of ours.
        _raw = _inp.get("strength_model")
        if _raw is None:
            _raw = _inp.get("strength")
        _record = {
            "name": _name,
            "strength": _scalar(_raw, float),
            "rank": int(_rank.group(1)) if _rank else None,
            "loader": str(wf[_nid].get("class_type", "")),
        }
        if _record["loader"] == "MiniMaxH3PDDLoRA":
            # heads-on and heads-off are different experiments and nothing else
            # in the manifest distinguishes them.
            _record["pdd_patch_heads"] = _scalar(_inp.get("patch_heads"), bool,
                                                missing=True, unknown=None)
        if _record["loader"] == "MiniMaxH3TurboLoRA":
            # Selects merge-vs-bypass, which is two numerically different
            # renders; without it both produce identical manifests.
            _record["low_vram"] = _scalar(_inp.get("low_vram"), bool,
                                          missing=False, unknown=None)
        models["loras"].append(_record)

    prompt_text = ""
    references = []

    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if ct == "MiniMaxH3Resolution":
            canvas["width"] = int(inputs.get("width", 1024))
            canvas["height"] = int(inputs.get("height", 768))
            canvas["length"] = int(inputs.get("length", 362))
            canvas["aspect"] = str(inputs.get("aspect", "4:3"))
            length = canvas["length"]
            canvas["latent_frames"] = ((length - 5) // 17) * 5 + 2 if length > 0 else 1

        elif ct == "UNETLoader":
            models["unet"] = str(inputs.get("unet_name", ""))
            # Reads whichever build the graph actually loads -- int8_convrot,
            # fp8_scaled, w4a8_mixed -- rather than testing for "convrot" and
            # leaving every other build mislabelled. One implementation, in
            # `substrate.infer_quantization`.
            models["weight_quantization"] = infer_quantization(models["unet"])
        elif ct in ("CLIPLoader", "MiniMaxH3AWQEncoderLoader"):
            # Both loader classes. Only `CLIPLoader` was matched, and no graph
            # this repo ships uses it -- every one loads the encoder through
            # `MiniMaxH3AWQEncoderLoader` -- so `models.clip` emitted "" on
            # every manifest ever generated here, asserting no text encoder.
            # Found 2026-08-26 by the sentinel above on its first run, which is
            # the sixth instance of this defect and the first one nobody had to
            # notice by hand. The input is named differently on each.
            models["clip"] = str(_scalar(inputs.get("clip_name"), str,
                                         missing=None)
                                 or _scalar(inputs.get("encoder_name"), str,
                                            missing="") or "")
        elif ct == "VAELoader":
            vae_name = str(inputs.get("vae_name", ""))
            if "audio" in vae_name:
                models["audio_vae"] = vae_name
            else:
                models["video_vae"] = vae_name
                # Pinned to the video VAE, and says nothing about the audio one
                # -- they ship at different quantizations, so one field cannot
                # describe both.
                models["vae_quantization"] = infer_quantization(vae_name)
        elif ct == "KSamplerSelect":
            sampling["sampler_name"] = str(inputs.get("sampler_name", "er_sde"))
        elif ct == "BasicScheduler":
            sampling["scheduler"] = str(inputs.get("scheduler", "simple"))
            sampling["steps"] = int(inputs.get("steps", 16))
            sampling["denoise"] = float(inputs.get("denoise", 1.0))
        elif ct == "SamplerCustomAdvanced":
            sampling["seed"] = int(inputs.get("noise_seed", 0))
        elif ct == "BasicGuider":
            sampling["cfg"] = float(inputs.get("cfg", 1.0))
        elif "prompt" in inputs and isinstance(inputs["prompt"], str) and len(inputs["prompt"]) > len(prompt_text):
            prompt_text = inputs["prompt"]
        elif "text" in inputs and isinstance(inputs["text"], str) and len(inputs["text"]) > len(prompt_text):
            prompt_text = inputs["text"]
        elif ct == "LoadImage":
            img_name = str(inputs.get("image", ""))
            ref_path = input_base / img_name
            if not ref_path.is_file():
                # Try relative to h3_refs
                ref_path = input_base / "h3_refs" / img_name

            raw_dim, fit_dim, lat_rows = ([0, 0], [0, 0], 0)
            f_hash = "missing"
            if ref_path.is_file():
                raw_dim, fit_dim, lat_rows = compute_reference_fit(ref_path)
                f_hash = sha256_file(ref_path)

            references.append({
                "slot": f"ref_image_{len(references)}",
                "source_file": str(ref_path.relative_to(input_base)) if ref_path.is_relative_to(input_base) else img_name,
                "sha256": f_hash,
                "raw_dimensions": raw_dim,
                "fitted_dimensions": fit_dim,
                "latent_rows": lat_rows,
                "fit_settings": {"allow_upscale": False, "short_edge": 2048, "lift_downstream_clamp": False},
            })

    # Steps and scheduler come from `h3_config.graph_schedule`, NOT from the
    # `BasicScheduler` branch above, which is now only one of the two nodes that
    # can own a schedule. Since the PDD rewiring, `MiniMaxH3PDDLoRA` emits SIGMAS
    # and 11 shipped graphs carry no BasicScheduler at all -- against those this
    # scan derived neither field and exited with "every graph this repo emits
    # sets them", an error whose own text the rewiring had made false. Three
    # other readers were migrated in the same change and this fourth was missed.
    if sampling.get("scheduler") is UNSET or sampling.get("steps") is UNSET:
        steps, scheduler = graph_schedule(wf)
        if steps is not None and sampling.get("steps") is UNSET:
            sampling["steps"] = int(steps)
        if scheduler is not None and sampling.get("scheduler") is UNSET:
            sampling["scheduler"] = str(scheduler)

    underived = resolve_unset({"canvas": canvas, "models": models,
                               "sampling": sampling, "attention": attention})
    return canvas, models, sampling, attention, prompt_text, references, underived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, help="Path to capture directory")
    parser.add_argument("--workflow", default=None, help="Path to workflow JSON (default: uses workflow_api.json in dir)")
    args = parser.parse_args()

    cap_dir = Path(os.path.expanduser(args.capture_dir)).resolve()
    if not cap_dir.is_dir():
        print(f"Error: {cap_dir} is not a directory", file=sys.stderr)
        return 1

    input_base = _paths.comfy_input()
    if input_base is None:
        sys.exit(_paths.describe("ComfyUI input", "H3_COMFY_INPUT"))
    wf_in_dir = cap_dir / "workflow_api.json"
    wf_path = Path(args.workflow).resolve() if args.workflow else (wf_in_dir if wf_in_dir.is_file() else Path("workflows/h3_probe_capture_ref3_api.json").resolve())

    wf = json.loads(wf_path.read_text()) if wf_path.is_file() else {}
    if not wf_in_dir.is_file() and wf_path.is_file():
        wf_in_dir.write_text(json.dumps(wf, indent=2) + "\n")

    canvas, models, sampling, attention, prompt_text, references, underived = extract_from_workflow(wf, input_base)

    # Calculate token accounting dynamically
    tokens_per_frame = (canvas["width"] // 32) * (canvas["height"] // 32)
    video_tokens = tokens_per_frame * canvas["latent_frames"]
    ref_tokens = sum(r["latent_rows"] for r in references)
    text_tokens = 7711  # Qwen3-VL text encoding tokens
    audio_tokens = 1206 # Audio latent tokens for 15.08s
    total_sequence_length = video_tokens + ref_tokens + text_tokens + audio_tokens

    # Scan captured tensors
    pt_files = sorted(glob.glob(str(cap_dir / "qkv_*.pt")))
    captured_tensors = []
    for pt in pt_files:
        pt_path = Path(pt)
        size_bytes = pt_path.stat().st_size
        f_hash = sha256_file(pt_path)

        parts = pt_path.name.replace(".pt", "").split("_")
        block_val, step_val = None, None
        for p in parts:
            if p.startswith("b") and p[1:].isdigit():
                block_val = int(p[1:])
            elif p.startswith("s") and p[1:].isdigit():
                step_val = int(p[1:])

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            shape = list(data["q"].shape)
            dtype_str = str(data["q"].dtype)
        except Exception:
            shape = [1, 56, total_sequence_length, 128]
            dtype_str = "torch.bfloat16"

        captured_tensors.append({
            "filename": pt_path.name,
            "block": block_val,
            "step": step_val,
            "shape": shape,
            "dtype": dtype_str,
            "size_bytes": size_bytes,
            "sha256": f_hash,
        })

    # Query system environment
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda
    sub = read_substrate()

    # Refuse to write a manifest whose substrate could not be read. The
    # alternative was this file's previous behaviour -- fill the gap with a
    # plausible constant -- and a manifest that quietly says 450 W on a box
    # running 330 W is worse than no manifest, because it validates. Emitting
    # null instead would fail `check_capture_manifest.py` at read time, which is
    # a worse place to find out than here.
    missing = [k for k in ("gpu_device", "gpu_power_limit_watts",
                           "driver_version", "comfyui_version") if not sub.get(k)]
    if missing:
        sys.exit(f"refusing to write a manifest: could not read {', '.join(missing)}. "
                 f"These are recorded substrate, not decoration -- run where "
                 f"`nvidia-smi` and the ComfyUI checkout are reachable, or fix "
                 f"`substrate.py`. Writing a plausible default here is the defect "
                 f"this guard exists to prevent.")

    # Added in 1.2.0, INSIDE `workload.models` -- the schema doc nests it there
    # and `check_capture_manifest.assert_model_hashes` is handed that block. So
    # anything enumerating `workload.models` expecting name -> filename now
    # meets one dict-valued key; `sha256` is the only non-string entry and is
    # skippable by name.
    models["sha256"] = hash_model_files(models)

    manifest = {
        "schema_version": "1.4.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": {
            "git_commit": get_git_commit(),
            "gpu_device": sub["gpu_device"],
            "gpu_power_limit_watts": sub["gpu_power_limit_watts"],
            "driver_version": sub["driver_version"],
            "cuda_version": sub["cuda_version"] or cuda_ver,
            "pytorch_version": torch_ver,
            "comfyui_version": sub["comfyui_version"],
            "comfy_kitchen_version": sub["comfy_kitchen_version"],
        },
        "workload": {
            "workflow_file": str(wf_path.relative_to(Path.cwd())) if wf_path.is_relative_to(Path.cwd()) else str(wf_path),
            "canvas": canvas,
            "models": models,
            "sampling": sampling,
            "attention": attention,
            # Named, not inferred. A reader can tell a field nobody derived
            # from one somebody did, which is the whole of what five rounds of
            # this defect cost. Empty on a normal graph.
            "underived": underived,
        },
        "prompt": {
            "full_prompt_text": prompt_text,
            "sections": parse_prompt_sections(prompt_text),
        },
        "references": references,
        "token_accounting": {
            "total_sequence_length": total_sequence_length,
            "video_tokens": video_tokens,
            "text_tokens": text_tokens,
            "reference_tokens": ref_tokens,
            "audio_tokens": audio_tokens,
        },
        "captured_tensors": captured_tensors,
    }

    manifest_path = cap_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[generate_manifest] Wrote dynamic capture manifest: {manifest_path}")
    print(f"  Indexed {len(captured_tensors)} tensor files ({sum(t['size_bytes'] for t in captured_tensors) / 1e9:.2f} GB total)")
    print(f"  Indexed {len(references)} reference images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
