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
"""

from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for name, spec in EXAMPLES.items():
        spec = dict(spec)
        note = spec.pop("note")
        g = graph(**spec)
        (args.out / name).write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
        used = sorted({v["class_type"] for v in g.values()})
        noncore = [c for c in used if c.startswith("MiniMaxH3PDD")]
        print(f"  {name}")
        print(f"     {note}")
        print(f"     {len(g)} nodes, {len(used)} classes; from the pack: {noncore or 'none'}")
    print(f"\nwrote {len(EXAMPLES)} example(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
