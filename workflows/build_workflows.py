#!/usr/bin/env python3
"""Generate the MiniMax H3 test workflows, in API format and UI format.

Why a generator instead of hand-edited JSON: the three bundled ComfyUI
templates are not equally editable. `video_minimax_h3_r2v` is a flat graph,
but t2v and i2v hide the entire sampler stack inside a subgraph named
"Image to Video (MiniMax H3)". Editing a subgraph by hand -- or converting
one to API format by hand -- is how you end up measuring a graph that is
subtly not the one you meant to run. Building them all from one description
keeps them identical everywhere they should be identical, and makes the
things that differ (which conditioning node, which checkpoint, whether a LoRA
is applied) obvious.

The sage node goes between `UNETLoader` and the sampler stack. Note that
MODEL forks to *two* consumers -- `BasicScheduler.model` and
`BasicGuider.model`. Rewiring only the guider leaves the scheduler reading
sigmas off the unpatched model; the render still succeeds, which is why the
mistake survives. Every graph here is generated from a single `model_src`
variable so the fork cannot drift.

Run it to regenerate:

    python3 build_workflows.py

It writes the JSON next to itself and validates every API graph against a
live ComfyUI's /object_info (or a cached copy passed with --object-info).
Validation is static -- nothing is submitted, nothing touches the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Model names, sampler settings, canvas geometry and the SolAttn knobs all
# used to live here in duplicate with the bench. Single source is
# h3_config.py -- see its docstring for why that matters.
from h3_config import (  # noqa: E402
    CANVAS, FPS, LENGTH, LONG_LENGTH, MODELS, REF_LORA, REF_LORA_STRENGTH,
    SAMPLING, SAGE_NODE, SEED, SIGMA_SHIFT, SOL_RECOMMENDED,
    TURBO_LORA, TURBO_LORA_STRENGTH, TURBO_SHIFT, TURBO_STEPS,
)

# Prompt for the long presets (345 frames, 14.375s). That needs a shot timeline,
# not one continuous beat -- the guide wants numbered shots with explicit cut
# times past a few seconds, and a 15s request against a 6s prompt leaves the
# model twelve seconds it was never told about.
LONG_T2V_PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, handheld, shallow depth of field. A medium shot frames a courier in a soaked red jacket standing over a bicycle at a city crosswalk in heavy evening rain, wet asphalt throwing back the signal lights, a brick facade with iron railings filling the background. The camera tracks right at medium amplitude and moderate speed as she snaps her helmet strap and pushes off.
[Shot 2] At 00:04.000, the shot cuts to a low tracking shot running alongside the bicycle as it crosses the junction, spray coming off the tyres, painted lane markings streaming past underneath.
[Shot 3] At 00:08.000, the camera whip pans up to a wide shot of the street as she cuts between two parked cars, pigeons scattering off the railings, neon shopfront signs reflected in the puddles.
[Shot 4] At 00:11.500, the shot changes to a close shot of her face under the helmet, rain streaking across the lens, as she glances back over her shoulder and then forward again, breathing hard.

overall_soundscape: steady heavy rain on asphalt and metal, tyre hiss through standing water, the click and rattle of a bicycle chain, a car horn twice in the middle distance, wings clattering as the pigeons take off, her breathing close and rhythmic under the helmet.

non_diegetic_music: none."""

# h264-mp4 rather than h265 or an nvenc variant: software x264 at crf 19 is
# the most portable mp4 there is, and the nvenc paths trade quality per bit
# for encode speed on a file that takes seconds to write next to a render
# that takes minutes. Switch to video/h265-mp4 if size matters more than
# playing everywhere.
VIDEO_FORMAT = "video/h264-mp4"

# Placeholder input filenames. These are whatever the local install happens
# to have; swap them for your own before running an i2v or r2v graph.
PLACEHOLDER_IMAGE_A = "1-man.png"
PLACEHOLDER_IMAGE_B = "2-mountain_landscape.png"

T2V_PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a lone lighthouse keeper on a wet stone balcony at dawn, wearing a heavy oilskin coat, the lamp housing glowing behind him. Grey-blue sea fog rolls past below the railing and gulls cross the frame. The camera pushes in with small amplitude at slow speed as he raises a brass telescope, holds it steady against his eye, then lowers it and turns toward the light. [Shot 2] At 00:03.000, the shot cuts to a close-up of the rotating lamp assembly, the beam sweeping past the lens and out into the fog.

overall_soundscape: A low sea swell breaks against stone under a steady wind, with gulls calling overhead. A distant foghorn sounds twice, and the lamp mechanism turns with a slow mechanical grind.

non_diegetic_music: Sustained low strings at a slow tempo with a single sparse piano figure, holding without a swell."""

I2V_PROMPT = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the subject shown in <Picture 1> holds its position, framing, lighting, and colors exactly as established in the image. The camera pushes in with small amplitude at slow speed while the subject begins to move, the surrounding scene staying continuous with the reference frame.

overall_soundscape: Quiet room tone with a low ambient hum continues throughout, joined by soft physical sounds from the subject's movement.

non_diegetic_music: N/A"""

R2V_PROMPT = """subject_definitions:
<Subject 1> is the main character in <Picture 1>, whose face, hair, and clothing are carried into the target video.
<Subject 2> is the environment in <Picture 2>, whose architecture, palette, and lighting are carried into the target video.

summary:
[reference generation] The target video places <Subject 1> inside <Subject 2> for a single continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - architecture, palette, and lighting are retained.

detailed_description:
The target video is in a cinematic live-action style with soft directional lighting.
[Shot 1] A medium shot establishes <Subject 2>, then <Subject 1> enters from the left and stops at the center of the frame. The camera trucks right with small amplitude at slow speed as <Subject 1> turns toward the light and looks off-screen.

overall_soundscape:
Steady interior room tone continues throughout, with soft footsteps and fabric movement as <Subject 1> crosses the frame.

non_diegetic_music:
N/A"""


# --------------------------------------------------------------------------
# API format
# --------------------------------------------------------------------------

sys.path.insert(0, str(HERE.parent))
from h3_rules import (  # noqa: E402
    aspect_in_range, describe_aspect_range, describe_length,
    duration_in_range, max_legal_length, min_legal_length,
)


def _ref_short_edge():
    """ComfyUI's reference short edge, read rather than repeated.

    `MiniMaxH3ReferenceFit` defaults this input to core's
    `REF_IMAGE_SHORT_EDGE`. Writing 2048 into the graph as a literal would be
    a second place to edit that agrees with the first only by inspection --
    if core ever moves the constant, the node's default moves and the shipped
    graphs quietly do not. There is no test that could tell those apart,
    because a duplicated decision has no observable disagreement until the
    day it disagrees.
    """
    # This script is designed to run without ComfyUI importable -- it
    # validates over HTTP -- so put the root on the path just for this.
    root = HERE.parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from comfy_extras.nodes_minimax_h3 import REF_IMAGE_SHORT_EDGE

    return REF_IMAGE_SHORT_EDGE


def _check_geometry(length, canvas):
    """Refuse to emit a graph the reference would reject.

    This config shipped 362 frames for a week. It is on the 17n+5 grid, it is
    inside ComfyUI's own 3600 limit, and it renders -- it is just 15.083s
    against a 15s ceiling the reference enforces and ComfyUI does not. Nothing
    in the pipeline said so, which is exactly the failure this repo exists to
    make loud, so the generator now holds the rule rather than a comment.
    """
    cv = dict(CANVAS, **canvas)
    if not duration_in_range(length):
        raise SystemExit(
            f"length {describe_length(length)} is outside MiniMax H3's 5-15s "
            f"window; legal counts are {min_legal_length()}-{max_legal_length()} "
            f"on the 17n+5 grid. Fix LONG_LENGTH/LENGTH in h3_config.py."
        )
    if not aspect_in_range(cv["width"], cv["height"]):
        raise SystemExit(
            f"canvas {cv['width']}x{cv['height']} is aspect "
            f"{cv['width'] / cv['height']:.3g}, outside H3's trained "
            f"{describe_aspect_range()} range."
        )


def build_api(task: str, *, sage: bool = True, prompt: str | None = None,
              length: int = LENGTH, seed: int = SEED,
              sol: dict | None = None, canvas_mode: str = "fit_to_canvas",
              stamp: bool = False, unet: str | None = None,
              lora: tuple[str, float] | None = None,
              steps: int | None = None, shift: dict | None = None,
              head_chunks: int | None = None, ref_upscale: bool = True,
              out_prefix: str | None = None, **canvas) -> dict:
    """API-format graph, submittable as {"prompt": <this>} to POST /prompt.

    Node ids match `bench/bench_e2e_h3.py` so a timing run and a hand-edited
    graph can be compared node-for-node; "10" is the sampler in every graph.

    `unet` overrides the checkpoint the task would otherwise pick. The two are
    separable because the ref-LoRA probe needs r2v *conditioning* driven by the
    *fl2va* checkpoint, which is not a combination any task name describes.
    `lora` is (name, strength) and inserts a LoraLoaderModelOnly.
    """
    if task not in ("t2v", "i2v", "r2v"):
        raise ValueError(task)
    _check_geometry(length, canvas)
    ref = task == "r2v"
    cv = dict(CANVAS, **canvas)
    prompt = prompt if prompt is not None else {
        "t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "r2v": R2V_PROMPT}[task]

    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet or MODELS["unet_ref2va" if ref else "unet_fl2va"],
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["clip"], "type": "minimax",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": SAMPLING["sampler"]}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": None, "scheduler": SAMPLING["scheduler"],
                         "steps": steps if steps is not None else SAMPLING["steps"],
                         "denoise": SAMPLING["denoise"]}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": None, "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 1]}},
        # Both decoders read the same packed AV latent and each pulls out its
        # own half; this is not a mistake in the wiring.
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        # VHS_VideoCombine instead of CreateVideo -> SaveVideo: one node, and
        # it muxes the audio itself. Node id 13; 14 is retired with SaveVideo.
        # The format sub-widgets (pix_fmt/crf/save_metadata/trim_to_audio) are
        # h264-mp4's own, and they are keyed here exactly as they are named in
        # /object_info's format spec -- VHS reads them by name, not position.
        # trim_to_audio stays False: H3 generates the pair jointly, so trimming
        # video to the audio track can only lose frames it meant to keep.
        "13": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["11", 0], "audio": ["12", 0],
                          "frame_rate": FPS, "loop_count": 0,
                          "filename_prefix": out_prefix or f"Video/h3_{task}",
                          "format": VIDEO_FORMAT, "pix_fmt": "yuv420p",
                          "crf": 19, "save_metadata": True,
                          "trim_to_audio": False,
                          "pingpong": False, "save_output": True}},
    }

    if ref:
        g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                             "prompt": prompt, "width": cv["width"], "height": cv["height"],
                             # 'max' rather than 'match', and the pairing is
                             # load-bearing: under 'match' the stock node sizes
                             # references from the video's pixel area and never
                             # reads the 2048 constant, so the fit nodes below
                             # would be undone and their two resamples wasted.
                             "length": length, "ref_image_size": "max",
                             # Autogrow slots are addressed by their flat dotted
                             # path; ComfyUI reassembles them into the nested
                             # dict the node signature expects. Slot ordinals are
                             # 0-based but the prompt tags are 1-based, so
                             # ref_image_0 is <Picture 1>.
                             "ref_images.ref_image_0": ["24", 0],
                             "ref_images.ref_image_1": ["25", 0]}}
        g["15"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_A}}
        g["16"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_B}}
        # One fit node per reference. ComfyUI clamps reference scaling with
        # min(1.0, 2048/short_edge) where the reference pipeline has none, so
        # a reference smaller than 2048 on its short side arrives under-sized
        # and identity fidelity comes out of those vision tokens. Ids 24/25.
        for nid, src in (("24", "15"), ("25", "16")):
            g[nid] = {"class_type": "MiniMaxH3ReferenceFit",
                      "inputs": {"image": [src, 0], "allow_upscale": ref_upscale,
                                 "short_edge": _ref_short_edge(),
                                 "lift_downstream_clamp": False}}
    else:
        inputs = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                  "width": cv["width"], "height": cv["height"], "length": length}
        if task == "i2v":
            # first_frame only. Wiring node 17's last_frame from a second
            # LoadImage turns this into the fl2va task the checkpoint is named
            # for; the model and every other node stay the same.
            #
            # The keyframe never reaches node 5 directly. MiniMaxH3ImageToVideo
            # stretches the first keyframe onto width/height non-uniformly --
            # 2.33x on a 3:4 still at the default canvas -- so node 17 fits it
            # first and hands over both the image and the size it was fitted to.
            # Node 5's own resize is then a bit-identical no-op.
            g["15"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_A}}
            # `length` goes through node 17 too, so the reference's 5-15s
            # window is enforced by the graph rather than only by the
            # generator. A graph edited in the UI afterwards -- which is the
            # normal way these get used -- keeps the check.
            g["17"] = {"class_type": "MiniMaxH3KeyframeCanvas",
                       "inputs": {"first_frame": ["15", 0], "mode": canvas_mode,
                                  "width": cv["width"], "height": cv["height"],
                                  "length": length}}
            inputs["first_frame"] = ["17", 2]
            inputs["width"] = ["17", 0]
            inputs["height"] = ["17", 1]
            inputs["length"] = ["17", 5]
        g["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}

    model_src = ["1", 0]
    if lora is not None:
        # Before the attention patches, not after. Either order renders -- a
        # LoRA patches weights and our node patches an attention function, so
        # they touch different surfaces -- but applying the LoRA clones the
        # ModelPatcher, and keeping that clone upstream of both attention
        # nodes avoids inserting it between the two that have to compose.
        # h3_config.CHAIN does not list this node and should not: it pins the
        # sage-then-Sol order, which is the part that is load-bearing, and a
        # LoRA in front of both is orthogonal to it.
        # Node id 18; 20/21/22 are already spoken for.
        g["18"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_src, "lora_name": lora[0],
                              "strength_model": lora[1]}}
        model_src = ["18", 0]
    # Always present, at the base checkpoint's own 12/3, so it changes nothing
    # by default. It is here to be edited: the turbo LoRAs carry their own
    # training shifts (the 768p 4-step wants 6/3), and a graph without this
    # node gives you nowhere to set that and no hint you needed to. Upstream of
    # sage so the sage-then-Sol adjacency below stays intact -- this patches
    # model sampling, which is a different surface from either of them.
    # Node id 19; 18 is the LoRA and 20/21/22 are already spoken for.
    g["19"] = {"class_type": "MiniMaxH3SigmaShift",
               "inputs": {"model": model_src,
                          **(shift if shift is not None else SIGMA_SHIFT)}}
    model_src = ["19", 0]
    if sage:
        g["20"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": model_src, **dict(
                       SAGE_NODE,
                       **({} if head_chunks is None
                          else {"head_chunks": head_chunks}))}}
        model_src = ["20", 0]
    if sol is not None:
        # After sage, never before -- SolAttn composes with the attention
        # patches it finds, and reversed it overwrites ours and you silently
        # get sage only. Node id 21 matches `bench/bench_e2e_h3.py`.
        g["21"] = {"class_type": "SolAttnPatch",
                   "inputs": {"model": model_src, **sol}}
        model_src = ["21", 0]
    # Last in the chain, because it asserts what the composition ended up
    # as, not what any one node intended. Sol-Attn negotiates with our
    # override through a duck-typed attribute that both sides rewrote within
    # a minute of each other once already; when that seam breaks the render
    # still succeeds and is quietly slower or numerically different. This
    # turns that into a refused render. `exercise` stays on: install-time
    # evidence is exactly what today has taught us not to trust.
    g["23"] = {"class_type": "SageChainAssert",
               "inputs": {"model": model_src, "require_override": True,
                          "require_forward_patch": True, "exercise": True,
                          "warn_only": False}}
    model_src = ["23", 0]

    # Reports what the assembled conditioning actually costs, before the
    # sampler runs. Pass-through, so it cannot change the render.
    g["26"] = {"class_type": "MiniMaxH3Preflight",
               "inputs": {"conditioning": ["5", 0], "samples": ["5", 1]}}

    # The fork. Both consumers, always, from the same variable.
    g["8"]["inputs"]["model"] = model_src
    g["9"]["inputs"]["model"] = model_src
    g["9"]["inputs"]["conditioning"] = ["26", 0]
    g["10"]["inputs"]["latent_image"] = ["26", 1]

    if stamp:
        # Bench only. Sits inline between the sampler and both decoders so it
        # has a real data dependency on the sampler's output -- ComfyUI orders
        # by dependency, not graph position, and a stamp with no such edge can
        # legally run BEFORE sampling and record pre-render state. It also
        # needs SIGMAS: n_sparse is the sigma window intersected with the
        # schedule and is readable from nothing else.
        g["22"] = {"class_type": "MiniMaxH3ProvenanceStamp",
                   "inputs": {"latent": ["10", 0], "model": model_src,
                              "sigmas": ["8", 0], "note": f"bench {task}"}}
        g["11"]["inputs"]["samples"] = ["22", 0]
        g["12"]["inputs"]["samples"] = ["22", 0]
    return g


# --------------------------------------------------------------------------
# UI format
# --------------------------------------------------------------------------

class UIGraph:
    """Minimal litegraph workflow writer.

    Field shapes are copied from the bundled `video_minimax_h3_r2v` template,
    which is the one H3 template that is already flat, so this emits the same
    dialect the frontend just loaded from disk.

    Deliberately no widget-to-input conversions and no helper nodes
    (ResolutionSelector, ComfyMathExpression, PrimitiveStringMultiline). The
    templates use those for convenience, but every one of them is another
    place a hand-edit can go wrong, and the point of these copies is to be
    easy to edit. Resolution, length and prompt are plain widget values on
    the conditioning node.
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node = 1
        self._next_link = 1

    def add(self, type_: str, pos, *, widgets=None, inputs=None, outputs=None,
            size=(320, 100), title=None):
        nid = self._next_node
        self._next_node += 1
        n = {
            "id": nid, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [dict(i) for i in (inputs or [])],
            "outputs": [dict(o) for o in (outputs or [])],
            "properties": {"Node name for S&R": type_},
        }
        if widgets is not None:
            # A dict stays a dict. Most nodes serialize widgets_values as a
            # positional list, but a node whose widget set depends on another
            # widget cannot -- VHS_VideoCombine adds pix_fmt/crf/... after
            # `format`, so position cannot address them and the frontend
            # writes a keyed object instead. `list(a_dict)` silently yields
            # the keys, which is a graph that loads and renders with every
            # setting wrong.
            n["widgets_values"] = (dict(widgets) if isinstance(widgets, dict)
                                   else list(widgets))
        if title:
            n["title"] = title
        self.nodes.append(n)
        return nid

    def _node(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def link(self, src, src_slot, dst, dst_input_name, type_):
        lid = self._next_link
        self._next_link += 1
        s, d = self._node(src), self._node(dst)
        s["outputs"][src_slot].setdefault("links", [])
        if s["outputs"][src_slot]["links"] is None:
            s["outputs"][src_slot]["links"] = []
        s["outputs"][src_slot]["links"].append(lid)
        for inp in d["inputs"]:
            if inp["name"] == dst_input_name:
                inp["link"] = lid
                break
        else:
            raise KeyError(f"{d['type']} has no input {dst_input_name!r}")
        self.links.append([lid, src, src_slot, dst, self._input_index(d, dst_input_name), type_])
        return lid

    @staticmethod
    def _input_index(node, name):
        return [i["name"] for i in node["inputs"]].index(name)

    def _topo_order(self):
        # `order` is advisory -- the frontend recomputes it -- but an
        # inconsistent value shows up as nodes drawn in a nonsense sequence,
        # so emit a real topological order.
        incoming = {n["id"]: set() for n in self.nodes}
        for lid, src, _ss, dst, _ds, _t in self.links:
            incoming[dst].add(src)
        order, placed = {}, set()
        i = 0
        while len(placed) < len(self.nodes):
            progressed = False
            for n in self.nodes:
                nid = n["id"]
                if nid in placed or not incoming[nid] <= placed:
                    continue
                order[nid], i = i, i + 1
                placed.add(nid)
                progressed = True
            if not progressed:
                raise RuntimeError("cycle in graph")
        for n in self.nodes:
            n["order"] = order[n["id"]]

    def dump(self, workflow_id: str) -> dict:
        self._topo_order()
        return {
            "id": workflow_id, "revision": 0,
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4,
        }


def _in(name, type_, *, optional=False, widget=False, label=None):
    d = {"name": name, "type": type_, "link": None}
    if label:
        d["label"] = label
    if optional:
        d["shape"] = 7
    if widget:
        d["widget"] = {"name": name}
    return d


def _out(name, type_):
    return {"name": name, "type": type_, "links": None}


# Text for the in-graph notes. Kept next to the builder rather than in
# docs/h3_geometry_and_nodes.md on purpose: that doc is the long form, this
# is what you need with the graph open. Numbers here come from
# comfy_extras/nodes_minimax_h3.py, not from lore.
_NOTE_GEOMETRY = """\
## You pick an aspect ratio. The resolution follows from it.

`adapt_canvas()` reads your two numbers as a ratio and derives the pixels:
short edge starts at 768, the area caps at 1,032,192 (768x1344), each axis
rounds to 32. Asking for 4K gives the same resolution as 720p at the same
ratio. Exactly 95 resolutions exist across the legal 1/4 to 4 aspect range.

Full table, the derivation, and the length and int32 axes:
`docs/h3_resolutions.md`.

## The fourteen worth knowing

| Ask for | Resolution | Video tokens/frame | Attention |
|---|---|---|---|
| 21:9 | 1536x672 | 1008 | 1.00x |
| 2:1 | 1440x704 | 990 | 0.96x |
| 16:9 | 1344x768 | 1008 | 1.00x |
| 5:3 | 1280x768 | 960 | 0.91x |
| 3:2 | 1152x768 | 864 | 0.73x |
| 4:3 | 1024x768 | 768 | 0.58x |
| 5:4 | 960x768 | 720 | 0.51x |
| 1:1 | 768x768 | 576 | 0.33x |
| 4:5 | 768x960 | 720 | 0.51x |
| 3:4 | 768x1024 | 768 | 0.58x |
| 2:3 | 768x1152 | 864 | 0.73x |
| 9:16 | 768x1344 | 1008 | 1.00x |
| 1:2 | 704x1440 | 990 | 0.96x |
| 9:21 | 672x1536 | 1008 | 1.00x |

All fourteen reproduce themselves, so typing one into width/height gives it
back. 1:1 costs a third of 16:9 at the same frame count, and attention
dominates the step, so this is the largest speed lever anywhere, larger than
any kernel or sparsity setting.

Attention goes as the square of the token count. Video tokens per frame are
`(w//32) * (h//32)`, which is symmetric, so portrait and landscape of a ratio
cost the same. 16:9 against 9:16 is a quality question, never a speed one.

## Where the 32 comes from

The VAE compresses space by 16, then the DiT patchifies that latent 2x2
before attending it. 16 x 2 = 32. Divisible by 16 alone leaves an odd latent
axis the patchify cannot tile.

Core's conditioning nodes do not apply `adapt_canvas()` to the video
resolution at all: width and height are plain ints at step 32, so what you
type is what you get. The 768 and the area cap describe the trained family,
not a limit the node enforces.

## Two things that surprise people

The short edge is not always 768. It is 768 only while the area cap does not
bind, roughly 3:4 through 7:4. Outside that the cap takes over: 21:9 is
1536x672, 9:21 is 672x1536.

1.00x is not the ceiling. Rounding to 32 can land above the 16:9 token count.
Ask for 23:7 and you get 1856x576, which is 1044 video tokens against 1008,
so 1.073x the attention for no extra pixels. That is the worst case in the
set. Nearby ratios behave: 29:9 gives 1824x576 at 1.036x. Stay on the
fourteen unless you have a reason.

## If you want this decided for you

`MiniMax H3 Keyframe Canvas` (this repo) derives the resolution from your
first keyframe the way the reference pipeline does, fits the keyframes onto
it, and reports the cost before you render. The first-frame graph is wired
that way. Text-to-video has no keyframe to derive from, so type a row above.

## Length rounds up to n % 17 == 5

Ask 200, get 209. Ask 300, get 311. Near the top: 311, 328, 345.

345 is the ceiling, not 362. ComfyUI's tooltip says ~124-362 and its node
accepts up to 3600, but the reference generates 5-15s at 24fps and applies
that ceiling after the rounding. 362 is 15.083s, so it is refused; 345 is
14.375s. There is no on-grid count at exactly 15.0s. Ask for 346 and you get
362, which is why the check has to run on the rounded number.

At 345 frames attention is ~76% of the step, against ~50% at 124, so long
clips are where sparsity and kernel work pay off most.

345 frames is the frame-count ceiling, not the sequence-length ceiling. At
1344x768 it is S=108,078, which is already past the fused-layout int32
crossing at 99,864 tokens. That is safe here only because this repo's node
refuses any sageattention without `sageattn_consume`. See the doc.
"""


_NOTE_NODES = """\
## Node order is load-bearing

```
Load Diffusion Model
  -> ModelSamplingMiniMaxH3       (sigma shift; anywhere before the fork)
  -> MiniMax H3 SageAttention     (this repo)
  -> SolAttnPatch                 (must be AFTER)
  -> BasicScheduler + BasicGuider (MODEL forks to BOTH)
```

**Sol-Attn must come second.** It composes with the attention patch it
finds; reversed, it overwrites ours and you silently get sage only, with no
error and no log line saying so.

**The sigma shift is here to be changed, not because it does anything at
12/3.** Those are the base checkpoint's training shifts, so the node is a
no-op as shipped. The turbo LoRAs inherit the sampler's shift instead of
carrying their own, and the 4-step v1.0 768p one was distilled at video
shift **6** -- and that is the variant trained at 1344x768, this canvas. Load
it without changing this and you sample it off a schedule it never saw.
Steps move with it too: 16 is a base-model number, these want 4 or 8. The
4-step v0.1 and 8-step v1.0 were both distilled at 12/3 and need no change
here.

**MODEL forks to two consumers.** Rewiring only the guider leaves the
scheduler reading sigmas off the unpatched model, and the render still
succeeds -- which is why that mistake survives.

## Check it is actually running, once per graph change

Turn `verbose` on in SolAttnPatch for one render, then off. You want three
lines. **Read them in the terminal** -- piping or redirecting block-buffers
the output and they may not appear even when everything is fine.

```
sage routing: arch=sm89 ... pv_accum=fp32+fp16 -> fp8_cuda++
[sol_attn] chaining onto an existing attention override
[sol_attn] sparse (1, ..., 56, 128) tau=2.0 int8 pointer
```

Line 1: sage engaged on the fast kernel. Line 3: sparse engaged at your tau.
**Line 2 is the order check** -- it only prints when Sol-Attn finds sage's
override already installed. Missing means the nodes are backwards and you
are paying full price for a render that otherwise looks fine.

## What each node is here for

- **ModelPreviewOverrideKJ** -- taeh3 preview, and it is arguably the
  largest optimization here rather than a convenience. Killing a bad seed at
  90s instead of 11 minutes saves ~9.5 min; the entire kernel and sparsity
  stack saves ~7 min per render. If one render in three is a bad seed the
  preview beats everything else combined -- and they compound rather than
  compete.
- **MiniMax H3 SageAttention** -- INT8-QK / FP8-PV kernel on all 50 DiT
  attention forwards, plus an `optimized_attention_override` registration.
  That second part is what lets Sol-Attn compose instead of bypassing sage.
- **SolAttnPatch** -- block-sparse attention. Settings are pinned from
  `workflows/h3_config.py`; edit there and regenerate, not here.

## Deliberately absent

- **MiniMaxH3MemoryEfficientSageAttentionPatch** (KJNodes) -- same job as
  our node, patches the same key, so they conflict. Ours also registers the
  override.
- **MiniMaxLowVRAMAttention** -- head chunking. ~1070 MiB saved, but 1000
  attention calls become 4000. On 24GB freed VRAM converts to wall-clock at
  a ~2.6% ceiling. Take it only if you are actually hitting OOM.
- **MiniMaxChunkFeedForward** -- at 362 frames attention peaks ~17.8 GiB
  against FFN's 9-12, so it chunks a peak that is not binding. Short-clip
  feature.
- **PathchSageAttentionKJ** -- global no-guard sage switch. Prefer the
  per-workflow node.
"""

# f-string, because the strength appears in the prose and the widget it
# describes comes from REF_LORA_STRENGTH. Hardcoding it here is how a graph
# ends up shipping a note that contradicts its own node.
def _probe_note(subject, companion, changed, compare, expect):
    """Note for a probe graph: one variable, its twin, and what to look at.

    A probe that does not name its companion and its seed is a graph with an
    unusual setting, not an experiment. Every one of these is identical to its
    twin except the line under "what differs", and they share
    `h3_config.SEED`, so anything you see between them is that line.
    """
    return f"""\
## Probe: {subject}

**Run this against `{companion}`.** Same seed ({SEED}), same prompt, same
canvas, same everything except one setting. That is the whole design: if the
seed moved between the two, the difference you are looking for would be
underneath the difference you are not.

**What differs:** {changed}

**What to compare:** {compare}

**What to expect:** {expect}

This is a probe, not a render config. If you like what one side does, change
the setting in the shipped graph rather than rendering from this file.
"""


_NOTE_SIZING = """\
## What the sizing nodes decide, and what Preflight tells you

**Preflight is pass-through.** It changes nothing. It reads the assembled
conditioning through the model's own `PackedLayout`, so the sequence length
it draws is the one attention will actually run at.

Read it top to bottom:

```
1152x768  trained family  864 video tokens/frame
124 frames (5.167s)  37 latent frames
sequence length 52,702
  video         31,968  ############........   60.7%
  references    17,216  #######.............   32.7%
  text           3,104  #...................    5.9%
  audio            414  ....................    0.8%
if the aspect ratio changed, same length:
  1:1   768x768      42,046    -20%
  16:9  1344x768     58,030    +10%
```

The percentages are the decision. Reference tokens are attended at every
sampling step exactly as video tokens are, so references at a third of the
sequence means a third of your attention cost is spent describing them.

**"trained family" vs "OUTSIDE trained family".** Core's conditioning nodes
take width and height as plain ints and never call `adapt_canvas`, so the 768
short edge and the 768x1344 area cap constrain nothing you type. 1024x1024 is
legal, renders, costs more per frame than 16:9, and is outside the family the
checkpoint was trained on. Outside is a choice, not an error -- but it should
be one you made on purpose.

## The two resolution nodes are not interchangeable

- A **keyframe** is patchified on the video's own latent grid, so its
  resolution must equal the video's. That is why *MiniMax H3 Keyframe
  Resolution* outputs width and height: the keyframe decides them.
- A **reference** is patchified on its own grid, so its resolution only sets
  how many vision tokens it contributes. That is why *MiniMax H3 Reference
  Resolution* does not output width and height.

You will never want both in one graph.

## Reference Resolution needs ref_image_size on `max`

This pairing is load-bearing, not tidiness. Under `match` the stock node
sizes references from the video's pixel area and never reads the 2048
constant, so the fit nodes would run, resample twice, and be undone. This
graph ships with `max` set. If you switch it back to `match`, delete the fit
nodes too or you are paying for nothing.

`allow_upscale` on is the released pipeline's behaviour: it scales until the
short side reaches 2048, enlarging small references. ComfyUI's own rule only
ever shrinks. Upscaling adds tokens, not detail, so whether it helps an
already-small source is unmeasured -- turn it off and watch the Preflight
percentages if you want the cheap version.

**The EXPERIMENTAL clamp lift is off and should stay off** unless you are
running an experiment and expect to discard the result. It monkeypatches a
core node for one call and pushes references past what the checkpoint was
conditioned at.
"""


_NOTE_TURBO = f"""\
## Turbo LoRA: what the training resolution means

This graph loads the **8-step v1.0** LoRA at {TURBO_STEPS} steps, shift
{TURBO_SHIFT["shift_video"]:g}/{TURBO_SHIFT["shift_audio"]:g}.

| LoRA | trained at | shift (v/a) | steps |
|---|---|---|---|
| 4-step v0.1 | 544p, **mixed aspect** | 12 / 3 | 4 |
| 8-step v1.0 (this graph) | 544p | 12 / 3 | 8 or 4 |
| 4-step v1.0 768p | **1344x768** | **6** / 3 | 4 |

**Two things move with the LoRA, and only one of them is the shift.** Steps
always move: 16 is a base-model number. The shift moves only for the 768p
one, which was distilled at video shift 6. The other two were distilled at
12/3, which is already the default, so for them the shift node stays put.
Changing one without the other is not a partial fix.

## The resolution question

A step-distillation LoRA learns to take bigger jumps along the schedule *at
the token count it saw*. 544p and 768p are roughly a factor of two apart in
tokens, so a 544p LoRA rendering at 1344x768 is working at about twice the
sequence length it was distilled on.

**You cannot satisfy both distributions at once, and that is the real
choice here.** MiniMax H3's own canvas rule is a 768 short edge with a
1344x768 area cap: that is what `adapt_canvas` enforces and what the
reference generates. 544p is below it. So:

- Render at **1344x768** and the base model is in its trained canvas while
  the 544p LoRA is off its distillation resolution.
- Render at **544p** and the LoRA is home while the base model is outside
  the canvas family it was trained on. Nothing stops you: the width and
  height on the conditioning node are plain ints at 32-px steps, so 544p is
  typeable. `MiniMaxH3KeyframeCanvas` is the node that refuses, which is why
  this graph is t2v.

**Which one costs less is not measured here.** Do not assume the LoRA's
resolution wins just because the LoRA is the thing you added.

**The 4-step v1.0 768p is the only one with no resolution gap at this
canvas** -- it was distilled at exactly 1344x768. The trade is aspect: it saw
one, where the 4-step v0.1 saw mixed aspect ratios. Render 1:1 or 9:16 and
the 768p LoRA is the off-distribution one.

Specs from `coderef/Minimax-H3-Turbo`, README model table.
"""


_NOTE_REF_LORA = f"""\
# This graph, and what to compare it against

This is `h3_image_ref_plus_text_to_video.json` with **one thing changed**:
where that graph loads `ref2va`, this one loads `fl2va` and applies Kijai's
extracted ref LoRA on top.

Everything else is shared by construction -- same seed, same prompt, same
canvas, same 362 frames, same 16 steps, same sage and Sol-Attn settings. Open
both, point them at the same reference images, run them. Any difference you
see is the LoRA.

**Set your own reference images first.** The two `LoadImage` nodes hold
whatever placeholders this install happened to have.

## What the LoRA is

The weight difference between `fl2va` and `ref2va`, extracted at rank 256
(`Kijai/MiniMax-H3-experimental`, 2026-08-08). Not a trained adapter -- a
whole-model delta, covering all 50 blocks, both token_refiner blocks, the
patch projections, `condition_proj` and the final layer, with full-rank
deltas on every norm and bias.

At strength **1.0** it is meant to turn fl2va into ref2va. Upstream's own
description is *"completely experimental, I don't even know if it has a use
case at this point"*, so treat the shipped {REF_LORA_STRENGTH} as a
starting point, not a settled answer.

Expect close, not identical, even when it works: rank 256 truncates the real
delta, and on the int8_convrot checkpoint the merge is a dequantize / add /
requantize round trip that loses a little more. A small gap is the expected
outcome, not a broken graph.

## Turning the strength dial

Below 1.0 you are interpolating toward plain fl2va -- between first/last-frame
keyframe conditioning and reference conditioning. That is the one thing two
separate checkpoints cannot give you, and it is the reason to keep this graph
around. Expect it to be ill-behaved before it is useful: the norm and bias
deltas interpolate linearly, which is a crude stand-in for interpolating two
models.

**Strength 0.0 and ctrl-B bypass are the same thing.** ComfyUI short-circuits
a LoRA whose strengths are all zero (`nodes.py:729`) and hands back the
untouched model, so either route renders true plain fl2va. Use whichever you
prefer -- just do not treat them as two different baselines.

Neither pays what the 1.0 arm pays. Applying the LoRA to a quantized
checkpoint is a dequantize / add / requantize round trip, and the
zero-strength route skips it. So part of any 1.0-against-0.0 difference is
that round trip, not the delta. To see the round trip on its own, render
**0.01** -- visually nil, but it does not short-circuit.

## One caveat if you are comparing carefully

Sol-Attn is on here, same as the shipped graph, because the point is to
compare like with like. But its window is a *percent* band that resolves
against the model's own sigma curve, and the LoRA changes the model -- so the
two graphs can end up running a different number of sparse steps. That is a
second difference on top of the LoRA.

It does not matter for "does this look right". It does matter if you are
judging a subtle quality difference. Bypass `SolAttnPatch` in **both** graphs
to remove it.
"""


def build_ui(task: str, *, sage: bool = True, prompt: str | None = None,
             steps: int | None = None, shift: dict | None = None,
             head_chunks: int | None = None, ref_upscale: bool = True,
             variant_note: str | None = None,
             length: int = LENGTH, seed: int = SEED, preview: bool = False,
             sol: dict | None = None, sol_enabled: bool = True,
             canvas_mode: str = "fit_to_canvas", stamp: bool = False,
             unet: str | None = None, lora: tuple[str, float] | None = None,
             out_prefix: str | None = None, title: str | None = None,
             **canvas) -> dict:
    ref = task == "r2v"
    cv = dict(CANVAS, **canvas)
    prompt = prompt if prompt is not None else {
        "t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "r2v": R2V_PROMPT}[task]
    g = UIGraph()

    unet_node = g.add("UNETLoader", (-1500, 0), size=(560, 90),
                      widgets=[unet or MODELS["unet_ref2va" if ref else "unet_fl2va"],
                               "default"],
                      outputs=[_out("MODEL", "MODEL")])
    clip = g.add("CLIPLoader", (-1500, 140), size=(560, 110),
                 widgets=[MODELS["clip"], "minimax", "default"],
                 outputs=[_out("CLIP", "CLIP")])
    vvae = g.add("VAELoader", (-1500, 300), size=(560, 70),
                 widgets=[MODELS["video_vae"]], outputs=[_out("VAE", "VAE")],
                 title="Load VAE (video)")
    avae = g.add("VAELoader", (-1500, 410), size=(560, 70),
                 widgets=[MODELS["audio_vae"]], outputs=[_out("VAE", "VAE")],
                 title="Load VAE (audio)")

    model_src = unet_node
    if lora is not None:
        # Before the attention patches -- see the matching note in build_api.
        # The strength widget is the one thing this graph exists to be swept,
        # so the node gets a title that says what its arm is.
        lora_node = g.add("LoraLoaderModelOnly", (-1500, 560), size=(560, 110),
                          widgets=[lora[0], lora[1]],
                          inputs=[_in("model", "MODEL")],
                          outputs=[_out("MODEL", "MODEL")],
                          title=f"Load LoRA (ref delta, strength {lora[1]})")
        g.link(unet_node, 0, lora_node, "model", "MODEL")
        model_src = lora_node

    # See the matching note in build_api. Titled with its values because the
    # whole reason it is in the graph is that a turbo LoRA needs them changed,
    # and a node showing "ModelSamplingMiniMaxH3" and nothing else does not
    # prompt anyone to look.
    sh = shift if shift is not None else SIGMA_SHIFT
    sigma_node = g.add("MiniMaxH3SigmaShift", (-1500, 700), size=(360, 110),
                       widgets=[sh["shift_video"], sh["shift_audio"]],
                       inputs=[_in("model", "MODEL")],
                       outputs=[_out("MODEL", "MODEL")],
                       title=f"Sigma shift (video {sh['shift_video']:g}, "
                             f"audio {sh['shift_audio']:g})")
    g.link(model_src, 0, sigma_node, "model", "MODEL")
    model_src = sigma_node

    sage_node = None
    if sage:
        sage_node = g.add("MiniMaxH3SageAttention", (-880, 0), size=(360, 110),
                          widgets=[SAGE_NODE["mode"],
                                   SAGE_NODE["patch_token_refiner"],
                                   SAGE_NODE["head_chunks"] if head_chunks is None
                                   else head_chunks],
                          inputs=[_in("model", "MODEL")],
                          outputs=[_out("MODEL", "MODEL")])
        g.link(model_src, 0, sage_node, "model", "MODEL")
        model_src = sage_node

    if sol is not None:
        # After sage, never before. SolAttn composes by walking the model's
        # existing object patches and wrapping the attention forwards it
        # finds; run first it has nothing to find, and ours then overwrites
        # its patch. Both orders load and render, which is exactly why it is
        # worth pinning in a generated graph instead of leaving to hand-wiring.
        #
        # Enabled when the graph is built for it, bypassed otherwise. Bypass
        # passes MODEL straight through, so a graph carrying a disabled
        # Sol-Attn node still loads and renders without the node installed.
        # The error-prone part is the ordering above, not the toggle.
        sol_node = g.add("SolAttnPatch", (-880, 190), size=(360, 330),
                         widgets=[sol["tau"], sol["start_percent"], sol["end_percent"],
                                  sol["min_tokens"], sol["int8_qk"],
                                  sol["sink_conditioning"], sol["morton"],
                                  sol["morton_curve"], sol["int8_pv"], sol["verbose"],
                                  sol["use_tma"], sol["dense_blocks"]],
                         # tau_profile, added by Sol-Attn 0e334dc: per-block tau
                         # overriding the base value. It is declared
                         # `force_input=True`, so it is a SOCKET, not a widget.
                         # An earlier version of this file emitted it as a 13th
                         # widget value instead. That was harmless in effect --
                         # it landed after dense_blocks, and LiteGraph drops
                         # widget values past the end of the widget list -- but it
                         # meant the node carried a widget count no build of
                         # Sol-Attn has ever had, and the socket was never
                         # declared at all. Left unconnected: one tau everywhere
                         # is what we ship.
                         #
                         # The API-graph validator cannot catch this class of bug:
                         # API graphs have no widget list, so widget/socket
                         # confusion is invisible there. That is what
                         # check_workflow_schema.py is for.
                         inputs=[_in("model", "MODEL"),
                                 _in("tau_profile", "STRING", optional=True)],
                         outputs=[_out("MODEL", "MODEL")],
                         title=("Patch Sol-Attn" if sol_enabled
                                else "Patch Sol-Attn (bypassed)"))
        if not sol_enabled:
            g._node(sol_node)["mode"] = 4
        g.link(model_src, 0, sol_node, "model", "MODEL")
        model_src = sol_node

    if preview:
        # The largest practical saving on a long clip, and not a kernel
        # change: a 362-frame render is ~17 min, so seeing step 3 is what
        # lets a bad seed die at 90 s instead of costing the whole run.
        #
        # It has to be this node rather than ComfyUI's built-in preview,
        # because the launcher passes --preview-method none globally; this
        # node sidesteps that by pushing its own frame to a DOM widget on
        # itself. taeh3 is the H3 tiny decoder (latent_channels 24,
        # patch_size 2) -- without it H3 has no approx VAE at all and
        # previews degrade to latent2rgb.
        #
        # preview_frames=4 rather than 1: a still frame catches a bad
        # composition, but the failures worth aborting a 17-minute render
        # for are motion failures, and those need more than one frame.
        prev_node = g.add("ModelPreviewOverrideKJ", (-460, 190), size=(360, 200),
                          widgets=[512, 80, True, 4, 8, "taeh3.safetensors"],
                          inputs=[_in("model", "MODEL"),
                                  _in("vae", "VAE", optional=True)],
                          outputs=[_out("MODEL", "MODEL")],
                          title="Preview (taeh3)")
        g.link(model_src, 0, prev_node, "model", "MODEL")
        model_src = prev_node

    img_a = img_b = None
    if ref:
        cond_inputs = [
            _in("clip", "CLIP"), _in("vae", "VAE"), _in("audio_vae", "VAE"),
            _in("ref_images.ref_image_0", "IMAGE", optional=True, label="ref_image_0"),
            _in("ref_images.ref_image_1", "IMAGE", optional=True, label="ref_image_1"),
            _in("ref_images.ref_image_2", "IMAGE", optional=True, label="ref_image_2"),
            _in("ref_videos.ref_video_0", "IMAGE", optional=True, label="ref_video_0"),
            _in("ref_video_audios.ref_video_audio_0", "AUDIO", optional=True,
                label="ref_video_audio_0"),
            _in("ref_audios.ref_audio_0", "AUDIO", optional=True, label="ref_audio_0"),
        ]
        cond = g.add("MiniMaxH3ReferenceToVideo", (-460, 0), size=(430, 620),
                     widgets=[prompt, cv["width"], cv["height"], length, "max"],
                     inputs=cond_inputs,
                     outputs=[_out("positive", "CONDITIONING"), _out("LATENT", "LATENT")])
        img_a = g.add("LoadImage", (-880, 640), size=(290, 330),
                      widgets=[PLACEHOLDER_IMAGE_A, "image"],
                      outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
        img_b = g.add("LoadImage", (-880, 1010), size=(290, 330),
                      widgets=[PLACEHOLDER_IMAGE_B, "image"],
                      outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
        g.link(vvae, 0, cond, "vae", "VAE")
        g.link(avae, 0, cond, "audio_vae", "VAE")
        # One fit node per reference, between LoadImage and the conditioning
        # node. See the matching note in build_api: paired with
        # ref_image_size on 'max', or the stock node undoes them.
        fits = []
        for i, (src, y) in enumerate(((img_a, 640), (img_b, 1010))):
            fit = g.add("MiniMaxH3ReferenceFit", (-580, y), size=(300, 150),
                        widgets=[ref_upscale, _ref_short_edge(), False],
                        inputs=[_in("image", "IMAGE")],
                        outputs=[_out("image", "IMAGE"),
                                 _out("vision_tokens", "INT")],
                        title=f"Reference {i + 1} resolution")
            g.link(src, 0, fit, "image", "IMAGE")
            g.link(fit, 0, cond, f"ref_images.ref_image_{i}", "IMAGE")
            fits.append(fit)
    else:
        cond_inputs = [_in("clip", "CLIP"), _in("vae", "VAE"),
                       _in("first_frame", "IMAGE", optional=True),
                       _in("last_frame", "IMAGE", optional=True)]
        if task == "i2v":
            # width/height/length arrive as links from the canvas node rather
            # than as typed widgets, so they need input sockets. `length` joins
            # them so the reference's 5-15s window is enforced by the graph and
            # survives editing in the UI, not only by the generator.
            cond_inputs += [_in("width", "INT", widget=True),
                            _in("height", "INT", widget=True),
                            _in("length", "INT", widget=True)]
        cond = g.add("MiniMaxH3ImageToVideo", (-460, 0), size=(430, 560),
                     widgets=[prompt, cv["width"], cv["height"], length],
                     inputs=cond_inputs,
                     outputs=[_out("positive", "CONDITIONING"), _out("LATENT", "LATENT")])
        g.link(vvae, 0, cond, "vae", "VAE")
        if task == "i2v":
            img_a = g.add("LoadImage", (-880, 900), size=(290, 330),
                          widgets=[PLACEHOLDER_IMAGE_A, "image"],
                          outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
            # The keyframe goes through the canvas node, never straight into
            # MiniMaxH3ImageToVideo -- that node stretches the first keyframe
            # onto width/height non-uniformly (2.33x on a 3:4 still at the
            # default canvas). Fitted first, its resize becomes a no-op.
            kfc = g.add("MiniMaxH3KeyframeCanvas", (-880, 640), size=(330, 230),
                        widgets=[canvas_mode, cv["width"], cv["height"], length],
                        inputs=[_in("first_frame", "IMAGE"),
                                _in("last_frame", "IMAGE", optional=True)],
                        # `length` is last because the node appends it there --
                        # inserting it beside width/height would shift every
                        # later slot in already-saved graphs.
                        outputs=[_out("width", "INT"), _out("height", "INT"),
                                 _out("first_frame", "IMAGE"),
                                 _out("last_frame", "IMAGE"),
                                 _out("attn_cost_vs_1to1", "FLOAT"),
                                 _out("length", "INT")])
            g.link(img_a, 0, kfc, "first_frame", "IMAGE")
            g.link(kfc, 2, cond, "first_frame", "IMAGE")
            g.link(kfc, 0, cond, "width", "INT")
            g.link(kfc, 1, cond, "height", "INT")
            g.link(kfc, 5, cond, "length", "INT")
    g.link(clip, 0, cond, "clip", "CLIP")

    noise = g.add("RandomNoise", (40, 0), size=(300, 110), widgets=[seed, "randomize"],
                  outputs=[_out("NOISE", "NOISE")])
    samp = g.add("KSamplerSelect", (40, 150), size=(300, 60),
                 widgets=[SAMPLING["sampler"]], outputs=[_out("SAMPLER", "SAMPLER")])
    sched = g.add("BasicScheduler", (40, 250), size=(300, 130),
                  widgets=[SAMPLING["scheduler"],
                           steps if steps is not None else SAMPLING["steps"],
                           SAMPLING["denoise"]],
                  inputs=[_in("model", "MODEL")], outputs=[_out("SIGMAS", "SIGMAS")])
    guider = g.add("BasicGuider", (40, 420), size=(300, 70),
                   inputs=[_in("model", "MODEL"), _in("conditioning", "CONDITIONING")],
                   outputs=[_out("GUIDER", "GUIDER")])
    sampler = g.add("SamplerCustomAdvanced", (400, 0), size=(320, 150),
                    inputs=[_in("noise", "NOISE"), _in("guider", "GUIDER"),
                            _in("sampler", "SAMPLER"), _in("sigmas", "SIGMAS"),
                            _in("latent_image", "LATENT")],
                    outputs=[_out("output", "LATENT"), _out("denoised_output", "LATENT")])
    vdec = g.add("VAEDecode", (780, 0), size=(260, 60),
                 inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                 outputs=[_out("IMAGE", "IMAGE")])
    adec = g.add("VAEDecodeAudio", (780, 110), size=(260, 60),
                 inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                 outputs=[_out("AUDIO", "AUDIO")])
    # One node for mux + save. Its widgets_values is a *dict*, not the
    # positional list every other node uses -- VHS adds format-dependent
    # widgets (pix_fmt, crf, ...) after `format`, so position cannot address
    # them. Shape copied from a frontend-written graph rather than guessed.
    save = g.add("VHS_VideoCombine", (1080, 0), size=(600, 520),
                 widgets={"frame_rate": FPS, "loop_count": 0,
                          "filename_prefix": out_prefix or f"Video/h3_{task}",
                          "format": VIDEO_FORMAT, "pix_fmt": "yuv420p",
                          "crf": 19, "save_metadata": True,
                          "trim_to_audio": False,
                          "pingpong": False, "save_output": True},
                 inputs=[_in("images", "IMAGE"),
                         _in("audio", "AUDIO", optional=True),
                         _in("meta_batch", "VHS_BatchManager", optional=True),
                         _in("vae", "VAE", optional=True)],
                 outputs=[_out("Filenames", "VHS_FILENAMES")])

    # See the matching note in build_api: last in the chain, asserting the
    # composition rather than any single node's intent.
    assert_node = g.add("SageChainAssert", (-480, 0), size=(360, 130),
                        widgets=[True, True, True, False],
                        inputs=[_in("model", "MODEL")],
                        outputs=[_out("model", "MODEL")],
                        title="Assert the attention chain composed")
    g.link(model_src, 0, assert_node, "model", "MODEL")
    model_src = assert_node

    g.link(model_src, 0, sched, "model", "MODEL")
    g.link(model_src, 0, guider, "model", "MODEL")
    # Pass-through, between conditioning and the sampler, so the report is
    # about the graph that is actually going to run.
    pre = g.add("MiniMaxH3Preflight", (-60, 640), size=(420, 260),
                inputs=[_in("conditioning", "CONDITIONING"),
                        _in("samples", "LATENT")],
                outputs=[_out("conditioning", "CONDITIONING"),
                         _out("samples", "LATENT"),
                         _out("sequence_length", "INT"),
                         _out("report", "STRING")],
                title="Preflight: what this render costs")
    g.link(cond, 0, pre, "conditioning", "CONDITIONING")
    g.link(cond, 1, pre, "samples", "LATENT")
    g.link(pre, 0, guider, "conditioning", "CONDITIONING")
    g.link(pre, 1, sampler, "latent_image", "LATENT")
    g.link(noise, 0, sampler, "noise", "NOISE")
    g.link(guider, 0, sampler, "guider", "GUIDER")
    g.link(samp, 0, sampler, "sampler", "SAMPLER")
    g.link(sched, 0, sampler, "sigmas", "SIGMAS")
    latent_src, latent_slot = sampler, 0
    if stamp:
        # Bench only. Inline between the sampler and both decoders so it has a
        # real data dependency on the sampler -- ComfyUI orders by dependency,
        # not graph position, and without that edge it can legally run BEFORE
        # sampling and record pre-render state. SIGMAS is what makes n_sparse
        # computable; nothing else exposes it.
        stampn = g.add("MiniMaxH3ProvenanceStamp", (780, 240), size=(330, 130),
                       widgets=[f"bench {task}"],
                       inputs=[_in("latent", "LATENT"), _in("model", "MODEL"),
                               _in("sigmas", "SIGMAS", optional=True)],
                       outputs=[_out("latent", "LATENT")])
        g.link(sampler, 0, stampn, "latent", "LATENT")
        g.link(model_src, 0, stampn, "model", "MODEL")
        g.link(sched, 0, stampn, "sigmas", "SIGMAS")
        latent_src, latent_slot = stampn, 0
    g.link(latent_src, latent_slot, vdec, "samples", "LATENT")
    g.link(vvae, 0, vdec, "vae", "VAE")
    g.link(latent_src, latent_slot, adec, "samples", "LATENT")
    g.link(avae, 0, adec, "vae", "VAE")
    g.link(vdec, 0, save, "images", "IMAGE")
    g.link(adec, 0, save, "audio", "AUDIO")

    # Guidance in the graph rather than in a doc nobody opens next to it.
    # MarkdownNote is in _UI_ONLY, so these never reach the API form and
    # cannot desync it.
    g.add("MarkdownNote", (-2180, 0), size=(620, 620), widgets=[_NOTE_GEOMETRY],
          title="Canvas + length: what is actually selectable")
    g.add("MarkdownNote", (-2180, 660), size=(620, 560), widgets=[_NOTE_NODES],
          title="Which nodes, and the order that matters")
    g.add("MarkdownNote", (-2860, 0), size=(640, 900), widgets=[_NOTE_SIZING],
          title="Resolution, references, and reading the preflight")
    if variant_note is not None:
        g.add("MarkdownNote", (-2180, 1280), size=(620, 760),
              widgets=[variant_note], title="What this graph is probing")
    elif lora is not None:
        g.add("MarkdownNote", (-2180, 1280), size=(620, 620),
              widgets=[_NOTE_REF_LORA], title="What this graph is probing")

    return g.dump(title or f"h3-{task}-sage")


# --------------------------------------------------------------------------
# Static validation against /object_info
# --------------------------------------------------------------------------

def load_object_info(source: str) -> dict:
    if source.startswith("http"):
        with urllib.request.urlopen(source.rstrip("/") + "/object_info", timeout=60) as r:
            return json.loads(r.read())
    return json.loads(Path(source).read_text())


def _combo_options(spec):
    """Combo option lists come in two shapes across ComfyUI node versions."""
    t = spec[0]
    if isinstance(t, list):
        return t
    if t == "COMBO":
        return (spec[1] or {}).get("options")
    return None


def validate_api(graph: dict, oi: dict, label: str) -> list[str]:
    errs = []

    def e(msg):
        errs.append(f"{label}: {msg}")

    for nid, node in graph.items():
        ct = node["class_type"]
        if ct not in oi:
            e(f"node {nid}: unknown class_type {ct!r}")
            continue
        spec = oi[ct]["input"]
        req = spec.get("required") or {}
        opt = spec.get("optional") or {}
        known = dict(req) | dict(opt)
        # Autogrow inputs are declared once but addressed as
        # "<input>.<prefix><i>"; expand the legal names.
        for name, s in list(known.items()):
            if s[0] != "COMFY_AUTOGROW_V3":
                continue
            tpl = (s[1] or {}).get("template") or {}
            inner = tpl.get("input") or {}
            inner_spec = next(iter((inner.get("required") or inner.get("optional") or {}).values()), None)
            for i in range(tpl.get("max", 0)):
                known[f"{name}.{tpl['prefix']}{i}"] = inner_spec

        # Format-dependent widgets. VHS_VideoCombine declares `format` as a
        # combo whose spec carries a per-format widget list (pix_fmt, crf,
        # save_metadata, ...), and reads those from **kwargs at run time --
        # `apply_format_widgets` warns and substitutes a default for any it
        # does not find. They are real inputs that /object_info does not
        # declare as inputs, so a plain known-name check calls every one of
        # them unknown. Third false-positive class this validator has had, all
        # the same shape: a node whose input set is not fully static.
        for spec in list(req.values()) + list(opt.values()):
            meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            for widgets in (meta.get("formats") or {}).values():
                for w in widgets:
                    if isinstance(w, list) and w and isinstance(w[0], str):
                        known.setdefault(w[0], None)

        given = node["inputs"]
        for name in req:
            if req[name][0] == "COMFY_AUTOGROW_V3":
                continue
            if name not in given:
                e(f"node {nid} ({ct}): missing required input {name!r}")
        for name, val in given.items():
            if name not in known:
                e(f"node {nid} ({ct}): unknown input {name!r}")
                continue
            s = known[name]
            if isinstance(val, list):  # a link
                src, slot = val
                if src not in graph:
                    e(f"node {nid} ({ct}).{name}: links to missing node {src!r}")
                    continue
                souts = oi[graph[src]["class_type"]]["output"]
                if slot >= len(souts):
                    e(f"node {nid} ({ct}).{name}: output slot {slot} out of range "
                      f"on node {src} ({graph[src]['class_type']})")
                    continue
                got = souts[slot]
                want = s[0] if s else None
                got_name = got if isinstance(got, str) else "COMBO"
                if want and isinstance(want, str) and want not in ("*",) and got_name != want:
                    e(f"node {nid} ({ct}).{name}: type {got_name} from node {src} "
                      f"does not match {want}")
                continue
            if s is None:
                continue
            opts = _combo_options(s)
            if opts is not None and val not in opts:
                e(f"node {nid} ({ct}).{name}: {val!r} is not an available option")
                continue
            meta = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            if s[0] in ("INT", "FLOAT") and isinstance(val, (int, float)):
                if "min" in meta and val < meta["min"]:
                    e(f"node {nid} ({ct}).{name}: {val} below min {meta['min']}")
                if "max" in meta and val > meta["max"]:
                    e(f"node {nid} ({ct}).{name}: {val} above max {meta['max']}")

        # H3-specific: frame count is snapped up to 17k+5 by the node, so an
        # off-grid `length` silently renders a different duration than asked.
        if ct in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo",
                  "EmptyMiniMaxH3LatentAV"):
            ln = given.get("length")
            if isinstance(ln, int) and ln % 17 != 5:
                e(f"node {nid} ({ct}): length {ln} is off the 17k+5 grid; "
                  f"the node will snap it up to {ln + (5 - ln % 17) % 17}")

    # The mistake this whole file exists to prevent.
    consumers = [(nid, n) for nid, n in graph.items()
                 if n["class_type"] in ("BasicScheduler", "BasicGuider")]
    srcs = {tuple(n["inputs"]["model"]) for _, n in consumers
            if isinstance(n["inputs"].get("model"), list)}
    if len(srcs) > 1:
        e(f"BasicScheduler and BasicGuider read MODEL from different sources {srcs}; "
          f"one of them is bypassing a model patch")
    return errs


def validate_ui(wf: dict, oi: dict, label: str) -> list[str]:
    """Self-consistency only. No server validates a UI graph, so this checks
    what the frontend would choke on: dangling links and slot mismatches."""
    errs = []

    def e(msg):
        errs.append(f"{label}: {msg}")

    by_id = {n["id"]: n for n in wf["nodes"]}
    declared = {l[0] for l in wf["links"]}
    for lid, src, ss, dst, ds, t in wf["links"]:
        if src not in by_id or dst not in by_id:
            e(f"link {lid}: endpoint missing")
            continue
        s, d = by_id[src], by_id[dst]
        if ss >= len(s["outputs"]):
            e(f"link {lid}: output slot {ss} out of range on {s['type']}")
        elif lid not in (s["outputs"][ss]["links"] or []):
            e(f"link {lid}: not listed on {s['type']} output {ss}")
        if ds >= len(d["inputs"]):
            e(f"link {lid}: input slot {ds} out of range on {d['type']}")
        elif d["inputs"][ds].get("link") != lid:
            e(f"link {lid}: not recorded on {d['type']} input {ds}")
    for n in wf["nodes"]:
        # Frontend-only nodes have no backend class, so they are absent from
        # /object_info by design. Rejecting them would be the validator being
        # confidently wrong rather than the graph being broken.
        if n["type"] in _FRONTEND_ONLY:
            continue
        if n["type"] not in oi:
            e(f"node {n['id']}: unknown type {n['type']!r}")
            continue
        for i, inp in enumerate(n["inputs"]):
            if inp.get("link") is not None and inp["link"] not in declared:
                e(f"node {n['id']} ({n['type']}) input {inp['name']}: dangling link")
            if inp.get("link") is None and inp.get("shape") != 7 and "widget" not in inp:
                e(f"node {n['id']} ({n['type']}): required input {inp['name']} unconnected")
        # widgets_values must cover every widget the node declares, in order,
        # including any that have been converted to inputs.
        spec = oi[n["type"]]["input"]
        # `force_input=True` turns a scalar input into a socket, so it owns no
        # widget value. Missing that is how tau_profile got emitted as a 13th
        # widget on a 12-widget node: this check allows a surplus (see below),
        # so a spurious extra value was invisible from here.
        def _is_widget(v):
            opts = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            if opts.get("forceInput"):
                return False
            return isinstance(v[0], list) or v[0] in (
                "INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3")

        widget_names = [k for k, v in ((spec.get("required") or {}) | (spec.get("optional") or {})).items()
                        if _is_widget(v)]
        got = len(n.get("widgets_values") or [])
        # RandomNoise / LoadImage carry an extra frontend-only widget
        # (control_after_generate, the upload button) that /object_info does
        # not report, so allow a surplus but never a shortfall.
        if got < len(widget_names):
            e(f"node {n['id']} ({n['type']}): {got} widget values for "
              f"{len(widget_names)} widgets {widget_names}")
    return errs


# --------------------------------------------------------------------------

# Nodes that are browser affordances rather than computation, so their
# absence from the API form is intentional and not drift.
#
# ModelPreviewOverrideKJ is the non-obvious one: it patches the model, but
# only to decode intermediate latents through taeh3 for display. Headless
# has nowhere to show them, and those decodes cost time that would land in
# any timing run as an unattributed confound. It belongs in the graph you
# watch and nowhere near the graph you measure.
_UI_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode",
            "ModelPreviewOverrideKJ"}

# Rendered entirely by the frontend, so they have no entry in /object_info.
# Subset of _UI_ONLY: ModelPreviewOverrideKJ is a real backend node that we
# exclude from the API form by choice, not by necessity.
_FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}


def _ui_settings(wf):
    """{class_type: widgets} for a UI graph, ignoring bypassed nodes."""
    return {n["type"]: n.get("widgets_values")
            for n in wf["nodes"]
            if n["type"] not in _UI_ONLY and n.get("mode", 0) == 0}


def _api_settings(wf):
    """{class_type: non-link inputs} for an API graph."""
    return {n["class_type"]: {k: v for k, v in n["inputs"].items()
                              if not isinstance(v, list)}
            for n in wf.values()}


def cross_check(written):
    """Report where a task's UI and API graphs disagree.

    Compares which nodes are present and, for the ones carrying settings we
    pin, that the pinned values match. Widget *order* differs between the two
    formats by design (UI is positional, API is keyed), so this checks the
    node set plus SolAttnPatch and MiniMaxH3SageAttention values explicitly
    rather than trying to align every widget by index.
    """
    by_task = {}
    for task, fmt, p, wf in written:
        by_task.setdefault(task, {})[fmt] = (p.name, wf)

    errs = []
    for task, forms in sorted(by_task.items()):
        if len(forms) < 2:
            continue
        ui_name, ui = forms["ui"]
        api_name, api = forms["api"]
        ui_s, api_s = _ui_settings(ui), _api_settings(api)

        only_ui = set(ui_s) - set(api_s) - _UI_ONLY
        only_api = set(api_s) - set(ui_s)
        for n in sorted(only_ui):
            errs.append(f"{task}: {n} in {ui_name} but not {api_name}")
        for n in sorted(only_api):
            errs.append(f"{task}: {n} in {api_name} but not {ui_name}")

        # Nodes whose values are compared, not just their presence. UI widgets
        # are positional in schema order; API inputs are keyed, so each entry
        # is the schema order of the widgets we care about.
        #
        # SolAttnPatch is here because its settings have actually drifted.
        # UNETLoader and LoraLoaderModelOnly joined it the moment `unet` and
        # `lora` became free builder parameters: before that the checkpoint
        # was derived from `task` inside both builders and the two formats
        # could not disagree about it, and now they can. Which checkpoint a
        # graph loads is exactly the class of difference this function exists
        # to catch, and the node-set check above cannot see it -- both formats
        # carry a UNETLoader either way.
        for cls, order in (
            ("SolAttnPatch",
             ["tau", "start_percent", "end_percent", "min_tokens",
              "int8_qk", "sink_conditioning", "morton", "morton_curve",
              "int8_pv", "verbose", "use_tma", "dense_blocks"]),
            ("UNETLoader", ["unet_name"]),
            ("LoraLoaderModelOnly", ["lora_name", "strength_model"]),
            # The shifts are here for the same reason as the checkpoint: they
            # are a free builder value that the two formats can now disagree
            # about, and a graph sampling off the wrong schedule renders
            # cleanly rather than failing.
            ("MiniMaxH3SigmaShift", ["shift_video", "shift_audio"]),
        ):
            if cls not in ui_s or cls not in api_s:
                continue
            widgets = ui_s[cls] or []
            for i, key in enumerate(order):
                if i >= len(widgets) or key not in api_s[cls]:
                    continue
                if widgets[i] != api_s[cls][key]:
                    errs.append(
                        f"{task}: {cls}.{key} is {widgets[i]!r} in "
                        f"{ui_name} but {api_s[cls][key]!r} in {api_name}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object-info", default="http://127.0.0.1:8188",
                    help="running ComfyUI base URL, or a path to a saved object_info.json")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = []

    # The ones you actually open in ComfyUI. Named for what they do, not for
    # the task abbreviation the code uses internally. All carry the taeh3
    # preview, which is what lets a bad seed die at ~90s instead of costing a
    # full render -- worth more than any kernel knob when render time is the
    # objective.
    # `label` keys the UI/API cross-check and has to be unique; `task` is what
    # the builder dispatches on. They were the same string until the ref-LoRA
    # graph arrived, which is a second r2v graph with a different model source.
    #
    # `extra` is the whole difference between the shipped ref graph and its
    # ref-LoRA sibling. Keeping it to one dict, on one line, next to the graph
    # it modifies is the point: the two are meant to be compared, so anything
    # that differs between them has to be visible in one place. Everything not
    # in `extra` -- seed, prompt, canvas, length, sampler, sage, Sol -- is
    # shared by construction and cannot drift apart.
    GRAPHS: tuple[tuple[str, str, str, str | None, dict[str, Any], str], ...] = (
        ("h3_text_to_video.json", "t2v", "t2v", LONG_T2V_PROMPT, {},
         "text -> video + audio"),
        ("h3_image_ref_plus_text_to_video.json", "r2v", "r2v", None, {},
         "reference image(s) + text -> video + audio"),
        ("h3_first_frame_to_video.json", "i2v", "i2v", None, {},
         "first frame + text -> video + audio (via MiniMaxH3KeyframeCanvas)"),
        ("h3_image_ref_plus_text_to_video_ref_lora.json", "r2v-reflora", "r2v", None,
         dict(unet=MODELS["unet_fl2va"], lora=(REF_LORA, REF_LORA_STRENGTH),
              out_prefix="video/h3_r2v_fl2va_ref_lora"),
         "same, but fl2va + the extracted ref LoRA instead of ref2va"),
        # t2v deliberately: the note explains that matching the LoRA's 544p
        # means leaving H3's own canvas rule, and MiniMaxH3KeyframeCanvas is
        # the node that refuses to, so an i2v turbo graph could not show the
        # choice it is describing.
        ("h3_text_to_video_turbo.json", "t2v-turbo", "t2v", LONG_T2V_PROMPT,
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT, variant_note=_NOTE_TURBO,
              out_prefix="video/h3_t2v_turbo_8step"),
         "text -> video + audio, via the 8-step turbo LoRA"),

        # --- probes: pairs, one variable, run against the named twin ---
        ("h3_probe_reference_upscale.json", "r2v-noupscale", "r2v", None,
         dict(ref_upscale=False, out_prefix="video/h3_probe_ref_noupscale",
              variant_note=_probe_note(
                  "does upscaling a small reference buy anything",
                  "h3_image_ref_plus_text_to_video.json",
                  "`allow_upscale` is OFF on both Reference Resolution nodes, "
                  "so references arrive at ComfyUI's own sizing instead of the "
                  "released pipeline's 2048 short edge.",
                  "Preflight's `references` line and percentage, then the "
                  "identity of the referenced subjects in the output. The "
                  "shipped graph spends roughly 13,900 more vision tokens on "
                  "the same two images.",
                  "Fewer tokens here, and a shorter sequence. Whether identity "
                  "is worse is the open question -- upscaling adds tokens, not "
                  "detail, and nobody has measured whether the checkpoint uses "
                  "them on an already-small source.")),
         "same references, without the reference pipeline's upscale"),

        ("h3_probe_square_canvas.json", "t2v-1to1", "t2v", LONG_T2V_PROMPT,
         dict(width=768, height=768,
              out_prefix="video/h3_probe_square",
              variant_note=_probe_note(
                  "what an aspect ratio actually costs",
                  "h3_text_to_video.json",
                  "768x768 instead of 1344x768. Both are inside the trained "
                  "family; only the shape changed.",
                  "Preflight's sequence length on each, and render time. "
                  "Attention is O(S^2) and dominates the step.",
                  "About a third of the attention cost at the same frame "
                  "count, which is the largest single lever in this pipeline "
                  "-- larger than any kernel or sparsity setting.")),
         "the same prompt on the cheapest legal canvas"),

        ("h3_probe_head_chunks.json", "t2v-chunk4", "t2v", LONG_T2V_PROMPT,
         dict(head_chunks=4, out_prefix="video/h3_probe_chunk4",
              variant_note=_probe_note(
                  "trading launches for VRAM headroom",
                  "h3_text_to_video.json",
                  "`head_chunks` 4 on the SageAttention node instead of 1.",
                  "Peak VRAM, and wall clock. Nothing about the output should "
                  "change: chunking splits the heads, it does not alter the "
                  "arithmetic.",
                  "Peak attention drops from 2862 MiB to 2645 at the default "
                  "canvas, because chunking rules out the v clone that only "
                  "pays unchunked. It costs 4 kernel launches per call, "
                  "measured at a ~2.6% wall-clock ceiling on a 24 GB 4090. "
                  "Take it to fit a render that otherwise will not fit, not "
                  "for speed.")),
         "the same render with the heads in 4 groups"),
    )

    for fname, label, task, prompt, extra, note in GRAPHS:
        wf = build_ui(task, sage=True, length=LONG_LENGTH, preview=True,
                      sol=SOL_RECOMMENDED, sol_enabled=True, prompt=prompt,
                      title=f"h3-{label}-sage", **extra)
        p = out / fname
        p.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
        written.append((label, "ui", p, wf))
        print(f"  {p.name}: {note}")

    # API-format copies of the same graphs, for driving a render over /prompt
    # without a browser. Same builder inputs, so they cannot describe a
    # different configuration than the set above.
    for fname, label, task, prompt, extra, _note in GRAPHS:
        # variant_note is guidance drawn on the canvas; the API form has no
        # node that carries it and _UI_ONLY would flag it as a desync.
        api_extra = {k: v for k, v in extra.items() if k != "variant_note"}
        wf = build_api(task, sage=True, length=LONG_LENGTH,
                       sol=SOL_RECOMMENDED, prompt=prompt, **api_extra)
        p = out / fname.replace(".json", "_api.json")
        p.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
        written.append((label, "api", p, wf))

    # Bench copies carrying MiniMaxH3ProvenanceStamp. Deliberately NOT the
    # shipped graphs: the stamp reads another pack's closure internals, so it
    # breaks when that pack changes, and a bench is where breakage is cheap.
    # API-only, so cross_check skips them (it needs both formats to compare).
    bench = out / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    for fname, task, prompt in (
        ("h3_text_to_video_stamped_api.json", "t2v", LONG_T2V_PROMPT),
        ("h3_image_ref_plus_text_to_video_stamped_api.json", "r2v", None),
        ("h3_first_frame_to_video_stamped_api.json", "i2v", None),
    ):
        wf = build_api(task, sage=True, length=LONG_LENGTH,
                       sol=SOL_RECOMMENDED, prompt=prompt, stamp=True)
        p = bench / fname
        p.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
        written.append((f"{task}-stamped", "api", p, wf))

    for _t, _f, p, _w in written:
        print(f"wrote {p.name}")

    # Cross-check the two formats of each task describe the same graph. The
    # per-format validators below only prove each is well-formed against
    # object_info; nothing there would notice the UI graph carrying a
    # SolAttnPatch the API graph lacks, which is exactly the state this file
    # was in before 2026-08-06.
    drift = cross_check(written)
    if drift:
        print("\nUI/API DRIFT:")
        for x in drift:
            print("  " + x)
        return 1
    print("UI/API cross-check: same node set and settings")

    if args.no_validate:
        return 0
    oi = load_object_info(args.object_info)
    errs = []
    for task, fmt, p, wf in written:
        errs += (validate_api if fmt == "api" else validate_ui)(wf, oi, p.name)
    if errs:
        print("\nvalidation FAILED:")
        for x in errs:
            print("  " + x)
        return 1
    print(f"\nvalidated {len(written)} graphs against object_info: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
