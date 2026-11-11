"""fl2va conditioning with the geometry seams measured here closed at one node.

Replaces `MiniMaxH3ImageToVideo` for t2va and the three keyframe signatures.
It does not touch ref2va; `MiniMaxH3ReferenceToVideo` is still core's.

**Why a node rather than compensating nodes around core's.** Every fix this
repo shipped before now worked by handing core a different input.
`MiniMaxH3KeyframeCanvas` pre-fit the keyframe, which left geometry owned by
two nodes in series. This node closes that ownership seam.

What it closes, each measured in this repo on 2026-08-21:

1. **Last-frame-only geometry.** Core picks stretch-versus-crop from which
   socket was wired, so a lone `last_frame` is cover-cropped into a canvas
   chosen elsewhere and loses whatever falls outside it. sglang selects the
   geometry anchor by semantic frame index (`prequeue.py:97-107`) and derives
   the canvas from it, so a lone last frame keeps its whole picture.
   `resolve_keyframe_geometry` implements that and this node is the only
   surface that can reach it -- `MiniMaxH3KeyframeCanvas` requires a
   `first_frame`.

2. **One geometry owner.** Canvas resolution and conditioning happen here, not
   in a node upstream feeding sizes into a node that resizes again.

3. **An empty prompt is refused.** Core pads to token 151643
   (`comfy_extras/nodes_minimax_h3.py:186-187`) and renders; LightX and sglang
   refuse. Refusing is cheap and the alternative conditions on a pad token.

**What it deliberately does NOT close, and why.**

*The Qwen processor's pixel bounds.* ComfyUI leaves them on the shared Qwen2-VL
helper's signature defaults, 3,136/12,845,056, where the release declares
65,536/16,777,216 for images. Two reasons this node does not fix it. It is
**inert here**: `bench/measure_qwen_bounds_bite.py` shows no legal H3 canvas
trips either bound, because every one is a multiple of the helper's rounding
factor and sits inside both -- it bites only on reference images past 3.0625:1,
which is the other node's path. And reaching it would mean patching
`cond_stage_model`, which `CLIP.clone()` shares **by reference**, so the patch
would leak into every other graph in the process. A wrong constant that
provably never fires is better than a correct constant installed unsafely.

*The VAE posterior.* Core takes the mean; the release samples under a pinned
seed 42. Closing it needs the log-variance, which `VAE.encode` never returns,
so a node would have to reach into `first_stage_model` -- and that skips
`load_models_gpu`, giving a CPU-resident model whenever the VAE is offloaded.
`vae_precision.py` is the safe shape for changing VAE behaviour: patch the VAE
object and let `VAE.encode` run normally. The posterior belongs in a node of
that shape, not this one.

Neither omission changes a shipped render today. Both are recorded in
`docs/research/official_weights_metadata.md`.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io
import node_helpers
from comfy_extras.nodes_minimax_h3 import _av_latent, _empty_av_latent

from .h3_rules import normalize_prompt
from .h3_spans import attach_span_masks
from .keyframe_canvas import resolve_keyframe_geometry
logger = logging.getLogger(__name__)


class MiniMaxH3Conditioning(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Conditioning",
            display_name="MiniMax H3 Conditioning",
            category="MiniMaxH3",
            description=(
                "fl2va conditioning that owns its own canvas and accepts a "
                "last frame on its own. Special-token ownership is native to "
                "ComfyUI's MiniMaxH3Tokenizer. Replaces "
                "MiniMaxH3ImageToVideo on the t2v and keyframe paths; ref2va "
                "uses its separate conditioning path."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="Video VAE. Required even with no "
                                            "keyframe, matching core."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input(
                    "last_frame", optional=True,
                    tooltip="Wire this alone for the last-frame-only "
                            "signature. Unlike core, a lone last frame anchors "
                            "the canvas instead of being cropped into one."),
                io.Combo.Input(
                    "canvas", options=["from_keyframe", "explicit"],
                    default="from_keyframe",
                    tooltip="from_keyframe derives the canvas from the anchor "
                            "keyframe, as the release does, and width/height "
                            "are ignored. explicit uses width/height and "
                            "cover-crops the anchor to fit."),
                io.Boolean.Input(
                    "single_frame", default=False, optional=True,
                    tooltip="On: the latent is one frame, a still image, and "
                            "length is not used. Text to image only: a first "
                            "or last frame cannot be wired with it, because "
                            "with one frame the keyframe would be the whole "
                            "picture."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"),
                     io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width=1344, height=768, length=124,
                first_frame=None, last_frame=None,
                canvas="from_keyframe", single_frame=False) -> io.NodeOutput:
        # Refused rather than padded. Core emits token 151643 for an empty
        # entry list and renders against it; both vendor runtimes reject the
        # request. Checked before any model work so the failure is cheap.
        if not prompt or not prompt.strip():
            raise ValueError(
                "MiniMaxH3Conditioning needs a prompt. ComfyUI's core node "
                "conditions on a pad token when the prompt is empty and "
                "renders anyway; LightX2V and sglang both refuse the request, "
                "and so does this."
            )

        if single_frame and (first_frame is not None or last_frame is not None):
            raise ValueError(
                "MiniMaxH3Conditioning: single_frame is text to image. With one "
                "frame a first or last frame would be the whole picture; unwire "
                "the keyframe, or turn single_frame off."
            )

        images, keyframes = [], []
        # Bound up front so the two blocks below cannot drift apart: they are
        # guarded by the same condition today, and a reader changing one has no
        # reason to notice the other depends on it.
        first_out = last_out = None
        if first_frame is not None or last_frame is not None:
            # One call owns the canvas, the anchor choice and both resizes.
            # `explicit` maps to the existing fit_to_canvas mode, which
            # cover-crops the anchor because the user chose the geometry.
            mode = "match_keyframe" if canvas == "from_keyframe" else "fit_to_canvas"
            width, height, first_out, last_out, _, length = (
                resolve_keyframe_geometry(
                    first_frame=first_frame, last_frame=last_frame,
                    mode=mode, width=width, height=height, length=length))

        if single_frame:
            # One frame, with the two audio steps the shared clock gives it.
            # Core's own empty latent cannot be asked for this: its
            # `temporal_shape` floors every length at five frames.
            latent, frame_count = _av_latent(width, height, 1, 2), 1
        else:
            latent, frame_count = _empty_av_latent(width, height, length)

        if first_out is not None or last_out is not None:
            # Presentation order is first then last, and only the frames
            # actually wired are pinned. `resolve_keyframe_geometry` fills both
            # slots with the anchor when one is missing, which is right for a
            # geometry answer and wrong for a row map -- pinning an absent
            # frame would anchor the target's final frame on a picture nobody
            # supplied.
            if first_frame is not None:
                images.append(first_out)
                keyframes.append({"resolved_frame_index": 0, "image": first_out})
            if last_frame is not None:
                images.append(last_out)
                keyframes.append({"resolved_frame_index": frame_count - 1,
                                  "image": last_out})

        tokens = clip.tokenize(normalize_prompt(prompt), images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        attach_span_masks(cond, clip, tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(
                cond, {"minimax_keyframes": keyframes})

        logger.info(
            "[h3] conditioning: %dx%d, %d frames, %s, %s keyframe(s)",
            width, height, frame_count, canvas, len(keyframes),
        )
        return io.NodeOutput(cond, latent)
