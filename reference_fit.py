"""Fit a reference image the way the reference pipeline does, including up.

`MiniMaxH3ReferenceToVideo` sizes reference images with

    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))     # nodes_minimax_h3.py

and the reference pipeline with

    scale = reference_image_short_edge / min(width, height)  # diffusers
                                    # modular_pipelines/minimax_h3/before_encoder.py

Same constant (2048), same round-to-32, one difference: the `min(1.0, ...)`.
ComfyUI never upscales. So a reference smaller than 2048 on its short edge
reaches the DiT at its original size, where the reference would have enlarged
it -- and reference tokens are latent rows, so a 512px reference contributes
16x fewer of them than the released pipeline gives it. Identity fidelity is
the whole job of a reference image, and it is being conditioned on a fraction
of the rows the model was built to see.

This custom node handles that still-open native ComfyUI gap for graphs that wire
it, the same way `MiniMaxH3KeyframeCanvas` handles the canvas divergence: it
does the resize itself, so the stock node's own scale becomes
`min(1.0, 2048/2048) = 1.0` and its resize is a no-op. It composes rather than
replacing core, and does not make the native gap closed.

**On the typed reference path this node is no longer needed.**
`MiniMaxH3AppendRefImage` carries `short_edge` and `allow_upscale` itself and
`MiniMaxH3ReferenceConditioning` performs one resize with the canvas in scope,
so the shipped graphs wire the loader straight to the append. This node stays
registered, and stays correct, because saved graphs outside this repo wire it
and a node that vanishes is a graph that stops loading. Its fit is idempotent
with the append's: a reference already at a 2048 short edge resolves the
append's `max` scale to 1.0.

**Two inputs on it are inert.** `lift_downstream_clamp` and
`keep_towers_matched` are retained so saved-graph widget positions stay valid,
and they do nothing. The first armed an override on `MiniMaxH3ReferenceToVideo`,
which no graph in this repo has ever wired, and which bound the constant by
value at import time in both consumers -- so the arm was invisible to them even
where the node was present. The second applied a Qwen ceiling this node cannot
know: it read Comfy's native `process_qwen2vl_images` default by introspection
and applied it universally, which is right for a native BF16 graph and wrong by
a large factor under the AWQ adapter. That decision moved to
`MiniMaxH3ReferenceConditioning.image_policy`, where the CLIP is in scope and
the answer is knowable. `qwen_max_pixels` and `clamp_to_qwen_ceiling` below stay
exported: `reference_video_fit.py` is a reporter for native-core paths, where
Comfy's default IS the right ceiling.

**Upscaling is not free and the node says so.** Reference rows ride through
every sampling step, so quadrupling the short edge multiplies those rows by
16 and the stock node's own tooltip already warns that `max` "can be several
times slower". The `latent_rows` output is there to make that visible before
you queue a render rather than after.

It also carries the reference's 1:4..4:1 check on reference images
(`before_encoder.py`), which ComfyUI does not have -- the same limit
`h3_rules` holds for the canvas.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE,
                                           REF_IMAGE_SHORT_EDGE, _resize)

try:
    from .h3_rules import aspect_in_range, describe_aspect_range
    from .reference_geometry import fit_reference_image, latent_rows
except ImportError:  # pragma: no cover - direct-module import for the checks
    from h3_rules import aspect_in_range, describe_aspect_range  # type: ignore[no-redef]
    from reference_geometry import (fit_reference_image,  # type: ignore[no-redef]
                                    latent_rows)

logger = logging.getLogger(__name__)


def qwen_max_pixels() -> int:
    """ComfyUI's own vision ceiling, read from the helper rather than copied.

    `process_qwen2vl_images` carries min/max pixels as signature defaults and
    `comfy/text_encoders/qwen3vl.py` calls it without overriding them, so the
    default IS the ceiling every H3 reference is subject to. Read by
    introspection so this node tracks ComfyUI instead of holding a second copy
    that goes stale silently -- the release ships a different value again, and
    `vendor_config.image_pixel_bounds()` is the authority for THAT one. These
    are two different numbers and conflating them is the whole defect below.
    """
    import inspect
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    param = inspect.signature(process_qwen2vl_images).parameters.get("max_pixels")
    if param is None or not isinstance(param.default, int):
        raise RuntimeError(
            "comfy.text_encoders.qwen_vl.process_qwen2vl_images no longer "
            "carries an integer `max_pixels` default, so this node cannot "
            "tell where Qwen will resize. Its layout changed; fix this rather "
            "than guessing a ceiling.")
    return int(param.default)


def clamp_to_qwen_ceiling(tw: int, th: int, ceiling: int):
    """Shrink (tw, th) under Qwen's ceiling, or return it unchanged.

    **What this is for.** Core hands ONE tensor to both towers: it VAE-encodes
    the reference and stashes the same object for the conditioner. Qwen then
    applies its own smart-resize inside the text encoder, after the VAE is
    already done. Above the ceiling that resize fires for Qwen alone, so the
    DiT receives latent rows at one resolution and hidden states at another for
    the same reference, and nothing says so.

    Pre-applying the shrink here puts both towers back on one size, because the
    tensor core encodes is the one this returns.

    Floors both dimensions to 32 rather than rounding. Rounding up could land
    back above the ceiling, which would leave the split in place while
    reporting that it had been closed. A multiple of 32 also makes Qwen's own
    round-to-32 the identity, so "under the ceiling" is enough to guarantee it
    will not resize.
    """
    if tw * th <= ceiling:
        return tw, th, None
    import math as _math
    scale = _math.sqrt(ceiling / (tw * th))
    nw = max(CANVAS_MULTIPLE, int(tw * scale) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
    nh = max(CANVAS_MULTIPLE, int(th * scale) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
    return nw, nh, (tw, th)


class MiniMaxH3ReferenceFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceFit",
            display_name="MiniMax H3 Reference Resolution",
            category="model/conditioning/minimax",
            description=(
                "Scales a reference image to MiniMax H3's 2048 short edge, "
                "upscaling when it is smaller -- which the reference pipeline "
                "does and ComfyUI does not. Wire it between Load Image and "
                "MiniMax H3 Reference to Video, with that node's "
                "ref_image_size on 'max', and its own resize becomes a no-op. "
                "Reference rows ride every sampling step, so read latent_rows "
                "before queueing."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input(
                    "allow_upscale", default=True, tooltip=(
                        "Scale the image until its shorter side reaches "
                        "short_edge, enlarging it if it is smaller. "
                        "On: matches the released pipeline, which upscales "
                        "unconditionally. A 1280x720 reference becomes "
                        "3648x2048, going from 880 to 7296 vision tokens. "
                        "Off: matches ComfyUI, which only ever shrinks. "
                        "The two differ by exactly this clamp and nothing "
                        "else. Upscaling adds tokens, not detail, so whether "
                        "it helps an already-small source is unmeasured here."
                    )),
                io.Int.Input(
                    "short_edge", default=REF_IMAGE_SHORT_EDGE, min=256,
                    max=4096, step=32, tooltip=(
                        "Scale the image until its SHORTER side reaches this, "
                        "then round to 32. A 1280x720 reference has a short "
                        "side of 720, so 2048 scales it by 2.844 in both "
                        "directions, giving 3648x2048 and 7296 vision tokens "
                        "against 880 unscaled. "
                        "2048 is a property of the released checkpoint, not a "
                        "derivation: the reference pipeline carries it beside "
                        "the canvas rules as `reference_image_short_edge` and "
                        "would change it for a different checkpoint. "
                        "Image references are deliberately exempt from the "
                        "768x1344 area cap that binds the video, which is why "
                        "a reference may legitimately reach 7.5 megapixels "
                        "when the video itself cannot exceed about one. "
                        "Lowering it trades identity fidelity for sequence "
                        "length, which is the actual cost knob here -- and "
                        "note that upscaling adds tokens, not detail, so "
                        "whether it helps an already-small source is "
                        "unmeasured."
                    )),
                # APPENDED, and they have to stay: widget values map
                # positionally in every saved graph, so removing either would
                # break every graph built while they worked. Both are INERT.
                # See the module docstring for what each used to do and where
                # the surviving half of it went.
                io.Boolean.Input(
                    "lift_downstream_clamp", default=False, optional=True,
                    display_name="RETIRED: lift the 2048 clamp",
                    tooltip=(
                        "RETIRED and ignored. This armed a one-call override "
                        "on MiniMaxH3ReferenceToVideo, a node no graph in this "
                        "repo wires; and both consumers of the constant it "
                        "rebound had already bound it by value at import, so "
                        "the arm was invisible to them anyway. Retained only "
                        "so saved graphs keep their widget positions."
                    )),
                io.Boolean.Input(
                    "keep_towers_matched", default=True, optional=True,
                    display_name="RETIRED: keep VAE and Qwen on one size",
                    tooltip=(
                        "RETIRED and ignored here. The job was real -- core "
                        "encodes one tensor with the VAE and hands the SAME "
                        "object to the conditioner, so a Qwen-side resize "
                        "above the ceiling splits the towers -- but this node "
                        "has no clip and cannot tell which ceiling applies. It "
                        "read Comfy's native process_qwen2vl_images default and "
                        "applied it universally, which the AWQ adapter makes "
                        "wrong by a large factor. Use "
                        "MiniMaxH3ReferenceConditioning.image_policy instead."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                # Named `latent_rows` since 2026-08-13. `_tokens` returns the
                # DiT's packed rows, `(h//32)*(w//32)`, not Qwen vision tokens
                # -- the two coincide below Qwen's max_pixels and diverge above
                # it. The description and the module docstring both already
                # called this `latent_rows`; the output did not. display_name
                # is free to change, unlike node_id and position.
                io.Int.Output(display_name="latent_rows"),
            ],
            # Hidden inputs are not part of `inputs=[]`, so adding them moves
            # no widget position. `prompt` is what lets this node see whether
            # the downstream node is actually on 'max'.
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image, allow_upscale=True, short_edge=REF_IMAGE_SHORT_EDGE,
                lift_downstream_clamp=False, keep_towers_matched=True
                ) -> io.NodeOutput:
        if lift_downstream_clamp:
            logger.warning(
                "[h3] lift_downstream_clamp is RETIRED and does nothing. It "
                "armed an override on MiniMaxH3ReferenceToVideo, which this "
                "repo has never wired. Untick it.")
        if image.shape[0] > 1:
            logger.warning(
                "[h3] reference carries %d images; using the first. Wire one "
                "node per reference instead.", image.shape[0])
        src_h, src_w = int(image.shape[1]), int(image.shape[2])

        # The reference refuses out-of-range reference images outright, the
        # same limit it applies to the canvas.
        if not aspect_in_range(src_w, src_h):
            raise RuntimeError(
                f"A MiniMax H3 reference image must be within "
                f"{describe_aspect_range()}; this one is {src_w}x{src_h} "
                f"({src_w / src_h:.3g}). Crop it before referencing it."
            )

        tw, th = fit_reference_image(
            src_w, src_h, size_policy="max", short_edge=short_edge,
            allow_upscale=allow_upscale)
        out = _resize(image[:1], tw, th, "disabled")
        # What the DiT actually pays: reference latents are 16x downsampled
        # spatially, and every one of these rows is attended at every step.
        tokens = latent_rows(tw, th)
        # Always against ComfyUI's current behaviour, so the log answers "what
        # is this node changing" rather than restating what it just did.
        stock_w, stock_h = fit_reference_image(
            src_w, src_h, size_policy="max", short_edge=short_edge,
            allow_upscale=False)
        stock_tokens = latent_rows(stock_w, stock_h)

        logger.info(
            "[h3] reference %dx%d -> %dx%d (allow_upscale=%s, short_edge=%d): "
            "%d latent rows, %.2gx ComfyUI's own sizing",
            src_w, src_h, tw, th, allow_upscale, short_edge, tokens,
            tokens / stock_tokens,
        )

        # Say "no change" out loud. The ratio above already carries it -- 1.0
        # prints as "1x" -- which is exactly the problem: it reads as a
        # successful resize at a glance, and a reader who sees this node in a
        # graph concludes the reference was fitted.
        #
        # INFO and not WARNING on purpose. Inert here is usually the correct
        # state, deliberately chosen, and a warning that fires on more than half
        # the graphs that wire it is how a project learns to ignore warnings --
        # worse than not having one.
        if tokens == stock_tokens:
            full = short_edge / min(src_w, src_h)
            if not allow_upscale and full > 1.0:
                logger.info(
                    "[h3] reference fit made NO CHANGE: allow_upscale is off "
                    "and this reference's short edge (%d) is already inside "
                    "the %d clamp, so this node reproduced ComfyUI's own "
                    "sizing exactly. Wired but inert -- turn allow_upscale on "
                    "to reach the %d the reference pipeline conditions at.",
                    min(src_w, src_h), short_edge, short_edge)
            else:
                logger.info(
                    "[h3] reference fit made NO CHANGE: %dx%d is already what "
                    "ComfyUI's own sizing produces at short_edge=%d.",
                    tw, th, short_edge)

        return io.NodeOutput(out, tokens)
