"""Fit a reference image the way the reference pipeline does, including up.

`MiniMaxH3ReferenceToVideo` sizes reference images with

    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))     # nodes_minimax_h3.py:226

and the reference pipeline with

    scale = reference_image_short_edge / min(width, height)  # diffusers
                                    # modular_pipelines/minimax_h3/before_encoder.py:490

Same constant (2048), same round-to-32, one difference: the `min(1.0, ...)`.
ComfyUI never upscales. So a reference smaller than 2048 on its short edge
reaches the DiT at its original size, where the reference would have enlarged
it -- and reference tokens are latent rows, so a 512px reference contributes
16x fewer of them than the released pipeline gives it. Identity fidelity is
the whole job of a reference image, and it is being conditioned on a fraction
of the rows the model was built to see.

This node closes that gap the same way `MiniMaxH3KeyframeCanvas` closes the
canvas one: it does the resize itself, so the stock node's own scale becomes
`min(1.0, 2048/2048) = 1.0` and its resize is a no-op. It composes rather
than replacing.

**Upscaling is not free and the node says so.** Reference rows ride through
every sampling step, so quadrupling the short edge multiplies those rows by
16 and the stock node's own tooltip already warns that `max` "can be several
times slower". The `latent_rows` output is there to make that visible before
you queue a render rather than after.

It also carries the reference's 1:4..4:1 check on reference images
(`before_encoder.py:488`), which ComfyUI does not have -- the same limit
`h3_rules` holds for the canvas.
"""

from __future__ import annotations

import contextlib
import logging

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE,
                                           REF_IMAGE_SHORT_EDGE, _resize)

try:
    from .h3_rules import aspect_in_range, describe_aspect_range
except ImportError:
    from h3_rules import aspect_in_range, describe_aspect_range  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class MiniMaxH3ReferenceFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceFit",
            display_name="MiniMax H3 Reference Fit",
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
                io.Combo.Input(
                    "mode", options=["reference", "down_only"],
                    default="reference", tooltip=(
                        "reference: scale to the 2048 short edge in both "
                        "directions, matching the released pipeline. "
                        "down_only: ComfyUI's current behaviour, which never "
                        "enlarges -- here so the two can be A/B'd without "
                        "rewiring the graph."
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
                # APPENDED, and it has to stay last: widget values map
                # positionally in every saved graph.
                io.Boolean.Input(
                    "lift_downstream_clamp", default=False, optional=True,
                    tooltip=(
                        "Only matters above 2048. MiniMax H3 Reference to "
                        "Video clamps with min(1.0, 2048/short_edge), so "
                        "anything larger this node produces is scaled straight "
                        "back and the setting appears to do nothing. Turning "
                        "this on lifts that clamp for exactly one downstream "
                        "call, then restores it. "
                        "Above 2048 is off-distribution: 2048 is what the "
                        "released checkpoint conditioned image references at. "
                        "Cost climbs quadratically -- 3072 is 16,416 vision "
                        "tokens against 7,296 -- per reference, on a sequence "
                        "that already crosses the int32 threshold at 345 "
                        "frames. Requires ref_image_size on 'max'; under "
                        "'match' the constant is never read and this logs that "
                        "it did nothing."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="vision_tokens"),
            ],
        )

    @classmethod
    def execute(cls, image, mode="reference", short_edge=REF_IMAGE_SHORT_EDGE,
                lift_downstream_clamp=False
                ) -> io.NodeOutput:
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

        full = short_edge / min(src_w, src_h)
        scale = min(1.0, full) if mode == "down_only" else full

        tw, th = _fit(src_w, src_h, scale)
        out = _resize(image[:1], tw, th, "disabled")
        # What the DiT actually pays: reference latents are 16x downsampled
        # spatially, and every one of these rows is attended at every step.
        tokens = _tokens(tw, th)
        # Always against ComfyUI's current behaviour, so the log answers "what
        # is this node changing" rather than restating what it just did.
        stock_tokens = _tokens(*_fit(src_w, src_h, min(1.0, full)))

        logger.info(
            "[h3] reference %dx%d -> %dx%d (%s, short_edge=%d): %d latent rows, "
            "%.2gx ComfyUI's own sizing", src_w, src_h, tw, th, mode,
            short_edge, tokens, tokens / stock_tokens,
        )
        if lift_downstream_clamp and short_edge > REF_IMAGE_SHORT_EDGE:
            arm_short_edge_override(short_edge)
        elif lift_downstream_clamp:
            logger.info(
                "[h3] lift_downstream_clamp is on but short_edge is %d, at or "
                "below the %d clamp, so there is nothing to lift.",
                short_edge, REF_IMAGE_SHORT_EDGE)

        return io.NodeOutput(out, tokens)


def _fit(w, h, scale):
    def snap(v):
        return max(CANVAS_MULTIPLE,
                   round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


def _tokens(w, h):
    """Vision tokens a reference of this pixel size contributes to the DiT.

    Two stages, not one. The VAE compresses space by 16, then the DiT
    patchifies that latent with `patch_size=(1, 2, 2)` before anything is
    attended (`comfy/ldm/minimax/model.py`, `patchify_video`). Counting only
    the VAE stage reports four times what the sequence actually carries,
    which is what this function did until 2026-08-11 and what the number in
    the 0.3.0 changelog entry came from.
    """
    return (h // 32) * (w // 32)


# --------------------------------------------------------------------------
# Lifting ComfyUI's 2048 clamp, for one downstream call
# --------------------------------------------------------------------------
#
# `MiniMaxH3ReferenceToVideo.execute` sizes image references with
# `min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))` (`nodes_minimax_h3.py:226`).
# The clamp is what lets this node work at all below 2048: we resize first,
# the stock scale resolves to 1.0, and its resize is a no-op. Above 2048 the
# same clamp scales our work back down, so `short_edge` is one-directional
# and a sweep past the default silently does nothing.
#
# `REF_IMAGE_SHORT_EDGE` is a module attribute read inside `execute` at call
# time, so rebinding it changes behaviour. Rebinding it *globally* would be
# the wrong fix: it is sticky for the process, invisible in the UI, and would
# reach graphs that do not contain this node -- the same silent-contamination
# class `nodes.py` guards against when it copies transformer_options. So the
# override is armed by this node, consumed by exactly one downstream call,
# and cleared in a `finally`.
#
# Above 2048 is off-distribution by construction: 2048 is what the released
# checkpoint conditioned image references at, carried in the reference
# pipeline as `ConfigSpec("reference_image_short_edge", 2048)`. This exists to
# make that measurable, not because bigger is better.

_PENDING_SHORT_EDGE = None
_WRAP_MARKER = "_h3_explorations_short_edge_wrapper"


@contextlib.contextmanager
def _rebound_short_edge(value):
    """Swap the module constant for the duration of one call."""
    import comfy_extras.nodes_minimax_h3 as core

    previous = core.REF_IMAGE_SHORT_EDGE
    core.REF_IMAGE_SHORT_EDGE = value
    try:
        yield
    finally:
        core.REF_IMAGE_SHORT_EDGE = previous


def _make_wrapper(original):
    """Wrap `ReferenceToVideo.execute` so an armed override applies once.

    Factored out so the behaviour is testable without a VAE, a CLIP or a
    model: `bench/check_short_edge_override.py` calls this with a stub.
    """
    def wrapper(*args, **kwargs):
        global _PENDING_SHORT_EDGE
        pending, _PENDING_SHORT_EDGE = _PENDING_SHORT_EDGE, None
        if pending is None:
            return original(*args, **kwargs)
        if kwargs.get("ref_image_size", "match") == "match":
            # In `match` the stock node never reads the constant -- it scales
            # to the generation's pixel area instead. An override that
            # silently does nothing in the default configuration is worse
            # than no override, so say so rather than appear to work.
            logger.warning(
                "[h3] short_edge override of %d ignored: ref_image_size is "
                "'match', which sizes references from the video's pixel area "
                "and never reads the 2048 constant. Set it to 'max'.", pending)
            return original(*args, **kwargs)
        logger.info("[h3] lifting the reference clamp to %d for one call "
                    "(off-distribution above 2048)", pending)
        with _rebound_short_edge(pending):
            return original(*args, **kwargs)

    setattr(wrapper, _WRAP_MARKER, True)
    return wrapper


def _install_wrapper():
    """Wrap the stock node once. Idempotent, and says so if someone else won.

    The chaining packs established the marker convention for exactly this;
    two packs each wrapping unaware of the other is the collision class this
    ecosystem keeps producing.
    """
    import comfy_extras.nodes_minimax_h3 as core

    node = core.MiniMaxH3ReferenceToVideo
    current = node.__dict__.get("execute")
    inner = current.__func__ if isinstance(current, classmethod) else current
    if getattr(inner, _WRAP_MARKER, False):
        return
    node.execute = classmethod(_make_wrapper(inner))


def arm_short_edge_override(value):
    """Arm the override for the next downstream ReferenceToVideo call."""
    global _PENDING_SHORT_EDGE
    _install_wrapper()
    _PENDING_SHORT_EDGE = value
