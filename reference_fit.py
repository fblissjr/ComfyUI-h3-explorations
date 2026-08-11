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
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="vision_tokens"),
            ],
        )

    @classmethod
    def execute(cls, image, mode="reference", short_edge=REF_IMAGE_SHORT_EDGE
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
