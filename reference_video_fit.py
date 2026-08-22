"""Say what a reference VIDEO will actually be conditioned at, and optionally set it.

**Why this exists.** `MiniMaxH3ReferenceFit` covers reference images. Nothing in
this pack touches reference *video*: `bench/preflight_graph.py` prices one
statically, but no node transforms or reports one. So for a reference video you
could not find out what resolution it finally reached, could not hold it under
Qwen's ceiling, and could not tell a deliberate downscale from an accidental
one. That silence is the defect -- not the sizing, which is defensible.

**Reporting is the deliverable here, not resizing.** Core re-applies its own
rule to whatever it is handed
(`comfy_extras/nodes_minimax_h3.py:315-320`): a canvas from `adapt_canvas`,
overridden by the source size rounded to 32 when the source has fewer pixels.
So a resize from this node survives only if it is a **fixed point** of that
rule, which downscales below the canvas area are and upscales are not -- an
upscale is simply capped back to the canvas. The node therefore defaults to
changing nothing and telling you what will happen.

**Not upscaling is the right default and this node does not argue with it.**
Reference rows are attended at every sampling step, so reference size taxes the
whole render rather than costing once; `docs/h3_references.md` prices it.
Deliberately passing downscaled reference video is sound, and the cost of the
alternative is measured while its benefit is not.

**What this does NOT do: frame rate.** `force_rate=24` on the loader owns that,
and `bench/check_ref_prompt_labels.py` gates it. Putting a second fps control
here would give two places to get it wrong.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE, adapt_canvas,
                                           _resize)
from .reference_fit import clamp_to_qwen_ceiling, qwen_max_pixels

logger = logging.getLogger(__name__)


def core_video_size(vw: int, vh: int) -> tuple[int, int]:
    """What core will resize a reference video to, given a source size.

    Mirrors `comfy_extras/nodes_minimax_h3.py:315-320` deliberately rather than
    importing, because core computes it inline inside `execute` and there is no
    function to call. **That makes it a copy, which this repo forbids without a
    reason: the reason is that it is a PREDICTION, and a prediction that stops
    matching is the thing worth reporting.** `bench/check_ref_video_prediction.py`
    holds it to core's real behaviour.
    """
    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return cw, ch


def latent_rows(w: int, h: int, frames: int) -> int:
    """Rows this reference video contributes, and they ride EVERY step.

    Space is compressed 16x by the VAE then patchified 2x2 by the DiT, so a
    frame contributes `(h//32) * (w//32)`. Time is compressed to the `17n+5`
    grid core snaps down to: `n//4 + 1` latent frames.
    """
    per_frame = (h // (CANVAS_MULTIPLE)) * (w // (CANVAS_MULTIPLE))
    n = frames
    while n % 17 != 5 and n > 5:
        n -= 1
    return per_frame * (n // 4 + 1)


class MiniMaxH3ReferenceVideoFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceVideoFit",
            display_name="MiniMax H3 Reference Video Resolution",
            category="model/conditioning/minimax",
            description=(
                "Reports what resolution a reference video will actually be "
                "conditioned at, what it costs in rows that ride every "
                "sampling step, and whether the VAE and Qwen will agree on it. "
                "Optionally downscales to a short edge. Defaults to changing "
                "nothing: reference rows tax the whole render, so a smaller "
                "reference is usually the right call and this node exists to "
                "make that choice visible rather than to override it."
            ),
            inputs=[
                io.Image.Input(
                    "frames", tooltip="Reference video frames, 24 fps. Use "
                                      "force_rate=24 on the loader."),
                io.Int.Input(
                    "short_edge", default=0, min=0, max=2048, step=32,
                    tooltip=(
                        "0 leaves the frames alone and only reports, which is "
                        "the default and usually correct. "
                        "Above 0, scales the shorter side to this and rounds "
                        "to 32. Only DOWNSCALES take effect: core re-derives "
                        "the size from its own canvas rule and an upscale is "
                        "capped straight back, so asking for more than the "
                        "canvas does nothing and this node says so."
                    )),
                io.Boolean.Input(
                    "keep_towers_matched", default=True, optional=True,
                    tooltip=(
                        "Hold the frames under Qwen's vision ceiling so the "
                        "VAE and the conditioner see the same picture. A "
                        "reference video is canvas-sized and sits far below "
                        "that ceiling, so this is inert in every ordinary "
                        "case and is here so the two reference nodes behave "
                        "the same way."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="frames"),
                io.Int.Output(display_name="latent_rows"),
            ],
        )

    @classmethod
    def execute(cls, frames, short_edge=0, keep_towers_matched=True
                ) -> io.NodeOutput:
        n = int(frames.shape[0])
        src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
        out = frames

        if short_edge > 0:
            scale = short_edge / min(src_w, src_h)
            tw = max(CANVAS_MULTIPLE,
                     round(src_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE,
                     round(src_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            if keep_towers_matched:
                tw, th, _ = clamp_to_qwen_ceiling(tw, th, qwen_max_pixels())
            if (tw, th) != (src_w, src_h):
                out = _resize(frames, tw, th, "disabled")
        else:
            tw, th = src_w, src_h

        # What core will do to what we just handed it. This is the number the
        # user actually gets, and it is not necessarily the one above.
        final_w, final_h = core_video_size(tw, th)
        rows = latent_rows(final_w, final_h, n)

        logger.info(
            "[h3] reference video %dx%d x%d frames -> core conditions it at "
            "%dx%d: %d latent rows, attended at EVERY sampling step.",
            src_w, src_h, n, final_w, final_h, rows)

        if short_edge > 0 and (final_w, final_h) != (tw, th):
            logger.warning(
                "[h3] reference video fit was OVERRIDDEN: this node produced "
                "%dx%d and core re-derived %dx%d from its own canvas rule. "
                "Only downscales below the canvas area survive; an upscale is "
                "capped straight back. Lower short_edge or leave it at 0.",
                tw, th, final_w, final_h)
        elif short_edge == 0:
            logger.info(
                "[h3] reference video fit is REPORTING ONLY (short_edge=0). "
                "Set it to downscale, which lowers rows on every step. Not "
                "upscaling is the cheaper default and the benefit of a larger "
                "reference is unmeasured.")

        ceiling = qwen_max_pixels()
        if final_w * final_h > ceiling:
            logger.warning(
                "[h3] reference video %dx%d is past Qwen's %d-pixel ceiling, "
                "so the VAE and the conditioner will see different "
                "resolutions of it and nothing downstream says so.",
                final_w, final_h, ceiling)

        return io.NodeOutput(out, rows)
