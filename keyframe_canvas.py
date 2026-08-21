"""Resolve an H3 canvas from a keyframe, the way the reference pipeline does.

`MiniMaxH3ImageToVideo` takes `width`/`height` as required inputs (default
1344x768) and stretches the first keyframe onto them. The stretch itself is
faithful -- the reference stretches the geometry anchor and cover-crops any
follower, deliberately, to match the released model's arithmetic. What ComfyUI
does not have is the default that normally makes the stretch a no-op: the
reference derives the canvas from the first keyframe when no size is given
(`MiniMaxH3ResizeStep` -> `resolve_canvas_size`) and then skips the resize
entirely once the keyframe already matches.

So the reference's deliberate-override branch is ComfyUI's default branch, and
an off-16:9 keyframe is silently distorted: measured 1.75x for a square source
at the default canvas and 2.33x for 3:4 portrait, carried by every frame of the
clip. This node closes that gap. `adapt_canvas` is ComfyUI's own port of
`resolve_canvas_size` -- same 768 short edge, same 768*1344 area cap, same
round-to-32 -- and it already sits in `nodes_minimax_h3.py`, just unused on the
keyframe path.

Verified in `bench/check_keyframe_canvas.py`: feeding the fitted image plus the
derived size makes both of the stock node's resize calls bit-identical
no-ops, so this composes with it rather than replacing it.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import adapt_canvas, _resize

logger = logging.getLogger(__name__)

# The reference's input rules, shared with anything else that needs them so
# the constants cannot drift. See h3_rules.py for what each one is and where
# in the reference it comes from.
try:
    from .h3_rules import (aspect_in_range, describe_aspect_range,
                           describe_length, duration_in_range, is_single_frame,
                           max_legal_length, min_legal_length, snap_length)
except ImportError:
    # ComfyUI loads this as a package; the bench scripts import it as a
    # top-level module with the repo root on sys.path. Both are legitimate
    # and the relative form only works for the first.
    from h3_rules import (aspect_in_range, describe_aspect_range,  # type: ignore[no-redef]
                          describe_length, duration_in_range, is_single_frame,
                          max_legal_length, min_legal_length, snap_length)


class MiniMaxH3KeyframeCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3KeyframeCanvas",
            display_name="MiniMax H3 Keyframe Resolution",
            category="model/conditioning/minimax",
            description=(
                "Derives the generation canvas from the first keyframe, matching the "
                "reference pipeline's default, and fits the keyframes onto it. Wire "
                "width/height and the fitted images into MiniMax H3 Image to Video: "
                "the keyframe then arrives already at canvas size, so that node's "
                "resize is a no-op and nothing is distorted."
            ),
            inputs=[
                io.Image.Input("first_frame"),
                # Options keep their order: a saved graph stores the chosen
                # string, but the default changed on 2026-08-13 from
                # fit_to_canvas to match_keyframe. Saved graphs carry their own
                # value and are unaffected; only a newly dropped node moves.
                io.Combo.Input("mode", options=["fit_to_canvas", "match_keyframe"],
                               default="match_keyframe", tooltip=(
                                   "match_keyframe (default): what the reference pipeline "
                                   "actually does when no size is given -- the canvas is "
                                   "derived from the keyframe's aspect and your width/height "
                                   "are ignored. Nothing is cropped, and the first frame is "
                                   "stretched while a last frame is cover-cropped, as in the "
                                   "reference. Use this for anything compared against diffusers. "
                                   "fit_to_canvas: you own the geometry -- BOTH keyframes are "
                                   "cover-cropped into the width/height you pass. That is the "
                                   "reference's deliberate-override branch, not its default, "
                                   "and it is lossy: a 3:4 photo forced to 1344x768 keeps 43% "
                                   "of its frame. It saves nothing at 1344x768, which is "
                                   "already the largest area adapt_canvas ever returns, so it "
                                   "only pays once you lower width/height on purpose.")),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32,
                             tooltip="Used by fit_to_canvas; ignored by match_keyframe."),
                io.Int.Input("height", default=768, min=32, max=16384, step=32,
                             tooltip="Used by fit_to_canvas; ignored by match_keyframe."),
                io.Image.Input("last_frame", optional=True),
                # Default moved 0 -> 124 on 2026-08-13. At 0 this output
                # forwards 0, and core's own min=5 does NOT catch it: a linked
                # input skips range validation entirely, so the render was a
                # 5-frame, 0.208s clip. 124 is the trained floor and matches
                # both core's default and MiniMaxH3Resolution's.
                io.Int.Input("length", default=124, min=0, max=3600, optional=True,
                             tooltip=(
                                 "Frame count to check and snap, passed straight "
                                 "through to MiniMax H3 Image to Video. Frames "
                                 "snap UP to the video VAE's 17n+5 grid, and "
                                 "the ceiling is 362 (15.083s), the longest "
                                 "length H3 was trained on. ComfyUI's node "
                                 "accepts up to 3600 with no ceiling at all. "
                                 "The reference pipeline stops at 345, one grid "
                                 "step lower -- portability, not a model limit. "
                                 "0 skips the check AND emits 0, which becomes "
                                 "a 5-frame clip downstream -- do not wire this "
                                 "output at 0."
                             )),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Image.Output(display_name="first_frame"),
                io.Image.Output(display_name="last_frame"),
                io.Float.Output(display_name="attn_cost_vs_1to1"),
                # Appended, NOT inserted next to width/height where it belongs
                # semantically. Output slots are positional in every saved
                # graph: putting `length` third would shift first_frame,
                # last_frame and attn_cost down one, and every existing
                # workflow wiring them would silently connect to the wrong
                # slot. Same reasoning as the node_id rule in CLAUDE.md.
                io.Int.Output(display_name="length"),
            ],
        )

    @classmethod
    # These MUST match the schema defaults above. ComfyUI does not inject a
    # schema default for an input a prompt omits -- the Python default is what
    # applies -- so the widget default only protects a node newly dropped in
    # the UI. An API-format prompt that leaves `length` out lands on the
    # signature's value, and at 0 that emits 0 from slot 5 and renders a
    # 5-frame 0.208s clip. The API path is how the benches are driven, so a
    # split between these two is a live bug on exactly the path that matters.
    # `bench/check_schema_defaults.py` asserts they agree, for every node here.
    def execute(cls, first_frame, mode="match_keyframe", width=1344, height=768,
                last_frame=None, length=124) -> io.NodeOutput:
        return io.NodeOutput(*resolve_keyframe_geometry(
            first_frame=first_frame, last_frame=last_frame, mode=mode,
            width=width, height=height, length=length))


def resolve_keyframe_geometry(first_frame=None, last_frame=None,
                              mode="match_keyframe", width=1344, height=768,
                              length=0):
    """(width, height, first_out, last_out, attn_cost, length) for a keyframe set.

    **Module-level so the geometry has exactly one implementation.** The node
    above is a thin wrapper and `conditioning.py` calls this directly. Every
    rule below -- the aspect refusal in `match_keyframe`, the snap-then-check
    length order, the anchor-versus-follower crop asymmetry -- was reasoned
    out once and must not be reasoned out twice.

    **`first_frame` is optional here and is not on the node.** The node keeps
    requiring it for compatibility; this function accepts a last-frame-only
    set because that is a released fl2va signature the node could never
    reach. The anchor is chosen by semantic frame index, which is sglang's
    rule at `prequeue.py:97-107` -- frame 0 sorts before the final-frame
    sentinel, so `first_frame` anchors when present and `last_frame` anchors
    when it is the only one. That is what makes a lone last frame keep its
    whole picture instead of being cover-cropped into a canvas chosen
    elsewhere.
    """
    anchor = first_frame if first_frame is not None else last_frame
    if anchor is None:
        raise ValueError("resolve_keyframe_geometry needs a first_frame or "
                         "a last_frame to anchor on")
    if anchor.shape[0] > 1:
        # the H3 node takes [:1] silently; say so rather than let a batch
        # look like it was used
        logger.warning(
            "[h3] the anchor keyframe carries %d images; MiniMax H3 uses only "
            "the first. Batch the prompt instead if you meant several renders.",
            anchor.shape[0],
        )
    src_h, src_w = int(anchor.shape[1]), int(anchor.shape[2])

    if mode == "match_keyframe":
        # The reference refuses here rather than resolving a canvas the
        # checkpoint was never trained on, and in this mode the aspect
        # comes from the image rather than from the user -- so nobody has
        # chosen it and nobody would otherwise be told. Raising is the
        # whole point: `adapt_canvas` would return a perfectly plausible
        # canvas and the render would just be bad.
        if not aspect_in_range(src_w, src_h):
            raise RuntimeError(
                f"MiniMax H3 was trained on aspect ratios from "
                f"{describe_aspect_range()}; this keyframe is "
                f"{src_w}x{src_h} ({src_w / src_h:.3g}). Crop it, or "
                f"switch to fit_to_canvas to choose the geometry yourself."
            )
        width, height = adapt_canvas(src_w, src_h)
        # Aspect now matches by construction, so "disabled" is a uniform
        # scale, not a stretch. Mirrors the reference's anchor path.
        anchor_crop = "disabled"
    else:
        # Round to the DiT's multiple of 32 and otherwise leave the size
        # alone. NOT adapt_canvas: that forces a 768 short edge and the area
        # cap, which would silently promote a 832x480 preview canvas to
        # 1344x768 -- a 6.7x attention increase from a node whose whole job
        # here is keeping render cost where the user put it.
        snapped_w = max(32, round(width / 32) * 32)
        snapped_h = max(32, round(height / 32) * 32)
        if (snapped_w, snapped_h) != (width, height):
            # step=32 constrains the UI, not an API submission, and the
            # API-format workflows in this repo are how benches are driven.
            # A size the DiT cannot patch would fail downstream with a
            # shape error that says nothing about where it came from.
            logger.warning(
                "[h3] canvas %dx%d is not a multiple of 32; "
                "snapped to %dx%d. The H3 latent cannot grid the original.",
                width, height, snapped_w, snapped_h,
            )
        width, height = snapped_w, snapped_h
        # Warn rather than raise: in this mode the user typed the geometry,
        # and this node's contract here is "you own it". Refusing a size
        # somebody deliberately entered would be this node overruling them,
        # which is the opposite of the mode. They still get told, because
        # the failure is a quality one and would otherwise be invisible.
        if not aspect_in_range(width, height):
            logger.warning(
                "[h3] canvas %dx%d is aspect %.3g, outside the %s range "
                "MiniMax H3 was trained on. It will render; expect the "
                "output to degrade rather than fail.",
                width, height, width / height, describe_aspect_range(),
            )
        # The user owns the geometry, so the anchor is cover-cropped rather
        # than stretched. NOTE: this is a deliberate divergence -- the
        # reference stretches the anchor even when width/height are given,
        # and only cover-crops followers. Cropping both keeps proportions
        # honest when the aspect was chosen on purpose; it costs edge
        # framing. Reference fidelity lives in match_keyframe.
        anchor_crop = "center"

    anchor_out = _resize(anchor[:1], width, height, anchor_crop)
    # The follower is cover-cropped in either mode, as in the reference.
    # Which frame is the follower depends on which one anchored: with a
    # lone last frame there is no follower and the anchor fills both slots.
    if first_frame is not None:
        first_out = anchor_out
        last_out = (_resize(last_frame[:1], width, height, "center")
                    if last_frame is not None else anchor_out)
    else:
        first_out = last_out = anchor_out

    # DiT tokens go as (h//32)*(w//32) and attention as their square, so
    # report cost against the cheapest canvas H3 will resolve (768x768).
    tokens = (height // 32) * (width // 32)
    cheapest = (768 // 32) ** 2
    attn_cost = round((tokens / cheapest) ** 2, 3)

    # Length is checked in the reference's order: snap to the VAE grid
    # first, then hold the duration ceiling against the RESULT. Checking
    # the request instead passes 346 and then renders 362.
    if length:
        snapped = snap_length(length)
        # Single frame is refused HERE and allowed on the reference path,
        # and that asymmetry is deliberate rather than an oversight. This
        # node feeds MiniMaxH3ImageToVideo, which pins a `last_frame` at
        # `frame_count - 1` -- frame 0 in a one-frame video, i.e. on top of
        # `first_frame`. Nobody has established what fl2va does at one
        # frame, so the shipped single-image path is ref2v. If that gets
        # measured, this branch is the thing to delete.
        if is_single_frame(snapped):
            raise RuntimeError(
                "length=1 (single-frame image mode) is not supported on "
                "the keyframe path. MiniMaxH3ImageToVideo anchors a "
                "last_frame at the final frame, which in a one-frame video "
                "is the first frame, and fl2va at one frame is unmeasured. "
                "Use MiniMaxH3ReferenceToVideo with reference images for "
                "single-image edits -- that is the path this repo ships "
                "and renders."
            )
        if not duration_in_range(length):
            raise RuntimeError(
                f"MiniMax H3 generates 5 to 15 seconds at 24fps; "
                f"{describe_length(length)} is outside that. Legal counts "
                f"run {min_legal_length()} to {max_legal_length()} on the "
                f"video VAE's 17n+5 grid."
                + (f" Note {length} snaps up to {snapped} before the "
                   f"ceiling applies." if snapped != length else "")
            )
        if snapped != length:
            logger.info("[h3] length %s", describe_length(length))
        length = snapped

    logger.info(
        "[h3] H3 canvas %dx%d (%s) from a %dx%d keyframe: "
        "aspect %.4f -> %.4f, attention ~%.2fx a 768x768 canvas",
        width, height, mode, src_w, src_h,
        src_w / src_h, width / height, attn_cost,
    )
    return width, height, first_out, last_out, attn_cost, length
