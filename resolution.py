"""Pick a MiniMax H3 resolution by shape, and see what it costs before you render.

Two things this exists to make visible, because nothing in the stock graph
says either one.

**Which resolutions the model was trained on.** `adapt_canvas()` derives a
resolution from an aspect ratio: short edge 768, area capped at 768*1344,
each axis rounded to 32. Sweeping the legal 1/4..4 aspect range through it
yields exactly 95 resolutions. But core's conditioning nodes never call it --
`MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo` and
`EmptyMiniMaxH3LatentAV` all take width and height as plain ints at
`min=32, step=32`. So those 95 are the *trained family*, not the legal set.
1024x1024 is legal, 32-divisible, renders fine, and is outside the family
because its area exceeds the cap. Nothing tells you that today.

**What the choice costs.** Attention is O(S^2) and dominates the step, so
aspect ratio is the largest single lever in the pipeline -- 1:1 costs a third
of 16:9 at the same length. The cost belongs in the dropdown you choose from,
not in a doc you read afterwards.

Divisibility by 32 is the one hard rule and it is architectural: the VAE
compresses space by 16, then the DiT patchifies that latent with
`patch_size=(1, 2, 2)` (`comfy/ldm/minimax/model.py`). 16 * 2 = 32. Divisible
by 16 alone leaves an odd latent axis the 2x2 patchify cannot tile.

The shape bands exist so no dropdown is long. All 95 are reachable: 48
landscape and square entries banded by ratio, their portrait mirrors (which
cost exactly the same, since tokens go as `(h//32)*(w//32)` and that is
symmetric), and `custom` for anything else including outside the family.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from comfy_api.latest import io

# Both spellings, matching the sibling nodes: relative when ComfyUI loads
# this as a package, absolute when a bench script imports it directly.
try:
    from .h3_rules import (aspect_in_range, describe_aspect_range,
                           describe_length, duration_in_range, duration_of,
                           snap_length)
except ImportError:  # pragma: no cover
    from h3_rules import (aspect_in_range, describe_aspect_range,  # type: ignore[no-redef]
                          describe_length, duration_in_range, duration_of,
                          snap_length)

logger = logging.getLogger(__name__)

# Ratio thresholds for the shape bands. Chosen so no band exceeds ~22
# entries; they carry no meaning to the model.
_BANDS = (
    ("ultrawide", 2.2, 99.0),
    ("wide", 1.5, 2.2),
    ("standard", 1.02, 1.5),
    ("square", 0.98, 1.02),
)


def _all_resolutions():
    """Every resolution `adapt_canvas` can produce, banded by shape.

    Swept rather than hardcoded, so this tracks upstream if the short edge,
    the area cap or the rounding ever change. 200k steps because a 4k sweep
    misses 576x1728.
    """
    from comfy_extras.nodes_minimax_h3 import adapt_canvas

    seen = {}
    for i in range(1, 200001):
        r = 0.25 + (4.0 - 0.25) * i / 200000
        seen[adapt_canvas(int(round(r * 100000)), 100000)] = True

    banded: dict[str, list[tuple[int, int]]] = {name: [] for name, _, _ in _BANDS}
    for name in ("portrait", "tall", "ultratall"):
        banded[name] = []
    for w, h in seen:
        ratio = w / h
        if w >= h:
            for name, lo, hi in _BANDS:
                if lo <= ratio < hi:
                    banded[name].append((w, h))
                    break
        else:
            # Portrait mirrors landscape at identical cost, so band it by the
            # ratio of its mirror rather than inventing separate thresholds.
            for name, mirrored in (("portrait", "standard"), ("tall", "wide"),
                                   ("ultratall", "ultrawide")):
                lo, hi = next((l, x) for n, l, x in _BANDS if n == mirrored)
                if lo <= h / w < hi:
                    banded[name].append((w, h))
                    break
    for name in banded:
        banded[name].sort(key=lambda wh: -wh[0] / wh[1])
    return banded


def _label(w, h):
    """Option text. Starts with WxH so `_parse` stays valid if the rest moves."""
    tokens = (w // 32) * (h // 32)
    base = (1344 // 32) * (768 // 32)
    return (f"{w}x{h}  {Fraction(w, h).limit_denominator(24)}  "
            f"{tokens} tok/frame  {(tokens / base) ** 2:.2f}x")


def _parse(label):
    w, h = label.split()[0].split("x")
    return int(w), int(h)


_RESOLUTIONS = None


def _resolutions():
    global _RESOLUTIONS
    if _RESOLUTIONS is None:
        _RESOLUTIONS = _all_resolutions()
    return _RESOLUTIONS


class MiniMaxH3Resolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        banded = _resolutions()
        options = []
        for name in ("ultrawide", "wide", "standard", "square",
                     "portrait", "tall", "ultratall"):
            entries = [_label(w, h) for w, h in banded[name]]
            options.append(io.DynamicCombo.Option(name, [
                io.Combo.Input(
                    f"{name}_resolution", options=entries, default=entries[0],
                    tooltip="Resolution, aspect ratio, video tokens per latent "
                            "frame, and attention cost against 16:9 at the same "
                            "length. Every entry here is inside the family the "
                            "model was trained on."),
            ]))
        options.append(io.DynamicCombo.Option("custom", [
            io.Int.Input("width", default=1344, min=32, max=16384, step=32,
                         tooltip="Must be a multiple of 32: the VAE compresses "
                                 "space by 16 and the DiT patchifies 2x2 on top "
                                 "of that."),
            io.Int.Input("height", default=768, min=32, max=16384, step=32,
                         tooltip="Multiple of 32, same reason as width."),
        ]))

        return io.Schema(
            node_id="MiniMaxH3Resolution",
            display_name="MiniMax H3 Resolution",
            category="model/conditioning/minimax",
            description=(
                "Pick a resolution by shape and see its cost before rendering. "
                "Wire width, height and length into MiniMax H3 Image to Video or "
                "Reference to Video. Custom reaches anything the DiT can patch, "
                "including resolutions outside the trained family -- the node "
                "says which side you are on rather than refusing."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "shape", options=options, tooltip=(
                        "Banded so no list is long. All 95 trained resolutions "
                        "are reachable: landscape and square by band, their "
                        "portrait mirrors (identical cost, since tokens go as "
                        "(h//32)*(w//32) and that is symmetric), or custom."
                    )),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=1, tooltip=(
                        "Frame count at 24 fps, rounded UP to the video VAE's "
                        "17n+5 temporal grid. 200 gives 209, 300 gives 311. The "
                        "reference generates 5-15s and applies that ceiling "
                        "after the rounding, so 345 (14.375s) is the largest "
                        "legal count and 346 rounds to 362 (15.083s), which is "
                        "over. This node warns rather than refusing."
                    )),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="length"),
                io.Int.Output(display_name="video_tokens"),
                io.Int.Output(display_name="tokens_per_frame"),
                io.Float.Output(display_name="attn_cost_vs_16_9"),
                io.String.Output(display_name="summary"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, shape, length=124) -> io.NodeOutput:
        from comfy_extras.nodes_minimax_h3 import adapt_canvas, temporal_shape

        # A DynamicCombo arrives as ONE nested dict, not as flattened kwargs:
        # the selected key under the input's own id, and the chosen option's
        # inputs alongside it. Core reads it the same way -- see
        # `comfy_extras/nodes_depth_anything_3.py`, `mode["mode"]` then
        # `mode["pose_method"]`.
        #
        # The first version of this read `shape.get("key")` and took its
        # option inputs from `**kw`, so every selection fell through to the
        # custom branch and every render came out 1344x768 whatever you
        # picked. It passed its test because the test called `execute` with
        # flattened kwargs -- I invented the caller instead of using the real
        # one, so the test agreed with the bug.
        key = shape if isinstance(shape, str) else shape["shape"]
        if key == "custom":
            width = int(shape["width"])
            height = int(shape["height"])
        else:
            width, height = _parse(shape[f"{key}_resolution"])

        notes = []
        if width % 32 or height % 32:
            # step=32 constrains the UI, not an API submission, and this repo
            # drives benches through the API format.
            raise RuntimeError(
                f"{width}x{height} is not a multiple of 32. The VAE compresses "
                f"space by 16 and the DiT patchifies 2x2 on top of that, so an "
                f"odd latent axis cannot be tiled."
            )

        snapped = snap_length(length)
        if snapped != length:
            notes.append(f"length {length} -> {snapped} on the 17n+5 grid")
        if not duration_in_range(snapped):
            notes.append(
                f"WARNING {describe_length(snapped)} is outside the reference's "
                f"5-15s window, which is the REFERENCE pipeline's ceiling "
                f"rather than a training limit -- 362 is trained. 345 is "
                f"the largest count the reference would emit")

        tokens_per_frame = (width // 32) * (height // 32)
        latent_frames = temporal_shape(snapped)[1]
        video_tokens = tokens_per_frame * latent_frames
        base = (1344 // 32) * (768 // 32)
        cost = (tokens_per_frame / base) ** 2

        in_family = adapt_canvas(width, height) == (width, height)
        if not aspect_in_range(width, height):
            notes.append(
                f"WARNING aspect {width / height:.3g} is outside the trained "
                f"range of {describe_aspect_range()}")
        elif not in_family:
            notes.append(
                f"outside the trained family: {width}x{height} is legal and "
                f"will render, but adapt_canvas maps this aspect to "
                f"{'x'.join(map(str, adapt_canvas(width, height)))}")

        summary = "\n".join([
            f"{width}x{height}  {Fraction(width, height).limit_denominator(24)}"
            f"  {'trained family' if in_family else 'outside trained family'}",
            f"{snapped} frames ({duration_of(snapped):.2f}s)  "
            f"{latent_frames} latent frames",
            f"{tokens_per_frame} video tokens/frame  {video_tokens:,} video tokens",
            f"attention {cost:.2f}x a 16:9 render at this length",
            *notes,
        ])
        logger.info("[h3] resolution %s", summary.replace("\n", " | "))

        unique_id = getattr(cls.hidden, "unique_id", None)
        if unique_id:
            try:
                from server import PromptServer
                PromptServer.instance.send_progress_text(summary, unique_id)
            except Exception:
                pass

        return io.NodeOutput(width, height, snapped, video_tokens,
                             tokens_per_frame, cost, summary)
