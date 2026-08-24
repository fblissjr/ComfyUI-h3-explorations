"""The one implementation of H3 reference-image sizing.

**Why this is a module and not a method.** Two consumers apply this arithmetic
and they do not share a process. `MiniMaxH3AppendRefImage` records the decision
inside a ComfyUI graph; the post-training calibration builder applies the same
decision to media named in a manifest, with no graph anywhere. The active plan
(`docs/research/qwen3-vl-special-tokens-post-training/canonical/active_plan.md`)
names two strata that are exactly this function's arguments -- a primary `max`
with upscaling off, and a separately named 2048-short-edge upscale-allowed
stress stratum -- and requires every row to record which one it came from. Two
implementations of that is the drift nothing would have caught, because both
copies would be individually correct and would disagree only on inputs neither
author tried.

**Stage one of two.** This is *upstream role sizing*: what geometry the
reference is prepared at before any Qwen processor sees it. The second stage --
what the selected Qwen still-image policy then does to it -- belongs to
`reference_conditioning._qwen_image_settings`, because it depends on which
encoder artifact is loaded and that is knowable only where the CLIP is in
scope. Keeping the two in separate modules is deliberate: they were conflated
in `reference_fit.py`, which read Comfy's native `process_qwen2vl_images`
default as though it were the ceiling for every deployment.
"""

from __future__ import annotations

import math

from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE

SIZE_POLICIES = ("match", "max")

__all__ = [
    "CANVAS_MULTIPLE",
    "REF_IMAGE_SHORT_EDGE",
    "SIZE_POLICIES",
    "fit_reference_image",
    "latent_rows",
    "snap_to_multiple",
]


def snap_to_multiple(value: float, scale: float = 1.0) -> int:
    """Round `value * scale` to the canvas multiple, never below one step."""
    return max(CANVAS_MULTIPLE,
               round(value * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def fit_reference_image(
    source_w: int, source_h: int, *, size_policy: str = "max",
    short_edge: int = REF_IMAGE_SHORT_EDGE, allow_upscale: bool = False,
    canvas_w: int | None = None, canvas_h: int | None = None,
) -> tuple[int, int]:
    """Return the `(width, height)` a reference image is prepared at.

    `match` sizes from the target canvas area, which is why `canvas_w` and
    `canvas_h` are required for it and meaningless for `max`. `max` sizes from
    `short_edge` alone and is therefore resolvable without a canvas -- that
    asymmetry is why the sizing knobs can live on the append node while the
    resize itself happens at the conditioner.

    `allow_upscale` applies to `max` ONLY. False reproduces ComfyUI, which
    clamps its scale with `min(1.0, ...)` and only ever shrinks; True
    reproduces the three serving implementations, which upscale
    unconditionally. `docs/h3_references.md` owns what that costs. Under
    `match` the flag is ignored -- see the branch below for why.
    """
    if size_policy not in SIZE_POLICIES:
        raise ValueError(
            f"unknown image size policy {size_policy!r}; "
            f"expected one of {SIZE_POLICIES}")
    if source_w < 1 or source_h < 1:
        raise ValueError(f"reference image has no area: {source_w}x{source_h}")

    if size_policy == "match":
        if not canvas_w or not canvas_h:
            raise ValueError(
                "size_policy='match' sizes a reference from the target canvas "
                "area, so canvas_w and canvas_h are required. 'max' is the "
                "canvas-independent policy.")
        # **`match` never enlarges, whatever `allow_upscale` says.** Core
        # clamps with `min(1.0, ...)` in BOTH of its modes, and `match` means
        # "cap at the target pixel area" -- a cap that could raise a small
        # reference is not a cap. Honouring `allow_upscale` here would also
        # silently change what every pre-fold saved graph does, since before
        # the fold this branch had no upscale knob to read at all. The append
        # node warns when the two are combined; this is what that warning is
        # telling the truth about.
        full = math.sqrt((canvas_w * canvas_h) / (source_w * source_h))
        scale = min(1.0, full)
    else:
        if short_edge < CANVAS_MULTIPLE:
            raise ValueError(
                f"short_edge must be at least {CANVAS_MULTIPLE}, got {short_edge}")
        full = short_edge / min(source_w, source_h)
        scale = full if allow_upscale else min(1.0, full)

    return snap_to_multiple(source_w, scale), snap_to_multiple(source_h, scale)


def latent_rows(width: int, height: int) -> int:
    """Packed rows a reference of this pixel size contributes to the DiT.

    Two stages, not one. The VAE compresses space by 16, then the DiT
    patchifies that latent with `patch_size=(1, 2, 2)` before anything is
    attended (`comfy/ldm/minimax/model.py`, `patchify_video`). Counting only
    the VAE stage reports four times what the sequence actually carries.

    These are not Qwen vision tokens. The two coincide below the selected
    still-image policy's ceiling and diverge above it.
    """
    return (height // 32) * (width // 32)
