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
    "IMAGE_POLICIES",
    "SIZE_POLICIES",
    "fit_reference_image",
    "qwen_image_settings",
    "qwen_image_size",
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


IMAGE_POLICIES = ("comfy", "release", "encoder")


def _policy_config():
    """The two declared still-image processor configs, imported lazily.

    Lazily and both ways: this module is imported as a package member by the
    nodes and as a top-level module by `bench/preflight_graph.py` and
    `bench/count_packed_rows.py`. `h3_awq_encoder` also pulls in `comfy_api`,
    which a caller that only wants `fit_reference_image` should not pay for.
    """
    try:
        from . import h3_awq_encoder, vendor_config
    except ImportError:  # pragma: no cover - top-level import for the tools
        import h3_awq_encoder  # type: ignore[no-redef]
        import vendor_config  # type: ignore[no-redef]
    return vendor_config, h3_awq_encoder


def qwen_image_settings(image_policy: str) -> tuple[tuple[int, int], dict]:
    """Return the bounds and geometry owned by the selected STILL policy.

    One ceiling has three live values and nothing could select between them:
    the installed ComfyUI code path's `process_qwen2vl_images` defaults, the
    loaded encoder artifact's snapshot, and the release's declaration.
    `reference_fit.py` read the first by introspection and applied it as though
    it were universal, which is right for a native BF16 graph and wrong by
    orders of magnitude under the AWQ adapter.

    `comfy` returns nothing to apply: it is the passthrough that leaves the
    still exactly as core would, which is what every graph got before this
    existed and therefore what the default has to be.
    """
    if image_policy == "comfy":
        raise ValueError(
            "the comfy still policy has no configured processor; callers must "
            "skip the Qwen stage entirely rather than ask for its settings")
    vendor_config, h3_awq_encoder = _policy_config()
    if image_policy == "release":
        return vendor_config.image_pixel_bounds(), vendor_config.patch_geometry()
    if image_policy == "encoder":
        return (h3_awq_encoder.source_image_pixel_bounds(),
                h3_awq_encoder.source_image_patch_geometry())
    raise ValueError(f"no configured Qwen processor for policy {image_policy!r}")


def qwen_image_size(width: int, height: int, image_policy: str) -> tuple[int, int]:
    """Return the selected still policy's Qwen view as ``(width, height)``.

    Pre-applying this is what puts both towers on one size. Core hands ONE
    tensor to the VAE and stashes the SAME object for the conditioner, so a
    Qwen-side resize that fires after the VAE has already encoded leaves the
    DiT holding latent rows at one resolution and hidden states at another for
    a single reference, silently.

    `smart_resize` is imported from the installed processor rather than copied,
    and it enforces a FLOOR as well as a ceiling: a still under the policy's
    `min_pixels` is enlarged by it. That is a real behaviour of the declared
    policy, not a bug to clamp away, and callers log it.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    (min_pixels, max_pixels), geometry = qwen_image_settings(image_policy)
    factor = int(geometry["patch_size"]) * int(geometry["merge_size"])
    target_h, target_w = smart_resize(
        height=height, width=width, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    return int(target_w), int(target_h)


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
