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
what the selected Qwen still-image policy then does to it -- is
`qwen_image_settings` below, and its `encoder` branch depends on which
encoder artifact is loaded. That is knowable only where the CLIP is in scope,
which is why the branch takes an explicit *contract* rather than reading a
module: until 2026-08-25 it read the current W4 artifact's snapshot whichever
CLIP the graph had loaded, so a stock `CLIPLoader` graph on `encoder` was
priced and pre-sized at bounds no loaded encoder declared. Keeping the two
stages in separate modules is deliberate: they were conflated in
`reference_fit.py`, which read Comfy's native `process_qwen2vl_images` default
as though it were the ceiling for every deployment.

**The contract.** `h3_awq_encoder.install_source_processors` stamps the
loaded artifact's declaration on the CLIP's transformer as
`_h3_encoder_contract`; `encoder_contract_from_clip` reads it back and
`effective_policy` says what `encoder` means when there is none: the native
path, because that is what a CLIP that declares nothing actually runs.
"""

from __future__ import annotations

import math

from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE

SIZE_POLICIES = ("match", "max")

ENCODER_CONTRACT_KEYS = ("source", "image_bounds", "image_geometry",
                         "video_bounds", "video_geometry")

__all__ = [
    "CANVAS_MULTIPLE",
    "REF_IMAGE_SHORT_EDGE",
    "ENCODER_CONTRACT_KEYS",
    "IMAGE_POLICIES",
    "SIZE_POLICIES",
    "effective_policy",
    "encoder_contract_from_clip",
    "fit_reference_image",
    "qwen_image_settings",
    "qwen_image_size",
    "latent_rows",
    "snap_to_multiple",
]


def encoder_contract_from_clip(clip) -> dict | None:
    """What the LOADED encoder declares, read off the CLIP. `None` is native.

    Branches on the observable, not on which loader node the user picked:
    the adapter stamps `_h3_encoder_contract` on the transformer it builds,
    and a CLIP from core's `CLIPLoader` carries no such attribute. A stamped
    contract missing a key is refused rather than partially applied.
    """
    model = clip
    for attribute in ("cond_stage_model", "qwen3vl_32b", "transformer"):
        model = getattr(model, attribute, None)
        if model is None:
            return None
    contract = getattr(model, "_h3_encoder_contract", None)
    if contract is None:
        return None
    missing = [key for key in ENCODER_CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(
            f"the loaded encoder's processor contract is missing {missing}; "
            "refusing to apply a partial declaration")
    return dict(contract)


def effective_policy(policy: str, contract: dict | None) -> str:
    """`encoder` on a CLIP that declares nothing IS the native path.

    Every other policy is its own answer. This is the only place that
    substitution is made, so a caller can log that it happened.
    """
    if policy == "encoder" and contract is None:
        return "comfy"
    return policy


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


def _vendor_config():
    """The release's declared processor configs, imported lazily.

    Lazily and both ways: this module is imported as a package member by the
    nodes and as a top-level module by `bench/preflight_graph.py` and
    `bench/count_packed_rows.py`, and a caller that only wants
    `fit_reference_image` should not pay for the import.
    """
    try:
        from . import vendor_config
    except ImportError:  # pragma: no cover - top-level import for the tools
        import vendor_config  # type: ignore[no-redef]
    return vendor_config


def qwen_image_settings(
    image_policy: str, contract: dict | None = None,
) -> tuple[tuple[int, int], dict]:
    """Return the bounds and geometry owned by the selected STILL policy.

    One ceiling has three live values and nothing could select between them:
    the installed ComfyUI code path's `process_qwen2vl_images` defaults, the
    loaded encoder artifact's declaration, and the release's declaration.
    `reference_fit.py` read the first by introspection and applied it as though
    it were universal, which is right for a native BF16 graph and wrong by
    orders of magnitude under the AWQ adapter.

    `comfy` returns nothing to apply: it is the passthrough that leaves the
    still exactly as core would, which is what every graph got before this
    existed and therefore what the default has to be.

    `encoder` needs the loaded encoder's `contract`
    (`encoder_contract_from_clip`), and refuses without one. A CLIP that
    declares nothing is resolved to `comfy` by `effective_policy` before this
    is reached; asking for encoder settings with no contract is a caller
    that skipped that step, not a case to paper over with a default.
    """
    if image_policy == "comfy":
        raise ValueError(
            "the comfy still policy has no configured processor; callers must "
            "skip the Qwen stage entirely rather than ask for its settings")
    if image_policy == "release":
        vendor_config = _vendor_config()
        return vendor_config.image_pixel_bounds(), vendor_config.patch_geometry()
    if image_policy == "encoder":
        if contract is None:
            raise ValueError(
                "the encoder still policy needs the loaded encoder's contract "
                "(reference_geometry.encoder_contract_from_clip); with none, "
                "resolve the policy through effective_policy first")
        return tuple(contract["image_bounds"]), dict(contract["image_geometry"])
    raise ValueError(f"no configured Qwen processor for policy {image_policy!r}")


def qwen_image_size(
    width: int, height: int, image_policy: str, contract: dict | None = None,
) -> tuple[int, int]:
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

    (min_pixels, max_pixels), geometry = qwen_image_settings(image_policy, contract)
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
