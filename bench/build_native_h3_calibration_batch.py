#!/usr/bin/env python3
"""Build the exact batches `oneshot` will consume, from the installed H3 path.

Gate 1 of `canonical/active_plan.md`. This produces one calibration *bundle*:
the Transformers-shaped batch tensors plus the full native-H3 presentation
record they were derived from, hashed field by field. It is not a launcher, it
loads no 32B weights, and it selects no calibration population.

**Nothing here reimplements presentation.** Every field that a defect could
change is produced by executing the code that already owns it:

| field | owner executed here |
|---|---|
| labels, ordering, timestamps, marker ids | `comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer` |
| vision-block length, image-pad expansion, attention mask | `comfy/sd1_clip.py::SDClipModel.process_tokens`, unmodified |
| H3 token tags | `comfy/text_encoders/minimax.py::token_tags_from_embeds_info` |
| M-RoPE inputs, DeepStack placement | `comfy/text_encoders/qwen3vl.py::Qwen3VL.build_image_inputs` |
| still resize and patch geometry | the release `Qwen2VLImageProcessor`, configured from `vendor_config/` |
| video resize and patch geometry | the release `Qwen3VLVideoProcessor`, configured from `vendor_config/` |
| upstream role sizing | `comfy_extras/nodes_minimax_h3.py` and `reference_conditioning.py` |

The one substitution is hidden *width*. The vision tower and token embedding
run at the released patch/merge/depth/DeepStack geometry with a reduced hidden
size, because every presentation field above is geometry and none of them
depends on that width. Weight-dependent claims are not made here; they belong
to `compare_transformers_comfy_layer50.py`.

Roles follow `canonical/active_plan.md`: a keyframe is placed on the resolved
H3 target canvas, an ordinary still uses `max` with no upstream upscale, and a
reference video takes the release 768-short-edge canvas rule with duration-aware
Qwen sampling and native two-frame presentation. Role, both geometry stages and
the media hash are recorded per media item, never per row.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`); it needs a GPU only
for the reduced-width tower and finishes in seconds.

    python bench/build_native_h3_calibration_batch.py --out <bundle dir>

`bench/prove_calibration_seam.py` then consumes the bundle in the pinned
`llm-compressor` environment. The bundle is the seam between two virtualenvs
that cannot import each other, and every tensor in it carries the hash the
prover re-derives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
import types
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
COMFY = REPO.parents[1]

DATASET_REPO = "StellarVoyager/H3-IR"
POOL = BENCH / "results" / "2026-08-24_h3_calibration_pool.jsonl"
BUNDLE_SCHEMA = "h3-native-calibration-bundle-v1"

IMAGE_PAD = 151655
VISION_START = 151652
VISION_END = 151653

# Reduced hidden width. Released depth, patch size, merge size and DeepStack
# indexes are read from the installed config and never reduced.
GEOMETRY_WIDTH = 64
GEOMETRY_VISION_WIDTH = 32
GEOMETRY_VISION_HEADS = 2

CONTRACT = re.compile(
    r"^H3 target contract \(authoritative\):\n(.*?)\n\nOriginal request:", re.S
)
MEDIA_LABEL = re.compile(r"<(Picture|Audio|Video) (\d+)>")

DEFAULT_FAMILIES = (
    "single-image",
    "multi-image-2-3",
    "keyframe-only",
    "keyframe-plus-reference",
    "video-reference",
)

# Deliberate defects, each named for the failure it stands in for. Every one is
# a defect that has actually shipped somewhere or that `active_plan.md` names as
# a stop condition; `bench/check_native_h3_presentation.py` requires each to
# change the presentation record, and `bench/prove_calibration_seam.py` requires
# each to fail the seam identity. A gate nobody has watched fail is not a gate.
MUTATIONS = {
    "chat-framing": "wrap the prompt in the Qwen chat template, as the completed "
                    "calibration run did",
    "first-image-only": "present only the first reference image, the rejected "
                        "preflight's slicing defect",
    "reorder-references": "reverse the declared request order",
    "timestamp-shift": "offset every video-block timestamp by half a second",
    "drop-temporal-repeat": "truncate an odd sampled-frame count instead of "
                            "repeat-padding it",
    "grid-shrink": "process stills at the current artifact's 200,704--301,056 "
                   "band instead of the release bounds",
    "drop-media": "omit the last declared media item",
    "token-tags-flip": "tag every vision position as text",
    "mm-types-zero": "emit all-zero mm_token_type_ids",
}


# --------------------------------------------------------------------------
# repository and dataset access


def _repo_package() -> types.ModuleType:
    """Import repo modules under a package name, with ComfyUI's root first.

    `bench/capture_h3_encoder_states.py` has its own copy of this because it is
    bound to one module and its code hash is recorded in the canonical layer-50
    benchmark artifacts; generalising it would invalidate that provenance for a
    refactor. The ordering matters: `docs/comfy_notes.md` records that a bare
    `import nodes` inside `comfy_extras` otherwise resolves to this repo's
    `nodes.py`.
    """
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    import nodes  # noqa: F401  ComfyUI's, imported before any comfy_extras

    name = "_h3_calibration_seam_pkg"
    if name in sys.modules:
        return sys.modules[name]
    package = types.ModuleType(name)
    package.__path__ = [str(REPO)]
    sys.modules[name] = package
    return package


def _repo_module(name: str):
    package = _repo_package()
    full = f"{package.__name__}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, REPO / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load repository module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def _dataset_root() -> tuple[Path, str]:
    spec = importlib.util.spec_from_file_location(
        "_h3_pool_builder", BENCH / "build_h3_calibration_pool.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load build_h3_calibration_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pinned_snapshot()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict:
    return {
        "sha256": _tensor_sha(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


# --------------------------------------------------------------------------
# release-declared processors


def _release_image_processor():
    """The still processor the release declares, configured from vendor_config.

    The release declares `image_processor_type: Qwen2VLImageProcessorFast`.
    That name is version-dependent -- in the two installed transformers builds
    it resolves to `Qwen2VLImageProcessor`, whose base is `TorchvisionBackend`,
    while `Qwen2VLImageProcessorPil` is a genuinely different implementation.
    So the class is selected by the observable backend and the resolution is
    recorded, per this repo's rule against branching on an implementation's
    name.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    declared = json.loads((REPO / "vendor_config" / "preprocessor_config.json").read_text())
    settings = {
        k: v for k, v in declared.items()
        if k not in ("processor_class", "image_processor_type")
    }
    processor = Qwen2VLImageProcessor(**settings)
    record = {
        "declared_type": declared["image_processor_type"],
        "resolved_class": type(processor).__name__,
        "resolved_backend": type(processor).__mro__[1].__name__,
        "size": dict(declared["size"]),
        "patch_size": declared["patch_size"],
        "temporal_patch_size": declared["temporal_patch_size"],
        "merge_size": declared["merge_size"],
        "image_mean": declared["image_mean"],
        "image_std": declared["image_std"],
        "resample": int(processor.resample),
        "config_sha256": _sha256_file(REPO / "vendor_config" / "preprocessor_config.json"),
    }
    return processor, record


def _current_artifact_image_processor():
    """The deployed artifact's snapshotted still processor, for `grid-shrink`.

    This is not an alternative v2 policy. It is the band the completed
    calibration run used -- 200,704 to 301,056 pixels -- and building a bundle
    with it is how the seam proof demonstrates that a wrong still policy is
    detected rather than absorbed.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    path = REPO / "config" / "qwen3vl_32b_minimax_h3_w4a16_awq" / "processor_config.json"
    declared = json.loads(path.read_text())["image_processor"]
    settings = {k: v for k, v in declared.items() if k != "image_processor_type"}
    processor = Qwen2VLImageProcessor(**settings)
    return processor, {
        "declared_type": declared["image_processor_type"],
        "resolved_class": type(processor).__name__,
        "resolved_backend": type(processor).__mro__[1].__name__,
        "size": dict(declared["size"]),
        "patch_size": declared["patch_size"],
        "temporal_patch_size": declared["temporal_patch_size"],
        "merge_size": declared["merge_size"],
        "image_mean": declared["image_mean"],
        "image_std": declared["image_std"],
        "resample": int(processor.resample),
        "config_sha256": _sha256_file(path),
        "note": "deployed artifact snapshot, used only by the grid-shrink mutation",
    }


def _release_video_processor():
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )

    path = REPO / "vendor_config" / "video_preprocessor_config.json"
    declared = json.loads(path.read_text())
    settings = {
        k: v for k, v in declared.items()
        if k not in ("processor_class", "video_processor_type")
    }
    processor = Qwen3VLVideoProcessor(**settings)
    record = {
        "declared_type": declared["video_processor_type"],
        "resolved_class": type(processor).__name__,
        "size": dict(declared["size"]),
        "patch_size": declared["patch_size"],
        "temporal_patch_size": declared["temporal_patch_size"],
        "merge_size": declared["merge_size"],
        "image_mean": declared["image_mean"],
        "image_std": declared["image_std"],
        "resample": int(processor.resample),
        "config_sha256": _sha256_file(path),
    }
    return processor, record


def _still_patches(image: torch.Tensor, processor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the release still processor on one `[1, H, W, 3]` float image.

    The uint8 round-and-clamp is the released serving boundary: the release
    decodes media into uint8 and resizes those pixels before its 1/255 rescale.
    `h3_awq_encoder.py::_source_image_patches` crosses the same boundary, and
    `reference_conditioning.py::_configured_qwen_video_frames` says why.
    """
    if image.ndim != 4 or image.shape[-1] != 3 or image.shape[0] != 1:
        raise ValueError(f"still must be [1,H,W,3], got {tuple(image.shape)}")
    pixels = image[0].detach().permute(2, 0, 1).to("cpu")
    if pixels.is_floating_point():
        pixels = pixels.mul(255).round().clamp_(0, 255).to(torch.uint8)
    else:
        pixels = pixels.to(torch.uint8)
    batch = processor.preprocess(pixels, return_tensors="pt")
    return batch["pixel_values"], batch["image_grid_thw"]


def _video_block_patches(frames: torch.Tensor, processor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the release video processor on one already-fitted two-frame block.

    Sampling and resizing happened upstream, at clip scope, under the release
    role policy; this call is the patchify only, so both are disabled. The
    result is compared against the installed `process_video_block` by
    `_check_video_block_agreement`.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video block must be [T,H,W,3], got {tuple(frames.shape)}")
    pixels = frames.detach().to("cpu")
    if pixels.is_floating_point():
        pixels = pixels.mul(255).round().clamp_(0, 255).to(torch.uint8)
    else:
        pixels = pixels.to(torch.uint8)
    batch = processor.preprocess(
        videos=[pixels.permute(0, 3, 1, 2)],
        do_sample_frames=False,
        do_resize=False,
        return_tensors="pt",
    )
    return batch["pixel_values_videos"], batch["video_grid_thw"]


def _check_video_block_agreement(frames: torch.Tensor, patches: torch.Tensor,
                                 grid: torch.Tensor) -> dict:
    """Independent arm: the installed Comfy block patchifier on the same pixels.

    `comfy/text_encoders/minimax.py::process_video_block` is a separate
    implementation of the same operation. Agreement is evidence that the
    release video processor is a faithful stand-in for the installed block
    path; a divergence is a finding, not a rounding detail to average away.
    """
    minimax = _comfy_minimax()
    comfy_patches, comfy_grid = minimax.process_video_block(frames.detach().cpu())
    return {
        "comfy_grid_thw": comfy_grid.cpu().tolist(),
        "release_grid_thw": grid.cpu().tolist(),
        "grids_equal": comfy_grid.cpu().tolist() == grid.cpu().tolist(),
        "shapes_equal": tuple(comfy_patches.shape) == tuple(patches.shape),
        "max_abs_delta": (
            float((comfy_patches.float() - patches.float()).abs().max())
            if tuple(comfy_patches.shape) == tuple(patches.shape) else None
        ),
    }


# --------------------------------------------------------------------------
# the installed presentation path, at released geometry and reduced width


def _comfy_minimax():
    import comfy.text_encoders.minimax as minimax

    return minimax


class _GeometryEmbedding(torch.nn.Module):
    """Stand-in token embedding. `process_tokens` uses only its shape."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, tokens, out_dtype=None):
        return torch.zeros(
            tokens.shape[0], tokens.shape[1], self.width,
            dtype=out_dtype or torch.float32, device=tokens.device,
        )


class _GeometryTransformer(torch.nn.Module):
    """The installed Qwen3-VL wiring with the released vision geometry.

    `preprocess_embed` is bound exactly as `h3_awq_encoder.py::
    install_source_processors` binds it -- same structure, release configuration
    instead of the current artifact's snapshot, because the v2 candidate's
    accepted contract is the release-declared processor.
    """

    def __init__(self, image_processor, video_processor, device: str):
        super().__init__()
        import comfy.ops
        from comfy.text_encoders.qwen3vl import (
            QWEN3VL_VISION,
            QWEN3VL_VISION_COMMON,
            Qwen3VLVisionModel,
        )

        released = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION["qwen3vl_32b"]}
        config = {
            **released,
            "out_hidden_size": GEOMETRY_WIDTH,
            "hidden_size": GEOMETRY_VISION_WIDTH,
            "intermediate_size": GEOMETRY_VISION_WIDTH * 2,
            "num_heads": GEOMETRY_VISION_HEADS,
        }
        self.geometry_config = {
            "released": {k: released[k] for k in sorted(released)},
            "reduced": {
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "num_heads": config["num_heads"],
                "out_hidden_size": config["out_hidden_size"],
            },
        }
        self.visual = Qwen3VLVisionModel(
            config, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast
        )
        generator = torch.Generator().manual_seed(0)
        self.visual.load_state_dict(
            {
                key: torch.randn(value.shape, generator=generator, dtype=torch.float32) * 0.02
                for key, value in self.visual.state_dict().items()
            },
            strict=True,
        )
        self.visual.eval().to(device)
        self._embedding = _GeometryEmbedding(GEOMETRY_WIDTH).to(device)
        self._image_processor = image_processor
        self._video_processor = video_processor
        self.blocks: list[dict] = []

    def get_input_embeddings(self):
        return self._embedding

    def preprocess_embed(self, embed, device):
        if embed.get("type") != "image":
            return None, None
        video_block = bool(embed.get("minimax_video_block", False))
        if video_block:
            patches, grid = _video_block_patches(embed["data"], self._video_processor)
            agreement = _check_video_block_agreement(embed["data"], patches, grid)
        else:
            patches, grid = _still_patches(embed["data"], self._image_processor)
            agreement = None
        merged, deepstack = self.visual(
            patches.to(device=device, dtype=torch.float32), grid.to(device)
        )
        self.blocks.append({
            "kind": "video_block" if video_block else "image",
            "patches": patches.detach().cpu(),
            "grid_thw": grid.detach().cpu(),
            "merged_tokens": int(merged.shape[0]),
            "deepstack_features": len(deepstack),
            "comfy_block_agreement": agreement,
        })
        return merged, {"grid": grid, "deepstack": deepstack}


class _PresentationHost:
    """Minimal host for the unmodified `SDClipModel.process_tokens`.

    That function reads exactly two attributes off `self`. Supplying them
    rather than constructing a 50-layer `MiniMaxH3ClipModel` keeps the real
    expansion, attention-mask and `embeds_info` logic while loading no weights.
    """

    special_tokens = {"pad": 151643}

    def __init__(self, transformer):
        self.transformer = transformer


# --------------------------------------------------------------------------
# H3-IR row -> ordered media with per-item role and both geometry stages


def parse_target_contract(row: dict) -> dict:
    user = [m for m in row["messages"] if m["role"] == "user"]
    if len(user) != 1:
        raise ValueError(f"{row['id']}: expected one user message, got {len(user)}")
    match = CONTRACT.search(user[0]["content"])
    if match is None:
        raise ValueError(f"{row['id']}: no H3 target contract in the user message")
    contract = {}
    for line in match.group(1).splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            contract[key.strip()] = value.strip()
    if "duration_seconds" not in contract or "available_media_labels" not in contract:
        raise ValueError(f"{row['id']}: target contract lacks duration or media labels")
    return contract


def declared_media_order(contract: dict) -> list[tuple[str, int]]:
    """Request order, read from the row's own `available_media_labels`.

    `canonical/native_h3_contract.md` requires reference items to stay in
    request order with an independent one-based counter per type. The dataset
    declares that order; it is not inferred from the media arrays, which are
    per-kind and cannot express interleaving.
    """
    labels = MEDIA_LABEL.findall(contract["available_media_labels"])
    if not labels:
        return []
    ordered = [(kind.lower(), int(index)) for kind, index in labels]
    for kind in ("picture", "audio", "video"):
        seen = [i for k, i in ordered if k == kind]
        if seen != list(range(1, len(seen) + 1)):
            raise ValueError(f"{kind} labels are not a 1..n run: {seen}")
    return ordered


def keyframe_declarations(row: dict) -> dict[int, str]:
    """One-based picture ordinal -> `first` or `last`, from the row's own IR."""
    return {
        int(n): where
        for n, where in re.findall(
            r"<Picture (\d+)> is the (first|last) frame", row.get("target_ir") or ""
        )
    }


def _decode_still(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).unsqueeze(0)


def _decode_clip(path: Path, max_frames: int) -> tuple[torch.Tensor, float]:
    import av
    import numpy as np

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None:
            raise ValueError(f"{path.name}: stream declares no average frame rate")
        loaded_fps = float(rate)
        for frame in container.decode(stream):
            frames.append(np.asarray(frame.to_rgb().to_ndarray(), dtype=np.uint8))
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"{path.name}: decoded zero frames")
    return torch.from_numpy(np.stack(frames)), loaded_fps


def _size_keyframe(image: torch.Tensor, where: str, geometry_nodes) -> tuple[torch.Tensor, dict]:
    """Place a keyframe on the resolved H3 target canvas.

    `MiniMaxH3ImageToVideo` stretches the first frame to the canvas and
    cover-crops the last one; a keyframe stands in for generated rows and
    shares the target grid, so canvas geometry is structural rather than
    conventional (`canonical/2026-08-24_keyframe_vs_reference_positioning.md`).
    """
    source_h, source_w = int(image.shape[1]), int(image.shape[2])
    canvas_w, canvas_h = geometry_nodes.adapt_canvas(source_w, source_h)
    crop = "disabled" if where == "first" else "center"
    resized = geometry_nodes._resize(image[:1].float() / 255.0, canvas_w, canvas_h, crop)
    return resized, {
        "policy": f"keyframe-{where}",
        "rule": "adapt_canvas then _resize",
        "crop": crop,
        "source": [source_w, source_h],
        "upstream": [canvas_w, canvas_h],
    }


STILL_POLICIES = ("max_no_upscale", "upscale_2048")


def _size_reference_still(image: torch.Tensor, geometry_nodes,
                          policy: str = "max_no_upscale") -> tuple[torch.Tensor, dict]:
    """The accepted v2 still policy, or the separately named stress stratum.

    `max_no_upscale` is the primary policy: the scale is `min(1.0, ...)`, so it
    is a ceiling at a 2048 short edge and never an upscale, and
    `reference_conditioning.py::_compile_reference_records` computes it the same
    way for the deployed path.

    `upscale_2048` is the stress stratum `active_plan.md` keeps separately
    named -- the 2048-short-edge serving convention with upstream upscaling
    allowed, which `MiniMaxH3ReferenceFit(allow_upscale=True)` performs on the
    deployed path. It is not an alternative primary policy and must not be
    mixed into one: it manufactures pixels, and its whole purpose is to measure
    the serving convention's cost without letting interpolated large references
    dominate the mix.
    """
    import math

    if policy not in STILL_POLICIES:
        raise ValueError(f"unknown still policy {policy!r}; expected {STILL_POLICIES}")
    source_h, source_w = int(image.shape[1]), int(image.shape[2])
    ratio = geometry_nodes.REF_IMAGE_SHORT_EDGE / min(source_w, source_h)
    scale = ratio if policy == "upscale_2048" else min(1.0, ratio)
    multiple = geometry_nodes.CANVAS_MULTIPLE
    target_w = max(multiple, round(source_w * scale / multiple) * multiple)
    target_h = max(multiple, round(source_h * scale / multiple) * multiple)
    source = image[:1].float() / 255.0
    # `_resize` is called even when the geometry is unchanged, because
    # `_compile_reference_records` calls it unconditionally and
    # `comfy.utils.common_upscale` has no identity short-circuit: its lanczos
    # pass runs anyway. Skipping it here would build a calibration input the
    # deployed path never produces, which is the defect class this gate exists
    # to catch. `identity_resample_max_abs_delta` records what that pass costs.
    resized = geometry_nodes._resize(source, target_w, target_h, "disabled")
    unchanged = (target_w, target_h) == (source_w, source_h)
    return resized, {
        "policy": f"reference-still-{policy.replace('_', '-')}",
        "rule": ("REF_IMAGE_SHORT_EDGE / short_edge" if policy == "upscale_2048"
                 else "min(1.0, REF_IMAGE_SHORT_EDGE / short_edge)")
                + ", round to CANVAS_MULTIPLE, then the unconditional _resize "
                  "the deployed reference compiler performs",
        "upscaling_allowed": policy == "upscale_2048",
        "scale": scale,
        "source": [source_w, source_h],
        "upstream": [target_w, target_h],
        "resized_upstream": not unchanged,
        "inside_ceiling": math.isclose(scale, 1.0),
        "identity_resample_max_abs_delta": (
            float((resized - source).abs().max()) if unchanged else None
        ),
    }


def _size_reference_video(clip: torch.Tensor, loaded_fps: float, duration_seconds: float,
                          geometry_nodes, reference_conditioning) -> tuple[torch.Tensor, list[float], dict]:
    """Release role policy, executed through the deployed reference code.

    `_prepare_reference_video` normalises to 24 fps and snaps the length to the
    model's 17k+5 grid; `adapt_canvas` supplies the 768-short-edge /
    1,032,192-pixel canvas; sampling steps at `FPS // 2` for the 2 fps Qwen
    view; `_release_qwen_video_frames` applies the release video processor's
    duration-aware `smart_resize` and its bicubic kernel.
    """
    frame_count = geometry_nodes.align_frame_count(
        max(5, int(round(duration_seconds * geometry_nodes.FPS)))
    )
    prepared = reference_conditioning._prepare_reference_video(clip, loaded_fps, frame_count)
    source_h, source_w = int(prepared.shape[1]), int(prepared.shape[2])
    canvas_w, canvas_h = geometry_nodes.adapt_canvas(source_w, source_h)
    fitted = geometry_nodes._resize(
        prepared.float() / 255.0, canvas_w, canvas_h, "disabled"
    )
    step = geometry_nodes.FPS // 2
    indices = list(range(0, int(fitted.shape[0]), step))
    sampled = fitted[indices]
    qwen_frames = reference_conditioning._release_qwen_video_frames(sampled)
    timestamps = [i / 2.0 for i in range(len(indices))]
    return qwen_frames, timestamps, {
        "policy": "reference-video-release",
        "rule": "align_frame_count, _prepare_reference_video, adapt_canvas, "
                "2 fps sample, release Qwen smart_resize",
        "target_frame_count": frame_count,
        "loaded_fps": loaded_fps,
        "decoded_frames": int(clip.shape[0]),
        "prepared_frames": int(prepared.shape[0]),
        "source": [source_w, source_h],
        "upstream": [canvas_w, canvas_h],
        "sampled_frames": len(indices),
        "sample_indices": indices,
        "qwen_view": [int(qwen_frames.shape[2]), int(qwen_frames.shape[1])],
        "timestamps": timestamps,
    }


def build_ordered_media(row: dict, contract: dict, root: Path,
                        geometry_nodes, reference_conditioning,
                        still_policy: str = "max_no_upscale") -> list[dict]:
    """Ordered reference items with per-item role, geometry and media hash."""
    order = declared_media_order(contract)
    keyframes = keyframe_declarations(row)
    duration = float(contract["duration_seconds"])
    images = list(row.get("images") or [])
    videos = list(row.get("videos") or [])
    declared_hashes = row.get("media_sha256") or {}

    items = []
    for kind, ordinal in order:
        if kind == "audio":
            items.append({
                "type": "audio",
                "label": f"<Audio {ordinal}>",
                "role": "reference-audio",
                "note": "text label only; no audio tensor reaches Qwen",
            })
            continue
        if kind == "picture":
            if ordinal > len(images):
                raise ValueError(f"{row['id']}: <Picture {ordinal}> has no media file")
            relative = images[ordinal - 1]
            path = root / relative
            decoded = _decode_still(path)
            where = keyframes.get(ordinal)
            if where is None:
                sized, geometry = _size_reference_still(
                    decoded, geometry_nodes, still_policy
                )
                role = "reference-still"
            else:
                sized, geometry = _size_keyframe(decoded, where, geometry_nodes)
                role = f"keyframe-{where}"
            items.append({
                "type": "image",
                "label": f"<Picture {ordinal}>",
                "role": role,
                "media_path": relative,
                "declared_sha256": declared_hashes.get(relative),
                "file_sha256": _sha256_file(path),
                "decoded": [int(decoded.shape[2]), int(decoded.shape[1])],
                "geometry": geometry,
                "data": sized,
            })
            continue
        if ordinal > len(videos):
            raise ValueError(f"{row['id']}: <Video {ordinal}> has no media file")
        relative = videos[ordinal - 1]
        path = root / relative
        cap = int(round(duration * 120)) + 240
        clip, loaded_fps = _decode_clip(path, cap)
        frames, timestamps, geometry = _size_reference_video(
            clip, loaded_fps, duration, geometry_nodes, reference_conditioning
        )
        items.append({
            "type": "video",
            "label": f"<Video {ordinal}>",
            "role": "reference-video",
            "media_path": relative,
            "declared_sha256": declared_hashes.get(relative),
            "file_sha256": _sha256_file(path),
            "decoded": [int(clip.shape[2]), int(clip.shape[1])],
            "geometry": geometry,
            "data": frames,
            "timestamps": timestamps,
        })
    return items


# --------------------------------------------------------------------------
# presentation -> Transformers batch


def _entries(tokens: dict) -> list:
    if set(tokens) != {"qwen3vl_32b"}:
        raise ValueError(f"unexpected tokenizer keys: {sorted(tokens)}")
    batches = tokens["qwen3vl_32b"]
    if len(batches) != 1:
        raise ValueError(f"expected one token batch, got {len(batches)}")
    return [row[0] for row in batches[0]]


def _expanded_ids(entries: list, embeds_info: list[dict]) -> list[int]:
    import numbers

    out: list[int] = []
    index = 0
    for entry in entries:
        if isinstance(entry, numbers.Integral):
            out.append(int(entry))
            continue
        if not isinstance(entry, dict) or index >= len(embeds_info):
            raise ValueError("tokenizer placeholder does not match processed embeds")
        out.extend([IMAGE_PAD] * int(embeds_info[index]["size"]))
        index += 1
    if index != len(embeds_info):
        raise ValueError("processed embeds remain after token expansion")
    return out


def _vision_spans(tags: list[int]) -> list[list[int]]:
    spans, start = [], None
    for index, tag in enumerate(tags):
        if tag == 0 and start is None:
            start = index
        elif tag != 0 and start is not None:
            spans.append([start, index - 1])
            start = None
    if start is not None:
        spans.append([start, len(tags) - 1])
    return spans


def _mm_token_type_ids(ids: list[int]) -> list[int]:
    """text 0 / image 1 / video 2, the derivation the release processor uses.

    Every H3 vision block -- still and two-frame video alike -- is image-keyed:
    `process_video_block` emits a `grid_t = 1` block on the image keys, and
    `canonical/2026-08-24_calibration_input_seam.md` measured that transformers
    labels it modality 1. So there is no type-2 position in an H3 batch, and
    `prove_calibration_seam.py` checks that against the release processor's own
    output rather than against this comment.
    """
    return [1 if token == IMAGE_PAD else 0 for token in ids]


def _scaffold_text(entries: list, inverse: dict) -> str:
    """Decoded text of everything up to the last vision block.

    This is the ordered-label, timestamp and marker scaffold -- the part a
    presentation defect changes. The prompt body that follows is covered by
    `prompt_sha256` and `expanded_token_ids_sha256`; repeating several kilobytes
    of it in every record would bury the field the reader came for.
    """
    last = -1
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            last = index
    if last < 0:
        return ""
    return "".join(
        inverse.get(e, "") for e in entries[: last + 2] if not isinstance(e, dict)
    )


CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def build_row(row: dict, contract: dict, root: Path, host, transformer,
              tokenizer, geometry_nodes, reference_conditioning,
              mutation: str | None = None,
              still_policy: str = "max_no_upscale") -> tuple[dict, dict]:
    import comfy.sd1_clip

    prompt = row.get("target_ir") or ""
    if not prompt:
        raise ValueError(f"{row['id']}: empty target_ir")
    if mutation == "chat-framing":
        prompt = CHAT_TEMPLATE.format(prompt)

    items = build_ordered_media(row, contract, root, geometry_nodes,
                                reference_conditioning, still_policy)
    if mutation == "reorder-references":
        items = list(reversed(items))
    elif mutation == "drop-media":
        media = [i for i, item in enumerate(items) if item["type"] != "audio"]
        if media:
            items = [item for i, item in enumerate(items) if i != media[-1]]
    elif mutation == "first-image-only":
        first = next((item for item in items if item["type"] == "image"), None)
        items = [first] if first is not None else []

    ref_items = []
    for item in items:
        if item["type"] == "audio":
            ref_items.append({"type": "audio"})
        elif item["type"] == "image":
            ref_items.append({"type": "image", "data": item["data"]})
        else:
            frames, stamps = item["data"], list(item["timestamps"])
            if mutation == "timestamp-shift":
                stamps = [t + 0.5 for t in stamps]
            elif mutation == "drop-temporal-repeat" and frames.shape[0] % 2 == 1:
                frames, stamps = frames[:-1], stamps[:-1]
            ref_items.append({"type": "video", "data": frames, "timestamps": stamps})

    tokens = tokenizer.tokenize_with_weights(prompt, minimax_ref_items=ref_items)
    entries = _entries(tokens)

    device = next(transformer.visual.parameters()).device
    transformer.blocks = []
    with torch.no_grad():
        embeds, attention, num_tokens, embeds_info = comfy.sd1_clip.SDClipModel.process_tokens(
            host, [entries], device
        )
        position_ids, visual_mask, deepstack = transformer_build_image_inputs(
            transformer, embeds, embeds_info
        )
    blocks = transformer.blocks

    tags = _comfy_minimax().token_tags_from_embeds_info(int(embeds.shape[1]), embeds_info).tolist()
    if mutation == "token-tags-flip":
        tags = [1] * len(tags)
    expanded = _expanded_ids(entries, embeds_info)
    if not (len(expanded) == len(tags) == int(embeds.shape[1])):
        raise ValueError(
            f"{row['id']}: expanded ids/tags/embeds disagree: "
            f"{len(expanded)}, {len(tags)}, {int(embeds.shape[1])}"
        )
    mask = attention.detach().cpu()[0].tolist()
    if num_tokens != [len(expanded)]:
        raise ValueError(f"{row['id']}: row carries masked or padded positions")
    if len(blocks) != len(embeds_info):
        raise ValueError(f"{row['id']}: {len(blocks)} vision blocks, {len(embeds_info)} embeds")

    pixel_values = torch.cat([b["patches"] for b in blocks], dim=0) if blocks else None
    image_grid_thw = torch.cat([b["grid_thw"] for b in blocks], dim=0) if blocks else None
    mm_types = _mm_token_type_ids(expanded)
    if mutation == "mm-types-zero":
        mm_types = [0] * len(mm_types)

    batch = {
        "input_ids": torch.tensor([expanded], dtype=torch.long),
        "attention_mask": torch.tensor([mask], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([mm_types], dtype=torch.int),
    }
    if pixel_values is not None:
        batch["pixel_values"] = pixel_values.contiguous()
        batch["image_grid_thw"] = image_grid_thw.to(torch.long).contiguous()

    marker_ids = set(range(151669, 151676))
    inverse = {v: k for k, v in tokenizer.qwen3vl_32b.tokenizer.get_vocab().items()}
    record = {
        "row_id": row["id"],
        "source": DATASET_REPO,
        "mutation": mutation,
        "primary_role": row.get("_primary_role"),
        "target_contract": contract,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "ordered_media": [
            {k: v for k, v in item.items() if k != "data"} for item in items
        ],
        "labels_in_order": [item["label"] for item in items],
        "pre_expansion_entries": [
            int(e) if not isinstance(e, dict) else {
                "type": e.get("type"),
                "minimax_video_block": bool(e.get("minimax_video_block", False)),
            }
            for e in entries
        ],
        "presentation_scaffold": _scaffold_text(entries, inverse),
        "expanded_token_ids_sha256": _json_sha(expanded),
        "sequence_length": len(expanded),
        "token_tags_sha256": _json_sha(tags),
        "vision_spans": _vision_spans(tags),
        "text_positions": sum(1 for t in tags if t == 1),
        "vision_positions": sum(1 for t in tags if t == 0),
        "marker_positions": [i for i, t in enumerate(expanded) if t in marker_ids],
        "marker_ids_present": sorted({t for t in expanded if t in marker_ids}),
        "attention_mask_sha256": _json_sha(mask),
        "mm_token_type_ids_sha256": _json_sha(mm_types),
        "embeds_info": [
            {"type": e.get("type"), "index": int(e["index"]), "size": int(e["size"])}
            for e in embeds_info
        ],
        "vision_blocks": [
            {
                "kind": b["kind"],
                "grid_thw": b["grid_thw"].tolist(),
                "merged_tokens": b["merged_tokens"],
                "deepstack_features": b["deepstack_features"],
                "patches": _tensor_record(b["patches"]),
                "comfy_block_agreement": b["comfy_block_agreement"],
            }
            for b in blocks
        ],
        "position_ids_sha256": _tensor_sha(position_ids) if position_ids is not None else None,
        "position_ids_shape": list(position_ids.shape) if position_ids is not None else None,
        "visual_mask_sha256": _tensor_sha(visual_mask) if visual_mask is not None else None,
        "deepstack_feature_count": len(deepstack) if deepstack else 0,
        "deepstack_feature_shapes": [list(d.shape) for d in deepstack] if deepstack else [],
        "batch_tensors": {k: _tensor_record(v) for k, v in batch.items()},
    }

    # The upstream-sized media travel with the bundle as uint8, the dtype the
    # release processors consume. Without them the prover could only re-hash the
    # patch tensors this builder produced, which is checking an implementation
    # against itself; with them it can re-derive `pixel_values` from an
    # independent processor arm on the same pixels.
    media: dict[str, torch.Tensor] = {}
    for index, item in enumerate(items):
        data = item.get("data")
        if not torch.is_tensor(data):
            continue
        pixels = data.detach().cpu()
        if pixels.is_floating_point():
            pixels = pixels.mul(255).round().clamp_(0, 255).to(torch.uint8)
        key = f"item{index:02d}_{item['type']}"
        media[key] = pixels.contiguous()
        record["ordered_media"][index]["upstream_media_key"] = key
        record["ordered_media"][index]["upstream_media"] = _tensor_record(media[key])
    return batch, record, media


def transformer_build_image_inputs(transformer, embeds, embeds_info):
    """Call the installed `Qwen3VL.build_image_inputs`, unbound.

    It reads only `embeds_info` and the embedding shape, so it produces the
    real Comfy M-RoPE position ids, visual-position mask and DeepStack
    concatenation without any language weights.
    """
    from comfy.text_encoders.qwen3vl import Qwen3VL

    return Qwen3VL.build_image_inputs(transformer, embeds, embeds_info)


# --------------------------------------------------------------------------


def _git_commit(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _provenance(revision: str, image_record: dict, video_record: dict) -> dict:
    import transformers

    return {
        "schema": BUNDLE_SCHEMA,
        "path_policy": "logical identifiers only; no machine-local paths",
        "dataset": {"repo_id": DATASET_REPO, "revision": revision},
        "repository_commit": _git_commit(REPO),
        "comfyui_commit": _git_commit(COMFY),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "processors": {"image": image_record, "video": video_record},
        "geometry_substitution": (
            "vision tower and token embedding run at released patch/merge/depth/"
            "DeepStack geometry with a reduced hidden width; no presentation "
            "field depends on that width and no weight-dependent claim is made"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="new bundle directory; must not exist")
    parser.add_argument("--row", action="append", help="H3-IR row id; repeatable")
    parser.add_argument(
        "--family", action="append",
        help=f"pool primary role; defaults to {', '.join(DEFAULT_FAMILIES)}",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--still-policy", choices=STILL_POLICIES, default="max_no_upscale",
        help="primary accepted policy, or the separately named 2048-upscale "
             "stress stratum",
    )
    parser.add_argument(
        "--mutate", choices=sorted(MUTATIONS),
        help="build a deliberately defective bundle; see MUTATIONS",
    )
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refuse to overwrite existing bundle directory: {out}")

    root, revision = _dataset_root()
    pool = [json.loads(line) for line in POOL.read_text().splitlines()]
    by_id = {r["id"]: r for r in pool}

    wanted: list[str] = []
    if args.row:
        for row_id in args.row:
            if row_id not in by_id:
                raise SystemExit(f"row {row_id} is not in the accepted pool")
            wanted.append(row_id)
    families = args.family or list(DEFAULT_FAMILIES)
    if not args.row:
        for family in families:
            members = sorted(
                (r["id"] for r in pool if r["primary_role"] == family)
            )
            if not members:
                raise SystemExit(f"no pooled row has primary role {family!r}")
            wanted.append(members[0])

    raw = {}
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in wanted:
            raw[row["id"]] = row
    missing = [row_id for row_id in wanted if row_id not in raw]
    if missing:
        raise SystemExit(f"rows absent from the pinned snapshot: {missing}")

    reference_conditioning = _repo_module("reference_conditioning")
    import comfy_extras.nodes_minimax_h3 as geometry_nodes

    if args.mutate == "grid-shrink":
        image_processor, image_record = _current_artifact_image_processor()
    else:
        image_processor, image_record = _release_image_processor()
    video_processor, video_record = _release_video_processor()
    transformer = _GeometryTransformer(image_processor, video_processor, args.device)
    host = _PresentationHost(transformer)
    tokenizer = _comfy_minimax().MiniMaxH3Tokenizer()

    out.mkdir(parents=True)
    from safetensors.torch import save_file

    records = []
    for row_id in wanted:
        row = dict(raw[row_id])
        row["_primary_role"] = by_id[row_id]["primary_role"]
        contract = parse_target_contract(row)
        batch, record, media = build_row(
            row, contract, root, host, transformer, tokenizer,
            geometry_nodes, reference_conditioning, mutation=args.mutate,
            still_policy=args.still_policy,
        )
        name = f"batch-{row_id}.safetensors"
        save_file({k: v.contiguous() for k, v in batch.items()}, out / name)
        record["batch_file"] = name
        record["batch_file_sha256"] = _sha256_file(out / name)
        if media:
            media_name = f"media-{row_id}.safetensors"
            save_file(media, out / media_name)
            record["media_file"] = media_name
            record["media_file_sha256"] = _sha256_file(out / media_name)
        records.append(record)
        blocks = record["vision_blocks"]
        print(
            f"{row_id} [{record['primary_role']}] len={record['sequence_length']} "
            f"vision={record['vision_positions']} text={record['text_positions']} "
            f"blocks={len(blocks)} grids={[b['grid_thw'][0][1:] for b in blocks]}",
            flush=True,
        )

    provenance = _provenance(revision, image_record, video_record)
    provenance["still_policy"] = args.still_policy
    provenance["mutation"] = args.mutate
    provenance["mutation_intent"] = MUTATIONS.get(args.mutate) if args.mutate else None
    manifest = {"provenance": provenance, "order": wanted, "rows": records}
    (out / "presentation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(records)} batches and presentation.json to the bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
