#!/usr/bin/env python3
"""Capture controlled MiniMax H3 layer-50 states from BF16 or current W4.

Run each arm in a separate process so the 32B encoders are never resident
together.  The supported modes are isolated weight comparisons from
``canonical/native_h3_contract.md``: both arms use one explicitly selected
processor policy and the installed native-H3 presentation.

Examples (ComfyUI venv, free GPU):

    python bench/capture_h3_encoder_states.py \
        --arm bf16 --processor-policy current_w4 --out /capture/bf16
    python bench/capture_h3_encoder_states.py \
        --arm w4 --processor-policy current_w4 --out /capture/w4
    python bench/compare_h3_encoder_captures.py \
        --reference /capture/bf16 --candidate /capture/w4 --out result.json

The built-in fixture population is deliberately a controlled substrate, not a
claim of corpus representativeness.  It covers raw T2VA, one-image I2VA,
two-image FL2VA, ordered image/audio/image Ref2VA, a real two-frame temporal
presentation, and dedicated dialogue ids.  Keyframes and video use the
1344x768 canvas geometry; Ref2VA stills use the independently implemented
2048-short-edge serving geometry.  Pixel tensors are deterministic coordinate
patterns; no random or dummy media enter a capture.

BF16 is loaded into the same installed ComfyUI 50-layer H3 implementation as
W4.  Only the source tensors needed by that implementation are mapped from the
14-shard Qwen3-VL directory.  This avoids comparing Transformers against Comfy
or layer 64 against layer 50.  W4 is loaded through the shipped repository
adapter.  Both then receive the adapter's exact source processor functions.

The manifest records the actual post-preprocessing visual patches and the full
pre-language-layer embedding tensor.  The comparator refuses metrics unless
those hashes and every presentation field match across arms.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import numbers
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
SCHEMA = "h3-layer50-capture-v1"
IMAGE_PAD = 151655
CURRENT_CONFIG_DIR = (
    REPO / "config" / "qwen3vl_32b_minimax_h3_w4a16_awq"
)
RELEASE_IMAGE_CONFIG = REPO / "vendor_config" / "preprocessor_config.json"
PROCESSOR_POLICIES = ("current_w4", "current_w4_release_image_bounds")
W4_DEFAULT = (
    COMFY
    / "models"
    / "text_encoders"
    / "qwen3vl_32b_minimax_h3_w4a16_awq.safetensors"
)
REPEAT_FIXTURES = {"t2va_text", "i2va_single_image"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict:
    return {
        "sha256": _tensor_sha(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _repo_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError as exc:
        raise ValueError(f"provenance path is outside the repository: {path.name}") from exc


def _source_locator(arm: str, source: Path) -> dict:
    """Portable model identity without storing a machine-local path."""
    resolved = source.resolve()
    return {
        "kind": "sharded_bf16_directory" if arm == "bf16" else "w4_safetensors",
        "logical_name": resolved.name,
        "selected_name": source.name,
        "selected_via_symlink": source.is_symlink(),
    }


def _processor_policy_spec(name: str) -> tuple[dict, dict]:
    """Return the effective image config and its complete provenance.

    The release-bounds arm deliberately changes only ``size`` on the current
    artifact's real Qwen2VLImageProcessor configuration. It does not conflate
    the bounds experiment with Comfy's float/bilinear native image path.
    """
    if name not in PROCESSOR_POLICIES:
        raise ValueError(f"unknown processor policy {name!r}")
    current_path = CURRENT_CONFIG_DIR / "processor_config.json"
    video_path = CURRENT_CONFIG_DIR / "video_preprocessor_config.json"
    current_container = json.loads(current_path.read_text())
    current = current_container.get("image_processor")
    if not isinstance(current, dict):
        raise ValueError(f"{current_path} has no image_processor object")
    effective = copy.deepcopy(current)
    source_files = [
        {
            "role": "current_w4_processor_snapshot",
            "path": _repo_path(current_path),
            "sha256": _sha256(current_path),
        },
        {
            "role": "current_w4_video_snapshot_unchanged",
            "path": _repo_path(video_path),
            "sha256": _sha256(video_path),
        },
    ]
    changed_fields = []
    if name == "current_w4_release_image_bounds":
        release = json.loads(RELEASE_IMAGE_CONFIG.read_text())
        release_size = release.get("size")
        if not isinstance(release_size, dict):
            raise ValueError(f"{RELEASE_IMAGE_CONFIG} has no size object")
        # This arm is meant to isolate pixel bounds. If any geometry or
        # normalization field diverges, silently calling it bounds-only would
        # invalidate that interpretation.
        shared_fields = (
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "image_mean",
            "image_std",
        )
        mismatched = {
            key: (current.get(key), release.get(key))
            for key in shared_fields
            if current.get(key) != release.get(key)
        }
        if mismatched:
            raise ValueError(
                "release image geometry/normalization differs from the current "
                f"artifact; bounds-only arm is invalid: {mismatched}"
            )
        effective["size"] = copy.deepcopy(release_size)
        changed_fields.append("size")
        source_files.append(
            {
                "role": "release_declared_image_bounds",
                "path": _repo_path(RELEASE_IMAGE_CONFIG),
                "sha256": _sha256(RELEASE_IMAGE_CONFIG),
            }
        )
    scope = (
        "its snapshotted image bounds"
        if name == "current_w4"
        else "release-declared image bounds only"
    )
    record = {
        "name": f"shared_{name}",
        "scope": (
            "weight-only shared policy; current W4 source processor with " + scope
        ),
        "changed_from_current_w4": changed_fields,
        "effective_still_bounds": copy.deepcopy(effective.get("size")),
        "effective_image_processor_config": effective,
        "effective_image_processor_config_sha256": _json_sha(effective),
        "effective_video_policy": "current_w4_snapshot_unchanged",
        "source_files": source_files,
    }
    return effective, record


def _artifact_declaration(h3, clip) -> dict | None:
    """Which snapshot the loaded artifact resolved to, before any override.

    Read off the CLIP the loader stamped, so it names the artifact actually
    open rather than the adapter's default. ``None`` for a CLIP that carries no
    stamp -- the BF16 arm before its processors are installed, and core's own
    ``CLIPLoader``.
    """
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    source = getattr(model, "_h3_processor_source", None)
    if not source:
        return None
    config_dir = Path(source)
    if not config_dir.is_dir():
        return {"snapshot": Path(source).name, "config_sha256": None}
    return {
        "snapshot": config_dir.name,
        "config_sha256": _sha256(config_dir / "config.json"),
        "declared_still_bounds": list(h3.source_image_pixel_bounds(config_dir)),
        "declared_video_bounds": list(h3.source_video_pixel_bounds(config_dir)),
    }


def _activate_processor_policy(h3, clip, name: str) -> tuple[dict, dict | None]:
    """Bind one policy through the adapter's public per-CLIP override seam.

    Returns the shared policy record and, separately, what the loaded artifact
    declared before the override. The policy is deliberately *shared*: an arm's
    own declaration is overridden so the comparison stays weight-only. Since
    2026-08-25 that override can be applied over an artifact whose declaration
    differs from the snapshot the policy is built from, so what was overridden
    is now recorded per arm instead of being invisible.
    """
    if Path(h3.CONFIG_DIR).resolve() != CURRENT_CONFIG_DIR.resolve():
        raise ValueError(
            f"adapter config root {h3.CONFIG_DIR} != expected {CURRENT_CONFIG_DIR}"
        )
    declaration = _artifact_declaration(h3, clip)
    effective, record = _processor_policy_spec(name)
    size = effective["size"]
    bounds = (int(size["shortest_edge"]), int(size["longest_edge"]))
    override = None if name == "current_w4" else bounds
    # The shared policy replaces the still-image processor AND rebinds the video
    # patchifier to the policy's own snapshot. It is only a still-image policy
    # while the two snapshots agree about video; otherwise the arm differs in
    # two places at once and is still labelled weight-only. The v1 and v2
    # snapshots agree today, which is exactly when a silent assumption forms.
    if declaration is not None and declaration.get("declared_video_bounds"):
        artifact = Path(
            clip.cond_stage_model.qwen3vl_32b.transformer._h3_processor_source
        )
        shared = (list(h3.source_video_pixel_bounds()),
                  h3.source_video_patch_geometry())
        theirs = (declaration["declared_video_bounds"],
                  h3.source_video_patch_geometry(artifact))
        if shared != theirs:
            raise ValueError(
                f"{artifact.name} declares a video view {theirs} where the "
                f"shared policy binds {shared}; a shared still-image policy "
                "cannot also move the video view"
            )
    h3.install_source_processors(clip, image_bounds=override)
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    actual = tuple(model._h3_image_bounds)
    if actual != bounds:
        raise ValueError(f"adapter bound image bounds {actual}, expected {bounds}")
    record["adapter_binding"] = {
        "api": "install_source_processors(image_bounds=...)",
        "bound_image_bounds": list(actual),
        "override_supplied": override is not None,
    }
    return record, declaration


def _git_commit(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(directory: Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "-C", str(directory), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _h3_module():
    """Load the repository adapter under a package name, as its checks do."""
    sys.path.insert(0, str(COMFY))
    package = types.ModuleType("_h3_encoder_capture_pkg")
    package.__path__ = [str(REPO)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.h3_awq_encoder", REPO / "h3_awq_encoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pattern(height: int, width: int, phase: int) -> torch.Tensor:
    """Deterministic [1,H,W,3] coordinate image in [0,1]."""
    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)
    red = torch.remainder(x + phase * 17, 257) / 256
    green = torch.remainder(y + phase * 29, 263) / 262
    blue = torch.remainder(x + y + phase * 41, 269) / 268
    return torch.stack(
        [red.expand(height, width), green.expand(height, width), blue], dim=-1
    ).unsqueeze(0)


def _fixtures() -> list[dict[str, Any]]:
    # These tensors represent media after H3's role-specific upstream sizing,
    # immediately before Qwen preprocessing. Keyframes/video are target-canvas
    # media; Ref2VA stills retain the 2048-short-edge serving geometry.
    canvas = _pattern(768, 1344, 1)
    ref_landscape = _pattern(2048, 3648, 2)
    ref_square = _pattern(2048, 2048, 3)
    ref_1080p_max = _pattern(1088, 1920, 6)
    video = torch.cat([_pattern(768, 1344, 4), _pattern(768, 1344, 5)], dim=0)
    return [
        {
            "fixture_id": "t2va_text",
            "family": "T2VA",
            "prompt": (
                "integrated_multimodal_description: [Shot 1] A static medium "
                "shot of a red bicycle beside a rain-dark wall. "
                "overall_soundscape: light rain. non_diegetic_music: N/A"
            ),
            "tokenize_kwargs": {},
            "ordered_media": [],
        },
        {
            "fixture_id": "i2va_single_image",
            "family": "I2VA",
            "prompt": (
                "At 0.00 seconds, Picture 1 supplies the opening composition. "
                "The camera slowly pushes toward the subject."
            ),
            "tokenize_kwargs": {"images": [canvas]},
            "ordered_media": [{"type": "image", "data": canvas}],
        },
        {
            "fixture_id": "fl2va_two_images",
            "family": "FL2VA",
            "prompt": (
                "Picture 1 defines the first frame and Picture 2 defines the "
                "last frame; motion between them remains continuous."
            ),
            "tokenize_kwargs": {"images": [canvas, canvas]},
            "ordered_media": [
                {"type": "image", "data": canvas},
                {"type": "image", "data": canvas},
            ],
        },
        {
            "fixture_id": "ref2va_ordered_multi_image",
            "family": "Ref2VA",
            "prompt": (
                "subject_definitions: Subject 1 follows Picture 1; Audio 1 "
                "defines the voice; Picture 2 defines the room. summary: "
                "[reference generation] Subject 1 enters the room and looks up."
            ),
            "tokenize_kwargs": {
                "minimax_ref_items": [
                    {"type": "image", "data": ref_landscape},
                    {"type": "audio"},
                    {"type": "image", "data": ref_square},
                ]
            },
            "ordered_media": [
                {"type": "image", "data": ref_landscape},
                {"type": "audio"},
                {"type": "image", "data": ref_square},
            ],
        },
        {
            "fixture_id": "ref2va_single_image",
            "family": "Ref2VA-still",
            "prompt": (
                "subject_definitions: Subject 1 follows Picture 1. summary: "
                "[reference generation] Subject 1 turns toward the window."
            ),
            "tokenize_kwargs": {
                "minimax_ref_items": [
                    {"type": "image", "data": ref_landscape},
                ]
            },
            "ordered_media": [
                {"type": "image", "data": ref_landscape},
            ],
        },
        {
            "fixture_id": "ref2va_single_image_1080p_max",
            "family": "Ref2VA-still-max-no-upscale",
            "prompt": (
                "subject_definitions: Subject 1 follows Picture 1. summary: "
                "[reference generation] Subject 1 turns toward the window."
            ),
            "tokenize_kwargs": {
                "minimax_ref_items": [
                    {"type": "image", "data": ref_1080p_max},
                ]
            },
            "ordered_media": [
                {"type": "image", "data": ref_1080p_max},
            ],
        },
        {
            "fixture_id": "ref2va_video_block",
            "family": "Ref2VA-video",
            "prompt": (
                "Video 1 defines the reference motion. The target holds the "
                "same direction of travel in a wider shot."
            ),
            "tokenize_kwargs": {
                "minimax_ref_items": [
                    {
                        "type": "video",
                        "data": video,
                        "timestamps": [0.0, 0.5],
                    }
                ]
            },
            "ordered_media": [
                {
                    "type": "video",
                    "data": video,
                    "timestamps": [0.0, 0.5],
                }
            ],
        },
        {
            "fixture_id": "t2va_dialogue_markers",
            "family": "T2VA-dialogue",
            "prompt": (
                "A woman turns toward the camera and says, "
                "<d>[English] We leave before sunrise.</d> She closes the door."
            ),
            "tokenize_kwargs": {},
            "ordered_media": [],
        },
    ]


def _real_ref_fixture(path: Path) -> dict[str, Any]:
    """Build a deterministic single-reference fixture from a decoded file."""
    import numpy as np
    from PIL import Image, ImageOps

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"real reference image absent: {path}")
    file_sha = _sha256(path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0)
    height, width = int(tensor.shape[1]), int(tensor.shape[2])
    return {
        "fixture_id": f"real_ref_{file_sha[:12]}",
        "family": "Ref2VA-still-real-heldout",
        "prompt": (
            "subject_definitions: Subject 1 follows Picture 1. summary: "
            "[reference generation] Subject 1 remains visually consistent "
            "while the camera changes angle."
        ),
        "tokenize_kwargs": {
            "minimax_ref_items": [{"type": "image", "data": tensor}]
        },
        "ordered_media": [{"type": "image", "data": tensor}],
        "source_files": [
            {
                "logical_name": path.name,
                "sha256": file_sha,
                "decoded_width": width,
                "decoded_height": height,
            }
        ],
    }


def _media_records(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = {"type": row["type"]}
        data = row.get("data")
        if torch.is_tensor(data):
            item["tensor"] = _tensor_record(data)
        if "timestamps" in row:
            item["timestamps"] = list(row["timestamps"])
        out.append(item)
    return out


def _wanted_bf16_key(name: str) -> bool:
    if name == "model.language_model.embed_tokens.weight":
        return True
    if name.startswith("model.visual."):
        return True
    prefix = "model.language_model.layers."
    if name.startswith(prefix):
        tail = name[len(prefix):]
        layer = tail.split(".", 1)[0]
        return layer.isdigit() and int(layer) < 50
    return False


def _target_bf16_key(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual."):]
    raise ValueError(f"unhandled BF16 key: {name}")


def bf16_inventory(root: Path) -> dict:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"BF16 index absent: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map") or {}
    selected = {k: v for k, v in weight_map.items() if _wanted_bf16_key(k)}
    if not selected:
        raise ValueError("BF16 index selected no H3 tensors")
    shards = sorted(set(selected.values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise ValueError(f"BF16 shards absent: {missing}")
    return {
        "index": index_path,
        "selected_keys": selected,
        "shards": [root / name for name in shards],
        "selected_tensor_count": len(selected),
        "mapped_shard_bytes": sum((root / name).stat().st_size for name in shards),
    }


def _load_bf16(root: Path, h3, embedding_directory):
    import comfy.sd

    inventory = bf16_inventory(root)
    by_shard: dict[str, list[str]] = {}
    for key, shard in inventory["selected_keys"].items():
        by_shard.setdefault(shard, []).append(key)

    state = {}
    print(
        f"mapping {inventory['selected_tensor_count']} H3 tensors from "
        f"{len(inventory['shards'])} BF16 shards",
        flush=True,
    )
    for path in inventory["shards"]:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for source_name in sorted(by_shard[path.name]):
                target_name = _target_bf16_key(source_name)
                if target_name in state:
                    raise ValueError(f"BF16 key collision after conversion: {target_name}")
                state[target_name] = handle.get_tensor(source_name)

    provided_shapes = {name: tuple(value.shape) for name, value in state.items()}
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state],
        embedding_directory=embedding_directory,
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    missing = sorted(set(expected) - set(provided_shapes))
    unexpected = sorted(set(provided_shapes) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(provided_shapes)
        if expected[name] != provided_shapes[name]
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "BF16 converted state does not exactly satisfy native H3: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"mismatched={mismatched[:3]}"
        )
    h3.install_source_processors(clip)
    h3._validate_native_tokenizer(clip)
    return clip, inventory


def _rebind_source_processors(h3, clip):
    """Re-install the artifact's processors WITHOUT losing which artifact it is.

    ``install_source_processors`` defaults to the adapter's own snapshot, which
    was harmless while one artifact existed and is not any more: called bare on
    a CLIP the loader resolved to a second generation, it silently rebinds that
    CLIP to the first generation's processor bounds and overwrites the stamp
    saying so. Read the stamp the loader left and re-install against it.
    """
    model = clip.cond_stage_model.qwen3vl_32b.transformer
    stamped = getattr(model, "_h3_processor_source", None)
    resolved = None
    if stamped and Path(stamped).is_dir() and \
            Path(stamped).resolve() != Path(h3.CONFIG_SOURCE).resolve():
        resolved = Path(stamped)
    h3.install_source_processors(clip, snapshot=resolved)
    after = getattr(model, "_h3_processor_source", None)
    if after != stamped:
        raise ValueError(
            f"re-installing processors moved the artifact from {stamped} to "
            f"{after}; the CLIP would preprocess under another artifact's config"
        )
    return clip


def _load_w4(path: Path, h3, embedding_directory):
    if not path.is_file():
        raise ValueError(f"W4 artifact absent: {path}")
    clip = h3._load_clip(
        str(path),
        embedding_directory,
        disable_dynamic=False,
        install_cache=False,
    )
    # _load_clip already installs these. Reinstalling makes the shared-policy
    # choice explicit and gives BF16/W4 one callable owner -- through the helper
    # above, so it cannot also change which artifact's config is bound.
    return _rebind_source_processors(h3, clip)


class InputRecorder:
    def __init__(self, clip, h3):
        self.clip = clip
        self.h3 = h3
        self.clip_model = clip.cond_stage_model.qwen3vl_32b
        self.model = self.clip_model.transformer
        self.image_bounds = tuple(self.model._h3_image_bounds)
        self._original_preprocess = self.model.preprocess_embed
        self._original_process_tokens = self.clip_model.process_tokens
        self.current: dict | None = None
        self.model.preprocess_embed = types.MethodType(self._preprocess, self.model)
        self.clip_model.process_tokens = types.MethodType(
            self._process_tokens, self.clip_model
        )

    def begin(self):
        if self.current is not None:
            raise RuntimeError("recorder already active")
        self.current = {"visual_patches": [], "process_calls": []}

    def finish(self) -> dict:
        if self.current is None:
            raise RuntimeError("recorder is not active")
        current = self.current
        self.current = None
        if len(current["process_calls"]) != 1:
            raise ValueError(
                f"expected one token-processing call, got {len(current['process_calls'])}"
            )
        current.update(current.pop("process_calls")[0])
        return current

    def _preprocess(self, this, embed, device):
        if self.current is None:
            raise RuntimeError("visual preprocessing happened outside a capture")
        if embed.get("type") != "image":
            return None, None
        video = bool(embed.get("minimax_video_block", False))
        if video:
            patches, grid = self.h3._source_video_block_patches(embed["data"], device)
        else:
            patches, grid = self.h3._source_image_patches(
                embed["data"], device, self.image_bounds
            )
        self.current["visual_patches"].append(
            {
                "kind": "video_block" if video else "image",
                **_tensor_record(patches),
                "grid_thw": grid.detach().cpu().tolist(),
                "source_media": _tensor_record(embed["data"]),
            }
        )
        merged, deepstack = this.visual(
            patches.to(device=device, dtype=torch.float32), grid
        )
        return merged, {"grid": grid, "deepstack": deepstack}

    def _process_tokens(self, this, tokens, device):
        if self.current is None:
            raise RuntimeError("token processing happened outside a capture")
        embeds, attention, num_tokens, embeds_info = self._original_process_tokens(
            tokens, device
        )
        position_ids, visual_mask, _ = self.model.build_image_inputs(
            embeds, embeds_info
        )
        grids = []
        infos = []
        for info in embeds_info:
            extra = info.get("extra")
            grid = extra.get("grid") if isinstance(extra, dict) else extra
            grid_list = grid.detach().cpu().tolist() if torch.is_tensor(grid) else None
            grids.extend(grid_list or [])
            infos.append(
                {
                    "type": info.get("type"),
                    "index": int(info["index"]),
                    "size": int(info["size"]),
                    "grid_thw": grid_list,
                }
            )
        self.current["process_calls"].append(
            {
                "input_embeds": _tensor_record(embeds),
                "attention_mask": attention.detach().cpu().tolist(),
                "attention_mask_sha256": _json_sha(attention.detach().cpu().tolist()[0]),
                "num_tokens": [int(x) for x in num_tokens],
                "embeds_info": infos,
                "grid_thw": grids,
                "position_ids_sha256": (
                    _tensor_sha(position_ids) if position_ids is not None else None
                ),
                "position_ids": (
                    {
                        "shape": list(position_ids.shape),
                        "dtype": str(position_ids.dtype).removeprefix("torch."),
                    }
                    if position_ids is not None
                    else None
                ),
                "visual_mask_sha256": (
                    _tensor_sha(visual_mask) if visual_mask is not None else None
                ),
            }
        )
        return embeds, attention, num_tokens, embeds_info


def _entries(tokens: dict) -> list:
    if set(tokens) != {"qwen3vl_32b"}:
        raise ValueError(f"unexpected tokenizer keys: {sorted(tokens)}")
    batches = tokens["qwen3vl_32b"]
    if len(batches) != 1:
        raise ValueError(f"expected one token batch, got {len(batches)}")
    return [row[0] for row in batches[0]]


def _expanded_ids(entries: list, embeds_info: list[dict]) -> list[int]:
    out = []
    embed_index = 0
    for entry in entries:
        if isinstance(entry, numbers.Integral):
            out.append(int(entry))
            continue
        if not isinstance(entry, dict) or embed_index >= len(embeds_info):
            raise ValueError("tokenizer placeholder does not match processed embeds")
        info = embeds_info[embed_index]
        out.extend([IMAGE_PAD] * int(info["size"]))
        embed_index += 1
    if embed_index != len(embeds_info):
        raise ValueError("processed embeds remain after token expansion")
    return out


def _vision_spans(tags: list[int]) -> list[list[int]]:
    spans = []
    start = None
    for index, tag in enumerate(tags):
        if tag == 0 and start is None:
            start = index
        elif tag != 0 and start is not None:
            spans.append([start, index - 1])
            start = None
    if start is not None:
        spans.append([start, len(tags) - 1])
    return spans


def _capture_one(clip, recorder: InputRecorder, fixture: dict) -> tuple[torch.Tensor, dict]:
    tokens = clip.tokenize(fixture["prompt"], **fixture["tokenize_kwargs"])
    entries = _entries(tokens)
    recorder.begin()
    out = clip.encode_from_tokens(tokens, return_dict=True)
    recorded = recorder.finish()

    hidden = out["cond"].detach().float().cpu()
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"unexpected hidden-state shape {tuple(hidden.shape)}")
    hidden = hidden[0].contiguous()
    tags_tensor = out.get("minimax_token_tags")
    if tags_tensor is None:
        raise ValueError("native H3 encode returned no minimax_token_tags")
    tags = [int(x) for x in tags_tensor.detach().cpu().flatten().tolist()]
    expanded = _expanded_ids(entries, recorded["embeds_info"])
    if not (len(expanded) == len(tags) == hidden.shape[0]):
        raise ValueError(
            f"expanded ids/tags/hidden disagree: {len(expanded)}, {len(tags)}, "
            f"{hidden.shape[0]}"
        )
    attention = recorded["attention_mask"]
    if len(attention) != 1 or len(attention[0]) != hidden.shape[0]:
        raise ValueError("attention mask does not align with hidden rows")
    if recorded["num_tokens"] != [hidden.shape[0]]:
        raise ValueError("fixture unexpectedly contains masked/padded rows")

    declared = recorder.h3._config("tokenizer_config.json").get(
        "extra_special_tokens"
    )
    if not isinstance(declared, list) or len(declared) != 20:
        raise ValueError("source tokenizer config does not declare 20 special tokens")
    vocab = recorder.clip.tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    marker_ids = {int(vocab[token]) for token in declared[13:]}
    if marker_ids != set(range(151669, 151676)):
        raise ValueError(f"unexpected H3 marker ids: {sorted(marker_ids)}")
    metadata = {
        "fixture_id": fixture["fixture_id"],
        "family": fixture["family"],
        "prompt": fixture["prompt"],
        "prompt_sha256": hashlib.sha256(fixture["prompt"].encode()).hexdigest(),
        "source_files": fixture.get("source_files", []),
        "sequence_length": int(hidden.shape[0]),
        "ordered_media": _media_records(fixture["ordered_media"]),
        "pre_expansion_entries": [
            int(x) if isinstance(x, numbers.Integral) else {
                "type": x.get("type"),
                "minimax_video_block": bool(x.get("minimax_video_block", False)),
            }
            for x in entries
        ],
        "expanded_token_ids": expanded,
        "expanded_token_ids_sha256": _json_sha(expanded),
        "visual_patches": recorded["visual_patches"],
        "grid_thw": recorded["grid_thw"],
        "embeds_info": recorded["embeds_info"],
        "vision_spans": _vision_spans(tags),
        "token_tags": tags,
        "token_tags_sha256": _json_sha(tags),
        "attention_mask": attention[0],
        "attention_mask_sha256": recorded["attention_mask_sha256"],
        "position_ids_sha256": recorded["position_ids_sha256"],
        "position_ids": recorded["position_ids"],
        "visual_mask_sha256": recorded["visual_mask_sha256"],
        "input_embeds_sha256": recorded["input_embeds"]["sha256"],
        "input_embeds": recorded["input_embeds"],
        "text_positions": [i for i, tag in enumerate(tags) if tag == 1],
        "vision_positions": [i for i, tag in enumerate(tags) if tag == 0],
        "marker_positions": [i for i, token in enumerate(expanded) if token in marker_ids],
        "hidden_state": {
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype).removeprefix("torch."),
        },
    }
    return hidden, metadata


def self_test() -> None:
    """Exercise the per-arm snapshot record and its guard, CPU only, no model.

    The capture itself needs the card. What is asserted here does not: whether
    an arm records which artifact declaration the shared policy overrode, and
    whether the guard fires when that artifact's video view disagrees with the
    policy's. Both were introduced on 2026-08-25 for a second artifact
    generation, and the guard is the kind that cannot be shown to work by the
    real artifacts, because v1 and v2 agree about video.
    """
    h3 = _h3_module()
    snapshots = h3._snapshot_dirs()
    assert len(snapshots) >= 2, "need a second snapshot to tell arms apart"

    def stub(source=None):
        transformer = types.SimpleNamespace()
        if source is not None:
            transformer._h3_processor_source = str(source)
        return types.SimpleNamespace(cond_stage_model=types.SimpleNamespace(
            qwen3vl_32b=types.SimpleNamespace(transformer=transformer)))

    # An unstamped CLIP -- core's loader, and the BF16 arm before its
    # processors are installed -- declares nothing rather than the default.
    assert _artifact_declaration(h3, stub()) is None

    # The re-install on the W4 load path must not move the CLIP to another
    # artifact's config. It did until 2026-08-25, by calling
    # `install_source_processors` bare: a second-generation CLIP came back
    # bound to v1's still-image ceiling with the stamp overwritten to match, so
    # the arm would have recorded v1 and preprocessed at v1's bounds while
    # holding v2's weights. Nothing downstream could have seen it, because the
    # stamp it would have been caught by is the thing that got overwritten.
    for config_dir in snapshots:
        clip = stub(config_dir)
        h3.install_source_processors(clip, snapshot=config_dir)
        _rebind_source_processors(h3, clip)
        model = clip.cond_stage_model.qwen3vl_32b.transformer
        assert Path(model._h3_processor_source) == config_dir, (
            f"re-install moved {config_dir.name} to {model._h3_processor_source}")
        assert model._h3_encoder_contract["source"] == config_dir.name
        assert tuple(model._h3_image_bounds) == tuple(
            h3.source_image_pixel_bounds(config_dir))

    seen = {}
    with tempfile.TemporaryDirectory(prefix="h3-capture-selftest-") as raw:
        # A stand-in for the artifact, so the record path is exercised without
        # streaming the real multi-gigabyte file through sha256 twice.
        stand_in = Path(raw) / "stand_in.safetensors"
        stand_in.write_bytes(b"")
        for config_dir in snapshots:
            clip = stub(config_dir)
            record, declaration = _activate_processor_policy(
                h3, clip, "current_w4_release_image_bounds")
            assert declaration["snapshot"] == config_dir.name, declaration
            model = _model_record("w4", stand_in, None, record, declaration)
            assert model["artifact_declaration_overridden"] == declaration
            # The point of the field: the shared policy record is identical
            # across arms -- which is what the comparator requires -- while the
            # model record says whose declaration was overridden to get there.
            seen[config_dir.name] = (
                record["effective_image_processor_config_sha256"],
                tuple(clip.cond_stage_model.qwen3vl_32b.transformer
                      ._h3_image_bounds),
            )
    assert len(set(seen.values())) == 1, (
        f"the shared policy did not bind identically across snapshots: {seen}")
    assert len(seen) == len(snapshots)

    # Deliberate violation: an artifact whose video view differs from the
    # policy's must stop the arm, not be labelled weight-only.
    other = next(d for d in snapshots if d != CURRENT_CONFIG_DIR)
    with tempfile.TemporaryDirectory(prefix="h3-capture-selftest-") as raw:
        moved = Path(raw) / other.name
        shutil.copytree(other, moved)
        video_path = moved / "video_preprocessor_config.json"
        video = json.loads(video_path.read_text())
        video["size"]["longest_edge"] = int(video["size"]["longest_edge"]) // 2
        video_path.write_text(json.dumps(video, indent=2) + "\n")
        try:
            _activate_processor_policy(h3, stub(moved), "current_w4")
        except ValueError as exc:
            assert "cannot also move the video view" in str(exc), exc
        else:
            raise AssertionError(
                "an artifact declaring a different video view was accepted "
                "into a shared still-image policy")

    print(f"ok: {len(snapshots)} snapshot(s) bind one shared policy identically, "
          "each arm records the declaration it overrode, and a divergent video "
          "view is refused")


def _model_record(
    arm: str,
    source: Path,
    inventory: dict | None,
    processor_policy: dict,
    artifact_declaration: dict | None = None,
) -> dict:
    """One arm's model identity, including whose declaration was overridden.

    ``artifact_declaration`` belongs here and not in ``processor_policy_record``
    on purpose. The comparator requires that record to be *equal* across arms
    (`compare_h3_encoder_captures.py::compare`), which is what makes
    "weight-only" mean anything; a per-artifact value in it would make every
    BF16-versus-candidate run refuse itself. The model record is per-arm by
    design, so this is where an arm says which artifact snapshot the shared
    policy was applied over.
    """
    if arm == "w4":
        resolved = source.resolve()
        print(f"hashing W4 artifact {resolved.name}", flush=True)
        files = [{"name": resolved.name, "size": resolved.stat().st_size, "sha256": _sha256(resolved)}]
        source_record = _source_locator(arm, source)
    else:
        print(f"hashing {len(inventory['shards'])} mapped BF16 shard files", flush=True)
        files = [
            {"name": p.name, "size": p.stat().st_size, "sha256": _sha256(p)}
            for p in inventory["shards"]
        ]
        index = inventory["index"]
        files.insert(
            0,
            {"name": index.name, "size": index.stat().st_size, "sha256": _sha256(index)},
        )
        source_record = {
            **_source_locator(arm, source),
            "selected_tensor_count": inventory["selected_tensor_count"],
            "mapped_shard_bytes": inventory["mapped_shard_bytes"],
        }
    record = {
        "arm": arm,
        "source": source_record,
        "files": files,
        "shared_processor_configs": processor_policy["source_files"],
        "effective_image_processor_config_sha256": processor_policy[
            "effective_image_processor_config_sha256"
        ],
    }
    if artifact_declaration is not None:
        record["artifact_declaration_overridden"] = artifact_declaration
    return record


def _provenance() -> dict:
    import comfy.model_management as model_management

    comfy_commit = _git_commit(COMFY)
    implementation_paths = [
        COMFY / "comfy" / "sd.py",
        COMFY / "comfy" / "sd1_clip.py",
        COMFY / "comfy" / "text_encoders" / "llama.py",
        COMFY / "comfy" / "text_encoders" / "minimax.py",
        COMFY / "comfy" / "text_encoders" / "qwen3vl.py",
        COMFY / "comfy" / "text_encoders" / "qwen_vl.py",
    ]
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": _git_commit(REPO),
        "repo_dirty": _git_dirty(REPO),
        "comfyui_commit": comfy_commit,
        "comfyui_dirty": _git_dirty(COMFY),
        "capture_script_sha256": _sha256(Path(__file__).resolve()),
        "h3_awq_adapter_sha256": _sha256(REPO / "h3_awq_encoder.py"),
        "implementation_files": [
            {
                "path": str(path.relative_to(COMFY)),
                "sha256": _sha256(path),
            }
            for path in implementation_paths
        ],
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "extra_reserved_vram_bytes": int(model_management.extra_reserved_memory()),
        "gpu": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }


def _fixture_inventory(fixtures: list[dict]) -> list[dict]:
    rows = []
    for fixture in fixtures:
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "family": fixture["family"],
                "prompt_sha256": hashlib.sha256(fixture["prompt"].encode()).hexdigest(),
                "source_files": fixture.get("source_files", []),
                "ordered_media": _media_records(fixture["ordered_media"]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("bf16", "w4"), required=False)
    parser.add_argument(
        "--self-test", action="store_true",
        help="CPU-only: assert the per-arm snapshot record and its guard",
    )
    parser.add_argument("--out", help="new capture directory; must not exist")
    parser.add_argument(
        "--bf16-dir",
        default=os.environ.get("H3_BF16_ENCODER_DIR"),
        help="BF16 sharded model directory; defaults to H3_BF16_ENCODER_DIR",
    )
    parser.add_argument("--w4-path", default=str(W4_DEFAULT))
    parser.add_argument(
        "--processor-policy",
        choices=PROCESSOR_POLICIES,
        default="current_w4",
        help=(
            "shared weight-only image policy; the release-bounds option keeps "
            "the current artifact processor implementation and changes only size"
        ),
    )
    parser.add_argument(
        "--fixture", action="append", help="capture only this fixture id; repeatable"
    )
    parser.add_argument(
        "--real-ref-image",
        action="append",
        help=(
            "replace built-in fixtures with one single-reference fixture per "
            "image path; repeatable"
        ),
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="inspect source/fixture inventory without loading a model or hashing weights",
    )
    parser.add_argument(
        "--reserve-vram-gib",
        type=float,
        default=0.0,
        help=(
            "additional Comfy dynamic-offload reserve; recorded in provenance "
            "and required to match between compared arms"
        ),
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.arm:
        parser.error("--arm is required unless --self-test is used")
    if args.reserve_vram_gib < 0:
        parser.error("--reserve-vram-gib must be nonnegative")

    if args.arm == "bf16" and not args.bf16_dir:
        parser.error("--bf16-dir or H3_BF16_ENCODER_DIR is required for the BF16 arm")
    source = Path(args.bf16_dir if args.arm == "bf16" else args.w4_path).expanduser()
    inventory = bf16_inventory(source) if args.arm == "bf16" else None
    fixtures = (
        [_real_ref_fixture(Path(path)) for path in args.real_ref_image]
        if args.real_ref_image
        else _fixtures()
    )
    if args.fixture:
        wanted = set(args.fixture)
        known = {row["fixture_id"] for row in fixtures}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"unknown fixture ids: {unknown}")
        fixtures = [row for row in fixtures if row["fixture_id"] in wanted]

    if args.inventory_only:
        _, processor_policy = _processor_policy_spec(args.processor_policy)
        result = {
            "arm": args.arm,
            "path_policy": "machine-local paths omitted; logical identifiers only",
            "source": _source_locator(args.arm, source),
            "processor_policy": processor_policy,
            "fixtures": _fixture_inventory(fixtures),
        }
        if inventory:
            result["bf16"] = {
                "selected_tensor_count": inventory["selected_tensor_count"],
                "mapped_shards": [p.name for p in inventory["shards"]],
                "mapped_shard_bytes": inventory["mapped_shard_bytes"],
            }
        elif source.is_file():
            with safe_open(source, framework="pt", device="cpu") as handle:
                result["w4"] = {
                    "tensor_count": len(list(handle.keys())),
                    "size": source.stat().st_size,
                    "metadata": handle.metadata(),
                }
        print(json.dumps(result, indent=2))
        return 0

    if not args.out:
        parser.error("--out is required unless --inventory-only is used")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for layer-50 capture")
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        raise SystemExit(f"refuse to overwrite existing capture directory: {out_dir}")
    out_dir.mkdir(parents=True)

    h3 = _h3_module()
    import folder_paths
    import comfy.model_management as model_management

    model_management.EXTRA_RESERVED_VRAM = int(
        args.reserve_vram_gib * 1024 * 1024 * 1024
    )

    embedding_directory = folder_paths.get_folder_paths("embeddings")
    if args.arm == "bf16":
        clip, inventory = _load_bf16(source, h3, embedding_directory)
    else:
        clip = _load_w4(source, h3, embedding_directory)
    processor_policy, artifact_declaration = _activate_processor_policy(
        h3, clip, args.processor_policy
    )

    vocab = clip.tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    vocab_sha = _json_sha(sorted((str(token), int(index)) for token, index in vocab.items()))
    recorder = InputRecorder(clip, h3)
    fixture_records = []
    for fixture in fixtures:
        fixture_id = fixture["fixture_id"]
        print(f"capturing {args.arm} {fixture_id}", flush=True)
        hidden, record = _capture_one(clip, recorder, fixture)
        if fixture_id in REPEAT_FIXTURES:
            repeated, repeated_record = _capture_one(clip, recorder, fixture)
            alignment_keys = (
                "expanded_token_ids_sha256",
                "ordered_media",
                "visual_patches",
                "grid_thw",
                "vision_spans",
                "token_tags_sha256",
                "attention_mask_sha256",
                "position_ids_sha256",
                "input_embeds_sha256",
            )
            if any(record[key] != repeated_record[key] for key in alignment_keys):
                raise ValueError(f"{fixture_id}: repeat changed recorded inputs")
            if not torch.equal(hidden, repeated):
                raise ValueError(f"{fixture_id}: repeat hidden state is not bit-identical")
            record["repeat_control"] = {"performed": True, "bit_identical": True}
        else:
            record["repeat_control"] = {"performed": False}

        hidden_path = out_dir / f"{fixture_id}.safetensors"
        save_file({"hidden": hidden}, str(hidden_path))
        record["hidden_state_file"] = hidden_path.name
        record["hidden_state_sha256"] = _sha256(hidden_path)
        fixture_records.append(record)

    manifest = {
        "schema_version": SCHEMA,
        "path_policy": "machine-local paths omitted; logical identifiers only",
        "arm": args.arm,
        "comparison_kind": "weight_only",
        "processor_policy": processor_policy["name"],
        "processor_policy_record": processor_policy,
        "output_tap": "raw state after language layer index 49; no final norm or lm_head",
        "fixture_population": "controlled deterministic substrate; not corpus-representative",
        "tokenizer_vocab_sha256": vocab_sha,
        "model": _model_record(args.arm, source, inventory, processor_policy,
                               artifact_declaration),
        "provenance": _provenance(),
        "fixtures": fixture_records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote complete capture {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
