"""Canonical dimensions, math constants, bus/regime enums, and checkpoint tables for `steering/`."""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path


def import_repo_module(name: str):
    """Import a root pack module (`control_vectors`, `h3_spans`, `h3_rules`, etc.) cleanly
    whether `REPO_ROOT` is on `sys.path` or loaded via ComfyUI's custom-node package,
    without ever colliding with same-named CLI scripts under `bench/`."""
    repo_root = Path(__file__).resolve().parent.parent
    existing = sys.modules.get(name)
    if existing is not None:
        ex_file = getattr(existing, "__file__", None)
        if ex_file and Path(ex_file).resolve().parent == repo_root:
            return existing
    for mod_key, mod_obj in list(sys.modules.items()):
        mod_paths = getattr(mod_obj, "__path__", None)
        if isinstance(mod_paths, (list, tuple)) and any(
            isinstance(p, (str, Path)) and Path(p).resolve() == repo_root for p in mod_paths
        ):
            try:
                return importlib.import_module(f".{name}", mod_key)
            except ImportError:
                pass
    resolved_sys_paths = set()
    for sp in sys.path:
        try:
            resolved_sys_paths.add(Path(sp or ".").resolve())
        except Exception:
            pass
    if repo_root in resolved_sys_paths:
        try:
            return importlib.import_module(name)
        except ImportError:
            pass
    pkg_name = "_h3pack"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[pkg_name] = pkg
    return importlib.import_module(f"{pkg_name}.{name}")

FORMAT_VERSION: str = "h3-control-vector/1"
SPAN_CACHE_FORMAT_VERSION: str = "h3-span-cache/1"

POST_REFINER_DIM: int = 5376
ENCODER_DIM: int = 5120

NORM_SHIFT_WINDOW: tuple[float, float] = (50.0, 85.0)
NORM_MATCHED_TARGET: float = 67.5
assert NORM_MATCHED_TARGET == sum(NORM_SHIFT_WINDOW) / 2.0

STACK_SHIFT_WARN_NORM: float = NORM_MATCHED_TARGET * 1.45
STACK_SMOOTH_KNEE_NORM: float = 95.5
STACK_SMOOTH_CEIL_NORM: float = 98.5

ERASURE_MAX_RANK: int = 8
COMPOSED_ERASURE_MAX_RANK: int = 16
MIN_ERASURE_PAIRS: int = 6
ERASURE_RIDGE_REL: float = 0.15
ERASURE_MATCH_TAU: float = 0.15

# Settled 4-bus Sound-Mixer architecture and Day-1 v1 active regime ("a" only).
STEER_BUSES: tuple[str, ...] = ("look", "motion", "audio", "hold")
STEER_REGIMES: tuple[str, ...] = ("a",)

FADER_MIN: float = -1.5
FADER_MAX: float = 2.0
DEFAULT_RELEASE_END_PERCENT: float = 0.33

NAME_RE = re.compile(r"^[a-z0-9_]+$")
CUSTOM_FAMILY_RE = re.compile(r"^custom:[0-9a-f]{12}$")
LLM_VECTOR_KEY_RE = re.compile(r"^vector\.llm\.\d+$")

VALID_MODES: tuple[str, ...] = ("t2va", "fl2va", "i2va", "l2va", "ref2va", "p_h3_image")
VALID_FRAMES: tuple[str, ...] = ("scene", "bare")
VALID_POOLING: tuple[str, ...] = ("scene_text", "all_tag1", "from_first_diff", "diff_span")
VALID_MEDIA_POOLING: tuple[str, ...] = (
    "scene_text",
    "look_only",
    "description_only",
    "soundscape_only",
    "all_tag1",
    "vision_tokens",
    "all_rows",
)
VALID_ALL_POOLING: tuple[str, ...] = (
    "scene_text",
    "look_only",
    "description_only",
    "soundscape_only",
    "all_tag1",
    "from_first_diff",
    "diff_span",
    "vision_tokens",
    "all_rows",
)

APPLY_MODES: tuple[str, ...] = (
    "add",
    "add_norm_matched",
    "add_norm_matched_erasure",
    "hold",
    "project_out",
    "amplify",
)
SCALE_MODES: tuple[str, ...] = ("raw", "norm_matched")

STEER_MODES: tuple[str, ...] = ("add_norm_matched_erasure", "add_norm_matched", "add", "project_out")
STEER_TARGETS: tuple[str, ...] = (
    "scene_only",
    "description_only",
    "look_only",
    "soundscape_only",
    "after_dialogue",
    "non_vision",
    "vision_tokens",
)
STEER_STREAMS: tuple[str, ...] = ("both", "video_only", "audio_only")
STEER_EDIT_PASSES: tuple[str, ...] = ("both", "edited_only", "original_only")
STEER_BLOCK_BANDS: tuple[str, ...] = (
    "all",
    "mid_late_20_49",
    "late_30_49",
    "early_0_19",
)
STEER_SPLIT_BACKENDS: tuple[str, ...] = (
    "auto_1x_kv",
    "legacy_2x_pass",
)
ROUTING_FAMILIES: tuple[str, ...] = ("fl2va", "ref2va", "h3_image", "fasth3")

DEFAULT_EXCLUDE_SHOTS: tuple[str, ...] = ("t2va_offpath_tortoise", "ref2va_look_anchor")
COMPILER_EXCLUDE_SHOTS: tuple[str, ...] = (
    "t2va_offpath_tortoise",
    "ref2va_look_anchor",
    "t2va_covered_market",
    "ref2va_market_stallholder",
    "ref2va_woman_blonde_street",
)

NON_H3_IMAGE_STEMS: frozenset[str] = frozenset(
    {
        "auctioneer_rapid_patter",
        "handheld_documentary_zoom",
        "operatic_bellow",
        "pyro_klaxon_audio",
        "reverb_cathedral",
        "vocal_strain",
        "whisper_asmr",
    }
)

CANONICAL_PROVENANCE_KEYS: tuple[str, ...] = (
    "author_kind",
    "requested_model",
    "resolved_model",
    "preset",
    "system_source",
    "system_prompt_sha256",
    "thinking",
    "reasoning_effort",
    "temperature",
    "seed",
    "repair_used",
    "input_tokens",
    "output_tokens",
    "created_utc",
)

SECTION_HEADER_PREFIXES: tuple[str, ...] = (
    "integrated_multimodal_description:",
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
SECTION_AND_SHOT_MARKERS: tuple[str, ...] = (*SECTION_HEADER_PREFIXES, "[Shot ")
MAX_SHARED_INTERIOR_WORDS: int = 5

FAMILY_TENSOR_KEYS: tuple[str, str, str] = (
    "condition_proj.weight",
    "condition_proj.bias",
    "token_refiner.final_norm.weight",
)

KNOWN_FAMILIES: dict[tuple[str, str, str], str] = {
    ("af3c58d7f2ad", "701e0673282f", "cf10a56a1137"): "fl2va",
    ("7c5485465531", "529e19c04005", "6ca615e4e3f8"): "ref2va",
    ("d6b17a52eb10", "2d8bca2b367d", "cf10a56a1137"): "fasth3",
    ("0c92b9024acc", "6f9a10f4d0ed", "036c13e58478"): "h3_image",
}

VIDEO_H3_FAMILIES: frozenset[str] = frozenset({"fl2va", "ref2va"})

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "format",
    "name",
    "quality",
    "pairs",
    "recipe",
    "family",
    "measured_under",
    "points",
    "quality_report",
    "provenance",
)

PIKA_FLEX_VARIANT_RE = re.compile(
    r"^(?:.*/)?(pika_h3_flex_(?:fl2va|ref2va))_(?:svd\d+_bf16|comfy_bf16|v1_avg_rank_\d+_bf16|kijai_bf16)\.safetensors$"
)

KNOWN_H3_CHECKPOINTS: dict[str, frozenset[str]] = {
    "fl2va": frozenset(
        {
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "minimax_h3_fl2va_int8_convrot.safetensors",
            "minimax_h3_fl2va_bf16.safetensors",
            "p-h3-image_int8_convrot.safetensors",
        }
    ),
    "ref2va": frozenset(
        {
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "minimax_h3_ref2va_int8_convrot.safetensors",
            "minimax_h3_ref2va_bf16.safetensors",
        }
    ),
    "h3_image": frozenset(
        {
            "minimax_h3_image_pruned_int8_convrot.safetensors",
            "minimax_h3_image_bf16.safetensors",
        }
    ),
}

KNOWN_REF2VA_GUIDE_LORAS: frozenset[str] = frozenset(
    {
        "minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors",
        "minimax_h3_inpainting_v1.0_r128_ffp.safetensors",
        "minimax_h3_camera_control_v1.0_r128_ffp.safetensors",
        "minimax_h3_motion_transfer_v1.0_r128_ffp.safetensors",
    }
)
