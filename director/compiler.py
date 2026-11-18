"""Runtime workflow compiler for Interactive Spatial Director.

Compiles spatial takes and anchor stills directly from shipped ``workflows/*.json``
templates and injects ``ModelPreviewOverrideKJ`` + ``taeh3`` / ``taeh3_image`` only
when the connected ComfyUI server exposes them in ``/object_info``.

Keeps top-level imports strictly stdlib so ``bench/check_imports.py`` passes offline.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS_DIR = _REPO_ROOT / "workflows"

INT32_TOKEN_LIMIT = 99864  # preflight._INT32_FUSED = 2**31 // (3 * 56 * 128)
TAEH3_VIDEO_FILE = "taeh3.safetensors"
TAEH3_IMAGE_FILE = "taeh3_image.safetensors"
MAX_VIDEO_PREVIEW_FRAMES = 27
PREVIEW_NODE_CLASS = "ModelPreviewOverrideKJ"
PREVIEW_NODE_ID = "165"
CLEAN_DECODE_NODE_ID = "162"
HUD_COMBINE_NODE_ID = "13"

ANCHOR_WORKFLOW_TEMPLATES: dict[str, dict[str, Any]] = {
    "h3_image": {
        "id": "h3_image",
        "label": "MiniMax-H3-Image (base)",
        "path": _WORKFLOWS_DIR / "image" / "h3_image_t2i_api.json",
    },
    "pika_image": {
        "id": "pika_image",
        "label": "Pika H3 Image (6-step)",
        "path": _WORKFLOWS_DIR / "image" / "h3_pika_image_t2i_api.json",
    },
    "ph3_image": {
        "id": "ph3_image",
        "label": "P-H3-Image (8-step)",
        "path": _WORKFLOWS_DIR / "image" / "h3_p_h3_image_t2i_api.json",
    },
}

IMAGE_EDIT_WORKFLOW_TEMPLATES: dict[str, dict[str, Any]] = {
    "h3_image": {
        "id": "h3_image",
        "label": "MiniMax-H3-Image Edit (default)",
        "path": _WORKFLOWS_DIR / "image" / "h3_image_edit_api.json",
    },
    "pika_image": {
        "id": "pika_image",
        "label": "Pika H3 1-Frame Edit (6-step)",
        "path": _WORKFLOWS_DIR / "image" / "h3_pika_image_edit_api.json",
    },
    "ph3_image": {
        "id": "ph3_image",
        "label": "P-H3-Image Edit (8-step)",
        "path": _WORKFLOWS_DIR / "image" / "h3_p_h3_image_edit_api.json",
    },
}

SPATIAL_WORKFLOW_SPECS: dict[str, dict[str, Any]] = {
    "r2v_portal": {
        "id": "r2v_portal",
        "label": "Reference Portal (R2V + optional Stamp)",
        "preferred": _WORKFLOWS_DIR / "h3_r2v_spatial_portal_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_world_targets_genie_pika_api.json",
        "modes": ["portal_path", "point_targets", "combined"],
        "has_ref_image": True,
        "has_first_frame": False,
        "supports_sprite": True,
    },
    "i2v_portal": {
        "id": "i2v_portal",
        "label": "First-Frame Portal (I2V)",
        "preferred": _WORKFLOWS_DIR / "h3_i2v_spatial_portal_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json",
        "modes": ["portal_path", "point_targets", "combined"],
        "has_ref_image": False,
        "has_first_frame": True,
        "supports_sprite": False,
    },
    "t2v_portal": {
        "id": "t2v_portal",
        "label": "Text-to-Video Portal (T2V)",
        "preferred": _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json",
        "modes": ["portal_path", "point_targets", "combined"],
        "has_ref_image": False,
        "has_first_frame": False,
        "supports_sprite": False,
    },
    "world_targets": {
        "id": "world_targets",
        "label": "World Targets + Sprite & SoundCues",
        "preferred": _WORKFLOWS_DIR / "h3_world_targets_genie_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_world_targets_genie_pika_api.json",
        "modes": ["point_targets", "portal_path", "combined"],
        "has_ref_image": True,
        "has_first_frame": False,
        "supports_sprite": True,
    },
}


def _find_comfy_root() -> Path | None:
    """Locate ComfyUI root across both custom_nodes checkouts and isolated worktrees."""
    import sys

    candidates: list[Path] = []
    env_comfy = (os.environ.get("H3_COMFY_DIR") or "").strip()
    if env_comfy:
        candidates.append(Path(env_comfy))
    candidates.extend(_REPO_ROOT.parents)
    for base in (sys.prefix, sys.executable):
        try:
            p = Path(base)
            candidates.extend(p.parents)
            # Also try symlink-resolved parents in case base is a custom wrapper outside ComfyUI.
            candidates.extend(p.resolve().parents)
        except Exception:
            pass
    for cand in candidates:
        if (cand / "comfy").is_dir() and (cand / "nodes.py").is_file():
            return cand
    return None


def _ensure_comfy_bootstrap() -> None:
    """Ensure ComfyUI root precedes repo root and cache ComfyUI ``nodes.py`` before pack imports."""
    import sys

    comfy_root = _find_comfy_root()
    if comfy_root is not None:
        root_str = str(comfy_root)
        if sys.path[:1] != [root_str]:
            while root_str in sys.path:
                sys.path.remove(root_str)
            sys.path.insert(0, root_str)
        try:
            import comfy.cli_args

            if "comfy.model_management" not in sys.modules:
                comfy.cli_args.args.cpu = True
        except Exception:
            pass
        if "nodes" not in sys.modules:
            try:
                import importlib

                importlib.import_module("nodes")
            except Exception:
                pass


def import_pack_module(name: str) -> Any:
    """Import a root pack module safely under an isolated package namespace (no ``__init__.py`` side effects)."""
    import importlib
    import sys
    import types

    _ensure_comfy_bootstrap()
    pkg_name = "_h3_director_pack"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_REPO_ROOT)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
    return importlib.import_module(f"{pkg_name}.{name}")


def _h3_rules() -> Any:
    return import_pack_module("h3_rules")


def _h3_prompt_edit() -> Any:
    return import_pack_module("h3_prompt_edit")


def _spatial_director() -> Any:
    return import_pack_module("spatial_director")


def _sound_cues() -> Any:
    return import_pack_module("sound_cues")


def all_on_both_clocks() -> tuple[int, ...]:
    """Return every frame length aligned on both the 17n+5 video and 40 Hz audio clocks."""
    rules = _h3_rules()
    return tuple(n for n in range(1, int(rules.MAX_LENGTH) + 1) if rules.on_both_clocks(n))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_spatial_template_path(mode: str) -> tuple[Path, bool]:
    """Return ``(template_path, used_shipped_preferred)`` for a spatial mode."""
    spec = SPATIAL_WORKFLOW_SPECS.get(mode)
    if spec is None:
        raise ValueError(f"Unknown spatial mode {mode!r}; expected one of {list(SPATIAL_WORKFLOW_SPECS)}")
    pref = Path(spec["preferred"])
    if pref.is_file():
        return pref, True
    fb = Path(spec["fallback"])
    if fb.is_file():
        return fb, False
    raise FileNotFoundError(f"Missing workflow templates for {mode!r}: {pref} and {fb}")


def available_workflows() -> list[dict[str, Any]]:
    """Describe spatial workflow modes available to the UI."""
    out: list[dict[str, Any]] = []
    for key, spec in SPATIAL_WORKFLOW_SPECS.items():
        pref = Path(spec["preferred"])
        fb = Path(spec["fallback"])
        out.append(
            {
                "id": key,
                "label": spec["label"],
                "modes": list(spec["modes"]),
                "available": pref.is_file() or fb.is_file(),
                "shipped": pref.is_file(),
                "has_ref_image": bool(spec["has_ref_image"]),
                "has_first_frame": bool(spec["has_first_frame"]),
                "supports_sprite": bool(spec["supports_sprite"]),
            }
        )
    return out


def available_anchor_workflows() -> list[dict[str, Any]]:
    """Describe still/anchor T2I workflow templates."""
    out: list[dict[str, Any]] = []
    for key, spec in ANCHOR_WORKFLOW_TEMPLATES.items():
        p = Path(spec["path"])
        out.append({"id": key, "label": spec["label"], "available": p.is_file()})
    return out


def available_image_edit_workflows() -> list[dict[str, Any]]:
    """Describe single-frame image-edit workflow templates for mask previews."""
    any_avail = any(Path(spec["path"]).is_file() for spec in IMAGE_EDIT_WORKFLOW_TEMPLATES.values())
    out: list[dict[str, Any]] = [
        {
            "id": "auto",
            "label": "Auto (Warm Resident — Pika ~4s / H3-Image HQ)",
            "available": any_avail,
        }
    ]
    for key, spec in IMAGE_EDIT_WORKFLOW_TEMPLATES.items():
        p = Path(spec["path"])
        out.append({"id": key, "label": spec["label"], "available": p.is_file()})
    return out


def resolve_image_edit_workflow(workflow: str | None, *, resident_backbone: str | None = None) -> str:
    """Resolve ``'auto'`` / empty masked-edit workflow against active resident VRAM backbone."""
    raw = str(workflow or "auto").strip()
    if raw in ("", "auto"):
        bb = str(resident_backbone or "").strip().lower()
        if bb == "pika":
            return "pika_image"
        if bb == "ph3_image":
            return "ph3_image"
        return "h3_image"
    return raw


def list_control_vectors() -> list[dict[str, Any]]:
    """Discover installed MiniMax H3 control vectors without importing Comfy runtime at module load."""
    dirs: list[Path] = []
    try:
        import folder_paths  # type: ignore

        for raw in folder_paths.get_folder_paths("control_vectors") or ():
            dirs.append(Path(raw))
    except Exception:
        pass
    if not dirs:
        try:
            ms_env = (os.environ.get("H3_MODEL_STORE") or "").strip()
            if ms_env:
                dirs.append(Path(ms_env) / "control_vectors")
        except Exception:
            pass
    comfy_root = _find_comfy_root()
    if comfy_root is not None:
        cv_guess = comfy_root / "models" / "control_vectors"
        if cv_guess.is_dir() and cv_guess not in dirs:
            dirs.append(cv_guess)

    seen_names: set[str] = set()
    vectors: list[dict[str, Any]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.safetensors")):
            rel = str(p.relative_to(d)).replace("\\", "/")
            stem = p.stem
            parts = stem.split("__")
            concept = parts[0] if parts else stem
            family = parts[1] if len(parts) > 1 else ""
            if rel in seen_names:
                continue
            seen_names.add(rel)
            modes = ["r2v_portal", "world_targets"] if "ref2va" in family else ["t2v_portal", "i2v_portal"]
            if not family:
                modes = list(SPATIAL_WORKFLOW_SPECS)
            vectors.append(
                {
                    "name": concept,
                    "file": rel,
                    "family": family or "unknown",
                    "available_modes": modes,
                }
            )
    return vectors


def _extract_combo_options(field_spec: Any) -> list[str]:
    """Handle both ComfyUI V1 `[list, meta]` and V3 `['COMBO', {'options': [...]}]` formats."""
    if isinstance(field_spec, (list, tuple)) and field_spec:
        if isinstance(field_spec[0], (list, tuple)):
            return [str(x) for x in field_spec[0]]
        if len(field_spec) >= 2 and isinstance(field_spec[1], dict) and isinstance(field_spec[1].get("options"), list):
            return [str(x) for x in field_spec[1]["options"]]
    if isinstance(field_spec, dict) and isinstance(field_spec.get("options"), list):
        return [str(x) for x in field_spec["options"]]
    return []


def resolve_kj_tiny_vae_option(object_info: dict[str, Any] | None, desired_filename: str) -> str | None:
    """Return matching ``tiny_vae`` combo string if ``ModelPreviewOverrideKJ`` exposes ``desired_filename``."""
    if not isinstance(object_info, dict):
        return None
    node_info = object_info.get(PREVIEW_NODE_CLASS)
    if not isinstance(node_info, dict):
        return None
    inp = node_info.get("input") or {}
    if not isinstance(inp, dict):
        return None
    for bucket in ("optional", "required"):
        group = inp.get(bucket)
        if not isinstance(group, dict):
            continue
        tiny_spec = group.get("tiny_vae")
        if tiny_spec is None:
            continue
        options = _extract_combo_options(tiny_spec)
        for opt in options:
            if opt == desired_filename or Path(opt).name == desired_filename:
                return opt
    return None


def has_kj_preview_override(object_info: dict[str, Any] | None, desired_filename: str = TAEH3_VIDEO_FILE) -> bool:
    """Return True only when ``ModelPreviewOverrideKJ`` is registered and exposes ``desired_filename``."""
    return resolve_kj_tiny_vae_option(object_info, desired_filename) is not None


_SAM3_PREFERRED_CKPTS: tuple[str, ...] = (
    "sam3.1_multiplex_fp16.safetensors",
    "sam3.1_multiplex_fp32.safetensors",
    "sam3_fp32.safetensors",
)


def resolve_sam3_checkpoint(object_info: dict[str, Any] | None) -> str | None:
    """Resolve an available SAM 3 / 3.1 checkpoint name from ``CheckpointLoaderSimple`` options."""
    if not isinstance(object_info, dict):
        return None
    loader = object_info.get("CheckpointLoaderSimple")
    if not isinstance(loader, dict):
        return None
    req = (loader.get("input") or {}).get("required") or {}
    options = _extract_combo_options(req.get("ckpt_name"))
    if not options:
        return None
    by_basename = {Path(opt).name: opt for opt in options}
    for pref in _SAM3_PREFERRED_CKPTS:
        if pref in options:
            return pref
        if pref in by_basename:
            return by_basename[pref]
    for opt in options:
        low = Path(opt).name.lower()
        if "sam3" in low and low.endswith(".safetensors"):
            return opt
    return None


def detect_sam3_capability(object_info: dict[str, Any] | None) -> dict[str, Any]:
    """Detect whether connected ComfyUI exposes ``SAM3_Detect`` and a SAM 3 / 3.1 checkpoint."""
    if not isinstance(object_info, dict):
        return {"available": False, "checkpoint": None, "has_corrections": False, "engine": "grabcut"}
    has_sam3 = isinstance(object_info.get("SAM3_Detect"), dict)
    ckpt = resolve_sam3_checkpoint(object_info)
    has_corr = isinstance(object_info.get("MiniMaxH3SAM31Corrections"), dict)
    available = bool(has_sam3 and ckpt)
    if available:
        engine = "sam3.1" if "3.1" in str(ckpt) else "sam3"
    else:
        engine = "grabcut"
    return {
        "available": available,
        "checkpoint": ckpt,
        "has_corrections": has_corr,
        "engine": engine,
    }


def has_spatial_combined_support(object_info: dict[str, Any] | None) -> bool:
    """Return True unless connected ``/object_info`` proves ``MiniMaxH3SpatialDirector`` lacks ``combined``/aux masks."""
    if not isinstance(object_info, dict):
        return True
    info = object_info.get("MiniMaxH3SpatialDirector")
    if not isinstance(info, dict):
        return True
    inp = info.get("input") or {}
    req = inp.get("required") or {}
    opt = inp.get("optional") or {}
    if not req and not opt:
        return True
    modes = _extract_combo_options(req.get("mode"))
    return ("combined" in modes) and ("include_mask" in opt) and ("exclude_mask" in opt)


def has_guide_latent_support(object_info: dict[str, Any] | None) -> bool:
    """Return True unless connected ``/object_info`` proves ``MiniMaxH3GuideLatent`` is absent."""
    if not isinstance(object_info, dict):
        return True
    if isinstance(object_info.get("MiniMaxH3GuideLatent"), dict):
        return True
    sd = object_info.get("MiniMaxH3SpatialDirector")
    if isinstance(sd, dict):
        inp = sd.get("input") or {}
        if inp.get("required") or inp.get("optional"):
            return False
    return True


def _normalize_sam3_points(raw_pts: Any, *, width: int, height: int) -> list[dict[str, int]]:
    """Convert ``[{"nx", "ny"}]`` or ``[{"x", "y"}]`` points into clamped integer pixel coords."""
    if not raw_pts:
        return []
    if not isinstance(raw_pts, (list, tuple)):
        raise ValueError("point list must be a JSON array")
    w_max = max(0, int(width) - 1)
    h_max = max(0, int(height) - 1)
    out: list[dict[str, int]] = []
    for item in raw_pts:
        if not isinstance(item, dict):
            continue
        if "nx" in item or "ny" in item:
            nx = max(0.0, min(1.0, float(item.get("nx", 0.5))))
            ny = max(0.0, min(1.0, float(item.get("ny", 0.5))))
            px = int(round(nx * float(w_max)))
            py = int(round(ny * float(h_max)))
        else:
            rx = float(item.get("x", 0.0))
            ry = float(item.get("y", 0.0))
            if bool(item.get("normalized", False)):
                px = int(round(max(0.0, min(1.0, rx)) * float(w_max)))
                py = int(round(max(0.0, min(1.0, ry)) * float(h_max)))
            else:
                px = int(round(rx))
                py = int(round(ry))
        out.append({"x": max(0, min(w_max, px)), "y": max(0, min(h_max, py))})
    return out


def compile_sam3_segment(
    params: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile an offline/live ComfyUI ``SAM3_Detect`` single-frame segmentation graph."""
    cfg = dict(params or {})
    image_ref = str(cfg.get("image_ref") or cfg.get("source_ref") or "").strip()
    if not image_ref:
        raise ValueError("image_ref is required for SAM3 segmentation")
    w = max(1, int(image_width))
    h = max(1, int(image_height))

    raw_pos = cfg.get("positive_points")
    if raw_pos is None and ("nx" in cfg or "ny" in cfg):
        raw_pos = [{"nx": float(cfg.get("nx", 0.5)), "ny": float(cfg.get("ny", 0.5))}]
    pos_px = _normalize_sam3_points(raw_pos, width=w, height=h)
    neg_px = _normalize_sam3_points(cfg.get("negative_points"), width=w, height=h)
    prompt_text = str(cfg.get("prompt") or "").strip()
    if not pos_px and not prompt_text:
        raise ValueError("At least one positive point or text prompt is required")

    cap = detect_sam3_capability(object_info)
    ckpt_name = str(cfg.get("checkpoint") or cap["checkpoint"] or _SAM3_PREFERRED_CKPTS[0])
    use_corrections = True if object_info is None else bool(cap["has_corrections"])

    graph: dict[str, Any] = {
        "200": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "202": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
    }
    if use_corrections:
        graph["201"] = {
            "class_type": "MiniMaxH3SAM31Corrections",
            "inputs": {
                "segmenter": ["200", 0],
                "segmenter_clip": ["200", 1],
                "correct_image_range": True,
                "correct_text_activation": True,
                "memory_selection": True,
            },
        }
        model_ref: list[Any] = ["201", 0]
        clip_ref: list[Any] = ["201", 1]
    else:
        model_ref = ["200", 0]
        clip_ref = ["200", 1]

    sam_inputs: dict[str, Any] = {
        "model": model_ref,
        "image": ["202", 0],
        "threshold": max(0.0, min(1.0, float(cfg.get("threshold", 0.5)))),
        "refine_iterations": max(0, min(5, int(cfg.get("refine_iterations", 2)))),
        "individual_masks": bool(cfg.get("individual_masks", False)),
    }
    if pos_px:
        graph["203"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(pos_px)}}
        sam_inputs["positive_coords"] = ["203", 0]
    if neg_px:
        graph["204"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(neg_px)}}
        sam_inputs["negative_coords"] = ["204", 0]
    if prompt_text and (not pos_px or bool(cfg.get("combine_text_and_points", False))):
        graph["205"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": clip_ref}}
        sam_inputs["conditioning"] = ["205", 0]

    graph["206"] = {"class_type": "SAM3_Detect", "inputs": sam_inputs}
    graph["207"] = {"class_type": "MaskToImage", "inputs": {"mask": ["206", 0]}}
    graph["208"] = {"class_type": "PreviewImage", "inputs": {"images": ["207", 0]}}
    return {
        "checkpoint": ckpt_name,
        "engine": "sam3.1" if "3.1" in ckpt_name else "sam3",
        "positive_points": pos_px,
        "negative_points": neg_px,
        "output_node_id": "208",
        "graph": graph,
    }


def video_preview_schedule(length: int) -> tuple[int, int]:
    """Return ``(preview_frames, preview_fps)`` for ``taeh3.safetensors`` at 1.0x real-time duration."""
    length_i = max(5, int(length))
    latent_steps = 5 * ((length_i - 5) // 17) + 2
    preview_frames = min(latent_steps, MAX_VIDEO_PREVIEW_FRAMES)
    if preview_frames >= latent_steps:
        preview_fps = 7  # ~7.06 Hz native H3 latent-step rate (39f->12@7fps, 90f->27@7fps)
    else:
        preview_fps = max(2, int(round(preview_frames * 24.0 / float(length_i))))
    return preview_frames, preview_fps


def _inject_kj_preview_override(
    graph: dict[str, Any],
    *,
    object_info: dict[str, Any] | None,
    tiny_vae_filename: str,
    preview_frames: int,
    preview_fps: int,
    enabled: bool = True,
) -> bool:
    """Insert node ``"165"`` right before guider ``"9"``'s ``model`` input when supported."""
    if not enabled:
        return False
    tiny_opt = resolve_kj_tiny_vae_option(object_info, tiny_vae_filename)
    if not tiny_opt:
        return False
    guider = graph.get("9")
    if not isinstance(guider, dict) or not isinstance(guider.get("inputs"), dict):
        return False
    prev_model = guider["inputs"].get("model")
    if prev_model is None:
        return False

    graph[PREVIEW_NODE_ID] = {
        "class_type": PREVIEW_NODE_CLASS,
        "inputs": {
            "model": copy.deepcopy(prev_model),
            "max_resolution": 672,
            "jpeg_quality": 80,
            "suppress_default_preview": True,
            "preview_frames": int(preview_frames),
            "preview_fps": int(preview_fps),
            "tiny_vae": tiny_opt,
        },
    }
    guider["inputs"]["model"] = [PREVIEW_NODE_ID, 0]
    return True


def _set_resolution_node(
    graph: dict[str, Any],
    *,
    width: int,
    height: int,
    length: int,
    canvas: str = "explicit",
) -> None:
    if width % 32 != 0 or height % 32 != 0 or width < 32 or height < 32:
        raise ValueError(f"Resolution {width}x{height} must be positive multiples of 32")
    node = graph.setdefault("27", {"class_type": "MiniMaxH3Resolution", "inputs": {}})
    if canvas == "same as the picture" and "15" in graph:
        node["inputs"] = {
            "shape": "same as the picture",
            "length": int(length),
            "picture": ["15", 0],
        }
    elif (width, height) == (1344, 768):
        node["inputs"] = {
            "shape": "wide",
            "shape.wide_resolution": "1344x768  7/4  1008 tok/frame  1.00x",
            "length": int(length),
        }
    elif (width, height) == (768, 768):
        node["inputs"] = {
            "shape": "square",
            "shape.square_resolution": "768x768  1  576 tok/frame  0.33x",
            "length": int(length),
        }
    else:
        node["inputs"] = {
            "shape": "custom",
            "shape.width": int(width),
            "shape.height": int(height),
            "length": int(length),
        }


def _rewrite_save_to_preview(graph: dict[str, Any]) -> None:
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") == "SaveImage":
            imgs = (node.get("inputs") or {}).get("images")
            node["class_type"] = "PreviewImage"
            node["inputs"] = {"images": imgs} if imgs is not None else {}


def _resolve_seed(params: dict[str, Any], template_graph: dict[str, Any]) -> int:
    raw = params.get("seed")
    if raw is not None and str(raw).strip() != "":
        val = int(raw)
        if val >= 0:
            return val
        return random.SystemRandom().randint(1, 2**31 - 1)
    n6 = template_graph.get("6", {}).get("inputs", {})
    if "noise_seed" in n6:
        return int(n6["noise_seed"])
    return random.SystemRandom().randint(1, 2**31 - 1)


def _aspect_shape_for_size(width: int | None, height: int | None, *, default: str = "4:3") -> str:
    """Map pixel dimensions to a valid ``MiniMaxH3ImageSize`` aspect ratio."""
    if not width or not height or int(width) <= 0 or int(height) <= 0:
        return default
    ratio = float(width) / float(height)
    if abs(ratio - 1.0) <= 0.12:
        return "1:1"
    if ratio >= 1.55:
        return "16:9"
    if ratio > 1.0:
        return "4:3"
    if ratio <= 1.0 / 1.55:
        return "9:16"
    return "3:4"


def _has_h3_video_sections(text: str) -> bool:
    low = (text or "").lower()
    return "integrated_multimodal_description:" in low and "overall_soundscape:" in low


def _has_h3_image_sections(text: str) -> bool:
    low = (text or "").lower()
    return (
        "subject_definitions:" in low
        or "retention_analysis:" in low
        or "detailed_description:" in low
        or "overall_soundscape:" in low
    )


def _describe_spatial_region(x: float, y: float) -> str:
    """Return a clean human-readable region description from normalized ``(x, y)``."""
    xc = max(0.0, min(1.0, float(x)))
    yc = max(0.0, min(1.0, float(y)))
    horiz = "left" if xc < 0.36 else ("right" if xc > 0.64 else "center")
    vert = "upper" if yc < 0.36 else ("lower" if yc > 0.64 else "middle")
    if horiz == "center" and vert == "middle":
        return "the center of the frame"
    if vert == "middle":
        return f"the {horiz} midground"
    if horiz == "center":
        return f"the {vert} center of the frame"
    return f"the {vert}-{horiz} region"


def _portal_active_span(
    events_doc: dict[str, Any],
    portal: list[dict[str, Any]],
    max_f: int,
) -> tuple[int, int]:
    """Return ``(start_frame, end_frame)`` for portal conditioning, holding a single keyframe across the clip."""
    raw_s = events_doc.get("portal_start")
    raw_e = events_doc.get("portal_end")
    default_s = int(round(float(portal[0].get("frame", 0)))) if portal else 0
    if raw_e is not None:
        default_e = int(round(float(raw_e)))
    elif len(portal) >= 2:
        default_e = int(round(float(portal[-1].get("frame", max_f))))
    else:
        default_e = max_f
    p_start = max(0, min(max_f, int(round(float(raw_s))) if raw_s is not None else default_s))
    p_end = max(p_start, min(max_f, default_e))
    return p_start, p_end


def _collect_spatial_beats(
    events_doc: dict[str, Any],
    *,
    director_mode: str,
    length: int,
    fallback_prompt: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Collect chronologically ordered spatial event beats and optional portal motion note."""
    max_f = max(0, int(length) - 1)
    targets = sorted(
        (t for t in (events_doc.get("targets") or ()) if isinstance(t, dict)),
        key=lambda t: (int(round(float(t.get("start", 0)))), int(t.get("id", 0))),
    )
    portal = sorted(
        (k for k in (events_doc.get("portal") or ()) if isinstance(k, dict)),
        key=lambda k: int(round(float(k.get("frame", 0)))),
    )

    fb = str(fallback_prompt or "").strip()
    beats: list[dict[str, Any]] = []
    trajectory_note = ""

    if len(portal) >= 2:
        k0, k1 = portal[0], portal[-1]
        dx = float(k1.get("x", 0.5)) - float(k0.get("x", 0.5))
        dy = float(k1.get("y", 0.5)) - float(k0.get("y", 0.5))
        if math.hypot(dx, dy) >= 0.14:
            r0 = _describe_spatial_region(float(k0.get("x", 0.5)), float(k0.get("y", 0.5)))
            r1 = _describe_spatial_region(float(k1.get("x", 0.5)), float(k1.get("y", 0.5)))
            t0 = f"{max(0, int(round(float(k0.get('frame', 0))))) / 24.0:.2f}"
            t1 = f"{max(0, int(round(float(k1.get('frame', max_f))))) / 24.0:.2f}"
            trajectory_note = f"sweeping smoothly from {r0} at {t0}s to {r1} at {t1}s"

    if director_mode in ("point_targets", "combined") and targets:
        has_custom_target_prompt = any(str(t.get("prompt") or "").strip() for t in targets)
        for t in targets:
            s_f = max(0, min(max_f, int(round(float(t.get("start", 0))))))
            e_f = max(s_f, min(max_f, int(round(float(t.get("end", max_f))))))
            reg = _describe_spatial_region(float(t.get("x", 0.5)), float(t.get("y", 0.5)))
            cue = str(t.get("cue") or "").strip().lower()
            cue = "" if cue == "none" else cue
            t_prompt = str(t.get("prompt") or "").strip()
            beats.append(
                {
                    "kind": "target",
                    "start": s_f,
                    "end": e_f,
                    "region": reg,
                    "prompt": t_prompt or fb,
                    "explicit_prompt": bool(t_prompt),
                    "cue": cue,
                }
            )
        if director_mode == "combined" and portal:
            p_explicit = [str(k.get("prompt") or "").strip() for k in portal if str(k.get("prompt") or "").strip()]
            p_start, p_end = _portal_active_span(events_doc, portal, max_f)
            mid_k = portal[len(portal) // 2] if len(portal) > 1 else portal[0]
            p_reg = _describe_spatial_region(float(mid_k.get("x", 0.5)), float(mid_k.get("y", 0.5)))
            if p_explicit or (fb and has_custom_target_prompt):
                beats.append(
                    {
                        "kind": "portal",
                        "start": p_start,
                        "end": p_end,
                        "region": trajectory_note or p_reg,
                        "prompt": p_explicit[0] if p_explicit else fb,
                        "explicit_prompt": bool(p_explicit),
                        "cue": "",
                    }
                )
        beats.sort(key=lambda b: (b["start"], b["end"]))
        return beats, (trajectory_note if director_mode == "combined" else "")

    if portal:
        explicit_kfs = [
            (
                max(0, min(max_f, int(round(float(k.get("frame", 0)))))),
                k,
                str(k.get("prompt") or "").strip(),
            )
            for k in portal
            if str(k.get("prompt") or "").strip()
        ]
        distinct_prompts = {p for _, _, p in explicit_kfs if p}
        if len(explicit_kfs) >= 2 and len(distinct_prompts) >= 2:
            for idx, (kf_f, k_item, k_prompt) in enumerate(explicit_kfs):
                next_f = explicit_kfs[idx + 1][0] if idx + 1 < len(explicit_kfs) else max_f
                end_f = max(kf_f, next_f - 1 if (idx + 1 < len(explicit_kfs) and next_f > kf_f) else max_f)
                beats.append(
                    {
                        "kind": "portal",
                        "start": kf_f,
                        "end": end_f,
                        "region": _describe_spatial_region(float(k_item.get("x", 0.5)), float(k_item.get("y", 0.5))),
                        "prompt": k_prompt,
                        "explicit_prompt": True,
                        "cue": "",
                    }
                )
            return beats, trajectory_note

        mid_k = portal[len(portal) // 2] if len(portal) > 1 else portal[0]
        p_start, p_end = _portal_active_span(events_doc, portal, max_f)
        p_prompt = explicit_kfs[0][2] if explicit_kfs else fb
        beats.append(
            {
                "kind": "portal",
                "start": p_start,
                "end": p_end,
                "region": _describe_spatial_region(float(mid_k.get("x", 0.5)), float(mid_k.get("y", 0.5))),
                "prompt": p_prompt,
                "explicit_prompt": bool(explicit_kfs),
                "cue": "",
            }
        )
    return beats, trajectory_note


def _events_have_per_event_prompts(events_doc: dict[str, Any]) -> bool:
    for key in ("targets", "portal"):
        for item in events_doc.get(key) or ():
            if isinstance(item, dict) and str(item.get("prompt") or "").strip():
                return True
    return False


def _summarize_focus_region_and_time(
    events_doc: dict[str, Any],
    *,
    director_mode: str,
    length: int,
) -> tuple[str, int, int, list[str], str]:
    """Extract primary normalized focus coordinates, onset/offset frames, cue words, and summary label."""
    fx, fy = 0.5, 0.5
    onset = 0
    offset = max(0, int(length) - 1)
    cues: list[str] = []

    targets = [t for t in (events_doc.get("targets") or ()) if isinstance(t, dict)]
    portal = [k for k in (events_doc.get("portal") or ()) if isinstance(k, dict)]

    if director_mode in ("point_targets", "combined") and targets:
        primary_t = targets[0]
        fx = float(primary_t.get("x", 0.5))
        fy = float(primary_t.get("y", 0.5))
        onset = max(0, min(int(length) - 1, int(round(float(primary_t.get("start", 0))))))
        offset = max(onset, min(int(length) - 1, int(round(float(primary_t.get("end", length - 1))))))
        for t in targets:
            preset = str(t.get("cue") or "").strip().lower()
            if preset and preset != "none" and preset not in cues:
                cues.append(preset)
    elif portal:
        mid_k = portal[len(portal) // 2] if len(portal) > 1 else portal[0]
        fx = float(mid_k.get("x", 0.5))
        fy = float(mid_k.get("y", 0.5))
        onset, offset = _portal_active_span(events_doc, portal, max(0, int(length) - 1))

    for key in ("portal_cues", "cues", "sound_cues"):
        for item in events_doc.get(key) or ():
            if isinstance(item, dict):
                preset = str(item.get("preset") or item.get("cue") or "").strip().lower()
                if preset and preset != "none" and preset not in cues:
                    cues.append(preset)

    region_phrase = _describe_spatial_region(fx, fy)
    summary = f"{region_phrase} @ f{onset}..{offset}"
    return region_phrase, onset, offset, cues, summary


def synthesize_spatial_prompts(
    *,
    mode: str,
    length: int,
    has_source_ref: bool,
    director_mode: str,
    events_doc: dict[str, Any],
    original_prompt: str = "",
    target_prompt: str = "",
    same_prompt: bool = False,
    edit_lines: str = "",
    has_sprite: bool = False,
    reaction_clauses: list[str] | None = None,
) -> tuple[str, str, str]:
    """Build clean, zero-hardcoded ``(original_prompt, edited_prompt, region_summary)``."""
    use_pic1 = bool(has_source_ref and mode in ("i2v_portal", "r2v_portal", "world_targets"))
    dur_s = f"{float(length) / 24.0:.3f}".rstrip("0").rstrip(".")
    region_phrase, onset, offset, cues, region_summary = _summarize_focus_region_and_time(
        events_doc,
        director_mode=director_mode,
        length=length,
    )
    onset_s = f"{float(onset) / 24.0:.2f}"
    offset_s = f"{float(offset) / 24.0:.2f}"

    raw_orig = (original_prompt or "").strip()
    if raw_orig:
        if _has_h3_video_sections(raw_orig) or edit_lines.strip():
            orig_out = raw_orig
        elif use_pic1:
            orig_out = (
                f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, "
                f"layout, and composition established in <Picture 1>, a steady {dur_s}-second shot of {raw_orig}. "
                f"Subtle natural micro-motion, light play, and environmental poise continue calmly while camera framing, "
                f"surfaces, and surrounding structure match <Picture 1> faithfully from first frame to last.\n\n"
                f"overall_soundscape:\nQuiet natural ambient room tone matching the scene acoustics with faint realistic textures.\n\n"
                f"non_diegetic_music:\nN/A"
            )
        else:
            orig_out = (
                f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, steady {dur_s}-second static shot showing "
                f"{raw_orig}. Balanced directional lighting, consistent architectural and natural textures, and calm ambient motion "
                f"hold cleanly across the full frame without sudden changes.\n\n"
                f"overall_soundscape:\nQuiet ambient room tone matching the visual space.\n\n"
                f"non_diegetic_music:\nN/A"
            )
    else:
        if use_pic1:
            sprite_note = (
                " Any foreground movable actor stays naturally grounded inside the space with realistic contact shadows and scale."
                if has_sprite
                else ""
            )
            orig_out = (
                f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, palette, "
                f"layout, and camera perspective established in <Picture 1>, a calm, steady {dur_s}-second shot unfolds while the scene "
                f"in <Picture 1> remains intact with gentle natural ambient micro-motion, subtle air and surface light continuity, "
                f"and zero unexpected visual disruptions.{sprite_note} The camera holds completely steady so every spatial edge aligns accurately with <Picture 1>.\n\n"
                f"overall_soundscape:\nQuiet natural ambient room tone matching the visual environment of <Picture 1>.\n\n"
                f"non_diegetic_music:\nN/A"
            )
        else:
            orig_out = (
                f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, steady {dur_s}-second static wide shot "
                f"with consistent natural lighting, cohesive spatial depth, and calm ambient continuity across the scene.\n\n"
                f"overall_soundscape:\nQuiet natural ambient room tone.\n\n"
                f"non_diegetic_music:\nN/A"
            )

    if same_prompt:
        return orig_out, orig_out, region_summary

    raw_target = (target_prompt or "").strip()
    if raw_target and _has_h3_video_sections(raw_target):
        edited_out = raw_target
    else:
        beats, trajectory_note = _collect_spatial_beats(
            events_doc,
            director_mode=director_mode,
            length=length,
            fallback_prompt=raw_target,
        )
        sprite_react = (
            " Nearby subjects react dynamically with natural illumination bounce, air impulse, and cast shadows."
            if has_sprite
            else ""
        )
        cue_audio_parts = [c.replace("_", " ") for c in cues]

        if len(beats) >= 2:
            visual_clauses: list[str] = []
            audio_clauses: list[str] = []
            for b in beats:
                b_on = f"{float(b['start']) / 24.0:.2f}"
                b_off = f"{float(b['end']) / 24.0:.2f}"
                b_act = str(b.get("prompt") or "").strip() or "localized transformation unfolding in place"
                b_reg = str(b.get("region") or region_phrase)
                visual_clauses.append(f"from {b_on}s to {b_off}s inside {b_reg}, {b_act}")
                cue_tag = f" ({b['cue'].replace('_', ' ')})" if b.get("cue") else ""
                audio_clauses.append(f"around {b_on} seconds{cue_tag} matching {b_act}")

            timeline_visual = "; ".join(visual_clauses)
            motion_note = f" The focal effect follows a path {trajectory_note}." if trajectory_note else ""
            audio_desc = f"Staggered localized acoustic events erupt {', and '.join(audio_clauses)} atop consistent natural scene ambience."
            region_summary = f"{len(beats)} beats @ f{beats[0]['start']}..{beats[-1]['end']}"

            if use_pic1:
                edited_out = (
                    f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                    f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, camera angle, "
                    f"palette, and spatial layout established in <Picture 1> across the {dur_s}-second take. "
                    f"In chronological sequence, {timeline_visual}, each interacting naturally with nearby surfaces, lighting, and reflections."
                    f"{motion_note}{sprite_react} Outside the active regions, all architecture, horizon, lighting, and surrounding scenery "
                    f"from <Picture 1> remain faithfully preserved.\n\n"
                    f"overall_soundscape:\n{audio_desc}\n\n"
                    f"non_diegetic_music:\nN/A"
                )
            else:
                edited_out = (
                    f"integrated_multimodal_description: [Shot 1] Live-action, cinematic {dur_s}-second shot. "
                    f"In chronological sequence, {timeline_visual}, casting realistic lighting, motion, and physical interaction "
                    f"into the surrounding environment while camera framing stays steady.{motion_note}\n\n"
                    f"overall_soundscape:\n{audio_desc}\n\n"
                    f"non_diegetic_music:\nN/A"
                )
        else:
            single_prompt = str(beats[0].get("prompt") or "").strip() if beats else ""
            action_desc = single_prompt or raw_target or "dynamic localized transformation unfolding inside the active region"
            motion_note = f" while {trajectory_note}" if trajectory_note else ""
            audio_desc = (
                f"Crisp localized acoustic event matching {action_desc} ({', '.join(cue_audio_parts)}) erupting around {onset_s} seconds atop consistent scene ambience."
                if cue_audio_parts
                else f"Localized sound effects matching {action_desc} arriving around {onset_s} seconds over subtle natural scene ambience."
            )
            if use_pic1:
                edited_out = (
                    f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
                    f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, camera angle, "
                    f"palette, and spatial layout established in <Picture 1> across the {dur_s}-second take. "
                    f"Focused inside {region_phrase} from {onset_s}s to {offset_s}s{motion_note}, {action_desc}, interacting naturally with nearby surfaces, "
                    f"lighting, and reflections.{sprite_react} Outside the active region, all architecture, horizon, lighting, and surrounding scenery "
                    f"from <Picture 1> remain faithfully preserved.\n\n"
                    f"overall_soundscape:\n{audio_desc}\n\n"
                    f"non_diegetic_music:\nN/A"
                )
            else:
                edited_out = (
                    f"integrated_multimodal_description: [Shot 1] Live-action, cinematic {dur_s}-second shot. "
                    f"Focused at {region_phrase} between {onset_s}s and {offset_s}s{motion_note}, {action_desc}, casting realistic lighting, motion, "
                    f"and physical interaction into the surrounding environment while camera framing stays steady.\n\n"
                    f"overall_soundscape:\n{audio_desc}\n\n"
                    f"non_diegetic_music:\nN/A"
                )

    clauses = [str(c).strip() for c in (reaction_clauses or []) if str(c).strip()]
    if clauses:
        suffix = " ".join(clauses)
        if suffix not in edited_out:
            edited_out = f"{edited_out.strip()} {suffix}".strip()

    return orig_out, edited_out, region_summary


def synthesize_image_edit_prompts(
    *,
    workflow_key: str,
    target_prompt: str,
    original_prompt: str = "",
    region_x: float = 0.5,
    region_y: float = 0.5,
) -> tuple[str, str, str]:
    """Build contract-matched ``(original_prompt, edited_prompt, region_phrase)`` for masked still edits."""
    region_phrase = _describe_spatial_region(region_x, region_y)
    edit_text = (target_prompt or "").strip()
    orig_text = (original_prompt or "").strip()

    if workflow_key == "pika_image":
        if orig_text and _has_h3_image_sections(orig_text):
            orig_out = orig_text
        else:
            desc = orig_text or "Everything in <Picture 1> remains completely unchanged with matching framing, lighting, textures, and depth."
            orig_out = (
                "subject_definitions:\n"
                "<Picture 1> is the source image and composition anchor.\n\n"
                "summary:\n"
                "[keyframe completion] Hold <Picture 1> completely unchanged.\n\n"
                "retention_analysis:\n"
                "<Picture 1> ([Shot 1] edited keyframe): fully_preserved - exact framing, subjects, lighting, and background remain intact.\n\n"
                "detailed_description:\n"
                f"Photographic still frame. [Shot 1] {desc} The camera is static and nothing moves; the frame holds this single still image.\n\n"
                "overall_soundscape:\nN/A\n\n"
                "non_diegetic_music:\nN/A"
            )
        if edit_text and _has_h3_image_sections(edit_text):
            edited_out = edit_text
        else:
            edited_out = (
                "subject_definitions:\n"
                "<Picture 1> is the source image and composition anchor.\n"
                f"<Subject 1> is {region_phrase} of <Picture 1>.\n\n"
                "summary:\n"
                f"[keyframe completion] Edit <Picture 1> so <Subject 1> becomes: {edit_text}.\n\n"
                "retention_analysis:\n"
                "<Picture 1> ([Shot 1] edited keyframe): partially_preserved - framing, camera angle, and surrounding lighting are kept.\n"
                f"<Subject 1> (appears in [Shot 1]): weak_reference - replaced in {region_phrase} with {edit_text}.\n\n"
                "detailed_description:\n"
                f"Photographic still frame. [Shot 1] All unmasked scenery and lighting of <Picture 1> remain intact. "
                f"Inside <Subject 1> ({region_phrase}), {edit_text}, matching the ambient key light, shadows, depth of field, "
                f"and realistic texture of <Picture 1>. The camera is static and nothing moves; the frame holds this single still image.\n\n"
                "overall_soundscape:\nN/A\n\n"
                "non_diegetic_music:\nN/A"
            )
        return orig_out, edited_out, region_phrase

    # MiniMax-H3-Image and P-H3-Image use sectionless visual prose with zero soundscape rows.
    if orig_text:
        orig_out = orig_text if "<Picture 1>" in orig_text else (
            f"The scene from <Picture 1> showing {orig_text}, preserved completely with identical framing, composition, lighting, and details."
        )
    else:
        orig_out = (
            "The complete photograph <Picture 1> is preserved with identical framing, camera perspective, "
            "subjects, surface textures, lighting, and background with zero edits or additions."
        )

    if "<Picture 1>" in edit_text and len(edit_text) >= 120:
        edited_out = edit_text
    else:
        edited_out = (
            f"The region at {region_phrase} of <Picture 1> is <Subject 1>. "
            f"The remaining framing, background, subjects, and ambient lighting of <Picture 1> are <Subject 2>. "
            f"The target image preserves <Subject 2> completely while transforming <Subject 1> into {edit_text}, "
            f"seamlessly matching the directional light, cast shadows, color balance, focus depth, and photographic detail of <Picture 1>."
        )
    return orig_out, edited_out, region_phrase


def compile_still(
    params: dict[str, Any] | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile an anchor still or stamp-cutout source generation workflow."""
    cfg = dict(params or {})
    workflow_key = str(cfg.get("workflow") or "h3_image").strip()
    if workflow_key not in ANCHOR_WORKFLOW_TEMPLATES:
        raise ValueError(
            f"Unknown anchor workflow {workflow_key!r}; expected one of {list(ANCHOR_WORKFLOW_TEMPLATES)}"
        )
    tpl_path = Path(ANCHOR_WORKFLOW_TEMPLATES[workflow_key]["path"])
    if not tpl_path.is_file():
        raise FileNotFoundError(f"Anchor workflow template missing: {tpl_path}")
    graph = _load_json(tpl_path)
    _rewrite_save_to_preview(graph)

    prompt_text = str(cfg.get("prompt") or "").strip()
    if not prompt_text:
        raise ValueError("Prompt cannot be empty when generating a still image")

    for_cutout = bool(cfg.get("for_cutout", False))
    if for_cutout and "backdrop" not in prompt_text.lower():
        prompt_text = (
            f"{prompt_text}, isolated centered full view on a clean uniform light grey studio backdrop, "
            f"crisp silhouette edges, soft neutral studio lighting, single subject"
        )

    seed = _resolve_seed(cfg, graph)
    if "6" in graph and isinstance(graph["6"].get("inputs"), dict):
        graph["6"]["inputs"]["noise_seed"] = seed

    if "5" in graph and isinstance(graph["5"].get("inputs"), dict):
        graph["5"]["inputs"]["prompt"] = prompt_text

    width = cfg.get("width")
    height = cfg.get("height")
    if workflow_key == "pika_image":
        if width is not None and height is not None:
            _set_resolution_node(graph, width=int(width), height=int(height), length=362)
    else:
        n4 = graph.get("4", {}).get("inputs", {})
        if isinstance(n4, dict):
            if cfg.get("shape"):
                n4["shape"] = str(cfg["shape"])
            elif for_cutout:
                n4["shape"] = "1:1"
            elif width is not None and height is not None:
                n4["shape"] = _aspect_shape_for_size(int(width), int(height), default=str(n4.get("shape") or "4:3"))
            mp_val = float(cfg["megapixels"]) if cfg.get("megapixels") is not None else float(n4.get("megapixels", 4.0))
            n4["megapixels"] = max(4.0, mp_val)
            default_gen_mp = 16.0 if workflow_key == "h3_image" else 8.0
            gen_mp_val = (
                float(cfg["generate_megapixels"])
                if cfg.get("generate_megapixels") is not None
                else float(n4.get("generate_megapixels", default_gen_mp))
            )
            n4["generate_megapixels"] = max(4.0, max(n4["megapixels"], gen_mp_val))

    preview_enabled = bool(cfg.get("preview_override", True))
    injected = _inject_kj_preview_override(
        graph,
        object_info=object_info,
        tiny_vae_filename=TAEH3_IMAGE_FILE,
        preview_frames=1,
        preview_fps=1,
        enabled=preview_enabled,
    )
    return {
        "workflow": workflow_key,
        "seed": seed,
        "for_cutout": for_cutout,
        "prompt": prompt_text,
        "preview_override_injected": injected,
        "graph": graph,
    }


def compile_masked_still_edit(
    params: dict[str, Any] | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a 1-frame masked image edit with ``MiniMaxH3EditGuider`` + ``taeh3_image`` preview."""
    cfg = dict(params or {})
    workflow_key = resolve_image_edit_workflow(
        cfg.get("workflow"),
        resident_backbone=cfg.get("resident_backbone"),
    )
    if workflow_key not in IMAGE_EDIT_WORKFLOW_TEMPLATES:
        raise ValueError(
            f"Unknown image edit workflow {workflow_key!r}; expected one of {list(IMAGE_EDIT_WORKFLOW_TEMPLATES)}"
        )
    tpl_path = Path(IMAGE_EDIT_WORKFLOW_TEMPLATES[workflow_key]["path"])
    if not tpl_path.is_file():
        raise FileNotFoundError(f"Image edit template missing: {tpl_path}")

    source_ref = str(cfg.get("source_ref") or cfg.get("ref_image") or cfg.get("anchor_image") or "").strip()
    if not source_ref:
        raise ValueError("source_ref is required for masked still edit preview")
    mask_ref = str(cfg.get("mask_ref") or cfg.get("mask_image") or "").strip()
    if not mask_ref:
        raise ValueError("mask_ref is required for masked still edit preview")
    target_prompt = str(cfg.get("target_prompt") or cfg.get("prompt") or "").strip()
    if not target_prompt:
        raise ValueError("target_prompt cannot be empty for masked still edit preview")

    region_x = float(cfg.get("region_x", 0.5))
    region_y = float(cfg.get("region_y", 0.5))
    orig_prompt, edited_prompt, region_phrase = synthesize_image_edit_prompts(
        workflow_key=workflow_key,
        target_prompt=target_prompt,
        original_prompt=str(cfg.get("original_prompt") or "").strip(),
        region_x=region_x,
        region_y=region_y,
    )

    graph = _load_json(tpl_path)
    _rewrite_save_to_preview(graph)

    for stale in ("165", "170", "171", "130", "131"):
        graph.pop(stale, None)

    seed = _resolve_seed(cfg, graph)
    if "6" in graph and isinstance(graph["6"].get("inputs"), dict):
        graph["6"]["inputs"]["noise_seed"] = seed

    graph["15"] = {"class_type": "LoadImage", "inputs": {"image": source_ref}}
    graph["170"] = {"class_type": "LoadImage", "inputs": {"image": mask_ref}}
    graph["171"] = {"class_type": "ImageToMask", "inputs": {"image": ["170", 0], "channel": "red"}}

    if workflow_key == "pika_image":
        if cfg.get("width") is not None and cfg.get("height") is not None and cfg.get("canvas") == "explicit":
            _set_resolution_node(graph, width=int(cfg["width"]), height=int(cfg["height"]), length=362, canvas="explicit")
        else:
            _set_resolution_node(graph, width=1344, height=768, length=362, canvas="same as the picture")
        graph["132"] = {
            "class_type": "MiniMaxH3EncodeReferences",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "references": ["50", 0],
                "width": ["27", 0],
                "height": ["27", 1],
                "length": ["27", 2],
                "video_policy": "comfy",
                "image_policy": "comfy",
                "single_frame": True,
            },
        }
        graph["5"] = {
            "class_type": "MiniMaxH3PromptOnReferences",
            "inputs": {"encoded_references": ["132", 0], "prompt": orig_prompt},
        }
        graph["131"] = {
            "class_type": "MiniMaxH3PromptOnReferences",
            "inputs": {"encoded_references": ["132", 0], "prompt": edited_prompt},
        }
        graph["26"] = {
            "class_type": "MiniMaxH3Preflight",
            "inputs": {
                "conditioning": ["5", 0],
                "samples": ["132", 1],
                "edited_conditioning": ["131", 0],
            },
        }
        graph["10"]["inputs"]["latent_image"] = ["26", 1]
        orig_cond_ref: list[Any] = ["26", 0]
        edited_cond_ref: list[Any] = ["26", 2]
    elif workflow_key == "ph3_image":
        graph.pop("16", None)
        n4 = graph.get("4", {}).get("inputs", {})
        if isinstance(n4, dict):
            n4["shape"] = "same as the picture"
            n4["picture"] = ["15", 0]
            mp_val = float(cfg["megapixels"]) if cfg.get("megapixels") is not None else 4.0
            n4["megapixels"] = max(4.0, mp_val)
            gen_mp_val = float(cfg["generate_megapixels"]) if cfg.get("generate_megapixels") is not None else 4.0
            n4["generate_megapixels"] = max(4.0, max(n4["megapixels"], gen_mp_val))
        n5_in = graph["5"]["inputs"]
        n5_in.pop("images.image_2", None)
        n5_in["images.image_1"] = ["15", 0]
        n5_in["prompt"] = orig_prompt
        n131_in = copy.deepcopy(n5_in)
        n131_in["prompt"] = edited_prompt
        graph["131"] = {"class_type": "MiniMaxH3PH3ImageConditioning", "inputs": n131_in}
        graph["10"]["inputs"]["latent_image"] = ["5", 1]
        orig_cond_ref = ["5", 0]
        edited_cond_ref = ["131", 0]
    else:  # h3_image (default MiniMax-H3-Image)
        n4 = graph.get("4", {}).get("inputs", {})
        if isinstance(n4, dict):
            n4["shape"] = "same as the picture"
            n4["picture"] = ["15", 0]
            mp_val = float(cfg["megapixels"]) if cfg.get("megapixels") is not None else 4.0
            n4["megapixels"] = max(4.0, mp_val)
            gen_mp_val = float(cfg["generate_megapixels"]) if cfg.get("generate_megapixels") is not None else 4.0
            n4["generate_megapixels"] = max(4.0, max(n4["megapixels"], gen_mp_val))
        graph["5"]["inputs"]["prompt"] = orig_prompt
        graph["131"] = {
            "class_type": "MiniMaxH3PromptOnImageReferences",
            "inputs": {
                "encoded_references": ["132", 0],
                "prompt": edited_prompt,
                "noise": ["6", 0],
            },
        }
        orig_cond_ref = ["5", 0]
        edited_cond_ref = ["131", 0]

    model_ref = copy.deepcopy((graph.get("9", {}).get("inputs") or {}).get("model") or ["58", 0])
    scale = max(0.0, min(4.0, float(cfg.get("scale", 1.8))))
    first_step = max(0, int(cfg.get("first_step", 0)))
    last_step = max(first_step, int(cfg.get("last_step", 1000)))
    graph["9"] = {
        "class_type": "MiniMaxH3EditGuider",
        "inputs": {
            "model": model_ref,
            "original": orig_cond_ref,
            "edited": edited_cond_ref,
            "scale": scale,
            "first_step": first_step,
            "align_positions": False,
            "last_step": last_step,
            "streams": "video_only",
            "audio_scale": 0.0,
            "mask": ["171", 0],
        },
    }

    preview_enabled = bool(cfg.get("preview_override", True))
    injected = _inject_kj_preview_override(
        graph,
        object_info=object_info,
        tiny_vae_filename=TAEH3_IMAGE_FILE,
        preview_frames=1,
        preview_fps=1,
        enabled=preview_enabled,
    )
    return {
        "workflow": workflow_key,
        "source_ref": source_ref,
        "mask_ref": mask_ref,
        "seed": seed,
        "scale": scale,
        "region_phrase": region_phrase,
        "original_prompt": orig_prompt,
        "edited_prompt": edited_prompt,
        "preview_override_injected": injected,
        "graph": graph,
    }


def _parse_events_doc(raw_events: Any, fallback_raw: str) -> tuple[dict[str, Any], str]:
    if raw_events is None:
        raw_str = fallback_raw
        return json.loads(raw_str), raw_str
    if isinstance(raw_events, str):
        doc = json.loads(raw_events)
        if not isinstance(doc, dict):
            raise ValueError("events_json must decode to a JSON object")
        return doc, raw_events
    if isinstance(raw_events, dict):
        doc = copy.deepcopy(raw_events)
        return doc, json.dumps(doc, indent=2)
    raise ValueError("events_json must be a JSON string or dict")


def _events_have_sound_cues(doc: dict[str, Any], params: dict[str, Any], *, director_mode: str = "portal_path") -> bool:
    explicit = params.get("sound_cues_enabled")
    if explicit is False:
        return False
    if director_mode in ("point_targets", "combined"):
        for t in doc.get("targets") or ():
            if isinstance(t, dict) and str(t.get("cue") or "none").strip().lower() not in ("", "none"):
                return True
    for key in ("portal_cues", "cues", "sound_cues"):
        items = doc.get(key) or params.get(key)
        if isinstance(items, list):
            for c in items:
                if isinstance(c, dict):
                    preset = str(c.get("preset") or c.get("cue") or "none").strip().lower()
                    if preset not in ("", "none"):
                        return True
    return bool(explicit)


def _feather_value(params: dict[str, Any], width: int, height: int, default_feather: float = 0.08) -> float:
    if params.get("feather") is not None:
        base = max(0.0, min(0.35, float(params["feather"])))
    elif params.get("feather_px") is not None:
        short_side = max(1, min(int(width), int(height)))
        base = max(0.0, min(0.35, float(params["feather_px"]) / float(short_side)))
    else:
        base = float(default_feather)
    if params.get("halo_radius") is not None:
        halo_val = max(0.0, min(0.35, float(params["halo_radius"])))
        if halo_val > 0.0:
            base = max(base, halo_val)
    return base


def _merge_dpad_inputs(events_doc: dict[str, Any], params: dict[str, Any]) -> bool:
    """Normalize optional ``dpad`` / ``dpad_keyframes`` into ``events_doc['dpad']``."""
    raw_dpad = params.get("dpad")
    raw_kf = params.get("dpad_keyframes")
    if isinstance(raw_dpad, dict):
        events_doc["dpad"] = copy.deepcopy(raw_dpad)
    if isinstance(raw_kf, dict):
        dpad_obj = events_doc.setdefault("dpad", {})
        if "waypoints" in raw_kf:
            dpad_obj["waypoints"] = copy.deepcopy(raw_kf["waypoints"])
        if "hops" in raw_kf:
            dpad_obj["hops"] = copy.deepcopy(raw_kf["hops"])
    elif isinstance(raw_kf, list):
        dpad_obj = events_doc.setdefault("dpad", {})
        dpad_obj["waypoints"] = copy.deepcopy(raw_kf)
    dpad_doc = events_doc.get("dpad")
    has_wps = isinstance(dpad_doc, dict) and bool(dpad_doc.get("waypoints") or dpad_doc.get("hops"))
    return bool(has_wps or raw_dpad is not None or raw_kf is not None)


def _apply_halo_couplings(
    events_doc: dict[str, Any],
    params: dict[str, Any],
    *,
    length: int,
    sprite_active: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministically offset post-onset sprite waypoints along portal->sprite radial vector."""
    raw_couplings = params.get("couplings") or []
    if not isinstance(raw_couplings, list) or not raw_couplings or not sprite_active:
        return [], []

    portals = events_doc.get("portal") or []
    applied: list[dict[str, Any]] = []
    reaction_clauses: list[str] = []

    raw_sprite_pos = params.get("sprite_pos")
    init_sx: float | None = None
    init_sy: float | None = None
    if isinstance(raw_sprite_pos, dict):
        init_sx = float(raw_sprite_pos.get("x", 0.5))
        init_sy = float(raw_sprite_pos.get("y", 0.72))
    elif isinstance(raw_sprite_pos, (list, tuple)) and len(raw_sprite_pos) >= 2:
        init_sx = float(raw_sprite_pos[0])
        init_sy = float(raw_sprite_pos[1])

    for entry in raw_couplings:
        if not isinstance(entry, dict):
            continue
        target_kind = str(entry.get("target_kind") or "sprite").strip().lower()
        if target_kind != "sprite":
            continue
        strength = max(0.0, min(1.0, float(entry.get("strength", 0.0))))
        if strength <= 0.0:
            continue
        c_mode = str(entry.get("mode") or "push").strip().lower()
        if c_mode not in ("push", "pull"):
            raise ValueError(f"Unsupported coupling mode {c_mode!r}; expected 'push' or 'pull'")

        p_idx = int(entry.get("portal_index", 0))
        if isinstance(portals, list) and portals:
            idx_clamped = max(0, min(len(portals) - 1, p_idx))
            portal_item = portals[idx_clamped] if isinstance(portals[idx_clamped], dict) else {}
        else:
            portal_item = {}
        x_p = float(portal_item.get("x", 0.5))
        y_p = float(portal_item.get("y", 0.5))
        onset = max(0, min(length - 1, int(round(float(portal_item.get("frame", 0))))))

        dpad_doc = events_doc.setdefault("dpad", {"waypoints": [], "hops": []})
        wps = dpad_doc.setdefault("waypoints", [])
        if not wps:
            sx0 = init_sx if init_sx is not None else 0.5
            sy0 = init_sy if init_sy is not None else 0.72
            wps.append({"frame": onset, "x": round(sx0, 4), "y": round(sy0, 4), "face": "right", "scale": 1.0})

        first_wp = wps[0] if isinstance(wps[0], dict) else {}
        x_s = init_sx if init_sx is not None else float(first_wp.get("x", 0.5))
        y_s = init_sy if init_sy is not None else float(first_wp.get("y", 0.72))

        vx = x_s - x_p
        vy = y_s - y_p
        dist = math.hypot(vx, vy)
        if dist > 1e-6:
            ux, uy = vx / dist, vy / dist
        else:
            ux, uy = 1.0, 0.0

        sign = -1.0 if c_mode == "pull" else 1.0
        disp = 0.12 * strength
        dx = sign * ux * disp
        dy = sign * uy * disp

        post_wps = [wp for wp in wps if isinstance(wp, dict) and int(round(float(wp.get("frame", 0)))) > onset]
        if not post_wps:
            post_f = min(length - 1, max(onset + 6, length // 2))
            if post_f > onset:
                new_wp = {"frame": post_f, "x": round(x_s, 4), "y": round(y_s, 4), "face": str(first_wp.get("face", "right")), "scale": float(first_wp.get("scale", 1.0))}
                wps.append(new_wp)
                post_wps = [new_wp]
            else:
                post_wps = [wps[-1]]

        for wp in post_wps:
            cur_x = float(wp.get("x", x_s))
            cur_y = float(wp.get("y", y_s))
            wp["x"] = round(max(0.08, min(0.92, cur_x + dx)), 4)
            wp["y"] = round(max(0.08, min(0.92, cur_y + dy)), 4)

        clause = str(entry.get("reaction_clause") or "").strip()
        if clause and clause not in reaction_clauses:
            reaction_clauses.append(clause)

        applied.append(
            {
                "portal_index": p_idx,
                "target_kind": "sprite",
                "mode": c_mode,
                "strength": round(strength, 4),
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "label": str(entry.get("label") or ""),
                "reaction_clause": clause,
            }
        )

    return applied, reaction_clauses


def _wire_model_chain(
    graph: dict[str, Any],
    *,
    control_vectors: list[dict[str, Any]],
    has_release_refs: bool,
    has_sprite: bool,
    has_cues: bool,
    object_info: dict[str, Any] | None,
    length: int,
    preview_override: bool,
    release_overrides: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Build deterministic model chain: 21(Sol) -> [115.. steers] -> [135 release] -> [165 TAE] -> 9(Guider)."""
    # Remove any template-carried steer/release/preview nodes first so chain is clean.
    for nid in list(graph):
        cls = graph[nid].get("class_type") if isinstance(graph[nid], dict) else ""
        if cls in ("MiniMaxH3Steer", "MiniMaxH3ReleaseReferences", PREVIEW_NODE_CLASS):
            del graph[nid]

    head: list[Any] = ["21", 0]
    steer_ids: list[str] = []

    for idx, vec in enumerate(control_vectors):
        if not isinstance(vec, dict):
            continue
        vec_name = str(vec.get("vector") or vec.get("file") or vec.get("name") or "").strip()
        if not vec_name or vec_name == "none":
            continue
        nid = str(115 + idx)
        steer_inputs: dict[str, Any] = {
            "model": copy.deepcopy(head),
            "vector": vec_name,
            "strength": float(vec.get("strength", 1.0)),
            "mode": str(vec.get("mode") or "add_norm_matched"),
            "target": str(vec.get("target") or "description_only"),
            "start_percent": float(vec.get("start_percent", 0.0)),
            "end_percent": float(vec.get("end_percent", 1.0)),
            "streams": str(vec.get("streams") or "both"),
            "hold_words": str(vec.get("hold_words") or ""),
            "hold_alpha": float(vec.get("hold_alpha", 1.0)),
            "allow_recipe_mismatch": bool(vec.get("allow_recipe_mismatch", False)),
            "edit_pass": str(vec.get("edit_pass") or "edited_only"),
        }
        if vec.get("fields"):
            steer_inputs["fields"] = list(vec["fields"])
        graph[nid] = {"class_type": "MiniMaxH3Steer", "inputs": steer_inputs}
        head = [nid, 0]
        steer_ids.append(nid)

    if has_release_refs or has_sprite or has_cues:
        ov = release_overrides or {}
        rel_inputs: dict[str, Any] = {
            "model": copy.deepcopy(head),
            "end_percent": float(ov.get("end_percent", 1.0)),
            "clean_level": float(ov.get("clean_level", 0.999)),
        }
        if has_cues:
            rel_inputs["audio_end_percent"] = float(ov.get("audio_end_percent", 0.60))
            rel_inputs["audio_clean_level"] = float(ov.get("audio_clean_level", 1.0))
        if has_sprite:
            rel_inputs["video_end_percent"] = float(ov.get("video_end_percent", 0.33))
        graph["135"] = {"class_type": "MiniMaxH3ReleaseReferences", "inputs": rel_inputs}
        head = ["135", 0]

    graph["9"]["inputs"]["model"] = copy.deepcopy(head)

    preview_frames, preview_fps = video_preview_schedule(int(length))
    injected = _inject_kj_preview_override(
        graph,
        object_info=object_info,
        tiny_vae_filename=TAEH3_VIDEO_FILE,
        preview_frames=preview_frames,
        preview_fps=preview_fps,
        enabled=preview_override,
    )
    return injected, steer_ids


def _compile_single_spatial(
    params: dict[str, Any],
    object_info: dict[str, Any] | None = None,
    *,
    scale_override: float | None = None,
) -> dict[str, Any]:
    rules = _h3_rules()
    mode = str(params.get("mode") or "r2v_portal").strip()
    tpl_path, used_shipped = resolve_spatial_template_path(mode)
    graph = _load_json(tpl_path)

    length = int(params.get("length", 90))
    clocks = all_on_both_clocks()
    if length not in clocks:
        raise ValueError(
            f"Spatial Director length={length} must lie on H3 on_both_clocks {list(clocks)}"
        )

    width = int(params.get("width", 1344))
    height = int(params.get("height", 768))
    canvas = str(params.get("canvas") or "explicit")

    # Resolve image sources for R2V / I2V vs T2V.
    source_ref = str(
        params.get("source_ref")
        or params.get("ref_image")
        or params.get("first_frame")
        or params.get("anchor_image")
        or ""
    ).strip()
    last_frame = str(params.get("last_frame") or "").strip()
    sprite_ref = str(
        params.get("sprite_ref")
        or params.get("sprite_cutout")
        or params.get("sprite_image")
        or ""
    ).strip()

    is_r2v = mode in ("r2v_portal", "world_targets")
    if sprite_ref and not is_r2v:
        raise ValueError("Sprite trajectory cutouts require ref2va mode ('r2v_portal' or 'world_targets')")
    if (is_r2v or mode == "i2v_portal") and not source_ref:
        raise ValueError("An anchor image is required before rendering in I2V/R2V mode; upload or generate an anchor scene first.")

    if is_r2v:
        graph.setdefault("15", {"class_type": "LoadImage", "inputs": {}})
        graph["15"]["inputs"]["image"] = source_ref
    elif mode == "i2v_portal":
        graph["15"] = {"class_type": "LoadImage", "inputs": {"image": source_ref}}
        for cond_id in ("5", "131"):
            graph[cond_id]["inputs"]["first_frame"] = ["15", 0]
        if last_frame:
            graph["16"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
            for cond_id in ("5", "131"):
                graph[cond_id]["inputs"]["last_frame"] = ["16", 0]
        else:
            graph.pop("16", None)
            for cond_id in ("5", "131"):
                graph[cond_id]["inputs"].pop("last_frame", None)
    else:  # t2v_portal
        graph.pop("15", None)
        graph.pop("16", None)
        for cond_id in ("5", "131"):
            graph[cond_id]["inputs"].pop("first_frame", None)
            graph[cond_id]["inputs"].pop("last_frame", None)

    tpl_length = int((graph.get("27") or {}).get("inputs", {}).get("length") or 141)
    _set_resolution_node(graph, width=width, height=height, length=length, canvas=canvas)

    # SpatialDirector ("160") event parsing BEFORE prompt synthesis so focus coordinates/cues feed prompts.
    n160 = graph["160"]["inputs"]
    default_dir_mode = "point_targets" if mode == "world_targets" else "portal_path"
    director_mode = str(params.get("director_mode") or (default_dir_mode if not used_shipped else n160.get("mode", default_dir_mode)))
    fallback_events = rules.DEFAULT_DIRECTOR_EVENTS_JSON if (not used_shipped and mode == "r2v_portal") else str(n160.get("events_json") or rules.DEFAULT_DIRECTOR_EVENTS_JSON)
    events_doc, _ = _parse_events_doc(params.get("events_json"), fallback_events)
    if params.get("events_json") is None and length != tpl_length:
        _scale_events_doc(events_doc, tpl_length, length)

    # Merge explicit portal_cues / dpad / halo couplings before validation, prompt synthesis & HUD wiring.
    if params.get("portal_cues"):
        events_doc["portal_cues"] = list(params["portal_cues"])
    has_dpad_inputs = _merge_dpad_inputs(events_doc, params)
    has_sprite = bool(sprite_ref) and is_r2v
    couplings_applied, reaction_clauses = _apply_halo_couplings(
        events_doc,
        params,
        length=length,
        sprite_active=bool(has_sprite or has_dpad_inputs),
    )

    if director_mode == "point_targets" and not events_doc.get("targets") and events_doc.get("portal"):
        # Allow mode switching seamlessly if user hasn't placed point targets yet.
        director_mode = "portal_path"

    events_str = json.dumps(events_doc, indent=2)
    _spatial_director().parse_director_events(events_str, mode=director_mode, length=length)

    # Scene-agnostic <Picture 1> prompt synthesis (never inherits shipped template prompt prose).
    raw_orig = str(params.get("original_prompt") or "").strip()
    same_prompt = bool(params.get("same_prompt", False))
    edit_lines = str(params.get("edit_lines") or "").strip()
    target_prompt = str(params.get("target_prompt") or "").strip()

    orig_prompt, synth_edit, region_summary = synthesize_spatial_prompts(
        mode=mode,
        length=length,
        has_source_ref=bool(source_ref),
        director_mode=director_mode,
        events_doc=events_doc,
        original_prompt=raw_orig,
        target_prompt=target_prompt,
        same_prompt=same_prompt,
        edit_lines=edit_lines,
        has_sprite=has_sprite,
        reaction_clauses=None,
    )
    graph["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": orig_prompt}}
    graph["5"]["inputs"]["prompt"] = ["133", 0]

    if same_prompt:
        graph.pop("130", None)
        graph.pop("134", None)
        graph["131"]["inputs"]["prompt"] = ["133", 0]
        prompt_mode = "same_prompt"
    elif edit_lines:
        _h3_prompt_edit().apply_edits(orig_prompt, edit_lines)
        graph.pop("134", None)
        graph["130"] = {
            "class_type": "MiniMaxH3PromptEdit",
            "inputs": {"original": ["133", 0], "edits": edit_lines},
        }
        graph["131"]["inputs"]["prompt"] = ["130", 0]
        prompt_mode = "edit_lines"
    else:
        graph.pop("130", None)
        graph["134"] = {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": synth_edit},
        }
        graph["131"]["inputs"]["prompt"] = ["134", 0]
        has_ev_prompts = _events_have_per_event_prompts(events_doc)
        prompt_mode = "target_prompt" if (target_prompt or has_ev_prompts) else "synthesized_hold"

    if not same_prompt and reaction_clauses:
        clause_suffix = " ".join(reaction_clauses)
        if "134" in graph:
            cur_val = str(graph["134"]["inputs"]["value"]).strip()
            if clause_suffix not in cur_val:
                graph["134"]["inputs"]["value"] = f"{cur_val} {clause_suffix}".strip()
        elif "130" in graph:
            edits_val = str(graph["130"]["inputs"].get("edits") or "")
            try:
                base_edited = _h3_prompt_edit().apply_edits(orig_prompt, edits_val)
            except Exception:
                base_edited = orig_prompt
            graph.pop("130", None)
            graph["134"] = {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {
                    "value": f"{base_edited.strip()} {clause_suffix}".strip() if clause_suffix not in base_edited else base_edited,
                },
            }
            graph["131"]["inputs"]["prompt"] = ["134", 0]

    feather_val = _feather_value(params, width, height, float(n160.get("feather", 0.08)))
    want_invert = bool(params.get("invert_mask", n160.get("invert_mask", False)))
    include_mask_ref = str(params.get("include_mask_ref") or "").strip()
    exclude_mask_ref = str(params.get("exclude_mask_ref") or "").strip()
    sprite_mask_ref = str(params.get("sprite_mask_ref") or "").strip()
    native_combined = has_spatial_combined_support(object_info)

    for stale_id in ("166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179"):
        graph.pop(stale_id, None)
    n160.pop("include_mask", None)
    n160.pop("exclude_mask", None)

    n160["width"] = ["27", 0]
    n160["height"] = ["27", 1]
    n160["length"] = ["27", 2]
    n160["feather"] = feather_val
    n160["events_json"] = events_str

    n161 = graph["161"]["inputs"]
    if params.get("hud_layout"):
        n161["hud_layout"] = str(params["hud_layout"])
    n161["title"] = (
        str(params["hud_title"])
        if params.get("hud_title") is not None
        else f"{mode.upper().replace('_', ' ')} · WORLD SANDBOX"
    )
    if params.get("hud_subtitle") is not None:
        n161["subtitle"] = str(params["hud_subtitle"])
    else:
        sub_hint = target_prompt or region_summary or "Anchored world dynamics"
        n161["subtitle"] = f"{sub_hint[:84]}..." if len(sub_hint) > 84 else sub_hint
    if params.get("ring_opacity") is not None:
        n161["ring_opacity"] = float(params["ring_opacity"])

    # Conditional Sprite Trajectory ("51").
    if has_sprite:
        ref_img = str(graph.get("15", {}).get("inputs", {}).get("image") or "")
        if sprite_ref == ref_img and "15" in graph:
            sprite_loader_ref: list[Any] = ["15", 0]
            graph.pop("52", None)
        else:
            graph["52"] = {"class_type": "LoadImage", "inputs": {"image": sprite_ref}}
            sprite_loader_ref = ["52", 0]
        n51_inputs: dict[str, Any] = {
            "image": sprite_loader_ref,
            "length": ["27", 2],
            "keyframes": ["160", 2],
            "canvas_width": int(params.get("sprite_canvas_width", 384)),
            "canvas_height": int(params.get("sprite_canvas_height", 224)),
            "sprite_scale": float(params.get("sprite_scale", 0.31)),
            "use_vae": True,
            "references": ["50", 0],
        }
        if sprite_mask_ref:
            graph["53"] = {"class_type": "LoadImage", "inputs": {"image": sprite_mask_ref}}
            graph["54"] = {"class_type": "ImageToMask", "inputs": {"image": ["53", 0], "channel": "red"}}
            n51_inputs["mask"] = ["54", 0]
        else:
            graph.pop("53", None)
            graph.pop("54", None)
        graph["51"] = {
            "class_type": "MiniMaxH3AppendSpriteTrajectory",
            "inputs": n51_inputs,
        }
        graph["132"]["inputs"]["references"] = ["51", 0]
    else:
        graph.pop("51", None)
        graph.pop("52", None)
        graph.pop("53", None)
        graph.pop("54", None)
        if is_r2v and "132" in graph:
            graph["132"]["inputs"]["references"] = ["50", 0]

    # Seed & noise.
    seed = _resolve_seed(params, graph)
    graph["6"]["inputs"]["noise_seed"] = seed

    # Native combined/aux masks OR zero-restart MaskComposite fallback when live server has 2-mode schema.
    fallback_cues_ref: Any = ["160", 1]
    if native_combined:
        n160["mode"] = director_mode
        n160["invert_mask"] = want_invert
        if include_mask_ref:
            graph["170"] = {"class_type": "LoadImage", "inputs": {"image": include_mask_ref}}
            graph["171"] = {"class_type": "ImageToMask", "inputs": {"image": ["170", 0], "channel": "red"}}
            n160["include_mask"] = ["171", 0]
        if exclude_mask_ref:
            graph["172"] = {"class_type": "LoadImage", "inputs": {"image": exclude_mask_ref}}
            graph["173"] = {"class_type": "ImageToMask", "inputs": {"image": ["172", 0], "channel": "red"}}
            n160["exclude_mask"] = ["173", 0]
        final_mask_ref: list[Any] = ["160", 0]
    else:
        has_portals = bool(events_doc.get("portal"))
        has_targets = bool(events_doc.get("targets"))
        need_mask_pipeline = (director_mode == "combined") or bool(include_mask_ref) or bool(exclude_mask_ref)
        if director_mode == "combined":
            if has_portals and has_targets:
                n160["mode"] = "portal_path"
                n160["invert_mask"] = False
                graph["166"] = {
                    "class_type": "MiniMaxH3SpatialDirector",
                    "inputs": {
                        "mode": "point_targets",
                        "width": ["27", 0],
                        "height": ["27", 1],
                        "length": ["27", 2],
                        "feather": feather_val,
                        "invert_mask": False,
                        "events_json": events_str,
                    },
                }
                graph["167"] = {
                    "class_type": "MaskComposite",
                    "inputs": {
                        "destination": ["160", 0],
                        "source": ["166", 0],
                        "x": 0,
                        "y": 0,
                        "operation": "add",
                    },
                }
                final_mask_ref = ["167", 0]
                deduped_cues = str(
                    _spatial_director().MiniMaxH3SpatialDirector.execute(
                        mode="combined",
                        width=64,
                        height=64,
                        length=length,
                        feather=feather_val,
                        invert_mask=False,
                        events_json=events_str,
                    )[1]
                )
                graph["168"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": deduped_cues}}
                fallback_cues_ref = ["168", 0]
            elif has_targets:
                n160["mode"] = "point_targets"
                n160["invert_mask"] = False
                final_mask_ref = ["160", 0]
            else:
                n160["mode"] = "portal_path"
                n160["invert_mask"] = False
                final_mask_ref = ["160", 0]
        else:
            n160["mode"] = director_mode
            n160["invert_mask"] = False if need_mask_pipeline else want_invert
            final_mask_ref = ["160", 0]

        if include_mask_ref:
            graph["170"] = {"class_type": "LoadImage", "inputs": {"image": include_mask_ref}}
            graph["171"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["170", 0],
                    "upscale_method": "bilinear",
                    "width": ["27", 0],
                    "height": ["27", 1],
                    "crop": "disabled",
                },
            }
            graph["172"] = {"class_type": "ImageToMask", "inputs": {"image": ["171", 0], "channel": "red"}}
            graph["173"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": final_mask_ref,
                    "source": ["172", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "add",
                },
            }
            final_mask_ref = ["173", 0]
        if need_mask_pipeline and want_invert:
            graph["174"] = {"class_type": "InvertMask", "inputs": {"mask": final_mask_ref}}
            final_mask_ref = ["174", 0]
        if exclude_mask_ref:
            graph["175"] = {"class_type": "LoadImage", "inputs": {"image": exclude_mask_ref}}
            graph["176"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["175", 0],
                    "upscale_method": "bilinear",
                    "width": ["27", 0],
                    "height": ["27", 1],
                    "crop": "disabled",
                },
            }
            graph["177"] = {"class_type": "ImageToMask", "inputs": {"image": ["176", 0], "channel": "red"}}
            graph["178"] = {"class_type": "InvertMask", "inputs": {"mask": ["177", 0]}}
            graph["179"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": final_mask_ref,
                    "source": ["178", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "multiply",
                },
            }
            final_mask_ref = ["179", 0]

    # Conditional Frame-0 GuideLatent pin ("169") for R2V so movable sprites do not unpin Frame 0,
    # followed by SoundCues ("70") -> Preflight ("26") -> EditGuider ("9").
    has_cues = _events_have_sound_cues(events_doc, params, director_mode=director_mode)
    latent_source = "132" if is_r2v else "5"
    lock_first_frame = bool(params.get("lock_first_frame", True))
    pin_guide_f0 = bool(
        is_r2v and source_ref and lock_first_frame and has_guide_latent_support(object_info)
    )
    if pin_guide_f0:
        graph["169"] = {
            "class_type": "MiniMaxH3GuideLatent",
            "inputs": {
                "positive": ["5", 0],
                "latent": [latent_source, 1],
                "vae": ["3", 0],
                "guide": ["15", 0],
                "frame_idx": 0,
                "single_frame": False,
                "edited_positive": ["131", 0],
            },
        }
        cond_pos_ref: list[Any] = ["169", 0]
        cond_lat_ref: list[Any] = ["169", 1]
        cond_edit_ref: list[Any] = ["169", 2]
    else:
        cond_pos_ref = ["5", 0]
        cond_lat_ref = [latent_source, 1]
        cond_edit_ref = ["131", 0]

    if has_cues:
        graph["70"] = {
            "class_type": "MiniMaxH3SoundCues",
            "inputs": {
                "positive": cond_pos_ref,
                "latent": cond_lat_ref,
                "audio_vae": ["4", 0],
                "cues": fallback_cues_ref,
                "snap_to_3f": True,
                "floor_dbfs": float(params.get("floor_dbfs", -28.0)),
                "seed": int(params.get("audio_seed", seed)),
                "edited_positive": cond_edit_ref,
            },
        }
        graph["26"]["inputs"] = {
            "conditioning": ["70", 0],
            "samples": ["70", 1],
            "edited_conditioning": ["70", 2],
        }
        edited_ref = ["70", 2]
    else:
        graph.pop("70", None)
        graph["26"]["inputs"] = {
            "conditioning": cond_pos_ref,
            "samples": cond_lat_ref,
            "edited_conditioning": cond_edit_ref,
        }
        edited_ref = cond_edit_ref

    n9 = graph["9"]["inputs"]
    n9["original"] = ["26", 0]
    n9["edited"] = edited_ref
    n9["mask"] = final_mask_ref
    if scale_override is not None:
        n9["scale"] = float(scale_override)
    elif params.get("scale") is not None:
        n9["scale"] = float(params["scale"])
    if params.get("audio_scale") is not None:
        n9["audio_scale"] = float(params["audio_scale"])
    if params.get("streams") is not None and str(params.get("streams")).strip():
        n9["streams"] = str(params["streams"]).strip()
    elif director_mode == "point_targets":
        n9["streams"] = "video_only"
    elif not used_shipped and mode == "r2v_portal":
        n9["streams"] = "independent"
    if params.get("first_step") is not None:
        n9["first_step"] = int(params["first_step"])
    if params.get("last_step") is not None:
        n9["last_step"] = int(params["last_step"])
    if params.get("align_positions") is not None:
        n9["align_positions"] = bool(params["align_positions"])

    # Control vectors + ReleaseReferences + ModelPreviewOverrideKJ.
    raw_vecs = params.get("control_vectors")
    if raw_vecs is None and params.get("control_vector"):
        cv = params["control_vector"]
        raw_vecs = [cv if isinstance(cv, dict) else {"vector": str(cv)}]
    vectors = list(raw_vecs) if isinstance(raw_vecs, list) else []

    preview_override = bool(params.get("preview_override", True))
    injected, steer_ids = _wire_model_chain(
        graph,
        control_vectors=vectors,
        has_release_refs=is_r2v,
        has_sprite=has_sprite,
        has_cues=has_cues,
        object_info=object_info,
        length=length,
        preview_override=preview_override,
        release_overrides=params.get("release_references"),
    )

    # Optional in-graph 2nd-pass Audio Refine (`MiniMaxH3FrozenVideoCache` + `MiniMaxH3AudioRefineMask`).
    for stale_ref_id in ("80", "81", "82", "83", "84", "85", "86", "87", "88"):
        graph.pop(stale_ref_id, None)
    audio_refine = bool(params.get("audio_refine", False))
    if audio_refine:
        audio_refine_denoise = max(0.05, min(1.0, float(params.get("audio_refine_denoise", 0.5))))
        audio_refine_steps = max(1, min(30, int(params.get("audio_refine_steps", 6))))
        rsrc: list[Any] = ["1", 0]
        graph["80"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": rsrc, "shift_video": 12.0, "shift_audio": 3.0},
        }
        rsrc = ["80", 0]
        if "58" in graph and isinstance(graph["58"].get("inputs"), dict):
            graph["81"] = {
                "class_type": "ModelAttentionBackend",
                "inputs": {"model": rsrc, "attention": graph["58"]["inputs"].get("attention", "comfy kitchen attention")},
            }
            rsrc = ["81", 0]
        if "21" in graph and isinstance(graph["21"], dict) and graph["21"].get("class_type") == "MiniMaxH3Sol":
            sol_inputs = {k: copy.deepcopy(v) for k, v in (graph["21"].get("inputs") or {}).items() if k != "model"}
            sol_inputs["model"] = rsrc
            graph["82"] = {"class_type": "MiniMaxH3Sol", "inputs": sol_inputs}
            rsrc = ["82", 0]
        graph["88"] = {
            "class_type": "MiniMaxH3FrozenVideoCache",
            "inputs": {
                "model": rsrc,
                "precision": "int4",
                "refresh": False,
                "refresh_every": 2,
                "verify": False,
                "halo": 0,
            },
        }
        rsrc = ["88", 0]
        graph["83"] = {
            "class_type": "MiniMaxH3AudioRefineMask",
            "inputs": {"latent": ["10", 0], "video_mask": 0.0, "audio_mask": 1.0},
        }
        graph["84"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": rsrc,
                "scheduler": "simple",
                "steps": audio_refine_steps,
                "denoise": audio_refine_denoise,
            },
        }
        graph["85"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        graph["86"] = {"class_type": "BasicGuider", "inputs": {"model": rsrc, "conditioning": edited_ref}}
        graph["87"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["86", 0],
                "sampler": ["85", 0],
                "sigmas": ["84", 0],
                "latent_image": ["83", 0],
            },
        }
        graph["11"]["inputs"]["samples"] = ["87", 0]
        graph["12"]["inputs"]["samples"] = ["87", 0]
    else:
        graph["11"]["inputs"]["samples"] = ["10", 0]
        graph["12"]["inputs"]["samples"] = ["10", 0]

    # Dual VHS_VideoCombine outputs ("13" HUD + "162" Clean), strictly temp-only (`save_output: False`).
    base_combine = copy.deepcopy(graph[HUD_COMBINE_NODE_ID])
    base_inputs = dict(base_combine.get("inputs") or {})
    base_inputs.update(
        {
            "images": ["161", 0],
            "audio": ["12", 0],
            "frame_rate": 24.0,
            "filename_prefix": f"h3_spatial_director/{mode}_hud",
            "save_output": False,
        }
    )
    graph[HUD_COMBINE_NODE_ID] = {"class_type": "VHS_VideoCombine", "inputs": base_inputs}

    clean_inputs = copy.deepcopy(base_inputs)
    clean_inputs["images"] = ["11", 0]
    clean_inputs["filename_prefix"] = f"h3_spatial_director/{mode}_clean"
    clean_inputs["save_output"] = False
    graph[CLEAN_DECODE_NODE_ID] = {"class_type": "VHS_VideoCombine", "inputs": clean_inputs}

    return {
        "mode": mode,
        "director_mode": director_mode,
        "seed": seed,
        "length": length,
        "width": width,
        "height": height,
        "used_shipped_template": used_shipped,
        "preview_override_injected": injected,
        "nodes_summary": {
            "has_sprite": has_sprite,
            "has_sprite_mask": bool(has_sprite and sprite_mask_ref),
            "guide_latent_pinned": pin_guide_f0,
            "has_include_mask": bool(include_mask_ref),
            "has_exclude_mask": bool(exclude_mask_ref),
            "spatial_combined_native": native_combined,
            "has_sound_cues": has_cues,
            "has_audio_refine": audio_refine,
            "has_preview_override": injected,
            "has_release_refs": "135" in graph,
            "control_vector_nodes": steer_ids,
            "prompt_mode": prompt_mode,
            "feather": round(float(n160["feather"]), 4),
            "halo_radius": round(max(0.0, min(0.35, float(params["halo_radius"]))) if params.get("halo_radius") is not None else 0.0, 4),
            "streams": str(n9.get("streams") or "independent"),
            "audio_gated_by_mask": str(n9.get("streams") or "independent") == "both",
            "couplings_applied": couplings_applied,
            "hud_save_output": bool(graph[HUD_COMBINE_NODE_ID]["inputs"].get("save_output")),
            "clean_save_output": bool(graph[CLEAN_DECODE_NODE_ID]["inputs"].get("save_output")),
        },
        "graph": graph,
    }


def compile_spatial_take(
    params: dict[str, Any] | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile primary spatial take graph and optional companion baseline (``scale=0.0``) graph."""
    cfg = dict(params or {})
    primary = _compile_single_spatial(cfg, object_info=object_info, scale_override=None)
    queue_baseline = bool(cfg.get("queue_baseline", False))
    baseline_graph = None
    if queue_baseline:
        base_cfg = copy.deepcopy(cfg)
        base_cfg["seed"] = primary["seed"]
        baseline_graph = _compile_single_spatial(base_cfg, object_info=object_info, scale_override=0.0)["graph"]
    return {
        "mode": primary["mode"],
        "director_mode": primary["director_mode"],
        "seed": primary["seed"],
        "length": primary["length"],
        "width": primary["width"],
        "height": primary["height"],
        "queue_baseline": queue_baseline,
        "preview_override_injected": primary["preview_override_injected"],
        "nodes_summary": primary["nodes_summary"],
        "graph": primary["graph"],
        "baseline_graph": baseline_graph,
    }


def _scale_frame(val: Any, src_len: int, dst_len: int) -> int:
    if isinstance(val, float) and 0.0 <= val <= 1.0 and not val.is_integer():
        raw = val * max(1, dst_len - 1)
    else:
        src_span = max(1, int(src_len) - 1)
        dst_span = max(1, int(dst_len) - 1)
        raw = float(val) * (float(dst_span) / float(src_span))
    return max(0, min(dst_len - 1, int(round(raw))))


def _snap_cue_frame_3f(frame: int, length: int) -> int:
    """Snap cue frame to a 3-frame (125 ms / 5 audio token) multiple strictly inside ``[0, length - 1]``."""
    max_mult = max(0, ((int(length) - 1) // 3) * 3)
    snapped = int(round(float(frame) / 3.0)) * 3
    return max(0, min(max_mult, snapped))


def snap_to_on_both_clocks(target_length: int) -> int:
    """Snap ``target_length`` to the nearest length on ``h3_rules.on_both_clocks``."""
    clocks = list(all_on_both_clocks())
    return min(clocks, key=lambda c: (abs(c - int(target_length)), -c))


def next_on_both_clocks(current_length: int) -> int:
    """Return next longer grid step on ``h3_rules.on_both_clocks``, clamping at max (345)."""
    clocks = sorted(all_on_both_clocks())
    for c in clocks:
        if c > int(current_length):
            return c
    return clocks[-1]


def compute_preflight_summary(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute offline PackedLayout token estimate and ``int32`` QKV limit status."""
    cfg = dict(params or {})
    clocks = all_on_both_clocks()
    length = int(cfg.get("length", 141))
    width = int(cfg.get("width", 1344))
    height = int(cfg.get("height", 768))
    mode = str(cfg.get("mode") or "r2v_portal")

    if length == 1:
        latent_steps = 1
        audio_tokens = 0
    else:
        latent_steps = 5 * ((length - 5) // 17) + 2
        audio_tokens = (length * 5) // 3

    tokens_per_frame = max(1, (width // 32) * (height // 32))
    video_tokens = latent_steps * tokens_per_frame

    ref_tokens = 0
    has_source_ref = bool(
        cfg.get("source_ref") or cfg.get("ref_image") or cfg.get("first_frame") or cfg.get("anchor_image")
    )
    if mode in ("r2v_portal", "world_targets"):
        ref_tokens += tokens_per_frame
        if bool(cfg.get("lock_first_frame", True)) and has_source_ref:
            ref_tokens += tokens_per_frame
        if cfg.get("sprite_ref") or cfg.get("sprite_cutout") or cfg.get("sprite_image"):
            sw = int(cfg.get("sprite_canvas_width", 384))
            sh = int(cfg.get("sprite_canvas_height", 224))
            ref_tokens += max(1, (sw // 32) * (sh // 32))
    elif mode == "i2v_portal":
        ref_tokens += tokens_per_frame

    per_event_text = ""
    raw_ev = cfg.get("events_json")
    if raw_ev is not None:
        try:
            ev_doc, _ = _parse_events_doc(raw_ev, "{}")
            per_event_text = " ".join(
                str(item.get("prompt") or "")
                for k in ("targets", "portal")
                for item in (ev_doc.get(k) or ())
                if isinstance(item, dict) and item.get("prompt")
            )
        except Exception:
            per_event_text = ""
    prompt_text = " ".join(
        [*(str(cfg.get(k) or "") for k in ("original_prompt", "target_prompt", "edit_lines")), per_event_text]
    ).strip()
    prompt_tokens_est = max(64, len(prompt_text) // 4 + 64)
    total_tokens = video_tokens + audio_tokens + ref_tokens + prompt_tokens_est
    exceeds_int32 = total_tokens >= INT32_TOKEN_LIMIT

    min_i2v_len = int(_h3_rules().min_legal_length())
    i2v_too_short = mode == "i2v_portal" and length != 1 and length < min_i2v_len

    warnings: list[str] = []
    if length != 1 and length not in clocks:
        warnings.append(f"length={length} is off on_both_clocks {list(clocks)}")
    if i2v_too_short:
        warnings.append(
            f"i2v_portal requires >= {min_i2v_len}f (5.0s, e.g. 141f); {length}f is supported on r2v_portal / world_targets"
        )
    if width % 32 or height % 32:
        warnings.append(f"resolution {width}x{height} must be divisible by 32")
    if exceeds_int32:
        warnings.append(
            f"total_tokens={total_tokens} exceeds int32 QKV stride threshold ({INT32_TOKEN_LIMIT})"
        )

    return {
        "ok": True,
        "length": length,
        "width": width,
        "height": height,
        "on_both_clocks": length in clocks,
        "latent_steps": latent_steps,
        "tokens_per_frame": tokens_per_frame,
        "video_tokens": video_tokens,
        "audio_tokens": audio_tokens,
        "ref_tokens": ref_tokens,
        "prompt_tokens_est": prompt_tokens_est,
        "total_tokens": total_tokens,
        "int32_limit": INT32_TOKEN_LIMIT,
        "exceeds_int32": exceeds_int32,
        "can_render": (
            (not exceeds_int32)
            and (width % 32 == 0)
            and (height % 32 == 0)
            and (length == 1 or length in clocks)
            and (not i2v_too_short)
        ),
        "warnings": warnings,
        "segments": {
            "video": video_tokens,
            "audio": audio_tokens,
            "references": ref_tokens,
            "prompt_est": prompt_tokens_est,
        },
    }


def _scale_events_doc(doc: dict[str, Any], src_len: int, dst_len: int) -> dict[str, Any]:
    """Scale spatial portal, target, dpad, and 3f cue timings in-place from ``src_len`` to ``dst_len``."""
    for p in doc.get("portal") or ():
        if isinstance(p, dict) and "frame" in p:
            p["frame"] = _scale_frame(p["frame"], src_len, dst_len)

    for t in doc.get("targets") or ():
        if not isinstance(t, dict):
            continue
        s = _scale_frame(t.get("start", 0), src_len, dst_len)
        e = _scale_frame(t.get("end", src_len - 1), src_len, dst_len)
        if str(t.get("cue") or "none").strip().lower() not in ("", "none"):
            s = _snap_cue_frame_3f(s, dst_len)
        if e <= s:
            e = min(dst_len - 1, s + 3)
            if e <= s and s > 0:
                s = max(0, e - 3)
        t["start"] = s
        t["end"] = e

    for c in doc.get("portal_cues") or ():
        if isinstance(c, dict) and "frame" in c:
            c["frame"] = _snap_cue_frame_3f(_scale_frame(c["frame"], src_len, dst_len), dst_len)

    dpad = doc.get("dpad")
    if isinstance(dpad, dict):
        for wp in dpad.get("waypoints") or ():
            if isinstance(wp, dict) and "frame" in wp:
                wp["frame"] = _scale_frame(wp["frame"], src_len, dst_len)
        for hp in dpad.get("hops") or ():
            if not isinstance(hp, dict):
                continue
            hs = _scale_frame(hp.get("start", 0), src_len, dst_len)
            he = _scale_frame(hp.get("end", hs + 6), src_len, dst_len)
            if he <= hs:
                he = min(dst_len - 1, hs + 3)
            hp["start"] = hs
            hp["end"] = he
    return doc


def escalate_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Scale spatial keyframes, sprite waypoints/hops, and 3f sound cues onto ``on_both_clocks``."""
    cfg = dict(params or {})
    src_len = int(cfg.get("length", 90))
    if src_len < 5:
        src_len = 90
    raw_target = cfg.get("target_length")
    dst_len = snap_to_on_both_clocks(int(raw_target)) if raw_target is not None else next_on_both_clocks(src_len)

    doc, _ = _parse_events_doc(cfg.get("events_json"), _h3_rules().DEFAULT_DIRECTOR_EVENTS_JSON)
    _scale_events_doc(doc, src_len, dst_len)

    extra_portal_cues = copy.deepcopy(cfg.get("portal_cues") or doc.get("portal_cues") or [])
    for c in extra_portal_cues:
        if isinstance(c, dict) and "frame" in c:
            c["frame"] = _snap_cue_frame_3f(_scale_frame(c["frame"], src_len, dst_len), dst_len)

    extra_dpad = copy.deepcopy(cfg.get("dpad_keyframes") or doc.get("dpad") or {})
    if isinstance(extra_dpad, dict):
        for wp in extra_dpad.get("waypoints") or ():
            if isinstance(wp, dict) and "frame" in wp:
                wp["frame"] = _scale_frame(wp["frame"], src_len, dst_len)
        for hp in extra_dpad.get("hops") or ():
            if not isinstance(hp, dict):
                continue
            hs = _scale_frame(hp.get("start", 0), src_len, dst_len)
            he = _scale_frame(hp.get("end", hs + 6), src_len, dst_len)
            if he <= hs:
                he = min(dst_len - 1, hs + 3)
            hp["start"] = hs
            hp["end"] = he

    preflight_cfg = dict(cfg)
    preflight_cfg["length"] = dst_len
    return {
        "ok": True,
        "source_length": src_len,
        "target_length": dst_len,
        "events_json": doc,
        "portal_cues": extra_portal_cues,
        "dpad_keyframes": extra_dpad,
        "preflight": compute_preflight_summary(preflight_cfg),
    }


def has_audio_refine_support(object_info: dict[str, Any] | None) -> bool:
    """Return True when ``MiniMaxH3AudioRefineMask`` + ``MiniMaxH3FrozenVideoCache`` are available."""
    if not isinstance(object_info, dict):
        return True
    has_mask = isinstance(object_info.get("MiniMaxH3AudioRefineMask"), dict)
    has_cache = isinstance(object_info.get("MiniMaxH3FrozenVideoCache"), dict)
    if has_mask and has_cache:
        return True
    if has_mask or has_cache:
        return False
    sd = object_info.get("MiniMaxH3SpatialDirector")
    if isinstance(sd, dict):
        inp = sd.get("input") or {}
        if inp.get("required") or inp.get("optional"):
            return False
    return True


def compile_audio_redub(
    params: dict[str, Any] | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a 2nd-pass video-frozen audio re-dub workflow (``FrozenVideoCache`` + ``AudioRefineMask``)."""
    cfg = dict(params or {})
    video_ref = str(cfg.get("video_ref") or cfg.get("source_video") or "").strip()
    if not video_ref:
        raise ValueError("video_ref is required for audio re-dub")

    tpl_path = _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json"
    if not tpl_path.is_file():
        raise FileNotFoundError(f"Audio re-dub base template missing: {tpl_path}")
    graph = _load_json(tpl_path)

    base_sol = copy.deepcopy(graph.get("21") or {})
    base_combine = copy.deepcopy(graph.get(HUD_COMBINE_NODE_ID) or {})

    for stale in ("7", "9", "10", "11", "19", "21", "58", "60", "130", "131", "134", "160", "161"):
        graph.pop(stale, None)

    length = snap_to_on_both_clocks(int(cfg.get("length", 90)))
    width = int(cfg.get("width", 1344))
    height = int(cfg.get("height", 768))
    _set_resolution_node(graph, width=width, height=height, length=length, canvas="explicit")

    seed = _resolve_seed(cfg, graph)
    graph["6"]["inputs"]["noise_seed"] = seed

    raw_prompt = str(
        cfg.get("audio_prompt")
        or cfg.get("target_prompt")
        or cfg.get("prompt")
        or graph.get("133", {}).get("inputs", {}).get("value")
        or ""
    ).strip()
    if not raw_prompt:
        raw_prompt = "Natural cinematic foley and ambience synced to on-screen motion."
    graph["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": raw_prompt}}
    graph["5"]["inputs"]["prompt"] = ["133", 0]

    graph["90"] = {"class_type": "LoadVideo", "inputs": {"file": video_ref}}
    graph["91"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["90", 0]}}
    graph["92"] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["91", 0],
            "upscale_method": "bilinear",
            "width": ["27", 0],
            "height": ["27", 1],
            "crop": "disabled",
        },
    }
    graph["93"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["92", 0], "vae": ["3", 0]}}

    denoise_raw = cfg.get("denoise")
    redub_mode = str(cfg.get("redub_mode") or "").strip().lower()
    if denoise_raw is not None and str(denoise_raw).strip() != "":
        denoise = max(0.05, min(1.0, float(denoise_raw)))
    elif redub_mode == "polish":
        denoise = 0.5
    else:
        denoise = 1.0
    resolved_redub_mode = "fresh" if denoise >= 0.999 else "polish"

    if cfg.get("use_distill") is not None:
        use_distill = bool(cfg["use_distill"])
    else:
        use_distill = bool(denoise >= 0.999)

    for aux_nid in ("94a", "94b"):
        graph.pop(aux_nid, None)
    if denoise >= 0.999:
        graph["94"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["5", 1]}}
        graph["95"] = {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["93", 0], "audio_latent": ["94", 1]},
        }
    else:
        graph["94a"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["5", 1]}}
        graph["94b"] = {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["93", 0], "audio_latent": ["94a", 1]},
        }
        graph["94"] = {
            "class_type": "MiniMaxH3EncodeTrack",
            "inputs": {"audio_vae": ["4", 0], "audio": ["91", 1], "level": "clip_guard"},
        }
        graph["95"] = {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["94b", 0], "audio_latent": ["94", 0]},
        }

    cues_str = str(cfg.get("cues_text") or "").strip()
    if not cues_str and (cfg.get("events_json") is not None or cfg.get("portal_cues")):
        minimal_fallback = '{"portal":[{"frame":0,"x":0.5,"y":0.5,"rx":0.2,"ry":0.2,"shape":"circle","label":"PORTAL"}]}'
        events_doc, _ = _parse_events_doc(cfg.get("events_json"), minimal_fallback)
        if cfg.get("events_json") is not None:
            raw_frames: list[int] = [0]
            for p in events_doc.get("portal") or ():
                if isinstance(p, dict) and p.get("frame") is not None:
                    raw_frames.append(int(float(p["frame"])))
            for t in events_doc.get("targets") or ():
                if isinstance(t, dict):
                    for fk in ("start", "end"):
                        if t.get(fk) is not None:
                            raw_frames.append(int(float(t[fk])))
            for c in events_doc.get("portal_cues") or ():
                if isinstance(c, dict) and c.get("frame") is not None:
                    raw_frames.append(int(float(c["frame"])))
            dp = events_doc.get("dpad")
            if isinstance(dp, dict):
                for wp in dp.get("waypoints") or ():
                    if isinstance(wp, dict) and wp.get("frame") is not None:
                        raw_frames.append(int(float(wp["frame"])))
                for hp in dp.get("hops") or ():
                    if isinstance(hp, dict):
                        for fk in ("start", "end"):
                            if hp.get(fk) is not None:
                                raw_frames.append(int(float(hp[fk])))
            max_doc_frame = max(raw_frames)
            src_len = int(cfg["source_length"]) if cfg.get("source_length") else (max_doc_frame + 1 if max_doc_frame >= length else length)
            if src_len != length and src_len > 1:
                _scale_events_doc(events_doc, src_len, length)
        if cfg.get("portal_cues"):
            clamped_cues = []
            for item in cfg["portal_cues"]:
                if not isinstance(item, dict):
                    continue
                c_copy = dict(item)
                f_val = max(0, min(length - 1, int(float(c_copy.get("frame", 0)))))
                c_copy["frame"] = _snap_cue_frame_3f(f_val, length)
                clamped_cues.append(c_copy)
            events_doc["portal_cues"] = clamped_cues
        if _events_have_sound_cues(events_doc, cfg, director_mode="combined"):
            if not events_doc.get("portal") and not events_doc.get("targets"):
                events_doc["portal"] = [
                    {"frame": 0, "x": 0.5, "y": 0.5, "rx": 0.2, "ry": 0.2, "shape": "circle", "label": "PORTAL"}
                ]
            cues_str = str(
                _spatial_director().MiniMaxH3SpatialDirector.execute(
                    mode="combined",
                    width=64,
                    height=64,
                    length=length,
                    feather=0.08,
                    invert_mask=False,
                    events_json=json.dumps(events_doc),
                )[1]
            ).strip()

    has_cues = bool(cues_str)
    if has_cues:
        _sound_cues().parse_sound_cues(cues_str, length, snap_to_3f=True)
        graph["70"] = {
            "class_type": "MiniMaxH3SoundCues",
            "inputs": {
                "positive": ["5", 0],
                "latent": ["95", 0],
                "audio_vae": ["4", 0],
                "cues": cues_str,
                "snap_to_3f": True,
                "floor_dbfs": float(cfg.get("floor_dbfs", -28.0)),
                "seed": int(cfg.get("audio_seed", seed)),
            },
        }
        cond_ref: list[Any] = ["70", 0]
        av_latent_ref: list[Any] = ["70", 1]
    else:
        graph.pop("70", None)
        cond_ref = ["5", 0]
        av_latent_ref = ["95", 0]

    graph["26"] = {
        "class_type": "MiniMaxH3Preflight",
        "inputs": {"conditioning": cond_ref, "samples": av_latent_ref},
    }
    graph["83"] = {
        "class_type": "MiniMaxH3AudioRefineMask",
        "inputs": {"latent": ["26", 1], "video_mask": 0.0, "audio_mask": 1.0},
    }

    if not use_distill:
        graph.pop("18", None)
        rsrc: list[Any] = ["1", 0]
    else:
        rsrc = ["18", 0]

    graph["80"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": rsrc, "shift_video": 12.0, "shift_audio": 3.0}}
    graph["81"] = {
        "class_type": "ModelAttentionBackend",
        "inputs": {"model": ["80", 0], "attention": "comfy kitchen attention"},
    }
    sol_inputs = {k: copy.deepcopy(v) for k, v in (base_sol.get("inputs") or {}).items() if k != "model"}
    sol_inputs.setdefault("tau", 1.0)
    sol_inputs.setdefault("start_percent", 0.2)
    sol_inputs.setdefault("end_percent", 1.0)
    sol_inputs.setdefault("min_tokens", 12288)
    sol_inputs.setdefault("sink_conditioning", "exact_kv_and_all_rows")
    sol_inputs.setdefault("verbose", True)
    sol_inputs.setdefault("dense_blocks", "49")
    sol_inputs.setdefault("quantizer", "rotated")
    sol_inputs.setdefault("token_routing", "off")
    sol_inputs["model"] = ["81", 0]
    graph["82"] = {"class_type": "MiniMaxH3Sol", "inputs": sol_inputs}
    graph["88"] = {
        "class_type": "MiniMaxH3FrozenVideoCache",
        "inputs": {"model": ["82", 0], "precision": "int4", "refresh": False, "refresh_every": 2, "verify": False, "halo": 0},
    }
    steps = max(1, min(30, int(cfg.get("steps", 6))))
    custom_sigmas = str(cfg.get("sigmas") or "").strip()
    if use_distill and denoise >= 0.999 and (steps == 6 or custom_sigmas):
        graph.pop("84", None)
        graph["60"] = {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": custom_sigmas or "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
            },
        }
        sigmas_ref: list[Any] = ["60", 0]
        schedule_kind = "ManualSigmas"
    else:
        graph.pop("60", None)
        graph["84"] = {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["88", 0], "scheduler": "simple", "steps": steps, "denoise": denoise},
        }
        sigmas_ref = ["84", 0]
        schedule_kind = "BasicScheduler"
    graph["85"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm" if use_distill else "euler"}}
    graph["86"] = {"class_type": "BasicGuider", "inputs": {"model": ["88", 0], "conditioning": ["26", 0]}}
    graph["87"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["86", 0],
            "sampler": ["85", 0],
            "sigmas": sigmas_ref,
            "latent_image": ["83", 0],
        },
    }

    graph["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["87", 0], "vae": ["4", 0]}}

    base_inputs = dict(base_combine.get("inputs") or {})
    base_inputs.update(
        {
            "images": ["92", 0],
            "audio": ["12", 0],
            "frame_rate": 24.0,
            "filename_prefix": "h3_spatial_director/audio_redub_hud",
            "save_output": False,
        }
    )
    graph[HUD_COMBINE_NODE_ID] = {"class_type": "VHS_VideoCombine", "inputs": base_inputs}

    clean_inputs = copy.deepcopy(base_inputs)
    clean_inputs["images"] = ["92", 0]
    clean_inputs["filename_prefix"] = "h3_spatial_director/audio_redub_clean"
    clean_inputs["save_output"] = False
    graph[CLEAN_DECODE_NODE_ID] = {"class_type": "VHS_VideoCombine", "inputs": clean_inputs}

    return {
        "mode": "audio_redub",
        "redub_mode": resolved_redub_mode,
        "video_ref": video_ref,
        "seed": seed,
        "length": length,
        "width": width,
        "height": height,
        "denoise": denoise,
        "steps": steps,
        "use_distill": use_distill,
        "has_audio_refine": True,
        "nodes_summary": {
            "has_sound_cues": has_cues,
            "has_audio_refine": True,
            "redub_mode": resolved_redub_mode,
            "schedule": schedule_kind,
            "use_distill": use_distill,
            "denoise": round(denoise, 3),
            "steps": steps,
            "video_source": "LoadVideo->VAEEncode",
            "audio_source": "fresh_empty" if denoise >= 0.999 else "MiniMaxH3EncodeTrack",
            "hud_save_output": False,
            "clean_save_output": False,
        },
        "graph": graph,
    }


def get_schema_data(object_info: dict[str, Any] | None = None, *, heylook_suggestions: bool = False) -> dict[str, Any]:
    """Return full schema metadata for ``GET /api/schema``."""
    cues = _sound_cues()
    sam3_cap = detect_sam3_capability(object_info)
    canvases = [
        {"label": "7:4 Wide (1344×768)", "width": 1344, "height": 768, "aspect": "7:4"},
        {"label": "4:3 Standard (1152×896)", "width": 1152, "height": 896, "aspect": "4:3"},
        {"label": "1:1 Square (768×768)", "width": 768, "height": 768, "aspect": "1:1"},
        {"label": "4:7 Tall (768×1344)", "width": 768, "height": 1344, "aspect": "4:7"},
    ]
    return {
        "ok": True,
        "lengths": list(all_on_both_clocks()),
        "sound_presets": dict(cues.SOUND_CUE_PRESETS),
        "canvases": canvases,
        "workflows": available_workflows(),
        "anchor_workflows": available_anchor_workflows(),
        "image_edit_workflows": available_image_edit_workflows(),
        "control_vectors": list_control_vectors(),
        "capabilities": {
            "kj_preview_override": has_kj_preview_override(object_info, TAEH3_VIDEO_FILE),
            "taeh3": has_kj_preview_override(object_info, TAEH3_VIDEO_FILE),
            "taeh3_image": has_kj_preview_override(object_info, TAEH3_IMAGE_FILE),
            "heylook_suggestions": bool(heylook_suggestions),
            "sam3": bool(sam3_cap["available"]),
            "sam3_engine": sam3_cap["engine"],
            "spatial_combined": has_spatial_combined_support(object_info),
            "guide_latent": has_guide_latent_support(object_info),
            "audio_refine": has_audio_refine_support(object_info),
        },
        "defaults": {
            "mode": "i2v_portal",
            "length": 141,
            "width": 1344,
            "height": 768,
            "steps": 6,
            "scale": 2.0,
            "audio_scale": 1.0,
            "streams": "independent",
            "first_step": 0,
            "last_step": 1000,
            "align_positions": False,
            "portal_shape": "circle",
            "feather": 0.08,
            "halo_radius": 0.0,
            "hud_layout": "director_card",
            "anchor_workflow": "h3_image",
            "preview_image_workflow": "auto",
            "preview_fps": 7,
        },
    }
