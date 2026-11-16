"""Runtime workflow compiler for Spatial World Director.

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
PREVIEW_NODE_CLASS = "ModelPreviewOverrideKJ"
PREVIEW_NODE_ID = "165"
CLEAN_DECODE_NODE_ID = "162"
HUD_COMBINE_NODE_ID = "13"

ANCHOR_WORKFLOW_TEMPLATES: dict[str, dict[str, Any]] = {
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
    "h3_image": {
        "id": "h3_image",
        "label": "MiniMax-H3-Image (base)",
        "path": _WORKFLOWS_DIR / "image" / "h3_image_t2i_api.json",
    },
}

SPATIAL_WORKFLOW_SPECS: dict[str, dict[str, Any]] = {
    "r2v_portal": {
        "id": "r2v_portal",
        "label": "Reference Portal (R2V + optional Stamp)",
        "preferred": _WORKFLOWS_DIR / "h3_r2v_spatial_portal_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_world_targets_genie_pika_api.json",
        "modes": ["portal_path", "point_targets"],
        "has_ref_image": True,
        "has_first_frame": False,
        "supports_sprite": True,
    },
    "i2v_portal": {
        "id": "i2v_portal",
        "label": "First-Frame Portal (I2V)",
        "preferred": _WORKFLOWS_DIR / "h3_i2v_spatial_portal_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json",
        "modes": ["portal_path", "point_targets"],
        "has_ref_image": False,
        "has_first_frame": True,
        "supports_sprite": False,
    },
    "t2v_portal": {
        "id": "t2v_portal",
        "label": "Text-to-Video Portal (T2V)",
        "preferred": _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_spatial_portal_pika_api.json",
        "modes": ["portal_path", "point_targets"],
        "has_ref_image": False,
        "has_first_frame": False,
        "supports_sprite": False,
    },
    "world_targets": {
        "id": "world_targets",
        "label": "World Targets + Sprite & SoundCues",
        "preferred": _WORKFLOWS_DIR / "h3_world_targets_genie_pika_api.json",
        "fallback": _WORKFLOWS_DIR / "h3_world_targets_genie_pika_api.json",
        "modes": ["point_targets", "portal_path"],
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


def compile_still(
    params: dict[str, Any] | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile an anchor still or stamp-cutout source generation workflow."""
    cfg = dict(params or {})
    workflow_key = str(cfg.get("workflow") or "pika_image").strip()
    if workflow_key not in ANCHOR_WORKFLOW_TEMPLATES:
        raise ValueError(
            f"Unknown anchor workflow {workflow_key!r}; expected one of {list(ANCHOR_WORKFLOW_TEMPLATES)}"
        )
    tpl_path = Path(ANCHOR_WORKFLOW_TEMPLATES[workflow_key]["path"])
    if not tpl_path.is_file():
        raise FileNotFoundError(f"Anchor workflow template missing: {tpl_path}")
    graph = _load_json(tpl_path)
    _rewrite_save_to_preview(graph)

    seed = _resolve_seed(cfg, graph)
    if "6" in graph and isinstance(graph["6"].get("inputs"), dict):
        graph["6"]["inputs"]["noise_seed"] = seed

    prompt_text = str(cfg.get("prompt") or "").strip()
    if prompt_text and "5" in graph and isinstance(graph["5"].get("inputs"), dict):
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
            if cfg.get("megapixels") is not None:
                n4["megapixels"] = float(cfg["megapixels"])
            if cfg.get("generate_megapixels") is not None:
                n4["generate_megapixels"] = float(cfg["generate_megapixels"])

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
    if director_mode == "point_targets":
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

    latent_steps = 5 * ((int(length) - 5) // 17) + 2
    injected = _inject_kj_preview_override(
        graph,
        object_info=object_info,
        tiny_vae_filename=TAEH3_VIDEO_FILE,
        preview_frames=latent_steps,
        preview_fps=7,  # Matches ~7.06 Hz TinyVAEDecoder; never fps=24.0
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

    if is_r2v:
        if source_ref:
            graph.setdefault("15", {"class_type": "LoadImage", "inputs": {}})
            graph["15"]["inputs"]["image"] = source_ref
    elif mode == "i2v_portal":
        img_name = source_ref or graph.get("15", {}).get("inputs", {}).get("image") or "1-man.png"
        graph["15"] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
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

    # Prompts: edit_lines OR natural target_prompt OR same_prompt=True.
    if "133" not in graph:
        default_orig = str(graph.get("5", {}).get("inputs", {}).get("prompt") or "")
        graph["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": default_orig}}
        graph["5"]["inputs"]["prompt"] = ["133", 0]

    if params.get("original_prompt") is not None and str(params["original_prompt"]).strip():
        graph["133"]["inputs"]["value"] = str(params["original_prompt"]).strip()
    orig_prompt = str(graph["133"]["inputs"]["value"])

    same_prompt = bool(params.get("same_prompt", False))
    edit_lines = str(params.get("edit_lines") or "").strip()
    target_prompt = str(params.get("target_prompt") or "").strip()

    if same_prompt:
        graph.pop("130", None)
        graph.pop("134", None)
        graph["131"]["inputs"]["prompt"] = ["133", 0]
        prompt_mode = "same_prompt"
    elif edit_lines:
        # Validate edit lines immediately against original prompt.
        _h3_prompt_edit().apply_edits(orig_prompt, edit_lines)
        graph.pop("134", None)
        graph["130"] = {
            "class_type": "MiniMaxH3PromptEdit",
            "inputs": {"original": ["133", 0], "edits": edit_lines},
        }
        graph["131"]["inputs"]["prompt"] = ["130", 0]
        prompt_mode = "edit_lines"
    elif target_prompt:
        graph.pop("130", None)
        graph["134"] = {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": target_prompt},
        }
        graph["131"]["inputs"]["prompt"] = ["134", 0]
        prompt_mode = "target_prompt"
    else:
        prompt_mode = "template_default"

    # SpatialDirector ("160") & DirectorHUD ("161").
    n160 = graph["160"]["inputs"]
    default_dir_mode = "point_targets" if mode == "world_targets" else "portal_path"
    director_mode = str(params.get("director_mode") or (default_dir_mode if not used_shipped else n160.get("mode", default_dir_mode)))
    fallback_events = rules.DEFAULT_DIRECTOR_EVENTS_JSON if (not used_shipped and mode == "r2v_portal") else str(n160.get("events_json") or rules.DEFAULT_DIRECTOR_EVENTS_JSON)
    events_doc, _ = _parse_events_doc(params.get("events_json"), fallback_events)
    if params.get("events_json") is None and length != tpl_length:
        _scale_events_doc(events_doc, tpl_length, length)

    # Merge explicit portal_cues / dpad / halo couplings before validation & HUD wiring.
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

    if director_mode == "point_targets" and not events_doc.get("targets") and events_doc.get("portal"):
        # Allow mode switching seamlessly if user hasn't placed point targets yet.
        director_mode = "portal_path"

    events_str = json.dumps(events_doc, indent=2)
    _spatial_director().parse_director_events(events_str, mode=director_mode, length=length)

    n160["mode"] = director_mode
    n160["width"] = ["27", 0]
    n160["height"] = ["27", 1]
    n160["length"] = ["27", 2]
    n160["feather"] = _feather_value(params, width, height, float(n160.get("feather", 0.08)))
    n160["invert_mask"] = bool(params.get("invert_mask", n160.get("invert_mask", False)))
    n160["events_json"] = events_str

    n161 = graph["161"]["inputs"]
    if params.get("hud_layout"):
        n161["hud_layout"] = str(params["hud_layout"])
    if params.get("hud_title") is not None:
        n161["title"] = str(params["hud_title"])
    if params.get("hud_subtitle") is not None:
        n161["subtitle"] = str(params["hud_subtitle"])
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
        graph["51"] = {
            "class_type": "MiniMaxH3AppendSpriteTrajectory",
            "inputs": {
                "image": sprite_loader_ref,
                "length": ["27", 2],
                "keyframes": ["160", 2],
                "canvas_width": int(params.get("sprite_canvas_width", 384)),
                "canvas_height": int(params.get("sprite_canvas_height", 224)),
                "sprite_scale": float(params.get("sprite_scale", 0.31)),
                "use_vae": True,
                "references": ["50", 0],
            },
        }
        graph["132"]["inputs"]["references"] = ["51", 0]
    else:
        graph.pop("51", None)
        graph.pop("52", None)
        if is_r2v and "132" in graph:
            graph["132"]["inputs"]["references"] = ["50", 0]

    # Seed & noise.
    seed = _resolve_seed(params, graph)
    graph["6"]["inputs"]["noise_seed"] = seed

    # Conditional SoundCues ("70") + Preflight ("26") + EditGuider ("9") positive inputs.
    has_cues = _events_have_sound_cues(events_doc, params, director_mode=director_mode)
    latent_source = "132" if is_r2v else "5"
    if has_cues:
        graph["70"] = {
            "class_type": "MiniMaxH3SoundCues",
            "inputs": {
                "positive": ["5", 0],
                "latent": [latent_source, 1],
                "audio_vae": ["4", 0],
                "cues": ["160", 1],
                "snap_to_3f": True,
                "floor_dbfs": float(params.get("floor_dbfs", -28.0)),
                "seed": int(params.get("audio_seed", seed)),
                "edited_positive": ["131", 0],
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
            "conditioning": ["5", 0],
            "samples": [latent_source, 1],
            "edited_conditioning": ["131", 0],
        }
        edited_ref = ["131", 0]

    n9 = graph["9"]["inputs"]
    n9["original"] = ["26", 0]
    n9["edited"] = edited_ref
    n9["mask"] = ["160", 0]
    if scale_override is not None:
        n9["scale"] = float(scale_override)
    elif params.get("scale") is not None:
        n9["scale"] = float(params["scale"])
    if params.get("audio_scale") is not None:
        n9["audio_scale"] = float(params["audio_scale"])
    if params.get("streams") is not None:
        n9["streams"] = str(params["streams"])
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
            "has_sound_cues": has_cues,
            "has_preview_override": injected,
            "has_release_refs": "135" in graph,
            "control_vector_nodes": steer_ids,
            "prompt_mode": prompt_mode,
            "feather": round(float(n160["feather"]), 4),
            "halo_radius": round(max(0.0, min(0.35, float(params["halo_radius"]))) if params.get("halo_radius") is not None else 0.0, 4),
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
    length = int(cfg.get("length", 90))
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
    if mode in ("r2v_portal", "world_targets"):
        ref_tokens += tokens_per_frame
        if cfg.get("sprite_ref") or cfg.get("sprite_cutout") or cfg.get("sprite_image"):
            sw = int(cfg.get("sprite_canvas_width", 384))
            sh = int(cfg.get("sprite_canvas_height", 224))
            ref_tokens += max(1, (sw // 32) * (sh // 32))
    elif mode == "i2v_portal":
        ref_tokens += tokens_per_frame

    prompt_text = " ".join(
        str(cfg.get(k) or "") for k in ("original_prompt", "target_prompt", "edit_lines")
    ).strip()
    prompt_tokens_est = max(64, len(prompt_text) // 4 + 64)
    total_tokens = video_tokens + audio_tokens + ref_tokens + prompt_tokens_est
    exceeds_int32 = total_tokens >= INT32_TOKEN_LIMIT

    warnings: list[str] = []
    if length != 1 and length not in clocks:
        warnings.append(f"length={length} is off on_both_clocks {list(clocks)}")
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
        "can_render": (not exceeds_int32) and (width % 32 == 0) and (height % 32 == 0) and (length == 1 or length in clocks),
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


def get_schema_data(object_info: dict[str, Any] | None = None, *, heylook_suggestions: bool = False) -> dict[str, Any]:
    """Return full schema metadata for ``GET /api/schema``."""
    cues = _sound_cues()
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
        "control_vectors": list_control_vectors(),
        "capabilities": {
            "kj_preview_override": has_kj_preview_override(object_info, TAEH3_VIDEO_FILE),
            "taeh3": has_kj_preview_override(object_info, TAEH3_VIDEO_FILE),
            "taeh3_image": has_kj_preview_override(object_info, TAEH3_IMAGE_FILE),
            "heylook_suggestions": bool(heylook_suggestions),
        },
        "defaults": {
            "mode": "r2v_portal",
            "length": 90,
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
            "anchor_workflow": "pika_image",
            "preview_fps": 7,
        },
    }
