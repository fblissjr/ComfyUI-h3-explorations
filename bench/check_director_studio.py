#!/usr/bin/env python3
"""Two-tier contract, security, compiler, and live Chrome CDP checks for Spatial Director Studio (`director/`).

Claims, i.e. what breaks if a case is deleted:
  compiler_contracts       `director/compiler.py` compiles `r2v_portal`, `i2v_portal`, `t2v_portal`,
                           `world_targets`, and `pika_t2i` stills with `save_output=False`, dual
                           `VHS_VideoCombine` (`"13"` HUD + `"162"` clean), `"51"`/`"70"` omitted when
                           unused, `queue_baseline=False` default (+ opt-in `scale=0.0` same-seed
                           companion), and dynamic `"165"` `ModelPreviewOverrideKJ` (`preview_fps=7` +
                           `tiny_vae="taeh3.safetensors"`, or `preview_frames=1, preview_fps=1` +
                           `taeh3_image.safetensors` for stills) injected iff `/object_info` exposes it.
  forget_isolation         `director/routes.py` `compute_forget_plan` and `delete_temp_files` remove
                           targeted director temp `images`/`audio`/`gifs` and orphaned
                           `h3_spatial_director/` uploads while protecting `$H3_OUTPUT_DIR/Video/`,
                           non-director history items, active queued prompts, and `keep_uploads`.
  escalate_clocks          `POST /api/escalate` scales portal/target/dpad/cue frames proportionally
                           across `h3_rules.on_both_clocks` (`39, 90, 141, 192, ...`) and snaps cues
                           to `3f` integer 40 Hz Audio VAE RoPE boundaries.
  static_and_route_audit   `director/static/` contains zero `https://`/`http://` CDN links, zero
                           inline `style=`/`onclick=`, zero `element.style.` mutations, zero
                           hardcoded `SOUND_CUE_PRESETS` or scene suggestion arrays, zero owner paths,
                           all relative JS imports resolve, and every `/api/...` route named in
                           `director/static/js/*.js` is registered in `director/server.py`.
  live_chrome_cdp          (Tier 2: `--live` or default when Chrome & ComfyUI exist) spawns isolated
                           `--cpu` ComfyUI (`127.0.0.1:8288`) + `director/server.py` (`127.0.0.1:8295`)
                           and drives headless Chrome (`1280x800`) via CDP WebSocket: zero console
                           errors, zero `securitypolicyviolation` CSP events (including `blob:` media
                           under `media-src`), Light Mode surface luminance (`> 220`), `Render Portal
                           Edit` & `Compare Original` fit inside `1280px` (`rect.right <= 1280`),
                           `#live-tae-pill` unclipped, `GET /api/schema` hydrates controls & detects
                           `ModelPreviewOverrideKJ`, anchor PNG upload, stage click contextual popover,
                           `Wheel` (`rx`/`ry`), `Shift+Wheel` (`halo_radius`), `Alt+Wheel` (`scale`),
                           `ArrowRight` `tablist` navigation in `Engine` drawer, `POST /api/compile`
                           emitting `"165"`, and simulated `"kj_preview_override"` frame compositing
                           onto `<canvas>`.

Usage:
    <comfy-venv-python> bench/check_director_studio.py            # Tier 1 + Tier 2 if Chrome & ComfyUI exist
    <comfy-venv-python> bench/check_director_studio.py --offline  # Tier 1 offline suite only
    <comfy-venv-python> bench/check_director_studio.py --live     # Force Tier 1 + Tier 2 (fail if missing)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import inspect
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _discover_comfy_root() -> Path:
    env_root = os.environ.get("COMFY_ROOT")
    if env_root and (Path(env_root) / "comfy").is_dir():
        return Path(env_root)
    for parent in REPO.parents:
        if (parent / "comfy").is_dir():
            return parent
    # Note: do not .resolve() sys.executable — .venv/bin/python may be a symlink.
    for parent in Path(sys.executable).parents:
        if (parent / "comfy").is_dir():
            return parent
    return REPO.parents[1]


COMFY_ROOT = _discover_comfy_root()
for _p in (str(REPO / "workflows"), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _lib  # noqa: E402

if (_lib.COMFY / "comfy").is_dir() is False and (COMFY_ROOT / "comfy").is_dir():
    _lib.COMFY = COMFY_ROOT

from _lib import bootstrap, case, finish, needs, skip  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

MOCK_OBJECT_INFO_WITH_KJ: dict[str, Any] = {
    "ModelPreviewOverrideKJ": {
        "input": {
            "required": {
                "model": ["MODEL", {}],
                "max_resolution": ["INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}],
                "jpeg_quality": ["INT", {"default": 80, "min": 30, "max": 100, "step": 1}],
                "suppress_default_preview": ["BOOLEAN", {"default": True}],
                "preview_frames": ["INT", {"default": 1, "min": 1, "max": 1024, "step": 1}],
                "preview_fps": ["INT", {"default": 12, "min": 1, "max": 60, "step": 1}],
            },
            "optional": {
                "vae": ["VAE", {}],
                "tiny_vae": [
                    "COMBO",
                    {
                        "options": ["none", "taeh3.safetensors", "taeh3_image.safetensors"],
                        "default": "none",
                    },
                ],
                "audio_vae": ["VAE", {}],
            },
        }
    },
    "MiniMaxH3AudioRefineMask": {"input": {"required": {}, "optional": {}}},
    "MiniMaxH3FrozenVideoCache": {"input": {"required": {}, "optional": {}}},
}

MOCK_OBJECT_INFO_WITHOUT_KJ: dict[str, Any] = {
    "MiniMaxH3SpatialDirector": {"input": {"required": {}, "optional": {}}},
    "VHS_VideoCombine": {"input": {"required": {}, "optional": {}}},
}

SAMPLE_EVENTS_90 = {
    "portal": [
        {"frame": 0, "x": 0.20, "y": 0.50, "rx": 0.22, "ry": 0.22, "shape": "circle", "label": "PORTAL"},
        {"frame": 45, "x": 0.50, "y": 0.50, "rx": 0.25, "ry": 0.25, "shape": "circle", "label": "PORTAL"},
        {"frame": 89, "x": 0.80, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "circle", "label": "PORTAL"},
    ],
    "targets": [
        {"id": 1, "label": "TARGET A", "start": 12, "end": 42, "ramp": 3, "x": 0.30, "y": 0.50, "radius": 0.18, "cue": "slam", "gain": 1.1},
        {"id": 2, "label": "TARGET B", "start": 52, "end": 82, "ramp": 3, "x": 0.72, "y": 0.50, "radius": 0.18, "cue": "hop", "gain": 1.0},
    ],
    "portal_cues": [
        {"frame": 14, "preset": "click", "gain": 0.9},
        {"frame": 47, "preset": "snap", "gain": 1.0},
    ],
    "dpad": {
        "waypoints": [
            {"frame": 0, "x": 0.22, "y": 0.74, "face": "right", "scale": 1.0},
            {"frame": 45, "x": 0.52, "y": 0.74, "face": "right", "scale": 1.0},
            {"frame": 89, "x": 0.78, "y": 0.74, "face": "left", "scale": 1.0},
        ],
        "hops": [{"start": 30, "end": 44, "lift": 0.24}],
    },
}


def _director_present() -> bool:
    d = REPO / "director"
    return (
        (d / "compiler.py").is_file()
        and (d / "routes.py").is_file()
        and (d / "server.py").is_file()
        and (d / "static" / "index.html").is_file()
    )


def _find_chrome() -> str | None:
    cand = Path("/usr/bin/google-chrome")
    if cand.is_file() and os.access(cand, os.X_OK):
        return str(cand)
    return (
        shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _load_director_modules():
    compiler_mod = importlib.import_module("director.compiler")
    routes_mod = importlib.import_module("director.routes")
    server_mod = importlib.import_module("director.server")
    return compiler_mod, routes_mod, server_mod


def _extract_graphs(res: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if isinstance(res, tuple):
        return res[0], (res[1] if len(res) > 1 else None)
    if isinstance(res, dict):
        if "graph" in res:
            return res["graph"], res.get("baseline_graph") or res.get("baseline")
        if "prompt" in res and isinstance(res["prompt"], dict):
            return res["prompt"], res.get("baseline_graph") or res.get("baseline")
        return res, None
    raise AssertionError(f"Unexpected compiler result type: {type(res).__name__}")


def _call_with_adaptive_sig(fn, payload: dict[str, Any], object_info: dict[str, Any] | None):
    sig = inspect.signature(fn)
    params = sig.parameters
    positional = [
        p for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if positional and positional[0].name in ("payload", "params", "spec", "req", "request", "body", "config", "data"):
        kwargs: dict[str, Any] = {}
        if "object_info" in params:
            kwargs["object_info"] = object_info
        return fn(payload, **kwargs)
    if len(positional) == 1 and positional[0].default is inspect.Parameter.empty and "mode" not in params:
        kwargs = {}
        if "object_info" in params:
            kwargs["object_info"] = object_info
        return fn(payload, **kwargs)
    call_kw = {k: v for k, v in payload.items() if k in params}
    if "object_info" in params:
        call_kw["object_info"] = object_info
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        for k, v in payload.items():
            call_kw.setdefault(k, v)
    return fn(**call_kw)


def _create_director_app(server_mod, *, comfy_url: str, state_dir: Path):
    sig = inspect.signature(server_mod.create_app)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "comfy_url" in params:
        kwargs["comfy_url"] = comfy_url
    if "state_dir" in params:
        kwargs["state_dir"] = state_dir
    if "allowed_hosts" in params:
        kwargs["allowed_hosts"] = None
    if not kwargs and len(params) >= 1:
        return server_mod.create_app(comfy_url)
    return server_mod.create_app(**kwargs)


def _nodes_by_class(graph: dict[str, Any], class_type: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (nid, node)
        for nid, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]


def _encode_test_png_bytes(width: int = 336, height: int = 192) -> bytes:
    from PIL import Image
    import numpy as np

    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, : width // 2, 0] = 210
    arr[:, : width // 2, 1] = 140
    arr[:, width // 2 :, 1] = 170
    arr[:, width // 2 :, 2] = 230
    arr[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = [245, 200, 60]
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _encode_solid_jpeg_b64(rgb: tuple[int, int, int] = (255, 0, 220), width: int = 336, height: int = 192) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_solid_mp4_b64(rgb: tuple[int, int, int] = (0, 235, 255), width: int = 320, height: int = 192, n_frames: int = 8) -> str:
    import av
    import numpy as np

    buf = io.BytesIO()
    frame_arr = np.empty((height, width, 3), dtype=np.uint8)
    frame_arr[:, :] = rgb
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=7)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(n_frames):
            vf = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_indexed_mp4_bytes(
    colors: list[tuple[int, int, int]] | None = None,
    width: int = 160,
    height: int = 96,
) -> bytes:
    import av
    import numpy as np

    palette = colors or [(230, 30, 30), (30, 220, 40), (30, 70, 235)]
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 1
        for rgb in palette:
            frame_arr = np.empty((height, width, 3), dtype=np.uint8)
            frame_arr[:, :] = rgb
            vf = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tier 1: Offline Checks
# ---------------------------------------------------------------------------

def check_compiler_contracts() -> str:
    compiler_mod, _routes_mod, _server_mod = _load_director_modules()

    base_events = json.dumps(SAMPLE_EVENTS_90)
    common_base = {
        "source_ref": "h3_spatial_director/test_anchor.png",
        "first_frame": "h3_spatial_director/test_anchor.png",
        "length": 90,
        "width": 1344,
        "height": 768,
        "seed": 1234567,
        "events_json": base_events,
        "target_prompt": "A tranquil stone courtyard opening into a swirling bioluminescent frost cavern.",
        "scale": 1.8,
        "first_step": 0,
        "last_step": 3,
        "streams": "independent",
        "audio_scale": 1.0,
        "queue_baseline": False,
    }

    # 1. All 4 spatial modes WITHOUT sprite cutout and WITHOUT cues -> "51"/"70" omitted, dual VHS save_output=False
    for mode in ("r2v_portal", "i2v_portal", "t2v_portal", "world_targets"):
        payload = dict(
            common_base,
            mode=mode,
            portal_cues=[],
            cues=[],
            sound_cues=[],
            sprite_ref="",
            events_json=json.dumps(
                {
                    "portal": SAMPLE_EVENTS_90["portal"],
                    "targets": [
                        {k: ("none" if k == "cue" else v) for k, v in t.items()}
                        for t in SAMPLE_EVENTS_90["targets"]
                    ],
                }
            ),
        )
        res = _call_with_adaptive_sig(compiler_mod.compile_spatial_take, payload, MOCK_OBJECT_INFO_WITH_KJ)
        g, baseline = _extract_graphs(res)
        assert baseline is None, f"{mode}: queue_baseline=False must not emit baseline graph"
        assert "13" in g and g["13"]["class_type"] == "VHS_VideoCombine", f"{mode}: missing HUD VHS '13'"
        assert "162" in g and g["162"]["class_type"] == "VHS_VideoCombine", f"{mode}: missing clean VHS '162'"
        assert g["13"]["inputs"].get("save_output") is False, f"{mode}: node 13 save_output must be False"
        assert g["162"]["inputs"].get("save_output") is False, f"{mode}: node 162 save_output must be False"
        assert g["13"]["inputs"]["images"] == ["161", 0], f"{mode}: node 13 must read HUD '161'"
        assert g["162"]["inputs"]["images"] == ["11", 0], f"{mode}: node 162 must read clean VAEDecode '11'"
        assert g["162"]["inputs"]["audio"] == g["13"]["inputs"]["audio"], f"{mode}: node 162 audio mismatch"
        assert "51" not in g and not _nodes_by_class(g, "MiniMaxH3AppendSpriteTrajectory"), (
            f"{mode}: node 51 must be omitted when no sprite cutout is attached"
        )
        assert "70" not in g and not _nodes_by_class(g, "MiniMaxH3SoundCues"), (
            f"{mode}: node 70 must be omitted when no cues are active"
        )
        # ModelPreviewOverrideKJ injected when /object_info exposes it + taeh3.safetensors
        assert "165" in g and g["165"]["class_type"] == "ModelPreviewOverrideKJ", f"{mode}: missing '165' preview override"
        po_in = g["165"]["inputs"]
        assert po_in.get("tiny_vae") == "taeh3.safetensors", f"{mode}: unexpected tiny_vae={po_in.get('tiny_vae')!r}"
        assert po_in.get("preview_fps") == 7 and isinstance(po_in.get("preview_fps"), int), (
            f"{mode}: preview_fps must be int 7, got {po_in.get('preview_fps')!r}"
        )
        assert po_in.get("preview_frames") == 5 * ((90 - 5) // 17) + 2, f"{mode}: unexpected preview_frames={po_in.get('preview_frames')}"
        assert po_in.get("suppress_default_preview") is True
        assert "audio_vae" not in po_in or po_in.get("audio_vae") is None
        assert g["9"]["inputs"]["model"] == ["165", 0], f"{mode}: guider '9' must consume ['165', 0]"

        # Absent /object_info -> "165" must be omitted
        res_no_kj = _call_with_adaptive_sig(compiler_mod.compile_spatial_take, payload, MOCK_OBJECT_INFO_WITHOUT_KJ)
        g_no_kj, _ = _extract_graphs(res_no_kj)
        assert "165" not in g_no_kj and not _nodes_by_class(g_no_kj, "ModelPreviewOverrideKJ"), (
            f"{mode}: '165' must be omitted when /object_info lacks ModelPreviewOverrideKJ"
        )
        assert g_no_kj["9"]["inputs"]["model"] != ["165", 0]

    assert compiler_mod.video_preview_schedule(39) == (12, 7)
    assert compiler_mod.video_preview_schedule(90) == (27, 7)
    assert compiler_mod.video_preview_schedule(141) == (27, 5)
    assert compiler_mod.video_preview_schedule(345) == (27, 2)
    res_345 = _call_with_adaptive_sig(compiler_mod.compile_spatial_take, {**common_base, "length": 345}, MOCK_OBJECT_INFO_WITH_KJ)
    g_345, _ = _extract_graphs(res_345)
    assert g_345["165"]["inputs"]["preview_frames"] == 27
    assert g_345["165"]["inputs"]["preview_fps"] == 2

    # 2. Sprite cutout + cues attached + queue_baseline=True + edit_lines
    payload_active = dict(
        common_base,
        mode="world_targets",
        sprite_ref="h3_spatial_director/cutout_1.png",
        portal_cues=[{"frame": 12, "preset": "slam", "gain": 1.1}, {"frame": 51, "preset": "hop", "gain": 1.0}],
        queue_baseline=True,
    )
    res_act = _call_with_adaptive_sig(compiler_mod.compile_spatial_take, payload_active, MOCK_OBJECT_INFO_WITH_KJ)
    g_act, g_base = _extract_graphs(res_act)
    assert "51" in g_act and g_act["51"]["class_type"] == "MiniMaxH3AppendSpriteTrajectory", "missing '51' when sprite_ref attached"
    assert "70" in g_act and g_act["70"]["class_type"] == "MiniMaxH3SoundCues", "missing '70' when cues attached"
    rel_nodes = _nodes_by_class(g_act, "MiniMaxH3ReleaseReferences")
    assert rel_nodes, "expected MiniMaxH3ReleaseReferences when sprite/sound cues are attached"
    assert abs(float(rel_nodes[0][1]["inputs"]["video_end_percent"]) - 0.33) < 1e-4
    assert abs(float(rel_nodes[0][1]["inputs"]["audio_end_percent"]) - 0.60) < 1e-4
    assert g_base is not None, "queue_baseline=True must emit companion baseline graph"
    assert float(g_base["9"]["inputs"]["scale"]) == 0.0, f"baseline guider scale must be 0.0, got {g_base['9']['inputs']['scale']}"
    assert g_base["6"]["inputs"]["noise_seed"] == g_act["6"]["inputs"]["noise_seed"], "baseline must share noise_seed"

    # edit_lines path creates MiniMaxH3PromptEdit ("130")
    payload_edits = dict(
        common_base,
        mode="t2v_portal",
        edit_lines="warm sunlit flagstones -> frozen midnight crystal shards",
        original_prompt="integrated_multimodal_description: [Shot 1] A courtyard of warm sunlit flagstones.\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: N/A",
    )
    res_edits = _call_with_adaptive_sig(compiler_mod.compile_spatial_take, payload_edits, MOCK_OBJECT_INFO_WITH_KJ)
    g_edits, _ = _extract_graphs(res_edits)
    assert "130" in g_edits and g_edits["130"]["class_type"] == "MiniMaxH3PromptEdit"

    # 4. Cross-Entity Spatial Halo Coupling: halo_radius widens feather, couplings shift sprite waypoints & append reaction_clause
    coupling_events = {
        "portal": [
            {"frame": 0, "x": 0.40, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "circle", "label": "PORTAL"},
            {"frame": 89, "x": 0.40, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "circle", "label": "PORTAL"},
        ],
        "dpad": {
            "waypoints": [
                {"frame": 0, "x": 0.60, "y": 0.50, "face": "right", "scale": 1.0},
                {"frame": 45, "x": 0.60, "y": 0.50, "face": "right", "scale": 1.0},
                {"frame": 89, "x": 0.60, "y": 0.50, "face": "right", "scale": 1.0},
            ],
            "hops": [],
        },
    }
    coupling_payload = dict(
        common_base,
        mode="r2v_portal",
        feather=0.08,
        halo_radius=0.25,
        sprite_ref="h3_spatial_director/cutout_1.png",
        events_json=json.dumps(coupling_events),
        couplings=[
            {
                "portal_index": 0,
                "target_kind": "sprite",
                "strength": 1.0,
                "mode": "push",
                "label": "Halo Push",
                "reaction_clause": "bioluminescent mist sweeps the tortoise rightward",
            }
        ],
        same_prompt=False,
    )
    res_couple = _call_with_adaptive_sig(compiler_mod.compile_spatial_take, coupling_payload, MOCK_OBJECT_INFO_WITH_KJ)
    g_couple, _ = _extract_graphs(res_couple)
    assert float(g_couple["160"]["inputs"]["feather"]) >= 0.25, (
        f"halo_radius=0.25 must widen feather >= 0.25, got {g_couple['160']['inputs']['feather']}"
    )
    # Prompt receives reaction_clause when same_prompt=False
    prompt_texts = [
        str(n.get("inputs", {}).get("value", "")) + str(n.get("inputs", {}).get("prompt", ""))
        for n in g_couple.values()
        if isinstance(n, dict)
    ]
    assert any("bioluminescent mist sweeps the tortoise rightward" in t for t in prompt_texts), (
        "reaction_clause must be appended into edited prompt when coupling is active and same_prompt=False"
    )
    # Post-onset sprite waypoint pushed away from portal (x_p=0.40 -> x_s=0.60 shifts +0.12 to ~0.72)
    coupled_ev = json.loads(g_couple["160"]["inputs"]["events_json"])
    coupled_wps = (coupled_ev.get("dpad") or {}).get("waypoints") or []
    assert any(float(wp["x"]) > 0.65 for wp in coupled_wps), (
        f"push coupling must offset sprite waypoint away from portal x=0.40 -> >0.65, got {coupled_wps}"
    )
    if isinstance(res_couple, dict) and isinstance(res_couple.get("nodes_summary"), dict):
        applied_list = res_couple["nodes_summary"].get("couplings_applied") or []
        assert len(applied_list) >= 1, res_couple["nodes_summary"]

    # 5. SAM 3.1 segmentation graph compiler contracts (points vs text prompt)
    mock_sam3_info = {
        **MOCK_OBJECT_INFO_WITH_KJ,
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["sam3.1_multiplex_fp16.safetensors", "sam3_fp32.safetensors"], {}]}}
        },
        "SAM3_Detect": {"input": {"required": {}, "optional": {}}},
        "MiniMaxH3SAM31Corrections": {"input": {"required": {}, "optional": {}}},
    }
    sam_pt = compiler_mod.compile_sam3_segment(
        {"image_ref": "h3_spatial_director/test_anchor.png", "positive_points": [{"nx": 0.5, "ny": 0.5}], "prompt": "ignore text"},
        image_width=1344,
        image_height=768,
        object_info=mock_sam3_info,
    )
    g_sam = sam_pt["graph"]
    assert g_sam["200"]["inputs"]["ckpt_name"] == "sam3.1_multiplex_fp16.safetensors"
    assert g_sam["201"]["class_type"] == "MiniMaxH3SAM31Corrections"
    assert g_sam["203"]["class_type"] == "PrimitiveString"
    assert json.loads(g_sam["203"]["inputs"]["value"]) == [{"x": 672, "y": 384}]
    assert g_sam["206"]["inputs"]["positive_coords"] == ["203", 0]
    assert "205" not in g_sam and "conditioning" not in g_sam["206"]["inputs"], (
        "Point click must not union CLIPTextEncode conditioning unless explicitly requested"
    )
    assert g_sam["207"]["class_type"] == "MaskToImage" and g_sam["208"]["class_type"] == "PreviewImage"

    sam_txt = compiler_mod.compile_sam3_segment(
        {"image_ref": "h3_spatial_director/test_anchor.png", "prompt": "bronze tortoise"},
        image_width=1344,
        image_height=768,
        object_info=mock_sam3_info,
    )
    assert sam_txt["graph"]["205"]["class_type"] == "CLIPTextEncode"
    assert sam_txt["graph"]["206"]["inputs"]["conditioning"] == ["205", 0]

    # 6. Combined mode + sprite_mask_ref + include_mask_ref + exclude_mask_ref (native AND pre-reload fallback)
    comb_payload = dict(
        common_base,
        mode="r2v_portal",
        director_mode="combined",
        sprite_ref="h3_spatial_director/cutout_1.png",
        sprite_mask_ref="h3_spatial_director/seg_cutout.png",
        include_mask_ref="h3_spatial_director/seg_include.png",
        exclude_mask_ref="h3_spatial_director/seg_exclude.png",
        invert_mask=True,
    )
    res_native = compiler_mod.compile_spatial_take(comb_payload, object_info=MOCK_OBJECT_INFO_WITH_KJ)
    g_nat = res_native["graph"]
    assert res_native["director_mode"] == "combined"
    assert g_nat["160"]["inputs"]["mode"] == "combined"
    assert g_nat["160"]["inputs"]["include_mask"] == ["171", 0]
    assert g_nat["160"]["inputs"]["exclude_mask"] == ["173", 0]
    assert g_nat["51"]["inputs"]["mask"] == ["54", 0]
    assert res_native["nodes_summary"]["has_sprite_mask"] is True
    assert res_native["nodes_summary"]["has_include_mask"] is True
    assert res_native["nodes_summary"]["has_exclude_mask"] is True
    assert res_native["nodes_summary"]["spatial_combined_native"] is True

    mock_pre_reload = {
        **MOCK_OBJECT_INFO_WITH_KJ,
        "MiniMaxH3SpatialDirector": {
            "input": {
                "required": {"mode": ["COMBO", {"options": ["portal_path", "point_targets"]}]},
                "optional": {"backdrop": ["IMAGE", {}]},
            }
        },
    }
    res_fb = compiler_mod.compile_spatial_take(comb_payload, object_info=mock_pre_reload)
    g_fb = res_fb["graph"]
    assert res_fb["nodes_summary"]["spatial_combined_native"] is False
    assert g_fb["160"]["inputs"]["mode"] == "portal_path"
    assert "include_mask" not in g_fb["160"]["inputs"] and "exclude_mask" not in g_fb["160"]["inputs"]
    assert g_fb["166"]["class_type"] == "MiniMaxH3SpatialDirector" and g_fb["166"]["inputs"]["mode"] == "point_targets"
    assert g_fb["167"]["class_type"] == "MaskComposite" and g_fb["167"]["inputs"]["operation"] == "add"
    assert g_fb["168"]["class_type"] == "PrimitiveStringMultiline" and g_fb["70"]["inputs"]["cues"] == ["168", 0]
    assert g_fb["171"]["class_type"] == "ImageScale" and g_fb["174"]["class_type"] == "InvertMask"
    assert g_fb["179"]["class_type"] == "MaskComposite" and g_fb["179"]["inputs"]["operation"] == "multiply"
    assert g_fb["9"]["inputs"]["mask"] == ["179", 0]

    # 7. Audio-Gate Guard (`streams` defaults & summary in `point_targets` vs explicit override)
    pt_no_streams = {
        k: v
        for k, v in common_base.items()
        if k != "streams"
    }
    pt_no_streams.update(mode="r2v_portal", director_mode="point_targets")
    res_pt_def = compiler_mod.compile_spatial_take(pt_no_streams, object_info=MOCK_OBJECT_INFO_WITH_KJ)
    assert res_pt_def["graph"]["9"]["inputs"]["streams"] == "video_only", res_pt_def["graph"]["9"]["inputs"]
    assert res_pt_def["nodes_summary"]["streams"] == "video_only"
    assert res_pt_def["nodes_summary"]["audio_gated_by_mask"] is False

    res_pt_both = compiler_mod.compile_spatial_take({**pt_no_streams, "streams": "both"}, object_info=MOCK_OBJECT_INFO_WITH_KJ)
    assert res_pt_both["graph"]["9"]["inputs"]["streams"] == "both"
    assert res_pt_both["nodes_summary"]["audio_gated_by_mask"] is True

    res_pt_ind = compiler_mod.compile_spatial_take({**pt_no_streams, "streams": "independent"}, object_info=MOCK_OBJECT_INFO_WITH_KJ)
    assert res_pt_ind["graph"]["9"]["inputs"]["streams"] == "independent"

    # 3. Anchor / Stamp Still (`compile_still` with `pika_image` -> `h3_pika_image_t2i` default)
    still_payload = {
        "workflow": "pika_image",
        "prompt": "integrated_multimodal_description: [Shot 1] A sunlit limestone archway.\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A",
        "width": 768,
        "height": 768,
        "seed": 424242,
    }
    res_still_kj = _call_with_adaptive_sig(compiler_mod.compile_still, still_payload, MOCK_OBJECT_INFO_WITH_KJ)
    g_still_kj, _ = _extract_graphs(res_still_kj)
    assert not _nodes_by_class(g_still_kj, "SaveImage"), "compile_still must rewrite SaveImage -> PreviewImage"
    assert _nodes_by_class(g_still_kj, "PreviewImage"), "compile_still must emit PreviewImage"
    assert "165" in g_still_kj and g_still_kj["165"]["class_type"] == "ModelPreviewOverrideKJ"
    assert g_still_kj["165"]["inputs"].get("tiny_vae") == "taeh3_image.safetensors"
    assert g_still_kj["165"]["inputs"].get("preview_frames") == 1
    assert g_still_kj["165"]["inputs"].get("preview_fps") == 1
    assert g_still_kj["9"]["inputs"]["model"] == ["165", 0]

    res_still_nokj = _call_with_adaptive_sig(compiler_mod.compile_still, still_payload, MOCK_OBJECT_INFO_WITHOUT_KJ)
    g_still_nokj, _ = _extract_graphs(res_still_nokj)
    assert "165" not in g_still_nokj and not _nodes_by_class(g_still_nokj, "ModelPreviewOverrideKJ")

    # 8. Slice 2C: 2nd-Pass Audio Refine (in-graph) & Standalone Audio Re-Dub
    assert compiler_mod.has_audio_refine_support(MOCK_OBJECT_INFO_WITH_KJ) is True
    assert compiler_mod.has_audio_refine_support({"MiniMaxH3SpatialDirector": {"input": {"required": {"mode": ["COMBO", {}]}}}}) is False
    res_ar = compiler_mod.compile_spatial_take(
        {**common_base, "mode": "t2v_portal", "director_mode": "combined", "audio_refine": True},
        object_info=MOCK_OBJECT_INFO_WITH_KJ,
    )
    g_ar = res_ar["graph"]
    for nid in ("80", "81", "82", "83", "84", "85", "86", "87", "88"):
        assert nid in g_ar, f"audio_refine=True missing node #{nid}"
    assert g_ar["88"]["class_type"] == "MiniMaxH3FrozenVideoCache"
    assert g_ar["83"]["class_type"] == "MiniMaxH3AudioRefineMask"
    assert g_ar["87"]["class_type"] == "SamplerCustomAdvanced"
    assert g_ar["11"]["inputs"]["samples"] == ["87", 0]
    assert g_ar["12"]["inputs"]["samples"] == ["87", 0]
    assert res_ar["nodes_summary"]["has_audio_refine"] is True

    res_rd1 = compiler_mod.compile_audio_redub(
        {
            "video_ref": "h3_spatial_director/shot1.mp4",
            "length": 90,
            "redub_mode": "fresh",
            "portal_cues": [{"frame": 14, "preset": "slam", "gain": 1.0}],
        },
        object_info=MOCK_OBJECT_INFO_WITH_KJ,
    )
    g_rd1 = res_rd1["graph"]
    assert res_rd1["redub_mode"] == "fresh" and res_rd1["use_distill"] is True
    assert g_rd1["90"]["class_type"] == "LoadVideo" and g_rd1["90"]["inputs"]["file"] == "h3_spatial_director/shot1.mp4"
    assert g_rd1["93"]["class_type"] == "VAEEncode"
    assert g_rd1["94"]["class_type"] == "LTXVSeparateAVLatent"
    assert g_rd1["95"]["class_type"] == "LTXVConcatAVLatent"
    assert g_rd1["70"]["class_type"] == "MiniMaxH3SoundCues" and "15: slam" in g_rd1["70"]["inputs"]["cues"]
    assert g_rd1["88"]["class_type"] == "MiniMaxH3FrozenVideoCache"
    assert g_rd1["83"]["class_type"] == "MiniMaxH3AudioRefineMask"
    assert g_rd1["60"]["class_type"] == "ManualSigmas" and "84" not in g_rd1
    assert g_rd1["87"]["inputs"]["sigmas"] == ["60", 0]
    assert g_rd1["85"]["inputs"]["sampler_name"] == "lcm" and "18" in g_rd1
    assert g_rd1["162"]["inputs"]["images"] == ["92", 0] and g_rd1["162"]["inputs"]["save_output"] is False
    assert g_rd1["13"]["inputs"]["images"] == ["92", 0] and g_rd1["13"]["inputs"]["save_output"] is False

    res_rd2 = compiler_mod.compile_audio_redub(
        {
            "video_ref": "h3_spatial_director/shot1.mp4",
            "length": 141,
            "source_length": 90,
            "redub_mode": "polish",
            "events_json": base_events,
        },
        object_info=MOCK_OBJECT_INFO_WITH_KJ,
    )
    g_rd2 = res_rd2["graph"]
    assert res_rd2["redub_mode"] == "polish" and res_rd2["use_distill"] is False
    assert g_rd2["94a"]["class_type"] == "LTXVSeparateAVLatent"
    assert g_rd2["94b"]["class_type"] == "LTXVConcatAVLatent"
    assert g_rd2["94"]["class_type"] == "MiniMaxH3EncodeTrack"
    assert g_rd2["95"]["inputs"]["video_latent"] == ["94b", 0]
    assert "18" not in g_rd2 and "60" not in g_rd2
    assert g_rd2["84"]["class_type"] == "BasicScheduler" and g_rd2["87"]["inputs"]["sigmas"] == ["84", 0]
    assert g_rd2["85"]["inputs"]["sampler_name"] == "euler"
    assert abs(float(g_rd2["84"]["inputs"]["denoise"]) - 0.5) < 1e-6

    res_lf = compiler_mod.compile_spatial_take(
        {**common_base, "mode": "i2v_portal", "last_frame": "h3_spatial_director/shot_end.png"},
        object_info=MOCK_OBJECT_INFO_WITH_KJ,
    )
    g_lf = res_lf["graph"]
    assert g_lf["16"]["class_type"] == "LoadImage" and g_lf["16"]["inputs"]["image"] == "h3_spatial_director/shot_end.png"
    assert g_lf["5"]["inputs"]["last_frame"] == ["16", 0] and g_lf["131"]["inputs"]["last_frame"] == ["16", 0]

    # 9. Scene-agnostic prompt synthesis (zero tortoise leakage) & missing-source_ref refusal on I2V/R2V
    for clean_mode in ("r2v_portal", "i2v_portal", "t2v_portal", "world_targets"):
        clean_res = compiler_mod.compile_spatial_take(
            {
                "mode": clean_mode,
                "source_ref": "h3_spatial_director/custom_scene.png" if clean_mode != "t2v_portal" else "",
                "length": 90,
                "width": 1344,
                "height": 768,
                "seed": 77,
                "target_prompt": "it bursts into glowing confetti",
                "events_json": json.dumps(SAMPLE_EVENTS_90),
            },
            object_info=MOCK_OBJECT_INFO_WITH_KJ,
        )
        cg = clean_res["graph"]
        blob = (
            str(cg.get("133", {}).get("inputs", {}).get("value", ""))
            + str(cg.get("134", {}).get("inputs", {}).get("value", ""))
            + str(cg.get("161", {}).get("inputs", {}).get("subtitle", ""))
        ).lower()
        assert "tortoise" not in blob and "turtle" not in blob, f"{clean_mode}: leaked hardcoded tortoise text"
        if clean_mode != "t2v_portal":
            assert "<Picture 1>" in str(cg["133"]["inputs"]["value"]), f"{clean_mode}: missing <Picture 1> anchor"
        assert "it bursts into glowing confetti" in str(cg["134"]["inputs"]["value"])

    for req_mode in ("i2v_portal", "r2v_portal", "world_targets"):
        raised = False
        try:
            compiler_mod.compile_spatial_take({"mode": req_mode, "length": 90, "target_prompt": "sparks"})
        except ValueError:
            raised = True
        assert raised, f"{req_mode} must refuse empty source_ref instead of falling back to hardcoded sample image"

    # 10. Cutout generation defaults to h3_image 4.0 MP floor & masked still edit works across all 3 image templates
    cutout_res = compiler_mod.compile_still(
        {"prompt": "a brass clockwork owl", "for_cutout": True},
        object_info=MOCK_OBJECT_INFO_WITH_KJ,
    )
    assert cutout_res["workflow"] == "h3_image" and cutout_res["for_cutout"] is True
    assert float(cutout_res["graph"]["4"]["inputs"]["megapixels"]) >= 4.0
    assert float(cutout_res["graph"]["4"]["inputs"]["generate_megapixels"]) >= 4.0
    assert cutout_res["graph"]["4"]["inputs"]["shape"] == "1:1"
    assert "backdrop" in str(cutout_res["graph"]["5"]["inputs"]["prompt"]).lower()

    for prev_wf in ("h3_image", "pika_image", "ph3_image"):
        m_prev = compiler_mod.compile_masked_still_edit(
            {
                "workflow": prev_wf,
                "source_ref": "h3_spatial_director/test_anchor.png",
                "mask_ref": "h3_spatial_director/test_mask.png",
                "target_prompt": "a swirling aurora portal",
                "seed": 1234,
            },
            object_info=MOCK_OBJECT_INFO_WITH_KJ,
        )
        mg = m_prev["graph"]
        assert not _nodes_by_class(mg, "SaveImage"), f"{prev_wf}: SaveImage forbidden"
        assert _nodes_by_class(mg, "PreviewImage"), f"{prev_wf}: PreviewImage required"
        assert mg["171"]["class_type"] == "ImageToMask"
        assert mg["9"]["class_type"] == "MiniMaxH3EditGuider" and mg["9"]["inputs"]["mask"] == ["171", 0]
        assert mg["165"]["class_type"] == "ModelPreviewOverrideKJ"
        assert mg["165"]["inputs"]["tiny_vae"] == "taeh3_image.safetensors"
        assert mg["165"]["inputs"]["preview_frames"] == 1 and mg["165"]["inputs"]["preview_fps"] == 1
        if prev_wf == "pika_image":
            assert mg["132"]["class_type"] == "MiniMaxH3EncodeReferences" and mg["132"]["inputs"]["single_frame"] is True
            assert mg["26"]["class_type"] == "MiniMaxH3Preflight"
            assert "overall_soundscape:" in m_prev["edited_prompt"].lower()
        else:
            assert "overall_soundscape:" not in m_prev["edited_prompt"].lower()

    return "all 4 spatial modes, zero-tortoise synthesis, masked image previews, dual VHS save_output=False, dynamic '165' preview_fps=7/omission, '51'/'70' omission, audio_refine + redub, baseline opt-in, and image stills verified"


def check_forget_plan_and_isolation() -> str:
    _compiler_mod, routes_mod, _server_mod = _load_director_modules()
    assert routes_mod.APP_MARK == "h3_spatial_director", f"unexpected APP_MARK {routes_mod.APP_MARK!r}"
    assert routes_mod.UPLOADS == "h3_spatial_director", f"unexpected UPLOADS {routes_mod.UPLOADS!r}"

    history = {
        "dir_1": {
            "prompt": [
                1,
                "dir_1",
                {
                    "15": {"class_type": "LoadImage", "inputs": {"image": "h3_spatial_director/orphan.png"}},
                    "16": {"class_type": "LoadImage", "inputs": {"image": "h3_spatial_director/kept_open.png"}},
                },
                {"h3_spatial_director": True},
            ],
            "outputs": {
                "13": {"gifs": [{"filename": "hud_1.mp4", "subfolder": "", "type": "temp"}]},
                "162": {
                    "gifs": [
                        {"filename": "clean_1.mp4", "subfolder": "", "type": "temp"},
                        {"filename": "exported_keep.mp4", "subfolder": "Video", "type": "output"},
                    ]
                },
                "17": {"images": [{"filename": "anchor_still.png", "subfolder": "dir_sub", "type": "temp"}]},
                "18": {"audio": [{"filename": "foley_1.wav", "subfolder": "", "type": "temp"}]},
            },
        },
        "other_lane": {
            "prompt": [
                2,
                "other_lane",
                {"15": {"class_type": "LoadImage", "inputs": {"image": "h3_spatial_director/other_upload.png"}}},
                {"h3_image_app": True},
            ],
            "outputs": {
                "13": {"gifs": [{"filename": "other_lane.mp4", "subfolder": "", "type": "temp"}]},
            },
        },
    }
    running = [
        [
            3,
            "q_run",
            {"15": {"class_type": "LoadImage", "inputs": {"image": "h3_spatial_director/queued.png"}}},
            {"h3_spatial_director": True},
        ]
    ]

    with tempfile.TemporaryDirectory(prefix="h3_dir_forget_") as tmp:
        tmp_root = Path(tmp) / "temp"
        in_root = Path(tmp) / "input"
        out_root = Path(tmp) / "out_root"
        out_video_dir = out_root / "Video"
        (in_root / "h3_spatial_director").mkdir(parents=True)
        (tmp_root / "dir_sub").mkdir(parents=True)
        out_video_dir.mkdir(parents=True)

        for rel in ("hud_1.mp4", "clean_1.mp4", "dir_sub/anchor_still.png", "foley_1.wav", "other_lane.mp4"):
            (tmp_root / rel).write_bytes(b"test")
        for up in ("orphan.png", "kept_open.png", "other_upload.png", "queued.png", "stray_orphan.png"):
            (in_root / "h3_spatial_director" / up).write_bytes(b"img")
        protected_export = out_video_dir / "exported_keep.mp4"
        protected_export.write_bytes(b"keep-export")

        plan = routes_mod.compute_forget_plan(
            history=history,
            queue_running=running,
            queue_pending=[],
            protected_uploads=["h3_spatial_director/kept_open.png"],
            temp_dir=tmp_root,
            input_dir=in_root,
            output_dir=out_root,
        )
        assert plan["history_ids_to_delete"] == ["dir_1"], f"expected only ['dir_1'], got {plan['history_ids_to_delete']}"
        temp_rels = {
            f"{e['subfolder']}/{e['filename']}" if e.get("subfolder") else e["filename"]
            for e in plan["temp_files_to_delete"]
        }
        assert {"hud_1.mp4", "clean_1.mp4", "dir_sub/anchor_still.png", "foley_1.wav"} <= temp_rels, temp_rels
        assert "Video/exported_keep.mp4" not in temp_rels
        assert "other_lane.mp4" not in temp_rels
        assert any(s.get("filename") == "exported_keep.mp4" for s in plan["skipped_non_temp"])
        assert "h3_spatial_director/stray_orphan.png" in plan["uploads_to_delete"]
        assert "h3_spatial_director/orphan.png" in plan["uploads_to_delete"]
        for prot in ("h3_spatial_director/kept_open.png", "h3_spatial_director/queued.png", "h3_spatial_director/other_upload.png"):
            assert prot in plan["protected_uploads"] and prot not in plan["uploads_to_delete"]

        routes_mod.delete_temp_files(plan, temp_dir=tmp_root, input_dir=in_root, output_dir=out_root)
        for gone in ("hud_1.mp4", "clean_1.mp4", "dir_sub/anchor_still.png", "foley_1.wav"):
            assert not (tmp_root / gone).exists(), f"{gone} should have been deleted"
        assert not (in_root / "h3_spatial_director" / "stray_orphan.png").exists()
        assert not (in_root / "h3_spatial_director" / "orphan.png").exists()
        for safe_up in ("kept_open.png", "queued.png", "other_upload.png"):
            assert (in_root / "h3_spatial_director" / safe_up).is_file(), f"{safe_up} should have been protected"
        assert (tmp_root / "other_lane.mp4").is_file()
        assert protected_export.is_file(), "$H3_OUTPUT_DIR/Video export must NEVER be touched"

        # When protected_uploads clears and queue drains, kept_open.png & queued.png become eligible
        plan2 = routes_mod.compute_forget_plan(
            history=history,
            queue_running=[],
            queue_pending=[],
            protected_uploads=[],
            temp_dir=tmp_root,
            input_dir=in_root,
            output_dir=out_root,
        )
        assert "h3_spatial_director/kept_open.png" in plan2["uploads_to_delete"]
        assert "h3_spatial_director/queued.png" in plan2["uploads_to_delete"]
        assert "h3_spatial_director/other_upload.png" not in plan2["uploads_to_delete"]
        routes_mod.delete_temp_files(plan2, temp_dir=tmp_root, input_dir=in_root, output_dir=out_root)
        assert not (in_root / "h3_spatial_director" / "kept_open.png").exists()
        assert not (in_root / "h3_spatial_director" / "queued.png").exists()
        assert (in_root / "h3_spatial_director" / "other_upload.png").is_file()
        assert protected_export.is_file()

    return "compute_forget_plan & delete_temp_files clean temp + orphans while protecting output/Video/, queue, and keep_uploads"


def check_server_routes_and_escalate() -> str:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    import h3_rules

    _compiler_mod, _routes_mod, server_mod = _load_director_modules()

    async def _run() -> None:
        stored_uploads: dict[str, bytes] = {
            "test_anchor.png": _encode_test_png_bytes(336, 192),
            "shot_src.mp4": _encode_indexed_mp4_bytes([(230, 30, 30), (30, 220, 40), (30, 70, 235)]),
        }
        submitted_prompts: list[dict[str, Any]] = []

        async def _stub_upload(req: web.Request) -> web.Response:
            reader = await req.multipart()
            fname = "up.png"
            sub = "h3_spatial_director"
            fbytes = b""
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "image":
                    fname = part.filename or fname
                    fbytes = await part.read()
                elif part.name == "subfolder":
                    sub = (await part.text()).strip()
            key = f"{sub}/{fname}" if sub else fname
            stored_uploads[key] = fbytes
            return web.json_response({"name": fname, "subfolder": sub, "type": "input"})

        async def _stub_view(req: web.Request) -> web.Response:
            q = req.rel_url.query
            fn = q.get("filename", "")
            sub = q.get("subfolder", "")
            key = f"{sub}/{fn}" if sub else fn
            raw = stored_uploads.get(key) or stored_uploads.get(fn)
            if raw is None:
                return web.Response(status=404, text="not found")
            ctype = "video/mp4" if fn.endswith(".mp4") else "image/png"
            return web.Response(body=raw, content_type=ctype)

        async def _stub_prompt(req: web.Request) -> web.Response:
            body = await req.json()
            submitted_prompts.append(body)
            return web.json_response({"prompt_id": f"stub_redub_pid_{len(submitted_prompts)}"})

        stub = web.Application()
        stub.router.add_get("/object_info", lambda _r: web.json_response(MOCK_OBJECT_INFO_WITH_KJ))
        stub.router.add_get("/system_stats", lambda _r: web.json_response({"system": {"ram_total": 64}, "devices": []}))
        stub.router.add_get("/queue", lambda _r: web.json_response({"queue_running": [], "queue_pending": []}))
        stub.router.add_get("/history", lambda _r: web.json_response({}))
        stub.router.add_post("/upload/image", _stub_upload)
        stub.router.add_get("/view", _stub_view)
        stub.router.add_post("/prompt", _stub_prompt)

        async with TestServer(stub) as comfy_srv:
            comfy_url = f"http://127.0.0.1:{comfy_srv.port}"
            with tempfile.TemporaryDirectory(prefix="h3_dir_srv_") as tmp_dir:
                app = _create_director_app(server_mod, comfy_url=comfy_url, state_dir=Path(tmp_dir))
                async with TestClient(TestServer(app)) as client:
                    # 1. Security headers + Host guard
                    r_home = await client.get("/")
                    assert r_home.status == 200
                    csp = r_home.headers.get("Content-Security-Policy", "")
                    assert "media-src 'self' blob: data:" in csp, f"CSP missing blob: media-src: {csp!r}"
                    assert "style-src 'self'" in csp and "script-src 'self'" in csp
                    r_evil = await client.get("/", headers={"Host": "evil.example.com"})
                    assert r_evil.status == 403, f"expected 403 on evil Host, got {r_evil.status}"

                    # 2. Schema hydration + capability detection
                    r_schema = await client.get("/api/schema")
                    assert r_schema.status == 200
                    schema = await r_schema.json()
                    assert schema["capabilities"]["kj_preview_override"] is True
                    assert schema["capabilities"]["taeh3"] is True
                    assert schema["capabilities"]["audio_refine"] is True
                    assert all(h3_rules.on_both_clocks(n) for n in schema["lengths"]), schema["lengths"]

                    # 3. Context suggest disabled cleanly without H3_HEYLOOK_URL
                    keep_heylook = os.environ.pop("H3_HEYLOOK_URL", None)
                    try:
                        r_sug = await client.post("/api/context_suggest", json={"prompt": "archway"})
                        assert r_sug.status == 200
                        sug = await r_sug.json()
                        assert sug == {"ok": True, "enabled": False, "suggestions": []}, sug
                    finally:
                        if keep_heylook is not None:
                            os.environ["H3_HEYLOOK_URL"] = keep_heylook

                    # 4. Escalate scaling across on_both_clocks + 3f sound cues
                    for src_len, dst_len in ((90, 141), (141, 192), (141, 90)):
                        ev = json.loads(json.dumps(SAMPLE_EVENTS_90))
                        if src_len != 90:
                            scale_f = (src_len - 1) / 89.0
                            for p in ev["portal"]:
                                p["frame"] = int(round(p["frame"] * scale_f))
                            for t in ev["targets"]:
                                t["start"] = int(round(t["start"] * scale_f))
                                t["end"] = max(t["start"] + 3, int(round(t["end"] * scale_f)))
                            for pc in ev["portal_cues"]:
                                pc["frame"] = int(round(pc["frame"] * scale_f))
                            for wp in ev["dpad"]["waypoints"]:
                                wp["frame"] = int(round(wp["frame"] * scale_f))
                            for hp in ev["dpad"]["hops"]:
                                hp["start"] = int(round(hp["start"] * scale_f))
                                hp["end"] = max(hp["start"] + 3, int(round(hp["end"] * scale_f)))
                        r_esc = await client.post(
                            "/api/escalate",
                            json={
                                "length": src_len,
                                "target_length": dst_len,
                                "events_json": json.dumps(ev),
                                "portal_cues": ev["portal_cues"],
                            },
                        )
                        assert r_esc.status == 200, await r_esc.text()
                        esc = await r_esc.json()
                        assert esc["ok"] is True
                        out_len = int(esc["target_length"])
                        assert out_len == dst_len and h3_rules.on_both_clocks(out_len), esc
                        out_ev = esc["events_json"]
                        if isinstance(out_ev, str):
                            out_ev = json.loads(out_ev)
                        assert out_ev["portal"][0]["frame"] == 0
                        assert out_ev["portal"][-1]["frame"] == out_len - 1
                        for t in out_ev.get("targets", []):
                            assert 0 <= t["start"] < t["end"] <= out_len - 1
                            if t.get("cue", "none") != "none":
                                assert t["start"] % 3 == 0, f"target start {t['start']} not 3f-snapped"
                        for pc in out_ev.get("portal_cues", []) + (esc.get("portal_cues") or []):
                            assert 0 <= pc["frame"] <= out_len - 1
                            assert pc["frame"] % 3 == 0, f"portal_cue frame {pc['frame']} not 3f-snapped"

                    # 5. POST /api/compile through server (including Halo Coupling summary)
                    r_cmp = await client.post(
                        "/api/compile",
                        json={
                            "mode": "r2v_portal",
                            "source_ref": "h3_spatial_director/test_anchor.png",
                            "sprite_ref": "h3_spatial_director/cutout_1.png",
                            "length": 141,
                            "width": 1344,
                            "height": 768,
                            "seed": 42,
                            "halo_radius": 0.22,
                            "couplings": [
                                {
                                    "portal_index": 0,
                                    "target_kind": "sprite",
                                    "strength": 0.8,
                                    "mode": "push",
                                    "reaction_clause": "radial breeze nudges subject right",
                                }
                            ],
                            "target_prompt": "Glowing bioluminescent frost portal inside stone courtyard",
                            "events_json": json.dumps(SAMPLE_EVENTS_90),
                        },
                    )
                    assert r_cmp.status == 200, await r_cmp.text()
                    cmp_data = await r_cmp.json()
                    assert cmp_data["ok"] is True and cmp_data["preview_override_injected"] is True
                    assert cmp_data["graph"]["165"]["inputs"]["preview_frames"] == 27
                    assert cmp_data["graph"]["165"]["inputs"]["preview_fps"] == 5
                    assert float(cmp_data["graph"]["160"]["inputs"]["feather"]) >= 0.22
                    assert len(cmp_data["nodes_summary"].get("couplings_applied") or []) >= 1, cmp_data["nodes_summary"]

                    # 6. POST /api/segment offline GrabCut path (cutout + exclude roles)
                    r_seg_cut = await client.post(
                        "/api/segment",
                        json={
                            "source_ref": "test_anchor.png",
                            "positive_points": [{"nx": 0.5, "ny": 0.5}],
                            "role": "cutout",
                        },
                    )
                    assert r_seg_cut.status == 200, await r_seg_cut.text()
                    seg_cut = await r_seg_cut.json()
                    assert seg_cut["ok"] is True and seg_cut["role"] == "cutout" and seg_cut["engine"] == "grabcut", seg_cut
                    assert seg_cut.get("mask_ref") and seg_cut.get("cutout_ref"), seg_cut
                    assert 0.01 <= float(seg_cut["coverage"]) <= 0.95, seg_cut

                    r_seg_exc = await client.post(
                        "/api/segment",
                        json={
                            "source_ref": "test_anchor.png",
                            "positive_points": [{"nx": 0.5, "ny": 0.5}],
                            "role": "exclude",
                        },
                    )
                    assert r_seg_exc.status == 200, await r_seg_exc.text()
                    seg_exc = await r_seg_exc.json()
                    assert seg_exc["ok"] is True and seg_exc["role"] == "exclude" and seg_exc.get("mask_ref"), seg_exc

                    # 7. Slice 2C: POST /api/extract_frame & POST /api/redub
                    from PIL import Image
                    import numpy as np

                    r_ext_tail = await client.post(
                        "/api/extract_frame",
                        json={"video": {"filename": "shot_src.mp4", "subfolder": "", "type": "temp"}, "frame_index": -1},
                    )
                    assert r_ext_tail.status == 200, await r_ext_tail.text()
                    ext_tail = await r_ext_tail.json()
                    assert ext_tail["ok"] is True and ext_tail["total_frames"] == 3 and ext_tail["frame_index"] == 2, ext_tail
                    assert ext_tail["ref"] in stored_uploads
                    with Image.open(io.BytesIO(stored_uploads[ext_tail["ref"]])) as tail_im:
                        tail_rgb = np.asarray(tail_im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
                    assert tail_rgb[2] > tail_rgb[0] + 80, f"expected blue tail frame, got RGB={tail_rgb}"

                    r_ext_0 = await client.post(
                        "/api/extract_frame",
                        json={"video": {"filename": "shot_src.mp4", "subfolder": "", "type": "temp"}, "frame_index": 0},
                    )
                    assert r_ext_0.status == 200, await r_ext_0.text()
                    ext_0 = await r_ext_0.json()
                    assert ext_0["ok"] is True and ext_0["frame_index"] == 0, ext_0
                    with Image.open(io.BytesIO(stored_uploads[ext_0["ref"]])) as f0_im:
                        f0_rgb = np.asarray(f0_im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
                    assert f0_rgb[0] > f0_rgb[2] + 80, f"expected red f0 frame, got RGB={f0_rgb}"

                    r_rd_dry = await client.post(
                        "/api/redub",
                        json={
                            "video_ref": "h3_spatial_director/shot_src.mp4",
                            "length": 90,
                            "redub_mode": "polish",
                            "dry_run": True,
                        },
                    )
                    assert r_rd_dry.status == 200, await r_rd_dry.text()
                    rd_dry = await r_rd_dry.json()
                    assert rd_dry["ok"] is True and rd_dry["dry_run"] is True and rd_dry["redub_mode"] == "polish"
                    assert rd_dry["graph"]["88"]["class_type"] == "MiniMaxH3FrozenVideoCache"
                    assert rd_dry["graph"]["84"]["class_type"] == "BasicScheduler"

                    r_rd_live = await client.post(
                        "/api/redub",
                        json={
                            "video": {"filename": "shot_src.mp4", "subfolder": "", "type": "temp"},
                            "length": 90,
                            "redub_mode": "polish",
                            "cues_text": "12: slam @1.0",
                        },
                    )
                    assert r_rd_live.status == 200, await r_rd_live.text()
                    rd_live = await r_rd_live.json()
                    # Silent shot_src.mp4 auto-promotes Polish -> Fresh (ManualSigmas + LTXVSeparateAVLatent)
                    assert rd_live["ok"] is True and rd_live["kind"] == "redub" and rd_live["redub_mode"] == "fresh", rd_live
                    assert rd_live["prompt_id"] == "stub_redub_pid_1"
                    assert rd_live["video_ref"].startswith("h3_spatial_director/shot_src_") and rd_live["video_ref"].endswith(".mp4"), rd_live
                    assert rd_live["video_ref"] in stored_uploads, stored_uploads.keys()
                    assert len(submitted_prompts) == 1

                    # 8. MP4 upload registers both Frame 0 anchor and playable "imported" take
                    from aiohttp import FormData

                    mp4_bytes = stored_uploads["shot_src.mp4"]
                    fd = FormData()
                    fd.add_field("file", mp4_bytes, filename="demo_drop.mp4", content_type="video/mp4")
                    r_up_mp4 = await client.post("/api/upload", data=fd)
                    assert r_up_mp4.status == 200, await r_up_mp4.text()
                    up_mp4 = await r_up_mp4.json()
                    assert up_mp4["ok"] is True and up_mp4["source_kind"] == "video_frame0"
                    assert up_mp4.get("imported_take_id") and up_mp4.get("video_ref") in stored_uploads, up_mp4
                    assert up_mp4.get("total_frames") == 3 and up_mp4.get("has_audio") is False, up_mp4

                    r_takes = await client.get("/api/takes")
                    takes_doc = await r_takes.json()
                    assert takes_doc.get("queue_ok") is True and isinstance(takes_doc.get("active_prompt_ids"), list), takes_doc
                    assert any(t.get("take_id") == rd_live["take_id"] and t.get("kind") == "redub" for t in takes_doc["takes"])
                    assert any(t.get("take_id") == up_mp4["imported_take_id"] and t.get("kind") == "imported" for t in takes_doc["takes"])

                    # 9. POST /api/preview_in_mask rasterizes circle PNG mask -> queues EditGuider + taeh3_image
                    n_sub_before = len(submitted_prompts)
                    r_mpv = await client.post(
                        "/api/preview_in_mask",
                        json={
                            "source_ref": "test_anchor.png",
                            "target_prompt": "a swirling azure vortex",
                            "region": {"x": 0.5, "y": 0.5, "rx": 0.22, "ry": 0.22, "shape": "circle"},
                            "feather": 0.08,
                            "seed": 55,
                        },
                    )
                    assert r_mpv.status == 200, await r_mpv.text()
                    mpv = await r_mpv.json()
                    assert mpv["ok"] is True and mpv["kind"] == "mask_preview"
                    assert mpv["mask_ref"] in stored_uploads, mpv
                    assert len(submitted_prompts) == n_sub_before + 1
                    mpv_graph = submitted_prompts[-1]["prompt"]
                    assert mpv_graph["9"]["class_type"] == "MiniMaxH3EditGuider"
                    assert mpv_graph["165"]["inputs"]["tiny_vae"] == "taeh3_image.safetensors"

                    # 10. POST /api/stamp_cutout with BOTH source_ref AND prompt triggers T2I for_cutout=True
                    r_stp = await client.post(
                        "/api/stamp_cutout",
                        json={
                            "source_ref": "test_anchor.png",
                            "prompt": "a small clockwork brass owl",
                            "seed": 77,
                        },
                    )
                    assert r_stp.status == 200, await r_stp.text()
                    stp = await r_stp.json()
                    assert stp["ok"] is True and stp.get("prompt_id"), f"Expected T2I cutout queue when prompt provided, got {stp}"

                    # 11. POST /api/context_suggest uses heylook POST /v1/messages (never /v1/chat/completions)
                    seen_heylook: dict[str, Any] = {}

                    async def _hl_models(_r: web.Request) -> web.Response:
                        return web.json_response({"data": [{"id": "qwen-vision", "capabilities": ["chat", "vision"]}]})

                    async def _hl_messages(req: web.Request) -> web.Response:
                        seen_heylook["req_id"] = req.headers.get("X-Request-ID")
                        seen_heylook["body"] = await req.json()
                        return web.json_response(
                            {
                                "content": [
                                    {"type": "thinking", "thinking": "ignoring chain of thought"},
                                    {
                                        "type": "text",
                                        "text": json.dumps(
                                            [{"title": "Frost Bloom", "target_prompt": "frost petals expand rapidly", "preset": "sparkle_burst"}]
                                        ),
                                    },
                                ]
                            }
                        )

                    hl_stub = web.Application()
                    hl_stub.router.add_get("/v1/models", _hl_models)
                    hl_stub.router.add_post("/v1/messages", _hl_messages)
                    hl_stub.router.add_post("/v1/chat/completions", lambda _r: web.Response(status=404, text="removed"))
                    async with TestServer(hl_stub) as hl_srv:
                        prev_hl = os.environ.get("H3_HEYLOOK_URL")
                        os.environ["H3_HEYLOOK_URL"] = f"http://127.0.0.1:{hl_srv.port}"
                        try:
                            r_hl = await client.post(
                                "/api/context_suggest",
                                json={"source_ref": "test_anchor.png", "original_prompt": "misty lake", "prompt": ""},
                            )
                            assert r_hl.status == 200, await r_hl.text()
                            hl_doc = await r_hl.json()
                            assert hl_doc["ok"] is True and hl_doc["enabled"] is True, hl_doc
                            assert hl_doc["suggestions"] and hl_doc["suggestions"][0]["title"] == "Frost Bloom", hl_doc
                            assert seen_heylook.get("req_id"), "Expected X-Request-ID on /v1/messages"
                            blocks = seen_heylook["body"]["messages"][0]["content"]
                            assert any(isinstance(b, dict) and b.get("type") == "image" for b in blocks), blocks
                        finally:
                            if prev_hl is None:
                                os.environ.pop("H3_HEYLOOK_URL", None)
                            else:
                                os.environ["H3_HEYLOOK_URL"] = prev_hl

                    # 12. GET /docs, /api/docs, /api/docs/file, and security confinement
                    r_docs_html = await client.get("/docs")
                    assert r_docs_html.status == 200 and "text/html" in r_docs_html.headers.get("Content-Type", "")
                    assert "Content-Security-Policy" in r_docs_html.headers

                    r_docs_cat = await client.get("/api/docs")
                    assert r_docs_cat.status == 200, await r_docs_cat.text()
                    docs_cat = await r_docs_cat.json()
                    assert docs_cat["ok"] is True and docs_cat["default_doc"] == "docs/wiki/spatial_director_guide.md"
                    assert docs_cat["groups"] and docs_cat["items"], docs_cat

                    r_docs_q = await client.get("/api/docs", params={"q": "Preview in Circle"})
                    assert r_docs_q.status == 200
                    docs_q = await r_docs_q.json()
                    assert any(r.get("path") == "docs/wiki/spatial_director_guide.md" for r in docs_q.get("results", [])), docs_q

                    r_doc_md = await client.get("/api/docs/file", params={"path": "docs/wiki/spatial_director_guide.md"})
                    assert r_doc_md.status == 200, await r_doc_md.text()
                    doc_md = await r_doc_md.json()
                    assert doc_md["ok"] is True and "Progressive 6-Level Tutorial" in doc_md["markdown"]

                    for bad_doc in ("../main.py", "h3.env", "director/server.py"):
                        r_bad_doc = await client.get("/api/docs/file", params={"path": bad_doc})
                        assert r_bad_doc.status == 400, f"expected 400 on {bad_doc!r}, got {r_bad_doc.status}"

    asyncio.run(_run())
    return "server CSP/Host guard, schema, /docs hub, /api/preview_in_mask, T2I /api/stamp_cutout, heylook /v1/messages, /api/segment, /api/extract_frame, /api/redub, /api/escalate, and /api/compile verified"


def check_static_and_route_audit() -> str:
    import sound_cues
    import h3_rules
    import check_no_owner_paths

    _compiler_mod, _routes_mod, server_mod = _load_director_modules()
    static_dir = REPO / "director" / "static"
    html_files = sorted(static_dir.rglob("*.html"))
    assert html_files, "no HTML files found under director/static/"
    css_files = sorted(static_dir.rglob("*.css"))
    svg_files = sorted(static_dir.rglob("*.svg"))
    js_files = sorted((static_dir / "js").rglob("*.js"))
    assert js_files, "no JS files found under director/static/js/"

    errors: list[str] = []
    allowed_xmlns = re.compile(r'xmlns(?::xlink)?=["\']http://www\.w3\.org/(?:2000/svg|1999/xlink)["\']')
    url_re = re.compile(r"(?:https?:)?//[^\s'\"`<>)]+", re.IGNORECASE)

    for f in [*html_files, *css_files, *svg_files, *js_files]:
        rel = f.relative_to(REPO).as_posix()
        txt = f.read_text(encoding="utf-8")
        cleaned = allowed_xmlns.sub("", txt)
        cleaned_no_comments = re.sub(r"//[^\n]*", "", re.sub(r"/\*[\s\S]*?\*/", "", cleaned))
        for m in url_re.finditer(cleaned_no_comments):
            errors.append(f"{rel}: external URL forbidden ({m.group(0)[:60]})")

    for hf in html_files:
        hrel = hf.relative_to(REPO).as_posix()
        htxt = hf.read_text(encoding="utf-8")
        if re.search(r"<style\b", htxt, re.IGNORECASE):
            errors.append(f"{hrel}: inline <style> tag forbidden")
        if re.search(r"\bstyle\s*=", htxt, re.IGNORECASE):
            errors.append(f"{hrel}: inline style= attribute forbidden")
        if re.search(r"\bon(?:click|change|input|submit|load|error|keydown|keyup|mousedown|mouseup|mousemove|wheel)\s*=", htxt, re.IGNORECASE):
            errors.append(f"{hrel}: inline on*= handler forbidden")
        if re.search(r"<option\b[^>]*value=['\"](?:slam|hop|snap|click|90|141)['\"]", htxt, re.IGNORECASE):
            errors.append(f"{hrel}: hardcoded preset/length <option> forbidden")

    forbidden_suggestions = (
        "Open swirling sea vortex",
        "Golden sunbeams",
        "Shatter keystone into spray",
        "SCENE-AWARE SUGGESTIONS",
    )
    fake_physics_patterns = (
        re.compile(r"\b1\.42\s*N\b"),
        re.compile(r"\bF\s*=\s*\d+(?:\.\d+)?\s*N\b"),
        re.compile(r"[+-]?\d+(?:\.\d+)?\s*m/s\b"),
        re.compile(r"Fluid\s+simulation", re.IGNORECASE),
        re.compile(r"~\s*2\s*s\b"),
    )
    for f in [*html_files, *js_files]:
        rel = f.relative_to(REPO).as_posix()
        raw = f.read_text(encoding="utf-8")
        no_comments = re.sub(r"//[^\n]*", "", re.sub(r"/\*[\s\S]*?\*/", "", raw))
        for pat in fake_physics_patterns:
            m_fp = pat.search(no_comments)
            if m_fp:
                errors.append(f"{rel}: fake Newtonian/fluid physics string {m_fp.group(0)!r} forbidden")
    for hf in html_files:
        hrel = hf.relative_to(REPO).as_posix()
        htxt = hf.read_text(encoding="utf-8")
        for phrase in forbidden_suggestions:
            if phrase in htxt:
                errors.append(f"{hrel}: hardcoded Stitch scene suggestion {phrase!r} forbidden")
    presets = sorted(sound_cues.SOUND_CUE_PRESETS.keys())
    clock_strs = [str(n) for n in h3_rules.chain_lengths()] + ["39", "90"]

    called_api_routes: set[str] = set()
    for js in js_files:
        rel = js.relative_to(REPO).as_posix()
        raw = js.read_text(encoding="utf-8")
        code = re.sub(r"//[^\n]*", "", re.sub(r"/\*[\s\S]*?\*/", "", raw))

        if re.search(r"\.style\s*(?:\.|\[)", code) or re.search(r"setAttribute\(\s*['\"]style['\"]", code):
            errors.append(f"{rel}: element.style mutation forbidden")
        if re.search(r"\bstyle\s*=", code) or re.search(r"\bon(?:click|change|input|submit)\s*=", code):
            errors.append(f"{rel}: inline style= or onclick= inside JS string forbidden")
        for phrase in forbidden_suggestions:
            if phrase in raw:
                errors.append(f"{rel}: hardcoded Stitch scene suggestion {phrase!r} forbidden")

        # No hardcoded preset or clock arrays `[...]` in any JS module
        for arr_m in re.finditer(r"\[([^\]]{1,400})\]", code):
            body = arr_m.group(1)
            preset_hits = [p for p in presets if re.search(rf"['\"`]{re.escape(p)}['\"`]", body)]
            if len(preset_hits) >= 2:
                errors.append(f"{rel}: hardcoded SOUND_CUE_PRESETS array {preset_hits}")
            clock_hits = [c for c in clock_strs if re.search(rf"(?<![0-9]){re.escape(c)}(?![0-9])", body)]
            if len(clock_hits) >= 3:
                errors.append(f"{rel}: hardcoded clock-length array {clock_hits}")

        # Non-synth JS modules must not even list multiple sound presets as literals
        if js.name != "foley_synth.js":
            hits = [p for p in presets if re.search(rf"['\"`]{re.escape(p)}['\"`]", code)]
            if len(hits) >= 2:
                errors.append(f"{rel}: literal sound preset names {hits} must come from /api/schema")

        # Relative ES module imports must resolve
        for imp_m in re.finditer(r"from\s+['\"](\.{1,2}/[^'\"]+)['\"]", code):
            target = (js.parent / imp_m.group(1)).resolve()
            if not target.is_file():
                errors.append(f"{rel}: broken ES import {imp_m.group(1)}")

        for api_m in re.finditer(r"['\"`](/api/[A-Za-z0-9_/\-]+)", code):
            called_api_routes.add(api_m.group(1))

    assert called_api_routes, "expected JS files to call /api/... endpoints"

    with tempfile.TemporaryDirectory(prefix="h3_dir_routes_") as tmp_dir:
        app = _create_director_app(server_mod, comfy_url="http://127.0.0.1:1", state_dir=Path(tmp_dir))
        registered_routes: set[str] = set()
        for route in app.router.routes():
            info = route.get_info()
            path = info.get("path") or info.get("formatter")
            if isinstance(path, str) and path.startswith("/api/"):
                registered_routes.add(path)
        missing = sorted(called_api_routes - registered_routes)
        if missing:
            errors.append(f"JS calls /api routes not registered in director/server.py: {missing}")

    # Zero owner paths or hardcoded tortoise/turtle strings across director/
    director_files = [p for p in sorted((REPO / "director").rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    for path, rel in check_no_owner_paths.text_files(REPO, director_files):
        data = path.read_bytes()
        for lineno, line in enumerate(data.splitlines(), 1):
            if any(pat.search(line) for pat in check_no_owner_paths.PATTERNS):
                errors.append(f"{rel}:{lineno}: owner path stored in director/")
            low_line = line.decode("utf-8", errors="ignore").lower()
            if "tortoise" in low_line or "turtle" in low_line:
                errors.append(f"{rel}:{lineno}: hardcoded tortoise/turtle text forbidden in director/")

    assert not errors, "\n  ".join(errors)
    return f"{len(js_files)} JS files audited, {len(called_api_routes)} /api routes verified, 0 CSP/inline/preset/path violations"


# ---------------------------------------------------------------------------
# Tier 2: Isolated --cpu ComfyUI (:8288) + Director (:8295) + Chrome CDP
# ---------------------------------------------------------------------------

def _wait_http_json(
    url: str,
    timeout_s: float = 45.0,
    proc: subprocess.Popen | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log_path and log_path.is_file():
                tail = "\n" + "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
            raise RuntimeError(f"Process exited prematurely with rc={proc.returncode} while waiting for {url}{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.25)
    tail = ""
    if log_path and log_path.is_file():
        tail = "\n" + "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}{tail}")


def _terminate_proc(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        if pgid == proc.pid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=6.0)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            pass


async def _run_cdp_browser(director_url: str, anchor_png: Path) -> str:
    import aiohttp

    chrome_bin = _find_chrome()
    assert chrome_bin, "google-chrome not found"

    cdp_port = _free_port()
    with tempfile.TemporaryDirectory(prefix="h3_dir_cdp_", ignore_cleanup_errors=True) as user_data:
        chrome_proc = subprocess.Popen(
            [
                chrome_bin,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--window-size=1280,800",
                f"--remote-debugging-port={cdp_port}",
                f"--user-data-dir={user_data}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            ws_url = None
            for _ in range(50):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=1.0) as r:
                        for target in json.loads(r.read().decode("utf-8")):
                            if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                                ws_url = target["webSocketDebuggerUrl"]
                                break
                    if ws_url:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.15)
            assert ws_url, "Failed to obtain Chrome CDP WebSocket URL"

            console_errors: list[str] = []
            runtime_exceptions: list[str] = []
            msg_id = 0

            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(ws_url, max_msg_size=16 * 1024 * 1024) as ws:
                    async def cdp(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
                        nonlocal msg_id
                        msg_id += 1
                        cid = msg_id
                        await ws.send_json({"id": cid, "method": method, "params": params or {}})
                        while True:
                            msg = await ws.receive()
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                raise RuntimeError(f"CDP websocket closed: {msg}")
                            data = json.loads(msg.data)
                            if "method" in data:
                                m = data["method"]
                                p = data.get("params") or {}
                                if m == "Runtime.exceptionThrown":
                                    runtime_exceptions.append(json.dumps(p))
                                elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
                                    console_errors.append(json.dumps(p.get("args")))
                                elif m == "Log.entryAdded":
                                    entry = p.get("entry") or {}
                                    if entry.get("level") == "error" and "favicon" not in str(entry.get("url", "")):
                                        console_errors.append(json.dumps(entry))
                                continue
                            if data.get("id") == cid:
                                if "error" in data:
                                    raise RuntimeError(f"CDP {method} failed: {data['error']}")
                                return data.get("result") or {}

                    async def js(expr: str) -> Any:
                        res = await cdp(
                            "Runtime.evaluate",
                            {"expression": expr, "awaitPromise": True, "returnByValue": True},
                        )
                        if "exceptionDetails" in res:
                            raise RuntimeError(f"JS exception: {res['exceptionDetails']}")
                        return (res.get("result") or {}).get("value")

                    await cdp("Page.enable")
                    await cdp("Runtime.enable")
                    await cdp("Log.enable")
                    await cdp("DOM.enable")
                    await cdp(
                        "Emulation.setDeviceMetricsOverride",
                        {"width": 1280, "height": 800, "deviceScaleFactor": 1, "mobile": False},
                    )
                    await cdp(
                        "Page.addScriptToEvaluateOnNewDocument",
                        {
                            "source": """
                            (() => {
                              window.__cspViolations = [];
                              document.addEventListener('securitypolicyviolation', (e) => {
                                window.__cspViolations.push({
                                  violatedDirective: e.violatedDirective,
                                  effectiveDirective: e.effectiveDirective,
                                  blockedURI: e.blockedURI,
                                  sourceFile: e.sourceFile,
                                  lineNumber: e.lineNumber
                                });
                              });
                              window.__directorSockets = [];
                              const OrigWS = window.WebSocket;
                              window.WebSocket = function(url, protocols) {
                                const inst = protocols ? new OrigWS(url, protocols) : new OrigWS(url);
                                window.__directorSockets.push(inst);
                                return inst;
                              };
                              window.WebSocket.prototype = OrigWS.prototype;
                              window.WebSocket.CONNECTING = OrigWS.CONNECTING;
                              window.WebSocket.OPEN = OrigWS.OPEN;
                              window.WebSocket.CLOSING = OrigWS.CLOSING;
                              window.WebSocket.CLOSED = OrigWS.CLOSED;
                            })();
                            """
                        },
                    )

                    await cdp("Page.navigate", {"url": director_url})
                    await cdp("Page.bringToFront")

                    # Wait for schema hydration + WS open
                    hydrated = False
                    for _ in range(50):
                        hydrated = bool(
                            await js(
                                "document.readyState === 'complete' && "
                                "document.querySelector('#stage-canvas') !== null && "
                                "document.querySelectorAll('button').length >= 3"
                            )
                        )
                        if hydrated:
                            break
                        await asyncio.sleep(0.15)
                    assert hydrated, "Studio page failed to hydrate DOM"

                    # Verify /api/schema from live ComfyUI :8288
                    schema = await js("fetch('/api/schema').then(r => r.json())")
                    assert schema["ok"] is True
                    assert schema["capabilities"]["kj_preview_override"] is True, schema["capabilities"]
                    assert schema["capabilities"]["taeh3"] is True, schema["capabilities"]

                    # Verify Light Mode surface luminance > 220
                    luma = await js(
                        """(() => {
                          function lum(bg) {
                            const m = String(bg).match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                            if (!m) return -1;
                            const [r, g, b] = [+m[1], +m[2], +m[3]];
                            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
                          }
                          const bodyL = lum(getComputedStyle(document.body).backgroundColor);
                          const hdr = document.querySelector('header');
                          const hdrL = hdr ? lum(getComputedStyle(hdr).backgroundColor) : -1;
                          return {bodyL, hdrL};
                        })()"""
                    )
                    assert luma["bodyL"] > 220, f"body luminance {luma['bodyL']} <= 220 (expected Light Mode)"
                    assert luma["hdrL"] > 220, f"header luminance {luma['hdrL']} <= 220 (expected Light Mode)"

                    # Verify Render Portal Edit, Compare Original, and Engine fit inside 1280px & pill unclipped
                    layout = await js(
                        """(() => {
                          function findBtn(id, textSub) {
                            const byId = document.getElementById(id);
                            if (byId) return byId;
                            return [...document.querySelectorAll('button')].find(b => b.textContent.includes(textSub)) || null;
                          }
                          const rBtn = findBtn('btn-render', 'Render Portal Edit');
                          const cBtn = findBtn('btn-compare-ab', 'Compare Original');
                          const eBtn = findBtn('btn-engine', 'Engine');
                          const pills = [
                            document.getElementById('live-tae-pill'),
                            document.getElementById('portal-tae-pill'),
                          ].filter(Boolean);
                          let pillClipped = false;
                          for (const pill of pills) {
                            let cur = pill.parentElement;
                            while (cur) {
                              const cs = getComputedStyle(cur);
                              if (cs.overflow === 'hidden' && parseFloat(cs.borderTopLeftRadius || '0') > 30) {
                                pillClipped = true;
                              }
                              cur = cur.parentElement;
                            }
                          }
                          return {
                            scrollWidth: document.documentElement.scrollWidth,
                            innerWidth: window.innerWidth,
                            renderRect: rBtn ? rBtn.getBoundingClientRect().toJSON() : null,
                            compareRect: cBtn ? cBtn.getBoundingClientRect().toJSON() : null,
                            engineRect: eBtn ? eBtn.getBoundingClientRect().toJSON() : null,
                            pillClipped,
                          };
                        })()"""
                    )
                    assert layout["scrollWidth"] <= 1280, f"Horizontal overflow at 1280px: {layout['scrollWidth']}"
                    for name_key in ("renderRect", "compareRect", "engineRect"):
                        rect = layout[name_key]
                        assert rect and rect["width"] > 0 and rect["height"] > 0, f"{name_key} missing/zero: {rect}"
                        assert rect["left"] >= 0 and rect["right"] <= 1280, f"{name_key} clipped at 1280px: {rect}"
                    assert not layout["pillClipped"], "#live-tae-pill is trapped inside a circular overflow:hidden clip"

                    # Verify blob: media under CSP media-src does not emit securitypolicyviolation
                    mp4_b64 = _encode_solid_mp4_b64()
                    await js(
                        f"""new Promise((resolve) => {{
                          const bin = atob({json.dumps(mp4_b64)});
                          const arr = new Uint8Array(bin.length);
                          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                          const blobUrl = URL.createObjectURL(new Blob([arr], {{type: 'video/mp4'}}));
                          const v = document.createElement('video');
                          v.muted = true;
                          v.playsInline = true;
                          v.src = blobUrl;
                          v.load();
                          setTimeout(() => {{ URL.revokeObjectURL(blobUrl); resolve(true); }}, 150);
                        }})"""
                    )

                    # Upload anchor PNG via DOM.setFileInputFiles
                    doc = await cdp("DOM.getDocument", {"depth": -1})
                    inp = await cdp(
                        "DOM.querySelector",
                        {"nodeId": doc["root"]["nodeId"], "selector": "#upload-anchor-input, input[type='file']"},
                    )
                    assert inp.get("nodeId"), "No file upload input found (#upload-anchor-input)"
                    await cdp("DOM.setFileInputFiles", {"nodeId": inp["nodeId"], "files": [str(anchor_png)]})

                    # Wait briefly for /api/upload + canvas paint
                    uploaded = False
                    for _ in range(30):
                        uploaded = bool(
                            await js(
                                """(() => {
                                  const c = document.getElementById('stage-canvas');
                                  if (!c || c.width < 10 || c.height < 10) return false;
                                  const d = c.getContext('2d').getImageData(Math.floor(c.width/2), Math.floor(c.height/2), 1, 1).data;
                                  return (d[0] + d[1] + d[2]) > 0;
                                })()"""
                            )
                        )
                        if uploaded:
                            break
                        await asyncio.sleep(0.15)
                    assert uploaded, "Anchor PNG upload did not render onto #stage-canvas"

                    # Click center of #stage-canvas -> contextual popover opens
                    c_rect = await js("document.getElementById('stage-canvas').getBoundingClientRect().toJSON()")
                    cx = c_rect["left"] + c_rect["width"] * 0.5
                    cy = c_rect["top"] + c_rect["height"] * 0.5
                    await cdp("Input.dispatchMouseEvent", {"type": "mousePressed", "x": cx, "y": cy, "button": "left", "clickCount": 1})
                    await cdp("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": cx, "y": cy, "button": "left", "clickCount": 1})
                    await asyncio.sleep(0.15)

                    pop = await js(
                        """(() => {
                          const p = document.getElementById('portal-card');
                          const tp = document.getElementById('target-prompt');
                          const hd = document.getElementById('portal-card-heading');
                          const diffWrap = document.getElementById('prompt-diff-preview');
                          const diffEdit = document.getElementById('diff-edit-line');
                          if (!p) return null;
                          const r = p.getBoundingClientRect().toJSON();
                          if (tp) {
                            tp.value = 'Swirling bioluminescent cavern portal';
                            tp.dispatchEvent(new Event('input', {bubbles: true}));
                            tp.dispatchEvent(new Event('change', {bubbles: true}));
                          }
                          return {
                            rect: r,
                            visible: getComputedStyle(p).display !== 'none',
                            hasPrompt: !!tp,
                            cardScrollHeight: p.scrollHeight,
                            cardClientHeight: p.clientHeight,
                            headingText: hd ? hd.textContent : '',
                            headingScrollW: hd ? hd.scrollWidth : 0,
                            headingClientW: hd ? hd.clientWidth : 0,
                            diffVisible: diffWrap ? getComputedStyle(diffWrap).display !== 'none' : false,
                            diffEditText: diffEdit ? diffEdit.textContent : '',
                          };
                        })()"""
                    )
                    assert pop and pop["visible"] and pop["hasPrompt"], f"Contextual popover #portal-card/#target-prompt missing: {pop}"
                    assert pop["rect"]["right"] <= 1280 and pop["rect"]["bottom"] <= 800, f"Popover clipped: {pop['rect']}"
                    assert pop["headingScrollW"] <= pop["headingClientW"] + 1, f"#portal-card-heading truncated: {pop}"
                    assert pop["diffVisible"] and "Swirling bioluminescent cavern portal" in pop["diffEditText"], f"#prompt-diff-preview did not update: {pop}"

                    # Helper: intercept one /api/render click to read exact serialized frontend state without queueing
                    async def _capture_render_payload() -> dict[str, Any]:
                        await js(
                            """(() => {
                              window.__capturedRenderReq = null;
                              const origFetch = window.fetch;
                              window.fetch = async function(input, init) {
                                const url = typeof input === 'string' ? input : (input && input.url) || '';
                                if (url.includes('/api/render')) {
                                  window.__capturedRenderReq = JSON.parse((init && init.body) || '{}');
                                  window.fetch = origFetch;
                                  return new Response(JSON.stringify({ok: true, prompt_id: 'sim-cdp', take_id: 'sim'}), {
                                    status: 200,
                                    headers: {'Content-Type': 'application/json'}
                                  });
                                }
                                return origFetch.apply(this, arguments);
                              };
                              const btn = document.getElementById('btn-render');
                              if (btn) btn.click();
                            })()"""
                        )
                        for _ in range(20):
                            got = await js("window.__capturedRenderReq")
                            if got is not None:
                                await asyncio.sleep(0.12)
                                return got
                            await asyncio.sleep(0.05)
                        raise AssertionError("Clicking #btn-render did not trigger POST /api/render")

                    def _first_portal(req_payload: dict[str, Any]) -> dict[str, Any]:
                        ev = req_payload.get("events_json", "{}")
                        if isinstance(ev, str):
                            ev = json.loads(ev)
                        portals = ev.get("portal") or []
                        assert portals, f"No portal keyframes in serialized payload: {ev}"
                        return portals[0]

                    p0_req = await _capture_render_payload()
                    assert p0_req.get("mode") == "i2v_portal", (
                        f"Expected uploaded anchor without cutout to default to i2v_portal, got {p0_req.get('mode')!r}"
                    )
                    sandbox_dom = await js(
                        """(() => ({
                          hasPreviewBtn: Boolean(document.getElementById('btn-preview-mask')),
                          hasOrigPrompt: Boolean(document.getElementById('original-prompt')),
                          hasPrevSelect: Boolean(document.getElementById('preview-workflow-select')),
                          hasEventTrack: Boolean(document.getElementById('event-chips-track')),
                          hasStatusOverlay: Boolean(document.getElementById('render-status-overlay')),
                        }))()"""
                    )
                    assert all(sandbox_dom.values()), f"Missing World Sandbox DOM elements: {sandbox_dom}"
                    p0 = _first_portal(p0_req)
                    rx0, ry0 = float(p0["rx"]), float(p0["ry"])
                    halo0 = float(p0_req.get("halo_radius", p0.get("halo_radius", 0.0)))
                    scale0 = float(p0_req.get("scale", 1.5))

                    # Wheel -> rx/ry changes
                    await cdp("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": cx, "y": cy, "deltaX": 0, "deltaY": -180, "modifiers": 0})
                    await asyncio.sleep(0.1)
                    p1_req = await _capture_render_payload()
                    p1 = _first_portal(p1_req)
                    assert abs(float(p1["rx"]) - rx0) > 1e-4 or abs(float(p1["ry"]) - ry0) > 1e-4, (
                        f"Wheel did not adjust rx/ry: before=({rx0},{ry0}) after=({p1['rx']},{p1['ry']})"
                    )

                    # Shift+Wheel -> halo_radius changes (Shift=8)
                    await cdp("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": cx, "y": cy, "deltaX": 0, "deltaY": -180, "modifiers": 8})
                    await asyncio.sleep(0.1)
                    p2_req = await _capture_render_payload()
                    p2 = _first_portal(p2_req)
                    halo2 = float(p2_req.get("halo_radius", p2.get("halo_radius", 0.0)))
                    assert halo2 != halo0 and halo2 > 0.0, (
                        f"Shift+Wheel did not adjust halo_radius: before={halo0} after={halo2}"
                    )

                    # Alt+Wheel -> scale changes (Alt=1)
                    await cdp("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": cx, "y": cy, "deltaX": 0, "deltaY": -180, "modifiers": 1})
                    await asyncio.sleep(0.1)
                    p3_req = await _capture_render_payload()
                    scale3 = float(p3_req.get("scale", 1.5))
                    assert abs(scale3 - scale0) > 1e-4, f"Alt+Wheel did not adjust scale: before={scale0} after={scale3}"

                    # Switch to Combined mode & toggle Pick Subject bar (1280x800 layout unclipped)
                    comb_state = await js(
                        """(() => {
                          document.getElementById('mode-btn-combined')?.click();
                          const btnComb = document.getElementById('mode-btn-combined');
                          const heading = document.getElementById('portal-card-heading');
                          const card = document.getElementById('portal-card');
                          const pick = document.getElementById('btn-sam-pick');
                          pick?.click();
                          const canvas = document.getElementById('stage-canvas');
                          const armed = Boolean(pick?.classList.contains('active') && canvas?.classList.contains('sam-pick-cursor'));
                          pick?.click();
                          const r = card.getBoundingClientRect();
                          return {
                            combinedActive: Boolean(btnComb?.classList.contains('active')),
                            headingText: heading ? heading.textContent.trim() : '',
                            headingScrollW: heading ? Math.ceil(heading.scrollWidth) : 0,
                            headingClientW: heading ? Math.floor(heading.clientWidth) : 0,
                            rect: { left: r.left, top: r.top, right: r.right, bottom: r.bottom },
                            armed,
                          };
                        })()"""
                    )
                    assert comb_state["combinedActive"], f"#mode-btn-combined not active: {comb_state}"
                    assert comb_state["headingScrollW"] <= comb_state["headingClientW"] + 1, (
                        f"#portal-card-heading truncated in combined mode: {comb_state}"
                    )
                    assert comb_state["rect"]["right"] <= 1280 and comb_state["rect"]["bottom"] <= 800, (
                        f"#portal-card clipped at 1280x800 with SAM bar: {comb_state['rect']}"
                    )
                    assert comb_state["armed"], f"#btn-sam-pick did not arm canvas cursor: {comb_state}"

                    # Open 3-Lane Keyframe Sequencer, add multi-hop arcs, unlink ry, cycle Audio Guard
                    await js("document.getElementById('btn-lanes-toggle')?.click()")
                    await asyncio.sleep(0.1)
                    tray_box = await js(
                        """(() => {
                          const t = document.getElementById('sequencer-tray');
                          if (!t || t.classList.contains('hidden')) return null;
                          const r = t.getBoundingClientRect();
                          return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, height: r.height };
                        })()"""
                    )
                    assert tray_box and tray_box["left"] >= 0 and tray_box["right"] <= 1280 and tray_box["bottom"] <= 800, (
                        f"#sequencer-tray invalid bounds at 1280x800: {tray_box}"
                    )
                    await js("document.getElementById('btn-seq-add-hop')?.click()")
                    await cdp("Input.dispatchKeyEvent", {"type": "rawKeyDown", "key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39, "modifiers": 8})
                    await cdp("Input.dispatchKeyEvent", {"type": "keyUp", "key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39, "modifiers": 8})
                    await js("document.getElementById('btn-seq-add-hop')?.click()")
                    await js("document.getElementById('btn-seq-ry-unlink')?.click()")
                    await js("document.getElementById('btn-audio-guard')?.click()")
                    await asyncio.sleep(0.1)

                    shot_out = (os.environ.get("DIRECTOR_CDP_SCREENSHOT") or "").strip()
                    if shot_out:
                        shot_res = await cdp("Page.captureScreenshot", {"format": "png"})
                        Path(shot_out).write_bytes(base64.b64decode(shot_res["data"]))

                    seq_req = await _capture_render_payload()
                    seq_ev = json.loads(seq_req.get("events_json", "{}"))
                    hops = (seq_ev.get("dpad") or {}).get("hops") or []
                    assert len(hops) >= 2, f"Expected >=2 multi-hop arcs without overwrite, got {hops}"
                    assert seq_req.get("streams") in ("video_only", "both", "independent"), seq_req.get("streams")
                    assert any(
                        p.get("shape") == "ellipse" and p.get("unlink_ry") for p in (seq_ev.get("portal") or [])
                    ), seq_ev.get("portal")

                    # Delete the last selected hop (second hop) with #btn-seq-delete
                    await js("document.getElementById('btn-seq-delete')?.click()")
                    seq_req_after_del = await _capture_render_payload()
                    hops_after = (json.loads(seq_req_after_del.get("events_json", "{}")).get("dpad") or {}).get("hops") or []
                    assert len(hops_after) == len(hops) - 1, f"Delete failed: before={len(hops)} after={len(hops_after)}"

                    await js("document.getElementById('btn-seq-close')?.click()")

                    # Open Engine drawer and verify ArrowRight tablist navigation
                    await js("document.getElementById('btn-engine').click()")
                    drawer_open = False
                    for _ in range(20):
                        drawer_open = bool(
                            await js(
                                """(() => {
                                  const d = document.getElementById('engine-drawer');
                                  if (!d) return false;
                                  const r = d.getBoundingClientRect();
                                  return r.width > 0 && r.height > 0 && getComputedStyle(d).visibility !== 'hidden';
                                })()"""
                            )
                        )
                        if drawer_open:
                            break
                        await asyncio.sleep(0.05)
                    assert drawer_open, "#engine-drawer did not become visible on click"
                    tab_state_0 = await js(
                        """(() => {
                          const drawer = document.getElementById('engine-drawer');
                          if (!drawer) return null;
                          const tabs = [...drawer.querySelectorAll('[role="tablist"] [role="tab"]')];
                          if (tabs.length < 2) return {count: tabs.length};
                          tabs[0].click();
                          tabs[0].focus();
                          return {
                            count: tabs.length,
                            selectedBefore: tabs.map(t => t.getAttribute('aria-selected')),
                            activeIdx: tabs.indexOf(document.activeElement),
                          };
                        })()"""
                    )
                    assert tab_state_0 and tab_state_0["count"] >= 2 and tab_state_0["activeIdx"] == 0, (
                        f"Engine drawer tablist focus failed: {tab_state_0}"
                    )
                    await cdp("Input.dispatchKeyEvent", {"type": "rawKeyDown", "key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39})
                    await cdp("Input.dispatchKeyEvent", {"type": "keyUp", "key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39})
                    await asyncio.sleep(0.1)
                    tab_state_1 = await js(
                        """(() => {
                          const drawer = document.getElementById('engine-drawer');
                          const tabs = [...drawer.querySelectorAll('[role="tablist"] [role="tab"]')];
                          return {
                            selectedAfter: tabs.map(t => t.getAttribute('aria-selected')),
                            activeIdx: tabs.indexOf(document.activeElement),
                          };
                        })()"""
                    )
                    assert tab_state_1["activeIdx"] == 1 and tab_state_1["selectedAfter"][1] == "true", (
                        f"ArrowRight tablist navigation failed: {tab_state_1}"
                    )

                    # Switch back to Steering tab and verify #chk-audio-refine propagates audio_refine=True
                    await js(
                        """(() => {
                          const tabs = [...document.querySelectorAll('#engine-drawer [role="tablist"] [role="tab"]')];
                          if (tabs[0]) tabs[0].click();
                          document.getElementById('chk-audio-refine')?.click();
                        })()"""
                    )
                    ar_req = await _capture_render_payload()
                    assert ar_req.get("audio_refine") is True, f"expected audio_refine=True after checking #chk-audio-refine, got {ar_req.get('audio_refine')}"
                    await js("document.getElementById('chk-audio-refine')?.click()")
                    ar_req_off = await _capture_render_payload()
                    assert ar_req_off.get("audio_refine") is False, f"expected audio_refine=False after unchecking #chk-audio-refine, got {ar_req_off.get('audio_refine')}"

                    # Verify #btn-stage-mute, #sam-refine-bar, and #redub-mode-select controls
                    ui_controls = await js(
                        """(() => {
                          const muteBtn = document.getElementById('btn-stage-mute');
                          const beforeMute = muteBtn ? muteBtn.textContent.trim() : '';
                          muteBtn?.click();
                          const afterMute = muteBtn ? muteBtn.textContent.trim() : '';
                          muteBtn?.click();
                          const samBar = document.getElementById('sam-refine-bar');
                          const pick = document.getElementById('btn-sam-pick');
                          pick?.click();
                          const samBarVisible = Boolean(samBar && !samBar.classList.contains('hidden'));
                          const cardRect = document.getElementById('portal-card')?.getBoundingClientRect();
                          pick?.click();
                          const rdSel = document.getElementById('redub-mode-select');
                          return {
                            beforeMute,
                            afterMute,
                            samBarVisible,
                            cardInBounds: Boolean(cardRect && cardRect.right <= 1280 && cardRect.bottom <= 800),
                            redubOpts: rdSel ? [...rdSel.options].map(o => o.value) : [],
                          };
                        })()"""
                    )
                    assert ui_controls["beforeMute"] != ui_controls["afterMute"], ui_controls
                    assert ui_controls["samBarVisible"] is True and ui_controls["cardInBounds"] is True, ui_controls
                    assert ui_controls["redubOpts"] == ["polish", "fresh"], ui_controls

                    # POST /api/compile from page using edited payload -> emits "165" ModelPreviewOverrideKJ
                    compile_res = await js(
                        f"fetch('/api/compile', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, "
                        f"body: JSON.stringify({json.dumps(p3_req)})}}).then(r => r.json())"
                    )
                    assert compile_res["ok"] is True and compile_res["preview_override_injected"] is True, compile_res
                    assert compile_res["graph"]["165"]["class_type"] == "ModelPreviewOverrideKJ", compile_res["graph"]["165"]
                    assert compile_res["graph"]["165"]["inputs"]["tiny_vae"] == "taeh3.safetensors", compile_res["graph"]["165"]
                    assert compile_res["graph"]["165"]["inputs"]["preview_frames"] == 27, (
                        f"expected preview_frames=27 for length={p3_req.get('length')}, got {compile_res['graph']['165']['inputs']}"
                    )
                    assert compile_res["graph"]["165"]["inputs"]["preview_fps"] == 7, (
                        f"expected preview_fps=7 for length={p3_req.get('length')}, got {compile_res['graph']['165']['inputs']}"
                    )

                    # Simulate "kj_preview_override" WebSocket messages (mp4 blob + jpeg) & verify canvas compositing
                    canvas_before = await js(
                        """(() => {
                          const c = document.getElementById('stage-canvas');
                          const ctx = c.getContext('2d');
                          const d = ctx.getImageData(0, 0, c.width, c.height).data;
                          let sum = 0;
                          for (let i = 0; i < d.length; i += 4) sum += d[i] + d[i + 1] + d[i + 2];
                          return sum;
                        })()"""
                    )
                    await js(
                        f"""(() => {{
                          const sockets = (window.__directorSockets || []).filter(s => s.url.includes('/api/ws'));
                          if (!sockets.length) throw new Error('No /api/ws WebSocket captured');
                          const ws = sockets[sockets.length - 1];
                          const msgMp4 = JSON.stringify({{
                            type: 'kj_preview_override',
                            data: {{
                              node_id: '165',
                              image: {json.dumps(mp4_b64)},
                              mime: 'video/mp4',
                              w: 320,
                              h: 192,
                              step: 3,
                              total: 6,
                              sigma: 0.8,
                              fps: 7.0
                            }}
                          }});
                          ws.dispatchEvent(new MessageEvent('message', {{data: msgMp4}}));
                        }})()"""
                    )
                    await asyncio.sleep(0.25)
                    jpeg_b64 = _encode_solid_jpeg_b64((255, 0, 220))
                    await js(
                        f"""(() => {{
                          const sockets = (window.__directorSockets || []).filter(s => s.url.includes('/api/ws'));
                          const ws = sockets[sockets.length - 1];
                          const msgJpg = JSON.stringify({{
                            type: 'kj_preview_override',
                            data: {{
                              node_id: '165',
                              image: {json.dumps(jpeg_b64)},
                              mime: 'image/jpeg',
                              w: 336,
                              h: 192,
                              step: 4,
                              total: 6,
                              sigma: 0.571429,
                              step_ms: 1420.0,
                              avg_step_ms: 1380.0,
                              delta: 0.14
                            }}
                          }});
                          ws.dispatchEvent(new MessageEvent('message', {{data: msgJpg}}));
                        }})()"""
                    )
                    composited = False
                    for _ in range(25):
                        res_comp = await js(
                            f"""(() => {{
                              const c = document.getElementById('stage-canvas');
                              const ctx = c.getContext('2d');
                              const d = ctx.getImageData(0, 0, c.width, c.height).data;
                              let sum = 0;
                              for (let i = 0; i < d.length; i += 4) sum += d[i] + d[i + 1] + d[i + 2];
                              const pill = document.getElementById('live-tae-pill');
                              return {{
                                changed: sum !== {canvas_before},
                                pillText: pill ? pill.textContent.replace(/\\s+/g, '') : '',
                                storedClientId: sessionStorage.getItem('h3_director_client_id') || '',
                                activeClientId: window.__SPATIAL_DIRECTOR__?.wsClient?.clientId || '',
                              }};
                            }})()"""
                        )
                        if (
                            res_comp["changed"]
                            and "4/6" in res_comp["pillText"]
                            and "Δ0.14" in res_comp["pillText"]
                            and res_comp["storedClientId"]
                            and res_comp["storedClientId"] == res_comp["activeClientId"]
                        ):
                            composited = True
                            break
                        await asyncio.sleep(0.12)
                    assert composited, "Simulated kj_preview_override frame did not composite onto #stage-canvas / update #live-tae-pill or persist clientId"

                    csp_violations = await js("window.__cspViolations")
                    assert csp_violations == [], f"CSP violations recorded on /: {csp_violations}"
                    assert not console_errors, f"Console errors recorded on /: {console_errors}"
                    assert not runtime_exceptions, f"Runtime exceptions recorded on /: {runtime_exceptions}"

                    guide_box = await js("document.getElementById('btn-docs-guide')?.getBoundingClientRect().toJSON()")
                    assert guide_box and guide_box["width"] > 0 and guide_box["right"] <= 1280, f"#btn-docs-guide invalid: {guide_box}"

                    # Verify /docs Documentation Hub renders spatial_director_guide.md, Mermaid SVG, TOC links & cross-doc nav
                    await cdp("Page.navigate", {"url": f"{director_url.rstrip('/')}/docs"})
                    docs_ready = False
                    for _ in range(40):
                        docs_state = await js(
                            """(() => {
                              const h1 = document.querySelector('#doc-article h1');
                              const svg = document.querySelector('#doc-article svg.mermaid-svg');
                              const tocLinks = document.querySelectorAll('#doc-toc-nav a.toc-link').length;
                              return {
                                h1: h1 ? h1.textContent.trim() : '',
                                hasSvg: Boolean(svg),
                                tocLinks,
                                scrollWidth: document.documentElement.scrollWidth,
                              };
                            })()"""
                        )
                        if (
                            docs_state
                            and "Interactive Spatial Director" in docs_state["h1"]
                            and docs_state["hasSvg"]
                            and docs_state["tocLinks"] >= 4
                        ):
                            docs_ready = True
                            break
                        await asyncio.sleep(0.15)
                    assert docs_ready, f"/docs failed to hydrate tutorial, Mermaid SVG, or TOC: {docs_state}"
                    assert docs_state["scrollWidth"] <= 1280, f"/docs horizontal overflow: {docs_state['scrollWidth']}"

                    # Navigate to Edit Guidance via sidebar or internal link
                    await js("document.querySelector(\"#doc-tree button[data-doc-path='docs/wiki/edit_guidance.md']\")?.click()")
                    switched = False
                    for _ in range(30):
                        h1_next = await js("document.querySelector('#doc-article h1')?.textContent.trim() || ''")
                        if "Edit Guidance" in h1_next:
                            switched = True
                            break
                        await asyncio.sleep(0.12)
                    assert switched, "Clicking Edit Guidance in /docs sidebar did not load docs/wiki/edit_guidance.md"

                    csp_violations_docs = await js("window.__cspViolations")
                    assert csp_violations_docs == [], f"CSP violations recorded on /docs: {csp_violations_docs}"
                    assert not console_errors, f"Console errors recorded across /docs: {console_errors}"
                    assert not runtime_exceptions, f"Runtime exceptions recorded across /docs: {runtime_exceptions}"

            return "1280x800 Light Mode, 0 CSP/console errors, / & /docs Markdown/Mermaid hub, blob media-src, upload, popover, i2v_portal default, Wheel/Shift+Wheel/Alt+Wheel, Engine tablist, compile '165', and kj_preview_override canvas composite ok"
        finally:
            _terminate_proc(chrome_proc, "google-chrome")
            shutil.rmtree(user_data, ignore_errors=True)


def check_tier2_live_cdp() -> str:
    comfy_main = COMFY_ROOT / "main.py"
    if not comfy_main.is_file():
        skip(f"ComfyUI main.py missing at {COMFY_ROOT}")
    if not _find_chrome():
        skip("google-chrome binary not found")
    if _port_open("127.0.0.1", 8288):
        raise AssertionError("Port 8288 is already in use; refusing to collide")
    if _port_open("127.0.0.1", 8295):
        raise AssertionError("Port 8295 is already in use; refusing to collide")

    comfy_proc: subprocess.Popen | None = None
    director_proc: subprocess.Popen | None = None

    with tempfile.TemporaryDirectory(prefix="h3_director_tier2_") as tmp:
        root = Path(tmp)
        base_dir = root / "comfy_base"
        cn_dir = base_dir / "custom_nodes"
        temp_dir = root / "comfy_temp"
        input_dir = root / "comfy_input"
        output_dir = root / "comfy_output"
        user_dir = root / "comfy_user"
        state_dir = root / "director_state"
        for d in (cn_dir, temp_dir, input_dir, output_dir, user_dir, state_dir):
            d.mkdir(parents=True, exist_ok=True)

        (cn_dir / "ComfyUI-h3-explorations").symlink_to(REPO)
        for dep in ("ComfyUI-VideoHelperSuite", "ComfyUI-KJNodes"):
            dep_path = COMFY_ROOT / "custom_nodes" / dep
            if dep_path.is_dir():
                (cn_dir / dep).symlink_to(dep_path)

        anchor_png = root / "anchor_test.png"
        anchor_png.write_bytes(_encode_test_png_bytes())

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["H3_OUTPUT_DIR"] = str(output_dir)
        env["H3_COMFY_OUTPUT"] = str(output_dir)
        env["H3_INPUT_DIR"] = str(input_dir)
        env["H3_COMFY_INPUT"] = str(input_dir)
        for bad_k in ("H3_CAPTURE", "H3_TELEMETRY", "H3_SESSION", "H3_TRACE"):
            env.pop(bad_k, None)

        cmd_comfy = [
            sys.executable,
            str(comfy_main),
            "--cpu",
            "--listen",
            "127.0.0.1",
            "--port",
            "8288",
            "--base-directory",
            str(base_dir),
            "--models-directory",
            str(COMFY_ROOT / "models"),
            "--temp-directory",
            str(temp_dir),
            "--input-directory",
            str(input_dir),
            "--output-directory",
            str(output_dir),
            "--user-directory",
            str(user_dir),
            "--database-url",
            "sqlite:///:memory:",
            "--preview-method",
            "none",
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "ComfyUI-h3-explorations",
            "ComfyUI-VideoHelperSuite",
            "ComfyUI-KJNodes",
            "--disable-assets",
            "--disable-product-tips",
            "--offline",
        ]
        extra_paths = COMFY_ROOT / "extra_model_paths.yaml"
        if extra_paths.is_file():
            cmd_comfy.extend(["--extra-model-paths-config", str(extra_paths)])

        cmd_dir = [
            sys.executable,
            "-m",
            "director.server",
            "--host",
            "127.0.0.1",
            "--port",
            "8295",
            "--comfy-url",
            "http://127.0.0.1:8288",
            "--state-dir",
            str(state_dir),
        ]

        comfy_log_path = root / "comfy_8288.log"
        dir_log_path = root / "director_8295.log"
        comfy_log = comfy_log_path.open("w", encoding="utf-8")
        dir_log = dir_log_path.open("w", encoding="utf-8")
        try:
            comfy_proc = subprocess.Popen(
                cmd_comfy,
                cwd=str(COMFY_ROOT),
                env=env,
                stdout=comfy_log,
                stderr=subprocess.STDOUT,
            )
            obj_info = _wait_http_json(
                "http://127.0.0.1:8288/object_info", timeout_s=60.0, proc=comfy_proc, log_path=comfy_log_path
            )
            assert "MiniMaxH3SpatialDirector" in obj_info, "ComfyUI :8288 missing MiniMaxH3SpatialDirector"
            assert "ModelPreviewOverrideKJ" in obj_info, "ComfyUI :8288 missing ModelPreviewOverrideKJ"

            director_proc = subprocess.Popen(
                cmd_dir,
                cwd=str(REPO),
                env=env,
                stdout=dir_log,
                stderr=subprocess.STDOUT,
            )
            schema = _wait_http_json(
                "http://127.0.0.1:8295/api/schema", timeout_s=20.0, proc=director_proc, log_path=dir_log_path
            )
            assert schema.get("ok") is True, f"Director :8295 bad schema: {schema}"

            return asyncio.run(_run_cdp_browser("http://127.0.0.1:8295/", anchor_png))
        finally:
            _terminate_proc(director_proc, "director.server")
            _terminate_proc(comfy_proc, "comfyui")
            comfy_log.close()
            dir_log.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Run Tier 1 offline checks only")
    mode.add_argument("--live", action="store_true", help="Force Tier 2 live Chrome CDP + --cpu ComfyUI (:8288/:8295)")
    args = parser.parse_args(argv)

    needs("director/ standalone Spatial Director studio", _director_present())
    bootstrap(cpu=True)

    print("Spatial Director Studio: Tier 1 offline + Tier 2 Chrome CDP")
    case("compiler builds 4 spatial modes + pika_t2i still with dual VHS, save_output=False, and dynamic '165'", check_compiler_contracts)
    case("routes.compute_forget_plan cleans temp + orphaned uploads while protecting output/Video/ & active keeps", check_forget_plan_and_isolation)
    case("server security, schema hydration, context_suggest disabled, and POST /api/escalate clock/3f scaling", check_server_routes_and_escalate)
    case("static audit: 0 CDN URLs, 0 inline styles/handlers, 0 hardcoded presets/suggestions, all /api routes wired", check_static_and_route_audit)

    run_tier2 = False
    if args.live:
        if not (COMFY_ROOT / "main.py").is_file():
            raise SystemExit(f"--live requested but ComfyUI main.py not found under {COMFY_ROOT}")
        if not _find_chrome():
            raise SystemExit("--live requested but google-chrome binary not found")
        run_tier2 = True
    elif not args.offline:
        run_tier2 = (COMFY_ROOT / "main.py").is_file() and bool(_find_chrome())

    if run_tier2:
        case("Tier 2 live --cpu ComfyUI (:8288) + Director (:8295) + headless Chrome CDP (1280x800)", check_tier2_live_cdp)
    else:
        print("  SKIP  Tier 2 live Chrome CDP (--offline or Chrome/ComfyUI absent)")

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
