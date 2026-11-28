#!/usr/bin/env python3
"""Ready-to-run live `:8188` matched-seed smoke suite for `MiniMaxH3SelfMaskRouter(step0_critic=True)`.

Tests three real-world scenes (never asking for bars/watermarks explicitly) in matched-seed
`off` vs `on` pairs (`90f`, `1344x768`, Pika H3 Flex 6-step LCM, `mask_source="thermostat_only"`):
  1. `tv_broadcast_storm`   — unrequested TV broadcast chyron / logo / border test
  2. `spaghetti_western`    — unrequested 1960s Techniscope anamorphic letterbox bar test
  3. `clean_street_control` — clean full-bleed control verifying 0 false-positive rollbacks (`1.00x`)

Writes videos and `report.json` under `$H3_OUTPUT_DIR/Video/step0_critic_smoke_2026-11-27/`
(never `data/`). Use `--dry-run` on CPU (`CUDA_VISIBLE_DEVICES=""`) to validate all graphs
against `:8188/object_info` without queueing any GPU work.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from workflows import h3_env as _h3_env  # noqa: E402

DEFAULT_SEED = 730451892
DEFAULT_LENGTH = 90
DEFAULT_OUT_SUBDIR = "Video/step0_critic_smoke_2026-11-27"
BASE_WORKFLOW = REPO_DIR / "workflows" / "h3_text_to_video_pika_api.json"

SCENES: dict[str, dict[str, str]] = {
    "tv_broadcast_storm": {
        "purpose": "Unrequested TV breaking-news chyron / watermark / border rescue check",
        "prompt": (
            "integrated_multimodal_description: [Shot 1] Live-action 1990s handheld television breaking news "
            "broadcast from a stormy coastal lighthouse pier at night, sheets of rain lashing dark wet basalt "
            "rocks and foaming ocean surf as a weather correspondent in a soaked bright-yellow rain slicker "
            "holds a black foam-topped broadcast microphone and shouts over the gale under harsh portable "
            "camera floodlights. The rotating lighthouse beam sweeps through heavy mist behind them.\n\n"
            "overall_soundscape: Howling coastal gale wind buffeting the microphone, crashing ocean waves "
            "against stone piers, and heavy driving rain.\n\n"
            "non_diegetic_music: N/A"
        ),
    },
    "spaghetti_western": {
        "purpose": "Unrequested 1960s Techniscope anamorphic letterbox bar rescue check",
        "prompt": (
            "integrated_multimodal_description: [Shot 1] Epic 1960s Italian Spaghetti Western shot on vintage "
            "35mm Techniscope anamorphic cinema lens with warm sun-bleached film grain, an extreme wide "
            "tracking shot of a lone weathered rider in a dusty woven poncho and wide-brimmed hat guiding a "
            "bay horse at a steady trot across vast red sandstone mesas in Monument Valley under blazing "
            "late-afternoon sunlight and crisp desert shadows.\n\n"
            "overall_soundscape: Dry desert wind whispering over red canyon grit, rhythmic steady horse "
            "hooves stepping on packed dirt, and soft creaking saddle leather.\n\n"
            "non_diegetic_music: N/A"
        ),
    },
    "clean_street_control": {
        "purpose": "Clean full-bleed control verifying zero false-positive Step-0 rollback (1.00x)",
        "prompt": (
            "integrated_multimodal_description: [Shot 1] Live-action, natural 35mm medium-wide tracking shot "
            "gliding smoothly down a sunlit European cobblestone street in a lively neighborhood morning "
            "market, warm afternoon sunlight grazing terracotta walls, hanging flower baskets, striped canvas "
            "awnings, and pedestrians strolling past wooden fruit and flower stalls.\n\n"
            "overall_soundscape: Warm open-air neighborhood street ambience, distant casual footsteps on "
            "cobblestones, gentle breeze rustling awning canvas, and soft birdsong.\n\n"
            "non_diegetic_music: N/A"
        ),
    },
}


def build_critic_graph(
    *,
    prompt: str,
    step0_critic: bool,
    filename_prefix: str,
    seed: int = DEFAULT_SEED,
    length: int = DEFAULT_LENGTH,
    base_workflow: Path = BASE_WORKFLOW,
) -> dict[str, Any]:
    """Attach `MiniMaxH3SelfMaskRouter(mask_source='thermostat_only', step0_critic=...)` onto the Pika T2V graph."""
    g = json.loads(base_workflow.read_text(encoding="utf-8"))
    g["27"]["inputs"]["length"] = int(length)
    g["5"]["inputs"]["prompt"] = str(prompt)
    g["6"]["inputs"]["noise_seed"] = int(seed)
    guider_model = g["9"]["inputs"]["model"]
    g["149"] = {
        "class_type": "MiniMaxH3SelfMaskRouter",
        "inputs": {
            "model": guider_model,
            "mask_source": "thermostat_only",
            "target_phrase": "",
            "threshold_z": 1.35,
            "softness": 0.25,
            "dilation_passes": 0,
            "isolation": "early_0_24_shield",
            "audio_zone": "base",
            "invert": False,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "replay_step0": False,
            "step0_critic": bool(step0_critic),
        },
    }
    g["9"]["inputs"]["model"] = ["149", 0]
    g["13"]["inputs"]["filename_prefix"] = str(filename_prefix)
    return g


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _validate_graph_against_server(server_url: str, label: str, graph: dict[str, Any]) -> None:
    obj_info = _http_json(f"{server_url}/object_info", timeout=15.0)
    if "MiniMaxH3SelfMaskRouter" not in obj_info:
        raise RuntimeError(f"{label}: MiniMaxH3SelfMaskRouter is not loaded on {server_url}")
    smr_in = obj_info["MiniMaxH3SelfMaskRouter"].get("input", {})
    all_smr = {**smr_in.get("required", {}), **smr_in.get("optional", {})}
    if "step0_critic" not in all_smr:
        raise RuntimeError(f"{label}: {server_url} has stale MiniMaxH3SelfMaskRouter without step0_critic")
    for nid, node in graph.items():
        ct = node["class_type"]
        if ct not in obj_info:
            raise RuntimeError(f"{label}: node {nid} unknown class_type {ct!r}")
        for k, v in node.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] not in graph:
                raise RuntimeError(f"{label}: node {nid}.{k} dangling link -> {v[0]}")


def _measure_video(path: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    top_l, bot_l, mid_l, sat_l, hi_l = [], [], [], [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        c_max = rgb.max(axis=-1)
        c_min = rgb.min(axis=-1)
        sat = np.where(c_max > 1e-6, (c_max - c_min) / np.maximum(c_max, 1e-6), 0.0)
        h = luma.shape[0]
        top_l.append(float(luma[: int(h * 0.10)].mean()))
        bot_l.append(float(luma[int(h * 0.90) :].mean()))
        mid_l.append(float(luma[int(h * 0.15) : int(h * 0.85)].mean()))
        sat_l.append(float(sat.mean()))
        hi_l.append(float((c_max >= 0.985).mean()))
    cap.release()
    return {
        "frames": len(top_l),
        "top_10pct_luma": round(float(np.mean(top_l)) if top_l else 0.0, 4),
        "bot_10pct_luma": round(float(np.mean(bot_l)) if bot_l else 0.0, 4),
        "mid_luma": round(float(np.mean(mid_l)) if mid_l else 0.0, 4),
        "mean_sat": round(float(np.mean(sat_l)) if sat_l else 0.0, 4),
        "hi_clip_frac": round(float(np.mean(hi_l)) if hi_l else 0.0, 4),
    }


def _pair_rgb_mae(path_a: Path, path_b: Path) -> float:
    import cv2
    import numpy as np

    ca, cb = cv2.VideoCapture(str(path_a)), cv2.VideoCapture(str(path_b))
    maes = []
    while True:
        oa, fa = ca.read()
        ob, fb = cb.read()
        if not (oa and ob):
            break
        if fa.shape == fb.shape:
            maes.append(float(np.abs(fa.astype(np.float32) - fb.astype(np.float32)).mean()))
    ca.release()
    cb.release()
    return round(float(np.mean(maes)) if maes else 0.0, 3)


def _queue_and_wait(server_url: str, label: str, graph: dict[str, Any]) -> tuple[Path, float]:
    t0 = time.time()
    resp = _http_json(f"{server_url}/prompt", payload={"prompt": graph}, timeout=20.0)
    pid = str(resp["prompt_id"])
    print(f"queued {label}: prompt_id={pid}", flush=True)
    while True:
        hist = _http_json(f"{server_url}/history/{pid}", timeout=10.0)
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status", {})
            dt = round(time.time() - t0, 1)
            if st.get("status_str") != "success":
                raise RuntimeError(f"{label} failed ({dt}s): {st.get('messages')}")
            mp4 = Path(entry["outputs"]["13"]["gifs"][0]["fullpath"])
            print(f"done   {label}: {dt}s -> {mp4}", flush=True)
            return mp4, dt
        time.sleep(2.5)


def main() -> int:
    _h3_env.load()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server base URL")
    p.add_argument(
        "--scenes",
        nargs="+",
        choices=list(SCENES.keys()),
        default=list(SCENES.keys()),
        help="Subset of scenes to run (default: all 3)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    p.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    p.add_argument("--dry-run", action="store_true", help="Build and validate all graphs against :8188 without queueing")
    p.add_argument("--allow-busy", action="store_true", help="Queue even if :8188 currently has active/pending jobs")
    args = p.parse_args()

    out_root = Path(os.environ.get("H3_OUTPUT_DIR", ""))
    if not args.dry_run and not str(out_root).strip():
        raise SystemExit("H3_OUTPUT_DIR is unset; source h3.env first")

    server = args.server.rstrip("/")
    if not args.dry_run and not args.allow_busy:
        q = _http_json(f"{server}/queue", timeout=10.0)
        if q.get("queue_running") or q.get("queue_pending"):
            raise SystemExit(
                f"REFUSED: {server} is busy (running={len(q.get('queue_running', []))}, "
                f"pending={len(q.get('queue_pending', []))}); pass --allow-busy or wait until idle"
            )

    graphs: list[tuple[str, str, bool, dict[str, Any]]] = []
    for scene_id in args.scenes:
        spec = SCENES[scene_id]
        for flag, tag in ((False, "off"), (True, "on")):
            label = f"{scene_id}_{tag}"
            prefix = f"{args.out_subdir}/{label}"
            g = build_critic_graph(
                prompt=spec["prompt"],
                step0_critic=flag,
                filename_prefix=prefix,
                seed=args.seed,
                length=args.length,
            )
            graphs.append((scene_id, label, flag, g))

    if args.dry_run:
        for _sid, label, _flag, g in graphs:
            _validate_graph_against_server(server, label, g)
            print(f"ok  dry-run validated {label} ({len(g)} nodes, step0_critic={g['149']['inputs']['step0_critic']})")
        return 0

    report: dict[str, Any] = {
        "seed": args.seed,
        "length": args.length,
        "scenes": {},
    }
    for scene_id in args.scenes:
        scene_out: dict[str, Any] = {"purpose": SCENES[scene_id]["purpose"], "arms": {}}
        paths: dict[str, Path] = {}
        for sid, label, flag, g in graphs:
            if sid != scene_id:
                continue
            mp4, dt = _queue_and_wait(server, label, g)
            arm_key = "on" if flag else "off"
            paths[arm_key] = mp4
            scene_out["arms"][arm_key] = {
                "mp4": mp4.name,
                "elapsed_s": dt,
                **_measure_video(mp4),
            }
        scene_out["off_vs_on_rgb_mae"] = _pair_rgb_mae(paths["off"], paths["on"])
        report["scenes"][scene_id] = scene_out

    report_path = out_root / args.out_subdir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
