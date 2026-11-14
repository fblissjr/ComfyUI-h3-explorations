#!/usr/bin/env python3
"""Phase A deciding A/B benchmark: MiniMaxH3EditGuider optional spatial/temporal mask.

Runs (or validates via `--dry-run`) the matched-seed tortoise blizzard test:
  1. `scout_base`      -- unedited Pika render (`BasicGuider`, `seed=730451892`, `length=124`)
  2. `sam_mask`        -- single-shot `SAM3_VideoTrack` + `SAM3_TrackToMask` on `scout_base`
  3. `eg_unmasked`     -- `MiniMaxH3EditGuider` heavy_blizzard (`first_step=1, last_step=3, scale=1.85`)
  4. `eg_mask_tight`   -- `InvertMask(tight_mask)` on `MiniMaxH3EditGuider.mask`
  5. `eg_mask_grow24`  -- `GrowMask(24) -> MaskToImage -> ImageBlur(12, 6.0) -> ImageToMask -> InvertMask`

Media artifacts are written strictly under `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_2026-11-14/`
(and copied to `$H3_INPUT_DIR/edit_guider_mask_ab_2026-11-14/` for downstream loads).
`data/edit_guider_mask_ab_2026-11-14/` receives JSON prompt/metric records only (zero media).
`--dry-run` writes zero files.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO_ROOT / "workflows") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "workflows"))

from _lib import bootstrap  # noqa: E402

bootstrap()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import edit_guidance  # noqa: E402
import h3_config  # noqa: E402
import h3_env  # noqa: E402
import h3_rules  # noqa: E402
import preflight_graph  # noqa: E402
from _lib import comfy_client, frames  # noqa: E402

RUN_TAG = "edit_guider_mask_ab_2026-11-14"
BASE_WORKFLOW = _REPO_ROOT / "workflows" / "h3_text_to_video_pika_edit_guidance_api.json"
PROMPT_FILE = _REPO_ROOT / "prompt_bank" / "t2va_offpath_tortoise.txt"

DEFAULT_LENGTH = 124
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
DEFAULT_SEED = 730451892
DEFAULT_SCALE = 1.85
DEFAULT_FIRST_STEP = 1
DEFAULT_LAST_STEP = 3
DEFAULT_SAM_PROMPT = "tortoise"

DEFAULT_EDITS = "\n".join([
    "on a quiet city street on an overcast morning -> on a wind-lashed city street in a heavy blizzard with thick snow piled on every surface and swirling flakes",
    "soundscape: the tortoise's claws scrape softly on the asphalt, with light traffic in the distance -> freezing gale wind howls across deep snowdrifts as thick snow crunches softly underfoot",
])


def resolve_input_root() -> Path:
    """Resolve `$H3_INPUT_DIR` from process env or gitignored `h3.env`."""
    env = h3_env.load()
    raw = os.environ.get("H3_INPUT_DIR") or env.get("H3_INPUT_DIR")
    if not raw:
        raise RuntimeError("H3_INPUT_DIR is not set (configure h3.env or export H3_INPUT_DIR)")
    return Path(raw)


def _load_base_graph() -> dict[str, Any]:
    return json.loads(BASE_WORKFLOW.read_text(encoding="utf-8"))


def _apply_geometry_and_seed(g: dict[str, Any], *, width: int, height: int, length: int, seed: int) -> None:
    if h3_rules.snap_length(length) != length:
        raise ValueError(f"length={length} is not a valid 17n+5 H3 length")
    g["27"]["inputs"]["length"] = int(length)
    g["6"]["inputs"]["noise_seed"] = int(seed)
    if width != DEFAULT_WIDTH or height != DEFAULT_HEIGHT:
        for cond_id in ("5", "131"):
            if cond_id in g:
                g[cond_id]["inputs"]["width"] = int(width)
                g[cond_id]["inputs"]["height"] = int(height)


def _wire_provenance_and_output(g: dict[str, Any], *, note: str, arm: str) -> None:
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {
            "latent": ["10", 0],
            "model": ["21", 0],
            "sigmas": ["60", 0],
            "clip": ["2", 0],
            "note": note,
        },
    }
    g["11"]["inputs"]["samples"] = ["30", 0]
    g["12"]["inputs"]["samples"] = ["30", 0]
    g["13"]["inputs"]["filename_prefix"] = f"Video/{RUN_TAG}/{arm}"


def build_scout_graph(
    orig_prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Unedited Pika `BasicGuider` render used both as baseline and SAM3 tracking source."""
    g = _load_base_graph()
    _apply_geometry_and_seed(g, width=width, height=height, length=length, seed=seed)
    g["133"]["inputs"]["value"] = orig_prompt
    g.pop("130", None)
    g.pop("131", None)
    g["26"]["inputs"].pop("edited_conditioning", None)
    g["9"] = {
        "class_type": "BasicGuider",
        "inputs": {
            "model": ["21", 0],
            "conditioning": ["26", 0],
        },
    }
    _wire_provenance_and_output(g, note="phaseA:scout_base", arm="scout_base")
    return g


def build_sam_mask_graph(
    *,
    scout_input_rel: str = f"{RUN_TAG}/scout_base.mp4",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    sam_prompt: str = DEFAULT_SAM_PROMPT,
    detection_threshold: float = 0.45,
    max_objects: int = 4,
    detect_interval: int = 12,
) -> dict[str, Any]:
    """Single-shot `SAM3_VideoTrack` + `SAM3_TrackToMask` lossless FFV1 mask extractor."""
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": scout_input_rel,
                "force_rate": 24,
                "custom_width": int(width),
                "custom_height": int(height),
                "frame_load_cap": int(length),
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": h3_config.SEGMENTER},
        },
        "3": {
            "class_type": "MiniMaxH3SAM31Corrections",
            "inputs": {
                "segmenter": ["2", 0],
                "segmenter_clip": ["2", 1],
                **h3_config.MASKED_JOB_SAM_CORRECTIONS,
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 1], "text": sam_prompt},
        },
        "5": {
            "class_type": "SAM3_VideoTrack",
            "inputs": {
                "images": ["1", 0],
                "model": ["3", 0],
                "conditioning": ["4", 0],
                "detection_threshold": float(detection_threshold),
                "max_objects": int(max_objects),
                "detect_interval": int(detect_interval),
            },
        },
        "6": {
            "class_type": "SAM3_TrackToMask",
            "inputs": {"track_data": ["5", 0], "object_indices": ""},
        },
        "7": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["6", 0]},
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 24.0,
                "filename_prefix": f"Video/{RUN_TAG}/tortoise_mask_tight",
                **h3_config.MASKED_JOB_LOSSLESS,
            },
        },
    }


def build_edit_guider_graph(
    *,
    arm: str,
    orig_prompt: str,
    edits: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    scale: float = DEFAULT_SCALE,
    first_step: int = DEFAULT_FIRST_STEP,
    last_step: int = DEFAULT_LAST_STEP,
    mask_mode: str | None = None,
    mask_input_rel: str = f"{RUN_TAG}/tortoise_mask_tight.mkv",
) -> dict[str, Any]:
    """Matched-seed `MiniMaxH3EditGuider` graph with `mask_mode` in `{None, 'tight', 'grow24'}`."""
    if mask_mode not in (None, "tight", "grow24"):
        raise ValueError(f"Unsupported mask_mode={mask_mode!r}")
    g = _load_base_graph()
    _apply_geometry_and_seed(g, width=width, height=height, length=length, seed=seed)
    g["133"]["inputs"]["value"] = orig_prompt
    g["130"]["inputs"]["edits"] = edits
    eg_inputs = g["9"]["inputs"]
    eg_inputs["scale"] = float(scale)
    eg_inputs["first_step"] = int(first_step)
    eg_inputs["last_step"] = int(last_step)
    eg_inputs["streams"] = "both"
    eg_inputs["audio_scale"] = 1.0

    if mask_mode is not None:
        g["200"] = {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": mask_input_rel,
                "force_rate": 24,
                "custom_width": int(width),
                "custom_height": int(height),
                "frame_load_cap": int(length),
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        }
        g["201"] = {
            "class_type": "ImageToMask",
            "inputs": {"image": ["200", 0], "channel": "red"},
        }
        if mask_mode == "tight":
            g["202"] = {"class_type": "InvertMask", "inputs": {"mask": ["201", 0]}}
            eg_inputs["mask"] = ["202", 0]
        else:
            # GrowMask adds a scout-lag margin; ImageBlur softens the foreground silhouette
            # contour (core FeatherMask fades canvas edges rather than interior shapes).
            g["202"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": ["201", 0], "expand": 24, "tapered_corners": True},
            }
            g["203"] = {"class_type": "MaskToImage", "inputs": {"mask": ["202", 0]}}
            g["204"] = {
                "class_type": "ImageBlur",
                "inputs": {"image": ["203", 0], "blur_radius": 12, "sigma": 6.0},
            }
            g["205"] = {
                "class_type": "ImageToMask",
                "inputs": {"image": ["204", 0], "channel": "red"},
            }
            g["206"] = {"class_type": "InvertMask", "inputs": {"mask": ["205", 0]}}
            eg_inputs["mask"] = ["206", 0]

    _wire_provenance_and_output(g, note=f"phaseA:{arm}", arm=arm)
    return g


def build_all_graphs(args: argparse.Namespace) -> tuple[str, str, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Construct all 5 benchmark graphs and validate edited prompt text."""
    orig_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    edited_prompt, edit_records = edit_guidance.apply_edits(orig_prompt, args.edits)
    if len(edit_records) != 2 or edited_prompt == orig_prompt:
        raise ValueError(f"Expected 2 applied edit clauses, got {len(edit_records)}: {edit_records!r}")
    graphs = {
        "scout_base": build_scout_graph(
            orig_prompt, width=args.width, height=args.height, length=args.length, seed=args.seed
        ),
        "sam_mask": build_sam_mask_graph(
            width=args.width, height=args.height, length=args.length, sam_prompt=args.sam_prompt
        ),
        "eg_unmasked": build_edit_guider_graph(
            arm="eg_unmasked",
            orig_prompt=orig_prompt,
            edits=args.edits,
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            scale=args.scale,
            first_step=args.first_step,
            last_step=args.last_step,
            mask_mode=None,
        ),
        "eg_mask_tight": build_edit_guider_graph(
            arm="eg_mask_tight",
            orig_prompt=orig_prompt,
            edits=args.edits,
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            scale=args.scale,
            first_step=args.first_step,
            last_step=args.last_step,
            mask_mode="tight",
        ),
        "eg_mask_grow24": build_edit_guider_graph(
            arm="eg_mask_grow24",
            orig_prompt=orig_prompt,
            edits=args.edits,
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            scale=args.scale,
            first_step=args.first_step,
            last_step=args.last_step,
            mask_mode="grow24",
        ),
    }
    return orig_prompt, edited_prompt, list(edit_records), graphs


def verify_dry_run(args: argparse.Namespace) -> int:
    """Validate all graphs statically via `preflight_graph.grade` with zero files written."""
    orig_prompt, edited_prompt, edit_records, graphs = build_all_graphs(args)

    for arm in ("scout_base", "eg_unmasked", "eg_mask_tight", "eg_mask_grow24"):
        g = graphs[arm]
        assert g["11"]["inputs"]["samples"] == ["30", 0], f"{arm}: VAEDecode not chained through ProvenanceStamp"
        assert g["12"]["inputs"]["samples"] == ["30", 0], f"{arm}: VAEDecodeAudio not chained through ProvenanceStamp"
        findings = preflight_graph.grade(g["5"], g, stem=arm)
        fails = [msg for lvl, msg in findings if lvl == "FAIL"]
        if fails:
            raise AssertionError(f"preflight_graph FAIL on {arm}: {fails}")

    edited_probe = copy.deepcopy(graphs["eg_unmasked"])
    edited_probe["133"]["inputs"]["value"] = edited_prompt
    edited_findings = preflight_graph.grade(edited_probe["5"], edited_probe, stem="eg_edited_resolved")
    edited_fails = [msg for lvl, msg in edited_findings if lvl == "FAIL"]
    if edited_fails:
        raise AssertionError(f"preflight_graph FAIL on resolved edited prompt: {edited_fails}")

    assert graphs["eg_mask_tight"]["9"]["inputs"]["mask"] == ["202", 0]
    assert graphs["eg_mask_tight"]["202"]["class_type"] == "InvertMask"
    assert graphs["eg_mask_grow24"]["9"]["inputs"]["mask"] == ["206", 0]
    assert graphs["eg_mask_grow24"]["202"]["class_type"] == "GrowMask"
    assert graphs["eg_mask_grow24"]["204"]["class_type"] == "ImageBlur"
    assert graphs["eg_mask_grow24"]["206"]["class_type"] == "InvertMask"

    print(
        f"[dry-run OK] 5 graphs validated ({args.width}x{args.height}, length={args.length}, "
        f"seed={args.seed}, scale={args.scale}, first_step={args.first_step}, last_step={args.last_step}); "
        f"0 preflight FAIL across 4 H3 graphs + resolved edited prompt ({len(edit_records)} clauses). "
        f"orig={len(orig_prompt)}ch edited={len(edited_prompt)}ch"
    )
    return 0


def _pick_saved_video(history_entry: dict[str, Any], out_root: Path, *, expected_ext: str, prefer_audio: bool) -> Path:
    items = comfy_client.collect_saved_files(history_entry)
    candidates: list[Path] = []
    for item in items:
        fn = str(item.get("filename") or "")
        if not fn.endswith(expected_ext):
            continue
        sub = str(item.get("subfolder") or "")
        path = (out_root / sub / fn) if sub else (out_root / fn)
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No *{expected_ext} saved output found in history outputs={items!r}")
    if prefer_audio:
        audio_hits = [p for p in candidates if p.stem.endswith("-audio")]
        if audio_hits:
            return audio_hits[0]
    return candidates[0]


def _load_rgb_frames(path: Path, size: tuple[int, int], count: int) -> np.ndarray:
    w, h = size
    raw = list(frames.stream(path, size, count=count, pix="rgb24"))
    if len(raw) != count:
        raise RuntimeError(f"{path.name}: expected {count} frames at {w}x{h}, got {len(raw)}")
    return np.stack([np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3) for buf in raw], axis=0)


def _load_gray_frames(path: Path, size: tuple[int, int], count: int) -> np.ndarray:
    w, h = size
    raw = list(frames.stream(path, size, count=count, pix="gray"))
    if len(raw) != count:
        raise RuntimeError(f"{path.name}: expected {count} mask frames at {w}x{h}, got {len(raw)}")
    return np.stack([np.frombuffer(buf, dtype=np.uint8).reshape(h, w) for buf in raw], axis=0)


def _compute_region_metrics(scout_rgb: np.ndarray, arm_rgb: np.ndarray, tight_mask_u8: np.ndarray) -> dict[str, float]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    diff = np.abs(arm_rgb.astype(np.float64) - scout_rgb.astype(np.float64)).mean(axis=-1)
    inside_sum = 0.0
    inside_px = 0
    outside_sum = 0.0
    outside_px = 0
    for f_idx in range(diff.shape[0]):
        fg = (tight_mask_u8[f_idx] >= 128).astype(np.uint8)
        dil = cv2.dilate(fg, kernel)
        in_mask = fg > 0
        out_mask = dil == 0
        if np.any(in_mask):
            inside_sum += float(diff[f_idx][in_mask].sum())
            inside_px += int(in_mask.sum())
        if np.any(out_mask):
            outside_sum += float(diff[f_idx][out_mask].sum())
            outside_px += int(out_mask.sum())
    return {
        "full_frame_mae": round(float(diff.mean()), 3),
        "inside_tortoise_mae": round(inside_sum / inside_px, 3) if inside_px else float("nan"),
        "outside_margin_mae": round(outside_sum / outside_px, 3) if outside_px else float("nan"),
        "inside_pixels": inside_px,
        "outside_pixels": outside_px,
    }


def _write_2x2_review_videos(
    out_dir: Path,
    arm_videos: dict[str, np.ndarray],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    cw, ch = width // 2, height // 2
    grid_mp4 = out_dir / "ab_2x2_stack.mp4"
    grid_cmd = [
        "nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24", "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(grid_mp4),
    ]
    proc = subprocess.Popen(grid_cmd, stdin=subprocess.PIPE)
    strip = frames.OneASecond(out_dir / "ab_2x2_stack", (width, height))
    placements = [
        ("scout_base", 0, 0),
        ("eg_unmasked", cw, 0),
        ("eg_mask_tight", 0, ch),
        ("eg_mask_grow24", cw, ch),
    ]
    n_frames = next(iter(arm_videos.values())).shape[0]
    try:
        for idx in range(n_frames):
            canvas = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(canvas)
            for label, ox, oy in placements:
                tile = Image.fromarray(arm_videos[label][idx]).resize((cw, ch), Image.Resampling.BILINEAR)
                canvas.paste(tile, (ox, oy))
                tag = f"{idx:03d} | {label}"
                draw.rectangle([ox + 8, oy + 8, ox + 12 + 7 * len(tag), oy + 26], fill=(0, 0, 0))
                draw.text((ox + 12, oy + 11), tag, fill=(255, 255, 255))
            raw = canvas.tobytes()
            assert proc.stdin is not None
            proc.stdin.write(raw)
            strip.write(raw, idx)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        code_grid = proc.wait()
        code_strip, parts = strip.close()
    if code_grid != 0 or code_strip != 0:
        raise RuntimeError(f"ffmpeg review encoding failed: grid={code_grid}, strip={code_strip}")
    return {"mp4": grid_mp4.name, "strip_parts": [Path(p["file"]).name for p in parts]}


def run_live_ab(args: argparse.Namespace) -> int:
    """Submit all 5 graphs sequentially to live `:8188`, analyze, and stack review strips."""
    if not comfy_client.has_input("MiniMaxH3EditGuider", "mask", base_url=args.url):
        print(
            "ERROR: live ComfyUI MiniMaxH3EditGuider does not advertise 'mask' yet; reload :8188 first.",
            file=sys.stderr,
        )
        return 2
    q = comfy_client.get_queue(base_url=args.url)
    if q.get("queue_running") or q.get("queue_pending"):
        raise RuntimeError(f"ComfyUI queue is not empty: {q}")

    out_root = h3_config.output_dir()
    out_dir = out_root / "Video" / RUN_TAG
    in_dir = resolve_input_root() / RUN_TAG
    data_dir = _REPO_ROOT / "data" / RUN_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    orig_prompt, edited_prompt, edit_records, graphs = build_all_graphs(args)

    def _submit_and_wait(name: str, expected_ext: str, *, prefer_audio: bool) -> tuple[str, float, Path]:
        t0 = time.monotonic()
        pid = comfy_client.submit_prompt(graphs[name], base_url=args.url)
        print(f"[queue] {name} -> {pid}", flush=True)
        entry = comfy_client.wait_for_prompt(pid, base_url=args.url)
        elapsed = round(time.monotonic() - t0, 2)
        saved = _pick_saved_video(entry, out_root, expected_ext=expected_ext, prefer_audio=prefer_audio)
        print(f"[done] {name} in {elapsed:.2f}s -> {saved.name}", flush=True)
        return pid, elapsed, saved

    prompts_record: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "length": args.length,
        "scale": args.scale,
        "first_step": args.first_step,
        "last_step": args.last_step,
        "sam_prompt": args.sam_prompt,
        "original_prompt": orig_prompt,
        "edits": args.edits,
        "edited_prompt": edited_prompt,
        "edit_records": edit_records,
        "graphs": graphs,
    }
    (data_dir / "prompts.json").write_text(json.dumps(prompts_record, indent=2), encoding="utf-8")

    results: dict[str, Any] = {"arms": {}}
    pid, elapsed, scout_mp4 = _submit_and_wait("scout_base", ".mp4", prefer_audio=True)
    shutil.copy2(scout_mp4, in_dir / "scout_base.mp4")
    results["arms"]["scout_base"] = {"prompt_id": pid, "elapsed_s": elapsed, "file": scout_mp4.name}

    pid, elapsed, sam_mkv = _submit_and_wait("sam_mask", ".mkv", prefer_audio=False)
    tight_mask = _load_gray_frames(sam_mkv, (args.width, args.height), args.length)
    mask_nonzero_frames = int(np.sum(np.max(tight_mask, axis=(1, 2)) > 0))
    if mask_nonzero_frames == 0 or int(tight_mask.max()) == 0:
        raise RuntimeError("SAM3_VideoTrack returned an all-zero tortoise mask; aborting before EditGuider arms")
    shutil.copy2(sam_mkv, in_dir / "tortoise_mask_tight.mkv")
    results["arms"]["sam_mask"] = {
        "prompt_id": pid,
        "elapsed_s": elapsed,
        "file": sam_mkv.name,
        "nonzero_frames": mask_nonzero_frames,
        "mean_area_pct": round(float((tight_mask >= 128).mean() * 100.0), 3),
    }

    scout_rgb = _load_rgb_frames(scout_mp4, (args.width, args.height), args.length)
    arm_videos: dict[str, np.ndarray] = {"scout_base": scout_rgb}

    for arm in ("eg_unmasked", "eg_mask_tight", "eg_mask_grow24"):
        pid, elapsed, arm_mp4 = _submit_and_wait(arm, ".mp4", prefer_audio=True)
        arm_rgb = _load_rgb_frames(arm_mp4, (args.width, args.height), args.length)
        arm_videos[arm] = arm_rgb
        metrics = _compute_region_metrics(scout_rgb, arm_rgb, tight_mask)
        results["arms"][arm] = {
            "prompt_id": pid,
            "elapsed_s": elapsed,
            "file": arm_mp4.name,
            **metrics,
        }

    results["review"] = _write_2x2_review_videos(out_dir, arm_videos, width=args.width, height=args.height)
    (data_dir / "ab_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {data_dir / 'ab_summary.json'} and review stack under OUT_DIR/{RUN_TAG}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate graphs statically (zero writes; default)")
    parser.add_argument("--run", action="store_true", help="Submit all 5 graphs to live :8188 ComfyUI")
    parser.add_argument("--url", default=comfy_client.DEFAULT_COMFY_URL, help="ComfyUI server base URL")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--first-step", type=int, default=DEFAULT_FIRST_STEP)
    parser.add_argument("--last-step", type=int, default=DEFAULT_LAST_STEP)
    parser.add_argument("--sam-prompt", default=DEFAULT_SAM_PROMPT)
    parser.add_argument("--edits", default=DEFAULT_EDITS)
    args = parser.parse_args(argv)
    if args.run:
        return run_live_ab(args)
    return verify_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
