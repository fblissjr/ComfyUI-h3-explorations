#!/usr/bin/env python3
"""Phase A deciding A/B benchmark: MiniMaxH3EditGuider optional spatial/temporal mask.

Runs (or validates via `--dry-run`) the matched-seed tortoise blizzard test:
  1. `scout_base`      -- unedited Pika render (`BasicGuider`, `seed=730451892`, `length=124`)
  2. `sam_mask`        -- single-shot `SAM3_VideoTrack` + `SAM3_TrackToMask` on `scout_base`
                          (`+`-delimited phrases unioned via `MaskComposite(operation="add")`)
  3. `eg_unmasked`     -- `MiniMaxH3EditGuider` heavy_blizzard (`first_step=2, last_step=3, scale=1.85`)
  4. `eg_mask_tight`   -- `InvertMask(tight_mask)` on `MiniMaxH3EditGuider.mask`
  5. `eg_mask_grow24`  -- `GrowMask(24) -> MaskToImage -> ImageBlur(12, sigma) -> ImageToMask -> InvertMask`

Media artifacts are written strictly under `$H3_OUTPUT_DIR/Video/<run_tag>/`.
`data/<run_tag>/` receives scrubbed JSON prompt/metric records only (zero media).
`--dry-run` writes zero files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
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

import h3_config  # noqa: E402
import h3_prompt_edit  # noqa: E402
import h3_rules  # noqa: E402
import preflight_graph  # noqa: E402
from _lib import comfy_client, frames  # noqa: E402

RUN_TAG = "edit_guider_mask_ab_fs2"
BASE_WORKFLOW = _REPO_ROOT / "workflows" / "h3_text_to_video_pika_edit_guidance_api.json"
PROMPT_FILE = _REPO_ROOT / "prompt_bank" / "t2va_offpath_tortoise.txt"

DEFAULT_LENGTH = 124
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
DEFAULT_SEED = 730451892
DEFAULT_SCALE = 1.85
DEFAULT_FIRST_STEP = 2
DEFAULT_LAST_STEP = 3
DEFAULT_STREAMS = "independent"
DEFAULT_AUDIO_SCALE = 1.0
# ImageBlur (`comfy_extras/nodes_post_processing.py:74-78`) builds its Gaussian kernel over normalized
# [-1, 1] coordinates rather than pixels; sigma=0.45 gives a Gaussian taper (6.0 acts nearly like a box).
DEFAULT_BLUR_SIGMA = 0.45
DEFAULT_SAM_PROMPT = "tortoise+candle"
DEFAULT_SAM_THRESHOLD = 0.45

DEFAULT_EDITS = "\n".join([
    "on a quiet city street on an overcast morning -> on a wind-lashed city street in a heavy blizzard with thick snow piled on every surface, including the tortoise's shell, and swirling flakes",
    "soundscape: the tortoise's claws scrape softly on the asphalt, with light traffic in the distance -> freezing gale wind howls across deep snowdrifts as thick snow crunches softly underfoot",
])


def _load_base_graph() -> dict[str, Any]:
    return json.loads(BASE_WORKFLOW.read_text(encoding="utf-8"))


def _scrub_paths(obj: Any, out_root: Path) -> Any:
    """Replace concrete `$H3_OUTPUT_DIR` mount paths with literal `$H3_OUTPUT_DIR` in records."""
    root_str = str(out_root).rstrip("/")
    if isinstance(obj, str):
        return obj.replace(root_str, "$H3_OUTPUT_DIR")
    if isinstance(obj, list):
        return [_scrub_paths(v, out_root) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub_paths(v, out_root) for k, v in obj.items()}
    return obj


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


def _wire_provenance_and_output(
    g: dict[str, Any],
    *,
    note: str,
    arm: str,
    run_tag: str = RUN_TAG,
) -> None:
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
    g["13"]["inputs"]["filename_prefix"] = f"Video/{run_tag}/{arm}"


def build_scout_graph(
    orig_prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    run_tag: str = RUN_TAG,
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
    _wire_provenance_and_output(g, note="phaseA:scout_base", arm="scout_base", run_tag=run_tag)
    return g


def _split_sam_phrases(sam_prompt: str) -> list[str]:
    phrases = [p.strip() for p in str(sam_prompt).split("+") if p.strip()]
    if not phrases:
        raise ValueError(f"sam_prompt must name at least one target phrase, got {sam_prompt!r}")
    return phrases


def _phrase_slug(idx: int, phrase: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return slug or f"p{idx}"


def build_sam_mask_graph(
    *,
    scout_video_path: str = f"$H3_OUTPUT_DIR/Video/{RUN_TAG}/scout_base.mp4",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    sam_prompt: str = DEFAULT_SAM_PROMPT,
    run_tag: str = RUN_TAG,
    detection_threshold: float = DEFAULT_SAM_THRESHOLD,
    max_objects: int = 4,
    detect_interval: int = 12,
) -> dict[str, Any]:
    """Single-shot `SAM3_VideoTrack` + `SAM3_TrackToMask` lossless FFV1 mask extractor (supports `+`-unioned phrases)."""
    phrases = _split_sam_phrases(sam_prompt)

    graph: dict[str, Any] = {
        "1": {
            "class_type": h3_config.MASKED_JOB_LOADER,
            "inputs": {
                "video": scout_video_path,
                "force_rate": h3_config.MASKED_JOB_FORCE_RATE,
                "custom_width": int(width),
                "custom_height": int(height),
                "frame_load_cap": int(length),
                "start_time": 0.0,
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
            "inputs": {"clip": ["3", 1], "text": phrases[0]},
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
    }
    mask_out_id = "6"
    phrase_mask_sources: list[tuple[int, str, str]] = [(0, phrases[0], "6")]
    for idx, phrase in enumerate(phrases[1:], start=1):
        clip_id = f"4_{idx}"
        track_id = f"5_{idx}"
        to_mask_id = f"6_{idx}"
        union_id = f"6_u{idx}"
        graph[clip_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 1], "text": phrase},
        }
        graph[track_id] = {
            "class_type": "SAM3_VideoTrack",
            "inputs": {
                "images": ["1", 0],
                "model": ["3", 0],
                "conditioning": [clip_id, 0],
                "detection_threshold": float(detection_threshold),
                "max_objects": int(max_objects),
                "detect_interval": int(detect_interval),
            },
        }
        graph[to_mask_id] = {
            "class_type": "SAM3_TrackToMask",
            "inputs": {"track_data": [track_id, 0], "object_indices": ""},
        }
        graph[union_id] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": [mask_out_id, 0],
                "source": [to_mask_id, 0],
                "x": 0,
                "y": 0,
                "operation": "add",
            },
        }
        phrase_mask_sources.append((idx, phrase, to_mask_id))
        mask_out_id = union_id

    if len(phrase_mask_sources) > 1:
        for p_idx, phrase, src_mask_id in phrase_mask_sources:
            slug = _phrase_slug(p_idx, phrase)
            graph[f"7_p{p_idx}"] = {
                "class_type": "MaskToImage",
                "inputs": {"mask": [src_mask_id, 0]},
            }
            graph[f"8_p{p_idx}"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": [f"7_p{p_idx}", 0],
                    "frame_rate": 24.0,
                    "filename_prefix": f"Video/{run_tag}/sam_phrase_{p_idx}_{slug}",
                    **h3_config.MASKED_JOB_LOSSLESS,
                },
            }

    graph["7"] = {
        "class_type": "MaskToImage",
        "inputs": {"mask": [mask_out_id, 0]},
    }
    graph["8"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["7", 0],
            "frame_rate": 24.0,
            "filename_prefix": f"Video/{run_tag}/tortoise_mask_tight",
            **h3_config.MASKED_JOB_LOSSLESS,
        },
    }
    return graph


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
    streams: str = DEFAULT_STREAMS,
    audio_scale: float = DEFAULT_AUDIO_SCALE,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    run_tag: str = RUN_TAG,
    mask_mode: str | None = None,
    mask_video_path: str = f"$H3_OUTPUT_DIR/Video/{RUN_TAG}/tortoise_mask_tight.mkv",
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
    eg_inputs["streams"] = str(streams)
    eg_inputs["audio_scale"] = float(audio_scale)

    if mask_mode is not None:
        g["200"] = {
            "class_type": h3_config.MASKED_JOB_LOADER,
            "inputs": {
                "video": mask_video_path,
                "force_rate": h3_config.MASKED_JOB_FORCE_RATE,
                "custom_width": int(width),
                "custom_height": int(height),
                "frame_load_cap": int(length),
                "start_time": 0.0,
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
                "inputs": {"image": ["203", 0], "blur_radius": 12, "sigma": float(blur_sigma)},
            }
            g["205"] = {
                "class_type": "ImageToMask",
                "inputs": {"image": ["204", 0], "channel": "red"},
            }
            g["206"] = {"class_type": "InvertMask", "inputs": {"mask": ["205", 0]}}
            eg_inputs["mask"] = ["206", 0]

    _wire_provenance_and_output(g, note=f"phaseA:{arm}", arm=arm, run_tag=run_tag)
    return g


def build_all_graphs(
    args: argparse.Namespace,
    *,
    scout_video_path: str | None = None,
    mask_video_path: str | None = None,
) -> tuple[str, str, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Construct all 5 benchmark graphs and validate edited prompt text."""
    run_tag = str(getattr(args, "run_tag", RUN_TAG) or RUN_TAG)
    streams = str(getattr(args, "streams", DEFAULT_STREAMS) or DEFAULT_STREAMS)
    audio_scale = float(getattr(args, "audio_scale", DEFAULT_AUDIO_SCALE))
    blur_sigma = float(getattr(args, "blur_sigma", DEFAULT_BLUR_SIGMA))
    sam_threshold = float(getattr(args, "sam_threshold", DEFAULT_SAM_THRESHOLD))
    scout_src = scout_video_path or f"$H3_OUTPUT_DIR/Video/{run_tag}/scout_base.mp4"
    mask_src = mask_video_path or f"$H3_OUTPUT_DIR/Video/{run_tag}/tortoise_mask_tight.mkv"

    orig_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    edited_prompt, edit_records = h3_prompt_edit.apply_edits(orig_prompt, args.edits)
    if len(edit_records) != 2 or edited_prompt == orig_prompt:
        raise ValueError(f"Expected 2 applied edit clauses, got {len(edit_records)}: {edit_records!r}")
    graphs = {
        "scout_base": build_scout_graph(
            orig_prompt,
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            run_tag=run_tag,
        ),
        "sam_mask": build_sam_mask_graph(
            scout_video_path=scout_src,
            width=args.width,
            height=args.height,
            length=args.length,
            sam_prompt=args.sam_prompt,
            run_tag=run_tag,
            detection_threshold=sam_threshold,
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
            streams=streams,
            audio_scale=audio_scale,
            blur_sigma=blur_sigma,
            run_tag=run_tag,
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
            streams=streams,
            audio_scale=audio_scale,
            blur_sigma=blur_sigma,
            run_tag=run_tag,
            mask_mode="tight",
            mask_video_path=mask_src,
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
            streams=streams,
            audio_scale=audio_scale,
            blur_sigma=blur_sigma,
            run_tag=run_tag,
            mask_mode="grow24",
            mask_video_path=mask_src,
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

    assert graphs["sam_mask"]["1"]["class_type"] == h3_config.MASKED_JOB_LOADER
    phrases = _split_sam_phrases(args.sam_prompt)
    if len(phrases) > 1:
        assert graphs["sam_mask"]["7"]["inputs"]["mask"] == [f"6_u{len(phrases) - 1}", 0]
        assert graphs["sam_mask"][f"6_u{len(phrases) - 1}"]["class_type"] == "MaskComposite"
        for p_idx in range(len(phrases)):
            assert graphs["sam_mask"][f"8_p{p_idx}"]["class_type"] == "VHS_VideoCombine"
    assert graphs["eg_mask_tight"]["200"]["class_type"] == h3_config.MASKED_JOB_LOADER
    assert graphs["eg_mask_tight"]["9"]["inputs"]["mask"] == ["202", 0]
    assert graphs["eg_mask_tight"]["202"]["class_type"] == "InvertMask"
    assert graphs["eg_mask_grow24"]["200"]["class_type"] == h3_config.MASKED_JOB_LOADER
    assert graphs["eg_mask_grow24"]["9"]["inputs"]["mask"] == ["206", 0]
    assert graphs["eg_mask_grow24"]["202"]["class_type"] == "GrowMask"
    assert graphs["eg_mask_grow24"]["204"]["class_type"] == "ImageBlur"
    assert graphs["eg_mask_grow24"]["206"]["class_type"] == "InvertMask"

    print(
        f"[dry-run OK] 5 graphs validated ({args.width}x{args.height}, length={args.length}, "
        f"seed={args.seed}, scale={args.scale}, first_step={args.first_step}, last_step={args.last_step}, "
        f"streams={getattr(args, 'streams', DEFAULT_STREAMS)}, sam_prompt={args.sam_prompt!r}); "
        f"0 preflight FAIL across 4 H3 graphs + resolved edited prompt ({len(edit_records)} clauses). "
        f"orig={len(orig_prompt)}ch edited={len(edited_prompt)}ch"
    )
    return 0


def _pick_saved_video(
    history_entry: dict[str, Any],
    out_root: Path,
    *,
    expected_ext: str,
    prefer_audio: bool,
    file_prefix: str | None = None,
) -> Path:
    items = comfy_client.collect_saved_files(history_entry)
    candidates: list[Path] = []
    for item in items:
        fn = str(item.get("filename") or "")
        if not fn.endswith(expected_ext):
            continue
        if file_prefix is not None and not (fn.startswith(f"{file_prefix}_") or fn == f"{file_prefix}{expected_ext}"):
            continue
        sub = str(item.get("subfolder") or "")
        path = (out_root / sub / fn) if sub else (out_root / fn)
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No *{expected_ext} saved output matching prefix={file_prefix!r} in history outputs={items!r}")
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


def _measure_audio_dbfs(mp4_path: Path, *, sr: int = 32000) -> dict[str, float]:
    """Decode mono float32 PCM from `mp4_path` and report RMS, peak, and >=2.5 kHz dBFS."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(mp4_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"ffmpeg audio decode failed on {mp4_path.name}: {res.stderr.decode('utf-8', 'replace')}")
    pcm = np.frombuffer(res.stdout, dtype=np.float32).astype(np.float64)
    n = int(pcm.size)
    if n == 0:
        return {"rms_dbfs": float("-inf"), "peak_dbfs": float("-inf"), "hi_2k5_dbfs": float("-inf")}
    rms = math.sqrt(float(np.mean(pcm * pcm)))
    peak = float(np.max(np.abs(pcm)))
    spec = np.fft.rfft(pcm)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    weights = np.full(spec.shape, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    hi_mask = freqs >= 2500.0
    hi_ms = float(np.sum((spec.real[hi_mask] ** 2 + spec.imag[hi_mask] ** 2) * weights[hi_mask])) / (float(n) * float(n))
    return {
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-12)), 2),
        "hi_2k5_dbfs": round(10.0 * math.log10(max(hi_ms, 1e-24)), 2),
    }


def _collect_provenance_stamp(prov_dir: Path, existing: set[Path], arm: str) -> dict[str, Any] | None:
    if not prov_dir.is_dir():
        return None
    target_note = f"phaseA:{arm}"
    for sp in sorted(set(prov_dir.glob("stamp_*.json")) - existing):
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("note") == target_note:
            return {
                "file": f"$H3_OUTPUT_DIR/provenance/{sp.name}",
                "note": data.get("note"),
                "sol": data.get("sol"),
                "builds_h3_explorations": (data.get("builds") or {}).get("h3_explorations"),
                "resolved": data.get("resolved"),
            }
    return None


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


def _histogram_p95(hist: np.ndarray) -> float:
    total = int(hist.sum())
    if total == 0:
        return float("nan")
    target = 0.95 * (total - 1)
    cdf = np.cumsum(hist)
    return float(int(np.searchsorted(cdf - 1, target, side="left")))


def _dark_centroid(frame_rgb: np.ndarray, region_mask: np.ndarray) -> tuple[float, float] | None:
    luma = (
        0.299 * frame_rgb[..., 0].astype(np.float32)
        + 0.587 * frame_rgb[..., 1].astype(np.float32)
        + 0.114 * frame_rgb[..., 2].astype(np.float32)
    )
    ys, xs = np.nonzero((luma < 85.0) & region_mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _compute_holdback_and_pose_metrics(
    scout_rgb: np.ndarray,
    unmasked_rgb: np.ndarray,
    arm_rgb: np.ndarray,
    tight_mask_u8: np.ndarray,
) -> dict[str, float]:
    """Measure within-run hold-back (`vs eg_unmasked`), frost, annulus snow, and `vs scout_base` stride shift."""
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dil49_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    dil65_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))

    core_px = rim_px = ann_px = out_px = fg_px = 0
    core_luma_sum = 0.0
    core_frost_cnt = rim_frost_cnt = ann_snow_cnt = out_snow_cnt = 0
    core_mae_unm_sum = core_mae_sct_sum = inside_mae_unm_sum = 0.0
    core_hist = np.zeros(256, dtype=np.int64)
    centroid_shifts: list[float] = []

    for f_idx in range(arm_rgb.shape[0]):
        fg_u8 = (tight_mask_u8[f_idx] >= 128).astype(np.uint8)
        fg = fg_u8 > 0
        core = cv2.erode(fg_u8, erode_k) > 0
        rim = fg & (~core)
        dil49 = cv2.dilate(fg_u8, dil49_k) > 0
        ann24 = dil49 & (~fg)
        out49 = ~dil49
        dil65 = cv2.dilate(fg_u8, dil65_k) > 0

        arm_f = arm_rgb[f_idx].astype(np.float32)
        luma = 0.299 * arm_f[..., 0] + 0.587 * arm_f[..., 1] + 0.114 * arm_f[..., 2]
        diff_unm = np.abs(arm_f - unmasked_rgb[f_idx].astype(np.float32)).mean(axis=-1)
        diff_sct = np.abs(arm_f - scout_rgb[f_idx].astype(np.float32)).mean(axis=-1)

        if np.any(fg):
            fg_px += int(fg.sum())
            inside_mae_unm_sum += float(diff_unm[fg].sum())
        if np.any(core):
            core_vals = luma[core]
            core_px += int(core_vals.size)
            core_luma_sum += float(core_vals.sum())
            core_frost_cnt += int((core_vals >= 185.0).sum())
            core_mae_unm_sum += float(diff_unm[core].sum())
            core_mae_sct_sum += float(diff_sct[core].sum())
            core_hist += np.bincount(np.clip(np.rint(core_vals), 0, 255).astype(np.uint8), minlength=256)
        if np.any(rim):
            rim_px += int(rim.sum())
            rim_frost_cnt += int((luma[rim] >= 185.0).sum())
        if np.any(ann24):
            ann_px += int(ann24.sum())
            ann_snow_cnt += int((luma[ann24] >= 185.0).sum())
        if np.any(out49):
            out_px += int(out49.sum())
            out_snow_cnt += int((luma[out49] >= 185.0).sum())

        c_sct = _dark_centroid(scout_rgb[f_idx], dil65)
        c_arm = _dark_centroid(arm_rgb[f_idx], dil65)
        if c_sct is not None and c_arm is not None:
            centroid_shifts.append(math.hypot(c_arm[0] - c_sct[0], c_arm[1] - c_sct[1]))

    return {
        "inside_mae_vs_unmasked": round(inside_mae_unm_sum / fg_px, 3) if fg_px else float("nan"),
        "eroded_core_mae_vs_unmasked": round(core_mae_unm_sum / core_px, 3) if core_px else float("nan"),
        "eroded_core_mae_vs_scout": round(core_mae_sct_sum / core_px, 3) if core_px else float("nan"),
        "eroded_core_luma_mean": round(core_luma_sum / core_px, 2) if core_px else float("nan"),
        "eroded_core_luma_p95": _histogram_p95(core_hist),
        "eroded_core_frost_185_pct": round(core_frost_cnt / core_px * 100.0, 3) if core_px else float("nan"),
        "tight_rim_frost_185_pct": round(rim_frost_cnt / rim_px * 100.0, 3) if rim_px else float("nan"),
        "annulus_24px_snow_185_pct": round(ann_snow_cnt / ann_px * 100.0, 3) if ann_px else float("nan"),
        "outside_49px_snow_185_pct": round(out_snow_cnt / out_px * 100.0, 3) if out_px else float("nan"),
        "dark_centroid_shift_mean_px": round(float(np.mean(centroid_shifts)), 2) if centroid_shifts else float("nan"),
        "dark_centroid_shift_max_px": round(float(np.max(centroid_shifts)), 2) if centroid_shifts else float("nan"),
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


def _write_tortoise_crop_sheet(
    out_dir: Path,
    arm_videos: dict[str, np.ndarray],
    tight_mask_u8: np.ndarray,
    *,
    frame_indices: tuple[int, ...] = (0, 40, 80, 120),
) -> str:
    """Write a native 1:1 crop sheet around the tortoise + candle across key frames under `out_dir`."""
    n_frames, h, w = tight_mask_u8.shape
    selected = [idx for idx in frame_indices if 0 <= idx < n_frames]
    if not selected:
        selected = [0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    pad = 96
    boxes: list[tuple[int, int, int, int]] = []
    for idx in selected:
        ys, xs = np.nonzero(tight_mask_u8[idx] >= 128)
        if xs.size == 0 or ys.size == 0:
            x0, y0, x1, y1 = max(0, w // 2 - 160), max(0, h // 2 - 120), min(w, w // 2 + 160), min(h, h // 2 + 120)
        else:
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(w, int(xs.max()) + 1 + pad)
            y1 = min(h, int(ys.max()) + 1 + pad)
        boxes.append((x0, y0, x1, y1))

    cell_w = max(x1 - x0 for x0, _, x1, _ in boxes)
    cell_h = max(y1 - y0 for _, y0, _, y1 in boxes)
    columns = ["contours", "scout_base", "eg_unmasked", "eg_mask_tight", "eg_mask_grow24"]
    gutter = 6
    sheet_w = len(columns) * cell_w + (len(columns) + 1) * gutter
    sheet_h = len(selected) * cell_h + (len(selected) + 1) * gutter
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for r_idx, (f_idx, (x0, y0, x1, y1)) in enumerate(zip(selected, boxes)):
        fg = (tight_mask_u8[f_idx] >= 128).astype(np.uint8)
        dil = cv2.dilate(fg, kernel)
        contour_rgb = arm_videos["scout_base"][f_idx].copy()
        tight_cnt, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grow_cnt, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_rgb, tight_cnt, -1, (0, 255, 255), 2)
        cv2.drawContours(contour_rgb, grow_cnt, -1, (255, 255, 0), 2)

        dx = (cell_w - (x1 - x0)) // 2
        dy = (cell_h - (y1 - y0)) // 2
        oy = gutter + r_idx * (cell_h + gutter)
        for c_idx, col in enumerate(columns):
            ox = gutter + c_idx * (cell_w + gutter)
            src = contour_rgb if col == "contours" else arm_videos[col][f_idx]
            crop = Image.fromarray(src[y0:y1, x0:x1])
            sheet.paste(crop, (ox + dx, oy + dy))
            tag = f"f{f_idx:03d} | {col}"
            draw.rectangle([ox + 4, oy + 4, ox + 10 + 7 * len(tag), oy + 20], fill=(0, 0, 0))
            draw.text((ox + 8, oy + 6), tag, fill=(255, 255, 255))

    out_name = "ab_tortoise_crops_f000_f040_f080_f120.png"
    sheet.save(out_dir / out_name)
    return out_name


def _find_existing_output(out_dir: Path, prefix: str, *, expected_ext: str, prefer_audio: bool) -> Path | None:
    if not out_dir.is_dir():
        return None
    if expected_ext == ".mp4" and prefer_audio:
        audio_hits = sorted(out_dir.glob(f"{prefix}_*-audio.mp4"))
        if audio_hits:
            return audio_hits[-1]
    hits = sorted(out_dir.glob(f"{prefix}_*{expected_ext}"))
    return hits[-1] if hits else None


def _expand_output_path(raw: str, out_root: Path) -> Path:
    root_str = str(out_root).rstrip("/")
    return Path(raw.replace("$H3_OUTPUT_DIR", root_str))


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _canonical_graph_for_resume(graph: dict[str, Any], out_root: Path, run_tag: str) -> str:
    scrubbed = _scrub_paths(copy.deepcopy(graph), out_root)
    return json.dumps(scrubbed, sort_keys=True).replace(f"Video/{run_tag}/", "Video/<RUN_TAG>/")


def _mask_f0_bbox(mask_f0: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask_f0 >= 128)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _verify_sam_phrases(
    out_dir: Path,
    phrases: list[str],
    size: tuple[int, int],
    length: int,
    sam_threshold: float,
) -> list[dict[str, Any]]:
    """Verify every `+`-split SAM phrase mask is non-empty on frame 0 and spatially plausible."""
    if len(phrases) <= 1:
        return []
    stats: list[dict[str, Any]] = []
    bboxes: dict[str, list[int]] = {}
    for p_idx, phrase in enumerate(phrases):
        prefix = f"sam_phrase_{p_idx}_{_phrase_slug(p_idx, phrase)}"
        p_path = _find_existing_output(out_dir, prefix, expected_ext=".mkv", prefer_audio=False)
        if p_path is None:
            raise RuntimeError(
                f"Missing per-phrase SAM3 mask output {prefix}_*.mkv in {out_dir}; cannot verify frame 0 for {phrase!r}"
            )
        p_mask = _load_gray_frames(p_path, size, length)
        f0_px = int((p_mask[0] >= 128).sum())
        nz_frames = int(np.sum(np.max(p_mask, axis=(1, 2)) >= 128))
        area_pct = round(float((p_mask >= 128).mean() * 100.0), 3)
        bbox = _mask_f0_bbox(p_mask[0])
        if f0_px == 0 or nz_frames == 0 or bbox is None:
            raise RuntimeError(
                f"SAM3 phrase {phrase!r} returned an empty mask on frame 0 "
                f"(frame0={f0_px} px, nonzero_frames={nz_frames}/{length}, threshold={sam_threshold}); "
                f"aborting before EditGuider arms"
            )
        bboxes[phrase.lower()] = bbox
        stats.append({
            "phrase": phrase,
            "file": str(p_path),
            "frame0_pixels": f0_px,
            "frame0_bbox": bbox,
            "nonzero_frames": nz_frames,
            "mean_area_pct": area_pct,
        })

    if "tortoise" in bboxes and "candle" in bboxes:
        tx0, ty0, tx1, _ = bboxes["tortoise"]
        cx0, cy0, cx1, _ = bboxes["candle"]
        if cx0 > tx1 or cx1 < tx0 or cy0 >= ty0:
            raise RuntimeError(
                f"SAM3 'candle' frame-0 mask bbox {bboxes['candle']} does not sit above/over "
                f"'tortoise' bbox {bboxes['tortoise']}; aborting before EditGuider arms"
            )
    return stats


def run_live_ab(args: argparse.Namespace) -> int:
    """Submit missing graphs sequentially to live `:8188`, analyze, and stack review strips."""
    if not comfy_client.has_input("MiniMaxH3EditGuider", "mask", base_url=args.url):
        print(
            "ERROR: live ComfyUI MiniMaxH3EditGuider does not advertise 'mask' yet; reload :8188 first.",
            file=sys.stderr,
        )
        return 2
    q = comfy_client.get_queue(base_url=args.url)
    if (q.get("queue_running") or q.get("queue_pending")) and not args.resume:
        raise RuntimeError(f"ComfyUI queue is not empty: {q}")

    run_tag = str(args.run_tag or RUN_TAG)
    out_root = h3_config.output_dir()
    out_dir = out_root / "Video" / run_tag
    prov_dir = out_root / "provenance"
    data_dir = _REPO_ROOT / "data" / run_tag
    if not args.resume and ((data_dir / "ab_summary.json").exists() or (data_dir / "prompts.json").exists()):
        raise RuntimeError(
            f"data/{run_tag}/ already contains run records; choose a fresh --run-tag (or pass --resume)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sam_threshold = float(getattr(args, "sam_threshold", DEFAULT_SAM_THRESHOLD))
    phrases = _split_sam_phrases(args.sam_prompt)
    saved_prompts = _load_json_dict(data_dir / "prompts.json") if args.resume else {}
    saved_summary = _load_json_dict(data_dir / "ab_summary.json") if args.resume else {}
    saved_graphs: dict[str, Any] = saved_prompts.get("graphs") if isinstance(saved_prompts.get("graphs"), dict) else {}
    saved_files_prev: dict[str, str] = (
        saved_prompts.get("saved_files") if isinstance(saved_prompts.get("saved_files"), dict) else {}
    )
    saved_arms_prev: dict[str, Any] = saved_summary.get("arms") if isinstance(saved_summary.get("arms"), dict) else {}

    orig_prompt, edited_prompt, edit_records, graphs = build_all_graphs(args)
    queued_graphs: dict[str, Any] = {}
    saved_arm_files: dict[str, str] = {}

    def _flush_prompts_record() -> None:
        prompts_record: dict[str, Any] = {
            "run_tag": run_tag,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "length": args.length,
            "scale": args.scale,
            "first_step": args.first_step,
            "last_step": args.last_step,
            "streams": args.streams,
            "audio_scale": args.audio_scale,
            "blur_sigma": args.blur_sigma,
            "sam_prompt": args.sam_prompt,
            "sam_threshold": sam_threshold,
            "original_prompt": orig_prompt,
            "edits": args.edits,
            "edited_prompt": edited_prompt,
            "edit_records": edit_records,
            "saved_files": saved_arm_files,
            "graphs": queued_graphs,
        }
        (data_dir / "prompts.json").write_text(
            json.dumps(_scrub_paths(prompts_record, out_root), indent=2),
            encoding="utf-8",
        )

    def _can_reuse(name: str, graph: dict[str, Any], candidate: Path) -> bool:
        if name == "sam_mask" and len(phrases) > 1:
            missing = [
                f"sam_phrase_{p_idx}_{_phrase_slug(p_idx, p)}"
                for p_idx, p in enumerate(phrases)
                if _find_existing_output(
                    out_dir, f"sam_phrase_{p_idx}_{_phrase_slug(p_idx, p)}", expected_ext=".mkv", prefer_audio=False
                ) is None
            ]
            if missing:
                print(f"[resume] {name} -> ignoring {candidate.name} (missing per-phrase outputs {missing}); re-running", flush=True)
                return False
        if name in saved_graphs:
            if _canonical_graph_for_resume(graph, out_root, run_tag) == _canonical_graph_for_resume(saved_graphs[name], out_root, run_tag):
                return True
            print(f"[resume] {name} -> ignoring stale {candidate.name} (saved graph settings differ); re-running", flush=True)
            return False
        if name == "scout_base" and not saved_graphs:
            baseline_ok = (
                orig_prompt == PROMPT_FILE.read_text(encoding="utf-8").strip()
                and args.width == DEFAULT_WIDTH
                and args.height == DEFAULT_HEIGHT
                and args.length == DEFAULT_LENGTH
                and args.seed == DEFAULT_SEED
            )
            if baseline_ok:
                print(
                    f"[resume] scout_base -> reusing copied baseline {candidate.name} "
                    f"(matched default prompt/{args.width}x{args.height}/length={args.length}/seed={args.seed})",
                    flush=True,
                )
                return True
            print(f"[resume] scout_base -> ignoring {candidate.name} (prompt/geometry/seed differ from baseline); re-running", flush=True)
            return False
        print(f"[resume] {name} -> ignoring unverified {candidate.name} (no saved settings in prompts.json); re-running", flush=True)
        return False

    def _submit_and_wait(
        name: str,
        graph: dict[str, Any],
        expected_ext: str,
        *,
        file_prefix: str,
        prefer_audio: bool,
        has_stamp: bool,
    ) -> tuple[str, float, float | None, Path, dict[str, Any] | None]:
        queued_graphs[name] = graph
        if args.resume:
            existing: Path | None = None
            prev_raw = saved_files_prev.get(name)
            if isinstance(prev_raw, str) and prev_raw:
                prev_path = _expand_output_path(prev_raw, out_root)
                if prev_path.is_file():
                    existing = prev_path
            if existing is None:
                existing = _find_existing_output(
                    out_dir, file_prefix, expected_ext=expected_ext, prefer_audio=prefer_audio
                )
            if existing is not None and _can_reuse(name, graph, existing):
                prev_arm = saved_arms_prev.get(name) if isinstance(saved_arms_prev.get(name), dict) else {}
                prev_exec_s = prev_arm.get("execution_s")
                prev_stamp = prev_arm.get("provenance") if has_stamp else None
                saved_arm_files[name] = str(existing)
                _flush_prompts_record()
                if name in saved_graphs:
                    print(f"[resume] {name} -> reusing verified {existing.name}", flush=True)
                return "reused", 0.0, prev_exec_s, existing, prev_stamp
        existing_stamps = set(prov_dir.glob("stamp_*.json")) if (has_stamp and prov_dir.is_dir()) else set()
        t0 = time.monotonic()
        pid = comfy_client.submit_prompt(graph, base_url=args.url)
        print(f"[queue] {name} -> {pid}", flush=True)
        entry = comfy_client.wait_for_prompt(pid, base_url=args.url)
        elapsed = round(time.monotonic() - t0, 2)
        execution_s = comfy_client.execution_duration_s(entry)
        saved = _pick_saved_video(
            entry, out_root, expected_ext=expected_ext, prefer_audio=prefer_audio, file_prefix=file_prefix
        )
        stamp = _collect_provenance_stamp(prov_dir, existing_stamps, name) if has_stamp else None
        saved_arm_files[name] = str(saved)
        _flush_prompts_record()
        exec_note = f" (gpu {execution_s:.2f}s)" if execution_s is not None else ""
        print(f"[done] {name} in {elapsed:.2f}s{exec_note} -> {saved.name}", flush=True)
        return pid, elapsed, execution_s, saved, stamp

    results: dict[str, Any] = {
        "run_tag": run_tag,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "length": args.length,
        "scale": args.scale,
        "first_step": args.first_step,
        "last_step": args.last_step,
        "streams": args.streams,
        "audio_scale": args.audio_scale,
        "blur_sigma": args.blur_sigma,
        "sam_prompt": args.sam_prompt,
        "sam_threshold": sam_threshold,
        "arms": {},
    }

    pid, elapsed, exec_s, scout_mp4, scout_stamp = _submit_and_wait(
        "scout_base", graphs["scout_base"], ".mp4", file_prefix="scout_base", prefer_audio=True, has_stamp=True
    )
    scout_audio = _measure_audio_dbfs(scout_mp4)
    results["arms"]["scout_base"] = {
        "prompt_id": pid,
        "elapsed_s": elapsed,
        "execution_s": exec_s,
        "file": str(scout_mp4),
        "provenance": scout_stamp,
        **scout_audio,
    }

    sam_graph = build_sam_mask_graph(
        scout_video_path=str(scout_mp4),
        width=args.width,
        height=args.height,
        length=args.length,
        sam_prompt=args.sam_prompt,
        run_tag=run_tag,
        detection_threshold=sam_threshold,
    )
    pid, elapsed, exec_s, sam_mkv, _ = _submit_and_wait(
        "sam_mask", sam_graph, ".mkv", file_prefix="tortoise_mask_tight", prefer_audio=False, has_stamp=False
    )
    tight_mask = _load_gray_frames(sam_mkv, (args.width, args.height), args.length)
    mask_nonzero_frames = int(np.sum(np.max(tight_mask, axis=(1, 2)) >= 128))
    combined_f0_px = int((tight_mask[0] >= 128).sum())
    if mask_nonzero_frames == 0 or combined_f0_px == 0:
        raise RuntimeError(
            f"SAM3_VideoTrack returned an empty mask (frame0={combined_f0_px} px, nonzero_frames={mask_nonzero_frames}); "
            "aborting before EditGuider arms"
        )
    phrase_stats = _verify_sam_phrases(out_dir, phrases, (args.width, args.height), args.length, sam_threshold)
    results["arms"]["sam_mask"] = {
        "prompt_id": pid,
        "elapsed_s": elapsed,
        "execution_s": exec_s,
        "file": str(sam_mkv),
        "frame0_pixels": combined_f0_px,
        "nonzero_frames": mask_nonzero_frames,
        "mean_area_pct": round(float((tight_mask >= 128).mean() * 100.0), 3),
        "phrases": phrase_stats,
    }

    scout_rgb = _load_rgb_frames(scout_mp4, (args.width, args.height), args.length)
    arm_videos: dict[str, np.ndarray] = {"scout_base": scout_rgb}

    for arm, mask_mode in (("eg_unmasked", None), ("eg_mask_tight", "tight"), ("eg_mask_grow24", "grow24")):
        arm_graph = build_edit_guider_graph(
            arm=arm,
            orig_prompt=orig_prompt,
            edits=args.edits,
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            scale=args.scale,
            first_step=args.first_step,
            last_step=args.last_step,
            streams=args.streams,
            audio_scale=args.audio_scale,
            blur_sigma=args.blur_sigma,
            run_tag=run_tag,
            mask_mode=mask_mode,
            mask_video_path=str(sam_mkv),
        )
        pid, elapsed, exec_s, arm_mp4, arm_stamp = _submit_and_wait(
            arm, arm_graph, ".mp4", file_prefix=arm, prefer_audio=True, has_stamp=True
        )
        arm_rgb = _load_rgb_frames(arm_mp4, (args.width, args.height), args.length)
        arm_videos[arm] = arm_rgb
        metrics = _compute_region_metrics(scout_rgb, arm_rgb, tight_mask)
        holdback = _compute_holdback_and_pose_metrics(scout_rgb, arm_videos["eg_unmasked"], arm_rgb, tight_mask)
        audio_metrics = _measure_audio_dbfs(arm_mp4)
        results["arms"][arm] = {
            "prompt_id": pid,
            "elapsed_s": elapsed,
            "execution_s": exec_s,
            "file": str(arm_mp4),
            "provenance": arm_stamp,
            **metrics,
            **holdback,
            **audio_metrics,
        }

    results["arms"]["scout_base"].update(
        _compute_holdback_and_pose_metrics(scout_rgb, arm_videos["eg_unmasked"], scout_rgb, tight_mask)
    )

    review_stack = _write_2x2_review_videos(out_dir, arm_videos, width=args.width, height=args.height)
    crop_sheet = _write_tortoise_crop_sheet(out_dir, arm_videos, tight_mask)
    results["review"] = {**review_stack, "crop_sheet": crop_sheet}

    _flush_prompts_record()
    (data_dir / "ab_summary.json").write_text(
        json.dumps(_scrub_paths(results, out_root), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {data_dir / 'ab_summary.json'} and review artifacts under $H3_OUTPUT_DIR/Video/{run_tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate graphs statically (zero writes; default)")
    parser.add_argument("--run", action="store_true", help="Submit all 5 graphs to live :8188 ComfyUI")
    parser.add_argument("--resume", action="store_true", help="Reuse verified matching outputs in --run-tag instead of re-running")
    parser.add_argument("--run-tag", default=RUN_TAG, help="Subdirectory tag under Video/ and data/")
    parser.add_argument("--url", default=comfy_client.DEFAULT_COMFY_URL, help="ComfyUI server base URL")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--first-step", type=int, default=DEFAULT_FIRST_STEP)
    parser.add_argument("--last-step", type=int, default=DEFAULT_LAST_STEP)
    parser.add_argument("--streams", default=DEFAULT_STREAMS, choices=("both", "video_only", "audio_only", "independent"))
    parser.add_argument("--audio-scale", type=float, default=DEFAULT_AUDIO_SCALE)
    parser.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA)
    parser.add_argument("--sam-prompt", default=DEFAULT_SAM_PROMPT)
    parser.add_argument("--sam-threshold", type=float, default=DEFAULT_SAM_THRESHOLD)
    parser.add_argument("--edits", default=DEFAULT_EDITS)
    args = parser.parse_args(argv)
    if args.run:
        return run_live_ab(args)
    return verify_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
