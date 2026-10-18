#!/usr/bin/env python3
"""Runtime Meta memory-selection patch and 4-arm scene measurement tool for SAM 3 / SAM 3.1.

Applies Meta's `cal_mem_score` + `frame_filter` memory selection (`mf_threshold = 0.01`)
at runtime to ComfyUI core's `SAM3Tracker` (`sam3_fp32.safetensors`) and `SAM31Tracker`
(`sam3.1_multiplex_fp32.safetensors`) via `ModelPatcher.add_object_patch` and
`CallbacksMP.ON_CLONE`, without editing any ComfyUI file. The patch is the node's
(`sam31_corrections.py`, imported here), so what this measures is what the chain runs.
`--face-probe` runs only the kitchen face probe (680-760 of `lead_fp_0613g`) on SAM 3.1,
through the node's own `attach` (`face_probe_through_node`).

Memory scores (`eff_iou_score`) are computed and stored per object (shape `[N_obj]`),
never averaged with `.mean()` across objects or multiplex slots, so a hidden target
tracked in a multi-object group is not admitted into memory on the strength of its
visible companions' scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = REPO_ROOT.parent.parent
for _p in (REPO_ROOT, COMFY_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comfy.cli_args

if not torch.cuda.is_available():
    comfy.cli_args.args.cpu = True

# The patch itself is the node's (`sam31_corrections.py`, "memory selection"): one implementation, which the chain's
# `MiniMaxH3SAM31Corrections` attaches to SAM 3.1 and this tool attaches to SAM 3 and SAM 3.1 alike, to measure both.
from sam31_corrections import (  # noqa: E402,F401
    MF_THRESHOLD,
    IoUTap,
    MemorySelectionStep,
    apply_memory_selection,
    attach_memory_selection,
    cal_mem_score_per_object,
    frame_filter_per_object,
    get_memory_selection_state,
)

SAM31_CHECKPOINT = "sam3.1_multiplex_fp32.safetensors"
SAM3_BASE_CHECKPOINT = "sam3_fp32.safetensors"


def track_masks_with_sam(
    model: Any,
    images: torch.Tensor,
    initial_masks: torch.Tensor,
) -> dict[str, Any]:
    """Run video tracking on `model` (`SAM3Tracker` or `SAM31Tracker`).

    Args:
        model: ComfyUI `ModelPatcher` wrapping `SAM3Model` or `SAM31Model`.
        images: `[N, H, W, 3]` float32 RGB frames in `[0, 1]`.
        initial_masks: `[N_obj, H, W]` or `[N_obj, 1, H, W]` binary seed masks on frame 0.

    Returns:
        dict with:
          - `masks`: bool numpy array `[N, N_obj, H, W]` at the input `(H, W)` resolution
          - `frame_scores`: dict[int, list[float]] per-frame `eff_iou_score` if memory selection ran
          - `selected_history`: dict[int, list[list[int]]] per-frame selected valid indices if memory selection ran
          - `elapsed_s`: wall time in seconds
    """
    import comfy.model_management
    from comfy.ldm.sam3.tracker import unpack_masks

    H, W = int(images.shape[1]), int(images.shape[2])
    images_nchw = images.permute(0, 3, 1, 2).contiguous()
    if initial_masks.dim() == 2:
        initial_masks = initial_masks.unsqueeze(0).unsqueeze(0)
    elif initial_masks.dim() == 3:
        initial_masks = initial_masks.unsqueeze(1)

    comfy.model_management.load_model_gpu(model)
    diff_model = model.model.diffusion_model

    model.patch_model(load_weights=False)
    try:
        active_step = get_memory_selection_state(model)
        if active_step is not None:
            active_step.reset_state(clear_runs=True)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = diff_model.forward_video(
                images_nchw,
                initial_masks=initial_masks,
                new_det_thresh=1.0,
                max_objects=int(initial_masks.shape[0]),
                detect_interval=1,
                pbar=None,
                target_device=model.load_device,
                target_dtype=model.model.manual_cast_dtype or model.model_dtype(),
            )
        elapsed_s = time.perf_counter() - t0

        frame_scores_out: dict[int, list[float]] = {}
        selected_out: dict[int, list[list[int]]] = {}
        if active_step is not None:
            frame_scores_out = {
                int(k): [round(float(x), 6) for x in v.tolist()]
                for k, v in sorted(active_step.merged_frame_scores().items())
            }
            selected_out = {
                int(k): [[int(i) for i in obj_list] for obj_list in v]
                for k, v in sorted(active_step.merged_selected_history().items())
            }
    finally:
        model.unpatch_model(unpatch_weights=False)

    if isinstance(result, dict):
        packed = result["packed_masks"]
        if packed is None:
            masks_out = np.zeros((images.shape[0], initial_masks.shape[0], H, W), dtype=bool)
        else:
            binary = unpack_masks(packed)  # [N, N_obj, 1008, 1008]
            n_f, n_o = int(binary.shape[0]), int(binary.shape[1])
            flat = binary.reshape(n_f * n_o, 1, binary.shape[2], binary.shape[3]).float()
            resized = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False) > 0.5
            masks_out = resized.reshape(n_f, n_o, H, W).cpu().numpy()
    elif isinstance(result, torch.Tensor):
        # SAM3Tracker.track_video returns float mask logits [N, N_obj, 1008, 1008]
        n_f, n_o = int(result.shape[0]), int(result.shape[1])
        flat = result.reshape(n_f * n_o, 1, result.shape[2], result.shape[3]).float()
        resized = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False) > 0.0
        masks_out = resized.reshape(n_f, n_o, H, W).cpu().numpy()
    else:
        raise TypeError(f"Unexpected forward_video return type: {type(result)!r}")

    return {
        "masks": masks_out,
        "frame_scores": frame_scores_out,
        "selected_history": selected_out,
        "elapsed_s": round(elapsed_s, 3),
    }


def compute_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute binary IoU between two 2D boolean masks."""
    a_b = np.asarray(a, dtype=bool)
    b_b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a_b, b_b).sum())
    union = int(np.logical_or(a_b, b_b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def compute_mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Return `(cx, cy)` centroid in pixel coordinates, or `None` if mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (round(float(xs.mean()), 2), round(float(ys.mean()), 2))


def summarize_track_series(
    masks: np.ndarray,
    obj_idx: int = 0,
    first_source_frame: int = 0,
    stride: int = 1,
    companion_masks: dict[str, np.ndarray] | None = None,
    baseline_masks: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build per-frame diagnostics (area, centroid, IoU vs baseline and companions).

    `masks` has shape `[N, N_obj, H, W]` (or `[N, H, W]`). When `stride == -1`
    (backward propagation from a later `pick_frame`), frame `i` maps to
    `first_source_frame - i`; when `stride == 1`, `first_source_frame + i`.
    """
    if masks.ndim == 4:
        series = masks[:, obj_idx]
    else:
        series = masks
    rows: list[dict[str, Any]] = []
    for i in range(series.shape[0]):
        src_f = first_source_frame + i * stride
        m = series[i]
        area = int(m.sum())
        centroid = compute_mask_centroid(m)
        row: dict[str, Any] = {
            "step_idx": i,
            "source_frame": src_f,
            "area_px": area,
            "centroid_xy": list(centroid) if centroid is not None else None,
        }
        if baseline_masks is not None and i < baseline_masks.shape[0]:
            row["iou_vs_off"] = round(compute_mask_iou(m, baseline_masks[i]), 4)
        if companion_masks:
            comp_ious: dict[str, float] = {}
            for cname, carr in companion_masks.items():
                if i < carr.shape[0]:
                    comp_ious[cname] = round(compute_mask_iou(m, carr[i]), 4)
            row["iou_vs_companions"] = comp_ious
        rows.append(row)
    return rows


def track_bidirectional_from_pick(
    model: Any,
    images: torch.Tensor,
    seed_masks: torch.Tensor,
    pick_frame: int,
) -> dict[str, Any]:
    """Track forward (`pick_frame..N-1`) and backward (`pick_frame..0`), mirroring `SubjectTrack`."""
    n_total = int(images.shape[0])
    n_obj = int(seed_masks.shape[0])
    H, W = int(images.shape[1]), int(images.shape[2])
    full_masks = np.zeros((n_total, n_obj, H, W), dtype=bool)
    full_scores: dict[int, list[float]] = {}
    full_selected: dict[int, list[list[int]]] = {}
    elapsed = 0.0

    # Forward pass: pick_frame .. n_total - 1
    fwd_imgs = images[pick_frame:]
    fwd_out = track_masks_with_sam(model, fwd_imgs, seed_masks)
    full_masks[pick_frame:] = fwd_out["masks"]
    elapsed += fwd_out["elapsed_s"]
    for k, v in fwd_out["frame_scores"].items():
        full_scores[pick_frame + k] = v
    for k, v in fwd_out["selected_history"].items():
        full_selected[pick_frame + k] = [
            [pick_frame + idx for idx in obj_list] for obj_list in v
        ]

    # Backward pass: pick_frame .. 0 (reversed)
    if pick_frame > 0:
        bwd_imgs = images[: pick_frame + 1].flip(0)
        bwd_out = track_masks_with_sam(model, bwd_imgs, seed_masks)
        # bwd_out["masks"][0] is pick_frame; [1:] is pick_frame - 1 down to 0
        full_masks[:pick_frame] = bwd_out["masks"][1:][::-1]
        elapsed += bwd_out["elapsed_s"]
        for k, v in bwd_out["frame_scores"].items():
            full_scores[pick_frame - k] = v
        for k, v in bwd_out["selected_history"].items():
            full_selected[pick_frame - k] = [
                [pick_frame - idx for idx in obj_list] for obj_list in v
            ]

    return {
        "masks": full_masks,
        "frame_scores": dict(sorted(full_scores.items())),
        "selected_history": dict(sorted(full_selected.items())),
        "elapsed_s": round(elapsed, 3),
    }


def _load_video_window(
    video_path: Path,
    first_frame: int,
    num_frames: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Decode `[num_frames, height, width, 3]` float32 RGB frames in `[0, 1]` using PyAV."""
    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx < first_frame:
                continue
            if len(frames) >= num_frames:
                break
            rgb = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            frames.append(rgb)
    if len(frames) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames starting at {first_frame} from {video_path.name}, "
            f"decoded {len(frames)}"
        )
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_video_window_by_seconds(
    video_path: Path,
    start_sec: float,
    num_frames: int,
    fps: float,
    width: int,
) -> torch.Tensor:
    """Decode `num_frames` at `fps` starting at `start_sec` via `LoadVideoFFmpegPath`, matching `subject_alone_or_in_a_group.py`."""
    import types
    import server

    if getattr(server.PromptServer, "instance", None) is None:
        server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    vhs_root = COMFY_ROOT / "custom_nodes" / "ComfyUI-VideoHelperSuite"
    if str(vhs_root) not in sys.path:
        sys.path.insert(0, str(vhs_root))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath

    loaded = LoadVideoFFmpegPath().load_video(
        video=str(video_path),
        force_rate=float(fps),
        custom_width=int(width),
        custom_height=0,
        frame_load_cap=int(num_frames),
        start_time=float(start_sec),
        format="AnimateDiff",
    )[0][..., :3]
    u8 = (loaded.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return torch.from_numpy(u8.astype(np.float32) / 255.0)


def _resolve_input_dir() -> Path:
    env_dir = os.environ.get("H3_INPUT_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        import folder_paths

        fp_in = Path(folder_paths.get_input_directory())
        if fp_in.is_dir():
            return fp_in
    except Exception:
        pass
    candidate = COMFY_ROOT / "input"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Could not locate H3_INPUT_DIR or ComfyUI input directory")


def _resolve_data_dir() -> Path:
    local_data = REPO_ROOT / "data"
    if (local_data / "2026-10-17_lead_fp_0613g").is_dir():
        return local_data
    shared_data = COMFY_ROOT / "custom_nodes" / "ComfyUI-h3-explorations" / "data"
    if shared_data.is_dir():
        return shared_data
    return local_data


FORBIDDEN_CAPTURE_MARKERS: tuple[str, ...] = ("blocked", "superseded")

EXPECTED_SCENE_CAPTURES: dict[str, dict[str, Any]] = {
    "kitchen_604_1009": {
        "capture_folder": "data/2026-10-17_lead_fp_0613g",
        "capture_date": "2026-10-17",
        "load": "lead_fp_0613g",
    },
    "hall_1306_1418": {
        "capture_folder": "data/2026-10-13_lead_fp_1306s",
        "companion_capture_folder": "data/2026-10-13_third_fp_1310c",
        "capture_date": "2026-10-13",
        "load": "lead_fp_1306s, third_fp_1310c",
    },
    "regain_1419_1465": {
        "capture_folder": "data/2026-10-15_lead_fp_1419d",
        "capture_date": "2026-10-15",
        "load": "lead_fp_1419d",
    },
}


def unpack_capture_mask_npz(
    npz_path: Path,
    width: int,
    key: str = "track",
) -> tuple[np.ndarray, str]:
    """Load and unpack a bit-packed `(N, H, W // 8)` mask array from `npz_path` and return `(bool_masks, sha256)`."""
    raw_bytes = npz_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    with np.load(npz_path) as z:
        if key not in z:
            raise KeyError(
                f"Expected key {key!r} in {npz_path.name}; available keys: {sorted(z.keys())}"
            )
        raw = z[key]
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[-1] * 8 >= width and raw.shape[-1] != width:
        unpacked = np.unpackbits(raw, axis=-1)[..., :width].astype(bool)
    else:
        unpacked = (raw > 0.5).astype(bool)
    return unpacked, digest


def validate_scene_capture(scene_name: str, cfg: dict[str, Any]) -> None:
    """Refuse blocked/superseded folders, unexpected captures, or mask paths missing `subjects/<subj>/`."""
    paths_to_check: list[Path] = []
    if "capture_folder" in cfg:
        paths_to_check.append(Path(cfg["capture_folder"]))
    if "companion_capture_folder" in cfg:
        paths_to_check.append(Path(cfg["companion_capture_folder"]))
    if "seed_npz" in cfg:
        paths_to_check.append(Path(cfg["seed_npz"]))
    for comp_entry in cfg.get("companion_npzs", {}).values():
        comp_path = comp_entry["npz"] if isinstance(comp_entry, dict) else comp_entry
        paths_to_check.append(Path(comp_path))
    if "face_subprobe" in cfg and "seed_npz" in cfg["face_subprobe"]:
        paths_to_check.append(Path(cfg["face_subprobe"]["seed_npz"]))

    for p in paths_to_check:
        parts_lower = [part.lower() for part in p.parts]
        for marker in FORBIDDEN_CAPTURE_MARKERS:
            if any(marker in part for part in parts_lower):
                raise ValueError(
                    f"Scene {scene_name!r} references a {marker} capture path ({p.name}); "
                    f"blocked or superseded captures must never be used."
                )

    for folder_key in ("capture_folder", "companion_capture_folder"):
        if folder_key in cfg:
            manifest_path = Path(cfg[folder_key]) / "manifest.json"
            if manifest_path.is_file():
                manifest_data = json.loads(manifest_path.read_text())
                status_str = str(manifest_data.get("status", "")).lower()
                for marker in FORBIDDEN_CAPTURE_MARKERS:
                    if manifest_data.get(marker) is True or marker in status_str:
                        raise ValueError(
                            f"Scene {scene_name!r} capture manifest {manifest_path.parent.name}/manifest.json "
                            f"is marked {marker}; blocked or superseded captures must never be used."
                        )

    if scene_name not in EXPECTED_SCENE_CAPTURES:
        return

    expected = EXPECTED_SCENE_CAPTURES[scene_name]
    exp_folder_name = Path(expected["capture_folder"]).name
    got_folder = Path(cfg.get("capture_folder", ""))
    if got_folder.name != exp_folder_name:
        raise ValueError(
            f"Scene {scene_name!r} must use capture folder {expected['capture_folder']!r}, "
            f"got {got_folder.name!r}."
        )
    if "companion_capture_folder" in expected:
        exp_comp_name = Path(expected["companion_capture_folder"]).name
        got_comp = Path(cfg.get("companion_capture_folder", ""))
        if got_comp.name != exp_comp_name:
            raise ValueError(
                f"Scene {scene_name!r} must use companion capture folder "
                f"{expected['companion_capture_folder']!r}, got {got_comp.name!r}."
            )

    if cfg.get("capture_date") != expected["capture_date"]:
        raise ValueError(
            f"Scene {scene_name!r} capture_date mismatch: expected {expected['capture_date']!r}, "
            f"got {cfg.get('capture_date')!r}."
        )
    if cfg.get("load") != expected["load"]:
        raise ValueError(
            f"Scene {scene_name!r} load mismatch: expected {expected['load']!r}, "
            f"got {cfg.get('load')!r}."
        )

    allowed_folders = {got_folder.resolve()}
    if "companion_capture_folder" in cfg:
        allowed_folders.add(Path(cfg["companion_capture_folder"]).resolve())

    npz_paths: list[Path] = [Path(cfg["seed_npz"])]
    for comp_entry in cfg.get("companion_npzs", {}).values():
        comp_path = comp_entry["npz"] if isinstance(comp_entry, dict) else comp_entry
        npz_paths.append(Path(comp_path))
    if "face_subprobe" in cfg and "seed_npz" in cfg["face_subprobe"]:
        npz_paths.append(Path(cfg["face_subprobe"]["seed_npz"]))

    for npz_p in npz_paths:
        # Must live under `<capture_folder>/subjects/<subj>/masks__*.npz`
        if npz_p.parent.parent.name != "subjects":
            raise ValueError(
                f"Scene {scene_name!r} mask path {npz_p} is missing 'subjects/<subj>/' directory structure."
            )
        cap_root = npz_p.parent.parent.parent.resolve()
        if cap_root not in allowed_folders:
            raise ValueError(
                f"Scene {scene_name!r} mask path {npz_p} belongs to capture {cap_root.name!r}, "
                f"which is not in the scene's explicit capture folders {[f.name for f in allowed_folders]}."
            )


def build_default_scenes() -> dict[str, dict[str, Any]]:
    """Define the 4 benchmark scenes with explicit, unblocked capture folders and `subjects/<subj>/` mask paths."""
    input_dir = _resolve_input_dir()
    data_dir = _resolve_data_dir()
    fun_video = input_dir / "mrpop" / "fun_0000_6394_24p_sound_in_step.mov"
    dance_video = input_dir / "lotsofpeopledance_0414_0720.mkv"

    kitchen_cap = data_dir / "2026-10-17_lead_fp_0613g"
    hall_lead_cap = data_dir / "2026-10-13_lead_fp_1306s"
    hall_third_cap = data_dir / "2026-10-13_third_fp_1310c"
    regain_cap = data_dir / "2026-10-15_lead_fp_1419d"

    return {
        "kitchen_604_1009": {
            "description": (
                "Kitchen 604-1009 (`lead` and companions `second`, `strangers` from `lead_fp_0613g`): "
                "focus windows 684-760 (head-bend occlusion) and 998-1009 (`lead` occluded "
                "behind `second` at 999-1002 and raw track jumps onto `second` at 1003-1009)."
            ),
            "video_path": fun_video,
            "capture_folder": kitchen_cap,
            "capture_date": "2026-10-17",
            "load": "lead_fp_0613g",
            "capture_first_frame": 613,
            "first_source_frame": 604,
            "num_frames": 406,
            "width": 1024,
            "height": 768,
            "pick_frame": 202,  # source frame 806
            "seed_npz": kitchen_cap / "subjects" / "lead" / "masks__shot_c_0604b.npz",
            "seed_frame_in_npz": 193,  # 613 + 193 = 806
            "companion_npzs": {
                "second": {
                    "npz": kitchen_cap / "subjects" / "second" / "masks__shot_c_0604b.npz",
                    "seed_frame_in_npz": 193,
                    "capture_first_frame": 613,
                },
                "strangers": {
                    "npz": kitchen_cap / "subjects" / "strangers" / "masks__shot_c_0604b.npz",
                    "seed_frame_in_npz": 193,
                    "capture_first_frame": 613,
                },
            },
            "focus_windows": [[684, 760], [998, 1009]],
            # Secondary sub-probe on 680-760 seeded on lead's face (`parts`) at source frame 680
            # where `lead`'s face is occluded on 687-707 (21 frames) and recovers at 708..760.
            "face_subprobe": {
                "first_source_frame": 680,
                "num_frames": 81,
                "pick_frame": 0,  # source frame 680
                "seed_npz": kitchen_cap / "subjects" / "lead" / "masks__shot_c_0604b.npz",
                "seed_frame_in_npz": 67,  # 613 + 67 = 680
                "focus_windows": [[684, 760]],
            },
        },
        "hall_1306_1418": {
            "description": (
                "Hall 1306-1418 (`lead` from `lead_fp_1306s` and `third`/dad from `third_fp_1310c`): "
                "seeded at pick_frame=4 (source 1310); `third` is occluded behind `lead` "
                "and `second` in the middle of the shot and steps out on 1392-1418."
            ),
            "video_path": fun_video,
            "capture_folder": hall_lead_cap,
            "companion_capture_folder": hall_third_cap,
            "capture_date": "2026-10-13",
            "load": "lead_fp_1306s, third_fp_1310c",
            "capture_first_frame": 1306,
            "first_source_frame": 1306,
            "num_frames": 113,
            "width": 1024,
            "height": 768,
            "pick_frame": 4,  # source frame 1310
            "seed_npz": hall_lead_cap / "subjects" / "lead" / "masks__lead_c_1306.npz",
            "seed_frame_in_npz": 4,  # 1306 + 4 = 1310
            "companion_npzs": {
                "third": {
                    "npz": hall_third_cap / "subjects" / "third" / "masks__third_c_1306.npz",
                    "seed_frame_in_npz": 0,  # 1310 + 0 = 1310
                    "capture_first_frame": 1310,
                },
            },
            "focus_windows": [[1392, 1418]],
        },
        "regain_1419_1465": {
            "description": (
                "Hall regain 1419-1465 (`lead`, `second`, `third` from `lead_fp_1419d`): "
                "seeded at pick_frame=12 (source 1431), `second` is occluded behind `lead` and `third` "
                "on 1437-1452 (16 frames), and re-latched on 1453-1465."
            ),
            "video_path": fun_video,
            "capture_folder": regain_cap,
            "capture_date": "2026-10-15",
            "load": "lead_fp_1419d",
            "capture_first_frame": 1419,
            "first_source_frame": 1419,
            "num_frames": 47,
            "width": 1024,
            "height": 768,
            "pick_frame": 12,  # source frame 1431
            "seed_npz": regain_cap / "subjects" / "lead" / "masks__shot_c_1419.npz",
            "seed_frame_in_npz": 12,  # 1419 + 12 = 1431
            "companion_npzs": {
                "second": {
                    "npz": regain_cap / "subjects" / "second" / "masks__shot_c_1419.npz",
                    "seed_frame_in_npz": 12,
                    "capture_first_frame": 1419,
                },
                "third": {
                    "npz": regain_cap / "subjects" / "third" / "masks__shot_c_1419.npz",
                    "seed_frame_in_npz": 12,
                    "capture_first_frame": 1419,
                },
            },
            "focus_windows": [[1435, 1465]],
        },
        "lotsofpeople": {
            "description": (
                "`lotsofpeopledance_0414_0720.mkv` (144 frames at 24 fps from 138.0s, width 1344, "
                "seeded on frame 4): tests both the `most central` dancer (lost in group of 16 "
                "around frame 95/115) and the `largest` front-row dancer (jumps to far-right "
                "dancer on frame 20 when tracked alone)."
            ),
            "video_path": dance_video,
            "capture_date": "2026-10-17",
            "load": "lotsofpeopledance_0414_0720_live_sam3_detect",
            "start_sec": 138.0,
            "num_frames": 144,
            "fps": 24.0,
            "width": 1344,
            "seed_frame": 4,
            "focus_windows": [[15, 45], [85, 143]],
        },
    }


def load_sam_model(checkpoint_name: str) -> Any:
    """Load `SAM3Model` or `SAM31Model` via ComfyUI's `load_state_dict_guess_config`."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from bench._lib.sam3_core import TEXT_PROJECTION_KEY

    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_name)
    sd, md = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    sd.pop(TEXT_PROJECTION_KEY, None)
    model, _, _, _ = comfy.sd.load_state_dict_guess_config(
        sd,
        output_vae=False,
        output_clip=False,
        output_model=True,
        metadata=md,
    )
    return model


def _align_capture_track_to_window(
    track_bool: np.ndarray,
    capture_first_frame: int,
    window_first_frame: int,
    window_num_frames: int,
) -> np.ndarray:
    """Place a `(N_cap, H, W)` capture track into the `(window_num_frames, H, W)` scene window."""
    h, w = int(track_bool.shape[1]), int(track_bool.shape[2])
    aligned = np.zeros((window_num_frames, h, w), dtype=bool)
    for i in range(window_num_frames):
        src_f = window_first_frame + i
        cap_idx = src_f - capture_first_frame
        if 0 <= cap_idx < track_bool.shape[0]:
            aligned[i] = track_bool[cap_idx]
    return aligned


def run_scene_four_arms(
    scene_name: str,
    models: dict[str, Any],
    out_dir: Path,
    mf_threshold: float = MF_THRESHOLD,
) -> dict[str, Any]:
    """Run the 4 arms (SAM 3.1 off/on, SAM 3 base off/on; alone and group) on `scene_name`."""
    scenes = build_default_scenes()
    if scene_name not in scenes:
        raise KeyError(f"Unknown scene {scene_name!r}; expected one of {sorted(scenes)}")
    cfg = scenes[scene_name]
    validate_scene_capture(scene_name, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scene_name == "lotsofpeople":
        return _run_lotsofpeople_scene(cfg, models, out_dir, mf_threshold=mf_threshold)

    width = int(cfg["width"])
    height = int(cfg["height"])
    first_src = int(cfg["first_source_frame"])
    num_frames = int(cfg["num_frames"])
    pick_frame = int(cfg["pick_frame"])
    pf_npz = int(cfg["seed_frame_in_npz"])
    cap_first = int(cfg.get("capture_first_frame", first_src))

    images = _load_video_window(
        cfg["video_path"],
        first_frame=first_src,
        num_frames=num_frames,
        width=width,
        height=height,
    )

    lead_npz_path = Path(cfg["seed_npz"])
    lead_masks_bool, lead_sha256 = unpack_capture_mask_npz(lead_npz_path, width=width, key="track")
    lead_seed = torch.from_numpy(lead_masks_bool[pf_npz].astype(np.float32))

    subject_masks_sha256: dict[str, str] = {"lead": lead_sha256}
    ref_companions: dict[str, np.ndarray] = {
        "saved_lead": _align_capture_track_to_window(
            lead_masks_bool,
            capture_first_frame=cap_first,
            window_first_frame=first_src,
            window_num_frames=num_frames,
        )
    }

    comp_seeds: list[torch.Tensor] = []
    comp_seed_by_name: dict[str, dict[str, Any]] = {}
    for comp_name, comp_info in cfg.get("companion_npzs", {}).items():
        comp_path = Path(comp_info["npz"])
        comp_pf = int(comp_info.get("seed_frame_in_npz", pf_npz))
        comp_first = int(comp_info.get("capture_first_frame", cap_first))
        comp_masks_bool, comp_sha = unpack_capture_mask_npz(comp_path, width=width, key="track")
        subject_masks_sha256[comp_name] = comp_sha
        ref_companions[f"saved_{comp_name}"] = _align_capture_track_to_window(
            comp_masks_bool,
            capture_first_frame=comp_first,
            window_first_frame=first_src,
            window_num_frames=num_frames,
        )
        cm = comp_masks_bool[comp_pf].astype(np.float32)
        if cm.sum() > 0:
            comp_tensor = torch.from_numpy(cm)
            comp_idx = 1 + len(comp_seeds)
            comp_seeds.append(comp_tensor)
            comp_seed_by_name[comp_name] = {
                "tensor": comp_tensor,
                "group_obj_idx": comp_idx,
            }

    group_seeds = torch.stack([lead_seed] + comp_seeds, dim=0)  # [1 + C, H, W]
    alone_lead_seed = lead_seed.unsqueeze(0)  # [1, H, W]

    cap_folder_rel = f"data/{Path(cfg['capture_folder']).name}"
    comp_cap_rel = (
        f"data/{Path(cfg['companion_capture_folder']).name}"
        if "companion_capture_folder" in cfg
        else None
    )

    scene_report: dict[str, Any] = {
        "scene": scene_name,
        "description": cfg["description"],
        "capture_folder": cap_folder_rel,
        "companion_capture_folder": comp_cap_rel,
        "capture_date": cfg["capture_date"],
        "load": cfg["load"],
        "masks_sha256": lead_sha256,
        "subject_masks_sha256": subject_masks_sha256,
        "first_source_frame": first_src,
        "num_frames": num_frames,
        "pick_frame": pick_frame,
        "pick_source_frame": first_src + pick_frame,
        "mf_threshold": mf_threshold,
        "n_group_objects": int(group_seeds.shape[0]),
        "focus_windows": cfg["focus_windows"],
        "arms": {},
    }

    # Determine which subjects to track:
    # - In kitchen_604_1009: target is `lead` (obj 0 in group, and alone), plus face_subprobe
    # - In hall_1306_1418: target is `lead` (obj 0) AND `third`/dad (`third` in companion_npzs)
    # - In regain_1419_1465: target is `lead` (obj 0) AND `second` (`second` in companion_npzs)
    targets: list[dict[str, Any]] = [
        {
            "name": "lead",
            "alone_seed": alone_lead_seed,
            "group_obj_idx": 0,
        }
    ]
    if scene_name == "hall_1306_1418" and "third" in comp_seed_by_name:
        targets.append(
            {
                "name": "third_dad",
                "alone_seed": comp_seed_by_name["third"]["tensor"].unsqueeze(0),
                "group_obj_idx": comp_seed_by_name["third"]["group_obj_idx"],
            }
        )
    elif scene_name == "regain_1419_1465" and "second" in comp_seed_by_name:
        targets.append(
            {
                "name": "second",
                "alone_seed": comp_seed_by_name["second"]["tensor"].unsqueeze(0),
                "group_obj_idx": comp_seed_by_name["second"]["group_obj_idx"],
            }
        )

    for model_key, base_model in models.items():
        model_arms: dict[str, Any] = {}
        for mem_on in (False, True):
            # 1) Alone calls for each target subject
            alone_results: dict[str, Any] = {}
            for tgt in targets:
                tname = tgt["name"]
                active_model = (
                    apply_memory_selection(base_model, mf_threshold=mf_threshold, focus_obj_idx=0)
                    if mem_on
                    else base_model.clone()
                )
                res = track_bidirectional_from_pick(
                    active_model, images, tgt["alone_seed"], pick_frame=pick_frame
                )
                alone_results[tname] = res

            # 2) Group call (all subjects together)
            group_results: dict[str, Any] = {}
            is_sam31 = "sam31" in model_key
            if len(comp_seeds) > 0:
                if is_sam31:
                    for tgt in targets:
                        tname = tgt["name"]
                        g_idx = int(tgt["group_obj_idx"])
                        active_model = (
                            apply_memory_selection(
                                base_model, mf_threshold=mf_threshold, focus_obj_idx=g_idx
                            )
                            if mem_on
                            else base_model.clone()
                        )
                        res = track_bidirectional_from_pick(
                            active_model, images, group_seeds, pick_frame=pick_frame
                        )
                        group_results[tname] = res
                else:
                    # SAM3Tracker.track_video loops _track_single_object per mask independently,
                    # so group object `g_idx` is identical to `alone` for that object; we still
                    # record it explicitly so the 4-arm x (alone, group) table is complete.
                    for tgt in targets:
                        tname = tgt["name"]
                        alone_res = alone_results[tname]
                        group_results[tname] = {
                            "masks": alone_res["masks"],
                            "frame_scores": alone_res["frame_scores"],
                            "selected_history": alone_res["selected_history"],
                            "elapsed_s": alone_res["elapsed_s"],
                            "note": "SAM3Tracker.track_video loops _track_single_object per object independently.",
                        }

            model_arms["mem_on" if mem_on else "mem_off"] = {
                "alone": alone_results,
                "group": group_results,
            }
        scene_report["arms"][model_key] = model_arms

    # Optional kitchen face sub-probe (680-760, 21-frame occlusion 687-707)
    if "face_subprobe" in cfg:
        sub = cfg["face_subprobe"]
        sub_imgs = images[
            sub["first_source_frame"] - first_src : sub["first_source_frame"] - first_src + sub["num_frames"]
        ]
        parts_bool, sub_sha256 = unpack_capture_mask_npz(
            Path(sub["seed_npz"]), width=width, key="parts"
        )
        face_arr = parts_bool[int(sub["seed_frame_in_npz"])]
        if face_arr.ndim == 3:
            face_arr = face_arr[0]
        face_seed = torch.from_numpy(face_arr.astype(np.float32)).unsqueeze(0)
        sub_arms: dict[str, Any] = {}
        for model_key, base_model in models.items():
            sub_arms[model_key] = {}
            for mem_on in (False, True):
                active_model = (
                    apply_memory_selection(base_model, mf_threshold=mf_threshold, focus_obj_idx=0)
                    if mem_on
                    else base_model.clone()
                )
                sub_arms[model_key]["mem_on" if mem_on else "mem_off"] = track_masks_with_sam(
                    active_model, sub_imgs, face_seed
                )
        scene_report["face_subprobe"] = {
            "first_source_frame": sub["first_source_frame"],
            "num_frames": sub["num_frames"],
            "masks_sha256": sub_sha256,
            "capture_date": cfg["capture_date"],
            "load": cfg["load"],
            "arms": _format_subprobe_summary(sub_arms, first_src=sub["first_source_frame"]),
        }

    summary = _compile_scene_summary(scene_report, targets, ref_companions)
    out_json = out_dir / f"summary__{scene_name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def _format_subprobe_summary(sub_arms: dict[str, Any], first_src: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_key, mdict in sub_arms.items():
        off_masks = mdict["mem_off"]["masks"][:, 0]
        on_masks = mdict["mem_on"]["masks"][:, 0]
        diffs: list[dict[str, Any]] = []
        for i in range(off_masks.shape[0]):
            iou = compute_mask_iou(on_masks[i], off_masks[i])
            off_area = int(off_masks[i].sum())
            on_area = int(on_masks[i].sum())
            if iou < 0.999 or off_area != on_area:
                diffs.append(
                    {
                        "source_frame": first_src + i,
                        "off_area_px": off_area,
                        "on_area_px": on_area,
                        "off_centroid": compute_mask_centroid(off_masks[i]),
                        "on_centroid": compute_mask_centroid(on_masks[i]),
                        "iou_on_vs_off": round(iou, 4),
                    }
                )
        out[model_key] = {
            "mem_off_elapsed_s": mdict["mem_off"]["elapsed_s"],
            "mem_on_elapsed_s": mdict["mem_on"]["elapsed_s"],
            "frame_scores_on": mdict["mem_on"]["frame_scores"],
            "selected_history_on": {
                str(first_src + k): [[first_src + x for x in o] for o in v]
                for k, v in mdict["mem_on"]["selected_history"].items()
            },
            "changed_frames_count": len(diffs),
            "changed_frames": diffs,
        }
    return out


def _compile_scene_summary(
    scene_report: dict[str, Any],
    targets: list[dict[str, Any]],
    ref_companions: dict[str, np.ndarray],
) -> dict[str, Any]:
    first_src = int(scene_report["first_source_frame"])
    n_frames = int(scene_report["num_frames"])
    focus_windows = scene_report["focus_windows"]

    compiled_arms: dict[str, Any] = {}
    for model_key, mdict in scene_report["arms"].items():
        compiled_arms[model_key] = {}
        for mode in ("alone", "group"):
            compiled_arms[model_key][mode] = {}
            for tgt in targets:
                tname = tgt["name"]
                if tname not in mdict["mem_off"].get(mode, {}):
                    continue
                off_res = mdict["mem_off"][mode][tname]
                on_res = mdict["mem_on"][mode][tname]
                obj_idx = (
                    0
                    if mode == "alone" or off_res["masks"].shape[1] == 1
                    else int(tgt["group_obj_idx"])
                )
                off_m = off_res["masks"][:, obj_idx]
                on_m = on_res["masks"][:, obj_idx]

                # Companion masks from the group run itself (if group) or saved reference
                live_companions: dict[str, np.ndarray] = dict(ref_companions)
                if mode == "group" and on_res["masks"].shape[1] > 1:
                    for o_i in range(on_res["masks"].shape[1]):
                        if o_i != obj_idx:
                            live_companions[f"group_obj_{o_i}_off"] = off_res["masks"][:, o_i]
                            live_companions[f"group_obj_{o_i}_on"] = on_res["masks"][:, o_i]

                off_series = summarize_track_series(
                    off_m, obj_idx=0, first_source_frame=first_src, companion_masks=live_companions
                )
                on_series = summarize_track_series(
                    on_m,
                    obj_idx=0,
                    first_source_frame=first_src,
                    companion_masks=live_companions,
                    baseline_masks=off_m,
                )

                changed_frames = [
                    {
                        "source_frame": on_series[i]["source_frame"],
                        "off_area_px": off_series[i]["area_px"],
                        "on_area_px": on_series[i]["area_px"],
                        "off_centroid_xy": off_series[i]["centroid_xy"],
                        "on_centroid_xy": on_series[i]["centroid_xy"],
                        "iou_vs_off": on_series[i]["iou_vs_off"],
                        "off_iou_vs_companions": off_series[i].get("iou_vs_companions", {}),
                        "on_iou_vs_companions": on_series[i].get("iou_vs_companions", {}),
                    }
                    for i in range(n_frames)
                    if on_series[i]["iou_vs_off"] < 0.999
                    or on_series[i]["area_px"] != off_series[i]["area_px"]
                ]

                focus_rows: list[dict[str, Any]] = []
                for w_start, w_end in focus_windows:
                    for i in range(n_frames):
                        sf = first_src + i
                        if w_start <= sf <= w_end:
                            score_val = on_res["frame_scores"].get(i)
                            sel_val = on_res["selected_history"].get(i)
                            focus_rows.append(
                                {
                                    "source_frame": sf,
                                    "off_area_px": off_series[i]["area_px"],
                                    "on_area_px": on_series[i]["area_px"],
                                    "off_centroid_xy": off_series[i]["centroid_xy"],
                                    "on_centroid_xy": on_series[i]["centroid_xy"],
                                    "iou_vs_off": on_series[i]["iou_vs_off"],
                                    "eff_iou_score": score_val,
                                    "selected_source_frames": (
                                        [[first_src + x for x in o] for o in sel_val]
                                        if sel_val is not None
                                        else None
                                    ),
                                    "off_iou_vs_companions": off_series[i].get("iou_vs_companions", {}),
                                    "on_iou_vs_companions": on_series[i].get("iou_vs_companions", {}),
                                }
                            )

                # Which frames had eff_iou_score[obj_idx] <= mf_threshold?
                filtered_out_frames: list[dict[str, Any]] = []
                for f_i, s_list in on_res["frame_scores"].items():
                    s_idx = min(obj_idx, len(s_list) - 1) if s_list else 0
                    if s_list and s_list[s_idx] <= scene_report["mf_threshold"]:
                        filtered_out_frames.append(
                            {
                                "source_frame": first_src + int(f_i),
                                "eff_iou_score": s_list[s_idx],
                                "all_object_scores": s_list,
                            }
                        )

                compiled_arms[model_key][mode][tname] = {
                    "mem_off_elapsed_s": off_res["elapsed_s"],
                    "mem_on_elapsed_s": on_res["elapsed_s"],
                    "total_frames": n_frames,
                    "changed_frames_count": len(changed_frames),
                    "filtered_out_frames_count": len(filtered_out_frames),
                    "filtered_out_frames": filtered_out_frames,
                    "changed_frames": changed_frames,
                    "focus_window_frames": focus_rows,
                }

    out: dict[str, Any] = {
        "scene": scene_report["scene"],
        "description": scene_report["description"],
        "capture_folder": scene_report["capture_folder"],
        "capture_date": scene_report["capture_date"],
        "load": scene_report["load"],
        "masks_sha256": scene_report["masks_sha256"],
        "subject_masks_sha256": scene_report["subject_masks_sha256"],
        "first_source_frame": first_src,
        "num_frames": n_frames,
        "pick_frame": scene_report["pick_frame"],
        "pick_source_frame": scene_report["pick_source_frame"],
        "mf_threshold": scene_report["mf_threshold"],
        "focus_windows": focus_windows,
        "arms": compiled_arms,
    }
    if scene_report.get("companion_capture_folder") is not None:
        out["companion_capture_folder"] = scene_report["companion_capture_folder"]
    if "face_subprobe" in scene_report:
        out["face_subprobe"] = scene_report["face_subprobe"]
    return out


def _run_lotsofpeople_scene(
    cfg: dict[str, Any],
    models: dict[str, Any],
    out_dir: Path,
    mf_threshold: float = MF_THRESHOLD,
) -> dict[str, Any]:
    """Run the 4 arms on `lotsofpeople` (alone and 16-dancer group)."""
    import types
    from comfy_extras.nodes_sam3 import SAM3_Detect
    from bench._lib.sam3_core import load_core
    from workflows.h3_config import SEGMENTER

    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO_ROOT)]
    sys.modules.setdefault("_h3pack", pkg)
    import _h3pack.subject_track as st

    images = _load_video_window_by_seconds(
        cfg["video_path"],
        start_sec=float(cfg["start_sec"]),
        num_frames=int(cfg["num_frames"]),
        fps=float(cfg["fps"]),
        width=int(cfg["width"]),
    )
    seed_frame = int(cfg["seed_frame"])

    # Detect all people on the seed frame once using stock SAM3_Detect, matching subject_alone_or_in_a_group.py
    stock_m, stock_clip = load_core(SEGMENTER)
    cond = stock_clip.encode_from_tokens_scheduled(stock_clip.tokenize("person:16"))
    with torch.inference_mode():
        det_out = SAM3_Detect.execute(
            stock_m,
            images[seed_frame : seed_frame + 1],
            conditioning=cond,
            threshold=0.5,
            individual_masks=True,
        )
        all_seed_masks = getattr(det_out, "args", det_out)[0].float().cpu()

    seed_masks_sha256 = hashlib.sha256(
        np.ascontiguousarray((all_seed_masks.numpy() > 0.5).astype(np.uint8)).tobytes()
    ).hexdigest()

    flat = (all_seed_masks > 0.5).flatten(1).float()
    area = all_seed_masks.flatten(1).sum(1)

    def _overlap_of(i: int, j: int) -> float:
        inter = float((flat[i] * flat[j]).sum())
        u = max(float(flat[i].sum() + flat[j].sum()) - inter, 1.0)
        m = max(min(float(flat[i].sum()), float(flat[j].sum())), 1.0)
        return max(inter / u, inter / m)

    def _distinct(order: list[int], k: int = 16) -> list[int]:
        chosen: list[int] = []
        for idx in order:
            if any(_overlap_of(idx, j) >= 0.5 for j in chosen):
                continue
            chosen.append(idx)
            if len(chosen) == k:
                break
        return chosen

    targets_cfg = [
        ("most_central", int(st.choose(all_seed_masks, [], "most central"))),
        ("largest", int(st.choose(all_seed_masks, [], "largest"))),
    ]

    # Track from seed_frame forward (frames seed_frame .. N-1)
    track_images = images[seed_frame:]
    n_track = int(track_images.shape[0])

    compiled_arms: dict[str, Any] = {}
    for model_key, base_model in models.items():
        compiled_arms[model_key] = {"alone": {}, "group": {}}
        is_sam31 = "sam31" in model_key

        for tname, tidx in targets_cfg:
            by_size = [tidx] + [i for i in area.argsort(descending=True).tolist() if i != tidx]
            ordered_indices = _distinct(by_size, 16)
            group_seed = all_seed_masks[ordered_indices]  # [<=16, H, W], target is index 0
            alone_seed = all_seed_masks[tidx : tidx + 1]  # [1, H, W]

            # Companion reference masks on seed frame for jump detection
            for mode, seed_tensor in (("alone", alone_seed), ("group", group_seed)):
                if mode == "group" and not is_sam31:
                    # SAM3Tracker tracks each object independently in a loop
                    alone_entry = compiled_arms[model_key]["alone"][tname]
                    compiled_arms[model_key]["group"][tname] = dict(
                        alone_entry,
                        note="SAM3Tracker.track_video loops _track_single_object per object independently.",
                    )
                    continue

                mode_runs: dict[str, Any] = {}
                for mem_on in (False, True):
                    active_model = (
                        apply_memory_selection(base_model, mf_threshold=mf_threshold, focus_obj_idx=0)
                        if mem_on
                        else base_model.clone()
                    )
                    mode_runs["mem_on" if mem_on else "mem_off"] = track_masks_with_sam(
                        active_model, track_images, seed_tensor
                    )

                off_res = mode_runs["mem_off"]
                on_res = mode_runs["mem_on"]
                off_m = off_res["masks"][:, 0]
                on_m = on_res["masks"][:, 0]

                changed_frames: list[dict[str, Any]] = []
                focus_rows: list[dict[str, Any]] = []
                for i in range(n_track):
                    clip_f = seed_frame + i
                    iou = round(compute_mask_iou(on_m[i], off_m[i]), 4)
                    off_area = int(off_m[i].sum())
                    on_area = int(on_m[i].sum())
                    off_c = compute_mask_centroid(off_m[i])
                    on_c = compute_mask_centroid(on_m[i])
                    row = {
                        "clip_frame": clip_f,
                        "off_area_px": off_area,
                        "on_area_px": on_area,
                        "off_centroid_xy": list(off_c) if off_c else None,
                        "on_centroid_xy": list(on_c) if on_c else None,
                        "iou_vs_off": iou,
                        "eff_iou_score": on_res["frame_scores"].get(i),
                    }
                    if iou < 0.999 or off_area != on_area:
                        changed_frames.append(row)
                    for w_s, w_e in cfg["focus_windows"]:
                        if w_s <= clip_f <= w_e:
                            focus_rows.append(row)

                filtered_out_frames = [
                    {
                        "clip_frame": seed_frame + int(f_i),
                        "eff_iou_score": s_list[0],
                        "all_object_scores": s_list,
                    }
                    for f_i, s_list in on_res["frame_scores"].items()
                    if s_list and s_list[0] <= mf_threshold
                ]

                compiled_arms[model_key][mode][tname] = {
                    "target_seed_idx": int(tidx),
                    "n_seeded_objects": int(seed_tensor.shape[0]),
                    "mem_off_elapsed_s": off_res["elapsed_s"],
                    "mem_on_elapsed_s": on_res["elapsed_s"],
                    "total_tracked_frames": n_track,
                    "changed_frames_count": len(changed_frames),
                    "filtered_out_frames_count": len(filtered_out_frames),
                    "filtered_out_frames": filtered_out_frames,
                    "changed_frames": changed_frames,
                    "focus_window_frames": focus_rows,
                }

    summary = {
        "scene": "lotsofpeople",
        "description": cfg["description"],
        "capture_date": cfg.get("capture_date", "2026-10-17"),
        "load": cfg.get("load", "lotsofpeopledance_0414_0720_live_sam3_detect"),
        "masks_sha256": seed_masks_sha256,
        "start_sec": cfg["start_sec"],
        "num_frames": cfg["num_frames"],
        "seed_frame": seed_frame,
        "n_detected_people_on_seed": int(all_seed_masks.shape[0]),
        "mf_threshold": mf_threshold,
        "arms": compiled_arms,
    }
    out_json = out_dir / "summary__lotsofpeople.json"
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def face_probe_through_node(model: Any, out_dir: Path) -> dict[str, Any]:
    """The kitchen face probe (680-760 of `lead_fp_0613g`, seeded on the lead's face at 680) through the NODE's `attach`.

    The same frames, seed and tracker call as `run_scene_four_arms`'s `face_subprobe`, on SAM 3.1, with the model
    corrected the way `MiniMaxH3SAM31Corrections` corrects it (`sam31_corrections.attach`). Four arms: memory selection
    off and on, each with the image range as the scene measurement ran it (not corrected) and as the chain runs it
    (corrected). Prints and writes, per arm, the first frame from 708 on where the face mask is not empty. Reads only
    the capture `build_default_scenes` names for the kitchen."""
    import sam31_corrections as node

    cfg = build_default_scenes()["kitchen_604_1009"]
    validate_scene_capture("kitchen_604_1009", cfg)
    sub = cfg["face_subprobe"]
    first, count = int(sub["first_source_frame"]), int(sub["num_frames"])
    width, height = int(cfg["width"]), int(cfg["height"])
    images = _load_video_window(cfg["video_path"], first_frame=first, num_frames=count, width=width, height=height)
    parts, parts_sha = unpack_capture_mask_npz(Path(sub["seed_npz"]), width=width, key="parts")
    face = parts[int(sub["seed_frame_in_npz"])]
    face = face[0] if face.ndim == 3 else face
    seed = torch.from_numpy(face.astype(np.float32)).unsqueeze(0)
    arms: dict[str, Any] = {}
    for image_range in (False, True):
        for memory in (False, True):
            m = model.clone()
            node.attach(m, None, image_range=image_range, text_activation=False, memory_selection=memory)
            got = track_masks_with_sam(m, images, seed)
            area = [int(x) for x in got["masks"][:, 0].reshape(count, -1).sum(axis=1)]
            back = next((first + i for i in range(708 - first, count) if area[i] > 0), None)
            key = f"range_{'corrected' if image_range else 'as_comfyui'}__memory_{'on' if memory else 'off'}"
            arms[key] = {"report": node.report(m, None), "first_frame_back_from_708": back, "face_area_px": area,
                         "elapsed_s": got["elapsed_s"]}
            print(f"[sam_memory_selection] face probe {key}: back at {back}", flush=True)
    out = {"what": "the kitchen face probe through sam31_corrections.attach", "frames": [first, first + count - 1],
           "capture_folder": f"data/{Path(cfg['capture_folder']).name}", "seed_npz": str(Path(sub["seed_npz"]).relative_to(Path(cfg["capture_folder"]).parent.parent)),
           "seed_frame_in_npz": int(sub["seed_frame_in_npz"]), "parts_sha256": parts_sha, "arms": arms}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "face_probe_through_node.json").write_text(json.dumps(out, indent=1))
    print(f"[sam_memory_selection] wrote {out_dir / 'face_probe_through_node.json'}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["kitchen_604_1009", "hall_1306_1418", "regain_1419_1465", "lotsofpeople"],
        help="Scene names to measure.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["sam31", "sam3"],
        default=["sam31", "sam3"],
        help="Which SAM checkpoints to run (`sam31`, `sam3`).",
    )
    parser.add_argument(
        "--mf-threshold",
        type=float,
        default=MF_THRESHOLD,
        help="Memory selection effective IoU threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_resolve_data_dir() / "2026-10-17_sam_memory_selection",
        help="Untracked data directory for JSON measurement outputs.",
    )
    parser.add_argument(
        "--face-probe",
        action="store_true",
        help="Run only the kitchen face probe on SAM 3.1, through the node's own attach (face_probe_through_node).",
    )
    args = parser.parse_args(argv)
    if args.face_probe:
        face_probe_through_node(load_sam_model(SAM31_CHECKPOINT), args.out_dir)
        return 0

    ckpt_map = {
        "sam31": SAM31_CHECKPOINT,
        "sam3": SAM3_BASE_CHECKPOINT,
    }
    loaded_models: dict[str, Any] = {}
    for mkey in args.models:
        ckpt = ckpt_map[mkey]
        print(f"[sam_memory_selection] loading {mkey} ({ckpt}) ...", flush=True)
        loaded_models[mkey] = load_sam_model(ckpt)

    all_summaries: dict[str, Any] = {}
    if args.out_dir.is_dir():
        for existing_json in sorted(args.out_dir.glob("summary__*.json")):
            if existing_json.name == "summary__all_scenes.json":
                continue
            try:
                existing_data = json.loads(existing_json.read_text())
                if isinstance(existing_data, dict) and "scene" in existing_data:
                    all_summaries[str(existing_data["scene"])] = existing_data
            except Exception:
                pass

    for sname in args.scenes:
        print(f"[sam_memory_selection] running scene {sname} ...", flush=True)
        summary = run_scene_four_arms(
            sname,
            models=loaded_models,
            out_dir=args.out_dir,
            mf_threshold=args.mf_threshold,
        )
        all_summaries[sname] = summary
        print(
            f"[sam_memory_selection] finished {sname} -> {args.out_dir / f'summary__{sname}.json'}",
            flush=True,
        )

    combined_path = args.out_dir / "summary__all_scenes.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"[sam_memory_selection] wrote {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
