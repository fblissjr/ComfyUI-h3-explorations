#!/usr/bin/env python3
"""GPU benchmark suite for SAM 3.1 Decoupled Multiplex Memory Selection (`SAM3-DMS`) and perception fixes.

Runs all verification and ablation experiments on real SAM 3.1 weights (`sam3.1_multiplex_fp32.safetensors`) via
ComfyUI's dynamic patcher (`ModelPatcherDynamic`):

  0. `n_obj_1_bit_identity_gate`: Proves current `attach(..., memory_selection=True)` on single-object tracks is
     pixel-for-pixel identical (`diff_px == 0`) to legacy `N_obj == 1` (`focus_obj_idx=0`, no encode filter).
  1. Experiment A (`multiplex_dms`): Compares `mem_off`, `focus0_legacy`, `dms_read_only`, `dms_full`, and
     `dms_full_plus_ptr` against per-object `alone` reference tracks across occluded `fun` multi-object scenes
     (`regain_1419_1465`, `hall_1306_1418`, `kitchen_occ_960_1009`, `kitchen_face_group_680_760`) plus an auto-seeded
     multi-person `fun_crowd_detect` window sorted smallest-first.
  2. Experiment B (`multimask_ptr`): Compares `multimask_ptr=False` vs `True` during propagation, logging how often
     `argmax(iou_pred) != 0` occurs, face recovery frame on `kitchen_face_680_760`, and tracking IoU impact.
  3. Experiment C (`detector_presence`): Compares `detector_presence=False` vs `True` across absent and present prompts
     in `SAM3_Detect` and comma-prompted `SAM3_VideoTrack` (`person, astronaut, dinosaur` and `astronaut, dinosaur`).

Usage:
    "$PY" bench/measure_sam31_dms_and_fixes.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(HERE), str(REPO), str(REPO / "workflows")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _lib import card_visible, needs  # noqa: E402

needs("a CUDA device to run SAM 3.1 DMS & perception benchmarks", card_visible())

from h3_config import SEGMENTER  # noqa: E402
from sam3_precision_arms import load_model, setup_core  # noqa: E402

_ORIG_ARGV = list(sys.argv[1:])
torch, mm, comfy, DYNAMIC = setup_core(False, 0)

import folder_paths  # noqa: E402
from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack  # noqa: E402

import sam31_corrections as L  # noqa: E402
import sam_memory_selection as sms  # noqa: E402
from subject_tracks import one_each  # noqa: E402


def _default_out_dir() -> Path:
    d = sms._resolve_data_dir()
    base = d.parent if d.name == "fun" else d
    return base / "2026-11-15_sam31_dms_and_fixes"


def _load_specific_frames(
    video_path: Path,
    frame_indices: list[int],
    width: int,
    height: int,
) -> dict[int, torch.Tensor]:
    """Decode sorted unique frame indices in a single pass as `[H, W, 3]` float32 `[0, 1]` CPU tensors."""
    import av

    wanted = sorted(set(int(i) for i in frame_indices))
    if not wanted:
        return {}
    wanted_set = set(wanted)
    max_idx = wanted[-1]
    out: dict[int, torch.Tensor] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx in wanted_set:
                rgb = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                out[idx] = torch.from_numpy(rgb.astype(np.float32) / 255.0)
            if idx >= max_idx:
                break
    missing = [i for i in wanted if i not in out]
    if missing:
        raise RuntimeError(f"Failed to decode frames {missing} from {video_path.name}")
    return out


def _attach_arm(stock_model: Any, arm: str) -> Any:
    """Build a cloned patcher for a named Experiment A/B tracker arm (`image_range=True`)."""
    m = stock_model.clone()
    if arm == "mem_off":
        L.attach(m, None, image_range=True, text_activation=False, memory_selection=False)
    elif arm == "focus0_legacy":
        L.attach(m, None, image_range=True, text_activation=False, memory_selection=False)
        L.attach_memory_selection(m, focus_obj_idx=0, record=True, one_object_only=False)
        m.object_patches.pop(L.ENCODE_MEM_PATCH, None)
        m.object_patches[L.MEMORY_PATCH].encode_filter = None
    elif arm == "dms_read_only":
        L.attach(m, None, image_range=True, text_activation=False, memory_selection=True)
        m.object_patches.pop(L.ENCODE_MEM_PATCH, None)
        m.object_patches[L.MEMORY_PATCH].encode_filter = None
    elif arm == "dms_full":
        L.attach(m, None, image_range=True, text_activation=False, memory_selection=True)
    elif arm == "dms_full_plus_ptr":
        L.attach(m, None, image_range=True, text_activation=False, memory_selection=True, multimask_ptr=True)
    else:
        raise ValueError(f"Unknown tracker arm: {arm!r}")
    return m


def _run_tracker_window(
    model: Any,
    images: torch.Tensor,
    seeds: torch.Tensor,
    pick_frame: int,
) -> dict[str, Any]:
    if pick_frame > 0:
        return sms.track_bidirectional_from_pick(model, images, seeds, pick_frame)
    return sms.track_masks_with_sam(model, images, seeds)


def _in_focus_window(src_frame: int, focus_windows: list[list[int]]) -> bool:
    return any(w0 <= src_frame <= w1 for w0, w1 in focus_windows)


def _eval_group_against_alone(
    group_res: dict[str, Any],
    alone_masks: np.ndarray,
    obj_names: list[str],
    first_source_frame: int,
    focus_windows: list[list[int]],
    mf_threshold: float = L.MF_THRESHOLD,
) -> dict[str, Any]:
    """Compute per-object and overall IoU metrics comparing a multi-object run to per-object `alone`."""
    g_masks = group_res["masks"]
    n_frames, n_obj = int(g_masks.shape[0]), int(g_masks.shape[1])
    scores_map: dict[int, list[float]] = group_res.get("frame_scores", {})

    per_obj: dict[str, Any] = {}
    all_ious: list[float] = []
    all_focus_ious: list[float] = []
    total_degraded = 0
    total_jumps = 0
    total_low_score = 0

    for o, name in enumerate(obj_names):
        ious: list[float] = []
        focus_ious: list[float] = []
        degraded_frames: list[int] = []
        jump_frames: list[int] = []
        for f_i in range(n_frames):
            src_f = first_source_frame + f_i
            a_m = alone_masks[f_i, o]
            g_m = g_masks[f_i, o]
            iou = sms.compute_mask_iou(g_m, a_m)
            ious.append(iou)
            if _in_focus_window(src_f, focus_windows):
                focus_ious.append(iou)
            if int(a_m.sum()) > 25 and iou < 0.5:
                degraded_frames.append(src_f)
            for c in range(n_obj):
                if c == o:
                    continue
                if int(alone_masks[f_i, c].sum()) > 25 and sms.compute_mask_iou(g_m, alone_masks[f_i, c]) > 0.5 and iou < 0.5:
                    jump_frames.append(src_f)
                    break

        low_score_frames = [
            first_source_frame + int(k)
            for k, s_list in scores_map.items()
            if o < len(s_list) and s_list[o] <= mf_threshold
        ]
        all_ious.extend(ious)
        all_focus_ious.extend(focus_ious)
        total_degraded += len(degraded_frames)
        total_jumps += len(jump_frames)
        total_low_score += len(low_score_frames)
        per_obj[name] = {
            "slot": o,
            "mean_iou_vs_alone": round(float(np.mean(ious)), 4),
            "min_iou_vs_alone": round(float(np.min(ious)), 4),
            "focus_mean_iou_vs_alone": round(float(np.mean(focus_ious)), 4) if focus_ious else None,
            "focus_min_iou_vs_alone": round(float(np.min(focus_ious)), 4) if focus_ious else None,
            "degraded_frames_lt_0_5": len(degraded_frames),
            "sample_degraded_src_frames": degraded_frames[:10],
            "companion_jump_frames": len(jump_frames),
            "sample_jump_src_frames": jump_frames[:10],
            "low_score_frames_le_mf": len(low_score_frames),
        }

    return {
        "elapsed_s": group_res["elapsed_s"],
        "mean_iou_vs_alone": round(float(np.mean(all_ious)), 4),
        "min_obj_mean_iou": round(min(v["mean_iou_vs_alone"] for v in per_obj.values()), 4),
        "focus_mean_iou_vs_alone": round(float(np.mean(all_focus_ious)), 4) if all_focus_ious else None,
        "total_degraded_frames_lt_0_5": total_degraded,
        "total_companion_jump_frames": total_jumps,
        "total_low_score_slot_frames": total_low_score,
        "per_object": per_obj,
    }


def run_n_obj_1_bit_identity_gate(stock_model: Any, scenes: dict[str, Any]) -> dict[str, Any]:
    """Verify current `attach(memory_selection=True)` matches legacy `N_obj == 1` (`focus=0`) with 0-pixel diff."""
    cfg = scenes["kitchen_604_1009"]
    sms.validate_scene_capture("kitchen_604_1009", cfg)
    sub = cfg["face_subprobe"]
    first, count = int(sub["first_source_frame"]), int(sub["num_frames"])
    w, h = int(cfg["width"]), int(cfg["height"])
    images = sms._load_video_window(cfg["video_path"], first_frame=first, num_frames=count, width=w, height=h)
    parts, _ = sms.unpack_capture_mask_npz(sub["seed_npz"], width=w, key="parts")
    face = parts[int(sub["seed_frame_in_npz"])]
    face = face[0] if face.ndim == 3 else face
    seed = torch.from_numpy(face.astype(np.float32)).unsqueeze(0)

    m_new = _attach_arm(stock_model, "dms_full")
    out_new = sms.track_masks_with_sam(m_new, images, seed)["masks"]

    m_leg = stock_model.clone()
    L._attach_range(m_leg)
    L.attach_memory_selection(m_leg, focus_obj_idx=0, record=False, one_object_only=True)
    m_leg.object_patches.pop(L.ENCODE_MEM_PATCH, None)
    m_leg.object_patches.pop(L.TRACK_LOOP_PATCH, None)
    out_leg = sms.track_masks_with_sam(m_leg, images, seed)["masks"]

    diff_px = int(np.sum(out_new != out_leg))
    assert diff_px == 0, f"N_obj==1 parity gate failed: diff_px={diff_px}"
    areas = [int(x) for x in out_new[:, 0].reshape(count, -1).sum(axis=1)]
    back = next((first + i for i in range(708 - first, count) if areas[i] > 0), None)
    print(f"[gate] N_obj==1 new vs legacy diff_px={diff_px}, first_frame_back_from_708={back}", flush=True)
    return {
        "diff_px": diff_px,
        "byte_identical": True,
        "frames": [first, first + count - 1],
        "first_frame_back_from_708": back,
    }


def _build_experiment_a_windows(
    stock_model: Any,
    stock_clip: Any,
    scenes: dict[str, Any],
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []

    # 1. regain_1419_1465: [second, lead, third] so occluded second sits in slot 0.
    cfg_r = scenes["regain_1419_1465"]
    sms.validate_scene_capture("regain_1419_1465", cfg_r)
    w_r, h_r = int(cfg_r["width"]), int(cfg_r["height"])
    imgs_r = sms._load_video_window(cfg_r["video_path"], int(cfg_r["first_source_frame"]), int(cfg_r["num_frames"]), w_r, h_r)
    lead_r, _ = sms.unpack_capture_mask_npz(cfg_r["seed_npz"], w_r)
    sec_r, _ = sms.unpack_capture_mask_npz(cfg_r["companion_npzs"]["second"]["npz"], w_r)
    thd_r, _ = sms.unpack_capture_mask_npz(cfg_r["companion_npzs"]["third"]["npz"], w_r)
    idx_r = int(cfg_r["seed_frame_in_npz"])
    seeds_r = torch.stack([
        torch.from_numpy(sec_r[int(cfg_r["companion_npzs"]["second"]["seed_frame_in_npz"])].astype(np.float32)),
        torch.from_numpy(lead_r[idx_r].astype(np.float32)),
        torch.from_numpy(thd_r[int(cfg_r["companion_npzs"]["third"]["seed_frame_in_npz"])].astype(np.float32)),
    ], dim=0)
    windows.append({
        "name": "regain_1419_1465",
        "images": imgs_r,
        "seeds": seeds_r,
        "obj_names": ["second", "lead", "third"],
        "pick_frame": int(cfg_r["pick_frame"]),
        "first_source_frame": int(cfg_r["first_source_frame"]),
        "focus_windows": cfg_r["focus_windows"],
    })

    # 2. hall_1306_1418: [third, lead] so occluded third sits in slot 0.
    cfg_h = scenes["hall_1306_1418"]
    sms.validate_scene_capture("hall_1306_1418", cfg_h)
    w_h, h_h = int(cfg_h["width"]), int(cfg_h["height"])
    imgs_h = sms._load_video_window(cfg_h["video_path"], int(cfg_h["first_source_frame"]), int(cfg_h["num_frames"]), w_h, h_h)
    lead_h, _ = sms.unpack_capture_mask_npz(cfg_h["seed_npz"], w_h)
    thd_h, _ = sms.unpack_capture_mask_npz(cfg_h["companion_npzs"]["third"]["npz"], w_h)
    seeds_h = torch.stack([
        torch.from_numpy(thd_h[int(cfg_h["companion_npzs"]["third"]["seed_frame_in_npz"])].astype(np.float32)),
        torch.from_numpy(lead_h[int(cfg_h["seed_frame_in_npz"])].astype(np.float32)),
    ], dim=0)
    windows.append({
        "name": "hall_1306_1418",
        "images": imgs_h,
        "seeds": seeds_h,
        "obj_names": ["third", "lead"],
        "pick_frame": int(cfg_h["pick_frame"]),
        "first_source_frame": int(cfg_h["first_source_frame"]),
        "focus_windows": cfg_h["focus_windows"],
    })

    # 3 & 4. kitchen windows sharing lead/second/strangers npzs.
    cfg_k = scenes["kitchen_604_1009"]
    sms.validate_scene_capture("kitchen_604_1009", cfg_k)
    w_k, h_k = int(cfg_k["width"]), int(cfg_k["height"])
    lead_k, _ = sms.unpack_capture_mask_npz(cfg_k["seed_npz"], w_k, key="track")
    lead_p, _ = sms.unpack_capture_mask_npz(cfg_k["seed_npz"], w_k, key="parts")
    sec_k, _ = sms.unpack_capture_mask_npz(cfg_k["companion_npzs"]["second"]["npz"], w_k, key="track")
    str_k, _ = sms.unpack_capture_mask_npz(cfg_k["companion_npzs"]["strangers"]["npz"], w_k, key="track")

    # 3. kitchen_occ_960_1009: [lead, second, strangers] at source 960 (npz idx 347).
    idx_960 = 960 - int(cfg_k["capture_first_frame"])
    imgs_k960 = sms._load_video_window(cfg_k["video_path"], 960, 50, w_k, h_k)
    seeds_k960 = torch.stack([
        torch.from_numpy(lead_k[idx_960].astype(np.float32)),
        torch.from_numpy(sec_k[idx_960].astype(np.float32)),
        torch.from_numpy(str_k[idx_960].astype(np.float32)),
    ], dim=0)
    windows.append({
        "name": "kitchen_occ_960_1009",
        "images": imgs_k960,
        "seeds": seeds_k960,
        "obj_names": ["lead", "second", "strangers"],
        "pick_frame": 0,
        "first_source_frame": 960,
        "focus_windows": [[998, 1009]],
    })

    # 4. kitchen_face_group_680_760: [lead_face, second, strangers] at source 680 (npz idx 67).
    sub = cfg_k["face_subprobe"]
    idx_680 = int(sub["seed_frame_in_npz"])
    imgs_k680 = sms._load_video_window(cfg_k["video_path"], 680, 81, w_k, h_k)
    face_seed = lead_p[idx_680][0] if lead_p[idx_680].ndim == 3 else lead_p[idx_680]
    seeds_k680 = torch.stack([
        torch.from_numpy(face_seed.astype(np.float32)),
        torch.from_numpy(sec_k[idx_680].astype(np.float32)),
        torch.from_numpy(str_k[idx_680].astype(np.float32)),
    ], dim=0)
    windows.append({
        "name": "kitchen_face_group_680_760",
        "images": imgs_k680,
        "seeds": seeds_k680,
        "obj_names": ["lead_face", "second", "strangers"],
        "pick_frame": 0,
        "first_source_frame": 680,
        "focus_windows": sub["focus_windows"],
    })

    # 5. fun_crowd_detect: probe candidates on fun_0000_6394_24p_sound_in_step.mov, sort smallest-first.
    fun_video = sms._resolve_input_dir() / "mrpop" / "fun_0000_6394_24p_sound_in_step.mov"
    cands = [620, 1320, 1800, 2200, 2600, 3000]
    cand_frames = _load_specific_frames(fun_video, cands, 1024, 768)
    m_det, c_det = L.corrected(stock_model, stock_clip, image_range=True, text_activation=True, memory_selection=False)
    cond_person = c_det.encode_from_tokens_scheduled(c_det.tokenize("person:12"))
    best_pick: tuple[int, list[torch.Tensor], list[int]] | None = None
    for cf in cands:
        with torch.inference_mode():
            masks_out, bbox_dicts = SAM3_Detect.execute(
                m_det, cand_frames[cf].unsqueeze(0), conditioning=cond_person, threshold=0.45, individual_masks=True
            )
        if masks_out.shape[0] == 0:
            continue
        scores = [float(d["score"]) for d in bbox_dicts[0]]
        kept, _ = one_each(masks_out, scores)
        valid_masks: list[tuple[int, torch.Tensor]] = []
        for idx_k in kept:
            mk = (masks_out[idx_k] > 0.5).float().cpu()
            area = int(mk.sum().item())
            if area >= 400:
                valid_masks.append((area, mk))
        valid_masks.sort(key=lambda pair: pair[0])
        if len(valid_masks) >= 4:
            if best_pick is None or len(valid_masks) > len(best_pick[1]):
                best_pick = (cf, [mk for _, mk in valid_masks], [a for a, _ in valid_masks])
        elif best_pick is None and valid_masks:
            best_pick = (cf, [mk for _, mk in valid_masks], [a for a, _ in valid_masks])

    if best_pick is None:
        raise RuntimeError("fun_crowd_detect found zero valid people across candidate frames")
    cf_best, crowd_masks, crowd_areas = best_pick
    print(f"[setup] fun_crowd_detect picked source frame {cf_best} with {len(crowd_masks)} people (areas={crowd_areas})", flush=True)
    imgs_crowd = sms._load_video_window(fun_video, first_frame=cf_best, num_frames=60, width=1024, height=768)
    windows.append({
        "name": f"fun_crowd_{cf_best}",
        "images": imgs_crowd,
        "seeds": torch.stack(crowd_masks, dim=0),
        "obj_names": [f"p{i}_{crowd_areas[i]}px" for i in range(len(crowd_masks))],
        "pick_frame": 0,
        "first_source_frame": cf_best,
        "focus_windows": [[cf_best, cf_best + 59]],
    })
    return windows


def run_experiment_a(stock_model: Any, windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment A: multi-object Multiplex DMS vs per-object `alone` reference."""
    arms_order = ("mem_off", "focus0_legacy", "dms_read_only", "dms_full", "dms_full_plus_ptr")
    results: dict[str, Any] = {}

    for win in windows:
        wname = win["name"]
        imgs = win["images"]
        seeds = win["seeds"]
        names = win["obj_names"]
        pick = win["pick_frame"]
        first_src = win["first_source_frame"]
        focus_w = win["focus_windows"]
        n_frames, H, W = int(imgs.shape[0]), int(imgs.shape[1]), int(imgs.shape[2])
        n_obj = len(names)

        print(f"\n[Exp A] === {wname} (N={n_frames}, objs={names}, pick={pick}) ===", flush=True)
        alone_masks = np.zeros((n_frames, n_obj, H, W), dtype=bool)
        alone_elapsed: dict[str, float] = {}
        for o, oname in enumerate(names):
            m_one = _attach_arm(stock_model, "dms_full")
            res_o = _run_tracker_window(m_one, imgs, seeds[o : o + 1], pick)
            alone_masks[:, o] = res_o["masks"][:, 0]
            alone_elapsed[oname] = res_o["elapsed_s"]

        win_arms: dict[str, Any] = {
            "alone": {"elapsed_s_per_obj": alone_elapsed, "elapsed_s_total": round(sum(alone_elapsed.values()), 3)}
        }
        for arm in arms_order:
            m_grp = _attach_arm(stock_model, arm)
            res_g = _run_tracker_window(m_grp, imgs, seeds, pick)
            metrics = _eval_group_against_alone(res_g, alone_masks, names, first_src, focus_w)
            win_arms[arm] = metrics
            print(
                f"  {arm:18s}: mean_iou={metrics['mean_iou_vs_alone']:.4f} focus_iou={metrics['focus_mean_iou_vs_alone']} "
                f"min_obj={metrics['min_obj_mean_iou']:.4f} degraded={metrics['total_degraded_frames_lt_0_5']} "
                f"jumps={metrics['total_companion_jump_frames']} low_scores={metrics['total_low_score_slot_frames']} "
                f"t={metrics['elapsed_s']:.2f}s",
                flush=True,
            )

        results[wname] = {
            "first_source_frame": first_src,
            "num_frames": n_frames,
            "pick_frame": pick,
            "focus_windows": focus_w,
            "objects": names,
            "seed_areas_px": [int(seeds[o].sum().item()) for o in range(n_obj)],
            "arms": win_arms,
        }
    return results


def _track_with_multimask_hook(
    stock_model: Any,
    imgs: torch.Tensor,
    seeds: torch.Tensor,
    pick: int,
    multimask_ptr: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run tracker (`memory_selection=True`) and record `argmax(iou_pred)` counts when propagation uses multimask."""
    m = _attach_arm(stock_model, "dms_full_plus_ptr" if multimask_ptr else "dms_full")
    n_obj = int(seeds.shape[0])
    stats = {"total_prop_obj_steps": 0, "branch_counts": [0, 0, 0], "nonzero_best_steps": 0}
    decoder = stock_model.model.diffusion_model.tracker.sam_mask_decoder
    tap = m.object_patches.get(L.MULTIMASK_TAP_PATCH)

    def _hook(module: Any, inputs: Any, output: Any) -> None:
        if tap is not None and not tap.armed:
            return
        if not (isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor)):
            return
        iou = output[1].detach()
        t_out = int(iou.shape[-1])
        if t_out <= 1:
            return
        rows = iou.reshape(-1, t_out)[:n_obj]
        best = torch.argmax(rows, dim=-1).tolist()
        for b in best:
            stats["total_prop_obj_steps"] += 1
            if b < len(stats["branch_counts"]):
                stats["branch_counts"][b] += 1
            if b != 0:
                stats["nonzero_best_steps"] += 1

    handle = decoder.register_forward_hook(_hook) if multimask_ptr else None
    try:
        out = _run_tracker_window(m, imgs, seeds, pick)
    finally:
        if handle is not None:
            handle.remove()

    total = max(stats["total_prop_obj_steps"], 1)
    stats["nonzero_rate"] = round(stats["nonzero_best_steps"] / total, 4) if stats["total_prop_obj_steps"] else 0.0
    return out, stats


def run_experiment_b(stock_model: Any, scenes: dict[str, Any], exp_a_windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment B: `correct_multimask_ptr=True` A/B on single-object and group occlusion sequences."""
    cfg_k = scenes["kitchen_604_1009"]
    sub = cfg_k["face_subprobe"]
    w_k, h_k = int(cfg_k["width"]), int(cfg_k["height"])
    imgs_face = sms._load_video_window(cfg_k["video_path"], int(sub["first_source_frame"]), int(sub["num_frames"]), w_k, h_k)
    lead_p, _ = sms.unpack_capture_mask_npz(sub["seed_npz"], width=w_k, key="parts")
    face = lead_p[int(sub["seed_frame_in_npz"])]
    face = face[0] if face.ndim == 3 else face
    face_seed = torch.from_numpy(face.astype(np.float32)).unsqueeze(0)

    by_name = {w["name"]: w for w in exp_a_windows}
    w_hall = by_name["hall_1306_1418"]
    w_reg = by_name["regain_1419_1465"]

    cases = [
        ("kitchen_face_680_760_N1", imgs_face, face_seed, 0, ["lead_face"], int(sub["first_source_frame"])),
        ("hall_third_1306_1418_N1", w_hall["images"], w_hall["seeds"][0:1], w_hall["pick_frame"], ["third"], w_hall["first_source_frame"]),
        ("hall_group_1306_1418_N2", w_hall["images"], w_hall["seeds"], w_hall["pick_frame"], w_hall["obj_names"], w_hall["first_source_frame"]),
        ("regain_second_1419_1465_N1", w_reg["images"], w_reg["seeds"][0:1], w_reg["pick_frame"], ["second"], w_reg["first_source_frame"]),
        ("regain_group_1419_1465_N3", w_reg["images"], w_reg["seeds"], w_reg["pick_frame"], w_reg["obj_names"], w_reg["first_source_frame"]),
    ]

    out_cases: dict[str, Any] = {}
    for cname, imgs, seeds, pick, obj_names, first_src in cases:
        res_off, _ = _track_with_multimask_hook(stock_model, imgs, seeds, pick, multimask_ptr=False)
        res_on, ptr_stats = _track_with_multimask_hook(stock_model, imgs, seeds, pick, multimask_ptr=True)
        m_off, m_on = res_off["masks"], res_on["masks"]
        n_f, n_o = int(m_off.shape[0]), int(m_off.shape[1])

        per_obj_iou: dict[str, float] = {}
        changed_frames: dict[str, int] = {}
        all_ious: list[float] = []
        for o, oname in enumerate(obj_names):
            o_ious = [sms.compute_mask_iou(m_off[i, o], m_on[i, o]) for i in range(n_f)]
            all_ious.extend(o_ious)
            per_obj_iou[oname] = round(float(np.mean(o_ious)), 4)
            changed_frames[oname] = sum(1 for v in o_ious if v < 0.999)

        entry: dict[str, Any] = {
            "num_frames": n_f,
            "objects": obj_names,
            "prop_branch_stats": ptr_stats,
            "mean_mask_iou_off_vs_on": round(float(np.mean(all_ious)), 4),
            "per_obj_mean_iou_off_vs_on": per_obj_iou,
            "changed_frames_iou_lt_0_999": changed_frames,
            "off_elapsed_s": res_off["elapsed_s"],
            "on_elapsed_s": res_on["elapsed_s"],
        }
        if cname == "kitchen_face_680_760_N1":
            area_off = [int(x) for x in m_off[:, 0].reshape(n_f, -1).sum(axis=1)]
            area_on = [int(x) for x in m_on[:, 0].reshape(n_f, -1).sum(axis=1)]
            entry["face_back_from_708_off"] = next((first_src + i for i in range(708 - first_src, n_f) if area_off[i] > 0), None)
            entry["face_back_from_708_on"] = next((first_src + i for i in range(708 - first_src, n_f) if area_on[i] > 0), None)

        print(
            f"[Exp B] {cname:28s}: nonzero_best={ptr_stats['nonzero_best_steps']}/{ptr_stats['total_prop_obj_steps']} "
            f"({ptr_stats['nonzero_rate']*100:.1f}%), iou_off_vs_on={entry['mean_mask_iou_off_vs_on']:.4f}, "
            f"changed={changed_frames}",
            flush=True,
        )
        out_cases[cname] = entry
    return out_cases


def run_experiment_c(stock_model: Any, stock_clip: Any) -> dict[str, Any]:
    """Experiment C: `correct_detector_presence=True` A/B on `SAM3_Detect` and comma-prompted `SAM3_VideoTrack`."""
    fun_video = sms._resolve_input_dir() / "mrpop" / "fun_0000_6394_24p_sound_in_step.mov"
    detect_indices = [620, 700, 1000, 1320, 1440, 2000, 3000, 4000]
    track_indices = list(range(620, 644))
    decoded = _load_specific_frames(fun_video, sorted(set(detect_indices + track_indices)), 1024, 768)

    detect_batch = torch.stack([decoded[i] for i in detect_indices], dim=0)
    track_clip_imgs = torch.stack([decoded[i] for i in track_indices], dim=0)

    m_off, c_off = L.corrected(stock_model, stock_clip, image_range=True, text_activation=True, memory_selection=True, detector_presence=False)
    m_on, c_on = L.corrected(stock_model, stock_clip, image_range=True, text_activation=True, memory_selection=True, detector_presence=True)

    absent_prompts = ["astronaut:8", "bicycle:8", "dinosaur:8", "umbrella:8"]
    present_prompts = ["person:8", "woman:8", "head:8"]
    detect_results: dict[str, Any] = {}

    for group_label, prompts in (("absent", absent_prompts), ("present", present_prompts)):
        detect_results[group_label] = {}
        for prompt in prompts:
            arm_stats: dict[str, Any] = {}
            for arm_label, model_p, clip_p in (("presence_off", m_off, c_off), ("presence_on", m_on, c_on)):
                cond = clip_p.encode_from_tokens_scheduled(clip_p.tokenize(prompt))
                with torch.inference_mode():
                    _, bbox_dicts = SAM3_Detect.execute(
                        model_p, detect_batch, conditioning=cond, threshold=0.0, refine_iterations=0, individual_masks=True
                    )
                all_scores = [float(d["score"]) for frame_dicts in bbox_dicts for d in frame_dicts]
                top1_per_frame = [max((float(d["score"]) for d in frame_dicts), default=0.0) for frame_dicts in bbox_dicts]
                arm_stats[arm_label] = {
                    "count_gt_0_3": sum(1 for s in all_scores if s > 0.3),
                    "count_gt_0_4": sum(1 for s in all_scores if s > 0.4),
                    "count_gt_0_5": sum(1 for s in all_scores if s > 0.5),
                    "max_score": round(max(all_scores, default=0.0), 4),
                    "mean_top1_score": round(float(np.mean(top1_per_frame)), 4),
                    "top1_per_frame": [round(s, 4) for s in top1_per_frame],
                }
            detect_results[group_label][prompt] = arm_stats
            print(
                f"[Exp C detect {group_label:7s}] {prompt:14s}: "
                f"OFF >0.3={arm_stats['presence_off']['count_gt_0_3']} >0.5={arm_stats['presence_off']['count_gt_0_5']} max={arm_stats['presence_off']['max_score']:.4f} | "
                f"ON >0.3={arm_stats['presence_on']['count_gt_0_3']} >0.5={arm_stats['presence_on']['count_gt_0_5']} max={arm_stats['presence_on']['max_score']:.4f}",
                flush=True,
            )

    video_track_results: dict[str, Any] = {}
    for prompt_str in ("astronaut, dinosaur", "person, astronaut, dinosaur"):
        vt_stats: dict[str, Any] = {}
        for arm_label, model_p, clip_p in (("presence_off", m_off, c_off), ("presence_on", m_on, c_on)):
            cond = clip_p.encode_from_tokens_scheduled(clip_p.tokenize(prompt_str))
            with torch.inference_mode():
                (track_data,) = SAM3_VideoTrack.execute(
                    track_clip_imgs,
                    model_p,
                    initial_mask=None,
                    conditioning=cond,
                    detection_threshold=0.4,
                    max_objects=16,
                    detect_interval=6,
                )
            pm = track_data["packed_masks"]
            spawned = 0 if pm is None else int(pm.shape[1])
            scores = [round(float(s), 4) for s in track_data.get("scores", [])]
            vt_stats[arm_label] = {"spawned_objects": spawned, "scores": scores}
        video_track_results[prompt_str] = vt_stats
        print(
            f"[Exp C track ] {prompt_str:28s}: OFF spawned={vt_stats['presence_off']['spawned_objects']} ({vt_stats['presence_off']['scores']}) | "
            f"ON spawned={vt_stats['presence_on']['spawned_objects']} ({vt_stats['presence_on']['scores']})",
            flush=True,
        )

    return {
        "detect_frames": detect_indices,
        "video_track_frames": [track_indices[0], track_indices[-1]],
        "detect": detect_results,
        "video_track": video_track_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir(), help="Directory for benchmark_summary.json.")
    args = parser.parse_args(_ORIG_ARGV if argv is None else argv)

    ckpt_path = folder_paths.get_full_path("checkpoints", SEGMENTER)
    needs(f"checkpoint `{SEGMENTER}` under checkpoints", ckpt_path is not None)
    stock_model, stock_clip, _ = load_model(torch, comfy, ckpt_path, None, keep_text_projection=True)
    scenes = sms.build_default_scenes()

    t_start = time.perf_counter()
    gate = run_n_obj_1_bit_identity_gate(stock_model, scenes)
    exp_a_windows = _build_experiment_a_windows(stock_model, stock_clip, scenes)
    exp_a = run_experiment_a(stock_model, exp_a_windows)
    exp_b = run_experiment_b(stock_model, scenes, exp_a_windows)
    exp_c = run_experiment_c(stock_model, stock_clip)
    total_s = round(time.perf_counter() - t_start, 2)

    summary = {
        "checkpoint": SEGMENTER,
        "dynamic_vram": bool(DYNAMIC),
        "patcher_class": type(stock_model).__name__,
        "mf_threshold": L.MF_THRESHOLD,
        "total_elapsed_s": total_s,
        "n_obj_1_bit_identity_gate": gate,
        "experiment_a_multiplex_dms": exp_a,
        "experiment_b_multimask_ptr": exp_b,
        "experiment_c_detector_presence": exp_c,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "benchmark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_path} in {total_s:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
