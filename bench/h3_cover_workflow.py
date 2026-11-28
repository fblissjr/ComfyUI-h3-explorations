#!/usr/bin/env python3
"""Generalized 2-step pure-H3 cover music-video CLI workflow (`prepare` -> `render` / `score`).

Automates everything except creative lyric authoring for arbitrary inputs:
  - `--ref-audio`: extracts clean phrase-bounded 8.0 s (`192f`, 32 kHz) vocal reference(s)
  - `--ref-images`: stages 1..K portrait/scene stills into `$H3_INPUT_DIR/cover_jobs/<job>/`
  - `--target-song`: runs GPU/CPU `MelBand-RoFormer + HDemucs` cascade (`bench/h3_stem_sep.py`),
    auto-selects register semitone shift (`vocal_formant=False`), aligns `39 + 51m` window
    seams onto vocal breath pauses, drafts 6-section `ref2va` prompt skeletons (with optional
    `.lrc`/`.txt` lyrics auto-slotted), runs uncached Pass-2 denoising by default (with
    opt-in `MiniMaxH3FrozenVideoCache` via `--frozen-video-cache`), stitches window outputs, and scores pitch lock via `bench/pitch_lock_metrics.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.functional as TAF

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap, comfy_client  # noqa: E402

bootstrap(cpu=None)

# Import repo-root pure-stdlib helpers cleanly without leaving `str(REPO)` on `sys.path`
# where root `nodes.py` would shadow ComfyUI's own `nodes.py`.
sys.path.insert(0, str(REPO))
import h3_prompt_contract as prompt_contract  # noqa: E402
import h3_rules  # noqa: E402

sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import h3_stem_sep as stem_sep  # noqa: E402
import pitch_lock_metrics  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
FPS = float(h3_rules.FPS)  # 24.0
YIN_FPS = float(yin_probe.DEFAULT_SR) / float(yin_probe.DEFAULT_HOP_LEN)  # 100.0 Hz (240-sample hop @ 24 kHz)
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_WIN_FRAMES = 294
DEFAULT_CTX_FRAMES = 39
DEFAULT_SEED = 910427311
PLACEHOLDER_PREFIX = "[PASTE_LYRICS_"

_SAFE_JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LRC_LINE_RE = re.compile(r"^((?:\[\d{1,3}:\d{2}(?:\.\d+)?\])+)\s*(.*)$")
_LRC_TAG_RE = re.compile(r"\[(\d{1,3}):(\d{2}(?:\.\d+)?)\]")


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _validate_job_name(job: str) -> str:
    clean = (job or "").strip()
    if not clean or not _SAFE_JOB_RE.match(clean) or ".." in clean:
        raise ValueError(f"Invalid job name {job!r}; use [A-Za-z0-9_.-] without '..'")
    return clean


def _job_paths(
    job: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    j = _validate_job_name(job)
    data_dir = REPO / "data" / "cover_jobs" / j
    paths: dict[str, Path] = {
        "data_dir": data_dir,
        "prompts_dir": data_dir / "prompts",
        "graphs_dir": data_dir / "graphs",
        "manifest": data_dir / "manifest.json",
    }
    if input_dir is not None:
        in_dir = input_dir / "cover_jobs" / j
        paths["in_dir"] = in_dir
        paths["stems_dir"] = in_dir / "stems"
    if output_dir is not None:
        out_dir = output_dir / "Video" / "cover_jobs" / j
        paths["out_dir"] = out_dir
        paths["slices_dir"] = out_dir / "slices"
        paths["vocal_stems_dir"] = out_dir / "vocal_stems"
    return paths


def _ensure_stereo(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.unsqueeze(0).repeat(2, 1).to(torch.float32)
    if wav.shape[0] == 1:
        return wav.repeat(2, 1).to(torch.float32)
    return wav[:2].to(torch.float32)


def _load_audio_np(p: Path, sr: int = 24000, channels: int = 1) -> np.ndarray:
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    )
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return arr if channels == 1 else arr.reshape(-1, channels).T


def _f0_percentiles(vals: np.ndarray) -> dict[str, float] | None:
    if len(vals) < 8:
        return None
    qs = np.percentile(vals, [10, 25, 50, 75, 90])
    return {
        "p10": round(float(qs[0]), 2),
        "p25": round(float(qs[1]), 2),
        "median": round(float(qs[2]), 2),
        "p75": round(float(qs[3]), 2),
        "p90": round(float(qs[4]), 2),
    }


def _pad_or_trim_np(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[-1] >= n:
        return arr[..., :n]
    pad_w = [(0, 0)] * (arr.ndim - 1) + [(0, n - arr.shape[-1])]
    return np.pad(arr, pad_w)


# ---------------------------------------------------------------------------
# 1. Automated Reference Selection, Register Alignment, & Seam Planning
# ---------------------------------------------------------------------------


def pick_best_ref_vocal_window(
    voc_44k: torch.Tensor,
    sr: int = 44100,
    *,
    dur_s: float = 8.0,
    hop_s: float = 0.25,
    boundary_s: float = 0.20,
    quiet_boundary_dbfs: float = -35.0,
    register_bias: str = "balanced",
    scan_s: float | None = None,
    explicit_start_s: float | None = None,
) -> dict[str, Any]:
    """Select the cleanest phrase-bounded 8.0 s (`192f`) reference vocal window."""
    if register_bias not in ("balanced", "low", "high"):
        raise ValueError(f"Unsupported register_bias {register_bias!r}")
    stereo = _ensure_stereo(voc_44k)
    mono_24k = TAF.resample(stereo.mean(dim=0, keepdim=True), orig_freq=sr, new_freq=24000).squeeze(0)
    f0_t, _, rms_t = yin_probe.yin_f0(mono_24k, sr=24000)
    f0 = f0_t.numpy()
    rms = rms_t.numpy()

    win_yf = int(round(dur_s * YIN_FPS))
    hop_yf = max(1, int(round(hop_s * YIN_FPS)))
    b_yf = max(1, min(win_yf // 4, int(round(boundary_s * YIN_FPS))))
    target_samples_32k = int(round(dur_s * stem_sep.SCAFFOLD_SR))

    n_frames = len(f0)
    scan_end = n_frames if scan_s is None or scan_s <= 0 else min(n_frames, int(round(scan_s * YIN_FPS)))
    global_voiced = np.isfinite(f0[:scan_end]) & (f0[:scan_end] > 0) & (rms[:scan_end] > -42.0)
    full_f0_stats = _f0_percentiles(f0[:scan_end][global_voiced])
    global_med = float(full_f0_stats["median"]) if full_f0_stats is not None else 200.0

    candidates: list[dict[str, Any]] = []
    if explicit_start_s is not None:
        max_sf = max(0, n_frames - win_yf)
        start_candidates = [max(0, min(max_sf, int(round(explicit_start_s * YIN_FPS))))]
    elif scan_end <= win_yf:
        start_candidates = [0]
    else:
        start_candidates = list(range(0, scan_end - win_yf + 1, hop_yf))

    for sf in start_candidates:
        ef = min(n_frames, sf + win_yf)
        sub_f0 = f0[sf:ef]
        sub_rms = rms[sf:ef]
        sub_v = np.isfinite(sub_f0) & (sub_f0 > 0) & (sub_rms > -42.0)
        v_frac = float(np.mean(sub_v)) if len(sub_v) else 0.0
        head_db = float(np.max(sub_rms[: min(b_yf, len(sub_rms))])) if len(sub_rms) else -80.0
        tail_db = float(np.max(sub_rms[-min(b_yf, len(sub_rms)) :])) if len(sub_rms) else -80.0
        penalty = 0.025 * max(0.0, head_db - quiet_boundary_dbfs) + 0.025 * max(0.0, tail_db - quiet_boundary_dbfs)

        stats = _f0_percentiles(sub_f0[sub_v])
        if stats is not None:
            iqr_oct = math.log2(max(stats["p75"], 1.0) / max(stats["p25"], 1.0))
            med_hz = stats["median"]
        else:
            iqr_oct = 0.0
            med_hz = global_med

        if register_bias == "balanced":
            reg_term = 0.10 * iqr_oct
        elif register_bias == "low":
            reg_term = -0.50 * math.log2(max(med_hz, 1.0) / max(global_med, 1.0))
        else:
            reg_term = 0.50 * math.log2(max(med_hz, 1.0) / max(global_med, 1.0))

        candidates.append(
            {
                "start_frame": sf,
                "start_s": round(sf / YIN_FPS, 4),
                "voiced_frac": round(v_frac, 4),
                "head_dbfs": round(head_db, 2),
                "tail_dbfs": round(tail_db, 2),
                "f0_stats": stats,
                "score": v_frac + reg_term - penalty,
            }
        )

    dense = [c for c in candidates if c["voiced_frac"] >= 0.35]
    pool = dense if dense else candidates
    best = max(pool, key=lambda c: c["score"])

    s0 = int(round(best["start_s"] * sr))
    seg_len = int(round(dur_s * sr))
    seg = stereo[:, s0 : s0 + seg_len]
    if seg.shape[1] < seg_len:
        seg = torch.nn.functional.pad(seg, (0, seg_len - seg.shape[1]))
    wav_32k = TAF.resample(seg, orig_freq=sr, new_freq=stem_sep.SCAFFOLD_SR)
    if wav_32k.shape[1] < target_samples_32k:
        wav_32k = torch.nn.functional.pad(wav_32k, (0, target_samples_32k - wav_32k.shape[1]))
    else:
        wav_32k = wav_32k[:, :target_samples_32k]
    peak = float(wav_32k.abs().max().item())
    if peak > 0.92:
        wav_32k = wav_32k * (0.92 / peak)

    return {
        "wav_32k": wav_32k,
        "start_s": best["start_s"],
        "duration_s": round(dur_s, 4),
        "voiced_frac": best["voiced_frac"],
        "head_dbfs": best["head_dbfs"],
        "tail_dbfs": best["tail_dbfs"],
        "f0_stats": best["f0_stats"],
        "full_ref_f0_stats": full_f0_stats,
        "register_bias": register_bias,
    }


def choose_cover_semitones(
    ref_f0_stats: dict[str, float] | None,
    tgt_voc_24k: torch.Tensor,
    *,
    explicit_semitones: float | None = None,
    explicit_formant: bool | None = None,
) -> dict[str, Any]:
    """Compute integer semitone transposition matching target vocals to `<Audio 1>` chest/belt range."""
    mono = tgt_voc_24k.mean(dim=0) if tgt_voc_24k.ndim == 2 else tgt_voc_24k
    f0_t, _, rms_t = yin_probe.yin_f0(mono.to(torch.float32), sr=24000)
    f0 = f0_t.numpy()
    rms = rms_t.numpy()

    active = rms > -42.0
    voiced = np.isfinite(f0) & (f0 > 0) & active
    n_active = int(np.sum(active))
    n_voiced = int(np.sum(voiced))
    active_periodic_frac = float(n_voiced) / max(n_active, 1)
    tgt_stats = _f0_percentiles(f0[voiced])

    warnings: list[str] = []
    if active_periodic_frac < 0.35:
        warnings.append(
            f"target vocal has low single-voice periodicity ({active_periodic_frac:.3f} < 0.35; "
            "stacked harmonies or shouted ad-libs likely) -- verify --semitones by ear"
        )

    if explicit_semitones is not None:
        st = float(explicit_semitones)
    elif tgt_stats is None or n_voiced < 24 or ref_f0_stats is None or not ref_f0_stats.get("median"):
        st = 0.0
        warnings.append("insufficient voiced frames to auto-detect semitones; defaulting to 0.0 st")
    else:
        raw_st = 12.0 * math.log2(float(ref_f0_stats["median"]) / float(tgt_stats["median"]))
        st = float(max(-12, min(12, round(raw_st))))
        ref_p90 = float(ref_f0_stats.get("p90") or ref_f0_stats["median"])
        ref_p75 = float(ref_f0_stats.get("p75") or ref_p90)
        ref_p10 = float(ref_f0_stats.get("p10") or ref_f0_stats["median"])
        while (
            (
                tgt_stats["p75"] * (2.0 ** (st / 12.0)) > max(ref_p90, ref_p75 * 1.06)
                or tgt_stats["p90"] * (2.0 ** (st / 12.0)) > ref_p90 * 1.12
            )
            and tgt_stats["p25"] * (2.0 ** ((st - 1.0) / 12.0)) >= ref_p10 * 0.92
            and st > -12.0
        ):
            st -= 1.0

    vocal_formant = bool(explicit_formant) if explicit_formant is not None else False
    shifted_med = (
        round(float(tgt_stats["median"]) * (2.0 ** (st / 12.0)), 2) if tgt_stats is not None else None
    )
    return {
        "semitones": st,
        "vocal_formant": vocal_formant,
        "active_periodic_frac": round(active_periodic_frac, 4),
        "tgt_f0_stats": tgt_stats,
        "shifted_tgt_med_hz": shifted_med,
        "warnings": warnings,
    }


def plan_breath_aligned_windows(
    voc_24k: torch.Tensor,
    *,
    n_windows: int | None = None,
    duration_s: float | None = None,
    start_s: float | None = None,
    window_frames: int = DEFAULT_WIN_FRAMES,
    context_frames: int = DEFAULT_CTX_FRAMES,
    search_pad_s: float = 3.0,
) -> dict[str, Any]:
    """Place `39 + 51m` window boundaries on vocal pauses and classify sung vs instrumental shots."""
    chain = h3_rules.chain_lengths()
    mono = (voc_24k.mean(dim=0) if voc_24k.ndim == 2 else voc_24k).detach().float().flatten()
    f0_t, _, y_rms_t = yin_probe.yin_f0(mono, sr=24000)
    yin_f0_arr = f0_t.numpy()
    yin_rms_arr = y_rms_t.numpy()

    hop_samps = 1000  # 24,000 Hz / 24 fps
    n_total_f = max(1, int(mono.numel() // hop_samps))
    if mono.numel() < hop_samps:
        pad_m = torch.nn.functional.pad(mono, (0, hop_samps - mono.numel()))
        rms = (20.0 * torch.log10(torch.sqrt(torch.mean(pad_m.unsqueeze(0) ** 2, dim=-1)).clamp(min=1e-7))).numpy()
    else:
        frames_24 = mono[: n_total_f * hop_samps].view(n_total_f, hop_samps)
        rms = (20.0 * torch.log10(torch.sqrt(torch.mean(frames_24 ** 2, dim=-1)).clamp(min=1e-7))).numpy()

    def _yin_slice(vf0: int, vf1: int) -> tuple[np.ndarray, np.ndarray]:
        y0 = max(0, min(len(yin_f0_arr) - 1, int(round(vf0 * YIN_FPS / FPS))))
        y1 = max(y0 + 1, min(len(yin_f0_arr), int(round(vf1 * YIN_FPS / FPS))))
        return yin_f0_arr[y0:y1], yin_rms_arr[y0:y1]

    if n_windows is None:
        if duration_s is not None:
            if int(round(duration_s * FPS)) <= window_frames and duration_s <= window_frames / FPS:
                n_win = 1
            else:
                stride = window_frames - context_frames
                n_win = 1 + max(0, math.ceil((duration_s * FPS - window_frames) / stride))
        else:
            active_idx = np.where(rms > -38.0)[0]
            end_f = int(active_idx[-1]) + 12 if len(active_idx) else min(n_total_f, window_frames)
            stride = window_frames - context_frames
            n_win = 1 if end_f <= window_frames else 1 + math.ceil((end_f - window_frames) / stride)
    else:
        n_win = int(n_windows)
    if n_win < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_win}")

    if n_win >= 2:
        if window_frames not in chain:
            raise ValueError(f"window_frames={window_frames} must be in {chain} for chained windows")
        if not h3_rules.on_both_clocks(context_frames) or context_frames >= window_frames:
            raise ValueError(f"context_frames={context_frames} must satisfy on_both_clocks (e.g. 39, 90) and be < {window_frames}")
    else:
        if h3_rules.snap_length(window_frames) != window_frames or not h3_rules.duration_in_range(window_frames):
            raise ValueError(f"window_frames={window_frames} must satisfy 17n+5 and 5.0..15.083 s")

    stride_frames = window_frames - context_frames
    total_frames = window_frames if n_win == 1 else window_frames + (n_win - 1) * stride_frames

    def _seam_cost(f0_start: int) -> tuple[float, float, float]:
        zone_peaks: list[float] = []
        zone_means: list[float] = []
        if n_win >= 2:
            for k in range(1, n_win):
                seam_head = f0_start + k * stride_frames
                z0 = max(0, min(n_total_f - 1, seam_head + context_frames - 5))
                z1 = max(z0 + 1, min(n_total_f, seam_head + context_frames))
                seg = rms[z0:z1]
                zone_peaks.append(float(np.max(seg)))
                zone_means.append(float(np.mean(seg)))
        end_z0 = max(0, min(n_total_f - 1, f0_start + total_frames - 6))
        end_z1 = max(end_z0 + 1, min(n_total_f, f0_start + total_frames))
        end_seg = rms[end_z0:end_z1]
        zone_peaks.append(float(np.max(end_seg)))
        zone_means.append(float(np.mean(end_seg)))
        max_p = max(zone_peaks)
        mean_m = float(np.mean(zone_means))
        return 0.6 * max_p + 0.4 * mean_m, max_p, mean_m

    max_start_f = max(0, n_total_f - total_frames)
    if start_s is not None:
        best_f0 = max(0, min(max_start_f, int(round(start_s * FPS))))
        best_cost, best_max_db, best_mean_db = _seam_cost(best_f0)
    else:
        act = (rms > -38.0).astype(np.int32)
        f_on = 0
        for idx in range(0, max(1, len(act) - 2)):
            if act[idx] and act[idx + 1] and act[idx + 2]:
                f_on = idx
                break
        cand_lo = max(0, f_on - int(round(2.5 * FPS)))
        cand_hi = min(max_start_f, f_on + int(round(search_pad_s * FPS)))
        if cand_hi < cand_lo:
            cand_hi = cand_lo
        best_f0 = cand_lo
        best_cost = float("inf")
        best_max_db = -80.0
        best_mean_db = -80.0
        for cf in range(cand_lo, cand_hi + 1):
            c, mp, mm = _seam_cost(cf)
            if c < best_cost:
                best_f0, best_cost, best_max_db, best_mean_db = cf, c, mp, mm

    half = window_frames // 2
    all_yf0, all_yrms = _yin_slice(best_f0, best_f0 + total_frames)
    global_voiced = np.isfinite(all_yf0) & (all_yf0 > 0) & (all_yrms > -42.0)
    global_med_hz = float(np.median(all_yf0[global_voiced])) if np.any(global_voiced) else 200.0
    global_med_db = float(np.median(rms[best_f0 : min(n_total_f, best_f0 + total_frames)]))

    windows: list[dict[str, Any]] = []
    for idx in range(n_win):
        wf0 = best_f0 + idx * stride_frames
        wf1 = min(n_total_f, wf0 + window_frames)
        w_f0, w_yrms = _yin_slice(wf0, wf1)
        w_v = np.isfinite(w_f0) & (w_f0 > 0) & (w_yrms > -42.0)
        w_med_hz = round(float(np.median(w_f0[w_v])), 2) if int(np.sum(w_v)) >= 8 else None
        w_rms = rms[wf0:wf1]
        act_rms = w_rms[w_rms > -42.0]
        w_act_db = round(float(np.median(act_rms)), 2) if len(act_rms) else -80.0

        def _shot_mode(s_start: int, s_end: int) -> str:
            sf0, syrms = _yin_slice(s_start, s_end)
            srms = rms[s_start:s_end]
            if len(srms) == 0:
                return "instrumental"
            sv = np.isfinite(sf0) & (sf0 > 0) & (syrms > -42.0)
            return "instrumental" if (float(np.mean(sv)) < 0.08 and float(np.percentile(srms, 90)) < -35.0) else "sung"

        s1_mode = _shot_mode(wf0, min(n_total_f, wf0 + half))
        s2_mode = _shot_mode(min(n_total_f, wf0 + half), wf1)

        if idx < n_win - 1:
            z0 = max(0, min(n_total_f - 1, (best_f0 + (idx + 1) * stride_frames) + context_frames - 5))
            z1 = max(z0 + 1, min(n_total_f, (best_f0 + (idx + 1) * stride_frames) + context_frames))
            seam_db = round(float(np.max(rms[z0:z1])), 2)
        else:
            ez0 = max(0, min(n_total_f - 1, best_f0 + total_frames - 6))
            ez1 = max(ez0 + 1, min(n_total_f, best_f0 + total_frames))
            seam_db = round(float(np.max(rms[ez0:ez1])), 2)

        windows.append(
            {
                "index": idx + 1,
                "start_s": round(wf0 / FPS, 4),
                "end_s": round((wf0 + window_frames) / FPS, 4),
                "shot1_span_s": [round(wf0 / FPS, 3), round((wf0 + half) / FPS, 3)],
                "shot2_span_s": [round((wf0 + half) / FPS, 3), round((wf0 + window_frames) / FPS, 3)],
                "shot1_mode": s1_mode,
                "shot2_mode": s2_mode,
                "med_f0_hz": w_med_hz,
                "active_dbfs": w_act_db,
                "register_role": "high",
                "seam_lock_dbfs": seam_db,
            }
        )

    meds = [float(w["med_f0_hz"]) for w in windows if w["med_f0_hz"] is not None]
    if len(meds) >= 2 and max(meds) / max(min(meds), 1.0) >= 1.12:
        split_hz = math.sqrt(min(meds) * max(meds))
        for w in windows:
            if w["med_f0_hz"] is not None:
                w["register_role"] = "low" if w["med_f0_hz"] < split_hz else "high"
            else:
                w["register_role"] = "low" if w["active_dbfs"] < global_med_db else "high"

    return {
        "start_s": round(best_f0 / FPS, 4),
        "duration_s": round(total_frames / FPS, 4),
        "total_frames": int(total_frames),
        "n_windows": n_win,
        "window_frames": int(window_frames),
        "context_frames": int(context_frames),
        "stride_frames": int(stride_frames),
        "seam_max_dbfs": round(best_max_db, 2),
        "seam_mean_dbfs": round(best_mean_db, 2),
        "global_med_f0_hz": round(global_med_hz, 2),
        "windows": windows,
    }


def resolve_window_denoises(
    windows: list[dict[str, Any]],
    denoise_spec: str | float = "0.29",
    *,
    verse_denoise: float = 0.26,
    chorus_denoise: float = 0.29,
) -> list[float]:
    """Resolve `"0.29"`, `"auto"`, or comma-separated per-window denoise floats."""
    n = len(windows)
    raw = str(denoise_spec).strip().lower()
    if raw == "auto":
        return [
            round(float(verse_denoise if w.get("register_role") == "low" else chorus_denoise), 4)
            for w in windows
        ]
    if "," in raw:
        vals = [float(part.strip()) for part in raw.split(",") if part.strip()]
        if len(vals) != n:
            raise ValueError(f"Expected {n} comma-separated denoise values, got {len(vals)}")
    else:
        val = float(raw)
        vals = [val] * n
    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Denoise value must be in (0, 1], got {v}")
    return [round(v, 4) for v in vals]


# ---------------------------------------------------------------------------
# 2. Lyrics Parsing & Contract-Valid Prompt Skeleton Generation
# ---------------------------------------------------------------------------


def parse_lyrics_for_windows(
    lyrics_path: Path | None,
    windows: list[dict[str, Any]],
) -> list[tuple[str | None, str | None]]:
    """Map `.lrc` timestamps or plain `.txt` lines onto each window's `[Shot 1]` / `[Shot 2]`."""
    n = len(windows)
    if lyrics_path is None:
        return [(None, None)] * n

    text = lyrics_path.read_text(encoding="utf-8")
    lrc_events: list[tuple[float, str]] = []
    for raw_line in text.splitlines():
        m = _LRC_LINE_RE.match(raw_line.strip())
        if m:
            lyric = m.group(2).strip()
            if not lyric:
                continue
            for mm, ss in _LRC_TAG_RE.findall(m.group(1)):
                lrc_events.append((int(mm) * 60.0 + float(ss), lyric))

    if lrc_events:
        lrc_events.sort(key=lambda item: item[0])
        out: list[tuple[str | None, str | None]] = []
        for w in windows:
            s1_lo, s1_hi = w["shot1_span_s"]
            s2_lo, s2_hi = w["shot2_span_s"]
            s1_lines = [txt for ts, txt in lrc_events if s1_lo <= ts < s1_hi]
            s2_lines = [txt for ts, txt in lrc_events if s2_lo <= ts < s2_hi]
            out.append((" / ".join(s1_lines) or None, " / ".join(s2_lines) or None))
        return out

    blocks = [blk.strip() for blk in re.split(r"\n\s*\n", text.strip()) if blk.strip()]
    if len(blocks) == n:
        res: list[tuple[str | None, str | None]] = []
        for blk in blocks:
            lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
            if len(lines) >= 2:
                mid = (len(lines) + 1) // 2
                res.append((" ".join(lines[:mid]), " ".join(lines[mid:])))
            else:
                res.append((lines[0], None))
        return res

    sung_slots: list[tuple[int, int]] = []
    for wi, w in enumerate(windows):
        if w.get("shot1_mode", "sung") == "sung":
            sung_slots.append((wi, 0))
        if w.get("shot2_mode", "sung") == "sung":
            sung_slots.append((wi, 1))
    flat_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    buckets: list[list[str | None]] = [[None, None] for _ in range(n)]
    if sung_slots and flat_lines:
        per = max(1, math.ceil(len(flat_lines) / len(sung_slots)))
        for idx, (wi, si) in enumerate(sung_slots):
            chunk = flat_lines[idx * per : (idx + 1) * per]
            if chunk:
                buckets[wi][si] = " ".join(chunk)
    return [(b[0], b[1]) for b in buckets]


def build_window_prompt_skeleton(
    *,
    window_idx: int,
    win_info: dict[str, Any],
    n_pictures: int,
    subject_desc: str = "solo acoustic singer-guitarist centered on stage at a dynamic black vocal microphone",
    stage_desc: str = "an intimate live studio concert stage with soft warm amber key light, deep velvet curtains, and quiet distant audience in background shadow",
    voice_desc: str = "raw, expressive chest-and-belt live solo singing vocal timbre matching <Audio 1>",
    aesthetic_desc: str = "warm live concert broadcast television aesthetic with crisp tungsten highlights, deep shadows, subtle filmic grain, and smooth highlight rolloff",
    soundscape_desc: str = "Intimate acoustic studio hall ambience with resonant wood sustain, crisp finger-pick contact on bronze acoustic strings, natural room depth, and subtle vocal breath.",
    shot1_lyrics: str | None = None,
    shot2_lyrics: str | None = None,
    window_frames: int = DEFAULT_WIN_FRAMES,
) -> str:
    """Build a 350-500 word `ref2va` 6-section prompt skeleton passing `h3_prompt_contract`."""
    if n_pictures < 1:
        raise ValueError(f"n_pictures must be >= 1, got {n_pictures}")
    pic_tags = " and ".join(f"<Picture {i}>" for i in range(1, n_pictures + 1))
    extra_pic = f"<Picture {min(2, n_pictures)}>"

    s1_mode = win_info.get("shot1_mode", "sung")
    s2_mode = win_info.get("shot2_mode", "sung")

    def _shot_perf(shot_num: int, mode: str, lyr: str | None) -> str:
        if mode == "instrumental" and not lyr:
            return (
                "With lips firmly closed in calm concentration and producing zero vocal sound, breathing noise, or humming of any kind, "
                "<Subject 1> lowers their gaze gently toward the fretboard and performs an expressive wordless instrumental passage, "
                "letting warm resonant acoustic chords, crisp wound-string harmonics, and low acoustic bass undertones ring smoothly across the quiet room "
                "while fretting fingers glide and shift chord shapes cleanly along the rosewood neck before settling on a sustained open ring."
            )
        phrase = lyr if lyr else f"{PLACEHOLDER_PREFIX}W{window_idx:02d}_SHOT{shot_num}]"
        return (
            "Leaning forward smoothly beside the dark mesh grille of the microphone with natural breathing posture and calm focus, "
            "<Subject 1> (S1) opens their throat and sings clearly in the expressive vocal timbre referenced from <Audio 1>: "
            f"<|lyrics_start|><d>[English] {phrase}</d><|lyrics_end|> "
            "Natural jaw articulation, throat resonance, cheek tension, and eyelid expression track every sung syllable and pitch contour cleanly while "
            "right-wrist strumming maintains steady rhythmic timing before settling comfortably into a gentle breath at the phrase ending."
        )

    shot1_perf = _shot_perf(1, s1_mode, shot1_lyrics)
    shot2_perf = _shot_perf(2, s2_mode, shot2_lyrics)

    shot1_text = (
        f"[Shot 1] A steady medium shot inside <Subject 2> frames <Subject 1> matching {pic_tags}, seated calmly on stage beneath warm overhead tungsten spotlights "
        "while the dimly lit distant audience watches quietly from deep background shadow and adds zero crowd chatter or room distraction. Subtle amber rim illumination "
        "catches stray hair strands, soft garment fabric texture, and polished natural wood grain beside softly draped background curtains and flickering stage lamps. "
        "Holding balanced upright posture directly in front of the slender black microphone stand, "
        f"{shot1_perf}"
    )
    shot2_text = (
        f"[Shot 2] The camera cuts smoothly to an intimate shoulders-up close-up inside <Subject 2>, isolating <Subject 1> beside {extra_pic}'s warm atmospheric backdrop "
        "with soft directional key light across forehead, cheekbones, and the matte black windscreen grille while distant drapes and lamps soften into creamy shallow-depth blur. "
        "Gentle low-amplitude push-in camera motion adds subtle live concert presence while keeping every facial detail, micro-expression, and shadow edge steady and sharp without motion smearing. "
        "Maintaining unbroken physical continuity from the opening medium angle, "
        f"{shot2_perf}"
    )

    body = (
        f"subject_definitions:\n"
        f"<Subject 1> is the {subject_desc} depicted in {pic_tags}.\n"
        f"<Subject 2> is {stage_desc} matching {extra_pic}.\n"
        f"<Audio 1> is the solo vocal-timbre reference for <Subject 1> (S1), supplying a {voice_desc}.\n\n"
        f"summary:\n"
        f"[reference generation + audio reference] Window {window_idx} of the live performance: <Subject 1> performs inside <Subject 2> across two shots "
        f"({win_info['start_s']:.2f}s to {win_info['end_s']:.2f}s) with vocal timbre guided by <Audio 1>.\n\n"
        f"retention_analysis:\n"
        f"<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - face, hair, clothing, posture, and instrument.\n"
        f"<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - lighting, stage layout, curtains, and quiet room background.\n"
        f"<Audio 1>: reference - solo vocal timbre and close-mic singing character.\n\n"
        f"detailed_description:\n"
        f"The target video maintains a {aesthetic_desc}. Every wrist motion, chord change, breath cadence, and facial expression stays closely aligned with the live audio track. "
        f"{shot1_text} {shot2_text}\n\n"
        f"overall_soundscape:\n"
        f"{soundscape_desc}\n\n"
        f"non_diegetic_music:\n"
        f"N/A"
    )

    refs = [prompt_contract.ImageRef(name=f"image_{i}") for i in range(1, n_pictures + 1)]
    refs.append(prompt_contract.AudioRef(name="audio_1"))
    mm = prompt_contract.describe_references(refs)
    fails, warns = prompt_contract.validate_h3_prompt_mechanics(
        body, mode="ref2va", manifest=mm, length=window_frames, enforce_min_words=True
    )
    if fails or warns:
        raise RuntimeError(f"Generated window {window_idx} prompt failed contract checks: fails={fails}, warns={warns}")
    return body


def find_unfilled_placeholders(prompts: list[str]) -> list[str]:
    """Return any prompt indices still carrying unedited `[PASTE_LYRICS_*]` placeholders."""
    hits: list[str] = []
    for idx, txt in enumerate(prompts, 1):
        found = re.findall(r"\[PASTE_LYRICS_[^\]]*\]", txt)
        if found:
            hits.append(f"window_{idx:02d}: {', '.join(found)}")
    return hits


# ---------------------------------------------------------------------------
# 3. API Graph Construction (`FrozenVideoCache` Pass 2 + Per-Window Schedulers)
# ---------------------------------------------------------------------------


def _wire_sol(g: dict[str, Any], *, in_id: str, shift_id: str, attn_id: str, sol_id: str) -> str:
    g[shift_id] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": [in_id, 0], "shift_video": 12.0, "shift_audio": 3.0}}
    g[attn_id] = {"class_type": "ModelAttentionBackend", "inputs": {"model": [shift_id, 0], "attention": "comfy kitchen attention"}}
    g[sol_id] = {
        "class_type": "MiniMaxH3Sol",
        "inputs": {
            "model": [attn_id, 0],
            "tau": 1.0,
            "start_percent": 0.2,
            "end_percent": 1.0,
            "min_tokens": 12288,
            "sink_conditioning": "exact_kv_and_all_rows",
            "verbose": True,
            "dense_blocks": "49",
            "quantizer": "rotated",
            "token_routing": "off",
            "rows": "per segment",
            "rows.text_rows": "exact",
            "rows.reference_rows": "exact",
            "rows.audio_rows": "exact",
            "rows.video_reference_rows": "routed",
        },
    }
    return sol_id


def build_cover_graph(
    manifest: dict[str, Any],
    prompts: list[str],
    *,
    window_denoises: list[float],
    pass2_steps: int = 12,
    use_frozen_video_cache: bool = False,
    aud_to_vid_gain: float = 1.0,
    mic_prompts: list[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build single- or multi-window ComfyUI API graph with stock-RoPE chaining (Pass-2 FVC opt-in)."""
    windows = manifest["windows"]
    n_win = len(windows)
    if len(prompts) != n_win or len(window_denoises) != n_win:
        raise ValueError(f"Mismatch: {n_win} windows, {len(prompts)} prompts, {len(window_denoises)} denoises")

    win_frames = int(manifest["window_frames"])
    ctx_frames = int(manifest["context_frames"])
    stride_s = round(int(manifest["stride_frames"]) / FPS, 3)
    ctx_s = round(ctx_frames / FPS, 3)
    width = int(manifest.get("width", DEFAULT_WIDTH))
    height = int(manifest.get("height", DEFAULT_HEIGHT))
    ref_images: list[str] = list(manifest["ref_images_rel"])
    if not ref_images:
        raise ValueError("manifest['ref_images_rel'] must not be empty")
    ref_register = manifest.get("ref_register", "single")
    out_prefix_dir = f"Video/cover_jobs/{manifest['job']}"

    g: dict[str, Any] = {}
    g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": h3_config.MODELS["unet_ref2va"], "weight_dtype": "default"}}
    g["2"] = {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": h3_config.MODELS["clip"], "device": "default"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["video_vae"]}}
    g["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["audio_vae"]}}
    g["18"] = {
        "class_type": "MiniMaxH3LoRABranch",
        "inputs": {
            "model": ["1", 0],
            "lora_name": h3_config.PIKA_REF2VA_SVD995_LORA,
            "strength": 1.0,
            "modules": "all",
            "blocks": "all",
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }
    pika_sol = _wire_sol(g, in_id="18", shift_id="19", attn_id="58", sol_id="21")
    base_sol = _wire_sol(g, in_id="1", shift_id="119", attn_id="158", sol_id="121")

    pic_policy = {"size_policy": "tiered", "qwen_view": "separate", "qwen_view.qwen_short_edge": 512}
    last_pic_node = ""
    for i, img_rel in enumerate(ref_images, 1):
        ld_id = f"15_p{i}"
        app_id = f"50_p{i}"
        g[ld_id] = {"class_type": "LoadImage", "inputs": {"image": img_rel}}
        app_inputs: dict[str, Any] = {"image": [ld_id, 0], **pic_policy}
        if last_pic_node:
            app_inputs["references"] = [last_pic_node, 0]
        g[app_id] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": app_inputs}
        last_pic_node = app_id

    def _add_ref_enc(tag: str, aud_rel: str) -> str:
        aud_ld = f"33_{tag}"
        aud_app = f"52_{tag}"
        enc_id = f"132_{tag}" if ref_register == "split" else "132"
        g[aud_ld] = {"class_type": "LoadAudio", "inputs": {"audio": aud_rel}}
        g[aud_app] = {"class_type": "MiniMaxH3AppendRefAudio", "inputs": {"audio": [aud_ld, 0], "references": [last_pic_node, 0]}}
        g[enc_id] = {
            "class_type": "MiniMaxH3EncodeReferences",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "references": [aud_app, 0],
                "width": width,
                "height": height,
                "length": win_frames,
                "video_policy": "comfy",
                "image_policy": "comfy",
            },
        }
        return enc_id

    if ref_register == "split" and manifest.get("ref_vocal_low_rel") and manifest.get("ref_vocal_high_rel"):
        enc_low = _add_ref_enc("low", manifest["ref_vocal_low_rel"])
        enc_high = _add_ref_enc("high", manifest["ref_vocal_high_rel"])
        enc_map = {"low": enc_low, "high": enc_high}
    else:
        enc_single = _add_ref_enc("main", manifest["ref_vocal_rel"])
        enc_map = {"low": enc_single, "high": enc_single}

    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": manifest["scaffold_rel"]}}
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
    g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": h3_config.PIKA_MANUAL_SIGMAS}}
    g["107"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}

    use_fvc = bool(use_frozen_video_cache and aud_to_vid_gain >= 1.0)
    if use_fvc:
        g["122_fvc"] = {
            "class_type": "MiniMaxH3FrozenVideoCache",
            "inputs": {
                "model": [base_sol, 0],
                "precision": "int4",
                "refresh": False,
                "refresh_every": 2,
                "verify": False,
                "halo": 0,
            },
        }
        pass2_base_model = "122_fvc"
    else:
        pass2_base_model = base_sol

    if aud_to_vid_gain >= 1.0 and mic_prompts is None:
        g["45"] = {
            "class_type": "MiniMaxH3QueryRouter",
            "inputs": {
                "model": [pika_sol, 0],
                "section_split": "soundscape_to_audio",
                "video_block_band": "all",
                "audio_block_band": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        }

    for k, win in enumerate(windows, 1):
        sfx = f"w{k}"
        enc_id = enc_map.get(win.get("register_role", "high"), next(iter(enc_map.values())))
        g[f"5_{sfx}"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": [enc_id, 0], "prompt": prompts[k - 1]}}
        g[f"26_{sfx}"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [f"5_{sfx}", 0], "samples": [enc_id, 1]}}

        if mic_prompts is not None or aud_to_vid_gain < 1.0:
            mic_txt = mic_prompts[k - 1] if mic_prompts is not None else prompts[k - 1]
            g[f"105_{sfx}"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": [enc_id, 0], "prompt": mic_txt}}
            g[f"126_{sfx}"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [f"105_{sfx}", 0], "samples": [enc_id, 1]}}
            g[f"45_{sfx}"] = {
                "class_type": "MiniMaxH3QueryRouter",
                "inputs": {
                    "model": [pika_sol, 0],
                    "section_split": "off",
                    "video_block_band": "all",
                    "audio_block_band": "all",
                    "aud_to_vid_gain": float(aud_to_vid_gain),
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                    "audio_conditioning": [f"126_{sfx}", 0],
                },
            }
            g[f"145_{sfx}"] = {
                "class_type": "MiniMaxH3QueryRouter",
                "inputs": {
                    "model": [base_sol, 0],
                    "section_split": "off",
                    "video_block_band": "all",
                    "audio_block_band": "all",
                    "aud_to_vid_gain": float(aud_to_vid_gain),
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                    "audio_conditioning": [f"126_{sfx}", 0],
                },
            }
            pass1_model = f"45_{sfx}"
            pass2_model = f"145_{sfx}"
            pass2_cond = f"126_{sfx}"
        else:
            pass1_model = "45"
            pass2_model = pass2_base_model
            pass2_cond = f"26_{sfx}"

        start_rel_s = round((k - 1) * stride_s, 3)
        if n_win == 1:
            g[f"49_{sfx}"] = {
                "class_type": "MiniMaxH3FreezeAudio",
                "inputs": {
                    "latent": [f"26_{sfx}", 1],
                    "audio_vae": ["4", 0],
                    "audio": ["48", 0],
                    "start_seconds": 0.0,
                    "audio_mask": 0.0,
                    "level": "clip_guard",
                },
            }
        else:
            fw_inputs: dict[str, Any] = {
                "latent": [f"26_{sfx}", 1],
                "audio_vae": ["4", 0],
                "audio": ["48", 0],
                "start_seconds": start_rel_s,
                "context_frames": ctx_frames,
                "audio_mask": 0.0,
                "level": "clip_guard",
                "context_noise": 0.35,
                "context_profile": "seam_taper",
            }
            if k > 1:
                prev_sfx = f"w{k - 1}"
                g[f"trim_prev_{sfx}"] = {
                    "class_type": "TrimAudioDuration",
                    "inputs": {"audio": [f"112_{prev_sfx}", 0], "start_index": stride_s, "duration": ctx_s},
                }
                g[f"trim_new_{sfx}"] = {
                    "class_type": "TrimAudioDuration",
                    "inputs": {"audio": ["48", 0], "start_index": round(start_rel_s + ctx_s, 3), "duration": stride_s},
                }
                g[f"hyb_aud_{sfx}"] = {
                    "class_type": "AudioConcat",
                    "inputs": {"audio1": [f"trim_prev_{sfx}", 0], "audio2": [f"trim_new_{sfx}", 0], "direction": "after"},
                }
                fw_inputs["previous"] = [f"110_{prev_sfx}", 0]
                fw_inputs["audio"] = [f"hyb_aud_{sfx}", 0]
                fw_inputs["start_seconds"] = 0.0
            g[f"49_{sfx}"] = {"class_type": "MiniMaxH3FreezeAudioWindow", "inputs": fw_inputs}

        g[f"6_{sfx}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + (k - 1)}}
        g[f"9_{sfx}"] = {"class_type": "BasicGuider", "inputs": {"model": [pass1_model, 0], "conditioning": [f"26_{sfx}", 0]}}
        g[f"10_{sfx}"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [f"6_{sfx}", 0],
                "guider": [f"9_{sfx}", 0],
                "sampler": ["7", 0],
                "sigmas": ["60", 0],
                "latent_image": [f"49_{sfx}", 0],
            },
        }
        g[f"152_{sfx}"] = {"class_type": "MiniMaxH3AudioRefineMask", "inputs": {"latent": [f"10_{sfx}", 0], "video_mask": 0.0, "audio_mask": 1.0}}
        g[f"160_{sfx}"] = {
            "class_type": "BasicScheduler",
            "inputs": {"model": [pass2_model, 0], "scheduler": "simple", "steps": int(pass2_steps), "denoise": float(window_denoises[k - 1])},
        }
        g[f"106_{sfx}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + 17 + (k - 1)}}
        g[f"109_{sfx}"] = {"class_type": "BasicGuider", "inputs": {"model": [pass2_model, 0], "conditioning": [pass2_cond, 0]}}
        g[f"110_{sfx}"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [f"106_{sfx}", 0],
                "guider": [f"109_{sfx}", 0],
                "sampler": ["107", 0],
                "sigmas": [f"160_{sfx}", 0],
                "latent_image": [f"152_{sfx}", 0],
            },
        }
        g[f"111_{sfx}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"110_{sfx}", 0], "vae": ["3", 0]}}
        g[f"112_{sfx}"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [f"110_{sfx}", 0], "vae": ["4", 0]}}
        g[f"113_{sfx}"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [f"111_{sfx}", 0],
                "audio": [f"112_{sfx}", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": f"{out_prefix_dir}/song_{sfx}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        }
        g[f"114_{sfx}"] = {
            "class_type": "SaveAudio",
            "inputs": {"audio": [f"112_{sfx}", 0], "filename_prefix": f"{out_prefix_dir}/song_aud_{sfx}"},
        }
    return g


# ---------------------------------------------------------------------------
# 4. Stitching & Pitch-Lock Scoring
# ---------------------------------------------------------------------------


def _decode_rgb_frames(mp4_path: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(mp4_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    n = len(raw) // (height * width * 3)
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(n, height, width, 3)


def stitch_cover_windows(
    manifest: dict[str, Any],
    output_dir: Path,
    window_mp4s: list[Path],
    window_flacs: list[Path] | None = None,
) -> dict[str, Any]:
    """Concatenate decoded H3 video frames and cosine-crossfade 32 kHz H3 audio into master MP4."""
    job = manifest["job"]
    out_dir = output_dir / "Video" / "cover_jobs" / job
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{job}_pure_h3_cover-audio.mp4"

    width = int(manifest.get("width", DEFAULT_WIDTH))
    height = int(manifest.get("height", DEFAULT_HEIGHT))
    win_frames = int(manifest["window_frames"])
    ctx_frames = int(manifest["context_frames"])
    stride_frames = int(manifest["stride_frames"])
    total_frames = int(manifest["total_frames"])

    vid_chunks: list[np.ndarray] = []
    seam_metrics: list[dict[str, Any]] = []
    local_means: list[float] = []
    for idx, mp4 in enumerate(window_mp4s):
        frames = _decode_rgb_frames(mp4, width=width, height=height)
        chunk = frames if idx == 0 else frames[ctx_frames:]
        adj = np.mean(np.abs(chunk[1:].astype(np.float32) - chunk[:-1].astype(np.float32)), axis=(1, 2, 3))
        local_means.append(float(np.mean(adj)))
        if vid_chunks:
            prev_last = vid_chunks[-1][-1].astype(np.float32)
            cur_first = chunk[0].astype(np.float32)
            seam_l1 = float(np.mean(np.abs(cur_first - prev_last)))
            base_adj = float(np.mean(adj[:10])) if len(adj) >= 10 else float(np.mean(adj))
            seam_metrics.append(
                {
                    "join": f"w{idx}->w{idx + 1}",
                    "frame_transition": f"f{sum(len(c) for c in vid_chunks) - 1}->f{sum(len(c) for c in vid_chunks)}",
                    "seam_l1": round(seam_l1, 3),
                    "post_10f_adj_mean": round(base_adj, 3),
                    "seam_ratio": round(seam_l1 / max(base_adj, 1e-6), 3),
                }
            )
        vid_chunks.append(chunk)

    all_frames = np.concatenate(vid_chunks, axis=0)
    assert len(all_frames) == total_frames, f"Expected {total_frames} frames, got {len(all_frames)}"

    sr = stem_sep.SCAFFOLD_SR
    total_samps = int(round(total_frames / FPS * sr))
    audio_sources = window_flacs if (window_flacs and len(window_flacs) == len(window_mp4s)) else window_mp4s
    stitched = np.zeros((2, total_samps), dtype=np.float32)

    if len(audio_sources) == 1:
        a = _pad_or_trim_np(_load_audio_np(audio_sources[0], sr=sr, channels=2), total_samps)
        stitched[:, :total_samps] = a
    else:
        overlap_samps = int(round(ctx_frames / FPS * sr))
        stride_samps = int(round(stride_frames / FPS * sr))
        win_samps = int(round(win_frames / FPS * sr))
        xfade_samps = min(overlap_samps, int(round(0.300 * sr)))
        xfade_offset = (overlap_samps - xfade_samps) // 2
        t = np.linspace(0.0, math.pi / 2.0, xfade_samps, dtype=np.float32)
        fade_in, fade_out = np.sin(t)[None, :], np.cos(t)[None, :]

        for idx, ap in enumerate(audio_sources):
            a = _pad_or_trim_np(_load_audio_np(ap, sr=sr, channels=2), win_samps)
            pos = idx * stride_samps
            if idx == 0:
                stitched[:, :win_samps] = a
            else:
                x0 = pos + xfade_offset
                x1 = x0 + xfade_samps
                stitched[:, x0:x1] = stitched[:, x0:x1] * fade_out + a[:, xfade_offset : xfade_offset + xfade_samps] * fade_in
                stitched[:, x1 : pos + win_samps] = a[:, xfade_offset + xfade_samps :]

    tmp_wav = out_dir / "_stitched_h3_audio.wav"
    stem_sep._write_wav_pcm16(tmp_wav, torch.from_numpy(stitched), sr)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0",
        "-i", str(tmp_wav),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(all_frames.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    tmp_wav.unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError(f"ffmpeg stitch failed with code {rc}")
    return {
        "stitched_mp4": str(out_mp4.relative_to(output_dir)),
        "frames": int(len(all_frames)),
        "duration_s": round(len(all_frames) / FPS, 3),
        "audio_samples_32k": int(total_samps),
        "seams": seam_metrics,
        "mean_adj_l1": round(float(np.mean(local_means)), 3),
    }


def _voicing_gap_breakdown(out_voc_24k: torch.Tensor, tgt_voc_24k: torch.Tensor, *, relaxed_thr: float = 0.28) -> dict[str, Any]:
    fo_s, cmndf_o, rms_o = (t.numpy() for t in yin_probe.yin_f0(out_voc_24k, sr=24000))
    ft_s = yin_probe.yin_f0(tgt_voc_24k, sr=24000)[0].numpy()
    fo_rel = yin_probe.yin_f0(out_voc_24k, sr=24000, threshold=relaxed_thr)[0].numpy()

    n = min(len(fo_s), len(ft_s), len(fo_rel))
    fo_s, cmndf_o, rms_o, ft_s, fo_rel = fo_s[:n], cmndf_o[:n], rms_o[:n], ft_s[:n], fo_rel[:n]
    tgt_v = np.isfinite(ft_s) & (ft_s > 0)
    out_v = np.isfinite(fo_s) & (fo_s > 0)
    n_tgt = int(np.sum(tgt_v))
    gap = tgt_v & (~out_v)
    n_sil = int(np.sum(gap & (rms_o < yin_probe.DEFAULT_MIN_DBFS)))
    n_raspy = int(np.sum(gap & (rms_o >= yin_probe.DEFAULT_MIN_DBFS) & (cmndf_o < relaxed_thr)))
    n_aper = int(np.sum(gap & (rms_o >= yin_probe.DEFAULT_MIN_DBFS) & (cmndf_o >= relaxed_thr)))

    rel_score = pitch_lock_metrics.score_f0(fo_rel, ft_s)
    rel_null = pitch_lock_metrics.lag_null(fo_rel, ft_s)
    rel_cov = float(np.sum(tgt_v & np.isfinite(fo_rel) & (fo_rel > 0))) / max(1, n_tgt)
    return {
        "n_tgt_voiced": n_tgt,
        "n_gap": int(np.sum(gap)),
        "below_min_dbfs_frac_of_tgt": round(n_sil / max(1, n_tgt), 3),
        "cmndf_015_to_028_frac_of_tgt": round(n_raspy / max(1, n_tgt), 3),
        "cmndf_ge_028_frac_of_tgt": round(n_aper / max(1, n_tgt), 3),
        "relaxed_thr": float(relaxed_thr),
        "relaxed_voiced_overlap_frac": round(rel_cov, 3),
        "relaxed_med_abs_cent_err": round(rel_score["med_abs_cent_err"], 1) if rel_score else None,
        "relaxed_in_tune_frac_50c": round(rel_score["in_tune_frac_50c"], 3) if rel_score else None,
        "relaxed_beats_frac": rel_null["med_abs_cent_err"]["beats_frac"] if rel_null else None,
        "relaxed_beats_all_shifts": bool(rel_null and rel_null["med_abs_cent_err"]["beats_frac"] == 1.0),
    }


def _find_latest(vid_dir: Path, prefix: str, ext: str) -> Path | None:
    m = sorted(vid_dir.glob(f"{prefix}*{ext}"))
    return m[-1] if m else None


def score_cover_job(manifest: dict[str, Any], *, input_dir: Path, output_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Separate rendered master & window vocals via `MelBand-RoFormer` and score against shifted target."""
    job = manifest["job"]
    out_dir = output_dir / "Video" / "cover_jobs" / job
    stems_out = out_dir / "vocal_stems"
    stems_out.mkdir(parents=True, exist_ok=True)
    master_mp4 = out_dir / f"{job}_pure_h3_cover-audio.mp4"
    if not master_mp4.is_file():
        raise FileNotFoundError(f"Missing stitched cover MP4: {master_mp4}")

    tgt_full_24k = _load_audio_np(input_dir / manifest["tgt_voc_rel"], sr=24000, channels=1)

    def _separate_cached(src_media: Path, stem_wav: Path) -> torch.Tensor:
        if (
            stem_wav.is_file()
            and stem_wav.stat().st_size > 44100
            and stem_wav.stat().st_mtime >= src_media.stat().st_mtime
        ):
            return torch.from_numpy(_load_audio_np(stem_wav, sr=24000, channels=1))
        wav_44k = torch.from_numpy(_load_audio_np(src_media, sr=44100, channels=2))
        dev = stem_sep._resolve_compute_device(None)
        voc_44k, _ = stem_sep.separate_vocals_roformer(wav_44k, dev)
        stem_sep._write_wav_pcm16(stem_wav, voc_44k, 44100)
        return TAF.resample(voc_44k.mean(dim=0, keepdim=True), 44100, 24000).squeeze(0)

    master_voc = _separate_cached(master_mp4, stems_out / f"{job}_voc.wav")
    master_metrics = pitch_lock_metrics.vocal_pitch_metrics(master_voc, torch.from_numpy(tgt_full_24k))
    master_metrics["voicing_gap"] = _voicing_gap_breakdown(master_voc, torch.from_numpy(tgt_full_24k))
    master_metrics["file"] = str(master_mp4.relative_to(output_dir))

    stride_24k = int(round(int(manifest["stride_frames"]) / FPS * 24000))
    win_24k = int(round(int(manifest["window_frames"]) / FPS * 24000))
    win_reports: dict[str, Any] = {}
    for k in range(1, len(manifest["windows"]) + 1):
        w_flac = _find_latest(out_dir, f"song_aud_w{k}_", ".flac")
        w_mp4 = _find_latest(out_dir, f"song_w{k}_", "-audio.mp4")
        src_aud = w_flac or w_mp4
        if src_aud is None:
            continue
        w_voc = _separate_cached(src_aud, stems_out / f"song_w{k}_voc.wav")
        s0 = (k - 1) * stride_24k
        tgt_slice = torch.from_numpy(tgt_full_24k[s0 : s0 + win_24k].copy())
        wm = pitch_lock_metrics.vocal_pitch_metrics(w_voc, tgt_slice)
        wm["voicing_gap"] = _voicing_gap_breakdown(w_voc, tgt_slice)
        win_reports[f"w{k}"] = wm

    report = {
        "job": job,
        "semitones": manifest["semitones"],
        "ref_register": manifest.get("ref_register", "single"),
        **master_metrics,
        "windows": win_reports,
    }
    stitch_p = data_dir / "song_stitch_meta.json"
    if stitch_p.is_file():
        report.update(json.loads(stitch_p.read_text(encoding="utf-8")))
    (data_dir / "song_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 5. CLI Entrypoints (`prepare`, `render`, `score`)
# ---------------------------------------------------------------------------


def _stage_ref_image(src: Path, dst: Path) -> None:
    ext = src.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        shutil.copy2(src, dst)
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-frames:v", "1", str(dst)], check=True)


def cmd_prepare(args: argparse.Namespace) -> int:
    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    paths = _job_paths(args.job, input_dir=input_dir, output_dir=output_dir)
    for key in ("data_dir", "prompts_dir", "graphs_dir", "in_dir", "stems_dir", "out_dir", "slices_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    ref_images_rel: list[str] = []
    for idx, src_img in enumerate(args.ref_images, 1):
        p = Path(src_img)
        if not p.is_file():
            raise FileNotFoundError(f"Missing --ref-images file: {p}")
        dst = paths["in_dir"] / f"ref_pic_{idx:02d}{p.suffix.lower() if p.suffix else '.png'}"
        _stage_ref_image(p, dst)
        ref_images_rel.append(str(dst.relative_to(input_dir)))

    ref_audio_p = Path(args.ref_audio)
    if not ref_audio_p.is_file():
        raise FileNotFoundError(f"Missing --ref-audio file: {ref_audio_p}")
    ref_stems_dir = paths["stems_dir"] / "ref"
    ref_voc_p = ref_stems_dir / "vocals.wav"
    if not ref_voc_p.is_file() or ref_voc_p.stat().st_mtime < ref_audio_p.stat().st_mtime:
        stem_sep.separate_to_dir(ref_audio_p, ref_stems_dir, recipe="vocals")
    ref_voc_44k, ref_sr = stem_sep._load_audio_native(ref_voc_p)

    ref_main = pick_best_ref_vocal_window(
        ref_voc_44k,
        sr=ref_sr,
        register_bias="balanced",
        scan_s=args.ref_scan_s,
        explicit_start_s=args.ref_start_s,
    )
    ref_vocal_path = paths["in_dir"] / "ref_vocal_192f.wav"
    stem_sep._write_wav_pcm16(ref_vocal_path, ref_main["wav_32k"], stem_sep.SCAFFOLD_SR)
    ref_vocal_rel = str(ref_vocal_path.relative_to(input_dir))

    ref_low_rel: str | None = None
    ref_high_rel: str | None = None
    ref_low_info: dict[str, Any] | None = None
    ref_high_info: dict[str, Any] | None = None
    if args.ref_register == "split":
        ref_low = pick_best_ref_vocal_window(ref_voc_44k, sr=ref_sr, register_bias="low", scan_s=args.ref_scan_s)
        ref_high = pick_best_ref_vocal_window(ref_voc_44k, sr=ref_sr, register_bias="high", scan_s=args.ref_scan_s)
        low_p = paths["in_dir"] / "ref_vocal_low_192f.wav"
        high_p = paths["in_dir"] / "ref_vocal_high_192f.wav"
        stem_sep._write_wav_pcm16(low_p, ref_low["wav_32k"], stem_sep.SCAFFOLD_SR)
        stem_sep._write_wav_pcm16(high_p, ref_high["wav_32k"], stem_sep.SCAFFOLD_SR)
        ref_low_rel = str(low_p.relative_to(input_dir))
        ref_high_rel = str(high_p.relative_to(input_dir))
        ref_low_info = {k: v for k, v in ref_low.items() if k != "wav_32k"}
        ref_high_info = {k: v for k, v in ref_high.items() if k != "wav_32k"}

    tgt_song_p = Path(args.target_song)
    if not tgt_song_p.is_file():
        raise FileNotFoundError(f"Missing --target-song file: {tgt_song_p}")
    tgt_stems_dir = paths["stems_dir"] / "target"
    tgt_voc_raw_p = tgt_stems_dir / "vocals.wav"
    if not tgt_voc_raw_p.is_file() or tgt_voc_raw_p.stat().st_mtime < tgt_song_p.stat().st_mtime:
        stem_sep.separate_to_dir(tgt_song_p, tgt_stems_dir, recipe="cascade")

    tgt_voc_24k = torch.from_numpy(_load_audio_np(tgt_voc_raw_p, sr=24000, channels=1))
    explicit_formant: bool | None
    if args.preserve_formant:
        explicit_formant = True
    elif args.vocal_formant == "true":
        explicit_formant = True
    elif args.vocal_formant == "false":
        explicit_formant = False
    else:
        explicit_formant = None
    semi_info = choose_cover_semitones(
        ref_main["f0_stats"],
        tgt_voc_24k,
        explicit_semitones=args.semitones,
        explicit_formant=explicit_formant,
    )
    win_plan = plan_breath_aligned_windows(
        tgt_voc_24k,
        n_windows=args.windows,
        duration_s=args.duration_s,
        start_s=args.start_s,
        window_frames=args.window_frames,
        context_frames=args.context_frames,
    )

    scaffold_p = paths["in_dir"] / "scaffold_full_32k.wav"
    tgt_voc_p = paths["in_dir"] / "target_voc_shifted_32k.wav"
    raw_orig_p = paths["in_dir"] / "target_raw_orig_32k.wav"
    scaf_meta = stem_sep.build_cover_scaffold(
        tgt_stems_dir,
        start_s=win_plan["start_s"],
        duration_s=win_plan["duration_s"],
        semitones=semi_info["semitones"],
        harmonic_db=args.harmonic_db,
        bass_db=args.bass_db,
        vocal_formant=semi_info["vocal_formant"],
        out_wav_32k=scaffold_p,
        out_voc_wav_32k=tgt_voc_p,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{win_plan['start_s']:.3f}", "-t", f"{win_plan['duration_s']:.3f}",
            "-i", str(tgt_stems_dir / "mix.wav"), "-ar", "32000", "-ac", "2", str(raw_orig_p),
        ],
        check=True,
    )

    # Export per-window 32 kHz scaffold preview slices into `$H3_OUTPUT_DIR` for easy lyric timing review
    scaf_32k = _load_audio_np(scaffold_p, sr=stem_sep.SCAFFOLD_SR, channels=2)
    win_samps = int(round(win_plan["window_frames"] / FPS * stem_sep.SCAFFOLD_SR))
    stride_samps = int(round(win_plan["stride_frames"] / FPS * stem_sep.SCAFFOLD_SR))
    for k in range(win_plan["n_windows"]):
        s0 = k * stride_samps
        slice_arr = _pad_or_trim_np(scaf_32k[:, s0 : s0 + win_samps], win_samps)
        stem_sep._write_wav_pcm16(paths["slices_dir"] / f"window_{k + 1:02d}_scaffold.wav", torch.from_numpy(slice_arr), stem_sep.SCAFFOLD_SR)

    lyrics_path = Path(args.lyrics) if args.lyrics else None
    lyric_pairs = parse_lyrics_for_windows(lyrics_path, win_plan["windows"])
    for k, win in enumerate(win_plan["windows"], 1):
        s1_lyr, s2_lyr = lyric_pairs[k - 1]
        p_txt = build_window_prompt_skeleton(
            window_idx=k,
            win_info=win,
            n_pictures=len(ref_images_rel),
            subject_desc=args.subject_desc,
            stage_desc=args.stage_desc,
            voice_desc=args.voice_desc,
            aesthetic_desc=args.aesthetic_desc,
            soundscape_desc=args.soundscape_desc,
            shot1_lyrics=s1_lyr,
            shot2_lyrics=s2_lyr,
            window_frames=win_plan["window_frames"],
        )
        (paths["prompts_dir"] / f"window_{k:02d}.txt").write_text(p_txt + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "job": _validate_job_name(args.job),
        "width": int(args.width),
        "height": int(args.height),
        "ref_register": args.ref_register,
        "ref_images_rel": ref_images_rel,
        "ref_vocal_rel": ref_vocal_rel,
        "ref_vocal_low_rel": ref_low_rel,
        "ref_vocal_high_rel": ref_high_rel,
        "scaffold_rel": str(scaffold_p.relative_to(input_dir)),
        "tgt_voc_rel": str(tgt_voc_p.relative_to(input_dir)),
        "raw_orig_rel": str(raw_orig_p.relative_to(input_dir)),
        "semitones": semi_info["semitones"],
        "inst_semitones": scaf_meta["inst_semitones"],
        "vocal_formant": semi_info["vocal_formant"],
        "harmonic_db": float(args.harmonic_db),
        "bass_db": float(args.bass_db),
        "ref_selection": {k: v for k, v in ref_main.items() if k != "wav_32k"},
        "ref_selection_low": ref_low_info,
        "ref_selection_high": ref_high_info,
        "pitch_analysis": semi_info,
        **win_plan,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "job": manifest["job"],
                "semitones": manifest["semitones"],
                "vocal_formant": manifest["vocal_formant"],
                "start_s": manifest["start_s"],
                "duration_s": manifest["duration_s"],
                "n_windows": manifest["n_windows"],
                "seam_max_dbfs": manifest["seam_max_dbfs"],
                "ref_vocal_start_s": manifest["ref_selection"]["start_s"],
                "ref_vocal_f0": manifest["ref_selection"]["f0_stats"],
                "warnings": semi_info["warnings"],
                "prompts_dir": str(paths["prompts_dir"].relative_to(REPO)),
            },
            indent=2,
        )
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    paths = _job_paths(args.job, input_dir=input_dir, output_dir=output_dir)
    if not paths["manifest"].is_file():
        raise FileNotFoundError(f"Missing manifest {paths['manifest']}; run `prepare` first")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    n_win = int(manifest["n_windows"])

    prompts = [(paths["prompts_dir"] / f"window_{k:02d}.txt").read_text(encoding="utf-8").strip() for k in range(1, n_win + 1)]
    unfilled = find_unfilled_placeholders(prompts)
    if unfilled and not args.allow_placeholders:
        raise SystemExit(
            "Unfilled lyric placeholder(s) found in prompts; edit prompt files or pass --lyrics / --allow-placeholders:\n  "
            + "\n  ".join(unfilled)
        )

    denoises = resolve_window_denoises(manifest["windows"], args.denoise)
    use_fvc_flag = bool(args.frozen_video_cache and not args.no_frozen_video_cache)
    graph = build_cover_graph(
        manifest,
        prompts,
        window_denoises=denoises,
        pass2_steps=args.pass2_steps,
        use_frozen_video_cache=use_fvc_flag,
        aud_to_vid_gain=args.aud_to_vid_gain,
        seed=args.seed,
    )
    gp = paths["graphs_dir"] / "cover_graph.json"
    gp.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")

    preflight_proc = subprocess.run(
        [sys.executable, str(REPO / "bench" / "preflight_graph.py"), str(gp)],
        capture_output=True,
        text=True,
    )
    pf_out = (preflight_proc.stdout + "\n" + preflight_proc.stderr).strip()
    pf_ok = preflight_proc.returncode == 0 and "\nFAIL" not in f"\n{pf_out}"
    print(f"[preflight] {manifest['job']} ({n_win}w, denoises={denoises}, fvc={use_fvc_flag and args.aud_to_vid_gain >= 1.0}): {'PASS' if pf_ok else 'FAIL'}")
    if not pf_ok:
        print(pf_out)
        return 1
    if args.dry_run:
        return 0

    while stem_sep._comfy_queue_busy():
        import time
        time.sleep(10.0)
    import time
    t0 = time.monotonic()
    pid = comfy_client.submit_prompt(graph, base_url=args.url, client_id=str(uuid.uuid4()))
    hist = comfy_client.wait_for_prompt(pid, base_url=args.url, timeout=max(3600.0, n_win * 1200.0))
    dt = comfy_client.execution_duration_s(hist) or (time.monotonic() - t0)

    vid_dir = paths["out_dir"]
    w_mp4s = [_find_latest(vid_dir, f"song_w{k}_", "-audio.mp4") for k in range(1, n_win + 1)]
    w_flacs = [_find_latest(vid_dir, f"song_aud_w{k}_", ".flac") for k in range(1, n_win + 1)]
    if any(p is None for p in w_mp4s):
        raise RuntimeError(f"Missing rendered window MP4s in {vid_dir}: {w_mp4s}")
    stitch_info = stitch_cover_windows(
        manifest,
        output_dir,
        [p for p in w_mp4s if p is not None],
        [p for p in w_flacs if p is not None] if all(w_flacs) else None,
    )
    stitch_info["wall_s"] = round(dt, 2)
    stitch_info["window_denoises"] = denoises
    (paths["data_dir"] / "song_stitch_meta.json").write_text(json.dumps(stitch_info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stitch_info, indent=2))

    if not args.no_score:
        report = score_cover_job(manifest, input_dir=input_dir, output_dir=output_dir, data_dir=paths["data_dir"])
        print(json.dumps(report, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    paths = _job_paths(args.job, input_dir=input_dir, output_dir=output_dir)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    report = score_cover_job(manifest, input_dir=input_dir, output_dir=output_dir, data_dir=paths["data_dir"])
    print(json.dumps(report, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generalized 2-step pure-H3 cover music-video workflow.")
    sub = ap.add_subparsers(dest="command", required=True)

    sp_prep = sub.add_parser("prepare", help="Separate stems, select ref vocal, align breath seams, and emit prompt skeletons.")
    sp_prep.add_argument("--job", required=True, help="Short job slug (creates data/cover_jobs/<job> and $H3_INPUT_DIR/cover_jobs/<job>).")
    sp_prep.add_argument("--ref-audio", required=True, help="Singer reference audio/video path.")
    sp_prep.add_argument("--ref-images", nargs="+", required=True, help="1..K singer/scene reference images.")
    sp_prep.add_argument("--target-song", required=True, help="Target song audio/video path.")
    sp_prep.add_argument("--lyrics", default=None, help="Optional .lrc or plain-text lyrics file.")
    sp_prep.add_argument("--semitones", type=float, default=None, help="Override auto integer semitone shift.")
    sp_prep.add_argument("--vocal-formant", choices=["auto", "true", "false"], default="auto", help="RubberBand formant preservation ('auto': False unless explicitly enabled).")
    sp_prep.add_argument("--preserve-formant", action="store_true", help="Deprecated alias for `--vocal-formant true`.")
    sp_prep.add_argument("--ref-start-s", type=float, default=None, help="Pin exact 8.0 s reference vocal slice start in seconds.")
    sp_prep.add_argument("--ref-scan-s", type=float, default=90.0, help="Max leading seconds of ref audio to scan (<=0 for full file, default 90.0).")
    sp_prep.add_argument("--ref-register", choices=["single", "split"], default="single", help="Single 8.0s <Audio 1> (default) or low/high register split.")
    sp_prep.add_argument("--windows", type=int, default=None, help="Explicit number of chained windows.")
    sp_prep.add_argument("--duration-s", type=float, default=None, help="Target duration in seconds when --windows is omitted.")
    sp_prep.add_argument("--start-s", type=float, default=None, help="Override breath-aligned start timestamp in seconds.")
    sp_prep.add_argument("--window-frames", type=int, default=DEFAULT_WIN_FRAMES, help="Frames per window (default 294 = 12.25 s).")
    sp_prep.add_argument("--context-frames", type=int, default=DEFAULT_CTX_FRAMES, help="Overlap context frames (default 39 = 1.625 s).")
    sp_prep.add_argument("--harmonic-db", type=float, default=-14.5, help="Harmonic stem level relative to vocals in dB.")
    sp_prep.add_argument("--bass-db", type=float, default=-12.0, help="Bass stem level relative to vocals in dB.")
    sp_prep.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    sp_prep.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    sp_prep.add_argument("--subject-desc", default="solo acoustic singer-guitarist centered on stage at a dynamic black vocal microphone")
    sp_prep.add_argument("--stage-desc", default="an intimate live studio concert stage with soft warm amber key light, deep velvet curtains, and quiet distant audience in background shadow")
    sp_prep.add_argument("--voice-desc", default="raw, expressive chest-and-belt live solo singing vocal timbre matching <Audio 1>")
    sp_prep.add_argument("--aesthetic-desc", default="warm live concert broadcast television aesthetic with crisp tungsten highlights, deep shadows, subtle filmic grain, and smooth highlight rolloff")
    sp_prep.add_argument("--soundscape-desc", default="Intimate acoustic studio hall ambience with resonant wood sustain, crisp finger-pick contact on bronze acoustic strings, natural room depth, and subtle vocal breath.")

    sp_ren = sub.add_parser("render", help="Preflight, render on :8188, stitch windows, and score pitch lock.")
    sp_ren.add_argument("--job", required=True, help="Prepared job slug.")
    sp_ren.add_argument("--denoise", default="0.29", help="Scalar float ('0.29'), 'auto' (0.26 low / 0.29 high), or CSV per window.")
    sp_ren.add_argument("--pass2-steps", type=int, default=12, help="Base Pass-2 scheduler steps (default 12 -> 41 steps at 0.29).")
    sp_ren.add_argument("--aud-to-vid-gain", type=float, default=1.0, help="Audio->video attention gain (< 1.0 disables FrozenVideoCache safely).")
    sp_ren.add_argument("--frozen-video-cache", action="store_true", help="Opt-in Pass-2 MiniMaxH3FrozenVideoCache(int4) speedup (default: False).")
    sp_ren.add_argument("--no-frozen-video-cache", action="store_true", help="Deprecated no-op; Pass-2 FrozenVideoCache is off by default.")
    sp_ren.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp_ren.add_argument("--url", default=DEFAULT_URL)
    sp_ren.add_argument("--allow-placeholders", action="store_true", help="Permit unedited [PASTE_LYRICS_*] placeholders for dry runs.")
    sp_ren.add_argument("--dry-run", action="store_true", help="Build graph + run bench/preflight_graph.py without queuing :8188.")
    sp_ren.add_argument("--no-score", action="store_true", help="Skip post-render MelBand-RoFormer vocal scoring.")

    sp_scr = sub.add_parser("score", help="Re-score an already stitched cover job with MelBand-RoFormer + pitch_lock_metrics.")
    sp_scr.add_argument("--job", required=True, help="Rendered job slug.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "render":
        return cmd_render(args)
    if args.command == "score":
        return cmd_score(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
