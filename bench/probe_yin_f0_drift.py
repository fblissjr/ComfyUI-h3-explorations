#!/usr/bin/env python3
"""M2 (`docs/wiki/next_steps.md`): Pure-PyTorch `YIN` F0 pitch-drift triage helper (0 renders, zero external ASR/WavLM).

Computes clean-room vectorized `YIN` fundamental frequency (`F0`, `65..600 Hz`) on
`-audio.mp4` or `.wav` tracks via `torch.fft.rfft`/`irfft`, accepts optional
`voice_spans.json` from `bench/voice_spans.py` (`voiced_spans_inclusive` + `fps`),
and reports median semitone shift + IQR spread against a `strength = 0.0` baseline
as advisory triage notes (never hard verdicts).

Usage:
    <comfy-venv-python> bench/probe_yin_f0_drift.py --clip TAKE-audio.mp4 [--voice-spans voice_spans.json]
    <comfy-venv-python> bench/probe_yin_f0_drift.py --baseline BASE-audio.mp4 --steered STEER1-audio.mp4 [STEER2-audio.mp4 ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import torch

DEFAULT_SR = 24000
DEFAULT_FRAME_LEN = 1024
DEFAULT_HOP_LEN = 240
DEFAULT_FMIN = 65.0
DEFAULT_FMAX = 600.0
DEFAULT_THRESHOLD = 0.15
DEFAULT_MIN_DBFS = -45.0


def load_mono_audio(
    path: Path,
    *,
    sr: int = DEFAULT_SR,
) -> torch.Tensor:
    """Decode mono float32 waveform `[-1, 1]` at `sr` via `ffmpeg`."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sr)),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg audio decode failed for {path.name}: {err or proc.returncode}")
    if not proc.stdout:
        raise RuntimeError(f"no audio stream samples decoded from {path.name}")
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(arr)


def yin_f0(
    mono: torch.Tensor,
    *,
    sr: int = DEFAULT_SR,
    frame_len: int = DEFAULT_FRAME_LEN,
    hop_len: int = DEFAULT_HOP_LEN,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    threshold: float = DEFAULT_THRESHOLD,
    min_dbfs: float = DEFAULT_MIN_DBFS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized pure-PyTorch `YIN` F0 estimator returning `(f0_hz, min_cmndf, rms_dbfs)`."""
    wav = mono.detach().float().flatten()
    if wav.numel() < frame_len:
        wav = torch.nn.functional.pad(wav, (0, frame_len - wav.numel()))

    tau_min = max(2, int(sr // fmax))
    tau_max_wanted = max(tau_min + 2, int(math.ceil(sr / fmin)))
    win_len = int(frame_len) - tau_max_wanted
    if win_len < 256:
        raise ValueError(f"frame_len={frame_len} too small for fmin={fmin} at sr={sr}")
    tau_max = min(tau_max_wanted, int(frame_len) - win_len)
    if tau_max <= tau_min:
        raise ValueError(f"invalid lag window tau_min={tau_min} >= tau_max={tau_max}")

    frames = wav.unfold(0, frame_len, hop_len)  # [N_frames, frame_len]
    n_frames = int(frames.shape[0])

    # Frame RMS in dBFS for silence gating before periodicity check
    rms = torch.sqrt(torch.mean(frames[:, :win_len] ** 2, dim=-1).clamp(min=1e-14))
    rms_dbfs = 20.0 * torch.log10(rms)

    # Exact difference function d_t(tau) = E0 + E_tau - 2 * cross_corr(x0, frames)
    x0 = frames[:, :win_len]
    e0 = (x0 * x0).sum(dim=-1, keepdim=True)
    cs = torch.cat(
        [
            torch.zeros((n_frames, 1), dtype=frames.dtype, device=frames.device),
            torch.cumsum(frames * frames, dim=-1),
        ],
        dim=-1,
    )
    taus = torch.arange(0, tau_max + 1, dtype=torch.long, device=frames.device)
    e_tau = cs[:, taus + win_len] - cs[:, taus]

    n_fft = 1 << (int(frame_len) + int(win_len) - 1).bit_length()
    spec_x0 = torch.fft.rfft(x0, n=n_fft, dim=-1)
    spec_fr = torch.fft.rfft(frames, n=n_fft, dim=-1)
    corr = torch.fft.irfft(spec_fr * spec_x0.conj(), n=n_fft, dim=-1)[:, : tau_max + 1]

    diff = (e0 + e_tau - 2.0 * corr).clamp(min=0.0)
    diff[:, 0] = 0.0

    # Cumulative mean normalized difference function (CMNDF)
    cmndf = torch.ones_like(diff)
    if tau_max >= 1:
        rsum = torch.cumsum(diff[:, 1:], dim=-1).clamp(min=1e-12)
        lags = torch.arange(1, tau_max + 1, dtype=diff.dtype, device=diff.device).unsqueeze(0)
        cmndf[:, 1:] = diff[:, 1:] * lags / rsum

    sub = cmndf[:, tau_min : tau_max + 1]
    min_cmndf = sub.min(dim=-1).values
    f0 = torch.full((n_frames,), float("nan"), dtype=torch.float32)

    cmndf_np = cmndf.cpu().numpy()
    rms_np = rms_dbfs.cpu().numpy()
    for i in range(n_frames):
        if rms_np[i] < min_dbfs:
            continue
        row = cmndf_np[i]
        # Canonical YIN: first tau >= tau_min dipping below threshold, then walk to local minimum
        hits = np.flatnonzero(row[tau_min : tau_max + 1] < threshold)
        if hits.size == 0:
            continue
        tau = int(tau_min + hits[0])
        while tau + 1 <= tau_max and row[tau + 1] < row[tau]:
            tau += 1
        tau_ref = float(tau)
        if 0 < tau < tau_max:
            y0, y1, y2 = float(row[tau - 1]), float(row[tau]), float(row[tau + 1])
            denom = 2.0 * (y0 - 2.0 * y1 + y2)
            if denom > 1e-12:
                delta = max(-0.5, min(0.5, (y0 - y2) / denom))
                tau_ref += delta
        if tau_ref > 0.0:
            f0[i] = float(sr) / tau_ref

    return f0, min_cmndf.cpu(), rms_dbfs.cpu()


def load_voice_span_mask(
    voice_spans_path: Path,
    n_hops: int,
    *,
    sr: int = DEFAULT_SR,
    frame_len: int = DEFAULT_FRAME_LEN,
    hop_len: int = DEFAULT_HOP_LEN,
) -> torch.Tensor:
    """Build boolean mask selecting YIN analysis hops overlapping `voiced_spans_inclusive`."""
    doc = json.loads(voice_spans_path.read_text(encoding="utf-8"))
    fps = float(doc.get("fps") or 24.0)
    spans = list(doc.get("voiced_spans_inclusive") or [])
    mask = torch.zeros((n_hops,), dtype=torch.bool)
    if not spans:
        return mask
    hop_indices = torch.arange(n_hops, dtype=torch.float64)
    vframe_mid = ((hop_indices * hop_len + 0.5 * frame_len) / float(sr)) * fps
    for pair in spans:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        lo, hi = float(pair[0]), float(pair[1])
        mask |= (vframe_mid >= (lo - 0.5)) & (vframe_mid <= (hi + 0.5))
    return mask


def summarize_f0_series(
    f0_hz: torch.Tensor,
    *,
    span_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize voiced F0 median (Hz), IQR (semitones), and voiced fraction."""
    if span_mask is None:
        active = torch.ones_like(f0_hz, dtype=torch.bool)
    else:
        sm = span_mask.bool().flatten()
        n_f = int(f0_hz.numel())
        if sm.numel() >= n_f:
            active = sm[:n_f]
        else:
            active = torch.zeros((n_f,), dtype=torch.bool, device=f0_hz.device)
            active[: sm.numel()] = sm.to(device=f0_hz.device)
    n_active = int(active.sum().item())
    valid = active & torch.isfinite(f0_hz) & (f0_hz > 0.0)
    vals = f0_hz[valid]
    if vals.numel() == 0:
        return {
            "active_hops": n_active,
            "voiced_hops": 0,
            "voiced_frac": 0.0,
            "median_f0_hz": None,
            "iqr_semitones": None,
        }
    med = float(torch.median(vals).item())
    q25 = float(torch.quantile(vals, 0.25).item())
    q75 = float(torch.quantile(vals, 0.75).item())
    iqr_st = 12.0 * math.log2(max(q75, 1e-6) / max(q25, 1e-6))
    return {
        "active_hops": n_active,
        "voiced_hops": int(vals.numel()),
        "voiced_frac": round(float(vals.numel()) / max(1, n_active), 4),
        "median_f0_hz": round(med, 2),
        "iqr_semitones": round(iqr_st, 3),
    }


def compare_f0_drift(
    base_f0: torch.Tensor,
    steered_f0: torch.Tensor,
    *,
    span_mask: torch.Tensor | None = None,
    drift_warn_semitones: float = 1.5,
) -> dict[str, Any]:
    """Compare a steered track against a `strength=0.0` baseline and emit advisory notes."""
    b_mask = span_mask[: base_f0.numel()] if span_mask is not None else None
    s_mask = span_mask[: steered_f0.numel()] if span_mask is not None else None
    b_sum = summarize_f0_series(base_f0, span_mask=b_mask)
    s_sum = summarize_f0_series(steered_f0, span_mask=s_mask)

    median_shift_st: float | None = None
    if b_sum["median_f0_hz"] is not None and s_sum["median_f0_hz"] is not None:
        median_shift_st = round(12.0 * math.log2(s_sum["median_f0_hz"] / b_sum["median_f0_hz"]), 3)

    paired_shift_st: float | None = None
    n_common = min(base_f0.numel(), steered_f0.numel())
    if n_common > 0:
        b_c = base_f0[:n_common]
        s_c = steered_f0[:n_common]
        m = torch.isfinite(b_c) & (b_c > 0.0) & torch.isfinite(s_c) & (s_c > 0.0)
        if span_mask is not None:
            m &= span_mask[:n_common].bool()
        if int(m.sum().item()) > 0:
            st_vec = 12.0 * torch.log2(s_c[m] / b_c[m])
            paired_shift_st = round(float(torch.median(st_vec).item()), 3)

    notes: list[str] = []
    if median_shift_st is not None and abs(median_shift_st) >= drift_warn_semitones:
        notes.append(
            f"median F0 shifted {median_shift_st:+.2f} semitones vs baseline — listen for vocal/pitch drift"
        )
    if b_sum["voiced_frac"] > 0.02 and s_sum["voiced_frac"] < 0.35 * b_sum["voiced_frac"]:
        notes.append(
            f"voiced fraction dropped ({b_sum['voiced_frac']:.3f} -> {s_sum['voiced_frac']:.3f}) — listen for vocal loss"
        )

    return {
        "baseline_median_f0_hz": b_sum["median_f0_hz"],
        "steered_median_f0_hz": s_sum["median_f0_hz"],
        "median_shift_semitones": median_shift_st,
        "paired_frame_shift_median_st": paired_shift_st,
        "iqr_base_st": b_sum["iqr_semitones"],
        "iqr_steered_st": s_sum["iqr_semitones"],
        "voiced_frac_base": b_sum["voiced_frac"],
        "voiced_frac_steered": s_sum["voiced_frac"],
        "triage_notes": notes,
    }


def _extract_f0(
    clip: Path,
    *,
    sr: int,
    threshold: float,
    min_dbfs: float,
) -> torch.Tensor:
    wav = load_mono_audio(clip, sr=sr)
    f0, _, _ = yin_f0(wav, sr=sr, threshold=threshold, min_dbfs=min_dbfs)
    return f0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", type=Path, help="single -audio.mp4 / .wav clip to profile")
    ap.add_argument("--baseline", type=Path, help="baseline (strength=0.0) -audio.mp4 / .wav clip")
    ap.add_argument("--steered", nargs="+", type=Path, help="one or more steered clips to compare against --baseline")
    ap.add_argument("--voice-spans", type=Path, help="optional voice_spans.json from bench/voice_spans.py")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR, help=f"analysis sample rate Hz (default {DEFAULT_SR})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"YIN CMNDF threshold (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--min-dbfs", type=float, default=DEFAULT_MIN_DBFS, help=f"silence gate dBFS (default {DEFAULT_MIN_DBFS})")
    ap.add_argument("--json", type=Path, help="optional output JSON path")
    args = ap.parse_args(argv)

    if args.clip:
        f0 = _extract_f0(args.clip, sr=args.sr, threshold=args.threshold, min_dbfs=args.min_dbfs)
        mask = load_voice_span_mask(args.voice_spans, f0.numel(), sr=args.sr) if args.voice_spans else None
        summary = {"clip": args.clip.name, **summarize_f0_series(f0, span_mask=mask)}
        print(json.dumps(summary, indent=2))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2) + "\n")
        return 0

    if not args.baseline or not args.steered:
        raise SystemExit("pass --clip CLIP or both --baseline BASE and --steered CLIP [CLIP ...]")

    base_f0 = _extract_f0(args.baseline, sr=args.sr, threshold=args.threshold, min_dbfs=args.min_dbfs)
    rows: list[dict[str, Any]] = []
    print("| Clip | Base F0 (Hz) | Steer F0 (Hz) | Shift (st) | Paired Shift (st) | Voiced (base->steer) | Triage Notes |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    for sp in args.steered:
        s_f0 = _extract_f0(sp, sr=args.sr, threshold=args.threshold, min_dbfs=args.min_dbfs)
        n_hops = max(base_f0.numel(), s_f0.numel())
        mask = load_voice_span_mask(args.voice_spans, n_hops, sr=args.sr) if args.voice_spans else None
        res = compare_f0_drift(
            base_f0,
            s_f0,
            span_mask=mask,
        )
        rows.append({"clip": sp.name, **res})
        bf = f"{res['baseline_median_f0_hz']:.1f}" if res["baseline_median_f0_hz"] is not None else "-"
        sf = f"{res['steered_median_f0_hz']:.1f}" if res["steered_median_f0_hz"] is not None else "-"
        ms = f"{res['median_shift_semitones']:+.2f}" if res["median_shift_semitones"] is not None else "-"
        ps = f"{res['paired_frame_shift_median_st']:+.2f}" if res["paired_frame_shift_median_st"] is not None else "-"
        vr = f"{res['voiced_frac_base']:.2f}->{res['voiced_frac_steered']:.2f}"
        notes = "; ".join(res["triage_notes"]) or "none"
        print(f"| `{sp.name}` | {bf} | {sf} | {ms} | {ps} | {vr} | {notes} |")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"baseline": args.baseline.name, "results": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
