#!/usr/bin/env python3
"""What the unscored part of a pitch-lock score is made of.

`pitch_lock_metrics.py` scores only the frames where the output vocal and the
target vocal both have a pitch (`voiced_overlap_frac`). This says what the
output is doing in the rest of the target's voiced frames, and whether it still
follows the melody there. It prints figures, not a verdict; what the unpitched
frames sound like needs listening.

- `frames`: splits the target's voiced frames into scored (a pitch in both),
  silent (output under YIN's level gate) and unpitched (output loud, no pitch
  found); gives the time shift at which the two voicings overlap most, the
  scored share in each quarter of the target's frames ranked by the target's own
  steadiness, level and pitch, and the longest scored and unpitched spans, in
  seconds, for finding them by ear; then
  loosens YIN's periodicity threshold on the output only and scores the frames
  that gain a pitch against the target and against the target rotated in time
  (the null of `pitch_lock_metrics.lag_null`, same shifts). A looser threshold
  takes harmonics for the pitch more often, so the in-tune share of gained
  frames is a floor on how much melody is there, never a ceiling.
- `agree`: two estimates of one output vocal (two separators, or a stem and the
  unseparated mix): in how many of the target's voiced frames each has a pitch,
  and how often they name the same one.
- `levels`: each stem's level against the mix, and the level of the mix with
  that stem taken out.
- `periodic`: the share of a file's active frames in which YIN finds a pitch,
  for comparing an output vocal with a real voice, and the quartiles of the
  pitch it finds, for comparing registers.
- `ltas`: the cover gate's timbre figure (cosine of two long-term mel spectra)
  for each file against a reference, with the reference's first half against
  its second beside it. The figure moves with what is in a clip (a bass under
  the voice, the register sung) as well as with whose voice it is, so read it
  only beside the same figure for the scaffold and the target vocal.

- `rests`: what a clip holds while the target vocal rests: its level there
  against its level while the vocal sings, how that sound is spread over
  `BANDS_HZ`, and how much of it each stem carries. A vocal stem that carries
  the rests' sound is not only a voice. It also gives the clip's band split
  over the sung frames, so a vocal stem's brightness can be set beside its
  target's.
- `under`: a clean voice with a bed mixed under it at each of `BED_LEVELS_DB`,
  and the share of the voice's pitched frames that keep a pitch. It says how
  loud an accompaniment must be before YIN loses the voice.

`frames --highpass HZ` filters the output first, to ask whether the band under
HZ (a bass left in the stem) is what hides the pitch.

    <comfy-venv-python> bench/pitch_lock_coverage.py frames [--highpass HZ] OUT_VOC TARGET_VOC [OUT_VOC TARGET_VOC ...]
    <comfy-venv-python> bench/pitch_lock_coverage.py agree STEM_A STEM_B TARGET_VOC
    <comfy-venv-python> bench/pitch_lock_coverage.py levels MIX STEM [STEM ...]
    <comfy-venv-python> bench/pitch_lock_coverage.py periodic FILE [FILE ...]
    <comfy-venv-python> bench/pitch_lock_coverage.py ltas REFERENCE FILE [FILE ...]
    <comfy-venv-python> bench/pitch_lock_coverage.py rests TARGET_VOC CLIP [STEM ...]
    <comfy-venv-python> bench/pitch_lock_coverage.py under VOICE BED
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pitch_lock_metrics as plm  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

RATE = plm.RATE
HOP_S = plm.HOP_S
IN_TUNE_CENTS = 50.0
# The first entry is the threshold every score in the records was made at.
THRESHOLDS = (yin_probe.DEFAULT_THRESHOLD, 0.25, 0.35, 0.50)
LADDER_STEP = 0.05
OVERLAP_SEARCH_S = 1.0
SPANS_LISTED = 4
SPAN_BRIDGE_FRAMES = 5
# A frame counts as active when it is within this of the clip's loud frames.
ACTIVE_BELOW_LOUD_DB = 20.0
LOUD_PERCENTILE = 95.0
LEVEL_RATE = 44100
# The mel settings of the cover gate's `ltas_cos_kurt_unplugged`, so the figures are comparable.
LTAS_N_FFT, LTAS_HOP, LTAS_MELS = 1024, 256, 64
# A frame is a rest when the target vocal is this far under its loud frames.
REST_BELOW_LOUD_DB = 30.0
BANDS_HZ = ((0, 150), (150, 1000), (1000, 4000), (4000, RATE // 2))
BAND_N_FFT = 2048
BED_LEVELS_DB = (-20.0, -15.0, -10.0, -5.0, 0.0)
HIGHPASS_PASSES = 4  # second-order sections in a row


def voiced(f0: np.ndarray) -> np.ndarray:
    return np.isfinite(f0) & (f0 > 0)


def _in_tune(fo: np.ndarray, ft: np.ndarray, frames: np.ndarray) -> float | None:
    """Share of `frames` (where both have a pitch) within `IN_TUNE_CENTS`, octaves folded, no transposition."""
    use = frames & voiced(fo) & voiced(ft)
    if int(use.sum()) < plm.MIN_SHARED_FRAMES:
        return None
    cents = 1200.0 * np.log2(fo[use] / ft[use])
    return float(np.mean(np.abs(((cents + 600.0) % 1200.0) - 600.0) <= IN_TUNE_CENTS))


def split_frames(fo: np.ndarray, rms_dbfs: np.ndarray, ft: np.ndarray) -> dict[str, int]:
    """Count the target's voiced frames by what the output is doing in them."""
    target, out = voiced(ft), voiced(fo)
    missed = target & ~out
    silent = missed & (rms_dbfs < yin_probe.DEFAULT_MIN_DBFS)
    return {
        "target_voiced": int(target.sum()),
        "scored": int((target & out).sum()),
        "silent": int(silent.sum()),
        "unpitched": int((missed & ~silent).sum()),
        "output_pitch_outside_target": int((out & ~target).sum()),
    }


def best_overlap(fo: np.ndarray, ft: np.ndarray, *, hop_s: float = HOP_S) -> dict[str, Any]:
    """Frames voiced in both at no shift, and at the shift within `OVERLAP_SEARCH_S` that has the most."""
    target, out = voiced(ft), voiced(fo)
    reach = round(OVERLAP_SEARCH_S / hop_s)
    counts = {lag: int((np.roll(target, lag) & out).sum()) for lag in range(-reach, reach + 1)}
    lag = max(counts, key=lambda k: (counts[k], -abs(k)))
    return {"at_no_shift": counts[0], "best": counts[lag], "best_shift_ms": round(lag * hop_s * 1000)}


def scored_by_quartile(values: np.ndarray, target: np.ndarray, out: np.ndarray) -> dict[str, list[float]]:
    """Share of the target's voiced frames that are scored, in each quarter of them ranked by `values`.

    Quarters are cut by rank, lowest first, so each holds the same number of frames.
    """
    frames = np.flatnonzero(target)
    order = frames[np.argsort(values[frames], kind="stable")]
    quarters = np.array_split(order, 4)
    return {
        "upper_edges": [round(float(values[q[-1]]), 3) for q in quarters if len(q)],
        "scored_frac": [round(float(out[q].mean()), 3) for q in quarters if len(q)],
    }


def longest_spans(frames: np.ndarray, *, hop_s: float = HOP_S) -> list[list[float]]:
    """The `SPANS_LISTED` longest runs of `frames` in seconds, in time order, for finding them by ear.

    Runs separated by `SPAN_BRIDGE_FRAMES` or fewer are one span.
    """
    runs: list[list[int]] = []
    for i in np.flatnonzero(frames):
        if runs and i - runs[-1][1] <= SPAN_BRIDGE_FRAMES + 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    longest = sorted(runs, key=lambda r: r[0] - r[1])[:SPANS_LISTED]
    return [[round(a * hop_s, 2), round((b + 1) * hop_s, 2)] for a, b in sorted(longest)]


def gained_frames_null(fo: np.ndarray, ft: np.ndarray, gained: np.ndarray, *, hop_s: float = HOP_S) -> dict[str, Any] | None:
    """In-tune share of the gained frames, aligned and against the target rotated in time."""
    aligned = _in_tune(fo, ft, gained)
    n = len(fo)
    step, edge = max(1, round(plm.LAG_STEP_S / hop_s)), round(plm.LAG_EDGE_S / hop_s)
    shifted = [v for v in (_in_tune(fo, np.roll(ft, lag), gained) for lag in range(edge, n - edge, step)) if v is not None]
    if aligned is None or len(shifted) < plm.MIN_LAGS:
        return None
    values = np.array(shifted)
    return {
        "in_tune_frac_50c": round(aligned, 3),
        "null_median": round(float(np.median(values)), 3),
        "null_best": round(float(values.max()), 3),
        "beats_frac": round(float(np.mean(values < aligned)), 3),
        "n_lags": len(shifted),
    }


def threshold_sweep(out_voc_24k: torch.Tensor, fo: np.ndarray, ft: np.ndarray) -> list[dict[str, Any]]:
    """Loosen YIN's periodicity threshold on the output only; score the frames that gain a pitch.

    The threshold climbs in `LADDER_STEP` steps and a frame keeps the pitch it had
    at the first threshold that gave it one. YIN takes the first dip under its
    threshold, so asking a noisy frame at a loose threshold returns an early
    shallow dip; asking at the tightest threshold that passes returns its deepest.
    """
    target, strict = voiced(ft), voiced(fo)
    merged = fo.copy()
    rows: list[dict[str, Any]] = []
    for threshold in np.arange(THRESHOLDS[0], THRESHOLDS[-1] + LADDER_STEP / 2, LADDER_STEP):
        threshold = round(float(threshold), 2)
        if threshold > THRESHOLDS[0]:
            loose = yin_probe.yin_f0(out_voc_24k, sr=RATE, threshold=threshold)[0].numpy()[: len(fo)]
            take = ~voiced(merged) & voiced(loose)
            merged[take] = loose[take]
        if threshold not in THRESHOLDS:
            continue
        gained = voiced(merged) & ~strict & target
        rows.append(
            {
                "threshold": threshold,
                "voiced_overlap_frac": round(float((voiced(merged) & target).sum()) / max(1, int(target.sum())), 3),
                "gained_frames": int(gained.sum()),
                "gained": gained_frames_null(merged, ft, gained) if int(gained.sum()) else None,
            }
        )
    return rows


def frames_report(out_voc_24k: torch.Tensor, tgt_voc_24k: torch.Tensor) -> dict[str, Any]:
    fo, cmndf, rms = (x.numpy() for x in yin_probe.yin_f0(out_voc_24k, sr=RATE))
    ft, tgt_cmndf, tgt_rms = (x.numpy() for x in yin_probe.yin_f0(tgt_voc_24k, sr=RATE))
    n = min(len(fo), len(ft))
    fo, cmndf, rms, ft, tgt_cmndf, tgt_rms = fo[:n], cmndf[:n], rms[:n], ft[:n], tgt_cmndf[:n], tgt_rms[:n]
    target, out = voiced(ft), voiced(fo)
    scored, missed = target & out, target & ~out

    def middle(values: np.ndarray, frames: np.ndarray) -> list[float] | None:
        return [round(float(v), 2) for v in np.percentile(values[frames], (25, 50, 75))] if frames.any() else None

    return {
        **split_frames(fo, rms, ft),
        "overlap": best_overlap(fo, ft),
        "output_dbfs_q25_50_75": {"scored": middle(rms, scored), "missed": middle(rms, missed)},
        "output_min_cmndf_q25_50_75": {"scored": middle(cmndf, scored), "missed": middle(cmndf, missed)},
        # Which of the target's frames keep a pitch: by how unsteady, how loud and how high the target is there.
        "scored_by_target_quartile": {
            "min_cmndf": scored_by_quartile(tgt_cmndf, target, out),
            "dbfs": scored_by_quartile(tgt_rms, target, out),
            "pitch_hz": scored_by_quartile(np.nan_to_num(ft), target, out),
        },
        "longest_spans_s": {
            "scored": longest_spans(scored),
            "unpitched": longest_spans(missed & (rms >= yin_probe.DEFAULT_MIN_DBFS)),
        },
        "threshold_sweep": threshold_sweep(out_voc_24k, fo, ft),
    }


def agreement(fa: np.ndarray, fb: np.ndarray, ft: np.ndarray) -> dict[str, Any]:
    """Within the target's voiced frames: which of two estimates of one output has a pitch, and whether they match."""
    n = min(len(fa), len(fb), len(ft))
    fa, fb, target = fa[:n], fb[:n], voiced(ft[:n])
    a, b = voiced(fa), voiced(fb)
    same = _in_tune(fa, fb, target)
    return {
        "target_voiced": int(target.sum()),
        "both": int((a & b & target).sum()),
        "a_only": int((a & ~b & target).sum()),
        "b_only": int((~a & b & target).sum()),
        "neither": int((~a & ~b & target).sum()),
        "same_pitch_frac_50c": None if same is None else round(same, 3),
    }


def periodic_share(f0: np.ndarray, cmndf: np.ndarray, rms_dbfs: np.ndarray) -> dict[str, Any]:
    """Share of the active frames (near the clip's loud level) in which YIN found a pitch."""
    active = rms_dbfs > np.percentile(rms_dbfs, LOUD_PERCENTILE) - ACTIVE_BELOW_LOUD_DB
    return {
        "active_frames": int(active.sum()),
        "pitched_frac_of_active": round(float((voiced(f0) & active).sum()) / max(1, int(active.sum())), 3),
        "min_cmndf_q25_50_75": [round(float(v), 2) for v in np.percentile(cmndf[active], (25, 50, 75))],
        # The register sung, over every pitched frame.
        "pitch_hz_q25_50_75": [round(float(v), 1) for v in np.percentile(f0[voiced(f0)], (25, 50, 75))] if voiced(f0).any() else None,
    }


def ltas_cosine(a_24k: torch.Tensor, b_24k: torch.Tensor) -> float:
    """Cosine of two clips' long-term log-mel spectra, each with its mean taken off."""
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, n_fft=LTAS_N_FFT, hop_length=LTAS_HOP, n_mels=LTAS_MELS)
    va, vb = (torch.log(mel(x.float()) + 1e-6).mean(dim=-1) for x in (a_24k, b_24k))
    va, vb = va - va.mean(), vb - vb.mean()
    return float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-12))


def highpass(wav_24k: torch.Tensor, hz: float) -> torch.Tensor:
    for _ in range(HIGHPASS_PASSES):
        wav_24k = torchaudio.functional.highpass_biquad(wav_24k, RATE, hz)
    return wav_24k


def _frame_power(wav_24k: torch.Tensor) -> np.ndarray:
    """Power per frequency bin and frame, `[bins, frames]`, on YIN's hop."""
    window = torch.hann_window(BAND_N_FFT)
    spec = torch.stft(wav_24k.float(), n_fft=BAND_N_FFT, hop_length=yin_probe.DEFAULT_HOP_LEN, window=window, center=False, return_complex=True)
    return (spec.abs() ** 2).numpy()


def _level_db(power: np.ndarray, frames: np.ndarray) -> float:
    return float(10.0 * np.log10(power[:, frames].sum() / max(1, int(frames.sum())) + 1e-20))


def _band_split(power: np.ndarray, frames: np.ndarray) -> dict[str, float]:
    freqs = np.fft.rfftfreq(BAND_N_FFT, 1.0 / RATE)
    total = power[:, frames].sum() + 1e-20
    return {
        f"{lo}-{hi}": round(float(10.0 * np.log10(power[(freqs >= lo) & (freqs < hi)][:, frames].sum() / total + 1e-12)), 1)
        for lo, hi in BANDS_HZ
    }


def rest_report(tgt_voc_24k: torch.Tensor, clip_24k: torch.Tensor, stems: dict[str, torch.Tensor]) -> dict[str, Any]:
    """What `clip` holds in the frames where the target vocal rests, and which stems carry it."""
    target, clip = _frame_power(tgt_voc_24k), _frame_power(clip_24k)
    n = min(target.shape[1], clip.shape[1])
    level = 10.0 * np.log10(target[:, :n].sum(axis=0) + 1e-20)
    loud = np.percentile(level, LOUD_PERCENTILE)
    rest, sung = level < loud - REST_BELOW_LOUD_DB, level > loud - ACTIVE_BELOW_LOUD_DB
    out: dict[str, Any] = {"rest_frames": int(rest.sum()), "sung_frames": int(sung.sum())}
    if not rest.any() or not sung.any():
        return out
    out["rest_re_sung_db"] = round(_level_db(clip[:, :n], rest) - _level_db(clip[:, :n], sung), 1)
    out["bands_in_rests_db"] = _band_split(clip[:, :n], rest)
    out["bands_all_frames_db"] = _band_split(clip[:, :n], np.ones(n, dtype=bool))
    # For a vocal stem given as the clip: its spectral balance while the target sings, to set beside the target's own.
    out["bands_sung_frames_db"] = _band_split(clip[:, :n], sung)
    out["stems_re_clip_in_rests_db"] = {}
    for name, stem in stems.items():
        power = _frame_power(stem)
        k = min(n, power.shape[1])
        out["stems_re_clip_in_rests_db"][name] = round(_level_db(power[:, :k], rest[:k]) - _level_db(clip[:, :k], rest[:k]), 1)
    return out


def kept_under_bed(voice_24k: torch.Tensor, bed_24k: torch.Tensor) -> dict[str, Any]:
    """Share of the voice's pitched frames that keep a pitch with `bed` mixed under it, by bed level re the voice."""
    n = min(voice_24k.numel(), bed_24k.numel())
    voice, bed = voice_24k[:n].float(), bed_24k[:n].float()
    clean = voiced(yin_probe.yin_f0(voice, sr=RATE)[0].numpy())
    scale = float(torch.sqrt(torch.mean(voice**2)) / (torch.sqrt(torch.mean(bed**2)) + 1e-12))
    kept = {}
    for level_db in BED_LEVELS_DB:
        mixed = voice + bed * scale * 10.0 ** (level_db / 20.0)
        still = voiced(yin_probe.yin_f0(mixed, sr=RATE)[0].numpy())
        kept[f"{level_db:+.0f}"] = round(float((still & clean).sum()) / max(1, int(clean.sum())), 3)
    bed_alone = periodic_share(*(x.numpy() for x in yin_probe.yin_f0(bed, sr=RATE)))
    return {"voice_pitched_frames": int(clean.sum()), "bed_pitched_frac_of_active": bed_alone["pitched_frac_of_active"], "kept_frac_by_bed_db": kept}


def _db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(1e-12, float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))))))


def stem_levels(mix: np.ndarray, stems: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Each stem's level against the mix, and the level of the mix with that stem subtracted."""
    out = {}
    for name, stem in stems.items():
        n = min(mix.shape[-1], stem.shape[-1])
        ref = _db(mix[..., :n])
        out[name] = {
            "stem_re_mix_db": round(_db(stem[..., :n]) - ref, 2),
            "mix_minus_stem_re_mix_db": round(_db(mix[..., :n] - stem[..., :n]) - ref, 2),
        }
    return out


def _load_stereo(path: Path) -> np.ndarray:
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(LEVEL_RATE), "-f", "f32le", "-"]
    )
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T


def _yin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0, cmndf, rms = yin_probe.yin_f0(yin_probe.load_mono_audio(path, sr=RATE), sr=RATE)
    return f0.numpy(), cmndf.numpy(), rms.numpy()


def _label(paths: list[Path]) -> list[str]:
    # Stems from one separator share a file name, so fall back to folder and name.
    unique = len({p.name for p in paths}) == len(paths)
    return [p.name if unique else f"{p.parent.name}/{p.name}" for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p_frames = sub.add_parser("frames", help="split the target's voiced frames and sweep the YIN threshold")
    p_frames.add_argument("pairs", nargs="+", type=Path, metavar="VOC", help="OUT_VOC TARGET_VOC, repeated")
    p_frames.add_argument("--highpass", type=float, default=None, metavar="HZ", help="filter each OUT_VOC above HZ first")
    p_agree = sub.add_parser("agree", help="compare two estimates of one output vocal")
    p_agree.add_argument("stem_a", type=Path)
    p_agree.add_argument("stem_b", type=Path)
    p_agree.add_argument("target", type=Path)
    p_levels = sub.add_parser("levels", help="stem levels against the mix")
    p_levels.add_argument("mix", type=Path)
    p_levels.add_argument("stems", nargs="+", type=Path)
    p_periodic = sub.add_parser("periodic", help="share of active frames with a pitch")
    p_periodic.add_argument("files", nargs="+", type=Path)
    p_ltas = sub.add_parser("ltas", help="long-term spectrum cosine against a reference")
    p_ltas.add_argument("reference", type=Path)
    p_ltas.add_argument("files", nargs="+", type=Path)
    p_rests = sub.add_parser("rests", help="what a clip holds while the target vocal rests")
    p_rests.add_argument("target", type=Path)
    p_rests.add_argument("clip", type=Path)
    p_rests.add_argument("stems", nargs="*", type=Path)
    p_under = sub.add_parser("under", help="pitched frames a voice keeps with a bed under it")
    p_under.add_argument("voice", type=Path)
    p_under.add_argument("bed", type=Path)
    args = parser.parse_args(argv)

    results: dict[str, Any]
    if args.mode == "frames":
        if len(args.pairs) % 2:
            parser.error("give files in pairs: OUT_VOC TARGET_VOC")
        outputs = args.pairs[0::2]

        def load_out(path: Path) -> torch.Tensor:
            wav = yin_probe.load_mono_audio(path, sr=RATE)
            return wav if args.highpass is None else highpass(wav, args.highpass)

        results = {
            name: {
                "target": target.name,
                "highpass_hz": args.highpass,
                **frames_report(load_out(out), yin_probe.load_mono_audio(target, sr=RATE)),
            }
            for name, out, target in zip(_label(outputs), outputs, args.pairs[1::2])
        }
    elif args.mode == "agree":
        results = {
            "a": args.stem_a.name if args.stem_a.name != args.stem_b.name else str(args.stem_a.parent.name),
            "b": args.stem_b.name if args.stem_a.name != args.stem_b.name else str(args.stem_b.parent.name),
            **agreement(_yin(args.stem_a)[0], _yin(args.stem_b)[0], _yin(args.target)[0]),
        }
    elif args.mode == "levels":
        results = stem_levels(_load_stereo(args.mix), dict(zip(_label(args.stems), map(_load_stereo, args.stems))))
    elif args.mode == "periodic":
        results = {name: periodic_share(*_yin(path)) for name, path in zip(_label(args.files), args.files)}
    elif args.mode == "rests":
        load = lambda path: yin_probe.load_mono_audio(path, sr=RATE)  # noqa: E731
        results = rest_report(load(args.target), load(args.clip), dict(zip(_label(args.stems), map(load, args.stems))))
    elif args.mode == "under":
        results = kept_under_bed(yin_probe.load_mono_audio(args.voice, sr=RATE), yin_probe.load_mono_audio(args.bed, sr=RATE))
    else:
        ref = yin_probe.load_mono_audio(args.reference, sr=RATE)
        half = ref.numel() // 2
        results = {
            "reference": args.reference.name,
            "reference_first_half_against_second": round(ltas_cosine(ref[:half], ref[half:]), 3),
            "ltas_cos": {
                name: round(ltas_cosine(yin_probe.load_mono_audio(path, sr=RATE), ref), 3)
                for name, path in zip(_label(args.files), args.files)
            },
        }
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
