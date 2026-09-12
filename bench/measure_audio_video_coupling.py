#!/usr/bin/env python3
"""Does the picture move with the audio? Two instruments with a null, per clip.

`docs/h3_audio_freeze.md` section 4 step 2 asks whether video generated
against a frozen track follows it. The owner's eye is the verdict; this is
the instrument that sits beside it, so a verdict has a number with a null
next to it rather than standing alone.

Two measures, both at the video's frame rate, both against the audio the
clip actually carries (or `--audio` for a track to test against instead):

- **beat**: whole-frame motion energy (mean absolute difference between
  consecutive greyscale frames, downscaled) against the audio's onset
  envelope (half-wave rectified spectral flux, one value per video frame).
  Reported as the Pearson correlation at lag zero and the best over a small
  lag window, plus a null: the same correlation against the envelope
  circularly shifted by random offsets, summarised as a z-score. A dancer
  who hits on the beats correlates; a dancer who moves regardless does not,
  whatever the track.
- **tempo**: does the motion have a periodic component at the track's
  tempo. The tempo is read off the audio's own onset envelope (the spectral
  peak between 60 and 200 bpm); the motion energy's power at that frequency
  and its first harmonic is divided by the median power across 0.5 to 8 Hz.
  A body hitting on the beats shows a ratio well above one; motion that
  ignores the track sits near one. The null is the same ratio at tempos
  drawn away from the track's, so the report says whether the track's tempo
  is special for that clip. Sample-wise correlation (the beat mode) is a
  harder test than the eye applies, because a limb that lands on the beat
  with a consistent lag still correlates poorly at lag zero.
- **mouth**: motion energy inside the lower part of the detected face
  (OpenCV's frontal Haar cascade, median box over the clip) against the
  audio's RMS envelope, and the ratio of lower-face motion while the audio
  is loud to while it is quiet. A mouth that moves with the words shows a
  ratio well above one and a positive correlation; a mouth that talks
  through the silence does not.

**What this is not.** A metric that ranks two arms is a claim about the
metric (`docs/checks.md`). Frame-difference energy is camera motion, cuts
and fabric as much as limbs; the Haar box is coarse and the lower-face
region catches chin and hands. The z-score says whether the coupling beats
chance for that clip; it does not say the coupling looks right. Read it
beside the owner's verdict, never instead of it.

## Running it

    <comfy venv python> bench/measure_audio_video_coupling.py \\
        --clip dancer_frozen=<path.mp4> --clip dancer_free=<path.mp4> \\
        --mode beat --out bench/results/<date>_..._coupling.json

`--audio <path>` tests every clip against that track instead of its own
soundtrack (the free twin against the frozen track, for instance).
Needs OpenCV and PyAV, both in the ComfyUI venv; no model, no CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 waveform and rate, decoded with PyAV as core's LoadAudio does."""
    import av
    with av.open(path) as af:
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n = stream.channels
        frames = []
        for fr in af.decode(streams=stream.index):
            buf = fr.to_ndarray()
            if buf.shape[0] != n:
                buf = buf.reshape(-1, n).T
            frames.append(buf.astype(np.float32))
        wav = np.concatenate(frames, axis=1)
    if wav.dtype.kind == "i":
        wav = wav / np.iinfo(wav.dtype).max
    return wav.mean(axis=0), int(sr)


def video_frames(path: str, width: int = 160):
    """Greyscale frames downscaled to `width`, and the frame rate."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return np.stack(frames), float(fps)


def face_box(path: str):
    """Median frontal-face box over the clip at full resolution, or None.

    Returns None when this OpenCV build has no cascade detector (the venv's
    build did not, 2026-09-12); the caller then falls back to a fixed
    centre-lower region and says so in the record.
    """
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    boxes = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 6 == 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = cascade.detectMultiScale(grey, 1.1, 5, minSize=(40, 40))
            if len(found):
                x, y, w, h = max(found, key=lambda b: b[2] * b[3])
                boxes.append((x, y, w, h))
        i += 1
    cap.release()
    if not boxes:
        return None
    b = np.median(np.array(boxes), axis=0).astype(int)
    return tuple(int(v) for v in b), len(boxes)


def onset_envelope(wav: np.ndarray, sr: int, n_frames: int, fps: float) -> np.ndarray:
    """Half-wave rectified spectral flux, one value per video frame."""
    hop = sr / fps
    win = int(2 ** np.ceil(np.log2(hop * 2)))
    window = np.hanning(win)
    prev = None
    out = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        c = int(round(i * hop))
        seg = wav[max(0, c - win // 2): c + win // 2]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        mag = np.abs(np.fft.rfft(seg * window))
        if prev is not None:
            out[i] = np.maximum(mag - prev, 0).sum()
        prev = mag
    return out


def rms_envelope(wav: np.ndarray, sr: int, n_frames: int, fps: float) -> np.ndarray:
    hop = sr / fps
    out = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a, b = int(round(i * hop)), int(round((i + 1) * hop))
        seg = wav[a:b]
        out[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
    return out


def motion_energy(frames: np.ndarray, region=None) -> np.ndarray:
    if region is not None:
        y0, y1, x0, x1 = region
        frames = frames[:, y0:y1, x0:x1]
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], d])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def coupling(motion: np.ndarray, env: np.ndarray, max_lag: int, seed: int = 0) -> dict:
    n = min(len(motion), len(env))
    m, e = motion[:n], env[:n]
    at0 = corr(m, e)
    best, best_lag = at0, 0
    for lag in range(-max_lag, max_lag + 1):
        c = corr(m[max(0, lag):n + min(0, lag)], e[max(0, -lag):n - max(0, lag)])
        if c > best:
            best, best_lag = c, lag
    rng = np.random.default_rng(seed)
    null = np.array([corr(m, np.roll(e, int(s))) for s in rng.integers(max_lag + 12, n - max_lag - 12, size=400)])
    z = (at0 - null.mean()) / (null.std() + 1e-12)
    return {"corr_lag0": round(at0, 4), "corr_best": round(best, 4), "best_lag_frames": best_lag,
            "null_mean": round(float(null.mean()), 4), "null_std": round(float(null.std()), 4),
            "z_lag0": round(float(z), 2), "frames": int(n)}


def _power_at(sig: np.ndarray, fps: float, hz: float) -> float:
    x = sig - sig.mean()
    x = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
    # sum over a narrow band around hz, one bin either side
    idx = np.argmin(np.abs(freqs - hz))
    lo, hi = max(0, idx - 1), min(len(spec) - 1, idx + 1)
    return float(spec[lo:hi + 1].sum())


def tempo_locking(motion: np.ndarray, env: np.ndarray, fps: float, seed: int = 0) -> dict:
    n = min(len(motion), len(env))
    m, e = motion[:n], env[:n]
    x = e - e.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= 1.0) & (freqs <= 200 / 60)
    tempo_hz = float(freqs[band][np.argmax(spec[band])])
    mspec = np.abs(np.fft.rfft((m - m.mean()) * np.hanning(n))) ** 2
    ref = (freqs >= 0.5) & (freqs <= 8.0)
    floor = float(np.median(mspec[ref])) + 1e-12
    def ratio(hz):
        return (_power_at(m, fps, hz) + _power_at(m, fps, 2 * hz)) / (2 * 3 * floor)
    at_tempo = ratio(tempo_hz)
    rng = np.random.default_rng(seed)
    others = []
    while len(others) < 300:
        hz = float(rng.uniform(1.0, 200 / 60))
        if abs(hz - tempo_hz) > 0.15 and abs(hz - tempo_hz / 2) > 0.15 and abs(hz - 2 * tempo_hz) > 0.15:
            others.append(ratio(hz))
    others = np.array(others)
    return {"tempo_bpm": round(tempo_hz * 60, 1), "power_ratio_at_tempo": round(at_tempo, 3),
            "null_median": round(float(np.median(others)), 3),
            "null_p95": round(float(np.percentile(others, 95)), 3),
            "beats_chance": bool(at_tempo > np.percentile(others, 95)), "frames": int(n)}


def measure(label: str, clip: str, mode: str, audio: str | None, max_lag: int) -> dict:
    frames, fps = video_frames(clip)
    wav, sr = load_audio(audio or clip)
    n = len(frames)
    rec = {"label": label, "clip": Path(clip).name, "audio": Path(audio).name if audio else "the clip's own",
           "fps": fps, "frames": n, "mode": mode}
    if mode == "beat":
        env = onset_envelope(wav, sr, n, fps)
        rec["beat"] = coupling(motion_energy(frames), env, max_lag)
    elif mode == "tempo":
        env = onset_envelope(wav, sr, n, fps)
        rec["tempo"] = tempo_locking(motion_energy(frames), env, fps)
    else:
        fb = face_box(clip)
        H, W = frames.shape[1], frames.shape[2]
        if fb is None:
            # No detector: the voice prompts frame one face at medium close-up,
            # centred, so the mouth sits in the centre-lower part of the frame.
            region = (int(0.50 * H), int(0.78 * H), int(0.35 * W), int(0.65 * W))
            box_note = {"face_box_full_res": None, "face_hits": 0,
                        "region_source": "fixed centre-lower region; no cascade detector in this OpenCV build"}
        else:
            (x, y, w, h), hits = fb
            # the detector ran at full resolution; the frames are downscaled
            import cv2
            cap = cv2.VideoCapture(clip)
            full_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cap.release()
            s = frames.shape[2] / full_w
            region = (int((y + 0.62 * h) * s), int((y + h) * s), int(x * s), int((x + w) * s))
            box_note = {"face_box_full_res": [int(x), int(y), int(w), int(h)], "face_hits": hits,
                        "region_source": "lower 38% of the median Haar face box"}
        motion = motion_energy(frames, region)
        env = rms_envelope(wav, sr, n, fps)
        loud = env > 0.2 * env.max()
        quiet = ~loud
        rec["mouth"] = {
            **box_note,
            "lower_face_region_small": list(region),
            "loud_frames": int(loud.sum()), "quiet_frames": int(quiet.sum()),
            "motion_loud_mean": round(float(motion[loud].mean()), 4) if loud.any() else None,
            "motion_quiet_mean": round(float(motion[quiet].mean()), 4) if quiet.any() else None,
            "loud_over_quiet": (round(float(motion[loud].mean() / (motion[quiet].mean() + 1e-9)), 3)
                                if loud.any() and quiet.any() else None),
            **coupling(motion, env, max_lag),
        }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", required=True, metavar="LABEL=PATH.mp4")
    ap.add_argument("--mode", choices=("beat", "tempo", "mouth"), required=True)
    ap.add_argument("--audio", default=None, help="test every clip against this track instead of its own")
    ap.add_argument("--max-lag", type=int, default=6, help="frames either side for the best-lag search")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for spec in args.clip:
        label, _, path = spec.partition("=")
        rows.append(measure(label, path, args.mode, args.audio, args.max_lag))
        r = rows[-1].get("beat") or rows[-1].get("tempo") or rows[-1].get("mouth")
        if "power_ratio_at_tempo" in r:
            print(f"  {label:<20} tempo {r['tempo_bpm']} bpm  ratio {r['power_ratio_at_tempo']}  "
                  f"null median {r['null_median']} p95 {r['null_p95']}  beats chance: {r['beats_chance']}")
        else:
            print(f"  {label:<20} " + (f"corr0 {r['corr_lag0']:+.3f}  best {r['corr_best']:+.3f}@{r['best_lag_frames']:+d}  z {r['z_lag0']:+.1f}"
                                        + (f"  loud/quiet {r['loud_over_quiet']}" if "loud_over_quiet" in r else "")
                                        if "corr_lag0" in r else str(r)))
    record = {"date": date.today().isoformat(), "mode": args.mode,
              "what": ("whole-frame motion energy against the audio onset envelope, per video frame, with a circular-shift null"
                       if args.mode == "beat" else
                       "power of the whole-frame motion energy at the audio's tempo (and first harmonic) over the median power across 0.5-8 Hz, against the same ratio at tempos drawn away from the track's"
                       if args.mode == "tempo" else
                       "lower-face motion energy against the audio RMS envelope, per video frame, with a circular-shift null; loud/quiet is the mean motion ratio between frames where the audio is above a fifth of its peak RMS and the rest"),
              "against": args.audio or "each clip's own soundtrack",
              "rows": rows,
              "caveats": ["an instrument beside the owner's verdict, not a ranking (docs/checks.md)",
                          "motion energy is camera, cuts and fabric as much as the subject; the Haar face box is coarse",
                          "z is against a circular-shift null of the same envelope, so it says the coupling beats chance for this clip and nothing about how it looks"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  record: {out.relative_to(REPO) if out.is_absolute() and REPO in out.parents else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
