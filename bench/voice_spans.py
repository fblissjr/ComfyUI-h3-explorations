#!/usr/bin/env python3
"""Where a voice is on a clip's track, by video frame, on this machine only.

    <python> bench/voice_spans.py separate CLIP OUT_DIR                  # the vocals stem and the rest, as wav
    <python> bench/voice_spans.py spans OUT_DIR [--fps 24]               # voice_per_frame.csv and voice_spans.json
    <python> bench/voice_spans.py cut OUT_DIR --frames 604-1050 [--band-db -12]   # a stem cut to a window's frames

**What it buys.** A masked render freezes the clip's audio, and whether a mouth should move depends on whether a
voice is there. Before this a "lip-sync" sentence went into a text for shots that had no voice on them at all,
and a render's mouth could not be set against the sound. `spans` gives each video frame a level for the voice
and for everything else and the frames the voice is on; `cut` gives a window's own stretch of the vocals stem,
to load in place of the clip's track when the model should hear the voice alone (the original audio is put
back at delivery, so what the model hears while sampling is free to differ).

**How.** `separate` splits the track with torchaudio's Hybrid Demucs from weights ALREADY in the torch hub cache.
It never fetches: with no weights it stops and says where it looked. Nothing leaves the machine. `spans` reads
the vocals stem's level per video frame; on a sung track the level has two separate modes (the voice, and the
stem's own silence), and a frame is voiced above `THRESHOLD_DB` between them. Gaps of `JOIN` frames or fewer are
closed and spans under `LEAST` frames dropped.

**Its control, printed by `spans`.** The longest unvoiced stretch with the level of the REST stem beside it: a
stretch where the band plays and the vocals stem stays silent is what shows the split works on this track. Read
it before trusting the spans. On the clip this was written on (2026-10-10) that stretch was the first 24
seconds, the band at about -22 dBFS and the vocals stem never above -47; a spectrogram of the mix, read by
eye, put the first voice within a frame of where the stem did.

**What it does not do.** Words: two models found on the machine were tried and neither is a transcriber for
singing (a voice-activity model read a whole sung stem as not speech; an English letter model gave scattered
words). Which person on screen the voice belongs to. A second voice under the lead: there is one vocals stem.
A quiet spoken line under a loud band may never reach the vocals stem; nothing here would say so.

Output goes where the caller says, which for a clip is the untracked `data/<date>_<name>_audio/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

#: torchaudio's cache file for `HDEMUCS_HIGH_MUSDB_PLUS`. Read, never fetched.
WEIGHTS = Path(os.path.expanduser("~/.cache/torch/hub/torchaudio/models/hdemucs_high_trained.pt"))
#: Seconds a chunk and the share of it cross-faded into the next. Inherited: torchaudio's own tutorial values.
SEGMENT, OVERLAP = 10.0, 0.25
#: The vocals stem's level, in dBFS, above which a video frame is voiced. Measured on one sung track,
#: 2026-10-10: voiced frames sat round -20 and the stem's silence under -70, with almost nothing between -60
#: and -40. `spans` prints the level's spread so a track that does not split this way shows it.
THRESHOLD_DB = -40.0
#: Unvoiced gaps of this many frames or fewer are closed, and voiced spans shorter than `LEAST` dropped.
#: Reasoned: a quarter of a second is a consonant or a catch of breath, not a rest.
JOIN, LEAST = 6, 4
RATE = 44100


def read_audio(path: str, rate: int = RATE) -> np.ndarray:
    """A file's audio as float32 [channels, samples] in -1..1, stereo, at `rate`, through ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(rate), "-f", "s16le", "-"],
                         capture_output=True).stdout
    return (np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0).reshape(-1, 2).T.copy()


def write_wav(path: Path, audio: np.ndarray, rate: int = RATE) -> None:
    from scipy.io import wavfile
    wavfile.write(str(path), rate, (np.clip(audio.T, -1.0, 1.0) * 32767.0).astype(np.int16))


def frame_levels(mono: np.ndarray, rate: int, fps: float) -> np.ndarray:
    """The level in dBFS of each video frame's stretch of a mono signal."""
    frames = int(round(len(mono) / rate * fps))
    out = np.zeros(frames, np.float32)
    for n in range(frames):
        seg = mono[int(round(n * rate / fps)):int(round((n + 1) * rate / fps))]
        out[n] = 20.0 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0, 1e-7))
    return out


def voiced_spans(level_db: np.ndarray, threshold: float = THRESHOLD_DB, join: int = JOIN, least: int = LEAST) -> list[list[int]]:
    """[first, last] inclusive frame runs above the threshold, gaps of `join` or fewer closed, short runs dropped."""
    runs, start = [], None
    for n, on in enumerate(list(level_db > threshold) + [False]):
        if on and start is None:
            start = n
        elif not on and start is not None:
            runs.append([start, n - 1])
            start = None
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= join:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [r for r in merged if r[1] - r[0] + 1 >= least]


def frame_cut(audio: np.ndarray, first: int, last: int, rate: int, fps: float) -> np.ndarray:
    """The samples of frames `first` to `last` inclusive: it starts on its first frame and is (last - first + 1) / fps long."""
    return audio[..., int(round(first * rate / fps)):int(round((last + 1) * rate / fps))]


def separate(a: argparse.Namespace) -> None:
    if not WEIGHTS.is_file():
        raise SystemExit(f"no separation weights at {WEIGHTS}: this tool does not download. Nothing was written")
    import torch
    import torchaudio
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = torchaudio.models.hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    torch.set_num_threads(int(a.threads))
    wav = torch.from_numpy(read_audio(a.clip))
    ref = wav.mean(0)
    mix = (wav - ref.mean()) / ref.std()
    chunk, hop, fade = int(SEGMENT * RATE), int(SEGMENT * RATE * (1 - OVERLAP)), int(SEGMENT * RATE * OVERLAP)
    total, weight, ramp = torch.zeros(4, 2, mix.shape[-1]), torch.zeros(mix.shape[-1]), torch.linspace(0, 1, fade)
    with torch.no_grad():
        for at in range(0, mix.shape[-1], hop):
            piece = mix[:, at:at + chunk]
            n = piece.shape[-1]
            est = model(torch.nn.functional.pad(piece, (0, chunk - n))[None])[0][..., :n]
            w = torch.ones(n)
            if at > 0:
                w[:min(fade, n)] = ramp[:min(fade, n)]
            if at + chunk < mix.shape[-1]:
                w[-fade:] = torch.minimum(w[-fade:], ramp.flip(0))
            total[..., at:at + n] += est * w
            weight[at:at + n] += w
            print(f"separated to {(at + n) / RATE:.1f} s", flush=True)
    stems = (total / weight.clamp(min=1e-6) * ref.std() + ref.mean()).numpy()
    write_wav(out / "vocals.wav", stems[3])
    write_wav(out / "rest.wav", stems[:3].sum(0))
    back = stems.sum(0) - wav.numpy()
    print(f"wrote {out}/vocals.wav and rest.wav; the two add back to the clip's audio to "
          f"{20 * np.log10(np.sqrt(np.mean(back ** 2)) / np.sqrt(np.mean(wav.numpy() ** 2))):.0f} dB")


def spans(a: argparse.Namespace) -> None:
    out = Path(a.out)
    vocals, rest = read_audio(str(out / "vocals.wav")).mean(0), read_audio(str(out / "rest.wav")).mean(0)
    lv, lr = frame_levels(vocals, RATE, a.fps), frame_levels(rest, RATE, a.fps)
    found = voiced_spans(lv, a.threshold)
    voiced = np.zeros(len(lv), bool)
    for lo, hi in found:
        voiced[lo:hi + 1] = True
    with open(out / "voice_per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "seconds", "vocals_stem_dbfs", "rest_stem_dbfs", "voiced"])
        for n in range(len(lv)):
            w.writerow([n, round(n / a.fps, 3), round(float(lv[n]), 1), round(float(lr[n]), 1), int(voiced[n])])
    quiet = voiced_spans(-lv, -a.threshold, 0, 1)                 # the unvoiced runs, by the same walk
    control = max(quiet, key=lambda r: r[1] - r[0], default=None)
    record = {"fps": a.fps, "frames": int(len(lv)), "threshold_dbfs": a.threshold, "join_gaps_up_to_frames": JOIN,
              "drop_spans_under_frames": LEAST, "voiced_spans_inclusive": found,
              "vocals_level_spread_dbfs": [round(float(x), 1) for x in np.percentile(lv, [5, 25, 50, 75, 95])]}
    if control:
        lo, hi = control
        record["control_longest_unvoiced"] = {"frames": [lo, hi], "vocals_stem_dbfs_max": round(float(lv[lo:hi + 1].max()), 1),
                                              "rest_stem_dbfs_median": round(float(np.median(lr[lo:hi + 1])), 1)}
    (out / "voice_spans.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(found)} voiced span(s) over {len(lv)} frames: {found}")
    print("vocals stem level at the 5th, 25th, 50th, 75th and 95th percentile, dBFS:", record["vocals_level_spread_dbfs"],
          "(two separate modes is a track this rule fits; one smear is not)")
    if control:
        c = record["control_longest_unvoiced"]
        print(f"control: frames {c['frames'][0]}-{c['frames'][1]} are unvoiced; the vocals stem never passes {c['vocals_stem_dbfs_max']} dBFS "
              f"there while the rest sits at {c['rest_stem_dbfs_median']}: if something is playing in that stretch, the split holds")


def cut(a: argparse.Namespace) -> None:
    out = Path(a.out)
    first, last = (int(x) for x in a.frames.split("-"))
    vocals = read_audio(str(out / "vocals.wav"))
    piece = frame_cut(vocals, first, last, RATE, a.fps)
    name = f"vocals_{first:04d}_{last:04d}.wav"
    if a.band_db is not None:
        piece = piece + frame_cut(read_audio(str(out / "rest.wav")), first, last, RATE, a.fps) * 10 ** (a.band_db / 20.0)
        name = f"vocals_band_{a.band_db:+.0f}db_{first:04d}_{last:04d}.wav"
    write_wav(out / name, piece)
    print(f"wrote {out / name}: {piece.shape[-1]} samples, {piece.shape[-1] / RATE:.4f} s for {last - first + 1} frames at {a.fps}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("separate")
    s.add_argument("clip")
    s.add_argument("out")
    s.add_argument("--threads", default=6)
    v = sub.add_parser("spans")
    v.add_argument("out")
    v.add_argument("--fps", type=float, default=24.0)
    v.add_argument("--threshold", type=float, default=THRESHOLD_DB)
    c = sub.add_parser("cut")
    c.add_argument("out")
    c.add_argument("--frames", required=True, metavar="FIRST-LAST", help="video frames, inclusive")
    c.add_argument("--fps", type=float, default=24.0)
    c.add_argument("--band-db", type=float, help="add the rest stem this many dB down; the voice alone when not given")
    a = p.parse_args()
    {"separate": separate, "spans": spans, "cut": cut}[a.mode](a)


if __name__ == "__main__":
    main()
