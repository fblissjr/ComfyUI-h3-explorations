#!/usr/bin/env python3
"""How closely does a `fully_copy` render reproduce its reference soundtrack?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
model, no server. ffmpeg on PATH.

**Why this exists.** The swap prompts ask for `<Audio 1>: fully_copy - reused
1:1 as the target video's complete final audio track`. H3 does not copy: it
GENERATES audio conditioned on the reference, so "did the copy work" is a
question about similarity, and on 2026-08-22 the only instrument for it was a
person saying the speech was "messed up". That is a real observation and it is
not comparable across arms. This makes it a number.

**What it measures.** Log-mel spectrograms of the render and of the matching
window of the reference, correlated frame-by-frame over time, then averaged.
Mel rather than waveform because the model regenerates rather than copies --
the output is never sample-aligned, so waveform correlation would read ~0 for a
perfect result and measure nothing.

**Read the numbers as ORDINAL, and only within one window length.** This is a
similarity proxy with no calibration: nothing establishes what value counts as
"speech intact", and a longer window averages differently from a shorter one.
Comparing a 15.083s render against 5.167s renders on this scale is exactly the
mistake it would invite. It is built to rank arms that share a window, and to
say whether a difference a listener reports is visible to any measure at all.

**The control is a mismatched pair.** Every render is also scored against a
LATER window of the same reference, which shares the speaker, the room and the
recording chain but not the content. That is the floor: a render scoring near
its own mismatched control is not reproducing anything specific, whatever the
absolute number looks like.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio

SR = 16000
REF = "/mnt/hub/ai/img/input/20260601_172336_00001-audio.mp4"


def load(path, start=0.0, duration=None):
    args = ["ffmpeg", "-v", "error"]
    if start:
        args += ["-ss", str(start)]
    args += ["-i", str(path)]
    if duration:
        args += ["-t", str(duration)]
    args += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    r = subprocess.run(args, capture_output=True, check=True)
    return torch.frombuffer(bytearray(r.stdout), dtype=torch.float32)


def logmel(w):
    m = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=256, n_mels=64)(w)
    return torch.log(m + 1e-6)


def corr(a, b):
    """Mean per-frame Pearson correlation over the overlapping frames."""
    A, B = logmel(a), logmel(b)
    n = min(A.shape[-1], B.shape[-1])
    A, B = A[..., :n], B[..., :n]
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    num = (A * B).sum(dim=0)
    den = A.norm(dim=0) * B.norm(dim=0) + 1e-12
    return float((num / den).mean())


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def main() -> int:
    out_dir = Path("/mnt/hub/ai/img/output/Video")
    arms = [
        ("len362-structured", "h3_r2v_swap_len362_00001-audio.mp4"),
        ("124f-structured-seed892", "h3_r2v_swap_structured_00001-audio.mp4"),
        ("124f-structured-seed893", "h3_r2v_swap_structured_00002-audio.mp4"),
        ("124f-concise-seed892", "h3_ref_video_swap_concise_concise_00001-audio.mp4"),
        ("124f-directive-seed892",
         "h3_ref_video_swap_directive_directive_00002-audio.mp4"),
    ]
    rows = []
    for label, name in arms:
        p = out_dir / name
        if not p.exists():
            print(f"  skip {label}: {name} not on disk")
            continue
        d = duration(p)
        gen = load(p)
        ref = load(REF, 0.0, d)
        # Control: the same length of reference from a window the render was
        # never conditioned on. Same speaker, same room, different content.
        ctrl_start = d + 1.0
        ctrl = load(REF, ctrl_start, d)
        r_match, r_ctrl = corr(gen, ref), corr(gen, ctrl)
        rows.append({"arm": label, "file": name, "seconds": round(d, 3),
                     "mel_corr_vs_own_window": round(r_match, 4),
                     "mel_corr_vs_control_window": round(r_ctrl, 4),
                     "margin_over_control": round(r_match - r_ctrl, 4)})
        print(f"  {label:<26} {d:6.3f}s   own {r_match:+.4f}   "
              f"control {r_ctrl:+.4f}   margin {r_match - r_ctrl:+.4f}")

    print("\n  Ordinal, and only within one window length. The margin over the "
          "\n  control is the load-bearing column: it is what says the render "
          "\n  reproduced THIS window rather than sounding generally like this "
          "\n  recording.")
    Path("bench/results/2026-08-22_soundtrack_reuse.json").write_text(
        json.dumps({"reference": REF, "sample_rate": SR,
                    "measure": "mean per-frame Pearson correlation of 64-bin log-mel",
                    "not_calibrated": "No value here means 'speech intact'. "
                                      "Ranks arms sharing a window length; "
                                      "cross-length comparison is invalid.",
                    "rows": rows}, indent=2))
    print("\n  wrote bench/results/2026-08-22_soundtrack_reuse.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
