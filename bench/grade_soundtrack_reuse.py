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

**Report thirds, not just a mean, and the mean alone hid the finding.** The
owner reported on 2026-08-22 that the structured arm "looks fine in some but a
little off in later parts sometimes" -- a degradation over TIME inside one
render, which a whole-window average cannot show. Split into thirds it is
plain: structured seed 894 runs 0.652 / 0.542 / 0.156 and averages 0.426, a
mid-pack number for a render that is strong for ten seconds and gone by the
end. Every arm declines; the prompt changes how fast. A mean over a window is
the wrong summary for a quantity that drifts within it.

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
    # Two render sets, scored in one run. **They must not be compared across
    # sets** -- this measure is uncalibrated and a 15.083s window averages
    # differently from a 5.167s one, which the module docstring states and
    # which the grouping here makes hard to ignore.
    sets = {
        "124f (bench-patched substrate; VOID as prompt evidence)": [
            ("124f-structured-s892", "h3_r2v_swap_structured_00001-audio.mp4"),
            ("124f-structured-s893", "h3_r2v_swap_structured_00002-audio.mp4"),
            ("124f-concise-s892",
             "h3_ref_video_swap_concise_concise_00001-audio.mp4"),
            ("124f-directive-s892",
             "h3_ref_video_swap_directive_directive_00002-audio.mp4"),
        ],
        "362f (shipped substrate, unpatched, matched seeds)": [
            ("362f-structured-s892", "h3_r2v_swap_structured_00003-audio.mp4"),
            ("362f-concise-s892",
             "h3_ref_video_swap_concise_concise_00002-audio.mp4"),
            ("362f-structured-s893", "h3_r2v_swap_structured_00004-audio.mp4"),
            ("362f-concise-s893",
             "h3_ref_video_swap_concise_concise_00003-audio.mp4"),
            ("362f-structured-s894", "h3_r2v_swap_structured_00005-audio.mp4"),
            ("362f-concise-s894",
             "h3_ref_video_swap_concise_concise_00004-audio.mp4"),
        ],
    }
    arms = [(label, name) for group in sets.values() for label, name in group]
    groups = {label: title for title, group in sets.items()
              for label, _ in group}
    rows = []
    seen_group = None
    for label, name in arms:
        if groups[label] != seen_group:
            seen_group = groups[label]
            print(f"\n  {seen_group}")
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
        # Thirds, because the quantity drifts inside the window it is
        # averaged over. See the module docstring.
        third = d / 3.0
        by_third = []
        for k in range(3):
            st = k * third
            g3 = load(p, st, third)
            r3 = load(REF, st, third)
            c3 = load(REF, st + ctrl_start / 2.0, third)
            by_third.append(round(corr(g3, r3) - corr(g3, c3), 3))
        rows.append({"arm": label, "file": name, "seconds": round(d, 3),
                     "mel_corr_vs_own_window": round(r_match, 4),
                     "mel_corr_vs_control_window": round(r_ctrl, 4),
                     "margin_over_control": round(r_match - r_ctrl, 4),
                     "margin_by_third": by_third})
        print(f"  {label:<24} margin {r_match - r_ctrl:+.4f}   "
              f"by third {by_third[0]:+.3f} {by_third[1]:+.3f} "
              f"{by_third[2]:+.3f}")

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
                    "sets": {t: [l for l, _ in g] for t, g in sets.items()},
                    "rows": rows}, indent=2))
    print("\n  wrote bench/results/2026-08-22_soundtrack_reuse.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
