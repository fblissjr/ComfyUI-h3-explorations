#!/usr/bin/env python3
"""Does one render arm's audio differ in timbre from another's, or is that the seed?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`); it needs torchaudio
and ffmpeg.

    bench/grade_arm_audio_spectrum.py \
        --arm stock='<output>/Video/marker_stock_*-audio.mp4' \
        --arm vendortokens='<output>/Video/marker_vendortokens_*-audio.mp4' \
        --out bench/results/DATE_arm_audio_spectrum.json

Written for a specific report -- one arm's speech sounding "through a tin can"
against another's -- and the shape generalises to any per-arm audio claim.

**Why a pair cannot answer this and six per arm can.** `CLAUDE.md`'s
different-sample rule: arms that differ in conditioning produce different
samples, not degraded copies of one sample. So a spectral difference between
clip A and clip B is a draw, not a measurement. What is not a draw is whether
the *between-arm* separation exceeds the *within-arm* spread across seeds. That
comparison is the entire point of this script, and every number it prints is
subordinate to it.

**The within-arm spread is the control.** If an arm's own clips scatter as
widely as the arms differ, there is nothing here and the script says so. A
version of this that reported only per-arm means would find a "difference"
every time, because two finite samples of anything differ.

**Descriptors chosen for the claim, not for completeness.** "Tin can" is
band-limited and boxy: energy pulled out of the low end and concentrated in the
low-mid, with the top rolled off. So: energy fractions in four bands summing to
1, the spectral centroid, and the 85% rolloff. The bands are a power *budget*,
not absolute levels, so a clip's loudness is already divided out of them.

**The RMS normalisation below is inert and is kept only to make that explicit.**
An earlier version of this comment claimed it stopped level differences leaking
into the band ratios. It does not: scaling the waveform scales numerator and
denominator of every ratio equally, and measured on a real clip the largest
difference it makes is 6e-08, which is float32 noise. Loudness is reported as
its own row instead, because a loudness difference is a real way two arms can
differ and silently dropping it would hide one.

Reports effect sizes; deliberately prints no p-value. At six clips an arm a
p-value would be theatre, and the honest output is the separation beside the
spread so a reader can see how much of one fits inside the other.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

# Four bands that split the way a "tin can" would: lows that a boxy sound
# loses, the low-mid it piles into, the presence range that carries
# intelligibility, and the top it rolls off.
BANDS = (("low_20_200", 20, 200), ("lowmid_200_1k", 200, 1000),
         ("presence_1k_4k", 1000, 4000), ("top_4k_16k", 4000, 16000))


# The H3 audio VAE is 32 kHz, so decoding at its native rate avoids a resample
# that would itself alter the top band this script measures.
SAMPLE_RATE = 32000


def _decode(path: Path):
    """(mono float waveform, sample rate), straight out of ffmpeg as raw f32.

    Deliberately not `torchaudio.load`: on this box it routes through
    TorchCodec, which is not installed, and a decode dependency that can vanish
    is not worth carrying for a stream ffmpeg already hands over as floats.
    """
    import torch
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, stdout=subprocess.PIPE).stdout
    if not raw:
        raise SystemExit(f"{path.name} has no decodable audio stream")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32), SAMPLE_RATE


def _descriptors(path: Path) -> dict:
    import torch
    w, sr = _decode(path)
    # Normalise loudness first: a quieter clip is not a boxier one, and without
    # this every band ratio would carry the level difference too.
    rms = float(w.pow(2).mean().sqrt())
    if rms > 0:
        w = w / rms   # inert for every ratio below; see the docstring

    spec = torch.stft(w, n_fft=2048, hop_length=512,
                      window=torch.hann_window(2048), return_complex=True).abs()
    power = spec.pow(2).mean(dim=1)              # average spectrum over time
    freqs = torch.linspace(0, sr / 2, power.shape[0])
    total = float(power.sum()) or 1.0

    out = {"rms": rms, "seconds": w.shape[0] / sr}
    for name, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(power[m].sum()) / total
    out["centroid_hz"] = float((freqs * power).sum() / power.sum())
    cum = torch.cumsum(power, 0) / power.sum()
    out["rolloff85_hz"] = float(freqs[int((cum >= 0.85).nonzero()[0])])
    return out


def _stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, var ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    metavar="LABEL=GLOB", help="repeatable; glob of that arm's clips")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        label, pattern = spec.split("=", 1)
        files = sorted(Path(p) for p in glob.glob(pattern))
        if not files:
            print(f"arm {label}: no clips matched {pattern}")
            return 2
        # Print every filename, not just the count. This script cannot see
        # which prompt produced a clip, so a glob that quietly spans two
        # experiments is invisible to it and obvious to a reader. It happened
        # on the first run of this file.
        print(f"arm {label}: {len(files)} clip(s)")
        for f in files:
            print(f"    {f.name}")
        arms[label] = [_descriptors(f) for f in files]

    if len(arms) != 2:
        print("this comparison needs exactly two arms")
        return 2
    (la, da), (lb, db) = arms.items()
    if len(da) != len(db):
        print(f"\n  arms are uneven ({len(da)} vs {len(db)}). Not fatal, but "
              f"the smaller arm's spread is the weaker estimate and the "
              f"comparison inherits that.")
    if min(len(da), len(db)) < 3:
        print("\n  fewer than three clips in an arm: the within-arm spread "
              "cannot be estimated, so nothing below is readable as a result")
    keys = [k for k in da[0] if k != "seconds"]

    print(f"\n{'descriptor':<18}{la + ' mean':>14}{'sd':>9}"
          f"{lb + ' mean':>16}{'sd':>9}{'sep/spread':>12}")
    record = {"arms": {la: da, lb: db}, "comparison": {}}
    verdict_rows = []
    for k in keys:
        ma, sa = _stats([d[k] for d in da])
        mb, sb = _stats([d[k] for d in db])
        # Between-arm separation measured in units of within-arm spread. This
        # ratio, not the means, is what the claim rests on.
        pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
        ratio = abs(ma - mb) / pooled if pooled > 0 else float("inf")
        record["comparison"][k] = {f"{la}_mean": ma, f"{la}_sd": sa,
                                   f"{lb}_mean": mb, f"{lb}_sd": sb,
                                   "separation_over_spread": ratio}
        verdict_rows.append((k, ratio))
        print(f"{k:<18}{ma:>14.4f}{sa:>9.4f}{mb:>16.4f}{sb:>9.4f}{ratio:>12.2f}")

    print("\n  sep/spread is the between-arm mean difference in units of the")
    print("  arms' own scatter across seeds. Below ~1 the arms sit inside each")
    print("  other's seed noise and the difference is not there.")
    strong = [k for k, r in verdict_rows if r >= 2.0]
    weak = [k for k, r in verdict_rows if 1.0 <= r < 2.0]
    if strong:
        print(f"\n  clears the spread by 2x or more: {', '.join(strong)}")
    elif weak:
        print(f"\n  marginal, between 1x and 2x: {', '.join(weak)}")
    else:
        print("\n  nothing clears the seed noise; on these clips the arms' "
              "audio is the same distribution")
    record["verdict"] = {"strong": strong, "marginal": weak}

    if args.out:
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
