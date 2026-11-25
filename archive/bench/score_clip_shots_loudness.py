#!/usr/bin/env python3
"""Two things a finished clip can be checked for without anyone watching it:
how many cuts it holds against the shots its prompt scripts, and how loud it
is against a reference clip of the same scene.

    python bench/score_clip_shots_loudness.py --shots 3 --reference DENSE-audio.mp4 \\
        --out data/results/<date>_<what>.json CLIP-audio.mp4 [...]

Written 2026-10-03 for the output check (`bench/results/2026-10-03_sol_output_check.md`).
On the twelve clips the owner judged on 2026-10-02, every sparse-attention
render of the three-shot market scene held a cut the script does not have and
the dense render did not, and the owner's audio verdicts on the dialogue scene
followed loudness against the dense clip. Both were found after the verdicts,
on two scenes; the record says what that does and does not support.

Cuts are `bench/measure_clip_delta.py::cut_times` (ffmpeg's scene score above
its `CUT_SCORE`), with every frame's score kept so a count can be re-read at
another threshold. Loudness is `bench/measure_clip_loudness.py::ebur128`.

It reports. `shots_match` is the one comparison it makes; whether a loudness
difference matters is not decided here. It sees an added or missing cut and a
level change, and nothing else: not identity, faces, artifacts, an action
done wrong inside a shot, or what is said. Basenames only reach the record.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_clip_delta import CUT_SCORE, cut_times  # noqa: E402
from measure_clip_loudness import ebur128, has_audio  # noqa: E402

#: Scene scores at or above this are kept per clip, so the count can be
#: re-read at another threshold. **Reasoned**: well under `CUT_SCORE`, and
#: above the score ordinary motion in these clips reaches.
KEEP_SCORE = 0.15


def scene_scores(path: Path) -> list[list[float]]:
    """[seconds, score] of every frame whose scene score is at least `KEEP_SCORE`."""
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-vf", "select='gte(scene,0)',metadata=print:file=-",
         "-f", "null", "-"], capture_output=True, text=True).stdout
    times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", out)]
    scores = [float(x) for x in re.findall(r"lavfi\.scene_score=([\d.]+)", out)]
    return [[round(t, 3), round(s, 3)] for t, s in zip(times, scores) if s >= KEEP_SCORE]


def score(path: Path, shots: int | None, ref_lufs: float | None) -> dict:
    cuts = cut_times(path)
    loud = ebur128(path) if has_audio(path) else None
    return {"clip": path.name, "cut_times_s": cuts, "cuts": len(cuts),
            "shots_match": None if shots is None else len(cuts) == shots - 1,
            "scene_scores": scene_scores(path), "loudness": loud,
            "loudness_delta_lu": None if loud is None or ref_lufs is None
            else round(loud["integrated_lufs"] - ref_lufs, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--shots", type=int, default=None, help="shots the prompt scripts")
    ap.add_argument("--reference", type=Path, default=None, help="the clip loudness is measured against")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ref = score(args.reference, args.shots, None) if args.reference else None
    ref_lufs = ref["loudness"]["integrated_lufs"] if ref and ref["loudness"] else None
    rows = [score(p, args.shots, ref_lufs) for p in args.clips]
    print(f"{'clip':<60} {'cuts':>4} {'script':>6} {'LUFS':>7} {'vs ref':>7}  cut times")
    for r in ([ref] if ref else []) + rows:
        lufs = f"{r['loudness']['integrated_lufs']:.1f}" if r["loudness"] else "-"
        delta = f"{r['loudness_delta_lu']:+.1f}" if r["loudness_delta_lu"] is not None else "ref" if r is ref else "-"
        match = "-" if r["shots_match"] is None else "ok" if r["shots_match"] else "NO"
        print(f"{r['clip'][-60:]:<60} {r['cuts']:>4} {match:>6} {lufs:>7} {delta:>7}  {r['cut_times_s']}")
    args.out.write_text(json.dumps({"measured_by": "bench/score_clip_shots_loudness.py", "scripted_shots": args.shots,
                                    "cut_score": CUT_SCORE, "keep_score": KEEP_SCORE,
                                    "reference": ref, "clips": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
