#!/usr/bin/env python3
"""Is a pair of clips the same take or two takes? Per-frame picture distance and audio envelope correlation.

    python bench/measure_pair_alignment.py A.mp4 B.mp4 [--label NAME] [--expect VERDICT]
    python bench/measure_pair_alignment.py --pairs pairs.json --out record.json

Clip paths are relative to ComfyUI's output directory (`bench/_paths.py::comfy_output`,
so set `H3_COMFY_OUTPUT` when no server owns the port), or absolute. Pass the
silent `.mp4`; the audio is read from its `-audio.mp4` sibling when one exists.

## Why

A reference metric (PSNR, SSIM, LPIPS, ColorVideoVDP) against the fully dense
render means something only when the two clips are the same take: the same
figures in the same places, differing by what the knob did to them. When the
knob moved the sample into a different take, the same numbers measure the
distance between two performances, and a better arm can score worse
(`docs/research/2026-09-19_evaluation_one_judge.md`, item 2;
`docs/research/2026-09-19_where_approximation_is_tolerated.md`, section 1.1).
This says which regime a pair is in, so a reader knows whether any reference
metric logged beside a verdict can be read at all.

## What it measures

- Picture: ffmpeg's `psnr` and `ssim` filters, B against A, frame by frame.
  Per frame: PSNR (average over planes) and SSIM for luma and for all planes.
  Frames ffmpeg reports at infinite PSNR are counted as identical and left out
  of the PSNR summary. Summaries: mean, 5th percentile, minimum and the
  0-based index of the minimum. The verdict reads luma SSIM, because chroma
  planes of two unrelated takes of one prompt still score high and pull the
  all-plane figure toward the middle.
- Sound: mono at `AUDIO_RATE`, an RMS envelope over `ENVELOPE_MS` frames, and
  its Pearson correlation between the two clips at lag zero and at the best
  lag within `MAX_LAG_S` either way (the method of the 2026-09-19 audio record,
  `bench/results/2026-09-19_audio_sage_vs_kitchen.md`).
- The graphs: every literal input that differs between the two embedded
  graphs, by `bench/diff_clip_graphs.py`'s rules, so the record says what the
  pair actually varies.

## The verdict, and how it can fail

The picture: `same_take` when mean luma SSIM is at least
`SAME_TAKE_MIN_LUMA_SSIM`, else `two_takes`; `identical` when every frame is
identical and the audio envelope is too. `aligned_frame_fraction` says how much
of the clip clears the line, for pairs that align in one shot and not the next.
The sound, separately: `aligned` at a best-lag envelope correlation of at
least `SOUND_ALIGNED_MIN_ENVELOPE_R`, else `diverged`. A reference picture
metric is readable on a `same_take` pair, an audio metric on an `aligned` one. A pairs file may give each pair an `expect`;
any pair whose verdict differs from its expectation makes the run exit 1. The
controls a record should carry: one clip against itself (`identical`) and one
known two-seed pair of one arm (`two_takes`). Without both, a green run could
be a tool that says the same word for everything.

Limits: it measures alignment, not quality. A same-take pair can hide a local
defect in its minimum frames; the minimum and its index are in the record for
that reason. Different resolutions or frame counts are refused, not padded.
No GPU.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
from diff_clip_graphs import flat, graph_of  # noqa: E402

#: Mean luma SSIM of B against A at or above which the PICTURE reads as the same
#: take, i.e. aligned closely enough that a reference metric measures the knob
#: and not a different performance. REASONED from a visual check, not measured
#: against a judge: in `bench/results/2026-09-19_pair_alignment.md` the two-seed
#: controls sit far below it, and every pair that sits between them and it
#: shares staging and cast but has figures in different places or poses. Its
#: first version placed the line lower, between a re-sampled pair and the
#: audio-aligned ones, and that record says why it moved.
SAME_TAKE_MIN_LUMA_SSIM = 0.70

#: Best-lag 20 ms envelope correlation at or above which the SOUND reads as the
#: same performance. INHERITED from the alignment test of
#: `bench/results/2026-09-19_audio_sage_vs_kitchen.md`. Picture and sound are
#: judged apart because they come apart: a pair can keep its dialogue and move
#: its figures, or the reverse.
SOUND_ALIGNED_MIN_ENVELOPE_R = 0.90

AUDIO_RATE = 16000     # Hz, mono; the envelope is all this reads, so the rate is not a fidelity choice
ENVELOPE_MS = 20       # inherited from the 2026-09-19 audio record's alignment test
MAX_LAG_S = 0.5        # inherited from the same record: the lag search, either way

VERDICTS = ("identical", "same_take", "two_takes")


def resolve(clip: str) -> Path:
    p = Path(clip)
    if p.is_absolute():
        return p
    return _paths.comfy_output() / p


def shown(path: Path) -> str:
    """The path as a record stores it: relative to the output root, never absolute."""
    try:
        return str(path.relative_to(_paths.comfy_output()))
    except ValueError:
        return path.name


def audio_sibling(video: Path) -> Path | None:
    sib = video.with_name(video.stem + "-audio.mp4")
    return sib if sib.is_file() else None


def probe(path: Path) -> dict:
    raw = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, check=True).stdout
    return orjson.loads(raw)["streams"][0]


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "p5": float(np.percentile(arr, 5)),
            "min": float(arr.min()), "argmin_frame": int(arr.argmin())}


def field(line: str, pattern: str) -> float:
    m = re.search(pattern, line)
    if m is None:
        raise SystemExit(f"refuse: ffmpeg's ssim log line has no {pattern!r}: {line!r}")
    return float(m.group(1))


def picture(a: Path, b: Path) -> dict:
    pa, pb = probe(a), probe(b)
    if (pa["width"], pa["height"]) != (pb["width"], pb["height"]):
        raise SystemExit(f"refuse: sizes differ, {pa['width']}x{pa['height']} against {pb['width']}x{pb['height']}")
    if pa.get("nb_frames") != pb.get("nb_frames"):
        raise SystemExit(f"refuse: frame counts differ, {pa.get('nb_frames')} against {pb.get('nb_frames')}")
    with tempfile.TemporaryDirectory() as tmp:
        plog, slog = Path(tmp) / "psnr.log", Path(tmp) / "ssim.log"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(a), "-i", str(b), "-lavfi",
                        f"[0:v][1:v]psnr=stats_file={plog};[0:v][1:v]ssim=stats_file={slog}",
                        "-f", "null", "-"], check=True)
        psnr_lines = plog.read_text().splitlines()
        ssim_lines = slog.read_text().splitlines()
    psnr = [m.group(1) for m in (re.search(r"psnr_avg:(\S+)", l) for l in psnr_lines) if m]
    luma = [field(l, r"Y:([0-9.]+)") for l in ssim_lines]
    every = [field(l, r"All:([0-9.]+)") for l in ssim_lines]
    finite = [float(v) for v in psnr if v != "inf"]
    return {
        "frames": len(luma),
        "size": [pa["width"], pa["height"]],
        "aligned_frame_fraction": sum(1 for v in luma if v >= SAME_TAKE_MIN_LUMA_SSIM) / len(luma),
        "identical_frames": sum(1 for v in psnr if v == "inf"),
        "psnr_db": summary(finite),
        "ssim_luma": summary(luma),
        "ssim_all": summary(every),
    }


def envelope(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar",
                          str(AUDIO_RATE), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    hop = AUDIO_RATE * ENVELOPE_MS // 1000
    n = len(x) // hop
    return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sound(a: Path | None, b: Path | None) -> dict | None:
    if a is None or b is None:
        return None
    ea, eb = envelope(a), envelope(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    max_lag = int(MAX_LAG_S * 1000 / ENVELOPE_MS)
    best_r, best_lag = None, 0
    for lag in range(-max_lag, max_lag + 1):
        xa, xb = (ea[lag:], eb[: n - lag]) if lag >= 0 else (ea[: n + lag], eb[-lag:])
        r = pearson(xa, xb)
        if r is not None and (best_r is None or r > best_r):
            best_r, best_lag = r, lag
    return {"envelope_frames": n, "envelope_r_lag0": pearson(ea, eb),
            "envelope_r_best": best_r, "best_lag_ms": best_lag * ENVELOPE_MS,
            "bit_identical": bool(np.array_equal(ea, eb))}


def graph_differences(a: Path, b: Path) -> list[str] | str:
    try:
        fa, fb = flat(graph_of(str(a))), flat(graph_of(str(b)))
    except Exception as exc:  # a clip with no embedded graph is reportable, not fatal
        return f"unreadable: {type(exc).__name__}"
    out = []
    for key in sorted(set(fa) | set(fb), key=str):
        va, vb = fa.get(key, "<absent>"), fb.get(key, "<absent>")
        if va != vb:
            cls, n, field = key
            out.append(f"{cls}[{n}].{field}")
    return out


def verdict(pic: dict, snd: dict | None) -> str:
    """The picture's verdict; the sound has its own (`sound_verdict`)."""
    if pic["identical_frames"] == pic["frames"] and (snd is None or snd["bit_identical"]):
        return "identical"
    return "same_take" if pic["ssim_luma"]["mean"] >= SAME_TAKE_MIN_LUMA_SSIM else "two_takes"


def sound_verdict(snd: dict | None) -> str:
    if snd is None:
        return "no_audio"
    if snd["bit_identical"]:
        return "identical"
    r = snd["envelope_r_best"]
    return "aligned" if r is not None and r >= SOUND_ALIGNED_MIN_ENVELOPE_R else "diverged"


def measure(label: str, a_arg: str, b_arg: str, expect: str | None) -> dict:
    a, b = resolve(a_arg), resolve(b_arg)
    for p in (a, b):
        if not p.is_file():
            raise SystemExit(f"refuse: no clip at {shown(p)}")
    pic = picture(a, b)
    snd = sound(audio_sibling(a), audio_sibling(b))
    row = {"label": label, "a": shown(a), "b": shown(b), "graph_differences": graph_differences(a, b),
           "picture": pic, "sound": snd, "verdict": verdict(pic, snd), "sound_verdict": sound_verdict(snd)}
    if expect is not None:
        row["expect"] = expect
        row["as_expected"] = row["verdict"] == expect
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("clips", nargs="*", help="A and B, when --pairs is not given")
    ap.add_argument("--label", default="pair")
    ap.add_argument("--expect", choices=VERDICTS, default=None)
    ap.add_argument("--pairs", type=Path, help='JSON list of {"label", "a", "b", optional "expect"}')
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.pairs:
        specs = orjson.loads(args.pairs.read_bytes())
    elif len(args.clips) == 2:
        specs = [{"label": args.label, "a": args.clips[0], "b": args.clips[1], "expect": args.expect}]
    else:
        ap.error("give two clips, or --pairs")
    for spec in specs:
        if spec.get("expect") not in (None, *VERDICTS):
            raise SystemExit(f"refuse: {spec['label']}: expect must be one of {VERDICTS}")

    rows = []
    for spec in specs:
        row = measure(spec["label"], spec["a"], spec["b"], spec.get("expect"))
        rows.append(row)
        pic, snd = row["picture"], row["sound"]
        env = "no audio" if snd is None else f"env r {snd['envelope_r_best']:.3f} at {snd['best_lag_ms']} ms"
        mark = "" if "expect" not in row else ("  ok" if row["as_expected"] else f"  FAIL, expected {row['expect']}")
        print(f"{row['label']:44s} ssim_y {pic['ssim_luma']['mean']:.3f} (min {pic['ssim_luma']['min']:.3f} "
              f"@{pic['ssim_luma']['argmin_frame']}, aligned {pic['aligned_frame_fraction']:.2f})  {env}  "
              f"-> {row['verdict']} / sound {row['sound_verdict']}{mark}")

    record = {"tool": "bench/measure_pair_alignment.py",
              "same_take_min_luma_ssim": SAME_TAKE_MIN_LUMA_SSIM,
              "sound_aligned_min_envelope_r": SOUND_ALIGNED_MIN_ENVELOPE_R,
              "audio": {"rate": AUDIO_RATE, "envelope_ms": ENVELOPE_MS, "max_lag_s": MAX_LAG_S},
              "pairs": rows}
    if args.out:
        args.out.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2) + b"\n")
    failed = [r["label"] for r in rows if r.get("as_expected") is False]
    if failed:
        print(f"FAIL: {len(failed)} pair(s) not as expected: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
