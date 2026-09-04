#!/usr/bin/env python3
"""EBU R128 loudness per rendered clip, grouped by scene and rung, as a record.

Written 2026-09-04 for one sentence: after scoring the ladder the owner said the
audio was good across the board but "a bit quieter on the pdd ones". That is a
claim about a level, and a level has a home the machine can re-derive. This
runs ffmpeg's `ebur128` filter (integrated loudness, loudness range, true peak)
over every clip an outputs record names, and writes one record with the per-arm
values and, per scene, each rung's integrated loudness against the scene's
`dense` rung.

    python bench/measure_clip_loudness.py \\
        --outputs bench/results/2026-09-03_ladder_outputs.json \\
        --output-root <share the launcher names> \\
        --out bench/results/2026-09-04_ladder_audio_loudness.json

What it is and is not. It is descriptive: one clip per arm, so a per-scene
delta is a paired draw from two different samples (CLAUDE.md's different-sample
rule), not a measurement of the rung. Five scenes give five paired draws, and
the most the record can say is how many share a sign. It reads the muxed
`-audio.mp4` the combine node wrote, which is what the blind singles carry
since 0.99.37; `bench/grade_arm_audio_spectrum.py` is the tool for a timbre
claim across seeds and refuses this ladder's one-clip arms for the reason its
docstring gives. Loudness here is a different question from that one.

Refusals: a clip named by the record that is not under the output root, a clip
with no audio stream, or an ffmpeg summary that does not parse. Basenames only
reach the record.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

_LUFS = re.compile(r"^\s*I:\s+(-?[\d.]+) LUFS", re.M)
_LRA = re.compile(r"^\s*LRA:\s+(-?[\d.]+) LU", re.M)
_PEAK = re.compile(r"^\s*Peak:\s+(-?[\d.]+) dBFS", re.M)


def ebur128(path: Path) -> dict:
    """Run ffmpeg's ebur128 summary on the first audio stream and parse it."""
    cmd = ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-map", "0:a:0",
           "-af", "ebur128=peak=true", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stderr
    if proc.returncode != 0:
        sys.exit(f"refuse: ffmpeg failed on {path.name}: {text.strip().splitlines()[-1] if text.strip() else proc.returncode}")
    m_i, m_lra, m_peak = _LUFS.search(text), _LRA.search(text), _PEAK.search(text)
    if not (m_i and m_lra and m_peak):
        sys.exit(f"refuse: could not parse the ebur128 summary for {path.name}")
    return {"integrated_lufs": float(m_i.group(1)),
            "loudness_range_lu": float(m_lra.group(1)),
            "true_peak_dbfs": float(m_peak.group(1))}


def has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def split_label(label: str) -> tuple[str, str]:
    """`<scene>_<rung>` as the ladder manifest spells it."""
    scene, _, rung = label.rpartition("_")
    return scene, rung


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--outputs", required=True,
                    help="an outputs record: {'arms': [{'label', 'outputs': [basename, ...]}]}")
    ap.add_argument("--output-root", default=os.environ.get("H3_COMFY_OUTPUT"),
                    help="the output directory the launcher names; clips are looked up under "
                         "<root>/Video/<basename> then <root>/<basename>. Default $H3_COMFY_OUTPUT")
    ap.add_argument("--baseline-rung", default="dense",
                    help="the rung every other rung in a scene is compared against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default=None, help="one sentence on why this record exists")
    args = ap.parse_args()
    if not args.output_root:
        sys.exit("refuse: no --output-root and H3_COMFY_OUTPUT is unset")
    root = Path(args.output_root)
    if not root.is_dir():
        sys.exit(f"refuse: output root is not a directory: {root}")

    outputs_path = Path(args.outputs)
    rec = json.loads(outputs_path.read_text())
    arms = rec.get("arms") or []
    if not arms:
        sys.exit(f"refuse: {outputs_path} names no arms")

    per_arm = []
    for arm in arms:
        names = arm.get("outputs") or []
        if len(names) != 1:
            sys.exit(f"refuse: arm {arm.get('label')!r} names {len(names)} outputs; one clip per arm expected")
        name = names[0]
        cands = [root / "Video" / name, root / name]
        path = next((c for c in cands if c.is_file()), None)
        if path is None:
            sys.exit(f"refuse: {name} not found under the output root (looked in Video/ and the root)")
        if not has_audio(path):
            sys.exit(f"refuse: {name} has no audio stream; measure the muxed -audio.mp4")
        scene, rung = split_label(arm["label"])
        row = {"label": arm["label"], "scene": scene, "rung": rung, "clip": name,
               "seed": arm.get("seed"), "graph": arm.get("graph"),
               "prompt_id": (arm.get("rendered") or {}).get("prompt_id")}
        row.update(ebur128(path))
        per_arm.append(row)
        print(f"  {arm['label']:>20}  I {row['integrated_lufs']:6.1f} LUFS  LRA {row['loudness_range_lu']:5.1f} LU  peak {row['true_peak_dbfs']:5.1f} dBFS")

    # Per scene, each rung against the baseline rung's integrated loudness.
    by_scene: dict[str, dict[str, dict]] = {}
    for row in per_arm:
        by_scene.setdefault(row["scene"], {})[row["rung"]] = row
    deltas: dict[str, dict[str, float]] = {}
    for scene, rungs in by_scene.items():
        base = rungs.get(args.baseline_rung)
        if base is None:
            sys.exit(f"refuse: scene {scene!r} has no {args.baseline_rung!r} rung to compare against")
        deltas[scene] = {rung: round(r["integrated_lufs"] - base["integrated_lufs"], 1)
                         for rung, r in rungs.items() if rung != args.baseline_rung}
    # Sign agreement per rung across scenes: the most one clip per arm can say.
    rung_names = sorted({r["rung"] for r in per_arm if r["rung"] != args.baseline_rung})
    sign_summary = {}
    for rung in rung_names:
        vals = [d[rung] for d in deltas.values() if rung in d]
        sign_summary[rung] = {
            "n_scenes": len(vals),
            "quieter_than_baseline": sum(1 for v in vals if v < 0),
            "louder_than_baseline": sum(1 for v in vals if v > 0),
            "equal": sum(1 for v in vals if v == 0),
            "delta_lufs_by_scene": {s: d[rung] for s, d in deltas.items() if rung in d},
            "median_delta_lufs": sorted(vals)[len(vals) // 2] if vals else None,
        }

    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": f"bench/measure_clip_loudness.py over {outputs_path.name}; ffmpeg ebur128=peak=true on the first audio stream of each muxed clip",
        "note": args.note,
        "reading": ("descriptive. One clip per arm, so a per-scene delta is a paired draw from two "
                    "different samples, not a measurement of the rung; the sign summary counts how "
                    "many scenes agree, which is the most this record can say. A seed sweep per arm "
                    "is what would turn a delta into a level difference."),
        "baseline_rung": args.baseline_rung,
        "units": {"integrated_lufs": "EBU R128 integrated loudness, LUFS (higher is louder)",
                  "loudness_range_lu": "EBU R128 loudness range, LU",
                  "true_peak_dbfs": "true peak, dBFS",
                  "delta_lufs": "rung integrated loudness minus the scene's baseline rung"},
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0],
        "per_arm": per_arm,
        "delta_vs_baseline_by_scene": deltas,
        "sign_summary_by_rung": sign_summary,
    }
    # basenames only: refuse if any absolute path slipped in
    def scrub(node, where="record"):
        if isinstance(node, dict):
            for k, v in node.items():
                scrub(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                scrub(v, f"{where}[{i}]")
        elif isinstance(node, str) and (node.startswith("/") or node.startswith("~") or "/home/" in node):
            sys.exit(f"refuse: {where} carries an absolute path: {node!r}")
    scrub(record)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print()
    for rung, s in sign_summary.items():
        print(f"{rung:>8}: quieter than {args.baseline_rung} in {s['quieter_than_baseline']} of {s['n_scenes']} scenes, "
              f"louder in {s['louder_than_baseline']}; median delta {s['median_delta_lufs']} LUFS")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
