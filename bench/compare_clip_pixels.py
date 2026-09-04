#!/usr/bin/env python3
"""Are two rendered clips the same render? Decoded pixels and samples, never bytes.

Written 2026-09-04 for one question: ComfyUI was pulled that day and the pull
turned the compiler's malloc graph on around the H3 forward, so a clip rendered
after it is in a different execution regime from the 2026-09-03 ladder. If a
dense rung re-rendered after the pull is pixel-identical to its 2026-09-03
clip, the regime changed timing only and the ladder's clips stay valid as blind
references for anything rendered later; if not, every cross-regime pairing
carries that caveat. `bench/verify_vsa_render.py` owns the mechanism this
reuses (decoded frames hashed, the container ignored, the arms identified from
the graph each file embeds) and explains why the mp4 bytes cannot answer it.

    python bench/compare_clip_pixels.py <a.mp4> <b.mp4> --out bench/results/<date>_<what>.json

What it reports. Video: the sha256 of the decoded RGB frames of each clip;
when they differ, per-frame mean and max absolute difference over the full
decoded frames, the count of frames that differ at all, and the first such
frame. Audio: the sha256 of the decoded PCM of each clip's first audio stream
and, when they differ, the max absolute sample difference. Provenance: the
seed and the set of node class types read from each file's embedded graph, so
the record says whether the two clips claim the same render.

Refusals: a clip that does not decode; two clips whose decoded frame counts or
frame sizes differ (that is not the same render and the per-frame statistics
would be meaningless); an absolute path in the record. Two clips whose
embedded seeds differ are compared anyway and the record says so: the caller
may be asking exactly that question.

Control, run on 2026-09-04 before the first real use: a clip against itself
reports identical on both streams; the ladder's stairwell dense against its
sage rung (same seed, different kernel) reports every frame differing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from verify_vsa_render import embedded_graph, seed_of  # noqa: E402


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]),
            "nb_frames": int(s.get("nb_frames") or 0), "rate": s.get("r_frame_rate")}


def decode_video(path: Path, w: int, h: int) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        sys.exit(f"refuse: could not decode video of {path.name}: {proc.stderr.decode()[:200]}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(raw) // (w * h * 3)
    if n == 0 or len(raw) != n * w * h * 3:
        sys.exit(f"refuse: decoded byte count of {path.name} is not whole frames at {w}x{h}")
    return raw.reshape(n, h, w, 3)


def decode_audio(path: Path) -> np.ndarray | None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le",
         "-acodec", "pcm_s16le", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.int16)


def graph_summary(path: Path) -> dict:
    g = embedded_graph(path)
    if g is None:
        return {"embedded_graph": False, "seed": None, "class_types": None}
    kinds = sorted({n.get("class_type") for n in g.values() if isinstance(n, dict)})
    return {"embedded_graph": True, "seed": seed_of(g), "class_types": kinds}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--out", default=None, help="record path; printed only when omitted")
    ap.add_argument("--note", default=None, help="one sentence on why this comparison exists")
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    for p in (a, b):
        if not p.is_file():
            sys.exit(f"refuse: not a file: {p.name}")

    pa, pb = probe(a), probe(b)
    if (pa["width"], pa["height"]) != (pb["width"], pb["height"]):
        sys.exit(f"refuse: frame sizes differ ({pa['width']}x{pa['height']} vs "
                 f"{pb["width"]}x{pb['height']}); not the same render")
    va = decode_video(a, pa["width"], pa["height"])
    vb = decode_video(b, pb["width"], pb["height"])
    if va.shape[0] != vb.shape[0]:
        sys.exit(f"refuse: decoded frame counts differ ({va.shape[0]} vs {vb.shape[0]}); not the same render")
    ha, hb = hashlib.sha256(va.tobytes()).hexdigest(), hashlib.sha256(vb.tobytes()).hexdigest()
    video = {"frames": int(va.shape[0]), "size": f"{pa['width']}x{pa['height']}",
             "sha256_a": ha, "sha256_b": hb, "identical": ha == hb}
    if ha != hb:
        d = np.abs(va.astype(np.int16) - vb.astype(np.int16))
        per_frame_mean = d.reshape(d.shape[0], -1).mean(axis=1)
        per_frame_max = d.reshape(d.shape[0], -1).max(axis=1)
        differing = np.nonzero(per_frame_max > 0)[0]
        video.update({
            "frames_differing": int(differing.size),
            "first_differing_frame": int(differing[0]) if differing.size else None,
            "mean_abs_diff_over_all_frames": float(per_frame_mean.mean()),
            "max_abs_diff": int(per_frame_max.max()),
            "per_frame_mean_abs_diff_quantiles": {
                q: float(np.quantile(per_frame_mean, float(q))) for q in ("0.0", "0.5", "1.0")},
            "units": "8-bit RGB levels, 0-255",
        })

    aa, ab = decode_audio(a), decode_audio(b)
    if aa is None or ab is None:
        audio = {"present_a": aa is not None, "present_b": ab is not None, "identical": None}
    else:
        sa, sb = hashlib.sha256(aa.tobytes()).hexdigest(), hashlib.sha256(ab.tobytes()).hexdigest()
        audio = {"samples_a": int(aa.size), "samples_b": int(ab.size),
                 "sha256_a": sa, "sha256_b": sb, "identical": sa == sb}
        if sa != sb:
            n = min(aa.size, ab.size)
            audio["max_abs_sample_diff"] = int(np.abs(aa[:n].astype(np.int32) - ab[:n].astype(np.int32)).max())
            audio["units"] = "16-bit PCM levels"

    ga, gb = graph_summary(a), graph_summary(b)
    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/compare_clip_pixels.py; ffmpeg decode to rgb24 and s16le, sha256 and numpy over the decoded arrays",
        "note": args.note,
        "clip_a": a.name, "clip_b": b.name,
        "provenance": {"a": ga, "b": gb,
                       "same_seed": (ga["seed"] == gb["seed"]) if ga["seed"] is not None and gb["seed"] is not None else None,
                       "same_class_types": (ga["class_types"] == gb["class_types"]) if ga["class_types"] and gb["class_types"] else None},
        "video": video, "audio": audio,
        "reading": ("identical on both streams means the same render; a differing video with the same seed and "
                    "the same class types means the substrate changed numerics between the two renders, "
                    "and the per-frame statistics say by how much in 8-bit levels"),
    }

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

    print(f"video: {'IDENTICAL' if video['identical'] else 'DIFFERENT'} over {video['frames']} frames at {video['size']}"
          + ("" if video["identical"] else
             f"; {video['frames_differing']} frames differ, first at {video['first_differing_frame']}, "
             f"mean abs diff {video['mean_abs_diff_over_all_frames']:.3f}, max {video['max_abs_diff']}"))
    print(f"audio: {audio.get('identical')}" + ("" if audio.get("identical") in (True, None) else
                                                f"; max abs sample diff {audio['max_abs_sample_diff']}"))
    print(f"provenance: same seed {record['provenance']['same_seed']}, same class types {record['provenance']['same_class_types']}")
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1))
        print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
