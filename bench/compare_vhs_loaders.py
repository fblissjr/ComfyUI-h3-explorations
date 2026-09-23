#!/usr/bin/env python3
"""Do VHS's two upload loaders hand the reference path the same frames? Measured, not read.

    <comfy-venv-python> bench/compare_vhs_loaders.py --out bench/results/<date>_vhs_loader_comparison.json

Written 2026-09-23 when the generated reference-video graphs moved from the
cv2 `VHS_LoadVideo` to `h3_config.REF_VIDEO_LOADER` (VHS's ffmpeg loader). The
two take different inputs (`skip_first_frames` / `select_every_nth` against
`start_time`) and select frames by different mechanisms, so a class swap is
only a no-op if it is shown to be one. Both are driven here through VHS's own
`load_video` with the shipped widgets (`force_rate` 24, `frame_load_cap`
`h3_config.REF_VIDEO_LENGTH`, start 0), and compared four ways:

  frames     Synthetic clips carry each frame's source index as binary blocks,
             so a loaded frame names the source frame it came from exactly. Per
             rate: both index lists, where they differ, and each loader's
             offset from the 24 fps grid (source time minus k/24).
  real clips The shipped placeholder clips. Each loaded frame is matched to a
             source frame decoded the way that loader decodes, near the grid
             position; the record keeps the match error, so a match that is
             not exact reads as a match that is not exact.
  colour     A 24 fps clip, where both loaders keep the same frames, against an
             accurate bt709 reference decode (rgb48, accurate rounding, full
             chroma interpolation): mean absolute error and per-channel bias,
             in 8-bit levels.
  metadata   `VHS_VIDEOINFO` key by key, and the audio each returns.

No GPU and no server; VHS is imported with a stand-in PromptServer, as the VHS
fork's own `tests/fork_metadata_png_only.py` does. The input directory comes
from `bench/_paths.py::comfy_input` (set `H3_COMFY_INPUT` when the server's
command line is not readable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CUSTOM_NODES = REPO.parent
COMFY = CUSTOM_NODES.parent
VHS = CUSTOM_NODES / "ComfyUI-VideoHelperSuite"
sys.path[:0] = [str(COMFY), str(VHS), str(REPO / "workflows"), str(REPO / "bench")]

from comfy.cli_args import args as _cli_args  # noqa: E402

_cli_args.cpu = True  # decoding only: VHS imports core, and core must not take the card from a render
import server  # noqa: E402


class _Anything:
    """Stand-in for the live PromptServer: VHS reads attributes at import."""
    def __getattr__(self, name):
        return _Anything()

    def __call__(self, *a, **k):
        return _Anything()


server.PromptServer.instance = _Anything()
import _paths  # noqa: E402
import cv2  # noqa: E402
import h3_config  # noqa: E402
from build_workflows import (PLACEHOLDER_VIDEO, PLACEHOLDER_VIDEO_SILENT,  # noqa: E402
                             REF_VIDEO_FORCE_RATE)
from videohelpersuite import load_video_nodes as L  # noqa: E402
from videohelpersuite.utils import ffmpeg_path  # noqa: E402

W, H, BITS, BLOCK, PITCH = 960, 544, 12, 64, 128
#: Rates a reference clip plausibly arrives at. 25 is the shipped placeholders'.
SYNTH_RATES = (24, 25, 30, 23.976, 29.97)
CAP = h3_config.REF_VIDEO_LENGTH
RATE = REF_VIDEO_FORCE_RATE


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, check=True, **kw)


def _ffmpeg_version() -> str:
    return _run([ffmpeg_path, "-version"], text=True).stdout.split("\n")[0]


def _git_head(path: Path) -> str:
    return _run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).stdout.strip()


def make_indexed_clip(path: Path, fps: float, n: int) -> None:
    """n frames at fps, each carrying its index as BITS white blocks, with a tone as long as the video."""
    frames = np.zeros((n, H, W, 3), np.uint8)
    for i in range(n):
        for b in range(BITS):
            if (i >> b) & 1:
                r, c = divmod(b, 6)
                frames[i, 64 + r * PITCH: 64 + r * PITCH + BLOCK, 64 + c * PITCH: 64 + c * PITCH + BLOCK] = 255
    _run([ffmpeg_path, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
          "-r", str(fps), "-i", "-", "-f", "lavfi", "-t", str(n / fps), "-i", "sine=frequency=440:sample_rate=48000",
          "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
          "-c:a", "aac", str(path)], input=frames.tobytes())


def read_indices(images) -> list[int]:
    out = []
    for f in images:
        v = 0
        for b in range(BITS):
            r, c = divmod(b, 6)
            m = 16  # sample the block's centre, clear of chroma bleed at its edges
            blk = f[64 + r * PITCH + m: 64 + r * PITCH + BLOCK - m, 64 + c * PITCH + m: 64 + c * PITCH + BLOCK - m]
            if float(blk.mean()) > 0.5:
                v |= 1 << b
        out.append(v)
    return out


def load_both(path: Path, cap: int = CAP):
    common = dict(video=str(path), force_rate=RATE, custom_width=0, custom_height=0, frame_load_cap=cap,
                  format="AnimateDiff")
    cv = L.load_video(**common, skip_first_frames=0, select_every_nth=1)
    ff = L.load_video(**common, start_time=0.0, generator=L.ffmpeg_frame_generator)
    return cv, ff


def grid_offsets_ms(indices, fps) -> dict:
    err = np.asarray(indices) / fps - np.arange(len(indices)) / RATE
    return {"max_abs_ms": round(float(np.abs(err).max()) * 1000, 2),
            "mean_signed_ms": round(float(err.mean()) * 1000, 2)}


def _waveform(audio):
    try:
        return audio["waveform"], None
    except Exception as exc:  # noqa: BLE001 -- VHS raises a bare Exception on a clip with no audio
        return None, type(exc).__name__


def metadata_and_audio(cv, ff) -> dict:
    (_, _, ca, ci), (_, _, fa, fi) = cv, ff
    (wa, ea), (wb, eb) = _waveform(ca), _waveform(fa)
    if wa is None or wb is None:
        audio = {"audio_identical": None, "audio_error_cv": ea, "audio_error_ff": eb}
    else:
        audio = {"audio_identical": bool(wa.shape == wb.shape and (wa == wb).all()), "audio_shape": list(wa.shape)}
    return {"video_info_differs": sorted(k for k in ci if ci[k] != fi.get(k)),
            "video_info_cv": ci, "video_info_ff": fi, **audio}


def synthetic(work: Path) -> list[dict]:
    rows = []
    for fps in SYNTH_RATES:
        n = int(round((CAP / RATE + 0.6) * fps))  # a little longer than the cap needs
        path = work / f"indexed_{fps}.mp4"
        make_indexed_clip(path, fps, n)
        cv, ff = load_both(path)
        a, b = read_indices(cv[0].numpy()), read_indices(ff[0].numpy())
        differ = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
        rows.append({"source_fps": fps, "source_frames": n, "cv_frames": len(a), "ff_frames": len(b),
                     "frames_where_selection_differs": len(differ), "first_differing": [
                         {"k": k, "cv_source_index": a[k], "ff_source_index": b[k]} for k in differ[:6]],
                     "max_index_gap": max((abs(a[k] - b[k]) for k in differ), default=0),
                     "cv_duplicates": len(a) - len(set(a)), "ff_duplicates": len(b) - len(set(b)),
                     "cv_offset_from_24fps_grid": grid_offsets_ms(a, fps),
                     "ff_offset_from_24fps_grid": grid_offsets_ms(b, fps),
                     **metadata_and_audio(cv, ff)})
        print(f"  synthetic {fps:>6} fps: {len(differ)} of {len(a)} frames select differently")
    return rows


def _decode(path: Path, pix_fmt: str, n: int, vf: str | None = None) -> np.ndarray:
    w, h = map(int, _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)], text=True).stdout.strip().split(","))
    cmd = [ffmpeg_path, "-v", "error", "-i", str(path), "-frames:v", str(n)]
    if vf:
        cmd += ["-vf", vf]
    raw = _run(cmd + ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]).stdout
    if pix_fmt == "rgb24":
        return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255
    if pix_fmt == "rgba64le":
        return (np.frombuffer(raw, "<u2").reshape(-1, h, w, 4)[..., :3].astype(np.float32) / 65535)
    return np.frombuffer(raw, "<u2").reshape(-1, h, w, 3).astype(np.float32) / 65535


def _match(loaded: np.ndarray, source: np.ndarray, fps: float) -> tuple[list[int], float, int]:
    """Each loaded frame's nearest source frame within 3 of its grid position; worst error; ties."""
    idx, worst, ties = [], 0.0, 0
    for k, frame in enumerate(loaded):
        centre = int(round(k * fps / RATE))
        cands = range(max(0, centre - 3), min(len(source), centre + 4))
        errs = [(float(np.abs(source[j] - frame).mean()), j) for j in cands]
        errs.sort()
        idx.append(errs[0][1])
        worst = max(worst, errs[0][0])
        ties += len(errs) > 1 and errs[1][0] - errs[0][0] < 1e-7
    return idx, worst, ties


def real_clip(path: Path) -> dict:
    fps = float(Fraction(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=r_frame_rate", "-of", "csv=p=0", str(path)], text=True).stdout.strip()))
    cv, ff = load_both(path)
    n = int(np.ceil(CAP * fps / RATE)) + 5
    ci, cworst, cties = _match(cv[0].numpy(), _decode(path, "rgb24", n), fps)
    fi, fworst, fties = _match(ff[0].numpy(), _decode(path, "rgba64le", n), fps)
    differ = [k for k, (x, y) in enumerate(zip(ci, fi)) if x != y]
    print(f"  real {path.name}: {len(differ)} of {len(ci)} frames select differently")
    return {"clip": path.name, "source_fps": fps, "cv_frames": len(ci), "ff_frames": len(fi),
            "cv_match_worst_mean_abs": cworst, "ff_match_worst_mean_abs": fworst,
            "cv_ambiguous_matches": cties, "ff_ambiguous_matches": fties,
            "frames_where_selection_differs": len(differ),
            "cv_first_last_source_index": [ci[0], ci[-1]], "ff_first_last_source_index": [fi[0], fi[-1]],
            "cv_offset_from_24fps_grid": grid_offsets_ms(ci, fps),
            "ff_offset_from_24fps_grid": grid_offsets_ms(fi, fps),
            **metadata_and_audio(cv, ff)}


def colour(path: Path) -> dict:
    cv, ff = load_both(path, cap=0)
    ref = _decode(path, "rgb48le", len(cv[0]),
                  vf="scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int+full_chroma_inp")
    out: dict = {"clip": path.name, "reference": "ffmpeg rgb48le, bt709 tv, accurate_rnd+full_chroma_int+full_chroma_inp",
           "units": "8-bit levels"}
    for name, img in (("cv", cv[0].numpy()), ("ff", ff[0].numpy())):
        d = (img - ref) * 255
        out[name] = {"mean_abs": round(float(np.abs(d).mean()), 3),
                     "p99_abs": round(float(np.percentile(np.abs(d), 99)), 3),
                     "bias_rgb": [round(float(d[..., c].mean()), 3) for c in range(3)]}
    print(f"  colour {path.name}: cv mean_abs {out['cv']['mean_abs']}, ff mean_abs {out['ff']['mean_abs']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--colour-clip", default="h3ref_24fps_6s.mp4",
                    help="a 24 fps clip in the input directory, where both loaders keep the same frames")
    args = ap.parse_args()
    inp = _paths.comfy_input()
    record = {"question": "do VHS_LoadVideo and VHS_LoadVideoFFmpeg hand H3 the same reference frames "
                          "under the shipped widgets",
              "conditions": {"ffmpeg": _ffmpeg_version(), "opencv": cv2.__version__,
                             "vhs_commit": _git_head(VHS), "h3_commit": _git_head(REPO),
                             "widgets": {"force_rate": RATE, "frame_load_cap": CAP,
                                         "cv": {"skip_first_frames": 0, "select_every_nth": 1},
                                         "ff": {"start_time": 0.0}},
                             "note": "the h3 commit is the one checked out when this ran; a dirty tree is not recorded"}}
    with tempfile.TemporaryDirectory() as tmp:
        record["synthetic"] = synthetic(Path(tmp))
    record["real_clips"] = [real_clip(inp / name) for name in (PLACEHOLDER_VIDEO, PLACEHOLDER_VIDEO_SILENT)]
    record["colour"] = colour(inp / args.colour_clip)
    args.out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
