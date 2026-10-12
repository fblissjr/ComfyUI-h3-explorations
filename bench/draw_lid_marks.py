#!/usr/bin/env python3
"""Draw a shot's eye openings on its body-model video: the motion video a masked load reads for expression.

    python bench/draw_lid_marks.py --mesh MESH.mp4 --points SHOT_points.npz --first 1575 --count 155 \\
        --out data/<job>/lead_mesh_with_lids_1575.mp4 --by <session> [--source SOURCE.mov] \\
        [--bare 1533-1540 --bare-by <who looked>] [--grow 3] [--mesh-first N] [--points-first N] [--source-first N]

The body-model video carries the head; its face is a neutral modelled one that never changes, so nothing in
it says how open the eyes are. This
fills the area between each eye's upper and lower lash line, from the pose model's points on the ORIGINAL
(`bench/pose_along_shot.py`), white with a thin black edge on the body model's own frame. Lids only: the same
points' lip lines were judged and are not drawn (`bench/_lib/face_points.py::lip_gap`).

Everything is named in and named out. It reads the mesh video and the points file it is given and writes the
video, a record beside it (`.json`) and, with `--source`, a sheet of a few frames laid over the original to
look at before the video is used. Nothing is looked up by a key and nothing made earlier is reused.

THE SIZE. Each eye's outline is grown about its own centre by one factor for the whole shot:
`h3_config.LID_MARKS["face_px"]` over the shot's median face height, or `--grow` to set it by hand. The
record says the factor and the marks' size on the canvas.

WHAT IT CANNOT KNOW, and says:
- Whether a head is turned away. The model still places lid points on the back of a head and scores them
  well. Frames to leave bare come from a person, by name (`--bare`, `--bare-by`); the tool leaves none bare
  by itself.
- Whether the lash lines are far enough apart to draw from. It prints the shot's lid gap in canvas pixels
  beside `h3_config.POSE_READING["lid_gap_px_to_draw"]`, says which side of that line the shot's median is and
  what the line is, and draws either way. Whether that is far enough is the reader's to say.
- Whether the points and the body-model video are on the source's clock. It is told which source frame each
  begins on and believes it. Three looks stand against a wrong answer, none of them this run's own word:
  with `--source` it sets how much the body-model video changes from frame to frame against how much the
  source does, a frame early and a frame late, and fails if a neighbour fits better; the sheet lays the
  marked frames over the original for a person; and `verify`, below, is for a second reader.

WHAT THE READ-BACK IS. After writing, the tool reads the video back and sets the white it finds on each frame
against the area drawn for that frame, one frame early and one late. That catches a file whose frames do not
hold what was drawn for them: one dropped, repeated or out of order, a mark on a frame left bare, a frame
with none. It cannot catch points given on the wrong clock, since the drawing and the area come from the same
rows.

    python bench/draw_lid_marks.py verify --video MARKED.mp4 --mesh MESH.mp4 --points SHOT_points.npz --first 1575 --count 155

`verify` reads a marked video somebody else made against a points file, with the reader's OWN account of
which frame each begins on. It is the same read-back, and there it does catch a wrong clock, because the
rows are chosen by the reader and not by the maker.

A passing run says the video holds the marks its points describe, on the frames they describe. It does not
say the marks are on her eyes: open the sheet.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402
from _lib import face_points as fp  # noqa: E402
from _lib.frames import FIT, FPS, probe, stream  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

#: Marks are drawn this many times larger and reduced, so their edges are smooth. Inherited: the one-off of
#: 2026-10-11 whose marks carried.
SUPER = 4
#: A pixel counts as a mark's white above this on every channel. Reasoned: the body model is mid grey and its
#: own highlights stay well under it; the encoder leaves a filled mark above it.
WHITE = 235
#: How much better the right frame must fit than a neighbour for the read-back to pass, as a difference of
#: correlations. Reasoned: on a shot whose eyes move, a one-frame slip costs several times this
#: (`bench/check_draw_lid_marks.py` holds a slipped file to it).
LAG_MARGIN = 0.01
#: How much of a mark is not white: the black edge and the encoder take about this many pixels of its outline,
#: all the way round. Measured: `bench/check_draw_lid_marks.py` draws marks of a dozen sizes and holds the white
#: it reads back to the outline's area less its length times this.
EDGE_PX = 1.0
#: A mark expected to leave less white than this (square pixels, both eyes) may leave none: it is listed as too
#: thin and is not a failure. One expected to leave more and found with none is. Reasoned: the spread of that
#: same measurement about its line.
THIN_PX2 = 80.0


def frames_arg(text: str) -> list[int]:
    """`1533-1540,1560` as source frame numbers."""
    out: list[int] = []
    for part in filter(None, (t.strip() for t in text.split(","))):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def draw(mesh: np.ndarray, rings: list[list[np.ndarray]], grow: float, bare: set[int]) -> np.ndarray:
    """The mesh frames [n, h, w, 3] with every frame's eye outlines filled, grown `grow` times about their centres."""
    marks = h3_config.LID_MARKS
    h, w = mesh.shape[1:3]
    out = np.empty_like(mesh)
    for f in range(len(mesh)):
        im = Image.fromarray(mesh[f]).resize((w * SUPER, h * SUPER), Image.BICUBIC)
        d = ImageDraw.Draw(im)
        for ring in ([] if f in bare else rings[f]):
            mid = ring.mean(0)
            poly = [tuple(v * SUPER) for v in mid + (ring - mid) * grow]
            d.polygon(poly, fill=tuple(marks["fill"]))
            d.line(poly + [poly[0]], fill=tuple(marks["edge"]), width=SUPER)
        out[f] = np.asarray(im.resize((w, h), Image.LANCZOS))
    return out


def write_video(path: Path, frames: np.ndarray) -> None:
    n, h, w = frames.shape[:3]
    enc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    enc.communicate(frames.tobytes())
    if enc.returncode:
        raise RuntimeError(f"ffmpeg could not write {path}")


def white_in(path: Path, mesh: np.ndarray) -> np.ndarray:
    """How many pixels a frame are a mark's white in the written video and were not white in the mesh it was drawn on."""
    n, h, w = mesh.shape[:3]
    got = np.stack(list(stream(str(path), (w, h), 0, n, pix="rgb24")))
    if len(got) != n:
        raise RuntimeError(f"{path} reads back {len(got)} frames, {n} were written")
    return ((got.min(-1) > WHITE) & ~(mesh.min(-1) > WHITE)).sum((1, 2)).astype(float)


def fit_by_lag(white: np.ndarray, area: np.ndarray, drawn: np.ndarray) -> dict[int, float]:
    """The correlation, over the frames that carry marks, of the white found with the area drawn, by frames of slip."""
    out = {}
    for lag in (-1, 0, 1):
        a = white[max(0, lag):len(white) - max(0, -lag)]
        b = area[max(0, -lag):len(area) - max(0, lag)]
        keep = drawn[max(0, lag):len(drawn) - max(0, -lag)] & drawn[max(0, -lag):len(drawn) - max(0, lag)]
        out[lag] = float(np.corrcoef(a[keep], b[keep])[0, 1]) if keep.sum() > 2 and a[keep].std() > 0 and b[keep].std() > 0 else float("nan")
    return out


def change(frames: np.ndarray) -> np.ndarray:
    """How much the picture changes from each frame to the next: the mean difference in grey levels."""
    g = frames.astype(np.float32).mean(-1) if frames.ndim == 4 else frames.astype(np.float32)
    return np.abs(np.diff(g, axis=0)).mean((1, 2))


def moves_with(source: np.ndarray, mesh: np.ndarray) -> dict[int, float]:
    """The correlation of the body-model video's change a frame with the source's, by frames of slip. No pose point."""
    ys, xs = np.nonzero((mesh.max(-1) > 24).any(0))                 # where the body model ever is: the rest of the picture moves for other reasons
    if len(ys):
        source, mesh = (v[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1] for v in (source, mesh))
    a, b = change(source), change(mesh)
    out = {}
    for lag in (-1, 0, 1):
        x, y = b[max(0, lag):len(b) - max(0, -lag)], a[max(0, -lag):len(a) - max(0, lag)]
        out[lag] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    return out


def source_frames(source: str, size: tuple[int, int], first: int, count: int, source_first: int = 0) -> np.ndarray:
    """The original's frames `first` on, fitted to the canvas as the loader fits them. `source_first` is the source
    frame of the video's frame 0: 0 for the original itself, the span's first for a video cut from it."""
    w, h = size
    got = list(stream(source, (w, h), first - source_first, count, vf=FIT.format(w=w, h=h), pix="rgb24")) if first >= source_first else []
    if len(got) != count:
        raise SystemExit(f"{source} gave {len(got)} of the {count} frames from source frame {first}, taking its frame 0 to be source frame {source_first}: "
                         "say which source frame it begins on with --source-first")
    return np.stack(got)


def slipped(lags: dict[int, float]) -> bool:
    """True when a neighbouring frame fits better than the frame itself by more than the margin."""
    other = [v for k, v in lags.items() if k != 0 and not np.isnan(v)]
    return bool(other and not np.isnan(lags[0]) and max(other) - lags[0] >= LAG_MARGIN)


def white_expected(rings: list[np.ndarray], factor: float) -> float:
    """The white a frame's marks should leave: the outlines' area less what the edge takes of it."""
    area = sum(fp.ring_area(r) for r in rings) * factor ** 2
    edge = sum(np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1).sum() for r in rings) * factor
    return float(max(area - edge * EDGE_PX, 0.0))


def read_back(video: Path, mesh: np.ndarray, area: np.ndarray, drawn: np.ndarray) -> dict:
    """The written video against the white each frame's marks should leave: what was found, how it fits by slip, a verdict."""
    white = white_in(video, mesh)
    lags = fit_by_lag(white, area, drawn)
    on_bare = int(white[~drawn].max()) if (~drawn).any() else 0
    empty = [int(f) for f in np.nonzero(drawn & (white == 0) & (area > THIN_PX2))[0]]
    thin = [int(f) for f in np.nonzero(drawn & (area <= THIN_PX2))[0]]
    other = [v for k, v in lags.items() if k != 0 and not np.isnan(v)]
    no_change = bool(np.isnan(lags[0]))                              # the eyes never move enough to tell a frame from its neighbour
    ahead = (lags[0] - max(other)) if other and not no_change else float("nan")
    passed = bool(on_bare == 0 and not empty and white[drawn].sum() > 0 and (no_change or not other or ahead >= LAG_MARGIN))
    return {"white": white, "lags": lags, "on_bare": on_bare, "marked_frames_with_no_white": empty, "too_thin": thin, "no_change": no_change,
            "passed": passed}


def sheet(path: Path, source_frames: np.ndarray, first: int, marked: np.ndarray, points: dict, count: int) -> list[int]:
    """A few frames in three panels (the original, the marked body model, the two blended), cut round the nose."""
    h, w = marked.shape[1:3]
    idx = points["index"]
    picks = sorted(set(np.linspace(0, count - 1, 6).astype(int).tolist()))
    src = source_frames
    side, tile = max(120, int(2.2 * np.median([fp.face_height(p, idx) for p in points["keypoints"]]))), 440
    rows = []
    for f in picks:
        x, y = points["keypoints"][f][idx["tip_of_nose"]]
        x0, y0 = int(min(max(x - side / 2, 0), w - side)), int(min(max(y - side * 0.6, 0), h - side))
        a, b = (Image.fromarray(v[f][y0:y0 + side, x0:x0 + side]) for v in (src, marked))
        row = Image.new("RGB", (tile * 3, tile))
        for i, t in enumerate((a, b, Image.blend(a, b, 0.5))):
            row.paste(t.resize((tile, tile), Image.LANCZOS), (i * tile, 0))
        ImageDraw.Draw(row).text((6, 6), f"source frame {first + f}: the original | the body model with the marks | the two blended", fill=(255, 255, 0))
        rows.append(row)
    out = Image.new("RGB", (tile * 3, tile * len(rows)))
    for i, r in enumerate(rows):
        out.paste(r, (0, i * tile))
    out.save(path, quality=90)
    return [first + f for f in picks]


def run(mesh_path: str, points_path: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        mesh_first: int | None = None, points_first: int | None = None, grow: float | None = None, bare: list[int] | None = None,
        bare_by: str = "", source: str = "", source_first: int = 0, say=print) -> dict:
    """Draw, write, read back, record. Returns the record; `passed` is False when the read-back fits a neighbour."""
    if bare and not bare_by:
        raise SystemExit("--bare names frames a person looked at: say who with --bare-by")
    points = fp.load(points_path, names)
    pf = points["first"] if points_first is None else points_first
    if pf is None:
        raise SystemExit(f"{points_path} does not say which source frame its first row is: give --points-first")
    off = first - pf
    if off < 0 or off + count > len(points["keypoints"]):
        raise SystemExit(f"{points_path} covers source frames {pf} to {pf + len(points['keypoints']) - 1}; frames {first} to {first + count - 1} were asked for")
    points = dict(points, keypoints=points["keypoints"][off:off + count], scores=points["scores"][off:off + count])
    idx = points["index"]
    w, h, have = probe(mesh_path)
    skip = first - (first if mesh_first is None else mesh_first)
    if skip < 0 or skip + count > have:
        raise SystemExit(f"{mesh_path} has {have} frames from source frame {first - skip}; frames {first} to {first + count - 1} were asked for")
    if points["size"] and tuple(points["size"]) != (w, h):
        raise SystemExit(f"the points were made on a picture of {points['size']} and the body-model video is {(w, h)}: they are not on one canvas")
    mesh = np.stack(list(stream(mesh_path, (w, h), skip, count, pix="rgb24")))
    rings = [fp.lid_rings(p, idx) for p in points["keypoints"]]
    face = float(np.median([fp.face_height(p, idx) for p in points["keypoints"]]))
    gap = float(np.median([sum(fp.lid_gap_px(p, idx)) / 2 for p in points["keypoints"]]))
    factor = float(grow) if grow else h3_config.LID_MARKS["face_px"] / face
    left = {f - first for f in (bare or []) if first <= f < first + count}
    marked = draw(mesh, rings, factor, left)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(out, marked)
    drawn = np.array([f not in left for f in range(count)])
    area = np.array([white_expected(rings[f], factor) if drawn[f] else 0.0 for f in range(count)])
    back = read_back(out, mesh, area, drawn)
    white, lags, on_bare, no_change = back["white"], back["lags"], back["on_bare"], back["no_change"]
    src = source_frames(source, (w, h), first, count, source_first) if source else None
    moves = moves_with(src, mesh) if src is not None else None
    off_clock = bool(moves and slipped(moves))
    passed = bool(back["passed"] and not off_clock)
    size = [float(np.median([np.ptp(r[:, k]) for f in range(count) for r in rings[f]]) * factor) for k in (0, 1)]
    line = h3_config.POSE_READING["lid_gap_px_to_draw"]
    record = {
        "video": out.name, "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames": [first, first + count - 1], "size": [w, h],
        "made_from": {"body_model_video": str(mesh_path), "its_first_frame_is_source_frame": first - skip, "points": str(points_path),
                      "their_first_row_is_source_frame": pf},
        "grown": round(factor, 3), "grown_by": "--grow" if grow else f"h3_config.LID_MARKS['face_px'] over the shot's median face height ({face:.0f} px)",
        "a_mark_on_the_canvas_px": {"wide": round(size[0], 1), "tall": round(size[1], 1)},
        "lid_gap_px_median": round(gap, 2), "lid_gap_px_to_draw": line,
        "lid_gap_median_is": ("under" if gap < line else "over") + " that line",
        "that_line_is": "a starting value between the median of the one shot whose marks carried and those of shots whose lash lines are under a pixel apart",
        "frames_left_bare": sorted(first + f for f in left), "left_bare_by": bare_by,
        "not_known": ["whether a head is turned away on any frame: the model places and scores lid points on the back of a head",
                      "whether the marks sit on her eyes: open the sheet"],
        "read_back": {"white_px_a_frame": {"least": int(white.min()), "median": int(np.median(white)), "most": int(white.max())},
                      "fit_by_frames_of_slip": {str(k): (None if np.isnan(v) else round(v, 4)) for k, v in lags.items()},
                      "white_on_a_bare_frame": on_bare, "marked_frames_with_no_white": [first + f for f in back["marked_frames_with_no_white"]],
                      "marks_too_thin_to_leave_much_white": {"frames": [first + f for f in back["too_thin"]],
                                                              "which_are": f"frames whose marks should leave under {THIN_PX2:g} square pixels of white, both eyes: the outline's area less about a pixel of its length"},
                      "eyes_never_change": no_change, "passed": back["passed"],
                      "what_it_cannot_catch": "points or a body-model video given on the wrong clock: the drawing and the area come from the same rows"},
        "body_model_against_the_source": ("not looked at: no --source" if moves is None else {
            "its_change_a_frame_against_the_sources_by_frames_of_slip": {str(k): (None if np.isnan(v) else round(v, 4)) for k, v in moves.items()},
            "a_neighbour_fits_better": off_clock,
            "says": ("a neighbouring frame fits better: the body-model video is NOT on the source's clock as told" if off_clock else
                     "no neighbour fits better" + ("" if not np.isnan(moves[0]) else "; nothing moves enough to tell"))}),
    }
    if source:
        sheet_path = out.with_name(out.stem + "_sheet.jpg")
        record["sheet"] = {"file": sheet_path.name, "frames": sheet(sheet_path, src, first, marked, points, count), "looked_at_by": ""}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"face {face:.0f} px tall at the median; lid gap {gap:.2f} px, {record['lid_gap_median_is']} of {line} px ({record['that_line_is']})")
    say(f"marks grown {factor:.2f} times: {size[0]:.0f} px wide and {size[1]:.0f} px tall on the canvas at the median")
    if back["too_thin"]:
        say(f"on {len(back['too_thin'])} of {count} frames the marks should leave under {THIN_PX2:g} square pixels of white, both eyes (the outline's area less about a pixel of its length)")
    say(f"frames left bare: {len(left)}" + (f", by {bare_by}" if left else " (none named; a head turned away is not found by this tool)"))
    say("read back: white a frame %d to %d; fit at no slip %s, one frame early %s, one late %s; white on a bare frame %d" % (
        white.min(), white.max(), *(("n/a" if np.isnan(lags[k]) else f"{lags[k]:.3f}") for k in (0, -1, 1)), on_bare))
    if moves is not None:
        say("the body model against the source, change a frame: fit at no slip %s, one frame early %s, one late %s%s" % (
            *(("n/a" if np.isnan(moves[k]) else f"{moves[k]:.3f}") for k in (0, -1, 1)), "  <- A NEIGHBOUR FITS BETTER" if off_clock else ""))
    say(("PASSED" if passed else "FAILED") + ": the video " + ("holds" if back["passed"] else "does NOT hold") + " the marks its points describe on the frames they describe"
        + ("; the eyes never change in this shot, so a slip of a frame could not be told" if no_change and passed else ""))
    say(f"wrote {out} and {out.with_suffix('.json').name}" + (f" and {record['sheet']['file']}: look at it before the video is used" if source else "; no sheet (give --source)"))
    record["passed"] = passed
    return record


def verify(video: str, mesh_path: str, points_path: str, first: int, count: int, *, names: list[str] | None = None,
           mesh_first: int | None = None, points_first: int | None = None, video_first: int | None = None, grow: float = 1.0, say=print) -> bool:
    """A marked video somebody made, read against a points file on the READER's account of each one's first frame."""
    points = fp.load(points_path, names)
    pf = points["first"] if points_first is None else points_first
    if pf is None:
        raise SystemExit(f"{points_path} does not say which source frame its first row is: give --points-first")
    off = first - pf
    if off < 0 or off + count > len(points["keypoints"]):
        raise SystemExit(f"{points_path} covers source frames {pf} to {pf + len(points['keypoints']) - 1}; frames {first} to {first + count - 1} were asked for")
    idx = points["index"]
    w, h, _ = probe(mesh_path)
    mesh = np.stack(list(stream(mesh_path, (w, h), first - (first if mesh_first is None else mesh_first), count, pix="rgb24")))
    got = np.stack(list(stream(video, (w, h), first - (first if video_first is None else video_first), count, pix="rgb24")))
    if len(mesh) != count or len(got) != count:
        raise SystemExit(f"asked for {count} frames; the body-model video gave {len(mesh)} and the marked video {len(got)}")
    white = ((got.min(-1) > WHITE) & ~(mesh.min(-1) > WHITE)).sum((1, 2)).astype(float)
    area = np.array([white_expected(fp.lid_rings(p, idx), grow) for p in points["keypoints"][off:off + count]])
    drawn = white > 0
    lags = fit_by_lag(white, area, drawn)
    bare = [first + int(f) for f in np.nonzero(~drawn)[0]]
    ok = bool(drawn.any() and not slipped(lags))
    say(f"frames {first} to {first + count - 1}: white a frame {int(white.min())} to {int(white.max())}; frames with no marks: {len(bare)} {bare[:40]}")
    say("white against the points' area: fit at no slip %s, one frame early %s, one late %s" % tuple(("n/a" if np.isnan(lags[k]) else f"{lags[k]:.3f}") for k in (0, -1, 1)))
    say("PASSED: no neighbouring frame's points fit the marks better" + ("; the eyes never change here, so a slip could not be told" if np.isnan(lags[0]) else "")
        if ok else "FAILED: " + ("the video has no marks" if not drawn.any() else "a neighbouring frame's points fit the marks better than the frame's own"))
    return ok


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        a = argparse.ArgumentParser(prog="draw_lid_marks.py verify", description="Read a marked video against a points file.")
        a.add_argument("--video", required=True); a.add_argument("--mesh", required=True); a.add_argument("--points", required=True)
        a.add_argument("--first", type=int, required=True); a.add_argument("--count", type=int, required=True)
        a.add_argument("--names"); a.add_argument("--mesh-first", type=int); a.add_argument("--points-first", type=int); a.add_argument("--video-first", type=int)
        a.add_argument("--grow", type=float, default=1.0, help="the factor the marks were grown by, from the video's record; 1 when not known")
        v = a.parse_args(sys.argv[2:])
        names = json.loads(Path(v.names).read_text()) if v.names else None
        return 0 if verify(v.video, v.mesh, v.points, v.first, v.count, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
                           video_first=v.video_first, grow=v.grow) else 1
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    a.add_argument("--mesh", required=True, help="the body-model video, at the canvas's size")
    a.add_argument("--points", required=True, help="the original's points along the shot (bench/pose_along_shot.py)")
    a.add_argument("--first", type=int, required=True, help="the shot's first source frame")
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--out", required=True, help="the motion video to write; its record and sheet go beside it")
    a.add_argument("--by", required=True, help="who is making it")
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--mesh-first", type=int, help="the source frame of the body-model video's frame 0; the shot's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    a.add_argument("--grow", type=float, help="set the factor by hand in place of the size rule")
    a.add_argument("--bare", default="", help="source frames to leave with no marks, as a person saw them: 1533-1540,1560")
    a.add_argument("--bare-by", default="", help="who looked")
    a.add_argument("--source", default="", help="the original video: writes a sheet to look at")
    a.add_argument("--source-first", type=int, default=0, help="the source frame of that video's frame 0, for one cut from the original; 0 when left out")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.mesh, v.points, v.first, v.count, Path(v.out), v.by, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
              grow=v.grow, bare=frames_arg(v.bare), bare_by=v.bare_by, source=v.source, source_first=v.source_first)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
