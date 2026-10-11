#!/usr/bin/env python3
"""Set a render's face beside its original's, frame by frame, from the pose model's points of each.

    python bench/read_render_against_original.py --original ORIGINAL_points.npz --render name=RENDER_points.npz \\
        [--render other=OTHER_points.npz ...] [--floor MAKE_A_points.npz MAKE_B_points.npz] \\
        [--open 1593-1604 --shut 1608-1699 --stretches-by <who looked>] [--out data/<job>/reading] [--names names.json]

The question a masked render of a face is read for is whether it moves with the original: are her eyes open
where the original's are and shut where they are shut, and is her head where the original's is. This prints
that, from points files made by `bench/pose_along_shot.py` on the original and on each render, on one canvas,
with the figures of `bench/_lib/face_points.py`. With `--out` it also writes one row a frame (`.csv`) and what
is printed (`.json`).

WHAT IT GIVES
- **The eye opening taken apart by where the original's eyes are open and where they are shut**, and the fall
  from the one to the other as a share of the original's. A mean over a whole shot hides everything this is
  for: a render whose eyes sit at one level can match the original's mean and follow nothing.
- **Where the head sits, in canvas pixels from the original's**, at three points that are not on the eyes and
  at the middle of the eyes, over the shot and on its worst stretch. Pixels and not a correlation: on
  2026-10-11 a head a few pixels off read as a perfect correlation.
- **The lip gap as a column, never as a mouth.** It is written to the table with the lips' score and nothing
  is said of it: it was judged and failed (`face_points.lip_gap`). A mouth is read by eye.

WHERE THE STRETCHES COME FROM. Best from a person who looked at the original: `--open` and `--shut` as source
frames, with `--stretches-by`. Without them the tool takes the third of the frames where the ORIGINAL's own
figure is highest and the third where it is lowest, and says that it did. Either way it prints how this
clip's figures sit against `h3_config.POSE_READING`, whose two lines were read off one close-up and are a
starting value: it counts the original's frames under, over and between them and says what the lines are. On a
small or a blurred face most frames can fall between them.

IT MEASURES; IT DOES NOT JUDGE. Like a tape measure, it says how far apart two things are and what they were
measured on. A figure near the original's is not good and one far from it is not bad: which is wanted depends
on the shot and on what the render was for, and the scales of these figures are not known yet. The words are
the reader's.

WHAT IT REFUSES TO SAY. A level without a floor. Two makes of ONE render at one seed (`--floor`) show how far
these figures move when nothing was changed; without them every difference is printed and none is called a
difference. With them, a figure is marked only when it is past the floor, and the floor is said for what it
is: it bounds two makes of one render and nothing about a second seed.

WHAT IT CANNOT KNOW
- Whether the points are right. On a face a few pixels tall, or a head turned away, the model still places
  them and scores them well: the tool prints the face's height and the lid gap in pixels beside the scores,
  and says when the gap is under `POSE_READING["lid_gap_px_to_draw"]`.
- Whether a render is good. It reads lids and a head's place. Likeness, a mouth, light and everything that
  only shows in motion are for a person on playback.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402
from _lib import face_points as fp  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

#: What the two lines of `h3_config.POSE_READING` for the eye opening are, said beside every count against them.
LINES_NOTE = ("On the one close-up those two lines were read off, a person looking at every frame agreed with the figure outside them and could not tell narrowed eyes from open ones between them; they are a starting value and not this clip's")
#: What the line for the lid gap is.
GAP_NOTE = "a starting value that lies between the median of the one shot whose marks carried and those of shots whose lash lines are under a pixel apart"
#: A stretch of this many frames is what "its worst stretch" means for the head. Reasoned: about a second and a
#: quarter at the lane's rate, long enough not to be one bad frame.
STRETCH = 30
HEAD = ("nose", "glabella", "chin", "eyes")


def frames_arg(text: str) -> list[int]:
    """`1593-1604,1700-1703` as source frame numbers."""
    out: list[int] = []
    for part in filter(None, (t.strip() for t in text.split(","))):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def columns(rows: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.array([np.nan if r[k] is None else r[k] for r in rows], float) for k in rows[0]}


def stretches(original: dict[str, np.ndarray], first: int, open_frames: list[int], shut_frames: list[int], by: str) -> dict:
    """Which rows count as the original's eyes open and shut, and who or what said so."""
    n = len(original["eyes"])
    if open_frames or shut_frames:
        if not (open_frames and shut_frames and by):
            raise SystemExit("--open and --shut come together, from a person who looked at the original: name them with --stretches-by")
        rows = {k: np.array([f - first for f in v if first <= f < first + n], int) for k, v in (("open", open_frames), ("shut", shut_frames))}
        if not len(rows["open"]) or not len(rows["shut"]):
            raise SystemExit(f"--open or --shut names no frame of this shot (source frames {first} to {first + n - 1})")
        if set(rows["open"]) & set(rows["shut"]):
            raise SystemExit("a frame is in both --open and --shut")
        return dict(rows, by=f"{by}, who looked at the original", how="given")
    order = np.argsort(original["eyes"], kind="stable")
    third = max(n // 3, 1)
    return {"open": np.sort(order[-third:]), "shut": np.sort(order[:third]),
            "by": "the tool: nobody said where the original's eyes are open", "how": "the third of the frames where the ORIGINAL's own figure is highest, and the third where it is lowest"}


def head_px(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The distance a frame between two tables' head points, in pixels."""
    return {k: np.hypot(a[f"{k}_x"] - b[f"{k}_x"], a[f"{k}_y"] - b[f"{k}_y"]) for k in HEAD}


def worst(d: np.ndarray, first: int) -> tuple[float, int, int]:
    """The mean over the worst stretch of `STRETCH` frames, and that stretch as source frames."""
    if len(d) <= STRETCH:
        return float(d.mean()), first, first + len(d) - 1
    run = np.convolve(d, np.ones(STRETCH) / STRETCH, mode="valid")
    i = int(run.argmax())
    return float(run[i]), first + i, first + i + STRETCH - 1


def eyes_of(t: dict[str, np.ndarray], s: dict) -> dict:
    return {"open": float(t["eyes"][s["open"]].mean()), "shut": float(t["eyes"][s["shut"]].mean())}


def read(original_path: str, renders: dict[str, str], *, floor: tuple[str, str] | None = None, names: list[str] | None = None,
         first: int | None = None, open_frames: list[int] | None = None, shut_frames: list[int] | None = None, stretches_by: str = "",
         out: Path | None = None, say=print) -> dict:
    """Read each render against the original. Returns what was printed, as data."""
    src = fp.load(original_path, names)
    first = src["first"] if first is None else first
    if first is None:
        raise SystemExit(f"{original_path} does not say which source frame its first row is: give --first")
    o = columns(fp.table(src, first))
    n = len(o["eyes"])
    tabs = {}
    for name, path in {**renders, **({"floor, first make": floor[0], "floor, second make": floor[1]} if floor else {})}.items():
        p = fp.load(path, names)
        if len(p["keypoints"]) != n:
            raise SystemExit(f"{path} has {len(p['keypoints'])} frames and the original {n}: they are not one shot")
        if src["size"] and p["size"] and tuple(p["size"]) != tuple(src["size"]):
            raise SystemExit(f"{path} was made on a picture of {p['size']} and the original on {src['size']}: they are not on one canvas")
        tabs[name] = columns(fp.table(p, first))
    s = stretches(o, first, open_frames or [], shut_frames or [], stretches_by)
    band = h3_config.POSE_READING
    eo = eyes_of(o, s)
    fall = eo["open"] - eo["shut"]
    res: dict = {"original": str(original_path), "frames": [first, first + n - 1], "stretches": {"open": [first + int(i) for i in s["open"]],
                 "shut": [first + int(i) for i in s["shut"]], "by": s["by"], "how": s["how"]}, "renders": {}}

    say(f"source frames {first} to {first + n - 1}; the original's eyes OPEN on {len(s['open'])} frames and SHUT on {len(s['shut'])}: {s['how']} ({s['by']})")
    # this clip against the starting lines
    under, over = int((o["eyes"] < band["eyes_shut_under"]).sum()), int((o["eyes"] > band["eyes_open_over"]).sum())
    face, gap = float(np.median(o["face_height_px"])), float(np.median(o["lid_gap_px"]))
    res["this_clip"] = {"face_height_px_median": round(face, 1), "lid_gap_px_median": round(gap, 2), "original_frames_under_the_shut_line": under,
                        "over_the_open_line": over, "between_the_two_lines": n - under - over, "the_two_lines_are": LINES_NOTE,
                        "lid_gap_median_under_its_line": bool(gap < band["lid_gap_px_to_draw"]), "that_line_is": GAP_NOTE,
                        "lid_score_median": round(float(np.median(o["score_lids"])), 3)}
    say(f"this clip: her face is {face:.0f} px tall and her lash lines {gap:.2f} px apart at the median"
        + f" (the line kept for that gap is {band['lid_gap_px_to_draw']} px, {GAP_NOTE}; this clip's median is {'under' if gap < band['lid_gap_px_to_draw'] else 'over'} it)"
        + f"; lid score {np.median(o['score_lids']):.2f}, which does not say lids were seen")
    say(f"   against the starting lines (one close-up's): {under} of the original's {n} frames are under the shut line ({band['eyes_shut_under']}), "
        f"{over} over the open line ({band['eyes_open_over']}), {n - under - over} between them. {LINES_NOTE}.")
    if fall <= 0:
        say(f"   in the stretches as given the original's open stretch reads no higher than its shut one ({eo['open']:.4f} against {eo['shut']:.4f}), so the fall below is of nothing")

    # the floor
    fl = None
    if floor:
        a, b = tabs.pop("floor, first make"), tabs.pop("floor, second make")
        ea, eb = eyes_of(a, s), eyes_of(b, s)
        hp = head_px(a, b)
        fl = {"eyes": max(abs(ea["open"] - eb["open"]), abs(ea["shut"] - eb["shut"])), "eyes_a_frame": float(np.abs(a["eyes"] - b["eyes"]).mean()),
              **{f"{k}_px": float(hp[k].mean()) for k in HEAD}}
        res["floor"] = {"what": "two makes of one render at one seed: how far the figures move when nothing was changed. It bounds nothing about a second seed.",
                        **{k: round(v, 5) for k, v in fl.items()}}
        say(f"THE FLOOR, two makes of one render: eye opening on a stretch {fl['eyes']:.4f}, a frame {fl['eyes_a_frame']:.4f}; "
            f"nose {fl['nose_px']:.2f} px. It bounds two makes of one render, not a second seed.")
    else:
        res["floor"] = None
        say("NO FLOOR GIVEN: every figure below is printed and none is called a difference. Two makes of one render (--floor) are what would allow it.")
    mark = lambda v, f: ("" if fl is None else ("  past the floor" if abs(v) > f else "  within the floor"))  # noqa: E731

    say("")
    say(f"EYE OPENING (lash-line gap over the face's height)   {'open':>9}{'shut':>9}{'fall':>9}{'of the original fall':>22}{'from the original, a frame':>28}")
    say(f"   {'the original':32}{eo['open']:9.4f}{eo['shut']:9.4f}{fall:+9.4f}{'1.00':>22}")
    for name, t in tabs.items():
        e = eyes_of(t, s)
        f = e["open"] - e["shut"]
        share = f / fall if fall > 0 else float("nan")
        apart = float(np.abs(t["eyes"] - o["eyes"]).mean())
        d_open, d_shut = e["open"] - eo["open"], e["shut"] - eo["shut"]
        hp = head_px(t, o)
        w = {k: worst(hp[k], first) for k in HEAD}
        res["renders"][name] = {
            "eyes": {"open": round(e["open"], 5), "shut": round(e["shut"], 5), "fall": round(f, 5), "share_of_the_originals_fall": None if np.isnan(share) else round(share, 3),
                     "from_the_original_on_the_open_stretch": round(d_open, 5), "on_the_shut_stretch": round(d_shut, 5), "a_frame": round(apart, 5),
                     "past_the_floor": None if fl is None else {"open": bool(abs(d_open) > fl["eyes"]), "shut": bool(abs(d_shut) > fl["eyes"])}},
            "head_px_from_the_original": {k: {"over_the_shot": round(float(hp[k].mean()), 2), "worst_stretch": round(w[k][0], 2), "its_frames": [w[k][1], w[k][2]],
                                              "past_the_floor": None if fl is None else bool(hp[k].mean() > fl[f"{k}_px"])} for k in HEAD},
            "face_height_px_median": round(float(np.median(t["face_height_px"])), 1),
            "lip_gap": "written to the table and not read: it is not a measure of an open or a shut mouth"}
        say(f"   {name:32}{e['open']:9.4f}{e['shut']:9.4f}{f:+9.4f}{share:22.2f}{apart:28.4f}")
        say(f"   {'':6}from the original: open stretch {d_open:+.4f}{mark(d_open, fl['eyes'] if fl else 0)}; shut stretch {d_shut:+.4f}{mark(d_shut, fl['eyes'] if fl else 0)}")
    say("")
    say(f"WHERE HER HEAD SITS, canvas pixels from the original's: over the shot, and on its worst {STRETCH} frames")
    for name in tabs:
        h = res["renders"][name]["head_px_from_the_original"]
        say(f"   {name:32}" + "   ".join(f"{k} {h[k]['over_the_shot']:.1f} ({h[k]['worst_stretch']:.1f} on {h[k]['its_frames'][0]}-{h[k]['its_frames'][1]})" for k in HEAD)
            + ("" if fl is None else ("  nose past the floor" if h["nose"]["past_the_floor"] else "  nose within the floor"))
            + f"   face {res['renders'][name]['face_height_px_median']:.0f} px tall (the original {face:.0f})")
    say("")
    say("HER MOUTH is not read here: the lip gap is in the table with the lips' score, and is not a measure of an open or a shut mouth. Look at the frames.")
    say("NOT READ: likeness, light, anything that only shows in motion. One seed a render unless a floor says otherwise.")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".csv"), "w", newline="") as fh:
            w_ = csv.writer(fh)
            keep = ("eyes", "lid_gap_px", "face_height_px", "lip_gap", "score_lids", "score_lips", "nose_x", "nose_y", "eyes_x", "eyes_y")
            w_.writerow(["source_frame", "stretch"] + [f"{name}:{k}" for name in ["original"] + list(tabs) for k in keep])
            which = {int(i): "open" for i in s["open"]} | {int(i): "shut" for i in s["shut"]}
            for i in range(n):
                w_.writerow([first + i, which.get(i, "")] + [f"{t[k][i]:.5f}" for t in [o] + list(tabs.values()) for k in keep])
        out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
        say(f"wrote {out.with_suffix('.csv')} and {out.with_suffix('.json').name}")
    return res


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    a.add_argument("--original", required=True, help="the original's points along the shot")
    a.add_argument("--render", action="append", required=True, metavar="NAME=POINTS", help="a render's points, with the name to print it under; give it again for more")
    a.add_argument("--floor", nargs=2, metavar=("MAKE_A", "MAKE_B"), help="the points of two makes of one render at one seed")
    a.add_argument("--first", type=int, help="the source frame of every file's row 0; the original's own when it says")
    a.add_argument("--open", default="", help="source frames where a person saw the original's eyes open: 1593-1604")
    a.add_argument("--shut", default="", help="source frames where a person saw them shut")
    a.add_argument("--stretches-by", default="", help="who looked")
    a.add_argument("--names", help="a json list of the points' names, for files made before they were kept")
    a.add_argument("--out", help="write the table and the reading here (a path without an extension)")
    v = a.parse_args()
    renders = dict(r.split("=", 1) for r in v.render)
    names = json.loads(Path(v.names).read_text()) if v.names else None
    read(v.original, renders, floor=tuple(v.floor) if v.floor else None, names=names, first=v.first, open_frames=frames_arg(v.open),
         shut_frames=frames_arg(v.shut), stretches_by=v.stretches_by, out=Path(v.out) if v.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
