#!/usr/bin/env python3
"""`bench/read_render_against_original.py` against faces whose every figure is chosen here.

The tool's whole use is to say whether a render's eyes and head go with its original's, and on 2026-10-11 three
readings of one render disagreed until each said what it measured. So this builds an original and renders out
of points whose opening and place a frame are numbers picked in this file
(`bench/_lib/face_points_fixture.py`), and holds what the tool PRINTS AND WRITES to them. The eye figure it is
held to is worked out here by hand from the fixture's geometry, not by the tool's code.

1. **Known openings and a known shift in, the same numbers out.** A render whose eyes are a chosen amount more
   open on the shut stretch and whose head sits a chosen number of pixels off, more on one stretch, reads as
   exactly that: the two stretch means, the fall and its share of the original's, the distance at each head
   point over the shot, and the worst stretch with its frames.
2. **A render at one level follows nothing, and its mean hides it.** A render whose eyes never move, set at the
   original's own mean, has the original's mean and none of its fall.
3. **No floor, no verdict.** With no floor the tool says so and marks no figure; nothing it returns says past
   or within. With two makes of one render it gives their distance, marks a render past it and one inside it,
   and says what a floor does not bound.
4. **Where the stretches come from is said.** Given by a named person they are used as given; with none, the
   thirds of the ORIGINAL's own figure, and the output says the tool chose them. Half a pair, no name, or a
   frame in both is refused.
5. **The mouth is never read.** The lip gap is a column of the table; the printed reading has one line about
   her mouth and it says the gap is not a measure of it, however far a render's lips are apart.
6. **This clip against the starting lines.** The counts of the original's frames under, over and between
   `h3_config.POSE_READING`'s two lines are the ones chosen here, and lash lines closer than
   `lid_gap_px_to_draw` are said to be under it.
7. **What it refuses, and what it writes.** Files of different lengths or canvases; no first frame. The table
   has one row a frame with the stretch each frame is in, and the reading is written as data.

No model, no card, no server, no ffmpeg.

    <comfy venv python> bench/check_read_render_against_original.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish  # noqa: E402
from _lib import face_points_fixture as fx  # noqa: E402

import read_render_against_original as rr  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

W, H, N, FIRST, FACE = 320, 240, 90, 200, 60.0
CENTRE = np.array([160.0, 100.0])
WORK = Path(tempfile.mkdtemp(prefix="check_read_render_"))

#: The original's opening a frame, pixels: open on its first thirty frames, shut after, with a little wobble so no
#: two frames tie. A person "looked" and gives frames 200-229 as open and 240-289 as shut.
OPEN_PX = np.concatenate([10 + 0.02 * np.arange(30), np.linspace(9.9, 2.6, 10), 2 + 0.01 * np.arange(50)])
OPEN_F, SHUT_F = list(range(200, 230)), list(range(240, 290))
#: A render that opens as far on the open stretch and stays 3 px more open on the shut one.
RENDER_PX = OPEN_PX + np.concatenate([np.zeros(30), np.linspace(0.3, 2.7, 10), np.full(50, 3.0)])
#: Its head: 5 px off everywhere (3 across, 4 up), 10 px on frames 240 to 269.
SHIFT = np.tile([3.0, -4.0], (N, 1))
SHIFT[40:70] = (6.0, -8.0)


def eyes(opening_px) -> np.ndarray:
    """The tool's eye figure for the fixture's eye, by hand: of an eye's five upper points the two ends touch the lower
    line and the other three stand sin(pi t) of (opening + sag) above the point under them, which is the nearest for
    openings this small; their mean over the face's height."""
    reach = np.sin(np.pi * np.array(fx.T)).sum() / len(fx.T)
    return reach * (np.asarray(opening_px, float) + fx.SAG) / FACE


def save(name: str, opening, centre=CENTRE, first: int | None = FIRST, size=(W, H), lips: float = 2.0, face_px: float = FACE) -> str:
    return str(fx.save(WORK / f"{name}_points.npz", fx.face(opening, centre, face_px=face_px, lips=lips), size, first=first))


ORIGINAL = save("original", OPEN_PX)
RENDER = save("render", RENDER_PX, CENTRE + SHIFT)


def run(renders: dict, **kw):
    lines: list[str] = []
    res = rr.read(ORIGINAL, renders, say=lines.append, **kw)
    return res, lines


def given() -> dict:
    return dict(open_frames=OPEN_F, shut_frames=SHUT_F, stretches_by="a person")


def known_in_known_out() -> str:
    res, _ = run({"render": RENDER}, **given())
    r = res["renders"]["render"]
    o_open, o_shut = eyes(OPEN_PX[:30]).mean(), eyes(OPEN_PX[40:]).mean()
    r_open, r_shut = eyes(RENDER_PX[:30]).mean(), eyes(RENDER_PX[40:]).mean()
    e = r["eyes"]
    for got, want, what in ((e["open"], r_open, "open"), (e["shut"], r_shut, "shut"), (e["fall"], r_open - r_shut, "fall"),
                            (e["share_of_the_originals_fall"], (r_open - r_shut) / (o_open - o_shut), "share"),
                            (e["on_the_shut_stretch"], r_shut - o_shut, "shut against the original"),
                            (e["from_the_original_on_the_open_stretch"], 0.0, "open against the original")):
        assert abs(got - want) < 2e-4, (what, got, want)
    head = r["head_px_from_the_original"]
    mean = (5.0 * 60 + 10.0 * 30) / 90
    for k in ("nose", "glabella", "chin"):
        assert abs(head[k]["over_the_shot"] - mean) < 0.02, (k, head[k])
        assert abs(head[k]["worst_stretch"] - 10.0) < 0.02 and head[k]["its_frames"] == [240, 269], (k, head[k])
    assert abs(head["eyes"]["over_the_shot"] - mean) < 1.5                       # the eyes' middle also moves with the lids
    assert res["frames"] == [200, 289] and res["stretches"]["open"] == OPEN_F and res["stretches"]["shut"] == SHUT_F
    return (f"eyes open {e['open']:.4f} and shut {e['shut']:.4f} as chosen, {e['share_of_the_originals_fall']:.2f} of the original's fall; "
            f"head {mean:.2f} px off over the shot and 10.00 on frames 240 to 269")


def one_level_follows_nothing() -> str:
    flat = save("flat", np.full(N, OPEN_PX.mean()))
    res, _ = run({"flat": flat}, **given())
    e = res["renders"]["flat"]["eyes"]
    whole = eyes(OPEN_PX).mean()
    assert abs((e["open"] + e["shut"]) / 2 - whole) < 1e-4 and abs(e["fall"]) < 1e-6 and abs(e["share_of_the_originals_fall"]) < 1e-3, e
    assert e["from_the_original_on_the_open_stretch"] < -0.01 and e["on_the_shut_stretch"] > 0.01, e
    return "a render held at the original's mean opening has none of its fall, and is under it where it is open and over it where it is shut"


def no_floor_no_verdict() -> str:
    res, lines = run({"render": RENDER}, **given())
    text = "\n".join(lines)
    assert res["floor"] is None and "NO FLOOR GIVEN" in text and "past the floor" not in text and "within the floor" not in text, text
    assert res["renders"]["render"]["eyes"]["past_the_floor"] is None
    assert all(v["past_the_floor"] is None for v in res["renders"]["render"]["head_px_from_the_original"].values())
    a = save("make_a", OPEN_PX + 0.10, CENTRE + (0.6, 0.0))                       # two makes: a tenth of a pixel of lid and 0.6 px of head apart
    near = save("near", OPEN_PX + 0.05, CENTRE + (0.3, 0.0))                      # inside that
    res, lines = run({"render": RENDER, "near": near}, floor=(ORIGINAL, a), **given())
    text = "\n".join(lines)
    want = eyes(0.10) - eyes(0.0)
    assert abs(res["floor"]["eyes"] - want) < 1e-5 and abs(res["floor"]["nose_px"] - 0.6) < 1e-3, res["floor"]
    assert "not a second seed" in text and "second seed" in res["floor"]["what"]
    far, close = res["renders"]["render"], res["renders"]["near"]
    assert far["eyes"]["past_the_floor"] == {"open": False, "shut": True}, far["eyes"]["past_the_floor"]
    assert close["eyes"]["past_the_floor"] == {"open": False, "shut": False}
    assert far["head_px_from_the_original"]["nose"]["past_the_floor"] is True and close["head_px_from_the_original"]["nose"]["past_the_floor"] is False
    assert "shut stretch" in text and "past the floor" in text and "within the floor" in text
    assert "floor, first make" not in res["renders"] and len(res["renders"]) == 2   # the two makes are the floor, not renders read
    return f"with none, nothing is marked; with two makes {want:.4f} and 0.6 px apart, the render 3 px more open on the shut stretch is past it and one inside it is within"


def where_the_stretches_come_from() -> str:
    res, lines = run({"render": RENDER})
    s = res["stretches"]
    assert "the tool" in s["by"] and "ORIGINAL" in s["how"] and len(s["open"]) == 30 and len(s["shut"]) == 30, s
    assert s["open"] == list(range(200, 230)) and s["shut"] == list(range(240, 270)), s          # the thirds of the original's own figure
    assert any("the tool" in line for line in lines[:1]), lines[:1]
    res, _ = run({"render": RENDER}, **given())
    assert "a person" in res["stretches"]["by"] and res["stretches"]["how"] == "given"
    for kw, word in ((dict(open_frames=OPEN_F), "come together"), (dict(open_frames=OPEN_F, shut_frames=SHUT_F), "--stretches-by"),
                     (dict(open_frames=OPEN_F, shut_frames=[229, 250], stretches_by="x"), "in both"),
                     (dict(open_frames=[5], shut_frames=SHUT_F, stretches_by="x"), "names no frame")):
        try:
            run({"render": RENDER}, **kw)
        except SystemExit as exc:
            assert word in str(exc), (kw, exc)
        else:
            raise AssertionError(f"accepted: {kw}")
    return "given stretches are used and named; with none the tool takes the thirds of the original's own figure and says so; half a pair, no name, an overlap and frames outside the shot are refused"


def the_mouth_is_never_read() -> str:
    gaping = save("gaping", RENDER_PX, CENTRE + SHIFT, lips=25.0)
    res, lines = run({"gaping": gaping}, out=WORK / "gaping_reading", **given())
    about = [line for line in lines if re.search(r"\b(mouth|lips?)\b", line.lower())]
    assert len(about) == 1 and "not a measure of an open or a shut mouth" in about[0], about
    assert "not read" in res["renders"]["gaping"]["lip_gap"]
    rows = list(csv.reader(open(WORK / "gaping_reading.csv")))
    col = rows[0].index("gaping:lip_gap")
    assert abs(float(rows[1][col]) - 25.0 / FACE) < 1e-3, rows[1][col]            # the gap is in the table, as a number
    assert not any(w in " ".join(lines).lower() for w in ("mouth is open", "mouth open", "mouth is shut", "sings"))
    return "a render with its lips 25 px apart is given one line, that the gap is not a measure of her mouth; the gap is a column of the table"


def this_clip_against_the_lines() -> str:
    band = h3_config.POSE_READING
    res, _ = run({"render": RENDER}, **given())
    e = eyes(OPEN_PX)
    c = res["this_clip"]
    assert c["original_frames_under_the_shut_line"] == int((e < band["eyes_shut_under"]).sum()), c
    assert c["over_the_open_line"] == int((e > band["eyes_open_over"]).sum()) and c["over_the_open_line"] > 0 and c["original_frames_under_the_shut_line"] == 0, c
    assert c["between_the_two_where_the_figure_was_not_readable"] == N - c["over_the_open_line"], c
    assert c["lid_gap_under_the_line_to_draw_from"] is False and abs(c["face_height_px_median"] - FACE) < 0.1
    small = save("small", OPEN_PX / 20, face_px=14.0)                             # a face 14 px tall with lash lines about a pixel apart
    lines2: list[str] = []
    res2 = rr.read(small, {"small render": save("small_render", RENDER_PX / 20, face_px=14.0)}, say=lines2.append, **given())
    assert res2["this_clip"]["lid_gap_under_the_line_to_draw_from"] is True and any("UNDER" in line for line in lines2), res2["this_clip"]
    assert any("does not say lids were seen" in line for line in lines2)
    return (f"{c['over_the_open_line']} of the original's frames are over the open line and {c['between_the_two_where_the_figure_was_not_readable']} between the two, "
            "as chosen; a face 14 px tall is said to be under the gap a mark can be drawn from")


def refusals_and_what_is_written() -> str:
    for what, renders, kw, word in (
            ("another length", {"short": save("short", RENDER_PX[:50])}, given(), "not one shot"),
            ("another canvas", {"big": save("big", RENDER_PX, size=(W * 2, H * 2))}, given(), "not on one canvas")):
        try:
            run(renders, **kw)
        except SystemExit as exc:
            assert word in str(exc), (what, exc)
        else:
            raise AssertionError(f"{what}: accepted")
    try:
        rr.read(save("nofirst", OPEN_PX, first=None), {"render": RENDER}, say=lambda *_: None)
    except SystemExit as exc:
        assert "--first" in str(exc), exc
    else:
        raise AssertionError("an original that does not say its first frame was accepted")
    res, lines = run({"render": RENDER}, out=WORK / "reading", **given())
    rows = list(csv.reader(open(WORK / "reading.csv")))
    head, body = rows[0], rows[1:]
    assert len(body) == N and [int(r[0]) for r in body] == list(range(FIRST, FIRST + N))
    assert [r[1] for r in body] == ["open"] * 30 + [""] * 10 + ["shut"] * 50, [r[1] for r in body][28:42]
    c = head.index("render:eyes")
    assert max(abs(float(r[c]) - w) for r, w in zip(body, eyes(RENDER_PX))) < 1e-4
    saved = json.loads((WORK / "reading.json").read_text())
    assert saved["renders"]["render"]["eyes"] == res["renders"]["render"]["eyes"] and saved["floor"] is None
    assert any(line.startswith("NOT READ") for line in lines)
    return "files of another length or canvas and an original with no first frame are refused; the table has a row a frame with its stretch, and the reading is written as it was printed"


case("known openings and a known shift in, the same numbers out", known_in_known_out)
case("a render at one level follows nothing", one_level_follows_nothing)
case("no floor, no verdict", no_floor_no_verdict)
case("where the stretches come from is said", where_the_stretches_come_from)
case("the mouth is never read", the_mouth_is_never_read)
case("this clip against the starting lines", this_clip_against_the_lines)
case("what it refuses, and what it writes", refusals_and_what_is_written)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
