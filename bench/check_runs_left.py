#!/usr/bin/env python3
"""`bench/runs_left.py` on a made table and made previews: each case is a way the queue the card reads could be wrong.

  the_queue_and_what_is_laid   runs past --after, the shot crossing it cut to the frames after it, the most frames left first; a
                               KEPT piece and one STANDING WITHOUT A KEEP said with their frames and not counted as left; a run
                               laid whole in its own list; an unsure subject listed and not run, one marked "no" not run at all
  nothing_dropped              a shot no preview covers, a preview with no part video of the subject, a shot with no run at the floor
                               (with a look's NOTHING TO MAKE) and an unknown `present`: each said with why, never dropped
  laid_and_left                frames a piece leaves as the original are left to make; a span under the shortest load said so
                               each run's restore spans cut to it, and what enters the face's box listed from ENTERING_PX up
  the_floor_and_the_preview    the runs cut by --least; a shot's own shot_c_<first> preview first, the next covering one where it lacks
                               the subject's part; a preview on another canvas read on its own
  the_command                  a kept --json output refused before anything is read

    <comfy venv python> bench/check_runs_left.py

Needs torch (the node's zoom, loaded without the server). No ffmpeg, no model, no card.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import propose_runs  # noqa: E402
import runs_left as rl  # noqa: E402

SIZE = (1024, 768)
LEAST_FRAMES = 10


def entry(first, last, subject, present="yes", stands=(), leaves=(), nothing=None):
    return {"first": first, "last": last, "subject": subject, "present": present, "nothing_to_make": nothing,
            "stands": [{"render": r, "frames": [list(f)], "the_owners_last_choice": {"said": "keep this one"} if kept else {"said": "wrong"},
                        "leaves_as_the_original": [list(x) for x in leaves]} for r, f, kept in stands]}


def preview(name, first, last, canvas=SIZE):
    return {"name": name, "first": first, "last": last, "canvas": list(canvas)}


def parts(px_of=lambda subject, f: 1000):
    """A part of `px_of` px a frame, boxed small in the middle of the frame (a zoom the node allows)."""
    def part_of(p, subject, first, last):
        if subject == "fourth" or (subject == "fifth" and p["name"] != "shot_c_2900c"):
            return f"the preview saved 0 part video(s) of {subject} where one is meant"
        px = np.array([px_of(subject, f) for f in range(first, last + 1)])
        hands = np.array([200 if 3010 <= f <= 3012 else 63 if f == 3020 else 0 for f in range(first, last + 1)])
        return {"px": px, "boxes": np.tile([480, 340, 544, 404], (len(px), 1)),
                "entering": {"Eyeglass": np.zeros(len(px), int), "hands": hands} if subject == "lead" else "not known: no class map"}
    return part_of


PLAN = {"subjects": {"lead": {"restore": [{"classes": ["Eyeglass"], "first": 3090, "last": 3150}, {"classes": ["Eyeglass"], "first": 3909, "last": 4052}]}}}


def queue(table, previews, part_of=None, least=480, plan=PLAN):
    return rl.queue(table, previews, part_of or parts(), least, 2049, 1, LEAST_FRAMES, SIZE, propose_runs.node_zoom().zoom_box, plan)


def the_queue_and_what_is_laid() -> str:
    table = {"made": "test", "reading": [
        entry(2000, 2040, "lead"),                                                   # wholly before: not run
        entry(2000, 2100, "lead"),                                                   # crosses 2049: 2050-2100
        entry(3000, 3099, "lead"),                                                   # nothing laid: 100 left
        entry(3100, 3199, "lead", stands=[("lead_3100", (3100, 3149), True)]),       # half kept: 3150-3199 left
        entry(3200, 3229, "second", stands=[("second_3200", (3190, 3240), False)]),  # laid whole, without a keep
        entry(3300, 3340, "third", present="unsure"),
        entry(3400, 3440, "lead", present="no")]}
    q = queue(table, [preview("shot_c_2000b", 2000, 2100), preview("shot_c_3000b", 3000, 3440)])
    order = [(r["run"], len(r["left"])) for r in q["to_make"]]
    # THE CASE THAT GOES RED when a laid frame is counted as left, the order is not the most frames left first, or a shot before
    # --after or one marked "no" is run
    assert order == [([3000, 3099], 100), ([2050, 2100], 51), ([3100, 3199], 50)], order
    half = q["to_make"][2]
    assert half["left"] == list(range(3150, 3200)) and half["laid"] == [{"render": "lead_3100", "frames": [3100, 3149], "kept": True,
                                                                        "leaves_as_the_original": []}], half
    assert [(r["run"], r["laid"][0]["kept"]) for r in q["laid_whole"]] == [([3200, 3229], False)], q["laid_whole"]
    assert [(s["subject"], s["first"]) for s in q["unsure"]] == [("third", 3300)] and not q["not_run"], q
    first = q["to_make"][0]
    # THE CASE THAT GOES RED when a restore span outside the run is printed with it, or one inside is cut wrong; or a frame under
    # ENTERING_PX is listed as entering
    assert first["restore"] == [{"classes": ["Eyeglass"], "frames": [3090, 3099]}] and q["to_make"][1]["restore"] == [], first["restore"]
    assert first["entering"] == {"hands": {"frames": [3010, 3011, 3012], "most_px": 200}}, first["entering"]
    assert q["laid_whole"][0]["entering"] == "not known: no class map", q["laid_whole"][0]["entering"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rl.print_queue(q, "t.json")
    said = out.getvalue()
    assert "LEFT 3150-3199 (50)" in said and "laid: KEPT lead_3100 3100-3149" in said and "STANDING WITHOUT A KEEP second_3200 3200-3229" in said \
        and "THE RECORD IS UNSURE, not run: third 3300-3340" in said and "restore spans inside: Eyeglass 3090-3099" in said \
        and "entering the face's box: hands 3010-3012 (most 200 px)" in said, said
    return "the most frames left first, the crossing shot cut at --after, kept and unkept pieces said and not left, laid whole apart, unsure listed"


def nothing_dropped() -> str:
    look = ["face not seen on 3100-3140 (a reader)"]
    table = {"made": "test", "reading": [entry(3000, 3040, "lead"), entry(5000, 5040, "lead"), entry(3000, 3040, "fourth"),
                                         entry(3100, 3140, "lead", nothing=look), entry(3000, 3040, "second", present="maybe")]}
    low = parts(lambda s, f: 100 if f >= 3100 else 1000)
    q = queue(table, [preview("shot_c_3000b", 3000, 3140)], low)
    why = {(s["subject"], s["first"]): s["why"] for s in q["not_run"]}
    # THE CASE THAT GOES RED when a shot with no preview, no part video, no run at the floor or an unknown `present` is left out silently
    assert why == {("lead", 5000): "no finished preview covers the shot",
                   ("fourth", 3000): "shot_c_3000b: the preview saved 0 part video(s) of fourth where one is meant"}, why
    assert [(s["first"], s["frames_clearing"], s["nothing_to_make"]) for s in q["no_run"]] == [(3100, 0, look)], q["no_run"]
    assert [(s["subject"], s["present"]) for s in q["present_otherwise"]] == [("second", "maybe")], q["present_otherwise"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rl.print_queue(q, "t.json")
    assert "NOTHING TO MAKE by a look: face not seen on 3100-3140" in out.getvalue() and "present 'maybe', not run" in out.getvalue(), out.getvalue()
    return ("a shot no preview covers, a preview with no part of the subject, a shot with no run at the floor (with its look) and an "
            "unknown `present` are each said, never dropped")


def laid_and_left() -> str:
    table = {"made": "test", "reading": [entry(3000, 3059, "lead", stands=[("lead_3000", (3000, 3059), True)], leaves=[(3010, 3019)]),
                                         entry(3100, 3159, "lead", stands=[("lead_3100", (3100, 3153), False)])]}
    q = queue(table, [preview("shot_c_3000b", 3000, 3159)])
    left = {r["run"][0]: r["left"] for r in q["to_make"]}
    # THE CASE THAT GOES RED when frames a piece leaves as the original are counted as laid
    assert left == {3000: list(range(3010, 3020)), 3100: list(range(3154, 3160))}, left
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rl.print_queue(q, "t.json")
    assert "LEFT 3154-3159 (6, under the shortest load) (6)" in out.getvalue() and "(leaves 3010-3019 as the original)" in out.getvalue(), out.getvalue()
    return "frames a piece leaves as the original are left to make; a span left under the shortest load is said so"


def the_floor_and_the_preview() -> str:
    table = {"made": "test", "reading": [entry(3000, 3059, "lead")]}
    low = parts(lambda s, f: 300 if 3020 <= f < 3040 else 1000)
    q = queue(table, [preview("shot_c_3000b", 3000, 3059)], low)
    assert [r["run"] for r in q["to_make"]] == [[3000, 3019], [3040, 3059]], [r["run"] for r in q["to_make"]]
    assert [r["run"] for r in queue(table, [preview("shot_c_3000b", 3000, 3059)], low, least=200)["to_make"]] == [[3000, 3059]]
    own, wide = preview("shot_c_3000b", 3000, 3059), preview("shot_c_2900c", 2900, 3100)
    assert [x["name"] for x in rl.previews_for([3000, 3059], [own, wide])] == ["shot_c_3000b", "shot_c_2900c"]
    # the shot's own preview without the subject's part: the next covering one is tried
    q = queue({"made": "test", "reading": [entry(3000, 3059, "fifth")]}, [own, wide])
    assert [r["preview"] for r in q["to_make"]] == ["shot_c_2900c"] and not q["not_run"], q
    # a preview on another canvas does not stop the rest
    q = queue({"made": "test", "reading": [entry(3000, 3059, "lead"), entry(4000, 4059, "lead")]},
              [preview("shot_c_3000b", 3000, 3059), preview("shot_c_4000b", 4000, 4059, (1344, 768))])
    assert [r["preview"] for r in q["to_make"]] == ["shot_c_3000b", "shot_c_4000b"], q
    return "the runs cut by --least; the shot's own preview first, the next covering one where it lacks the part; each preview on its canvas"


def the_command() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        kept = Path(tmp) / "q.json"
        kept.write_text("{}")
        try:
            rl.main(["--table", str(Path(tmp) / "none.json"), "--plan", str(Path(tmp) / "plan.json"), "--json", str(kept)])
            raise AssertionError("a kept output was not refused")
        except SystemExit as e:
            assert "a kept output is not replaced" in str(e), str(e)
        assert kept.read_text() == "{}"
    return "a kept --json output refused before anything is read"


case("the queue and what is laid", the_queue_and_what_is_laid)
case("nothing dropped", nothing_dropped)
case("laid and left", laid_and_left)
case("the floor and the preview", the_floor_and_the_preview)
case("the command", the_command)
sys.exit(finish())
