#!/usr/bin/env python3
"""`bench/others_present.py` on made previews, a cast file and a shots record: each case is a way the answer to "are there
people in the shot who are not ours" could be wrong.

  ours_and_others              each tracker's own person is ours, the owner's answer by box over the tracker's pick on that table
                               and frame, "none of the outlined people" takes the pick away, another tracker's person is ours;
                               the rest are others, with box, share and the gap to ours
  the_answer_a_shot            yes where a cut's shown frame has an other; no on a short shot with none; cannot say on a shot
                               longer than LONG_SHOT_FRAMES with none, with the frames not shown; no preview
  the_owners_words             a numbered person the owner's words in the shot's note call not a person counts as no one, matched
                               by place to the box it had on the page they answered on, and listed with their words; a box that
                               does not overlap is not carried
  the_videos_not_people        in the shots the plan's not_people names, a lone small other touching none of ours is no one, said
                               with its box; one touching, a large one, two others or another shot stay someone
  the_newest_preview           of a shot's previews, the later letter, then the later day, finished; a run writes over nothing

    <comfy venv python> bench/check_others_present.py

No model, no card, no server.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import others_present as op  # noqa: E402


def table(people, pick, state="seeded", first=0, last=19, shown=5) -> dict:
    """A tracker's shot table of one cut: `people` as (number, x, y, w, h, held by another tracker)."""
    return {"people_are": "every detection on the frame, in one numbering for every tracker of a job", "shots": [
        {"first_frame": first, "last_frame": last, "shown_frame": shown, "subject": {"state": state, "person": pick},
         "people": [{"person": n, "box": [x, y, w, h], "share_of_frame": round(w * h / (100 * 100), 4), "held_by_another_tracker": held}
                    for n, x, y, w, h, held in people]}]}


PEOPLE = [(1, 0, 0, 10, 10, False), (2, 30, 0, 20, 40, False), (3, 60, 0, 20, 40, True), (4, 90, 90, 5, 5, False)]


def preview(tables: dict, first=100, name="shot_c_0100") -> dict:
    return {"name": name, "first": first, "last": first + 19, "tables": {k: (f"{name}_{k}_shots_00001_shots.json", t) for k, t in tables.items()}}


def ours_and_others() -> str:
    pv = preview({"a": table(PEOPLE, 2)})
    got = op.shot_others({"first": 100, "last": 119}, pv, [])
    cut = got["cuts"][0]
    # THE CASE THAT GOES RED when the tracker's own person or another tracker's is said to be someone else
    assert [o["person"] for o in cut["others"]] == [1, 4] and cut["ours"] == {"a": "person 2, the tracker's seeded"}, cut
    assert cut["others"][0]["box"] == [0, 0, 10, 10] and cut["others"][0]["gap_px_to_ours"] == 20.0 and cut["others"][0]["share_of_frame"] == 0.01, cut["others"]
    # the owner's answer on that table and frame stands over the pick; "none of the outlined people" takes the pick away
    answer = {"subject": "a", "table": "shot_c_0100_a_shots_00001_shots.json", "frame": 105, "box": [0, 0, 10, 10], "answer": "here"}
    cut = op.shot_others({"first": 100, "last": 119}, pv, [answer])["cuts"][0]
    assert [o["person"] for o in cut["others"]] == [2, 4] and cut["ours"]["a"] == "person 1, by the owner's answer", cut
    cut = op.shot_others({"first": 100, "last": 119}, pv, [dict(answer, box=None, answer="none of the outlined people")])["cuts"][0]
    assert [o["person"] for o in cut["others"]] == [1, 2, 4] and cut["ours"]["a"] == "the owner: none of the outlined people", cut
    # an answer on another frame or another table is not this one's
    for other in (dict(answer, frame=106), dict(answer, table="shot_c_0100b_a_shots_00001_shots.json")):
        assert [o["person"] for o in op.shot_others({"first": 100, "last": 119}, pv, [other])["cuts"][0]["others"]] == [1, 4]
    cut = op.shot_others({"first": 100, "last": 119}, preview({"a": table(PEOPLE, 2, state="picked")}), [])["cuts"][0]
    assert "PICKED" in cut["ours"]["a"], cut["ours"]
    assert op.gap_px((0, 0, 10, 10), (13, 14, 20, 20)) == 5.0 and op.gap_px((0, 0, 10, 10), (5, 5, 20, 20)) == 0.0
    return ("the tracker's own person and another tracker's are ours; the owner's answer by box on that table and frame over the pick, none taking it "
            "away; an answer elsewhere is not this one's; the others with box, share and gap")


def the_answer_a_shot() -> str:
    alone = [(1, 30, 0, 20, 40, False)]
    assert op.shot_others({"first": 100, "last": 119}, preview({"a": table(PEOPLE, 2)}), [])["others_present"] == "yes"
    assert op.shot_others({"first": 100, "last": 119}, preview({"a": table(alone, 1)}), [])["others_present"] == "no"
    n = op.LONG_SHOT_FRAMES + 1
    long = preview({"a": table(alone, 1, last=n - 1)})
    long["last"] = 100 + n - 1
    got = op.shot_others({"first": 100, "last": 100 + n - 1}, long, [])
    # THE CASE THAT GOES RED when a long shot with nobody else on its one shown frame is said to have nobody
    assert got["others_present"] == "cannot say" and got["frames_not_shown"] == n - 1, got
    assert op.shot_others({"first": 100, "last": 119}, None, [])["others_present"] == "no preview"
    two = {"a": table(alone, 1), "b": table([(1, 30, 0, 20, 40, False)], 1, first=10, last=19, shown=15)}
    two["a"]["shots"][0]["last_frame"] = 9
    got = op.shot_others({"first": 100, "last": 119}, preview(two), [])
    assert got["frames_shown"] == [105, 115] and got["others_present"] == "no", got
    return f"yes with an other on a shown frame; no on a short shot with none; cannot say past {op.LONG_SHOT_FRAMES} frames, with the frames not shown; no preview"


def the_owners_words() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "answered").mkdir()
        (d / "answered" / "shot_c_0100_a_shots_00001_shots.json").write_text(json.dumps(table(PEOPLE, 2)))
        page = {"built": "2026-01-01T10:00:00Z", "previews": [{"job": "shot_c_0100", "frames": [100, 119], "tables": str(d / "answered")}], "page": "p"}
        note = ("a session's words | the owner on the who page (frame 100, answers.jsonl line 7, 2026-01-01T11:00): 1 = a picture of someone on a cereal box, "
                "ignore this\n2 = the lead")
        shot = {"first": 100, "last": 119, "note": note}
        named = op.not_a_person(shot, [page])
        assert named == [{"person": 1, "words": "a picture of someone on a cereal box, ignore this", "line": 7, "page": "p", "preview": "shot_c_0100",
                          "box": (0.0, 0.0, 10.0, 10.0)}], named
        got = op.shot_others(shot, preview({"a": table(PEOPLE, 2)}), [], [page])
        # THE CASE THAT GOES RED when the owner's words that a numbered person is a picture are not read
        assert [o["person"] for o in got["cuts"][0]["others"]] == [4] and got["not_people"][0]["answers_line"] == 7 and got["others_present"] == "yes", got
        moved = [(1, 50, 50, 10, 10, False), (2, 30, 0, 20, 40, False)]
        got = op.shot_others(shot, preview({"a": table(moved, 2)}), [], [page])
        assert [o["person"] for o in got["cuts"][0]["others"]] == [1] and not got["not_people"], "a box in another place was carried"
        assert got["words_not_carried"] == [{"person": 1, "answers_line": 7, "answered_on": "shot_c_0100", "why": "not carried: another frame or place"}], got
        alone = op.shot_others(shot, preview({"a": table([(1, 0, 0, 10, 10, False), (2, 30, 0, 20, 40, False)], 2)}), [], [page])
        assert alone["others_present"] == "no" and alone["not_people"], alone
        # of the pages built before the words, the newest: an older page numbered person 1 elsewhere
        (d / "older").mkdir()
        (d / "older" / "shot_c_0100_a_shots_00001_shots.json").write_text(json.dumps(table([(1, 70, 70, 10, 10, False)], 1)))
        older = dict(page, built="2026-01-01T09:00:00Z", previews=[dict(page["previews"][0], tables=str(d / "older"))])
        assert op.not_a_person(shot, [older, page])[0]["box"] == (0.0, 0.0, 10.0, 10.0), "an older page's numbering was read"
        late = dict(page, built="2026-01-02T00:00:00Z")
        assert op.not_a_person(shot, [late])[0]["box"] is None, "a page built after the words was read"
    return "the owner's words in the note count a numbered person as no one, matched by place to the box on the page they answered on; another place, or a later page, not"


def the_videos_not_people() -> str:
    """The owner's word of 2026-10-14 on the girl printed on the cereal box: in the shots the plan's `not_people` names, a lone
    small other touching none of ours counts as no one, said with its box; anything else there stays someone."""
    rule = [{"what": "THE BOX", "shots": [[100, 119]], "by": "the owner"}]
    ours = [(2, 30, 0, 20, 40, False)]
    def got(people, shot=(100, 119)):
        return op.shot_others({"first": shot[0], "last": shot[1]}, preview({"a": table(people, 2)}), [], None, rule)
    one = got(ours + [(4, 90, 90, 5, 5, False)])
    # THE CASE THAT GOES RED when the box keeps a kitchen shot "yes", or a person touching ours is taken for it
    assert one["others_present"] == "no" and one["not_people"][0]["box"] == [90, 90, 95, 95] and one["not_people"][0]["owners_words"] == "THE BOX", one
    assert got(ours + [(4, 50, 0, 5, 5, False)])["others_present"] == "yes", "a small other touching ours was taken for the box"
    assert got(ours + [(4, 90, 90, 30, 30, False)])["others_present"] == "yes", "a large other was taken for the box"
    assert got(ours + [(4, 90, 90, 5, 5, False), (5, 0, 90, 5, 5, False)])["others_present"] == "yes", "two others were taken for the box"
    far = op.shot_others({"first": 200, "last": 219}, preview({"a": table(ours + [(4, 90, 90, 5, 5, False)], 2)}, first=200), [], None, rule)
    assert far["others_present"] == "yes", "the box's rule reached a shot it does not name"
    return "in a shot the plan names, a lone small other touching none of ours is no one, with its box; one touching, a large one, two, or another shot stay"


def the_newest_preview() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for folder, name, status in (("2026-01-01_shot_c_0100_job", "shot_c_0100", "success"), ("2026-01-02_shot_c_0100b_job", "shot_c_0100b", "success"),
                                     ("2026-01-03_shot_c_0100c_job", "shot_c_0100c", "failed"), ("2026-01-01_shot_c_0200_job", "shot_c_0200", "success")):
            (d / folder).mkdir()
            (d / folder / "job.json").write_text(json.dumps({"job": {"name": name, "output": "Video/v/s", "first": 100, "frames": 20, "subjects": []}}))
            (d / folder / "preview.queued.json").write_text(json.dumps({"status": status}))
        got = op.newest_previews(d, Path("/share"))
        assert sorted(got) == sorted([f"{d / '2026-01-02_shot_c_0100b_job' / 'job.json'}=/share/Video/v/s/shot_c_0100b",
                                      f"{d / '2026-01-01_shot_c_0200_job' / 'job.json'}=/share/Video/v/s/shot_c_0200"]), got
        (d / "out.json").write_text("kept")
        try:
            op.main(["--shots", "s", "--cast", "c", "--json", str(d / "out.json"), "--md", str(d / "out.md")])
            raise AssertionError("a kept output was written over")
        except SystemExit as e:
            assert "a kept output is not replaced" in str(e), e
    return "the later letter, then the later day, of a finished preview; a failed one is not read; a kept output is not replaced"


case("ours and others", ours_and_others)
case("the answer a shot", the_answer_a_shot)
case("the owner's words that a numbered person is not one", the_owners_words)
case("the video's not_people", the_videos_not_people)
case("the newest preview of a shot", the_newest_preview)
sys.exit(finish())
