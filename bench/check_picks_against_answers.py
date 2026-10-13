#!/usr/bin/env python3
"""`bench/picks_against_answers.py` keeps what a tracker picked and what a session read against what the owner clicked; each case is a way a line could say the wrong thing of one of the three.

Why it exists: the owner asked (2026-10-13) that the three are kept as data to be read over time. A line that calls a
right pick wrong, or counts a number against a number of another picture, would be read for months.

Six previews made in a temporary folder, each with real track videos (ffmpeg), on a canvas of 64 by 48:

  a_wrong_pick_is_wrong_both_ways        the tracker took the larger person and the owner named the smaller: no by the track, no
                                         by the numbers; the owner's LAST click is the one read and the earlier one is counted
  another_pictures_number_is_not_compared   a subject whose own table shows another frame than the picture answered on: the
                                         numbers are not compared, the track on the answered frame decides, and the line says so
  not_outlined_and_not_on_frame          `not outlined` with the track on an outlined person is a miss; `not on frame` with the
                                         track off is right
  no_click_is_kept_and_counted_nowhere   a subject the owner did not click for has a line, with no agreement
  the_two_ways_can_differ_and_say_so     a track on the named person at an overlap under the line: no by the track, yes by the
                                         numbers, said on the line, and the person the track lies on most is the named one
  how_the_pick_was_chosen                seeded; the largest on a frame the owner marked; on the shot's middle; on a frame named otherwise
  the_features_are_the_pictures_own      people outlined, the named person's rank, share, band, place and height, and how alike
                                         the named and the held person are, from the table's signatures
  the_sessions_cells                     said right and it was not; said wrong and it was right; said wrong and it was; could not
                                         tell; `thinks_it_is` compared where written and said to be absent where not
  the_owners_words_are_on_the_line       whole, with their line
  an_unsure_answer_is_counted_apart      a shot marked as one the owner's own words were unsure of keeps its lines whole, carries the
                                         words' line numbers and who read them so, and is out of the table's blocks; a mark on no
                                         such shot, or naming a line that is not the owner's words on it, is refused
  a_batch_is_written_once                a dry run writes nothing; a batch is appended; the same name is refused; --follows must
                                         name a batch that is there; a second batch leaves the first as it was
  the_table_is_counts_for_one_batch      four blocks and the session's cells, saying it is one batch and what it cannot see

    <python> bench/check_picks_against_answers.py

No model, no card; numpy and ffmpeg for the track videos.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import case, finish  # noqa: E402
from _lib.frames import write_mask_video  # noqa: E402

import picks_against_answers as pa  # noqa: E402

W, H = 64, 48
SMALL, LARGE = [4, 4, 10, 20], [30, 4, 20, 40]            # two people as a table writes a box: x, y, width, height


def person(n: int, box, sig) -> dict:
    return {"person": n, "box": box, "share_of_frame": round(box[2] * box[3] / (W * H), 4), "detector_score": 0.9, "held_by_another_tracker": False,
            "has_head": n == 1, "signatures": sig}


PEOPLE = [person(1, SMALL, [[1.0, 0.0], [1.0, 0.0]]), person(2, LARGE, [[0.6, 0.8], [0.8, 0.6]])]


def table(frames: int, shown: int, picked, state="picked", seed="", people=None, score=1.5) -> dict:
    return {"version": 2, "size": [W, H], "pick": "largest", "pick_frame": shown, "pick_seed": seed, "cuts": [], "shots": [
        {"shot": 1, "first_frame": 0, "last_frame": frames - 1, "frames": frames, "shown_frame": shown, "people": PEOPLE if people is None else people,
         "subject": {"state": state, "person": picked, "why": "the table's own words", "similarity": None, "also_held_by_another_tracker": False},
         "frames_with_subject": frames - 2, "track_score": [score] * frames}]}


def mask(frames: int, box, on=None) -> np.ndarray:
    m = np.zeros((frames, H, W), bool)
    if box is not None:
        m[:, box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = True
    if on is not None:
        m[[f for f in range(frames) if f not in on]] = False
    return m


class Made:
    def __init__(self, root: Path):
        self.root, self.specs, self.answers, self.notes = root, [], [], []
        self.shots = {"schema": "h3_shots/1", "video": "v.mp4", "span": [0, 699], "subjects": ["a", "b"], "shots": []}

    def preview(self, name: str, first: int, frames: int, subjects: dict, pick_at=None, marked=()) -> None:
        """`subjects` is {label: (the table, the track's mask or None)}."""
        tables, job = self.root / "out" / name, self.root / "jobs" / f"{name}_job"
        tables.mkdir(parents=True)
        job.mkdir(parents=True)
        listed = []
        for label, (t, m) in subjects.items():
            (tables / f"{name}_{label}_shots_00001_shots.json").write_text(json.dumps(t))
            if m is not None:
                write_mask_video(tables / f"{name}_{label}_track_00001.mkv", m)
                listed.append(f"Video/x/{name}/{name}_{label}_track_00001.mkv")
        (job / "job.json").write_text(json.dumps({"job": {"name": name, "first": first, "frames": frames, "subjects": [
            {"label": label, "pick": "largest", **({"pick_at": pick_at} if pick_at is not None else {})} for label in subjects]}}))
        (job / "preview.built.json").write_text(json.dumps({"commit": "abc1234", "built_at": "2026-01-02T10:00:00-05:00", "tracker_mask_version": 14}))
        (job / "preview.queued.json").write_text(json.dumps({"status": "success", "seconds": 21, "files": listed}))
        self.specs.append(f"{job / 'job.json'}={tables}")
        self.shots["shots"].append({"first": first, "last": first + frames - 1, "who": {
            k: {"present": "yes", "how": "", "by": "", "frames_on": list(marked) if k == "a" else [], "frames_boxed": []} for k in ("a", "b")}})

    def click(self, picture: str, frame: int, subject: str, chosen: str, who: str = "owner") -> None:
        self.answers.append({"at": f"2026-01-03T10:00:{len(self.answers):02d}Z", "who": who, "row": "who", "what": "choice", "subject": subject, "chosen": chosen,
                             "table": f"{picture}_shots_00001_shots.json", "source_frame": frame})

    def say(self, frame: int, words: str) -> None:
        self.answers.append({"at": f"2026-01-03T10:00:{len(self.answers):02d}Z", "who": "owner", "row": "who", "what": "words", "words": words, "source_frame": frame})

    def read(self, name: str, first: int, **subjects) -> None:
        self.notes.append({"first": first, "job_file": f"inputs/{name}.json", "recorded_at": "2026-01-02T12:00:00Z",
                           "subjects": [{"label": k, "picked_number": 2, "picked_looks_like": "a tall one", "why": "free text", **v} for k, v in subjects.items()]})

    def files(self) -> dict:
        paths = {k: self.root / f"{k}.json{'l' if k != 'shots' else ''}" for k in ("shots", "answers", "notes")}
        paths["shots"].write_text(json.dumps(self.shots))
        for k in ("answers", "notes"):
            paths[k].write_text("".join(json.dumps(x) + "\n" for x in getattr(self, k)))
        return paths

    def lines(self, unsure=None) -> dict:
        p = self.files()
        answers = [(n, json.loads(t)) for n, t in enumerate(p["answers"].read_text().splitlines(), 1)]
        got = pa.lines_of(self.shots, pa.previews(self.specs), answers, pa.notes_of([str(p["notes"])]), "B1", "a check", "NOW", "c0ffee", pa.seed_agrees(),
                          unsure=unsure)
        return {(x["preview"]["job"], x["subject"]): x for x in got}


def build(root: Path) -> Made:
    m = Made(root)
    # pv: two subjects. a's own picture is frame 110, where the owner answers; b's own picture is frame 120.
    m.preview("pv", 100, 40, {"a": (table(40, 10, 2), mask(40, LARGE)), "b": (table(40, 20, 1), mask(40, LARGE))}, pick_at=119)
    m.click("pv_a", 110, "a", "2")
    m.click("pv_a", 110, "a", "1")                      # the owner's last click for a: the smaller person
    m.click("pv_a", 110, "b", "2")
    m.click("pv_a", 110, "a", "2", who="a session (a test of the page)")
    m.say(100, "1 is small, in a picture on the wall\n2 is her")
    m.read("pv", 100, a={"right_person": "yes"}, b={"right_person": "no"})
    # solo: seeded on person 1; the owner says not outlined
    m.preview("solo", 200, 30, {"a": (table(30, 5, 1, state="taken (seeded)", seed="is the pick"), mask(30, SMALL))})
    m.click("solo_a", 205, "a", "not outlined")
    m.read("solo", 200, a={"right_person": "no"})
    # gone: the owner says not on this frame, and the track is off there (on from frame 10)
    m.preview("gone", 300, 20, {"a": (table(20, 4, 1), mask(20, SMALL, on=range(10, 20)))})
    m.click("gone_a", 304, "a", "not on frame")
    m.read("gone", 300, a={"right_person": "cannot tell"})
    # quiet: no click
    m.preview("quiet", 400, 20, {"a": (table(20, 4, 1), mask(20, SMALL))})
    # near: the pick is the named person by number, and the track covers half of their box
    m.preview("near", 500, 20, {"a": (table(20, 5, 2), mask(20, [30, 4, 20, 20]))}, pick_at=505)
    m.click("near_a", 505, "a", "2")
    # good: the pick is the named person, the track is their box, picked on a frame the owner had marked
    m.preview("good", 600, 20, {"a": (table(20, 10, 2), mask(20, LARGE))}, pick_at=610, marked=[610])
    m.click("good_a", 610, "a", "2")
    m.read("good", 600, a={"right_person": "yes", "thinks_it_is": 2})
    m.say(505, "MAYBE it is 2?")                        # the owner's own doubt on the shot of `near`; the last line of the answers
    return m


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        made = build(Path(tmp))
        got = made.lines()
        ag = lambda key: got[key]["agreement"]["tracker_against_owner"]        # noqa: E731

        def wrong_pick() -> str:
            x, a = got[("pv", "a")], ag(("pv", "a"))
            assert x["owner"]["click"] == "1" and x["owner"]["clicks_before_the_last"] == 1 and x["owner"]["line"] == 2, x["owner"]
            assert a["by_the_track"] is False and a["by_the_numbers"] is False and a["agrees"] is False and "the_two_ways_differ" not in a, a
            assert a["overlap_of_the_track_with_the_named_box"] == 0.0 and a["the_track_lies_most_on"] == {"person": 2, "overlap": 1.0}, a
            assert a["line"] == pa.seed_agrees() and "a starting value" in a["line_is"], a
            return "the larger person taken, the smaller named: wrong by the track and by the numbers; the last of two clicks read, a session's click not at all"

        def another_picture() -> str:
            x, a = got[("pv", "b")], ag(("pv", "b"))
            assert x["tracker"]["person"] == 1 and x["tracker"]["on_frame"] == 120 and x["owner"]["frame"] == 110 and x["owner"]["picture"].startswith("pv_a_"), x["tracker"]
            assert x["session"]["number_is_of_another_picture"] is True and a["by_the_numbers"] is None, (x["session"], a)
            assert a["by_the_track"] is True and a["agrees"] is True and a["overlap_of_the_track_with_the_named_box"] == 1.0, a
            return "b's own table shows frame 120 and the owner answered on frame 110: no number is compared, and the track on 110 decides"

        def not_outlined_and_not_on_frame() -> str:
            s, g = ag(("solo", "a")), ag(("gone", "a"))
            assert s["agrees"] is False and "lies on one of them" in s["how"] and s["by_the_numbers"] is None, s
            assert g["agrees"] is True and g["the_track_on_that_frame"]["on"] is False, g
            return "not outlined, with the track on an outlined person, is a miss; not on this frame, with the track off, is right"

        def no_click() -> str:
            x = got[("quiet", "a")]
            assert x["owner"]["click"] == pa.NO_CLICK and x["agreement"]["tracker_against_owner"]["agrees"] is None and x["features"]["people_outlined"] is None, x["owner"]
            return "a subject with no click has its line and is in no agreement"

        def two_ways() -> str:
            a = ag(("near", "a"))
            assert a["by_the_track"] is False and a["by_the_numbers"] is True and a["agrees"] is False, a
            assert a["overlap_of_the_track_with_the_named_box"] == 0.5 and a["share_of_the_track_inside_the_named_box"] == 1.0 and a["the_track_lies_most_on"]["person"] == 2, a
            assert a["the_two_ways_differ"].startswith("by the track no, by the numbers yes"), a
            return "a track over half the named person's box: no by the track, yes by the numbers, and the line says the two differ"

        def how_chosen() -> str:
            have = {k[0]: v["tracker"]["how_chosen"] for k, v in got.items() if k[1] == "a"}
            assert (have["solo"], have["good"], have["pv"], have["near"], have["quiet"]) == (pa.HOW[0], pa.HOW[1], pa.HOW[2], pa.HOW[3], pa.HOW[4]), have
            return "seeded; a frame the owner marked; the shot's middle; a frame named otherwise; the tracker's own frame"

        def features() -> str:
            f = got[("pv", "a")]["features"]
            n = f["the_named_person"]
            assert f["people_outlined"] == 2 and f["people_with_a_head"] == 1 and n["rank_by_size"] == 2 and n["share_band"] == pa.SHARE_BANDS[2][1], f
            assert n["height_px"] == 20 and n["centre"] == [round(9 / W, 3), round(14 / H, 3)] and n["touches_the_frames_edge"] is False, n
            assert n["share_over_the_largest"] == round(PEOPLE[0]["share_of_frame"] / PEOPLE[1]["share_of_frame"], 4), n
            assert f["the_person_the_track_lies_most_on"] == {"person": 2, "share_of_the_frame": PEOPLE[1]["share_of_frame"], "rank_by_size": 1}, f
            assert f["likeness_of_the_named_to_the_one_the_track_lies_on"] == 0.6 and f["the_tracks_score_on_the_frame"] == 1.5 and f["shot_frames"] == 40, f
            assert f["subjects_in_the_job"] == 2 and got[("pv", "b")]["features"]["place_in_the_trackers_order"] == 2
            assert got[("good", "a")]["features"]["likeness_of_the_named_to_the_one_the_track_lies_on"] is None, "a person was compared with themselves"
            p = got[("pv", "a")]["preview"]
            assert (p["commit"], p["tracker_mask_version"], p["seconds"], p["first"]) == ("abc1234", 14, 21, 100), p
            return "the named person is second by size of two, in the 5-to-20% band; the held and the named are 0.6 alike by the table's signatures"

        def session() -> str:
            cell = lambda key: got[key]["agreement"]["session_against_owner"]        # noqa: E731
            assert cell(("pv", "a"))["cell"] == "said the pick was right, and it was not" and cell(("pv", "b"))["cell"] == "said the pick was wrong, and it was right"
            assert cell(("solo", "a"))["cell"] == "said the pick was wrong, and it was" and cell(("gone", "a"))["cell"] == "could not tell"
            assert cell(("near", "a"))["cell"] == "no reading by a session"
            assert cell(("good", "a"))["cell"] == "said the pick was right, and it was" and cell(("good", "a"))["the_reader_named_the_owners_person"] is True
            assert cell(("pv", "a"))["thinks_it_is"].startswith("not written") and "the_reader_named_the_owners_person" not in cell(("pv", "a"))
            assert got[("pv", "a")]["session"]["picked_looks_like"] == "a tall one" and got[("pv", "a")]["session"]["line"] == 1
            return "the four cells and `could not tell`; a reader's `thinks_it_is` is compared where written and said to be absent where not"

        def words() -> str:
            said = got[("pv", "a")]["owner"]["words_on_the_shot"]
            assert [(w["line"], w["words"]) for w in said] == [(5, "1 is small, in a picture on the wall\n2 is her")] and got[("solo", "a")]["owner"]["words_on_the_shot"] == []
            return "the owner's words on the shot are on each of its lines, whole, and on no other shot's"

        def unsure() -> str:
            last = len(made.answers)
            mark = {(500, 519): {"lines": [last], "read_so_by": "the lead, reading the owner's words"}}
            marked = made.lines(mark)
            x = marked[("near", "a")]
            assert x["owner"]["unsure"] == mark[(500, 519)] and x["owner"]["click"] == "2" and x["agreement"] == got[("near", "a")]["agreement"], x["owner"]
            assert all(v["owner"]["unsure"] is None for k, v in marked.items() if k != ("near", "a")), "a shot nobody marked is marked"
            text = pa.table_text(list(marked.values()), "B1")
            assert "OF THE CLICKS, 1 are on shots where the owner's own words say they were unsure (shots 500-519; read so by the lead, reading the owner's words)" in text, text[:700]
            assert "counted APART, not in the blocks: the tracker right 0, wrong 1, cannot be said 0. The blocks below are of the other 5." in text, text[:900]
            assert pa.HOW[3] not in text.split("ONE BATCH")[1], "the unsure shot's line is in a block"                   # `near` was the only pick on a frame named otherwise
            assert "No shot is marked as one the owner was unsure of: the blocks below are of all 6." in pa.table_text(list(got.values()), "B1")
            for bad, words in (({(500, 518): mark[(500, 519)]}, "is not a shot of the shots file"), ({(500, 519): {"lines": [1], "read_so_by": "x"}}, "are not the owner's typed words on a frame of that shot")):
                try:
                    made.lines(bad)
                except SystemExit as stop:
                    assert words in str(stop), str(stop)
                else:
                    raise AssertionError(f"not refused: {bad}")
            return "an unsure shot's line is whole, carries the words' lines and who read them so, and is counted apart from the blocks"

        def batches() -> str:
            p = made.files()
            out = Path(tmp) / "kept" / "picks.jsonl"
            cmd = [sys.executable, str(HERE / "picks_against_answers.py"), "--shots", str(p["shots"]), "--answers", str(p["answers"]), "--notes", str(p["notes"]),
                   "--by", "a check", "--out", str(out)] + [w for s in made.specs for w in ("--preview", s)]
            dry = subprocess.run(cmd + ["--batch", "day1", "--dry-run"], capture_output=True, text=True)
            assert dry.returncode == 0 and not out.exists() and "DRY: nothing written; 7 line(s)" in dry.stdout, dry.stdout + dry.stderr
            first = subprocess.run(cmd + ["--batch", "day1"], capture_output=True, text=True)
            kept = out.read_text()
            assert first.returncode == 0 and len(kept.splitlines()) == 7 and all(json.loads(t)["batch"] == "day1" for t in kept.splitlines()), first.stdout + first.stderr
            again = subprocess.run(cmd + ["--batch", "day1"], capture_output=True, text=True)
            assert again.returncode != 0 and "already holds a batch named 'day1'" in again.stderr + again.stdout and out.read_text() == kept, again.stderr
            lost = subprocess.run(cmd + ["--batch", "day2", "--follows", "nowhere"], capture_output=True, text=True)
            assert lost.returncode != 0 and "holds no batch of that name" in lost.stderr + lost.stdout and out.read_text() == kept, lost.stderr
            second = subprocess.run(cmd + ["--batch", "day1_again", "--follows", "day1"], capture_output=True, text=True)
            now = out.read_text()
            assert second.returncode == 0 and now.startswith(kept) and len(now.splitlines()) == 14 and json.loads(now.splitlines()[-1])["follows"] == "day1", second.stderr
            return "a dry run writes nothing; a batch is appended once; a batch made again follows the first and leaves it as it was"

        def the_table() -> str:
            text = pa.table_text(list(got.values()), "B1")
            assert "batch B1: 7 line(s)" in text and "6 with a click by the owner, 1 with none" in text and "ONE BATCH. Counts, not rates" in text, text[:400]
            assert text.count("THE TRACKER'S PICK AGAINST THE OWNER'S CLICK, by ") == 4 and "THE SESSION'S READING AGAINST THE OWNER'S CLICK" in text
            row = next(t for t in text.splitlines() if t.strip().startswith(pa.HOW[2]))
            assert "right   1" in row and "wrong   1" in row, row                                   # pv: a wrong, b right, both picked on the shot's middle
            assert "`thinks_it_is` (the person a reader thinks IS the subject) is written on 1 of the batch's readings" in text
            assert "lines on which the track and the plain numbers differ: 1" in text and "under the line: 1" in text and "CANNOT SEE: whether the owner's click is right" in text
            none = pa.table_text([got[("pv", "a")]], "B1")
            assert "this batch's readers were not asked" in none, none
            return "four blocks of counts and the session's cells, for one batch, with what it cannot see"

        case("a_wrong_pick_is_wrong_both_ways", wrong_pick)
        case("another_pictures_number_is_not_compared", another_picture)
        case("not_outlined_and_not_on_frame", not_outlined_and_not_on_frame)
        case("no_click_is_kept_and_counted_nowhere", no_click)
        case("the_two_ways_can_differ_and_say_so", two_ways)
        case("how_the_pick_was_chosen", how_chosen)
        case("the_features_are_the_pictures_own", features)
        case("the_sessions_cells", session)
        case("the_owners_words_are_on_the_line", words)
        case("an_unsure_answer_is_counted_apart", unsure)
        case("a_batch_is_written_once", batches)
        case("the_table_is_counts_for_one_batch", the_table)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
