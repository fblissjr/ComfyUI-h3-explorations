#!/usr/bin/env python3
"""`bench/shots_record.py` joins two shots of a shots record or cuts one, keeping every answer whole; each case is a way an answer could be lost, moved to the wrong cut, or a record left that no longer covers its video.

  a_merge_keeps_both_halves_whole       the one shot holds both halves' words headed by their frames, both `by`, the frame lists joined, both
                                        notes and a line saying when, by whom and why, and the two shots as they were under `merged_from`
  a_merge_of_two_that_differ_is_refused where a subject is yes on one half and no on the other, until --stands says which stands and why
  only_neighbours_merge                 a shot and one that is not the next, or a frame that begins no shot, is refused
  a_split_keeps_the_answers_on_their_cut   the half the answers are on keeps its entries whole; the other half says what it was told and what the
                                        whole shot said; both keep the shot as it was under `split_from`
  an_answer_on_the_other_cut_is_refused a frame in an entry's lists or in its words that lies on the other half stops the split
  a_split_needs_a_word_for_everybody    the other half with a subject nobody spoke for is refused; so is a cut on the shot's first frame or outside it
  a_dry_run_writes_nothing              and prints the shots before and after
  the_record_is_written_as_its_others_are   one space of indent and no last newline, every other shot byte for byte, its span still covered, a
                                        line added to `written_by`; the coverage tool's own reader takes it

    <python> bench/check_shots_record.py

The standard library only; no model, no card.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shot_coverage  # noqa: E402
import shots_record as sr  # noqa: E402

TOOL = HERE / "shots_record.py"


def record() -> dict:
    who = lambda a, b, **more: {"a": {"present": a, "how": f"a is {a} here", "by": "owner, day one", **more}, "b": {"present": b, "how": "frame 12: not on this frame", "by": "owner, day one"}}     # noqa: E731
    return {"schema": "h3_shots/1", "video": "v.mp4", "span": [0, 99], "subjects": ["a", "b"], "written_by": "a session", "shots": [
        {"first": 0, "last": 9, "who": who("no", "no")},
        {"first": 10, "last": 29, "who": who("yes", "no", frames_on=[], frames_boxed=[12]), "note": "THE OWNER'S NOTE ON 10"},
        {"first": 30, "last": 49, "who": {"a": {"present": "yes", "how": "frame 33 (the source itself): on this frame, not outlined", "by": "owner, day two", "frames_on": [33], "frames_boxed": []},
                                          "b": {"present": "no", "how": "frame 33: not on this frame", "by": "owner, day two"}}, "scene_score": 0.4},
        {"first": 50, "last": 99, "who": {"a": {"present": "yes", "how": "frame 60: on this frame; frame 70: not on this frame", "by": "owner", "frames_on": [60], "frames_boxed": []},
                                          "b": {"present": "no", "how": "frame 60: not on this frame", "by": "owner"}}, "scene_score": 0.9}]}


def run(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args[:1], "--shots", str(path), *args[1:]], capture_output=True, text=True)


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f": {str(why)[:800]}"))
        fails += not ok

    r0 = record()
    a, b = r0["shots"][1], r0["shots"][2]
    one = sr.merged(a, b, ["a", "b"], "one continuous picture", "a check", "NOW")
    case("a_merge_keeps_both_halves_whole",
         (one["first"], one["last"]) == (10, 49) and one["who"]["a"]["present"] == "yes" and one["who"]["b"]["present"] == "no"
         and one["who"]["a"]["how"] == "10-29: a is yes here; 30-49: frame 33 (the source itself): on this frame, not outlined"
         and one["who"]["a"]["by"] == "10-29: owner, day one; 30-49: owner, day two" and one["who"]["a"]["frames_on"] == [33] and one["who"]["a"]["frames_boxed"] == [12]
         and "frames_on" not in one["who"]["b"] and one["merged_from"] == [a, b] and one["merged_from"][1]["scene_score"] == 0.4
         and one["note"] == "10-29: THE OWNER'S NOTE ON 10 || MERGED NOW by a check from the record's two shots 10-29 and 30-49: one continuous picture", one)
    differ = copy.deepcopy(b)
    differ["who"]["b"]["present"] = "yes"
    try:
        sr.merged(a, differ, ["a", "b"], "w", "c", "NOW")
        refused = ""
    except sr.Refused as e:
        refused = str(e)
    ruled = sr.merged(a, differ, ["a", "b"], "w", "c", "NOW", {"b": ("yes", "she walks in on 31")})
    case("a_merge_of_two_that_differ_is_refused", "b is `no` on 10-29 and `yes` on 30-49" in refused and "--stands" in refused
         and ruled["who"]["b"]["present"] == "yes" and "the two halves differ (no and yes) and `yes` stands: she walks in on 31 (c)" in ruled["who"]["b"]["how"], refused)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "shots.json"
        path.write_text(json.dumps(r0, indent=1))
        was = path.read_text()
        far = run(path, "merge", "--first", "10", "--second", "50", "--why", "w", "--by", "c")
        none = run(path, "merge", "--first", "10", "--second", "31", "--why", "w", "--by", "c")
        case("only_neighbours_merge", far.returncode == 5 and "does not begin on 50" in far.stdout and none.returncode == 5 and "not both the first frame of a shot" in none.stdout
             and path.read_text() == was, far.stdout + none.stdout)
        long = r0["shots"][3]
        other = {"a": ("unsure", "no answer of the owner's is on a frame of these"), "b": ("unsure", "no answer is on these")}
        try:
            sr.split(long, 65, "first", other, ["a", "b"], "w", "c", "NOW")
            astray = ""
        except sr.Refused as e:
            astray = str(e)
        first, second = sr.split(long, 80, "first", other, ["a", "b"], "another picture from 80", "a check", "NOW", "A SESSION'S READING: a is the one in the hat")
        case("a_split_keeps_the_answers_on_their_cut",
             (first["first"], first["last"], second["first"], second["last"]) == (50, 79, 80, 99) and first["who"] == long["who"] and first["scene_score"] == 0.9
             and second["who"]["a"] == {"present": "unsure", "how": "no answer of the owner's is on a frame of these; the whole shot 50-99 was `yes` (frame 60: on this frame; frame 70: not on this frame), said on the other cut",
                                        "by": "a check", "frames_on": [], "frames_boxed": []}
             and first["split_from"] == long == second["split_from"] and "scene_score" not in second
             and first["note"] == "SPLIT NOW by a check from the record's shot 50-99 at 80: another picture from 80"
             and second["note"] == "SPLIT NOW by a check from the record's shot 50-99 at 80: another picture from 80 || A SESSION'S READING: a is the one in the hat", (first, second))
        # the same shot kept on its SECOND half: refused, since its answers (60, 70) are then on the other cut
        try:
            sr.split(long, 80, "second", other, ["a", "b"], "w", "c", "NOW")
            wrong_half = ""
        except sr.Refused as e:
            wrong_half = str(e)
        case("an_answer_on_the_other_cut_is_refused", "a's entry names frame(s) [70]" in astray and "divided by a person" in astray
             and "a's entry names frame(s) [60, 70]" in wrong_half, astray + " | " + wrong_half)
        nobody = run(path, "split", "--shot", "50", "--at", "80", "--keeps", "first", "--other", "a=unsure:nothing known", "--why", "w", "--by", "c")
        edge = run(path, "split", "--shot", "50", "--at", "50", "--keeps", "first", "--other", "a=unsure:x", "--other", "b=unsure:x", "--why", "w", "--by", "c")
        out = run(path, "split", "--shot", "50", "--at", "120", "--keeps", "first", "--other", "a=unsure:x", "--other", "b=unsure:x", "--why", "w", "--by", "c")
        bad = run(path, "split", "--shot", "50", "--at", "80", "--keeps", "first", "--other", "a=maybe:x", "--other", "b=unsure:x", "--why", "w", "--by", "c")
        case("a_split_needs_a_word_for_everybody", nobody.returncode == 5 and "none was given for b" in nobody.stdout and edge.returncode == 5 and out.returncode == 5
             and "not inside 50-99 past its first frame" in edge.stdout + out.stdout and bad.returncode == 5 and "is not `SUBJECT=yes|no|unsure:words`" in bad.stdout
             and path.read_text() == was, nobody.stdout + edge.stdout + bad.stdout)
        dry = run(path, "merge", "--first", "10", "--second", "30", "--why", "one picture", "--by", "c", "--dry-run")
        case("a_dry_run_writes_nothing", dry.returncode == 0 and path.read_text() == was and "BEFORE:" in dry.stdout and "AFTER" in dry.stdout
             and "DRY: nothing written" in dry.stdout and '"last": 49' in dry.stdout and "3 shots" in dry.stdout, dry.stdout[-600:] + dry.stderr)
        real = run(path, "merge", "--first", "10", "--second", "30", "--why", "one  picture,\n looked at", "--by", "c")
        cut = run(path, "split", "--shot", "50", "--at", "80", "--keeps", "first", "--other", "a=unsure:nothing known", "--other", "b=unsure:nothing known", "--why", "another picture", "--by", "c")
        text = path.read_text()
        now = json.loads(text)
        ok = real.returncode == 0 and cut.returncode == 0 and text == json.dumps(now, indent=1) and not text.endswith("\n")
        ok = ok and [(s["first"], s["last"]) for s in now["shots"]] == [(0, 9), (10, 49), (50, 79), (80, 99)] and now["shots"][0] == r0["shots"][0]
        ok = ok and now["written_by"].startswith("a session; shots 10-29 and 30-49 merged into 10-49 by bench/shots_record.py for c at ") and "one picture, looked at" in now["written_by"] \
            and "shot 50-99 split into 50-79 and 80-99" in now["written_by"] and {k: v for k, v in now.items() if k not in ("shots", "written_by")} == {k: v for k, v in r0.items() if k not in ("shots", "written_by")}
        try:
            shot_coverage.load_shots(path)
        except SystemExit as stop:
            ok = False
            text = str(stop)
        case("the_record_is_written_as_its_others_are", ok, real.stdout[-300:] + cut.stdout[-300:] + text[:300])
    print(f"\n{'all' if not fails else fails} {'of ' if fails else ''}8 case(s) {'FAILED' if fails else 'ok'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
