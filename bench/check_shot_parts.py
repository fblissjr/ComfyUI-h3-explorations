#!/usr/bin/env python3
"""`bench/shot_parts.py` lists the shots with nobody and, for each shot and subject present, the frames their part is on; each case is a way a row could count the wrong frames, read the wrong preview or call something it should only show.

Why it exists: a session takes its shots to build from this table, and the owner reads from it where nothing can be
replaced. Its work is counting a saved mask on the right frames of the right preview and saying what stands there in
the coverage tool's own words; each of those can be off by a frame, a preview or a word.

  nobody_and_unsure                 list 1 is the shots where every subject is `no`; an `unsure` subject is under its own heading and its shot is not in list 1
  the_part_is_counted_on_its_frames a saved part that is large, then small, then gone: there, at or over the floor, under it and not there, as counts and as runs in source frames
  each_part_apart                   every part name and every class of a class map on any frame, on every row whatever the plan names, and hair that shows with no face reads so
  nothing_to_make_is_a_persons      a row whose plan parts a person's look says are seen on no frame is "present, nothing to make under this plan"; a seeded row whose
                                    plan parts are under the floor on every frame is only a candidate for a look
  seeded_or_picked                  from the preview's own table; the frames the track is off are counted and named
  the_last_preview_listed_is_read   where two cover a shot and track the subject, the last listed is read, the other is named, and the row says how many of the shot's frames it covers
  no_preview_says_so                a row no preview covers, and a row whose preview does not track the subject, each say which and carry what stands there alone
  one_window_or_more                the window the builder would take, and the frames over the model's ceiling
  what_stands_is_the_coverage_tools the row's state is `shot_coverage.coverage()`'s for the same files, with the piece laid and the owner's line on it
  the_owners_words_are_whole        every words line of the owner on the shot's frames, and no session's; the plan's dated note on a stretch of the shot is on that subject's row and no other
  no_frame_is_a_heading_and_the_rest_are_sorted   the row whose part is on no frame is under that heading; the others are in order of the frames the part is there, fewest first; no row is called anything
  pure_runs_and_figures             the two pure functions on small arrays

    <python> bench/check_shot_parts.py

numpy and ffmpeg (the mask videos are made here, small); no model, no card.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "workflows"))
import capture_masked_run as cap  # noqa: E402
import masked_job  # noqa: E402
import shot_coverage  # noqa: E402
import shot_parts  # noqa: E402
from _lib.frames import FPS, write_mask_video  # noqa: E402

TOOL = HERE / "shot_parts.py"
W, H = 128, 96
FLOOR = int(cap.PART_LEAST_PX)
LARGE, SMALL = (slice(10, 80), slice(10, 100)), (slice(10, 30), slice(10, 30))        # 6300 px and 400 px
assert 400 < FLOOR <= 6300, "the made parts no longer sit either side of the floor: size them again"


def mask(frames: int, where: dict) -> np.ndarray:
    """[frames, H, W] of bool: `where` is {(first, last): region}."""
    m = np.zeros((frames, H, W), bool)
    for (a, b), region in where.items():
        m[(slice(a, b + 1),) + region] = True
    return m


def class_video(path: Path, classes: np.ndarray) -> None:
    """A class map as the part node saves one: the grey level is the class's index."""
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "ffv1", "-level", "3", str(path)], stdin=subprocess.PIPE)
    enc.stdin.write(classes.astype(np.uint8).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0


def preview(folder: Path, name: str, first: int, frames: int, subjects: dict) -> str:
    """`subjects` is {label: {"track": mask, "parts": mask, "classes": array or None, "seed": str, "state": str}}."""
    out = folder / name
    out.mkdir()
    for label, s in subjects.items():
        write_mask_video(out / f"{name}_{label}_track_00001.mkv", s["track"])
        write_mask_video(out / f"{name}_{label}_parts_00001.mkv", s["parts"])
        if s.get("classes") is not None:
            class_video(out / f"{name}_{label}_classes_00001.mkv", s["classes"])
        (out / f"{name}_{label}_shots_00001_shots.json").write_text(json.dumps({"pick_seed": s.get("seed", ""), "shots": [
            {"shot": 1, "first_frame": 0, "last_frame": frames - 1, "subject": {"state": s.get("state", "picked")}}]}))
    job = folder / f"{name}.json"
    job.write_text(json.dumps({"job": {"name": name, "first": first, "frames": frames, "subjects": [{"label": k} for k in subjects]}}))
    return f"{job}={out}"


def main() -> int:
    fails = ran = 0

    def case(name, ok, why=""):
        nonlocal fails, ran
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f": {str(why)[:900]}"))
        fails += not ok
        ran += 1

    names = cap._pack("sapiens2_parts").CLASS_NAMES
    hair, face = names.index("Hair"), names.index("Face_Neck")
    longest = shot_parts.h3_config.MAX_LENGTH    # one window holds a load up to the ceiling (2026-10-15)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        last = 199 + longest + 20
        shots = {"schema": "h3_shots/1", "video": "t.mp4", "span": [100, last + 10], "subjects": ["a", "b"], "shots": [
            {"first": 100, "last": 119, "who": {"a": {"present": "yes"}, "b": {"present": "no"}}},
            {"first": 120, "last": 139, "who": {"a": {"present": "no"}, "b": {"present": "no"}}},
            {"first": 140, "last": 159, "who": {"a": {"present": "unsure", "how": "two say", "by": "x"}, "b": {"present": "yes"}}},
            {"first": 160, "last": 199, "who": {"a": {"present": "yes"}, "b": {"present": "yes"}}, "note": "A NOTE ON 160"},
            {"first": 200, "last": last, "who": {"a": {"present": "yes", "seen_looks": [
                {"part": "face_and_neck", "first": 200, "last": last, "is": "not seen", "why": "A LOOK: HER LEGS ONLY", "by": "a reader", "at": "2026-01-02T00:00:00Z"}]},
                "b": {"present": "no"}}},
            # nobody is `yes` here and one is `unsure`: not a shot with nobody
            {"first": last + 1, "last": last + 10, "who": {"a": {"present": "unsure", "how": "a back", "by": "y"}, "b": {"present": "no"}}}]}
        plan = {"schema": "h3_plan/1", "subjects": {"a": {"region": "face", "parts": ["face_and_neck"]},
                                                    "b": {"region": "face and hair", "parts": ["hair", "face_and_neck"], "keep_outs": ["held", "hands"]}},
                "load_prefixes": {"a": ["a_"], "b": ["b_"]},
                "departures": [{"first": 170, "last": 175, "subject": "a", "note": "THE BACK OF A HEAD, NO LOAD", "by": "a session's reading"}]}
        (d / "shots.json").write_text(json.dumps(shots))
        (d / "plan.json").write_text(json.dumps(plan))
        (d / "t.txt").write_text("100-119 /share/Video/x/a_one/a_one_00001.mp4 100\n")
        (d / "renders.json").write_text(json.dumps({"renders": [
            {"name": "a_one", "file": "Video/x/a_one/a_one_00001.mp4", "first": 100, "last": 119, "motion_wired": False, "as_the_graph_says": {"replace": "the wired parts"}},
            {"name": "b_spare", "file": "Video/x/b_spare/b_spare_00001.mp4", "first": 160, "last": 199, "motion_wired": False, "as_the_graph_says": {"replace": "the wired parts"}}]}))
        (d / "verdicts.jsonl").write_text(json.dumps({"render": "Video/x/a_one/a_one_00001.mp4", "at": "2026-01-01T00:00:00Z", "who": "owner", "words": "KEEP a_one"}) + "\n")
        (d / "answers.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [
            {"who": "owner", "row": "who", "what": "words", "source_frame": 160, "at": "2026-01-01T10:00:00Z", "words": "NOT SURE ANYTHING CAN BE REPLACED HERE"},
            {"who": "owner", "row": "who", "what": "words", "source_frame": 161, "at": "2026-01-01T10:00:01Z", "words": "A SECOND LINE"},
            {"who": "mrx", "row": "who", "what": "words", "source_frame": 160, "at": "2026-01-01T10:00:02Z", "words": "A SESSION'S"}]))
        # p1, shot 100-119, a: large on 100-109, small on 110-114, gone on 115-119; the track off on 118-119; picked
        p1 = preview(d, "p1", 100, 20, {"a": {"track": mask(20, {(0, 17): LARGE}), "parts": mask(20, {(0, 9): LARGE, (10, 14): SMALL})}})
        # p2, 140-199. a: the part on no frame of 140-159 and large on 160-199. b: hair on every frame, a face with it on 140-169 only; seeded
        classes = np.zeros((60, H, W), np.uint8)
        classes[:, 10:80, 10:60] = hair
        classes[:30, 10:80, 60:120] = face
        p2 = preview(d, "p2", 140, 60, {
            "a": {"track": mask(60, {(0, 59): LARGE}), "parts": mask(60, {(20, 59): LARGE})},
            "b": {"track": mask(60, {(0, 59): LARGE}), "parts": classes > 0, "classes": classes, "seed": "frame 3: 1, 1, 5, 5", "state": "taken (seeded)"}})
        # p3, 160-179, a alone, listed after p2: it is the one read for a on 160-199, and it covers half of that shot; small on every frame
        p3 = preview(d, "p3", 160, 20, {"a": {"track": mask(20, {(0, 19): LARGE}), "parts": mask(20, {(0, 19): SMALL}), "seed": "frame 0: 1, 1, 5, 5"}})
        cmd = [sys.executable, str(TOOL), "--shots", str(d / "shots.json"), "--plan", str(d / "plan.json"), "--table", str(d / "t.txt"), "--renders", str(d / "renders.json"),
               "--verdicts", str(d / "verdicts.jsonl"), "--answers", str(d / "answers.jsonl"), "--preview", p1, "--preview", p2, "--preview", p3,
               "--json", str(d / "out.json"), "--md", str(d / "out.md")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        out, md = json.loads((d / "out.json").read_text()), (d / "out.md").read_text()
        row = {(x["first"], x["subject"]): x for x in out["rows"]}
        # who else is in each shot is printed per shot, from the same previews (the lead's order of 2026-10-14); these made tables number
        # nobody on a shown frame, so it is not known, said, and no cast file was handed in
        case("others_present_per_shot", len(out["others_present"]) == len(json.loads((d / "shots.json").read_text())["shots"])
             and "## Other people in the shot:" in md and "with no cast file handed in" in out["others_present_from"]
             and all(r["others_present"] in ("not known", "no preview") for r in out["others_present"]), out.get("others_present"))

        case("nobody_and_unsure", out["nobody"] == [{"first": 120, "last": 139, "frames": 20}] and [(u["first"], u["subject"]) for u in out["unsure"]] == [(140, "a"), (last + 1, "a")]
             and len(out["rows"]) == 7 and "## List 1. Nobody: 1 shot(s), 20 frames, of 6 shots" in md and "- 140-159, a: two say (x)" in md, (out["nobody"], out["unsure"]))
        p = row[(100, "a")]["part"]
        case("the_part_is_counted_on_its_frames", (p["frames"], p["frames_there"], p["frames_at_or_over_the_floor"], p["frames_under_the_floor"], p["frames_not_there"]) == (20, 15, 10, 5, 5)
             and p["runs_there"] == [[100, 114]] and p["runs_not_there"] == [[115, 119]] and p["runs_at_or_over_the_floor"] == [[100, 109]]
             and p["px"] == {"least": 400, "median": 6300, "most": 6300} and p["floor_px"] == FLOOR and p["median_over_the_floor"] == round(6300 / FLOOR, 2)
             and row[(100, "a")]["preview"]["size"] == [W, H], p)
        apart = row[(160, "b")]["parts_apart"]
        case("each_part_apart", set(apart) == set(cap._pack("sapiens2_parts").PARTS) and apart["hair"]["frames_there"] == 40 and apart["face_and_neck"]["frames_there"] == 10
             and all(apart[k]["frames_there"] == 0 for k in apart if k not in ("hair", "face_and_neck"))
             and {k: v["frames_there"] for k, v in row[(160, "b")]["classes_seen"].items()} == {"Hair": 40, "Face_Neck": 10}
             and "every class on any frame: Face_Neck on 10 (10 at or over the floor); Hair on 40 (0 at or over the floor)" in md
             and apart["face_and_neck"]["runs_there"] == [[160, 169]] and apart["face_and_neck"]["runs_not_there"] == [[170, 199]] and row[(160, "b")]["part"]["frames_there"] == 40
             and row[(140, "b")]["parts_apart"]["face_and_neck"]["frames_there"] == 20 and row[(100, "a")]["parts_apart"] == {}
             and "`hair` apart (a part the plan replaces): there on 40 of 40 frames" in md and "`face_and_neck` apart (a part the plan replaces): there on 10 of 40 frames" in md, {k: (v["frames_there"], v["runs_there"]) for k, v in apart.items()})
        far_a, small = row[(200, "a")], row[(160, "a")]
        case("nothing_to_make_is_a_persons", far_a["stands"]["state"] == "present, nothing to make under this plan" and far_a["nothing_to_make"]
             and "A LOOK: HER LEGS ONLY" in far_a["nothing_to_make"][0] and "NOTHING TO MAKE UNDER THIS PLAN, a person's look: face_and_neck not seen" in md
             and small["stands"]["state"] != "present, nothing to make under this plan" and "a_candidate_for_a_look" in small
             and "A CANDIDATE FOR A LOOK" in md and "a_candidate_for_a_look" not in row[(100, "a")], (far_a.get("nothing_to_make"), small.get("a_candidate_for_a_look")))
        # the candidate needs a SEEDED tracker: a picked one may hold somebody else (the lead's ruling, 2026-10-14)
        sh = {"video": "t.mp4", "span": [0, 9], "subjects": ["a"], "shots": [{"first": 0, "last": 9, "who": {"a": {"present": "yes"}}}]}
        pl = {"subjects": {"a": {"region": "face", "parts": ["face_and_neck"]}}}
        covered = shot_coverage.coverage(sh, pl, [], [], [], [])
        tiny = lambda seeded: lambda pv, label, parts, names, menu: {"size": [W, H], "track_px": np.full(10, 900), "part_px": np.full(10, 5), "apart_px": {},  # noqa: E731
                                                                     "class_px": {}, "missing": [], "seeded": seeded, "cuts": []}
        made = [{"name": "pz", "spec": "pz", "first": 0, "last": 9, "subjects": {"a": {}}}]
        picked = shot_parts.lists(sh, pl, covered, made, [], measure=tiny(False))["rows"][0]
        seeded = shot_parts.lists(sh, pl, covered, made, [], measure=tiny(True))["rows"][0]
        case("a_candidate_needs_a_seeded_tracker", "a_candidate_for_a_look" not in picked and "a_candidate_for_a_look" in seeded
             and seeded["stands"]["state"] == "source, no line", (picked.get("a_candidate_for_a_look"), seeded.get("a_candidate_for_a_look")))
        t = row[(100, "a")]["tracker"]
        case("seeded_or_picked", t["seeded"] is False and (t["frames_on"], t["frames_off"], t["runs_off"]) == (18, 2, [[118, 119]]) and row[(160, "b")]["tracker"]["seeded"] is True
             and row[(160, "a")]["tracker"]["seeded"] is True and "PICKED, which may be somebody else; on 18 frames, off 2 (118-119)" in md and "SEEDED from a person's answer" in md, t)
        pv = row[(160, "a")]["preview"]
        case("the_last_preview_listed_is_read", pv["name"] == "p3" and pv["others_handed_in"] == ["p2"] and pv["covers"] == [160, 179] and pv["frames_covered"] == 20
             and row[(160, "a")]["part"]["frames"] == 20 and row[(160, "a")]["part"]["frames_under_the_floor"] == 20 and row[(160, "a")]["part"]["px"]["median"] == 400
             and "preview p3 (160-179), read on 20 of the shot's 40 frames; also handed in and not read: p2" in md, pv)
        far, other = row[(200, "a")], None
        r2 = subprocess.run(cmd[:cmd.index("--preview")] + ["--preview", p3, "--json", str(d / "o2.json"), "--md", str(d / "o2.md")], capture_output=True, text=True)
        if r2.returncode == 0:
            other = {(x["first"], x["subject"]): x for x in json.loads((d / "o2.json").read_text())["rows"]}[(160, "b")]
        case("no_preview_says_so", far["preview"] is None and far["part"] is None and far["no_preview"] == "no preview handed in covers this shot and tracks this subject"
             and far["stands"]["state"] and other is not None and other["no_preview"] == "a preview handed in covers this shot and does not track this subject"
             and "### No part read: 2 row(s)" in md and "NO PREVIEW READ: no preview handed in covers this shot" in md, (far.get("no_preview"), other and other.get("no_preview"), r2.stderr[-300:]))
        case("one_window_or_more", row[(100, "a")]["window"]["one_window"] is True and row[(100, "a")]["window"]["the_builder_would_take"] == 22 < min(masked_job.chain_lengths())
             and far["window"]["one_window"] is False and far["window"]["frames_over_the_longest"] == 20 and f"MORE THAN ONE WINDOW: 20 frames over the ceiling ({longest})" in md, far["window"])
        read_table, read_intents = shot_coverage._delivery()
        theirs = shot_coverage.coverage(shot_coverage.load_shots(d / "shots.json"), shot_coverage.load_plan(d / "plan.json", ["a", "b"]), read_table(str(d / "t.txt")),
                                        read_intents(str(d / "t.txt")), json.loads((d / "renders.json").read_text())["renders"],
                                        [json.loads(x) for x in (d / "verdicts.jsonl").read_text().splitlines()])
        want = {(x["first"], x["subject"]): x["state"] for x in theirs["rows"]}
        laid = row[(100, "a")]["stands"]["laid"]
        case("what_stands_is_the_coverage_tools", all(row[k]["stands"]["state"] == want[k] for k in row) and len(laid) == 1 and laid[0]["piece"].endswith("a_one_00001.mp4")
             and [o["name"] for o in row[(160, "b")]["stands"]["other_renders"]] == ["b_spare"] and "another render, not in the table: b_spare (160-199)" in md
             and (not laid[0]["owner_said"] or "KEEP a_one" in " ".join(laid[0]["owner_said"])), (laid, want))
        words = [w["words"] for w in row[(160, "a")]["the_owners_words"]]
        case("the_owners_words_are_whole", words == ["NOT SURE ANYTHING CAN BE REPLACED HERE", "A SECOND LINE"] and row[(100, "a")]["the_owners_words"] == []
             and "A SESSION'S" not in md and md.count("NOT SURE ANYTHING CAN BE REPLACED HERE") == 2 and "the shot's note: A NOTE ON 160" in md
             and row[(160, "a")]["the_plans_notes"] == ["170-175: THE BACK OF A HEAD, NO LOAD (a session's reading)"] and row[(160, "b")]["the_plans_notes"] == []
             and md.count("the plan's note on a stretch of it: 170-175: THE BACK OF A HEAD, NO LOAD (a session's reading)") == 1, words)
        none_at = md.index("### The part is on NO frame of the preview: 1 row(s)")
        some_at = md.index("### The part is on some frame: 4 row(s), fewest frames first, no line drawn")
        heads = [line for line in md[some_at:md.index("### No part read")].splitlines() if line.startswith("### ") and "-" in line]
        there = [row[(int(m.group(1)), m.group(2))]["part"]["frames_there"] for m in (re.match(r"### (\d+)-\d+, (\w+)", h) for h in heads)]
        case("no_frame_is_a_heading_and_the_rest_are_sorted", "### 140-159, a (UNSURE in the shots record)" in md[none_at:some_at] and row[(140, "a")]["part"]["frames_there"] == 0
             and there == sorted(there) and len(there) == 4 and "nothing to replace" not in md.lower() and "CANNOT SEE: a part model's miss reads as 'not there'" in md, (heads, there))
    f = shot_parts.figures(np.array([0, 5, 9, 0, 0, 12]), 50, 10)
    case("pure_runs_and_figures", shot_parts.runs([True, True, False, True], 7) == [[7, 8], [10, 10]] and shot_parts.runs([], 3) == []
         and (f["frames_there"], f["frames_at_or_over_the_floor"], f["frames_under_the_floor"], f["frames_not_there"]) == (3, 1, 2, 3)
         and f["runs_there"] == [[51, 52], [55, 55]] and f["runs_not_there"] == [[50, 50], [53, 54]] and f["px"] == {"least": 5, "median": 9, "most": 12}
         and shot_parts.figures(np.zeros(3, int), 0, 10)["px"] is None, f)
    print(f"\n{'all' if not fails else fails} {'of ' if fails else ''}{ran} case(s) {'FAILED' if fails else 'ok'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
