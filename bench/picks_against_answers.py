#!/usr/bin/env python3
"""What the models thought against what the owner said, kept a batch at a time: one line a subject of a preview, with what each got right.

    <python> bench/picks_against_answers.py --shots data/<stem>/shots.json --answers data/<stem>/poc/answers.jsonl \\
        --preview JOB.json=TABLES_DIR [--preview ...] [--notes PREVIEWS_RUN.jsonl ...] \\
        [--owner-unsure "FIRST-LAST=LINE[,LINE]:who read it so"] ... \\
        --batch NAME [--follows BATCH] --by SESSION --out data/<stem>/picks_against_answers.jsonl [--dry-run]

**What it buys.** The owner asked (2026-10-13) that what the models thought is kept against what they said, as data
that can be read over time for whether the picks are getting better. Three things say who a subject is on a shot:
a tracker picks a person, a session reads the tracker's tile and says whether the pick is right, and the owner
clicks on the who page. Until this tool the three sat in three files and were compared once, by eye, and the
comparison was thrown away. This writes one line for every subject of every preview given, holding all three and
the figures to read them across, appended to one file under a batch name. It decides nothing and corrects nothing.

**One line holds** (`SCHEMA`):

- what it is of: the batch, the video, the shot, the subject, the preview (its job's name, first frame, day, build
  commit, tracker mask version, how long it ran).
- `tracker`: its state and person number on its own table, the frame it picked or was seeded on, the table's own
  `why`, and how the pick was chosen (`HOW`): seeded; the largest person on a frame the owner had marked the subject
  on BEFORE the preview was built (a plain frame's answer, by its own time in the answers file); on the shot's
  middle; on a frame named otherwise.
- `session`: a reader's line on that tile from a notes file (`who_page.py --notes` takes the same file): the number
  the tracker picked, what that person looks like, whether the reader thinks it is the right person and why.
  FROM THE NEXT BATCH ON A READER ALSO WRITES `thinks_it_is`: the number, on the subject's own tile, of the person
  the reader thinks IS the subject, or `none of the outlined people`; the first batch's readers were not asked, and
  where they said a pick was wrong the person they meant is only in the free text of `why`, which is not read.
- `owner`: their last click for the subject on the shot's picture (a number, `not outlined`, `not on frame`, or no
  click), its line and time, how many times they clicked before it, and their typed words on the shot, whole.
- `agreement`: of the tracker with the owner, and of the session with the owner, each with how it was decided.
- `features`: figures on the frame the owner answered, from the preview's own tables and track.

**A person's number is of one picture.** The tracker's number is on the subject's own table and frame; the owner
clicked on the one picture the page showed, which may be another tracker's, of another frame. So whether the
tracker held the person the owner named is decided from the SUBJECT'S OWN TRACK on the frame the owner answered
(one frame of the preview's track video) against THE BOX of the person the owner named there: the overlap of the
two boxes, and the share of the track's pixels inside the named box. Both figures are kept. "Agrees" is the
overlap at or over `subject_track.SEED_AGREES`, the tracker's own line for two boxes being one person: a starting
value, reasoned and measured on nothing, and said so on every line. Where the subject's table shows the same
frame, the plain comparison of the two numbers is kept beside it, and a line on which the two ways differ says so.
The person the track lies on most is kept too: a track on the named person at an overlap under the line is a
different thing from a track on somebody else.

**The table it prints** is counts in four blocks (how the pick was chosen; the named person's rank by size; their
share of the frame in bands; the people outlined), for one batch, with what it was counted on and what it cannot
see. No rate, no verdict, no pass or fail. The other features are in the lines for whoever asks.

**An answer the owner was unsure of is not a clean reference.** The owner's click is what the other two are held
to, and their own words sometimes say they could not tell ("maybe it is 3"). `--owner-unsure` marks a shot's lines
with the words' line numbers and who read them so: a person's reading of the owner's words, never this tool's.
Such lines are kept whole and counted APART from the table's blocks.

**A batch is written once.** A name already in the file is refused. A batch made again (the owner changed a
click) takes a new name and `--follows` the one it replaces; nothing is rewritten.

**What it cannot see.** Whether the owner's click is right: it is the reference here, not a measurement. Anything
about a subject the owner did not click for: the line is kept, so what was asked is counted, and it is in no
agreement cell. What a reader who said "wrong" thought was right, until readers write `thinks_it_is`. Why a
tracker picked as it did, past its table's own words. One batch says nothing about getting better; it is what
the next is read against.
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

SCHEMA = "h3_picks_against_answers/1"
WHO_ROWS = ("b20", "who")
#: The who page's buttons that are not a number, as the answers file holds them.
NOT_OUTLINED, NOT_ON_FRAME = "not outlined", ("not on frame", "not in shot")
NO_CLICK = "NO CLICK"
#: How a tracker's pick was chosen, as one of these.
HOW = ("seeded", "largest, on a frame the owner had marked the subject on", "largest, on the shot's middle",
       "largest, on a frame named otherwise", "largest, on the tracker's own frame")
#: The bands a person's share of the frame is counted in. Reasoned, a starting value: plain bands a person can
#: hold in their head, a face in an inset under the first and a close-up over the last. Not from any clip.
SHARE_BANDS = ((0.01, "under 1% of the frame"), (0.05, "1% to under 5%"), (0.20, "5% to under 20%"), (None, "20% or more"))
#: The table's bands for the people outlined on a picture and for a person's rank by size. Reasoned, starting values:
#: a table of counts on a few dozen lines reads as more than it holds when every count has a row of its own. The
#: exact figures are in the lines.
PEOPLE_BANDS = ((1, "1 outlined"), (2, "2 outlined"), (3, "3 outlined"), (6, "4 to 6 outlined"), (None, "7 or more outlined"))
RANK_BANDS = ((1, "rank 1, the largest"), (2, "rank 2"), (3, "rank 3"), (None, "rank 4 or lower"))
LINE_IS = "subject_track.SEED_AGREES, the tracker's own line for two boxes being one person: a starting value, reasoned and measured on nothing"


def seed_agrees(repo: Path = REPO) -> float:
    """`subject_track.SEED_AGREES`, read from the source and not imported: the module needs ComfyUI."""
    for line in ast.parse((repo / "subject_track.py").read_text()).body:
        if isinstance(line, ast.Assign) and any(getattr(t, "id", "") == "SEED_AGREES" for t in line.targets):
            return float(ast.literal_eval(line.value))
    raise SystemExit("subject_track.py no longer sets SEED_AGREES: the line two boxes are held to has nowhere to be read from")


def box_overlap(a, b) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1); 0 when either is missing (`subject_tracks.box_overlap`)."""
    if a is None or b is None:
        return 0.0
    shared = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union > 0 else 0.0


def corners(box) -> tuple[float, float, float, float]:
    """A shot table's box (x, y, width, height) as (x0, y0, x1, y1)."""
    return float(box[0]), float(box[1]), float(box[0]) + float(box[2]), float(box[1]) + float(box[3])


def mask_box(mask):
    """A mask's box (x0, y0, x1, y1), far side exclusive, or None for an empty one."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def track_on_frame(path: Path, size, at: int, frame: int):
    """One frame of a preview's track video as a mask at `size`, or None when the video does not reach the frame."""
    from _lib.frames import read_mask
    mask, covered = read_mask(str(path), tuple(size), int(at), int(frame), 1)
    return mask[0] if bool(covered[0]) else None


def previews(specs: list[str]) -> list[dict]:
    """Each `JOB.json=TABLES_DIR`: the job, each subject's table and track video, and the records beside the job file."""
    out = []
    for spec in specs:
        job_path, _, folder = spec.partition("=")
        if not folder:
            raise SystemExit(f"--preview {spec!r}: write JOB.json=TABLES_DIR")
        job, beside = json.loads(Path(job_path).read_text())["job"], Path(job_path).parent
        read = lambda name: json.loads((beside / name).read_text()) if (beside / name).is_file() else {}   # noqa: E731
        built, queued = read("preview.built.json"), read("preview.queued.json")
        tables, tracks = {}, {}
        for s in job["subjects"]:
            p = Path(folder) / f"{job['name']}_{s['label']}_shots_00001_shots.json"
            if not p.is_file():
                raise SystemExit(f"{spec}: no table {p}")
            tables[s["label"]] = json.loads(p.read_text())
            listed = [Path(f).name for f in queued.get("files") or [] if f"_{s['label']}_track_" in Path(f).name and f.endswith(".mkv")]
            tracks[s["label"]] = Path(folder) / listed[0] if len(listed) == 1 and (Path(folder) / listed[0]).is_file() else None
        out.append({"name": job["name"], "first": int(job["first"]), "last": int(job["first"]) + int(job["frames"]) - 1, "job": job,
                    "tables": tables, "tracks": tracks, "built": built, "queued": queued, "job_file": str(job_path)})
    return out


def owner_lines(answers: list[tuple[int, dict]]) -> tuple[dict, list]:
    """The owner's clicks by picture and subject ({(table, frame): {subject: the last click, with how many came
    before}}), and their typed words lines. Only lines whose `who` is `owner` on a who row."""
    clicks: dict = {}
    words = []
    for n, d in answers:
        if d.get("who") != "owner" or d.get("row") not in WHO_ROWS:
            continue
        if d.get("what") == "words" and str(d.get("words", "")).strip() and str(d.get("source_frame", "")).isdigit():
            words.append({"line": n, "at": d.get("at"), "frame": int(d["source_frame"]), "words": str(d["words"])})
        elif d.get("what") == "choice" and d.get("subject") and d.get("table") and d.get("table") != "plain" and str(d.get("source_frame", "")).isdigit():
            tile = clicks.setdefault((str(d["table"]), int(d["source_frame"])), {})
            before = tile.get(str(d["subject"]), {"before": -1})["before"] + 1
            tile[str(d["subject"])] = {"chosen": str(d.get("chosen", "")), "line": n, "at": d.get("at"), "before": before}
    return clicks, words


def notes_of(paths: list[str]) -> dict:
    """A session's readings by (the preview's job name, its first frame, the subject): `who_page.py --notes` takes the same files."""
    out = {}
    for path in paths:
        for n, text in enumerate(Path(path).read_text().splitlines(), 1):
            if not text.strip():
                continue
            row = json.loads(text)
            for s in row.get("subjects", []):
                out[(Path(row["job_file"]).stem, int(row["first"]), str(s.get("label")))] = {**s, "recorded_at": row.get("recorded_at"), "file": str(path), "line": n}
    return out


def marked_before(answers: list[tuple[int, dict]], subject: str, built_at) -> set[int]:
    """The frames the owner had marked a subject ON, by a plain frame's `on this frame, not outlined`, BEFORE a preview
    was built. Read from the answers file with each line's own time, never from the shots record's frame fields: those
    are written again after every answer, and on the first batch they called a pick on the shot's middle "a frame the
    owner had marked" because the owner's click on that preview had since boxed that very frame."""
    built = None if not built_at else datetime.fromisoformat(str(built_at))
    out, last = set(), {}
    for _n, d in answers:
        if d.get("who") == "owner" and d.get("row") in WHO_ROWS and d.get("table") == "plain" and d.get("what") == "choice" and d.get("subject") == subject \
                and str(d.get("source_frame", "")).isdigit():
            when = datetime.fromisoformat(str(d["at"]).replace("Z", "+00:00")) if d.get("at") else None
            if built is None or built.tzinfo is None or when is None or when < built:
                last[int(d["source_frame"])] = str(d.get("chosen"))
    for frame, chosen in last.items():
        if chosen == NOT_OUTLINED:
            out.add(frame)
    return out


def how_chosen(table: dict, state: str, subject_job: dict, shot: dict, marked: set[int]) -> str:
    if table.get("pick_seed") or "seeded" in str(state):
        return HOW[0]
    at = subject_job.get("pick_at")
    if at is None:
        return HOW[4]
    if int(at) in marked:
        return HOW[1]
    return HOW[2] if int(at) == (shot["first"] + shot["last"]) // 2 else HOW[3]


def likeness(a: dict | None, b: dict | None) -> float | None:
    """How alike two people of one table are: the lowest, over the places both carry a signature in, of the two
    signatures' product (they are unit length), which is the number the tracker itself holds a line against."""
    if not a or not b:
        return None
    pairs = [(x, y) for x, y in zip(a.get("signatures") or [], b.get("signatures") or []) if x is not None and y is not None]
    return round(min(sum(p * q for p, q in zip(x, y)) for x, y in pairs), 4) if pairs else None


def band(share: float | None) -> str:
    if share is None:
        return "no person named"
    return next(words for top, words in SHARE_BANDS if top is None or share < top)


def tracker_against_owner(click: dict | None, named: dict | None, track, people: list[dict], same_frame: bool, picked: int | None, line: float) -> dict:
    """Did the tracker hold who the owner said: by the subject's track against the named person's box, and by the two numbers."""
    if click is None:
        return {"agrees": None, "how": "the owner made no click for this subject on this shot's picture"}
    chosen = click["chosen"]
    out: dict = {"agrees": None, "by_the_track": None, "by_the_numbers": None, "line": line, "line_is": LINE_IS}
    if track is None:
        out["how"] = "the preview's record lists no one track video of the subject that reaches the frame the owner answered"
    else:
        on = bool(track.any())
        tbox = mask_box(track)
        each = sorted(((box_overlap(tbox, corners(p["box"])), int(p["person"])) for p in people), reverse=True) if on else []
        out["the_track_on_that_frame"] = {"on": on, "share_of_the_frame": round(float(track.mean()), 5), "box": None if tbox is None else [int(v) for v in tbox]}
        out["the_track_lies_most_on"] = {"person": each[0][1], "overlap": round(each[0][0], 4)} if each else None
        if chosen.isdigit():
            if named is None:
                out["how"] = f"the picture's table has no person {chosen}"
            else:
                x0, y0, x1, y1 = (int(round(v)) for v in corners(named["box"]))
                inside = float(track[y0:y1, x0:x1].sum()) / float(track.sum()) if on else 0.0
                overlap = box_overlap(tbox, corners(named["box"]))
                out.update(by_the_track=overlap >= line, overlap_of_the_track_with_the_named_box=round(overlap, 4),
                           share_of_the_track_inside_the_named_box=round(inside, 4),
                           how="the overlap of the subject's track's box with the named person's box, on the frame the owner answered")
        elif chosen in NOT_ON_FRAME:
            out.update(by_the_track=not on, how="the owner says the subject is not on the frame: the track is off there, or it is not")
        elif chosen == NOT_OUTLINED:
            if not on:
                out.update(by_the_track=False, how="the owner says the subject is on the frame and not outlined, and the track is off there")
            elif each and each[0][0] >= line:
                out.update(by_the_track=False, how="the owner says the subject is none of the outlined people, and the track lies on one of them")
            else:
                out["how"] = ("the owner says the subject is none of the outlined people, and the track lies on none of them at the line: whether what "
                              "it lies on is the subject cannot be said from the files")
    if chosen.isdigit() and same_frame and picked is not None:
        out["by_the_numbers"] = picked == int(chosen)
    out["agrees"] = out["by_the_track"] if out["by_the_track"] is not None else out["by_the_numbers"]
    if out["by_the_track"] is not None and out["by_the_numbers"] is not None and out["by_the_track"] != out["by_the_numbers"]:
        out["the_two_ways_differ"] = (f"by the track {'yes' if out['by_the_track'] else 'no'}, by the numbers {'yes' if out['by_the_numbers'] else 'no'}: "
                                      "the track's box and the picked person's are held to a line, and a number is not")
    return out


def session_against_owner(reading: dict | None, tracker: dict, click: dict | None, of_another_picture) -> dict:
    if reading is None:
        return {"cell": "no reading by a session"}
    said, right = str(reading.get("right_person")), tracker.get("agrees")
    if right is None:
        cell = "the pick's rightness cannot be said, so neither can the reading's"
    elif said == "yes":
        cell = "said the pick was right, and it was" if right else "said the pick was right, and it was not"
    elif said == "no":
        cell = "said the pick was wrong, and it was" if not right else "said the pick was wrong, and it was right"
    else:
        cell = "could not tell"
    out = {"said_the_pick_was_right": said, "the_pick_was_right_by_the_owner": right, "cell": cell}
    thinks = reading.get("thinks_it_is")
    if thinks is None:
        out["thinks_it_is"] = "not written: the reader was not asked who the subject IS"
    elif click is not None and of_another_picture is False and click["chosen"].isdigit() and str(thinks).isdigit():
        out["thinks_it_is"], out["the_reader_named_the_owners_person"] = int(thinks), int(thinks) == int(click["chosen"])
    else:
        out["thinks_it_is"] = thinks
    return out


def lines_of(shots: dict, pvs: list[dict], answers: list[tuple[int, dict]], notes: dict, batch: str, by: str, now: str, commit: str,
             line: float, track=track_on_frame, follows: str = "", unsure: dict | None = None) -> list[dict]:
    """One line a subject of each preview's job and each shot of the record the preview lies in. `unsure` is
    {(shot's first, last): {"lines": the owner's words lines that say so, "read_so_by": who}}."""
    clicks, words = owner_lines(answers)
    for (first, last), mark in (unsure or {}).items():
        if not any((s["first"], s["last"]) == (first, last) for s in shots["shots"]):
            raise SystemExit(f"--owner-unsure names frames {first}-{last}, which is not a shot of the shots file")
        stray = [n for n in mark["lines"] if not any(w["line"] == n and first <= w["frame"] <= last for w in words)]
        if stray:
            raise SystemExit(f"--owner-unsure {first}-{last}: line(s) {stray} are not the owner's typed words on a frame of that shot")
    out = []
    for pv in pvs:
        names = {f"{pv['name']}_{label}_shots_00001_shots.json": label for label in pv["tables"]}
        for shot in [s for s in shots["shots"] if s["first"] <= pv["last"] and s["last"] >= pv["first"]]:
            said = [w for w in words if shot["first"] <= w["frame"] <= shot["last"]]
            for place, sj in enumerate(pv["job"]["subjects"], 1):
                label, own = sj["label"], pv["tables"][sj["label"]]
                mine = [(c[label]["line"], tname, frame, c[label]) for (tname, frame), c in clicks.items()
                        if tname in names and shot["first"] <= frame <= shot["last"] and label in c]
                click_line, tname, frame, click = max(mine) if mine else (None, None, None, None)
                k = None if frame is None else frame - pv["first"]
                own_shot = next((t for t in own["shots"] if k is not None and int(t["first_frame"]) <= k <= int(t["last_frame"])), None) \
                    or next((t for t in own["shots"] if t.get("people")), own["shots"][0])
                state = own_shot.get("subject") or {}
                picked = state.get("person")
                tile = pv["tables"][names[tname]] if tname else None
                tile_shot = next((t for t in tile["shots"] if int(t["shown_frame"]) == k), None) if tile else None
                people = (tile_shot or {}).get("people") or []
                named = next((p for p in people if click and click["chosen"].isdigit() and int(p["person"]) == int(click["chosen"])), None)
                same_frame = None if frame is None else pv["first"] + int(own_shot["shown_frame"]) == frame
                path = pv["tracks"].get(label)
                mask = track(path, own["size"], pv["first"], frame) if (path is not None and frame is not None) else None
                agreement = tracker_against_owner(click, named, mask, people, bool(same_frame), picked, line)
                reading = notes.get((pv["name"], pv["first"], label))
                by_size = sorted(people, key=lambda p: -float(p.get("share_of_frame") or 0.0))
                rank = lambda p: None if p is None else 1 + [int(q["person"]) for q in by_size].index(int(p["person"]))        # noqa: E731
                held = next((p for p in people if agreement.get("the_track_lies_most_on") and int(p["person"]) == agreement["the_track_lies_most_on"]["person"]), None)
                w, h = own["size"]
                scores = [v for v in own_shot.get("track_score") or [] if v is not None]
                at_k = None if k is None else k - int(own_shot["first_frame"])
                score_here = (own_shot.get("track_score") or [None])[at_k] if at_k is not None and 0 <= at_k < len(own_shot.get("track_score") or []) else None
                features = {
                    "measured_on": "the picture the owner answered on (its table's people) and the subject's own table and track; None where there was no click",
                    "people_outlined": len(people) if tile_shot else None, "people_with_a_head": sum(1 for p in people if p.get("has_head")) if tile_shot else None,
                    "the_named_person": None if named is None else {
                        "share_of_the_frame": named.get("share_of_frame"), "share_band": band(named.get("share_of_frame")), "rank_by_size": rank(named),
                        "share_over_the_largest": round(float(named["share_of_frame"]) / float(by_size[0]["share_of_frame"]), 4) if by_size[0].get("share_of_frame") else None,
                        "detector_score": named.get("detector_score"), "has_head": named.get("has_head"), "held_by_another_tracker": named.get("held_by_another_tracker"),
                        "box": named["box"], "height_px": named["box"][3], "centre": [round((named["box"][0] + named["box"][2] / 2) / w, 3), round((named["box"][1] + named["box"][3] / 2) / h, 3)],
                        "touches_the_frames_edge": bool(named["box"][0] <= 1 or named["box"][1] <= 1 or named["box"][0] + named["box"][2] >= w - 1 or named["box"][1] + named["box"][3] >= h - 1)},
                    "the_person_the_track_lies_most_on": None if held is None else {"person": int(held["person"]), "share_of_the_frame": held.get("share_of_frame"), "rank_by_size": rank(held)},
                    "likeness_of_the_named_to_the_one_the_track_lies_on": None if named is None or held is None or named is held else likeness(named, held),
                    "the_tracks_score_on_the_frame": score_here, "the_tracks_score_median_over_its_shot": round(statistics.median(scores), 3) if scores else None,
                    "shot_frames": shot["last"] - shot["first"] + 1, "frames_the_tracker_has_the_subject_on": own_shot.get("frames_with_subject"),
                    "cuts_the_tracker_made_in_the_preview": len(own.get("cuts") or []), "subjects_in_the_job": len(pv["job"]["subjects"]), "place_in_the_trackers_order": place}
                out.append({
                    "schema": SCHEMA, "batch": batch, "follows": follows or None, "made_at": now, "made_by": by, "tool_commit": commit,
                    "video": shots.get("video"), "shot": [shot["first"], shot["last"]], "subject": label,
                    "preview": {"job": pv["name"], "first": pv["first"], "frames": pv["last"] - pv["first"] + 1, "job_file": pv["job_file"],
                                "built_at": pv["built"].get("built_at"), "commit": pv["built"].get("commit"), "tracker_mask_version": pv["built"].get("tracker_mask_version"),
                                "seconds": pv["queued"].get("seconds")},
                    "tracker": {"state": state.get("state"), "person": picked, "on_frame": pv["first"] + int(own_shot["shown_frame"]), "why": state.get("why"),
                                "similarity": state.get("similarity"), "also_held_by_another_tracker": state.get("also_held_by_another_tracker"),
                                "how_chosen": how_chosen(own, str(state.get("state")), sj, shot, marked_before(answers, label, pv["built"].get("built_at")))},
                    "session": None if reading is None else {
                        **{f: reading.get(f) for f in ("picked_number", "picked_looks_like", "right_person", "why", "thinks_it_is", "recorded_at", "file", "line")},
                        "number_is_of_another_picture": None if same_frame is None else not same_frame},
                    "owner": {"click": click["chosen"] if click else NO_CLICK, "line": click_line, "at": click["at"] if click else None,
                              "clicks_before_the_last": click["before"] if click else None, "picture": tname, "frame": frame, "words_on_the_shot": said,
                              "unsure": (unsure or {}).get((shot["first"], shot["last"]))},
                    "agreement": {"tracker_against_owner": agreement,
                                  "session_against_owner": session_against_owner(reading, agreement, click, None if same_frame is None else not same_frame)},
                    "features": features})
    return out


def table_text(lines: list[dict], batch: str) -> str:
    """Counts for one batch, in four blocks. Figures with what they were counted on and what they cannot see."""
    every = [x for x in lines if x["owner"]["click"] != NO_CLICK]
    doubted = [x for x in every if x["owner"].get("unsure")]
    clicked = [x for x in every if not x["owner"].get("unsure")]
    word = lambda v: "right" if v is True else "wrong" if v is False else "cannot be said"                                    # noqa: E731

    def block(title: str, key) -> list[str]:
        """`key(x)` is (the row's place in its block, its words)."""
        cells: dict = {}
        for x in clicked:
            cells.setdefault(key(x), {"right": 0, "wrong": 0, "cannot be said": 0})[word(x["agreement"]["tracker_against_owner"]["agrees"])] += 1
        rows = [f"  {name:58s} right {c['right']:3d}   wrong {c['wrong']:3d}   cannot be said {c['cannot be said']:3d}" for (_place, name), c in sorted(cells.items())]
        return [f"THE TRACKER'S PICK AGAINST THE OWNER'S CLICK, by {title}:"] + (rows or ["  (no line with a click)"]) + [""]
    named = lambda x: x["features"]["the_named_person"] or {}                                                                # noqa: E731
    no_number = lambda x: (99, f"the owner's click was `{x['owner']['click']}`")                                              # noqa: E731
    banded = lambda value, bands: next((n, words) for n, (top, words) in enumerate(bands) if top is None or value <= top)     # noqa: E731
    shares = [words for _top, words in SHARE_BANDS]
    apart = {"right": 0, "wrong": 0, "cannot be said": 0}
    for x in doubted:
        apart[word(x["agreement"]["tracker_against_owner"]["agrees"])] += 1
    out = [f"picks against answers, batch {batch}: {len(lines)} line(s), one a subject of a preview's job and a shot; {len(every)} with a click by the owner, "
           f"{len(lines) - len(every)} with none (kept, in no count below).",
           (f"OF THE CLICKS, {len(doubted)} are on shots where the owner's own words say they were unsure (shots "
            + ", ".join(sorted({f"{x['shot'][0]}-{x['shot'][1]}" for x in doubted})) + "; read so by "
            + "; ".join(sorted({str(x['owner']['unsure']['read_so_by']) for x in doubted})) + f"). They are not clean answers and are counted APART, not in the blocks: "
            f"the tracker right {apart['right']}, wrong {apart['wrong']}, cannot be said {apart['cannot be said']}. The blocks below are of the other {len(clicked)}."
            if doubted else f"No shot is marked as one the owner was unsure of: the blocks below are of all {len(clicked)}."),
           "ONE BATCH. Counts, not rates: nothing here says the picks are getting better or worse; the next batch is read against this one.",
           f"'right' is the subject's track on the frame the owner answered overlapping the named person's box at or over the line ({LINE_IS}).", ""]
    out += block("how the pick was chosen", lambda x: (HOW.index(x["tracker"]["how_chosen"]), x["tracker"]["how_chosen"]))
    out += block("the named person's rank by size among the people outlined", lambda x: banded(named(x)["rank_by_size"], RANK_BANDS) if named(x) else no_number(x))
    out += block("the named person's share of the frame", lambda x: (shares.index(named(x)["share_band"]), named(x)["share_band"]) if named(x) else no_number(x))
    out += block("the people outlined on the picture", lambda x: banded(x["features"]["people_outlined"], PEOPLE_BANDS))
    cells: dict = {}
    for x in clicked:
        cells[x["agreement"]["session_against_owner"]["cell"]] = cells.get(x["agreement"]["session_against_owner"]["cell"], 0) + 1
    out += ["THE SESSION'S READING AGAINST THE OWNER'S CLICK (the reader said whether the tracker's pick was right):"] \
        + [f"  {name:66s} {n:3d}" for name, n in sorted(cells.items())] + [""]
    asked = sum(1 for x in lines if x["session"] and x["session"].get("thinks_it_is") is not None)
    differ = sum(1 for x in clicked if x["agreement"]["tracker_against_owner"].get("the_two_ways_differ"))
    near = sum(1 for x in clicked if x["agreement"]["tracker_against_owner"].get("by_the_track") is False and named(x)
               and (x["agreement"]["tracker_against_owner"].get("the_track_lies_most_on") or {}).get("person") == int(x["owner"]["click"]))
    out += [f"`thinks_it_is` (the person a reader thinks IS the subject) is written on {asked} of the batch's readings"
            + (": this batch's readers were not asked, so a reading that said 'wrong' is not counted as naming anybody." if not asked else "."),
            f"lines on which the track and the plain numbers differ: {differ}. 'wrong' lines whose track lies on the named person more than on anybody else, under the line: {near}.",
            "CANNOT SEE: whether the owner's click is right (it is the reference); anything about a subject the owner did not click for; why a tracker picked as it "
            "did; whether the line suits these pictures (it was reasoned, not measured). The other features (likeness, place, height, scores, lengths) are in the lines."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--shots", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--preview", action="append", default=[], metavar="JOB.json=TABLES_DIR", required=True)
    ap.add_argument("--notes", action="append", default=[], metavar="PREVIEWS_RUN.jsonl", help="a session's readings of the previews' tiles, as `who_page.py --notes` takes them")
    ap.add_argument("--owner-unsure", action="append", default=[], metavar="FIRST-LAST=LINE[,LINE]:who read it so",
                    help="a shot on which the owner's own words say they could not tell: the answers lines of those words, and who read them so; repeatable")
    ap.add_argument("--batch", required=True, help="the batch's name, by its previews' day; one already in the file is refused")
    ap.add_argument("--follows", default="", help="the batch this one is made again from (the owner changed a click): it must be in the file")
    ap.add_argument("--by", required=True, help="the session")
    ap.add_argument("--out", required=True, help="the file the batch is appended to")
    ap.add_argument("--dry-run", action="store_true", help="print the table and write nothing")
    a = ap.parse_args()
    out = Path(a.out)
    was = out.read_text() if out.is_file() else ""
    there = {json.loads(t)["batch"] for t in was.splitlines() if t.strip()}
    if a.batch in there:
        raise SystemExit(f"{out} already holds a batch named {a.batch!r}: a batch is written once. Made again, it takes a new name and --follows {a.batch}")
    if a.follows and a.follows not in there:
        raise SystemExit(f"--follows {a.follows!r}: {out} holds no batch of that name ({', '.join(sorted(there)) or 'it is empty'})")
    answers = [(n, json.loads(t)) for n, t in enumerate(Path(a.answers).read_text().splitlines(), 1) if t.strip()]
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    unsure = {}
    for text in a.owner_unsure:
        try:
            span, rest = text.split("=", 1)
            numbers, who = rest.split(":", 1)
            first, last = (int(v) for v in span.split("-"))
            unsure[(first, last)] = {"lines": [int(v) for v in numbers.split(",")], "read_so_by": " ".join(who.split())}
            if not unsure[(first, last)]["read_so_by"]:
                raise ValueError
        except ValueError:
            raise SystemExit(f"--owner-unsure {text!r} is not `FIRST-LAST=LINE[,LINE]:who read it so`") from None
    lines = lines_of(json.loads(Path(a.shots).read_text()), previews(a.preview), answers, notes_of(a.notes), a.batch, a.by, now, commit, seed_agrees(), follows=a.follows,
                     unsure=unsure)
    print(table_text(lines, a.batch))
    if a.dry_run:
        print(f"DRY: nothing written; {len(lines)} line(s) would be appended to {out}")
        return 0
    if (out.read_text() if out.is_file() else "") != was:
        raise SystemExit(f"{out} changed while this ran; nothing is written. Run again.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    print(f"appended {len(lines)} line(s) to {out} as batch {a.batch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
