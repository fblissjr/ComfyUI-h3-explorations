#!/usr/bin/env python3
"""Who else is in a shot: the people a preview's trackers numbered that are none of the video's subjects, a shot at a time.

    <python> bench/others_present.py --shots data/<stem>/shots.json --cast data/<stem>/cast/cast.jsonl \\
        (--preview JOB.json=TABLES_DIR [--preview ...] | --newest-previews-in data --share OUTPUT_DIR) --json OUT.json --md OUT.md

**What it is for.** The owner asked (2026-10-14) why "are there people in the shot who are not ours" was not in a
load's job file. A preview's trackers number every person their detector finds on a shot's shown frame, in one
numbering for every tracker of the job (`shot_table.py`, the table's `people_are`), and the owner said on the who
pages which number is which subject (the cast file). So everyone else numbered is someone not ours. Nothing here
needs the card: it reads the tables the previews saved and the owner's answers.

**A shot, as it reads it.** The record's shot; the preview handed in that covers it (where two do, the last
listed, as `bench/shot_parts.py` reads them); each of its trackers' cuts that lie on the shot, each with its shown
frame. On each cut, OURS is each tracker's own person (its `subject.person`), corrected by the owner's answer for
that subject on that very table and frame where there is one (`--cast`: a box is matched to the numbered person it
overlaps most; "none of the outlined people" means the tracker's person is not that subject), and any person
another tracker's mask lies on (`held_by_another_tracker`). OTHERS are the rest: each with its number, box, share
of the frame and the gap in px to the nearest of ours (0 where the boxes meet).

**The answer a shot.** `yes` where any cut's shown frame has someone else; `no` where none has and the shot is at
most `LONG_SHOT_FRAMES` long; `cannot say` where none has and it is longer: the frames between the shown ones were
not looked at, and someone may walk in there (how many frames were not shown is said). A shot no preview covers is
`no preview`; one whose preview's tables have no cut on it, `not known`.

**What it cannot see.** A person the detector did not find on the shown frame; anyone on frames that were not
shown (said as `cannot say` only past the length line); whether a numbered person the owner never answered about is
the tracker's subject (the tracker's pick is taken, said as PICKED where it was not seeded); the pixels of anyone
(the whole-frame strangers pass, `bench/strangers.py`, makes those, on the card).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from picks_against_answers import box_overlap, corners  # noqa: E402
from who_page import LONG_SHOT_FRAMES  # noqa: E402

#: The least overlap at which an answer's box names a numbered person. Reasoned, a starting value: the cast file's box
#: is the table's own box of the person clicked, so it overlaps its person almost whole and any other far less.
ANSWER_OVERLAP = 0.5


def gap_px(a, b) -> float:
    """The gap between two boxes (x0, y0, x1, y1) in px; 0 where they meet or overlap."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return round((dx * dx + dy * dy) ** 0.5, 1)


#: A video's `not_people` (its plan: what, the shots it shows in, who said so): in those shots, a LONE other on a shown frame
#: smaller than this share of the frame that touches none of ours counts as no one, each call said with its box. The owner's
#: word of 2026-10-14 on the girl printed on the cereal box ("we've seen that a million times"); the share is the lead's, a
#: starting value: the box's other measured 0.0069-0.0128 of the frame on the kitchen shots.
NOT_PEOPLE_SHARE = 0.02
#: The owner's words on a shot that say a numbered person is not a person (the shots record's note carries them, as written
#: on the who page): "N = a picture of someone on a cereal box, ignore this" (604-1009, answers.jsonl line 1033).
NOT_A_PERSON = re.compile(r"^\s*(\d+)\s*=\s*(.*\b(?:ignore|not a person|a picture of)\b.*)$", re.I | re.M)
#: A who-page line as the shots record's note carries it.
NOTE_LINE = re.compile(r"the owner on the who page \(frame (\d+), answers\.jsonl line (\d+), ([^)]*)\): (.*?)(?= \| |$)", re.S)


def not_a_person(shot: dict, pages: list[dict]) -> list[dict]:
    """The numbered people the owner's words in the shot's note say are not people, each with the box it had on the page
    they answered on: the newest who page (`pages`, page.json) built before the words whose preview covers the frame."""
    out = []
    for frame, line, when, words in NOTE_LINE.findall(str(shot.get("note") or "")):
        for number, said in NOT_A_PERSON.findall(words):
            at = when.replace(" ", "T")
            seen = [p for p in pages if str(p.get("built", "")) <= at + "Z" and any(int(v["frames"][0]) <= int(frame) <= int(v["frames"][1]) for v in p["previews"])]
            if not seen:
                out.append({"person": int(number), "words": said.strip(), "line": int(line), "box": None, "why_no_box": "no who page built before the words covers the frame"})
                continue
            page = max(seen, key=lambda p: p["built"])
            v = next(v for v in page["previews"] if int(v["frames"][0]) <= int(frame) <= int(v["frames"][1]))
            box = None
            for t in sorted(Path(v["tables"]).glob(f"{v['job']}_*_shots_*_shots.json")):
                for cut in json.loads(t.read_text()).get("shots") or []:
                    if int(v["frames"][0]) + int(cut["first_frame"]) <= int(frame) <= int(v["frames"][0]) + int(cut["last_frame"]):
                        box = next((corners(p["box"]) for p in cut.get("people") or [] if int(p["person"]) == int(number)), box)
            out.append({"person": int(number), "words": said.strip(), "line": int(line), "page": page.get("page", ""), "preview": v["job"], "box": box})
    return out


def cut_others(cuts: dict, answers: list[dict], first: int) -> dict:
    """One shot-table cut as each tracker of a preview saw it (`cuts`: {subject: (table file name, the cut)}), with the
    owner's answers: the people on its shown frame, which are ours, and the others."""
    any_cut = next(iter(cuts.values()))[1]
    shown = first + int(any_cut["shown_frame"])
    people = {int(p["person"]): p for p in any_cut.get("people") or []}
    ours, how = {}, {}
    for subject, (table, cut) in cuts.items():
        said = [x for x in answers if x.get("subject") == subject and x.get("table") == table and int(x.get("frame", -1)) == shown]
        pick = (cut.get("subject") or {}).get("person")
        if said:
            last = said[-1]
            if last.get("box"):
                best = max(people.values(), key=lambda p: box_overlap(corners(p["box"]), tuple(last["box"])), default=None)
                if best is not None and box_overlap(corners(best["box"]), tuple(last["box"])) >= ANSWER_OVERLAP:
                    ours[int(best["person"])], how[subject] = subject, f"person {int(best['person'])}, by the owner's answer"
                else:
                    how[subject] = "the owner's box matches no numbered person"
            else:
                how[subject] = "the owner: none of the outlined people"
        elif pick is not None:
            ours.setdefault(int(pick), subject)
            state = (cut.get("subject") or {}).get("state", "")
            how[subject] = f"person {int(pick)}, the tracker's {'pick (PICKED: nobody said who)' if state == 'picked' else state}"
        else:
            how[subject] = "the tracker holds nobody here"
    for p in people.values():
        if p.get("held_by_another_tracker") and int(p["person"]) not in ours:
            ours[int(p["person"])] = "another tracker's"
    mine = [corners(people[k]["box"]) for k in ours if k in people]
    others = [{"person": k, "box": [round(v) for v in corners(p["box"])], "share_of_frame": p.get("share_of_frame"),
               "gap_px_to_ours": min((gap_px(corners(p["box"]), m) for m in mine), default=None)}
              for k, p in sorted(people.items()) if k not in ours]
    return {"shown_frame": shown, "people": len(people), "ours": how, "others": others}


def shot_others(shot: dict, preview: dict | None, cast: list[dict], pages: list[dict] | None = None, not_people: list[dict] | None = None) -> dict:
    """The answer for a record shot: its preview's cuts on it, and yes / no / cannot say / no preview. An other the owner's
    words in the shot's note say is not a person (`not_a_person`) counts as no one, matched by place and size to the box
    it had on the page they answered on, and is listed apart with their words."""
    a, b = int(shot["first"]), int(shot["last"])
    if preview is None:
        return {"frames": [a, b], "others_present": "no preview", "why": "no preview handed in covers this shot"}
    by_cut: dict[tuple[int, int], dict] = {}
    for subject, (table_name, table) in preview["tables"].items():
        for cut in table.get("shots") or []:
            lo, hi = preview["first"] + int(cut["first_frame"]), preview["first"] + int(cut["last_frame"])
            if lo <= b and hi >= a and "shown_frame" in cut:      # a cut that numbers nobody on a shown frame says nothing of who is there
                by_cut.setdefault((lo, hi), {})[subject] = (table_name, cut)
    if not by_cut:
        return {"frames": [a, b], "preview": preview["name"], "others_present": "not known", "why": "its preview's tables have no cut on this shot"}
    cuts = [dict(cut_others(c, cast, preview["first"]), frames=[lo, hi]) for (lo, hi), c in sorted(by_cut.items())]
    named = [x for x in not_a_person(shot, pages or []) if x["box"] is not None]
    for c in cuts:
        keep = []
        for o in c["others"]:
            hit = next((x for x in named if box_overlap(tuple(o["box"]), tuple(x["box"])) >= ANSWER_OVERLAP), None)
            if hit:
                c.setdefault("not_people", []).append(dict(o, owners_words=hit["words"], answers_line=hit["line"], answered_on=hit["preview"],
                                                           overlap=round(box_overlap(tuple(o["box"]), tuple(hit["box"])), 3)))
            else:
                keep.append(o)
        c["others"] = keep
        # the video's not_people: a lone small other touching none of ours, in a shot the record names
        rules = [r for r in not_people or [] if any(lo <= a and b <= hi for lo, hi in r.get("shots") or [])]
        if rules and len(c["others"]) == 1:
            o = c["others"][0]
            if (o.get("share_of_frame") or 1) < NOT_PEOPLE_SHARE and (o.get("gap_px_to_ours") is None or o["gap_px_to_ours"] > 0):
                c.setdefault("not_people", []).append(dict(o, owners_words=rules[0]["what"], answers_line=None, answered_on="the plan's not_people",
                                                           by=rules[0].get("by", ""), overlap=None))
                c["others"] = []
    shown = sorted({c["shown_frame"] for c in cuts if a <= c["shown_frame"] <= b})
    with_others = [c for c in cuts if c["others"]]
    n = b - a + 1
    if with_others:
        answer = "yes"
    elif n > LONG_SHOT_FRAMES:
        answer = "cannot say"
    else:
        answer = "no"
    largest = max((o for c in with_others for o in c["others"]), key=lambda o: o.get("share_of_frame") or 0, default=None)
    return {"frames": [a, b], "preview": preview["name"], "others_present": answer, "frames_shown": shown,
            "frames_not_shown": n - len(shown), "frames_with_others": [c["shown_frame"] for c in with_others],
            "most_others_on_a_frame": max((len(c["others"]) for c in cuts), default=0),
            "largest_other": largest, "nearest_gap_px": min((o["gap_px_to_ours"] for c in with_others for o in c["others"] if o["gap_px_to_ours"] is not None), default=None),
            "not_people": [x for c in cuts for x in c.get("not_people") or []],
            # the owner's words that matched no numbered person of this preview: said, never dropped (the lead, 2026-10-14)
            "words_not_carried": [{"person": x["person"], "answers_line": x["line"], "answered_on": x["preview"], "why": "not carried: another frame or place"}
                                  for x in named if not any(n["answers_line"] == x["line"] and n["person"] == x["person"]
                                                            for c in cuts for n in c.get("not_people") or [])],
            "cuts": cuts, "tables_list_every_detection": all("every detection" in str(t.get("people_are", "")) for _, t in preview["tables"].values())}


def read_preview(spec: str) -> dict:
    """`JOB.json=TABLES_DIR`: the preview's job and each subject's shot table in the folder its trackers saved into."""
    job_file, sep, tables = spec.partition("=")
    if not sep:
        raise SystemExit(f"--preview {spec!r}: write JOB.json=TABLES_DIR")
    job = json.loads(Path(job_file).read_text())["job"]
    out = {}
    for s in job["subjects"]:
        found = sorted(Path(tables).glob(f"{job['name']}_{s['label']}_shots_*_shots.json"))
        if found:
            out[s["label"]] = (found[-1].name, json.loads(found[-1].read_text()))
    return {"name": job["name"], "first": int(job["first"]), "last": int(job["first"]) + int(job["frames"]) - 1, "tables": out, "job_file": job_file}


def newest_previews(data: Path, share: Path) -> list[str]:
    """`JOB.json=TABLES_DIR` of the newest finished preview of each shot among `data`'s `<day>_shot_c_<first>[letter]_job`
    folders (the later letter, then the later day), its tables in the folder its job wrote to under `share`."""
    best: dict[str, tuple] = {}
    for folder in sorted(data.glob("*_shot_c_*_job")):
        ran = folder / "preview.queued.json"
        if not (folder / "job.json").is_file() or not ran.is_file() or json.loads(ran.read_text()).get("status") != "success":
            continue
        job = json.loads((folder / "job.json").read_text())["job"]
        base = job["name"].rstrip("abcdefghijklmnopqrstuvwxyz")
        key = (job["name"][len(base):], folder.name[:10])
        if base not in best or key > best[base][0]:
            best[base] = (key, f"{folder / 'job.json'}={share / job['output'] / job['name']}")
    return [spec for _, spec in sorted(best.values(), key=lambda x: x[1])]


def summary(row: dict) -> dict:
    """A shot's answer as a job file keeps it: the answer and its figures, without the cuts."""
    big = row.get("largest_other")
    return ({k: row[k] for k in ("others_present", "preview", "frames_shown", "frames_not_shown", "frames_with_others", "most_others_on_a_frame",
                                 "nearest_gap_px", "why", "not_people", "words_not_carried") if k in row}
            | ({"largest_other_share_of_frame": big.get("share_of_frame")} if big else {}))


def said(row: dict) -> str:
    """A shot's answer in words, for a job's note and its defaults block."""
    x = row["others_present"]
    if x == "yes":
        return (f"yes: someone not ours on {len(row['frames_with_others'])} of the {len(row['frames_shown'])} frame(s) the preview {row['preview']} shows "
                f"({', '.join(map(str, row['frames_with_others']))}), at most {row['most_others_on_a_frame']} on a frame, the nearest {row['nearest_gap_px']} px from ours")
    if x == "cannot say":
        return f"cannot say: nobody else on the frame(s) shown ({', '.join(map(str, row['frames_shown']))}), and {row['frames_not_shown']} frames were not shown"
    if x == "no":
        return f"no: nobody else on the frame(s) the preview {row['preview']} shows ({', '.join(map(str, row['frames_shown']))}), and the shot is short"
    return f"{x}: {row.get('why', '')}"


def of_shot(shot: dict, previews: list[dict]) -> dict | None:
    """The preview that covers the shot and has a table: the last listed, as shot_parts reads them."""
    covering = [p for p in previews if p["tables"] and p["first"] <= shot["first"] and shot["last"] <= p["last"]]
    return covering[-1] if covering else None


def md(rows: list[dict], made: str) -> str:
    lines = [f"# Other people present, a shot at a time", "", made, "",
             "| frames | answer | preview | frames shown | with others | most on a frame | the largest other (share of the frame) | nearest gap to ours (px) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        big = r.get("largest_other")
        lines.append(f"| {r['frames'][0]}-{r['frames'][1]} | {r['others_present']}" + (f" ({r['frames_not_shown']} frames not shown)" if r["others_present"] == "cannot say" else "")
                     + f" | {r.get('preview', '')} | {', '.join(map(str, r.get('frames_shown') or [])) or '-'} | {', '.join(map(str, r.get('frames_with_others') or [])) or '-'}"
                     + f" | {r.get('most_others_on_a_frame', '-')} | {big['share_of_frame'] if big else '-'} | {r.get('nearest_gap_px') if r.get('nearest_gap_px') is not None else '-'} |")
        for x in r.get("words_not_carried") or []:
            lines.append(f"|  | the owner's words on person {x['person']} (answers.jsonl line {x['answers_line']}, on {x['answered_on']}): {x['why']} | | | | | | |")
        for x in r.get("not_people") or []:
            lines.append(f"|  | counted as no one: person {x['person']} at {x['box']} ({x.get('share_of_frame')} of the frame), by "
                         + (f"the owner's words (answers.jsonl line {x['answers_line']}, on {x['answered_on']}, overlap {x['overlap']})" if x.get("answers_line")
                            else f"the plan's not_people ({x.get('by', '')})") + f": \"{x['owners_words']}\" | | | | | | |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--shots", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--plan", help="the video's plan: its `not_people` (what, the shots, who said so) count a lone small other there as no one")
    ap.add_argument("--preview", action="append", default=[], metavar="JOB.json=TABLES_DIR")
    ap.add_argument("--newest-previews-in", metavar="DATA_DIR", help="read the newest finished shot_c_* preview of each shot in this folder, in place of --preview")
    ap.add_argument("--share", help="the server's output folder the previews wrote to (with --newest-previews-in)")
    ap.add_argument("--who-page", action="append", default=[], metavar="PAGE.json",
                    help="a who page's page.json (bench/who_page.py): where the owner's words in a shot's note name a numbered person as not a person, "
                         "the page they answered on gives that person's box; repeatable")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args(argv)
    for out in (a.json, a.md):
        if Path(out).exists():
            raise SystemExit(f"refused: {out} is there; a kept output is not replaced")
    shots = json.loads(Path(a.shots).read_text())
    cast = [json.loads(line) for line in open(a.cast) if line.strip()]
    if a.newest_previews_in:
        if a.preview or not a.share:
            raise SystemExit("--newest-previews-in takes --share and no --preview")
        a.preview = newest_previews(Path(a.newest_previews_in), Path(a.share))
    previews = [read_preview(p) for p in a.preview]
    pages = []
    for x in a.who_page:
        page = json.loads(Path(x).read_text())
        if isinstance(page.get("previews"), list):
            pages.append(dict(page, page=x))
    plan = json.loads(Path(a.plan).read_text()) if a.plan else {}
    rows = [shot_others(dict(s, first=int(s["first"]), last=int(s["last"])), of_shot(dict(s, first=int(s["first"]), last=int(s["last"])), previews), cast, pages,
                        plan.get("not_people")) for s in shots["shots"]]
    made = (f"Made {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} by bench/others_present.py from {a.shots}, {a.cast} and "
            f"{len(previews)} preview(s); a shot longer than {LONG_SHOT_FRAMES} frames with nobody else on its shown frames is `cannot say` "
            "(who_page.LONG_SHOT_FRAMES, a starting value). Figures of the previews' detections, not of the pixels.")
    Path(a.json).write_text(json.dumps({"made": made, "rows": rows}, indent=1) + "\n")
    Path(a.md).write_text(md(rows, made))
    counts = {k: sum(1 for r in rows if r["others_present"] == k) for k in ("yes", "no", "cannot say", "no preview")}
    print(f"wrote {a.json} and {a.md}: {len(rows)} shots, " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
