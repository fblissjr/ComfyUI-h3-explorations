#!/usr/bin/env python3
"""The masked lane's scoreboard: every judged shot of every window, each render's metrics beside the eye's verdicts.

One row per render per shot: what the turn metric says, what the motion
metric says, and what a person said, with who and on what. It runs no model
and reads no media: the poses come from `bench/measure_subject_yaw.py`'s
records, the verdicts and the provenance of every clip from
`bench/turn_metric_eye_verdicts.json`.

**What it buys** (card `build-metric-suite`): one page that answers "which
arm holds the turn, follows the motion and keeps the look, on which clip",
for as many clips as the verdicts file names, so a choice between two arms
is not made on the one clip somebody happened to look at. It is accepted
when the owner reads an arm's row and it says what they saw.

**What goes in.** The verdicts file's `windows`: a stretch of a source clip
with where it came from, what it may be used for, and `poses`, the pose
record made from it. Its `sets`: one shot of a window, judged by a named
reader. A window with no `poses` is listed as not measured, with the
command that would make the record. A shot the source has and nobody judged
is measured and shown with no eye column.

**What comes out, per shot and per render:**
- turn: the end facing's distance from the source's in degrees, for the
  shoulders and for the head, the chin lift's distance beside them, and
  the verdict, which is the shoulders' at the tolerance
  (`measure_subject_yaw.py::TOLERANCE_DEGREES`).
- motion: `followed` and `in_step` over the joints the source moves, the
  best time shift, `stray`, and the verdict at
  `measure_subject_motion.py::FOLLOWS`. The parts the source moves are
  named, since a score on the arms and a score on the whole body are
  different claims.
- look: the eye's verdict only. No look metric exists yet.
- per shot, per metric: whether the metric orders the renders as the eye
  did, and how many of its verdicts match where the eye said yes or no.

**What it cannot say.** A shot with one verdict from the eye orders
nothing: agreement there is on the verdict alone. Every limit of the two
metrics carries over; this adds none and removes none.

    <python> bench/run_lane_benchmark.py --out data/results/<date>_lane_scoreboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import measure_subject_motion as M  # noqa: E402
import measure_subject_yaw as Y  # noqa: E402

VERDICTS = HERE / "turn_metric_eye_verdicts.json"
# which of a metric's verdicts an eye's yes and no stand for; a partial matches neither
MATCHES = {"turn": {"yes": "holds", "no": "fails"}, "motion": {"yes": "follows", "no": "does not follow"}}


def agreement(field: str, rows: dict[str, dict], measured: dict[str, float | None]) -> dict | None:
    """How a metric stands against the eye on one shot: the order, and the verdicts where the eye said yes or no."""
    eye = {label: row[field]["eye"] for label, row in rows.items() if row[field].get("eye")}
    if not eye:
        return None
    order = Y.ranks_as_the_eye(measured, eye)
    firm = {label: v for label, v in eye.items() if v in MATCHES[field]}
    same = [label for label, v in firm.items() if rows[label][field].get("verdict") == MATCHES[field][v]]
    return {
        "judged": len(eye),
        "eye_verdicts": sorted(set(eye.values()), key=lambda v: Y.ORDER[v]),
        "orders_as_the_eye": order["same_order"] if len(set(eye.values())) > 1 else None,
        "out_of_order": order["disagreements"],
        "verdicts_matching": len(same), "of_yes_or_no": len(firm),
        "verdicts_not_matching": sorted(set(firm) - set(same)),
    }


def read_shot(record: dict, shot: list[int], judged: dict | None) -> dict:
    """One shot of a pose record: every render's turn and motion, with the eye's verdicts where a set gives them."""
    clips = (judged or {}).get("clips", {})
    eye = {field: {label: row[field] for label, row in clips.items() if row.get(field)} for field in ("turn", "motion", "look")}
    turn = Y.judge(record, shot, eye["turn"], Y.TOLERANCE_DEGREES)
    motion = M.read_shot(record, shot, eye["motion"], curves=False, say=lambda *_a: None)
    rows = {}
    for label in record["clips"]:
        t, m = turn["clips"][label], motion["clips"][label]["in_body"]
        rows[label] = {
            "turn": {"end_difference": t.get("end_difference"), "largest_turn": t.get("largest_turn"),
                     "mean_difference": t.get("mean_difference"), "verdict": t["verdict"],
                     "head_end_difference": t["head"].get("end_difference"),
                     "head_mean_difference": t["head"].get("mean_difference"),
                     "end_lift_source": t["head"].get("end_lift_source"), "end_lift": t["head"].get("end_lift"),
                     "mean_lift_difference": t["head"].get("mean_lift_difference")},
            "motion": {"followed": m["followed"], "in_step": m["in_step"],
                       "in_step_parts": {p: v["in_step"] for p, v in m["parts"].items() if v["moved"]},
                       "best_shift_frames": m.get("best_shift_frames"),
                       "followed_at_best_shift": m.get("followed_at_best_shift"), "stray": m["stray"],
                       "parts": {p: v["followed"] for p, v in m["parts"].items() if v["moved"]},
                       "verdict": m["verdict"]},
            "look": {},
        }
        for field in ("turn", "motion", "look"):
            if label in eye[field]:
                rows[label][field]["eye"] = eye[field][label]
    first = next(iter(motion["clips"].values()), None)
    out = {
        "shot": list(shot), "samples": turn["samples"],
        "source": {"largest_turn": turn["source"]["largest_turn"],
                   "moved_parts": sorted({p for p, v in first["in_body"]["parts"].items() if v["moved"]}) if first else []},
        "rows": rows,
        "against_the_eye": {
            "turn": agreement("turn", rows, {label: r["turn"]["end_difference"] for label, r in rows.items()}),
            "motion": agreement("motion", rows, {label: (None if r["motion"]["followed"] is None
                                                         else round(1.0 - r["motion"]["followed"], 3))
                                                 for label, r in rows.items()}),
        },
    }
    if judged:
        out["what_happens"] = judged.get("what_happens")
        out["judged"] = judged.get("judged")
    return out


def build(spec: dict, only: str | None = None) -> dict:
    """The scoreboard for every window the verdicts file names."""
    board = {"script": "bench/run_lane_benchmark.py", "tolerance_degrees": Y.TOLERANCE_DEGREES,
             "follows_at": M.FOLLOWS, "windows": {}}
    for name, window in spec["windows"].items():
        if only and only != name:
            continue
        entry = {key: window.get(key) for key in ("file", "start_second", "rate", "frames", "provenance", "permission", "poses")}
        sets = {set_name: s for set_name, s in spec["sets"].items() if s.get("window") == name}
        poses = REPO / window["poses"] if window.get("poses") else None
        if poses is None or not poses.is_file():
            entry["not_measured"] = ("no pose record is named for this window" if poses is None
                                     else f"{window['poses']} is not on disk")
            entry["sets_waiting"] = sorted(sets)
            board["windows"][name] = entry
            continue
        record = json.loads(poses.read_text())
        entry["subject_box"] = record.get("subject_box", "the kept mask's")
        entry["shots"] = []
        covered = []
        for set_name, judged in sets.items():
            shot = read_shot(record, judged["shot"], judged)
            shot["set"] = set_name
            entry["shots"].append(shot)
            covered.append(list(judged["shot"]))
        for shot in record.get("shots") or []:
            if list(shot) not in covered:
                entry["shots"].append(read_shot(record, list(shot), None))
        entry["shots"].sort(key=lambda s: s["shot"])
        board["windows"][name] = entry
    return board


def _n(value, digits: int = 2) -> str:
    if value is None:
        return ""
    text = f"{value:.{digits}f}"
    return text.lstrip("-") if float(text) == 0 else text      # no "-0.00"


def as_markdown(board: dict, title: str) -> str:
    lines = [f"# {title}", "", "lane: masked", "",
             "Generated by `bench/run_lane_benchmark.py` from the pose records and "
             "`bench/turn_metric_eye_verdicts.json`. Every figure is a distance from the source, in degrees: where "
             "the shoulders face at the end of the shot, where the head faces at the end, and how far the chin's "
             f"lift is from the source's on average. The turn verdict is the shoulders', holds at or under "
             f"{board['tolerance_degrees']}. Motion: `followed` over the joints the source moves, 0 for a subject who "
             f"never moves, follows at or above {board['follows_at']}; `in step` is the same motion at any size, "
             "1 the source's, 0 unrelated or still, and gives no verdict. An empty eye cell is a render nobody "
             "judged on that question. The best time shift, `stray` and every part are in the `.json`.", ""]
    for name, window in board["windows"].items():
        lines += [f"## {name}: `{window['file']}` from {window['start_second']} s", "",
                  f"- provenance: {window['provenance']}", f"- permission: {window['permission']}"]
        if window.get("not_measured"):
            lines += [f"- **not measured**: {window['not_measured']}", ""]
            continue
        lines += [f"- poses: `{window['poses']}`; the subject's box is {window['subject_box']}", ""]
        for shot in window["shots"]:
            a, b = shot["shot"]
            head = f"### frames {a} to {b}" + (f" (`{shot['set']}`)" if shot.get("set") else " (not judged)")
            lines += [head, ""]
            if shot.get("what_happens"):
                lines += [shot["what_happens"].capitalize() + ".", ""]
            lines += [f"The source's largest turn here is {_n(shot['source']['largest_turn'], 1)} degrees and it moves: "
                      + (", ".join(shot["source"]["moved_parts"]) or "nothing past the floor") + ".", ""]
            if shot.get("judged"):
                j = shot["judged"]
                lines += [f"Judged by {j['by']} on {j['on']}." + (f" {j['blind'].capitalize()}." if j.get("blind") else ""), ""]
            lines += ["| render | shoulders, end | head, end | chin lift, mean | turn | eye | followed | in step | hands in step "
                      "| motion | eye | look, eye |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for label, row in shot["rows"].items():
                t, m = row["turn"], row["motion"]
                lines.append(f"| `{label}` | {_n(t['end_difference'], 1)} | {_n(t['head_end_difference'], 1)} | "
                             f"{_n(t['mean_lift_difference'], 1)} | {t['verdict']} | {t.get('eye', '')} | "
                             f"{_n(m['followed'])} | {_n(m['in_step'])} | {_n(m['in_step_parts'].get('hands'))} | "
                             f"{m['verdict']} | {m.get('eye', '')} | {row['look'].get('eye', '')} |")
            lines.append("")
            for field in ("turn", "motion"):
                against = shot["against_the_eye"][field]
                if not against:
                    continue
                order = {True: "orders the renders as the eye did", False: "does NOT order the renders as the eye did",
                         None: "has nothing to order (the eye gave one verdict)"}[against["orders_as_the_eye"]]
                said = (f"{against['verdicts_matching']} of {against['of_yes_or_no']} verdicts match where the eye said yes or no"
                        if against["of_yes_or_no"] else "the eye gave no yes or no to match")
                wrong = f" (not matching: {', '.join(against['verdicts_not_matching'])})" if against["verdicts_not_matching"] else ""
                lines.append(f"- {field}: {order}; {said}{wrong}.")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--window", help="only this window")
    ap.add_argument("--title", default="The masked lane's scoreboard")
    ap.add_argument("--out", required=True, type=Path, help="path without a suffix; .md and .json are written")
    args = ap.parse_args()
    board = build(json.loads(args.verdicts.read_text()), args.window)
    board["verdicts"] = args.verdicts.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(board, indent=1) + "\n")
    args.out.with_suffix(".md").write_text(as_markdown(board, args.title))
    for name, window in board["windows"].items():
        if window.get("not_measured"):
            print(f"{name}: not measured, {window['not_measured']}")
            continue
        for shot in window["shots"]:
            got = {f: a and (a["orders_as_the_eye"], f"{a['verdicts_matching']}/{a['of_yes_or_no']}")
                   for f, a in shot["against_the_eye"].items()}
            print(f"{name} {shot['shot']}: {len(shot['rows'])} render(s), against the eye {got}")
    print(f"wrote {args.out.with_suffix('.md').name} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
