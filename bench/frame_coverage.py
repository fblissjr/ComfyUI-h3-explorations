#!/usr/bin/env python3
"""Per frame, per shot and subject: is the person tracked, which laid piece covers the frame, and did that piece's run
carry a mask for her there; otherwise the original shows. Three lists out.

    <python> bench/frame_coverage.py --table data/<...>/t_<...>.json [--span A-B] [--json OUT.json] [--md OUT.md]

**Why it exists** (the owner's order of 2026-10-16, `docs/wiki/decisions.md`). In the kitchen a person laid as swapped
turned back into the original halfway through the shot, and nothing in the ledger or the pages said so: the table
counted a gap against the whole shot, not against the frames she is in, two renders of one person meeting were not
flagged, and the ledger read only the frames a person's `drop=` emptied, not the floor's or the grade's. The method is
the flip survey's of 2026-10-15 (the part model plus our floor emptied every flip it found), ported, not imported.

**What it reads.** The delivery table's JSON: its `reading` (each shot and subject in it) and its `rows` (the laid
pieces, each tied to its capture's run by `run=<capture>:<run>`). For each frame of each shot and subject:

- tracked: the SAM 3.1 track's px for her on that frame, from `subjects/<label>/masks__<by>.npz` of a capture a row of
  that shot is tied to (the piece's own capture first, then any other of the shot's, since a capture carries every
  subject's track over its own frames). Where the tied captures cannot tell, the run says why ("no tied capture
  covers this frame", or one covers it and holds no track of her there) and "tracking not known", never "not
  tracked", and names any capture of the same clip under `data/` that no row ties to and that holds a track of her
  over those frames: a capture that could tell, not read (the brief reads the captures the rows tie to);
- a shot the table's reading calls `unsure` is read like a `yes` one, and its runs say `present`;
- covered: the laid piece for her on that frame, if any;
- carried: the px of the mask the node was given for her on that frame (`runs/<run>/region.npz` `carried`), with
  the capture manifest's held frames (filled from a neighbouring frame);
- otherwise the original shows, and inside a piece, why its run carried nothing there (`emptied`).

**The three lists**, each a run of frames with the shot, subject and piece:

1. the original shows while she is tracked (track px > 0), with or without a piece;
2. seams: two different laid renders of one person meeting on one shot, with both render days;
3. frames emptied inside a laid piece while she is tracked (part of 1), by cause: `drop=` (a person's word), floor
   (the grade's "too small"), grade (the grade emptied it for another reason, its own `why` in `part_grades.json`), nothing saved (the saved part was empty), the run's
   own no-mask list (`left_as_the_source_with_no_mask`, an effect the run recorded, not a cause), a cut; "not in the
   record" where none of those says. Every cause that holds is said, joined by `+`.

**What it cannot see.** The picture. A frame whose mask was carried can still come out looking like the original (H3
may regenerate a face like the source's); deciding that wants a per-frame likeness of the delivered video, which no
record holds. Track px > 0 says SAM 3.1 put some pixels on her, not that her face is visible: a run's px are printed
so a reader sees "barely in". A two-subject run's `carried` is both subjects' union. A frame the capture did not read
is said as that. Nothing here is a threshold of degree; green or red only for a row whose `run=` does not resolve or
a file it names that is missing.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from piece_gaps import day, runs  # noqa: E402

#: set bits of a byte, to count a packed mask's px without unpacking it
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
CANNOT = ("the picture: a frame whose mask was carried can still come out as the original (no per-frame likeness of the delivered "
          "video is read); whether a tracked person's face is visible (track px > 0 is SAM 3.1's pixels on her, printed so 'barely in' "
          "shows; whether her face shows on such frames is a playback question); a two-subject run's carried mask is both subjects' union; a face-and-hair run whose face emptied while its hair was carried counts as carried (the px are the union); a restore= that gives the source back at assembly inside a carried mask is not read; a subject the shots file says is not in a shot is not walked (the table's not_in_the_shot); an older capture's subject named by an alias is not matched as untied; a frame the tied captures cannot tell is 'tracking not known' with why, and an untied capture holding her track is named, never read")


def px(packed: np.ndarray) -> np.ndarray:
    """Px a frame of a packed bool mask [n, h, ceil(w/8)]."""
    return POP[packed.reshape(len(packed), -1)].sum(1, dtype=np.int64)


def rel(path: Path) -> str:
    """A path from the repo root when it is under it, else whole (a check's temporary folder)."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def frames_of(spans) -> set[int]:
    """A manifest's frame list, as spans [[a, b], ...] or single frames."""
    out: set[int] = set()
    for x in spans or []:
        if isinstance(x, (list, tuple)):
            out.update(range(int(x[0]), int(x[1]) + 1))
        else:
            out.add(int(x))
    return out


class Failure(Exception):
    """A hard failure: a row whose `run=` does not resolve, or a file it names that is missing."""


class Capture:
    """One capture folder's records, read once."""

    def __init__(self, folder: Path):
        if not (folder / "manifest.json").is_file():
            raise Failure(f"{folder}: no manifest.json")
        self.folder = folder
        self.m = json.loads((folder / "manifest.json").read_text())
        self.first, self.n = int(self.m["first_frame"]), int(self.m["frames"])
        self._track: dict = {}
        self._run: dict = {}

    def sighting(self, label: str) -> dict | None:
        s = next((s for s in self.m.get("subjects", []) if s["label"] == label), None)
        if s is None:
            return None
        return next((x for x in s["sightings"] if (self.folder / "subjects" / label / f"masks__{x['by']}.npz").is_file()), s["sightings"][0])

    def track(self, label: str) -> dict | None:
        """Her SAM 3.1 track px and covered flags on this capture's frames, and the saved part's px; None when the
        capture has no masks file for her."""
        if label not in self._track:
            seen = self.sighting(label)
            path = self.folder / "subjects" / label / f"masks__{seen['by']}.npz" if seen else None
            if path is None or not path.is_file():
                self._track[label] = None
            else:
                z = np.load(path)
                self._track[label] = {"px": px(z["track"]), "covered": z["covered"].astype(bool),
                                      "part_px": px(z["parts"]) if "parts" in z.files else None, "file": rel(path)}
        return self._track[label]

    def track_at(self, label: str, f: int) -> int | None:
        t = self.track(label)
        k = f - self.first
        if t is None or not 0 <= k < self.n or not t["covered"][k]:
            return None
        return int(t["px"][k])

    def run(self, name: str) -> dict:
        if name not in self._run:
            entry = next((r for r in self.m.get("runs", []) if r["name"] == name), None)
            if entry is None:
                raise Failure(f"{self.folder.name}: no run {name!r} in its manifest")
            rz = self.folder / "runs" / name / "region.npz"
            if not rz.is_file():
                raise Failure(f"{rz}: missing")
            z = np.load(rz)
            self._run[name] = {"entry": entry, "carried_px": px(z["carried"]), "read": z["read"].astype(bool),
                               "file": rel(rz)}
        return self._run[name]

    def grades(self, label: str) -> dict[int, dict]:
        p = self.folder / "subjects" / label / "part_grades.json"
        return {int(g["source_frame"]): g for g in json.loads(p.read_text())["frames"]} if p.is_file() else {}


def why_emptied(cap: Capture, run: dict, label: str, f: int, grades: dict[int, dict]) -> str:
    """Why a run carried no mask for her on frame f, every cause the records give, in the order the chain makes the
    part; "not in the record" when none says."""
    seen = cap.sighting(label) or {}
    entry = run["entry"]
    causes = []
    if f in frames_of(entry.get("left_as_the_source_across_a_cut")):
        causes.append("a cut")
    if f in frames_of(seen.get("parts_emptied_on_source_frames")):
        causes.append("drop=")
    g = grades.get(f)
    if g and g.get("did") == "emptied":
        causes.append("floor" if str(g.get("why") or "").startswith("too small") else "grade")
    elif f in frames_of(seen.get("parts_emptied_by_grade_on_source_frames")):
        causes.append("floor or grade (part_grades.json does not say which)")
    t = cap.track(label)
    k = f - cap.first
    if t is not None and t["part_px"] is not None and 0 <= k < cap.n and t["part_px"][k] == 0:
        causes.append("nothing saved")
    if f in frames_of(entry.get("left_as_the_source_with_no_mask")):
        causes.append("the run's no-mask list")
    return "+".join(causes) or "not in the record"


def emptied(cap: Capture, run_name: str, label: str, span: list[int]) -> dict:
    """The frames inside `span` on which the run carried no mask for `label`, each with its cause; the rule the ledger's
    `left_by_the_load` reads too (one home). Frames the capture did not read are listed apart, not as emptied."""
    run = cap.run(run_name)
    grades = cap.grades(label)
    out, unread = {}, []
    for f in range(max(span[0], cap.first), min(span[1], cap.first + cap.n - 1) + 1):
        k = f - cap.first
        if not run["read"][k]:
            unread.append(f)
        elif run["carried_px"][k] == 0:
            out[f] = why_emptied(cap, run, label, f, grades)
    return {"frames": out, "not_read": unread, "read_from": run["file"]}


def resolve(row: dict, caps: dict[str, Capture]) -> tuple[Capture, str]:
    if not row.get("run") or ":" not in row["run"]:
        raise Failure(f"row {row.get('name')} {row.get('first')}-{row.get('last')}: no run= to tie it to a capture")
    folder, name = row["run"].rsplit(":", 1)
    path = (REPO / folder).resolve()
    if folder not in caps:
        if not path.is_dir():
            raise Failure(f"row {row.get('name')}: run={row['run']} names no folder")
        caps[folder] = Capture(path)
    cap = caps[folder]
    cap.run(name)
    return cap, name


def untied_index(clips: set[str], tied: set[str], data: Path) -> list[tuple[str, int, int, set[str]]]:
    """Every capture of these clips under data/ that no row ties to: (folder, first, last, subjects with a masks file).
    Named beside a run no tied capture can tell, as a capture that could; never read for the lists (the brief: the
    captures the rows tie to)."""
    out = []
    for mp in sorted(data.glob("*/manifest.json")):
        if mp.parent.name in tied:
            continue
        try:
            m = json.loads(mp.read_text())
            if m.get("clip") not in clips:
                continue
            first, n = int(m["first_frame"]), int(m["frames"])
        except (ValueError, KeyError, TypeError):
            continue
        labels = {s["label"] for s in m.get("subjects", []) if any((mp.parent / "subjects" / s["label"]).glob("masks__*.npz"))}
        out.append((mp.parent.name, first, first + n - 1, labels))
    return out


def written_by_render(table: dict) -> dict[str, str]:
    out = {}
    for r in table.get("reading", []):
        for s in r.get("stands", []):
            if s.get("written"):
                out[s["render"]] = s["written"]
    return out


def coverage(table: dict, span: list[int] | None = None, data: Path = REPO / "data") -> dict:
    caps: dict[str, Capture] = {}
    failures = []
    tied, failed = [], []
    for row in table["rows"]:
        try:
            cap, name = resolve(row, caps)
            tied.append((row, cap, name))
        except Failure as e:
            failures.append(str(e))
            failed.append(row)
    written = written_by_render(table)
    untied = untied_index({c.m.get("clip") for c in caps.values()}, {c.folder.name for c in caps.values()}, data)
    shots, list1, list2, unknown, empty, carried_unknown = [], [], [], [], [], []
    for rd in table.get("reading", []):
        a, b, label = int(rd["first"]), int(rd["last"]), rd["subject"]
        if span and (b < span[0] or a > span[1]):
            continue
        present = rd.get("present")                                 # "yes" or "unsure": both are read, and said
        lo, hi = (max(a, span[0]), min(b, span[1])) if span else (a, b)
        in_shot = [(r, c, n) for r, c, n in tied if r["last"] >= a and r["first"] <= b]
        mine = [(r, c, n) for r, c, n in in_shot if r["subject"] == label]
        failed_mine = [r for r in failed if r.get("subject") == label and r["last"] >= a and r["first"] <= b]
        empt = {}
        for r, c, n in mine:
            e = emptied(c, n, label, [max(r["first"], lo), min(r["last"], hi)])
            empt[id(r)] = e
        cols = {"frames": [], "track_px": [], "track_from": [], "piece": [], "carried_px": [], "state": []}
        for f in range(lo, hi + 1):
            piece = next(((r, c, n) for r, c, n in mine if r["first"] <= f <= r["last"]), None)
            # the piece's capture, then the shot's, then any capture a row of another shot ties to that reaches this frame
            order = ([piece] if piece else []) + [x for x in in_shot if x is not piece] + [x for x in tied if x not in in_shot]
            tpx, tfrom = None, None
            for r, c, n in order:
                v = c.track_at(label, f)
                if v is not None:
                    tpx, tfrom = v, c.folder.name
                    break
            if tpx is None:
                tfrom = ("no tied capture covers this frame" if not any(c.first <= f < c.first + c.n for _, c, _ in order)
                         else "a tied capture covers this frame and holds no track of her on it")
            carried = None
            bad = next((r for r in failed_mine if r["first"] <= f <= r["last"]), None)
            if piece is None and bad is not None:
                state = f"not known: the laid piece {bad.get('name')} has a run= that does not resolve"
            elif piece is None:
                state = "original: no piece laid"
            else:
                r, c, n = piece
                run = c.run(n)
                k = f - c.first
                if not 0 <= k < c.n:
                    state = "not known: the piece's capture does not cover this frame"
                elif not run["read"][k]:
                    state = "not known: the capture did not read this frame of the run"
                else:
                    carried = int(run["carried_px"][k])
                    if carried == 0:
                        state = "original: emptied inside the piece (" + empt[id(r)]["frames"].get(f, "not in the record") + ")"
                    elif f in frames_of((c.sighting(label) or {}).get("parts_held_on_source_frames")):
                        state = "carried (held from a neighbouring frame)"
                    else:
                        state = "carried"
            for key, v in (("frames", f), ("track_px", tpx), ("track_from", tfrom), ("piece", piece[0]["name"] if piece else None),
                           ("carried_px", carried), ("state", state)):
                cols[key].append(v)
        shots.append({"shot": [a, b], "subject": label, "present": present, **cols})
        # list 1 and its side lists: runs of equal (piece, state) where the original shows
        cur = None

        def close():
            if cur is None:
                return
            pxs = [x for x in cur["px"] if x is not None]
            item = {"shot": [a, b], "subject": label, "present": present, "frames": [cur["f0"], cur["f1"]], "count": cur["f1"] - cur["f0"] + 1,
                    "piece": cur["piece"], "shot_has_a_piece_of_her": bool(mine), "state": cur["state"], "track_from": sorted(cur["from"])}
            if cur["kind"] == "tracked":
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)}
                list1.append(item)
            elif cur["kind"] == "unknown":
                item["tracking"] = ("not known: " + cur["state"].removeprefix("not known: ")) if cur["state"].startswith("not known") \
                    else f"{cur['why']}: tracking not known"
                item["untied_captures_with_her_track"] = [n for n, a_, b_, labels in untied
                                                          if label in labels and a_ <= cur["f1"] and b_ >= cur["f0"]]
                unknown.append(item)
            elif cur["kind"] == "untracked":
                empty.append(item)
            elif cur["kind"] == "carried_not_known":
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)}
                carried_unknown.append(item)

        for i, f in enumerate(cols["frames"]):
            st, tpx = cols["state"][i], cols["track_px"][i]
            if st.startswith("not known"):
                kind = "unknown" if tpx is None else "carried_not_known"   # her track known; the piece's mask is not
            elif not st.startswith("original"):
                kind = None
            elif tpx is None:
                kind = "unknown"
            elif tpx > 0:
                kind = "tracked"
            else:
                kind = "untracked"
            why = cols["track_from"][i] if tpx is None else None
            key = (kind, cols["piece"][i], st, why)
            if cur is not None and cur["key"] == key and f == cur["f1"] + 1:
                cur["f1"] = f
                cur["px"].append(tpx)
                if cols["track_from"][i] and tpx is not None:
                    cur["from"].add(cols["track_from"][i])
            else:
                close()
                cur = None if kind is None else {"key": key, "kind": kind, "f0": f, "f1": f, "piece": cols["piece"][i], "state": st, "why": why,
                                                 "px": [tpx], "from": set() if tpx is None else {cols["track_from"][i]} - {None}}
        close()
        # list 2: two different laid renders of her meeting inside this shot
        laid = sorted((r for r, _, _ in mine), key=lambda r: r["first"])
        for x, y in zip(laid, laid[1:]):
            if y["first"] == x["last"] + 1 and x["file"] != y["file"] and lo <= x["last"] and y["first"] <= hi:
                list2.append({"shot": [a, b], "subject": label, "at": [x["last"], y["first"]],
                              "before": {"piece": x["name"], "written": written.get(x["name"]), "day": day(written.get(x["name"]))},
                              "after": {"piece": y["name"], "written": written.get(y["name"]), "day": day(written.get(y["name"]))}})
    list3 = [x for x in list1 if x["piece"] is not None]
    return {"failures": failures, "shots": shots, "original_while_tracked": list1, "seams": list2,
            "emptied_inside_a_piece": list3, "original_tracking_not_known": unknown, "original_her_track_empty": empty,
            "carried_not_known": carried_unknown, "runs": flat(list1, list2, unknown, empty, written, carried_unknown)}


def flat(list1: list[dict], list2: list[dict], unknown: list[dict], empty: list[dict], written: dict[str, str],
         carried_unknown: list[dict] | None = None) -> list[dict]:
    """One list of runs for a page, by frame (the lead's shape, 2026-10-16). The kinds are disjoint: a run of list 1
    inside a piece is `emptied_inside`, one with no piece `original_while_tracked` (`shot_has_a_piece_of_her` tells a
    turn-back inside a shot she has a piece in from a shot with no piece of her at all); a seam's first and last are
    the two frames that meet; `original_tracking_not_known` is a run the tied captures cannot tell, said so it is not
    read as none, with any untied capture that holds her track; `original_her_track_empty` is the original on frames
    her tied track is 0 px; `carried_not_known` is a run her track is known on and whether the laid piece carried a
    mask is not (its run= does not resolve, its capture does not cover the frame, or did not read it)."""
    out = []
    for x in list1:
        inside = x["piece"] is not None
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
                    "kind": "emptied_inside" if inside else "original_while_tracked",
                    "pieces": [x["piece"]] if inside else [], "render_days": [day(written.get(x["piece"]))] if inside else [],
                    "cause": x["state"].split("(", 1)[1].rstrip(")") if inside else "no piece laid", "track_px_median": x["track_px"]["median"],
                    "shot_has_a_piece_of_her": x["shot_has_a_piece_of_her"], "present": x["present"]})
    for x in list2:
        out.append({"first": x["at"][0], "last": x["at"][1], "shot": x["shot"], "subject": x["subject"], "kind": "seam",
                    "pieces": [x["before"]["piece"], x["after"]["piece"]], "render_days": [x["before"]["day"], x["after"]["day"]],
                    "cause": None, "track_px_median": None})
    for x in unknown:
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
                    "kind": "original_tracking_not_known", "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
                    "cause": x["tracking"], "track_px_median": None, "untied_captures_with_her_track": x["untied_captures_with_her_track"],
                    "present": x["present"]})
    for x in empty:
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
                    "kind": "original_her_track_empty", "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
                    "cause": x["state"], "track_px_median": 0, "present": x["present"]})
    for x in carried_unknown or []:
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
                    "kind": "carried_not_known", "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
                    "cause": x["state"], "track_px_median": x["track_px"]["median"], "present": x["present"]})
    return sorted(out, key=lambda r: (r["first"], r["subject"], r["kind"]))


def counts(out: dict) -> dict:
    by_cause: dict[str, int] = {}
    for x in out["emptied_inside_a_piece"]:
        c = x["state"].split("(", 1)[1].rstrip(")")
        by_cause[c] = by_cause.get(c, 0) + x["count"]
    untracked = sum(x["count"] for x in out["original_her_track_empty"])
    nopiece = [x for x in out["original_while_tracked"] if x["piece"] is None]
    return {"original_while_tracked": {"runs": len(out["original_while_tracked"]), "frames": sum(x["count"] for x in out["original_while_tracked"])},
            "of_which_no_piece_laid": {"in_a_shot_with_a_piece_of_her": sum(x["count"] for x in nopiece if x["shot_has_a_piece_of_her"]),
                                       "in_a_shot_with_no_piece_of_her": sum(x["count"] for x in nopiece if not x["shot_has_a_piece_of_her"])},
            "seams": len(out["seams"]),
            "emptied_inside_a_piece": {"runs": len(out["emptied_inside_a_piece"]), "frames": sum(x["count"] for x in out["emptied_inside_a_piece"]),
                                       "frames_by_cause": by_cause},
            "original_tracking_not_known_frames": sum(x["count"] for x in out["original_tracking_not_known"]),
            "original_and_her_track_empty_frames": untracked}


def fr(x) -> str:
    return f"{x[0]}-{x[1]}" if x[0] != x[1] else str(x[0])


def markdown(out: dict) -> str:
    c = out["counts"]
    lines = [f"# Frame coverage: {out['table']}", "", out["made"], "", f"Command: `{out['command']}`", "",
             f"Measured on: {out['measured_on']}", "", f"Cannot see: {out['cannot_see']}", ""]
    if out["failures"]:
        lines += ["## FAILED: rows whose run= does not resolve, or a file missing", ""] + [f"- {x}" for x in out["failures"]] + [""]
    lines += ["## Counts", "", "```", json.dumps(c, indent=1), "```", "",
              "## 1. The original shows while she is tracked", "",
              "| shot | subject | frames | count | piece | a piece of her in the shot | state | track px least / median / most | track read from |",
              "|---|---|---|---|---|---|---|---|---|"]
    for x in out["original_while_tracked"]:
        t = x["track_px"]
        lines.append(f"| {fr(x['shot'])} | {x['subject']}{'' if x['present'] == 'yes' else ' (' + str(x['present']) + ')'} | {fr(x['frames'])} | "
                     f"{x['count']} | {x['piece'] or '(none laid)'} | {'yes' if x['shot_has_a_piece_of_her'] else 'no'} | {x['state']} | "
                     f"{t['least']} / {t['median']} / {t['most']} | {', '.join(x['track_from'])} |")
    lines += ["", "## 2. Seams: two laid renders of one person meeting", "", "| shot | subject | at | before (rendered) | after (rendered) |", "|---|---|---|---|---|"]
    for x in out["seams"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {x['at'][0]}/{x['at'][1]} | {x['before']['piece']} ({x['before']['day']}) | "
                     f"{x['after']['piece']} ({x['after']['day']}) |")
    lines += ["", "## 3. Emptied inside a laid piece while she is tracked (part of 1), by cause", "",
              "| shot | subject | frames | count | piece | cause |", "|---|---|---|---|---|---|"]
    for x in out["emptied_inside_a_piece"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece']} | {x['state'].split('(', 1)[1].rstrip(')')} |")
    lines += ["", "## The original shows, and the tied captures cannot tell whether she is tracked", "",
              "A capture named in the last column holds her track on some of these frames and is tied to no row; it is not read.", "",
              "| shot | subject | frames | count | why not known | untied captures with her track |", "|---|---|---|---|---|---|"]
    for x in out["original_tracking_not_known"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['tracking']} | "
                     f"{', '.join(x['untied_captures_with_her_track']) or '-'} |")
    lines += ["", "## She is tracked and whether the laid piece carried a mask is not known", "",
              "| shot | subject | frames | count | piece | why | track px median |", "|---|---|---|---|---|---|---|"]
    for x in out["carried_not_known"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece'] or '-'} | {x['state']} | {x['track_px']['median']} |")
    lines += ["", "## The original shows on frames her tied track is 0 px", "", "| shot | subject | frames | count | piece | state |", "|---|---|---|---|---|---|"]
    for x in out["original_her_track_empty"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece'] or '(none laid)'} | {x['state']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--table", required=True, help="a delivery table's .json (bench/delivery_table.py)")
    ap.add_argument("--span", help="A-B: only the shots that meet these frames, cut to them")
    ap.add_argument("--json", help="where the JSON goes; beside the table when left out")
    ap.add_argument("--md", help="where the .md goes; beside the table when left out")
    a = ap.parse_args()
    table_path = Path(a.table)
    table = json.loads(table_path.read_text())
    span = [int(x) for x in a.span.split("-")] if a.span else None
    out = coverage(table, span)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stem = "frame_coverage_" + table_path.stem.removeprefix("t_") + (f"_{span[0]:04d}_{span[1]:04d}" if span else "")
    out = {"table": str(table_path), "span": span,
           "made": f"Made {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} at commit {commit}",
           "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
           "measured_on": "the table's reading (shots and subjects) and rows (laid pieces, run=); each tied capture's SAM 3.1 track "
                          "(subjects/<label>/masks__<by>.npz), its run's carried mask (runs/<run>/region.npz), its manifest's held, drop=, "
                          "grade and no-mask lists and part_grades.json",
           "cannot_see": CANNOT, **out}
    out["counts"] = counts(out)
    jp = Path(a.json) if a.json else table_path.parent / f"{stem}.json"
    mp = Path(a.md) if a.md else table_path.parent / f"{stem}.md"
    jp.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    mp.write_text(markdown(out))
    print(json.dumps(out["counts"], indent=1))
    print(f"wrote {jp}\nwrote {mp}")
    for x in out["failures"]:
        print("FAIL", x)
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
