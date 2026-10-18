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
  subject's track over its own frames). Where the tied captures cannot tell, the run says why and either
  "the original: no piece made for these frames (no capture here, so tracking is not shown)" ("no tied capture
  covers this frame") or "the original: no piece made for these frames, and the capture there holds no track of her"
  (one covers it and holds no track of her on it), never "not
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

**Left out by a look** (the lead's ruling of 2026-10-16, as built). Frames the shots record says the subject's part is
NOT SEEN on (`who[label].seen_looks`, entered by `bench/shots_look.py`; a later `seen` takes them back) are their own
kind, with who looked and why, and never in list 1. A laid piece's own job `left_out` is NOT taken as a look: it says
"his face is behind his blurred hand" on one load and "left for a later load ... not ruled out" on another (lead_fp_0852b
on the kitchen's 604-845, the very frames the owner saw turn back), so it is said beside the list-1 run it covers
(`a_jobs_left_out_says`) and the frames stay in list 1. A look that rules frames the original goes into the shots record.
The ledger and the shots record are the ones the table's `made` names, or `--ledger` and `--shots`.

**Laid but barely changed** (the lead's order of 2026-10-16, on the owner's "I thought we built a gap checker!?": 806 in
the 2003Z cut, where lead_fp_0805d, "only what changed", barely changed its own region and the view said no gap). The
assembler records what each piece changed on each frame (the clip's `<clip>.mp4.check.json`, `regions.pieces.<piece>.
changed_px_per_frame`, by its own `changed`); nothing is recomputed here. Each frame a piece carried a mask on, its changed
px are set against the px of the region its run carried (`runs/<run>/region.npz`), both counted at the canvas. A share under `BARELY_CHANGED` is a run of `laid_but_barely_changed`, with the shares
printed. The check is the one the table's own assembly command writes (`--check` names another); the gap check is
refused when it is missing or its pieces and frames are not the table's rows, never run without it. A frame the piece's
job dropped or emptied carried no mask and is its own kind, never this one.

**Region far below its own size** (the lead's order of 2026-10-16, on the owner's "finds her... ear... at 805"): a frame
whose carried region is under `FAR_BELOW_ITS_OWN_SIZE` of the piece's own median carried px (over the row's frames that
carried one): the piece changed what it carried, and what it carried was a scrap of her, so her face shows round it. Said
with the frame's px and the median. A frame that is both is said as this kind.

**By shot**, worst first in each of two lists (`by_shot`, the lead's ruling of 2026-10-16): (a) `remake`, the frames a laid
piece fails on or no piece is laid where she is tracked (the original while tracked, by no piece and emptied inside one;
laid but barely changed; a region far below its own size), which a remake answers; (b) `capture`, the original where
nobody's track is tied (the original: no piece made for these frames (no capture here, so tracking is not shown)), which a capture answers, not a remake. A look's frames and an empty track are
in neither.

**What it cannot see.** The picture. A frame whose mask was carried can still come out looking like the original (H3
may regenerate a face like the source's); deciding that wants a per-frame likeness of the delivered video, which no
record holds. The changed share is an upper bound on what changed inside the region: the assembler's changed px reach
past it (the node's feather, a spill), and a piece laid by its region counts the region whole, so a region-laid piece that
drew the original comes out over the line. Track px > 0 says SAM 3.1 put some pixels on her, not that her face is visible: a run's px are printed
so a reader sees "barely in". A two-subject run's `carried` is both subjects' union. A frame the capture did not read
is said as that. Nothing here is a threshold of degree; green or red only for a row whose `run=` does not resolve or
a file it names that is missing.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
          "shows; whether her face shows on such frames is a playback question); a two-subject run's carried mask is both subjects' union; a face-and-hair run whose face emptied while its hair was carried counts as carried (the px are the union); a restore= that gives the source back at assembly inside a carried mask is not read; a subject the shots file says is not in a shot is not walked (the table's not_in_the_shot); an older capture's subject named by an alias is not matched as untied; a frame the tied captures cannot tell is 'the original: no piece made for these frames (no capture here, so tracking is not shown)' with why, and an untied capture holding her track is named, never read")


#: The share of the carried region a laid piece must change on a frame, under which the frame is `laid_but_barely_changed`.
#: **Measured** starting value, set between the owner's 806 and 807 on the 2003Z cut at the corrected scale
#: (`fun_0000_6394_joined_2026-10-16T2003Z_from_laid.mp4.check.json`; both counts at the canvas, Gemini's D17, 2026-10-16):
#: lead_fp_0805d's 806 is 0.226 (1810 of 8008 px; the owner: it "doesn't swap") and 807 0.404 (3735 of 9235; "seems to swap"),
#: its 828-832 0.511 to 0.566 (no complaint). It was first set on shares understated by the loader crop's area over the canvas's
#: (1.93 on this video). The shares are printed so a reader sees how far under.
BARELY_CHANGED = 0.3

#: The share of a piece's own median carried px under which a frame's carried region is `region_far_below_its_own_size`: the
#: piece changed what it carried, and what it carried was a scrap of her (her face hidden), so the original shows round it.
#: **Measured** starting value, on the same cut (2642 carried frames, 2026-10-16): lead_fp_0805d's 805 is 0.041 (486 px against
#: its median 11853; the owner: "finds her... ear... at 805") and lead_fp_1419d's 1448 0.077; lead_fp_0751c's 751 is 0.100, and
#: thirteen frames fall under 0.15, eleven of them lead_fp_1419d's and third_fp_1419d's. No break in the shares: a line, not a
#: gap, and the px and the median are printed beside it.
FAR_BELOW_ITS_OWN_SIZE = 0.1


class Refused(Exception):
    """The gap check is not run: the clip's check it reads is missing or is not of this table."""


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


def left_out_by_looks(table: dict, ledger: dict | None, shots: dict | None) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], dict]]:
    """Two maps of (subject, frame) -> {who, why, from}.

    `looks`: the frames the shots record says the subject's part is NOT SEEN on (`who[label].seen_looks`,
    `bench/shots_look.py`; a later `seen` takes an earlier `not seen` back). These are their own kind, "left out by a
    look", never a gap (the lead's ruling of 2026-10-16).

    `jobs`: the frames a laid piece's own job left out (`job.left_out`, from the job file the ledger's record of the
    render names), carried beside those frames as context and NOT taken out of list 1: a job's left_out is sometimes a
    look ("his face is behind his blurred hand") and sometimes "left for a later load ... not ruled out" (lead_fp_0852b
    on 604-845, the kitchen's own gaps), and only free text tells them apart. A look that rules frames the original
    is entered into the shots record."""
    import re
    looks: dict[tuple[str, int], dict] = {}
    jobs: dict[tuple[str, int], dict] = {}
    for row in table.get("rows", []):
        r = ((ledger or {}).get("renders") or {}).get(row.get("file")) or {}
        rec = next((c for c in reversed(r.get("records") or []) if c.get("job_file")), None)
        if rec is None or not (REPO / rec["job_file"]).is_file():
            continue
        job = json.loads((REPO / rec["job_file"]).read_text()).get("job") or {}
        m = re.search(r"written by (\S+) at (\d{4}-\d\d-\d\dT[\d:]+Z?)", str(job.get("note") or ""))
        who = f"the job of {row.get('name')}" + (f", written by {m.group(1)} at {m.group(2)}" if m else "")
        for x in job.get("left_out") or []:
            for f in range(int(x["first"]), int(x["first"]) + int(x["frames"])):
                jobs[(x["subject"], f)] = {"who": who, "why": x.get("why"), "from": rec["job_file"]}
    for shot in (shots or {}).get("shots", []):
        for label, w in (shot.get("who") or {}).items():
            for look in w.get("seen_looks") or []:
                for f in range(int(look["first"]), int(look["last"]) + 1):
                    if look.get("is") == "not seen":
                        looks[(label, f)] = {"who": look.get("by"), "why": look.get("why"), "from": "the shots record's seen_looks"}
                    elif look.get("is") == "seen":
                        looks.pop((label, f), None)
    return looks, jobs


def named_in(table: dict, pattern: str) -> Path | None:
    import re
    m = re.search(pattern, str(table.get("made") or ""))
    return REPO / m.group(1) if m and (REPO / m.group(1)).is_file() else None


def written_by_render(table: dict) -> dict[str, str]:
    out = {}
    for r in table.get("reading", []):
        for s in r.get("stands", []):
            if s.get("written"):
                out[s["render"]] = s["written"]
    return out


def check_of(table: dict, path: Path | None = None) -> tuple[Path, dict]:
    """The clip's check (`<clip>.mp4.check.json`, the assembler's): the one the table's own assembly command writes (its
    `--out`), or `path`. Refused when it is missing, or when its pieces and frames are not the table's rows."""
    if path is None:
        cmd = shlex.split(str((table.get("assembly") or {}).get("command") or ""))
        if "--out" not in cmd[:-1]:
            raise Refused("the table names no assembled clip (its assembly command has no --out): give the clip's check with --check")
        path = Path(cmd[cmd.index("--out") + 1] + ".check.json")
    path = Path(path)
    if not path.is_file():
        raise Refused(f"{path}: missing. The gap check reads what each piece changed from the clip's check, so it is not run without "
                      "it: assemble the table first (its assembly command), or name the check with --check")
    check = json.loads(path.read_text())
    left = [(str(r["file"]), int(r["first"]), int(r["last"])) for r in table["rows"]]
    strays = []
    for x in check.get("table") or []:
        p, a, b = str(x["piece"]), int(x["frames"][0]), int(x["frames"][1])
        i = next((i for i, (f, fa, fb) in enumerate(left) if (fa, fb) == (a, b) and (p == f or p.endswith("/" + f.lstrip("/")))), None)
        if i is None:
            strays.append(f"{Path(p).name} {a}-{b}")
        else:
            left.pop(i)
    if strays or left:
        raise Refused(f"{path} is not of this table: pieces in the check and not the table: {strays or 'none'}; rows of the table not in "
                      f"the check: {[f'{Path(f).name} {a}-{b}' for f, a, b in left] or 'none'}")
    return path, check


def changed_of(check: dict, row: dict) -> dict[int, int]:
    """The px the assembler's `changed` called changed on each frame of a row's piece, from the clip's check."""
    piece = next((p for p in check["regions"]["pieces"] if p == row["file"] or p.endswith("/" + row["file"].lstrip("/"))), None)
    if piece is None:
        raise Refused(f"the clip's check holds no changed px for {row['name']} ({row['file']})")
    v = check["regions"]["pieces"][piece]
    a, b = (int(x) for x in v["frames"])
    if len(v["changed_px_per_frame"]) != b - a + 1:
        raise Refused(f"the clip's check holds {len(v['changed_px_per_frame'])} changed counts for {row['name']} over {a}-{b}: not one a frame")
    return {a + k: int(x) for k, x in enumerate(v["changed_px_per_frame"])}


def coverage(table: dict, span: list[int] | None = None, data: Path = REPO / "data", ledger: Path | None = None, shots: Path | None = None,
             check: Path | None = None) -> dict:
    check_path, clip_check = check_of(table, check)
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
    ledger = ledger or named_in(table, r"from the ledger `([^`]+)`")
    shots_file = shots or named_in(table, r"`([^`]*shots\.json)`")
    looked, job_left = left_out_by_looks(table, json.loads(ledger.read_text()) if ledger else None, json.loads(shots_file.read_text()) if shots_file else None)
    shots, list1, list2, unknown, empty, carried_unknown, looked_out, barely, small = [], [], [], [], [], [], [], [], []
    for rd in table.get("reading", []):
        a, b, label = int(rd["first"]), int(rd["last"]), rd["subject"]
        if span and (b < span[0] or a > span[1]):
            continue
        present = rd.get("present")                                 # "yes" or "unsure": both are read, and said
        lo, hi = (max(a, span[0]), min(b, span[1])) if span else (a, b)
        in_shot = [(r, c, n) for r, c, n in tied if r["last"] >= a and r["first"] <= b]
        mine = [(r, c, n) for r, c, n in in_shot if r["subject"] == label]
        failed_mine = [r for r in failed if r.get("subject") == label and r["last"] >= a and r["first"] <= b]
        empt, changed_px, own_size = {}, {}, {}
        for r, c, n in mine:
            e = emptied(c, n, label, [max(r["first"], lo), min(r["last"], hi)])
            empt[id(r)] = e
            changed_px[id(r)] = changed_of(clip_check, r)
            own = c.run(n)
            sizes = [int(own["carried_px"][f - c.first]) for f in range(r["first"], r["last"] + 1)
                     if 0 <= f - c.first < c.n and own["read"][f - c.first] and own["carried_px"][f - c.first] > 0]
            own_size[id(r)] = float(median(sizes)) if sizes else None
        cols = {"frames": [], "track_px": [], "track_from": [], "piece": [], "carried_px": [], "changed_share": [], "state": []}
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
            carried, share = None, None
            bad = next((r for r in failed_mine if r["first"] <= f <= r["last"]), None)
            if piece is None and bad is not None:
                state = f"not known: the laid piece {bad.get('name')} has a run= that does not resolve"
            elif piece is None and (label, f) in looked:
                state = "left out by a look: " + str(looked[(label, f)]["why"])
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
                    else:
                        # what the assembler's `changed` called changed, against the region carried: both counted at the canvas (the
                        # assembler lays at the canvas whatever --size, and refuses a capture of another size; Gemini's D17, 2026-10-16)
                        share = changed_px[id(r)].get(f, 0) / carried
                        if own_size[id(r)] and carried < FAR_BELOW_ITS_OWN_SIZE * own_size[id(r)]:
                            state = f"region far below its own size (its median {int(own_size[id(r)])} px)"
                        elif share < BARELY_CHANGED:
                            state = "laid but barely changed"
                        elif f in frames_of((c.sighting(label) or {}).get("parts_held_on_source_frames")):
                            state = "carried (held from a neighbouring frame)"
                        else:
                            state = "carried"
            for key, v in (("frames", f), ("track_px", tpx), ("track_from", tfrom), ("piece", piece[0]["name"] if piece else None),
                           ("carried_px", carried), ("changed_share", None if share is None else round(share, 4)), ("state", state)):
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
                said = [job_left[(label, f)] for f in range(cur["f0"], cur["f1"] + 1) if (label, f) in job_left]
                if said:
                    # context, not a verdict: what a laid piece's job said of these frames (see left_out_by_looks)
                    item["a_jobs_left_out_says"] = {k: said[0][k] for k in ("who", "why", "from")} | {"frames": len(said)}
                list1.append(item)
            elif cur["kind"] == "unknown":
                words = ("the original: no piece made for these frames, and the capture there holds no track of her"
                         if "holds no track of her" in str(cur["why"] or "")
                         else "the original: no piece made for these frames (no capture here, so tracking is not shown)")
                item["tracking"] = ("not known: " + cur["state"].removeprefix("not known: ")) if cur["state"].startswith("not known") \
                    else f"{cur['why']}: {words}"
                item["untied_captures_with_her_track"] = [n for n, a_, b_, labels in untied
                                                          if label in labels and a_ <= cur["f1"] and b_ >= cur["f0"]]
                unknown.append(item)
            elif cur["kind"] == "untracked":
                empty.append(item)
            elif cur["kind"] == "looked_out":
                item.update({k: looked[(label, cur["f0"])][k] for k in ("who", "why", "from")})
                looked_out.append(item)
            elif cur["kind"] == "carried_not_known":
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)}
                carried_unknown.append(item)
            elif cur["kind"] == "barely":
                sh = cur["share"]
                item["changed_share"] = {"least": min(sh), "median": round(median(sh), 4), "most": max(sh), "under": BARELY_CHANGED}
                barely.append(item)
            elif cur["kind"] == "small":
                item["carried_px"] = cur["carried"]
                item["its_median_px"] = int(cur["state"].rsplit("median ", 1)[1].split(" ")[0])
                item["changed_share"] = cur["share"]
                small.append(item)

        for i, f in enumerate(cols["frames"]):
            st, tpx = cols["state"][i], cols["track_px"][i]
            if st == "laid but barely changed":
                kind = "barely"
            elif st.startswith("region far below its own size"):
                kind = "small"
            elif st.startswith("left out by a look"):
                kind = "looked_out"
            elif st.startswith("not known"):
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
                cur["share"].append(cols["changed_share"][i])
                cur["carried"].append(cols["carried_px"][i])
                if cols["track_from"][i] and tpx is not None:
                    cur["from"].add(cols["track_from"][i])
            else:
                close()
                cur = None if kind is None else {"key": key, "kind": kind, "f0": f, "f1": f, "piece": cols["piece"][i], "state": st, "why": why,
                                                 "px": [tpx], "share": [cols["changed_share"][i]], "carried": [cols["carried_px"][i]],
                                                 "from": set() if tpx is None else {cols["track_from"][i]} - {None}}
        close()
        # list 2: two different laid renders of her meeting inside this shot
        laid = sorted((r for r, _, _ in mine), key=lambda r: r["first"])
        for x, y in zip(laid, laid[1:]):
            if y["first"] == x["last"] + 1 and x["file"] != y["file"] and lo <= x["last"] and y["first"] <= hi:
                list2.append({"shot": [a, b], "subject": label, "at": [x["last"], y["first"]],
                              "before": {"piece": x["name"], "written": written.get(x["name"]), "day": day(written.get(x["name"]))},
                              "after": {"piece": y["name"], "written": written.get(y["name"]), "day": day(written.get(y["name"]))}})
    list3 = [x for x in list1 if x["piece"] is not None]
    runs = flat(list1, list2, unknown, empty, written, carried_unknown, looked_out, barely, small)
    return {"failures": failures, "shots": shots, "original_while_tracked": list1, "seams": list2,
            "emptied_inside_a_piece": list3, "original_tracking_not_known": unknown, "original_her_track_empty": empty,
            "carried_not_known": carried_unknown, "left_out_by_a_look": looked_out, "laid_but_barely_changed": barely,
            "region_far_below_its_own_size": small,
            "read_for_looks": {"ledger": str(ledger) if ledger else None, "shots": str(shots_file) if shots_file else None},
            "clip_check": {"file": str(check_path), "sha256": hashlib.sha256(check_path.read_bytes()).hexdigest()},
            "runs": runs, "by_shot": by_shot(runs)}


#: The kinds of run counted per shot (`by_shot`), in two lists the lead ruled apart (2026-10-16): what a remake answers (a laid
#: piece failing, or no piece where she is tracked), and what a capture answers (nobody's track tied).
ORIGINAL_SHOWING = {"remake": ("original_while_tracked", "emptied_inside", "laid_but_barely_changed", "region_far_below_its_own_size"),
                    "capture": ("original_tracking_not_known",)}


def by_shot(runs: list[dict]) -> dict[str, list[dict]]:
    """Per shot, the frames of each kind where the original shows (`ORIGINAL_SHOWING`), worst first, in its two lists: which shots
    to remake first, and which want a capture tied. A frame two subjects show the original on is counted for each."""
    return {which: ranked(runs, kinds) for which, kinds in ORIGINAL_SHOWING.items()}


def ranked(runs: list[dict], kinds: tuple) -> list[dict]:
    shots: dict[tuple, dict] = {}
    for r in runs:
        if r["kind"] in kinds:
            x = shots.setdefault(tuple(r["shot"]), {"shot": list(r["shot"]), "frames": 0, "by_kind": {}, "subjects": set()})
            n = r["last"] - r["first"] + 1
            x["frames"] += n
            x["by_kind"][r["kind"]] = x["by_kind"].get(r["kind"], 0) + n
            x["subjects"].add(r["subject"])
    return [{**x, "subjects": sorted(x["subjects"])} for x in sorted(shots.values(), key=lambda x: (-x["frames"], x["shot"]))]


def flat(list1: list[dict], list2: list[dict], unknown: list[dict], empty: list[dict], written: dict[str, str],
         carried_unknown: list[dict] | None = None, looked_out: list[dict] | None = None, barely: list[dict] | None = None,
         small: list[dict] | None = None) -> list[dict]:
    """One list of runs for a page, by frame (the lead's shape, 2026-10-16). The kinds are disjoint: a run of list 1
    inside a piece is `emptied_inside`, one with no piece `original_while_tracked` (`shot_has_a_piece_of_her` tells a
    turn-back inside a shot she has a piece in from a shot with no piece of her at all); a seam's first and last are
    the two frames that meet; `original_tracking_not_known` is a run the tied captures cannot tell, said so it is not
    read as none, with any untied capture that holds her track; `original_her_track_empty` is the original on frames
    her tied track is 0 px; `carried_not_known` is a run her track is known on and whether the laid piece carried a
    mask is not (its run= does not resolve, its capture does not cover the frame, or did not read it); `laid_but_barely_changed`
    is a run a piece carried a mask on and changed less than `BARELY_CHANGED` of the region, the shares said;
    `region_far_below_its_own_size` a run whose carried region is under `FAR_BELOW_ITS_OWN_SIZE` of the piece's median, the px said."""
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
    for x in looked_out or []:
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"], "kind": "left_out_by_a_look",
                    "pieces": [], "render_days": [], "cause": x["why"], "who": x["who"], "from": x["from"], "track_px_median": None,
                    "present": x["present"]})
    for x in carried_unknown or []:
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
                    "kind": "carried_not_known", "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
                    "cause": x["state"], "track_px_median": x["track_px"]["median"], "present": x["present"]})
    for x in barely or []:
        sh = x["changed_share"]
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"], "kind": "laid_but_barely_changed",
                    "pieces": [x["piece"]], "render_days": [day(written.get(x["piece"]))],
                    "cause": f"{sh['least']:.3f} to {sh['most']:.3f} of its carried region changed (under {BARELY_CHANGED})",
                    "changed_share": sh, "track_px_median": None, "present": x["present"]})
    for x in small or []:
        out.append({"first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
                    "kind": "region_far_below_its_own_size", "pieces": [x["piece"]], "render_days": [day(written.get(x["piece"]))],
                    "cause": f"carried {', '.join(str(p) for p in x['carried_px'])} px against its median {x['its_median_px']} "
                             f"(under {FAR_BELOW_ITS_OWN_SIZE} of it)",
                    "carried_px": x["carried_px"], "its_median_px": x["its_median_px"], "track_px_median": None, "present": x["present"]})
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
            "original_and_her_track_empty_frames": untracked,
            "left_out_by_a_look_frames": sum(x["count"] for x in out.get("left_out_by_a_look", [])),
            "laid_but_barely_changed": {"runs": len(out["laid_but_barely_changed"]), "frames": sum(x["count"] for x in out["laid_but_barely_changed"])},
            "region_far_below_its_own_size": {"runs": len(out["region_far_below_its_own_size"]),
                                              "frames": sum(x["count"] for x in out["region_far_below_its_own_size"])}}


def fr(x) -> str:
    return f"{x[0]}-{x[1]}" if x[0] != x[1] else str(x[0])


def markdown(out: dict) -> str:
    c = out["counts"]
    lines = [f"# Frame coverage: {out['table']}", "", out["made"], "", f"Command: `{out['command']}`", "",
             f"Measured on: {out['measured_on']}", "", f"Cannot see: {out['cannot_see']}", ""]
    if out["failures"]:
        lines += ["## FAILED: rows whose run= does not resolve, or a file missing", ""] + [f"- {x}" for x in out["failures"]] + [""]
    for which, title in (("remake", "the shots to remake: a laid piece fails, or no piece where she is tracked"),
                         ("capture", "the shots that want a capture tied: the original, nobody's track tied")):
        kinds = ORIGINAL_SHOWING[which]
        lines += [f"## By shot, worst first: {title}", "",
                  "| shot | frames | " + " | ".join(kinds) + " | subjects |", "|---|---|" + "---|" * len(kinds) + "---|"]
        for x in out["by_shot"][which]:
            lines.append(f"| {fr(x['shot'])} | {x['frames']} | " + " | ".join(str(x["by_kind"].get(k, 0)) for k in kinds) + f" | {', '.join(x['subjects'])} |")
        lines += [""]
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
    lines += ["", "## The original shows, and the tied captures cannot tell whether she is tracked (the original: no piece made for these frames (no capture here, so tracking is not shown))", "",
              "A capture named in the last column holds her track on some of these frames and is tied to no row; it is not read.", "",
              "| shot | subject | frames | count | why not known | untied captures with her track |", "|---|---|---|---|---|---|"]
    for x in out["original_tracking_not_known"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['tracking']} | "
                     f"{', '.join(x['untied_captures_with_her_track']) or '-'} |")
    lines += ["", "## Left out by a look (a laid piece's job, or the shots record): the original, on a person's word", "",
              "| shot | subject | frames | count | who | why |", "|---|---|---|---|---|---|"]
    for x in out.get("left_out_by_a_look", []):
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['who']} | {str(x['why'])[:300]} |")
    lines += ["", "## She is tracked and whether the laid piece carried a mask is not known", "",
              "| shot | subject | frames | count | piece | why | track px median |", "|---|---|---|---|---|---|---|"]
    for x in out["carried_not_known"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece'] or '-'} | {x['state']} | {x['track_px']['median']} |")
    lines += ["", f"## Laid but barely changed: a piece carried a mask and changed under {BARELY_CHANGED} of its region", "",
              f"Read from the clip's check `{out['clip_check']['file']}` (regions.pieces.<piece>.changed_px_per_frame, the assembler's `changed`), "
              "against each run's carried region.", "",
              "| shot | subject | frames | count | piece | changed share least / median / most |", "|---|---|---|---|---|---|"]
    for x in out["laid_but_barely_changed"]:
        sh = x["changed_share"]
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece']} | {sh['least']} / {sh['median']} / {sh['most']} |")
    lines += ["", f"## Region far below its own size: a carried region under {FAR_BELOW_ITS_OWN_SIZE} of the piece's own median", "",
              "| shot | subject | frames | count | piece | carried px | its median px |", "|---|---|---|---|---|---|---|"]
    for x in out["region_far_below_its_own_size"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece']} | "
                     f"{', '.join(str(p) for p in x['carried_px'])} | {x['its_median_px']} |")
    lines += ["", "## The original shows on frames her tied track is 0 px", "", "| shot | subject | frames | count | piece | state |", "|---|---|---|---|---|---|"]
    for x in out["original_her_track_empty"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece'] or '(none laid)'} | {x['state']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--table", required=True, help="a delivery table's .json (bench/delivery_table.py)")
    ap.add_argument("--span", help="A-B: only the shots that meet these frames, cut to them")
    ap.add_argument("--ledger", help="the render ledger whose records name each laid piece's job file; the one the table names when left out")
    ap.add_argument("--shots", help="the shots record whose looks are read; the one the table names when left out")
    ap.add_argument("--check", help="the assembled clip's <clip>.mp4.check.json; the one the table's assembly command writes when left out")
    ap.add_argument("--json", help="where the JSON goes; beside the table when left out")
    ap.add_argument("--md", help="where the .md goes; beside the table when left out")
    a = ap.parse_args()
    table_path = Path(a.table)
    table = json.loads(table_path.read_text())
    span = [int(x) for x in a.span.split("-")] if a.span else None
    try:
        out = coverage(table, span, ledger=Path(a.ledger) if a.ledger else None, shots=Path(a.shots) if a.shots else None,
                       check=Path(a.check) if a.check else None)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stem = "frame_coverage_" + table_path.stem.removeprefix("t_") + (f"_{span[0]:04d}_{span[1]:04d}" if span else "")
    out = {"table": str(table_path), "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(), "span": span,
           "made": f"Made {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} at commit {commit}",
           "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
           "measured_on": "the table's reading (shots and subjects) and rows (laid pieces, run=); each tied capture's SAM 3.1 track "
                          "(subjects/<label>/masks__<by>.npz), its run's carried mask (runs/<run>/region.npz), its manifest's held, drop=, "
                          "grade and no-mask lists and part_grades.json; the clip's check (each piece's changed px a frame, the assembler's)",
           "cannot_see": CANNOT, **out}
    out["counts"] = counts(out)
    jp = Path(a.json) if a.json else table_path.parent / f"{stem}.json"
    mp = Path(a.md) if a.md else table_path.parent / f"{stem}.md"
    jp.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    mp.write_text(markdown(out))
    print(json.dumps(out["counts"], indent=1))
    for which, title in (("remake", "TO REMAKE (a laid piece fails, or no piece where she is tracked)"),
                         ("capture", "TO CAPTURE (the original, nobody's track tied)")):
        print(f"BY SHOT, worst first, {title}:")
        for x in out["by_shot"][which][:15]:
            print(f"  {fr(x['shot'])}: {x['frames']} frame(s): " + ", ".join(f"{k} {v}" for k, v in x["by_kind"].items()) + f" ({', '.join(x['subjects'])})")
    print(f"wrote {jp}\nwrote {mp}")
    for x in out["failures"]:
        print("FAIL", x)
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
