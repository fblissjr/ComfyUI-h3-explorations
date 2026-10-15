#!/usr/bin/env python3
"""Whether a load made again can rest on the look an earlier capture of it had: track, part, region and reference video.

    <python> bench/look_carry.py --new data/<day>_<load> --read data/<day>_<earlier load> [--by WHO --note-file WORDS.txt]
                                 [--reference-checked CHECK.txt --checked-frames 660,670,676,680]

**What it is for.** A load made again (a new reference, a new text, a mend that does not touch its masks) is owed a look
like any load, and a blind reader's look costs the card time. The card measured by hand, every time, whether the new
capture's track, part and region were equal to, or a strict subset of, an earlier capture a blind reader had read, and the
lead then recorded a look resting on that reading (mrcircle's `measure_against.py`, ported here: mrpen's of 2026-10-13).

**What it prints**, by source frame, on the frames both captures hold (the new one may start inside the earlier one): the
subject's track and part (equal, a subset, or LARGER, with the px added and where), and the planned region (equal, a
subset, or LARGER by how many 16 px cells and where); the rules new and gone between the two captures' flags, and the new
one's verdict; and which reading the earlier capture's last look rests on (its words, and a blind reading beside it).
Then the reference video each capture's load names in its job file (`motion_file`, `motion_file_2`; the job is found beside
the capture as `<capture>_job/job.plan.json`, or named with `--new-job`/`--read-job`): its path, its sha256 now, when the
file was last written, and whether that is before the earlier capture's strip was drawn (`looks/<load>.shown.json`,
`made`). A reference remade in place keeps its path (the 10/14 mouth mend moved the old ones aside and wrote the new ones
under the old names), so the reference is THE SAME only when the earlier capture's file has not been written since the
strip its reader saw (the earliest time a look kept of it: a redraw does not move it), both digests are equal and each
begins on the same source frame; anything else, a job not found among them, is CHANGED. Two loads naming no reference are
THE SAME, as nothing. The frames the
earlier look's words name inside its load are printed, since a check of a changed reference is asked for on those.

**With `--by`** it records a look on the new capture (`capture_masked_run.py looked`, one line in its `looks.json`) that
rests on that reading, ONLY when every frame of every mask is equal or a subset, the new load's frames all inside the earlier
one's, and the earlier capture has a reading to rest on: a look in its `looks.json`, or a blind reading beside it
(`look_blind/reading*.md`). Otherwise it refuses and says a reader is owed. The look is the `--by`
person's, with their own words (`--note-file`) and the figures; a look by the capture's maker is refused by the render step.
Where the masks carry and the reference CHANGED, it records only with the check of the new reference named: the checker's
words (`--reference-checked`, a file: who looked at what, and when) and the frames looked at (`--checked-frames`, each
inside the new load). The look's note then says the reference changed, both digests, the frames checked, and which frames
the earlier look's words name that the check did not; that last is printed, not refused, since it is read from words (the
look's note only, not the blind reading's file).

**What it cannot see.** A subset of the masks is not a subset of what a render draws: the text may differ, and the reading
said what it saw then. What the earlier reference was when it was CHANGED (its file is gone from the path, or was never
named); a file's written time is the filesystem's (the share's clock, read in this shell's zone), and a copy that keeps an old
time would pass for unchanged, while a rewrite of the same bytes reads as CHANGED. A job whose reference key is not
`motion_file` reads as naming none. Whether the
new reference stays the same file until the render reads it. Whether the earlier reading was right. The picture.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The masks of the subject compared frame by frame (the capture's `subjects/<subject>/masks__<preview>.npz`, packed bits).
MASKS = ("track", "parts")


def load_of(capture: Path) -> dict:
    """A capture's load: its run's name, subject, first source frame and frame count, the preview its subject was seen by."""
    m = json.loads((capture / "manifest.json").read_text())
    if not m.get("runs"):
        raise SystemExit(f"{capture.name} holds no run")
    run = m["runs"][0]
    subject = run["subject"]
    seen = next(s for s in m["subjects"] if s["label"] == subject)["sightings"][0]["by"]
    region = np.load(capture / "runs" / run["name"] / "region.npz")["region"]
    return {"capture": capture, "name": run["name"], "subject": subject, "by": seen, "first": int(run.get("first_source_frame", m["first_frame"])),
            "frames": int(region.shape[0]), "region": region > 0, "planned": bool(run.get("planned"))}


def where(cells: np.ndarray) -> list[int] | None:
    """The box (x0, y0, x1, y1) of a frame's True cells, or None."""
    ys, xs = np.nonzero(cells)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def compare(new: dict, read: dict) -> dict:
    """Frame by frame on the new load's frames: each mask and the region equal, a subset, or larger, with what was added and where."""
    shift = new["first"] - read["first"]
    n = new["frames"]
    out = {"frames": [new["first"], new["first"] + n - 1], "read_frames": [read["first"], read["first"] + read["frames"] - 1],
           "outside_the_read": [new["first"] + i for i in range(n) if not 0 <= shift + i < read["frames"]], "masks": {}}
    inside = [i for i in range(n) if 0 <= shift + i < read["frames"]]
    a_masks = np.load(read["capture"] / "subjects" / read["subject"] / f"masks__{read['by']}.npz")
    b_masks = np.load(new["capture"] / "subjects" / new["subject"] / f"masks__{new['by']}.npz")
    for k in MASKS:
        rows = {}
        for i in inside:
            x, y = a_masks[k][shift + i], b_masks[k][i]
            added = int(np.unpackbits(y & ~x).sum())
            gone = int(np.unpackbits(x & ~y).sum())
            rows[new["first"] + i] = "equal" if not added and not gone else "subset" if not added else f"LARGER by {added} px"
        out["masks"][k] = rows
    rows = {}
    for i in inside:
        x, y = read["region"][shift + i], new["region"][i]
        added = y & ~x
        rows[new["first"] + i] = ("equal" if (x == y).all() else "subset" if not added.any()
                                  else f"LARGER by {int(added.sum())} cells at {where(added)}")
        out.setdefault("region_cells", {})[new["first"] + i] = [int(x.sum()), int(y.sum())]
    out["masks"]["region"] = rows
    out["larger"] = {k: [f for f, v in r.items() if v.startswith("LARGER")] for k, r in out["masks"].items()}
    out["carries"] = not out["outside_the_read"] and not any(out["larger"].values())
    return out


def rules(new: dict, read: dict) -> dict:
    """The rules new and gone between the two captures' flags, and the new one's verdict."""
    fa = json.loads((read["capture"] / "flags.json").read_text())
    fb = json.loads((new["capture"] / "flags.json").read_text())
    key = lambda f: (f["rule"], f.get("segment") if isinstance(f.get("segment"), str) else "", f.get("other") or "")  # noqa: E731
    a, b = set(map(key, fa["flags"])), set(map(key, fb["flags"]))
    return {"gone": sorted(a - b), "new": sorted(b - a), "verdict": fb.get("verdict")}


def reading_of(read: dict) -> dict:
    """What the earlier capture's last look rests on: its words, by and at, and any blind reading kept beside it."""
    looks = json.loads((read["capture"] / "looks.json").read_text())["looks"] if (read["capture"] / "looks.json").is_file() else []
    blind = sorted(str(p) for p in read["capture"].glob("look_blind/reading*.md"))
    last = looks[-1] if looks else None
    return {"look": {"by": last["by"], "at": last["at"], "note": last.get("note", "")} if last else None, "blind_readings": blind}


def job_of(capture: Path, run: str, given: str | None = None) -> Path | None:
    """The job file the capture's load was built from: the one named, else `<capture>_job/job.plan.json` beside it, when it holds the load."""
    path = Path(given) if given else capture.parent / f"{capture.name}_job" / "job.plan.json"
    if not path.is_file():
        if given:
            raise SystemExit(f"no job file {path}")
        return None
    file = json.loads(path.read_text())
    names = [x.get("name") for x in file.get("loads", [])]
    of = (file.get("job") or {}).get("capture")
    if of and Path(of).name != capture.name:
        if given:
            raise SystemExit(f"{path} is the job of {of}, not of {capture}")
        return None
    if run not in names:
        if given:
            raise SystemExit(f"{path} holds no load {run}; it has {', '.join(str(x) for x in names) or 'none'}")
        return None
    return path


def when(text: str | None) -> datetime.datetime | None:
    return datetime.datetime.fromisoformat(text) if text else None


def reference_of(load: dict, job: Path | None) -> dict:
    """The reference video(s) the load names in its job, each with its digest now, when the file was last written and the source frame
    it begins on, and when the capture's strip was first drawn: the earliest `made` of the load's strip among the copies each look
    kept (`looks.json`, `shown`; a redraw does not move them) and the shown record beside the strip (a redraw rewrites it). Local
    time, as the capture writes both."""
    capture, name = load["capture"], load["name"]
    shown = capture / "looks" / f"{name}.shown.json"
    kept = capture / "looks.json"
    times = [r.get("made") for x in (json.loads(kept.read_text())["looks"] if kept.is_file() else []) for r in x.get("shown", [])
             if r.get("load") == name] + [json.loads(shown.read_text()).get("made") if shown.is_file() else None]
    times = [t for t in times if t]
    strip = min(times, key=when) if times else None
    out = {"job": str(job) if job else None, "strip_made": strip, "files": []}
    if job is None:
        return out
    file = json.loads(job.read_text())
    entry = next(x for x in file["loads"] if x.get("name") == name)
    job_first = (file.get("job") or {}).get("first")              # the builder's rule: a reference with no `motion_first` begins on the job's first frame
    for x in ("", "_2"):
        f = entry.get(f"motion_file{x}")
        if not f:
            continue
        p = Path(f)
        one = {"file": f, "sha256": None, "written": None, "first": entry.get(f"motion_first{x}", job_first)}
        if p.is_file():
            one["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            one["written"] = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        out["files"].append(one)
    return out


def same_reference(new_ref: dict, read_ref: dict, read_name: str) -> dict:
    """THE SAME only when both jobs are found, name as many references, every earlier file was last written before the earlier strip
    was drawn (so it is what that strip showed) and every digest is equal. Anything else is CHANGED, with why."""
    def changed(why):
        return {"same": False, "why": why}
    for ref, who in ((read_ref, f"the earlier capture ({read_name})"), (new_ref, "the new capture")):
        if ref["job"] is None:
            return changed(f"no job file found for {who}: its reference is not known")
    if len(new_ref["files"]) != len(read_ref["files"]):
        return changed(f"the earlier load names {len(read_ref['files'])} reference video(s) and the new one {len(new_ref['files'])}")
    strip = when(read_ref["strip_made"])
    for a, b in zip(read_ref["files"], new_ref["files"]):
        for one in (a, b):
            if one["sha256"] is None:
                return changed(f"{one['file']} is not there to read")
        if strip is None:
            return changed("the earlier capture says nothing of when its strip was drawn")
        if when(a["written"]) >= strip:
            return changed(f"{a['file']} was written at {a['written']}, not before the earlier strip was drawn ({read_ref['strip_made']}): "
                           "it is not the file the earlier reader saw")
        if a["sha256"] != b["sha256"]:
            return changed(f"{Path(a['file']).name} {a['sha256'][:12]} and {Path(b['file']).name} {b['sha256'][:12]} differ")
        if a["first"] != b["first"]:
            return changed(f"{Path(a['file']).name} begins on source frame {a['first']} for the earlier load and {b['first']} for the new one")
    return {"same": True, "why": ("neither load names a reference video" if not new_ref["files"] else
                                  "the earlier file unwritten since its strip and every digest equal")}


def frames_named(note: str, first: int, last: int) -> list[int]:
    """The source frames a look's words name inside its load: a number, or a range ("813-818", "813-18") cut to the load. A number
    inside a name (`lead_fp_0630b`) or a decimal is not one; a count in the load's span ("820 px") is, and "from 658" names 658 only."""
    out = set()
    for m in re.finditer(r"(?<![\w.])(\d{2,5})(?:\s*[-\u2013]\s*(\d{1,5}))?(?!\w|\.\d)", note):
        a, tail = m.group(1), m.group(2) or m.group(1)
        b = int(a[:len(a) - len(tail)] + tail) if len(tail) < len(a) else int(tail)
        out.update(f for f in range(int(a), b + 1) if first <= f <= last)
    return sorted(out)


def spans(frames: list[int]) -> str:
    out, i = [], 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append(str(frames[i]) if i == j else f"{frames[i]}-{frames[j]}")
        i = j + 1
    return ", ".join(out) or "none"


def summary(c: dict) -> list[str]:
    lines = []
    for k, rows in c["masks"].items():
        eq = sum(1 for v in rows.values() if v == "equal")
        sub = sum(1 for v in rows.values() if v in ("equal", "subset"))
        lines.append(f"{k}: equal on {eq} of {len(rows)}; equal or a subset on {sub}"
                     + ("; LARGER on " + ", ".join(f"{f} ({rows[f][len('LARGER '):]})" for f in c["larger"][k][:12]) + (" ..." if len(c["larger"][k]) > 12 else "")
                        if c["larger"][k] else ""))
    if c["outside_the_read"]:
        lines.append(f"frames of the new load outside the earlier one: {c['outside_the_read'][0]}-{c['outside_the_read'][-1]} ({len(c['outside_the_read'])})")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--new", required=True, help="the new load's capture folder (on the node's plan)")
    ap.add_argument("--read", required=True, help="the earlier capture a reader read")
    ap.add_argument("--by", help="record a look on the new capture resting on that reading, as this person, when it carries")
    ap.add_argument("--note-file", help="with --by: the person's own words, a file; the figures are added to them")
    ap.add_argument("--new-job", help="the new load's job file, where it is not <new>_job/job.plan.json")
    ap.add_argument("--read-job", help="the earlier load's job file, where it is not <read>_job/job.plan.json")
    ap.add_argument("--reference-checked", help="with --by, where the reference CHANGED: the checker's words on the new reference (who, "
                                                "what, when), a file")
    ap.add_argument("--checked-frames", help="with --reference-checked: the source frames of the new reference looked at, comma separated")
    a = ap.parse_args(argv)
    new, read = load_of(Path(a.new)), load_of(Path(a.read))
    if new["subject"] != read["subject"]:
        raise SystemExit(f"the new load is of {new['subject']!r} and the earlier one of {read['subject']!r}: not one subject's load made again")
    c, r, reading = compare(new, read), rules(new, read), reading_of(read)
    new_ref = reference_of(new, job_of(new["capture"], new["name"], a.new_job))
    read_ref = reference_of(read, job_of(read["capture"], read["name"], a.read_job))
    ref = same_reference(new_ref, read_ref, read["name"])
    named = frames_named(reading["look"]["note"], read["first"], read["first"] + read["frames"] - 1) if reading["look"] else []
    print(f"{new['name']} ({new['frames']} frames from {new['first']}) against {read['name']} ({read['frames']} from {read['first']}), {new['subject']}:")
    for line in summary(c):
        print("  " + line)
    print(f"  rules gone: {r['gone']} | rules new: {r['new']} | the new capture's verdict: {r['verdict']}")
    look = reading["look"]
    print("  the earlier capture's last look: " + (f"{look['by']}, {look['at']}: {look['note'][:300]}" if look else "NONE")
          + ("; blind reading(s): " + ", ".join(reading["blind_readings"]) if reading["blind_readings"] else ""))
    for side, one in (("earlier", read_ref), ("new", new_ref)):
        print(f"  the {side} reference, from {one['job'] or 'NO JOB FILE FOUND'}: "
              + ("; ".join(f"{f['file']} sha256 {(f['sha256'] or 'NOT THERE')[:16]} written {f['written']}" for f in one["files"]) or "none named")
              + f"; its strip drawn {one['strip_made']}")
    print(f"  the reference: {'THE SAME' if ref['same'] else 'CHANGED'} ({ref['why']})")
    print(f"  the frames the earlier look's words name: {spans(named)}")
    print("  DOES NOT CARRY: " + ("frames outside the earlier load" if c["outside_the_read"] else "a mask is LARGER on " + ", ".join(k for k, v in c["larger"].items() if v))
          + ": a reader is owed" if not c["carries"] else
          "  CARRIES: every frame of every mask equal or a subset, and the same reference" if ref["same"] else
          "  THE MASKS CARRY AND THE REFERENCE CHANGED: it carries only with the check of the new reference named "
          "(--reference-checked WORDS.txt --checked-frames F,F,...), at the frames the earlier reader named")
    if not a.by:
        return 0
    if not c["carries"] or not (look or reading["blind_readings"]) or not new["planned"]:
        why = ("it does not carry" if not c["carries"] else "the earlier capture has no look and no blind reading to rest on" if not (look or reading["blind_readings"])
               else "the new capture is not on the node's plan")
        raise SystemExit(f"no look recorded: {why}. A reader is owed")
    if not a.note_file or not Path(a.note_file).read_text().strip():
        raise SystemExit("--note-file: the person's own words on what they rest the look on, a file")
    checked: list[int] = []
    if not ref["same"]:
        if not a.reference_checked or not Path(a.reference_checked).read_text().strip() or not a.checked_frames:
            raise SystemExit(f"no look recorded: the reference CHANGED ({ref['why']}). Name the check of the new reference: "
                             "--reference-checked WORDS.txt (who looked at what, and when) and --checked-frames F,F,...")
        try:
            checked = sorted({int(x) for x in a.checked_frames.split(",") if x.strip()})
        except ValueError as e:
            raise SystemExit(f"--checked-frames: source frame numbers, comma separated ({e})") from e
        outside = [f for f in checked if not new["first"] <= f < new["first"] + new["frames"]]
        if not checked or outside:
            raise SystemExit(f"--checked-frames: {spans(outside) if outside else 'none'} not inside the new load "
                             f"({new['first']}-{new['first'] + new['frames'] - 1})")
        unchecked = [f for f in named if f not in checked]
        if unchecked:
            print(f"  the earlier look's words name {spans(unchecked)}, not among the frames checked: said in the look's note", flush=True)
    import capture_masked_run as cap
    eq = {k: sum(1 for v in rows.values() if v == "equal") for k, rows in c["masks"].items()}
    note = (" ".join(Path(a.note_file).read_text().split())
            + f" RESTS ON the reading of {read['capture'].name}: "
            + "; ".join(([f"its look by {look['by']} at {look['at']}"] if look else []) + ([f"the blind reading {', '.join(reading['blind_readings'])}"] if reading["blind_readings"] else []))
            + f". By bench/look_carry.py: track, part and region equal or a subset on all {new['frames']} frames (equal: "
            + ", ".join(f"{k} {v}" for k, v in eq.items()) + f"); rules gone {r['gone']}, new {r['new']}; verdict {r['verdict']}."
            + (" The reference THE SAME: " + (", ".join(f"{Path(f['file']).name} sha256 {f['sha256'][:16]}" for f in new_ref["files"]) or "none named")
               + "." if ref["same"] else
               f" THE REFERENCE CHANGED ({ref['why']}): the earlier "
               + (", ".join(f"{Path(f['file']).name} sha256 {(f['sha256'] or 'not there')[:16]} written {f['written']}" for f in read_ref["files"]) or "none named")
               + ", the new " + (", ".join(f"{Path(f['file']).name} sha256 {(f['sha256'] or 'not there')[:16]} written {f['written']}" for f in new_ref["files"]) or "none named")
               + f". The new reference checked at {spans(checked)}: " + " ".join(Path(a.reference_checked).read_text().split())
               + (" The earlier look's words name no frame inside its load." if not named else
                  f" The earlier look's words name {spans(named)}"
                  + (f"; not among the frames checked: {spans([f for f in named if f not in checked])}." if [f for f in named if f not in checked] else
                     "; every one was checked."))))
    cap.looked(argparse.Namespace(capture=str(new["capture"]), by=a.by, loads=new["name"], note=note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
