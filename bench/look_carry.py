#!/usr/bin/env python3
"""Whether a load made again can rest on the look an earlier capture of it had: track, part, region and reference video.

    <python> bench/look_carry.py --new data/<day>_<load> --read data/<day>_<earlier load> [--old-reference OLD.mp4]
                                 [--by WHO --note-file WORDS.txt --reference-checked CHECK.txt --checked-frames 658-680
                                  --not-checked 821-821=WHY]

**What it is for.** A load made again (a new reference, a new text, a mend that does not touch its masks) is owed a look
like any load, and a blind reader's look costs the card time. The card measured by hand, every time, whether the new
capture's track, part and region were equal to, or a strict subset of, an earlier capture a blind reader had read, and the
lead then recorded a look resting on that reading (mrcircle's `measure_against.py`, ported here: mrpen's of 2026-10-13).

**What it prints**, by source frame, on the frames both captures hold (the new one may start inside the earlier one): the
subject's track and part (equal, a subset, or LARGER, with the px added and where), and the planned region (equal, a
subset, or LARGER by how many 16 px cells and where); the rules new and gone between the two captures' flags, and the new
one's verdict; and which reading the earlier capture's last look rests on (its words, and a blind reading beside it).

**The reference video**, read through the builder's own view of each load (`masked_job_graph.one_subject_view` and
`motion_of`: a pair's second reference, a mesh or no video; the job beside the capture, `<capture>_job/job.plan.json`, or
`--new-job`/`--read-job`). It is THE SAME only when both show the model the same kind of motion video, as many files, each
beginning on the same source frame at the same short edge, and each earlier file the one its reading saw: by the digest the
strip's record kept (`shown.json` `motion[*].sha256`, which the strip's own change (mrvid, 2026-10-15) adds; no strip drawn before
it has one) where there is one, else by the file's time
(written before that strip; a reference remade in place keeps its path) and its digest now. The strip is the one the last
look rests on (that look's own copy of the record); with no look, no later than the capture's own written time. The new job's
reference record must be no older than the file it names (`REFERENCE_RECORD_SLACK_S`), or no look rests on it. Anything else, a job not found among them, is CHANGED. The new capture's own strip must have been drawn with
the file there now, or no look rests on it.

**Where the reference CHANGED, a check of the new one is owed** on the frames, inside both loads, where the two references'
records say they drew differently (`job.reference.drawn`: the mouth a frame at a time, its inside, the eye's form and the parts
left out; every frame when a job holds no record), where their pixels differ (`REFERENCE_DIFF`, with `--old-reference` once
for each reference: the earlier file as its reader saw it, by the kept digest or by its name and a time before the strip,
refused when it is not; a frame one file lacks or at another size is owed), and where the earlier look's words name a frame
(`frames_named`: not the load's own span in its header, not a count). Each is printed. When none owes a frame, it asks for
`--old-reference`.

**With `--by`** it records a look on the new capture (`capture_masked_run.py looked`, one line in its `looks.json`) that
rests on that reading, ONLY when every frame of every mask is equal or a subset, the new load's frames all inside the earlier
one's, and the earlier capture has a reading to rest on: a look in its `looks.json`, or a blind reading beside it
(`look_blind/reading*.md`). Otherwise it refuses and says a reader is owed. The look is the `--by`
person's, with their own words (`--note-file`) and the figures; a look by the capture's maker is refused by the render step.
Where the masks carry and the reference CHANGED, it records only with the check of the new reference named: the checker's
words (`--reference-checked`, a file: who looked at what, and when), the frames looked at (`--checked-frames`, inside the new
load), and every frame owed either among them or given a reason (`--not-checked A-B=WHY`): else it refuses, naming them (the
lead, 2026-10-15: 630c's look said every frame was checked while the reader's 659-671 were not). The look's note says the
reference changed, both digests, the frames owed and why, the frames checked, and those not, with their reasons.

**What it cannot see.** A subset of the masks is not a subset of what a render draws: the text may differ, and the reading
said what it saw then. What the earlier reference was when it was CHANGED and no `--old-reference` is given (its file is gone
from the path, or was never named), so its pixels are not compared. Without a kept digest, a file's written time is the
filesystem's (the share's clock, read in this shell's zone): a copy that keeps an old time passes for unchanged, a rewrite of
the same bytes reads as CHANGED, and a write in the same second as the strip counts as CHANGED. A blind reading's own file is
not read for frames, and "from 658" names 658 only. The earlier words name what
the reader wrote, which may be of the masks and not the reference: a frame owed that way can be given a reason. Whether the
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
    """What the load showed the model as its reference video(s), read through the builder's own view of the load
    (`masked_job_graph.one_subject_view`, so a pair's second reference is `motion_file_2`, and `motion_of`, so a mesh or no video
    is said): each file with its digest now, when it was last written, the source frame it begins on and its short edge; and the
    strip the capture's reading saw: its time and, where the strip's record kept them (`shown.json` `motion[*].sha256`, from
    2026-10-15), the digests it was drawn with. The strip is the one the last look rests on (that look's own copy of the record,
    which a redraw does not move); with no look, the record beside the strip, no later than the first blind reading's file."""
    capture, name = load["capture"], load["name"]
    shown_file = capture / "looks" / f"{name}.shown.json"
    kept = capture / "looks.json"
    rows = [r for x in (json.loads(kept.read_text())["looks"] if kept.is_file() else []) for r in x.get("shown", []) if r.get("load") == name]
    if rows:
        seen, seen_from = rows[-1], "the strip its last look rests on (that look's own record)"
    else:
        seen = json.loads(shown_file.read_text()) if shown_file.is_file() else {}
        seen_from = "the record beside the strip"
        status = capture / "status.json"
        written = json.loads(status.read_text()).get("written") if status.is_file() else None
        blind = [datetime.datetime.fromtimestamp(b.stat().st_mtime).isoformat(timespec="seconds") for b in capture.glob("look_blind/reading*.md")]
        # with no look's own copy, the strip a blind reader saw was drawn after the capture was last written and before the reading:
        # the capture's own time is the earliest it can have been, so a strip drawn again since cannot pass for the read one
        if blind and written and seen.get("made") and when(written) < when(seen["made"]):
            seen = {"made": written}
            seen_from = "the capture's own written time (a blind reading only: the strip it read was drawn no earlier)"
    digests = {str(m.get("file")): m.get("sha256") for m in seen.get("motion") or [] if m.get("sha256")}
    out = {"job": str(job) if job else None, "strip_made": seen.get("made"), "strip_from": seen_from, "files": [], "motion": None}
    if job is None:
        return out
    sys.path.insert(0, str(HERE.parent / "workflows"))
    import masked_job_graph
    file = json.loads(job.read_text())
    entry = next(x for x in file["loads"] if x.get("name") == name)
    try:
        view = masked_job_graph.one_subject_view({**(file.get("defaults") or {}), **entry})
    except ValueError as e:
        out["job"], out["why"] = None, f"{job.name}: {e}"
        return out
    out["motion"] = masked_job_graph.motion_of(file.get("job") or {"subjects": []}, view) if (file.get("job") or {}).get("subjects") \
        else view.get("motion")
    job_first = (file.get("job") or {}).get("first")              # the builder's rule: a reference with no `motion_first` begins on the job's first frame
    for x in ("", "_2"):
        f = view.get(f"motion_file{x}")
        if not f:
            continue
        p = Path(f)
        one = {"file": f, "sha256": None, "written": None, "first": view.get(f"motion_first{x}", job_first),
               "edge": view.get(f"motion_edge{x}"), "seen_sha256": digests.get(str(f))}
        if p.is_file():
            one["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            one["written"] = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        out["files"].append(one)
    return out


def same_reference(new_ref: dict, read_ref: dict, read_name: str) -> dict:
    """THE SAME only when both jobs are found and show the model the same: the same kind of motion video, as many reference files,
    each beginning on the same source frame at the same short edge, and each of the earlier files the one its reading saw: by the
    digest its strip's record kept where it kept one, else by the file's time (written before that strip) and its digest now.
    Anything else is CHANGED, with why. Same-second times count as CHANGED."""
    def changed(why):
        return {"same": False, "why": why}
    for ref, who in ((read_ref, f"the earlier capture ({read_name})"), (new_ref, "the new capture")):
        if ref["job"] is None:
            return changed(ref.get("why") or f"no job file found for {who}: its reference is not known")
    if new_ref["motion"] != read_ref["motion"]:
        return changed(f"the earlier load shows the model {read_ref['motion']!r} as its motion and the new one {new_ref['motion']!r}")
    if len(new_ref["files"]) != len(read_ref["files"]):
        return changed(f"the earlier load names {len(read_ref['files'])} reference video(s) and the new one {len(new_ref['files'])}")
    strip = when(read_ref["strip_made"])
    for a, b in zip(read_ref["files"], new_ref["files"]):
        if b["sha256"] is None:
            return changed(f"{b['file']} is not there to read")
        if a["first"] != b["first"] or a["edge"] != b["edge"]:
            return changed(f"{Path(a['file']).name} begins on source frame {a['first']} at a short edge of {a['edge']} for the earlier load, "
                           f"and {b['first']} at {b['edge']} for the new one")
        if a["seen_sha256"]:
            if a["seen_sha256"] != b["sha256"]:
                return changed(f"the earlier strip was drawn with {Path(a['file']).name} {a['seen_sha256'][:12]} and the new one names "
                               f"{Path(b['file']).name} {b['sha256'][:12]}")
            continue
        if a["sha256"] is None:
            return changed(f"{a['file']} is not there to read")
        if strip is None:
            return changed("the earlier capture says nothing of when its strip was drawn")
        if when(a["written"]) >= strip:
            return changed(f"{a['file']} was written at {a['written']}, not before the earlier strip was drawn ({read_ref['strip_made']}): "
                           "it is not the file the earlier reader saw")
        if a["sha256"] != b["sha256"]:
            return changed(f"{Path(a['file']).name} {a['sha256'][:12]} and {Path(b['file']).name} {b['sha256'][:12]} differ")
    return {"same": True, "why": (f"neither load shows the model a reference video ({new_ref['motion']})" if not new_ref["files"] else
                                  "by the digests the earlier strip's record kept" if all(a["seen_sha256"] for a in read_ref["files"]) else
                                  "the earlier file unwritten since its strip and every digest equal")}


#: What a drawn reference shows a frame at a time, read from a job's reference record (`job.reference.drawn`, `head_reference`):
#: the mouth by the lip lines, by the class map's own mouth, or none.
MOUTH_STATES = ("the lip lines", "the class map", "no mouth")


def drawn_frames(job: Path | None, first: int, last: int) -> dict:
    """A frame at a time on first-last, what the job's reference record says it drew: the mouth (`MOUTH_STATES`), the eye's form, the
    mouth's inside and the parts left out. Empty where the job has no reference record."""
    if job is None:
        return {}
    said = ((json.loads(job.read_text()).get("job") or {}).get("reference") or {}).get("drawn")
    if not said:
        return {}
    mouth = said.get("mouth") or {}
    def inside(spans_, f):
        return any(a <= f <= b for a, b in spans_ or [])
    bare = [s for rule in (mouth.get("no_mouth_on") or {}).values() for s in rule]
    out = {}
    for f in range(first, last + 1):
        if not mouth.get("drawn") or inside(bare, f):
            m = "no mouth"
        elif "frames_by_the_lip_lines" in mouth:
            m = "the lip lines" if inside(mouth["frames_by_the_lip_lines"], f) else "the class map" if inside(mouth.get("frames_by_the_class_map"), f) else "no mouth"
        else:
            m = mouth.get("drawn_as", "the lip lines")
        shut = said.get("eyes_shut") or {}
        eyes = said.get("iris") if not inside(shut.get("drawn_shut_on") or shut.get("frames"), f) else f"shut, drawn {shut.get('form')!r}"
        out[f] = (m, eyes, mouth.get("inside"), tuple(said.get("parts_left_out") or ()))
    return out


#: How long after the reference's file its record may be written and still be its record (`job.reference.made`): the tool writes the
#: video, then its sheet and its record. Reasoned: on 630's remake of 2026-10-14 the record followed the video by 3 seconds.
REFERENCE_RECORD_SLACK_S = 120

#: Two references' frames differ where this many pixels or more differ by more than `level` grey levels in any channel.
#: Inherited: mrarc's ref_diff_frames.py (2026-10-15); on 630's old and remade references the frames whose drawing did not change
#: read 0 such pixels and those that did 166 to 953.
REFERENCE_DIFF = dict(level=40, px=1)


def pixel_frames(old: str, old_first: int, new: str, new_first: int, first: int, last: int) -> list[int]:
    """The source frames on first-last whose pixels differ between two reference videos (`REFERENCE_DIFF`), each read from its own
    first source frame."""
    import cv2
    def frames(path, at):
        v, f, out = cv2.VideoCapture(path), at, {}
        while True:
            ok, x = v.read()
            if not ok:
                return out
            if first <= f <= last:
                out[f] = x
            f += 1
    a, b = frames(old, old_first), frames(new, new_first)
    if not a or not b:
        raise SystemExit(f"no frame of {first}-{last} could be read from {old if not a else new}")
    # a frame one file lacks, or at another size, cannot be compared: it is owed, not passed
    return sorted(f for f in range(first, last + 1) if f not in a or f not in b or a[f].shape != b[f].shape
                  or int((np.abs(a[f].astype(int) - b[f].astype(int)).max(2) > REFERENCE_DIFF["level"]).sum()) >= REFERENCE_DIFF["px"])


def frames_named(note: str, first: int, last: int, whole: list[tuple[int, int]] = ()) -> list[int]:
    """The source frames a look's words name inside first-last: a number, or a range ("813-818", "813-18", "672 to 680") cut to it.
    Not one: a number inside a name (`lead_fp_0630b`), a decimal, a count ("154 frames"), or a range that is a load's own whole span
    (`whole`: "(69 frames, 1466-1534)" heads a reading). "from 658" names 658 only."""
    out = set()
    for m in re.finditer(r"(?<![\w.])(\d{2,5})(?:\s*(?:[-\u2013]|to)\s*(\d{1,5}))?(?!\w|\.\d)(?!\s*frames?\b)", note):
        a, tail = m.group(1), m.group(2) or m.group(1)
        b = int(a[:len(a) - len(tail)] + tail) if len(tail) < len(a) else int(tail)
        if m.group(2) and (int(a), b) in set(whole):
            continue
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
    ap.add_argument("--checked-frames", help="with --reference-checked: the source frames of the new reference looked at, comma separated "
                                             "(F or A-B)")
    ap.add_argument("--not-checked", action="append", default=[], metavar="A-B=WHY",
                    help="with --reference-checked: frames the check is owed on and was not made, each with why; repeatable")
    ap.add_argument("--old-reference", action="append", default=[],
                    help="the earlier load's reference video as its reader saw it (a remake moves it aside, *_SUPERSEDED_*), one for each "
                         "reference the load names, in its order: the frames whose pixels differ from the new one are then owed a check")
    a = ap.parse_args(argv)
    new, read = load_of(Path(a.new)), load_of(Path(a.read))
    if new["subject"] != read["subject"]:
        raise SystemExit(f"the new load is of {new['subject']!r} and the earlier one of {read['subject']!r}: not one subject's load made again")
    c, r, reading = compare(new, read), rules(new, read), reading_of(read)
    new_ref = reference_of(new, job_of(new["capture"], new["name"], a.new_job))
    read_ref = reference_of(read, job_of(read["capture"], read["name"], a.read_job))
    ref = same_reference(new_ref, read_ref, read["name"])
    # the new capture's own strip: the look recorded on it rests on what that strip showed, so its reference must be the file there now
    stale = [one["file"] for one in new_ref["files"] if one["sha256"] and (
        (one["seen_sha256"] and one["seen_sha256"] != one["sha256"])
        or (not one["seen_sha256"] and new_ref["strip_made"] and when(one["written"]) >= when(new_ref["strip_made"])))]
    lo, hi = max(new["first"], read["first"]), min(new["first"] + new["frames"], read["first"] + read["frames"]) - 1
    whole = [(read["first"], read["first"] + read["frames"] - 1), (new["first"], new["first"] + new["frames"] - 1)]
    named = frames_named(reading["look"]["note"], lo, hi, whole) if reading["look"] else []
    owed: dict[str, list[int]] = {}
    if not ref["same"]:
        # THE FRAMES THE CHECK IS OWED ON (the lead, 2026-10-15, on the outside reviewer's G1): where the two references' records say
        # they drew differently, where their pixels differ (with --old-reference), and where the earlier look's words name a frame
        a_, b_ = drawn_frames(job_of(read["capture"], read["name"], a.read_job), lo, hi), drawn_frames(job_of(new["capture"], new["name"], a.new_job), lo, hi)
        if a_ and b_:
            owed["the references' records say they drew it differently"] = [f for f in range(lo, hi + 1) if a_.get(f) != b_.get(f)]
        else:
            owed["a job holds no reference record, so nothing says where they drew alike"] = list(range(lo, hi + 1))
        if a.old_reference:
            if len(a.old_reference) != len(read_ref["files"]) or len(new_ref["files"]) != len(read_ref["files"]):
                raise SystemExit(f"--old-reference: {len(a.old_reference)} given; the earlier load names {len(read_ref['files'])} reference "
                                 f"video(s) and the new one {len(new_ref['files'])}: one for each, in order")
            begins = lambda one, load: int(one["first"]) if one["first"] is not None else load["first"]  # noqa: E731
            differ: set[int] = set()
            for given, was, now in zip(a.old_reference, read_ref["files"], new_ref["files"]):
                old = Path(given)
                if not old.is_file():
                    raise SystemExit(f"--old-reference {old}: not a file")
                digest = hashlib.sha256(old.read_bytes()).hexdigest()
                stamp = datetime.datetime.fromtimestamp(old.stat().st_mtime).isoformat(timespec="seconds")
                seen = was["seen_sha256"]
                if (seen and seen != digest) or (not seen and (old.name != Path(was["file"]).name or read_ref["strip_made"] is None
                                                               or when(stamp) >= when(read_ref["strip_made"]))):
                    raise SystemExit(f"--old-reference {old.name} ({digest[:12]}, written {stamp}) is not the file the earlier strip was drawn with"
                                     + (f" ({seen[:12]})" if seen else f" ({Path(was['file']).name}, drawn {read_ref['strip_made']})"))
                differ.update(pixel_frames(str(old), begins(was, read), now["file"], begins(now, new), lo, hi))
            owed["their pixels differ"] = sorted(differ)
        owed["the earlier look's words name it"] = named
    must = sorted({f for fs in owed.values() for f in fs})
    # the new job's reference record against the file it names: a record older than the file says nothing of what the file holds
    record_made = ((((json.loads(job_of(new["capture"], new["name"], a.new_job).read_text()).get("job") or {}).get("reference") or {}).get("made"))
                   if job_of(new["capture"], new["name"], a.new_job) else None)
    newer = [one["file"] for one in new_ref["files"] if record_made and one["written"] and
             datetime.datetime.fromtimestamp(Path(one["file"]).stat().st_mtime, datetime.timezone.utc)
             > datetime.datetime.fromisoformat(record_made.replace("Z", "+00:00")) + datetime.timedelta(seconds=REFERENCE_RECORD_SLACK_S)]
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
    for why, fs in owed.items():
        print(f"  a check of the new reference is owed where {why}: {spans(fs)}")
    if not ref["same"] and not a.old_reference:
        print("  the pixels were not compared: give --old-reference (the earlier reference as its reader saw it) to owe them too")
    print("  DOES NOT CARRY: " + ("frames outside the earlier load" if c["outside_the_read"] else "a mask is LARGER on " + ", ".join(k for k, v in c["larger"].items() if v))
          + ": a reader is owed" if not c["carries"] else
          "  CARRIES: every frame of every mask equal or a subset, and the same reference" if ref["same"] else
          "  THE MASKS CARRY AND THE REFERENCE CHANGED: it carries only with the check of the new reference named "
          "(--reference-checked WORDS.txt --checked-frames F,A-B,...), every frame owed checked or given a reason (--not-checked A-B=WHY)")
    if newer:
        print(f"  THE NEW JOB'S REFERENCE RECORD IS OLDER THAN THE FILE IT NAMES: {', '.join(newer)} was written after {record_made}; "
              "build the job's reference again before a look rests on it")
    if stale:
        print(f"  THE NEW CAPTURE'S OWN STRIP WAS DRAWN WITH ANOTHER FILE: {', '.join(stale)} changed after it ({new_ref['strip_made']}); draw the "
              "strip again before a look rests on it")
    if not a.by:
        return 0
    if not c["carries"] or not (look or reading["blind_readings"]) or not new["planned"]:
        why = ("it does not carry" if not c["carries"] else "the earlier capture has no look and no blind reading to rest on" if not (look or reading["blind_readings"])
               else "the new capture is not on the node's plan")
        raise SystemExit(f"no look recorded: {why}. A reader is owed")
    if not a.note_file or not Path(a.note_file).read_text().strip():
        raise SystemExit("--note-file: the person's own words on what they rest the look on, a file")
    if stale:
        raise SystemExit(f"no look recorded: the new capture's strip was drawn with another file than {', '.join(stale)} holds now: draw it again")
    if newer:
        raise SystemExit(f"no look recorded: the new job's reference record is older than {', '.join(newer)}: build its reference again")
    checked: list[int] = []
    skipped: list[tuple[int, int, str]] = []
    if not ref["same"]:
        if not a.reference_checked or not Path(a.reference_checked).read_text().strip() or not a.checked_frames:
            raise SystemExit(f"no look recorded: the reference CHANGED ({ref['why']}). Name the check of the new reference: "
                             "--reference-checked WORDS.txt (who looked at what, and when) and --checked-frames F,A-B,...")
        try:
            checked = sorted({f for x in a.checked_frames.split(",") if x.strip()
                              for f in range(int(x.split("-")[0]), int(x.split("-")[-1]) + 1)})
        except ValueError as e:
            raise SystemExit(f"--checked-frames: source frames, F or A-B, comma separated ({e})") from e
        outside = [f for f in checked if not new["first"] <= f < new["first"] + new["frames"]]
        if not checked or outside:
            raise SystemExit(f"--checked-frames: {spans(outside) if outside else 'none'} not inside the new load "
                             f"({new['first']}-{new['first'] + new['frames'] - 1})")
        for x in a.not_checked:
            rng, sep, why = x.partition("=")
            try:
                f0, f1 = (int(v) for v in (rng.split("-") * 2)[:2])
            except ValueError:
                raise SystemExit(f"--not-checked {x!r} is not A-B=WHY") from None
            if not sep or not why.strip() or f0 > f1:
                raise SystemExit(f"--not-checked {x!r} is not A-B=WHY, with the why")
            skipped.append((f0, f1, " ".join(why.split())))
        if not must:
            raise SystemExit("no look recorded: the reference CHANGED and nothing says on which frames (the records drew alike, no words name "
                             "a frame, no --old-reference): give --old-reference so its pixels say")
        # THE CASE THAT REFUSES: a frame the check is owed on, neither checked nor given a reason (630c, 2026-10-15: 659-671 left
        # out of a check while the earlier reader named 658-680)
        left = [f for f in must if f not in checked and not any(f0 <= f <= f1 for f0, f1, _ in skipped)]
        if left:
            raise SystemExit(f"no look recorded: a check of the new reference is owed on {spans(left)}, neither in --checked-frames nor in "
                             "--not-checked A-B=WHY (" + "; ".join(f"{why}: {spans([f for f in fs if f in left])}" for why, fs in owed.items()
                                                               if any(f in left for f in fs)) + ")")
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
               + " A check was owed " + ("; ".join(f"where {why}: {spans(fs)}" for why, fs in owed.items()) or "on no frame")
               + ("" if a.old_reference else " (the pixels not compared: no --old-reference)")
               + (". Not checked: " + "; ".join(f"{f0}-{f1}, {why}" for f0, f1, why in skipped) + "." if skipped else
                  ". Every frame owed was checked.")))
    cap.looked(argparse.Namespace(capture=str(new["capture"]), by=a.by, loads=new["name"], note=note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
