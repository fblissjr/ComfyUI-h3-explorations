#!/usr/bin/env python3
"""Whether a load made again can rest on the look an earlier capture of it had: track, part and region frame by frame.

    <python> bench/look_carry.py --new data/<day>_<load> --read data/<day>_<earlier load> [--by WHO --note-file WORDS.txt]

**What it is for.** A load made again (a new reference, a new text, a mend that does not touch its masks) is owed a look
like any load, and a blind reader's look costs the card time. The card measured by hand, every time, whether the new
capture's track, part and region were equal to, or a strict subset of, an earlier capture a blind reader had read, and the
lead then recorded a look resting on that reading (mrcircle's `measure_against.py`, ported here: mrpen's of 2026-10-13).

**What it prints**, by source frame, on the frames both captures hold (the new one may start inside the earlier one): the
subject's track and part (equal, a subset, or LARGER, with the px added and where), and the planned region (equal, a
subset, or LARGER by how many 16 px cells and where); the rules new and gone between the two captures' flags, and the new
one's verdict; and which reading the earlier capture's last look rests on (its words, and a blind reading beside it).

**With `--by`** it records a look on the new capture (`capture_masked_run.py looked`, one line in its `looks.json`) that
rests on that reading, ONLY when every frame of every mask is equal or a subset, the new load's frames all inside the earlier
one's, and the earlier capture has a reading to rest on: a look in its `looks.json`, or a blind reading beside it
(`look_blind/reading*.md`). Otherwise it refuses and says a reader is owed. The look is the `--by`
person's, with their own words (`--note-file`) and the figures; a look by the capture's maker is refused by the render step.

**What it cannot see.** A subset of the masks is not a subset of what a render draws: the reference and the text may
differ, and the reading said what it saw then. Whether the earlier reading was right. The picture.
"""
from __future__ import annotations

import argparse
import json
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
    a = ap.parse_args(argv)
    new, read = load_of(Path(a.new)), load_of(Path(a.read))
    if new["subject"] != read["subject"]:
        raise SystemExit(f"the new load is of {new['subject']!r} and the earlier one of {read['subject']!r}: not one subject's load made again")
    c, r, reading = compare(new, read), rules(new, read), reading_of(read)
    print(f"{new['name']} ({new['frames']} frames from {new['first']}) against {read['name']} ({read['frames']} from {read['first']}), {new['subject']}:")
    for line in summary(c):
        print("  " + line)
    print(f"  rules gone: {r['gone']} | rules new: {r['new']} | the new capture's verdict: {r['verdict']}")
    look = reading["look"]
    print("  the earlier capture's last look: " + (f"{look['by']}, {look['at']}: {look['note'][:300]}" if look else "NONE")
          + ("; blind reading(s): " + ", ".join(reading["blind_readings"]) if reading["blind_readings"] else ""))
    print("  CARRIES: every frame of every mask equal or a subset" if c["carries"] else
          "  DOES NOT CARRY: " + ("frames outside the earlier load" if c["outside_the_read"] else "a mask is LARGER on " + ", ".join(k for k, v in c["larger"].items() if v))
          + ": a reader is owed")
    if not a.by:
        return 0
    if not c["carries"] or not (look or reading["blind_readings"]) or not new["planned"]:
        why = ("it does not carry" if not c["carries"] else "the earlier capture has no look and no blind reading to rest on" if not (look or reading["blind_readings"])
               else "the new capture is not on the node's plan")
        raise SystemExit(f"no look recorded: {why}. A reader is owed")
    if not a.note_file or not Path(a.note_file).read_text().strip():
        raise SystemExit("--note-file: the person's own words on what they rest the look on, a file")
    import capture_masked_run as cap
    eq = {k: sum(1 for v in rows.values() if v == "equal") for k, rows in c["masks"].items()}
    note = (" ".join(Path(a.note_file).read_text().split())
            + f" RESTS ON the reading of {read['capture'].name}: "
            + "; ".join(([f"its look by {look['by']} at {look['at']}"] if look else []) + ([f"the blind reading {', '.join(reading['blind_readings'])}"] if reading["blind_readings"] else []))
            + f". By bench/look_carry.py: track, part and region equal or a subset on all {new['frames']} frames (equal: "
            + ", ".join(f"{k} {v}" for k, v in eq.items()) + f"); rules gone {r['gone']}, new {r['new']}; verdict {r['verdict']}.")
    cap.looked(argparse.Namespace(capture=str(new["capture"]), by=a.by, loads=new["name"], note=note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
