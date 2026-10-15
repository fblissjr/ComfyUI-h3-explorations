#!/usr/bin/env python3
"""`bench/look_carry.py` on made captures: each case is a way a load made again could rest on a reading it no longer matches.

  equal_and_subset_carry       track, part and region equal, or the new one a subset, on every frame: it carries; a frame of the new
                               load inside the earlier one when it starts later
  larger_does_not_carry        a px added to the track or the part, or a cell to the region, on one frame: LARGER, with how much and
                               where, and it does not carry; a frame outside the earlier load does not carry
  a_look_only_when_it_carries  with --by: a look recorded on the new capture resting on the earlier reading (its look or its blind
                               reading), with the person's words and the figures; refused when it does not carry, with no reading to
                               rest on, without words, or by the capture's maker; rules new and gone said
  a_changed_reference_needs_its_check
                               the masks equal and the reference remade IN PLACE (same path, written after the earlier strip; both
                               digests read now are equal): CHANGED; so is a different file with other bytes, and a load whose job is
                               not found. With --by it records only with the check of the new reference named (words, frames inside
                               the load), and the note says the reference changed and which frames the earlier words name unchecked

    <comfy venv python> bench/check_look_carry.py

No model, no card, no ffmpeg.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import look_carry as lc  # noqa: E402

N, H, W = 6, 48, 64


def capture(root: Path, name: str, first: int, track: np.ndarray, parts: np.ndarray, region: np.ndarray, flags=(), maker="the card",
            look=None, blind=False, written="2026-01-01T10:00:00", strip_made="2026-01-01T11:00:00", reference=b"a reference",
            ref_path=None, ref_written="2026-01-01T09:00:00", job=True, motion_first=None) -> Path:
    c = root / f"2026-01-01_{name}"
    (c / "runs" / name).mkdir(parents=True)
    (c / "subjects" / "a").mkdir(parents=True)
    (c / "looks").mkdir()
    (c / "manifest.json").write_text(json.dumps({"first_frame": first, "runs": [{"name": name, "subject": "a", "first_source_frame": first, "planned": True}],
                                                 "subjects": [{"label": "a", "sightings": [{"by": "pv"}]}]}))
    np.savez(c / "runs" / name / "region.npz", region=region)
    np.savez(c / "subjects" / "a" / "masks__pv.npz", track=np.packbits(track, axis=-1), parts=np.packbits(parts, axis=-1))
    (c / "flags.json").write_text(json.dumps({"verdict": "clear", "flags": [{"rule": r} for r in flags]}))
    (c / "status.json").write_text(json.dumps({"written": written, "made_by": maker}))
    (c / "looks" / f"{name}.shown.json").write_text(json.dumps({"load": name, "made": strip_made, "frames": [first, first + len(region) - 1]}))
    if look:
        (c / "looks.json").write_text(json.dumps({"looks": [look]}))
    if blind:
        (c / "look_blind").mkdir()
        (c / "look_blind" / "reading.md").write_text("a blind reading")
    if job:                                                             # the job beside the capture, as the chain writes it, naming the reference
        ref = Path(ref_path) if ref_path else root / f"{name}_reference.mp4"
        ref.write_bytes(reference)
        at = datetime.datetime.fromisoformat(ref_written).timestamp()
        os.utime(ref, (at, at))
        (root / f"{c.name}_job").mkdir()
        (root / f"{c.name}_job" / "job.plan.json").write_text(json.dumps({"job": {"capture": str(c)}, "loads": [{"name": name, "motion_file": str(ref),
                                                                                          **({"motion_first": motion_first} if motion_first else {})}]}))
    return c


def masks(n=N):
    track = np.zeros((n, H, W * 8), bool)
    track[:, 10:30, 10:60] = True
    parts = np.zeros_like(track)
    parts[:, 12:20, 20:40] = True
    region = np.zeros((n, H, W), bool)
    region[:, 5:20, 5:30] = True
    return track, parts, region


def run(argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = lc.main(argv)
    return code, out.getvalue()


def equal_and_subset_carry() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t, p, r = masks()
        read = capture(d, "a_old", 100, t, p, r, flags=["part_in_pieces"], blind=True)
        t2, p2, r2 = t.copy(), p.copy(), r.copy()
        p2[2, 12:14, 20:40] = False                                     # the part smaller on one frame
        r2[3, 5:7, 5:30] = False                                        # the region smaller on another
        new = capture(d, "a_new", 101, t2[1:], p2[1:], r2[1:], flags=["text_names_a_voice_where_there_is_none"])
        c = lc.compare(lc.load_of(new), lc.load_of(read))
        # THE CASE THAT GOES RED when a subset is called larger, or the new load's frames are matched off by one
        assert c["carries"] and c["masks"]["parts"][102] == "subset" and c["masks"]["region"][103] == "subset" and c["masks"]["track"][101] == "equal", c["masks"]
        code, said = run(["--new", str(new), "--read", str(read)])
        assert code == 0 and "  CARRIES:" in said and "the reference: THE SAME" in said and "rules gone: [('part_in_pieces', '', '')]" in said and "rules new: [('text_names_a_voice_where_there_is_none', '', '')]" in said, said
    return "equal or a subset on every frame carries, a new load starting inside the earlier one matched frame for frame; rules new and gone said"


def larger_does_not_carry() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t, p, r = masks()
        read = capture(d, "a_old", 100, t, p, r, blind=True)
        for what, change in (("track", lambda t, p, r: t.__setitem__((4, 40, 300), True)), ("parts", lambda t, p, r: p.__setitem__((4, 40, 300), True)),
                             ("region", lambda t, p, r: r.__setitem__((4, slice(30, 32), slice(40, 43)), True))):
            t2, p2, r2 = t.copy(), p.copy(), r.copy()
            change(t2, p2, r2)
            new = capture(d / what, "a_new", 100, t2, p2, r2) if (d / what).mkdir() is None else None
            c = lc.compare(lc.load_of(new), lc.load_of(read))
            # THE CASE THAT GOES RED when one px or one cell more on one frame lets the look carry
            assert not c["carries"] and c["larger"][what] == [104], (what, c["larger"])
        assert c["masks"]["region"][104] == "LARGER by 6 cells at [40, 30, 43, 32]", c["masks"]["region"][104]
        assert lc.compare(lc.load_of(new), lc.load_of(read))["masks"]["track"][104] == "equal"
        long = capture(d / "long", "a_new", 98, *masks()) if (d / "long").mkdir() is None else None
        c = lc.compare(lc.load_of(long), lc.load_of(read))
        assert not c["carries"] and c["outside_the_read"] == [98, 99], c["outside_the_read"]
    return "a px more in the track or the part, or a cell more in the region, on one frame does not carry, said with how much and where; frames outside the earlier load do not"


def a_look_only_when_it_carries() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        words = d / "words.txt"
        words.write_text("THE LEAD'S OWN WORDS on what was compared")
        t, p, r = masks()
        read = capture(d, "a_old", 100, t, p, r, blind=True, look={"by": "a reader", "at": "2026-01-01T09:00:00", "note": "every frame looked at"})
        new = capture(d, "a_new", 100, t, p, r)
        code, said = run(["--new", str(new), "--read", str(read), "--by", "the lead", "--note-file", str(words)])
        looks = json.loads((new / "looks.json").read_text())["looks"]
        # THE CASE THAT GOES RED when a look is not recorded where it carries, or says nothing of what it rests on
        assert code == 0 and len(looks) == 1 and looks[0]["by"] == "the lead" and looks[0]["loads"] == ["a_new"], looks
        assert looks[0]["note"].startswith("THE LEAD'S OWN WORDS") and "RESTS ON the reading of 2026-01-01_a_old: its look by a reader at 2026-01-01T09:00:00; the blind reading" in looks[0]["note"] \
            and "equal or a subset on all 6 frames" in looks[0]["note"] and "The reference THE SAME: a_new_reference.mp4 sha256" in looks[0]["note"], looks[0]["note"]
        # a blind reading alone is a reading to rest on
        only_blind = capture(d / "b", "a_old", 100, t, p, r, blind=True) if (d / "b").mkdir() is None else None
        new2 = capture(d / "c", "a_new", 100, t, p, r) if (d / "c").mkdir() is None else None
        assert run(["--new", str(new2), "--read", str(only_blind), "--by", "the lead", "--note-file", str(words)])[0] == 0
        for setup, words_, by, refusal in (
                ("larger", words, "the lead", "it does not carry"), ("unread", words, "the lead", "no look and no blind reading"),
                ("no words", d / "empty.txt", "the lead", "--note-file"), ("maker", words, "the card", "made this capture")):
            (d / "empty.txt").write_text("")
            base = d / setup
            base.mkdir()
            r2 = r.copy()
            if setup == "larger":
                r2[0, 40, 60] = True
            old = capture(base, "a_old", 100, t, p, r, blind=setup != "unread")
            nw = capture(base / "n", "a_new", 100, t, p, r2) if (base / "n").mkdir() is None else None
            try:
                run(["--new", str(nw), "--read", str(old), "--by", by, "--note-file", str(words_)])
                raise AssertionError(f"recorded: {setup}")
            except SystemExit as e:
                assert refusal in str(e), (setup, str(e))
            assert not (nw / "looks.json").exists(), f"a look was written: {setup}"
    return ("a look is recorded where it carries, with the person's words, what it rests on (a look, a blind reading) and the figures; refused where it "
            "does not carry, with nothing to rest on, without words, or by the capture's maker")


def a_changed_reference_needs_its_check() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        words, check = d / "words.txt", d / "check.txt"
        words.write_text("THE LEAD'S OWN WORDS")
        check.write_text("A CHECKER looked at the new reference's mouth at 10:00")
        t, p, r = masks()
        reader = {"by": "a reader", "at": "2026-01-01T12:00:00", "note": "the drawn mouth off her face on 102-104 (lead_fp_0100b, 1.5 px)"}
        shared = d / "the_reference.mp4"
        read = capture(d, "a_old", 100, t, p, r, blind=True, look=reader, reference=b"the old drawing", ref_path=shared)
        # remade in place: the same path, new bytes, written after the earlier strip (11:00); both digests read now are equal
        new = capture(d / "n", "a_new", 100, t, p, r, reference=b"the new drawing", ref_path=shared, ref_written="2026-01-02T09:00:00") \
            if (d / "n").mkdir() is None else None
        # the earlier strip redrawn after the mend: its shown record moves past the file, the look's own copy does not
        (read / "looks" / "a_old.shown.json").write_text(json.dumps({"load": "a_old", "made": "2026-01-03T09:00:00", "frames": [100, 105]}))
        looks = json.loads((read / "looks.json").read_text())
        looks["looks"][0]["shown"] = [{"load": "a_old", "made": "2026-01-01T11:00:00"}]
        (read / "looks.json").write_text(json.dumps(looks))
        code, said = run(["--new", str(new), "--read", str(read)])
        # THE CASE THAT GOES RED when a reference remade under its old name passes as the same by its path, its digest now, or a
        # strip redrawn after it
        assert code == 0 and "the reference: CHANGED" in said and "not the file the earlier reader saw" in said and "  CARRIES:" not in said \
            and "THE MASKS CARRY AND THE REFERENCE CHANGED" in said and "the frames the earlier look's words name: 102-104" in said, said
        for setup, kw, why in (("other bytes", {"reference": b"another drawing"}, "differ"), ("no job", {"job": False}, "no job file found"),
                               ("begins later", {"motion_first": 101}, "begins on source frame None for the earlier load and 101")):
            (d / setup).mkdir()
            other = capture(d / setup, "a_new", 100, t, p, r, **kw)
            old = capture(d / setup / "o", "a_old", 100, t, p, r, blind=True) if (d / setup / "o").mkdir() is None else None
            assert "the reference: CHANGED" in run(["--new", str(other), "--read", str(old)])[1] and why in run(["--new", str(other), "--read", str(old)])[1], setup
        base = ["--new", str(new), "--read", str(read), "--by", "the lead", "--note-file", str(words)]
        for extra, refusal in (([], "the reference CHANGED"), (["--reference-checked", str(check)], "the reference CHANGED"),
                               (["--reference-checked", str(check), "--checked-frames", "102,107"], "107 not inside the new load")):
            try:
                run(base + extra)
                raise AssertionError(f"recorded: {extra}")
            except SystemExit as e:
                assert refusal in str(e), (extra, str(e))
            assert not (new / "looks.json").exists(), f"a look was written: {extra}"
        code, said = run(base + ["--reference-checked", str(check), "--checked-frames", "102, 103"])
        note = json.loads((new / "looks.json").read_text())["looks"][0]["note"]
        # THE CASE THAT GOES RED when the look does not say the reference changed, what checked it, or what the earlier words name unchecked
        assert code == 0 and "THE REFERENCE CHANGED" in note and "The new reference checked at 102-103: A CHECKER looked" in note \
            and "not among the frames checked: 104" in note and "name 104, not among the frames checked" in said, (note, said)
    return ("a reference remade in place, other bytes, or a job not found is CHANGED; a look then records only with the new reference's "
            "check named, inside the load, and says the change, both digests and the earlier words' frames left unchecked")


case("equal and subset carry", equal_and_subset_carry)
case("larger does not carry", larger_does_not_carry)
case("a look only when it carries", a_look_only_when_it_carries)
case("a changed reference needs its check", a_changed_reference_needs_its_check)
sys.exit(finish())
