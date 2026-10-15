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
import hashlib
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
            ref_path=None, ref_written="2026-01-01T09:00:00", job=True, motion_first=None, job_block=None, load=None) -> Path:
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
        entry = load(str(ref)) if load else {"name": name, "motion_file": str(ref), **({"motion_first": motion_first} if motion_first else {})}
        (root / f"{c.name}_job" / "job.plan.json").write_text(json.dumps({"job": {"capture": str(c), **(job_block or {})}, "loads": [entry]}))
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
        same = {"reference": {"drawn": {"mouth": {"drawn": True, "drawn_as": "the lip lines"}}, "made": "2026-01-03T00:00:00Z"}}
        read = capture(d, "a_old", 100, t, p, r, blind=True, look=reader, reference=b"the old drawing", ref_path=shared, job_block=same)
        # remade in place: the same path, new bytes, written after the earlier strip (11:00); both digests read now are equal
        new = capture(d / "n", "a_new", 100, t, p, r, reference=b"the new drawing", ref_path=shared, ref_written="2026-01-02T09:00:00",
                      strip_made="2026-01-02T11:00:00", job_block=same) \
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
                               ("begins later", {"motion_first": 101}, "begins on source frame None at a short edge of None for the earlier load, and 101")):
            (d / setup).mkdir()
            other = capture(d / setup, "a_new", 100, t, p, r, **kw)
            old = capture(d / setup / "o", "a_old", 100, t, p, r, blind=True) if (d / setup / "o").mkdir() is None else None
            assert "the reference: CHANGED" in run(["--new", str(other), "--read", str(old)])[1] and why in run(["--new", str(other), "--read", str(old)])[1], setup
        base = ["--new", str(new), "--read", str(read), "--by", "the lead", "--note-file", str(words)]
        for extra, refusal in (([], "the reference CHANGED"), (["--reference-checked", str(check)], "the reference CHANGED"),
                               (["--reference-checked", str(check), "--checked-frames", "102,107"], "107 not inside the new load"),
                               # THE CASE THAT GOES RED when a frame the earlier words name is neither checked nor given a reason (630c)
                               (["--reference-checked", str(check), "--checked-frames", "102,103"], "a check of the new reference is owed on 104"),
                               (["--reference-checked", str(check), "--checked-frames", "102,103", "--not-checked", "104-104"], "is not A-B=WHY")):
            try:
                run(base + extra)
                raise AssertionError(f"recorded: {extra}")
            except SystemExit as e:
                assert refusal in str(e), (extra, str(e))
            assert not (new / "looks.json").exists(), f"a look was written: {extra}"
        code, said = run(base + ["--reference-checked", str(check), "--checked-frames", "102-103", "--not-checked", "104-104=HER HAND OVER IT"])
        note = json.loads((new / "looks.json").read_text())["looks"][0]["note"]
        # THE CASE THAT GOES RED when the look does not say the reference changed, what checked it, what was owed, or what was left and why
        assert code == 0 and "THE REFERENCE CHANGED" in note and "The new reference checked at 102-103: A CHECKER looked" in note \
            and "where the earlier look's words name it: 102-104" in note and "Not checked: 104-104, HER HAND OVER IT." in note, (note, said)
    return ("a reference remade in place, other bytes, or a job not found is CHANGED; a look then records only with the new reference's "
            "check named, inside the load, and says the change, both digests and the earlier words' frames left unchecked")


def what_the_check_is_owed_on() -> str:
    """The outside reviewer's G1 (2026-10-15): the frames owed a check come from the references' own records and pixels as well as the
    earlier words, read without the reader's header span or counts; a digest the strip's record kept decides alone; a pair's second
    reference and the kind of motion video are compared."""
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t, p, r = masks()
        # the words: a header span, a count and "A to B", inside a load of 100-105
        assert lc.frames_named("Every frame of a_old (6 frames, 100-105): 6 frames; the mouth off on 101 to 102 and 104", 100, 105,
                               [(100, 105)]) == [101, 102, 104]
        # THE CASE THAT GOES RED when a header span counts as named (7 of 21 captures of 10/14 named every frame) or a count does
        assert lc.frames_named("(105 frames, 100-105)", 100, 105, [(100, 105)]) == [], "the header named frames"
        # the records: the earlier drew the class map's mouth on every frame, the new one no mouth on 103-105
        old_drawn = {"reference": {"drawn": {"mouth": {"drawn": True, "drawn_as": "the class map"}, "iris": "contrast"}}}
        new_drawn = {"reference": {"drawn": {"mouth": {"drawn": True, "drawn_as": "the class map", "frames_by_the_class_map": [[100, 102]],
                                                       "frames_by_the_lip_lines": [], "no_mouth_on": {"THE GATE": [[103, 105]]}}, "iris": "contrast"}}}
        def mp4(path, frames):
            v = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24, (32, 24))
            for x in frames:
                v.write(x)
            v.release()
            return path
        plain = [np.full((24, 32, 3), 90, np.uint8) for _ in range(6)]
        moved = [x.copy() for x in plain]
        moved[1][5:10, 5:10] = 255                                      # frame 101 drawn differently
        (d / "aside").mkdir()
        old_ref = mp4(d / "aside" / "shared.mkv", plain)                # moved aside under its own name, as a remake does
        read = capture(d, "a_old", 100, t, p, r, blind=True, look={"by": "a reader", "at": "2026-01-01T12:00:00", "note": "fine on 105"},
                       ref_path=d / "shared.mkv", reference=old_ref.read_bytes(), job_block=old_drawn)
        new = capture(d / "n", "a_new", 100, t, p, r, ref_path=d / "shared.mkv", reference=mp4(d / "tmp.mkv", moved).read_bytes(),
                      ref_written="2026-01-02T09:00:00", strip_made="2026-01-02T11:00:00", job_block=new_drawn) if (d / "n").mkdir() is None else None
        at = datetime.datetime.fromisoformat("2026-01-01T09:00:00").timestamp()
        os.utime(old_ref, (at, at))
        code, said = run(["--new", str(new), "--read", str(read), "--old-reference", str(old_ref)])
        # THE CASE THAT GOES RED when the records' or the pixels' frames are not owed a check
        assert "the references' records say they drew it differently: 103-105" in said and "their pixels differ: 101" in said \
            and "the earlier look's words name it: 105" in said, said
        # the new capture's own reference written again after its strip: a look cannot rest on that strip
        touched = d / "shared.mkv"
        os.utime(touched, (datetime.datetime.fromisoformat("2026-01-04T09:00:00").timestamp(),) * 2)
        words = d / "w.txt"
        words.write_text("WORDS")
        try:
            run(["--new", str(new), "--read", str(read), "--by", "the lead", "--note-file", str(words)])
            raise AssertionError("a look rested on a strip drawn with another file")
        except SystemExit as e:
            # THE CASE THAT GOES RED when a look rests on a new strip whose reference changed after it was drawn
            assert "the new capture's strip was drawn with another file" in str(e), str(e)
        os.utime(touched, (datetime.datetime.fromisoformat("2026-01-02T09:00:00").timestamp(),) * 2)
        (d / "late").mkdir()
        late = d / "late" / "shared.mkv"
        late.write_bytes(old_ref.read_bytes())                          # written now: after the earlier strip, so not what its reader saw
        try:
            run(["--new", str(new), "--read", str(read), "--old-reference", str(late)])
            raise AssertionError("an old reference written after the strip was taken")
        except SystemExit as e:
            assert "is not the file the earlier strip was drawn with" in str(e), str(e)
        # a digest the look's strip record kept decides alone: the old file gone, the same bytes now carry; other bytes do not
        for bytes_, same in ((b"the drawing", True), (b"another drawing", False)):
            e = d / f"digest_{same}"
            e.mkdir()
            kept = {"by": "a reader", "at": "2026-01-01T12:00:00", "note": "", "shown": [{"load": "a_old", "made": "2026-01-01T11:00:00",
                    "motion": [{"file": str(e / "gone.mkv"), "first": None, "sha256": hashlib.sha256(b"the drawing").hexdigest()}]}]}
            old = capture(e, "a_old", 100, t, p, r, blind=True, look=kept, ref_path=e / "gone.mkv")
            (e / "gone.mkv").unlink()
            nw = capture(e / "n", "a_new", 100, t, p, r, reference=bytes_, ref_path=e / "gone.mkv",
                         ref_written="2026-01-03T09:00:00") if (e / "n").mkdir() is None else None
            got = run(["--new", str(nw), "--read", str(old)])[1]
            # THE CASE THAT GOES RED when a kept digest is not what decides
            assert ("the reference: THE SAME (by the digests the earlier strip's record kept)" in got) == same, (same, got)
        # the kind of motion video, and a pair's second reference
        f = d / "mode"
        f.mkdir()
        old = capture(f, "a_old", 100, t, p, r, blind=True, load=lambda ref: {"name": "a_old", "motion": "none"})
        nw = capture(f / "n", "a_new", 100, t, p, r, load=lambda ref: {"name": "a_new", "motion": "mesh"}) if (f / "n").mkdir() is None else None
        assert "the earlier load shows the model 'none' as its motion and the new one 'mesh'" in run(["--new", str(nw), "--read", str(old)])[1]
        g = d / "pair"
        g.mkdir()
        pair = lambda name, second: (lambda ref: {"name": name, "subjects": [{"subject": "a", "region": "face", "motion_file": ref},  # noqa: E731
                                                                            {"subject": "b", "region": "face", "motion_file": str(g / second)}]})
        (g / "b1.mkv").write_bytes(b"b one")
        (g / "b2.mkv").write_bytes(b"b two")
        for b in ("b1.mkv", "b2.mkv"):
            os.utime(g / b, (at, at))
        old = capture(g, "a_old", 100, t, p, r, blind=True, ref_path=g / "a.mkv", load=pair("a_old", "b1.mkv"))
        nw = capture(g / "n", "a_new", 100, t, p, r, ref_path=g / "a.mkv", load=pair("a_new", "b2.mkv")) if (g / "n").mkdir() is None else None
        got = run(["--new", str(nw), "--read", str(old)])[1]
        # THE CASE THAT GOES RED when a two-subject load's references are not read (both would be "none named": THE SAME)
        assert "the reference: CHANGED (b1.mkv" in got and "b2.mkv" in got, got
    return ("owed a check: where the references' records drew differently, where their pixels differ (an old reference written after the "
            "strip refused), and where the earlier words name a frame (no header span, no count, \"A to B\" read); a kept digest decides "
            "alone; the kind of motion video and a pair's second reference compared")


def the_reader_named_gaps() -> str:
    """The cold reader's gaps of 2026-10-15, each a way a changed reference could still pass unchecked."""
    import cv2
    def mp4(path, frames):
        v = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24, (frames[0].shape[1], frames[0].shape[0]))
        for x in frames:
            v.write(x)
        v.release()
        return path
    rec = lambda **k: {"reference": {"drawn": {"mouth": {"drawn": True, "drawn_as": "the lip lines"}, "iris": "contrast", **k}}}  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t, p, r = masks()
        words = d / "w.txt"
        words.write_text("WORDS")
        check = d / "c.txt"
        check.write_text("A CHECKER")
        def pair(name, old_kw, new_kw, ref_old=b"old", ref_new=b"new"):
            (d / name).mkdir()
            old = capture(d / name, "a_old", 100, t, p, r, blind=True, look={"by": "x", "at": "2026-01-01T12:00:00", "note": ""},
                          reference=ref_old, **old_kw)
            (d / name / "n").mkdir()
            new = capture(d / name / "n", "a_new", 100, t, p, r, reference=ref_new, ref_written="2026-01-02T09:00:00",
                          strip_made="2026-01-02T11:00:00", **new_kw)
            return new, old
        # a job with no reference record owes every frame; an eye drawn another way owes every frame
        new, old = pair("norecord", {}, {"job_block": rec()})
        assert "a job holds no reference record, so nothing says where they drew alike: 100-105" in run(["--new", str(new), "--read", str(old)])[1]
        new, old = pair("iris", {"job_block": rec()}, {"job_block": rec(iris="filled")})
        assert "the references' records say they drew it differently: 100-105" in run(["--new", str(new), "--read", str(old)])[1]
        # nothing owed: the records alike, no words, no old reference: refused, asking for the pixels
        new, old = pair("nothing", {"job_block": rec()}, {"job_block": rec()})
        try:
            run(["--new", str(new), "--read", str(old), "--by", "the lead", "--note-file", str(words), "--reference-checked", str(check), "--checked-frames", "102"])
            raise AssertionError("one checked frame passed with nothing owed")
        except SystemExit as e:
            # THE CASE THAT GOES RED when a changed reference with nothing said of where passes on one checked frame
            assert "nothing says on which frames" in str(e), str(e)
        # an old reference at another size: every frame owed, not dropped
        new, old = pair("size", {"job_block": rec()}, {"job_block": rec()})
        (d / "size" / "aside").mkdir()
        small = mp4(d / "size" / "aside" / "a_old_reference.mp4", [np.zeros((12, 16, 3), np.uint8)] * 6)
        mp4(Path(json.loads((Path(str(new) + "_job") / "job.plan.json").read_text())["loads"][0]["motion_file"]), [np.zeros((24, 32, 3), np.uint8)] * 6)
        at = datetime.datetime.fromisoformat("2026-01-01T09:00:00").timestamp()
        os.utime(small, (at, at))
        got = run(["--new", str(new), "--read", str(old), "--old-reference", str(small)])[1]
        assert "their pixels differ: 100-105" in got, got
        # a kept digest: an old reference with other bytes is refused; a new strip drawn with another digest refuses the look
        keep = lambda digest, f: {"by": "x", "at": "2026-01-01T12:00:00", "note": "", "shown": [{"load": "a_old", "made": "2026-01-01T11:00:00",  # noqa: E731
                                  "motion": [{"file": f, "sha256": digest}]}]}
        (d / "kept").mkdir()
        f = str(d / "kept" / "a_old_reference.mp4")
        old = capture(d / "kept", "a_old", 100, t, p, r, look=keep(hashlib.sha256(b"what was seen").hexdigest(), f), job_block=rec())
        (d / "kept" / "n").mkdir()
        new = capture(d / "kept" / "n", "a_new", 100, t, p, r, reference=b"new", ref_written="2026-01-02T09:00:00", strip_made="2026-01-02T11:00:00",
                      job_block=rec())
        (d / "kept" / "aside").mkdir()
        other = d / "kept" / "aside" / "a_old_reference.mp4"
        other.write_bytes(b"not what was seen")
        try:
            run(["--new", str(new), "--read", str(old), "--old-reference", str(other)])
            raise AssertionError("an old reference with another digest was taken")
        except SystemExit as e:
            assert "is not the file the earlier strip was drawn with" in str(e), str(e)
        shown = new / "looks" / "a_new.shown.json"
        shown.write_text(json.dumps({**json.loads(shown.read_text()), "motion": [{"file": str(d / "kept" / "n" / "a_new_reference.mp4"),
                                                                                    "sha256": hashlib.sha256(b"drawn with").hexdigest()}]}))
        try:
            run(["--new", str(new), "--read", str(old), "--by", "the lead", "--note-file", str(words)])
            raise AssertionError("a look rested on a strip drawn with another digest")
        except SystemExit as e:
            assert "the new capture's strip was drawn with another file" in str(e), str(e)
        # the new job's reference record older than its file
        new, old = pair("record", {"job_block": rec()}, {"job_block": {**rec(), "reference": {**rec()["reference"], "made": "2026-01-01T00:00:00Z"}}})
        try:
            run(["--new", str(new), "--read", str(old), "--by", "the lead", "--note-file", str(words)])
            raise AssertionError("a look rested on a reference record older than its file")
        except SystemExit as e:
            assert "reference record is older than" in str(e), str(e)
        # the builder's rule for the motion video: a whole-subject load of a subject with a pose is the mesh, against one that says none
        new, old = pair("mesh", {"job_block": {"subjects": [{"label": "a"}]}, "load": lambda ref: {"name": "a_old"}},
                        {"job_block": {"subjects": [{"label": "a"}]}, "load": lambda ref: {"name": "a_new", "motion": "none"}})
        assert "the earlier load shows the model 'mesh' as its motion and the new one 'none'" in run(["--new", str(new), "--read", str(old)])[1]
        # a blind reading alone: the strip's bound is the capture's own time, so a reference remade between it and a redrawn strip is CHANGED
        (d / "blind").mkdir()
        old = capture(d / "blind", "a_old", 100, t, p, r, blind=True, written="2026-01-01T10:00:00", strip_made="2026-01-05T11:00:00",
                      ref_written="2026-01-01T10:30:00", reference=b"same", job_block=rec())
        (d / "blind" / "n").mkdir()
        new = capture(d / "blind" / "n", "a_new", 100, t, p, r, reference=b"same", ref_path=json.loads(
            (Path(str(old) + "_job") / "job.plan.json").read_text())["loads"][0]["motion_file"], ref_written="2026-01-01T10:30:00",
            strip_made="2026-01-06T11:00:00", job_block=rec())
        # THE CASE THAT GOES RED when a strip drawn again after a blind reading passes a reference remade before it
        assert "the reference: CHANGED" in run(["--new", str(new), "--read", str(old)])[1]
    return ("a job with no reference record, or an eye drawn another way, owes every frame; nothing owed asks for the pixels; an old reference "
            "at another size owes every frame; a kept digest refuses an old reference with other bytes and a new strip drawn with another; a "
            "record older than its file refuses; the builder's mesh-or-none rule compared; a blind reading's strip bounded by the capture's own time")


case("equal and subset carry", equal_and_subset_carry)
case("larger does not carry", larger_does_not_carry)
case("a look only when it carries", a_look_only_when_it_carries)
case("a changed reference needs its check", a_changed_reference_needs_its_check)
case("what the check is owed on, and what shows the model", what_the_check_is_owed_on)
case("the gaps a cold reader named", the_reader_named_gaps)
sys.exit(finish())
