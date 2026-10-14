#!/usr/bin/env python3
"""`bench/look_carry.py` on made captures: each case is a way a load made again could rest on a reading it no longer matches.

  equal_and_subset_carry       track, part and region equal, or the new one a subset, on every frame: it carries; a frame of the new
                               load inside the earlier one when it starts later
  larger_does_not_carry        a px added to the track or the part, or a cell to the region, on one frame: LARGER, with how much and
                               where, and it does not carry; a frame outside the earlier load does not carry
  a_look_only_when_it_carries  with --by: a look recorded on the new capture resting on the earlier reading (its look or its blind
                               reading), with the person's words and the figures; refused when it does not carry, with no reading to
                               rest on, without words, or by the capture's maker; rules new and gone said

    <comfy venv python> bench/check_look_carry.py

No model, no card, no ffmpeg.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import look_carry as lc  # noqa: E402

N, H, W = 6, 48, 64


def capture(root: Path, name: str, first: int, track: np.ndarray, parts: np.ndarray, region: np.ndarray, flags=(), maker="the card",
            look=None, blind=False, written="2026-01-01T10:00:00", strip_made="2026-01-01T11:00:00") -> Path:
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
        assert code == 0 and "CARRIES" in said and "rules gone: [('part_in_pieces', '', '')]" in said and "rules new: [('text_names_a_voice_where_there_is_none', '', '')]" in said, said
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
            and "equal or a subset on all 6 frames" in looks[0]["note"], looks[0]["note"]
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


case("equal and subset carry", equal_and_subset_carry)
case("larger does not carry", larger_does_not_carry)
case("a look only when it carries", a_look_only_when_it_carries)
sys.exit(finish())
