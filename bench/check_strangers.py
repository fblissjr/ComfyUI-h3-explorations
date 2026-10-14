#!/usr/bin/env python3
"""The strangers keep-out (`bench/strangers.py`) on a made preview: what it takes out, where it writes, what it refuses.

A preview of two subjects is made in a temporary folder: a source wider than the canvas (the loader crops it), white
blocks on black, the preview beginning two frames into it, and the two subjects' tracks moving a frame at a time. The
part model is handed in as "every white pixel is a person", so what is checked is the tool's arithmetic, its clocks
and its files, not the model.

1. **A subject is not a stranger, nor is the rim round them, by a disc.** A white pixel within
   `h3_config.MASKED_STRANGERS["grow_px"]` of a track (straight out at exactly the grow, or diagonally inside the
   disc) is not in the file; one a pixel further straight out, or diagonally outside the disc and inside its square,
   is. Both subjects are taken out.
2. **The clocks and the fit.** The source is read from the preview's first frame and cropped to the canvas as the
   lane's loader crops it; each track's frame 0 is the preview's first frame, and a track that moves is taken out
   where it is on each frame.
3. **A lost track is held**: a frame where a subject's track is empty takes its last frame with one, and the record
   lists the frame.
4. **Where it writes** is the path the job step looks for, with a record the job step holds it to
   (`held_to_its_record`): another preview's record, another grow or other tracks are named.
5. **It writes over nothing**: a file that is there, two tracks for a subject, a track not of the canvas's size, a
   source shorter than the preview and an output key that names no video are refused, with nothing written."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, needs  # noqa: E402

needs("ffmpeg on the path", bool(shutil.which("ffmpeg")))

from _lib.frames import read_mask, write_mask_video  # noqa: E402

import strangers as st  # noqa: E402

W, H, N, FIRST, SRC_W, LEAD = 64, 48, 4, 2, 80, 2          # the preview: 4 frames from source frame 2; the source is 80 wide, cropped to 64
GROW = st.h3_config.MASKED_STRANGERS["grow_px"]
import capture_masked_run as _cap  # noqa: E402
_NAMES = _cap._pack("sapiens2_parts").CLASS_NAMES
FACE, BODY = _NAMES.index(st.FACE_CLASS), _NAMES.index("Torso")
#: The part model, faked: white is a person's body, mid grey a face (`preview(faces=...)`), black nobody.
MODEL = lambda pic: np.where(pic[..., 0] > 200, BODY, np.where(pic[..., 0] > 90, FACE, 0)).astype(np.uint8)  # noqa: E731


def refused(fn, *words: str) -> None:
    try:
        fn()
    except st.Refused as e:
        assert all(w in str(e) for w in words), f"refused, but not with {words}: {e}"
        return
    raise AssertionError(f"not refused: {words}")


def picture():
    """On the canvas, a frame of the preview each: the people, subject a's track, subject b's track."""
    people, a, b = (np.zeros((N, H, W), bool) for _ in range(3))
    for i in range(N):
        a[i, 10:30, 4 + 2 * i:14 + 2 * i] = True           # a moves right two pixels a frame
    b[:, 10:24, 40:46] = True
    people |= a | b
    for i in range(N):
        e = 14 + 2 * i                                      # one past a's right edge on this frame
        people[i, 12, e - 1 + GROW] = True                  # straight out at exactly the grow: hers
        people[i, 14, e + GROW] = True                      # a pixel further: a stranger
        people[i, 30 - 1 + 6, e - 1 + 6] = True             # diagonally 6,6 off her lower right corner: outside the disc, inside its square
        people[i, 30 - 1 + 5, e - 1 + 5] = True             # diagonally 5,5: inside the disc
    people[:, 40:46, 30:40] = True                          # a stranger well off both
    return people, a, b


def preview(root: Path, people: np.ndarray, tracks: dict, frames: int = N, listed_twice: bool = False, own: dict | None = None,
            faces: np.ndarray | None = None) -> Path:
    out = root / "output" / "Video/v/s/pv"
    out.mkdir(parents=True)
    src = root / "source.mkv"
    # source frames 0..LEAD-1 are all white (a reader on the wrong clock sees everybody); then the preview's frames,
    # each set 8 px into a frame 80 wide, the band outside the crop white
    canvas = np.concatenate([np.ones((LEAD, H, W), bool), people[:frames]])
    wide = np.ones((len(canvas), H, SRC_W), bool)
    wide[:, :, 8:8 + W] = canvas
    level = wide.astype(np.uint8) * 255
    if faces is not None:                                   # a face, as the part model would call it: mid grey
        f = np.zeros_like(wide)
        f[LEAD:, :, 8:8 + W] = faces[:frames]
        level[f] = 128
    rgb = np.repeat(level[..., None], 3, -1)
    enc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SRC_W}x{H}", "-r", "24", "-i", "-",
                          "-c:v", "ffv1", "-pix_fmt", "bgr0", str(src)], input=rgb.tobytes())
    assert enc.returncode == 0
    files = []
    for label, m in tracks.items():
        write_mask_video(out / f"pv_{label}_track_00001.mkv", m)
        files.append(f"Video/v/s/pv/pv_{label}_track_00001.mkv")
    for name, m in (own or {}).items():                    # e.g. "a_held_things": a subject's own file beside the track
        write_mask_video(out / f"pv_{name}_00001.mkv", m)
        files.append(f"Video/v/s/pv/pv_{name}_00001.mkv")
    if listed_twice:
        files.append("Video/v/s/pv/pv_a_track_00002.mkv")
    folder = root / "data" / "2026-01-01_pv_job"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"job": {"name": "pv", "first": FIRST, "frames": N, "source": str(src), "canvas": [W, H], "output": "Video/v/s",
                                                         "subjects": [{"label": k} for k in tracks]}}))
    (folder / "preview.queued.json").write_text(json.dumps({"status": "success", "files": files}))
    return folder


def a_subject_and_their_rim_are_not_strangers() -> str:
    people, a, b = picture()
    got = st.strangers_of(people, [a, b], GROW)
    for i in range(N):
        e = 14 + 2 * i
        assert not got[i, 12, e - 1 + GROW] and got[i, 14, e + GROW], "the grow is not exactly its px straight out"
        assert got[i, 35, e + 5] and not got[i, 34, e + 4], "the grow is not a disc"
    assert got[:, 40:46, 30:40].all() and not got[:, 10:24, 40:46].any(), "a stranger off the grow was taken out, or b was not"
    assert st.strangers_of(people, [a], GROW)[:, 10:24, 40:46].all(), "with one track taken out, the other subject should be in the file"
    return f"a pixel {GROW} px straight out of a track is taken out and one at {GROW + 1} is not; 5,5 diagonally is taken out and 6,6 is not (a disc); both subjects are"


def where_it_writes() -> str:
    people, a, b = picture()
    a_lost = a.copy()
    a_lost[2] = False                                       # the tracker lost her on the preview's third frame
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = preview(root, people, {"a": a_lost, "b": b})
        said = []
        rec = st.run(folder, root / "output", "the session", MODEL, "a fake model", say=said.append)
        path = root / "output" / "Video/v/keep_outs/pv/pv_strangers.mkv"
        import shot_to_load as stl
        assert path == root / "output" / stl.KEEP_OUTS_AT.format(video="v", preview="pv") / "pv_strangers.mkv" and Path(rec["file"]) == path, rec["file"]
        back = read_mask(str(path), (W, H), 0, 0, N)[0]
        held = a_lost.copy()
        held[2] = a_lost[1]                                 # her lost frame takes the frame before it
        # THE CASE THAT GOES RED on a wrong clock, a source not cropped as the loader crops it, or a lost track left empty
        want = st.handed_back(st.strangers_of(people, [held, b], GROW), {"a": st.grown(held, GROW), "b": st.grown(b, GROW)}, np.zeros_like(people), FIRST)[0]
        assert (back == want).all(), "the file is not the strangers of the preview's own frames, cropped, with her lost frame held"
        assert not back[2, 10:30, 10:14].any(), "the subject is a stranger on the frame her track was lost"
        w = json.loads(path.with_suffix(".json").read_text())
        assert w["grow_px"] == GROW and w["first_source_frame"] == FIRST and w["frames_a_track_is_empty_and_held"] == {"a": [FIRST + 2]} and w["source_size"] == [SRC_W, H], w
        assert any("does not find" in x for x in w["cannot_see"]) and said and f"strangers on {N} of them" in said[0], said
        tracks = {"a": str(root / "output/Video/v/s/pv/pv_a_track_00001.mkv"), "b": str(root / "output/Video/v/s/pv/pv_b_track_00001.mkv")}
        assert st.held_to_its_record(path, "pv", FIRST, N, tracks) == [] and w["frame_hash"] and w["hold_frames"] == st.h3_config.MASKED_STRANGERS["hold_frames"]
        assert any("record's frames" in x for x in st.held_to_its_record(path, "pv", FIRST, N + 1, tracks))
        moved = path.with_name("moved.mkv")
        shutil.copy(path, moved)
        shutil.copy(path.with_suffix(".json"), moved.with_suffix(".json"))
        assert any("record's file" in x for x in st.held_to_its_record(moved, "pv", FIRST, N, tracks))
        write_mask_video(path.with_name("changed.mkv"), ~back)
        shutil.copy(path.with_name("changed.mkv"), path)
        assert any("not the ones its record hashed" in x for x in st.held_to_its_record(path, "pv", FIRST, N, tracks)), "a changed file was taken"
        assert any("preview" in x for x in st.held_to_its_record(path, "pv2", FIRST, N, tracks))
        assert any("first_source_frame" in x for x in st.held_to_its_record(path, "pv", FIRST + 1, N, tracks))
        assert any("subjects_taken_out" in x for x in st.held_to_its_record(path, "pv", FIRST, N, {"a": tracks["a"]}))
        refused(lambda: st.run(folder, root / "output", "the session", MODEL, "m"), "is there", "never replaced")
        path.with_suffix(".json").unlink()
        assert any("is not there beside it" in x for x in st.held_to_its_record(path, "pv", FIRST, N, tracks))
    return "the file is the strangers of the preview's frames read on its own clock and cropped as the loader crops, her lost frame held; its record holds it to its preview"


def it_writes_over_nothing() -> str:
    people, a, b = picture()
    nothing = lambda root: not (root / "output/Video/v/keep_outs").exists()   # noqa: E731
    for kw, tracks, words in (({"listed_twice": True}, {"a": a, "b": b}, ("lists 2 track videos for 'a'",)),
                              ({"frames": N - 1}, {"a": a, "b": b}, ("gave", "nothing written")),
                              ({}, {"a": a[:, :24, :32], "b": b}, ("is 32x24", "nothing written")),
                              ({}, {"a": a[:N - 1], "b": b}, (f"with {N - 1} frames", "nothing written")),
                              ({}, {"a": np.zeros_like(a), "b": b}, ("empty on every frame", "Nothing written"))):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refused(lambda: st.run(preview(root, people, tracks, **kw), root / "output", "s", MODEL, "m"), *words)
            assert nothing(root), f"a file was written on a refused run: {words}"
    refused(lambda: st.file_of({"name": "pv", "output": "elsewhere"}, Path("/o")), "not Video/<video>/<session>")
    return ("two tracks for one subject, a source shorter than the preview, a track not of the canvas's size or shorter than the preview, a track empty on "
            "every frame and an output key with no video are refused, and nothing is written")


def nothing_of_a_subjects_own_is_a_stranger() -> str:
    """The lead's rule of 2026-10-14, on mrcircle's 1156b: her track leaves out her black hat as a held thing, and the
    strangers file took the hat (1164) and slivers at her mitts (1186) as somebody else's. Her held things and her parts, as
    her preview saves them, are taken out with her track."""
    people, a, b = picture()
    hat = np.zeros_like(a)
    hat[:, 0:1, 0:6] = True                                 # above her track, past the grow and not touching it: her hat as a held thing
    mitt = np.zeros_like(a)
    mitt[:, 39:42, 0:3] = True                              # below her track's corner, past the grow: a sliver of her part outside it
    people = people | hat | mitt
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bare = st.run(preview(root, people, {"a": a, "b": b}), root / "output", "s", MODEL, "m", say=lambda *_: None)
        before = read_mask(bare["file"], (W, H), 0, 0, N)[0]
    assert before[:, 0, 2].all() and before[:, 40, 1].all(), "the fixture's hat or mitt is not a stranger without her own files: the case reads nothing"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec = st.run(preview(root, people, {"a": a, "b": b}, own={"a_held_things": hat, "a_parts": mitt}), root / "output", "s", MODEL, "m", say=lambda *_: None)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
        # THE CASE THAT GOES RED when a subject's held things or parts are not taken out
        assert not (back & (hat | mitt)).any(), "her hat or her part outside her track is a stranger"
        assert back[:, 40:46, 30:40].all(), "the stranger off everybody's own was taken out"
        assert rec["taken_out"] == st.OWN_FORM and [Path(f).name for f in rec["subjects_own_taken_out"]["a"]] == ["pv_a_held_things_00001.mkv", "pv_a_parts_00001.mkv"], rec
        tracks = {k: str(root / f"output/Video/v/s/pv/pv_{k}_track_00001.mkv") for k in ("a", "b")}
        assert st.held_to_its_record(Path(rec["file"]), "pv", FIRST, N, tracks) == []
        old = json.loads(Path(rec["file"]).with_suffix(".json").read_text())
        old.pop("taken_out")
        Path(rec["file"]).with_suffix(".json").write_text(json.dumps(old))
        assert any("made before 2026-10-14's mend" in x for x in st.held_to_its_record(Path(rec["file"]), "pv", FIRST, N, tracks)), "a file of tracks only was taken"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        refused(lambda: st.run(preview(root, people, {"a": a, "b": b}, own={"a_held_things": hat, "a_held_things_x": hat}), root / "output", "s", MODEL, "m"),
                "lists 2 held_things videos for 'a'")
    return "her held things and her part outside her track are not strangers, the record names them, and a file made of tracks only is made again"


def a_piece_touching_her_with_no_face_is_hers() -> str:
    """The lead's ruling (A) of 2026-10-14, on 1156b: the tracker left out her hat lifted in her hand (1164) and her mitt's
    fingertips, past the grow, and her held-things file held none of them. A piece of what is left that touches her own,
    grown, and holds no face is hers, and is listed; a piece that holds a face stays a stranger however close (3167: the
    other woman's chin and face against the lead's)."""
    people, a, b = picture()
    hat = np.zeros_like(a)
    for i in range(N):
        hat[i, 0:10 - GROW, 6 + 2 * i:16 + 2 * i] = True     # above her track, reaching down to just past her grow: lifted in her hand
    tips = np.zeros_like(a)
    for i in range(N):
        e = 14 + 2 * i
        tips[i, 20:22, e + GROW:e + GROW + 6] = True        # straight out of her right edge, past the grow and touching it
    other = np.zeros_like(a)
    other[:, 30 + GROW:30 + GROW + 6, 4:12] = True          # under her, touching her grow: another woman's chin and face
    face = other.copy()
    people = people | hat | tips | other
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        said: list[str] = []
        rec = st.run(preview(root, people, {"a": a, "b": b}, faces=face), root / "output", "s", MODEL, "m", say=said.append)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
    # THE CASE THAT GOES RED when a faceless piece touching her stays a stranger, or a face touching her is handed back
    assert not (back & (hat | tips)).any(), "her lifted hat or her fingertips past the grow are strangers"
    assert back[:, 30 + GROW + 1:30 + GROW + 6, 5:11].all(), "a face touching her was handed back to her"
    assert back[:, 40:46, 30:40].all(), "a stranger touching nobody was handed back"
    got = rec["handed_back"]
    # four a frame: the hat, the fingertips, and the fixture's two lone pixels just past her grow (picture())
    assert {g["subject"] for g in got} == {"a"} and sorted({g["frame"] for g in got}) == [FIRST + i for i in range(N)] and len(got) == 4 * N, got
    assert f"a: {4 * N} piece(s) handed back on {N} frame(s)" in rec["handed_back_said"] and rec["handed_back_said"] in said[0], (rec["handed_back_said"], said)
    assert any("arm or hand reaching across" in x for x in rec["cannot_see"])
    # handed_back alone: a face pixel anywhere in the piece keeps it
    s1 = np.zeros((1, 8, 8), bool)
    s1[0, 2:5, 2:5] = True
    own = np.zeros_like(s1)
    own[0, 2:5, 5:7] = True
    f1 = np.zeros_like(s1)
    f1[0, 4, 2] = True
    assert st.handed_back(s1, {"x": own}, f1, 0)[1] == [] and st.handed_back(s1, {"x": own}, np.zeros_like(s1), 0)[1][0]["px"] == 9
    return "a faceless piece touching her own, grown (a hat lifted, fingertips), is hers and listed; a piece with a face stays a stranger however close"


def a_lost_track_is_held_a_little() -> str:
    """A blind reader (2026-10-14): a subject who left was held at their last place to the end of the shot, keeping a
    newcomer to that spot out of the strangers file; and the start of a preview was held to the first later frame
    with no test of it."""
    reach = int(st.h3_config.MASKED_STRANGERS["hold_frames"])
    n = 2 * reach + 6
    t = np.zeros((n, 4, 4), bool)
    t[2:4, 1, 1] = True                                     # on frames 2 and 3, then gone
    out, held, empty = st.held_track(t)
    assert held == [0, 1] + list(range(4, 4 + reach)) and empty == list(range(4 + reach, n)), (held, empty)
    assert out[0, 1, 1] and out[3 + reach, 1, 1] and not out[4 + reach].any(), "a frame within reach was not held, or one beyond it was"
    late = np.zeros((n, 4, 4), bool)
    late[reach + 1:, 2, 2] = True                           # on from frame reach+1: frame 0 is reach+1 away, frame 1 is reach
    _, held, empty = st.held_track(late)
    assert empty == [0] and held == list(range(1, reach + 1)), (held, empty)
    return f"an empty frame is held to the nearest frame with one when it is at most {reach} away, before or (at the start) after; beyond, left empty and listed"


case("a subject, and their rim within the grow, is not a stranger", a_subject_and_their_rim_are_not_strangers)
case("where it writes, its clocks, and its record", where_it_writes)
case("it writes over nothing", it_writes_over_nothing)
case("a lost track is held a little, and no further", a_lost_track_is_held_a_little)
case("nothing of a subject's own is a stranger", nothing_of_a_subjects_own_is_a_stranger)
case("a piece touching her with no face is hers; a face stays a stranger", a_piece_touching_her_with_no_face_is_hers)
sys.exit(finish())
