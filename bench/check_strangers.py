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
4. **Where it writes** is the path the job step looks for, with the whole frame's classes beside it on the preview's frames
   (`whole_frame_classes`, read by the capture's her-hair-in-hole pass) and a record the job step holds it to
   (`held_to_its_record`): another preview's record, another grow or other tracks are named.
5. **It writes over nothing**: a file that is there, two tracks for a subject, a track not of the canvas's size, a
   source shorter than the preview and an output key that names no video are refused, with nothing written.
6. **Only another person is a stranger** (the owner, 2026-10-23): a piece touching a subject with a core of its own is
   handed back where it lies on no other person the detector found, and stays a stranger with one pixel on one; a
   detection is a subject's when more than `h3_config.MASKED_STRANGERS["subject_share"]` of it lies on her track, and
   another person's is held `hold_frames` either side, as a track is.
   One file serves every subject's load, so another subject's own detection is another person.
7. **The people pass** writes the detector's people of the preview's own frames, and `run` makes no file without them;
   the detector is a stand-in here, as the part model is, and how the preview's tracker asked is read off its graph.
8. **The card freed first**: from the command line each pass asks the server to free the card (the request stood in for)
   before its model, and `run`'s people child exits before the part model loads."""
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
#: The part model, faked: white is Apparel (a person the core test cannot tell: no core), mid grey a face, dark grey an arm
#: (`preview(levels=...)`), black nobody.
FACE, ARM, NOT_CORE, HAIR = _NAMES.index("Face_Neck"), _NAMES.index("Left_Lower_Arm"), _NAMES.index("Apparel"), _NAMES.index("Hair")
MODEL = lambda pic: np.select([pic[..., 0] > 200, pic[..., 0] > 90, pic[..., 0] > 40, pic[..., 0] > 15], [NOT_CORE, FACE, ARM, HAIR], 0).astype(np.uint8)  # noqa: E731


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


#: How the fixture's tracker asked for people, as `strangers.tracker_asks` reads it off a graph.
ASKS = {"checkpoint": "sam.safetensors", "corrections": {"correct_image_range": True}, "phrase": "person", "threshold": 0.5, "max_people": 16}


def detector(found: list, seen: list | None = None):
    """A stand-in for SAM 3.1: frame by frame, the masks `found` holds for it (each a [N, H, W] person, in their numbering), every
    picture it is handed kept in `seen`."""
    calls = iter(range(10 ** 6))

    def detect(pic):
        i = next(calls)
        if seen is not None:
            seen.append(pic.copy())
        got = [m[i] for m in found if m[i].any()]
        return (np.stack(got) if got else np.zeros((0, H, W), bool)), [0.9] * len(got)
    return detect


def encode(src: Path, people: np.ndarray, frames: int, levels: dict | None) -> None:
    # source frames 0..LEAD-1 are all white (a reader on the wrong clock sees everybody); then the preview's frames,
    # each set 8 px into a frame 80 wide, the band outside the crop white
    canvas = np.concatenate([np.ones((LEAD, H, W), bool), people[:frames]])
    wide = np.ones((len(canvas), H, SRC_W), bool)
    wide[:, :, 8:8 + W] = canvas
    level = wide.astype(np.uint8) * 255
    for v, m in (levels or {}).items():                     # a class the part model would call it: 128 a face, 64 an arm (MODEL)
        f = np.zeros_like(wide)
        f[LEAD:, :, 8:8 + W] = m[:frames]
        level[f] = v
    rgb = np.repeat(level[..., None], 3, -1)
    enc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SRC_W}x{H}", "-r", "24", "-i", "-",
                          "-c:v", "ffv1", "-pix_fmt", "bgr0", str(src)], input=rgb.tobytes())
    assert enc.returncode == 0


def preview(root: Path, people: np.ndarray, tracks: dict, frames: int = N, listed_twice: bool = False, own: dict | None = None,
            levels: dict | None = None, found: list | None = None, seen: list | None = None) -> Path:
    """A preview of `tracks` on a source of `people`, and its people file made by the tool's own pass from a stand-in detector that
    finds `found` (each a person, [N, H, W]; by default the subjects' tracks alone, so a piece touching a subject lies on nobody else).
    `frames` short of the preview cuts the source after the people pass."""
    out = root / "output" / "Video/v/s/pv"
    out.mkdir(parents=True)
    src = root / "source.mkv"
    encode(src, people, N, levels)
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
    st.people_pass(folder, root / "output", "the session", detector(list(tracks.values()) if found is None else found, seen), ASKS, say=lambda *_: None)
    if frames < N:
        encode(src, people, frames, levels)
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
        # THE CASE THAT GOES RED when the whole frame's classes are not kept beside the file, or not on the preview's own frames
        kept = np.load(w["whole_frame_classes"])
        assert Path(w["whole_frame_classes"]) == path.with_name("pv_strangers_classes.npz") and int(kept["first"]) == FIRST, w["whole_frame_classes"]
        assert kept["classes"].shape == (N, H, W) and ((kept["classes"] > 0) == people).all(), "the kept classes are not the model's on the preview's frames"
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
    # the people file is the people pass's, beside where the strangers file would go: nothing of the strangers run's is there
    nothing = lambda root: not [f for f in (root / "output/Video/v/keep_outs/pv").glob("*") if "_people" not in f.name]   # noqa: E731
    for kw, tracks, words in (({"listed_twice": True}, {"a": a, "b": b}, ("lists 2 track videos for 'a'",)),
                              ({"frames": N - 1}, {"a": a, "b": b}, ("gave", "nothing written")),
                              ({}, {"a": a[:, :24, :32], "b": b}, ("is 32x24", "nothing written")),
                              ({}, {"a": a[:N - 1], "b": b}, (f"with {N - 1} frames", "nothing written")),
                              ({}, {"a": np.zeros_like(a), "b": b}, ("empty on every frame", "Nothing written"))):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refused(lambda: st.run(preview(root, people, tracks, found=[b], **kw), root / "output", "s", MODEL, "m"), *words)
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
        assert any("made before this form" in x for x in st.held_to_its_record(Path(rec["file"]), "pv", FIRST, N, tracks)), "a file of tracks only was taken"
        old["taken_out"] = st.OWN_FORM.split("; and")[0] + "; and a piece touching them with no body core of its own, or whose only core is hair meeting theirs"
        Path(rec["file"]).with_suffix(".json").write_text(json.dumps(old))
        # THE CASE THAT GOES RED when a file made before 2026-10-23's rule (no detector's people) is taken
        assert any("made before this form" in x for x in st.held_to_its_record(Path(rec["file"]), "pv", FIRST, N, tracks)), "a file of the 10-14 form was taken"
        old.update(taken_out=st.OWN_FORM, subject_share=0.25)
        Path(rec["file"]).with_suffix(".json").write_text(json.dumps(old))
        assert any("record's subject_share" in x for x in st.held_to_its_record(Path(rec["file"]), "pv", FIRST, N, tracks)), "another subject_share was taken"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        refused(lambda: st.run(preview(root, people, {"a": a, "b": b}, own={"a_held_things": hat, "a_held_things_x": hat}), root / "output", "s", MODEL, "m"),
                "lists 2 held_things videos for 'a'")
    return "her held things and her part outside her track are not strangers, the record names them, and a file made of tracks only is made again"


def a_piece_touching_her_with_no_core_is_hers() -> str:
    """The lead's ruling (A) of 2026-10-14, on 1156b: the tracker left out her hat lifted in her hand (1164) and her mitt's
    fingertips, past the grow, and her held-things file held none of them. A piece of what is left that touches her own,
    grown, and holds no body core (`CORE_CLASSES`) is hers, and is listed; a piece that holds a face stays a stranger
    however close (3167: the other woman's chin and face against the lead's), and so does an arm reaching across her
    (the lead's refinement the same day, on an outside reviewer's point)."""
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
    arm = np.zeros_like(a)
    for i in range(N):
        e = 14 + 2 * i
        arm[i, 25:27, e + GROW:e + GROW + 8] = True         # out of her right edge past the grow, touching it: a crowd's arm across her
    people = people | hat | tips | other | arm
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        said: list[str] = []
        # the other woman and the crowd's arm are people the detector found: on them, a face or an arm stays a stranger
        rec = st.run(preview(root, people, {"a": a, "b": b}, levels={128: face, 64: arm}, found=[a, b, other, arm]), root / "output", "s", MODEL, "m",
                     say=said.append)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
    # THE CASE THAT GOES RED when a piece with no core touching her stays a stranger, or a face or an arm touching her is handed back
    assert not (back & (hat | tips)).any(), "her lifted hat or her fingertips past the grow are strangers"
    assert back[:, 30 + GROW + 1:30 + GROW + 6, 5:11].all(), "a face touching her was handed back to her"
    loose = arm & ~st.grown(a, GROW) & ~st.grown(b, GROW)
    assert loose.any() and (back & loose).sum() == loose.sum(), "an arm touching her was handed back to her"
    assert back[:, 40:46, 30:40].all(), "a stranger touching nobody was handed back"
    got = rec["handed_back"]
    # four a frame: the hat, the fingertips, and the fixture's two lone pixels just past her grow (picture())
    assert {g["subject"] for g in got} == {"a"} and sorted({g["frame"] for g in got}) == [FIRST + i for i in range(N)] and len(got) == 4 * N, got
    assert f"a: {4 * N} piece(s) handed back on {N} frame(s)" in rec["handed_back_said"] and rec["handed_back_said"] in said[0], (rec["handed_back_said"], said)
    assert any("hand, hat or prop alone" in x for x in rec["cannot_see"]) and rec["core_classes"] == list(st.CORE_CLASSES)
    assert "Apparel" not in st.CORE_CLASSES and not any("Hand" in c for c in st.CORE_CLASSES) and all(c in _NAMES for c in st.CORE_CLASSES)
    # handed_back alone: a core pixel anywhere in the piece keeps it
    s1 = np.zeros((1, 8, 8), bool)
    s1[0, 2:5, 2:5] = True
    own = np.zeros_like(s1)
    own[0, 2:5, 5:7] = True
    f1 = np.zeros_like(s1)
    f1[0, 4, 2] = True
    assert st.handed_back(s1, {"x": own}, f1, 0)[1] == [] and st.handed_back(s1, {"x": own}, np.zeros_like(s1), 0)[1][0]["px"] == 9
    return ("a piece touching her own, grown, with no body core (a hat lifted, fingertips) is hers and listed; a piece with a face or an arm stays a "
            "stranger however close")


def her_own_hair_past_her_track() -> str:
    """The lead's ruling of 2026-10-14 on 1097c and 1156c: her flying hair just outside her track was a stranger, since Hair is
    a body core. A piece whose only core is Hair, meeting the Hair inside her grown own, is hers; one that meets her where she
    has no hair, or holds another core, stays a stranger, and the record says what kept it."""
    people, a, b = picture()
    her_hair = np.zeros_like(a)
    for i in range(N):
        her_hair[i, 2:14, 6 + 2 * i:12 + 2 * i] = True      # her hair: the top rows of her track and on up through her grow, as hair runs
    flying = np.zeros_like(a)
    for i in range(N):
        flying[i, 0:2, 6 + 2 * i:12 + 2 * i] = True         # above her, past the grow, touching it: her hair flying
    other = np.zeros_like(a)
    other[:, 29 + GROW:29 + GROW + 3, 4:9] = True           # under her, touching her grow where she has no hair: someone's hair
    neck = np.zeros_like(a)
    neck[:, 24 + GROW:24 + GROW + 2, 40:46] = True          # under b, touching b's grow: hair with a face in it
    neck_face = np.zeros_like(a)
    neck_face[:, 24 + GROW + 1, 40:46] = True
    people = people | her_hair | flying | other | neck
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # someone's hair under her and the hair with a face under b are people the detector found
        rec = st.run(preview(root, people, {"a": a, "b": b}, levels={30: her_hair | flying | other | (neck & ~neck_face), 128: neck_face},
                             found=[a, b, other, neck]), root / "output", "s", MODEL, "m", say=lambda *_: None)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
    # THE CASE THAT GOES RED when her own hair past her track is a stranger, or a stranger's hair meeting her where she has none is hers
    assert not (back & flying).any(), "her own hair past her track is a stranger"
    assert (back & other & ~st.grown(a, GROW)).sum() == (other & ~st.grown(a, GROW)).sum() and (other & ~st.grown(a, GROW)).any(), "hair touching her where she has none was handed back"
    assert (back & neck & ~st.grown(b, GROW)).any(), "hair with a face in it was handed back"
    by_hair = [g for g in rec["handed_back"] if g.get("by") == "its hair meets hers"]
    assert {g["subject"] for g in by_hair} == {"a"} and len(by_hair) == N, rec["handed_back"]
    kept = rec["kept_though_touching"]
    assert any(k["touches"] == ["a"] and set(k["core"]) == {"Hair"} for k in kept) and any(k["touches"] == ["b"] and "Face_Neck" in k["core"] for k in kept), kept
    assert any("hair touching hers" in x for x in rec["cannot_see"]) and rec["hair_class"] == "Hair"
    return ("her hair past her track, meeting her own hair, is hers; hair touching her where she has none, and hair with a face, stay strangers, "
            "each with the classes that kept it in the record")


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


def her_own_clothing_on_no_other_person_is_hers() -> str:
    """The owner's rule (2026-10-14, restated 2026-10-23): what a subject wears is hers. On shot_c_0014e, where she is alone, her
    own clothing past her track was kept as a stranger because clothing is a body core (`kept_though_touching` in its record). A
    piece touching her with a core of its own is a stranger only where it lies on another person the detector found."""
    people, a, b = picture()
    sleeve, crowd, lone, far = (np.zeros_like(a) for _ in range(4))
    for i in range(N):
        e = 14 + 2 * i
        sleeve[i, 17:20, e + GROW:e + GROW + 3] = True      # her sleeve, out of her right edge past the grow, touching it (and not b)
        crowd[i, 25:27, e + GROW:e + GROW + 5] = True       # the same shape lower down: someone else's arm
    lone[:, 44:47, 56:62] = True                            # clothing touching nobody, on no detection
    people = people | sleeve | crowd | lone
    her = a | sleeve                                        # the detector's mask of her takes her sleeve in
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec = st.run(preview(root, people, {"a": a, "b": b}, levels={64: sleeve | crowd | lone}, found=[her, b, crowd]),
                     root / "output", "s", MODEL, "m", say=lambda *_: None)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
    out_of = lambda m: m & ~st.grown(a, GROW) & ~st.grown(b, GROW)   # noqa: E731
    # THE CASE THAT GOES RED when her own clothing on no other person stays a stranger, or another's on a detection is handed back
    assert out_of(sleeve).any() and not (back & sleeve).any(), "her own sleeve, on no other person, is a stranger"
    assert (back & out_of(crowd)).sum() == out_of(crowd).sum() and out_of(crowd).any(), "an arm lying on another person was handed back to her"
    assert (back & lone).sum() == lone.sum(), "a piece touching nobody was handed back"
    moved = [g for g in rec["handed_back"] if g["by"] == st.NO_OTHER]
    assert len(moved) == N and {g["subject"] for g in moved} == {"a"} and all(g["core"] == {"Left_Lower_Arm": g["px"]} for g in moved), moved
    said = rec["handed_back_on_no_other_person"]
    assert said["pieces"] == N and sorted(said["by_frame"]) == [str(FIRST + i) for i in range(N)], said
    # her detection takes her sleeve in: each piece says so, and the record counts them apart from pieces on no one's detection (a
    # detection holding her and a stranger would hand the stranger's back the same way: a look takes the largest)
    assert all(g["on_her_detection_px"] == g["px"] for g in moved), moved
    assert said["on_her_own_detection"]["pieces"] == N and said["on_no_detection"]["pieces"] == 0, said
    kept = [k for k in rec["kept_though_touching"] if "a" in k["touches"] and "Left_Lower_Arm" in k.get("core", {})]
    assert len(kept) == N and all(k["on_people"] == [3] and k["on_people_px"] == [k["px"]] for k in kept), kept
    assert rec["other_people_a_frame"] == [1] * N and rec["subject_share"] == st.h3_config.MASKED_STRANGERS["subject_share"], rec["other_people_a_frame"]
    assert any("the detector does not find" in x for x in rec["cannot_see"]) and st.NO_OTHER in rec["handed_back_said"], rec["handed_back_said"]
    # handed_back alone: ONE pixel of another person keeps a piece with a core; none hands it back
    s1, own, c1 = np.zeros((1, 8, 8), bool), np.zeros((1, 8, 8), bool), np.zeros((1, 8, 8), bool)
    s1[0, 2:5, 2:5], own[0, 2:5, 5:7], c1[0, 3, 3] = True, True, True
    dot = np.zeros((8, 8), bool)
    dot[4, 2] = True
    assert st.handed_back(s1, {"x": own}, c1, 0, others=[[(2, dot)]])[2][0]["on_people"] == [2], "a piece with one pixel on another person was handed back"
    assert st.handed_back(s1, {"x": own}, c1, 0, others=[[]])[1][0]["by"] == st.NO_OTHER, "a piece with a core on nobody else was kept"
    assert st.handed_back(s1, {"x": own}, c1, 0)[1] == [], "with no people given, a piece with a core was handed back"
    return ("her own clothing past her track, touching her and on no other person the detector found, is hers and listed by frame with its core; "
            "the same piece with a pixel on another person, and a piece touching nobody, stay strangers")


def a_detection_is_hers_by_her_track() -> str:
    """Which detection is a subject (`others_of`): one with more than `subject_share` of its pixels on her track is hers, so her own
    clothing on it is no other person's; one at the share or under is another person, and a frame where her track holds her and no
    detection is hers is listed (the detector missed her, or gave her one mask with someone else)."""
    share = float(st.h3_config.MASKED_STRANGERS["subject_share"])
    t = np.zeros((3, 4, 10), bool)
    t[:2, :, :5] = True                                     # her track: 20 px a frame, and gone on the third
    over, at = np.zeros((4, 10), bool), np.zeros((4, 10), bool)
    over[:, :6] = True                                      # 20 of 24 on her: hers
    at[:, 3:5] = True
    at[:, 5:7] = True                                       # 8 on her and 8 off: exactly half
    others, unseen = st.others_of([[(1, over)], [(1, at)], []], {"s": t}, share)
    # a frame her track does not hold her is not listed as one the detector missed her on
    assert share == 0.5 and [o[3] for o in others[0]] == ["s"] and [(o[0], o[3]) for o in others[1]] == [(1, None)] and others[2] == [], others
    assert unseen == {"s": [1]}, unseen
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        people, a, b = picture()
        lost = a.copy()
        lost[2] = False
        rec = st.run(preview(root, people, {"a": a, "b": b}, found=[lost, b]), root / "output", "s", MODEL, "m", say=lambda *_: None)
    assert rec["frames_a_subject_has_no_detection"] == {"a": [FIRST + 2]}, rec["frames_a_subject_has_no_detection"]
    return (f"a detection more than {share} on her track is hers and one at {share} is another person's; a frame where her track holds her and "
            "nobody's detection is hers is listed")


def a_person_dropped_for_a_moment_is_held() -> str:
    """`held_others`: a person the detector found within `hold_frames` either side of a frame is still another person there,
    as a track is held (on shot_c_4912g the CPU's SAM 3.1 dropped the dancer in blue passing the third subject on 4936-4937,
    blurred, and found her on 4935 and 4938). A piece on her only on another frame stays a stranger, listed with that frame;
    beyond the reach it is handed back."""
    reach = int(st.h3_config.MASKED_STRANGERS["hold_frames"])
    n = 2 * reach + 2
    s1, own = np.zeros((n, 8, 8), bool), np.zeros((n, 8, 8), bool)
    s1[:, 2:5, 2:5], own[:, 2:5, 5:7] = True, True
    found = np.zeros((8, 8), bool)
    found[1:6, 1:5] = True                                  # the dancer, found on frame 0 alone
    others = [[(2, found, 0, None)]] + [[] for _ in range(n - 1)]
    held = st.held_others(others)
    assert len(held) == n and [len(h) for h in held] == [1] * (reach + 1) + [0] * (n - reach - 1), [len(h) for h in held]
    _, given, kept = st.handed_back(s1, {"x": own}, s1, 0, others=held)
    # THE CASE THAT GOES RED when a dropped person is not held, or held past the reach
    assert [k["frame"] for k in kept] == list(range(reach + 1)) and [g["frame"] for g in given] == list(range(reach + 1, n)), (kept, given)
    assert kept[0]["on_people"] == [2] and "on_people_held" not in kept[0] and kept[1]["on_people"] == [] and kept[1]["on_people_held"] == [[0, 2, 9]], kept[:2]
    assert all(g["by"] == st.NO_OTHER for g in given)
    # and in a run: the crowd's arm is found on the preview's first frame only, and stays a stranger on every frame within the reach
    people, a, b = picture()
    crowd = np.zeros_like(a)
    for i in range(N):
        e = 14 + 2 * i
        crowd[i, 25:27, e + GROW:e + GROW + 5] = True
    seen_once = np.zeros_like(a)
    seen_once[0, 24:28, 20:40] = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec = st.run(preview(root, people | crowd, {"a": a, "b": b}, levels={64: crowd}, found=[a, b, seen_once]), root / "output", "s", MODEL, "m",
                     say=lambda *_: None)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
    loose = crowd & ~st.grown(a, GROW) & ~st.grown(b, GROW)
    assert N - 1 <= reach and (back & loose).sum() == loose.sum(), "an arm found on the first frame alone was handed back on the frames after it"
    kept = sorted((k["frame"], bool(k.get("on_people")), bool(k.get("on_people_held"))) for k in rec["kept_though_touching"] if "Left_Lower_Arm" in k.get("core", {}))
    assert kept == [(FIRST, True, False)] + [(FIRST + i, False, True) for i in range(1, N)], kept
    return f"a person found {reach} frames away still keeps a piece a stranger, listed with the frame they were found on; {reach + 1} away, the piece is hers"


def one_file_serves_every_subjects_load() -> str:
    """The strangers file is one a preview and serves every subject's load (the reviewer's point on the build, 2026-10-23): a piece
    is handed back only where, for EACH subject it touches, it lies on no person but that subject. A piece of B's arm on B's own
    detection that touches A, alone or with B, stays a stranger, so it is kept out of A's load; touching B alone, it is B's."""
    grow = 2
    t = {"a": np.zeros((1, 40, 40), bool), "b": np.zeros((1, 40, 40), bool)}
    t["a"][0, 5:20, 5:15], t["b"][0, 5:20, 25:35] = True, True
    own = {k: st.grown(v, grow) for k, v in t.items()}
    out = []
    for name, rows, cols in (("touches a", (22, 24), (8, 13)), ("touches a and b", (10, 12), (17, 23)), ("touches b", (22, 24), (28, 33))):
        piece = np.zeros((1, 40, 40), bool)
        piece[0, rows[0]:rows[1], cols[0]:cols[1]] = True
        det_b = t["b"][0] | piece[0]                        # B's detection takes the piece in: B's arm
        others, _ = st.others_of([[(1, t["a"][0]), (2, det_b)]], t, 0.5)
        assert [o[3] for o in others[0]] == ["a", "b"], others
        _, given, kept = st.handed_back(piece, own, piece, 0, others=st.held_others(others))
        touches = (kept or given)[0].get("touches") or [(given[0]["subject"])] + given[0].get("touches_also", [])
        out.append((name, sorted(touches), "kept" if kept else given[0]["by"]))
        if kept:
            assert kept[0]["on_subjects"] == ["b"] and kept[0]["on_people"] == [2], kept
    # THE CASE THAT GOES RED when a subject's own detection is no other person to the subject beside her
    assert out == [("touches a", ["a"], "kept"), ("touches a and b", ["a", "b"], "kept"), ("touches b", ["b"], st.NO_OTHER)], out
    return "a piece on B's detection touching A, or A and B, stays a stranger (out of A's load); touching B alone, it is B's"


def the_card_is_freed_first() -> str:
    """`people` and `run` each load a model in a process of their own beside ComfyUI, which keeps its models on the card after a
    render: from the command line each asks the server at `--host` to free the card (`free_server`, `load_chain.free_card`)
    before its model touches the card, says so, and carries on where no server answers. `run` makes a missing people file first in
    a child process told the same host, which exits before the part model loads; a child that fails leaves nothing written."""
    import contextlib
    import io
    import urllib.error
    import load_chain
    people, a, b = picture()
    calls: list = []
    real_free, real_run, real_seg, real_sam = load_chain.free_card, subprocess.run, st.seg_model, st.sam_people
    graph = {"100": {"class_type": st.CHECKPOINT_NODE, "inputs": {"ckpt_name": "sam.safetensors"}},
             "1005": {"class_type": st.TRACKER_NODE, "inputs": {"segmenter": ["100", 0], "subject_phrase": "person", "detection_threshold": 0.5,
                                                                "max_people": 16}}}

    def child(ok: bool):
        def fake(argv, *x, **k):
            if len(argv) > 2 and str(argv[1]).endswith("strangers.py") and argv[2] == "people":
                calls.append(("people child", list(map(str, argv))))
                if ok:
                    st.people_pass(Path(argv[argv.index("--preview") + 1]), Path(argv[argv.index("--output") + 1]), "s", detector([a, b]), ASKS,
                                   say=lambda *_: None)
                return subprocess.CompletedProcess(argv, 0 if ok else 3)
            return real_run(argv, *x, **k)
        return fake
    try:
        load_chain.free_card = lambda host: (calls.append(("free", host)), "freed")[1]
        st.seg_model = lambda folder: (calls.append(("part model", folder)), MODEL)[1]
        def fresh(asks):
            calls.append(("detector", asks["phrase"]))
            d = detector([a, b])
            d.device = "a stand-in"
            return d
        st.sam_people = fresh
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = preview(root, people, {"a": a, "b": b})
            (folder / "preview.json").write_text(json.dumps(graph))
            pfile = st.people_file_of(json.loads((folder / "job.json").read_text())["job"], root / "output")
            gone = lambda: (pfile.unlink(), pfile.with_suffix(".json").unlink())   # noqa: E731
            base = ["--preview", str(folder), "--by", "s", "--output", str(root / "output"), "--host", "h:1"]
            # the people pass from the command line: freed, then the detector
            gone()
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                assert st.main(["people", *base]) == 0
            assert calls == [("free", "h:1"), ("detector", "person")] and "the card freed before the model loads" in said.getvalue(), (calls, said.getvalue())
            # no server: said, and carried on
            calls.clear()
            gone()

            def down(host):
                raise urllib.error.URLError("no server")
            load_chain.free_card = down
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                assert st.main(["people", *base]) == 0
            assert calls == [("detector", "person")] and "did not free the card" in said.getvalue() and pfile.is_file(), (calls, said.getvalue())
            load_chain.free_card = lambda host: (calls.append(("free", host)), "freed")[1]
            # THE CASE THAT GOES RED when `run` does not make its people file in a child told the host, or frees the card after the part model
            calls.clear()
            gone()
            subprocess.run = child(True)
            with contextlib.redirect_stdout(io.StringIO()):
                assert st.main(["run", *base, "--model-folder", "/store/m"]) == 0
            subprocess.run = real_run
            assert [c[0] for c in calls] == ["people child", "free", "part model"] and calls[0][1][-2:] == ["--host", "h:1"], calls
            assert (root / "output/Video/v/keep_outs/pv/pv_strangers.mkv").is_file()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = preview(root, people, {"a": a, "b": b})
            pfile = st.people_file_of(json.loads((folder / "job.json").read_text())["job"], root / "output")
            pfile.unlink()
            pfile.with_suffix(".json").unlink()
            calls.clear()
            subprocess.run = child(False)
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                assert st.main(["run", "--preview", str(folder), "--by", "s", "--output", str(root / "output"), "--model-folder", "/store/m"]) == 5
            subprocess.run = real_run
            assert [c[0] for c in calls] == ["people child"] and "the people pass ended with 3" in said.getvalue(), (calls, said.getvalue())
            assert calls[0][1][-2:] == ["--host", load_chain.HOST] and not (root / "output/Video/v/keep_outs/pv/pv_strangers.mkv").exists()
    finally:
        load_chain.free_card, subprocess.run, st.seg_model, st.sam_people = real_free, real_run, real_seg, real_sam
    return ("each pass asks the server to free the card before its model, says so, and carries on with no server; `run`'s child makes the "
            "people file told the same host and exits before the part model; a failed child leaves nothing written")


def sam_people_numbers_left_to_right() -> str:
    """`sam_people`, with core's detect node, the checkpoint loader and the tracker's phrase stood in for: the people come back in
    shot_table's person order (left to right by the centre column of the mask), each score with its mask, and the phrase asked is
    the tracker's own `counted` phrase."""
    import types
    import torch
    asked: list = []
    masks = torch.zeros((3, 8, 40))
    masks[0, :, 30:36], masks[1, :, 0:6], masks[2, :, 15:21] = 1, 1, 1      # right, left, middle, as the detector returned them

    class Detect:
        @staticmethod
        def execute(model, image, conditioning=None, threshold=None, individual_masks=None):
            asked.append((conditioning, threshold, individual_masks, tuple(image.shape)))
            return types.SimpleNamespace(args=(masks, [[{"score": 0.9}, {"score": 0.8}, {"score": 0.7}]]))

    class Clip:
        def tokenize(self, text):
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return f"cond of {tokens}"
    fakes = {"comfy_extras.nodes_sam3": types.SimpleNamespace(SAM3_Detect=Detect), "_lib.sam3_core": types.SimpleNamespace(load_core=lambda c: ("model", Clip())),
             "_h3pack.subject_track": types.SimpleNamespace(counted=lambda phrase, most: f"{phrase}:{most}")}
    kept = {k: sys.modules.get(k) for k in fakes}
    try:
        sys.modules.update(fakes)
        detect = st.sam_people({**ASKS, "corrections": None})
        got, scores = detect(np.zeros((8, 40, 3), np.uint8))
    finally:
        for k, v in kept.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    # THE CASE THAT GOES RED when the people are not put in the shot table's order, or a score parts from its mask
    assert [int(np.nonzero(m)[1].mean()) for m in got] == [2, 17, 32] and scores == [0.8, 0.7, 0.9], (scores,)
    assert asked == [("cond of person:16", 0.5, True, (1, 8, 40, 3))], asked
    return "the detector's people come back left to right, each score with its mask, asked with the tracker's own phrase"


def the_people_pass() -> str:
    """The people file (`people_pass`, read by `people_on`): the detector is handed the preview's own frames on its own clock,
    cropped as the loader crops; each frame's people come back as it gave them, numbered from 1; and `run` takes them only from
    a file of this preview, its frames and its canvas, unchanged since its record hashed it. Without one it refuses before
    the part model runs, naming the command."""
    people, a, b = picture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seen: list = []
        folder = preview(root, people, {"a": a, "b": b}, seen=seen)
        pjob = json.loads((folder / "job.json").read_text())["job"]
        path = root / "output/Video/v/keep_outs/pv/pv_people.npz"
        assert st.people_file_of(pjob, root / "output") == path and path.is_file(), path
        # THE CASE THAT GOES RED on a wrong clock or no loader crop for the detector
        assert len(seen) == N and all(((seen[i][..., 0] > 127) == people[i]).all() for i in range(N)), "the detector was not handed the preview's frames"
        found, r = st.people_on(pjob, root / "output")
        assert [[p for p, _ in f] for f in found] == [[1, 2]] * N and all((found[i][0][1] == a[i]).all() and (found[i][1][1] == b[i]).all() for i in range(N))
        assert r["asked_as"] == ASKS and r["people_a_frame"] == [2] * N and r["frames"] == N and r["first_source_frame"] == FIRST, r
        refused(lambda: st.people_pass(folder, root / "output", "s", detector([a, b]), ASKS), "is there", "never replaced")
        for key, wrong in (("preview", "pv2"), ("frames", N + 1), ("canvas", [W + 1, H])):
            rp = path.with_suffix(".json")
            keep = rp.read_text()
            rp.write_text(json.dumps({**json.loads(keep), key: wrong}))
            refused(lambda: st.people_on(pjob, root / "output"), f"says {key}", "nothing written")
            rp.write_text(keep)
        keep = path.read_bytes()
        got = dict(np.load(path).items())
        got["person"] = got["person"][::-1].copy()          # the same arrays, the people numbered otherwise: another file
        np.savez_compressed(path, **got)
        # THE CASE THAT GOES RED when a people file changed since its record hashed it is taken
        assert path.read_bytes() != keep
        refused(lambda: st.people_on(pjob, root / "output"), "not the file its record hashed")
        path.write_bytes(keep)
        assert st.people_on(pjob, root / "output")[1]["sha256"], "the file put back was refused"
        bad = lambda pic: (np.zeros((1, H - 1, W), bool), [0.9])  # noqa: E731
        path.unlink()
        path.with_suffix(".json").unlink()
        refused(lambda: st.people_pass(folder, root / "output", "s", bad, ASKS), "the detector gave 1 masks", "nothing written")
        assert not path.exists()
        refused(lambda: st.run(folder, root / "output", "s", MODEL, "m"), "pv_people.npz or its record is not there", "bench/strangers.py people --preview")
        assert not (root / "output/Video/v/keep_outs/pv/pv_strangers.mkv").exists(), "a strangers file was written with no people file"
    return ("the detector is handed the preview's own frames, cropped; its people are read back as it numbered them from a file of this preview "
            "and frames only, unchanged; no strangers file is made without one")


def how_the_tracker_asked() -> str:
    """`tracker_asks` reads the preview's own graph: the checkpoint, the corrections wired between it and the tracker (none for the
    stock model), the phrase, the threshold and the most people, and refuses a graph it cannot read that from."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        refused(lambda: st.tracker_asks(folder), "holds no preview.json")
        tracker = {"class_type": st.TRACKER_NODE, "inputs": {"segmenter": ["101", 0], "subject_phrase": "person", "detection_threshold": 0.5, "max_people": 16}}
        graph = {"100": {"class_type": st.CHECKPOINT_NODE, "inputs": {"ckpt_name": "sam3.1.safetensors"}},
                 "101": {"class_type": st.CORRECTIONS_NODE, "inputs": {"segmenter": ["100", 0], "correct_image_range": True, "correct_text_activation": False}},
                 "1005": tracker}
        (folder / "preview.json").write_text(json.dumps(graph))
        got = st.tracker_asks(folder)
        assert got == {"checkpoint": "sam3.1.safetensors", "corrections": {"correct_image_range": True, "correct_text_activation": False},
                       "phrase": "person", "threshold": 0.5, "max_people": 16}, got
        graph["1105"] = {"class_type": st.TRACKER_NODE, "inputs": {**tracker["inputs"], "segmenter": ["100", 0]}}
        (folder / "preview.json").write_text(json.dumps(graph))
        refused(lambda: st.tracker_asks(folder), "2 trackers that ask for people differently")
        graph["1105"]["inputs"]["segmenter"] = ["101", 0]
        (folder / "preview.json").write_text(json.dumps(graph))
        assert st.tracker_asks(folder) == got, "two trackers asking alike were refused"
        graph["1005"]["inputs"]["max_people"] = ["7", 0]
        (folder / "preview.json").write_text(json.dumps(graph))
        refused(lambda: st.tracker_asks(folder), "does not read")
        (folder / "preview.json").write_text(json.dumps({"100": graph["100"]}))
        refused(lambda: st.tracker_asks(folder), "has no tracker")
    return "the checkpoint, the corrections between it and the tracker, the phrase, threshold and most people are read off the graph; trackers asking differently are refused"


def another_person_the_part_model_missed_is_a_stranger() -> str:
    """2026-10-27, on 4647k: whole-frame Sapiens2 missed the dark-haired bystander on 4686-4693 (c == 0) while SAM 3.1 found
    2,000-3,500 px of her. Every other person the detector found (`others[i]` where `o[3] is None`) is included where the part
    model found nobody, with a core so `handed_back` keeps them even when touching a subject; a subject's own detection (`o[3]`
    not None) is not."""
    people, a, b = picture()
    bystander = np.zeros_like(a)
    for i in range(N):
        e = 14 + 2 * i
        bystander[i, 16:20, e + GROW:e + GROW + 3] = True       # touching a's grow (clear of b's), and black in `people` so MODEL gives c == 0
    her_det = a.copy()
    for i in range(N):
        e = 14 + 2 * i
        her_det[i, 20:22, e + GROW:e + GROW + 3] = True         # a sliver on her own detection (>0.5 on a) where MODEL gives c == 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec = st.run(preview(root, people, {"a": a, "b": b}, found=[her_det, b, bystander]), root / "output", "s", MODEL, "m", say=lambda *_: None)
        back = read_mask(rec["file"], (W, H), 0, 0, N)[0]
    assert back[bystander].all(), "a bystander the detector found where the part model found nobody was left out or handed back"
    assert not (back & (her_det & ~a)).any(), "a subject's own detection where the part model found nobody became a stranger"
    return "another person the detector found where the part model found nobody stays a stranger, even touching a subject; her own detection does not"


case("a subject, and their rim within the grow, is not a stranger", a_subject_and_their_rim_are_not_strangers)
case("another person the part model missed is a stranger", another_person_the_part_model_missed_is_a_stranger)
case("where it writes, its clocks, and its record", where_it_writes)
case("it writes over nothing", it_writes_over_nothing)
case("a lost track is held a little, and no further", a_lost_track_is_held_a_little)
case("nothing of a subject's own is a stranger", nothing_of_a_subjects_own_is_a_stranger)
case("a piece touching her with no body core is hers; a face or an arm stays a stranger", a_piece_touching_her_with_no_core_is_hers)
case("her own hair past her track is hers", her_own_hair_past_her_track)
case("her own clothing on no other person the detector found is hers", her_own_clothing_on_no_other_person_is_hers)
case("a detection is hers when it lies mostly on her track", a_detection_is_hers_by_her_track)
case("a person the detector drops for a moment is held", a_person_dropped_for_a_moment_is_held)
case("one file serves every subject's load", one_file_serves_every_subjects_load)
case("the people pass, and no strangers file without it", the_people_pass)
case("the card is freed first, and run's people child exits before the part model", the_card_is_freed_first)
case("sam_people numbers the people left to right", sam_people_numbers_left_to_right)
case("the people are asked for as the preview's tracker asked", how_the_tracker_asked)
sys.exit(finish())
