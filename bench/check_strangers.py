#!/usr/bin/env python3
"""The strangers keep-out (`bench/strangers.py`) on a made preview: what it takes out, where it writes, what it refuses.

A preview of two subjects on four frames is made in a temporary folder: a source of white blocks on black, the two
subjects' tracks, the preview's job file and record. The part model is handed in as "every white pixel is a
person", so what is checked is the tool's arithmetic and its files, not the model.

1. **A subject is not a stranger, nor is the rim round them.** A white pixel inside a subject's track, or within
   `h3_config.MASKED_STRANGERS["grow_px"]` of it, is not in the file; one further off is. Both subjects are taken out.
2. **Where it writes** is the path the job step (`bench/shot_to_load.py`) looks for, on the preview's frames, with a
   record beside it saying the grow and the figures.
3. **It writes over nothing**: a file that is there is refused, as are a preview listing two tracks for a subject, a
   source shorter than the preview, and an output key that names no video."""
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

W, H, N = 64, 48, 4
GROW = st.h3_config.MASKED_STRANGERS["grow_px"]


def refused(fn, *words: str) -> None:
    try:
        fn()
    except st.Refused as e:
        assert all(w in str(e) for w in words), f"refused, but not with {words}: {e}"
        return
    raise AssertionError(f"not refused: {words}")


def picture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The people on the frame, subject a's track, subject b's track, and where strangers are, [N, H, W] each."""
    people, a, b = (np.zeros((N, H, W), bool) for _ in range(3))
    a[:, 10:30, 4:20] = True
    b[:, 10:30, 40:50] = True
    people |= a | b
    people[:, 10:30, 20:20 + GROW - 2] = True             # the part model past a's edge, within the grow: hers
    people[:, 40:46, 4:30] = True                         # a stranger well off both
    people[:, 10:30, 50 + GROW + 3:W] = True              # a stranger beside b, further than the grow
    want = people & ~st.grown(a, GROW) & ~st.grown(b, GROW)
    return people, a, b, want


def preview(root: Path, people: np.ndarray, a: np.ndarray, b: np.ndarray, frames: int = N, listed_twice: bool = False) -> Path:
    out = root / "output" / "Video/v/s/pv"
    out.mkdir(parents=True)
    src = root / "source.mkv"
    rgb = np.repeat((people[:frames].astype(np.uint8) * 255)[..., None], 3, -1)
    enc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                          "-c:v", "ffv1", "-pix_fmt", "bgr0", str(src)], input=rgb.tobytes())
    assert enc.returncode == 0
    files = []
    for label, m in (("a", a), ("b", b)):
        write_mask_video(out / f"pv_{label}_track_00001.mkv", m)
        files.append(f"Video/v/s/pv/pv_{label}_track_00001.mkv")
    if listed_twice:
        files.append("Video/v/s/pv/pv_a_track_00002.mkv")
    folder = root / "data" / "2026-01-01_pv_job"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"job": {"name": "pv", "first": 0, "frames": N, "source": str(src), "canvas": [W, H], "output": "Video/v/s",
                                                         "subjects": [{"label": "a"}, {"label": "b"}]}}))
    (folder / "preview.queued.json").write_text(json.dumps({"status": "success", "files": files}))
    return folder


def a_subject_and_their_rim_are_not_strangers() -> str:
    people, a, b, want = picture()
    got = st.strangers_of(people, [a, b], GROW)
    assert (got == want).all(), "the strangers are not the people less both tracks grown"
    assert not got[:, 10:30, 20:20 + GROW - 2].any(), "her rim within the grow is a stranger"
    assert got[:, 40:46, 4:30].any() and got[:, 10:30, 50 + GROW + 3:W].all(), "a stranger off the grow was taken out"
    assert st.strangers_of(people, [a], GROW)[:, 10:30, 40:50].all(), "with one track taken out, the other subject should be in the file"
    return f"a white pixel within {GROW} px of either subject's track is taken out, both subjects are, and a person further off stays"


def where_it_writes() -> str:
    people, a, b, want = picture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder = preview(root, people, a, b)
        said = []
        rec = st.run(folder, root / "output", "the session", lambda pic: pic[..., 0] > 127, "a fake model", say=said.append)
        path = root / "output" / "Video/v/keep_outs/pv/pv_strangers.mkv"
        import shot_to_load as stl
        assert path == root / "output" / stl.KEEP_OUTS_AT.format(video="v", preview="pv") / "pv_strangers.mkv" and Path(rec["file"]) == path, rec["file"]
        back = read_mask(str(path), (W, H), 0, 0, N)[0]
        assert (back == want).all(), "the file is not the strangers"
        written = json.loads(path.with_suffix(".json").read_text())
        assert written["grow_px"] == GROW and written["figures"]["frames_with_a_stranger"] == N and written["subjects_taken_out"].keys() == {"a", "b"}, written
        assert "a person the part model does not find" in written["cannot_see"] and said and "strangers on 4 of them" in said[0], said
        refused(lambda: st.run(folder, root / "output", "the session", lambda pic: pic[..., 0] > 127, "m"), "is there", "never replaced")
    return "the file is at the job step's KEEP_OUTS_AT path, on the preview's frames, equal to the strangers; its record says the grow, both subjects and what it cannot see"


def it_writes_over_nothing() -> str:
    people, a, b, _ = picture()
    model = lambda pic: pic[..., 0] > 127                 # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        refused(lambda: st.run(preview(root, people, a, b, listed_twice=True), root / "output", "s", model, "m"), "lists 2 track videos for 'a'")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        refused(lambda: st.run(preview(root, people, a, b, frames=N - 1), root / "output", "s", model, "m"), "gave 3 frames", "nothing written")
        assert not (root / "output/Video/v/keep_outs").exists() or not any((root / "output/Video/v/keep_outs").rglob("*.mkv")), "a file was written on a short source"
    refused(lambda: st.file_of({"name": "pv", "output": "elsewhere"}, Path("/o")), "not Video/<video>/<session>")
    return "two tracks for one subject, a source shorter than the preview, and an output key with no video are refused, and nothing is written"


case("a subject, and their rim within the grow, is not a stranger", a_subject_and_their_rim_are_not_strangers)
case("where it writes, and its record", where_it_writes)
case("it writes over nothing", it_writes_over_nothing)
sys.exit(finish())
