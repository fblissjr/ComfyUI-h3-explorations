#!/usr/bin/env python3
"""The stacked delivery clip (`bench/stack_delivery.py`) on a clip and pieces whose every frame is known by construction.

Small files are made with ffmpeg in a temporary folder: a source and a delivery clip of twelve frames whose
pictures differ on every frame; two pieces' mask pictures and two reference videos that are one flat grey level a
frame, a different level on every frame, so a panel says which frame of which file it shows by its level; the
picture beside each piece with a saved graph as ComfyUI writes one; and a table of those pieces.

1. **The table is read through the assembler's readers**: a row's piece, its first frame, whose it is and what it
   gives back; `--alias` renames a subject on the picture, in the rows and in the `source` lines alike. A
   reference's path is read out of a picture's metadata, so it is opened only when it is a plain video file on
   this machine: a URL, a protocol string, a name that is no video's and a file that is not there are each
   refused with the reason, and a real video is kept (the control).
2. **What the middle row reads, frame by frame**: a row's mask picture on the frame counted from the PIECE's
   first frame, and its reference on the frame counted from the GRAPH's first frame, which differ for a load
   that continued a stored window (the piece begins later than its graph loads from). A frame no row covers
   has no panel. The words name the pieces, what each gives back, each reference with its frame, and the
   table's reason where a subject is left as the source.
3. **The built file shows those frames**: read back, each panel's grey level is the one that frame of that file
   was made with, the control being the level the wrong frame would have; a frame with no render is black in
   the middle; the top row is the source's own frame and the bottom the clip's, each nearer its own frame than
   the frames either side.
4. **`check` reads the file back**: the frame counts, the sound's packets against the clip's, no frame where a
   neighbour fits better. Its measure, given a row shifted by a frame, says so (the control).
5. **A table that lays nothing is refused.**

Needs ffmpeg and ffprobe, and Pillow. No model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_stack_delivery.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import stack_delivery as sd  # noqa: E402

W, H, FRAMES = 256, 192, 12
MASK = lambda n: 40 + 30 * n          # a mask picture's grey level on its frame n   # noqa: E731
REF = lambda n: 30 + 20 * n           # a reference's grey level on its frame n       # noqa: E731
NEAR = 9                              # grey levels: two encodes and a resize between a level made and a level read


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def flat(path: Path, size: str, frames: int, level: str) -> None:
    """A video that is one grey level a frame, `level` an expression of the frame number N."""
    ffmpeg("-f", "lavfi", "-i", f"nullsrc=size={size}:rate=24,format=gray,geq=lum='{level}'", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path))


def picture(path: Path, first: int, ref: Path | str | None, ref_start: int = 0) -> None:
    """The picture beside a piece, with the graph that made it: where it loaded from, and its reference wired in."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    load = lambda video, at: {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": video, "start_time": (at - 0.4) / 24.0 if at else 0.0}}
    graph = {"1": load("/input/copy.mov", first), "9": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"frames": ["1", 0]}}}
    if ref is not None:
        graph["6"] = load(str(ref), ref_start)
        graph["9"]["inputs"]["motion_video"] = ["6", 0]
    info = PngInfo()
    info.add_text("prompt", json.dumps(graph))
    Image.new("RGB", (2, 2)).save(path, pnginfo=info)


class Made:
    """The files of one small delivery: built once, read by every case."""
    def __init__(self, root: Path):
        self.root = root
        self.source, self.clip = root / "source.mp4", root / "clip.mp4"
        for path, look in ((self.source, "testsrc2"), (self.clip, "testsrc2")):
            extra = [] if path is self.source else ["-vf", "hue=h=90,negate"]
            ffmpeg("-f", "lavfi", "-i", f"{look}=size={W}x{H}:rate=24", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
                   "-frames:v", str(FRAMES), "-shortest", *extra, "-pix_fmt", "yuv420p", "-c:a", "aac", str(path))
        self.ref_a, self.ref_b = root / "ref_a.mp4", root / "ref_b.mp4"
        flat(self.ref_a, f"{W}x{H}", 11, "30+20*N")
        flat(self.ref_b, f"{W}x{H}", 11, "30+20*N")
        # piece a: source frames 2 to 7, its file and its graph both begin on 2, its reference from the reference's frame 3
        # piece b: a load that CONTINUED a stored window: its graph loaded from 5, its file begins on 8; reference from frame 0
        self.a, self.b = root / "pa_00001.mp4", root / "pb_00001.mp4"
        for piece, first, ref, ref_start, frames in ((self.a, 2, self.ref_a, 3, 6), (self.b, 5, self.ref_b, 0, 3)):
            flat(piece, f"{W}x{H}", frames, "128")
            flat(piece.with_name(piece.stem + "_with_mask.mp4"), f"{W}x{H * 2}", frames, "40+30*N")
            picture(piece.with_suffix(".png"), first, ref, ref_start)
        self.table = root / "t.txt"
        self.table.write_text(f"# a table for the check\n2-7 {self.a} 2 subject=a restore=b:whole\n8-10 {self.b} 8 subject=b\n"
                              "source 0-1 subject=a not in the shot\nsource 11-11 subject=b turned away\n")
        self.out = root / "stacked.mp4"

    def build(self, **more) -> None:
        given = dict(source=str(self.source), table=str(self.table), joined=str(self.clip), out=str(self.out), work=str(self.root / "work"),
                     tag="", alias=[], copy_rate=24.0, only=None)
        given.update(more)
        sd.build(argparse.Namespace(**given))


def grey(path: Path, crop: str, w: int, h: int) -> np.ndarray:
    """Every frame of a file, one crop of it, grey: [n, h, w]."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", crop, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(float)


MADE: list[Made] = []
HOLD = tempfile.TemporaryDirectory()


def made() -> Made:
    if not MADE:
        MADE.append(Made(Path(HOLD.name)))
    return MADE[0]


def the_table_as_read() -> str:
    m = made()
    rows, intents = sd.read_rows(str(m.table), 24.0, {})
    assert [(r["first"], r["last"], r["name"], r["piece_first"], r["graph_first"], r["subject"], r["less"]) for r in rows] == [
        (2, 7, "pa", 2, 2, "a", ["b:whole"]), (8, 10, "pb", 8, 5, "b", [])], rows
    assert rows[0]["ref"] == (str(m.ref_a), 3) and rows[1]["ref"] == (str(m.ref_b), 0) and rows[0]["mask"].endswith("pa_00001_with_mask.mp4"), rows
    assert [(i["first"], i["subject"], i["why"]) for i in intents] == [(0, "a", "not in the shot"), (11, "b", "turned away")], intents
    renamed, their_intents = sd.read_rows(str(m.table), 24.0, {"b": "the second subject"})
    assert renamed[0]["less"] == ["the second subject:whole"] and renamed[1]["subject"] == "the second subject" and their_intents[1]["subject"] == "the second subject"
    # a piece with no picture beside it: no reference, and its graph's first frame is taken as the piece's
    bare = m.root / "bare.txt"
    bare.write_text(f"3-4 {m.root / 'nothing_00001.mp4'} 3\n")
    (only,), _ = sd.read_rows(str(bare), 24.0, {})
    assert (only["graph_first"], only["ref"], only["subject"]) == (3, None, None), only
    # A REFERENCE'S PATH IS READ FROM A PICTURE'S METADATA: only a plain video file on this machine is opened. A URL, a
    # protocol string, a file that is no video by its name and a file that is not there each leave the row with no
    # reference and the reason; the control is the same piece naming a real video, which keeps it
    odd = m.root / "odd_00001.mp4"
    odd.write_bytes(b"")
    (m.root / "notes.txt").write_text("not a video")
    odd_table = m.root / "odd.txt"
    odd_table.write_text(f"3-4 {odd} 3\n")
    for named, why in (("http://example.invalid/x.mp4", "not a path on this machine"), ("concat:/etc/hostname|/etc/hosts", "not a path on this machine"),
                       (str(m.root / "notes.txt"), "does not end in one of"), (str(m.root / "gone.mp4"), "not a file on this machine")):
        picture(odd.with_suffix(".png"), 3, named)          # as written: a URL keeps its two slashes
        (row,), _ = sd.read_rows(str(odd_table), 24.0, {})
        assert row["ref"] is None and why in row["ref_refused"] and sd.panels_of(3, [row])[1:] == [], (named, row)
    picture(odd.with_suffix(".png"), 3, m.ref_a)
    (row,), _ = sd.read_rows(str(odd_table), 24.0, {})
    assert row["ref"] == (str(m.ref_a), 0) and row["ref_refused"] is None, row
    return ("a row's piece, first frame, subject and what it gives back; an alias renames a subject everywhere; a piece with no picture has no "
            "reference; a reference named in a picture's metadata is opened only when it is a plain video file on this machine")


def what_each_frame_reads() -> str:
    m = made()
    rows, intents = sd.read_rows(str(m.table), 24.0, {})
    kinds = lambda f: [(p["kind"], Path(p["file"]).name, p["frame"]) for p in sd.panels_of(f, rows)]
    assert kinds(0) == [] and kinds(11) == [], "a frame no row covers has a panel"
    assert kinds(4) == [("mask", "pa_00001_with_mask.mp4", 2), ("reference", "ref_a.mp4", 5)], kinds(4)
    # the continued load: its mask picture counts from the piece's first frame (8), its reference from the graph's (5)
    assert kinds(9) == [("mask", "pb_00001_with_mask.mp4", 1), ("reference", "ref_b.mp4", 4)], kinds(9)
    words = sd.words_of(4, rows, intents, sd.panels_of(4, rows))
    assert words == ["ON THIS FRAME: pa (a's), less b:whole", "reference video given to pa: ref_a.mp4, its frame 5"], words
    assert sd.words_of(0, rows, intents, []) == ["ON THIS FRAME: the original, no render laid", "left as the original on purpose, a: not in the shot"]
    both = rows + [dict(rows[1], first=2, last=7, name="pc", ref=None)]
    assert sd.words_of(4, both, intents, sd.panels_of(4, both))[:2] == ["ON THIS FRAME: pa (a's), less b:whole", "and with it: pc (b's)"]
    return ("a mask picture is read on the frame counted from the piece's first frame and a reference on the frame counted from the graph's, "
            "which differ for a continued load; the words name the pieces, what each gives back, each reference's frame and the table's reasons")


def the_built_file() -> str:
    m = made()
    m.build()
    line, bar = sd.sizes(H)
    w, total, n = sd.probe(str(m.out))
    assert (w, total, n) == (W, H * 3 + bar, FRAMES), (w, total, n)
    # two panels a frame here, tiled at half size across the middle row: the mask at the left, the reference at the right.
    # Read at each tile's lower right quarter, clear of its label.
    pw, ph = W // 2, H // 2
    top = H + ph // 2
    mask = grey(m.out, f"crop={pw // 2}:{ph // 2}:{pw // 2}:{top + ph // 2}", pw // 2, ph // 2).mean(axis=(1, 2))
    ref = grey(m.out, f"crop={pw // 2}:{ph // 2}:{pw + pw // 2}:{top + ph // 2}", pw // 2, ph // 2).mean(axis=(1, 2))
    for f, mask_frame, ref_frame, wrong_ref in ((2, 0, 3, None), (4, 2, 5, None), (7, 5, 8, None), (8, 0, 3, 0), (9, 1, 4, 1), (10, 2, 5, 2)):
        assert abs(mask[f] - MASK(mask_frame)) < NEAR, f"frame {f}: the mask panel reads {mask[f]:.0f}, its frame {mask_frame} was made {MASK(mask_frame)}"
        assert abs(ref[f] - REF(ref_frame)) < NEAR, f"frame {f}: the reference panel reads {ref[f]:.0f}, its frame {ref_frame} was made {REF(ref_frame)}"
        if wrong_ref is not None:      # the control: counted from the piece's first frame, the continued load would show this level
            assert abs(ref[f] - REF(wrong_ref)) > NEAR, f"frame {f}: the fixture cannot tell the graph's frame from the piece's"
    for f in (0, 1, 11):
        assert mask[f] < NEAR and ref[f] < NEAR, f"frame {f}: no row covers it and the middle row is not black ({mask[f]:.0f}, {ref[f]:.0f})"
    # the top row is the source and the bottom the clip, each its own frame: below the label, against the files themselves
    cut = H // 4
    for name, y, path in (("the original row", 0, m.source), ("the clip row", H * 2 + bar, m.clip)):
        row = grey(m.out, f"crop={W}:{H - cut}:0:{y + cut}", W, H - cut)
        own = grey(path, f"crop={W}:{H - cut}:0:{cut}", W, H - cut)
        for f in range(1, FRAMES - 1):
            here, beside = np.abs(row[f] - own[f]).mean(), min(np.abs(row[f] - own[f - 1]).mean(), np.abs(row[f] - own[f + 1]).mean())
            assert here < beside, f"{name}, frame {f}: {here:.2f} from its own frame and {beside:.2f} from a neighbour"
    other = grey(m.out, f"crop={W}:{H - cut}:0:{cut}", W, H - cut)
    assert np.abs(other[5] - grey(m.clip, f"crop={W}:{H - cut}:0:{cut}", W, H - cut)[5]).mean() > 20, "the fixture's source and clip cannot be told apart"
    assert line >= 10 and bar % 2 == 0
    return ("read back, each panel shows the frame of the file it should, by its level, and the continued load's reference is the graph's frame "
            "and not the piece's; a frame with no render is black in the middle; the top row is the source's frame and the bottom the clip's")


def the_read_back() -> str:
    m = made()
    if not m.out.is_file():
        m.build()
    got = sd.read_back(str(m.source), str(m.clip), str(m.out))
    assert (got["frames"], got["clip_frames"], got["canvas"]) == (FRAMES, FRAMES, [W, H]) and got["sound_packets"]["the_same"], got
    for key in ("the_original_row", "the_clip_row"):
        assert got[key]["a_neighbour_fits_better"] == [] and got[key]["own_frame_fits_best"] >= FRAMES - 2, (key, got[key])
    # the control for the measure itself: a row that is its source one frame late is said to fit a neighbour
    rng = np.random.default_rng(3)
    ref = rng.integers(0, 255, (8, 192, 256)).astype(np.int16)
    late = np.concatenate([ref[:1], ref[:-1]])
    assert len(sd.fits(late, ref)["a_neighbour_fits_better"]) >= 6 and sd.fits(ref, ref)["a_neighbour_fits_better"] == []
    return "the frame counts, the sound's packets and each row against its file read as built; a row one frame late is said to fit a neighbour"


def a_table_that_lays_nothing() -> str:
    m = made()
    empty = m.root / "empty.txt"
    empty.write_text("source 0-11 subject=a nothing rendered yet\n")
    try:
        m.build(table=str(empty), out=str(m.root / "never.mp4"))
    except SystemExit as e:
        assert "lays no piece" in str(e), str(e)
    else:
        raise AssertionError("a table with no piece was stacked")
    assert not (m.root / "never.mp4").exists()
    return "a table with no piece is refused and nothing is written"


case("the table as read, through the assembler's readers", the_table_as_read)
case("what the middle row reads on each frame", what_each_frame_reads)
case("the built file shows those frames", the_built_file)
case("check: the file read back", the_read_back)
case("a table that lays nothing", a_table_that_lays_nothing)

if __name__ == "__main__":
    code = finish()
    HOLD.cleanup()
    sys.exit(code)
