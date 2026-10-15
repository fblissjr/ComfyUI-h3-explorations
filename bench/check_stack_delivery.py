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
   refused with the reason, and a real video is kept (the control). A playlist under a video's name, which
   passes a check of the name and which ffmpeg left to itself follows (the control), does not read: the reader
   tells ffmpeg the container by the suffix.
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
                     tag="", alias=[], copy_rate=24.0, only=None, first=0)
        given.update(more)
        sd.build(argparse.Namespace(**given))


def grey(path: Path, crop: str, w: int, h: int) -> np.ndarray:
    """Every frame of a file, one crop of it, grey: [n, h, w]."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", crop, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(float)


def tile_level(path: Path, rect: list[int]) -> np.ndarray:
    """The mean grey level of the middle half of one tile of the middle row (`sd.layout`'s [x, y, w, h]), on every frame."""
    x, y, w, h = rect
    cw, ch = w // 4 * 2, h // 4 * 2                        # even: ffmpeg's crop on a yuv420p file rounds a size down to one
    return grey(path, f"crop={cw}:{ch}:{(x + w // 4) // 2 * 2}:{(H + y + h // 4) // 2 * 2}", cw, ch).mean(axis=(1, 2))


#: The middle row's ground where a subject's column has no piece on a frame (`stack_delivery.build`), as grey.
SLATE = 24


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
    # A PLAYLIST UNDER A VIDEO'S NAME passes a check of the name, so the reader must not follow it: ffmpeg is told
    # the container by the suffix and the file does not read. The control: left to probe the content, ffmpeg DOES
    # follow this same file to the video it lists, so the fixture is a working playlist and the gap was real
    playlist = m.root / "playlist.mp4"
    playlist.write_text("ffconcat version 1.0\nfile 'source.mp4'\n")
    assert sd.not_a_plain_video(str(playlist)) is None, "the fixture's playlist does not pass the check of its name"
    followed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                               "-of", "csv=p=0", str(playlist)], capture_output=True, text=True).stdout.strip().strip(",")
    assert followed == str(FRAMES), f"the control: ffmpeg left to itself reads {followed!r} frames through the playlist, not the source's {FRAMES}"
    reader = sd.Reader(str(playlist), 0, f"scale={W}:{H}", W, H)
    got = reader.frame(0)
    reader.close()
    assert got is None, "a playlist named as a video was followed"
    real = sd.Reader(str(m.ref_a), 0, f"scale={W}:{H}", W, H)
    assert real.frame(0) is not None, "a real video does not read with its container told"
    real.close()
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
    w, total, n = sd.probe(str(m.out))
    got = sd.read_back(str(m.source), str(m.clip), str(m.out))
    bar = got["size"][1] - 3 * H
    assert (w, n) == (W, FRAMES) and got["canvas"] == [W, H] and bar > 0 and bar % 2 == 0, (w, total, n, got["canvas"])
    # one column a subject for the whole clip, in the order each first lays a piece: a's (pa, 2-7), then b's (pb, 8-10). Each column's
    # mask picture and its reference read where `sd.layout` lays them, and nowhere else (C1, 2026-10-15: tiles moved as pieces came and went)
    assert sd.columns_of(sd.read_rows(str(m.table), 24.0, {})[0], 0, FRAMES - 1) == ["a", "b"]
    cols = sd.layout(W, H, 2)
    mask_a, mask_b = tile_level(m.out, cols[0]["mask"]), tile_level(m.out, cols[1]["mask"])
    ref_a, ref_b = tile_level(m.out, cols[0]["whole"]), tile_level(m.out, cols[1]["whole"])
    for f, mask_frame, ref_frame, wrong_ref, mask, ref, other in ((2, 0, 3, None, mask_a, ref_a, mask_b), (4, 2, 5, None, mask_a, ref_a, mask_b),
                                                                  (7, 5, 8, None, mask_a, ref_a, mask_b), (8, 0, 3, 0, mask_b, ref_b, mask_a),
                                                                  (9, 1, 4, 1, mask_b, ref_b, mask_a), (10, 2, 5, 2, mask_b, ref_b, mask_a)):
        assert abs(mask[f] - MASK(mask_frame)) < NEAR, f"frame {f}: the mask panel reads {mask[f]:.0f}, its frame {mask_frame} was made {MASK(mask_frame)}"
        assert abs(ref[f] - REF(ref_frame)) < NEAR, f"frame {f}: the reference panel reads {ref[f]:.0f}, its frame {ref_frame} was made {REF(ref_frame)}"
        assert abs(other[f] - SLATE) < NEAR, f"frame {f}: the other subject's column, which has no piece here, reads {other[f]:.0f}"
        if wrong_ref is not None:      # the control: counted from the piece's first frame, the continued load would show this level
            assert abs(ref[f] - REF(wrong_ref)) > NEAR, f"frame {f}: the fixture cannot tell the graph's frame from the piece's"
    for f in (0, 1, 11):
        for name, lv in (("a's mask", mask_a), ("b's mask", mask_b), ("a's reference", ref_a)):
            assert abs(lv[f] - SLATE) < NEAR, f"frame {f}: no row covers it and {name} tile reads {lv[f]:.0f}, not the bare ground"
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
    return ("read back, each subject's column shows the frame of the file it should, by its level, in its own place for the whole clip, and the "
            "continued load's reference is the graph's frame and not the piece's; a column with no piece on a frame is bare; the top row is the "
            "source's frame and the bottom the clip's")


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


def a_clip_of_a_span() -> str:
    """THE FAULT OF 2026-10-14: six clips of one shot each were stacked as if each began on source frame 0. The original row
    and the masks row were of other frames, the burned number was the clip's own count, and the check passed, since it
    read the source from 0 too."""
    m = made()
    k, n = 4, 7                                                    # the clip is source frames 4 to 10
    span, out = m.root / "span.mp4", m.root / "span_stacked.mp4"
    ffmpeg("-i", str(m.clip), "-vf", f"select='between(n\\,{k}\\,{k + n - 1})',setpts=N/24/TB", "-an", "-frames:v", str(n), "-pix_fmt", "yuv420p", str(span))
    m.build(joined=str(span), out=str(out), first=k)
    assert sd.probe(str(out))[2] == n, sd.probe(str(out))
    cut = H // 4
    row = grey(out, f"crop={W}:{H - cut}:0:{cut}", W, H - cut)
    own = grey(m.source, f"crop={W}:{H - cut}:0:{cut}", W, H - cut)
    for i in range(n):
        here, from_zero = np.abs(row[i] - own[k + i]).mean(), np.abs(row[i] - own[i]).mean()
        assert here < from_zero and here < 6, f"stacked frame {i}: the original row is {here:.2f} from source frame {k + i} and {from_zero:.2f} from source frame {i}"
    # the middle row is the table's rows on the SOURCE frame: piece a's mask on 4-7, piece b's on 8-10 (as the_built_file reads them, at index f - k)
    cols = sd.layout(W, H, 2)
    mask_a, mask_b = tile_level(out, cols[0]["mask"]), tile_level(out, cols[1]["mask"])
    for f, mask_frame, mask in ((4, 2, mask_a), (7, 5, mask_a), (8, 0, mask_b), (10, 2, mask_b)):
        assert abs(mask[f - k] - MASK(mask_frame)) < NEAR, f"source frame {f}: the mask panel reads {mask[f - k]:.0f}, its frame {mask_frame} was made {MASK(mask_frame)}"
    right = sd.read_back(str(m.source), str(span), str(out), k)
    assert right["the_clip_begins_on_source_frame"] == k and right["the_original_row"]["a_neighbour_fits_better"] == [] and right["the_original_row"]["own_frame_fits_best"] >= n - 2, right
    # THE CONTROL: the same file held to the source from 0, as the check did, does not fit
    wrong = sd.read_back(str(m.source), str(span), str(out), 0)["the_original_row"]
    assert wrong["mean_to_its_own_frame"] > 3 * max(right["the_original_row"]["mean_to_its_own_frame"], 0.5), (wrong, right["the_original_row"])
    # --only outside the clip's own frames is refused; and a build that is not told where the clip begins lays the wrong rows (the control for the fixture)
    try:
        m.build(joined=str(span), out=str(m.root / "x.mp4"), first=k, only="0-3")
        raise AssertionError("--only outside the span clip's frames was taken")
    except SystemExit as e:
        assert "is not inside the clip's frames, 4 to 10" in str(e), e
    untold = m.root / "span_untold.mp4"
    m.build(joined=str(span), out=str(untold), first=0)
    bad = grey(untold, f"crop={W}:{H - cut}:0:{cut}", W, H - cut)
    assert np.abs(bad[2] - own[k + 2]).mean() > np.abs(bad[2] - own[2]).mean(), "the fixture cannot tell a clip of a span from a clip from 0"
    return "a clip of source frames 4 to 10, told so: the original row and the panels are the source's 4 to 10, and the read-back held from 0 does not fit"


def the_clips_own_check_and_whole_words() -> str:
    """The joined clip's check (`<clip>.check.json`) on its frames, in the piece's own column, and every word kept (C1, 2026-10-15:
    a 1-frame flicker the check had flagged, d005, showed nothing on the stack, and the words under the middle row were cut)."""
    m = made()
    long_why = "pb_00001.mp4 is the original on 1 frame(s) a row gives it, " + "and this sentence runs on well past one line of the block " * 3
    check = Path(str(m.clip) + ".check.json")
    check.write_text(json.dumps({"flags": [{"id": "d005", "rule": "piece_changes_nothing", "level": "iffy", "piece": "pb_00001.mp4", "subject": "b",
                                            "source_frames": [[9, 9]], "why": long_why}]}))
    out = m.root / "checked.mp4"
    try:
        m.build(out=str(out))
    finally:
        check.unlink()
    recs = [json.loads(x) for x in out.with_name(out.stem + ".frames.jsonl").read_text().splitlines()]
    at = {r["source_frame"]: r for r in recs}
    assert at[9]["columns"]["b"]["check_flags"] == ["d005"] and at[9]["check_flags"] == ["d005"], at[9]
    assert at[8]["columns"]["b"]["check_flags"] == [] and at[9]["columns"]["a"] is None, (at[8], at[9])
    head = json.loads(out.with_name(out.stem + ".json").read_text())
    # the check's flag is a row of its own for the whole clip (C2, 2026-10-15), drawn whole, bright on its frame only
    assert " ".join(long_why.split()) in " ".join(head["flag_lines"]["d005"]), "the check's words are not whole in its row"
    assert not any(line.endswith("...") for r in recs for line in r["words"]) and not any(x.endswith("...") for v in head["flag_lines"].values() for x in v), \
        "a line of words was cut"
    assert at[9]["changes"] == ["flag d005 starts"] and at[10]["changes"] == ["flag d005 ends"], (at[9]["changes"], at[10]["changes"])
    assert head["columns"] == ["a", "b"] and head["check_flags"][0]["id"] == "d005", head["columns"]
    return "the clip's own check on its frame and in its piece's column, its words whole; no line cut; the columns and flags in the sidecar"


def each_piece_read_through_its_job() -> str:
    """A piece's inputs are read from its job folder, found by its queued record, with the output folder taken from the
    piece's own path (2026-10-15: run with no H3_OUTPUT_DIR the reader refused every piece and the stack called it "no queued
    record"); a refused read is said as refused, with its reason."""
    import what_h3_got
    m = made()
    data = m.root / "data"
    job = data / "2026-01-01_pa_job"
    job.mkdir(parents=True, exist_ok=True)
    under = "Video/v/pa/pa_00001.mp4"
    (job / "pa.render.queued.json").write_text(json.dumps({"files": [under]}))
    rows = [{"name": "pa", "piece": f"/share/out/{under}"}, {"name": "pb", "piece": "/share/out/Video/v/pb/pb_00001.mp4"}]
    seen, held = [], what_h3_got.read
    try:
        what_h3_got.read = lambda folder, load, stage, output=None: seen.append((folder.name, load, stage, output)) or {"inputs": []}
        given, refused = sd.given_of_pieces(rows, data)
        assert seen == [("2026-01-01_pa_job", "pa", "render", Path("/share/out/"))] and list(given) == ["pa"] and refused == {}, (seen, given)
        def refuse(*_a, **_k):
            raise SystemExit("output directory is not where this server writes")
        what_h3_got.read = refuse
        given, refused = sd.given_of_pieces(rows, data)
        assert given == {} and refused == {"pa": "output directory is not where this server writes"}, refused
    finally:
        what_h3_got.read = held
    return "pa read from its job folder with the output folder its own path gives; pb, queued by no record, left out; a refused read kept with its reason"


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
case("a clip of a span that does not begin on frame 0", a_clip_of_a_span)
case("the clip's own check, and the words whole", the_clips_own_check_and_whole_words)
case("each piece read through its job", each_piece_read_through_its_job)
case("a table that lays nothing", a_table_that_lays_nothing)

if __name__ == "__main__":
    code = finish()
    HOLD.cleanup()
    sys.exit(code)
