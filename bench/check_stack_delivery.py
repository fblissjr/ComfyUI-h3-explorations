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
2. **What the right column reads, frame by frame**: a row's mask picture on the frame counted from the PIECE's
   first frame, and its reference on the frame counted from the GRAPH's first frame, which differ for a load
   that continued a stored window (the piece begins later than its graph loads from). A frame no row covers
   has no panel. The words name the pieces, what each gives back, each reference with its frame, and the
   table's reason where a subject is left as the source.
3. **The built file shows those frames**: read back, each panel's grey level is the one that frame of that file
   was made with, the control being the level the wrong frame would have; a column with no piece on a frame is
   the column's bare ground; the top row is the source's own frame and the bottom the clip's, each nearer its
   own frame than the frames either side. A clip of a span is stacked on its source frames (`--first`); the
   clip's own check flags are in their piece's column and in the records whole.
4. **`check` reads the file back**: the frame counts, the sound's packets against the clip's, no frame where a
   neighbour fits better. Its measure, given a row shifted by a frame, says so (the control).
5. **A table that lays nothing is refused**, and **a clip of another size than the source** (named, both sizes):
   a clip of the source's size shows a band on its bottom row the loader's cut takes (the control).

From the blind read of 2026-10-16 (`internal/claude/2026-10-16_mrtony/review_stack_view.md`), on a second fixture whose
source (288x224) is larger than the canvas (256x192), whose clip is the source with U and V moved by 3 levels on every
frame (lossless) and a red box on 2-7, with a zoomed piece, a piece whose mask picture runs out, a queued record with a
day, and a per-frame gap check's runs:

6. **The middle row is the assembler's own test**: its changed pixels on every frame are `assemble_delivery.changed`
   on the two files as yuv420p, 0 where U and V moved by 3; the RGB test it used marks pixels there (the control).
7. **Nothing drawn on the picture**: each of the three rows held pixel for pixel to the picture it should be (the
   middle one recomputed with the assembler's test), its edge rows and label corner included; only a 1 px zoom
   outline and the encode's own smear by a legend or the changed edge are not read.
8. **Every chip said**: `original_tracking_not_known` drawn; many chips on one frame wrap inside the source's width
   and the legend grows to hold them; an unknown kind stops the build; a check flag carries its level.
9. **Who is laid comes from the table**: a piece whose mask picture runs out is still named in the legend and its
   column, its panel said as not read with why.
10. **The day from a record, the head crop said**: the queued record's day, "day not recorded" where none; "(no points:
    no head crop)" where the reference has no points; the zoom records by piece file, one that does not read kept
    with its reason.
11. **`check` reads the canvas and the width**: the canvas and the source's size apart; a file whose metadata gives
    another width is refused; an odd source is refused before anything is encoded.

From the second blind read of 2026-10-16 (`review2_stack_view.md`):

12. **What was not read is said**: with no `--coverage` and no check record, legend 2 says so on every frame and never
    "no gap or flag"; that is said only where both were read. A subject left as the original on purpose is in legend 1.
13. **The bar is off what H3 saw**: a `piece_changes_nothing` bar is a line of the block's label strip; the mask
    picture's bottom is not red, and every line of the label lies inside the block, above the picture.
14. **A trial of some frames reads back**: a build of frames 3-8 is read back from the stack's own record; nothing is cut
    off the rows for a label.
15. **Each piece at its own canvas**: a 320x192 piece beside two of 256x192 shows its whole mask picture, and its zoom box
    goes through the loader's crop of its own canvas.
16. **A refused reference is said** NOT OPENED, with the refusal.
17. **The one-a-second parts** are beside the stack only when both encoders finished: a failed main encode leaves nothing.
18. **No stack without its sidecar**: a sidecar that fails to write is a hard failure naming the stack, and nothing says
    "written" first. The stack's own sidecar is read back (`sidecar.read`): the roles it names and the stack's bytes.

From the owner's go of 2026-10-16 (the squares kept as the original), on a region file written by the node's own
`video_mask.save_window_region` and a track handed as the load's "the subject's mask":

19. **A square kept inside her track** is bright yellow in the WHAT H3 SAW panel, with its chip; a square the node
    regenerated is bare.
20. **A square kept outside her track** is a quiet shade, with no chip.
21. **A render with no region files** says so in its label strip.

From the owner's review page of 2026-10-16 (Chrome failed on level 6.0 views), with the level's macroblock limit lowered
for the fixture:

22. **Every chip said, within level 5.1**: the view stays within the limit, the busiest frame's chips that legend 2 cannot
    hold are in the right column below the blocks, every chip of every frame is in one or the other, and the file is
    tagged level 5.1.
23. **A layout over the level is refused** before anything is encoded, naming the sizes.

From the outside reviewer's read of the squares (`internal/gemini/2026-10-16_review_view_tokens_D13.md`), each on a variant
of the squares fixture:

24. **B1, her track over the whole latent step**: a token hers only on the later frames of its step is inside on the first.
25. **B2, a pair load takes both tracks**: a token in the second of two tracks is inside.
26. **B3, a window of another run is left unread**: a leftover window file over the same frames is named, not counted.
27. **B4, an unlaid frame draws no squares**: the frame whose mask is empty shows none and says why; the frame before has its own.
28. **B5, a track loaded at its own size**: a loader left at 0 by 0 reads the track at the file's size.

From the whole video of 2026-10-16, refused by one row, and then not starting in the browser:

29. **Three subjects and a busy frame within the level**: three blocks and the overflow fit the right column at a limit where
    rounding each block up came to one row over.
30. **The view starts without its end**: the view and each one-a-second part have their index (`moov`) before their media
    and within their first 4 MB; the view's sidecar is of its final bytes.

From the candidate-vs-current comparison mode (`compare`, mode (c), 2026-10-17 `C71` / `C66`):

31. **Compare a candidate cut against the current cut**: three 100%-size left-column picture rows (`ORIGINAL`, `CURRENT CUT`,
    `CANDIDATE CUT`, nothing drawn over any of them), burned-in `frame {f:05d}` in legend 1, two right-column blocks per
    compared subject (`CURRENT (a)` and `CANDIDATE (a)`, each showing its own `_with_mask.mp4` and reference frame), audio
    trimmed and muxed for the span from `--source` even when `--joined` is silent, `<out>.mp4.json` sidecar and
    `<stem>.alignment.txt` written with `ALIGNMENT_NOTE`, offsets `-2..+2` MAE (`best_offset == 0`) and `vs_own < vs_other`,
    and shifted/swapped controls caught by `compare_alignment` and `read_back`.
32. **Candidate-piece shorthand and unchanged subjects**: `--candidate-piece` synthesises the candidate table over the
    current table, unchanged subjects stay in `legend_1` as `(both)` without right-column blocks unless `--subject` names
    them, and `check` reads back a `layout=compare` stack.
33. **Compare refusals**: missing or conflicting candidate table/piece flags, malformed `--candidate-piece`, a candidate cut
    of another size than the source, mismatched spans without `--only`, `--only` outside the candidate cut, a candidate table
    laying no piece on the span, and an unknown `--subject` are each refused in the tool's own words with no file written.

What these cases cannot see: the legends and labels are read from the drawing code's own records (`legend_1`,
`legend_2`, `chips_drawn`, `label_lines`, `label_boxes`), not from their pixels; a mark over 60 levels on under 0.4% of
a row, or within the 5x5 band round the changed edge, passes "nothing drawn on the picture"; the first fixture scans
the repo's own `data/` for queued records.

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
    """The mean grey level of the middle half of one tile of the right column (`sd.right_tiles`'s [x, y, w, h], inside the
    column, which begins at the source's width), on every frame."""
    x, y, w, h = rect
    cw, ch = w // 4 * 2, h // 4 * 2                        # even: ffmpeg's crop on a yuv420p file rounds a size down to one
    return grey(path, f"crop={cw}:{ch}:{(W + x + w // 4) // 2 * 2}:{(y + h // 4) // 2 * 2}", cw, ch).mean(axis=(1, 2))


def head_of(out: Path) -> dict:
    """A built stack's own record (`<stem>.stack.json`)."""
    return json.loads(out.with_name(out.stem + ".stack.json").read_text())


def frames_of(out: Path) -> dict[int, dict]:
    """A built stack's per-frame records, by source frame."""
    return {r["source_frame"]: r for r in (json.loads(x) for x in out.with_name(out.stem + ".stack.frames.jsonl").read_text().splitlines())}


def legends_of(head: dict) -> list[int]:
    """The two legends' heights as the build wrote them (a stack from before 2026-10-16's fixes had one height for both)."""
    return list(head.get("legends") or [head["legend"]] * 2)


def tiles(out: Path | None = None) -> list[dict]:
    """The right column's blocks as the build laid them, read from its own record (the legends' heights, and so the column's
    scale, follow what the clip's frames say)."""
    return head_of(out or made().out)["right_tiles"]


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
    legs = legends_of(head_of(m.out))
    # the layout of 2026-10-16: three rows at the source's own size and two legends on the left, the canvas's column on the right
    assert (w, total, n) == (W + W, 3 * H + sum(legs), FRAMES) and got["canvas"] == [W, H], (w, total, n, got["canvas"])
    # one column a subject for the whole clip, in the order each first lays a piece: a's (pa, 2-7), then b's (pb, 8-10). Each column's
    # mask picture and its reference read where `sd.right_tiles` lays them, and nowhere else (C1, 2026-10-15: tiles moved as pieces came and went)
    assert sd.columns_of(sd.read_rows(str(m.table), 24.0, {})[0], 0, FRAMES - 1) == ["a", "b"]
    cols = tiles()
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
    for name, y, path in (("the original row", 0, m.source), ("the clip row", H * 2 + sum(legs), m.clip)):
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
    cols = tiles(out)
    mask_a, mask_b = tile_level(out, cols[0]["mask"]), tile_level(out, cols[1]["mask"])
    for f, mask_frame, mask in ((4, 2, mask_a), (7, 5, mask_a), (8, 0, mask_b), (10, 2, mask_b)):
        assert abs(mask[f - k] - MASK(mask_frame)) < NEAR, f"source frame {f}: the mask panel reads {mask[f - k]:.0f}, its frame {mask_frame} was made {MASK(mask_frame)}"
    right = sd.read_back(str(m.source), str(span), str(out), k)
    assert right["the_clip_begins_on_source_frame"] == k and right["the_original_row"]["a_neighbour_fits_better"] == [] and right["the_original_row"]["own_frame_fits_best"] >= n - 2, right
    # THE CONTROL: the same file held to the source from 0, as the check did, does not fit. Since the second blind read the check takes
    # the frames from the stack's own record, so the control reads a copy with no record beside it, where the check falls back to --first
    bare = m.root / "bare"
    bare.mkdir(exist_ok=True)
    ffmpeg("-i", str(out), "-c", "copy", "-map_metadata", "0", str(bare / out.name))
    wrong = sd.read_back(str(m.source), str(span), str(bare / out.name), 0)["the_original_row"]
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
    recs = [json.loads(x) for x in out.with_name(out.stem + ".stack.frames.jsonl").read_text().splitlines()]
    at = {r["source_frame"]: r for r in recs}
    assert at[9]["columns"]["b"]["check_flags"] == ["d005"] and at[9]["check_flags"] == ["d005"], at[9]
    assert at[8]["columns"]["b"]["check_flags"] == [] and at[9]["columns"]["a"] is None, (at[8], at[9])
    head = json.loads(out.with_name(out.stem + ".stack.json").read_text())
    # from 2026-10-16 no words are drawn but the legends: the check's flag is a chip on its frames, and its words are whole in the records
    assert head["check_flags"][0]["why"] == long_why, "the check's words are not whole in the records"
    assert not any(line.endswith("...") for r in recs for line in r["words"]), "a line of words was cut"
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
        what_h3_got.read = lambda folder, load, stage, output=None: seen.append((folder.name, load, stage, output)) or {"inputs": [], "folder": folder.name}
        given, refused = sd.given_of_pieces(rows, data)
        assert seen == [("2026-01-01_pa_job", "pa", "render", Path("/share/out/"))] and list(given) == ["pa_00001.mp4"] and refused == {}, (seen, given)
        # TWO FILES OF ONE LOAD (the second blind read of 2026-10-16): each is read from the record that names it, keyed by its file,
        # and neither takes the other's (until then both were keyed by the load's name and the second was never read)
        job2 = data / "2026-01-02_pa_job"
        job2.mkdir(parents=True, exist_ok=True)
        (job2 / "pa.render.queued.json").write_text(json.dumps({"files": ["Video/v/pa/pa_00002.mp4"]}))
        two = rows + [{"name": "pa", "piece": "/share/out/Video/v/pa/pa_00002.mp4"}]
        given, refused = sd.given_of_pieces(two, data)
        assert {k: v["folder"] for k, v in given.items()} == {"pa_00001.mp4": "2026-01-01_pa_job", "pa_00002.mp4": "2026-01-02_pa_job"}, given
        def refuse(*_a, **_k):
            raise SystemExit("output directory is not where this server writes")
        what_h3_got.read = refuse
        given, refused = sd.given_of_pieces(rows, data)
        assert given == {} and refused == {"pa_00001.mp4": "output directory is not where this server writes"}, refused
    finally:
        what_h3_got.read = held
        import shutil
        shutil.rmtree(data / "2026-01-02_pa_job", ignore_errors=True)
    return ("pa read from its job folder with the output folder its own path gives, keyed by its file; two files of one load each read from "
            "its own record; pb, queued by no record, left out; a refused read kept with its reason")


def refused(fn, *words) -> str:
    """`fn` stops with SystemExit whose words hold every one of `words`; its words, for the case's line."""
    try:
        fn()
    except SystemExit as stop:
        assert all(w in str(stop) for w in words), (str(stop), words)
        return str(stop)
    raise AssertionError(f"not refused: {words}")


def the_whole_frame_shown() -> str:
    """C3, 2026-10-15: the stack cut the original and the clip as the loader cuts them, and hid a band at the delivered frame's
    edge. The bottom row is the clip at its own size, the source's: a clip of the source's size (288x224, a source larger
    than the 256x192 canvas) with one bright row along its bottom: the loader's cut takes that row (the control, read from
    the file through FIT), and the stack's clip row shows it, unscaled; the read-back still fits. A clip of ANOTHER size than
    the source (264x200 against 256x192) is refused, naming both sizes: until the blind read of 2026-10-16 it was stretched
    to the source's size without a word, and the middle row marked the misregistration as changed."""
    m, b = made(), big()
    banded = m.root / "banded.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x3c3c3c:s=264x200:r=24", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
           "-frames:v", str(FRAMES), "-shortest", "-vf", "drawbox=x=0:y=199:w=264:h=1:color=white:t=fill", "-pix_fmt", "yuv444p",
           "-c:v", "libx264", "-qp", "0", "-c:a", "aac", str(banded))
    said = refused(lambda: m.build(joined=str(banded), out=str(m.root / "banded_stacked.mp4")), "264 by 200", "256 by 192")
    assert not (m.root / "banded_stacked.mp4").exists(), "a refused clip left a file"
    same = b.root / "banded.mp4"
    ffmpeg("-f", "lavfi", "-i", f"color=c=0x3c3c3c:s={BW}x{BH}:r=24", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
           "-frames:v", str(FRAMES), "-shortest", "-vf", f"drawbox=x=0:y={BH - 1}:w={BW}:h=1:color=white:t=fill", "-pix_fmt", "yuv420p",
           "-c:v", "libx264", "-qp", "0", "-c:a", "aac", str(same))
    cut = grey(same, sd.FIT.format(w=W, h=H), W, H)
    assert cut[:, -2:, W // 4:3 * W // 4].max() < 120, "the control: the loader's cut keeps the band, so the fixture cannot show it hidden"
    out = b.root / "banded_stacked.mp4"
    b.build(joined=str(same), out=str(out), coverage="")
    got = sd.read_back(str(b.source), str(same), str(out))
    legs = legends_of(head_of(out))
    row = grey(out, f"crop={BW}:{BH}:0:{2 * BH + sum(legs)}", BW, BH)
    edge = row[:, -2:, BW // 4:3 * BW // 4].max(axis=(1, 2))
    assert (edge > 150).all(), f"the band at the clip's bottom edge is not in the stack's clip row: {edge}"
    assert got["the_clip_row"]["a_neighbour_fits_better"] == [], got["the_clip_row"]
    return (f"a clip of another size is refused ({said.splitlines()[0][:90]}...); a one-row band at a source-size clip's bottom, cut by the "
            f"loader's fit, shows in the stack's clip row (grey {int(edge.min())}+)")


# ---------------------------------------------------------------- a source larger than the canvas (the blind read of 2026-10-16)

BW, BH = 288, 224           # a source larger than the 256x192 canvas: the loader's crop is 288x216 from row 4 (`assemble_delivery.crop_of`)
SHIFT = 3                   # levels added to U and V on every frame of the big clip: under CHANGE, so the assembler's own test calls it unchanged
BOX = (60, 60, 80, 60)      # x, y, w, h of a red box drawn on the big clip on source frames 2-7: a real change
ZOOM_BOX = [64, 48, 192, 144]   # pa's zoom box at the canvas; at the source (72, 58, 216, 166)
LONG = 10                   # gap runs on source frame 10: more chips than one line of a 288-wide legend holds


class Big:
    """A source 288x224, larger than the canvas; a clip of it (lossless) with U and V moved by SHIFT on every frame and a red box
    on 2-7; the same two pieces, pa with a zoom record and pb with a zoom record that does not read and a mask picture of one
    frame (its frames 9 and 10 have none); a queued record for pa with its day; and a per-frame gap check's runs."""
    def __init__(self, m: Made):
        import shutil
        self.root = m.root / "big"
        self.root.mkdir(exist_ok=True)
        self.source, self.clip = self.root / "source.mp4", self.root / "clip.mp4"
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={BW}x{BH}:rate=24", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
               "-frames:v", str(FRAMES), "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", str(self.source))
        x, y, w, h = BOX
        ffmpeg("-i", str(self.source), "-vf", f"lutyuv=u=val+{SHIFT}:v=val+{SHIFT},drawbox=x={x}:y={y}:w={w}:h={h}:color=red:t=fill:enable='between(n,2,7)'",
               "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-c:a", "copy", str(self.clip))
        for piece in (m.a, m.b):
            for f in (piece, piece.with_suffix(".png"), piece.with_name(piece.stem + "_with_mask.mp4")):
                shutil.copy(f, self.root / f.name)
        self.a, self.b = self.root / m.a.name, self.root / m.b.name
        flat(self.root / "pb_00001_with_mask.mp4", f"{W}x{H * 2}", 1, "40+30*N")           # one frame of three: 9 and 10 have none
        (self.root / "pa_00001_zoomed.json").write_text(json.dumps({"file": "h3 zoomed piece", "run": "pa_00001", "fixed": True, "canvas": [W, H],
                                                                    "source_size": [W, H], "frames": 6, "box": ZOOM_BOX, "zoom": 2.0}))
        flat(self.root / "pa_00001_zoomed.mp4", f"{W}x{H}", 6, "128")
        (self.root / "pb_00001_zoomed.json").write_text(json.dumps({"file": "not a zoom record"}))
        self.table = self.root / "t.txt"
        self.table.write_text(f"2-7 {self.a} 2 subject=a\n8-10 {self.b} 8 subject=b\n")
        self.data = self.root / "data"
        job = self.data / "2026-10-14_pa_job"
        job.mkdir(parents=True, exist_ok=True)
        (job / "pa.render.queued.json").write_text(json.dumps({"files": ["big/pa_00001.mp4"], "queued_at": "2026-10-14T09:00:00-05:00"}))
        self.coverage = self.root / "coverage.json"
        runs = [{"first": 3, "last": 4, "subject": "a", "kind": "original_while_tracked", "cause": "track 900 px"},
                {"first": 9, "last": 9, "subject": "b", "kind": "original_tracking_not_known", "cause": "no tied capture covers this frame"}]
        runs += [{"first": 10, "last": 10, "subject": "b", "kind": "seam", "cause": f"seam number {i} between two pieces"} for i in range(LONG)]
        self.coverage.write_text(json.dumps({"runs": runs}))
        self.out = self.root / "stacked.mp4"

    def build(self, more_inputs: list | None = None, **more) -> None:
        import what_h3_got
        given = dict(source=str(self.source), table=str(self.table), joined=str(self.clip), out=str(self.out), work=str(self.root / "work"),
                     tag="", alias=[], copy_rate=24.0, only=None, first=0, data=str(self.data), coverage=str(self.coverage))
        given.update(more)
        # what pa's load was given: a reference with no points record, so it has no head crop (and, where a case hands them, more inputs)
        held = what_h3_got.read
        what_h3_got.read = lambda folder, load, stage, output=None: {"inputs": [
            {"tag": "<Video 1>", "role": "the reference", "file": str(self.a), "record": {"its_first_frame_is_source_frame": 0},
             "loader": {"custom": [W, H]}, "reference_frames": 11}] + list(more_inputs or [])}
        try:
            sd.build(argparse.Namespace(**given))
        finally:
            what_h3_got.read = held


BIG: list[Big] = []


def big() -> Big:
    if not BIG:
        BIG.append(Big(made()))
    return BIG[0]


def built_big() -> Big:
    b = big()
    if not b.out.is_file():
        b.build()
    return b


def raw(path: Path, pix: str, w: int, h: int, crop: str = "") -> list[bytes]:
    """Every frame of a file as raw bytes of `pix`, unscaled (or one crop of it)."""
    got = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), *(["-vf", crop] if crop else []), "-f", "rawvideo", "-pix_fmt", pix, "-"],
                         capture_output=True, check=True).stdout
    n = w * h * (3 if pix == "rgb24" else 1) if pix != "yuv420p" else w * h * 3 // 2
    return [got[i * n:(i + 1) * n] for i in range(len(got) // n)]


def the_assemblers_own_test() -> str:
    """Q1 of the blind read: the middle row marked a change by the largest of R, G and B, where the assembler takes the largest
    of Y, U and V (`assemble_delivery.difference`). On the big clip every frame is the source with U and V moved by SHIFT,
    which the assembler's test calls unchanged, and 2-7 also carry a red box. The row's count of changed pixels, on every
    frame, is the assembler's own `changed` on the two files read as yuv420p; on the frames without the box it is 0. The
    control: the RGB test the row used marks pixels on those same frames, so the fixture tells the two apart."""
    import cv2
    b = built_big()
    asm = sd._assembler()
    src, clip = raw(b.source, "yuv420p", BW, BH), raw(b.clip, "yuv420p", BW, BH)
    srgb, crgb = raw(b.source, "rgb24", BW, BH), raw(b.clip, "rgb24", BW, BH)
    recs = frames_of(b.out)
    for f in range(FRAMES):
        own = int(asm.changed(asm.planes(clip[f], BW, BH), asm.planes(src[f], BW, BH)).sum())
        assert recs[f]["changed_px"] == own, f"frame {f}: the middle row marks {recs[f]['changed_px']} px changed, the assembler's test {own}"
        if not 2 <= f <= 7:
            assert own == 0, f"the fixture: frame {f}, with U and V moved by {SHIFT}, is changed to the assembler's test ({own} px)"
    s = np.frombuffer(srgb[0], np.uint8).reshape(BH, BW, 3).astype(np.int16)
    j = np.frombuffer(crgb[0], np.uint8).reshape(BH, BW, 3).astype(np.int16)
    d = cv2.blur(np.abs(s - j).max(axis=2).astype(np.float32), (asm.CHANGE_MEAN, asm.CHANGE_MEAN))
    rgb_says = int(cv2.morphologyEx((d > asm.CHANGE).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).sum())
    assert rgb_says > 0, "the control: the RGB test marks nothing on frame 0 either, so the fixture cannot tell the two tests apart"
    return (f"on all {FRAMES} frames the row's count is the assembler's `changed` on yuv420p; frame 0 (U, V moved by {SHIFT}): 0 px, where the "
            f"RGB test marks {rgb_says}; box frames {recs[2]['changed_px']} px")


def nothing_drawn_on_the_picture() -> str:
    """The owner's spec: the legends never cover the picture. On the big stack, with a zoomed piece and a subject shown as the
    original while tracked (frames 3-4): the top row is the source and the bottom the clip in every row, the label corner and
    the edge rows (0-9 and the last 10) included; the middle row is the source dimmed with the cut where the assembler's test
    says it changed, edged white, everywhere except a 1 px zoom outline kept off the edge rows and columns. Each row's pixels
    are held to the picture the build should have laid, decoded as the stack is."""
    import cv2
    b = built_big()
    asm = sd._assembler()
    head = head_of(b.out)
    l1, l2 = legends_of(head)
    total_w = sd.probe(str(b.out))[0]
    stack = raw(b.out, "rgb24", total_w, 3 * BH + l1 + l2)
    src, clip = raw(b.source, "yuv420p", BW, BH), raw(b.clip, "yuv420p", BW, BH)
    srgb, crgb = raw(b.source, "rgb24", BW, BH), raw(b.clip, "rgb24", BW, BH)
    zb = [72, 58, 216, 166]                                   # pa's zoom box at the source, by `assemble_delivery.box_at_the_source`
    outline = np.zeros((BH, BW), bool)
    outline[zb[1] - 2:zb[3] + 2, zb[0] - 2:zb[2] + 2] = True
    outline[zb[1] + 3:zb[3] - 3, zb[0] + 3:zb[2] - 3] = False
    worst = {}
    for f in range(FRAMES):
        frame = np.frombuffer(stack[f], np.uint8).reshape(3 * BH + l1 + l2, total_w, 3).astype(np.int16)
        s = np.frombuffer(srgb[f], np.uint8).reshape(BH, BW, 3).astype(np.int16)
        j = np.frombuffer(crgb[f], np.uint8).reshape(BH, BW, 3).astype(np.int16)
        hard = asm.changed(asm.planes(clip[f], BW, BH), asm.planes(src[f], BW, BH)).astype(np.uint8)
        mid = (s * 0.4).astype(np.int16)
        mid[hard.astype(bool)] = j[hard.astype(bool)]
        edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        mid[edge.astype(bool)] = 255
        # the stack's own encode smears colour a pixel or two either side of the white edge round the red box (the same chroma
        # subsampling as at a legend); those pixels, and the zoom outline's, are not read
        skip = cv2.dilate(edge, np.ones((5, 5), np.uint8)).astype(bool) | (outline if 2 <= f <= 7 else False)
        for name, y, want, skip in (("ORIGINAL", 0, s, None), ("WHAT WAS CHANGED", BH + l1, mid, skip),
                                    ("THE CUT", 2 * BH + l1 + l2, j, None)):
            off = np.abs(frame[y:y + BH, :BW] - want).max(axis=2)
            if skip is not None:
                off[skip] = 0
            # the edge rows and the label corner: a row's mean off the picture, and the share of pixels far off it. The stack's own
            # yuv420p encode smears colour over the two rows of a picture that touch a legend (measured on this fixture, 2026-10-16:
            # up to 14.8 levels on those two, 6.6 at most on the rows past them, with nothing drawn), so rows 2-9 and the last 10 but 2 are read
            edge_rows = np.concatenate([off[2:10], off[-10:-2]]).mean(axis=1).max()
            corner = off[:40, :120].mean()
            far = float((off > 60).mean())
            worst[name] = max(worst.get(name, (0, 0, 0)), (round(float(edge_rows), 1), round(float(corner), 1), round(far, 4)))
            assert edge_rows < 12 and corner < 12 and far < 0.004, (f"frame {f}, the {name} row: something is drawn over the picture: an edge row "
                                                                    f"{edge_rows:.1f} levels off on average, the label corner {corner:.1f}, {far:.2%} of pixels over 60 off")
    return "worst (edge row mean, label corner mean, share over 60 levels) by row: " + "; ".join(f"{k} {v}" for k, v in worst.items())


def every_chip_said() -> str:
    """Chips are facts read from records, none dropped or cut without a word: a kind of the gap check's with no words stops the build
    naming it; `original_tracking_not_known` (the most common kind on 2026-10-16, and dropped until then) is drawn; LONG chips on
    one frame wrap to more lines, each drawn whole inside the source's width; a check flag is shown with its level."""
    b = built_big()
    recs = frames_of(b.out)
    assert "b: TRACKING NOT KNOWN (no tied capture covers this frame)" in recs[9]["chips"], recs[9]["chips"]
    drawn = recs[10]["chips_drawn"]
    assert [c["words"] for c in drawn] == recs[10]["chips"] and len(drawn) == LONG, (drawn, recs[10]["chips"])
    assert all(0 <= c["box"][0] and c["box"][2] <= BW for c in drawn), [c["box"] for c in drawn]
    lines = sorted({c["box"][1] for c in drawn})
    l2 = legends_of(head_of(b.out))[1]
    assert len(lines) > 1 and all(c["box"][3] <= l2 for c in drawn), (lines, l2)
    # an unknown kind stops the build, naming it
    odd = b.root / "odd_coverage.json"
    odd.write_text(json.dumps({"runs": [{"first": 3, "last": 3, "subject": "a", "kind": "a_kind_nobody_named", "cause": ""}]}))
    said = refused(lambda: b.build(coverage=str(odd), out=str(b.root / "never.mp4")), "a_kind_nobody_named")
    assert not (b.root / "never.mp4").exists()
    # a check flag with its level
    check = Path(str(b.clip) + ".check.json")
    check.write_text(json.dumps({"flags": [{"id": "d009", "rule": "jump", "level": "iffy", "piece": "pa_00001.mp4", "subject": "a",
                                            "source_frames": [[5, 5]], "why": "the changed area steps"}]}))
    flagged = b.root / "flagged.mp4"
    try:
        b.build(out=str(flagged))
    finally:
        check.unlink()
    assert "d009 jump (iffy)" in [c["words"] for c in frames_of(flagged)[5]["chips_drawn"]], frames_of(flagged)[5]["chips_drawn"]
    return f"tracking not known drawn; {LONG} chips on frame 10 on {len(lines)} lines inside {BW} px; an unknown kind refused ({said[:70]}...); a flag with its level"


def who_is_laid_from_the_table() -> str:
    """Q3 of the blind read: the legend said ORIGINAL for a laid subject whose panel did not read. pb lays 8-10, and its mask
    picture has one frame: on 9 and 10 the legend and the records still name pb, and its panel is said as not read, with why."""
    b = built_big()
    recs = frames_of(b.out)
    for f in (9, 10):
        col = recs[f]["columns"]["b"]
        assert col is not None and col["piece"] == "pb" and col["mask_frame"] is None and "not read" in col["mask_not_read"], (f, col)
        assert recs[f]["legend"]["b"].startswith("b: pb"), recs[f]["legend"]
        assert "what H3 saw: not read" in col["label"] and col["label"].startswith("b: pb"), col["label"]
    assert recs[8]["columns"]["b"]["mask_frame"] == 0 and recs[8]["columns"]["b"].get("mask_not_read") is None, recs[8]["columns"]["b"]
    assert recs[1]["legend"]["a"] == "a: ORIGINAL" and recs[1]["columns"]["a"] is None, recs[1]
    return f"on 9 and 10 the legend names pb and its panel says {recs[9]['columns']['b']['mask_not_read']!r}"


def the_day_and_the_head_crop() -> str:
    """The render's day is read from the queued record that names the piece (`queued_at`), never the file's time; a piece no record
    names says "day not recorded". A reference with no points has no head crop, and its block says so (the test was always true)."""
    b = built_big()
    head, recs = head_of(b.out), frames_of(b.out)
    assert list(head["given"]) == ["pa_00001.mp4"], list(head["given"])
    # THE STACK'S OWN SIDECAR, read back (the sidecar's second read, finding G): the roles it names and the stack's bytes. Only
    # `sidecar.read(video)` and its `inputs` roles are used; the rest of its form is another session's to change
    import hashlib
    import sidecar as sidecar_tool
    side = sidecar_tool.read(b.out)
    roles = {i["role"]: i["path"] for i in side["inputs"]}
    for role, path in (("the source", b.source), ("the joined clip", b.clip), ("the table", b.table), ("the per-frame gap check", b.coverage)):
        assert role in roles and Path(roles[role]).resolve() == Path(path).resolve(), (role, roles.get(role))
    assert side["video_sha256"] == hashlib.sha256(b.out.read_bytes()).hexdigest(), "the sidecar is not of the stack's bytes"
    assert recs[4]["legend"]["a"] == "a: pa (10/14)", recs[4]["legend"]
    assert recs[9]["legend"]["b"] == "b: pb (day not recorded)", recs[9]["legend"]
    assert "(no points: no head crop)" in recs[4]["columns"]["a"]["label"], recs[4]["columns"]["a"]["label"]
    # the zoom records: pa's read and keyed by its file, pb's refused and kept with its reason
    assert head["zooms"]["pa_00001.mp4"]["box"] == [72, 58, 216, 166], head["zooms"]
    assert "pb_00001.mp4" in head["zoom_not_read"] and "not a zoom record" in head["zoom_not_read"]["pb_00001.mp4"], head.get("zoom_not_read")
    return f"pa (10/14), pb (day not recorded); pa's block: {recs[4]['columns']['a']['label'][-60:]!r}; pb's zoom record not read, with its reason"


def check_reads_the_canvas() -> str:
    """`check` reports the canvas the pieces are (256x192) and the source's size (288x224) apart, and holds the picture's width to
    the source's and the right column's as its metadata says: a file whose metadata says another width is refused. An odd source
    is refused before anything is encoded."""
    b = built_big()
    got = sd.read_back(str(b.source), str(b.clip), str(b.out))
    assert got["canvas"] == [W, H] and got["source_size"] == [BW, BH], (got["canvas"], got.get("source_size"))
    said = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "csv=p=0", str(b.out)], capture_output=True, text=True).stdout.strip()
    wrong = b.root / "wrong_width.mp4"
    ffmpeg("-i", str(b.out), "-c", "copy", "-metadata", "comment=" + said.replace(f"right={W}", f"right={W + 64}"), str(wrong))
    refused(lambda: sd.read_back(str(b.source), str(b.clip), str(wrong)), "wide")
    odd = b.root / "odd.mp4"
    ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={BW}x{BH + 1}:r=24", "-frames:v", str(FRAMES), "-pix_fmt", "yuv444p", "-c:v", "libx264", "-qp", "0", str(odd))
    refused(lambda: b.build(source=str(odd), joined=str(odd), out=str(b.root / "odd_stacked.mp4"), coverage=""), "odd")
    assert not (b.root / "odd_stacked.mp4").exists() and not (b.root / "work" / "odd_stacked.video.mp4").exists()
    return f"canvas {got['canvas']}, source {got['source_size']}; a width its metadata does not give is refused; an odd source refused up front"


# ---------------------------------------------------------------- the second blind read of 2026-10-16 (review2_stack_view.md)

def barred() -> Path:
    """The big stack built with a check record beside the clip: d005 `piece_changes_nothing` on pb's frame 8."""
    b = built_big()
    out = b.root / "barred.mp4"
    if not out.is_file():
        check = Path(str(b.clip) + ".check.json")
        check.write_text(json.dumps({"flags": [{"id": "d005", "rule": "piece_changes_nothing", "level": "iffy", "piece": "pb_00001.mp4", "subject": "b",
                                                "source_frames": [[8, 8]], "why": "pb changes nothing on 8"}]}))
        try:
            b.build(out=str(out))
        finally:
            check.unlink()
    return out


def what_was_not_read_is_said() -> str:
    """Finding 1: with no `--coverage` and no `<clip>.check.json`, legend 2 said "no gap or flag on this frame" about gaps nobody
    checked. Now it says "gaps not checked: no --coverage" and "check flags not read: <why>", and the plain "no gap or flag" only
    where both were read. Finding 9: a subject the table leaves as the original on purpose gets its words in legend 1."""
    m = made()
    if not m.out.is_file():
        m.build()
    plain = frames_of(m.out)
    for f in range(FRAMES):
        two = plain[f]["legend_2"]
        assert "gaps not checked: no --coverage" in two and any(t.startswith("check flags not read: ") for t in two), (f, two)
        assert "no gap or flag on this frame" not in two, (f, two)
    assert "a: ORIGINAL on purpose (not in the shot)" in plain[0]["legend_1"], plain[0]["legend_1"]
    assert "b: ORIGINAL on purpose (turned away)" in plain[11]["legend_1"], plain[11]["legend_1"]
    both = frames_of(barred())
    assert both[0]["legend_2"][0] == "no gap or flag on this frame", both[0]["legend_2"]
    return f"unread: {plain[0]['legend_2'][:2]}; both read, nothing on frame 0: {both[0]['legend_2'][0]!r}; intents in legend 1"


def the_bar_off_what_h3_saw() -> str:
    """Finding 4: the `piece_changes_nothing` bar was painted over the bottom of the mask picture (the node's key line). Now it is a
    line of the block's label strip; the strip holds every line of the label, wrapped inside the block, above the mask picture."""
    out = barred()
    recs, head = frames_of(out), head_of(out)
    tile = head["right_tiles"][head["columns"].index("b")]          # b's block
    x, y, w, h = tile["mask"]
    l1, l2 = legends_of(head)
    total_w = sd.probe(str(out))[0]
    frame = np.frombuffer(raw(out, "rgb24", total_w, 3 * BH + l1 + l2)[8], np.uint8).reshape(3 * BH + l1 + l2, total_w, 3).astype(int)
    bottom = frame[y + h - 8:y + h - 2, BW + x + w // 4:BW + x + 3 * w // 4]
    redness = float((bottom[..., 0] - bottom[..., 1]).mean())
    assert redness < 20, f"frame 8: the bottom of b's mask picture is {redness:.0f} levels redder than green: a bar is drawn over what H3 saw"
    col = recs[8]["columns"]["b"]
    assert any("d005: pb CHANGES 0 PX HERE" in t for t in col["label_lines"]), col["label_lines"]
    bx, by, bw, bh = tile["block"]
    for box in col["label_boxes"]:
        assert bx <= box[0] and box[2] <= bx + bw and by <= box[1] and box[3] <= y, (box, tile)
    assert len(col["label_lines"]) > 1, col["label_lines"]
    return f"no bar over b's mask picture (red less green {redness:.1f}); its label on {len(col['label_lines'])} lines, all above the picture and inside the block"


def a_trial_of_some_frames_reads_back() -> str:
    """Finding 3: `check` held the clip row to the clip from its frame 0 and the original row to `--first`, so a stack built with
    `--only` read back wrong. Now it takes the frames from the stack's own record. Finding 9: no label is drawn on the rows, so
    nothing is cut off the top before the rows are compared, and the printed lines do not speak of a label."""
    m = made()
    out = m.root / "only.mp4"
    m.build(out=str(out), only="3-8")
    got = sd.read_back(str(m.source), str(m.clip), str(out))
    for key in ("the_original_row", "the_clip_row"):
        assert got[key]["a_neighbour_fits_better"] == [] and got[key]["own_frame_fits_best"] >= 4, (key, got[key])
    assert got["frames_stacked"] == [3, 8] and got["the_original_row"]["cut"] == 0, (got.get("frames_stacked"), got["the_original_row"])
    assert not any("label" in line for line in sd.check_lines(got, 0)), sd.check_lines(got, 0)
    return f"frames 3-8 of the clip: each row fits its own frame on {got['the_original_row']['own_frame_fits_best']} and {got['the_clip_row']['own_frame_fits_best']} of 6"


def the_mixed_build() -> tuple[Big, Path]:
    """The big fixture with a third piece, pc, of ANOTHER canvas (320x192): its mask picture carries a bright band on its right 32
    columns, its zoom record is of its own canvas, and its saved graph names a reference that is refused (a URL)."""
    b = built_big()
    out = b.root / "mixed.mp4"
    if not out.is_file():
        pc = b.root / "pc_00001.mp4"
        flat(pc, "320x192", 4, "128")
        ffmpeg("-f", "lavfi", "-i", "nullsrc=size=320x384:rate=24,format=gray,geq=lum='if(gte(X\\,288)\\,250\\,60)'", "-frames:v", "4",
               "-pix_fmt", "yuv420p", str(b.root / "pc_00001_with_mask.mp4"))
        picture(b.root / "pc_00001.png", 3, "http://example.invalid/ref.mp4")
        (b.root / "pc_00001_zoomed.json").write_text(json.dumps({"file": "h3 zoomed piece", "run": "pc_00001", "fixed": True, "canvas": [320, 192],
                                                                "source_size": [320, 192], "frames": 4, "box": ZOOM_BOX, "zoom": 2.0}))
        flat(b.root / "pc_00001_zoomed.mp4", "320x192", 4, "128")
        table = b.root / "t_mixed.txt"
        table.write_text(f"2-7 {b.a} 2 subject=a\n8-10 {b.b} 8 subject=b\n3-6 {pc} 3 subject=c\n")
        b.build(table=str(table), out=str(out), coverage="")
    return b, out


def each_piece_at_its_own_canvas() -> str:
    """Finding 2: every mask picture was cut, and every zoom box placed, at the FIRST row's canvas. A piece of another canvas (pc,
    320x192, beside two of 256x192) is read at its own: its whole mask picture (the band on its right 32 columns) shows in its tile,
    and its zoom box is placed through the loader's crop of ITS canvas, as the assembler lays it."""
    b, out = the_mixed_build()
    head = head_of(out)
    asm = sd._assembler()
    crop = asm.crop_of(BW, BH, 320, 192)
    inside, _ = asm.box_at_the_source(ZOOM_BOX, crop, (320, 192))
    want = [inside[0] + crop[2], inside[1] + crop[3], inside[2] + crop[2], inside[3] + crop[3]]
    assert head["zooms"]["pc_00001.mp4"]["box"] == want, (head["zooms"]["pc_00001.mp4"]["box"], want)
    l1, l2 = legends_of(head)
    total_w = sd.probe(str(out))[0]
    x, y, w, h = head["right_tiles"][head["columns"].index("c")]["mask"]
    grey_ = np.frombuffer(raw(out, "rgb24", total_w, 3 * BH + l1 + l2)[4], np.uint8).reshape(3 * BH + l1 + l2, total_w, 3).astype(float).mean(axis=2)
    tile = grey_[y + h // 4:y + 3 * h // 4, BW + x:BW + x + w]
    band, rest = tile[:, -max(2, w // 20):].mean(), tile[:, w // 4:w // 2].mean()
    assert band > 180 and rest < 110, f"pc's tile: its right edge reads {band:.0f} (its band was made 250), its middle {rest:.0f} (made 60)"
    return f"pc's zoom box {want} through its own crop; its tile shows its band ({band:.0f}) beside its ground ({rest:.0f})"


def a_refused_reference_said() -> str:
    """Finding 5: a reference the tool refused to open was labelled "no reference wired". Now the label says it was not opened, and why."""
    b, out = the_mixed_build()
    label = frames_of(out)[4]["columns"]["c"]["label"]
    assert "no reference wired" not in label and "NOT OPENED" in label and "not a path on this machine" in label, label
    return f"pc's label: {label[label.index('NOT OPENED') - 15:][:90]!r}"


def one_a_second_beside_the_stack_only_when_whole() -> str:
    """Finding 6: the one-a-second parts were written straight beside `--out`, and stayed there when the main encoder failed. Now they
    are encoded under `--work` and moved beside `--out` only after both encoders succeed."""
    m = made()
    out = m.root / "slow" / "slow.mp4"
    out.parent.mkdir(exist_ok=True)
    work = m.root / "slow_work"
    work.mkdir(exist_ok=True)
    blocked = work / "slow.video.mp4"
    blocked.mkdir(exist_ok=True)                                      # the main encoder cannot write its working file
    refused(lambda: m.build(out=str(out), work=str(work), one_a_second=True), "encoder")
    left = sorted(p.name for p in out.parent.iterdir())
    assert left == [], f"a failed build left beside --out: {left}"
    blocked.rmdir()
    m.build(out=str(out), work=str(work), one_a_second=True)
    beside = sorted(p.name for p in out.parent.glob("slow.1fps.*"))
    assert beside and not list(work.glob("*.1fps.*")), (beside, sorted(p.name for p in work.iterdir()))
    return f"a failed main encode leaves nothing beside --out; a whole build moves {beside} there"


def no_stack_without_its_sidecar() -> str:
    """Finding 7: the sidecar was written last and only SystemExit from its reader was caught: a refusal or error left the stack on the
    share with no sidecar. Now the sidecar is written before success is said, and any failure of it is a hard failure naming the stack."""
    import contextlib
    import io
    import sidecar as sidecar_tool
    m = made()
    out = m.root / "nosidecar.mp4"
    held = sidecar_tool.write

    def broken(*_a, **_k):
        raise RuntimeError("another file holds the sidecar's name")
    said = io.StringIO()
    sidecar_tool.write = broken
    try:
        with contextlib.redirect_stdout(said):
            words = refused(lambda: m.build(out=str(out)), str(out), "sidecar", "another file holds")
    finally:
        sidecar_tool.write = held
    assert "written " not in said.getvalue(), said.getvalue()[-400:]
    return f"refused: {words[:120]}..."


# ---------------------------------------------------------------- the squares kept as the original (the owner's go, 2026-10-16)

TRACK_BOX = (64, 48, 160, 144)      # x0, y0, x1, y1 of pa's subject's mask and of her track, on the 256x192 canvas
MARGIN = 32                         # the margin the region file says the node grew her mask by
INSIDE_UNIT, OUTSIDE_UNIT = (2, 3), (0, 1)    # token (row, column) the fixture takes out of the grown region: in her track, and outside it
UNIT = 32                           # a whole token's side on the 256x192 canvas: two 16-px cells


def squares_variant(name: str, empty_frames=(), tracks=None, more_windows=(), custom=(W, H)) -> tuple[Big, Path]:
    """The big fixture with a region file for pa, written by the node's own `video_mask.save_window_region`: her mask is TRACK_BOX
    (left empty on `empty_frames` of the window), grown by MARGIN, and two of the grown region's tokens are kept as the original
    (taken out of `tokens`), one inside her track and one outside it. `more_windows` are more region files beside it, as
    (file name, first_frame, trim). Her track or tracks are handed as the load's "the subject's mask", each a list of
    (box, ffmpeg `enable` expression or None) drawn white on black, loaded at `custom`. pb has no region files."""
    import capture_masked_run as capm
    import shutil
    import torch
    b = built_big()
    out = b.root / f"{name}.mp4"
    if out.is_file():
        return b, out
    vm = capm._pack("video_mask")
    lt = 3
    frames = sum(vm.run_lengths(lt))
    mask = torch.zeros(frames, H, W)
    x0, y0, x1, y1 = TRACK_BOX
    mask[:, y0:y1, x0:x1] = 1.0
    for k in empty_frames:
        mask[k] = 0.0
    tokens = vm.token_mask(vm.grow(mask, MARGIN), lt, H // 16, W // 16, True)
    for r, c in (INSIDE_UNIT, OUTSIDE_UNIT):
        tokens[:, 2 * r:2 * r + 2, 2 * c:2 * c + 2] = 0.0
    windows = b.root / "pa_windows"
    shutil.rmtree(windows, ignore_errors=True)
    windows.mkdir()
    for file, first, trim in (("pa_window_1_region.npz", 0, 0),) + tuple(more_windows):
        vm.save_window_region(str(windows / file), mask, tokens, {"edge": vm.EDGE_TOKENS, "composite": "whole region"}, MARGIN, first, trim)
    inputs = []
    for i, boxes in enumerate(tracks or [[(TRACK_BOX, None)]]):
        track = b.root / "capture" / "subjects" / "a" / f"track__{name}_{i}.mkv"
        track.parent.mkdir(parents=True, exist_ok=True)
        draw = ",".join(f"drawbox=x={bx[0]}:y={bx[1]}:w={bx[2] - bx[0]}:h={bx[3] - bx[1]}:color=white:t=fill" + (f":enable='{on}'" if on else "")
                        for bx, on in boxes)
        ffmpeg("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=24", "-frames:v", str(FRAMES), "-vf", draw, "-c:v", "libx264", "-qp", "0",
               "-pix_fmt", "yuv420p", str(track))
        inputs.append({"role": "the subject's mask", "file": str(track), "loader": {"first_frame": 0, "custom": list(custom)}})
    try:
        b.build(out=str(out), coverage="", more_inputs=inputs)
    finally:
        shutil.rmtree(windows, ignore_errors=True)
    return b, out


def the_squares_build() -> tuple[Big, Path]:
    """The squares fixture as `squares_variant` makes it by default: one window, her track TRACK_BOX on every frame."""
    return squares_variant("squares")


def stack_rgb(out: Path, index: int) -> np.ndarray:
    """One frame of a built big stack as [h, w, 3] float."""
    l1, l2 = legends_of(head_of(out))
    total_w = sd.probe(str(out))[0]
    return np.frombuffer(raw(out, "rgb24", total_w, 3 * BH + l1 + l2)[index], np.uint8).reshape(3 * BH + l1 + l2, total_w, 3).astype(float)


def unit_colour(out: Path, index: int, rc) -> np.ndarray:
    """The mean colour of the middle half of one canvas token in pa's tile (column a) on stacked frame `index`."""
    head = head_of(out)
    x, y, w, h = head["right_tiles"][head["columns"].index("a")]["mask"]
    k, (r, c) = w / W, rc
    frame = stack_rgb(out, index)
    return frame[round(y + (r + 0.25) * UNIT * k):round(y + (r + 0.75) * UNIT * k),
                 round(BW + x + (c + 0.25) * UNIT * k):round(BW + x + (c + 0.75) * UNIT * k)].reshape(-1, 3).mean(axis=0)


def yellow(colour: np.ndarray) -> bool:
    return bool(colour[0] > 180 and colour[1] > 160 and colour[2] < 110)


def her_track_over_the_whole_step() -> str:
    """B1 of the outside review (2026-10-16): a unit is inside her track when it is more than `OWN_TRACK_CELL_SHARE` hers on ANY
    frame of its latent step, as `capture_masked_run.grid_carve` and `node_taken_inside` count it; the panel tested the one
    frame shown. Here her track covers the inside token only from load frame 3 on: on source frame 3 (load frame 1, in the step of
    load frames 1-4) the token is inside her track, yellow and counted."""
    b, out = squares_variant("step", tracks=[[((64, 100, 160, 144), "lt(n,3)"), (TRACK_BOX, "gte(n,3)")]])
    col = frames_of(out)[3]["columns"]["a"]
    colour = unit_colour(out, 3, INSIDE_UNIT)
    assert col["kept_squares"].get("inside") == 1 and yellow(colour), (col["kept_squares"], colour.round().tolist())
    return f"source frame 3: {col['kept_squares']['inside']} inside, drawn {colour.round().tolist()}"


def a_pair_load_takes_both_tracks() -> str:
    """B2: a load of two subjects is handed both tracks ("the subject's mask" twice); a unit is inside when it is inside their union.
    The inside token lies in the second track only."""
    b, out = squares_variant("pair", tracks=[[((64, 48, 96, 144), None)], [((96, 48, 160, 144), None)]])
    col = frames_of(out)[4]["columns"]["a"]
    colour = unit_colour(out, 4, INSIDE_UNIT)
    assert col["kept_squares"].get("inside") == 1 and yellow(colour), (col["kept_squares"], colour.round().tolist())
    return f"frame 4: {col['kept_squares']['inside']} inside the union of two tracks, drawn {colour.round().tolist()}"


def a_window_of_another_run_left_unread() -> str:
    """B3: the windows are read as the capture reads them (`read_saved_regions`): each starts laying where the one before ended
    (`first_frame + trim`), and files after the one that reaches the render's end are another run's, left unread and named. A
    leftover window file over the same frames does not make two counts, nor a second read of one track frame."""
    b, out = squares_variant("leftover", more_windows=(("pa_window_2_region.npz", 0, 0),))
    col = frames_of(out)[4]["columns"]["a"]
    read = head_of(out)["kept_squares_read"]["pa_00001.mp4"]
    assert col["kept_squares"].get("inside") == 1 and col["kept_squares"].get("window") == "pa_window_1_region.npz", col["kept_squares"]
    assert read.get("left_unread") == ["pa_window_2_region.npz"], read
    return f"frame 4 from pa_window_1 alone ({col['kept_squares']['inside']} inside); pa_window_2 left unread and named"


def an_unlaid_frame_draws_no_squares() -> str:
    """B4: on a frame the piece does not lay (its mask empty there, or the cut gate shut), the composite shows the source, so no
    square is drawn or counted as kept. Here the mask is empty on window frame 3 (source frame 5), inside a step whose other frames
    carry it."""
    b, out = squares_variant("unlaid", empty_frames=(3,))
    col = frames_of(out)[5]["columns"]["a"]
    colour = unit_colour(out, 5, INSIDE_UNIT)
    assert not yellow(colour) and not col["kept_squares"].get("inside") and col["kept_squares"].get("not_laid"), (col["kept_squares"], colour.round().tolist())
    assert frames_of(out)[4]["columns"]["a"]["kept_squares"].get("inside") == 1, "the control: the frame before, laid, has its square"
    return f"source frame 5: {col['kept_squares']['not_laid']!r}; nothing drawn ({colour.round().tolist()})"


def a_track_loaded_at_its_own_size() -> str:
    """B5: a loader that leaves its size to the file (custom 0 by 0) hands the track at the file's own size; the panel reads it there
    and never at 0 by 0."""
    b, out = squares_variant("native", custom=(0, 0))
    col = frames_of(out)[4]["columns"]["a"]
    assert col["kept_squares"].get("inside") == 1 and not col["kept_squares"].get("not_split"), col["kept_squares"]
    return f"frame 4: {col['kept_squares']['inside']} inside, the track read at its own {W} by {H}"


def squares_on_frame_4() -> tuple[dict, dict]:
    """pa's column on source frame 4 of the squares build: its record, and the mean colour of three tokens of its tile (the one kept
    inside her track, the one kept outside it, and one the node regenerated)."""
    b, out = the_squares_build()
    recs, head = frames_of(out), head_of(out)
    l1, l2 = legends_of(head)
    total_w = sd.probe(str(out))[0]
    frame = np.frombuffer(raw(out, "rgb24", total_w, 3 * BH + l1 + l2)[4], np.uint8).reshape(3 * BH + l1 + l2, total_w, 3).astype(float)
    x, y, w, h = head["right_tiles"][head["columns"].index("a")]["mask"]
    k = w / W

    def unit(rc):                                                  # the middle half of one token of the canvas, in the tile
        r, c = rc
        return frame[round(y + (r + 0.25) * UNIT * k):round(y + (r + 0.75) * UNIT * k),
                     round(BW + x + (c + 0.25) * UNIT * k):round(BW + x + (c + 0.75) * UNIT * k)].reshape(-1, 3).mean(axis=0)
    return recs[4]["columns"]["a"], {"inside": unit(INSIDE_UNIT), "outside": unit(OUTSIDE_UNIT), "bare": unit((4, 6))}


def a_square_kept_inside_her_track() -> str:
    """The owner's go of 2026-10-16: each WHAT H3 SAW panel shows the squares the node kept as the original, from the render's own
    region files: the grown region's tokens (`token_mask(grow(mask, margin))`) less the region it regenerated (`tokens`). One
    inside the subject's own track (more than `OWN_TRACK_CELL_SHARE` hers) is bright yellow, with a chip in the label strip."""
    col, at = squares_on_frame_4()
    inside, bare = at["inside"], at["bare"]
    assert inside[0] > 180 and inside[1] > 160 and inside[2] < 110, f"the square kept inside her track is not bright yellow: {inside.round()}"
    assert abs(bare - MASK(2)).max() < NEAR, f"a square the node regenerated is drawn over: {bare.round()} (made {MASK(2)})"
    assert "1 square kept as the original inside a's outline" in col["label_lines"], col["label_lines"]
    assert col["kept_squares"]["inside"] == 1, col["kept_squares"]
    return f"inside {inside.round().tolist()}, yellow, chip {col['label_lines'][-1]!r}; a regenerated square bare {bare.round().tolist()}"


def a_square_kept_outside_her_track() -> str:
    """A square kept as the original outside her track (a fair trim on a neighbour or a prop) is a quiet shade, with no chip."""
    col, at = squares_on_frame_4()
    outside, bare = at["outside"], at["bare"]
    assert outside[2] - outside[0] > 8 and abs(outside - bare).max() > 15 and outside[0] < 150, \
        f"the square kept outside is not the quiet shade: {outside.round()} beside the bare {bare.round()}"
    assert not any("outside" in line for line in col["label_lines"]), col["label_lines"]
    assert col["kept_squares"]["outside"] == 1, col["kept_squares"]
    return f"outside {outside.round().tolist()}, quiet, no chip"


def a_render_with_no_region_files_said() -> str:
    """pb has no region files: its label strip says so, never a silent blank."""
    b, out = the_squares_build()
    said = " ".join(frames_of(out)[9]["columns"]["b"]["label_lines"])          # the strip's lines, as wrapped
    assert "squares kept as the original not shown: no window region files beside the render" in said, said
    return f"pb: {said[said.index('squares kept'):]!r}"


# ---------------------------------------------------------------- a view every player plays: H.264 level 5.1 (the owner's review page, 2026-10-16)

STACK_W = BW + W                    # the big fixture's view is the source's width and the canvas's column: 544, 34 macroblocks across


def with_level_limit(macroblocks: int, fn):
    """`fn()` with the level's macroblock limit lowered to `macroblocks` (the fixture is far smaller than 3840 rows), then put back."""
    held = getattr(sd, "LEVEL_MACROBLOCKS", None)
    sd.LEVEL_MACROBLOCKS = macroblocks
    try:
        return fn()
    finally:
        if held is None:
            del sd.LEVEL_MACROBLOCKS
        else:
            sd.LEVEL_MACROBLOCKS = held


def every_chip_said_within_the_level() -> str:
    """The owner's review page crashed Chrome on 2448x4172 views (level 6.0): legend 2 grew to the busiest of 6395 frames. Every
    view stays within level 5.1's macroblocks, and the chips that do not fit legend 2 go to the right column below what H3 saw,
    each said on its own frame. Here the limit is lowered to 52 macroblock rows (832 px at 544 wide): the busiest frame (LONG
    chips on 10) overflows, every chip of every frame is said in legend 2 or the right column, and the file is tagged level 5.1."""
    b = built_big()
    out = b.root / "level.mp4"
    with_level_limit(34 * 52, lambda: b.build(out=str(out)))
    w_, h_, _ = sd.probe(str(out))
    assert w_ == STACK_W and h_ <= 52 * 16, f"the view is {w_} by {h_}: over the {52 * 16} rows the lowered level allows"
    recs, head = frames_of(out), head_of(out)
    for f, r in recs.items():
        said = [c["words"] for c in r["chips_drawn"]] + [c["words"] for c in r.get("chips_overflow") or []]
        assert sorted(said) == sorted(r["chips"]), (f, said, r["chips"])
    over = recs[10].get("chips_overflow") or []
    assert over, "frame 10's chips all fit legend 2: the fixture does not overflow"
    blocks_end = max(t["block"][1] + t["block"][3] for t in head["right_tiles"])
    for c in over:
        x0, y0, x1, y1 = c["box"]
        assert 0 <= x0 and x1 <= head["right_column"] and blocks_end <= y0 and y1 <= h_, (c, blocks_end, head["right_column"], h_)
    level = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=level", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True).stdout.strip()
    assert level == "51", f"the view is tagged level {level}, not 5.1"
    return f"{w_} by {h_} within {52 * 16}; frame 10: {len(recs[10]['chips_drawn'])} chips in legend 2, {len(over)} in the right column; level {level}"


def three_subjects_and_a_busy_frame_within_the_level() -> str:
    """The whole video of 2026-10-16 (three subjects, a busy frame) was refused by one row: "the right column (1024 by 3828)
    cannot hold its 3 block(s) (3273 rows) and the chips legend 2 cannot hold (556 rows)". The blocks are scaled to the rows the
    overflow leaves, and three blocks each rounded up by a fraction of a row came to one over. Here a table of three subjects
    (pa, pb and pc) with frame 10's LONG chips, the limit lowered to 55 macroblock rows (880 at 608 wide, where the probe of the
    fix found the same one-row refusal): the view builds within it, its three blocks and the overflow both in the right column,
    every chip said."""
    b, _ = the_mixed_build()
    out = b.root / "three.mp4"
    width = BW + 320
    try:
        with_level_limit(-(-width // 16) * 55, lambda: b.build(table=str(b.root / "t_mixed.txt"), out=str(out)))
    except SystemExit as e:
        raise AssertionError(f"refused: {e}") from None
    w_, h_, _ = sd.probe(str(out))
    assert h_ <= 55 * 16, f"the view is {w_} by {h_}, over {55 * 16}"
    recs, head = frames_of(out), head_of(out)
    assert head["columns"] == ["a", "c", "b"], head["columns"]
    blocks_end = max(t["block"][1] + t["block"][3] for t in head["right_tiles"])
    over = recs[10].get("chips_overflow") or []
    assert over and all(blocks_end <= c["box"][1] and c["box"][3] <= h_ for c in over), (blocks_end, over)
    for f, r in recs.items():
        said = [c["words"] for c in r["chips_drawn"]] + [c["words"] for c in r.get("chips_overflow") or []]
        assert sorted(said) == sorted(r["chips"]), (f, said, r["chips"])
    return f"{w_} by {h_} within {55 * 16}: three blocks to row {blocks_end}, frame 10's {len(over)} overflow chips under them; scale {head['right_scale']:.2f}"


def a_layout_over_the_level_refused() -> str:
    """A layout that cannot fit level 5.1 (here the limit lowered below the three picture rows) is refused before anything is
    encoded, naming the sizes; never an unplayable file."""
    b = built_big()
    out = b.root / "never_level.mp4"
    said = refused(lambda: with_level_limit(34 * 40, lambda: b.build(out=str(out))), "level 5.1", str(STACK_W))
    assert not out.exists() and not (b.root / "work" / "never_level.video.mp4").exists()
    return f"refused: {said[:150]}..."


def top_atoms(path: Path) -> list[tuple[str, int, int]]:
    """The top-level boxes of an mp4 as (type, offset, size), read from its header bytes alone."""
    import struct
    out, at, end = [], 0, path.stat().st_size
    with open(path, "rb") as fh:
        while at < end:
            fh.seek(at)
            head = fh.read(16)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head[:8])
            if size == 1:
                size = struct.unpack(">Q", head[8:16])[0]
            elif size == 0:
                size = end - at
            out.append((kind.decode("latin-1"), at, size))
            at += size
    return out


#: How far into a view its index (the `moov` box) may begin for a browser to start it without reading to the end. Reasoned: a
#: player's first range request; the 716 MB whole-video view of 2026-10-16 had it at its end and did not start.
FIRST_BYTES = 4 * 1024 * 1024


def the_view_starts_without_its_end() -> str:
    """The whole-video view (716 MB) would not start in the owner's browser: its index (`moov`) was written at the END of the file.
    A built view, and each one-a-second part, has its `moov` before its media (`mdat`) and within its first FIRST_BYTES; and the
    view's sidecar is of its final bytes (written after the faststart pass)."""
    import sidecar as sidecar_tool
    m = made()
    out = m.root / "fast" / "fast.mp4"
    out.parent.mkdir(exist_ok=True)
    m.build(out=str(out), one_a_second=True)
    for file in [out] + sorted(out.parent.glob("fast.1fps.*.mp4")):
        atoms = top_atoms(file)
        kinds = [k for k, _, _ in atoms]
        moov = next((at for k, at, _ in atoms if k == "moov"), None)
        assert "moov" in kinds and "mdat" in kinds and kinds.index("moov") < kinds.index("mdat") and moov < FIRST_BYTES, \
            f"{file.name}: its boxes are {kinds}: the index is not at the front"
    sidecar_tool.read(out)                                         # refuses unless the sidecar is of the bytes on disk
    return f"moov before mdat in the view and its {len(list(out.parent.glob('fast.1fps.*.mp4')))} one-a-second part(s); the sidecar is of the final bytes"


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


# ---------------------------------------------------------------- candidate-vs-current comparison mode (`compare`, mode (c), 2026-10-17 C71 / C66)

CAND_MASK = lambda n: 60 + 20 * n      # pcand's mask picture grey level on its frame n, distinct from pa's MASK(n) = 40 + 30 * n   # noqa: E731


class CompareFixture:
    """A candidate piece `pcand_00001.mp4` on frames 2-7 for subject a (its mask picture `60+20*N`, wired to `ref_b` from frame 1),
    a candidate cut `cand_span.mp4` of 6 frames (source frames 2-7, `--candidate-first 2`) and `cand_full.mp4` of 12 frames (with a
    distinct visual look so `|candidate - current| >= 4`), and a silent current cut `silent_clip.mp4` to verify audio is muxed from
    `--source` trimmed to the span when `--joined` has no audio stream."""
    def __init__(self, m: Made):
        self.m = m
        self.root = m.root / "cmp"
        self.root.mkdir(exist_ok=True)
        self.pcand = self.root / "pcand_00001.mp4"
        flat(self.pcand, f"{W}x{H}", 6, "160")
        flat(self.pcand.with_name("pcand_00001_with_mask.mp4"), f"{W}x{H * 2}", 6, "60+20*N")
        picture(self.pcand.with_suffix(".png"), 2, m.ref_b, 1)
        self.cand_table = self.root / "t_cand.txt"
        self.cand_table.write_text(f"2-7 {self.pcand} 2 subject=a restore=b:whole\n8-10 {m.b} 8 subject=b\n"
                                   "source 0-1 subject=a not in the shot\nsource 11-11 subject=b turned away\n")
        self.silent_clip = self.root / "silent_clip.mp4"
        ffmpeg("-i", str(m.clip), "-an", "-c:v", "copy", str(self.silent_clip))
        self.cand_full = self.root / "cand_full.mp4"
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=24", "-frames:v", str(FRAMES),
               "-vf", "hue=h=180,vflip", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", "0", str(self.cand_full))
        self.cand_span = self.root / "cand_span.mp4"
        ffmpeg("-i", str(self.cand_full), "-vf", "trim=start_frame=2:end_frame=8,setpts=PTS-STARTPTS",
               "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", "0", str(self.cand_span))
        self.out = self.root / "cmp_span.mp4"

    def compare(self, **more) -> None:
        given = dict(
            source=str(self.m.source), table=str(self.m.table), joined=str(self.silent_clip),
            candidate_joined=str(self.cand_span), candidate_table=str(self.cand_table), candidate_piece="",
            candidate_piece_first=None, out=str(self.out), work=str(self.root / "work"), current_tag="", tag="",
            alias=[], subject="", copy_rate=24.0, only="2-7", first=0, candidate_first=2,
            data=str(self.root / "data"), coverage="", one_a_second=False,
        )
        given.update(more)
        sd.compare(argparse.Namespace(**given))


CMP_FIX: list[CompareFixture] = []


def cmp_fix() -> CompareFixture:
    if not CMP_FIX:
        CMP_FIX.append(CompareFixture(made()))
    return CMP_FIX[0]


def compare_candidate_against_current() -> str:
    """Mode (c) (`C71` / `C66`): compares a candidate cut (`cand_span.mp4`, frames 2-7) against the current cut (`silent_clip.mp4`)
    and `ORIGINAL` (`source.mp4`). Verifies:
    - Left column: three 100%-size rows (`ORIGINAL`, `CURRENT CUT`, `CANDIDATE CUT`), nothing drawn over any of the three rows,
      and `frame 00002`..`frame 00007` plus `"ABOVE: ORIGINAL"` / `"BELOW: CURRENT CUT"` / `"BELOW: CANDIDATE CUT"` in the legends.
    - Right column: two blocks (`current:a` and `candidate:a`), each reading its own `_with_mask.mp4` (`MASK` vs `CAND_MASK`) and
      its own reference frame (`ref_a` from 3 vs `ref_b` from 1), with `"CURRENT (a): WHAT H3 SAW, pa"` and `"CANDIDATE (a): WHAT H3 SAW, pcand"` labels.
    - Audio: trimmed and muxed for frames 2-7 from `--source` even when `--joined` (`silent_clip.mp4`) has no audio stream.
    - Level 5.1, faststart (`moov` before `mdat`), `.mp4.json` sidecar (`sidecar.read`), `<stem>.alignment.txt` with `ALIGNMENT_NOTE`,
      offsets `-2..+2` MAE (`best_offset == 0` for all three rows) and `vs_own < vs_other`, plus shifted/swapped controls."""
    import sidecar as sidecar_tool
    cf = cmp_fix()
    if not cf.out.is_file():
        cf.compare()
    head, recs = head_of(cf.out), frames_of(cf.out)
    assert head["layout"] == "compare" and head["columns"] == ["current:a", "candidate:a"], head
    assert head["frames"] == [2, 7] and head["candidate_begins_on_source_frame"] == 2, head
    l1, l2 = legends_of(head)
    w_, h_, n_ = sd.probe(str(cf.out))
    assert (w_, h_, n_) == (W + head["right_column"], 3 * H + l1 + l2, 6), (w_, h_, n_)
    # Burned-in frame numbers and row headers in legends
    for f in range(2, 8):
        r1, r2 = recs[f]["legend_1"], recs[f]["legend_2"]
        assert r1[0] == "ABOVE: ORIGINAL" and r1[1] == f"frame {f:05d}" and any("BELOW: CURRENT CUT" in s for s in r1), (f, r1)
        assert any("a: current pa" in s and "-> candidate pcand" in s for s in r1), (f, r1)
        assert any("BELOW: CANDIDATE CUT" in s for s in r2), (f, r2)
        assert recs[f]["columns"]["current:a"]["label"].startswith("CURRENT (a): WHAT H3 SAW, pa"), recs[f]["columns"]["current:a"]["label"]
        assert recs[f]["columns"]["candidate:a"]["label"].startswith("CANDIDATE (a): WHAT H3 SAW, pcand"), recs[f]["columns"]["candidate:a"]["label"]
    # Right-column tiles read their own mask and reference grey levels on every frame
    t_cur, t_cand = head["right_tiles"][0], head["right_tiles"][1]
    cur_mask = tile_level(cf.out, t_cur["mask"])
    cand_mask = tile_level(cf.out, t_cand["mask"])
    cur_ref = tile_level(cf.out, t_cur["whole"])
    cand_ref = tile_level(cf.out, t_cand["whole"])
    for idx, f in enumerate(range(2, 8)):
        assert abs(cur_mask[idx] - MASK(f - 2)) < NEAR, f"frame {f} current mask: got {cur_mask[idx]:.1f}, want {MASK(f - 2)}"
        assert abs(cand_mask[idx] - CAND_MASK(f - 2)) < NEAR, f"frame {f} candidate mask: got {cand_mask[idx]:.1f}, want {CAND_MASK(f - 2)}"
        assert abs(cur_ref[idx] - REF(3 + f - 2)) < NEAR, f"frame {f} current ref: got {cur_ref[idx]:.1f}, want {REF(3 + f - 2)}"
        assert abs(cand_ref[idx] - REF(1 + f - 2)) < NEAR, f"frame {f} candidate ref: got {cand_ref[idx]:.1f}, want {REF(1 + f - 2)}"
        # Control: current mask is far from candidate mask on frame 2 (40 vs 60)
        if idx == 0:
            assert abs(cur_mask[idx] - CAND_MASK(0)) > NEAR, (cur_mask[idx], CAND_MASK(0))
    # Nothing drawn over the three picture rows (ORIGINAL, CURRENT CUT, CANDIDATE CUT)
    stack = raw(cf.out, "rgb24", w_, h_)
    srgb = raw(cf.m.source, "rgb24", W, H)[2:8]
    jrgb = raw(cf.silent_clip, "rgb24", W, H)[2:8]
    crgb = raw(cf.cand_span, "rgb24", W, H)
    for idx in range(6):
        frame = np.frombuffer(stack[idx], np.uint8).reshape(h_, w_, 3).astype(np.int16)
        for name, y, want_bytes in (("ORIGINAL", 0, srgb[idx]), ("CURRENT CUT", H + l1, jrgb[idx]), ("CANDIDATE CUT", 2 * H + l1 + l2, crgb[idx])):
            want = np.frombuffer(want_bytes, np.uint8).reshape(H, W, 3).astype(np.int16)
            off = np.abs(frame[y:y + H, :W] - want).max(axis=2)
            edge_rows = np.concatenate([off[2:10], off[-10:-2]]).mean(axis=1).max()
            corner = off[:40, :120].mean()
            assert edge_rows < 12 and corner < 12, f"frame {idx + 2} {name}: edge_rows={edge_rows:.1f}, corner={corner:.1f}"
    # Audio muxed from source even though silent_clip has no audio
    assert not sd._has_audio(str(cf.silent_clip)) and sd._has_audio(str(cf.out)), "audio was not muxed from --source when --joined was silent"
    # Level 5.1 & faststart & sidecar
    level = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=level", "-of", "csv=p=0", str(cf.out)],
                           capture_output=True, text=True).stdout.strip()
    assert level == "51", f"compare view is tagged level {level}, not 5.1"
    atoms = top_atoms(cf.out)
    kinds = [k for k, _, _ in atoms]
    assert "moov" in kinds and "mdat" in kinds and kinds.index("moov") < kinds.index("mdat")
    side = sidecar_tool.read(cf.out)
    roles = {i["role"] for i in side["inputs"]}
    for must in ("the source", "the joined clip", "the table", "the candidate clip", "the candidate table",
                 "the stack's records", "the stack's per-frame records", "the stack's alignment check"):
        assert must in roles, (must, roles)
    # Alignment file and read_back check
    align_path = cf.out.with_name("cmp_span.alignment.txt")
    assert align_path.is_file() and sd.ALIGNMENT_NOTE in align_path.read_text(), align_path
    got = sd.read_back(str(cf.m.source), str(cf.silent_clip), str(cf.out))
    assert got["sound_packets"]["stacked"] and got["sound_packets"]["the_same"], got["sound_packets"]
    for row_key in ("the_original_row", "the_current_row", "the_candidate_row"):
        assert got[row_key]["a_neighbour_fits_better"] == [], (row_key, got[row_key])
    al = got["alignment"]
    assert al["note"] == sd.ALIGNMENT_NOTE, al
    for row_key in ("original", "current", "candidate"):
        assert al[row_key]["best_offset"] == 0, (row_key, al[row_key])
    assert al["current"]["vs_own"] < al["current"]["vs_other"] and al["candidate"]["vs_own"] < al["candidate"]["vs_other"], al
    # Control: shifting the candidate row by +1 frame or swapping current and candidate is caught by compare_alignment
    s_g = grey(cf.m.source, f"crop={W}:{H}:0:0", W, H)
    j_g = grey(cf.silent_clip, f"crop={W}:{H}:0:0", W, H)
    c_g = grey(cf.cand_span, f"crop={W}:{H}:0:0", W, H)
    shifted = sd.compare_alignment(s_g[2:8], j_g[2:8], np.concatenate([c_g[1:], c_g[-1:]]), s_g[2:8], j_g[2:8], c_g, 2)
    assert shifted["candidate"]["best_offset"] == 1, shifted["candidate"]
    swapped = sd.compare_alignment(s_g[2:8], c_g, j_g[2:8], s_g[2:8], j_g[2:8], c_g, 2)
    assert swapped["current"]["vs_own"] > swapped["current"]["vs_other"] and swapped["candidate"]["vs_own"] > swapped["candidate"]["vs_other"], swapped
    return (f"{w_}x{h_} compare stack on 2-7: current:a ({cur_mask[0]:.0f}/{cur_ref[0]:.0f}) vs candidate:a ({cand_mask[0]:.0f}/{cand_ref[0]:.0f}), "
            f"audio={got['sound_packets']['stacked'][:16]}, offsets=0, own={al['candidate']['vs_own']}<other={al['candidate']['vs_other']}")


def compare_candidate_piece_and_unchanged_subject() -> str:
    """`--candidate-piece` synthesises the candidate table by replacing overlapping rows for that subject in `--table`: on the full
    0-11 clip where subject `a` changes (`pa` -> `pcand` on 2-7) and subject `b` is unchanged (`pb` on 8-10 in both), only `a` gets
    `current:a` and `candidate:a` right-column blocks while `b` is recorded in `unchanged_subjects` and named as `(both)` in
    `legend_1`; naming `--subject a,b` includes both subjects in the right column; and `sd.check` passes on the built
    compare stack."""
    import contextlib
    import io
    cf = cmp_fix()
    out = cf.root / "cmp_full.mp4"
    cf.compare(joined=str(cf.m.clip), candidate_joined=str(cf.cand_full), candidate_table="",
               candidate_piece=str(cf.pcand), candidate_piece_first=2, only=None, candidate_first=0, out=str(out))
    head, recs = head_of(out), frames_of(out)
    assert head["columns"] == ["current:a", "candidate:a"] and head["unchanged_subjects"] == ["b"], head
    assert "b: pb (day not recorded) (both)" in recs[9]["legend_1"], recs[9]["legend_1"]
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        sd.check(argparse.Namespace(source=str(cf.m.source), joined=str(cf.m.clip), out=str(out), first=0, candidate_joined=None, candidate_first=None))
    assert sd.ALIGNMENT_NOTE in said.getvalue() and "(best +0)" in said.getvalue(), said.getvalue()
    # Naming both subjects explicitly with --subject puts both in the right column
    out_both = cf.root / "cmp_both.mp4"
    cf.compare(joined=str(cf.m.clip), candidate_joined=str(cf.cand_full), candidate_table="",
               candidate_piece=str(cf.pcand), candidate_piece_first=2, subject="a,b", only="6-8", candidate_first=0, out=str(out_both))
    assert head_of(out_both)["columns"] == ["current:a", "candidate:a", "current:b", "candidate:b"], head_of(out_both)["columns"]
    return f"unchanged b in legend_1, right column {head['columns']}; --subject a,b -> 4 blocks; check passed"


def compare_refusals() -> str:
    """Every compare-mode guard refuses in the tool's own words with no traceback and leaves no output file."""
    cf = cmp_fix()
    never = cf.root / "never_cmp.mp4"
    refused(lambda: cf.compare(candidate_table="", candidate_piece="", out=str(never)), "compare requires --candidate-table or --candidate-piece")
    refused(lambda: cf.compare(candidate_piece=str(cf.pcand), out=str(never)), "pass either --candidate-table or --candidate-piece, not both")
    refused(lambda: cf.compare(candidate_table="", candidate_piece=str(cf.root / "no_such_piece.mp4"), out=str(never)), "candidate piece is not a file")
    refused(lambda: cf.compare(only=None, out=str(never)), "give --only FIRST-LAST for the span to compare")
    refused(lambda: cf.compare(only="0-7", out=str(never)), "is not inside the candidate clip's frames, 2 to 7 of the source")
    refused(lambda: cf.compare(subject="nobody", out=str(never)), "--subject 'nobody' has no piece on frames 2-7 in either table")
    empty = cf.root / "empty_cand.txt"
    empty.write_text("source 0-11 subject=a nothing rendered\n")
    refused(lambda: cf.compare(candidate_table=str(empty), out=str(never)), "lays no piece on frames 2-7: there is nothing to compare")
    wrong_sz = cf.root / "wrong_cand.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=264x200:r=24", "-frames:v", "6", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", "0", str(wrong_sz))
    refused(lambda: cf.compare(candidate_joined=str(wrong_sz), out=str(never)), "264 by 200", "256 by 192")
    assert not never.exists()
    # CLI refusal has no Python traceback
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "stack_delivery.py"), "compare",
                           "--source", str(cf.m.source), "--table", str(cf.m.table), "--joined", str(cf.silent_clip),
                           "--candidate-joined", str(cf.cand_span), "--work", str(cf.root / "work"), "--out", str(never)],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "Traceback" not in proc.stderr and "compare requires --candidate-table or --candidate-piece" in proc.stderr, proc.stderr
    return "8 compare-mode refusals + CLI refusal with no traceback verified"


case("the table as read, through the assembler's readers", the_table_as_read)
case("what the right column reads on each frame", what_each_frame_reads)
case("the built file shows those frames", the_built_file)
case("check: the file read back", the_read_back)
case("a clip of a span that does not begin on frame 0", a_clip_of_a_span)
case("the clip's own check, and the words whole", the_clips_own_check_and_whole_words)
case("each piece read through its job", each_piece_read_through_its_job)
case("the whole delivered frame, its edge too; a clip of another size refused", the_whole_frame_shown)
case("a table that lays nothing", a_table_that_lays_nothing)
case("the middle row is the assembler's own test", the_assemblers_own_test)
case("nothing drawn on the picture", nothing_drawn_on_the_picture)
case("every chip said, none dropped or cut", every_chip_said)
case("who is laid comes from the table, an unread panel said", who_is_laid_from_the_table)
case("the render's day from a record; no head crop said", the_day_and_the_head_crop)
case("check reads the canvas and the width", check_reads_the_canvas)
case("what was not read is said, never 'no gap'; intents in legend 1", what_was_not_read_is_said)
case("the bar is off what H3 saw; the label wraps in its strip", the_bar_off_what_h3_saw)
case("a trial of some frames reads back", a_trial_of_some_frames_reads_back)
case("each piece at its own canvas", each_piece_at_its_own_canvas)
case("a refused reference is said", a_refused_reference_said)
case("the one-a-second parts beside the stack only when it is whole", one_a_second_beside_the_stack_only_when_whole)
case("no stack without its sidecar", no_stack_without_its_sidecar)
case("a square kept inside her track: yellow, with its chip", a_square_kept_inside_her_track)
case("a square kept outside her track: quiet, no chip", a_square_kept_outside_her_track)
case("a render with no region files is said", a_render_with_no_region_files_said)
case("B1: her track over the whole latent step", her_track_over_the_whole_step)
case("B2: a pair load takes both tracks", a_pair_load_takes_both_tracks)
case("B3: a window of another run is left unread", a_window_of_another_run_left_unread)
case("B4: an unlaid frame draws no squares", an_unlaid_frame_draws_no_squares)
case("B5: a track loaded at its own size", a_track_loaded_at_its_own_size)
case("every chip said, within H.264 level 5.1", every_chip_said_within_the_level)
case("a layout over level 5.1 is refused", a_layout_over_the_level_refused)
case("three subjects and a busy frame, within the level", three_subjects_and_a_busy_frame_within_the_level)
case("the view starts without its end: moov at the front", the_view_starts_without_its_end)
case("compare: candidate cut against current cut, panels, audio, sidecar, alignment", compare_candidate_against_current)
case("compare: --candidate-piece shorthand, unchanged subjects, and check subcommand", compare_candidate_piece_and_unchanged_subject)
case("compare: refusals in the tool's own words", compare_refusals)

if __name__ == "__main__":
    code = finish()
    HOLD.cleanup()
    sys.exit(code)

