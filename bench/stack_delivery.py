#!/usr/bin/env python3
"""A delivery clip stacked for the owner's eye: the original, what each render on the frame was given, the clip.

    <python> bench/stack_delivery.py build --source SOURCE --table TABLE.txt --joined CLIP.mp4 --out STACKED.mp4 \\
        --work SCRATCH_FOLDER [--tag WORDS] [--alias OLD=NEW]... [--copy-rate 24] [--only FIRST-LAST] [--first SOURCE_FRAME] \\
        [--data DATA_FOLDER] [--coverage FRAME_COVERAGE.json] [--one-a-second]
    <python> bench/stack_delivery.py compare --source SOURCE --table CURRENT_TABLE.txt --joined CURRENT_CLIP.mp4 \\
        [--candidate-table CAND_TABLE.txt] [--candidate-joined CAND_CLIP.mp4] [--candidate-piece PIECE.mp4] \\
        [--candidate-piece-first FRAME] [--subject WHO] --out STACKED.mp4 --work SCRATCH_FOLDER \\
        [--current-tag WORDS] [--tag WORDS] [--alias OLD=NEW]... [--copy-rate 24] [--only FIRST-LAST] \\
        [--first SOURCE_FRAME] [--candidate-first SOURCE_FRAME] [--data DATA_FOLDER] [--coverage FRAME_COVERAGE.json] [--one-a-second]
    <python> bench/stack_delivery.py check --source SOURCE --joined CLIP.mp4 [--candidate-joined CAND_CLIP.mp4] --out STACKED.mp4 [--first SOURCE_FRAME]

**What it buys.** A render is handed over stacked, never bare: the masked input on top, the render below, frame
numbers burned in, so a thing seen on playback is reported with the file and the frame. `capture_masked_run.py
video` does that for one load. A delivery clip is many loads laid by a table, and twice (2026-10-11,
2026-10-13) the whole clip's stacked form was made by a one-off script beside the table. This is that script
as a tool: it reads the SAME table the assembler built the clip from, so the stack shows the clip's own pieces.

**One picture a source frame: three rows at the source's own size on the left, the canvas's column on the right**
(the owner, 2026-10-16: "JUST those 3 things ... at REAL resolution", a legend that does not cover the picture, and one
file the owner, the sessions and the outside reviewer all watch). Top to bottom on the left:

    ORIGINAL           the source frame, unscaled
    legend 1           "ABOVE: ORIGINAL", the frame number, and for each subject the piece the TABLE lays on the frame with its
                       render's day (`queued_at` of the queued record that names the piece, "day not recorded" when none), or
                       ORIGINAL, or "ORIGINAL on purpose (<the table's reason>)" for a subject its `source` lines leave; then
                       "BELOW: WHAT WAS CHANGED" with its key: the loader's crop in words (one for each canvas the pieces
                       have), each zoomed piece's box in words in its subject's colour, and which subjects the red bands name
    WHAT WAS CHANGED   the source dimmed, and the clip at full strength wherever the assembler's own test calls it changed
                       (`assemble_delivery.changed` on both frames read as yuv420p: one test, one copy), its edge white; each
                       zoomed piece's box (`assemble_delivery.box_at_the_source`) as a 1 px outline in its subject's colour
    legend 2           the chips, wrapped to as many lines as the clip's busiest frame needs, up to what H.264 level 5.1
                       leaves (the height is one for the whole clip; a frame's chips past it go to the right column, below): the per-frame gap check's runs (`bench/frame_coverage.py`, `--coverage`), every kind it writes in
                       `CHIP_WORDS`, and the joined clip's own check flags (`<clip>.check.json`) each with its level ("iffy"):
                       readings of degree, by that check's thresholds, not verdicts. What was not read is said on every
                       frame: "gaps not checked: no --coverage", "check flags not read: <why>"; "no gap or flag on this frame"
                       only where both were read. Then "BELOW: THE CUT" (or `--tag`)
    THE CUT            the delivery clip, unscaled; it must be the source's size (`assemble_delivery.py --size source`):
                       a clip of another size is refused, never stretched
    right column       a block a subject, the same block for the whole clip (C1, 2026-10-15), as wide as the largest canvas:
                       its label strip, WHAT H3 SAW (the piece's own mask picture, the lower half of its `_with_mask.mp4`,
                       cut at the piece's own canvas and set in its tile at its own shape; where it does not read, "what H3
                       saw: not read, <why>" in its place), and under it the reference its graph wires, on the frame the
                       graph gave it, as a head crop following the face (the strip's own rule, `capture_masked_run.head_boxes`;
                       "(no points: no head crop)" when there is none) and whole; a reference the tool refused to open is
                       said NOT OPENED with the refusal. The label strip holds the block's words, wrapped, in a font scaled
                       with the blocks, and each `piece_changes_nothing` flag of its piece as a red bar of its own: the strip
                       is one height for the clip, as tall as the longest label can be. The pictures are scaled by one
                       printed factor only when the blocks do not fit (`right_tiles`). A green border on the head crop: the
                       text encoder's frame.
                       THE SQUARES KEPT AS THE ORIGINAL (the owner's go, 2026-10-16), on the mask picture only: from the
                       render's own region files (`<render stem>_windows/*_region.npz`, read by the node's own
                       `video_mask.load_window_region`), the units of the grown region the node did not regenerate,
                       `token_mask(grow(mask, margin)) & ~tokens` with the node's own helpers, in the window that lays the
                       frame, the windows read as the capture reads them (each following on from the last, stopped at the
                       render's end, a later run's files left unread and named: `KeptSquares`). Nothing on a frame the piece
                       does not lay (its mask empty, or the cut gate shut). Split by the subject's own track as the load was
                       handed it, every track of a load of two subjects together: a unit more than
                       `capture_masked_run.OWN_TRACK_CELL_SHARE` hers on any frame of its latent step (as the capture's
                       `grid_carve` and `node_taken_inside` count it, through its own `_units`) is filled bright yellow and
                       counted in a chip ("N squares kept as the original inside <subject>'s outline"); one outside her track
                       gets a quiet wash and no chip (`KEPT_INSIDE`, `KEPT_OUTSIDE`: fills, never a line). A render with no
                       region files, windows that do not follow on, a file that does not read, or a track not read is said in
                       the label strip, never left blank.

**Within H.264 level 5.1** (`LEVEL`, `LEVEL_MACROBLOCKS`, `LEVEL_SIDE`, `most_height`: 3840 rows at 2448 wide), which every
browser plays (the owner's review page crashed Chrome on 2448x4172 views, level 6.0, 2026-10-16). The picture rows stay at the
source's size and legend 1 is whole, its words as short as they allow and its lines tight; legend 2 takes the rows left. A
frame whose chips legend 2 cannot hold keeps what fits, says "N more on this frame: in the right column, below what H3 saw",
and lays the rest there, under the blocks, headed by its frame number (`chips_overflow` in its record): never cropped or
dropped. The blocks are scaled down to make that room when they must. A layout that cannot fit (the rows and legend 1 alone
over the level, or the blocks and the overflow over the right column) is refused before anything is encoded, naming the
sizes. The file is encoded with `-level 5.1`.

**What is drawn where.** On the three left rows nothing but, on the middle row, the white edge of the changed region
and the zoom outlines: 1 px, in the subject's colour, and never over the picture's outer 10 rows or columns (`EDGE_CLEAR`, where the loader's crop cuts and the assembler
carries the edge, `assemble_delivery.carry_edge_rows`). Every word, and the red mark for a subject shown as the
original while tracked (`--coverage`'s `original_while_tracked`), is in the legends: the mark is a red band along
the bottom of legend 1 and the top of legend 2, framing the middle row from outside it. The words drawn are in the
records whole, with each chip's drawn box.

**Its records**: `<stem>.stack.json` beside the video (the columns, where each tile lies, what each piece was given,
each piece's canvas, the zooms by piece file and the zoom records that did not read with why, the days and where each
came from, the check's flags or why they were not read, whether the gaps were checked, the frames stacked and the
source frame the clip begins on, both legends' heights, which region files and which track each piece's kept squares
were read from or why not; everything keyed by a piece is keyed by its FILE) and `<stem>.stack.frames.jsonl`, a line a source frame: what each column
showed and its label, the legend's words, the chips as drawn, the changed pixels, every word. `<out>.json` is the
video's own sidecar (`bench/sidecar.py`). The file's metadata names the stack's record with the source's size, the
canvas, both legends' heights and the right column's width, which `check` reads back. The sidecar is written before
success is said; a sidecar the tool refuses or fails to write is a hard failure naming the stack. `--one-a-second`
also writes the same frames at one a second and no sound, in parts no larger than the outside reviewer's reader takes
(`_lib.frames.OneASecond`), each named by its source frames and listed in the records: encoded under `--work` and
moved beside `--out` only once both encoders have finished whole.

**What is read from where.** The table through the assembler's own two readers, so the format has one home. A
piece's reference video and the source frame its graph starts on from the graph ComfyUI saved in the picture
beside the piece: a load that continued a stored window loads from an earlier frame than its file begins on,
and its reference is taken from the graph's frame, not the file's. Each piece's canvas from its own mask picture
(else the piece itself). The rate and the sound from the clip.

**Refused before anything is encoded**: a table that lays nothing; a source of odd width or height (yuv420p does
not hold it); a clip of another size than the source; a piece whose canvas is not known (neither it nor its mask
picture is there); a `--coverage` run of a kind with no words.

**`check`** reads the stacked file back: its size against its metadata (the width the source's and the right
column's, the height three rows and the two legends), its frame count against the clip's, its sound's packets
against the clip's, and each of the top and bottom rows held against the file it came from on its own frame and on
the frames either side (small, grey, whole: no label is drawn on them). The frames stacked and the clip's first source
frame come from the stack's own record, so a build of some frames (`--only`) reads back right; `--first` is used only
where the record is not beside the file. Printed as figures, with the source's size and the canvas apart.

**A reference's path comes out of a picture's metadata**, not from whoever runs this, so it is opened only when
it is a plain video file on this machine (`not_a_plain_video`), and every file here is opened as a local file,
by no other protocol, and as the container its suffix names (`CONTAINER`): ffmpeg is told what the file is and
does not decide from its content, so a playlist under a video's name is not followed. A reference that is not opened is said when the table is read and has no panel.

**What it cannot show or see.** The parts in their own colours (not in a render's files); the still a render
took a face from (it is in the records); a zoomed render's mask at its whole-frame place (its mask picture is of
the zoomed box, as the node drew it); the assembly's restored pixels as a picture (they are in the check's flags,
not drawn); two pieces of one subject on one frame (the first is shown, the second said); a change under
`assemble_delivery.CHANGE` or thinner than its 3x3 opening (the assembler's own test drops it). The read-back
cannot see the middle row, the legends or the right column: whether a mask picture or a reference is on the right
frame rests on the table's and the graphs' own figures (printed by `build`), on the records and on a look. A held
or still stretch of picture fits its neighbours as well as itself, so a tie there says nothing either way.

The working file is written under `--work` and only the finished file goes to `--out`: the share is not scratch. The view and
its one-a-second parts are written with their index at the front (`-movflags +faststart`), so a browser starts them without
reading to the end, and the sidecar is written after, of the final bytes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib.frames import FIT, probe  # noqa: E402
from frame_coverage import CHIP_NO_TRACK_WORDS, CHIP_WORDS, LOST_KINDS, chip_words_of  # noqa: E402

#: Lines of words in the block under the middle row of the layout before 2026-10-16, and a line's height at a canvas 768
#: tall: kept only so `read_back` can read a stack of that layout. Reasoned then: two pieces, their two references and
#: three subjects left as the source is the most one frame of a three-subject clip has had.
LINES, LINE_AT_768 = 7, 22
#: A subject's colour on the picture (its zoom box, its legend words, its right-hand block), in the order of the columns.
COLOURS = ((255, 120, 90), (90, 200, 255), (150, 230, 110), (240, 200, 70))
#: THE MOST A VIEW'S FRAME MAY BE: H.264 level 5.1, which every browser's decoder plays (the owner's review page crashed Chrome,
#: "Aw Snap", error code 5, on the whole-video views of 2026-10-16: 2448x4172, level 6.0, because legend 2 grew to the busiest of
#: 6395 frames; every view at or under 3724 tall, level 5.1, played). Inherited from the standard's level table: 36864 macroblocks a
#: frame, and no side over 4096 px (the decoders' own limit, which the lead set with it). A layout over it is refused before encoding.
LEVEL = "5.1"
LEVEL_MACROBLOCKS = 36864
LEVEL_SIDE = 4096


def most_height(width: int) -> int:
    """The tallest even frame within `LEVEL_MACROBLOCKS` and `LEVEL_SIDE` at `width` (3840 at 2448). Read at call time, so a
    check can lower the limit."""
    return min(LEVEL_SIDE, LEVEL_MACROBLOCKS // -(-int(width) // 16) * 16) // 2 * 2


#: Source pixels at each side of the picture no outline is drawn over. Reasoned: the loader's crop cut 6 rows top and bottom
#: of this video (1424x1080 at 1024x768) and the assembler carries the laid edge 8 rows past it (`assemble_delivery.CARRY_FULL_PX`);
#: a fault in those rows (2823b's bottom, 2026-10-15) must be seen bare.
EDGE_CLEAR = 10


def _assembler():
    spec = importlib.util.spec_from_file_location("assemble_delivery", HERE / "assemble_delivery.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: What a reference video's file may end in. A reference's path is read out of a picture file's metadata, so it is
#: not this tool's caller's word: only a plain video file on disk is opened, never a URL, a protocol string or a
#: playlist that would have ffmpeg fetch something else.
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")
#: The container ffmpeg is TOLD a file is, by its suffix, so it never decides from the file's content. Left to
#: itself ffmpeg probes: a text file named `x.mp4` that is a playlist is read as a playlist and the files it
#: lists are opened, which a check of the name alone does not stop. With the format forced such a file simply
#: does not read. (The QuickTime reader does not follow a file's references to other files unless asked to.)
CONTAINER = {".mp4": "mov", ".mov": "mov", ".mkv": "matroska", ".webm": "matroska"}


def not_a_plain_video(path: str) -> str | None:
    """Why a path read from a saved graph is not opened, or None when it is a plain video file on this machine."""
    if "://" in path or path.split(":", 1)[0] in ("concat", "subfile", "data", "pipe", "fd", "cache", "crypto", "async", "http", "https", "tcp", "udp", "rtmp", "ftp"):
        return "it is not a path on this machine"
    if Path(path).suffix.lower() not in VIDEO_SUFFIXES:
        return f"its name does not end in one of {', '.join(VIDEO_SUFFIXES)}"
    if not Path(path).is_file():
        return "it is not a file on this machine"
    return None


def frame_of(seconds: float, rate: float) -> int:
    """A loader's start time as a frame of its copy: the job graphs set it a fraction of a frame early so the loader
    lands on the frame, and an exact start comes to the same frame."""
    return math.ceil(seconds * rate - 1e-3)


def _loader(graph: dict, link, depth: int = 0) -> dict | None:
    node = graph.get(str(link[0])) if isinstance(link, list) and link else None
    if node is None or depth > 8:
        return None
    if isinstance(node["inputs"].get("video"), str):
        return node["inputs"]
    for v in node["inputs"].values():
        if isinstance(v, list) and len(v) == 2:
            got = _loader(graph, v, depth + 1)
            if got:
                return got
    return None


def graph_reads(piece: str, rate: float) -> tuple[int | None, tuple[str, int] | None]:
    """From the graph saved in a piece's picture: the source frame it loaded from, and its reference video with the
    frame it starts on. (None, None) when there is no picture, no graph or no Masked Source in it; a reference
    of None where the graph wires none."""
    png = Path(piece).with_suffix(".png")
    if not png.is_file():
        return None, None
    from PIL import Image
    try:
        graph = json.loads(Image.open(png).info.get("prompt") or "")
    except (ValueError, OSError):
        return None, None
    node = [n for n in graph.values() if n["class_type"] == "MiniMaxH3MaskedSource"]
    if len(node) != 1:
        return None, None
    ins = node[0]["inputs"]
    frames = _loader(graph, ins.get("frames"))
    direct = graph.get(str(ins["motion_video"][0])) if isinstance(ins.get("motion_video"), list) else None
    ref = direct["inputs"] if direct is not None and isinstance(direct["inputs"].get("video"), str) else None     # a file, wired straight in
    return (frame_of(float(frames.get("start_time", 0.0)), rate) if frames else None,
            (ref["video"], frame_of(float(ref.get("start_time", 0.0)), rate)) if ref else None)


def read_rows(table: str, rate: float, aliases: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """The table's rows, each with its mask picture, its graph's first frame and its reference; and its `source` lines."""
    tool = _assembler()
    name = lambda label: aliases.get(label, label)
    rows = []
    for r in tool.read_table(table):
        graph_first, ref = graph_reads(r["piece"], rate)
        less = []
        for label, wanted in r["restore"]:
            text = tool.restore_text(label, wanted)
            less.append(name(label) + text[len(label):])
        refused = not_a_plain_video(ref[0]) if ref else None
        if refused:
            print(f"the reference the saved graph of {Path(r['piece']).name} names is NOT OPENED ({refused}): {ref[0]!r}", flush=True)
            ref = None
        rows.append({"first": r["first"], "last": r["last"], "piece": r["piece"], "piece_first": r["piece_first"],
                     "name": Path(r["piece"]).stem.rsplit("_", 1)[0], "subject": name(r["subject"]) if r.get("subject") else None, "less": less,
                     "mask": str(Path(r["piece"]).with_name(Path(r["piece"]).stem + "_with_mask.mp4")),
                     "graph_first": r["piece_first"] if graph_first is None else graph_first, "ref": ref, "ref_refused": refused})
    return rows, [{**i, "subject": name(i["subject"])} for i in tool.read_intents(table)]


def panels_of(f: int, rows: list[dict]) -> list[dict]:
    """What the middle row reads for source frame `f`, in the table's order: for each row on the frame its mask
    picture's frame, then its reference's frame where it has one."""
    out = []
    for r in rows:
        if not r["first"] <= f <= r["last"]:
            continue
        out.append({"kind": "mask", "row": r, "file": r["mask"], "frame": f - r["piece_first"], "words": f"MASKED for {r['name']}"})
        if r["ref"]:
            out.append({"kind": "reference", "row": r, "file": r["ref"][0], "frame": r["ref"][1] + f - r["graph_first"],
                        "words": f"REFERENCE VIDEO given to {r['name']}"})
    return out


def words_of(f: int, rows: list[dict], intents: list[dict], panels: list[dict]) -> list[str]:
    """The block's lines for source frame `f`: the pieces, the references, the subjects left as the source."""
    active = [r for r in rows if r["first"] <= f <= r["last"]]
    lines = [("ON THIS FRAME: " if i == 0 else "and with it: ") + r["name"] + (f" ({r['subject']}'s)" if r["subject"] else "")
             + "".join(", less " + x for x in r["less"]) for i, r in enumerate(active)]
    lines = lines or ["ON THIS FRAME: the original, no render laid"]
    lines += [f"reference video given to {p['row']['name']}: {Path(p['file']).name}, its frame {p['frame']}" for p in panels if p["kind"] == "reference"]
    lines += [f"left as the original on purpose, {i['subject']}: {i['why']}" for i in intents if i["first"] <= f <= i["last"]]
    return lines


class Reader:
    """Frames of one file from a frame on, in order, as [h, w, 3] (or [h, w] grey, or the flat bytes of a yuv420p frame).
    A frame that cannot be read is None."""
    def __init__(self, path: str, skip: int, vf: str, w: int, h: int, pix: str = "rgb24"):
        self.at = skip
        self.size, self.shape = {"rgb24": (w * h * 3, (h, w, 3)), "gray": (w * h, (h, w)), "yuv420p": (w * h * 3 // 2, (w * h * 3 // 2,))}[pix]
        told = CONTAINER.get(Path(path).suffix.lower())
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-protocol_whitelist", "file", *(["-f", told] if told else []),
                                   "-i", "file:" + str(Path(path).resolve()),
                                   "-vf", f"select='gte(n\\,{max(skip, 0)})'" + ("," + vf if vf else ""),
                                   "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix, "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def frame(self, n: int):
        buf = None
        while self.at <= n:
            buf = self.p.stdout.read(self.size)
            self.at += 1
        return None if buf is None or len(buf) < self.size else np.frombuffer(buf, np.uint8).reshape(self.shape)

    def close(self) -> None:
        self.p.stdout.close()
        self.p.kill()
        self.p.wait()


#: The original and the clip as whole frames in their rows: scaled to fit, the rest black. The loader's cut (`FIT`) took
#: the rows a fault at the delivered frame's edge lies in (2026-10-15: a step of 33-41 grey levels along 2823b's bottom).
WHOLE = "scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"


def _rate(path: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                          capture_output=True, text=True, check=True).stdout.strip().strip(",")


def sizes(h: int) -> tuple[int, int]:
    """(a line's height, the block's height, even) for a canvas `h` tall, in the layout before 2026-10-16 (read back only)."""
    line = max(10, round(LINE_AT_768 * h / 768))
    return line, 2 * ((line * LINES + 6 + 1) // 2)


def columns_of(rows: list[dict], first: int, last: int) -> list[str]:
    """The middle row's columns, one a subject, in the order each first lays a piece on the frames stacked: a subject keeps
    its column for the whole clip (C1, 2026-10-15: the tiles jumped 192 px and swapped rows as pieces began and ended)."""
    out = []
    for r in sorted((r for r in rows if r["first"] <= last and r["last"] >= first), key=lambda r: (r["first"], rows.index(r))):
        who = r["subject"] or r["name"]
        if who not in out:
            out.append(who)
    return out


def right_tiles(w: int, h: int, n: int, total_h: int, label_px: int = 34) -> tuple[float, list[dict]]:
    """The right column's blocks, one a subject, as [x, y, w, h] inside the column (which is `w` wide, the render canvas's own
    width): the block, its label strip (`label_px` tall, as [x, y, w, h]: the block's words and its check's bars, never over a
    picture), the piece's mask picture at the canvas's size, and under it the reference's head crop and the reference whole.
    The pictures are scaled by one factor, printed in the label, only when `n` blocks do not fit `total_h`."""
    body = h + h // 2 + 12
    sc = min(1.0, max(0.05, (total_h / max(1, n) - label_px) / body))
    # a block's height rounded DOWN: n blocks each rounded up by a fraction of a row came to one row over the column (the whole
    # video of 2026-10-16, three blocks: 3273 rows where 3272 were left under the chips legend 2 could not hold)
    bh, mw, mh, hs = label_px + int(body * sc), round(w * sc), round(h * sc), round(h // 2 * sc)
    ww = min(w - hs - 24, round(hs * w / h))
    out = []
    for i in range(n):
        y = i * bh
        ry = y + label_px + mh + 4
        out.append({"block": [0, y, w, bh], "label": [8, y + 4, w - 16, label_px - 8], "mask": [(w - mw) // 2, y + label_px, mw, mh],
                    "head": [8, ry, hs, hs], "whole": [hs + 16, ry, ww, round(ww * h / w)]})
    return sc, out


#: The colours of the squares kept as the original in a WHAT H3 SAW panel: (colour, its weight over the picture). Inside the
#: subject's own track bright yellow, over most of the square; outside it a quiet blue-grey wash. Reasoned: the first must not
#: be missed and the second must not be read as a fault. Fills only, never a line.
KEPT_INSIDE = ((255, 230, 0), 0.65)
KEPT_OUTSIDE = ((30, 40, 70), 0.55)


class KeptSquares:
    """The squares a render's node kept as the original, frame by frame, read from the render's own region files.

    THE WINDOWS, as the capture reads them (`capture_masked_run.read_saved_regions`): each window's file
    (`video_mask.save_window_region`, found by `capture_masked_run.window_region_files`, in window order) is read by the node's
    own `video_mask.load_window_region`; each lays its frames from its `first_frame + trim` to its end, and must start where the
    one before ended. Files after the one that reaches the render's last frame are an earlier, longer run's, left unread and
    named (`left_unread`). A gap, or files that stop short of the render's end, and nothing is shown, with the reason.

    THE TAKEN SQUARES of a window's latent step are the grown region's units less the region it regenerated:
    `token_mask(grow(mask, margin)) & ~tokens`, with the node's own `grow` and `token_mask`, in whole tokens or latent cells as its
    `edge` setting says, of the window's picture (the zoomed one for a zoomed window; the panel shows that picture too). On a frame
    the piece does not lay (its mask empty there, or `video_mask.cut_gate` shut across a cut) the composite shows the source, and
    nothing is drawn or counted (`not_laid`).

    THE SPLIT is by the subject's own TRACK, as the load was handed it: every "the subject's mask" input of what it was given (a
    load of two subjects is handed both, and the union is hers), each read at its loader's size (the file's own where the loader
    leaves it 0 by 0) and brought into the window's picture as the node brings a mask (`crop_mask` for a zoomed window, `fit_mask`
    otherwise). A unit is inside when it is more than `capture_masked_run.OWN_TRACK_CELL_SHARE` hers on ANY frame of its latent
    step, as `capture_masked_run.grid_carve` and `node_taken_inside` count it, through their own `_units`. Each window's frames of
    the track are read once, in order, and kept as one bool a unit and step."""

    def __init__(self, row: dict, given: dict | None, given_why: str | None, capm, vm, load_first: int, render_frames: int | None = None):
        self.row, self.vm, self.capm, self.load_first = row, vm, capm, int(load_first)
        self.windows, self.why, self.tracks, self.track_why, self.left_unread, self._hers = [], None, [], None, [], {}
        files = capm.window_region_files(row["piece"])
        name = Path(row["piece"]).stem.rsplit("_", 1)[0]
        if not files:
            self.why = f"no window region files beside the render ({name}_windows/{name}_window_*_region.npz)"
            return
        end = 0
        try:
            for f in files:
                if render_frames is not None and end >= render_frames:
                    self.left_unread.append(f.name)             # an earlier, longer run's file: not this render's
                    continue
                got = vm.load_window_region(str(f))
                mask, tokens = got["mask"], got["tokens"]
                start, trim, count = int(got["first_frame"]), int(got["trim"]), int(mask.shape[0])
                if start + trim != end:
                    self.windows, self.why = [], (f"{f.name} lays from load frame {start + trim} and the window before it ended on {end}: "
                                                  "the windows do not follow on")
                    return
                lt, lh, lw = (int(v) for v in tokens.shape)
                whole = (got["source"].get("edge") or vm.EDGE_TOKENS) == vm.EDGE_TOKENS
                grown = vm.token_mask(vm.grow(mask, got["margin"]), lt, lh, lw, whole) > 0.5
                taken = (grown & ~(tokens > 0.5)).numpy()
                gate = (vm.cut_gate(mask, lt, got["source"].get("cuts"), start) > 0.5).numpy()
                has_mask = (mask > 0.5).flatten(1).any(dim=1).numpy()
                self.windows.append({"file": f.name, "start": start, "trim": trim, "count": count, "zoom": got.get("zoom"),
                                     "picture": (int(mask.shape[1]), int(mask.shape[2])), "cells": (lh, lw), "whole": whole,
                                     "taken": taken[:, ::2, ::2] if whole else taken, "runs": np.cumsum([0] + vm.run_lengths(lt)),
                                     "gate": gate, "has_mask": has_mask})
                end = start + count
        except Exception as e:                      # a file the node's own reader refuses
            self.windows, self.why = [], f"a window region file does not read ({type(e).__name__}: {e})"
            return
        if render_frames is not None and end < render_frames:
            self.windows, self.why = [], f"the window region files lay {end} load frames and the render has {render_frames}"
            return
        if given is None:
            self.track_why = f"what the load was given is not read ({given_why})"
            return
        tracks = [x for x in given.get("inputs") or [] if x.get("role") == "the subject's mask" and x.get("file")]
        if not tracks:
            self.track_why = "what the load was given names no \"the subject's mask\""
            return
        for t in tracks:
            refused = not_a_plain_video(t["file"])
            if refused:
                self.tracks, self.track_why = [], f"{Path(t['file']).name}: {refused}"
                return
            loader = t.get("loader") or {}
            size = tuple(int(v or 0) for v in (loader.get("custom") or ())[:2])
            if len(size) != 2 or not all(size):
                # a loader that leaves its size to the file hands the track at the file's own size (never 0 by 0)
                try:
                    size = tuple(probe(t["file"])[:2])
                except (ValueError, IndexError, OSError) as e:
                    size = (0, 0)
                    why = str(e)
                else:
                    why = "its own size reads as 0"
                if not all(size):
                    self.tracks, self.track_why = [], f"{Path(t['file']).name}: its loader leaves the size to the file and {why}"
                    return
            self.tracks.append((t["file"], int(loader.get("first_frame") or 0), size))

    def _hers_of(self, wd: dict):
        """[latent steps, units] of bool: more than the capture's share hers on any frame of the step, the union of her tracks,
        in window `wd`'s picture. Read once a window; None where a track has no frame at all here."""
        import torch
        if wd["file"] in self._hers:
            return self._hers[wd["file"]]
        lh, lw = wd["cells"]
        h, w = wd["picture"]
        union = None
        for file, first, (cw, ch) in self.tracks:
            reader = Reader(file, first + wd["start"], FIT.format(w=cw, h=ch), cw, ch, "gray")
            frames, last = [], None
            for i in range(wd["count"]):
                got = reader.frame(first + wd["start"] + i)
                last = got if got is not None else last     # a frame past the track's end holds its last, as the node holds a window's
                frames.append(last)
            reader.close()
            if last is None:
                self._hers[wd["file"]] = None
                return None
            frames = [x if x is not None else np.zeros_like(last) for x in frames]
            t = torch.from_numpy(np.stack(frames) > 127).to(torch.float32)
            if wd["zoom"]:
                t = self.vm.crop_mask(t, wd["zoom"]["box"], int(wd["zoom"]["canvas"][0]), int(wd["zoom"]["canvas"][1]))
            elif tuple(t.shape[1:]) != (h, w):
                t = self.vm.fit_mask(t, w, h)
            union = t if union is None else torch.maximum(union, t)
        hers = (self.capm._units(self.vm, union > 0.5, lh, lw, wd["whole"]) > _own_share(self.capm)).numpy()
        runs = wd["runs"]
        steps = np.stack([hers[runs[s]:runs[s + 1]].any(axis=0) for s in range(len(runs) - 1)])
        self._hers[wd["file"]] = steps[:, ::2, ::2] if wd["whole"] else steps
        return self._hers[wd["file"]]

    def on(self, f: int) -> dict:
        """What frame `f` shows: {"windows": {file: {"inside", "outside", "taken"}}, "draw": (window, inside units, outside units)
        or None, "not_split": why or None, "not_laid": why or None} or {"not_read": why}."""
        if self.why:
            return {"not_read": self.why}
        n = f - self.load_first
        wd = next((x for x in self.windows if x["start"] + x["trim"] <= n < x["start"] + x["count"]), None)
        if wd is None:
            return {"not_read": f"no window region file lays load frame {n} (source frame {f})"}
        k = n - wd["start"]
        if not (wd["gate"][k] and wd["has_mask"][k]):
            why = "its mask is empty on this frame" if not wd["has_mask"][k] else "the cut gate leaves this frame as the source"
            return {"windows": {wd["file"]: {"inside": 0, "outside": 0, "taken": 0}}, "draw": None, "not_split": None,
                    "not_laid": f"the piece lays nothing here ({why}): the composite shows the source"}
        step = int(np.searchsorted(wd["runs"], k, side="right")) - 1
        taken = wd["taken"][step]
        hers = self._hers_of(wd) if self.tracks else None
        not_split = self.track_why if not self.tracks else (None if hers is not None else "her track has no frame in this window")
        inside = taken & hers[step] if hers is not None else np.zeros_like(taken)
        outside = taken & ~inside
        return {"windows": {wd["file"]: {"inside": int(inside.sum()), "outside": int(outside.sum()), "taken": int(taken.sum())}},
                "draw": (wd, inside, outside), "not_split": not_split, "not_laid": None}

    def paint(self, img: np.ndarray, draw) -> np.ndarray:
        """The picture with the squares filled (no line anywhere): inside the track `KEPT_INSIDE`, outside it `KEPT_OUTSIDE`. The
        units are the window's grid laid over the panel's picture, which shows the window's picture at its own size."""
        wd, inside, outside = draw
        out = img.astype(np.float32)
        ph, pw = out.shape[:2]
        lh, lw = wd["cells"]
        span = 2 if wd["whole"] else 1
        for units, (colour, weight) in ((outside, KEPT_OUTSIDE), (inside, KEPT_INSIDE)):
            for r, c in zip(*np.nonzero(units)):
                y0, y1 = round(r * span * ph / lh), round(min((r + 1) * span, lh) * ph / lh)
                x0, x1 = round(c * span * pw / lw), round(min((c + 1) * span, lw) * pw / lw)
                out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - weight) + np.array(colour, np.float32) * weight
        return out.clip(0, 255).astype(np.uint8)

    def close(self) -> None:
        """Nothing is held open: each window's track frames are read once and closed (kept for the callers that close it)."""


def _own_share(capm) -> float:
    """The capture's own threshold for a unit counted as the subject's (`capture_masked_run.OWN_TRACK_CELL_SHARE`), imported."""
    return float(capm.OWN_TRACK_CELL_SHARE)


def squares_words(who: str, kept: dict) -> list[tuple[str, tuple]]:
    """The label strip's lines for the squares kept as the original on a frame: the count inside her outline (a chip, yellow) where
    there are any; both counts where two windows lay the frame; the count unsplit where her track is not read; and why nothing is
    shown where the region files are not there or do not read. Nothing for squares kept outside her track alone."""
    if "not_read" in kept:
        return [(f"squares kept as the original not shown: {kept['not_read']}", (200, 200, 200))]
    counts = kept["windows"]
    if kept.get("not_split"):
        n = sum(v["taken"] for v in counts.values())
        return [(f"{n} square{'s' if n != 1 else ''} kept as the original, not split: {kept['not_split']}", (230, 180, 60))] if n else []
    if len(counts) > 1:
        return [(f"{len(counts)} windows lay this frame: " + "; ".join(f"{name}: {v['inside']} kept inside {who}'s outline, {v['outside']} outside"
                                                                     for name, v in counts.items()), KEPT_INSIDE[0])]
    n = next(iter(counts.values()))["inside"]
    return [(f"{n} square{'s' if n != 1 else ''} kept as the original inside {who}'s outline", KEPT_INSIDE[0])] if n else []


def job_folders(data: Path) -> dict[str, tuple[Path, str, str | None]]:
    """Each render's file name to the job folder that queued it, the render's path under the output folder, and when it was
    queued as the record says (`queued_at`, in the clock the record was written in; None where it has none), from every
    `*_job/<load>.render.queued.json` under `data`."""
    out = {}
    for q in sorted(data.glob("*_job/*.render.queued.json")):
        try:
            held = json.loads(q.read_text())
        except ValueError:
            continue
        files = held.get("files") or []
        if files:
            out[Path(files[0]).name] = (q.parent, files[0], held.get("queued_at"))
    return out


def render_days(rows: list[dict], folders: dict[str, tuple[Path, str, str | None]]) -> dict[str, tuple[str, str]]:
    """Each piece's file name to (its render's day as MM/DD, where that was read): the `queued_at` of the queued record that names
    it, as the record wrote it; else the date its job folder's name begins with; else "day not recorded". Never the file's time:
    a copy or a touch moves that (the blind read of 2026-10-16)."""
    import re
    out = {}
    for r in rows:
        name = Path(r["piece"]).name
        found = folders.get(name)
        at = re.match(r"\d{4}-(\d\d)-(\d\d)", (found[2] or "") if found else "")
        named = re.match(r"\d{4}-(\d\d)-(\d\d)_", found[0].name) if found else None
        if at:
            out[name] = (f"{at.group(1)}/{at.group(2)}", f"queued_at of {found[0].name}")
        elif named:
            out[name] = (f"{named.group(1)}/{named.group(2)}", f"the date its job folder {found[0].name} is named by (its queued record has no queued_at)")
        else:
            out[name] = ("day not recorded", "no queued record names its file")
    return out


def given_of_pieces(rows: list[dict], data: Path) -> dict[str, dict]:
    """What each piece's load was handed, read from its own graph (`bench/what_h3_got.py`), by the piece's FILE name: two files of
    one load are two pieces, each read from the record that names it. A piece whose job folder is not found is shown from its
    picture's graph as before, and said so."""
    import what_h3_got
    folders = job_folders(data)
    out, refused = {}, {}
    for r in rows:
        file = Path(r["piece"]).name
        found = folders.get(file)
        if found is None or file in out:
            continue
        folder, under, _queued = found
        load = Path(r["piece"]).stem.rsplit("_", 1)[0]
        # the output folder is the piece's own path less the part the queued record names under it (2026-10-15: a run with no
        # H3_OUTPUT_DIR in its environment had the reader refuse every piece, for the zoom record alone)
        piece = str(r["piece"])
        output = Path(piece[:-len(under)]) if piece.endswith(under) else None
        try:
            out[file] = what_h3_got.read(folder, load, "render", output=output)
        except SystemExit as e:
            refused[file] = str(e)
            print(f"what H3 was given for {file} could not be read from {folder.name}: {e}", flush=True)
    return out, refused


def flow(items: list[dict], font, width: int, gap: int) -> list[dict]:
    """Items laid left to right in lines no wider than `width`, every word kept: an item that does not fit the line's room
    goes on to the next line, and one wider than a whole line is broken between its words (a word wider than a line, between
    its letters) over as many lines as it needs, each part drawn whole. An item is {"text", "colour", "boxed"}, and
    "break": True begins a new line. Returns the parts as {"item": index, "text", "x", "line", "w"}, x from 0."""
    pad = box_pad(font)
    wide = lambda t, boxed: round(font.getlength(t)) + (2 * pad if boxed else 0)  # noqa: E731
    parts, x, line = [], 0, 0
    for i, it in enumerate(items):
        if it.get("break") and parts:
            x, line = 0, line + 1
        chunks = [it["text"]]
        if wide(it["text"], it["boxed"]) > width:
            chunks, cur = [], ""
            for word in it["text"].split():
                while wide(word, it["boxed"]) > width:
                    n = len(word)
                    while n > 1 and wide(word[:n], it["boxed"]) > width:
                        n -= 1
                    if cur:
                        chunks.append(cur)
                        cur = ""
                    chunks.append(word[:n])
                    word = word[n:]
                if cur and wide(cur + " " + word, it["boxed"]) > width:
                    chunks.append(cur)
                    cur = word
                else:
                    cur = (cur + " " + word).strip()
            if cur:
                chunks.append(cur)
        for c in chunks:
            w_ = wide(c, it["boxed"])
            if x > 0 and x + w_ > width:
                x, line = 0, line + 1
            parts.append({"item": i, "text": c, "x": x, "line": line, "w": w_})
            x += w_ + gap
    return parts


def box_pad(font) -> int:
    """A chip's padding round its words, for a font."""
    return max(2, round(font.size / 4))


def line_height(font) -> int:
    """One line of a legend: a chip's box (the font's own height and its padding) and a gap under it."""
    return round(font.size * 1.25) + 2 * box_pad(font) + max(2, font.size // 4)


def build(a: argparse.Namespace) -> None:
    if getattr(a, "candidate_table", "") or getattr(a, "candidate_joined", "") or getattr(a, "candidate_piece", ""):
        compare(a)
        return
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    asm = _assembler()
    # THE SIZES, BEFORE ANYTHING IS READ OR ENCODED (the blind read of 2026-10-16). The rows are the source's own size and the
    # encoder writes yuv420p, which holds no odd width or height (libx264 failed after every frame had been encoded). The clip is
    # the bottom row at that same size: until then a clip of another size (the assembler's `--size canvas`) was stretched to it
    # without a word, and the middle row marked the misregistration as changed everywhere there was texture.
    sw, sh, _ = probe(a.source)
    if sw % 2 or sh % 2:
        raise SystemExit(f"{a.source} is {sw} by {sh}: an odd width or height does not encode as yuv420p, and the stack's rows are the "
                         "source's own size; nothing written")
    jw, jh, total = probe(a.joined)
    if (jw, jh) != (sw, sh):
        raise SystemExit(f"{a.joined} is {jw} by {jh} and the source {a.source} is {sw} by {sh}: the stack's bottom row is the clip at the "
                         "source's own size and is never stretched to it. Build the clip at the source's size (`assemble_delivery.py --size "
                         "source`); nothing written")
    aliases = dict(x.split("=", 1) for x in a.alias)
    rows, intents = read_rows(a.table, a.copy_rate, aliases)
    if not rows:
        raise SystemExit(f"{a.table} lays no piece: there is nothing to stack beside the original")
    # EACH PIECE'S OWN CANVAS, from its mask picture (the lower half of its `_with_mask.mp4`), else the piece itself: its mask picture
    # is cut, its reference read and its zoom box placed at that size, as the assembler lays each piece at its own (the second blind
    # read of 2026-10-16: every row was cut at the first row's, and a wider piece lost its right side without a word). The right
    # column is as wide and tall as the largest; a piece of another shape sits in its tile at its own shape.
    canvas_of: dict[str, tuple[int, int]] = {}
    for r in rows:
        if Path(r["mask"]).is_file():
            mw_, mh_, _ = probe(r["mask"])
            canvas_of[Path(r["piece"]).name] = (mw_, mh_ // 2)
        elif Path(r["piece"]).is_file():
            canvas_of[Path(r["piece"]).name] = probe(r["piece"])[:2]
        else:
            raise SystemExit(f"{r['piece']}: neither it nor {Path(r['mask']).name} is there, so its canvas is not known; nothing written")
        r["canvas"] = canvas_of[Path(r["piece"]).name]
    w, h = max(c[0] for c in canvas_of.values()), max(c[1] for c in canvas_of.values())
    k = h / 768.0

    def font_px(px: int, bold: bool = True):
        for path in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(path, max(8, px))
            except OSError:
                pass
        return ImageFont.load_default()

    font = lambda size, bold=True: font_px(round(size * k), bold)  # noqa: E731

    for r in rows:
        print(f"{r['first']:5d}-{r['last']:<5d} {r['name']:34s} file starts at {r['piece_first']}, its graph at {r['graph_first']}; reference: "
              + (f"{Path(r['ref'][0]).name} from its frame {r['ref'][1]}" if r["ref"] else "none wired"), flush=True)
    # THE SOURCE FRAME THE CLIP BEGINS ON (`--first`): a clip of a span that does not begin on frame 0. Every frame below is a SOURCE
    # frame: the table's rows, the original read, the number burned in; the clip's own frame is that less `base`. Until 2026-10-14
    # a span clip was stacked as if it began on 0: the original and the masks rows were of other frames and the number was the
    # clip's own count (six per-shot clips, seen on two frames by a reader; the check had passed, reading the source from 0 too).
    base = int(getattr(a, "first", 0) or 0)
    first_f, last_f = (int(v) for v in a.only.split("-")) if a.only else (base, base + total - 1)
    if first_f < base or last_f > base + total - 1:
        raise SystemExit(f"--only {a.only} is not inside the clip's frames, {base} to {base + total - 1} of the source")
    cols = columns_of(rows, first_f, last_f)
    data = Path(getattr(a, "data", "") or HERE.parent / "data")
    given, refused = given_of_pieces(rows, data) if data.is_dir() else ({}, {})
    # the render's day, from the queued record that names the piece (never the file's time)
    days = render_days(rows, job_folders(data) if data.is_dir() else {})
    # what each load was given, its head crops and its sampled frames, all by the piece's FILE (two files of one load are two pieces)
    unread = [Path(r["piece"]).name for r in rows if Path(r["piece"]).name not in given]
    why_unread = {n: (f"its record could not be read: {refused[n].splitlines()[0][:160]}" if n in refused else
                      f"no queued record names its file under {data}" + ("" if data.is_dir() else " (the folder is not there: give --data)"))
                  for n in dict.fromkeys(unread)}
    for n, why in why_unread.items():
        print(f"what H3 was given is not read for {n}: {why}; its column shows its picture's graph alone", flush=True)
    # the joined clip's own check (`bench/assemble_delivery.py` writes it beside the clip): each flag on its frames, whole. Where it is
    # not read the legend says so on every frame: "no flag" is said only of flags that were read
    check_file = Path(a.joined + ".check.json")
    flags, flags_not_read = [], None
    if not check_file.is_file():
        flags_not_read = f"no {check_file.name} beside the clip"
    else:
        try:
            flags = json.loads(check_file.read_text()).get("flags") or []
        except ValueError as e:
            flags_not_read = f"{check_file.name} does not read ({e})"
    if flags_not_read:
        print(f"check flags not read: {flags_not_read}", flush=True)
    gaps_checked = bool(getattr(a, "coverage", ""))
    if not gaps_checked:
        print("gaps not checked: no --coverage; the legend says so on every frame", flush=True)
    import capture_masked_run as capm                          # the strip's own head crop and the node's region files, one rule each
    heads, sampled = {}, {}
    for file, g in given.items():
        v = next((x for x in g["inputs"] if x.get("tag") == "<Video 1>"), None)
        rf = ((v or {}).get("record") or {}).get("its_first_frame_is_source_frame")
        if v and capm is not None and rf is not None:
            cw_, ch_ = (v.get("loader") or {}).get("custom") or canvas_of[file]
            # the boxes are the reference's own frames, indexed by the reference frame a panel shows; None where it has no points
            got = capm.head_boxes(v, (cw_ or canvas_of[file][0], ch_ or canvas_of[file][1]), (rf, rf + (v.get("reference_frames") or 1) - 1))
            heads[file] = (got, (cw_ or canvas_of[file][0], ch_ or canvas_of[file][1]))
        for e in (v or {}).get("encoder_frames") or []:
            for x in e["sampled"]:
                if not x["held"] and x["source_frame"] is not None:
                    sampled.setdefault(file, set()).add(x["source_frame"])

    def notes_of(f: int, panels: list[dict]) -> list[tuple[str, tuple]]:
        out = [(t, (255, 235, 60)) for t in words_of(f, rows, intents, panels)]
        for r in rows:
            file = Path(r["piece"]).name
            if r["first"] <= f <= r["last"] and file in unread:
                out.append((f"{file}: what H3 was given is not read ({why_unread[file]}): its column shows its picture's graph", (255, 170, 60)))
        return out
    # the part of the source the loader kept, (w, h, x, y), for each canvas the pieces have
    crop_for = {c: asm.crop_of(sw, sh, *c) for c in sorted(set(canvas_of.values()))}
    # each zoomed piece's box at its true place in the source, by the piece's FILE (two files of one load are two pieces); a zoom
    # record that does not read is kept with its reason
    zooms, zoom_not_read = {}, {}
    for r in rows:
        file = Path(r["piece"]).name
        try:
            z = asm.zoomed_of(r["piece"], asm.probe(r["piece"]))
        except SystemExit as e:
            z = None
            zoom_not_read[file] = str(e)
            print(f"{file}: its zoom record is not read ({e}); no zoom box is drawn for it", flush=True)
        if z:
            crop = crop_for[canvas_of[file]]
            inside, _moved = asm.box_at_the_source(z["box"], crop, canvas_of[file])
            zooms[file] = {"box": [inside[0] + crop[2], inside[1] + crop[3], inside[2] + crop[2], inside[3] + crop[3]], "zoom": z.get("zoom")}
    day_of = lambda r: days[Path(r["piece"]).name][0]                # noqa: E731
    colour_of = {who: COLOURS[i % len(COLOURS)] for i, who in enumerate(cols)}
    # the per-frame gap check's runs (bench/frame_coverage.py), when handed in, as chips on their frames: every kind has words, and a
    # kind with none stops the build here, before anything is encoded
    chips_at: dict[int, list[tuple[str, str]]] = {}
    lost_at: dict[int, set[str]] = {}
    if getattr(a, "coverage", ""):
        runs = json.loads(Path(a.coverage).read_text()).get("runs") or []
        unknown = sorted({run["kind"] for run in runs} - set(CHIP_WORDS))
        if unknown:
            raise SystemExit(f"{a.coverage} has runs of kind {', '.join(unknown)}, which this tool has no words for (`CHIP_WORDS`): a chip "
                             "is not dropped without a word; nothing written")
        for run in runs:
            who = aliases.get(run["subject"], run["subject"])
            words_ = chip_words_of(run)
            for f in range(max(run["first"], first_f), min(run["last"], last_f) + 1):
                chips_at.setdefault(f, []).append((who, words_))
                if run["kind"] in LOST_KINDS:
                    lost_at.setdefault(f, set()).add(who)

    def laid_on(f: int) -> tuple[dict[str, dict], list[tuple[str, dict]]]:
        """The rows the TABLE lays on a frame, by subject (the first of a subject's), and any second row of a subject."""
        by, extra = {}, []
        for r in rows:
            if r["first"] <= f <= r["last"]:
                who = r["subject"] or r["name"]
                if who in by:
                    extra.append((who, r))
                else:
                    by[who] = r
        return by, extra

    WHITE, GREY, RED = (255, 255, 255), (170, 170, 170), (235, 50, 50)

    def lost_on(f: int) -> list[str]:
        return sorted(lost_at.get(f, set()))

    def legends_of(f: int) -> tuple[list[dict], list[dict], dict[str, str], list[dict]]:
        """A frame's two legends as items for `flow`, the who-is-laid words by subject, and the live check flags."""
        laid, _ = laid_on(f)
        who_words = {who: (f"{who}: {laid[who]['name']} ({day_of(laid[who])})" if who in laid else f"{who}: ORIGINAL") for who in cols}
        # a subject the table leaves as the original on purpose, with the table's reason
        purpose = []
        for i in intents:
            if i["first"] <= f <= i["last"]:
                words_ = f"{i['subject']}: ORIGINAL on purpose ({i['why']})"
                if i["subject"] in cols and i["subject"] not in laid:
                    who_words[i["subject"]] = words_
                else:
                    purpose.append(words_)
        one = [{"text": "ABOVE: ORIGINAL", "colour": WHITE, "boxed": False}, {"text": f"frame {f:05d}", "colour": WHITE, "boxed": False}]
        one += [{"text": who_words[who], "colour": colour_of[who] if who in laid else GREY, "boxed": False} for who in cols]
        one += [{"text": t, "colour": GREY, "boxed": False} for t in purpose]
        # legend 1's words as short as they allow: the view's height is held to level 5.1 (`most_height`)
        one.append({"text": "BELOW: WHAT WAS CHANGED (the assembler's test), edged white", "colour": (230, 230, 230), "boxed": False, "break": True})
        for c, crop in crop_for.items():
            if (crop[0], crop[1]) != (sw, sh):
                one.append({"text": "loader's crop" + (f" for {c[0]}x{c[1]}" if len(crop_for) > 1 else "")
                            + f": x {crop[2]}-{crop[2] + crop[0] - 1}, y {crop[3]}-{crop[3] + crop[1] - 1}", "colour": GREY, "boxed": False})
        for who, r in laid.items():
            z = zooms.get(Path(r["piece"]).name)
            if z:
                b = z["box"]
                one.append({"text": f"{r['name']}" + (f" {z['zoom']:.2f}x zoom" if z["zoom"] else " zoom") + f": x {b[0]}-{b[2] - 1}, "
                            f"y {b[1]}-{b[3] - 1}", "colour": colour_of.get(who, WHITE), "boxed": False})
        lost = lost_on(f)
        if lost:
            one.append({"text": f"RED BANDS: {', '.join(lost)} original while tracked", "colour": RED, "boxed": False})
        live = [fl for fl in flags if any(x <= f <= y for x, y in fl["source_frames"])]
        two = [{"text": f"{who}: {t}", "colour": colour_of.get(who, WHITE), "boxed": True, "chip": True} for who, t in chips_at.get(f, [])]
        two += [{"text": f"{fl['id']} {fl['rule']} ({fl.get('level') or 'level not recorded'})", "colour": (255, 150, 60), "boxed": True, "chip": True}
                for fl in live]
        # what was not read is said on every frame; "no gap or flag" only where both were read and neither has anything here
        if not gaps_checked:
            two.append({"text": "gaps not checked: no --coverage", "colour": (200, 160, 60), "boxed": True})
        if flags_not_read:
            two.append({"text": f"check flags not read: {flags_not_read}", "colour": (200, 160, 60), "boxed": True})
        two = two or [{"text": "no gap or flag on this frame", "colour": (110, 110, 110), "boxed": True}]
        two.append({"text": "BELOW: " + (a.tag or "THE CUT"), "colour": WHITE, "boxed": False, "break": True})
        return one, two, who_words, live
    # THE LEGENDS' HEIGHTS, one each for the whole clip: as many lines as the clip's busiest frame needs, so a chip is never cut
    leg_f = font(22 * sh / 1080 * 768 / h)                           # a size in pixels of a 1080-tall source, scaled with the source
    margin = max(4, round(10 * sh / 1080))
    gap = max(6, round(24 * sh / 1080))
    band = max(3, round(8 * sh / 1080))                              # the red band each legend gives the middle row
    pad = max(2, round(4 * sh / 1080))
    room = sw - 2 * margin
    laid_out = {}
    lines1 = lines2 = 1
    for f in range(first_f, last_f + 1):
        one, two, who_words, live = legends_of(f)
        p1, p2 = flow(one, leg_f, room, gap), flow(two, leg_f, room, max(4, gap // 2))
        laid_out[f] = (one, two, p1, p2, who_words, live)
        lines1, lines2 = max(lines1, p1[-1]["line"] + 1), max(lines2, p2[-1]["line"] + 1)
    lh = line_height(leg_f)
    lh1 = round(leg_f.size * 1.35)                                   # legend 1 has no boxes: its lines are as tight as the words
    L1 = 2 * ((pad + lines1 * lh1 + pad + band + 1) // 2)
    # WITHIN H.264 LEVEL 5.1 (`most_height`): the picture rows stay at the source's size and legend 1 is whole; legend 2 takes what is
    # left, and the chips of a frame that do not fit it go to the right column, below what H3 saw, said on their own frame
    W = sw + w
    W -= W % 2
    most = most_height(W)
    cap = (most - 3 * sh - L1 - band - 2 * pad) // lh                 # legend 2's lines within the level, its last "BELOW: THE CUT"
    if W > LEVEL_SIDE or cap < 3:
        raise SystemExit(f"the view would be {W} by at least {3 * sh + L1 + 2 * ((band + 2 * pad + 3 * lh + 1) // 2)} (three rows of {sh}, "
                         f"legend 1 of {L1}, legend 2 of three lines), over H.264 level {LEVEL}'s {W} by {most} ({LEVEL_MACROBLOCKS} macroblocks, "
                         f"no side over {LEVEL_SIDE}): a view a browser cannot play is never written; nothing written")
    overflow_at: dict[int, tuple[list[dict], list[dict]]] = {}
    over_lines = 0
    if lines2 > cap:
        keep = cap - 2                                                # chip lines kept, then the note, then "BELOW: THE CUT"
        rw_room = w - 16
        for f, (one, two, p1, p2, who_words, live) in laid_out.items():
            if p2[-1]["line"] + 1 <= cap:
                continue
            chips_ = two[:-1]
            parts_ = flow(chips_, leg_f, room, max(4, gap // 2))
            last_line = {}
            for part in parts_:
                last_line[part["item"]] = max(last_line.get(part["item"], 0), part["line"])
            stay = [it for i, it in enumerate(chips_) if last_line[i] < keep]
            moved = [it for i, it in enumerate(chips_) if last_line[i] >= keep]
            two_ = stay + [{"text": f"{len(moved)} more on this frame: in the right column, below what H3 saw", "colour": (200, 200, 200),
                            "boxed": False, "break": True}, two[-1]]
            items_o = [{"text": f"frame {f:05d}, legend 2 continued:", "colour": WHITE, "boxed": False}] + [{**it, "break": i == 0} for i, it in enumerate(moved)]
            parts_o = flow(items_o, leg_f, rw_room, max(4, gap // 2))
            overflow_at[f] = (items_o, parts_o)
            over_lines = max(over_lines, parts_o[-1]["line"] + 1)
            laid_out[f] = (one, two_, p1, flow(two_, leg_f, room, max(4, gap // 2)), who_words, live)
        lines2 = cap
    L2 = 2 * ((band + pad + lines2 * lh + pad + 1) // 2)
    total_h = 3 * sh + L1 + L2
    overflow_px = 16 + over_lines * lh if over_lines else 0           # the right column's rows under the blocks, for the overflow
    blocks_h = total_h - overflow_px

    def block_items(who: str, row: dict | None, mask_why: str | None, at_: int | None, has_crop: bool, ref_why: str | None,
                    bars: list[str], sc: float, colour: tuple, squares: list[tuple[str, tuple]] = ()) -> list[dict]:
        """A block's label as items for `flow`: whose it is, what H3 saw and the reference under it; then each bar of the clip's
        check on its piece (`piece_changes_nothing`), each on a line of its own, in the label strip and never over a picture."""
        if row is None:
            return [{"text": f"{who}: the original on this frame (no piece)", "colour": (200, 200, 200), "boxed": False}]
        name = row["name"]
        words = (f"{who}: " + (f"WHAT H3 SAW, {name}" if mask_why is None else f"{name}, what H3 saw: not read (why, below)")
                 + (f" (scaled {sc:.2f})" if sc < 1 else "")
                 + (f"; under it its reference, frame {at_}" + ("" if has_crop else " (no points: no head crop)") if at_ is not None else
                    f"; its reference: {ref_why}" if ref_why else
                    f"; its reference is NOT OPENED ({row['ref_refused']})" if row.get("ref_refused") else "; no reference wired"))
        return ([{"text": words, "colour": colour, "boxed": False}] + [{"text": t, "colour": WHITE, "boxed": True, "break": True, "bar": True} for t in bars]
                + [{"text": t, "colour": c, "boxed": True, "break": True} for t, c in squares])

    def bars_of(row: dict, live: list[dict]) -> list[str]:
        return [f"{fl['id']}: {row['name']} CHANGES 0 PX HERE: THE ORIGINAL SHOWS" for fl in live
                if fl["rule"] == "piece_changes_nothing" and (fl.get("piece") or "").startswith(row["name"])]

    def worst_items(who: str, row: dict, lab_f) -> int:
        """The most lines a block of this row's can need, for the strip's height: the mask picture not read, the longest of its
        reference words, every bar of its piece at once, and the scale shown."""
        bars = bars_of(row, flags)
        kept = kept_of[Path(row["piece"]).name]
        name_ = kept.windows[0]["file"] if kept.windows else ""
        worst = [squares_words(who, {"not_read": kept.why or "no window region file lays load frame 99999 (source frame 99999)"}),
                 squares_words(who, {"windows": {name_: {"inside": 99999, "outside": 99999, "taken": 99999}}, "not_split": None}),
                 squares_words(who, {"windows": {name_: {"inside": 99999, "outside": 99999, "taken": 99999}},
                                     "not_split": kept.track_why or "her track has no frame in this window"}),
                 squares_words(who, {"windows": {w_["file"]: {"inside": 99999, "outside": 99999, "taken": 99999} for w_ in kept.windows[:2]} or {"": {}},
                                     "not_split": None}) if len(kept.windows) > 1 else []]
        sq = max(worst, key=lambda x: len(x[0][0]) if x else 0)
        forms = [block_items(who, row, "not read", 99999, False, None, bars, 0.5, WHITE, sq)]
        if row["ref"]:
            forms.append(block_items(who, row, "not read", None, False, f"not read, {Path(row['ref'][0]).name} has no frame 99999", bars, 0.5, WHITE, sq))
        return max(flow(items, lab_f, w - 16, 4)[-1]["line"] + 1 for items in forms)
    # THE SQUARES KEPT AS THE ORIGINAL, from each render's own region files and its subject's own track (the owner's go, 2026-10-16)
    vm = capm._pack("video_mask")
    kept_of = {}
    for r in rows:
        file = Path(r["piece"]).name
        g = given.get(file)
        src_in = next((x for x in (g or {}).get("inputs") or [] if x.get("role") == "the source"), None)
        load_first = ((src_in or {}).get("loader") or {}).get("first_frame")
        origin = r["graph_first"] if load_first is None else load_first
        rendered = (r["piece_first"] - origin) + probe(r["piece"])[2] if Path(r["piece"]).is_file() else None   # the load frames the render holds
        kept_of[file] = KeptSquares(r, g, why_unread.get(file), capm, vm, origin, rendered)
        k_ = kept_of[file]
        print(f"{file}: squares kept as the original " + (f"not shown: {k_.why}" if k_.why else f"from {len(k_.windows)} window file(s)"
              + (f", not split: {k_.track_why}" if k_.track_why else f", split by her track{'s' if len(k_.tracks) > 1 else ''} " + ", ".join(Path(t[0]).name for t in k_.tracks))
              + (f"; left unread, another run's: {', '.join(k_.left_unread)}" if k_.left_unread else "")), flush=True)
    # THE LABEL STRIP, one height for the whole clip: as many lines as the longest a block's label can be, in a font scaled with
    # the blocks (`sc`); a larger strip scales the blocks down, which shrinks the font, until the strip holds the label
    chip_f = font(20 * sh / 1080 * 768 / h)
    label_px = 34
    for _ in range(8):
        sc_factor, blocks = right_tiles(w, h, len(cols), blocks_h, label_px)
        lab_f = font_px(round(chip_f.size * sc_factor))
        lines_ = max([worst_items(r["subject"] or r["name"], r, lab_f) for r in rows] + [1])
        need = 8 + lines_ * line_height(lab_f)
        if need <= label_px:
            break
        label_px = need
    sc = sc_factor
    rw = w                                                           # the right column is the largest canvas's own width
    blocks_end = max((t["block"][1] + t["block"][3] for t in blocks), default=0)
    if overflow_px and blocks_end > blocks_h:
        raise SystemExit(f"the right column ({rw} by {total_h}) cannot hold its {len(cols)} block(s) ({blocks_end} rows) and the chips legend 2 "
                         f"cannot hold ({overflow_px} rows) within H.264 level {LEVEL}; nothing written")
    if W * total_h > LEVEL_MACROBLOCKS * 256 or max(W, total_h) > LEVEL_SIDE:
        raise SystemExit(f"the view would be {W} by {total_h}, over H.264 level {LEVEL}; nothing written")

    def lay(d, items: list[dict], parts: list[dict], x0: int, y0: int, step: int) -> list[dict]:
        """Parts of a legend drawn from (x0, y0), a line every `step` rows; the chips as drawn, each with its box."""
        bp = box_pad(leg_f)
        boxes: dict[int, list[int]] = {}
        bottom = max(2, leg_f.size // 4)
        for part in parts:
            it = items[part["item"]]
            x, y = x0 + part["x"], y0 + part["line"] * step
            if it["boxed"]:
                d.rectangle((x, y, x + part["w"], y + step - bottom), fill=(40, 40, 40), outline=it["colour"], width=max(1, bp // 2))
            d.text((x + (bp if it["boxed"] else 0), y + (bp if it["boxed"] else 0)), part["text"], font=leg_f, fill=it["colour"])
            if it.get("chip"):
                b = boxes.setdefault(part["item"], [x, y, x + part["w"], y + step - bottom])
                b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x + part["w"]), max(b[3], y + step - bottom)
        return [{"words": items[i]["text"], "box": b} for i, b in sorted(boxes.items())]

    def legend(items: list[dict], parts: list[dict], height: int, top: int, red: bool, red_at: str, step: int) -> tuple[Image.Image, list[dict]]:
        """A legend strip the left column wide, its parts drawn from `top`, and the chips as drawn (their boxes in the strip)."""
        img = Image.new("RGB", (sw, height), (14, 14, 14))
        d = ImageDraw.Draw(img)
        drawn = lay(d, items, parts, margin, top, step)
        if red:
            d.rectangle((0, height - band, sw - 1, height - 1) if red_at == "bottom" else (0, 0, sw - 1, band - 1), fill=RED)
        return img, drawn

    def outline(img: np.ndarray, box: list[int], colour: tuple) -> None:
        """A 1 px outline of a box (x0, y0 inclusive, x1, y1 exclusive), drawn only clear of the picture's outer EDGE_CLEAR rows and columns."""
        x0, y0, x1, y1 = box[0], box[1], box[2] - 1, box[3] - 1
        lo_x, lo_y, hi_x, hi_y = EDGE_CLEAR, EDGE_CLEAR, sw - EDGE_CLEAR - 1, sh - EDGE_CLEAR - 1
        xa, xb, ya, yb = max(x0, lo_x), min(x1, hi_x), max(y0, lo_y), min(y1, hi_y)
        for y in (y0, y1):
            if lo_y <= y <= hi_y and xa <= xb:
                img[y, xa:xb + 1] = colour
        for x in (x0, x1):
            if lo_x <= x <= hi_x and ya <= yb:
                img[ya:yb + 1, x] = colour

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    silent = str(work / (Path(a.out).stem + ".video.mp4"))
    sidecar = Path(a.out).with_name(Path(a.out).stem + ".stack.json")   # the stack's own records; `<out>.json` is the video's sidecar
    tag = f"comment=stack_delivery layout=real source={sw}x{sh} canvas={w}x{h} legends={L1}+{L2} right={rw} sidecar={sidecar.name}"
    # the original and the clip, each read twice: as rgb24 for the rows, and as yuv420p for the assembler's own test of a change
    src, joined = Reader(a.source, first_f, "", sw, sh), Reader(a.joined, first_f - base, "", sw, sh)
    src_yuv, joined_yuv = Reader(a.source, first_f, "", sw, sh, "yuv420p"), Reader(a.joined, first_f - base, "", sw, sh, "yuv420p")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{total_h}", "-r", _rate(a.joined), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-level", LEVEL, "-preset", "faster", silent], stdin=subprocess.PIPE)
    from _lib.frames import OneASecond
    # the one-a-second parts are encoded under --work and moved beside --out only once both encoders have finished whole
    slow = OneASecond(work / Path(a.out).stem, (W, total_h), ["-metadata", tag, "-level", LEVEL]) if getattr(a, "one_a_second", False) else None
    stopped = None
    readers: dict[tuple, Reader] = {}
    missing: dict[str, list[int]] = {}
    records = []
    for f in range(first_f, last_f + 1):
        panels = panels_of(f, rows)
        keep = {(id(p["row"]), p["kind"]) for p in panels}
        for key in [key for key in readers if key not in keep]:
            readers.pop(key).close()
        # who is laid comes from the TABLE's rows on the frame; a panel that does not read is said in its column, never turns it to ORIGINAL
        laid, extra = laid_on(f)
        for who, r in extra:
            missing.setdefault(f"a second piece of {who} on one frame (the first is shown)", []).append(f)
        by_col: dict[str, dict] = {who: {"row": r} for who, r in laid.items()}
        for p in panels:
            key = (id(p["row"]), p["kind"])
            who = p["row"]["subject"] or p["row"]["name"]
            if by_col.get(who, {}).get("row") is not p["row"]:
                continue
            if key not in readers:
                cw_, ch_ = p["row"]["canvas"]
                readers[key] = Reader(p["file"], p["frame"], f"crop={cw_}:{ch_}:0:{ch_}" if p["kind"] == "mask" else f"scale={cw_}:{ch_}", cw_, ch_)
            got = readers[key].frame(p["frame"])
            if got is None:
                missing.setdefault(f"{p['row']['name']} {p['kind']}", []).append(f)
                by_col[who][p["kind"] + "_not_read"] = ("not read, " + (f"{Path(p['file']).name} is not there" if not Path(p["file"]).is_file()
                                                                        else f"{Path(p['file']).name} has no frame {p['frame']}"))
                continue
            by_col[who][p["kind"]] = (got, p["frame"])
        s, j = src.frame(f), joined.frame(f - base)
        sy, jy = src_yuv.frame(f), joined_yuv.frame(f - base)
        if s is None or j is None or sy is None or jy is None:
            raise SystemExit(f"frame {f}: the {'source' if s is None or sy is None else 'clip'} could not be read; nothing written")
        one, two, p1, p2, who_words, live_flags = laid_out[f]
        lost = lost_on(f)
        rec = {"source_frame": f, "clip_frame": f - base, "columns": {}, "check_flags": [fl["id"] for fl in live_flags],
               "chips": [f"{who}: {t}" for who, t in chips_at.get(f, [])], "legend": who_words, "red_bands": lost,
               "legend_1": [it["text"] for it in one], "legend_2": [it["text"] for it in two]}
        # the middle-left row: the source dimmed, the clip at full strength wherever the assembler's own test calls it changed
        # (`assemble_delivery.changed`, on both frames as yuv420p), its edge white; each zoomed piece's box as a 1 px outline
        hard = asm.changed(asm.planes(jy, sw, sh), asm.planes(sy, sw, sh)).astype(np.uint8)
        middle = (s.astype(np.int16) * 0.4).astype(np.uint8)
        middle[hard.astype(bool)] = j[hard.astype(bool)]
        middle[cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)] = (255, 255, 255)
        rec["changed_px"] = int(hard.sum())
        for who, r in laid.items():
            z = zooms.get(Path(r["piece"]).name)
            if z:
                outline(middle, z["box"], colour_of.get(who, WHITE))
        leg1, _ = legend(one, p1, L1, pad, bool(lost), "bottom", lh1)
        leg2, drawn = legend(two, p2, L2, band + pad, bool(lost), "top", lh)
        rec["chips_drawn"] = drawn
        # the right column: what H3 saw, a block a subject, at the canvas's own size (scaled only when the blocks do not fit)
        right = Image.new("RGB", (rw, total_h), (22, 24, 30))
        dr = ImageDraw.Draw(right)
        lh_ = line_height(lab_f)

        def put(img: np.ndarray, rect: list[int], interpolation) -> None:
            """A picture into a tile at its own shape, centred: a piece of another canvas than the column's is not stretched."""
            x0, y0, tw, th = rect
            ih, iw = img.shape[:2]
            f_ = min(tw / iw, th / ih)
            nw, nh = max(1, round(iw * f_)), max(1, round(ih * f_))
            right.paste(Image.fromarray(cv2.resize(img, (nw, nh), interpolation=interpolation)), (x0 + (tw - nw) // 2, y0 + (th - nh) // 2))

        def draw_label(items: list[dict], t: dict) -> tuple[list[str], list[list[int]]]:
            """A block's label in its strip, wrapped and never cut: its lines and each part's box, in the column's pixels."""
            lx, ly, lw, _lh = t["label"]
            lines, boxes = {}, []
            for part in flow(items, lab_f, lw, 4):
                it = items[part["item"]]
                x, y = lx + part["x"], ly + part["line"] * lh_
                box = [x, y, x + part["w"], y + lh_ - max(2, lab_f.size // 4)]
                if it.get("bar"):
                    dr.rectangle(tuple(box), fill=(120, 20, 20))
                dr.text((x + (box_pad(lab_f) if it["boxed"] else 0), y + box_pad(lab_f)), part["text"], font=lab_f, fill=it["colour"])
                lines.setdefault(part["line"], []).append(part["text"])
                boxes.append(box)
            return [" ".join(v) for _, v in sorted(lines.items())], boxes
        for who, t in zip(cols, blocks):
            slot = by_col.get(who)
            bx, by, bw, bh = t["block"]
            dr.rectangle((bx, by, bx + bw - 1, by + bh - 1), outline=colour_of[who], width=2)
            if slot is None:
                draw_label(block_items(who, None, None, None, False, None, [], sc, WHITE), t)
                rec["columns"][who] = None
                continue
            row = slot["row"]
            name, file = row["name"], Path(row["piece"]).name
            rcw, rch = row["canvas"]
            hb = heads.get(file)
            has_crop = bool(hb and hb[0])                       # head_boxes gives None where the reference has no points
            at_ = slot["reference"][1] if "reference" in slot else None
            kept = kept_of[file].on(f)
            items = block_items(who, row, slot.get("mask_not_read"), at_, has_crop, slot.get("reference_not_read"), bars_of(row, live_flags), sc,
                                colour_of[who], squares_words(row["subject"] or who, kept))
            label_lines, label_boxes = draw_label(items, t)
            col = {"piece": name, "file": file, "canvas": [rcw, rch], "label": items[0]["text"], "label_lines": label_lines, "label_boxes": label_boxes,
                   "kept_squares": ({"not_read": kept["not_read"]} if "not_read" in kept else
                                    {**(next(iter(kept["windows"].values())) if len(kept["windows"]) == 1 else {"windows": kept["windows"]}),
                                     "window": next(iter(kept["windows"])) if len(kept["windows"]) == 1 else None,
                                     "unit": ("a token (two latent cells a side)" if kept["draw"][0]["whole"] else "a latent cell") if kept["draw"] else None,
                                     "not_split": kept.get("not_split"), "not_laid": kept.get("not_laid"), "own_track_share": _own_share(capm)}),
                   "mask_frame": None, "mask_not_read": slot.get("mask_not_read"), "reference_frame": None,
                   "reference_not_read": slot.get("reference_not_read"), "reference_refused": row.get("ref_refused"),
                   "encoder_sample": f in sampled.get(file, set()), "zoom_box_at_source": (zooms.get(file) or {}).get("box"),
                   "check_flags": [fl["id"] for fl in live_flags if (fl.get("piece") or "").startswith(name)]}
            x0_, y0_, mw_, mh_ = t["mask"]
            if "mask" in slot:
                shown = slot["mask"][0]
                if "not_read" not in kept and kept["draw"] is not None:
                    shown = kept_of[file].paint(shown, kept["draw"])      # the squares kept as the original, filled, no line
                put(shown, t["mask"], cv2.INTER_AREA)
                col["mask_frame"] = slot["mask"][1]
            else:
                # in the mask picture's place: what H3 saw is not read, and why, wrapped inside the tile and never cut
                dr.rectangle((x0_, y0_, x0_ + mw_ - 1, y0_ + mh_ - 1), outline=(120, 120, 120), width=1)
                why_items = [{"text": f"what H3 saw: {slot['mask_not_read']}", "colour": (200, 200, 200), "boxed": False}]
                for part in flow(why_items, lab_f, mw_ - 16, 0):
                    dr.text((x0_ + 8, y0_ + 8 + part["line"] * lh_), part["text"], font=lab_f, fill=(200, 200, 200))
            if "reference" in slot:
                ref, at = slot["reference"]
                col["reference_frame"] = at
                _hx, ry, hs, _ = t["head"]
                if has_crop:
                    boxes = hb[0]
                    i = min(max(at, 0), len(boxes) - 1)
                    sx_, sy_ = rcw / hb[1][0], rch / hb[1][1]
                    bx0, by0, bx1, by1 = (round(boxes[i][0] * sx_), round(boxes[i][1] * sy_), round(boxes[i][2] * sx_), round(boxes[i][3] * sy_))
                    right.paste(Image.fromarray(cv2.resize(ref[by0:by1, bx0:bx1], (hs, hs), interpolation=cv2.INTER_CUBIC)), (8, ry))
                    if col["encoder_sample"]:
                        dr.rectangle((6, ry - 2, 8 + hs + 1, ry + hs + 1), outline=(90, 230, 110), width=3)
                put(ref, t["whole"], cv2.INTER_AREA)
            rec["columns"][who] = col
        if f in overflow_at:
            # the chips legend 2 cannot hold on this frame, in the right column below what H3 saw (level 5.1 holds the view's height)
            items_o, parts_o = overflow_at[f]
            rec["chips_overflow"] = lay(dr, items_o, parts_o, 8, blocks_end + 8, lh)
        rec["words"] = [t for t, _ in notes_of(f, panels)]
        records.append(rec)
        frame = Image.new("RGB", (W, total_h))
        frame.paste(Image.fromarray(s), (0, 0))
        frame.paste(leg1, (0, sh))
        frame.paste(Image.fromarray(middle), (0, sh + L1))
        frame.paste(leg2, (0, 2 * sh + L1))
        frame.paste(Image.fromarray(j), (0, 2 * sh + L1 + L2))
        frame.paste(right, (sw, 0))
        data_ = frame.tobytes()
        try:
            enc.stdin.write(data_)
        except BrokenPipeError:
            stopped = f"the encoder stopped taking frames at source frame {f}"
            break
        if slow is not None:
            slow.write(data_, f)
    try:
        enc.stdin.close()
    except BrokenPipeError:
        stopped = stopped or "the encoder stopped taking frames"
    slow_code, slow_parts = slow.close() if slow is not None else (0, [])
    code = enc.wait()
    for r in readers.values():
        r.close()
    for r in (src, joined, src_yuv, joined_yuv):
        r.close()
    for k_ in kept_of.values():
        k_.close()
    if stopped or code or slow_code:
        for part in work.glob(Path(a.out).stem + ".1fps.*"):
            part.unlink()
        if Path(silent).is_file():
            Path(silent).unlink()
        raise SystemExit(f"the encoder failed ({stopped or f'exit {code or slow_code}'}); nothing written beside {a.out}")
    # the index (`moov`) at the FRONT (`+faststart`): a browser starts the view without reading to its end (the 716 MB whole-video view
    # of 2026-10-16 did not start in the owner's browser with it at the end). The one-a-second parts are written so by `OneASecond`.
    # The sidecar below is written after this, so it is of the final bytes
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", a.joined, "-map", "0:v", "-map", "1:a?", "-c", "copy", "-metadata", tag,
                    "-movflags", "+faststart", a.out], check=True)
    Path(silent).unlink()
    import shutil
    for part in slow_parts:
        moved = Path(a.out).with_name(Path(part["file"]).name)
        shutil.move(part["file"], moved)
        part["file"] = str(moved)
    for prev, rec in zip([None] + records[:-1], records):
        ch = []
        if prev is not None:
            ch += [f"flag {x} starts" for x in rec["check_flags"] if x not in prev["check_flags"]]
            ch += [f"flag {x} ends" for x in prev["check_flags"] if x not in rec["check_flags"]]
            for who in cols:
                a_, b_ = (prev["columns"].get(who) or {}).get("piece"), (rec["columns"].get(who) or {}).get("piece")
                if a_ != b_:
                    ch.append(f"{who}: {b_ or 'the original'} from this frame" + (f" (was {a_})" if a_ else ""))
        rec["changes"] = ch
    sidecar.write_text(json.dumps({"clip": a.out, "one_a_second": slow_parts, "joined": a.joined, "table": a.table, "source": a.source,
                                   "frames": [first_f, last_f], "clip_begins_on_source_frame": base, "canvas": [w, h],
                                   "canvases": {file: list(c) for file, c in canvas_of.items()}, "source_size": [sw, sh], "legends": [L1, L2],
                                   "right_column": rw, "right_scale": sc, "label_strip": label_px, "level": LEVEL, "most_height": most,
                                   "legend_2_lines": lines2, "overflow_rows": overflow_px, "frames_overflowing": sorted(overflow_at),
                                   "loader_crops": {f"{c[0]}x{c[1]}": list(crop) for c, crop in crop_for.items()}, "zooms": zooms, "zoom_not_read": zoom_not_read,
                                   "gaps_checked": gaps_checked, "check_flags_not_read": flags_not_read,
                                   "kept_squares_read": {file: (k_.why or {"windows": [w_["file"] for w_ in k_.windows],
                                                                          "split_by": [t[0] for t in k_.tracks], "not_split": k_.track_why,
                                                                          "left_unread": k_.left_unread})
                                                         for file, k_ in kept_of.items()},
                                   "days": {name: {"day": d, "from": src_} for name, (d, src_) in days.items()},
                                   "colours": colour_of, "coverage": getattr(a, "coverage", "") or None,
                                   "columns": cols, "right_tiles": blocks,
                                   "source_is": ("the original the stack's top row shows and the assembler laid the pieces on (--source); each piece's own "
                                                 "graph names the file it was rendered from (given[piece].inputs, role 'the source'), which may be another "
                                                 "copy of the same clip"),
                                   "rendered_from": sorted({x["file"] for g in given.values() for x in g["inputs"] if x["role"] == "the source"}),
                                   "given": given, "given_not_read": sorted(set(unread)), "check_flags": flags,
                                   "records": Path(sidecar).with_suffix(".frames.jsonl").name}, indent=1, default=str) + "\n")
    with open(sidecar.with_suffix(".frames.jsonl"), "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    # the video's sidecar (the owner, 2026-10-16: every video says how it was made): this command, its inputs, and the joined
    # clip's pieces with their windows, carried from the clip's own sidecar where it has one. Written BEFORE success is said, and
    # any refusal or error of it is a hard failure naming the stack: a stack is never left without its sidecar in silence
    import sidecar as sidecar_tool
    try:
        pieces, carried = sidecar_tool.read(a.joined).get("pieces"), "from the joined clip's own sidecar"
    except (SystemExit, Exception) as e:
        pieces, carried = None, f"the joined clip's sidecar was not read, so its pieces are not carried ({type(e).__name__}: {e})"
    try:
        side = sidecar_tool.write(a.out, inputs={"the source": a.source, "the table": a.table, "the joined clip": a.joined,
                                       **({"the per-frame gap check": a.coverage} if getattr(a, "coverage", "") else {}),
                                       "the stack's records": str(sidecar), "the stack's per-frame records": str(sidecar.with_suffix(".frames.jsonl"))},
                        extra={"layout": "real (2026-10-16)", "source_size": [sw, sh], "canvas": [w, h], "legends": [L1, L2], "right_scale": sc_factor,
                               "one_a_second": slow_parts, "pieces_are": carried}, pieces=pieces)
    except (SystemExit, Exception) as e:
        raise SystemExit(f"THE STACK {a.out} IS WRITTEN BUT HAS NO SIDECAR: the sidecar tool refused or failed ({type(e).__name__}: {e}). It is not "
                         "to be handed over until its sidecar is written") from None
    print(f"written {a.out} ({last_f - first_f + 1} frames, {W} by {total_h}: the source's {sw} by {sh} rows, legends of {L1} and {L2}, "
          f"the canvas's {w} column)")
    for part in slow_parts:
        print(f"written {part['file']} (source frames {part['frames'][0]}-{part['frames'][1]} at one a second, no sound; {part['bytes']} bytes)")
    print(f"its records: {sidecar} and {sidecar.with_suffix('.frames.jsonl').name}")
    print(f"its sidecar: {side}")
    print("columns, one a subject for the whole clip:", ", ".join(cols))
    if zoom_not_read:
        print("zoom records not read, so no box is drawn for these pieces:", ", ".join(sorted(zoom_not_read)))
    print("panels or words that could not be shown, as first frame, last frame, count:", {name: (v[0], v[-1], len(v)) for name, v in missing.items()} or "none")


def _small(path: str, vf: str, n: int, first: int = 0) -> np.ndarray:
    """`n` frames of a file from its frame `first`, small and grey."""
    w, h = 256, 192
    r = Reader(path, first, vf + f",scale={w}:{h}", w, h, "gray")
    out = []
    for f in range(first, first + n):
        got = r.frame(f)
        if got is None:
            break
        out.append(got.astype(np.int16))
    r.close()
    return np.stack(out)


def fits(row: np.ndarray, ref: np.ndarray, cut: int = 0) -> dict:
    """A stacked row against the file it came from, both small and grey, less its top `cut` rows (the label a stack of the
    layout before 2026-10-16 drew there; none since): on how many frames its own frame fits best, ties a neighbour, or a
    neighbour fits better, and the mean differences."""
    n = min(len(row), len(ref))
    d = lambda step: np.array([np.abs(row[f, cut:] - ref[min(max(f + step, 0), n - 1), cut:]).mean() for f in range(n)])
    here, before, after = d(0), d(-1), d(1)
    if n > 1:
        before[0], after[-1] = after[0], before[-1]              # the ends have one neighbour
    other = np.minimum(before, after)
    return {"frames": n, "own_frame_fits_best": int((here < other).sum()), "ties_a_neighbour": int((here == other).sum()),
            "a_neighbour_fits_better": [int(f) for f in np.flatnonzero(here > other)], "mean_to_its_own_frame": round(float(here.mean()), 2),
            "greatest_to_its_own_frame": round(float(here.max()), 2), "mean_to_the_nearer_neighbour": round(float(other.mean()), 2), "cut": cut}


def _has_audio(path: str) -> bool:
    return bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
                                "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())


def _fps_float(path: str) -> float:
    r = _rate(path)
    if "/" in r:
        num, den = r.split("/", 1)
        return float(num) / max(float(den), 1e-9)
    return float(r or 24.0)


def _parse_only(text: str) -> tuple[int, int]:
    try:
        a, b = (int(v) for v in text.split("-"))
    except ValueError:
        raise SystemExit(f"--only {text!r} is not FIRST-LAST; nothing written") from None
    if a > b:
        raise SystemExit(f"--only {text}: the first frame is after the last; nothing written")
    return a, b


#: C71 note printed in `<stem>.alignment.txt` and carried in the alignment record so nobody reads a pass as more than it is.
ALIGNMENT_NOTE = ("Note: when two cuts differ by under a level on most pixels, own-vs-other margins are small "
                  "even where offset-0 frame alignment is sharp; read both figures as data, not as a binary verdict.")


def offsets_mae(row: np.ndarray, ref: np.ndarray, cut: int = 0) -> dict:
    """Mean grey difference between `row` and `ref` at frame offsets -2, -1, 0, +1, +2, and the offset that fits best."""
    n = min(len(row), len(ref))
    by_step = {}
    for step in (-2, -1, 0, 1, 2):
        key = f"{step:+d}" if step else "0"
        vals = [np.abs(row[f, cut:] - ref[min(max(f + step, 0), n - 1), cut:]).mean() for f in range(n)]
        by_step[key] = round(float(np.mean(vals)), 2) if vals else 0.0
    best = min((-2, -1, 0, 1, 2), key=lambda s: (by_step[f"{s:+d}" if s else "0"], abs(s)))
    return {"offsets": by_step, "best_offset": int(best)}


def compare_alignment(orig_row: np.ndarray, cur_row: np.ndarray, cand_row: np.ndarray,
                      src_ref: np.ndarray, cur_ref: np.ndarray, cand_ref: np.ndarray, first_frame: int = 0) -> dict:
    """Each row of a compare stack held against its named source at offsets -2..+2, and the current and candidate rows held
    against their own source and the other source on pixels where the two sources differ (`|candidate - current| >= 4`)."""
    n = min(len(orig_row), len(cur_row), len(cand_row), len(src_ref), len(cur_ref), len(cand_ref))
    orig_row, cur_row, cand_row = orig_row[:n], cur_row[:n], cand_row[:n]
    src_ref, cur_ref, cand_ref = src_ref[:n], cur_ref[:n], cand_ref[:n]
    diff = np.abs(cand_ref - cur_ref)
    mask = diff >= 4.0
    per_frame_px = mask.reshape(n, -1).sum(axis=1) if n else np.zeros(0, dtype=int)
    peak_i = int(np.argmax(per_frame_px)) if n else 0
    if not mask.any():
        mask = diff > 0
    if not mask.any():
        mask = np.ones_like(diff, dtype=bool)
    cur_own = round(float(np.abs(cur_row[mask] - cur_ref[mask]).mean()), 2) if n else 0.0
    cur_other = round(float(np.abs(cur_row[mask] - cand_ref[mask]).mean()), 2) if n else 0.0
    cand_own = round(float(np.abs(cand_row[mask] - cand_ref[mask]).mean()), 2) if n else 0.0
    cand_other = round(float(np.abs(cand_row[mask] - cur_ref[mask]).mean()), 2) if n else 0.0
    return {
        "note": ALIGNMENT_NOTE,
        "frames": n,
        "first_source_frame": int(first_frame),
        "peak_source_frame": int(first_frame + peak_i),
        "peak_clip_frame": int(peak_i),
        "peak_diff_pixels_ge_4": int(per_frame_px[peak_i]) if n else 0,
        "total_diff_pixels_ge_4": int(per_frame_px.sum()) if n else 0,
        "original": offsets_mae(orig_row, src_ref),
        "current": {**offsets_mae(cur_row, cur_ref), "vs_own": cur_own, "vs_other": cur_other},
        "candidate": {**offsets_mae(cand_row, cand_ref), "vs_own": cand_own, "vs_other": cand_other},
    }


def compare(a: argparse.Namespace) -> None:
    """Mode (c): compare a candidate cut/render against what is currently laid on the span, within H.264 level 5.1.
    Left column (3 rows at the source's own size + 2 legends): ORIGINAL, CURRENT CUT, CANDIDATE CUT.
    Right column: for each compared subject, CURRENT WHAT H3 SAW + REF and CANDIDATE WHAT H3 SAW + REF."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    asm = _assembler()
    sw, sh, src_total = probe(a.source)
    if sw % 2 or sh % 2:
        raise SystemExit(f"{a.source} is {sw} by {sh}: an odd width or height does not encode as yuv420p, and the stack's rows are the "
                         "source's own size; nothing written")
    jw, jh, total = probe(a.joined)
    if (jw, jh) != (sw, sh):
        raise SystemExit(f"{a.joined} is {jw} by {jh} and the source {a.source} is {sw} by {sh}: the stack's middle row is the current cut at the "
                         "source's own size and is never stretched to it. Build the clip at the source's size (`assemble_delivery.py --size "
                         "source`); nothing written")
    cand_table_arg = getattr(a, "candidate_table", "") or ""
    cand_joined_arg = getattr(a, "candidate_joined", "") or ""
    cand_piece_arg = getattr(a, "candidate_piece", "") or ""
    if not cand_table_arg and not cand_piece_arg:
        raise SystemExit("compare requires --candidate-table or --candidate-piece; nothing written")
    if cand_table_arg and cand_piece_arg:
        raise SystemExit("pass either --candidate-table or --candidate-piece, not both; nothing written")
    if cand_piece_arg and not Path(cand_piece_arg).is_file():
        raise SystemExit(f"{cand_piece_arg}: candidate piece is not a file; nothing written")
    if cand_joined_arg:
        cand_joined = cand_joined_arg
    elif cand_piece_arg:
        pw, ph, _ = probe(cand_piece_arg)
        if (pw, ph) != (sw, sh):
            raise SystemExit(f"{cand_piece_arg} is {pw} by {ph} and the source {a.source} is {sw} by {sh}: give --candidate-joined built at the "
                             "source's size (`assemble_delivery.py --size source`); nothing written")
        cand_joined = cand_piece_arg
    else:
        raise SystemExit(f"compare requires --candidate-joined at the source's size ({sw} by {sh}); nothing written")
    cjw, cjh, ctotal = probe(cand_joined)
    if (cjw, cjh) != (sw, sh):
        raise SystemExit(f"{cand_joined} is {cjw} by {cjh} and the source {a.source} is {sw} by {sh}: the stack's bottom row is the candidate cut "
                         "at the source's own size and is never stretched to it. Build the candidate clip at the source's size "
                         "(`assemble_delivery.py --size source`); nothing written")
    base = int(getattr(a, "first", 0) or 0)
    cand_first_raw = getattr(a, "candidate_first", None)
    if cand_first_raw is not None:
        cand_base = int(cand_first_raw)
    elif getattr(a, "candidate_piece_first", None) is not None and not cand_joined_arg:
        cand_base = int(a.candidate_piece_first)
    else:
        cand_base = base
    if getattr(a, "only", None):
        first_f, last_f = _parse_only(a.only)
    elif (base, base + total - 1) == (cand_base, cand_base + ctotal - 1):
        first_f, last_f = base, base + total - 1
    else:
        raise SystemExit(f"--joined covers {base} to {base + total - 1} and --candidate-joined covers {cand_base} to {cand_base + ctotal - 1}: "
                         "give --only FIRST-LAST for the span to compare; nothing written")
    if first_f < base or last_f > base + total - 1:
        raise SystemExit(f"--only {first_f}-{last_f} is not inside the current clip's frames, {base} to {base + total - 1} of the source")
    if first_f < cand_base or last_f > cand_base + ctotal - 1:
        raise SystemExit(f"--only {first_f}-{last_f} is not inside the candidate clip's frames, {cand_base} to {cand_base + ctotal - 1} of the source")
    if first_f < 0 or last_f >= src_total:
        raise SystemExit(f"frames {first_f}-{last_f} are not inside the source {a.source} (0 to {src_total - 1}); nothing written")
    aliases = dict(x.split("=", 1) for x in a.alias)
    rows, intents = read_rows(a.table, a.copy_rate, aliases)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    raw_subj = getattr(a, "subject", "") or ""
    subj_list = [aliases.get(s.strip(), s.strip()) for s in (raw_subj if isinstance(raw_subj, (list, tuple)) else str(raw_subj).split(",")) if s.strip()]
    if cand_table_arg:
        cand_table_path = cand_table_arg
        cand_rows, cand_intents = read_rows(cand_table_path, a.copy_rate, aliases)
    else:
        subj = subj_list[0] if subj_list else (rows[0]["subject"] if rows and rows[0].get("subject") else "a")
        subj_rows = [r for r in rows if (r["subject"] or r["name"]) == subj and r["first"] <= last_f and r["last"] >= first_f]
        cp_first = (int(a.candidate_piece_first) if getattr(a, "candidate_piece_first", None) is not None
                    else (subj_rows[0]["first"] if subj_rows and not getattr(a, "only", None) else first_f))
        p_frames = probe(cand_piece_arg)[2]
        cp_span_first = max(first_f, cp_first)
        cp_span_last = min(last_f, cp_first + p_frames - 1)
        if cp_span_first > cp_span_last:
            raise SystemExit(f"{cand_piece_arg} ({p_frames} frames from {cp_first}) does not overlap frames {first_f}-{last_f}; nothing written")
        synth_lines = [f"# synthesized candidate table for {Path(cand_piece_arg).name}"]
        for r in rows:
            r_who = r["subject"] or r["name"]
            if r_who != subj or r["last"] < cp_span_first or r["first"] > cp_span_last:
                restore_s = "".join(f" restore={x}" for x in r.get("less") or [])
                synth_lines.append(f"{r['first']}-{r['last']} {r['piece']} {r['piece_first']}"
                                   + (f" subject={r['subject']}" if r.get("subject") else "") + restore_s)
        synth_lines.append(f"{cp_span_first}-{cp_span_last} {Path(cand_piece_arg).resolve()} {cp_first} subject={subj}")
        for i in intents:
            synth_lines.append(f"source {i['first']}-{i['last']} subject={i['subject']} {i['why']}")
        synth_file = work / (Path(a.out).stem + ".candidate.table.txt")
        synth_file.write_text("\n".join(synth_lines) + "\n")
        cand_table_path = str(synth_file)
        cand_rows, cand_intents = read_rows(cand_table_path, a.copy_rate, aliases)
    if not rows and not cand_rows:
        raise SystemExit(f"{a.table} and {cand_table_path} lay no piece: there is nothing to compare")
    if not [r for r in cand_rows if r["first"] <= last_f and r["last"] >= first_f]:
        raise SystemExit(f"{cand_table_path} lays no piece on frames {first_f}-{last_f}: there is nothing to compare; nothing written")
    all_rows = rows + cand_rows
    canvas_of: dict[str, tuple[int, int]] = {}
    for r in all_rows:
        p_key = str(Path(r["piece"]).resolve())
        if Path(r["mask"]).is_file():
            mw_, mh_, _ = probe(r["mask"])
            canvas_of[p_key] = (mw_, mh_ // 2)
        elif Path(r["piece"]).is_file():
            canvas_of[p_key] = probe(r["piece"])[:2]
        else:
            raise SystemExit(f"{r['piece']}: neither it nor {Path(r['mask']).name} is there, so its canvas is not known; nothing written")
        r["canvas"] = canvas_of[p_key]
        r["key"] = p_key
    w, h = max(c[0] for c in canvas_of.values()), max(c[1] for c in canvas_of.values())
    k = h / 768.0

    def font_px(px: int, bold: bool = True):
        for path in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(path, max(8, px))
            except OSError:
                pass
        return ImageFont.load_default()

    font = lambda size, bold=True: font_px(round(size * k), bold)  # noqa: E731
    for tag_label, r_list in (("current", rows), ("candidate", cand_rows)):
        for r in r_list:
            print(f"[{tag_label:9s}] {r['first']:5d}-{r['last']:<5d} {r['name']:34s} file starts at {r['piece_first']}, its graph at {r['graph_first']}; reference: "
                  + (f"{Path(r['ref'][0]).name} from its frame {r['ref'][1]}" if r["ref"] else "none wired"), flush=True)
    cur_cols = columns_of(rows, first_f, last_f)
    cand_cols = columns_of(cand_rows, first_f, last_f)
    all_cols = list(dict.fromkeys(cur_cols + cand_cols))

    def active_sig(r_list: list[dict], who: str) -> list[tuple]:
        return [(max(r["first"], first_f), min(r["last"], last_f), r["key"], r["piece_first"], tuple(r["less"]))
                for r in r_list if (r["subject"] or r["name"]) == who and r["first"] <= last_f and r["last"] >= first_f]

    if subj_list:
        missing_subj = [s for s in subj_list if s not in all_cols]
        if missing_subj:
            raise SystemExit(f"--subject {missing_subj[0]!r} has no piece on frames {first_f}-{last_f} in either table; nothing written")
        compared_cols = list(dict.fromkeys(subj_list))
    else:
        compared_cols = [who for who in all_cols if active_sig(rows, who) != active_sig(cand_rows, who)] or list(all_cols)
    all_table_names = {r["name"] for r in rows + cand_rows if len(r["name"]) >= 2}
    for flag, side_name, tag_val, r_list in (
        ("--current-tag", "current", getattr(a, "current_tag", "") or "", rows),
        ("--tag", "candidate", getattr(a, "tag", "") or "", cand_rows),
    ):
        if not tag_val:
            continue
        span_rows = [r for r in r_list if r["first"] <= last_f and r["last"] >= first_f]
        span_names = list(dict.fromkeys(r["name"] for r in span_rows))
        for tok in re.findall(r"[A-Za-z0-9_.-]+", tag_val):
            if (tok in all_table_names or re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", tok)) and tok not in span_names:
                raise SystemExit(
                    f"{flag} {tag_val!r} names {tok!r}, which is not laid in the {side_name} table on frames "
                    f"{first_f}-{last_f} (laid: {', '.join(span_names) or 'none'}); nothing written"
                )
        for who in all_cols:
            who_rows = [r for r in span_rows if (r["subject"] or r["name"]) == who]
            who_names = list(dict.fromkeys(r["name"] for r in who_rows))
            mentioned = [n for n in who_names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", tag_val)]
            if mentioned and len(mentioned) < len(who_names):
                laid_desc = ", ".join(f"{r['name']} ({max(r['first'], first_f)}-{min(r['last'], last_f)})" for r in who_rows)
                missing_names = [n for n in who_names if n not in mentioned]
                raise SystemExit(
                    f"{flag} {tag_val!r} names only {', '.join(mentioned)} for {who} on frames {first_f}-{last_f}, "
                    f"leaving out {', '.join(missing_names)} (laid: {laid_desc}); name every piece laid for {who} on the span "
                    f"or pass a cut label; nothing written"
                )
    slots = [{"col_key": f"{side}:{who}", "side": side, "who": who,
              "title": f"{'CURRENT' if side == 'current' else 'CANDIDATE'} ({who})"}
             for who in compared_cols for side in ("current", "candidate")]
    data = Path(getattr(a, "data", "") or HERE.parent / "data")
    given, refused = given_of_pieces(all_rows, data) if data.is_dir() else ({}, {})
    days = render_days(all_rows, job_folders(data) if data.is_dir() else {})
    unread = [Path(r["piece"]).name for r in all_rows if Path(r["piece"]).name not in given]
    why_unread = {n: (f"its record could not be read: {refused[n].splitlines()[0][:160]}" if n in refused else
                      f"no queued record names its file under {data}" + ("" if data.is_dir() else " (the folder is not there: give --data)"))
                  for n in dict.fromkeys(unread)}
    for n, why in why_unread.items():
        print(f"what H3 was given is not read for {n}: {why}; its column shows its picture's graph alone", flush=True)
    check_file = Path(a.joined + ".check.json")
    cand_check_file = Path(cand_joined + ".check.json")
    flags, flags_not_read = [], None
    if not check_file.is_file() and not cand_check_file.is_file():
        flags_not_read = f"no {check_file.name} or {cand_check_file.name} beside the clips"
    else:
        for cf in (check_file, cand_check_file):
            if cf.is_file():
                try:
                    for fl in json.loads(cf.read_text()).get("flags") or []:
                        if fl not in flags:
                            flags.append(fl)
                except ValueError as e:
                    flags_not_read = f"{cf.name} does not read ({e})"
    if flags_not_read:
        print(f"check flags not read: {flags_not_read}", flush=True)
    gaps_checked = bool(getattr(a, "coverage", ""))
    if not gaps_checked:
        print("gaps not checked: no --coverage; the legend says so on every frame", flush=True)
    import capture_masked_run as capm
    heads, sampled = {}, {}
    for r in all_rows:
        file = Path(r["piece"]).name
        p_key = r["key"]
        g = given.get(file)
        if not g or p_key in heads:
            continue
        v = next((x for x in g["inputs"] if x.get("tag") == "<Video 1>"), None)
        rf = ((v or {}).get("record") or {}).get("its_first_frame_is_source_frame")
        if v and capm is not None and rf is not None:
            cw_, ch_ = (v.get("loader") or {}).get("custom") or canvas_of[p_key]
            got = capm.head_boxes(v, (cw_ or canvas_of[p_key][0], ch_ or canvas_of[p_key][1]), (rf, rf + (v.get("reference_frames") or 1) - 1))
            heads[p_key] = (got, (cw_ or canvas_of[p_key][0], ch_ or canvas_of[p_key][1]))
        for e in (v or {}).get("encoder_frames") or []:
            for x in e["sampled"]:
                if not x["held"] and x["source_frame"] is not None:
                    sampled.setdefault(p_key, set()).add(x["source_frame"])
    crop_for = {c: asm.crop_of(sw, sh, *c) for c in sorted(set(canvas_of.values()))}
    zooms, zoom_not_read = {}, {}
    for r in all_rows:
        file = Path(r["piece"]).name
        p_key = r["key"]
        try:
            z = asm.zoomed_of(r["piece"], asm.probe(r["piece"]))
        except SystemExit as e:
            z = None
            zoom_not_read[file] = str(e)
            print(f"{file}: its zoom record is not read ({e}); no zoom box is drawn for it", flush=True)
        if z:
            crop = crop_for[canvas_of[p_key]]
            inside, _moved = asm.box_at_the_source(z["box"], crop, canvas_of[p_key])
            zooms[file] = {"box": [inside[0] + crop[2], inside[1] + crop[3], inside[2] + crop[2], inside[3] + crop[3]], "zoom": z.get("zoom")}
    day_of = lambda r: days[Path(r["piece"]).name][0]  # noqa: E731
    colour_of = {who: COLOURS[i % len(COLOURS)] for i, who in enumerate(all_cols)}
    chips_at: dict[int, list[tuple[str, str]]] = {}
    lost_at: dict[int, set[str]] = {}
    if getattr(a, "coverage", ""):
        runs = json.loads(Path(a.coverage).read_text()).get("runs") or []
        unknown = sorted({run["kind"] for run in runs} - set(CHIP_WORDS))
        if unknown:
            raise SystemExit(f"{a.coverage} has runs of kind {', '.join(unknown)}, which this tool has no words for (`CHIP_WORDS`): a chip "
                             "is not dropped without a word; nothing written")
        for run in runs:
            who = aliases.get(run["subject"], run["subject"])
            words_ = chip_words_of(run)
            for f in range(max(run["first"], first_f), min(run["last"], last_f) + 1):
                chips_at.setdefault(f, []).append((who, words_))
                if run["kind"] in LOST_KINDS:
                    lost_at.setdefault(f, set()).add(who)

    def laid_on_list(r_list: list[dict], f: int) -> tuple[dict[str, dict], list[tuple[str, dict]]]:
        by, extra = {}, []
        for r in r_list:
            if r["first"] <= f <= r["last"]:
                who = r["subject"] or r["name"]
                if who in by:
                    extra.append((who, r))
                else:
                    by[who] = r
        return by, extra

    WHITE, GREY, RED = (255, 255, 255), (170, 170, 170), (235, 50, 50)

    def lost_on(f: int) -> list[str]:
        return sorted(lost_at.get(f, set()))

    cur_title = getattr(a, "current_tag", "") or "CURRENT CUT"
    cand_title = getattr(a, "tag", "") or "CANDIDATE CUT"

    def legends_of(f: int) -> tuple[list[dict], list[dict], dict[str, str], list[dict]]:
        cur_laid, _ = laid_on_list(rows, f)
        cand_laid, _ = laid_on_list(cand_rows, f)
        who_words = {}
        for who in all_cols:
            c_part = f"{cur_laid[who]['name']} ({day_of(cur_laid[who])})" if who in cur_laid else "ORIGINAL"
            n_part = f"{cand_laid[who]['name']} ({day_of(cand_laid[who])})" if who in cand_laid else "ORIGINAL"
            if who in compared_cols:
                who_words[who] = f"{who}: current {c_part} -> candidate {n_part}"
            else:
                who_words[who] = f"{who}: {c_part} (both)" if c_part == n_part else f"{who}: current {c_part} -> candidate {n_part}"
        purpose = []
        for i in intents + [x for x in cand_intents if x not in intents]:
            if i["first"] <= f <= i["last"]:
                words_ = f"{i['subject']}: ORIGINAL on purpose ({i['why']})"
                if words_ not in purpose:
                    purpose.append(words_)
        one = [{"text": "ABOVE: ORIGINAL", "colour": WHITE, "boxed": False}, {"text": f"frame {f:05d}", "colour": WHITE, "boxed": False}]
        one += [{"text": who_words[who], "colour": colour_of[who] if (who in cur_laid or who in cand_laid) else GREY, "boxed": False} for who in all_cols]
        one += [{"text": t, "colour": GREY, "boxed": False} for t in purpose]
        one.append({"text": "BELOW: " + cur_title, "colour": (230, 230, 230), "boxed": False, "break": True})
        for c, crop in crop_for.items():
            if (crop[0], crop[1]) != (sw, sh):
                one.append({"text": "loader's crop" + (f" for {c[0]}x{c[1]}" if len(crop_for) > 1 else "")
                            + f": x {crop[2]}-{crop[2] + crop[0] - 1}, y {crop[3]}-{crop[3] + crop[1] - 1}", "colour": GREY, "boxed": False})
        for who, r in cur_laid.items():
            z = zooms.get(Path(r["piece"]).name)
            if z:
                b = z["box"]
                one.append({"text": f"{r['name']}" + (f" {z['zoom']:.2f}x zoom" if z["zoom"] else " zoom") + f": x {b[0]}-{b[2] - 1}, "
                            f"y {b[1]}-{b[3] - 1}", "colour": colour_of.get(who, WHITE), "boxed": False})
        lost = lost_on(f)
        if lost:
            one.append({"text": f"RED BANDS: {', '.join(lost)} original while tracked", "colour": RED, "boxed": False})
        live = [fl for fl in flags if any(x <= f <= y for x, y in fl["source_frames"])]
        two = [{"text": f"{who}: {t}", "colour": colour_of.get(who, WHITE), "boxed": True, "chip": True} for who, t in chips_at.get(f, [])]
        two += [{"text": f"{fl['id']} {fl['rule']} ({fl.get('level') or 'level not recorded'})", "colour": (255, 150, 60), "boxed": True, "chip": True}
                for fl in live]
        if not gaps_checked:
            two.append({"text": "gaps not checked: no --coverage", "colour": (200, 160, 60), "boxed": True})
        if flags_not_read:
            two.append({"text": f"check flags not read: {flags_not_read}", "colour": (200, 160, 60), "boxed": True})
        two = two or [{"text": "no gap or flag on this frame", "colour": (110, 110, 110), "boxed": True}]
        two.append({"text": "BELOW: " + cand_title, "colour": WHITE, "boxed": False, "break": True})
        for who, r in cand_laid.items():
            z = zooms.get(Path(r["piece"]).name)
            if z:
                b = z["box"]
                two.append({"text": f"{r['name']}" + (f" {z['zoom']:.2f}x zoom" if z["zoom"] else " zoom") + f": x {b[0]}-{b[2] - 1}, "
                            f"y {b[1]}-{b[3] - 1}", "colour": colour_of.get(who, WHITE), "boxed": False})
        return one, two, who_words, live

    leg_f = font(22 * sh / 1080 * 768 / h)
    margin = max(4, round(10 * sh / 1080))
    gap = max(6, round(24 * sh / 1080))
    band = max(3, round(8 * sh / 1080))
    pad = max(2, round(4 * sh / 1080))
    room = sw - 2 * margin
    laid_out = {}
    lines1 = lines2 = 1
    for f in range(first_f, last_f + 1):
        one, two, who_words, live = legends_of(f)
        p1, p2 = flow(one, leg_f, room, gap), flow(two, leg_f, room, max(4, gap // 2))
        laid_out[f] = (one, two, p1, p2, who_words, live)
        lines1, lines2 = max(lines1, p1[-1]["line"] + 1), max(lines2, p2[-1]["line"] + 1)
    lh = line_height(leg_f)
    lh1 = round(leg_f.size * 1.35)
    L1 = 2 * ((pad + lines1 * lh1 + pad + band + 1) // 2)
    W = sw + w
    W -= W % 2
    most = most_height(W)
    cap = (most - 3 * sh - L1 - band - 2 * pad) // lh
    if W > LEVEL_SIDE or cap < 3:
        raise SystemExit(f"the view would be {W} by at least {3 * sh + L1 + 2 * ((band + 2 * pad + 3 * lh + 1) // 2)} (three rows of {sh}, "
                         f"legend 1 of {L1}, legend 2 of three lines), over H.264 level {LEVEL}'s {W} by {most} ({LEVEL_MACROBLOCKS} macroblocks, "
                         f"no side over {LEVEL_SIDE}): a view a browser cannot play is never written; nothing written")
    overflow_at: dict[int, tuple[list[dict], list[dict]]] = {}
    over_lines = 0
    if lines2 > cap:
        keep = cap - 2
        rw_room = w - 16
        for f, (one, two, p1, p2, who_words, live) in laid_out.items():
            if p2[-1]["line"] + 1 <= cap:
                continue
            tail_idx = next((i for i, it in enumerate(two) if it.get("text", "").startswith("BELOW: ")), len(two) - 1)
            chips_ = two[:tail_idx]
            tail_items = two[tail_idx:]
            parts_ = flow(chips_, leg_f, room, max(4, gap // 2))
            last_line = {}
            for part in parts_:
                last_line[part["item"]] = max(last_line.get(part["item"], 0), part["line"])
            stay = [it for i, it in enumerate(chips_) if last_line[i] < keep]
            moved = [it for i, it in enumerate(chips_) if last_line[i] >= keep]
            two_ = stay + [{"text": f"{len(moved)} more on this frame: in the right column, below what H3 saw", "colour": (200, 200, 200),
                            "boxed": False, "break": True}] + tail_items
            items_o = [{"text": f"frame {f:05d}, legend 2 continued:", "colour": WHITE, "boxed": False}] + [{**it, "break": i == 0} for i, it in enumerate(moved)]
            parts_o = flow(items_o, leg_f, rw_room, max(4, gap // 2))
            overflow_at[f] = (items_o, parts_o)
            over_lines = max(over_lines, parts_o[-1]["line"] + 1)
            laid_out[f] = (one, two_, p1, flow(two_, leg_f, room, max(4, gap // 2)), who_words, live)
        lines2 = cap
    L2 = 2 * ((band + pad + lines2 * lh + pad + 1) // 2)
    total_h = 3 * sh + L1 + L2
    overflow_px = 16 + over_lines * lh if over_lines else 0
    blocks_h = total_h - overflow_px

    def block_items(title: str, row: dict | None, mask_why: str | None, at_: int | None, has_crop: bool, ref_why: str | None,
                    bars: list[str], sc: float, colour: tuple, squares: list[tuple[str, tuple]] = ()) -> list[dict]:
        if row is None:
            return [{"text": f"{title}: the original on this frame (no piece)", "colour": (200, 200, 200), "boxed": False}]
        name = row["name"]
        words = (f"{title}: " + (f"WHAT H3 SAW, {name}" if mask_why is None else f"{name}, what H3 saw: not read (why, below)")
                 + (f" (scaled {sc:.2f})" if sc < 1 else "")
                 + (f"; under it its reference, frame {at_}" + ("" if has_crop else " (no points: no head crop)") if at_ is not None else
                    f"; its reference: {ref_why}" if ref_why else
                    f"; its reference is NOT OPENED ({row['ref_refused']})" if row.get("ref_refused") else "; no reference wired"))
        return ([{"text": words, "colour": colour, "boxed": False}] + [{"text": t, "colour": WHITE, "boxed": True, "break": True, "bar": True} for t in bars]
                + [{"text": t, "colour": c, "boxed": True, "break": True} for t, c in squares])

    def bars_of(row: dict, live: list[dict]) -> list[str]:
        return [f"{fl['id']}: {row['name']} CHANGES 0 PX HERE: THE ORIGINAL SHOWS" for fl in live
                if fl["rule"] == "piece_changes_nothing" and (fl.get("piece") or "").startswith(row["name"])]

    vm = capm._pack("video_mask")
    kept_of = {}
    for r in all_rows:
        file = Path(r["piece"]).name
        p_key = r["key"]
        if p_key in kept_of:
            continue
        g = given.get(file)
        src_in = next((x for x in (g or {}).get("inputs") or [] if x.get("role") == "the source"), None)
        load_first = ((src_in or {}).get("loader") or {}).get("first_frame")
        origin = r["graph_first"] if load_first is None else load_first
        rendered = (r["piece_first"] - origin) + probe(r["piece"])[2] if Path(r["piece"]).is_file() else None
        kept_of[p_key] = KeptSquares(r, g, why_unread.get(file), capm, vm, origin, rendered)
        k_ = kept_of[p_key]
        print(f"{file}: squares kept as the original " + (f"not shown: {k_.why}" if k_.why else f"from {len(k_.windows)} window file(s)"
              + (f", not split: {k_.track_why}" if k_.track_why else f", split by her track{'s' if len(k_.tracks) > 1 else ''} " + ", ".join(Path(t[0]).name for t in k_.tracks))
              + (f"; left unread, another run's: {', '.join(k_.left_unread)}" if k_.left_unread else "")), flush=True)

    def worst_items(title: str, who: str, row: dict, lab_f) -> int:
        bars = bars_of(row, flags)
        kept = kept_of[row["key"]]
        name_ = kept.windows[0]["file"] if kept.windows else ""
        worst = [squares_words(who, {"not_read": kept.why or "no window region file lays load frame 99999 (source frame 99999)"}),
                 squares_words(who, {"windows": {name_: {"inside": 99999, "outside": 99999, "taken": 99999}}, "not_split": None}),
                 squares_words(who, {"windows": {name_: {"inside": 99999, "outside": 99999, "taken": 99999}},
                                     "not_split": kept.track_why or "her track has no frame in this window"}),
                 squares_words(who, {"windows": {w_["file"]: {"inside": 99999, "outside": 99999, "taken": 99999} for w_ in kept.windows[:2]} or {"": {}},
                                     "not_split": None}) if len(kept.windows) > 1 else []]
        sq = max(worst, key=lambda x: len(x[0][0]) if x else 0)
        forms = [block_items(title, row, "not read", 99999, False, None, bars, 0.5, WHITE, sq)]
        if row["ref"]:
            forms.append(block_items(title, row, "not read", None, False, f"not read, {Path(row['ref'][0]).name} has no frame 99999", bars, 0.5, WHITE, sq))
        return max(flow(items, lab_f, w - 16, 4)[-1]["line"] + 1 for items in forms)

    chip_f = font(20 * sh / 1080 * 768 / h)
    label_px = 34
    for _ in range(8):
        sc_factor, blocks = right_tiles(w, h, len(slots), blocks_h, label_px)
        lab_f = font_px(round(chip_f.size * sc_factor))
        lines_ = max([worst_items(f"CANDIDATE ({r['subject'] or r['name']})", r["subject"] or r["name"], r, lab_f) for r in all_rows] + [1])
        need = 8 + lines_ * line_height(lab_f)
        if need <= label_px:
            break
        label_px = need
    sc = sc_factor
    rw = w
    blocks_end = max((t["block"][1] + t["block"][3] for t in blocks), default=0)
    if overflow_px and blocks_end > blocks_h:
        raise SystemExit(f"the right column ({rw} by {total_h}) cannot hold its {len(slots)} block(s) ({blocks_end} rows) and the chips legend 2 "
                         f"cannot hold ({overflow_px} rows) within H.264 level {LEVEL}; nothing written")
    if W * total_h > LEVEL_MACROBLOCKS * 256 or max(W, total_h) > LEVEL_SIDE:
        raise SystemExit(f"the view would be {W} by {total_h}, over H.264 level {LEVEL}; nothing written")

    def lay(d, items: list[dict], parts: list[dict], x0: int, y0: int, step: int) -> list[dict]:
        bp = box_pad(leg_f)
        boxes: dict[int, list[int]] = {}
        bottom = max(2, leg_f.size // 4)
        for part in parts:
            it = items[part["item"]]
            x, y = x0 + part["x"], y0 + part["line"] * step
            if it["boxed"]:
                d.rectangle((x, y, x + part["w"], y + step - bottom), fill=(40, 40, 40), outline=it["colour"], width=max(1, bp // 2))
            d.text((x + (bp if it["boxed"] else 0), y + (bp if it["boxed"] else 0)), part["text"], font=leg_f, fill=it["colour"])
            if it.get("chip"):
                b = boxes.setdefault(part["item"], [x, y, x + part["w"], y + step - bottom])
                b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x + part["w"]), max(b[3], y + step - bottom)
        return [{"words": items[i]["text"], "box": b} for i, b in sorted(boxes.items())]

    def legend(items: list[dict], parts: list[dict], height: int, top: int, red: bool, red_at: str, step: int) -> tuple[Image.Image, list[dict]]:
        img = Image.new("RGB", (sw, height), (14, 14, 14))
        d = ImageDraw.Draw(img)
        drawn = lay(d, items, parts, margin, top, step)
        if red:
            d.rectangle((0, height - band, sw - 1, height - 1) if red_at == "bottom" else (0, 0, sw - 1, band - 1), fill=RED)
        return img, drawn

    silent = str(work / (Path(a.out).stem + ".video.mp4"))
    sidecar = Path(a.out).with_name(Path(a.out).stem + ".stack.json")
    alignment_file = Path(a.out).with_name(Path(a.out).stem + ".alignment.txt")
    tag = f"comment=stack_delivery layout=compare source={sw}x{sh} canvas={w}x{h} legends={L1}+{L2} right={rw} sidecar={sidecar.name}"
    src = Reader(a.source, first_f, "", sw, sh)
    joined = Reader(a.joined, first_f - base, "", sw, sh)
    cand_reader = Reader(cand_joined, first_f - cand_base, "", sw, sh)
    joined_yuv = Reader(a.joined, first_f - base, "", sw, sh, "yuv420p")
    cand_yuv = Reader(cand_joined, first_f - cand_base, "", sw, sh, "yuv420p")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{total_h}", "-r", _rate(a.joined), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-level", LEVEL, "-preset", "faster", silent], stdin=subprocess.PIPE)
    from _lib.frames import OneASecond
    slow = OneASecond(work / Path(a.out).stem, (W, total_h), ["-metadata", tag, "-level", LEVEL]) if getattr(a, "one_a_second", False) else None
    stopped = None
    readers: dict[tuple, Reader] = {}
    missing: dict[str, list[int]] = {}
    records = []
    for f in range(first_f, last_f + 1):
        cur_panels = panels_of(f, rows)
        cand_panels = panels_of(f, cand_rows)
        all_panels = cur_panels + cand_panels
        keep = {(id(p["row"]), p["kind"]) for p in all_panels}
        for key in [key for key in readers if key not in keep]:
            readers.pop(key).close()
        cur_laid, cur_extra = laid_on_list(rows, f)
        cand_laid, cand_extra = laid_on_list(cand_rows, f)
        for side_name, extra in (("current", cur_extra), ("candidate", cand_extra)):
            for who, _r in extra:
                missing.setdefault(f"a second piece of {who} on one frame in {side_name} (the first is shown)", []).append(f)
        by_slot: dict[str, dict] = {}
        for slot in slots:
            who = slot["who"]
            laid_map = cur_laid if slot["side"] == "current" else cand_laid
            if who in laid_map:
                by_slot[slot["col_key"]] = {"row": laid_map[who]}
        for side_name, p_list in (("current", cur_panels), ("candidate", cand_panels)):
            for p in p_list:
                who = p["row"]["subject"] or p["row"]["name"]
                col_key = f"{side_name}:{who}"
                if by_slot.get(col_key, {}).get("row") is not p["row"]:
                    continue
                key = (id(p["row"]), p["kind"])
                if key not in readers:
                    cw_, ch_ = p["row"]["canvas"]
                    readers[key] = Reader(p["file"], p["frame"], f"crop={cw_}:{ch_}:0:{ch_}" if p["kind"] == "mask" else f"scale={cw_}:{ch_}", cw_, ch_)
                got = readers[key].frame(p["frame"])
                if got is None:
                    missing.setdefault(f"{p['row']['name']} {p['kind']}", []).append(f)
                    by_slot[col_key][p["kind"] + "_not_read"] = ("not read, " + (f"{Path(p['file']).name} is not there" if not Path(p["file"]).is_file()
                                                                                 else f"{Path(p['file']).name} has no frame {p['frame']}"))
                    continue
                by_slot[col_key][p["kind"]] = (got, p["frame"])
        s, j, c = src.frame(f), joined.frame(f - base), cand_reader.frame(f - cand_base)
        jy, cy = joined_yuv.frame(f - base), cand_yuv.frame(f - cand_base)
        if s is None or j is None or c is None or jy is None or cy is None:
            which = "source" if s is None else ("current clip" if j is None or jy is None else "candidate clip")
            raise SystemExit(f"frame {f}: the {which} could not be read; nothing written")
        one, two, p1, p2, who_words, live_flags = laid_out[f]
        lost = lost_on(f)
        hard_diff = asm.changed(asm.planes(cy, sw, sh), asm.planes(jy, sw, sh)).astype(np.uint8)
        abs_diff = np.abs(c.astype(np.int16) - j.astype(np.int16))
        rec = {"source_frame": f, "clip_frame": f - base, "candidate_clip_frame": f - cand_base, "columns": {},
               "check_flags": [fl["id"] for fl in live_flags], "chips": [f"{who}: {t}" for who, t in chips_at.get(f, [])],
               "legend": who_words, "red_bands": lost, "legend_1": [it["text"] for it in one], "legend_2": [it["text"] for it in two],
               "changed_px": int(hard_diff.sum()),
               "diff_vs_current": {"changed_px": int(hard_diff.sum()), "mean": round(float(abs_diff.mean()), 3), "max": int(abs_diff.max())}}
        leg1, _ = legend(one, p1, L1, pad, bool(lost), "bottom", lh1)
        leg2, drawn = legend(two, p2, L2, band + pad, bool(lost), "top", lh)
        rec["chips_drawn"] = drawn
        right = Image.new("RGB", (rw, total_h), (22, 24, 30))
        dr = ImageDraw.Draw(right)
        lh_ = line_height(lab_f)

        def put(img: np.ndarray, rect: list[int], interpolation) -> None:
            x0, y0, tw, th = rect
            ih, iw = img.shape[:2]
            f_ = min(tw / iw, th / ih)
            nw, nh = max(1, round(iw * f_)), max(1, round(ih * f_))
            right.paste(Image.fromarray(cv2.resize(img, (nw, nh), interpolation=interpolation)), (x0 + (tw - nw) // 2, y0 + (th - nh) // 2))

        def draw_label(items: list[dict], t: dict) -> tuple[list[str], list[list[int]]]:
            lx, ly, lw, _lh = t["label"]
            lines, boxes = {}, []
            for part in flow(items, lab_f, lw, 4):
                it = items[part["item"]]
                x, y = lx + part["x"], ly + part["line"] * lh_
                box = [x, y, x + part["w"], y + lh_ - max(2, lab_f.size // 4)]
                if it.get("bar"):
                    dr.rectangle(tuple(box), fill=(120, 20, 20))
                dr.text((x + (box_pad(lab_f) if it["boxed"] else 0), y + box_pad(lab_f)), part["text"], font=lab_f, fill=it["colour"])
                lines.setdefault(part["line"], []).append(part["text"])
                boxes.append(box)
            return [" ".join(v) for _, v in sorted(lines.items())], boxes

        for slot_meta, t in zip(slots, blocks):
            col_key, who, title = slot_meta["col_key"], slot_meta["who"], slot_meta["title"]
            slot = by_slot.get(col_key)
            bx, by, bw, bh = t["block"]
            dr.rectangle((bx, by, bx + bw - 1, by + bh - 1), outline=colour_of[who], width=2)
            if slot is None:
                draw_label(block_items(title, None, None, None, False, None, [], sc, WHITE), t)
                rec["columns"][col_key] = None
                continue
            row = slot["row"]
            name, file, p_key = row["name"], Path(row["piece"]).name, row["key"]
            rcw, rch = row["canvas"]
            hb = heads.get(p_key)
            has_crop = bool(hb and hb[0])
            at_ = slot["reference"][1] if "reference" in slot else None
            kept = kept_of[p_key].on(f)
            items = block_items(title, row, slot.get("mask_not_read"), at_, has_crop, slot.get("reference_not_read"), bars_of(row, live_flags), sc,
                                colour_of[who], squares_words(row["subject"] or who, kept))
            label_lines, label_boxes = draw_label(items, t)
            col = {"side": slot_meta["side"], "subject": who, "piece": name, "file": file, "canvas": [rcw, rch],
                   "label": items[0]["text"], "label_lines": label_lines, "label_boxes": label_boxes,
                   "kept_squares": ({"not_read": kept["not_read"]} if "not_read" in kept else
                                    {**(next(iter(kept["windows"].values())) if len(kept["windows"]) == 1 else {"windows": kept["windows"]}),
                                     "window": next(iter(kept["windows"])) if len(kept["windows"]) == 1 else None,
                                     "unit": ("a token (two latent cells a side)" if kept["draw"][0]["whole"] else "a latent cell") if kept["draw"] else None,
                                     "not_split": kept.get("not_split"), "not_laid": kept.get("not_laid"), "own_track_share": _own_share(capm)}),
                   "mask_frame": None, "mask_not_read": slot.get("mask_not_read"), "reference_frame": None,
                   "reference_not_read": slot.get("reference_not_read"), "reference_refused": row.get("ref_refused"),
                   "encoder_sample": f in sampled.get(p_key, set()), "zoom_box_at_source": (zooms.get(file) or {}).get("box"),
                   "check_flags": [fl["id"] for fl in live_flags if (fl.get("piece") or "").startswith(name)]}
            x0_, y0_, mw_, mh_ = t["mask"]
            if "mask" in slot:
                shown = slot["mask"][0]
                if "not_read" not in kept and kept["draw"] is not None:
                    shown = kept_of[p_key].paint(shown, kept["draw"])
                put(shown, t["mask"], cv2.INTER_AREA)
                col["mask_frame"] = slot["mask"][1]
            else:
                dr.rectangle((x0_, y0_, x0_ + mw_ - 1, y0_ + mh_ - 1), outline=(120, 120, 120), width=1)
                why_items = [{"text": f"what H3 saw: {slot['mask_not_read']}", "colour": (200, 200, 200), "boxed": False}]
                for part in flow(why_items, lab_f, mw_ - 16, 0):
                    dr.text((x0_ + 8, y0_ + 8 + part["line"] * lh_), part["text"], font=lab_f, fill=(200, 200, 200))
            if "reference" in slot:
                ref, at = slot["reference"]
                col["reference_frame"] = at
                _hx, ry, hs, _ = t["head"]
                if has_crop:
                    boxes = hb[0]
                    i = min(max(at, 0), len(boxes) - 1)
                    sx_, sy_ = rcw / hb[1][0], rch / hb[1][1]
                    bx0, by0, bx1, by1 = (round(boxes[i][0] * sx_), round(boxes[i][1] * sy_), round(boxes[i][2] * sx_), round(boxes[i][3] * sy_))
                    right.paste(Image.fromarray(cv2.resize(ref[by0:by1, bx0:bx1], (hs, hs), interpolation=cv2.INTER_CUBIC)), (8, ry))
                    if col["encoder_sample"]:
                        dr.rectangle((6, ry - 2, 8 + hs + 1, ry + hs + 1), outline=(90, 230, 110), width=3)
                put(ref, t["whole"], cv2.INTER_AREA)
            rec["columns"][col_key] = col
        if f in overflow_at:
            items_o, parts_o = overflow_at[f]
            rec["chips_overflow"] = lay(dr, items_o, parts_o, 8, blocks_end + 8, lh)
        rec["words"] = (["CURRENT: " + w_ for w_ in words_of(f, rows, intents, cur_panels)]
                        + ["CANDIDATE: " + w_ for w_ in words_of(f, cand_rows, cand_intents, cand_panels)])
        records.append(rec)
        frame = Image.new("RGB", (W, total_h))
        frame.paste(Image.fromarray(s), (0, 0))
        frame.paste(leg1, (0, sh))
        frame.paste(Image.fromarray(j), (0, sh + L1))
        frame.paste(leg2, (0, 2 * sh + L1))
        frame.paste(Image.fromarray(c), (0, 2 * sh + L1 + L2))
        frame.paste(right, (sw, 0))
        data_ = frame.tobytes()
        try:
            enc.stdin.write(data_)
        except BrokenPipeError:
            stopped = f"the encoder stopped taking frames at source frame {f}"
            break
        if slow is not None:
            slow.write(data_, f)
    try:
        enc.stdin.close()
    except BrokenPipeError:
        stopped = stopped or "the encoder stopped taking frames"
    slow_code, slow_parts = slow.close() if slow is not None else (0, [])
    code = enc.wait()
    for r in readers.values():
        r.close()
    for r in (src, joined, cand_reader, joined_yuv, cand_yuv):
        r.close()
    for k_ in kept_of.values():
        k_.close()
    if stopped or code or slow_code:
        for part in work.glob(Path(a.out).stem + ".1fps.*"):
            part.unlink()
        if Path(silent).is_file():
            Path(silent).unlink()
        raise SystemExit(f"the encoder failed ({stopped or f'exit {code or slow_code}'}); nothing written beside {a.out}")
    if (first_f, last_f) == (base, base + total - 1) and _has_audio(a.joined):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", a.joined, "-map", "0:v", "-map", "1:a?", "-c", "copy",
                        "-metadata", tag, "-movflags", "+faststart", a.out], check=True)
    else:
        audio_pick = next(((p, f0_) for p, f0_ in ((a.source, first_f), (a.joined, first_f - base), (cand_joined, first_f - cand_base)) if _has_audio(p)), None)
        if audio_pick is not None:
            audio_src, audio_f0 = audio_pick
            fps = _fps_float(audio_src)
            start_s = max(0.0, audio_f0 / fps)
            dur_s = (last_f - first_f + 1) / fps
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}", "-i", audio_src,
                            "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-metadata", tag, "-movflags", "+faststart", a.out], check=True)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-c", "copy", "-metadata", tag, "-movflags", "+faststart", a.out], check=True)
    Path(silent).unlink()
    import shutil
    for part in slow_parts:
        moved = Path(a.out).with_name(Path(part["file"]).name)
        shutil.move(part["file"], moved)
        part["file"] = str(moved)
    col_keys = [s["col_key"] for s in slots]
    for prev, rec in zip([None] + records[:-1], records):
        ch = []
        if prev is not None:
            ch += [f"flag {x} starts" for x in rec["check_flags"] if x not in prev["check_flags"]]
            ch += [f"flag {x} ends" for x in prev["check_flags"] if x not in rec["check_flags"]]
            for ck in col_keys:
                a_, b_ = (prev["columns"].get(ck) or {}).get("piece"), (rec["columns"].get(ck) or {}).get("piece")
                if a_ != b_:
                    ch.append(f"{ck}: {b_ or 'the original'} from this frame" + (f" (was {a_})" if a_ else ""))
        rec["changes"] = ch
    stack_payload = {"layout": "compare", "clip": a.out, "one_a_second": slow_parts,
                     "joined": a.joined, "table": a.table, "candidate_joined": cand_joined, "candidate_table": cand_table_path,
                     "source": a.source, "frames": [first_f, last_f], "clip_begins_on_source_frame": base,
                     "candidate_begins_on_source_frame": cand_base, "canvas": [w, h],
                     "canvases": {Path(r["piece"]).name: list(r["canvas"]) for r in all_rows},
                     "source_size": [sw, sh], "legends": [L1, L2], "right_column": rw, "right_scale": sc, "label_strip": label_px,
                     "level": LEVEL, "most_height": most, "legend_2_lines": lines2, "overflow_rows": overflow_px,
                     "frames_overflowing": sorted(overflow_at), "loader_crops": {f"{c[0]}x{c[1]}": list(crop) for c, crop in crop_for.items()},
                     "zooms": zooms, "zoom_not_read": zoom_not_read, "gaps_checked": gaps_checked, "check_flags_not_read": flags_not_read,
                     "kept_squares_read": {Path(r["piece"]).name: (kept_of[r["key"]].why or {"windows": [w_["file"] for w_ in kept_of[r["key"]].windows],
                                                                                             "split_by": [t[0] for t in kept_of[r["key"]].tracks],
                                                                                             "not_split": kept_of[r["key"]].track_why,
                                                                                             "left_unread": kept_of[r["key"]].left_unread})
                                           for r in all_rows},
                     "days": {name: {"day": d, "from": src_} for name, (d, src_) in days.items()},
                     "colours": colour_of, "coverage": getattr(a, "coverage", "") or None,
                     "compared_subjects": compared_cols, "unchanged_subjects": [who for who in all_cols if who not in compared_cols],
                     "columns": col_keys, "right_tiles": blocks,
                     "source_is": ("the original the stack's top row shows and the assembler laid the pieces on (--source); the middle row is "
                                   "--joined (current cut) and the bottom row is --candidate-joined (candidate cut)"),
                     "rendered_from": sorted({x["file"] for g in given.values() for x in g["inputs"] if x["role"] == "the source"}),
                     "given": given, "given_not_read": sorted(set(unread)), "check_flags": flags,
                     "records": Path(sidecar).with_suffix(".frames.jsonl").name}
    sidecar.write_text(json.dumps(stack_payload, indent=1, default=str) + "\n")
    with open(sidecar.with_suffix(".frames.jsonl"), "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    got = read_back(a.source, a.joined, a.out, base, cand_joined, cand_base)
    stack_payload["alignment"] = got["alignment"]
    stack_payload["alignment_file"] = alignment_file.name
    sidecar.write_text(json.dumps(stack_payload, indent=1, default=str) + "\n")
    alignment_file.write_text("\n".join(check_lines(got, base)) + "\n")
    import sidecar as sidecar_tool
    pieces_all = []
    carried_parts = []
    for label_s, clip_p in (("current", a.joined), ("candidate", cand_joined)):
        try:
            ps = sidecar_tool.read(clip_p).get("pieces")
            if ps:
                pieces_all.extend(ps)
            carried_parts.append(f"{label_s}: from {Path(clip_p).name}'s sidecar")
        except (SystemExit, Exception) as e:
            carried_parts.append(f"{label_s}: sidecar not read ({type(e).__name__}: {e})")
    try:
        side = sidecar_tool.write(
            a.out,
            inputs={
                "the source": a.source,
                "the table": a.table,
                "the joined clip": a.joined,
                "the candidate table": cand_table_path,
                "the candidate clip": cand_joined,
                **({"the candidate piece": cand_piece_arg} if cand_piece_arg else {}),
                **({"the per-frame gap check": a.coverage} if getattr(a, "coverage", "") else {}),
                "the stack's records": str(sidecar),
                "the stack's per-frame records": str(sidecar.with_suffix(".frames.jsonl")),
                "the stack's alignment check": str(alignment_file),
            },
            extra={
                "layout": "compare (mode c, 2026-10-17)",
                "frames": [first_f, last_f],
                "source_size": [sw, sh],
                "canvas": [w, h],
                "legends": [L1, L2],
                "right_scale": sc_factor,
                "compared_subjects": compared_cols,
                "columns": col_keys,
                "alignment": got["alignment"],
                "one_a_second": slow_parts,
                "pieces_are": "; ".join(carried_parts),
            },
            pieces=pieces_all or None,
        )
    except (SystemExit, Exception) as e:
        raise SystemExit(f"THE STACK {a.out} IS WRITTEN BUT HAS NO SIDECAR: the sidecar tool refused or failed ({type(e).__name__}: {e}). It is not "
                         "to be handed over until its sidecar is written") from None
    print(f"written {a.out} ({last_f - first_f + 1} frames, {W} by {total_h}: the source's {sw} by {sh} rows, legends of {L1} and {L2}, "
          f"the canvas's {w} column)")
    for part in slow_parts:
        print(f"written {part['file']} (source frames {part['frames'][0]}-{part['frames'][1]} at one a second, no sound; {part['bytes']} bytes)")
    print(f"its records: {sidecar}, {sidecar.with_suffix('.frames.jsonl').name} and {alignment_file.name}")
    print(f"its sidecar: {side}")
    print("columns:", ", ".join(col_keys))
    if zoom_not_read:
        print("zoom records not read, so no box is drawn for these pieces:", ", ".join(sorted(zoom_not_read)))
    print("panels or words that could not be shown, as first frame, last frame, count:", {name: (v[0], v[-1], len(v)) for name, v in missing.items()} or "none")


def read_back(source: str, joined: str, out: str, first: int = 0, candidate_joined: str | None = None, candidate_first: int | None = None) -> dict:
    """`first` is the source frame the clip begins on: the original row is held to the source from THAT frame. For a stack of
    the layout of 2026-10-16 (`real` or `compare`) the frames stacked and the clip's first frame are taken from the stack's own
    record (the `.stack.json` its metadata names; `--first` only where there is none), so a build of some frames (`--only`)
    reads back right; the reading holds `source_size` (the rows) and `canvas` (the right column's) apart, and the file's width
    and height to what its metadata says; a file of another size is refused."""
    w, total_h, n = probe(out)
    nj = probe(joined)[2]
    said = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "csv=p=0", out], capture_output=True, text=True).stdout
    import re
    real = re.search(r"stack_delivery layout=(real|compare) source=(\d+)x(\d+) canvas=(\d+)x(\d+) (?:legends=(\d+)\+(\d+)|legend=(\d+)) right=(\d+)", said)
    if real:
        layout_kind = real.group(1)
        sw, sh, cw, ch = (int(real.group(i)) for i in (2, 3, 4, 5))
        l1, l2 = (int(real.group(6)), int(real.group(7))) if real.group(6) else (int(real.group(8)),) * 2
        rw = int(real.group(9))
        if 3 * sh + l1 + l2 != total_h:
            raise SystemExit(f"{out} is {w} by {total_h}: not three rows of {sh} and legends of {l1} and {l2} as its own metadata says")
        if w != sw + rw - (sw + rw) % 2:
            raise SystemExit(f"{out} is {w} wide: not the source's {sw} and the right column's {rw} as its own metadata says")
        native = f"scale={sw}:{sh}"
        named = re.search(r"sidecar=(\S+)", said)
        record = Path(out).with_name(named.group(1)) if named else None
        held = json.loads(record.read_text()) if record is not None and record.is_file() else {}
        f0 = int((held.get("frames") or [first])[0])
        base = int(held.get("clip_begins_on_source_frame", first))
        md5 = lambda p: subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", "0:a", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True).stdout.strip()
        sound = (md5(out), md5(joined))
        if layout_kind == "compare":
            cand_path = candidate_joined or held.get("candidate_joined")
            if not cand_path or not Path(cand_path).is_file():
                raise SystemExit(f"{out} is a compare stack and its candidate clip is not found (give --candidate-joined)")
            cand_base = int(candidate_first if candidate_first is not None else held.get("candidate_begins_on_source_frame", base))
            nc = probe(cand_path)[2]
            orig_s = _small(out, f"crop={sw}:{sh}:0:0", n)
            cur_s = _small(out, f"crop={sw}:{sh}:0:{sh + l1}", n)
            cand_s = _small(out, f"crop={sw}:{sh}:0:{2 * sh + l1 + l2}", n)
            src_ref = _small(source, native, n, f0)
            cur_ref = _small(joined, native, n, f0 - base)
            cand_ref = _small(cand_path, native, n, f0 - cand_base)
            orig_fit = fits(orig_s, src_ref)
            cur_fit = fits(cur_s, cur_ref)
            cand_fit = fits(cand_s, cand_ref)
            align = compare_alignment(orig_s, cur_s, cand_s, src_ref, cur_ref, cand_ref, f0)
            same_span = (f0, f0 + n - 1) == (base, base + nj - 1)
            sound_same = (bool(sound[0]) and sound[0] == sound[1]) if (same_span and sound[1]) else bool(sound[0])
            return {"layout": "compare", "frames": n, "clip_frames": nj, "candidate_clip_frames": nc,
                    "size": [w, total_h], "canvas": [cw, ch], "source_size": [sw, sh], "legends": [l1, l2],
                    "sound_packets": {"stacked": sound[0], "clip": sound[1], "the_same": sound_same},
                    "the_clip_begins_on_source_frame": base, "the_candidate_begins_on_source_frame": cand_base,
                    "frames_stacked": held.get("frames"),
                    "frames_from": f"the stack's record {record.name}" if held else "--first (no stack record beside the file)",
                    "the_original_row": orig_fit, "the_current_row": cur_fit, "the_clip_row": cur_fit,
                    "the_candidate_row": cand_fit, "alignment": align}
        return {"frames": n, "clip_frames": nj, "size": [w, total_h], "canvas": [cw, ch], "source_size": [sw, sh], "legends": [l1, l2],
                "sound_packets": {"stacked": sound[0], "clip": sound[1], "the_same": bool(sound[0]) and sound[0] == sound[1]},
                "the_clip_begins_on_source_frame": base, "frames_stacked": held.get("frames"),
                "frames_from": f"the stack's record {record.name}" if held else "--first (no stack record beside the file)",
                "the_original_row": fits(_small(out, f"crop={sw}:{sh}:0:0", n), _small(source, native, n, f0)),
                "the_clip_row": fits(_small(out, f"crop={sw}:{sh}:0:{2 * sh + l1 + l2}", n), _small(joined, native, n, f0 - base))}
    m = re.search(r"stack_delivery canvas=(\d+)x(\d+) block=(\d+)", said)
    h, bar = (int(m.group(2)), int(m.group(3))) if m else (None, None)
    if h is None:
        for cand in range(total_h // 3, 0, -1):                   # the canvas's height: three rows and the block
            if cand * 3 + sizes(cand)[1] == total_h:
                h, bar = cand, sizes(cand)[1]
                break
    if h is None or h * 3 + bar != total_h:
        raise SystemExit(f"{out} is {w} by {total_h}: not three rows and a block of words as this tool lays them")
    # a stack that says its rows are whole frames is read back as such; one from before was cut as the loader cuts
    fit = (WHOLE if m and "rows=whole" in said else FIT).format(w=w, h=h)
    md5 = lambda p: subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", "0:a", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True).stdout.strip()
    sound = (md5(out), md5(joined))
    return {"frames": n, "clip_frames": nj, "size": [w, total_h], "canvas": [w, h], "sound_packets": {"stacked": sound[0], "clip": sound[1], "the_same": bool(sound[0]) and sound[0] == sound[1]},
            "the_clip_begins_on_source_frame": first,
            "the_original_row": fits(_small(out, f"crop={w}:{h}:0:0", n), _small(source, fit, n, first), cut=24),
            "the_clip_row": fits(_small(out, f"crop={w}:{h}:0:{h * 2 + bar}", n), _small(joined, fit, n), cut=24)}


def check_lines(got: dict, first: int) -> list[str]:
    """What `check` prints of a reading (`read_back`), a line each: the frames, the sound, the rows, and (for a compare stack)
    the offset -2..+2 and own-vs-other alignment figures."""
    s = got["sound_packets"]
    rows = f", rows at the source's {got['source_size'][0]} by {got['source_size'][1]}" if got.get("source_size") else ""
    begins = got.get("the_clip_begins_on_source_frame", first)
    stacked = f"; source frames {got['frames_stacked'][0]}-{got['frames_stacked'][1]} stacked" if got.get("frames_stacked") else ""
    out = [f"frames: the stacked file {got['frames']}, the clip {got['clip_frames']}, which begins on source frame {begins}{stacked}"
           + (f" (from {got['frames_from']})" if got.get("frames_from") else "") + f"; picture {got['size'][0]} by {got['size'][1]}"
           f"{rows}, canvas {got['canvas'][0]} by {got['canvas'][1]}",
           f"sound packets: stacked {s['stacked'] or 'none'}, the clip {s['clip'] or 'none'}: {'the same' if s['the_same'] else 'NOT the same'}"]
    row_pairs = (("ORIGINAL row against the source", "the_original_row"),
                 ("CURRENT row against the current clip", "the_current_row"),
                 ("CANDIDATE row against the candidate clip", "the_candidate_row")) if got.get("layout") == "compare" else (
                 ("ORIGINAL row against the source", "the_original_row"), ("CLIP row against the clip", "the_clip_row"))
    for name, key in row_pairs:
        r = got[key]
        worse = r["a_neighbour_fits_better"]
        cut = f", less its top {r['cut']} rows (the label of the layout before 2026-10-16)" if r.get("cut") else ""
        out.append(f"{name}, {r['frames']} frames at 256 by 192, grey{cut}: its own frame fits best on {r['own_frame_fits_best']}, ties a neighbour on "
                   f"{r['ties_a_neighbour']}, a neighbour fits better on {len(worse)}" + (f" (frames {', '.join(map(str, worse[:20]))})" if worse else "")
                   + f"; mean difference to its own frame {r['mean_to_its_own_frame']} of 255, greatest {r['greatest_to_its_own_frame']}; to the nearer neighbour "
                   f"{r['mean_to_the_nearer_neighbour']}")
    if got.get("alignment"):
        al = got["alignment"]
        out.append(al["note"])
        out.append(f"diff pixels (|candidate - current| >= 4 at 256x192): peak source frame {al['peak_source_frame']} "
                   f"(clip frame {al['peak_clip_frame']}, {al['peak_diff_pixels_ge_4']} px), total {al['total_diff_pixels_ge_4']} px")
        for label, key in (("original", "original"), ("current", "current"), ("candidate", "candidate")):
            offs = al[key]["offsets"]
            seq = " / ".join(f"{k}:{offs[k]}" for k in ("-2", "-1", "0", "+1", "+2"))
            own_other = f"; on diff pixels vs own {al[key]['vs_own']}, vs other {al[key]['vs_other']}" if "vs_own" in al[key] else ""
            out.append(f"alignment {label}: offsets {seq} (best {al[key]['best_offset']:+d}){own_other}")
    return out


def check(a: argparse.Namespace) -> None:
    print("\n".join(check_lines(read_back(a.source, a.joined, a.out, a.first, getattr(a, "candidate_joined", None),
                                          getattr(a, "candidate_first", None)), a.first)))


def _add_candidate_args(parser: argparse.ArgumentParser, required_mode: bool = False) -> None:
    parser.add_argument("--candidate-table", default="", help="the delivery table of the candidate cut (mode c)")
    parser.add_argument("--candidate-joined", default="", help="the candidate delivery clip at the source's size (mode c)")
    parser.add_argument("--candidate-piece", default="", help="a single candidate render piece to compare when --candidate-table is not given")
    parser.add_argument("--candidate-piece-first", type=int, default=None, metavar="SOURCE_FRAME",
                        help="the source frame --candidate-piece's frame 0 is (defaults to the span's first frame)")
    parser.add_argument("--subject", default="", help="the subject whose current and candidate renders are compared in the right column")
    parser.add_argument("--current-tag", default="", help="words for the middle row's title in legend 1 in place of CURRENT CUT")
    parser.add_argument("--candidate-first", type=int, default=None, metavar="SOURCE_FRAME",
                        help="the source frame --candidate-joined's first frame is (defaults to --first)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    for mode_name, help_text in (("build", "write the stacked file"),
                                 ("compare", "write a mode (c) comparison stack: original, current cut, candidate cut, and both working columns")):
        b = sub.add_parser(mode_name, help=help_text)
        b.add_argument("--source", required=True, help="the original the clip was built against")
        b.add_argument("--table", required=True, help="the delivery table the current clip was built from")
        b.add_argument("--joined", required=True, help="the current delivery clip")
        b.add_argument("--out", required=True)
        b.add_argument("--work", required=True, help="a scratch folder for the working file; never the share")
        b.add_argument("--tag", default="", help="words for the bottom row's title in legend 2 in place of THE CUT / CANDIDATE CUT")
        b.add_argument("--alias", action="append", default=[], metavar="OLD=NEW", help="a subject's name in the table shown as another on the picture; repeatable")
        b.add_argument("--copy-rate", type=float, default=24.0, help="the rate of the copy the pieces' graphs load: a loader's start is in its seconds")
        b.add_argument("--only", metavar="FIRST-LAST", help="a trial on these source frames alone")
        b.add_argument("--first", type=int, default=0, metavar="SOURCE_FRAME", help="the source frame the clip's first frame is, for a clip of a span that does not begin on 0")
        b.add_argument("--data", default="", help="the folder of the job folders (data/), where each piece's queued graph is found; the repo's own when not given")
        b.add_argument("--coverage", default="", help="the per-frame gap check's JSON (bench/frame_coverage.py): its runs are drawn as chips on their frames")
        b.add_argument("--one-a-second", action="store_true", help="also write the same frames at one a second, no sound, in parts under the outside "
                       "reviewer's file size, <out>.1fps.<first>-<last>.mp4: for an agent's video reader that takes one frame a second")
        _add_candidate_args(b, required_mode=(mode_name == "compare"))
    c = sub.add_parser("check", help="read a stacked file back against the source and the clip(s)")
    c.add_argument("--source", required=True)
    c.add_argument("--joined", required=True)
    c.add_argument("--candidate-joined", default=None, help="the candidate clip, when reading back a compare stack with no record beside it")
    c.add_argument("--out", required=True)
    c.add_argument("--first", type=int, default=0, metavar="SOURCE_FRAME", help="the source frame the clip's first frame is: the original row is held to the source from there")
    c.add_argument("--candidate-first", type=int, default=None, metavar="SOURCE_FRAME", help="the source frame the candidate clip's first frame is")
    a = p.parse_args()
    {"build": build, "compare": compare, "check": check}[a.mode](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

