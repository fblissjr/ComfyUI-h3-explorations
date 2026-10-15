#!/usr/bin/env python3
"""A delivery clip stacked for the owner's eye: the original, what each render on the frame was given, the clip.

    <python> bench/stack_delivery.py build --source SOURCE --table TABLE.txt --joined CLIP.mp4 --out STACKED.mp4 \\
        --work SCRATCH_FOLDER [--tag WORDS] [--alias OLD=NEW]... [--copy-rate 24] [--only FIRST-LAST] [--first SOURCE_FRAME]
    <python> bench/stack_delivery.py check --source SOURCE --joined CLIP.mp4 --out STACKED.mp4

**What it buys.** A render is handed over stacked, never bare: the masked input on top, the render below, frame
numbers burned in, so a thing seen on playback is reported with the file and the frame. `capture_masked_run.py
video` does that for one load. A delivery clip is many loads laid by a table, and twice (2026-10-11,
2026-10-13) the whole clip's stacked form was made by a one-off script beside the table. This is that script
as a tool: it reads the SAME table the assembler built the clip from, so the stack shows the clip's own pieces.

**One picture a source frame, the frame number on it, three rows and a block of words.**

    ORIGINAL          the source, fitted to the canvas as the lane's loader fits it (`_lib.frames.FIT`)
    WHAT WAS GIVEN    one column a subject, in the order each first lays a piece, the same column for the whole clip
                      (C1, 2026-10-15: tiles jumped and swapped rows as pieces came and went): the piece's own mask
                      picture (the lower half of its `_with_mask.mp4`), and under it the reference its graph wires, on
                      the frame the graph gave it for this source frame, whole and as a head crop following the face
                      (the strip's own rule, `capture_masked_run.head_boxes`, from the reference's points). Each piece's
                      inputs are read through `bench/what_h3_got.py` from its job folder, found by its queued record
                      under `--data`. A column with no piece on a frame says the subject is the original there.
    THE CLIP          the delivery clip, fitted the same way

The words between the middle and bottom rows, every word kept (a line too long goes on to the next; the block is as
tall as the frame that needs the most): what the mask pictures' colours are; the pieces on this frame in the
table's order, each with whose it is and what its row gives back to the source (`less`); each reference video's
file and frame; the table's own `source` line wherever a subject is left as the source on purpose; and every flag
of the joined clip's own check (`<clip>.check.json`) on this frame, whole. A piece the check says changes nothing
on a frame (`piece_changes_nothing`) is badged in its column there.

**Its records**: `<out>.json` beside the table (the columns, where each tile lies, what each piece was given, the
check's flags) and `<out>.frames.jsonl`, a line a source frame with what each column showed and every word; the
file names them in its metadata, with the canvas and the block's height that `check` reads back. `--one-a-second`
also writes the same frames at one a second and no sound, in parts no larger than the outside reviewer's reader
takes (`_lib.frames.OneASecond`), each named by its source frames and listed in the records.

**What is read from where.** The table through the assembler's own two readers, so the format has one home. A
piece's reference video and the source frame its graph starts on from the graph ComfyUI saved in the picture
beside the piece: a load that continued a stored window loads from an earlier frame than its file begins on,
and its reference is taken from the graph's frame, not the file's. The canvas from the first piece's mask
picture. The rate and the sound from the clip.

**`check`** reads the stacked file back: its frame count against the clip's, its sound's packets against the
clip's, and each of the top and bottom rows held against the file it came from on its own frame and on the
frames either side (small, grey, below the label). Printed as figures.

**A reference's path comes out of a picture's metadata**, not from whoever runs this, so it is opened only when
it is a plain video file on this machine (`not_a_plain_video`), and every file here is opened as a local file,
by no other protocol, and as the container its suffix names (`CONTAINER`): ffmpeg is told what the file is and
does not decide from its content, so a playlist under a video's name is not followed. A reference that is not opened is said when the table is read and has no panel.

**What it cannot show or see.** The parts in their own colours (not in a render's files); the still a render
took a face from (it is in the records); a zoomed render's mask at its whole-frame place (its mask picture is of
the zoomed box, as the node drew it); the assembly's restored pixels as a picture (they are named in the words and
in the check's flags, not drawn); two pieces of one subject on one frame (the first is shown, the second said). The read-back cannot see the middle row: whether a mask picture or a reference is on the
right frame rests on the table's and the graphs' own figures (printed by `build`) and on a look. A held or
still stretch of picture fits its neighbours as well as itself, so a tie there says nothing either way.

The working file is written under `--work` and only the finished file goes to `--out`: the share is not scratch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib.frames import FIT, probe  # noqa: E402

#: Lines of words in the block under the middle row, and a line's height at a canvas 768 tall. Reasoned: two
#: pieces, their two references and three subjects left as the source is the most one frame of a three-subject
#: clip has had.
LINES, LINE_AT_768 = 7, 22


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
    """Frames of one file from a frame on, in order, as [h, w, 3] (or [h, w] grey). A frame that cannot be read is None."""
    def __init__(self, path: str, skip: int, vf: str, w: int, h: int, pix: str = "rgb24"):
        self.at, c = skip, 3 if pix == "rgb24" else 1
        self.size, self.shape = w * h * c, (h, w, 3) if c == 3 else (h, w)
        told = CONTAINER.get(Path(path).suffix.lower())
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-protocol_whitelist", "file", *(["-f", told] if told else []),
                                   "-i", "file:" + str(Path(path).resolve()),
                                   "-vf", f"select='gte(n\\,{max(skip, 0)})'," + vf,
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


def _rate(path: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                          capture_output=True, text=True, check=True).stdout.strip().strip(",")


def sizes(h: int) -> tuple[int, int]:
    """(a line's height, the block's height, even) for a canvas `h` tall."""
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


def layout(w: int, h: int, n: int) -> list[dict]:
    """Where each column's tiles lie in the middle row (w by h), as [x, y, w, h]: its label, the piece's mask picture
    (4:3, at most a little over half the row's height), and under it the reference's head crop and the reference whole."""
    k = h / 768.0
    cw = w // max(1, n)
    lab = round(26 * k)
    mw = min(cw - 8, int(0.52 * h * 4 / 3))
    mw -= mw % 2
    mh = mw * 3 // 4
    y = lab + mh + round(8 * k)
    rh = h - y - round(40 * k)
    hs = min(rh, (cw - 12) // 2)
    rw = min(cw - hs - 12, rh * 4 // 3)
    out = []
    for i in range(n):
        x0 = i * cw
        out.append({"column": [x0, 0, cw, h], "label": [x0 + 4, 2, cw - 8, lab], "mask": [x0 + (cw - mw) // 2, lab, mw, mh],
                    "head": [x0 + 4, y, hs, hs], "whole": [x0 + hs + 8, y, rw, rw * 3 // 4], "caption_y": y + max(hs, rw * 3 // 4) + 4})
    return out


def job_folders(data: Path) -> dict[str, Path]:
    """Each render's file name to the job folder that queued it, from every `*_job/<load>.render.queued.json` under `data`."""
    out = {}
    for q in sorted(data.glob("*_job/*.render.queued.json")):
        try:
            files = json.loads(q.read_text()).get("files") or []
        except ValueError:
            continue
        if files:
            out[Path(files[0]).name] = q.parent
    return out


def given_of_pieces(rows: list[dict], data: Path) -> dict[str, dict]:
    """What each piece's load was handed, read from its own graph (`bench/what_h3_got.py`), by piece name. A piece whose job
    folder is not found is shown from its picture's graph as before, and said so."""
    import what_h3_got
    folders = job_folders(data)
    out = {}
    for r in rows:
        folder = folders.get(Path(r["piece"]).name)
        if folder is None or r["name"] in out:
            continue
        load = Path(r["piece"]).stem.rsplit("_", 1)[0]
        try:
            out[r["name"]] = what_h3_got.read(folder, load, "render")
        except SystemExit as e:
            print(f"what H3 was given for {r['name']} could not be read from {folder.name}: {e}", flush=True)
    return out


def build(a: argparse.Namespace) -> None:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    aliases = dict(x.split("=", 1) for x in a.alias)
    rows, intents = read_rows(a.table, a.copy_rate, aliases)
    if not rows:
        raise SystemExit(f"{a.table} lays no piece: there is nothing to stack beside the original")
    mw, mh, _ = probe(rows[0]["mask"])
    w, h = mw, mh // 2
    k = h / 768.0

    def font(size: float, bold: bool = True):
        for path in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(path, max(8, round(size * k)))
            except OSError:
                pass
        return ImageFont.load_default()

    big, mid, small, plain = font(40), font(20), font(15), font(15, bold=False)
    line_h = max(10, round(LINE_AT_768 * k))

    def label(img, text, size, xy=(8, 6), colour=(255, 235, 60)):
        d = ImageDraw.Draw(img)
        box = d.textbbox(xy, text, font=size)
        d.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        d.text(xy, text, font=size, fill=colour)

    def wrapped(text: str, size, width: int) -> list[str]:
        """Every word kept: a line too wide goes on to the next, never cut."""
        out, line = [], ""
        for word in text.split():
            if line and size.getlength(line + " " + word) > width:
                out.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        return out + ([line] if line else [])

    for r in rows:
        print(f"{r['first']:5d}-{r['last']:<5d} {r['name']:34s} file starts at {r['piece_first']}, its graph at {r['graph_first']}; reference: "
              + (f"{Path(r['ref'][0]).name} from its frame {r['ref'][1]}" if r["ref"] else "none wired"), flush=True)
    total = probe(a.joined)[2]
    # THE SOURCE FRAME THE CLIP BEGINS ON (`--first`): a clip of a span that does not begin on frame 0. Every frame below is a SOURCE
    # frame: the table's rows, the original read, the number burned in; the clip's own frame is that less `base`. Until 2026-10-14
    # a span clip was stacked as if it began on 0: the original and the masks rows were of other frames and the number was the
    # clip's own count (six per-shot clips, seen on two frames by a reader; the check had passed, reading the source from 0 too).
    base = int(getattr(a, "first", 0) or 0)
    first_f, last_f = (int(v) for v in a.only.split("-")) if a.only else (base, base + total - 1)
    if first_f < base or last_f > base + total - 1:
        raise SystemExit(f"--only {a.only} is not inside the clip's frames, {base} to {base + total - 1} of the source")
    cols = columns_of(rows, first_f, last_f)
    tiles = layout(w, h, len(cols))
    data = Path(getattr(a, "data", "") or HERE.parent / "data")
    given = given_of_pieces(rows, data) if data.is_dir() else {}
    unread = [r["name"] for r in rows if r["name"] not in given]
    if unread:
        print(f"what H3 was given is not read for {', '.join(dict.fromkeys(unread))}: no queued record names their files under {data}"
              + ("" if data.is_dir() else " (the folder is not there: give --data)") + "; their columns show the picture's graph alone", flush=True)
    # the joined clip's own check (`bench/assemble_delivery.py` writes it beside the clip): each flag on its frames, whole
    check_file = Path(a.joined + ".check.json")
    flags = json.loads(check_file.read_text()).get("flags") or [] if check_file.is_file() else []
    capm = None
    if any(g for g in given.values()):
        import capture_masked_run as capm                      # the strip's own head crop, one rule for both clips
    heads, sampled = {}, {}
    for name, g in given.items():
        v = next((x for x in g["inputs"] if x.get("tag") == "<Video 1>"), None)
        row = next(r for r in rows if r["name"] == name)
        rf = ((v or {}).get("record") or {}).get("its_first_frame_is_source_frame")
        if v and capm is not None and rf is not None:
            cw_, ch_ = (v.get("loader") or {}).get("custom") or [w, h]
            # the boxes are the reference's own frames, indexed by the reference frame a panel shows
            got = capm.head_boxes(v, (cw_ or w, ch_ or h), (rf, rf + (v.get("reference_frames") or 1) - 1))
            heads[name] = (got, (cw_ or w, ch_ or h))
        for e in (v or {}).get("encoder_frames") or []:
            for x in e["sampled"]:
                if not x["held"] and x["source_frame"] is not None:
                    sampled.setdefault(name, set()).add(x["source_frame"])
    # the block of words under the middle row: every line of every frame, wrapped and never cut; as tall as the most any frame needs
    # the mask pictures are the node's own review (`video_mask.py::window_layers`), whose legend is too small to read here
    # one line on the marks (the rest is in the records); then every flag of the joined clip's check in a row of its own for the whole
    # clip, bright on its frames and dim off them (C2, 2026-10-15: the live flags alone reflowed on every frame); then this frame's words
    marks_line = "MASK PICTURES: red the mask given, cyan what else regenerates, white what the render kept. GREEN BORDER: the text encoder's frame."
    spans = lambda fl: ", ".join(f"{x}-{y}" if x != y else str(x) for x, y in fl["source_frames"])
    flag_lines = [[one for one in wrapped(f"{fl['id']} {fl['level'].upper()} on {spans(fl)}: {fl['why']}", plain, w - 16)] for fl in flags]

    def notes_of(f: int, panels: list[dict]) -> list[tuple[str, tuple]]:
        out = [(t, (255, 235, 60)) for t in words_of(f, rows, intents, panels)]
        for r in rows:
            if r["first"] <= f <= r["last"] and r["name"] in unread:
                out.append((f"{r['name']}: what H3 was given is not read (no queued record of it found): its column shows its picture's graph",
                            (255, 170, 60)))
        return out
    lines_at = {}
    for f in range(first_f, last_f + 1):
        lines_at[f] = [(one, c) for text, c in notes_of(f, panels_of(f, rows)) for one in wrapped(text, plain, w - 16)]
    marks_lines = wrapped(marks_line, plain, w - 16)
    fixed = len(marks_lines) + sum(len(x) for x in flag_lines)
    bar = 2 * (((fixed + max(len(v) for v in lines_at.values())) * line_h + 12 + 1) // 2)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    silent = str(work / (Path(a.out).stem + ".video.mp4"))
    fit = FIT.format(w=w, h=h)
    sidecar = Path(a.table).with_name(Path(a.out).stem + ".json")
    tag = f"comment=stack_delivery canvas={w}x{h} block={bar} sidecar={sidecar}"
    src, joined = Reader(a.source, first_f, fit, w, h), Reader(a.joined, first_f - base, fit, w, h)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h * 3 + bar}", "-r", _rate(a.joined), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-preset", "medium", silent], stdin=subprocess.PIPE)
    from _lib.frames import OneASecond
    slow = OneASecond(Path(a.out).with_suffix(""), (w, h * 3 + bar), ["-metadata", tag]) if getattr(a, "one_a_second", False) else None
    readers: dict[tuple, Reader] = {}
    missing: dict[str, list[int]] = {}
    records = []
    for f in range(first_f, last_f + 1):
        panels = panels_of(f, rows)
        keep = {(id(p["row"]), p["kind"]) for p in panels}
        for key in [key for key in readers if key not in keep]:
            readers.pop(key).close()
        by_col: dict[str, dict] = {}
        for p in panels:
            key = (id(p["row"]), p["kind"])
            if key not in readers:
                readers[key] = Reader(p["file"], p["frame"], f"crop={w}:{h}:0:{h}" if p["kind"] == "mask" else f"scale={w}:{h}", w, h)
            got = readers[key].frame(p["frame"])
            if got is None:
                missing.setdefault(f"{p['row']['name']} {p['kind']}", []).append(f)
                continue
            who = p["row"]["subject"] or p["row"]["name"]
            slot = by_col.setdefault(who, {"row": p["row"]})
            if slot["row"] is not p["row"]:
                missing.setdefault(f"a second piece of {who} on one frame (the first is shown)", []).append(f)
                continue
            slot[p["kind"]] = (got, p["frame"])
        mid_img = Image.new("RGB", (w, h), (0, 0, 0))
        d = ImageDraw.Draw(mid_img)
        rec = {"source_frame": f, "clip_frame": f - base, "columns": {}, "check_flags": [fl["id"] for fl in flags if any(x <= f <= y for x, y in fl["source_frames"])]}
        for who, t in zip(cols, tiles):
            x0, y0, cw, ch = t["column"]
            d.rectangle((x0, y0, x0 + cw - 1, y0 + ch - 1), fill=(22, 24, 30))
            slot = by_col.get(who)
            if slot is None:
                label(mid_img, f"{who}: the original on this frame (no piece)", small, (t["label"][0], t["label"][1]), (200, 200, 200))
                rec["columns"][who] = None
                continue
            row = slot["row"]
            name = row["name"]
            live = [fl for fl in flags if any(x <= f <= y for x, y in fl["source_frames"]) and (fl.get("piece") or "").startswith(name)]
            label(mid_img, f"{who}: {name}", small, (t["label"][0], t["label"][1]))
            col = {"piece": name, "mask_frame": None, "reference_frame": None, "encoder_sample": f in sampled.get(name, set()),
                   "check_flags": [fl["id"] for fl in live]}
            if "mask" in slot:
                x, y, tw, th = t["mask"]
                mid_img.paste(Image.fromarray(cv2.resize(slot["mask"][0], (tw, th), interpolation=cv2.INTER_AREA)), (x, y))
                col["mask_frame"] = slot["mask"][1]
            if "reference" in slot:
                ref, at = slot["reference"]
                col["reference_frame"] = at
                hb = heads.get(name)
                boxes = hb[0] if hb else None
                x, y, tw, th = t["whole"]
                mid_img.paste(Image.fromarray(cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)), (x, y))
                if boxes:
                    i = min(max(at, 0), len(boxes) - 1)
                    sx, sy = w / hb[1][0], h / hb[1][1]
                    bx0, by0, bx1, by1 = (round(boxes[i][0] * sx), round(boxes[i][1] * sy), round(boxes[i][2] * sx), round(boxes[i][3] * sy))
                    x, y, hs, _ = t["head"]
                    mid_img.paste(Image.fromarray(cv2.resize(ref[by0:by1, bx0:bx1], (hs, hs), interpolation=cv2.INTER_CUBIC)), (x, y))
                    if col["encoder_sample"]:
                        d.rectangle((x - 2, y - 2, x + hs + 1, y + hs + 1), outline=(90, 230, 110), width=3)
                    cap = f"its reference, frame {at}: head, and whole"
                else:
                    cap = f"its reference, frame {at}, whole (no points beside it: no head crop)"
                d.text((t["head"][0], t["caption_y"]), cap, font=small, fill=(230, 230, 230))
            for fl in live:
                if fl["rule"] == "piece_changes_nothing":
                    # the original shows on this frame: the badge says what the piece's own tiles show here
                    x, y, tw, th = t["mask"]
                    said_ = wrapped(f"{fl['id']}: {name} CHANGES 0 PX HERE: THE ORIGINAL SHOWS", small, tw - 12)
                    tall = len(said_) * round(20 * k) + round(8 * k)
                    d.rectangle((x, y + th - tall, x + tw, y + th), fill=(120, 20, 20))
                    for i_, one in enumerate(said_):
                        d.text((x + 6, y + th - tall + round(4 * k) + i_ * round(20 * k)), one, font=small, fill=(255, 255, 255))
            rec["columns"][who] = col
        s, j = src.frame(f), joined.frame(f - base)
        if s is None or j is None:
            raise SystemExit(f"frame {f}: the {'source' if s is None else 'clip'} could not be read; nothing written")
        top, bot = Image.fromarray(s), Image.fromarray(j)
        label(top, f"ORIGINAL   frame {f:05d}", big)
        label(bot, f"{a.tag or 'THE CLIP'}   frame {f:05d}", big)
        block = Image.new("RGB", (w, bar), (18, 18, 18))
        bd = ImageDraw.Draw(block)
        y_ = 4
        for one in marks_lines:
            bd.text((8, y_), one, font=plain, fill=(170, 170, 170))
            y_ += line_h
        for fl, lines_ in zip(flags, flag_lines):
            on = fl["id"] in rec["check_flags"]
            for one in lines_:
                bd.text((8, y_), one, font=plain, fill=(255, 150, 60) if on else (95, 95, 95))
                y_ += line_h
        y_ += 4
        for text, colour in lines_at[f]:
            bd.text((8, y_), text, font=plain, fill=colour)
            y_ += line_h
        rec["words"] = [t for t, _ in lines_at[f]]
        records.append(rec)
        frame = Image.new("RGB", (w, h * 3 + bar))
        frame.paste(top, (0, 0))
        frame.paste(mid_img, (0, h))
        frame.paste(block, (0, h * 2))
        frame.paste(bot, (0, h * 2 + bar))
        data_ = frame.tobytes()
        enc.stdin.write(data_)
        if slow is not None:
            slow.write(data_, f)
    enc.stdin.close()
    slow_code, slow_parts = slow.close() if slow is not None else (0, [])
    if enc.wait() or slow_code:
        raise SystemExit("the encoder failed; nothing written")
    for r in readers.values():
        r.close()
    src.close()
    joined.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", a.joined, "-map", "0:v", "-map", "1:a?", "-c", "copy", "-metadata", tag, a.out], check=True)
    Path(silent).unlink()
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
                                   "frames": [first_f, last_f], "canvas": [w, h], "block": bar, "columns": cols, "layout": tiles,
                                   "source_is": ("the original the stack's top row shows and the assembler laid the pieces on (--source); each piece's own "
                                                 "graph names the file it was rendered from (given[piece].inputs, role 'the source'), which may be another "
                                                 "copy of the same clip"),
                                   "rendered_from": sorted({x["file"] for g in given.values() for x in g["inputs"] if x["role"] == "the source"}),
                                   "given": given, "given_not_read": sorted(set(unread)), "check_flags": flags,
                                   "flag_lines": dict(zip((fl["id"] for fl in flags), flag_lines)), "records": Path(sidecar).with_suffix(".frames.jsonl").name}, indent=1, default=str) + "\n")
    with open(sidecar.with_suffix(".frames.jsonl"), "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"written {a.out} ({last_f - first_f + 1} frames, {w} by {h * 3 + bar})")
    for part in slow_parts:
        print(f"written {part['file']} (source frames {part['frames'][0]}-{part['frames'][1]} at one a second, no sound; {part['bytes']} bytes)")
    print(f"its records: {sidecar} and {sidecar.with_suffix('.frames.jsonl').name}")
    print("columns, one a subject for the whole clip:", ", ".join(cols))
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


def fits(row: np.ndarray, ref: np.ndarray, cut: int = 24) -> dict:
    """A stacked row against the file it came from, both small and grey, below the label (`cut` rows): on how many
    frames its own frame fits best, ties a neighbour, or a neighbour fits better, and the mean differences."""
    n = min(len(row), len(ref))
    d = lambda step: np.array([np.abs(row[f, cut:] - ref[min(max(f + step, 0), n - 1), cut:]).mean() for f in range(n)])
    here, before, after = d(0), d(-1), d(1)
    if n > 1:
        before[0], after[-1] = after[0], before[-1]              # the ends have one neighbour
    other = np.minimum(before, after)
    return {"frames": n, "own_frame_fits_best": int((here < other).sum()), "ties_a_neighbour": int((here == other).sum()),
            "a_neighbour_fits_better": [int(f) for f in np.flatnonzero(here > other)], "mean_to_its_own_frame": round(float(here.mean()), 2),
            "greatest_to_its_own_frame": round(float(here.max()), 2), "mean_to_the_nearer_neighbour": round(float(other.mean()), 2)}


def read_back(source: str, joined: str, out: str, first: int = 0) -> dict:
    """`first` is the source frame the clip begins on: the original row is held to the source from THAT frame."""
    w, total_h, n = probe(out)
    nj = probe(joined)[2]
    # the canvas and the block as the build wrote them into the file (since 2026-10-15 the block is as tall as the clip's words
    # need); a file from before says nothing, and its canvas is found as that build laid it
    said = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "csv=p=0", out], capture_output=True, text=True).stdout
    import re
    m = re.search(r"stack_delivery canvas=(\d+)x(\d+) block=(\d+)", said)
    h, bar = (int(m.group(2)), int(m.group(3))) if m else (None, None)
    if h is None:
        for cand in range(total_h // 3, 0, -1):                   # the canvas's height: three rows and the block
            if cand * 3 + sizes(cand)[1] == total_h:
                h, bar = cand, sizes(cand)[1]
                break
    if h is None or h * 3 + bar != total_h:
        raise SystemExit(f"{out} is {w} by {total_h}: not three rows and a block of words as this tool lays them")
    fit = FIT.format(w=w, h=h)
    md5 = lambda p: subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", "0:a", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True).stdout.strip()
    sound = (md5(out), md5(joined))
    return {"frames": n, "clip_frames": nj, "size": [w, total_h], "canvas": [w, h], "sound_packets": {"stacked": sound[0], "clip": sound[1], "the_same": bool(sound[0]) and sound[0] == sound[1]},
            "the_clip_begins_on_source_frame": first,
            "the_original_row": fits(_small(out, f"crop={w}:{h}:0:0", n), _small(source, fit, n, first)),
            "the_clip_row": fits(_small(out, f"crop={w}:{h}:0:{h * 2 + bar}", n), _small(joined, fit, n))}


def check_lines(got: dict, first: int) -> list[str]:
    """What `check` prints of a reading (`read_back`), a line each: the frames, the sound, the two rows."""
    s = got["sound_packets"]
    out = [f"frames: the stacked file {got['frames']}, the clip {got['clip_frames']}, which begins on source frame {first}; picture {got['size'][0]} by {got['size'][1]}, canvas {got['canvas'][0]} by {got['canvas'][1]}",
           f"sound packets: stacked {s['stacked'] or 'none'}, the clip {s['clip'] or 'none'}: {'the same' if s['the_same'] else 'NOT the same'}"]
    for name, key in (("ORIGINAL row against the source", "the_original_row"), ("CLIP row against the clip", "the_clip_row")):
        r = got[key]
        worse = r["a_neighbour_fits_better"]
        out.append(f"{name}, {r['frames']} frames at 256 by 192, grey, below the label: its own frame fits best on {r['own_frame_fits_best']}, ties a neighbour on "
                   f"{r['ties_a_neighbour']}, a neighbour fits better on {len(worse)}" + (f" (frames {', '.join(map(str, worse[:20]))})" if worse else "")
                   + f"; mean difference to its own frame {r['mean_to_its_own_frame']} of 255, greatest {r['greatest_to_its_own_frame']}; to the nearer neighbour "
                   f"{r['mean_to_the_nearer_neighbour']}")
    return out


def check(a: argparse.Namespace) -> None:
    print("\n".join(check_lines(read_back(a.source, a.joined, a.out, a.first), a.first)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="write the stacked file")
    b.add_argument("--source", required=True, help="the original the clip was built against")
    b.add_argument("--table", required=True, help="the delivery table the clip was built from")
    b.add_argument("--joined", required=True, help="the delivery clip")
    b.add_argument("--out", required=True)
    b.add_argument("--work", required=True, help="a scratch folder for the working file; never the share")
    b.add_argument("--tag", default="", help="words for the bottom row's label in place of THE CLIP")
    b.add_argument("--alias", action="append", default=[], metavar="OLD=NEW", help="a subject's name in the table shown as another on the picture; repeatable")
    b.add_argument("--copy-rate", type=float, default=24.0, help="the rate of the copy the pieces' graphs load: a loader's start is in its seconds")
    b.add_argument("--only", metavar="FIRST-LAST", help="a trial on these source frames alone (the sound is then the whole clip's)")
    b.add_argument("--first", type=int, default=0, metavar="SOURCE_FRAME", help="the source frame the clip's first frame is, for a clip of a span that does not begin on 0")
    b.add_argument("--data", default="", help="the folder of the job folders (data/), where each piece's queued graph is found; the repo's own when not given")
    b.add_argument("--one-a-second", action="store_true", help="also write the same frames at one a second, no sound, in parts under the outside "
                   "reviewer's file size, <out>.1fps.<first>-<last>.mp4: for an agent's video reader that takes one frame a second")
    c = sub.add_parser("check", help="read a stacked file back against the source and the clip")
    c.add_argument("--source", required=True)
    c.add_argument("--joined", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--first", type=int, default=0, metavar="SOURCE_FRAME", help="the source frame the clip's first frame is: the original row is held to the source from there")
    a = p.parse_args()
    {"build": build, "check": check}[a.mode](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
