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
    WHAT WAS GIVEN    for each row of the table on this frame: the node's own picture of what it masked (the
                      lower half of the piece's `_with_mask.mp4`) and, where the graph saved in the piece's
                      picture wires one, its reference video on the frame the graph gave it for this source
                      frame. One panel fills the row; more are tiled at half size. No row: it says so.
    THE CLIP          the delivery clip, fitted the same way

The words between the middle and bottom rows: the pieces on this frame in the table's order, each with whose it
is and what its row gives back to the source (`less`); each reference video's file and frame; and the table's
own `source` line wherever a subject is left as the source on purpose.

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
took a face from; a zoomed render's mask at its whole-frame place (its mask picture is of the zoomed box, as
the node drew it). The read-back cannot see the middle row: whether a mask picture or a reference is on the
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


def build(a: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont
    aliases = dict(x.split("=", 1) for x in a.alias)
    rows, intents = read_rows(a.table, a.copy_rate, aliases)
    if not rows:
        raise SystemExit(f"{a.table} lays no piece: there is nothing to stack beside the original")
    mw, mh, _ = probe(rows[0]["mask"])
    w, h = mw, mh // 2
    line_h, bar = sizes(h)
    k = h / 768.0

    def font(size: float, bold: bool = True):
        for path in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(path, max(8, round(size * k)))
            except OSError:
                pass
        return ImageFont.load_default()

    big, mid, small, plain = font(40), font(20), font(15), font(15, bold=False)

    def label(img, text, size, xy=(8, 6)):
        d = ImageDraw.Draw(img)
        box = d.textbbox(xy, text, font=size)
        d.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        d.text(xy, text, font=size, fill=(255, 235, 60))

    def fitted(d, text, size, width):
        if d.textlength(text, font=size) <= width:
            return text
        while text and d.textlength(text + " ...", font=size) > width:
            text = text[:-1]
        return text + " ..."

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
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    silent = str(work / (Path(a.out).stem + ".video.mp4"))
    fit = FIT.format(w=w, h=h)
    src, joined = Reader(a.source, first_f, fit, w, h), Reader(a.joined, first_f - base, fit, w, h)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h * 3 + bar}", "-r", _rate(a.joined), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-preset", "medium", silent], stdin=subprocess.PIPE)
    readers: dict[tuple, Reader] = {}
    missing: dict[str, list[int]] = {}
    most = 0
    for f in range(first_f, last_f + 1):
        panels = panels_of(f, rows)
        keep = {(id(p["row"]), p["kind"]) for p in panels}
        for key in [key for key in readers if key not in keep]:
            readers.pop(key).close()
        shown = []
        for p in panels:
            key = (id(p["row"]), p["kind"])
            if key not in readers:
                readers[key] = Reader(p["file"], p["frame"], f"crop={w}:{h}:0:{h}" if p["kind"] == "mask" else f"scale={w}:{h}", w, h)
            got = readers[key].frame(p["frame"])
            if got is None:
                missing.setdefault(f"{p['row']['name']} {p['kind']}", []).append(f)
            else:
                shown.append((got, p["words"]))
        most = max(most, len(shown))
        mid_img = Image.new("RGB", (w, h), (0, 0, 0))
        if not shown:
            d = ImageDraw.Draw(mid_img)
            d.text((round(40 * k), round(300 * k)), "NO RENDER ON THIS FRAME", font=big, fill=(255, 235, 60))
            d.text((round(40 * k), round(360 * k)), "the row below is the original", font=mid, fill=(230, 230, 230))
        elif len(shown) == 1:
            mid_img = Image.fromarray(shown[0][0])
            label(mid_img, shown[0][1], mid)
        else:
            pw, ph = w // 2, h // 2
            for i, (arr, said) in enumerate(shown[:4]):
                tile = Image.fromarray(arr).resize((pw, ph), Image.BILINEAR)
                label(tile, said, small)
                mid_img.paste(tile, ((i % 2) * pw, (i // 2) * ph if len(shown) > 2 else ph // 2))
        s, j = src.frame(f), joined.frame(f - base)
        if s is None or j is None:
            raise SystemExit(f"frame {f}: the {'source' if s is None else 'clip'} could not be read; nothing written")
        top, bot = Image.fromarray(s), Image.fromarray(j)
        label(top, f"ORIGINAL   frame {f:05d}", big)
        label(bot, f"{a.tag or 'THE CLIP'}   frame {f:05d}", big)
        block = Image.new("RGB", (w, bar), (18, 18, 18))
        d = ImageDraw.Draw(block)
        notes = words_of(f, rows, intents, panels)
        if len(notes) > LINES:
            missing.setdefault("lines of words that did not fit the block", []).append(f)
        pieces = max(1, sum(r["first"] <= f <= r["last"] for r in rows))
        for i, text in enumerate(notes[:LINES]):
            d.text((8, 4 + i * line_h), fitted(d, text, plain, w - 16), font=plain, fill=(255, 235, 60) if i < pieces else (220, 220, 220))
        frame = Image.new("RGB", (w, h * 3 + bar))
        frame.paste(top, (0, 0))
        frame.paste(mid_img, (0, h))
        frame.paste(block, (0, h * 2))
        frame.paste(bot, (0, h * 2 + bar))
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("the encoder failed; nothing written")
    for r in readers.values():
        r.close()
    src.close()
    joined.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", a.joined, "-map", "0:v", "-map", "1:a?", "-c", "copy", a.out], check=True)
    Path(silent).unlink()
    print(f"written {a.out} ({last_f - first_f + 1} frames, {w} by {h * 3 + bar})")
    print("most panels on one frame's middle row:", most, "(four are drawn)")
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
    h = None
    for cand in range(total_h // 3, 0, -1):                       # the canvas's height: three rows and the block
        if cand * 3 + sizes(cand)[1] == total_h:
            h = cand
            break
    if h is None:
        raise SystemExit(f"{out} is {w} by {total_h}: not three rows and a block of words as this tool lays them")
    bar, fit = sizes(h)[1], FIT.format(w=w, h=h)
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
