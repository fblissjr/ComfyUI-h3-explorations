#!/usr/bin/env python3
"""Clips of the same source frames, one row each, stacked for the owner to judge one against another, with sound.

    <python> bench/stack_rows.py build --name LOAD --source SOURCE --first F --last L --out STACKED.mp4 --work FOLDER \\
        --records data/<video>/review/rows \\
        [--mask ID=MASK.mkv@FRAME]... [--input-label WORDS] \\
        --row "WORDS=CLIP.mp4@FRAME[:canvas|:whole|:top|:bottom]"... [--sound FILE@FRAME] [--one-a-second]
    <python> bench/stack_rows.py check --out STACKED.mp4

**What it buys.** A test of a new way of making a piece is judged against what the video has there today, on the same
frames, with the sound (the owner, through the lead, 2026-10-15: the two-person load of 852-958 against the two
pieces it would replace). `stack_delivery.py` stacks a delivery clip by its table and `stack_labeled_clips.py` stacks
clips with no sound and no frame of the source in common; this lines up any number of clips by the source frame each
begins on, so a row can be added (a labelled-reference arm, a chained arm) by one more `--row`.

**One picture a source frame, a row a clip, in the order given, each under a strip of words**: what the row is, the
file, and the source frame. The rows are all in the source's own frame, its whole picture letterboxed into the row:

    the input row     (when `--mask` is given, first) the source with each subject's mask tinted in its own colour,
                      outlined, and its id written at its top; a mask is at the canvas, as the lane's loader cuts the
                      source (`_lib.frames.LOADER_CROP`), and is laid back where that cut came from
    a row             `:whole`, a clip of the source's whole frame (a joined delivery clip); `:canvas`, a render at the
                      canvas, laid back where the loader's cut came from, so the band the cut drops is black and
                      every row's picture is in the same place; `:top`/`:bottom`, the top or bottom half of a clip
                      twice as tall (a render's `_with_mask.mp4`), at the canvas. `FRAME` is the source frame the
                      clip's first frame shows. Each row and mask is laid back from its own canvas: a 1344x768 render
                      beside 1024x768 masks is cut from another part of the source, and is laid back there.

The sound is the source's over the same frames, or `--sound`'s, cut at its own file's frame rate from its first video
frame; the stacked file runs at the source's rate (`fun.mp4` is at 24000/1001, its `_24p_sound_in_step` copy at 24).
A sound file is refused when it has no video (no frames to cut it by), a variable rate, no sound, or does not hold
the span. A row whose clip does not hold every frame from `--first` to `--last` is refused: a row is not padded. Rows
are matched by frame number; a row at another rate than the source is said, not refused (a joined clip of `fun.mp4`
beside the `.mov`'s render is the same frames).

**Its records**: `<stem>.json` under `--records` (the rows, files and the frames each was read from, the filter each was
laid with, the masks, the size of each band), never beside the clip: the clip goes on the share and data does not
(until 2026-10-15 it was written beside the clip). The clip's metadata names the record's whole path, and `check`
reads it from there. `--one-a-second` writes the same frames at one a second and no
sound in parts for the outside reviewer's reader (`_lib.frames.OneASecond`).

**`check`** reads the stacked file back: its frame count against the span (red if not equal), whether it carries
sound and how long, and each clip row held against its own file on the first, middle and last frames and the frames
either side of each, the mean differences printed and the one that fits best named: a row on the wrong frame reads
best a frame off. Figures, not a verdict; a held or still stretch fits its neighbours as well as itself. The input row
is drawn on and is not held.

**What it cannot show.** A zoomed render (its picture is of a box, not the canvas): give a render's `_with_mask.mp4`
top half as a `:top` row only if it was rendered on the whole frame. A file whose packets outnumber its frames (an
edit list) passes the count up front and stops part way with "ran out". Which mask is which subject rests on the file
given with each id. The working file is written under `--work`; only the finished file goes to `--out`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib.frames import FPS, probe, stream  # noqa: E402

#: The width of a row, in pixels. Reasoned: five rows of a 4:3 source at this width (each 720 under a 54 strip) stack
#: under 3,900 pixels, inside the 4,096 a hardware decoder commonly takes; a laptop screen plays it scaled. The
#: one-a-second parts take their size from it.
ROW_WIDTH = 960
#: The words' strip above each row, in pixels (two lines: what the row is, then its file), and its type size. Reasoned,
#: read at the row width above.
STRIP, TYPE = 54, 20
#: A subject's colour on the input row, in the order the masks are given. Reasoned: far apart and away from skin's hues
#: (a pink and an orange were in this list until a reader set them at a flushed face's and a tan's, 2026-10-15).
COLOURS = ((40, 150, 255), (60, 220, 90), (170, 90, 255), (0, 220, 220), (240, 240, 60))
#: How much of a subject's colour a masked pixel takes on the input row.
TINT = 0.45
KINDS = ("canvas", "whole", "top", "bottom")
SPEC = re.compile(r"^(?P<label>.+?)=(?P<path>.+)@(?P<first>\d+)(?::(?P<kind>[a-z]+))?$")


def spec(text: str, kinds: tuple[str, ...] = KINDS, default: str = "whole") -> dict:
    m = SPEC.match(text)
    if not m:
        raise SystemExit(f"{text!r} is not WORDS=FILE@FRAME[:kind]")
    kind = m["kind"] or default
    if kind not in kinds:
        raise SystemExit(f"{text!r}: the kind is one of {', '.join(kinds)}")
    return {"label": m["label"].strip(), "file": m["path"], "first": int(m["first"]), "kind": kind}


def loader_box(source: tuple[int, int], canvas: tuple[int, int]) -> tuple[float, float, float, float]:
    """Where in the source the lane's loader cuts the canvas from: (x, y, w, h) in source pixels, centred, the canvas's
    shape (`_lib.frames.LOADER_CROP`: the whole width when the canvas is the wider, else the whole height)."""
    sw, sh = source
    ar, a = canvas[0] / canvas[1], sw / sh
    w, h = (sw, sw / ar) if ar > a else (sh * ar, sh)
    return (sw - w) / 2, (sh - h) / 2, w, h


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def geometry(source: tuple[int, int], canvas: tuple[int, int] | None) -> dict:
    """The row's size, where in it the source's whole frame lies, and (given a canvas) where that canvas's cut lies."""
    sw, sh = source
    w = ROW_WIDTH
    h = even(w * sh / sw)
    out = {"row": [w, h], "whole": f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"}
    if canvas is not None:
        x, y, cw, ch = loader_box(source, canvas)
        k = w / sw
        bw, bh = even(cw * k), even(ch * k)
        bx, by = int(round(x * k)), int(round(y * k))
        bx, by = min(bx, w - bw), min(by, h - bh)
        out["box"] = [bx, by, bw, bh]
        out["canvas"] = f"scale={bw}:{bh},pad={w}:{h}:{bx}:{by}"
    return out


def laid(row: dict, source: tuple[int, int]) -> dict:
    """A row or mask with its own canvas (its file's size, or half its height for `:top`/`:bottom`), the box that canvas
    lies in, and the filter that lays it there."""
    if row["kind"] == "whole":
        return {**row, "vf": geometry(source, None)["whole"]}
    w, h = size_of(row["file"])
    canvas = (w, h // 2) if row["kind"] in ("top", "bottom") else (w, h)
    g = geometry(source, canvas)
    half = {"top": "crop=iw:ih/2:0:0,", "bottom": "crop=iw:ih/2:0:ih/2,"}.get(row["kind"], "")
    return {**row, "canvas": list(canvas), "box": g["box"], "vf": half + g["canvas"]}


def _font(size: int):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(d, text: str, font, room: float) -> str:
    """The words cut to the room with an ellipsis; the whole words are in the records."""
    if d.textlength(text, font=font) <= room:
        return text
    while d.textlength(text + "…", font=font) > room and len(text) > 4:
        text = text[:-1]
    return text + "…"


def strip(width: int, words: str, file: str, frame: int):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, STRIP), (0, 0, 0))
    d, f, small = ImageDraw.Draw(img), _font(TYPE), _font(TYPE - 5)
    tag = f"frame {frame}"
    tw = d.textlength(tag, font=f)
    d.text((width - tw - 10, 4), tag, fill=(255, 230, 120), font=f)
    d.text((10, 4), _fit(d, words, f, width - tw - 30), fill=(255, 255, 255), font=f)
    d.text((10, TYPE + 12), _fit(d, file, small, width - 20), fill=(170, 170, 170), font=small)
    return np.asarray(img)


def outline(m: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(m)
    edge[1:] |= m[1:] != m[:-1]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    return edge & m


def draw_masks(picture: np.ndarray, masks: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """The input row: each subject's mask tinted in its own colour and outlined, its id at the top of its mask."""
    from PIL import Image, ImageDraw
    out = picture.astype(np.float32)
    for n, (_id, m) in enumerate(masks):
        c = np.array(COLOURS[n % len(COLOURS)], np.float32)
        on = m > 127
        out[on] = out[on] * (1 - TINT) + c * TINT
        out[outline(on)] = c
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    d, f = ImageDraw.Draw(img), _font(TYPE + 2)
    for n, (ident, m) in enumerate(masks):
        ys, xs = np.nonzero(m > 127)
        if not len(ys):
            continue
        x, y = int(np.median(xs[ys <= ys.min() + 8])), int(ys.min())
        w = d.textlength(ident, font=f)
        x0, y0 = int(max(2, min(img.width - w - 8, x - w / 2))), max(2, y - TYPE - 10)
        d.rectangle((x0 - 4, y0 - 2, x0 + w + 4, y0 + TYPE + 6), fill=COLOURS[n % len(COLOURS)])
        d.text((x0, y0), ident, fill=(0, 0, 0), font=f)
    return np.asarray(img)


def _ratio(r: str) -> float:
    num, _, den = r.partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def video_of(path: str) -> dict | None:
    """A file's first video stream: its rate as ffmpeg writes it (`24000/1001`) and as a number, its average rate and
    the time its first frame is at; None for a file with no video. The source of this video is not always at 24:
    `fun.mp4` is at 24000/1001 and its `_24p_sound_in_step` copy at 24, the same frames, so a sound cut by seconds must
    be cut at its own file's rate (found 2026-10-15: at 24 for both, the cut drifts a frame every 1,000)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,avg_frame_rate,start_time", "-of", "json", path], capture_output=True, text=True).stdout
    streams = json.loads(out or "{}").get("streams") or []
    if not streams:
        return None
    v = streams[0]
    return {"rate": v["r_frame_rate"], "per_second": _ratio(v["r_frame_rate"]), "average": _ratio(v.get("avg_frame_rate", "0/0")),
            "start": float(v.get("start_time") or 0.0)}


def rate_of(path: str) -> tuple[str, float]:
    v = video_of(path)
    if v is None:
        raise SystemExit(f"{path} has no video: it has no frames to number")
    return v["rate"], v["per_second"]


def sound_of(path: str) -> dict | None:
    """A file's first sound stream: when it starts and how long it is; None for a file with no sound."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time,duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    streams = json.loads(out or "{}").get("streams") or []
    if not streams:
        return None
    return {"start": float(streams[0].get("start_time") or 0.0), "duration": float(streams[0].get("duration") or 0.0)}


def size_of(path: str) -> tuple[int, int]:
    w, h, _n = probe(path)
    return w, h


def build(a: argparse.Namespace) -> None:
    first, last = a.first, a.last
    n = last - first + 1
    if n < 1:
        raise SystemExit("--last is before --first")
    rows = [spec(r) for r in a.row]
    masks = [spec(m, ("canvas",), "canvas") for m in a.mask]
    if not rows:
        raise SystemExit("give at least one --row")
    for r in rows + masks + [{"file": a.source, "first": 0, "label": "the source"}]:
        if not Path(r["file"]).is_file():
            raise SystemExit(f"{r['label']}: no file {r['file']}")
        held = probe(r["file"])[2]
        if r["first"] > first or r["first"] + held - 1 < last:
            raise SystemExit(f"{r['label']}: {r['file']} holds source frames {r['first']}-{r['first'] + held - 1}, "
                             f"not every frame of {first}-{last}; a row is not padded")
    source = size_of(a.source)
    g = geometry(source, None)
    w, h = g["row"]
    band = STRIP + h
    rows = [laid(r, source) for r in rows]
    masks = [laid(m, source) for m in masks]
    for r in rows + masks:
        r["file"] = str(Path(r["file"]).resolve())
    rows_in = ([{"label": a.input_label, "file": str(Path(a.source).resolve()), "first": 0, "kind": "input", "vf": g["whole"]}]
               if masks else []) + rows
    tall = band * len(rows_in)
    sound_file, sound_first = (a.sound.rsplit("@", 1)[0], int(a.sound.rsplit("@", 1)[1])) if a.sound else (a.source, 0)
    rate, per_second = rate_of(a.source)                   # the stacked file runs at the source's rate
    sv, sa = video_of(sound_file), sound_of(sound_file)    # and its sound is cut at its own file's rate
    if sv is None:
        raise SystemExit(f"--sound {sound_file} has no video: give the video file that carries the sound, whose frames number it")
    if sv["average"] and abs(sv["average"] - sv["per_second"]) > sv["per_second"] * 1e-3:
        raise SystemExit(f"--sound {sound_file} has a variable rate ({sv['rate']}, on average {sv['average']:.4f}): a frame is no fixed time")
    if sa is None:
        raise SystemExit(f"{sound_file} carries no sound")
    start, length = sv["start"] + (first - sound_first) / sv["per_second"], n / per_second
    if first < sound_first or start < sa["start"] - 1e-3 or start + length > sa["start"] + sa["duration"] + 1e-3:
        raise SystemExit(f"the sound of {sound_file} (from source frame {sound_first}, {sa['start']:.3f}-"
                         f"{sa['start'] + sa['duration']:.3f} s) does not hold source frames {first}-{last} ({start:.3f}-{start + length:.3f} s)")
    for r in rows:
        rr = rate_of(r["file"])
        if abs(rr[1] - per_second) > per_second * 1e-3:
            print(f"  {r['label']} runs at {rr[0]} and the source at {rate}: matched by frame number, not by time")

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = Path(a.records).resolve()
    if records == out.parent.resolve():
        raise SystemExit("--records is the clip's own folder: the record is data and goes under data/, not beside a clip")
    records.mkdir(parents=True, exist_ok=True)
    kept = records / (out.stem + ".json")
    silent = work / (out.stem + ".silent.mp4")
    record = {"made": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "name": a.name,
              "source": str(Path(a.source).resolve()), "first": first, "last": last, "row": [w, h], "strip": STRIP,
              "rows": [{**r, "frames_read": [first - r["first"], last - r["first"]] if r["kind"] != "input" else [first, last]}
                       for r in rows_in],
              "masks": [{"id": m["label"], "file": m["file"], "first": m["first"]} for m in masks],
              "rate": rate, "sound": {"file": str(Path(sound_file).resolve()), "first": sound_first, "rate": sv["rate"],
                                      "from_seconds": start, "seconds": length}, "argv": sys.argv[1:]}
    tag = f"comment=stack_rows first={first} last={last} rows={len(rows_in)} row={w}x{h} strip={STRIP} records={kept}"
    print(f"{a.name}: source frames {first}-{last} ({n}), {len(rows_in)} row(s) of {w}x{h} under a {STRIP}-pixel strip")
    for r in rows_in:
        print(f"  {r['label']}: {r['file']} ({r['kind']}, from its frame {first - r['first'] if r['kind'] != 'input' else first})")

    readers = []
    for r in rows_in:
        readers.append(stream(r["file"], (w, h), first - r["first"], n, r["vf"], "rgb24"))
    mask_readers = [(m["label"], stream(m["file"], (w, h), first - m["first"], n, m["vf"], "gray")) for m in masks]
    enc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-threads", "4", "-f", "rawvideo", "-pix_fmt",
                            "rgb24", "-s", f"{w}x{tall}", "-r", rate, "-i", "-", "-c:v", "libx264", "-crf", "18",
                            "-preset", "medium", "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)
    slow = None
    if a.one_a_second:
        from _lib.frames import OneASecond
        slow = OneASecond(out.with_suffix(""), (w, tall), ["-metadata", tag])
    made = 0
    for k in range(n):
        f = first + k
        bands = []
        mk = [(ident, next(r, None)) for ident, r in mask_readers]
        for (ident, m), mask in zip(mk, masks):
            if m is None:
                raise SystemExit(f"the mask of {ident}: {mask['file']} ran out at source frame {f}")
        for r, reader in zip(rows_in, readers):
            pic = next(reader, None)
            if pic is None:
                raise SystemExit(f"{r['label']}: {r['file']} ran out at source frame {f}")
            if r["kind"] == "input":
                pic = draw_masks(pic, mk)
            under = (Path(r["file"]).name if r["kind"] != "input" else
                     "masks: " + ", ".join(f"{m['label']} {Path(m['file']).parent.name}/{Path(m['file']).name}" for m in masks))
            bands += [strip(w, r["label"], under, f), pic]
        frame = np.ascontiguousarray(np.concatenate(bands, axis=0)).tobytes()
        enc.stdin.write(frame)
        if slow is not None:
            slow.write(frame, f)
        made += 1
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg could not write the stacked picture")
    parts = []
    if slow is not None:
        code, parts = slow.close()
        if code:
            raise SystemExit("ffmpeg could not write the one-a-second parts")
    joined = work / (out.stem + ".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", sound_file, "-map", "0:v", "-map", "1:a:0",
                    "-c:v", "copy", "-af", f"atrim=start={start}:duration={length},asetpts=PTS-STARTPTS", "-c:a", "aac",
                    "-b:a", "192k", "-metadata", tag, "-movflags", "+faststart", str(joined)], check=True)
    record["frames_written"], record["one_a_second"] = made, parts
    shutil.move(str(joined), out)
    silent.unlink(missing_ok=True)
    kept.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out} ({made} frames) and its record {kept}" + (f", {len(parts)} one-a-second part(s)" if parts else ""))


def record_of(out: Path) -> Path:
    """The record's path, from the clip's own metadata (`records=` in its comment)."""
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout
    m = re.search(r"records=(\S+\.json)", tags)
    if not m:
        raise SystemExit(f"{out} names no record in its metadata: not a clip this tool made")
    return Path(m.group(1))


def check(a: argparse.Namespace) -> int:
    out = Path(a.out)
    rec = json.loads(record_of(out).read_text())
    w, h = rec["row"]
    n = rec["last"] - rec["first"] + 1
    _w, tall, frames = probe(str(out))
    hard = []
    if frames != n:
        hard.append(f"{frames} frames, not the span's {n}")
    if tall != (rec["strip"] + h) * len(rec["rows"]):
        hard.append(f"{tall} pixels tall, not {len(rec['rows'])} rows of {rec['strip'] + h}")
    dur = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of",
                          "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip()
    per_second = rate_of(str(out))[1]
    print(f"{out.name}: {frames} frames for source frames {rec['first']}-{rec['last']} ({n}) at {rec.get('rate', FPS)}; sound "
          + (f"{float(dur):.3f} s against {n / per_second:.3f} s" if dur else "NONE"))
    if not dur:
        hard.append("no sound")
    if frames != n or tall != (rec["strip"] + h) * len(rec["rows"]):
        for line in hard:
            print(f"FAIL  {line}")
        return 1                                           # the rows are not read from a file not of the span's shape
    for i, r in enumerate(rec["rows"]):
        if r["kind"] == "input":
            print(f"  row {i + 1} {r['label']}: drawn on, not held")
            continue
        said = []
        for k in sorted({0, n // 2, n - 1}):
            got = next(stream(str(out), (w, tall), k, 1, "", "rgb24"))
            y = i * (rec["strip"] + h) + rec["strip"]
            mine = got[y:y + h].astype(np.int16)
            at = rec["first"] - r["first"] + k
            fit = {}
            for d in (-1, 0, 1):
                if at + d < 0:
                    continue
                want = next(stream(r["file"], (w, h), at + d, 1, r["vf"], "rgb24"), None)
                if want is not None:
                    fit[d] = float(np.abs(mine - want.astype(np.int16)).mean())
            best = min(fit, key=fit.get)
            said.append(f"{rec['first'] + k}: " + " ".join(f"{d:+d} {v:.2f}" for d, v in fit.items())
                        + ("" if best == 0 else f" (fits {best:+d} best)"))
        print(f"  row {i + 1} {r['label']}: mean difference to its own file a frame either side ({'; '.join(said)})")
    for line in hard:
        print(f"FAIL  {line}")
    return 1 if hard else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True, help="the load the clip is for: the review item's heading")
    b.add_argument("--source", required=True)
    b.add_argument("--first", type=int, required=True)
    b.add_argument("--last", type=int, required=True)
    b.add_argument("--mask", action="append", default=[], help="ID=MASK@FRAME: a subject's mask at the canvas, from its source frame")
    b.add_argument("--input-label", default="the masked input")
    b.add_argument("--row", action="append", default=[], help="WORDS=CLIP@FRAME[:canvas|:whole|:top|:bottom]; repeatable, in order")
    b.add_argument("--sound", help="FILE@FRAME: the sound's file and the source frame it begins on (default: the source)")
    b.add_argument("--out", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--records", required=True, help="the folder under data/ the record goes in (never the clip's own)")
    b.add_argument("--one-a-second", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "build":
        build(a)
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
