#!/usr/bin/env python3
"""`bench/stack_rows.py` on made videos: every row on the source frame it says, laid where its picture came from.

1. **Each row shows the source frame of its stacked frame.** A made source whose frames are flat greys, each its own
   level, a `:whole` clip of it that begins on another frame, a `:canvas` clip cut as the lane's loader cuts it and
   beginning on a third, and a `:top` clip twice as tall: on every stacked frame each row reads the level of that
   source frame, and the strip says that frame. The file holds the span's frames and the sound's length.
2. **A canvas row lies where the loader's cut came from.** On a made picture with a ramp across and down, the canvas cut
   (`_lib.frames.FIT`) laid back by `geometry` matches the whole frame inside the box, and the band the cut dropped
   is black.
3. **The input row**: a subject's mask tints that subject's pixels in its colour, and no pixel outside it.
4. **The sound at the source's own rate**: a source at 24000/1001 (as `fun.mp4` is) whose sound is silent until its frame
   1100 and a tone after: the stack of 1100-1110 starts on the tone, where a cut at 24 would start 46 ms early, in the
   silence.
5. **Refused and red**: a row whose clip does not hold every frame of the span is refused; `check` on the good file
   exits 0, and on a file cut short exits 1.

No card, no server. ffmpeg.

    <python> bench/check_stack_rows.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import case, finish  # noqa: E402
from _lib.frames import FIT, stream  # noqa: E402

import stack_rows as sr  # noqa: E402

TOOL = HERE / "stack_rows.py"
SRC = (712, 540)          # the shape of the clip the tool was made for (1424x1080), at half size
CANVAS = (512, 384)
N = 30


def level(n: int) -> int:
    return 30 + (n * 7) % 190


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def raw_video(path: Path, frames, size, pix="rgb24", codec=("-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv444p"), audio=False):
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", "24", "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={N / 24}", "-c:a", "aac", "-shortest"]
    p = subprocess.Popen(cmd + [*codec, str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, path


def world(t: Path) -> dict:
    w, h = SRC
    src = t / "source.mov"
    raw_video(src, (np.full((h, w, 3), level(n), np.uint8) for n in range(N)), SRC, audio=True)
    whole = t / "whole.mp4"                                                    # source frames 3..
    ff("-i", str(src), "-vf", "select=gte(n\\,3)", "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "1", str(whole))
    canvas = t / "canvas.mp4"                                                  # source frames 5.., the loader's cut
    ff("-i", str(src), "-vf", "select=gte(n\\,5)," + FIT.format(w=CANVAS[0], h=CANVAS[1]), "-fps_mode", "passthrough",
       "-c:v", "libx264", "-crf", "1", str(canvas))
    wide = t / "wide.mp4"                                                      # source frames 6.., a 1344x768-shaped canvas
    ff("-i", str(src), "-vf", "select=gte(n\\,6)," + FIT.format(w=448, h=256), "-fps_mode", "passthrough",
       "-c:v", "libx264", "-crf", "1", str(wide))
    tall = t / "with_mask.mp4"                                                 # source frames 4.., the picture on top
    ff("-i", str(src), "-vf", "select=gte(n\\,4)," + FIT.format(w=CANVAS[0], h=CANVAS[1]) + ",pad=iw:ih*2:0:0:white",
       "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "1", str(tall))
    mask = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
    mask[100:260, 150:300] = 255
    mk = t / "mask.mkv"                                                        # source frames 2..
    raw_video(mk, (mask for _ in range(N - 2)), CANVAS, "gray", ("-c:v", "ffv1"))
    return {"src": src, "whole": whole, "canvas": canvas, "wide": wide, "tall": tall, "mask": mk}


def build(t: Path, w: dict, first: int, last: int, rows: list[str], out: str = "o.mp4") -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "build", "--name", "made", "--source", str(w["src"]), "--first", str(first),
           "--last", str(last), "--mask", f"a={w['mask']}@2", "--out", str(t / "out" / out), "--work", str(t / "work")]
    for r in rows:
        cmd += ["--row", r]
    return subprocess.run(cmd, capture_output=True, text=True)


def rows_on_their_frames() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        w = world(t)
        first, last = 10, 20
        got = build(t, w, first, last, [f"whole={w['whole']}@3:whole", f"canvas={w['canvas']}@5:canvas", f"top={w['tall']}@4:top",
                                        f"wide={w['wide']}@6:canvas"])
        assert got.returncode == 0, got.stderr[-2000:]
        out = t / "out" / "o.mp4"
        rec = json.loads(out.with_suffix(".json").read_text())
        rw, rh = rec["row"]
        band = rec["strip"] + rh
        n = last - first + 1
        by, wide_by = rec["rows"][2]["box"][1], rec["rows"][4]["box"][1]
        assert wide_by > by + 10, (wide_by, by)                                # its own canvas: a wider one is cut lower
        frames = list(stream(str(out), (rw, band * 5), 0, None, "", "rgb24"))
        assert len(frames) == n, len(frames)
        for k, f in enumerate(frames):
            want = level(first + k)
            for i in range(1, 5):                                              # the four clip rows
                y = i * band + rec["strip"]
                centre = f[y + rh // 2 - 20:y + rh // 2 + 20, rw // 2 + 120:rw // 2 + 160].astype(float).mean()
                # THE CASE THAT GOES RED when a row shows another source frame than the stacked frame's
                assert abs(centre - want) < 4, (k, i, centre, want)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
                                    "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
        assert abs(dur - n / 24) < 0.03, dur
        assert subprocess.run([sys.executable, str(TOOL), "check", "--out", str(out)], capture_output=True).returncode == 0
        # the band the loader dropped is black on a canvas row, the source on a whole row
        y2, y1 = 2 * band + rec["strip"], 1 * band + rec["strip"]
        if by > 1:
            assert frames[0][y2:y2 + by - 1].mean() < 8 and frames[0][y1:y1 + by - 1].mean() > 20, (by, frames[0][y2:y2 + by].mean())
        y4 = 4 * band + rec["strip"]                                            # the wide canvas: its own band is black
        assert frames[0][y4:y4 + wide_by - 1].mean() < 8 and frames[0][y4 + wide_by + 4:y4 + wide_by + 20].mean() > 20
        # the sound refused where it cannot be cut right: begun after the span's first frame, and a file with no video
        late = subprocess.run([sys.executable, str(TOOL), "build", "--name", "made", "--source", str(w["src"]), "--first", "10",
                               "--last", "20", "--row", f"whole={w['whole']}@3:whole", "--sound", f"{w['src']}@12",
                               "--out", str(t / "out" / "late.mp4"), "--work", str(t / "work")], capture_output=True, text=True)
        assert late.returncode != 0 and "does not hold source frames" in late.stderr, late.stderr[-400:]
        wav = t / "sound.wav"
        ff("-i", str(w["src"]), "-vn", str(wav))
        bare = subprocess.run([sys.executable, str(TOOL), "build", "--name", "made", "--source", str(w["src"]), "--first", "10",
                               "--last", "20", "--row", f"whole={w['whole']}@3:whole", "--sound", f"{wav}@0",
                               "--out", str(t / "out" / "bare.mp4"), "--work", str(t / "work")], capture_output=True, text=True)
        assert bare.returncode != 0 and "has no video" in bare.stderr, bare.stderr[-400:]
        # refused: a row that does not hold the span
        short = build(t, w, 3, first, [f"canvas={w['canvas']}@5:canvas"], "short.mp4")     # the clip begins on 5
        assert short.returncode != 0 and "not every frame" in short.stderr, short.stderr[-500:]
        # red: a file cut short
        cut = t / "out" / "cut.mp4"
        ff("-i", str(out), "-frames:v", str(n - 3), "-c:v", "libx264", "-crf", "18", "-c:a", "copy", str(cut))
        cut.with_suffix(".json").write_text(out.with_suffix(".json").read_text())
        assert subprocess.run([sys.executable, str(TOOL), "check", "--out", str(cut)], capture_output=True).returncode == 1
    return (f"a whole, a canvas, a top-half and a wider canvas row, each beginning on its own source frame, read each "
            f"stacked frame's source frame on all {n}, the wider canvas laid in its own box; the sound is the span's; a "
            f"short row, a sound begun late and a sound with no video refused; a cut file red")


def canvas_where_its_cut_came_from() -> str:
    w, h = SRC
    x = np.linspace(0, 255, w)[None, :, None]
    y = np.linspace(0, 255, h)[:, None, None]
    ramp = np.concatenate([np.broadcast_to(x, (h, w, 1)), np.broadcast_to(y, (h, w, 1)), np.full((h, w, 1), 128.0)], 2)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src = t / "ramp.png"
        raw = t / "ramp.rgb"
        raw.write_bytes(ramp.astype(np.uint8).tobytes())
        ff("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", str(raw), str(src))
        cut = t / "cut.png"
        ff("-i", str(src), "-vf", FIT.format(w=CANVAS[0], h=CANVAS[1]), str(cut))
        g = sr.geometry(SRC, CANVAS)
        rw, rh = g["row"]
        whole = next(stream(str(src), (rw, rh), 0, 1, g["whole"], "rgb24")).astype(float)
        back = next(stream(str(cut), (rw, rh), 0, 1, g["canvas"], "rgb24")).astype(float)
        bx, by, bw, bh = g["box"]
        inside = np.abs(whole[by + 2:by + bh - 2, bx + 2:bx + bw - 2] - back[by + 2:by + bh - 2, bx + 2:bx + bw - 2]).mean()
        # THE CASE THAT GOES RED when the canvas is laid back off the place the loader cut it from
        assert inside < 3, inside
        # the control: ten pixels off across, read on the channel that ramps across, reads off
        shifted = np.abs(whole[by + 2:by + bh - 2, bx + 2:bx + bw - 12, 0] - back[by + 2:by + bh - 2, bx + 12:bx + bw - 2, 0]).mean()
        assert shifted > 2 and shifted > 5 * inside, (shifted, inside)
    return f"the canvas laid back matches the whole frame inside its box (mean {inside:.2f}); ten pixels off it reads {shifted:.2f}"


def the_input_row() -> str:
    pic = np.full((200, 300, 3), 100, np.uint8)
    m = np.zeros((200, 300), np.uint8)
    m[50:150, 100:200] = 255
    got = sr.draw_masks(pic, [("a", m)]).astype(int)
    c = np.array(sr.COLOURS[0])
    inside = got[80:120, 130:170].reshape(-1, 3).mean(0)
    want = 100 * (1 - sr.TINT) + c * sr.TINT
    assert np.abs(inside - want).max() < 2, (inside, want)
    outside = np.concatenate([got[160:, :].reshape(-1, 3), got[:, 220:].reshape(-1, 3)])
    assert (outside == 100).all(), "a pixel outside the mask was drawn on"
    return "the mask's pixels take the subject's colour, none outside it"


def the_sound_at_its_rate() -> str:
    at, rate = 1100, 24000 / 1001
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src = t / "ntsc.mp4"
        ff("-f", "lavfi", "-i", f"color=gray:s=64x48:r=24000/1001:d={1200 / rate}", "-f", "lavfi", "-i",
           f"aevalsrc='if(gte(t\\,{at / rate})\\,0.5*sin(2*PI*1000*t)\\,0)':s=48000:d={1200 / rate}",
           "-c:v", "libx264", "-crf", "1", "-c:a", "pcm_s16le", "-shortest", str(t / "ntsc.mov"))
        ff("-i", str(t / "ntsc.mov"), "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", str(src))
        got = subprocess.run([sys.executable, str(TOOL), "build", "--name", "made", "--source", str(src), "--first", str(at),
                              "--last", str(at + 10), "--row", f"it={src}@0:whole", "--out", str(t / "o.mp4"), "--work",
                              str(t / "w")], capture_output=True, text=True)
        assert got.returncode == 0, got.stderr[-1500:]
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(t / "o.mp4"), "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
                             capture_output=True).stdout
        x = np.frombuffer(pcm, np.int16).astype(float)
        head = float(np.sqrt((x[480:1440] ** 2).mean()))                     # 10-30 ms in: past the encoder's ramp
        # THE CASE THAT GOES RED when the sound is cut at 24 a second: its first 46 ms would be silence
        assert head > 3000, head
    return f"a source at 24000/1001 cut from its frame {at}: the sound starts on the tone (rms {head:.0f} 10-30 ms in)"


case("each row on the source frame it says", rows_on_their_frames)
case("the sound at the source's own rate", the_sound_at_its_rate)
case("a canvas row where the loader's cut came from", canvas_where_its_cut_came_from)
case("the input row", the_input_row)

if __name__ == "__main__":
    sys.exit(finish())
