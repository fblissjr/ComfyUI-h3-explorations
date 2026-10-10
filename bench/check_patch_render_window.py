#!/usr/bin/env python3
"""`bench/patch_render_window.py join` against a render and a patch whose answers are known by construction.

The join is the step that puts a redone stretch back into a finished render. If it is a frame off, or takes
the patch on frames outside the hole, or changes the render's colour or its audio, the file that is watched is
wrong in a way the redraw is then blamed for. This makes a small "finished render" with the lane's own writer
(a soft picture that drifts, a tone under it) and a patch window over part of it in which every frame is marked:
the patch's frames carry a painted rectangle, the hole's frames a second one. Then it joins and reads the file.

1. **Every frame is in its place.** The joined file has the render's frame count and rate; outside the hole each
   frame is the render's own (one more encode from it and nearer it than its neighbours); inside the hole each
   frame is the patch's frame for that place, and not the patch's frame one earlier or later.
2. **Only the hole's frames come from the patch.** The patch's frames just outside the hole, which carry the first
   rectangle and not the second, are not in the joined file: there the render's picture is, with no rectangle.
3. **The colour and the audio are the render's.** The four colour fields of the joined file are the render's, its
   planes have no bias against the frames that went in, and its audio is the render's packets, byte for byte.
4. **The control.** The same join with the window given one frame late puts every hole frame one off, and the
   check of case 1 says so.
5. **A render from before the writer tagged its files, patched by a file written after.** The render holds BT.601
   values with no tag and the patch BT.709 with one. Read as a player reads it, the joined file is the render as it
   plays outside the hole and the patch as it plays inside it: no step in colour where the patch begins, and the file says what it
   holds. Joined by a plain concatenation the hole's frames were two to three levels off in every channel, in a
   file with no tag (measured 2026-10-10); that is why the join goes through `bench/assemble_delivery.py`.

## Running it

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_patch_render_window.py

Imports the pack's writer, so ComfyUI core must be importable; no card and no server. The `build` half of the
tool (the one-window graph) is not covered here: it needs a render's own graph and masks.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap, case, finish  # noqa: E402

bootstrap(cpu=True)        # the writer is imported for its ffmpeg call; nothing here uses a device
import torch  # noqa: E402

import assemble_delivery as assembler  # noqa: E402  (its readers; the join under test is patch_render_window's)

W, H = 256, 192
FRAMES, RATE = 60, 24
WINDOW_FIRST, WINDOW_LENGTH = 20, 22          # the patch covers render frames 20-41
HOLE = (26, 35)                               # the frames to take from it, inclusive
MARK_PATCH = (30, 30, 90, 90)                 # on every frame of the patch
MARK_HOLE = (150, 100, 220, 160)              # on the patch's frames inside the hole only
PICTURE = "nb_colors=5:c0=0x905030:c1=0x2a5fa0:c2=0xc8b450:c3=0x3c6432:c4=0xa04682:x0=40:y0=30:x1=360:y1=270:seed=7"
TOOL = Path(__file__).resolve().parent / "patch_render_window.py"


def _song():
    """The pack's song node as a module of a stand-in package; the repo root stays off `sys.path`."""
    pkg = sys.modules.get("_h3pack")
    if pkg is None:
        pkg = types.ModuleType("_h3pack")
        pkg.__path__ = [str(REPO)]
        sys.modules["_h3pack"] = pkg
    return importlib.import_module("_h3pack.audio_freeze_song")


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, **kw)
    assert proc.returncode == 0, f"{' '.join(str(c) for c in cmd[:6])} ...: {proc.stderr.decode(errors='replace')[-400:]}"
    return proc.stdout


def planes(path) -> np.ndarray:
    """Every frame of a file as yuv420p planes in the file's own levels, [n, bytes]."""
    raw = run([assembler.FFMPEG, "-v", "error", "-an", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, W * H * 3 // 2).astype(np.int16)


def luma(frame: np.ndarray) -> np.ndarray:
    return frame[:W * H].reshape(H, W)


def marked(frame: np.ndarray, base: np.ndarray, rect) -> float:
    """How far a frame's luma is from a base frame's inside a rectangle, a few pixels in."""
    x0, y0, x1, y1 = rect
    return float(np.abs(luma(frame) - luma(base))[y0 + 4:y1 - 4, x0 + 4:x1 - 4].mean())


with tempfile.TemporaryDirectory() as _tmp:
    TMP = Path(_tmp)
    song = _song()
    raw = run([assembler.FFMPEG, "-v", "error", "-f", "lavfi", "-i", f"gradients=size={W}x{H}:rate={RATE}:speed=0.03:{PICTURE}",
               "-frames:v", str(FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    picture = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(FRAMES, H, W, 3).copy()).float() / 255.0
    silent = str(TMP / "render_silent.mp4")
    assert song._write_frames_mp4(silent, picture, 12) == FRAMES
    RENDER = str(TMP / "render.mp4")
    run([assembler.FFMPEG, "-y", "-v", "error", "-i", silent, "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-ac", "2", "-b:a", "96k", "-t", f"{FRAMES / RATE:.6f}", RENDER])

    def patch_frames() -> torch.Tensor:
        out = picture[WINDOW_FIRST:WINDOW_FIRST + WINDOW_LENGTH].clone()
        x0, y0, x1, y1 = MARK_PATCH
        out[:, y0:y1, x0:x1] = (out[:, y0:y1, x0:x1] + 0.5) % 1.0
        x0, y0, x1, y1 = MARK_HOLE
        a, b = HOLE[0] - WINDOW_FIRST, HOLE[1] - WINDOW_FIRST + 1
        # brighter by a different amount on every hole frame, so a frame taken one off is told from its neighbour
        for k in range(a, b):
            out[k, y0:y1, x0:x1] = (out[k, y0:y1, x0:x1] * 0.2 + 0.1 + 0.07 * (k - a)).clamp(0, 1)
        return out
    PATCH = str(TMP / "patch.mp4")
    assert song._write_frames_mp4(PATCH, patch_frames(), 12) == WINDOW_LENGTH

    def join(name: str, window_first: int = WINDOW_FIRST) -> str:
        out = str(TMP / f"{name}.mp4")
        proc = subprocess.run([sys.executable, str(TOOL), "join", "--render", RENDER, "--patch", PATCH, "--window", str(window_first),
                               "--hole", str(HOLE[0]), str(HOLE[1]), "--out", out], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-400:] or proc.stdout[-400:]
        return out

    JOINED = join("joined")
    render, patch, joined = planes(RENDER), planes(PATCH), planes(JOINED)

    def misplaced(joined_: np.ndarray) -> list[int]:
        """Hole frames that are nearer the patch's frame one earlier or later than the patch's frame for their place."""
        bad = []
        for f in range(HOLE[0], HOLE[1] + 1):
            k = f - WINDOW_FIRST
            own = np.abs(joined_[f] - patch[k]).mean()
            near = min(np.abs(joined_[f] - patch[j]).mean() for j in (k - 1, k + 1) if 0 <= j < len(patch))
            if not own < near:
                bad.append(f)
        return bad

    def in_place() -> str:
        info, base = assembler.probe(JOINED), assembler.probe(RENDER)
        assert info["frames"] == FRAMES == len(joined) and info["rate"] == base["rate"], (info["frames"], info["rate"], base["rate"])
        outside = [f for f in range(FRAMES) if not HOLE[0] <= f <= HOLE[1]]
        worst = max(float(np.abs(joined[f] - render[f]).mean()) for f in outside)
        assert worst < 1.0, f"a frame outside the hole is {worst:.2f} levels from the render's"
        for f in outside[1:-1]:
            own = np.abs(joined[f] - render[f]).mean()
            assert own < min(np.abs(joined[f] - render[f - 1]).mean(), np.abs(joined[f] - render[f + 1]).mean()), f"frame {f} is nearer a neighbour of the render's"
        bad = misplaced(joined)
        assert not bad, f"hole frames {bad} are not the patch's frames for their place"
        inside = max(float(np.abs(joined[f] - patch[f - WINDOW_FIRST]).mean()) for f in range(HOLE[0], HOLE[1] + 1))
        assert inside < 1.0, f"a hole frame is {inside:.2f} levels from the patch's"
        return f"{FRAMES} frames; outside the hole at most {worst:.2f} levels from the render, inside at most {inside:.2f} from the patch"

    def only_the_hole() -> str:
        for f in (HOLE[0] - 1, HOLE[1] + 1, WINDOW_FIRST, WINDOW_FIRST + WINDOW_LENGTH - 1):
            got = marked(joined[f], render[f], MARK_PATCH)
            assert got < 2.0, f"frame {f} is outside the hole and carries the patch's mark ({got:.1f} levels)"
        for f in (HOLE[0], HOLE[1]):
            assert marked(joined[f], render[f], MARK_PATCH) > 30 and marked(joined[f], render[f], MARK_HOLE) > 8, f"hole frame {f} is not the patch's"
        return "the patch's frames either side of the hole are not in the file"

    def colour_and_audio() -> str:
        says, was = assembler.probe(JOINED)["says"], assembler.probe(RENDER)["says"]
        assert says == was, f"the joined file says {says} about its colour; the render says {was}"
        outside = [f for f in range(FRAMES) if not HOLE[0] <= f <= HOLE[1]]
        n = W * H
        bias = [float(np.mean([(joined[f][a:b] - render[f][a:b]).mean() for f in outside])) for a, b in ((0, n), (n, n * 5 // 4), (n * 5 // 4, n * 3 // 2))]
        assert max(abs(x) for x in bias) < assembler.BIAS, f"the joined file sits {[round(x, 2) for x in bias]} levels (Y, U, V) from the render outside the hole"
        a, b = assembler.audio_packets(RENDER), assembler.audio_packets(JOINED)
        assert [p[1:3] for p in a] == [p[1:3] for p in b], f"the audio is not the render's packets ({len(a)} against {len(b)})"
        return f"colour {says}; bias {[round(x, 2) for x in bias]}; {len(b)} audio packets, the render's"

    def a_frame_late() -> str:
        late = planes(join("joined_late", WINDOW_FIRST + 1))
        bad = misplaced(late)
        assert len(bad) >= HOLE[1] - HOLE[0], f"a window given one frame late left only {bad} out of place"
        return f"{len(bad)} of the hole's {HOLE[1] - HOLE[0] + 1} frames read as out of place"

    def an_older_render() -> str:
        rgb = (picture * 255.0).round().to(torch.uint8).numpy()
        old = str(TMP / "render_older_form.mp4")
        run([assembler.FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(RATE), "-i", "-",
             "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100", "-map", "0:v", "-map", "1:a", "-vf", "scale=out_color_matrix=bt601",
             "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-bsf:v", "h264_metadata=matrix_coefficients=2", "-c:a", "aac", "-ac", "2",
             "-b:a", "96k", "-t", f"{FRAMES / RATE:.6f}", old], input=rgb.tobytes())
        assert assembler.probe(old)["matrix"] == "unknown", "the stand-in for an older render carries a matrix tag"
        out = str(TMP / "joined_older.mp4")
        proc = subprocess.run([sys.executable, str(TOOL), "join", "--render", old, "--patch", PATCH, "--window", str(WINDOW_FIRST),
                               "--hole", str(HOLE[0]), str(HOLE[1]), "--out", out], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-400:] or proc.stdout[-400:]

        def as_seen(path):
            raw_ = run([assembler.FFMPEG, "-v", "error", "-an", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"])
            return np.round(np.frombuffer(raw_, "<u2").reshape(-1, H, W, 4)[..., :3] / 65535 * 255)
        seen, went_in, patched = as_seen(out), as_seen(old), as_seen(PATCH)      # each file as a player shows it
        outside = np.concatenate([(seen[f] - went_in[f]).reshape(-1, 3) for f in (HOLE[0] - 3, HOLE[0] - 1, HOLE[1] + 1, HOLE[1] + 3)]).mean(0)
        inside = np.concatenate([(seen[f] - patched[f - WINDOW_FIRST]).reshape(-1, 3) for f in (HOLE[0], HOLE[0] + 1, HOLE[1])]).mean(0)
        assert np.abs(outside).max() < 1.0, f"outside the hole the joined file reads {np.round(outside, 2)} (R, G, B) from the render as it plays"
        assert np.abs(inside).max() < 1.0, f"inside the hole the joined file reads {np.round(inside, 2)} (R, G, B) from the patch's picture: a step in colour at the hole"
        says = assembler.probe(out)["says"]
        assert says == assembler.SAYS, f"the joined file says {says} about its colour"
        return f"read as a player reads it: {np.round(outside, 2)} outside the hole, {np.round(inside, 2)} inside; the file says {says}"

    case("every frame is in its place", in_place)
    case("only the hole's frames come from the patch", only_the_hole)
    case("the colour and the audio are the render's", colour_and_audio)
    case("control: the window given one frame late", a_frame_late)
    case("an older render patched by a file written today", an_older_render)
sys.exit(finish())
