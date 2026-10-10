#!/usr/bin/env python3
"""The delivery assembler against a clip and pieces whose answers are known by construction.

`bench/assemble_delivery.py` writes the file that gets watched and says it proved the file whole. A proof
that cannot fail proves nothing, and a merge that looks plausible and gives one pass's pixels to another
would be trusted. So this makes a small clip with a picture that changes every frame and a tone under it,
makes pieces from it with the lane's own writer (`audio_freeze_song._write_frames_mp4`), paints rectangles
of known place and size into them, and reads the tool's answers back.

1. **A piece that changed nothing is the original.** Frames taken as the loader takes them and written by
   the writer, untouched: the tool finds no changed pixel in them. This is the case that fails if the
   writer's conversion and the tool's copy of it (`TO_BT709`) drift apart, or the fit does. A piece in the
   writer's older form (no matrix tag, BT.601 values) is read as that and is the original too.
2. **A painted rectangle is found where it is**: the changed region's box within `SLACK` pixels of the
   rectangle and its area between the rectangle's and the rectangle grown by `SLACK`.
3. **A delivery of pieces and original ranges passes its own check**: the frame count, the rate, the four
   colour fields, the source's audio packets from the first one, every frame nearest its own.
4. **The controls: three files that must fail.** A frame dropped and another shown twice with the count and
   the stamps intact; the audio encoded again; and, for a span from the middle, the track cut straight from
   the source, which begins at the picture's last keyframe and not at the span.
5. **A span from the middle** built by the tool begins its track within `HEAD_PACKETS` packets of the span.
6. **Two pieces on the same frames.** Rectangles apart: each shows its own piece's pixels and nothing is
   shared. Rectangles that overlap: the shared pixels are counted, and with no capture the later row shows
   there. With a capture whose masks put the shared pixels in the FIRST row's subject alone, the first row
   shows there: a mask outranks the table's order.
7. **The flags.** A piece that changes pixels on frames where its subject has no mask; one whose rectangle
   is far from its subject's mask; one whose changed area steps up, named at the frame of the step; one that changes
   nothing. And the case that
   must raise none: the rectangle on its subject's mask, the same size throughout. The tables land in the
   capture folder.
8. **At the source's size** (`--size source`). The file is the source's size and passes its check; away from
   the rectangle every pixel is the source's own, the rows the loader's crop dropped included; the rectangle is
   where the crop and the scale put it, within `SLACK`; a piece that kept every pixel, scaled up, is nearer the
   source's picture where it is laid than three pixels to any side; and a piece painted up to the canvas's top edge raises
   the flag that says the source's rows above it were left as they were.
9. **A class given back to the source** (`restore=<subject>.<Class>` on a row). A small box of one class inside
   the painted rectangle: there the file is the original and round it the piece, at both sizes; the pixels given
   back are counted, and the join flag is raised because they sit in the middle of a large change. The same box
   where the piece changed nothing gives nothing back and raises nothing. A class by its index means the same
   as by its name, a class the part model does not have is refused, and a restore whose subject has no class
   map on some frames says which.
10. **Another subject given back whole** (`restore=<subject>` with no class). Two tracked masks over the painted
    rectangle, the piece's own subject and another, overlapping: where only the other's mask is, the file is the
    original; where both are, and where only the piece's own is, it is the piece. Without a run for the piece
    in the capture nothing is taken out of the other's mask, so the overlap goes back too; and with
    `restore=<subject>:whole` it goes back whether or not there is a run.
11. **A piece that spills over a cut.** A second clip with a hard cut in it and no audio. A piece painted up to
    one frame past the cut raises the flag on that frame; painted up to the cut, or on well past it (a subject who
    is in both shots), it raises nothing; and a piece given as several rows raises each of its flags once. The
    flag says whether the cut falls inside a latent step of the piece's load (`loop_plan.step_span`): it does for
    a load starting at frame 0 and does not for the same pictures loaded from frame 2, where the cut is on a step's
    edge. The delivery of a clip with no audio passes and says none was written.
12. **Whose a pixel is, from the capture's owner map.** Two tracks that both claim a strip, and an `owners.npz`
    that gives half the strip to the other subject and marks the rest contested. `restore=<subject>` gives back
    what that subject owns, the half of the strip included, and not the contested half, which the plain tracked
    masks would have left to the piece whole; and two pieces that both changed the strip each show on the half
    their subject owns. The record says an owner map was used.
13. **What the record says the file did to a subject.** A second subject whose tracked mask lies half on the
    painted rectangle, with a class map that puts one class on the painted half and another on the rest: the
    record's `subjects` table reads about half of its tracked pixels off the source, all of the one class and none
    of the other, at both sizes; with `restore=` on the row it reads none; the floor (pixels no track claims and
    no piece changed) reads none either way; the reading itself is tested on a picture running from black to
    white, where a file's luma at the canvas must be its own fitted original at the dark and bright ends too (read
    as `gray`, ffmpeg widened the range and a quarter of untouched footage read as off the source); and `--compare`
    of the two records prints both figures on one line.
14. **A patch as a later row.** A piece, and a patch of it: a second file made FROM THE PIECE (not from the
    original), repainted in part on a few frames. Given as a later row over those frames, the patch shows wherever
    it differs from the original, which is everything the piece changed there and what the patch redrew; on the
    other frames the piece shows; and the frames are flagged as changed by two pieces, settled by order, since
    both are the same subject's. That is a redone stretch laid over the render it was cut from, in one table. The
    control: the same file read against the table without the patch row fails, naming the hole's frames, though
    every frame is in order and no other proof fails.
15. **A patch as a later row, with a capture.** The piece has a run in the capture and so a known subject; its
    patch has none. The patch still shows on its frames, with nothing said on its row and with `subject=` naming
    whose it is; where a third piece of ANOTHER subject changed the same pixels and the first subject holds them,
    the patch shows there and not the piece it replaces nor the other subject's pass; and the record's
    `rows_shown` reads every row as showing where it was meant to. The first delivery built this way gave the
    patched frames back to the piece (the rule asked which changing piece's subject held a pixel, and the patch
    had no subject to answer with) and every proof passed: `rows_shown` is the proof that now fails on it.
16. **A locked file is not built over.** With the delivery's name in its folder's `LOCKED.md`, the same build is
    refused before anything is written, the file's bytes are what they were, and the refusal names a free name
    beside it; `--check-only` still reads it and passes; with the file's bytes changed it fails, saying the file is
    not the one that was locked; and a name the lock file does not list builds as before.

## Running it

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_assemble_delivery.py

Imports the pack's writer, so ComfyUI core must be importable; no card and no server.
"""
from __future__ import annotations

import importlib
import json
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

import assemble_delivery as tool  # noqa: E402

W, H = 256, 192            # the canvas
SW, SH = 400, 304          # the source: a shape the loader has to crop to the canvas's
FRAMES = 72
RATE = "24000/1001"
SLACK = 6                  # pixels a found box may sit from the painted one: the averaging in `changed`, the 4:2:0
                           # chroma and the codec's ringing each move an edge by a pixel or two
# The picture's colours and line, all named: left to the filter they are drawn at random on every run, seed or
# no seed (measured 2026-10-10: three runs, three pictures), and a check whose clip changes cannot hold a margin.
PICTURE = "nb_colors=5:c0=0x905030:c1=0x2a5fa0:c2=0xc8b450:c3=0x3c6432:c4=0xa04682:x0=40:y0=30:x1=360:y1=270:seed=7"
RECT_A = (40, 40, 104, 120)      # x0, y0, x1, y1
RECT_B = (160, 60, 220, 150)     # apart from A
RECT_C = (80, 80, 150, 150)      # overlaps A in x 80-104, y 80-120


def _song():
    """The pack's song node as a module of a stand-in package. The repo root itself stays off `sys.path`: its
    `nodes.py` would shadow core's."""
    pkg = sys.modules.get("_h3pack")
    if pkg is None:
        pkg = types.ModuleType("_h3pack")
        pkg.__path__ = [str(REPO)]
        sys.modules["_h3pack"] = pkg
    return importlib.import_module("_h3pack.audio_freeze_song")


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, **kw)
    assert proc.returncode == 0, f"{' '.join(str(c) for c in cmd[:6])} ...: {proc.stderr.decode(errors='replace')[-300:]}"
    return proc.stdout


def make_source(path: Path) -> None:
    """A soft picture that drifts, different on every frame, tagged BT.709, with a tone under it. One keyframe, so
    a straight cut has a long head. Soft and slow on purpose: on a hard test pattern that moves fast (`testsrc2`)
    the writer's own codec noise reaches twice `CHANGE` and a piece that kept every pixel reads as changed
    (measured 2026-10-10); `CHANGE` was measured on footage, and that is the picture it holds for."""
    run([tool.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:{PICTURE}",
         "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100", "-frames:v", str(FRAMES), "-shortest",
         "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv", "-c:v", "libx264", "-crf", "12",
         "-g", "250", "-pix_fmt", "yuv420p", *tool.BT709_TAGS, "-c:a", "aac", "-ac", "2", "-b:a", "96k", str(path)])


def loaded(source: Path, first: int, count: int) -> torch.Tensor:
    """Source frames as the loader hands them to the lane: its filter, 16 bits, divided by 65535."""
    raw = run([tool.FFMPEG, "-v", "error", "-an", "-i", str(source), "-pix_fmt", "rgba64le", "-vf",
               f"select='between(n\\,{first}\\,{first + count - 1})'," + tool.loader_fit(W, H), "-fps_mode", "passthrough",
               "-f", "rawvideo", "-"])
    a = np.frombuffer(raw, np.dtype(np.uint16).newbyteorder("<")).reshape(count, H, W, 4)[..., :3]
    return torch.from_numpy((a / 65535.0).astype(np.float32))


def painted(frames: torch.Tensor, rect, frames_from: int = 0) -> torch.Tensor:
    """The frames with a rectangle shifted half of full scale in every channel: far over `CHANGE` on any picture."""
    out = frames.clone()
    x0, y0, x1, y1 = rect
    out[frames_from:, y0:y1, x0:x1] = (out[frames_from:, y0:y1, x0:x1] + 0.5) % 1.0
    return out


def table(path: Path, rows) -> Path:
    path.write_text("".join(f"{a}-{b} {piece} {first}\n" for a, b, piece, first in rows))
    return path


def deliver(tmp: Path, name: str, rows, span: str, *more) -> tuple[int, dict, Path]:
    out = tmp / f"{name}.mp4"
    proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(tmp / "source.mp4"),
                           "--table", str(table(tmp / f"{name}.txt", rows)), "--span", span, "--out", str(out), *more],
                          capture_output=True, text=True)
    report = Path(str(out) + ".check.json")
    assert report.is_file(), f"no check json: {proc.stderr[-300:] or proc.stdout[-300:]}"
    return proc.returncode, json.loads(report.read_text()), out


def recheck(tmp: Path, name: str, rows, span: str, out: Path) -> tuple[int, dict]:
    proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(tmp / "source.mp4"),
                           "--table", str(table(tmp / f"{name}.txt", rows)), "--span", span, "--out", str(out), "--check-only"],
                          capture_output=True, text=True)
    return proc.returncode, json.loads(Path(str(out) + ".check.json").read_text())


def capture(folder: Path, first: int, frames: int, subjects: dict, runs: list, classes: dict | None = None) -> Path:
    """A capture folder as `bench/capture_masked_run.py` lays one out, holding only what the assembler reads.
    `classes[label]` is a class map [frames, H, W] of class indices."""
    folder.mkdir(parents=True)
    manifest = {"name": folder.name, "first_frame": first, "frames": frames, "size": [W, H],
                "subjects": [{"label": k, "sightings": [{"by": "made"}]} for k in subjects], "runs": runs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    for label, mask in subjects.items():
        (folder / "subjects" / label).mkdir(parents=True)
        np.savez_compressed(folder / "subjects" / label / "masks__made.npz", track=np.packbits(mask, axis=-1),
                            covered=np.ones(frames, bool))
        if classes and label in classes:
            np.savez_compressed(folder / "subjects" / label / "classes__made.npz", classes=classes[label])
    return folder


def mask_of(rect, frames: int, grow: int = 0) -> np.ndarray:
    m = np.zeros((frames, H, W), bool)
    x0, y0, x1, y1 = rect
    m[:, max(0, y0 - grow):y1 + grow, max(0, x0 - grow):x1 + grow] = True
    return m


def frame_planes(source: Path, rows, frame: int, captures=None):
    """The one frame the tool would feed for `frame`, as planes, and the record of it."""
    record = {"rows": {}, "shared": {}}
    for r in rows:
        r.setdefault("matrix", tool.probe(r["piece"])["matrix"])
    buf = next(tool.fed_frames(source, [(frame, frame, rows)], W, H, record, captures=captures))
    return tool.planes(buf, W, H), record


def inside(planes, other, rect) -> float:
    x0, y0, x1, y1 = rect
    return float(np.abs(planes[0][y0 + SLACK:y1 - SLACK, x0 + SLACK:x1 - SLACK] - other[0][y0 + SLACK:y1 - SLACK, x0 + SLACK:x1 - SLACK]).mean())


with tempfile.TemporaryDirectory() as _tmp:
    TMP = Path(_tmp)
    SOURCE = TMP / "source.mp4"
    make_source(SOURCE)
    song = _song()
    WHOLE = loaded(SOURCE, 0, FRAMES)

    def write(name: str, frames: torch.Tensor) -> str:
        path = str(TMP / f"{name}.mp4")
        assert song._write_frames_mp4(path, frames, 19) == len(frames), "the writer did not report the frames it was given"
        return path

    KEPT = write("kept", WHOLE[10:40])                         # source frames 10-39, nothing changed
    PIECE_A = write("piece_a", painted(WHOLE[10:40], RECT_A))  # source frames 10-39
    PIECE_B = write("piece_b", painted(WHOLE[10:40], RECT_B))
    PIECE_C = write("piece_c", painted(WHOLE[10:40], RECT_C))
    LATE = write("late", painted(WHOLE[40:60], RECT_B))        # source frames 40-59

    def planes_of(piece: str, frame: int, first: int = 10):
        row = {"piece": piece, "piece_first": first, "matrix": tool.probe(piece)["matrix"]}
        return tool.planes(next(tool.piece_frames(row, frame, frame, W, H)), W, H)

    def original(frame: int):
        return tool.planes(next(tool.original_frames(SOURCE, frame, frame, W, H)), W, H)

    def nothing_changed() -> str:
        worst, top = 0, 0.0
        for f in (10, 25, 39):
            diff = tool.difference(planes_of(KEPT, f), original(f))
            worst, top = max(worst, int(tool.changed(planes_of(KEPT, f), original(f), diff).sum())), max(top, float(diff.max()))
        assert worst == 0, f"a piece that kept every pixel reads {worst} changed pixels against the tool's original"
        old = str(TMP / "kept_older_form.mp4")
        frames = (WHOLE[10:40].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy().tobytes()
        run([tool.FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
             "-vf", "scale=out_color_matrix=bt601", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
             "-bsf:v", "h264_metadata=matrix_coefficients=2", old], input=frames)
        assert tool.probe(old)["matrix"] == "unknown", f"the stand-in for an older file is tagged {tool.probe(old)['matrix']}"
        for f in (10, 39):
            px = int(tool.changed(planes_of(old, f), original(f)).sum())
            assert px == 0, f"a piece in the writer's older form reads {px} changed pixels"
        return f"as the writer writes today (largest difference {top:.1f} levels against a line of {tool.CHANGE:g}), and in its older untagged form"

    def rectangle_found() -> str:
        for f in (10, 39):
            m = tool.changed(planes_of(PIECE_A, f), original(f))
            box = tool.box_of(m)
            assert box is not None and all(abs(a - b) <= SLACK for a, b in zip(box, RECT_A)), f"frame {f}: box {box}, painted {RECT_A}"
            least = (RECT_A[2] - RECT_A[0]) * (RECT_A[3] - RECT_A[1])
            most = (RECT_A[2] - RECT_A[0] + 2 * SLACK) * (RECT_A[3] - RECT_A[1] + 2 * SLACK)
            assert least <= m.sum() <= most, f"frame {f}: {int(m.sum())} px, the rectangle is {least}"
        return f"within {SLACK} px"

    ROWS = [(10, 29, PIECE_A, 10), (40, 59, LATE, 40)]
    built: dict = {}

    def delivery_passes() -> str:
        code, r, out = deliver(TMP, "whole", ROWS, f"0-{FRAMES - 1}")
        built["whole"] = out
        assert code == 0 and r["verdict"] == "passes", r["failures"]
        v, o, a = r["video"], r["order"], r["audio"]
        assert v["frames"] == FRAMES and v["rate"] == RATE and v["timestamps_off_their_place"] == 0, v
        assert v["says_range_matrix_transfer_primaries"] == tool.SAYS, v
        assert o["decoded"] == FRAMES and o["nearer_a_neighbour"] == 0 and o["nearest_their_own_fed_frame"] == FRAMES, o
        assert a["first_source_packet"] == 0 and a["mean_difference_in_place"] < a["nearest_other_place"][1], a
        assert max(abs(x) for x in o["bias_y_u_v"]) < tool.BIAS, o
        assert a["head_the_edit_list_skips_s"] <= tool.HEAD_PACKETS * 1024 / 44100 + 1e-4, a    # the encoder's own priming packet
        assert [s[:2] for s in r["segments"]] == [[0, 9], [10, 29], [30, 39], [40, 59], [60, FRAMES - 1]], r["segments"]
        return f"{FRAMES} frames, {a['packets']} of the source's audio packets"

    def controls_fail() -> str:
        whole = built["whole"]
        tags = ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", *tool.BT709_TAGS, "-video_track_timescale", "24000"]
        broken = TMP / "dropped_and_doubled.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(whole), "-vf",
             "select='not(eq(n\\,20))',loop=loop=1:size=1:start=44,setpts=N*1001/24000/TB," + tool.SAY_BT709, "-r", RATE,
             *tags, "-c:a", "copy", str(broken)])
        code, r = recheck(TMP, "whole", ROWS, f"0-{FRAMES - 1}", broken)
        assert r["video"]["frames"] == FRAMES and r["video"]["timestamps_off_their_place"] == 0, "the control changed the count or the stamps"
        assert code == 1 and any("nearer a neighbour" in f for f in r["failures"]), f"a dropped and a doubled frame passed: {r['failures']}"
        assert r["order"]["first_nearer_a_neighbour"][0] == 20, r["order"]
        again = TMP / "audio_encoded_again.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(whole), "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(again)])
        code, r = recheck(TMP, "whole", ROWS, f"0-{FRAMES - 1}", again)
        assert code == 1 and any("not a run of the source's packets" in f for f in r["failures"]), f"re-encoded audio passed: {r['failures']}"
        code, r, mid = deliver(TMP, "middle", ROWS, "30-60")
        assert code == 0, r["failures"]
        built["middle"] = r
        straight = TMP / "cut_straight.mp4"
        start, length = 30 * 1001 / 24000, 31 * 1001 / 24000
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(mid), "-ss", f"{start:.6f}", "-t", f"{length:.6f}", "-i", str(SOURCE),
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(straight)])
        code, r = recheck(TMP, "middle", ROWS, "30-60", straight)
        assert code == 1 and any("before the span" in f for f in r["failures"]), \
            f"a track cut straight from the source passed: {r['failures']} {r['audio']}"
        return f"a frame dropped and one doubled; audio encoded again; a track begun {r['audio']['head_the_edit_list_skips_s']} s before its span"

    def middle_span() -> str:
        r = built["middle"]
        a = r["audio"]
        packet = 1024 / 44100
        assert r["verdict"] == "passes" and 0 < a["head_the_edit_list_skips_s"] <= tool.HEAD_PACKETS * packet + 1e-4, a
        assert a["mean_difference_in_place"] < a["nearest_other_place"][1] and abs(a["sync_error_s"]) < 1.5 / 44100, a
        return f"the track begins {a['head_the_edit_list_skips_s']} s before the span"

    def two_pieces() -> str:
        f = 20
        a, b, c, o = planes_of(PIECE_A, f), planes_of(PIECE_B, f), planes_of(PIECE_C, f), original(f)
        rows = [{"piece": PIECE_A, "piece_first": 10}, {"piece": PIECE_B, "piece_first": 10}]
        got, record = frame_planes(SOURCE, rows, f)
        assert record["shared"][f]["px"] == 0, f"rectangles apart share {record['shared'][f]['px']} px"
        assert inside(got, a, RECT_A) < 1.0 and inside(got, b, RECT_B) < 1.0, "a piece's rectangle is not its own pixels in the merge"
        corner = (4, 150, 60, 188)
        assert inside(got, o, corner) < 1.5, "the merge changed pixels neither piece changed"
        shared_rect = (RECT_C[0], RECT_C[1], RECT_A[2], RECT_A[3])       # x 80-104, y 80-120
        rows = [{"piece": PIECE_A, "piece_first": 10}, {"piece": PIECE_C, "piece_first": 10}]
        got, record = frame_planes(SOURCE, rows, f)
        area = (shared_rect[2] - shared_rect[0]) * (shared_rect[3] - shared_rect[1])
        px = record["shared"][f]["px"]
        assert area <= px <= (shared_rect[2] - shared_rect[0] + 2 * SLACK) * (shared_rect[3] - shared_rect[1] + 2 * SLACK), \
            f"{px} shared px, the rectangles overlap on {area}"
        assert record["shared"][f]["by_mask"] == 0 and inside(got, c, shared_rect) < 1.0, "with no capture the later row does not show in the overlap"
        only_a = RECT_A[2] - 20
        assert inside(got, a, (RECT_A[0], RECT_A[1], RECT_C[0], RECT_A[3])) < 1.0 and only_a, "the first row lost pixels only it changed"
        folder = capture(TMP / "cap_overlap", 10, 30, {"a": mask_of(RECT_A, 30, grow=SLACK), "c": mask_of((RECT_A[2] + SLACK, RECT_C[1], RECT_C[2], RECT_C[3]), 30)},
                         [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16},
                          {"name": "run_c", "render": "piece_c.mp4", "subject": "c", "margin_px": 16}])
        caps = tool.Captures([folder], (W, H))
        got, record = frame_planes(SOURCE, [{"piece": PIECE_A, "piece_first": 10}, {"piece": PIECE_C, "piece_first": 10}], f, caps)
        assert record["shared"][f]["by_mask"] == record["shared"][f]["px"] > 0, record["shared"][f]
        assert inside(got, a, shared_rect) < 1.0, "the shared pixels lie in the first row's subject and the later row still shows there"
        return f"apart: nothing shared; overlapping: {px} px, the later row by order, the first row when its subject's mask holds them"

    def flags() -> str:
        grown = painted(painted(WHOLE[10:40], RECT_A), (120, 20, 250, 180), frames_from=24)       # from source frame 34 a second, larger change
        jumpy = write("jumpy", grown)
        on = mask_of(RECT_A, 30, grow=4)
        gone = on.copy()
        gone[10:20] = False                                                                       # source frames 20-29: no mask
        far = mask_of((200, 150, 250, 190), 30)                                                    # a mask in the other corner
        cases = {
            "good": (PIECE_A, on, []),
            "gone": (PIECE_A, gone, ["piece_changes_where_its_subject_is_not"]),
            "far": (PIECE_A, far, ["piece_changes_away_from_its_subject"]),
            "idle": (KEPT, on, ["piece_changes_nothing"]),
        }
        for name, (piece, mask, want) in cases.items():
            folder = capture(TMP / f"cap_{name}", 10, 30, {"a": mask},
                             [{"name": "run_a", "render": Path(piece).name, "subject": "a", "margin_px": 16}])
            code, r, out = deliver(TMP, f"flag_{name}", [(10, 39, piece, 10)], "10-39", "--capture", str(folder))
            got = [f["rule"] for f in r["flags"]]
            assert code == 0 and got == want, f"{name}: flags {got}, expected {want}"
            if name == "gone":
                assert r["flags"][0]["source_frames"] == [[20, 29]] and r["flags"][0]["run"] == "run_a", r["flags"][0]
                written = folder / "runs" / "run_a" / f"changed__{out.stem}.csv"
                assert written.is_file() and len(written.read_text().splitlines()) == 31, "the table is not in the capture folder"
                assert json.loads((folder / f"delivery__{out.stem}.json").read_text())["flags"] == r["flags"]
        code, r, _ = deliver(TMP, "flag_jump", [(10, 39, jumpy, 10)], "10-39")
        got = {f["rule"]: f for f in r["flags"]}
        assert list(got) == ["changed_area_steps"], list(got)
        assert got["changed_area_steps"]["source_frames"] == [[34, 34]], got["changed_area_steps"]
        assert r["pieces_with_no_capture"] == ["jumpy.mp4"], r["pieces_with_no_capture"]
        return "no mask, a mask far away, a step in area, nothing changed; none on a piece that stays on its subject"

    def at_the_sources_size() -> str:
        code, r, out = deliver(TMP, "full", [(10, 29, PIECE_A, 10)], "5-34", "--size", "source")
        assert code == 0 and r["verdict"] == "passes" and r["size"] == [SW, SH], (r["failures"], r["size"])
        cw, ch, x0, y0 = tool.crop_of(SW, SH, W, H)
        assert (cw, ch, x0, y0) == (400, 300, 0, 2) == tuple(r["the_loader's_crop_of_the_source"][k] for k in ("width", "height", "x", "y"))

        def luma(path, frame):
            raw = run([tool.FFMPEG, "-v", "error", "-i", str(path), "-vf", f"select='eq(n\\,{frame})'", "-fps_mode", "passthrough",
                       "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
            return np.frombuffer(raw, np.uint8).reshape(SH, SW).astype(np.float32)
        ours, theirs = luma(out, 15), luma(SOURCE, 20)             # the file starts at source frame 5
        moved = np.abs(ours - theirs) > 30                           # the paint moves a pixel by half of full scale
        want = [x0 + RECT_A[0] * cw / W, y0 + RECT_A[1] * ch / H, x0 + RECT_A[2] * cw / W, y0 + RECT_A[3] * ch / H]
        box = tool.box_of(moved)
        assert box is not None and all(abs(a - b) <= SLACK for a, b in zip(box, want)), f"the rectangle is at {box}, the crop and scale put it at {want}"
        far = np.ones_like(moved)
        far[int(want[1]) - 4 * SLACK:int(want[3]) + 4 * SLACK, int(want[0]) - 4 * SLACK:int(want[2]) + 4 * SLACK] = False
        away = float(np.abs(ours - theirs)[far].mean())
        assert away < 1.0, f"away from the rectangle the file is {away:.2f} levels from the source's own picture"
        dropped = float(np.abs(ours[:y0] - theirs[:y0]).mean())
        assert dropped < 1.5, f"the rows the loader's crop dropped are {dropped:.2f} levels from the source's"
        before = float(np.abs(luma(out, 2) - luma(SOURCE, 7)).mean())
        assert before < 1.0, f"a frame no piece covers is {before:.2f} levels from the source's"
        # the scale-up lands on the source: a piece that kept every pixel, laid over the whole crop, is nearest the
        # source's own picture where it is put and not a few pixels to either side
        base = tool.planes(next(tool.source_frames(SOURCE, 20, 20, SW, SH)), SW, SH)
        laid = tool.planes(tool.laid_over(base, [planes_of(KEPT, 20)], np.zeros((H, W), np.int8), (cw, ch, x0, y0)), SW, SH)[0]
        inner = (slice(y0 + 8, y0 + ch - 8), slice(x0 + 8, x0 + cw - 8))
        here = float(np.abs(laid[inner] - base[0][inner]).mean())
        beside = min(float(np.abs(np.roll(laid, (dy, dx), (0, 1))[inner] - base[0][inner]).mean()) for dy, dx in ((0, 3), (0, -3), (3, 0), (-3, 0)))
        assert here < beside, f"a kept piece scaled up is {here:.2f} levels from the source in place and {beside:.2f} three pixels aside"
        top = write("to_the_top", painted(WHOLE[10:40], (100, 0, 160, 40)))
        code, r, _ = deliver(TMP, "full_top", [(10, 29, top, 10)], "10-29", "--size", "source")
        assert code == 0 and [f["rule"] for f in r["flags"]] == ["piece_changes_up_to_the_loader's_crop"], [f["rule"] for f in r["flags"]]
        code, r, _ = deliver(TMP, "canvas_top", [(10, 29, top, 10)], "10-29")
        assert r["flags"] == [], "at the canvas's size nothing is beyond the crop, and the flag was raised"
        return (f"the rectangle within {SLACK} px of where the crop and scale put it; {away:.2f} levels from the source elsewhere; "
                f"a kept piece scaled up {here:.2f} from the source in place, {beside:.2f} three pixels aside")

    def a_class_given_back() -> str:
        names = tool.class_names()
        apparel = names.index("Apparel")
        thing = (60, 60, 76, 80)                          # inside RECT_A: a small thing the piece painted over
        aside = (180, 20, 196, 40)                        # where the piece changed nothing
        run_a = [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}]

        def class_map(rect, frames=30):
            c = np.zeros((frames, H, W), np.uint8)
            c[:, rect[1]:rect[3], rect[0]:rect[2]] = apparel
            return c

        on = capture(TMP / "cap_restore_on", 10, 30, {"a": mask_of(RECT_A, 30, grow=4)}, run_a, {"a": class_map(thing)})
        off = capture(TMP / "cap_restore_off", 10, 30, {"a": mask_of(RECT_A, 30, grow=4)}, run_a, {"a": class_map(aside)})
        short = capture(TMP / "cap_restore_short", 10, 15, {"a": mask_of(RECT_A, 15, grow=4)}, run_a, {"a": class_map(thing, 15)})
        def tabled(name, text, *more):
            path = TMP / f"{name}.txt"
            path.write_text(text)
            out = TMP / f"{name}.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(SOURCE), "--table", str(path),
                                   "--span", "10-39", "--out", str(out), *more], capture_output=True, text=True)
            report = Path(str(out) + ".check.json")
            return proc, (json.loads(report.read_text()) if report.is_file() else None), out

        line = f"10-39 {PIECE_A} 10 restore=a.Apparel\n"
        proc, r, out = tabled("restore_on", line, "--capture", str(on))
        assert proc.returncode == 0 and r is not None, proc.stderr[-300:]
        piece = r["regions"]["pieces"][PIECE_A]
        area = (thing[2] - thing[0]) * (thing[3] - thing[1])
        grown = (thing[2] - thing[0] + 2 * tool.RESTORE_GROW + 2) * (thing[3] - thing[1] + 2 * tool.RESTORE_GROW + 2)
        assert all(area <= px <= grown for px in piece["restored_px_per_frame"]), (area, piece["restored_px_per_frame"][:4])
        assert [f["rule"] for f in r["flags"]] == ["restored_pixels_beside_a_large_change"], [f["rule"] for f in r["flags"]]
        assert r["flags"][0]["source_frames"] == [[10, 39]], r["flags"][0]["source_frames"]
        table_file = on / "runs" / "run_a" / f"changed__{out.stem}.csv"
        assert "restored_px" in table_file.read_text().splitlines()[0], "the capture's table has no restored column"
        row = {"piece": PIECE_A, "piece_first": 10, "restore": [("a", [apparel])]}
        got, _ = frame_planes(SOURCE, [row], 20, tool.Captures([on], (W, H)))
        o, a = original(20), planes_of(PIECE_A, 20)
        core = (thing[0] + 2, thing[1] + 2, thing[2] - 2, thing[3] - 2)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1]:rect[3], rect[0]:rect[2]] - y[0][rect[1]:rect[3], rect[0]:rect[2]]).mean())
        assert box_mean(got, o, core) < 1.0, f"inside the restored box the frame is {box_mean(got, o, core):.2f} levels from the original"
        rest = (RECT_A[0] + SLACK, 90, RECT_A[2] - SLACK, RECT_A[3] - SLACK)
        assert box_mean(got, a, rest) < 1.0, "away from the restored box the piece's own pixels are gone"
        proc, r, out = tabled("restore_on_full", line, "--capture", str(on), "--size", "source")
        assert proc.returncode == 0 and r is not None, proc.stderr[-300:]
        cw, ch, x0, y0 = tool.crop_of(SW, SH, W, H)

        def luma(path, frame):
            raw = run([tool.FFMPEG, "-v", "error", "-i", str(path), "-vf", f"select='eq(n\\,{frame})'", "-fps_mode", "passthrough",
                       "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
            return np.frombuffer(raw, np.uint8).reshape(SH, SW).astype(np.float32)
        ours, theirs = luma(out, 10), luma(SOURCE, 20)
        big = [int(x0 + core[0] * cw / W) + 2, int(y0 + core[1] * ch / H) + 2, int(x0 + core[2] * cw / W) - 2, int(y0 + core[3] * ch / H) - 2]
        back = float(np.abs(ours - theirs)[big[1]:big[3], big[0]:big[2]].mean())
        assert back < 1.5, f"at the source's size the restored box is {back:.2f} levels from the source"
        proc, r, _ = tabled("restore_off", line, "--capture", str(off))
        piece = r["regions"]["pieces"][PIECE_A]
        assert proc.returncode == 0 and set(piece["restored_px_per_frame"]) == {0} and r["flags"] == [], (piece["restored_px_per_frame"][:3], r["flags"])
        proc, r, _ = tabled("restore_index", f"10-39 {PIECE_A} 10 restore=a.{apparel}\n", "--capture", str(on))
        assert proc.returncode == 0 and [f["rule"] for f in r["flags"]] == ["restored_pixels_beside_a_large_change"], "a class by its index is not the class by its name"
        proc, r, _ = tabled("restore_unknown", f"10-39 {PIECE_A} 10 restore=a.Earring\n", "--capture", str(on))
        assert proc.returncode != 0 and "not a class of the part model" in proc.stderr, proc.stderr[-200:]
        proc, r, _ = tabled("restore_short", line, "--capture", str(short))
        blind = [f for f in r["flags"] if f["rule"] == "restore_has_no_mask"]
        assert blind and blind[0]["source_frames"] == [[25, 39]], [f["rule"] for f in r["flags"]]
        return f"{area} to {grown} px given back a frame; the original inside the box at both sizes; nothing where the piece changed nothing"

    def a_subject_given_back() -> str:
        # RECT_A is x 40-104, y 40-120. The piece's own subject holds its left part, another subject its right part,
        # and they overlap in the middle
        own, other = (40, 40, 80, 120), (64, 40, 104, 120)
        only_other, both, only_own = (86, 50, 100, 110), (68, 50, 76, 110), (46, 50, 58, 110)
        masks = {"a": mask_of(own, 30), "b": mask_of(other, 30)}
        with_run = capture(TMP / "cap_subject", 10, 30, masks, [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}])
        no_run = capture(TMP / "cap_subject_no_run", 10, 30, masks, [])
        o, a = original(20), planes_of(PIECE_A, 20)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1]:rect[3], rect[0]:rect[2]] - y[0][rect[1]:rect[3], rect[0]:rect[2]]).mean())
        row = {"piece": PIECE_A, "piece_first": 10, "restore": [("b", None)]}
        got, record = frame_planes(SOURCE, [dict(row)], 20, tool.Captures([with_run], (W, H)))
        assert box_mean(got, o, only_other) < 1.0, f"where only the other subject is, the frame is {box_mean(got, o, only_other):.2f} from the original"
        assert box_mean(got, a, both) < 1.0, "where both masks are, the piece did not keep its pixels"
        assert box_mean(got, a, only_own) < 1.0, "where only the piece's own subject is, the piece's pixels are gone"
        given = record["rows"][PIECE_A][20]["restored_px"]
        area = (other[2] - own[2]) * (other[3] - other[1])                       # the other's mask outside the piece's own
        assert area <= given <= area + 2 * tool.RESTORE_GROW * (other[3] - other[1] + 40), (area, given)
        got, _ = frame_planes(SOURCE, [dict(row)], 20, tool.Captures([no_run], (W, H)))
        assert box_mean(got, o, both) < 1.0 and box_mean(got, o, only_other) < 1.0, "with no run for the piece the other's whole mask was not given back"
        assert tool.read_table(table(TMP / "subject_row.txt", [(10, 39, PIECE_A, "10 restore=b")]))[0]["restore"] == [("b", None)]
        whole = {"piece": PIECE_A, "piece_first": 10, "restore": [("b", "whole")]}
        got, _ = frame_planes(SOURCE, [whole], 20, tool.Captures([with_run], (W, H)))
        assert box_mean(got, o, both) < 1.0 and box_mean(got, o, only_other) < 1.0, "`:whole` did not give back what both masks claim"
        assert box_mean(got, a, only_own) < 1.0, "`:whole` gave back the piece's own subject where the other is not"
        assert tool.read_table(table(TMP / "whole_row.txt", [(10, 39, PIECE_A, "10 restore=b:whole")]))[0]["restore"] == [("b", "whole")]
        return f"{given} px given back on the frame read ({area} in the other's mask alone); the overlap stays the piece's, and goes back with `:whole`"

    def spill_over_a_cut() -> str:
        cut_at, frames = 20, 40
        clip = TMP / "with_a_cut.mp4"
        other = PICTURE.replace("c0=0x905030", "c0=0x203c8c").replace("c2=0xc8b450", "c2=0x50a0c8").replace("x0=40:y0=30", "x0=300:y0=40")
        run([tool.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:{PICTURE}",
             "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:{other},eq=brightness=-0.35", "-filter_complex",
             f"[0:v]trim=end_frame={cut_at},setpts=PTS-STARTPTS[a];[1:v]trim=end_frame={frames - cut_at},setpts=PTS-STARTPTS[b];"
             "[a][b]concat=n=2:v=1,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[v]",
             "-map", "[v]", "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", *tool.BT709_TAGS, str(clip)])
        whole = loaded(clip, 0, frames)
        luma = whole.mean(dim=-1) * 255.0
        moved = float((luma[cut_at] - luma[cut_at - 1]).abs().mean())
        assert moved > 1.5 * tool.CUT, f"the clip's cut moves the picture {moved:.0f} levels, too near the tool's line of {tool.CUT:g} to test it"

        def piece(name, last_painted):
            out = whole.clone()
            x0, y0, x1, y1 = RECT_A
            out[5:last_painted + 1, y0:y1, x0:x1] = (out[5:last_painted + 1, y0:y1, x0:x1] + 0.5) % 1.0
            return write(name, out)

        def flags_of(name, last_painted):
            path = TMP / f"{name}.txt"
            path.write_text(f"0-{frames - 1} {piece(name + '_piece', last_painted)} 0\n")
            out = TMP / f"{name}.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(clip), "--table", str(path),
                                   "--span", f"0-{frames - 1}", "--out", str(out)], capture_output=True, text=True)
            r = json.loads(Path(str(out) + ".check.json").read_text())
            assert proc.returncode == 0 and r["verdict"] == "passes", r["failures"]
            return r
        def flags_of_rows(name, last_painted):
            """The same piece given as two rows that meet inside the first shot: its flags must not double."""
            path = TMP / f"{name}.txt"
            made = piece(name + "_piece", last_painted)
            path.write_text(f"0-9 {made} 0\n10-{frames - 1} {made} 0\n")
            out = TMP / f"{name}.mp4"
            subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(clip), "--table", str(path),
                            "--span", f"0-{frames - 1}", "--out", str(out)], capture_output=True, text=True)
            return json.loads(Path(str(out) + ".check.json").read_text())["flags"]
        r = flags_of("spill_one", cut_at)
        spill = [f for f in r["flags"] if f["rule"] == "piece_changes_across_a_cut"]
        assert spill and spill[0]["source_frames"] == [[cut_at, cut_at]], [(f["rule"], f["source_frames"]) for f in r["flags"]]
        assert isinstance(r["audio"], str) and "no audio" in r["audio"], r["audio"]
        for name, last in (("spill_none", cut_at - 1), ("spill_follows", cut_at + 12)):
            r = flags_of(name, last)
            assert not [f for f in r["flags"] if f["rule"] == "piece_changes_across_a_cut"], f"{name}: a piece that {('ends at the cut' if last < cut_at else 'goes on past it')} was flagged"
        # what the flag says about the latent step: the piece's load starts at source frame 0, so the cut at 20 falls
        # inside a step; the same pictures as a load starting at frame 2 put the cut on a step's edge
        lp = importlib.import_module("_h3pack.loop_plan")
        assert lp.step_span(cut_at)[0] != cut_at and lp.step_span(cut_at - 2)[0] == cut_at - 2, \
            f"core's step cycle {lp.FRAME_PER_TOKEN} no longer puts frame {cut_at} inside a step and {cut_at - 2} on an edge: the case needs new frames"
        note = spill[0]["figures"]["at_each_cut"][0]
        a, n = lp.step_span(cut_at)
        assert note["latent_step"] == [a, a + n - 1] and note["the_cut_splits_it"] and note["it_holds_the_spill"], note
        late = whole[2:].clone()
        x0, y0, x1, y1 = RECT_A
        late[3:cut_at - 2 + 1, y0:y1, x0:x1] = (late[3:cut_at - 2 + 1, y0:y1, x0:x1] + 0.5) % 1.0
        path = TMP / "spill_edge.txt"
        path.write_text(f"2-{frames - 1} {write('spill_edge_piece', late)} 2\n")
        out = TMP / "spill_edge.mp4"
        subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(clip), "--table", str(path), "--span", f"0-{frames - 1}",
                        "--out", str(out)], capture_output=True, text=True)
        edge = [f for f in json.loads(Path(str(out) + ".check.json").read_text())["flags"] if f["rule"] == "piece_changes_across_a_cut"]
        assert edge and edge[0]["figures"]["at_each_cut"][0]["the_cut_splits_it"] is False and "not that mechanism" in edge[0]["why"], edge
        again = flags_of_rows("spill_rows", cut_at)
        assert [f["rule"] for f in again].count("piece_changes_across_a_cut") == 1, "a piece on two rows raised its flag twice"
        return ("one frame past the cut is flagged, once, and said to be inside a latent step; the same spill from a load whose step ends "
                "at the cut is said not to be; ending at the cut, or going on well past it, is not flagged")

    def by_the_owner_map() -> str:
        # RECT_A is x 40-104, y 40-120. Subject a's track holds x 40-80, b's x 64-104: both claim x 64-80.
        own, other = (40, 40, 80, 120), (64, 40, 104, 120)
        owner = np.full((30, H, W), 255, np.uint8)
        owner[:, 40:120, 40:64] = 0                    # a alone
        owner[:, 40:120, 80:104] = 1                   # b alone
        owner[:, 40:80, 64:80] = 1                     # the strip's top half: b's, by the class maps
        owner[:, 80:120, 64:80] = 254                  # its bottom half: contested
        b_half, disputed, only_other, only_own = (67, 46, 77, 74), (67, 86, 77, 114), (86, 50, 100, 110), (46, 50, 58, 110)
        folder = capture(TMP / "cap_owners", 10, 30, {"a": mask_of(own, 30), "b": mask_of(other, 30)},
                         [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16},
                          {"name": "run_c", "render": "piece_c.mp4", "subject": "b", "margin_px": 16}])
        np.savez_compressed(folder / "owners.npz", owner=owner, labels=np.array(["a", "b"]), nobody=255, contested=254)
        caps = tool.Captures([folder], (W, H))
        o, a, c = original(20), planes_of(PIECE_A, 20), planes_of(PIECE_C, 20)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1]:rect[3], rect[0]:rect[2]] - y[0][rect[1]:rect[3], rect[0]:rect[2]]).mean())
        got, _ = frame_planes(SOURCE, [{"piece": PIECE_A, "piece_first": 10, "restore": [("b", None)]}], 20, caps)
        assert box_mean(got, o, only_other) < 1.0 and box_mean(got, o, b_half) < 1.0, "what the other subject owns was not given back"
        assert box_mean(got, a, disputed) < 1.0 and box_mean(got, a, only_own) < 1.0, "a contested pixel, or the piece's own, was given back"
        assert caps.used_owner_map, "the owner map was not read"
        # two pieces that both changed the strip: RECT_C is x 80-150, y 80-150, so use a piece painted on the strip itself
        strip = write("strip", painted(WHOLE[10:40], (60, 40, 84, 120)))
        s_ = planes_of(strip, 20)
        rows = [{"piece": strip, "piece_first": 10}, {"piece": PIECE_A, "piece_first": 10}]
        folder2 = capture(TMP / "cap_owners_2", 10, 30, {"a": mask_of(own, 30), "b": mask_of(other, 30)},
                          [{"name": "run_s", "render": "strip.mp4", "subject": "b", "margin_px": 16},
                           {"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}])
        np.savez_compressed(folder2 / "owners.npz", owner=owner, labels=np.array(["a", "b"]), nobody=255, contested=254)
        got, record = frame_planes(SOURCE, rows, 20, tool.Captures([folder2], (W, H)))
        assert box_mean(got, s_, b_half) < 1.0, "a pixel both pieces changed and the first row's subject owns went to the later row"
        assert box_mean(got, a, disputed) < 1.0, "a contested pixel both pieces changed did not fall to the later row"
        assert record["shared"][20]["by_mask"] > 0 and record["shared"][20]["by_order"] > 0, record["shared"][20]
        return f"a restore gives back what the map says the other owns and not what it calls contested; of the pixels two pieces changed, {record['shared'][20]['by_mask']} went by the map and {record['shared'][20]['by_order']} by order"

    def the_subjects_table() -> str:
        # subject b: x 80-128, y 50-110. RECT_A covers x 40-104, so its left half (x 80-104) is painted, the right is not.
        b = (80, 50, 128, 110)
        cmap = np.zeros((30, H, W), np.uint8)
        cmap[:, 50:110, 80:104] = 4                    # on the painted half
        cmap[:, 50:110, 104:128] = 23                  # on the unpainted half
        names = tool.class_names()
        folder = capture(TMP / "cap_table", 10, 30, {"a": mask_of((40, 40, 78, 120), 30), "b": mask_of(b, 30)},
                         [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}], {"b": cmap})
        got = {}
        for name, restore, size in (("table_plain", "", "canvas"), ("table_restore", " restore=b", "canvas"), ("table_plain_full", "", "source")):
            path = TMP / f"{name}.txt"
            path.write_text(f"10-39 {PIECE_A} 10{restore}\n")
            out = TMP / f"{name}.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(SOURCE), "--table", str(path), "--span", "10-39",
                                   "--out", str(out), "--capture", str(folder), "--size", size], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr[-300:] or proc.stdout[-300:]
            got[name] = json.loads(Path(str(out) + ".check.json").read_text())["subjects"]
        for name in ("table_plain", "table_plain_full"):
            t = got[name]
            sub = t["subjects"]["b"]
            assert 40 <= sub["tracked_off_pct"] <= 60, f"{name}: half of b's tracked pixels are painted, the table reads {sub['tracked_off_pct']}%"
            hair, cloth = sub["by_class"][names[4]], sub["by_class"][names[23]]
            assert hair["off_pct_in_its_track"] > 85 and cloth["off_pct_in_its_track"] < 2, (name, hair, cloth)
            assert t["floor_off_pct"] < 0.5, f"{name}: the floor reads {t['floor_off_pct']}%"
        back = got["table_restore"]["subjects"]["b"]
        assert back["tracked_off_pct"] < 2 and back["by_class"][names[4]]["off_pct_in_its_track"] < 2, back
        # the reading itself, on a picture that runs from black to white: the tool's read of a file at the canvas must
        # be the fitted original of that same file, dark and bright ends included
        ramp = TMP / "ramp.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
             f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.02:nb_colors=2:c0=black:c1=white:x0=0:y0=0:x1={SW}:y1={SH}:seed=1",
             "-frames:v", "6", "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv", "-c:v", "libx264",
             "-crf", "10", "-pix_fmt", "yuv420p", *tool.BT709_TAGS, str(ramp)])
        read = list(tool.luma_at_canvas(ramp, W, H))
        fitted = [tool.planes(b, W, H)[0] for b in tool.original_frames(ramp, 0, 5, W, H)]
        assert len(read) == 6 and min(float(f.min()) for f in fitted) < 40 and max(float(f.max()) for f in fitted) > 200, "the ramp does not reach dark and bright"
        apart = max(float((np.abs(r - f) > tool.OFF).mean()) for r, f in zip(read, fitted))
        assert apart < 0.005, f"{100 * apart:.1f}% of a file's own pixels read as off its fitted original: the two readings are in different levels"
        proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--compare", str(TMP / "table_plain.mp4.check.json"),
                               str(TMP / "table_restore.mp4.check.json")], capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.strip().startswith("b: its tracked mask")), "")
        assert proc.returncode == 0 and "->" in line and "restore=b" in proc.stdout, proc.stdout[-400:] or proc.stderr[-300:]
        before, after = (float(x) for x in line.split(")")[1].split(";")[0].split("->"))
        assert 40 <= before <= 60 and after < 2, line
        return f"b's tracked mask {got['table_plain']['subjects']['b']['tracked_off_pct']:.1f}% off the source, {back['tracked_off_pct']:.1f}% with the restore; the floor {got['table_plain']['floor_off_pct']}%"

    def a_patch_as_a_later_row() -> str:
        hole = (18, 24)
        redo = (60, 60, 96, 110)                           # inside RECT_A: what the patch draws again
        base = loaded(Path(PIECE_A), 0, 30)                # the piece's own frames, as a loader hands them over
        patched = base.clone()
        x0, y0, x1, y1 = redo
        a, b = hole[0] - 10, hole[1] - 10 + 1
        patched[a:b, y0:y1, x0:x1] = (patched[a:b, y0:y1, x0:x1] * 0.3 + 0.05).clamp(0, 1)
        patch = write("patch_of_a", patched)
        rows = [{"piece": PIECE_A, "piece_first": 10}, {"piece": patch, "piece_first": 10}]
        o, piece_, patch_ = original(20), planes_of(PIECE_A, 20), planes_of(patch, 20)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3] - y[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3]).mean())
        assert box_mean(patch_, piece_, redo) > 20 and box_mean(patch_, o, redo) > 20, "the patch's redo is not different enough to tell"
        got, record = frame_planes(SOURCE, [dict(r) for r in rows], 20)
        rest_of_a = (RECT_A[0], RECT_A[1], redo[0], RECT_A[3])       # the part of the piece's rectangle the patch left alone
        assert box_mean(got, patch_, redo) < 1.0, "on a hole frame the redone part is not the patch's"
        assert box_mean(got, patch_, rest_of_a) < 1.0 and box_mean(got, piece_, rest_of_a) < 1.5, "on a hole frame the rest of the piece's change is gone"
        assert box_mean(got, o, (150, 20, 240, 60)) < 1.0, "pixels neither changed are not the original's"
        shared = record["shared"][20]
        assert shared["px"] > 0 and shared["by_mask"] == 0, shared
        code, r, out = deliver(TMP, "patch_row", [(10, 39, PIECE_A, 10), (hole[0], hole[1], patch, 10)], "10-39")
        assert code == 0 and r["verdict"] == "passes", r["failures"]
        both = [f for f in r["flags"] if f["rule"] == "two_pieces_change_the_same_pixels"]
        assert both and both[0]["source_frames"] == [[hole[0], hole[1]]], [(f["rule"], f["source_frames"]) for f in r["flags"]]
        assert [seg[:2] + [len(seg[2])] for seg in r["segments"]] == [[10, 17, 1], [18, 24, 2], [25, 39, 1]], r["segments"]
        assert r["order"]["squares_apart"] == 0, r["order"]
        # the control: the same file read against the table WITHOUT the patch row is the wrong picture over a part of
        # seven frames, every frame in order, and nothing else in the check says so
        code, wrong = recheck(TMP, "patch_row_without_the_patch", [(10, 39, PIECE_A, 10)], "10-39", out)
        said = [f for f in wrong["failures"] if "hold a part" in f]
        assert code == 1 and said and f"[[{hole[0]}, {hole[1]}]]" in said[0], f"a file with another row over part of it passed: {wrong['failures']}"
        assert wrong["order"]["nearer_a_neighbour"] == 0 and len(wrong["failures"]) == 1, (wrong["order"], wrong["failures"])
        return (f"on the hole's frames the patch shows over the piece ({shared['px']} px both changed, by order); elsewhere the piece; "
                f"read against the table without the patch row the file fails on {wrong['order']['squares_apart']} squares of frames "
                f"{hole[0]}-{hole[1]}, its frames in order and no other proof failing")

    def a_patch_with_a_capture() -> str:
        hole = (18, 24)
        redo = (60, 60, 96, 110)
        patched = loaded(Path(PIECE_A), 0, 30).clone()
        x0, y0, x1, y1 = redo
        a, b = hole[0] - 10, hole[1] - 10 + 1
        patched[a:b, y0:y1, x0:x1] = (patched[a:b, y0:y1, x0:x1] * 0.3 + 0.05).clamp(0, 1)
        patch = write("patch_with_capture", patched)
        # subject a holds the whole of RECT_A; subject c holds RECT_C less its overlap with RECT_A
        c_mask = mask_of(RECT_C, 30)
        c_mask[:, RECT_A[1]:RECT_A[3], RECT_A[0]:RECT_A[2]] = False
        folder = capture(TMP / "cap_patch", 10, 30, {"a": mask_of(RECT_A, 30, grow=4), "c": c_mask},
                         [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16},
                          {"name": "run_c", "render": "piece_c.mp4", "subject": "c", "margin_px": 16}])
        caps = tool.Captures([folder], (W, H))
        patch_, piece_, c_ = planes_of(patch, 20), planes_of(PIECE_A, 20), planes_of(PIECE_C, 20)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3] - y[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3]).mean())
        only_redo = (redo[0], redo[1], RECT_C[0], redo[3])                 # redrawn by the patch, not touched by piece C
        triple = (RECT_C[0], RECT_C[1], redo[2], redo[3])                  # changed by the piece, its patch and piece C; a holds it
        premise = [round(box_mean(patch_, piece_, only_redo), 1), round(box_mean(patch_, c_, triple), 1), round(box_mean(patch_, piece_, triple), 1)]
        assert min(premise) > 8, f"the three pieces are not different enough where they meet to tell which shows: {premise}"
        for said in ({}, {"subject": "a"}):
            rows = [{"piece": PIECE_A, "piece_first": 10}, {"piece": patch, "piece_first": 10, **said}]
            got, record = frame_planes(SOURCE, [dict(r) for r in rows], 20, caps)
            assert box_mean(got, patch_, only_redo) < 1.0, f"with {said or 'nothing said'} on its row the patch does not show over the piece it patches"
            rows = [{"piece": PIECE_A, "piece_first": 10}, {"piece": patch, "piece_first": 10, **said}, {"piece": PIECE_C, "piece_first": 10}]
            got, record = frame_planes(SOURCE, [dict(r) for r in rows], 20, caps)
            if said:
                assert box_mean(got, patch_, triple) < 1.0, "where a holds a pixel three pieces changed, a's patch does not show"
        lines = "".join(f"{a_}-{b_} {piece} {first}{more}\n" for a_, b_, piece, first, more in (
            (10, 39, PIECE_A, 10, ""), (hole[0], hole[1], patch, 10, " subject=a"), (10, 39, PIECE_C, 10, "")))
        (TMP / "patch_capture.txt").write_text(lines)
        out = TMP / "patch_capture.mp4"
        proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(SOURCE), "--table", str(TMP / "patch_capture.txt"),
                               "--span", "10-39", "--out", str(out), "--capture", str(folder)], capture_output=True, text=True)
        r = json.loads(Path(str(out) + ".check.json").read_text())
        assert proc.returncode == 0 and r["verdict"] == "passes", r["failures"]
        shares = {row["row"]: row["share"] for row in r["rows_shown"]}
        assert len(shares) == 3 and all(v is not None and v >= tool.SHOWN for v in shares.values()), shares
        assert r["pieces_with_no_capture"] == [], r["pieces_with_no_capture"]
        return f"the patch shows with and without `subject=`; where three pieces changed a pixel a holds, a's patch shows; rows shown {sorted(shares.values())}"

    def a_locked_file() -> str:
        folder = TMP / "locked"
        folder.mkdir()
        rows = [(10, 39, PIECE_A, 10)]

        def go(name, *more):
            out = folder / f"{name}.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(SOURCE), "--table", str(table(folder / f"{name}.txt", rows)),
                                   "--span", "10-39", "--out", str(out), *more], capture_output=True, text=True)
            return proc, out
        proc, out = go("accepted")
        assert proc.returncode == 0, proc.stderr[-300:]
        was = out.read_bytes()
        md5 = tool.md5_of(str(out))
        (folder / tool.LOCKS).write_text(f"# Locked outputs\n\nSome words.\n\n- `accepted.mp4`  md5 {md5}  locked today (a note, with `backticks` in it).\n"
                                         "  A second line naming `other.mp4` in passing.\n")
        proc, _ = go("accepted")
        said = proc.stderr + proc.stdout
        assert proc.returncode != 0 and "is locked" in said and "accepted_b.mp4" in said, f"a build over a locked file was not refused: {said[-300:]}"
        assert out.read_bytes() == was, "the refused build changed the locked file"
        proc, _ = go("accepted", "--check-only")
        record = json.loads(Path(str(out) + ".check.json").read_text())
        assert proc.returncode == 0 and record["locked"]["md5_now"] == md5 and out.read_bytes() == was, (proc.returncode, record.get("locked"))
        proc, other = go("other")
        assert proc.returncode == 0 and other.is_file(), "a name only mentioned in the lock file's prose was refused"
        out.write_bytes(other.read_bytes()[:-7] + b"changed")
        proc, _ = go("accepted", "--check-only")
        record = json.loads(Path(str(out) + ".check.json").read_text())
        assert proc.returncode == 1 and any("not the file that was accepted" in f for f in record["failures"]), record["failures"]
        return "a build over it is refused and names accepted_b.mp4; a re-read passes and leaves its bytes; changed bytes fail; a name not listed builds"

    case("a piece that changed nothing is the original", nothing_changed)
    case("a painted rectangle is found where it is", rectangle_found)
    case("a delivery of pieces and original ranges passes its own check", delivery_passes)
    case("controls: three files that must fail", controls_fail)
    case("a span from the middle begins its track at the span", middle_span)
    case("two pieces on the same frames", two_pieces)
    case("the flags", flags)
    case("at the source's size", at_the_sources_size)
    case("a class given back to the source", a_class_given_back)
    case("another subject given back whole", a_subject_given_back)
    case("a piece that spills over a cut", spill_over_a_cut)
    case("whose a pixel is, from the owner map", by_the_owner_map)
    case("what the record says the file did to a subject", the_subjects_table)
    case("a patch as a later row", a_patch_as_a_later_row)
    case("a patch as a later row, with a capture", a_patch_with_a_capture)
    case("a locked file is not built over", a_locked_file)
sys.exit(finish())
