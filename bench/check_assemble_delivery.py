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
   is far from its subject's mask; one whose changed area jumps; one that changes nothing. And the case that
   must raise none: the rectangle on its subject's mask, the same size throughout. The tables land in the
   capture folder.

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
    run([tool.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:nb_colors=5:seed=7",
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


def capture(folder: Path, first: int, frames: int, subjects: dict, runs: list) -> Path:
    """A capture folder as `bench/capture_masked_run.py` lays one out, holding only what the assembler reads."""
    folder.mkdir(parents=True)
    manifest = {"name": folder.name, "first_frame": first, "frames": frames, "size": [W, H],
                "subjects": [{"label": k, "sightings": [{"by": "made"}]} for k in subjects], "runs": runs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    for label, mask in subjects.items():
        (folder / "subjects" / label).mkdir(parents=True)
        np.savez_compressed(folder / "subjects" / label / "masks__made.npz", track=np.packbits(mask, axis=-1),
                            covered=np.ones(frames, bool))
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
        worst = 0
        for f in (10, 25, 39):
            worst = max(worst, int(tool.changed(planes_of(KEPT, f), original(f)).sum()))
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
        return "as the writer writes today, and in its older untagged form"

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
        assert list(got) == ["piece_changes_far_more_than_it_usually_does"], list(got)
        assert got["piece_changes_far_more_than_it_usually_does"]["source_frames"] == [[34, 39]], got
        assert r["pieces_with_no_capture"] == ["jumpy.mp4"], r["pieces_with_no_capture"]
        return "no mask, a mask far away, a jump in area, nothing changed; none on a piece that stays on its subject"

    case("a piece that changed nothing is the original", nothing_changed)
    case("a painted rectangle is found where it is", rectangle_found)
    case("a delivery of pieces and original ranges passes its own check", delivery_passes)
    case("controls: three files that must fail", controls_fail)
    case("a span from the middle begins its track at the span", middle_span)
    case("two pieces on the same frames", two_pieces)
    case("the flags", flags)
sys.exit(finish())
