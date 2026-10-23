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
    `restore=<subject>:whole` it goes back whether or not there is a run. The same capture written with other
    labels for the two people gives the same picture when read through `--alias`, with its owner map and its own
    flag under the table's names; two labels of one capture read as one subject are refused.
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
    their subject owns. The record says an owner map was used. A class restore naming another subject than the
    row's own keeps off what the map gives the row's own subject, its rim included, and still gives back its own
    pixels and the contested ones; the row's own subject's class is given back wherever its class map has it, the
    other subject's pixels included; with no owner map nothing is kept off.
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
17. **A shot where a named subject is the source's own.** On the clip with a cut, a piece that changes its subject
    on the first shot and nothing on the second. With the subject tracked on the second shot the build fails and
    names the shot's frames; so it does when the tracker called them absent there and the capture's own flag, or
    a shot table, says people were detected. It passes, with the shot listed under its reading, when the table
    says so in a `source` line (the words kept), when the people detected are another subject's that is laid
    there, when nobody was detected, and when nothing is known. A `source` line over a shot that has something
    laid, and one that covers no whole shot, fail. Consecutive frames that each move over the line are one cut, not
    a shot a frame; and where a capture's shot table covers the frames, its cuts are the cuts. The frames of the first shot where the subject is tracked and
    nothing is laid yet are listed with the longest run. The first locked file passed every proof with a shot of
    29 frames on which nothing was laid for either of its subjects; this is the proof that would have failed it.
18. **A frame its run carried no mask on.** A piece painted on every frame, and a capture whose run carried a mask
    on all frames but five. With the run's carried mask the node's own, those five frames are the original and the
    others the piece's; the record's flag names the five and says they were left; the shots proof does not count
    them as laid. With the same mask said to be read off a review, nothing is dropped and the flag says that. With
    no carried mask in the capture the piece is laid on every frame, as before.
19. **A row that ends, or begins, on the source's own picture.** A piece painted on every frame of a row whose
    paint thins to nothing over its last five frames, its subject tracked throughout: the build fails and names
    those frames; the same thinning at the start fails as "begins"; a piece that holds to both ends passes with
    its two ends near the middle's level in the record. Cut short of the thinned frames the row passes (and the
    frames are then listed as tracked with nothing laid). And where this reading passes but the capture's own
    reading of the render (`runs/<run>/changed.json`) says the render ends on the source on frames the row lays,
    the build fails on the capture's word. Three real short loads ended this way and every other proof passed
    them.
20. **A piece rendered zoomed in.** A piece, the picture it was rendered from at twice the size inside a box, and
    the json saying the box, made so that the piece is exactly that picture reduced into the box. At the source's
    size the file passes, its record names the zoomed file and reads the two as agreeing, and inside the region the
    delivered picture carries more of the zoomed file's fine stripes than the same piece laid without it; at the
    canvas's size the zoomed file is not used and the file is the plain piece's. With the box in the json moved
    eight pixels the build fails on the agreement and names the piece. Refused before anything is read: no zoomed
    file beside the json, a zoomed file with another frame count, a box not fixed, another run's name.
21. **A row that names its capture's run** (`run=<capture folder>:<run name>`). A capture made as a plan, before its
    render, whose run names no render, as every load of the masked lane's does. With `subject=` alone on the row
    nothing is read under the mask the run carried and no table is written into the capture (the control: the
    state before 2026-10-13). With the run named, the five frames the run carried nothing on are the source's
    and are named, and the capture gets its table. Refused before anything is written: a folder that is not a
    capture given, a run the capture does not have, a run of another subject than the row's, a run that names
    another render, no `--capture` at all, and a token that is no run.
22. **A thing worn on the head, placed on the piece's own face** (2026-10-23). A plate whose glasses are a white
    box, her Eyeglass labelled there, the render's face points the film's moved by a known shift: the white comes to
    the box moved and the film's place shows the piece, at both sizes; a hat on the same row moved by the ears' shift;
    held where no point is scored or the shift
    jumps; kept where the film has it, byte for byte, where her hand given back in place touches it; nothing laid and no hole where there is no label; with `--worn-fill` the nearest labelled frame's mask and
    pixels moved by this frame's render point less that frame's film point; clipped off the piece; refused with no
    render points or film points (the plain restore, flagged, the build passing); the shift smoothed where one
    frame's reading steps; `:filmed` and a hand restore the same bytes.

## Running it

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_assemble_delivery.py

Imports the pack's writer, so ComfyUI core must be importable; no card and no server.
"""
from __future__ import annotations

import importlib
import json
import shutil
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


def make_source(path: Path, pcm: bool = False) -> None:
    """A soft picture that drifts, different on every frame, tagged BT.709, with a tone under it. One keyframe, so
    a straight cut has a long head. Soft and slow on purpose: on a hard test pattern that moves fast (`testsrc2`)
    the writer's own codec noise reaches twice `CHANGE` and a piece that kept every pixel reads as changed
    (measured 2026-10-10); `CHANGE` was measured on footage, and that is the picture it holds for."""
    run([tool.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:{PICTURE}",
         "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100", "-frames:v", str(FRAMES), "-shortest",
         "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv", "-c:v", "libx264", "-crf", "12",
         "-g", "250", "-pix_fmt", "yuv420p", *tool.BT709_TAGS,
         *(["-c:a", "pcm_s16le", "-ac", "2", "-f", "mov"] if pcm else ["-c:a", "aac", "-ac", "2", "-b:a", "96k"]), str(path)])


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


def faint(frames: torch.Tensor, rect, by: float = 4 / 255) -> torch.Tensor:
    """The frames with a rectangle raised `by` in every channel: under `CHANGE`, as a whole-region render's barely-changed pixels are."""
    out = frames.clone()
    x0, y0, x1, y1 = rect
    out[:, y0:y1, x0:x1] = (out[:, y0:y1, x0:x1] + by).clamp(0.0, 1.0)
    return out


def graph_beside(piece: str, composite: str | None) -> None:
    """The PNG VHS saves beside a render, holding the graph's `prompt` with the Masked Source's composite; None writes none."""
    png = Path(piece).with_suffix(".png")
    png.unlink(missing_ok=True)
    if composite is not None:
        from PIL import Image, PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", json.dumps({"104": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"composite": composite}}}))
        Image.new("RGB", (4, 4)).save(png, pnginfo=info)


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
    if "--check-only" not in more and out.is_file():
        # every built clip has its sidecar (bench/sidecar.py), of its bytes, naming the table, the source, the check record and each piece
        import sidecar
        try:
            side = sidecar.read(out)
        except SystemExit as e:
            raise AssertionError(f"a built clip's sidecar does not hold: {e}") from None
        roles = {x["role"]: x["path"] for x in side["inputs"]}
        assert roles["the table"] == str(tmp / f"{name}.txt") and roles["the source"] == str(tmp / "source.mp4") \
            and roles["the check record"] == str(report), roles
        lo, hi = (int(v) for v in span.split("-"))
        laid = [r for r in rows if r[0] <= hi and r[1] >= lo]          # the rows the span lays; the others are listed as not laid
        assert [v for k, v in roles.items() if k.startswith("piece ") and "zoomed" not in k] == [str(Path(r[2]).resolve()) for r in laid], roles
        assert side["extra"]["table_rows_not_laid"] == [[r[0], r[1], str(r[2])] for r in rows if r not in laid], side["extra"]
        assert "no " in side["extra"]["pieces_why"] and side.get("pieces") is None, side["extra"]   # no table JSON here: said, not guessed
    return proc.returncode, json.loads(report.read_text()), out


def recheck(tmp: Path, name: str, rows, span: str, out: Path) -> tuple[int, dict]:
    import sidecar
    try:
        before = sidecar.read(out)
    except SystemExit:
        before = None
    proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(tmp / "source.mp4"),
                           "--table", str(table(tmp / f"{name}.txt", rows)), "--span", span, "--out", str(out), "--check-only"],
                          capture_output=True, text=True)
    report = json.loads(Path(str(out) + ".check.json").read_text())
    if out.is_file():
        # --check-only writes the sidecar again: its verdict is this check's, and a build's sidecar of the same bytes is kept as `earlier`
        try:
            side = sidecar.read(out)
        except SystemExit as e:
            raise AssertionError(f"--check-only left the clip with no sidecar of its bytes: {e}") from None
        # --check-only appends the check; the build's record (its command, inputs, pieces) stays whole
        assert side["checks"][-1]["verdict"] == report["verdict"] and "--check-only" in side["checks"][-1]["command"], side.get("checks")
        if before is not None:
            assert {k: side.get(k) for k in ("command", "inputs", "extra", "pieces")} == {k: before.get(k) for k in ("command", "inputs", "extra", "pieces")}, \
                "a check replaced the build's record"
    return proc.returncode, report


def capture(folder: Path, first: int, frames: int, subjects: dict, runs: list, classes: dict | None = None,
            flags: list | None = None, shots: dict | None = None, carried: dict | None = None, region: dict | None = None) -> Path:
    """A capture folder as `bench/capture_masked_run.py` lays one out, holding only what the assembler reads.
    `classes[label]` is a class map [frames, H, W] of class indices; `flags` is its flags.json; `shots[label]` is
    that subject's shot table as [(first frame, last frame, people detected, the tracker's word)]; `carried[run]`
    is the mask that run carried, [frames, H, W], as its windows saved it."""
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
        if shots and label in shots:
            table = {"cuts": [a for a, _b, _n, _w in shots[label]][1:],
                     "shots": [{"first_frame": a, "last_frame": b, "people": [{"person": k + 1} for k in range(n)],
                                "subject": {"state": word}} for a, b, n, word in shots[label]]}
            (folder / "subjects" / label / "shots__made.json").write_text(json.dumps(table))
    if flags is not None:
        (folder / "flags.json").write_text(json.dumps({"flags": flags}))
    for name in set(carried or {}) | set(region or {}):
        (folder / "runs" / name).mkdir(parents=True)
        np.savez_compressed(folder / "runs" / name / "region.npz", read=np.ones(frames, bool),
                            **({"carried": np.packbits(carried[name], axis=-1)} if name in (carried or {}) else {}),
                            **({"region": region[name]} if name in (region or {}) else {}))
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

    def pcm_by_its_samples() -> str:
        """A PCM track (fun's source: pcm_s16le in a .mov) copied into the file can be the source's samples cut into
        other packets (fun's 1408Z cut). Decoded in place, sample for sample, it is the source's track and passes with a
        note; a PCM track with its samples changed fails. Until 2026-10-21 any packet split but the source's failed."""
        d = TMP / "pcm"
        d.mkdir()
        make_source(d / "source.mp4", pcm=True)
        code, r, out = deliver(d, "whole_pcm", ROWS, f"0-{FRAMES - 1}")
        assert code == 0 and r["verdict"] == "passes", r["failures"]
        # the same samples in other packets: PCM to PCM in 1000-sample packets is lossless
        split = d / "split_otherwise.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(out), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
             "-af", "asetnsamples=n=1000:p=0", "-c:a", "pcm_s16le", str(split)])
        assert np.array_equal(tool.pcm(out), tool.pcm(split)), "the fixture changed the samples"
        assert [p[1:3] for p in tool.audio_packets(out)] != [p[1:3] for p in tool.audio_packets(split)], "the fixture kept the packets"
        code, r = recheck(d, "whole_pcm", ROWS, f"0-{FRAMES - 1}", split)
        a = r["audio"]
        assert code == 0 and r["verdict"] == "passes", r["failures"]
        assert a["a_run_of_the_source's_packets"] is False and a["decoded_samples_are_the_source's_in_place"] is True and a.get("note"), a
        # the same samples, delayed half a second: decoded, they are the same bytes, so the place is read off the packets
        late = d / "split_and_late.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(out), "-itsoffset", "0.5", "-i", str(split), "-map", "0:v:0", "-map", "1:a:0",
             "-c", "copy", str(late)])
        assert np.array_equal(tool.pcm(late), tool.pcm(out)), "the delayed fixture changed the samples"
        code, r = recheck(d, "whole_pcm", ROWS, f"0-{FRAMES - 1}", late)
        assert code == 1 and any("off the span" in f for f in r["failures"]), (r["failures"], r["audio"])
        # the source's own samples from the start, cut short at half the span
        short = d / "split_and_short.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(split), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
             "-af", f"atrim=end_sample={FRAMES * 44100 * 1001 // 24000 // 2}", "-c:a", "pcm_s16le", str(short)])
        code, r = recheck(d, "whole_pcm", ROWS, f"0-{FRAMES - 1}", short)
        assert code == 1 and any("samples where the span is" in f for f in r["failures"]), (r["failures"], r["audio"])
        # a sample off: the same packets' worth, at half the level
        quieter = d / "samples_changed.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(out), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
             "-af", "volume=0.5", "-c:a", "pcm_s16le", str(quieter)])
        code, r = recheck(d, "whole_pcm", ROWS, f"0-{FRAMES - 1}", quieter)
        assert code == 1 and any("decoded samples are not the source's in place" in f for f in r["failures"]), r["failures"]
        # a span from the middle: the samples are held at the span's own place in the source, not from its start
        code, r, mid = deliver(d, "middle_pcm", ROWS, "30-60")
        assert code == 0 and r["verdict"] == "passes", r["failures"]
        mid_split = d / "middle_split_otherwise.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(mid), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
             "-af", "asetnsamples=n=1000:p=0", "-c:a", "pcm_s16le", str(mid_split)])
        code, r = recheck(d, "middle_pcm", ROWS, "30-60", mid_split)
        m = r["audio"]
        assert code == 0 and m.get("decoded_samples_are_the_source's_in_place") is True, (r["failures"], m)
        return (f"{a['samples_compared']} samples, the source's in place in other packets, pass; the same delayed half a second, "
                f"cut to half its length, or at half the level fails; "
                f"a span from the middle held at its own place ({m['samples_compared']} samples)")

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
            # a piece that changed nothing while its subject is tracked on every frame is also a shot with nothing
            # laid for a named subject, and that is a failure of the build, not only a flag
            fails = [f for f in r["failures"] if "nothing laid for a" in f and "tracked, nothing laid" in f]
            assert got == want and (code, len(r["failures"])) == ((1, 1) if name == "idle" else (0, 0)) and bool(fails) == (name == "idle"), \
                f"{name}: flags {got}, expected {want}; exit {code}, failures {r['failures']}"
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

    def the_cut_rows_carry_the_edge() -> str:
        """The lead's ruling of 2026-10-15 (the clips reviewer's C3): where a laid region reaches the edge of the loader's crop, the rows
        or columns cut beyond it take the laid edge as far along it as the piece reaches, in full for CARRY_FULL_PX and fading to the
        source over CARRY_FADE_PX; elsewhere, under a restore, and past that depth they stay the source's."""
        def lay(sw, sh, box, reach_rows=None, reach_cols=None, held=None, reach_top=None, reach_right=None):
            cw, ch, x0, y0 = box
            base = [np.full((sh, sw), 80.0, np.float32), np.full((sh // 2, sw // 2), 120.0, np.float32), np.full((sh // 2, sw // 2), 120.0, np.float32)]
            for k, sc in enumerate((1, 2, 2)):                 # inside the crop the source is 50 / 100; its own cut picture 80 / 120
                base[k][y0 // sc:(y0 + ch) // sc, x0 // sc:(x0 + cw) // sc] = 50.0 if k == 0 else 100.0
            piece = [np.full((H, W), 200.0, np.float32), np.full((H // 2, W // 2), 160.0, np.float32), np.full((H // 2, W // 2), 160.0, np.float32)]
            owner = np.full((H, W), -1, np.int8)
            if reach_rows is not None:
                owner[H - 40:, reach_rows] = 0
            if reach_cols is not None:
                owner[reach_cols, :40] = 0
            if reach_top is not None:
                owner[:40, reach_top] = 0
            if reach_right is not None:
                owner[reach_right, W - 40:] = 0
            raw = np.frombuffer(tool.laid_over(base, [piece], owner, box, [held] if held is not None else None), np.uint8)
            y = raw[:sw * sh].reshape(sh, sw).astype(int)
            u = raw[sw * sh:sw * sh + (sw // 2) * (sh // 2)].reshape(sh // 2, sw // 2).astype(int)
            v = raw[sw * sh + (sw // 2) * (sh // 2):].reshape(sh // 2, sw // 2).astype(int)
            return y, u, v
        cw, ch, x0, y0 = tool.crop_of(SW, SH, W, H)
        assert y0 > 0 and y0 + ch < SH and x0 == 0, (cw, ch, x0, y0)
        y, u, v = lay(SW, SH, (cw, ch, x0, y0), reach_rows=slice(0, W // 2))   # the piece reaches the bottom on the left half, not the top
        left, right, below = slice(20, int(0.4 * SW)), slice(int(0.65 * SW), SW - 20), slice(y0 + ch, SH)
        # THE CASE THAT GOES RED when the cut rows under a region that reaches the edge keep the source (a band), or take the render where
        # nothing reached it, in luma or either chroma plane
        assert (y[below, left] >= 195).all() and (y[below, right] == 80).all() and (y[:y0] == 80).all(), (y[below, left].min(), y[below, right].max())
        for c in (u, v):
            assert (c[(y0 + ch) // 2:, 10:int(0.2 * SW)] >= 155).all() and (c[(y0 + ch) // 2:, int(0.35 * SW):SW // 2 - 10] == 120).all(), c[-1, ::40]
        import cv2
        # A MATTE ROW'S SOFT BAND REACHES THE EDGE (the D18 read): it owns no pixel there, and what it lays there is carried all the same.
        # The band is the bottom 40 rows of the left half at half weight; the piece there is the node's blend, half render (200) and half
        # the canvas's source (50): 125, laid on the sharp source as (1 - 0.5) * 50 + 0.5 * 200 = 125
        base_m = [np.full((SH, SW), 80.0, np.float32), np.full((SH // 2, SW // 2), 120.0, np.float32), np.full((SH // 2, SW // 2), 120.0, np.float32)]
        for k_, sc in enumerate((1, 2, 2)):
            base_m[k_][y0 // sc:(y0 + ch) // sc, x0 // sc:(x0 + cw) // sc] = 50.0 if k_ == 0 else 100.0
        orig_m = [np.full((H, W), 50.0, np.float32), np.full((H // 2, W // 2), 100.0, np.float32), np.full((H // 2, W // 2), 100.0, np.float32)]
        mt = np.zeros((H, W), np.float32)
        mt[H - 40:, :W // 2] = 0.5
        piece_m = [o_ + (r_ - o_) * (mt if o_.shape == mt.shape else cv2.resize(mt, (o_.shape[1], o_.shape[0]), interpolation=cv2.INTER_AREA))
                   for o_, r_ in zip(orig_m, (np.full((H, W), 200.0, np.float32), np.full((H // 2, W // 2), 160.0, np.float32),
                                              np.full((H // 2, W // 2), 160.0, np.float32)))]
        raw_m = np.frombuffer(tool.laid_over(base_m, [piece_m], np.full((H, W), -1, np.int8), (cw, ch, x0, y0), None, None, exact=[True],
                                             orig=orig_m, mats=[mt]), np.uint8)
        ym = raw_m[:SW * SH].reshape(SH, SW).astype(int)
        edge_row = ym[y0 + ch - 1, left]
        # THE CASE THAT GOES RED when a matte row's soft band is not counted in what reaches the edge, and the cut rows keep the source
        assert (abs(edge_row - 125) <= 2).all() and (abs(ym[y0 + ch:y0 + ch + tool.CARRY_FULL_PX, left] - 125) <= 2).all() \
            and (ym[below, right] == 80).all(), (
                f"a matte row's soft band at the crop edge was not carried into the cut rows: edge {int(edge_row.min())}..{int(edge_row.max())}, "
                f"cut rows {int(ym[y0 + ch:y0 + ch + 3, left].min())}, unreached {int(ym[below, right].max())}")
        # THE FEATHER'S INNER EDGE AT THE SOURCE'S SIZE (the second D18 read): a matte row's full region and its soft band are one layer, laid in
        # one pass; laid in two, where the upscale blends their weights a quarter of the change fell back toward what lies beneath, on a ring
        sw3, sh3 = 2 * W, 2 * H
        orig3 = [np.full((H, W), 50.0, np.float32), np.full((H // 2, W // 2), 100.0, np.float32), np.full((H // 2, W // 2), 100.0, np.float32)]
        base3 = [cv2.resize(x, (2 * x.shape[1], 2 * x.shape[0]), interpolation=cv2.INTER_NEAREST) for x in orig3]
        m3 = np.zeros((H, W), np.float32)
        m3[40:120, 40:160] = 1.0
        for d in range(1, 7):                                                # a ramp of six pixels round it, the node's soft band
            ring = np.zeros((H, W), bool)
            ring[40 - d:120 + d, 40 - d:160 + d] = True
            ring[40 - d + 1:120 + d - 1, 40 - d + 1:160 + d - 1] = False
            m3[ring] = 1.0 - d / 7.0
        owner3 = np.where(m3 >= 1.0, 0, -1).astype(np.int8)
        piece3 = [o_ + 150.0 * (m3 if o_.shape == m3.shape else cv2.resize(m3, (o_.shape[1], o_.shape[0]), interpolation=cv2.INTER_AREA)) for o_ in orig3]
        raw3 = np.frombuffer(tool.laid_over([b.copy() for b in base3], [piece3], owner3, (sw3, sh3, 0, 0), None, None, exact=[True], orig=orig3,
                                            mats=[m3]), np.uint8)
        y3 = raw3[:sw3 * sh3].reshape(sh3, sw3).astype(np.float32)
        want3 = 50.0 + cv2.resize(150.0 * m3, (sw3, sh3), interpolation=cv2.INTER_CUBIC)
        cover3 = cv2.resize((m3 >= 0.25).astype(np.float32), (sw3, sh3), interpolation=cv2.INTER_NEAREST) > 0
        worst3 = float(np.abs(y3 - want3)[cover3].max())
        # THE CASE THAT GOES RED when the full region and its band are laid in two passes, and a ring of the change falls back
        assert worst3 < 4.0, f"at the feather's inner edge the laid frame is {worst3:.1f} levels from the change scaled up (one row alone, 2x)"
        # under a restore that takes the edge out of the piece, nothing is carried
        held = np.zeros((H, W), np.float32)
        held[H - 60:, :] = 1.0
        yh, _u, _v = lay(SW, SH, (cw, ch, x0, y0), reach_rows=slice(0, W // 2), held=held)
        assert (yh[below] == 80).all(), "a restore's edge was carried"
        # a source cut at the sides: the columns take the laid edge where the piece reaches the left edge (its top half here)
        sw2, sh2 = 440, 300
        box2 = tool.crop_of(sw2, sh2, W, H)
        assert box2[2] > 0 and box2[3] == 0, box2
        ys, _u, _v = lay(sw2, sh2, box2, reach_cols=slice(0, H // 2))
        near, far = slice(box2[2] - tool.CARRY_FULL_PX, box2[2]), slice(0, box2[2] - tool.CARRY_FULL_PX - tool.CARRY_FADE_PX)
        assert (ys[20:int(0.4 * sh2), near] >= 195).all() and (ys[int(0.65 * sh2):sh2 - 20, :box2[2]] == 80).all(), ys[::40, :box2[2]]
        assert box2[2] <= tool.CARRY_FULL_PX + tool.CARRY_FADE_PX or (ys[20:int(0.4 * sh2), far] == 80).all(), ys[60, :box2[2]]
        # the top and the right sides, each where the piece reaches it alone
        yt, _u, _v = lay(SW, SH, (cw, ch, x0, y0), reach_top=slice(0, W // 2))
        assert (yt[:y0, 20:int(0.4 * SW)] >= 195).all() and (yt[:y0, int(0.65 * SW):SW - 20] == 80).all() and (yt[below] == 80).all(), yt[:y0, ::50]
        yr, _u, _v = lay(sw2, sh2, box2, reach_right=slice(0, H // 2))
        edge_r = box2[2] + box2[0]
        assert (yr[20:int(0.4 * sh2), edge_r:edge_r + tool.CARRY_FULL_PX] >= 195).all() and (yr[int(0.65 * sh2):sh2 - 20, edge_r:] == 80).all(), yr[60, edge_r:]
        # a deep cut is not one stretched row: in full for CARRY_FULL_PX, then back to the source past CARRY_FULL_PX + CARRY_FADE_PX
        sh3 = H * 2 + 80
        box3 = (W * 2, H * 2, 0, 40)
        yd, _u, _v = lay(W * 2, sh3, box3, reach_rows=slice(0, W // 2))
        last = 40 + H * 2 - 1
        assert (yd[last + 1:last + 1 + tool.CARRY_FULL_PX, 20:W // 2] >= 195).all(), yd[last + 1:last + 12, 40]
        assert (yd[last + 1 + tool.CARRY_FULL_PX + tool.CARRY_FADE_PX:, 20:W // 2] == 80).all(), yd[last + 14:last + 22, 40]
        middle = yd[last + tool.CARRY_FULL_PX + tool.CARRY_FADE_PX // 2, 20:W // 2]
        assert (middle > 100).all() and (middle < 180).all(), middle[:4]                 # half way through the fade, half way between
        _y, ud, _v = lay(W * 2, sh3, box3, reach_rows=slice(0, W // 2))
        clast = (40 + H * 2) // 2 - 1
        assert (ud[clast + 1:clast + 1 + tool.CARRY_FULL_PX // 2, 10:W // 4] >= 155).all() and \
            (ud[clast + 1 + (tool.CARRY_FULL_PX + tool.CARRY_FADE_PX) // 2 + 1:, 10:W // 4] == 120).all(), ud[clast + 1:clast + 12, 20]
        return f"the {SH - y0 - ch} cut rows below take the laid edge where the piece reaches it ({int(y[below, left].mean())}), the source elsewhere"

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
        # ON HER HEAD ONLY (`@head`, the lead's ruling of 2026-10-15: her pearl earring is Apparel, and so is her collar): the class's
        # pieces that touch her own Hair or Face_Neck come back, the ones off her head do not
        hair = names.index("Hair")
        cmap = np.zeros((H, W), np.uint8)
        cmap[60:80, 40:60] = hair                         # her hair
        cmap[60:80, 62:70] = apparel                      # an earring 2 px off her hair: within RESTORE_ON_HEAD_PX
        cmap[20:40, 180:196] = apparel                    # a collar or strap, off her head

        class OneMap:
            class_px_left: dict = {}

            def classes(self, label, frame, prefer=None):
                return cmap
        the_run = (None, None, {"name": "run_a", "subject": "a"})
        given = lambda wanted: tool.restore_weight({"piece": PIECE_A, "restore": [("a", wanted)]}, OneMap(), the_run, 20, (H, W))[1]  # noqa: E731
        # her own `@head` Apparel is worn on the head, so since 2026-10-23 it is placed on the piece's own face (`worn_layer`) and given
        # back in place only as filmed (`:filmed`): the @head pieces are read the same either way (`class_mask`)
        on_head, plain = given(tool.OnHeadFilmed([apparel])), given([apparel])
        # THE CASE THAT GOES RED when a piece off her head is given back by `@head`, or the one on it is not
        assert on_head[60:80, 62:70].all() and not on_head[20:40, 180:196].any(), "@head gave back a piece off her head, or not the one on it"
        assert (tool.class_mask(cmap, tool.OnHead([apparel])) == on_head).all(), "the worn layer's @head pieces are not the in-place restore's"
        assert not given(tool.OnHead([apparel])).any(), "her own @head Apparel was given back at the film's place, not left to her face"
        assert plain[20:40, 180:196].all() and plain[60:80, 62:70].all(), "the control: the plain class restore gives back both pieces"
        cmap[60:80, 40:60] = 0                            # no head on the frame: nothing comes back
        assert not given(tool.OnHeadFilmed([apparel])).any(), "@head gave back a piece on a frame with no head"
        (TMP / "at_head.txt").write_text(f"10-39 {PIECE_A} 10 restore=a.Apparel@head\n")
        read = tool.read_table(TMP / "at_head.txt")[0]["restore"]
        assert read == [("a", [apparel])] and isinstance(read[0][1], tool.OnHead) and tool.restore_text(*read[0]) == "a.Apparel@head", read
        (TMP / "at_neck.txt").write_text(f"10-39 {PIECE_A} 10 restore=a.Apparel@neck\n")
        try:
            tool.read_table(TMP / "at_neck.txt")
            raise AssertionError("`@neck` was taken")
        except SystemExit as e:
            assert "only `head` is known" in str(e), e
        return f"{area} to {grown} px given back a frame; the original inside the box at both sizes; nothing where the piece changed nothing; `@head` only on her head"

    def a_worn_thing_on_its_face() -> str:
        """A THING WORN ON THE HEAD IS PLACED ON THE PIECE'S OWN FACE (the owner, 2026-10-23), by one point's shift, translation only.
        A plate whose glasses are a white box on her forehead's picture; her Eyeglass labelled there; the render's points the film's moved
        by (DX, DY). LIVE: the box's white comes to the box moved, the piece shows at the film's place (a plate read at the new place, or
        a hole at the old one, fails). The eyes' middle where the nose root scores low. HELD: a frame with no point scored, and a frame
        whose shift jumps (the back of her head), take the nearest read frame's shift. SKIPPED: a frame with no label lays nothing and
        punches no hole. FILL (`--worn-fill`): a frame with no label borrows the nearest labelled frame's mask and pixels, moved by THIS
        frame's render point less THAT frame's film point. CLIP: moved off the piece, nothing is laid past its area grown by the margin.
        NO-OP: `:filmed`, and a class not worn on the head, are the plain restore at the film's place, byte for byte."""
        names = tool.class_names()
        glass, hand = names.index("Eyeglass"), names.index("Left_Hand")
        box = (60, 60, 76, 72)                            # x0, y0, x1, y1 at the canvas: her glasses, inside RECT_A
        DX, DY, MOVE = 16, 8, 6                           # the render's face from the film's; the film's own move on frame 15
        cw, ch, x0, y0 = tool.crop_of(SW, SH, W, H)
        s = cw / W
        plate = TMP / "source_glasses.mp4"
        run([tool.FFMPEG, "-y", "-v", "error", "-i", str(SOURCE), "-vf",
             f"drawbox=x={round(x0 + box[0] * s)}:y={round(y0 + box[1] * s)}:w={round((box[2] - box[0]) * s)}:h={round((box[3] - box[1]) * s)}:color=white:t=fill,"
             "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv", "-c:v", "libx264", "-crf", "12", "-g", "250",
             "-pix_fmt", "yuv420p", *tool.BT709_TAGS, "-c:a", "copy", str(plate)])
        piece = write("piece_g", painted(loaded(plate, 10, 30), RECT_A))
        frames = 30                                       # source frames 10-39, the piece's
        bare = {15} | set(range(35, 40))                  # no label: 15 and 35-39
        cmap = np.zeros((frames, H, W), np.uint8)
        for k in range(frames):
            if 10 + k not in bare:
                cmap[k, box[1]:box[3], box[0]:box[2]] = glass
        held_in_hand = 30                                  # her hand, given back where the film has it, holds the glasses' edge
        cmap[held_in_hand - 10, box[1]:box[3], box[2]:box[2] + 8] = hand
        outside, sliver = 33, 34                           # her glasses reach a third past the piece (RECT_A); a sliver past it
        cmap[outside - 10, box[1]:box[3], RECT_A[2]:RECT_A[2] + 8] = glass
        cmap[sliver - 10, box[1]:box[3], RECT_A[2] + 4:RECT_A[2] + 5] = glass      # clear of what `changed` bleeds past RECT_A
        here = {"left_eye": (62, 64), "right_eye": (74, 64), "center_of_nose_root": (68, 64), "left_ear": (50, 66), "right_ear": (86, 66)}
        pnames = list(here)
        ears = [pnames.index("left_ear"), pnames.index("right_ear")]
        EX, EY = 4, 12                                     # the render's EARS from the film's: where a hat goes, not the glasses
        film = np.array([[here[x] for x in pnames]] * frames, np.float32)
        film[15 - 10] += (MOVE, 0)                        # the film's head moves on 15 (the render's follows it)
        rend = film + np.array([DX, DY], np.float32)
        rend[:, ears] += np.array([EX - DX, EY - DY], np.float32)
        rend[24 - 10] += (6, 0)                           # 24: the shift steps 6 px across for a frame, under JUMP_PX: read, and smoothed
        rend[28 - 10] += (60, 0)                          # 28: the render's face 60 px off its neighbours' (the back of her head)
        fs, rs = np.full((frames, len(pnames)), 0.9, np.float32), np.full((frames, len(pnames)), 0.9, np.float32)
        rs[12 - 10, pnames.index("center_of_nose_root")] = 0.1     # 12: the nose root scored low: the eyes' middle
        rs[20 - 10] = 0.1                                           # 20: nothing scored: held
        run_g = [{"name": "run_g", "render": "piece_g.mp4", "subject": "a", "margin_px": 16}]

        def made(folder, render_points):
            cap = capture(TMP / folder, 10, frames, {"a": mask_of(RECT_A, frames, grow=4)}, run_g, {"a": cmap})
            np.savez(TMP / f"{folder}_film.npz", keypoints=film, scores=fs, names=np.array(pnames), size=np.array([W, H]), first_frame=np.array(10))
            np.savez(cap / "run_g_render_points.npz", keypoints=render_points, scores=rs, names=np.array(pnames), size=np.array([W, H]),
                     first_frame=np.array(10))
            (TMP / f"{folder}_job").mkdir()
            (TMP / f"{folder}_job" / "job.render.json").write_text(json.dumps({"job": {"capture": f"data/{folder}", "reference": {"points": str(TMP / f"{folder}_film.npz")}},
                                                                             "loads": [{"name": "run_g"}]}))
            return cap
        cap = made("cap_worn", rend)
        caps = tool.Captures([cap], (W, H))
        row = {"first": 10, "last": 39, "piece": piece, "piece_first": 10, "restore": [("a", [glass]), ("a", [hand])]}

        def at(n, fill=False, c=caps, r=None):
            record = {"rows": {}, "shared": {}}
            rr = dict(r or row, matrix=tool.probe(piece)["matrix"])
            buf = next(tool.fed_frames(plate, [(n, n, [rr])], W, H, record, captures=c, fill_worn=fill))
            return tool.planes(buf, W, H), record.get("worn", {}).get((piece, 10, 39, "a.Eyeglass"), {}).get("frames", {}).get(n)

        def plate_at(n):
            return tool.planes(next(tool.original_frames(plate, n, n, W, H)), W, H)

        def piece_at(n):
            return tool.planes(next(tool.piece_frames({"piece": piece, "piece_first": 10, "matrix": tool.probe(piece)["matrix"]}, n, n, W, H)), W, H)

        def mean_off(x, y, rect, dx=0, dy=0):
            # x inside `rect` moved by (dx, dy) against y inside `rect`
            a0, b0, a1, b1 = rect
            return float(np.abs(x[0][b0 + dy:b1 + dy, a0 + dx:a1 + dx] - y[0][b0:b1, a0:a1]).mean())
        core = (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2)
        film_only = (box[0] + 1, box[1] + 1, box[0] + DX - 1, box[3] - 1)     # the film's place the moved box does not cover
        g22 = plate_at(22)
        assert mean_off(g22, g22, core, DX, DY) > 40, "the fixture: the plate at the moved place must not be the glasses' white"
        # LIVE (22): the plate's white box comes to the box moved; the film's place shows the piece
        got, p = at(22)
        assert p["status"] == "live" and p["from_source_frame"] == 22 and p["translation"] == [DX, DY] and p["point"] == "the nose root", p
        assert p["eyes_before_px"] == [DX, DY] and p["eyes_after_px"] == [0.0, 0.0] and p["clipped_px"] == 0 and p["laid_px"] > 0, p
        moved = mean_off(got, g22, core, DX, DY)
        # THE CASE THAT GOES RED when the weight is moved and the plate read at the new place, or a hole punched at the old one
        assert moved < 2.0, f"the moved box is {moved:.2f} levels from the plate's white glasses"
        assert mean_off(got, piece_at(22), film_only) < 1.0, "at the film's place the piece is not shown: a hole of the plate is there"
        # at the source's size: the plate's own pixels, moved (DX, DY) canvas pixels through the loader's crop and scale
        full = tool.planes(next(tool.fed_frames(plate, [(22, 22, [dict(row, matrix=tool.probe(piece)["matrix"])])], W, H, captures=caps, full=(SW, SH))), SW, SH)
        src = tool.planes(next(tool.source_frames(plate, 22, 22, SW, SH)), SW, SH)
        big = [int(x0 + box[0] * s) + 3, int(y0 + box[1] * s) + 3, int(x0 + box[2] * s) - 3, int(y0 + box[3] * s) - 3]
        at_full = mean_off(full, src, big, int(round(DX * s)), int(round(DY * s)))
        assert at_full < 3.0 and mean_off(src, src, big, int(round(DX * s)), int(round(DY * s))) > 40, f"at the source's size the moved box is {at_full:.2f} levels off"
        # the eyes' middle where the nose root scores low (12)
        _got, p = at(12)
        assert p["status"] == "live" and p["point"] == "the eyes' middle" and p["translation"] == [DX, DY], p
        # SMOOTHED: 24's shift reads 6 px further across than its neighbours'; the shift laid there and beside it is the three frames' mean
        for n in (23, 24, 25):
            got, p = at(n)
            # THE CASE THAT GOES RED when the shift laid is a frame's own reading and not the mean over SMOOTH_REACH either side
            assert p["status"] == "live" and p["translation"] == [DX + 2.0, DY] and p["raw"] == ([DX + 6.0, DY] if n == 24 else [DX, DY]), p
        assert mean_off(got, plate_at(25), core, DX + 2, DY) < 2.0, "the smoothed shift is not the one laid"
        # HELD: 20 has no point scored, 28's shift jumps 60 px from its neighbours'
        for n, came in ((20, 19), (28, 27)):
            got, p = at(n)
            assert p["status"] == "held" and p["held_from"] == came and p["translation"] == [DX, DY], p
            assert mean_off(got, plate_at(n), core, DX, DY) < 2.0, f"frame {n} does not lay the held shift"
        assert p["jump_px"] is not None and p["jump_px"] > tool.JUMP_PX and "back of her head" in p["why"], p
        # SKIPPED (15, the default): no label, nothing laid, no hole
        got, p = at(15)
        assert p["status"] == "skipped" and p["laid_px"] == 0 and "filling is off" in p["why"], p
        p15 = piece_at(15)
        moved_core = (core[0] + DX, core[1] + DY, core[2] + DX, core[3] + DY)
        assert mean_off(got, p15, core) < 1.0 and mean_off(got, p15, moved_core) < 1.0, "a frame with no label laid something"
        # FILL (`--worn-fill`, 15): 14's mask and pixels, moved by the render's point on 15 less the film's on 14 = (DX + MOVE, DY)
        got, p = at(15, fill=True)
        assert p["status"] == "filled" and p["from_source_frame"] == 14 and p["translation"] == [DX + MOVE, DY] and p["film_moved_px"] == [MOVE, 0.0], p
        # THE CASE THAT GOES RED when a borrowed frame is moved by this frame's own shift (DX, DY): the white lands MOVE px short
        assert mean_off(got, plate_at(14), core, DX + MOVE, DY) < 2.0, "the filled frame is not 14's box moved by 15's render less 14's film"
        short = (box[0] + DX + 1, core[1], box[0] + DX + MOVE - 1, core[3])
        assert mean_off(got, p15, short) < 1.0, "the filled frame laid white where this frame's own shift would put it"
        # AS FILMED (30): her hand, given back where the film has it, touches the glasses, so moving them would pull them off it: they are
        # given back in place, as before, and nothing is laid at the moved place
        got, p = at(held_in_hand)
        assert p["status"] == "as filmed" and p["would_be"] == "live" and p["laid_px"] == 0 and p["under_what_is_given_back_px"] > 0 \
            and p["past_the_piece_px"] == 0 and "held there" in p["why"], p
        g30 = plate_at(held_in_hand)
        # THE CASE THAT GOES RED when a thing that reaches past what the piece shows is moved all the same
        assert mean_off(got, g30, core) < 2.0, "the glasses held in her hand were not given back where the film has them"
        in_place, _ = at(held_in_hand, r=dict(row, restore=[("a", tool.Filmed([glass])), ("a", [hand])]))
        assert tool.to_bytes(got) == tool.to_bytes(in_place), "a frame kept as filmed is not the plain restore, byte for byte"
        # a third of it outside what the piece lays: kept as filmed; a sliver outside: moved, the sliver counted
        _got, p = at(outside)
        assert p["status"] == "as filmed" and "outside what the piece lays" in p["why"], p
        _got, p = at(sliver)
        assert p["status"] == "live" and 0 < p["past_the_piece_px"] <= tool.PAST_SHARE * p["label_px_on_the_frame"], p
        # THE WHOLE ROW, built, by default and with the fill: the counts and the flags
        tabled = TMP / "worn.txt"
        tabled.write_text(f"10-39 {piece} 10 restore=a.Eyeglass restore=a.Left_Hand\n")
        got_r = {}
        for name, more in (("worn", []), ("worn_fill", ["--worn-fill"])):
            out = TMP / f"{name}.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(plate), "--table", str(tabled), "--span", "10-39",
                                   "--out", str(out), "--capture", str(cap), *more], capture_output=True, text=True)
            got_r[name] = json.loads(Path(str(out) + ".check.json").read_text())
            assert proc.returncode == 0, (name, proc.stderr[-400:], got_r[name]["failures"])
        w0, w1 = (got_r[k]["worn_on_its_face"][0] for k in ("worn", "worn_fill"))
        assert w0["frames"] == {"live": 20, "held": 2, "filled": 0, "as filmed": 2, "skipped": 6} and not w0["fill"], w0["frames"]
        assert w1["frames"] == {"live": 20, "held": 2, "filled": 3, "as filmed": 2, "skipped": 3} and w1["fill"] and got_r["worn_fill"]["worn_fill"], w1["frames"]
        assert w1["frames_by_status"]["filled"] == [[15, 15], [35, 36]] and w1["frames_by_status"]["skipped"] == [[37, 39]], w1["frames_by_status"]
        rules0, rules1 = ({f["rule"] for f in got_r[k]["flags"]} for k in ("worn", "worn_fill"))
        assert {"worn_thing_held", "worn_thing_not_laid", "worn_thing_kept_as_filmed", "worn_thing_partly_past_the_piece"} <= rules0 \
            and "worn_thing_taken_from_another_frame" not in rules0, rules0
        assert "worn_thing_taken_from_another_frame" in rules1, rules1
        # CLIP: the render's face 60 px off on every frame, so the moved box falls outside RECT_A grown by the margin
        far = made("cap_worn_far", film + np.array([DX + 60, DY], np.float32))
        # A HAT (Apparel `@head`, touching her Hair) on the same row moves by the EARS' shift, the glasses by the nose root's
        hair_cap = np.zeros((frames, H, W), np.uint8)
        hair_cap[:] = cmap
        hair_cap[:, 36:58, 56:80] = names.index("Hair")
        hair_cap[:, 44:54, 62:74] = names.index("Apparel")
        hatted = capture(TMP / "cap_worn_hat", 10, frames, {"a": mask_of(RECT_A, frames, grow=4)}, run_g, {"a": hair_cap})
        np.savez(hatted / "run_g_render_points.npz", keypoints=rend, scores=rs, names=np.array(pnames), size=np.array([W, H]), first_frame=np.array(10))
        (TMP / "cap_worn_hat_job").mkdir()
        (TMP / "cap_worn_hat_job" / "job.render.json").write_text(json.dumps({"job": {"capture": "data/cap_worn_hat", "reference": {"points": str(TMP / "cap_worn_film.npz")}},
                                                                             "loads": [{"name": "run_g"}]}))
        record = {"rows": {}, "shared": {}}
        hat_row = dict(row, restore=[("a", [glass]), ("a", tool.OnHead([names.index("Apparel")]))], matrix=tool.probe(piece)["matrix"])
        next(tool.fed_frames(plate, [(22, 22, [hat_row])], W, H, record, captures=tool.Captures([hatted], (W, H))))
        by_token = {k[3]: v["frames"][22] for k, v in record["worn"].items()}
        # THE CASE THAT GOES RED when a hat is moved by the face's points, which follow a turned face and not the head
        assert by_token["a.Apparel@head"]["point"] == "the ears' middle" and by_token["a.Apparel@head"]["translation"] == [EX, EY], by_token
        assert by_token["a.Eyeglass"]["point"] == "the nose root" and by_token["a.Eyeglass"]["translation"] == [DX, DY], by_token
        got, p = at(22, c=tool.Captures([far], (W, H)))
        assert p["status"] == "live" and p["clipped_px"] > 0, p
        beyond = (RECT_A[2] + 16 + 2, core[1], core[2] + DX + 60, core[3])
        assert mean_off(got, piece_at(22), beyond) < 1.0, "the thing was laid outside the piece's area grown by its margin"
        # NO POINTS, NO STOP (the lead's call, 2026-10-23): a row whose render's points are missing, or whose job names no film points,
        # gives the thing back where the film has it, byte for byte the plain restore, and the build passes with the missing file flagged
        plain_row = dict(row, restore=[("a", tool.Filmed([glass])), ("a", [hand])])
        plain, _ = at(22, r=plain_row)
        job = TMP / "cap_worn_job" / "job.render.json"
        kept = job.read_text()
        for name, undo in (("render", None), ("film", kept)):
            if name == "render":
                (cap / "run_g_render_points.npz").rename(cap / "aside.npz")
            else:
                job.write_text(json.dumps({**json.loads(kept), "job": {**json.loads(kept)["job"], "reference": {}}}))
            out2 = TMP / f"worn_no_{name}_points.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(plate), "--table", str(tabled), "--span", "10-39",
                                   "--out", str(out2), "--capture", str(cap)], capture_output=True, text=True)
            r2 = json.loads(Path(str(out2) + ".check.json").read_text())
            said = "run_g_render_points.npz are not there" if name == "render" else "names no film face points"
            flag = [f for f in r2["flags"] if f["rule"] == "worn_thing_not_placed_no_points"]
            # THE CASE THAT GOES RED when a row with no points stops the build, or is laid otherwise than as filmed
            assert proc.returncode == 0 and "Traceback" not in proc.stderr and out2.exists() and flag and said in flag[0]["why"] \
                and said in proc.stdout and r2["worn_on_its_face"][0]["frames"]["as filmed"] == 30, (name, proc.stderr[-300:], r2["failures"])
            got, p = at(22)
            assert p["status"] == "as filmed" and tool.to_bytes(got) == tool.to_bytes(plain), f"with no {name} points the row is not the plain restore"
            if name == "render":
                (cap / "aside.npz").rename(cap / "run_g_render_points.npz")
            else:
                job.write_text(undo)
        caps = tool.Captures([cap], (W, H))                 # read the points again, as they are
        # NO-OP: `:filmed`, and a class not worn on the head, are the plain restore at the film's place, byte for byte
        (TMP / "filmed.txt").write_text(f"10-39 {piece} 10 restore=a.Eyeglass:filmed\n")
        filmed = tool.read_table(TMP / "filmed.txt")[0]["restore"]
        assert isinstance(filmed[0][1], tool.Filmed) and tool.restore_text(*filmed[0]) == "a.Eyeglass:filmed", filmed
        hands = capture(TMP / "cap_worn_hand", 10, frames, {"a": mask_of(RECT_A, frames, grow=4)}, run_g, {"a": np.where(cmap == glass, hand, 0).astype(np.uint8)})
        for n in (22, 15):
            as_filmed, rec_f = at(n, r=dict(row, restore=filmed))  # its glasses `:filmed`, no hand
            by_hand, _ = at(n, c=tool.Captures([hands], (W, H)), r=dict(row, restore=[("a", [hand])]))
            # THE CASE THAT GOES RED when `:filmed`, or a class not on the head, is moved
            assert tool.to_bytes(as_filmed) == tool.to_bytes(by_hand) and rec_f is None, f"frame {n}: `:filmed` is not the plain restore"
        got, _ = at(22, r=dict(row, restore=filmed))
        assert mean_off(got, g22, core) < 2.0, "`:filmed` did not give the glasses back at the film's place"
        assert tool.worn_of(dict(row, restore=[("b", [glass])]), caps.run_of(piece)) == [], "another subject's glasses were placed on the row's own face"
        (TMP / "filmed_bad.txt").write_text(f"10-39 {piece} 10 restore=a.Eyeglass:moved\n")
        try:
            tool.read_table(TMP / "filmed_bad.txt")
            raise AssertionError("`:moved` was taken")
        except SystemExit as e:
            assert "only `filmed` is known" in str(e), e
        return (f"the plate's white glasses moved {DX},{DY} ({moved:.2f} levels at the canvas, {at_full:.2f} at the source's size), the piece at the "
                f"film's place; held on a frame with no point and on a 60 px jump; 15 skipped, and filled from 14 by {DX + MOVE},{DY}; "
                f"default {w0['frames']}, with the fill {w1['frames']}; clipped off the piece; smoothed where 24 steps; with no render or film points the plain restore, flagged, the build passing; "
                "`:filmed` and a hand restore the same bytes")

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
        # the same capture written with other labels for the same two people, read through --alias: the same picture,
        # the owner map and the capture's own flag under the table's names
        other_words = capture(TMP / "cap_subject_other_words", 10, 30, {"lead": masks["a"], "second": masks["b"]},
                              [{"name": "run_a", "render": "piece_a.mp4", "subject": "lead", "margin_px": 16}],
                              flags=[{"id": "f001", "rule": tool.PEOPLE_FLAG, "subject": "second", "source_frames": [[10, 12]], "figures": {"people": 1}}])
        np.savez_compressed(other_words / "owners.npz", owner=np.where(masks["b"], 1, np.where(masks["a"], 0, 255)).astype(np.uint8),
                            labels=np.array(["lead", "second"]), nobody=255, contested=254)
        renamed = tool.Captures([other_words], (W, H), {"lead": "a", "second": "b"})
        got_alias, _ = frame_planes(SOURCE, [dict(row)], 20, renamed)
        assert box_mean(got_alias, o, only_other) < 1.0 and box_mean(got_alias, a, only_own) < 1.0, "through --alias the other subject was not given back"
        assert renamed.run_of(PIECE_A)[2]["subject"] == "a" and renamed.mask("b", 20).any() and renamed.owned("b", 20).any() \
            and renamed.said_absent_with_people()[0]["subject"] == "b" and renamed.mask("second", 20) is None, "a label was not read as the table's name"
        try:
            tool.Captures([other_words], (W, H), {"lead": "a", "second": "a"})
        except SystemExit as exc:
            assert "same subject" in str(exc), exc
        else:
            raise AssertionError("two labels of one capture read as one subject were accepted")
        return f"{given} px given back on the frame read ({area} in the other's mask alone); the overlap stays the piece's, and goes back with `:whole`"

    def a_clip_with_a_cut(cut_at: int, frames: int) -> Path:
        clip = TMP / "with_a_cut.mp4"
        if clip.is_file():
            return clip
        other = PICTURE.replace("c0=0x905030", "c0=0x203c8c").replace("c2=0xc8b450", "c2=0x50a0c8").replace("x0=40:y0=30", "x0=300:y0=40")
        run([tool.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:{PICTURE}",
             "-f", "lavfi", "-i", f"gradients=size={SW}x{SH}:rate={RATE}:speed=0.03:{other},eq=brightness=-0.35", "-filter_complex",
             f"[0:v]trim=end_frame={cut_at},setpts=PTS-STARTPTS[a];[1:v]trim=end_frame={frames - cut_at},setpts=PTS-STARTPTS[b];"
             "[a][b]concat=n=2:v=1,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[v]",
             "-map", "[v]", "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", *tool.BT709_TAGS, str(clip)])
        return clip

    def spill_over_a_cut() -> str:
        cut_at, frames = 20, 40
        clip = a_clip_with_a_cut(cut_at, frames)
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
        # two captures cover the frame: a long load that calls both subjects absent there, listed FIRST, and the one
        # that sees them. A row with no run of its own (`subject=`) must read the one that sees them, or its restore
        # gives nothing back
        absent = capture(TMP / "cap_owners_absent", 10, 30, {"a": np.zeros((30, H, W), bool), "b": np.zeros((30, H, W), bool)},
                         [{"name": "run_long", "render": "some_long_render.mp4", "subject": "a", "margin_px": 16}])
        np.savez_compressed(absent / "owners.npz", owner=np.full((30, H, W), 255, np.uint8), labels=np.array(["a", "b"]), nobody=255, contested=254)
        both = tool.Captures([absent, folder], (W, H))
        got, _ = frame_planes(SOURCE, [{"piece": write("piece_a_again", loaded(Path(PIECE_A), 0, 30)), "piece_first": 10, "subject": "a", "restore": [("b", None)]}], 20, both)
        assert box_mean(got, o, only_other) < 1.0 and box_mean(got, a, only_own) < 1.5, \
            "a row with no run read the first capture covering the frame, which calls both subjects absent, and gave nothing back"
        assert both.mask("a", 20).any() and both.owned("b", 20).any() and not both.mask("a", 20, prefer=absent).any(), "the fallback, or a run's own capture's word"
        # a class restore of ANOTHER subject's class keeps off what the map gives the row's own subject: b's class map
        # calls the whole rectangle its apparel, a's side included, as a class map cut to a track and its margin does
        # beside another person, and the row is a's
        apparel = tool.class_names().index("Apparel")
        reaching = np.zeros((30, H, W), np.uint8)
        reaching[:, RECT_A[1]:RECT_A[3], RECT_A[0]:RECT_A[2]] = apparel
        runs = [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}]
        with_map = capture(TMP / "cap_class_owned", 10, 30, {"a": mask_of(own, 30), "b": mask_of(other, 30)}, runs, {"a": reaching, "b": reaching})
        np.savez_compressed(with_map / "owners.npz", owner=owner, labels=np.array(["a", "b"]), nobody=255, contested=254)
        no_map = capture(TMP / "cap_class_unowned", 10, 30, {"a": mask_of(own, 30), "b": mask_of(other, 30)}, runs, {"b": reaching})
        row = [{"piece": PIECE_A, "piece_first": 10, "restore": [("b", [apparel])]}]
        kept_to = tool.Captures([with_map], (W, H))
        got, rec = frame_planes(SOURCE, [dict(row[0])], 20, kept_to)
        assert box_mean(got, a, only_own) < 1.0, "another subject's class restore gave back pixels the map gives the row's own subject"
        edge = (62, 46, 64, 74)          # a's last two columns beside b's half of the strip: the restore's grown rim must not reach them
        assert box_mean(got, a, edge) < box_mean(got, o, edge), "the restore's rim was grown onto pixels the map gives the row's own subject"
        whole_class = (RECT_A[3] - RECT_A[1]) * (RECT_A[2] - RECT_A[0])
        counted = rec["rows"][PIECE_A][20]["class_px"]
        assert counted == whole_class - (RECT_A[3] - RECT_A[1]) * 24, f"the record counts {counted} class pixels given back, of a class map of {whole_class} with 1920 on another's"
        assert box_mean(got, o, only_other) < 1.0 and box_mean(got, o, disputed) < 1.0, "the class was not given back on its own subject's pixels, or on contested ones"
        left = kept_to.class_px_left[(PIECE_A, "b", 20)]
        assert left == (RECT_A[3] - RECT_A[1]) * 24, f"{left} class pixels were left as another subject's; a's side of the rectangle is 24 wide"
        got, _ = frame_planes(SOURCE, [dict(row[0])], 20, tool.Captures([no_map], (W, H)))
        assert box_mean(got, o, only_own) < 1.0, "with no owner map the class is given back wherever the class map has it, as before"
        # the row's OWN subject's class, reaching onto the other subject: given back there too, since on a third
        # party's pixels the source is the right picture (a rule that kept it off them left the piece's change there)
        own_class = tool.Captures([with_map], (W, H))
        got, _ = frame_planes(SOURCE, [{"piece": PIECE_A, "piece_first": 10, "restore": [("a", [apparel])]}], 20, own_class)
        assert box_mean(got, o, only_other) < 1.0 and box_mean(got, o, only_own) < 1.0, "the row's own subject's class was kept off pixels another subject owns"
        assert not own_class.class_px_left, own_class.class_px_left
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
        # depth-settled `owners.npz` with `restore=b:whole`:
        # 1) `a` in front of `b` (z_a=1.20m, z_b=1.95m): `a` owns the overlap `b_half` (0), and `a`'s carried part
        #    reaches `disputed` (even if `b` owned `disputed`): `restore=b:whole` gives back `only_other` and never
        #    punches into `a`'s owned `b_half` or `a`'s carried `disputed`.
        owner_a_front = owner.copy()
        owner_a_front[:, 40:80, 64:80] = 0              # `a` owns top half of overlap by depth
        owner_a_front[:, 80:120, 64:80] = 1             # even if `b` owned bottom half, `a`'s carried mask protects it when `a` is in front
        carried_a = np.zeros((30, H, W), bool)
        carried_a[:, 40:120, 40:80] = True
        folder_af = capture(TMP / "cap_owners_af", 10, 30, {"a": mask_of(own, 30), "b": mask_of(other, 30)},
                            [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}],
                            carried={"run_a": carried_a})
        z_af = np.tile(np.array([[1.20], [1.95]], np.float32), (1, 30))
        np.savez_compressed(folder_af / "owners.npz", owner=owner_a_front, labels=np.array(["a", "b"]),
                            nobody=255, contested=254, depth_settled=np.bool_(True), z_by_frame=z_af)
        caps_af = tool.Captures([folder_af], (W, H))
        got_af, _ = frame_planes(SOURCE, [{"piece": PIECE_A, "piece_first": 10, "restore": [("b", "whole")]}], 20, caps_af)
        assert box_mean(got_af, o, only_other) < 1.0, "depth-settled restore=b:whole did not give back b's uncontested pixels"
        assert box_mean(got_af, a, b_half) < 1.0 and box_mean(got_af, a, disputed) < 1.0, (
            "depth-settled restore=b:whole punched a hole in front subject a's owned or carried pixels"
        )
        _, hard_af = tool.restore_weight({"piece": PIECE_A, "restore": [("b", "whole")]}, caps_af, caps_af.run_of(PIECE_A), 20, (H, W))
        assert not hard_af[80:120, 64:80].any() and hard_af[40:120, 80:104].all(), (
            "depth-settled restore=b:whole hard mask did not exclude front subject a's carried mask"
        )
        # 2) `b` in front of `a` (z_a=1.95m, z_b=1.20m): `b` owns `b_half` (1), so `restore=b:whole` gives back `b_half`
        #    and `only_other` while leaving `only_own` as `a`.
        folder_bf = capture(TMP / "cap_owners_bf", 10, 30, {"a": mask_of(own, 30), "b": mask_of(other, 30)},
                            [{"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16}],
                            carried={"run_a": carried_a})
        z_bf = np.tile(np.array([[1.95], [1.20]], np.float32), (1, 30))
        np.savez_compressed(folder_bf / "owners.npz", owner=owner, labels=np.array(["a", "b"]),
                            nobody=255, contested=254, depth_settled=np.bool_(True), z_by_frame=z_bf)
        caps_bf = tool.Captures([folder_bf], (W, H))
        got_bf, _ = frame_planes(SOURCE, [{"piece": PIECE_A, "piece_first": 10, "restore": [("b", "whole")]}], 20, caps_bf)
        assert box_mean(got_bf, o, only_other) < 1.0 and box_mean(got_bf, o, b_half) < 1.0, (
            "depth-settled restore=b:whole with b in front did not give back b's owned overlap"
        )
        assert box_mean(got_bf, a, only_own) < 1.0, "depth-settled restore=b:whole gave back a's own pixels"
        return f"a restore gives back what the map says the other owns and not what it calls contested (and gates `:whole` by depth when `depth_settled`); of the pixels two pieces changed, {record['shared'][20]['by_mask']} went by the map and {record['shared'][20]['by_order']} by order; another subject's class restore leaves the {left} px the map gives the row's own subject, and the row's own class is given back wherever its map has it"

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

    def two_people_and_a_keep_out() -> str:
        """`claimed_by_more_than_one_track` is of two PEOPLE; a person's track under a keep-out of theirs is by design."""
        a, b = mask_of((40, 40, 100, 120), 4), mask_of((90, 40, 150, 120), 4)        # two people, 10 columns of 80 rows in common
        clothes = mask_of((40, 80, 100, 120), 4)                                         # a keep-out on a: the lower half of a
        cmap = np.zeros((4, H, W), np.uint8)
        cmap[a] = 4
        folder = capture(TMP / "cap_people", 10, 4, {"a": a, "b": b, "clothes": clothes}, [], {"a": cmap, "b": cmap})
        caps = tool.Captures([folder], (W, H))
        assert caps.people() == ["a", "b"], caps.people()
        tally = tool.Subjects(caps)
        flat = np.zeros((H, W), np.float32)
        for f in range(10, 14):
            tally.add(f, flat, flat, np.zeros((H, W), bool))
        got = tally.report()
        two, beside = got["claimed_by_more_than_one_track"], got["a_person_and_a_keep_out_by_design"]
        assert two["px"] == 4 * 10 * 80 and two["frames"] == 4 and two["counted_between"] == ["a", "b"], two
        # clothes on a: 60 by 40, of which the 10 columns b also holds are already two people's
        assert beside["px"] == 4 * 50 * 40 and beside["keep_outs"] == ["clothes"] and beside["frames"] == 4, beside
        assert set(got["subjects"]) == {"a", "b", "clothes"}, "a keep-out lost its own row of the table"
        # with no class map anywhere nothing says who is a person, and every label is counted as one, as before
        bare = capture(TMP / "cap_no_maps", 10, 4, {"a": a, "clothes": clothes}, [])
        plain = tool.Subjects(tool.Captures([bare], (W, H)))
        plain.add(10, flat, flat, np.zeros((H, W), bool))
        assert plain.report()["claimed_by_more_than_one_track"]["px"] == 60 * 40 and plain.report()["a_person_and_a_keep_out_by_design"]["px"] == 0
        return f"two people share {two['px']} px; a person under their own keep-out, {beside['px']} px, is counted beside it and not in it"

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

    def shots_left_as_the_source() -> str:
        cut_at, frames = 20, 40
        clip = a_clip_with_a_cut(cut_at, frames)
        whole = loaded(clip, 0, frames)

        def painted_on(name, rect, first, last, gap=()):
            out = whole.clone()
            x0, y0, x1, y1 = rect
            for f in range(first, last + 1):
                if f not in gap:
                    out[f, y0:y1, x0:x1] = (out[f, y0:y1, x0:x1] + 0.5) % 1.0
            return write(name, out)
        # its subject on the first shot only, from that shot's first frame to its last (a piece that began or ended
        # as the original there would be the ends proof's business), with four frames left alone in the middle
        piece_a = painted_on("shots_a", RECT_A, 0, cut_at - 1, gap=(8, 9, 10, 11))
        piece_b = painted_on("shots_b", RECT_B, 0, frames - 1)          # another subject, on both shots
        on_both, first_only = mask_of(RECT_A, frames, grow=4), mask_of(RECT_A, frames, grow=4)
        first_only[cut_at:] = False
        other = mask_of(RECT_B, frames, grow=4)
        run_a = {"name": "run_a", "render": Path(piece_a).name, "subject": "a", "margin_px": 16}
        run_b = {"name": "run_b", "render": Path(piece_b).name, "subject": "b", "margin_px": 16}
        flag = {"id": "f001", "rule": tool.PEOPLE_FLAG, "subject": "a", "source_frames": [[cut_at, frames - 1]],
                "figures": {"people": 2, "similarity": 0.6, "line": 0.8}}

        def go(name, folder, *lines, pieces=((piece_a, ""),)):
            path = TMP / f"{name}.txt"
            path.write_text("".join(f"0-{frames - 1} {p} 0{more}\n" for p, more in pieces) + "".join(line + "\n" for line in lines))
            out = TMP / f"{name}.mp4"
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(clip), "--table", str(path), "--span", f"0-{frames - 1}",
                                   "--out", str(out), *(("--capture", str(folder)) if folder else ())], capture_output=True, text=True)
            report = Path(str(out) + ".check.json")
            assert report.is_file(), proc.stderr[-300:] or proc.stdout[-300:]
            r = json.loads(report.read_text())
            return proc.returncode, r, {(tuple(e["frames"]), e["subject"]): e for e in r["shots"]["where_a_named_subject_is_the_source's_own"]}
        second = ((cut_at, frames - 1), "a")
        # tracked on the second shot, nothing laid there
        tracked = capture(TMP / "shots_tracked", 0, frames, {"a": on_both}, [run_a])
        code, r, own = go("shots_tracked", tracked)
        assert r["shots"]["cuts"] == [cut_at], r["shots"]["cuts"]
        said = [f for f in r["failures"] if "nothing laid for a" in f]
        assert code == 1 and said and f"{cut_at}-{frames - 1}" in said[0] and own[second]["reading"] == "tracked, nothing laid", (code, r["failures"], own)
        assert f"{cut_at}-{frames - 1}" in r["verdict_line"] and r["verdict_line"].startswith("FAILS"), r["verdict_line"]
        first = r["shots"]["shots"][0]["subjects"]["a"]
        assert first["reading"] == "laid" and first["tracked_with_nothing_laid"]["runs"] == [[8, 11]] and first["tracked_with_nothing_laid"]["longest"] == 4, first
        # the same, answered in the table
        code, r, own = go("shots_intent", tracked, f"source {cut_at}-{frames - 1} subject=a not in the brief for this shot")
        assert code == 0 and own[second]["reading"] == "by intent" and own[second]["would_have_read"] == "tracked, nothing laid" \
            and own[second]["why"] == "not in the brief for this shot" and "1 by intent" in r["verdict_line"], (code, r["failures"], own, r["verdict_line"])
        # an answer that contradicts the file, and one that fits no shot
        code, r, _ = go("shots_contradicted", tracked, f"source {cut_at}-{frames - 1} subject=a ok", f"source 0-{cut_at - 1} subject=a not so")
        assert code == 1 and any("have something of theirs laid" in f for f in r["failures"]), r["failures"]
        code, r, _ = go("shots_no_shot", tracked, f"source {cut_at + 3}-{cut_at + 9} subject=a part of a shot")
        assert code == 1 and any("covers no whole shot" in f for f in r["failures"]), r["failures"]
        # called absent on the second shot: the capture's own flag says people were detected
        code, r, own = go("shots_flagged", capture(TMP / "shots_flagged", 0, frames, {"a": first_only}, [run_a], flags=[flag]))
        assert code == 1 and own[second]["reading"] == "not tracked, people detected" and own[second]["the_capture_said"]["flag"] == "f001", (code, own)
        # called absent, and a shot table counts one person there
        table = [(0, cut_at - 1, 1, "picked"), (cut_at, frames - 1, 1, "absent")]
        code, r, own = go("shots_table", capture(TMP / "shots_table", 0, frames, {"a": first_only}, [run_a], shots={"a": table}))
        assert code == 1 and own[second]["reading"] == "not tracked, people detected" and own[second]["people_detected"] == 1, (code, own)
        # the one person counted there is another subject, who is laid there
        both = capture(TMP / "shots_other", 0, frames, {"a": first_only, "b": other}, [run_a, run_b], shots={"a": table})
        code, r, own = go("shots_other", both, pieces=((piece_a, ""), (piece_b, "")))
        assert code == 0 and own[second]["reading"] == "not tracked, the people detected are other subjects'" \
            and own[second]["other_subjects_there"] == ["b"] and ((cut_at, frames - 1), "b") not in own, (code, r["failures"], own)
        # nobody detected; and nothing known
        nobody = [(0, cut_at - 1, 1, "picked"), (cut_at, frames - 1, 0, "absent")]
        code, r, own = go("shots_nobody", capture(TMP / "shots_nobody", 0, frames, {"a": first_only}, [run_a], shots={"a": nobody}))
        assert code == 0 and own[second]["reading"] == "not tracked, nobody detected", (code, r["failures"], own)
        code, r, own = go("shots_unknown", capture(TMP / "shots_unknown", 0, frames, {"a": first_only}, [run_a]))
        assert code == 0 and own[second]["reading"] == "not known" and "1 not known" in r["verdict_line"], (code, r["failures"], own, r["verdict_line"])
        # with no capture no subject is named, and the record says that nothing was asked
        code, r, own = go("shots_bare", None)
        assert code == 0 and not own and "no row's subject is known" in r["verdict_line"] and r["shots"]["cuts"] == [cut_at], (code, own, r["verdict_line"])
        # a blur: three consecutive frames each over the line are one cut at the first, not three shots of a frame
        moved = {n: 2.0 for n in range(1, 40)}
        moved.update({20: 60.0, 21: 55.0, 22: 48.0, 31: 70.0})
        section, problems = tool.shots_of({"rows": {}}, [], None, (0, 39), moved, [])
        assert [e["frames"] for e in section["shots"]] == [[0, 19], [20, 30], [31, 39]] and not problems \
            and section["runs_of_frames_over_the_line_read_as_one_cut"] == [[20, 22]], (section["cuts"], section["runs_of_frames_over_the_line_read_as_one_cut"])
        # where a capture's shot table covers the frames its cuts are the cuts: a cut this tool's line misses is
        # taken from it, and a move this tool's line calls a cut inside one of its shots is not one

        class Tables:
            def shot_tables(self):
                return [{"subject": "a", "capture": "made", "cuts": [10, 25],
                         "shots": [{"frames": [0, 9], "people": 1, "state": "picked"}, {"frames": [10, 24], "people": 1, "state": "taken"},
                                   {"frames": [25, 34], "people": 1, "state": "taken"}]},
                        # a load made for one shot: its table holds no cut, and its first frame is a cut of the source
                        {"subject": "a", "capture": "one shot", "cuts": [], "shots": [{"frames": [35, 39], "people": 1, "state": "picked"}]}]

            def said_absent_with_people(self):
                return []

            def mask(self, label, n, prefer=None):
                return None
        section, problems = tool.shots_of({"rows": {}}, [], Tables(), (0, 39), {**moved, 35: 66.0, 37: 50.0}, [])
        # the cut at 35 is this tool's own: it is the second table's FIRST frame, which that table has no word on;
        # the move at 37, inside it, is not a cut because the table has none there
        assert [e["frames"] for e in section["shots"]] == [[0, 9], [10, 24], [25, 34], [35, 39]] \
            and section["frames_a_shot_table_has_the_word_on"] == [[1, 34], [36, 39]], (section["cuts"], section["frames_a_shot_table_has_the_word_on"])
        apart = section["cuts_a_capture_puts_elsewhere"][0]
        assert apart["its_cuts_not_found_here"] == [10, 25] and apart["cuts_here_it_does_not_have"] == [20, 31], apart
        assert section["the_record_against_this_reading"] is None and "this tool's own" in section["cuts_are"], "with no shots record the section says one was read"
        # THE SHOTS RECORD'S CUTS (--shots), where no capture's shot table has the word. With no capture: the record's are the
        # cuts, this tool's own reading is set beside them both ways, and nothing fails. The record has a cut at 25 the
        # picture does not show (two grey pictures, or a record that split one shot) and none at 31, where the picture moves
        section, problems = tool.shots_of({"rows": {}}, [], None, (0, 39), moved, [], record_cuts=[0, 20, 25, 80])
        assert [e["frames"] for e in section["shots"]] == [[0, 19], [20, 24], [25, 39]] and not problems, (section["cuts"], problems)
        told = section["the_record_against_this_reading"]
        assert told["cuts_the_record_has_and_this_reading_does_not_find"] == [25] and told["cuts_this_reading_finds_and_the_record_does_not_have"] == [31] \
            and told["cuts_the_record_has_where_a_shot_table_has_the_word_and_has_none"] == [] and "the shots record's" in section["cuts_are"], told
        # with a capture's shot table: the table still has the word on the frames it covers (a record cut at 30 and one at
        # 37, inside its shots, are not cuts and are listed), and the record has it on the one frame no table covers, 35
        section, problems = tool.shots_of({"rows": {}}, [], Tables(), (0, 39), moved, [], record_cuts=[0, 10, 30, 35, 37])
        assert [e["frames"] for e in section["shots"]] == [[0, 9], [10, 24], [25, 34], [35, 39]] and not problems, (section["cuts"], problems)
        told = section["the_record_against_this_reading"]
        assert told["cuts_the_record_has_where_a_shot_table_has_the_word_and_has_none"] == [30, 37] \
            and told["cuts_the_record_has_and_this_reading_does_not_find"] == [35] and told["cuts_this_reading_finds_and_the_record_does_not_have"] == [], told
        # a record with no cut at 35: the two shots either side are then one, since neither a table nor the record cuts there
        section, _ = tool.shots_of({"rows": {}}, [], Tables(), (0, 39), {**moved, 35: 66.0}, [], record_cuts=[0, 10])
        assert [e["frames"] for e in section["shots"]][-1] == [25, 39] and section["the_record_against_this_reading"]["cuts_this_reading_finds_and_the_record_does_not_have"] == [35]
        (TMP / "not_a_record.json").write_text(json.dumps({"schema": "h3_plan/1", "shots": []}))
        (TMP / "a_record.json").write_text(json.dumps({"schema": "h3_shots/1", "shots": [{"first": 0, "last": 9}, {"first": 10, "last": 39}]}))
        assert tool.record_cuts_of(TMP / "a_record.json") == [0, 10]
        # A TABLE OF ONE SHOT says a subject with no row there is the original (the first per-shot review clip,
        # 2026-10-14, failed on it): kept when it lies inside the span, on the shots' edges, and the shots record names
        # that subject on every shot it covers. The first form kept it on the name alone, and the first reader's probe
        # passed a line off the span and one over part of a shot; each of those fails now, as does a name the record
        # has only on another shot, a line with no record, and one beside a row whose subject is not known
        (TMP / "a_record_who.json").write_text(json.dumps({"schema": "h3_shots/1", "shots": [
            {"first": 0, "last": 19, "who": {"a": {"present": "yes"}}}, {"first": 20, "last": 39, "who": {"a": {}, "c": {"present": "no"}}}]}))
        who = tool.record_subjects_of(TMP / "a_record_who.json")
        assert who == [(0, 19, {"a": "yes"}), (20, 39, {"a": None, "c": "no"})], who
        one = lambda first, last, label="c": [{"first": first, "last": last, "subject": label, "why": "not in the shot by the shots record"}]
        section, problems = tool.shots_of({"rows": {}}, [], None, (0, 39), moved, one(20, 39), record_cuts=[0, 20], record_subjects=who)
        kept = section["said_of_subjects_no_row_has"]
        assert not problems and len(kept) == 1 and kept[0]["frames"] == [20, 39] and kept[0]["subject"] == "c" and kept[0]["the_shots_record_lists_them"] \
            and "the shots' edges" in kept[0]["checked_against"] and kept[0]["the_shots_record_says_present"] == ["no"] \
            and any("the table's word alone" in m for m in section["missing"]), (problems, kept)
        for line, subjects, said in ((one(20, 39), [(0, 19, {"a": "yes", "c": "yes"}), (20, 39, {"a": "yes"})], "does not list them on 20-39"),
                                     (one(5000, 6000), who, "outside the build's span 0-39"),
                                     (one(23, 30), who, "does not begin and end on the shots' edges"),
                                     (one(0, 39), who, "does not list them on 0-19"),
                                     (one(20, 39, "cc"), who, "does not list them on 20-39"),
                                     (one(20, 39), None, "no shots record was given")):
            section, problems = tool.shots_of({"rows": {}}, [], None, (0, 39), moved, line, record_cuts=[0, 20], record_subjects=subjects)
            assert not section["said_of_subjects_no_row_has"] and len(problems) == 1 and f"names {line[0]['subject']}, a subject no row of it has" in problems[0] \
                and said in problems[0], (line, problems)
        assert "whether they have a row cannot be said" in tool.no_row_fault(one(20, 39)[0], [(0, 19), (20, 39)], (0, 39), who, ["piece_x.mp4"])
        # the verdict line says what the record says: not in the shot where it says no, the original where it says yes or unsure
        words = tool.no_row_words(kept + [{**kept[0], "subject": "d", "the_shots_record_says_present": ["unsure", "yes"]}])
        assert "c is not in the shot on 20-39 (the shots record: present no)" in words and "d is the original on 20-39 (the shots record: present unsure/yes)" in words \
            and tool.no_row_words([]) == "", words
        # and through shots_of itself: a row whose subject is not known (no capture to say whose it is) refuses the line
        section, problems = tool.shots_of({"rows": {}}, [{"piece": "/x/piece_x.mp4"}], None, (0, 39), moved, one(20, 39), record_cuts=[0, 20], record_subjects=who)
        assert not section["said_of_subjects_no_row_has"] and len(problems) == 1 and "piece_x.mp4" in problems[0] and "cannot be said" in problems[0], problems
        try:
            tool.record_cuts_of(TMP / "not_a_record.json")
            raise AssertionError("a file that is not a shots record was read as one")
        except SystemExit as stop:
            assert "not a video's shots record" in str(stop), stop
        return ("a shot with its subject tracked and nothing laid fails by its frames; so does one called absent with people detected, by the "
                "capture's flag or a shot table; a `source` line answers it and its words are kept; another subject's person, nobody, and "
                "nothing known are listed and pass; a `source` line over a laid shot or over part of one fails; one for a subject no row has "
                "is kept inside the span, on the shots' edges, where the record names them, and fails off the span, over part of a shot, where the record "
                "does not name them, with no record, and beside a row whose subject is not known")

    def no_mask_no_lay() -> str:
        on = mask_of(RECT_A, 30, grow=4)
        held = on.copy()
        held[12:17] = False                                             # source frames 22-26: the run carried nothing
        o_in, a_in = original(24), planes_of(PIECE_A, 24)
        o_out, a_out = original(30), planes_of(PIECE_A, 30)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3] - y[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3]).mean())
        assert box_mean(a_in, o_in, RECT_A) > 20, "the piece does not differ from the original where it is painted"
        results = {}
        # the node's own carried mask is known by the window files the run's entry lists, whatever words describe it
        for name, said in (("saved", {"carried_is": "the mask each window was given", "files": ["piece_a_window_1_region.npz"]}),
                           ("review", {"carried_is": "read from the review"}), ("none", None)):
            run = {"name": "run_a", "render": "piece_a.mp4", "subject": "a", "margin_px": 16, **(said or {})}
            folder = capture(TMP / f"cap_carried_{name}", 10, 30, {"a": on}, [run], carried={"run_a": held} if said else None)
            code, r, out = deliver(TMP, f"carried_{name}", [(10, 39, PIECE_A, 10)], "10-39", "--capture", str(folder))
            flag = [f for f in r["flags"] if f["rule"] == "piece_changed_where_its_run_carried_no_mask"]
            got_in, got_out = planes_of(str(out), 24), planes_of(str(out), 30)
            results[name] = (code, r, flag, box_mean(got_in, o_in, RECT_A), box_mean(got_in, a_in, RECT_A), box_mean(got_out, a_out, RECT_A))
        code, r, flag, to_orig, to_piece, outside = results["saved"]
        assert code == 0 and to_orig < 1.5 and to_piece > 20 and outside < 1.5, \
            f"with the node's own carried mask empty on a frame the row still shows there ({to_orig:.1f} from the original, {to_piece:.1f} from the piece)"
        assert flag and flag[0]["source_frames"] == [[22, 26]] and flag[0]["figures"]["left_as_the_source"] is True, flag
        first = r["shots"]["shots"][0]["subjects"]["a"]
        assert first["frames_laid"] == 25 and first["tracked_with_nothing_laid"]["runs"] == [[22, 26]], first
        assert not [f for f in r["flags"] if f["rule"] == "piece_changes_nothing"], "the frames left as the source's were also called a piece changing nothing"
        # the same at the source's size, where a row is laid over the source and not the other way round
        code, r, out = deliver(TMP, "carried_saved_full", [(10, 39, PIECE_A, 10)], "10-39", "--capture", str(TMP / "cap_carried_saved"), "--size", "source")
        luma = list(tool.luma_at_canvas(str(out), W, H))
        x0, y0, x1, y1 = RECT_A
        inside = lambda got, want: float(np.abs(got - want[0])[y0 + 3:y1 - 3, x0 + 3:x1 - 3].mean())   # noqa: E731
        assert code == 0 and inside(luma[24 - 10], o_in) < 3.0 and inside(luma[30 - 10], a_out) < 3.0, \
            f"at the source's size: {inside(luma[24 - 10], o_in):.1f} from the original on a frame with no mask, {inside(luma[30 - 10], a_out):.1f} from the piece on one with"
        code, r, flag, to_orig, to_piece, outside = results["review"]
        assert code == 0 and to_piece < 1.5 and flag and flag[0]["figures"]["left_as_the_source"] is False and flag[0]["source_frames"] == [[22, 26]], \
            f"a carried mask read off a review dropped frames, or did not say so ({to_piece:.1f} from the piece; {flag})"
        code, r, flag, to_orig, to_piece, outside = results["none"]
        assert code == 0 and to_piece < 1.5 and not flag, "with no carried mask in the capture a frame was dropped or flagged"
        return "five frames the node's own carried mask is empty on are the original and are named; a review's reading drops nothing and says so; no carried mask, nothing changes"

    def a_run_named_on_the_row() -> str:
        # a capture made as a plan, before its render: its run names no render, as every load of the lane's does
        on = mask_of(RECT_A, 30, grow=4)
        held = on.copy()
        held[12:17] = False                                             # source frames 22-26: the run carried nothing
        planned = {"name": "run_a", "planned": True, "render": None, "subject": "a", "margin_px": 16,
                   "carried_is": "the mask each window was given", "files": ["piece_a_window_1_planned_region.npz"]}
        folder = capture(TMP / "cap_planned", 10, 30, {"a": on, "b": mask_of(RECT_B, 30)}, [planned, {**planned, "name": "run_b", "subject": "b"},
                                                                                    {**planned, "name": "run_c", "render": "another.mp4"}],
                         carried={"run_a": held})
        o_in, a_in = original(24), planes_of(PIECE_A, 24)

        def box_mean(x, y, rect):
            return float(np.abs(x[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3] - y[0][rect[1] + 3:rect[3] - 3, rect[0] + 3:rect[2] - 3]).mean())
        carried_flag = lambda r: [f for f in r["flags"] if f["rule"] == "piece_changed_where_its_run_carried_no_mask"]   # noqa: E731
        # the row as it was read until now: whose it is and no run. Nothing is read under the mask the run carried
        code, r, out = deliver(TMP, "planned_subject_only", [(10, 39, PIECE_A, "10 subject=a")], "10-39", "--capture", str(folder))
        assert code == 0 and not carried_flag(r) and not r["written_to_captures"] and box_mean(planes_of(str(out), 24), a_in, RECT_A) < 1.5, \
            "a row with subject= alone was read under a run's mask: the case below would prove nothing"
        # the same row naming its run: the five frames the run carried nothing on are the source's and are named
        code, r, out = deliver(TMP, "planned_run_named", [(10, 39, PIECE_A, f"10 subject=a run={folder}:run_a")], "10-39", "--capture", str(folder))
        flag = carried_flag(r)
        got = planes_of(str(out), 24)
        assert code == 0 and flag and flag[0]["source_frames"] == [[22, 26]] and flag[0]["figures"]["left_as_the_source"] is True and flag[0]["run"] == "run_a", \
            f"a row that names its run was not read under the mask that run carried: {flag}"
        assert box_mean(got, o_in, RECT_A) < 1.5 and box_mean(got, a_in, RECT_A) > 20, "the frame the run carried nothing on still shows the piece"
        assert not r["pieces_with_no_capture"] and any(w.endswith("runs/run_a/changed__planned_run_named.csv") for w in r["written_to_captures"]), r["written_to_captures"]
        assert tool.read_table(table(TMP / "run_row.txt", [(10, 39, PIECE_A, "10 run=data/a_capture:its:run")]))[0]["run"] == ("data/a_capture:its", "run")
        # what is refused, each before anything is written
        refusals = {"a folder that is not a capture given": (f"10 subject=a run={TMP / 'cap_elsewhere'}:run_a", ["--capture", str(folder)], "not one of the captures given"),
                    "a run the capture does not have": (f"10 subject=a run={folder}:run_z", ["--capture", str(folder)], "holds no run named 'run_z'"),
                    "a run of another subject": (f"10 subject=a run={folder}:run_b", ["--capture", str(folder)], "the run is of 'b'"),
                    "a run that names another render": (f"10 subject=a run={folder}:run_c", ["--capture", str(folder)], "names another render"),
                    "no capture given at all": (f"10 subject=a run={folder}:run_a", [], "no --capture was given"),
                    "a token that is no run": ("10 subject=a run=run_a", ["--capture", str(folder)], "is not `run=<capture folder>:<run name>`")}
        for name, (rest, more, words) in refusals.items():
            stem = "refused_" + name.replace(" ", "_")
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(TMP / "source.mp4"), "--table", str(table(TMP / f"{stem}.txt", [(10, 39, PIECE_A, rest)])),
                                   "--span", "10-39", "--out", str(TMP / f"{stem}.mp4"), *more], capture_output=True, text=True)
            assert proc.returncode != 0 and words in proc.stderr + proc.stdout and not (TMP / f"{stem}.mp4").exists(), f"{name}: not refused in words before the build: {(proc.stderr or proc.stdout)[-200:]}"
        return ("a row that names the run of a capture made before its render is read under the mask that run carried (five frames left as the source, "
                "named; its table written into the capture); with subject= alone it is not; six malformed ties are refused before anything is written")

    def ends_on_the_source() -> str:
        base = WHOLE[10:40]
        on = mask_of(RECT_A, 30, grow=4)
        x0, y0, x1, y1 = RECT_A

        def thinning(name, strength):
            """A piece whose paint in the rectangle has `strength[k]` of its full step on frame k of thirty."""
            out = base.clone()
            for k, v in enumerate(strength):
                out[k, y0:y1, x0:x1] = (out[k, y0:y1, x0:x1] + 0.35 * v).clamp(0, 1)
            return write(name, out)
        full = [1.0] * 30
        fades_out = thinning("ends_out", full[:25] + [0.6, 0.45, 0.3, 0.15, 0.0])
        fades_in = thinning("ends_in", [0.0, 0.15, 0.3, 0.45, 0.6] + full[5:])
        holds = thinning("ends_hold", full)

        def go(name, piece, last=39, changed_json=None):
            run = {"name": "run_a", "render": Path(piece).name, "subject": "a", "margin_px": 16}
            folder = capture(TMP / f"cap_{name}", 10, 30, {"a": on}, [run])
            if changed_json is not None:
                (folder / "runs" / "run_a").mkdir(parents=True, exist_ok=True)
                (folder / "runs" / "run_a" / "changed.json").write_text(json.dumps({"ends": changed_json}))
            code, r, _ = deliver(TMP, name, [(10, last, piece, 10)], "10-39", "--capture", str(folder))
            return code, r, r["ends"]["rows"][0]
        code, r, row = go("ends_out", fades_out)
        said = [f for f in r["failures"] if "ends on the source's own picture" in f]
        # the run is every frame from the end under two thirds of the middle: the five thinned ones (0.6 of full and less)
        assert code == 1 and said and row["end"]["on_the_source"] == [35, 39] and row["start"]["on_the_source"] is None, (code, r["failures"], row)
        assert "39" in r["verdict_line"] and r["verdict_line"].startswith("FAILS"), r["verdict_line"]
        code, r, row = go("ends_in", fades_in)
        assert code == 1 and any("begins on the source's own picture" in f for f in r["failures"]) and row["start"]["on_the_source"][0] == 10 \
            and row["end"]["on_the_source"] is None, (code, r["failures"], row)
        code, r, row = go("ends_hold", holds)
        assert code == 0 and row["end"]["on_the_source"] is None and row["start"]["on_the_source"] is None \
            and 0.85 < row["end"]["tail_levels_over_middle"] < 1.15 and 0.85 < row["end"]["outermost_frame_over_middle"] < 1.15, (code, r["failures"], row)
        held_to = row["end"]["outermost_frame_over_middle"]
        # the row cut short of the thinned frames passes; what is left of the shot is said, not failed
        code, r, row = go("ends_cut", fades_out, last=35)
        assert code == 0 and row["end"]["on_the_source"] is None, (code, r["failures"], row)
        assert r["shots"]["shots"][0]["subjects"]["a"]["tracked_with_nothing_laid"]["runs"] == [[36, 39]], r["shots"]["shots"][0]["subjects"]["a"]
        # this reading passes, and the capture's own reading of the render says it ends on the source on frames the row lays
        theirs = {"level": 30.0, "starts_on_the_source": None, "ends_on_the_source": {"source_frames": [37, 39], "frames": 3}}
        code, r, row = go("ends_by_capture", holds, changed_json=theirs)
        assert code == 1 and any("by its capture's reading" in f and "37-39" in f for f in r["failures"]) and r["ends"]["the_captures_read"][0]["ends_on_the_source"]["frames"] == 3, \
            (code, r["failures"])
        code, r, row = go("ends_by_capture_cut", holds, last=36, changed_json=theirs)
        assert code == 0, f"a row cut short of the frames its capture names was failed on them: {r['failures']}"
        return (f"paint thinning to nothing over a row's last five frames fails by its frames, and at its start as 'begins'; a row that holds reads "
                f"{held_to:.2f} of its middle on its last frame and passes; cut short of the thinned frames it passes; the capture's own reading "
                "of the render fails a row that lays the frames it names")

    def laid_by_its_region() -> str:
        """The lead's rulings of 2026-10-16: a piece whose records all say composite "whole region" is laid by its run's region
        and what it changed past it, at both sizes; another subject's row keeps what it holds and she does not; "only what
        changed", no graph, a run not the node's own, a run or a window that says another composite, and a composite the node
        has not each keep `changed`, with why; a piece laid by both rules is flagged by its frames."""
        BOX = (32, 32, 144, 128)                                         # the region H3 regenerated, cell-aligned: 16-px cells 2..8, 2..7
        PAST = (60, 132, 90, 150)                                        # a change past the region's edge, as the node's feather lays one
        piece = write("fw_00001", painted(faint(WHOLE[10:40], BOX), PAST))
        cells = np.zeros((30, H // 16, W // 16), bool)
        cells[:, BOX[1] // 16:BOX[3] // 16, BOX[0] // 16:BOX[2] // 16] = True
        OTHER = (104, 40, 170, 128)                                      # another subject, over her region beyond her own mask

        def windows(*composites):
            folder = TMP / "fw_windows"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir()
            for k, c in enumerate(composites, 1):
                np.savez(folder / f"fw_window_{k}_region.npz", meta=json.dumps({"source": {"composite": c}}))

        def run_of(**more):
            return {"name": "run_f", "render": "fw_00001.mp4", "subject": "a", "margin_px": 16, "files": ["fw_window_1_region.npz"],
                    "composite": "whole region", "masked_source": {"composite": "whole region"}, **more}

        def capture_of(tag, run, read=None):
            folder = capture(TMP / f"cap_fw_{tag}", 10, 30, {"a": mask_of(RECT_A, 30), "b": mask_of(OTHER, 30)}, [run], region={"run_f": cells})
            if read is not None:
                z = dict(np.load(folder / "runs" / "run_f" / "region.npz"))
                np.savez_compressed(folder / "runs" / "run_f" / "region.npz", **{**z, "read": read})
            return folder

        def near(luma, want, rect):
            x0, y0, x1, y1 = rect
            return float(np.abs(luma - want[0])[y0 + SLACK:y1 - SLACK, x0 + SLACK:x1 - SLACK].mean())
        o, f = original(24), planes_of(piece, 24)
        assert 2.0 < inside(f, o, RECT_A) < tool.CHANGE, f"the fixture: the faint piece is {inside(f, o, RECT_A):.2f} levels off the original, not under CHANGE"
        graph_beside(piece, "whole region")
        windows("whole region")
        tail = np.ones(30, bool)
        tail[25:] = False                                                # source frames 35-39: no region read
        folder = capture_of("main", run_of(), tail)
        code, r, out = deliver(TMP, "fw_source", [(10, 39, piece, 10)], "10-39", "--capture", str(folder), "--size", "source")
        luma = list(tool.luma_at_canvas(str(out), W, H))[24 - 10]
        row = r["regions"]["pieces"][piece]["laid_by"]
        # THE CASE THAT GOES RED when a whole-region piece's barely-changed pixels are laid back as the source
        assert code == 0 and near(luma, f, RECT_A) + 1.0 < near(luma, o, RECT_A), f"at the source's size: {near(luma, f, RECT_A):.2f} from the piece, {near(luma, o, RECT_A):.2f} from the original"
        # THE CASE THAT GOES RED when the region alone is laid and a change past its edge is dropped (fix 6)
        assert near(luma, f, PAST) + 5.0 < near(luma, o, PAST), f"past the region's edge: {near(luma, f, PAST):.2f} from the piece, {near(luma, o, PAST):.2f} from the original"
        # THE CASE THAT GOES RED when the counts or the switch of rule are not said (fixes 4 and 5)
        assert row["rule"] == "region" and row["frames_by_region"] == 25 and row["frames_by_changed"] == 5, row
        flag = [x for x in r["flags"] if x["rule"] == "piece_laid_by_two_rules"]
        assert flag and flag[0]["source_frames"] == [[35, 39]], flag
        # at the canvas's size, as a later row over another subject's piece that changed him only past her region: her region is laid,
        # and his pixels inside it are his row's (the original there)
        his_piece = write("his_00001", painted(WHOLE[10:40], (150, 50, 168, 120)))     # his row changes him past her region, not inside it
        code_c, r_c, out_c = deliver(TMP, "fw_canvas", [(10, 39, his_piece, "10 subject=b"), (10, 39, piece, 10)], "10-39", "--capture", str(folder))
        got = planes_of(str(out_c), 24)
        row_c = r_c["regions"]["pieces"][piece]["laid_by"]
        assert code_c == 0 and inside(got, f, RECT_A) + 1.0 < inside(got, o, RECT_A), (code_c, r_c["failures"], inside(got, f, RECT_A), inside(got, o, RECT_A))
        assert row_c["frames_by_region"] == 25 and row_c["frames_by_changed"] == 5, row_c
        # THE CASE THAT GOES RED when another person inside her region is laid from her render (fix 1)
        his = (110, 48, 140, 120)
        assert inside(got, o, his) + 1.0 < inside(got, f, his), f"another subject inside her region: {inside(got, f, his):.2f} from her piece, {inside(got, o, his):.2f} from the original"
        # each record that does not say "whole region" keeps `changed`, with why (fixes 2, 3, 7)
        cases = {"only what changed": ("only what changed", run_of(), ("whole region",), "rule"),
                 "no graph": (None, run_of(), ("whole region",), "no fw_00001.png beside the piece"),
                 "not the node's own": ("whole region", run_of(files=None), ("whole region",), "is not the node's own"),
                 "the run says another": ("whole region", run_of(composite="only what changed"), ("whole region",), "the capture's run records the composite 'only what changed'"),
                 "the graph's record says another": ("whole region", run_of(masked_source={"composite": "only what changed"}), ("whole region",), "the capture's graph records"),
                 "a window says another": ("whole region", run_of(), ("whole region", "only what changed"), "its windows do not all say"),
                 "no windows": ("whole region", run_of(), (), "no fw_windows/*_region.npz"),
                 "not the node's word": ("half a region", run_of(), ("whole region",), "is not one of the node's")}
        for tag, (composite, run, wins, words) in cases.items():
            graph_beside(piece, composite)
            windows(*wins)
            folder = capture_of(tag.replace(" ", "_").replace("'", ""), run)
            _code, r, out = deliver(TMP, f"fw_{tag.replace(' ', '_').replace(chr(39), '')}", [(10, 39, piece, 10)], "10-39", "--capture", str(folder), "--size", "source")
            luma = list(tool.luma_at_canvas(str(out), W, H))[24 - 10]
            row = r["regions"]["pieces"][piece]["laid_by"]
            # laid by `changed`, a piece under CHANGE is not laid: its pixels are the original's
            assert row["rule"] == "changed" and row["frames_by_region"] == 0 and near(luma, o, RECT_A) + 1.0 < near(luma, f, RECT_A), (tag, row)
            assert tag == "only what changed" or words in row["why"], (tag, row)
        graph_beside(piece, None)
        shutil.rmtree(TMP / "fw_windows", ignore_errors=True)
        return ("a whole-region piece laid by its region and what it changed past it, at the source's size and as a later row at the canvas's; another "
                "subject inside her region keeps his own; the counts and a switch of rule said; each record that does not say whole region keeps `changed`, said")

    def a_zoomed_piece() -> str:
        import torch.nn.functional as F
        base = WHOLE[10:40]
        box = (64, 48, 192, 144)                                   # at the canvas; twice its size is the canvas: zoom 2
        bx0, by0, bx1, by1 = box
        inside = base[:, by0:by1, bx0:bx1].permute(0, 3, 1, 2)
        zoomed = F.interpolate(inside, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).clone()
        zx0, zy0, zx1, zy1 = 60, 40, 200, 150                      # what the render changed, in the zoomed picture
        # stripes one zoomed pixel wide: reduced to the canvas they are a flat grey, so the piece cannot hold them
        # and only a picture taken from the zoomed file can
        stripes = (torch.arange(zx0, zx1) % 2).float().view(1, 1, -1, 1)
        # and bands sixteen zoomed pixels wide, which the piece does hold: what a box in the wrong place disagrees on
        bands = ((torch.arange(zx0, zx1) // 16) % 2).float().view(1, 1, -1, 1)
        zoomed[:, zy0:zy1, zx0:zx1] = (zoomed[:, zy0:zy1, zx0:zx1] * 0.4 + 0.1 + 0.25 * bands + 0.25 * stripes).clamp(0, 1)
        down = F.avg_pool2d(zoomed.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        piece = base.clone()
        px0, py0, px1, py1 = bx0 + zx0 // 2, by0 + zy0 // 2, bx0 + zx1 // 2, by0 + zy1 // 2
        piece[:, py0:py1, px0:px1] = down[:, zy0 // 2:zy1 // 2, zx0 // 2:zx1 // 2]      # the node's own laying back, exactly

        def pair(name, frames=None, **said):
            path = write(name, piece)
            write(name + "_zoomed", zoomed if frames is None else zoomed[:frames])
            note = {"file": tool.ZOOM_FILE, "version": 1, "run": name, "piece": f"{name}.mp4", "zoomed": f"{name}_zoomed.mp4", "frames": 30,
                    "canvas": [W, H], "source_size": [W, H], "box": list(box), "fixed": True, "zoom": 2.0, "region_cut_at_box_sides": [], **said}
            (TMP / f"{name}_zoomed.json").write_text(json.dumps(note))
            return path

        def detail(path) -> float:
            """How much the delivered picture changes from one source pixel to the next, inside the region."""
            cw, ch, x0, y0 = tool.crop_of(SW, SH, W, H)
            y = tool.planes(next(tool.source_frames(str(path), 10, 10, SW, SH)), SW, SH)[0]
            a, b = x0 + (px0 + 4) * cw // W, x0 + (px1 - 4) * cw // W
            c, d = y0 + (py0 + 4) * ch // H, y0 + (py1 - 4) * ch // H
            return float(np.abs(np.diff(y[c:d, a:b], axis=1)).mean())
        code, r, with_it = deliver(TMP, "zoom_full", [(10, 39, pair("zoom_good"), 10)], "10-39", "--size", "source")
        z = r["zoomed"][0] if r.get("zoomed") else None
        assert code == 0 and z and z["picture_from"] == "zoom_good_zoomed.mp4" and z["agreement_levels_worst_frame"][0] < tool.ZOOM_AGREES / 2 \
            and z["frames_read"] == 30 and z["region_px_outside_the_box_most"] == 0, (code, r["failures"], z)
        code, r, without = deliver(TMP, "zoom_plain", [(10, 39, write("zoom_no_files", piece), 10)], "10-39", "--size", "source")
        assert code == 0 and r["zoomed"] == [], (code, r["zoomed"])
        fine, coarse = detail(with_it), detail(without)
        assert fine > 1.3 * coarse, f"the picture laid from the zoomed file carries no more of its stripes ({fine:.2f}) than the piece laid alone ({coarse:.2f})"
        # at the canvas's size the zoomed file is not used: the file is the piece's
        code, r, small = deliver(TMP, "zoom_canvas", [(10, 39, pair("zoom_good_canvas"), 10)], "10-39")
        code2, _r, small_plain = deliver(TMP, "zoom_canvas_plain", [(10, 39, write("zoom_no_files_canvas", piece), 10)], "10-39")
        apart = float(np.abs(planes_of(str(small), 20)[0] - planes_of(str(small_plain), 20)[0]).mean())
        assert code == 0 and code2 == 0 and "not used" in r["zoomed"][0]["picture_from"] and apart < 0.5, (code, code2, r["zoomed"], apart)
        # the box in the json moved: the two are no longer one picture inside the region
        moved = [bx0 + 8, by0, bx1 + 8, by1]
        code, r, _ = deliver(TMP, "zoom_moved", [(10, 39, pair("zoom_moved_box", box=moved), 10)], "10-39", "--size", "source")
        assert code == 1 and any("is not the piece's picture inside the region" in f and "zoom_moved_box.mp4" in f for f in r["failures"]), (code, r["failures"])
        # refused before anything is read
        refused = {}
        for name, break_it in (("no zoomed file", lambda n: (TMP / f"{n}_zoomed.mp4").unlink()), ("another frame count", None),
                               ("a box not fixed", None), ("another run's name", None)):
            stem = "zoom_bad_" + name.replace(" ", "_").replace("'", "")
            path = pair(stem, **({"fixed": False} if name == "a box not fixed" else {"run": "somebody_else"} if name == "another run's name" else {}),
                        frames=20 if name == "another frame count" else None)
            if break_it:
                break_it(stem)
            proc = subprocess.run([sys.executable, str(Path(tool.__file__)), "--source", str(SOURCE), "--table", str(table(TMP / f"{stem}.txt", [(10, 39, path, 10)])),
                                   "--span", "10-39", "--size", "source", "--out", str(TMP / f"{stem}_out.mp4")], capture_output=True, text=True)
            refused[name] = (proc.returncode != 0 and "_zoomed.json" in (proc.stderr + proc.stdout) and not (TMP / f"{stem}_out.mp4").exists())
        assert all(refused.values()), f"not refused before anything was written: {[k for k, v in refused.items() if not v]}"
        return (f"laid from the zoomed file the region changes {fine:.1f} levels a source pixel against {coarse:.1f} from the piece alone; the two "
                f"agree to {z['agreement_levels_worst_frame'][0]} levels; a box moved eight pixels fails; four malformed pairs are refused; "
                "at the canvas's size the zoomed file is not used")

    def laid_by_the_matte() -> str:
        """The lead's ruling of 2026-10-16, "the exact matte": a piece whose windows saved the node's own matte is laid wherever the node
        blended any of its render in, and by nothing rebuilt. Each piece here is made by the node's own functions (`lay_window`, and
        `lay_back` on a zoomed load whose box is off the 16-px grid at 1.333x), its render four levels off the source inside the region
        (806's signature: under `CHANGE`), and delivered at the source's size and at the canvas's as a later row: inside the matte the
        delivery is the node's composite to within encode noise, and outside it the source's."""
        vm = sys.modules["_h3pack.video_mask"]
        import cv2 as cv
        base = WHOLE[10:36]                                               # 26 frames: eight latent steps (`run_lengths`)
        cells = torch.zeros(8, H // 16, W // 16)
        cells[:, 3:8, 4:11] = 1.0                                         # x 64-176, y 48-128 of the picture the node regenerates in
        mask = torch.zeros(26, H, W)
        mask[:, 60:116, 76:164] = 1.0
        src = {"composite": vm.COMPOSITE_REGION, "feather_pixels": 6, "cuts": None, "change_threshold": vm.CHANGE_THRESHOLD}
        made = {}
        for tag, box in (("plain", None), ("zoomed", (38, 30, 230, 174))):
            if box is None:
                pixels = base
            else:
                zoom = {"box": list(box), "source_size": [W, H], "canvas": [W, H], "scale": W / (box[2] - box[0]),
                        "render_region": vm.RENDER_ZOOM_REPLACED, "set_by": vm.SET_BY_LARGEST}
                pixels = vm.crop_frames(base, box, W, H)
            render = faint(pixels, (64, 48, 176, 128))
            composite, alpha, _lines = vm.lay_window(render, pixels, cells, mask, src, 8, 0)
            if box is None:
                piece_frames, matte = composite, vm.matte_of(alpha)
            else:
                piece_frames, matte = vm.lay_back(render, alpha, base, zoom, weight=True)
            stem = f"mt_{tag}_00001"
            path = write(stem, piece_frames)
            if box is not None:
                write(stem + "_zoomed", composite)
                (TMP / f"{stem}_zoomed.json").write_text(json.dumps({"file": tool.ZOOM_FILE, "version": 2, "run": stem, "piece": f"{stem}.mp4",
                    "zoomed": f"{stem}_zoomed.mp4", "frames": 26, "canvas": [W, H], "source_size": [W, H], "box": list(box), "fixed": True,
                    "zoom": zoom["scale"], "region_cut_at_box_sides": []}))
            folder = TMP / f"mt_{tag}_windows"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir()
            vm.save_window_region(str(folder / f"mt_{tag}_window_1_region.npz"), mask, cells, src, 8, 0, 0, matte=matte,
                                  piece={"stem": stem, "first": 0, "written": 26})
            (folder / f"mt_{tag}_window_1.mp4").write_bytes(b"")
            made[tag] = (path, matte.numpy(), piece_frames)
        plain = write("mt_base_00001", base)
        k = 24 - 10
        said = []
        for tag, (path, matte, piece_frames) in made.items():
            m = matte[k]
            laid = cv.erode((m == 255).astype(np.uint8), np.ones((2 * SLACK + 1,) * 2, np.uint8)).astype(bool)
            off = ~cv.dilate((m > 0).astype(np.uint8), np.ones((2 * SLACK + 1,) * 2, np.uint8)).astype(bool)
            assert laid.sum() > 500 and off.sum() > 500, (tag, int(laid.sum()), int(off.sum()))
            f, o = planes_of(path, 24), original(24)
            assert 2.0 < float(np.abs(f[0] - o[0])[laid].mean()) < tool.CHANGE, f"the fixture: {tag}'s render is not under CHANGE from the source"
            for size in ("source", "canvas"):
                name = f"mt_{tag}_{size}"
                rows = [(10, 35, plain, 10), (10, 35, path, 10)]
                code, r, out = deliver(TMP, name, rows, "10-35", *(("--size", "source") if size == "source" else ()))
                got = list(tool.luma_at_canvas(str(out), W, H))[k] if size == "source" else planes_of(str(out), 24)[0]
                row = r["regions"]["pieces"][path]["laid_by"]
                near_piece, near_source = float(np.abs(got - f[0])[laid].mean()), float(np.abs(got - o[0])[laid].mean())
                outside = float(np.abs(got - o[0])[off].mean())
                # THE CASE THAT GOES RED when a piece under CHANGE where its matte is 1 is not laid wholly (806), or is laid by anything but its matte
                assert code == 0 and row["rule"] == "matte" and row["frames_by_matte"] == 26, (tag, size, code, r["failures"], row)
                assert near_piece < 1.5 and near_piece + 1.0 < near_source, f"{tag} at the {size}'s size: {near_piece:.2f} from the node's composite, {near_source:.2f} from the source"
                assert outside < 1.5, f"{tag} at the {size}'s size: {outside:.2f} from the source outside the matte"
                said.append(f"{tag}/{size} {near_piece:.2f}")
        # TWO PIECES THAT TOUCH (a Gemini reader's finding, 2026-10-16): each matte-laid row's weight is its own pixels, never grown or blurred,
        # so across the shared edge every pixel is one node composite or the other, not a feathered mix of the two
        touching = {}
        for tag, cols, xs in (("left", (4, 8), (76, 124)), ("right", (8, 12), (132, 180))):
            c2 = torch.zeros(8, H // 16, W // 16)
            c2[:, 3:8, cols[0]:cols[1]] = 1.0
            m2 = torch.zeros(26, H, W)
            m2[:, 60:116, xs[0]:xs[1]] = 1.0
            render = painted(base, (cols[0] * 16, 48, cols[1] * 16, 128))
            comp, alpha, _ = vm.lay_window(render, base, c2, m2, src, 8, 0)
            path = write(f"mt_{tag}_00001", comp)
            folder = TMP / f"mt_{tag}_windows"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir()
            vm.save_window_region(str(folder / f"mt_{tag}_window_1_region.npz"), m2, c2, src, 8, 0, 0, matte=vm.matte_of(alpha),
                                  piece={"stem": f"mt_{tag}_00001", "first": 0, "written": 26})
            (folder / f"mt_{tag}_window_1.mp4").write_bytes(b"")
            touching[tag] = (path, vm.matte_of(alpha).numpy()[k].astype(np.float32) / 255.0, render)
        a_ = planes_of(touching["left"][0], 24)[0]
        m_r, r_r = touching["right"][1], planes_of(write("mt_right_render_00001", touching["right"][2]), 24)[0]
        # where the two soft bands overlap, the later render over the earlier piece by the later matte (as the overlap case below holds);
        # elsewhere each is its own composite
        want_t = m_r * r_r + (1.0 - m_r) * a_
        band = (slice(64, 112), slice(110, 146))
        for size in ("canvas", "source"):
            code, r, out = deliver(TMP, f"mt_touch_{size}", [(10, 35, plain, 10), (10, 35, touching["left"][0], 10), (10, 35, touching["right"][0], 10)], "10-35",
                                   *(("--size", "source") if size == "source" else ()))
            got = list(tool.luma_at_canvas(str(out), W, H))[k] if size == "source" else planes_of(str(out), 24)[0]
            apart = float(np.abs(got - want_t)[band].mean())
            # THE CASE THAT GOES RED when a matte-laid row is grown and blurred where two pieces touch
            assert code == 0 and apart < 2.0, f"two touching pieces at the {size}'s size: {apart:.2f} levels from the later render over the earlier piece across the shared edge"
            said.append(f"touching/{size} {apart:.2f}")
        # TWO PIECES THAT OVERLAP (Gemini's D18, finding 2): the later row's soft band lies inside the earlier row's region, where the node
        # blended its render against the source; laid there, the earlier row is what lies beneath, so the band is the later render blended
        # over the earlier piece by the node's own weight, not over the source
        over = {}
        for tag, cols, xs in (("under", (4, 10), (76, 156)), ("over", (8, 12), (132, 180)), ("third", (7, 11), (118, 166))):
            c2 = torch.zeros(8, H // 16, W // 16)
            c2[:, 3:8, cols[0]:cols[1]] = 1.0
            m2 = torch.zeros(26, H, W)
            m2[:, 60:116, xs[0]:xs[1]] = 1.0
            render = (painted(base, (cols[0] * 16, 48, cols[1] * 16, 128)) if tag == "under" else
                      faint(base, (cols[0] * 16, 48, cols[1] * 16, 128), by=60 / 255 if tag == "over" else -50 / 255))
            comp, alpha, _ = vm.lay_window(render, base, c2, m2, src, 8, 0)
            path = write(f"mt_{tag}_00001", comp)
            folder = TMP / f"mt_{tag}_windows"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir()
            vm.save_window_region(str(folder / f"mt_{tag}_window_1_region.npz"), m2, c2, src, 8, 0, 0, matte=vm.matte_of(alpha),
                                  piece={"stem": f"mt_{tag}_00001", "first": 0, "written": 26})
            (folder / f"mt_{tag}_window_1.mp4").write_bytes(b"")
            over[tag] = (path, vm.matte_of(alpha).numpy()[k].astype(np.float32) / 255.0, render)
        u_ = planes_of(over["under"][0], 24)[0]
        m_, m3 = over["over"][1], over["third"][1]
        r_y = planes_of(write("mt_over_render_00001", over["over"][2]), 24)[0]
        r3 = planes_of(write("mt_third_render_00001", over["third"][2]), 24)[0]
        rows_band = np.zeros_like(m_, dtype=bool)
        rows_band[64:112] = True
        # THE WHOLE OVERLAP (the D18 read): wherever the later row's matte is above zero over the earlier row's region, its soft band and its
        # full region alike; the law is the later render over what lies beneath, by its own matte
        whole_over = rows_band & (m_ > 0.02) & (over["under"][1] > 0.02)
        want = m_ * r_y + (1.0 - m_) * u_
        assert whole_over.sum() > 300 and float(np.abs(u_ - original(24)[0])[whole_over].mean()) > 20, "the fixture: the earlier row does not change the overlap"
        assert (m_[whole_over] > 0.99).any() and ((m_[whole_over] > 0.05) & (m_[whole_over] < 0.95)).any(), "the fixture: the overlap holds no full or no soft band"
        three_over = rows_band & (m3 > 0.02) & ((m_ > 0.02) | (over["under"][1] > 0.02))
        want3 = m3 * r3 + (1.0 - m3) * want
        for size in ("canvas", "source"):
            code, r, out = deliver(TMP, f"mt_overlap_{size}", [(10, 35, plain, 10), (10, 35, over["under"][0], 10), (10, 35, over["over"][0], 10)], "10-35",
                                   *(("--size", "source") if size == "source" else ()))
            got = list(tool.luma_at_canvas(str(out), W, H))[k] if size == "source" else planes_of(str(out), 24)[0]
            apart = float(np.abs(got - want)[whole_over].mean())
            # THE CASE THAT GOES RED when a matte row is laid against the source over the row beneath it, in its soft band or its full region
            assert code == 0 and apart < 3.0, f"two overlapping pieces at the {size}'s size: the whole overlap is {apart:.2f} levels from the later render over the earlier piece"
            said.append(f"overlap/{size} {apart:.2f}")
            code, r, out = deliver(TMP, f"mt_three_{size}", [(10, 35, plain, 10), (10, 35, over["under"][0], 10), (10, 35, over["over"][0], 10),
                                                            (10, 35, over["third"][0], 10)], "10-35", *(("--size", "source") if size == "source" else ()))
            got = list(tool.luma_at_canvas(str(out), W, H))[k] if size == "source" else planes_of(str(out), 24)[0]
            apart3 = float(np.abs(got - want3)[three_over].mean())
            # THE CASE THAT GOES RED when three overlapping matte rows are not each laid over what lies beneath
            assert code == 0 and apart3 < 3.0, f"three overlapping pieces at the {size}'s size: {apart3:.2f} levels from each render laid over the ones beneath"
            said.append(f"three/{size} {apart3:.2f}")
        # A MATTE ROW BESIDE A PLAIN ONE (the D18 read, finding 2): a later row laid by what it changed feathers only into pixels nobody laid;
        # the matte row's soft band is laid by it, so the plain row's grown feather never pulls that band toward its own picture
        plain_beside = painted(base, (192, 64, 230, 112))                 # a change right beside the over row's soft band (x 132-180, its band to ~190)
        beside_path = write("mt_beside_00001", plain_beside)
        m_o = over["over"][1]
        band_beside = rows_band & (m_o > 0.02) & (m_o < 0.98)
        band_beside[:, :176] = False
        band_beside[:, 190:] = False                                      # off the pixels the plain row's own change claims (its `changed` spreads past it)
        want_b = m_o * r_y + (1.0 - m_o) * original(24)[0]
        assert band_beside.sum() > 50, "the fixture: no soft band of the matte row lies within reach of the plain row's feather"
        for size in ("canvas", "source"):
            code, r, out = deliver(TMP, f"mt_beside_{size}", [(10, 35, plain, 10), (10, 35, over["over"][0], 10), (10, 35, beside_path, 10)], "10-35",
                                   *(("--size", "source") if size == "source" else ()))
            got = list(tool.luma_at_canvas(str(out), W, H))[k] if size == "source" else planes_of(str(out), 24)[0]
            apart_b = float(np.abs(got - want_b)[band_beside].mean())
            # THE CASE THAT GOES RED when a plain row's grown feather pulls a matte row's soft band toward its own picture
            assert code == 0 and apart_b < 2.0, f"a matte row beside a plain one at the {size}'s size: its soft band is {apart_b:.2f} levels from its own law"
            said.append(f"beside/{size} {apart_b:.2f}")
        # A MATTE ROW THAT IS ALL SOFT BAND (the D18 read): its weight never reaches full, so it owns no pixel outright; it is laid all the same
        c4 = torch.zeros(8, H // 16, W // 16)
        c4[:, 3:8, 4:11] = 1.0
        render4 = painted(base, (64, 48, 176, 128))
        _comp4, alpha4, _ = vm.lay_window(render4, base, c4, mask, src, 8, 0)
        alpha4 = alpha4 * 0.8
        comp4 = vm.composite(render4, base, alpha4)
        soft_path = write("mt_soft_00001", comp4)
        folder4 = TMP / "mt_soft_windows"
        shutil.rmtree(folder4, ignore_errors=True)
        folder4.mkdir()
        vm.save_window_region(str(folder4 / "mt_soft_window_1_region.npz"), mask, c4, src, 8, 0, 0, matte=vm.matte_of(alpha4),
                              piece={"stem": "mt_soft_00001", "first": 0, "written": 26})
        (folder4 / "mt_soft_window_1.mp4").write_bytes(b"")
        m4 = vm.matte_of(alpha4).numpy()[k]
        assert m4.max() < 255 and (m4 > 150).sum() > 300, "the fixture: the matte reaches full, or holds no strong band"
        code, r, out = deliver(TMP, "mt_soft_source", [(10, 35, plain, 10), (10, 35, soft_path, 10)], "10-35", "--size", "source")
        got4 = list(tool.luma_at_canvas(str(out), W, H))[k]
        p4, o4 = planes_of(soft_path, 24)[0], original(24)[0]
        strong = m4 > 150
        # THE CASE THAT GOES RED when a matte row with no full pixel is skipped at the source's size
        assert code == 0 and float(np.abs(got4 - p4)[strong].mean()) + 5.0 < float(np.abs(got4 - o4)[strong].mean()), (
            f"an all-soft matte row with no full pixel was skipped at the source's size: {float(np.abs(got4 - p4)[strong].mean()):.2f} from piece vs "
            f"{float(np.abs(got4 - o4)[strong].mean()):.2f} from source")
        shutil.rmtree(folder4, ignore_errors=True)
        # A PIECE LAID BY TWO RULES: its second window saved no matte, so its frames keep `changed`, and the flag names them
        path, matte, piece_frames = made["plain"]
        folder = TMP / "mt_plain_windows"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        vm.save_window_region(str(folder / "mt_plain_window_1_region.npz"), mask[:13], cells, src, 8, 0, 0, matte=torch.from_numpy(matte[:13]),
                              piece={"stem": "mt_plain_00001", "first": 0, "written": 13})
        vm.save_window_region(str(folder / "mt_plain_window_2_region.npz"), mask[13:], cells, src, 8, 13, 0,
                              piece={"stem": "mt_plain_00001", "first": 13, "written": 13})
        for n_ in (1, 2):
            (folder / f"mt_plain_window_{n_}.mp4").write_bytes(b"")
        code, r, _out = deliver(TMP, "mt_two_rules", [(10, 35, path, 10)], "10-35")
        row = r["regions"]["pieces"][path]["laid_by"]
        flag = [x for x in r["flags"] if x["rule"] == "piece_laid_by_two_rules"]
        # THE CASE THAT GOES RED when a piece laid by its matte on some frames and by `changed` on others is not flagged
        assert row["frames_by_matte"] == 13 and row["frames_by_changed"] == 13 and flag and flag[0]["source_frames"] == [[23, 35]], (
            f"a piece laid by two rules was not flagged on its frames: {row}, {flag}")
        # A CONTINUED RUN (the matte's blind read, finding 1): its first window holds the context's frames before the piece begins, so the
        # piece's frame k is the window's frame trim + k; read at k, the matte is `trim` frames early. Here the window holds 3 frames of context
        # whose matte is the next frames' reversed; and A SECOND RUN OF THE PREFIX in the same folder (finding 3) is not read
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        ctx_mask = torch.cat([mask[:3], mask])
        early = torch.cat([torch.zeros_like(torch.from_numpy(matte[:3])), torch.from_numpy(matte)])     # the context's frames: nothing laid
        vm.save_window_region(str(folder / "mt_plain_window_1_region.npz"), ctx_mask, cells, src, 8, 0, 3, matte=early,
                              piece={"stem": "mt_plain_00001", "first": 0, "written": 26})
        (folder / "mt_plain_window_1.mp4").write_bytes(b"")
        vm.save_window_region(str(folder / "mt_plain_window_2_region.npz"), mask, cells, src, 8, 26, 0, matte=torch.zeros_like(early[:26]),
                              piece={"stem": "mt_plain_00002", "first": 0, "written": 26})
        (folder / "mt_plain_window_2.mp4").write_bytes(b"")
        reader = tool.Matte(path)
        # THE CASE THAT GOES RED when a continued run's matte is read `trim` frames early, or another run's file is read as this piece's
        assert matte[0].any() and all(np.array_equal(reader.at(q), matte[q]) for q in (0, 1, 2, 14, 25)) and reader.gaps == [] \
            and reader.other_stems == ["mt_plain_window_2_region.npz (mt_plain_00002)"], (
                f"continued run matte or other stem misread: gaps={reader.gaps}, other_stems={reader.other_stems}")
        # another run's window of the prefix silences nothing (the D18 read): a frame no window of this piece wrote is refused beside it
        try:
            reader.at(30)
        except SystemExit as e:
            # THE CASE THAT GOES RED when another run's region file silences the refusal on every frame
            assert "no window of the piece wrote it" in str(e), f"unexpected refusal message: {e}"
        else:
            raise AssertionError("a frame no window wrote was read beside another run's window")
        # every window saved its matte and still no window wrote a frame: refused, never laid by another rule
        (folder / "mt_plain_window_2.mp4").unlink()
        try:
            tool.Matte(path).at(30)
        except SystemExit as e:
            assert "no window of the piece wrote it" in str(e), f"unexpected refusal message: {e}"
        else:
            raise AssertionError("a frame no window wrote was read")
        # Gemini's D18, finding 4: a window that saved no matte silences the refusal only on its own frames
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        vm.save_window_region(str(folder / "mt_plain_window_1_region.npz"), mask[:13], cells, src, 8, 0, 0, matte=torch.from_numpy(matte[:13]),
                              piece={"stem": "mt_plain_00001", "first": 0, "written": 13})
        vm.save_window_region(str(folder / "mt_plain_window_2_region.npz"), mask[13:], cells, src, 8, 13, 0, piece={"stem": "mt_plain_00001", "first": 13, "written": 13})
        for n_ in (1, 2):
            (folder / f"mt_plain_window_{n_}.mp4").write_bytes(b"")
        bare_reader = tool.Matte(path)
        assert bare_reader.at(15) is None, "a frame of the window with no matte was not returned as None"
        try:
            bare_reader.at(30)
        except SystemExit as e:
            # THE CASE THAT GOES RED when a window with no matte silences the refusal on every frame
            assert "no window of the piece wrote it" in str(e), f"unexpected refusal message: {e}"
        else:
            raise AssertionError("a frame no window wrote was read beside a window with no matte")
        # Gemini's D18, finding 3: a render that saved its windows' regions for this piece and no matte in any is said, not silent
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        vm.save_window_region(str(folder / "mt_plain_window_1_region.npz"), mask, cells, src, 8, 0, 0, piece={"stem": "mt_plain_00001", "first": 0, "written": 26})
        (folder / "mt_plain_window_1.mp4").write_bytes(b"")
        said_lb = tool.laid_by_of(path)
        # THE CASE THAT GOES RED when such a piece falls back to the old rules with nothing said
        assert said_lb["rule"] != "matte" and said_lb.get("windows_without_a_matte") == [1] and "no matte in any" in said_lb.get("matte_missing", ""), (
            f"a piece whose region files saved no matte in any fell back without reporting matte_missing: {said_lb.get('matte_missing')!r}")
        # THE SOFT BAND AND THE OPAQUE CORE AT THE SOURCE'S SIZE (r16n_01 / rethink.md A1): on a textured source `sharp`, a matte-laid row
        # follows `view += keep * ((up - (1 - m) * was) - m * view)`: where m = 0.25 it keeps three-quarters of the sharp source and adds a
        # quarter of the render, and where m = 1.0 it is the render alone with none of `sharp - was` embossed onto it
        sw_, sh_, cw_, ch_ = 64, 48, 32, 24
        rng = np.random.default_rng(3)
        sharp = [rng.uniform(40, 200, (sh_, sw_)).astype(np.float32), np.full((sh_ // 2, sw_ // 2), 128.0, np.float32), np.full((sh_ // 2, sw_ // 2), 128.0, np.float32)]
        canvas_src = [cv.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv.INTER_AREA) for x in sharp]
        wgt = np.zeros((ch_, cw_), np.float32)
        wgt[:, 6:15] = 1.0
        wgt[:, 17:26] = 0.25
        render = [np.full_like(x, 180.0 if q == 0 else 128.0) for q, x in enumerate(canvas_src)]
        piece_p = [c + (r - c) * (wgt if c.shape == wgt.shape else cv.resize(wgt, (c.shape[1], c.shape[0]), interpolation=cv.INTER_AREA))
                   for c, r in zip(canvas_src, render)]
        owner_ = np.where(wgt >= 1.0, 0, -1).astype(np.int8)
        got_ = np.frombuffer(tool.laid_over(sharp, [piece_p], owner_, (sw_, sh_, 0, 0), [None], None, exact=[True], orig=canvas_src, mats=[wgt]), np.uint8)
        got_y = got_[:sw_ * sh_].reshape(sh_, sw_).astype(np.float32)
        core_band = (slice(4, 44), slice(16, 26))
        soft_band = (slice(4, 44), slice(38, 48))
        want_soft = 0.75 * sharp[0] + 0.25 * 180.0
        # THE CASE THAT GOES RED when `sharp - was` is embossed onto `m == 1` (`view += (up - was) * wgt`), or the soft band loses `sharp`
        assert float(np.abs(got_y - 180.0)[core_band].max()) <= 1.0, f"opaque core embossed with sharp - was: max {float(np.abs(got_y - 180.0)[core_band].max()):.2f}"
        assert float(np.abs(got_y - want_soft)[soft_band].mean()) < 1.5, f"the soft band at the source's size: {float(np.abs(got_y - want_soft)[soft_band].mean()):.2f}"
        # ONE COMPOSITING LAW PROPERTY TEST (r16n_01 / C38): 12 random 1-, 2-, and 3-piece matte stacks (odd pixel coordinates, soft feathers)
        # across both sizes (`canvas` and `source`) and all three planes (`Y`, `U`, `V`), graded at the WORST PIXEL (`max`) in each weight regime
        # (`0`, `between`, `1`), plus a piece with no matte laid by its fallback (`changed`) alone and in the stack
        prng = np.random.default_rng(17)
        pw_, ph_, psw, psh = 64, 48, 128, 96
        pyy, pxx = np.mgrid[0:psh, 0:psw].astype(np.float32)
        psharp = [
            (110.0 + 30.0 * np.sin(pxx / 7.0) * np.cos(pyy / 5.0) + 18.0 * ((-1.0) ** (pxx + pyy))).astype(np.float32),
            (128.0 + 25.0 * np.cos(pxx[::2, ::2] / 6.0) + 10.0 * ((-1.0) ** (pxx[::2, ::2] + pyy[::2, ::2]))).astype(np.float32),
            (128.0 - 25.0 * np.sin(pyy[::2, ::2] / 6.0) + 10.0 * ((-1.0) ** (pxx[::2, ::2] + pyy[::2, ::2]))).astype(np.float32),
        ]
        porig = [cv.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv.INTER_AREA) for p in psharp]
        worst_law = {sz: {pl: {"0": 0.0, "between": 0.0, "1": 0.0} for pl in ("Y", "U", "V")} for sz in ("canvas", "source")}
        for trial in range(12):
            n_p = (trial % 3) + 1
            t_mats, t_pieces, t_renders = [], [], []
            for _i in range(n_p):
                rx0, ry0 = int(prng.integers(6, 20)), int(prng.integers(6, 14))
                rx1, ry1 = int(prng.integers(rx0 + 22, 58)), int(prng.integers(ry0 + 18, 42))
                raw_m = np.zeros((ph_, pw_), np.float32)
                raw_m[ry0:ry1, rx0:rx1] = 1.0
                tm = np.round(np.clip((cv.GaussianBlur(raw_m, (0, 0), 3.0) - 0.15) / 0.70, 0.0, 1.0) * 255.0) / 255.0
                vals = (float(prng.uniform(45.0, 215.0)), float(prng.uniform(75.0, 185.0)), float(prng.uniform(75.0, 185.0)))
                tm_c = cv.resize(tm, (pw_ // 2, ph_ // 2), interpolation=cv.INTER_AREA)
                t_pieces.append([tm * vals[0] + (1.0 - tm) * porig[0], tm_c * vals[1] + (1.0 - tm_c) * porig[1], tm_c * vals[2] + (1.0 - tm_c) * porig[2]])
                t_mats.append(tm)
                t_renders.append(vals)
            t_owner, _ = tool.owners([tm == 1.0 for tm in t_mats], [None] * n_p)
            got_c = tool.planes(tool.merged(t_pieces, t_owner, [None] * n_p, porig, exact=[True] * n_p, mats=t_mats), pw_, ph_)
            want_c, z_c, o_c = [p.copy() for p in porig], [np.ones(p.shape, bool) for p in porig], [np.zeros(p.shape, bool) for p in porig]
            for tm, vals in zip(t_mats, t_renders):
                tm_c = cv.resize(tm, (pw_ // 2, ph_ // 2), interpolation=cv.INTER_AREA)
                for k_, (mp, vk) in enumerate(((tm, vals[0]), (tm_c, vals[1]), (tm_c, vals[2]))):
                    want_c[k_] = (1.0 - mp) * want_c[k_] + mp * vk
                    z_c[k_] &= (mp == 0)
                    o_c[k_] = np.where(mp > 0, mp == 1, o_c[k_])
            for k_, pl in enumerate(("Y", "U", "V")):
                err = np.abs(got_c[k_] - np.clip(np.round(want_c[k_]), 0, 255))
                worst_law["canvas"][pl]["0"] = max(worst_law["canvas"][pl]["0"], float(err[z_c[k_]].max()))
                worst_law["canvas"][pl]["between"] = max(worst_law["canvas"][pl]["between"], float(err[~z_c[k_] & ~o_c[k_]].max()))
                worst_law["canvas"][pl]["1"] = max(worst_law["canvas"][pl]["1"], float(err[o_c[k_]].max()))
            got_s = tool.planes(tool.laid_over(psharp, t_pieces, t_owner, (psw, psh, 0, 0), [None] * n_p, None, exact=[True] * n_p, orig=porig, mats=t_mats), psw, psh)
            want_s, z_s, o_s = [p.copy() for p in psharp], [np.ones(p.shape, bool) for p in psharp], [np.zeros(p.shape, bool) for p in psharp]
            for tm, vals in zip(t_mats, t_renders):
                tm_c = cv.resize(tm, (pw_ // 2, ph_ // 2), interpolation=cv.INTER_AREA)
                lin_y = cv.resize(tm, (psw, psh), interpolation=cv.INTER_LINEAR)
                lin_c = cv.resize(tm_c, (psw // 2, psh // 2), interpolation=cv.INTER_LINEAR)
                my = np.where(lin_y > 0, np.clip(cv.resize(tm, (psw, psh), interpolation=cv.INTER_CUBIC), 0.0, 1.0), 0.0)
                mc = np.where(lin_c > 0, np.clip(cv.resize(tm_c, (psw // 2, psh // 2), interpolation=cv.INTER_CUBIC), 0.0, 1.0), 0.0)
                cy = cv.resize(cv.erode((tm == 1).astype(np.uint8), np.ones((5, 5), np.uint8)), (psw, psh), interpolation=cv.INTER_NEAREST) > 0
                cc = cv.resize(cv.erode((tm_c == 1).astype(np.uint8), np.ones((5, 5), np.uint8)), (psw // 2, psh // 2), interpolation=cv.INTER_NEAREST) > 0
                for k_, (mp, vk, ck) in enumerate(((my, vals[0], cy), (mc, vals[1], cc), (mc, vals[2], cc))):
                    want_s[k_] = (1.0 - mp) * want_s[k_] + mp * vk
                    z_s[k_] &= (mp == 0)
                    o_s[k_] = np.where(mp > 0, ck, o_s[k_])
            for k_, pl in enumerate(("Y", "U", "V")):
                err = np.abs(got_s[k_] - np.clip(np.round(want_s[k_]), 0, 255))
                worst_law["source"][pl]["0"] = max(worst_law["source"][pl]["0"], float(err[z_s[k_]].max()))
                worst_law["source"][pl]["between"] = max(worst_law["source"][pl]["between"], float(err[~z_s[k_] & ~o_s[k_]].max()))
                if o_s[k_].any():
                    worst_law["source"][pl]["1"] = max(worst_law["source"][pl]["1"], float(err[o_s[k_]].max()))
        for pl in ("Y", "U", "V"):
            assert worst_law["canvas"][pl] == {"0": 0.0, "between": 0.0, "1": 0.0}, (
                f"property test failed on canvas plane {pl}: {worst_law['canvas'][pl]}")
            assert worst_law["source"][pl]["0"] == 0.0 and worst_law["source"][pl]["1"] == 0.0 and worst_law["source"][pl]["between"] <= 2.0, (
                f"property test failed on source plane {pl}: {worst_law['source'][pl]}")
        # A piece with no matte (`exact=False`, `mats[i]=None`, e.g. 1039_g before its matte is written) is laid by its fallback (`changed`)
        # both as an earlier row and as a later row in a stack with a matte piece, at both sizes across all planes
        fb_piece = [p.copy() for p in porig]
        fb_piece[0][4:40, 2:28], fb_piece[1][2:20, 1:14], fb_piece[2][2:20, 1:14] = 210.0, 170.0, 90.0
        fb_hard = tool.changed(fb_piece, porig, tool.difference(fb_piece, porig))
        m_right_raw = np.zeros((ph_, pw_), np.float32)
        m_right_raw[8:40, 36:58] = 1.0
        m_right = np.round(np.clip((cv.GaussianBlur(m_right_raw, (0, 0), 3.0) - 0.15) / 0.70, 0.0, 1.0) * 255.0) / 255.0
        m_right_c = cv.resize(m_right, (pw_ // 2, ph_ // 2), interpolation=cv.INTER_AREA)
        p_right = [m_right * 60.0 + (1.0 - m_right) * porig[0], m_right_c * 100.0 + (1.0 - m_right_c) * porig[1], m_right_c * 150.0 + (1.0 - m_right_c) * porig[2]]
        for pcs, exs, mts, hards in (([fb_piece, p_right], [False, True], [None, m_right], [fb_hard, m_right == 1.0]),
                                     ([p_right, fb_piece], [True, False], [m_right, None], [m_right == 1.0, fb_hard])):
            fb_owner, _ = tool.owners(hards, [None, None])
            fb_c = tool.planes(tool.merged(pcs, fb_owner, [None, None], porig, exact=exs, mats=mts), pw_, ph_)
            fb_s = tool.planes(tool.laid_over(psharp, pcs, fb_owner, (psw, psh, 0, 0), [None, None], None, exact=exs, orig=porig, mats=mts), psw, psh)
            fb_err_c = [float(np.abs(fb_c[0][8:36, 6:24] - 210.0).max()), float(np.abs(fb_c[1][4:18, 3:12] - 170.0).max()), float(np.abs(fb_c[2][4:18, 3:12] - 90.0).max())]
            fb_err_s = [float(np.abs(fb_s[0][16:72, 12:48] - 210.0).max()), float(np.abs(fb_s[1][8:36, 6:24] - 170.0).max()), float(np.abs(fb_s[2][8:36, 6:24] - 90.0).max())]
            assert fb_err_c == [0.0, 0.0, 0.0] and fb_err_s == [0.0, 0.0, 0.0], (
                f"fallback piece with no matte not laid across Y/U/V: canvas {fb_err_c}, source {fb_err_s}")
        law_fig = (f"canvas Y/U/V 0={worst_law['canvas']['Y']['0']:.1f}/{worst_law['canvas']['U']['0']:.1f}/{worst_law['canvas']['V']['0']:.1f} "
                   f"between={worst_law['canvas']['Y']['between']:.1f}/{worst_law['canvas']['U']['between']:.1f}/{worst_law['canvas']['V']['between']:.1f} "
                   f"1={worst_law['canvas']['Y']['1']:.1f}/{worst_law['canvas']['U']['1']:.1f}/{worst_law['canvas']['V']['1']:.1f}; "
                   f"source Y/U/V 0={worst_law['source']['Y']['0']:.1f}/{worst_law['source']['U']['0']:.1f}/{worst_law['source']['V']['0']:.1f} "
                   f"between={worst_law['source']['Y']['between']:.1f}/{worst_law['source']['U']['between']:.1f}/{worst_law['source']['V']['between']:.1f} "
                   f"1={worst_law['source']['Y']['1']:.1f}/{worst_law['source']['U']['1']:.1f}/{worst_law['source']['V']['1']:.1f}")
        for tag in ("plain", "zoomed", "left", "right", "under", "over", "third"):
            shutil.rmtree(TMP / f"mt_{tag}_windows", ignore_errors=True)
        return ("laid by the node's own matte, unzoomed and zoomed 1.333x off the grid, at the source's size and the canvas's: inside the matte "
                "within " + ", ".join(said) + " levels of the node's composite (four levels off the source), the source outside it; two touching "
                "pieces and two or three overlapping ones laid as the later render over the earlier piece by its own matte; a window with no matte is flagged by its frames; a continued run's matte read from its written frames, another run's file not read; "
                f"the soft band and opaque core at the source's size keep the sharp source in the band and emboss none of it on the core; property test worst pixel ({law_fig}), and a piece with no matte laid by its fallback (canvas/source Y/U/V 0.0/0.0/0.0)")

    case("the cut rows carry the laid edge where a region reaches it", the_cut_rows_carry_the_edge)
    case("a piece that changed nothing is the original", nothing_changed)
    case("a painted rectangle is found where it is", rectangle_found)
    case("a delivery of pieces and original ranges passes its own check", delivery_passes)
    case("controls: three files that must fail", controls_fail)
    case("a span from the middle begins its track at the span", middle_span)
    case("a PCM track is held to its decoded samples", pcm_by_its_samples)
    case("two pieces on the same frames", two_pieces)
    case("the flags", flags)
    case("at the source's size", at_the_sources_size)
    case("a class given back to the source", a_class_given_back)
    case("another subject given back whole", a_subject_given_back)
    case("a piece that spills over a cut", spill_over_a_cut)
    case("whose a pixel is, from the owner map", by_the_owner_map)
    case("what the record says the file did to a subject", the_subjects_table)
    case("two people's tracks, and a person's under a keep-out", two_people_and_a_keep_out)
    case("a patch as a later row", a_patch_as_a_later_row)
    case("a patch as a later row, with a capture", a_patch_with_a_capture)
    case("a locked file is not built over", a_locked_file)
    case("a shot where a named subject is the source's own", shots_left_as_the_source)
    case("a frame its run carried no mask on", no_mask_no_lay)
    case("a row that names its capture's run", a_run_named_on_the_row)
    case("a row that ends or begins on the source's own picture", ends_on_the_source)
    case("a piece rendered zoomed in", a_zoomed_piece)
    case("a whole-region piece laid by its region", laid_by_its_region)
    case("a piece laid by the node's own matte", laid_by_the_matte)
    case("a thing worn on the head, placed on the piece's own face", a_worn_thing_on_its_face)
sys.exit(finish())
