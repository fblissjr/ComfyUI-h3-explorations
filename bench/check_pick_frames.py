#!/usr/bin/env python3
"""The frame picker (`bench/_lib/pick_frames.py`) and its command (`bench/pick_frames.py`), on spans made for the purpose.

Evenly spaced frames hid 5915-5918 of second_hfp_5889e (2026-10-21), where the sapiens face count fell at 5916 and
she leaves between 5917 and 5918 (the owner). Each case below is one way a picker can hide a change again.

Claims, i.e. what breaks if a case is deleted:

  between_the_even_picks      a series that falls to zero at frame 27 of a 46-frame span (5916 in 5889-5934's numbers), which a
                              4-frame evenly spaced sheet misses and a 6-frame one shows with neither neighbour (the
                              control asserts both), is picked at 26, 27 and 28, the reason at 27 saying which series
                              went and from what to what
  a_label_changes             a categorical series (a mouth mode) changing is picked at that frame, from and to; two
                              in one second: one picked and the other listed, both picked at `per_second=2`
  selective                   flicker under `LEVEL_SHARE` of the subject's own median is no change; a one-frame dip
                              is no step (`HOLD`); a weaker change in a second that has one is listed under
                              `also_changed_not_picked` with its lower score and not picked
  an_island                   a region on one frame with none on both sides (4647d's 4706 in form) is picked as an island
  a_place_jumps               a place that moves more than `JUMP_SHARE` of its size in a frame is picked; drift is not
  floor_first_last_and_cap    first and last are always picked and each aligned `FLOOR`-frame second holds a picked
                              frame, under a cap too: on a 362-frame span under `SHEET_CAP`, quiet, with one change, and
                              with two flags, the floor alone passes the cap, so the change keeps its frame and both
                              flanks, nothing is dropped and the output says by how much it is over; on a 100-frame span
                              there is room; a cap that can be met takes the weakest changes' flanks first; a frame two
                              changes put there keeps both reasons; two identical flags give one reason, and two
                              identical candidate flags are picked together; a weaker change that joined a flank frame
                              the cap takes is named in `dropped` and the cap text
  flanks_and_names            a change that lost its second's room to a stronger one but falls on a flag's frame keeps
                              the frames either side; a weaker change on a frame listed only as a flank joins it as a
                              reason, not "not picked"; a count that falls under the level floor is named so, and
                              "gone" only where it is zero; an island is named from both sides; a peak reads "peak V
                              over line L"
  refusals                    bad input to the command is refused in its own words with no traceback: a reversed or
                              malformed span, a malformed size, a cap or a room a second under 1, a missing capture folder, video, render, clip, points,
                              cast or laid file, a bad @FIRST, an unreadable cast file; a change clip with no window
                              to cut is refused and no file is written
  the_picture_peaks           a lossless clip with a hard change at one frame: the command's `--video` picks it, and
                              the reasons say `picture_whole peak`
  a_capture_folder            the command on a capture folder made for the purpose: its counts, a likely-to-fail flag,
                              statements, the owner's cast box (on a frame where the track is empty) and the laid edges
                              are each a reason; an iffy flag's edges in a second a stronger change holds are listed
                              under `also_changed_not_picked` with `SPAN_SCORE`, not picked; each frame carries source_frame, clip_frame, clip_seconds; a file that is not
                              there is named in `absent`
  the_change_clip             `--change-clip` holds each change window, a frame a second, with a card before each

    <comfy venv python> bench/check_pick_frames.py [--mutate linspace|unselective]

`--mutate linspace` swaps the picker for six evenly spaced frames (what the sheet writers used before 2026-10-21) and
runs every case against it: the cases that defend against that must go red. `--mutate unselective` takes away the
level floor, the hold and the room a second (the picker as first built, which picked 45 of 5889e's 46 frames): the
cases that defend selectivity must go red. Needs ffmpeg and ffprobe for the last
three cases; no model, no card, no server.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, needs, refused  # noqa: E402
from _lib import pick_frames as pf  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

import pick_frames as cli  # noqa: E402
from _lib import frames as fr  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="check_pick_frames_"))
FIRST, N = 5889, 46


def evenly(first: int, last: int, series=None, spans=None, events=None, *, floor: int = pf.FLOOR, cap=None, per_second=1) -> dict:
    """The mutation: six evenly spaced frames, what the sheet writers picked before the shared picker."""
    frames = sorted(set(np.linspace(first, last, 6).astype(int).tolist()))
    return {"span": [first, last], "changes": [], "also_changed_not_picked": [], "dropped": [], "series": {}, "over_the_cap": False, "rules": {},
            "frames": [{"source_frame": f, "clip_frame": f - first, "clip_seconds": round((f - first) / 24, 3),
                        "reasons": [{"role": "floor", "short": "evenly spaced"}]} for f in frames]}


def run_cli(args: list[str]) -> None:
    """The command in this process; a refusal is a failed case here, not the end of the check."""
    try:
        cli.main(args)
    except SystemExit as e:
        raise AssertionError(f"refused: {e}") from None


def picked(result: dict) -> dict[int, list[dict]]:
    return {p["source_frame"]: p["reasons"] for p in result["frames"]}


def between_the_even_picks() -> str:
    gone = FIRST + 27
    # the control: a 4-frame even sheet (5889, 5904, 5919, 5934: the one that missed it) has none of 26-28; a 6-frame one
    # has 27 alone, and neither frame either side, so it cannot show the face going
    for k, has in ((4, set()), (6, {gone})):
        even = set(np.linspace(FIRST, FIRST + N - 1, k).astype(int).tolist())
        assert {gone - 1, gone, gone + 1} & even == has, (k, sorted(even))
    face = [3000.0] * 27 + [0.0] * (N - 27)
    got = picked(pf.pick(FIRST, FIRST + N - 1, {"sapiens_face_px": face}))
    assert {gone - 1, gone, gone + 1} <= set(got), sorted(got)
    at = [r for r in got[gone] if r["role"] == "at"]
    assert at and at[0]["change"]["by"] == "sapiens_face_px" and at[0]["change"]["kind"] == "gone" \
        and at[0]["change"]["from"] == 3000 and at[0]["change"]["to"] == 0, got[gone]
    assert any(r["role"] == "before" for r in got[gone - 1]) and any(r["role"] == "after" for r in got[gone + 1])
    return f"a count gone at {gone}: {gone - 1}, {gone}, {gone + 1} picked; a 4-frame even sheet has none of them, a 6-frame one {gone} alone"


def a_label_changes() -> str:
    mode = ["closed_lip_line"] * 17 + ["lip_lines"] * 4 + ["closed_lip_line"] * (N - 21)
    both = pf.pick(FIRST, FIRST + N - 1, {"mouth_mode": mode}, per_second=2)
    got = picked(both)
    for f, a, b in ((FIRST + 17, "closed_lip_line", "lip_lines"), (FIRST + 21, "lip_lines", "closed_lip_line")):
        assert f in got and any(r["role"] == "at" and r["change"]["from"] == a and r["change"]["to"] == b for r in got[f]), (f, got.get(f))
    one = pf.pick(FIRST, FIRST + N - 1, {"mouth_mode": mode})       # both in the first second: one picked, one listed
    at = {c["frame"] for c in one["changes"]} | {c["frame"] for c in one["also_changed_not_picked"]}
    assert at == {FIRST + 17, FIRST + 21} and len(one["also_changed_not_picked"]) == 1, (one["changes"], one["also_changed_not_picked"])
    return "the mouth mode's two changes in one second: one picked and one listed by default, both picked at two a second"


def an_island() -> str:
    region = [17664] * 56 + [0, 0, 0, 15104, 0, 0] + [17000] * 17
    first = 4647
    got = picked(pf.pick(first, first + len(region) - 1, {"region_px": region}))
    assert 4706 in got and any(r["role"] == "at" and r["change"]["kind"] == "island of 1" for r in got[4706]), got.get(4706)
    return "a region on 4706 alone, nothing on 4703-4705 and 4707: picked as an island of 1"


def selective() -> str:
    """What makes the list a choice: flicker under the subject's own level is no change; a one-frame blip is no step;
    a weaker change in a second that already has one is listed with its score, not picked."""
    face = [8000.0] * 20 + [0.0, 34.0, 0.0, 90.0, 180.0, 0.0, 2.0] + [0.0] * (N - 27)
    got = pf.pick(FIRST, FIRST + N - 1, {"sapiens_face_px": face})
    read = [(c["frame"], c["kind"]) for c in got["changes"] + got["also_changed_not_picked"]]
    assert read == [(FIRST + 20, "gone")], read
    blip = [5000.0] * 10 + [2000.0] + [5000.0] * 9 + [2000.0] * (N - 20)
    got = pf.pick(FIRST, FIRST + N - 1, {"track_px": blip})
    assert [(c["frame"], c["kind"]) for c in got["changes"]] == [(FIRST + 20, "step")], got["changes"]
    two = [5000.0] * 6 + [2000.0] * 6 + [0.0] * (N - 12)          # a step at 6 and the track gone at 12, one second
    got = pf.pick(FIRST, FIRST + N - 1, {"track_px": two})
    assert [(c["frame"], c["kind"]) for c in got["changes"]] == [(FIRST + 12, "gone")], got["changes"]
    left = got["also_changed_not_picked"]
    assert [(c["frame"], c["kind"]) for c in left] == [(FIRST + 6, "step")] and 0 < left[0]["score"] < got["changes"][0]["score"], left
    assert FIRST + 6 not in picked(got)
    return ("flicker of 2-180 px after 8000 is no change; a one-frame dip is no step; of a step and a disappearance in one "
            "second the disappearance is picked and the step listed with its lower score")


def a_place_jumps() -> str:
    place = [(500.0 + 2 * i, 300.0, 40.0) for i in range(30)] + [(700.0, 300.0, 40.0)] * 16
    got = pf.pick(FIRST, FIRST + N - 1, {"head_track_place": place})
    jumps = [c["frame"] for c in got["changes"] if c["kind"] == "jumps"]
    assert jumps == [FIRST + 30], got["changes"]
    return "a head track that drifts 2 px a frame is not picked; one that moves 142 px at once is, at that frame"


def every_second_held(result: dict) -> bool:
    """Whether each `FLOOR`-frame second of the span holds a picked frame."""
    first, last = result["span"]
    kept = {p["source_frame"] for p in result["frames"]}
    return all(any(first + k * pf.FLOOR <= f <= min(first + (k + 1) * pf.FLOOR - 1, last) for f in kept)
               for k in range((last - first) // pf.FLOOR + 1))


def floor_first_last_and_cap() -> str:
    quiet = pf.pick(0, 99)
    frames = [p["source_frame"] for p in quiet["frames"]]
    assert frames[0] == 0 and frames[-1] == 99 and every_second_held(quiet) and len(frames) == 5, frames
    two = picked(pf.pick(0, 99, {"a": [1] * 50 + [0] * 50, "b": ["x"] * 50 + ["y"] * 50}))
    assert {r["change"]["by"] for r in two[50] if r["role"] == "at"} == {"a", "b"}, two[50]
    many = pf.pick(0, 99, {"a": [float(i % 2) * 100 for i in range(100)]}, cap=10)   # an island on every frame
    assert len(many["frames"]) == 100 and many["over_the_cap"], len(many["frames"])
    # long spans under the sheet writers' cap: a change's own frame, the written-down frames and a frame in each second stay
    quiet = pf.sheet_picks(0, 362)
    assert every_second_held(quiet) and quiet["over_the_cap"] and "over the cap by" in quiet["cap"], quiet["cap"]
    face = {"a": [5000.0] * 200 + [0.0] * 162}
    flags = [{"name": "flag x", "frames": [50, 60]}, {"name": "flag y", "frames": [300, 310]}]
    for long in (pf.sheet_picks(0, 362, face), pf.sheet_picks(0, 362, face, flags)):
        # the floor alone passes the cap, so dropping the change's flanks could not reach it: they stay, and the overrun is said
        kept = {p["source_frame"] for p in long["frames"]}
        assert {199, 200, 201} <= kept and every_second_held(long) and not long["dropped"], (sorted(kept), long["dropped"])
        assert long["over_the_cap"] and f"over the cap by {len(kept) - pf.SHEET_CAP}" in long["cap"], long["cap"]
    assert {50, 61, 300, 311} <= kept, sorted(kept)
    roomy = pf.sheet_picks(0, 100, {"a": [5000.0] * 50 + [0.0] * 50})
    assert {49, 50, 51} <= {p["source_frame"] for p in roomy["frames"]} and roomy["cap"].endswith("within it"), roomy
    busy = {"a": [0] * 30 + [5] * 30 + [0] * 140, "b": [1.0] * 100 + [3.0] * 100, "c": [9000] * 150 + [0] + [9000] * 49}
    capped = pf.pick(0, 199, busy, spans=[{"name": "flag sure", "frames": [120, 130]}], cap=12)
    kept = {p["source_frame"] for p in capped["frames"]}
    assert {c["frame"] for c in capped["changes"]} | {0, 199} <= kept and every_second_held(capped) and len(kept) <= 12, sorted(kept)
    # weakest first (the step at 100, then of the equal changes of state the latest): 100, 150, 60 lose their flanks; 30 keeps them
    assert [d["frame"] for d in capped["dropped"]] == [60, 100, 150] and {29, 31} <= kept \
        and not any(f in kept for d in capped["dropped"] for f in d["frames_dropped"]), capped["dropped"]
    same = pf.pick(0, 99, spans=[{"name": "flag x", "frames": [40, 50]}, {"name": "flag x", "frames": [40, 50]}])
    assert [r["short"] for r in picked(same)[40]] == ["flag x starts"], picked(same)[40]
    twins = pf.pick(0, 99, {"a": [5000.0] * 41 + [0.0] * 59}, spans=[{"name": "flag iffy", "frames": [40, 50], "record": False}] * 2)
    listed = [(c["frame"], c["by"]) for c in twins["also_changed_not_picked"]]
    assert (40, "flag iffy") in [(c["frame"], c["by"]) for c in twins["changes"]] and (40, "flag iffy") not in listed \
        and not twins["dropped"], (twins["changes"], listed, twins["dropped"])
    # a weaker change that joined a flank frame the cap takes is named with that frame, not moved to "not picked" in silence
    joins = pf.pick(0, 99, {"a": [5000.0] * 50 + [0.0] * 50, "b": ["x"] * 51 + ["y"] * 49}, cap=5)
    kept = {p["source_frame"] for p in joins["frames"]}
    lost = [(d["frame"], d.get("its_frame_dropped")) for d in joins["dropped"] if "its_frame_dropped" in d]
    assert 51 not in kept and lost == [(51, 51)] and "went with them" in joins["cap"] and not any(c["by"] == "b" for c in joins["also_changed_not_picked"]), \
        (sorted(kept), joins["dropped"], joins["cap"])
    return ("first, last and a frame in each second; both reasons kept; on a long span the floor alone passes the sheet's cap, so the "
            "change keeps its flanks and the overrun is said; a cap that can be met takes the weakest changes' flanks first; twin "
            "flags are picked together; a join the cap takes is named")


def flanks_and_names() -> str:
    face = [8000.0] * 20 + [500.0] * 10 + [0.0] * (N - 30)
    strong = [9000.0] * 10 + [0.0] * (N - 10)                   # a stronger change earlier in the same second
    mode = ["x"] * 21 + ["y"] * (N - 21)                        # a weaker change on 21, a flank of 20
    got = pf.pick(FIRST, FIRST + N - 1, {"sapiens_face_px": face, "track_px": strong, "mode": mode},
                  spans=[{"name": "flag sure", "frames": [FIRST + 20, FIRST + 25]}])
    at = picked(got)
    # 20 lost its second's room to 10, but joins the flag's frame and keeps its flanks
    assert {FIRST + 9, FIRST + 10, FIRST + 11, FIRST + 19, FIRST + 20, FIRST + 21} <= set(at), sorted(at)
    assert any(r["role"] == "at" and r["change"]["by"] == "mode" for r in at[FIRST + 21]), at[FIRST + 21]   # joins the flank frame
    assert not any(c["by"] == "mode" for c in got["also_changed_not_picked"]), got["also_changed_not_picked"]
    kinds = {(c["frame"], c["kind"]) for c in got["changes"]}
    assert (FIRST + 20, "falls under the level floor") in kinds and (FIRST + 30, "gone") not in kinds, kinds  # 500 to 0: both nothing
    gone = pf.pick(FIRST, FIRST + N - 1, {"sapiens_face_px": [8000.0] * 20 + [0.0] * (N - 20)})
    assert [(c["frame"], c["kind"]) for c in gone["changes"]] == [(FIRST + 20, "gone")], gone["changes"]
    island = picked(pf.pick(0, 40, {"region_px": [0] * 20 + [15104] + [1] + [0] * 19}))
    assert "region_px island of 1 0>15104>1" in [r["short"] for r in island[20]], island[20]
    peak = pf.pick(0, 40, {"picture_whole": {"values": [None] + [1.0] * 19 + [134.886] + [1.0] * 20, "rules": ("peak",)}})
    label = [r["short"] for r in picked(peak)[20] if r["role"] == "at"]
    assert label == ["picture_whole peak 134.9 over line 3"], label
    return ("a change that lost its second's room keeps its flanks on a flag's frame; a weaker change on a flank frame joins it; "
            "8000 to 500 falls under the level floor, 8000 to 0 is gone; an island named from both sides; a peak's label")


def refusals() -> str:
    cap = capture_folder()
    video = clip(WORK / "refuse.mkv", moving(8))
    me = [sys.executable, str(Path(cli.__file__))]
    bad_cast = WORK / "bad_cast.jsonl"
    bad_cast.write_text("not json\n")
    said = []
    for args, words in ((["--span", "10-5"], "comes before its first"), (["--span", "abc"], "is not A-B"),
                        (["--span", "0-7", "--cap", "0"], "1 or more"), (["--span", "0-7", "--per-second", "0"], "1 or more"),
                        (["--span", "0-7", "--size", "1024"], "is not WxH"), (["--capture", str(WORK / "nowhere")], "no such folder"),
                        (["--span", "0-7", "--video", str(WORK / "none.mkv")], "no such file"),
                        (["--span", "0-7", "--video", f"{video}@x"], "a whole number"),
                        (["--span", "0-7", "--video", str(video), "--render", str(WORK / "none.mp4")], "no such file"),
                        (["--span", "0-7", "--points", str(WORK / "none.npz")], "no such file"),
                        (["--span", "0-7", "--laid", str(WORK / "none.jsonl"), "--load", "x"], "no such file"),
                        (["--span", "0-7", "--cast", str(bad_cast), "--subject", "second"], "cannot be read"),
                        (["--capture", str(cap), "--change-clip", str(WORK / "x.mp4"), "--change-clip-from", str(WORK / "none.mp4")], "no such file")):
        said.append(refused(me + args, words))
    empty = WORK / "empty.mp4"
    said.append(refused(me + ["--span", "0-7", "--change-clip", str(empty), "--change-clip-from", str(video)], "no window to cut"))
    assert not empty.exists(), "an empty change clip was written"
    return f"{len(said)} bad inputs refused in the tool's own words; an empty change clip refused and not written"


def clip(path: Path, frames: np.ndarray) -> Path:
    n, h, w = frames.shape
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24",
                            "-i", "-", "-c:v", "ffv1", "-level", "3", str(path)], stdin=subprocess.PIPE)
    enc.communicate(frames.tobytes())
    assert enc.returncode == 0
    return path


def moving(n: int = N, cut: int = 20) -> np.ndarray:
    """A block that moves a pixel a frame on grey, and the whole picture changing at frame `cut`."""
    out = np.full((n, 96, 128), 60, np.uint8)
    for k in range(n):
        out[k, 30:50, 10 + k:30 + k] = 200
    out[cut:] = 255 - out[cut:]
    return out


def the_picture_peaks() -> str:
    video = clip(WORK / "cut.mkv", moving())
    out = WORK / "peak.json"
    run_cli(["--span", f"{FIRST}-{FIRST + N - 1}", "--video", f"{video}@{FIRST}", "--size", "128x96", "--out", str(out)])
    got = picked(json.loads(out.read_text()))
    at = FIRST + 20
    assert at in got and any(r["role"] == "at" and r["change"]["by"] == "picture_whole" and r["change"]["kind"] == "peak" for r in got[at]), got.get(at)
    return f"the hard change at {at} picked by the whole picture's peak"


def capture_folder() -> Path:
    cap = WORK / "2026-10-21_second_x"
    (cap / "subjects" / "second").mkdir(parents=True, exist_ok=True)
    rows = [{"source_frame": FIRST + i, "source_track_px": 0 if i < 9 else 5000, "sapiens_face_px": 800, "mouth_mode": "closed_lip_line",
             "face_outside_region_px": 12 if i == 30 else 0} for i in range(N)]
    (cap / "facts_pass.json").write_text(json.dumps({"what_measured": {"subject": "second", "run": "second_x", "window_source_frames": [FIRST, FIRST + N - 1]},
                                                     "frames": rows}))
    (cap / "flags.json").write_text(json.dumps({"span": [FIRST, FIRST + N - 1], "blocking": [], "flags": [
        {"rule": "track_empty", "subject": "second", "level": "likely to fail", "id": "f1", "source_frames": [[FIRST + 36, FIRST + 37]]},
        {"rule": "part_not_visible", "subject": "second", "level": "iffy", "id": "f2", "source_frames": [[FIRST + 14, FIRST + 15]]},
        {"rule": "someone_elses", "subject": "third", "source_frames": [[FIRST + 12, FIRST + 12]]}]}))
    (cap / "no_mask_statements.json").write_text(json.dumps({"statements": [{"subject": "second", "source_frames": [FIRST, FIRST + 8], "by": "a reader"}]}))
    return cap


def a_capture_folder() -> str:
    cap = capture_folder()
    cast = WORK / "cast.jsonl"
    cast.write_text(json.dumps({"schema": "h3_cast/1", "subject": "second", "frame": FIRST + 4, "answer": "here", "box": [355.0, 144.0, 41.0, 53.0],
                                "size": [1024, 768], "by": "the owner", "at": "2026-10-21T00:00:00Z", "table": "t.json"}) + "\n")
    laid = WORK / "laid.jsonl"
    laid.write_text(json.dumps({"schema": "h3_laid/1", "first": FIRST, "last": FIRST + N - 1, "subject": "second", "load": "second_x",
                                "file": "x.mp4", "sha256": "0", "owner": "unconfirmed", "by": "the check", "frames": [[FIRST, FIRST + 26]]}) + "\n")
    out = WORK / "capture.json"
    run_cli(["--capture", str(cap), "--cast", str(cast), "--laid", str(laid), "--load", "second_x", "--out", str(out)])
    res = json.loads(out.read_text())
    got = picked(res)
    want = {FIRST + 4: "owner's cast", FIRST + 9: "source_track_px", FIRST + 30: "face_outside_region_px",
            FIRST + 36: "flag track_empty", FIRST + 38: "flag track_empty", FIRST + 27: "laid second_x"}
    for f, by in want.items():
        assert f in got and any(r["role"] == "at" and r["change"]["by"].startswith(by) for r in got[f]), (f, by, got.get(f))
    assert FIRST + 12 not in got or not any(r.get("change", {}).get("by") == "flag someone_elses" for r in got[FIRST + 12]), "another subject's flag was read"
    # an iffy flag's edges (14, 16) share the first second with the track appearing (9), which outranks them: listed, not picked
    assert FIRST + 14 not in got and FIRST + 16 not in got, (got.get(FIRST + 14), got.get(FIRST + 16))
    iffy = [c for c in res["also_changed_not_picked"] if c["by"] == "flag part_not_visible"]
    assert [c["frame"] for c in iffy] == [FIRST + 14, FIRST + 16] and all(c["score"] == pf.SPAN_SCORE for c in iffy), res["also_changed_not_picked"]
    one = res["frames"][1]
    assert set(one) == {"source_frame", "clip_frame", "clip_seconds", "reasons"} and one["clip_frame"] == one["source_frame"] - FIRST \
        and one["clip_seconds"] == round(one["clip_frame"] / 24, 3), one
    assert any("frames.json" in x for x in res["absent"]) and any("head_tube" in x for x in res["absent"]) and res["cannot_see"], res["absent"]
    return ("the owner's box on a frame the track is empty, the track appearing, a face outside the region, a likely-to-fail flag's edges "
            "and the laid edge are each a reason; an iffy flag's edges are listed, not picked; another subject's flag is not read; what was "
            "not there is named")


def the_change_clip() -> str:
    video = clip(WORK / "stacked.mkv", moving())
    out = WORK / "changes.mp4"
    res_path = WORK / "clip.json"
    run_cli(["--span", f"{FIRST}-{FIRST + N - 1}", "--video", f"{video}@{FIRST}", "--size", "128x96", "--out", str(res_path),
              "--change-clip", str(out), "--change-clip-from", f"{video}@{FIRST}"])
    res = json.loads(res_path.read_text())["change_clip"]
    assert res["windows"] == [[FIRST + 19, FIRST + 21]], res["windows"]
    w, h, n = fr.probe(str(out))
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout.strip().strip(",")
    assert (w, h) == (128, 96) and n == 4 == res["seconds"] and rate == "1/1", (w, h, n, rate, res)
    levels = [float(f.mean()) for f in fr.stream(str(out), (w, h), 0, n)]
    want = [float(m.mean()) for m in moving()[19:22]]
    assert levels[0] < 40 and all(abs(a - b) < 6 for a, b in zip(levels[1:], want)), (levels, want)
    return "the change's window 19-21, after a dark card, a frame a second: four seconds, each frame its own"


MUTATE = sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None
if MUTATE == "linspace":
    print("MUTATION: the picker is six evenly spaced frames; the cases that defend against it must FAIL")
    pf.pick = evenly
elif MUTATE == "unselective":
    print("MUTATION: no level floor, no hold, no room a second; the cases that defend selectivity must FAIL")
    whole = pf.pick
    pf.LEVEL_SHARE, pf.HOLD = 1e-9, 1
    pf.pick = lambda *a, **k: whole(*a, **{**k, "per_second": 10 ** 6})
elif MUTATE:
    raise SystemExit(f"--mutate is linspace or unselective, not {MUTATE!r}")
case("between the even picks", between_the_even_picks)
case("a label changes", a_label_changes)
case("an island", an_island)
case("selective", selective)
case("a place jumps", a_place_jumps)
case("floor, first, last and the cap", floor_first_last_and_cap)
case("flanks and names", flanks_and_names)
case("refusals", refusals)
case("the picture peaks", the_picture_peaks)
case("a capture folder", a_capture_folder)
case("the change clip", the_change_clip)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
