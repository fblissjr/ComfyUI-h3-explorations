#!/usr/bin/env python3
"""`bench/propose_runs.py` on made part masks: each case is a way a proposed load could be wrong.

  the_frames_and_the_floors    px a frame, the whole mask or its largest piece; each floor said a frame; the runs cut by the first
  the_runs                     runs of the shortest load or more where the part clears the floor, joined across --bridge empty frames
                               (listed mid-run, those with no part apart), the frames under the floor at their edges listed, the rest
                               left as the original
  the_nodes_own_zoom           each run's zoom is video_mask.zoom_box itself on the run's boxes: a small part zooms, one that crosses
                               the frame is refused with the node's own words
  the_command                  on a made preview folder: the part file the preview saved, the stretch held inside the preview, a kept
                               output refused

    <comfy venv python> bench/check_propose_runs.py

Needs ffmpeg and torch (the node's module is loaded without the server). No model, no card.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
from _lib.frames import write_mask_video  # noqa: E402

import propose_runs as pr  # noqa: E402

W, H = 1024, 768


def made(n: int, sizes: list[int], speck: bool = False, at=(400, 300)) -> np.ndarray:
    """A part mask of n frames: a square of `sizes[f]` px a frame (0 for none) at `at`, and with `speck` a 4 px speck far off."""
    m = np.zeros((n, H, W), bool)
    for f, s in enumerate(sizes):
        side = int(round(s ** 0.5))
        if side:
            m[f, at[1]:at[1] + side, at[0]:at[0] + side] = True
        if speck:
            m[f, 700:702, 900:902] = True
    return m


def the_frames_and_the_floors() -> str:
    m = made(3, [900, 0, 400], speck=True)
    px, boxes = pr.part_frames(m, largest=False)
    assert list(px) == [904, 4, 404] and boxes[1].tolist() == [900, 700, 902, 702], (px, boxes)
    px, boxes = pr.part_frames(m, largest=True)
    # THE CASE THAT GOES RED when a speck counts toward the part read as its largest piece, or makes its box
    assert list(px) == [900, 4, 400] and boxes[0].tolist() == [400, 300, 430, 330], (px, boxes)
    out = pr.propose(np.array([900, 4, 400]), boxes, 100, [480, 4096], 1, 1, (W, H), pr.node_zoom().zoom_box)
    assert [(f["frame"], f["clears_480"], f["clears_4096"]) for f in out["frames"]] == [(100, True, False), (101, False, False), (102, False, False)], out["frames"]
    return "px a frame of the whole mask or its largest piece, its box from that piece; each floor said a frame"


def the_runs() -> str:
    ok = np.array([0] + [1] * 14 + [0] + [1] * 5 + [0, 0] + [1] * 3 + [0] + [1] * 20, bool)
    # THE CASE THAT GOES RED when a run is not joined across one empty frame, is joined across two, or a short run is kept
    assert pr.runs_of(ok, 1, 12) == [(1, 20), (23, 46)], pr.runs_of(ok, 1, 12)
    assert pr.runs_of(ok, 0, 12) == [(1, 14), (27, 46)], pr.runs_of(ok, 0, 12)
    sizes = [100] + [2000] * 14 + [0] + [2000] * 5 + [100, 100] + [2000] * 3 + [100] + [2000] * 20
    m = made(len(sizes), sizes)
    px, boxes = pr.part_frames(m, largest=True)
    out = pr.propose(px, boxes, 100, [480], 1, 12, (W, H), pr.node_zoom().zoom_box)
    r1, r2 = out["runs"]
    assert r1["frames"] == [101, 120] and r1["mid_run_under_the_floor"] == [115] and r1["no_part_mid_run"] == [115], r1
    assert r1["edge_frames_under_the_floor_next_to_it"] == [100, 121, 122], r1["edge_frames_under_the_floor_next_to_it"]
    assert r2["frames"] == [123, 146] and r2["mid_run_under_the_floor"] == [126], r2
    assert out["left_as_the_original"] == [[100, 100], [121, 122]], out["left_as_the_original"]
    return "runs of the shortest load or more joined across --bridge empty frames, listed mid-run and with no part apart; the edges; the rest the original"


def the_nodes_own_zoom() -> str:
    zoom_box = pr.node_zoom().zoom_box
    import torch
    small = pr.propose(np.full(12, 2000), np.tile([400, 300, 445, 345], (12, 1)), 0, [480], 1, 12, (W, H), zoom_box)["runs"][0]["zoom"]
    box, _ = zoom_box(torch.tensor(np.tile([400, 300, 445, 345], (12, 1))), W, H, W, H)
    # THE CASE THAT GOES RED when the zoom is not the node's own
    assert small["zooms"] == round(W / (box[2] - box[0]), 2) and small["box"] == list(box) and small["zooms"] >= 1.25, small
    wide = np.tile([0, 0, 1000, 700], (12, 1))
    big = pr.propose(np.full(12, 2000), wide, 0, [480], 1, 12, (W, H), zoom_box)["runs"][0]["zoom"]
    assert big["zooms"] is None and "under 1.25 times" in big["why"], big
    return "each run's zoom is the node's zoom_box on its boxes: a small part zooms with its box, one across the frame is refused in the node's words"


def the_command() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "job").mkdir()
        (d / "share" / "Video/v/s/pv").mkdir(parents=True)
        (d / "job" / "job.json").write_text(json.dumps({"job": {"name": "pv", "first": 100, "frames": 14, "canvas": [W, H], "output": "Video/v/s"}}))
        (d / "job" / "preview.queued.json").write_text(json.dumps({"status": "success", "files": ["Video/v/s/pv/pv_a_parts_00001.mkv"]}))
        write_mask_video(d / "share" / "Video/v/s/pv/pv_a_parts_00001.mkv", made(14, [2000] * 14))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert pr.main(["--preview", str(d / "job"), "--subject", "a", "--least", "480", "--share", str(d / "share"), "--json", str(d / "o.json")]) == 0
        got = json.loads((d / "o.json").read_text())
        assert got["runs"][0]["frames"] == [100, 113] and got["part_file"] == "Video/v/s/pv/pv_a_parts_00001.mkv" and "PROPOSE 100-113" in out.getvalue(), got
        for argv, words in ((["--first", "90"], "is not inside the preview's"), (["--subject", "b"], "saved 0 part video(s)"), (["--json", str(d / "o.json")], "kept output")):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pr.main(["--preview", str(d / "job"), "--subject", "a", "--least", "480", "--share", str(d / "share"), *argv])
                raise AssertionError(f"taken: {argv}")
            except SystemExit as e:
                assert words in str(e), (argv, e)
    return "the part file the preview saved; the stretch held inside the preview; another subject's missing part and a kept output refused"


case("the frames and the floors", the_frames_and_the_floors)
case("the runs", the_runs)
case("the node's own zoom", the_nodes_own_zoom)
case("the command", the_command)
sys.exit(finish())
