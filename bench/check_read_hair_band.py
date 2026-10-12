#!/usr/bin/env python3
"""`bench/read_hair_band.py` prints a hair band as figures; each case is a way a figure could say something the frames do not.

Why it exists: the reader was built to put numbers beside a thing judged on playback (hair drawn on one frame and
given back on the next under a head-and-hair part). A figure that reads "switching" on a band that holds still, or
"still" on one that toggles, would be trusted over the picture. So each figure is held on frames whose answer is
known by construction, with the case that must NOT read as a fault beside it.

1. **The cause, from the plans.** On made cells: the band's cells that are open, that are not, and that changed
   since the frame before, for a region that holds still against one that opens and closes; a frame with no band
   counts nothing.
2. **Drawn, and drawn and undrawn.** A band redrawn the same on every frame reads no switching; one whose upper
   half is redrawn on every other frame reads half the band switching each time; and a SOURCE that moves under a render that follows it reads the source's
   own change and no switching of the render's (the control: movement is not flicker).
3. **The lower edge.** Drawn pixels that reach a known depth read that depth; a reach that moves reads its step;
   nothing drawn reads 0, and a frame with no band reads nothing.
4. **What the pair differs in besides the band.** A different reference video and a different text are each
   named; the band's own keys and the load's name are not; two loads alike but for the band differ in nothing.
5. **From files, the plans alone.** Two build records, their captures' regions and the two mask videos the band
   record names, written in a temporary folder: the band's pixels are the keep-out less the cut copy, the table
   has a row a frame, and the reader refuses a pair of other frames, a `--band` record with no band, and renders
   given without the source.

    <python> bench/check_read_hair_band.py

Needs ffmpeg (the mask videos of case 5). No model, no card, no render: the renders' path through ffmpeg is not
covered here, and the tool's first real run on a pair of renders is where that is met.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
from _lib.frames import write_mask_video  # noqa: E402

import read_hair_band as rb  # noqa: E402

H, W, CELL, N = 96, 160, 16, 6


def rect(x0: int, y0: int, x1: int, y1: int, frames: int = N) -> np.ndarray:
    m = np.zeros((frames, H, W), bool)
    m[:, y0:y1, x0:x1] = True
    return m


def the_cause() -> str:
    band = rect(32, 48, 96, 80)                                    # 4 cells wide, 2 deep: 8 cells
    band[5] = False                                                # the last frame has no band
    cells = rb.cells_any(band, CELL)
    still = np.zeros((N, H // CELL, W // CELL), bool)
    still[:, 3:5, 2:6] = True                                      # every band cell open, on every frame
    rows = rb.cause(cells, still)
    assert [(r["band_cells"], r["open"], r["not_open"], r["changed"]) for r in rows[:5]] == [(8, 8, 0, None)] + [(8, 8, 0, 0)] * 4, rows
    assert (rows[5]["band_cells"], rows[5]["open"], rows[5]["changed"]) == (0, 0, 0) and rows[0]["region_cells"] == 8, rows[5]
    moving = still.copy()
    moving[1::2, 4, 2:6] = False                                   # the lower row of cells is given back on every other frame
    rows = rb.cause(cells, moving)
    assert [(r["open"], r["not_open"], r["changed"]) for r in rows[:5]] == [(8, 0, None), (4, 4, 4), (8, 0, 4), (4, 4, 4), (8, 0, 4)], rows
    elsewhere = still.copy()
    elsewhere[1::2, 0, 0] = True                                   # a cell far from the band opens and closes: not the band's change
    assert [r["changed"] for r in rb.cause(cells, elsewhere)[1:5]] == [0, 0, 0, 0], rb.cause(cells, elsewhere)
    return ("a region that holds still changes no band cell; one that gives a row back every other frame changes that row each time; "
            "a cell that changes outside the band is not counted")


def drawn_and_undrawn() -> str:
    band, region = rect(32, 48, 96, 80), rect(16, 16, 112, 96)
    source = np.full((N, H, W), 100, np.uint8)
    steady = source.copy()
    steady[:, 48:64, 32:96] = 160                                  # the band's upper half redrawn, the same on every frame
    rows = rb.pictures(band, region, source, steady)
    assert all(r["floor"] == 0.0 and r["drawn_share"] == 0.5 for r in rows), rows
    assert rows[0]["switch_share"] is None and all(r["switch_share"] == 0.0 for r in rows[1:]), [r["switch_share"] for r in rows]
    toggles = source.copy()
    toggles[::2, 48:64, 32:96] = 160                               # redrawn on every other frame only
    rows = rb.pictures(band, region, source, toggles)
    assert [r["drawn_share"] for r in rows] == [0.5, 0.0] * 3 and all(r["switch_share"] == 0.5 for r in rows[1:]), rows
    # the control: the SOURCE changes under the band from frame to frame and the render follows it: nothing is drawn and undrawn
    walks = source.copy()
    for n in range(N):
        walks[n, 48:80, 32 + 8 * n:48 + 8 * n] = 200
    follows = walks.copy()
    follows[:, 16:32, 16:112] = 30                                 # it redraws elsewhere in its region, not in the band
    rows = rb.pictures(band, region, walks, follows)
    assert all(r["drawn_share"] == 0.0 and r["switch_share"] == 0.0 for r in rows[1:]) and all(r["source_moved_share"] > 0 for r in rows[1:]), rows
    # a load that left nothing of the frame alone has no floor, and nothing is read on it
    nothing = rb.pictures(band, np.ones((N, H, W), bool), source, steady)
    assert all(r["floor"] is None and r["drawn_share"] is None for r in nothing), nothing
    return "a band redrawn alike reads no switching, half a band redrawn every other frame reads that half, and a moving source under a render that follows it reads the source's change alone"


def the_lower_edge() -> str:
    band = rect(32, 48, 96, 80, 1)[0]
    drawn = np.zeros((H, W), bool)
    drawn[48:60, 32:96] = True                                     # twelve rows down from the band's top
    assert rb.edge_depth(drawn, band) == 12.0
    assert rb.edge_depth(np.zeros((H, W), bool), band) == 0.0 and rb.edge_depth(drawn, np.zeros((H, W), bool)) is None
    region = rect(16, 16, 112, 96, 3)
    source = np.full((3, H, W), 100, np.uint8)
    render = source.copy()
    render[0, 48:60, 32:96], render[1, 48:60, 32:96], render[2, 48:76, 32:96] = 160, 160, 160
    rows = rb.pictures(rect(32, 48, 96, 80, 3), region, source, render)
    assert [(r["edge_px"], r["edge_step_px"]) for r in rows] == [(12.0, None), (12.0, 0.0), (28.0, 16.0)], rows
    return "drawn pixels twelve rows into the band read 12; a reach that moves to 28 reads a step of 16; nothing drawn reads 0"


def besides_the_band() -> str:
    a = {"load": {"name": "x_band", "first": 10, "hair_band": 0.6, "motion_file": "/m/one.mp4", "seed": 7}, "text": {"file": "t1.txt", "sha256": "aa"}}
    b = {"load": {"name": "x", "first": 10, "motion_file": "/m/two.mp4", "seed": 7}, "text": {"file": "t2.txt", "sha256": "bb"}}
    told = rb.differences(a, b)
    assert len(told) == 2 and told[0].startswith("motion_file: '/m/one.mp4' in the band load, '/m/two.mp4' in the pair") and "t1.txt" in told[1], told
    alike = {"load": {**a["load"], "name": "x", "hair_band": None}, "text": a["text"]}
    assert rb.differences(a, alike) == [], rb.differences(a, alike)
    return "a different reference and a different text are named; the band's own key and the name are not; alike but for the band is nothing"


def from_files() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        keep = rect(32, 32, 96, 96)
        cut = keep.copy()
        cut[:, 48:80, 32:96] = False                               # the band: 64 by 32 taken out of the keep-out
        write_mask_video(folder / "keep.mkv", keep)
        write_mask_video(folder / "cut.mkv", cut)
        region = np.zeros((N, H // CELL, W // CELL), bool)
        region[:, 2:5, 2:6] = True

        def record(name: str, band: bool, first: int = 100, opened: np.ndarray = region) -> str:
            capture = folder / f"capture_{name}"
            (capture / "runs" / name).mkdir(parents=True)
            (capture / "manifest.json").write_text(json.dumps({"first_frame": 100, "frames": N, "size": [W, H],
                                                               "runs": [{"name": name, "carried_from_the_node": True}]}))
            np.savez_compressed(capture / "runs" / name / "region.npz", region=opened)
            rec = {"load": {"name": name, "first": first, "frames": N if first == 100 else N - 2}, "capture": str(capture), "text": {"sha256": "aa"},
                   "hair_band": {"file": str(folder / "cut.mkv"), "its_first_frame_is_source_frame": 100,
                                 "made_from": {"keep_out": str(folder / "keep.mkv"), "keep_outs_first_frame_is_source_frame": 100}} if band else None}
            path = folder / f"{name}.built.json"
            path.write_text(json.dumps(rec))
            return str(path)

        shut = region.copy()
        shut[:, 4] = False                                          # the pair gives the band's lower row of cells back
        args = lambda **kw: argparse.Namespace(**{"band": None, "pair": None, "render_band": None, "render_pair": None, "source": None, **kw})  # noqa: E731
        out = rb.read(args(band=record("with", True), pair=record("without", False, opened=shut)))
        assert out["band_px"] == [64 * 32] * N and out["cell_px"] == CELL and "pictures" not in out, out["band_px"]
        rows = rb.table(out)
        assert len(rows) == N and rows[0]["source_frame"] == 100, rows[0]
        assert (rows[0]["band_cells__band"], rows[0]["open__band"], rows[0]["open__pair"], rows[0]["not_open__pair"]) == (8, 8, 4, 4), rows[0]
        assert rows[1]["changed__band"] == 0 and rows[1]["changed__pair"] == 0 and rows[0]["region_cells__band"] == 12 and rows[0]["region_cells__pair"] == 8
        assert out["regions_are"] == {"with": "the node's plan", "without": "the node's plan"} and out["the_pair_differs_besides_the_band_in"] == []
        for kw, word in ((dict(band=record("a", True), pair=record("b", False, first=102)), "a pair is a load of the same frames"),
                         (dict(band=record("c", False), pair=record("d", False)), "has no hair band"),
                         (dict(band=record("e", True), pair=record("f", False), render_band="r.mp4"), "the renders are read together")):
            try:
                rb.read(args(**kw))
            except SystemExit as stop:
                assert word in str(stop), stop
            else:
                raise AssertionError(f"read where the refusal `{word}` was due")
    return "the band is the keep-out less the cut copy; a row a frame from two records and their captures; three refusals by name"


case("the cause, from the plans", the_cause)
case("drawn, and drawn and undrawn", drawn_and_undrawn)
case("the lower edge", the_lower_edge)
case("what the pair differs in besides the band", besides_the_band)
case("from files, the plans alone", from_files)
sys.exit(finish())
