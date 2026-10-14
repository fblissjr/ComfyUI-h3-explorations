#!/usr/bin/env python3
"""The look's rooms and the plan check (`bench/look_rooms.py`): what a room holds, when it is joined, and how the
node's own plan is read.

1. **A hairline room** holds his hair where the class map has it over his face on any frame within the hold (the
   fringe it flips), and his hair within the band of Face_Neck on any frame within the hold; nothing further into
   his hair, nothing off his track, nothing on another subject's track, nothing past the hold.
2. **A look** names its rooms by kind: an unknown kind, a look with no `change` room, a subject with no look and a
   look with no `look_by` are refused; a room's own settings win over `h3_config.LOOK_HAIRLINE`.
3. **The room is joined to the capture's part, after the capture**: a capture not finished and verified is refused.
4. **The node's region, read from its own files** (written by `video_mask.save_window_region`, the node's own
   writer): each written frame takes its own latent step's cells (`video_mask.run_lengths`, past one cycle of
   `FRAME_PER_TOKEN`), a frame the plan leaves as the source holds none, a zoomed window lands in its box as the
   node lays its render, and a plan that continues a stored window, a zoom for another canvas, a gap between windows,
   a window file not there and frames no window writes are refused.
5. **The figures**: a room pixel outside the region is named by source frame, and none when inside; hair that
   switches between regenerated and kept is counted; a notch is a kept cell three or four of whose neighbours are
   regenerated, and a cell at the grid's edge has only the neighbours inside it.
6. **Where it writes** is the job step's parts folder (`shot_to_load.PARTS_AT`).
7. **A room only where the capture kept a part**: on a frame the capture left as the source, the room is empty.

NOT HELD: the `room` and `check` steps end to end on a real job (a capture's folder, the server's files); the job
file's rewrite; what the model draws; whether a band of 16 px is right for another face. The region reader was held
by hand on 2026-10-14 to two real plans (docs/checks.md)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import look_rooms as lr  # noqa: E402

HAIR, FACE = lr.class_index("Hair"), lr.class_index("Face_Neck")


def refused(fn, *words: str) -> None:
    try:
        fn()
    except lr.Refused as e:
        assert all(w in str(e) for w in words), f"refused, but not with {words}: {e}"
        return
    raise AssertionError(f"not refused: {words}")


def a_hairline_room() -> str:
    n, hold, band = 9, 2, 3
    cls = np.zeros((n, 20, 20), np.uint8)
    cls[:, 10:, :] = FACE                     # his face below row 10
    cls[:, :10, :] = HAIR                     # his hair above it
    cls[4, 10, 5] = HAIR                      # a curl over his face on frame 4 only
    track = np.ones((n, 20, 20), bool)
    track[:, :, 19] = False                   # off his track
    others = np.zeros_like(track)
    others[:, 9, 10] = True                   # another subject's track inside his band
    room = lr.hairline_room(cls, track, hold, band, others)
    assert room[2, 10, 5] and room[6, 10, 5] and not room[1, 10, 5] and not room[7, 10, 5], "the curl's hold"
    assert room[0, 10 - band, 0] and not room[0, 10 - band - 1, 0], "the band's reach into his hair"
    assert not room[0, 15, 0], "his face that is never hair is the part's, not the room's"
    assert not room[:, :, 19].any(), "off his track"
    assert room[0, 9, 9] and not room[:, 9, 10].any(), "another subject's track"
    return f"the curl over his face held {hold} frames either side and no further; the band {band} px into his hair and no further; nothing off his track or on another's"


def a_look_names_its_rooms() -> str:
    cls = np.zeros((3, 8, 8), np.uint8)
    track = np.ones((3, 8, 8), bool)
    refused(lambda: lr.rooms_of({"change": [{"room": "beard"}]}, cls, track), "beard", "hairline")
    refused(lambda: lr.rooms_of({"change": []}, cls, track), "no `change` room")
    refused(lambda: lr.look_of({"subjects": {"third": {}}}, "third"), "no `look`")
    refused(lambda: lr.look_of({"subjects": {"third": {"look": {"change": []}}}}, "third"), "look_by")
    _, made = lr.rooms_of({"change": [{"room": "hairline", "band_px": 5}]}, cls, track)
    assert made[0]["settings"]["band_px"] == 5 and made[0]["settings"]["hold_frames"] == lr.h3_config.LOOK_HAIRLINE["hold_frames"], made
    return "an unknown kind, no change room, no look and no look_by refused; a room's own band_px wins over h3_config's"


def only_where_the_capture_kept_a_part() -> str:
    cls = np.zeros((5, 20, 20), np.uint8)
    cls[:, 10:, :] = FACE
    cls[:, :10, :] = HAIR
    track = np.ones((5, 20, 20), bool)
    part = np.zeros_like(track)
    part[[0, 1, 3, 4], 12:, :] = True                       # the capture left frame 2 as the source
    room, _ = lr.rooms_of({"change": [{"room": "hairline"}]}, cls, track, None, part)
    assert room[1].any() and room[3].any() and not room[2].any(), [int(x) for x in room.sum((1, 2))]
    room, _ = lr.rooms_of({"change": [{"room": "hairline"}]}, cls, track)
    assert room[2].any(), "with no part given, every frame keeps its room"
    return "a frame with no part keeps no room: the capture's word that it stays the source holds"


def after_the_capture() -> str:
    with tempfile.TemporaryDirectory() as t:
        ctx = {"capture": Path(t), "subject": "third", "load": {}, "others_named": []}
        refused(lambda: lr.capture_of(ctx), "not finished and verified", "after the chain's `first`")
        (Path(t) / "status.json").write_text(json.dumps({"state": "done"}))
        (Path(t) / "verify.json").write_text(json.dumps({"failed": ["x"]}))
        refused(lambda: lr.capture_of(ctx), "not finished and verified")
    return "a capture not finished, or not verified, is refused: the room joins the part the capture made"


def plan_of(folder: Path, windows: list[dict], zoom=None, **more) -> None:
    (folder / "x_plan.json").write_text(json.dumps({"plan": "h3 song plan", "windows": windows, "zoom": zoom, **more}))


def write_window(folder: Path, number: int, tokens: np.ndarray, frames: int, first: int, size, zoom=None) -> str:
    import torch
    vm = lr.cap()._pack("video_mask")
    name = f"x_window_{number}_planned_region.npz"
    w, h = size
    vm.save_window_region(str(folder / name), torch.zeros(frames, h, w), torch.from_numpy(tokens.astype(np.float32)),
                          {"zoom": zoom}, 0, first, 0)
    return name


def the_nodes_region() -> str:
    vm = lr.cap()._pack("video_mask")
    runs = vm.run_lengths(7)                                   # 21 frames: 1,4,4,4,4 then 1,4
    frames, size = sum(runs), (24, 16)
    tokens = np.zeros((7, 4, 6), bool)
    for k in range(7):
        tokens[k, k % 4, k % 6] = True                         # each step its own cell
    with tempfile.TemporaryDirectory() as t:
        folder = Path(t)
        f1 = write_window(folder, 1, tokens, frames, 0, size)
        plan_of(folder, [{"number": 1, "region_file": f1, "first_frame": 0, "first_written_frame": 0, "last_written_frame": frames - 1,
                          "frames_left_as_source": [3]}])
        region, _, cells = lr.node_region(folder, "x", frames, size)
        starts = np.cumsum([0] + runs)
        for k in range(7):
            for f in range(starts[k], starts[k + 1]):
                if f == 3:
                    assert not region[f].any(), "a frame left as the source holds no region"
                    continue
                y, x = (k % 4) * 4, (k % 6) * 4
                assert region[f, y:y + 4, x:x + 4].all() and region[f].sum() == 16, f"frame {f} is not on step {k}"
        assert sum(len(c) for c in cells) == 7
        refused(lambda: lr.node_region(folder, "x", frames + 1, size), f"write {frames} of the load's {frames + 1}")
        refused(lambda: lr.node_region(folder, "x", frames, (32, 16)), "planned at 24x16")
        plan_of(folder, [{"number": 1, "region_file": f1, "first_frame": 0, "first_written_frame": 2, "last_written_frame": frames - 1}])
        refused(lambda: lr.node_region(folder, "x", frames, size), "writes from frame 2")
        plan_of(folder, [{"number": 1, "region_file": f1, "first_frame": 0, "first_written_frame": 0, "last_written_frame": frames - 1}], continue_from="y")
        refused(lambda: lr.node_region(folder, "x", frames, size), "continues a stored window")
        plan_of(folder, [{"number": 1, "region_file": "gone.npz", "first_frame": 0, "first_written_frame": 0, "last_written_frame": frames - 1}])
        refused(lambda: lr.node_region(folder, "x", frames, size), "gone.npz", "not beside the plan")
        # zoomed: a box of 12x8 at (8, 4), its window the canvas (24x16) at 2x
        zoom = {"box": [8, 4, 20, 12], "source_size": [24, 16], "canvas": [24, 16], "scale": 2.0}
        one = np.zeros((1, 4, 6), bool)
        one[0, 0, 0] = True
        fz = write_window(folder, 1, one, 1, 0, size, zoom)
        plan_of(folder, [{"number": 1, "region_file": fz, "first_frame": 0, "first_written_frame": 0, "last_written_frame": 0}], zoom=zoom)
        region, _, _ = lr.node_region(folder, "x", 1, size)
        assert region[0, 4:6, 8:10].all() and region[0].sum() == 4, np.argwhere(region[0])
        fz = write_window(folder, 1, one, 1, 0, size, dict(zoom, canvas=[48, 32]))
        refused(lambda: lr.node_region(folder, "x", 1, size), "for a canvas of [48, 32]")
    return f"each written frame on its own step over {len(runs)} steps ({runs}); a frame left as the source empty; a zoomed cell in its box; six refusals"


def the_figures() -> str:
    n = 3
    region = np.zeros((n, 10, 10), bool)
    region[:, 4:, 2:8] = True
    room = np.zeros_like(region)
    room[:, 5, 3] = True
    hair = np.zeros_like(region)
    hair[:, 3:5, :] = True
    face = np.zeros_like(region)
    face[:, 5:, 2:8] = True
    fig = lr.figures_of(region, room, hair, face, 100)
    assert fig["room_px_outside_the_region"]["frames"] == [], fig
    room[1, 1, 1] = True
    region[2, 3, 2:8] = True                     # a row of his hair regenerated on the last frame only
    fig = lr.figures_of(region, room, hair, face, 100)
    assert fig["room_px_outside_the_region"]["frames"] == [101] and fig["room_px_outside_the_region"]["px"] == [1], fig
    assert fig["hair_px_switching_from_the_frame_before"]["px"] == 6 and fig["hair_px_switching_from_the_frame_before"]["frames"] == 1, fig
    cells = np.ones((1, 5, 5), bool)
    cells[0, 2, 2] = False                       # four neighbours regenerated: a notch
    cells[0, 0, 4] = False                       # a corner: two neighbours inside the grid, not a notch
    assert lr.notches(cells) == 1, lr.notches(cells)
    cells[0, 0, 2] = False                       # an edge cell with its three inside neighbours regenerated: a notch
    assert lr.notches(cells) == 2, lr.notches(cells)
    return "a room pixel outside the region named by source frame; switching hair counted; notches counted inside the grid only"


def where_it_writes() -> str:
    import shot_to_load
    assert lr.PARTS_AT == shot_to_load.PARTS_AT, (lr.PARTS_AT, shot_to_load.PARTS_AT)
    return f"the job step's parts folder, {lr.PARTS_AT}"


case("a hairline room: the fringe and the band, held, on his track, off another's", a_hairline_room)
case("a look names its rooms, and what it does not name is refused", a_look_names_its_rooms)
case("a room only where the capture kept a part", only_where_the_capture_kept_a_part)
case("the room is joined after the capture", after_the_capture)
case("the node's own region, from its own files", the_nodes_region)
case("the figures the check prints", the_figures)
case("where it writes", where_it_writes)
sys.exit(finish())
