#!/usr/bin/env python3
"""The look's rooms and the plan check (`bench/look_rooms.py`): what a room holds, and how the node's own plan is read.

1. **A hairline room** holds his hair where the class map has it over his face on any frame within the hold (the
   fringe it flips), and his hair within the band of Face_Neck on any frame within the hold; nothing further into
   his hair, nothing off his track, nothing past the hold.
2. **A look** names its rooms by kind: an unknown kind, a look with no `change` room, a subject with no look and a
   look with no `look_by` are refused; a room's own settings win over `h3_config.LOOK_HAIRLINE`.
3. **The node's latent steps** are core's: the copy here equals `FRAME_PER_TOKEN` in core's model, read from its
   source text.
4. **The node's region, laid back**: a whole-frame window's cells cover the frame; a zoomed window's cells land in
   its box, a cell the box's size over the grid; each written frame takes its own step's cells; a window file that
   is not there, and frames no window writes, are refused.
5. **The figures**: a room pixel outside the region is named by source frame, and none when inside; hair that
   switches between regenerated and kept is counted; a notch is a kept cell three or four of whose neighbours are
   regenerated.
6. **Where it writes** is the job step's parts folder (`shot_to_load.PARTS_AT`).

NOT HELD: the `room` and `check` steps end to end on a real job (a preview's folder, the server's files); what the
model draws; whether a band of 16 px is right for another face."""
from __future__ import annotations

import json
import re
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
    room = lr.hairline_room(cls, track, hold, band)
    assert room[2, 10, 5] and room[6, 10, 5] and not room[1, 10, 5] and not room[7, 10, 5], "the curl's hold"
    assert room[0, 10 - band, 0] and not room[0, 10 - band - 1, 0], "the band's reach into his hair"
    assert not room[0, 15, 0], "his face that is never hair is the part's, not the room's"
    assert not room[:, :, 19].any(), "off his track"
    return f"the curl over his face held {hold} frames either side and no further; the band {band} px into his hair and no further; nothing off his track"


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


def the_steps_are_cores() -> str:
    core = Path(lr.REPO).parents[1] / "comfy" / "ldm" / "minimax" / "model.py"
    found = re.search(r"^FRAME_PER_TOKEN\s*=\s*\(([^)]*)\)", core.read_text(), re.M)
    assert found, f"no FRAME_PER_TOKEN in {core}"
    assert tuple(int(x) for x in found.group(1).split(",") if x.strip()) == lr.FRAME_PER_TOKEN, found.group(0)
    assert lr.steps_of(12) == [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    return f"{lr.FRAME_PER_TOKEN} is core's; frame 0 alone, then fours"


def window(folder: Path, name: str, number: int, first: int, frames: int, tokens: np.ndarray, zoom=None) -> dict:
    meta = {"version": 2, "first_frame": first, "trim": 0, "frames": frames, "height": 8, "width": 8, "source": {}, "zoom": zoom}
    np.savez_compressed(folder / f"{name}_window_{number}_planned_region.npz", tokens=tokens.astype(np.uint8), meta=np.asarray(json.dumps(meta)))
    return {"number": number}


def the_nodes_region_laid_back() -> str:
    whole = np.zeros((4, 6), bool)
    whole[0, 0] = True
    got = lr.laid_back(whole, None, (24, 16))
    assert got[:4, :4].all() and got.sum() == 16, "a whole-frame cell is a sixth of the width"
    box = {"box": [8, 4, 20, 13], "scale": 2.0}
    got = lr.laid_back(whole, box, (24, 16))
    assert got[4:6, 8:10].all() and got.sum() == 4 and not got[:4].any(), "a zoomed cell lands in the box, the box's size over the grid"
    with tempfile.TemporaryDirectory() as t:
        folder = Path(t)
        steps = np.zeros((3, 4, 6), bool)
        steps[0, 0, 0] = steps[1, 1, 1] = steps[2, 2, 2] = True
        wins = [window(folder, "x", 1, 0, 9, steps)]
        wins[0].update(first_written_frame=0, last_written_frame=8)
        (folder / "x_plan.json").write_text(json.dumps({"windows": wins}))
        region, _ = lr.node_region(folder, "x", 9, (24, 16))
        assert region[0, 0, 0] and region[1, 4, 4] and region[4, 4, 4] and region[5, 8, 8] and not region[5, 4, 4], "each frame its own step"
        refused(lambda: lr.node_region(folder, "x", 10, (24, 16)), "write 9 of the load's 10")
        (folder / "x_plan.json").write_text(json.dumps({"windows": wins + [{"number": 2, "first_written_frame": 9, "last_written_frame": 9}]}))
        refused(lambda: lr.node_region(folder, "x", 10, (24, 16)), "window 2", "not there")
    return "whole-frame and zoomed cells land where the node rendered them; each written frame takes its step; a missing window or an unwritten frame refused"


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
    fig = lr.figures_of(region, room, hair, face, 100, 1.0)
    assert fig["room_px_outside_the_region"]["frames"] == [], fig
    room[1, 1, 1] = True
    region[2, 3, 2:8] = True                     # a row of his hair regenerated on the last frame only
    fig = lr.figures_of(region, room, hair, face, 100, 1.0)
    assert fig["room_px_outside_the_region"]["frames"] == [101] and fig["room_px_outside_the_region"]["px"] == [1], fig
    assert fig["hair_px_switching_from_the_frame_before"]["px"] == 6 and fig["hair_px_switching_from_the_frame_before"]["frames"] == 1, fig
    cells = np.ones((1, 5, 5), bool)
    cells[0, 2, 2] = False
    cells[0, 0, 4] = False
    assert lr.notches(cells) == 2, lr.notches(cells)
    cells[0, 1:4, 3] = False
    return f"a room pixel outside the region named by source frame; switching hair counted; notches counted ({lr.notches(cells)} after a cut)"


def where_it_writes() -> str:
    import shot_to_load
    assert lr.PARTS_AT == shot_to_load.PARTS_AT, (lr.PARTS_AT, shot_to_load.PARTS_AT)
    return f"the job step's parts folder, {lr.PARTS_AT}"


case("a hairline room: the fringe and the band, held, on his track", a_hairline_room)
case("a look names its rooms, and what it does not name is refused", a_look_names_its_rooms)
case("the node's latent steps are core's", the_steps_are_cores)
case("the node's own region, laid back through its box", the_nodes_region_laid_back)
case("the figures the check prints", the_figures)
case("where it writes", where_it_writes)
sys.exit(finish())
