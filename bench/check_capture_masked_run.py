#!/usr/bin/env python3
"""The capture tool's tables against masks whose answers are known by construction, for three subjects.

`bench/capture_masked_run.py` turns saved masks into per-frame rows with a `subject` on each, and rows across
subjects. A table that looks plausible and is wrong would be trusted, so this builds rectangles whose areas,
overlaps and margins can be counted by hand and reads the rows back.

1. **One subject's row.** A rectangle of known size: its share of the frame, its box, one piece; a part that
   is a known fraction of it, wholly inside it; a frame the mask video did not cover is marked so and carries
   no share, so an empty mask and no mask are different rows.
2. **Across subjects.** Three subjects, any pair of which the table must set against each other: two that
   overlap by a counted number of pixels, a third that touches neither.
3. **Two sightings of one subject.** The same rectangle seen by two runs reads 1; the second shifted by half
   its width reads the intersection over union that shift gives.
4. **A run's region and what was given back.** A region that leaves out the cells another subject stands in:
   the cells given back are the cells of the grown mask on that subject, counted here with the node's own
   `grow`; a region that kept them reports none. The control: moving the other subject out of the margin's
   reach makes both columns zero.
5. **The control for the whole table.** One subject's mask moved by a cell changes its own box and the pair's
   overlap and nothing about the third subject.
6. **A plan's region.** Worked out from the masks by the Masked Source's rule: no cell holding the subject is
   left out; a token the other subject touches is left out unless the subject's own mask is in it; with
   nobody kept out the margin reaches them (the control); and a region in whole tokens holds no part token.
7. **The preflight's rules**, each with the case that must NOT raise it: a shot taken just above the line
   against one well above it; a shot called absent with somebody on screen, which is a shot to look at
   whatever it scored, against the same shot inside frames the caller says the subject is not in; a shot
   taken in frames the caller bars; a part that is empty, spilled, a third of its size or moved for a few frames against
   the same part held steady, which raises nothing; and a text with a voice sentence over unvoiced frames
   against the same text over voiced ones, and a denial that is not read as a voice.
10. **The look.** A render at the source's level over the area reads 0, one at the level of a render that held
   reads 1, a frame with no area reads nothing, and a reference no different from the source has no lift
   (which the command refuses).
9. **Segments.** From a class map: the classes a part mask is made of are read back from the mask; each
   segment `<label>.<class>` has its pixels per frame; what lies inside a run's region and is not the carried
   part is counted in pixels and cells, the subject's own and another subject's; and a segment kept out of
   the margin is no longer inside (the control).
8. **The held part.** On a subject that moves a pixel a frame, a part emptied on one frame and put at the
   subject's feet on another is filled exactly from its neighbours; every other frame is byte for byte what
   was given; with frames chosen by the caller only those are filled; a part with nothing wrong is not
   touched; and the middle of a gap longer than `HOLD_REACH` is left and listed.

No video, no model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_capture_masked_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import capture_masked_run as cap  # noqa: E402

H, W, N = 96, 160, 4
CELL = cap.CELL
MARGIN = 16


def rect(x0: int, y0: int, x1: int, y1: int, frames: int = N) -> np.ndarray:
    m = np.zeros((frames, H, W), bool)
    m[:, y0:y1, x0:x1] = True
    return m


ALL = np.ones(N, bool)
A = rect(32, 32, 64, 64)            # 32 by 32
B = rect(48, 32, 96, 64)            # 48 by 32, sharing 16 by 32 with A
C = rect(128, 0, 144, 16)           # 16 by 16, away from both


def one_subject() -> str:
    covered = ALL.copy()
    covered[3] = False
    part = rect(32, 32, 64, 48)     # the top half of A
    rows = cap.subject_rows("a", "run1", 100, A, covered, part)
    r = rows[1]
    assert r["subject"] == "a" and r["seen_by"] == "run1" and r["source_frame"] == 101, r
    assert r["track_share"] == round(32 * 32 / (H * W), 5), r["track_share"]
    assert r["track_box"] == [32, 32, 64, 64] and r["track_pieces"] == 1 and r["track_step"] == 1.0, r
    assert r["parts_of_track"] == 0.5 and r["parts_inside_track"] == 1.0, r
    assert r["parts_centre_in_box"] == [round(15.5 / 32, 3), round(7.5 / 32, 3)], r["parts_centre_in_box"]
    assert rows[3]["covered"] == 0 and "track_share" not in rows[3], rows[3]
    two = A.copy()
    two[:, 80:90, 0:10] = True
    assert cap.subject_rows("a", "run1", 0, two, ALL)[0]["track_pieces"] == 2
    return "share, box, pieces, the part's fraction and place; an uncovered frame carries no share"


def sightings(a=A, b=B, c=C) -> dict:
    return {"a": {"run1": (a, ALL, None)}, "b": {"run1": (b, ALL, None)}, "c": {"run1": (c, ALL, None)}}


def across() -> str:
    row = cap.cross_rows(100, N, sightings(), {})[0]
    assert row["masks_overlap__a__b"] == round(16 * 32 / (H * W), 5), row
    assert row["masks_overlap_of_smaller__a__b"] == 0.5, row
    assert row["masks_overlap__a__c"] == 0.0 and row["masks_overlap__b__c"] == 0.0, row
    assert sum(k.startswith("masks_overlap__") for k in row) == 3, sorted(row)
    return "three subjects, three pairs, each counted"


def two_sightings() -> str:
    seen = {"a": {"run1": (A, ALL, None), "run2": (A.copy(), ALL, None), "run3": (rect(48, 32, 80, 64), ALL, None)}}
    row = cap.cross_rows(0, N, seen, {})[0]
    assert row["same_subject__a__run1__run2"] == 1.0, row
    assert row["same_subject__a__run1__run3"] == round(16 / 48, 4), row
    return "the same mask reads 1, a half-width shift a third"


def _grow():
    return cap._pack("video_mask").grow


def _run(other: np.ndarray, keep_them_out: bool) -> dict:
    """A run on subject a with a fixed margin: its region is the grown mask's cells, less b's own cells when
    `keep_them_out`, which is what the Masked Source does with a wired `others`."""
    import torch
    grown = _grow()(torch.from_numpy(A).to(torch.float32), MARGIN).numpy() > 0.5
    region = cap.cells_any(grown)
    theirs = cap.cells_any(other & ~A)
    if keep_them_out:
        region = region & ~theirs
    reached, back = cap.taken_back(A, other, region, MARGIN, _grow())
    return {"subject": "a", "others": ["b"], "region": region, "carried": A, "read": ALL,
            "reached": {"b": reached}, "back": {"b": back}, "expect": int((cap.cells_any(grown & other & ~A)).sum(axis=(1, 2))[0])}


def given_back() -> str:
    beside = rect(64, 32, 112, 64)                      # b starts where a ends: inside the margin's reach
    kept = _run(beside, True)
    row = cap.cross_rows(0, N, sightings(b=beside), {"pass_a": kept})[0]
    assert kept["expect"] > 0, "the fixture's margin never reaches the other subject"
    assert row["cells_given_back__pass_a__b"] == kept["expect"], (row, kept["expect"])
    assert row["margin_reached__pass_a__b"] > 0 and row["region_on__pass_a__b"] == 0.0, row
    assert row["region_cells_holding__pass_a__b"] == 0, row
    taken = _run(beside, False)
    row = cap.cross_rows(0, N, sightings(b=beside), {"pass_a": taken})[0]
    assert row["cells_given_back__pass_a__b"] == 0 and row["region_on__pass_a__b"] > 0, row
    assert row["region_cells_holding__pass_a__b"] == taken["expect"], (row, taken["expect"])
    far = rect(120, 64, 152, 96)                        # the control: out of the margin's reach
    away = _run(far, True)
    row = cap.cross_rows(0, N, sightings(b=far), {"pass_a": away})[0]
    assert row["cells_given_back__pass_a__b"] == 0 and row["margin_reached__pass_a__b"] == 0.0, row
    rows = cap.run_rows("pass_a", "a", 0, kept["region"], A, ALL)
    assert rows[0]["region_cells"] == int(kept["region"][0].sum()) and rows[0]["margin_share"] > 0, rows[0]
    return f"{kept['expect']} cells given back when kept out, none when taken or out of reach"


def two_runs() -> str:
    one, two = _run(rect(64, 32, 112, 64), False), _run(rect(64, 32, 112, 64), False)
    two = {**two, "subject": "b", "region": cap.cells_any(rect(64, 32, 112, 64)), "reached": {}, "back": {}}
    row = cap.cross_rows(0, N, sightings(b=rect(64, 32, 112, 64)), {"pass_a": one, "pass_b": two})[0]
    shared = int((one["region"][0] & two["region"][0]).sum())
    assert shared > 0 and row["regions_overlap_cells__pass_a__pass_b"] == shared, (row, shared)
    return f"two passes' regions share {shared} cells and the row says so"


def moved() -> str:
    before = cap.cross_rows(0, N, sightings(), {})[0]
    after = cap.cross_rows(0, N, sightings(a=rect(32 + CELL, 32, 64 + CELL, 64)), {})[0]
    assert after["masks_overlap__a__b"] != before["masks_overlap__a__b"], "moving a subject changed nothing"
    assert after["masks_overlap__b__c"] == before["masks_overlap__b__c"], "moving a changed the pair without it"
    box = cap.subject_rows("a", "run1", 0, rect(32 + CELL, 32, 64 + CELL, 64), ALL)[0]["track_box"]
    assert box == [32 + CELL, 32, 64 + CELL, 64], box
    return "a mask moved by a cell moves its box and its pair's overlap, and nothing else"


def plan() -> str:
    beside = rect(64, 32, 112, 64)                      # starts where a ends: in the margin's reach, never in a's cells
    cells = cap.planned_region(A, [beside], MARGIN, _grow(), whole_tokens=False)
    tokens = cap.planned_region(A, [beside], MARGIN, _grow(), whole_tokens=True)
    assert not (cap.cells_any(A) & ~cells).any() and not (cap.cells_any(A) & ~tokens).any(), "a cell holding the subject was left out"
    assert not (cells & cap.cells_any(beside))[0].any(), "a cell the other subject touches, with none of the subject, was taken"
    alone = cap.planned_region(A, [], MARGIN, _grow(), whole_tokens=False)
    assert (alone & cap.cells_any(beside))[0].any(), "the control: with nobody kept out the margin reaches them"
    over = rect(56, 32, 112, 64)                        # overlaps a: a token holding both stays the subject's
    shared = cap.planned_region(A, [over], MARGIN, _grow(), whole_tokens=True)
    both = cap.whole_tokens_of(cap.cells_any(A)) & cap.whole_tokens_of(cap.cells_any(over))
    assert both.any() and (shared | ~both).all(), "a token holding the subject and the other was given up"
    theirs_only = cap.whole_tokens_of(cap.cells_any(over)) & ~cap.whole_tokens_of(cap.cells_any(A))
    assert not (shared & theirs_only).any(), "a token of the other subject alone was taken"
    assert (cap.whole_tokens_of(tokens) == tokens).all(), "a region in whole tokens holds part of a token"
    return f"{int(cells[0].sum())} cells, {int(tokens[0].sum())} in whole tokens; a token with both stays, one of the other alone goes"


def _table(*shots) -> dict:
    return {"match": 0.8, "shots": [{"first_frame": a, "last_frame": b, "people": [{}] * people,
                                     "subject": {"state": state, "similarity": sim}} for a, b, state, sim, people in shots]}


def shot_rules() -> str:
    table = _table((0, 9, "picked", None, 1), (10, 19, "taken", 0.81, 2), (20, 29, "taken", 0.97, 2),
                   (30, 39, "absent", 0.75, 1), (40, 49, "absent", 0.4, 1), (50, 59, "absent", None, 0))
    flags, shots = cap.flag_shots("a", "run1", table, 100, [])
    got = {(f["rule"], f["source_frames"][0][0]): f["level"] for f in flags}
    assert got == {("taken_near_the_line", 110): "iffy", ("absent_with_people_on_screen", 130): "iffy",
                   ("absent_with_people_on_screen", 140): "iffy"}, got
    said, _ = cap.flag_shots("a", "run1", table, 100, [[140, 149]])
    assert {f["source_frames"][0][0]: f["level"] for f in said if f["rule"].startswith("absent")} == {130: "iffy", 140: "likely fine"}, said
    assert len(shots) == 6 and shots[2]["level"] == "likely fine", shots
    barred, _ = cap.flag_shots("a", "run1", table, 100, [[120, 125]])
    assert [(f["rule"], f["level"]) for f in barred if f["source_frames"] == [[120, 129]]] == [("taken_where_not_expected", "likely to fail")], barred
    rows = cap.subject_rows("a", "run1", 100, np.concatenate([A, np.zeros_like(A)]), np.ones(2 * N, bool))
    empty = cap.flag_track("a", "run1", rows, _table((0, 2 * N - 1, "picked", None, 1)), 100)
    assert empty and empty[0]["source_frames"] == [[100 + N, 100 + 2 * N - 1]], empty
    assert not cap.flag_track("a", "run1", rows, _table((0, N - 1, "picked", None, 1), (N, 2 * N - 1, "absent", 0.2, 0)), 100)
    return "near the line, far from it, barred frames, and a track empty only where the subject was taken"


def part_rules() -> str:
    frames = 40
    track = rect(32, 16, 96, 80, frames)
    steady = rect(48, 16, 80, 32, frames)
    rows = cap.subject_rows("a", "run1", 0, track, np.ones(frames, bool), steady)
    assert cap.flag_parts("a", "run1", rows) == [], "a part held steady on its subject raised a flag"
    bad = steady.copy()
    bad[10] = False                                 # empty
    bad[20] = False
    bad[20, 16:32, 96:128] = True                   # off the subject
    bad[30] = False
    bad[30, 16:24, 56:64] = True                    # a twelfth of its size
    bad[35] = False
    bad[35, 64:80, 48:80] = True                    # the same size, at the bottom of the box
    rules = {f["rule"]: f["source_frames"] for f in cap.flag_parts("a", "run1", cap.subject_rows("a", "run1", 0, track, np.ones(frames, bool), bad))}
    assert rules["part_empty_on_the_subject"] == [[10, 10]], rules
    assert [20, 20] in rules["part_off_the_subject"], rules
    assert [30, 30] in rules["part_changes_size"], rules
    assert [35, 35] in rules["part_moves_on_the_subject"], rules
    return "empty, spilled, shrunk and moved are each named on their frame; a steady part raises nothing"


def held_part() -> str:
    frames = 40
    track = np.zeros((frames, H, W), bool)
    part = np.zeros((frames, H, W), bool)
    for n in range(frames):                         # the subject walks a pixel a frame; the part rides at its top
        track[n, 16:80, 20 + n:60 + n] = True
        part[n, 16:32, 28 + n:52 + n] = True
    broken = part.copy()
    broken[20] = False                              # empty on one frame
    broken[30] = False
    broken[30, 64:80, 28 + 30:52 + 30] = True       # at the subject's feet on another
    rows = cap.subject_rows("a", "run1", 0, track, np.ones(frames, bool), broken)
    held, which, left = cap.held_parts(track, broken, rows)
    assert which == [20, 30] and left == [], (which, left)
    for n in which:
        assert (held[n] == part[n]).all(), (n, cap.overlap(held[n], part[n]))
    same = [n for n in range(frames) if n not in which]
    assert (held[same] == broken[same]).all(), "a frame no rule named was changed"
    chosen, which, _ = cap.held_parts(track, broken, rows, only=[[120, 120]], first=100)
    assert which == [20] and (chosen[30] == broken[30]).all(), "a frame the caller did not choose was filled"
    steady, none, _ = cap.held_parts(track, part, cap.subject_rows("a", "run1", 0, track, np.ones(frames, bool), part))
    assert none == [] and (steady == part).all(), "a part with nothing wrong was held"
    gone = part.copy()
    gone[5:35] = False                              # a gap longer than the reach either way from its middle
    _, which, left = cap.held_parts(track, gone, cap.subject_rows("a", "run1", 0, track, np.ones(frames, bool), gone))
    assert left and set(left) == set(range(5, 35)) - set(which) and all(abs(n - 4) <= cap.HOLD_REACH or abs(n - 35) <= cap.HOLD_REACH for n in which), (which, left)
    return "an empty frame and a moved one are filled exactly from their neighbours; only the chosen frames when chosen; a long gap is left"


def segments() -> str:
    names = ("Background", "Face", "Hair", "Hand")
    mine = np.zeros((N, H, W), np.uint8)
    mine[:, 32:48, 32:64] = 1                       # a face, 16 by 32
    mine[:, 16:32, 32:64] = 2                       # hair above it
    theirs = np.zeros((N, H, W), np.uint8)
    theirs[:, 32:48, 64:80] = 3                     # somebody's hand beside the face
    part = mine == 1
    assert cap.part_classes(part, mine, names) == [1], cap.part_classes(part, mine, names)
    rows = cap.segment_rows("a", "run1", 100, mine, names)
    assert rows[0] == {"frame": 0, "source_frame": 100, "subject": "a", "seen_by": "run1", "a.Face": 512, "a.Hair": 512}, rows[0]
    region = cap.planned_region(part, [], MARGIN, _grow(), whole_tokens=False)
    inside = cap.segments_in_region(100, region, part, {"a": mine, "b": theirs}, names)[0]
    assert "a.Face__px" not in inside, "the carried part was counted as something else inside the region"
    assert inside["a.Hair__px"] == 512 and inside["a.Hair__cells"] == 2, inside
    assert inside["b.Hand__px"] == 256 and inside["b.Hand__cells"] == 1, inside
    kept = cap.planned_region(part, [theirs > 0], MARGIN, _grow(), whole_tokens=False)
    assert "b.Hand__px" not in cap.segments_in_region(100, kept, part, {"a": mine, "b": theirs}, names)[0], "kept out, and still inside"
    return "a part's classes read back from its mask; hair and a neighbour's hand inside the region counted; none once kept out"


def under_a_doubted_part() -> str:
    classes = np.zeros((4, H, W), np.uint8)
    classes[0, 32:48, 32:64] = 1                    # the part itself
    classes[1, 32:48, 32:64] = 2                    # hair where the part should be
    classes[2, 32:48, 32:64] = 3                    # a hand there; frame 3 has nothing labelled
    filled = np.zeros((4, H, W), bool)
    filled[:, 32:48, 32:64] = True
    other = np.zeros((4, H, W), bool)
    what = cap.under_doubted([1], 2, filled, [0, 1, 2, 3], classes, [other])
    assert [what[f]["mostly"] for f in range(4)] == ["the part", "their own hair", "their other classes", "nothing labelled"], what
    other[3, 32:48, 32:64] = True
    assert cap.under_doubted([1], 2, filled, [3], classes, [other])[3]["mostly"] in ("another subject", "nothing labelled")
    classes[3, 32:48, 32:64] = 3
    assert cap.under_doubted([1], 2, filled, [3], classes, [other])[3]["another subject"] == 1.0
    return "the part, hair, another class and nothing are each named from what lies under the fill"


def the_look() -> str:
    source = np.array([40.0, 41.0, 39.0, 40.0, np.nan, 40.0])
    held = np.array([80.0, 81.0, 79.0])             # a render that held the new subject, on the first three frames
    like_original = np.array([42.0, 41.0, 40.0, 43.0, np.nan, 39.0])
    like_new = source + 40.0
    figure, lift = cap.look_figure(source, like_original, held, None)
    assert lift == 40.0 and np.nanmax(np.abs(figure)) < 0.1 and np.isnan(figure[4]), (lift, figure)
    figure, _ = cap.look_figure(source, like_new, held, None)
    assert np.allclose(figure[~np.isnan(figure)], 1.0), figure
    flat, lift = cap.look_figure(source, like_new, source[:3], None)
    assert lift == 0.0, "a reference no different from the source gave a lift"
    return "a render at the source's level reads 0, one at the held render's reads 1, a frame with no area reads nothing"


def text_rules() -> str:
    sings = "She is in a kitchen. She performs the main voice on the track as it plays."
    denies = "She is in a kitchen. She does not speak or sing at any point."
    quiet = cap.flag_text(sings, [[500, 600]], 100, 50)
    assert [f["rule"] for f in quiet] == ["text_names_a_voice_where_there_is_none"], quiet
    assert cap.flag_text(sings, [[120, 130]], 100, 50) == [], "a voice sentence over voiced frames was flagged"
    assert cap.flag_text(denies, [[500, 600]], 100, 50) == [], "a denial was read as a voice sentence"
    over = cap.flag_text(denies, [[120, 130]], 100, 50)
    assert [f["rule"] for f in over] == ["voiced_frames_and_no_voice_sentence"] and over[0]["source_frames"] == [[120, 130]], over
    return "a voice sentence over silence, a denial over a voice; neither the other way round"


case("one subject's row", one_subject)
case("across three subjects", across)
case("two sightings of one subject", two_sightings)
case("a run's region and the cells given back", given_back)
case("two runs on the same cells", two_runs)
case("control: a mask moved on purpose", moved)
case("a plan's region from the masks", plan)
case("preflight: shots and tracks", shot_rules)
case("preflight: a part that leaves its subject", part_rules)
case("the held part", held_part)
case("segments and what is inside a region", segments)
case("what lies under a doubted part", under_a_doubted_part)
case("the look between the original and a render that held", the_look)
case("preflight: the text against the voice", text_rules)
sys.exit(finish())
