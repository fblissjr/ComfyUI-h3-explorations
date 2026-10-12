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
   against one well above it; a shot called absent with somebody on screen, at the top until it is answered
   whatever it scored, against the same shot inside frames the caller says the subject is not in; a shot
   taken in frames the caller bars; a part that is empty, spilled, a third of its size or moved for a few frames against
   the same part held steady, which raises nothing; and a text with a voice sentence over unvoiced frames
   against the same text over voiced ones, and a denial that is not read as a voice.
14. **A mouth and its timing.** On drawn mouths: open reads well above shut; a tilted head and a stray label
   elsewhere change nothing; no mouth gives no reading; a smaller face makes the same mouth read larger. On a
   series: the same mouth two frames late is placed two frames late; a mouth found elsewhere in the frame is
   not scored as the same mouth.
13. **What a run changed.** On a frame whose difference from the source is known by construction: a face drawn
   again, a hairline left alone and the half of a neighbour's hand inside the region each read at their own
   figure; a subject's pixels inside and outside the region are told apart; with nothing labelled outside the
   region the floor is the picture's own outside it, and a region over the whole frame has none.
12. **A region carried across a cut.** Through `loop_plan.split_steps`, which owns the arithmetic: a cut two
   frames into a latent step names the step's two frames on the side the subject is not on; a subject on
   both sides makes the whole step shared and nothing across; a cut on a step's own edge and a frame lost
   inside a shot name nothing; and the frames across are the ones the node's `cut_gate` leaves as the source.
11. **Whose a pixel is.** A pixel one track claims is that subject's; of two claimants it is the one whose class
   map names it; named by both or by neither it is contested and not guessed; and with no class map every
   pixel both claim is contested.
10. **The look.** A render at the source's level over the area reads 0, one at the level of a render that held
   reads 1, a frame with no area reads nothing, and a reference no different from the source has no lift
   (which the command refuses).
9. **Segments.** From a class map: the classes a part mask is made of are read back from the mask; each
   segment `<label>.<class>` has its pixels per frame; what lies inside a run's region and is not the carried
   part is counted in pixels and cells, the subject's own and another subject's; and a segment kept out of
   the margin is no longer inside (the control). A class labelled on one subject in the strip beside its
   outline, inside another subject's track, is counted as inside the other's and none of its own.
8. **The held part.** On a subject that moves a pixel a frame, a part emptied on one frame and put at the
   subject's feet on another is filled exactly from its neighbours; every other frame is byte for byte what
   was given; with frames chosen by the caller only those are filled; a part with nothing wrong is not
   touched; and the middle of a gap longer than `HOLD_REACH` is left and listed.

15. **The frames a render's strip shows** (`render_span`). A render that begins on the span's first frame and
   is as long shows every frame and leaves nothing out; one that begins before the span skips its own frames
   before it; one that begins inside the span (a load whose first frames are left as the source) is shown from
   there, on the render's frame 0, with the span's frames before it named; one that ends first has the frames
   after it named; and a render with no frame inside the span is refused with both ranges.

No video, no model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_capture_masked_run.py
"""
from __future__ import annotations

import json
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
    twice = cap.cross_rows(100, N, {"a": {"run1": (A, ALL, None)}, "b": {"run1": (A.copy(), ALL, None)}}, {})[0]
    assert twice["masks_overlap_of_smaller__a__b"] == 1.0 > cap.SAME_PERSON > row["masks_overlap_of_smaller__a__b"], \
        "one person tracked twice does not read above the line two overlapping people read under"
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
    kept = cap.planned_region(A, [], MARGIN, _grow(), whole_tokens=False, keep=[rect(32, 32, 48, 48)])
    assert not (kept & cap.cells_any(rect(32, 32, 48, 48))).any() and kept.sum() == alone.sum() - N, "keep did not win over the subject's own cell"
    return f"{int(cells[0].sum())} cells, {int(tokens[0].sum())} in whole tokens; a token with both stays, one of the other alone goes"


def _table(*shots) -> dict:
    return {"match": 0.8, "shots": [{"first_frame": a, "last_frame": b, "people": [{}] * people,
                                     "subject": {"state": state, "similarity": sim}} for a, b, state, sim, people in shots]}


def shot_rules() -> str:
    table = _table((0, 9, "picked", None, 1), (10, 19, "taken", 0.81, 2), (20, 29, "taken", 0.97, 2),
                   (30, 39, "absent", 0.75, 1), (40, 49, "absent", 0.4, 1), (50, 59, "absent", None, 0))
    flags, shots = cap.flag_shots("a", "run1", table, 100, [])
    got = {(f["rule"], f["source_frames"][0][0]): f["level"] for f in flags}
    # an absence with somebody on screen is unresolved, at the top, whatever the score, until it is answered
    assert got == {("taken_near_the_line", 110): "iffy", ("absent_with_people_on_screen", 130): "likely to fail",
                   ("absent_with_people_on_screen", 140): "likely to fail"}, got
    said, _ = cap.flag_shots("a", "run1", table, 100, [[140, 149]])
    assert {f["source_frames"][0][0]: f["level"] for f in said if f["rule"].startswith("absent")} == {130: "likely to fail", 140: "likely fine"}, said
    part, _ = cap.flag_shots("a", "run1", table, 100, [[140, 145]])
    assert {f["source_frames"][0][0]: f["level"] for f in part if f["rule"].startswith("absent")}[140] == "likely to fail", \
        "half a shot named in --not-in answered the whole shot"
    # answered on the tracker: `shot N: none` leaves it absent and typed; `shot N: person K` makes it taken
    typed = _table((0, 9, "absent (corrected)", 0.75, 1), (10, 19, "absent", 0.4, 2), (20, 29, "taken (corrected)", 0.67, 2))
    typed["shots"][1]["corrected"] = "none"
    answered, states = cap.flag_shots("a", "run1", typed, 100, [])
    assert [(f["rule"], f["level"], f["figures"]["answered_by"]) for f in answered] == \
        [("absent_with_people_on_screen", "likely fine", "a correction typed on the tracker")] * 2, \
        "a typed absence was not taken as answered, or a typed take under the line was called a guess"
    assert [x["state"] for x in states] == ["absent", "absent", "taken"], states
    # a take under the tracker's own line (its rule for the only person in a shot) is not vouched for by the tracker
    under = _table((0, 9, "taken (only person)", 0.59, 2), (10, 19, "taken (only person)", 0.87, 1), (20, 29, "taken (corrected)", 0.59, 2))
    low, _ = cap.flag_shots("a", "run1", under, 100, [])
    assert [(f["rule"], f["level"], f["source_frames"]) for f in low] == [("taken_under_its_own_line", "likely to fail", [[100, 109]])], low
    # a named subject with no mask on any frame of the load
    nobody = cap.subject_rows("a", "run1", 100, np.zeros_like(A), np.ones(N, bool))
    never = cap.flag_never("a", "run1", nobody, _table((0, N - 1, "absent", None, 0)), 100)
    assert [(f["rule"], f["level"], f["source_frames"], f["figures"]["a_shot_is_picked"]) for f in never] == \
        [("subject_never_found", "likely to fail", [[100, 100 + N - 1]], False)], never
    assert cap.flag_never("a", "run1", cap.subject_rows("a", "run1", 100, A, np.ones(N, bool)), None, 100) == [], "a subject with a mask was called never found"
    # a shot table that MARKS the people an earlier tracker holds: they do not make an absence a question
    held = {"match": 0.8, "shots": [{"first_frame": 0, "last_frame": 9, "people": [{"held_by_another_tracker": True}, {"held_by_another_tracker": True}],
                                     "subject": {"state": "absent", "similarity": None}}]}
    assert cap.flag_shots("a", "run1", held, 100, [])[0] == [], "an absence beside people another tracker holds was asked about"
    held["shots"][0]["people"][0] = {"held_by_another_tracker": False}
    assert [f["rule"] for f in cap.flag_shots("a", "run1", held, 100, [])[0]] == ["absent_with_people_on_screen"], "one person nobody holds, and no question"
    assert cap.shot_state({"subject": {"state": "taken (corrected)"}}) == ("taken", True)
    assert cap.shot_state({"subject": {"state": "taken"}, "corrected": ""}) == ("taken", False)
    assert len(shots) == 6 and shots[2]["level"] == "likely fine", shots
    barred, _ = cap.flag_shots("a", "run1", table, 100, [[120, 125]])
    assert [(f["rule"], f["level"]) for f in barred if f["source_frames"] == [[120, 129]]] == [("taken_where_not_expected", "likely to fail")], barred
    rows = cap.subject_rows("a", "run1", 100, np.concatenate([A, np.zeros_like(A)]), np.ones(2 * N, bool))
    empty = cap.flag_track("a", "run1", rows, _table((0, 2 * N - 1, "picked", None, 1)), 100)
    assert empty and empty[0]["source_frames"] == [[100 + N, 100 + 2 * N - 1]], empty
    assert not cap.flag_track("a", "run1", rows, _table((0, N - 1, "picked", None, 1), (N, 2 * N - 1, "absent", 0.2, 0)), 100)
    held = cap.flag_track("a", "run1", rows, _table((0, N - 1, "picked", None, 1), (N, 2 * N - 1, "taken (corrected)", 0.6, 1)), 100)
    assert held and held[0]["source_frames"] == [[100 + N, 100 + 2 * N - 1]], "a corrected shot with no track was not seen as taken"
    # a load that holds a shot its subject has no mask on: frames 100-119, cuts at 106 and 112
    there = np.r_[np.ones(6, bool), np.zeros(6, bool), np.ones(8, bool)]
    assert cap.shots_without(there, [106, 112], 100) == [[106, 111]]
    there[8] = True
    assert cap.shots_without(there, [106, 112], 100) == [], "a shot with a mask on one frame was called bare"
    assert cap.shots_without(np.zeros(20, bool), [], 100) == [[100, 119]] and cap.shots_without(np.zeros(20, bool), [100, 300], 100) == [[100, 119]]
    return ("near the line, far from it, barred frames; an absence with people on screen at the top until a correction or "
            "--not-in for the whole shot answers it; a corrected shot read as taken; a shot of a load with no mask of its subject")


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
    track_a, track_b = rect(32, 16, 64, 48), rect(64, 32, 96, 64)
    reach = mine.copy()
    reach[:, 32:48, 64:72] = 3                      # a hand labelled on a, in the strip beside a's outline, inside b's track
    whose = {r["segment"]: r for r in cap.whose_rows("a", 100, reach, track_a, {"b": track_b}, names) if r["frame"] == 0}
    assert whose["a.Face"]["in_own_track"] == 512 and whose["a.Face"]["in_other_track"] == 0 and whose["a.Face"]["in_neither"] == 0, whose["a.Face"]
    assert whose["a.Hand"] == {"frame": 0, "source_frame": 100, "segment": "a.Hand", "px": 128, "in_own_track": 0, "in_other_track": 128,
                               "other": "b", "in_neither": 0}, whose["a.Hand"]
    picked = cap.class_mask_of(mine, ["Hair"], names)
    assert picked.sum() == N * 512 and not (picked & part).any(), "the class mask of Hair is not the hair"
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


def owners() -> str:
    one, two = rect(32, 32, 64, 64), rect(48, 32, 96, 64)          # both claim x 48-64
    map_one = np.zeros((N, H, W), np.uint8)
    map_two = np.zeros((N, H, W), np.uint8)
    map_one[:, 32:48, 48:64] = 5                    # one's class map names the top half of the shared strip
    map_two[:, 40:56, 48:64] = 7                    # two's names rows 40-56 of it: rows 40-48 named by both, 56-64 by neither
    owner, labels, shared, contested = cap.owner_map({"one": one, "two": two}, {"one": map_one, "two": map_two})
    f = owner[0]
    assert labels == ["one", "two"] and shared[0] == 16 * 32, (labels, shared[0])
    assert (f[32:64, 32:48] == 0).all() and (f[32:64, 64:96] == 1).all(), "a pixel one track claims is not that subject's"
    assert (f[32:40, 48:64] == 0).all(), "named by one's class map alone, and not one's"
    assert (f[48:56, 48:64] == 1).all(), "named by two's class map alone, and not two's"
    assert (f[40:48, 48:64] == cap.CONTESTED).all() and (f[56:64, 48:64] == cap.CONTESTED).all(), "named by both or by neither, and guessed"
    assert (f[0:32] == cap.NOBODY).all() and contested[0] == 2 * 8 * 16, contested[0]
    lent = cap.owned_class_mask(map_one, owner, 0, ["Hand"], ("Background", "a", "b", "c", "d", "Hand"))
    assert lent.sum() == N * 8 * 16 and (lent[0] == (f == 0) & (map_one[0] == 5)).all(), "a class taken where its subject does not own the pixel"
    # whose pixels a part sits on: a part of one's over columns 40-56 is half one's own, the rest as the strip was settled
    part = rect(40, 32, 56, 64)
    whose = cap.part_owners(part, owner, labels, "one")
    assert whose["pixels"] == N * 16 * 32 and whose["shares"] == {"one": 0.625, "two": 0.125, "contested": 0.25, "nobody": 0.0}, whose
    assert whose["frames_on_another_or_contested"] == N and whose["most_px_on_one_frame"] == 8 * 24, whose
    assert cap.part_owners(part, owner, labels, "one", 100)["source_frames_on_another_or_contested"] == [[100, 100 + N - 1]]
    clear = cap.part_owners(rect(32, 32, 48, 64), owner, labels, "one")
    assert clear["shares"]["one"] == 1.0 and clear["frames_on_another_or_contested"] == 0, clear
    assert cap.part_owners(np.zeros_like(part), owner, labels, "one") is None, "an empty part was given shares"
    bare, _, _, left = cap.owner_map({"one": one, "two": two}, {})
    assert left[0] == 16 * 32, "with no class map every pixel both claim is contested"
    return "one claimant owns; of two, the one whose class map names the pixel; named by both or neither is contested, never guessed"


def across_a_cut() -> str:
    vm = cap._pack("video_mask")
    runs = vm.run_lengths(5)                        # the frames under five latent steps
    total = sum(runs)
    cut = runs[0] + runs[1] + 2                     # a cut two frames into the third step
    present = np.zeros(total, bool)
    present[cut:] = True                            # the subject is in the shot after the cut only
    across, shared = cap.cut_frames([100 + cut], 100, present)
    named = [100 + runs[0] + runs[1], 100 + runs[0] + runs[1] + 1]
    assert across == named and shared == [], (across, shared)
    assert cap.cut_frames([100 + runs[0] + runs[1]], 100, present) == ([], []), "a cut on a step's own edge was named"
    both, shared = cap.cut_frames([100 + cut], 100, np.ones(total, bool))
    assert both == [] and shared == list(range(100 + runs[0] + runs[1], 100 + runs[0] + runs[1] + runs[2])), \
        "a subject on both sides of a cut inside one step: the whole step is shared, nothing is across"
    lost = np.ones(total, bool)
    lost[cut - 1] = False
    assert cap.cut_frames([], 100, lost) == ([], []), "a frame lost inside a shot, with no cut, was named"
    if hasattr(vm, "cut_gate"):                     # the node's own gate: the frames it leaves as the source
        import torch
        mask = torch.zeros(total, 4, 4)
        mask[cut:] = 1.0
        gate = vm.cut_gate(mask, 5, [cut])
        assert [100 + i for i, g in enumerate(gate.tolist()) if g == 0.0] == named, (gate.tolist(), named)
    return "the frames of a step on the far side of a cut are across; a subject on both sides makes the step shared; an edge or a lost frame names nothing"


def what_changed() -> str:
    names = ("Background", "Face", "Hair", "Hand")
    mine = np.zeros((H, W), np.uint8)
    mine[32:48, 32:64] = 1
    mine[16:32, 32:64] = 2
    theirs = np.zeros((H, W), np.uint8)
    theirs[32:48, 64:96] = 3
    region = np.zeros((H, W), bool)
    region[16:48, 32:80] = True                     # the face, the hair, and half of their hand
    diff = np.full((H, W), 2.0)
    diff[32:48, 32:64] = 20.0                       # the face was drawn again
    diff[32:48, 64:80] = 9.0                        # and the half of the hand inside the region; the hair was not
    one = cap.frame_changes(diff, region, {"a": mine, "b": theirs}, {"a": mine > 0, "b": theirs > 0}, names)
    assert one["floor"] == 2.0, "with nothing labelled outside the region the floor is not the picture's own outside it"
    everywhere = cap.frame_changes(diff, np.ones((H, W), bool), {"a": mine}, {}, names)
    assert everywhere["floor"] is None, "a region over the whole frame, and a floor"
    assert one["segments"] == {"a.Face": (512, 20.0), "a.Hair": (512, 2.0), "b.Hand": (256, 9.0)}, one["segments"]
    assert one["subjects"]["b"] == {"inside_px": 256, "inside": 9.0, "outside": 2.0}, one["subjects"]
    assert one["subjects"]["a"]["outside"] is None, "a subject wholly inside the region has no outside"
    big = np.zeros((H, W), np.uint8)
    big[64:96, 0:160] = 2                           # enough labelled pixels outside the region for a floor
    assert cap.frame_changes(diff, region, {"a": mine, "c": big}, {}, names)["floor"] == 2.0
    # a render that ends on the source: level 40, then a slide to the floor over the last frames
    frames = list(range(100, 130))
    held = [40.0] * 30
    fade = [40.0] * 22 + [34.0, 30.0, 24.0, 18.0, 12.0, 9.0, 6.0, 3.0]
    assert cap.source_at_the_ends(frames, held, 2.0)["ends_on_the_source"] is None, "a render that held to its last frame"
    end = cap.source_at_the_ends(frames, fade, 2.0)
    assert end["ends_on_the_source"]["source_frames"] == [124, 129] and end["ends_on_the_source"]["share_of_the_level"] == 0.07, end
    assert end["starts_on_the_source"] is None
    start = cap.source_at_the_ends(frames, fade[::-1], 2.0)
    assert start["starts_on_the_source"]["source_frames"] == [100, 105] and start["ends_on_the_source"] is None, start
    dip = [40.0] * 15 + [5.0] + [40.0] * 14          # one low frame in the middle is not an end
    assert cap.source_at_the_ends(frames, dip, 2.0)["ends_on_the_source"] is None
    assert "nothing was replaced" in cap.source_at_the_ends(frames, [4.0] * 30, 2.0)["why_not_read"]
    assert "frames carry a mask" in cap.source_at_the_ends(frames[:5], [40.0] * 5, 2.0)["why_not_read"]
    return "a redrawn face, an untouched hairline and half of a neighbour's hand each read at their own difference"


def _ellipse(cx: int, cy: int, long: int, short: int, angle: float = 0.0) -> np.ndarray:
    import cv2
    img = np.zeros((H, W), np.uint8)
    cv2.ellipse(img, (cx, cy), (long, short), angle, 0, 360, 1, -1)
    return img.astype(bool)


def mouths() -> str:
    face = np.zeros((6, H, W), bool)
    face[:, 8:88, 40:120] = True
    shut, wide = _ellipse(80, 60, 14, 3), _ellipse(80, 60, 14, 10)
    tilted = _ellipse(80, 60, 14, 10, 40.0)
    speck = wide.copy()
    speck[4:7, 4:7] = True                           # a stray label far from the mouth
    opening, centre = cap.mouth_openings(np.stack([shut, wide, tilted, speck, np.zeros((H, W), bool), wide]), face)
    assert opening[1] > 1.6 * opening[0], (opening[0], opening[1])
    assert abs(opening[2] - opening[1]) < 0.05 * opening[1], "a tilted head changed how open the mouth reads"
    assert abs(opening[3] - opening[1]) < 0.02 * opening[1] and abs(centre[3][0] - 80) < 1, "a stray label moved the reading"
    assert np.isnan(opening[4]) and np.isnan(centre[4]).all(), "no mouth, and a reading"
    small = cap.mouth_openings(np.stack([wide]), np.stack([face[0]]))[0][0]
    half = np.zeros((1, H, W), bool)
    half[:, 8:88, 40:80] = True
    assert cap.mouth_openings(np.stack([wide]), half)[0][0] > 1.3 * small, "a smaller face did not make the same mouth read larger"
    t = np.arange(120, dtype=float)
    series = np.sin(t / 5.0) + 0.3 * np.sin(t / 1.7)
    late = np.r_[np.full(2, np.nan), series[:-2]]   # the same mouth two frames late
    by = cap.shifted_agreement(series, late)
    assert max((v, k) for k, v in by.items() if v is not None)[1] == 2 and by[2] > 0.99, by
    where = np.tile([80.0, 60.0], (120, 1))
    score = cap.score_mouth(series, where, late, where)
    assert score["best_shift_arm_late_positive"] == 2 and score["level_difference_median"] is not None, score
    elsewhere = cap.score_mouth(series, where, series, where + 100.0)
    assert elsewhere["frames_same_mouth"] == 0, "a mouth found somewhere else was scored as the same mouth"
    sung = cap.with_the_voice(series, late)          # a mouth that follows the voice's level, two frames late
    assert sung["with_the_vocal_level_best"][0] == 2 and sung["with_the_vocal_level_best"][1] > 0.99, sung
    assert sung["with_the_vocal_level_at_no_shift"] < sung["with_the_vocal_level_best"][1], sung
    idle = cap.with_the_voice(series, np.sin(t * 2.9 + 1.0))    # a mouth that moves on its own
    assert abs(idle["with_the_vocal_level_at_no_shift"]) < 0.2, idle
    assert cap.with_the_voice(np.full(120, np.nan), series)["with_the_vocal_level_best"] is None, "no voice level, and an agreement"
    named = cap.frames_inside("103-105", 100, 8)
    assert named.tolist() == [False, False, False, True, True, True, False, False] and cap.frames_inside(None, 100, 8).all()
    # a mouth open with no voice: 60 frames, the source wide open on 20-35 and again on 50-53, a voice on 40-59
    opening = np.full(60, 0.05)
    opening[20:36], opening[50:54], opening[28] = 0.3, 0.3, np.nan      # one frame of the run with no mouth seen
    spoken = np.zeros(60)
    spoken[40:] = 1
    shut, follows = np.full(60, 0.04), opening.copy()
    found = cap.open_runs(opening, spoken, {"shut": shut, "follows": follows}, first=100)
    assert [r["source_frames"] for r in found["runs"]] == [[120, 135]], "the long unvoiced run, and only it: " + str(found["runs"])
    one = found["runs"][0]["renders"]
    assert one["shut"]["share_of_the_source"] < 0.2 and one["follows"]["share_of_the_source"] == 1.0, one
    assert cap.open_runs(opening, np.ones(60), {}, 100)["runs"] == [], "an open mouth on voiced frames was called open with no voice"
    assert "does not cover" in cap.open_runs(opening, np.full(60, np.nan), {}, 100)["why_none"], "an unknown voice was read as silence"
    assert cap.open_runs(opening, np.ones(60), {}, 100)["why_none"] == "every frame is voiced"
    brief = np.full(60, 0.05)
    brief[20:28] = 0.3
    assert cap.open_runs(brief, spoken, {}, 100)["runs"] == [], "eight frames were called a run"
    return ("open reads above shut, the same tilted or with a stray label; a mouth two frames late is placed two frames late, "
            "against another mouth and against the voice's level; a stretch of source frames scores only its own frames; "
            "a long run of the source's mouth open on unvoiced frames is found, a voiced or a brief one is not")


def saved_regions() -> str:
    """Two windows' region files read back as one run's region, frame for frame.

    Window 1 is 9 frames from frame 0 of the load; window 2 is 9 frames from frame 4 and its video leaves
    5 off the front, so the two tile 13 frames. The canvas is 96 by 160, a token 32 px."""
    import tempfile
    import torch
    vm = cap._pack("video_mask")
    settings = {"composite": "whole region", "feather_pixels": 0, "change_threshold": None, "cuts": [7],
                "grow_pixels": 8, "grow_by": "a fixed margin", "replace": "subject", "edge": "whole tokens"}

    def one(x0: int, present: slice) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(9, H, W)
        mask[present, 32:64, x0:x0 + 32] = 1.0
        tokens = torch.zeros(3, H // 32, W // 32)
        tokens[:, 1, x0 // 32] = 1.0
        return mask, tokens

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "pass_windows"
        folder.mkdir()
        render = str(Path(tmp) / "pass_00001.mp4")
        assert cap.window_region_files(render) == [], "no files, and some found"
        first_mask, first_tokens = one(32, slice(0, 9))
        # the second window's subject is gone after the cut at frame 7 of the load (its own frame 3)
        second_mask, second_tokens = one(96, slice(0, 3))
        vm.save_window_region(str(folder / "pass_window_2_region.npz"), second_mask, second_tokens, settings, 8, 4, 5)
        vm.save_window_region(str(folder / "pass_window_1_region.npz"), first_mask, first_tokens, settings, 8, 0, 0)
        (folder / "pass_window_x_region.npz").write_bytes(b"not a window")
        found = cap.window_region_files(render)
        assert [f.name for f in found] == ["pass_window_1_region.npz", "pass_window_2_region.npz"], found
        # the load's frame 0 is source frame 100; the span is source frames 98 to 113
        region, carried, read, how = cap.read_saved_regions(found, 98, 16, (W, H), at=100, render_frames=13)
        assert read.tolist() == [False] * 2 + [True] * 13 + [False], read.tolist()
        assert region[2:11, 2, 2].all() and not region[2:11, 2, 6].any(), "window 1's frames do not carry window 1's region"
        assert not region[11:15, 2, 2].any(), "window 2's frames carry window 1's region"
        # window 2's own frames 5 to 8 are frames 9 to 12 of the load, all without the subject, in a step the file
        # gives a region. A file written now is from a node whose composite never lays a frame that has no mask of
        # its own: they are left as the source, and named
        assert not region[11:15].any(), "a frame with no mask of its own was read as laid, from a file whose node does not lay it"
        assert how["left_as_the_source_with_no_mask"] == [109, 110, 111, 112] and how["left_as_the_source_across_a_cut"] == [], how
        assert carried[2:11, 32:64, 32:64].all() and not carried[11:15].any(), "the carried mask is not each window's own"
        assert how["files"] == [f.name for f in found], how
        # a cut inside a step with the subject on one side: the other side is left as the source and named
        gated_mask, gated_tokens = one(96, slice(0, 7))
        vm.save_window_region(str(folder / "pass_window_2_region.npz"), gated_mask, gated_tokens, {**settings, "cuts": [11]}, 8, 4, 5)
        region, carried, read, how = cap.read_saved_regions(found, 100, 13, (W, H), render_frames=13)
        # frames 11 and 12 are across the cut from the subject AND have no mask of their own: a file written now names
        # them by the wider rule, and either way they are the source's
        assert how["left_as_the_source_with_no_mask"] + how["left_as_the_source_across_a_cut"] == [111, 112], how
        assert region[9:11, 2, 6].all() and not region[11:13].any() and read[11:13].all(), "a frame across a cut kept its region"
        # a load shorter than its last window: the window's tail is held frames the render does not write
        region, carried, read, how = cap.read_saved_regions(found, 100, 13, (W, H), render_frames=11)
        assert read.tolist() == [True] * 11 + [False] * 2 and not region[11:].any() and how["held_frames_past_the_render"] == 2, how
        assert how["left_as_the_source_across_a_cut"] == [], "a frame past the render's end was named as across a cut"
        # a window file after the one that reaches the render's end is an earlier, longer run's
        vm.save_window_region(str(folder / "pass_window_3_region.npz"), gated_mask, gated_tokens, settings, 8, 40, 5)
        with_stale = cap.window_region_files(render)
        region, carried, read, how = cap.read_saved_regions(with_stale, 100, 13, (W, H), render_frames=13)
        assert read.all() and how["files_of_another_run_left_unread"] == ["pass_window_3_region.npz"], how
        assert cap.read_saved_regions(with_stale, 100, 13, (W, H))[0] is None, "with no render length, a gap before a stale window was read"
        # a window rendered zoomed in holds its region in the zoomed picture's space: refused, not read on the canvas's grid
        real = vm.load_window_region
        vm.load_window_region = lambda path: {**real(path), "zoom": {"box": [0, 0, 80, 48], "scale": 2.0}}
        try:
            refused = cap.read_saved_regions(found, 100, 13, (W, H), render_frames=13)
        finally:
            vm.load_window_region = real
        assert refused[0] is None and refused[3]["zoomed"] and "zoomed in" in refused[3]["refused"], refused[3]
        # files that do not reach the render's end, or leave a gap, are not this render's
        assert cap.read_saved_regions(found, 100, 13, (W, H), render_frames=21)[0] is None, "a longer render, and its region read"
        assert cap.read_saved_regions(found[1:], 100, 13, (W, H))[0] is None, "a window missing from the front, and its region read"
    return ("two windows tile one run's region, each frame from the window whose video holds it; a frame left as the "
            "source across a cut has none and is named; files that do not fit the render are refused")


def graded_holds() -> str:
    """A fill is taken only where it lies on the part's own classes better than the saved part; neither: emptied."""
    face = rect(40, 20, 60, 40, 6)                   # the part's class sits here on frames 0 to 3
    classes = np.where(face, 2, 0).astype(np.uint8)
    classes[4:] = 0                                  # frames 4 and 5: turned away, no face class at all
    arm = rect(90, 20, 110, 40, 6)                   # somewhere the class map does not call the part
    saved = face.copy()
    saved[1] = arm[1]                                # frame 1: the saved part is off the part (a node's own hold)
    saved[2] = False                                 # frame 2: the part model lost it
    saved[4] = arm[4]                                # frame 4: a part on something else, turned away
    saved[5] = False                                 # frame 5: rightly empty
    fill = saved.copy()
    fill[0] = arm[0]                                 # frame 0: the saved part is right and the fill lands on an arm
    fill[1], fill[2] = face[1], face[2]              # frames 1 and 2: the fill is on the part
    fill[5] = arm[5]                                 # frame 5: a fill drawn where there is no part
    out, did = cap.grade_holds(saved, fill, classes, [2])
    assert (out[0] == saved[0]).all() and did[0]["did"].startswith("saved") and did[0]["fill_on_its_classes"] == 0.0, did[0]
    assert (out[1] == face[1]).all() and did[1]["did"] == "filled" and did[1]["saved_on_its_classes"] == 0.0, did[1]
    assert (out[2] == face[2]).all() and did[2] == {"did": "filled", "saved_on_its_classes": None, "fill_on_its_classes": 1.0}, did[2]
    assert 3 not in did and (out[3] == saved[3]).all(), "a frame nobody doubted was graded"
    assert not out[4].any() and did[4]["did"] == "emptied", "a saved part on something else was kept"
    assert not out[5].any() and did[5]["did"] == "emptied" and did[5]["saved_on_its_classes"] is None, did[5]
    # a fill that is on the part but less so than the saved part does not replace it
    half = face.copy()
    half[0, 20:40, 50:70] = True                     # the fill: half on the part, half off, and larger
    kept, why = cap.grade_holds(face, half, classes, [2])
    assert (kept[0] == face[0]).all() and why[0]["did"].startswith("saved"), why[0]
    # a part the class map vouches for, and too small to draw: emptied by size, with its pixels in the decision
    small, said = cap.grade_holds(face, face, classes, [2], least_px=500)        # the face here is 400 px
    assert not small[:4].any() and said[3] == {"saved_on_its_classes": 1.0, "fill_on_its_classes": None, "did": "emptied",
                                               "why": "too small to be drawn as the part", "px": 400, "least_px": 500}, said[3]
    large, quiet = cap.grade_holds(face, face, classes, [2], least_px=400)
    assert (large[3] == face[3]).all() and 3 not in quiet, "a part at the least size was emptied"
    return ("a fill on an arm is refused where the saved part is right; a fill on the part replaces a saved part that is "
            "off it or lost; a frame with no part under either is emptied; an undoubted frame is left alone")


def verifying() -> str:
    """The first check of a job: what was written together, at the canvas, over the span, and the two readers."""
    def video(name: str, at: str, frames: int = 20, size=(W, H)) -> dict:
        return {"file": name, "written": f"2026-10-10T{at}", "size": list(size), "frames": frames}

    def manifest(inputs: dict, runs: list, at: int = 100, shots: bool = True, classes: bool = True) -> dict:
        return {"size": [W, H], "first_frame": 100, "frames": 20, "runs": runs, "subjects": [{"label": "a", "sightings": [
            {"by": "p", "first_source_frame": at, "shots": shots, "classes": "c.mkv" if classes else None, "inputs": inputs}]}]}

    good = {"track": video("t.mkv", "13:44:10"), "parts": video("r.mkv", "13:45:50"), "shots": {"file": "s.json", "written": "2026-10-10T13:44:02"}}
    agree = {"frames_compared": 20, "cells": 1200, "cells_differing": 0, "frames_differing": [], "carried_overlap_median": 0.994,
             "carried_overlap_min": 0.99, "carried_over_a_pixel_off_median": 0.0005, "carried_over_a_pixel_off_max": 0.003}
    run = {"name": "r", "planned": False, "subject": "a", "files": ["w1.npz"], "readers": agree}
    by = lambda checks: {c["check"]: c for c in checks}
    clean = cap.verify_checks(manifest(good, [run]), [])
    assert all(c["ok"] for c in clean) and len(clean) == 5, [(c["check"], c["ok"]) for c in clean]
    # a morning's mask beside an afternoon's: the folder that held two runs under one name
    mixed = by(cap.verify_checks(manifest({**good, "parts": video("r.mp4", "09:17:00")}, [run]), []))
    assert mixed["inputs_written_together"]["ok"] is False and "09:17:00" in mixed["inputs_written_together"]["why"], mixed
    # a pose table from a pass of its own does not make the masks two runs
    posed = by(cap.verify_checks(manifest({**good, "pose": {"file": "p.json", "written": "2026-10-10T09:00:00"}}, [run]), []))
    assert posed["inputs_written_together"]["ok"] is True
    small = by(cap.verify_checks(manifest({**good, "parts": video("r.mkv", "13:45:50", size=(W // 2, H // 2))}, [run]), []))
    assert small["inputs_at_the_canvas"]["ok"] is False and small["inputs_written_together"]["ok"] is True
    short = by(cap.verify_checks(manifest({**good, "track": video("t.mkv", "13:44:10", frames=12)}, [run]), []))
    assert short["inputs_reach_the_span"]["ok"] is False and "111" in short["inputs_reach_the_span"]["why"], short
    late = by(cap.verify_checks(manifest(good, [run], at=104), []))
    assert late["inputs_reach_the_span"]["ok"] is False, "masks that start after the span's first frame were let through"
    old = by(cap.verify_checks(manifest(None, [run]), []))
    assert old["inputs_written_together"]["ok"] is None, "a capture with no recorded inputs was passed"
    # the readers: a region that differs, a render with no saved files, files that were refused
    off = by(cap.verify_checks(manifest(good, [{**run, "readers": {**agree, "cells_differing": 30, "frames_differing": [[3, 4]]}}]), []))
    assert off["region_saved_against_region_read_back"]["ok"] is False
    loose = by(cap.verify_checks(manifest(good, [{**run, "readers": {**agree, "carried_over_a_pixel_off_median": 0.05}}]), []))
    assert loose["region_saved_against_region_read_back"]["ok"] is False
    # a small mask's plain overlap is low for no fault, and is not what is judged
    face = by(cap.verify_checks(manifest(good, [{**run, "readers": {**agree, "carried_overlap_median": 0.97}}]), []))
    assert face["region_saved_against_region_read_back"]["ok"] is True, "a small mask's overlap failed two readers that agree to the pixel"
    older = by(cap.verify_checks(manifest(good, [{**run, "readers": {k: v for k, v in agree.items() if "pixel_off" not in k}}]), []))
    assert older["region_saved_against_region_read_back"]["ok"] is None
    none = by(cap.verify_checks(manifest(good, [{"name": "r", "planned": False, "subject": "a"}]), []))
    assert none["region_saved_against_region_read_back"]["ok"] is None, "a render with no saved files was called verified"
    refused = by(cap.verify_checks(manifest(good, [{"name": "r", "planned": False, "subject": "a", "saved_regions_refused": "a gap"}]), []))
    assert refused["region_saved_against_region_read_back"]["ok"] is False
    # a plan that carries a part needs the class map; a track empty in a taken shot fails; no shot table cannot be checked
    plan_run = {"name": "pl", "planned": True, "subject": "a", "carried_is": "the part"}
    assert by(cap.verify_checks(manifest(good, [plan_run], classes=False), []))["class_map_for_a_planned_part"]["ok"] is False
    assert by(cap.verify_checks(manifest(good, [plan_run]), []))["class_map_for_a_planned_part"]["ok"] is True
    empty = {"rule": "track_empty_in_a_taken_shot", "subject": "a", "seen_by": "p", "source_frames": [[104, 105]], "why": "no mask"}
    assert by(cap.verify_checks(manifest(good, [run]), [empty]))["track_where_the_shot_table_says_taken"]["ok"] is False
    # a person's statement accounts for an empty frame it covers, by subject; a frame it does not cover still fails
    gone = {**empty, "source_frames": [[104, 109]], "figures": {"frames": 5, "each": [104, 105, 107, 108, 109]}}
    walked = {"subject": "a", "source_frames": [104, 109], "why": "walks out of the shot", "by": "someone"}
    told = by(cap.verify_checks(manifest(good, [run]), [gone], [walked]))["track_where_the_shot_table_says_taken"]
    assert told["ok"] is True and "walks out of the shot" in told["why"] and told["accounted_for"] == [walked], told
    part = by(cap.verify_checks(manifest(good, [run]), [gone], [{**walked, "source_frames": [104, 107]}]))["track_where_the_shot_table_says_taken"]
    assert part["ok"] is False and part["source_frames"] == [[108, 109]], part
    over = {"name": "pl", "planned": True, "subject": "a", "carried_is": "the track", "load": {"first_frame": 100, "frames": 8}}
    clash = by(cap.verify_checks(manifest(good, [over]), [gone], [walked]))
    assert clash["no_mask_statement_against_a_load"]["ok"] is False and clash["no_mask_statement_against_a_load"]["source_frames"] == [[104, 105], [107, 107]], clash
    short = by(cap.verify_checks(manifest(good, [{**over, "load": {"first_frame": 100, "frames": 4}}]), [gone], [walked]))
    assert "no_mask_statement_against_a_load" not in short, "a load that ends before the excused frames was refused"
    other = by(cap.verify_checks(manifest(good, [run]), [gone], [{**walked, "subject": "b"}]))["track_where_the_shot_table_says_taken"]
    assert other["ok"] is False and "accounted_for" not in other, "another subject's statement accounted for this one's frames"
    assert by(cap.verify_checks(manifest({k: v for k, v in good.items() if k != "shots"}, [run], shots=False), []))[
        "track_where_the_shot_table_says_taken"]["ok"] is None
    # the comparison itself: one cell and a shaved mask
    region = np.zeros((4, H // 16, W // 16), bool)
    region[:, 2, 3] = True
    carried = rect(40, 30, 80, 70)
    other, shaved = region.copy(), rect(41, 30, 80, 70)
    other[1, 4, 4] = True
    read = np.array([True, True, True, False])
    got = cap.readers_agreement((region, carried, read), (other, shaved, np.ones(4, bool)))
    assert got["frames_compared"] == 3 and got["cells_differing"] == 1 and got["frames_differing"] == [[1, 1]], got
    assert 0.97 < got["carried_overlap_median"] < 0.98, got
    assert got["carried_over_a_pixel_off_max"] == 0.0, "a mask shaved by one pixel has pixels over a pixel from the other"
    moved = cap.readers_agreement((region, carried, read), (region, rect(46, 30, 86, 70), np.ones(4, bool)))
    assert moved["carried_over_a_pixel_off_median"] > 0.1, moved
    assert cap.VERIFY_FAILED not in (0, 1, 2, cap.GATE_BLOCKED)
    return ("masks of two runs, another canvas, short of the span or late are each named; a pose table of its own pass is not; "
            "a region that differs, refused files or no saved files do not pass; a check that cannot be made is not a pass")


def _body(subject: str, box: str = "given", left: bool = True, right: bool = True, outside: int = 0) -> dict:
    return {"person": 0, "subject": subject, "bbox": [0, 0, 10, 10], "box_source": box, "keypoints_2d": [[0.0, 0.0]] * 70,
            "keypoints_outside_frame": outside, "left_hand": {"crop_side_px": 80.0, "decoder_used": left},
            "right_hand": {"crop_side_px": 40.0, "decoder_used": right}}


def pose_tables() -> str:
    """A pose table read for one subject, and what it says before its mesh is used as a motion video."""
    table = {"schema": cap.POSE_SCHEMA, "hand_refinement": True, "first_source_frame": 100, "frames": [
        {"frame": 0, "source_frame": 100, "people": [_body("a"), _body("b")]},
        {"frame": 1, "source_frame": 101, "people": [_body("a", box="whole frame"), _body("b")]},
        {"frame": 2, "source_frame": 102, "people": [_body("a", box="whole frame", right=False)]},
        {"frame": 3, "source_frame": 103, "people": [_body("a", right=False, outside=30), _body("a")]},
        {"frame": 4, "source_frame": 104, "people": [_body("b")]},
        {"frame": 5, "source_frame": 105, "people": [_body("a")]}]}
    rows = cap.pose_rows(table, "a", 101, 4)
    assert [r["source_frame"] for r in rows] == [101, 102, 103, 104] and [r["frame"] for r in rows] == [0, 1, 2, 3], rows
    assert [r["bodies"] for r in rows] == [1, 1, 2, 0] and "box_source" not in rows[3], rows
    assert rows[2]["keypoints_outside_share"] == round(30 / 70, 3) and rows[1]["right_hand_refined"] is False, rows
    moved = cap.pose_rows(table, "a", 201, 4, at=200)
    assert [r["source_frame"] for r in moved] == [201, 202, 203, 204], "`at` did not move the table's frame 0"
    lone = cap.pose_rows({**table, "frames": [{"frame": 0, "source_frame": 100, "people": [_body("sample")]}]}, "a", 100, 1)
    assert lone[0]["bodies"] == 1 and lone[0]["named_in_table"] == "sample", "a table under one other name was not taken"
    assert cap.pose_rows(table, "c", 100, 6)[0]["bodies"] == 0, "a label the table does not have took somebody's body"
    try:
        cap.pose_rows({**table, "schema": "h3_body_pose_table/2"}, "a", 100, 6)
    except SystemExit as stop:
        assert "h3_body_pose_table/2" in str(stop)
    else:
        raise AssertionError("a table of another schema was read")
    masks = [{"source_frame": n, "track_share": 0.1} for n in (101, 102, 103, 104)]
    flags = {f["rule"] + ":" + str(f["figures"].get("side", "")): f for f in cap.flag_pose("a", "run", rows, masks, {101}, True)}
    whole = flags["pose_fitted_to_the_whole_frame:"]
    assert whole["level"] == cap.LEVELS[2] and whole["source_frames"] == [[101, 102]] and whole["figures"]["with_another_subject_in_frame"] == 1, whole
    assert flags["several_bodies_under_one_name:"]["source_frames"] == [[103, 103]]
    assert flags["no_pose_where_the_subject_is:"]["source_frames"] == [[104, 104]]
    assert flags["hand_not_refined:right_hand"]["source_frames"] == [[102, 103]] and "hand_not_refined:left_hand" not in flags, flags.keys()
    assert flags["pose_mostly_outside_the_frame:"]["source_frames"] == [[103, 103]]
    alone = cap.flag_pose("a", "run", rows, masks, set(), False)
    assert next(f for f in alone if f["rule"] == "pose_fitted_to_the_whole_frame")["level"] == cap.LEVELS[1], "nobody else in frame, and top level"
    assert not any(f["rule"] == "hand_not_refined" for f in alone), "refinement off, and a hand flagged as not refined"
    assert cap.flag_pose("a", "run", cap.pose_rows(table, "a", 100, 1), masks[:0], {100}, True) == [], "a clean frame was flagged"
    # what a body is doing, from 3D keypoints: x to the image's right, y down, z away from the camera
    k = np.zeros((70, 3))
    k[5], k[6], k[9], k[10] = (0.2, 0.0, 3.0), (-0.2, 0.0, 3.0), (0.15, 0.5, 3.0), (-0.15, 0.5, 3.0)   # shoulders, hips: facing the camera
    k[3], k[4], k[0] = (0.0, -0.25, 3.08), (0.0, -0.25, 2.92), (0.1, -0.25, 3.0)    # ears and nose: the head turned side-on
    k[41], k[62] = (0.1, -0.2, 3.0), (0.3, 0.5, 3.0)                               # right wrist by the nose, left wrist at the hip
    state = cap.pose_state(k)
    assert abs(state["body_yaw"]) < 0.5 and abs(state["hip_yaw"]) < 0.5, state
    assert abs(abs(state["head_yaw"]) - 90) < 0.5 and abs(state["head_lift"]) < 0.5, state
    assert state["right_wrist_to_nose"] < 0.15 and state["left_wrist_to_nose"] > 1.4, state
    lifted = k.copy()
    lifted[0] = (0.1, -0.35, 3.0)                    # the nose above the ears' line: the chin up
    assert cap.pose_state(lifted)["head_lift"] > 30, cap.pose_state(lifted)
    far = k * 3.0                                    # the same pose three times as far and as large
    assert cap.pose_state(far) == state, "the same pose at another distance reads differently"
    with_3d = {**table, "frames": [{"frame": 0, "source_frame": 100, "people": [{**_body("a"), "keypoints_3d": k.tolist()}]}]}
    assert cap.pose_rows(with_3d, "a", 100, 1)[0]["right_wrist_to_nose"] == state["right_wrist_to_nose"]
    assert "head_yaw" not in rows[0], "a table with no 3D keypoints grew a facing"
    return ("a subject's bodies by name and frame, under `at` and under one other name; the whole-frame box is top level only "
            "with another subject in frame; two bodies, no body, an unrefined hand and a body mostly out of frame are named; "
            "from 3D keypoints the body's and the head's facing, the chin's lift and a hand at the face, the same at any distance")


def loads() -> str:
    """A pass as a list of loads: each a plan on its own frames, its steps counted from its own first frame."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pass.json"
        path.write_text(json.dumps({"defaults": {"margin": 64, "others": ["b", "c"], "still": "/x/still.png"},
                                    "loads": [{"first": 278, "frames": 16, "subject": "a", "text": "shot one"},
                                              {"first": 415, "frames": 23, "subject": "a", "text": "shot two", "margin": 32,
                                               "name": "second", "audio": None}]}))
        got = cap.loads_from_file(str(path))
        assert [n for n, _ in got] == ["a_0278", "second"], got
        assert got[0][1] == {"margin": "64", "others": "b+c", "still": "/x/still.png", "frames": "16", "subject": "a",
                             "text": "shot one", "at": "278"}, got[0][1]
        assert got[1][1]["margin"] == "32" and got[1][1]["at"] == "415" and "audio" not in got[1][1], got[1][1]
        for bad, word in (({"loads": [{"first": 1, "frames": 2}]}, "lacks subject"),
                          ({"loads": [{"first": 1, "frames": 2, "subject": "a"}] * 2}, "two loads are named")):
            path.write_text(json.dumps(bad))
            try:
                cap.loads_from_file(str(path))
            except SystemExit as stop:
                assert word in str(stop), stop
            else:
                raise AssertionError(f"a pass file with {word} was read")
    # a span of 20 frames from source frame 100; a load of 6 frames from 104, one from 96, one with no end
    assert np.nonzero(cap.load_frames(100, 20, 104, 6))[0].tolist() == [4, 5, 6, 7, 8, 9]
    assert np.nonzero(cap.load_frames(100, 20, 96, 6))[0].tolist() == [0, 1]
    assert cap.load_frames(100, 20, 110, None).tolist() == [False] * 10 + [True] * 10
    row = np.arange(20) >= 12
    assert cap.from_the_load(row, 100, 96).tolist() == [False] * 4 + row.tolist(), "a load from before the span"
    assert cap.from_the_load(row, 100, 110).tolist() == row[10:].tolist(), "a load from inside the span"
    assert cap.from_the_load(row, 100, 110, 4).tolist() == row[10:14].tolist(), "a load does not end on its last frame"
    # the reason for a load a shot: the same cut and subject, read from a long load and from the shot's own first frame
    runs = cap._pack("video_mask").run_lengths(6)
    cut = 100 + runs[0] + runs[1] + 2               # two frames into the long load's third step
    present = (100 + np.arange(sum(runs))) >= cut    # the subject is in the shot after the cut
    long_load, _ = cap.cut_frames([cut], 100, cap.from_the_load(present, 100, 100))
    assert long_load == [cut - 2, cut - 1], long_load
    own, shared = cap.cut_frames([cut], cut, cap.from_the_load(present, 100, cut))
    assert own == [] and shared == [], "a load that starts on its shot's first frame has a cut inside a step"
    # a load that ends one frame before a cut, at a length that is not a whole number of steps: the cut is not in it
    before = np.ones(sum(runs), bool)
    ends = runs[0] + runs[1] + 3
    assert cap.cut_frames([100 + ends], 100, cap.from_the_load(before, 100, 100, ends)) == ([], []), "a cut after a load's end was named"
    assert cap.cut_frames([100 + ends], 100, cap.from_the_load(before, 100, 100))[1] != [], "the control: the same cut inside a longer load"
    # where a load starts against where its subject is largest
    assert cap.small_start(np.array([0.02, 0.05, 0.20, 0.10])) == {"first_frame_share": 0.02, "largest_share": 0.2, "largest_on_load_frame": 2}
    assert cap.small_start(np.array([0.11, 0.20, 0.10])) is None, "a load that starts at over half its largest was named"
    assert cap.small_start(np.zeros(5)) is None and cap.small_start(np.zeros(0)) is None, "no subject, and a small start"
    return ("a pass file is one plan a load, defaults under each; a load holds its own frames only; a load from its shot's "
            "first frame has no cut inside a step where a long load names two frames; a start at under half the largest is named")


def the_gate() -> str:
    """A flag at the top level blocks until it is overridden as it stands; an override needs a who and a why."""
    import argparse
    import contextlib
    import io
    import tempfile
    top = {"id": "f001", "rule": "two_tracks_on_one_person", "level": cap.LEVELS[2], "subject": "a", "other": "b",
           "source_frames": [[10, 20]]}
    iffy = {"id": "f002", "rule": "taken_near_the_line", "level": cap.LEVELS[1], "subject": "a", "source_frames": [[30, 40]]}
    assert cap.gate_verdict([iffy], []) == {"verdict": "clear", "blocking": [], "overridden": []}, "an iffy flag blocked"
    assert cap.gate_verdict([top, iffy], [])["blocking"] == ["f001"]
    # what a render did against a flag is not an override
    seen = {"flag": "f001", "key": cap.flag_key(top), "happened": False}
    assert cap.gate_verdict([top], [seen])["verdict"] == "blocked", "an outcome was taken for an override"
    passed = {"flag": "f001", "key": cap.flag_key(top), "overridden": True}
    assert cap.gate_verdict([top, iffy], [passed]) == {"verdict": "clear", "blocking": [], "overridden": ["f001"]}
    # the same flag under another number is the same flag; the flag with more frames is not
    assert cap.gate_verdict([{**top, "id": "f007"}], [passed])["verdict"] == "clear", "a renumbered flag lost its override"
    grown = {**top, "source_frames": [[10, 26]]}
    assert cap.gate_verdict([grown], [passed])["blocking"] == ["f001"], "a flag that grew kept its override"
    assert cap.flag_key(top) != cap.flag_key({**top, "other": "c"}), "two flags about different subjects share a key"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "flags.json").write_text(json.dumps({"verdict": "blocked", "blocking": ["f001"], "flags": [top, iffy]}))

        def run(happened: str, **more) -> None:
            given = {"capture": str(folder), "flag": "f001", "happened": happened, "render": None, "note": "", "by": "", **more}
            with contextlib.redirect_stdout(io.StringIO()):
                cap.outcome(argparse.Namespace(**given))

        for missing in ({"by": "someone"}, {"note": "a reason"}, {"by": " ", "note": " "}):
            try:
                run("overridden", **missing)
            except SystemExit as stop:
                assert "needs --by" in str(stop), stop
            else:
                raise AssertionError(f"an override was recorded with {missing} only")
        assert not (folder / "outcomes.json").exists(), "a refused override was written"
        run("no", render="r.mp4")
        assert json.loads((folder / "flags.json").read_text())["verdict"] == "blocked", "an outcome cleared the gate"
        run("overridden", by="someone", note="a reason")
        after = json.loads((folder / "flags.json").read_text())
        assert after["verdict"] == "clear" and after["overridden"] == ["f001"] and len(after["flags"]) == 2, after
        last = json.loads((folder / "outcomes.json").read_text())["outcomes"][-1]
        assert last["said"] == "overridden by someone: a reason" and "happened" not in last, last
    assert cap.GATE_BLOCKED not in (0, 1, 2)
    return ("a top-level flag blocks, an iffy one does not; an override needs who and why and clears the flag as it stands, "
            "under any number; a flag that grew blocks again; an outcome is not an override")


def planned_regions() -> str:
    """A pass that has not rendered, read from the plan and the region files the node's preview wrote.

    A 13-frame load in two windows of 9: window 2 starts on frame 4, keeps 5 and writes 4, and its last
    four frames are held past the load's end. The plan says which window writes a frame; the capture does
    not work it out."""
    import tempfile
    import torch
    vm = cap._pack("video_mask")
    settings = {"composite": "whole region", "feather_pixels": 0, "change_threshold": None, "cuts": [], "grow_pixels": 16,
                "grow_by": "a fixed margin", "replace": "subject", "edge": "latent cells"}

    def window(x0: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(9, H, W)
        mask[:, 32:64, x0:x0 + 32] = 1.0
        tokens = torch.zeros(3, H // 32, W // 32)
        tokens[:, 1, x0 // 32] = 1.0
        return mask, tokens

    def row(number: int, first: int, trim: int, writes: int, left=()) -> dict:
        return {"number": number, "first_frame": first, "frames": 9, "trim": trim, "frames_written": writes,
                "first_written_frame": first + trim, "last_written_frame": first + trim + writes - 1, "text": f"text {number}",
                "region_file": f"pass_window_{number}_planned_region.npz", "frames_left_as_source": list(left), "regenerating_share": 0.07}

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "pass_windows"
        folder.mkdir()
        for number, x0, first, trim in ((1, 32, 0, 0), (2, 96, 4, 5)):
            vm.save_window_region(str(folder / f"pass_window_{number}_planned_region.npz"), *window(x0), settings, 16, first, trim)
        plan = {"plan": "h3 song plan", "version": 1, "source": {"frames": 13, **settings},
                "windows": [row(1, 0, 0, 9), row(2, 4, 5, 4, left=(11,))]}
        (folder / "pass_plan.json").write_text(json.dumps(plan))
        assert cap.plan_file(str(folder)).name == "pass_plan.json" and cap.plan_file(str(folder / "pass_plan.json")).name == "pass_plan.json"
        # the load's frame 0 is source frame 100; the span is source frames 98 to 113
        region, carried, read, how = cap.read_planned_regions(folder / "pass_plan.json", 98, 16, (W, H), at=100)
        assert read.tolist() == [False] * 2 + [True] * 13 + [False], "frames past the load's end, or before it, were read"
        assert region[2:11, 2, 2].all() and not region[2:11, 2, 6].any(), "window 1's frames do not carry window 1's region"
        assert region[11:13, 2, 6].all() and region[14, 2, 6] and not region[11:15, 2, 2].any(), "window 2 writes with its own region"
        assert not region[13].any() and read[13] and how["left_as_the_source_across_a_cut"] == [111], "the plan's gated frame kept a region"
        assert carried[11:15, 32:64, 96:128].all() and how["frames_written"] == 13 and how["planned_windows"][1]["writes_source_frames"] == [109, 112], how
        assert how["source_settings"]["grow_pixels"] == 16 and how["source_settings"]["edge"] == "latent cells"
        # a load shorter than the span, with nothing saying where it starts, is refused: it is not put at the span's first frame
        assert "say which source frame the load starts on" in cap.read_planned_regions(folder / "pass_plan.json", 98, 16, (W, H))[3]["refused"]
        assert cap.read_planned_regions(folder / "pass_plan.json", 100, 13, (W, H))[1] is not None, "a load as long as the span needs no `at`"
        # a window the plan could not plan, a file that is not there, and a gap are each refused with the reason
        unplanned = {**plan, "windows": [row(1, 0, 0, 9), {**row(2, 4, 5, 4), "region_file": None, "why": "starts past the source's 13 frames"}]}
        (folder / "pass_plan.json").write_text(json.dumps(unplanned))
        assert "starts past" in cap.read_planned_regions(folder / "pass_plan.json", 100, 13, (W, H))[3]["refused"]
        (folder / "pass_plan.json").write_text(json.dumps({**plan, "windows": [row(1, 0, 0, 9), row(2, 6, 5, 2)]}))
        assert "ended on 9" in cap.read_planned_regions(folder / "pass_plan.json", 100, 13, (W, H))[3]["refused"]
        # a run that continues a stored window: one window that keeps 5 frames of that window's and writes 4. The kept
        # frames are not a gap, carry no region and are not read; the same plan with no `continue_from` is a gap, and
        # so is a continuation whose first window writes from anywhere but the end of its kept frames
        vm.save_window_region(str(folder / "pass_window_1_planned_region.npz"), *window(96), settings, 16, 0, 5)
        going_on = {**plan, "continue_from": "stored_window_1.safetensors", "windows": [row(1, 0, 5, 4)]}
        (folder / "pass_plan.json").write_text(json.dumps(going_on))
        region, carried, read, how = cap.read_planned_regions(folder / "pass_plan.json", 98, 16, (W, H), at=100)
        assert read.tolist() == [False] * 7 + [True] * 4 + [False] * 5, "a continuation's kept frames were read, or its written ones were not"
        assert region[7:11, 2, 6].all() and not region[:7].any(), "the kept frames of a stored window carry a region"
        assert how["kept_from_a_stored_window"] == {"file": "stored_window_1.safetensors", "source_frames": [100, 104]}, how
        assert how["frames_written"] == 9 and how["planned_windows"][0]["writes_source_frames"] == [105, 108], how
        assert how["planned_windows"][0]["kept_frames"] == 5 and how["planned_windows"][0]["first_frame"] == 100, how
        (folder / "pass_plan.json").write_text(json.dumps({**going_on, "continue_from": ""}))
        assert "ended on 0" in cap.read_planned_regions(folder / "pass_plan.json", 98, 16, (W, H), at=100)[3]["refused"]
        (folder / "pass_plan.json").write_text(json.dumps({**going_on, "windows": [{**row(1, 0, 5, 4), "first_written_frame": 6}]}))
        assert "ended on 0" in cap.read_planned_regions(folder / "pass_plan.json", 98, 16, (W, H), at=100)[3]["refused"]
        vm.save_window_region(str(folder / "pass_window_1_planned_region.npz"), *window(32), settings, 16, 0, 0)
        (folder / "pass_plan.json").write_text(json.dumps(plan))
        (folder / "pass_window_2_planned_region.npz").unlink()
        assert "not beside the plan" in cap.read_planned_regions(folder / "pass_plan.json", 100, 13, (W, H))[3]["refused"]
        (folder / "other_plan.json").write_text("{}")
        try:
            cap.plan_file(str(folder))
        except SystemExit as stop:
            assert "2 plan file(s)" in str(stop)
        else:
            raise AssertionError("a folder with two plans was read as one")
    # a frame with no mask of its own inside a latent step that has one is lent the step's region
    step = np.zeros((6, H // 16, W // 16), bool)
    step[1:5, 2, 3] = True                           # one step over frames 1 to 4
    own = rect(48, 32, 64, 48, 6)
    own[[0, 2, 5]] = False                           # frame 2 was emptied on purpose; 0 and 5 are outside the step
    assert cap.lent_frames(own, step, np.ones(6, bool)) == [2], "the emptied frame inside the step, and only it"
    assert cap.lent_frames(own, step, np.array([1, 1, 0, 1, 1, 1], bool)) == [], "a frame that was not read was called lent"
    return ("the plan says which window writes a frame and which frame is left as the source; a held tail is not read; an "
            "unplanned window, a missing file, a gap and two plans in one folder are each refused; the kept frames of a run "
            "that continues a stored window are not a gap and carry no region; a frame with no mask inside a step that has "
            "one is named as lent")


def held_tails() -> str:
    """Before anything samples: is the model shown, as a frame to keep, the subject it is replacing.

    A window of 9 frames (latent steps 0 | 1-4 | 5-8) whose load ends inside it; the rest is the last frame held."""
    import tempfile
    import torch
    vm = cap._pack("video_mask")

    def window(real: int, tail_open: bool, on_last: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(9, H, W)
        mask[:real, 32:64, 32:64] = 1.0
        if not on_last:
            mask[real - 1] = 0.0
        tokens = torch.zeros(3, H // 32, W // 32)
        tokens[:2 if real > 1 else 1, 1, 1] = 1.0
        if tail_open:
            mask[real:, 32:64, 32:64] = 1.0
            tokens[2, 1, 1] = 1.0
        return mask, tokens

    plate = cap.held_tail_facts(*window(5, False), 4, 100, vm.run_lengths)
    assert plate["held_frames"] == 4 and plate["held_share"] == round(4 / 9, 3) and plate["held_frames_shown_clean"] == 4, plate
    assert plate["subject_on_the_last_real_frame"] and plate["step_with_real_and_held_source_frames"] is None, "the held frames start on a step's edge"
    inside = cap.held_tail_facts(*window(4, False), 5, 100, vm.run_lengths)
    # the load ends inside the step of frames 1 to 4: that step has a region (three masked frames), so its held frame is
    # not clean; the four held frames of the last step are
    assert inside["step_with_real_and_held_source_frames"] == [101, 103] and inside["held_frames_shown_clean"] == 4, inside
    opened = cap.held_tail_facts(*window(5, True), 4, 100, vm.run_lengths)
    assert opened["held_frames_shown_clean"] == 0, "a tail with its region open was read as shown clean"
    assert cap.held_tail_facts(*window(9, False), 0, 100, vm.run_lengths)["held_frames_shown_clean"] == 0, "a load that fills its window has a tail"
    gone = cap.held_tail_facts(*window(5, False, on_last=False), 4, 100, vm.run_lengths)
    assert gone["held_frames_shown_clean"] == 4 and not gone["subject_on_the_last_real_frame"], gone
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "subjects" / "a").mkdir(parents=True)
        (folder / "subjects" / "a" / "per_frame.json").write_text(json.dumps({"rows": [{"seen_by": "p", "track_share": 0.015}] * 5}))

        def manifest(*wins, planned=True) -> dict:
            run = {"name": "pl", "planned": planned, "subject": "a",
                   "planned_windows": [{"window": i + 1, "writes_source_frames": [100, 104], **w_} for i, w_ in enumerate(wins)] or None}
            return {"first_frame": 100, "subjects": [{"label": "a", "sightings": [{"by": "p"}]}], "runs": [run]}

        raised = cap.flag_held_tail(manifest(plate), folder)
        assert [(f["rule"], f["level"]) for f in raised] == [("original_shown_in_the_held_tail", cap.LEVELS[2])], raised
        assert raised[0]["figures"]["subject_share_of_frame_median"] == 0.015 and "held_tail" in raised[0]["why"] and "1.50%" in raised[0]["why"], raised[0]
        assert cap.flag_held_tail(manifest(opened), folder) == [], "the remedy set, and the flag raised"
        # a window that writes more frames than its load holds: the frame past the end is a held copy laid on the next source frame
        long = cap.flag_held_tail(manifest({**opened, "first_frame": 100, "writes_source_frames": [100, 105]}), folder)
        assert [(f["rule"], f["level"], f["source_frames"]) for f in long] == [("plan_writes_past_the_load", cap.LEVELS[2], [[105, 105]])], long
        assert cap.flag_held_tail(manifest({**opened, "first_frame": 100}), folder) == [], "a plan that writes exactly its load was flagged"
        assert cap.flag_held_tail(manifest(gone), folder) == [], "the subject is not on the held frame, and the flag raised"
        blind = cap.flag_held_tail(manifest(), folder)
        assert [(f["rule"], f["level"]) for f in blind] == [("held_tail_not_checked", cap.LEVELS[1])], "an estimate with no windows was passed in silence"
        assert cap.flag_held_tail(manifest(plate, planned=False), folder) == [], "a rendered run was gated as a plan"
    # a continuation: the head faces away on the last kept frame and turns to the camera over the new frames
    def at(frame: int, yaw: float, lift: float = 0.0, wrist: float = 0.9) -> dict:
        return {"source_frame": frame, "head_yaw": yaw, "head_lift": lift, "left_wrist_to_nose": wrist, "right_wrist_to_nose": 1.2}

    table = {f: at(f, -76.0) for f in range(100, 110)}
    table.update({110 + i: at(110 + i, -76.0 + 8.0 * i, -3.0 * i, 0.9 - 0.06 * i) for i in range(10)})
    found = cap.turn_from_kept(table, 109, (110, 119))
    assert found == {"head_turn_degrees": 72.0, "reached_on_source_frame": 119, "chin_change_degrees": 27.0, "a_wrist_nearer_the_nose_by": 0.54}, found
    assert cap.turn_from_kept(table, 105, (106, 109))["head_turn_degrees"] == 0.0, "a subject who does not turn"
    assert cap.turn_from_kept({170: at(170, 170.0), 171: at(171, -170.0)}, 170, (171, 171))["head_turn_degrees"] == 20.0, "the turn went the long way round"
    assert cap.turn_from_kept({}, 109, (110, 119)) is None and cap.turn_from_kept({109: {"source_frame": 109}}, 109, (110, 119)) is None
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "subjects" / "a").mkdir(parents=True)
        win = {"window": 2, "first_frame": 100, "kept_frames": 10, "writes_source_frames": [110, 119]}
        m = {"subjects": [{"label": "a", "sightings": [{"by": "p"}]}],
             "runs": [{"name": "pl", "planned": True, "subject": "a", "planned_windows": [{"window": 1, "kept_frames": 0}, win]}]}
        blind = cap.flag_kept(m, folder)
        assert [f["rule"] for f in blind] == ["kept_frames_not_checked"], "no pose table, and the continuation passed in silence"
        (folder / "subjects" / "a" / "pose__p.json").write_text(json.dumps({"rows": list(table.values())}))
        far = cap.flag_kept(m, folder)
        assert [(f["rule"], f["level"], f["source_frames"]) for f in far] == [("kept_frames_far_from_the_pose_ahead", cap.LEVELS[1], [[110, 119]])], far
        still = {**m, "runs": [{**m["runs"][0], "planned_windows": [{**win, "first_frame": 96, "writes_source_frames": [106, 109]}]}]}
        assert cap.flag_kept(still, folder) == [], "a continuation whose source holds its pose was flagged"
    # the figures a per-clip rule reads: a span's shots from its cuts, and the head across one shot
    assert cap.shot_edges(100, 30, [90, 100, 110, 110, 125, 130, 200]) == [[100, 109], [110, 124], [125, 129]]
    assert cap.shot_edges(100, 30, []) == [[100, 129]]
    across = cap.head_across([{"source_frame": 99}] + [table[f] for f in range(105, 120)])
    assert across["head_yaw_first"] == -76.0 and across["head_yaw_last"] == -4.0 and across["largest_turn_from_the_first_degrees"] == 72.0, across
    assert across["reached_on_source_frame"] == 119 and across["chin_degrees"] == [-27.0, 0.0] and across["a_wrist_nearest_the_nose"] == 0.36, across
    assert cap.head_across([{"source_frame": 99}]) is None
    return ("a load's last frame held past its end and shown clean with the subject on it blocks; a tail with its region open, "
            "a load that fills its window and a last frame without the subject do not; an estimate with no windows says it was not checked; "
            "a continuation whose source turns away from its kept frames' pose is named, one that holds its pose is not")


def face_sizes() -> str:
    """A face a few tokens tall is named before a face-only pass; a larger one, or a part that is not a face, is not."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "subjects" / "a").mkdir(parents=True)

        def manifest(height: int, made=("Face_Neck", "Upper_Lip")) -> dict:
            parts = rect(40, 8, 80, 8 + height, 6)
            parts[5] = False                         # a frame with no part does not count
            np.savez_compressed(folder / "subjects" / "a" / "masks__p.npz", parts=np.packbits(parts, axis=-1), track=np.packbits(parts, axis=-1))
            return {"size": [W, H], "first_frame": 100, "frames": 6, "runs": [{"name": "r", "subject": "a"}],
                    "subjects": [{"label": "a", "sightings": [{"by": "p", "part_is_made_of": list(made)}]}]}

        token = cap.TOKEN_PX
        cap.TOKEN_PX = 16                            # this check's canvas is 96 px tall: a token of 16 puts the line at 64 px
        try:
            small = cap.flag_face_size(manifest(24), folder)
            assert [(f["rule"], f["level"], f["figures"]["tokens"]) for f in small] == [("face_small_in_tokens", cap.LEVELS[1], 1.5)], small
            assert cap.flag_face_size(manifest(int(cap.FACE_TOKENS * 16)), folder) == [], "a face at the line was named"
            assert cap.flag_face_size(manifest(24, made=("Upper_Clothing",)), folder) == [], "a part that is not a face was sized as one"
            nobody = manifest(24)
            nobody["runs"] = []
            assert cap.flag_face_size(nobody, folder) == [], "a subject no run replaces was sized"
        finally:
            cap.TOKEN_PX = token
    # a part in pieces: the head, and a sliver somewhere else that carries the part's class
    head = rect(40, 8, 80, 56, 4)
    sliver = rect(120, 70, 140, 76, 4)
    sliver[3] = False
    piece, stray = cap.largest_piece(head | sliver)
    assert (piece == head).all() and stray.tolist() == [120, 120, 120, 0], stray.tolist()
    assert cap.largest_piece(np.zeros((2, H, W), bool))[1].tolist() == [0, 0]
    return ("a face under the line in tokens is named, with its height; a face at the line, a part that is not a face, and a "
            "subject no run replaces are not; a part's pieces away from its largest are counted and can be dropped")


def who_is_who() -> str:
    """A shot on which some tracker's call is nobody's word blocks, one flag a shot, and shows the question to ask."""
    def cell(ask="", word="") -> dict:
        return {"called": "taken", "called_person": 1, "answered_by": word, "ask": ask, "numbers_on_its_tile": [1, 2], "tile_frame": 406}

    q = "shot 2, lead's tile (frame 406): which numbered person is lead? 1, 2, or none"
    file = {"first_source_frame": 600, "not_picked": {}, "rows": [
        {"shot": 1, "frames": [0, 405], "status": "answered", "why": "", "note": "a person's word: lead (picked), second (picked)",
         "trackers": {"lead": cell(word="picked"), "second": cell(word="picked")}},
        {"shot": 2, "frames": [406, 434], "status": "ask", "why": "nobody has said who lead, second are in this shot", "note": "",
         "trackers": {"lead": cell(ask=q), "second": cell(ask=q.replace("lead", "second"))}},
        {"shot": 3, "frames": [435, 446], "status": "ask", "why": "lead and second have one person on one frame: two trackers on one person",
         "note": "", "trackers": {"lead": cell(word="picked"), "second": cell(word="corrected by hand")}},
        {"shot": 4, "frames": [447, 500], "status": "answered", "why": "", "note": "every detection in this shot has no head: things, not people",
         "trackers": {"lead": cell(), "second": cell()}}]}
    flags = cap.flag_who(file, 1000, 50)             # the capture holds source frames 1000 to 1049: shots 2 and 3, and the start of 4
    assert [(f["rule"], f["level"], f["source_frames"], f["subject"]) for f in flags] == [
        ("who_is_who_unanswered", cap.LEVELS[2], [[1006, 1034]], None), ("who_is_who_unanswered", cap.LEVELS[2], [[1035, 1046]], None)], flags
    assert q in flags[0]["why"] and q.replace("lead", "second") in flags[0]["why"], "one flag a shot did not carry every tracker's question"
    assert "two trackers on one person" in flags[1]["why"], "a row with no question did not show its reason"
    assert set(flags[0]["figures"]) == {"shot", "status", "frames_are_source_frames"}, flags[0]["figures"]
    # a file written before the tool was cut: only `agrees` is settled, and a row with no question shows its why
    older = {"first_source_frame": 600, "rows": [
        {"shot": 1, "frames": [406, 434], "status": "agrees", "trackers": {"lead": {"ask": ""}}},
        {"shot": 2, "frames": [435, 440], "status": "differs", "why": "the sum says otherwise", "trackers": {"lead": {"ask": q, "would_type": "shot 2: person 1"}}},
        {"shot": 3, "frames": [441, 446], "status": "unresolved", "why": "no look to compare on", "note": "3 look(s) not read", "trackers": {"lead": {"ask": ""}}}]}
    old = cap.flag_who(older, 1000, 50)
    assert [f["source_frames"] for f in old] == [[[1035, 1040]], [[1041, 1046]]] and q in old[0]["why"] and "shot 2: person 1" not in old[0]["why"], old
    assert "no look to compare on" in old[1]["why"] and "[3 look(s) not read]" in old[1]["why"], old[1]["why"]
    cause = cap.flag_who({**file, "not_picked": {"second": "second's tracker has its subject on no shot of this load"}}, 1000, 50)
    assert cause[0]["why"].startswith("FIRST: second picked nobody (second's tracker has its subject on no shot of this load)"), cause[0]["why"]
    # several tracked subjects and no file at all is said, and blocks; one tracked subject has nothing to decide jointly
    def tracked(*labels, tables=True) -> dict:
        return {"first_frame": 1000, "frames": 50, "subjects": [{"label": x, "sightings": [{"by": "p", "shots": "t.json" if tables else None}]} for x in labels]}
    unrun = cap.flag_who_not_run(tracked("lead", "second"))
    assert [(f["rule"], f["level"], f["source_frames"], f["subject"]) for f in unrun] == [("who_is_who_not_run", cap.LEVELS[2], [[1000, 1049]], None)], unrun
    assert cap.flag_who_not_run(tracked("lead")) == [] and cap.flag_who_not_run(tracked("lead", "second", tables=False)) == [], \
        "one tracked subject, or masks with no shot table, was asked for a joint decision"
    answered = {**file, "rows": [{**r, "status": "answered"} for r in file["rows"]]}
    assert cap.flag_who(answered, 1000, 50) == [], "rows that are answered were flagged"
    assert cap.flag_who(file, 100, 50) == [], "rows outside the capture's span were flagged"
    unplaced = cap.flag_who({**file, "first_source_frame": None}, 1000, 50)
    assert len(unplaced) == 2 and unplaced[0]["source_frames"] == [[406, 434]] and not unplaced[0]["figures"]["frames_are_source_frames"], unplaced[0]
    return ("a shot with a call that is nobody's word blocks, one flag a shot, with every tracker's question or the row's reason; an "
            "answered row and one outside the span are not flagged; a file from before the cut is read by the same rule")


def gate_by_shot() -> str:
    """The gate's answer per shot and subject is built from the capture's own two files, and a clear cell still says what was taken."""
    def subject(state, likeness, share, masked, whose=None, typed=False) -> dict:
        return {"on_screen": {"state": state, "likeness": likeness, "typed": typed}, "share_of_the_frame": None if share is None else {"median": share},
                "frames_with_a_mask": masked, "frames": 10, "part_pixels_owned_by": whose}

    crossing = {"frames_on_another_or_contested": 2, "source_frames_on_another_or_contested": [[108, 109]], "most_px_on_one_frame": 40, "shares": {}}
    record = {"shots": [
        {"shot": 1, "source_frames": [100, 109], "who_is_who": {"status": "answered"},
         "subjects": {"a": subject("taken", 0.86, 0.48, 10, crossing), "b": subject("absent", None, None, 0)}},
        {"shot": 2, "source_frames": [110, 119], "who_is_who": None,
         "subjects": {"a": subject("absent", 0.7, None, 0), "b": subject("taken", 0.9, 0.2, 10, typed=True)}}]}
    flags = {"blocking": ["f001", "f003", "f004"], "flags": [
        {"id": "f001", "rule": "absent_with_people_on_screen", "subject": "a", "source_frames": [[110, 119]]},
        {"id": "f002", "rule": "part_empty_on_the_subject", "subject": "a", "source_frames": [[104, 104]]},      # overridden: not in `blocking`
        {"id": "f003", "rule": "who_is_who_unanswered", "subject": None, "source_frames": [[110, 119]]},
        {"id": "f004", "rule": "subject_never_found", "subject": "b", "source_frames": [[100, 104]]}]}
    one, two = cap.by_shot(record, flags)
    assert one["subjects"]["a"]["blocked_by"] == [] and one["subjects"]["a"]["likeness"] == 0.86 and one["subjects"]["a"]["share_of_the_frame"] == 0.48, one
    assert one["subjects"]["a"]["part_on_another_or_contested"]["source_frames_on_another_or_contested"] == [[108, 109]], one
    assert one["subjects"]["b"]["blocked_by"] == ["subject_never_found"] and one["every_subject"] == [] and one["who_is_who"] == "answered", one
    assert two["subjects"]["a"]["blocked_by"] == ["absent_with_people_on_screen"] and two["subjects"]["b"]["blocked_by"] == [], two
    assert two["every_subject"] == ["who_is_who_unanswered"] and two["who_is_who"] is None, two
    assert two["subjects"]["b"]["typed"] and not one["subjects"]["a"]["typed"], "a typed state was not told from the tracker's own"
    # per load: a flag that names the run, blocking, overridden or a warning; kept and held frames; who else is inside
    loads = {"loads": [{"run": "r1", "subject": "a", "source_frames": [100, 109], "windows": [{"kept_frames": 0, "held_frames_shown_clean": 4}, {"kept_frames": 90}],
                        "voice": {"voiced_frames": 0, "frames": 10, "source_mouth_open_on_unvoiced_frames": 6},
                        "inside_the_region": {"other_subjects": {"b": {"frames": 3, "source_frames": [[101, 103]], "most_share_of_its_mask": 0.5}}}},
                       {"run": "r2", "subject": "b", "source_frames": [110, 119], "windows": None, "inside_the_region": {}}]}
    said = {"blocking": ["f001"], "overridden": ["f002"], "flags": [
        {"id": "f001", "rule": "region_over_another_subject", "level": cap.LEVELS[2], "run": "r1", "other": "b", "source_frames": [[101, 103]]},
        {"id": "f002", "rule": "region_on_a_frame_with_no_mask", "level": cap.LEVELS[2], "run": "r1", "source_frames": [[105, 105]]},
        {"id": "f003", "rule": "segment_inside_region", "level": cap.LEVELS[1], "run": "r1", "source_frames": [[100, 109]]},
        {"id": "f004", "rule": "segment_inside_region", "level": cap.LEVELS[1], "run": "r1", "source_frames": [[100, 104]]},
        {"id": "f005", "rule": "part_empty_on_the_subject", "level": cap.LEVELS[2], "subject": "a", "source_frames": [[100, 100]]}]}
    r1, r2 = cap.by_load(loads, said)
    assert [f["rule"] for f in r1["blocked_by"]] == ["region_over_another_subject"] and r1["blocked_by"][0]["other"] == "b", r1
    assert [f["id"] for f in r1["overridden"]] == ["f002"] and r1["warnings"] == {"segment_inside_region": 2}, r1
    assert r1["kept_frames"] == 90 and r1["held_frames_shown_clean"] == 4 and r1["other_subjects_inside"]["b"]["frames"] == 3, r1
    assert r2["blocked_by"] == [] and r2["kept_frames"] is None and r2["other_subjects_inside"] == {}, r2
    assert r1["voice"] == {"voiced_frames": 0, "frames": 10, "source_mouth_open_on_unvoiced_frames": 6} and r2["voice"] is None, (r1["voice"], r2["voice"])
    return ("a flag counts on the shots its frames touch and only while it blocks; a flag with no subject is the shot's; "
            "a clear cell carries the take's state, likeness and size")


def a_look() -> str:
    """A look is one line by somebody who did not make the capture, holding what each strip it looked at showed."""
    import hashlib
    import tempfile
    status = {"state": "done", "written": "2026-10-10T17:00:00", "made_by": "maker"}
    row = {"load": "a_1", "frames": [100, 132], "text_label": "hall", "text_frames": [100, 140], "text_by": "writer", "text_sha256": "ab" * 32,
           "strip": "a_1.mp4", "made": "2026-10-10T17:30:00"}
    entry = cap.look_entry(status, [row], "reader", "every frame and the text, on the strip", "2026-10-10T18:00:00")
    assert entry == {"by": "reader", "at": "2026-10-10T18:00:00", "loads": ["a_1"], "note": "every frame and the text, on the strip", "shown": [row],
                     "capture_written": "2026-10-10T17:00:00", "maker": "maker"}, entry
    assert cap.look_entry({"written": "2026-10-10T17:00:00"}, [row], "reader", "all of it", "now")["maker"] == "not recorded"
    for bad, why in (((status, [row], "maker", "looked", "now"), "the maker's own look was taken"),
                     ((status, [], "reader", "looked", "now"), "a look of no strip was taken"),
                     ((status, [{**row, "made": "2026-10-10T16:59:00"}], "reader", "looked", "now"), "a strip drawn before the capture was written was taken"),
                     ((status, [row], "reader", " ", "now"), "a look with no note was taken")):
        try:
            cap.look_entry(*bad)
        except SystemExit:
            continue
        raise AssertionError(why)
    # what the strip shows of a load: its frames, its text whole, and where the text was written from; the hash is of the stripped text
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "hall.txt").write_text("\n  She walks down the hall.\nShe stops.  \n")
        # the shape of a real job file: `texts` is in the `job` block, where the builder reads it (a made-up job with it at
        # the top level kept this case green while no real job could ever be recorded, 2026-10-10)
        job = {"job": {"name": "j", "texts": {"hall": {"file": "hall.txt", "frames": [100, 140], "by": "writer"}}}, "defaults": {"subject": "a"},
               "loads": [{"name": "a_1", "first": 100, "frames": 33, "text": "hall"}, {"name": "a_2", "first": 150, "frames": 9, "text": "old"}]}
        (folder / "job.json").write_text(json.dumps(job))
        (folder / "a_2.plan.txt").write_text("An older text.\n")
        got = cap.load_and_text(folder / "job.json", "a_1")
        assert got["frames"] == [100, 132] and got["recorded"] and got["text"] == "She walks down the hall.\nShe stops." and got["text_by"] == "writer" \
            and got["text_frames"] == [100, 140] and got["text_sha256"] == hashlib.sha256(got["text"].encode()).hexdigest(), got
        # the hash is the job builder's rule (the string it writes into the graph), restated in the capture so a strip can be drawn
        # for a job the build refuses. Two functions are held together by a case and not by a comment: the same file through both
        (folder / "hall.txt").write_text("\n\n  She walks down the hall.\r\nShe stops.  \n\n")
        mine = cap.load_and_text(folder / "job.json", "a_1")
        sys.path.insert(0, str(Path(cap.__file__).resolve().parent))
        import masked_job
        try:
            theirs = masked_job.text_of(job["job"], {**job["defaults"], **job["loads"][0]}, folder)
        except (SystemExit, Exception) as exc:           # the builder's half is not in this tree yet, or refuses this small job
            theirs = exc
        compared = isinstance(theirs, dict) and "sha256" in theirs
        if compared:
            assert theirs["sha256"] == mine["text_sha256"], f"the builder hashes {theirs['sha256']} and the capture {mine['text_sha256']} for one file"
        top = {**job, "job": {"name": "j"}, "texts": job["job"]["texts"]}       # the block in the wrong place is no record at all
        (folder / "top.json").write_text(json.dumps(top))
        assert not cap.load_and_text(folder / "top.json", "a_1")["recorded"], "a texts block outside the job block was read as the job's"
        old = cap.load_and_text(folder / "job.json", "a_2")
        assert not old["recorded"] and old["text"] == "An older text." and old["text_from"] == "a_2.plan.txt" and "text_sha256" not in old, old
        try:
            cap.load_and_text(folder / "job.json", "a_9")
        except SystemExit:
            pass
        else:
            raise AssertionError("a load the job does not have was shown")
    lines = cap.wrap_text("one two three four\n\nfive", 9, len)
    assert lines == ["one two", "three", "four", "", "five"] and " ".join(x for x in lines if x).split() == "one two three four five".split(), lines
    return ("a look holds what each strip showed; the maker's own, a look of no strip, a strip older than the capture and an empty note "
            "are refused; a load's text is shown whole with the frames it was written from, or marked not recorded; "
            + ("the builder's hash of one file equals the capture's" if compared else
               "THE BUILDER'S HASH WAS NOT COMPARED (its text_of does not give one in this tree)"))


def a_load_that_lays_nothing() -> str:
    """Where the subject is tracked and a load's own mask is empty with no region on the frame, nothing is laid: named per load."""
    carried = rect(32, 32, 64, 64, 6)
    carried[2:5] = False                             # frames 2 to 4: the part is empty
    region = np.zeros((6, H // 16, W // 16), bool)
    region[[0, 1, 2, 5], 2, 2] = True                # frame 2 sits in a step that has a region (a lent frame, another rule's)
    read = np.array([True] * 5 + [False])            # frame 5 is not the load's
    tracked = np.array([True, True, True, True, False, True])
    assert cap.bare_frames(carried, region, read, tracked) == [3], "frames with a region, without the subject, or outside the load were named"
    assert cap.bare_frames(rect(32, 32, 64, 64, 6), region, read, tracked) == [], "a load that lays its mask on every frame was named"
    return "a frame the load reads, with the subject tracked, an empty mask and no region is named; a lent frame, an untracked one and one outside the load are not"


def a_strip_of_a_render() -> str:
    """Which frames the stacked strip shows of a render that does not begin or end where the capture's span does."""
    whole = cap.render_span(1093, 41, 1093, 41)
    assert whole == {"skip": 0, "lo": 0, "frames": 41, "left_out": []}, whole
    early = cap.render_span(1093, 41, 1090, 44)          # the render began three frames before the span
    assert early == {"skip": 3, "lo": 0, "frames": 41, "left_out": []}, early
    late = cap.render_span(1093, 41, 1098, 36)           # frames 1093-1097 were left as the source: the render begins at 1098
    assert late == {"skip": 0, "lo": 5, "frames": 36, "left_out": [{"frames": [1093, 1097], "why": "before the render"}]}, late
    # the strip's frame k is span frame lo + k and render frame skip + k: the same source frame on both rows
    assert all(1093 + late["lo"] + k == 1098 + late["skip"] + k for k in range(late["frames"]))
    assert all(1093 + early["lo"] + k == 1090 + early["skip"] + k for k in range(early["frames"]))
    short = cap.render_span(1093, 41, 1098, 20)          # and it ends before the span does
    assert short["frames"] == 20 and short["left_out"] == [{"frames": [1093, 1097], "why": "before the render"},
                                                           {"frames": [1118, 1133], "why": "after the render"}], short
    for outside in ((1093, 41, 1134, 10), (1093, 41, 1000, 93)):
        try:
            cap.render_span(*outside)
            raise AssertionError(f"a render with no frame inside the span was taken: {outside}")
        except SystemExit as e:
            assert "no frame is in both" in str(e) and "1093-1133" in str(e), e
    return "a render inside the span is shown from its own frame 0 with the span's frames before and after it named; one outside it is refused"


def text_rules() -> str:
    sings = "She is in a room. She performs the main voice on the track as it plays."
    denies = "She is in a room. She does not speak or sing at any point."
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
case("what a run changed", what_changed)
case("a mouth and its timing", mouths)
case("whose a pixel is", owners)
case("the held part", held_part)
case("segments and what is inside a region", segments)
case("what lies under a doubted part", under_a_doubted_part)
case("the look between the original and a render that held", the_look)
case("preflight: a region carried across a cut", across_a_cut)
case("a run's region from the files its windows saved", saved_regions)
case("a pass read from the node's own plan", planned_regions)
case("preflight: the original shown in a held tail", held_tails)
case("preflight: a face only a few tokens tall", face_sizes)
case("preflight: a shot where a tracker's call is nobody's word", who_is_who)
case("shots: the gate's answer per shot and subject", gate_by_shot)
case("the look: what a strip shows of a load and its text, and the line that records it", a_look)
case("preflight: a load that lays nothing where its subject is", a_load_that_lays_nothing)
case("a fill of a doubted part, graded by the class map", graded_holds)
case("verify: the capture reads what the nodes wrote", verifying)
case("a pose table as an input", pose_tables)
case("a pass as a list of loads", loads)
case("preflight: the gate and its override", the_gate)
case("preflight: the text against the voice", text_rules)
case("the strip: the frames a render's strip shows, and the span's frames it leaves out", a_strip_of_a_render)
sys.exit(finish())
