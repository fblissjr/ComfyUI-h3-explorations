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

16. **The look of a load of two subjects.** The load is read by the builder's own rule: its first subject's, with
   which label is whose in the list's order (`<Subject N>`, the still `<Picture N>`, the reference `<Video N>`)
   and where each reference begins (the load's `motion_first`, else the job's first frame); a load whose two
   subjects are one is refused in the builder's words. In the masks row each subject's part is filled in its
   own colour, the one margin in the first's, a pixel both tracks claim stays the first's, and the legend has a
   line each; a run told one subject is drawn as before (the control).

17. **Neither subject of a load of two is another subject to it.** The cross table makes no column for the
   run's region on either; the rule about a region over another subject and the rule about another subject's
   segment inside the region flag neither and still flag a third subject; the second subject's own part is not
   a segment "inside the region", and what else of theirs is inside it is their own, to name in the text; and
   told one subject, the second is flagged as before (the control).

18. **The floor under a small part is a starting value, held against the clip's own sizes.** A capture sizes a
   subject's part on its own frames (the least, the most, five percentiles; the frames under the floor and
   under a quarter, a half and twice the starting value), says who set the floor and which frames it emptied,
   and takes another floor from `least=`: the grade empties under that one and no other, and 0 empties
   nothing. A sighting that was not graded says why nothing was emptied; a part on no frame has no sizes. A
   capture that exists is sized from its record, or from its own saved masks when it was made before sizes
   were recorded. Not covered: the `files` command's own reading of `least=` from a `--mask`, which takes mask
   videos; it was run by hand on one capture's inputs the day it was written (`docs/wiki/decisions.md`).
   The starting floor is held at the size the load renders at (`starting_floor`, 2026-10-14): PART_LEAST_PX over
   the square of the node's own zoom_box on the part's boxes, so a far face is kept where the canvas value
   emptied it; not scaled with `zoom=off` or where the node will not zoom; a run at another zoom names the frames
   whose emptying would change (`floor_against_the_run`).

19. **A mask's `at` is the source frame its preview's files begin on** (inside the verify case). A sighting read
   from another frame than its preview's own job says, for the masks or for the shot table, fails with the
   frames it is off; the right frame passes; a preview whose job folder is not found, or is one of two of the
   name, is not checked and is neither.

21. **No keep-out carves the subject's own track** (`keep_out_inside_own_track`, the owner, 2026-10-16). A keep-out that
   is not one of our people (a mask with no shot table: strangers, held, her hands or clothes) given back cells that lie
   mostly inside the load subject's own track is counted a frame at a time with the cells named, and the gate takes it
   as a hard flag that no override clears; another person given back cells, and a keep-out beside her track, are not.
   Seen red with the column not written, with people taken as keep-outs, and with an override clearing it.
   THE CARVE (`carve_keep_out`): a hands keep-out over her face gives back cells inside her track uncarved and none
   carved, with the pixels and cells it lost; through the files step (its readers stood in), the keep-out's track
   written for the builder is the carved one and the fault column reads 0 on it, another person's track untouched.
   Seen red with the carve skipped in the files step (the fault column nonzero).

22. **The carve on the node's own grid** (`grid_carve`, `node_taken_inside`, the lead's ruling of 2026-10-16). On a plan and
   a region file written by the node's own `save_window_region`, and counted by the node's own `window`: a keep-out over
   cells that are hers on one frame of a latent step and on others of it is given units by the node after the canvas
   carve and none after the grid carve; and on a zoomed load (a box off the canvas grid, 1.333) a band just outside her
   track inside a zoomed unit mostly hers, the same. Through the files step on a plan, the keep-out written for the
   builder is the grid-carved one; `flag_runs` reads the node's count as the hard fault. Seen red with the step pooling
   taken out (frame by frame), with the zoom not laid back (the canvas grid), and with the files step not carving on the plan.

20. **No largest piece of a face or hair** (`refuse_pieces_on_a_head`, `h3_config.NO_PIECES_ON`, the owner, 2026-10-16).
   Run through the files step with its readers stood in: `pieces=largest` on a part the class map says is a face,
   or hair, stops before a piece is dropped; on a part of clothing it reaches the drop; with no class map it stops.
   Seen red with the call taken out of the files step.

No video, no model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_capture_masked_run.py
"""
from __future__ import annotations

import argparse
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


def _run(other: np.ndarray, keep_them_out: bool, carried: np.ndarray = A) -> dict:
    """A run on subject a with a fixed margin: its region is the grown mask's cells, less b's own cells when
    `keep_them_out`, which is what the Masked Source does with a wired `others`. `carried` is a's part (A by default)."""
    import torch
    grown = _grow()(torch.from_numpy(carried).to(torch.float32), MARGIN).numpy() > 0.5
    region = cap.cells_any(grown)
    theirs = cap.cells_any(other & ~carried)
    if keep_them_out:
        region = region & ~theirs
    reached, back = cap.taken_back(carried, other, region, MARGIN, _grow())
    return {"subject": "a", "others": ["b"], "region": region, "carried": carried, "read": ALL,
            "reached": {"b": reached}, "back": {"b": back}, "expect": int((cap.cells_any(grown & other & ~carried)).sum(axis=(1, 2))[0])}


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
    # WHOSE empty track it is. For a subject a load replaces it is the top level; for one who is only an OTHER of the loads it is data
    # with figures, and blocks nothing; a subject of one load and an other of another is a subject
    taken = _table((0, 2 * N - 1, "picked", None, 1))
    load_of_b = [{"name": "ld", "subject": "b", "others": ["a"], "first_source_frame": 100}]
    assert empty[0]["rule"] == "track_empty_in_a_taken_shot" and empty[0]["level"] == cap.LEVELS[2], "with no runs handed in, an empty track was not held as a subject's"
    own = cap.flag_track("a", "run1", rows, taken, 100, [{"name": "ld", "subject": "a", "others": ["b"]}])
    assert [(f["rule"], f["level"]) for f in own] == [("track_empty_in_a_taken_shot", cap.LEVELS[2])], "THE LOAD'S OWN SUBJECT with no mask in a taken shot did not fail"
    pair = cap.flag_track("a", "run1", rows, taken, 100, [{"name": "ld", "subject": "b", "subjects": ["b", "a"], "others": []}])
    assert pair[0]["rule"] == "track_empty_in_a_taken_shot", "the second subject of a load of two was held as an other"
    both = cap.flag_track("a", "run1", rows, taken, 100, load_of_b + [{"name": "l2", "subject": "a", "others": ["b"]}])
    assert both[0]["rule"] == "track_empty_in_a_taken_shot", "a subject of one load who is an other of another was let through as an other"
    nobody = cap.flag_track("a", "run1", rows, taken, 100, [{"name": "ld", "subject": "b", "others": []}])
    assert nobody[0]["rule"] == "track_empty_in_a_taken_shot", "a subject named by no load was let through as an other"
    cells = np.zeros((2 * N, H // cap.CELL, W // cap.CELL), bool)
    cells[:, 0, 0] = True                                           # the load's region is the top left cell on every frame
    box = rows[N - 1]["track_box"]
    as_other = cap.flag_track("a", "run1", rows, taken, 100, load_of_b, {"ld": (100, cells)}, [W, H])
    assert [(f["rule"], f["level"]) for f in as_other] == [("other_has_no_track_in_a_taken_shot", cap.LEVELS[0])], as_other
    fig = as_other[0]["figures"]
    want = round(float(np.hypot(max(0, box[0] - cap.CELL), max(0, box[1] - cap.CELL))), 1)
    assert fig["an_other_of"] == ["ld"] and fig["each"] == list(range(100 + N, 100 + 2 * N)) and fig["cannot_see"] == cap.OTHER_CANNOT_SEE, fig
    assert fig["beside"] == [{"source_frames": [100 + N, 100 + 2 * N - 1], "after": None, "before": {
        "source_frame": 100 + N - 1, "share_of_the_frame": rows[N - 1]["track_share"], "px": int(round(rows[N - 1]["track_share"] * W * H)), "box": box}}], fig["beside"]
    gap = fig["region_gap_px"]["ld"][0]
    assert gap["least_px"] == want and gap["frames_with_a_region"] == N and gap["their_last_box_is_of_frame"] == 100 + N - 1 and want > 0, (gap, want, box)
    assert "is an OTHER of ld" in as_other[0]["why"] and f"at least {want} px" in as_other[0]["why"] and "CANNOT SEE: whether they are hidden" in as_other[0]["why"], as_other[0]["why"]
    assert cap.flag_track("a", "run1", rows, taken, 100, load_of_b)[0]["figures"]["region_gap_px"]["ld"][0]["least_px"] is None, "a gap was printed with no saved region to measure"
    assert cap.box_gap([32, 32, 40, 40], np.zeros((4, 4), bool), 16) is None and cap.box_gap([0, 0, 20, 20], np.eye(4, dtype=bool), 16) == 0.0
    assert cap.box_gap([40, 8, 48, 12], np.eye(4, dtype=bool)[:1].repeat(4, 0) * np.array([[1, 0, 0, 0]], bool), 16) == 24.0, "a box 24 px right of the first column's cells"
    # the gate and verify: an other's flag blocks neither; the own subject's blocks both
    assert cap.gate_verdict([{**as_other[0], "id": "f001"}], [])["verdict"] == "clear" and cap.gate_verdict([{**own[0], "id": "f001"}], [])["verdict"] == "blocked"
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
        vm.load_window_region = lambda path: {**real(path), "zoom": {"box": [0, 0, 80, 48], "scale": 2.0, "canvas": [W * 2, H * 2]}}
        try:
            refused = cap.read_saved_regions(found, 100, 13, (W, H), render_frames=13)
        finally:
            vm.load_window_region = real
        # (a zoomed window is READ since 2026-10-13, `zoomed_regions` below; one whose zoom is for another canvas is still refused)
        assert refused[0] is None and refused[3]["zoomed"] and "for a canvas of" in refused[3]["refused"], refused[3]
        # files that do not reach the render's end, or leave a gap, are not this render's
        assert cap.read_saved_regions(found, 100, 13, (W, H), render_frames=21)[0] is None, "a longer render, and its region read"
        assert cap.read_saved_regions(found[1:], 100, 13, (W, H))[0] is None, "a window missing from the front, and its region read"
    return ("two windows tile one run's region, each frame from the window whose video holds it; a frame left as the "
            "source across a cut has none and is named; files that do not fit the render are refused")


def zoomed_regions() -> str:
    """A load rendered zoomed in, read onto the canvas where the node lays it. Refused by both readers until 2026-10-13.

    The canvas is 160 by 96. The box is its middle half, 80 by 48 at (32, 16): a zoom of 2. The subject is one cell of
    the canvas, (64..80, 32..48); cut to the box as the node cuts it, it is one token of the zoomed picture."""
    import tempfile
    import torch
    vm = cap._pack("video_mask")
    zoom = {"box": [32, 16, 112, 64], "source_size": [W, H], "canvas": [W, H], "scale": 2.0, "render_region": vm.RENDER_ZOOM_REPLACED, "set_by": "a check"}
    settings = {"composite": "whole region", "feather_pixels": 0, "change_threshold": None, "cuts": [], "grow_pixels": 16,
                "grow_by": "a fixed margin", "replace": "subject", "edge": "latent cells", "zoom": zoom}
    whole = torch.zeros(9, H, W)
    whole[:, 32:48, 64:80] = 1.0
    zoomed = vm.crop_mask(whole, zoom["box"], W, H)
    assert bool((zoomed[0, 32:64, 64:96] == 1).all()) and float(zoomed[0].sum()) == 32 * 32, "the node's own cut of the box is not the one this case was written for"
    tokens = torch.zeros(3, H // 32, W // 32)
    tokens[:, 1, 2] = 1.0
    mask, share, why = cap.zoomed_on_the_canvas(vm, zoomed, tokens, zoom, (W, H))
    assert why is None and bool(((mask > 0.5) == (whole > 0.5)).all()), "the zoomed mask laid back is not the mask it was cut from"
    cells = share > 0
    assert cells.shape == (9, H // 16, W // 16) and cells[:, 2, 4].all() and int(cells[0].sum()) == 1, f"the zoomed token is not the canvas cell it covers: {cells[0].nonzero().tolist()}"
    # the three counts: a token that is one whole cell counts once by each rule; one that lies across cells counts more by the first
    assert cap.zoomed_cells_counted(list(share)) == {"any_pixel": 9, "more_than_half": 9, "every_pixel": 9, "counted_over": "the frames read"}
    across = cap.zoomed_on_the_canvas(vm, zoomed, tokens, {**zoom, "box": [40, 24, 120, 72]}, (W, H))[1]      # the same box, half a cell over and down
    counted = cap.zoomed_cells_counted([across[0]])
    assert counted["any_pixel"] == 4 and counted["every_pixel"] == 0, counted
    # THE ROUND TRIP on a box that is on no grid (100 by 60 at (30, 18), a zoom of 1.6): the node's cut, then this reader's
    # laying back. The line: no more pixels differ than the mask's own one-pixel ring. Reasoned, not measured on a render: each
    # of the two scalings places an edge within a pixel of where it was. What was measured is printed.
    off = {**zoom, "box": [30, 18, 130, 78], "scale": 1.6}
    odd = torch.zeros(3, H, W)
    odd[:, 30:60, 60:90] = 1.0
    back, _, why = cap.zoomed_on_the_canvas(vm, vm.crop_mask(odd, off["box"], W, H), torch.zeros(1, H // 32, W // 32), off, (W, H))
    differ, ring = int(((back[0] > 0.5) != (odd[0] > 0.5)).sum()), 2 * (30 + 30)
    assert why is None and differ <= ring, f"{differ} pixels differ after the round trip, over the ring of {ring}"
    # another canvas than the capture's, and a mask of another size, are refused in words
    assert "for a canvas of [160, 96] and the capture is 320x192" in cap.zoomed_on_the_canvas(vm, zoomed, tokens, zoom, (W * 2, H * 2))[2]
    # the margin: the node's figure is of the zoomed picture; the capture's is canvas pixels, with the node's beside it
    assert cap.canvas_margin(64, {"zoom": zoom}) == (32, 64) and cap.canvas_margin(64, {}) == (64, None) and cap.canvas_margin(None, {"zoom": zoom}) == (None, None)

    def row(writes: int) -> dict:
        return {"number": 1, "first_frame": 0, "frames": 9, "trim": 0, "frames_written": writes, "first_written_frame": 0,
                "last_written_frame": writes - 1, "text": "t", "region_file": "pass_window_1_planned_region.npz", "frames_left_as_source": [],
                "regenerating_share": 0.07}

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "pass_windows"
        folder.mkdir()
        vm.save_window_region(str(folder / "pass_window_1_planned_region.npz"), zoomed, tokens, settings, 16, 0, 0)
        plan = {"plan": "h3 song plan", "version": 2, "zoom": zoom, "source": {"frames": 9, **settings}, "windows": [row(9)]}
        (folder / "pass_plan.json").write_text(json.dumps(plan))
        region, carried, read, how = cap.read_planned_regions(folder / "pass_plan.json", 100, 9, (W, H), at=100)
        assert read.all() and region[:, 2, 4].all() and int(region.sum()) == 9, "a zoomed plan's region is not on the canvas cell it covers"
        assert (carried == (whole.numpy() > 0.5)).all() and how["zoom"] == zoom and "up to one cell more" in how["zoomed_cells_are"], how.get("zoom")
        assert how["zoomed_cells_counted"]["any_pixel"] == how["zoomed_cells_counted"]["every_pixel"] == 9, how["zoomed_cells_counted"]
        # the same plan read as if it were of the whole frame would put the region on the zoomed picture's own cells: not there
        assert not region[:, 2:4, 4:6].all(), "the region is on the zoomed picture's cells, not laid back"
        # a window that says another box than the plan's, and one that says none, are not this run's
        vm.save_window_region(str(folder / "pass_window_1_planned_region.npz"), zoomed, tokens, {**settings, "zoom": off}, 16, 0, 0)
        assert "one box for the whole load" in cap.read_planned_regions(folder / "pass_plan.json", 100, 9, (W, H), at=100)[3]["refused"]
        vm.save_window_region(str(folder / "pass_window_1_planned_region.npz"), zoomed, tokens, {**settings, "zoom": None}, 16, 0, 0)
        assert "one box for the whole load" in cap.read_planned_regions(folder / "pass_plan.json", 100, 9, (W, H), at=100)[3]["refused"]
        # a zoomed RENDER's windows, read the same way; and a second window of another box refused
        render = str(Path(tmp) / "pass_00001.mp4")
        vm.save_window_region(str(folder / "pass_window_1_region.npz"), zoomed, tokens, settings, 16, 0, 0)
        found = cap.window_region_files(render)
        region, carried, read, how = cap.read_saved_regions(found, 100, 9, (W, H), render_frames=9)
        assert read.all() and region[:, 2, 4].all() and int(region.sum()) == 9 and (carried == (whole.numpy() > 0.5)).all() and how["zoom"] == zoom, how
        vm.save_window_region(str(folder / "pass_window_2_region.npz"), zoomed, tokens, {**settings, "zoom": None}, 16, 9, 0)
        mixed = cap.read_saved_regions(cap.window_region_files(render), 100, 18, (W, H), render_frames=18)
        assert mixed[0] is None and mixed[3]["zoomed"] and "one box for the whole load" in mixed[3]["refused"], mixed[3]
        # the review of a zoomed run is of the box: said from a window's own file, or from a graph that asks for a zoom when no window saved a region
        (folder / "pass_window_2_region.npz").unlink()
        asks = {"1": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"render_region": vm.RENDER_ZOOM_REPLACED}}}
        assert "its review is of the box" in cap.review_is_of_a_box(found, None, vm) and "cannot be read" in cap.review_is_of_a_box([], asks, vm)
        whole_frame = {"1": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"render_region": vm.RENDER_WHOLE}}}
        assert cap.review_is_of_a_box([], whole_frame, vm) is None and cap.review_is_of_a_box([], None, vm) is None
        vm.save_window_region(str(folder / "pass_window_1_region.npz"), zoomed, tokens, {**settings, "zoom": None}, 16, 0, 0)
        assert cap.review_is_of_a_box(found, asks, vm) is None, "a window that says it was not zoomed was refused on the graph's asking"
    return (f"a zoomed plan and a zoomed render's windows are read onto the canvas where the node lays them; the node's cut of a box laid back is the mask "
            f"it was cut from (on a box on no grid {differ} pixel(s) differ, of a ring of {ring}); another box, another canvas and a zoomed run's review are refused")


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


def a_part_held_on_a_persons_word() -> str:
    """`word=` (bench/shot_to_load.py job --hold-part; the lead's ruling of 2026-10-15 on 2823b's 3098-3104 and 5935's 6008 and 6020):
    a frame the grade left empty takes the nearest graded part, moved along the line between the parts either side and
    cut to the track, at any distance and at the load's edge, never the floor's own speck; a standing part is untouched; a frame with no
    track stays empty and says so."""
    n = 30
    face = np.zeros((n, 96, 160), bool)
    for f in range(n):
        face[f, 20:40 + (f > 16), 30 + f:50 + f] = True                  # one px to the right a frame; a row taller from frame 17
    track = rect(10, 5, 150, 90, n)
    track[25] = False                                                   # frame 25: the subject is not tracked
    classes = np.where(face, 2, 0).astype(np.uint8)
    speck = rect(90, 60, 95, 65, n)                                     # 25 px, on the part's class: under the floor
    saved = face.copy()
    for f in (5, 6, 7):                                                 # the floor's specks, where the part model lost her face
        classes[f] = np.where(speck[f], 2, 0)
        saved[f] = speck[f]
    for f in list(range(10, 23)) + [25, 29]:                            # a run of thirteen lost; the untracked frame; the last frame
        classes[f] = 0
        saved[f] = False
    word = {5, 6, 7, 27, 29, 25} | set(range(10, 23))
    without, _ = cap.grade_holds(saved, saved, classes, [2], least_px=100)
    held, said = cap.grade_holds(saved, saved, classes, [2], least_px=100, word=word, track=track)
    # THE CASE THAT GOES RED when the word gives back the floor's speck (5935's 5995 at 17 px), leaves the ends of a long run empty, or
    # takes a shape that does not follow the part
    assert not without[5:8].any() and not without[10:23].any(), "the fixture's frames were not emptied without the word"
    # each takes its NEAREST part's shape: 10-16 the short one of frame 9, 17-22 the tall one of frame 23 (the shapes differ either side)
    for f in [5, 6, 7] + list(range(10, 23)):
        nearest = face[f] if f < 10 or f > 22 else np.roll(face[9 if f - 9 <= 23 - f else 23], f - (9 if f - 9 <= 23 - f else 23), axis=-1)
        assert (held[f] == nearest).all() and said[f]["did"] == "filled on a person's word", (f, said.get(f), int(held[f].sum()))
        assert said[f]["px_on_the_track"] == said[f]["px"], said[f]
    # the load's last frame has a part on one side only: it takes that part where it is, as held_parts does (no line to follow)
    assert (held[29] == face[28]).all() and said[29]["did"] == "filled on a person's word", said[29]
    assert said[10]["from_frames"] == [9] and said[22]["from_frames"] == [23] and said[29]["from_frames"] == [28], (said[10], said[22], said[29])
    assert not held[25].any() and said[25]["did"] == "left empty: the word not held" and said[25]["word_not_held"] == cap.WORD_NOT_HELD, said[25]
    assert 27 not in said and (held[27] == face[27]).all(), "a frame whose part stands was graded for the word"
    # a load with no part on any frame: nothing to hold from, said
    none, said_none = cap.grade_holds(np.zeros_like(saved), np.zeros_like(saved), np.zeros_like(classes), [2], word={3, 4}, track=track)
    assert not none.any() and said_none[3]["word_not_held"] == cap.WORD_NOT_HELD and said_none[4]["did"] == "left empty: the word not held", said_none
    try:
        cap.grade_holds(saved, saved, classes, [2], word={5})
    except ValueError as e:
        assert "give `track`" in str(e), e
    else:
        raise AssertionError("a word was held with no track to cut it to")
    # the files step hands the mask's word and the track to the grade, with the lost frames as before
    every = saved.copy()
    given = cap.grade_given({"word": "3101-3102"}, {0: {"mostly": "nothing labelled"}, 1: {"mostly": "her own classes"}}, every, 3090, 20, track)
    assert given["word"] == {11, 12} and given["lost"] == {0} and given["lost_fill"] is every and given["track"] is track, given
    assert cap.word_frames("3101-3104+3098", 3090, 20) == {8, 11, 12, 13, 14}
    for spans, words in (("3105-3112", "not inside the capture's frames 3090-3109"), ("3104-3098", "runs backwards"),
                         ("3098-3100-3104", "is not FRAME or FIRST-LAST"), ("31x8", "is not FRAME or FIRST-LAST")):
        try:
            cap.word_frames(spans, 3090, 20)
        except SystemExit as stop:
            assert words in str(stop), stop
        else:
            raise AssertionError(f"word={spans} was read")
    cap.word_refused("a", {"word": "1-2", "classes": "c.mkv"}, True)
    cap.word_refused("a", {}, False)
    for m, has, covers, words in (({"word": "1-2", "classes": "c.mkv"}, False, True, "no part"),
                                  ({"word": "1-2", "classes": "c.mkv", "grade": "off"}, True, True, "grade=off"),
                                  ({"word": "1-2"}, True, True, "no class map"),
                                  ({"word": "1-2", "classes": "c.mkv"}, True, False, "does not cover the whole span"),
                                  ({"word": "1-4", "classes": "c.mkv", "drop": "3-9"}, True, True, "both name 3-4"),
                                  ({"word": "31x8", "classes": "c.mkv", "drop": "3-9"}, True, True, "is not FRAME or FIRST-LAST")):
        try:
            cap.word_refused("a", m, has, covers)
        except SystemExit as stop:
            assert words in str(stop), stop
        else:
            raise AssertionError(f"word= taken with {words}")
    return ("a frame named by a person's word and left empty takes the nearest graded part, moved with the part and cut to "
            "the track, on a run of any length and at the load's edge, never the floor's speck; a standing part is untouched; an untracked frame "
            "stays empty with WORD_NOT_HELD; the word reaches the grade with the track; spans parsed strictly; refused with no grade, an "
            "uncovered mask, or drop= on the same frames")


def a_short_fill_where_nothing_was_labelled() -> str:
    """751b's 761 (2026-10-15): the class map labelled nothing on her face for one frame, a speck under the floor was left, and the
    floor emptied the frame between two with her face. A run of at most SHORT_FILL such frames, a part either side, takes the fill;
    nothing else changes."""
    n = 10
    face = rect(40, 20, 60, 40, n)                                      # 400 px a frame
    classes = np.where(face, 2, 0).astype(np.uint8)
    speck = rect(90, 60, 95, 65, n)                                     # 25 px, off her face
    saved = face.copy()
    for f in (3, 6, 7, 8, 0):                                           # frames where the part model labelled nothing
        classes[f] = 0
        classes[f][speck[f]] = 2                                        # the speck lies on the part's class
        saved[f] = speck[f]
    fill = face.copy()
    lost = {3, 6, 7, 8, 0}
    before, said_before = cap.grade_holds(saved, fill, classes, [2], least_px=100)
    after, said = cap.grade_holds(saved, fill, classes, [2], least_px=100, lost=lost)
    # THE CASE THAT GOES RED when 761 is left emptied, or a longer run or one at the load's edge is filled
    assert not before[3].any() and said_before[3]["did"] == "emptied", said_before[3]
    assert (after[3] == face[3]).all() and said[3]["did"] == "filled where nothing was labelled" and "least_px" not in said[3], said[3]
    assert not after[6:9].any() and not after[0].any(), "a run of three, or one at the load's edge, was filled"
    # a frame whose saved part stands (805b's 805, 821, 833) is untouched though nothing was labelled under the fill
    part = rect(90, 60, 102, 72, n)                                     # 144 px, on the part's class: above the floor, and kept
    own = np.where(face, 2, 0).astype(np.uint8)
    own[3] = np.where(part[3], 2, 0)                                     # frame 3: nothing labelled where the fill would go, her part kept
    stands = face.copy()
    stands[3] = part[3]
    kept, said_kept = cap.grade_holds(stands, fill, own, [2], least_px=100, lost={3})
    # THE CASE THAT GOES RED when a saved part that stands is replaced (805b's 805, 821 and 833), its neighbours both with a part
    assert kept[2].any() and kept[4].any() and (kept[3] == part[3]).all() and said_kept[3]["did"].startswith("saved part kept"), said_kept.get(3)
    # a neighbour with no part: not filled; the last frame: not filled; a fill under the floor: not taken
    alone, no_fill = saved.copy(), fill.copy()
    alone[2], no_fill[2] = False, False                                  # frame 2: no part, and none offered
    lone, said_lone = cap.grade_holds(alone, no_fill, classes, [2], least_px=100, lost={3})
    assert not lone[2].any(), said_lone.get(2)
    assert not lone[3].any(), "a frame was filled with no part on the frame before it"
    end_saved, end_classes = face.copy(), np.where(face, 2, 0).astype(np.uint8)
    end_saved[n - 1], end_classes[n - 1] = speck[n - 1], np.where(speck[n - 1], 2, 0)
    end, _ = cap.grade_holds(end_saved, fill, end_classes, [2], least_px=100, lost={n - 1})
    assert not end[n - 1].any(), "the load's last frame was filled"
    tiny, _ = cap.grade_holds(saved, speck, classes, [2], least_px=100, lost={3}, lost_fill=speck)
    assert not tiny[3].any(), "a fill under the floor was taken"
    # the fill's shape is the unlimited one (`lost_fill`), not a hold's that put the speck back
    wide, said_wide = cap.grade_holds(saved, saved, classes, [2], least_px=100, lost={3}, lost_fill=face)
    assert (wide[3] == face[3]).all() and said_wide[3]["px"] == 400, said_wide[3]
    return ("a frame emptied between two with a part, where nothing was labelled, takes the fill; a run of three, one at the load's edge and a "
            "frame whose saved part stands are left as they were")


def the_floor_at_the_size_it_renders() -> str:
    """The starting floor held at the size the load renders at (2026-10-14): a load the node zooms by Z gives the model its part
    Z squared larger, so the floor on the canvas is PART_LEAST_PX / Z^2, Z the node's own zoom_box on the part's boxes; least= is
    not scaled; zoom=off (the job's whole_frame) is not; at on-plan, a run whose zoom differs says the frames that would change."""
    import torch
    W_, H_ = 1024, 768
    vm = cap._pack("video_mask")
    small = np.zeros((20, H_, W_), bool)
    small[:, 300:340, 500:530] = True                                  # a far face, 1200 px: under the canvas floor
    small[5, 300:310, 500:530] = False
    small[5, 300:302, 500:530] = True                                  # one frame of 60 px
    least, by, z = cap.starting_floor(small, (W_, H_), whole_frame=False)
    box, why = vm.zoom_box(torch.tensor(np.tile([500, 300, 530, 340], (20, 1))), W_, H_, W_, H_)
    # THE CASE THAT GOES RED when the zoom is not the node's own, or the floor is not the starting value at the zoom's size
    assert z["box"] == list(box) and z["scale"] == round(W_ / (box[2] - box[0]), 4) and z["scale"] == 2.0, (z, box, why)
    assert least == cap.PART_LEAST_PX // 4 == cap.floor_on_the_canvas(2.0) and "zoomed 2.00x" in by and f"{least} px on the canvas" in by, (least, by)
    # the grade under it keeps the far face and empties the 60 px frame; under the canvas value it would empty all of them
    kept, said = cap.grade_holds(small, small, np.ones((20, H_, W_), np.uint8), [1], least)
    assert kept[0].any() and not kept[5].any() and said[5]["least_px"] == least, said.get(5)
    assert not cap.grade_holds(small, small, np.ones((20, H_, W_), np.uint8), [1], cap.PART_LEAST_PX)[0].any(), "the control: the canvas floor empties every frame"
    # THE CASE THAT GOES RED when a whole-frame load's floor is scaled
    whole = cap.starting_floor(small, (W_, H_), whole_frame=True)
    assert whole[0] == cap.PART_LEAST_PX and "zoom=off" in whole[1] and whole[2] is None, whole
    wide = np.zeros((3, H_, W_), bool)
    wide[:, 10:760, 10:1010] = True                                     # across the frame: the node does not zoom
    w = cap.starting_floor(wide, (W_, H_), whole_frame=False)
    assert w[0] == cap.PART_LEAST_PX and w[2]["box"] is None and "under 1.25 times" in w[1], w
    # at on-plan: the zoom the floor used against the run's
    px = np.array([5000, 2000, 1100, 900, 0, 3000])
    assert cap.floor_against_the_run(px, 100, np.ones(6, bool), 2.0, 2.0) is None
    ag = cap.floor_against_the_run(px, 100, np.ones(6, bool), 2.0, 1.33)      # used 1024 px; at 1.33x 2316 px
    # THE CASE THAT GOES RED when a run's other zoom is not said with the frames whose emptying it would change
    assert ag["floor_px_used"] == 1024 and ag["floor_px_at_the_run_zoom"] == cap.floor_on_the_canvas(1.33) \
        and ag["frames_whose_emptying_would_change"] == [[101, 102]] and ag["frames"] == 2 and ag["would"].startswith("be emptied"), ag
    back = cap.floor_against_the_run(px, 100, np.array([1, 0, 1, 1, 1, 1], bool), 1.0, 2.0)
    assert back["frames_whose_emptying_would_change"] == [[102, 102], [105, 105]] and back["would"].startswith("be kept"), back
    return ("the starting floor is PART_LEAST_PX at the zoom the node's zoom_box gives the part's boxes, so a far face is kept where the canvas floor "
            "emptied it; not scaled on the whole frame, by zoom=off or where the node will not zoom; a run whose zoom differs names the frames")


def the_floor_against_the_clip() -> str:
    """The floor that empties a small part is a starting value: a capture sizes the part on its own frames, says
    where the floor falls among them and what it emptied, and takes another floor from `least=`."""
    import tempfile
    parts = rect(40, 20, 60, 40, 6)                  # 400 px on frames 0 to 3
    parts[4] = False
    parts[4, 20:23, 40:44] = True                    # frame 4: a sliver of 12 px
    parts[5] = False                                 # frame 5: no part, which is not a size
    found = cap.part_sizes(parts, 100, 100, cap.LEAST_GIVEN, emptied=[4])
    assert (found["frames"], found["frames_with_a_part"]) == (6, 5) and found["px"] == {
        "least": 12, "p10": 167, "p25": 400, "median": 400, "p75": 400, "p90": 400, "most": 400}, found
    assert (found["floor_px"], found["floor_set_by"], found["frames_under_the_floor"], found["source_frames_under_the_floor"]) == \
        (100, cap.LEAST_GIVEN, 1, [[104, 104]]), found
    # the counts under a quarter, a half, one and two times the STARTING value are always there, and under the floor in use
    assert found["frames_under_px"] == {"100": 1, **{str(cap.PART_LEAST_PX * k // 4): 5 for k in (1, 2, 4, 8)}}, found["frames_under_px"]
    assert found["emptied_by_the_floor"] == {"frames": 1, "source_frames": [[104, 104]]} and found["cannot_see"] == cap.SIZES_CANNOT_SEE
    said = "\n".join(cap.size_lines("a's part", found))
    for words in ("on 5 of 6 frames", "least 12", "median 400", "the floor is 100 px", cap.LEAST_GIVEN, "source frames 104",
                  "EMPTIED BY THE FLOOR in the part to load: 1 frame(s)", "least=PX", "cannot see"):
        assert words in said, f"the printed sizes lack {words!r}:\n{said}"
    # the starting value says of itself that it is not this clip's; a floor of 0 is under nothing and empties nothing
    start = cap.part_sizes(parts, 100, cap.PART_LEAST_PX, cap.LEAST_STARTING, emptied=[0, 1, 2, 3, 4])
    assert start["frames_under_the_floor"] == 5 and "not set from this clip" in "\n".join(cap.size_lines("a", start)), start
    none = cap.part_sizes(parts, 100, 0, cap.LEAST_GIVEN, emptied=[])
    assert none["frames_under_the_floor"] == 0 and "0" not in none["frames_under_px"] and "emptied no frame" in "\n".join(cap.size_lines("a", none))
    # a sighting whose part was not graded says why nothing was emptied; a part on no frame has no sizes
    off = cap.part_sizes(parts, 100, cap.PART_LEAST_PX, cap.LEAST_STARTING, not_applied="grade=off on this --mask")
    assert off["emptied_by_the_floor"] is None and "nothing was emptied by size on this sighting: grade=off" in "\n".join(cap.size_lines("a", off))
    empty = cap.part_sizes(np.zeros_like(parts), 100, cap.PART_LEAST_PX, cap.LEAST_STARTING)
    assert empty["px"] is None and "on no frame" in cap.size_lines("a", empty)[0]
    # the floor a capture is given is the one the grade uses (`grade_holds`): 100 empties the sliver and leaves the face
    classes = np.where(rect(40, 20, 60, 40, 6), 2, 0).astype(np.uint8)
    held, decided = cap.grade_holds(parts, parts, classes, [2], least_px=100)
    assert held[:4].all(axis=(1, 2)).tolist() == [False] * 4 and held[:4].any(axis=(1, 2)).all() and not held[4].any(), "a floor of 100 px"
    assert [f for f, d in decided.items() if "least_px" in d] == [4] and decided[4]["px"] == 12, decided
    # `sizes` on a folder: a capture that recorded its sizes is read; an earlier one is measured again from its own
    # masks, with the starting value and the frames its grades say were emptied for size
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "2026-01-01_x"
        (folder / "subjects" / "a").mkdir(parents=True)
        np.savez_compressed(folder / "subjects" / "a" / "masks__p.npz", parts=np.packbits(parts, axis=-1), track=np.packbits(parts, axis=-1))
        manifest = {"size": [W, H], "first_frame": 100, "frames": 6, "runs": [],
                    "subjects": [{"label": "a", "sightings": [{"by": "p", "parts": "a_parts.mkv"}]},
                                 {"label": "b", "sightings": [{"by": "p", "parts": None}]}]}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        (folder / "subjects" / "a" / "part_grades.json").write_text(json.dumps({"frames": [
            {"source_frame": 104, "did": "emptied", "why": "too small to be drawn as the part", "px": 12, "least_px": cap.PART_LEAST_PX},
            {"source_frame": 101, "did": "filled"}]}))
        (where, again, how), = cap.sizes_of(folder)
        assert "2026-01-01_x a (by p)" in where and "source frames 100-105" in where and "measured now" in how, (where, how)
        assert again["px"] == found["px"] and again["floor_px"] == cap.PART_LEAST_PX and again["floor_set_by"] == cap.LEAST_STARTING \
            and again["emptied_by_the_floor"] == {"frames": 1, "source_frames": [[104, 104]]}, again
        manifest["subjects"][0]["sightings"][0]["part_sizes"] = found
        (folder / "manifest.json").write_text(json.dumps(manifest))
        (_, read, how), = cap.sizes_of(folder)
        assert read == found and "as the capture recorded it" in how, how
    return ("a part's size on its own frames, the floor and who set it, the frames under it and the frames it emptied are figures a "
            "capture prints and records; the starting value says it is not the clip's; `least=` is the floor the grade uses and 0 "
            "empties nothing; a capture that exists is sized from its record, or from its own masks when it has none")


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
    # A LOAD'S OWN PART (2026-10-14: the first load to carry a held part failed here): made a day later from the preview's saved part, it is
    # held to that part; one made from a file of another run is not
    derived = by(cap.verify_checks(manifest({**good, "parts": {**video("held.mkv", "23:59:00"), "made_from": video("r.mkv", "13:45:50")}}, [run]), []))
    assert derived["inputs_written_together"]["ok"] is True and "held.mkv was made by the job step from r.mkv" in derived["inputs_written_together"]["why"], derived
    stray = by(cap.verify_checks(manifest({**good, "parts": {**video("held.mkv", "23:59:00"), "made_from": video("r.mkv", "09:17:00")}}, [run]), []))
    assert stray["inputs_written_together"]["ok"] is False and "made from r.mkv at 09:17:00" in stray["inputs_written_together"]["why"], stray
    # A TRACK COPIED ON A PERSON'S WORD (`track_from=`, 2026-10-15: second_hfp_0604f failed here, its copy written a day after the preview):
    # held to the preview track it was cut from, as a load's own part is
    copied = by(cap.verify_checks(manifest({**good, "track": {**video("t_not.mkv", "23:59:00"), "made_from": video("t.mkv", "13:44:10")}}, [run]), []))
    # THE CASE THAT GOES RED when a track copy is held to its own write time, not to the preview track it came from
    assert copied["inputs_written_together"]["ok"] is True and "t_not.mkv was made by the job step from t.mkv" in copied["inputs_written_together"]["why"], copied
    # a pose table from a pass of its own does not make the masks two runs
    posed = by(cap.verify_checks(manifest({**good, "pose": {"file": "p.json", "written": "2026-10-10T09:00:00"}}, [run]), []))
    assert posed["inputs_written_together"]["ok"] is True
    # A MASK'S `at` AGAINST THE PREVIEW'S OWN JOB: a sighting read from another source frame than its preview's files
    # begin on is a hard failure with the number of frames it is off; one read from the right frame passes (the
    # control); one whose preview's job is not found is not checked, by name, and is neither
    off = manifest(good, [run], at=104)
    (wrong,), none = cap.at_checks(off, {"p": (100, "2026-01-01_p_job")})
    assert wrong["ok"] is False and "at=104" in wrong["why"] and "begin on source frame 100" in wrong["why"] and "4 frame(s) off the picture" in wrong["why"] and none == [], wrong
    (right,), none = cap.at_checks(manifest(good, [run]), {"p": (100, "2026-01-01_p_job")})
    assert right["ok"] is True and none == [], right
    shots_off = manifest(good, [run])
    shots_off["subjects"][0]["sightings"][0]["shots_first_source_frame"] = 108
    assert cap.at_checks(shots_off, {"p": (100, "j")})[0][0]["ok"] is False, "a shot table read from another frame than the preview's first passed"
    made, unchecked = cap.at_checks(off, {"p": (None, "no job folder data/<day>_p_job holds a job.json")})
    assert made == [] and unchecked == [{"check": "at_is_the_previews_first_frame", "about": "a by p", "why": "no job folder data/<day>_p_job holds a job.json"}], unchecked
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        assert cap.preview_first_of("p", data) == (None, "no job folder data/<day>_p_job holds a job.json")
        (data / "2026-01-01_p_job").mkdir()
        (data / "2026-01-01_p_job" / "job.json").write_text(json.dumps({"job": {"first": 100}}))
        assert cap.preview_first_of("p", data) == (100, "2026-01-01_p_job")
        (data / "2026-01-02_p_job").mkdir()
        (data / "2026-01-02_p_job" / "job.json").write_text(json.dumps({"job": {"first": 300}}))
        assert cap.preview_first_of("p", data) == (None, "2 job folders are named for p"), "two previews of one name: which one is not said, so nothing is checked"
    small = by(cap.verify_checks(manifest({**good, "parts": video("r.mkv", "13:45:50", size=(W // 2, H // 2))}, [run]), []))
    assert small["inputs_at_the_canvas"]["ok"] is False and small["inputs_written_together"]["ok"] is True
    short = by(cap.verify_checks(manifest({**good, "track": video("t.mkv", "13:44:10", frames=12)}, [run]), []))
    assert short["inputs_reach_the_span"]["ok"] is False and "111" in short["inputs_reach_the_span"]["why"], short
    late = by(cap.verify_checks(manifest(good, [run], at=104), []))
    assert late["inputs_reach_the_span"]["ok"] is False, "masks that start after the span's first frame were let through"
    old = by(cap.verify_checks(manifest(None, [run]), []))
    assert old["inputs_written_together"]["ok"] is None, "a capture with no recorded inputs was passed"
    # the files a --mask names, as the manifest keeps them: a file the job step made today from a preview file carries that file as
    # `made_from`, a load's own part (`parts_from=`) and another subject's track copied with frames emptied (`track_from=`, 2026-10-15)
    fake = lambda path: {"file": Path(path).name}                                                              # noqa: E731
    got = cap.sighting_inputs(cap.spec("b:track=/o/copy.mkv,track_from=/o/pv_b_track.mkv,shots=/o/s.json,parts=/o/own.mkv,parts_from=/o/p.mkv")[1], fake)
    # THE CASE THAT GOES RED when a track copied on a person's word is kept with nothing of the preview track it came from
    assert got["track"] == {"file": "copy.mkv", "made_from": {"file": "pv_b_track.mkv"}} and got["parts"]["made_from"] == {"file": "p.mkv"} \
        and "made_from" not in got["shots"] and set(got) == {"track", "parts", "shots"}, got
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
    as_other = {"rule": "other_has_no_track_in_a_taken_shot", "subject": "a", "seen_by": "p", "source_frames": [[104, 105]], "why": "an OTHER of ld",
                "figures": {"frames": 2, "each": [104, 105], "an_other_of": ["ld"]}}
    passed = by(cap.verify_checks(manifest(good, [run]), [as_other]))["track_where_the_shot_table_says_taken"]
    assert passed["ok"] is True and passed["an_other_of"] == ["ld"] and passed["why"] == "an OTHER of ld" and passed["source_frames"] == [[104, 105]], passed
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
        # a load of two subjects reaches the capture as its first subject's, with the second beside it for the plan
        path.write_text(json.dumps({"loads": [{"name": "ab", "first": 1, "frames": 9, "margin": 16, "subjects": [
            {"subject": "a", "region": "face and hair", "others": ["c"]}, {"subject": "b", "region": "whole"}]}]}))
        (name, spec), = cap.loads_from_file(str(path))
        assert name == "ab" and spec["subject"] == "a" and spec["others"] == "c" and spec["carried"] == "held", spec
        assert json.loads(spec["also"]) == {"subject": "b", "region": "whole", "carried": "track"}, spec["also"]
        # THE PLAN'S RUN OF A PAIR (`load_chain`'s capture on the plan): the --run read from the node's plan and the file's load of the
        # same name are ONE plan, the --run's, carrying the file's `also` and nothing else of it; a run that rendered is not a plan; a load
        # no --run names stays a plan of its own; a --run of one subject is as it was
        run = ("ab", {"preview": "/w", "subject": "a", "at": "1"})
        got = cap.plans_with_the_files([run, ("done", {"render": "/r.mp4", "subject": "a"})], [(name, spec), ("other", {"subject": "a", "at": "5"})])
        assert got == [("other", {"subject": "a", "at": "5"}), ("ab", {**run[1], "also": spec["also"]})], got
        assert cap.plans_with_the_files([run], [("other", {"subject": "a"})]) == [("other", {"subject": "a"}), run]
    # a pair's second subject's own things under a name the first's holds (`hands_lead`): its own words, with whose, never another person
    assert cap.keep_out_words("hands") == "the subject's hands and arms" and cap.keep_out_words("hands_lead") == "lead's hands and arms", cap.keep_out_words("hands_lead")
    assert cap.keep_out_words("lead") == cap.keep_out_words("third_party") == "another tracked subject"
    # the second subject's share of what a two-subject load carries: its track when it is replaced whole, else its held part, else its part
    track, part, held = rect(96, 16, 144, 80), rect(104, 24, 120, 40), rect(104, 24, 124, 44)
    assert (cap.second_carried({"subject": "b", "carried": "track"}, track, part, held) == track).all()
    assert (cap.second_carried({"subject": "b", "carried": "held"}, track, part, held) == held).all()
    assert (cap.second_carried({"subject": "b", "carried": "held"}, track, part, None) == part).all()
    try:
        cap.second_carried({"subject": "b", "region": "face", "carried": "held"}, track, None, None)
    except SystemExit as stop:
        assert "no part mask" in str(stop), stop
    else:
        raise AssertionError("a second subject replaced by a part it has no mask of was planned")
    # the plan is worked out from both parts as one mask, as the node is given them: no cell holding either is left out
    first_part = rect(32, 32, 48, 48)
    union = cap.planned_region(first_part | held, [], MARGIN, _grow())
    alone = cap.planned_region(first_part, [], MARGIN, _grow())
    assert union[cap.cells_any(first_part)].all() and union[cap.cells_any(held)].all() and not alone[cap.cells_any(held)].any(), \
        "the plan of two subjects' parts did not cover both, or the plan of one covered the other's"
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
    # A WINDOW ON THE MODEL'S OWN GRID THAT IS NOT A CHAINED LENGTH (2026-10-13: a load one window holds takes the shortest
    # length of the video grid that holds it): a 39-frame window, twelve latent steps, holding a 24-frame load. The reader
    # takes the window's length from the mask it is handed and its steps from the node's own run lengths, and knows no list
    steps = vm.run_lengths(12)
    assert sum(steps) == 39, steps
    short_mask, short_tokens = torch.zeros(39, H, W), torch.zeros(12, H // 32, W // 32)
    short_mask[:, 32:64, 32:64] = 1.0                         # the region open over the held frames, as the node holds it
    short_tokens[:, 1, 1] = 1.0
    short = cap.held_tail_facts(short_mask, short_tokens, 15, 100, vm.run_lengths)
    edge = next(sum(steps[:k]) for k in range(len(steps)) if sum(steps[:k]) <= 24 < sum(steps[:k + 1]))
    assert (short["frames"], short["held_frames"], short["held_share"], short["held_frames_shown_clean"]) == (39, 15, round(15 / 39, 3), 0), short
    assert short["step_with_real_and_held_source_frames"] == (None if edge == 24 else [100 + edge, 123]) and short["subject_on_the_last_real_frame"], short
    short_tokens[-3:] = 0.0                                    # the same window with its last three steps given no region: those held frames are shown clean
    assert cap.held_tail_facts(short_mask, short_tokens, 15, 100, vm.run_lengths)["held_frames_shown_clean"] == sum(steps[-3:]), "a short window's clean held frames were not counted"
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
        # a plan that already noises its kept frames is told so, not told to noise them (the cold reader of 2026-10-15)
        assert far[0]["why"].endswith("or noise the kept ones (`context_noise`)"), far[0]["why"]
        noised = cap.flag_kept({**m, "runs": [{**m["runs"][0], "planned_context_noise": 0.8}]}, folder)
        # THE CASE THAT GOES RED when the plan's noise is not read into the advice
        assert noised[0]["why"].endswith("or noise them more: the plan shows them at noise 0.8 (`context_noise`)"), noised[0]["why"]
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
            # a load of two subjects: the SECOND subject's small face is named too (a fact about a subject is read once per subject)
            pair = manifest(24)
            (folder / "subjects" / "z").mkdir(parents=True, exist_ok=True)
            big = rect(100, 8, 140, 8 + int(cap.FACE_TOKENS * 16), 6)
            np.savez_compressed(folder / "subjects" / "z" / "masks__p.npz", parts=np.packbits(big, axis=-1), track=np.packbits(big, axis=-1))
            pair["subjects"].insert(0, {"label": "z", "sightings": [{"by": "p", "part_is_made_of": ["Face_Neck"]}]})
            pair["runs"] = [{"name": "r", "subject": "z"}]
            assert cap.flag_face_size(pair, folder) == [], "the control: told one subject, the other's face is nobody's to size"
            pair["runs"] = [{"name": "r", "subject": "z", "subjects": ["z", "a"]}]
            assert [f["subject"] for f in cap.flag_face_size(pair, folder)] == ["a"], cap.flag_face_size(pair, folder)
        finally:
            cap.TOKEN_PX = token
    # a part in pieces: the head, and a sliver somewhere else that carries the part's class
    head = rect(40, 8, 80, 56, 4)
    sliver = rect(120, 70, 140, 76, 4)
    sliver[3] = False
    piece, stray = cap.largest_piece(head | sliver)
    assert (piece == head).all() and stray.tolist() == [120, 120, 120, 0], stray.tolist()
    assert cap.largest_piece(np.zeros((2, H, W), bool))[1].tolist() == [0, 0]
    # what the option took away is said a frame at a time, with what it cannot see: the flag is silent once it is set
    took = cap.pieces_taken(stray, 949)
    assert (took["frames"], took["frames_read"], took["px_most"], took["px_by_source_frame"]) == (3, 4, 120, [[949, 120], [950, 120], [951, 120]]), took
    said = cap.pieces_lines("a", took)
    assert "TOOK PIXELS OF a's PART AWAY on 3 of 4 frame(s), up to 120 px" in said[0] and "949: 120, 950: 120, 951: 120" in said[0] \
        and said[1].strip().startswith("cannot see: whether a piece taken away was the part"), said
    none = cap.pieces_lines("a", cap.pieces_taken(np.zeros(4, np.int64), 949))
    assert len(none) == 1 and "took nothing away" in none[0], none
    return ("a face under the line in tokens is named, with its height; a face at the line, a part that is not a face, and a "
            "subject no run replaces are not; a part's pieces away from its largest are counted and can be dropped, and what "
            "dropping them took away is said a frame at a time")


def no_pieces_on_a_head() -> str:
    """`pieces=largest` refused at the files step on a part made of a face or hair, whoever wrote the job file or command."""
    import tempfile
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    part = rect(40, 8, 80, 56)

    def run(cls: str | None) -> str:
        cmap = np.zeros((N, H, W), np.uint8)
        if cls:
            cmap[part] = names.index(cls)
        stood_in = {k: getattr(cap, k) for k in ("probe", "read_mask", "read_classes", "largest_piece")}

        class Reached(Exception):
            pass

        def reached(_parts):
            raise Reached
        cap.probe, cap.read_mask, cap.read_classes = (lambda _p: (W, H, N)), (lambda *_a: (part, np.ones(N, bool))), (lambda *_a: cmap)
        cap.largest_piece = reached
        spec = {"track": "t.mkv", "parts": "p.mkv", "pieces": "largest", **({"classes": "c.mkv"} if cls else {})}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cap._files(argparse.Namespace(first=100, frames=N, name="x", date="2026-01-01", source="s.mp4"), Path(tmp), [("a", spec)], [], [])
        except Reached:
            return "the drop"
        except SystemExit as e:
            return str(e)
        finally:
            for k, v in stood_in.items():
                setattr(cap, k, v)
        raise AssertionError("the files step neither stopped nor reached the drop")
    # THE CASES THAT GO RED when the files step reads a face or hair with the largest piece
    for cls in ("Face_Neck", "Hair"):
        said = run(cls)
        assert said.startswith(f"subject 'a': pieces=largest on a part made of {cls}") and "e6993b5e" in said, said
    assert run("Upper_Clothing") == "the drop", "a part of clothing was refused the largest piece"
    assert "pieces=largest with no classes= video" in run(None), run(None)
    return "the largest piece refused on a face and on hair at the files step, before a piece is dropped; a part of clothing reaches it; no class map stops"


def no_keep_out_inside_her_track() -> str:
    """The owner, 2026-10-16: nothing puts the original back inside a person's region. A keep-out's cells given back that lie
    inside her own track are counted and named, and block the gate with no override; another person's do not."""
    import argparse
    import contextlib
    import io
    import tempfile
    body = rect(16, 16, 96, 96)                          # her track: her head (A, the carried part) and the body round it
    keep = rect(64, 32, 80, 64)                          # a keep-out beside her head, inside her track and the margin's reach
    run = _run(keep, True)
    row = cap.cross_rows(0, N, sightings(a=body, b=keep), {"pass_a": run})[0]
    # THE CASE THAT GOES RED when the cells given back inside her own track are not counted
    assert row["cells_given_back__pass_a__b"] == 2 and row.get("cells_given_back_inside_own_track__pass_a__b") == 2, row
    assert row["cells_given_back_inside_own_track_at__pass_a__b"] == "2,4;3,4", row
    beside = cap.cross_rows(0, N, sightings(a=A, b=keep), {"pass_a": run})[0]        # her track her head alone: the cells are not hers
    assert beside["cells_given_back__pass_a__b"] == 2 and beside["cells_given_back_inside_own_track__pass_a__b"] == 0, beside
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "runs" / "r").mkdir(parents=True)
        rows = [{"source_frame": 100 + n, **{f"cells_given_back_inside_own_track__r__{k}": (2 if n in (1, 2) else 0) for k in ("held", "b")},
                 **{f"cells_given_back_inside_own_track_at__r__{k}": ("2,4;3,4" if n in (1, 2) else "") for k in ("held", "b")}} for n in range(4)]
        (folder / "frames.json").write_text(json.dumps({"rows": rows}))
        (folder / "runs" / "r" / "per_frame.json").write_text(json.dumps({"rows": []}))
        m = {"subjects": [{"label": "a", "sightings": [{"by": "p", "shots": True}]}, {"label": "b", "sightings": [{"by": "p", "shots": True}]},
                          {"label": "held", "sightings": [{"by": "p", "shots": False}]}], "runs": [{"name": "r", "subject": "a", "planned": True}]}
        hits = [f for f in cap.flag_runs(m, folder) if f["rule"] == "keep_out_inside_own_track"]
        # THE CASE THAT GOES RED when a person is taken for a keep-out, or the keep-out is not flagged
        assert [(f["other"], f["level"], f["hard"], f["source_frames"]) for f in hits] == [("held", cap.LEVELS[2], True, [[101, 102]])], hits
        assert "101: 2,4;3,4" in hits[0]["why"] and "a's own track" in hits[0]["why"] and "2026-10-16" in hits[0]["why"], hits[0]["why"]
        flag = {**hits[0], "id": "f001"}
        over = {"flag": "f001", "key": cap.flag_key(flag), "overridden": True}
        # THE CASE THAT GOES RED when an override clears it
        assert cap.gate_verdict([flag], [over]) == {"verdict": "blocked", "blocking": ["f001"], "overridden": []}, cap.gate_verdict([flag], [over])
        (folder / "flags.json").write_text(json.dumps({"verdict": "blocked", "blocking": ["f001"], "flags": [flag]}))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cap.outcome(argparse.Namespace(capture=str(folder), flag="f001", happened="overridden", render=None, note="a reason", by="someone"))
        except SystemExit as e:
            assert "no override" in str(e), e
        else:
            raise AssertionError("an override of a hard flag was recorded")
        assert not (folder / "outcomes.json").exists(), "a refused override was written"
    # THE CARVE, on a hands keep-out over her face: given back inside her track uncarved, none carved
    hands = rect(32, 48, 64, 64)                         # her hands over the lower half of her face: her part is the face less them
    face = A & ~hands
    before = cap.cross_rows(0, N, sightings(a=body, b=hands), {"pass_a": _run(hands, True, face)})[0]
    carved, px, cells = cap.carve_keep_out(hands, body)
    after = cap.cross_rows(0, N, sightings(a=body, b=carved), {"pass_a": _run(carved, True, face)})[0]
    assert before["cells_given_back_inside_own_track__pass_a__b"] > 0, before
    assert after["cells_given_back_inside_own_track__pass_a__b"] == 0 and not carved.any(), after
    assert px.tolist() == [int(hands[0].sum())] * N and cells.tolist() == [int(cap.cells_any(hands)[0].sum())] * N, (px, cells)
    # her own pixels go too, in a cell that is mostly not hers; the keep-out's pixels outside her track stay
    wide = rect(16, 16, 100, 96)                         # her edge at x 100, a quarter into the cell 96-112: that cell is not hers
    edge = rect(96, 40, 110, 56)                         # a keep-out across it
    cut, px_edge, _ = cap.carve_keep_out(edge, wide)
    assert (cut == (edge & ~wide)).all() and px_edge.tolist() == [int((edge & wide)[0].sum())] * N, px_edge
    return "a keep-out's cells given back inside her own track counted and named; a hard flag no override clears; another person's and one beside her track not; " \
           f"a hands keep-out over her face carved: {before['cells_given_back_inside_own_track__pass_a__b']} cells given back inside her track uncarved, none carved"


def the_carve_in_the_files_step() -> str:
    """Through the files step, its readers stood in: the keep-out's track the builder will hand the node is written carved; a
    person's is not. Stops at the first table written after the masks are read."""
    import tempfile
    body, hands, other = rect(16, 16, 96, 96), rect(32, 48, 64, 64), rect(80, 16, 130, 90)    # the other person stands over her edge
    masks_given = {"a.mkv": body, "hands.mkv": hands, "b.mkv": other}
    written: dict = {}
    stood_in = {k: getattr(cap, k) for k in ("probe", "read_mask", "write_mask_video", "write_table", "input_facts")}

    class Reached(Exception):
        pass

    def table(*_a, **_k):
        raise Reached
    cap.probe = lambda _p: (W, H, N)
    cap.read_mask = lambda path, *_a: (masks_given[Path(path).name].copy(), np.ones(N, bool))
    cap.write_mask_video = lambda path, m: written.__setitem__(Path(path).parent.name, m)
    cap.write_table = table
    cap.input_facts = lambda _p: {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            shots = Path(tmp) / "s.json"
            shots.write_text("{}")
            # the keep-out first on the command line, and the replaced subject with no shot table: she is read first all the same
            given = [("hands", {"track": "hands.mkv"}), ("a", {"track": "a.mkv"}), ("b", {"track": "b.mkv", "shots": str(shots)})]
            args_ = argparse.Namespace(first=100, frames=N, name="x", date="2026-01-01", source="s.mp4")
            try:
                cap._files(args_, Path(tmp), given, [], [("r", {"subject": "a", "others": "b+hands"})])
            except Reached:
                pass
            except SystemExit as e:
                raise AssertionError(f"the keep-out was read before the replaced subject with no shot table: {e}") from None
            # ON A PLAN (the on-plan rebuild): the keep-out written for the builder is carved on the node's grid as well
            # zoomed 1.333 in a box whose units start at x 4: the unit 88-100 is two thirds hers (her edge at 96), and a band 96-100
            # outside her lies in it, which the canvas carve leaves (its cell 96-112 is not hers)
            zoom = {"box": [4, 0, 124, 72], "source_size": [W, H], "canvas": [W, H], "scale": 160 / 120, "render_region": "part", "set_by": "the box"}
            plan = _a_node_plan(Path(tmp), A & ~hands, zoom, 2)            # the plan's own frames: 5, from the capture's first frame
            band = rect(96, 30, 100, 90)
            masks_given["band.mkv"] = band
            given = given + [("band", {"track": "band.mkv"})]
            on_plan = {}
            cap.write_mask_video = lambda path, m: on_plan.__setitem__(Path(path).parent.name, m)
            try:
                cap._files(args_, Path(tmp) / "planned", given, [], [("r", {"subject": "a", "others": "b+hands+band", "preview": str(plan), "at": "100"})])
            except Reached:
                pass
            cap.write_mask_video = lambda path, m: written.__setitem__(Path(path).parent.name, m)
            canvas_only = cap.carve_keep_out(band, body)[0]
            want = cap.grid_carve(canvas_only, body, plan, 100, 100)[0]
            assert want.sum() < canvas_only.sum(), "the fixture: the grid carve takes nothing the canvas carve leaves"
            # THE CASE THAT GOES RED when the files step does not carve on the plan's grid
            assert (on_plan["band"] == want).all(), "the files step did not carve the keep-out on the node's grid"
            given = given[:-1]
            # THE CASE THAT GOES RED when a keep-out is written uncarved because the subject it is carved out of was never read
            gone = dict(written)
            try:
                cap._files(args_, Path(tmp) / "two", [("hands", {"track": "hands.mkv"}), ("b", {"track": "b.mkv", "shots": str(shots)})], [],
                           [("r", {"subject": "a", "others": "b+hands"})])
            except Reached:
                raise AssertionError("a keep-out was read with the subject it is carved out of never given") from None
            except SystemExit as e:
                assert "keep-out 'hands': a, whose track it is carved out of, has no --mask read before it" in str(e), e
            assert written["hands"] is gone["hands"], "the keep-out was written before the stop"
    finally:
        for k, v in stood_in.items():
            setattr(cap, k, v)
    # THE CASE THAT GOES RED when the files step does not carve: the copy the builder hands the node gives back cells inside her track
    row = cap.cross_rows(0, N, sightings(a=body, b=written["hands"]), {"pass_a": _run(written["hands"], True, A & ~hands)})[0]
    assert row["cells_given_back_inside_own_track__pass_a__b"] == 0 and not written["hands"].any(), row
    assert (written["b"] == other).all() and (written["a"] == body).all(), "a person's track was carved"
    return "the keep-out's track written for the builder is carved out of her track; a person's is untouched"


def _a_node_plan(folder: Path, part: np.ndarray, zoom: dict | None, latent_t: int, edge: str | None = None) -> Path:
    """A song node plan with one window and its region file, written by the node's own `save_window_region`: her part as the
    node has it (the zoomed picture under a zoom), a fixed margin of 40, latent cells of 16 px."""
    import torch
    vm = cap._pack("video_mask")
    frames = sum(vm.run_lengths(latent_t))
    if len(part) < frames:                                            # the window's frames past the given ones hold the last, as the node does
        part = np.concatenate([part, np.repeat(part[-1:], frames - len(part), axis=0)])
    src = {"zoom": zoom, "grow_pixels": 40, "grow_by": vm.GROW_FIXED, "edge": edge or vm.EDGE_CELLS, "feather_pixels": 0}
    vm.save_window_region(str(folder / "w1.npz"), cap._picture(vm, part[:frames], zoom, (W, H)), torch.zeros(latent_t, H // 16, W // 16), src, 40, 0, 0)
    (folder / "p_plan.json").write_text(json.dumps({"plan": "h3 song plan", "zoom": zoom, "width": W, "height": H,
                                                    "source": {k: v for k, v in src.items() if k != "zoom"},
                                                    "windows": [{"number": 1, "first_frame": 0, "frames": frames, "first_written_frame": 0,
                                                                 "last_written_frame": frames - 1, "region_file": "w1.npz"}]}))
    return folder / "p_plan.json"


def the_carve_on_the_nodes_grid() -> str:
    """The lead's ruling of 2026-10-16: the node pins a unit of its grid to the source over a whole latent step, on the zoomed
    picture's grid for a zoomed load, so the carve is made there: per window, per step, per unit. Counted by the node's own
    `window` on the carved keep-outs."""
    import tempfile
    vm = cap._pack("video_mask")
    lt = 2
    f = sum(vm.run_lengths(lt))                                       # 1 + 4 frames: the second step is frames 1-4

    def box(x0, y0, x1, y1, on=None):
        m = np.zeros((f, H, W), bool)
        m[on if on is not None else slice(None), y0:y1, x0:x1] = True
        return m
    said = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, zoom, edge, part, mine, keep in (
                # her part one cell; her track also two cells on frame 2 only (not the step's first); a keep-out over them on frames 1, 3, 4
                ("a latent step", None, vm.EDGE_CELLS, box(80, 32, 96, 48), box(80, 32, 96, 48) | box(48, 32, 64, 64, [2]), box(50, 34, 62, 62, [1, 3, 4])),
                # the same in whole tokens of 2x2 cells: counted in tokens, not cells (Gemini's note, 2026-10-16)
                ("whole tokens", None, vm.EDGE_TOKENS, box(80, 32, 96, 48), box(80, 32, 96, 48) | box(32, 32, 64, 64, [2]), box(34, 34, 62, 62, [1, 3, 4])),
                # (the box at x 40: the zoomed units start at canvas x 40 + 12k, so the unit 76-88 is two thirds hers up to her edge at 84,
                # the band 84-88 lies in it and in the canvas cell 80-96, which is not hers; her units read as canvas pixels without the
                # zoom would fall on 32-64, far from the band)
                ("a zoomed unit", {"box": [40, 24, 160, 96], "source_size": [W, H], "canvas": [W, H], "scale": 160 / 120, "render_region": "part",
                                   "set_by": "the box"}, vm.EDGE_CELLS, box(60, 40, 72, 60), box(60, 30, 84, 80), box(84, 30, 88, 80))):
            folder = Path(tmp) / tag.replace(" ", "_")
            folder.mkdir()
            plan = _a_node_plan(folder, part, zoom, lt, edge)

            def taken(keep_, where):
                # the region remade on these keep-outs by the node's own `window`, and the units they take from it
                return cap.node_taken_inside(None, cap.predict_regions(plan, 0, 0, keep_, None, (W, H), folder / where), mine, 0, 0)
            canvas = cap.carve_keep_out(keep, mine)[0]
            before = taken(canvas, "before")
            carved, lost = cap.grid_carve(canvas, mine, plan, 0, 0)
            after = taken(carved, "after")
            # THE CASES THAT GO RED when the carve is made frame by frame, or on the canvas's grid for a zoomed load
            assert before.sum() > 0, f"{tag}: the fixture: the node takes nothing after the canvas carve ({before.tolist()})"
            assert after.sum() == 0 and lost.sum() > 0, f"{tag}: the node still takes {after.tolist()} after the grid carve"
            if edge == vm.EDGE_TOKENS:
                # THE CASE THAT GOES RED when whole tokens are counted as cells: the two columns of cells under her are one token
                assert set(before.tolist()) <= {0, 1}, f"whole tokens counted as cells: {before.tolist()}"
            said.append(f"{tag}: the node took {before.tolist()} units a frame after the canvas carve, none after")
        # A WINDOW'S FRAMES PAST THE CAPTURE HOLD ITS LAST, as the node holds them (`video_mask.held_mask`; Gemini's note on 0604g window 2)
        tail = cap._window_frames(box(0, 0, 16, 16)[:3], {"first_frame": 1}, 0, 0, 4)
        assert tail[0].any() and tail[1].any() and tail[2].any() and tail[3].any(), "the frames past the capture are not its last held"
        # NO PIXEL WIDER WITHOUT A ZOOM: the cells are the node's exactly; a keep-out beside a carved unit keeps its pixels
        folder = Path(tmp) / "exact"
        folder.mkdir()
        part, mine = box(80, 32, 96, 48), box(80, 32, 96, 48) | box(48, 32, 64, 64, [1])
        beside = box(50, 34, 66, 62, [2, 3, 4])                       # 64-66 lies in the cell 64-80, not hers
        carved, _ = cap.grid_carve(beside, mine, _a_node_plan(folder, part, None, lt), 0, 0)
        assert carved[2:, 34:62, 64:66].all() and not carved[2:, 34:62, 50:64].any(), "a pixel wider was carved with no zoom"
        # A PLAN THAT IS NOT THE CAPTURE'S, OR A WINDOW WHOSE STEPS DO NOT HOLD ITS FRAMES, IS REFUSED (the read's finding 4)
        bad = json.loads((folder / "p_plan.json").read_text())
        (folder / "p_plan.json").write_text(json.dumps({**bad, "width": W * 2}))
        try:
            cap.node_windows(folder / "p_plan.json", (W, H))
        except SystemExit as e:
            assert "the plan is 320x96 and the capture 160x96" in str(e), e
        else:
            raise AssertionError("a plan of another size was read")
        (folder / "p_plan.json").write_text(json.dumps({**bad, "source": {**bad["source"], "grow_by": "the subject's size"}}))
        try:
            cap.node_windows(folder / "p_plan.json", (W, H))
        except SystemExit as e:
            assert "a fixed margin only" in str(e), e
        else:
            raise AssertionError("a grow by the subject's size was read")
        (folder / "p_plan.json").write_text(json.dumps(bad))
        import torch
        vm.save_window_region(str(folder / "w1.npz"), torch.zeros(9, H, W), torch.zeros(lt, H // 16, W // 16),
                              {"grow_pixels": 40, "grow_by": vm.GROW_FIXED, "edge": vm.EDGE_CELLS}, 40, 0, 0)   # 2 steps (5 frames), a mask of 9
        try:
            cap.node_windows(folder / "p_plan.json", (W, H))
        except SystemExit as e:
            assert "2 latent steps hold 5 frames and its mask has 9" in str(e), e
        else:
            raise AssertionError("a window whose steps do not hold its frames was read")
        said.append("frames past the capture held; no pixel wider without a zoom; a plan of another size, a grow by her size and steps that do not hold "
                    "the frames refused")
        # ANOTHER PERSON among the render's others is in both arms of the count: what the keep-outs take is counted, never what a person keeps
        folder = Path(tmp) / "a_person"
        folder.mkdir()
        part, mine = box(80, 32, 96, 48), box(80, 32, 96, 48) | box(48, 32, 64, 64, [1])
        person = box(50, 34, 62, 62, [2, 3, 4])                       # a person where the keep-out stood: he keeps those units out, and that is no fault
        plan = _a_node_plan(folder, part, None, lt)
        alone = cap.node_taken_inside(person, cap.predict_regions(plan, 0, 0, person, None, (W, H), folder / "p"), mine, 0, 0)
        assert alone.sum() == 0, f"a person's units counted as the keep-outs': {alone.tolist()}"
        said.append("a person's own units are not counted")
    return "; ".join(said)


def the_floor_holds() -> str:
    """Build (B), the lead's ruling of 2026-10-16: a tracked person's part under the floor, or collapsed against its neighbours, is held
    from the nearest frame whose part stands (never from a fragment), with its own small part on her track kept; not where no face shows
    (`noface`); and a hold whose shape lies mostly off her track places nothing ("hold missed her track"), floor or word."""
    face = rect(40, 20, 60, 40, 9)                                   # 400 px on every frame, then thinned below
    parts = face.copy()
    parts[3] = False
    parts[3, 22:25, 54:64] = True                                    # 3: a 30 px sliver, under the floor of 50, reaching past the face's place on her track
    parts[4] = False
    parts[4, 30:35, 44:55] = True                                    # 4: a 55 px stub, over the floor and 0.1375 of its neighbours: collapsed
    parts[5] = False
    parts[5, 22:24, 42:52] = True                                    # 5: 20 px, where a person said no face shows
    parts[8] = False
    parts[8, 22:24, 64:69] = True                                    # 8: 10 px on a track that lies beside her face's place
    classes = np.where(parts, 2, 0).astype(np.uint8)
    track = rect(36, 16, 64, 44, 9)
    track[8] = False
    track[8, 16:44, 62:70] = True
    held, did = cap.grade_holds(parts, parts, classes, [2], least_px=50, track=track, noface={5})
    # THE CASE THAT GOES RED when a small part on a tracked frame is emptied, and the original shows round her
    assert did[3]["did"] == "held by the floor rule" and held[3].sum() >= 300 and did[3]["own_part_kept_on_the_track_px"] == 30, did[3]
    # THE CASE THAT GOES RED when her own small part is thrown away and only the moved shape is laid
    assert held[3][parts[3]].all() and not face[3][parts[3]].all(), "the frame's own part on her track was not kept (or the fixture's part lies inside the moved shape)"
    # THE CASE THAT GOES RED when a stub beside a full face is called a good frame, or a hold comes from a fragment
    assert did[4]["did"] == "held by the floor rule" and "collapsed" in did[4]["why"] and did[4]["from_frames"][0] in (2, 6), did[4]
    assert did[3]["from_frames"][0] == 2, did[3]
    # THE CASE THAT GOES RED when a part is held where a person said no face shows
    assert not held[5].any() and did[5]["did"] == "emptied" and "no-face-on" in did[5].get("not_held", ""), did[5]
    # THE CASE THAT GOES RED when a hold that lands off her track is placed as a patch beside her
    assert not held[8].any() and did[8]["did"].startswith("emptied: hold missed her track") and did[8]["px_on_the_track"] == 0, did[8]
    # the same on a person's word: a word hold that lands off her track places nothing, and says so
    bare = np.where(np.arange(9)[:, None, None] == 8, False, face)
    worded, said = cap.grade_holds(bare, bare, np.where(face, 2, 0).astype(np.uint8), [2], word={8}, track=track)
    assert not worded[8].any() and said[8]["did"].startswith("left empty: hold missed her track"), said[8]
    # with no track given, the floor empties as it did
    old, said_old = cap.grade_holds(parts, parts, classes, [2], least_px=50)
    assert not old[3].any() and said_old[3]["did"] == "emptied" and "not_held" not in said_old[3], said_old[3]
    # A HOLD NEVER COMES FROM A HELD FRAME (Gemini's D16 on 0613f's 626-629, the lead's ruling): 1 and 3 are slivers, 2 was emptied by the
    # off-head rule; each takes its shape from 0 or 4, whose parts stand at her head, and none from another held frame
    chain = rect(40, 20, 60, 40, 5)
    chain[1] = False
    chain[1, 22:24, 42:52] = True
    chain[2] = False
    chain[3] = False
    chain[3, 22:24, 42:52] = True
    held_c, did_c = cap.grade_holds(chain, chain, np.where(rect(40, 20, 60, 40, 5), 2, 0).astype(np.uint8), [2], least_px=50,
                                    track=rect(36, 16, 64, 44, 5), off_head_fragments={2})
    # THE CASE THAT GOES RED when a hold is taken from a frame that was itself held (a held-from-held chain)
    assert [did_c[f]["from_frames"][0] for f in (1, 2, 3)] in ([0, 0, 4], [0, 4, 4]) and all(did_c[f]["did"] == "held by the floor rule" for f in (1, 2, 3)), \
        {f: did_c[f] for f in (1, 2, 3)}
    # THE FLOOR READ (the lead's ruling): every emptying before any hold, and every hold, word and floor, from frames standing as their own
    # part: 1 and 6 are held on a person's word, 2 and 4 are slivers, 3 is the job's drop= with a full part saved
    mixed = rect(40, 20, 60, 40, 8)
    for f in (1, 6):
        mixed[f] = False
    for f in (2, 4):
        mixed[f] = False
        mixed[f, 22:24, 42:52] = True
    held_m, did_m = cap.grade_holds(mixed, mixed, np.where(rect(40, 20, 60, 40, 8), 2, 0).astype(np.uint8), [2], least_px=50,
                                    track=rect(36, 16, 64, 44, 8), word={1, 6}, drop={3})
    # THE CASE THAT GOES RED when a floor hold takes a frame held on a person's word, or a dropped frame, as its source
    assert did_m[2]["from_frames"] == [0] and did_m[4]["from_frames"] == [5] and did_m[1]["from_frames"] == [0] and did_m[6]["from_frames"] in ([5], [7]), \
        {f: (did_m[f]["did"], did_m[f].get("from_frames")) for f in (1, 2, 4, 6)}
    assert not held_m[3].any() and did_m[3]["why"] == "dropped by the job (drop=)", did_m[3]
    # D22 ITEM 2 (order): a drop or noface frame with a huge false-positive blob (3000 px) beside a real 400 px standing part must be
    # emptied BEFORE sizes/COLLAPSE_SHARE are computed, so it never collapses its standing neighbour; and a drop/noface frame in `lost`
    # or `word` must never be re-filled by SHORT_FILL or word_shapes
    order_parts = rect(40, 20, 60, 40, 6)                             # 400 px standing part on frames 0, 1, 3, 5
    order_parts[2] = rect(10, 10, 70, 60, 1)[0]                       # frame 2 (drop): 3000 px false-positive blob (400 <= 0.15 * 3000 = 450!)
    order_parts[4] = rect(10, 10, 70, 60, 1)[0]                       # frame 4 (noface): 3000 px false-positive blob
    order_cls = np.where(order_parts, 2, 0).astype(np.uint8)
    order_track = rect(5, 5, 80, 60, 6)
    held_o, did_o = cap.grade_holds(order_parts, order_parts, order_cls, [2], least_px=50, track=order_track,
                                    drop={2}, noface={4}, lost={2, 4}, lost_fill=rect(40, 20, 60, 40, 6), word={4})
    assert 1 not in did_o and 3 not in did_o and (held_o[1] == order_parts[1]).all() and (held_o[3] == order_parts[3]).all(), (
        f"drop/noface blob collapsed standing neighbour or was not emptied first: {did_o}"
    )
    assert not held_o[2].any() and did_o[2]["did"] == "emptied" and "drop=" in did_o[2]["why"], did_o[2]
    assert not held_o[4].any() and did_o[4]["did"] == "emptied" and "no-face-on" in did_o[4].get("not_held", ""), did_o[4]
    # D22 ITEM 3 (report, not a threshold): every held frame records from_frames, source_px, and (when points given) face_points_inside,
    # so a hip hold (0/4 face points inside) and a head hold (4/4 face points inside) are distinguished in part_grades.json, manifest.json,
    # flags.json, and preflight stdout
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pts_p = Path(tmp) / "pts.npz"
        names_p = ["l_centerpoint_of_upper_lash_line", "center_of_glabella", "tip_of_nose", "tip_of_chin"]
        kp_p = np.zeros((9, len(names_p), 2), np.float32)
        kp_p[:] = [[50, 30], [45, 25], [55, 25], [50, 32]]           # all 4 face points inside rect(40, 20, 60, 40)
        kp_p[4] = [[15, 75], [12, 70], [18, 70], [15, 78]]           # frame 4: her real face is at (15, 75), outside the hip hold at (40..60, 20..40)!
        sc_p = np.ones((9, len(names_p)), np.float32)
        np.savez(pts_p, keypoints=kp_p, scores=sc_p, size=np.array([W, H]), names=np.array(names_p), first_frame=np.array(100))
        held_r, did_r = cap.grade_holds(parts, parts, classes, [2], least_px=50, track=track, noface={5}, points=str(pts_p), first_frame=100)
        assert did_r[3]["from_frames"] == [2] and did_r[3]["source_px"] == 400 and did_r[3].get("face_points_inside") == {"inside": 4, "of": 4}, did_r[3]
        assert did_r[4]["from_frames"] in ([2], [6]) and did_r[4]["source_px"] == 400 and did_r[4].get("face_points_inside") == {"inside": 0, "of": 4}, did_r[4]
    return ("a sliver held with its own part kept; a collapsed stub held from a full frame; no hold where no face shows; a hold off her track "
            "placed nowhere, floor or word; a hold never from a held, word-held or dropped frame; drop/noface emptied before collapse/fill; "
            "each held frame reports its source, source_px and face_points_inside")


def off_her_head() -> str:
    """The off-head rule (the lead's rulings of 2026-10-16, on 1010f's hat square and 0613f's skirt): a speck far from her face is dropped
    (1036: no part, so no region unit, stands there); her neck beside her face and above her shoulders, and a face split in two, are kept;
    a large piece below her shoulders goes (850's skirt); a frame left with nothing at her head is a fragment; a frame with no face points
    is left as it was, and said."""
    import tempfile
    assert all(p in cap.HEAD_POINT_NAMES for p in ("tip_of_chin", "center_of_labiomental_groove", "neck")), cap.HEAD_POINT_NAMES
    names = ["left_shoulder", "right_shoulder", "nose", "left_eye", "right_eye", "left_ear", "right_ear", "tip_of_chin",
             "center_of_labiomental_groove", "neck", "center_of_glabella", "tip_of_nose", "l_centerpoint_of_upper_lash_line",
             "r_centerpoint_of_upper_lash_line"]
    n = 10
    kp = np.zeros((n, len(names), 2), np.float32)
    sc = np.ones((n, len(names)), np.float32)
    face_at = {"nose": (50, 30), "left_eye": (45, 25), "right_eye": (55, 25), "left_ear": (43, 28), "right_ear": (57, 28), "tip_of_chin": (50, 39),
               "center_of_labiomental_groove": (50, 37), "neck": (50, 42), "center_of_glabella": (50, 23), "tip_of_nose": (50, 31),
               "l_centerpoint_of_upper_lash_line": (45, 25), "r_centerpoint_of_upper_lash_line": (55, 25), "left_shoulder": (35, 55), "right_shoulder": (65, 55)}
    for i, name in enumerate(names):
        kp[:, i] = face_at[name]
    sc[4] = 0.0                                                      # frame 4: no point over the line
    # Frame 5 (D22 item 4 & D22b item 1): bent over the table (0613f's 684-707) — her face points reach past her shoulder line
    # (shoulders at y=35 -> sh_line=39; tip_of_nose at y=40 >= 39 -> bent=True, line = 39 + hh(25) = 64), while tip_of_chin at (55, 49)
    # has low score (0.1 <= line_score). Hanging hair splits her Face_Neck into:
    # - upper face (40..48, 20..42),
    # - a neck piece (40..44, 45..51) outside head_box [39..61, 19..44] with no face point near, chained to upper face below sh_line=39,
    # - a split chin piece (52..58, 47..53) >margin from upper face and neck, with low-score tip_of_chin (55, 49) inside it,
    # - a disconnected hand under her chin (42..54, 58..63) inside [hx0..hx1] and <= line, with 0 face points on its edge (dropped!),
    # - a body patch outside [hx0..hx1] at (70..82, 44..62) (dropped!).
    kp[5, names.index("left_shoulder")] = (35, 35)
    kp[5, names.index("right_shoulder")] = (65, 35)
    kp[5, names.index("tip_of_nose")] = (50, 40)
    kp[5, names.index("neck")] = (50, 40)
    kp[5, names.index("tip_of_chin")] = (55, 49)
    sc[5, names.index("tip_of_chin")] = 0.1
    # Frames 6, 7, 8 (D22 item 5.1): each of tip_of_chin (frame 6), center_of_labiomental_groove (frame 7), and neck (frame 8) is the ONLY
    # lower head point scored on that frame (at x=55, y=46), and a jaw/neck piece at x=39..43, y=47..50 (near left_ear x=43, far from x=55
    # so face_near is False, and 7 px below the upper face piece at y=20..40 so chain-walk with margin=4 is False) lies inside head_box
    # ONLY when that point is in HEAD_POINT_NAMES!
    for f, active_pt in ((6, "tip_of_chin"), (7, "center_of_labiomental_groove"), (8, "neck")):
        for nm in ("tip_of_chin", "center_of_labiomental_groove", "neck"):
            sc[f, names.index(nm)] = 1.0 if nm == active_pt else 0.0
            kp[f, names.index(nm)] = (55, 46)
    parts = rect(40, 20, 60, 40, n)                                  # her face, 400 px
    parts[:5, 6:8, 131:133] = True                                   # 0-4: a 4 px speck far off, by a raised hat (1036): 0.01 of her face
    parts[1, 30:44, 62:70] = True                                    # 1: her neck beside the face, outside her head box (x to 61), 2 px from the face
    parts[1, 45:75, 64:68] = True                                    # 1: her arm, chained to the neck 1 px below it, reaching past the shoulder line
    parts[2, 70:95, 30:70] = True                                    # 2: a 1000 px piece below her shoulders (850's skirt)
    parts[3] = False
    parts[3, 20:40, 40:49] = True                                    # 3: her face split in two by a hand
    parts[3, 20:40, 51:60] = True
    parts[5] = False
    parts[5, 20:42, 40:48] = True                                    # 5: upper face sliver (176 px)
    parts[5, 45:51, 40:44] = True                                    # 5: chained neck piece outside head_box, below sh_line=39, <= line=64 (24 px)
    parts[5, 47:53, 52:58] = True                                    # 5: split chin piece >margin from face/neck, with tip_of_chin inside (36 px)
    parts[5, 58:63, 42:54] = True                                    # 5: disconnected hand under chin inside hx0..hx1, 0 face points on edge (dropped!)
    parts[5, 44:62, 70:82] = True                                    # 5: body patch outside head box [39..61] (dropped!)
    for f in (6, 7, 8):
        parts[f, 47:50, 39:43] = True                                # 6, 7, 8: side-jaw/neck piece at x=39..43, y=47..50 (inside head_box only via active_pt)
    # Frame 9 (D22 item 6.4): connected L-shaped piece whose single bounding box [50..70, 35..65] overlaps head_box [39..61, 19..46]
    # while all its actual pixels lie outside head_box (vertical arm x=65..70, y=35..65; bottom arm x=50..70, y=60..65, below sh_line=59)
    parts[9, 35:65, 65:70] = True
    parts[9, 60:65, 50:70] = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "points.npz"
        np.savez(path, keypoints=kp, scores=sc, size=np.array([W, H]), names=np.array(names), first_frame=np.array(100))
        kept, said, fragments, blind = cap.off_head(parts, str(path), 100, 4)
        # Upright subject with slightly tilted shoulders (46, 52) and neck at 47: a disconnected hand under her chin must be dropped
        kp_hand = kp[:1].copy()
        kp_hand[0, names.index("left_shoulder")] = (30, 46)
        kp_hand[0, names.index("right_shoulder")] = (70, 52)
        kp_hand[0, names.index("neck")] = (50, 47)
        hand_p = Path(tmp) / "hand_pts.npz"
        np.savez(hand_p, keypoints=kp_hand, scores=sc[:1], size=np.array([W, H]), names=np.array(names), first_frame=np.array(100))
        hand_parts = np.zeros((1, H, W), bool)
        hand_parts[0, 20:40, 40:60] = True
        hand_parts[0, 42:51, 48:52] = True
        hand_parts[0, 53:60, 48:52] = True
        hand_parts[0, 62:80, 44:58] = True
        kept_hand, said_hand, _, _ = cap.off_head(hand_parts, str(hand_p), 100, 4)
        assert not kept_hand[0, 62:80, 44:58].any() and not kept_hand[0, 53:60, 48:52].any(), (
            f"hand or chest under chin on upright tilted-shoulder frame was kept: {said_hand}"
        )
        skirt_only = parts.copy()
        skirt_only[0] = False
        skirt_only[0, 70:95, 30:70] = True
        _k, _s, frags_only, _b = cap.off_head(skirt_only, str(path), 100, 4)
        # D22 item 6.1: points file of another size (or no size) is refused
        bad_size = Path(tmp) / "bad_size.npz"
        np.savez(bad_size, keypoints=kp, scores=sc, size=np.array([W * 2, H * 2]), names=np.array(names), first_frame=np.array(100))
        try:
            cap.off_head(parts, str(bad_size), 100, 4)
            raise AssertionError("off_head accepted a points file of another size")
        except (ValueError, SystemExit) as exc:
            assert "size" in str(exc), exc
        # D22 item 5.2, item 3, item 6.3: run cap._files and cap.preflight with points= on --mask, verifying off_head runs in files,
        # records parts_dropped_off_her_head and part_sizes measured after off_head, and preflight writes held_frames to flags.json and stdout
        f_dir = Path(tmp) / "cap_files"
        stood_in = {k: getattr(cap, k) for k in ("probe", "read_mask", "read_classes", "write_mask_video", "input_facts", "picture_rows")}
        written_vids: dict[str, np.ndarray] = {}
        cls_names = list(cap._pack("sapiens2_parts").CLASS_NAMES)
        face_cls = cls_names.index("Face_Neck")
        f_parts = parts[:6].copy()
        f_parts[0] = False
        f_parts[0, 70:80, 30:50] = True                              # frame 0: only skirt below shoulders -> off_head empties it -> floor holds from frame 1!
        f_track = rect(30, 15, 75, 95, 6)
        f_cmap = np.where(f_parts, face_cls, 0).astype(np.uint8)
        f_shots = Path(tmp) / "shots.json"
        f_shots.write_text(json.dumps({"shots": [{"shot": 1, "first_frame": 0, "last_frame": 5, "frames": 6,
                                                  "subject": {"person": 1, "state": "taken"}, "frames_with_subject": 6,
                                                  "frames_without_subject": [], "frames_on_others": [], "track_score": [5.0] * 6}]}))
        f_src = Path(tmp) / "src.mp4"
        f_src.write_bytes(b"")
        try:
            cap.probe = lambda _p: (W, H, 6)
            cap.read_mask = lambda p, *_a: ((f_track.copy() if "track" in str(p) else f_parts.copy()), np.ones(6, bool))
            cap.read_classes = lambda *_a: f_cmap.copy()
            cap.write_mask_video = lambda p, m: written_vids.__setitem__(Path(p).name, m.copy())
            cap.input_facts = lambda p: {"file": str(p), "written": "2026-10-16T12:00:00", "size": [W, H], "frames": 6}
            cap.picture_rows = lambda *_a: [(0.0, 0.0)] * 6
            args_f = argparse.Namespace(first=100, frames=6, name="x", date="2026-01-01", source=str(f_src), loads=None, voice=None)
            spec_a = {"track": "track.mkv", "parts": "parts.mkv", "classes": "classes.mkv",
                      "by": "pv", "at": "100", "least": "50", "points": str(path), "points_margin": "4"}
            cap._files(args_f, f_dir, [("a", spec_a)], [], [])
            written_ht_vids: dict[str, np.ndarray] = {}
            cap.write_mask_video = lambda p, m: written_ht_vids.__setitem__(Path(p).name, m.copy())
            args_ht = argparse.Namespace(first=100, frames=6, name="x_ht", date="2026-01-01", source=str(f_src), loads=None, voice=None, head_tube=True)
            cap._files(args_ht, Path(tmp) / "cap_ht_files", [("a", dict(spec_a))], [], [])
        finally:
            for k, v in stood_in.items():
                setattr(cap, k, v)
        f_manifest = json.loads((f_dir / "manifest.json").read_text())
        f_sighting = f_manifest["subjects"][0]["sightings"][0]
        assert "parts_dropped_off_her_head" in f_sighting and f_sighting["parts_dropped_off_her_head"].get("frames_dropped", 0) >= 1, f_sighting
        assert not written_vids["parts__pv.mkv"][2, 70:95, 30:70].any(), "files step did not prune off-head piece from parts__pv.mkv"
        assert not written_ht_vids["parts__pv.mkv"][2, 70:95, 30:70].any() and not written_ht_vids["parts_held__pv.mkv"][2, 70:95, 30:70].any(), (
            "head_tube resurrected off-head piece from classes.mkv into parts__pv.mkv / parts_held__pv.mkv"
        )
        assert written_ht_vids["parts__pv.mkv"][2, 20:40, 40:60].all(), "head_tube altered head pixels on frame 2"
        assert "after off_head" in f_sighting["part_sizes"].get("measured_on", ""), f_sighting["part_sizes"]
        f_pg = json.loads((f_dir / "subjects" / "a" / "part_grades.json").read_text())
        assert "after off_head" in f_pg.get("measured_on", ""), f_pg
        assert f_pg.get("held_frames") and f_pg["held_frames"][0]["source_frame"] == 100 and f_pg["held_frames"][0]["from_source_frames"] == [101] \
            and f_pg["held_frames"][0]["source_px"] == 512 and (f_pg["held_frames"][0].get("face_points_inside") or {}).get("inside") == 5, f_pg.get("held_frames")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cap.preflight(argparse.Namespace(capture=str(f_dir), not_in=[], own_keep_out=[], given_back=[], named_empty=[],
                                             chin_points=None, who_is_who=None, text=None, voice_spans=None, gate=False))
        pf_out = buf.getvalue()
        f_flags = json.loads((f_dir / "flags.json").read_text())
        assert "held_frames" in f_flags and f_flags["held_frames"]["a"][0]["source_frame"] == 100, f_flags.get("held_frames")
        assert "100" in pf_out and "from 101" in pf_out and "5/5" in pf_out, pf_out
    # THE CASE THAT GOES RED when a speck far from her face stays in the part (1036's square on the hat)
    assert not kept[:4, 6:8, 131:133].any() and all("speck" in d["why"] for f in range(4) for d in said.get(f, []) if d["box"][0] == 131), said
    # THE CASE THAT GOES RED when her own neck, split from her face, is dropped
    assert kept[1, 30:44, 62:70].all(), said.get(1)
    # THE CASE THAT GOES RED when the chain walks from her neck down her arm past the shoulder line
    assert not kept[1, 45:75, 64:68].any() and any(d["box"] == [64, 45, 68, 75] and "shoulder" in d["bound"] for d in said[1]), said[1]
    # THE CASE THAT GOES RED when a large piece below her shoulders is kept (850's skirt, 5592 px)
    assert not kept[2, 70:95, 30:70].any() and any("off her head" in d["why"] and "shoulder" in d["bound"] for d in said[2]), said[2]
    # a face split in two is kept whole: both halves lie at her face points
    assert kept[3].sum() == parts[3, :, :128].sum(), said.get(3)
    # THE CASE THAT GOES RED when her real chin and neck under her bent head are dropped (0613f's 684-707), or a hand/body patch stays
    assert (kept[5, 20:42, 40:48].all() and kept[5, 45:51, 40:44].all() and kept[5, 47:53, 52:58].all()
            and not kept[5, 58:63, 42:54].any() and not kept[5, 44:62, 70:82].any()), said.get(5)
    # THE CASE THAT GOES RED when tip_of_chin, center_of_labiomental_groove, or neck is removed from HEAD_POINT_NAMES
    for f, nm in ((6, "tip_of_chin"), (7, "center_of_labiomental_groove"), (8, "neck")):
        assert kept[f, 47:50, 39:43].all(), f"frame {f} ({nm}): side-jaw/neck piece inside head_box was dropped: {said.get(f)}"
    # THE CASE THAT GOES RED when in_head tests bounding-box overlap instead of pixels inside head_box
    assert not kept[9, 35:65, 65:70].any() and not kept[9, 60:65, 50:70].any(), f"frame 9 L-shaped piece outside head_box was kept: {said.get(9)}"
    # THE CASE THAT GOES RED when a frame left with nothing at her head is not a fragment for the floor rule
    assert frags_only == {0} and fragments == set(), (frags_only, fragments)
    # no face points: the part as it was, and said
    assert blind == [4] and np.array_equal(kept[4], parts[4]), blind
    # the floor rule holds a frame the off-head rule left with nothing at her head, from the nearest frame whose part stands
    emptied = rect(40, 20, 60, 40, n)
    emptied[0] = False
    held, did = cap.grade_holds(emptied, emptied, np.where(rect(40, 20, 60, 40, n), 2, 0).astype(np.uint8), [2], least_px=50,
                                track=rect(36, 16, 64, 44, n), off_head_fragments={0})
    # THE CASE THAT GOES RED when a frame left with nothing at her head is left as the original
    assert did[0]["did"] == "held by the floor rule" and held[0].sum() == 400, did[0]
    return "a far speck dropped; her neck and bent-head chin/neck kept; a hand under chin, piece below shoulders and L-box outside head_box dropped; HEAD_POINT_NAMES and files step verified"


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
        # a load of two subjects, in the builder's shape: read by the builder's own rule, the first subject's, with which label is whose
        two = {"job": {"name": "j", "first": 90, "subjects": [{"label": "b", "still": "b_job.png"}],
                       "texts": {"hall": {"file": "hall.txt", "frames": [100, 140], "by": "writer"}}},
               "loads": [{"name": "ab", "first": 100, "frames": 33, "text": "hall", "margin": 32, "subjects": [
                   {"subject": "a", "region": "face and hair", "still": "a.png", "motion_file": "/x/a_ref.mp4", "motion_first": 96, "others": ["hands"]},
                   {"subject": "b", "region": "face", "motion_file": "/x/b_ref.mp4"}]}]}
        (folder / "two.json").write_text(json.dumps(two))
        got = cap.load_and_text(folder / "two.json", "ab")
        assert got["subject"] == "a" and got["frames"] == [100, 132] and got["recorded"], got
        assert got["pair"] == [
            {"subject": "a", "region": "face and hair", "is": "<Subject 1>", "picture": "<Picture 1>", "still": "a.png", "video": "<Video 1>"},
            {"subject": "b", "region": "face", "is": "<Subject 2>", "picture": "<Picture 2>", "still": "b_job.png", "video": "<Video 2>"}], got["pair"]
        # where each reference begins: the load's own word, else the job's first frame (the builder's rule)
        assert got["motion"] == [{"file": "/x/a_ref.mp4", "first": 96}, {"file": "/x/b_ref.mp4", "first": 90}], got["motion"]
        assert "pair" not in cap.load_and_text(folder / "job.json", "a_1") and cap.load_and_text(folder / "job.json", "a_1")["motion"] == []
        two["loads"][0]["subjects"][1]["subject"] = "a"                      # both subjects the same: the builder's refusal, by name
        (folder / "same.json").write_text(json.dumps(two))
        try:
            cap.load_and_text(folder / "same.json", "ab")
        except SystemExit as e:
            assert "both of its `subjects` are 'a'" in str(e), e
        else:
            raise AssertionError("a load whose two subjects are one was shown")
    lines = cap.wrap_text("one two three four\n\nfive", 9, len)
    assert lines == ["one two", "three", "four", "", "five"] and " ".join(x for x in lines if x).split() == "one two three four five".split(), lines
    return ("a look holds what each strip showed; the maker's own, a look of no strip, a strip older than the capture and an empty note "
            "are refused; a load's text is shown whole with the frames it was written from, or marked not recorded; "
            + ("the builder's hash of one file equals the capture's" if compared else
               "THE BUILDER'S HASH WAS NOT COMPARED (its text_of does not give one in this tree)"))


def what_the_node_will_render() -> str:
    """The lead, 2026-10-14: every blind reader of that day read a strip that said nowhere whether its load renders zoomed, nor its
    window, nor the defaults it carries. The strip says them, on 1097c's figures (zoomed) and 0604d's (the whole frame)."""
    import tempfile
    defaults = [{"what": "zoom", "carries": "zoomed in where the node's box allows"}, {"what": "the eye", "carries": "white sclera and black iris"}]
    zoomed = {"zoom": {"box": [60, 26, 628, 452], "source_size": [1024, 768], "canvas": [1024, 768], "scale": 1.8028169014084507, "set_by": "the box"},
              "planned_windows": [{"window": 1, "writes_source_frames": [1097, 1133], "first_frame": 1097, "frames": 39, "held_frames": 2, "kept_frames": 0}]}
    whole = {"zoom": None, "planned_windows": [{"window": 1, "writes_source_frames": [604, 948], "first_frame": 604, "frames": 345, "held_frames": 0, "kept_frames": 0}]}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        job = d / "j.json"
        job.write_text(json.dumps({"job": {"capture": str(d / "cap"), "defaults": defaults}}))
        z = cap.render_said(zoomed, job, "lead_fp_1097c")
        # THE CASE THAT GOES RED when the strip leaves out the zoom, the window's held frames or the defaults
        assert z[0].startswith("RENDERED ZOOMED IN 1.80x on the box [60, 26, 628, 452] of the 1024x768 frame"), z
        assert z[1] == "window 1: 39 frames from source frame 1097, writing 1097-1133; 2 held frame(s), 0 kept", z
        assert z[2] == "THE DEFAULTS THIS LOAD CARRIES: zoom: zoomed in where the node's box allows; the eye: white sclera and black iris", z
        (d / "cap_job").mkdir()
        (d / "cap_job" / "second_hfp_0604d.plan.queued.json").write_text(json.dumps({"out": "rendered on the whole frame: the subject with its room takes "
                                                                                             "1080x810 of the 1024x768 the canvas shows, which is under 1.25 times\n[1] 00:00"}))
        w = cap.render_said(whole, job, "second_hfp_0604d")
        assert w[0] == ("RENDERED ON THE WHOLE FRAME: the node's words, the subject with its room takes 1080x810 of the 1024x768 the canvas shows, "
                        "which is under 1.25 times"), w
        assert cap.render_said(whole, d / "none.json", "x")[0] == "RENDERED ON THE WHOLE FRAME (the node's plan names no zoom)"
    return "the strip says zoomed in by how much and on which box, or the whole frame with the node's words why; each window's frames and held frames; the defaults"


def a_strip_of_two_subjects() -> str:
    """The strip's masks row for a load of two subjects: each subject's part in its own colour, one margin, a legend line each."""
    red, blue = (255, 0, 0), (0, 0, 255)
    a_track, b_track = rect(16, 16, 64, 80, 1), rect(96, 16, 144, 80, 1)
    subjects = [{"label": "a", "colour": red, "track": a_track}, {"label": "b", "colour": blue, "track": b_track}]
    carried = rect(32, 32, 48, 48, 1) | rect(112, 32, 128, 48, 1)           # both parts as one mask, as the node is given them
    region = np.zeros((1, H // CELL, W // CELL), bool)
    region[0, 1:4, 1:9] = True
    run = {"name": "ab", "subject": "a", "planned": True, "region": region, "carried": carried, "back": {}}
    plate = np.full((H, W, 3), 100, np.uint8)
    one = cap.draw_masks(plate, 0, subjects, [run]).astype(int)
    two = cap.draw_masks(plate, 0, subjects, [{**run, "subjects": ["a", "b"]}]).astype(int)
    in_a, in_b, margin = (40, 40), (40, 120), (40, 72)                      # (y, x): inside a's part, inside b's part, between them
    assert one[in_a][0] > one[in_a][2] and one[in_b][0] > one[in_b][2], "the control: told one subject, both parts are drawn in its colour"
    assert two[in_a][0] > two[in_a][2] and two[in_b][2] > two[in_b][0], (two[in_a], two[in_b])
    assert (two[margin] == one[margin]).all() and two[margin][0] > two[margin][2], "the one margin is the first subject's colour"
    assert (two[in_a] == one[in_a]).all(), "the first subject's part changed colour when the second was named"
    both = rect(32, 32, 48, 48, 1)                                          # a pixel both tracks claim stays the first's
    overlap = cap.draw_masks(plate, 0, [subjects[0], {**subjects[1], "track": b_track | both}], [{**run, "subjects": ["a", "b"]}]).astype(int)
    assert overlap[in_a][0] > overlap[in_a][2], overlap[in_a]
    both_ = cap.pinned_legend(subjects, [{**run, "subjects": ["a", "b"]}], [{}], ["a", "b"])
    assert [(e["label"], e["kind"]) for e in both_] == [("a", "replaced"), ("b", "replaced")], both_
    alone = cap.pinned_legend(subjects, [run], [{}], ["a"])
    assert [(e["label"], e["kind"]) for e in alone] == [("a", "replaced"), ("b", "other")], alone
    return "each subject's part is filled in its own colour and the margin in the first's; told one subject the picture is as before; a legend line each"


def the_strips_marks_and_records() -> str:
    """The review strip of 2026-10-15 (C1): a given-back cell hatched in its keep-out's colour, never white; a keep-out pinned in
    the legend on a frame where its share of the region is 0 and it still has cells given back (that frame's line was dropped
    before); an overridden flag said so; the frame records' events; the head crop's size fixed while its centre follows the face."""
    import tempfile
    red, blue = (255, 0, 0), (0, 0, 255)
    subjects = [{"label": "a", "colour": red, "track": rect(16, 16, 64, 80, 2)}, {"label": "hands", "colour": blue, "track": rect(96, 16, 144, 80, 2)}]
    region = np.zeros((2, H // CELL, W // CELL), bool)
    region[:, 1:5, 1:5] = True
    back = np.zeros((2, H // CELL, W // CELL), bool)
    back[1, 3, 7] = True                                            # one cell given back to hands, on frame 1 only, inside its outline
    run = {"name": "r", "subject": "a", "planned": True, "region": region, "carried": rect(32, 32, 48, 48, 2), "back": {"hands": back}}
    plate = np.full((H, W, 3), 100, np.uint8)
    img = cap.draw_masks(plate, 1, subjects, [run]).astype(int)
    y0, x0 = 3 * CELL, 7 * CELL
    diag = img[y0 + CELL - 1, x0]                                    # the long stroke starts at the cell's bottom left
    # THE CASE THAT GOES RED when a given-back cell is drawn white, whatever claimed it
    assert tuple(diag) == blue, f"a cell given back to hands is drawn {tuple(diag)}, not hands' colour"
    assert tuple(cap.draw_masks(plate, 0, subjects, [run]).astype(int)[y0 + CELL - 1, x0]) != blue, "the control: no cell on frame 0"
    rows = [{"region_on__r__hands": "0.0", "cells_given_back__r__hands": "0"}, {"region_on__r__hands": "0.0", "cells_given_back__r__hands": "1"}]
    legend = cap.pinned_legend(subjects, [run], rows, ["a"])
    assert [(e["label"], e["kind"]) for e in legend] == [("a", "replaced"), ("hands", "keep-out")], legend
    flags_in = [{"id": "f001", "level": "likely to fail", "rule": "region_over_another_subject", "subject": "a", "other": "hands", "run": "r",
                 "source_frames": [[10, 10]], "why": "w"},
                {"id": "f002", "level": "iffy", "rule": "x", "subject": None, "run": None, "source_frames": [[11, 11]]},
                {"id": "f003", "level": "likely fine", "rule": "y", "run": None, "source_frames": [[10, 11]]},
                {"id": "f004", "level": "iffy", "rule": "z", "run": None, "source_frames": [[50, 60]]},
                {"id": "f005", "level": "iffy", "rule": "z", "run": "another", "source_frames": [[10, 11]]}]
    flags = cap.flag_rows(flags_in, {"f001"}, {"r"}, (10, 11))
    assert [f["id"] for f in flags] == ["f001", "f002"] and flags[0]["overridden"] and "OVERRIDDEN" in flags[0]["words"], flags
    seg = cap.flag_rows([{**flags_in[1], "id": "f006", "rule": "segment_inside_region", "segment": "a.Hair"}], set(), {"r"}, (10, 11))
    assert "(a.Hair)" in seg[0]["words"], seg                     # a flag of one segment names it
    m = {"first_frame": 10}
    table = {"a": [{"covered": True, "track_share": 0.1}] * 2, "hands": [{"covered": True, "track_share": 0.02, "track_pieces": 3}] * 2}
    run_table = {"r": [{"region_share": 0.05, "margin_share": 0.02}, {"region_share": 0.0, "margin_share": 0.0}]}
    given = {"inputs": [{"role": "the source", "tag": None, "loader": {"first_frame": 10}},
                        {"role": "<Video 1>", "tag": "<Video 1>", "loader": {"first_frame": 0}, "reference_frames": 1,
                         "encoder_frames": [{"window_first": 0, "window_frames": 2, "sampled": [{"source_frame": 10, "reference_frame": 0, "held": False}]}]}]}
    recs = cap.strip_records(m, subjects, [run], rows, table, run_table, flags, given, 0, 2, 0, True, legend, ["hands"])
    assert recs[0]["encoder_samples"] == [{"tag": "<Video 1>", "reference_frame": 0}] and recs[1]["window"] == {"number": 1, "frame": 1}, recs
    assert recs[1]["references"][0]["held"] and not recs[0]["references"][0]["held"], recs[1]["references"]
    assert recs[1]["keep_outs"]["hands"]["cells_given_back"] == 1 and "1 margin cells kept as the original" in recs[1]["legend"][1], recs[1]["legend"]
    ch = recs[1]["changes"]
    for want in ("flag f001 ends", "flag f002 starts", "nothing regenerates from this frame", "hands: cells given back start (0 to 1)",
                 "<Video 1> runs out: its last frame from here"):
        assert want in ch, (want, ch)
    assert recs[0]["changes"] == ["the text encoder is shown <Video 1> frame 0 here"], recs[0]["changes"]
    # the head crop: a face moving right across a reference, read through the record's box and scale
    with tempfile.TemporaryDirectory() as tmp:
        names = ["l_upper_lash_line", "r_upper_lash_line", "center_of_glabella", "tip_of_nose", "tip_of_chin"]
        kp = np.zeros((20, len(names), 2), np.float32)
        for f in range(20):
            kp[f] = [[100 + 5 * f, 100], [140 + 5 * f, 100], [120 + 5 * f, 95], [120 + 5 * f, 120], [120 + 5 * f, 150]]
        pts = Path(tmp) / "p.npz"
        np.savez(pts, keypoints=kp, scores=np.ones((20, len(names)), np.float32), names=np.array(names), first_frame=np.array(500), size=np.array([1024, 768]))
        video = {"record": {"box": [0, 0, 1024, 768], "scale": [2.0, 2.0], "points": {"file": str(pts), "first_row_is_source_frame": 500}}}
        boxes = cap.head_boxes(video, (1024, 768), (500, 519))
        sides = {b[2] - b[0] for b in boxes} | {b[3] - b[1] for b in boxes}
        assert len(boxes) == 20 and len(sides) == 1, boxes
        assert boxes[-1][0] > boxes[0][0] and boxes[10][1] == boxes[0][1], (boxes[0], boxes[10], boxes[-1])
        assert cap.head_boxes({"record": {}}, (1024, 768), (500, 519)) is None
        # a chin whose point is not read: the record's face height stands for it, so the crop reaches below the nose
        sc = np.ones((20, len(names)), np.float32)
        sc[:, names.index("tip_of_chin")] = 0.0
        np.savez(pts, keypoints=kp, scores=sc, names=np.array(names), first_frame=np.array(500), size=np.array([1024, 768]))
        short = cap.head_boxes(video, (1024, 768), (500, 519))
        tall = cap.head_boxes({"record": {**video["record"], "face_height_px": {"in_this_video": 110.0}}}, (1024, 768), (500, 519))
        assert tall[0][3] - tall[0][1] > short[0][3] - short[0][1] and tall[0][3] >= 2 * 95 + 110, (short[0], tall[0])
        # C3, 2026-10-15: a head tipped back on a close-up, quickly: the face drops 120 px in three frames and its chin reaches far
        # down on them. Each frame's points lie inside its own box, and the box keeps one size
        kp2 = kp.copy()
        for f in range(10, 13):
            kp2[f] += [0, 120]
            kp2[f][names.index("tip_of_chin")] += [0, 40]
        np.savez(pts, keypoints=kp2, scores=np.ones((20, len(names)), np.float32), names=np.array(names), first_frame=np.array(500), size=np.array([1024, 768]))
        tipped = cap.head_boxes(video, (1024, 768), (500, 519))
        assert len({b[2] - b[0] for b in tipped}) == 1, tipped
        for f, b in enumerate(tipped):
            xs, ys = kp2[f][:, 0] * 2.0, kp2[f][:, 1] * 2.0           # through the record's scale
            assert b[0] <= xs.min() and xs.max() <= b[2] and b[1] <= ys.min() and ys.max() <= b[3], (f, b, ys.min(), ys.max())
    # a keep-out the capture tracks no mask of is still a legend line, never a fault
    words = cap.legend_words({"label": "strangers", "colour": (1, 2, 3), "kind": "keep-out"}, {"subjects": {}, "runs": {}, "keep_outs": {}})
    assert "no tracked mask of it in this capture" in words, words
    # what the composite kept (the song node's save_kept): read from a window's region file by source frame, drawn white on the
    # whole frame through the zoom box; a region file without it gives nothing, as every render before 2026-10-15
    with tempfile.TemporaryDirectory() as tmp:
        kept = np.zeros((3, 48, 64), bool)
        kept[1, 12:36, 16:48] = True                                   # frame 1 of the window: a block in the zoomed picture
        meta = {"first_frame": 2, "width": 64, "zoom": {"box": [32, 16, 96, 64]}}
        reg = Path(tmp) / "w_region.npz"
        np.savez(reg, kept=np.packbits(kept, axis=-1), meta=np.array(json.dumps(meta)))
        plain = Path(tmp) / "p_region.npz"
        np.savez(plain, meta=np.array(json.dumps(meta)))
        given_k = {"inputs": [{"role": "the source", "tag": None, "loader": {"first_frame": 100}}], "zoom": {"windows": [{"region": str(reg)}]}}
        got_k = cap.kept_of(given_k)
        assert sorted(got_k) == [102, 103, 104] and not got_k[102][0].any() and got_k[103][0].sum() == 24 * 32, sorted(got_k)
        assert cap.kept_of({**given_k, "zoom": {"windows": [{"region": str(plain)}]}}) == {}
        canvas = np.zeros((96, 160, 3), np.uint8)
        px = cap.draw_kept(canvas, *got_k[103])
        # the box is 64x48 for a 64x48 picture: the block lands at x 48-80, y 28-52 of the whole frame, outlined white
        assert px == 32 * 24 and tuple(canvas[28, 60]) == cap.WHITE and tuple(canvas[40, 60]) == (0, 0, 0), (px, canvas[28, 60], canvas[40, 60])
    return ("a given-back cell in its keep-out's colour; a keep-out at 0% of the region still pinned with its cell; an override said; "
            "what the composite kept read by source frame and drawn white through the zoom box; "
            "events and the encoder's frames in the records; the head crop one size, following the face")


def neither_is_another() -> str:
    """No subject of a load of two is "another subject" to it: the cross table, the region rule and the segments rule."""
    import tempfile
    third = rect(96, 32, 128, 64)
    seen = {"a": {"p": (A, ALL, None)}, "b": {"p": (rect(64, 32, 96, 64), ALL, None)}, "c": {"p": (third, ALL, None)}}
    region = np.zeros((N, H // CELL, W // CELL), bool)
    region[:, 2:4, 2:8] = True                                        # over a, b and c alike
    run = {"subject": "a", "others": [], "region": region, "carried": A | rect(64, 32, 96, 64), "read": ALL, "reached": {}, "back": {}}
    alone = cap.cross_rows(0, N, seen, {"ab": run})[0]
    assert alone["region_on__ab__b"] > 0 and alone["region_on__ab__c"] > 0 and "region_on__ab__a" not in alone, "the control: one subject, and b is another"
    pair = cap.cross_rows(0, N, seen, {"ab": {**run, "subjects": ["a", "b"]}})[0]
    assert "region_on__ab__b" not in pair and "region_cells_holding__ab__b" not in pair and pair["region_on__ab__c"] == alone["region_on__ab__c"], pair
    assert cap.subjects_of({"subject": "a"}) == ["a"] and cap.subjects_of({"subject": "a", "subjects": ["a", "b"]}) == ["a", "b"]
    # a rule that is a fact about a subject walks each run once per subject; a run of one subject is the very same run
    single = {"name": "r", "subject": "a"}
    walked = cap.by_subject([single, {"name": "ab", "subject": "a", "subjects": ["a", "b"], "load": {"first_frame": 5}}])
    assert walked[0] is single and [(r["name"], r["subject"]) for r in walked] == [("r", "a"), ("ab", "a"), ("ab", "b")], walked
    assert walked[2]["load"] == {"first_frame": 5} and walked[2]["subjects"] == ["a", "b"], walked[2]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "runs" / "ab").mkdir(parents=True)
        frames = [{"source_frame": 100 + n, "region_on__ab__b": 0.5, "region_on__ab__c": 0.5} for n in range(N)]     # a capture made before the change
        (folder / "frames.json").write_text(json.dumps({"rows": frames}))
        (folder / "runs" / "ab" / "per_frame.json").write_text(json.dumps({"rows": [
            {"source_frame": 100 + n, "region_share": 0.2, "margin_share": 0.15} for n in range(N)]}))
        px = cap.SEGMENT_PX
        (folder / "runs" / "ab" / "segments_in_region.json").write_text(json.dumps({"rows": [
            {"frame": n, "source_frame": 100 + n, "b.Face__px": px, "b.Face__cells": 2, "b.Hand__px": px, "b.Hand__cells": 1,
             "c.Hand__px": px, "c.Hand__cells": 1} for n in range(N)]}))
        def manifest(**more) -> dict:
            return {"subjects": [{"label": "a", "sightings": [{"by": "p", "part_is_made_of": ["Face"]}]},
                                 {"label": "b", "sightings": [{"by": "p", "part_is_made_of": ["Face", "Hair"]}]},
                                 {"label": "c", "sightings": [{"by": "p"}]}],
                    "runs": [{"name": "ab", "subject": "a", "planned": True, **more}]}
        def named(flags, rule):
            return sorted((f.get("other") or f.get("segment"), f["subject"]) for f in flags if f["rule"] == rule)
        one, two = cap.flag_runs(manifest(), folder), cap.flag_runs(manifest(subjects=["a", "b"]), folder)
        assert named(one, "region_over_another_subject") == [("b", "a"), ("c", "a")], "the control: told one subject, the region over b is flagged"
        assert named(two, "region_over_another_subject") == [("c", "a")], named(two, "region_over_another_subject")
        mostly = [f["why"] for f in two if f["rule"] == "region_mostly_not_the_subject"]
        assert len(mostly) == 1 and "is not a and b " in mostly[0], mostly
        one, two = cap.flag_segments(manifest(), folder), cap.flag_segments(manifest(subjects=["a", "b"]), folder)
        assert named(one, "segment_inside_region") == [("b.Face", "a"), ("b.Hand", "a"), ("c.Hand", "a")], named(one, "segment_inside_region")
        # the second subject's own part is what is replaced; what else is theirs inside the region is their own, to name in the text
        assert named(two, "segment_inside_region") == [("b.Hand", "b"), ("c.Hand", "a")], named(two, "segment_inside_region")
        why = {f["segment"]: f["why"] for f in two if f["rule"] == "segment_inside_region"}
        assert "name it in the text or keep it out" in why["b.Hand"] and "c's own pass must win there" in why["c.Hand"], why
    return ("told two subjects, the cross table has no column for the region on either, the region rule and the segments rule flag "
            "neither as another and still flag a third; told one, the second is flagged as before")


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


def what_the_job_says_is_its_own() -> str:
    """The lead's rulings of 2026-10-14 on the card's stops: frames the job empties on purpose are said with its reason a level
    below the top; a region over the subject's OWN keep-out (her hands, held things) drops a level only where something
    gives it back on every frame it is over, and otherwise says what would; another person's stays as it was."""
    import tempfile
    run = {"name": "r", "subject": "a", "planned": True}
    flags = cap.lays_nothing(run, [10, 11, 12, 20], {10: "THE JOB'S DROP", 11: "THE JOB'S DROP", 12: "A LOOK"}, 50)
    named = next(f for f in flags if f["rule"] == "load_lays_nothing_where_named")
    top = next(f for f in flags if f["rule"] == "load_lays_nothing_on_frames")
    # THE CASE THAT GOES RED when a frame the job named still blocks, or one nobody named does not
    assert named["level"] == cap.LEVELS[1] and named["source_frames"] == [[10, 12]] and "10-11: THE JOB'S DROP" in named["why"] and "12-12: A LOOK" in named["why"], named
    assert top["level"] == cap.LEVELS[2] and top["source_frames"] == [[20, 20]] and top["figures"] == {"frames": 1, "frames_of_the_load": 50}, top
    assert [f["rule"] for f in cap.lays_nothing(run, [10], {}, 50)] == ["load_lays_nothing_on_frames"]
    told = cap.told_by_the_job(argparse.Namespace(own_keep_out=["hands=Left_Hand+Right_Hand", "held="], given_back=["hands:100-102=THE RESTORE"],
                                                  named_empty=["a:5-6=THE DROP"]))
    assert told == {"own": {"hands": ["Left_Hand", "Right_Hand"], "held": []}, "given": {"hands": [(100, 102, "THE RESTORE")]},
                    "named": {"a": [(5, 6, "THE DROP")]}, "chin": None}, told
    for bad, words in ((dict(given_back=["clothes:1-2=X"]), "not one of the --own-keep-out"), (dict(given_back=["hands:1-2="]), "with the why"),
                       (dict(named_empty=["a:9-2=X"]), "with the why"), (dict(named_empty=["a:x=X"]), "not NAME:FIRST-LAST")):
        try:
            cap.told_by_the_job(argparse.Namespace(**{"own_keep_out": ["hands=Left_Hand"], "given_back": [], "named_empty": [], **bad}))
            raise AssertionError(f"taken: {bad}")
        except SystemExit as e:
            assert words in str(e), (bad, str(e))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "runs" / "r").mkdir(parents=True)
        (folder / "frames.json").write_text(json.dumps({"rows": [{"source_frame": 100 + n, "region_on__r__hands": 0.5, "region_on__r__held": 0.5,
                                                                    "region_on__r__b": 0.5} for n in range(6)]}))
        (folder / "runs" / "r" / "per_frame.json").write_text(json.dumps({"rows": [{"source_frame": 100 + n, "region_share": 0.2, "margin_share": 0.0}
                                                                                     for n in range(6)]}))
        m = {"subjects": [{"label": x, "sightings": [{"by": "p"}]} for x in ("a", "b", "hands", "held")], "runs": [run]}
        own = {"hands": ["Left_Hand", "Right_Hand"], "held": []}
        by = lambda fl: {f["other"]: f for f in fl if f["rule"] == "region_over_another_subject"}       # noqa: E731
        before = by(cap.flag_runs(m, folder))
        assert {k: f["level"] for k, f in before.items()} == {"b": cap.LEVELS[2], "hands": cap.LEVELS[2], "held": cap.LEVELS[2]}, "the control"
        whole = by(cap.flag_runs(m, folder, own, {"hands": [(100, 105, "THE PLAN'S RESTORE")]}))
        assert whole["hands"]["level"] == cap.LEVELS[1] and "a's OWN, and on every one of those frames it is given back: THE PLAN'S RESTORE" in whole["hands"]["why"], whole["hands"]
        assert whole["held"]["level"] == cap.LEVELS[2] and "no class a restore span can name, so nothing gives it back on 100-105" in whole["held"]["why"], whole["held"]
        assert whole["b"] == before["b"], "another person's region changed"
        part = by(cap.flag_runs(m, folder, own, {"hands": [(100, 102, "THE PLAN'S RESTORE")]}))["hands"]
        assert part["level"] == cap.LEVELS[2] and "given back on 3 of the frames (THE PLAN'S RESTORE)" in part["why"] \
            and "a restore span of the plan naming Left_Hand+Right_Hand on 103-105 would give those pixels back" in part["why"], part
        assert by(cap.flag_runs(m, folder, own, {}))["hands"]["level"] == cap.LEVELS[2], "her own hands with nothing given back went quiet"
    return ("frames the job names are said with its reason and do not block, others do; her own keep-out drops a level only where given back on every "
            "frame, and says which restore would answer it; another person's is unchanged; the gate's job flags parsed and refused")


def held_split() -> str:
    """The lead, 2026-10-14, on mrpre's read: held's pixels inside the region are split before they count. Edge slivers (gone under a
    HELD_THIN_PX opening) and her clothing (Upper_Clothing, Apparel, or Background below her chin) are data with frames and px; the
    rest stays the flag, its px named. Made on the two mechanisms of the day: on 0014d her chest skin is Face_Neck below the bib's fold,
    so only the chin takes the fold (810 px left without it, 54 with); on 1010d a thick sliver above her chin stays (157 px)."""
    import tempfile
    W_, H_ = 1024, 768
    names = list(cap._pack("sapiens2_parts").CLASS_NAMES)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for d in ("runs/r", "subjects/a", "subjects/held"):
            (folder / d).mkdir(parents=True)
        track, held = np.zeros((2, H_, W_), bool), np.zeros((2, H_, W_), bool)
        classes = np.zeros((2, H_, W_), np.uint8)
        track[:, 100:700, 300:700] = True
        classes[:, 100:700, 300:700] = names.index("Background")
        classes[:, 150:650, 450:550] = names.index("Face_Neck")          # her chest skin called Face_Neck, down to row 649
        classes[0, 500:520, 320:340] = names.index("Upper_Clothing")
        held[0, 300:303, 300:401] = True                                  # a sliver 3 px thick: 303 px
        held[0, 460:480, 600:620] = True                                  # 14d: the bib's fold, Background under her chin and above row 649: 400 px
        held[0, 500:520, 320:340] = True                                  # on her Upper_Clothing: 400 px
        held[1, 340:350, 640:652] = True                                  # 1010d: a thick sliver above her chin: 120 px
        np.savez(folder / "subjects" / "a" / "masks__p.npz", track=np.packbits(track, axis=-1), parts=np.packbits(track, axis=-1))
        np.savez(folder / "subjects" / "held" / "masks__p.npz", track=np.packbits(held, axis=-1))
        np.savez(folder / "subjects" / "a" / "classes__p.npz", classes=classes)
        np.savez(folder / "runs" / "r" / "region.npz", region=np.ones((2, H_ // 16, W_ // 16), bool))
        (folder / "frames.json").write_text(json.dumps({"rows": [{"source_frame": f, "region_on__r__held": 0.5} for f in (120, 121)]}))
        (folder / "runs" / "r" / "per_frame.json").write_text(json.dumps({"rows": []}))
        m = {"size": [W_, H_], "first_frame": 120, "subjects": [{"label": x, "sightings": [{"by": "p"}]} for x in ("a", "held")],
             "runs": [{"name": "r", "subject": "a", "planned": True, "first_source_frame": 120}]}
        pts = folder / "points.npz"            # the reference's points at half the canvas: row 200 is 400; 121's, a guess, is set high, above the sliver
        kp = np.zeros((3, 2, 2), np.float32)
        kp[:, 1] = [[300, 200], [300, 150], [300, 999]]
        np.savez(pts, keypoints=kp, scores=np.array([[1, 0.9], [1, 0.04], [1, 0.5]], np.float32), names=np.array(["nose", "tip_of_chin"]),
                 first_frame=np.array(120), size=np.array([512, 384]))
        chin = cap.chin_rows(str(pts), (W_, H_))
        assert chin == {120: (400, 0.9), 121: (300, 0.04)}, chin             # scaled to the canvas; a point off it is no point
        own = {"held": []}
        got = {f["rule"]: f for f in cap.flag_runs(m, folder, own, {}, chin) if f.get("other") == "held"}
        # THE CASE THAT GOES RED when slivers or her clothing count against the load, or what is left is not named with its px
        assert got["region_over_held_slivers"]["level"] == cap.LEVELS[0] and got["region_over_held_slivers"]["figures"]["px_a_frame"] == {120: 303}, got["region_over_held_slivers"]
        cl = got["region_over_her_clothing"]
        assert cl["level"] == cap.LEVELS[0] and cl["figures"]["px_a_frame"] == {120: 800} and "her clothing regenerated inside the region" in cl["why"], cl
        assert cl["figures"]["below_her_chin_from"][120] == "the chin point, row 400 and down, score 0.9", cl["figures"]
        left = got["region_over_another_subject"]
        # THE CASE THAT GOES RED when a chin point scored under CHIN_LEAST_SCORE is taken: set high, it would silence 121's sliver as her clothing
        assert left["source_frames"] == [[121, 121]] and "LEFT after the edge slivers and her clothing: 120 px, on 121 (120 px)" in left["why"] \
            and f"chin point scored under {cap.CHIN_LEAST_SCORE} on 121-121: her Face_Neck's lowest row there" in left["why"], left
        assert left["figures"]["below_her_chin_from"][121] == f"CHIN POINT SCORED UNDER {cap.CHIN_LEAST_SCORE} (0.04): her Face_Neck's lowest row, 649", left["figures"]
        # THE CASE THAT GOES RED when "below her chin" is read off Face_Neck's lowest row where a chin point is given: 14d's fold stays flagged
        no_chin = {f["rule"]: f for f in cap.flag_runs(m, folder, own, {}, {}) if f.get("other") == "held"}["region_over_another_subject"]
        assert no_chin["source_frames"] == [[120, 121]] and "LEFT after the edge slivers and her clothing: 520 px, on 120 (400 px), 121 (120 px)" in no_chin["why"] \
            and "no chin point on 120-121: her Face_Neck's lowest row there" in no_chin["why"], no_chin["why"]
        # nothing left: no flag at all, the data stays
        held[1] = False
        np.savez(folder / "subjects" / "held" / "masks__p.npz", track=np.packbits(held, axis=-1))
        rules = [f["rule"] for f in cap.flag_runs(m, folder, own, {}, chin) if f.get("other") == "held"]
        assert rules == ["region_over_held_slivers", "region_over_her_clothing"], rules
        # another subject's mask is not split, and with no class map the held flag is as before
        (folder / "subjects" / "a" / "classes__p.npz").unlink()
        assert [f["rule"] for f in cap.flag_runs(m, folder, own, {}, chin) if f.get("other") == "held"] == ["region_over_another_subject"]
    return ("held's pixels in the region split: edge slivers and her clothing (Upper_Clothing, Apparel, Background below the reference's chin point) are data "
            "with frames and px; the rest stays the flag with its px; without a chin point Face_Neck's lowest row, said so; no class map, as before")


def specks_of_another() -> str:
    """The lead, 2026-10-14: on second_hfp_0014d the strangers mask was specks of 1-657 px on her own apron edge and wrist, at most
    46 px of it inside the region, read as "up to 100%". A frame with under REGION_ON_OTHER_LEAST_PX of another's pixels inside the
    region is said as data, with its size; a frame over it counts as before."""
    import tempfile
    W_, H_ = 1024, 768
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "runs" / "r").mkdir(parents=True)
        (folder / "subjects" / "strangers").mkdir(parents=True)
        shares = {100: 46 / (W_ * H_), 101: 657 / (W_ * H_), 102: 4000 / (W_ * H_)}         # 14d's figures, and one frame of a real someone
        (folder / "frames.json").write_text(json.dumps({"rows": [{"source_frame": f, "region_on__r__strangers": 1.0 if f < 102 else 0.5} for f in shares]}))
        (folder / "subjects" / "strangers" / "per_frame.json").write_text(json.dumps({"rows": [{"source_frame": f, "track_share": v} for f, v in shares.items()]}))
        (folder / "runs" / "r" / "per_frame.json").write_text(json.dumps({"rows": []}))
        m = {"size": [W_, H_], "subjects": [{"label": x, "sightings": [{"by": "p"}]} for x in ("a", "strangers")], "runs": [{"name": "r", "subject": "a", "planned": True}]}
        got = {f["rule"]: f for f in cap.flag_runs(m, folder) if f.get("other") == "strangers"}
        # THE CASE THAT GOES RED when specks of another's mask count against a load as "up to 100%"
        specks, real = got["region_over_another_subject_specks"], got["region_over_another_subject"]
        assert specks["level"] == cap.LEVELS[0] and specks["source_frames"] == [[100, 101]] and specks["figures"]["px_inside_most"] == 657, specks
        assert real["level"] == cap.LEVELS[2] and real["source_frames"] == [[102, 102]] and "at most 2000 px of it inside the region" in real["why"], real
        (folder / "frames.json").write_text(json.dumps({"rows": [{"source_frame": f, "region_on__r__strangers": 1.0} for f in (100, 101)]}))
        got = [f["rule"] for f in cap.flag_runs(m, folder) if f.get("other") == "strangers"]
        assert got == ["region_over_another_subject_specks"], "specks alone still made a top-level flag"
        # THE CASE THAT GOES RED when the floor reaches a subject of ours: a small face of 700 px inside the region still counts (the lead)
        (folder / "subjects" / "third").mkdir()
        (folder / "subjects" / "third" / "per_frame.json").write_text(json.dumps({"rows": [{"source_frame": 100, "track_share": 700 / (W_ * H_)}]}))
        (folder / "frames.json").write_text(json.dumps({"rows": [{"source_frame": 100, "region_on__r__third": 1.0}]}))
        m3 = dict(m, subjects=[{"label": x, "sightings": [{"by": "p"}]} for x in ("a", "third")])
        third = [f for f in cap.flag_runs(m3, folder) if f.get("other") == "third"]
        assert [f["rule"] for f in third] == ["region_over_another_subject"] and third[0]["level"] == cap.LEVELS[2], third
    return (f"under {cap.REGION_ON_OTHER_LEAST_PX} px of the strangers mask inside the region a frame is data with its size; over it counts, with the "
            "px said; a subject of ours counts at any size")


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
    # A sentence that names a reference picture says what is taken from a still, never what is performed (2026-10-13: the
    # lead's retention line was read as a voice denied, and the same line with no "not" as a voice performed).
    kept = ("<Subject 1> (appears in [Shot 1]): partially_preserved - only the girl's face is retained, wherever it shows: her eyes, eyebrows, "
            "nose, mouth and skin; the hair, the clothing and the background of <Picture 1> are not used.")
    kept_plain = "<Subject 1> (appears in [Shot 1]): partially_preserved - only the face is retained: the eyes, eyebrows, nose, mouth and skin of <Picture 1>."
    for line in (kept, kept_plain):
        assert cap.voice_sentences(line) == ([], []), f"a retention line was read as a voice sentence: {cap.voice_sentences(line)}"
        assert cap.flag_text("She is in a room. " + line, [[500, 600]], 100, 50) == [], "a retention line over silence was flagged as naming a voice"
        said = cap.flag_text("She is in a room. " + line, [[120, 130]], 100, 50)
        assert [(f["rule"], f["level"]) for f in said] == [("voiced_frames_and_no_voice_sentence", cap.LEVELS[0])] and "says nothing of one" in said[0]["why"], said
    # beside a real voice sentence, or a real denial, the retention line changes nothing
    assert cap.flag_text(sings + " " + kept, [[120, 130]], 100, 50) == [] and cap.voice_sentences(denies + " " + kept)[1] == ["She does not speak or sing at any point."]
    # A sentence defining a subject by a reference video is read by its head (2026-10-14: the drawn mouth's words were read as a voice on
    # lead_fp_1097c, which has none). 1097c's own sentences, as its text file holds them, and the same sentence as load_text writes it now.
    drawn_1097c = ("<Subject 2> is the movements of the head and the facial expressions performed by the figure in <Video 1>: a closer view of the head "
                   "and shoulders of a plain grey body model of a person on a black ground, posed as the person moves, with its eyes, eyebrows, nose "
                   "and mouth drawn on its face as thin white outlines; inside each eye's outline the white of the eye is white and the iris black, and "
                   "they open, close and look where the person's eyes do; the mouth is outlined, dark inside where it is open, and it opens and closes "
                   "where the person's mouth does.")
    sound_1097c = "overall_soundscape: Natural ambient atmosphere continues throughout the shot.\n\nnon_diegetic_music: A song plays throughout."
    import load_text
    written_now = load_text.subject_2({"mouth": {"drawn": True, "inside": "the class map"}, "iris": "whole"})
    for line in (drawn_1097c, written_now, sound_1097c):
        # THE CASE THAT GOES RED when the reference video's drawn mouth, or the music line, is read as a voice performed
        assert cap.voice_sentences(line) == ([], []), (line[:60], cap.voice_sentences(line))
    assert cap.flag_text("subject_definitions:\n" + drawn_1097c + "\n\n" + sound_1097c, [], 1097, 37) == []
    # the head still counts, and the singing sentence load_text writes beside it is still a voice sentence
    sung_head = "<Subject 2> is the singing and the lip movements performed by the figure in <Video 1>: a grey body model on a black ground."
    assert cap.voice_sentences(sung_head)[0] == [sung_head], cap.voice_sentences(sung_head)
    assert cap.voice_sentences(drawn_1097c + "\n<Subject 1> (S1) sings, her mouth opening wide on the sung phrases.")[0] == [
        "<Subject 1> (S1) sings, her mouth opening wide on the sung phrases."]
    return ("a voice sentence over silence, a denial over a voice; neither the other way round; a sentence naming a reference picture is neither, "
            "whatever its words; one defining a subject by a reference video is read by its head (1097c's drawn mouth is no voice)")


def head_tube_in_the_files_step() -> str:
    """When --head-tube is enabled, the files step bounds the head tube by the person track, bridges 1-frame dropouts, and writes provenance."""
    import tempfile
    n = 4
    cls_names = list(cap._pack("sapiens2_parts").CLASS_NAMES)
    face_cls = cls_names.index("Face_Neck")
    torso_cls = cls_names.index("Torso")
    # Person track covers x=20..60, y=10..80 across all 4 frames.
    ht_track = rect(20, 10, 60, 80, n)
    # Head part is at x=28..52, y=14..38 on all frames, except:
    # - Frame 1 also has an off-person false-positive Face_Neck blob at x=80..96, y=20..36 (outside ht_track)
    # - Frame 2 has a 1-frame Sapiens2-Seg dropout on the head (empty at x=28..52, y=14..38)
    ht_parts = rect(28, 14, 52, 38, n)
    ht_parts[1, 20:36, 80:96] = True
    ht_parts[2, 14:38, 28:52] = False
    ht_cmap = np.where(ht_parts, face_cls, 0).astype(np.uint8)
    ht_cmap[:, 44:70, 26:54] = torso_cls

    stood_in = {k: getattr(cap, k) for k in ("probe", "read_mask", "read_classes", "write_mask_video", "input_facts", "picture_rows")}
    with tempfile.TemporaryDirectory() as tmp:
        f_src = Path(tmp) / "src.mp4"
        f_src.write_bytes(b"")
        off_dir = Path(tmp) / "cap_off"
        on_dir = Path(tmp) / "cap_on"
        written_off: dict[str, np.ndarray] = {}
        written_on: dict[str, np.ndarray] = {}
        try:
            cap.probe = lambda _p: (W, H, n)
            cap.read_mask = lambda p, *_a: ((ht_track.copy() if "track" in str(p) else ht_parts.copy()), np.ones(n, bool))
            cap.read_classes = lambda *_a: ht_cmap.copy()
            cap.input_facts = lambda p: {"file": str(p), "written": "2026-10-17T12:00:00", "size": [W, H], "frames": n}
            cap.picture_rows = lambda *_a: [(0.0, 0.0)] * n

            cap.write_mask_video = lambda p, m: written_off.__setitem__(Path(p).name, m.copy())
            args_off = argparse.Namespace(first=100, frames=n, name="off", date="2026-01-01", source=str(f_src), loads=None, voice=None, head_tube=False)
            spec_a = {"track": "track.mkv", "parts": "parts.mkv", "classes": "classes.mkv", "by": "pv", "at": "100", "least": "50"}
            cap._files(args_off, off_dir, [("a", dict(spec_a))], [], [])

            cap.write_mask_video = lambda p, m: written_on.__setitem__(Path(p).name, m.copy())
            args_on = argparse.Namespace(first=100, frames=n, name="on", date="2026-01-01", source=str(f_src), loads=None, voice=None, head_tube=True)
            cap._files(args_on, on_dir, [("a", dict(spec_a))], [], [])

            try:
                cap._files(args_off, Path(tmp) / "bad", [("a", {**spec_a, "head_tube": "maybe"})], [], [])
                raise AssertionError("accepted invalid head_tube=maybe")
            except SystemExit as exc:
                assert "head_tube=1 or --head-tube" in str(exc), exc
        finally:
            for k, v in stood_in.items():
                setattr(cap, k, v)

        assert not (off_dir / "subjects" / "a" / "head_tube__pv.json").is_file(), "head_tube ran when off by default"
        off_sighting = json.loads((off_dir / "manifest.json").read_text())["subjects"][0]["sightings"][0]
        assert "head_tube" not in off_sighting, off_sighting

        prov_path = on_dir / "subjects" / "a" / "head_tube__pv.json"
        npz_path = on_dir / "subjects" / "a" / "head_tube__pv.npz"
        assert prov_path.is_file() and npz_path.is_file(), "head_tube provenance and masks not written when enabled"
        prov = json.loads(prov_path.read_text())
        assert prov["outside_person_track_clipped_px"] == [0, 16 * 16, 0, 0], prov["outside_person_track_clipped_px"]
        assert not (written_on["parts__pv.mkv"] & ~ht_track).any(), "parts__pv.mkv leaked outside person_track"
        assert int(written_on["parts__pv.mkv"][2, 14:38, 28:52].sum()) == 24 * 24, "1-frame dropout on frame 2 was not bridged by head_tube"
        on_sighting = json.loads((on_dir / "manifest.json").read_text())["subjects"][0]["sightings"][0]
        assert on_sighting.get("head_tube") is True and on_sighting.get("head_tube_provenance") == "subjects/a/head_tube__pv.json", on_sighting
        for k in ("facing_frames", "side_with_eye_frames", "back_of_head_frames", "proposed_drop_frames"):
            assert k in on_sighting.get("head_tube_summary", {}), f"missing {k} in manifest sighting head_tube_summary: {on_sighting}"
        on_pg = json.loads((on_dir / "subjects" / "a" / "part_grades.json").read_text())
        assert on_pg.get("head_tube") is True and on_pg.get("head_tube_provenance") == "subjects/a/head_tube__pv.json", on_pg
        for k in ("facing_frames", "side_with_eye_frames", "back_of_head_frames", "proposed_drop_frames"):
            assert k in on_pg, f"missing {k} in part_grades.json: {on_pg}"
        on_classes = np.load(on_dir / "subjects" / "a" / "classes__pv.npz")["classes"]
        assert (on_classes[:, 44:70, 26:54] == torso_cls).all(), "non-head body classes in classes__pv.npz were lost"
    return "off by default; when on, bounds head tube by person_track, bridges 1-frame dropout, preserves torso classes, and writes provenance"


def back_of_head_and_no_face_never_held_by_grade() -> str:
    """A back_of_head frame or a noface frame is never filled by neighbour fill, short-gap fill, or the floor rule;
    back_of_head is a proposal that never empties a standing part by itself; and a person's word= overrides back_of_head."""
    n = 6
    track = rect(20, 10, 68, 80, n)
    # Standing face part is 24x24 = 576 px (above least_px=480) on frames 0, 2, 4, 5.
    parts = rect(28, 16, 52, 40, n)
    # Frame 1: a 25 px speck (5x5 < least_px=480) that the floor rule would normally hold and fill from frame 0/2.
    parts[1] = False
    parts[1, 20:25, 34:39] = True
    # Frame 3: completely empty (0 px) between standing frames 2 and 4, where neighbour/short-gap fill would normally fill.
    parts[3] = False

    filled = rect(28, 16, 52, 40, n)
    cmap = np.zeros((n, H, W), dtype=np.uint8)
    cmap[parts] = 1  # Face_Neck where standing part / speck exists
    cmap[3, 16:40, 28:52] = 1  # Offer valid Face_Neck under frame 3's fill to test neighbour/short-gap exclusion on back_of_head
    cmap[4, 16:40, 28:52] = 1  # Standing part on frame 4

    # 1. Control without back_of_head or noface: frame 1 is held by the floor rule (filled to 576 px), frame 3 is filled.
    ctrl_kept, ctrl_grade = cap.grade_holds(
        parts, filled, cmap, [1], least_px=480, lost={3}, track=track
    )
    assert ctrl_grade[1]["did"] == "held by the floor rule" and int(ctrl_kept[1].sum()) == 576, (ctrl_grade.get(1), int(ctrl_kept[1].sum()))
    assert int(ctrl_kept[3].sum()) == 576, (ctrl_grade.get(3), int(ctrl_kept[3].sum()))

    # 2. With back_of_head={1, 3, 4}:
    #    - Frame 1 (speck < least_px) is NOT held by the floor rule; it is emptied and records not_held="back of the head..."
    #    - Frame 3 (empty gap) is NOT filled by neighbour/short-gap fill; it stays empty.
    #    - Frame 4 (standing 576 px >= least_px, no drop=) is NOT emptied by back_of_head (proposal only!).
    boh_kept, boh_grade = cap.grade_holds(
        parts, filled, cmap, [1], least_px=480, lost={3}, track=track, back_of_head={1, 3, 4}
    )
    assert boh_grade[1]["did"] == "emptied" and not boh_kept[1].any(), (
        f"back_of_head frame 1 was held by the floor rule: grade={boh_grade.get(1)} px={int(boh_kept[1].sum())}"
    )
    assert "back of the head" in boh_grade[1].get("not_held", ""), boh_grade[1]
    assert not boh_kept[3].any(), (
        f"back_of_head frame 3 was filled across gap: grade={boh_grade.get(3)} px={int(boh_kept[3].sum())}"
    )
    assert int(boh_kept[4].sum()) == 576, (
        f"back_of_head proposal emptied standing frame 4 without drop=: grade={boh_grade.get(4)} px={int(boh_kept[4].sum())}"
    )

    # 3. Person's word= overrides back_of_head and fills frame 1 from its neighbour:
    word_kept, word_grade = cap.grade_holds(
        parts, filled, cmap, [1], least_px=480, word={1}, track=track, back_of_head={1}
    )
    assert int(word_kept[1].sum()) == 576, (
        f"person's word= did not override back_of_head on frame 1: grade={word_grade.get(1)} px={int(word_kept[1].sum())}"
    )
    assert word_grade[1]["did"] == "filled on a person's word", word_grade[1]

    # 4. Person's noface= prevents the floor rule from holding frame 1 even when not back_of_head:
    nf_kept, nf_grade = cap.grade_holds(
        parts, filled, cmap, [1], least_px=480, track=track, noface={1}
    )
    assert nf_grade[1]["did"] == "emptied" and not nf_kept[1].any(), (
        f"noface frame 1 was held by the floor rule: grade={nf_grade.get(1)}"
    )
    assert nf_grade[1].get("not_held") == "no face on a person's word (--no-face-on)", nf_grade[1]
    return "back_of_head and noface are never held by the floor rule or neighbour fill; back_of_head empties nothing by itself; word= overrides back_of_head"


def depth_in_owner_map_and_the_files_step() -> str:
    """`--depth` (`bench/capture_masked_run.py`): settles contested pixels in `owner_map` when core depth medians
    separate (`|z_a - z_b| > max(0.03, 0.03 * min(z_a, z_b))`), leaves background slits and co-planar overlaps
    contested, clips a back subject's `parts` off a front subject's owned pixels in `files`, preserves keep-out
    pixels strictly in front of the replaced subject in `carve_keep_out` and `grid_carve`, and refuses on canvas
    or overlap-frame mismatches with no traceback."""
    import argparse
    import tempfile
    from _lib import refused
    one, two = rect(16, 16, 64, 64), rect(48, 16, 96, 64)          # overlap in x=48..64, y=16..64 (16x48 = 768 px)
    map_one = np.where(one, 1, 0).astype(np.uint8)
    map_two = np.where(two, 1, 0).astype(np.uint8)
    # Without depth: both class maps claim the overlap, so all 768 overlap pixels are CONTESTED.
    ow_no, lbls, sh_no, cont_no = cap.owner_map({"one": one, "two": two}, {"one": map_one, "two": map_two})
    assert lbls == ["one", "two"] and int(sh_no[0]) == 768 and int(cont_no[0]) == 768, (sh_no[0], cont_no[0])

    # Build synthetic depth: `one` (back) at 1.95m, `two` (front) at 1.20m; in overlap y=16..56 is at 1.20m (front),
    # while y=56..64 is a 3.50m background slit behind the back subject (`3.50 > 1.95 + 0.02`).
    depth = np.full((N, H, W), 2.50, np.float32)
    valid = np.ones((N, H, W), bool)
    depth[:, 16:64, 16:48] = 1.95
    depth[:, 16:64, 64:96] = 1.20
    depth[:, 16:56, 48:64] = 1.20
    depth[:, 56:64, 48:64] = 3.50
    ow_yes, _, sh_yes, cont_yes = cap.owner_map({"one": one, "two": two}, {"one": map_one, "two": map_two}, depth=depth, valid=valid)
    assert int(sh_yes[0]) == 768 and int(cont_yes[0]) == 16 * 8, (sh_yes[0], cont_yes[0])
    assert (ow_yes[:, 16:56, 48:64] == 1).all(), "foreground overlap was not given to front subject `two`"
    assert (ow_yes[:, 56:64, 48:64] == cap.CONTESTED).all(), "background slit behind back subject was not left CONTESTED"

    # Co-planar subjects (`|z_one - z_two| <= tau_z`): depth does not guess, overlap stays CONTESTED.
    depth_close = depth.copy()
    depth_close[:, 16:64, 16:48] = 1.50
    depth_close[:, 16:64, 64:96] = 1.52
    _, _, _, cont_close = cap.owner_map({"one": one, "two": two}, {"one": map_one, "two": map_two}, depth=depth_close, valid=valid)
    assert int(cont_close[0]) == 768, f"co-planar overlap within tau_z was guessed: {cont_close[0]}"

    # Keep-out carving with depth: a foreground keep-out at 1.10m in front of `one` (1.95m) is preserved; one at 1.95m is carved.
    keep_fg = rect(24, 24, 40, 40)
    depth_fg = depth.copy()
    depth_fg[:, 24:40, 24:40] = 1.10
    carved_fg, px_fg, _ = cap.carve_keep_out(keep_fg, one, depth=depth_fg, valid=valid)
    assert (carved_fg == keep_fg).all() and int(px_fg.sum()) == 0, "foreground keep-out in front of subject was carved"
    carved_co, px_co, _ = cap.carve_keep_out(keep_fg, one, depth=depth, valid=valid)
    assert not carved_co.any() and int(px_co[0]) == 16 * 16, "co-planar keep-out inside subject track was not carved"

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        plan_p = _a_node_plan(tmp, rect(20, 20, 44, 44), None, 2)
        band = rect(44, 20, 48, 44)  # inside the cell 32..48 that `one` (16..64) owns
        depth_band = depth.copy()
        depth_band[:, 20:44, 44:48] = 1.10
        g_fg, lost_fg = cap.grid_carve(band, one, plan_p, 100, 100, depth=depth_band, valid=valid)
        assert (g_fg == band).all() and int(lost_fg.sum()) == 0, "grid_carve carved a foreground keep-out"
        g_co, lost_co = cap.grid_carve(band, one, plan_p, 100, 100, depth=depth, valid=valid)
        assert not g_co.any() and int(lost_co.sum()) > 0, "grid_carve left a co-planar keep-out on a shared cell"

        # Refusal cases on read_depth_npz: canvas mismatch, missing overlap frames, and missing keys.
        bad_canvas = tmp / "bad_canvas.npz"
        np.savez_compressed(bad_canvas, depth=np.ones((N, H * 2, W * 2), np.float32), mask=np.ones((N, H * 2, W * 2), bool),
                            frames=np.arange(100, 100 + N, dtype=np.int32))
        refused(lambda: cap.read_depth_npz(bad_canvas, (W, H), 100, N), "--depth bad_canvas.npz", "depth canvas is 320x192")

        short_frames = tmp / "short_frames.npz"
        np.savez_compressed(short_frames, depth=depth[:1], mask=valid[:1], frames=np.array([100], dtype=np.int32))
        refused(lambda: cap.read_depth_npz(short_frames, (W, H), 100, N, required_frames={100, 101, 102}),
                "--depth short_frames.npz", "lacks overlap source frame(s) 101-102")

        bad_keys = tmp / "bad_keys.npz"
        np.savez_compressed(bad_keys, depth=depth[:1], frames=np.array([100], dtype=np.int32))
        refused(lambda: cap.read_depth_npz(bad_keys, (W, H), 100, N), "--depth bad_keys.npz", "missing mask")

        # Full `_files` step with `--depth`: back subject `one`'s part spills onto front subject `two`'s owned overlap (x=48..60)
        # and is clipped before `held_parts`/`grade_holds`, while `owners.npz` records `depth_settled=True` and `z_by_frame`.
        good_depth = tmp / "good_depth.npz"
        np.savez_compressed(good_depth, depth=depth, mask=valid, frames=np.arange(100, 100 + N, dtype=np.int32))
        part_one = rect(24, 20, 60, 44)  # 24..48 is inside `one` alone (24x24=576 px); 48..60 spills into `two`'s owned overlap (12x24=288 px)
        part_two = rect(52, 20, 80, 44)
        masks_given = {"one.mkv": one, "two.mkv": two, "p1.mkv": part_one, "p2.mkv": part_two}
        classes_given = {"c1.mkv": map_one, "c2.mkv": map_two}
        stood_in = {k: getattr(cap, k) for k in ("probe", "read_mask", "read_classes", "write_mask_video", "picture_rows", "input_facts")}
        cap.probe = lambda _p: (W, H, N)
        cap.read_mask = lambda path, *_a: (masks_given[Path(path).name].copy(), np.ones(N, bool))
        cap.read_classes = lambda path, *_a: classes_given[Path(path).name].copy()
        cap.write_mask_video = lambda *_a: None
        cap.picture_rows = lambda *_a: [(0.0, 0.0)] * N
        cap.input_facts = lambda _p: {}
        try:
            out_cap = tmp / "cap_depth"
            args_ = argparse.Namespace(first=100, frames=N, name="cap_depth", date="2026-01-01", source="s.mp4",
                                       loads=None, voice=None, head_tube=False, depth=str(good_depth))
            given = [
                ("one", {"track": "one.mkv", "parts": "p1.mkv", "classes": "c1.mkv", "least": "0"}),
                ("two", {"track": "two.mkv", "parts": "p2.mkv", "classes": "c2.mkv", "least": "0"}),
            ]
            cap._files(args_, out_cap, given, [], [])
            ow_npz = np.load(out_cap / "owners.npz")
            assert bool(ow_npz["depth_settled"]) is True and ow_npz["z_by_frame"].shape == (2, N), ow_npz.files
            m_one = np.load(out_cap / "subjects" / "one" / "masks__preview.npz")
            p1_out = np.unpackbits(m_one["parts"], axis=-1)[:, :, :W].astype(bool)
            assert int(p1_out[0].sum()) == 24 * 24 and not p1_out[:, :, 48:60].any(), (
                f"back subject `one`'s part spill onto `two` was not clipped: {int(p1_out[0].sum())}"
            )
            man = json.loads((out_cap / "manifest.json").read_text())
            s_one = next(s for s in man["subjects"] if s["label"] == "one")["sightings"][0]
            assert s_one["parts_clipped_by_depth"]["px"] == N * 12 * 24, s_one.get("parts_clipped_by_depth")
            assert man["owners"]["depth"]["settled_by_depth_px"] == N * 16 * 40, man["owners"]

            # One-subject load shape (load_chain.py::mask_specs passes classes= only for the load's subject, and track= only for others):
            # one-sided class map does not pre-settle shared pixels, and owners.npz still includes `two` so `--depth` settles the overlap.
            out_one_subj = tmp / "cap_depth_one_subj"
            given_one_subj = [
                ("one", {"track": "one.mkv", "parts": "p1.mkv", "classes": "c1.mkv", "least": "0"}),
                ("two", {"track": "two.mkv"}),
            ]
            cap._files(args_, out_one_subj, given_one_subj, [], [])
            ow_one_subj = np.load(out_one_subj / "owners.npz")
            assert list(ow_one_subj["labels"]) == ["one", "two"] and bool(ow_one_subj["depth_settled"]) is True
            m_one_s = np.load(out_one_subj / "subjects" / "one" / "masks__preview.npz")
            p1_out_s = np.unpackbits(m_one_s["parts"], axis=-1)[:, :, :W].astype(bool)
            assert int(p1_out_s[0].sum()) == 24 * 24 and not p1_out_s[:, :, 48:60].any()
        finally:
            for k, v in stood_in.items():
                setattr(cap, k, v)
    return "owner_map settles overlap by core depth and leaves background slits and co-planar overlaps contested; files --depth clips back-subject part spills and preserves foreground keep-outs (including on one-subject loads); canvas and frame mismatches refused"


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
case("a load rendered zoomed in, read onto the canvas", zoomed_regions)
case("a pass read from the node's own plan", planned_regions)
case("preflight: the original shown in a held tail", held_tails)
case("preflight: a face only a few tokens tall", face_sizes)
case("preflight: a shot where a tracker's call is nobody's word", who_is_who)
case("shots: the gate's answer per shot and subject", gate_by_shot)
case("the look: what a strip shows of a load and its text, and the line that records it", a_look)
case("preflight: a load that lays nothing where its subject is", a_load_that_lays_nothing)
case("preflight: what the job says is its own", what_the_job_says_is_its_own)
case("preflight: specks of another are data", specks_of_another)
case("a strip says what the node will render", what_the_node_will_render)
case("the strip's marks and its frame records", the_strips_marks_and_records)
case("the floor at the size the load renders at", the_floor_at_the_size_it_renders)
case("preflight: held's slivers and her clothing are data", held_split)
case("a fill of a doubted part, graded by the class map", graded_holds)
case("a short fill where nothing was labelled", a_short_fill_where_nothing_was_labelled)
case("a part held on a person's word", a_part_held_on_a_persons_word)
case("the floor under a small part: a starting value, held against the clip's own sizes", the_floor_against_the_clip)
case("verify: the capture reads what the nodes wrote", verifying)
case("a pose table as an input", pose_tables)
case("a pass as a list of loads", loads)
case("preflight: the gate and its override", the_gate)
case("preflight: the text against the voice", text_rules)
case("the strip: the frames a render's strip shows, and the span's frames it leaves out", a_strip_of_a_render)
case("the strip: a load of two subjects, each in its own colour", a_strip_of_two_subjects)
case("a load of two subjects: neither is another subject to it", neither_is_another)
case("no largest piece of a face or hair, at the files step", no_pieces_on_a_head)
case("no keep-out carves the subject's own track", no_keep_out_inside_her_track)
case("the floor holds", the_floor_holds)
case("off her head", off_her_head)
case("the carve, in the files step", the_carve_in_the_files_step)
case("the carve on the node's own grid", the_carve_on_the_nodes_grid)
case("the head tube, in the files step", head_tube_in_the_files_step)
case("back of the head and no-face frames are never held by the grade", back_of_head_and_no_face_never_held_by_grade)
case("depth in the owner map and the files step", depth_in_owner_map_and_the_files_step)
sys.exit(finish())
