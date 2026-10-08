#!/usr/bin/env python3
"""The model-free judgements about a tracked subject (`subject_tracks.py`), on made-up masks. No model, no card.

Each case is a way a track nobody should trust could be reported as held, or a correction could land on the wrong shot:

  a_strip_is_not_a_subject        the shapes the independent run of 2026-10-07 saw pass for subjects (a strip along the
                                  bottom edge, one along the right edge, a fragment of a few hundred pixels) fail the
                                  shape test, a scatter of specks fails, and a person, a small far person and a person
                                  cut by the frame's edge pass. THE CONTROL: counting non-empty masks calls every one
                                  of them a subject.
  the_same_at_any_size            every shape gets the same verdict on a 1008 grid, a 252 grid and a 1344 x 760 frame.
  a_slid_track_is_doubted         a track that leaves its person for the frame's edge while the detector still sees
                                  the person is doubted from just after the look before the one that disagrees until
                                  the next look or the first empty frame; KNOWN LIMIT, asserted: a swap onto a
                                  neighbour the detector also sees is not doubted;
                                  a track the detector agrees with is never doubted; an empty stretch is never
                                  doubted; a track seeded again after a loss starts with no doubt against it; frames
                                  before the first look are not doubted.
  trusted_is_both                 a frame is trusted only with a plausible mask the detector does not contradict.
  a_held_figure_is_one_run        a figure that moves a little every frame, seeded in the middle of its track, is one
                                  run from the first frame to the last; the seed has no step; a seed frame with no
                                  mask gives an empty run, and a seed outside the track is refused.
  a_jump_is_cut                   a mask that leaves its figure for another with no empty frame between is cut at the
                                  jump, tracked forward and tracked backward, and the frames past it are outside the
                                  run though each is steady on the other figure. THE CONTROLS: a count of non-empty
                                  masks and the shape test call the whole track held; with the line at zero the whole
                                  track is one run.
  a_return_after_a_gap_is_outside  the run stops at the first empty frame; a mask that comes back, in the same place
                                  or another, is outside it, and its step across the gap says which with how many
                                  frames lie between.
  a_gallery_is_taken_inside_the_run  the Subject Track's own `gallery_frames` given the run takes no frame from after
                                  a jump. THE CONTROL: given the whole track, as today's node gives it, it does.
  the_frame_beside_a_jump_is_no_gallery_frame  a frame that lies on its figure and has a small piece on the next,
                                  one frame before the mask jumps there, is kept by the cut (it shares most of its
                                  pixels with the frame before) and left out of the gallery's span; its box ratio and
                                  centre step stand out and decide nothing. A run that ends on an empty frame or the
                                  track's end loses no frame, and the seed is never left out. THE CONTROL: the Subject
                                  Track's own `gallery_frames` given the run as the cut leaves it takes that frame.
  a_jump_without_warning_costs_a_good_frame  KNOWN COST, asserted: `gallery_span` leaves out the frame beside
                                  every jump, and a jump can come with no warning. The frame before it is then an
                                  ordinary frame of the subject (the same mask as the one before, by every measure)
                                  and is left out all the same. Seen on 2026-10-07 at a second jump on the same
                                  stretch as the first (`bench/results/2026-10-07_subject_track_calls_on_masks.md`).
                                  The trim was reasoned from one jump; this is a case it does not describe, and it
                                  costs one good frame and adds no bad one.
  a_creep_is_not_caught           KNOWN LIMIT, asserted so it is never read as covered: a mask that grows from one
                                  figure over the next and shrinks onto it ends sharing nothing with its seed and is
                                  one run, because no single step shares too little; the area ratio shows the growth.
                                  This is a test of continuity, not of identity.
  taking_back_is_by_likeness_then_place  after a loss, a clear leader is taken on likeness wherever they stand and
                                  nobody under the line is; a call too close by likeness goes to the candidate where the
                                  subject last was, only if that one is near, like-sized and clearly nearer than the
                                  others. THE CONTROLS: with the place switched off, or with no last place, the same
                                  close call takes nobody, as today's node does.
  a_clear_leader_elsewhere_is_refused  with `leader_must_be_near`, a clear leader far from where the subject last was,
                                  or of another size, is refused and the reason says so; near, it is taken; with no
                                  last place it is taken on likeness, as across a cut. THE CONTROL: without the option
                                  the far leader is taken, which is the swap today's node made on a crowd.
  a_slow_move_off_passes_the_default  KNOWN LIMIT, asserted: a mask that leaves its figure in steps that each share
                                  a little more than the default line, and then goes empty, is one run to the empty
                                  frame and loses no frame to the gallery's span, though its last frame shares nothing
                                  with its seed. Seen on 2026-10-07 in one arm (the largest figure on the crowd clip's
                                  hard stretch, corrected, seeded with the three nearest, as fed; its twin with the
                                  input moved one level stepped under the line and was cut):
                                  `bench/results/2026-10-07_subject_alone_or_in_a_group.json`, "crowd, hard, the
                                  largest". The line was not moved on it: the gap between that step and a held
                                  figure's least step is thin on both sides, on one figure. A caller's higher line
                                  cuts it, which the case also asserts.
  taking_back_by_place_asks_where_first  the one candidate over the line who stands where the subject last was is
                                  taken though another scores higher somewhere else; nobody is when none stands
                                  there, when two who may be asked do, when the one who stands there may not be asked,
                                  or when there is no last place. First by a margin takes a candidate under the line
                                  and refuses one who is not first. The result names the likeness rule that ran, the
                                  count asked and returned and whether the cap was reached, and the taken candidate's
                                  rank and lead. THE CONTROL: `take_back`, likeness first, takes the higher score
                                  standing somewhere else on the same candidates. KNOWN LIMIT, asserted: two figures
                                  who have changed places are taken wrongly.
  stray_specks_go_and_the_subject_stays_whole  a piece of a tracked mask that is tiny beside the frame's largest piece
                                  AND away from it is removed, and the frames and pixels are counted; a large piece far
                                  away (an arm past somebody's head), a few pixels beside the subject, a frame's only
                                  piece and an empty frame are untouched; each test alone removes nothing; the reach
                                  follows the subject's size; a speck straight above goes; a diagonal tail off the
                                  subject's corner stays theirs; the input is not written to unless asked, and in
                                  place gives the same answer. THE CONTROL: the input mask holds the speck.
  the_one_in_the_way_is_named_by_the_box  of the detections on a frame that are not the subject, the one whose mask
                                  covers most of the subject's box is named, whether or not the detector also returned
                                  the subject; a neighbour whose edge touches the box is not; with nobody in the box
                                  nobody is named. THE CONTROL: by mask against mask the person in front overlaps the
                                  subject's mask by nothing, which is why the box is asked.
  notes_say_a_place_in_the_clip   the four forms of a corrections line read as written; a frame and its time are the
                                  same place; each kind of typo is refused with the line quoted, never skipped.
  a_note_lands_in_its_shot        a note is placed by where the load starts in the clip, the same text serves two
                                  loads, and a note outside a load is returned as outside, not dropped.
  the_text_to_type_round_trips    for every frame of a load, at two rates and a start that is not a whole second,
                                  `place_text` read back by `parse_notes` lands on that frame's shot.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_subject_tracks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from _lib import case, finish  # noqa: E402

import torch  # noqa: E402

import subject_tracks as S  # noqa: E402


def _blank(h: int, w: int) -> torch.Tensor:
    return torch.zeros((h, w), dtype=torch.bool)


def _shapes(h: int, w: int) -> dict[str, tuple[torch.Tensor, bool]]:
    """Named masks on an h x w frame, each with whether it should count as a subject. Sizes are shares of the frame."""
    def box(y0, y1, x0, x1):
        m = _blank(h, w)
        m[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)] = True
        return m

    out = {
        "a person": (box(0.30, 0.90, 0.40, 0.55), True),
        "a small far person": (box(0.50, 0.58, 0.20, 0.23), True),
        "a person cut by the left edge": (box(0.20, 0.95, 0.0, 0.12), True),
        "a strip along the bottom edge": (box(0.975, 1.0, 0.0, 1.0), False),
        "a strip along the right edge": (box(0.0, 1.0, 0.985, 1.0), False),
        "a fragment": (box(0.50, 0.515, 0.50, 0.515), False),
        "nothing": (_blank(h, w), False),
    }
    specks = _blank(h, w)
    specks[::max(h // 12, 1), ::max(w // 12, 1)] = True
    specks[1::max(h // 12, 1), ::max(w // 12, 1)] = True
    specks[::max(h // 12, 1), 1::max(w // 12, 1)] = True
    out["a scatter of specks"] = (specks if specks.sum() >= S.FRAGMENT_SHARE * h * w else box(0, 0, 0, 0), False)
    return out


def a_strip_is_not_a_subject():
    shapes = _shapes(1008, 1008)
    got = {name: bool(S.plausible(m)) for name, (m, _) in shapes.items()}
    wrong = [name for name, (_, want) in shapes.items() if got[name] != want]
    assert not wrong, f"wrong on: {wrong}"
    held_by_count = [name for name, (m, want) in shapes.items() if bool(m.any()) and not want]
    assert len(held_by_count) >= 4, "the control: a count of non-empty masks should call the strips, the fragment and the specks subjects"
    stacked = torch.stack([m for m, _ in shapes.values()])
    assert S.plausible(stacked).tolist() == [want for _, want in shapes.values()], "a stack of masks is judged differently from each alone"
    return f"{sum(got.values())} of {len(got)} shapes pass; a non-empty count would pass {len(held_by_count)} that should not"


def the_same_at_any_size():
    for h, w in ((252, 252), (760, 1344), (1008, 1008)):
        wrong = [name for name, (m, want) in _shapes(h, w).items() if bool(S.plausible(m)) != want and name != "a scatter of specks"]
        assert not wrong, f"at {w}x{h}, wrong on: {wrong}"
    return "the same verdicts at 252x252, 1344x760 and 1008x1008"


def _clip_with_a_slide():
    """Twelve frames, 64 x 64: the detector sees one person at the same place throughout. The track is on them for
    frames 0 to 3, slides to the bottom edge for 4 to 7, is empty for 8 and 9, and is back on them for 10 and 11."""
    person = _blank(64, 64)
    person[16:48, 24:36] = True
    strip = _blank(64, 64)
    strip[62:, :] = True
    track = torch.stack([person] * 4 + [strip] * 4 + [_blank(64, 64)] * 2 + [person] * 2)
    return track, person


def a_slid_track_is_doubted():
    track, person = _clip_with_a_slide()
    looks = {f: person[None] for f in (2, 5, 8, 10)}
    got = S.doubted(track, looks).tolist()
    want = [False] * 3 + [True] * 5 + [False] * 4      # from after the last look (frame 2) to the look on frame 8, where it is empty
    assert got == want, f"doubted frames: {[i for i, d in enumerate(got) if d]}"
    assert not S.doubted(torch.stack([person] * 12), looks).any(), "a track the detector agrees with was doubted"
    late = S.doubted(track, {5: person[None]}).tolist()
    assert late == [False] * 5 + [True] * 3 + [False] * 4, f"with one look, doubted frames: {[i for i, d in enumerate(late) if d]}"
    nobody = S.doubted(track, {1: torch.zeros((0, 64, 64), dtype=torch.bool)}).tolist()
    assert nobody[:8] == [False] + [True] * 7 and not any(nobody[8:10]), "a look that finds nobody should doubt the track from there, and never an empty frame"
    # THE KNOWN LIMIT, asserted so it is never read as covered: a track that slid onto a NEIGHBOUR the detector also sees
    neighbour = _blank(64, 64)
    neighbour[16:48, 44:56] = True
    swapped = torch.stack([person] * 4 + [neighbour] * 8)
    both = torch.stack([person, neighbour])
    assert not S.doubted(swapped, {f: both for f in (2, 5, 8, 10)}).any(), "the watch claimed to see a swap onto a detected neighbour; it cannot"
    return "doubted on frames 3 to 7 (back to the look before); a regained track starts clean; KNOWN LIMIT: a swap onto a detected neighbour is not seen"


def trusted_is_both():
    track, person = _clip_with_a_slide()
    looks = {f: person[None] for f in (2, 5, 8, 10)}
    got = S.trusted(track, looks).tolist()
    assert got == [True] * 3 + [False] * 7 + [True] * 2, f"trusted frames: {[i for i, t in enumerate(got) if t]}"
    # frame 3 is still on the person: only the watch's reach back to the look before distrusts it
    assert bool(S.plausible(track[3])) and bool(S.doubted(track, looks)[3]), "frame 3 should be plausible and doubted"
    # with no look at all the strip frames fail on shape alone
    assert S.trusted(track, {}).tolist() == [True] * 4 + [False] * 6 + [True] * 2, "with no looks, trust should be the shape test"
    return "trusted on frames 0 to 2 and 10, 11; frame 3 is doubted though plausible, 4 to 7 fail on shape, 8 and 9 are empty"


def _figure(left: int, right: int | None = None) -> torch.Tensor:
    """A figure on a 64 x 64 frame: rows 16 to 47, columns `left` to `right` (twelve wide when `right` is not given)."""
    m = _blank(64, 64)
    m[16:48, left:(left + 12 if right is None else right)] = True
    return m


HERE_, THERE_ = 10, 40      # the left columns of two figures standing apart, sharing no pixel


def a_held_figure_is_one_run():
    track = torch.stack([_figure(10 + f) for f in range(24)])        # one column a frame: a figure walking across
    got = S.unbroken(track, seed=12)
    assert (got.first, got.end, got.before, got.after) == (0, 24, S.STOPPED_END, S.STOPPED_END), got
    steps = [v for f, v in enumerate(got.overlap) if f != 12 and v is not None]
    assert got.overlap[12] is None and len(steps) == 23 and all(v > 0.8 for v in steps), got.overlap
    assert all(v == 1 for f, v in enumerate(got.frames_apart) if f != 12), got.frames_apart
    assert all(v is not None and abs(v - 1.0) < 1e-6 for f, v in enumerate(got.area_ratio) if f != 12), got.area_ratio
    assert S.unbroken(track.to(torch.float32), seed=12).overlap == got.overlap, "float masks are judged differently from bool ones"
    nobody = torch.stack([_figure(10)] * 3 + [_blank(64, 64)] + [_figure(10)] * 3)
    empty = S.unbroken(nobody, seed=3)
    assert (empty.first, empty.end, empty.before, empty.after) == (3, 3, S.STOPPED_EMPTY, S.STOPPED_EMPTY), empty
    for bad in (-1, 7):
        try:
            S.unbroken(nobody, seed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a seed on frame {bad} of a 7-frame track was accepted")
    return f"24 frames, seeded on 12: one run, the least step {min(steps):.2f}; an empty seed frame gives an empty run"


def _clip_with_a_jump() -> torch.Tensor:
    """Twenty frames: the mask is on one figure for frames 0 to 9 and on another, sharing no pixel, for 10 to 19. No empty frame."""
    return torch.stack([_figure(HERE_)] * 10 + [_figure(THERE_)] * 10)


def a_jump_is_cut():
    track = _clip_with_a_jump()
    got = S.unbroken(track, seed=2)
    assert (got.first, got.end, got.after) == (0, 10, S.STOPPED_MOVED), got
    assert got.overlap[10] == 0.0 and (got.centre_step[10] or 0.0) > 0.5 and got.frames_apart[10] == 1, (got.overlap[10], got.centre_step[10])
    assert all(v == 1.0 for v in got.overlap[11:]), "past the jump each frame is steady on the other figure, and still outside the run"
    back = S.unbroken(track, seed=15)                    # the same track seeded on the other figure: tracked backward into the jump
    assert (back.first, back.end, back.before, back.after) == (10, 20, S.STOPPED_MOVED, S.STOPPED_END), back
    # THE CONTROLS: what was counted until 2026-10-07 calls this track held on every frame, and so does the line at zero
    assert int(track.flatten(1).any(1).sum()) == 20 and bool(S.plausible(track).all()), "the control: non-empty and plausible on every frame"
    off = S.unbroken(track, seed=2, moved_off=0.0)
    assert (off.first, off.end) == (0, 20), f"with the line at zero the jump should pass, got {(off.first, off.end)}"
    return "cut at the jump from either side; a count of masks, the shape test and the line at zero all pass the whole track"


def a_return_after_a_gap_is_outside():
    gap = [_blank(64, 64)] * 3
    elsewhere = S.unbroken(torch.stack([_figure(HERE_)] * 6 + gap + [_figure(THERE_)] * 3), seed=0)
    same = S.unbroken(torch.stack([_figure(HERE_)] * 6 + gap + [_figure(HERE_)] * 3), seed=0)
    for got in (elsewhere, same):
        assert (got.first, got.end, got.after) == (0, 6, S.STOPPED_EMPTY), got
        assert got.overlap[6:9] == [None] * 3 and got.frames_apart[9] == 4, (got.overlap, got.frames_apart)
    assert elsewhere.overlap[9] == 0.0 and same.overlap[9] == 1.0, (elsewhere.overlap[9], same.overlap[9])
    return "the run ends at the first empty frame; the step across the gap is 0 for a return elsewhere and 1 for a return in place"


def _subject_track():
    """`subject_track.py`, the node's module, loaded as `check_subject_track.py` loads it; only for its `gallery_frames`."""
    import importlib.util
    import types
    sys.path.insert(0, str(REPO.parent.parent))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.subject_track", REPO / "subject_track.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.subject_track"] = module
    spec.loader.exec_module(module)
    return module


def a_gallery_is_taken_inside_the_run():
    node = _subject_track()
    track = _clip_with_a_jump().to(torch.float32)
    run = S.unbroken(track, seed=2)
    inside = [f + run.first for f in node.gallery_frames(track[run.first:run.end])]
    assert inside and all(run.first <= f < run.end for f in inside) and max(inside) < 10, inside
    # THE CONTROL: today's node hands `gallery_frames` the whole tracked piece
    whole = node.gallery_frames(track)
    assert any(f >= 10 for f in whole), f"the control: the whole track's gallery should hold frames from after the jump, got {whole}"
    return f"inside the run: frames {inside}; from the whole track, as today: {whole}, {sum(f >= 10 for f in whole)} of them after the jump"


def _clip_with_a_straddle() -> torch.Tensor:
    """Twenty frames, the shape of the jump seen on 2026-10-07: on one figure for frames 0 to 8; on frame 9 on that
    figure with a small piece on the next; on the next figure alone from frame 10. No empty frame."""
    straddle = _figure(HERE_)
    straddle[40:48, THERE_:THERE_ + 4] = True
    return torch.stack([_figure(HERE_)] * 9 + [straddle] + [_figure(THERE_)] * 10)


def the_frame_beside_a_jump_is_no_gallery_frame():
    track = _clip_with_a_straddle()
    run = S.unbroken(track, seed=2)
    assert (run.first, run.end, run.after) == (0, 10, S.STOPPED_MOVED), run
    assert (run.overlap[9] or 0.0) > 0.9, f"the straddle frame should share most of its pixels with the frame before, got {run.overlap[9]}"
    assert S.gallery_span(run) == (0, 9), S.gallery_span(run)
    # what a report can show of that frame, and what nothing is decided on
    assert (run.box_ratio[9] or 0.0) > 2 and (run.centre_step[9] or 0.0) > 0.3 and (run.area_ratio[9] or 9.0) < 1.2, (run.box_ratio[9], run.centre_step[9], run.area_ratio[9])
    assert all(abs((v or 0.0) - 1.0) < 1e-6 for v in run.box_ratio[1:9] if v is not None), run.box_ratio
    # tracked backward into the same jump: the run's first frame is the one left out
    flipped = torch.flip(track, dims=[0])
    back = S.unbroken(flipped, seed=17)
    assert (back.first, back.end, back.before) == (10, 20, S.STOPPED_MOVED) and S.gallery_span(back) == (11, 20), (back, S.gallery_span(back))
    # nothing is lost where the run did not stop on a move, and the seed is never left out
    whole = S.unbroken(torch.stack([_figure(HERE_)] * 6 + [_blank(64, 64)] * 3), seed=0)
    assert S.gallery_span(whole) == (0, 6) and whole.after == S.STOPPED_EMPTY, (S.gallery_span(whole), whole.after)
    alone = S.unbroken(torch.stack([_figure(HERE_)] + [_figure(THERE_)] * 3), seed=0)
    assert (alone.first, alone.end) == (0, 1) and S.gallery_span(alone) == (0, 1), "the seed frame was left out of its own gallery"
    # THE CONTROL: the node's own gallery, given the run as the cut leaves it, takes the straddle frame
    node = _subject_track()
    as_cut = [f + run.first for f in node.gallery_frames(track[run.first:run.end].to(torch.float32))]
    first, end = S.gallery_span(run)
    spanned = [f + first for f in node.gallery_frames(track[first:end].to(torch.float32))]
    assert 9 in as_cut, f"the control: the run as cut should put the straddle frame in the gallery, got {as_cut}"
    assert 9 not in spanned and spanned, spanned
    return f"the cut keeps frame 9 (overlap {run.overlap[9]:.2f}, box ratio {run.box_ratio[9]:.1f}); the gallery's span leaves it out; the node's gallery on the run as cut takes it: {as_cut}"


def a_jump_without_warning_costs_a_good_frame():
    track = _clip_with_a_jump()                          # frames 0 to 9 on one figure, 10 on another: no straddle frame
    run = S.unbroken(track, seed=2)
    assert (run.first, run.end, run.after) == (0, 10, S.STOPPED_MOVED), run
    # the frame beside the jump is ordinary: the same mask as the frame before, by every measure returned
    assert (run.overlap[9], run.area_ratio[9], run.box_ratio[9], run.centre_step[9]) == (1.0, 1.0, 1.0, 0.0), (run.overlap[9], run.area_ratio[9], run.box_ratio[9], run.centre_step[9])
    assert bool((track[9] == track[2]).all()), "frame 9 should be the subject's own mask"
    # THE KNOWN COST: it is left out of the gallery's span all the same, and nothing from after the jump comes in
    assert S.gallery_span(run) == (0, 9), S.gallery_span(run)
    return "KNOWN COST: a jump with no straddle frame; the frame before it is the subject's own mask and the gallery's span still leaves it out"


def a_creep_is_not_caught():
    # the mask's right edge grows from one figure over the next, three columns a frame, then its left edge follows
    grow = [_figure(HERE_, right) for right in range(HERE_ + 12, THERE_ + 13, 3)]
    shrink = [_figure(left, THERE_ + 12) for left in range(HERE_ + 3, THERE_ + 1, 3)]
    track = torch.stack(grow + shrink)
    assert not bool((track[0] & track[-1]).any()), "the creep should end on a figure sharing nothing with the seed's"
    got = S.unbroken(track, seed=0)
    steps = [v for v in got.overlap[1:] if v is not None]
    ratios = [v for v in got.area_ratio[1:] if v is not None]
    assert len(steps) == len(ratios) == int(track.shape[0]) - 1, "every frame after the seed should have a step"
    # THE KNOWN LIMIT: one run, every step far over the line; only the area says anything happened
    assert (got.first, got.end, got.after) == (0, int(track.shape[0]), S.STOPPED_END), "the step test claimed to see a creep; it cannot"
    assert min(steps) > 2 * S.MOVED_OFF, min(steps)
    grown = max(int(m.sum()) for m in track) / int(track[0].sum())
    assert grown > 3 and max(ratios) > 1.1, (grown, max(ratios))
    return (f"KNOWN LIMIT: {int(track.shape[0])} frames from one figure onto another, the least step {min(steps):.2f}, one run; "
            f"the mask grew to {grown:.1f} times its seed on the way")


def notes_say_a_place_in_the_clip():
    text = "frame 1310: person 2\n# a comment\n\n1:23.5: nobody\n83.5 s = take the best\nFrame 3055 : line 0.75\n0:02: leave it alone\n"
    notes = S.parse_notes(text, rate=24.0)
    got = [(round(n.seconds, 4), n.action, n.value) for n in notes]
    want = [(round(1310 / 24, 4), S.SUBJECT, 2.0), (83.5, S.NOBODY, None), (83.5, S.TAKE_BEST, None), (round(3055 / 24, 4), S.LINE, 0.75), (2.0, S.LEAVE_ALONE, None)]
    assert got == want, f"read as {got}"
    assert S.parse_notes("frame 2004: nobody", 24.0)[0].seconds == S.parse_notes("1:23.5: nobody", 24.0)[0].seconds, "frame 2004 at 24 a second is 1:23.5"
    for bad in ("fraem 12: person 2", "1:75: nobody", "frame 3: dance", "frame 3: person 0", "shot 2: person 1", "frame 3 person 2"):
        try:
            S.parse_notes("frame 1: nobody\n" + bad, 24.0)
        except ValueError as exc:
            assert bad.split(":")[0].split()[0] in str(exc) or bad in str(exc), f"`{bad}` was refused without quoting it: {exc}"
        else:
            raise AssertionError(f"`{bad}` was accepted")
    return f"{len(notes)} notes read; six kinds of typo refused with the line quoted"


def a_note_lands_in_its_shot():
    notes = S.parse_notes("frame 100: person 2\nframe 400: nobody\nframe 5000: take the best", rate=24.0)
    first_load = [(0, 120), (120, 345)]                 # the clip's first 345 frames
    second_load = [(0, 60), (60, 345)]                  # the next 345, starting at clip frame 345
    placed, outside = S.notes_for_shots(notes, first_load, first_seconds=0.0, rate=24.0)
    assert {k: [n.text for n in v] for k, v in placed.items()} == {0: ["frame 100: person 2"]}, f"first load: {placed}"
    assert [n.text for n in outside] == ["frame 400: nobody", "frame 5000: take the best"]
    placed, outside = S.notes_for_shots(notes, second_load, first_seconds=345 / 24.0, rate=24.0)
    assert {k: [n.text for n in v] for k, v in placed.items()} == {0: ["frame 400: nobody"]}, f"second load: {placed}"
    assert [n.text for n in outside] == ["frame 100: person 2", "frame 5000: take the best"]
    return "one text, two loads: each note lands in the load and shot that hold its frame, the rest are returned as outside"


def the_text_to_type_round_trips():
    shots = [(0, 37), (37, 38), (38, 200), (200, 345)]
    for rate in (24.0, 25.0):
        for first in (0.0, 14.375, 345 / rate, 100.5 / rate, 101.5 / rate):      # the last two start on a half frame
            for start, end in shots:
                for frame in (start, (start + end) // 2, end - 1):
                    text = S.place_text(frame, first, rate)
                    note = S.parse_notes(text.split(" (")[0] + ": nobody", rate)[0]
                    placed, outside = S.notes_for_shots([note], shots, first, rate)
                    assert not outside and list(placed) == [shots.index((start, end))], f"{text} at rate {rate}, start {first}: landed in {list(placed)}, outside {len(outside)}"
    sample = S.place_text(10, 54.2, 24.0)
    assert sample.startswith("frame ") and "(" in sample and ":" in sample, sample
    # a time that rounds up to the minute is written as the next minute, and what is written can be typed back
    for frame in (1439, 1440, 2879):
        text = S.place_text(frame, 0.0, 24.0)
        assert ":60" not in text, text
        S.parse_notes(text.split("(")[1].rstrip(")") + ": nobody", 24.0)
    return f"every shot's first, middle and last frame, at 24 and 25 a second and five starts, two on a half frame; e.g. `{sample}`"


def _box(cx: float, cy: float, w: float = 0.1, h: float = 0.3):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def taking_back_is_by_likeness_then_place():
    last = _box(0.5, 0.5)
    here, there, far = _box(0.52, 0.5), _box(0.2, 0.5), _box(0.9, 0.5)
    rule = dict(line=0.88, lead=0.03)
    # a clear leader is taken on likeness alone, wherever they stand
    assert S.take_back([0.95, 0.90], [far, here], last, **rule)[:2] == (0, S.TOOK_LEADER)
    # nobody over the line: nobody, however near
    assert S.take_back([0.87, 0.60], [here, far], last, **rule)[:2] == (None, S.NOBODY_OVER)
    # too close to call by likeness: the one where the subject was, whichever of the two scored higher
    assert S.take_back([0.92, 0.91], [there, here], last, **rule)[:2] == (1, S.TOOK_NEAREST)
    assert S.take_back([0.91, 0.92], [there, here], last, **rule)[:2] == (1, S.TOOK_NEAREST)
    # ... but not when both stand where the subject was, or the near one is another size, or nobody is near
    assert S.take_back([0.92, 0.91], [_box(0.47, 0.5), here], last, **rule)[:2] == (None, S.TOO_CLOSE), "two candidates equally near were told apart by place"
    assert S.take_back([0.92, 0.91], [there, _box(0.52, 0.5, 0.3, 0.9)], last, **rule)[:2] == (None, S.TOO_CLOSE), "a candidate three times the size was taken as the subject"
    assert S.take_back([0.92, 0.91], [there, far], last, **rule)[:2] == (None, S.TOO_CLOSE), "a candidate far from where the subject was was taken"
    # a third person who is near but under the line, or not close in likeness, does not enter the tie
    assert S.take_back([0.92, 0.91, 0.70], [there, here, _box(0.5, 0.5)], last, **rule)[:2] == (1, S.TOOK_NEAREST)
    # THE CONTROLS: with the place switched off, or no last place (across a cut), the same close call takes nobody,
    # which is what today's node does; and `take_best` takes the higher score
    assert S.take_back([0.92, 0.91], [there, here], last, by_position=False, **rule)[:2] == (None, S.TOO_CLOSE)
    assert S.take_back([0.92, 0.91], [there, here], None, **rule)[:2] == (None, S.TOO_CLOSE)
    assert S.take_back([0.92, 0.91], [there, here], last, take_best=True, **rule)[:2] == (0, S.TOOK_BEST)
    detail = S.take_back([0.92, 0.91], [there, here], last, **rule)[2]
    assert detail["lead"] == 0.01 and detail["distances_in_diagonals"][0] < 0.1 and len(detail["distances_in_diagonals"]) == 2, detail
    m = _blank(100, 200)
    m[20:60, 50:90] = True
    assert S.box_of(m) == (0.25, 0.2, 0.45, 0.6) and S.box_of(_blank(4, 4)) is None
    return "a clear leader by likeness; a close call by place, only when one candidate is near, like-sized and clearly nearer; nobody otherwise"


def a_clear_leader_elsewhere_is_refused():
    last = _box(0.36, 0.77, 0.11, 0.28)
    far, near = _box(0.88, 0.76, 0.13, 0.35), _box(0.34, 0.83, 0.13, 0.30)
    rule = dict(line=0.88, lead=0.03)
    # THE CONTROL: the highest likeness and the widest lead, half the frame away, is taken by likeness alone
    assert S.take_back([0.963, 0.874], [far, near], last, **rule)[:2] == (0, S.TOOK_LEADER)
    which, why, detail = S.take_back([0.963, 0.874], [far, near], last, leader_must_be_near=True, **rule)
    assert (which, why) == (None, S.LEADER_ELSEWHERE) and detail["leader_distance_in_diagonals"] > S.NEAR_DIAGONALS, (which, why, detail)
    # the same scores with the leader where the subject was: taken, and the distance is in the report
    which, why, detail = S.take_back([0.963, 0.874], [near, far], last, leader_must_be_near=True, **rule)
    assert (which, why) == (0, S.TOOK_LEADER) and detail["leader_distance_in_diagonals"] < S.NEAR_DIAGONALS, (which, why, detail)
    # a leader in the right place at three times the size is not the subject either
    assert S.take_back([0.963, 0.874], [_box(0.36, 0.77, 0.33, 0.84), far], last, leader_must_be_near=True, **rule)[:2] == (None, S.LEADER_ELSEWHERE)
    # no last place (the first look of a shot after a cut): likeness is all there is
    assert S.take_back([0.963, 0.874], [far, near], None, leader_must_be_near=True, **rule)[:2] == (0, S.TOOK_LEADER)
    # the option does not touch a close call or a refusal under the line
    assert S.take_back([0.92, 0.91], [far, near], last, leader_must_be_near=True, **rule)[:2] == (1, S.TOOK_NEAREST)
    assert S.take_back([0.87, 0.60], [near, far], last, leader_must_be_near=True, **rule)[:2] == (None, S.NOBODY_OVER)
    return "a clear leader far from the subject's last place, or of another size, is refused with its distance reported; without the option it is taken"


def a_slow_move_off_passes_the_default():
    # seven columns a frame on a figure twelve wide: each step shares 5 of 19 columns, a little over the default line
    track = torch.stack([_figure(HERE_)] * 4 + [_figure(HERE_ + 7 * k) for k in (1, 2, 3)] + [_blank(64, 64)] * 3)
    assert not bool((track[6] & track[0]).any()), "the last mask should share nothing with the seed's"
    got = S.unbroken(track, seed=0)
    steps = [v for v in got.overlap[4:7] if v is not None]
    assert len(steps) == 3 and all(S.MOVED_OFF < v < 0.5 for v in steps), steps
    # THE KNOWN LIMIT: one run up to the empty frame, and the span trims nothing because the run did not stop on a move
    assert (got.first, got.end, got.after) == (0, 7, S.STOPPED_EMPTY), "the default line claimed to see a slow move-off; it does not"
    assert S.gallery_span(got) == (0, 7), S.gallery_span(got)
    # a caller's higher line cuts it at its first step, and the span then leaves out the frame before
    higher = S.unbroken(track, seed=0, moved_off=0.5)
    assert (higher.first, higher.end, higher.after) == (0, 4, S.STOPPED_MOVED) and S.gallery_span(higher) == (0, 3), (higher.first, higher.end, S.gallery_span(higher))
    return f"KNOWN LIMIT: three steps of {steps[0]:.2f}, over the default {S.MOVED_OFF}, then empty: one run of 7 frames, all gallery material; a line of 0.5 cuts it at the first"


def taking_back_by_place_asks_where_first():
    last = _box(0.36, 0.77, 0.11, 0.28)
    here, far, far2 = _box(0.34, 0.80, 0.12, 0.30), _box(0.88, 0.76, 0.13, 0.35), _box(0.60, 0.76, 0.12, 0.30)
    rule = dict(asked=16, line=0.88, lead=0.03)
    # the candidate at the place is taken though a clear leader stands somewhere else
    which, why, detail = S.take_back_by_place([0.963, 0.90], [far, here], last, **rule)
    assert (which, why) == (1, S.TOOK_AT_THE_PLACE), (which, why, detail)
    took = detail["taken"]
    assert took["rank"] == 2 and took["lead_over_the_others"] < 0 and took["over_the_line"] and took["overlap_with_the_place"] > S.AT_THE_PLACE > took["next_overlap_with_the_place"], took
    assert (detail["likeness"], detail["asked"], detail["returned"], detail["cap_reached"]) == (S.OVER_THE_LINE, 16, 2, False), detail
    # THE CONTROL: likeness first takes the leader, half the frame away, which is the swap today's node made
    assert S.take_back([0.963, 0.90], [far, here], last, line=0.88, lead=0.03)[:2] == (0, S.TOOK_LEADER)
    # nobody stands there: nobody, however clear the leader
    assert S.take_back_by_place([0.963, 0.90], [far, far2], last, **rule)[:2] == (None, S.NOBODY_THERE)
    # two who may be asked stand there: place cannot say which
    assert S.take_back_by_place([0.92, 0.91], [here, _box(0.37, 0.78, 0.12, 0.29)], last, **rule)[:2] == (None, S.TWO_THERE)
    # ... but a second figure there who is under the line is not asked, and does not block the first
    assert S.take_back_by_place([0.92, 0.70], [here, _box(0.37, 0.78, 0.12, 0.29)], last, **rule)[:2] == (0, S.TOOK_AT_THE_PLACE)
    # the one who stands there is under the line: not taken, and the report says somebody stood there
    which, why, detail = S.take_back_by_place([0.95, 0.80], [far, here], last, **rule)
    assert (which, why) == (None, S.NOBODY_THERE) and detail["candidates_at_the_place"] == 1 and detail["may_be_asked"] == 1, (which, why, detail)
    # no last place (across a cut): nobody, and no box is read
    assert S.take_back_by_place([0.963, 0.90], [far, here], None, **rule)[:2] == (None, S.NO_LAST_PLACE)
    # FIRST BY A MARGIN: the line is not asked; the first by the margin is taken where they stand at the place
    first = dict(rule, likeness=S.FIRST_BY_A_MARGIN)
    which, why, detail = S.take_back_by_place([0.80, 0.70], [here, far], last, **first)
    assert (which, why) == (0, S.TOOK_AT_THE_PLACE) and detail["likeness"] == S.FIRST_BY_A_MARGIN and not detail["taken"]["over_the_line"], (which, why, detail)
    assert S.take_back_by_place([0.80, 0.70], [here, far], last, **rule)[:2] == (None, S.NOBODY_THERE), "over the line should refuse the same candidate"
    # ... first, but not by the margin; and first by the margin, but somewhere else
    assert S.take_back_by_place([0.80, 0.79], [here, far], last, **first)[:2] == (None, S.NOBODY_THERE)
    assert S.take_back_by_place([0.90, 0.963], [here, far], last, **first)[:2] == (None, S.NOBODY_THERE)
    # the cap: as many came back as were asked for, so the subject may not be among them
    full = S.take_back_by_place([0.5] * 16, [far] * 16, last, **rule)[2]
    assert full["cap_reached"] and full["returned"] == 16 and not S.take_back_by_place([0.5] * 9, [far] * 9, last, **rule)[2]["cap_reached"]
    assert S.take_back_by_place([], [], last, **rule)[:2] == (None, S.NOBODY_THERE)
    try:
        S.take_back_by_place([0.9], [here], last, likeness="the best", **rule)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown likeness rule was accepted")
    assert abs(S.box_overlap((0.0, 0.0, 0.2, 0.2), (0.1, 0.0, 0.3, 0.2)) - 1 / 3) < 1e-9 and S.box_overlap(None, last) == 0.0
    # THE KNOWN LIMIT, asserted so it is never read as covered: the subject has walked off and another figure stands
    # in their place. Place takes that figure, with every number in order.
    which, why, _ = S.take_back_by_place([0.90, 0.95], [here, far], last, **rule)
    assert (which, why) == (0, S.TOOK_AT_THE_PLACE), "place claimed to know who stands there; it cannot"
    return "by place first: the one who may be asked and stands where the subject was; nobody when none or two do; KNOWN LIMIT: whoever stands there is taken"


def the_one_in_the_way_is_named_by_the_box():
    subject = _blank(64, 64)
    subject[16:48, 16:44] = True
    subject[30:48, 24:36] = False                      # where somebody stands in front, the subject's mask has a gap
    front = _blank(64, 64)
    front[30:60, 24:36] = True
    beside = _blank(64, 64)
    beside[16:48, 43:55] = True                        # a neighbour: one column inside the subject's box
    far = _figure(2, 10)
    which, why, detail = S.in_the_way(subject, torch.stack([far, beside, front]))
    assert (which, why) == (2, S.THE_ONE_IN_THE_WAY) and detail["the_subject_among_them"] is None, (which, why, detail)
    assert detail["share_of_the_box"] > 0.2 and detail["next_share_of_the_box"] < S.IN_THE_BOX, detail
    # THE CONTROL: mask against mask, the person in front shares nothing with the subject
    assert detail["overlap_with_the_subjects_mask"] == 0.0 and not bool((front & subject).any()), detail
    # the detector also returned the subject: set aside, and the same person is named
    which, why, detail = S.in_the_way(subject, torch.stack([subject, front, beside]).to(torch.float32))
    assert (which, why) == (1, S.THE_ONE_IN_THE_WAY) and detail["the_subject_among_them"]["detection"] == 0, (which, why, detail)
    # only a neighbour at the edge, or nobody, or no subject: nobody is named
    which, why, detail = S.in_the_way(subject, torch.stack([far, beside]))
    assert (which, why) == (None, S.NOBODY_IN_THE_WAY) and 0 < detail["most_of_the_box"] < S.IN_THE_BOX, (which, why, detail)
    assert S.in_the_way(subject, torch.zeros((0, 64, 64), dtype=torch.bool))[:2] == (None, S.NOBODY_IN_THE_WAY)
    assert S.in_the_way(subject, subject[None])[:2] == (None, S.NOBODY_IN_THE_WAY), "the subject was named as standing in their own way"
    assert S.in_the_way(_blank(64, 64), front[None])[:2] == (None, S.NOBODY_IN_THE_WAY)
    return "named by the share of the subject's box a detection covers; the subject set aside; a neighbour at the edge is not named; mask against mask says nobody"


def stray_specks_go_and_the_subject_stays_whole():
    h, w = 256, 448
    body = _blank(h, w)
    body[60:200, 180:260] = True                       # the subject: 140 by 80
    arm = _blank(h, w)
    arm[70:110, 300:330] = True                        # a large piece of them past somebody's head: detached, kept
    sliver = _blank(h, w)
    sliver[120:124, 262:265] = True                    # a few pixels just off their outline: tiny, near, kept
    speck = _blank(h, w)
    speck[20, 430] = True                              # one pixel far away, on somebody else
    fleck = _blank(h, w)
    fleck[230:236, 20:28] = True                       # a larger stray, still tiny beside the subject and far
    clip = torch.stack([body, body | speck, body | arm | speck, body | sliver | fleck, _blank(h, w), speck]).to(torch.float32)
    out, frames, pixels = S.drop_specks(clip)
    # THE CONTROL: today's mask carries the speck, and what is downstream widens it into a block
    assert bool(clip[1, 20, 430]) and bool(clip[3, 232, 24]), "the case holds no speck; it tests nothing"
    assert frames == [1, 2, 3] and pixels == 1 + 1 + 48, (frames, pixels)
    assert torch.equal(out[0], clip[0]) and torch.equal(out[1], body.float()), "a lone subject changed, or the speck stayed"
    assert torch.equal(out[2], (body | arm).float()), "a large detached piece of the subject was dropped with the speck"
    assert torch.equal(out[3], (body | sliver).float()), "a small piece beside the subject was dropped, or a far fleck kept"
    assert not bool(out[4].any()) and torch.equal(out[5], clip[5]), "an empty frame changed, or a frame's only piece was removed"
    assert out.dtype == clip.dtype and not bool((out > clip).any()), "the mask gained pixels or changed type"
    # up and down as well as across: a speck straight above the subject's head, beyond the reach, goes
    above = _blank(h, w)
    above[2, 220] = True                               # 58 px above the body's top row, inside its columns
    got, hit, _n = S.drop_specks((body | above).float()[None])
    assert hit == [0] and not bool(got[0, 2, 220]), "a speck straight above the subject was kept: the gap is measured across only"
    # eight neighbours: a one-pixel diagonal tail off the subject's corner is part of them, however far it runs
    tail = _blank(h, w)
    for k in range(1, 51):                             # fifty pixels: its far end is beyond the reach
        tail[200 + k - 1, 260 + k - 1] = True          # from the corner pixel beyond (199, 259), down and to the right
    whole = (body | tail).float()[None]
    assert S.drop_specks(whole)[1] == [] and torch.equal(S.drop_specks(whole)[0], whole), \
        "a diagonal tail joined to the subject by a corner was split off and dropped: four neighbours were used"
    # the mask given is not written to, unless the caller says nothing else holds it
    before = clip.clone()
    S.drop_specks(clip)
    assert torch.equal(clip, before), "the input mask was written to"
    mine = clip.clone()
    same, frames_in_place, pixels_in_place = S.drop_specks(mine, in_place=True)
    assert same is mine and torch.equal(mine, out) and (frames_in_place, pixels_in_place) == (frames, pixels), "in place gave another answer"
    # each test alone keeps a piece: a tiny piece inside the reach, and a far piece that is not tiny
    near = S.drop_specks((body | sliver).float()[None])[1]
    big_far = _blank(h, w)
    big_far[10:40, 400:440] = True                     # 1,200 px, a tenth of the subject, far away
    assert near == [] and S.drop_specks((body | big_far).float()[None])[1] == [], "one test alone removed a piece"
    # the reach scales with the subject: the same speck at the same distance from a subject a tenth the size is far
    small = _blank(h, w)
    small[100:130, 200:216] = True
    close = _blank(h, w)
    close[100, 250] = True                             # 34 px from the small subject, 0 px inside the large one's reach
    assert S.drop_specks((small | close).float()[None])[1] == [0] and S.drop_specks((body | close).float()[None])[1] == []
    return "a far speck goes; a large detached piece, a small near piece, a lone piece and an empty frame stay; the reach follows the subject's size"


def main() -> int:
    for fn in (a_strip_is_not_a_subject, the_same_at_any_size, a_slid_track_is_doubted, trusted_is_both,
               a_held_figure_is_one_run, a_jump_is_cut, a_return_after_a_gap_is_outside, a_gallery_is_taken_inside_the_run,
               the_frame_beside_a_jump_is_no_gallery_frame, a_jump_without_warning_costs_a_good_frame, a_creep_is_not_caught, a_slow_move_off_passes_the_default, taking_back_is_by_likeness_then_place, a_clear_leader_elsewhere_is_refused, taking_back_by_place_asks_where_first, the_one_in_the_way_is_named_by_the_box, stray_specks_go_and_the_subject_stays_whole, notes_say_a_place_in_the_clip, a_note_lands_in_its_shot,
               the_text_to_type_round_trips):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
