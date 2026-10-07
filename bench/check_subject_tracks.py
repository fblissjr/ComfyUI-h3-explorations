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
  taking_back_is_by_likeness_then_place  after a loss, a clear leader is taken on likeness wherever they stand and
                                  nobody under the line is; a call too close by likeness goes to the candidate where the
                                  subject last was, only if that one is near, like-sized and clearly nearer than the
                                  others. THE CONTROLS: with the place switched off, or with no last place, the same
                                  close call takes nobody, as today's node does.
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
        for first in (0.0, 14.375, 345 / rate):
            for start, end in shots:
                for frame in (start, (start + end) // 2, end - 1):
                    text = S.place_text(frame, first, rate)
                    note = S.parse_notes(text.split(" (")[0] + ": nobody", rate)[0]
                    placed, outside = S.notes_for_shots([note], shots, first, rate)
                    assert not outside and list(placed) == [shots.index((start, end))], f"{text} at rate {rate}, start {first}: landed in {list(placed)}, outside {len(outside)}"
    sample = S.place_text(10, 54.2, 24.0)
    assert sample.startswith("frame ") and "(" in sample and ":" in sample, sample
    return f"every shot's first, middle and last frame, at 24 and 25 a second and three starts; e.g. `{sample}`"


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


def main() -> int:
    for fn in (a_strip_is_not_a_subject, the_same_at_any_size, a_slid_track_is_doubted, trusted_is_both,
               taking_back_is_by_likeness_then_place, notes_say_a_place_in_the_clip, a_note_lands_in_its_shot,
               the_text_to_type_round_trips):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
