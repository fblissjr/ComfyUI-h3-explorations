"""What can be said about a tracked subject without a model: is a mask a subject at all, does anything still agree with
a track, and where in the clip does a correction point.

The SAM 3.1 tracker node (`sam31_track.py`, not built yet) follows subjects through a shot and has to say when a track
cannot be trusted. The model-free pieces of that live here, so they can be checked on made-up masks and reused by the
bench tools that accept or reject the tracker:

**The shape test** (`plausible`). On 2026-10-07 an independent run found a tracked mask that slid off its person onto a
thin strip along the bottom edge of the frame and stayed "on" for hundreds of frames, and others that shrank to a few
hundred pixels. A count of non-empty masks scores those as subjects held. A mask passes here only if it is not a
fragment, not mostly inside a band along the frame's border, and not a sparse scatter inside its own box. It is a test of
shape, not of identity: a mask on the wrong person passes.

**The watch** (`doubted`). The tracker never checks a track against anything; a track that slides onto nothing (a wall,
the frame's edge, a blur) stays non-empty. The detector's answer on a frame is an independent look. A track is doubted
when a look finds no detection overlapping its mask, and the doubt reaches back to the look before, since nobody knows
where between the two it went wrong. WHAT IT CANNOT SEE: a track that slid onto a NEIGHBOUR. The neighbour is detected,
so the detector agrees with the mask. Catching that takes a likeness test against the subject, which is the tracker
node's to make, not this function's.
`trusted` is the two together, per frame: what the report shows, and the only frames a motion reference may be built from.

**The step test** (`unbroken`). On 2026-10-07 a subject followed alone had its mask moved by the tracker onto another
figure half the frame away with no empty frame between; the shape test passed every frame, a re-find then took the figure
the track was on, and the gallery it was judged against already held both figures
(`bench/results/2026-10-07_subject_regain_looks.md`). So each frame's mask is compared with the last mask before it, the
track is cut at the first empty frame or the first step that shares too little, and only the part that reaches the seed
without a cut is the subject's: the only frames a gallery may be taken from. IT IS A TEST OF CONTINUITY, NOT OF IDENTITY.
It sees a jump. It does not see a creep, a mask that grows over two figures and then shrinks onto the other, because
every step of that shares most of its pixels with the one before; the area and the box centre are returned beside the
overlap so a report can show one, and nothing is cut on them.

**Taking a subject back after a loss** (`take_back`). On hard crowd footage a change of the input too small to see
moved a regain's lead over the next person by about the lead the Subject Track requires
(`bench/results/2026-10-07_subject_track_under_nudge.md`), so likeness alone cannot settle the closest cases. Where the
subject was when last trusted can: among candidates too close to call, the one nearest that place, of a like size and
clearly nearer than the others, is taken; otherwise nobody is. Every number it used goes back for the report.

**Places in the clip** (`parse_notes`, `place_text`). A correction or a per-shot line names a place in the CLIP, as a
time or a frame, not a shot number inside one load of frames, so one text serves every load of a long clip. Nobody
computes an address: `place_text` writes the exact words to type for a shot.

Nothing here is specific to people: a subject is whatever the tracker's phrase asked for.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

#: reasoned, as shares of the frame so they hold at any size: about 500 pixels of the model's 1008 x 1008 grid. The
#: fragments the independent run saw were a few hundred pixels (2026-10-07); a far subject's head is larger than this.
FRAGMENT_SHARE = 0.0005
#: reasoned: about 25 pixels of the model's 1008, the width of the strip the independent run saw along the bottom edge.
BORDER_SHARE_OF_SIDE = 0.025
#: reasoned: a subject cut by the frame's edge has most of its mask further in than the band; a strip along the edge has
#: nearly all of it inside.
IN_BORDER_MOST = 0.5
#: reasoned: a person fills a third to a half of their own box; a scatter of specks across the frame fills very little.
BOX_FILL_LEAST = 0.1
#: reasoned, not measured: the overlap (intersection over union) at which a detection counts as the same thing as a
#: track's mask. Tracker and detector masks of one person differ at the edges, so the line is well under 1. To be read
#: against the base rate `bench/` measures (how often a track nobody doubts has no detection behind it).
AGREE_AT = 0.3


def plausible(masks: torch.Tensor) -> torch.Tensor:
    """[..., H, W] masks (bool, or above 0.5) to [...] bool: the mask could be a subject by its shape."""
    m = masks if masks.dtype == torch.bool else masks > 0.5
    h, w = int(m.shape[-2]), int(m.shape[-1])
    area = m.flatten(-2).sum(-1)
    band = torch.zeros((h, w), dtype=torch.bool, device=m.device)
    by, bx = max(round(h * BORDER_SHARE_OF_SIDE), 1), max(round(w * BORDER_SHARE_OF_SIDE), 1)
    band[:by] = band[-by:] = True
    band[:, :bx] = band[:, -bx:] = True
    in_band = (m & band).flatten(-2).sum(-1)
    rows, cols = m.any(-1), m.any(-2)
    height = h - rows.float().argmax(-1) - rows.flip(-1).float().argmax(-1)
    width = w - cols.float().argmax(-1) - cols.flip(-1).float().argmax(-1)
    return (area >= FRAGMENT_SHARE * h * w) & (in_band <= IN_BORDER_MOST * area) & (area >= BOX_FILL_LEAST * height * width)


def best_overlap(mask: torch.Tensor, detections: torch.Tensor) -> float:
    """The highest intersection over union of one [H, W] mask with any of [N, H, W] detections; 0 with none."""
    m = mask if mask.dtype == torch.bool else mask > 0.5
    d = detections if detections.dtype == torch.bool else detections > 0.5
    if d.shape[0] == 0 or not bool(m.any()):
        return 0.0
    inter = (d & m).flatten(1).sum(1).float()
    union = (d | m).flatten(1).sum(1).float().clamp(min=1)
    return float((inter / union).max())


def doubted(track: torch.Tensor, looks: dict[int, torch.Tensor], agree_at: float = AGREE_AT) -> torch.Tensor:
    """[F, H, W] track and {frame: [N, H, W] detections on it} to [F] bool: the frames the detector does not vouch for.

    On a looked-at frame the track is doubted when it has a mask and no detection overlaps it by `agree_at`. The doubt
    covers every frame after the look before (the last time the detector was asked, whatever it said) up to the next
    look, or up to the first frame the track is empty, whichever comes first: an empty track claims nothing, and a
    track seeded again after a loss starts with no doubt against it. With no look before, the doubt starts at the
    disagreeing look itself. A detection of somebody else under the mask counts as agreement: see the module's note.
    """
    t = track if track.dtype == torch.bool else track > 0.5
    out = torch.zeros(t.shape[0], dtype=torch.bool)
    on = t.flatten(1).any(1).cpu()
    frames = sorted(f for f in looks if 0 <= f < t.shape[0])
    for i, f in enumerate(frames):
        if not bool(on[f]) or best_overlap(t[f], looks[f]) >= agree_at:
            continue
        start = frames[i - 1] + 1 if i else f
        gone = (~on[start:f]).nonzero()
        if gone.numel():                       # the track was empty between the two looks: the doubt starts after that
            start = start + int(gone[-1]) + 1
        until = frames[i + 1] if i + 1 < len(frames) else int(t.shape[0])
        empty = (~on[f:until]).nonzero()
        if empty.numel():
            until = f + int(empty[0])
        out[start:until] = True
    return out


def trusted(track: torch.Tensor, looks: dict[int, torch.Tensor], agree_at: float = AGREE_AT) -> torch.Tensor:
    """[F] bool: the frames where the track has a mask that passes the shape test and the detector does not contradict."""
    t = track if track.dtype == torch.bool else track > 0.5
    return plausible(t).cpu() & ~doubted(t, looks, agree_at)


# ---- has the mask stayed on one figure

#: reasoned, not measured: a mask that shares less than this with the last mask before it, by intersection over union,
#: has moved off the figure it was on. One clip at one frame rate stands behind it: the least step of a figure that was
#: held, on three windows at 24 frames a second, is in `bench/results/2026-10-07_subject_regain_looks.md` (the table of
#: tracks handed back) and sits well clear of it. A lower frame rate or a small fast figure moves a held mask further
#: in one step, so this is the default of `unbroken`'s `moved_off` and never a constant a caller cannot set.
MOVED_OFF = 0.2

STOPPED_EMPTY, STOPPED_MOVED, STOPPED_END = "an empty frame", "the mask moved off", "the end of the track"


@dataclass(frozen=True)
class Unbroken:
    """`unbroken`'s result: the part of one tracked call that reaches its seed without a cut, and every frame's step.

    A frame's step is taken against the last frame with a mask on the seed's side of it: the frame before for a frame
    after the seed, the frame after for a frame before it. Each list has one entry per frame, None where there is no
    step: on the seed, on an empty frame, and where no frame on the seed's side has a mask.
    """
    first: int                       # the run's first frame
    end: int                         # one past its last; equal to `first` when the seed frame has no mask
    before: str                      # why the run starts where it does: one of the STOPPED_ reasons
    after: str                       # why it ends where it does
    overlap: list[float | None]      # intersection over union with that mask; the run is cut on this alone
    area_ratio: list[float | None]   # this mask's area over that mask's: above 1 it grew
    centre_step: list[float | None]  # how far the box centre moved, in diagonals of that mask's box
    frames_apart: list[int | None]   # how many frames away that mask is: 1 unless empty frames lie between


def unbroken(track: torch.Tensor, seed: int, moved_off: float = MOVED_OFF) -> Unbroken:
    """[F, H, W] masks of ONE tracked call and the frame it was seeded on, to the run around the seed that stayed on one figure.

    Walking away from the seed in each direction, the run stops before the first frame that is empty or whose mask
    shares less than `moved_off` with the last mask before it. Frames past a stop are not in the run whatever they
    hold: a mask that comes back after a gap is a re-find's to judge, and its step across the gap is here for that
    (`overlap`, with `frames_apart` saying how old the mask it was compared with is).

    A TEST OF CONTINUITY, NOT OF IDENTITY: see the module's note on what it cannot see. `area_ratio` and `centre_step`
    are returned for the report and cut nothing.

    One tracked call, not a stitched piece. The Subject Track fills a run of empty frames backward as well as forward
    from the frame it takes somebody on (`subject_track.py`, the regain), so a finished piece has a seam where a later
    call's backward fill meets an earlier call's track, and a seam reads as a jump. Give each call's own frames and seed.
    """
    t = track if track.dtype == torch.bool else track > 0.5
    n = int(t.shape[0])
    if not 0 <= int(seed) < n:
        raise ValueError(f"unbroken: seed frame {seed} is outside the track's {n} frame(s)")
    seed = int(seed)
    flat = t.flatten(1)
    area = [int(a) for a in flat.sum(1)]
    overlap: list[float | None] = [None] * n
    area_ratio: list[float | None] = [None] * n
    centre_step: list[float | None] = [None] * n
    frames_apart: list[int | None] = [None] * n

    def walk(frames) -> tuple[int | None, str]:
        """The steps of `frames`, in order away from the seed; the first frame that stops the run, and why."""
        last = seed if area[seed] else None
        stop, why = None, STOPPED_END
        for f in frames:
            if area[f] and last is not None:
                shared = int((flat[f] & flat[last]).sum())
                overlap[f] = shared / max(area[f] + area[last] - shared, 1)
                area_ratio[f] = area[f] / area[last]
                centre_step[f] = _away(box_of(t[f]), box_of(t[last]))[0]
                frames_apart[f] = abs(f - last)
            if stop is None and not area[f]:
                stop, why = f, STOPPED_EMPTY
            elif stop is None and overlap[f] is not None and overlap[f] < moved_off:
                stop, why = f, STOPPED_MOVED
            if area[f]:
                last = f
        return stop, why

    late, after = walk(range(seed + 1, n))
    early, before = walk(range(seed - 1, -1, -1))
    if not area[seed]:
        return Unbroken(seed, seed, STOPPED_EMPTY, STOPPED_EMPTY, overlap, area_ratio, centre_step, frames_apart)
    return Unbroken(0 if early is None else early + 1, n if late is None else late, before, after,
                    overlap, area_ratio, centre_step, frames_apart)


# ---- taking a subject back after a loss

#: reasoned, not measured: a candidate is "where the subject was" when its box centre is within this many diagonals of
#: the subject's last trusted box. One diagonal is a person moving about their own size.
NEAR_DIAGONALS = 1.0
#: reasoned, not measured: and when its box area is within this factor of the subject's last trusted box.
SIZE_FACTOR = 2.0
#: reasoned, not measured: and when no other close candidate is nearer than this many times its distance.
CLEARLY_NEARER = 1.5
#: reasoned, not measured: two candidates both within this many diagonals of the place stand in the same place, and
#: place cannot tell them apart whatever the ratio of two small distances says. About half a person's width.
SAME_PLACE = 0.15

TOOK_LEADER, TOOK_BEST, TOOK_NEAREST, NOBODY_OVER, TOO_CLOSE = "a clear leader", "the best, whatever its lead", "the nearest of those too close to call", "nobody over the line", "too close to call"
LEADER_ELSEWHERE = "a clear leader, but not where the subject last was"


def box_of(mask: torch.Tensor) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) of an [H, W] mask in shares of the frame, or None when it is empty."""
    m = mask if mask.dtype == torch.bool else mask > 0.5
    if not bool(m.any()):
        return None
    rows, cols = m.any(1).nonzero(), m.any(0).nonzero()
    h, w = float(m.shape[0]), float(m.shape[1])
    return float(cols[0]) / w, float(rows[0]) / h, (float(cols[-1]) + 1) / w, (float(rows[-1]) + 1) / h


def _away(box, last) -> tuple[float, float]:
    """How far `box` is from `last` in diagonals of `last`, and the ratio of their areas (never under 1)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    lx, ly = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
    diagonal = max(((last[2] - last[0]) ** 2 + (last[3] - last[1]) ** 2) ** 0.5, 1e-6)
    area, was = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9), max((last[2] - last[0]) * (last[3] - last[1]), 1e-9)
    return (((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5) / diagonal, max(area / was, was / area)


def take_back(scores: list[float], boxes: list, last_box, line: float, lead: float,
              take_best: bool = False, by_position: bool = True, leader_must_be_near: bool = False) -> tuple[int | None, str, dict]:
    """Which candidate on a looked-at frame is the subject, after a loss inside a shot.

    `scores` are the candidates' likenesses to the subject, `boxes` their boxes (`box_of`), `last_box` the subject's
    box on the last frame its track was trusted (None when there is none, as across a cut). A candidate must reach
    `line`. The best is taken when it leads the next by `lead`, or whatever its lead with `take_best`. Otherwise the
    candidates within `lead` of the best are too close to call by likeness, and with `by_position` the one nearest
    `last_box` is taken if it is near, of a like size, and clearly nearer than the others; else nobody is.

    `leader_must_be_near` also asks a clear leader to be near `last_box` and of a like size when there is one. Measured
    2026-10-07 on a crowd of alike figures: of four clear leaders the Subject Track took after a loss, three stood
    somewhere else, and the one with the highest likeness and the widest lead was the farthest
    (`bench/results/2026-10-07_subject_regain_looks.md`). A clear leader by likeness does not by itself say who.

    Returns (the index or None, one of the reasons above, the numbers behind it for the report).
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best = scores[order[0]] if order else -1.0
    nxt = scores[order[1]] if len(order) > 1 else -1.0
    detail = {"best": round(float(best), 4), "next": round(float(nxt), 4), "lead": round(float(best - nxt), 4), "line": line, "lead_required": lead}
    if not order or best < line:
        return None, NOBODY_OVER, detail
    if best - nxt >= lead:
        if leader_must_be_near and last_box is not None and boxes[order[0]] is not None:
            d, ratio = _away(boxes[order[0]], last_box)
            detail["leader_distance_in_diagonals"], detail["leader_size_ratio"] = round(d, 3), round(ratio, 3)
            if d > NEAR_DIAGONALS or ratio > SIZE_FACTOR:
                return None, LEADER_ELSEWHERE, detail
        return order[0], TOOK_LEADER, detail
    if take_best:
        return order[0], TOOK_BEST, detail
    if not by_position or last_box is None:
        return None, TOO_CLOSE, detail
    close = [i for i in order if scores[i] >= line and best - scores[i] < lead and boxes[i] is not None]
    away = sorted((_away(boxes[i], last_box) + (i,) for i in close), key=lambda t: t[0])
    detail["distances_in_diagonals"] = [round(d, 3) for d, _, _ in away]
    detail["size_ratios"] = [round(r, 3) for _, r, _ in away]
    if not away:
        return None, TOO_CLOSE, detail
    d, ratio, i = away[0]
    if d <= NEAR_DIAGONALS and ratio <= SIZE_FACTOR and (len(away) == 1 or away[1][0] >= max(d * CLEARLY_NEARER, SAME_PLACE)):
        return i, TOOK_NEAREST, detail
    return None, TOO_CLOSE, detail


# ---- places in the clip

@dataclass(frozen=True)
class Note:
    """One line of a corrections text: a place in the clip, in seconds from its start, and what to do with the shot there."""
    seconds: float
    action: str            # one of ACTIONS
    value: float | None    # the subject's number for `subject`, the line for `line`; None otherwise
    text: str              # the line as typed, for the report


SUBJECT, NOBODY, TAKE_BEST, LEAVE_ALONE, LINE = "subject", "nobody", "take the best", "leave alone", "line"
ACTIONS = (SUBJECT, NOBODY, TAKE_BEST, LEAVE_ALONE, LINE)
_PLACE = r"(?:frame\s*(?P<frame>\d+)|(?P<min>\d+):(?P<sec>\d{1,2}(?:\.\d+)?)|(?P<only>\d+(?:\.\d+)?)\s*s)"
_NOTE = re.compile(rf"^{_PLACE}\s*[:=]\s*(?P<what>.+?)\s*$", re.IGNORECASE)
_WHAT = (
    (re.compile(r"^(?:person|subject)\s*(\d+)$", re.IGNORECASE), SUBJECT),
    (re.compile(r"^(?:nobody|none|no one)$", re.IGNORECASE), NOBODY),
    (re.compile(r"^take the best$", re.IGNORECASE), TAKE_BEST),
    (re.compile(r"^leave (?:it )?alone$", re.IGNORECASE), LEAVE_ALONE),
    (re.compile(r"^line\s*(\d*\.\d+|\d+)$", re.IGNORECASE), LINE),
)


def parse_notes(text: str, rate: float) -> list[Note]:
    """A corrections text to its notes. One per line; blank lines and lines starting with `#` are skipped.

        frame 1310: person 2        the shot holding that frame of the clip takes subject 2 of its preview
        1:23.5: nobody              that shot is left alone
        83.5 s: take the best       that shot takes the best candidate whatever `when unsure` says
        frame 3055: line 0.75       that shot is judged against its own match line

    A frame is a frame of the clip at `rate`, the rate the frames were loaded at. A line that is not understood raises
    with the line quoted: a typo must not silently leave a shot as it was.
    """
    notes = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _NOTE.match(line)
        if m is None:
            raise ValueError(f"corrections: could not read `{line}`. Write a place and what to do there, such as "
                             "`frame 1310: person 2`, `1:23.5: nobody`, `83.5 s: take the best` or `frame 3055: line 0.75`.")
        if m["frame"] is not None:
            seconds = int(m["frame"]) / float(rate)
        elif m["min"] is not None:
            if float(m["sec"]) >= 60:
                raise ValueError(f"corrections: `{line}` has {m['sec']} seconds after the colon; write minutes:seconds.")
            seconds = int(m["min"]) * 60 + float(m["sec"])
        else:
            seconds = float(m["only"])
        for pattern, action in _WHAT:
            w = pattern.match(m["what"])
            if w is not None:
                value = float(w.group(1)) if w.groups() else None
                if action == SUBJECT and value is not None and value < 1:
                    raise ValueError(f"corrections: `{line}` names subject {int(value)}; the preview numbers subjects from 1.")
                notes.append(Note(seconds, action, value, line))
                break
        else:
            raise ValueError(f"corrections: in `{line}`, `{m['what']}` is not something to do with a shot. Use `person N`, "
                             "`nobody`, `take the best`, `leave alone` or `line X`.")
    return notes


def _clip_frame(seconds: float, rate: float) -> int:
    """The clip frame a time names: the nearest, a half going up. One rule for both directions, so a load that starts on
    a half frame cannot put a typed place one frame off (`round` is half-to-even, and two separate rounds could part)."""
    return int(math.floor(float(seconds) * float(rate) + 0.5))


def notes_for_shots(notes: list[Note], shots: list[tuple[int, int]], first_seconds: float, rate: float) -> tuple[dict[int, list[Note]], list[Note]]:
    """Which notes land in which shot of this load, and which land in none.

    `shots` are (start, end) frame ranges within the load, end exclusive. `first_seconds` is where in the clip the load's
    first frame sits. Returns ({shot index from 0: its notes, in the order typed}, the notes outside this load).
    """
    placed: dict[int, list[Note]] = {}
    outside = []
    for note in notes:
        frame = _clip_frame(note.seconds, rate) - _clip_frame(first_seconds, rate)
        for i, (start, end) in enumerate(shots):
            if start <= frame < end:
                placed.setdefault(i, []).append(note)
                break
        else:
            outside.append(note)
    return placed, outside


def place_text(frame_in_load: int, first_seconds: float, rate: float) -> str:
    """The words that address a frame of this load in a corrections text, as a clip frame and a time: `frame 1310 (0:54.6)`.

    The frame form is the one to type; it round-trips exactly through `parse_notes`. The time is for a reader.
    """
    clip_frame = _clip_frame(first_seconds, rate) + int(frame_in_load)
    tenths = int(math.floor(clip_frame / float(rate) * 10 + 0.5))        # rounded once, before the minutes are split off: never `0:60.0`
    return f"frame {clip_frame} ({tenths // 600}:{tenths % 600 / 10:04.1f})"
