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
overlap so a report can show one, and nothing is cut on them. `gallery_span` is the part of a run a gallery may be
taken from: the run less the frame beside a jump, which was seen to lie on both figures at once.

**One detection a thing** (`one_each`). The detector can return one person several times, as masks nested in each
other: on 2026-10-10 a small, half-hidden person on one frame came back as three detections at one place, were numbered
as three people, and their three labels were drawn on top of each other
(`bench/results/2026-10-10_who_is_who_across_shots.md`). Detections are joined when most of the smaller mask lies inside
the larger, and the one the detector scored highest stands for them. WHAT IT CANNOT TELL: one person wholly in front of
another whose mask was drawn through them. That would be joined too, so every join is counted and reported.

**Taking a subject back after a loss** (`take_back`). On hard crowd footage a change of the input too small to see
moved a regain's lead over the next person by about the lead the Subject Track requires
(`bench/results/2026-10-07_subject_track_under_nudge.md`), so likeness alone cannot settle the closest cases. Where the
subject was when last trusted can: among candidates too close to call, the one nearest that place, of a like size and
clearly nearer than the others, is taken; otherwise nobody is. Every number it used goes back for the report.
`take_back_by_place` turns the order round, after the same day showed a clear leader by likeness standing somewhere
else: a candidate has to stand where the subject last was, likeness only says who may be asked (over a line, or first by
a margin), and the result says whether the detector returned as many as it was asked for.

**Places in the clip** (`parse_notes`, `place_text`). A correction or a per-shot line names a place in the CLIP, as a
time or a frame, not a shot number inside one load of frames, so one text serves every load of a long clip. Nobody
computes an address: `place_text` writes the exact words to type for a shot.

**Stray specks** (`drop_specks`, 2026-10-08). A tracked mask can carry a few pixels far from its subject on a frame:
one to a few hundred, on somebody else's jacket. Everything downstream widens what it is given (the part node shows the
model the mask widened, the Masked Source grows the part by its margin and rounds it out to whole tokens for a latent
step's frames), so a speck of a few pixels became a block of the crowd regenerated for several frames, nowhere near the
subject. Per frame the largest connected piece is the subject; another piece is dropped only when it is BOTH tiny beside
that piece and away from it. Either alone keeps it, because a subject is often in several pieces: an arm seen past the
head of the person in front, a hand between two people. A hand at arm's length is kept by its size however far it is; a
sliver of sleeve is kept by being near however small.

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


#: Two detections are one thing when at least this share of the smaller mask lies inside the larger. Reasoned, with one
#: frame of one clip seen (2026-10-10, three nested masks of one person, each wholly inside the next): two people side
#: by side share an edge, not most of the smaller one's area.
SAME_THING = 0.8


def one_each(masks: torch.Tensor, scores: list[float], same: float = SAME_THING) -> tuple[list[int], dict[int, list[int]]]:
    """Which of a frame's detections stand for a thing each, once nested ones are joined.

    `masks` is [N, H, W] and `scores` the detector's score of each. Returns (the detections kept, in their own order,
    {a kept detection: every detection it stands for, itself first}).

    A THING IS A LARGER DETECTION AND WHATEVER IS NESTED IN IT. Taken largest first, a detection either lies `same`
    inside one already taken, and then belongs to the thing of the one that holds most of it, or it starts a thing
    of its own. So a whole person, their head and their torso are one thing though the head lies outside the torso
    (both lie in the whole); and two people whose masks share a strip stay two when something small lies in that
    strip, a hand across somebody or a held thing, which goes to whoever holds more of it. Two rules were tried and
    lost before this one, both found by mrcorn's cold reads on 2026-10-10: choosing as it went in score order split
    the whole, head and torso into two people and lost the whole; joining everything linked through any other
    detection made one thing of two people with a hand between them, and of three people in a row.

    Of a thing the highest scored detection is kept, the larger on a tie. That can be a PART (a head scored above
    the whole): `kept_share` says how much of the thing's largest mask the kept one is, and the tracker reports it,
    since the kept mask is what is sized, compared and seeded.
    """
    on = [(m > 0.5) for m in masks]
    area = [int(o.sum()) for o in on]
    score = [float(scores[i]) if i < len(scores) else 0.0 for i in range(len(on))]
    thing: dict[int, int] = {}                       # detection: the detection that started its thing
    for i in sorted(range(len(on)), key=lambda k: (-area[k], -score[k], k)):
        inside = [(int((on[i] & on[k]).sum()), k) for k in thing] if area[i] else []
        held = [(shared, -area[k], -k) for shared, k in inside if shared >= same * area[i]]
        thing[i] = thing[-max(held)[2]] if held else i
    members: dict[int, list[int]] = {}
    for i, start in thing.items():
        members.setdefault(start, []).append(i)
    stands = {}
    for those in members.values():
        best = min(those, key=lambda k: (-score[k], -area[k], k))
        stands[best] = [best] + sorted((k for k in those if k != best), key=lambda k: (-score[k], -area[k], k))
    return sorted(stands), stands


def kept_share(masks: torch.Tensor, stands: dict[int, list[int]]) -> dict[int, float]:
    """For each kept detection that stands for others: its area as a share of the largest mask it stands for."""
    out = {}
    for kept, those in stands.items():
        if len(those) > 1:
            areas = [int((masks[i] > 0.5).sum()) for i in those]
            out[kept] = areas[0] / max(max(areas), 1)
    return out


#: A piece of a tracked mask is a speck only under this share of the frame's largest piece. Measured on one load, the
#: tracker's own mask as a no-sampling preview wrote it (2026-10-08; the masking board, finding `mhi-08`, has the
#: figures): every stray piece was under it and every detached piece that was the subject and lay off the largest
#: piece's box was several times over it.
SPECK_SHARE = 0.005
#: ...and only when its box lies farther than this from the largest piece's box, as a share of the square root of the
#: largest piece's area, so it scales with the subject. Measured on the same load: every stray lay beyond it; the small
#: pieces that were the subject lay inside the largest piece's own box.
SPECK_REACH = 0.4


def drop_specks(mask: torch.Tensor, share: float = SPECK_SHARE, reach: float = SPECK_REACH, in_place: bool = False):
    """A [N, H, W] tracked mask without its stray specks: (the mask, the frames changed, the pixels removed in all).

    Per frame the largest connected piece (eight neighbours) is taken as the subject. Another piece is removed when its
    area is under `share` of that piece's AND its box is more than `reach` x the square root of that piece's area from
    that piece's box on either axis. Nothing else changes: a frame of one piece, an empty frame and every piece that is
    large or near come back as they were. The largest piece is never removed, so a frame that holds only a speck keeps
    it: whether the subject is there at all is not this function's question.

    The mask given is not written to unless `in_place`, which a caller sets when nothing else holds the tensor: a
    clip's mask at the canvas is gigabytes, and a copy of it on every run buys nothing there. It is read a frame at a
    time for the same reason.
    """
    import numpy as np                 # here: the two uses in this module
    from scipy import ndimage          # core's own dependency; imported here so the rest of the module needs none
    out = mask if in_place else mask.clone()
    eight = np.ones((3, 3), dtype=bool)
    frames, pixels = [], 0
    for f in range(int(mask.shape[0])):
        on = (mask[f] > 0.5).cpu().numpy()
        rows = np.flatnonzero(on.any(axis=1))
        if not rows.size:
            continue
        cols = np.flatnonzero(on.any(axis=0))
        y0, x0 = int(rows[0]), int(cols[0])
        labels, count = ndimage.label(on[y0:int(rows[-1]) + 1, x0:int(cols[-1]) + 1], structure=eight)
        if count < 2:
            continue
        areas = np.bincount(labels.ravel())[1:]
        big = int(areas.argmax())
        boxes = ndimage.find_objects(labels)
        by, bx = boxes[big]
        far = float(reach) * math.sqrt(float(areas[big]))
        gone = []
        for i in range(count):
            if i == big or areas[i] >= float(share) * areas[big]:
                continue
            sy, sx = boxes[i]
            gap = max(by.start - sy.stop, sy.start - by.stop, bx.start - sx.stop, sx.start - bx.stop, 0)
            if gap > far:
                gone.append(i + 1)
        if gone:
            kill = np.isin(labels, gone)
            out[f, y0:y0 + kill.shape[0], x0:x0 + kill.shape[1]][torch.from_numpy(kill)] = 0
            frames.append(f)
            pixels += int(kill.sum())
    return out, frames, pixels


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
    seed: int                        # the frame the call was seeded on
    first: int                       # the run's first frame
    end: int                         # one past its last; equal to `first` when the seed frame has no mask
    before: str                      # why the run starts where it does: one of the STOPPED_ reasons
    after: str                       # why it ends where it does
    overlap: list[float | None]      # intersection over union with that mask; the run is cut on this alone
    area_ratio: list[float | None]   # this mask's area over that mask's: above 1 it grew
    box_ratio: list[float | None]    # this mask's box area over that mask's box area: a far piece moves this, not the area
    centre_step: list[float | None]  # how far the box centre moved, in diagonals of that mask's box
    frames_apart: list[int | None]   # how many frames away that mask is: 1 unless empty frames lie between


def unbroken(track: torch.Tensor, seed: int, moved_off: float = MOVED_OFF) -> Unbroken:
    """[F, H, W] masks of ONE tracked call and the frame it was seeded on, to the run around the seed that stayed on one figure.

    Walking away from the seed in each direction, the run stops before the first frame that is empty or whose mask
    shares less than `moved_off` with the last mask before it. Frames past a stop are not in the run whatever they
    hold: a mask that comes back after a gap is a re-find's to judge, and its step across the gap is here for that
    (`overlap`, with `frames_apart` saying how old the mask it was compared with is).

    A TEST OF CONTINUITY, NOT OF IDENTITY: see the module's note on what it cannot see. `area_ratio`, `box_ratio` and
    `centre_step` are returned for the report and cut nothing. Frames for a gallery come from `gallery_span`, not from
    the run as it stands.

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
    box_ratio: list[float | None] = [None] * n
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
                box, was = box_of(t[f]), box_of(t[last])
                if box is not None and was is not None:      # both frames have a mask, so both have a box
                    box_ratio[f] = ((box[2] - box[0]) * (box[3] - box[1])) / max((was[2] - was[0]) * (was[3] - was[1]), 1e-9)
                    centre_step[f] = _away(box, was)[0]
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
        return Unbroken(seed, seed, seed, STOPPED_EMPTY, STOPPED_EMPTY, overlap, area_ratio, box_ratio, centre_step, frames_apart)
    return Unbroken(seed, 0 if early is None else early + 1, n if late is None else late, before, after,
                    overlap, area_ratio, box_ratio, centre_step, frames_apart)


def gallery_span(run: Unbroken) -> tuple[int, int]:
    """The frames of an unbroken run a gallery may be taken from, as (first, one past the last).

    The run itself, less its outermost frame on a side where it stopped because the mask moved off. Reasoned from one
    case, with no number in it: on 2026-10-07 the frame before a jump already lay on two figures at once, the one the
    track was on and a small piece on the one it jumped to, while still sharing most of its pixels with the frame before
    (`bench/results/2026-10-07_subject_regain_looks.md` and the per-frame track beside it; one stretch, as fed and with
    the input moved one level, a figure the owner did not mean). `unbroken` keeps that frame, since by overlap it is
    still the subject's mask; a gallery must not, because the Subject Track's `gallery_frames` takes the frame with the
    largest mask by rule and that frame is the likeliest to be it. The seed frame is never left out: its mask was
    picked, not tracked. A side that stopped on an empty frame or the track's end loses nothing. A second jump on the
    same stretch came with no such frame before it (`bench/results/2026-10-07_subject_track_calls_on_masks.md`): there
    the trim costs one good frame and adds no bad one, and a trim of one frame is not shown to be enough for a mask
    that leaves over several.

    `box_ratio` and `centre_step` on the frame left out say whether it was such a frame; nothing is decided on them
    until there is more than one case.
    """
    first, end = run.first, run.end
    if run.after == STOPPED_MOVED and end - 1 > run.seed:
        end -= 1
    if run.before == STOPPED_MOVED and first < run.seed:
        first += 1
    return first, end


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


# ---- taking a subject back by where they last were

#: read off one stretch, not measured as a rule: a candidate is "where the subject last was" when its box overlaps the
#: subject's last box by this much, by intersection over union. In the by-place table of
#: `bench/results/2026-10-07_subject_regain_looks.md` the candidates over the likeness line fall in two groups with
#: nothing between, at the picked figure's place and elsewhere, and this sits in the gap. Four labelled takes, one
#: stretch, a figure who stays where they stand: the shape is supported and the number is not, so it is an argument.
AT_THE_PLACE = 0.3

OVER_THE_LINE, FIRST_BY_A_MARGIN = "over the line", "first, by a margin"
TOOK_AT_THE_PLACE, NOBODY_THERE, TWO_THERE, NO_LAST_PLACE = ("the one where the subject last was", "nobody where the subject last was",
                                                           "more than one where the subject last was", "no last place to judge by")


def box_overlap(a, b) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1); 0 when either is missing."""
    if a is None or b is None:
        return 0.0
    shared = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union > 0 else 0.0


def take_back_by_place(scores: list[float], boxes: list, last_box, asked: int, line: float, lead: float,
                       likeness: str = OVER_THE_LINE, at_the_place: float = AT_THE_PLACE) -> tuple[int | None, str, dict]:
    """Which candidate on a looked-at frame is the subject, after a loss inside a shot, judged by place before likeness.

    `take_back` asks likeness first and place only to break a tie. On a crowd of alike figures a clear leader by
    likeness was mostly a figure standing somewhere else, while by overlap with the subject's box the candidates over
    the line fell in two groups with nothing between (`bench/results/2026-10-07_subject_regain_looks.md`, read off
    recorded looks on one stretch; no labels beyond a strip looked at). So here a candidate has to stand where the
    subject last was, and likeness only says who may be asked. Place is not identity either: two figures who change
    places, or a subject who walks off while another steps in, are taken wrongly, and nothing here sees it.

    `scores` are the candidates' likenesses, `boxes` their boxes (`box_of`), `last_box` the subject's box on the last
    frame that is the subject's without doubt: the last frame of `gallery_span`, not the last frame of the track, which
    can be a frame beside a jump lying on two figures. `asked` is how many detections the caller asked the detector
    for; place only helps when the subject is among those returned, so the result says whether that cap was reached.

    Who may be asked, by `likeness`:
      `OVER_THE_LINE`      every candidate at or above `line`.
      `FIRST_BY_A_MARGIN`  the candidate with the highest score, if it leads the next by `lead`; `line` is not asked.
                           Against one signature of the subject a held track cleared a fixed line on under half its
                           looks and was first on nearly all (`bench/results/2026-10-07_subject_likeness_in_a_group.md`:
                           tracks compared, not detections; one pass).
    Of those, the one whose box overlaps `last_box` by `at_the_place` is taken. Nobody is taken when none does, when
    more than one does, or when there is no last place (across a cut, which is the match's to judge).

    Returns (the index or None, the reason, the numbers behind it): which likeness rule ran, the count asked and
    returned, how many candidates stand at the place at all, and for the one taken its score, its rank, its lead over
    the best of the others and whether it is over the line, so the rule that did not run can be read off the same look.
    """
    if likeness not in (OVER_THE_LINE, FIRST_BY_A_MARGIN):
        raise ValueError(f"take_back_by_place: likeness is `{OVER_THE_LINE}` or `{FIRST_BY_A_MARGIN}`, not {likeness!r}")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best = scores[order[0]] if order else -1.0
    nxt = scores[order[1]] if len(order) > 1 else -1.0
    detail = {"likeness": likeness, "line": line, "lead_required": lead, "at_the_place": at_the_place, "asked": int(asked),
              "returned": len(scores), "cap_reached": len(scores) >= int(asked),
              "best": round(float(best), 4), "next": round(float(nxt), 4), "lead": round(float(best - nxt), 4)}
    if last_box is None:
        return None, NO_LAST_PLACE, detail
    overlaps = [box_overlap(b, last_box) for b in boxes]
    there = [i for i in order if overlaps[i] >= at_the_place]
    detail["candidates_at_the_place"] = len(there)
    if likeness == OVER_THE_LINE:
        may = [i for i in order if scores[i] >= line]
    else:
        may = [order[0]] if order and best - nxt >= lead else []
    detail["may_be_asked"] = len(may)
    here = [i for i in may if overlaps[i] >= at_the_place]
    if len(here) != 1:
        return None, (TWO_THERE if here else NOBODY_THERE), detail
    i = here[0]
    others = max((scores[k] for k in order if k != i), default=-1.0)
    detail["taken"] = {"score": round(float(scores[i]), 4), "rank": order.index(i) + 1, "lead_over_the_others": round(float(scores[i] - others), 4),
                       "over_the_line": bool(scores[i] >= line), "overlap_with_the_place": round(overlaps[i], 3),
                       "next_overlap_with_the_place": round(max((overlaps[k] for k in order if k != i), default=0.0), 3)}
    return i, TOOK_AT_THE_PLACE, detail


# ---- who stands in the subject's way

#: reasoned, not measured: a detection counts as standing in the subject's box when its mask covers at least this share
#: of that box. A person in front covers a good part of it; a neighbour's shoulder at its edge covers a sliver. No
#: stretch has been measured for it, so it is an argument.
IN_THE_BOX = 0.05

THE_ONE_IN_THE_WAY, NOBODY_IN_THE_WAY = "the detection most inside the subject's box", "nobody else inside the subject's box"


def in_the_way(subject: torch.Tensor, detections: torch.Tensor, in_the_box: float = IN_THE_BOX,
               same_at: float = AGREE_AT) -> tuple[int | None, str, dict]:
    """Which detection on a looked-at frame stands most in the subject's way without being the subject.

    For keeping a person who stands in front of the subject out of the region that is regenerated: they are followed
    in a tracker call of their own and their mask is kept (the Masked Source's `keep`). This only names who.

    `subject` is the subject's [H, W] mask on the frame, `detections` the [N, H, W] masks a fresh detect returned
    there. The detection that is the subject (the one overlapping the subject's mask most, if by `same_at`) is set
    aside. Of the rest, the one whose mask covers the largest share of the subject's BOX is named, if it covers
    `in_the_box` of it. The box and not the mask, because the detector's masks of two people hardly overlap even
    where one stands in front of the other (`bench/results/2026-10-07_subject_track_calls_on_masks.md`, the fresh
    detect at each look): by mask against mask the person in front scores as nobody.

    It cannot say who is in front and who behind, only who shares the subject's box; and with as many detections
    returned as were asked for, the one in the way may not be among them. The choice is a first guess for a shot,
    to be shown and corrected like the pick.

    Returns (the index or None, the reason, the numbers behind it).
    """
    m = subject if subject.dtype == torch.bool else subject > 0.5
    d = detections if detections.dtype == torch.bool else detections > 0.5
    box = box_of(m)
    detail: dict = {"in_the_box": in_the_box, "detections": int(d.shape[0]), "the_subject_among_them": None}
    if box is None or d.shape[0] == 0:
        return None, NOBODY_IN_THE_WAY, detail
    h, w = int(m.shape[0]), int(m.shape[1])
    x0, y0, x1, y1 = round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)
    shared = (d & m).flatten(1).sum(1).float()
    on_the_subject = (shared / (d | m).flatten(1).sum(1).float().clamp(min=1)).tolist()
    me = max(range(len(on_the_subject)), key=lambda i: on_the_subject[i])
    if on_the_subject[me] >= same_at:
        detail["the_subject_among_them"] = {"detection": me, "overlap": round(on_the_subject[me], 3)}
    else:
        me = None
    covers = (d[:, y0:y1, x0:x1].flatten(1).sum(1).float() / max((y1 - y0) * (x1 - x0), 1)).tolist()
    order = sorted((i for i in range(len(covers)) if i != me), key=lambda i: -covers[i])
    if not order or covers[order[0]] < in_the_box:
        detail["most_of_the_box"] = round(covers[order[0]], 3) if order else None
        return None, NOBODY_IN_THE_WAY, detail
    i = order[0]
    detail.update({"share_of_the_box": round(covers[i], 3), "overlap_with_the_subjects_mask": round(on_the_subject[i], 3),
                   "next_share_of_the_box": (round(covers[order[1]], 3) if len(order) > 1 else None)})
    return i, THE_ONE_IN_THE_WAY, detail


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
