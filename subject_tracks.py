"""What can be said about a tracked subject without a model: is a mask a subject at all, does anything still agree with
a track, and where in the clip does a correction point.

The SAM 3.1 tracker node (`sam31_track.py`, not built yet) follows subjects through a shot and has to say when a track
cannot be trusted. Three model-free pieces of that live here, so they can be checked on made-up masks and reused by the
bench tools that accept or reject the tracker:

**The shape test** (`plausible`). On 2026-10-07 an independent run found a tracked mask that slid off its person onto a
thin strip along the bottom edge of the frame and stayed "on" for hundreds of frames, and others that shrank to a few
hundred pixels. A count of non-empty masks scores those as subjects held. A mask passes here only if it is not a
fragment, not mostly inside a band along the frame's border, and not a sparse scatter inside its own box. It is a test of
shape, not of identity: a mask on the wrong person passes.

**The watch** (`doubted`). The tracker never checks a track against anything; a track that slides onto a neighbour, or
onto nothing, stays non-empty. The detector's answer on a frame is an independent look. A track is doubted on a looked-at
frame when it has a mask there and no detection overlaps it, and a doubt lasts until a later look agrees again.
`trusted` is the two together, per frame: what the report shows, and the only frames a motion reference may be built from.

**Places in the clip** (`parse_notes`, `place_text`). A correction or a per-shot line names a place in the CLIP, as a
time or a frame, not a shot number inside one load of frames, so one text serves every load of a long clip. Nobody
computes an address: `place_text` writes the exact words to type for a shot.

Nothing here is specific to people: a subject is whatever the tracker's phrase asked for.
"""
from __future__ import annotations

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
    covers every frame from that look up to the next look, or up to the first frame the track is empty, whichever comes
    first: an empty track claims nothing, and a track seeded again after a loss starts with no doubt against it. Frames
    before the first look are not doubted.
    """
    t = track if track.dtype == torch.bool else track > 0.5
    out = torch.zeros(t.shape[0], dtype=torch.bool)
    on = t.flatten(1).any(1).cpu()
    frames = sorted(f for f in looks if 0 <= f < t.shape[0])
    for i, f in enumerate(frames):
        if not bool(on[f]) or best_overlap(t[f], looks[f]) >= agree_at:
            continue
        until = frames[i + 1] if i + 1 < len(frames) else int(t.shape[0])
        empty = (~on[f:until]).nonzero()
        if empty.numel():
            until = f + int(empty[0])
        out[f:until] = True
    return out


def trusted(track: torch.Tensor, looks: dict[int, torch.Tensor], agree_at: float = AGREE_AT) -> torch.Tensor:
    """[F] bool: the frames where the track has a mask that passes the shape test and the detector does not contradict."""
    t = track if track.dtype == torch.bool else track > 0.5
    return plausible(t).cpu() & ~doubted(t, looks, agree_at)


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


def notes_for_shots(notes: list[Note], shots: list[tuple[int, int]], first_seconds: float, rate: float) -> tuple[dict[int, list[Note]], list[Note]]:
    """Which notes land in which shot of this load, and which land in none.

    `shots` are (start, end) frame ranges within the load, end exclusive. `first_seconds` is where in the clip the load's
    first frame sits. Returns ({shot index from 0: its notes, in the order typed}, the notes outside this load).
    """
    placed: dict[int, list[Note]] = {}
    outside = []
    for note in notes:
        frame = int(round((note.seconds - float(first_seconds)) * float(rate)))
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
    clip_frame = int(round(float(first_seconds) * float(rate))) + int(frame_in_load)
    seconds = clip_frame / float(rate)
    return f"frame {clip_frame} ({int(seconds // 60)}:{seconds % 60:04.1f})"
