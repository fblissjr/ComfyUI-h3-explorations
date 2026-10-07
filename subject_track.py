"""One person, followed through a clip with cuts, as one mask per frame.

The owner's case (2026-10-04): a music video with several people, cutaways and
coloured lighting, in which one performer is to be replaced. Core's
`SAM3_VideoTrack` with a text prompt is not built for it. SAM 3's video data
was checked to have no scene cuts (its paper, the verification step of the
video annotation), so across a cut the tracker starts new objects: on the
owner's band clip the lead came out as three hand-picked object numbers. With
a text prompt it also has an object cap that detection stops at for good, a
keep-alive count that blanks a track the detector stops matching, and a
detector mask that overwrites the tracked one on a confident frame
(`comfy/ldm/sam3/tracker.py::track_video_with_detection`,
`_match_and_add_detections`). `docs/research/masking/2026-10-04_mrhf.md` has
the reading.

Upstream SAM 3 added a caller-mask request for this on 2026-09-18
(`coderef/sam3`, "Condition the video tracker on a caller mask"): tell the
tracker which object is meant, from a mask, at any frame. It is not served by
the multiplex model the 3.1 checkpoint is, and core's port does not have it.
This node does the same thing with what core has: it splits the clip into
shots, picks the subject once, finds the same person on each other shot, and
tracks every shot from that person's mask with no text prompt, which is
core's `initial_mask` path and has none of the three behaviours above.

**The steps** (`follow`):

1. Cuts (`cut_scores`, `find_cuts`, `auto_cuts`): one minus the correlation
   of consecutive frames' gradient maps at a small size. A cut changes where
   the edges are; a lighting change does not. ffmpeg's scene score failed on
   the band clip under its coloured gels; this score separated eight of its
   nine cuts from every other frame (the ninth is a jump cut inside a
   cutaway). The threshold is a value the user names, or automatic: the
   middle of the widest gap in the clip's own scores, since two clips already
   disagreed about one fixed value. The score is taken on the picture: bars
   at a frame's edges that stay one flat value through the clip (`flat_borders`)
   are left out, because their edges are the same in every frame and pull
   every cut's score down. On a 4:3 picture padded onto the wide canvas the
   same eleven cuts scored 0.86 and under where the bare picture's scored
   0.87 and over, and the automatic threshold found none
   (`bench/results/2026-10-06_bordered_source_cuts.md`).
2. People: each shot is looked at on one frame a little way in
   (`PROBE_OFFSET`; the first frame after a cut is where blur and a dissolve
   land). SAM 3 is asked for `subject_phrase` there. Core's detector returns
   one detection per phrase unless the phrase carries a count
   (`comfy/text_encoders/sam3_clip.py::_parse_prompts`, `person:8`), so the
   node asks for up to `max_people` (`counted`).
3. The pick (`choose`, `main_subject`): `pick` names a rule, the largest, the
   most central or the best match for the phrase. Named a frame, the node
   applies the rule there. Left automatic, it applies the rule on every
   shot's frame and takes the person the rule favours for the most frames of
   the clip: a lead is the largest person in the long shots, and a cutaway's
   largest person is the largest only there. A shot's favourite on whom no
   head is found has no say while another shot's has one. The detector takes
   things for people (step 7's microphone), and a thing alone in the longest
   shot would otherwise be the subject: on the first window of the one-person
   clip the opening shot is the longest and shows only the microphone where
   it is judged, and the microphone was picked and masked for the whole shot
   (`bench/results/2026-10-06_subject_track_defaults.md`). The report names
   a favourite left out this way.
4. Each other shot: its people are compared with the subject by the vision
   trunk's features pooled under the top third of each mask, the head and
   shoulders (`top_third`, `signature`). The other people on the pick frame
   are certainly not the subject, so their average signature is subtracted
   from every signature before comparing (`relative`, `similarity`). A shot
   whose best is at or above the cut is seeded there; one below it is probed
   every `PROBE_STRIDE` frames, so a subject who walks in late or turns round
   late is still found; a shot with nobody above it is left empty.
5. The cut between "the subject" and "somebody else" (`auto_match`): a value the
   user names, or automatic. Automatic takes everything at or above a floor,
   and moves the cut up to the middle of the widest gap in the shots' best
   scores when that gap lies above the floor: the widest gap is taken as the
   line between the subject and everybody else. The report prints the scores
   with the cut marked, so a wrong cut can be seen.
6. Two places, not one. SAM 3 is also asked for `head_phrase` on every frame
   that is looked at, and each person gets the head that lies inside their
   mask (`head_of`). A person is compared with the subject under the top
   third of the mask and under the head, and the lower of the two counts. A
   person on whom no head is found is no match. Measured on three clips
   (`docs/research/masking/2026-10-04_mrhf.md`): on a clip of young women in
   a car the top third alone took two other women as the lead, and the head
   alone took a wrong person on the band clip; the lower of the two took the
   right shots on both.
7. A clip with one person in it. When the pick frame shows nobody else and
   the match is automatic, there is nobody to mistake the subject for, and
   the similarity is the wrong judge: it fell to 0.73 for the same singer
   between a full-length shot and a close-up (mrblue's card run,
   2026-10-04). So a shot under the line is probed to its end, and its best
   frame showing exactly one person with a head is taken, whatever it scores.
   The head is what keeps a thing out: the detector marks a hanging
   microphone as a person in that clip's empty opening, and on every frame it
   was the one detection with no head. The report and the tile say the shot
   was taken this way. A cutaway to a different lone person would be taken
   too; naming a value for `match` turns the rule off.
8. Tracking: from the seed frame forward to the shot's end and backward to its
   start, each a tracker call with the seed mask as `initial_mask`.
9. A subject let go inside a shot is looked for again (`regain`). A tracker
   call is seeded once and never detects again, so when it lets the subject
   go, every later frame of the shot is empty however long the subject is
   back in view. Measured on one clip on the card, a single shot with no
   cut: the mask ended on frame 680 of 1,065 and never came back. So a run of frames the track leaves empty is treated
   is probed every `PROBE_STRIDE` frames from the side that is tracked, and
   each person a probed frame shows is compared with the subject AS THIS
   SHOT'S OWN TRACK SHOWED THEM: a gallery of up to `GALLERY_MOST`
   signatures taken under the tracked mask on frames spread over the part
   of the shot that was tracked (`gallery_of`). A person's score is their
   best plain similarity to the gallery, the lower of the two places a
   person is compared (step 6). The first frame whose best person scores at
   or above `REGAIN_SAME`, and at least `REGAIN_MARGIN` above the next
   person on that frame, seeds the track again, run both ways over the
   empty run. Nobody is taken on a rule about being alone. Measured on one
   clip on the card, 2026-10-06, with a phrase that returns one detection a
   frame: after the track let go, that one detection was a region at the
   frame's edge with no head on four of the eight frames looked at, two
   edge regions on one, nothing on one, and the subject on three; and on a
   later frame it was a person whose mask shares no pixel with the
   subject's track. What is still
   empty afterwards stays empty, and the report and the shot table name
   those frames and what every probe scored. Its limits, plainly: the
   gallery is what the first track held, so a first pick on the wrong
   person is not put right by this; a subject who comes back looking
   unlike every gallery frame (turned away, far smaller) is not found; and
   a subject who has really left costs a probe every `PROBE_STRIDE` frames
   to the shot's end. A shot corrected by hand is not searched.

**How far the matching can be trusted.** SAM 3's trunk is trained to say what
a thing is, not who, so people in the same clothes score close together. On
the owner's band clip (six people in one sweatshirt,
`docs/research/masking/2026-10-04_mrhf.md`): pooled under the whole mask a
neighbour scores as close to the pick as the lead in another shot does; under
the top third with the pick frame's other people subtracted, on the card, the
lead scored 0.92 and 0.94 on his other two shots and the best wrong person
0.71. Seen from behind he scores inside the others' range, so a shot in which
he never faces the camera is not found, and the report and the preview show
it as absent with its best value. One clip. Every value that decides the mask
is an input or is stated in the report.

**Keeping the mask** is not this node's job. `MiniMaxH3MaskedSource` keeps a
finished mask across runs (`mask_store.py`) and on a hit never asks for its
`mask` input, so core does not run this node at all. Two things here serve
that. `MASK_VERSION` on the node goes into the kept mask's key: bump it when
a change would give a different mask from the same inputs and settings (the
cut score, the pick, the signature, the automatic rules, how a shot is
seeded and tracked), and
not for a tooltip or the report's wording. And the preview and the report are
shown as the node's own UI, so nothing has to be wired to see them: a preview
or save node on either output would make core run the tracker on every queue,
whatever the Masked Source decided. The outputs exist; a shipped graph leaves
them unwired.

**The shot table** (`shot_table.py`) is the same per-shot state as a table a
person reviews once: per shot, who was found, numbered left to right as on
the preview's outlines, who was taken and why. It is the fourth output, and
its text is shown under the report. It reads `follow`'s result and changes
nothing about how anyone is followed, so it does not move `MASK_VERSION`.

**A correction** (`corrections`, `parse_corrections`) fixes one shot by hand
with two numbers: `shot 3: person 2` takes person 2 of shot 3, `shot 3: none`
leaves the shot alone. The shot number is the one the report and the tile
print; the person number is the one drawn on that shot's tile
(`shot_table.person_order`). It is applied on the frame the tile shows, from
the mask the node already holds there, so the numbers a person reads are the
numbers that apply, and nothing is detected again. The automatic pass still
decides which frame each tile shows, and does so the same way with or without
corrections; a corrected shot is tracked once, from the corrected seed. The
design is mrhf's (2026-10-05, on the masking board: number the outlines the
tile already draws, and a correction is two numbers typed off it), and the
numbering is mrteal's (`shot_table.py`). The owner's ask: the car clip's wrong
shots fixed without a frontend widget, and "the simpler elegant solution is
always the better one".

Every phrase SAM 3 is given is an input: `subject_phrase` and `head_phrase`.

Nothing here patches core. It calls core's own nodes (`SAM3_Detect`,
`SAM3_VideoTrack`, `SAM3_TrackToMask`) and the SAM 3 model's vision trunk.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

from . import shot_table

logger = logging.getLogger(__name__)

#: The size frames are reduced to for the cut score, width by height. Small on
#: purpose: a cut moves every edge, and grain or a gesture should not count.
#: Reasoned, not measured.
CUT_SIZE = (192, 108)
#: A row or column at a frame's edge is border when all of it, in every frame,
#: lies within this of one value (frames are 0 to 1). Reasoned, not measured:
#: wide enough for a coded black bar's noise after the reduction to
#: `CUT_SIZE`, far under the spread of any strip that holds a picture.
BORDER_FLAT = 4.0 / 255.0
#: The default of `cut_threshold`, used when `cuts` is `at a value`, and what
#: automatic falls back to when the clip's scores show no clear gap. Reasoned
#: from two clips, neither of which it suits as a fixed value: see `CUT_FLOOR`.
CUT_THRESHOLD = 0.9
#: Automatic cuts: only a gap whose upper side is at or above this is taken as
#: the line between cuts and everything else. Measured on two clips
#: (`docs/research/masking/2026-10-04_mrhf.md`): the band segment's cuts score
#: 0.99 and above and its other frames 0.76 and below; the one-person clip's
#: cuts 0.84 and above and its other frames 0.46 and below.
CUT_FLOOR = 0.8
#: Automatic cuts: the gap has to be at least this wide. Reasoned.
CUT_MIN_GAP = 0.1
#: How far into a shot the frame it is judged on lies. Reasoned: past the
#: frame a cut detector off by one, a dissolve or motion blur lands on.
PROBE_OFFSET = 4
#: Frames between probes of a shot whose first look found no match.
#: Reasoned: half a second at the pack's frame rate.
PROBE_STRIDE = 12
#: A run of frames a track leaves empty inside its shot is searched when it is
#: at least this long. Reasoned: the search looks every `PROBE_STRIDE` frames,
#: so a shorter run may hold no frame to look at.
REGAIN_MIN_RUN = PROBE_STRIDE
#: The most times one shot's track is seeded again. Reasoned: a bound on the
#: cost of a shot whose track keeps letting go; not measured.
REGAIN_MOST = 8
#: The most frames of a shot's own track kept as the gallery a person found
#: after a loss is compared with. Reasoned: enough to hold the turns and sizes
#: of one shot, at one trunk pass and one head detection each; not measured.
GALLERY_MOST = 8
#: A person is the subject at or above this plain similarity to a gallery of
#: the subject. Measured on one clip on the card, 2026-10-06, one shot with no
#: cut (`bench/results/2026-10-06_subject_track_regain_and_handover.md`): the
#: subject found again after the track let go scored 0.89 to 0.93 against the
#: shot's own gallery, at a quarter of the size the gallery's frames show, and
#: 0.95 to 0.97 where the gallery was close in time and size; another person
#: the detector returned for the phrase scored 0.86, and detections at the
#: frame's edge 0.70 to 0.74. The line sits between 0.86 and 0.89: three
#: hundredths on one clip, which is thin. At `PLAIN_SAME` (0.93), where this
#: started, the subject was refused after the loss.
REGAIN_SAME = 0.88
#: ...and at least this far above the next person on the same frame. FIRST
#: ROUGH PASS: reasoned, not measured.
REGAIN_MARGIN = 0.03
#: Automatic matching, similarity relative to the pick frame's other people:
#: nothing below this is the subject. Measured on one clip on the card, the
#: band segment with five others on the pick frame: the lead 0.92 and 0.94,
#: the best wrong person 0.71 (`docs/research/masking/2026-10-04_mrhf.md`).
MATCH_FLOOR = 0.8
#: The same floor on the plain similarity, used when the pick frame shows
#: nobody else. Measured on the same clip by a CPU probe: the lead facing the
#: camera 0.945 and above, everyone else 0.89 and below.
PLAIN_FLOOR = 0.91
#: Automatic matching: a gap in the shots' best scores at least this wide,
#: above the floor, moves the cut to its middle. Reasoned: wider than the
#: spread among true matches seen so far and narrower than the gap to the
#: wrong people.
MIN_GAP = 0.1
#: The default of `match_threshold`, used when `match` is `at a value`. The
#: middle of the gap measured on the band clip on the card.
MATCH_THRESHOLD = 0.82
#: Automatic pick: two shots' favoured people are the same person at or above
#: this plain similarity. Measured by the CPU probe above. It only decides a
#: vote, so being a little off costs little.
PLAIN_SAME = 0.93
#: The default of `detection_threshold`. Inherited: core's node default
#: (`comfy_extras/nodes_sam3.py::SAM3_Detect.define_schema`).
DETECTION_THRESHOLD = 0.5
#: The default of `subject_phrase`.
SUBJECT_PHRASE = "person"
#: The default of `head_phrase`: what SAM 3 is asked for to find each
#: person's head, the second place a match has to hold.
HEAD_PHRASE = "head"
#: The default of `max_people`: the most detections of the phrase taken from a
#: frame. Reasoned: more than a stage usually shows, and each one costs a mask
#: refinement only on the few frames that are probed.
MAX_PEOPLE = 16
#: Width of a preview tile in pixels. Reasoned: wide enough to read the label.
TILE_WIDTH = 768
#: The side SAM 3's vision trunk takes. Inherited: core's detect node scales
#: every frame to it (`comfy_extras/nodes_sam3.py::SAM3_Detect.execute`).
TRUNK_SIDE = 1008

PICK_LARGEST = "largest"
PICK_CENTRAL = "most central"
PICK_SCORE = "best match for the phrase"
PICKS = (PICK_LARGEST, PICK_CENTRAL, PICK_SCORE)

AUTOMATIC = "automatic"
PICK_ON_FRAME = "a frame I name"
AT_VALUE = "at a value"


def counted(phrase: str, most: int) -> str:
    """`phrase` in core's SAM 3 prompt syntax, each comma-separated part asking for up to `most` detections.

    A part that already carries its own `:N` keeps it. Core reads `name:N` as
    at most N detections of `name`, and a bare `name` as one.

    One phrase asking for one detection is written bare. Core's tokenizer
    takes a short cut for that case and encodes the text as typed
    (`comfy/text_encoders/sam3_clip.py::SAM3TokenizerWrapper.tokenize_with_weights`),
    so `person:1` would reach the text encoder with its `:1` in it.
    """
    parts = [p.strip() for p in str(phrase).split(",") if p.strip()]
    if not parts:
        raise ValueError("subject_phrase is empty: say what SAM 3 should look for, for instance `person`")
    if len(parts) == 1:
        own = re.match(r"^(.+?)\s*:\s*([\d.]+)\s*$", parts[0])
        name, count = (own.group(1).strip(), max(1, round(float(own.group(2))))) if own else (parts[0], max(int(most), 1))
        if count == 1:
            return name
    return ", ".join(p if re.match(r"^.+?\s*:\s*[\d.]+\s*$", p) else f"{p}:{max(int(most), 1)}" for p in parts)


def flat_borders(small: torch.Tensor) -> tuple[int, int, int, int]:
    """How many rows and columns of `small` ([F, h, w]) are border: (top, bottom, left, right).

    A row or column is border when every value in it, in every frame, lies
    within `BORDER_FLAT` of one value: the bars of a picture padded or
    letterboxed into another shape. Counted inward from each edge to the first
    line that is not. A strip that is flat in one shot and not in the next is
    picture, and so is a still background, which is not one flat value. A clip
    that is flat all over has no border, since it has no picture either.
    """
    if small.shape[0] == 0:
        return 0, 0, 0, 0
    low, high = small.amin(dim=0), small.amax(dim=0)
    rows = ((high.amax(dim=1) - low.amin(dim=1)) <= BORDER_FLAT).tolist()
    cols = ((high.amax(dim=0) - low.amin(dim=0)) <= BORDER_FLAT).tolist()

    def run(flags) -> int:
        n = 0
        for flat in flags:
            if not flat:
                break
            n += 1
        return n

    top, bottom, left, right = run(rows), run(rows[::-1]), run(cols), run(cols[::-1])
    if top + bottom >= len(rows) or left + right >= len(cols):
        return 0, 0, 0, 0
    return top, bottom, left, right


def borders_line(found: dict, width: int, height: int) -> str:
    """The report's words about borders left out of the cut score, in the frames' own pixels; empty when there are none."""
    top, bottom, left, right = (found or {}).get("borders", (0, 0, 0, 0))
    w, h = CUT_SIZE
    sides = [(top, "top", height / h), (bottom, "bottom", height / h), (left, "left", width / w), (right, "right", width / w)]
    named = [f"about {int(round(n * scale))} px at the {side}" for n, side, scale in sides if n]
    return ("flat borders left out of the cut score: " + ", ".join(named)) if named else ""


def cut_scores(frames: torch.Tensor, found: dict | None = None) -> torch.Tensor:
    """[F, H, W, C] frames to [F - 1] scores; entry i is the step into frame i + 1.

    One minus the correlation of the two frames' gradient-magnitude maps at
    `CUT_SIZE`. About 0 for a held shot, well under 1 for a lighting change or
    ordinary movement, about 1 across a cut.

    Flat borders (`flat_borders`) are left out, with the one line beside each
    that the reduction mixes with the bar, so the score is the picture's. A
    clip with no border scores exactly as it did before borders were looked
    for. `found`, when given, receives `borders`: the rows and columns left
    out at `CUT_SIZE`, as (top, bottom, left, right).
    """
    w, h = CUT_SIZE
    parts = []
    for i in range(0, frames.shape[0], 64):
        grey = frames[i:i + 64, ..., :3].to(torch.float32).mean(dim=-1, keepdim=True).movedim(-1, 1)
        parts.append(F.interpolate(grey, size=(h, w), mode="area")[:, 0])
    if not parts:
        return torch.zeros(0)
    small = torch.cat(parts, dim=0)
    top, bottom, left, right = flat_borders(small)
    if found is not None:
        found["borders"] = (top, bottom, left, right)
    # beside a bar the reduction's line is part bar and part picture: an edge in every frame, so it goes too
    small = small[:, top + bool(top):h - bottom - bool(bottom), left + bool(left):w - right - bool(right)]
    edges = (small[:, 1:, 1:] - small[:, 1:, :-1]).abs() + (small[:, 1:, 1:] - small[:, :-1, 1:]).abs()
    e = edges.flatten(1)
    e = e - e.mean(dim=1, keepdim=True)
    e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return (1.0 - (e[1:] * e[:-1]).sum(dim=1)).cpu()


def find_cuts(scores: torch.Tensor, threshold: float) -> list[int]:
    """The frames that start a new shot: every frame whose step in scores at or above `threshold`."""
    return [int(i) + 1 for i in (scores >= float(threshold)).nonzero().flatten()]


def auto_cuts(scores) -> float:
    """The cut threshold from the clip's own step scores.

    The middle of the widest gap between neighbouring scores whose upper side
    is at or above `CUT_FLOOR`, when that gap is at least `CUT_MIN_GAP` wide:
    cuts are few and score high, everything else is many and scores lower.
    With no such gap, `CUT_THRESHOLD`.
    """
    ordered = sorted((float(v) for v in scores), reverse=True)
    widest, at = 0.0, float(CUT_THRESHOLD)
    for high, low in zip(ordered, ordered[1:]):
        if high < CUT_FLOOR:
            break
        if high - low >= CUT_MIN_GAP and high - low > widest:
            widest, at = high - low, (high + low) / 2
    return at


def cuts_line(scores, at: float, named: bool, most: int = 12) -> str:
    """The report's line about the cut threshold: the highest step scores, in order, with ` | ` at the threshold."""
    top = sorted((float(v) for v in scores), reverse=True)[:most]
    above = " ".join(f"{v:.2f}" for v in top if v >= at)
    below = " ".join(f"{v:.2f}" for v in top if v < at)
    return (f"cut threshold: {'the value named' if named else 'automatic'}, {at:.2f}; "
            f"highest steps: {above or '(none)'} | {below or '(none)'}")


def shot_ranges(n_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each shot, in order, covering every frame."""
    starts = [0] + [c for c in sorted(set(cuts)) if 0 < c < n_frames]
    return [(s, e) for s, e in zip(starts, starts[1:] + [int(n_frames)]) if e > s]


def choose(masks: torch.Tensor, scores: list[float], pick: str) -> int | None:
    """Which of the detections on the pick frame is the subject. None when there are none.

    `masks` is [N, H, W] and `scores` the detector's score for each, as core
    returns them (highest score first).
    """
    if masks.shape[0] == 0:
        return None
    if pick == PICK_SCORE:
        return int(max(range(len(scores)), key=lambda i: scores[i])) if scores else 0
    on = masks > 0.5
    if pick == PICK_LARGEST:
        return int(on.flatten(1).sum(dim=1).argmax())
    if pick == PICK_CENTRAL:
        h, w = masks.shape[-2:]
        ys = torch.arange(h, dtype=torch.float32).view(1, h, 1) / max(h - 1, 1) - 0.5
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, w) / max(w - 1, 1) - 0.5
        area = on.flatten(1).sum(dim=1).clamp(min=1).to(torch.float32)
        cy = (on * ys).flatten(1).sum(dim=1) / area
        cx = (on * xs).flatten(1).sum(dim=1) / area
        return int((cx ** 2 + cy ** 2).argmin())
    raise ValueError(f"unknown pick {pick!r}; one of {list(PICKS)}")


def top_third(mask: torch.Tensor) -> torch.Tensor:
    """A [H, W] mask cut to the top third of the rows it covers: the head and shoulders of a standing person.

    People in a clip often share their clothes and never their heads, so the
    signature is taken here. A mask that covers nothing comes back unchanged.
    """
    rows = (mask > 0.5).any(dim=1).nonzero().flatten()
    if rows.numel() == 0:
        return mask
    top, bottom = int(rows.min()), int(rows.max())
    out = mask.clone()
    out[top + max(1, (bottom - top + 1) // 3):] = 0
    return out


def signature(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """The trunk's features averaged under a mask, unit length. [C, h, w] and [H, W] to [C].

    None when the mask covers no feature cell.
    """
    m = F.interpolate(mask[None, None].to(torch.float32), size=features.shape[-2:], mode="area")[0, 0]
    total = m.sum()
    if float(total) < 1e-3:
        return None
    v = (features.to(torch.float32) * m).sum(dim=(1, 2)) / total
    return v / v.norm().clamp(min=1e-6)


def relative(sig: torch.Tensor | None, centre: torch.Tensor | None) -> torch.Tensor | None:
    """A signature with `centre` subtracted, unit length again. Unchanged when there is no centre."""
    if sig is None or centre is None:
        return sig
    v = sig - centre
    return v / v.norm().clamp(min=1e-6)


def similarity(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    """Cosine similarity of two signatures; -1 when either is missing."""
    if a is None or b is None:
        return -1.0
    return float((a * b).sum())


def head_of(person: torch.Tensor, heads: torch.Tensor) -> torch.Tensor | None:
    """The head that belongs to a person: of the heads lying mostly inside the person's mask, the highest.

    `person` is [H, W] and `heads` [N, H, W]. None when no head does.
    """
    inside, best, best_top = person > 0.5, None, None
    for head in heads:
        on = head > 0.5
        area = int(on.sum())
        if area == 0 or int((on & inside).sum()) * 2 < area:
            continue
        top = int(on.any(dim=1).nonzero().min())
        if best_top is None or top < best_top:
            best, best_top = head, top
    return best


def _views(sig) -> tuple:
    """A person's signatures as a tuple: one per place they are compared. A lone signature is one view."""
    return tuple(sig) if isinstance(sig, (tuple, list)) else (sig,)


def _has_head(sig) -> bool:
    """Whether a person's last view exists: with SAM's callables that is the head."""
    return _views(sig)[-1] is not None


def _width(mask: torch.Tensor) -> int:
    """How many columns a [H, W] mask covers: a rough measure of how closely a person is framed."""
    return int((mask > 0.5).any(dim=0).sum())


def auto_match(scores: list[float], floor: float) -> float:
    """Where the subject ends and somebody else begins, from the shots' best scores.

    The floor, or the middle of the widest gap between neighbouring scores
    when that gap is at least `MIN_GAP` wide, its upper side is at or above
    the floor and its middle is too. With no such gap everything at or above
    the floor is the subject, which is right when they are in every shot.
    """
    ordered = sorted((float(s) for s in scores if s >= 0), reverse=True)
    widest, cut = 0.0, float(floor)
    for high, low in zip(ordered, ordered[1:]):
        if high >= floor and high - low >= MIN_GAP and high - low > widest:
            widest, cut = high - low, (high + low) / 2
    return max(cut, float(floor))


@dataclass
class Shot:
    """What `follow` did with one shot, for the report and the preview."""
    start: int
    end: int
    probe: int                   # the frame the shot is first judged on
    seed: int | None = None      # the frame the subject was taken on; None when absent
    best: float = -1.0           # the best similarity seen
    shown: int = 0               # the frame the preview shows: the seed, or where the best was seen
    index: int | None = None     # which detection on `shown` is the subject, or the best one when absent
    candidates: int = 0          # detections on `shown`
    picked: bool = False         # the shot the pick was made in
    lone: bool = False           # taken under the line, as the only person on its frame
    width: int = 0               # columns the mask on `shown` covers
    seen: int = 0                # the most detections on any frame looked at
    corrected: str = ""          # what a correction said of this shot: `person 2`, `none`, or nothing
    # each time the track was seeded again inside the shot: (the first frame it had left empty, the frame it was
    # seeded on, detections there, similarity to the shot's gallery, the next person's similarity or -1)
    regained: list[tuple[int, int, int, float, float]] = field(default_factory=list)
    searched: list[tuple[int, int]] = field(default_factory=list)   # empty runs searched with nothing found: (first, last)
    # every frame probed after a loss: (frame, detections, best similarity to the gallery, the next person's or -1)
    probes: list[tuple[int, int, float, float]] = field(default_factory=list)
    gallery: list[int] = field(default_factory=list)                # the frames the gallery was taken on


@dataclass
class Followed:
    """`follow`'s result."""
    shots: list[Shot]
    pieces: dict[int, torch.Tensor] = field(default_factory=dict)   # {first frame: [n, H, W]}
    pick_frame: int | None = None    # None when nothing was detected to pick
    others: int = 0                  # people on the pick frame the comparison is relative to
    match: float = 0.0               # the similarity a shot had to reach
    pick_width: int = 0              # columns the subject's mask covers on the pick frame
    looks: list[tuple[int, int, int, float]] = field(default_factory=list)   # (shot, frame, detections, best similarity)
    views: int = 1                   # places a person is compared: head and shoulders, and the head
    views_used: int = 1              # of those, how many the subject has on the pick frame
    passed_over: list[tuple[int, int]] = field(default_factory=list)   # (shot, frame): a favourite with no head, not counted for the pick
    # the subject as the picked shot's own track showed them: the frames, and each frame's signature per place compared
    gallery_frames: list[int] = field(default_factory=list)
    gallery: list[tuple] = field(default_factory=list)
    gallery_given: int = 0           # signatures handed in from an earlier run, which then chose the pick
    # with a gallery handed in, every frame looked at for the pick: (frame, detections, best similarity, the next person's or -1)
    pick_probes: list[tuple[int, int, float, float]] = field(default_factory=list)


_CORRECTION = re.compile(r"^shot\s*(\d+)\s*[:=]?\s*(?:person\s*(\d+)|(none))$", re.IGNORECASE)


def parse_corrections(text: str, n_shots: int) -> dict[int, int | None]:
    """`shot 3: person 2` and `shot 5: none`, one per line (or separated by `;`), as {shot: person or None}.

    Shots and people count from 1, as the report and the tiles print them. A line that is not a correction, a
    shot the clip does not have and a shot named twice are refused by name.
    """
    out: dict[int, int | None] = {}
    for piece in re.split(r"[\n;]", str(text or "")):
        piece = piece.strip()
        if not piece:
            continue
        found = _CORRECTION.match(piece)
        if found is None:
            raise ValueError(f"corrections: cannot read `{piece}`; write `shot 3: person 2` or `shot 3: none`, one per line")
        shot, person = int(found[1]), (None if found[3] else int(found[2]))
        if not 1 <= shot <= int(n_shots):
            raise ValueError(f"corrections: `{piece}` names shot {shot}, and the clip has {int(n_shots)} shot(s)")
        if person is not None and person < 1:
            raise ValueError(f"corrections: `{piece}` names person {person}; people are numbered from 1")
        if shot in out:
            raise ValueError(f"corrections: shot {shot} is corrected twice")
        out[shot] = person
    return out


def main_subject(shots: list[Shot], pick: str, detect, sign,
                 passed_over: list[tuple[int, int]] | None = None) -> tuple[int, int] | None:
    """The frame and detection to pick when no frame is named.

    The rule is applied on every shot's probe frame. The people it favours are
    grouped by plain similarity, and the group covering the most frames wins;
    within it, the frame showing the most people, then the earliest.

    A favourite with no head is not counted while any favourite has one, the
    rule a match already follows: with no head it is likelier a thing than a
    person. Each one left out is appended to `passed_over` as (shot number,
    frame). When no favourite has a head, all of them count.
    """
    winners, headed = [], []
    for shot in shots:
        masks, scores = detect(shot.probe)
        i = choose(masks, scores, pick)
        if i is not None:
            sig = sign(shot.probe, masks[i])
            winners.append((shot, i, _views(sig)[0], int(masks.shape[0])))
            headed.append(_has_head(sig))
    if not winners:
        return None
    if any(headed) and not all(headed):
        if passed_over is not None:
            passed_over.extend((shots.index(w[0]) + 1, w[0].probe) for w, has in zip(winners, headed) if not has)
        winners = [w for w, has in zip(winners, headed) if has]

    def same(a, b) -> bool:
        return a is b or similarity(a[2], b[2]) >= PLAIN_SAME

    def frames_won(a) -> int:
        return sum(w[0].end - w[0].start for w in winners if same(a, w))

    lead = max(winners, key=lambda a: (frames_won(a), -a[0].start))
    shot, i, _, _ = max((w for w in winners if same(lead, w)), key=lambda w: (w[3], -w[0].start))
    return shot.probe, i


def empty_runs(piece: torch.Tensor) -> list[tuple[int, int]]:
    """The runs of frames a tracked piece [n, H, W] leaves empty, as (first, one past the last) in the piece's own numbers."""
    on = (piece > 0.5).flatten(1).any(dim=1).tolist()
    runs, first = [], None
    for i, v in enumerate(on + [True]):
        if not v and first is None:
            first = i
        elif v and first is not None:
            runs.append((first, i))
            first = None
    return runs


def gallery_frames(piece: torch.Tensor, most: int = GALLERY_MOST) -> list[int]:
    """Up to `most` frames of a tracked piece that carry a mask; the piece's own numbers.

    Spread evenly over the frames that carry one, and the frames where the mask is smallest and largest are among
    them, each in place of the evenly spread frame nearest to it: on the card a subject found again scored lower
    the further his size was from the gallery's (0.89 to 0.91 at a quarter of it, 0.95 to 0.97 close to it), so
    the gallery should hold the sizes the track saw and not only its times. Measured on one clip, 2026-10-06; on
    that clip the smallest frame was the last one, was in the gallery anyway and was never the best match.
    """
    area = (piece > 0.5).flatten(1).sum(dim=1)
    on = (area > 0).nonzero().flatten().tolist()
    if len(on) <= most:
        return on
    chosen = sorted({on[round(k * (len(on) - 1) / (most - 1))] for k in range(most)})
    for extreme in (min(on, key=lambda f: (int(area[f]), f)), max(on, key=lambda f: (int(area[f]), -f))):
        if extreme not in chosen:
            nearest = min(chosen, key=lambda f: abs(f - extreme))
            chosen[chosen.index(nearest)] = extreme
    return sorted(set(chosen))


def gallery_scores(found: torch.Tensor, frame: int, gallery: list[tuple], sign) -> list[float]:
    """Each detection's plain similarity to a gallery of the subject: its best over the gallery, the lower of the places compared.

    The places compared are the ones every gallery entry has. A detection that lacks one of them scores -1: with no
    head where the subject always had one, it is likelier a thing or a person cut off by the frame than the subject.
    """
    held = [k for k in range(len(gallery[0])) if all(g[k] is not None for g in gallery)] if gallery else []
    out = []
    for m in found:
        views = _views(sign(frame, m))
        if not held or any(k >= len(views) or views[k] is None for k in held):
            out.append(-1.0)
            continue
        out.append(min(max(similarity(g[k], views[k]) for g in gallery) for k in held))
    return out


def clear_best(sims: list[float]) -> tuple[int | None, float, float]:
    """Which detection is the subject by `gallery_scores`, with its score and the next person's: at or above
    `REGAIN_SAME` and `REGAIN_MARGIN` clear of the next. None when nobody is."""
    order = sorted(range(len(sims)), key=lambda k: -sims[k])
    best = sims[order[0]] if order else -1.0
    second = sims[order[1]] if len(order) > 1 else -1.0
    taken = order[0] if order and best >= REGAIN_SAME and best - second >= REGAIN_MARGIN else None
    return taken, float(best), float(second)


def follow(n_frames: int, cuts: list[int], pick: str, pick_frame: int | None, match_threshold: float | None,
           detect: Callable[[int], tuple[torch.Tensor, list[float]]],
           sign: Callable[[int, torch.Tensor], torch.Tensor | None],
           track: Callable[[int, int, int, torch.Tensor], torch.Tensor],
           stride: int = PROBE_STRIDE, offset: int = PROBE_OFFSET,
           corrections: dict[int, int | None] | None = None, gallery: list[tuple] | None = None) -> Followed:
    """The subject's mask per frame, shot by shot. The model work is in three callables.

    `detect(frame)` returns that frame's detections, [N, H, W] and their
    scores. `sign(frame, mask)` returns a mask's signature on that frame, or
    a tuple of them, one per place a person is compared; a match is as good
    as its worst view, and a person lacking a view the subject has is no match.
    `track(start, end, seed, mask)` returns the masks of frames `start` to
    `end` exclusive, tracked from `mask` on `seed`, in frame order.

    `pick_frame` None picks automatically (`main_subject`); `match_threshold`
    None cuts automatically (`auto_match`).

    `corrections` is {shot number from 1: person number from 1, or None for
    nobody} (`parse_corrections`). The automatic pass runs as it would
    without them, so each shot's `shown` frame is the one its tile shows
    either way; a corrected shot is then seeded on that frame from the
    person of that number there, or left empty.

    `gallery` is the subject as an earlier run's track showed them
    (`Followed.gallery` of that run, through the shot table). With one and
    no frame named, the pick is not made by `pick`'s rule: each shot's probe
    frame and then every `stride`-th frame is looked at, and the first frame
    with a person who is the subject by the gallery (`clear_best`) is the
    pick. When nobody is, nothing is picked and every mask is empty: a gap
    is the honest answer where the rule would take somebody else.
    """
    corrections = dict(corrections or {})
    shots = [Shot(s, e, probe=min(s + max(int(offset), 0), e - 1)) for s, e in shot_ranges(n_frames, cuts)]
    for shot in shots:
        shot.shown = shot.probe
    for number in corrections:
        if not 1 <= int(number) <= len(shots):
            raise ValueError(f"corrections: shot {number} is named, and the clip has {len(shots)} shot(s)")
    by_hand = {id(shots[int(number) - 1]) for number in corrections}   # tracked once, from the corrected seed
    result = Followed(shots)
    given = list(gallery or [])
    result.gallery_given = len(given)
    if pick_frame is None and given:
        entry = None
        probes = [s.probe for s in shots]
        step = max(int(stride), 1)
        for f in probes + [f for f in range(0, int(n_frames), step) if f not in set(probes)]:
            found, _ = detect(f)
            which, best, second = clear_best(gallery_scores(found, f, given, sign))
            result.pick_probes.append((int(f), int(found.shape[0]), best, second))
            if which is not None:
                entry = (int(f), which)
                break
    elif pick_frame is None:
        entry = main_subject(shots, pick, detect, sign, result.passed_over)
    else:
        if not 0 <= int(pick_frame) < int(n_frames):
            raise ValueError(f"pick_frame {pick_frame} is outside the clip's {n_frames} frames")
        masks, scores = detect(int(pick_frame))
        which = choose(masks, scores, pick)
        entry = None if which is None else (int(pick_frame), which)
    if entry is None:
        _correct(result, corrections, detect, track)
        return result
    frame0, which = entry
    masks0, _ = detect(frame0)
    picked = masks0[which]
    mine = _views(sign(frame0, picked))
    theirs = [_views(sign(frame0, m)) for i, m in enumerate(masks0) if i != which]
    use = [k for k, v in enumerate(mine) if v is not None]         # the views the subject has
    others = [t for t in theirs if t[0] is not None]
    centres = []
    for k in range(len(mine)):
        have = [t[k] for t in theirs if t[k] is not None]
        centres.append(torch.stack(have, dim=0).mean(dim=0) if have else None)
    reference = tuple(relative(v, c) for v, c in zip(mine, centres))
    result.pick_frame, result.others, result.pick_width = frame0, len(others), _width(picked)
    result.views, result.views_used = len(mine), len(use)

    def alike(sig) -> float | None:
        """A person's similarity to the subject: the lowest over the views the subject has. None when one is missing."""
        views = _views(sig)
        if not use or any(views[k] is None for k in use):
            return None
        return min(similarity(reference[k], relative(views[k], centres[k])) for k in use)

    def judge(f: int) -> tuple[float, int | None, int, torch.Tensor]:
        """Frame `f`'s best similarity, which detection has it, how many there have a head, and the detections."""
        found, _ = detect(f)
        sigs = [sign(f, m) for m in found]
        sims = [alike(v) for v in sigs]
        heads = sum(1 for v in sigs if _has_head(v))
        able = [k for k, v in enumerate(sims) if v is not None]
        if not able:
            return -1.0, None, heads, found
        i = max(able, key=lambda k: sims[k])
        return sims[i], i, heads, found

    def look(shot: Shot, f: int) -> tuple[float, int | None, int]:
        """`judge` on frame `f` for a shot being placed; the shot remembers its best."""
        score, i, heads, found = judge(f)
        shot.seen = max(shot.seen, int(found.shape[0]))
        result.looks.append((shots.index(shot) + 1, f, int(found.shape[0]), score))
        if i is not None and score > shot.best:
            shot.best, shot.shown, shot.index = score, f, i
            shot.candidates, shot.width = int(found.shape[0]), _width(found[i])
        return score, i, heads

    def regain(shot: Shot) -> None:
        """Look again for a subject the track let go inside `shot`, and seed the track again where they are found.

        A run of empty frames is probed from the side that is tracked: forward after a loss, backward into a
        run the shot opens on. Each person on a probed frame is compared with the shot's gallery, the subject
        as this shot's own track showed them; the first frame whose best person is at or above `REGAIN_SAME`
        and `REGAIN_MARGIN` clear of the next person there is taken. What the shot's tile shows is not moved.
        """
        piece = result.pieces.get(shot.start)
        if piece is None:
            return
        step, done, known = max(int(stride), 1), set(), None
        while len(shot.regained) < REGAIN_MOST:
            runs = [(a + shot.start, b + shot.start) for a, b in empty_runs(piece)
                    if b - a >= REGAIN_MIN_RUN and a + shot.start not in done]
            if not runs:
                break
            if known is None:
                # what the track held before anything was seeded again, with whatever an earlier run handed in
                if shot.picked and result.gallery:
                    shot.gallery, own = list(result.gallery_frames), list(result.gallery)
                else:
                    shot.gallery = [f + shot.start for f in gallery_frames(piece)]
                    own = [_views(sign(f, (piece[f - shot.start] > 0.5).to(torch.float32))) for f in shot.gallery]
                known = given + own
            a, b = runs[0]
            done.add(a)
            frames = range(a, b, step) if a > shot.start else range(b - 1, a - 1, -step)
            for f in frames:
                found, _ = detect(f)
                which, best, second = clear_best(gallery_scores(found, f, known, sign))
                shot.probes.append((f, int(found.shape[0]), best, second))
                if which is not None:
                    piece[a - shot.start:b - shot.start] = track(a, b, f, found[which])
                    shot.regained.append((a, f, int(found.shape[0]), best, second))
                    break
            else:
                shot.searched.append((a, b - 1))

    rest = [s for s in shots if not s.start <= frame0 < s.end]
    first = {id(s): look(s, s.probe) for s in rest}
    floor = MATCH_FLOOR if others else PLAIN_FLOOR
    result.match = float(match_threshold) if match_threshold is not None else auto_match([s.best for s in rest], floor)
    # nobody else on the pick frame and nothing named: there is nobody to mistake the subject for
    alone = match_threshold is None and not others

    for shot in shots:
        if shot.start <= frame0 < shot.end:
            shot.picked, shot.seed, shot.shown, shot.index = True, frame0, frame0, which
            shot.best, shot.candidates, shot.width = 1.0, int(masks0.shape[0]), result.pick_width
            if id(shot) not in by_hand:
                piece = result.pieces[shot.start] = track(shot.start, shot.end, frame0, picked)
                # the subject as this track shows them, for a loss later in the shot and for the run after this one
                result.gallery_frames = [f + shot.start for f in gallery_frames(piece)]
                result.gallery = [_views(sign(f, (piece[f - shot.start] > 0.5).to(torch.float32))) for f in result.gallery_frames]
                regain(shot)
            continue
        taken = lone = None        # (similarity, frame, detection)
        later = [f for f in range(shot.start, shot.end, max(int(stride), 1)) if f != shot.probe]
        for f in [shot.probe] + later:
            score, i, heads = first[id(shot)] if f == shot.probe else look(shot, f)
            if i is None:
                continue
            if score >= result.match:
                taken = (score, f, i)
                break
            if alone and heads == 1 and (lone is None or score > lone[0]):
                lone = (score, f, i)
        if taken is None and lone is not None:
            taken, shot.lone = lone, True
        if taken is None:
            continue
        score, seed, i = taken
        seeds, _ = detect(seed)
        shot.seed, shot.shown, shot.index, shot.best = seed, seed, i, score
        shot.candidates, shot.width = int(seeds.shape[0]), _width(seeds[i])
        if id(shot) not in by_hand:
            result.pieces[shot.start] = track(shot.start, shot.end, seed, seeds[i])
            regain(shot)
    _correct(result, corrections, detect, track)
    return result


def _correct(result: Followed, corrections: dict[int, int | None], detect, track) -> None:
    """Apply the corrections to `follow`'s result: each on the frame its shot's tile shows."""
    for number, person in sorted(corrections.items()):
        shot = result.shots[int(number) - 1]
        if person is None:
            shot.seed, shot.lone, shot.corrected = None, False, "none"
            result.pieces.pop(shot.start, None)
            continue
        people, _ = detect(shot.shown)
        which = shot_table.detection_of(people, int(person))
        if which is None:
            raise ValueError(
                f"corrections: shot {number} has {int(people.shape[0])} person(s) on frame {shot.shown}, the frame "
                f"its tile shows, so there is no person {int(person)} to take")
        shot.seed, shot.index, shot.lone, shot.corrected = shot.shown, which, False, f"person {int(person)}"
        shot.candidates, shot.width = int(people.shape[0]), _width(people[which])
        result.pieces[shot.start] = track(shot.start, shot.end, shot.shown, people[which])


def assemble(n_frames: int, height: int, width: int, pieces: dict[int, torch.Tensor]) -> torch.Tensor:
    """[n_frames, height, width] of 0 or 1: the tracked pieces in place, zeros where the subject is absent."""
    out = torch.zeros((int(n_frames), int(height), int(width)), dtype=torch.float32)
    for start, piece in pieces.items():
        out[start:start + piece.shape[0]] = (piece > 0.5).to(torch.float32)
    return out


def _state(shot: Shot) -> str:
    if shot.corrected:
        return "absent (corrected)" if shot.seed is None else "taken (corrected)"
    if shot.picked:
        return "picked"
    if shot.seed is None:
        return "absent"
    return "taken (only person)" if shot.lone else "taken"


def _regained(shot: Shot) -> str:
    """The report's words for a track seeded again inside its shot, and for the runs searched in vain; empty when neither."""
    said = [f"let go on frame {a} and found again on frame {f} (similarity {score:.2f} to the shot's own track, "
            + (f"the next person there {second:.2f}, " if second >= 0 else "") + f"{n} detection(s) there)"
            for a, f, n, score, second in shot.regained]
    if shot.searched:
        best = max((p[2] for p in shot.probes), default=-1.0)
        said.append("searched and not found on frames " + ", ".join(f"{a}-{b}" for a, b in shot.searched)
                    + (f" (the best person on any frame looked at scored {best:.2f}; the line is {REGAIN_SAME:.2f})"
                       if best >= 0 else " (no detection on any frame looked at)"))
    return "".join("; " + text for text in said)


def _framing(shot: Shot, found: Followed) -> str:
    """Words for a shot framed much closer or wider than the pick frame, where the top third is another part of a person."""
    if not shot.width or not found.pick_width:
        return ""
    ratio = shot.width / found.pick_width
    if 2 / 3 <= ratio <= 1.5:
        return ""
    return f"; framed differently, the mask is {ratio:.1f} times as wide as on the pick frame"


def report(found: Followed, cuts: list[int], pick: str, phrase: str, named_frame: bool, named_value: bool,
           seconds: float, cutting: str | None = None) -> str:
    """What was found, in words: the cuts, the pick, the line between the subject and others, each shot, the cost.

    `cutting` is `cuts_line`'s sentence about the cut threshold, when the caller has one.
    """
    shots = found.shots
    lines = [f"{len(shots)} shot(s); cuts at frame(s) {cuts if cuts else 'none'}"]
    if cutting:
        lines.append(cutting)
    if found.pick_frame is None and found.gallery_given:
        best = max((p[2] for p in found.pick_probes), default=-1.0)
        lines.append(f"nobody matching `{phrase}` on the {len(found.pick_probes)} frame(s) looked at is the subject an earlier run "
                     f"handed in (the best scored {best:.2f}; the line is {REGAIN_SAME:.2f}): no subject, every mask is empty")
    elif found.pick_frame is None:
        lines.append(f"nothing matched `{phrase}` to pick from: no subject, every mask is empty")
    else:
        how = "the frame named" if named_frame else "chosen automatically: the person that rule favours for most of the clip"
        if found.gallery_given and not named_frame:
            taken = next((p for p in found.pick_probes if p[0] == found.pick_frame), None)
            passed = [p for p in found.pick_probes if p[0] != found.pick_frame]
            how = (f"chosen as the subject an earlier run handed in: similarity {taken[2]:.2f} to its {found.gallery_given} "
                   f"signature(s)" + (f", the next person there {taken[3]:.2f}" if taken[3] >= 0 else "")
                   + (f"; {len(passed)} earlier frame(s) refused, the best of them {max(p[2] for p in passed):.2f}" if passed else ""))
            lines.append(f"subject: the `{phrase}` on frame {found.pick_frame} ({how})")
        else:
            lines.append(f"subject: the {pick} `{phrase}` on frame {found.pick_frame} ({how})")
        if found.passed_over:
            lines.append("not counted for the pick, no head was found on them: the "
                         + f"{pick} `{phrase}` of " + ", ".join(f"shot {n} (frame {f})" for n, f in found.passed_over))
        if found.others:
            lines.append(f"similarity is relative to the {found.others} other(s) on that frame")
        else:
            lines.append("nobody else on that frame: plain similarity, where other people score about 0.8 to 0.9"
                         + ("" if named_value else "; a shot's best frame showing one person with a head is taken whatever it scores"))
        if found.views > 1:
            lines.append("a match has to hold in two places, the head and shoulders and the head; the lower one counts"
                         if found.views_used == found.views else
                         "no head was found on the subject on that frame: matched by the head and shoulders alone")
        scores = sorted((s.best for s in shots if not s.picked and s.best >= 0), reverse=True)
        above = " ".join(f"{v:.2f}" for v in scores if v >= found.match)
        below = " ".join(f"{v:.2f}" for v in scores if v < found.match)
        lines.append(f"match: {'the value named' if named_value else 'automatic'}, {found.match:.2f}; "
                     f"shots' scores: {above or '(none)'} | {below or '(none)'}")
    for n, s in enumerate(shots, 1):
        span = f"[{n}] frames {s.start}-{s.end - 1}"
        if s.corrected:
            lines.append(f"{span}: corrected by hand, {s.corrected} of the {s.candidates} detection(s) on frame {s.shown}"
                         if s.seed is not None else f"{span}: corrected by hand, nobody taken")
        elif s.picked:
            lines.append(f"{span}: the picked shot, {s.candidates} detection(s) on frame {s.seed}{_regained(s)}")
        elif s.seed is not None:
            how = "as the only person there, " if s.lone else ""
            lines.append(f"{span}: taken on frame {s.seed}, {how}similarity {s.best:.2f}, "
                         f"{s.candidates} detection(s) there{_framing(s, found) if s.lone else ''}{_regained(s)}")
        else:
            why = (f"best similarity {s.best:.2f} on frame {s.shown}{_framing(s, found)}" if s.best >= 0 else
                   "no detection" if not s.seen else f"up to {s.seen} detection(s), none that could be compared")
            lines.append(f"{span}: absent ({why})")
    lines.append(f"{seconds:.0f} s")
    return "\n".join(lines)


def _outline(mask: torch.Tensor, width: int) -> torch.Tensor:
    """The inner edge of a [H, W] mask, `width` pixels thick."""
    m = (mask > 0.5).to(torch.float32)[None, None]
    inner = -F.max_pool2d(-m, 2 * width + 1, stride=1, padding=width)
    return (m - inner)[0, 0] > 0.5


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an older Pillow has one size
        return ImageFont.load_default()


def preview(frames: torch.Tensor, mask: torch.Tensor, shots: list[Shot], detect) -> torch.Tensor:
    """One labelled frame per shot, [shots, h, TILE_WIDTH, 3].

    The frame is the one the shot was taken on, or where its best candidate
    was seen. The subject's mask is tinted. Every detection there is outlined:
    green for the one taken, orange for the best candidate of an absent shot,
    white for the rest. Each outline carries its person number at the top,
    the number the shot table and a correction use (`shot_table.person_order`:
    left to right). The label gives the shot, its frames, its score and
    whether it was picked, taken or absent.
    """
    height, width = int(frames.shape[1]), int(frames.shape[2])
    th = max(2, round(height * TILE_WIDTH / width))
    tiles = []
    for n, s in enumerate(shots, 1):
        img = frames[s.shown, ..., :3].to(torch.float32).cpu().clone()
        tint = mask[s.shown].unsqueeze(-1) * 0.35
        img = img * (1.0 - tint) + torch.tensor([1.0, 0.1, 0.1]) * tint
        found, _ = detect(s.shown)
        for i, m in enumerate(found):
            if i == s.index:
                colour, thick = ([0.1, 1.0, 0.2] if s.seed is not None else [1.0, 0.6, 0.0]), 4
            else:
                colour, thick = [1.0, 1.0, 1.0], 1
            img[_outline(m, thick)] = torch.tensor(colour)
        small = F.interpolate(img.movedim(-1, 0)[None], size=(th, TILE_WIDTH), mode="area")[0].movedim(0, -1)
        pil = Image.fromarray((small.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        score = "" if s.picked or s.best < 0 else f"  {s.best:.2f}"
        text = f"{n}  frames {s.start}-{s.end - 1}{score}  {_state(s)}"
        draw, font = ImageDraw.Draw(pil), _font(22)
        box = draw.textbbox((8, 6), text, font=font)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        label_bottom, label_right = box[3] + 6, box[2] + 8
        subject = shot_table.person_number(found, s.index)
        for number, col, row in shot_table.label_points(found):
            x, y = col * TILE_WIDTH / width, row * th / height + 3
            nb = draw.textbbox((0, 0), str(number), font=font)
            w, h = nb[2] - nb[0] + 10, nb[3] - nb[1] + 8
            x = min(max(x - w / 2, 0), TILE_WIDTH - w)
            if y < label_bottom and x < label_right:      # clear of the shot's own label
                y = label_bottom
            y = min(y, th - h)
            ink = (255, 255, 255) if number != subject else ((40, 255, 60) if s.seed is not None else (255, 160, 0))
            draw.rectangle((x, y, x + w, y + h), fill=(0, 0, 0))
            draw.text((x + 5 - nb[0], y + 4 - nb[1]), str(number), fill=ink, font=font)
        tiles.append(torch.from_numpy(np.asarray(pil).copy()).to(torch.float32) / 255.0)
    return torch.stack(tiles, dim=0) if tiles else torch.zeros((1, th, TILE_WIDTH, 3))


def _sam_callables(segmenter, segmenter_clip, frames: torch.Tensor, phrase: str, detection_threshold: float,
                   max_people: int = MAX_PEOPLE, head_phrase: str = HEAD_PHRASE):
    """`detect`, `sign` and `track` on core's SAM 3: its detect and track nodes, and the model's vision trunk."""
    import comfy.model_management
    import comfy.utils
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_TrackToMask, SAM3_VideoTrack  # core's nodes

    cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(counted(phrase, max_people)))
    head_cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(counted(head_phrase, max_people)))
    head_masks: dict[int, torch.Tensor] = {}
    detections: dict[int, tuple[torch.Tensor, list[float]]] = {}
    features: dict[int, torch.Tensor] = {}

    def detect(f: int):
        if f not in detections:
            out = SAM3_Detect.execute(segmenter, frames[f:f + 1], conditioning=cond,
                                      threshold=float(detection_threshold), individual_masks=True)
            masks, boxes = getattr(out, "args", out)[:2]
            scores = [float(b.get("score", 0.0)) for b in (boxes[0] if boxes else [])]
            detections[f] = (masks.to(torch.float32).cpu(), scores[:int(masks.shape[0])])
        return detections[f]

    def sign(f: int, mask: torch.Tensor):
        if f not in features:
            comfy.model_management.load_model_gpu(segmenter)
            device = comfy.model_management.get_torch_device()
            dtype = segmenter.model.get_dtype()
            x = comfy.utils.common_upscale(frames[f:f + 1, ..., :3].movedim(-1, 1), TRUNK_SIDE, TRUNK_SIDE,
                                           "bilinear", crop="disabled").to(device=device, dtype=dtype)
            trunk = segmenter.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x)
            trunk = trunk[-1] if isinstance(trunk, (list, tuple)) else trunk
            features[f] = trunk[0].to(torch.float32).cpu()
        if f not in head_masks:
            out = SAM3_Detect.execute(segmenter, frames[f:f + 1], conditioning=head_cond,
                                      threshold=float(detection_threshold), individual_masks=True)
            head_masks[f] = getattr(out, "args", out)[0].to(torch.float32).cpu()
        head = head_of(mask, head_masks[f])
        return signature(features[f], top_third(mask)), (None if head is None else signature(features[f], head))

    def run(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = SAM3_VideoTrack.execute(images, segmenter, initial_mask=mask[None].to(torch.float32),
                                      conditioning=None, detection_threshold=float(detection_threshold),
                                      max_objects=1, detect_interval=1)
        data = getattr(out, "args", out)[0]
        masks = SAM3_TrackToMask.execute(data, "")
        return getattr(masks, "args", masks)[0].to(torch.float32).cpu()

    def track(start: int, end: int, seed: int, mask: torch.Tensor) -> torch.Tensor:
        forward = run(frames[seed:end], mask)
        if seed == start:
            return forward
        backward = run(torch.flip(frames[start:seed + 1], dims=[0]), mask)
        return torch.cat([torch.flip(backward, dims=[0])[:-1], forward], dim=0)

    return detect, sign, track


def _selection(value, key: str, nested: str):
    """A DynamicCombo's selection and its nested value (None when the option has none).

    It arrives as one nested dict, the selection under the input's own id. A
    bare string is accepted for a direct call from a check.
    """
    if isinstance(value, str):
        return value, None
    return value[key], value.get(nested)


class MiniMaxH3SubjectTrack(io.ComfyNode):
    #: Part of a kept mask's key (`mask_store.mask_key`). Bump when the same
    #: inputs and settings would give a different mask. 2: the comparison became
    #: relative to the pick frame's other people. 3: every detection of the
    #: phrase is taken, not core's one per phrase. 4: the automatic pick and
    #: the automatic match, and each shot judged a few frames in. 5: the cut
    #: threshold from the clip's scores and inclusive, and a lone person taken
    #: under the line when the pick frame shows nobody else. 6: a match has to
    #: hold on the head as well, and the lone person has to have a head. 7: a
    #: shot's favourite with no head is not counted for the automatic pick. 8:
    #: the cut score leaves a clip's flat borders out.
    MASK_VERSION = 9

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SubjectTrack",
            display_name="MiniMax H3 Subject Track (one person across cuts)",
            category="model/latent/minimax",
            description=(
                "Follows one person through a clip that has cuts and other people in it, and gives one mask "
                "per frame, empty where they are not on screen. Wire the mask into a Masked Source node. "
                "The preview shows who it took in each shot and the report says why."),
            inputs=[
                io.Image.Input("frames", tooltip="The source video's frames, the same ones the Masked Source node gets."),
                io.Model.Input("segmenter", tooltip="The SAM 3 checkpoint's model."),
                io.Clip.Input("segmenter_clip", tooltip="The SAM 3 checkpoint's text encoder."),
                io.String.Input("subject_phrase", default=SUBJECT_PHRASE,
                                tooltip="What SAM 3 is asked to find in each shot. `person` finds everyone."),
                # the default is `most central` since 2026-10-07 (the owner): on the lane's crowd clip
                # `largest` picked a figure at the frame's edge on all three test windows and
                # `most central` the person meant (bench/results/2026-10-07_subject_track_under_nudge.md)
                io.Combo.Input("pick", options=list(PICKS), default=PICK_CENTRAL,
                               tooltip=("Which of the people found is the subject: the one covering the most of "
                                        "the frame, the one nearest its centre, or SAM 3's highest score.")),
                io.DynamicCombo.Input(
                    "pick_on",
                    options=[
                        io.DynamicCombo.Option(AUTOMATIC, []),
                        io.DynamicCombo.Option(PICK_ON_FRAME, [
                            io.Int.Input("pick_frame", default=0, min=0, max=1_000_000,
                                         tooltip="The frame `pick` is applied on."),
                        ]),
                    ],
                    tooltip=("Where the subject is picked. `automatic` takes the person `pick` favours for most "
                             "of the clip. `a frame I name` applies `pick` on one frame.")),
                io.DynamicCombo.Input(
                    "match",
                    options=[
                        io.DynamicCombo.Option(AUTOMATIC, []),
                        io.DynamicCombo.Option(AT_VALUE, [
                            io.Float.Input("match_threshold", default=MATCH_THRESHOLD, min=-1.0, max=1.0, step=0.01,
                                           tooltip="A person in another shot is the subject at or above this similarity."),
                        ]),
                    ],
                    tooltip=("How alike a person in another shot has to be to count as the subject. `automatic` "
                             "sets it from the shots' scores. `at a value` uses yours.")),
                io.DynamicCombo.Input(
                    "cuts",
                    options=[
                        io.DynamicCombo.Option(AUTOMATIC, []),
                        io.DynamicCombo.Option(AT_VALUE, [
                            io.Float.Input("cut_threshold", default=CUT_THRESHOLD, min=0.0, max=2.0, step=0.01,
                                           tooltip="Two consecutive frames at least this different are a cut."),
                        ]),
                    ],
                    tooltip=("How different two consecutive frames have to be to count as a cut. `automatic` "
                             "sets it from the clip's own scores. `at a value` uses yours.")),
                io.Float.Input("detection_threshold", default=DETECTION_THRESHOLD, min=0.0, max=1.0, step=0.01,
                               advanced=True, tooltip="SAM 3's score threshold for a detection of the phrase."),
                io.Int.Input("max_people", default=MAX_PEOPLE, min=1, max=64, advanced=True,
                             tooltip="The most matches of the phrase taken from one frame."),
                # appended 2026-10-04
                io.String.Input("head_phrase", default=HEAD_PHRASE, advanced=True,
                                tooltip=("What SAM 3 is asked to find on each person so they can be told apart. "
                                         "A person in another shot has to match the subject there as well.")),
                # appended 2026-10-05
                io.String.Input("corrections", default="", multiline=True, optional=True,
                                tooltip=("Fixes a shot the node got wrong, one per line. `shot 3: person 2` takes "
                                         "person 2 of shot 3; `shot 3: none` leaves shot 3 alone. The shot's "
                                         "number and each person's number are the ones on the preview.")),
                # appended 2026-10-06
                io.String.Input("subject_from", default="", multiline=False, optional=True,
                                tooltip=("Optional. Follows the same person as an earlier run did: the `shot_table` "
                                         "output of that run's Subject Track wired here, or the path of the "
                                         "`..._shots.json` it saved.\n\nThe person is then picked by how they looked "
                                         "in that run, not by `pick`; if nobody here looks like them, nothing is "
                                         "picked.\n\nFor a long clip rendered in pieces. Leave empty otherwise.")),
            ],
            outputs=[
                io.Mask.Output(display_name="mask", tooltip="One mask per frame at the frames' size; empty where the subject is absent."),
                io.Image.Output(display_name="preview", tooltip="One labelled frame per shot, each person's outline numbered."),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="shot_table",
                                 tooltip="The shots as a table, in JSON: who was found in each, numbered as on the preview, "
                                         "who was taken and why. A Save Shot Table node writes it out for review."),
            ],
        )

    @classmethod
    def execute(cls, frames, segmenter, segmenter_clip, pick_on, match, cuts, subject_phrase=SUBJECT_PHRASE,
                pick=PICK_CENTRAL, detection_threshold=DETECTION_THRESHOLD, max_people=MAX_PEOPLE,
                head_phrase=HEAD_PHRASE, corrections="", subject_from="") -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        where, pick_frame = _selection(pick_on, "pick_on", "pick_frame")
        how, match_threshold = _selection(match, "match", "match_threshold")
        cutting, cut_threshold = _selection(cuts, "cuts", "cut_threshold")
        if where not in (AUTOMATIC, PICK_ON_FRAME) or how not in (AUTOMATIC, AT_VALUE) or cutting not in (AUTOMATIC, AT_VALUE):
            raise ValueError(f"unknown pick_on {where!r}, match {how!r} or cuts {cutting!r}")
        named_frame, named_value, named_cut = where == PICK_ON_FRAME, how == AT_VALUE, cutting == AT_VALUE
        if named_frame and pick_frame is None:
            raise ValueError("pick_on `a frame I name` needs its `pick_frame`")
        if named_value and match_threshold is None:
            raise ValueError("match `at a value` needs its `match_threshold`")
        if named_cut and cut_threshold is None:
            raise ValueError("cuts `at a value` needs its `cut_threshold`")
        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        began = time.perf_counter()
        seen: dict = {}
        steps = cut_scores(frames, seen)
        cut_at = float(cut_threshold) if named_cut else auto_cuts(steps)
        found_cuts = find_cuts(steps, cut_at)
        by_hand = parse_corrections(corrections, len(shot_ranges(n, found_cuts)))
        handed = shot_table.gallery_from(subject_from) if str(subject_from or "").strip() else None
        detect, sign, track = _sam_callables(segmenter, segmenter_clip, frames, subject_phrase, detection_threshold,
                                             int(max_people), head_phrase)
        with torch.no_grad():
            found = follow(n, found_cuts, pick, int(pick_frame) if named_frame else None,
                           float(match_threshold) if named_value else None, detect, sign, track, corrections=by_hand,
                           gallery=handed)
        mask = assemble(n, h, w, found.pieces)
        text = report(found, found_cuts, pick, subject_phrase, named_frame, named_value, time.perf_counter() - began,
                      cutting="\n".join(filter(None, [cuts_line(steps, cut_at, named_cut), borders_line(seen, w, h)])))
        logger.info("[h3] MiniMaxH3SubjectTrack: %s", text.replace("\n", "; "))
        tiles = preview(frames, mask, found.shots, detect)
        table = shot_table.build(found, detect, mask, state=_state, phrase=subject_phrase, pick=pick,
                                 named_frame=named_frame, named_value=named_value, cuts=found_cuts)
        both = text + "\n\n" + shot_table.as_text(table)
        shown = {**ui.PreviewImage(tiles, cls=cls).as_dict(), **ui.PreviewText(both).as_dict()}
        return io.NodeOutput(mask, tiles, text, shot_table.as_json(table), ui=shown)
