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
   to the shot's end. A shot corrected by hand is searched the same way
   (since 2026-10-07): the correction says who the subject is on the frame
   the tile shows, not that the tracker will hold them to the shot's end.
   Until then a corrected shot was tracked once and left empty from the
   frame the tracker let go, which the first render to pin a small subject
   by hand showed as the original person for the rest of the window.

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

**Nothing keeps this node's mask between runs.** Until 2026-10-10
`MiniMaxH3MaskedSource` kept a finished mask on disk under a key
(`mask_store.py`) and on a hit never asked for its `mask` input, so core did
not run this node at all; that store was removed under the owner's rule that
nothing an earlier run made is reused by a key (`docs/wiki/decisions.md`), and
the hand-bumped `MASK_VERSION` on the node went on 2026-10-24: the code stamp
(`bench/_lib/code_stamp.py`) on every derived file and `ran_code` in the shot
table (the file and the `sha256` of its bytes as imported in the server) are the
authority for which code made a preview or a correction. The preview and the report are
shown as the node's own UI, so nothing has to be wired to see them; the
outputs exist, and a shipped graph leaves them unwired.

**The shot table** (`shot_table.py`) is the same per-shot state as a table a
person reviews once: per shot, who was found, numbered left to right as on
the preview's outlines, who was taken and why. It is the fourth output, and
its text is shown under the report. It reads `follow`'s result and changes
nothing about how anyone is followed.

**One detection a person** (`one_detection_each`, 2026-10-10). The detector
can return one person several times, as masks nested in each other; they
were numbered as several people, their labels drawn on top of each other,
and "the largest" could be a second, looser mask of a person. Detections
that are one thing (`subject_tracks.one_each`) are joined before anything
else sees them, the one the detector scored highest standing for them,
unless that one is under half of the largest and so may be a part: then
they are left as they came, each numbered. The report, the shot table and
the tile's header count what was joined and what was left. A correction typed by
number before this change may name another person now: the numbers count
the people left.

**A seed** (`seeds`, `parse_seeds`, `_seed`; 2026-10-10) is the same thing as
a correction written so that a TOOL can write it: `frame 410: 543, 253, 157, 186`
takes, in the shot that holds frame 410 of this load, the detection on that
frame whose box agrees with the box given (x, y, width, height, as the shot
table writes boxes), and `frame 410: none` leaves that shot empty. A
correction is two numbers read off this tracker's tile, on the frame this
tracker's own pass chose to show, so nothing that has not seen that tile
can write one; a seed names its own frame. A seed that fits nobody leaves
the shot EMPTY and says so (`absent (seed refused)`, with the nearest
detection's box and overlap): a tool that named a person and silently got
somebody else is the worst outcome, and an empty shot with people on it is
already flagged. A seed and a correction on one shot is an error. The
frame is the LOAD's: the node cannot know where its load starts in the
source, so whoever writes the seed does that subtraction, once. A seed on
the PICKED shot that names a person is the pick: every other shot is
judged against the person it names, and the gallery is made from the
seeded track. Two things a seed shares with a correction and does not
fix: on a load where the pick rule finds nobody at all, seeds are applied
and the unseeded shots stay empty, and a seeded track that lets go there
is not looked at again; and a typed correction on the picked shot still
changes that shot alone.

**Every owner's box is a seed on its frame** (2026-10-23, the owner's go).
A shot may carry a box on each of several frames (`parse_seeds`). Each is
named on its own frame (`name_seed`): the detection whose box agrees with
it at `SEED_AGREES`; else, since a box drawn by hand rarely agrees with a
detection's tight box, the one detection no other tracker holds whose mask
overlaps the box at `SEED_OVERLAPS` (refused when the box holds two
people); else what SAM 3 segments inside the box from the box alone (core's
box prompt), clipped to the box and less other trackers' people, never the
rectangle. Each taken box is tracked out to its neighbours' frames and the
tracks are kept frame by frame by the tracker's own score (`merge_tracks`),
not split at a midpoint between boxes. The shot table's `seeds` says what
each box took, how, the frames its track was kept on, and why one was
refused. An empty run holding an owner's box is searched again whatever its
length (`regain`). The head is tracked within each stretch the person
track holds her, seeded inside it (`_track_head_span`): on 1419d it had been
seeded across seventeen empty frames and lost back across them. A shot of
one box whose track has no empty frame takes the same calls as before. A
box marked `body` (`SEED_BODY`: drawn round her arm or body where her head
is hidden) seeds her person track and never her head track. A box the owner
DREW (`drawn`, `SEED_DRAWN`) is approximate, never an outline (the owner,
2026-10-23): it names the one person lying mostly inside it, refuses one
that straddles two people or lies on another tracked subject, and only
where nobody qualifies asks SAM 3's box prompt, unclipped (`name_drawn`).

**What a refused shot leaves behind** (2026-10-10). A shot judged after a
cut used to leave one number, its best similarity. `follow` now keeps, for
every frame it judged, every detection's signature and its likeness to the
subject in each place compared (`Followed.judged`), and the shot table
writes them: `looks` per shot, and the signatures of the people on the shown
frame (`shot_table.LOOKS_ARE`, `SIGNATURES_ARE`). The signatures are there
so a reader can compare a shot's people with ANOTHER tracker's subject: on
the load that prompted this, the plain likeness could not name a person
against a line and could say which of two named subjects a person was more
like (`bench/results/2026-10-10_who_is_who_across_shots.md`). And when the
automatic line stands above its floor, the report and the table say which
two shots' scores it was put between (`moved_by_a_gap`): if the subject is
in both, the lower one was refused by the gap.

**A correction** (`corrections`, `parse_corrections`) fixes one shot by hand
with two numbers: `shot 3: person 2` takes person 2 of shot 3, `shot 3: none`
leaves the shot alone. The shot number is the one the report and the tile
print; the person number is the one drawn on that shot's tile
(`shot_table.person_order`). It is applied on the frame the tile shows, from
the mask the node already holds there, so the numbers a person reads are the
numbers that apply, and nothing is detected again. The automatic pass still
decides which frame each tile shows, and does so the same way with or without
corrections; a corrected shot is tracked from the corrected seed and, where
that track lets go, looked for again like any other shot. The
design is mrhf's (2026-10-05, on the masking board: number the outlines the
tile already draws, and a correction is two numbers typed off it), and the
numbering is mrteal's (`shot_table.py`). The owner's ask: the car clip's wrong
shots fixed without a frontend widget, and "the simpler elegant solution is
always the better one".

Every phrase SAM 3 is given is an input: `subject_phrase` and `head_phrase`.

**Two nodes, one person** (`others`, `held_people`; 2026-10-10). Each node
follows one person and knows nothing of another node in the same graph. On
the first clip with two people followed, the second node's pick rule took the
first node's person for four shots, and every frame of both masks was the
same mask (the day's captures under `data/`; `ON_OTHERS` has the figures).
`others` is the earlier nodes' mask wired into a later one: a detection lying
mostly on it is HELD, and a held detection is never a candidate, anywhere a
person is chosen (the pick rule, a shot's match, a re-find, the people the
scores are taken relative to). It is still LISTED: a frame's people are
always everybody the detector found, in one numbering, with the held ones
marked on the tile and in the shot table. The first form removed them from
the list. That made a later tracker's numbers a different list from an
earlier one's on the same frame, made its looks incomplete (a reader took
"whoever is left" for a finding), and hid a subject from her own tracker
when an earlier one had wrongly taken her (the same day's three-tracker
run). Judging did not move: the tracker compares and takes among the people
it did before. A typed correction or a seed MAY name a held person, since
one of the two trackers is then wrong and a person is saying which; the
report and the table say so. With `others` wired the "only person" rule is
off altogether: in a job with several named people a lone person may be any
of them. When a track moves onto a held person in the middle of a shot
(`frames_on_others`), those frames are emptied before `regain` runs so they
count as lost and the re-find search can run on them, while the report and the
shot table record what was cut.

**The tracker's own score** (`ScoreTap`; 2026-10-10). ComfyUI's tracker
computes, per frame, a score that its object is there, and its node returns
one number for the whole object. The score was seen falling before a loss
and was neither read nor saved
(`docs/research/postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`,
item 4). Here the tracker object's step is wrapped for the length of one
track call, and each frame's score goes into the shot table under its shot.
Nothing reads it yet: whether it warns of a loss is a test still to run. A
record that could be out of step with its frames is not written at all, and
the report says why.

Nothing here patches core. It calls core's own nodes (`SAM3_Detect`,
`SAM3_VideoTrack`, `SAM3_TrackToMask`) and the SAM 3 model's vision trunk;
the score is read by wrapping one method on the loaded tracker object and
putting it back.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

from . import shot_table, subject_tracks

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
#: so a shorter run may hold no frame to look at. A run holding an owner's box
#: is searched whatever its length, on the box's frame first (2026-10-23).
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
#: The default of `track_head`: whether `MiniMaxH3SubjectTrack` also propagates
#: the subject's head with SAM 3.1 across each shot (`head_mask` output).
#: Measured on the kitchen lead 751-785 mask test (2026-10-19) and decided by the
#: owner (C198, C199, `docs/wiki/decisions.md`, 2026-10-19): replaces `False`;
#: tested on one shot so far.
TRACK_HEAD = True
#: The default of `max_people`: the most detections of the phrase taken from a
#: frame. Reasoned: more than a stage usually shows, and each one costs a mask
#: refinement only on the few frames that are probed.
MAX_PEOPLE = 16
#: Width of a preview tile in pixels. Reasoned: wide enough to read the label.
TILE_WIDTH = 768
#: The side SAM 3's vision trunk takes. Inherited: core's detect node scales
#: every frame to it (`comfy_extras/nodes_sam3.py::SAM3_Detect.execute`).
TRUNK_SIDE = 1008

#: A detection with more than this share of its pixels on the people other trackers hold is not a candidate (the
#: `others` input). measured, one clip (2026-10-10, the day's captures under `data/`): where two Subject Track nodes
#: held one person, the share of one mask on the other was 1.000 on every frame of two loads; where they held two
#: different people through a two-person scene, heads close and one passing behind the other, it never passed 0.039.
#: Half sits in that gap.
ON_OTHERS = 0.5

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
        # Distance from the centre in the frame's own proportions: both axes in units of the frame's width. Until
        # 2026-10-07 each axis ran -0.5 to 0.5, which measures a 16:9 frame as if it were square and counts a step
        # up or down about 1.8 times a step sideways. On the two frames where the subject's detection is known
        # this changes neither pick (`decisions.md`, that date), so it is a correction of the measure, not a
        # fix of a known miss. Measuring from the head and shoulders was tried on the same frames and was worse.
        ys = (torch.arange(h, dtype=torch.float32).view(1, h, 1) / max(h - 1, 1) - 0.5) * (h / w)
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
    gap = auto_gap(scores, floor)
    return float(floor) if gap is None else (gap[0] + gap[1]) / 2


def auto_gap(scores: list[float], floor: float) -> tuple[float, float] | None:
    """The two scores `auto_match` puts its line between, (the lower, the higher); None when the floor stands."""
    ordered = sorted((float(s) for s in scores if s >= 0), reverse=True)
    widest, ends = 0.0, None
    for high, low in zip(ordered, ordered[1:]):
        if high >= floor and high - low >= MIN_GAP and high - low > widest:
            widest, ends = high - low, (low, high)
    return ends if ends is not None and (ends[0] + ends[1]) / 2 > float(floor) else None


@dataclass
class Shot:
    """What `follow` did with one shot, for the report and the preview."""
    start: int
    end: int
    probe: int                   # the frame the shot is first judged on
    seed: int | None = None      # the frame the subject was taken on; None when absent
    best: float = -1.0           # the best similarity seen
    first: float = -1.0          # the similarity on the shot's first look, which is what the automatic line is set from
    shown: int = 0               # the frame the preview shows: the seed, or where the best was seen
    index: int | None = None     # which detection on `shown` is the subject, or the best one when absent
    candidates: int = 0          # detections on `shown`
    picked: bool = False         # the shot the pick was made in
    lone: bool = False           # taken under the line, as the only person on its frame
    # a frame where one person was left only because another tracker's people were set aside: not taken as "the only
    # person", whatever it scored there. None when that did not happen in this shot
    also_held: bool = False       # a correction or a seed named a person an earlier tracker holds
    width: int = 0               # columns the mask on `shown` covers
    seen: int = 0                # the most detections on any frame looked at
    corrected: str = ""          # what a correction said of this shot: `person 2`, `none`, or nothing
    # what a seed said of this shot, or None: {"frame", "box" (x, y, w, h) or None for nobody, "person" it matched on
    # that frame or None, "overlap" of the two boxes, "refused" (bool), "nearest_box" of the best detection or None,
    # and since 2026-10-23 "how" it was taken (`shot_table.SEED_HOW`), "px" of the seed mask, "why" a refusal}
    seeded: dict | None = None
    # a shot seeded from several boxes (2026-10-23): every seed's record as `seeded` holds one, in frame order; a taken one
    # also carries `first_frame` and `last_frame` (the frames its track ran over) and `kept_on` ([first, last] runs where
    # its track was the one kept). None for one seed or none; `seeded` is then the first taken seed's record
    seeded_all: list[dict] | None = None
    # frames of a shot seeded from several boxes left empty because two boxes' tracks held different people there
    # (`merge_tracks`); empty otherwise
    disagreed: list[int] = field(default_factory=list)
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
    floor: float = 0.0               # the floor the automatic line started from; the line is above it only by a gap
    gap: tuple[float, float] | None = None   # the two first-look scores an automatic line was put between, or None
    # a seed on the picked shot: "is the pick" when the detection it names became the reference every other shot is
    # judged against; "took nobody" when it was `none` or refused and the pick rule's own choice stayed the reference;
    # "names no detection" when its boxes were taken only by segmenting inside them (2026-10-23), the rule's choice
    # then staying the reference too; "body boxes only" when its only boxes are round her body, which are never a reference
    pick_seed: str = ""
    rule_pick: tuple[int, int] | None = None   # (frame, detection) the pick rule chose, when a seed replaced it
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
    # every frame a shot was judged on after a cut, and the pick frame: {frame: one entry per detection there, in the
    # detector's order: (its signature per place compared, its likeness to the subject per place or None, the lowest
    # of those or None, its whole-mask signature or None)}. On the pick frame the likeness is None: that frame is what
    # the others are compared with.
    judged: dict[int, list[tuple]] = field(default_factory=dict)
    gallery_whole: list = field(default_factory=list)   # the gallery frames' whole-mask signatures, or None each
    on_others: list[int] = field(default_factory=list)  # frames cut because the track lay mostly on `others`
    head_pieces: dict[int, torch.Tensor] = field(default_factory=dict)  # {first frame: [n, H, W]} head masks when `track_head=True`
    # where each shot's head came from, a frame at a time: "stretch" (tracked inside the stretch the person track holds her),
    # "span" (main's head track of the span, where the stretch found none), "span, top third" (main's stand-in: no head
    # found in the span) (`_track_head_span`)
    head_from: dict[int, dict[int, str]] = field(default_factory=dict)
    head_mode: str = "span"                                  # "span" (main's, the default) or "per stretch" (`parse_head_mode`)


#: A seed's box and a detection's are the same person at or over this overlap (intersection over union). Reasoned,
#: one clip's worth of nothing: on one frame the same person's two boxes are nearly one box, and the line the
#: corrections tool reads a tile by is the same (`bench/who_is_who_across_shots.py::settle_tables`). A STARTING VALUE
#: under the owner's rule of 2026-10-10: where a join kept a tighter mask than the box a seed was written from, a
#: refusal's message gives the nearest box and its overlap, which is the evidence for moving it.
SEED_AGREES = 0.9
#: An owner's box no detection's box agrees with (`SEED_AGREES`) takes the detection whose MASK overlaps the box (as a
#: region, intersection over union) at least this much, among the detections no other tracker holds (`name_seed`).
#: Reasoned, not measured: a person's mask fills about half of its own tight box, so with the mask inside a box drawn by
#: hand the overlap is that half over how much larger the drawn box is, and 0.3 takes a drawn box up to two thirds again
#: the area of the detection's own. It errs strict on purpose: a box it does not take goes to the box-prompted
#: segmentation, which takes what is inside the box anyway, while a box taken too loosely would name somebody else.
SEED_OVERLAPS = 0.3
#: "Mostly", inherited from `head_of` (a head mostly inside a person is theirs): a second detection that passes
#: `SEED_OVERLAPS` is a part of the first when at least this share of its mask lies on the first's, and another person
#: otherwise, which refuses the box as holding two; and a box-prompted segmentation lying more than this share on
#: people other trackers hold is refused as theirs.
SEED_MOSTLY = 0.5
#: How a seed's box was taken (`name_seed`), in the order tried; the words are the record's `how`. One copy, in
#: `shot_table.py`, which words the table's rows by them.
SEED_BY_BOX, SEED_BY_OVERLAP, SEED_BY_SEGMENT, SEED_REFUSED, SEED_BY_INSIDE, SEED_BY_PROMPT = shot_table.SEED_HOW
#: The word a seed line carries when its box was DRAWN by the owner on a sheet (`bench/cast_file.py box`): approximate,
#: "just to give you a rough idea" (the owner, 2026-10-23). It says which person is whom, never an outline, so it is
#: named by `name_drawn`, not by the agreement of boxes.
SEED_DRAWN = "drawn"
#: A drawn box names the detected person at least this share of whose mask lies inside it. "Mostly", inherited from
#: `head_of` (a head mostly inside a person is theirs). Reasoned, not measured: a rough box drawn round one person holds
#: most of what the detector found of them, however loosely drawn, and a person it was not drawn round has most of
#: their mask outside it unless the box straddles them, which `name_drawn` refuses.
DRAWN_INSIDE = 0.5
#: What SAM 3's box prompt segments from a box is taken only as ONE PERSON the detector found too (`prompt_seed`): it
#: must lie mostly (`SEED_MOSTLY`) on one detection no other tracked subject is, and no more than this share of it on
#: any other detection or on another tracked subject's track. Reasoned, not measured: two people's right masks share
#: little more than the pixels along the edge where one passes the other, and a segmentation that took another person
#: (the lead's dress under 4647's boxes on 4674-4677), two people as one blob, or a thing nobody detected as a person
#: (the blind read of 2026-10-23: a chair) is refused and recorded as doubtful.
PROMPT_ON_OTHERS = 0.25
#: A person mask is HEAD-SIZED, and seeds the head track itself in a stretch with no head detected, when it has at
#: most this many times the pixels of her own head on the nearest frame of the span where `head_of` found one
#: (`_head_seed`). Reasoned from one shot, a STARTING VALUE: on `fun`'s preview shot_c_1419d (2026-10-22, the second
#: subject's track and head track videos) only her head peeks out on 1431 and her person mask there is 1,895 px with no
#: head found in it (1.6 times her head track on 1455-1465, which holds 1,166-1,205 px a frame), and her person mask on
#: those frames, where a head was found, holds 2,905-4,892 px (2.5 to 4.2 times her head there). Two sits between.
HEAD_SIZED = 2.0
#: The word a seed line ends in when its box encloses the subject's BODY and not her head (2026-10-23, the lead's
#: reading of the owner's sheets: on 4647 and 4912 some boxes sit on her arm or at the table where her head is hidden).
#: Such a box seeds her person track only, never her head track (`_seed`, `_track_head_span`).
SEED_BODY = "body"
#: The word a line ends in when its box is ANOTHER SUBJECT'S on that frame (`bench/cast_file.py seeds` writes the other
#: subjects' boxes on each frame this subject is boxed on): not a seed, but who this tracker must not take there, so the
#: first tracker in a job's order, which is given no `others`, knows them too (2026-10-23, the blind read).
SEED_OTHER = "other"
#: The line the seeds text carries when the job has other subjects (`bench/shot_to_load.py preview` writes it): this tracker
#: then refuses a box prompt, as doubtful, on a frame where it knows none of them, having no earlier tracker's mask and no
#: other subject's box there (2026-10-23, the blind re-read: the first tracker of a job took the lead's dress).
_JOB_OTHERS = re.compile(r"^job\s*:\s*(\d+)\s+other\s+subjects?$", re.IGNORECASE)


#: The line that switches a preview's head track to PER STRETCH (`bench/shot_to_load.py preview --head-per-stretch WHY`):
#: the head seeded and tracked inside each stretch the person track holds her, with main's span head where a stretch gives
#: none. Without it the head track is main's span head track, exactly as main (the owner, 2026-10-23: an opt-in, so no saved
#: or rendered shot can lose a head).
HEAD_PER_STRETCH = "per stretch"
_HEAD_MODE = re.compile(r"^head\s*:\s*per\s+stretch(?:\s*:\s*(.+))?$", re.IGNORECASE)


def parse_head_mode(text: str) -> tuple[str, str]:
    """Which head track the seeds text asks for: ("per stretch", the why) where it carries `head: per stretch: WHY`, else
    ("span", "")."""
    for piece in re.split(r"[\n;]", str(text or "")):
        found = _HEAD_MODE.match(piece.strip())
        if found is not None:
            return HEAD_PER_STRETCH, " ".join(str(found[1] or "").split())
    return "span", ""


def parse_job_others(text: str) -> int:
    """How many other subjects the seeds text says the job has (`job: 2 other subjects`), 0 where it says none."""
    for piece in re.split(r"[\n;]", str(text or "")):
        found = _JOB_OTHERS.match(piece.strip())
        if found is not None:
            return int(found[1])
    return 0
_SEED = re.compile(r"^frame\s*(\d+)\s*[:=]\s*(?:(none)|(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
                   r"(?:\s+(drawn))?(?:\s+(body))?(?:\s+(other))?)$", re.IGNORECASE)


def parse_seeds(text: str, shots: list[tuple[int, int]]) -> dict[int, list[tuple[int, tuple | None]]]:
    """`frame 410: 543, 253, 157, 186`, `frame 412: 540, 300, 60, 90 body` and `frame 439: none`, one per line (or `;`).

    Returns {shot number from 1: [(frame, (x, y, width, height) or None for nobody), ...] in frame order}; a box
    marked `drawn` (the owner drew it, approximate: `SEED_DRAWN`) carries "drawn" after its box, and one marked `body`
    (it encloses her body, not her head: `SEED_BODY`) "body", in that order: (frame, box, "drawn", "body"). The frame
    is the load's own and picks the shot; `shots` is the load's (first frame, one past the last) per shot. A shot may
    hold a box on each of several frames (2026-10-23: every box the owner drew is a seed on its frame; until then two
    seeds in one shot were refused and the cast file passed one); `_seed` tracks from each and keeps, frame by frame,
    the track the tracker's own score holds best (`merge_tracks`). A line that is not a seed, a frame the load does
    not have, a box with no area, two seeds on one frame, and `none` beside another seed of its shot (two words for one
    shot) are refused by name. A line ending in `other` is another subject's box (`SEED_OTHER`), read by
    `parse_other_boxes` and passed over here.
    """
    out: dict[int, list[tuple[int, tuple | None]]] = {}
    for piece in re.split(r"[\n;]", str(text or "")):
        piece = piece.strip()
        if not piece or _JOB_OTHERS.match(piece) or _HEAD_MODE.match(piece):
            continue
        found = _SEED.match(piece)
        if found is None:
            raise ValueError(f"seeds: cannot read `{piece}`; write `frame 410: 543, 253, 157, 186` (x, y, width, height), "
                             "the same ending in `drawn` for a box drawn by hand and then `body` for a box round her body and not her "
                             "head, or `frame 410: none`, one per line")
        frame = int(found[1])
        number = next((n for n, (start, end) in enumerate(shots, 1) if start <= frame < end), None)
        if number is None:
            raise ValueError(f"seeds: `{piece}` names frame {frame}, and the load has frames 0 to {shots[-1][1] - 1 if shots else -1}")
        box = None if found[2] else tuple(float(found[k]) for k in (3, 4, 5, 6))
        if box is not None and (box[2] <= 0 or box[3] <= 0):
            raise ValueError(f"seeds: `{piece}` gives a box with no area")
        if found[9]:
            continue                      # another subject's box: `parse_other_boxes`
        given = out.setdefault(number, [])
        span = f"shot {number} (frames {shots[number - 1][0]}-{shots[number - 1][1] - 1})"
        if any(seed[0] == frame for seed in given):
            raise ValueError(f"seeds: {span} is seeded twice on frame {frame}: one box a frame")
        if given and (box is None or any(seed[1] is None for seed in given)):
            raise ValueError(f"seeds: {span} is seeded with `none` beside another seed, on frames "
                             f"{', '.join(str(seed[0]) for seed in given)} and {frame}: two words for one shot")
        given.append((frame, box) + ((SEED_DRAWN,) if found[7] else ()) + ((SEED_BODY,) if found[8] else ()))
    return {number: sorted(given, key=lambda seed: seed[0]) for number, given in out.items()}


def parse_other_boxes(text: str, shots: list[tuple[int, int]]) -> dict[int, list[tuple[tuple, bool]]]:
    """The other subjects' boxes the seeds text carries (`... other`, `SEED_OTHER`): {frame: [(box, drawn)]}. Read after
    `parse_seeds` has refused whatever it refuses in the same text."""
    out: dict[int, list[tuple[tuple, bool]]] = {}
    parse_seeds(text, shots)
    for piece in re.split(r"[\n;]", str(text or "")):
        found = _SEED.match(piece.strip())
        if found is not None and found[9]:
            out.setdefault(int(found[1]), []).append((tuple(float(found[k]) for k in (3, 4, 5, 6)), bool(found[7])))
    return out


def others_named(detect, other_boxes: dict[int, list[tuple[tuple, bool]]], held=None, in_box=None):
    """`held` widened by the detections the other subjects' boxes name on each frame (`parse_other_boxes`): a callable,
    frame -> the indices another tracked subject is there, carrying `held`'s joined mask as `.others`. Each box is named
    THE SAME WAY ITS OWN SUBJECT'S TRACKER NAMES IT (`name_seed`: a tile's box by agreement, then by its mask's overlap,
    then by the box prompt; a drawn one by `name_drawn`), and the detection it takes, or the one its box prompt matched,
    is another subject's here (2026-10-23, the independent review: a box their tracker took by overlap or by the prompt
    was missed). Used where a seed is named, so a tracker's seed is held to every other subject the owner boxed, not only
    to the trackers that ran before it (the first in a job's order is given no `others`; the blind read)."""
    seen: dict[int, frozenset] = {}

    def off(f: int) -> frozenset:
        f = int(f)
        if f not in seen:
            named = set()
            for box, drawn in other_boxes.get(f, []):
                got = name_seed(detect, f, box, None, in_box, drawn=drawn)
                k = got.index if got.mask is not None and got.index is not None else (got.record.get("prompt") or {}).get("index")
                if got.mask is not None and k is not None:
                    named.add(int(k))
            seen[f] = frozenset(named) | (held(f) if held is not None else frozenset())
        return seen[f]

    off.others = getattr(held, "others", None)
    return off


def seed_list(given) -> list[tuple[int, tuple | None]]:
    """A shot's seeds as `parse_seeds` gives them, a list in frame order; a lone (frame, box), the form before
    2026-10-23, is read as a list of one."""
    return sorted(given if isinstance(given, list) else [given], key=lambda seed: int(seed[0]))


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


def _free(masks: torch.Tensor, held: frozenset) -> list[int]:
    """The detections of a frame that no other tracker holds, as indices into that frame's list.

    The one road to a candidate for every choice THE TRACKER makes: the pick (by rule, on a named frame, or from a
    handed gallery), a shot's best match, a re-find, and the people the scores are taken relative to. A PERSON'S
    WORD goes round it, in two places and on purpose: a typed correction (`_correct`) and a seed (`seed_names`)
    choose among everybody, and may name a held person."""
    return [k for k in range(int(masks.shape[0])) if k not in held]


def choose_free(masks: torch.Tensor, scores: list[float], pick: str, held: frozenset) -> int | None:
    """`choose` among the detections no other tracker holds; the index is into the frame's whole list."""
    free = _free(masks, held)
    if not free:
        return None
    i = choose(masks[torch.tensor(free, dtype=torch.long)], [scores[k] for k in free if k < len(scores)], pick)
    return None if i is None else free[i]


def main_subject(shots: list[Shot], pick: str, detect, sign,
                 passed_over: list[tuple[int, int]] | None = None, held=None) -> tuple[int, int] | None:
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
        off = held(shot.probe) if held is not None else frozenset()
        i = choose_free(masks, scores, pick, off)
        if i is not None:
            sig = sign(shot.probe, masks[i])
            winners.append((shot, i, _views(sig)[0], len(_free(masks, off))))
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


def filled_runs(piece: torch.Tensor) -> list[tuple[int, int]]:
    """The runs of frames a tracked piece [n, H, W] holds somebody on, as (first, one past the last) in its own numbers."""
    on = (piece > 0.5).flatten(1).any(dim=1).tolist()
    runs, first = [], None
    for i, v in enumerate(on + [False]):
        if v and first is None:
            first = i
        elif not v and first is not None:
            runs.append((first, i))
            first = None
    return runs


def _scored(track, start: int, end: int, seed: int, mask: torch.Tensor, record: bool = False):
    """One track call and the tracker's own score per frame of it: (masks [end - start], [score or None], [why none]).

    `track.scored` gives the scores without writing them into the node's record (`_sam_callables`); a stand-in with no
    such attribute is called as it is, and has no score to give. `record` False takes the head tracker's call, which
    never writes the record (`track.without_scores`)."""
    scored = getattr(track, "scored", None)
    if scored is not None:
        return scored(int(start), int(end), int(seed), mask)
    plain = track if record else getattr(track, "without_scores", track)
    return plain(int(start), int(end), int(seed), mask), [None] * (int(end) - int(start)), []


#: When several boxes' tracks hold somebody on one frame, a switch from the box kept on the frame before to another
#: is made only where the other's mask overlaps the mask kept on the frame before at least this much (intersection over
#: union). Reasoned, not measured: at 24 frames a second a person moves little between frames, so a track still on her
#: holds most of the pixels it held a frame earlier, and a jump under half is to another region or another person (the
#: blind read of 2026-10-23: with scores alternating frame to frame the merge flickered between two people).
MERGE_CONTINUOUS = 0.5


def same_person(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two [H, W] masks of one frame are one person: the smaller lies mostly (`SEED_MOSTLY`) on the larger, so a
    track of her head and a track of her whole body are one, and two people side by side are two."""
    on_a, on_b = a > 0.5, b > 0.5
    small = min(int(on_a.sum()), int(on_b.sum()))
    return small > 0 and float((on_a & on_b).sum()) >= SEED_MOSTLY * small


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    on_a, on_b = a > 0.5, b > 0.5
    union = int((on_a | on_b).sum())
    return float((on_a & on_b).sum()) / union if union else 0.0


def _one_detection_holds(detect, f: int, masks: list[torch.Tensor]) -> bool:
    """Whether one person detection of frame `f` holds every mask mostly (`SEED_MOSTLY`): parts of one person, such as
    her arm tracked from a box round her body and her face from a box round her head."""
    if detect is None:
        return False
    people, _ = detect(int(f))
    return any(all(float(((m > 0.5) & (p > 0.5)).sum()) >= SEED_MOSTLY * max(float((m > 0.5).sum()), 1.0) for m in masks) for p in people)


def merge_tracks(start: int, end: int, runs: list[tuple], detect=None) -> tuple[torch.Tensor, list, list, list]:
    """Several seeds' tracks of `[start, end)` as one, frame by frame (2026-10-23, the owner's go: every owner box is a seed).

    `runs` is (seed frame, first, one past the last, masks [last - first], score per frame or None) per seed. On a seed's
    own frame its own track is kept: the box there is the owner's word. Elsewhere, of the tracks that hold somebody on
    the frame:
    - when two of them do not overlap as one person (`same_person`) and no person detection of the frame (`detect`,
      the person tracker's; None for heads) holds them both, they hold DIFFERENT PEOPLE: the frame is left empty and
      listed, since one track has moved onto somebody else and nothing here can say which (the blind read of
      2026-10-23). Held by one detection, they are parts of one person (her arm from a body box, her face from a head
      box) and the frame keeps both;
    - else the one whose tracker's own score there is highest (over 0 it takes the object as present, `ScoreTap`); a
      track whose score could not be read ranks under any that was, and a tie, or no score at all, goes to the seed
      nearer the frame, then the earlier; but a switch away from the seed kept on the frame before, while that seed still
      holds her, is made only when the new mask overlaps the one kept before at `MERGE_CONTINUOUS` or more.
    A frame no track holds anybody on stays empty, with the highest score any track gave there. Returns (masks
    [end - start], the score kept per frame, the seed frame kept per frame or None, the frames left empty because two
    tracks held different people)."""
    some = runs[0][3]
    out = torch.zeros((int(end) - int(start),) + tuple(some.shape[1:]), dtype=some.dtype)
    scores, kept, disagreed = [None] * (int(end) - int(start)), [None] * (int(end) - int(start)), []
    before = None                                  # (seed, mask) kept on the frame before, None after an empty frame
    for f in range(int(start), int(end)):
        here = [(int(seed), masks[f - lo], sc[f - lo]) for seed, lo, hi, masks, sc in runs if lo <= f < hi]
        own = [h for h in here if h[0] == f]
        holding = [h for h in here if bool((h[1] > 0.5).any())]
        pick = own[0] if own else None
        if pick is None and holding:
            if any(not same_person(x[1], y[1]) for i, x in enumerate(holding) for y in holding[i + 1:]):
                if _one_detection_holds(detect, f, [h[1] for h in holding]):
                    best = max(holding, key=lambda h: (h[2] is not None, h[2] if h[2] is not None else 0.0, -abs(f - h[0]), -h[0]))
                    joined = torch.stack([h[1] for h in holding]).amax(dim=0)
                    pick = (best[0], joined, best[2])
                else:
                    disagreed.append(f)
            else:
                pick = max(holding, key=lambda h: (h[2] is not None, h[2] if h[2] is not None else 0.0, -abs(f - h[0]), -h[0]))
                stay = next((h for h in holding if before is not None and h[0] == before[0]), None)
                if stay is not None and pick[0] != stay[0] and _iou(pick[1], before[1]) < MERGE_CONTINUOUS:
                    pick = stay
        if pick is not None:
            out[f - start], scores[f - start], kept[f - start] = pick[1], pick[2], pick[0]
            before = (pick[0], pick[1])
        else:
            known = [h[2] for h in here if h[2] is not None]
            scores[f - start] = max(known) if known else None
            before = None
    return out, scores, kept, disagreed


def _head_seed(head_fn, piece: torch.Tensor, shot: Shot, frame: int, person: torch.Tensor, lo: int, hi: int,
               span: tuple[int, int], fallback: bool = True) -> tuple[int, torch.Tensor] | None:
    """Where the head track of one stretch `[lo, hi)` is seeded, and from what: (frame, mask), or None.

    In order: the head `head_of` finds inside `person` on `frame`; the nearest of up to `GALLERY_MOST` frames spread over
    THE SAME STRETCH that has one; `person` itself where it is head-sized (`HEAD_SIZED`, against her own head on the
    nearest frame of `span` where one is found and not occluded smaller: sized there, never seeded there). None where
    neither is: the frames are then filled from main's span head track, top third and all (`_span_head`, `_track_head_span`).
    Never a frame outside the stretch: on 1419d the head was seeded across an empty run and lost back across it (2026-10-23).
    With the stretch the whole span this is what the head track did before (2026-10-18), call for call: the head-sized
    step looks only on frames the stretch's search did not, which there are none; the top third main took here comes
    from main's span head track, from the same seed and over the same span (`_span_head`).
    `fallback` False (a stretch the owner boxed only round her body, `SEED_BODY`) takes a head SAM finds and nothing
    else: her arm at the table is neither head-sized nor has a head in its top third (2026-10-23, the lead's reading)."""
    h_frame, h_seed = int(frame), head_fn(int(frame), person)
    tried = set()          # frames whose person track was asked for a head already, and had none
    if h_seed is None or not bool((h_seed > 0.5).any()):
        stretch = piece[lo - shot.start:hi - shot.start]
        for k in sorted(gallery_frames(stretch), key=lambda k: abs((lo + k) - h_frame)):
            f = lo + k
            m = head_fn(f, (piece[f - shot.start] > 0.5).to(torch.float32))
            if m is not None and bool((m > 0.5).any()):
                h_frame, h_seed = f, m
                break
            tried.add(f)
    if (h_seed is None or not bool((h_seed > 0.5).any())) and not fallback:
        return None
    if h_seed is None or not bool((h_seed > 0.5).any()):
        h_frame, h_seed = int(frame), None
        area = int((person > 0.5).sum())
        whole = piece[span[0] - shot.start:span[1] - shot.start]
        for k in sorted(gallery_frames(whole), key=lambda k: abs((span[0] + k) - h_frame)):
            f = span[0] + k
            if f in tried:
                continue
            m = head_fn(f, (piece[f - shot.start] > 0.5).to(torch.float32))
            if m is not None and bool((m > 0.5).any()):
                if area and area <= HEAD_SIZED * int((m > 0.5).sum()):
                    h_seed = (person > 0.5).to(torch.float32)
                    break
    if h_seed is None or not bool((h_seed > 0.5).any()):
        return None
    p_at_seed = person if h_frame == int(frame) else piece[h_frame - shot.start]
    clipped = ((h_seed > 0.5) & (p_at_seed > 0.5)).to(torch.float32)
    if not bool((clipped > 0.5).any()):
        clipped = (h_seed > 0.5).to(torch.float32)
    return h_frame, clipped


def _track_head_span(result: Followed, shot: Shot, start: int, end: int, seed_frame: int, seed_person: torch.Tensor,
                     sign, track, seeds: list[tuple[int, torch.Tensor]] | None = None, bodies: list[int] | None = None,
                     anchored: bool = False, body_lead: bool = False) -> None:
    """The subject's head over `[start, end)`, tracked WITHIN EACH STRETCH THE PERSON TRACK HOLDS HER, clipped to it.

    Each run of frames `result.pieces[shot.start]` holds somebody on inside the span is seeded inside itself
    (`_head_seed`) and tracked over itself alone, so the head is never seeded on the far side of an empty run and
    followed back across it (2026-10-23: on 1419d, seeded at 1431 where only her head peeks out and no head was found,
    it was seeded on 1454, past seventeen empty frames, and held nothing on 1428-1436). A run is seeded from
    `seed_person` on `seed_frame` when that frame is in it, else from the person track on its frame nearest
    `seed_frame`. `seeds`, (frame, mask) of every owner box a shot's track was seeded from (`_seed`), seed a run they
    lie in instead; a run holding two or more is tracked from each up to its neighbours' frames and kept frame by frame
    by the tracker's own score (`merge_tracks`). `bodies`, the frames of owner boxes round her BODY (`SEED_BODY`), never
    seed a head: they are not in `seeds`, and a run whose only owner boxes are body boxes takes its head only from a
    head SAM finds inside her track there, with no head-sized or top-third stand-in (`_head_seed`, `fallback`). The
    stand-ins run only where an owner's box that is not a body box anchors the stretch (`seeds`, or `seed_frame` with
    `anchored`, which `_seed` passes for its one box): never from a body box, a stretch holding no box, a tracker's own
    pick or a subject found again (2026-10-23, the blind read: a stretch holding only her arm made the arm her head). A
    frame she is held on that its stretch gives no head takes MAIN'S SPAN HEAD there (`_span_head`: seeded on the span's
    seed frame or its nearest gallery frame, else the top third of her mask, never from a box round her body, and
    tracked across empty runs as main did), so the head never covers less than main's did (2026-10-23, the lead on the
    reader's census); `result.head_from` says which a frame took. A span the person track holds on every frame, with
    one box, is tracked exactly as it was before this: one call over the span, seeded on the same frame from the same
    mask, and no second one."""
    piece = result.pieces.get(shot.start)
    if piece is None:
        return
    head_fn = getattr(sign, "head", None)
    if head_fn is None:
        return
    h_piece = result.head_pieces.get(shot.start)
    if h_piece is None:
        h_piece = torch.zeros_like(piece)
        result.head_pieces[shot.start] = h_piece
    span_piece = piece[start - shot.start:end - shot.start]
    h_piece[start - shot.start:end - shot.start] = 0.0
    marks = result.head_from.setdefault(shot.start, {})
    for f in range(int(start), int(end)):
        marks.pop(f, None)
    if result.head_mode != HEAD_PER_STRETCH:
        # THE DEFAULT: main's span head track, exactly as main (the owner, 2026-10-23), from the shot's first head box or its
        # own seed; never a top third from a box round her body
        span, third = _span_head(head_fn, piece, shot, start, end, seed_frame, seed_person, track, stand_in=not body_lead)
        if span is not None:
            h_piece[start - shot.start:end - shot.start] = span.to(h_piece.dtype)
            for k in range(end - start):
                if bool(span[k].any()):
                    marks[start + k] = "span, top third" if third else "span"
        return
    owner = sorted(((int(f), m, True) for f, m in (seeds or [])), key=lambda s: s[0])
    body = sorted({int(f) for f in (bodies or [])})
    stretches = filled_runs(span_piece)
    # one stretch over the whole span searched from the span's own seed alone: its search, and its track where it found a
    # head, are main's own; the span pass below then makes neither again
    same_as_main = False
    seeded = False
    for ra, rb in stretches:
        a, b = start + ra, start + rb
        inside = [(f, m, True) for f, m, _ in owner if a <= f < b]
        if not inside:
            boxed_body = [f for f in body if a <= f < b]
            if boxed_body:
                # boxed round her body only: a head SAM finds in her track, from the box nearest the seed, or none
                f = min(boxed_body, key=lambda g: (abs(g - int(seed_frame)), g))
                inside = [(f, (piece[f - shot.start] > 0.5).to(torch.float32), False)]
            elif a <= int(seed_frame) < b:
                # the stand-ins run only where an owner's box round her head (or a tile's box) anchors the stretch
                inside = [(int(seed_frame), seed_person, bool(anchored))]
            else:
                f = min(max(int(seed_frame), a), b - 1)
                inside = [(f, (piece[f - shot.start] > 0.5).to(torch.float32), False)]   # a stretch with no box
        if (ra, rb) == (0, end - start) and len(inside) == 1 and inside[0][0] == int(seed_frame) and inside[0][1] is seed_person:
            same_as_main = True
        # each seed looks for its head between its neighbours' frames, so two seeds' heads are two seeds
        heads = []
        for k, (f, m, fallback) in enumerate(inside):
            lo = inside[k - 1][0] if k else a
            hi = inside[k + 1][0] + 1 if k + 1 < len(inside) else b
            got = _head_seed(head_fn, piece, shot, f, m, lo, hi, (start, end), fallback=fallback)
            if got is not None and all(got[0] != h[0] for h in heads):
                heads.append(got)
        if not heads:
            continue
        seeded = True
        if len(heads) == 1:
            masks = _scored(track, a, b, heads[0][0], heads[0][1])[0]
        else:
            heads.sort(key=lambda h: h[0])
            runs = []
            for k, (f, m) in enumerate(heads):
                lo = heads[k - 1][0] if k else a
                hi = heads[k + 1][0] + 1 if k + 1 < len(heads) else b
                got, scores, _ = _scored(track, lo, hi, f, m)
                runs.append((f, lo, hi, got, scores))
            masks = merge_tracks(a, b, runs)[0]           # a frame two heads disagree on is left headless
        h_piece[a - shot.start:b - shot.start] = ((masks > 0.5) & (span_piece[ra:rb] > 0.5)).to(torch.float32)
    # NEVER LESS THAN MAIN (2026-10-23, the lead on the reader's census: 835 frames of 38 saved shots where main held a head
    # that a stretch alone would not; the owner's rule, never leave originals): where a frame she is held on has no head from
    # its stretch, main's span head track fills it, recorded per frame (`head_from`: "stretch", "span" or "span, top third").
    # Run only where such a frame is, and not where the one stretch was main's own call, so a whole, headed span costs nothing
    k0 = start - shot.start
    held_on = [bool((span_piece[k] > 0.5).any()) for k in range(end - start)]
    stretch_on = [bool((h_piece[k0 + k] > 0.5).any()) for k in range(end - start)]
    for k in range(end - start):
        if stretch_on[k]:
            marks[start + k] = "stretch"
    if any(h and not s_ for h, s_ in zip(held_on, stretch_on)) and not (same_as_main and seeded):
        span, third = _span_head(head_fn, piece, shot, start, end, seed_frame, seed_person, track, stand_in=not body_lead,
                                 searched=same_as_main)
        if span is not None:
            for k in range(end - start):
                if held_on[k] and not stretch_on[k] and bool(span[k].any()):
                    h_piece[k0 + k] = span[k].to(h_piece.dtype)
                    marks[start + k] = "span, top third" if third else "span"


def _span_head(head_fn, piece: torch.Tensor, shot: Shot, start: int, end: int, seed_frame: int, seed_person: torch.Tensor,
               track, stand_in: bool, searched: bool = False) -> tuple[torch.Tensor | None, bool]:
    """MAIN'S HEAD TRACK of a span, as it was before 2026-10-23, call for call: the head `head_of` finds inside
    `seed_person` on `seed_frame`, else on the nearest of up to `GALLERY_MOST` frames spread over the whole span, else
    the top third of `seed_person` (`stand_in`: never from a box round her body), tracked over the span across any
    empty run and cut to the person track. `searched`: the one stretch of the span already made this same search from this
    same seed and found nothing, so it is not made again (main made it once). Returns ([span frames, H, W] bool or None,
    whether the top third seeded it)."""
    span_piece = piece[start - shot.start:end - shot.start]
    if not bool((span_piece > 0.5).any()):
        return None, False
    h_frame, h_seed = int(seed_frame), (None if searched else head_fn(int(seed_frame), seed_person))
    if (h_seed is None or not bool((h_seed > 0.5).any())) and not searched:
        for k in sorted(gallery_frames(span_piece), key=lambda k: abs((start + k) - h_frame)):
            f = start + k
            m = head_fn(f, (piece[f - shot.start] > 0.5).to(torch.float32))
            if m is not None and bool((m > 0.5).any()):
                h_frame, h_seed = f, m
                break
    third = False
    if (h_seed is None or not bool((h_seed > 0.5).any())) and stand_in:
        h_frame, h_seed, third = int(seed_frame), top_third(seed_person), True
    if h_seed is None or not bool((h_seed > 0.5).any()):
        return None, False
    p_at_seed = seed_person if h_frame == int(seed_frame) else piece[h_frame - shot.start]
    clipped = ((h_seed > 0.5) & (p_at_seed > 0.5)).to(torch.float32)
    if not bool((clipped > 0.5).any()):
        clipped = (h_seed > 0.5).to(torch.float32)
    tracked = _scored(track, start, end, h_frame, clipped)[0]
    return (tracked > 0.5) & (span_piece > 0.5), third


def follow(n_frames: int, cuts: list[int], pick: str, pick_frame: int | None, match_threshold: float | None,
           detect: Callable[[int], tuple[torch.Tensor, list[float]]],
           sign: Callable[[int, torch.Tensor], torch.Tensor | None],
           track: Callable[[int, int, int, torch.Tensor], torch.Tensor],
           stride: int = PROBE_STRIDE, offset: int = PROBE_OFFSET,
           corrections: dict[int, int | None] | None = None, gallery: list[tuple] | None = None,
           sign_whole: Callable[[int, torch.Tensor], torch.Tensor | None] | None = None,
           seeds: dict[int, tuple[int, tuple | None]] | None = None,
           held=None, track_head: bool = TRACK_HEAD, other_boxes: dict | None = None, job_others: int = 0,
           head_per_stretch: bool = False) -> Followed:
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

    `seeds` is {shot number from 1: [(frame, box or None), ...]} (`parse_seeds`;
    a lone (frame, box) is read as a list of one): the automatic pass runs as
    it would without them, and a seeded shot is then tracked from what each
    box names on its own frame (`name_seed`), left empty for `None`, or left
    empty and marked refused when no box takes anybody (`_seed`). Several
    boxes in one shot are each tracked from and kept frame by frame by the
    tracker's own score (`merge_tracks`). A shot may not carry both a seed
    and a correction. An empty run holding a box is searched whatever its
    length (`regain`).

    `held` is None, or a callable giving, for a frame, which of `detect`'s
    detections there another tracker holds (`held_people`). A held detection
    is listed and never chosen. With `held` given the rule "the only person
    on the frame is taken whatever it scores" is off: in a job with several
    named people a lone person may be any of them (on 2026-10-10 the third of
    three trackers took another subject that way, and a second took a
    subject small in an inset twice).

    `sign_whole(frame, mask)`, when given, is a person's signature under
    their WHOLE mask. NOTHING HERE IS DECIDED ON IT: it is kept beside
    everything judged and beside the gallery (`Followed.judged`,
    `gallery_whole`). It was measured once, for a reader that settled a shot
    for several trackers' subjects together: on that shot it and the top third
    failed on different frames (`bench/results/2026-10-10_who_is_who_across_shots.md`).
    That reader was cut the same day and nothing reads the signature now.
    """
    corrections = dict(corrections or {})
    # `seeds` is a local name further down (a seed frame's detections), so the argument is held under another
    given_seeds = {int(n): seed_list(v) for n, v in (seeds or {}).items()}
    for number in sorted(set(given_seeds) & {int(n) for n in corrections}):
        raise ValueError(f"shot {number} has both a seed (frame {given_seeds[number][0][0]}) and a typed correction: two words for one shot. "
                         "Remove one")
    in_box = getattr(sign, "in_box", None)
    # who a seed may not take: the people earlier trackers hold, and those the other subjects' boxes name on its frame
    seed_held = others_named(detect, other_boxes, held, in_box) if other_boxes else held

    def blind(f: int) -> str:
        """Why a box prompt cannot be held to the job's other subjects on frame `f`, or "" where it can."""
        if job_others > 0 and held is None and not (other_boxes or {}).get(int(f)):
            return (f"the job has {job_others} other subject(s) and this tracker knows none of them on this frame (no earlier "
                    "tracker's mask, no other subject's box here), so a box prompt cannot be held to them: DOUBTFUL")
        return ""
    shots = [Shot(s, e, probe=min(s + max(int(offset), 0), e - 1)) for s, e in shot_ranges(n_frames, cuts)]
    for shot in shots:
        shot.shown = shot.probe
    for number in corrections:
        if not 1 <= int(number) <= len(shots):
            raise ValueError(f"corrections: shot {number} is named, and the clip has {len(shots)} shot(s)")
    for number, these in given_seeds.items():
        for frame, *_ in these:
            if not 1 <= int(number) <= len(shots) or not shots[int(number) - 1].start <= int(frame) < shots[int(number) - 1].end:
                raise ValueError(f"seeds: frame {frame} is not in shot {number}")
    # tracked from the corrected or seeded frame, below, and not by the automatic pass
    by_hand = {id(shots[int(number) - 1]) for number in list(corrections) + list(given_seeds)}
    # every frame an owner's box stands on, by shot: a run `regain` meets holding one is searched whatever its length
    boxed = {id(shots[int(number) - 1]): [int(f) for f, box, *_ in these if box is not None] for number, these in given_seeds.items()}
    bodied = {id(shots[int(number) - 1]): [int(f) for f, box, *words in these if box is not None and SEED_BODY in words]
              for number, these in given_seeds.items()}
    result = Followed(shots)
    result.head_mode = HEAD_PER_STRETCH if head_per_stretch else "span"
    off = held if held is not None else (lambda f: frozenset())
    given = list(gallery or [])
    result.gallery_given = len(given)
    if pick_frame is None and given:
        entry = None
        probes = [s.probe for s in shots]
        step = max(int(stride), 1)
        for f in probes + [f for f in range(0, int(n_frames), step) if f not in set(probes)]:
            found, _ = detect(f)
            free = _free(found, off(f))
            which, best, second = clear_best(gallery_scores(found[torch.tensor(free, dtype=torch.long)], f, given, sign))
            result.pick_probes.append((int(f), int(found.shape[0]), best, second))
            if which is not None:
                entry = (int(f), free[which])
                break
    elif pick_frame is None:
        entry = main_subject(shots, pick, detect, sign, result.passed_over, held)
    else:
        if not 0 <= int(pick_frame) < int(n_frames):
            raise ValueError(f"pick_frame {pick_frame} is outside the clip's {n_frames} frames")
        masks, scores = detect(int(pick_frame))
        which = choose_free(masks, scores, pick, off(int(pick_frame)))
        entry = None if which is None else (int(pick_frame), which)
    if entry is None:
        _correct(result, corrections, detect, track, held, sign=sign, track_head=track_head)
        _seed(result, given_seeds, detect, track, held, sign=sign, track_head=track_head, in_box=in_box, seed_held=seed_held, blind=blind)
        return result
    frame0, which = entry
    # A SEED ON THE PICKED SHOT IS THE PICK (mrcorn's cold read, 2026-10-10). A tool seeds that shot precisely when
    # the pick rule chose the wrong person; were the seed to change that shot alone, every other shot would still be
    # judged against the rule's person and one mask would follow two people, with nothing said. So the detection
    # the seed names, on the seed's frame, becomes the reference. A seed there that takes nobody (`none`, or
    # refused) cannot be a reference: the rule's pick stays one, and the report says which person that is.
    # Of several boxes on the picked shot (2026-10-23), the first in frame order that names a DETECTION is the pick: a
    # reference is a detection, and one is needed. A shot whose boxes were all taken by the box-prompted segmentation
    # (`SEED_BY_SEGMENT`, `SEED_BY_PROMPT`) names none: the rule's pick stays the reference and the report says so ("names no detection").
    on_pick = next((n for n, s in enumerate(shots, 1) if s.start <= frame0 < s.end), None)
    if on_pick in given_seeds:
        named, segmented = None, False
        for seed_frame, seed_box, *words in given_seeds[on_pick]:
            if seed_box is None:
                continue
            if SEED_BODY in words:
                continue                  # a box round her body alone is never the reference others are judged against
            got = name_seed(detect, seed_frame, seed_box, seed_held, in_box, drawn=SEED_DRAWN in words, blind=blind(seed_frame))
            if got.mask is not None and got.index is not None:
                named = (int(seed_frame), int(got.index))
                break
            segmented = segmented or got.mask is not None
        bodies_only = any(box is not None and SEED_BODY in words for _, box, *words in given_seeds[on_pick])
        if named is not None:
            if named != (int(frame0), int(which)):
                result.rule_pick = (int(frame0), int(which))
            (frame0, which), result.pick_seed = named, "is the pick"
        else:
            result.pick_seed = "names no detection" if segmented else "body boxes only" if bodies_only else "took nobody"
    masks0, _ = detect(frame0)
    picked = masks0[which]
    mine = _views(sign(frame0, picked))
    beside = [_views(sign(frame0, m)) for i, m in enumerate(masks0) if i != which]      # everybody else there, for the table
    free0 = set(_free(masks0, off(frame0)))
    # the scores are taken relative to the people there NOBODY ELSE HOLDS, as they were when held people were not listed
    theirs = [v for i, v in zip([i for i in range(int(masks0.shape[0])) if i != which], beside) if i in free0]
    use = [k for k, v in enumerate(mine) if v is not None]         # the views the subject has
    others = [t for t in theirs if t[0] is not None]
    centres = []
    for k in range(len(mine)):
        have = [t[k] for t in theirs if t[k] is not None]
        centres.append(torch.stack(have, dim=0).mean(dim=0) if have else None)
    reference = tuple(relative(v, c) for v, c in zip(mine, centres))
    result.pick_frame, result.others, result.pick_width = frame0, len(others), _width(picked)
    result.judged[int(frame0)] = [(mine if i == which else beside[i - (i > which)], [None] * len(mine), None,
                                   None if sign_whole is None else sign_whole(frame0, masks0[i]))
                                  for i in range(int(masks0.shape[0]))]
    result.views, result.views_used = len(mine), len(use)

    def per_view(sig) -> list[float | None]:
        """A person's similarity to the subject in each place compared; None where the person or the subject lacks it."""
        views = _views(sig)
        return [similarity(reference[k], relative(views[k], centres[k])) if k in use and k < len(views) and views[k] is not None else None
                for k in range(len(mine))]

    def alike(sig) -> float | None:
        """A person's similarity to the subject: the lowest over the views the subject has. None when one is missing."""
        each = per_view(sig)
        if not use or any(each[k] is None for k in use):
            return None
        return min(each[k] for k in use)

    def judge(f: int) -> tuple[float, int | None, int, torch.Tensor]:
        """Frame `f`'s best similarity, which detection has it, how many there have a head, and the detections."""
        found, _ = detect(f)
        sigs = [sign(f, m) for m in found]
        sims = [alike(v) for v in sigs]
        result.judged[int(f)] = [(_views(v), per_view(v), a, None if sign_whole is None else sign_whole(f, m))
                                 for v, a, m in zip(sigs, sims, found)]
        free = set(_free(found, off(f)))              # a held person is listed and signed, and is never the best
        heads = sum(1 for k, v in enumerate(sigs) if k in free and _has_head(v))
        able = [k for k, v in enumerate(sims) if v is not None and k in free]
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
        # A RUN HOLDING AN OWNER'S BOX IS SEARCHED WHATEVER ITS LENGTH, and on the box's frame first (2026-10-23: on 1419d
        # nine empty frames, 1419-1427, held the owner's box on 1423 and were never searched, being under REGAIN_MIN_RUN).
        # A box a seed took holds its own frame, so the boxes found here are ones refused, or frames cut as another's
        mine = boxed.get(id(shot), [])
        while len(shot.regained) < REGAIN_MOST:
            runs = [(a + shot.start, b + shot.start) for a, b in empty_runs(piece)
                    if (b - a >= REGAIN_MIN_RUN or any(a <= f - shot.start < b for f in mine)) and a + shot.start not in done]
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
            frames = list(range(a, b, step) if a > shot.start else range(b - 1, a - 1, -step))
            boxes_here = [f for f in mine if a <= f < b]
            frames = boxes_here + [f for f in frames if f not in boxes_here]
            for f in frames:
                found, _ = detect(f)
                free = _free(found, off(f))
                which, best, second = clear_best(gallery_scores(found[torch.tensor(free, dtype=torch.long)], f, known, sign))
                shot.probes.append((f, int(found.shape[0]), best, second))
                if which is not None:
                    which = free[which]
                    piece[a - shot.start:b - shot.start] = track(a, b, f, found[which])
                    _cut_on_others(result, shot, held, replace=False)
                    if track_head:
                        # found again by likeness, no box: a head only where SAM finds one, and a body box's frame is one
                        _track_head_span(result, shot, a, b, f, found[which], sign, track, bodies=bodied.get(id(shot)))
                    shot.regained.append((a, f, int(found.shape[0]), best, second))
                    break
            else:
                shot.searched.append((a, b - 1))

    rest = [s for s in shots if not s.start <= frame0 < s.end]
    first = {id(s): look(s, s.probe) for s in rest}
    for shot in rest:
        shot.first = shot.best
    floor = MATCH_FLOOR if others else PLAIN_FLOOR
    result.floor = float(floor)
    result.gap = None if match_threshold is not None else auto_gap([s.first for s in rest], floor)
    result.match = float(match_threshold) if match_threshold is not None else auto_match([s.first for s in rest], floor)
    # nobody else on the pick frame and nothing named: there is nobody to mistake the subject for
    # ...and no other tracker in the job: with `others` wired a lone person may be any of the job's named people
    alone = match_threshold is None and not others and held is None

    for shot in shots:
        if shot.start <= frame0 < shot.end:
            shot.picked, shot.seed, shot.shown, shot.index = True, frame0, frame0, which
            shot.best, shot.candidates, shot.width = 1.0, int(masks0.shape[0]), result.pick_width
            if id(shot) not in by_hand:
                piece = result.pieces[shot.start] = track(shot.start, shot.end, frame0, picked)
                _cut_on_others(result, shot, held)
                if track_head:
                    _track_head_span(result, shot, shot.start, shot.end, frame0, picked, sign, track)
                # the subject as this track shows them, for a loss later in the shot and for the run after this one
                result.gallery_frames = [f + shot.start for f in gallery_frames(piece)]
                result.gallery = [_views(sign(f, (piece[f - shot.start] > 0.5).to(torch.float32))) for f in result.gallery_frames]
                if sign_whole is not None:
                    result.gallery_whole = [sign_whole(f, (piece[f - shot.start] > 0.5).to(torch.float32)) for f in result.gallery_frames]
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
            _cut_on_others(result, shot, held)
            if track_head:
                _track_head_span(result, shot, shot.start, shot.end, seed, seeds[i], sign, track)
            regain(shot)
    _correct(result, corrections, detect, track, held, sign=sign, track_head=track_head)
    # A correction says who, not that the tracker will hold them: where the corrected track lets go, look again as
    # for any other shot. The gallery is the corrected track's own frames (`regain`), so the search is for that person.
    for number, person in sorted(corrections.items()):
        if person is not None:
            regain(result.shots[int(number) - 1])
    _seed(result, given_seeds, detect, track, held, sign=sign, track_head=track_head, in_box=in_box, seed_held=seed_held, blind=blind)
    if result.pick_seed in ("is the pick", "names no detection"):
        # the subject as the seeded track shows them, as the picked shot's own track would have: for a loss later in
        # that shot, for a reader of the table, and for the run after this one. A shot seeded by segmentation alone is
        # still the owner's word about who she is, which the rule's pick is not
        shot = result.shots[on_pick - 1]
        piece = result.pieces.get(shot.start)
        if piece is not None:
            result.gallery_frames = [f + shot.start for f in gallery_frames(piece)]
            result.gallery = [_views(sign(f, (piece[f - shot.start] > 0.5).to(torch.float32))) for f in result.gallery_frames]
            if sign_whole is not None:
                result.gallery_whole = [sign_whole(f, (piece[f - shot.start] > 0.5).to(torch.float32)) for f in result.gallery_frames]
    for number in sorted(given_seeds):
        if result.shots[int(number) - 1].seed is not None:
            regain(result.shots[int(number) - 1])
    return result


def _cut_on_others(result: Followed, shot: Shot, held, *, replace: bool = True) -> None:
    """Empty frames in `shot` where the track moved mostly onto another tracker's person (`frames_on_others`).

    Run before `gallery_frames` and before `regain`, so frames where the tracker jumped onto another person are
    never sampled into the gallery and count as empty runs for `regain` to search. Skipped when a correction or
    seed explicitly named a held person (`shot.also_held`), where two masks sit on one person on purpose until
    the other tracker is corrected."""
    if replace and result.on_others:
        result.on_others = [f for f in result.on_others if not shot.start <= f < shot.end]
    other_masks = getattr(held, "others", None)
    if other_masks is None or shot.also_held:
        return
    piece = result.pieces.get(shot.start)
    if piece is None:
        return
    most = float(getattr(held, "most", ON_OTHERS))
    local = frames_on_others(piece, other_masks[shot.start:shot.end], most)
    if local:
        piece[local] = 0.0
        if shot.start in result.head_pieces:
            result.head_pieces[shot.start][local] = 0.0
        result.on_others = sorted(set(result.on_others).union(f + shot.start for f in local))


def _correct(result: Followed, corrections: dict[int, int | None], detect, track, held=None,
             *, sign=None, track_head: bool = False) -> None:
    """Apply the corrections to `follow`'s result: each on the frame its shot's tile shows.

    A correction may name a person another tracker holds: a person's word outranks a tracker's take, and one of the
    two trackers is then wrong. The shot remembers it (`also_held`), for the report and the table."""
    for number, person in sorted(corrections.items()):
        shot = result.shots[int(number) - 1]
        if person is None:
            shot.seed, shot.lone, shot.corrected, shot.also_held = None, False, "none", False
            result.pieces.pop(shot.start, None)
            result.head_pieces.pop(shot.start, None)
            _cut_on_others(result, shot, held)
            continue
        people, _ = detect(shot.shown)
        which = shot_table.detection_of(people, int(person))
        if which is None:
            raise ValueError(
                f"corrections: shot {number} has {int(people.shape[0])} person(s) on frame {shot.shown}, the frame "
                f"its tile shows, so there is no person {int(person)} to take")
        shot.seed, shot.index, shot.lone, shot.corrected = shot.shown, which, False, f"person {int(person)}"
        shot.also_held = held is not None and which in held(int(shot.shown))
        shot.candidates, shot.width = int(people.shape[0]), _width(people[which])
        result.pieces[shot.start] = track(shot.start, shot.end, shot.shown, people[which])
        _cut_on_others(result, shot, held)
        if track_head and sign is not None:
            _track_head_span(result, shot, shot.start, shot.end, shot.shown, people[which], sign, track)


def seed_names(detect, frame: int, box: tuple) -> tuple[torch.Tensor, int | None, dict]:
    """The detection on `frame` a seed's box names: (that frame's detections, which one or None, the seed's record).

    The one whose box overlaps the seed's most; it is refused, and nobody is named, under `SEED_AGREES`.
    Among EVERYBODY on the frame, a person another tracker holds included: a seed is a person's word (or a tool's
    carrying one), and `_seed` marks the shot when the one named is held. A box read off a tile overlaps its own
    detection wholly; a box typed by hand between two people who overlap names the nearer, held or not.
    """
    people, _ = detect(int(frame))
    wanted = (box[0], box[1], box[0] + box[2], box[1] + box[3])
    theirs = [shot_table._box(m) for m in people]
    overlaps = [0.0 if b is None else subject_tracks.box_overlap((b[0], b[1], b[0] + b[2], b[1] + b[3]), wanted) for b in theirs]
    best = max(range(len(overlaps)), key=lambda k: (overlaps[k], -k)) if overlaps else None
    record = {"frame": int(frame), "box": [float(v) for v in box],
              "person": None if best is None else shot_table.person_number(people, best),
              "overlap": None if best is None else float(overlaps[best]), "refused": best is None or overlaps[best] < SEED_AGREES,
              "nearest_box": None if best is None or theirs[best] is None else [int(v) for v in theirs[best]]}
    return people, best, record


@dataclass
class SeedTaken:
    """What one owner's box named on its frame (`name_seed`)."""
    frame: int
    people: torch.Tensor          # the frame's detections, [N, H, W]
    index: int | None             # the detection taken; the nearest by box where refused; None where segmented
    mask: torch.Tensor | None     # the seed mask [H, W]; None where refused
    record: dict                  # the record the shot keeps (`Shot.seeded`)
    body: bool = False            # the box encloses her body, not her head (`SEED_BODY`): her person track only


def box_region(box: tuple, height: int, width: int) -> torch.Tensor:
    """[H, W] bool, True inside a box (x, y, width, height) taken outward to whole pixels and clipped to the frame."""
    x0, y0 = max(int(math.floor(float(box[0]))), 0), max(int(math.floor(float(box[1]))), 0)
    x1 = min(int(math.ceil(float(box[0]) + float(box[2]))), int(width))
    y1 = min(int(math.ceil(float(box[1]) + float(box[3]))), int(height))
    out = torch.zeros((int(height), int(width)), dtype=torch.bool)
    out[y0:y1, x0:x1] = True
    return out


def name_seed(detect, frame: int, box: tuple, held=None, in_box=None, drawn: bool = False, blind: str = "") -> SeedTaken:
    """What an owner's box on `frame` names, tried in order, and the record that says how (2026-10-23, the owner's go).

    1. `seed_names`: the detection whose box agrees with it at `SEED_AGREES`, among everybody (as before: a box read
       off a tile agrees with its own detection wholly).
    2. Else the detection whose MASK overlaps the box, as a region, at `SEED_OVERLAPS` or more, the most of any, among
       the detections no other tracker holds (`held`): a box drawn by hand rarely agrees with a detection's tight box at
       0.9. Refused when a second detection also passes and is not a part of the first (`SEED_MOSTLY` of it on the
       first's mask): the box holds two people, and taking one would name somebody without a word to say who.
    3. Else, when no detection passes, what SAM 3 segments inside the box from the box itself (`in_box`, core's
       `SAM3_Detect` box prompt; `_sam_callables`), clipped to the box, taken only as one person the detector found
       too and less every pixel another tracker holds (`prompt_seed`): NEVER THE RECTANGLE. Refused, and recorded as
       doubtful, when it matches no person detection, is another tracked subject, or runs onto another person.
    A refusal says why in the record (`why`) and keeps the nearest detection by box, as a refusal always has.

    A box the owner DREW (`drawn`, the seed line's `drawn` word) is approximate, never an outline, and is named by
    `name_drawn` instead: none of the three steps above is tried for it. `blind`, when given, says why this tracker
    cannot hold a box prompt to the job's other subjects on this frame: step 3 (and `name_drawn`'s) is then refused as
    doubtful and SAM is not asked."""
    people, best, record = seed_names(detect, frame, box)
    # `mask_overlap`: the taken detection's mask against the box as a region, written where step 2 took it
    record.update(how=SEED_REFUSED if record["refused"] else SEED_BY_BOX, px=None, why="", mask_overlap=None)
    frame = int(frame)
    if drawn:
        return name_drawn(people, best, record, frame, box, held, in_box, blind)
    if not record["refused"]:
        record["px"] = int((people[best] > 0.5).sum())
        return SeedTaken(frame, people, best, people[best], record)
    height, width = int(people.shape[-2]), int(people.shape[-1])
    region = box_region(box, height, width)
    off = held(frame) if held is not None else frozenset()

    def fills(k: int) -> float:
        on = people[k] > 0.5
        union = int((on | region).sum())
        return float((on & region).sum()) / union if union else 0.0

    passing = sorted((k for k in range(int(people.shape[0])) if k not in off and fills(k) >= SEED_OVERLAPS),
                     key=lambda k: (-fills(k), k))
    if passing:
        top = passing[0]
        on_top = people[top] > 0.5
        others_too = [k for k in passing[1:] if float(((people[k] > 0.5) & on_top).sum()) < SEED_MOSTLY * float((people[k] > 0.5).sum())]
        if others_too:
            record["why"] = (f"{1 + len(others_too)} detections' masks overlap the box at {SEED_OVERLAPS:.2f} or more and are not parts of "
                             f"one another (persons {', '.join(str(shot_table.person_number(people, k)) for k in [top] + others_too)}): "
                             "the box holds more than one person")
            return SeedTaken(frame, people, best, None, record)
        b = shot_table._box(people[top])
        record.update(refused=False, how=SEED_BY_OVERLAP, person=shot_table.person_number(people, top),
                      overlap=subject_tracks.box_overlap((b[0], b[1], b[0] + b[2], b[1] + b[3]),
                                                         (box[0], box[1], box[0] + box[2], box[1] + box[3])),
                      nearest_box=[int(v) for v in b], mask_overlap=round(fills(top), 3), px=int(on_top.sum()))
        return SeedTaken(frame, people, top, people[top], record)
    if in_box is None:
        record["why"] = (f"no detection's mask overlaps the box at {SEED_OVERLAPS:.2f} or more, and this tracker has no "
                         "box-prompted segmentation to take what is inside it")
        return SeedTaken(frame, people, best, None, record)
    if blind:
        record.update(doubtful=True, why=f"no detection's mask overlaps the box at {SEED_OVERLAPS:.2f} or more, and {blind}")
        return SeedTaken(frame, people, best, None, record)
    seed, facts = prompt_seed(in_box(frame, box), people, off, getattr(held, "others", None), frame, clip=region)
    record.update(doubtful=facts["doubtful"], prompt=facts)
    if seed is None:
        record["why"] = f"no detection's mask overlaps the box at {SEED_OVERLAPS:.2f} or more, and {facts['why']}"
        return SeedTaken(frame, people, best, None, record)
    record.update(refused=False, how=SEED_BY_SEGMENT, px=int((seed > 0.5).sum()))
    return SeedTaken(frame, people, None, seed, record)


def prompt_seed(cut, people: torch.Tensor, off, theirs, frame: int, clip: torch.Tensor | None = None) -> tuple[torch.Tensor | None, dict]:
    """What SAM 3's box prompt segmented from a box, held to the detections of its frame (2026-10-23, the blind read).

    `cut` is the prompt's mask [H, W] or None, `off` the detections another tracked subject is (held by an earlier
    tracker, or named by another subject's box on this frame), `theirs` the earlier trackers' joined mask [frames, H, W]
    or None, `clip` the box's region for a tile's box (exact) and None for a drawn one (approximate: a rough box would
    cut her off). Taken only as one person: the segmentation must lie mostly (`SEED_MOSTLY`) on one detection that is
    not `off`, and no more than `PROMPT_ON_OTHERS` of it on any other detection or on `theirs`; the seed is then the
    segmentation less `theirs` and less EVERY OTHER DETECTED PERSON's mask, so what bled onto another person (their arm,
    under the line) is never kept, for the first tracker of a job too, which has no `theirs` (2026-10-23, the independent
    review). A detection lying mostly (`SEED_MOSTLY`) on the matched one is a part of that person (her head detected
    apart from her body): it neither counts as another person nor is taken out. Anything else is refused as DOUBTFUL,
    with why: it matches no person detection (a thing), it is another tracked subject, or it runs onto another person or
    subject (two people as one blob). Returns (the seed or None, {"why", "doubtful", "person": the detection matched or
    None, "index": its index, "on_person", "on_others"})."""
    facts = {"why": "", "doubtful": False, "person": None, "index": None, "on_person": None, "on_others": None}
    got = (cut > 0.5) if cut is not None else None
    if got is not None and clip is not None:
        got = got & clip
    if got is None or not bool(got.any()):
        facts["why"] = "SAM 3's box prompt segmented nothing" + (" inside the box" if clip is not None else " from it")
        return None, facts
    area = float(got.sum())
    shares = [float((got & (people[k] > 0.5)).sum()) / area for k in range(int(people.shape[0]))]
    held_share = float((got & (theirs[frame] > 0.5)).sum()) / area if theirs is not None else 0.0
    facts["doubtful"] = True
    if not shares or max(shares) < SEED_MOSTLY:
        facts["why"] = (f"what SAM 3's box prompt segmented matches no person detection (at most {max(shares, default=0.0):.2f} of it "
                        f"on any one; it must lie {SEED_MOSTLY:.2f} on one): DOUBTFUL, a thing or nobody")
        return None, facts
    k = max(range(len(shares)), key=lambda j: (shares[j], -j))
    facts.update(person=shot_table.person_number(people, k), index=int(k), on_person=round(shares[k], 3), on_others=round(held_share, 3))
    on_k = people[k] > 0.5
    part_of_k = {j for j in range(len(shares)) if j != k
                 and float(((people[j] > 0.5) & on_k).sum()) >= SEED_MOSTLY * max(float((people[j] > 0.5).sum()), 1.0)}
    if k in off:
        facts["why"] = (f"what SAM 3's box prompt segmented lies {shares[k]:.2f} on person {facts['person']}, another tracked "
                        "subject: DOUBTFUL")
        return None, facts
    over = [j for j in range(len(shares)) if j != k and j not in part_of_k and shares[j] > PROMPT_ON_OTHERS]
    if over or held_share > PROMPT_ON_OTHERS:
        facts["why"] = ("what SAM 3's box prompt segmented runs onto "
                        + "; ".join([f"person {shot_table.person_number(people, j)} ({shares[j]:.2f} of it)" for j in over]
                                    + ([f"another tracked subject's track ({held_share:.2f} of it)"] if held_share > PROMPT_ON_OTHERS else []))
                        + f" beside person {facts['person']} (the line is {PROMPT_ON_OTHERS:.2f}): DOUBTFUL, more than one person")
        return None, facts
    facts["doubtful"] = False
    left = got & ~(theirs[frame] > 0.5) if theirs is not None else got.clone()
    for j in range(len(shares)):
        if j != k and j not in part_of_k:
            left &= ~(people[j] > 0.5)
    return left.to(torch.float32), facts


def name_drawn(people: torch.Tensor, best: int | None, record: dict, frame: int, box: tuple, held=None, in_box=None,
               blind: str = "") -> SeedTaken:
    """What a box the owner DREW on a sheet names (2026-10-23, the owner: "the boxes i drew were approximates!!! they were
    just to give you a rough idea"): which person is whom, roughly; never an outline.

    1. Every detection with any pixel in the box is ranked by the share of its mask inside the box. When the first is a
       person another tracker holds (another tracked subject: the lead's dress under 4647's boxes on 4674-4677), the box
       is refused: it lies on someone already followed, and nothing here may guess past that.
    2. The detections with at least `DRAWN_INSIDE` of their mask inside the box qualify; one lying mostly (`SEED_MOSTLY`)
       on another qualifier's mask is a part of that person and not a second one. Two people qualifying refuses the box
       (it straddles them). One, no other tracker's, is taken. The record keeps the share of its mask inside the box
       (`inside`) and the share of the box it covers (`covers`).
    3. Only where nobody qualifies: SAM 3's box prompt (`in_box`), NOT clipped to the box, since a rough box would cut
       her off, and held to the frame's detections as a tile's is (`prompt_seed`): one person, not another tracked
       subject, not running onto anyone else, or refused as doubtful.
    The box itself is never the seed."""
    record.update(refused=True, how=SEED_REFUSED, drawn=True, inside=None, covers=None)
    height, width = int(people.shape[-2]), int(people.shape[-1])
    region = box_region(box, height, width)
    area = max(int(region.sum()), 1)
    off = held(frame) if held is not None else frozenset()
    on = [people[k] > 0.5 for k in range(int(people.shape[0]))]
    inside = {k: float((m & region).sum()) / max(float(m.sum()), 1.0) for k, m in enumerate(on) if bool((m & region).any())}
    ranked = sorted(inside, key=lambda k: (-inside[k], -float((on[k] & region).sum()), k))
    said = lambda k: f"person {shot_table.person_number(people, k)} ({inside[k]:.2f} of their mask inside it)"   # noqa: E731
    if ranked:
        record.update(inside=round(inside[ranked[0]], 3), covers=round(float((on[ranked[0]] & region).sum()) / area, 3))
    if ranked and ranked[0] in off:
        record["why"] = (f"the person lying most inside the drawn box, {said(ranked[0])}, is another tracked subject's: the box "
                         "lies on someone already followed, and this tracker does not guess past that")
        return SeedTaken(frame, people, best, None, record)
    qualify = [k for k in ranked if inside[k] >= DRAWN_INSIDE]
    persons = [k for k in qualify if not any(j != k and float((on[k] & on[j]).sum()) >= SEED_MOSTLY * float(on[k].sum())
                                             and float(on[j].sum()) > float(on[k].sum()) for j in qualify)]
    if len(persons) > 1:
        record["why"] = (f"{len(persons)} people lie mostly inside the drawn box ({', '.join(said(k) for k in persons)}): it straddles "
                         "them, and a rough box cannot say which is meant")
        return SeedTaken(frame, people, best, None, record)
    if persons:
        k = persons[0]
        if k in off:
            record["why"] = f"the one person lying mostly inside the drawn box, {said(k)}, is another tracked subject's"
            return SeedTaken(frame, people, best, None, record)
        b = shot_table._box(people[k])
        record.update(refused=False, how=SEED_BY_INSIDE, person=shot_table.person_number(people, k),
                      overlap=subject_tracks.box_overlap((b[0], b[1], b[0] + b[2], b[1] + b[3]),
                                                         (box[0], box[1], box[0] + box[2], box[1] + box[3])),
                      nearest_box=[int(v) for v in b], inside=round(inside[k], 3), covers=round(float((on[k] & region).sum()) / area, 3),
                      px=int(on[k].sum()))
        return SeedTaken(frame, people, k, people[k], record)
    if in_box is None:
        record["why"] = (f"nobody lies mostly ({DRAWN_INSIDE:.2f}) inside the drawn box, and this tracker has no box-prompted "
                         "segmentation to look with")
        return SeedTaken(frame, people, best, None, record)
    if blind:
        record.update(doubtful=True, why=f"nobody lies mostly ({DRAWN_INSIDE:.2f}) inside the drawn box, and {blind}")
        return SeedTaken(frame, people, best, None, record)
    seed, facts = prompt_seed(in_box(frame, box), people, off, getattr(held, "others", None), frame, clip=None)
    record.update(doubtful=facts["doubtful"], prompt=facts)
    if seed is None:
        record["why"] = f"nobody lies mostly ({DRAWN_INSIDE:.2f}) inside the drawn box, and {facts['why']}"
        return SeedTaken(frame, people, best, None, record)
    record.update(refused=False, how=SEED_BY_PROMPT, px=int((seed > 0.5).sum()), covers=round(float(((seed > 0.5) & region).sum()) / area, 3))
    return SeedTaken(frame, people, None, seed, record)


def _seed(result: Followed, seeds: dict[int, list[tuple[int, tuple | None]]], detect, track, held=None,
          *, sign=None, track_head: bool = False, in_box=None, seed_held=None, blind=None) -> None:
    """Apply the seeds to `follow`'s result: every box on its own frame, by what it names there (`name_seed`).

    ONE BOX: the shot is tracked from it over the whole shot, as it always was. SEVERAL (2026-10-23, the owner's go:
    every owner box is a seed on its frame): each box taken is tracked from its own frame out to its neighbours'
    frames (the shot's ends for the first and last), so every frame between two boxes is tracked from both, and the
    tracks are kept frame by frame by the tracker's own score (`merge_tracks`), not split at a midpoint. Each taken
    box's record says the frames its track ran over and where it was kept (`first_frame`, `last_frame`, `kept_on`). A
    box refused takes no frames and says why; a shot whose every box is refused is left empty. The shot is shown on,
    and its `seeded` is, the first box taken. The head is tracked within each stretch the merged track holds her,
    from the boxes inside it (`_track_head_span`)."""
    for number, these in sorted(seeds.items()):
        these = seed_list(these)
        shot = result.shots[int(number) - 1]
        frame = int(these[0][0])
        shot.lone, shot.shown = False, frame
        # a belt: a seeded shot is in `by_hand`, so the automatic pass has tracked nothing here
        result.pieces.pop(shot.start, None)
        result.head_pieces.pop(shot.start, None)
        _cut_on_others(result, shot, held)
        if these[0][1] is None:                     # `none`, which `parse_seeds` takes only alone in its shot
            shot.seed, shot.index = None, None
            shot.seeded = {"frame": frame, "box": None, "person": None, "overlap": None, "refused": False, "nearest_box": None,
                           "how": "nobody", "px": None, "why": "", "mask_overlap": None}
            continue
        named = [name_seed(detect, f, box, seed_held if seed_held is not None else held, in_box, drawn=SEED_DRAWN in words,
                           blind=blind(f) if blind is not None else "")
                 for f, box, *words in these]
        for x, (_, _, *word) in zip(named, these):
            x.body = SEED_BODY in word
            x.record["encloses"] = SEED_BODY if x.body else None
        taken = [x for x in named if x.mask is not None]
        lead = taken[0] if taken else named[0]
        shot.seeded, shot.shown, shot.candidates = lead.record, lead.frame, int(lead.people.shape[0])
        if len(named) > 1:
            for x in named:
                x.record.update(first_frame=None, last_frame=None, kept_on=[])
            shot.seeded_all = [x.record for x in named]
        if not taken:
            # the nearest detection stays outlined on the tile as the closest; nobody is taken
            shot.seed, shot.index = None, lead.index
            continue
        shot.seed, shot.index, shot.width = lead.frame, lead.index, _width(lead.mask)
        shot.also_held = held is not None and any(x.index is not None and x.index in held(x.frame) for x in taken)
        if len(taken) == 1:
            result.pieces[shot.start] = track(shot.start, shot.end, lead.frame, lead.mask)
            if len(named) > 1:
                lead.record.update(first_frame=int(shot.start), last_frame=int(shot.end) - 1,
                                   kept_on=[[int(shot.start) + a, int(shot.start) + b - 1] for a, b in filled_runs(result.pieces[shot.start])])
        else:
            runs, trouble = [], []
            for k, x in enumerate(taken):
                lo = taken[k - 1].frame if k else shot.start
                hi = taken[k + 1].frame + 1 if k + 1 < len(taken) else shot.end
                masks, scores, why = _scored(track, lo, hi, x.frame, x.mask, record=True)
                runs.append((x.frame, lo, hi, masks, scores))
                trouble.extend(why)
                x.record.update(first_frame=int(lo), last_frame=int(hi) - 1)
            piece, scores, kept, disagreed = merge_tracks(shot.start, shot.end, runs, detect)
            result.pieces[shot.start] = piece
            shot.disagreed = list(disagreed)
            keep = getattr(track, "keep_scores", None)
            if keep is not None:
                keep(int(shot.start), scores, trouble)
            for x in taken:
                x.record["kept_on"] = shot_table._runs([k == x.frame for k in kept], int(shot.start))
        _cut_on_others(result, shot, held)
        if track_head and sign is not None:
            # a body box seeds her person track and never her head's: its frame is passed so no head is made from it
            heads = [(x.frame, x.mask) for x in taken if not x.body]
            # the head's own seed is the first box round her head (its frame, her mask there), never a body box before it:
            # a shot is headless-by-body only when every box it took is a body box (2026-10-23, the independent review)
            ref = next((x for x in taken if not x.body), lead)
            _track_head_span(result, shot, shot.start, shot.end, ref.frame, ref.mask, sign, track,
                             seeds=heads if len(taken) > 1 else None, bodies=[x.frame for x in taken if x.body] or None,
                             anchored=not ref.body, body_lead=all(x.body for x in taken))


def _frame_list(frames: list[int], most: int = 16) -> str:
    """Frames for a report line: the first `most`, and how many more."""
    shown = ", ".join(str(f) for f in frames[:most])
    return shown if len(frames) <= most else f"{shown} and {len(frames) - most} more"


def share_on(mask: torch.Tensor, others: torch.Tensor) -> float:
    """The share of a [H, W] mask's pixels that lie on `others`, a [H, W] mask of other people. 0 for an empty mask."""
    on = mask > 0.5
    area = int(on.sum())
    return float((on & (others > 0.5)).sum()) / area if area else 0.0


def held_people(detect, others: torch.Tensor, most: float = ON_OTHERS):
    """Which of a frame's detections another tracker holds: a callable, frame -> the held indices into `detect(frame)`.

    `others` is [frames, H, W], every earlier tracker's mask joined. A detection with more than `most` of its pixels
    on that frame's `others` is HELD: it stays in the frame's list, numbered like anybody, and is never chosen
    (`_free` is the one road to a candidate). `detect` is the detector AFTER duplicates were joined, as it was when
    held people were removed from the list: the mark is read on the detection that stands for a person, never on
    one of the detections joined into it. The frames asked about are kept on the callable as `seen`."""
    seen: dict[int, frozenset] = {}

    def held(f: int) -> frozenset:
        f = int(f)
        if f not in seen:
            masks, _ = detect(f)
            seen[f] = frozenset(i for i in range(int(masks.shape[0])) if share_on(masks[i], others[f]) > most)
        return seen[f]

    held.seen = seen
    held.others = others
    held.most = most
    return held


def one_detection_each(detect, joined: dict[int, int], same: float = subject_tracks.SAME_THING,
                       parts: dict[int, float] | None = None, unsure: dict[int, int] | None = None):
    """`detect` with detections that are another detection of the same thing left out (`subject_tracks.one_each`).

    The one the detector scored highest stands for them, so a person is numbered once on the tile, is one candidate
    for the pick, a shot's match and a re-find, and "as many people as subjects" can be counted. `joined[frame]` is
    how many were left out on each frame looked at.

    NOT JOINED WHEN THE ONE KEPT WOULD BE A PART (mrcorn's cold read, 2026-10-10). The highest scored of a thing can
    be a head or a hand; kept, it would be what is sized, compared and seeded, and the whole would have no number
    for a correction to name. So a thing whose kept detection is under `A_PART_UNDER` of its largest mask
    (`subject_tracks.kept_share`) is left as the detector returned it, every detection numbered, which is what it
    cost before this existed: an extra number. `unsure[frame]` counts those detections. `parts[frame]` is the
    smallest share among the things that WERE joined there, so it is never under the line.
    """
    kept: dict[int, tuple[torch.Tensor, list[float]]] = {}

    def detect_one_each(f: int):
        if f not in kept:
            masks, scores = detect(f)
            _, stands = subject_tracks.one_each(masks, scores, same)
            shares = subject_tracks.kept_share(masks, stands)
            keep, left, doubted, safe = [], 0, 0, []
            for best, those in stands.items():
                if len(those) > 1 and shares[best] < A_PART_UNDER:
                    keep.extend(those)
                    doubted += len(those)
                else:
                    keep.append(best)
                    left += len(those) - 1
                    if len(those) > 1:
                        safe.append(shares[best])
            keep.sort()
            joined[int(f)] = left
            if unsure is not None and doubted:
                unsure[int(f)] = doubted
            if parts is not None and safe:
                parts[int(f)] = min(safe)
            kept[f] = (masks[keep], [scores[i] for i in keep if i < len(scores)])
        return kept[f]

    return detect_one_each


#: A thing is joined only when the detection kept for it is at least this share of the largest mask it stands for;
#: under it the kept one may be a part of the person and the detections are left as they came. Reasoned: a tighter
#: mask of the same person is most of the looser one (the one real frame seen, 2026-10-10: the kept mask was about
#: nine tenths of the largest by area); a head or a hand is well under half of a whole person.
A_PART_UNDER = 0.5


def joined_report(shots, joined: dict[int, int], parts: dict[int, float] | None = None, unsure: dict[int, int] | None = None) -> str:
    """The report's lines about detections joined as one thing, and about those left alone as possibly one; empty when neither."""
    frames = sorted(f for f, n in joined.items() if n)
    parts, unsure = parts or {}, unsure or {}
    lines = []
    if frames:
        lines.append(f"the detector returned the same thing more than once on {len(frames)} of the {len(joined)} frame(s) looked at "
                     f"({sum(joined.values())} extra detection(s)): each is joined with the one it scored highest and is not numbered")
    if unsure:
        lines.append(f"on {len(unsure)} frame(s) detections that may be one thing were NOT joined, because the one the detector "
                     f"scored highest is under half of the largest and may be a part: {_frame_list(sorted(unsure))}. Each is numbered; "
                     "a correction can take the whole or the part")
    for number, shot in enumerate(shots, 1):
        n, doubt = joined.get(int(shot.shown), 0), unsure.get(int(shot.shown), 0)
        if n:
            share = parts.get(int(shot.shown))
            lines.append(f"[{number}] {n} of them on the tile's frame, {shot.shown}"
                         + ("" if share is None else f"; the detection kept is {share:.2f} of the largest it stands for")
                         + (f". This shot is corrected by hand ({shot.corrected}): the numbers count the people left, so check the "
                            "number against this tile" if shot.corrected else ""))
        if doubt:
            lines.append(f"[{number}] {doubt} detections on the tile's frame, {shot.shown}, may be one thing and are numbered separately")
    return "\n".join(lines)


#: The report says the shot table's size when its text is longer than this, in characters. Chosen: a megabyte.
TABLE_SAID_OVER = 1_000_000


def frames_on_others(mask: torch.Tensor, others: torch.Tensor, most: float = ON_OTHERS) -> list[int]:
    """The frames where this tracker's own mask lies mostly on `others`: it is on a person another tracker holds.

    A tracker can move its mask onto a neighbour with no empty frame between (`subject_tracks.py`, "The step test"),
    and with `others` wired that is visible for nothing (`_cut_on_others` empties those frames before `regain` runs).
    """
    on, taken = mask > 0.5, others > 0.5
    area = on.flatten(1).sum(dim=1)
    both = (on & taken).flatten(1).sum(dim=1)
    return [int(f) for f in ((area > 0) & (both.to(torch.float32) > most * area.to(torch.float32))).nonzero().flatten().tolist()]


def others_report(shots, held: dict[int, frozenset], on_others: list[int]) -> str:
    """The report's lines about the `others` input: who is held, which shots that left with no subject, a person's word
    that names a held person, and the frames cut where this track lay mostly on another tracker's person. `held` is
    `held_people`'s `seen`."""
    looked = sorted(held)
    lines = [f"others: {sum(len(held[f]) for f in looked)} detection(s) on {sum(1 for f in looked if held[f])} of the {len(looked)} "
             "frame(s) looked at lie mostly on a person another tracker holds: listed and numbered, marked held, never candidates"]
    for number, shot in enumerate(shots, 1):
        there = sum(len(held[f]) for f in looked if shot.start <= f < shot.end)
        if shot.also_held:
            lines.append(f"[{number}] THE PERSON NAMED BY HAND HERE IS ALSO HELD BY ANOTHER TRACKER (frame {shot.shown}): one of "
                         "the two trackers is wrong, and two masks are on one person until the other is corrected"
                         + (". AND THIS IS THE PICK: every other shot is judged against that person, whom the other tracker "
                            "holds wherever it follows them, so this tracker can take nobody there until the other is corrected"
                            if shot.picked else ""))
        if there and shot.seed is None:
            lines.append(f"[{number}] no subject: every person compared there was another tracker's, or nobody was "
                         f"like enough ({there} held detection(s) on the frames looked at)")
    if on_others:
        runs = shot_table._runs([f in set(on_others) for f in range(on_others[0], on_others[-1] + 1)], on_others[0])
        lines.append(f"cut {len(on_others)} frame(s) where this track lay mostly on another tracker's person: "
                     + ", ".join(f"{a}-{b}" for a, b in runs[:8]) + (" and more" if len(runs) > 8 else ""))
    return "\n".join(lines)


class ScoreTap:
    """While entered, records the object score a tracker gives at each frame it steps through.

    ComfyUI's SAM 3 tracker computes a score per frame for whether its object is there (`object_score_logits`,
    `comfy/ldm/sam3/tracker.py::track_step`) and its node returns one number per object, not per frame. Here the
    tracker object's `track_step` is wrapped for the length of one call and put back after: nothing in ComfyUI is
    edited. `seen[frame]` is that frame's score for the one object tracked, as the tracker's own number (over 0
    it takes the object as present): kept as that and not as a probability, which on an easy stretch rounds to 1
    on every frame and would show a fall only once it was over. `trouble` says in words why the record cannot be
    used (no such method, a frame stepped twice, more than one object), and is empty when it can. A tracker with
    no `track_step` records nothing and says so.
    """

    def __init__(self, tracker):
        self.tracker, self.seen, self.trouble = tracker, {}, ""
        self._wrapped, self._was = False, None

    def __enter__(self):
        step = getattr(self.tracker, "track_step", None)
        if not callable(step):
            self.trouble = "the tracker has no `track_step` to read a score from"
            return self

        def recording(*args, **kwargs):
            out = step(*args, **kwargs)
            frame = kwargs.get("frame_idx", args[0] if args else None)
            logits = out.get("object_score_logits") if isinstance(out, dict) else None
            if frame is None or logits is None:
                self.trouble = self.trouble or "a step returned no frame number or no object score"
            elif int(frame) in self.seen:
                self.trouble = self.trouble or f"frame {int(frame)} was stepped more than once"
            elif int(logits.numel()) != 1:
                self.trouble = self.trouble or f"{int(logits.numel())} object scores on a frame, where one object is tracked"
            else:
                self.seen[int(frame)] = float(logits.detach().float().flatten()[0])
            return out

        # an attribute of this object only; the class is not touched. Whatever the object itself already carried
        # under that name (another wrapper, an outer tap) is remembered and put back, not deleted (mrcorn's cold read)
        self._was = vars(self.tracker).get("track_step")
        self.tracker.track_step = recording
        self._wrapped = True
        return self

    def __exit__(self, *exc):
        if self._wrapped:
            if self._was is None:
                vars(self.tracker).pop("track_step", None)
            else:
                self.tracker.track_step = self._was
            self._wrapped = False
        return False


def scores_for(seen: dict[int, float], trouble: str, frames: int, backwards: bool = False) -> tuple[list[float | None], str]:
    """One tracked run's scores in clip order: ([score or None] for its `frames` frames, lowest clip frame first; why not).

    `seen` is a `ScoreTap`'s record in the run's own numbers, 0 being the frame it was seeded on. A run fed
    `backwards` was seeded on its highest clip frame, so its record is turned round. The frame a run is seeded on
    is given its mask and may not be stepped, so it alone may be missing. Any other gap, or any trouble the tap
    met, gives no scores at all and the reason: a list that might be out of step is worse than none.
    """
    if trouble:
        return [None] * frames, trouble
    missing = [k for k in range(1, frames) if k not in seen]
    if missing or any(k < 0 or k >= frames for k in seen):
        return [None] * frames, (f"the tracker stepped {len(seen)} frame(s) of a run of {frames}"
                                 + (f", frame {missing[0]} of the run among those it did not" if missing else ""))
    local = [seen.get(k) for k in range(frames)]
    return (local[::-1] if backwards else local), ""


def assemble(n_frames: int, height: int, width: int, pieces: dict[int, torch.Tensor]) -> torch.Tensor:
    """[n_frames, height, width] of 0 or 1: the tracked pieces in place, zeros where the subject is absent."""
    out = torch.zeros((int(n_frames), int(height), int(width)), dtype=torch.float32)
    for start, piece in pieces.items():
        out[start:start + piece.shape[0]] = (piece > 0.5).to(torch.float32)
    return out


def _seed_said(given: dict) -> str:
    """One owner's box in the report's words: what it took on its frame and how (`name_seed`), or why it was refused."""
    at = f"load frame {given['frame']}" + (" (a box round her body: her person track only, never her head)"
                                          if given.get("encloses") == SEED_BODY else "")
    if given["refused"] and given.get("drawn"):
        return f"SEED REFUSED on {at}, a box drawn by hand {[round(v) for v in given['box']]}: {given['why']}"
    if given["refused"]:
        near = ("there is no detection on that frame" if given["nearest_box"] is None else
                f"the nearest detection is person {given['person']}, box {given['nearest_box']}, overlap {given['overlap']:.2f} "
                f"(the line is {SEED_AGREES:.2f}). A tighter box of the same person means the line is too strict here; "
                "another person means the seed is wrong")
        return (f"SEED REFUSED on {at}, box {[round(v) for v in given['box']]}: {near}"
                + (f"; {given['why']}" if given.get("why") else ""))
    how = given.get("how", SEED_BY_BOX)
    if how == SEED_BY_INSIDE:
        return (f"seeded on {at}: took person {given['person']}, the one person lying mostly inside the box drawn by hand ({given['inside']:.2f} "
                f"of their mask inside it, the line {DRAWN_INSIDE:.2f}; covering {given['covers']:.2f} of the box)")
    if how == SEED_BY_PROMPT:
        return (f"seeded on {at}: nobody lay mostly inside the box drawn by hand, so SAM 3's box prompt from it, not clipped to it, "
                f"{given['px']} px less another tracked subject's")
    if how == SEED_BY_SEGMENT:
        return (f"seeded on {at}: no detection agreed with the box, so SAM 3 segmented inside it from the box itself, "
                f"{given['px']} px, less anything another tracker holds")
    if how == SEED_BY_OVERLAP:
        return (f"seeded on {at}: took person {given['person']}, whose mask overlaps the box {given['mask_overlap']:.2f} "
                f"(the line is {SEED_OVERLAPS:.2f}; their boxes {given['overlap']:.2f}, under the {SEED_AGREES:.2f} a box read off a tile meets)")
    return f"seeded on {at}: took person {given['person']}, box overlap {given['overlap']:.2f}"


def _state(shot: Shot) -> str:
    if shot.seeded:
        return "absent (seed refused)" if shot.seeded["refused"] else "absent (seeded)" if shot.seed is None else "taken (seeded)"
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


def moved_by_a_gap(found: Followed, named_value: bool) -> tuple[float, float] | None:
    """The two scores the automatic line was put between, (the lower, the higher), when it stands above its floor.

    `auto_match` moves the line to the middle of the widest gap in the shots' scores ON THEIR FIRST LOOK, and those
    are the scores returned here (`Followed.gap`), not the shots' final bests: a refused shot is looked at again on
    every stride frame and its best can rise, even past the line (mrcorn's cold read, 2026-10-10). A gap is right
    when the shots under it show somebody else. When the subject is in both, it lies between two right answers: on
    2026-10-10 a subject at 0.80 was refused under a line of 0.85 set between her own two shots
    (`bench/results/2026-10-10_who_is_who_across_shots.md`). None when the value was named, or the floor stands.
    """
    return None if named_value or found.pick_frame is None else found.gap


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
        elif found.pick_seed == "is the pick":
            lines.append(f"subject: the `{phrase}` a seed names on frame {found.pick_frame}; every other shot is judged against them"
                         + ("" if found.rule_pick is None else
                            f". The pick rule's own choice ({pick}, frame {found.rule_pick[0]}) was somebody else or another frame and is not used"))
        else:
            lines.append(f"subject: the {pick} `{phrase}` on frame {found.pick_frame} ({how})")
        if found.pick_seed == "took nobody":
            lines.append(f"THE PICKED SHOT'S SEED TOOK NOBODY, so the other shots were judged against the pick rule's person on frame "
                         f"{found.pick_frame}, whom this mask does not hold in that shot. Seed the other shots too, or name the pick")
        if found.pick_seed == "body boxes only":
            lines.append(f"THE PICKED SHOT'S BOXES ARE ROUND HER BODY ONLY, never a reference, so the other shots were judged against "
                         f"the pick rule's person on frame {found.pick_frame}, who may not be the person boxed. Seed the other shots too, "
                         "or box her head on this one")
        if found.pick_seed == "names no detection":
            lines.append(f"THE PICKED SHOT'S SEEDS NAME NO DETECTION (each was taken by segmenting inside its box), so the other shots "
                         f"were judged against the pick rule's person on frame {found.pick_frame}, who may not be the person boxed. "
                         "Seed the other shots too, or name the pick")
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
        gap = moved_by_a_gap(found, named_value)
        if gap is not None:
            under = [(n, s) for n, s in enumerate(shots, 1) if not s.picked and abs(s.first - gap[0]) < 1e-9]
            which = ", ".join(f"shot {n}, whose best over the whole shot is {s.best:.2f}" + ("" if s.seed is None else " and which was taken")
                              for n, s in under)
            lines.append(f"the automatic line is above its floor ({found.floor:.2f}): it was put in the gap between two shots' "
                         f"scores on their first look, {gap[0]:.2f} and {gap[1]:.2f}. The lower is " + (which or "a shot")
                         + ("; on its first look it was over the floor, so the gap alone refused it there" if gap[0] >= found.floor else
                            "; on its first look it was under the floor as well")
                         + ". If the subject is in both shots, look at the lower one's tile and correct it or name the value")
    for n, s in enumerate(shots, 1):
        span = f"[{n}] frames {s.start}-{s.end - 1}"
        if s.seeded_all:
            each = "; ".join(_seed_said(given) + ("" if given["refused"] else
                                                  f", tracked over {given['first_frame']}-{given['last_frame']} and kept on "
                                                  + (", ".join(f"{a}-{b}" for a, b in given["kept_on"]) or "no frame"))
                             for given in s.seeded_all)
            lines.append(f"{span}: seeded from {len(s.seeded_all)} boxes, each tracked out to its neighbours' frames and kept frame "
                         f"by frame where the tracker's own score holds her best: {each}"
                         + (f". THE BOXES' TRACKS HELD DIFFERENT PEOPLE on {_frame_list(s.disagreed)}: left empty there" if s.disagreed else "")
                         + (_regained(s) if s.seed is not None else ". EVERY SEED WAS REFUSED: the shot is left empty"))
        elif s.seeded:
            given = s.seeded
            if given["box"] is None:
                lines.append(f"{span}: seeded on load frame {given['frame']}: nobody")
            elif given["refused"]:
                lines.append(f"{span}: {_seed_said(given)}. The shot is left empty")
            elif given.get("how", SEED_BY_BOX) == SEED_BY_BOX:
                lines.append(f"{span}: seeded on load frame {given['frame']}: took person {given['person']} of the {s.candidates} "
                             f"detection(s) there, box overlap {given['overlap']:.2f}"
                             + ("; a box round her body: her person track only, never her head" if given.get("encloses") == SEED_BODY else "")
                             + _regained(s))
            else:
                lines.append(f"{span}: {_seed_said(given)}, of the {s.candidates} detection(s) there{_regained(s)}")
        elif s.corrected:
            lines.append(f"{span}: corrected by hand, {s.corrected} of the {s.candidates} detection(s) on frame {s.shown}"
                         f"{_regained(s)}" if s.seed is not None else f"{span}: corrected by hand, nobody taken")
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


def tile_label(number: int, shot: Shot, joined: dict[int, int] | None = None, unsure: dict[int, int] | None = None) -> str:
    """A tile's header: the shot, its frames, its score, its state, and how many duplicate detections its frame had.

    The last is said here because the header is where a person typing a correction looks, and a join can be wrong
    (`subject_tracks.one_each`, what it cannot tell): a person missing from the numbers has an explanation in sight.
    """
    score = "" if shot.picked or shot.best < 0 else f"  {shot.best:.2f}"
    if shot.seeded:
        # the automatic pass's best likeness in this shot is somebody the seed may not have named: not printed beside
        # a seeded state. A seed's own figure is how well its box and the detection agree
        score = "" if shot.seeded["overlap"] is None else f"  box {shot.seeded['overlap']:.2f}"
        if shot.seeded_all:
            score = f"  {sum(not g['refused'] for g in shot.seeded_all)} of {len(shot.seeded_all)} boxes"
        elif shot.seeded.get("how") in (SEED_BY_SEGMENT, SEED_BY_PROMPT):
            score = "  segmented in its box"
        elif shot.seeded.get("how") == SEED_BY_INSIDE:
            score = f"  drawn box {shot.seeded['inside']:.2f} inside"
    extra = int((joined or {}).get(int(shot.shown), 0))
    doubt = int((unsure or {}).get(int(shot.shown), 0))
    return (f"{number}  frames {shot.start}-{shot.end - 1}{score}  {_state(shot)}"
            + (f"  ({extra} duplicate{'s' if extra != 1 else ''} joined)" if extra else "")
            + (f"  ({doubt} may be one, not joined)" if doubt else ""))


def preview(frames: torch.Tensor, mask: torch.Tensor, shots: list[Shot], detect, joined: dict[int, int] | None = None,
            unsure: dict[int, int] | None = None, held=None) -> torch.Tensor:
    """One labelled frame per shot, [shots, h, TILE_WIDTH, 3].

    The frame is the one the shot was taken on, or where its best candidate
    was seen. The subject's mask is tinted. Every detection there is outlined:
    green for the one taken, orange for the best candidate of an absent shot,
    white for the rest. Each outline carries its person number at the top,
    the number the shot table and a correction use (`shot_table.person_order`:
    left to right). The label gives the shot, its frames, its score and
    whether it was picked, taken or absent, and how many detections on that
    frame were joined with another as the same thing (`tile_label`).
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
        text = tile_label(n, s, joined, unsure)
        draw, font = ImageDraw.Draw(pil), _font(22)
        box = draw.textbbox((8, 6), text, font=font)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        label_bottom, label_right = box[3] + 6, box[2] + 8
        subject = shot_table.person_number(found, s.index)
        # a person another tracker holds keeps their number and says so: one numbering on every tracker's tile
        theirs = held(int(s.shown)) if held is not None else frozenset()
        marked = {n for n, index in enumerate(shot_table.person_order(found), 1) if index in theirs}
        for number, col, row in shot_table.label_points(found):
            name = f"{number} held" if number in marked else str(number)
            x, y = col * TILE_WIDTH / width, row * th / height + 3
            nb = draw.textbbox((0, 0), name, font=font)
            w, h = nb[2] - nb[0] + 10, nb[3] - nb[1] + 8
            x = min(max(x - w / 2, 0), TILE_WIDTH - w)
            if y < label_bottom and x < label_right:      # clear of the shot's own label
                y = label_bottom
            y = min(y, th - h)
            ink = (255, 255, 255) if number != subject else ((40, 255, 60) if s.seed is not None else (255, 160, 0))
            draw.rectangle((x, y, x + w, y + h), fill=(0, 0, 0))
            draw.text((x + 5 - nb[0], y + 4 - nb[1]), name, fill=ink, font=font)
        tiles.append(torch.from_numpy(np.asarray(pil).copy()).to(torch.float32) / 255.0)
    return torch.stack(tiles, dim=0) if tiles else torch.zeros((1, th, TILE_WIDTH, 3))


def _sam_callables(segmenter, segmenter_clip, frames: torch.Tensor, phrase: str, detection_threshold: float,
                   max_people: int = MAX_PEOPLE, head_phrase: str = HEAD_PHRASE,
                   track_scores: dict | None = None):
    """`detect`, `sign` and `track` on core's SAM 3: its detect and track nodes, and the model's vision trunk.

    `track_scores`, when given a dict, is filled as tracks are made: `track_scores[frame]` is the tracker's own
    object score on that frame of the clip (`ScoreTap`), or None where it could not be read, and
    `track_scores["trouble"]` is a list of why, one line per run that gave none. A later track over the same
    frames replaces the earlier one's scores, as it replaces its mask.
    """
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

    def trunk_of(f: int) -> torch.Tensor:
        if f not in features:
            comfy.model_management.load_model_gpu(segmenter)
            device = comfy.model_management.get_torch_device()
            dtype = segmenter.model.get_dtype()
            x = comfy.utils.common_upscale(frames[f:f + 1, ..., :3].movedim(-1, 1), TRUNK_SIDE, TRUNK_SIDE,
                                           "bilinear", crop="disabled").to(device=device, dtype=dtype)
            trunk = segmenter.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x)
            trunk = trunk[-1] if isinstance(trunk, (list, tuple)) else trunk
            features[f] = trunk[0].to(torch.float32).cpu()
        return features[f]

    def whole(f: int, mask: torch.Tensor):
        """The whole mask's signature. Kept for the shot table and compared by nobody here (`follow`, `sign_whole`)."""
        return signature(trunk_of(f), mask)

    def head_on(f: int, mask: torch.Tensor):
        if f not in head_masks:
            out = SAM3_Detect.execute(segmenter, frames[f:f + 1], conditioning=head_cond,
                                      threshold=float(detection_threshold), individual_masks=True)
            head_masks[f] = getattr(out, "args", out)[0].to(torch.float32).cpu()
        return head_of(mask, head_masks[f])

    def sign(f: int, mask: torch.Tensor):
        trunk_of(f)
        head = head_on(f, mask)
        return signature(features[f], top_third(mask)), (None if head is None else signature(features[f], head))

    sign.whole = whole
    sign.head = head_on

    boxed: dict = {}

    def in_box(f: int, box) -> torch.Tensor | None:
        """What SAM 3 segments inside a box on frame `f` from the box alone: core's `SAM3_Detect` with `bboxes` and no
        text, its box-prompt path (`comfy_extras/nodes_sam3.py`, "Box prompts: SAM decoder path"). [H, W], or None when it
        returns no mask. Read once a frame and box. `name_seed` clips it to the box and takes out other trackers' people."""
        key = (int(f), tuple(round(float(v), 3) for v in box))
        if key not in boxed:
            out = SAM3_Detect.execute(segmenter, frames[f:f + 1], conditioning=None,
                                      bboxes={"x": float(box[0]), "y": float(box[1]), "width": float(box[2]), "height": float(box[3])},
                                      threshold=float(detection_threshold), individual_masks=True)
            masks = getattr(out, "args", out)[0].to(torch.float32).cpu()
            boxed[key] = masks[0] if masks.ndim == 3 and int(masks.shape[0]) else None
        return boxed[key]

    sign.in_box = in_box

    def run(images: torch.Tensor, mask: torch.Tensor, first: int, backwards: bool = False):
        """Track `images` from `mask` on their first frame. `first` is the clip frame of the lowest-numbered one.

        (masks, the tracker's own score per frame in clip order, why there is none or "")."""
        with ScoreTap(getattr(getattr(segmenter.model, "diffusion_model", None), "tracker", None)) as tap:
            out = SAM3_VideoTrack.execute(images, segmenter, initial_mask=mask[None].to(torch.float32),
                                          conditioning=None, detection_threshold=float(detection_threshold),
                                          max_objects=1, detect_interval=1)
        count = int(images.shape[0])
        values, why = scores_for(tap.seen, tap.trouble, count, backwards=backwards)
        # The frame a run is seeded on is GIVEN its mask, and the score the tracker returns there is the constant
        # it writes for a frame it was told about, not a judgement. Written as null, or a reader would see
        # "present, high" on exactly the frames a track or a re-find was seeded (mrcorn's cold read).
        values[-1 if backwards else 0] = None
        data = getattr(out, "args", out)[0]
        masks = SAM3_TrackToMask.execute(data, "")
        return (getattr(masks, "args", masks)[0].to(torch.float32).cpu(), values,
                f"frames {first}-{first + count - 1}: {why}" if why else "")

    def scored(start: int, end: int, seed: int, mask: torch.Tensor):
        """`track` without writing the record: (masks [end - start], score per frame of `[start, end)`, [why none])."""
        forward, f_values, f_why = run(frames[seed:end], mask, seed)
        if seed == start:
            return forward, f_values, [w for w in (f_why,) if w]
        backward, b_values, b_why = run(torch.flip(frames[start:seed + 1], dims=[0]), mask, start, backwards=True)
        # the seed frame is the last of the backward run and the first of the forward one, null in both
        return (torch.cat([torch.flip(backward, dims=[0])[:-1], forward], dim=0), b_values[:-1] + f_values,
                [w for w in (f_why, b_why) if w])

    def keep_scores(first: int, values: list, trouble: list[str]) -> None:
        """Write a track's scores into the record over the frames it covers, replacing what an earlier track wrote there."""
        if track_scores is None:
            return
        for k in range(len(values)):          # this track replaces whatever was tracked over these frames before
            track_scores.pop(first + k, None)
        for k, value in enumerate(values):
            track_scores[first + k] = value
        for why in trouble:
            track_scores.setdefault("trouble", []).append(why)

    def track(start: int, end: int, seed: int, mask: torch.Tensor, record_scores: bool = True) -> torch.Tensor:
        masks, values, trouble = scored(start, end, seed, mask)
        if record_scores:
            keep_scores(start, values, trouble)
        return masks

    track.without_scores = lambda start, end, seed, mask: track(start, end, seed, mask, record_scores=False)
    # for several seeds of one shot, kept frame by frame by these scores (`merge_tracks`) and recorded once, merged
    track.scored = scored
    track.keep_scores = keep_scores

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
                # the default is `most central` since 2026-10-07 (the owner): on the three windows tested
                # `largest` and `most central` were different detections, and `most central` the person
                # meant (docs/wiki/decisions.md, 2026-10-07, which says where the evidence is)
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
                # appended 2026-10-10
                io.Mask.Input("others", optional=True,
                              tooltip=("Optional. The people other Subject Track nodes already follow, one mask per "
                                       "frame: wire their `mask` here (several joined into one). A person lying "
                                       "mostly on it is never taken as this node's subject, so two nodes cannot "
                                       "follow the same person. Such a person stays outlined and numbered on the "
                                       "preview, marked `held`. A shot where only those people are on screen has "
                                       "no subject.")),
                io.String.Input("seeds", default="", multiline=True, optional=True,
                                tooltip=("Optional. Tells the tracker which person to follow, by a box on a frame of "
                                         "this load. Written by the analysis, or by hand. One line a box:\n"
                                         "frame 410: 543, 253, 157, 186   (the box's left, top, width, height)\n"
                                         "frame 412: 540, 300, 60, 90 body   (a box round their body, not their head: "
                                         "it seeds the person, never the head)\n"
                                         "frame 414: 500, 280, 90, 120 drawn   (a box drawn by hand, approximate: it names "
                                         "the one person lying mostly inside it, and is refused across two)\n"
                                         "frame 439: none   (nobody in that shot)\n"
                                         "job: 2 other subjects   (written by the tools: a box prompt is then refused on a "
                                         "frame where this tracker knows none of them)\n"
                                         "head: per stretch: WHY   (written by the tools: the head tracked inside each stretch "
                                         "the person track holds; without it the head track is as it always was)\n"
                                         "The frame counts from 0 in the frames given to this node, not in the source "
                                         "video. A shot may have boxes on several frames: each is tracked from, and the "
                                         "track the tracker scores best is kept frame by frame. A box no detected person "
                                         "matches takes the person whose outline overlaps it, else what SAM 3 finds inside "
                                         "it; one that takes nobody is refused and the report says why. A seed in the shot "
                                         "the subject is picked in is the pick: every other shot is then matched against "
                                         "the person it names. A shot cannot have both a seed and a correction, or `none` "
                                         "beside a box.")),
                # appended 2026-10-18
                io.Boolean.Input("track_head", default=TRACK_HEAD, optional=True,
                                 tooltip=("When on (the default), also propagates the subject's head mask in each shot "
                                          "from the head detection on that shot's seed frame, clipped to the person "
                                          "track, and emits it on `head_mask`.")),
            ],
            outputs=[
                io.Mask.Output(display_name="mask", tooltip="One mask per frame at the frames' size; empty where the subject is absent."),
                io.Image.Output(display_name="preview", tooltip="One labelled frame per shot, each person's outline numbered."),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="shot_table",
                                 tooltip="The shots as a table, in JSON: who was found in each, numbered as on the preview, "
                                         "who was taken and why. A Save Shot Table node writes it out for review."),
                io.Mask.Output(display_name="head_mask",
                               tooltip="When `track_head` is on, the subject's propagated head mask per frame, clipped "
                                       "to `mask`; empty when `track_head` is off."),
            ],
        )

    @classmethod
    def execute(cls, frames, segmenter, segmenter_clip, pick_on, match, cuts, subject_phrase=SUBJECT_PHRASE,
                pick=PICK_CENTRAL, detection_threshold=DETECTION_THRESHOLD, max_people=MAX_PEOPLE,
                head_phrase=HEAD_PHRASE, corrections="", subject_from="", others=None, seeds="",
                track_head=TRACK_HEAD) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        if others is not None:
            if others.ndim == 4 and int(others.shape[-1]) == 1:
                others = others[..., 0]
            if others.ndim != 3 or tuple(others.shape) != tuple(frames.shape[:3]):
                raise ValueError(f"others must be one mask per frame at the frames' size, {tuple(frames.shape[:3])}; got "
                                 f"{tuple(others.shape)}: wire the `mask` of a Subject Track that was given the same frames")
            others = (others > 0.5).cpu()
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
        seeded = parse_seeds(seeds, shot_ranges(n, found_cuts))
        other_boxes = parse_other_boxes(seeds, shot_ranges(n, found_cuts))
        job_others = parse_job_others(seeds)
        head_mode, head_mode_why = parse_head_mode(seeds)
        handed = shot_table.gallery_from(subject_from) if str(subject_from or "").strip() else None
        track_scores: dict = {}
        detect, sign, track = _sam_callables(segmenter, segmenter_clip, frames, subject_phrase, detection_threshold,
                                             int(max_people), head_phrase, track_scores=track_scores)
        joined: dict[int, int] = {}
        parts: dict[int, float] = {}
        unsure: dict[int, int] = {}
        detect = one_detection_each(detect, joined, parts=parts, unsure=unsure)
        # after the join, as it always was: the mark is read on the detection that stands for a person
        held = held_people(detect, others) if others is not None else None
        with torch.no_grad():
            found = follow(n, found_cuts, pick, int(pick_frame) if named_frame else None,
                           float(match_threshold) if named_value else None, detect, sign, track, corrections=by_hand,
                           gallery=handed, sign_whole=getattr(sign, "whole", None), seeds=seeded,
                           track_head=bool(track_head), other_boxes=other_boxes, job_others=job_others,
                           head_per_stretch=head_mode == HEAD_PER_STRETCH, held=held)
        # Stray specks of the tracker's go before anything reads the mask: the tiles, the shot table and every node
        # after this one widen what they are given (`subject_tracks.drop_specks`).
        # In place: `assemble` returns a tensor nothing else holds, and a copy of a clip's mask is gigabytes.
        mask, speck_frames, speck_pixels = subject_tracks.drop_specks(assemble(n, h, w, found.pieces), in_place=True)
        text = report(found, found_cuts, pick, subject_phrase, named_frame, named_value, time.perf_counter() - began,
                      cutting="\n".join(filter(None, [cuts_line(steps, cut_at, named_cut), borders_line(seen, w, h)])))
        if speck_frames:
            text += (f"\nstray specks removed from the mask: {speck_pixels} px on {len(speck_frames)} frame(s), "
                     f"each tiny beside the subject and away from them: {_frame_list(speck_frames)}")
        if joined_report(found.shots, joined, parts, unsure):
            text += "\n" + joined_report(found.shots, joined, parts, unsure)
        on_others = None
        if others is not None:
            on_others = frames_on_others(mask, others)
            also_held = {f for s in found.shots if s.also_held for f in range(s.start, s.end)}
            left = [f for f in on_others if f not in also_held]
            if left:
                mask[left] = 0.0
            on_others = sorted(set(found.on_others).union(left))
            text += "\n" + others_report(found.shots, held.seen, on_others)
        if track_head:
            head_raw, _, _ = subject_tracks.drop_specks(
                assemble(n, h, w, found.head_pieces), in_place=True,
            )
            head_mask = ((head_raw > 0.5) & (mask > 0.5)).to(torch.float32)
            head_frames = int((head_mask > 0.5).flatten(1).any(dim=1).sum())
            text += f"\nhead track (track_head=True): {head_frames} of {n} frame(s)"
        else:
            head_mask = torch.zeros((n, h, w), dtype=torch.float32)
            head_frames = 0
            text += "\nhead track (track_head=False): off"
        trouble = track_scores.get("trouble") or []
        if trouble:
            text += ("\nthe tracker's own score per frame could not be read on " + f"{len(trouble)} run(s), and is left "
                     "out of the shot table there: " + "; ".join(trouble[:4]) + (" and more" if len(trouble) > 4 else ""))
        logger.info("[h3] MiniMaxH3SubjectTrack: %s", text.replace("\n", "; "))
        tiles = preview(frames, mask, found.shots, detect, joined, unsure, held)
        table = shot_table.build(found, detect, mask, state=_state, phrase=subject_phrase, pick=pick,
                                 named_frame=named_frame, named_value=named_value, cuts=found_cuts,
                                 held=held, on_others=on_others,
                                 track_scores=track_scores, joined=joined, gap=moved_by_a_gap(found, named_value),
                                 parts=parts, unsure=unsure)
        table["track_head"] = bool(track_head)
        # WHICH CODE RAN, taken in this process when the module was imported (`RAN_CODE`): the observable for what the
        # server runs, where `bench/_lib/code_stamp.py` reads the file on disk
        table["ran_code"] = dict(RAN_CODE)
        # which head track ran: main's span head track (the default) or per stretch (`head: per stretch: WHY`), with the why
        table["head_mode"] = head_mode
        if head_mode == HEAD_PER_STRETCH:
            table["head_mode_why"] = head_mode_why
        table["head_frames"] = head_frames
        table_json = shot_table.as_json(table)
        if len(table_json) > TABLE_SAID_OVER:
            # the signatures of every shot's people are most of it (mrcorn's cold read: nobody should meet it first in a browser)
            text += (f"\nthe shot table is {len(table_json) / 1e6:.1f} MB as text: it carries the signatures of the people on "
                     "every shot's shown frame")
        both = text + "\n\n" + shot_table.as_text(table)
        shown = {**ui.PreviewImage(tiles, cls=cls).as_dict(), **ui.PreviewText(both).as_dict()}
        return io.NodeOutput(mask, tiles, text, table_json, head_mask, ui=shown)


def ran_code(path=None) -> dict:
    """The code this process runs, for the shot table (`ran_code`): this module's file and the sha256 of its bytes as
    they were read when it was imported. Nothing else showed which `subject_track.py` the server ran:
    `bench/_lib/code_stamp.py` reads the file on disk, which after an edit or a pull is the new file whatever the
    server imported. After a restart a session holds the preview's `ran_code.sha256` to `sha256sum subject_track.py` at HEAD."""
    from pathlib import Path
    path = Path(path or __file__)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


#: Taken once, when the module is imported in the server: what the file held then, not what it holds now.
RAN_CODE = ran_code()
