"""A source video and a subject mask, for masked video-to-video.

The owner's case (2026-10-04): take a music video, keep its audio and
everything outside one subject, and regenerate that subject from a reference
still. Core already has the mechanism. A latent `noise_mask` is a per-token
timestep, not an attention mask: `comfy/ldm/minimax/model.py::mask_row_values`
pools it to the DiT's 2x2 patch grid and `MiniMaxH3Model._forward` pins a
preserved token at the conditioning timestep, with the clean latent
re-injected every step (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`).
This module supplies what core leaves to the graph: getting a per-frame pixel
mask onto that grid without losing the subject, and putting the untouched
pixels back afterwards.

**Why not stock `SetLatentNoiseMask`.** It writes a flat mask, so the sampler
pads ones for the audio stream and the track regenerates
(`audio_freeze.py`, "Trap this node guards"). And core resizes a mask with
trilinear interpolation (`comfy/utils.py::reshape_mask`), which samples it:
the video VAE packs frames in runs of `FRAME_PER_TOKEN`, and a subject that
moves inside a run can fall between samples and stay in the plate.

**The reduction** (`token_mask`), every step a max so no subject pixel is
dropped: grow in pixels, max-pool to the latent cells, max over each temporal
run, then max over each 2x2 patch, written back onto the cells. The last step
is core's own (`mask_row_values`), done here so that the region this module
reports, the region the sampler blends and the region the composite restores
are the same set of tokens. Binary on purpose: a fractional value is a
partial-strength row, which is a different experiment.

**The composite** (`pixel_alpha`, `composite`). A preserved token does not
decode back to the source exactly: the VAE round trip and the pinned-timestep
blend both move it (`bench/results/2026-09-25_distill_audio_s1.md` is the
record for a wholly frozen video). So outside the regenerated tokens the
source's own pixels go back, with the boundary feathered. `grow_pixels` is
what keeps the feather on background: the blend reaches `feather_pixels` into
the regenerated region, and the subject sits at least `grow_pixels` inside it.
`lay_window` is the whole of it as the song node runs it (the weight, the cut
gate, the blend), and `save_window_region` writes what it was run with beside
each window, so a finished render can be laid again off the server.

**The margin's size** (`grow_by`, 2026-10-08). `a fixed margin` is
`grow_pixels` on every frame. `the subject's size` takes it per frame from the
mask's own area (`margins`: a share of its square root, steadied over time by
a running median, capped at `grow_pixels`, never under `feather_pixels`), for
a subject who is small in the frame: a fixed margin there is several times
their own area and the region holds the people round them. Every place that
widens a source's mask reads `source_margins`, half of it where half was read
before, so the region, the preview, the report and the composite move
together. The motion reference's widening does not follow: it stays half of
`grow_pixels` on every frame (`motion_widening` says why). The token grid still
rounds the region out to whole tokens, which a small margin cannot go under.

**The people round the subject** (`others`, 2026-10-08). A mask of people who
are not the subject. The margin does not grow over them: a token that holds
any of their pixels and none of the subject's own mask stays the source's. A
token that holds any pixel of the subject's own mask still regenerates,
whoever else is in it. That is the difference from `keep`, which wins over the
subject too and is for a thing the subject holds: on a subject who is small,
most of their own mask lies in tokens they share with somebody, and `keep` on
a neighbour would hand those back to the source, original subject and all
(`bench/check_video_mask.py` item 14 holds the two apart). "The subject's own
mask" is the mask this node settled on, so under `the wired parts` it is the
part and not the person: a mask of everybody that includes the subject then
takes the margin off their own body beside the part, and the subject is to be
left out of it. Whether the rule should read the tracked subject as their own
is open. Why it exists: with two
people inside the region the model drew the new subject on the other one
(`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`, section 5).
Not rendered. Three rules since 2026-10-15 (the outside reviewer's G2, ruled by
the lead; `bench/results/2026-10-15_hair_flashes.md`), since a keep-out holes a
cell for the whole of its latent step: a piece of the others under one cell's
area on at most half of its step's frames is dropped before the cut
(`drop_specks`); a one-cell hole they punch with the region all round it, that
they hold briefly and do not fill, is closed (`close_lone_holes`); and a piece
of the region the keep-outs or `keep` cut off from the subject is given back
(`drop_islands`).

**What gets replaced** (`replace`, owner 2026-10-04: "it should be a choice on
a node"). `whole subject` regenerates the tracked person. `head and hair`
keeps the body below the hair, its clothes and its movement as the source's
own pixels, so nothing about the action has to be prompted: the body turns
because it is the original's. The part is found by core's SAM 3 detector on
the frames the subject is in, from phrases the user can read and change
(`part_phrases`), and kept only where it lies on the subject's own mask, so a
neighbour's hair is not taken.

The region is **everything of the subject down to where the part ends**, not
the part's own outline. The first render regenerated the outline alone (head
plus a mass of long hair), and the model filled it with a head that size: "a
giant bobblehead" (owner). Nothing in a head-shaped hole pins the head's
scale. Drawing the neck and shoulders with it does, because they have to
meet the kept torso (a peer session's suggestion,
`docs/research/masking/2026-10-04_mrhf.md`). With short hair the region is
little more than the head.

A frame the detector misses would otherwise regenerate nothing and show the
original's head, so a miss takes the nearest found frame's cut, and the log
says how many did. The node's `mask` output is the mask it used, for a
preview.

**Painting the subject out before the encode** (`paint_out`; a peer session's
finding, `docs/research/masking/2026-10-04_mrhf.md`). The video VAE's encoder
is convolutional, so a kept token beside the mask was encoded with the old
subject inside its receptive field, and those tokens are re-injected clean on
every step. A faint remnant of the original measured beside the regenerated
subject is the suspected result. With `paint_out` the subject's pixels are
filled from their surroundings (`fill_subject`) in the frames that are
encoded; the composite still restores the true source. Core's own H3 inpaint
path hides the original before its encode too
(`comfy_extras/nodes_minimax_h3.py::MiniMaxH3FunControlPatch`). Off by
default until a matched render says it helps.

**What the composite keeps** (`composite`; owner's notes on the first clip
and a peer's measurements and design, `docs/research/masking/2026-10-04_mrhf.md`).
The regenerated tokens hold three kinds of pixel: the new subject, where the
old subject stood, and margin that is neither. The margin is background the
model had to invent, and it is where the flicker and the cut-out look were:
the grown border, the backdrop a moving original swept, and anything the
tracker marked by mistake. `whole region` keeps all three from the render.
`only what changed` keeps the render where it differs from the source by
more than `change_threshold` (the new subject and its shadow; the old
subject's place, which is also forced by its mask) and restores the source
where the model merely repainted the backdrop. No second detector pass and
no phrase: a neighbour the model reproduced faithfully simply stays source.
Then a generous `grow_pixels` costs nothing visible, so a replacement that
reaches past the original's outline is not cut off. It does not fix where
the old subject stood and the new one does not: that needs a plate with
nobody in it.

**Painting out, as built, made things worse** (owner, on the proof render:
"much worse than before"). It stays off. The likely reason (inferred): the
original's trace in the kept tokens also tells the model where the subject
is and how big, and a hole smaller than the original's hair leaves a fringe
the model continues as dark lines.

**A motion reference** (`motion_reference`, 2026-10-05). The per-token
mechanism carries no movement: the replaced subject faces the camera while
the original turns, and every way of putting the source into the target rows
(a late start, plain or softened) brought the original's look with its pose
(`bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`). The channel the
model was trained to take motion from is a video reference, so this node can
ask the song node to show each window to the model as `<Video 1>` as well:
the subject alone on grey, or the whole window, at a short edge that sets its
cost, with or without the video model's copy (`motion_vae`; off is the
encoder-only form, a few thousand text tokens per window). The song node
builds it per window from the source it already holds (`window_frames`,
`motion_reference`), so nothing is wired and no copy of the clip is kept. The
prompt names the relationship, never the action: the masking board's route
1, `docs/research/masking/2026-10-05_mryellow.md` section 7.

**A second wired video** (`motion_video_2`, 2026-10-12). Under `a video I
wire` the node takes a second video beside the first, with its own short
edge (`motion_short_edge_2`), and the song node shows it to the model as the
next video reference after the first: `<Video 2>` when the reference chain
holds no video of its own. It is for a load whose text gives each reference
a subject of its own, the way the vendor guide writes several references
(`docs/prompting.md`, section 9): a body for the movement and a closer view
of a face, say. It is cut window by window as the first is (`wired_motion`),
and `motion_vae` is one switch for both. A second video with no first is
refused. No render had been made with it when this was written.

**Zoomed in on the subject** (`subject only, zoomed in`, 2026-10-06). `subject
only` scales the whole frame down and greys the rest, so a subject that is
small in the frame is a handful of the encoder's tokens and most of the
picture is grey. This choice shows the same pixels in a box around the
subject: one fixed box per shot, so the subject still travels and moves
against a frame that holds still, and the framing changes only where the
source cuts. The box is taken from the TRACKED subject, not from what is
replaced: the source record now carries the subject's box per frame
(`subject_boxes`, from the tracker's mask before it is cut to a part),
because on a parts graph nothing downstream of this node knew where the whole
subject was. `motion_reference` states the rule that bounds the picture.

**What the wired parts cover** (`part_warning`, 2026-10-06). With `replace`
on `the wired parts` the region is the part node's mask, and a part model
that labelled someone else leaves a sliver of it on the subject: the original
then stays in the render and nothing said so. Before the part replaces the
mask, the node counts how much of the tracker's mask the part covers
(`part_coverage.py`), logs the figures, and puts the one-line warning in the
source record under `part_warning`, None when no frame is in doubt. The song
node's report and the prompt node's summary show it. It is a report: nothing
is refused and no mask changes.

The record this node returns, by key, beyond the node's own inputs: `frames`
and `mask` (the frames and the region used), `shot_table` and `replace` (read
by the prompt node), `subject_boxes` (the tracked subject's box per frame),
and `part_warning`.

**A mask is temporary data.** It is made by the preprocessing step for the
job that reads it, and nothing hands a mask from one run to another by a key
(the owner's rule, 2026-10-10: nothing an earlier run made is reused by a
key; a result is made again, or handed in explicitly as a named file or a
wired input; `AGENTS.md`, "Operative rules", and `docs/wiki/decisions.md`).
From 2026-10-04 this node kept its finished mask on disk (`reuse_mask`,
`mask_store.py`) under a key made from the upstream graph, a sample of the
frames and hand-bumped version numbers. The key could not see a change to
the code that makes a mask, a stale mask hid a fix on 2026-10-07 and the
store was switched off in code that day; it was removed with its input on
2026-10-10. A mask made once and used again is a file a person wires in.

**The shot table travels with the mask** (2026-10-05, `shot_table.py`). The
Subject Track's table says who was found and taken in each shot, and a
person reviews it in place of a render. Wired into `shot_table` here, it is
asked for with the mask (both are lazy and both come from the tracker) and
handed on in the source bundle, where the song node writes it beside the
video under the render's own number.

Design from two third-party nodes, read and not run:
`coderef/comfyui_dagthomas/nodes/h3/mouth_guard.py` (pixel grow, max-pool,
max per run; it protects where this regenerates) and
`coderef/ComfyUI-H3-Motion-Context-MultiRef/h3_v2v_fractional.py::_mask_to_video_latent`
(the per-run reduction). Neither feathers a composite.

**What stays the original inside the region** (`keep`, 2026-10-07). The
region is everything the mask and its margin cover, and a mask of one person
covers what that person holds. Under the noise a prop is gone: the model sees
it only in the motion reference, small and at two frames a second, and draws
what the prompt makes likely. An optional second mask names what to keep. It
is taken out of the token mask after the grow, in whole tokens (`window`),
so those tokens reach the model clean like the rest of the plate, and the
composite, which reads the same tokens, shows the source there. Reasoned,
not yet rendered.

Nothing here patches core.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

from .h3_rules import on_both_clocks
from .part_coverage import RECORD_KEY as PART_WARNING
from .part_coverage import coverage, summarise

logger = logging.getLogger(__name__)

H3MaskedSource = io.Custom("H3_MASKED_SOURCE")

#: Frames per chunk for the pixel-space pools and blends: bounds the transient
#: memory of a window. Reasoned, not measured.
CHUNK = 48

#: The `replace` choices. The second finds a part with core's SAM 3 detector.
REPLACE_WHOLE = "whole subject"
REPLACE_PART = "head and hair"
#: The third regenerates exactly the part node's mask (`sapiens2_parts.py`, wired on `parts`),
#: grown like any other mask; head and hair keeps its top-down rule (mrorange, 2026-10-05).
REPLACE_PARTS = "the wired parts"
#: The default of `grow_pixels`. The owner's choice on one clip, 2026-10-04:
#: twice a DiT token of canvas "preserved identity better" than one, which
#: was the first, reasoned value. One seed each.
GROW_PIXELS = 64
#: The `grow_by` choices. The first is the node as it was: `grow_pixels` on every frame. The second takes the
#: margin from the subject's own size on each frame, with `grow_pixels` as its cap, because a margin fixed in
#: pixels is a thin border round a close subject and several times a small one's own area: the region then
#: holds the people round them (`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`, section 5;
#: the owner, 2026-10-07: "so we made the hole too big"). Not the default until a pair has been judged.
GROW_FIXED = "a fixed margin"
GROW_SUBJECT = "the subject's size"
GROW_BY = (GROW_FIXED, GROW_SUBJECT)
#: The `edge` choices: how fine the region's edge is. The first is the node as it was: a token is regenerated or
#: kept whole. The second hands the sampler the mask per latent cell, half a token's side. Core labels a token by
#: the most regenerated of its cells and puts the source back cell by cell (`comfy/model_base.py`,
#: `MiniMaxH3.scale_latent_inpaint` and `_token_grid_masks`; read 2026-10-09), so a kept cell inside a regenerated
#: token is the source's in the result. vllm-omni's mask editing does the same. Why it is here: rounding out to
#: whole tokens is what keeps the region at about twice a small subject's own area with no margin at all (the
#: masking board, finding mhi-03). Not the default until a pair has been judged (the owner, 2026-10-09).
EDGE_TOKENS = "whole tokens"
EDGE_CELLS = "latent cells"
EDGES = (EDGE_TOKENS, EDGE_CELLS)
#: Under `the subject's size`: the margin as a share of the square root of the mask's area on the frame.
#: Reasoned, 2026-10-08, not rendered: a little over `GROW_PIXELS` against that root on the close subject the
#: fixed margin was chosen on, so a subject that close keeps the margin it has today.
GROW_SHARE = 0.15
#: Frames to each side that the area's running median reads before the margin is taken from it (`steady`).
#: Reasoned: a median is not moved by outliers that number under half its window, so a part mask that collapses
#: on fewer than this many frames of any stretch that long cannot pull the margin down with it. On the clip this
#: was written for the frames a part report put in doubt stay under that in every such stretch, with little to
#: spare; the part node now holds those frames (`sapiens2_parts.py`), so the area seldom collapses there at all.
GROW_STEADY = 24
#: The `composite` choices. The second is the default: it is the composite of
#: the render the owner called the best (2026-10-04, one clip).
COMPOSITE_REGION = "whole region"
COMPOSITE_CHANGED = "only what changed"
#: The default of `change_threshold`: how far the render must differ from the
#: source, as a fraction of full scale averaged over the colour channels and a
#: small neighbourhood, to be kept. Reasoned from a peer's measurements on the
#: first clip (invented backdrop a few levels from the source, a subject tens
#: of levels), not tuned.
CHANGE_THRESHOLD = 0.05
#: The neighbourhood the difference is averaged over, in pixels to each side.
#: Reasoned, not measured.
CHANGE_BLUR = 4
#: The default of `part_phrases`: the union of the detections is the part.
#: "hair" as well as "head" because long hair lies on the chest and back,
#: outside any head. Reasoned, not measured.
PART_PHRASES = "hair, head"
#: The default of `part_margin`: a part is kept where it lies on the subject's
#: mask widened by this many pixels of the source frame, because the two masks
#: come from different passes and their edges do not coincide. Reasoned, not
#: measured.
PART_MARGIN = 8
#: The default of `part_threshold`, the detector's score threshold for a part.
#: Inherited: core's node default
#: (`comfy_extras/nodes_sam3.py::SAM3_Detect.define_schema`).
PART_THRESHOLD = 0.5
#: `start_from` choices: what the regenerated tokens start from.
#:
#: `noise` is the shipped render: the schedule runs from its first knot and
#: nothing of the source is under the mask. `the original's top, softened`
#: starts the schedule late, so that what is in the latent under the mask
#: shows through at the first step, and decides by token what that is: the
#: top of the subject keeps a grey, blurred copy of the source, which carried
#: which way the original faces on the one window it was tried on, and the
#: rest of the subject starts from nothing, so the original's clothes are not
#: there to be carried with it
#: (`bench/results/2026-10-04_masked_v2v_turn_soft_arms.md` is what led here:
#: every late start that held the whole subject changed the clothes).
START_NOISE = "noise"
START_TOP = "the original's top, softened"
#: The default of `start_top`: the share of the subject's height, from the
#: top, that keeps the softened source. Reasoned from one clip: the head and
#: shoulders of a standing figure, and short of where the band clip's
#: original has hair reaching his chest, since a softened hair mass is the
#: likeliest thing to bring long hair back. The same share with the body
#: filled in pixels gave half a turn in the record above; nothing was tried
#: with the body at zero. A detected head (the part menu) is the intended
#: second choice for this region and is not built.
START_TOP_SHARE = 0.3
#: The default of `start_blur`, pixels at the render's size. Seen on one
#: window, same record: at this blur the softened source still turned him on
#: both seeds tried, and at twice it no longer did.
START_BLUR = 16
#: The default of `start_knots`: how many knots late the schedule starts.
#: Seen, same record: two knots left a softened source as a flat shape.
START_KNOTS = 1
#: `motion_reference` choices: what of the source window, if anything, the
#: song node appends to the reference chain as a video, so the model is shown
#: the original's movement through the channel it was trained to take motion
#: from (`docs/research/masking/2026-10-05_mryellow.md`, section 2, route A).
#: Nothing in the per-token-timestep mechanism carries movement: the arms of
#: 2026-10-04 showed that whatever enters the target rows brings its look.
MOTION_NONE = "none"
MOTION_SUBJECT = "subject only"
MOTION_FRAME = "whole frame"
#: The subject alone, in a box around it and not in the whole frame, so the
#: encoder's tokens are spent on the subject. Owner, 2026-10-06, after a small
#: subject on a 16:9 canvas did not have its movement followed: zoom in.
#: `motion_reference` has the rule.
MOTION_ZOOM = "subject only, zoomed in"
#: A video the user wires, one frame per source frame, shown as the movement in place of anything cut from
#: the source: a body mesh, a pose, a map of a mouth. Owner, 2026-10-10: a signal can say how a person moves
#: without the text saying it, and without the original's look. On one shot, one seed, a body mesh given this
#: way drew an action the same text did not draw from the subject-on-grey reference. Appended, so the choices
#: before it keep their place.
MOTION_WIRED = "a video I wire"
#: Removed on 2026-10-10 with their code (`docs/wiki/decisions.md`): a sixth choice, `a video I wire, zoomed in`
#: (one render, no difference), and three inputs for the reference's look, a blur by a share of the box, a blur
#: in pixels and grey (reasoned, never rendered). A saved graph that names the sixth choice is refused by name.
MOTIONS = (MOTION_NONE, MOTION_SUBJECT, MOTION_ZOOM, MOTION_FRAME, MOTION_WIRED)
#: THE HELD TAIL (`held_mask`). A window given past its source's last frame holds that frame WITH ITS MASK, so
#: the region stays open and the model draws the new subject standing still there. Until 2026-10-10 the held
#: frames had an empty mask (a plate: the source's last frame with the ORIGINAL subject in it, shown clean
#: for the rest of the window), and for a day an input `held_tail` chose between the two. Measured that day
#: on three loads of one clip, one seed each (`docs/wiki/window_context.md`): under the plate a short load
#: drew the still's person on its first frames and the original on its last; with the region held open the
#: fall at the end was gone on every one, and the owner read the result as swapped on playback. The plate
#: and the input were removed the same day (`docs/wiki/decisions.md`). A held frame past the TRACK's end is
#: not written; one inside the track (a track longer than the source, usually audio that was not cut to the
#: load) is written, and the song node's report says how many.
#: Where the region is rendered (`render_region`, 2026-10-10). `the whole frame` is the render as it has been:
#: the source fitted to the canvas. The zoomed choice crops the load to ONE box round what is replaced, scales
#: that box to the canvas, runs the whole render on it as if it were the clip, and lays the region back into
#: the whole frame (`zoomed_source`, `lay_back`). Cut down on 2026-10-10 to what its one render used, until
#: the owner rules on that render (`docs/wiki/decisions.md`): a second choice round the tracked subject, an
#: input for the largest zoom, the rules that let a zoomed run continue another, and the cut of a wired motion
#: video to the box are removed. From then until 2026-10-13 a zoomed run with a wired motion video was refused;
#: since then it takes the video AS IT IS WIRED, not cut to the box (`zoomed_source` has what that does and does
#: not do; the owner's go is the Open List's row p04, answered 2026-10-12). Why a zoom at all: a subject small in the
#: frame is a few of the model's tokens. The evidence when written is one clip and mixed (the owner on
#: playback, 2026-10-10): a face part about 94 px tall on the canvas did not read as replaced where one about
#: 164 px had, and the same small face then did read as replaced once its load no longer faded at the end
#: (the held tail, above); so how much is size is not settled, and smaller subjects than that one are in the clip.
#: The zoom buys tokens while the region is drawn; the window's video is
#: still the canvas, so it does not buy delivered pixels there. The laid picture in the zoomed space is
#: written beside each window for a delivery built at the source's size. Untested on a render when written.
RENDER_WHOLE = "the whole frame"
RENDER_ZOOM_REPLACED = "zoomed in on what is replaced"
RENDER_REGIONS = (RENDER_WHOLE, RENDER_ZOOM_REPLACED)
#: WHAT THE CONTEXT FRAMES SHOW (`context_shows`, 2026-10-22). A one-window load shorter than its window is given the
#: source's real frames after it (before it only at the clip's end; `workflows/masked_job_graph.py::window_pad`, which
#: has why), tossed after the decode.
#: `as filmed` (the default, and the only behavior from 2026-10-21 until this input): nothing is regenerated on them,
#: so the original subject is in the window unmasked there, which helped the second subject (her real lowered gaze
#: beside the shot; the owner's verdicts of 2026-10-22). `face covered`: the subject's region is held open on them,
#: the shot's first frame's mask over the frames before and its last frame's over the frames after (`pad_context`),
#: grown by the margin as on every frame, so the original's face is not in the frozen context. Reasoned from the beard
#: rule (`docs/masked_v2v/AGENTS.md` §2: 3-4 frames of unmasked beard in a window pull the beard onto the new face)
#: and from the held tail above (a region held open on frames past the shot is the node's measured choice there);
#: not yet rendered when written. The keep-outs (`keep`, `others`) stay empty on those frames in both forms: `keep`
#: holds the source's own pixels inside the region and would show the face there, and an `others` held from the edge
#: frame could cut the subject's real face out of the held region where it has moved. More regenerated on a frame
#: that is tossed costs no laid pixel. A plan's subject chooses it (`subjects.<s>.context`, `bench/shot_to_load.py`).
#: CONTEXT ACROSS A CUT IS HELD (`hold_context`, 2026-10-22). Of those context frames, the ones that lie across a cut
#: from the shot (the builder knows the shot: `workflows/masked_job_graph.py::held_context`, carried in this node's
#: `_meta` as `held_before` and `held_after`, the outermost frames of each side) are copies of the shot's own edge frame
#: WITH that frame's masks (the subject's, the tracked one, `keep` and `others`), as a short window had before 2026-10-21
#: (the held tail, above), whatever `context_shows` says. Measured on one load (`second_hfp_4479c`, 2026-10-22): with
#: real frames of the neighbouring shots in its window, the box's brightness tracked the source on every frame but the
#: shot's end, where it rose above it, a seam at the cut; H3's latent steps each span several frames, so the
#: shot's edge frame shared a step with the other shot. The track under held frames is silence (the song node), as it
#: was past a track's end. A graph built before this carries neither key, and its context frames stay real.
#: THE GUARD ON A HELD SIDE (`held_edge_falls_back`, the owner's choice of 2026-10-22; updated 2026-10-24 to step back
#: to the nearest good shot frame). When a held side's edge frame has no region, or one under this share of the shot's
#: own middle region (the median area of its frames with a mask, read from the load's own mask, up front), that side's
#: held context frames hold the nearest frame inside the shot whose region is at least this share of the middle (leaving
#: the kept shot frames untouched), or fall back to real frames only when the whole shot has no mask, and the record
#: says which side and which shot frame was held (`held_fell_back`). Why: held copies of an edge frame with no region
#: repeat the original subject untouched over every held frame; on second_hfp_4912e the region was 0 px on 4946 beside
#: 9147 px median, and stepping back to shot frame 30 (4942, 2958 px) holds a masked frame across the cut while keeping
#: 4946 itself as filmed (`docs/wiki/decisions.md`, 2026-10-21, 2026-10-24). REASONED, a starting value from that one
#: load: a quarter catches an empty or nearly empty edge and a subject mostly hidden there, and leaves one who has only
#: turned or moved back.
HELD_EDGE_LEAST = 0.25
CONTEXT_AS_FILMED = "as filmed"
CONTEXT_FACE_COVERED = "face covered"
CONTEXT_SHOWS = (CONTEXT_AS_FILMED, CONTEXT_FACE_COVERED)
#: ZOOM TO A TARGET HEAD SIZE, SWITCHED OFF BY DEFAULT (the owner's go of 2026-10-22, and the owner's word the same day
#: to land it off and test it first: a load asks for it with `zoom_up_to`). A zoomed load's zoom is the least of three:
#: the box round the region with its room (`ZOOM_ROOM`), the load's `zoom_up_to` (`ZOOM_LARGEST` unless it asks), and
#: the zoom that takes the replaced region's height on the canvas (`region_height`) to `ZOOM_HEAD_TARGET`; except that
#: the target never holds a load under `ZOOM_LARGEST` where its box allows that. At the default `zoom_up_to`, which is
#: `ZOOM_LARGEST`, the target changes nothing: every load is zoomed as from 2026-10-10 (the box, or a cap of 2), which a
#: replay of every zoomed render graph under `data/` on 2026-10-22 gave to the pixel (80 zoomed, 31 on the whole frame;
#: `docs/wiki/decisions.md`). Asked up to `ZOOM_MOST_ASKED`, a load whose box held it under 2 is as it was
#: (second_hfp_0415n: 1.83, set by the box), one whose region reaches the target at 2 or less stays at 2
#: (second_hfp_1039e), and a region too small to reach it at 2 zooms on (second_hfp_4602d: 2.0, 3.28 asked up to 4,
#: set by the target). The region is what the load replaces: on a head-and-hair load the head (its median height
#: matched the body-parts model's hair and face height on the second subject's loads, 256 against 256 px on 0415n, 89
#: against 89 on 4479c; 5461e is the exception, 78 against 176), on a face-only load the face, which the target then
#: takes to the height a head would have had. The zoom enlarges whatever the node was handed: a graph whose loader
#: fits the clip to the canvas hands canvas pixels, and the plate the model sees is then those enlarged by the whole
#: zoom. Only the region is laid back. `set_by` in the record says which bound set it (`SET_BY_*`).
#:
#: The largest zoom a load takes unless it asks for more (`zoom_up_to`), and the zoom the target never holds a load
#: under. Reasoned when it was set (516e06e2, 2026-10-10; the lead's arithmetic on one case): about twice takes the
#: face that failed to the size of the one that was accepted. Every zoomed load to 2026-10-22 was made under it.
ZOOM_LARGEST = 2.0
#: The most a load may ask for with `zoom_up_to`. Reasoned: 4 times the canvas the loader hands is about 2.9 times
#: the film's own pixels on fun (1424x1068 fitted to 1024x768: 4 x 1024 / 1424); the plate round the region is that
#: film enlarged bilinearly, softer than anything a delivery shows, and the model draws the region to sit in that
#: plate, so past it a head gains tokens and may lose detail to the soft plate. The smallest region of the second
#: subject's 61 renders measured on 2026-10-22 (second_hfp_1419c, 48 px at 1x) reaches about 190 canvas px at 4.
#: Nothing was rendered past 2 when written.
ZOOM_MOST_ASKED = 4.0
#: The replaced region's height a zoom aims at, in canvas pixels: `region_height` times the zoom. It acts only on a
#: load that asks past `ZOOM_LARGEST`. Reasoned from the second subject's 61 renders measured on 2026-10-22 (the
#: lead's table: head height by the body-parts model's hair and face classes, on the canvas, and the hair's lightness
#: against her face's): drawn dark at 96 and 154 px with all her hair inside the hole (second_hfp_1419c, 1093c, both at
#: the cap of 2; the lead's reading); light at about 180 px (4479b to e, 4602c and d, 1010g) and at 219 (4726c); 4912e
#: (226 px, all her hair in the hole) came out darker than those. Size is not shown to be the cause: 1093c's and
#: 1419c's texts also said "short blonde hair ending at her shoulders", and on 4479e taking those words out alone
#: fixed a dark nape (the lead, 2026-10-22); every whole-frame load with her head at 283 to 456 px came out light.
#: 300 sits with those, above every dark one. A starting value, for the test that decides it (the 1093 pair).
ZOOM_HEAD_TARGET = 300
#: The smallest zoom worth doing: under it the load is rendered on the whole frame and the report says so.
#: Reasoned: a quarter larger is a token or two on a face, for a second resize of everything laid back.
ZOOM_SMALLEST = 1.25
#: Room left round the framed boxes on each side, as a share of their union's own width and height, before
#: the box is taken out to the canvas's shape. Reasoned: the region grows past the mask by its margin and the
#: model needs plate round the region on every side; a quarter each way leaves the subject half the picture.
ZOOM_ROOM = 0.25
#: What `zoomed_source` is told of the window a run continues, beside None (it was rendered on the whole frame)
#: and its `zoom` record (it was rendered in that box): the run continues nothing, or the stored window has no
#: region file to say (`stored_zoom`).
NOT_CONTINUED = "not continued"
NO_REGION_FILE = "no region file"
#: A zoom's `set_by`: the box with its room, the largest (the load's `zoom_up_to`), the region's height to
#: `ZOOM_HEAD_TARGET`, or `ZOOM_LARGEST` where the load asked for more and the target was reached under it. At the
#: default only the first two, as before 2026-10-22.
SET_BY_BOX, SET_BY_LARGEST = "the box", "the largest zoom"
SET_BY_HEAD, SET_BY_ANY = "the head's target height", "the zoom any load may take"
#: Room left around the subject's box on each side, in canvas pixels, before
#: it is taken out to the canvas multiple. Reasoned: one of the encoder's
#: merged tokens (patch 16 by merge 2), so a limb at the edge of the box is
#: not at the edge of the picture. Not rendered yet.
MOTION_BOX_ROOM = 32
#: Shorter side of that reference, in pixels. Reasoned, 2026-10-05: core never
#: enlarges a reference video, so this sets its pixel area; with the VAE copy
#: on, a 384 short edge costs a quarter of the 768 canvas's rows
#: (`docs/h3_references.md`, "Budget by pixel area"). Encoder-only, it sets
#: how much the text encoder sees of the subject at two frames per second.
MOTION_SHORT_EDGE = 384
#: The lazy inputs core is asked for through `check_lazy_status`: the mask always, the segmenter and its text
#: encoder only when `replace` cuts a part with them.
LAZY_FOR_MASK = ("mask", "segmenter", "segmenter_clip")
#: The tracker's shot table (`shot_table.py`), lazy with the mask.
LAZY_FOR_TABLE = "shot_table"
#: The part node's mask, lazy with the mask and read only for `the wired parts` (2026-10-05).
LAZY_FOR_PARTS = "parts"


def run_lengths(latent_t: int) -> list[int]:
    """Pixel frames under each latent step: `FRAME_PER_TOKEN`, cyclic."""
    return [FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(int(latent_t))]


def fit_frames(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[N, H, W, C] to [N, height, width, 3], centre-cropped to the canvas's shape.

    The same call core's own H3 nodes fit a clip with
    (`comfy_extras/nodes_minimax_h3.py`, `common_upscale(..., "center")`).
    """
    out = comfy.utils.common_upscale(frames[..., :3].movedim(-1, 1), int(width), int(height), "bilinear", "center")
    return out.movedim(1, -1)


def _fit_crop(old_w: int, old_h: int, width: int, height: int) -> tuple[int, int]:
    """The columns and rows `comfy.utils.common_upscale`'s centre crop takes off each side of an
    `old_w` by `old_h` picture on its way to `width` by `height`. `fit_mask` says why it is restated."""
    old_aspect, new_aspect = old_w / old_h, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
    return x, y


def fit_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[N, h, w] to [N, height, width] through the crop `fit_frames` makes.

    The crop is `comfy.utils.common_upscale`'s centre crop, restated because
    that function then interpolates, and interpolation samples: a thin mask
    feature can fall between samples on the way down. Here a mask that is
    being made smaller is max-pooled, so nothing is lost; one being made
    larger is interpolated, where nothing can be. `bench/check_video_mask.py`
    holds this crop to core's.
    """
    width, height = int(width), int(height)
    old_h, old_w = int(mask.shape[-2]), int(mask.shape[-1])
    x, y = _fit_crop(old_w, old_h, width, height)
    m = mask.to(torch.float32).narrow(-2, y, old_h - y * 2).narrow(-1, x, old_w - x * 2).unsqueeze(1)
    if m.shape[-2] >= height and m.shape[-1] >= width:
        return F.adaptive_max_pool2d(m, (height, width))[:, 0]
    return F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)[:, 0]


#: A dilation of at least this many pixels is done on a mask reduced by
#: `GROW_COARSE`, which costs a small fraction of the full-size pool. The
#: result covers at least the asked distance and at most `GROW_COARSE - 1`
#: pixels more. Reasoned: the reduction is far below a DiT token.
GROW_COARSE_FROM = 16
GROW_COARSE = 4


def grow(mask: torch.Tensor, pixels) -> torch.Tensor:
    """Dilate a [N, H, W] mask by at least `pixels` in every direction (a square max-pool).

    Exact below `GROW_COARSE_FROM`. From there the mask is max-pooled down by
    `GROW_COARSE`, dilated there and brought back, so it never covers less
    than asked and the cost stays flat as the distance grows.

    `pixels` is one number for every frame, or a [N] tensor with each frame's
    own (`margins`): frames that share a distance are dilated together, and a
    frame's result does not depend on which others it was given with.
    """
    if torch.is_tensor(pixels):
        each = pixels.to(torch.long).flatten()
        if int(each.shape[0]) != int(mask.shape[0]):
            raise ValueError(f"{int(each.shape[0])} margins for a mask of {int(mask.shape[0])} frames")
        out = None
        for p in each.unique().tolist():
            rows = (each == p).nonzero()[:, 0].to(mask.device)
            grown = grow(mask[rows], int(p))
            if out is None:
                out = grown.new_zeros(tuple(mask.shape))
            out[rows] = grown
        return mask.clone() if out is None else out
    p = int(pixels)  # zero is a kernel of one, which returns the mask
    coarse = p >= GROW_COARSE_FROM
    k = -(-p // GROW_COARSE) if coarse else p
    parts = []
    for i in range(0, mask.shape[0], CHUNK):
        m = mask[i:i + CHUNK].unsqueeze(1)
        if coarse:
            m = F.max_pool2d(m, GROW_COARSE, ceil_mode=True)
        m = F.max_pool2d(m, (1, 2 * k + 1), stride=1, padding=(0, k))
        m = F.max_pool2d(m, (2 * k + 1, 1), stride=1, padding=(k, 0))
        if coarse:
            m = m.repeat_interleave(GROW_COARSE, dim=-2).repeat_interleave(GROW_COARSE, dim=-1)
            m = m[..., :mask.shape[-2], :mask.shape[-1]]
        parts.append(m[:, 0])
    return torch.cat(parts, dim=0)


def area_share(mask: torch.Tensor) -> torch.Tensor:
    """The share of the frame a [N, H, W] mask covers on each frame, [N] in 0..1.

    Counted a chunk of frames at a time and never as a float copy of the clip: the node takes this on every
    run, whatever `grow_by` is.
    """
    pixels = float(int(mask.shape[1]) * int(mask.shape[2]))
    counts = [(mask[i:i + CHUNK] > 0.5).flatten(1).sum(dim=1) for i in range(0, int(mask.shape[0]), CHUNK)]
    return (torch.cat(counts).to(torch.float32) / pixels) if counts else torch.zeros(0, dtype=torch.float32)


def steady(values: torch.Tensor, reach: int = GROW_STEADY) -> torch.Tensor:
    """A [N] series' running median over `reach` frames to each side, read from the frames above zero only.

    A median, not a mean: a run of outliers shorter than half the window does not move it, and a step (a cut
    to a shot where the subject is another size) stays a step on the frame it falls on. A frame with nothing
    on it takes its neighbours' value, which nothing reads, since there is no mask there to grow; a stretch
    with nothing anywhere in reach is zero.
    """
    r = int(reach)
    v = values.to(torch.float32).clone()
    v[v <= 0] = float("nan")
    padded = F.pad(v[None, None], (r, r), value=float("nan"))[0, 0]
    return torch.nan_to_num(padded.unfold(0, 2 * r + 1, 1).nanmedian(dim=1).values, nan=0.0)


def margins(area: torch.Tensor | None, pixels: int, grow_pixels: int, feather_pixels: int, grow_by: str):
    """How far the mask is widened on each frame, in pixels of a frame that holds `pixels` of them.

    `a fixed margin` is `grow_pixels`, returned as the number it is, so `grow` does exactly what it did before
    the choice existed. `the subject's size` is `GROW_SHARE` of the square root of the mask's steadied area
    (`area`: `area_share` of the mask the region is grown from, over the whole clip), a [N] tensor, never above
    `grow_pixels` and never below `feather_pixels`: the node refuses a feather wider than the margin because
    the blend would reach the subject's own pixels, and that has to hold on every frame.

    The area is a share of the frame, so the same record gives the margin on the source's own frames (the
    preview) and on the render canvas (a window). The fit's centre crop changes that share by the little it
    crops; nothing corrects for it.
    """
    if grow_by not in GROW_BY:
        raise ValueError(f"unknown grow_by {grow_by!r}; one of {list(GROW_BY)}")
    cap = int(grow_pixels)
    if grow_by == GROW_FIXED or area is None:
        return cap
    side = (steady(area) * float(pixels)).sqrt()
    return (GROW_SHARE * side).ceil().long().clamp(min=min(int(feather_pixels), cap), max=cap)


def source_margins(source: dict, first_frame: int, frames: int, pixels: int):
    """The margin on each of `frames` frames of a source from `first_frame`, for a frame of `pixels` pixels.

    Every place that widens a source's mask reads it here, so the region the model regenerates, the preview,
    the report and the composite cannot disagree. One number under `a fixed margin`. Frames past the source's
    end take the cap: the mask is empty there (`window_frames`).
    """
    each = margins(source.get("subject_area"), pixels, source["grow_pixels"], source.get("feather_pixels", 0),
                   source.get("grow_by", GROW_FIXED))
    if not torch.is_tensor(each):
        return each
    each = each[int(first_frame):int(first_frame) + int(frames)]
    short = int(frames) - int(each.shape[0])
    if short > 0:
        # past the source's end the held frames take the last frame's own margin, as they take its mask
        # (`held_mask`)
        tail = int(each[-1]) if int(each.shape[0]) else int(source["grow_pixels"])
        each = torch.cat([each, each.new_full((short,), tail)])
    return each


def motion_widening(source: dict) -> int:
    """How far the motion reference widens the subject before the rest goes grey: half of `grow_pixels`, on
    every frame, under either `grow_by`.

    Not `source_margins`. **Measured** 2026-10-08 on one clip's masks, outside the node: taken from the
    subject's size, the widening greyed part of the tracked subject above the part's lowest row on some
    frames of the windows where the subject is small, and the fixed half-margin far less of it (the masking
    board, finding `mhi-04`, has the figures). A reference that hides a raised arm is worse than one that
    shows a neighbour's edge, so only the region scales.
    """
    return int(source["grow_pixels"]) // 2


def margin_note(each) -> str:
    """A margin for a report: `64 px`, or its range over the frames it was taken on."""
    if not torch.is_tensor(each):
        return f"{int(each)} px"
    low, high = int(each.min()), int(each.max())
    return f"{low} px" if low == high else f"{low} to {high} px"


def token_mask(mask: torch.Tensor, latent_t: int, lat_h: int, lat_w: int, whole_tokens: bool = True) -> torch.Tensor:
    """A [F, H, W] pixel mask (above 0.5 = regenerate) to [latent_t, lat_h, lat_w] of 0 or 1.

    F must be the window's frame count, the sum of `run_lengths(latent_t)`.
    Every value in a 2x2 patch is the same, which is what core's own pooling
    would make of it. With `whole_tokens` off the mask is left per latent
    cell (`EDGE_CELLS`): a cell is regenerated when any pixel under it on any
    frame of its run is, and core does the pooling to tokens itself.
    """
    runs = run_lengths(latent_t)
    if int(mask.shape[0]) != sum(runs):
        raise ValueError(f"the mask has {int(mask.shape[0])} frames; a {latent_t}-step window is {sum(runs)}")
    binary = (mask > 0.5).to(torch.float32)
    cells = torch.cat([F.adaptive_max_pool2d(binary[i:i + CHUNK].unsqueeze(1), (int(lat_h), int(lat_w)))[:, 0]
                       for i in range(0, binary.shape[0], CHUNK)], dim=0)
    steps, at = [], 0
    for n in runs:
        steps.append(cells[at:at + n].amax(dim=0))
        at += n
    m = torch.stack(steps, dim=0)
    if not whole_tokens:
        return m.contiguous()
    # core's patch pooling (`mask_row_values`): replicate-pad to even, max per 2x2
    m = F.pad(m.unsqueeze(1), (0, lat_w % 2, 0, lat_h % 2), mode="replicate")[:, 0]
    tokens = m.reshape(latent_t, m.shape[-2] // 2, 2, m.shape[-1] // 2, 2).amax(dim=(2, 4))
    return tokens.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[:, :lat_h, :lat_w].contiguous()


#: A KEEP-OUT SPECK DOES NOT HOLE A STEP (the outside reviewer's G2 of 2026-10-15, ruled by the lead that day): the others' mask takes
#: a cell out of the region for the whole of its latent step (`token_mask` pools a step's frames), so a speck on one or two frames
#: holes four, and the subject's hair is drawn on some steps and not the next (second_hfp_0014d: 57-60 by a strangers speck of 89 px,
#: 159-162 at both collars by held specks of 9-183 px). A piece of the others' mask (4-connected, on one frame) under `SPECK_CELLS`
#: cells' area, present at its place on at most `SPECK_STEP_SHARE` of its step's frames, is dropped from the others before the region
#: is cut. A step of one frame (a window's first) has no speck dropped: half of one frame is none, and one frame holes only itself.
#: **Measured** one cell, a starting value, on that window's saved region: the flash on 159-162 needs 256 px at a 16 px cell
#: (its piece on 161 is 183 px), and one cell mends the three flashes the reviewer placed there (the record,
#: `bench/results/2026-10-15_hair_flashes.md`). **Reasoned** the half: a real face can be one cell on a small shot (the caveat of
#: 2026-10-14), and a face is on most frames of its step; a speck that is is kept.
SPECK_CELLS = 1.0
SPECK_STEP_SHARE = 0.5


def drop_specks(away: torch.Tensor, latent_t: int, lat_h: int, lat_w: int) -> torch.Tensor:
    """The others' mask [F, H, W] less its specks (`SPECK_CELLS`, `SPECK_STEP_SHARE`), a frame and a step at a time."""
    import cv2
    h, w = int(away.shape[-2]), int(away.shape[-1])
    floor = SPECK_CELLS * (h / lat_h) * (w / lat_w)
    on = (away > 0.5).cpu().numpy()
    gone = np.zeros_like(on)
    at = 0
    for n in run_lengths(latent_t):
        step = on[at:at + n]
        for i in range(n):
            k, lab, stats, _ = cv2.connectedComponentsWithStats(step[i].astype(np.uint8), connectivity=4)
            for j in range(1, k):
                if stats[j, cv2.CC_STAT_AREA] >= floor:
                    continue
                x, y, bw, bh = (int(v) for v in stats[j, :4])
                piece = lab[y:y + bh, x:x + bw] == j
                frames_on = sum(bool((step[g, y:y + bh, x:x + bw] & piece).any()) for g in range(n))
                if frames_on <= SPECK_STEP_SHARE * n:
                    gone[at + i, y:y + bh, x:x + bw] |= piece
        at += n
    return away * torch.from_numpy(~gone).to(device=away.device, dtype=away.dtype)


#: A ONE-CELL HOLE IS CLOSED (G2's 2C, ruled with the specks): a cell the others took out of the grown region, not the subject's own,
#: with all four of its neighbours in the region, is given back to the region where the others' px in it are under the specks' floor
#: on every frame of its step and the others are on it on at most `SPECK_STEP_SHARE` of those frames (the specks' guard, the lead's
#: ruling of 2026-10-15). On second_hfp_0014d's 163-166 a strangers piece of 286-418 px, over one cell, holed one cell in the middle
#: of her hair with 40 and 105 px in it on two of the four frames. A cell a keep-out fills, or holds on most of its step, stays
#: open, whoever it is: a small face held all through its step, which the specks rule keeps out, is not closed back in either. In
#: whole tokens the cell is a token, and the floor is still the specks' (one cell's area).
def close_lone_holes(tokens: torch.Tensor, grown: torch.Tensor, own: torch.Tensor, whole: bool, away: torch.Tensor) -> torch.Tensor:
    """`tokens` with each enclosed one-cell (or one-token) hole of the others (`away`, [F, H, W]) given back where the others' px in
    it are under the specks' floor on every frame of its step and on it on at most `SPECK_STEP_SHARE` of those frames."""
    latent_t, lat_h, lat_w = (int(v) for v in tokens.shape)
    h, w = int(away.shape[-2]), int(away.shape[-1])
    floor = SPECK_CELLS * (h / lat_h) * (w / lat_w)
    px = torch.cat([F.adaptive_avg_pool2d((away[i:i + CHUNK] > 0.5).float().unsqueeze(1), (lat_h, lat_w))[:, 0]
                    for i in range(0, int(away.shape[0]), CHUNK)], dim=0) * (h * w / (lat_h * lat_w))
    if whole:
        px = F.pad(px.unsqueeze(1), (0, lat_w % 2, 0, lat_h % 2))[:, 0]
        px = px.reshape(px.shape[0], px.shape[-2] // 2, 2, px.shape[-1] // 2, 2).sum(dim=(2, 4))
    brief, at = [], 0
    for n in run_lengths(latent_t):
        step = px[at:at + n]
        brief.append(((step > 0.5).sum(dim=0) <= SPECK_STEP_SHARE * n) & (step.amax(dim=0) < floor))
        at += n
    brief = torch.stack(brief, dim=0).to(tokens.device)
    grid = (lambda t: t[:, ::2, ::2]) if whole else (lambda t: t)
    t, g, o = grid(tokens) > 0.5, grid(grown) > 0.5, grid(own) > 0.5
    p = F.pad(t.float(), (1, 1, 1, 1)) > 0.5
    ring = p[:, :-2, 1:-1] & p[:, 2:, 1:-1] & p[:, 1:-1, :-2] & p[:, 1:-1, 2:]
    closed = (t | (~t & g & ~o & ring & brief)).float()
    if whole:
        closed = closed.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[:, :lat_h, :lat_w]
    return closed.to(tokens.dtype).contiguous()


#: AN ISLAND OF THE REGION WITH NONE OF THE SUBJECT IS DROPPED (G2's (b), ruled with the specks): where the keep-outs cut the margin,
#: a piece of it can lie cut off from the subject (an apron, a chest, a bowl below her arms), regenerated with nothing of hers in it.
#: A 4-connected piece of the region on a step that shares no cell with the subject's own mask is given back to the source.
#: **Measured** on second_hfp_0604d and 0014d's saved windows: 43 such pieces, none holding a pixel of her hair or face in her class
#: map, and the reviewer saw none holding rendered hair. The others never take a cell of the subject's own mask, so her own hair is
#: never an island; what it can drop is margin room where new hair would fall outside her carried mask (0604d's 799: a cell the
#: specks rule gave back, cut off, so dropped again), and a piece `keep` cuts off. Not measured on a window where either holds hair.
def drop_islands(tokens: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
    """`tokens` less each piece of a step that shares no cell with `own`."""
    import cv2
    out = tokens.clone()
    for k in range(int(tokens.shape[0])):
        n, lab = cv2.connectedComponents((tokens[k] > 0.5).cpu().numpy().astype(np.uint8), connectivity=4)
        if n <= 1:
            continue
        lab = torch.from_numpy(lab)
        mine = torch.unique(lab[(own[k].cpu() > 0.5) & (lab > 0)])
        off = (lab > 0) & ~torch.isin(lab, mine)
        out[k][off.to(out.device)] = 0
    return out


def pixel_alpha(tokens: torch.Tensor, height: int, width: int, feather_pixels: int) -> torch.Tensor:
    """The blend weight per pixel frame, [F, height, width]: 1 keeps the render, 0 the source.

    Each latent step's mask covers the frames of its run. The feather is a
    box blur of `feather_pixels`, so the ramp is centred on the token edge
    and reaches that far to each side.
    """
    runs = torch.tensor(run_lengths(tokens.shape[0]), device=tokens.device)
    f = int(feather_pixels)
    out = []
    for i in range(0, tokens.shape[0], 12):
        a = F.interpolate(tokens[i:i + 12].unsqueeze(1), size=(int(height), int(width)), mode="nearest")
        # zero is a kernel of one, which returns the hard edge
        a = F.avg_pool2d(F.pad(a, (f, f, f, f), mode="replicate"), 2 * f + 1, stride=1)
        out.append(a[:, 0].repeat_interleave(runs[i:i + 12], dim=0))
    return torch.cat(out, dim=0)


def changed_alpha(images: torch.Tensor, source: torch.Tensor, tokens: torch.Tensor,
                  old_subject: torch.Tensor, feather_pixels: int, old_margin: int,
                  threshold: float) -> torch.Tensor:
    """The blend weight that keeps the render only where it changed the picture. [F, H, W].

    The change is the absolute difference from the source, averaged over the
    channels and over `CHANGE_BLUR` pixels, ramped from 0 at 0.6 of
    `threshold` to 1 at 1.4 of it, so a pixel hovering near the threshold
    fades and does not switch. The old subject's mask widened by `old_margin`
    is always kept. The result is held at its maximum over each temporal run
    (the mask's own grain), cut to the regenerated tokens, widened by the
    feather to close small holes and blurred by it, and never exceeds the
    whole-region weight.
    """
    f, t = int(feather_pixels), float(threshold)
    height, width = int(images.shape[1]), int(images.shape[2])
    region = pixel_alpha(tokens, height, width, 0) > 0.5
    b = CHANGE_BLUR
    parts = []
    for i in range(0, images.shape[0], CHUNK):
        d = (images[i:i + CHUNK] - source[i:i + CHUNK].to(images.device, images.dtype)).abs().mean(dim=-1).unsqueeze(1)
        d = F.avg_pool2d(F.pad(d, (b, b, b, b), mode="replicate"), 2 * b + 1, stride=1)[:, 0]
        parts.append(((d - 0.6 * t) / max(0.8 * t, 1e-6)).clamp(0.0, 1.0))
    keep = torch.maximum(torch.cat(parts, dim=0), (grow(old_subject.to(torch.float32), old_margin) > 0.5).float())
    keep = keep * region
    held, at = [], 0
    for n in run_lengths(tokens.shape[0]):
        held.append(keep[at:at + n].amax(dim=0, keepdim=True).expand(n, -1, -1))
        at += n
    wide = grow(torch.cat(held, dim=0), f)
    soft = [F.avg_pool2d(F.pad(wide[i:i + CHUNK].unsqueeze(1), (f, f, f, f), mode="replicate"), 2 * f + 1, stride=1)[:, 0]
            for i in range(0, wide.shape[0], CHUNK)]
    return torch.minimum(torch.cat(soft, dim=0), pixel_alpha(tokens, height, width, f))


def composite(images: torch.Tensor, source: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """`alpha * images + (1 - alpha) * source`, [F, H, W, 3] with alpha [F, H, W]."""
    if tuple(images.shape[:3]) != tuple(source.shape[:3]) or tuple(images.shape[:3]) != tuple(alpha.shape):
        raise ValueError(
            f"the decoded window {tuple(images.shape)}, the source {tuple(source.shape)} and the blend "
            f"weight {tuple(alpha.shape)} do not cover the same frames")
    out = torch.empty_like(images)
    for i in range(0, images.shape[0], CHUNK):
        a = alpha[i:i + CHUNK].unsqueeze(-1).to(images.device, images.dtype)
        out[i:i + CHUNK] = images[i:i + CHUNK] * a + source[i:i + CHUNK].to(images.device, images.dtype) * (1.0 - a)
    return out


def source_cuts(frames: torch.Tensor, table: str = "") -> list[int]:
    """The frames of a source that start a new shot, in order.

    From the Subject Track's shot table when one is wired, so they are the
    cuts the mask was tracked with. Otherwise from the tracker's own detector
    on these frames (`subject_track.cut_scores`, `auto_cuts`), so a mask that
    was loaded from a file, or made by another node, still has them.
    """
    ranges = shot_ranges({"shot_table": table})
    if ranges:
        return sorted({int(a) for a, _b in ranges if int(a) > 0})
    from . import subject_track   # here, not at the top: the tracker's module is not needed to import this one
    scores = subject_track.cut_scores(frames)
    if not int(scores.numel()):
        return []
    return subject_track.find_cuts(scores, subject_track.auto_cuts(scores))


def cut_gate(mask: torch.Tensor, latent_t: int, cuts, first_frame: int = 0) -> torch.Tensor:
    """Which frames of a window the composite may lay anything on: [F] of 1 or 0.

    One latent step is a run of pixel frames (`run_lengths`), and the region
    is one per step (`token_mask`), so a step whose run straddles a cut
    carries the subject's region onto the frames of the other shot. The
    sampler cannot leave those cells alone; the composite can leave them
    unlaid. A frame is 0 when its run is split by a cut, the subject's mask
    is empty on every frame of the run on this frame's side of it, and it is
    not empty on another side. `cuts` are frames of the source that start a
    shot (`source_cuts`) and `first_frame` the window's first.

    Found 2026-10-10: a whole-subject pass repainted the first or last frame
    or two of the next shot at four cuts of one clip (the assembler's own
    check).

    This is the rule for a cut only, and since later the same day it is no
    longer the whole of what the composite leaves alone. Until then this
    docstring said "not a rule about an empty mask: a frame the tracker lost
    inside a shot has no cut beside it and is laid as before, covered by its
    run". That was reversed (the lead, 2026-10-10) after two renders laid
    something on frames whose own mask was empty with no cut near them, one
    of them a face drawn for a single frame where the source shows none:
    `unlaid_frames` now leaves EVERY frame with no mask of its own as the
    source, and this function names the ones among them that lie across a
    cut, for the report, the plan and the preflight.
    """
    present = (mask > 0.5).flatten(1).any(dim=1)
    gate = torch.ones(int(mask.shape[0]), dtype=torch.float32)
    inside = sorted({int(c) - int(first_frame) for c in (cuts or [])})
    at = 0
    for n in run_lengths(latent_t):
        edges = [at] + [c for c in inside if at < c < at + n] + [at + n]
        sides = list(zip(edges, edges[1:]))
        has = [bool(present[a:b].any()) for a, b in sides]
        if len(sides) > 1 and any(has):
            for (a, b), on in zip(sides, has):
                if not on:
                    gate[a:b] = 0.0
        at += n
    return gate


def unlaid_frames(mask: torch.Tensor, tokens: torch.Tensor, cuts, first_frame: int = 0) -> tuple[list[int], list[int]]:
    """The frames of a window the composite lays nothing on, as two lists of window frames: (across, bare).

    The rule is one line: **a frame whose own fitted mask is empty is never laid.** A latent step's region is
    the maximum over its frames, so a frame with no mask can still sit under a region lent by its neighbours
    in the step, and whatever the sampler drew there would be laid on a frame nobody asked to change.
    `across` are those of them `cut_gate` names, on the far side of a cut from the subject inside one step.
    `bare` are the rest that a region reaches (their step has a regenerated token): a frame the part node
    emptied because the subject is turned away, a frame the tracker dropped. Frames
    with no mask in a step with no region are not listed: nothing was there to lay. The case met first: a
    subject who ENTERS mid-shot has empty frames before her first masked one, and those that share her first
    latent step are left as the source, as anyone would want.

    The node cannot tell a frame that is rightly empty from one the tracker lost while the subject is plainly
    there. On the first this is right. On the second it shows one frame of the ORIGINAL in a run of the new
    subject, so the rule is only as good as the mask: such a frame is the preflight's to block before a
    render (`bench/capture_masked_run.py`), until the mask is fixed or someone has written that it is rightly
    empty. The sampler still regenerates the step either way.
    """
    latent_t = int(tokens.shape[0])
    present = (mask > 0.5).flatten(1).any(dim=1)
    gated = cut_gate(mask, latent_t, cuts, first_frame) < 0.5
    reached = (tokens > 0.5).flatten(1).any(dim=1).repeat_interleave(
        torch.tensor(run_lengths(latent_t), device=tokens.device)).to(present.device)
    across = [int(f) for f in gated.nonzero().flatten()]
    bare = [int(f) for f in (~present & reached & ~gated.to(present.device)).nonzero().flatten()]
    return across, bare


def lay_window(images: torch.Tensor, pixels: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor,
               source: dict, margin, first_frame: int):
    """A decoded window laid over its source, as the song node writes it: the frames, the weight, the report's lines.

    `images` is the decode, `pixels`, `tokens` and `mask` what `window` returned for it, `margin` its
    `source_margins` and `first_frame` where the window starts in the source. The weight is the whole
    region's (`pixel_alpha`) or, under `only what changed`, `changed_alpha`'s; it is then taken off every
    frame whose own mask is empty (`unlaid_frames`: across a cut from the subject, or with no cut beside it);
    `composite` blends. Until 2026-10-10 such frames were laid, and for a day a hidden key reproduced that
    for the tool below; it is gone, so a render from before that date, laid again, differs on those frames.
    One function, so
    that a saved window can be laid again with exactly what the render ran: `bench/recomposite_window.py`
    decodes a window's stored latent and calls this, with the settings the render used or with one changed.
    """
    height, width = int(images.shape[1]), int(images.shape[2])
    lines = []
    if source.get("composite") == COMPOSITE_CHANGED:
        # the render is kept only where it changed the picture or the old subject stood
        alpha = changed_alpha(images, pixels, tokens, mask, source["feather_pixels"], margin // 2,
                              source["change_threshold"])
        whole = float(pixel_alpha(tokens, height, width, 0).mean())
        lines.append("composite keeps only what changed: "
                     f"{100.0 * float((alpha > 0.5).float().mean()) / max(whole, 1e-6):.0f}% of the "
                     "regenerated pixels, the source restored in the rest")
    else:
        alpha = pixel_alpha(tokens, height, width, source["feather_pixels"])
    # a latent step's region covers its whole run of frames, so it reaches frames with no mask of their own:
    # the next shot's across a cut, a frame the part was emptied on, a held frame. Those stay the source's
    first = int(first_frame)
    across, bare = unlaid_frames(mask, tokens, source.get("cuts"), first)
    if across or bare:
        gate = torch.ones(int(mask.shape[0]), dtype=torch.float32)
        gate[across + bare] = 0.0
        alpha = alpha * gate[:, None, None].to(alpha)
    if across:
        lines.append(f"{len(across)} frame(s) lie across a cut from the subject inside "
                     "one latent step and are left as the source: frame(s) "
                     + ", ".join(str(first + f) for f in across))
    if bare:
        lines.append(f"{len(bare)} frame(s) have no mask of their own and no cut beside them, and are left "
                     "as the source: frame(s) " + ", ".join(str(first + f) for f in bare))
    return composite(images, pixels, alpha), alpha, lines


#: What of a source's record a window's composite reads, saved with the window's region (`save_window_region`)
#: so the window can be laid again without the graph. `lay_window` reads the first four; the rest say how the
#: region was made.
REGION_SETTINGS = ("composite", "feather_pixels", "change_threshold", "cuts", "grow_pixels", "grow_by", "replace", "edge")


def save_window_region(path: str, mask: torch.Tensor, tokens: torch.Tensor, source: dict, margin,
                       first_frame: int, trim: int, kept: torch.Tensor | None = None, matte: torch.Tensor | None = None,
                       piece: dict | None = None) -> str:
    """Write what a window's composite was run with, beside the window's latent, losslessly.

    The window's fitted mask (bits, packed along the width) and its token region, as `window` returned them,
    its margin, where it starts in the source, how many frames its video leaves off the front, and
    `REGION_SETTINGS` of the source. With the window's stored latent and the source's own frames that is
    everything `lay_window` takes, so a render can be laid again without its tracker or its graph. A render
    whose tracker ran inside its own graph used to leave its mask nowhere. Written whole or not at all.

    With `kept` (the song node's `save_kept`, 2026-10-15): what the composite kept of the render, its weight over
    one half on each of the window's frames (the white outline of the mask review), packed like the mask. A
    review clip draws it from here; until then it was only burned into the review's picture.

    With `matte` (the lead's ruling of 2026-10-16, "the exact matte"): the weight the window's written video was blended
    with, on every one of the window's frames (the trimmed ones too), at the WHOLE canvas whatever the zoom ([F, height,
    width] uint8, 0 to 255: `lay_window`'s weight, or `lay_back`'s on a zoomed window). The assembler lays a piece by it
    and by nothing it would rebuild. It changes nothing H3 is given: the mask and the tokens are written as before.

    With `piece` (the matte's blind read, 2026-10-16): which finished file the window's frames went into and where, {"stem": the output's
    stem (`<prefix>_NNNNN`), "first": the piece frame its first written frame is, "written": how many it wrote}. The window folder is
    shared by every run of a prefix, and a continued run's first window writes from its context's end; the assembler reads the matte by
    these and refuses another stem's file.
    """
    settings = {k: source.get(k) for k in REGION_SETTINGS}
    meta = {"first_frame": int(first_frame), "trim": int(trim), "frames": int(mask.shape[0]),
            "height": int(mask.shape[1]), "width": int(mask.shape[2]), "source": settings,
            # None, or the box this window was rendered in (`zoomed_source`): the mask and the tokens are then
            # of the ZOOMED picture, and a reader that lays them or counts their cells maps through the box
            "zoom": source.get("zoom")}
    if source.get("pad_before") or source.get("pad_after"):
        meta["pad_before"] = int(source.get("pad_before", 0))
        meta["pad_after"] = int(source.get("pad_after", 0))
        # what the context frames showed (`CONTEXT_SHOWS`; a source from before 2026-10-22 carries none, and was `as filmed`),
        # and under `face covered` the window frames of them across a cut from the shot (`context_across_cut`)
        meta["context_shows"] = str(source.get("context_shows") or CONTEXT_AS_FILMED)
        if source.get("context_held_across_a_cut"):
            meta["context_held_across_a_cut"] = [int(f) for f in source["context_held_across_a_cut"]]
        if source.get("context_held_empty"):
            # the sides whose edge frame had no mask, so nothing was held there (`before`, `after`)
            meta["context_held_empty"] = [str(e) for e in source["context_held_empty"]]
        for side in ("held_before", "held_after"):
            # how many context frames on that side were held copies across a cut (`CONTEXT ACROSS A CUT IS HELD`)
            if source.get(side):
                meta[side] = int(source[side])
        if source.get("held_fell_back"):
            # the held sides that kept real frames, each with why (`HELD_EDGE_LEAST`)
            meta["held_fell_back"] = [str(x) for x in source["held_fell_back"]]
    extra = {}
    if matte is not None:
        if tuple(matte.shape[:1]) != tuple(mask.shape[:1]) or matte.dtype != torch.uint8:
            raise ValueError(f"the matte {tuple(matte.shape)} {matte.dtype} is not the window's {int(mask.shape[0])} frames of uint8")
        meta["matte"] = {"what": "the weight the window's video was blended with, 0 to 255, every frame of the window (its trimmed frames too)",
                         "space": "the whole canvas", "height": int(matte.shape[1]), "width": int(matte.shape[2])}
        extra["matte"] = matte.cpu().numpy()
    if piece is not None:
        meta["piece"] = {"stem": str(piece["stem"]), "first": int(piece["first"]), "written": int(piece["written"])}
    if kept is not None:
        meta["kept"] = "what the composite kept of the render: its weight over one half, the window's frames, packed like the mask"
        extra["kept"] = np.packbits((kept > 0.5).cpu().numpy(), axis=-1)
    part = path + ".part"
    with open(part, "wb") as f:
        np.savez_compressed(
            f, mask=np.packbits((mask > 0.5).cpu().numpy(), axis=-1), tokens=(tokens > 0.5).cpu().numpy().astype(np.uint8),
            margin=(margin.cpu().numpy() if torch.is_tensor(margin) else np.asarray(int(margin))).astype(np.int64),
            meta=np.asarray(json.dumps(meta)), **extra)
    os.replace(part, path)
    return path


#: What a padded window's region file says of its context frames (`save_window_region`): a tool that saves a region file
#: again from one it read carries these with `padding_of`, so a `face covered` window is never relabeled `as filmed`
#: (found by a blind read of the change that made them, 2026-10-22: `bench/recomposite_piece.py` dropped the last two).
PADDING_KEYS = ("pad_before", "pad_after", "context_shows", "context_held_across_a_cut", "context_held_empty", "held_before",
                "held_after", "held_fell_back")


def padding_of(region: dict) -> dict:
    """The padding keys (`PADDING_KEYS`) of a region as `load_window_region` returns it, or of a plan's window row: what
    to hand `save_window_region` again with the source's settings. Empty for a window that was not padded."""
    return {k: region[k] for k in PADDING_KEYS if region.get(k)}


def load_window_region(path: str) -> dict:
    """A window's saved region as `lay_window` takes it: `mask` [F, H, W] and `tokens` [latent_t, h, w] of 0 or 1
    (and `kept` [F, H, W] of 0 or 1 when the render saved it, `save_kept`; and `matte` [F, H, W] of 0 to 1 at the whole
    canvas, the weight the window's video was blended with, when the render saved it),
    `margin` (a number, or a [F] tensor when it was each frame's own), `first_frame`, `trim`, `source`, and
    `zoom`: None, or the box of the load's frames the window was rendered in, when the mask and the tokens are
    of that box scaled to the canvas and not of the whole frame (`zoomed_source`). A padded window's also carries
    `pad_before`, `pad_after` and `context_shows` (`CONTEXT_SHOWS`; absent in a file saved before 2026-10-22, when the
    context was `as filmed`), with `context_held_across_a_cut` where a held context frame lies across a cut,
    `context_held_empty` where an edge frame had no mask to hold, and `held_before` and `held_after` where context
    frames across a cut were held copies of the shot's edge frame (`hold_context`; absent before 2026-10-22, when they
    were real), with `held_fell_back` naming a side that kept real frames because its edge frame had too small a region
    (`HELD_EDGE_LEAST`) (`PADDING_KEYS`)."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        mask = np.unpackbits(z["mask"], axis=-1)[..., :int(meta["width"])]
        margin = z["margin"]
        out = {"mask": torch.from_numpy(mask).to(torch.float32), "tokens": torch.from_numpy(z["tokens"]).to(torch.float32),
               "margin": int(margin) if margin.ndim == 0 else torch.from_numpy(margin).long()}
        if "kept" in z.files:
            out["kept"] = torch.from_numpy(np.unpackbits(z["kept"], axis=-1)[..., :int(meta["width"])]).to(torch.float32)
        if "matte" in z.files:
            out["matte"] = torch.from_numpy(z["matte"]).to(torch.float32) / 255.0
    return {**out, "first_frame": int(meta["first_frame"]), "trim": int(meta["trim"]), "source": dict(meta["source"]),
            "zoom": meta.get("zoom"),
            **({"pad_before": int(meta.get("pad_before", 0)), "pad_after": int(meta.get("pad_after", 0)),
                **{k: meta[k] for k in PADDING_KEYS[2:] if k in meta}}
               if (meta.get("pad_before") or meta.get("pad_after")) else {})}


@dataclass(frozen=True)
class OverlayLayer:
    """One thing the mask view draws over the source, and one entry of its legend.

    `frames(at, end)` returns the layer's mask for frames `at` to `end` of the window, [n, H, W], true
    where it is drawn; a function and not a tensor so a layer that is cheap to make a piece at a time
    (the token region) never exists whole. `outline` draws the mask's edge and leaves its inside alone.
    The list is the design (2026-10-06): today's layers are the tracked subject, what else regenerates
    and what the composite kept; a part node's classes or a tracker's other people are more layers,
    each with its name and colour, and the legend is built from whatever the list holds.
    """

    name: str
    colour: tuple[float, float, float]
    frames: Callable[[int, int], torch.Tensor]
    strength: float = 0.5
    outline: bool = False


#: The mask view's colours and strengths. The owner's first look was a throwaway overlay in this red
#: and this cyan; kept so the two read alike.
OVERLAY_SUBJECT = ((1.0, 0.16, 0.16), 0.5)
#: What the view calls the mask it draws in that colour, by the Masked Source's `replace`. The mask
#: is the tracked subject only for the whole subject; for the other two it is a part of them, and
#: the two can sit in different places: on 2026-10-06 the tracker held the subject on every frame of
#: a clip while the wired parts' mask lay on other people for its first seconds. A legend that
#: called that mask "the tracked subject" would have said the opposite of what the render did.
OVERLAY_MASK_NAMES = {REPLACE_WHOLE: "the tracked subject", REPLACE_PART: "the head and hair",
                      REPLACE_PARTS: "the parts taken"}
OVERLAY_REGION = ((0.0, 0.78, 1.0), 0.4)
OVERLAY_KEPT = (1.0, 1.0, 1.0)
#: An outline's width in pixels. Reasoned: visible at the canvas's size, thin enough to read what is under it.
OVERLAY_OUTLINE = 2
#: The legend's text height as a share of the frame's, and its floor in pixels. Reasoned: small and out of
#: the centre, still legible on a phone.
LEGEND_SHARE = 0.022
LEGEND_MIN = 11


def token_region(tokens: torch.Tensor, height: int, width: int) -> Callable[[int, int], torch.Tensor]:
    """The regenerated region as an `OverlayLayer.frames`: the token mask the sampler was given, brought
    back to pixels with no feather, for a piece that starts and ends on whole cycles of `FRAME_PER_TOKEN`."""
    cycle, per = len(FRAME_PER_TOKEN), sum(FRAME_PER_TOKEN)

    def frames(at: int, end: int) -> torch.Tensor:
        if at % per:
            raise ValueError(f"a piece of the token region must start on a cycle of {per} frames; got frame {at}")
        steps = tokens[at // per * cycle:-(-end // per) * cycle]
        return (pixel_alpha(steps, int(height), int(width), 0) > 0.5)[:end - at]
    return frames


def mask_layer_name(replace: str) -> str:
    """The legend's name for the mask a source carries, from its `replace` (`OVERLAY_MASK_NAMES`). A value
    this does not know is refused: a wrong name here is a picture that misleads."""
    if replace not in OVERLAY_MASK_NAMES:
        raise ValueError(f"no name for the mask of replace {replace!r}; one of {list(OVERLAY_MASK_NAMES)}")
    return OVERLAY_MASK_NAMES[replace]


def window_layers(mask: torch.Tensor, tokens: torch.Tensor, height: int, width: int, replace: str,
                  alpha: torch.Tensor | None = None) -> list[OverlayLayer]:
    """What a masked window's view shows today, most specific first: the mask the source carries, named for
    what it is by the source's `replace` (`mask_layer_name`); what else regenerates; and with `alpha` (the
    composite's weight under `only what changed`) what was kept from the render, as an outline."""
    layers = [OverlayLayer(mask_layer_name(replace), OVERLAY_SUBJECT[0], lambda at, end: mask[at:end] > 0.5, OVERLAY_SUBJECT[1]),
              OverlayLayer("what else regenerates", OVERLAY_REGION[0], token_region(tokens, height, width), OVERLAY_REGION[1])]
    if alpha is not None:
        layers.append(OverlayLayer("what the render kept", OVERLAY_KEPT, lambda at, end: alpha[at:end] > 0.5, 1.0, outline=True))
    return layers


def overlay_legend(layers: list[OverlayLayer], height: int, width: int, note: str = "") -> tuple[torch.Tensor, float]:
    """The legend as a patch, [h, w, 3], and its opacity, for the bottom-left corner of a frame.

    One line saying what each colour is, built from the layers in the order given, because the file is
    opened cold, away from the graph and its log. A filled swatch for a filled layer, an empty one for an
    outline, then `note` (the window's regenerated share, from the song node). Drawn once per window with
    Pillow's own font; what does not fit the frame's width is cut off.
    """
    from PIL import Image, ImageDraw, ImageFont

    size = max(LEGEND_MIN, int(round(int(height) * LEGEND_SHARE)))
    font = ImageFont.load_default(size=size)
    pad, gap = size // 2, size
    widths = [int(font.getlength(layer.name)) for layer in layers]
    tail = (gap + int(font.getlength(note))) if note else 0
    w = min(int(width), pad + sum(size + pad // 2 + n for n in widths) + gap * max(len(layers) - 1, 0) + tail + pad)
    h = size + 2 * pad
    patch = Image.new("RGB", (w, h), (0, 0, 0))
    draw, x = ImageDraw.Draw(patch), pad
    for layer, n in zip(layers, widths):
        rgb = tuple(int(round(255 * c)) for c in layer.colour)
        draw.rectangle([x, pad, x + size - 1, pad + size - 1], fill=None if layer.outline else rgb, outline=rgb, width=2)
        draw.text((x + size + pad // 2, pad - 1), layer.name, fill=(255, 255, 255), font=font)
        x += size + pad // 2 + n + gap
    if note:
        draw.text((x, pad - 1), note, fill=(200, 200, 200), font=font)
    colour = torch.frombuffer(bytearray(patch.tobytes()), dtype=torch.uint8).reshape(h, w, 3).to(torch.float32) / 255.0
    return colour, 0.75


def overlay_pieces(pixels: torch.Tensor, layers: list[OverlayLayer], latent_t: int, start: int = 0, note: str = ""):
    """The frames of a window's mask view, [n, H, W, 3] a cycle of latent steps at a time, from frame `start`.

    `pixels` are the window's source frames at the canvas (`window`). Each pixel is tinted by the first
    filled layer in the list that covers it and by no other, so tints never mix: a list is given most
    specific first (a subject before the region around it, a part before the person it is on). Outlines go
    over the fills; the legend, with `note` after it, goes in the bottom-left corner of every frame. Nothing here is the size of the window: a piece is one cycle of
    `FRAME_PER_TOKEN`, cut where `changed_alpha_on` cuts, and a window of `latent_t` steps has
    `sum(run_lengths(latent_t))` frames.
    """
    height, width = int(pixels.shape[1]), int(pixels.shape[2])
    legend, opacity = overlay_legend(layers, height, width, note)
    lh, lw = int(legend.shape[0]), int(legend.shape[1])
    cycle, runs, at, k = len(FRAME_PER_TOKEN), run_lengths(latent_t), 0, int(OVERLAY_OUTLINE)
    for i in range(0, int(latent_t), cycle):
        end = at + sum(runs[i:i + cycle])
        if end > int(start):
            px = pixels[at:end, ..., :3].to(torch.float32).clone()
            taken = torch.zeros(px.shape[:3], dtype=torch.bool)
            for layer in (x for x in layers if not x.outline):
                where = layer.frames(at, end) & ~taken      # a pixel takes the first layer that covers it, and only that tint
                tint = torch.tensor(layer.colour, dtype=px.dtype)
                px = torch.where(where.unsqueeze(-1), px * (1.0 - layer.strength) + tint * layer.strength, px)
                taken |= where
            for layer in (x for x in layers if x.outline):
                inside = layer.frames(at, end).to(torch.float32).unsqueeze(1)
                eroded = 1.0 - F.max_pool2d(1.0 - inside, 2 * k + 1, stride=1, padding=k)
                edge = (inside - eroded)[:, 0] > 0.5
                px = torch.where(edge.unsqueeze(-1), torch.tensor(layer.colour, dtype=px.dtype), px)
            px[:, height - lh:, :lw] = px[:, height - lh:, :lw] * (1.0 - opacity) + legend * opacity
            yield px[max(int(start) - at, 0):].clamp(0.0, 1.0)
        at = end


def first_frames(pieces, count: int):
    """`pieces` cut off after `count` frames: a last window's view stops where its video does when the
    track ends first (`loop_plan.frames_kept`)."""
    left = int(count)
    for piece in pieces:
        if left <= 0:
            return
        yield piece[:left]
        left -= int(piece.shape[0])


def render_over(images: torch.Tensor, pieces):
    """Each piece of a mask view with the same frames of the render above it, [n, 2H, W, 3]: the mask review's
    picture. `images` are the window's frames as written (after the trim, and without a tail past the track),
    so the pieces must start where they do; the view stops where the render does."""
    yield from (torch.cat([images[at:at + int(piece.shape[0]), ..., :3].to(piece.dtype), piece], dim=1)
                for at, piece in _placed(first_frames(pieces, int(images.shape[0]))))


def _placed(pieces):
    at = 0
    for piece in pieces:
        yield at, piece
        at += int(piece.shape[0])


def _push_pull(image: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fill where `weight` is 0 from where it is not: halve until the hole closes, then come back up.

    `image` is [N, C, H, W] already multiplied by `weight` [N, 1, H, W].
    """
    known = weight > 0
    estimate = image / weight.clamp(min=1e-6)
    if bool(known.all()) or min(image.shape[-2:]) <= 1:
        return estimate
    coarse = _push_pull(F.avg_pool2d(image, 2, ceil_mode=True), F.avg_pool2d(weight, 2, ceil_mode=True))
    up = F.interpolate(coarse, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return torch.where(known, estimate, up)


def fill_subject(pixels: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
    """[F, H, W, 3] with the pixels under `hole` [F, H, W] replaced by a fill from their surroundings.

    Low frequency on purpose: these pixels are regenerated and never shown,
    and the fill only has to stop a kept token's encoder seeing the subject.
    Each frame is filled alone. Outside the hole nothing changes.
    """
    out = pixels.clone()
    for i in range(0, pixels.shape[0], CHUNK):
        h = (hole[i:i + CHUNK] > 0.5).to(pixels.dtype).unsqueeze(1)
        if not bool(h.any()):
            continue
        img = pixels[i:i + CHUNK].movedim(-1, 1)
        fill = _push_pull(img * (1.0 - h), 1.0 - h)
        out[i:i + CHUNK] = (img * (1.0 - h) + fill * h).movedim(1, -1)
    return out


def select_part(subject: torch.Tensor, part: torch.Tensor, margin: int = PART_MARGIN) -> torch.Tensor:
    """The part of a subject: `part` where it lies on `subject` widened by `margin` pixels. [N, H, W] each."""
    return ((part > 0.5) & (grow(subject.to(torch.float32), margin) > 0.5)).to(torch.float32)


def part_bottom(part: torch.Tensor) -> torch.Tensor:
    """Per frame, the lowest row a part reaches, or -1 where the frame has none. [N] long."""
    rows = (part > 0.5).any(dim=-1)                       # [N, H]
    index = torch.arange(rows.shape[-1], device=part.device)
    return torch.where(rows, index, torch.full_like(index, -1)).amax(dim=-1)


def carry_missing(bottoms: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Give each frame with no part (-1) the nearest found frame's value; (filled, how many were carried)."""
    found = (bottoms >= 0).nonzero().flatten()
    missing = (bottoms < 0).nonzero().flatten()
    if not found.numel() or not missing.numel():
        return bottoms, 0
    nearest = found[(missing[:, None] - found[None, :]).abs().argmin(dim=1)]
    out = bottoms.clone()
    out[missing] = bottoms[nearest]
    return out, int(missing.numel())


def above(subject: torch.Tensor, bottoms: torch.Tensor) -> torch.Tensor:
    """The subject's mask down to row `bottoms[n]` inclusive, per frame. [N, H, W]."""
    rows = torch.arange(subject.shape[-2], device=subject.device)[None, :, None]
    return ((subject > 0.5) & (rows <= bottoms[:, None, None])).to(torch.float32)


def soften_subject(pixels: torch.Tensor, hole: torch.Tensor, blur: int) -> torch.Tensor:
    """[F, H, W, 3] with the pixels under `hole` [F, H, W] reduced to their coarse shape: no colour, blurred.

    What a late start is given in place of the source inside the region.
    Luminance only, blurred by a Gaussian of `blur` pixels that averages
    inside the hole alone, so the background does not bleed in. Outside the
    hole nothing changes.
    """
    sigma = max(float(blur), 0.5)
    radius = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).to(pixels.device, pixels.dtype)

    def smooth(x: torch.Tensor) -> torch.Tensor:        # [N, 1, H, W]
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))

    out = pixels.clone()
    for i in range(0, pixels.shape[0], CHUNK):
        h = (hole[i:i + CHUNK] > 0.5).to(pixels.dtype).unsqueeze(1)
        if not bool(h.any()):
            continue
        grey = pixels[i:i + CHUNK, ..., :3].mean(dim=-1).unsqueeze(1)
        soft = smooth(grey * h) / smooth(h).clamp(min=1e-4)
        img = pixels[i:i + CHUNK].movedim(-1, 1)
        out[i:i + CHUNK] = (img * (1.0 - h) + soft.expand(-1, img.shape[1], -1, -1) * h).movedim(1, -1)
    return out


def top_of(hole: torch.Tensor, share: float) -> torch.Tensor:
    """The top `share` of the rows each frame's hole covers, [F, H, W] of 0 or 1. An empty frame stays empty."""
    on = hole > 0.5
    rows = on.any(dim=2)                                                    # [F, H]
    index = torch.arange(hole.shape[1], device=hole.device).unsqueeze(0)
    first = torch.where(rows, index, hole.shape[1]).min(dim=1).values
    last = torch.where(rows, index, -1).max(dim=1).values
    cut = first + (float(share) * (last - first + 1).clamp(min=0)).round().long()
    return (on & (index < cut.unsqueeze(1)).unsqueeze(2)).to(torch.float32)


def start_zero_tokens(source: dict, mask: torch.Tensor, tokens: torch.Tensor,
                      first_frame: int) -> torch.Tensor | None:
    """The regenerated tokens whose latent starts from nothing, [latent_t, lat_h, lat_w] of 0 or 1. None when the start is noise.

    `mask` is the window's fitted mask and `tokens` its token mask, as
    `window` returns them, and `first_frame` where the window starts in the
    source, which the margin is read by (`source_margins`); it has no default,
    so a caller cannot read another window's margins by leaving it out. The subject's
    tokens (the mask widened by half of the margin, where the softening is)
    are zeroed except those the top
    share touches. Tokens of the margin beyond that keep the source: they are
    background, and what they hold is what the composite wants there anyway.
    """
    if source.get("start_from", START_NOISE) == START_NOISE:
        return None
    hole = grow(mask, source_margins(source, first_frame, int(mask.shape[0]),
                                     int(mask.shape[1]) * int(mask.shape[2])) // 2)
    shape = tuple(int(n) for n in tokens.shape)
    body = token_mask(hole, *shape)
    keep = token_mask(top_of(hole, float(source["start_top"])), *shape)
    return (tokens > 0.5).to(torch.float32) * body * (1.0 - keep)


def window_frames(source: dict, first_frame: int, frames: int, width: int, height: int):
    """One window of a source on the render canvas: its fitted frames, its fitted mask, frames held.

    The source can run out inside a loop's last window: the missing frames
    repeat the last one, with the last frame's mask held with it
    (`held_mask`). A window that starts past the source's end is refused.
    """
    have = int(source["frames"].shape[0])
    first_frame, frames = int(first_frame), int(frames)
    if first_frame >= have:
        raise ValueError(
            f"the source video has {have} frames and this window starts at frame {first_frame}: "
            "load more of it (the loader's frame cap), or shorten the run")
    pixels = fit_frames(source["frames"][first_frame:first_frame + frames], width, height)
    mask = fit_mask(source["mask"][first_frame:first_frame + frames], width, height)
    short = frames - int(pixels.shape[0])
    if short > 0:
        pixels = torch.cat([pixels, pixels[-1:].expand(short, -1, -1, -1)], dim=0)
        mask = held_mask(mask, short)
    return pixels, mask, short


def held_mask(mask: torch.Tensor, short: int) -> torch.Tensor:
    """A window's fitted mask (the subject's, or `keep`'s, or `others`') with `short` frames added for the frames
    held past the source's end: the last frame's own, so the region stays open there. One function for the
    three masks, so the region, what is kept in it and who is kept out of it stay in step on the held frames."""
    return torch.cat([mask, mask[-1:].expand(short, -1, -1)], dim=0)


def pad_context(mask: torch.Tensor, before: int, after: int, shows: str = CONTEXT_AS_FILMED) -> torch.Tensor:
    """A shot's [n, H, W] mask on its padded window, [before + n + after, H, W] float (`CONTEXT_SHOWS` has the rule):
    empty on the context frames `as filmed`; `face covered`, the shot's first frame's mask repeated over the frames
    before and its last frame's over the frames after. The subject's mask and the tracked one go through this; the
    keep-outs are padded empty in both forms."""
    if shows not in CONTEXT_SHOWS:
        raise ValueError(f"unknown context_shows {shows!r}; it is `{CONTEXT_AS_FILMED}` or `{CONTEXT_FACE_COVERED}`")
    mask = mask.to(torch.float32)
    if shows == CONTEXT_AS_FILMED or not int(mask.shape[0]):
        return F.pad(mask, (0, 0, 0, 0, int(before), int(after)))
    return torch.cat([mask[:1].expand(int(before), -1, -1), mask, mask[-1:].expand(int(after), -1, -1)], dim=0)


def hold_context(x, before: int, after: int, *, src_before: int | None = None, src_after: int | None = None):
    """A padded window's [N, ...] tensor (frames, or a mask) with its first `before` frames copies of frame
    `src_before` (defaulting to `before`) and its last `after` copies of frame `src_after` (defaulting to
    `n - after - 1`) (`CONTEXT ACROSS A CUT IS HELD`): the shot's edge frame, the nearest frame inside the shot
    with a region at least `HELD_EDGE_LEAST` of the shot's middle region (`held_edge_source`), or the innermost
    real context frame where a side holds real ones too. The middle slice `x[before:n - after]` is untouched.
    None stays None. One function for the frames and every mask, so the picture, the region, what is kept in it
    and who is kept out of it stay in step on held frames, as `held_mask` keeps them past a source's end."""
    if x is None or not (before or after):
        return x
    n, before, after = int(x.shape[0]), int(before), int(after)
    if before + after >= n:
        raise ValueError(f"{before} held frame(s) before and {after} after leave none of a {n}-frame window to hold")
    sb = before if src_before is None else int(src_before)
    sa = (n - after - 1) if src_after is None else int(src_after)
    return torch.cat(([x[sb:sb + 1].expand(before, *x.shape[1:])] if before else []) + [x[before:n - after]]
                     + ([x[sa:sa + 1].expand(after, *x.shape[1:])] if after else []), dim=0)


def held_edge_source(mask: torch.Tensor, held_before: int, held_after: int) -> tuple[int | None, int | None, list[str]]:
    """For a shot's [n, H, W] region `mask` (before padding) and its held counts `(held_before, held_after)`:
    when a held side's edge frame (`0` for `before`, `n - 1` for `after`) has `edge == 0` or
    `edge < HELD_EDGE_LEAST * middle`, finds the nearest frame inside the shot with `area >= HELD_EDGE_LEAST * middle`
    (searching forward from `0` for `before`, backward from `n - 1` for `after`). Returns
    `(shot_before, shot_after, lines)` where `shot_before` / `shot_after` is `None` when the edge frame itself is
    good, the stepped-to shot frame index (`0 .. n - 1`) when a nearest good frame is found, or `-1` when no frame in
    the shot has a mask (`on.numel() == 0`), and `lines` are the `held_fell_back` strings."""
    n_shot = int(mask.shape[0])
    area = (mask > 0.5).flatten(1).sum(dim=1)
    on = area[area > 0]
    middle = float(on.float().median()) if on.numel() else 0.0
    least = HELD_EDGE_LEAST * middle
    picked: dict[str, int | None] = {"before": None, "after": None}
    out: list[str] = []
    for side, held, k, which, order in (
        ("before", held_before, 0, "first", range(n_shot)),
        ("after", held_after, n_shot - 1, "last", range(n_shot - 1, -1, -1)),
    ):
        edge = int(area[k]) if n_shot else 0
        if held and (edge == 0 or edge < least):
            good = next((j for j in order if int(area[j]) > 0 and int(area[j]) >= least), -1) if on.numel() else -1
            picked[side] = good
            if good >= 0:
                out.append(
                    f"{side}: the shot's {which} frame's region is {edge} px, under {HELD_EDGE_LEAST:g} of the shot's "
                    f"middle {middle:.0f} px, so its {held} context frame(s) there hold shot frame {good} ({int(area[good])} px)"
                )
            else:
                out.append(
                    f"{side}: the shot's {which} frame's region is {edge} px, under {HELD_EDGE_LEAST:g} of the shot's "
                    f"middle {middle:.0f} px, so its {held} context frame(s) there stay the source's real frames"
                )
    return picked["before"], picked["after"], out


def held_edge_falls_back(mask: torch.Tensor, held_before: int, held_after: int) -> list[str]:
    """Of a shot's held sides, the ones whose edge frame's region is too small to hold (`HELD_EDGE_LEAST`), each said
    as a line: which side, the edge frame's area, the shot's middle one, and which nearest good shot frame was held
    in its place (or that real frames were kept when the whole shot has no mask). `mask` is the shot's own [n, H, W]
    region, before it is padded. A side that holds nothing is not read."""
    return held_edge_source(mask, held_before, held_after)[2]


def context_across_cut(cuts, before: int, n_shot: int, after: int) -> list[int]:
    """The context frames of a padded window (window frames, 0 at its first) that lie across a cut from the shot: a
    frame before it with a cut between it and the shot's first frame, or after it with a cut between the shot's last
    frame and it. `cuts` are window frames that start a shot (`source_cuts`). Under `face covered` the mask held there
    is the shot's edge frame's, and the subject may stand elsewhere in that other shot, or not be in it."""
    end = int(before) + int(n_shot)
    cut = sorted({int(c) for c in cuts or []})
    return ([f for f in range(int(before)) if any(f < c <= int(before) for c in cut)]
            + [f for f in range(end, end + int(after)) if any(end <= c <= f for c in cut)])


def runs_said(frames) -> str:
    """Frames as runs for a log line: `0..13, 127..140`."""
    runs: list[list[int]] = []
    for f in sorted(int(x) for x in frames):
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return ", ".join(f"{a}..{b}" if b > a else str(a) for a, b in runs)


def wired_motion(source: dict, first_frame: int, frames: int, width: int, height: int, which: int = 1) -> torch.Tensor:
    """One window of the video wired as the motion reference, [F, h, w, 3], cut where the window is cut.

    The wired video runs beside the source, frame for frame, so a window that starts at source frame k is shown
    the wired video from frame k: the same slice `window_frames` takes of the source. It is fitted to the canvas
    as the source's frames are and then scaled to `motion_short_edge`, like every other motion reference; a
    source that runs out inside the last window repeats its last frame, as the source's own frames do.

    `which` is 1 for `motion_video` and 2 for `motion_video_2`, which has its own short edge and is cut the same way.
    """
    if which not in (1, 2):
        raise ValueError(f"a source carries two wired videos, 1 and 2; asked for {which!r}")
    tail = "" if which == 1 else "_2"
    video = source.get("motion_frames" + tail)
    if video is None:
        raise ValueError(f"motion_reference `{source.get('motion_reference', MOTION_WIRED)}` and no `motion_video{tail}` "
                         "on the source: wire one")
    have = int(video.shape[0])
    first_frame, frames = int(first_frame), int(frames)
    if first_frame >= have:
        raise ValueError(f"the wired motion video has {have} frames and this window starts at frame {first_frame}")
    pixels = fit_frames(video[first_frame:first_frame + frames], width, height)
    short = frames - int(pixels.shape[0])
    if short > 0:
        pixels = torch.cat([pixels, pixels[-1:].expand(short, -1, -1, -1)], dim=0)
    blank = torch.zeros(tuple(pixels.shape[:3]), dtype=torch.float32, device=pixels.device)
    return motion_reference(pixels, blank, MOTION_FRAME, int(source["motion_short_edge" + tail]), 0)


def _tracked_boxes(mask: torch.Tensor) -> torch.Tensor:
    """The subject's box on each frame of a [N, H, W] mask: [N, 4] long, (x0, y0, x1, y1) with the far side
    exclusive, a row of -1 where the mask is empty. The pack's one box function, `sapiens2_parts.mask_boxes`.

    Kept as one function on purpose. The shot table is to carry the subject's box per frame; that day this
    reads the table and computes nothing, and nothing else changes.
    """
    from .sapiens2_parts import mask_boxes   # imported here: that module imports this one at load
    return mask_boxes(mask)


def fit_boxes(boxes: torch.Tensor, old_w: int, old_h: int, width: int, height: int) -> torch.Tensor:
    """Boxes on `old_w` by `old_h` frames, on the render canvas, through the crop `fit_frames` makes.

    Never inside the fitted mask's own box: a near edge rounds down and a far edge up. A box the crop leaves
    nothing of, like a frame with no subject, is a row of -1.
    """
    old_w, old_h, width, height = int(old_w), int(old_h), int(width), int(height)
    x, y = _fit_crop(old_w, old_h, width, height)
    sx, sy = width / (old_w - 2 * x), height / (old_h - 2 * y)
    b = boxes.to(torch.float64)
    out = torch.stack([torch.floor((b[:, 0] - x) * sx).clamp(0, width), torch.floor((b[:, 1] - y) * sy).clamp(0, height),
                       torch.ceil((b[:, 2] - x) * sx).clamp(0, width), torch.ceil((b[:, 3] - y) * sy).clamp(0, height)],
                      dim=1).to(torch.long)
    out[(boxes[:, 0] < 0) | (out[:, 2] <= out[:, 0]) | (out[:, 3] <= out[:, 1])] = -1
    return out


def shot_ranges(source: dict) -> list[tuple[int, int]]:
    """(first frame, last frame) of each shot in the table a source carries, in order; empty with no table."""
    text = str(source.get("shot_table") or "")
    if not text:
        return []
    import json   # here, not at the top: the one use
    try:
        return [(int(s["first_frame"]), int(s["last_frame"])) for s in json.loads(text)["shots"]]
    except (ValueError, KeyError, TypeError):
        logger.warning("[h3] the source's shot table could not be read for its cuts; the zoom takes one box for the window")
        return []


def shot_boxes(boxes: torch.Tensor, ranges: list[tuple[int, int]], width: int, height: int,
               room: int = MOTION_BOX_ROOM, first_frame: int = 0) -> torch.Tensor:
    """One fixed box per shot: [N, 4] long, every frame carrying its shot's box, -1 through a shot the subject is never in.

    `boxes` are the subject's own, one a frame, for frames `first_frame` onwards of the clip `ranges` counts
    in. A shot's box is the union of the subject's boxes over those of its frames that are here, widened by
    `room` and then outward to the canvas multiple, inside the frame. A union, so a subject that is covered
    and uncovered inside a shot does not move the box; fixed, so the subject's travel is still travel. With no
    ranges the frames are one shot. Frames no range covers take the shot before them.
    """
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE as multiple   # core's constant, as `motion_reference`
    n, width, height = int(boxes.shape[0]), int(width), int(height)
    out = torch.full((n, 4), -1, dtype=torch.long)
    starts = sorted({min(max(int(a) - int(first_frame), 0), n) for a, _b in ranges} | {0})
    for a, b in zip(starts, starts[1:] + [n]):
        rows = boxes[a:b]
        rows = rows[rows[:, 0] >= 0]
        if not int(rows.shape[0]):
            continue
        x0, y0 = int(rows[:, 0].min()) - int(room), int(rows[:, 1].min()) - int(room)
        x1, y1 = int(rows[:, 2].max()) + int(room), int(rows[:, 3].max()) + int(room)
        x0, y0 = max(x0 // multiple * multiple, 0), max(y0 // multiple * multiple, 0)
        x1, y1 = min(-(-x1 // multiple) * multiple, width), min(-(-y1 // multiple) * multiple, height)
        out[a:b] = torch.tensor([x0, y0, x1, y1], dtype=torch.long)
    return out


def window_boxes(source: dict, first_frame: int, frames: int, width: int, height: int) -> torch.Tensor | None:
    """`shot_boxes` for one window of a source, on the render canvas; None when the source carries no boxes.

    The box of a shot is taken over the window's own frames of it, so a subject that shrinks through a long
    shot is framed for this window and not for the whole shot. Frames past the source's end, which
    `window_frames` fills with the last one and its mask, carry the last frame's box.
    """
    # whose box: a pass that replaces a part frames the part, a whole-subject pass the tracked person. A rule
    # and not an input since 2026-10-10 (`motion_box` chose between the two for a day; no render showed a
    # difference, and a face pass framed on the whole person shows a face small in a box the size of a person)
    whole = source.get("replace", REPLACE_WHOLE) == REPLACE_WHOLE
    rows = source.get("subject_boxes") if whole else source.get("region_boxes")
    if rows is None:
        return None
    first_frame, frames = int(first_frame), int(frames)
    part = rows[first_frame:first_frame + frames]
    short = frames - int(part.shape[0])
    if short > 0:
        part = torch.cat([part, part[-1:].expand(short, -1)], dim=0)
    fitted = fit_boxes(part, int(source["frames"].shape[2]), int(source["frames"].shape[1]), width, height)
    return shot_boxes(fitted, shot_ranges(source), width, height, first_frame=first_frame)


def _reference_size(h: int, w: int, short_edge: int) -> tuple[int, int]:
    """(height, width) of the whole frame as a reference: the shorter side `short_edge`, rounded to the canvas multiple."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE  # core's constant; imported here so a check needs no server
    scale = float(short_edge) / float(min(h, w))
    return (max(CANVAS_MULTIPLE, int(round(h * scale / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, int(round(w * scale / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE))


def zoom_plan(boxes: torch.Tensor, h: int, w: int, short_edge: int):
    """How `subject only, zoomed in` lays out a window of `h` by `w` frames: (height, width, runs), or None.

    `boxes` is `shot_boxes`' result for the window. Each run is (first frame, stop, box, scale, (height,
    width) of the box once scaled) for a stretch of frames that share a box; frames in no run have no subject
    and are grey. None means "show the whole frame, as `subject only` does": every box is the whole frame, no
    frame has a subject, or nothing finer than the whole frame fits the budget.

    The rule, as four things that are always true of a plan (`bench/check_video_mask.py` holds each):
    1. the picture holds at most the pixels the whole-frame reference at `short_edge` holds, so the zoom
       never costs more than `subject only`, with no widget of its own;
    2. every shot is shown at a scale at least the whole frame's, so the subject is never smaller than in
       `subject only`;
    3. a box that is the whole frame gives `subject only`'s picture, value for value (the None above);
    4. no shot is enlarged past the canvas's own pixels, as the reference compiler never enlarges a video.
    A video has one size, so a window's shots share one picture and each box is fitted inside it at its own
    scale, centred on grey. The picture's shape is chosen among those inside the budget, on the canvas
    multiple: the one that shows the window's frames largest, with no shot under the whole frame's scale.
    The whole-frame reference's own shape is always among them, so there is always a plan or a None.
    """
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE as multiple
    h, w, n = int(h), int(w), int(boxes.shape[0])
    th, tw = _reference_size(h, w, short_edge)
    budget, floor = th * tw, min(th / h, tw / w)
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or not torch.equal(boxes[i], boxes[start]):
            box = tuple(int(v) for v in boxes[start])
            if box[0] >= 0:
                runs.append((start, i, box))
            start = i
    if not runs or floor >= 1.0 or all(box == (0, 0, w, h) for _a, _z, box in runs):
        return None
    import math   # here, not at the top: the one use
    best = None
    for out_h in range(multiple, budget // multiple + 1, multiple):
        for out_w in range(multiple, budget // out_h // multiple * multiple + 1, multiple):
            scales = [min(1.0, out_h / (y1 - y0), out_w / (x1 - x0)) for _a, _z, (x0, y0, x1, y1) in runs]
            if min(scales) < floor - 1e-12:
                continue
            # how large the window's frames are shown: the mean of log scale over frames, so a shot counts by
            # its length and doubling one shot weighs what doubling another does; then the smaller picture
            shown = sum(math.log(s) * (z - a) for (a, z, _box), s in zip(runs, scales))
            key = (round(shown, 9), -out_h * out_w, -out_h)
            if best is None or key > best[0]:
                best = (key, out_h, out_w, scales)
    if best is None:
        return None
    _key, out_h, out_w, scales = best
    return out_h, out_w, [(a, z, box, s, (max(1, int((box[3] - box[1]) * s + 1e-9)), max(1, int((box[2] - box[0]) * s + 1e-9))))
                          for (a, z, box), s in zip(runs, scales)]


def zoom_note(boxes: torch.Tensor, h: int, w: int, short_edge: int) -> str:
    """One clause for a report: what `zoom_plan` made of a window's boxes."""
    plan = zoom_plan(boxes, h, w, short_edge)
    if plan is None:
        return "the whole frame is shown, as `subject only`: the subject's box is the frame, is absent, or cannot be shown larger"
    out_h, out_w, runs = plan
    return (f"{len(runs)} box(es) on the canvas, "
            + ", ".join(f"{x1 - x0}x{y1 - y0} at {scale:.2f} of canvas scale" for _a, _z, (x0, y0, x1, y1), scale, _size in runs)
            + f"; the picture is {out_w}x{out_h}")


def motion_reference(pixels: torch.Tensor, mask: torch.Tensor, mode: str, short_edge: int, margin,
                     boxes: torch.Tensor | None = None):
    """The window as the model is shown it as a video reference, [F, h, w, 3], or None for `none`.

    `subject only` keeps the pixels under the mask widened by `margin` (one
    number, or each frame's own as `grow` takes it) and sets
    the rest to mid grey, so the reference carries how the subject moves and
    nothing of the scene the kept rows already hold. `whole frame` keeps the
    window as it is. Either is scaled so its shorter side is `short_edge`,
    rounded to the canvas multiple with the aspect kept; the reference
    compiler never enlarges a video, so this is what sets its cost.

    `subject only, zoomed in` is `subject only` shown in `boxes` (`window_boxes`:
    one fixed box per shot, around the tracked subject or the replaced part) and not in the whole
    frame. `zoom_plan` lays it out and states the rule: never more pixels than
    `subject only` at this `short_edge`, never a smaller subject, the same
    picture when the box is the frame, never enlarged past the canvas.
    """
    if mode == MOTION_NONE:
        return None
    if mode not in MOTIONS:
        raise ValueError(f"unknown motion_reference {mode!r}; one of {list(MOTIONS)}")
    n, h, w = int(pixels.shape[0]), int(pixels.shape[1]), int(pixels.shape[2])
    th, tw = _reference_size(h, w, short_edge)
    keep = grow(mask.to(torch.float32), margin) if mode in (MOTION_SUBJECT, MOTION_ZOOM) else None
    plan = None
    if mode == MOTION_ZOOM:
        if boxes is None or int(boxes.shape[0]) != n:
            raise ValueError(
                f"motion_reference `{MOTION_ZOOM}` needs the subject's box on each of the window's {n} frames "
                "(`window_boxes`); the source record carries them as `subject_boxes`, and as `region_boxes` for a "
                "pass that replaces a part")
        plan = zoom_plan(boxes, h, w, short_edge)
    if plan is not None:
        out_h, out_w, runs = plan
        shown = torch.full((n, out_h, out_w, 3), 0.5, dtype=torch.float32, device=pixels.device)
        for first, stop, (x0, y0, x1, y1), _scale, (sh, sw) in runs:
            top, left = (out_h - sh) // 2, (out_w - sw) // 2
            for i in range(first, stop, CHUNK):
                j = min(i + CHUNK, stop)
                chunk = pixels[i:j, y0:y1, x0:x1, :3].to(torch.float32)
                m = (keep[i:j, y0:y1, x0:x1] > 0.5).to(chunk.dtype).unsqueeze(-1)
                chunk = chunk * m + 0.5 * (1.0 - m)
                if (sh, sw) != (y1 - y0, x1 - x0):
                    chunk = F.interpolate(chunk.movedim(-1, 1), size=(sh, sw), mode="bilinear",
                                          align_corners=False, antialias=True).movedim(1, -1)
                shown[i:j, top:top + sh, left:left + sw] = chunk.clamp(0.0, 1.0)
        return shown
    out = []
    for i in range(0, n, CHUNK):
        chunk = pixels[i:i + CHUNK, ..., :3].to(torch.float32)
        if keep is not None:
            m = (keep[i:i + CHUNK] > 0.5).to(chunk.dtype).unsqueeze(-1)
            chunk = chunk * m + 0.5 * (1.0 - m)
        if (th, tw) != (h, w):
            chunk = F.interpolate(chunk.movedim(-1, 1), size=(th, tw), mode="bilinear",
                                  align_corners=False, antialias=True).movedim(1, -1)
        out.append(chunk.clamp(0.0, 1.0))
    return torch.cat(out, dim=0)


#: The preview strip: this many frames sampled evenly over the clip, each row this tall. Reasoned.
PREVIEW_ROWS = 6
PREVIEW_HEIGHT = 192


def preview_strip(frames: torch.Tensor, mask: torch.Tensor, grow_pixels, motion: str, short_edge: int,
                  margin, boxes: torch.Tensor | None = None, others: torch.Tensor | None = None,
                  wired: torch.Tensor | None = None, wired_2: torch.Tensor | None = None,
                  short_edge_2: int | None = None) -> torch.Tensor:
    """[1, H, W, 3]: sampled frames down the strip, the regenerated region tinted red on the plate, and, when a
    motion reference is on, what the encoder is shown beside each. The dry-run review looks at this and the
    tracker's tiles before anything samples.

    With `boxes` (the zoomed reference's box on every frame of the clip) each plate also carries its box as an
    outline, so the picture beside it can be found on the frame. The boxes here are each shot's over the whole
    clip; a window takes its own from its own frames (`window_boxes`), which is never larger.

    `grow_pixels` and `margin` are one number each, or every frame's own over the clip (`margins`).

    With `others` (the people the margin does not grow over) their pixels are tinted blue and taken out of the
    red wherever they are not the subject's own. Pixel for pixel, where the model works in whole tokens
    (`window`): a look at who is kept out, not the token mask."""
    n = int(frames.shape[0])
    idx = torch.linspace(0, n - 1, steps=min(PREVIEW_ROWS, n)).round().long()
    f = frames[idx, ..., :3].to(torch.float32)
    grow_pixels = grow_pixels[idx] if torch.is_tensor(grow_pixels) else grow_pixels
    margin = margin[idx] if torch.is_tensor(margin) else margin
    region = (grow(mask[idx].to(torch.float32), grow_pixels) > 0.5).unsqueeze(-1).to(f.dtype)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=f.dtype, device=f.device)
    away = None
    if others is not None:
        away = ((others[idx] > 0.5) & ~(mask[idx] > 0.5)).unsqueeze(-1).to(f.dtype)
        region = region * (1.0 - away)
    plate = f * (1.0 - region) + (0.5 * f + 0.5 * red) * region
    if away is not None:
        blue = torch.tensor([0.0, 0.3, 1.0], dtype=f.dtype, device=f.device)
        plate = plate * (1.0 - away) + (0.5 * f + 0.5 * blue) * away
    shown = None if boxes is None else boxes[idx]
    if shown is not None:
        cyan = torch.tensor([0.0, 1.0, 1.0], dtype=f.dtype, device=f.device)
        line = max(2, int(f.shape[1]) // 128)                      # an outline that survives the tile's scale-down
        for row, (x0, y0, x1, y1) in enumerate(shown.tolist()):
            if x0 < 0:
                continue
            plate[row, y0:y0 + line, x0:x1] = cyan
            plate[row, max(y1 - line, y0):y1, x0:x1] = cyan
            plate[row, y0:y1, x0:x0 + line] = cyan
            plate[row, y0:y1, max(x1 - line, x0):x1] = cyan
    tiles = [plate]
    if motion == MOTION_WIRED:
        # the wired video's own frames, fitted as the plate is: what the model is shown as movement (`wired_motion`)
        shown_wired = None if wired is None else fit_frames(wired[idx], int(f.shape[2]), int(f.shape[1]))
        ref = None if shown_wired is None else motion_reference(shown_wired, mask[idx], MOTION_FRAME, short_edge, 0, shown)
    else:
        ref = motion_reference(frames[idx], mask[idx], motion, short_edge, margin, shown)
    if ref is not None:
        tiles.append(ref.to(f.dtype).to(f.device))
    if motion == MOTION_WIRED and wired_2 is not None:
        # the second wired video, beside the first, at its own short edge
        second = motion_reference(fit_frames(wired_2[idx], int(f.shape[2]), int(f.shape[1])), mask[idx], MOTION_FRAME,
                                  short_edge if short_edge_2 is None else short_edge_2, 0)
        tiles.append(second.to(f.dtype).to(f.device))
    h = PREVIEW_HEIGHT
    scaled = []
    for t in tiles:
        w = max(8, int(round(int(t.shape[2]) * h / int(t.shape[1]))))
        scaled.append(F.interpolate(t.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False,
                                    antialias=True).movedim(1, -1))
    row = torch.cat(scaled, dim=2)                                   # the plate and the reference side by side
    strip = row.reshape(1, int(row.shape[0]) * h, int(row.shape[2]), 3)   # the sampled frames down the strip
    return strip.clamp(0.0, 1.0)


def stored_zoom(latent_path: str):
    """What a stored window's own region file says of its framing, for a run that continues it: None (the whole
    frame), its `zoom` record (one box of the frames it was handed), or `NO_REGION_FILE`. The file is the one
    the song node writes beside the window's latent (`loop_resume.region_path`); only its header is read."""
    path = str(latent_path)[:-len(".safetensors")] + "_region.npz" if str(latent_path).endswith(".safetensors") else ""
    if not path or not os.path.isfile(path):
        return NO_REGION_FILE
    with np.load(path) as z:
        return json.loads(str(z["meta"])).get("zoom")


def box_sides_cut(tokens: torch.Tensor, zoom: dict) -> list[str]:
    """The sides of a zoomed window's box its regenerated region reaches, where the box's side is inside the
    frame: the region is cut off there (the picture ends), and `lay_back` fades it into the whole frame at the
    box's edge. A side of the box that is the frame's own edge is not named: a region at the frame's edge is
    the frame's. A smaller zoom would hold more of the picture (`zoom_box`)."""
    old_w, old_h = (int(v) for v in zoom["source_size"])
    width, height = (int(v) for v in zoom["canvas"])
    cx, cy = _fit_crop(old_w, old_h, width, height)
    x0, y0, x1, y1 = (int(v) for v in zoom["box"])
    on = tokens > 0.5
    sides = (("left", on[:, :, 0], x0 - cx > 1), ("top", on[:, 0, :], y0 - cy > 1),
             ("right", on[:, :, -1], old_w - cx - x1 > 1), ("bottom", on[:, -1, :], old_h - cy - y1 > 1))
    return [name for name, edge, inside in sides if inside and bool(edge.any())]


def region_height(boxes: torch.Tensor, old_h: int, height: int, cy: int) -> float | None:
    """The replaced region's height on the canvas at a zoom of one: the median over the frames that have one of its
    box's height (`boxes` [N, 4] on the load's own frames, a row of -1 where there is none), through the fit to the
    canvas (`_fit_crop`). None when no frame has a region. The median, so a frame where hair flies or the part
    model takes the neck moves it little."""
    rows = boxes[boxes[:, 0] >= 0]
    if not int(rows.shape[0]):
        return None
    tall = (rows[:, 3] - rows[:, 1]).to(torch.float64)
    return float(tall.median()) * float(height) / float(int(old_h) - 2 * int(cy))


def zoom_box(boxes: torch.Tensor, old_w: int, old_h: int, width: int, height: int, largest: float | None = None,
             target: float | None = None, any_load: float | None = None):
    """The one box a load is rendered in, and what set it: ((x0, y0, x1, y1), one of `SET_BY_BOX`, `SET_BY_LARGEST`,
    `SET_BY_HEAD`, `SET_BY_ANY`), or (None, why not).

    `boxes` are [N, 4] on the load's own `old_w` by `old_h` frames, a row of -1 where there is none. The box is
    their union over the load, with `ZOOM_ROOM` round it, taken out to the canvas's shape, and never smaller than
    the zoom allows: at most `largest`, at most the zoom that takes the region's height (`region_height`) to
    `target` canvas pixels, and never held under `any_load` by the target (the constants' comment has the rule).
    Each left as None is its constant (`largest` and `any_load` `ZOOM_LARGEST`, `target` `ZOOM_HEAD_TARGET`), read when
    called. With `largest` no more than `any_load` (the default) this is the rule of 2026-10-10 (the box, or a cap). In
    whole EVEN pixels of the load's frames, far side exclusive, exactly the canvas's proportions, inside the part of
    the frame the canvas shows (`_fit_crop`), moved inside it and not shrunk. Fixed for the load: a box that
    followed the subject would turn the subject's travel into the background's. Even and whole, so a tool outside
    can cut the same box from the file (chroma is half size). None when the subject is in no frame, or when the box
    comes to under `ZOOM_SMALLEST` times: a subject that crosses the frame lands there, and is rendered on the
    whole frame.
    """
    import math   # here, not at the top: the one use
    old_w, old_h, width, height = int(old_w), int(old_h), int(width), int(height)
    cx, cy = _fit_crop(old_w, old_h, width, height)
    shown_w, shown_h = old_w - 2 * cx, old_h - 2 * cy
    rows = boxes[boxes[:, 0] >= 0]
    if not int(rows.shape[0]):
        return None, "the subject is in no frame of the load"
    x0, y0 = max(int(rows[:, 0].min()), cx), max(int(rows[:, 1].min()), cy)
    x1, y1 = min(int(rows[:, 2].max()), old_w - cx), min(int(rows[:, 3].max()), old_h - cy)
    if x1 <= x0 or y1 <= y0:
        return None, "the subject is outside what the canvas shows of the frame"
    # the canvas's shape in its smallest whole terms, doubled: the two cannot both be even, and the box is
    g = math.gcd(width, height)
    unit_w, unit_h = 2 * width // g, 2 * height // g
    by_box = max((x1 - x0) * (1.0 + 2.0 * ZOOM_ROOM) / unit_w, (y1 - y0) * (1.0 + 2.0 * ZOOM_ROOM) / unit_h)
    # the most the zoom may be: the region's height to the target, between the zoom any load may take and the largest
    tall = region_height(boxes, old_h, height, cy)
    largest = max(float(ZOOM_LARGEST if largest is None else largest), 1.0)
    any_load = max(float(ZOOM_LARGEST if any_load is None else any_load), 1.0)
    target = float(ZOOM_HEAD_TARGET if target is None else target)
    to_target = target / tall if tall else float("inf")
    most = max(min(largest, to_target), min(any_load, largest))
    by_cap = shown_w / (most * unit_w)
    k = max(1, math.ceil(max(by_box, by_cap) - 1e-9))
    box_w, box_h = k * unit_w, k * unit_h
    left_lo, top_lo = cx + cx % 2, cy + cy % 2
    left_hi, top_hi = (old_w - cx - box_w) // 2 * 2, (old_h - cy - box_h) // 2 * 2
    if shown_w / box_w < ZOOM_SMALLEST - 1e-9 or left_hi < left_lo or top_hi < top_lo:
        return None, (f"the subject with its room takes {box_w}x{box_h} of the {shown_w}x{shown_h} the canvas shows, "
                      f"which is under {ZOOM_SMALLEST:g} times")
    left = min(max(int(round(((x0 + x1) / 2.0 - box_w / 2.0) / 2.0)) * 2, left_lo), left_hi)
    top = min(max(int(round(((y0 + y1) / 2.0 - box_h / 2.0) / 2.0)) * 2, top_lo), top_hi)
    if by_box > by_cap:
        why = SET_BY_BOX
    elif most >= largest:
        why = SET_BY_LARGEST
    elif most > any_load:
        why = SET_BY_HEAD
    else:
        why = SET_BY_ANY
    return (left, top, left + box_w, top + box_h), why


def crop_frames(frames: torch.Tensor, box, width: int, height: int) -> torch.Tensor:
    """[N, H, W, C] cut to `box` and scaled to [N, height, width, 3]: a whole-pixel cut, then the call core fits
    a clip with (`fit_frames`), with no crop of its own because `zoom_box` gives the canvas's proportions."""
    x0, y0, x1, y1 = (int(v) for v in box)
    out = [comfy.utils.common_upscale(frames[i:i + CHUNK, y0:y1, x0:x1, :3].movedim(-1, 1), int(width), int(height),
                                      "bilinear", "disabled").movedim(1, -1)
           for i in range(0, int(frames.shape[0]), CHUNK)]
    return torch.cat(out, dim=0)


def crop_mask(mask: torch.Tensor, box, width: int, height: int) -> torch.Tensor:
    """[N, H, W] cut to `box` and brought to [N, height, width] of 0 or 1, as `fit_mask` brings a mask to the
    canvas (pooled when it is made smaller, so nothing is lost; interpolated when larger), then cut at a half:
    the mask a tracker run on the zoomed picture would hand over."""
    x0, y0, x1, y1 = (int(v) for v in box)
    out = [(fit_mask(mask[i:i + CHUNK, y0:y1, x0:x1], width, height) > 0.5).to(torch.float32)
           for i in range(0, int(mask.shape[0]), CHUNK)]
    return torch.cat(out, dim=0)


def _canvas_box(zoom: dict) -> tuple[float, float, float, float]:
    """A zoom's box on the whole-frame canvas, (x0, y0, x1, y1) and not whole numbers: through the fit
    `fit_frames` makes of the load's frames (`_fit_crop`, then each side by its own scale)."""
    old_w, old_h = (int(v) for v in zoom["source_size"])
    width, height = (int(v) for v in zoom["canvas"])
    cx, cy = _fit_crop(old_w, old_h, width, height)
    sx, sy = width / (old_w - 2 * cx), height / (old_h - 2 * cy)
    x0, y0, x1, y1 = (int(v) for v in zoom["box"])
    return (x0 - cx) * sx, (y0 - cy) * sy, (x1 - cx) * sx, (y1 - cy) * sy


def render_zoom_note(zoom: dict, source: dict) -> str:
    """One line for a report: the box a load is rendered in, the zoom it came to and what set it, and how much
    of that zoom is enlargement of the pixels the node was handed (all of it, when the graph's loader had
    already fitted the clip to the canvas)."""
    x0, y0, x1, y1 = zoom["box"]
    old_w, old_h = zoom["source_size"]
    width, height = zoom["canvas"]
    scale, enlarged = float(zoom["scale"]), float(width) / float(x1 - x0)
    return (f"ZOOMED {scale:.2f}x: rendered {zoom['render_region']}: one box for the load, {x1 - x0}x{y1 - y0} at ({x0}, {y0}) "
            f"of its {old_w}x{old_h} frames, shown {scale:.2f} times larger than on the whole frame (set by "
            + {SET_BY_LARGEST: f"the largest zoom, {float(zoom.get('up_to', ZOOM_LARGEST)):g}",
               SET_BY_HEAD: f"the replaced region's height to {ZOOM_HEAD_TARGET:g} canvas px, the load asking up to "
                            f"{float(zoom.get('up_to', ZOOM_LARGEST)):g}",
               SET_BY_ANY: f"the zoom any load may take, {ZOOM_LARGEST:g}: the region reaches {ZOOM_HEAD_TARGET:g} canvas px under it"}
            .get(zoom["set_by"], "the box with its room")
            + f"); the node was handed {old_w}x{old_h} frames for a {width}x{height} canvas, so "
            + (f"each pixel it was handed is shown enlarged {enlarged:.2f} times: the surroundings the model sees are "
               "the handed frames enlarged by that much, not a sharper picture" if enlarged > 1.0 + 1e-9
               else "nothing is enlarged: the box holds at least the canvas's pixels")
            + ("" if (old_w, old_h) != (width, height) else
               " (the frames arrived already at the canvas's size: a loader that hands the clip's own frames would "
               "give the box more pixels)")
            + f"; grow and feather are pixels of the zoomed picture, so {int(source['grow_pixels'])} and "
            f"{int(source['feather_pixels'])} px there are about {int(source['grow_pixels']) / scale:.0f} and "
            f"{int(source['feather_pixels']) / scale:.0f} px on the whole frame; each window's video is the whole "
            "frame with the region laid back, the zoomed picture is written beside it, and the mask review is of the "
            "ZOOMED picture, not of the whole frame")


def wired_under_zoom_note(source: dict) -> str:
    """One line for a report: a zoomed run's wired motion video was handed on as it was wired (`zoomed_source`)."""
    videos = "videos are" if source.get("motion_frames_2") is not None else "video is"
    return (f"ZOOMED, and the wired motion {videos} NOT cut to the box: the model is shown what a whole-frame run of this "
            "load would be shown. A video with a framing of its own (the body model cut to the head) is the same "
            "reference either way; one that runs beside the whole frame shows its figure at the frame's place and "
            "size, beside a plate that is the box enlarged")


def zoomed_source(source: dict, width: int, height: int, continued=NOT_CONTINUED) -> tuple[dict, list[str]]:
    """A source as the render runs on it, and the report's lines: the source itself under `the whole frame`,
    or a copy whose pictures are one box of the load scaled to the canvas.

    Made once a run, by the song node, because the canvas is known there and not at the Masked Source. In the
    copy `frames`, `mask`, `keep` and `others` are the box at the canvas's size, the two
    box tables and the subject's area are of that picture, `zoom` says what was done (`box` in whole even
    pixels of the load's own frames, `source_size`, `canvas`, `scale`, `render_region`, `set_by`) and `whole`
    is the source as it came. Everything between here and the composite then runs unchanged on a clip that
    happens to be a crop: `window`, the margins, the token region, the motion reference, the encode, the
    sampler, the decode and `lay_window`. `lay_back` puts the result into the whole frame.

    One box for the whole LOAD, not one a window: a continuation's kept frames are the window before's latent,
    so two windows with two boxes would hand the model a context of another framing. For the same reason a
    load with a cut inside it is refused: a box held across a cut frames nobody on one side of it. The pictures
    are made on every run; nothing keeps them.

    A ZOOMED RUN CONTINUES NOTHING AND IS NOT CONTINUED (`continue_from`). A continuation is a separate run on
    another load, and its kept frames are the stored window's latent, in that window's framing: `continued` is
    what that window's region file says (`stored_zoom`). A whole-frame run on a zoomed window is refused, and
    so is a zoomed run on any stored window. For a day a zoomed run could take the stored window's box; no
    render had done it, and it was removed on 2026-10-10. A zoomed choice that comes to the whole frame (the
    subject too large for a zoom) is a whole-frame run for this.

    A WIRED MOTION VIDEO IS HANDED ON AS IT WAS WIRED (2026-10-13; refused from 2026-10-10 until then). The zoom
    does nothing to it: `motion_frames`, `motion_frames_2` and their short edges are the copy's as they are the
    source's, so `wired_motion` gives a zoomed window the same pictures, to the bit, that it gives the same window
    of a whole-frame run. What that means depends on the video, and the node cannot tell which it was given:
    - a video with a framing of its own (the body model cut to one box round the head, `bench/head_reference.py`)
      is in neither the frame's coordinates nor the box's on a whole-frame run either, and is the same reference
      here. This is what the pairing is for: a small face gets the tokens of the zoom and the reference of the form.
    - a video that runs beside the WHOLE FRAME (a whole-frame body model, a pose) still shows the whole frame: its
      figure stands where the subject stands in the frame, at the frame's size, while the plate the model is
      given is the box enlarged. Nothing cuts it to the box (the cut was removed on 2026-10-10 after one render
      that showed no difference, and is not back). The report's line says the video was not cut.
    The zoom does not make the reference larger, sharper or differently timed, and the reference does not move
    the box: the box is from `region_boxes` alone. No render had been made with the pair when this was written.

    ONE EDGE, left open: a whole-frame run that continues a stored window with NO region file is let through
    as a whole-frame continuation. A zoomed window always writes its region file except when the save itself
    failed (the song node reports "the window's region was NOT saved" and goes on); such a window, continued
    by a whole-frame run, hands over kept frames of a box and nothing says so.
    """
    choice = source.get("render_region", RENDER_WHOLE)
    stored = continued if isinstance(continued, dict) else None
    if choice == RENDER_WHOLE:
        if stored is not None:
            raise ValueError(f"this run renders `{RENDER_WHOLE}` and the window it continues was rendered zoomed in, in the "
                             f"box {stored['box']} of {stored['source_size'][0]}x{stored['source_size'][1]} frames: its kept "
                             "frames are of that box. A zoomed window is not continued: continue a whole-frame run")
        return source, []
    if choice not in RENDER_REGIONS:
        raise ValueError(f"unknown render_region {choice!r}; one of {list(RENDER_REGIONS)}")
    frames = source["frames"]
    n, old_h, old_w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    rows = source.get("region_boxes")
    if rows is None:
        raise ValueError(f"render_region `{choice}` needs the replaced part's box on each frame, and this source carries "
                         "none: it was not made by a Masked Source that tracked a subject")
    up_to = float(source.get("zoom_up_to", ZOOM_LARGEST))
    box, why = zoom_box(rows, old_w, old_h, width, height, largest=up_to)
    if box is None:
        if stored is not None:      # a whole-frame run after all, and the window it continues was zoomed
            raise ValueError(f"this load is rendered on the whole frame ({why}) and the window it continues was rendered "
                             f"zoomed in, in the box {stored['box']}: a zoomed window is not continued")
        return source, [f"render_region `{choice}` asked, and the load is rendered on the whole frame: {why}"]
    if continued != NOT_CONTINUED:
        raise ValueError(f"render_region `{choice}` would render this load in the box {list(box)}, and the run continues "
                         "a stored window: a zoomed run continues nothing, because its kept frames would be of another "
                         f"framing. Render `{RENDER_WHOLE}`, or start this load as its own run")
    cx, _cy = _fit_crop(old_w, old_h, int(width), int(height))
    scale = (old_w - 2 * cx) / (box[2] - box[0])
    zoom = {"box": [int(v) for v in box], "source_size": [old_w, old_h], "canvas": [int(width), int(height)],
            "scale": float(scale), "render_region": choice, "set_by": why,
            # the most the load asked for (`zoom_up_to`), from 2026-10-22; a record without it was made under ZOOM_LARGEST
            "up_to": up_to}
    names = [k for k in ("mask", "keep", "others") if source.get(k) is not None]
    made = {"frames": crop_frames(frames, box, width, height)}
    for k in names:
        m = source[k]
        made[k] = crop_mask(m[..., 0] if m.ndim == 4 else m, box, width, height)
    out = {**source, **made, "zoom": zoom, "whole": source}
    out["subject_area"] = area_share(made["mask"])
    out["region_boxes"] = _tracked_boxes(made["mask"])
    tracked = source.get("subject_boxes")
    if tracked is not None:
        # the tracked subject's box on the zoomed picture: never inside its own, and -1 where the box leaves it out
        k = float(width) / (box[2] - box[0])
        b = tracked.to(torch.float64)
        moved = torch.stack([torch.floor((b[:, 0] - box[0]) * k).clamp(0, int(width)),
                             torch.floor((b[:, 1] - box[1]) * k).clamp(0, int(height)),
                             torch.ceil((b[:, 2] - box[0]) * k).clamp(0, int(width)),
                             torch.ceil((b[:, 3] - box[1]) * k).clamp(0, int(height))], dim=1).to(torch.long)
        moved[(tracked[:, 0] < 0) | (moved[:, 2] <= moved[:, 0]) | (moved[:, 3] <= moved[:, 1])] = -1
        out["subject_boxes"] = moved
    lines = [render_zoom_note(zoom, source)]
    if source.get("motion_frames") is not None:
        lines.append(wired_under_zoom_note(source))
    return out, lines


def lay_back(decode: torch.Tensor, alpha: torch.Tensor, whole: torch.Tensor, zoom: dict, weight: bool = False):
    """A zoomed window's render laid into the whole frame: [F, height, width, 3].

    `decode` is the window's decode in the zoomed space and `alpha` the weight `lay_window` blended it with
    there, so `only what changed` was decided where both sides of the difference went through the same
    enlargement and on the tokens the sampler regenerated. `whole` is the window's whole-frame pixels
    (`window_frames` of the source as it came). The decode times its weight, and the weight, are scaled down
    to the box's size on the canvas (bilinear, antialiased) and placed at the box's own position, which is not
    on whole canvas pixels (`_canvas_box`); the result is `whole * (1 - weight) + decode * weight`. Weighted
    before it is scaled, so the enlarged plate is in no pixel and a soft edge is not weighted twice; where
    the scaled weight is 0 the pixel is `whole`'s own, bit for bit, inside the box as well as outside it.

    With `weight`, also the weight each canvas pixel was blended with ([F, height, width] uint8, 0 to 255; 0 outside
    the box): the matte the window's video was laid by, saved beside its region (`save_window_region`, `matte`).
    """
    if tuple(decode.shape[:3]) != tuple(alpha.shape) or int(decode.shape[0]) != int(whole.shape[0]):
        raise ValueError(f"the zoomed decode {tuple(decode.shape)}, its weight {tuple(alpha.shape)} and the whole "
                         f"frames {tuple(whole.shape)} do not cover the same frames")
    import math   # here, not at the top: the one use
    height, width = int(whole.shape[1]), int(whole.shape[2])
    fx0, fy0, fx1, fy1 = _canvas_box(zoom)
    small_w, small_h = max(1, int(round(fx1 - fx0))), max(1, int(round(fy1 - fy0)))
    ix0, iy0 = max(0, math.floor(fx0)), max(0, math.floor(fy0))
    ix1, iy1 = min(width, math.ceil(fx1)), min(height, math.ceil(fy1))
    # where each canvas pixel of that patch reads the scaled picture, as `grid_sample` counts: -1..1 over it
    xs = ((torch.arange(ix0, ix1, dtype=torch.float32) + 0.5 - fx0) / (fx1 - fx0)) * 2.0 - 1.0
    ys = ((torch.arange(iy0, iy1, dtype=torch.float32) + 0.5 - fy0) / (fy1 - fy0)) * 2.0 - 1.0
    grid = torch.stack([xs[None, :].expand(iy1 - iy0, -1), ys[:, None].expand(-1, ix1 - ix0)], dim=-1)
    exact = (small_w, small_h) == (ix1 - ix0, iy1 - iy0) and fx0 == ix0 and fy0 == iy0
    out = whole.clone()
    matte = torch.zeros(tuple(whole.shape[:3]), dtype=torch.uint8) if weight else None
    for i in range(0, int(decode.shape[0]), CHUNK):
        a = alpha[i:i + CHUNK].unsqueeze(1).to(torch.float32)
        both = torch.cat([decode[i:i + CHUNK].movedim(-1, 1).to(torch.float32) * a, a], dim=1)
        small = F.interpolate(both, size=(small_h, small_w), mode="bilinear", align_corners=False, antialias=True)
        if not exact:
            small = F.grid_sample(small, grid[None].expand(int(small.shape[0]), -1, -1, -1).to(small),
                                  mode="bilinear", padding_mode="zeros", align_corners=False)
        w = small[:, 3:4].clamp(0.0, 1.0).movedim(1, -1).to(out.device, out.dtype)
        laid = small[:, :3].movedim(1, -1).to(out.device, out.dtype)
        patch = out[i:i + CHUNK, iy0:iy1, ix0:ix1]
        out[i:i + CHUNK, iy0:iy1, ix0:ix1] = torch.where(w > 0.0, patch * (1.0 - w) + laid, patch)
        if matte is not None:
            matte[i:i + CHUNK, iy0:iy1, ix0:ix1] = (w[..., 0].cpu() * 255.0).round().to(torch.uint8)
    return (out, matte) if weight else out


def matte_of(alpha: torch.Tensor) -> torch.Tensor:
    """A composite's weight as the matte a window's region file keeps (`save_window_region`): [F, H, W] uint8, 0 to 255."""
    return (alpha.detach().to(torch.float32).clamp(0.0, 1.0).cpu() * 255.0).round().to(torch.uint8)


def window(source: dict, first_frame: int, frames: int, width: int, height: int,
           latent_t: int, lat_h: int, lat_w: int):
    """One window of a source: its fitted frames, the frames to encode, its token mask, its fitted mask, frames held.

    With a `keep` mask on the source the token mask is the grown region less
    every token the keep mask touches: the model is given those tokens clean,
    as it is the rest of the plate, and regenerates around them.

    With an `others` mask it is the grown region less every token that holds
    a pixel of the others and none of the subject's own mask: the margin does
    not grow over them, and no token of the subject is given up. Their specks
    are dropped first (`drop_specks`) and a one-cell hole they leave is closed
    after (`close_lone_holes`). Taken out before `keep`, which still wins over
    everything. With either wired, a piece of the region cut off from the
    subject is given back last (`drop_islands`).

    The frames to encode are the fitted frames themselves, or with `paint_out`
    a copy with the subject filled in: the mask widened by half of
    the margin (`source_margins`: `grow_pixels`, or under `grow_by` the
    subject's size each frame's own), which takes the soft edge SAM leaves on a fast limb and
    keeps the other half of the margin real background for the kept tokens.
    That hole is inside the regenerated tokens, so no filled pixel is shown.

    A loop's last window ends at or past the end of its track
    (`loop_plan.py`, "Lengths"), so the source can run out inside it. The
    missing frames repeat the last one with its mask held (`held_mask`): past
    the track they are not written (`loop_plan.frames_kept`).
    A window that starts past the source's end is refused; that is a source
    that does not belong to this track.
    """
    pixels, mask, short = window_frames(source, first_frame, frames, width, height)
    margin = source_margins(source, first_frame, frames, int(width) * int(height))
    whole = source.get("edge", EDGE_TOKENS) == EDGE_TOKENS
    tokens = token_mask(grow(mask, margin), latent_t, lat_h, lat_w, whole)
    if source.get("others") is not None:
        away = fit_mask(source["others"][int(first_frame):int(first_frame) + int(frames)], width, height)
        if short > 0:
            away = held_mask(away, short)
        # a speck of the others on a frame or two of a step does not hole the whole step (`drop_specks`)
        away = drop_specks(away, latent_t, lat_h, lat_w)
        # In whole tokens, as the region is: a token the others touch is the source's unless the subject's own
        # mask, before any margin, has a pixel in it on one of its frames.
        theirs = token_mask(away, latent_t, lat_h, lat_w, whole)
        own = token_mask(mask, latent_t, lat_h, lat_w, whole)
        grown = tokens
        tokens = tokens * (1.0 - theirs * (1.0 - own))
        # a one-cell hole the others punched with the region all round it is given back (`close_lone_holes`)
        tokens = close_lone_holes(tokens, grown, own, whole, away)
    if source.get("keep") is not None:
        # After the grow, so the margin cannot run back over what is kept, and in whole tokens: a token
        # that holds any kept pixel on any of its frames is kept. The composite and the review read
        # `tokens`, so the source's own pixels are what is shown there.
        held = fit_mask(source["keep"][int(first_frame):int(first_frame) + int(frames)], width, height)
        if short > 0:
            held = held_mask(held, short)
        tokens = tokens * (1.0 - token_mask(held, latent_t, lat_h, lat_w, whole))
    if source.get("others") is not None or source.get("keep") is not None:
        # a piece of the region the keep-outs cut off from the subject is given back to the source (`drop_islands`)
        tokens = drop_islands(tokens, token_mask(mask, latent_t, lat_h, lat_w, whole))
    encode = pixels
    if source.get("paint_out"):
        encode = fill_subject(pixels, grow(mask, margin // 2))
    elif source.get("start_from", START_NOISE) != START_NOISE:
        # a late start shows what is encoded here; `start_zero_tokens` then empties the body's tokens
        encode = soften_subject(pixels, grow(mask, margin // 2), int(source["start_blur"]))
    return pixels, encode, tokens, mask, short


def detect_part(segmenter, segmenter_clip, frames: torch.Tensor, where: torch.Tensor,
                phrases: tuple[str, ...], threshold: float = PART_THRESHOLD) -> torch.Tensor:
    """The union of core's SAM 3 detections of `phrases` on frames `where`, one mask each, [len(where), H, W]."""
    from comfy_extras.nodes_sam3 import SAM3_Detect  # core's node; imported here so a check needs no SAM
    parts = []
    for i in range(0, where.shape[0], CHUNK):
        chunk = frames[where[i:i + CHUNK]]
        union = torch.zeros(tuple(chunk.shape[:3]), dtype=torch.float32)
        for phrase in phrases:
            cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(phrase))
            out = SAM3_Detect.execute(segmenter, chunk, conditioning=cond, threshold=float(threshold))
            union = torch.maximum(union, getattr(out, "args", out)[0].to(union))
        parts.append(union)
    return torch.cat(parts, dim=0)


#: The widest margin and blend the node's widgets take. Reasoned, as they stood in the schema: the schema and
#: the queue-time validation read them from here so the two cannot drift.
GROW_PIXELS_MAX = 512
FEATHER_PIXELS_MAX = 128


def settings_refusal(grow_pixels, feather_pixels, paint_out=False, start_from=START_NOISE) -> str | None:
    """Why these settings cannot run, or None: the refusals that need only the node's own widgets.

    One function for both moments. The node's queue-time validation calls it, so a graph with such a setting is
    refused when it is submitted; `execute` calls it too, for a value that arrives through a link and is not
    known until then. Until 2026-10-10 these were raised only when the node ran, which on a busy queue is after
    everything ahead of it (a preview with a feather wider than its margin waited a quarter of an hour to fail).
    A refusal that needs a tensor (`keep`, `others`, the frames' shape) cannot be made here and stays in `execute`.
    """
    if int(feather_pixels) > int(grow_pixels):
        return (f"feather_pixels {int(feather_pixels)} is wider than grow_pixels {int(grow_pixels)}: the blend "
                "would reach the subject's own pixels and bring the original back at its edge")
    if start_from != START_NOISE and paint_out:
        return ("paint_out fills the subject in before the encode, so a softened start has nothing to "
                "soften: turn one of the two off")
    return None


class MiniMaxH3MaskedSource(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedSource",
            display_name="MiniMax H3 Masked Source (video to video)",
            category="model/latent/minimax",
            description=(
                "A source video and a mask over the subject to replace. Wire it into the song "
                "node's `source`: each window then starts from the source's own frames, "
                "regenerates only the masked subject, and puts the source's pixels back "
                "everywhere else. The mask is one frame per source frame, 1 on the subject, "
                "as SAM3 Track to Mask gives it."),
            inputs=[
                io.Image.Input("frames", tooltip="The source video's frames at 24 fps, from its start."),
                # lazy: asked for by `check_lazy_status`, with whichever other inputs `replace` needs
                io.Mask.Input("mask", lazy=True, tooltip="One mask per frame, 1 on the subject to replace."),
                # appended 2026-10-04
                io.Combo.Input("replace", options=[REPLACE_WHOLE, REPLACE_PART, REPLACE_PARTS], default=REPLACE_WHOLE,
                               tooltip=("What is regenerated; the first of the three choices a user makes here. `whole subject` replaces the person, clothes "
                                        "and movement included, so the prompt has to say what they do. "
                                        "`head and hair` replaces the subject from the top of the head down "
                                        "to where the hair ends and keeps the rest of the body, its clothes "
                                        "and its movement; it needs `segmenter` and `segmenter_clip`, and "
                                        "costs a detector pass per phrase on each frame the subject is in."
                                        " `the wired parts` regenerates exactly the mask on `parts`, from the part node.")),
                io.Mask.Input(LAZY_FOR_PARTS, optional=True, lazy=True,
                              tooltip=("The part node's `parts` output (MiniMaxH3SubjectParts): one mask per frame, 1 on "
                                       "the chosen parts of the subject. Read only for `the wired parts`, which "
                                       "regenerates exactly that region; SAM is not asked for a phrase.")),
                # appended 2026-10-05, optional so a saved graph keeps running. The
                # defaults are the shipped render as it was; `subject only` with the
                # VAE copy off is the arm the masking board calls route 1. Read by
                # the song node, which builds the reference per window.
                io.Combo.Input("motion_reference", options=list(MOTIONS),
                               default=MOTION_NONE, optional=True,
                               tooltip=("Also show the model the original's movement, as a video reference the "
                                        "prompt names as <Video 1>.\n\n"
                                        "none (default): the model sees the still and the prompt only.\n\n"
                                        "subject only: the source window with everything outside the subject "
                                        "grey, so the model sees how the subject moves and nothing of the "
                                        "scene.\n\n"
                                        "subject only, zoomed in: the same, shown in a box around the subject "
                                        "(around the replaced part, on a pass that replaces a part) "
                                        "and not in the whole frame, so a subject that is small in the frame "
                                        "is seen larger for the same cost or less. The box holds still through "
                                        "a shot and changes only where the video cuts; when a window holds "
                                        "shots framed differently, each sits in the middle of one shared "
                                        "picture on grey. Use it when the model does not follow a small "
                                        "subject's movement.\n\n"
                                        "whole frame: the source window as it is.\n\n"
                                        "a video I wire: the video on `motion_video`, which runs beside the "
                                        "source frame for frame: a body mesh, a pose, a map of a mouth. It says "
                                        "how the subject moves without showing the original, and each window "
                                        "is shown its own part of it.\n\n"
                                        "The prompt has to say what the video provides, for example that the "
                                        "subject's motion and timing come from <Video 1>. Costs text-encoder "
                                        "tokens on every window, and with motion_vae on, rows on every "
                                        "sampling step.")),
                io.Int.Input("motion_short_edge", default=MOTION_SHORT_EDGE, min=32, max=1024, step=32, optional=True,
                             tooltip=("Shorter side, in pixels, of the motion reference the model is shown, "
                                      "rounded to 32. Smaller is cheaper; raise it if the model cannot make "
                                      "out the subject. Zoomed in, it is the most the picture may cost: the "
                                      "zoomed picture never holds more pixels than the whole frame at this "
                                      "short edge would.")),
                io.Boolean.Input("motion_vae", default=False, optional=True,
                                 tooltip=("Off (default): the motion reference reaches the model through the "
                                          "text encoder only, at two frames per second. Cheap.\n\n"
                                          "On: the video model also gets its own copy, which costs rows on "
                                          "every sampling step and may not fit the card at a long window. "
                                          "Read the song node's report before queueing.")),
                io.Int.Input("grow_pixels", default=GROW_PIXELS, min=0, max=GROW_PIXELS_MAX,
                             tooltip=("How far the mask is widened before it reaches the model, in pixels of "
                                      "the render canvas. Raise it when the replacement is cut off at its "
                                      "edge. The hole's shape is all that tells the model where the original "
                                      "stood, so a wider one lets the new subject stand somewhere else and "
                                      "uncover what the original hid.")),
                io.Int.Input("feather_pixels", default=8, min=0, max=FEATHER_PIXELS_MAX,
                             tooltip=("Width of the blend between the regenerated region and the source's "
                                      "own pixels, to each side of the boundary. Raise it if the boundary "
                                      "shows; it cannot exceed grow_pixels.")),
                io.Boolean.Input("paint_out", default=False,
                                 tooltip=("Fill the subject in with its surroundings before the source is "
                                          "encoded, so the kept picture around the mask carries no trace of "
                                          "them. Turn it on if a faint remnant of the original shows beside "
                                          "the new subject. Costs a fill per frame, no sampling time.")),
                io.Model.Input("segmenter", optional=True, lazy=True,
                               tooltip="The SAM 3 model that tracked the mask. Read only for `head and hair`."),
                io.Clip.Input("segmenter_clip", optional=True, lazy=True,
                              tooltip="The SAM 3 checkpoint's text encoder. Read only for `head and hair`."),
                io.String.Input("part_phrases", default=PART_PHRASES,
                                tooltip=("What SAM 3 is asked to find on the subject for `head and hair`, "
                                         "separated by commas; everything found is used together. Change "
                                         "it when the `mask` output shows the wrong region. Each phrase "
                                         "costs one more detector pass.")),
                io.Float.Input("part_threshold", default=PART_THRESHOLD, min=0.0, max=1.0, step=0.01,
                               tooltip=("How sure SAM 3 has to be before a detection of a phrase counts, for "
                                        "`head and hair`. Lower it when the log says many frames were "
                                        "carried from a neighbour; raise it when the `mask` output takes "
                                        "something that is not the part.")),
                io.Int.Input("part_margin", default=PART_MARGIN, min=0, max=256,
                             tooltip=("How far past the subject's own mask a detected part may reach and "
                                      "still count as the subject's, in pixels of the source frame. Raise it "
                                      "when the `mask` output clips the part at the subject's edge; lower it "
                                      "when it takes a neighbour's.")),
                io.Combo.Input("composite", options=[COMPOSITE_REGION, COMPOSITE_CHANGED], default=COMPOSITE_CHANGED,
                               tooltip=("What is kept from the render. `whole region` keeps everything that was "
                                        "regenerated, margin included. `only what changed` keeps the render "
                                        "where it differs from the source (the new subject, and where the old "
                                        "one stood) and puts the source back where the model only repainted "
                                        "the background: choose it when the area around the subject flickers "
                                        "or looks cut out. Costs a comparison per frame, no model.")),
                io.Float.Input("change_threshold", default=CHANGE_THRESHOLD, min=0.0, max=1.0, step=0.005,
                               tooltip=("For `only what changed`: how different the render must be from the "
                                        "source to be kept, as a fraction of full brightness. Lower it if parts "
                                        "of the new subject go missing; raise it if flicker around the subject "
                                        "remains.")),
                # appended 2026-10-05: the per-token late start (`START_TOP`)
                io.Combo.Input("start_from", options=[START_NOISE, START_TOP], default=START_NOISE, optional=True,
                               tooltip=("What the new subject starts from. `noise` (default): nothing of the "
                                        "original is under the mask.\n\n"
                                        "`the original's top, softened`: sampling starts late, the top of the "
                                        "subject from a grey blur of the original and the rest from nothing, "
                                        "so the model can see which way the original faces without seeing "
                                        "its clothes. Try it when the new subject should turn with the "
                                        "original. Costs a blur per frame, and saves the steps skipped.")),
                io.Float.Input("start_top", default=START_TOP_SHARE, min=0.05, max=1.0, step=0.05, optional=True,
                               tooltip=("For a softened start: the share of the subject's height, from the top, "
                                        "that keeps the blurred original. Lower it if the original's hair comes "
                                        "back; raise it if the subject does not follow the original.")),
                io.Int.Input("start_blur", default=START_BLUR, min=1, max=128, optional=True,
                             tooltip=("For a softened start: the blur, in pixels. Raise it if the original's "
                                      "look comes through; lower it if the subject does not follow.")),
                io.Int.Input("start_knots", default=START_KNOTS, min=1, max=4, optional=True,
                             tooltip=("For a softened start: how many steps of the schedule are skipped. More "
                                      "shows more of the blur in the result.")),
                # appended 2026-10-05 (`shot_table.py`): lazy with the mask
                io.String.Input(LAZY_FOR_TABLE, optional=True, lazy=True, force_input=True,
                                tooltip=("The Subject Track's `shot_table` output: who was found and taken in "
                                         "each shot. The song node writes it next to the video.")),
                # appended 2026-10-07 (the owner: "keep the prop rather than describe it"). A plain mask
                # from any node; unwired, nothing changes.
                io.Mask.Input("keep", optional=True,
                              tooltip=("Optional. One mask per source frame of what must stay the original even "
                                       "inside the region: something the subject holds, a person standing close, "
                                       "anything passing in front. It wins over the region after grow_pixels, so "
                                       "the margin cannot grow back over it, and the source's own pixels are "
                                       "shown there. The model sees what is kept and draws around it.\n\n"
                                       "Kept in whole tokens: a block of the picture that holds any kept pixel "
                                       "is kept, for the few frames that block spans, so a little of what "
                                       "surrounds a small object stays too.\n\n"
                                       "Example: a second Subject Track with the phrase `cigarette`, its mask "
                                       "wired here.")),
                # appended 2026-10-08: a margin fixed in pixels is several times a small subject's own area
                io.Combo.Input("grow_by", options=list(GROW_BY), default=GROW_FIXED, optional=True,
                               tooltip=("How wide the margin round the mask is.\n\n"
                                        "a fixed margin: grow_pixels on every frame.\n\n"
                                        "the subject's size: the margin shrinks with the subject, frame by "
                                        "frame, and is never more than grow_pixels. Use it when the subject is "
                                        "small in the frame and the region takes in the people round them: the "
                                        "model may then draw the new subject on one of those people. The motion "
                                        "reference is not changed by this. The song node's report gives the "
                                        "margin each window used.")),
                # appended 2026-10-08: with two people inside the region the model drew the new subject on the
                # other one. A plain mask from any node; unwired, nothing changes.
                io.Mask.Input("others", optional=True,
                              tooltip=("Optional. People who are not the subject: the margin does not grow over "
                                       "them. One mask per source frame, at the frames' own size, of everyone "
                                       "the region should leave alone; a mask of another length or size is "
                                       "refused.\n\n"
                                       "A block of the picture that holds any of them and none of the subject "
                                       "stays the original. A block that holds any of what is replaced is "
                                       "still regenerated, so none of it is given up.\n\n"
                                       "Leave the subject out of this mask. Where only a part of them is "
                                       "replaced, a mask that covers the rest of their body takes the margin "
                                       "off it.\n\n"
                                       "Use `keep` for a thing the subject holds; use this for the people "
                                       "standing round them. The preview shows them in blue.")),
                # appended 2026-10-09: a trial of a finer edge, the provenance beside `EDGES`
                io.Combo.Input("edge", options=list(EDGES), default=EDGE_TOKENS, optional=True,
                               tooltip=("How fine the edge of the replaced region is.\n\n"
                                        "whole tokens: the region is rounded out to the model's blocks of 32 "
                                        "pixels. A block is regenerated or kept whole.\n\n"
                                        "latent cells: the region is rounded out to 16 pixels. Inside a "
                                        "block that is regenerated, the cells outside the region come back as "
                                        "the original. Use it when the subject is small and the region takes "
                                        "in the people round them. Untested on playback.")),
                # appended 2026-10-10 (the owner: a signal for the movement, not words for it)
                io.Image.Input("motion_video", optional=True,
                               tooltip=("Optional, for motion_reference `a video I wire`. One frame per source "
                                        "frame, any size: it is fitted to the render's canvas as the source "
                                        "is. What the model is shown as the movement, in place of anything cut "
                                        "from the source: a body mesh of the original, a pose, a map of where "
                                        "a mouth opens. The prompt has to say what <Video 1> is and what is "
                                        "taken from it. With motion_vae on, the video model gets its own copy "
                                        "at every frame; off, the text encoder sees two frames a second.")),
                # appended 2026-10-10: a small subject rendered zoomed in (`RENDER_REGIONS`, `zoomed_source`)
                io.Combo.Input("render_region", options=list(RENDER_REGIONS), default=RENDER_WHOLE, optional=True,
                               tooltip=("Where the replaced region is drawn. Use the zoomed choice when the face "
                                        "or part being replaced is small in the frame and comes out looking "
                                        "like nobody in particular.\n\n"
                                        "the whole frame (default): the picture is rendered as it is.\n\n"
                                        "zoomed in on what is replaced: the load is cropped to one box round "
                                        "the replaced part (on a face pass, the head), rendered at that size, "
                                        "and the result is put back into the whole frame, at most zoom_up_to times. "
                                        "The report says the "
                                        "zoom it came to; a part that crosses the frame is zoomed less, and one "
                                        "that fills it is rendered on the whole frame.\n\n"
                                        "One box for the whole load, so load one shot at a time: a load with "
                                        "a cut in it is refused, and so is a run that continues another. "
                                        "A wired motion video is shown to the model as it is, not cut to the "
                                        "box: wire one framed on the subject (a body model cut to the head), "
                                        "not one of the whole frame. Costs no sampling time. The zoomed "
                                        "picture is also written beside each window. grow_pixels and "
                                        "feather_pixels are then pixels of the zoomed picture.")),
                # appended 2026-10-12 (the owner: more than one reference video, each with a subject of its own)
                io.Image.Input("motion_video_2", optional=True,
                               tooltip=("Optional: a second video beside `motion_video`, for motion_reference "
                                        "`a video I wire`. One frame per source frame, any size. The model is "
                                        "shown it as the next video after the first: <Video 2> when no other "
                                        "video reference is wired into the song node. Use it when the prompt "
                                        "gives each video something of its own to carry, a body for the "
                                        "movement and a closer view of a face, say; the prompt has to say what "
                                        "each one is. It costs as the first does, at its own short edge, and "
                                        "motion_vae switches the video model's copy of both. Needs "
                                        "`motion_video` wired too.")),
                io.Int.Input("motion_short_edge_2", default=MOTION_SHORT_EDGE, min=32, max=1024, step=32, optional=True,
                             tooltip=("Shorter side, in pixels, of the second motion video as the model is "
                                      "shown it. Does nothing unless `motion_video_2` is wired.")),
                # appended 2026-10-22 (the owner: one system for more than one scenario; `CONTEXT_SHOWS` has the rule)
                io.Combo.Input("context_shows", options=list(CONTEXT_SHOWS), default=CONTEXT_AS_FILMED, optional=True,
                               tooltip=("What the window's extra frames show, when the frames are longer than the mask "
                                        "(a shot padded with the source's real frames before and after it, which are "
                                        "dropped after the decode). Does nothing when the mask covers every frame.\n\n"
                                        "as filmed (default): nothing is regenerated on them; the original person is "
                                        "seen there as the camera saw them.\n\n"
                                        "face covered: the region is held open on them, the shot's first frame's mask "
                                        "on the frames before and its last frame's on the frames after, grown by the "
                                        "margin, so the original's face or beard is not left in view for the model to "
                                        "copy. Where those frames are another shot, the held mask may not be where the "
                                        "person is; the report names those frames.\n\n"
                                        "A job's graph marks the extra frames that lie across a cut, in another shot: "
                                        "those are copies of the shot's own first or last frame with its region open, "
                                        "whichever is chosen here.")),
                # appended 2026-10-22 (the owner: the zoom to a target head size lands switched off, one load asks for it to test)
                io.Float.Input("zoom_up_to", default=ZOOM_LARGEST, min=ZOOM_SMALLEST, max=ZOOM_MOST_ASKED, step=0.25,
                               optional=True,
                               tooltip=(f"The most a zoomed load is enlarged. {ZOOM_LARGEST:g} (default): the box round the "
                                        f"part sets the zoom, at most {ZOOM_LARGEST:g} times, as every zoomed load has been. "
                                        f"Above {ZOOM_LARGEST:g}: a part too small to be about {ZOOM_HEAD_TARGET} px tall "
                                        f"on the canvas at {ZOOM_LARGEST:g} times is enlarged until it is, up to this, "
                                        "where the box allows; the picture round it is the frame enlarged that much, so "
                                        "it is soft. Does nothing unless render_region is zoomed in.")),
            ],
            outputs=[H3MaskedSource.Output(display_name="source"),
                     io.Mask.Output(display_name="mask",
                                    tooltip=("The mask this node used, one per source frame, before grow_pixels: "
                                             "preview it to see what will be replaced.")),
                     io.Image.Output(display_name="preview",
                                     tooltip=("A strip of sampled frames: the plate with the regenerated region "
                                              "tinted red, and beside it what the text encoder is shown as the "
                                              "motion reference when one is on. Look before rendering."))],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, frames=None, replace=REPLACE_WHOLE, **kwargs):
        # None is a connected input core has not run yet; an unconnected
        # optional input is absent. `frames` is not lazy, so it is here.
        wanted = {REPLACE_PART: LAZY_FOR_MASK, REPLACE_PARTS: (LAZY_FOR_MASK[0], LAZY_FOR_PARTS)}.get(replace, LAZY_FOR_MASK[:1]) + (LAZY_FOR_TABLE,)
        return [name for name in wanted if name in kwargs and kwargs[name] is None]

    @classmethod
    def validate_inputs(cls, grow_pixels, feather_pixels, paint_out=False, start_from=START_NOISE) -> bool | str:
        """Refuse at queue time what `settings_refusal` would refuse at run time.

        Only these four are named: core skips its own range and choice tests for an input a validation
        function names (`execution.py::validate_inputs`), so those tests are made again here and every other
        input keeps core's. A value that comes through a link is None at this point and is left to `execute`.
        """
        if grow_pixels is None or feather_pixels is None:
            return True
        if not 0 <= int(grow_pixels) <= GROW_PIXELS_MAX:
            return f"grow_pixels {int(grow_pixels)} is outside 0 to {GROW_PIXELS_MAX}"
        if not 0 <= int(feather_pixels) <= FEATHER_PIXELS_MAX:
            return f"feather_pixels {int(feather_pixels)} is outside 0 to {FEATHER_PIXELS_MAX}"
        if start_from is not None and start_from not in (START_NOISE, START_TOP):
            return f"unknown start_from {start_from!r}; one of {[START_NOISE, START_TOP]}"
        return settings_refusal(grow_pixels, feather_pixels, bool(paint_out),
                                START_NOISE if start_from is None else start_from) or True

    @classmethod
    # `mask` has no default: it is required in the schema. Core hands a lazy input it was not asked to run as
    # None, which `check_lazy_status` does not allow for the mask, and `execute` refuses if it happens.
    def execute(cls, frames, mask, grow_pixels=GROW_PIXELS, feather_pixels=8, replace=REPLACE_WHOLE, paint_out=False,
                segmenter=None, segmenter_clip=None, part_phrases=PART_PHRASES,
                part_threshold=PART_THRESHOLD, part_margin=PART_MARGIN, composite=COMPOSITE_CHANGED,
                change_threshold=CHANGE_THRESHOLD, motion_reference=MOTION_NONE,
                motion_short_edge=MOTION_SHORT_EDGE, motion_vae=False, start_from=START_NOISE,
                start_top=START_TOP_SHARE, start_blur=START_BLUR, start_knots=START_KNOTS,
                shot_table=None, parts=None, keep=None, grow_by=GROW_FIXED, others=None,
                edge=EDGE_TOKENS, motion_video=None,
                render_region=RENDER_WHOLE, motion_video_2=None,
                motion_short_edge_2=MOTION_SHORT_EDGE, context_shows=CONTEXT_AS_FILMED,
                zoom_up_to=ZOOM_LARGEST) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        if grow_by not in GROW_BY:
            raise ValueError(f"unknown grow_by {grow_by!r}; one of {list(GROW_BY)}")
        if edge not in EDGES:
            raise ValueError(f"unknown edge {edge!r}; one of {list(EDGES)}")
        if replace not in (REPLACE_WHOLE, REPLACE_PART, REPLACE_PARTS):
            raise ValueError(f"unknown replace {replace!r}; one of {[REPLACE_WHOLE, REPLACE_PART, REPLACE_PARTS]}")
        if start_from not in (START_NOISE, START_TOP):
            raise ValueError(f"unknown start_from {start_from!r}; one of {[START_NOISE, START_TOP]}")
        refusal = settings_refusal(grow_pixels, feather_pixels, paint_out, start_from)
        if refusal:
            raise ValueError(refusal)
        if composite not in (COMPOSITE_REGION, COMPOSITE_CHANGED):
            raise ValueError(f"unknown composite {composite!r}; one of {[COMPOSITE_REGION, COMPOSITE_CHANGED]}")
        if motion_reference not in MOTIONS:
            raise ValueError(f"unknown motion_reference {motion_reference!r}; one of {list(MOTIONS)}")
        if render_region not in RENDER_REGIONS:
            raise ValueError(f"unknown render_region {render_region!r}; one of {list(RENDER_REGIONS)}")
        if not ZOOM_SMALLEST <= float(zoom_up_to) <= ZOOM_MOST_ASKED:
            raise ValueError(f"zoom_up_to {float(zoom_up_to):g} is outside {ZOOM_SMALLEST:g} to {ZOOM_MOST_ASKED:g}")
        if context_shows not in CONTEXT_SHOWS:
            raise ValueError(f"unknown context_shows {context_shows!r}; it is `{CONTEXT_AS_FILMED}` or `{CONTEXT_FACE_COVERED}`")
        if motion_reference == MOTION_WIRED and motion_video is None:
            raise ValueError(f"motion_reference is `{motion_reference}` and nothing is wired to `motion_video`: wire the "
                             "video that shows the movement, or choose another motion_reference")
        n_frames = int(frames.shape[0])
        n_shot = (int(mask.shape[0]) if mask is not None and getattr(mask, "ndim", 0) >= 3
                  and 0 < int(mask.shape[0]) < n_frames and on_both_clocks(n_frames) else n_frames)
        hidden = getattr(cls, "hidden", None)
        prompt = getattr(hidden, "prompt", None) if hidden is not None else None
        uid = getattr(hidden, "unique_id", None) if hidden is not None else None
        meta = (prompt.get(str(uid), {}).get("_meta") if isinstance(prompt, dict) and uid is not None else None) or {}
        if n_shot < n_frames:
            extra = n_frames - n_shot
            if "pad_before" in meta or "pad_after" in meta:
                pad_before, pad_after = int(meta.get("pad_before", 0)), int(meta.get("pad_after", 0))
                if pad_before < 0 or pad_after < 0 or pad_before + pad_after != extra:
                    raise ValueError(f"window pad ({pad_before}, {pad_after}) does not add up to the {extra} context "
                                     f"frame(s) around a {n_shot}-frame shot in a {n_frames}-frame window")
            else:
                # a graph that does not say: every extra frame after the shot, as the job builder pads
                # (`workflows/masked_job_graph.py::window_pad`, the owner's word of 2026-10-22; split evenly until then)
                pad_before, pad_after = 0, extra
            # of those, the ones across a cut from the shot, held (`CONTEXT ACROSS A CUT IS HELD`); none in a graph from before
            held_before, held_after = int(meta.get("held_before", 0)), int(meta.get("held_after", 0))
            if not (0 <= held_before <= pad_before and 0 <= held_after <= pad_after):
                raise ValueError(f"held context ({held_before}, {held_after}) is more than the window's context frames "
                                 f"({pad_before}, {pad_after}), or under none")
        else:
            pad_before = pad_after = held_before = held_after = 0
        if motion_video is not None:
            if motion_reference != MOTION_WIRED:
                raise ValueError(f"`motion_video` is wired and motion_reference is `{motion_reference}`: set it to "
                                 f"`{MOTION_WIRED}` to use the video, or unwire it. It is never used in silence")
            if motion_video.ndim != 4 or int(motion_video.shape[0]) != n_shot:
                raise ValueError(
                    f"`motion_video` is {tuple(motion_video.shape)} and the source has {n_shot} "
                    "frames: it needs one frame per source frame, made from the same `frames`, so that a window "
                    "is shown the movement of its own frames")
        if motion_video_2 is not None:
            if motion_video is None:
                raise ValueError("`motion_video_2` is wired and `motion_video` is not: a second video comes after a "
                                 "first. Wire the one video to `motion_video`")
            if motion_video_2.ndim != 4 or int(motion_video_2.shape[0]) != n_shot:
                raise ValueError(
                    f"`motion_video_2` is {tuple(motion_video_2.shape)} and the source has {n_shot} "
                    "frames: it needs one frame per source frame, as `motion_video` does")
        if keep is not None:
            if keep.ndim == 4 and int(keep.shape[-1]) == 1:
                keep = keep[..., 0]
            if tuple(keep.shape) != (n_shot, *frames.shape[1:3]):
                raise ValueError(
                    f"`keep` is {tuple(keep.shape)} and the frames {(n_shot, *frames.shape[1:3])}: it needs one mask "
                    "per source frame at the frames' own size, from a node that ran on the same `frames`")
            if paint_out or start_from != START_NOISE:
                # both show a changed copy of the source under the subject, which a kept token would put on screen
                raise ValueError(
                    "`keep` shows the source's own pixels inside the region, and paint_out and a softened start "
                    "both change those pixels before the encode: turn them off, or unwire `keep`")
            keep = (keep > 0.5).to(torch.float32)
        if others is not None:
            if others.ndim == 4 and int(others.shape[-1]) == 1:
                others = others[..., 0]
            if others.ndim != 3 or int(others.shape[0]) != n_shot:
                raise ValueError(
                    f"`others` has {int(others.shape[0])} masks and the source {n_shot} frames: it "
                    "needs one mask per source frame, made on the same `frames` this node takes. Nothing is "
                    "stretched or cut to fit, since a mask on the wrong frame keeps the wrong people out")
            if tuple(others.shape[1:]) != tuple(frames.shape[1:3]):
                raise ValueError(
                    f"`others` is {int(others.shape[2])}x{int(others.shape[1])} and the frames are "
                    f"{int(frames.shape[2])}x{int(frames.shape[1])}: it needs the frames' own size")
            if paint_out or start_from != START_NOISE:
                # as for `keep`: a token left to the source would show the changed copy of it
                raise ValueError(
                    "`others` leaves the source's own pixels where the margin would have grown, and paint_out "
                    "and a softened start both change those pixels before the encode: turn them off, or unwire "
                    "`others`")
            others = (others > 0.5).to(torch.float32)
        table = str(shot_table or "")
        if mask is None:
            raise ValueError("the Masked Source was run with no mask: wire one (a tracker's, or a mask video loaded "
                             "from a file)")
        tracked = mask          # the tracker's own mask, before it is cut to a part below
        mask, note, part_warning = cls._settle_mask(frames[pad_before:pad_before + n_shot], mask, replace, segmenter,
                                                    segmenter_clip, part_phrases, part_threshold, part_margin, parts)
        cuts = source_cuts(frames, table)
        context = {}            # what the context frames took, for the record and each window's region file
        shot_before, shot_after, fell_back = (
            held_edge_source(mask, held_before, held_after) if (held_before or held_after) else (None, None, [])
        )
        if fell_back:
            if shot_before == -1:
                held_before = 0
                shot_before = None
            if shot_after == -1:
                held_after = 0
                shot_after = None
            for line in fell_back:
                logger.info("[h3] MiniMaxH3MaskedSource: held context falls back: %s", line)
        if n_shot < n_frames:
            tracked_2d = tracked[..., 0] if tracked.ndim == 4 else tracked
            # the subject's mask, and the tracked one its boxes come from, by `context_shows`; the keep-outs empty in
            # both forms (`CONTEXT_SHOWS` has why)
            tracked = pad_context(tracked_2d, pad_before, pad_after, context_shows)
            mask = pad_context(mask, pad_before, pad_after, context_shows)
            context = {"context_shows": context_shows, **({"held_fell_back": fell_back} if fell_back else {})}
            if context_shows == CONTEXT_FACE_COVERED:
                across = context_across_cut(cuts, pad_before, n_shot, pad_after)
                if across:
                    context["context_held_across_a_cut"] = across
                # an edge frame with no mask holds nothing: those context frames are as filmed after all, and the record says so
                empty = ([] if not pad_before or bool((mask[pad_before] > 0.5).any()) else ["before"]) \
                    + ([] if not pad_after or bool((mask[pad_before + n_shot - 1] > 0.5).any()) else ["after"])
                if empty:
                    context["context_held_empty"] = empty
                    logger.info("[h3] MiniMaxH3MaskedSource: context `%s`: the shot's %s frame has no mask, so nothing is held over "
                                "the frames %s it: they show the source as filmed", context_shows,
                                " and ".join({"before": "first", "after": "last"}[e] for e in empty), " and ".join(empty))
                logger.info("[h3] MiniMaxH3MaskedSource: context `%s`: the shot's first frame's mask (%d px) held over the %d "
                            "frame(s) before it, its last frame's (%d px) over the %d after%s", context_shows,
                            int((mask[pad_before] > 0.5).sum()), pad_before, int((mask[pad_before + n_shot - 1] > 0.5).sum()),
                            pad_after, (f"; {len(across)} of them lie across a cut from the shot, where the subject may stand "
                                        f"elsewhere or nowhere: window frame(s) {runs_said(across)}" if across else ""))
            if keep is not None:
                keep = F.pad(keep, (0, 0, 0, 0, pad_before, pad_after))
            if others is not None:
                others = F.pad(others, (0, 0, 0, 0, pad_before, pad_after))
            if held_before or held_after:
                # across a cut, the shot's edge frame (or nearest frame >= HELD_EDGE_LEAST * middle) held with its masks in place of the other shot's frames
                src_b = (pad_before + shot_before) if shot_before is not None else None
                src_a = (pad_before + shot_after) if shot_after is not None else None
                frames, tracked, mask, keep, others = (
                    hold_context(x, held_before, held_after, src_before=src_b, src_after=src_a)
                    for x in (frames, tracked, mask, keep, others)
                )
                context.update({k: v for k, v in (("held_before", held_before), ("held_after", held_after)) if v})
                logger.info("[h3] MiniMaxH3MaskedSource: %d context frame(s) before the shot and %d after it lie across a "
                            "cut: they are held copies of the shot's edge frame with its masks, not the other shot's "
                            "frames; %d before and %d after are the source's own", held_before, held_after,
                            pad_before - held_before, pad_after - held_after)
            if motion_video is not None:
                motion_video = torch.cat(
                    ([motion_video[:1].expand(pad_before, -1, -1, -1)] if pad_before else [])
                    + [motion_video]
                    + ([motion_video[-1:].expand(pad_after, -1, -1, -1)] if pad_after else []), dim=0)
            if motion_video_2 is not None:
                motion_video_2 = torch.cat(
                    ([motion_video_2[:1].expand(pad_before, -1, -1, -1)] if pad_before else [])
                    + [motion_video_2]
                    + ([motion_video_2[-1:].expand(pad_after, -1, -1, -1)] if pad_after else []), dim=0)
        elif context_shows != CONTEXT_AS_FILMED:
            logger.info("[h3] MiniMaxH3MaskedSource: context `%s` and the mask covers all %d frames: no context frames, "
                        "nothing held", context_shows, n_frames)
        # The whole subject's box per frame, from the tracker's mask and not from the part it was just
        # cut to: on a parts graph nothing after this node knows where the subject is otherwise.
        boxes = _tracked_boxes(tracked[..., 0] if tracked.ndim == 4 else tracked)
        covered = float((mask > 0.5).any(dim=0).float().mean())
        # The mask's area on each frame as a share of the frame: what `grow_by` takes the margin from, here on
        # the source's own frames and in every window on the canvas (`source_margins`).
        area = area_share(mask)
        each = margins(area, int(frames.shape[1]) * int(frames.shape[2]), grow_pixels, feather_pixels, grow_by)
        logger.info("[h3] MiniMaxH3MaskedSource: %d frames, replace `%s`%s, the mask touches %.1f%% of the "
                    "frame over the clip, grow %s (%s), feather %d px%s, composite keeps the %s", int(frames.shape[0]),
                    replace, note, 100.0 * covered, margin_note(each), grow_by, int(feather_pixels),
                    ", subject painted out before the encode" if paint_out else "", composite)
        if keep is not None:
            both = (grow(mask.to(torch.float32), each) > 0.5) & (keep > 0.5)
            logger.info("[h3] MiniMaxH3MaskedSource: keep is wired: it lies inside the grown region on %d of %d "
                        "frames, and those tokens stay the source's", int(both.flatten(1).any(dim=1).sum()),
                        int(frames.shape[0]))
        if others is not None:
            reach = (grow(mask.to(torch.float32), each) > 0.5) & (others > 0.5) & ~(mask > 0.5)
            logger.info("[h3] MiniMaxH3MaskedSource: others is wired: the margin would have reached them on %d of %d "
                        "frames; those tokens stay the source's unless the subject's own mask is in them",
                        int(reach.flatten(1).any(dim=1).sum()), int(frames.shape[0]))
        if motion_reference != MOTION_NONE:
            logger.info("[h3] MiniMaxH3MaskedSource: motion reference %s at a %d short edge, %s%s", motion_reference,
                        int(motion_short_edge), "with the video model's copy" if motion_vae else "text encoder only",
                        "" if motion_video_2 is None else f"; a second wired video at a {int(motion_short_edge_2)} short edge")
        # The mask's own box per frame: on a parts graph the part's, where `boxes` is the tracked subject's.
        # A zoomed reference is framed on the part when a part is replaced and on the person otherwise
        # (`window_boxes` has the rule).
        region_boxes = _tracked_boxes(mask)
        zoomed = None
        if motion_reference == MOTION_ZOOM:
            # for the preview and the log: each shot's box over the whole clip, at the frames' own size
            framed, what = (boxes, "the tracked subject") if replace == REPLACE_WHOLE else (region_boxes, "what is replaced")
            zoomed = shot_boxes(framed, shot_ranges({"shot_table": table}), int(frames.shape[2]), int(frames.shape[1]))
            sizes = sorted({(x1 - x0, y1 - y0) for x0, y0, x1, y1 in zoomed.tolist() if x0 >= 0})
            logger.info("[h3] MiniMaxH3MaskedSource: zoomed in on %s, one box a shot, %s on %dx%d "
                        "frames over the clip; each window takes its own from its own frames", what,
                        ", ".join(f"{bw}x{bh}" for bw, bh in sizes) or "no box (the subject is in no frame)",
                        int(frames.shape[2]), int(frames.shape[1]))
        return io.NodeOutput({"frames": frames, "mask": mask, "grow_pixels": int(grow_pixels),
                              # the margin's rule and what it reads (`source_margins`): [N] in 0..1
                              "grow_by": grow_by, "subject_area": area,
                              # how fine the region's edge is (`EDGES`; `window`)
                              "edge": edge,
                              "feather_pixels": int(feather_pixels), "paint_out": bool(paint_out),
                              "composite": composite, "change_threshold": float(change_threshold),
                              "motion_reference": motion_reference, "motion_short_edge": int(motion_short_edge),
                              "motion_vae": bool(motion_vae),
                              # [N, h, w, 3] or None: the video wired as the movement (`wired_motion`)
                              "motion_frames": motion_video,
                              # the second wired video and its own short edge, or None (`wired_motion`, `which=2`)
                              "motion_frames_2": motion_video_2, "motion_short_edge_2": int(motion_short_edge_2),
                              # [N, H, W] of 0 or 1, or None: what stays the source's inside the region (`window`)
                              "keep": keep,
                              # [N, H, W] of 0 or 1, or None: people the margin does not grow over (`window`)
                              "others": others,
                              # the part mask's coverage of the tracked subject, when it is in doubt
                              # (`part_coverage.py`); shown by the song node and the prompt node
                              PART_WARNING: part_warning,
                              "start_from": start_from, "start_top": float(start_top),
                              "start_blur": int(start_blur), "start_knots": int(start_knots),
                              # [N, 4] long, (x0, y0, x1, y1) on these frames with the far side exclusive, a
                              # row of -1 where the subject is absent: the TRACKED subject, whatever is replaced
                              "subject_boxes": boxes,
                              # the same for the mask this record carries (the part, on a parts graph), and which
                              # of the two a zoomed reference and a blur's share read (`window_boxes`)
                              "region_boxes": region_boxes,
                              # where the region is rendered; the box is made where the canvas is known
                              # (`zoomed_source`, called by the song node)
                              "render_region": render_region,
                              # the most a zoomed load is enlarged (`zoom_up_to`; `zoomed_source`)
                              "zoom_up_to": float(zoom_up_to),
                              # read by the prompt node (`masked_prompt.py`), which describes what is replaced
                              "shot_table": table, "replace": replace,
                              # the frames that start a shot: the composite lays nothing across one of them
                              # from the subject (`cut_gate`)
                              "cuts": cuts,
                              **({"pad_before": pad_before, "pad_after": pad_after} if (pad_before or pad_after) else {}),
                              # on a padded window: what its context frames show (`CONTEXT_SHOWS`), and under `face
                              # covered` those of them across a cut from the shot (`context_across_cut`)
                              **context},
                             mask.to(torch.float32),
                             preview_strip(frames, mask, each, motion_reference,
                                           int(motion_short_edge), int(grow_pixels) // 2, zoomed, others,
                                           motion_video, motion_video_2, int(motion_short_edge_2)))

    @classmethod
    def _settle_mask(cls, frames, mask, replace, segmenter, segmenter_clip, part_phrases, part_threshold,
                     part_margin, parts=None):
        """The mask this node uses, from the tracked one: checked against the frames, and cut to the part for
        `head and hair`. Returns it with what the log line says about it, and with the part mask's coverage
        warning (`part_coverage.Summary.warning`): a line when the wired parts leave frames in doubt, else None."""
        if mask.ndim == 4 and int(mask.shape[-1]) == 1:
            mask = mask[..., 0]
        if mask.ndim != 3:
            raise ValueError(f"frames must be [N, H, W, C] and mask [N, H, W]; got {tuple(frames.shape)} and {tuple(mask.shape)}")
        if int(mask.shape[0]) != int(frames.shape[0]):
            raise ValueError(
                f"{int(frames.shape[0])} frames and {int(mask.shape[0])} masks: the mask must be tracked "
                "over the same frames the source node loaded")
        if tuple(mask.shape[1:]) != tuple(frames.shape[1:3]):
            raise ValueError(
                f"the mask is {int(mask.shape[2])}x{int(mask.shape[1])} and the frames are "
                f"{int(frames.shape[2])}x{int(frames.shape[1])}: each is cropped to the canvas by its own "
                "shape, so a mask of another shape would land shifted. Track the mask on these frames")
        note = ""
        warning = None
        if replace == REPLACE_PART:
            phrases = tuple(p.strip() for p in str(part_phrases).split(",") if p.strip())
            if not phrases:
                raise ValueError("part_phrases is empty: name what SAM 3 should find on the subject, e.g. `hair, head`")
            if segmenter is None or segmenter_clip is None:
                raise ValueError(
                    f"replace `{replace}` finds the part with SAM 3: wire the SAM 3 checkpoint's model into "
                    "`segmenter` and its text encoder into `segmenter_clip`")
            where = (mask > 0.5).flatten(1).any(dim=1).nonzero().flatten()
            region = torch.zeros_like(mask, dtype=torch.float32)
            carried = 0
            if where.numel():
                subject = mask[where].to(torch.float32)
                found = select_part(subject, detect_part(segmenter, segmenter_clip, frames, where, phrases,
                                                         part_threshold).to(mask.device), int(part_margin))
                bottoms, carried = carry_missing(part_bottom(found))
                if bool((bottoms < 0).all()):
                    raise ValueError(
                        f"SAM 3 found none of {list(phrases)} on the subject in any frame: change `part_phrases`, "
                        "or set `replace` to whole subject")
                region[where] = above(subject, bottoms)
            note = (f", from {list(phrases)} at threshold {float(part_threshold):g}, found on {int(where.numel()) - carried} of the {int(where.numel())} "
                    f"frames the subject is in" + (f" and carried from the nearest frame on {carried}" if carried else ""))
            mask = region
        elif replace == REPLACE_PARTS:
            if parts is None:
                raise ValueError(
                    f"replace `{replace}` regenerates the part node's mask: wire MiniMaxH3SubjectParts' `parts` "
                    "output into `parts`, or set `replace` to head and hair for the SAM phrase path")
            if parts.ndim == 4 and int(parts.shape[-1]) == 1:
                parts = parts[..., 0]
            if tuple(parts.shape) != tuple(mask.shape):
                raise ValueError(
                    f"`parts` is {tuple(parts.shape)} and the mask {tuple(mask.shape)}: the part node must run "
                    "on the same frames and the same subject mask this node takes")
            # How much of the tracked subject the part covers, counted before the part replaces the mask:
            # a part model that labelled someone else leaves a sliver here and the original in the render.
            # A report: nothing is refused and no mask changes.
            told = summarise(coverage(mask, parts.to(mask.device)))
            for line in told.lines():
                logger.info("[h3] MiniMaxH3MaskedSource: the wired parts: %s", line)
            warning = told.warning()
            where = (mask > 0.5).flatten(1).any(dim=1).nonzero().flatten()
            region = torch.zeros_like(mask, dtype=torch.float32)
            found_on = 0
            if where.numel():
                subject = mask[where].to(torch.float32)
                # the part node already cut its mask to the subject widened by its own margin;
                # cutting again here only guards a part node run on another subject
                part = select_part(subject, parts[where].to(mask.device).to(torch.float32), int(part_margin))
                found_on = int((part > 0.5).flatten(1).any(dim=1).sum())
                if not found_on:
                    raise ValueError(
                        "`parts` is empty on every frame the subject is in: tick a part on the part node, "
                        "or set `replace` to whole subject")
                region[where] = part
            note = (f", from the wired parts, present on {found_on} of the {int(where.numel())} frames the "
                    "subject is in")
            mask = region
        return mask, note, warning
