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
is refused and no mask changes. A mask kept from an earlier run arrives
without the part mask, so the key is None there.

The record this node returns, by key, beyond the node's own inputs: `frames`
and `mask` (the frames and the region used), `shot_table` and `replace` (read
by the prompt node), `subject_boxes` (the tracked subject's box per frame),
and `part_warning`.

**The mask is kept across runs** (`reuse_mask`; owner, 2026-10-04: "save the
mask"). Tracking the subject, and finding the part for `head and hair`, cost
more than a window of sampling after every restart, and a clip's mask does
not change between the arms of a test. The node keeps its finished mask on
disk under a key made from everything that decides it, and asks core for
`mask`, `segmenter` and `segmenter_clip` only when nothing kept matches, so on
a hit the tracker and the detector never run. A hit is the same bytes as the
tracked mask. `mask_store.py` has the key, the format and the budget;
`MASK_KEY_SKIP` below is the list of this node's inputs that do not change
its mask.

**The shot table travels with the mask** (2026-10-05, `shot_table.py`). The
Subject Track's table says who was found and taken in each shot, and a
person reviews it in place of a render. Wired into `shot_table` here, it is
asked for only when the mask is (both are lazy and both come from the
tracker), kept in the mask's file, read back on a hit, and handed on in the
source bundle, where the song node writes it beside the video under the
render's own number. Wiring it does not change the mask's key. A mask kept
without a table is a miss the first time a graph wires one.

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

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from comfy_api.latest import io

import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

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
#: This node's inputs that do not change the mask it settles on: they act on
#: the mask afterwards (the grow, the feather, the composite) or on the frames.
#: Everything else on the node, and everything upstream of it, is in the key a
#: kept mask is found by (`mask_store.mask_key`). Reasoned, from `execute`: the
#: mask is final before any of these is read. `bench/check_mask_store.py`
#: holds both directions.
#: The kept mask on disk, as a whole: read and written only while this is True. **False since 2026-10-07 by the
#: owner's decision, and to be turned on by a code change and nothing else**: `reuse_mask` on the node has no effect
#: while it is False. A kept mask is only as fresh as the `MASK_VERSION` of the nodes that made it; a change to how a
#: subject is followed that forgot to bump one was served the old mask that evening, and the fix it carried looked
#: like it had failed. **Before it is turned back on, the store has to detect that the code changed**: the key holds
#: the inputs and hand-bumped version numbers, and nothing derived from the code that makes the mask, so it cannot
#: tell a mask made by yesterday's logic from today's (the owner, the same evening).
MASK_REUSE_ENABLED = False

MASK_KEY_SKIP = ("grow_pixels", "grow_by", "feather_pixels", "paint_out", "composite", "change_threshold", "reuse_mask",
                 "motion_reference", "motion_short_edge", "motion_vae",
                 "start_from", "start_top", "start_blur", "start_knots", "shot_table", "keep")
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
MOTIONS = (MOTION_NONE, MOTION_SUBJECT, MOTION_ZOOM, MOTION_FRAME)
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
#: The inputs core is asked for only when no kept mask matches.
LAZY_FOR_MASK = ("mask", "segmenter", "segmenter_clip")
#: The tracker's shot table (`shot_table.py`), lazy with the mask and kept
#: with it: a kept mask that carries a table spares the tracker for both, and
#: one that carries none is a miss for a graph that wires the table, once.
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
        each = torch.cat([each, each.new_full((short,), int(source["grow_pixels"]))])
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


def token_mask(mask: torch.Tensor, latent_t: int, lat_h: int, lat_w: int) -> torch.Tensor:
    """A [F, H, W] pixel mask (above 0.5 = regenerate) to [latent_t, lat_h, lat_w] of 0 or 1.

    F must be the window's frame count, the sum of `run_lengths(latent_t)`.
    Every value in a 2x2 patch is the same, which is what core's own pooling
    would make of it.
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
    # core's patch pooling (`mask_row_values`): replicate-pad to even, max per 2x2
    m = F.pad(m.unsqueeze(1), (0, lat_w % 2, 0, lat_h % 2), mode="replicate")[:, 0]
    tokens = m.reshape(latent_t, m.shape[-2] // 2, 2, m.shape[-1] // 2, 2).amax(dim=(2, 4))
    return tokens.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[:, :lat_h, :lat_w].contiguous()


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
    repeat the last one with nothing masked (`window` says why). A window that
    starts past the source's end is refused.
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
        mask = torch.cat([mask, torch.zeros((short,) + tuple(mask.shape[1:]), dtype=mask.dtype, device=mask.device)], dim=0)
    return pixels, mask, short


def _tracked_boxes(mask: torch.Tensor) -> torch.Tensor:
    """The subject's box on each frame of a [N, H, W] mask: [N, 4] long, (x0, y0, x1, y1) with the far side
    exclusive, a row of -1 where the mask is empty. The pack's one box function, `sapiens2_parts.mask_boxes`.

    Kept as one function on purpose. The shot table is to carry the subject's box per frame (it lands with
    the kept mask's removal); that day this reads the table and computes nothing, and nothing else changes.
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
    `window_frames` fills with the last one unmasked, have no subject of their own: they carry their shot's
    box like any frame the subject is off, and show grey in it.
    """
    rows = source.get("subject_boxes")
    if rows is None:
        return None
    first_frame, frames = int(first_frame), int(frames)
    part = rows[first_frame:first_frame + frames]
    short = frames - int(part.shape[0])
    if short > 0:
        part = torch.cat([part, torch.full((short, 4), -1, dtype=part.dtype)], dim=0)
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
    one fixed box per shot, around the tracked subject) and not in the whole
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
                "(`window_boxes`); the source record carries them as `subject_boxes`")
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
                  margin, boxes: torch.Tensor | None = None) -> torch.Tensor:
    """[1, H, W, 3]: sampled frames down the strip, the regenerated region tinted red on the plate, and, when a
    motion reference is on, what the encoder is shown beside each. The dry-run review looks at this and the
    tracker's tiles before anything samples.

    With `boxes` (the zoomed reference's box on every frame of the clip) each plate also carries its box as an
    outline, so the picture beside it can be found on the frame. The boxes here are each shot's over the whole
    clip; a window takes its own from its own frames (`window_boxes`), which is never larger.

    `grow_pixels` and `margin` are one number each, or every frame's own over the clip (`margins`)."""
    n = int(frames.shape[0])
    idx = torch.linspace(0, n - 1, steps=min(PREVIEW_ROWS, n)).round().long()
    f = frames[idx, ..., :3].to(torch.float32)
    grow_pixels = grow_pixels[idx] if torch.is_tensor(grow_pixels) else grow_pixels
    margin = margin[idx] if torch.is_tensor(margin) else margin
    region = (grow(mask[idx].to(torch.float32), grow_pixels) > 0.5).unsqueeze(-1).to(f.dtype)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=f.dtype, device=f.device)
    plate = f * (1.0 - region) + (0.5 * f + 0.5 * red) * region
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
    ref = motion_reference(frames[idx], mask[idx], motion, short_edge, margin, shown)
    if ref is not None:
        tiles.append(ref.to(f.dtype).to(f.device))
    h = PREVIEW_HEIGHT
    scaled = []
    for t in tiles:
        w = max(8, int(round(int(t.shape[2]) * h / int(t.shape[1]))))
        scaled.append(F.interpolate(t.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False,
                                    antialias=True).movedim(1, -1))
    row = torch.cat(scaled, dim=2)                                   # the plate and the reference side by side
    strip = row.reshape(1, int(row.shape[0]) * h, int(row.shape[2]), 3)   # the sampled frames down the strip
    return strip.clamp(0.0, 1.0)


def window(source: dict, first_frame: int, frames: int, width: int, height: int,
           latent_t: int, lat_h: int, lat_w: int):
    """One window of a source: its fitted frames, the frames to encode, its token mask, its fitted mask, frames held.

    With a `keep` mask on the source the token mask is the grown region less
    every token the keep mask touches: the model is given those tokens clean,
    as it is the rest of the plate, and regenerates around them.

    The frames to encode are the fitted frames themselves, or with `paint_out`
    a copy with the subject filled in: the mask widened by half of
    the margin (`source_margins`: `grow_pixels`, or under `grow_by` the
    subject's size each frame's own), which takes the soft edge SAM leaves on a fast limb and
    keeps the other half of the margin real background for the kept tokens.
    That hole is inside the regenerated tokens, so no filled pixel is shown.

    A loop's last window ends at or past the end of its track
    (`loop_plan.py`, "Lengths"), so the source can run out inside it. The
    missing frames repeat the last one with nothing masked: they are past the
    track and are not written (`loop_plan.frames_kept`), and a held plate is
    what a frozen row should see.
    A window that starts past the source's end is refused; that is a source
    that does not belong to this track.
    """
    pixels, mask, short = window_frames(source, first_frame, frames, width, height)
    margin = source_margins(source, first_frame, frames, int(width) * int(height))
    tokens = token_mask(grow(mask, margin), latent_t, lat_h, lat_w)
    if source.get("keep") is not None:
        # After the grow, so the margin cannot run back over what is kept, and in whole tokens: a token
        # that holds any kept pixel on any of its frames is kept. The composite and the review read
        # `tokens`, so the source's own pixels are what is shown there.
        held = fit_mask(source["keep"][int(first_frame):int(first_frame) + int(frames)], width, height)
        if short > 0:
            held = torch.cat([held, torch.zeros((short,) + tuple(held.shape[1:]), dtype=held.dtype,
                                                device=held.device)], dim=0)
        tokens = tokens * (1.0 - token_mask(held, latent_t, lat_h, lat_w))
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


class MiniMaxH3MaskedSource(io.ComfyNode):
    #: Part of a kept mask's key (`mask_store.py`). Raise it when a change
    #: would give a different mask from the same inputs and settings: that is
    #: `_settle_mask` and what it calls. The grow, the feather and the
    #: composite act after the mask and do not count.
    MASK_VERSION = 1

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
                # lazy since 2026-10-04: not asked for when a kept mask matches (`check_lazy_status`)
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
                                        "and not in the whole frame, so a subject that is small in the frame "
                                        "is seen larger for the same cost or less. The box holds still through "
                                        "a shot and changes only where the video cuts; when a window holds "
                                        "shots framed differently, each sits in the middle of one shared "
                                        "picture on grey. Use it when the model does not follow a small "
                                        "subject's movement.\n\n"
                                        "whole frame: the source window as it is.\n\n"
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
                io.Int.Input("grow_pixels", default=GROW_PIXELS, min=0, max=512,
                             tooltip=("How far the mask is widened before it reaches the model, in pixels of "
                                      "the render canvas. Raise it when the replacement is cut off at its "
                                      "edge. The hole's shape is all that tells the model where the original "
                                      "stood, so a wider one lets the new subject stand somewhere else and "
                                      "uncover what the original hid.")),
                io.Int.Input("feather_pixels", default=8, min=0, max=128,
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
                # appended 2026-10-04 (`mask_store.py`)
                # Off by default since 2026-10-07 (the owner: off until this lane is ready for production). A kept
                # mask is only as fresh as the `MASK_VERSION` of the nodes that made it, and a change to how a
                # subject is followed that forgets to bump one is served the old mask with no sign of it: that
                # evening a fix's confirmation render showed the unfixed result for exactly that reason.
                io.Boolean.Input("reuse_mask", default=False, optional=True,
                                 tooltip=("DISABLED FOR NOW: this switch has no effect. Every run tracks afresh "
                                          "and keeps nothing, whatever it is set to, until kept masks are turned "
                                          "back on in the code.\n\n"
                                          "What it does when enabled: the finished mask is kept on disk, and a later "
                                          "run with the same video and the same mask settings uses it without "
                                          "tracking again, also after a restart.")),
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
                # appended 2026-10-05 (`shot_table.py`): lazy with the mask and kept with it
                io.String.Input(LAZY_FOR_TABLE, optional=True, lazy=True, force_input=True,
                                tooltip=("The Subject Track's `shot_table` output. It is kept with the mask, "
                                         "so a render that reuses a kept mask still has it, and the song node "
                                         "writes it next to the video.")),
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
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[H3MaskedSource.Output(display_name="source"),
                     io.Mask.Output(display_name="mask",
                                    tooltip=("The mask this node used, one per source frame, before grow_pixels: "
                                             "preview it to see what will be replaced.")),
                     io.Image.Output(display_name="preview",
                                     tooltip=("A strip of sampled frames: the plate with the regenerated region "
                                              "tinted red, and beside it what the text encoder is shown as the "
                                              "motion reference when one is on. Look before rendering."))],
        )

    @classmethod
    def _mask_key(cls, frames, reuse_mask):
        """The key this node's mask is kept under, or None: turned off, or no queued prompt to read (a direct call)."""
        hidden = getattr(cls, "hidden", None)
        prompt, node_id = getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None)
        if not MASK_REUSE_ENABLED or not reuse_mask or frames is None or not isinstance(prompt, dict) or node_id is None:
            return None
        from . import mask_store
        return mask_store.mask_key(prompt, node_id, frames, skip=MASK_KEY_SKIP)

    @classmethod
    def check_lazy_status(cls, frames=None, replace=REPLACE_WHOLE, reuse_mask=False, **kwargs):
        # None is a connected input core has not run yet; an unconnected
        # optional input is absent. `frames` is not lazy, so it is here.
        wanted = {REPLACE_PART: LAZY_FOR_MASK, REPLACE_PARTS: (LAZY_FOR_MASK[0], LAZY_FOR_PARTS)}.get(replace, LAZY_FOR_MASK[:1]) + (LAZY_FOR_TABLE,)
        missing = [name for name in wanted if name in kwargs and kwargs[name] is None]
        if not missing:
            return []
        key = cls._mask_key(frames, reuse_mask)
        if key is not None:
            from . import mask_store
            # with the table wired, only a kept mask that carries one is a hit
            if mask_store.has(key, tuple(frames.shape[:3]), with_table=LAZY_FOR_TABLE in kwargs):
                return []           # a kept mask matches: the tracker and the detector do not run
        return missing

    @classmethod
    # `mask` has no default: it is required in the schema, and core hands a lazy input it was not asked to run
    # as None, which is what a hit on a kept mask looks like here.
    def execute(cls, frames, mask, grow_pixels=GROW_PIXELS, feather_pixels=8, replace=REPLACE_WHOLE, paint_out=False,
                segmenter=None, segmenter_clip=None, part_phrases=PART_PHRASES,
                part_threshold=PART_THRESHOLD, part_margin=PART_MARGIN, composite=COMPOSITE_CHANGED,
                change_threshold=CHANGE_THRESHOLD, reuse_mask=False, motion_reference=MOTION_NONE,
                motion_short_edge=MOTION_SHORT_EDGE, motion_vae=False, start_from=START_NOISE,
                start_top=START_TOP_SHARE, start_blur=START_BLUR, start_knots=START_KNOTS,
                shot_table=None, parts=None, keep=None, grow_by=GROW_FIXED) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        if int(feather_pixels) > int(grow_pixels):
            raise ValueError(
                f"feather_pixels {int(feather_pixels)} is wider than grow_pixels {int(grow_pixels)}: the blend "
                "would reach the subject's own pixels and bring the original back at its edge")
        if grow_by not in GROW_BY:
            raise ValueError(f"unknown grow_by {grow_by!r}; one of {list(GROW_BY)}")
        if replace not in (REPLACE_WHOLE, REPLACE_PART, REPLACE_PARTS):
            raise ValueError(f"unknown replace {replace!r}; one of {[REPLACE_WHOLE, REPLACE_PART, REPLACE_PARTS]}")
        if start_from not in (START_NOISE, START_TOP):
            raise ValueError(f"unknown start_from {start_from!r}; one of {[START_NOISE, START_TOP]}")
        if start_from != START_NOISE and paint_out:
            raise ValueError(
                "paint_out fills the subject in before the encode, so a softened start has nothing to "
                "soften: turn one of the two off")
        if composite not in (COMPOSITE_REGION, COMPOSITE_CHANGED):
            raise ValueError(f"unknown composite {composite!r}; one of {[COMPOSITE_REGION, COMPOSITE_CHANGED]}")
        if motion_reference not in MOTIONS:
            raise ValueError(f"unknown motion_reference {motion_reference!r}; one of {list(MOTIONS)}")
        if keep is not None:
            if keep.ndim == 4 and int(keep.shape[-1]) == 1:
                keep = keep[..., 0]
            if tuple(keep.shape) != tuple(frames.shape[:3]):
                raise ValueError(
                    f"`keep` is {tuple(keep.shape)} and the frames {tuple(frames.shape[:3])}: it needs one mask "
                    "per source frame at the frames' own size, from a node that ran on the same `frames`")
            if paint_out or start_from != START_NOISE:
                # both show a changed copy of the source under the subject, which a kept token would put on screen
                raise ValueError(
                    "`keep` shows the source's own pixels inside the region, and paint_out and a softened start "
                    "both change those pixels before the encode: turn them off, or unwire `keep`")
            keep = (keep > 0.5).to(torch.float32)
        key = cls._mask_key(frames, reuse_mask)
        kept = None
        if key is not None:
            from . import mask_store
            kept = mask_store.load(key, tuple(frames.shape[:3]))
        table = str(shot_table or "")
        part_warning = None     # a kept mask arrives without the part mask, so there is nothing to count
        if kept is not None:
            mask, note = kept, ", mask kept from an earlier run (nothing tracked)"
            # The tracker did not run, so the subject's boxes are the kept REGION's. This branch goes with
            # the kept mask's removal (`reuse_mask`, `mask_store`): delete it with them.
            boxes = _tracked_boxes(mask)
            if motion_reference == MOTION_ZOOM and replace != REPLACE_WHOLE:
                note += ", the zoom framed on the kept region and not the whole subject"
            if table:
                # the tracker ran for its table alone: the kept file had none (kept before the
                # table existed, or by a graph that did not wire it). Keep it for the next run.
                if mask_store.table(key) != table:
                    mask_store.save(key, mask, table)
                    note += ", its shot table kept with it now"
            else:
                table = mask_store.table(key)
        else:
            if mask is None:
                # `check_lazy_status` found a kept mask and it did not read here: it has been removed
                raise ValueError(
                    "the mask kept for this video could not be read and has been removed: queue the "
                    "workflow again and it will be tracked afresh")
            tracked = mask          # the tracker's own mask, before it is cut to a part below
            mask, note, part_warning = cls._settle_mask(frames, mask, replace, segmenter, segmenter_clip,
                                                        part_phrases, part_threshold, part_margin, parts)
            # The whole subject's box per frame, from the tracker's mask and not from the part it was just
            # cut to: on a parts graph nothing after this node knows where the subject is otherwise.
            boxes = _tracked_boxes(tracked[..., 0] if tracked.ndim == 4 else tracked)
            if key is not None:
                seconds = mask_store.save(key, mask, table)
                note += f", mask kept for the next run ({seconds:.0f} s to write)"
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
        if motion_reference != MOTION_NONE:
            logger.info("[h3] MiniMaxH3MaskedSource: motion reference %s at a %d short edge, %s", motion_reference,
                        int(motion_short_edge), "with the video model's copy" if motion_vae else "text encoder only")
        zoomed = None
        if motion_reference == MOTION_ZOOM:
            # for the preview and the log: each shot's box over the whole clip, at the frames' own size
            zoomed = shot_boxes(boxes, shot_ranges({"shot_table": table}), int(frames.shape[2]), int(frames.shape[1]))
            sizes = sorted({(x1 - x0, y1 - y0) for x0, y0, x1, y1 in zoomed.tolist() if x0 >= 0})
            logger.info("[h3] MiniMaxH3MaskedSource: zoomed in on the tracked subject, one box a shot, %s on %dx%d "
                        "frames over the clip; each window takes its own from its own frames",
                        ", ".join(f"{bw}x{bh}" for bw, bh in sizes) or "no box (the subject is in no frame)",
                        int(frames.shape[2]), int(frames.shape[1]))
        return io.NodeOutput({"frames": frames, "mask": mask, "grow_pixels": int(grow_pixels),
                              # the margin's rule and what it reads (`source_margins`): [N] in 0..1
                              "grow_by": grow_by, "subject_area": area,
                              "feather_pixels": int(feather_pixels), "paint_out": bool(paint_out),
                              "composite": composite, "change_threshold": float(change_threshold),
                              "motion_reference": motion_reference, "motion_short_edge": int(motion_short_edge),
                              "motion_vae": bool(motion_vae),
                              # [N, H, W] of 0 or 1, or None: what stays the source's inside the region (`window`)
                              "keep": keep,
                              # the part mask's coverage of the tracked subject, when it is in doubt
                              # (`part_coverage.py`); shown by the song node and the prompt node
                              PART_WARNING: part_warning,
                              "start_from": start_from, "start_top": float(start_top),
                              "start_blur": int(start_blur), "start_knots": int(start_knots),
                              # [N, 4] long, (x0, y0, x1, y1) on these frames with the far side exclusive, a
                              # row of -1 where the subject is absent: the TRACKED subject, whatever is replaced
                              "subject_boxes": boxes,
                              # read by the prompt node (`masked_prompt.py`), which describes what is replaced
                              "shot_table": table, "replace": replace},
                             mask.to(torch.float32),
                             preview_strip(frames, mask, each, motion_reference,
                                           int(motion_short_edge), int(grow_pixels) // 2, zoomed))

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
