"""Body parts and a soft matte on one tracked person, from Sapiens2.

What it is for (the masked video-to-video lane, `docs/wiki/masked_v2v.md`):
the Masked Source's `head and hair` finds its part by asking SAM 3 for a
phrase on every frame the subject is in, which cost minutes before sampling on
the band clip and misses frames (`bench/results/2026-10-04_masked_v2v_band.md`,
`band_head_w1`). Sapiens2's segmentation model labels every pixel of a person
with one of `CLASS_NAMES` in one forward pass per frame, and its matting model
gives an alpha for the person's edge. A part menu, a garment swap and a redub
of the mouth are the same map read with a different choice of classes.

**What the subject holds** (`Found.holds`, the `held` output, 2026-10-07). The
map has no class for a cigarette or a cup, and the tracker's mask of a person
takes in what they hold, so inside the subject's own mask a pixel this model
calls background is something attached to them that is not them. Taken near
the lips or a hand (`HELD_ANCHORS`, `held_near`), it is a mask of the held
thing with no name asked of any model and nothing small tracked on its own:
it rides on the subject's track. Wired to the Masked Source's `keep`, the
thing stays the original's while the person is replaced. It cannot see a
thing the tracker's mask leaves out (a thin tip past the subject's edge), and
a part of the body the model fails to label near a hand counts as held.
Reasoned and checked on a painted frame; not yet run on a clip.

**The steps** (`subject_parts`), per frame the subject is in:

0. The subject alone (`alone`): before the crop is taken, every pixel further
   than `ALONE_MARGIN` from the tracked subject's mask is replaced by the
   colour the crop is filled with off the frame, which is zero to the model.
   The model is trained on one person in a frame; in a crowd it can label a
   neighbour in front of the subject and leave the subject background, and
   the cut in step 3 then keeps almost nothing. It is done on the frame, so
   the crop's surroundings are flat too. Both models are run on that one
   crop: the matte's alpha is then of the same person the labels are. The
   margin is never less than `subject_margin`, so nothing this node returns
   was computed on a replaced pixel. This changes what the model is shown
   and not the rule about what is kept: step 3 is as it was. **Open**: what
   the matting model does at the person's edge with a flat surround a few
   pixels outside it is not measured; the masked graphs wire `parts`, not
   `matte`, so no render has depended on it.
1. The crop (`mask_boxes`, `crop_box`, `take_crops`): the box of the tracked
   subject's mask, widened by `crop_margin`, then widened on its short side to
   the model's own shape so the person is not squashed, and resized to the
   model's working size. The lane's frames are landscape with several people;
   the model is trained on one person in a portrait frame
   (`coderef/sapiens2/docs/SEG.md`, the size in each folder's
   `preprocessor_config.json`). Where the box leaves the frame it is filled
   with the mean colour, which is zero after normalisation.
2. The label map: the logits are brought back to the box's size the way
   upstream's own tool does it (bilinear, then the highest class;
   `coderef/sapiens2/sapiens/dense/tools/vis/vis_seg.py`) and laid back on
   the frame (`paste_back`).
3. The cut to the subject: a label counts only where it lies on the subject's
   mask widened by `subject_margin`, so a neighbour inside the crop gives
   nothing. Everything this node returns lies inside that widened mask.
4. The matte, when a matting model is loaded: its alpha is the whole person's,
   not a part's. It is kept on the chosen part, and on what the label map
   calls background within `matte_reach` pixels of the part, which is where a
   soft edge lies outside a hard one. A border between two parts of the same
   person (hair against a collar) stays hard: it is not an edge of the
   person. With no matting model the matte is the part mask.
5. A missed frame (`hold`): a frame where the subject is on screen and none of
   the chosen classes is found on them takes the part of the nearest frame
   that has one, moved from that frame's subject box to this one's and cut to
   this frame's subject. A frame the subject is not in stays empty: holding a
   part into it would regenerate pixels of a shot they are not in.

Nothing is temporal in the model, so a label map can flicker from frame to
frame. The report says on how many frames each class was seen, which frames
were held, and how much of the subject the parts cover (`part_coverage.py`):
"found" means one pixel, and a part that covers a sliver of the subject leaves
the original in the render.

**The class names** are not in the checkpoint: its config calls them
`LABEL_0` to `LABEL_28`. `CLASS_NAMES` is upstream's table, and whether the
checkpoint's index order is that table's is what
`bench/results/2026-10-05_sapiens2_first_frame.md` records.

**Memory.** Each model is given to core as a patcher and asked for with
`load_models_gpu`, so core moves the DiT out when this node runs and moves
these out when the sampler needs the card. A transformers model is held
through `_Held` because core writes `.device` on what a patcher holds.

Nothing here patches core, and nothing is imported from `coderef/`. The models
are the installed transformers' own classes on a local folder.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

from .part_coverage import Coverage, coverage, ranges, summarise
from .video_mask import PART_MARGIN, grow

logger = logging.getLogger(__name__)

H3Sapiens2 = io.Custom("H3_SAPIENS2")

#: The folder type the loader lists: `models/sapiens2/`, one sub-folder per
#: model as Hugging Face lays it out (`config.json`, `model.safetensors`,
#: `preprocessor_config.json`).
FOLDER = "sapiens2"
#: The architecture each kind of folder names in its `config.json`.
SEG_ARCH = "Sapiens2ForSemanticSegmentation"
MATTE_ARCH = "Sapiens2ForImageMatting"
#: The `matting` choice that loads no matting model.
NO_MATTING = "none"

#: The segmentation classes by index. Inherited: upstream's `DOME_CLASSES_29`
#: (`coderef/sapiens2/sapiens/dense/src/datasets/seg/seg_utils.py`), the same
#: list as `coderef/sapiens2/docs/SEG.md`. The checkpoint's config carries no
#: names, so the order is checked against a frame, not read from a file:
#: `bench/results/2026-10-05_sapiens2_first_frame.md`.
CLASS_NAMES = (
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm",
    "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
    "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
    "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth",
    "Upper_Teeth", "Tongue",
)
#: The preview's colour for each class. Inherited: the same upstream table.
PALETTE = (
    (50, 50, 50), (255, 218, 0), (14, 204, 182), (128, 200, 255), (255, 0, 109), (189, 0, 204), (255, 0, 218),
    (0, 160, 204), (0, 255, 145), (204, 0, 131), (182, 0, 255), (255, 109, 0), (0, 255, 255), (72, 0, 255),
    (204, 131, 0), (255, 0, 0), (72, 255, 0), (189, 204, 0), (182, 255, 0), (102, 0, 204), (32, 72, 204),
    (0, 145, 255), (14, 204, 0), (0, 128, 72), (235, 205, 119), (115, 227, 112), (157, 113, 143), (132, 93, 50),
    (82, 21, 114),
)
BACKGROUND = 0

_MOUTH = ("Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")
#: The menu: each boolean input and the classes it takes. Reasoned: the face
#: carries its glasses and its mouth, since the label map cuts both out of
#: `Face_Neck` and a face is never wanted with holes in it; `mouth` alone is
#: the redub's region.
PARTS = {
    "hair": ("Hair",),
    "face_and_neck": ("Face_Neck", "Eyeglass") + _MOUTH,
    "upper_clothing": ("Upper_Clothing",),
    "lower_clothing": ("Lower_Clothing",),
    "hands": ("Left_Hand", "Right_Hand"),
    "mouth": _MOUTH,
}
#: The menu entries on by default. Reasoned: the region the Masked Source's
#: `head and hair` asks SAM 3 for, which is what this node replaces first.
PARTS_ON = ("hair", "face_and_neck")

#: The default of `crop_margin`, in pixels of the source frame. Reasoned, not
#: measured: room for hair and a raised hand outside the tracker's box, and a
#: little of the scene around the person as the model's training frames have.
CROP_MARGIN = 32
#: The default of `subject_margin`. Inherited: `video_mask.PART_MARGIN`, the
#: same allowance for the same reason (two models' edges do not coincide).
SUBJECT_MARGIN = PART_MARGIN
#: The default of `matte_reach`, in pixels of the source frame. Reasoned, not
#: measured: a soft edge is a few pixels wide at this lane's frame sizes.
MATTE_REACH = 8
#: How far past the tracked subject's mask the part model still sees the
#: picture, in pixels of the source frame; further out it is shown flat
#: colour. **Tested at this value and no other, on one window of each of
#: two clips** (`bench/results/2026-10-06_part_model_shown_subject_alone.md`):
#: on one, the parts cover the subject from the window's first frame where
#: without it they lay off the subject for a stretch; on the other, where
#: the parts were already on the subject, the coverage is the same to within
#: a few points and the part fills more of this margin. **Reasoned floor**:
#: `subject_parts` never uses less than `subject_margin`, since a label
#: counts out to there.
ALONE_MARGIN = 8
#: `held`: what the subject holds, with no name asked of any model. The
#: tracker's mask of a person takes in what they hold, and the part model has
#: no class for it, so inside the subject's own mask a pixel the part model
#: calls background is something attached to them that is not them. It is
#: taken only near the lips or a hand (`HELD_ANCHORS`), where a thing is held.
#: The owner's idea and Gemini's hand-anchor sketch, 2026-10-07; the mouth is
#: here because the owner's first example starts there.
HELD_ANCHORS = _MOUTH + ("Left_Hand", "Right_Hand")
#: The default of `held_near`, in pixels of the source frame: how far from the
#: lips or a hand a held thing may reach. Reasoned, not measured: about a
#: hand's length at this lane's frame sizes, so a cone or a cup is covered and
#: a thing on the other side of the body is not.
HELD_NEAR = 64
#: The default of `held_smallest`, in pixels: fewer than this on a frame is
#: taken for speckle at the mask's edge and dropped. Reasoned, not measured.
HELD_SMALLEST = 24
#: Crops per forward pass. Reasoned, not measured: the head's last layers hold
#: the working size at tens of channels per crop, which is small next to the
#: weights, and a larger batch buys little on one card.
BATCH = 4
#: The most frames the preview shows, spread over the frames the subject is
#: in. Reasoned: enough to see a change across the clip in one glance.
PREVIEW_TILES = 8
#: A preview tile's height; its width follows the model's shape.
TILE_HEIGHT = 512


# ------------------------------------------------------------------ the menu

def _plain(name: str) -> str:
    return "".join(c for c in str(name).lower() if c.isalnum())


def chosen_classes(menu: dict[str, bool], other_classes: str = "") -> tuple[int, ...]:
    """The class indices a choice takes: the menu's entries that are on, and the named classes.

    A name is matched without case, spaces or underscores. An unknown name and an empty choice are refused.
    """
    by_name = {_plain(n): i for i, n in enumerate(CLASS_NAMES)}
    taken: set[int] = set()
    for entry, on in menu.items():
        if entry not in PARTS:
            raise ValueError(f"unknown part {entry!r}; the menu is {list(PARTS)}")
        if on:
            taken.update(by_name[_plain(n)] for n in PARTS[entry])
    for name in (p.strip() for p in str(other_classes or "").split(",")):
        if not name:
            continue
        if _plain(name) not in by_name:
            raise ValueError(f"other_classes names {name!r}, which is not a Sapiens2 class; the classes are: "
                             + ", ".join(CLASS_NAMES[1:]))
        taken.add(by_name[_plain(name)])
    taken.discard(BACKGROUND)
    if not taken:
        raise ValueError("no part is chosen: turn on at least one of " + ", ".join(PARTS)
                         + ", or name a class in other_classes")
    return tuple(sorted(taken))


# ------------------------------------------------------------- the geometry

def mask_boxes(mask: torch.Tensor) -> torch.Tensor:
    """Per frame, the box of a [N, H, W] mask as (x0, y0, x1, y1), the far side exclusive; -1 where empty. [N, 4] long."""
    on = mask > 0.5
    _, h, w = on.shape
    cols, rows = on.any(dim=1), on.any(dim=2)                      # [N, W], [N, H]
    x0, y0 = cols.to(torch.uint8).argmax(dim=1), rows.to(torch.uint8).argmax(dim=1)
    x1 = w - cols.flip(1).to(torch.uint8).argmax(dim=1)
    y1 = h - rows.flip(1).to(torch.uint8).argmax(dim=1)
    boxes = torch.stack([x0, y0, x1, y1], dim=1).to(torch.long)
    boxes[~cols.any(dim=1)] = -1
    return boxes


def crop_box(box, margin: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """A subject's box widened by `margin` and then, on its short side, to the shape of `size` (height, width).

    Centred on the subject. It may leave the frame; `take_crops` fills what does.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0 + 2 * int(margin), y1 - y0 + 2 * int(margin)
    # whole pixels, never smaller than the widened box: w * size_h / size_w, rounded up
    if h * size[1] < w * size[0]:
        tall, wide = -(-w * size[0] // size[1]), w
    else:
        tall, wide = h, -(-h * size[1] // size[0])
    left = (x0 + x1 - wide) // 2
    top = (y0 + y1 - tall) // 2
    return left, top, left + wide, top + tall


def _inside(box, height: int, width: int):
    """The part of a box that lies on the frame, and where that part sits in the box: two (x0, y0, x1, y1), or None."""
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return (fx0, fy0, fx1, fy1), (fx0 - x0, fy0 - y0, fx1 - x0, fy1 - y0)


def take_crops(frames: torch.Tensor, boxes, size: tuple[int, int], fill) -> torch.Tensor:
    """Each frame's box as a [3, size] picture: [n, H, W, 3] in, [n, 3, h, w] out, `fill` where the box leaves the frame."""
    height, width = int(frames.shape[1]), int(frames.shape[2])
    colour = torch.as_tensor(fill, dtype=torch.float32).view(3, 1, 1)
    out = []
    for frame, box in zip(frames, boxes):
        canvas = colour.expand(3, box[3] - box[1], box[2] - box[0]).clone()
        seen = _inside(box, height, width)
        if seen is not None:
            (fx0, fy0, fx1, fy1), (bx0, by0, bx1, by1) = seen
            canvas[:, by0:by1, bx0:bx1] = frame[fy0:fy1, fx0:fx1, :3].to(torch.float32).movedim(-1, 0)
        out.append(F.interpolate(canvas[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0])
    return torch.stack(out, dim=0)


def alone(frames: torch.Tensor, subject: torch.Tensor, margin: int, fill) -> torch.Tensor:
    """Frames with everything further than `margin` from the subject's mask replaced by `fill`: [n, H, W, C] and
    [n, H, W] in, [n, H, W, 3] out. A copy; the frames given are not written to."""
    seen = (grow(subject.to(torch.float32), int(margin)) > 0.5).unsqueeze(-1)
    colour = torch.as_tensor(fill, dtype=torch.float32).view(1, 1, 1, 3)
    return torch.where(seen, frames[..., :3].to(torch.float32), colour)


def paste_back(crop: torch.Tensor, box, height: int, width: int) -> torch.Tensor:
    """A [bh, bw] map made on a box, laid on an empty [height, width] frame; what lies off the frame is dropped."""
    out = torch.zeros((height, width), dtype=crop.dtype)
    seen = _inside(box, height, width)
    if seen is not None:
        (fx0, fy0, fx1, fy1), (bx0, by0, bx1, by1) = seen
        out[fy0:fy1, fx0:fx1] = crop[by0:by1, bx0:bx1]
    return out


def hold(part: torch.Tensor, from_box, to_box, hard: bool) -> torch.Tensor:
    """A [H, W] part moved from one subject box to another: what lay at a place in the first lies at the same place in the second."""
    height, width = part.shape
    fx0, fy0, fx1, fy1 = (int(v) for v in from_box)
    tx0, ty0, tx1, ty1 = (int(v) for v in to_box)
    piece = part[fy0:fy1, fx0:fx1].to(torch.float32)[None, None]
    moved = F.interpolate(piece, size=(ty1 - ty0, tx1 - tx0), mode="bilinear", align_corners=False)[0, 0]
    out = torch.zeros((height, width), dtype=torch.float32)
    out[ty0:ty1, tx0:tx1] = (moved > 0.5).to(torch.float32) if hard else moved.clamp(0.0, 1.0)
    return out


def _widened(box, margin: int, height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in box)
    return max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, width), min(y1 + margin, height)


# ---------------------------------------------------------------- the parts

@dataclass
class Found:
    parts: torch.Tensor                      # [N, H, W], 1 on the chosen classes of the subject
    matte: torch.Tensor                      # [N, H, W], the soft version, or `parts` with no matting model
    present: torch.Tensor                    # [N] bool, the subject is on the frame
    found: torch.Tensor                      # [N] bool, a chosen class was found on them there
    seen: torch.Tensor                       # [N, classes] long, pixels of each class on the subject
    held: list[int] = field(default_factory=list)       # frames given a neighbour's part
    boxes: dict[int, tuple] = field(default_factory=dict)        # the crop box of each frame run
    labels: dict[int, torch.Tensor] = field(default_factory=dict)  # the label map of each frame in `keep`
    coverage: Coverage | None = None         # how much of the subject `parts` covers, per frame
    holds: torch.Tensor | None = None        # [N, H, W], 1 on what the subject holds (`HELD_ANCHORS`), or None
    seconds: float = 0.0                     # the whole pass, both models


def subject_parts(frames: torch.Tensor, subject: torch.Tensor, classes: tuple[int, ...],
                  seg: Callable[[torch.Tensor], torch.Tensor], matting: Callable[[torch.Tensor], torch.Tensor] | None,
                  *, size: tuple[int, int], mean, std, crop_margin: int = CROP_MARGIN,
                  subject_margin: int = SUBJECT_MARGIN, matte_reach: int = MATTE_REACH, hold_missing: bool = True,
                  batch: int = BATCH, keep: tuple[int, ...] = (), show_alone: bool = True,
                  held_near: int = HELD_NEAR, held_smallest: int = HELD_SMALLEST) -> Found:
    """The chosen classes on the tracked subject, per frame. The module docstring has the steps.

    `seg` takes normalised crops [B, 3, h, w] and returns logits [B, classes, h', w']; `matting` returns alpha
    [B, 1, h', w'] in 0..1, or is None. Both are callables so a check can stand in for the models.
    `show_alone` is step 0. It is not an input of the node: off, the models are shown the picture as it is,
    which is what the node did until 2026-10-06 and what a check needs as its control.

    `Found.holds` is what the subject holds: inside their own mask (not the widened one: the ring around
    it is real background, and keeping that would keep the original's outline), labelled background by the
    part model, within `held_near` pixels of the lips or a hand, and at least `held_smallest` pixels on the
    frame. No frame is given a neighbour's: a held thing comes and goes.
    """
    n, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    tight = mask_boxes(subject)
    present = tight[:, 0] >= 0
    where = present.nonzero().flatten().tolist()
    out = Found(parts=torch.zeros((n, height, width), dtype=torch.float32),
                matte=torch.zeros((n, height, width), dtype=torch.float32), present=present,
                found=torch.zeros(n, dtype=torch.bool), seen=torch.zeros((n, len(CLASS_NAMES)), dtype=torch.long))
    wanted = torch.tensor(classes, dtype=torch.long)
    anchors = torch.tensor([CLASS_NAMES.index(c) for c in HELD_ANCHORS], dtype=torch.long)
    out.holds = torch.zeros((n, height, width), dtype=torch.float32)
    began = time.perf_counter()
    for i in range(0, len(where), int(batch)):
        index = where[i:i + int(batch)]
        boxes = [crop_box(tight[f], crop_margin, size) for f in index]
        shown = (alone(frames[index], subject[index], max(ALONE_MARGIN, int(subject_margin)), mean)
                 if show_alone else frames[index])
        crops = (take_crops(shown, boxes, size, mean) - mean_t) / std_t
        logits = seg(crops)
        alphas = matting(crops) if matting is not None else None
        wide = grow(subject[index].to(torch.float32), int(subject_margin)) > 0.5
        soft = []                      # per frame: the alpha on the frame, and where the label map says background
        anchored, unlabelled = [], []
        for j, (f, box) in enumerate(zip(index, boxes)):
            shape = (box[3] - box[1], box[2] - box[0])
            scores = F.interpolate(logits[j:j + 1].to(torch.float32), size=shape, mode="bilinear", align_corners=False)
            label = paste_back(scores[0].argmax(dim=0).to(torch.uint8).cpu(), box, height, width)
            on_subject = wide[j]
            chosen = torch.isin(label.to(torch.long), wanted) & on_subject
            out.seen[f] = torch.bincount(label[on_subject].to(torch.long), minlength=len(CLASS_NAMES))
            out.parts[f] = chosen.to(torch.float32)
            out.found[f] = bool(chosen.any())
            out.boxes[f] = box
            anchored.append(torch.isin(label.to(torch.long), anchors) & on_subject)
            unlabelled.append((label == BACKGROUND) & (subject[f] > 0.5))
            if f in keep:
                out.labels[f] = label
            if alphas is not None:
                alpha = F.interpolate(alphas[j:j + 1].to(torch.float32), size=shape, mode="bilinear",
                                      align_corners=False)[0, 0].clamp(0.0, 1.0).cpu()
                soft.append((paste_back(alpha, box, height, width), (label == BACKGROUND) & on_subject))
            else:
                out.matte[f] = out.parts[f]
        near_anchor = grow(torch.stack(anchored).to(torch.float32), int(held_near)) > 0.5
        for j, f in enumerate(index):
            thing = unlabelled[j] & near_anchor[j]
            if int(thing.sum()) >= int(held_smallest):
                out.holds[f] = thing.to(torch.float32)
        if alphas is not None:
            # one dilation for the batch: measured on CPU, 2026-10-05, a frame at a time it cost about three
            # times as much per frame
            near = grow(out.parts[index], int(matte_reach)) > 0.5
            for j, f in enumerate(index):
                alpha, background = soft[j]
                region = (out.parts[f] > 0.5) | (near[j] & background)
                out.matte[f] = alpha * region.to(torch.float32)
    out.seconds = time.perf_counter() - began
    if where and not bool(out.found.any()):
        raise ValueError(
            "Sapiens2 found none of " + ", ".join(CLASS_NAMES[c] for c in classes) + " on the subject in any frame: "
            "choose other parts, or check that `subject_mask` is the mask of these frames' subject")
    if hold_missing:
        have = out.found.nonzero().flatten()
        for f in (present & ~out.found).nonzero().flatten().tolist():
            g = int(have[(have - f).abs().argmin()])
            src = _widened(tight[g], int(subject_margin), height, width)
            dst = _widened(tight[f], int(subject_margin), height, width)
            on_subject = grow(subject[f:f + 1].to(torch.float32), int(subject_margin))[0]
            out.parts[f] = hold(out.parts[g], src, dst, hard=True) * (on_subject > 0.5)
            out.matte[f] = hold(out.matte[g], src, dst, hard=False) * (on_subject > 0.5)
            out.held.append(f)
    # of what is returned, after the hold. `seen` is counted on the widened subject, so the labelled share is too.
    counts = out.seen.to(torch.float64)
    out.coverage = coverage(subject, out.parts,
                            labelled=counts[:, BACKGROUND + 1:].sum(dim=1) / counts.sum(dim=1).clamp(min=1.0))
    return out


# -------------------------------------------------- the report and the preview

def report(found: Found, classes: tuple[int, ...], crop_margin: int, subject_margin: int, matte_reach: int,
           hold_missing: bool, size: tuple[int, int], matting_name: str | None) -> str:
    n = int(found.present.shape[0])
    present = int(found.present.sum())
    lines = [f"Sapiens2 parts: the subject is on {present} of {n} frames"
             + (f", {found.seconds / present:.2f} s per frame" if present else "")]
    lines.append("classes taken: " + ", ".join(CLASS_NAMES[c] for c in classes))
    lines.append(f"the models were shown the subject alone: the picture further than "
                 f"{max(ALONE_MARGIN, int(subject_margin))} px from the tracked subject's mask was replaced by flat "
                 "colour, so nobody else is there to label")
    missed = (found.present & ~found.found).nonzero().flatten().tolist()
    if not present:
        lines.append("nothing to find: `subject_mask` is empty on every frame")
    elif not missed:
        lines.append("a taken class was found on every frame the subject is in (one pixel counts; the coverage "
                     "lines say how much)")
    elif hold_missing:
        lines.append(f"not found on {len(missed)} frames, which took the nearest found frame's part: {ranges(found.held)}")
    else:
        lines.append(f"not found on {len(missed)} frames, left empty (hold_missing is off): {ranges(missed)}")
    on_frames = (found.seen > 0).sum(dim=0).tolist()
    seen = [f"{CLASS_NAMES[c]} {on_frames[c]}" for c in sorted(range(1, len(CLASS_NAMES)), key=lambda c: -on_frames[c])
            if on_frames[c]]
    lines.append("classes seen on the subject, by frames: " + (", ".join(seen) if seen else "none"))
    if found.coverage is not None and present:
        lines.extend(summarise(found.coverage).lines())
    if found.holds is not None and present:
        sizes = found.holds.flatten(1).sum(dim=1)
        on = sizes > 0
        lines.append(f"held: something attached to the subject that is not body or clothing, near the lips or a "
                     f"hand, on {int(on.sum())} of {present} frames"
                     + (f" (median {int(sizes[on].median())} px, frames {ranges(on.nonzero().flatten().tolist())})"
                        if bool(on.any()) else ""))
    if found.boxes:
        tall = [b[3] - b[1] for b in found.boxes.values()]
        lines.append(f"crop: margin {int(crop_margin)} px, {min(tall)} to {max(tall)} px tall before the resize to "
                     f"{size[1]}x{size[0]}; a label counts on the subject's mask widened by {int(subject_margin)} px")
    lines.append(f"matte: alpha from {matting_name}, on the part and on background within {int(matte_reach)} px of it"
                 if matting_name else "matte: the part mask (no matting model is loaded)")
    return "\n".join(lines)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an older Pillow has one size
        return ImageFont.load_default()


def preview_frames(present: torch.Tensor, most: int = PREVIEW_TILES) -> tuple[int, ...]:
    """Up to `most` of the frames the subject is in, spread evenly over them."""
    where = present.nonzero().flatten()
    if not where.numel():
        return ()
    pick = torch.linspace(0, where.numel() - 1, min(int(most), int(where.numel()))).round().to(torch.long)
    return tuple(sorted({int(where[p]) for p in pick}))


def preview(frames: torch.Tensor, found: Found, size: tuple[int, int]) -> torch.Tensor:
    """The label map over the crop for each kept frame, side by side, with the classes present named below. [1, h, w, 3].

    Each class has its colour; the chosen part is outlined in white.
    """
    tile = (TILE_HEIGHT, max(2, round(TILE_HEIGHT * size[1] / size[0])))
    palette = torch.tensor(PALETTE, dtype=torch.float32) / 255.0
    tiles, shown = [], set()
    for f in sorted(found.labels):
        label, box = found.labels[f].to(torch.long), found.boxes[f]
        img = frames[f, ..., :3].to(torch.float32).cpu()
        tint = (label != BACKGROUND).to(torch.float32).unsqueeze(-1) * 0.6
        img = img * (1.0 - tint) + palette[label] * tint
        part = found.parts[f][None, None]
        edge = (part - (-F.max_pool2d(-part, 5, stride=1, padding=2)))[0, 0] > 0.5
        img[edge] = 1.0
        small = take_crops(img[None], [box], tile, (0.0, 0.0, 0.0))[0].movedim(0, -1)
        pil = Image.fromarray((small.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        draw, font = ImageDraw.Draw(pil), _font(20)
        text = f"frame {f}" + (" (held)" if f in found.held else "")
        where = draw.textbbox((8, 6), text, font=font)
        draw.rectangle((where[0] - 6, where[1] - 4, where[2] + 6, where[3] + 4), fill=(0, 0, 0))
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        tiles.append(np.asarray(pil).copy())
        shown.update(int(c) for c in label.unique().tolist() if c != BACKGROUND)
    if not tiles:
        return torch.zeros((1, tile[0], tile[1], 3))
    row = np.concatenate(tiles, axis=1)
    font = _font(18)
    legend = Image.new("RGB", (row.shape[1], 10), (0, 0, 0))
    x, y, line = 8, 8, 28
    places = []
    for c in sorted(shown):
        wide = int(ImageDraw.Draw(legend).textlength(CLASS_NAMES[c], font=font)) + 44
        if x + wide > row.shape[1] and x > 8:
            x, y = 8, y + line
        places.append((c, x, y))
        x += wide
    legend = Image.new("RGB", (row.shape[1], y + line + 4), (0, 0, 0))
    draw = ImageDraw.Draw(legend)
    for c, x, y in places:
        draw.rectangle((x, y + 2, x + 18, y + 20), fill=PALETTE[c])
        draw.text((x + 24, y), CLASS_NAMES[c], fill=(255, 255, 255), font=font)
    sheet = np.concatenate([row, np.asarray(legend)], axis=0)
    return torch.from_numpy(sheet.copy()).to(torch.float32)[None] / 255.0


# ---------------------------------------------------------------- the models

class _Held(torch.nn.Module):
    """A transformers model as a plain module. Core's patcher writes `.device` on the module it holds
    (`comfy/model_patcher.py::ModelPatcher.load`), and on a transformers model `device` is a read-only property."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net


def _roots() -> list[str]:
    """Every `sapiens2` model folder core knows, with `models/sapiens2/` registered on first use: core has no such type."""
    import folder_paths
    if FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=True)
    return folder_paths.get_folder_paths(FOLDER)


def list_models(arch: str, roots: list[str] | None = None) -> list[str]:
    """The sub-folders of `models/sapiens2/` whose `config.json` names `arch`, by folder name."""
    roots = _roots() if roots is None else roots
    names = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            try:
                with open(os.path.join(root, name, "config.json"), encoding="utf-8") as handle:
                    if arch in (json.load(handle).get("architectures") or []):
                        names.append(name)
            except (OSError, ValueError):
                continue
    return names


def model_folder(name: str, roots: list[str] | None = None) -> str:
    for root in (_roots() if roots is None else roots):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    raise ValueError(f"no Sapiens2 model folder named {name!r} under models/{FOLDER}/")


def working_size(folder: str) -> tuple[tuple[int, int], tuple, tuple]:
    """The size (height, width), mean and std a model folder's own `preprocessor_config.json` gives."""
    with open(os.path.join(folder, "preprocessor_config.json"), encoding="utf-8") as handle:
        pre = json.load(handle)
    return ((int(pre["size"]["height"]), int(pre["size"]["width"])), tuple(pre["image_mean"]), tuple(pre["image_std"]))


def _patcher(net: torch.nn.Module):
    import comfy.model_management as mm
    import comfy.model_patcher
    # The plain patcher and a full load: these modules have none of core's
    # cast-on-use weights, so a partly loaded one would run across two devices.
    return comfy.model_patcher.ModelPatcher(_Held(net.eval()), load_device=mm.get_torch_device(),
                                            offload_device=mm.unet_offload_device())


def _runner(patcher, read: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """A patcher's forward as a callable on normalised crops: `read` is the output field, `logits` or `alphas`."""
    device = patcher.load_device

    def run(crops: torch.Tensor) -> torch.Tensor:
        # bf16 on a card. Inherited: upstream's own inference setting for both
        # tasks (`mixed_precision="bf16"` in the 1b seg and matting configs
        # under `coderef/sapiens2/sapiens/dense/configs/`).
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return getattr(patcher.model.net(pixel_values=crops.to(device)), read).to(torch.float32)
    return run


class MiniMaxH3Sapiens2Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Sapiens2Loader",
            display_name="MiniMax H3 Sapiens2 Loader (parts and matte)",
            category="model/latent/minimax",
            description=(
                "Loads Sapiens2's body-part segmentation model, and its matting model if one is chosen, "
                "from models/sapiens2/. Each model is a folder as Hugging Face lays it out. Wire the "
                "output into a Subject Parts node."),
            inputs=[
                io.Combo.Input("segmentation", options=list_models(SEG_ARCH),
                               tooltip="The folder under models/sapiens2/ that holds the body-part segmentation model."),
                io.Combo.Input("matting", options=[NO_MATTING] + list_models(MATTE_ARCH), default=NO_MATTING,
                               tooltip=("The folder that holds the matting model, which gives the Subject Parts "
                                        "node's `matte` a soft edge. `none` loads no matting model and the "
                                        "matte is then the hard part mask; it saves one model pass per frame.")),
            ],
            outputs=[H3Sapiens2.Output(display_name="sapiens2")],
        )

    @classmethod
    def execute(cls, segmentation, matting=NO_MATTING) -> io.NodeOutput:
        from transformers import Sapiens2ForImageMatting, Sapiens2ForSemanticSegmentation
        folder = model_folder(segmentation)
        size, mean, std = working_size(folder)
        net = Sapiens2ForSemanticSegmentation.from_pretrained(folder, local_files_only=True)
        if int(net.config.num_labels) != len(CLASS_NAMES):
            raise ValueError(
                f"{segmentation} has {int(net.config.num_labels)} classes and this node knows the "
                f"{len(CLASS_NAMES)}-class map: it would name the parts wrongly")
        held = {"seg": _patcher(net), "matting": None, "size": size, "mean": mean, "std": std,
                "seg_name": segmentation, "matting_name": None}
        if matting != NO_MATTING:
            matte_folder = model_folder(matting)
            if working_size(matte_folder) != (size, mean, std):
                raise ValueError(f"{matting} and {segmentation} do not share a working size and normalisation: "
                                 "the two models are run on the same crops")
            held["matting"] = _patcher(Sapiens2ForImageMatting.from_pretrained(matte_folder, local_files_only=True))
            held["matting_name"] = matting
        logger.info("[h3] MiniMaxH3Sapiens2Loader: %s%s, working size %dx%d", segmentation,
                    f" and {matting}" if held["matting"] is not None else ", no matting model", size[1], size[0])
        return io.NodeOutput(held)


class MiniMaxH3SubjectParts(io.ComfyNode):
    #: Part of a kept mask's key when this node sits upstream of a Masked
    #: Source (`mask_store.mask_versions`). Raise it when the same inputs and
    #: settings would give a different mask: the crop, the map back, the cut to
    #: the subject, the menu's classes, the matte's region, how a frame is held.
    MASK_VERSION = 2

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SubjectParts",
            display_name="MiniMax H3 Subject Parts (Sapiens2)",
            category="model/latent/minimax",
            description=(
                "Finds body parts on one tracked person, per frame: hair, face, clothing, hands, mouth. "
                "Gives a mask of the chosen parts and a soft-edged matte of them. The preview shows the "
                "label map on a few frames and the report says what was found."),
            inputs=[
                H3Sapiens2.Input("sapiens2", tooltip="The models, from the Sapiens2 Loader node."),
                io.Image.Input("frames", tooltip="The source video's frames, the same ones the subject was tracked on."),
                io.Mask.Input("subject_mask", tooltip=("One mask per frame, 1 on the person whose parts are wanted, "
                                                       "empty where they are not on screen: the Subject Track "
                                                       "node's `mask`.")),
                io.Boolean.Input("hair", default="hair" in PARTS_ON, tooltip="Take the hair."),
                io.Boolean.Input("face_and_neck", default="face_and_neck" in PARTS_ON,
                                 tooltip="Take the face and neck, with glasses and the mouth."),
                io.Boolean.Input("upper_clothing", default="upper_clothing" in PARTS_ON,
                                 tooltip="Take what is worn on the upper body."),
                io.Boolean.Input("lower_clothing", default="lower_clothing" in PARTS_ON,
                                 tooltip="Take what is worn on the lower body."),
                io.Boolean.Input("hands", default="hands" in PARTS_ON, tooltip="Take both hands."),
                io.Boolean.Input("mouth", default="mouth" in PARTS_ON,
                                 tooltip="Take the lips, teeth and tongue, without the rest of the face."),
                io.String.Input("other_classes", default="",
                                tooltip=("More classes to take, by name, separated by commas. The names are: "
                                         + ", ".join(CLASS_NAMES[1:]) + ".")),
                io.Int.Input("crop_margin", default=CROP_MARGIN, min=0, max=1024,
                             tooltip=("How far past the subject's mask the picture given to the model reaches, in "
                                      "pixels of the source frame. Raise it when the preview shows hair or a "
                                      "hand cut off at the crop's edge.")),
                io.Int.Input("subject_margin", default=SUBJECT_MARGIN, min=0, max=256,
                             tooltip=("How far past the subject's mask a part may reach and still count as the "
                                      "subject's, in pixels of the source frame. Raise it when `parts` is "
                                      "clipped at the subject's edge; lower it when it takes a neighbour's.")),
                io.Int.Input("matte_reach", default=MATTE_REACH, min=0, max=256,
                             tooltip=("How far outside the part the matte's soft edge may extend, in pixels of "
                                      "the source frame. Raise it when fine hair is cut short in `matte`. Read "
                                      "only when a matting model is loaded.")),
                io.Boolean.Input("hold_missing", default=True,
                                 tooltip=("On: a frame where the subject is on screen and none of the chosen parts "
                                          "is found on them takes the part of the nearest frame that has one, "
                                          "moved to where the subject is. Off: that frame's mask is left empty.")),
                # appended 2026-10-07 with the `held` output; saved graphs keep running
                io.Int.Input("held_near", default=HELD_NEAR, min=1, max=512, optional=True,
                             tooltip=("For the `held` output: how far from the lips or a hand a held thing may "
                                      "reach, in pixels of the source frame. Raise it when a long thing is cut "
                                      "short; lower it when `held` takes in a bag or a strap.")),
                io.Int.Input("held_smallest", default=HELD_SMALLEST, min=1, max=65536, optional=True,
                             tooltip=("For the `held` output: a frame with fewer pixels than this is left empty. "
                                      "Raise it when `held` flickers on with specks at the subject's edge.")),
            ],
            outputs=[
                io.Mask.Output(display_name="parts", tooltip="One mask per frame, 1 on the chosen parts of the subject."),
                io.Mask.Output(display_name="matte",
                               tooltip=("The same parts with the matting model's soft edge; the same as `parts` "
                                        "when no matting model is loaded.")),
                io.Image.Output(display_name="preview", tooltip="The label map on a few frames, with the classes named."),
                io.String.Output(display_name="report"),
                io.Mask.Output(display_name="held",
                               tooltip=("One mask per frame of what the subject holds: inside their mask, not body "
                                        "or clothing by the part model, near the lips or a hand. No name is asked "
                                        "for it. Wire it to the Masked Source's `keep` so a cigarette, a cone or a "
                                        "microphone stays the original's while the person is replaced. Empty on a "
                                        "frame where nothing is held.")),
            ],
        )

    @classmethod
    def execute(cls, sapiens2, frames, subject_mask, hair="hair" in PARTS_ON,
                face_and_neck="face_and_neck" in PARTS_ON, upper_clothing="upper_clothing" in PARTS_ON,
                lower_clothing="lower_clothing" in PARTS_ON, hands="hands" in PARTS_ON, mouth="mouth" in PARTS_ON,
                other_classes="", crop_margin=CROP_MARGIN, subject_margin=SUBJECT_MARGIN, matte_reach=MATTE_REACH,
                hold_missing=True, held_near=HELD_NEAR, held_smallest=HELD_SMALLEST) -> io.NodeOutput:
        mask = check_inputs(frames, subject_mask)
        classes = chosen_classes({"hair": hair, "face_and_neck": face_and_neck, "upper_clothing": upper_clothing,
                                  "lower_clothing": lower_clothing, "hands": hands, "mouth": mouth}, other_classes)
        import comfy.model_management as mm
        models = [p for p in (sapiens2["seg"], sapiens2["matting"]) if p is not None]
        mm.load_models_gpu(models, force_full_load=True)
        seg = _runner(sapiens2["seg"], "logits")
        matting = _runner(sapiens2["matting"], "alphas") if sapiens2["matting"] is not None else None
        size = tuple(sapiens2["size"])
        present = mask_boxes(mask)[:, 0] >= 0
        found = subject_parts(frames, mask, classes, seg, matting, size=size, mean=sapiens2["mean"],
                              std=sapiens2["std"], crop_margin=int(crop_margin), subject_margin=int(subject_margin),
                              matte_reach=int(matte_reach), hold_missing=bool(hold_missing),
                              keep=preview_frames(present), held_near=int(held_near),
                              held_smallest=int(held_smallest))
        text = report(found, classes, crop_margin, subject_margin, matte_reach, bool(hold_missing), size,
                      sapiens2["matting_name"])
        logger.info("[h3] MiniMaxH3SubjectParts: %s", text.replace("\n", "; "))
        sheet = preview(frames, found, size)
        shown = {**ui.PreviewImage(sheet, cls=cls).as_dict(), **ui.PreviewText(text).as_dict()}
        return io.NodeOutput(found.parts, found.matte, sheet, text, found.holds, ui=shown)


def check_inputs(frames: torch.Tensor, subject_mask: torch.Tensor) -> torch.Tensor:
    """The subject mask as [N, H, W], refused by name when it is not the mask of these frames."""
    if frames.ndim != 4:
        raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
    mask = subject_mask
    if mask.ndim == 4 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"subject_mask must be [N, H, W]; got {tuple(subject_mask.shape)}")
    if int(mask.shape[0]) != int(frames.shape[0]):
        raise ValueError(
            f"subject_mask has {int(mask.shape[0])} frames and frames has {int(frames.shape[0])}: the mask "
            "must be tracked over the same frames")
    if tuple(mask.shape[1:]) != tuple(frames.shape[1:3]):
        raise ValueError(
            f"subject_mask is {int(mask.shape[2])}x{int(mask.shape[1])} and frames is "
            f"{int(frames.shape[2])}x{int(frames.shape[1])}: track the mask on these frames")
    return mask
