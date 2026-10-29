#!/usr/bin/env python3
"""The masked-source reduction and composite, and the ways each could keep the old subject.

`video_mask.py` is the module; `audio_freeze_song.py` uses it. Masked
video-to-video fails quietly: a subject pixel the mask loses is a piece of the
original left in the render, and the clip still plays. Each item is one way
that could happen.

1. **No subject frame is dropped in time.** A subject present in one frame of
   a run must regenerate that run's latent step. The control is core's own
   resize (`comfy.utils.reshape_mask`, trilinear) on the same mask, which
   must lose it, or this module is not buying what it says.
2. **The mask is already on core's token grid.** `token_mask`'s output run
   through `comfy/ldm/minimax/model.py::mask_row_values` comes back unchanged,
   so the region this module reports and composites is the region the model
   regenerates.
3. **The feather stays off the subject.** With `grow_pixels >= feather_pixels`
   the blend weight is exactly 1 on every original subject pixel, and exactly
   0 further than the feather from a regenerated token. The node refuses the
   other ordering.
4. **The composite returns the source where nothing regenerates**, bit for
   bit, and the render where the weight is 1.
5. **A window that outruns the source holds its last frame unmasked**, and a
   window that starts past the source is refused.
6. **The mask is cropped as the frames are, and loses nothing on the way
   down.** `fit_mask` restates core's centre crop so it can max-pool; a mask
   and its frames fitted from the same odd shape must still coincide, a
   one-pixel line must survive a large downscale (the control: core's
   bilinear resize of the same mask drops it below one half), and the node
   refuses a mask whose shape is not its frames'.
7. **The paint-out hides the subject from the encode and nothing else.**
   With `paint_out` the frames to encode differ from the fitted frames only
   under the hole, the hole covers every subject pixel, no filled pixel lies
   outside the regenerated tokens (so none can be shown), and a bright
   subject on a flat ground is gone from the encode frames. Off, the encode
   frames are the fitted frames.
8. **A part is taken only from the subject, and a missed frame is never
   left showing the original.** `select_part` keeps a part where it lies on
   the subject's mask and drops the same part on a neighbour. The region is
   the subject down to the part's lowest row (`above`), so it holds the
   subject's own pixels above that row and none below. A frame with no part
   takes the nearest found frame's row and is counted (`carry_missing`); with
   nothing found anywhere nothing is invented. The node refuses a part with
   no segmenter wired, an empty phrase list and an unknown choice, and its
   second output is the mask it used. Its third is the preview strip: sampled
   frames with the region tinted, the motion reference beside them when on.
   `the wired parts` (2026-10-05) regenerates exactly the part node's mask on
   `parts`, kept to the subject, asks for nothing else lazily, and refuses an
   unwired or empty one.
9. **`only what changed` keeps a subject and restores the margin.** With a
   render that equals the source except where the new subject is, the weight
   is 1 on the new subject and on every old-subject pixel inside the
   regenerated tokens, 0 in margin further than the feather and the
   difference's own blur from either, never above the whole-region weight,
   and 0 everywhere when the render repainted the source faithfully (a
   tracker's false positive). A difference under the ramp's foot is not
   kept. The node refuses an unknown choice.
10. **Every graph that wires a Masked Source wires it whole**: into a song
   node's `source`, its mask tracked over the same frames it carries, and the
   song node's track taken from the same loader as those frames.
12. **The mask review shows what regenerates, and nothing else is touched.**
   `overlay_pieces` over a window with a known subject: every frame from the
   trim on, a pixel of the subject in the subject's colour and only that, a
   pixel of the regenerated margin in the region's, a pixel far outside the
   source's own; the region is the token mask the sampler was given, piece
   for piece; an outline only with the composite's weight; the legend names
   every layer and sits in the bottom-left corner; a layer added to the list
   is drawn and named with no other change; and `render_over` puts the same
   frames of the render above. The song node's `save_mask_review` is its last
   input, optional and on, is left out of a stored window's key, and every
   shipped graph that wires a source writes it.
11. **The loader loads the frames the song node's plan reads, and no more.**
   The tracker works on every frame the loader hands it, so a cap past what
   the plan reads for the graph's extent, window and context is tracking
   nothing renders (`loop_plan.frames_read`; a spare window of it until
   2026-10-06). A cap under it would hold the last frame where the plan
   expects picture. The count is the planner's own, not restated here.

13. **A margin taken from the subject's size holds the region to the
   subject, and a fixed one is untouched.** `grow_by` (2026-10-08). Under `a
   fixed margin` the margin is the number `grow_pixels` and a per-frame
   margin of that number grows the same mask, bit for bit. Under `the
   subject's size`, on a subject that shrinks through a clip: the region on
   the close frames is the fixed margin's; on the small frames the region
   over the subject's own area is under `GROW_REGION_BOUND`, where the fixed
   margin's is over it (the control, or the bound proves nothing); the margin
   never passes `grow_pixels` and never goes under `feather_pixels`; a run of
   frames on which the mask collapses to a sliver moves the margin by no more
   than `GROW_STEP`, where the same rule with no steadying drops it to the
   floor (the second control); a window's tokens under it lie inside the
   fixed margin's; a window reads its own frames' margins and a window past
   the source's end takes the cap; on a canvas of another size than the
   frames, from a start where the margins are not the clip's first, the
   tokens, the paint-out's hole and the late start's softened hole and
   emptied tokens are all the canvas's margins on the window's own frames,
   and `start_zero_tokens` cannot be called without the window's start; the
   floor is the feather; the song node hands the composite the window's own
   margin; the motion reference's widening stays half
   of `grow_pixels` under either choice; and the node carries the choice and the
   area on its record, refuses an unknown choice, keeps it out of the kept
   mask's key, and the shared config holds the default.

14. **The margin stays off the people round the subject, and the subject
   loses nothing.** `others` (2026-10-08). With a neighbour standing against
   the subject: no token that holds the subject's own mask is given up; no
   token that holds the neighbour and none of the subject regenerates; every
   other token is as it was. The control is the same mask on `keep`, which
   must cost the subject tokens, or the two inputs are not told apart. A mask
   that also covers the mask the region is grown from changes nothing (which
   is the whole subject only when that mask is); `keep` still wins where both
   are wired; a neighbour on one frame acts on that latent step only; the
   node refuses another frame count or size and the two settings `keep`
   refuses, by name; unwired nothing changes; the preview shows them.

15. **The finer edge loses no subject pixel and changes no label.** `edge`
   (2026-10-09). With `latent cells` the mask is left per latent cell: every
   masked pixel is still under a regenerated cell on its run; the cells are
   inside the tokens the default makes, and fewer on a mask of single pixels
   (the control: a mask that already fills whole tokens gives the same
   answer both ways, so the saving comes from the rounding and nothing
   else); core's own pooling of the cell mask gives the rows the default's
   tokens give, so the model's labels do not move and only what is put back
   does. `window` reads the choice from the record, an unset record is the
   default bit for bit, and the node refuses an unknown choice.
16. **A setting the node will refuse is refused when the graph is queued.**
    (2026-10-10.) `settings_refusal` names a feather wider than the margin and
    a softened start with `paint_out`, and passes the shipped defaults; the
    node's `validate_inputs` returns that same message for the same values,
    True for the defaults, True for a value that is a link and so not known
    yet, and a message for a margin, a feather or a `start_from` outside what
    the schema allows (core stops testing an input a validation function
    names, so the range is tested there). The control: `execute` still raises
    the same message, for a value that only arrives through a link.
17. **A wired motion video is cut where the window is cut.** `motion_video`
    (2026-10-10): with `a video I wire`, a window that starts at source frame
    k is shown the wired video from frame k, fitted to the canvas and scaled
    to the short edge asked, and a window that runs past the video's end
    repeats its last frame. The control: a second window's reference is not
    the first window's, and equals the wired video's later frames, so the
    cut is by the window and not from frame zero (which is what a reference
    video appended to the chain gets). The choice with nothing wired, a video
    wired under another choice, and a video of another length are each
    refused by name; the preview strip shows the wired video beside the
    plate; `motion_video` is where it was appended and optional; and the song
    node takes the wired branch. **A second wired video** (2026-10-12) is
    carried with its own short edge, is cut by the window as the first is and
    leaves the first as it was; it widens the preview strip; one with no
    first, or of another length, is refused by name; its two inputs are on
    the end and optional; the static label plan gives the two videos
    `<Video 1>` and `<Video 2>` in order. **What a reference costs**
    (`audio_freeze_song.reference_costs`): a line a reference, named in
    order, with the tokens and rows the pricing module gives for it, "no copy
    for the video model" where there is none, a last line that is their sum,
    nothing for no references, and one line and no error when the count fails.
18. **A tracked mask becomes one box a frame, and none where the subject is
    not.** `subject_boxes.frame_boxes` (2026-10-10), for a node that takes
    boxes: the box is the mask's own bounds, widened by the margin and held
    inside the frame; a frame with an empty mask gets an empty list, never the
    whole frame (the control: core's own fallback for an empty mask is the
    whole frame, which is the body drawn where nobody is); and the list has
    the shape core's SAM 3D Body prediction reads, checked through core's own
    reader when it imports.
19. **The composite lays nothing across a cut from the subject.** One latent
    step is a run of frames and the region is one per step, so a run that
    straddles a cut carries the subject's region onto the other shot's
    frames (2026-10-10: a whole-subject pass repainted a frame or two of the
    next shot at four cuts). `cut_gate` is 0 on the frames of such a run
    that lie on a side of the cut the subject is on no frame of, and 1
    everywhere else. The controls: a frame the tracker lost with no cut
    beside it stays 1 (it is covered by its run, as before); a cut with the
    subject on both sides stays 1; a cut that falls on a run's edge splits
    nothing; a run the subject is on no frame of is left alone; and the same
    cut named in the source's frames is found through `first_frame`.
    `source_cuts` reads the cuts from a wired shot table and otherwise from
    the Subject Track's own detector, which finds a cut made here and none
    in a held shot; the source record carries them, and the song node gates
    the weight before it composites.
20. **A window is laid by one function, and a render saves what it was laid
    with.** `lay_window` is the song node's composite (the weight, the cut
    gate, the blend) and the song node calls nothing else for it: under
    `only what changed` and under `whole region` its frames are the pieces'
    own answer, the weight it returns is the weight it laid, a gated frame
    is the source bit for bit and no frame without a cut beside it moves,
    and its lines are the report's. `save_window_region` and
    `load_window_region` (2026-10-10: a render whose tracker ran in its own
    graph left its mask nowhere, so a composite could not be run again on
    it): the mask, the token region, the margin (one number, or each
    frame's own), the window's first frame, its trim and the settings read
    back as written, and a window laid from the file is the window laid
    from the tensors, bit for bit; a soft mask is saved as the 0 or 1 every
    reader makes of it. The song node writes the file beside the window's
    latent, removes a stale one with the stale latent, and counts no reused
    window (08c3cb12 shadowed that count inside the loop; the count went with
    stored-window reuse on 2026-10-10).
21. **The song node's loop, run.** Every item above reads a function or the
    node's source; none ran the loop, and a variable shadowed inside it
    shipped (08c3cb12). Here `MiniMaxH3AudioFreezeSong.execute` runs whole
    on a canvas of a few latent cells: two windows with context over a real
    Masked Source record with two cuts, each inside a latent step, and a
    subject on the first and third shots. The stand-ins are the model's
    side only: a sampler that marks the cells it was asked to regenerate, a
    video VAE whose decode is the window's own source with those cells
    painted bright, an audio VAE of the right shapes, a conditioning node
    that returns a token. The windows, the plan, the Masked Source, the
    composite, the writes and the join are the node's own, through ffmpeg.
    The report opens with the render's path and says which frames lie across
    a cut, and they are the frames `loop_plan.split_steps` names for the
    load. In the joined video those frames are the source's and a frame the
    subject is on carries the bright cells. Each window leaves a region
    file that reads back as `window` makes the window's mask and tokens,
    with its first frame and trim; a stale one from an earlier run is
    replaced; and with `keep_windows` off the folder is gone. A preview
    of the same graph samples nothing, asks for the source alone, and
    writes a planned region for each window that is the region the render
    saved, and a plan file whose written frames, trims, split steps and
    frames left as the source are the render's; it removes an earlier
    plan's file and touches none of the render's; on a continued run
    (`continue_from`) the plan's trim is the context, as the render's is,
    for the first window too. With
    `context_noise` the mask the second window's sampler is handed carries
    the value on the context's region cells, 0 on the context's plate, and
    is the default run's everywhere else.

24. **What the context frames show** (`context_shows`, 2026-10-22). A short
   load's padded frames carry an empty mask `as filmed` and, `face covered`,
   the shot's first frame's mask before it and its last frame's after it,
   the keep-outs empty either way; the steps wholly before the shot then
   regenerate; a load that fills its window is the same under both; the
   record, the region file and the plan's row say which form it took and
   which held frames lie across a cut, and the plan's mask frames stay the
   shot's.

No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_video_mask.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
WORKFLOWS = REPO / "workflows"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WORKFLOWS))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context for a shape check; the sibling checks do the same
import comfy.utils  # noqa: E402
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, mask_row_values  # noqa: E402
import h3_config  # noqa: E402


def check_keep(problems):
    """The optional `keep` mask (2026-10-07): it wins over the region after the grow, in whole tokens, and unwired
    nothing changes. The control: the same keep applied BEFORE the grow, which the grow then runs back over."""
    frames = torch.rand(FRAMES, 72, 128, 3)
    mask = torch.zeros(FRAMES, 72, 128)
    mask[:, 20:44, 40:72] = 1.0
    keep = torch.zeros(FRAMES, 72, 128)
    keep[:, 30:34, 50:54] = 1.0                       # a small thing inside the subject, on every frame
    base = {"frames": frames, "mask": mask, "grow_pixels": 8, "feather_pixels": 0}
    plain = vm.window(base, 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    none = vm.window(dict(base, keep=None), 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    if not torch.equal(plain, none):
        problems.append("keep: an unwired keep changed the token mask")
    held = vm.token_mask(vm.fit_mask(keep, W, H), LATENT_T, LAT_H, LAT_W)
    got = vm.window(dict(base, keep=keep), 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    if float(held.sum()) == 0.0 or float((plain * held).sum()) == 0.0:
        problems.append("keep: the case does not put the kept thing inside the region; it tests nothing")
    if float((got * held).max()) != 0.0:
        problems.append("keep: a token the keep mask touches still regenerates")
    if not torch.equal(got, plain * (1.0 - held)):
        problems.append("keep: tokens the keep mask does not touch changed")
    if float(got.sum()) == 0.0:
        problems.append("keep: nothing regenerates with a small keep inside a large region")
    # the control: subtracted before the grow, the 8 px margin covers a 4 px hole again and nothing is kept
    before = vm.token_mask(vm.grow(vm.fit_mask(mask * (1.0 - keep), W, H), 8), LATENT_T, LAT_H, LAT_W)
    if float((before * held).max()) == 0.0:
        problems.append("keep: the control (keep taken out before the grow) also kept the tokens; the case "
                        "cannot tell after from before")
    # in time: kept on some frames only, a token is kept for the whole latent step that holds a kept frame
    runs = vm.run_lengths(LATENT_T)
    some = torch.zeros_like(keep)
    some[sum(runs[:2]):sum(runs[:2]) + 1] = keep[0]   # one frame, the first of the third latent step
    t = vm.window(dict(base, keep=some), 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    kept_steps = [k for k in range(LATENT_T) if float((plain[k] - t[k]).abs().sum()) > 0.0]
    if kept_steps != [2]:
        problems.append(f"keep on one frame of latent step 2 changed steps {kept_steps}")
    # the node: shape and the two settings that change the pixels under the subject are refused
    for label, kw, bad in (("another size", {}, torch.zeros(FRAMES, 36, 64)),
                           ("paint_out", {"paint_out": True}, keep),
                           ("a softened start", {"start_from": vm.START_TOP}, keep)):
        try:
            vm.MiniMaxH3MaskedSource.execute(frames, mask, keep=bad, **kw)
            problems.append(f"keep: {label} was accepted")
        except ValueError:
            pass
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, keep=keep.unsqueeze(-1))
    src = out.args[0] if hasattr(out, "args") else out[0]
    if src.get("keep") is None or tuple(src["keep"].shape) != tuple(keep.shape):
        problems.append("keep: the node did not carry a [N, H, W, 1] keep mask onto the source as [N, H, W]")


#: Item 13's bound on the pixel region over the subject's own area where the subject is small, before the token
#: grid rounds it out. Reasoned: a square subject of side s with a margin of `GROW_SHARE` of s on every side is
#: (1 + 2 x 0.15) squared, about 1.7 times its area; the feather's floor and the coarse grow's rounding lift that
#: on a subject a few tokens across, and 4 leaves room for both while staying under what the fixed margin gives.
GROW_REGION_BOUND = 4.0
#: How far, in pixels, a collapsed stretch may move the steadied margin on any frame. Reasoned: one quantum of
#: the coarse grow (`GROW_COARSE`), under which two margins dilate alike.
GROW_STEP = 4


def check_grow_by(problems):
    """Item 13. A square subject that is close, shrinks, and stays small, with a stretch where its mask collapses."""
    n, h, w, cap, feather = 150, 192, 320, 16, 2
    side = [120] * 40 + [int(round(120 - (120 - 8) * i / 59)) for i in range(60)] + [8] * 50
    mask = torch.zeros(n, h, w)
    cy, cx = 112, 176                                    # the middle of a token, so a small region can sit in one
    for i, s_ in enumerate(side):
        y0, x0 = cy - s_ // 2, cx - s_ // 2
        mask[i, y0:y0 + s_, x0:x0 + s_] = 1.0
    clean = mask.clone()
    doubt = range(60, 70)                                # the part mask on a sliver, as a part report puts in doubt
    for i in doubt:
        mask[i] = 0.0
        mask[i, cy:cy + 2, cx:cx + 2] = 1.0
    px = h * w
    fixed = vm.margins(vm.area_share(mask), px, cap, feather, vm.GROW_FIXED)
    if fixed != cap or torch.is_tensor(fixed):
        problems.append(f"grow_by: a fixed margin is {fixed!r}, not the number grow_pixels")
    if not torch.equal(vm.grow(mask, torch.full((n,), cap)), vm.grow(mask, cap)):
        problems.append("grow_by: a per-frame margin of grow_pixels on every frame does not grow what the number does")
    mixed = torch.tensor([0, 3, 16, 40] * 2)
    by_frame = torch.stack([vm.grow(clean[i:i + 1], int(p))[0] for i, p in zip(range(36, 44), mixed.tolist())])
    if not torch.equal(vm.grow(clean[36:44], mixed), by_frame):
        problems.append("grow_by: frames grown together by their own margins differ from each grown alone")
    try:
        vm.grow(mask, torch.full((n - 1,), cap))
        problems.append("grow_by: a margin per frame for the wrong number of frames was accepted")
    except ValueError:
        pass

    each = vm.margins(vm.area_share(mask), px, cap, feather, vm.GROW_SUBJECT)
    if not torch.is_tensor(each) or tuple(each.shape) != (n,):
        problems.append(f"grow_by: the subject's size gave {type(each).__name__}, not one margin per frame")
        return
    if int(each.max()) > cap or int(each.min()) < feather:
        problems.append(f"grow_by: margins run {int(each.min())} to {int(each.max())}, outside feather {feather} "
                        f"to grow_pixels {cap}")
    close, small = slice(0, 30), slice(120, 150)
    if not torch.equal(vm.grow(mask[close], each[close]), vm.grow(mask[close], cap)):
        problems.append("grow_by: on the close frames the region is not the fixed margin's")

    def over_subject(grown, frames):
        return float((grown > 0.5).flatten(1).sum(dim=1).float().div(mask[frames].flatten(1).sum(dim=1)).max())
    scaled_ratio = over_subject(vm.grow(mask[small], each[small]), small)
    fixed_ratio = over_subject(vm.grow(mask[small], cap), small)
    if fixed_ratio <= GROW_REGION_BOUND:
        problems.append(f"grow_by: the control (the fixed margin on the small frames) is {fixed_ratio:.1f} times "
                        f"the subject, not over the bound {GROW_REGION_BOUND}; the case tests nothing")
    if scaled_ratio >= GROW_REGION_BOUND:
        problems.append(f"grow_by: on the small frames the region is {scaled_ratio:.1f} times the subject, "
                        f"the bound is {GROW_REGION_BOUND}")
    # steadied: the collapsed stretch moves no frame's margin by more than a quantum; unsteadied it does
    calm = vm.margins(vm.area_share(clean), px, cap, feather, vm.GROW_SUBJECT)
    moved = int((each - calm).abs().max())
    if moved > GROW_STEP:
        problems.append(f"grow_by: ten collapsed frames moved the margin by {moved} px, more than {GROW_STEP}")

    def unsteadied(m):
        return (vm.GROW_SHARE * (vm.steady(vm.area_share(m), 0) * px).sqrt()).ceil().long().clamp(min=feather, max=cap)
    if int((unsteadied(mask) - unsteadied(clean)).abs().max()) <= GROW_STEP:
        problems.append("grow_by: the control (no steadying) was not moved by the collapsed frames either; the "
                        "case cannot tell a steadied margin from a raw one")
    step = int((each[1:] - each[:-1]).abs().max())
    if step > GROW_STEP:
        problems.append(f"grow_by: the margin jumps {step} px between two frames of one shot")
    # a cut: the size steps, and the margin steps on the same frame, not smeared over the window
    cut = torch.cat([vm.area_share(clean[:1]).expand(60), vm.area_share(clean[-1:]).expand(60)])
    m_cut = vm.margins(cut, px, cap, feather, vm.GROW_SUBJECT)
    if len(set(m_cut[:60].tolist())) != 1 or len(set(m_cut[60:].tolist())) != 1 or int(m_cut[59]) == int(m_cut[60]):
        problems.append("grow_by: a step in the subject's size (a cut) is not a step in the margin on that frame")
    # frames with nobody: no margin to take, and they do not drag their neighbours' down
    gap = vm.area_share(clean).clone()
    gap[125:135] = 0.0
    if not torch.equal(vm.margins(gap, px, cap, feather, vm.GROW_SUBJECT)[:125], calm[:125]):
        problems.append("grow_by: frames with no subject changed the margin of the frames before them")

    # a window: its own frames' margins, inside the fixed margin's tokens; past the source's end the cap
    frames = torch.rand(n, h, w, 3)
    base = {"frames": frames, "mask": mask, "grow_pixels": cap, "feather_pixels": feather}
    record = dict(base, grow_by=vm.GROW_SUBJECT, subject_area=vm.area_share(mask))
    lat_t = 7
    span = sum(vm.run_lengths(lat_t))
    for first in (0, 120):
        t_fixed = vm.window(base, first, span, w, h, lat_t, h // 16, w // 16)[2]
        t_scaled = vm.window(record, first, span, w, h, lat_t, h // 16, w // 16)[2]
        if float((t_scaled * (1.0 - t_fixed)).max()) != 0.0:
            problems.append(f"grow_by: a window from frame {first} regenerates a token the fixed margin does not")
        if first == 0 and not torch.equal(t_scaled, t_fixed):
            problems.append("grow_by: a window on the close frames does not regenerate the fixed margin's tokens")
        if first == 120 and float(t_scaled.sum()) >= float(t_fixed.sum()):
            problems.append("grow_by: a window on the small frames regenerates as many tokens as the fixed margin")
    got = vm.source_margins(record, 120, 40, px)
    if tuple(got.shape) != (40,) or not torch.equal(got[:30], each[120:150]) or set(got[30:].tolist()) != {int(each[149])}:
        problems.append("grow_by: a window's margins are not its own frames', with the last frame's held past the source's end")
    if vm.source_margins(base, 120, 40, px) != cap or vm.source_margins({"mask": mask, "grow_pixels": cap}, 0, 4, px) != cap:
        problems.append("grow_by: a record without the choice does not take the fixed margin")
    if vm.margin_note(cap) != f"{cap} px" or vm.margin_note(each) != f"{int(each.min())} to {int(each.max())} px":
        problems.append("grow_by: the report's margin is not the number or the range")
    # A window on a canvas of another size than its frames, from a start where the margins differ from the
    # clip's first frames: every hole and the tokens are the canvas's margins on the window's own frames.
    cw, ch, first = 2 * w, 2 * h, 60
    want = vm.margins(vm.area_share(mask), cw * ch, cap, feather, vm.GROW_SUBJECT)[first:first + span]
    if torch.equal(want, vm.margins(vm.area_share(mask), px, cap, feather, vm.GROW_SUBJECT)[first:first + span]) \
            or torch.equal(want, vm.margins(vm.area_share(mask), cw * ch, cap, feather, vm.GROW_SUBJECT)[:span]) \
            or torch.equal(want // 2, torch.full_like(want, cap // 2)):
        problems.append("grow_by: the canvas case's margins equal the source's, the first frames' or the fixed "
                        "half; it cannot tell them apart")
    fitted = vm.fit_mask(mask[first:first + span], cw, ch)
    shape = (lat_t, ch // 16, cw // 16)
    got = vm.window(record, first, span, cw, ch, *shape)
    if not torch.equal(got[2], vm.token_mask(vm.grow(fitted, want), *shape)):
        problems.append("grow_by: a window's tokens on another canvas are not the canvas's margins on its own frames")
    painted = vm.window(dict(record, paint_out=True), first, span, cw, ch, *shape)
    if not torch.equal(painted[1], vm.fill_subject(painted[0], vm.grow(fitted, want // 2))):
        problems.append("grow_by: the paint-out's hole is not half the window's own margins")
    late = dict(record, start_from=vm.START_TOP, start_top=0.3, start_blur=4, start_knots=1)
    soft = vm.window(late, first, span, cw, ch, *shape)
    if not torch.equal(soft[1], vm.soften_subject(soft[0], vm.grow(fitted, want // 2), 4)):
        problems.append("grow_by: the late start's softened hole is not half the window's own margins")
    hole = vm.grow(fitted, want // 2)
    by_hand = (soft[2] > 0.5).float() * vm.token_mask(hole, *shape) * (1.0 - vm.token_mask(vm.top_of(hole, 0.3), *shape))
    if not torch.equal(vm.start_zero_tokens(late, fitted, soft[2], first), by_hand):
        problems.append("grow_by: the late start's emptied tokens are not from the window's own margins")
    if torch.equal(vm.start_zero_tokens(late, fitted, soft[2], 0), by_hand):
        problems.append("grow_by: the late start's tokens do not depend on where the window starts; the case tests nothing")
    try:
        vm.start_zero_tokens(late, fitted, soft[2])
        problems.append("grow_by: start_zero_tokens ran without being told where its window starts")
    except TypeError:
        pass
    # the floor is the feather, and it is the floor that lifts a small subject's margin
    if int(vm.margins(vm.area_share(clean), px, cap, 6, vm.GROW_SUBJECT).min()) != 6 \
            or int(vm.margins(vm.area_share(clean), px, cap, 0, vm.GROW_SUBJECT).min()) >= 6:
        problems.append("grow_by: the margin's floor is not feather_pixels")
    # the composite's old-subject margin is half the window's own (`lay_window`, item 20), read where the window starts
    song_text = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "margin, int(round(w.start * FPS)))" not in song_text \
            or 'source["feather_pixels"], margin // 2,' not in (REPO / "video_mask.py").read_text(encoding="utf-8") \
            or "margin = video_mask.source_margins(source, int(round(w.start * FPS)), w.frames," not in song_text \
            or "video_mask.start_zero_tokens(source, src_mask, src_tokens, int(round(w.start * FPS)))" not in song_text:
        problems.append("grow_by: the song node does not hand the composite and the late start the window's own margins")
    # the motion reference's widening does not follow the choice: half of grow_pixels under either
    if vm.motion_widening(record) != cap // 2 or vm.motion_widening(base) != cap // 2:
        problems.append("grow_by: the motion reference's widening is not half of grow_pixels under both choices")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "video_mask.motion_widening(source)" not in song:
        problems.append("grow_by: the song node does not take the motion reference's widening from motion_widening")

    # the node, its schema and the shared config
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=cap, feather_pixels=feather, grow_by=vm.GROW_SUBJECT)
    src = out.args[0] if hasattr(out, "args") else out[0]
    if src.get("grow_by") != vm.GROW_SUBJECT or not torch.equal(src.get("subject_area"), vm.area_share(mask)):
        problems.append("grow_by: the node's record does not carry the choice and the mask's area per frame")
    plain = vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=cap, feather_pixels=feather)
    plain = plain.args[0] if hasattr(plain, "args") else plain[0]
    if plain.get("grow_by") != vm.GROW_FIXED or vm.source_margins(plain, 0, n, px) != cap:
        problems.append("grow_by: the node's default is not the fixed margin")
    try:
        vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_by="a guess")
        problems.append("grow_by: an unknown choice was accepted")
    except ValueError:
        pass
    last = [i for i in vm.MiniMaxH3MaskedSource.define_schema().inputs if i.id == "grow_by"]
    if len(last) != 1 or not last[0].optional or last[0].default != vm.GROW_FIXED or list(last[0].options) != list(vm.GROW_BY):
        problems.append("grow_by: it is not an optional input of the node with the fixed margin as its default")
    if h3_config.MASKED_SOURCE.get("grow_by") != vm.GROW_FIXED:
        problems.append("grow_by: h3_config.MASKED_SOURCE does not hold the node's default")


def check_edge(problems):
    torch.manual_seed(0)
    mask = torch.zeros(FRAMES, LAT_H * 16, LAT_W * 16)
    for _ in range(6):
        f, y, x = (int(torch.randint(0, n, (1,))) for n in mask.shape)
        mask[f, y, x] = 1.0
    tokens = vm.token_mask(mask, LATENT_T, LAT_H, LAT_W)
    cells = vm.token_mask(mask, LATENT_T, LAT_H, LAT_W, whole_tokens=False)
    if tuple(cells.shape) != tuple(tokens.shape) or not bool(((cells == 0) | (cells == 1)).all()):
        problems.append(f"the cell mask is {tuple(cells.shape)} with values outside 0 and 1")
        return
    # no subject pixel is lost: each masked pixel sits under a regenerated cell on its run
    at = 0
    for step, run in enumerate(vm.run_lengths(LATENT_T)):
        under = cells[step].repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1)
        if bool((mask[at:at + run].amax(dim=0) > under).any()):
            problems.append(f"latent step {step}: a masked pixel is not under a regenerated cell")
        at += run
    if bool((cells > tokens).any()):
        problems.append("a cell is regenerated outside the token the default regenerates")
    if not float(cells.sum()) < float(tokens.sum()):
        problems.append("on a mask of single pixels the cell edge regenerates no less than whole tokens")
    # the control: a mask that already fills whole tokens is the same both ways
    full = torch.zeros(FRAMES, LAT_H * 16, LAT_W * 16)
    full[:, 32:96, 64:128] = 1.0
    if not torch.equal(vm.token_mask(full, LATENT_T, LAT_H, LAT_W),
                       vm.token_mask(full, LATENT_T, LAT_H, LAT_W, whole_tokens=False)):
        problems.append("a mask that fills whole tokens differs between the two edges: the saving is not the rounding")
    # core labels the same rows from either mask
    rows_cells = mask_row_values(cells, LATENT_T, LAT_H, LAT_W)
    rows_tokens = mask_row_values(tokens, LATENT_T, LAT_H, LAT_W)
    if rows_cells is None or rows_tokens is None or not torch.equal(rows_cells, rows_tokens):
        problems.append("core's pooling labels different rows from the cell mask and from the token mask")
    # the window reads the record; an unset record is the default
    frames = torch.rand(FRAMES, H, W, 3)
    small = torch.zeros(FRAMES, H, W)
    small[:, 20:27, 40:45] = 1.0
    base = {"frames": frames, "mask": small, "grow_pixels": 0, "feather_pixels": 0}
    plain = vm.window(base, 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    named = vm.window(dict(base, edge=vm.EDGE_TOKENS), 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    fine = vm.window(dict(base, edge=vm.EDGE_CELLS), 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)[2]
    if not torch.equal(plain, named):
        problems.append("a record that names `whole tokens` differs from one that names no edge")
    if not torch.equal(plain, vm.token_mask(vm.fit_mask(small, W, H), LATENT_T, LAT_H, LAT_W)):
        problems.append("a record with no edge is not the default token mask")
    if not torch.equal(fine, vm.token_mask(vm.fit_mask(small, W, H), LATENT_T, LAT_H, LAT_W, whole_tokens=False)):
        problems.append("`latent cells` on the record is not the cell mask in the window")
    try:
        vm.MiniMaxH3MaskedSource.execute(frames, small, edge="something else")
        problems.append("an unknown edge was accepted by the node")
    except ValueError as err:
        if "edge" not in str(err):
            problems.append(f"an unknown edge was refused without naming it: {err}")
    # appended inputs keep their place (`APPENDED`)
    got = _appended("edge")
    if got is None or not got.optional or got.default != vm.EDGE_TOKENS:
        problems.append("edge: it is not where it was appended (the input before `motion_video`), optional, "
                        "defaulting to whole tokens")


def check_region_rules(problems):
    """The outside reviewer's G2 of 2026-10-15, ruled by the lead: a keep-out speck under one cell on at most half its step's frames
    does not hole the step (`drop_specks`); a one-cell hole of the keep-outs with the region all round it is closed
    (`close_lone_holes`); a piece of the region the keep-outs cut off from the subject is given back (`drop_islands`). Each at its
    bound, with its control."""
    n, h, w = FRAMES, 128, 192
    frames = torch.rand(n, h, w, 3)
    mask = torch.zeros(n, h, w)
    mask[:, 40:88, 72:104] = 1.0                       # the subject: cells rows 2-5, cols 4-6 of 16 px
    shape = (LATENT_T, h // 16, w // 16)
    args = (0, n, w, h) + shape
    cells = {"frames": frames, "mask": mask, "grow_pixels": 32, "feather_pixels": 0, "edge": vm.EDGE_CELLS}
    plain = vm.window(cells, *args)[2]
    step2 = list(range(sum(vm.run_lengths(LATENT_T)[:2]), sum(vm.run_lengths(LATENT_T)[:3])))   # latent step 2's four frames
    def speck(on, deep=10):
        o = torch.zeros(n, h, w)
        for f in on:
            o[f, 98:98 + deep, 88:104] = 1.0           # in the margin under the subject, across cells (6, 5) and (6, 6): two cells,
        return vm.window(dict(cells, others=o), *args)[2]   # so the one-cell close cannot give them back
    # THE CASE THAT GOES RED when a speck on one frame of a step holes the step, or a speck on most of its frames does not
    if not torch.equal(speck(step2[:1]), plain) or not torch.equal(speck(step2[:2]), plain):
        problems.append("region rules: a speck under one cell on one or two frames of a four-frame step took cells out of the region")
    if float(speck(step2[:3])[2, 6, 5]) != 0.0 or float(speck(step2[:3])[2, 6, 6]) != 0.0:
        problems.append("region rules: a speck on three of its step's four frames was dropped (a small face stays out)")
    if float(speck(step2[:1], deep=17)[2, 6, 5]) != 0.0:
        problems.append("region rules: a piece over one cell's area was dropped as a speck")
    if float(speck(step2[:3])[1, 6, 5]) != 1.0:
        problems.append("region rules: a speck of step 2 took a cell out of step 1")
    # a speck across a step boundary (the last two frames of step 2, the first two of step 3) is on half of each step: dropped from
    # both. Grouped by any other steps than the node's (`run_lengths`), three of its frames fall in one step and it holes it
    step3 = list(range(sum(vm.run_lengths(LATENT_T)[:3]), sum(vm.run_lengths(LATENT_T)[:4])))
    if not torch.equal(speck(step2[2:] + step3[:2]), plain):
        problems.append("region rules: a speck on two frames each side of a step boundary took cells out (its steps are not the node's)")
    # THE CASE THAT GOES RED when a one-cell hole with the region all round it stays open, a wider one, one the keep-out fills or one
    # held all through its step is closed (a small face), or the close is not in whole tokens where the region is. The piece reaches
    # from the subject's own cell (5, 6), which no keep-out takes, into cell (6, 6): 264 px, over the floor, so not a speck
    def holed(on, rows=(84, 106), cols=(98, 110), base=cells, a=args, size=(h, w)):
        o = torch.zeros(n, *size)
        for f in on:
            o[f, rows[0]:rows[1], cols[0]:cols[1]] = 1.0
        return vm.window(dict(base, others=o), *a)[2]
    if float(plain[2, 6, 6]) != 1.0 or float(holed(step2[:2])[2, 6, 6]) != 1.0:
        problems.append("region rules: a one-cell hole on two of its step's frames with the region all round it was not closed")
    if float(holed(step2)[2, 6, 6]) != 0.0:
        problems.append("region rules: a one-cell hole held all through its step was closed (a small face stays out)")
    if float(holed(step2[:2], rows=(84, 112), cols=(96, 112))[2, 6, 6]) != 0.0:
        problems.append("region rules: a cell the keep-out fills was closed")
    if float(holed(step2[:2], cols=(98, 126))[2, 6, 6]) != 0.0:
        problems.append("region rules: a hole two cells wide was closed")
    # in whole tokens (the default edge) the hole is a token: a 256 px square frame, the subject a token block in its middle
    big = torch.zeros(n, 256, 256)
    big[:, 96:160, 96:160] = 1.0                       # tokens rows 3-4, cols 3-4 of 32 px; grown by 64, tokens 1-6
    tok = {"frames": torch.rand(n, 256, 256, 3), "mask": big, "grow_pixels": 64, "feather_pixels": 0}
    targs = (0, n, 256, 256, LATENT_T, 16, 16)
    whole_plain = vm.window(tok, *targs)[2]
    got = holed(step2[:2], rows=(140, 170), cols=(100, 112), base=tok, a=targs, size=(256, 256))   # 120 px in token (5, 3)
    if float(whole_plain[2, 10:12, 6:8].min()) != 1.0 or not torch.equal(got, whole_plain):
        problems.append("region rules: in whole tokens a one-token hole with the region all round it was not closed")
    # THE CASE THAT GOES RED when a piece of the region the keep-outs cut off from the subject is kept
    band = torch.zeros(n, h, w)
    band[:, 96:112, :] = 1.0                           # a held thing across the frame under her: row 7 is cut off from her
    cut = vm.window(dict(cells, others=band), *args)[2]
    if float(plain[:, 7].max()) != 1.0 or float(cut[:, 7].max()) != 0.0 or float(cut[:, 6].max()) != 0.0:
        problems.append("region rules: the margin cut off from the subject by a keep-out stayed in the region")
    if not torch.equal(cut[:, :6], plain[:, :6]):
        problems.append("region rules: the region joined to the subject changed where a keep-out cut off a piece below it")
    kept = vm.window(dict(cells, keep=band), *args)[2]
    if float(kept[:, 7].max()) != 0.0:
        problems.append("region rules: a piece cut off by `keep` stayed in the region")


def check_queue_time_refusals(problems):
    node = vm.MiniMaxH3MaskedSource
    if vm.settings_refusal(vm.GROW_PIXELS, 8) is not None:
        problems.append("settings_refusal refuses the shipped margin and feather")
    wide = vm.settings_refusal(0, 8)
    soft = vm.settings_refusal(vm.GROW_PIXELS, 8, True, vm.START_TOP)
    if not wide or "wider than grow_pixels" not in wide:
        problems.append(f"settings_refusal does not name a feather wider than the margin: {wide!r}")
    if not soft or "paint_out" not in soft:
        problems.append(f"settings_refusal does not name a softened start with paint_out: {soft!r}")
    if vm.settings_refusal(8, 8) is not None or vm.settings_refusal(vm.GROW_PIXELS, 8, True) is not None:
        problems.append("settings_refusal refuses a feather equal to the margin, or paint_out alone")
    if node.validate_inputs(vm.GROW_PIXELS, 8) is not True:
        problems.append("validate_inputs does not pass the shipped defaults")
    if node.validate_inputs(0, 8) != wide or node.validate_inputs(vm.GROW_PIXELS, 8, True, vm.START_TOP) != soft:
        problems.append("validate_inputs does not return settings_refusal's message for the same values")
    if node.validate_inputs(None, 8) is not True or node.validate_inputs(vm.GROW_PIXELS, None) is not True:
        problems.append("validate_inputs refuses a margin or feather that is a link, which it cannot know yet")
    for label, got in (("a margin over the schema's largest", node.validate_inputs(vm.GROW_PIXELS_MAX + 1, 8)),
                       ("a negative feather", node.validate_inputs(vm.GROW_PIXELS, -1)),
                       ("a feather over the schema's largest", node.validate_inputs(vm.GROW_PIXELS_MAX, vm.FEATHER_PIXELS_MAX + 1)),
                       ("an unknown start_from", node.validate_inputs(vm.GROW_PIXELS, 8, False, "from somewhere"))):
        if not isinstance(got, str):
            problems.append(f"validate_inputs passes {label}: it names the input, so core no longer tests it")
    ranges = {i.id: (i.min, i.max) for i in node.define_schema().inputs if i.id in ("grow_pixels", "feather_pixels")}
    if ranges != {"grow_pixels": (0, vm.GROW_PIXELS_MAX), "feather_pixels": (0, vm.FEATHER_PIXELS_MAX)}:
        problems.append(f"the schema's ranges {ranges} are not the constants the validation tests against")
    # the control: the run-time path still refuses, with the same words
    frames, mask = torch.rand(FRAMES, H, W, 3), torch.zeros(FRAMES, H, W)
    mask[:, 20:40, 30:50] = 1.0
    try:
        node.execute(frames, mask, grow_pixels=0, feather_pixels=8)
        problems.append("execute runs a feather wider than the margin")
    except ValueError as e:
        if str(e) != wide:
            problems.append(f"execute refuses a feather wider than the margin in other words: {e}")


def check_wired_motion(problems):
    node = vm.MiniMaxH3MaskedSource
    SHORT = 64                                   # half the canvas's short side, so the scale-down is exercised
    n = FRAMES
    frames = torch.rand(n, H, W, 3)
    mask = torch.zeros(n, H, W)
    mask[:, 20:40, 30:50] = 1.0
    # a video whose every frame is one flat level, its own frame number: any slice of it names its frames
    levels = (torch.arange(n, dtype=torch.float32) + 1.0) / (n + 1.0)
    video = levels.view(n, 1, 1, 1).expand(n, H // 2, W // 2, 3).contiguous()
    src = {"frames": frames, "mask": mask, "motion_reference": vm.MOTION_WIRED, "motion_short_edge": SHORT,
           "motion_frames": video}
    first, count = 5, 9
    got = vm.wired_motion(src, first, count, W, H)
    th, tw = vm._reference_size(H, W, SHORT)
    if tuple(got.shape) != (count, th, tw, 3):
        problems.append(f"wired motion: a window's reference is {tuple(got.shape)}, not {(count, th, tw, 3)}")
        return
    seen = got.mean(dim=(1, 2, 3))
    if not torch.allclose(seen, levels[first:first + count], atol=1e-4):
        problems.append("wired motion: a window starting at frame 5 is not shown the wired video from frame 5")
    zero = vm.wired_motion(src, 0, count, W, H).mean(dim=(1, 2, 3))
    if torch.allclose(seen, zero, atol=1e-4):
        problems.append("control failed: the window at frame 5 and the window at frame 0 are shown the same frames, "
                        "so the cut is not by the window")
    tail = vm.wired_motion(src, n - 3, count, W, H).mean(dim=(1, 2, 3))
    if not torch.allclose(tail[:3], levels[n - 3:], atol=1e-4) or not torch.allclose(tail[3:], levels[-1].expand(count - 3), atol=1e-4):
        problems.append("wired motion: a window that runs past the video's end does not repeat its last frame")
    for label, kwargs, word in (
            ("the choice with nothing wired", dict(motion_reference=vm.MOTION_WIRED), "motion_video"),
            ("a video under another choice", dict(motion_reference=vm.MOTION_SUBJECT, motion_video=video), "never used in silence"),
            ("a video of another length", dict(motion_reference=vm.MOTION_WIRED, motion_video=video[:-1]), "one frame per source frame")):
        try:
            node.execute(frames, mask, **kwargs)
            problems.append(f"wired motion: {label} was accepted")
        except ValueError as exc:
            if word not in str(exc):
                problems.append(f"wired motion: the refusal of {label} does not say so: {exc}")
    out = node.execute(frames, mask, motion_reference=vm.MOTION_WIRED, motion_video=video, motion_short_edge=SHORT)
    record, strip = out.args[0], out.args[2]
    if record.get("motion_frames") is not video or record.get("motion_reference") != vm.MOTION_WIRED:
        problems.append("wired motion: the source record does not carry the wired video and its choice")
    plain = node.execute(frames, mask).args[2]
    if int(strip.shape[2]) <= int(plain.shape[2]):
        problems.append("wired motion: the preview strip is no wider than with no motion reference: the wired "
                        "video is not shown beside the plate")
    # `a video I wire, zoomed in` was removed on 2026-10-10 (one render, no difference): the choice is refused by
    # name, the list ends on the wired video, and a record that carries boxes changes nothing of the whole-frame one
    marked = video.clone()
    marked[:, 10:20, 15:25] = 1.0
    boxed = dict(node.execute(frames, mask, motion_reference=vm.MOTION_WIRED, motion_video=marked, motion_short_edge=SHORT).args[0])
    if not torch.equal(vm.wired_motion(boxed, first, count, W, H), vm.wired_motion(dict(src, motion_frames=marked), first, count, W, H)):
        problems.append("wired motion: the whole-frame choice changed when the record carries boxes")
    try:
        node.execute(frames, mask, motion_reference="a video I wire, zoomed in", motion_video=marked)
        problems.append("wired motion: the removed zoomed choice was accepted")
    except ValueError as exc:
        if "unknown motion_reference" not in str(exc):
            problems.append(f"wired motion: the refusal of the removed zoomed choice does not name the choices: {exc}")
    if vm.MOTIONS != (vm.MOTION_NONE, vm.MOTION_SUBJECT, vm.MOTION_ZOOM, vm.MOTION_FRAME, vm.MOTION_WIRED) \
            or any(hasattr(vm, name) for name in ("MOTION_WIRED_ZOOM", "wired_ground", "WIRED", "ZOOMED")):
        problems.append(f"motion_reference: the choices are the five, in place, and the zoomed wired one is gone with its code: {vm.MOTIONS}")
    got = _appended("motion_video")
    if got is None or not got.optional:
        problems.append("motion_video: it is not where it was appended (`APPENDED`) and optional")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "video_mask.wired_motion(source, int(round(w.start * FPS)), w.frames, width, height)" not in song \
            or "if motion == video_mask.MOTION_WIRED:" not in song:
        problems.append("wired motion: the song node does not cut the wired video at the window's own start")
    # a second wired video (2026-10-12): its own frames, its own short edge, cut by the window as the first is
    SHORT_2 = 32
    video_2 = (1.0 - levels).view(n, 1, 1, 1).expand(n, H // 4, W // 4, 3).contiguous()
    out = node.execute(frames, mask, motion_reference=vm.MOTION_WIRED, motion_video=video, motion_short_edge=SHORT,
                       motion_video_2=video_2, motion_short_edge_2=SHORT_2)
    both, strip_2 = out.args[0], out.args[2]
    if both.get("motion_frames_2") is not video_2 or both.get("motion_short_edge_2") != SHORT_2 or both.get("motion_frames") is not video:
        problems.append("second wired video: the source record does not carry both videos and the second's own short edge")
    second = vm.wired_motion(both, first, count, W, H, which=2)
    if tuple(second.shape) != (count, *vm._reference_size(H, W, SHORT_2), 3):
        problems.append(f"second wired video: a window's second reference is {tuple(second.shape)}, not at its own short edge")
    elif not torch.allclose(second.mean(dim=(1, 2, 3)), 1.0 - levels[first:first + count], atol=1e-4):
        problems.append("second wired video: a window starting at frame 5 is not shown the second video from frame 5")
    if not torch.equal(vm.wired_motion(both, first, count, W, H), vm.wired_motion(src, first, count, W, H)):
        problems.append("second wired video: wiring it changed what the first shows")
    if int(strip_2.shape[2]) <= int(strip.shape[2]):
        problems.append("second wired video: the preview strip is no wider than with one wired video")
    for label, kwargs, word in (
            ("a second video with no first", dict(motion_reference=vm.MOTION_NONE, motion_video_2=video_2), "comes after a first"),
            ("a second video of another length", dict(motion_reference=vm.MOTION_WIRED, motion_video=video, motion_video_2=video_2[:-1]),
             "one frame per source frame")):
        try:
            node.execute(frames, mask, **kwargs)
            problems.append(f"second wired video: {label} was accepted")
        except ValueError as exc:
            if word not in str(exc):
                problems.append(f"second wired video: the refusal of {label} does not say so: {exc}")
    try:
        vm.wired_motion(src, first, count, W, H, which=2)
        problems.append("second wired video: a source with one video gave a second")
    except ValueError as exc:
        if "motion_video_2" not in str(exc):
            problems.append(f"second wired video: asking a one-video source for the second does not name the input: {exc}")
    for name in ("motion_video_2", "motion_short_edge_2"):
        where = _appended(name)
        if where is None or not where.optional:
            problems.append(f"{name}: it is not where it was appended (`APPENDED`) and optional")
    if "which=2)" not in song or 'source.get("motion_frames_2") is not None' not in song:
        problems.append("second wired video: the song node does not cut it at the window's own start, after the first")
    # what a reference costs, as the song node's report prints it: a line a reference, and one for all
    afs = importlib.import_module("_h3pack.audio_freeze_song")
    rc = importlib.import_module("_h3pack.reference_conditioning")
    report = importlib.import_module("_h3pack.reference_report")
    refs = tuple(rc.RuntimeVideoReference(frames=vm.wired_motion(both, first, count, W, H, which=k), loaded_fps=24.0, soundtrack=None,
                                          use_vae=copy) for k, copy in ((1, True), (2, False)))
    said = afs.reference_costs(refs, W, H, count)
    priced = report.price_references(refs, W, H, count)
    if len(said) != 3 or not said[0].startswith("<Video 1> costs") or not said[1].startswith("<Video 2> costs"):
        problems.append(f"reference costs: two wired videos must give a line each, named in order, and one for all: {said}")
    else:
        one, two = priced.items
        if f"{one.qwen_tokens:,} tokens" not in said[0] or f"{one.dit_rows:,} rows" not in said[0] or one.dit_rows <= 0:
            problems.append(f"reference costs: the first video's line does not carry its tokens and its rows: {said[0]}")
        if f"{two.qwen_tokens:,} tokens" not in said[1] or "no copy for the video model" not in said[1] or two.dit_rows != 0:
            problems.append(f"reference costs: a video with no copy for the video model must say so: {said[1]}")
        if f"{priced.vision_tokens:,} tokens" not in said[2] or f"{priced.dit_reference_rows:,} rows" not in said[2]:
            problems.append(f"reference costs: the last line is not the sum over the references: {said[2]}")
    if afs.reference_costs((), W, H, count) != [] or afs.reference_costs(None, W, H, count) != []:
        problems.append("reference costs: no references must print nothing")
    broken = afs.reference_costs(("not a reference",), W, H, count)
    if len(broken) != 1 or "was not counted" not in broken[0]:
        problems.append(f"reference costs: a count that fails must be one line and never an error: {broken}")


def check_subject_boxes(problems):
    spec = importlib.util.spec_from_file_location("_h3pack.subject_boxes", REPO / "subject_boxes.py")
    sb = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.subject_boxes"] = sb
    spec.loader.exec_module(sb)
    mask = torch.zeros(4, H, W)
    mask[0, 20:40, 30:50] = 1.0
    mask[1, 0:10, 0:12] = 1.0                    # against the frame's corner
    mask[3, H - 6:H, W - 9:W] = 1.0             # against the far corner; frame 2 is empty
    plain = sb.frame_boxes(mask, 0, 1)             # these masks are a few hundred pixels: the floor is set aside here
    if plain != [[{"x": 30, "y": 20, "width": 20, "height": 20}], [{"x": 0, "y": 0, "width": 12, "height": 10}], [],
                 [{"x": W - 9, "y": H - 6, "width": 9, "height": 6}]]:
        problems.append(f"subject boxes: the boxes are not the masks' bounds, or an empty frame has one: {plain}")
    wide = sb.frame_boxes(mask, 8, 1)
    if wide[0] != [{"x": 22, "y": 12, "width": 36, "height": 36}] or wide[1] != [{"x": 0, "y": 0, "width": 20, "height": 18}] \
            or wide[3] != [{"x": W - 17, "y": H - 14, "width": 17, "height": 14}] or wide[2] != []:
        problems.append(f"subject boxes: a margin does not widen the box and stop at the frame's edge: {wide}")
    if sb.frame_boxes(mask.unsqueeze(-1), 0, 1) != plain:
        problems.append("subject boxes: a mask with a trailing channel is read differently")
    out = sb.MiniMaxH3SubjectBoxes.execute(mask, 0, 1)
    if out.args[0] != plain or "3 of 4 frames" not in out.args[1] or "frame 2" not in out.args[1]:
        problems.append(f"subject boxes: the node's boxes or its report are not the function's: {out.args[1]}")
    # the floor (2026-10-10): a mask that covers something, and too little of it to be a person, gets no box
    floored = sb.frame_boxes(mask)
    if floored != [[], [], [], []] or sb.too_small(mask) != [0, 1, 3]:
        problems.append(f"subject boxes: masks of a few hundred pixels are under the floor and got boxes: {floored}")
    said = sb.MiniMaxH3SubjectBoxes.execute(mask, 0).args[1]
    if "0 of 4 frames" not in said or "empty on 1 of them" not in said or "under 2048 px on 3" not in said:
        problems.append(f"subject boxes: the report does not tell an empty frame from one under the floor: {said}")
    try:
        from comfy_extras.nodes_sam3d_body import _per_frame_bboxes_from_detections
        read = _per_frame_bboxes_from_detections(plain, 4)
        if [tuple(b.shape) for b in read] != [(1, 4), (1, 4), (0, 4), (1, 4)] or read[0].tolist() != [[30.0, 20.0, 50.0, 40.0]]:
            problems.append("subject boxes: core's own reader does not read the list as one box a frame and none "
                            "on the empty one")
        from comfy_extras.sam3d_body.utils import _bbox_from_mask
        if _bbox_from_mask(mask[2]).tolist() != [0.0, 0.0, float(W), float(H)]:
            print("note  core's fallback for an empty mask is no longer the whole frame: the control in item 18 "
                  "has nothing to stand against")
    except ImportError as exc:
        print(f"note  core's SAM 3D Body reader did not import ({exc}): the list's shape was not checked against it")


def check_cut_gate(problems):
    runs = vm.run_lengths(LATENT_T)
    starts = [sum(runs[:k]) for k in range(len(runs))]
    # the third run, split one frame in: the subject is on its first frame only
    k = next(i for i, n in enumerate(runs) if i >= 2 and n >= 3)
    a, n = starts[k], runs[k]
    cut = a + 1

    def mask_on(frames):
        m = torch.zeros(FRAMES, H, W)
        for f in frames:
            m[f, 40:60, 50:70] = 1.0
        return m

    before = mask_on(range(0, cut))                     # there up to the cut, gone after it
    gate = vm.cut_gate(before, LATENT_T, [cut])
    want = torch.ones(FRAMES)
    want[cut:a + n] = 0.0
    if not torch.equal(gate, want):
        _fail(problems, f"cut gate: a run split at frame {cut} with the subject before it gave {gate.tolist()}; "
                        f"the frames {cut} to {a + n - 1} after the cut must be 0 and every other frame 1")
    after = mask_on(range(cut, FRAMES))                 # the other way round: the subject's shot starts at the cut
    gate = vm.cut_gate(after, LATENT_T, [cut])
    want = torch.ones(FRAMES)
    want[a:cut] = 0.0
    if not torch.equal(gate, want):
        _fail(problems, f"cut gate: with the subject only after the cut the frames before it in the run must be 0; got {gate.tolist()}")
    if not bool(vm.cut_gate(before, LATENT_T, []).all()):
        _fail(problems, "cut gate: with no cut a frame the subject's mask is empty on was left unlaid; a frame the "
                        "tracker lost inside a shot must stay covered by its run")
    lost = mask_on([f for f in range(FRAMES) if f != cut])
    if not bool(vm.cut_gate(lost, LATENT_T, []).all()) or not bool(vm.cut_gate(mask_on(range(FRAMES)), LATENT_T, [cut]).all()):
        _fail(problems, "cut gate: a lost frame with no cut, or a cut with the subject on both sides, must change nothing")
    if not bool(vm.cut_gate(before, LATENT_T, [a]).all()):
        _fail(problems, "cut gate: a cut on a run's first frame splits no run and must change nothing")
    if not bool(vm.cut_gate(mask_on(range(0, a)), LATENT_T, [cut]).all()):
        _fail(problems, "cut gate: a run the subject is on no frame of was gated; there is nothing of the subject to hold back")
    if not torch.equal(vm.cut_gate(before, LATENT_T, [1000 + cut], first_frame=1000), vm.cut_gate(before, LATENT_T, [cut])):
        _fail(problems, "cut gate: a cut named in the source's frames is not found through the window's first frame")
    # the cuts themselves: from a shot table when there is one, else the tracker's detector on the frames
    table = json.dumps({"shots": [{"first_frame": 0, "last_frame": 9}, {"first_frame": 10, "last_frame": 30},
                                  {"first_frame": 31, "last_frame": 40}]})
    if vm.source_cuts(torch.zeros(41, 8, 8, 3), table) != [10, 31]:
        _fail(problems, f"cut gate: the cuts of a three-shot table came back as {vm.source_cuts(torch.zeros(41, 8, 8, 3), table)}")
    torch.manual_seed(7)
    one, two = torch.rand(1, 108, 192, 3), torch.rand(1, 108, 192, 3)
    clip = torch.cat([one.expand(12, -1, -1, -1), two.expand(9, -1, -1, -1)], dim=0)
    if vm.source_cuts(clip) != [12]:
        _fail(problems, f"cut gate: two held pictures joined at frame 12 gave the cuts {vm.source_cuts(clip)}")
    if vm.source_cuts(one.expand(12, -1, -1, -1)):
        _fail(problems, "cut gate: a held shot was given a cut")
    # the song node gates through `lay_window` (item 20, which lays a window across a cut); the record carries the cuts
    # (taken once before the padding since 2026-10-22, which reads them for the held context frames across a cut)
    vm_text = (REPO / "video_mask.py").read_text(encoding="utf-8")
    if "        cuts = source_cuts(frames, table)\n" not in vm_text or '"cuts": cuts,' not in vm_text:
        _fail(problems, "cut gate: the Masked Source's record does not carry the source's cuts")


def check_lay_window(problems):
    """Item 20. The song node's composite as one function, and the file a window's region is saved in."""
    import tempfile

    grow_px, feather = 32, 8
    runs = vm.run_lengths(LATENT_T)
    starts = [sum(runs[:k]) for k in range(len(runs))]
    STARTS_OF = starts
    k = next(i for i, n in enumerate(runs) if i >= 2 and n >= 3)
    cut = starts[k] + 1                                  # one frame into a run: the subject's shot ends here
    old = torch.zeros(FRAMES, H, W)
    old[:cut, 48:80, 80:112] = 0.8                       # soft, as a tracker's mask is; gone after the cut
    tokens = vm.token_mask(vm.grow(old, grow_px), LATENT_T, LAT_H, LAT_W)
    pixels = torch.full((FRAMES, H, W, 3), 0.5)
    render = pixels.clone()
    render[:, 40:72, 96:128] = 0.9                       # the new subject, drawn on every frame of every run
    across = list(range(cut, starts[k] + runs[k]))
    first = 1000
    record = {"composite": vm.COMPOSITE_CHANGED, "feather_pixels": feather, "change_threshold": vm.CHANGE_THRESHOLD,
              "cuts": [first + cut], "grow_pixels": grow_px, "grow_by": vm.GROW_FIXED, "replace": vm.REPLACE_WHOLE,
              "edge": vm.EDGE_TOKENS}

    out, alpha, lines = vm.lay_window(render, pixels, tokens, old, record, grow_px, first)
    plain = vm.changed_alpha(render, pixels, tokens, old, feather, grow_px // 2, vm.CHANGE_THRESHOLD)
    gate = vm.cut_gate(old, LATENT_T, record["cuts"], first)
    if not torch.equal(alpha, plain * gate[:, None, None]) or not torch.equal(out, vm.composite(render, pixels, alpha)):
        _fail(problems, "lay window: under `only what changed` the frames are not the changed weight, gated, blended")
    if not torch.equal(out[across], pixels[across]) or not bool((plain[across] > 0.5).any()):
        _fail(problems, f"lay window: frames {across} lie across the cut and must be the source bit for bit, with a "
                        "weight the gate had something to take off")
    # A frame with no mask of its own is never laid (2026-10-10), cut or no cut: with the cuts taken away the
    # same frames are still the source, and the report says why in its other line. No key lays them as
    # renders before that day did: a record that still carries the old one is laid like any other.
    free, _a, quiet = vm.lay_window(render, pixels, tokens, old, {**record, "cuts": []}, grow_px, first)
    if not torch.equal(out, free) or len(quiet) != 2 or "no mask of their own and no cut beside them" not in quiet[1] \
            or not quiet[1].endswith(", ".join(str(first + f) for f in across)):
        _fail(problems, f"lay window: with no cut named, the frames with no mask must still be the source and be named: {quiet}")
    keyed, _a, said = vm.lay_window(render, pixels, tokens, old, {**record, "cuts": [], "lay_frames_with_no_mask": True},
                                    grow_px, first)
    if not torch.equal(keyed, free) or said != quiet:
        _fail(problems, "lay window: the key that laid frames with no mask as renders before 2026-10-10 did is gone, and a "
                        "record that still carries it must be laid like any other")
    if len(lines) != 2 or "only what changed" not in lines[0] or "lie across a cut" not in lines[1] \
            or not lines[1].endswith(", ".join(str(first + f) for f in across)):
        _fail(problems, f"lay window: the report's lines are {lines} with the cut")
    # mid-shot, no cut anywhere: one frame's mask emptied inside a step whose other frames have it. That frame
    # is the source bit for bit, its neighbours carry the render, and a frame with no mask in a step with no
    # region at all is not named
    lost = STARTS_OF[1] + 1
    holed = old.clone()
    holed[lost] = 0.0
    hole_tokens = vm.token_mask(vm.grow(holed, grow_px), LATENT_T, LAT_H, LAT_W)
    got, _a, told = vm.lay_window(render, pixels, hole_tokens, holed, {**record, "cuts": []}, grow_px, first)
    names = vm.unlaid_frames(holed, hole_tokens, [], first)
    if not torch.equal(got[lost], pixels[lost]) or torch.equal(got[lost - 1], pixels[lost - 1]) or torch.equal(got[lost + 1], pixels[lost + 1]):
        _fail(problems, f"lay window: frame {lost}, whose own mask is empty inside a step that has a region, must be the source, "
                        "and the frames beside it must carry the render")
    if names[0] or lost not in names[1] or names[1] != [lost] + across or FRAMES - 1 in names[1] \
            or f"{first + lost}" not in told[-1].split("frame(s) ")[-1]:
        _fail(problems, f"lay window: the frames with no mask a region reaches are {names[1]}; they must be frame {lost} and the "
                        f"split step's {across}, and no frame of a step with no region")
    plain_region = {**record, "composite": vm.COMPOSITE_REGION, "cuts": []}
    region_alpha = vm.pixel_alpha(tokens, H, W, feather)
    _w, n_alpha, n_lines = vm.lay_window(render, pixels, tokens, old, plain_region, grow_px, first)
    keep_frames = torch.ones(FRAMES)
    keep_frames[across] = 0.0
    if not torch.equal(n_alpha, region_alpha * keep_frames[:, None, None]) or len(n_lines) != 1:
        _fail(problems, "lay window: under `whole region` the weight must be the region's own on every frame with a mask and 0 "
                        "on a frame with none")

    # the file: what was written comes back, and a window laid from it is the window laid from the tensors
    each = torch.full((FRAMES,), grow_px, dtype=torch.long)
    each[::2] = grow_px // 2
    with tempfile.TemporaryDirectory() as tmp:
        for margin in (grow_px, each):
            path = vm.save_window_region(str(Path(tmp) / "w_region.npz"), old, tokens, record, margin, first, 5)
            got = vm.load_window_region(path)
            same_margin = torch.equal(got["margin"], margin) if torch.is_tensor(margin) else got["margin"] == margin
            if not torch.equal(got["mask"], (old > 0.5).float()) or not torch.equal(got["tokens"], tokens) or not same_margin \
                    or got["first_frame"] != first or got["trim"] != 5 or got["source"] != {key: record.get(key) for key in vm.REGION_SETTINGS}:
                _fail(problems, f"window region: the file does not read back as written (margin {vm.margin_note(margin)})")
                continue
            want = vm.lay_window(render, pixels, tokens, old, record, margin, first)[0]
            again = vm.lay_window(render, pixels, got["tokens"], got["mask"], got["source"], got["margin"], got["first_frame"])[0]
            if not torch.equal(want, again):
                _fail(problems, f"window region: a window laid from its file is not the window laid from its tensors "
                                f"(margin {vm.margin_note(margin)})")
        # a file from a render of before the no-mask rule is read like any other: nothing reproduces that render
        import numpy as np
        with np.load(path) as z:
            parts = {k: z[k] for k in z.files}
        meta = json.loads(str(parts["meta"]))
        now = vm.load_window_region(path)["source"]
        meta["version"] = 1
        old_path = str(Path(tmp) / "old_region.npz")
        with open(old_path, "wb") as fh:
            np.savez_compressed(fh, **{**parts, "meta": np.asarray(json.dumps(meta))})
        if vm.load_window_region(old_path)["source"] != now or "lay_frames_with_no_mask" in now:
            _fail(problems, "window region: a version 1 file must read back with the settings a file written now has, and no "
                            "key that lays frames with no mask")
        Path(old_path).unlink()
        if sorted(x.name for x in Path(tmp).iterdir()) != ["w_region.npz"]:
            _fail(problems, f"window region: the write left {sorted(x.name for x in Path(tmp).iterdir())} behind")

    song_text = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    # the one `cut_gate` the song node's file may hold is the plan writer's, which lays nothing: it names the
    # frames a render will leave as the source (`write_plan`, above the node's class)
    node_text = song_text[song_text.find("class MiniMaxH3AudioFreezeSong(io.ComfyNode):"):]
    if "video_mask.lay_window(images, src_pixels, src_tokens, src_mask, source," not in node_text \
            or any(name in node_text for name in ("video_mask.composite(", "video_mask.changed_alpha(", "video_mask.cut_gate(",
                                                   "video_mask.unlaid_frames(", "lay_frames_with_no_mask")) \
            or any(name in song_text for name in ("video_mask.composite(", "video_mask.changed_alpha(")):
        _fail(problems, "lay window: the song node must lay a window through `video_mask.lay_window` and through nothing else")
    if "video_mask.save_window_region(region_path, src_mask, src_tokens, source, margin," not in song_text \
            or "loop_resume.review_path(work_dir, filename, w.number), region_path, zoomed_path):" not in song_text:
        _fail(problems, "window region: the song node does not save a window's region beside its latent, or keeps a stale one")
    # 08c3cb12 shadowed the node's count of reused windows inside its loop, and the report's first line gave a
    # frame number for it. The count went with stored-window reuse (2026-10-10); the loop must not grow another
    if "        first = " in song_text or "reused" in song_text.split("class MiniMaxH3AudioFreezeSong(io.ComfyNode):")[1].split("def execute(")[1]:
        _fail(problems, "lay window: the song node's `execute` counts or names reused windows again; nothing an earlier run made is reused")


#: The most the plate inside a zoomed run's box may be from the handed frame, in levels, read through the
#: file's codec (`check_song_loop`). Measured 2026-10-10 on this fixture: 2 as built, 47 when the laid picture
#: went back whole in place of the decode under its weight.
PLATE_OFF_MOST = 24.0


def check_song_loop(problems):
    """Item 21. The song node's own loop over two windows of a masked source, the model's side stood in for."""
    import importlib
    import os
    import subprocess
    import tempfile

    import comfy.nested_tensor
    import folder_paths
    song = importlib.import_module("_h3pack.audio_freeze_song")
    af = importlib.import_module("_h3pack.audio_freeze")
    plan = importlib.import_module("_h3pack.loop_plan")

    width, height, window, context = 64, 48, min(plan.CHAIN_LENGTHS), 39
    total = 2 * window - context
    # two cuts, each a frame or two into a latent step, and the subject on the first and third shots
    cuts = [plan.step_span(96)[0] + 2, plan.step_span(176)[0] + 1]
    on = [(0, cuts[0] - 1), (cuts[1], total - 1)]
    frames = torch.full((total, height, width, 3), 0.3) + torch.linspace(0.0, 0.2, width)[None, None, :, None]
    frames[cuts[0]:cuts[1]] = 0.25
    mask = torch.zeros(total, height, width)
    for a, b in on:
        mask[a:b + 1, 16:32, 16:40] = 1.0
    # a table whole enough to be written beside the render, as the node writes the tracker's
    shots = importlib.import_module("_h3pack.shot_table")
    rows = [{"shot": n + 1, "first_frame": a, "last_frame": b - 1, "frames": b - a, "shown_frame": a, "people": [{}],
             "subject": {"person": 1 if (a, b - 1) in on else None, "state": "taken" if (a, b - 1) in on else "absent", "why": "made for the check"},
             "closest_person": None, "corrected": "", "on_screen": (a, b - 1) in on,
             "frames_with_subject": b - a if (a, b - 1) in on else 0, "frames_without_subject": [], "caption": ""}
            for n, (a, b) in enumerate(zip([0] + cuts, cuts + [total]))]
    table = json.dumps({"table": "h3 shot table", "version": shots.TABLE_VERSION, "frames": total, "size": [width, height],
                        "subject_phrase": "person", "pick": "largest", "pick_frame": None, "cuts": cuts, "shots": rows})
    made = vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=8, feather_pixels=2, composite=vm.COMPOSITE_CHANGED,
                                            shot_table=table)
    source = getattr(made, "args", made)[0]
    if source.get("cuts") != cuts:
        _fail(problems, f"song loop: the record's cuts are {source.get('cuts')}, not the table's {cuts}; the run below would test nothing")
        return
    split = plan.split_steps(cuts, 0, on, total)
    across = sorted(f for step in split for f in step["across"])
    if not across or any(step["shared"] for step in split):
        _fail(problems, f"song loop: the fixture's cuts lay {across} across a cut; it needs some, and none shared")
        return

    class Encoder:
        layer_idx = None
        patcher = types.SimpleNamespace(patches_uuid="none")

    class AudioVAE:
        audio_sample_rate = 32000

        def spacial_compression_encode(self):
            return 800

        def encode(self, samples_last):
            return torch.zeros(1, 32, 2, samples_last.shape[1] // 800)

    class VideoVAE:
        """The decode of a window is its own source with every cell the sampler marked painted bright."""

        def encode(self, pixels):
            self.seen = pixels.clone()
            steps = next(t for t in range(1, int(pixels.shape[0]) + 1) if af.pixel_frames(t) == int(pixels.shape[0]))
            return torch.zeros(1, 24, steps, height // 16, width // 16)

        def decode(self, latent):
            marked = latent[0, 0] > 0.5
            runs = torch.tensor(vm.run_lengths(int(marked.shape[0])))
            out = self.seen.clone()
            out[marked.repeat_interleave(runs, dim=0).repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)] = 0.95
            return out

    class Sampler:
        """Marks, in the latent's first channel, the cells the noise mask asks it to regenerate."""

        def __init__(self, model):
            pass

        def set_conds(self, conds):
            pass

        masks: list = []        # the video mask each window's sampler was handed, in order

        def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
            video, audio = latent_image.unbind()
            video = video.clone()
            Sampler.masks.append(denoise_mask.unbind()[0].clone())
            asked = denoise_mask.unbind()[0].expand_as(video[:, 0:1]) > 0.5
            video[:, 0:1] = torch.where(asked, torch.ones_like(video[:, 0:1]), video[:, 0:1])
            return comfy.nested_tensor.NestedTensor((video, audio))

    class Conditioning:
        @staticmethod
        def execute(*args, **kwargs):
            return ([["a stand-in", {}]],)

    track = {"waveform": torch.zeros(1, 2, total * 32000 // 24), "sample_rate": 32000}

    def run(out_dir: str, prefix: str, record=None, preview=False, **more):
        real = (song.Guider_Basic, song.MiniMaxH3Conditioning, folder_paths.get_output_directory,
                song.comfy.sample.fix_empty_latent_channels, song.latent_preview.prepare_callback)
        song.Guider_Basic, song.MiniMaxH3Conditioning = Sampler, Conditioning
        folder_paths.get_output_directory = lambda: out_dir
        song.comfy.sample.fix_empty_latent_channels = lambda model, latent, *a, **k: latent
        song.latent_preview.prepare_callback = lambda *a, **k: None
        try:
            # a preview is handed None for every lazy input but the source, as core hands it
            models = (None,) * 4 if preview else (object(), Encoder(), VideoVAE(), AudioVAE())
            got = song.MiniMaxH3AudioFreezeSong.execute(
                *models, track, None if preview else object(), None if preview else torch.linspace(1.0, 0.0, 5), "a prompt", "",
                preview, width, height, window, context, "whole", 7, 0.0, "clip_guard", prefix, 4,
                save_metadata_png=False, source=source if record is None else record, **more)
        finally:
            (song.Guider_Basic, song.MiniMaxH3Conditioning, folder_paths.get_output_directory,
             song.comfy.sample.fix_empty_latent_channels, song.latent_preview.prepare_callback) = real
        return getattr(got, "args", got)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t" / "run_windows"
        folder.mkdir(parents=True)
        (folder / "run_window_1_region.npz").write_bytes(b"left by an earlier run")
        try:
            path, report = run(tmp, "t/run")[:2]
        except Exception as exc:  # noqa: BLE001 -- the loop raising at all is the finding
            _fail(problems, f"song loop: the node's loop raised over two windows of a masked source: {type(exc).__name__}: {exc}")
            return
        lines = report.splitlines()
        if not lines[0].startswith("rendered -> ") or "NOT saved" in report or "FAILED" in report:
            _fail(problems, f"song loop: the report opens {lines[0][:60]!r}, or says a region or a review failed: "
                            + "; ".join(x for x in lines if "NOT saved" in x or "FAILED" in x))
        said = sorted(int(f) for x in lines if "lie across a cut" in x for f in x.rsplit("frame(s) ", 1)[1].split(", "))
        if said != across:
            _fail(problems, f"song loop: the report leaves frames {said} as the source; split_steps names {across} for the load")
        size = width * height * 3
        raw = subprocess.run([song._ffmpeg(), "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True).stdout
        if len(raw) != total * size:
            _fail(problems, f"song loop: the joined video holds {len(raw) // size} frames of {total}")
            return
        video = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(total, height, width, 3).float() / 255.0
        off = ((video - frames).abs().flatten(1).amax(dim=1) * 255.0).tolist()      # the furthest pixel of each frame, in levels
        painted = [f for f in range(total) if off[f] > 64]
        expected = [f for a, b in on for f in range(a, b + 1)]
        if painted != expected:
            wrong = sorted(set(painted) ^ set(expected))
            _fail(problems, f"song loop: {len(painted)} frames of the joined video carry the regenerated cells and the subject is on "
                            f"{len(expected)}; they differ on frames {wrong[:12]}, where the frames across a cut are {across}")
        for number, first, trim in ((1, 0, 0), (2, window - context, context)):
            name = folder / f"run_window_{number}_region.npz"
            try:
                got = vm.load_window_region(str(name))
            except Exception as exc:  # noqa: BLE001 -- a missing or stale file is the finding
                _fail(problems, f"song loop: window {number}'s region file does not read: {type(exc).__name__}: {exc}")
                continue
            steps = next(t for t in range(1, window + 1) if af.pixel_frames(t) == window)
            want = vm.window(source, first, window, width, height, steps, height // 16, width // 16)
            if not torch.equal(got["tokens"], want[2]) or not torch.equal(got["mask"], (want[3] > 0.5).float()) \
                    or (got["first_frame"], got["trim"]) != (first, trim) or got["source"]["cuts"] != cuts:
                _fail(problems, f"song loop: window {number}'s region file is not the window's own mask, tokens, first frame {first} and trim {trim}")
        beside = sorted(x.name for x in (Path(tmp) / "t").iterdir() if x.is_file())
        stem = Path(path).stem
        if beside != sorted([stem + ".mp4", stem + "_with_mask.mp4", stem + shots.SUFFIX_JSON, stem + shots.SUFFIX_TEXT]):
            _fail(problems, f"song loop: beside the render are {beside}; it owes the video, its mask review and the shot table")
        # `context_noise` (2026-10-10): the mask the second window's sampler is handed carries the value on the
        # context's steps where the region is, 0 on the context's plate, and is the default run's on every new step
        # over a subject who is on every frame, so the context holds region and plate both
        everywhere = torch.zeros(total, height, width)
        everywhere[:, 0:8, 0:8] = 1.0                    # small and in a corner: its region is one token of this canvas, not all of it
        held = getattr(vm.MiniMaxH3MaskedSource.execute(frames, everywhere, grow_pixels=8, feather_pixels=2,
                                                        composite=vm.COMPOSITE_CHANGED, shot_table=table), "args", None)[0]
        Sampler.masks.clear()
        try:
            run(tmp, "w/run", record=held)
            plain = [m.clone() for m in Sampler.masks]
            Sampler.masks.clear()
            noisy_report = run(tmp, "v/run", record=held, context_noise=0.5)[1]
        except Exception as exc:  # noqa: BLE001 -- the loop raising at all is the finding
            _fail(problems, f"song loop: the loop raised with context_noise 0.5: {type(exc).__name__}: {exc}")
            return
        noisy = [m.clone() for m in Sampler.masks]
        steps = next(t for t in range(1, window + 1) if af.pixel_frames(t) == window)
        ctx = next(t for t in range(1, steps) if af.pixel_frames(t) == context)
        region = vm.window(held, window - context, window, width, height, steps, height // 16, width // 16)[2][:ctx] > 0.5
        if len(plain) != 2 or len(noisy) != 2 or not torch.equal(plain[0], noisy[0]):
            _fail(problems, f"song loop: context_noise changed the first window's mask, or a window was not sampled ({len(plain)}, {len(noisy)})")
        elif not region.any() or region.all():
            _fail(problems, "song loop: the fixture's context has no region, or nothing but region; the case below would show nothing")
        else:
            was, now = plain[1][0, 0], noisy[1][0, 0]
            if bool(was[:ctx].any()) or not bool((now[:ctx][region] == 0.5).all()) or bool(now[:ctx][~region].any()) \
                    or not torch.equal(was[ctx:], now[ctx:]):
                _fail(problems, "song loop: with context_noise 0.5 the second window's mask must be 0.5 on the context's region cells, "
                                "0 on the context's plate and unchanged on its new steps, and all 0 on the context at the default")
        if "shown at noise 0.5" not in noisy_report:
            _fail(problems, "song loop: the report does not say the context was shown with noise")
        # a preview's plan (2026-10-10): the same prefix as the render above, nothing sampled, and what it writes
        # is what that render wrote
        ask = song.MiniMaxH3AudioFreezeSong.check_lazy_status
        if ask(preview=True, source=None, model=None, vae=None, clip=None) != ["source"] or ask(preview=True, model=None) != [] \
                or "source" not in ask(preview=False, source=None, model=None) \
                or ask(preview=True, source=None, references=None, model=None, clip=None, sigmas=None) != ["source", "references"]:
            _fail(problems, "song loop: a preview must ask for the source and the reference chain when they are wired, and for nothing else")
        rendered = {n: (folder / f"run_window_{n}_region.npz").read_bytes() for n in (1, 2)}
        (folder / "run_window_9_planned_region.npz").write_bytes(b"left by an earlier plan")
        Sampler.masks.clear()
        try:
            said = run(tmp, "t/run", preview=True)[1]
        except Exception as exc:  # noqa: BLE001 -- a preview raising at all is the finding
            _fail(problems, f"song loop: a preview with a source raised: {type(exc).__name__}: {exc}")
            return
        if Sampler.masks or "plan written with nothing sampled: 2 planned region file(s)" not in said:
            _fail(problems, f"song loop: a preview sampled, or does not say what it wrote: {said.splitlines()[-2:]}")
        # what each window's references will cost, in the plan (2026-10-12): counted, not encoded
        if "costs:" in said or "the references in all" in said:
            _fail(problems, "song loop: a plan with no reference of any kind prints a reference's cost")
        have = int(source["frames"].shape[0])
        level = torch.linspace(0.1, 0.9, have).view(have, 1, 1, 1)
        both = dict(source, motion_reference=vm.MOTION_WIRED, motion_vae=True, motion_short_edge=64, motion_short_edge_2=32,
                    motion_frames=level.expand(have, height // 2, width // 2, 3).contiguous(),
                    motion_frames_2=(1.0 - level).expand(have, height // 4, width // 4, 3).contiguous())
        try:
            planned = run(tmp, "costs/run", record=both, preview=True)[1].splitlines()
        except Exception as exc:  # noqa: BLE001
            _fail(problems, f"song loop: a preview of a load with two wired videos raised: {type(exc).__name__}: {exc}")
            planned = []
        if planned:
            windows_planned = sorted({int(line[1:line.index("]")]) for line in planned if line.startswith("[") and "costs:" in line})
            lone = types.SimpleNamespace(start=0.0, frames=window, number=1)
            want = [f"[1] {line}" for line in song.reference_costs(song.window_references(both, None, lone, width, height)[0], width, height, window, planned=True)]
            got = [line for line in planned if line.startswith("[1] ") and ("costs:" in line or "the references in all" in line)]
            if windows_planned != [1, 2] or got != want or len(want) != 3:
                _fail(problems, f"song loop: a plan must print, for each window, what each of its references costs and the sum: windows "
                                f"{windows_planned}, window 1 said {got}, the count is {want}")
            elif "<Video 1> costs" not in want[0] or "<Video 2> costs" not in want[1] or "A PLAN'S COUNT" not in want[2] \
                    or "A PLAN'S COUNT" in " ".join(song.reference_costs(song.window_references(both, None, lone, width, height)[0], width, height, window)):
                _fail(problems, f"song loop: a plan's cost lines must name each video and say they are a plan's count, and a render's must not: {want}")
            if Sampler.masks:
                _fail(problems, "song loop: counting what the references cost sampled something")
        short = dict(both, motion_frames=both["motion_frames"][:window // 2], motion_frames_2=None)
        try:
            uncounted = run(tmp, "short/run", record=short, preview=True)[1]
        except Exception as exc:  # noqa: BLE001
            _fail(problems, f"song loop: a plan whose second window has no frames of the wired video raised: {type(exc).__name__}: {exc}")
            uncounted = ""
        if "[2] what the references cost was not counted:" not in uncounted or "[1] <Video 1> costs" not in uncounted:
            _fail(problems, "song loop: a window whose references cannot be counted must say so in one line and leave the rest of the plan standing")
        if (folder / "run_window_9_planned_region.npz").exists() or any(
                (folder / f"run_window_{n}_region.npz").read_bytes() != rendered[n] for n in (1, 2)):
            _fail(problems, "song loop: a preview left an earlier plan's file in place, or touched a render's region file")
        for n in (1, 2):
            try:
                planned = vm.load_window_region(str(folder / f"run_window_{n}_planned_region.npz"))
            except Exception as exc:  # noqa: BLE001
                _fail(problems, f"song loop: window {n}'s planned region does not read: {type(exc).__name__}: {exc}")
                continue
            real = vm.load_window_region(str(folder / f"run_window_{n}_region.npz"))
            if not torch.equal(planned["tokens"], real["tokens"]) or not torch.equal(planned["mask"], real["mask"]) \
                    or any(planned[k] != real[k] for k in ("margin", "first_frame", "trim", "source")):
                _fail(problems, f"song loop: window {n}'s planned region is not the region the render of the same graph saved")
        try:
            plan_file = json.loads((folder / "run_plan.json").read_text())
        except Exception as exc:  # noqa: BLE001
            _fail(problems, f"song loop: the plan file does not read: {type(exc).__name__}: {exc}")
            return
        rows = plan_file["windows"]
        left = sorted(f for r in rows for f in r["frames_left_as_source_and_written"])
        wrote = [(r["first_written_frame"], r["last_written_frame"]) for r in rows]
        if left != across or wrote != [(0, window - 1), (window, total - 1)] or [r["trim"] for r in rows] != [0, context] \
                or sorted({tuple(s["step"]) for r in rows for s in r["split_steps"]}) != sorted(tuple(s["step"]) for s in split) \
                or plan_file["source"]["cuts"] != cuts or plan_file["plan"] != song.PLAN_NAME:
            _fail(problems, f"song loop: the plan names frames {left} as left to the source and windows writing {wrote}; the "
                            f"render left {across} and wrote two windows from 0 and from {window}")
        # a CONTINUED run, where a plan is used most: its first window takes its context from a stored window, so
        # its trim is the context and not 0. The plan re-derives that trim; the render takes it from the window node.
        resume = importlib.import_module("_h3pack.loop_resume")
        audio_steps = int(round(window * 5 / 3))
        stored = resume.save_window(str(Path(tmp)), "before", 1, comfy.nested_tensor.NestedTensor(
            (torch.zeros(1, 24, steps, height // 16, width // 16), torch.zeros(1, 32, 2, audio_steps))), 0, 0.0, window)
        try:
            run(tmp, "c/run", continue_from=stored)
            run(tmp, "c/run", preview=True, continue_from=stored)
        except Exception as exc:  # noqa: BLE001
            _fail(problems, f"song loop: a continued run, or its preview, raised: {type(exc).__name__}: {exc}")
            return
        there = Path(tmp) / "c" / "run_windows"
        plan_c = json.loads((there / "run_plan.json").read_text())
        for n, row in enumerate(plan_c["windows"], start=1):
            planned, real = (vm.load_window_region(str(there / f"run_window_{n}{kind}.npz")) for kind in ("_planned_region", "_region"))
            if planned["trim"] != real["trim"] or real["trim"] != context or row["trim"] != context \
                    or planned["first_frame"] != real["first_frame"] or not torch.equal(planned["tokens"], real["tokens"]):
                _fail(problems, f"song loop: on a continued run window {n}'s plan has trim {planned['trim']} and the render "
                                f"{real['trim']}; both must be the context, {context}, and the regions the same")
        if plan_c["windows"][0]["first_written_frame"] != context or plan_c["continue_from"] != Path(stored).name:
            _fail(problems, f"song loop: a continued run's plan says its first window writes from frame "
                            f"{plan_c['windows'][0]['first_written_frame']}; it writes from {context}, after the context it was handed")
        # RENDERED ZOOMED IN (2026-10-10): one shot, a small subject, frames handed at the canvas's own size as the
        # job graphs hand them, so the box is enlarged; the plate is a fine texture, which an enlargement and a
        # reduction would blur. The run is made on one box of the load; the video is the whole frame with the
        # region laid back and its plate untouched; the zoomed picture and what says its box are written beside
        # it; both windows share the box; the plan says it too
        texture = torch.rand(height, width, generator=torch.Generator().manual_seed(11)) * 0.4 + 0.2
        big = texture[None, :, :, None].expand(total, -1, -1, 3).clone()
        small = torch.zeros(total, height, width)
        small[:, 20:24, 28:34] = 1.0
        one_shot = json.dumps({"table": "h3 shot table", "version": shots.TABLE_VERSION, "frames": total, "size": [width, height],
                               "subject_phrase": "person", "pick": "largest", "pick_frame": None, "cuts": [], "shots": [
                                   {"shot": 1, "first_frame": 0, "last_frame": total - 1, "frames": total, "shown_frame": 0, "people": [{}],
                                    "subject": {"person": 1, "state": "taken", "why": "made for the check"}, "closest_person": None,
                                    "corrected": "", "on_screen": True, "frames_with_subject": total, "frames_without_subject": [],
                                    "caption": ""}]})
        far = getattr(vm.MiniMaxH3MaskedSource.execute(big, small, grow_pixels=4, feather_pixels=0, composite=vm.COMPOSITE_REGION,
                                                       shot_table=one_shot, render_region=vm.RENDER_ZOOM_REPLACED), "args", None)[0]
        try:
            zpath, zreport = run(tmp, "z/run", record=far)[:2]
            zplan_said = run(tmp, "z/run", record=far, preview=True)[1]
        except Exception as exc:  # noqa: BLE001 -- the loop raising at all is the finding
            _fail(problems, f"song loop, zoomed: the loop or its preview raised: {type(exc).__name__}: {exc}")
            return
        zfolder, zstem = Path(tmp) / "z" / "run_windows", Path(zpath).stem
        if "rendered zoomed in on what is replaced" not in zreport or "NOT written" in zreport or "NOT saved" in zreport or "FAILED" in zreport \
                or "rendered zoomed in on what is replaced" not in zplan_said:
            _fail(problems, "song loop, zoomed: the report or the preview does not say the run is zoomed, or says a file failed: "
                            + "; ".join(x for x in zreport.splitlines() if "NOT " in x or "FAILED" in x))
        try:
            record = json.loads((Path(tmp) / "z" / (zstem + "_zoomed.json")).read_text())
        except Exception as exc:  # noqa: BLE001
            _fail(problems, f"song loop, zoomed: the box's file beside the render does not read: {type(exc).__name__}: {exc}")
            return
        bx0, by0, bx1, by1 = record["box"]
        if (record["file"], record["frames"], record["canvas"], record["source_size"], record["fixed"], record["piece"], record["zoomed"]) != \
                (song.ZOOMED_NAME, total, [width, height], [width, height], True, zstem + ".mp4", zstem + "_zoomed.mp4") \
                or (bx1 - bx0, by1 - by0) != (width // 2, height // 2) or abs(record["zoom"] - 2.0) > 1e-9:
            _fail(problems, f"song loop, zoomed: the box's file says {record}")

        def read(name):
            got = subprocess.run([song._ffmpeg(), "-v", "error", "-i", str(name), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 capture_output=True).stdout
            if len(got) != total * size:
                return None
            return torch.frombuffer(bytearray(got), dtype=torch.uint8).reshape(total, height, width, 3).float() / 255.0
        piece, closer = read(zpath), read(Path(tmp) / "z" / (zstem + "_zoomed.mp4"))
        if piece is None or closer is None:
            _fail(problems, f"song loop, zoomed: the joined video and the joined zoomed picture must each hold {total} frames of the canvas")
            return
        whole_frames = vm.fit_frames(big, width, height)
        cut = vm.crop_frames(big, record["box"], width, height)
        # the painted cells are tokens of the ZOOMED picture: 16 px there, 8 px here, inside the box
        lit, lit_closer = (piece - whole_frames).abs().amax(dim=-1) * 255.0 > 64, (closer - cut).abs().amax(dim=-1) * 255.0 > 64
        inside = torch.zeros(height, width, dtype=torch.bool)
        inside[by0:by1, bx0:bx1] = True
        per_frame, per_closer = lit.flatten(1).sum(dim=1).float(), lit_closer.flatten(1).sum(dim=1).float()
        if not bool(lit.flatten(1).any(dim=1).all()) or bool(lit[:, ~inside].any()) or not bool(lit_closer.flatten(1).any(dim=1).all()):
            _fail(problems, "song loop, zoomed: every frame of the video must carry the regenerated cells inside the box and nowhere else, "
                            "and every frame of the zoomed picture must carry them")
        elif not 0.15 < float((per_frame / per_closer).median()) < 0.4:
            _fail(problems, f"song loop, zoomed: the regenerated cells take {float(per_frame.median()):.0f} px of a frame of the video and "
                            f"{float(per_closer.median()):.0f} of the zoomed picture; at 2 times that is about a quarter")
        # the plate inside the box, clear of the cells that were regenerated, is the handed frame's own and not
        # that frame enlarged and reduced again: what would be laid if the laid picture went back in place of
        # the decode under its weight. In levels, through the file's codec
        near = torch.nn.functional.max_pool2d(lit.float()[:, None], 5, stride=1, padding=2)[:, 0] > 0
        plate_off = float(((piece - whole_frames).abs().amax(dim=-1) * 255.0)[(~near) & inside[None]].max())
        if plate_off > PLATE_OFF_MOST:
            _fail(problems, f"song loop, zoomed: the plate inside the box is {plate_off:.1f} levels from the handed frame's own at most; "
                            "only the region is laid back, so it must be the frame as it was handed in")
        regions = []
        for number in (1, 2):
            try:
                regions.append(vm.load_window_region(str(zfolder / f"run_window_{number}_region.npz")))
            except Exception as exc:  # noqa: BLE001
                _fail(problems, f"song loop, zoomed: window {number}'s region file does not read: {type(exc).__name__}: {exc}")
                return
            if not (zfolder / f"run_window_{number}_zoomed.mp4").is_file():
                _fail(problems, f"song loop, zoomed: window {number} left no zoomed picture beside its video")
        plan_z = json.loads((zfolder / "run_plan.json").read_text())
        if regions[0]["zoom"] is None or regions[0]["zoom"] != regions[1]["zoom"] or regions[0]["zoom"]["box"] != record["box"] \
                or plan_z.get("zoom") != regions[0]["zoom"]:
            _fail(problems, f"song loop, zoomed: both windows' region files and the plan must say the one box the load was rendered in "
                            f"({record['box']}): {regions[0]['zoom']}, {regions[1]['zoom']}, {plan_z.get('zoom')}")
        zsource = vm.zoomed_source(far, width, height)[0]
        want = vm.window(zsource, window - context, window, width, height, steps, height // 16, width // 16)
        if not torch.equal(regions[1]["tokens"], want[2]) or not torch.equal(regions[1]["mask"], (want[3] > 0.5).float()):
            _fail(problems, "song loop, zoomed: a window's region file must hold the mask and tokens of the ZOOMED picture")
        zbeside = sorted(x.name for x in (Path(tmp) / "z").iterdir() if x.is_file())
        if zbeside != sorted([zstem + ".mp4", zstem + "_with_mask.mp4", zstem + "_zoomed.mp4", zstem + "_zoomed.json",
                              zstem + shots.SUFFIX_JSON, zstem + shots.SUFFIX_TEXT]):
            _fail(problems, f"song loop, zoomed: beside the render are {zbeside}")
        # A ZOOMED LOAD WITH A WIRED MOTION VIDEO, through the loop (2026-10-13): its plan is made, says the video
        # was not cut to the box, and counts the reference a whole-frame plan of the same load counts
        far_frames = int(far["frames"].shape[0])
        far_wired = dict(far, motion_reference=vm.MOTION_WIRED, motion_vae=True, motion_short_edge=64,
                         motion_frames=torch.linspace(0.1, 0.9, far_frames).view(far_frames, 1, 1, 1).expand(far_frames, height, width, 3).contiguous())
        try:
            zoomed_plan = run(tmp, "zw/run", record=far_wired, preview=True)[1].splitlines()
            whole_plan = run(tmp, "ww/run", record={**far_wired, "render_region": vm.RENDER_WHOLE}, preview=True)[1].splitlines()
        except Exception as exc:  # noqa: BLE001 -- the loop raising at all is the finding
            _fail(problems, f"song loop, zoomed: a plan of a zoomed load with a wired motion video raised: {type(exc).__name__}: {exc}")
        else:
            costs = [[line for line in plan if "<Video 1> costs" in line] for plan in (zoomed_plan, whole_plan)]
            if not any("motion video is NOT cut to the box" in line for line in zoomed_plan) \
                    or any("NOT cut to the box" in line for line in whole_plan):
                _fail(problems, "song loop, zoomed: a zoomed plan with a wired motion video must say the video was not cut, and a whole-frame plan must not")
            if not costs[0] or costs[0] != costs[1]:
                _fail(problems, f"song loop, zoomed: the wired video of a zoomed plan must cost what it costs on the whole frame: {costs[0]} against {costs[1]}")
        try:
            y_plan = run(tmp, "y/run", record={**far, "cuts": [7]}, preview=True)[1]
            if "rendered zoomed in on what is replaced" not in y_plan:
                _fail(problems, f"song loop, zoomed: a load with a context cut in its window did not extend the zoom: {y_plan}")
        except ValueError as exc:
            _fail(problems, f"song loop, zoomed: a load with a cut in its window was refused instead of extending the zoom: {exc}")
        # CONTINUING a stored window: a zoomed run continues nothing, whatever the stored window says of its
        # framing; and a whole-frame run is refused on a stored window that says it was zoomed
        for said_of_it in ("has no region file", "says it was zoomed"):
            try:
                run(tmp, "k/run", record=far, continue_from=stored)
                _fail(problems, f"song loop, zoomed: a zoomed run continued a stored window that {said_of_it}")
            except ValueError as exc:
                if "a zoomed run continues nothing" not in str(exc):
                    _fail(problems, f"song loop, zoomed: the refusal of a zoomed continuation does not say why: {exc}")
            vm.save_window_region(stored[:-len(".safetensors")] + "_region.npz", regions[0]["mask"], regions[0]["tokens"],
                                  {"zoom": regions[0]["zoom"]}, 4, 0, 0)
        try:
            run(tmp, "j/run", continue_from=stored)
            _fail(problems, "song loop, zoomed: a whole-frame run continued a window that was rendered zoomed in")
        except ValueError as exc:
            if "was rendered zoomed in" not in str(exc):
                _fail(problems, f"song loop, zoomed: the refusal of a whole-frame run on a zoomed window does not say so: {exc}")
        if record.get("region_cut_at_box_sides") is None or any("region_cut_at_box_sides" not in r for r in plan_z["windows"]):
            _fail(problems, "song loop, zoomed: the box's file and each planned window must say which sides of the box the region is cut at")
        zgone = run(tmp, "x/run", record=far, keep_windows=False)
        if os.path.isdir(Path(tmp) / "x" / "run_windows") or not os.path.isfile(zgone[0]) \
                or not os.path.isfile(zgone[0][:-len(".mp4")] + "_zoomed.mp4"):
            _fail(problems, "song loop, zoomed: with keep_windows off the windows folder must go and the joined zoomed picture stay")
        gone = run(tmp, "u/run", keep_windows=False)
        if os.path.isdir(Path(tmp) / "u" / "run_windows") or not os.path.isfile(gone[0]):
            _fail(problems, f"song loop: with keep_windows off the windows folder still holds "
                            f"{sorted(os.listdir(Path(tmp) / 'u' / 'run_windows')) if os.path.isdir(Path(tmp) / 'u' / 'run_windows') else 'nothing'}"
                            ", or the joined video is missing")


def check_motion_look(problems):
    """Item 22, as it stands since 2026-10-10. The blur, the grey and the box choice were removed with their code
    (reasoned, built, never rendered); what is held is that they are gone and what replaced the box choice: a
    zoomed reference is framed on the replaced part when a part is replaced, and on the tracked person otherwise."""
    node = vm.MiniMaxH3MaskedSource
    n = FRAMES
    frames = torch.rand(n, H, W, 3)
    person = torch.zeros(n, H, W)
    person[:, 16:112, 32:96] = 1.0
    face = torch.zeros(n, H, W)
    face[:, 20:44, 52:76] = 1.0
    inputs = {i.id for i in node.define_schema().inputs}
    gone = ("motion_box", "motion_blur_share", "motion_blur_pixels", "motion_grey")
    names = ("blur_frames", "grey_frames", "blur_sigmas", "look_note", "motion_look", "look_boxes", "window_look_note",
             "BOX_SUBJECT", "BOX_REPLACED", "BOXES", "MOTION_BLUR_SHARE", "MOTION_BLUR_PIXELS", "LUMA")
    if inputs & set(gone) or any(hasattr(vm, name) for name in names):
        problems.append(f"motion look: a removed input or its code is back: {sorted(inputs & set(gone))}, "
                        f"{[name for name in names if hasattr(vm, name)]}")
    for kw in (dict(motion_grey=True), dict(motion_blur_share=0.1), dict(motion_box="what is replaced")):
        try:
            node.execute(frames, person, motion_reference=vm.MOTION_SUBJECT, **kw)
            problems.append(f"motion look: the node still takes {list(kw)[0]}")
        except TypeError:
            pass
    whole = node.execute(frames, person, grow_pixels=8, motion_reference=vm.MOTION_ZOOM, motion_short_edge=64).args[0]
    part = node.execute(frames, person, grow_pixels=8, replace=vm.REPLACE_PARTS, parts=face, part_margin=0,
                        motion_reference=vm.MOTION_ZOOM, motion_short_edge=64).args[0]
    if any(k in whole for k in gone):
        problems.append("motion look: the record still carries a removed key")
    on_person, on_part = vm.window_boxes(whole, 0, n, W, H), vm.window_boxes(part, 0, n, W, H)
    if on_person is None or on_part is None:
        problems.append("motion look: a record made by the node carries no boxes for a zoomed reference")
        return
    area = lambda b: int((b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1]))
    if not area(on_part) < area(on_person) or int(on_part[0, 0]) > 52 or int(on_part[0, 2]) < 76 \
            or int(on_person[0, 0]) > 32 or int(on_person[0, 2]) < 96:
        problems.append(f"motion look: a pass that replaces a part must frame its zoomed reference on the part "
                        f"({on_part[0].tolist()}) and a whole-subject pass on the person ({on_person[0].tolist()})")
    # the reference is the frames it was before the look existed: no argument left that could change them
    pixels, mask, _held = vm.window_frames(whole, 0, n, W, H)
    ref = vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, 64, 4)
    try:
        vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, 64, 4, None, None)
        problems.append("motion look: `motion_reference` still takes a ground or a look")
    except TypeError:
        pass
    if ref is None or float(ref.min()) < 0.0 or float(ref.max()) > 1.0:
        problems.append("motion look: the subject-only reference is not a picture in range")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "motion_look" in song or "window_look_note" in song or "look_boxes" in song:
        problems.append("motion look: the song node still reads a look")


def check_render_zoom(problems):
    """Item 24. A small subject rendered zoomed in: the zoom to a target head size a load may ask for (2026-10-22),
    the one box of a load, the source made of it, the lay back, and a wired motion video handed on as it was wired
    (2026-10-13; refused from 2026-10-10 until then)."""
    _zoom_to_head(problems)
    _render_zoom(problems)


def _zoom_to_head(problems):
    """1b. ZOOM TO A TARGET HEAD SIZE, SWITCHED OFF BY DEFAULT (the owner's go and word of 2026-10-22). On fun's canvas,
    the frames handed at its size: at the default every head is zoomed as the cap of 2 zoomed it, to the pixel; a load
    that asks up to `ZOOM_MOST_ASKED` (`zoom_up_to`) has a small head zoomed past 2 until it is `ZOOM_HEAD_TARGET` px
    tall, and no further than it asked; a head the target reaches under 2, and a subject whose box holds it under 2,
    are zoomed as before even then. Red on the rule before (a load asking for 4 had every small head at 4)."""
    target, most = getattr(vm, "ZOOM_HEAD_TARGET", None), getattr(vm, "ZOOM_MOST_ASKED", 4.0)
    by_head, by_any = getattr(vm, "SET_BY_HEAD", "the head's target height"), getattr(vm, "SET_BY_ANY", "the zoom any load may take")
    w, h = 1024, 768

    def head(x0, y0, wide, tall, frames=6, step=0):
        """A head `wide` by `tall` px from (x0, y0), moving `step` px right a frame, with one frame it is not in."""
        return torch.tensor([[x0 + i * step, y0, x0 + i * step + wide, y0 + tall] for i in range(frames)] + [[-1] * 4])

    def zoom_of(rows, **kw):
        import inspect
        kw = {k: v for k, v in kw.items() if k in inspect.signature(vm.zoom_box).parameters}   # the rule before had no any_load
        box, why = vm.zoom_box(rows, w, h, w, h, **kw)
        return (None, why) if box is None else (w / (box[2] - box[0]), why, box)
    # SWITCHED OFF: at the default the 90 px head and the 48 px head stay at the cap of 2, as on every render to 2026-10-22
    for name, rows in (("90 px", head(470, 300, 60, 90)), ("48 px", head(490, 320, 30, 48))):
        off = zoom_of(rows)
        if off[0] is None or off[0] != vm.ZOOM_LARGEST or vm.ZOOM_LARGEST != 2.0 or off[1] != vm.SET_BY_LARGEST:
            _fail(problems, f"render zoom, to the head: at the default a {name} head is zoomed to the cap of 2, set by the largest zoom; it came to {off}")
    # the second subject's heads (the lead's table, 2026-10-22): 4602d's, 90 px, came to 2 under the cap
    small = zoom_of(head(470, 300, 60, 90), largest=most)
    if small[0] is None or not (2.0 < small[0] <= 300 / 90 + 1e-9) or small[1] != by_head or 90 * small[0] < 300 * (1 - 1 / 32):
        _fail(problems, f"render zoom, to the head: a head 90 px tall on the canvas is zoomed past 2 until it is about {target} px tall, "
                        f"set by the head's target; it came to {small}")
    tiny = zoom_of(head(490, 320, 30, 48), largest=most)
    if tiny[0] is None or abs(tiny[0] - 4.0) > 1e-9 or tiny[1] != vm.SET_BY_LARGEST or most != 4.0:
        _fail(problems, f"render zoom, to the head: a head 48 px tall (1419c's), asking up to 4, is zoomed to 4 and no more; it came to {tiny}")
    three = zoom_of(head(490, 320, 30, 48), largest=3.0)
    if three[0] is None or abs(three[0] - 3.0) > 0.05 or three[0] > 3.0 + 1e-9 or three[1] != vm.SET_BY_LARGEST:
        _fail(problems, f"render zoom, to the head: the same head asking up to 3 is zoomed to about 3 and no more; it came to {three}")
    # a head that moves across the frame: the box with its room bounds it before the target does
    roams = zoom_of(head(200, 300, 30, 48, step=60), largest=most)
    if roams[0] is None or roams[1] != vm.SET_BY_BOX or roams[0] >= 300 / 48:
        _fail(problems, f"render zoom, to the head: a small head that crosses the frame is zoomed only as far as its box with its room allows; {roams}")
    else:
        x0, y0, x1, y1 = roams[2]
        if x0 > 200 or x1 < 200 + 5 * 60 + 30 or (x1 - x0) < 330 * (1 + 2 * vm.ZOOM_ROOM) - 1e-6:
            _fail(problems, f"render zoom, to the head: the box {roams[2]} must hold the head's travel with its room")
    # unchanged: a head the target reaches at 2 or less (1039e's, 197 px), and one whose box holds it under 2 (0415n's)
    for name, rows, by in (("a head 197 px tall", head(450, 250, 120, 197), by_any),
                           ("0415n's head and its travel", torch.tensor([[513, 90, 886, 346], [540, 104, 870, 370], [-1] * 4]), vm.SET_BY_BOX)):
        now, then = zoom_of(rows, largest=most), zoom_of(rows)
        if now[0] is None or then[0] is None or now[2] != then[2] or now[1] != by:
            _fail(problems, f"render zoom, to the head: {name} is zoomed as the cap of 2 zoomed it ({then}), set by {by!r}; it came to {now}")
    tall = getattr(vm, "region_height", None)
    if tall is None or tall(head(470, 300, 60, 90), 768, 768, 0) != 90.0 or tall(head(0, 0, 10, 40)[:3], 1536, 768, 0) != 20.0:
        _fail(problems, "render zoom, to the head: the region's height is the median of its box's height over the frames it is in, on the canvas")
    # the run itself: the zoomed source says what set its zoom, and the report names the target
    n = 2
    rows = head(470, 300, 60, 90, frames=n)[:n]
    mask = torch.zeros(n, h, w)
    mask[:, 300:390, 470:530] = 1.0
    source = {"frames": torch.zeros(n, h, w, 3), "mask": mask, "region_boxes": rows, "render_region": vm.RENDER_ZOOM_REPLACED,
              "grow_pixels": 8, "feather_pixels": 2}
    z, said = vm.zoomed_source({**source, "zoom_up_to": most}, w, h)
    zoom = z.get("zoom") or {}
    if zoom.get("set_by") != by_head or not 2.0 < float(zoom.get("scale", 0)) <= 300 / 90 + 1e-9 or zoom.get("up_to") != most \
            or not said or f"height to {target} canvas px" not in said[0]:
        _fail(problems, f"render zoom, to the head: a zoomed source asking up to {most:g} records what set its zoom and what it asked, and the "
                        f"report says the target: {zoom}, {said[:1]}")
    plain = (vm.zoomed_source(source, w, h)[0].get("zoom") or {})
    if plain.get("scale") != 2.0 or plain.get("set_by") != vm.SET_BY_LARGEST or plain.get("up_to") != 2.0:
        _fail(problems, f"render zoom, to the head: a source that asks nothing is zoomed at the cap of 2 and says so: {plain}")
    # the node: `zoom_up_to` appended last, optional, 2 by default, and a value past the most refused in the node's words
    where = _appended("zoom_up_to")
    if where is None or not where.optional or where.default != 2.0 or where.max != most:
        _fail(problems, "render zoom, to the head: `zoom_up_to` must be the Masked Source's last input (`APPENDED`), optional, 2 by default, "
                        f"at most {most:g}")
    frames, region = torch.zeros(4, 48, 64, 3), torch.zeros(4, 48, 64)
    region[:, 20:28, 28:36] = 1.0
    table = json.dumps({"shots": [{"first_frame": 0, "last_frame": 3}]})
    try:
        asked = vm.MiniMaxH3MaskedSource.execute(frames, region, shot_table=table, render_region=vm.RENDER_ZOOM_REPLACED, zoom_up_to=4.0).args[0]
        if asked.get("zoom_up_to") != 4.0 or vm.MiniMaxH3MaskedSource.execute(frames, region, shot_table=table).args[0].get("zoom_up_to") != 2.0:
            _fail(problems, "render zoom, to the head: the Masked Source's record carries the zoom_up_to it was given, 2 when none")
        vm.MiniMaxH3MaskedSource.execute(frames, region, shot_table=table, zoom_up_to=most + 1.0)
        _fail(problems, f"render zoom, to the head: the Masked Source took a zoom_up_to past {most:g}")
    except (TypeError, ValueError) as exc:
        if "zoom_up_to" not in str(exc) or isinstance(exc, TypeError):
            _fail(problems, f"render zoom, to the head: the Masked Source's zoom_up_to: {type(exc).__name__}: {exc}")


def _render_zoom(problems):
    import importlib
    import math
    import tempfile
    # 1. the box: whole even pixels of the load's frames, the canvas's proportions exactly, inside what the
    #    canvas shows, round the subject with its room, never under what `largest` allows
    for old_w, old_h, width, height in ((1424, 1080, 1024, 768), (1024, 768, 1024, 768), (1920, 1080, 1344, 768), (356, 270, 256, 192)):
        cx, cy = vm._fit_crop(old_w, old_h, width, height)
        shown_w, shown_h = old_w - 2 * cx, old_h - 2 * cy
        mx, my = old_w // 2, old_h // 2
        cases = {"small, in the middle": ((mx - shown_w // 40, my - shown_h // 30, mx + shown_w // 40, my + shown_h // 30), 2.0, vm.SET_BY_LARGEST),
                 # from the first EVEN column and row the canvas shows: the box is even, so an odd crop's first
                 # column is the one pixel a box cannot reach
                 "small, in a corner": ((cx + cx % 2, cy + cy % 2, cx + shown_w // 20, cy + shown_h // 15), 2.0, vm.SET_BY_LARGEST),
                 "a quarter of the frame": ((mx - shown_w // 8, my - shown_h // 8, mx + shown_w // 8, my + shown_h // 8), 8.0, vm.SET_BY_BOX)}
        for name, (subject, largest, by) in cases.items():
            rows = torch.tensor([list(subject)] * 3 + [[-1] * 4])
            box, why = vm.zoom_box(rows, old_w, old_h, width, height, largest, any_load=largest)
            if box is None:
                _fail(problems, f"render zoom: {name} on {old_w}x{old_h} for {width}x{height} gave no box: {why}")
                continue
            x0, y0, x1, y1 = box
            scale = shown_w / (x1 - x0)
            if any(v % 2 for v in box) or (x1 - x0) * height != (y1 - y0) * width \
                    or x0 < cx or y0 < cy or x1 > old_w - cx or y1 > old_h - cy \
                    or x0 > subject[0] or y0 > subject[1] or x1 < subject[2] or y1 < subject[3]:
                _fail(problems, f"render zoom: {name} on {old_w}x{old_h} for {width}x{height}: the box {box} must be whole even "
                                f"pixels in the canvas's proportions, inside what the canvas shows and round the subject {subject}")
            if why != by or scale > largest + 1e-9 or scale < vm.ZOOM_SMALLEST:
                _fail(problems, f"render zoom: {name} on {old_w}x{old_h}: set by {why!r} at {scale:.3f} times; it is set by {by} "
                                f"and never over the largest zoom, {largest:g}")
            if by == vm.SET_BY_LARGEST and scale < largest * 0.9:
                _fail(problems, f"render zoom: {name} on {old_w}x{old_h}: a small subject comes to {scale:.3f} times under a cap of {largest:g}")
            if by == vm.SET_BY_BOX and ((x1 - x0) < (subject[2] - subject[0]) * (1 + 2 * vm.ZOOM_ROOM) - 1e-6
                                    or (y1 - y0) < (subject[3] - subject[1]) * (1 + 2 * vm.ZOOM_ROOM) - 1e-6):
                _fail(problems, f"render zoom: {name} on {old_w}x{old_h}: the box {box} leaves the subject less than its room")
        big = torch.tensor([[cx + shown_w // 10, cy + shown_h // 10, old_w - cx - shown_w // 10, old_h - cy - shown_h // 10]])
        if vm.zoom_box(big, old_w, old_h, width, height, 2.0)[0] is not None \
                or vm.zoom_box(torch.full((4, 4), -1), old_w, old_h, width, height, 2.0)[0] is not None:
            _fail(problems, f"render zoom: on {old_w}x{old_h} a subject that fills the frame, or one in no frame, was given a box")

    # 2. the source made of the box: handed frames twice the canvas each way, so a zoom of 2 is a plain cut
    n, big_h, big_w, width, height = 20, 96, 128, 64, 48
    frames = (torch.linspace(0.1, 0.5, big_w)[None, None, :, None] + torch.linspace(0.0, 0.3, big_h)[None, :, None, None]).expand(n, -1, -1, 3).clone()
    mask = torch.zeros(n, big_h, big_w)
    mask[:, 44:52, 60:72] = 1.0
    table = json.dumps({"shots": [{"first_frame": 0, "last_frame": n - 1}]})
    node = vm.MiniMaxH3MaskedSource
    rec = node.execute(frames, mask, grow_pixels=8, feather_pixels=2, shot_table=table,
                       render_region=vm.RENDER_ZOOM_REPLACED).args[0]
    if (rec.get("render_region"), rec.get("cuts")) != (vm.RENDER_ZOOM_REPLACED, []) or "largest_zoom" in rec or vm.ZOOM_LARGEST != 2.0:
        _fail(problems, f"render zoom: the record carries {rec.get('render_region')!r} and cuts {rec.get('cuts')}; it carries no "
                        "largest_zoom, which is the constant ZOOM_LARGEST (2.0 in this fixture's arithmetic)")
        return
    plain = {**rec, "render_region": vm.RENDER_WHOLE}
    same, said = vm.zoomed_source(plain, width, height)
    if same is not plain or said:
        _fail(problems, "render zoom: under `the whole frame` the source must come back as it went in, with nothing said")
    z, said = vm.zoomed_source(rec, width, height)
    zoom = z.get("zoom")
    if zoom is None or z.get("whole") is not rec or len(said) != 1 or "handed 128x96 frames for a 64x48 canvas" not in said[0] \
            or not said[0].startswith("ZOOMED 2.00x: "):
        _fail(problems, f"render zoom: a zoomed source must carry its zoom and the source as it came, and say what it was handed: {said}")
        return
    x0, y0, x1, y1 = zoom["box"]
    if (x1 - x0, y1 - y0) != (width, height) or abs(zoom["scale"] - 2.0) > 1e-9 or zoom["source_size"] != [big_w, big_h] \
            or zoom["canvas"] != [width, height] or zoom["set_by"] != vm.SET_BY_LARGEST:
        _fail(problems, f"render zoom: on frames twice the canvas a small subject under a cap of 2 is a {width}x{height} box at 2 times; got {zoom}")
        return
    if not torch.equal(z["frames"], frames[:, y0:y1, x0:x1]) or not torch.equal(z["mask"], mask[:, y0:y1, x0:x1]):
        _fail(problems, "render zoom: at a scale of one the zoomed frames and mask are the box cut out, value for value")
    want_box = torch.tensor([60 - x0, 44 - y0, 72 - x0, 52 - y0])
    if not torch.equal(z["region_boxes"][0], want_box) or not torch.equal(z["subject_boxes"][0], want_box) \
            or abs(float(z["subject_area"][0]) - 12 * 8 / (width * height)) > 1e-6:
        _fail(problems, f"render zoom: the box tables and the subject's area must be of the zoomed picture: {z['region_boxes'][0].tolist()}, "
                        f"{z['subject_boxes'][0].tolist()}, {float(z['subject_area'][0]):.4f}")
    fitted, fitted_mask, _held = vm.window_frames(z, 0, 9, width, height)
    if not torch.equal(fitted, z["frames"][:9]) or not torch.equal(fitted_mask, z["mask"][:9]):
        _fail(problems, "render zoom: a window of a zoomed source is its frames and mask as they are: the fit to the canvas must change nothing")
    again = vm.zoomed_source(rec, width, height)[0]
    if again["frames"] is z["frames"] or not torch.equal(again["frames"], z["frames"]) or not torch.equal(again["mask"], z["mask"]):
        _fail(problems, "render zoom: a second run makes the zoomed pictures again, the same pictures and not the same tensors: "
                        "nothing an earlier run made is kept")
    if vm.zoomed_source({**rec, "cuts": [7]}, width, height)[0].get("zoom") != z["zoom"]:
        _fail(problems, "render zoom: a source with a cut in its window must extend the shot's one zoom box across the window")
    for bad, word in (({**rec, "region_boxes": None}, "box on each frame"),
                      ({**rec, "render_region": "closer"}, "unknown render_region")):
        try:
            vm.zoomed_source(bad, width, height)
            _fail(problems, f"render zoom: a source that should be refused ({word}) was not")
        except ValueError as exc:
            if word not in str(exc):
                _fail(problems, f"render zoom: the refusal does not say {word!r}: {exc}")
    # A ZOOMED RUN CONTINUES NOTHING AND IS NOT CONTINUED: its kept frames would be of another framing. For a day
    # a zoomed run took the stored window's box; that went on 2026-10-10 with no render having done it
    other = {**zoom, "box": [x0 - 2, y0 - 2, x1 - 2, y1 - 2]}
    for record, stored, word in ((plain, zoom, "was rendered zoomed in"), (rec, None, "a zoomed run continues nothing"),
                                 (rec, vm.NO_REGION_FILE, "a zoomed run continues nothing"),
                                 (rec, other, "a zoomed run continues nothing"), (rec, zoom, "a zoomed run continues nothing")):
        try:
            vm.zoomed_source(record, width, height, stored)
            _fail(problems, f"render zoom: a continuation that should be refused ({word}) was not")
        except ValueError as exc:
            if word not in str(exc):
                _fail(problems, f"render zoom: the continuation's refusal does not say {word!r}: {exc}")
    # A ZOOMED RUN TAKES A WIRED MOTION VIDEO AS IT IS (2026-10-13; refused from 2026-10-10 until then). The zoom
    # does nothing to the video and the video does nothing to the zoom: the same box and pictures as with none
    # wired, the same reference a whole-frame run of the load is shown, to the bit, and a line of the report that
    # says the video was not cut. The control: the same video cut to the box is NOT that reference, so a cut
    # that crept back in would be seen here.
    wired = torch.rand(n, big_h, big_w, 3, generator=torch.Generator().manual_seed(7))
    with_video = dict(rec, motion_reference=vm.MOTION_WIRED, motion_frames=wired)
    try:
        zv, said_v = vm.zoomed_source(with_video, width, height)
    except ValueError as exc:
        _fail(problems, f"render zoom: a zoomed run with a wired motion video was refused: {exc}")
    else:
        if zv.get("zoom") != zoom or not torch.equal(zv["frames"], z["frames"]) or not torch.equal(zv["mask"], z["mask"]) \
                or not torch.equal(zv["region_boxes"], z["region_boxes"]):
            _fail(problems, "render zoom: a wired motion video changed the box or the zoomed pictures; the box is from the replaced part alone")
        shown = vm.wired_motion(zv, 4, 9, width, height)
        on_whole = vm.wired_motion({**with_video, "render_region": vm.RENDER_WHOLE}, 4, 9, width, height)
        if zv.get("motion_frames") is not wired or not torch.equal(shown, on_whole):
            _fail(problems, "render zoom: a zoomed window's wired motion video must be the one a whole-frame run of the load is shown, to the bit")
        cut = vm.wired_motion({**zv, "motion_frames": vm.crop_frames(wired, zoom["box"], big_w, big_h)}, 4, 9, width, height)
        if cut.shape != shown.shape or torch.equal(cut, shown):
            _fail(problems, "render zoom: the fixture cannot tell a wired video cut to the box from one handed on as it is")
        if len(said_v) != 2 or said_v[0] != said[0] or "motion video is NOT cut to the box" not in said_v[1]:
            _fail(problems, f"render zoom: a zoomed run with a wired motion video must say, on a line of its own, that the video was not cut: {said_v}")
        two, said_two = vm.zoomed_source(dict(with_video, motion_frames_2=wired.flip(0), motion_short_edge_2=32), width, height)
        if two.get("motion_frames_2") is None or len(said_two) != 2 or "motion videos are NOT cut to the box" not in said_two[1]:
            _fail(problems, f"render zoom: a second wired video on a zoomed run is carried and said as the first is: {said_two}")
    for stored in (None, vm.NO_REGION_FILE, vm.NOT_CONTINUED):
        if vm.zoomed_source(plain, width, height, stored)[0] is not plain:
            _fail(problems, f"render zoom: a whole-frame run continuing a whole-frame window ({stored!r}) must run as it always has")
    with tempfile.TemporaryDirectory() as tmp:
        latent = str(Path(tmp) / "run_window_1.safetensors")
        cells = torch.ones(1, height // 16, width // 16)
        if vm.stored_zoom(latent) != vm.NO_REGION_FILE:
            _fail(problems, "render zoom: a stored window with no region file must say so")
        vm.save_window_region(latent[:-len(".safetensors")] + "_region.npz", z["mask"][:1], cells, {"zoom": zoom}, 8, 0, 0)
        if vm.stored_zoom(latent) != zoom:
            _fail(problems, f"render zoom: a stored window's framing is its region file's: {vm.stored_zoom(latent)}")
        vm.save_window_region(latent[:-len(".safetensors")] + "_region.npz", z["mask"][:1], cells, {}, 8, 0, 0)
        if vm.stored_zoom(latent) is not None:
            _fail(problems, "render zoom: a stored window rendered on the whole frame must read as None")
    # the region reaching a side of the box that is inside the frame is cut off there, and named; a side that
    # is the frame's own edge is not
    cells = torch.zeros(2, height // 16, width // 16)
    cells[1, 1, 0] = 1.0             # the left column on a middle row, and the top row on a middle column: no corner
    cells[0, 0, 1] = 1.0
    at_edge = {**zoom, "box": [0, 0, width, height]}
    if vm.box_sides_cut(cells, zoom) != ["left", "top"] or vm.box_sides_cut(cells, at_edge) != [] \
            or vm.box_sides_cut(torch.zeros_like(cells), zoom) != []:
        _fail(problems, f"render zoom: the sides a region is cut at are {vm.box_sides_cut(cells, zoom)} and, with the box in the "
                        f"frame's corner, {vm.box_sides_cut(cells, at_edge)}; they are left and top, and none")
    wide = torch.zeros(n, big_h, big_w)
    wide[:, 10:86, 12:116] = 1.0
    filled = node.execute(frames, wide, grow_pixels=8, feather_pixels=2, shot_table=table,
                          render_region=vm.RENDER_ZOOM_REPLACED).args[0]
    back, said = vm.zoomed_source(filled, width, height)
    if back is not filled or len(said) != 1 or "rendered on the whole frame" not in said[0]:
        _fail(problems, f"render zoom: a subject that fills the frame is rendered on the whole frame, and the report says so: {said}")
    where = _appended("render_region")
    if where is None or not where.optional or where.default != vm.RENDER_WHOLE \
            or vm.RENDER_REGIONS != (vm.RENDER_WHOLE, vm.RENDER_ZOOM_REPLACED) \
            or any(i.id == "largest_zoom" for i in node.define_schema().inputs) \
            or any(hasattr(vm, name) for name in ("RENDER_ZOOM_SUBJECT", "ZOOM_LARGEST_MAX", "CONTINUED_BOX")):
        _fail(problems, "render zoom: `render_region` must be where it was appended (`APPENDED`), optional, the whole frame by "
                        "default, with one zoomed choice; the largest zoom is a constant and no input")
    try:
        node.execute(frames, mask, shot_table=table, largest_zoom=3.0)
        _fail(problems, "render zoom: the node still takes a `largest_zoom`")
    except TypeError:
        pass
    for kw, word in ((dict(render_region="closer"), "render_region"), (dict(render_region="zoomed in on the tracked subject"), "render_region")):
        try:
            node.execute(frames, mask, shot_table=table, **kw)
            _fail(problems, f"render zoom: the Masked Source took {kw}")
        except ValueError as exc:
            if word not in str(exc):
                _fail(problems, f"render zoom: the Masked Source's refusal of {kw} does not name {word}: {exc}")

    # 3. the lay back, on a box that lands between canvas pixels: nothing shifts, the plate is the whole
    #    frame's own wherever the weight is 0, and a weight of 1 is the decode
    old_w, old_h, width, height = 356, 270, 256, 192
    yy, xx = torch.meshgrid(torch.arange(old_h, dtype=torch.float32), torch.arange(old_w, dtype=torch.float32), indexing="ij")
    picture = 0.5 + 0.25 * torch.sin(xx / 17.0) * torch.cos(yy / 13.0) + 0.2 * torch.sin((xx + 2.0 * yy) / 29.0)
    frames = picture[None, :, :, None].expand(3, -1, -1, 3).clone()
    rows = torch.tensor([[150, 110, 190, 150]] * 3)
    box, _why = vm.zoom_box(rows, old_w, old_h, width, height, 2.0)
    cx, _cy = vm._fit_crop(old_w, old_h, width, height)
    zoom = {"box": list(box), "source_size": [old_w, old_h], "canvas": [width, height], "scale": (old_w - 2 * cx) / (box[2] - box[0]),
            "render_region": vm.RENDER_ZOOM_REPLACED, "set_by": vm.SET_BY_LARGEST}
    fx0, fy0, fx1, fy1 = vm._canvas_box(zoom)
    if all(float(v).is_integer() for v in (fx0, fy0, fx1, fy1)):
        _fail(problems, f"render zoom: the fixture's box lands on whole canvas pixels ({fx0}, {fy0}); the case below needs one that does not")
    whole = vm.fit_frames(frames, width, height)
    zoomed = vm.crop_frames(frames, box, width, height)
    alpha = torch.zeros(3, height, width)
    alpha[:, 50:150, 60:200] = 1.0
    laid = vm.lay_back(zoomed, alpha, whole, zoom)
    err = float((laid - whole).abs().max())
    shifted = float((vm.lay_back(zoomed, alpha, whole, {**zoom, "box": [box[0] + 4, box[1], box[2] + 4, box[3]]}) - whole).abs().max())
    if err > 0.02 or err > 0.5 * shifted:
        _fail(problems, f"render zoom: the zoomed plate laid back is {err:.4f} from the whole frame's own at most (a box 4 px off "
                        f"gives {shifted:.4f}); the lay back must put the picture where it was cut from")
    # the rectangle's place on the canvas, a pixel of slack each way for the filter's reach
    k = (fx1 - fx0) / width
    rx0, ry0, rx1, ry1 = fx0 + 60 * k, fy0 + 50 * k, fx0 + 200 * k, fy0 + 150 * k
    outside = torch.ones(height, width, dtype=torch.bool)
    outside[max(0, math.floor(ry0) - 2):math.ceil(ry1) + 2, max(0, math.floor(rx0) - 2):math.ceil(rx1) + 2] = False
    if not torch.equal(laid[:, outside], whole[:, outside]) or not torch.equal(vm.lay_back(zoomed, torch.zeros_like(alpha), whole, zoom), whole):
        _fail(problems, "render zoom: where the weight is 0 the laid-back frame must be the whole frame's own pixels, bit for bit, "
                        "inside the box as well as outside it")
    flat = vm.lay_back(torch.full_like(zoomed, 0.9), alpha, whole, zoom)
    inner = flat[:, math.ceil(ry0) + 2:math.floor(ry1) - 2, math.ceil(rx0) + 2:math.floor(rx1) - 2]
    if float((inner - 0.9).abs().max()) > 1e-4:
        _fail(problems, f"render zoom: under a weight of 1 the laid-back frame is the decode; it is {float((inner - 0.9).abs().max()):.5f} off")
    try:
        vm.lay_back(zoomed, alpha[:2], whole, zoom)
        _fail(problems, "render zoom: a weight that covers other frames than the decode was laid")
    except ValueError:
        pass
    # THE EXACT MATTE (the lead's ruling of 2026-10-16): the weight lay_back returns is the weight it blended with, at the whole canvas.
    # A white decode over a black frame lays the weight itself, so the laid picture and the matte agree to a level
    soft = vm.pixel_alpha(torch.ones(1, height // 16, width // 16) * (torch.rand(1, height // 16, width // 16) > 0.5), height, width, 6)[:1].expand(3, -1, -1)
    laid_w, matte = vm.lay_back(torch.ones_like(zoomed), soft.contiguous(), torch.zeros_like(whole), zoom, weight=True)
    if matte.dtype != torch.uint8 or tuple(matte.shape) != tuple(whole.shape[:3]):
        _fail(problems, f"render zoom: the matte is {matte.dtype} {tuple(matte.shape)}, not uint8 over the whole frames {tuple(whole.shape[:3])}")
    elif float((laid_w[..., 0] * 255.0 - matte.float()).abs().max()) > 1.0 or not bool((matte > 0).any()):
        # THE CASE THAT GOES RED when the saved matte is not the weight the window's video was blended with
        _fail(problems, f"render zoom: the matte is {float((laid_w[..., 0] * 255.0 - matte.float()).abs().max()):.2f} levels from the weight the frame was laid with")
    if not torch.equal(vm.lay_back(torch.ones_like(zoomed), soft.contiguous(), torch.zeros_like(whole), zoom), laid_w):
        _fail(problems, "render zoom: asking for the matte changed the frame lay_back lays")
    with tempfile.TemporaryDirectory() as tmp:
        back = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "m.npz"), alpha, torch.ones(1, height // 16, width // 16), {}, 8, 0, 0,
                                                           matte=matte))
        if "matte" not in back or float((back["matte"] * 255.0 - matte.float()).abs().max()) > 0.0:
            _fail(problems, "render zoom: the matte saved with the window's region does not come back as it was saved")
        try:
            vm.save_window_region(str(Path(tmp) / "bad.npz"), alpha, torch.ones(1, height // 16, width // 16), {}, 8, 0, 0, matte=matte[:2])
            _fail(problems, "render zoom: a matte that covers other frames than the window's was saved")
        except ValueError:
            pass
    song_text = (REPO / "audio_freeze_song.py").read_text()
    for words in ("images, matte = video_mask.lay_back(decoded, alpha, video_mask.window_frames(", "zoom, weight=True)",
                  "matte = video_mask.matte_of(alpha)", "kept=kept_weight, matte=matte,",
                  'piece={"stem": stem, "first": int(round(w.start * FPS)) + int(trim) - head, "written": written})'):
        if words not in song_text:
            _fail(problems, f"render zoom: the song node does not save the weight its window's video was blended with ({words!r} not in it)")

    # 4. the region file says the box, and a file with none says None
    with tempfile.TemporaryDirectory() as tmp:
        tokens = torch.ones(1, height // 16, width // 16)
        settings = {"composite": vm.COMPOSITE_REGION, "feather_pixels": 0, "zoom": zoom}
        got = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "z.npz"), alpha[:1], tokens, settings, 8, 0, 0))
        bare = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "w.npz"), alpha[:1], tokens, {"composite": vm.COMPOSITE_REGION}, 8, 0, 0))
        if got["zoom"] != zoom or bare["zoom"] is not None or "zoom" in got["source"]:
            _fail(problems, f"render zoom: a region file must hand back the box its window was rendered in, and None for a whole-frame window: {got['zoom']}, {bare['zoom']}")


def check_held_tail(problems):
    """Item 23. What a window is given past the source's last frame: that frame, with its mask held so the region
    stays open. The plate tail and the input that chose it were removed on 2026-10-10."""
    node = vm.MiniMaxH3MaskedSource
    have = sum(vm.run_lengths(3)) + 2                 # the source ends two frames into the window's fourth latent step
    frames = torch.rand(have, H, W, 3)
    mask = torch.zeros(have, H, W)
    mask[:, 32:64, 48:80] = 1.0
    mask[-1] = 0.0
    mask[-1, 64:96, 80:112] = 1.0                     # the last frame's subject is somewhere else: the tail must be THIS one
    keep = torch.zeros(have, H, W)
    keep[-1, 70:74, 90:94] = 1.0
    shape = (LATENT_T, LAT_H, LAT_W)
    record = node.execute(frames, mask, grow_pixels=8, keep=keep).args[0]
    if "held_tail" in record or any(i.id == "held_tail" for i in node.define_schema().inputs) \
            or any(hasattr(vm, name) for name in ("TAIL_PLATE", "TAIL_OPEN", "TAILS")):
        problems.append("held tail: the input, its record key or its choices are back; the open tail is the only behaviour")
    try:
        node.execute(frames, mask, held_tail="the last frame, as the source's")
        problems.append("held tail: the node still takes a `held_tail`")
    except TypeError:
        pass
    got = vm.window(record, 0, FRAMES, W, H, *shape)
    real = vm.window(record, 0, have, W, H, 3, LAT_H, LAT_W) if have == sum(vm.run_lengths(3)) else None
    if got[4] != FRAMES - have or not torch.equal(got[0][have:], got[0][have - 1:have].expand(FRAMES - have, -1, -1, -1)):
        problems.append("held tail: the frames past the source's end must be its last frame, held, and counted")
    last = vm.fit_mask(mask[-1:], W, H)[0]
    if not all(torch.equal(got[3][f], last) for f in range(have, FRAMES)):
        problems.append("held tail: the held frames must carry the LAST frame's mask, so the region stays open on them")
    alone = vm.token_mask(vm.grow(last[None], 8), 1, LAT_H, LAT_W)[0]
    held_keep = vm.token_mask(vm.fit_mask(keep[-1:], W, H), 1, LAT_H, LAT_W)[0]
    want = alone * (1.0 - held_keep)
    if not bool(want.any()) or not bool(held_keep.any()) or not all(torch.equal(got[2][k], want) for k in range(4, LATENT_T)):
        problems.append("held tail: on the steps wholly past the source's end the region must be the last frame's own, with what "
                        "`keep` held on that frame still kept")
    del real
    each = dict(record, grow_by=vm.GROW_SUBJECT, subject_area=vm.area_share(mask), feather_pixels=2)
    margins = vm.source_margins(each, 0, FRAMES, W * H)
    if not torch.is_tensor(margins) or int(margins[-1]) != int(margins[have - 1]):
        problems.append("held tail: under the subject's size the held frames must take the last frame's margin")
    boxes = vm.window_boxes(dict(record, replace=vm.REPLACE_PARTS), 0, FRAMES, W, H)
    if boxes is None or not torch.equal(boxes[have:], boxes[have - 1:have].expand(FRAMES - have, -1)) or int(boxes[-1, 0]) < 0:
        problems.append("held tail: the held frames must carry the last frame's box, as they carry its mask")
    # a held frame has a mask of its own, so the composite lays it: the no-mask rule names none of them
    across, bare = vm.unlaid_frames(got[3], got[2], [], 0)
    if any(f >= have for f in across + bare):
        problems.append(f"held tail: the composite would leave held frames {[f for f in across + bare if f >= have]} as the source")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if '"its last frame is held with its region open"' not in song or "held frame(s) are INSIDE the track and are written" not in song:
        problems.append("held tail: the song node's report does not say the held frames keep their region open, or that some "
                        "of them are inside the track and written")

    # A 35-frame shot in a 39-frame window (`C38`, 2026-10-21): `MiniMaxH3MaskedSource.execute` pads 2 context frames
    # before and 2 after (`pad_before=2, pad_after=2`) with empty mask, keeping all 39 real source frames (`held == 0`),
    # and `save_window_region` / `load_window_region` record `pad_before=2, pad_after=2` while storing the 35-frame mask.
    frames_39 = torch.rand(39, H, W, 3)
    mask_35 = torch.zeros(35, H, W)
    mask_35[:, 32:64, 48:80] = 1.0
    motion_35 = torch.linspace(0.1, 0.9, 35).view(35, 1, 1, 1).expand(35, H, W, 3).contiguous()
    from types import SimpleNamespace
    prompt_39 = {"10": {"_meta": {"pad_before": 2, "pad_after": 2}}}
    node.hidden = SimpleNamespace(prompt=prompt_39, unique_id="10")
    try:
        rec_39 = node.execute(frames_39, mask_35, grow_pixels=8, motion_reference=vm.MOTION_WIRED,
                              motion_video=motion_35).args[0]
    finally:
        node.hidden = None
    if (rec_39.get("pad_before"), rec_39.get("pad_after")) != (2, 2) or rec_39["mask"].shape[0] != 39:
        problems.append(f"context padding (35 in 39): expected pad_before=2, pad_after=2 and 39 mask frames, got {rec_39.get('pad_before')}, {rec_39.get('pad_after')}, {rec_39['mask'].shape}")
    if bool(rec_39["mask"][:2].any()) or bool(rec_39["mask"][37:].any()) or not bool(rec_39["mask"][2:37].all(dim=0).any()):
        problems.append("context padding (35 in 39): context frames before (0..1) and after (37..38) must have empty mask and shot frames (2..36) keep the mask")
    if rec_39["motion_frames"].shape[0] != 39 or not torch.equal(rec_39["motion_frames"][0], motion_35[0]) or not torch.equal(rec_39["motion_frames"][-1], motion_35[-1]):
        problems.append("context padding (35 in 39): motion_video was not padded to 39 frames by repeating its first and last frame")
    win_39 = vm.window(rec_39, 0, 39, W, H, 12, LAT_H, LAT_W)
    if win_39[4] != 0:
        problems.append(f"context padding (35 in 39): held tail must be 0 when 39 real source frames are loaded, got {win_39[4]}")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = vm.save_window_region(str(Path(tmp) / "w1_region.npz"), win_39[3], win_39[2], rec_39, 8, 0, 2)
        loaded_reg = vm.load_window_region(reg_path)
        if (loaded_reg["pad_before"], loaded_reg["pad_after"], loaded_reg["trim"], int(loaded_reg["mask"].shape[0])) != (2, 2, 2, 39):
            problems.append(f"context padding (35 in 39): saved region must carry pad_before=2, pad_after=2, trim=2 and 39 window frames, got {loaded_reg['pad_before']}, {loaded_reg['pad_after']}, {loaded_reg['trim']}, {loaded_reg['mask'].shape}")


def check_context_shows(problems):
    """Item 24. What the real frames a short load is padded with show (`context_shows`, 2026-10-22): `as filmed`, nothing
    held there; `face covered`, the shot's first frame's mask held over the frames before and its last frame's over the
    frames after, the keep-outs empty in both forms; a load that fills its window unchanged by either; and the form, with
    the held frames that lie across a cut, in the record, each window's region file and the plan's rows."""
    import tempfile
    from types import SimpleNamespace
    if getattr(vm, "CONTEXT_SHOWS", None) != ("as filmed", "face covered") or not hasattr(vm, "pad_context"):
        problems.append("context: the node has no `context_shows` choice of `as filmed` and `face covered` (`CONTEXT_SHOWS`, `pad_context`)")
        return
    where = _appended("context_shows")
    if where is None or not where.optional or where.default != vm.CONTEXT_AS_FILMED or list(where.options) != list(vm.CONTEXT_SHOWS):
        problems.append("context: `context_shows` is not appended last, optional, of the two plain-word choices, `as filmed` by default")
    # the pure padding: a shot of 3 frames whose subject moves, 2 frames before it and 1 after
    shot = torch.zeros(3, 4, 4)
    shot[0, 0, 0], shot[1, 1, 1], shot[2, 2, 2] = 1.0, 1.0, 1.0
    filmed, covered = vm.pad_context(shot, 2, 1), vm.pad_context(shot, 2, 1, vm.CONTEXT_FACE_COVERED)
    if tuple(filmed.shape) != (6, 4, 4) or bool(filmed[:2].any()) or bool(filmed[5:].any()) or not torch.equal(filmed[2:5], shot):
        problems.append("context: `as filmed` must pad the shot's mask with empty frames before and after and leave its own frames")
    if not (torch.equal(covered[0], shot[0]) and torch.equal(covered[1], shot[0]) and torch.equal(covered[5], shot[2])
            and torch.equal(covered[2:5], shot)):
        problems.append("context: `face covered` must hold the shot's FIRST frame's mask over the frames before and its LAST "
                        "frame's over the frames after, and leave its own frames")
    try:
        vm.pad_context(shot, 2, 1, "covered")
        problems.append("context: an unknown `context_shows` was padded, not refused")
    except ValueError:
        pass
    # which held frames lie across a cut: cuts on the shot's own first frame and on the frame after its last
    if vm.context_across_cut([2, 5], 2, 3, 1) != [0, 1, 5] or vm.context_across_cut([], 2, 3, 1) != [] \
            or vm.context_across_cut([1], 2, 3, 1) != [0]:
        problems.append(f"context: the held frames across a cut are wrong: {vm.context_across_cut([2, 5], 2, 3, 1)}, "
                        f"{vm.context_across_cut([1], 2, 3, 1)}")
    # the node: a 35-frame shot in a 39-frame window, 2 before and 2 after, the subject moving so first and last differ
    node = vm.MiniMaxH3MaskedSource
    frames_39 = torch.rand(39, H, W, 3)
    mask_35 = torch.zeros(35, H, W)
    for k in range(35):
        mask_35[k, 32:64, 20 + 2 * k:52 + 2 * k] = 1.0
    others_35 = torch.zeros(35, H, W)
    others_35[:, 90:110, 150:170] = 1.0
    table = json.dumps({"shots": [{"first_frame": 0, "last_frame": 1}, {"first_frame": 2, "last_frame": 36},
                                  {"first_frame": 37, "last_frame": 38}]})
    node.hidden = SimpleNamespace(prompt={"10": {"_meta": {"pad_before": 2, "pad_after": 2}}}, unique_id="10")
    try:
        recs = {shows: node.execute(frames_39, mask_35, grow_pixels=8, others=others_35, shot_table=table,
                                    context_shows=shows).args[0] for shows in vm.CONTEXT_SHOWS}
        plain = node.execute(frames_39, mask_35, grow_pixels=8, others=others_35, shot_table=table).args[0]
    finally:
        node.hidden = None
    filmed, covered = recs[vm.CONTEXT_AS_FILMED], recs[vm.CONTEXT_FACE_COVERED]
    if not torch.equal(plain["mask"], filmed["mask"]) or plain.get("context_shows") != vm.CONTEXT_AS_FILMED:
        problems.append("context: a node given no `context_shows` must pad as `as filmed` does and say so")
    if bool(filmed["mask"][:2].any()) or bool(filmed["mask"][37:].any()):
        problems.append("context: `as filmed` must leave the context frames' mask empty")
    if not (all(torch.equal(covered["mask"][k], mask_35[0]) for k in (0, 1))
            and all(torch.equal(covered["mask"][k], mask_35[-1]) for k in (37, 38))
            and torch.equal(covered["mask"][2:37], filmed["mask"][2:37])):
        problems.append("context: `face covered` must hold the first shot frame's mask on frames 0..1 and the last's on 37..38 "
                        "and change no frame of the shot")
    boxes = covered["subject_boxes"]
    if not (torch.equal(boxes[0], boxes[2]) and torch.equal(boxes[38], boxes[36])) or int(filmed["subject_boxes"][0, 0]) >= 0:
        problems.append("context: the tracked subject's box must be held with the mask under `face covered`, and absent `as filmed`")
    if bool(covered["others"][:2].any()) or bool(covered["others"][37:].any()):
        problems.append("context: the keep-outs must stay empty on the context frames under `face covered` as `as filmed`")
    if covered.get("context_shows") != vm.CONTEXT_FACE_COVERED or covered.get("context_held_across_a_cut") != [0, 1, 37, 38] \
            or "context_held_across_a_cut" in filmed:
        problems.append(f"context: the record must say `face covered` and the held frames across a cut, [0, 1, 37, 38]: "
                        f"{covered.get('context_shows')!r}, {covered.get('context_held_across_a_cut')!r}")
    lt = next(t for t in range(1, 40) if sum(vm.run_lengths(t)) == 39)
    early = [s for s in range(lt) if sum(vm.run_lengths(lt)[:s + 1]) <= 2]      # latent steps wholly on the frames before
    win = {shows: vm.window(rec, 0, 39, W, H, lt, LAT_H, LAT_W) for shows, rec in recs.items()}
    if not early or any(bool(win[vm.CONTEXT_AS_FILMED][2][s].any()) for s in early) \
            or not all(bool(win[vm.CONTEXT_FACE_COVERED][2][s].any()) for s in early):
        problems.append(f"context: the latent step(s) {early} wholly on the frames before the shot must regenerate under `face "
                        "covered` and nothing `as filmed`")
    # the region file and the plan's rows say which form the context took; the plan's mask frames are the shot's
    afs = importlib.import_module("_h3pack.audio_freeze_song")
    for shows, rec in recs.items():
        with tempfile.TemporaryDirectory() as tmp:
            got = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "w_region.npz"), win[shows][3], win[shows][2],
                                                              rec, 8, 0, 2))
            if got.get("context_shows") != shows or (shows == vm.CONTEXT_FACE_COVERED) != ("context_held_across_a_cut" in got):
                problems.append(f"context: the region file of a `{shows}` window reads back {got.get('context_shows')!r}, "
                                f"{got.get('context_held_across_a_cut')!r}")
            one = SimpleNamespace(number=1, start=0.0, frames=39, text="t")
            afs.write_plan(rec, [one], [35], 0, False, W, H, tmp, "c", {})
            plan = json.loads(Path(afs.loop_resume.plan_path(tmp, "c")).read_text())
            row = plan["windows"][0]
            if row.get("context_shows") != shows or plan["frames_with_the_mask"] != [[0, 34]]:
                problems.append(f"context: the `{shows}` plan says {row.get('context_shows')!r} and the mask on "
                                f"{plan['frames_with_the_mask']}; the shot's own frames are [[0, 34]]")
    # a load that fills its window: unchanged by either form, and no context in the record
    full = {shows: node.execute(frames_39, torch.cat([mask_35[:2], mask_35, mask_35[-2:]]), grow_pixels=8,
                                context_shows=shows).args[0] for shows in vm.CONTEXT_SHOWS}
    if not torch.equal(full[vm.CONTEXT_AS_FILMED]["mask"], full[vm.CONTEXT_FACE_COVERED]["mask"]) \
            or any("context_shows" in r or "pad_before" in r for r in full.values()):
        problems.append("context: a load with no context frames must come out the same under either form, with no context key")
    try:
        node.execute(frames_39, mask_35, grow_pixels=8, context_shows="the face covered")
        problems.append("context: the node took an unknown `context_shows`")
    except ValueError as exc:
        if "`as filmed` or `face covered`" not in str(exc):
            problems.append(f"context: the refusal does not name the two choices in plain words: {exc}")
    # an edge frame with no mask holds nothing, and the record says which side (data, never a refusal)
    late = mask_35.clone()
    late[-3:] = 0.0                                    # the subject has left by the shot's last frame
    node.hidden = SimpleNamespace(prompt={"10": {"_meta": {"pad_before": 2, "pad_after": 2}}}, unique_id="10")
    try:
        gone = node.execute(frames_39, late, grow_pixels=8, context_shows=vm.CONTEXT_FACE_COVERED).args[0]
    finally:
        node.hidden = None
    if gone.get("context_held_empty") != ["after"] or bool(gone["mask"][37:].any()) or not torch.equal(gone["mask"][0], late[0]):
        problems.append(f"context: a last frame with no mask must hold nothing after it and say `after`: {gone.get('context_held_empty')!r}")
    # a region file saved again from one read keeps every padding key (`padding_of`; a re-composite once dropped two of them)
    with tempfile.TemporaryDirectory() as tmp:
        first = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "a.npz"), win[vm.CONTEXT_FACE_COVERED][3],
                                                            win[vm.CONTEXT_FACE_COVERED][2], {**covered, "context_held_empty": ["after"],
                                                                                               "held_before": 1, "held_after": 2,
                                                                                               "held_fell_back": ["after: a check"]}, 8, 0, 2))
        again = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "b.npz"), first["mask"], first["tokens"],
                                                            {**first["source"], **vm.padding_of(first)}, 8, 0, 2))
        if vm.padding_of(again) != vm.padding_of(first) or set(vm.padding_of(first)) != set(vm.PADDING_KEYS):
            problems.append(f"context: a region saved again with `padding_of` lost a key: {vm.padding_of(first)} -> {vm.padding_of(again)}")
    check_held_context(problems, frames_39, mask_35, others_35, afs)


def check_held_context(problems, frames_39, mask_35, others_35, afs):
    """Context frames across a cut are held copies of the shot's edge frame (`video_mask.hold_context`, the owner's "seam
    problem" of 2026-10-22): the 35-frame shot in its 39-frame window, 2 context frames each side. The builder marks the
    held ones in `_meta` (`held_before`, `held_after`; `workflows/masked_job_graph.py::held_context`, held in
    `bench/check_masked_job.py`). Red on the node before it: the context frames stayed the other shot's real frames."""
    import tempfile
    from types import SimpleNamespace
    node = vm.MiniMaxH3MaskedSource

    def run(meta, shows=vm.CONTEXT_AS_FILMED):
        node.hidden = SimpleNamespace(prompt={"10": {"_meta": meta}}, unique_id="10")
        try:
            return node.execute(frames_39, mask_35, grow_pixels=8, others=others_35, context_shows=shows).args[0]
        finally:
            node.hidden = None

    # a whole shot, a cut on each side: every context frame held, with the edge frame's masks, whatever context_shows says
    for shows in vm.CONTEXT_SHOWS:
        rec = run({"pad_before": 2, "pad_after": 2, "held_before": 2, "held_after": 2}, shows)
        f, m, o = rec["frames"], rec["mask"], rec["others"]
        if not (all(torch.equal(f[k], frames_39[2]) for k in (0, 1)) and all(torch.equal(f[k], frames_39[36]) for k in (37, 38))
                and torch.equal(f[2:37], frames_39[2:37])):
            problems.append(f"held: under `{shows}` a whole shot's context frames must be copies of its first frame before and its "
                            "last after, its own frames unchanged")
        if not (all(torch.equal(m[k], m[2]) and torch.equal(o[k], o[2]) for k in (0, 1))
                and all(torch.equal(m[k], m[36]) and torch.equal(o[k], o[36]) for k in (37, 38)) and bool(m[0].any())):
            problems.append(f"held: under `{shows}` a held frame must carry its edge frame's mask (the region open) and keep-outs")
        if not torch.equal(rec["subject_boxes"][0], rec["subject_boxes"][2]) or (rec.get("held_before"), rec.get("held_after")) != (2, 2):
            problems.append(f"held: under `{shows}` the box must be held with the mask and the record say 2 and 2: "
                            f"{rec.get('held_before')!r}, {rec.get('held_after')!r}")
    # a load inside a longer shot that ends after it: before is the shot's own footage (real), after is held
    rec = run({"pad_before": 2, "pad_after": 2, "held_before": 0, "held_after": 2})
    if not torch.equal(rec["frames"][:2], frames_39[:2]) or bool(rec["mask"][:2].any()) \
            or not all(torch.equal(rec["frames"][k], frames_39[36]) for k in (37, 38)) or "held_before" in rec:
        problems.append("held: a side where the shot continues must keep its real frames, as filmed, and only the cut side be held")
    # one side partly real: the held frame copies the innermost real context frame, with its (empty) mask
    rec = run({"pad_before": 2, "pad_after": 2, "held_before": 1, "held_after": 0})
    if not (torch.equal(rec["frames"][0], frames_39[1]) and torch.equal(rec["frames"][1], frames_39[1])
            and torch.equal(rec["frames"][37:], frames_39[37:]) and not bool(rec["mask"][:2].any())):
        problems.append("held: a held frame outside a real context frame must copy that frame, with its mask")
    # a graph built before the change names no held frames: real frames, as it was rendered
    rec = run({"pad_before": 2, "pad_after": 2})
    if not torch.equal(rec["frames"], frames_39) or "held_before" in rec or "held_after" in rec:
        problems.append("held: a graph that names no held frames must keep every context frame real")
    try:
        run({"pad_before": 2, "pad_after": 2, "held_before": 3, "held_after": 0})
        problems.append("held: more held frames than context frames on a side was taken, not refused")
    except ValueError as exc:
        if "is more than the window's context frames" not in str(exc):
            problems.append(f"held: the refusal of too many held frames does not say so in its own words: {exc}")
    # the region file and the plan row say how many were held, side by side
    rec = run({"pad_before": 2, "pad_after": 2, "held_before": 2, "held_after": 1})
    lt = next(t for t in range(1, 40) if sum(vm.run_lengths(t)) == 39)
    win = vm.window(rec, 0, 39, W, H, lt, LAT_H, LAT_W)
    with tempfile.TemporaryDirectory() as tmp:
        got = vm.load_window_region(vm.save_window_region(str(Path(tmp) / "h_region.npz"), win[3], win[2], rec, 8, 0, 2))
        afs.write_plan(rec, [SimpleNamespace(number=1, start=0.0, frames=39, text="t")], [35], 0, False, W, H, tmp, "h", {})
        row = json.loads(Path(afs.loop_resume.plan_path(tmp, "h")).read_text())["windows"][0]
    if (got.get("held_before"), got.get("held_after"), row.get("held_before"), row.get("held_after")) != (2, 1, 2, 1):
        problems.append(f"held: the region file and the plan row must say 2 held before and 1 after: {got.get('held_before')!r}, "
                        f"{got.get('held_after')!r}, {row.get('held_before')!r}, {row.get('held_after')!r}")
    # THE GUARD (`video_mask.HELD_EDGE_LEAST`, the owner's choice of 2026-10-22; updated 2026-10-24, from second_hfp_4912e's empty 4946):
    # a held side whose edge frame has no region, or one under that share of the shot's middle region, holds the nearest good shot frame
    # into the held padding slice, leaves the kept shot frames untouched, keeps `held_after` active, and says which shot frame was held
    def edged(last):
        m = mask_35.clone()
        m[-1] = 0.0
        m[-1, 32:32 + last, 60:60 + last] = 1.0 if last else 0.0
        node.hidden = SimpleNamespace(prompt={"10": {"_meta": {"pad_before": 2, "pad_after": 2, "held_before": 2, "held_after": 2}}},
                                      unique_id="10")
        try:
            return node.execute(frames_39, m, grow_pixels=8, others=others_35).args[0]
        finally:
            node.hidden = None
    middle = 32 * 32
    small = int((vm.HELD_EDGE_LEAST * middle) ** 0.5) - 1          # a square edge region just under the guard's share
    large = int((vm.HELD_EDGE_LEAST * middle) ** 0.5) + 2          # and just over it
    for last, what in ((0, "an empty"), (small, "a nearly empty")):
        rec = edged(last)
        if not (all(torch.equal(rec["frames"][k], frames_39[35]) for k in (37, 38))
                and all(torch.equal(rec["mask"][k], rec["mask"][35]) for k in (37, 38))
                and all(torch.equal(rec["others"][k], rec["others"][35]) for k in (37, 38))
                and bool(rec["mask"][37:].any())
                and torch.equal(rec["frames"][36], frames_39[36])
                and int((rec["mask"][36] > 0.5).sum()) == last * last
                and all(torch.equal(rec["frames"][k], frames_39[2]) for k in (0, 1))):
            problems.append(f"held: {what} last frame must hold shot frame 33 (window index 35) into the after padding while leaving kept frame 36 untouched")
        said = rec.get("held_fell_back") or []
        if (rec.get("held_before"), rec.get("held_after")) != (2, 2) or len(said) != 1 or not said[0].startswith("after") \
                or f"is {last * last} px" not in said[0] or "hold shot frame 33 (1024 px)" not in said[0]:
            problems.append(f"held: {what} last frame's step-back to shot frame 33 is not recorded with active held_after=2: {said!r}, "
                            f"{rec.get('held_before')!r}, {rec.get('held_after')!r}")
    rec = edged(large)
    if rec.get("held_after") != 2 or rec.get("held_fell_back") or not torch.equal(rec["frames"][38], frames_39[36]):
        problems.append(f"held: a last frame with a region over the guard's share ({large * large} of {middle} px) must be held")
    with tempfile.TemporaryDirectory() as tmp:
        fb = edged(0)
        afs.write_plan(fb, [SimpleNamespace(number=1, start=0.0, frames=39, text="t")], [35], 0, False, W, H, tmp, "f", {})
        row = json.loads(Path(afs.loop_resume.plan_path(tmp, "f")).read_text())["windows"][0]
    if not (row.get("held_fell_back") or [""])[0].startswith("after") or "hold shot frame 33" not in row["held_fell_back"][0] \
            or row.get("held_after") != 2:
        problems.append(f"held: the plan row does not record held_after=2 and the step-back to shot frame 33: {row!r}")
    # second_hfp_4912e's actual tail (35 frames in 39, pad_before=0, pad_after=4, held_before=0, held_after=4):
    # shot frames 0..29 have 9147 px, shot frame 30 has 2958 px (>= 0.25 * 9147 = 2286.75 px), shot frames 31..33 have
    # 2100, 1500, 934 px (< 2286.75 px), and shot frame 34 has 0 px; steps back over 34, 33, 32, 31 to shot frame 30,
    # holding frames_39[30] into 35..38 while leaving kept frames 31..34 untouched
    m_4912e = torch.zeros(35, H, W)
    for k, px in [(i, 9147) for i in range(30)] + [(30, 2958), (31, 2100), (32, 1500), (33, 934), (34, 0)]:
        m_4912e[k].view(-1)[:px] = 1.0
    node.hidden = SimpleNamespace(prompt={"10": {"_meta": {"pad_before": 0, "pad_after": 4, "held_before": 0, "held_after": 4}}},
                                  unique_id="10")
    try:
        rec_4912e = node.execute(frames_39, m_4912e, grow_pixels=8, others=others_35).args[0]
    finally:
        node.hidden = None
    if not (rec_4912e.get("held_after") == 4
            and all(torch.equal(rec_4912e["frames"][k], frames_39[30]) for k in range(35, 39))
            and all(torch.equal(rec_4912e["mask"][k], m_4912e[30]) for k in range(35, 39))
            and all(torch.equal(rec_4912e["frames"][k], frames_39[k]) for k in range(31, 35))
            and not bool(rec_4912e["mask"][34].any())
            and "hold shot frame 30 (2958 px)" in (rec_4912e.get("held_fell_back") or [""])[0]):
        problems.append(f"held: 4912e tail must step back over 34..31 to shot frame 30 (2958 px) for 35..38 and leave 31..34 untouched: "
                        f"{rec_4912e.get('held_after')!r}, {rec_4912e.get('held_fell_back')!r}")
    # second_hfp_4647g's empty tail (`HELD_EDGE_EMPTY_RUN`): when the outermost 4+ shot frames are all 0 px while earlier frames
    # have a mask, the subject is hidden/absent at that edge of the shot, so the 0 px edge frame (34) itself is held across 35..38
    # without stepping back into the shot and injecting an earlier mask into a boundary latent step
    m_4647g = mask_35.clone()
    m_4647g[-6:] = 0.0
    node.hidden = SimpleNamespace(prompt={"10": {"_meta": {"pad_before": 0, "pad_after": 4, "held_before": 0, "held_after": 4}}},
                                  unique_id="10")
    try:
        rec_4647g = node.execute(frames_39, m_4647g, grow_pixels=8, others=others_35).args[0]
    finally:
        node.hidden = None
    if not (rec_4647g.get("held_after") == 4
            and not rec_4647g.get("held_fell_back")
            and all(torch.equal(rec_4647g["frames"][k], frames_39[34]) for k in range(35, 39))
            and not bool(rec_4647g["mask"][29:].any())):
        problems.append(f"held: 4647g hidden tail (>= {vm.HELD_EDGE_EMPTY_RUN} empty edge frames) must hold edge frame 34 with empty mask: "
                        f"{rec_4647g.get('held_after')!r}, {rec_4647g.get('held_fell_back')!r}")
    # a graph that does not say how its window is padded: every extra frame after the shot, as the builder pads
    # (`workflows/masked_job_graph.py::window_pad`, the owner's word of 2026-10-22; split evenly until then)
    node.hidden = None
    rec = node.execute(frames_39, mask_35, grow_pixels=8).args[0]
    if (rec.get("pad_before"), rec.get("pad_after")) != (0, 4) or not bool(rec["mask"][0].any()) or bool(rec["mask"][35:].any()):
        problems.append(f"held: a window with no padding said must put its 4 extra frames after the shot: "
                        f"{rec.get('pad_before')!r}, {rec.get('pad_after')!r}")
    # the song node: the model hears silence under the held frames, the real track elsewhere; the given value is untouched
    rate = 48000
    wave = torch.ones(1, 2, int(39 / 24 * rate))
    quiet = afs.silenced({"waveform": wave, "sample_rate": rate}, 2, 39 - 1)["waveform"]
    edge_a, edge_b = int(round(2 / 24 * rate)), int(round(38 / 24 * rate))
    if bool(quiet[..., :edge_a].any()) or bool(quiet[..., edge_b:].any()) or not bool((quiet[..., edge_a:edge_b] == 1).all()) \
            or not bool((wave == 1).all()):
        problems.append("held: the track must be silent under the held frames and as given elsewhere, the value given unchanged")
    # what the song node hands the model (`heard_under_held`, which its execute calls): the counts the Masked Source KEPT,
    # after its guard, and silence under them; a side that fell back, or a run with no source, hears the track as given
    track = {"waveform": wave, "sample_rate": rate}
    got = afs.heard_under_held(track, {"held_before": 0, "held_after": 8}, 39, 0, 8)
    end = int(round(31 / 24 * rate))
    if got[:2] != (0, 8) or bool(got[2]["waveform"][..., end:].any()) or not bool((got[2]["waveform"][..., :end] == 1).all()):
        problems.append(f"held: the song node must hear silence under the 8 held frames after the shot and the track before: {got[:2]}")
    fell = afs.heard_under_held(track, {"held_fell_back": ["after: a check"]}, 39, 0, 8)
    none = afs.heard_under_held(track, None, 39, 0, 0)
    if fell[:2] != (0, 0) or fell[2] is not track or none[2] is not track:
        problems.append("held: a side that fell back, or a run with no source, must hear the track as given")
    song = (REPO / "audio_freeze_song.py").read_text()
    if "= heard_under_held(audio, source, total, pad_before, pad_after)" not in song:
        problems.append("held: the song node's execute does not take its track from `heard_under_held`")


def check_others(problems):
    """Item 14. A subject with a neighbour standing against them: the margin stays off the neighbour and the
    subject loses no token. The control: the same mask on `keep`, which gives the subject holes."""
    n, h, w = FRAMES, 128, 192
    frames = torch.rand(n, h, w, 3)
    mask = torch.zeros(n, h, w)
    mask[:, 40:88, 72:104] = 1.0                       # the subject: one token column and a bit, 48 by 32
    others = torch.zeros(n, h, w)
    others[:, 56:120, 100:150] = 1.0                    # a neighbour overlapping the subject's right edge and below
    base = {"frames": frames, "mask": mask, "grow_pixels": 32, "feather_pixels": 0}
    shape = (LATENT_T, h // 16, w // 16)
    args = (0, n, w, h) + shape
    plain = vm.window(base, *args)[2]
    if not torch.equal(vm.window(dict(base, others=None), *args)[2], plain):
        problems.append("others: unwired, the token mask changed")
    mine = vm.token_mask(mask, *shape)
    theirs = vm.token_mask(others, *shape)
    got = vm.window(dict(base, others=others), *args)[2]
    shared, only_theirs = mine * theirs, theirs * (1.0 - mine)
    if float(shared.sum()) == 0.0 or float((plain * only_theirs).sum()) == 0.0:
        problems.append("others: the case puts no neighbour in the subject's tokens or in the margin; it tests nothing")
    if float((mine * (1.0 - got)).max()) != 0.0:
        problems.append("others: a token that holds the subject's own mask was given up")
    if float((got * only_theirs).max()) != 0.0:
        problems.append("others: the margin grew over a token that holds a neighbour and none of the subject")
    if not torch.equal(got, plain * (1.0 - only_theirs)):
        problems.append("others: tokens the neighbour does not touch changed")
    # the control: the same mask on `keep` hands tokens of the subject back to the source
    kept = vm.window(dict(base, keep=others), *args)[2]
    if float((mine * (1.0 - kept)).sum()) == 0.0:
        problems.append("others: the control (the neighbour's mask on `keep`) cost the subject no token; the case "
                        "cannot tell the two inputs apart")
    # a mask that also covers the mask the region is grown from changes nothing; under `the wired parts` that
    # mask is the part, so this is not a promise about a mask that covers the whole person
    if not torch.equal(vm.window(dict(base, others=torch.maximum(others, mask)), *args)[2], got):
        problems.append("others: a mask that also covers the subject's own mask changed the tokens")
    # with `keep` wired too, keep still wins over everything it touches
    held = torch.zeros(n, h, w)
    held[:, 60:64, 80:84] = 1.0
    both = vm.window(dict(base, others=others, keep=held), *args)[2]
    if not torch.equal(both, got * (1.0 - vm.token_mask(held, *shape))):
        problems.append("others: with keep wired too, keep does not win over the tokens it touches")
    # in time: a neighbour on one frame of a step keeps the margin off for that step only
    runs = vm.run_lengths(LATENT_T)
    once = torch.zeros_like(others)
    once[sum(runs[:2])] = others[0]
    t = vm.window(dict(base, others=once), *args)[2]
    if [k for k in range(LATENT_T) if not torch.equal(t[k], plain[k])] != [2]:
        problems.append("others: a neighbour on one frame of latent step 2 changed other steps")
    # a window reads its own frames of it: a neighbour who is there in the clip's second half only
    twice = {"frames": torch.cat([frames, frames]), "mask": torch.cat([mask, mask]), "grow_pixels": 32,
             "feather_pixels": 0, "others": torch.cat([torch.zeros_like(others), others])}
    if not torch.equal(vm.window(twice, 0, n, w, h, *shape)[2], plain)             or not torch.equal(vm.window(twice, n, n, w, h, *shape)[2], got):
        problems.append("others: a window did not read its own frames of the mask")
    # a window past the source's end: the held frames carry the last frame's mask, and its others with it
    late = vm.window(dict(base, others=others), n - 5, n, w, h, *shape)
    if late[4] != n - 5 or not torch.equal(late[3][5:], late[3][4:5].expand(n - 5, -1, -1)):
        problems.append("others: a window past the source's end did not hold its last frame with its mask")
    # the node: its record, its refusals by name, the key, the schema, the preview
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=32, feather_pixels=0, others=others.unsqueeze(-1))
    src = out.args[0] if hasattr(out, "args") else out[0]
    if src.get("others") is None or tuple(src["others"].shape) != tuple(others.shape):
        problems.append("others: the node did not carry a [N, H, W, 1] mask onto the source as [N, H, W]")
    strip = (out.args if hasattr(out, "args") else out)[2]
    bare = vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=32, feather_pixels=0)
    bare_src, bare_strip = (bare.args if hasattr(bare, "args") else bare)[0], (bare.args if hasattr(bare, "args") else bare)[2]
    if bare_src.get("others") is not None or torch.equal(strip, bare_strip):
        problems.append("others: unwired the record carries one, or the preview does not show the people kept out")
    for label, bad, kw, word in (("another frame count", others[:-1], {}, f"{n - 1} masks"),
                                 ("another size", torch.zeros(n, h // 2, w // 2), {}, "size"),
                                 ("paint_out", others, {"paint_out": True}, "paint_out"),
                                 ("a softened start", others, {"start_from": vm.START_TOP}, "softened")):
        try:
            vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=32, feather_pixels=0, others=bad, **kw)
            problems.append(f"others: {label} was accepted")
        except ValueError as exc:
            if word not in str(exc):
                problems.append(f"others: the refusal of {label} does not say so: {exc}")
    inputs = vm.MiniMaxH3MaskedSource.define_schema().inputs
    # appended inputs keep their place: a later one goes after, never between (`APPENDED`)
    got = _appended("others")
    if got is None or not got.optional:
        problems.append("others: it is not where it was appended (the input before `edge`) and optional")


#: The Masked Source's last inputs, in the order they were appended. A saved workflow stores a widget's value
#: by position, so a new input goes on the end and none of these moves: each case below that owns one of them
#: asks `_appended` for it, and adding an input is one name on the end of this tuple.
APPENDED = ("others", "edge", "motion_video", "render_region", "motion_video_2", "motion_short_edge_2", "context_shows", "zoom_up_to", "encode_frames")


def _appended(name: str):
    """The Masked Source's input `name` when the node's last inputs are `APPENDED` in order, else None."""
    inputs = vm.MiniMaxH3MaskedSource.define_schema().inputs
    if tuple(i.id for i in inputs[-len(APPENDED):]) != APPENDED:
        return None
    return next(i for i in inputs if i.id == name)


def _fail(problems, text: str) -> None:
    """Record a failed case. `check_motion_reference` called this from 0.190.7 on with nothing
    defining it here, so a failing case there was a NameError that ended the run before the checks
    after it (found by mrop, 2026-10-06, when a control went red by that crash)."""
    problems.append(text)


def _load(name: str = "video_mask"):
    """A root module of the pack as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location(f"_h3pack.{name}", REPO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_h3pack.{name}"] = module
    spec.loader.exec_module(module)
    return module


vm = _load()

SOURCE = "MiniMaxH3MaskedSource"
SONG = "MiniMaxH3AudioFreezeSong"
# a small canvas on the model's grid: 16 px per latent cell, 2 cells per token
W, H, LAT_W, LAT_H = 192, 128, 12, 8
LATENT_T = 7
FRAMES = sum(vm.run_lengths(LATENT_T))


def check_temporal(problems):
    mask = torch.zeros(FRAMES, H, W)
    # one frame only, in the middle of the third run, a block the size of one cell
    run_start = sum(vm.run_lengths(2))
    frame = run_start + 1
    mask[frame, 32:48, 64:80] = 1.0
    tokens = vm.token_mask(mask, LATENT_T, LAT_H, LAT_W)
    if float(tokens[2].max()) != 1.0:
        problems.append("a subject present in one frame of a run did not regenerate that run's latent step")
    if float(tokens.sum()) != float(tokens[2].sum()):
        problems.append("a one-frame subject marked a latent step outside its own run")
    if sorted(tokens.unique().tolist()) not in ([0.0, 1.0], [1.0]):
        problems.append(f"token_mask is not binary: {tokens.unique().tolist()}")
    # the control: core's resize of the same mask
    core = comfy.utils.reshape_mask(mask, (1, 24, LATENT_T, LAT_H, LAT_W))[0, 0]
    if float(core.max()) >= 0.5:
        problems.append(
            "control failed: core's trilinear resize kept a one-frame subject at half strength or more "
            f"({float(core.max()):.3f}), so the per-run maximum is not what protects it")
    try:
        vm.token_mask(mask[:-1], LATENT_T, LAT_H, LAT_W)
        problems.append("a mask one frame short of the window was accepted")
    except ValueError:
        pass
    if tuple(FRAME_PER_TOKEN) != tuple(vm.run_lengths(len(FRAME_PER_TOKEN))):
        problems.append("run_lengths does not follow core's FRAME_PER_TOKEN")


def check_token_grid(problems):
    torch.manual_seed(0)
    for lat_h, lat_w in ((LAT_H, LAT_W), (7, 11)):  # the odd grid takes core's replicate pad
        # a few single pixels, so most tokens stay preserved and the pooling has edges to move
        mask = torch.zeros(FRAMES, lat_h * 16, lat_w * 16)
        for _ in range(6):
            f, y, x = (int(torch.randint(0, n, (1,))) for n in mask.shape)
            mask[f, y, x] = 1.0
        tokens = vm.token_mask(mask, LATENT_T, lat_h, lat_w)
        pad_h, pad_w = lat_h + lat_h % 2, lat_w + lat_w % 2
        rows = mask_row_values(tokens, LATENT_T, pad_h, pad_w)
        if rows is None:
            problems.append("core read a partly masked window as fully generating")
            continue
        again = rows.reshape(LATENT_T, pad_h // 2, pad_w // 2)
        back = again.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[:, :lat_h, :lat_w]
        if not torch.equal(back, tokens):
            problems.append(f"core's patch pooling changes token_mask's output on a {lat_h}x{lat_w} grid")


def check_feather(problems):
    grow, feather = 32, 8
    subject = torch.zeros(FRAMES, H, W)
    subject[:, 48:80, 80:112] = 1.0
    tokens = vm.token_mask(vm.grow(subject, grow), LATENT_T, LAT_H, LAT_W)
    alpha = vm.pixel_alpha(tokens, H, W, feather)
    if tuple(alpha.shape) != (FRAMES, H, W):
        problems.append(f"pixel_alpha returned {tuple(alpha.shape)} for {FRAMES} frames of {H}x{W}")
        return
    if float(alpha[subject > 0.5].min()) != 1.0:
        problems.append("the feather reaches the subject's own pixels: the original would show at its edge")
    region = torch.nn.functional.interpolate(tokens.unsqueeze(1), size=(H, W), mode="nearest")[:, 0]
    region = region.repeat_interleave(torch.tensor(vm.run_lengths(LATENT_T)), dim=0)
    far = vm.grow(region, feather) < 0.5
    if far.any() and float(alpha[far].max()) != 0.0:
        problems.append("the blend weight is above 0 further than the feather from a regenerated token")
    # the node refuses a feather wider than the grow
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.zeros(2, 8, 8), grow_pixels=4, feather_pixels=8)
        problems.append("the node accepted a feather wider than the grow")
    except ValueError:
        pass
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(3, 8, 8, 3), torch.zeros(2, 8, 8))
        problems.append("the node accepted a mask with a different frame count from the frames")
    except ValueError:
        pass


def check_composite(problems):
    torch.manual_seed(1)
    render, source = torch.rand(FRAMES, H, W, 3), torch.rand(FRAMES, H, W, 3)
    alpha = torch.zeros(FRAMES, H, W)
    alpha[:, :64] = 1.0
    out = vm.composite(render, source, alpha)
    if not torch.equal(out[:, 64:], source[:, 64:]):
        problems.append("the composite does not return the source bit for bit where nothing regenerates")
    if not torch.equal(out[:, :64], render[:, :64]):
        problems.append("the composite does not return the render where the weight is 1")
    try:
        vm.composite(render[:-1], source, alpha)
        problems.append("a render one frame short of the source was composited")
    except ValueError:
        pass


def check_window(problems):
    have = FRAMES + 10
    frames = torch.rand(have, 72, 128, 3)
    mask = torch.zeros(have, 72, 128)
    mask[:, 20:40, 40:60] = 1.0
    source = {"frames": frames, "mask": mask, "grow_pixels": 0, "feather_pixels": 0}
    pixels, encode, tokens, _mask, held = vm.window(source, 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
    if tuple(pixels.shape) != (FRAMES, H, W, 3) or tuple(tokens.shape) != (LATENT_T, LAT_H, LAT_W) or held:
        problems.append(f"a whole window came back as {tuple(pixels.shape)}, {tuple(tokens.shape)}, held {held}")
    if encode is not pixels:
        problems.append("with paint_out off the frames to encode are not the fitted frames themselves")
    # the last window of a loop: the source runs out inside it
    start = have - 12
    pixels, _encode, tokens, _mask, held = vm.window(source, start, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
    if held != FRAMES - 12:
        problems.append(f"a window 12 frames from the source's end reported {held} held frames, expected {FRAMES - 12}")
    elif not torch.equal(pixels[12:], pixels[11:12].expand(FRAMES - 12, -1, -1, -1)):
        problems.append("the frames past the source's end are not its last frame held")
    elif not torch.equal(_mask[12:], _mask[11:12].expand(FRAMES - 12, -1, -1)) or float(_mask[11].max()) == 0.0 \
            or float(tokens[-1].max()) == 0.0:
        problems.append("the held frames past the source's end must carry the last frame's mask, so their steps regenerate "
                        "(the open tail, the only behaviour since 2026-10-10)")
    try:
        vm.window(source, have, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
        problems.append("a window starting past the source's end was accepted")
    except ValueError:
        pass


def check_fit(problems):
    # a wide source onto a narrower canvas and a tall one onto a wider: both crops
    for src_h, src_w in ((270, 480), (400, 300)):
        frames = torch.zeros(1, src_h, src_w, 3)
        mask = torch.zeros(1, src_h, src_w)
        y0, y1, x0, x1 = src_h // 3, src_h // 2, src_w // 3, src_w // 2
        frames[:, y0:y1, x0:x1] = 1.0
        mask[:, y0:y1, x0:x1] = 1.0
        f = vm.fit_frames(frames, W, H)[..., 0]
        m = vm.fit_mask(mask, W, H)
        if tuple(m.shape) != (1, H, W):
            problems.append(f"fit_mask returned {tuple(m.shape)} for a {W}x{H} canvas")
            continue
        # every pixel the fitted frame shows as subject is inside the fitted mask
        if bool(((f > 0.5) & (m < 0.5)).any()):
            problems.append(f"a {src_w}x{src_h} mask and its frames do not coincide after the fit")
    line = torch.zeros(1, 1080, 1920)
    line[:, :, 1001] = 1.0
    if float(vm.fit_mask(line, W, H).max()) != 1.0:
        problems.append("a one-pixel line was lost when the mask was fitted down")
    core = comfy.utils.common_upscale(line.unsqueeze(1), W, H, "bilinear", "center")
    if float(core.max()) >= 0.5:
        problems.append("control failed: core's bilinear resize kept a one-pixel line at half strength or more")
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.zeros(2, 8, 16))
        problems.append("the node accepted a mask whose shape is not its frames'")
    except ValueError:
        pass


def check_paint_out(problems):
    grow_px = 32
    frames = torch.full((FRAMES, H, W, 3), 0.2)
    mask = torch.zeros(FRAMES, H, W)
    mask[:, 48:80, 80:112] = 1.0
    frames[:, 48:80, 80:112] = 1.0  # a bright subject on a flat ground
    source = {"frames": frames, "mask": mask, "grow_pixels": grow_px, "feather_pixels": 8, "paint_out": True}
    pixels, encode, tokens, _mask, _held = vm.window(source, 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
    if not torch.equal(pixels, frames):
        problems.append("paint_out changed the frames the composite restores")
    changed = (encode != pixels).any(dim=-1)
    hole = vm.grow(mask, grow_px // 2) > 0.5
    if bool((changed & ~hole).any()):
        problems.append("paint_out changed pixels outside its hole")
    if bool(((mask > 0.5) & ~changed).any()):
        problems.append("a subject pixel survived the paint-out")
    region = torch.nn.functional.interpolate(tokens.unsqueeze(1), size=(H, W), mode="nearest")[:, 0]
    region = region.repeat_interleave(torch.tensor(vm.run_lengths(LATENT_T)), dim=0) > 0.5
    if bool((changed & ~region).any()):
        problems.append("a filled pixel lies outside the regenerated tokens: the composite would not hide it")
    if float(encode.max()) > 0.25:
        problems.append(f"the bright subject is still in the frames to encode (max {float(encode.max()):.3f})")
    # a fill of nothing is nothing
    if not torch.equal(vm.fill_subject(frames, torch.zeros(FRAMES, H, W)), frames):
        problems.append("fill_subject changed frames with an empty hole")


def check_part(problems):
    subject = torch.zeros(4, 64, 96)
    subject[:, 10:50, 20:40] = 1.0
    part = torch.zeros(4, 64, 96)
    part[0, 10:20, 22:38] = 1.0   # the subject's own, down to row 19
    part[0, 10:20, 70:86] = 1.0   # a neighbour's, far from the subject
    part[3, 10:30, 22:38] = 1.0   # frames 1 and 2 have none
    got = vm.select_part(subject, part)
    if float(got[0, 10:20, 22:38].min()) != 1.0:
        problems.append("select_part dropped the part that lies on the subject")
    if float(got[:, :, 60:].max()) != 0.0:
        problems.append("select_part kept a neighbour's part")
    bottoms = vm.part_bottom(got)
    if bottoms.tolist() != [19, -1, -1, 29]:
        problems.append(f"part_bottom read {bottoms.tolist()}, expected [19, -1, -1, 29]")
    filled, carried = vm.carry_missing(bottoms)
    if carried != 2 or filled.tolist() != [19, 19, 29, 29]:
        problems.append(f"carry_missing gave {filled.tolist()} carrying {carried}; each miss takes its nearest found frame")
    none, n = vm.carry_missing(torch.tensor([-1, -1]))
    if n != 0 or none.tolist() != [-1, -1]:
        problems.append("carry_missing invented a part where no frame had one")
    region = vm.above(subject, filled)
    if float(region[0, 10:20, 20:40].min()) != 1.0 or float(region[0, 20:].max()) != 0.0:
        problems.append("above does not hold exactly the subject's rows down to the part's lowest row")
    if float(region[:, :, 60:].max()) != 0.0 or float(region[1, 10:20, 20:40].min()) != 1.0:
        problems.append("above took pixels off the subject, or left a carried frame empty")
    for kwargs, what in (({"replace": "head and hair"}, "a part with no segmenter wired"),
                         ({"replace": "left arm"}, "an unknown replace"),
                         ({"replace": "head and hair", "segmenter": object(), "segmenter_clip": object(),
                           "part_phrases": " , "}, "an empty phrase list")):
        try:
            vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.ones(2, 8, 8), **kwargs)
            problems.append(f"the node accepted {what}")
        except ValueError:
            pass
    out = vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.ones(2, 8, 8))
    out = getattr(out, "args", out)
    if len(out) != 3 or not torch.equal(out[1], out[0]["mask"].to(torch.float32)):
        problems.append("the node's second output is not the mask it used")
    strip = out[2]
    if tuple(strip.shape) != (1, min(vm.PREVIEW_ROWS, 2) * vm.PREVIEW_HEIGHT, vm.PREVIEW_HEIGHT, 3):
        problems.append(f"the preview strip of two 8x8 frames should be [1, 2*{vm.PREVIEW_HEIGHT}, {vm.PREVIEW_HEIGHT}, 3], got {tuple(strip.shape)}")
    frames = torch.zeros(3, 16, 16, 3)
    subject = torch.zeros(3, 16, 16); subject[:, 2:14, 4:12] = 1.0
    with_ref = vm.MiniMaxH3MaskedSource.execute(frames, subject, motion_reference=vm.MOTION_SUBJECT, motion_short_edge=32)
    with_ref = getattr(with_ref, "args", with_ref)
    if int(with_ref[2].shape[2]) <= int(strip.shape[2]) * 0 + vm.PREVIEW_HEIGHT:
        problems.append("with a motion reference on, the preview strip must show the reference beside the plate")
    # the wired parts: exactly the part node's mask, kept to the subject
    parts = torch.zeros(3, 16, 16); parts[:, 4:8, 6:10] = 1.0; parts[:, 0:2, 0:2] = 1.0   # a part on him and a blob off him
    out_parts = vm.MiniMaxH3MaskedSource.execute(frames, subject, replace=vm.REPLACE_PARTS, parts=parts, part_margin=0)
    out_parts = getattr(out_parts, "args", out_parts)
    region = out_parts[0]["mask"]
    if not bool((region[:, 4:8, 6:10] > 0.5).all()) or bool((region[:, 0:2, 0:2] > 0.5).any()) or bool((region[:, 10:14] > 0.5).any()):
        problems.append("`the wired parts` must regenerate the part where it lies on the subject and nothing else")
    # the part mask's coverage of the tracked subject travels in the record (`part_coverage.py`): a line when
    # frames are in doubt, None when they are not and on every other `replace`. Counted on the tracker's mask
    # and the part mask as wired, before the part replaces the mask.
    pc = sys.modules["_h3pack.part_coverage"]
    key = pc.RECORD_KEY
    if key not in out_parts[0] or out_parts[0][key] is not None:
        problems.append(f"a part that covers the same share of the subject on every frame is in doubt: {out_parts[0].get(key)!r}")
    doubted = torch.zeros(3, 16, 16); doubted[:2, 2:14, 4:12] = 1.0; doubted[2, 2:3, 4:5] = 1.0   # a sliver on the last frame
    record = vm.MiniMaxH3MaskedSource.execute(frames, subject, replace=vm.REPLACE_PARTS, parts=doubted, part_margin=0)
    record = getattr(record, "args", record)[0]
    want = pc.summarise(pc.coverage(subject, doubted)).warning()
    if want is None or record.get(key) != want or "1 of 3 frames" not in str(record.get(key)):
        problems.append(f"the record's warning for a part that is a sliver on one frame is {record.get(key)!r}, not {want!r}")
    if pc.record_line(record) != f"the Masked Source warns: {want}" or pc.record_line(out_parts[0]) is not None or pc.record_line(None) is not None:
        problems.append("record_line does not give one line for a record with a warning and None for one without, or for no record")
    whole = getattr(vm.MiniMaxH3MaskedSource.execute(frames, subject), "args", None)[0]
    if key not in whole or whole[key] is not None:
        problems.append("a Masked Source with no part wired carries a part warning, or does not carry the key at all")
    song_source = (REPO / "audio_freeze_song.py").read_text()
    if song_source.count("part_coverage.record_line(source)") < 2 or "lines.append(part_coverage.record_line(source))" not in song_source:
        problems.append("the song node's report no longer shows the Masked Source's part warning in its source block")
    try:
        vm.MiniMaxH3MaskedSource.execute(frames, subject, replace=vm.REPLACE_PARTS)
        problems.append("`the wired parts` with nothing on `parts` was accepted")
    except ValueError:
        pass
    try:
        vm.MiniMaxH3MaskedSource.execute(frames, subject, replace=vm.REPLACE_PARTS, parts=torch.zeros(3, 16, 16))
        problems.append("`the wired parts` with an empty part mask was accepted")
    except ValueError:
        pass
    asked = vm.MiniMaxH3MaskedSource.check_lazy_status(frames=frames, replace=vm.REPLACE_PARTS,
                                                       mask=None, parts=None, segmenter=None, segmenter_clip=None, shot_table=None)
    if sorted(asked) != sorted(["mask", vm.LAZY_FOR_PARTS, vm.LAZY_FOR_TABLE]):
        problems.append(f"`the wired parts` must ask lazily for the mask, the parts and the table only, not {asked}")


def check_changed_alpha(problems):
    grow_px, feather = 32, 8
    old = torch.zeros(FRAMES, H, W)
    old[:, 48:80, 80:112] = 1.0
    tokens = vm.token_mask(vm.grow(old, grow_px), LATENT_T, LAT_H, LAT_W)
    source = torch.full((FRAMES, H, W, 3), 0.5)
    new = torch.zeros(FRAMES, H, W, dtype=torch.bool)
    new[:, 40:72, 96:128] = True        # the new subject, shifted: part inside the old outline, part in margin
    render = source.clone()
    render[new] = 0.9
    blank = torch.zeros(FRAMES, H, W)
    t = vm.CHANGE_THRESHOLD
    alpha = vm.changed_alpha(render, source, tokens, old, feather, grow_px // 2, t)
    whole = vm.pixel_alpha(tokens, H, W, feather)
    region = vm.pixel_alpha(tokens, H, W, 0) > 0.5
    deep = whole >= 1.0                 # further than the feather inside the region
    if tuple(alpha.shape) != (FRAMES, H, W):
        problems.append(f"changed_alpha returned {tuple(alpha.shape)}")
        return
    if bool((alpha > whole + 1e-6).any()):
        problems.append("the changed-only weight exceeds the whole-region weight somewhere")
    if float(alpha[new & deep].min()) != 1.0:
        problems.append("a new-subject pixel well inside the region is not fully kept")
    if float(alpha[(old > 0.5) & deep].min()) != 1.0:
        problems.append("an old-subject pixel is not fully kept: the original would show through")
    kept = new | (vm.grow(old, grow_px // 2) > 0.5)
    margin = region & ~(vm.grow(kept.float(), 2 * feather + vm.CHANGE_BLUR) > 0.5)
    if not bool(margin.any()):
        problems.append("the changed-alpha case has no margin to test")
    elif float(alpha[margin].max()) != 0.0:
        problems.append("margin the render left as the source keeps the render")
    # a faithful repaint with no old subject under it: everything is restored
    if float(vm.changed_alpha(source.clone(), source, tokens, blank, feather, grow_px // 2, t).max()) != 0.0:
        problems.append("a faithful repaint of the source keeps some of the render")
    # a difference under the ramp's foot is noise, not change
    faint = source + 0.5 * t
    if float(vm.changed_alpha(faint, source, tokens, blank, feather, grow_px // 2, t).max()) != 0.0:
        problems.append("a difference of half the threshold was kept")
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.ones(2, 8, 8), composite="the nice bits")
        problems.append("the node accepted an unknown composite")
    except ValueError:
        pass


def check_motion_reference(problems):
    """`motion_reference`: `none` is None; `subject only` keeps the subject (widened by the margin) and greys the
    rest; `whole frame` keeps everything; the short edge is honoured and rounded to the canvas multiple."""
    vm = _load()
    torch.manual_seed(0)
    pixels = torch.rand(4, 64, 96, 3)
    mask = torch.zeros(4, 64, 96)
    mask[:, 16:48, 32:64] = 1.0
    if vm.motion_reference(pixels, mask, vm.MOTION_NONE, 64, 0) is not None:
        _fail(problems, "motion_reference: `none` must return None")
    whole = vm.motion_reference(pixels, mask, vm.MOTION_FRAME, 64, 0)
    if whole is None or tuple(whole.shape) != (4, 64, 96, 3) or not torch.allclose(whole, pixels, atol=1e-6):
        _fail(problems, "motion_reference: `whole frame` at the source's own short edge must be the frames unchanged")
    subject = vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, 64, 0)
    inside = subject[:, 16:48, 32:64]
    outside = torch.cat([subject[:, :16].flatten(), subject[:, 48:].flatten(), subject[:, :, :32].flatten(), subject[:, :, 64:].flatten()])
    if not torch.allclose(inside, pixels[:, 16:48, 32:64], atol=1e-6):
        _fail(problems, "motion_reference: `subject only` changed the subject's own pixels")
    if not torch.allclose(outside, torch.full_like(outside, 0.5), atol=1e-6):
        _fail(problems, "motion_reference: `subject only` must set everything outside the subject to mid grey")
    widened = vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, 64, 8)
    if torch.allclose(widened[:, 8:16, 32:64], torch.full((4, 8, 32, 3), 0.5), atol=1e-6):
        _fail(problems, "motion_reference: the margin must widen what is kept of the subject")
    small = vm.motion_reference(pixels, mask, vm.MOTION_FRAME, 32, 0)
    if tuple(small.shape) != (4, 32, 64, 3):
        _fail(problems, f"motion_reference: a 32 short edge on 64x96 frames should give 32x64, got {tuple(small.shape)}")
    try:
        vm.motion_reference(pixels, mask, "sideways", 64, 0)
        _fail(problems, "motion_reference: an unknown mode was accepted")
    except ValueError:
        pass
    import reference_order as ro
    if ro.MOTION_NONE != vm.MOTION_NONE or ro.MASKED_SOURCE_CLASS != "MiniMaxH3MaskedSource":
        _fail(problems, "reference_order's literals for the Masked Source must equal video_mask's: the static label plan reads them")
    song_inputs = {"source": ["9", 0]}
    g = {"9": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"motion_reference": vm.MOTION_SUBJECT}}}
    if ro.assign_labels(ro.plan_for(song_inputs, g)) != ["<Video 1>"]:
        _fail(problems, "a Masked Source with a motion reference must add one <Video N> to the song node's static label plan")
    g["9"]["inputs"].update(motion_reference=vm.MOTION_WIRED, motion_video=["3", 0], motion_video_2=["4", 0])
    if ro.assign_labels(ro.plan_for(song_inputs, g)) != ["<Video 1>", "<Video 2>"]:
        _fail(problems, "a Masked Source with a second wired video must add two <Video N> to the static label plan, in order")
    g["9"]["inputs"] = {"motion_reference": vm.MOTION_NONE}
    if ro.assign_labels(ro.plan_for(song_inputs, g)) != []:
        _fail(problems, "a Masked Source with no motion reference must add nothing to the static label plan")


def check_motion_zoom(problems):
    """`subject only, zoomed in`: the box is one a shot, around the TRACKED subject, on the token grid; and the
    four things `zoom_plan` promises of the picture, each with a case that would fail without it."""
    vm = _load()
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE as M
    torch.manual_seed(0)
    H, W, SHORT = 256, 384, 128                 # the whole-frame reference of these frames is 128x192
    pixels = torch.rand(6, H, W, 3)

    def subject(*boxes):
        m = torch.zeros(len(boxes), H, W)
        for i, b in enumerate(boxes):
            if b is not None:
                m[i, b[1]:b[3], b[0]:b[2]] = 1.0
        return m

    # ---- the boxes: one a shot, a union, on the grid, in the frame, changing only at a cut
    small = [(200, 100, 230, 150), (210, 100, 240, 150), (220, 110, 250, 160)]          # a subject that travels
    mask = subject(*small, None, (40, 40, 100, 200), (40, 40, 100, 200))
    rows = vm._tracked_boxes(mask)
    if rows.tolist()[0] != [200, 100, 230, 150] or rows.tolist()[3] != [-1, -1, -1, -1]:
        _fail(problems, f"zoom: the subject's box per frame is (x0, y0, x1, y1), far side exclusive, -1 when absent; got {rows.tolist()[:4]}")
    per_shot = vm.shot_boxes(rows, [(0, 2), (3, 5)], W, H)
    first, second = per_shot[0].tolist(), per_shot[4].tolist()
    if not (per_shot[:3] == per_shot[0]).all() or not (per_shot[3:] == per_shot[3]).all() or first == second:
        _fail(problems, "zoom: a shot's frames must share one box, and two shots must not share theirs")
    for label, box, members in (("first", first, small), ("second", second, [(40, 40, 100, 200)])):
        if any(v % M for v in box) or box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
            _fail(problems, f"zoom: the {label} shot's box {box} is off the canvas multiple or outside the frame")
        if any(b[0] < box[0] or b[1] < box[1] or b[2] > box[2] or b[3] > box[3] for b in members):
            _fail(problems, f"zoom: the {label} shot's box {box} does not hold the subject on every one of its frames")
        if any(b[0] - box[0] < min(vm.MOTION_BOX_ROOM, b[0]) for b in members):
            _fail(problems, f"zoom: the {label} shot's box {box} leaves less than the room asked on its near side")
    if vm.shot_boxes(rows, [], W, H)[0].tolist() != vm.shot_boxes(rows, [(0, 5)], W, H)[0].tolist():
        _fail(problems, "zoom: with no shot table the frames are one shot")
    if vm.shot_boxes(rows[3:4], [(0, 0)], W, H).tolist() != [[-1, -1, -1, -1]]:
        _fail(problems, "zoom: a shot the subject is never in must have no box")
    # a window later in the clip counts the table's frames from its own first frame
    late = vm.shot_boxes(rows[2:], [(0, 2), (3, 5)], W, H, first_frame=2)
    if late[0].tolist() == late[2].tolist() or late[1].tolist() != [-1, -1, -1, -1] and late[1].tolist() != late[2].tolist():
        _fail(problems, f"zoom: a window that starts inside a shot must cut where the table cuts; got {late.tolist()}")

    # ---- the boxes on the canvas agree with the mask on the canvas
    wide = torch.zeros(2, 96, 256); wide[0, 20:70, 30:90] = 1.0; wide[1, 5:90, 200:250] = 1.0
    for cw, ch in ((128, 96), (256, 96), (128, 48)):
        fitted = vm.fit_boxes(vm._tracked_boxes(wide), 256, 96, cw, ch)
        own = vm._tracked_boxes(vm.fit_mask(wide, cw, ch))
        for f, o in zip(fitted.tolist(), own.tolist()):
            if o[0] < 0:
                continue
            if f[0] > o[0] or f[1] > o[1] or f[2] < o[2] or f[3] < o[3] or max(abs(a - b) for a, b in zip(f, o)) > 1:
                _fail(problems, f"zoom: a box fitted to {cw}x{ch} is {f} and the fitted mask's own box is {o}")

    # ---- the four promises
    full = torch.tensor([[0, 0, W, H]] * 6)
    today = vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, SHORT, 4)
    budget = int(today.shape[1]) * int(today.shape[2])
    floor = min(int(today.shape[1]) / H, int(today.shape[2]) / W)
    layouts = {
        "one small box": per_shot[:3].repeat(2, 1),
        "two shots": per_shot,
        "a wide box and a tall one": torch.tensor([[0, 96, W, 160]] * 3 + [[160, 0, 224, H]] * 3),
        "the whole frame and a small box": torch.tensor([[0, 0, W, H]] * 3 + [[192, 96, 256, 160]] * 3),
        "a tall box that is most of the frame": torch.tensor([[0, 0, 352, H]] * 6),
    }
    for name, boxes in layouts.items():
        got = vm.motion_reference(pixels, mask, vm.MOTION_ZOOM, SHORT, 4, boxes)
        if int(got.shape[1]) * int(got.shape[2]) > budget:
            _fail(problems, f"zoom, {name}: the picture is {tuple(got.shape[1:3])}, more pixels than the whole-frame reference's {tuple(today.shape[1:3])}")
        if int(got.shape[1]) % M or int(got.shape[2]) % M or int(got.shape[0]) != 6:
            _fail(problems, f"zoom, {name}: the picture {tuple(got.shape)} is off the canvas multiple or drops frames")
        plan = vm.zoom_plan(boxes, H, W, SHORT)
        for _a, _z, box, scale, size in (plan[2] if plan else ()):
            if scale < floor - 1e-9:
                _fail(problems, f"zoom, {name}: the box {box} is shown at {scale:.3f}, smaller than the whole frame's {floor:.3f}")
            if scale > 1.0 + 1e-9 or size[0] > box[3] - box[1] or size[1] > box[2] - box[0]:
                _fail(problems, f"zoom, {name}: the box {box} is enlarged past the canvas's own pixels (scale {scale:.3f})")
    if not torch.equal(vm.motion_reference(pixels, mask, vm.MOTION_ZOOM, SHORT, 4, full), today):
        _fail(problems, "zoom: a box that is the whole frame must give `subject only`'s picture, value for value")
    if not torch.equal(vm.motion_reference(pixels, subject(*[None] * 6), vm.MOTION_ZOOM, SHORT, 4, torch.full((6, 4), -1)),
                       vm.motion_reference(pixels, subject(*[None] * 6), vm.MOTION_SUBJECT, SHORT, 4)):
        _fail(problems, "zoom: with the subject in no frame the picture must be `subject only`'s")
    # the same boxes always give the same picture: the window keep's key holds the boxes, not the picture
    for name, boxes in layouts.items():
        again = vm.zoom_plan(boxes.clone(), H, W, SHORT)
        if again != vm.zoom_plan(boxes, H, W, SHORT) or not torch.equal(
                vm.motion_reference(pixels, mask, vm.MOTION_ZOOM, SHORT, 4, boxes),
                vm.motion_reference(pixels.clone(), mask.clone(), vm.MOTION_ZOOM, SHORT, 4, boxes.clone())):
            _fail(problems, f"zoom, {name}: the same boxes gave two pictures")
    tall = vm.zoom_plan(torch.tensor([[160, 0, 256, H]] * 6), H, W, SHORT)
    if tall is None or tall[0] <= tall[1] or tall[2][0][3] <= floor + 1e-9:
        _fail(problems, f"zoom: a tall box must get a tall picture and be shown larger than the whole frame shows it; got {tall}")
    if vm.zoom_plan(per_shot, H, W, 4 * H) is not None:
        _fail(problems, "zoom: a short edge past the frame's own leaves nothing to zoom; the plan must be None")

    # a small box is shown at the canvas's own pixels: the crop, value for value, with grey outside the subject
    one = per_shot[:3]
    got = vm.motion_reference(pixels[:3], mask[:3], vm.MOTION_ZOOM, SHORT, 0, one)
    x0, y0, x1, y1 = one[0].tolist()
    if tuple(got.shape[1:3]) != (y1 - y0, x1 - x0):
        _fail(problems, f"zoom: a box under the budget must be shown at its own size {(y1 - y0, x1 - x0)}, got {tuple(got.shape[1:3])}")
    else:
        sx0, sy0, sx1, sy1 = small[1]
        if not torch.equal(got[1, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0], pixels[1, sy0:sy1, sx0:sx1]):
            _fail(problems, "zoom: the subject's own pixels changed inside its box")
        outside = got[1].clone(); outside[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = 0.5
        if not torch.allclose(outside, torch.full_like(outside, 0.5)):
            _fail(problems, "zoom: everything in the box that is not the subject must be mid grey")
        # the box holds still, so the subject's travel is still travel: ten columns between frames 0 and 1
        cols = [int((got[i] != 0.5).any(dim=-1).any(dim=0).nonzero()[0]) for i in (0, 1)]
        if cols[1] - cols[0] != small[1][0] - small[0][0]:
            _fail(problems, f"zoom: the subject moved {small[1][0] - small[0][0]} columns in the frame and {cols[1] - cols[0]} in its box")
    # two shots of different size share one picture: the smaller sits centred on grey
    both = vm.motion_reference(pixels, mask, vm.MOTION_ZOOM, SHORT, 0, per_shot)
    plan = vm.zoom_plan(per_shot, H, W, SHORT)
    if plan is None or len(plan[2]) != 2:
        _fail(problems, "zoom: two shots with their own boxes must be two runs of one plan")
    else:
        (_a, _z, _box, _s, (sh, sw)) = min(plan[2], key=lambda r: r[4][0] * r[4][1])
        top, left = (plan[0] - sh) // 2, (plan[1] - sw) // 2
        frame = both[0] if plan[2][0][4] == (sh, sw) else both[4]
        border = frame.clone(); border[top:top + sh, left:left + sw] = 0.5
        if not torch.allclose(border, torch.full_like(border, 0.5)):
            _fail(problems, "zoom: a shot smaller than the picture must sit centred on grey")
        # and the subject is where a centred box puts it: the first shot is at the canvas's own scale here
        (_a, _z, (bx0, by0, _bx1, _by1), scale, (fh, fw)) = plan[2][0]
        sx0, sy0, sx1, sy1 = small[0]
        at_top, at_left = (plan[0] - fh) // 2 + sy0 - by0, (plan[1] - fw) // 2 + sx0 - bx0
        if scale != 1.0 or (plan[0] - fh) // 2 == 0 or not torch.equal(
                both[0, at_top:at_top + sy1 - sy0, at_left:at_left + sx1 - sx0], pixels[0, sy0:sy1, sx0:sx1]):
            _fail(problems, f"zoom: the first shot's box must sit in the middle of a taller picture with the subject's pixels in place; plan {plan}")
    for bad in (None, per_shot[:2]):
        try:
            vm.motion_reference(pixels, mask, vm.MOTION_ZOOM, SHORT, 4, bad)
            _fail(problems, "zoom: a window with no box per frame was accepted")
        except ValueError:
            pass

    # ---- the node: the record carries the TRACKED subject's boxes, whatever is replaced, and the window reads them
    frames = torch.full((6, H, W, 3), 0.3)      # flat, so the outline is the only cyan on the plate
    parts = torch.zeros(6, H, W); parts[:, 100:120, 205:225] = 1.0; parts[3:, 60:90, 50:90] = 1.0
    table = json.dumps({"shots": [{"first_frame": 0, "last_frame": 2}, {"first_frame": 3, "last_frame": 5}]})
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, replace=vm.REPLACE_PARTS, parts=parts, part_margin=0,
                                           motion_reference=vm.MOTION_ZOOM, motion_short_edge=SHORT, shot_table=table)
    out = getattr(out, "args", out)
    record = out[0]
    if not torch.equal(record["subject_boxes"], rows):
        _fail(problems, "zoom: the source record's `subject_boxes` must be the tracker's mask's boxes, not the replaced part's")
    if vm.shot_ranges(record) != [(0, 2), (3, 5)] or vm.shot_ranges({"shot_table": ""}) != [] or vm.shot_ranges({"shot_table": "{"}) != []:
        _fail(problems, "zoom: the cuts must be read from the record's shot table, and an empty or unreadable one is no cuts")
    # a pass that replaces a part frames the PART (a rule since 2026-10-10, where an input chose for a day); the
    # same record read as a whole-subject pass frames the tracked person
    of_part = vm.shot_boxes(record["region_boxes"], [(0, 2), (3, 5)], W, H)
    win, person = vm.window_boxes(record, 0, 6, W, H), vm.window_boxes(dict(record, replace=vm.REPLACE_WHOLE), 0, 6, W, H)
    if win is None or not torch.equal(win, of_part) or person is None or not torch.equal(person, per_shot) or torch.equal(of_part, per_shot):
        _fail(problems, "zoom: on a canvas the size of the frames a part pass's boxes must be the part's shots' boxes and a "
                        "whole-subject pass's the tracked person's")
    held = vm.window_boxes(record, 4, 6, W, H)
    if held is None or held[0].tolist() != of_part[4].tolist() or not (held == held[0]).all():
        _fail(problems, f"zoom: frames past the source's end carry their shot's box; got {None if held is None else held.tolist()}")
    if vm.window_boxes({"frames": frames}, 0, 6, W, H) is not None:
        _fail(problems, "zoom: a source with no boxes must say so with None")
    plain = getattr(vm.MiniMaxH3MaskedSource.execute(frames, mask, motion_reference=vm.MOTION_SUBJECT,
                                                     motion_short_edge=SHORT), "args", None)
    plate_w = int(round(W * vm.PREVIEW_HEIGHT / H))
    if int(out[2].shape[1]) != int(plain[2].shape[1]) or int(out[2].shape[2]) <= plate_w:
        _fail(problems, "zoom: the preview strip must show the zoomed picture beside each plate")
    if int(out[2].shape[2]) == int(plain[2].shape[2]):
        _fail(problems, "zoom: the picture beside the plate must be the zoomed one, which on these boxes is not the whole frame's shape")
    cyan = (out[2][0, :, :plate_w, 1] - out[2][0, :, :plate_w, 0]) > 0.3
    if not bool(cyan.any()) or bool(((plain[2][0, :, :plate_w, 1] - plain[2][0, :, :plate_w, 0]) > 0.3).any()):
        _fail(problems, "zoom: the preview's plate must carry the box as an outline, and only when zoomed in")
    if "box" not in vm.zoom_note(per_shot, H, W, SHORT) or "whole frame" not in vm.zoom_note(full, H, W, SHORT):
        _fail(problems, "zoom: the report's clause must say the boxes, or that the whole frame is shown")
    if vm.MOTION_ZOOM not in vm.MOTIONS or len(set(vm.MOTIONS)) != len(vm.MOTIONS) \
            or vm.MOTIONS[:4] != (vm.MOTION_NONE, vm.MOTION_SUBJECT, vm.MOTION_ZOOM, vm.MOTION_FRAME):
        _fail(problems, "zoom: MOTIONS must list each choice once, the first four in the place they shipped in")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "video_mask.window_boxes(source, int(round(w.start * FPS)), w.frames, width, height)" not in song \
            or 'int(source["motion_short_edge"]), video_mask.motion_widening(source),' not in song:
        _fail(problems, "zoom: the song node no longer builds the window's boxes and hands them to the motion reference")


def check_late_start(problems):
    """`start_from`: `noise` changes nothing; the softened start greys only the subject in the frames to encode and
    empties only the body's tokens, leaving the top share, the margin and every kept token; off by default."""
    grow_px = 48                                   # wider than a token, so the margin beyond the subject has tokens of its own
    w, h, lat_w, lat_h = 256, 160, 16, 10
    torch.manual_seed(0)
    frames = torch.rand(FRAMES, h, w, 3)
    mask = torch.zeros(FRAMES, h, w)
    mask[:, 20:140, 96:160] = 1.0
    frames[:, 20:140, 96:160] = torch.tensor([0.1, 0.8, 0.3])       # a green subject
    base = {"frames": frames, "mask": mask, "grow_pixels": grow_px, "feather_pixels": 4, "paint_out": False}

    def start(**kw):
        return {**base, "start_from": vm.START_TOP, "start_top": 0.3, "start_blur": 8, "start_knots": 1, **kw}

    pixels, encode, tokens, fitted, _held = vm.window({**base, "start_from": vm.START_NOISE}, 0, FRAMES, w, h, LATENT_T, lat_h, lat_w)
    if not torch.equal(pixels, encode):
        problems.append("start_from `noise` changed the frames to encode")
    if vm.start_zero_tokens({**base, "start_from": vm.START_NOISE}, fitted, tokens, 0) is not None or vm.start_zero_tokens(base, fitted, tokens, 0) is not None:
        problems.append("start_from `noise`, or a source without the key, empties tokens")

    pixels, encode, tokens, fitted, _held = vm.window(start(), 0, FRAMES, w, h, LATENT_T, lat_h, lat_w)
    if not torch.equal(pixels, frames):
        problems.append("a softened start changed the frames the composite restores")
    inside = encode[0, 30:130, 104:152]
    if not bool(((inside[..., 0] - inside[..., 1]).abs() < 1e-5).all()) or float(inside.std()) > 0.05:
        problems.append("the subject is not a grey blur in the frames to encode: its colour or its detail would start the render")
    changed = (encode != pixels).any(dim=-1)
    if bool((changed & ~(vm.grow(mask, grow_px // 2) > 0.5)).any()):
        problems.append("a softened start changed pixels outside the subject's hole")
    empty = vm.start_zero_tokens(start(), fitted, tokens, 0)
    if tuple(empty.shape) != (LATENT_T, lat_h, lat_w):
        problems.append(f"the emptied tokens are {tuple(empty.shape)}, not the window's token grid")
        return
    rows = empty[0].amax(dim=1)
    if float(rows[1]) or float(rows[2]):
        problems.append(f"token rows under the top of the subject are emptied; they must keep the softened source: {rows.tolist()}")
    if not all(float(rows[r]) for r in (5, 6, 7)):
        problems.append(f"token rows under the body are not emptied, so the original's clothes would start the render: {rows.tolist()}")
    if float((empty * (1.0 - tokens)).sum()) != 0.0:
        problems.append("a kept token is emptied: the plate would lose source pixels")
    # a token is a 2x2 patch of latent cells, so the subject's edge at cell 5 takes cells 4 and 5 together
    if float(empty[:, :, :4].sum()) != 0.0 or float(empty[:, :, 12:].sum()) != 0.0:
        problems.append("margin tokens beside the subject are emptied; they hold background and must keep it")
    if float(tokens[:, :, :4].sum() + tokens[:, :, 12:].sum()) == 0.0:
        problems.append("the control failed: this canvas has no regenerated margin tokens beside the subject to leave alone")
    if float(vm.start_zero_tokens(start(start_top=1.0), fitted, tokens, 0).sum()) != 0.0:
        problems.append("with the whole subject kept, tokens are still emptied")
    if float(empty.sum()) == 0.0:
        problems.append("the control failed: nothing is emptied at the default share either")
    gone = mask.clone()
    gone[0] = 0.0                                   # a frame the subject is not in
    px2, enc2, _t, _m, _h = vm.window(start(mask=gone), 0, FRAMES, w, h, LATENT_T, lat_h, lat_w)
    if not torch.equal(enc2[0], px2[0]):
        problems.append("a frame the subject is not in was softened")
    schema = vm.MiniMaxH3MaskedSource.define_schema()
    inputs = {i.id: i for i in schema.inputs}
    names = ("start_from", "start_top", "start_blur", "start_knots")
    for name, default in zip(names, (vm.START_NOISE, vm.START_TOP_SHARE, vm.START_BLUR, vm.START_KNOTS)):
        if name not in inputs or getattr(inputs[name], "default", None) != default or not getattr(inputs[name], "optional", False):
            problems.append(f"`{name}` is not an optional input defaulting to {default!r}: the shipped render would change")


def check_mask_review(problems):
    """Item 12. Each case was seen red against a deliberate break of the view (2026-10-06, thirteen
    breaks in a scratch copy), and each fails on a line below. Three of the breaks make the view
    raise where the others make it draw wrongly, and those are caught and named here so a red says
    what it is: an unknown `replace` that is no longer refused, a token region cut on the wrong
    cycle, and a view not stopped where a cut render stops. That last one would raise in a real
    render too (`render_over`'s concatenation refuses the mismatch); it would not misdraw.
    """
    grow_px = 32
    subject = torch.zeros(FRAMES, H, W)
    subject[:, 48:80, 80:112] = 1.0
    tokens = vm.token_mask(vm.grow(subject, grow_px), LATENT_T, LAT_H, LAT_W)
    source = torch.rand(FRAMES, H, W, 3) * 0.2 + 0.4
    render = source.clone()
    render[:, 40:88, 72:120] = 0.9
    alpha = vm.changed_alpha(render, source, tokens, subject, 8, grow_px // 2, vm.CHANGE_THRESHOLD)
    region = vm.pixel_alpha(tokens, H, W, 0) > 0.5
    reg = vm.token_region(tokens, H, W)
    per = sum(vm.FRAME_PER_TOKEN)
    for at in range(0, FRAMES, per):
        end = min(at + per, FRAMES)
        try:
            piece = reg(at, end)
        except Exception as exc:  # noqa: BLE001 -- a region that cannot be drawn is the finding; everything below draws it
            problems.append(f"mask review: the token region for frames {at}-{end} could not be drawn "
                            f"({type(exc).__name__}: {exc}): it is not cut on the cycle the sampler's tokens are")
            return
        if not torch.equal(piece, region[at:end]):
            problems.append(f"mask review: the token region for frames {at}-{end} is not the region the sampler was given")
    # The first layer is named for what the source's mask is, by its `replace`: only the whole
    # subject's mask is "the tracked subject" (2026-10-06, a parts mask that lay off the subject).
    for value, name in ((vm.REPLACE_WHOLE, "the tracked subject"), (vm.REPLACE_PART, "the head and hair"),
                        (vm.REPLACE_PARTS, "the parts taken")):
        got = vm.window_layers(subject, tokens, H, W, value)
        if got[0].name != name or [layer.name for layer in got[1:]] != ["what else regenerates"]:
            problems.append(f"mask review: with replace {value!r} the layers are {[layer.name for layer in got]}")
    try:
        vm.window_layers(subject, tokens, H, W, "something else")
        problems.append("mask review: a replace the view has no name for was drawn under some name")
    except ValueError:
        pass
    except Exception as exc:  # noqa: BLE001 -- any other failure is not the refusal that names the choices
        problems.append(f"mask review: a replace the view has no name for raised {type(exc).__name__}, "
                        "not the refusal that lists the choices")
    song_text = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "window_layers(src_mask, src_tokens, height, width, source[\"replace\"], kept)" not in song_text:
        problems.append("mask review: the song node does not name the mask's layer from the source's `replace`")
    layers = vm.window_layers(subject, tokens, H, W, vm.REPLACE_WHOLE, alpha)
    if [layer.name for layer in layers] != ["the tracked subject", "what else regenerates", "what the render kept"] \
            or [layer.outline for layer in layers] != [False, False, True]:
        problems.append(f"mask review: today's layers are {[(l.name, l.outline) for l in layers]}")
    legend, _opacity = vm.overlay_legend(layers, H, W, "12% of this window regenerates")
    lh = int(legend.shape[0])
    trim = 5
    out = torch.cat(list(vm.overlay_pieces(source, layers, LATENT_T, trim, "12% of this window regenerates")))
    if tuple(out.shape) != (FRAMES - trim, H, W, 3):
        problems.append(f"mask review: {tuple(out.shape)} for a {FRAMES}-frame window trimmed by {trim}")
        return
    f = 3                                               # a frame of the output; frame f + trim of the window
    src = source[f + trim]

    def tinted(colour, strength, y, x):
        want = src[y, x] * (1.0 - strength) + torch.tensor(colour) * strength
        return bool(torch.allclose(out[f, y, x], want, atol=1e-5))
    kept_edge = (alpha[f + trim] > 0.5)
    if not tinted(*vm.OVERLAY_SUBJECT, 60, 96):
        problems.append("mask review: a pixel of the tracked subject is not in the subject's colour alone")
    white = (out[f] == torch.tensor(vm.OVERLAY_KEPT)).all(dim=-1)
    white[H - lh:] = False                              # the legend's rows have their own white
    if not bool(white.any()) or not bool(kept_edge[white].all()):
        problems.append("mask review: no outline of what the render kept, or one drawn outside it")
    above = torch.zeros(H, W, dtype=torch.bool)
    above[:H - lh] = True                               # everything but the legend's rows
    margin = (region[f + trim] & ~(subject[f + trim] > 0.5) & ~white & above).nonzero()
    far = (~region[f + trim] & above).nonzero()
    if not margin.numel() or not all(tinted(*vm.OVERLAY_REGION, y, x) for y, x in margin[::7].tolist()):
        problems.append("mask review: the regenerated margin is not in the region's colour")
    if not far.numel() or not all(bool(torch.equal(out[f, y, x], src[y, x])) for y, x in far[::37].tolist()):
        problems.append("mask review: a pixel outside the regenerated region is not the source's")
    plain = torch.cat(list(vm.overlay_pieces(source, vm.window_layers(subject, tokens, H, W, vm.REPLACE_WHOLE), LATENT_T)))
    gone = (plain[f + trim] == torch.tensor(vm.OVERLAY_KEPT)).all(dim=-1)
    gone[H - lh:] = False
    if bool(gone.any()):
        problems.append("mask review: an outline was drawn with no composite weight given")
    if bool(torch.equal(plain[0, H - lh:, :8], source[0, H - lh:, :8])):
        problems.append("mask review: no legend in the bottom-left corner")
    extra = layers + [vm.OverlayLayer("a class", (0.2, 0.9, 0.3), lambda at, end: torch.zeros(end - at, H, W, dtype=torch.bool)
                                      .index_fill_(2, torch.arange(0, 8), True))]
    more = torch.cat(list(vm.overlay_pieces(source, extra, LATENT_T, trim)))
    want = src[8, 4] * 0.5 + torch.tensor((0.2, 0.9, 0.3)) * 0.5
    if not bool(torch.allclose(more[f, 8, 4], want, atol=1e-5)) or not tinted(*vm.OVERLAY_SUBJECT, 60, 96):
        problems.append("mask review: a layer added to the list is not drawn, or changed the layers above it")
    if int(vm.overlay_legend(extra, H, 4000)[0].shape[1]) <= int(vm.overlay_legend(layers, H, 4000)[0].shape[1]):
        problems.append("mask review: the legend did not grow with a layer added")
    stacked = torch.cat(list(vm.render_over(render[trim:], vm.overlay_pieces(source, layers, LATENT_T, trim))))
    if tuple(stacked.shape) != (FRAMES - trim, 2 * H, W, 3) or not torch.equal(stacked[:, :H], render[trim:]) \
            or not torch.equal(stacked[:, H:], torch.cat(list(vm.overlay_pieces(source, layers, LATENT_T, trim)))):
        problems.append("mask review: the stacked picture is not the render's frames over the view's")
    # a last window whose tail lies past the track: the view stops where the render does
    try:
        short = torch.cat(list(vm.render_over(render[trim:FRAMES - 3], vm.overlay_pieces(source, layers, LATENT_T, trim))))
    except RuntimeError as exc:
        problems.append("mask review: with the render's tail cut the view runs past the render and cannot be "
                        f"stacked on it ({str(exc)[:80]})")
        short = None
    if short is not None and (int(short.shape[0]) != FRAMES - 3 - trim or not torch.equal(short, stacked[:FRAMES - 3 - trim])):
        problems.append("mask review: with the render's tail cut the stacked picture does not stop with the render")
    if sum(int(x.shape[0]) for x in vm.first_frames(vm.overlay_pieces(source, layers, LATENT_T, trim), 9)) != 9:
        problems.append("mask review: `first_frames` did not stop at the count asked")
    # the song node's switch, read from its source: appended last, optional, on; and out of a stored window's key
    import ast
    tree = ast.parse((REPO / "audio_freeze_song.py").read_text(encoding="utf-8"))
    schema = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "define_schema")
    inputs = next(k.value for c in ast.walk(schema) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "inputs")
    names = [getattr(e.args[0], "value", None) if getattr(e, "args", None) else None for e in inputs.elts]
    at = names.index("save_mask_review") if "save_mask_review" in names else -1
    kw = {k.arg: getattr(k.value, "value", None) for k in inputs.elts[at].keywords} if at >= 0 else {}
    # appended after `source`, so a saved workflow's widget values keep their places (inputs after it are later appends)
    if at < 0 or "source" not in names or at < names.index("source") or kw.get("default") is not True or kw.get("optional") is not True:
        problems.append("mask review: `save_mask_review` is not appended after the song node's `source`, optional and on by default")
    check_review_robust(problems, song_text)


def check_review_robust(problems, song_text):
    """A long render must not be cost by its review, nor join a review that is not of its windows.

    The writer and the wrapper run for real here (ffmpeg, a few small frames); the song node's use of
    them is read from its source, since nothing here can run the node: its windows need the models.
    What that leaves unrun is the node's own lines between the two.
    """
    import importlib
    import os
    import tempfile
    import comfy.model_management as mm
    song = importlib.import_module("_h3pack.audio_freeze_song")
    frames = torch.rand(5, 96, 160, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "w_1_with_mask.mp4")
        if song._write_review_mp4(path, iter([frames]), 160, 96, 19) != 5 or os.listdir(tmp) != ["w_1_with_mask.mp4"]:
            problems.append(f"mask review: a review written whole is not one file under its final name: {os.listdir(tmp)}")
        os.remove(path)

        def cut_short():
            yield frames
            raise RuntimeError("the encode was cut short")
        try:
            song._write_review_mp4(path, cut_short(), 160, 96, 19)
            problems.append("mask review: a review whose frames stopped part way did not raise")
        except RuntimeError:
            pass
        if os.listdir(tmp):
            problems.append(f"mask review: a review cut short left {os.listdir(tmp)} behind; a file under the final "
                            "name would be joined by the next run as if it were whole")

    def breaks():
        raise RuntimeError("no room left")
    why = song._review_or_reason(breaks, "a test")
    if not why or "RuntimeError" not in why or "no room left" not in why:
        problems.append(f"mask review: a review that failed was reported as {why!r}, not as what went wrong")
    if song._review_or_reason(lambda: None, "a test") is not None:
        problems.append("mask review: a review that worked was reported as a failure")

    def stopped():
        raise mm.InterruptProcessingException()
    try:
        song._review_or_reason(stopped, "a test")
        problems.append("mask review: an interrupt during a review was swallowed; the run would go on")
    except mm.InterruptProcessingException:
        pass
    # The node's side, from its source. The review's write goes through the whole-or-absent writer and
    # runs under the wrapper; a window that renders removes its old review with its old latent; and
    # what a render owes whether or not the review worked comes after the wrapped join, not inside it.
    if "_write_pieces_mp4(loop_resume.review_path" in song_text or "_write_pieces_mp4(path, video_mask.first_frames" in song_text \
            or song_text.count("_write_review_mp4(") != 2:
        problems.append("mask review: the song node writes a review straight to its final name")
    if "why = _review_or_reason(window_review, f\"window {w.number}\")" not in song_text \
            or "why = _review_or_reason(joined_review, \"the run\")" not in song_text:
        problems.append("mask review: a failure in a review would fail the render")
    if "for stale in (latent_path, loop_resume.review_path(work_dir, filename, w.number), region_path, zoomed_path):" not in song_text:
        problems.append("mask review: a window that renders again keeps its old review for a later run to join")
    at = song_text.find("why = _review_or_reason(joined_review")
    owed = [song_text.find(mark, at) for mark in ("write_metadata_png(os.path.join(full_out", "shot_table.write_beside(source",
                                                   "return io.NodeOutput(")]
    if at < 0 or min(owed) < 0 or owed != sorted(owed):
        problems.append("mask review: the render's metadata, shot table or outputs no longer follow the review's join")


def check_loader_cap(problems, path, graph, loader_id, song, plan):
    """Item 11: the source loader's cap against what the song node's plan reads."""
    import math
    loader, ins = graph.get(loader_id, {}), song["inputs"]
    cap = loader.get("inputs", {}).get("frame_load_cap")
    if cap is None:
        problems.append(f"{path.name}: the source loader {loader_id} has no frame_load_cap to hold to the plan")
        return
    if ins.get("extent") != "first_seconds":
        problems.append(f"{path.name}: the song node's extent is {ins.get('extent')!r}, so how many frames the "
                        "loader must hold cannot be read from the graph")
        return
    total = int(math.ceil(float(ins["extent.seconds"]) * plan.FPS))     # as `audio_freeze_song.py::execute`
    reads = plan.frames_read(total, int(ins["window_frames"]), int(ins["context_frames"]), str(ins.get("timeline") or ""))
    if int(cap) != reads:
        problems.append(f"{path.name}: the loader's frame_load_cap is {int(cap)} and the song node's plan reads "
                        f"{reads} frames for {ins['extent.seconds']} s at windows of {ins['window_frames']} with "
                        f"{ins['context_frames']} of context" + (": the tracker works on frames nothing renders"
                                                                 if int(cap) > reads else ": the plan runs out of picture"))


def check_encode_frames(problems):
    """`encode_frames` feeds `window`'s `encode` without changing `source["frames"]`, the masks, or the zoom box."""
    n, h, w = 9, 96, 160
    frames = torch.full((n, h, w, 3), 0.2)
    encode = torch.linspace(0.3, 0.9, h * w, dtype=torch.float32).reshape(1, h, w, 1).expand(n, h, w, 3).clone()
    mask = torch.zeros(n, h, w)
    mask[:, 24:56, 48:96] = 1.0
    got = _appended("encode_frames")
    if got is None or not got.optional:
        problems.append("encode_frames: it is not where it was appended (at the end of `APPENDED`) and optional")
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=16, feather_pixels=4, encode_frames=encode)
    src = (out.args if hasattr(out, "args") else out)[0]
    if src.get("encode_frames") is None or not torch.equal(src["encode_frames"], encode):
        problems.append("encode_frames: the node did not carry `encode_frames` onto the source dict")
    if not torch.equal(src["frames"], frames):
        problems.append("encode_frames: wiring `encode_frames` altered `source['frames']`")
    win = vm.window(src, 0, n, w, h, 3, 12, 20)
    if not torch.allclose(win[0], frames, atol=1e-5) or not torch.allclose(win[1], encode, atol=1e-5):
        problems.append("encode_frames: `window` did not keep `pixels` as `frames` and take `encode` from `source['encode_frames']`")
    zsrc, _zlines = vm.zoomed_source(dict(src, render_region="zoomed in on what is replaced", zoom_up_to=2.0), w, h)
    expected_zenc = vm.crop_frames(encode, zsrc["zoom"]["box"], w, h) if zsrc.get("zoom") else None
    if (expected_zenc is None or zsrc.get("encode_frames") is None
            or torch.equal(expected_zenc, encode) or not torch.equal(zsrc["encode_frames"], expected_zenc)):
        problems.append("encode_frames: `zoomed_source` did not crop `encode_frames` alongside `frames`")
    win39_f = torch.full((39, h, w, 3), 0.2)
    shot35_e = torch.full((35, h, w, 3), 0.7)
    shot35_m = torch.zeros(35, h, w)
    shot35_m[:, 24:56, 48:96] = 1.0
    out39 = vm.MiniMaxH3MaskedSource.execute(win39_f, shot35_m, grow_pixels=16, feather_pixels=4, encode_frames=shot35_e)
    src39 = (out39.args if hasattr(out39, "args") else out39)[0]
    if tuple(src39["encode_frames"].shape) != (39, h, w, 3):
        problems.append(f"encode_frames: 35-frame shot in 39-frame window gave {tuple(src39['encode_frames'].shape)}")
    for label, bad in (("another frame count", encode[:-1]), ("another size", torch.zeros(n, h // 2, w // 2, 3))):
        try:
            vm.MiniMaxH3MaskedSource.execute(frames, mask, grow_pixels=16, feather_pixels=4, encode_frames=bad)
            problems.append(f"encode_frames: {label} was accepted")
        except ValueError as exc:
            if "encode_frames" not in str(exc):
                problems.append(f"encode_frames: refusal of {label} did not name `encode_frames`: {exc}")


def check_graphs(problems):
    seen = 0
    plan = _load("loop_plan")
    for path in h3_config.graph_paths(WORKFLOWS, include_bench=True):
        graph = json.loads(path.read_text())
        for nid, node in graph.items():
            if not isinstance(node, dict) or node.get("class_type") != SOURCE:
                continue
            seen += 1
            ins = node["inputs"]
            users = [n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == SONG
                     and n["inputs"].get("source") == [nid, 0]]
            if not users:
                problems.append(f"{path.name}: Masked Source {nid} feeds no song node's `source`")
                continue
            frames_from = ins["frames"][0]
            # walk the mask back to the frames its tracker saw: the pack's Subject Track
            # takes them itself; core's chain reaches them through its track data
            mask_node = graph[ins["mask"][0]]
            if "track_data" in mask_node["inputs"]:
                track = graph.get(mask_node["inputs"]["track_data"][0], {})
                tracked_from = track.get("inputs", {}).get("images", [None])[0]
            else:
                tracked_from = mask_node["inputs"].get("frames", [None])[0]
            if tracked_from != frames_from:
                problems.append(f"{path.name}: the mask of Masked Source {nid} was not tracked over its own frames")
            # A review graph is the one place the tiles are wired: its song node is on `preview`,
            # so it samples nothing.
            reviewing = all(n["inputs"].get("preview") is True for n in users)
            for out in (1, 2):  # a consumer of the tracker's preview or report runs it on every queue
                if not reviewing and mask_node.get("class_type") == "MiniMaxH3SubjectTrack" and any(
                        v == [ins["mask"][0], out] for n in graph.values() if isinstance(n, dict)
                        for v in n.get("inputs", {}).values()):
                    problems.append(f"{path.name}: the Subject Track's preview or report is wired, which defeats the kept mask")
            for song in users:
                check_loader_cap(problems, path, graph, frames_from, song, plan)
                if song["inputs"].get("save_mask_review") is not True:
                    problems.append(f"{path.name}: a graph that wires a Masked Source does not write `save_mask_review`")
                if song["inputs"].get("audio", [None])[0] != frames_from:
                    problems.append(f"{path.name}: the song node's track is not the audio of the source video")
                if "segmenter" not in ins or "segmenter_clip" not in ins:
                    problems.append(f"{path.name}: Masked Source {nid} has no segmenter wired, so `replace` cannot be changed without rewiring")
                if "references" not in song["inputs"]:
                    problems.append(f"{path.name}: a masked source with no reference still to replace the subject from")
    if not seen:
        problems.append("no shipped graph wires a Masked Source; item 10 checked nothing")


def main() -> int:
    problems: list[str] = []
    for check in (check_temporal, check_token_grid, check_feather, check_composite, check_window, check_keep, check_edge, check_fit, check_paint_out, check_part, check_changed_alpha, check_motion_reference, check_motion_zoom, check_late_start, check_grow_by, check_others, check_region_rules, check_queue_time_refusals, check_wired_motion, check_subject_boxes, check_cut_gate, check_lay_window, check_song_loop, check_motion_look, check_held_tail, check_context_shows, check_render_zoom, check_mask_review, check_encode_frames, check_graphs):
        check(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    the masked source keeps every subject frame, sits on core's token grid, takes a keep mask out "
              "after the grow in whole tokens and only then, feathers off the "
              "subject, composites exactly, holds a short source, crops the mask as the frames, paints out only "
              "inside the regenerated tokens, takes a part only from the subject, restores the margin under "
              "`only what changed`, builds a motion reference on grey or whole at the short edge asked, softens only the subject and empties only its body's tokens for a late start, is wired whole in every graph, its loader loads the frames the plan reads, the mask review shows what regenerates, and a margin taken from the subject's size holds the region under its bound where a fixed one does not, and the margin stays off the people round the subject without costing the subject a token, and a keep-out's speck does not hole a step, a one-cell hole is closed and a piece cut off from the subject is given back, and a setting it will refuse is refused at queue time in the same words, and a wired motion video is cut where the window is cut, and a tracked mask becomes one box a frame with none where the subject is not, and nothing is laid across a cut from the subject, and a short load's padded frames show the subject as filmed or with the region held from the shot's edge frames, as the load says")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
