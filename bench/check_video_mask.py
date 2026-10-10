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
    plate; `motion_video` is the node's last input, optional, and not in the
    kept mask's key; and the song node takes the wired branch.
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
    latent, removes a stale one with the stale latent, and never assigns the
    count of reused windows again inside its loop (08c3cb12 did, and the
    report's first line then gave a frame number for it).
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
    The report opens with no window reused and says which frames lie across
    a cut, and they are the frames `loop_plan.split_steps` names for the
    load. In the joined video those frames are the source's and a frame the
    subject is on carries the bright cells. Each window leaves a region
    file that reads back as `window` makes the window's mask and tokens,
    with its first frame and trim; a stale one from an earlier run is
    replaced; and with `keep_windows` off the folder is gone.

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
            vm.MiniMaxH3MaskedSource.execute(frames, mask, reuse_mask=False, keep=bad, **kw)
            problems.append(f"keep: {label} was accepted")
        except ValueError:
            pass
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, reuse_mask=False, keep=keep.unsqueeze(-1))
    src = out.args[0] if hasattr(out, "args") else out[0]
    if src.get("keep") is None or tuple(src["keep"].shape) != tuple(keep.shape):
        problems.append("keep: the node did not carry a [N, H, W, 1] keep mask onto the source as [N, H, W]")
    if "keep" not in vm.MASK_KEY_SKIP:
        problems.append("keep: it is applied after the subject's mask is final and must not be in the kept mask's key")


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
    if tuple(got.shape) != (40,) or not torch.equal(got[:30], each[120:150]) or set(got[30:].tolist()) != {cap}:
        problems.append("grow_by: a window's margins are not its own frames', held at the cap past the source's end")
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
    if "grow_by" not in vm.MASK_KEY_SKIP:
        problems.append("grow_by: it acts after the mask is final and must not be in the kept mask's key")
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
    inputs = vm.MiniMaxH3MaskedSource.define_schema().inputs
    # appended inputs keep their place: `motion_video` (2026-10-10) came after it
    if inputs[-2].id != "edge" or not inputs[-2].optional or inputs[-2].default != vm.EDGE_TOKENS:
        problems.append("edge: it is not where it was appended (the input before `motion_video`), optional, "
                        "defaulting to whole tokens")
    if "edge" not in vm.MASK_KEY_SKIP:
        problems.append("`edge` is not in MASK_KEY_SKIP: a change of edge would track the subject again")


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
    inputs = node.define_schema().inputs
    if inputs[-1].id != "motion_video" or not inputs[-1].optional:
        problems.append("motion_video: it is not the node's last input and optional")
    if "motion_video" not in vm.MASK_KEY_SKIP:
        problems.append("motion_video: it is shown to the model and does not make the mask, so it must not be in the "
                        "kept mask's key")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "video_mask.wired_motion(source, int(round(w.start * FPS)), w.frames, width, height)" not in song:
        problems.append("wired motion: the song node does not cut the wired video at the window's own start")


def check_subject_boxes(problems):
    spec = importlib.util.spec_from_file_location("_h3pack.subject_boxes", REPO / "subject_boxes.py")
    sb = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.subject_boxes"] = sb
    spec.loader.exec_module(sb)
    mask = torch.zeros(4, H, W)
    mask[0, 20:40, 30:50] = 1.0
    mask[1, 0:10, 0:12] = 1.0                    # against the frame's corner
    mask[3, H - 6:H, W - 9:W] = 1.0             # against the far corner; frame 2 is empty
    plain = sb.frame_boxes(mask)
    if plain != [[{"x": 30, "y": 20, "width": 20, "height": 20}], [{"x": 0, "y": 0, "width": 12, "height": 10}], [],
                 [{"x": W - 9, "y": H - 6, "width": 9, "height": 6}]]:
        problems.append(f"subject boxes: the boxes are not the masks' bounds, or an empty frame has one: {plain}")
    wide = sb.frame_boxes(mask, 8)
    if wide[0] != [{"x": 22, "y": 12, "width": 36, "height": 36}] or wide[1] != [{"x": 0, "y": 0, "width": 20, "height": 18}] \
            or wide[3] != [{"x": W - 17, "y": H - 14, "width": 17, "height": 14}] or wide[2] != []:
        problems.append(f"subject boxes: a margin does not widen the box and stop at the frame's edge: {wide}")
    if sb.frame_boxes(mask.unsqueeze(-1)) != plain:
        problems.append("subject boxes: a mask with a trailing channel is read differently")
    out = sb.MiniMaxH3SubjectBoxes.execute(mask, 0)
    if out.args[0] != plain or "3 of 4 frames" not in out.args[1] or "frame 2" not in out.args[1]:
        problems.append(f"subject boxes: the node's boxes or its report are not the function's: {out.args[1]}")
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
    if '"cuts": source_cuts(frames, table)' not in (REPO / "video_mask.py").read_text(encoding="utf-8"):
        _fail(problems, "cut gate: the Masked Source's record does not carry the source's cuts")


def check_lay_window(problems):
    """Item 20. The song node's composite as one function, and the file a window's region is saved in."""
    import tempfile

    grow_px, feather = 32, 8
    runs = vm.run_lengths(LATENT_T)
    starts = [sum(runs[:k]) for k in range(len(runs))]
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
    free, _a, quiet = vm.lay_window(render, pixels, tokens, old, {**record, "cuts": []}, grow_px, first)
    moved = sorted({int(f) for f in (out != free).flatten(1).any(dim=1).nonzero().flatten()})
    if moved != across:
        _fail(problems, f"lay window: with and without the cut the frames that differ are {moved}, not the gated {across}")
    if len(lines) != 2 or "only what changed" not in lines[0] or not lines[1].endswith(", ".join(str(first + f) for f in across)) \
            or len(quiet) != 1:
        _fail(problems, f"lay window: the report's lines are {lines} with the cut and {quiet} without")
    whole, w_alpha, w_lines = vm.lay_window(render, pixels, tokens, old, {**record, "composite": vm.COMPOSITE_REGION, "cuts": []},
                                            grow_px, first)
    if not torch.equal(w_alpha, vm.pixel_alpha(tokens, H, W, feather)) or not torch.equal(whole, vm.composite(render, pixels, w_alpha)) \
            or w_lines:
        _fail(problems, "lay window: under `whole region` the weight is not the region's own, or a line was reported")

    # the file: what was written comes back, and a window laid from it is the window laid from the tensors
    each = torch.full((FRAMES,), grow_px, dtype=torch.long)
    each[::2] = grow_px // 2
    with tempfile.TemporaryDirectory() as tmp:
        for margin in (grow_px, each):
            path = vm.save_window_region(str(Path(tmp) / "w_region.npz"), old, tokens, record, margin, first, 5)
            got = vm.load_window_region(path)
            same_margin = torch.equal(got["margin"], margin) if torch.is_tensor(margin) else got["margin"] == margin
            if not torch.equal(got["mask"], (old > 0.5).float()) or not torch.equal(got["tokens"], tokens) or not same_margin \
                    or got["first_frame"] != first or got["trim"] != 5 or got["source"] != {key: record[key] for key in vm.REGION_SETTINGS}:
                _fail(problems, f"window region: the file does not read back as written (margin {vm.margin_note(margin)})")
                continue
            want = vm.lay_window(render, pixels, tokens, old, record, margin, first)[0]
            again = vm.lay_window(render, pixels, got["tokens"], got["mask"], got["source"], got["margin"], got["first_frame"])[0]
            if not torch.equal(want, again):
                _fail(problems, f"window region: a window laid from its file is not the window laid from its tensors "
                                f"(margin {vm.margin_note(margin)})")
        if sorted(x.name for x in Path(tmp).iterdir()) != ["w_region.npz"]:
            _fail(problems, f"window region: the write left {sorted(x.name for x in Path(tmp).iterdir())} behind")

    song_text = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    if "video_mask.lay_window(images, src_pixels, src_tokens, src_mask, source," not in song_text \
            or any(name in song_text for name in ("video_mask.composite(", "video_mask.changed_alpha(", "video_mask.cut_gate(")):
        _fail(problems, "lay window: the song node must lay a window through `video_mask.lay_window` and through nothing else")
    if "video_mask.save_window_region(region_path, src_mask, src_tokens, source, margin," not in song_text \
            or "loop_resume.review_path(work_dir, filename, w.number), region_path):" not in song_text:
        _fail(problems, "window region: the song node does not save a window's region beside its latent, or keeps a stale one")
    if song_text.count("        first = ") != 1 or "        first = len(reused)" not in song_text:
        _fail(problems, "lay window: the song node assigns `first` (its count of reused windows) more than once")


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

        def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
            video, audio = latent_image.unbind()
            video = video.clone()
            asked = denoise_mask.unbind()[0].expand_as(video[:, 0:1]) > 0.5
            video[:, 0:1] = torch.where(asked, torch.ones_like(video[:, 0:1]), video[:, 0:1])
            return comfy.nested_tensor.NestedTensor((video, audio))

    class Conditioning:
        @staticmethod
        def execute(*args, **kwargs):
            return ([["a stand-in", {}]],)

    track = {"waveform": torch.zeros(1, 2, total * 32000 // 24), "sample_rate": 32000}

    def run(out_dir: str, prefix: str, **more):
        real = (song.Guider_Basic, song.MiniMaxH3Conditioning, folder_paths.get_output_directory,
                song.comfy.sample.fix_empty_latent_channels, song.latent_preview.prepare_callback)
        song.Guider_Basic, song.MiniMaxH3Conditioning = Sampler, Conditioning
        folder_paths.get_output_directory = lambda: out_dir
        song.comfy.sample.fix_empty_latent_channels = lambda model, latent, *a, **k: latent
        song.latent_preview.prepare_callback = lambda *a, **k: None
        try:
            got = song.MiniMaxH3AudioFreezeSong.execute(
                object(), Encoder(), VideoVAE(), AudioVAE(), track, object(), torch.linspace(1.0, 0.0, 5), "a prompt", "",
                False, width, height, window, context, "whole", 7, 0.0, "clip_guard", prefix, 4,
                save_metadata_png=False, source=source, **more)
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
        if not lines[0].startswith("0 reused -> ") or "NOT saved" in report or "FAILED" in report:
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
        gone = run(tmp, "u/run", keep_windows=False)
        if os.path.isdir(Path(tmp) / "u" / "run_windows") or not os.path.isfile(gone[0]):
            _fail(problems, f"song loop: with keep_windows off the windows folder still holds "
                            f"{sorted(os.listdir(Path(tmp) / 'u' / 'run_windows')) if os.path.isdir(Path(tmp) / 'u' / 'run_windows') else 'nothing'}"
                            ", or the joined video is missing")


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
    # a window past the source's end: the held frames have no mask and no others
    late = vm.window(dict(base, others=others), n - 5, n, w, h, *shape)
    if late[4] != n - 5 or float(late[2][-1].max()) != 0.0:
        problems.append("others: a window past the source's end did not hold its last frame unmasked")
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
    # appended inputs keep their place: a later one goes after, never between (`edge`, 2026-10-09, is the next,
    # then `motion_video`, 2026-10-10)
    if inputs[-3].id != "others" or not inputs[-3].optional:
        problems.append("others: it is not where it was appended (the input before `edge`) and optional")
    if "others" not in vm.MASK_KEY_SKIP:
        problems.append("others: it acts after the mask is final and must not be in the kept mask's key")


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
    elif float(tokens[-1].max()) != 0.0:
        problems.append("the held frames past the source's end are masked: they would regenerate")
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
    asked = vm.MiniMaxH3MaskedSource.check_lazy_status(frames=frames, replace=vm.REPLACE_PARTS, reuse_mask=False,
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
    if not all(k in vm.MASK_KEY_SKIP for k in ("motion_reference", "motion_short_edge", "motion_vae")):
        _fail(problems, "motion_reference's three inputs must not enter the kept mask's key: they do not change the mask")
    import reference_order as ro
    if ro.MOTION_NONE != vm.MOTION_NONE or ro.MASKED_SOURCE_CLASS != "MiniMaxH3MaskedSource":
        _fail(problems, "reference_order's literals for the Masked Source must equal video_mask's: the static label plan reads them")
    song_inputs = {"source": ["9", 0]}
    g = {"9": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"motion_reference": vm.MOTION_SUBJECT}}}
    if ro.assign_labels(ro.plan_for(song_inputs, g)) != ["<Video 1>"]:
        _fail(problems, "a Masked Source with a motion reference must add one <Video N> to the song node's static label plan")
    g["9"]["inputs"]["motion_reference"] = vm.MOTION_NONE
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
    out = vm.MiniMaxH3MaskedSource.execute(frames, mask, replace=vm.REPLACE_PARTS, parts=parts, part_margin=0, reuse_mask=False,
                                           motion_reference=vm.MOTION_ZOOM, motion_short_edge=SHORT, shot_table=table)
    out = getattr(out, "args", out)
    record = out[0]
    if not torch.equal(record["subject_boxes"], rows):
        _fail(problems, "zoom: the source record's `subject_boxes` must be the tracker's mask's boxes, not the replaced part's")
    if vm.shot_ranges(record) != [(0, 2), (3, 5)] or vm.shot_ranges({"shot_table": ""}) != [] or vm.shot_ranges({"shot_table": "{"}) != []:
        _fail(problems, "zoom: the cuts must be read from the record's shot table, and an empty or unreadable one is no cuts")
    win = vm.window_boxes(record, 0, 6, W, H)
    if win is None or not torch.equal(win, per_shot):
        _fail(problems, "zoom: a window's boxes on a canvas the size of the frames must be the shots' boxes")
    held = vm.window_boxes(record, 4, 6, W, H)
    if held is None or held[0].tolist() != per_shot[4].tolist() or not (held == held[0]).all():
        _fail(problems, f"zoom: frames past the source's end carry their shot's box; got {None if held is None else held.tolist()}")
    if vm.window_boxes({"frames": frames}, 0, 6, W, H) is not None:
        _fail(problems, "zoom: a source with no boxes must say so with None")
    plain = getattr(vm.MiniMaxH3MaskedSource.execute(frames, mask, reuse_mask=False, motion_reference=vm.MOTION_SUBJECT,
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
        if name not in vm.MASK_KEY_SKIP:
            problems.append(f"`{name}` is not in MASK_KEY_SKIP: changing it would track the subject afresh")


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
    if song_text.count('source["replace"]') < 2 or "window_layers(src_mask, src_tokens, height, width, source[\"replace\"], kept)" not in song_text:
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
    if "save_mask_review" not in _load("loop_resume").SONG_PER_WINDOW:
        problems.append("mask review: turning `save_mask_review` on or off would re-render every stored window")
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
    # The node's side, from its source. Both writes go through the whole-or-absent writer and both
    # run under the wrapper; a window that renders removes its old review with its old latent; and
    # what a render owes whether or not the review worked comes after the wrapped join, not inside it.
    if "_write_pieces_mp4(loop_resume.review_path" in song_text or "_write_pieces_mp4(path, video_mask.first_frames" in song_text \
            or song_text.count("_write_review_mp4(") != 3:
        problems.append("mask review: the song node writes a review straight to its final name")
    if "why = _review_or_reason(window_review, f\"window {w.number}\")" not in song_text \
            or "why = _review_or_reason(joined_review, \"the run\")" not in song_text:
        problems.append("mask review: a failure in a review would fail the render")
    if "for stale in (latent_path, loop_resume.review_path(work_dir, filename, w.number), region_path):" not in song_text:
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
            # so it samples nothing and there is no render whose kept mask could be defeated.
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
    for check in (check_temporal, check_token_grid, check_feather, check_composite, check_window, check_keep, check_edge, check_fit, check_paint_out, check_part, check_changed_alpha, check_motion_reference, check_motion_zoom, check_late_start, check_grow_by, check_others, check_queue_time_refusals, check_wired_motion, check_subject_boxes, check_cut_gate, check_lay_window, check_song_loop, check_mask_review, check_graphs):
        check(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    the masked source keeps every subject frame, sits on core's token grid, takes a keep mask out "
              "after the grow in whole tokens and only then, feathers off the "
              "subject, composites exactly, holds a short source, crops the mask as the frames, paints out only "
              "inside the regenerated tokens, takes a part only from the subject, restores the margin under "
              "`only what changed`, builds a motion reference on grey or whole at the short edge asked, softens only the subject and empties only its body's tokens for a late start, is wired whole in every graph, its loader loads the frames the plan reads, the mask review shows what regenerates, and a margin taken from the subject's size holds the region under its bound where a fixed one does not, and the margin stays off the people round the subject without costing the subject a token, and a setting it will refuse is refused at queue time in the same words, and a wired motion video is cut where the window is cut, and a tracked mask becomes one box a frame with none where the subject is not, and nothing is laid across a cut from the subject")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
