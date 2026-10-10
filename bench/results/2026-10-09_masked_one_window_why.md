# Why three masked renders of one window each came out wrong: ten more renders, each one change, and what was measured (2026-10-09)

lane: masked video to video
verdict: on one window of one clip with one lead: the wrong-way start is the node's own text on a window that opens on the lead's back, and the sentences saying which way he faces fix it alone; the other man's look came at one seed of two and not without the motion video; the half cap is in the model's draw and tied to that one opening; the tracker's mask sits on its own frame throughout; about half of what is redrawn is not the lead, and a lower margin brings the other man's look back for a stretch. Partly watched by the owner; nothing is a default.

**Read this first.** One source video, one lead (one still, one set of
clothing), one scene, one window, and one render a cell. The owner watched
six of these as they landed and their notes are below; every other reading
is a still or a measurement, and a still cannot show what must hold over
time. Nothing here is claimed for another clip, person or setting. This
continues [`2026-10-09_masked_text_and_edge_one_window.md`](2026-10-09_masked_text_and_edge_one_window.md),
whose three renders are the control, the text render and the edge render
named here, and it corrects two things that record said.

**Two corrections to that record.** The window is two frames before the one
intended, not one: `masked_render_against_source.py align`, tested the same
day on a clip cut from known frames, puts the renders on the copy's frames
the json gives, and the loader's own ffmpeg arguments reproduce it exactly
(the copy carries two frames before time zero). And the reading of the cut
cap as the composite's doing is refuted below.

## What was asked

The owner, on playback: the control faces the wrong way and walks backward,
becomes the other man at about five seconds and a third person at the end;
the text render is right but for half a cap missing on its first frames; the
edge render is the control. Then: find out why, and be sure; check whether
anything cached is used; check the frame rule, seams and whether the
tracker's frames are the model's frames; and the margin, which is what the
day set out to test.

## The renders

Each differs from the control, or from the text render where it says so, in
the one thing named. The json beside this file
([`2026-10-09_masked_one_window_why.json`](2026-10-09_masked_one_window_why.json))
has each one's prompt id, file, report lines and what ComfyUI's cache served
it.

| render | the change | what it asks |
|---|---|---|
| t1 | the text render, `composite` = `whole region` | did the composite remove half the cap |
| t2 | the control, next seed | is the control one seed's draw |
| t3 | the control, no motion reference | does the other man's look come through the motion video |
| t4 | the control plus the sentences saying which way he faces | which sentence matters |
| t5 | the control with only the retained face narrowed | which sentence matters |
| t6 | the control on the window at the clip's start, which opens on his face | does the default text fail only when the face is hidden |
| t7 | the text render started twelve frames earlier | does the cut cap follow the opening or the place |
| t8 | the text render, fixed 16 px margin | what a lower margin does |
| t9 | the text render, no margin and no blend | the same |
| t10 | t9 with `edge` = `latent cells` | the smallest region the lane makes |

## Which way he faces

`bench/measure_subject_yaw.py`, every second frame, the box from the
tracker's own mask saved as a video. Zero is facing the camera and 180 the
back. The json has every curve.

| | faces the camera, first third | middle third | last third | last frame with his back to it | mean difference from the source |
|---|---|---|---|---|---|
| source | 2% | 5% | 24% | 230 | |
| control | 95% | 5% | 59% | 204 | 78 |
| control, seed 2 (t2) | 45% | 12% | 78% | 220 | 70 |
| no motion reference (t3) | 100% | 88% | 83% | 180 | 125 |
| face sentence only (t5) | 28% | 10% | 34% | 242 | 45 |
| view sentences only (t4) | 0% | 0% | 68% | 190 | 36 |
| text | 0% | 0% | 56% | 196 | 31 |
| margin 16 (t8) | 0% | 10% | 46% | 208 | 32 |
| margin 0 (t9) | 10% | 7% | 39% | 210 | 33 |
| margin 0, latent cells (t10) | 0% | 0% | 37% | 220 | 23 |

- **The wrong-way start is the node's text on this window.** With it he
  faces the camera at the start at both seeds, where the source shows a back.
  The sentences that say which way he faces put it right alone (t4);
  narrowing the retained face alone does not (t5).
- **The motion video is what turns him at all.** Without it he faces the
  camera through most of the window (t3).
- **Every render turns to the camera before the source does**, by one to two
  seconds, the text render included. The three with a lower margin turn
  later, nearer the source.
- **The default text is not wrong everywhere.** On the window that opens on
  his face (t6) the stills read show him turn with the source and stay the
  lead. That window was not measured, only read.

## Who he is

A measure for this clip only: of the pixels a render changed from the
source, how many are the lead's blue T-shirt and how many a white shirt,
which the source's man and the crowd wear. The json has the frame each
render is lost from, by its rule.

- The control and t5 (same seed, nearly the same text) draw a white shirt
  from the middle of the window on. The control at the next seed (t2) does
  not, and neither does the control without the motion video (t3). So the
  other man's look is not every seed's, and on the one pair that tests it,
  it did not come without the motion video.
- The text render and t4 keep the T-shirt throughout.
- **All three lower margins draw the white shirt for a stretch in the middle
  and then recover**, with the text that otherwise holds. It is drawn, not
  shown through: measured against the tracker's own mask, almost none of the
  original's shirt lies outside the redrawn region at any margin (the json
  has the count). The stretch overlaps the frames on which the part model
  reused another frame's mask; that is an overlap, not a shown cause.

## The cap

- **Not the composite.** t1 keeps the whole region and is the text render on
  the cap to within the codec (`landing`, a box on the cap, in the json).
- **Not the region's edge.** The missing half is inside the redrawn region
  on the mask review.
- **Tied to that opening.** Started twelve frames earlier (t7), the cap is
  whole on the stills read, on its own first frames and on the frame the
  original start opened on. The owner saw the cut in t1 and t4, which share
  the original start.
- **Not known:** whether the half is absent or drawn black on a near-black
  background, and why that opening does it.

## The halo, and the margin

- **About half of what is redrawn is not the lead**, and three to five times
  his area where he is small (read off the mask review; the json has it by
  frame). The finer edge alone moves that little.
- **The pixels just inside the region's edge flicker at about five times the
  untouched pixels beside them**, in every render measured
  (`masked_render_against_source.py flicker`). A lower margin does not
  change that figure; it changes how much such area there is.
- **The composite keeps most of the region** on this crowd (each render's
  own report line, in the json): the model has to invent the people behind
  him, the invention differs from the real ones, and a test that asks only
  whether a pixel changed keeps it.
- **A lower margin has a cost here**: the white shirt above, and one render
  with a stray frame facing the camera (t9, in the facing curves).

## Is the tracker on the model's frames

- One loader feeds the tracker, the part model, the Masked Source and the
  Song node (read off the graph). There is one frame list.
- `masked_render_against_source.py timing`, with the tracker's own mask:
  each frame's mask fits the source's frame of the same number on every
  stretch of the window, by a wide lead. Its control: masks shifted two and
  three frames on purpose are reported at exactly those shifts on every
  stretch. With the mask recovered from the review it is not clean, because
  that mask is grown past his outline and he is shrinking in frame.
- The shot table beside each render has him on every frame with no loss.
- The part model's reused masks, on the frames the earlier record lists, are
  the one place a frame carries a mask cut from another.
- Frame count, canvas and joins: the count fits the frame rule
  (`video_mask.run_lengths` sums to it), the canvas is whole tokens, and a
  single window has no join.

## Was anything cached

Nothing from this pack: no shipped graph turns a reuse on, and
`bench/check_mask_store.py` item 9 now holds that. ComfyUI's own node cache
served the loader, the track and the parts to most of these renders (the
json has the count for each), so they rendered from one mask; the render
after a changed Masked Source ran that node again. `docs/comfy_notes.md`,
"What is cached, and by whom", says how it is keyed.

## What this does not settle

- One seed for every arm but the control. The sentence result is one pair
  each.
- t3 changes the reference and the text that names it together.
- Why the model draws the other man's look at a low margin, and why the cap
  is cut at that opening.
- Whether any of it holds on another clip or another lead.

## What follows, for the owner to choose

- The node's text says nothing about the view and keeps the face in every
  frame; on a window that opens on a back that is the measured fault. The
  tracker does not know which way a subject faces; the body model used for
  the facing measure does.
- A composite that keeps the new subject by its own outline, found in the
  render, is the piece that would let the region stay generous without the
  invented crowd; a lower margin by itself traded the halo for the other
  man's look on this window.
- The tools that captured and analysed the 2026-10-08 final live in a
  session folder under `internal/` and are written for that one clip. Three
  of their jobs are now modes of `bench/masked_render_against_source.py`
  (`align`, `timing`, and `flicker` on the measured frame); the numbered
  review and the mask recovery are not in the tracked tree.
