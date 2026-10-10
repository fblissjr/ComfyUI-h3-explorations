# One load per subject shot against one long load, for a whole-subject masked pass (2026-10-10)

lane: masked video to video
verdict: not judged. Two shots of one clip, one seed, one render each way, read from tables and from stills; nobody has watched either. With the same region to the cell, the load made for one shot changed about a third of the region and left the things in front of the subject as the source's; the long load over several shots changed over half of it and redrew them.

**Read this first.** Two shots is not a sample, and both are the same
subject in the same setting. The two ways also differ in more than length:
the per-shot load is one short window whose tail is a held last frame, and
the long load is two long windows with other shots between the subject's.
Which of those does it is not separated here. A builder default rests on
this record; it is a first read.

**What was asked.** A whole-subject pass over a stretch with several cuts
was first rendered as one load, every shot the subject is in sharing two
long windows with the shots the subject is not in. The same two shots were then
rendered each as its own load, same still, text, seed and motion video (a
body mesh), and set against the long load's frames of the same shots.

## How

Each shot is a capture folder (`bench/capture_masked_run.py files`,
untracked `data/`) with both renders as runs, the region read from each
render's own review. The numbers are in the json beside this file
([`2026-10-10_one_load_per_shot_against_one_long_load.json`](2026-10-10_one_load_per_shot_against_one_long_load.json)),
one row a shot and render: the region's share of the frame; the share of
the region more than a threshold from the source; the mean difference from
the source on the subject's mask and in a fixed box beside it that holds
things which are not the subject; the change from the frame before inside the region, beside
the source's; and the `look` figure. The stills were read on five frames a
shot, source over long load over per-shot load.

## What it shows

- **The region is the same** in both renders of a shot, to a tenth of a
  percent of the frame. What differs is what was done inside it.
- **The per-shot load changed less of it**, on both shots, and its
  difference from the source in the box beside the subject is well under
  half the long load's. As seen in stills, what that box holds is the
  source's own in the per-shot load and is redrawn in the long load.
- **Movement inside the region, frame to frame.** On the first shot the
  per-shot load changes about as much as the source does and the long load
  less; on the second both are near the source's. As seen in stills the
  per-shot load's pose is the source's on all five frames of the first
  shot, where the long load holds one pose.
- **The pose measure separates them on one shot and not on the other**
  (the json's `motion` block). On the first shot the source moves one
  joint and holds the rest: the per-shot load's moves in step with it at
  about half the size, and the long load's does not move with it at all.
  On the second shot, where the source moves three joints, both renders
  follow about equally. So "the long load holds one
  pose" is true of the first shot and not of the second, and the difference
  that holds on both shots is the one above: how much of the region was
  redrawn, and whether what is beside the subject stays the source's.
- **Both show the still's person.** The `look` reads higher for the
  per-shot loads, partly because their subject sits where the original
  sat, which that figure also rewards.

## What it does not show

The first shot's pose pass ran on the CPU and the second's on the card;
the json says which. The measure's `follows` line was set on another
clip, so its verdict words are in the json and are not leaned on here.
Whether
a per-shot load is better on a long shot, where it needs more than one
window. Whether the difference is the load's length, the single window, or
the absence of other shots in it. Anything about how either plays.
Fingers and held things are not joints: what a subject does with objects
is in the stills and the box figure, not in the pose measure.
