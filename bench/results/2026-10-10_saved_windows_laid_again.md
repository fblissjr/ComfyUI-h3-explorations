# Saved windows of masked renders, decoded and laid again (2026-10-10)

lane: masked video to video
verdict: not judged. Figures read from decoded latents and tables; nobody watched a clip for this. One clip, one subject, one seed a render.

**Read this first.** Every region here was rebuilt from the mask read back
off each render's own review (a tinted, compressed picture), because these
renders were made before a window saved its region. The tables give, for
each window, how many cells the rebuilt token region differs from the
region the review shows. The figures taken from the decode alone (how far
the decode is from the source in a box) do not use the mask at all.

**What was asked.** Three things, each answerable from files a render
already left, with one decode a window and no sampling:

1. Does the cut gate (`video_mask.cut_gate`, 0.254.0) return exactly the
   frames one render repainted across its cuts, and touch no other?
2. Things beside the subject came out as the source's own in a load made
   for one shot and were redrawn in a long load of the same shot
   ([`2026-10-10_one_load_per_shot_against_one_long_load.md`](2026-10-10_one_load_per_shot_against_one_long_load.md)).
   Is that the sampler's doing or the composite's?
3. Are the frames on the subject's side of a latent step that a cut splits
   any different from their neighbours?

## How

`bench/recomposite_window.py` (its docstring is the method): a window's
stored latent decoded once through the shipped video VAE and kept, then
laid by `video_mask.lay_window`, the function the song node calls, as
rendered and with a setting changed. The numbers are in "Tables" below,
each under the name its finding points at; the file they were made from
and the tool's own tables are under the untracked
`data/2026-10-10_recomposite_decodes/`. The kept decodes were removed at
the day's end: each is made again from its stored latent with
`--decode-only`. (Until the day's rule about this folder, the figures were
a tracked `.json` beside this file.)

## What it shows

- **The gate returns the repainted frames and nothing else.** Laid with no
  gate (as rendered) and with it, the two windows of the faulty render
  differ on the frames its delivery's check had found, each of them the
  source bit for bit under the gate, and on no other frame
  (`gate_on_mom_alone_mesh`). How far the first laying is from each window's
  own video is beside it (`as_rendered_off_its_own_video_levels`): the size
  of the difference on the untouched plate of those files.
- **The things beside the subject: the sampler.** In the box, the decode is
  already far from the source in the long load and close to it in the
  per-shot load, on both shots, before any composite; the composite moves
  both figures by little (`box_before_and_after_the_composite`). So the
  per-shot window regenerated those cells close to the source, and the long
  window did not. Why the sampler did is not answered here.
- **A split step leaves something at the composite even with the gate.**
  On the subject's own frames of each of the four split steps, `only what
  changed` keeps more of the render than on the neighbouring step of the
  same shot (`split_steps_of_mom_alone_mesh`, `kept_more_by`). The cause is
  in the code: `changed_alpha` holds its maximum over a step's frames, and
  the frames across the cut differ from the source over the whole region,
  so the whole region is kept on the subject's side too. The gate takes the
  far frames out of the blend, not out of that maximum.
- **Fine detail on those frames: no clear difference.** The decode's detail
  inside what is kept, over the source's there, is level with the
  neighbouring step at the two cuts where the subject's shot ends, and
  somewhat lower at the two where it begins. Whether that is the split step
  or a shot's first frames cannot be told from one render.

- **A short load that faded back to the original: the sampler drew it.**
  Added the same afternoon. One load capped at its shot's end, laid from
  the region file the render itself saved (the first use of one), drew the
  still's person on its first frames and the original on its last. Under
  the subject's mask the composite kept all of the render on every frame of
  the shot, so it gave nothing back; and the decode's own distance from the
  source under that mask falls frame by frame toward the held frames, to
  near the plate's floor on the shot's last frame
  (`short_load_fade_mom_shot_1010`). So the fade is in the latent. Of the
  held frames the track made the render write, the first carries the render
  (it shares a latent step with the shot's last frames) and the rest are
  the source whole (`held_frames_written`).

- **A frame with no mask of its own, lent a region by its latent step: the
  rule on its own case.** Two face passes that saved their own regions,
  laid as rendered and with every frame that has no mask left as the source
  (`video_mask.unlaid_frames`). The frames that differ are the frames a
  peer had found with an empty mask and something laid on them, and one
  held frame past a load's end that is not in its video; each is the source
  bit for bit under the rule, and no other frame of either window moves
  (`no_mask_rule_on_two_face_passes`, with what the render had laid in the
  lent cells of each).

- **The first render made zoomed in, read from its own files.** Added the
  same evening with `bench/recomposite_window.py --read-files`, which
  decodes nothing; one load, one seed. The lay back leaves the
  plate alone: outside the box, and inside the box clear of the region,
  the piece is as far from the fitted source as the accepted whole-frame
  render of the same load is over the same areas, figure for figure, so
  what is there is the file's codec and nothing of the zoom. Inside the
  region the zoomed picture written beside the render, scaled into the
  box, agrees with the piece to a few levels, which is the agreement the
  delivery tool will hold a zoomed piece to. The part takes a larger
  share of the canvas than it took on the whole frame; both shares are in
  the record (`first_zoomed_render_read_from_its_files`). Nothing here
  says how the face looks.

## Tables

Levels are 8-bit levels. Frame numbers are frames of the source's 24 fps copy. Each table's title is the
name its finding above points at. The file of every figure these were made from is
`data/2026-10-10_recomposite_decodes/saved_windows_laid_again.json` (untracked).

### `gate_on_mom_alone_mesh`

| window | frames written | frames that differ with the gate | of those, the source bit for bit | not the source | laying off its own video, median / most | rebuilt region: cells differing of cells |
|---|---|---|---|---|---|---|
| window 1 | 345 | 294 | 294 | none | 2.091 / 3.946 | 961 of 248505 |
| window 2 | 255 | 294, 414, 438, 482, 483 | 294, 414, 438, 482, 483 | none | 2.751 / 4.442 | 703 of 108051 |

### `split_steps_of_mom_alone_mesh`

| cut at | subject's frames of the split step | neighbouring step | kept px, split step | kept px, neighbour | kept more by | detail of render over source: split step / neighbour |
|---|---|---|---|---|---|---|
| 294 | 291-293 | 287-290 | 330558 | 304425 | 8.6% | 1.145 / 1.133 |
| 415 | 415-417 | 418-421 | 339826 | 313720 | 8.3% | 1.144 / 1.224 |
| 438 | 435-437 | 431-434 | 308256 | 288886 | 6.7% | 1.216 / 1.227 |
| 484 | 484-485 | 486-489 | 337974 | 310687 | 8.8% | 1.089 / 1.157 |

### `box_before_and_after_the_composite`

| shot | load | frames | decode off source in the box, median (min to max) | laid off source, median | share of the box kept | rebuilt region: cells differing of cells |
|---|---|---|---|---|---|---|
| copy 278-293 | long load | 16 | 26.2 (24.342 to 28.586) | 25.8 | 0.92 | 961 of 248505 |
| copy 278-293 | per shot load | 16 | 9.6 (8.092 to 13.987) | 8.9 | 0.87 | 68 of 23624 |
| copy 415-437 | long load | 23 | 24.0 (21.52 to 25.841) | 23.4 | 0.83 | 703 of 108051 |
| copy 415-437 | per shot load | 23 | 8.6 (6.302 to 9.937) | 7.5 | 0.69 | 124 of 32188 |

### `short_load_fade_mom_shot_1010`

Laid from the region file the render saved; 41 frames written. Under the subject's mask, every second frame:

| source frame | mask px | decode off source | laid off source | share of the mask kept |
|---|---|---|---|---|
| 1010 | 9001 | 49.86 | 49.548 | 1.0 |
| 1012 | 12093 | 66.71 | 66.703 | 1.0 |
| 1014 | 17425 | 74.658 | 74.465 | 1.0 |
| 1016 | 15592 | 69.735 | 69.729 | 1.0 |
| 1018 | 12469 | 58.251 | 58.197 | 1.0 |
| 1020 | 13511 | 50.304 | 50.082 | 0.9993 |
| 1022 | 11722 | 46.839 | 46.731 | 1.0 |
| 1024 | 11984 | 40.708 | 40.692 | 1.0 |
| 1026 | 13521 | 44.18 | 44.099 | 1.0 |
| 1028 | 12818 | 44.942 | 44.856 | 1.0 |
| 1030 | 12112 | 39.72 | 39.644 | 1.0 |
| 1032 | 10858 | 30.589 | 30.548 | 1.0 |
| 1034 | 9529 | 25.042 | 25.037 | 1.0 |
| 1036 | 8072 | 18.117 | 18.064 | 1.0 |
| 1038 | 8478 | 4.902 | 4.892 | 1.0 |

`held_frames_written`: [{"source_frame": 1039, "kept_px": 29689}, {"source_frame": 1040, "kept_px": 0}, {"source_frame": 1041, "kept_px": 0}, {"source_frame": 1042, "kept_px": 0}, {"source_frame": 1043, "kept_px": 0}, {"source_frame": 1044, "kept_px": 0}, {"source_frame": 1045, "kept_px": 0}, {"source_frame": 1046, "kept_px": 0}, {"source_frame": 1047, "kept_px": 0}, {"source_frame": 1048, "kept_px": 0}, {"source_frame": 1049, "kept_px": 0}, {"source_frame": 1050, "kept_px": 0}]

### `no_mask_rule_on_two_face_passes`

| render | frames written | frames that differ under the rule | of those written | now the source | not the source | lent cells / px kept / laid off source in them, as rendered |
|---|---|---|---|---|---|---|
| face_street_0294_c | 121 | 358, 409 | 358, 409 | 358, 409 | none | 358: 32 / 7789 / 10.131; 409: 50 / 12410 / 25.187 |
| face_street_0438 | 46 | 450, 456, 457, 458, 484 | 450, 456, 457, 458 | 450, 456, 457, 458, 484 | none | 450: 57 / 13506 / 6.856; 456: 33 / 7720 / 7.42; 457: 33 / 7720 / 13.369; 458: 33 / 7720 / 18.951; 484: 58 / 14458 / 5.076 |

### `first_zoomed_render_read_from_its_files`

Box [208, 62, 848, 542] of 1024x768 frames, 1.6 times, set by the box; 46 frames; frames left as the source: 452, 453, 454. Share of the canvas regenerating: 0.0559 zoomed, 0.0173 on the whole frame.

| area | the zoomed render: mean / 99.9th / worst | the whole-frame render, same area: mean / 99.9th / worst |
|---|---|---|
| outside the box | 3.318 / 31.0 / 98.0 | 3.326 / 31.0 / 98.0 |
| inside the box clear of the region | 3.667 / 36.0 / 101.0 | 3.666 / 36.0 / 102.0 |
| inside the region | 21.245 / 193.0 / 248.0 | 22.256 / 187.0 / 253.0 |

The zoomed picture laid into the box against the piece, inside the region, clear of its edge by 8 canvas pixels: median 2.321, 99th 7.9, over 562876 pixels. the baseline's `inside_the_region` is the whole-frame render read over the ZOOMED run's region, not over its own: the two renders came from two captures of the shot. A reading made by hand the same evening, before the tool had this mode, gave the same plate figures to a few hundredths of a level; this block replaces it.

`decode_seconds_on_an_empty_card`: {"345_frames": [32.1, 28.5, 28.3], "192_frames": [17.1], "141_frames": [10.7, 10.7], "note": "first of the 345s includes the VAE's load"}

## What it does not show

Whether any of these frames looks worse: the detail figure is a mean over
the kept area and says nothing of a face or a hand. Why a long window
redraws what a short one leaves. Anything about a second clip, subject or
seed. And the laid figures carry the read-back mask's error; the tool's
kept share reads a few points above the node's own report on the per-shot
windows.

## What follows

Renders made from 0.262.0 save each window's region, so the next use needs
no read-back. The leftover in `changed_alpha` is logged in
`data/CAPTURE_GAPS.md`; it goes away with one load per shot, and its fix
otherwise is the same change as a region per side of a cut.
