# Saved windows of masked renders, decoded and laid again (2026-10-10)

lane: masked video to video
verdict: not judged. Figures read from decoded latents and tables; nobody watched a clip for this. One clip, one subject, one seed a render.

**Read this first.** Every region here was rebuilt from the mask read back
off each render's own review (a tinted, compressed picture), because these
renders were made before a window saved its region. The json gives, for
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
rendered and with a setting changed. The numbers are in
[`2026-10-10_saved_windows_laid_again.json`](2026-10-10_saved_windows_laid_again.json),
built from the tool's tables; the tables and the kept decodes are under the
untracked `data/2026-10-10_recomposite_decodes/`.

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
