bump: minor

### Added

- **The Masked Source can take its margin from the subject's size**
  (`grow_by`, a new last input; `video_mask.margins`). `a fixed margin`, the
  default, is the node as it was: `grow_pixels` on every frame, and every
  mask it grows is the same mask bit for bit. `the subject's size` takes the
  margin per frame from the mask's own area (`GROW_SHARE` of its square
  root), with `grow_pixels` as the cap and `feather_pixels` as the floor,
  so the node's rule that the blend stays off the subject holds on every
  frame. The area is steadied by a running median (`GROW_STEADY`) before
  the margin is taken, so a part mask that collapses for a stretch does not
  pull the margin down, and a cut's step stays on its frame. Why: a margin
  fixed in pixels is a thin border round a close subject and several times
  a small one's own area, and the region then holds the people round them
  (`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`,
  section 5; the masking board's `top-the-hole-is-too-big`). The share is
  reasoned and has not rendered; no shipped graph sets the choice.
- **The choice moves the region and what follows from it, not the motion
  reference.** Every place that widens a source's mask for the region reads
  one helper (`video_mask.source_margins`): the region's tokens, the
  paint-out and late-start holes, the preview strip, the `keep` overlap line
  and the composite's old-subject margin. Per window the Song node's report
  gives the margin the region used, as its range over the window. The motion
  reference's widening stays half of `grow_pixels` on every frame
  (`video_mask.motion_widening`): taken from the subject's size it greyed
  part of the subject on some frames of one clip's masks, measured outside
  the node (the masking board, finding `mhi-04`).
- `bench/check_video_mask.py` item 13 holds it, with two controls that must
  be red: the fixed margin on a small subject is over the bound the scaled
  one must stay under, and the same rule with no steadying is moved by a
  collapsed stretch.

### Changed

- `window_keep.py`'s source-encode key carries the choice beside
  `grow_pixels`, and `feather_pixels` with it when the margin follows the
  subject, where the feather is the margin's floor: an encode kept under one
  is not handed to a run on the other. The conditioning's key does not need
  either, since the motion reference does not follow the choice.
- `video_mask.start_zero_tokens` takes the window's first frame and has no
  default for it, so the late start cannot read another window's margins.
- `bench/node_id_manifest.json` records the Masked Source's appended input.
- `h3_config.MASKED_SOURCE` holds `grow_by` at the node's default, so a
  rebuilt masked graph carries the input.
- `bench/check_mask_store.py` reads the record's new per-frame area as a
  tensor, and holds it equal between a kept and a tracked run.
