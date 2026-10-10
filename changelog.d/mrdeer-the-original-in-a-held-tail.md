bump: minor

### Added

- **The gate checks, before anything samples, that no planned window shows
  the model a picture of the subject it is replacing as a frame to keep**
  (`bench/capture_masked_run.py preflight`, `flag_held_tail`,
  `original_shown_in_the_held_tail`, top level). A window longer than its
  load is filled with the load's last frame, held; with the tail as the
  plate those frames reach the model clean, and when the subject's mask is
  on that last frame they are the original. From the node's own plan
  (`--run NAME:preview=`), per window: the frames held, how many of them are
  shown clean (`held_tail_facts`: a held frame is clean when its latent step
  has no region), whether the subject is on the last real frame, and the
  latent step that is part real and part held, which goes first. The flag
  names the remedy: the Masked Source's `held_tail` with its region open, or
  a load that fills its window; with the remedy set the plan shows no held
  frame clean and nothing is raised. Found on 2026-10-10 on two per-shot
  loads that ended on the original (a fade over the last twenty frames on a
  subject at a small share of the frame, over the last four on a large
  face); read back through the rule, both raise it and it names the frames
  that went first. A plan worked out from the masks has no windows and
  cannot be checked: `held_tail_not_checked` says so once, iffy. Pinned in
  `bench/check_capture_masked_run.py`.
- **`changed` reads a render's two ends against the source every time it
  runs** (`source_at_the_ends`; `ends` and `under_the_carried_mask_by_frame`
  in `changed.json`): the difference from the source under the mask the run
  carried, the load's level, and whether the first or the last frames fall
  to the source (`starts_on_the_source`, `ends_on_the_source`, with the run's
  frames). It is the outcome that grades the rule above, not a substitute
  for it. On the day's renders it names the two loads known to have ended on
  the original and a third nobody had seen (a 46-frame face load whose last
  four frames fall), and is silent on four that held.
