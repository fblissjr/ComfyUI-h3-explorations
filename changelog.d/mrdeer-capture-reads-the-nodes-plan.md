bump: minor

### Added

- `bench/capture_masked_run.py files --run NAME:preview=<windows folder or
  plan file>,subject=LABEL[,others=..]`: a pass that has not rendered, read
  from what the song node's preview wrote (`<name>_plan.json` and each
  window's `_planned_region.npz`, 0.274.0) instead of worked out from the
  masks. The region is the one each window will be given, per latent step;
  the window that writes a frame, the held tail of a short load and the
  frames the composite will leave as the source across a cut are the plan's
  own. A window the plan could not plan, a region file that is not beside
  the plan, a gap between windows and two plans in one folder are each
  refused with the reason. `--plan` stays as the what-if without the card
  and its docstring calls what it works out an estimate. Pinned in
  `bench/check_capture_masked_run.py` on a plan and region files written by
  `video_mask.save_window_region`; not yet run on a preview's own files,
  which exist from the next server start.
