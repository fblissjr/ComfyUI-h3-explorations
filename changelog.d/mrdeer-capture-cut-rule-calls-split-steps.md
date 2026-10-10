bump: patch

### Changed

- `bench/capture_masked_run.py`'s cut rule calls `loop_plan.split_steps` in place of its own copy of the latent-step arithmetic (`window_plan` and `straddled_frames` are gone; `cut_frames` turns a mask's presence into the ranges that function takes). It needs no window plan any more: the steps' edges are fixed for a whole load. It names the same frames as before on the three captures it had been run on.

### Added

- A preflight flag, `region_shared_across_a_cut`: the frames of a latent step in which a subject is on both sides of a cut, where each side is given the other side's region too and no gate covers it.
