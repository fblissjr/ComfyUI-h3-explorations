bump: minor

### Added

- A preflight rule in `bench/capture_masked_run.py`, `region_carried_across_a_cut`: the frames beside a cut that a latent step carries a subject's region onto, per window, from the masks, the shot table's cuts and the window plan (read from a render's graph, or `window=` and `context=` on a plan). It is the rule of the node's `video_mask.cut_gate`, worked out with no sampling; the check compares the two when the tree has the gate. On the render where the assembler had found the other subject repainted at four cuts, it names those five frames and no others. A `--mask` takes `shots_at=` for a shot table that counts from another first frame than its mask video.
