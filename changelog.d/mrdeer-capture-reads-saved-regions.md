bump: minor

### Changed

- `bench/capture_masked_run.py files` takes a run's region from the files
  its windows saved beside their latents (`<name>_window_N_region.npz`,
  `video_mask.save_window_region`) when the render has them: each frame from
  the window whose video holds it, the token region over each latent step's
  frames, and none on a frame `video_mask.cut_gate` left as the source
  across a cut, which the manifest names. Until now the region was read back
  by un-tinting the render's review video against the source, which cannot
  be done for a render whose source was another render and is a reading
  where this is the record. A render made before the files were written, or
  `region=review` on `--run`, is read from the review as before; files that
  do not tile the render from its first frame to its last are refused and
  the manifest says why. `bench/check_capture_masked_run.py` pins it on two
  windows written by `save_window_region` itself.
