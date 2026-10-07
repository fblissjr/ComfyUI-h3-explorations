bump: minor

### Changed

- **The masked graphs load the source at the canvas in one resize.** The
  generator gave the loader the canvas width alone, so a 16:9 file was
  scaled by the loader and then cropped and scaled again by the fit, both
  bilinear. It now gives the canvas height as well
  (`workflows/build_workflows.py`, `freeze_song_source`), and the loader
  crops to the canvas's shape and scales in the one ffmpeg pass that sets
  the frame rate; `video_mask.fit_frames` then has nothing to do. Measured
  on eight frames of one 16:9 clip, emulating both paths with ffmpeg, mean
  absolute Laplacian: 1.96 on the two-step path, 2.45 in one pass with the
  loader's filter, 2.59 in one pass with Lanczos, which sglang uses and the
  loader does not expose. Twelve generated graphs change (the masked and
  daily mask graphs); a 4:3 file at a 4:3 canvas was never affected.
  `bench/check_widget_deviations.py` declares the height beside the width.
  Every masked render of a 16:9 clip before this had the softer source. The
  tracker and the part model now see frames at the canvas size, so kept
  masks are recomputed; the effect on SAM's masks is measured next and is
  not in this entry.
