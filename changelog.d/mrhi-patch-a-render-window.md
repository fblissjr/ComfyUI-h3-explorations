bump: minor

### Added

- **`bench/patch_render_window.py`: redo one stretch of a finished masked
  render, and join it back.** The masked lane is run on the finished render
  itself as its source, for one window, with the hole open only on the
  frames to redo; every other frame of the window is the render's own
  picture, kept whole, so the redraw is held at both ends by what is already
  there and nothing is continued across a seam. First used on 2026-10-08
  (the masking board, finding `mhi-09`): one window, one seed.
  - `build` cuts the two mask clips for the window from the mask videos of
    the render's own load (nothing is tracked again), closed outside the
    hole, with stray specks dropped by `subject_tracks.drop_specks` and the
    part cut to the cleaned subject by `video_mask.select_part`; cuts the
    motion video from the source's frames with `video_mask.motion_reference`
    and the render graph's own settings; writes the render's graph changed
    in the fewest places that make it this window, and prints those places.
    It says which frames the hole really covers after the model's frame
    grouping, and refuses a length the planner would not render as one
    window.
  - `join` takes the hole's frames from the patch and everything else from
    the render, by frame index, and prints the per-frame numbers either side
    of each end of the hole: the patch against the render in the region and
    on the plate, the frame-to-frame change in the region, the brightness
    inside the mask.
  - Three rules are written beside the code, each from the first use: open
    the hole where render and source agree in pose; put its ends on the
    edges of the model's frame groups; append the motion video as a
    reference, since the Masked Source cuts its own only from its source.

### Not done

- It queues nothing, and no shipped workflow carries the technique. Each
  mode was run once, on the files of the first use: `build` gave the same
  clips, frame for frame, and the same nodes and wiring as the hand-built
  graph that rendered; `join` gave a file of the render's frame count whose
  frames sit on the render's index outside the hole and on the patch's
  inside it. No check holds either mode.
