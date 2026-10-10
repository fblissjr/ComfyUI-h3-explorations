bump: minor

### Added

- **`bench/masked_render_against_source.py` gains two modes and takes the
  measured frame.** `align` says which source frames a render's first,
  middle and last frames are and how the source was fitted to the canvas, by
  frame number and never by time; `timing` says whether the mask on each
  frame is the subject's outline on that frame of the source, from a
  tracker's mask saved as a video or from the render's own mask review, read
  over stretches of frames; and `flicker` takes `--start-frame` and `--fit`,
  so it reads the source the render was made from where it used a start time
  and a stretch. Each mode's control is in its docstring: a clip cut from
  known frames, and masks shifted on purpose.
- **`bench/check_mask_store.py` item 9: no shipped graph reuses another
  run's result, and only a graph named as a probe wires a cache** (the
  owner, 2026-10-09). The reuse inputs and cache nodes are listed in the
  check and held to the pack's source, so a new one cannot ship unlisted.
  Red controls: a made-up graph with a reuse on, a cache node without the
  probe name, an unlisted switch.
- **`docs/comfy_notes.md`, "What is cached, and by whom"**: how ComfyUI's own
  node cache is keyed, where it can serve something stale, and where this
  pack's own reuse is switched and checked.
- **A record of ten more renders of the one masked window, each one change**
  (`bench/results/2026-10-09_masked_one_window_why.md`, with its json): which
  way the lead faces in each against the source, who is drawn, the cap, the
  halo and the margin, and whether the tracker's frames are the model's.

### Fixed

- **The earlier record of that window** says where the later one corrects
  it: the window is two frames before the one intended, and the cut cap is
  not the composite's doing.
