bump: minor

### Changed

- **`bench/patch_render_window.py join` goes through
  `bench/assemble_delivery.py`: one joiner.** The join is a table of one row
  (the hole's frames from the patch, over the render as the source), so a
  patched render is checked by decode as every delivery is and its colour
  is right whatever form its inputs were written in. As a plain
  concatenation the join left the hole's frames two to three levels off in
  every channel, in a file with no colour tag, whenever the render had been
  written before the song node converted and tagged its files and the patch
  after; with both written the same way it was correct. Its report of the
  hole's two ends is unchanged.

### Added

- **`bench/check_patch_render_window.py`**, the join's first check: every
  frame in its place, only the hole's frames from the patch, the render's
  colour and audio, a control (the window given one frame late), and the
  older render patched by a newer file, which the old join failed.
- **`bench/assemble_delivery.py`: the across-a-cut flag says whether the
  cut falls inside a latent step of the piece's load**
  (`loop_plan.step_span`, counted from the piece's first frame) and whether
  that step holds the spilled frames. If it does, the spill is the known
  way a region is carried over a cut; a spill at a cut on a step's edge is
  something else and the flag says so. On the render that showed the fault
  it names all four cuts as inside a step.
- `docs/wiki/masked_v2v.md` points at `bench/region_against_plate.py`.
