bump: minor

### Added

- **`bench/assemble_delivery.py --size source`: the delivery at the source's
  own size.** Every frame is the source's own picture as it decodes, never
  scaled, and only what a render changed is scaled up from the canvas and
  laid over it at the place the loader's crop took it from (`crop_of`,
  `laid_over`); the rows or columns that crop dropped stay the source's.
  Every row of the table is laid by region in this mode, the first as well,
  and a render whose change reaches the canvas's edge where the crop cut is
  flagged (`piece_changes_up_to_the_loader's_crop`), since beyond it only
  the source's picture exists. `--soften` is refused here: measured
  2026-10-10 on two fixed-camera renders of one clip, the scale-up alone
  brought the regenerated region to about the sharpness and frame-to-frame
  change of the source's own pixels beside it, where at the canvas's size
  it read crisper. Not judged on playback. The default stays `canvas`.
- **`bench/check_assemble_delivery.py`, an eighth case** for it: the file is
  the source's size and passes its check, pixels away from a painted
  rectangle are the source's own, the rectangle is where the crop and the
  scale put it, a render that kept every pixel is nearer the source where it
  is laid than three pixels to any side, and the edge flag is raised by a
  render painted to the canvas's top and not at the canvas's size.
  `owners` now settles shared pixels for both sizes.
