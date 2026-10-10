bump: minor

### Added

- **`bench/assemble_delivery.py` reads a capture's `owners.npz` for whose a
  pixel is.** Two tracked masks can claim one pixel, and then neither mask
  says whose it is. Where a capture given holds an owner map
  (`bench/capture_masked_run.py::owner_map`) that covers the frame and lists
  the subject, "the subject's pixels" means the pixels it owns: for a pixel
  two pieces both changed, and for `restore=<subject>`, which then gives
  back what that subject owns, the pixels the map settled in its favour
  included, and nothing the map marks contested. With no owner map the
  tracked masks are used as before, and the check record's `whose_pixels`
  says which it was. Measured 2026-10-10 on one whole-person render over the
  frames where the two tracks claim the most (an arm across the other
  person): of the pixels both tracks claim, the share left more than twelve
  levels from the source went from about two thirds under the tracked masks
  to under a fiftieth under the owner map.
  `bench/check_assemble_delivery.py` has a twelfth case.
