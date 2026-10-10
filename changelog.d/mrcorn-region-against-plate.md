bump: minor

### Added

- **`bench/region_against_plate.py`: how a regenerated region sits in its
  plate, in numbers.** For a render and the clip it was made from: detail
  in three bands of the luma with the grain taken out, the texture of flat
  areas beside the size of edges, grain as the frame-to-frame change where
  the picture is still, and tone and cast, each for the region, for what
  stood there before, and for the plate just outside it; the region is read
  from where the render differs from the fitted original, so no mask is
  needed. It also reads the region under a few trial blurs, which is how
  `assemble_delivery.py --soften` gets its sigma. It was a session script
  until now and a finding rests on it: run on two fixed-camera renders of
  one clip on 2026-10-10 it refuted the guess that the region was cleaner
  than its plate (it was crisper at the edges and busier, with tone and
  cast already matching), and showed that at the source's own size the
  scale-up alone brings the region to the plate's softness.
- **`bench/check_region_against_plate.py`**: noise of a known size comes
  back at that size and not as detail; a blur of a known sigma is read as
  softer and lands on the tool's own trial row; a lift and a tint are read
  where they were put; the region counted is the region changed. Three
  deliberate breaks of the tool each turned it red.
