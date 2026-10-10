bump: minor

### Added

- **`bench/assemble_delivery.py`: a row lays nothing on a frame its run
  carried no mask on.** A masked render regenerates by latent step, and a
  step is several frames: where a subject's part is emptied on one frame of
  a step and present on another, the sampler still draws a region on the
  emptied frame, and a delivery carried six such frames (a face under a
  hat's brim for one frame where the source has none). A run's capture holds
  the mask each window carried (`runs/<run>/region.npz`); where the manifest
  says it is the one each window saved, a row whose carried mask is empty on
  a frame leaves that frame as the source's, at both sizes, and the flag
  `piece_changed_where_its_run_carried_no_mask` names the frames and the
  pixels. A carried mask that was only read off a review picture drops
  nothing and the same flag says so. It covers renders that exist; the cause
  is in the node's composite.
- `bench/check_assemble_delivery.py`: case 18.
- `bench/results/2026-10-10_delivery_files_and_their_records.md`: the dated
  lines, and the file built under the rule measured against the one before.
