bump: patch

### Added

- **A second probe graph for the frozen video cache on a masked window**:
  `h3_probe_v2v_masked_upper_song_ref2va_motion_cache_api.json`, the
  upper-body graph with `MiniMaxH3FrozenVideoCache` on the song node's model
  and nothing else changed. A probe; no shipped or daily graph carries the
  cache.

### Changed

- All generated graphs were rebuilt and validated against the server
  restarted on 0.216.0; none changed. The generator's comment above the
  first cached probe pointed at the retired run's record and now points at
  the corrected pair.
