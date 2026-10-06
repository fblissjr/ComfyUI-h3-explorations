bump: minor

### Changed

- **The frozen-row cache's masked-window use is retired; the node stays for
  the audio-refine pass.** On the lane's most favourable masked window the
  cache engaged cleanly and saved nothing (a plain cached step 36.75 s
  against a Sol-Attn stock step's 33.6 s; the window 463.8 s against
  462.9 s, `bench/results/2026-10-06_frozen_cache_masked_window.md`), while
  on the refine pass the sampler halves
  (`bench/results/2026-09-25_frozen_cache_s1.md`). The owner, 2026-10-06:
  deprecate what has negative value, cite why, keep the code where it is
  plainly unused; and keep the audio-refine use. `frozen_video_cache.py::_gate`
  now sends any call that regenerates a video row stock with the reason;
  `halo` leaves the node, `h3_config.FROZEN_VIDEO_CACHE` and the four refine
  graphs; `verify` loses the video's ring-and-interior split and the
  text-cached second pass, so it costs what the stage-split record measured
  again; the generator's `masked_cache` argument and the probe graph
  `h3_probe_v2v_masked_song_ref2va_motion_cache` are gone. Kept because
  each guards the refine pass (mrhand's read at the cut): the live-share
  limit under refine provenance, the per-row audio mask reading, the
  context-window, misfit-mask and kept-content guards, and the stage
  timers. `bench/check_frozen_video_cache.py` items 9 to 16 (the masked
  use) became one item holding the retirement and those guards on the
  refine layout. `archive/frozen_cache_masked/` holds the module, the check
  and the graph as they stood at 25103f0e, with a README citing both
  records and what would be worth reopening (mrhand's end-of-day note:
  the attention call off the square costs about 2.8 times the dense
  square per query-key pair, which is the whole miss).

The node manifest loses the one name. A schema change: the server was
restarted by the owner before the rebuild. Not yet run on the card after
the cut; the refine pass through the cut module is mrhand's test tonight.
