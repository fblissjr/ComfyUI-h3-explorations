bump: patch

### Changed

- `docs/h3_references.md`: a dated note beside "Set `force_rate=24` on the loader": that setting puts a reference's time labels right by dropping or repeating frames, which on a 25 fps source is one dropped frame a second (measured 2026-10-08 as camera steps in every render from one such clip); a source the result must stay in step with is loaded as an every-frame 24 fps copy with `force_rate` 0 and the delivery re-stamped. `workflows/h3_config.py`: the comment on `MASKED_SOURCE` no longer says the margin taken from the subject's size has not rendered (it has, and has not been judged as a pair). `docs/wiki/decisions.md` logs what both used to say. Comments and prose only: no value, default or graph changes.
