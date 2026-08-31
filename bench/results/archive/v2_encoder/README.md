# v2 text-encoder calibration lane

**Closed 2026-08-27.** The v2 AWQ encoder was rejected at Gate 5 and the lane
was closed rather than adopted. What ships instead is `ENCODER_INT8` -- read
`h3_config.MODELS["clip"]` for the artifact of record, never this directory and
never a doc sentence.

These are the gate measurements, the calibration-pool selection and its
near-duplicate adjudication, the layer-49/50 cross-stack comparisons against
transformers, the kernel- and storage-axis sweeps, the GPTQ host-budget probes,
and the sequential-floor envelope pilots.

The write-ups that consume them are
`docs/research/qwen3-vl-special-tokens-post-training/`, which is itself a
closed tree; its links were repointed here when these files moved.

**What did not move with the lane**, because something live still reads it:
the calibration pool jsonl (several `bench/` scripts open it), the
component map, `2026-08-25_four_encoders_holdout_layer50.json`
(`workflows/h3_config.py` and three bench scripts),
`2026-08-25_released_encoder_is_stock.json` (CLAUDE.md cites it as settled),
and the marker-embedding records. Those stayed in `bench/results/`.
