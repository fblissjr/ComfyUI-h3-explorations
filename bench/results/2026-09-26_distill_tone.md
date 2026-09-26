# What the owner calls "washed out / contrasty", measured, 2026-09-26

**A description, not a ranking.** One seed, two scenes. The instrument is
`bench/measure_clip_tone.py`: report-only, its own scale, and its docstring
defines every column. The numbers are in `2026-09-26_distill_tone.json`. The
clips are the three-way comparison's
(`2026-09-26_distill_compare_s1.md`), plus the undistilled base model
(`h3_text_to_video`, 16 steps, rows `diner_base` and `subway_base` in that
record's JSONL) at the same seed and prompt text, as the model's own look.

## The owner's words, and the columns that follow them

The owner, on diner: FlashGen "more contrasty" than PDD8, and FastH3 "more
washed out / contrasty than flashgen was, even".

| diner | base | PDD8 | FlashGen | FastH3 |
|---|---|---|---|---|
| white (Y p99) | 0.605 | 0.88 | 0.938 | 0.956 |
| mid (Y p50) | 0.194 | 0.272 | 0.278 | 0.298 |
| rms_contrast | 0.151 | 0.177 | 0.196 | 0.211 |

| subway | base | PDD8 | FlashGen | FastH3 |
|---|---|---|---|---|
| white | 0.801 | 0.921 | 0.929 | 0.984 |
| rms_contrast | 0.199 | 0.241 | 0.257 | 0.261 |
| clipped | 0.0024 | 0.002 | 0.0011 | 0.0137 |

- **Every distill brightens and stretches the tone curve** against the base
  model. The order is PDD8, then FlashGen, then FastH3, on both scenes. That is
  the owner's ordering.
- **What the owner's words map onto:** the white point and the RMS contrast
  track "contrasty / washed out" on both scenes. Chroma does not: FastH3 is the
  most saturated on diner, so "washed out" here is not desaturation. It reads
  as bright, near-clipped highlights on a steeper curve. On subway, FastH3
  clips about six times the base's share of highlights.
- **The base is low-key.** On diner its white point sits at 0.61, a dim night
  interior. The distills lift it toward full scale.
- **Template and contract FastH3 measure alike** (rms 0.211 on both). The
  contract's changes did not move the tone.

## Caveats

- The base clips decode through the INT8 video VAE, shipped by 0.151.0 before
  they rendered. The distill clips decode through fp16. The owner could not
  tell the two decoders apart on a matched pair
  (`2026-09-26_vae_decoders_345f`, the VAE session's record). This run has not
  measured whether the decoder moves these columns.
- **Why distills do this is not tested here.** Inference: every distill
  trained to match the teacher in few steps shows it, the two
  distribution-matching distills (FlashGen's VSD, FastH3's DMD2) most strongly.

## Next

- Whether the look can be pulled back toward the base's without losing the
  distill's coherence. Candidates, each a probe:
  - FlashGen at a strength below 1;
  - the first step on the base model and the rest on the distill;
  - a tone match in post against the base's curve.
