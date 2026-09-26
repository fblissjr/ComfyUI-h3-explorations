# PDD8, FlashGen and FastH3 as each is best built today, 2026-09-26

**A first look, not a verdict.** One seed, two scenes, unjudged by the owner.
The manifest is `bench/distill_compare_arms.json`, and the rows are
`2026-09-26_distill_compare_s1.jsonl`.

**The arms:**
- PDD8: `h3_text_to_video_pdd`.
- FlashGen: `h3_text_to_video_flashgen`, rank 64 applied at the call.
- FastH3: `h3_probe_t2v_fasth3_8step_contract`, FastVideo's contract.

**Setup:** seed 730451892, 1344x768, 345 frames. The diner FlashGen and FastH3
clips were already rendered at this seed and code, in the rows `fast_ship` and
`contract` of the two earlier records.

## Checks that need no judgement

- **Every arm rendered.** No row carries `error` or `suspect_cache_hit`.
- **Sampler time**, `sampler_s` in the rows:
  - PDD8: 245 s (diner), 241 s (subway);
  - FastH3: 200 s (subway), 199 s (diner, `2026-09-26_fasth3_contract_s1.jsonl`);
  - FlashGen: 130 s (subway), 132 s (diner, `fast_ship`).
- **Audio level** (`volumedetect`, mean / max dB):

  | scene | PDD8 | FlashGen | FastH3 |
  |---|---|---|---|
  | diner | -31.1 / -10.6 | -32.3 / -11.3 | -29.0 / -9.8 |
  | subway | -21.7 / -6.8 | -18.9 / -2.7 | -14.0 / -1.4 |

  FastH3's subway max sits 1.4 dB under full scale. Loudness is not quality;
  a listen should be loudness-matched.

## By eye, not a ranking

Four frames per arm, at frames 24, 120, 216 and 312.

- **Diner:**
  - all three arrive at the prompt's view in through the window by the last
    frame;
  - PDD8 is the softest and warmest;
  - FlashGen is photographic, with rain on the glass;
  - FastH3 carries the most background detail and the most legible neon
    (mirrored from inside).
- **Subway:**
  - PDD8's frame 216 is an empty corridor, with the runner out of frame;
  - FlashGen keeps the runner in every sampled frame, including on the
    escalator;
  - FastH3 keeps him in every frame too, with the heaviest motion blur, and
    ends on the fall.
- **FastH3 at its contract does not look washed out on either scene.** That
  was the owner's reading of the 0.141.0 template clip.

## Clips

On the output share under `Video/`, each with a `-audio.mp4` companion:
- **diner:**
  - `text_to_video_pdd_diner_pdd8_00003`
  - `text_to_video_flashgen_fast_ship_00001`
  - `h3_probe_t2v_fasth3_8step_contract_contract_00001`
- **subway:**
  - `text_to_video_pdd_subway_pdd8_00004`
  - `text_to_video_flashgen_subway_flashgen_00001`
  - `h3_probe_t2v_fasth3_8step_contract_subway_fasth3_00001`

## Next

- **The owner's look**, three-way per scene.
- **Then a blind session** (`h3-ab-session`) at a second seed, with the dense
  baseline, if one of the three is to be the default.
