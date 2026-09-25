# Distill and audio-recovery session, throwaway first run, 2026-09-25

**A wiring run, not a comparison.** One seed and two scenes. Nothing here is
quoted as a quality result. The manifest is `bench/distill_audio_arms.json`,
and the rows are `2026-09-25_distill_audio_s1.jsonl`, graph sha and substrate
per row.

**Arms:**
- `pdd8`: the shipped PDD8 t2v graph.
- `pdd8_refine`: `pdd8` plus an audio-only refine pass (`audio_refine.py`).
- `fasth3`: FastH3 8-step V2 on core's VSA.
- `flashgen`: the FlashGen 4-step LoRA.
- `flashgen_refine`: `flashgen` plus the refine pass.

**Scenes:** `t2va_diner_breakup` (dialogue over music) and
`t2va_subway_chase` (motion). Seed 730451892, 1344x768, 345 frames, all on
the kitchen chain at 0.141.0.

## What happened to the run

- **A power outage** cut it after four rows: the diner warmup, `pdd8`,
  `pdd8_refine` and `fasth3`. After the reboot the server was restarted by
  the `docs/comfy_notes.md` recipe, and the remaining arms were run with a
  fresh warmup and appended to the same file. The diner arms therefore span
  two server sessions.
- **Every arm rendered.** No row is flagged `error` or `suspect_cache_hit`.
- **Each refine arm reused its base arm's first pass from ComfyUI's cache**
  (same nodes, same inputs, run back to back). `per_node_s` shows only the
  refine sampler, node 87, so the row's `sampler_s` is the refine pass alone.
  The output is unaffected: the first pass is identical by construction.
- **The warmups ran at seed 730451891** (the runner offsets the warmup seed so
  the real row is not a cache hit). They are not a same-seed repeat of
  anything.

## Checks that need no judgement

**Is the refine arm's video the base arm's video?** Not bit for bit. The
decoded frames differ on all four pairs. On diner PDD8 against PDD8 plus
refine, the PSNR averages about 46 dB (ffmpeg `psnr`: minimum 43, maximum 51).
That is visually indistinguishable, so the pairs still judge audio. This run
cannot say whether the difference comes from the VAE decode not being
bit-reproducible across runs, or from the arithmetic of the masked blend. A
same-seed repeat of one graph after a cache clear would decide it.

**Audio level**, `ffmpeg volumedetect` on each clip's muxed audio, in dB. The
table is descriptive: one seed, and loudness is not quality.

| scene | pdd8 | pdd8_refine | fasth3 | flashgen | flashgen_refine |
|---|---|---|---|---|---|
| diner, mean | -31.1 | -27.2 | -27.6 | -32.9 | -28.6 |
| diner, max | -11.2 | -10.5 | -8.1 | -12.4 | -9.3 |
| subway, mean | -21.8 | -20.1 | -14.0 | -18.0 | -14.5 |
| subway, max | -6.7 | -6.4 | -2.6 | -0.7 | -0.9 |

The refine pass raises mean loudness on every pair. That is the direction the
energy-loss finding predicts (`docs/research/pdd/audio_under_pdd.md`, "What
actually holds"). Whether it sounds better is the owner's ear to decide.

**Sampler time** per row is in the JSONL. On this card: the refine pass costs
about as much as a four-step distill pass, and FastH3's eight VSA steps cost
about what PDD8's eight do. Cache state: warm after each warmup; the refine
rows are refine-only, as noted above.

## The clips

On the output share under `Video/`, each with a `-audio.mp4` companion:

| arm | diner | subway |
|---|---|---|
| pdd8 | `text_to_video_pdd_diner_pdd8_00002` | `text_to_video_pdd_subway_pdd8_00003` |
| pdd8_refine | `h3_probe_t2v_pdd8_audio_refine_diner_pdd8_refine_00001` | `h3_probe_t2v_pdd8_audio_refine_subway_pdd8_refine_00001` |
| fasth3 | `h3_probe_t2v_fasth3_8step_diner_fasth3_00001` | `h3_probe_t2v_fasth3_8step_subway_fasth3_00001` |
| flashgen | `h3_probe_t2v_flashgen_4step_diner_flashgen_00002` | `h3_probe_t2v_flashgen_4step_subway_flashgen_00002` |
| flashgen_refine | `h3_probe_t2v_flashgen_4step_audio_refine_diner_flashgen_refine_00001` | `h3_probe_t2v_flashgen_4step_audio_refine_subway_flashgen_refine_00001` |

The `_00001` FlashGen clips are the warmups, at a different seed.

## Next

A quoted session follows the `h3-ab-session` process. It adds the dense
baseline and a second seed, and it pairs the arms, not the throwaway clips.
