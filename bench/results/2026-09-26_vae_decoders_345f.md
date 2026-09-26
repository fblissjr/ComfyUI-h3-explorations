# Video decoders at 345 frames: fp16, INT8 ConvRot, taeh3, 2026-09-26

**Question.** What do the INT8 ConvRot video VAE and `h3_config.DRAFT_VAE`
(taeh3) cost and save against the shipped fp16 decoder, at the owner's
length? Also, is the draft route's keeper decode lossless?
`docs/open_experiments.md` #31 (observables 1 and 2), and the INT8 lane,
which the owner reopened for measurement the same day ("lets try").

**Latent.** The shipped FlashGen t2v graph (`h3_text_to_video_flashgen`) at
seed 730451892, 1344x768, 345 frames, as saved by
`h3_text_to_video_flashgen_draft`. Latent `[1, 24, 102, 48, 84]`.

## Keeper path: lossless

`bench/results/2026-09-26_draft_decode_s1.jsonl`, four rows: a ship warmup,
ship, draft, and `keep`. `keep` is `h3_decode_saved_latent` on the draft's two
saved halves.

- **The draft reused the ship arm's cached sampler output.** Same graph
  upstream of the decoders and same seed, so ComfyUI's cache served node 10.
  The draft row's `sampler_s` is null and its total is the decode alone. So
  the saved latent is the ship render's latent, which is what makes the
  comparison below clean.
- **The keeper clip is identical to the ship clip** on video (345 frames) and
  audio, by decoded pixels and PCM
  (`2026-09-26_draft_keeper_vs_ship_pixels.json`,
  `2026-09-26_draft_keeper_vs_ship_pixels_audio.json`). The SaveLatent and
  LoadLatent round trip loses nothing.
- In the server, VAEDecode took 29.9 s for ship and 2.5 s for the draft, both
  with the DiT resident.

## Decoders on one latent

`bench/compare_vae_decoders.py`, record `2026-09-26_vae_decoders_345f.json`.
Each arm ran in a fresh process with `--fast fp16_accumulation` (as `start.sh`
launches the server), on an otherwise idle card: the server was up with its
models unloaded. Steady-state time is the second of two decodes. Peak is
`max_memory_allocated`, weights plus activations.

| arm | steady s | peak MiB | PSNR vs fp16 dB (worst frame) | max abs | motion ratio |
|---|---|---|---|---|---|
| fp16 | 29.55 | 5651 | reference | | |
| fp16 repeat | 29.63 | 5651 | bit-identical | 0 | 1.0 |
| INT8 ConvRot | 17.36 | 3295 | 54.91 (53.42) | 29 | 1.0002 |
| taeh3 | 2.04 | 279 | 26.66 (22.69) | 242 | 0.9039 |

Controls: determinism held (the fp16 repeat is bit-identical), and taeh3 sits
far below INT8, so the metric separates decoders.

## Reading

- **INT8 at the owner's length.** It is faster than the 124-frame measurement
  of 2026-08-10 recorded in the CHANGELOG, saves about the same resident
  memory, and the error is small and flat in time: the motion ratio is at 1,
  so there is no flicker. The two PSNR figures are not comparable. The
  earlier one was on encoded clips of paired renders, under a core and
  kitchen that have since changed (kitchen #167's VAE kernels, core #16436's
  tile blending). This one is raw decoder output from one latent.
  **Not measured:** whether any of it is visible. It is a pixel metric, not
  a judgement.
- **taeh3.** Roughly an order of magnitude faster than fp16 and nearly free
  in memory, at a fidelity that is plainly a preview. Its motion ratio below
  1 reads as smoothing. Whether a keep made on it survives is #31
  observable 3, which needs the owner.
- **What did not run:** one latent and one scene. VRAM here is the decoder in
  isolation. In a render the DiT may be resident too, depending on
  ComfyUI's offload at decode time.
