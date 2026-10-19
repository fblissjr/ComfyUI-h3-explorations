# One still as an encoder-only reference: time, and the owner's look (2026-10-03)

The owner asked what happens when a reference is encoded "with just the
vision tower and not the vae". The conditioning node already allows it (leave
`vae` unwired: `docs/h3/references.md`, "Encoder-only references"), so this is
one render of the shipped graph that way, beside the shipped render at the
same seed.

## What ran

- **Graph:** `workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json`
  with `vae` and `audio_vae` removed from the conditioning node's inputs and
  nothing else changed (a copy under `internal/`, not a shipped graph).
  Market scene, 1344x768, 345 frames, one still at a 2048 short edge shown to
  the text encoder at that size, Sol at the 0.185.0 defaults.
- **Seed 730451892**, a server started for it and unarmed. Row:
  `2026-10-03_encoder_only_reference.jsonl`.
- **Beside it:** the shipped graph at the same seed, first render after a
  server start as well (`2026-10-03_sol_container_protocol_e2e.jsonl`).

## Time

| | shipped | encoder only |
|---|--:|--:|
| packed sequence, tokens | 119,485 | 112,125 |
| sampler, s | 338.8 | 272.6 |
| total, s | 389.1 | 320.7 |

Both are first renders after a start, so they carry the same load costs; a
second render of either would be faster by about the same amount
(`2026-10-03_r2v_finish_time_budget.md`). The difference is the reference's
latent rows leaving every block of every evaluation.

## The owner's look

"look the same to me, cant tell the difference and couldnt tell u which one
was encoder only if i didnt know. seems to hold the ref image well." (the
owner, 2026-10-03, on `..._savelat_encoder_only_00001-audio.mp4` against
`..._savelat_ship_00004-audio.mp4`.)

## Loudness and cuts, measured later the same day

`bench/score_clip_shots_loudness.py` on the clips already on disk, nothing
rendered. EBU R128 integrated loudness against the dense render of the market
scene at the shared view (`dbp_market__dense`), and scene cuts against the
three shots the prompt scripts:

| clip, market scene, seed 730451892 | LUFS | against dense | cuts |
|---|--:|--:|--:|
| dense attention, shared view (the reference) | -21.0 | | 2 |
| shipped (sparse attention, shared view) | -21.3 | -0.3 LU | 4 |
| encoder-only still (sparse, shared view to the encoder) | -22.4 | -1.4 LU | 4 |
| 512 encoder view (sparse) | -21.3 | -0.3 LU | 4 |

- On this scene dropping the still's VAE rows makes the mix about 1 LU
  quieter than the shipped clip, and the 512 view does not move it. The
  louder mix the 512 view shows on the backstage dialogue scene
  (`2026-10-03_backstage_qview512_clip.json`) is not here.
- Every sparse clip holds two cuts the script does not have, the encoder-only
  one included; that is the sparse-attention finding of
  `2026-10-03_sol_output_check.md`, not a reference effect.
- One scene, one seed, one clip per row.

## What this does not establish

- **Not blind.** The filenames say which is which, and the owner knew.
- **One scene, one still, one seed.** No face in close-up, no second
  reference, no product with fine detail.
- **Not the vendor's path.** sglang gives the still to both the encoder and
  the VAE (`docs/research/sglang_h3_pipeline.md`, "Reference stills").
- Nothing about a reference video or audio.

So it is a lead, and no default moved. What followed: `use_vae` on
`MiniMaxH3AppendRefImage`, so one reference can be encoder-only without
unwiring the VAE for all of them.
