# Pika H3 Flex: how close each implementation runs to Pika's release

last updated: 2026-11-22 (the pack's Pika ref2va graphs size reference stills by Pika's rule, `tiered`; sglang's Pika follows the release on references, keyframes and audio length; Pika answered how the distill was trained; [`../sweeps/2026-11-22.md`](../sweeps/2026-11-22.md)); 2026-11-15 (page created from the 2026-11-15 sweep)

Pika H3 Flex is the owner's default distill. Its contract (knots, re-noise sampler, shifts) is in
`../../references.md`, "Pika H3 Flex", and `workflows/h3_config.py` (`PIKA_*`). This page tracks how each engine
matches Pika's own numerics beyond the contract. That matters for seed-exact parity, and possibly for quality.

## Agreements across engines (2026-11-15; last three items 2026-11-22)

- **The contract.** LightX2V's new six-step configs (`1053bb0a`;
  `coderef/LightX2V/configs/minimax_h3/distill/6steps/minimax_h3_pika_*_6step.json`) have the same knots, shifts and
  no-CFG as `h3_config.PIKA_*`. sglang still validates shifts 12/3.
- **Separate RNG streams.** Three engines seed the initial noise with `seed`, the re-noise with `seed + 1`, and the
  condition noise with `seed + 7919`. They are sglang (`f98e9611bc`, #47468;
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py::minimax_h3_imgvid_cond_noise_aug_rows`), LightX2V (`1053bb0a`) and vllm-omni
  (draft #9777). Draws are in Pika's own tensor layout, and drawing in token order changes the result (vllm-omni
  #9777's note).
- **LoRA factors at bf16, unmerged.** sglang casts the release's fp32 factors to bf16
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py`, `load_lora_adapter`). vllm-omni's native loader casts A and B to the
  model dtype. `MiniMaxH3LoRABranch` casts its factors to the compute dtype (`lora_branch.py`), so it aligns.
- **TF32 matmul off.** sglang sets `torch.set_float32_matmul_precision("highest")`. Core sets no global TF32 flag, so
  torch's default (off) applies. That is the agent's reasoning, not run.
- **The training schedule is the inference schedule** (Pika issue #9, answered 2026-11-20 by a Pika contributor):
  distillation ran at shifts 12/3 on knots `[1000, 750, 500, 250, 100, 40, 0]`, which are
  `h3_config.PIKA_CONTRACT_KNOTS`. SFT and PDMD were both LoRAs, and the released adapter holds ReFL+DPO merged
  weights.
- **Decoded audio ends with the video.** sglang's Pika trims the decoded audio to the video's duration
  (`trim_audio_to_video=True`, `44a391941a`), as the release does. The audio latent count is the same rounding of
  `frames / 24 * 40` in sglang, Pika and core, so the two lengths already agree on every `h3_rules.on_both_clocks`
  length; the pack's `VHS_VideoCombine` mux uses `-shortest` (agent's reading).
- **Reference stills are enlarged by canvas tier** (2026-11-22). Pika's release takes a 2048 short edge from the
  768 tier up and 1024 below it (`coderef/Pika-H3-Flex/src/pika_h3/core/buckets.py::reference_short_edge_for_canvas`).
  The pack's Pika ref2va graphs send `size_policy="tiered"` (`h3_config.PIKA_REF_SIZE_POLICY`,
  `reference_geometry.py::tier_short_edge`), the same rule. sglang's Pika uses `diffusers` since `44a391941a`
  (#51935), the same size from the 768 tier up; no sglang mode has the 1024 tier
  ([`reference_sizing.md`](reference_sizing.md)).

## Differences

- fl2va rounds the clean condition rows to bf16 before the noise is added, ref2va does not (sglang, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::_apply_condition_noise_aug`).
  Not compared with our path.
- **Keyframes.** Pika's release center-crops each keyframe at source resolution, then resizes
  (`coderef/Pika-H3-Flex/src/pika_h3/core/conditioning.py::fit_keyframe`).
  - sglang's Pika has done the same since `44a391941a` (`keyframe_fit="source_crop"`).
  - The pack's follower keyframe is cropped the same way.
  - Under `canvas="from_keyframe"` the pack's anchor is scaled without a crop
    (`keyframe_canvas.py::resolve_keyframe_geometry`, `match_keyframe`). The canvas follows the anchor's aspect,
    so the difference is only the canvas's rounding to 32 px.
- vllm-omni #9777 adds `minimax_h3_reference_numerics=true` to keep the official bf16 rounding points. It claims
  near-identity with Pika's outputs on t2va/fl2va; ref2va is not covered by the claim.

## Open

- Pika issue #10: garbled speech mid-clip on R2V with the Flex LoRA, clean on the base model; Pika asked for the
  assets ([`../horizon.md`](../horizon.md)). Issue #9 was answered 2026-11-20 (Agreements, above).
- Whether our condition-noise draw order and seed offsets match these. Not compared.
