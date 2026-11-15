# Pika H3 Flex: how close each implementation runs to Pika's release

last updated: 2026-11-15 (page created from the 2026-11-15 sweep)

Pika H3 Flex is the owner's default distill. Its contract (knots, re-noise sampler, shifts) is in
`../../references.md`, "Pika H3 Flex", and `workflows/h3_config.py` (`PIKA_*`). This page tracks how each engine
matches Pika's own numerics beyond the contract. That matters for seed-exact parity, and possibly for quality.

## Agreements across engines (2026-11-15)

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

## Differences

- fl2va rounds the clean condition rows to bf16 before the noise is added, ref2va does not (sglang, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::_apply_condition_noise_aug`).
  Not compared with our path.
- sglang's Pika inherits sglang's new reference default and does not enlarge small stills. Pika's own `buckets.yaml`
  does ([`reference_sizing.md`](reference_sizing.md)).
- vllm-omni #9777 adds `minimax_h3_reference_numerics=true` to keep the official bf16 rounding points. It claims
  near-identity with Pika's outputs on t2va/fl2va; ref2va is not covered by the claim.

## Open

- Pika issue #9 (unanswered): whether training was LoRA or full, what the training shift was, and what the adapter
  holds ([`../horizon.md`](../horizon.md)).
- Whether our condition-noise draw order and seed offsets match these. Not compared.
