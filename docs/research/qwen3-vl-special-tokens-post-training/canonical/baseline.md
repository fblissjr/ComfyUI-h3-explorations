# Canonical baseline: facts, observations, and unknowns

**Status:** Authoritative baseline
**Last updated:** 2026-08-25

## Owner objective

- **OWNER-DECISION:** The target is a strong multimodal H3 conditioning
  encoder. Reference images and reference videos pass through Qwen3-VL and are
  first-class workloads; text-only T2VA evidence is not sufficient to approve
  a new quantization or post-training result.
- This objective does not imply that AWQ improves the BF16 vision tower. The
  v2 quantization experiment leaves that tower unchanged and asks whether the
  W4 language stack better preserves the fused visual, DeepStack, and text
  distributions H3 actually consumes.

## Released encoder and H3 boundary

- **SOURCE:** The encoder is Qwen3-VL 32B with 64 language decoder layers,
  hidden size 5120, 64 attention heads, 8 key/value heads, and a 27-block vision
  tower. H3 consumes the first 50 language layers.
- **SOURCE:** Native ComfyUI returns the unnormalized output after language
  layer index 49. It does not apply the full-Qwen final RMSNorm or LM head to
  H3 conditioning. See the installed
  [`minimax.py`](../../../../../../comfy/text_encoders/minimax.py).
- **SOURCE:** The released tokenizer assigns these seven additional special
  tokens:

  | ID | Token |
  |---:|---|
  | 151669 | `<d>` |
  | 151670 | `</d>` |
  | 151671 | `<|cutoff|>` |
  | 151672 | `<|lyrics_start|>` |
  | 151673 | `<|lyrics_end|>` |
  | 151674 | `<|caption_start|>` |
  | 151675 | `<|caption_end|>` |

- **MEASURED, 2026-08-25:** The released text encoder is byte-identical to
  stock `Qwen/Qwen3-VL-32B-Instruct`: all 14 `text_encoder/` shards of the
  release carry the same Hub LFS SHA-256 and size as the 14 stock shards, so
  every tensor, the embedding table included, is the stock release and no
  post-training of the encoder shipped. What MiniMax runs behind its API is
  not observable from the release. Producer and record:
  [`check_released_encoder_is_stock.py`](../../../../bench/check_released_encoder_is_stock.py),
  [`2026-08-25_released_encoder_is_stock.json`](../../../../bench/results/2026-08-25_released_encoder_is_stock.json).
- **MEASURED:** Those seven embedding rows match the corresponding unused stock
  Qwen3-VL tail rows and fall on the untrained control distribution. The
  release did not language-model-train those rows. Evidence and controls are in
  [`official_weights_metadata.md`](../../official_weights_metadata.md) and
  [`2026-08-21_h3_token_embeddings.json`](../../../../bench/results/2026-08-21_h3_token_embeddings.json).
- **MEASURED:** Dedicated IDs, ordinary BPE spellings, and stripped markers
  produce materially different layer-50 states on shared ordinary-token spans.
  This is encoder evidence only; it does not identify which representation the
  DiT prefers. See
  [`2026-08-21_h3_marker_token_states.json`](../../../../bench/results/2026-08-21_h3_marker_token_states.json).

## Caption-marker result

- **OWNER-OBSERVED:** The completed 2026-08-23 caption-marker arms demonstrated
  that the caption marker can affect H3 behavior in the tested setup. The
  marker must not be described as inert or as “doing nothing.”
- The bounded conclusion is **behavioral activity in that setup**. It does not
  by itself establish the effect size across seeds, whether every effect is
  beneficial, the behavior of the other six tokens, or which tokenizer
  realization MiniMax used during DiT training.
- The tracked arm document still describes its earlier review state and should
  receive a separate outcome write-up when the final observations and artifacts
  are ready: [`2026-08-23_caption_marker_arms.md`](../../../../internal/prompts/2026-08-23_caption_marker_arms.md).

## Current W4A16 AWQ artifact

- **MEASURED:** The single-file artifact is 20,394,199,288 bytes and contains
  all 64 decoder layers on disk: 448 group-128 W4 decoder linears and 448 scale
  tensors. The adapter retains layers 0--49, or 350 W4 linears, for H3.
- **MEASURED:** The artifact preserves the vision tower, DeepStack mergers, and
  token embedding table in BF16. This is a storage fact, not proof of zero
  downstream degradation.
- **SOURCE:** The repo adapter performs the 64-to-50-layer truncation and routes
  W4 execution through `comfy_kitchen.gemv_awq_w4a16`. See
  [`h3_awq_encoder.py`](../../../../h3_awq_encoder.py).
- **MEASURED:** A real loader smoke recorded approximately 14.97 GB of staged
  H3-relevant weights. This is not a total peak-VRAM guarantee and must not be
  added to separately estimated components that it may already include. See
  [`h3_awq_encoder.md`](../../../h3_awq_encoder.md).

## What the successful calibration run actually used

- **MEASURED:** Exactly 96 rows were processed: 50 from
  `StellarVoyager/H3-IR`, 30 from local extracted MP4 metadata/frame 0, and 16
  from `oakmindai/minimax_h3_avatar_500`. No fallback duplication occurred.
- **MEASURED:** Every row presented exactly one image plus text. The successful
  run contained no text-only row, multi-image array, FL2VA pair, or video block.
- **MEASURED:** The calibration path used the Hugging Face chat template with
  `add_generation_prompt=True`; it did not use native raw H3 presentation.
- **MEASURED:** Still images were constrained to 200,704--301,056 pixels for
  that run.
- **MEASURED from the audited reconstruction:** `<d>` and `</d>` each occur 133
  times across 80 of the 96 reconstructed prompts. Caption, lyrics, and cutoff
  tokens have zero occurrences. Reconstructed sequence lengths span 437--1,879
  tokens. The row-level supporting artifact remains explicitly a reconstruction,
  not an original run-emitted manifest:
  [`2026-08-24_awq_calibration_manifest.jsonl`](../archive/brainstorming/2026-08-24/gemini/2026-08-24_awq_calibration_manifest.jsonl).
- **MEASURED:** All 64 language layers were quantized. The successful run did
  not calibrate only the 50 layers H3 later consumes.

## Autograd and numerical fidelity

- **MEASURED:** Backward through the installed
  `comfy_kitchen.gemv_awq_w4a16` raises because no autograd formula is
  registered. This is a property of the installed execution path, not proof
  that the mathematical operation can never have a backward implementation.
- **MEASURED:** Existing validation checked structure, dtypes, shapes, loading,
  and a layer-50 output shape. It did not compute BF16-versus-W4 cosine
  similarity, MSE, relative L2, or a downstream quality equivalence.
- **MEASURED, bounded substrate:** The current artifact now has a controlled
  six-family BF16-versus-W4 layer-50 baseline under its own processor policy,
  plus held-out single-reference processor-policy measurements. The result is
  fixture-level evidence rather than a population estimate. See
  [`2026-08-24_layer50_processor_policy_benchmark.md`](2026-08-24_layer50_processor_policy_benchmark.md).
- **UNKNOWN:** Population-level current-W4 drift on the accepted
  component-disjoint H3-IR holdout, including deployed-path preprocessing and
  role-stratified reference-video results.
- **UNKNOWN:** Whether a better native-H3 calibration materially improves that
  drift or downstream generations.
- The released depth contract and the local encoder variants' embedding/layer
  inventories are recorded separately in
  [`encoder_depth_and_embedding.md`](encoder_depth_and_embedding.md).

## Training-history and marker unknowns

- **UNKNOWN:** Which exact tokenizer realization was used for MiniMax's DiT
  training. Untrained rows do not distinguish dedicated-ID training from a
  legacy-BPE training path.
- **UNKNOWN:** Whether every one of the seven dedicated IDs appeared in DiT
  training data.
- **UNKNOWN:** The detailed learned downstream mechanism of each marker. Token
  names and prompt syntax do not prove claims about specific attention
  transitions, pitch, tempo, padding bleed, or spatial binding.
- **UNKNOWN:** Whether dialogue post-training is useful. That decision requires
  controlled DiT-level baselines; encoder representation distance alone is a
  proxy.
- **UNKNOWN:** A correct and practical W4 backward implementation, its peak
  memory, its runtime, and its numerical agreement with a BF16 reference.

## Claims that are not canonical

The following numbers and conclusions remain unsupported and must not be used
as planning facts:

- 512-sample or 256-sample successful calibration populations;
- four calibration buckets in the successful run;
- native multi-image or video calibration in the successful run;
- “98--99% identical” requantized weights;
- a 19.27 GB training peak or 4.73 GB guaranteed headroom;
- 15-minute or 90-minute training durations;
- a 22% diffusion-noise floor;
- five prompts by three seeds confirming H1, H2, or H3;
- exact BPE-to-single-token representation transplantation; and
- clean generation or valid tensor shapes proving BF16 fidelity.
