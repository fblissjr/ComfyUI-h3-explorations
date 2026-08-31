# Transformers-versus-Comfy Qwen3-VL implementation parity

**Status:** Measured implementation-level and released-configuration result
**Observation date:** 2026-08-24
**Scope:** M-RoPE position IDs, vision merged output, DeepStack features, and
the released-weight precision seam

## Why this boundary matters

The intended v2 calibration path drives Transformers'
`Qwen3VLForConditionalGeneration` through `llm-compressor`, while deployed H3
conditioning uses ComfyUI's Qwen3-VL implementation. Calibration would collect
the wrong activation distribution if the two implementations disagreed on
position IDs or vision-tower arithmetic even when they received identical
inputs and weights.

Two probes compare these independent implementations and include deliberate
mutations that must make the comparisons fail:

- [`probe_mrope_implementation_parity.py`](../../../../bench/probe_mrope_implementation_parity.py)
- [`probe_vision_tower_parity.py`](../../../../bench/probe_vision_tower_parity.py)

## Results

**MEASURED.** M-RoPE position IDs agreed exactly across six fixtures: single
square, wide, and tall blocks; two equal blocks; three mixed blocks; and a
nine-block Ref2VA-shaped case. Shifting one vision span on one side produced 24
mismatched positions, so the check detects the structural defect it is intended
to catch.

- [`2026-08-24_mrope_implementation_parity.json`](../../../../bench/results/2026-08-24_mrope_implementation_parity.json)

**MEASURED.** The vision probe transferred 39 shared state-dict keys directly,
without name mapping, from one explicitly seeded finite float32 state dict into
both small models. Across the same six geometry families, the largest absolute
difference over the post-merger output and DeepStack features was
`2.384185791015625e-7`. Perturbing one Transformers merger weight was detected.

| Fixture | patch rows | worst maximum absolute difference |
|---|---:|---:|
| single square block | 16 | `5.960e-8` |
| single wide block | 48 | `2.384e-7` |
| single tall block | 48 | `1.788e-7` |
| two equal blocks | 32 | `8.941e-8` |
| three mixed blocks | 112 | `1.490e-7` |
| nine blocks, Ref2VA-shaped | 224 | `1.192e-7` |

- [`2026-08-24_vision_tower_parity.json`](../../../../bench/results/2026-08-24_vision_tower_parity.json)

## Two false-green hazards now fixed by the probe

**SOURCE.** Transformers exposes the post-merger vision result as
`pooler_output`; `last_hidden_state` is the pre-merge state. The probe unwraps
the result by field name. Comparing ComfyUI's merged result to
`last_hidden_state` would compare different tensors and can evade a simple
shape check at released dimensions.

**SOURCE.** ComfyUI's ops allocate parameters uninitialized because they expect
checkpoint loading. The probe does not compare framework initializers. It
constructs one seeded finite state dict, loads it strictly into both models,
and checks all weights are finite before comparison.

## Released weights expose a configuration-level precision gap

The float32 proxy result above remains valid for shared arithmetic. It did not
represent the deployed configuration: ComfyUI stores the released vision
weights in BF16, casts its manual-cast linears to the FP32 activation dtype, but
performs position-embedding lookup and interpolation at the stored BF16 dtype.
Plain Transformers FP32 and BF16 each choose one dtype for both operations, so
neither reproduces that combination.

**MEASURED.** On identical real patches and the released vision weights:

| comparison | relative L2 |
|---|---:|
| ComfyUI FP32 parameters versus Transformers FP32 | 0.001379 |
| deployed ComfyUI BF16 parameters versus ComfyUI FP32 parameters | 0.018735 |
| deployed ComfyUI versus Transformers FP32 | 0.018805 |
| deployed ComfyUI versus Transformers BF16 | 0.095507 |

The comparison includes the merged output and DeepStack features. A merger
weight mutation moves the matched-precision comparison by more than an order of
magnitude, so the probe can distinguish the claimed agreement from a wrong
weight.

- [`probe_released_vision_precision.py`](../../../../bench/probe_released_vision_precision.py)
- [`2026-08-24_released_vision_precision.json`](../../../../bench/results/2026-08-24_released_vision_precision.json)

**MEASURED.** The same difference is already present at the language model's
layer-0 input. On a released-weight two-picture fixture, text positions were
bit-identical while vision positions had relative L2 0.018805. After decoder
layer 49, Transformers FP32 versus deployed ComfyUI measured relative L2
0.002615 on text positions and 0.393034 on vision positions. The aggregate is
therefore dominated by visual-row share and is not a reliable wrong-layer
control unless results are split by position class.

- [`compare_transformers_comfy_layer50.py`](../../../../bench/compare_transformers_comfy_layer50.py)
- [`2026-08-24_crossstack_layer50_mixed.json`](../../../../bench/results/archive/v2_encoder/2026-08-24_crossstack_layer50_mixed.json)
- [`2026-08-24_crossstack_layer50_controls.json`](../../../../bench/results/archive/v2_encoder/2026-08-24_crossstack_layer50_controls.json)

This narrowed the earlier conclusion rather than retracting it. The two
implementations agree closely when their precision configuration is matched;
plain Transformers FP32 and BF16 do not match the deployed configuration.

## Gate 1B: the accepted calibration precision policy

`probe_position_embedding_parity.py` isolated the remaining position-embedding
difference step by step. On four real released-weight geometry fixtures,
ComfyUI and Transformers produced identical interpolation indices and BF16
coefficients. The difference was the BF16 reduction order: Transformers uses a
generic `.sum(1)`, while ComfyUI explicitly adds the four weighted terms. The
explicit four-term reduction reproduced ComfyUI's position-embedding tensor
bit-for-bit on all four fixtures. Corrupt-index and corrupt-weight controls
moved the output.

[`h3_calibration_precision.py`](../../../../bench/h3_calibration_precision.py)
implements that result as the calibration-only `comfy_exact` policy:

- released BF16 position-embedding values;
- BF16 interpolation coefficients;
- ComfyUI's explicit four-term reduction order; and
- FP32 active vision and decoder compute.

The policy is scoped to one model instance and one entering thread, restores
every modified object after success or exception, refuses an unrecognized
Transformers source expression, and does not edit the checkpoint, installed
package, or deployed ComfyUI model. The executable guard and its mutation
control are
[`check_calibration_precision_policy.py`](../../../../bench/check_calibration_precision_policy.py).

**MEASURED.** `comfy_exact` passed the predeclared Gate 1B rule on every real
fixture. The table reports the worst relative L2 across the merged vision
output and all three DeepStack outputs against deployed ComfyUI:

| fixture grid | plain FP32 | native BF16 | `comfy_exact` |
|---|---:|---:|---:|
| one `44 x 40` block | 0.009712 | 0.045732 | 0.001255 |
| two `18 x 18` blocks | 0.008357 | 0.036253 | 0.000945 |
| one `48 x 84` keyframe block | 0.010883 | 0.051106 | 0.001153 |
| `84 x 48` keyframe plus `22 x 22` reference | 0.018805 | 0.095507 | 0.001175 |

At the raw layer-49 boundary, `comfy_exact` was closer than plain FP32 and
native BF16 on vision positions for all four fixtures and did not worsen text
positions relative to FP32. The largest remaining vision-position relative L2
was 0.059267 on the single `44 x 40` fixture. This policy is therefore the
closest measured calibration approximation, not a claim of bitwise full-stack
identity.

The stable evidence is grouped by fixture:

| fixture | interpolation | vision/DeepStack | layer 49 |
|---|---|---|---|
| single image | [position](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_position_single_image.json) | [vision](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_vision_single_image.json) | [layer 49](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_layer49_single_image.json) |
| multi-image | [position](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_position_multi_image.json) | [vision](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_vision_multi_image.json) | [layer 49](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_layer49_multi_image.json) |
| keyframe | [position](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_position_keyframe_only.json) | [vision](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_vision_keyframe_only.json) | [layer 49](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_layer49_keyframe_only.json) |
| mixed keyframe/reference | [position](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_position_mixed_keyframe_reference.json) | [vision](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_vision_mixed_keyframe_reference.json) | [layer 49](../../../../bench/results/archive/v2_encoder/2026-08-24_gate1b_layer49_mixed_keyframe_reference.json) |

## Decision and limits

**DECISION.** The implementation-level M-RoPE and vision/DeepStack arithmetic
gate and the released-configuration Gate 1B precision gate are closed.
Subsequent calibration feasibility work uses `comfy_exact`. Plain FP32, native
BF16, and the earlier generic-sum hybrid remain comparison arms; lower resource
cost cannot promote one of them to the candidate policy.

This does not establish:

- bitwise full-stack equality at released weights;
- equality of image/video preprocessing before the vision tower;
- BF16/CUDA kernel bitwise identity;
- correct media, grid, label, or token handoff into `oneshot`;
- calibration memory or runtime feasibility; or
- numerical or perceptual fidelity of any quantized checkpoint.

The first vision probe deliberately uses a small float32 configuration. Its
result establishes shared arithmetic under that configuration. The Gate 1B
matrix establishes the accepted precision policy on four released-weight
fixtures; it is not a population estimate, quantization result, or proof that
the policy is feasible through the real AWQ modifier on the RTX 4090.
