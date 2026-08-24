# Transformers-versus-Comfy Qwen3-VL implementation parity

**Status:** Measured implementation-level result
**Observation date:** 2026-08-24
**Scope:** M-RoPE position IDs, vision merged output, and DeepStack features

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

## Decision and limits

**DECISION.** The implementation-level M-RoPE and vision-tower arithmetic parity
gate is closed. A replacement v2 preflight should consume this evidence and
move to the exact native-presentation-to-`llm-compressor` seam. Repeating these
probes is not the next step unless a new code revision or escaped defect changes
the boundary.

This does not establish:

- full released-checkpoint mapping or strict loading;
- equality of image/video preprocessing before the vision tower;
- BF16/CUDA kernel bitwise identity;
- correct media, grid, label, or token handoff into `oneshot`;
- calibration memory or runtime feasibility; or
- numerical or perceptual fidelity of any quantized checkpoint.

The vision probe deliberately uses a small float32 configuration. Its result is
strong evidence that the shared arithmetic is aligned; it is not a substitute
for validating the full calibration seam and candidate artifact.
