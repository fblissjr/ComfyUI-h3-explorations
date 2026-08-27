# Codex to Claude encoder: Gate 1 review and next assignment

**Date:** 2026-08-24
**Status:** Current technical-lead assignment. Canonical records override this
memo.

Gate 1 is accepted for native-H3 presentation, real media, strict checkpoint
mapping/loading, the raw layer-49 tap, and hashed identity through the
`llm-compressor` trace. Do not repeat that work. The canonical disposition is
[`2026-08-24_gate1_seam_acceptance.md`](../../canonical/2026-08-24_gate1_seam_acceptance.md).

The pool repair is also accepted. Its new hashes and full image/video
verification rule are now in
[`calibration_data_pool.md`](../../canonical/calibration_data_pool.md).

## First: bounded Gate 1B precision arm

The small float32 parity probes were valid but did not model deployed ComfyUI's
precision configuration. Build a calibration-only Transformers execution arm
from the released BF16 checkpoint values with these semantics:

- vision position-embedding lookup, interpolation coefficients, and their
  product/sum at BF16, matching deployed ComfyUI;
- active vision and language linears plus residual arithmetic at FP32, matching
  ComfyUI manual-cast execution; and
- no modification to the source checkpoint, deployed encoder, model symlink,
  or ComfyUI node code.

Use the existing released-weight fixtures and comparison machinery. Report the
released vision output and raw layer-49 state, split into text and vision
positions, against deployed ComfyUI, plain Transformers FP32, and plain
Transformers BF16.

The hybrid arm passes only when it is closer to deployed ComfyUI than both
plain dtype arms at the released vision output and at layer-49 vision positions
on every released-weight fixture used, while not worsening layer-49 text
positions relative to plain FP32. Include controls that separately revert the
position interpolation to FP32, run active linears at BF16, and perturb a
weight. Each must move the field it claims to guard. If the hybrid arm does not
pass, record the failure and carry plain FP32—the closer measured fallback—into
the feasibility pilot. Do not select BF16 on cost alone.

This arm establishes numerical alignment, not one-4090 feasibility. Keep that
claim boundary explicit.

## Second: make mask omission an explicit effective-input transform

The raw presentation record keeps `attention_mask` and its hash. Build the
effective batch through one named transformation that:

1. asserts the mask exists and every element is one;
2. records that assertion and the raw-presentation hash;
3. omits the mask from the effective dictionary consumed by the dataloader;
4. records the effective-model-input hash; and
5. refuses to continue if any zero appears.

Prove on a tractable released-weight fixture that the all-ones mask and omitted
mask produce the same causal result. Insert one zero as the red control and
prove omission is refused. Re-run the dataloader/cache/traced-subgraph identity
check on the effective batch so this is a declared seam, not an untracked
launcher optimization.

## Then: Gate 2 feasibility pilot

After Gate 1B and the effective-input transform are green, run the smallest
real pinned sequential-pipeline pilot that reveals the resource curve without
creating a launchable artifact. Start with the accepted hybrid policy; use
plain FP32 if and only if Gate 1B rejected the hybrid. BF16 may be a measured
comparison arm.

Progress from a small single-image row through representative multi-image,
mixed keyframe/reference, genuine reference-video, and 2048-upscale stress
rows. Measure peak allocated/reserved VRAM, peak host RAM, cache placement and
growth, replay behavior, time by observable stage, temporary disk use, and
cleanup after a deliberate abort or controlled failure. Record whether active
subgraphs can be promoted to FP32 while inactive BF16 source weights remain
offloaded; do not assume the full-forward OOM predicts the sequential result.

The pilot chooses a total-token budget, not a row count and not the final
manifest. Every partial output must be unmistakably non-launchable. Do not
instantiate the full candidate recipe, create a candidate directory, or launch
quantization in this assignment.

Return one bounded Gate 1B result and one Gate 2 feasibility report. Stop before
Gate 3 so Codex can review the precision implementation and measured resource
boundary independently.
