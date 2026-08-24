# Gate 1 native-H3 calibration-seam acceptance

**Status:** Accepted Gate 1 and Gate 1B evidence
**Observation date:** 2026-08-24
**Evidence commit:** `6255afa`
**Boundary:** no quantization, recipe, candidate directory, deployment change,
or launch authorization

## Accepted findings

**MEASURED.** Five real decoded H3-IR fixtures cover single-image,
multi-image, keyframe, mixed keyframe-plus-reference, and genuine
reference-video roles under the accepted v2 geometry policy. The installed H3
path produces their labels, ordering, timestamps, token IDs, token tags,
multimodal token types, patches, grids, spans, M-RoPE inputs, and DeepStack
placement. The calibration process consumes a hashed bundle because the pinned
ComfyUI and `llm-compressor` environments cannot import each other; identity is
re-established at the bundle, preconstructed dataloader,
`IntermediatesCache`, and traced-subgraph boundaries.

**MEASURED.** Independent vendor-shaped and Transformers-processor arms agree
with the installed presentation wherever their presentation contracts can
express the same row. Mutation controls detect chat wrapping, first-image
slicing, media reorder or loss, timestamp drift, missing temporal repeat, grid
change, missing multimodal token types, and a builder disconnected from the
dataloader. Token tags are correctly detected by the presentation gate and are
correctly invisible to the Qwen/`oneshot` chain because they are consumed later
by H3's DiT.

**MEASURED.** The released checkpoint is a strict bijection with the
Transformers calibration model: 1,058 tensors, all 64 decoder layers, three
DeepStack mergers, and no missing, unexpected, or mismatched keys. The
candidate target boundary is unchanged: quantize the decoder linears while the
351 vision/DeepStack tensors and input embedding remain BF16. The raw
unnormalized state after decoder layer 49 is the H3 output tap.

The detailed evidence and controls are in
[`2026-08-24-gate1-seam-audit.md`](../brainstorming/claude-encoder/2026-08-24-gate1-seam-audit.md).

## Corrections consumed by this acceptance

The accepted pool now opens and hashes every declared image and video with no
role exemption. The repaired provenance is authoritative in
[`calibration_data_pool.md`](calibration_data_pool.md).

The earlier small float32 parity probe established shared arithmetic, not the
precision configuration deployed by ComfyUI. Released-weight measurements show
that plain Transformers FP32 and BF16 do not reproduce ComfyUI's BF16 position
interpolation, explicit four-term BF16 reduction, and FP32 active compute. The
calibration-only `comfy_exact` policy reproduces the position embedding
bit-for-bit and passed the released-weight Gate 1B matrix, with a bounded
remaining full-stack residual.
[`2026-08-24_transformers_comfy_parity.md`](2026-08-24_transformers_comfy_parity.md)
owns that correction and the Gate 1B acceptance rule.

## Effective attention-mask rule

**MEASURED.** On the released-weight comparison fixture, keeping versus
omitting an all-ones mask under the same forced math backend was bit-identical
at the raw layer-49 state. The mask therefore added no masking semantics on
that eligible input; omitting it preserved causal attention.

Changing from forced math to the automatic kernel policy while also omitting
the mask produced relative L2 0.000420. That comparison establishes
kernel-policy sensitivity, not the identity of the automatically selected
backend. The availability API does not identify selection, and availability
must be queried at the real tensor shape and causal mode. The artifact labels
that arm `auto`, not as a named fused kernel. Earlier long-fixture memory and
timing behavior remains a resource observation, not backend attribution.

- [`2026-08-24_effective_mask_equivalence.json`](../../../../bench/results/2026-08-24_effective_mask_equivalence.json)

**DECISION.** Keep the raw attention mask and its hash in the presentation
record. Before the effective calibration batch is built, assert that every
element is one. Only then omit `attention_mask` from the dictionary handed to
the dataloader and traced graph. Record both the raw-presentation hash and the
effective-model-input hash. A mask containing any zero is not eligible for this
normalization and must stop the run. A red control must insert a zero and prove
that omission is refused. The seam identity proof for subsequent gates applies
to the effective batch after this declared normalization.

## Gate 1B and Gate 2 boundary

Gate 1 is accepted for presentation, media, checkpoint mapping, and trace
identity. Gate 1B accepts `comfy_exact` as the calibration execution policy.
Neither gate modifies the deployed ComfyUI model or source checkpoint.

Gate 2A may now measure the no-modifier sequential floor. That run validates
cache/replay mechanics and establishes a lower resource bound only. It cannot
set the calibration population because it does not instantiate AWQ observers or
modifier state. Gate 2B must run a bounded real AWQ modifier before the plan can
freeze an absolute row or token budget.
