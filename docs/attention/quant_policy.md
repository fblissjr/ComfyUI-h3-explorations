# H3 quantization policy: which blocks get which precision

A quantized H3 is a per-block policy: for each of the fifty DiT blocks, what the attention kernel quantizes q/k/v to,
and what precision the linears are stored at. This page holds the policy, one row per decision with its status and
evidence. An empty status means nobody has measured the row. The mechanism behind the attention rows is
[`block49_quant_error.md`](block49_quant_error.md): unrotated INT8 attention loses accuracy on the blocks whose
`k_norm.weight` is lopsided (45, 48, 49 on every H3 checkpoint), and rotation removes most of it.

## The policy table (attention)

| where | attention | status | evidence |
|---|---|---|---|
| Sol's routed steps, every block not in `dense_blocks` | Sol, INT8, `quantizer="rotated"` (q/k Hadamard-rotated before INT8) | shipped since 2026-09-27 (owner) | lowest Sol quantization error on every graded cell of the shipped PDD8 t2v and ref2va captures, cheaper than `balanced` (`bench/results/2026-09-27_sol_redesign_test2.md`). `balanced` (`qk_balance`) shipped from 2026-09-15 to 2026-09-27 |
| the dense steps (before `start_percent`), calls under `min_tokens`, and the blocks in `dense_blocks` | kitchen `int8_attention` through core's Model Attention Backend (`h3_config.DENSE_BACKEND_NODE`), which rotates q/k before INT8; no sage | shipped since 2026-09-15 (owner) | immune to the loud channels by construction, a third of Sol's and sage's error on block 49 ([`data/results/2026-09-15_ck_int8_attention_block49.json`](../../data/results/2026-09-15_ck_int8_attention_block49.json)); below sage at every captured cell, step 4 included ([`data/results/2026-09-15_dense_kernels_by_step.json`](../../data/results/2026-09-15_dense_kernels_by_step.json)); same wall time as the sage-dense chain (`bench/results/2026-09-15_block49_community_chain.md`) |
| `dense_blocks` | `h3_config.SOL_DENSE_TAIL`, `"38,39,40,41,42,49"` | shipped since 2026-10-02 (owner, kept by decision) | [`sol_dense_blocks.md`](sol_dense_blocks.md). The 2026-09-25 tail `"45,48,49"` was the unrotated-INT8 answer; under rotation block 48's error is 7.14% and 45's 9.55%, while routing error peaks at blocks 38 to 43 |
| 45, 48, 49 on bf16 (`MiniMaxH3ExactBlocks`) | ComfyUI's own attention at those blocks | opt-in, on the sage-chain arms `h3_probe_t2v_exact_tail` and `h3_probe_t2v_policy` | the best arm on three scenes on the sage chain; against every free lever on, a small edge ("cannot tell them apart" to "maybe faces") at about a sixth more render time ([`block49_quant_error.md`](block49_quant_error.md), section 6c). Not re-judged since rotation shipped |

**An open question about the dense kernel.** On one seed of the market scene, every configuration whose dense steps ran
kitchen's rotated INT8 kernel lost the hand in the coin drop, and every configuration on sage or bf16 dense steps kept
it (`bench/results/2026-09-15_block49_community_chain.md`, the controls section), while every capture instrument puts
the kitchen kernel below sage (`bench/results/2026-09-15_dense_kernels_*.json`). The per-call error grade cannot see that flaw. Eleven reproduction renders, default
chain against the sage levers chain on six scenes, are the test; their scoring was pending at last record
(`bench/results/2026-09-15_block49_repro_batch.md`, stacks under `Video/block49_repro/stacks/`).

## The policy table (linears)

| blocks | linears | status | evidence |
|---|---|---|---|
| all | int8 convrot (`qkv_proj`, `out_proj`, `fc1`, `fc2`) | shipped | inherited; no sensitivity measurement exists here |
| tail | bf16? | unmeasured | TaoMate (deprecated, history only) protected 0, 1, 47, 48, 49 on its W8A8 path: a near-miss against our loud set, and a different surface |

The loud channels are created by the k_norm gain after the qkv projection, so the convrot INT8 linears never see them
(read from the model code, not measured).

## LoRAs on the int8 checkpoints

A LoRA merged by `LoraLoaderModelOnly` is requantized into each int8 layer with stochastic rounding
(`comfy/ops.py::resolve_cast_module_with_vbar`). A delta well below one int8 step survives only in expectation, under
rounding noise several times its size. How far below a step each distill's delta sits, per layer:
[`bench/results/2026-09-26_int8_lora_requant.json`](../../bench/results/2026-09-26_int8_lora_requant.json), from
`bench/probe_int8_lora_requant.py`. `MiniMaxH3LoRABranch` (`lora_branch.py`) applies a LoRA at the call instead and
leaves the int8 weight as shipped.

**The policy since 0.154.0 (owner, 2026-09-26): every LoRA on an int8 checkpoint goes through the branch.**

- The generator's `lora_branch` defaults on.
- `MiniMaxH3PDDLoRA` applies PDD's backbone and refiner the same way (`backbone_apply`).
- A merge survives only on a named control arm. Which graphs still merge is a walk of `graph_paths` for
  `LoraLoaderModelOnly` and for `backbone_apply: "merge"`.
- PDD's adaln update stays merged: it lands on an fp16 weight, not int8.

## The plan, in tiers

Each tier is usable alone; each later tier removes a cost of the one before.

| tier | what | status |
|---|---|---|
| 0 | the policy as a graph: `h3_probe_t2v_policy` carries the channel-balance node on the loud blocks, sage `fp8++ balanced`, and 45/48/49 on exact attention. The market scene's porter morphing as he turns is the regression witness | built 2026-09-15; a sage-chain arm (`bench/check_attention_defaults.py::FLOOR_STEMS`) |
| 1 | the per-head balance factor inside Sol's own quantizer, so the routed steps get what the sage steps get | built, graded and installed 2026-09-15 (`qk_balance`): about twice what the weights fold removes from Sol's INT8 term at block 49, neutral at blocks 0 and 32 ([`bench/results/2026-09-15_channel_balance_kernel_b{49,0,32}_s15.json`](../../bench/results/2026-09-15_channel_balance_kernel_b49_s15.json)); one to two percent of a Sol call; shipped 2026-09-15 to 2026-09-27 |
| 2 | INT8 that survives the loud blocks: a Hadamard rotation of q/k inside Sol's quantizer, as kitchen's `int8_attention` already does. Orthogonal, so exact for the scores, and it flattens per-token spikes no per-channel factor reaches | built 2026-09-15 (kitchen fork `rotate`); on the block-49 capture Sol's INT8 term is plain 0.0265, balanced 0.0193, rotated 0.0140, both 0.0126, below kitchen's rotated dense kernel's 0.0166; block 0 improves slightly ([`bench/results/2026-09-15_sol_rotate_b{49,0}_s15.json`](../../bench/results/2026-09-15_sol_rotate_b49_s15.json)). About fifteen percent of a Sol call as built (serial per-row Hadamard; a warp-per-row form is the optimization). The default since 2026-09-27 |
| 3 | a sensitivity-ranked bake for the linears: capture one step's linear inputs across all fifty blocks, score each linear's tolerance for 4 bits with kitchen's own quantizers, bake a mixed checkpoint with the tail protected on evidence | not started |

How Tier 2 sits in the kernel: the rotation (kitchen's `apply_convrot_sign128` and `convrot128` from
`quant_qk_int8.cu`: a fixed sign diagonal, then a normalized Walsh-Hadamard over the 128 channels) is applied in
`quant_q_rows`, `quant_k_rows` (after the key-mean centring), `centroid_quant` and `prep_pooled_quant`, everywhere a row
becomes int8. The routing threshold, kmean, kcvar and the coarse branch stay unrotated, as `qk_balance` left them;
they are read against each other, never against an int8 carrier. The off path is byte-identical. What rotation and
the balance do to the route itself is in [`sol_error.md`](sol_error.md), "The route itself under INT8"
([`bench/results/2026-09-15_sol_route_b{49,0}_s15.json`](../../bench/results/2026-09-15_sol_route_b49_s15.json); cost
in `bench/results/2026-09-15_sol_timing_b49_s15.json`). With every lever plus Sol's rotation, token routing gave the
best Sol number on block 49, and token routing without the balance still hurt it
([`bench/results/2026-09-15_sol_token_aug_x_options_b49_s15.json`](../../bench/results/2026-09-15_sol_token_aug_x_options_b49_s15.json)).

## The public pack

The channel-balance node and the checkpoint scan are published as
[ComfyUI-H3-Quant](https://github.com/fblissjr/ComfyUI-H3-Quant), from this repo's `standalone/h3_quant`; this pack
defers to it when installed. The kernel forms go upstream through comfy-kitchen, not through it.
