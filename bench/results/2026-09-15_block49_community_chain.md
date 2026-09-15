# The community chain on the market prompt: kitchen int8 attention dense + Sol (2026-09-15)

The chain most H3 users run: ComfyUI core's Model Attention Backend on
"comfy kitchen attention" (kitchen's `int8_attention`, which rotates q/k
before INT8 and is immune to the loud-channel mechanism,
`2026-09-15_ck_int8_attention_block49.json`) as the dense kernel, Sol on
the routed steps, sage absent. Our Sol node stands in for core's
block-sparse node on the same kernel. Graphs `h3_probe_t2v_ck*_api.json`
(generator `dense_attn="ck"`), market prompt, seed 730451892, the seed of
every other market arm today. Served wheel `0.2.34+sol.5284cfb`.

Server log for these renders (`user/comfyui_8188.log`): "[h3-sol] chaining
onto an existing attention override" (the backend node's), "chain assert,
call-time: no sage: probes ... reached no sage kernel", and for the third
arm "[h3-sol] keeping blocks [45, 48, 49] dense of 50".

| clip | arm | what carries the block-49 mechanism | wall time |
|---|---|---|---|
| ck | as most people run it | Sol's routed steps only; the dense steps are on the rotated kernel | 500 s |
| ck_balanced | + Sol `qk_balance` | nothing unrotated and unbalanced | 500 s |
| ck_dense_tail | + blocks 45/48/49 handed to the dense kernel (`dense_blocks`) | nothing at those blocks; Sol elsewhere | 516 s |

For comparison, our own chain on the same prompt and seed: sage-dense
default 508 s, all levers 511 s, levers plus bf16 tail 569 s. The dense tail
costs 16 s here against 60 s there, because this chain's fallback is the
INT8 rotated kernel, not bf16.

## Outputs

`Video/h3_probe_t2v_ck_00001-audio.mp4`, `Video/h3_probe_t2v_ck_balanced_00001-audio.mp4`,
`Video/h3_probe_t2v_ck_dense_tail_00001-audio.mp4`; captioned stack
`Video/block49_eval/stack_market_community_chain.mp4`. Prompt ids:
- `h3_probe_t2v_ck`: `0e6a1247-6e2c-4d5f-b51b-e43169d4ab26`
- `h3_probe_t2v_ck_balanced`: `2ad0ed48-5f4d-4359-8276-3012c1bf42a5`
- `h3_probe_t2v_ck_dense_tail`: `b4eceff6-99eb-40e0-af67-3c4ae6db872b`

## What it decides

Against `h3_t2v_00019` (our sage-dense default, the porter morphs) and
`h3_probe_t2v_levers_00001` (our chain with every lever on): if the plain
community chain already looks like our levers arm, the dense half of our
problem was our Sage node and nobody else's; if it morphs like our default,
the routed steps carry it visibly for everyone on block-sparse attention
and the Sol balance or the dense tail is worth shipping to them.

## Owner's scoring

(unfilled)

## The controls, 2026-09-15 evening: the coins from nowhere are the kitchen dense kernel's

The owner: "in all 3 of those clips - ck, ck balanced, and ck dense tail,
the coins fall from nowhere in all 3." Frames at 0.5 s spacing from 6 to
11.5 s, same prompt and seed (730451892):

| arm | dense-step kernel | routed steps | the coin beat | wall time |
|---|---|---|---|---|
| ck, ck_balanced, ck_dense_tail | kitchen `int8_attention` (rotated INT8) | Sol | both hands stay on the crate from 7.5 to 9.0 s; no hand goes near the tins; coins appear anyway | 500 / 500 / 516 s |
| `h3_probe_t2v_dense` (new) | ComfyUI's own bf16 attention | none (no Sol) | a hand reaches over at 7.5 s, is over the open tin at 8.0 and 8.5 s, withdraws at 9.0 s | 1785 s |
| `h3_probe_t2v_sol_nosage` | ComfyUI's own bf16 attention | Sol | the same hand, the same beat, a near-identical take to the fully dense arm | 742 s |
| the sage arms of the morning | sage fp8++ | Sol | a hand drops coins (small and fast in the levers clip, a stack in the policy clip) | 508 / 511 / 569 s |
| `h3_probe_t2v_sage_rotate` (owner's ask) | sage fp8++ balanced + balance node | Sol balanced AND rotated | a hand over the open tin at 8.0 and 8.5 s | 508 s |

Reading. The action is decided on the dense steps (the first fifth of the
schedule): the three kitchen-dense arms share the flaw and diverge only
later, and the two bf16-dense arms are almost one take. With Sol present
in one bf16 arm and absent in the other and the hand present in both, Sol
is cleared. The dense-step kernel is the variable, and the kitchen rotated
INT8 kernel is the one that loses the hand. One seed; it needs a second
before it is more than a strong hint. It is also the first sign that the
kernel with the lowest measured block-49 error at the last step has a
visible cost of its own, which no grade so far could see: every grade is
at step 15, and composition is set at steps 4 to 8, where kitchen's INT8
P and V may not behave like sage's fp8 ones. That grade is next.

Costs from the same table: bf16 on the dense steps only costs about half
again (742 s); bf16 everywhere three and a half times (1785 s).

**The early-step grade, same night** (`bench/results/2026-09-15_dense_kernels_by_step.json`):
kitchen's dense kernel has the LOWER error at every captured cell, step 4
included. The missing hand is not explained by per-call error against fp32
attention; the instrument that ranks the kernels cannot see whatever the
clips show. One seed of clips against a lower number everywhere. What
would settle it: the same pair (kitchen dense + Sol balanced against sage
with every lever) on a second seed and a second scene, before the default
moves again in either direction.
