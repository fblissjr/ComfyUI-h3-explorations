# Block-49 reproduction batch: default chain vs sage levers chain

Date: 2026-09-15. Server: ComfyUI on the shared :8188, kitchen wheel
0.2.34+sol.757657f, sage fork v0.7.19, h3 pack at 9d79a1b. Model: MiniMax
H3 int8 convrot. One seed per pair, same prompt file per scene.

Question: on the market seed every arm using kitchen's rotated INT8 dense
kernel lost the hand on the coin drop while sage and bf16 arms kept it,
although every capture instrument grades kitchen's kernel lower error
(`2026-09-15_dense_kernels_*.json`). Does that split reproduce on other
scenes and a second market seed?

Arms:

- default: the shipped chain after 28d8ee5 (kitchen int8 dense backend via
  Model Attention Backend + Sol qk_balance), graph `h3_text_to_video`.
- sagelevers: sage fp8++ balanced + MiniMaxH3ChannelBalance fold + Sol
  qk_balance, graph `h3_probe_t2v_levers`.

Outputs: `Video/block49_repro/<scene>_<arm>_s<seed>_00001-audio.mp4`; the
kitchen scene's default arm is the earlier `Video/block49_kitchen/default_s730451892_00001-audio.mp4`
(same chain, rendered in the kitchen batch). Side-by-side stacks for scoring:
`Video/block49_repro/stacks/stack_<scene>_default_vs_sagelevers.mp4`
(built by `bench/stack_labeled_clips.py`, default on top).

Wall time (seconds, execution_start to execution_success from server history):

| scene | seed | default | sagelevers |
|---|---|---|---|
| diner | 730451892 | 501 | 492 |
| kitchen | 730451892 | (kitchen batch) | 486 |
| market | 20260915 | 501 | 494 |
| hardware_aisle_short | 730451892 | 500 | 489 |
| post_office | 730451892 | 493 | 482 |
| noodle_bar | 730451892 | 497 | 487 |

All eleven succeeded. The sage levers chain is consistently a few seconds
faster than the kitchen dense chain on this batch, opposite to the market
seed-730451892 timing (`docs/h3_quant_policy.md`, wall-time table), so
the ordering of the two chains is within run-to-run noise.

Scoring: owner, by eye, per stack. Unscored at the time of writing.
Fill in below.

| scene | default keeps the hand on object drops | sagelevers keeps it | notes |
|---|---|---|---|
| diner | | | |
| kitchen | | | |
| market s20260915 | | | |
| hardware_aisle_short | | | |
| post_office | | | |
| noodle_bar | | | |
