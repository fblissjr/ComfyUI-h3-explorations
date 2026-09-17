# Sol options on a scene already judged clean: rotate, token routing, and the generated sage chain

Date: 2026-09-17. Noodle bar scene (`prompt_bank/t2va_noodle_bar.txt`), seed
730451892, base 16-step t2v at full length, `workflows/h3_text_to_video_api.json`
with one thing changed per arm by `bench/run_graph_arms.py`; the sage arm is the
graph `build_workflows.py --chain sage` writes. comfy-kitchen 0.2.34+sol.b532e28.
Rows: `2026-09-17_sol_options_noodlebar_arms.jsonl`. Reference clip: the
default-chain render of the same scene and seed from the block-49 reproduction
batch, which the owner judged to have nothing wrong.

| arm | what differs from the default chain | sampler s |
|---|---|---|
| rotate | Sol `rotate` on | 480 |
| tokens_measured | `token_routing` = measured blocks (0, 24, 32, 40) | 467 |
| tokens_early_middle | `token_routing` = every block but the last five | 479 |
| tokens_all_rotate | `token_routing` = all blocks, with `rotate` on | 495 |
| sage_chain | sage `fp8++ balanced` under Sol, no kitchen dense node | 459 |

For scale, the same graph unchanged samples in about 468 s
(`2026-09-17_market_prompt_fix_and_rotate_timing.jsonl`, the rotate on/off
pair, which put rotate's cost at about 15 s on the serial kernel).

Clips: `Video/sol_options_noodlebar/`. Stacks, default on top:
`Video/sol_options_noodlebar/stacks/stack_noodlebar_default_vs_<arm>.mp4`.
These are the first clips with token routing on that anyone can judge, and the
first render of a generated sage-chain graph (it rendered; until now that set
had only passed schema validation).

Scoring: owner, by eye. Unscored.

| arm | against the default clip | notes |
|---|---|---|
| rotate | | |
| tokens_measured | | |
| tokens_early_middle | | |
| tokens_all_rotate | | |
| sage_chain | | |

## Second round, same day: the token reorder, rotate on the ported kernel, sage in its rotated mode

Rows: `2026-09-17_sol_options_noodlebar_arms_2.jsonl` (normal server) and
`2026-09-17_sol_reorder_noodlebar_arms.jsonl` (server started with
`--disable-comfy-compiler`, because the reorder and ComfyUI's memory compiler do
not work together yet; CHANGELOG 0.122.1). Kitchen wheel 0.2.34+sol.36e29f1,
sage fork v0.7.20. The three reorder-round clips share that flag, so they
compare with each other and not with the clips above.

| arm | what differs | sampler s |
|---|---|---|
| rotate_fastkernel | Sol `rotate` on, the ported in-register kernel (first render after a restart) | 466 |
| sage_chain_rotated | sage `fp8++ rotated` under Sol | 460 |
| nocompiler_default | default chain, memory compiler off | 466 |
| nocompiler_reorder3d_tau10 | `morton` on, curve `3d`, tau 1.0 | 465 |
| nocompiler_reorder3d_tau13 | `morton` on, curve `3d`, tau 1.3 | 436 |

Three timing readings. `rotate` now costs nothing at render level (466 s against
the default's 468 s; it was about 15 s on the serial kernel). Turning the memory
compiler off did not slow the default render. The reorder itself is free, and
at tau 1.3 it takes about 30 s off the sampling, which is the speed side of the
capture result (`2026-09-17_sol_orderings.md`): fewer routed blocks for about
the same measured error.

Stacks: `stack_noodlebar_reorder3d.mp4` (plain order, reorder at tau 1.0,
reorder at tau 1.3) and `stack_noodlebar_sage_balanced_vs_rotated.mp4` (default
chain, sage balanced, sage rotated). The rotate_fastkernel clip needs no stack:
the ported kernel is bit-identical to the one that rendered the `rotate` clip
above.

| stack | verdict | notes |
|---|---|---|
| reorder3d: tau 1.0 against plain order | | |
| reorder3d: tau 1.3 against plain order | | |
| sage rotated against sage balanced | | |
