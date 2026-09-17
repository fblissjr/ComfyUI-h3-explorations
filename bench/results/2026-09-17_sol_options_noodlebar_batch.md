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
