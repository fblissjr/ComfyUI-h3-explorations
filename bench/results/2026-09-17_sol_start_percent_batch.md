# start_percent on the default chain: 0.2 against 0.1 against 0.0

Date: 2026-09-17. Model: MiniMax H3, int8 convrot checkpoint, base 16-step
t2v, 345 frames at 1344x768. Default chain (kitchen int8 dense under Sol,
`qk_balance` on), graph `workflows/h3_text_to_video_api.json` with only
`MiniMaxH3SolAttn.start_percent` (and the prompt, for the second scene)
patched by `bench/run_graph_arms.py`. Seed 730451892 on every render.
comfy-kitchen 0.2.34+sol.b532e28, pack at 7e44b2d, RTX 4090.

Why: `start_percent` had never been measured at any value
(`docs/SOLATTN.md`), and `2026-09-17_sol_stage_profile.md` priced it: the
four dense steps it forces cost more attention time than the twelve Sol
steps. At shift 12 on 16 steps, 0.2 leaves four steps dense, 0.1 two, 0.0 none.

Timing rows: `2026-09-17_sol_start_percent_arms.jsonl` (five renders; the
market 0.2 arm is the same morning's baseline,
`2026-09-17_default_chain_walltime_baseline.jsonl`).

| scene | start_percent | dense steps | sampler s | total s |
|---|---|---|---|---|
| market | 0.2 | 4 | (baseline row: total only) | 501 |
| market | 0.1 | 2 | 419 | 460 |
| market | 0.0 | 0 | 368 | 402 |
| post office | 0.2 | 4 | 459 | 496 |
| post office | 0.1 | 2 | 407 | 441 |
| post office | 0.0 | 0 | 357 | 391 |

Each pair of dense steps turned sparse is 50 to 58 s, which is what the live
per-call times predicted (about 25 s per step). 0.0 is a fifth off the render.

Clips: `Video/sol_start_percent/<scene>_sp<value>_s730451892_*-audio.mp4`.
Stacks for scoring, 0.2 on top, 0.1 in the middle, 0.0 at the bottom:
`Video/sol_start_percent/stacks/stack_market_start_percent.mp4` and
`stack_postoffice_start_percent.mp4`.

What to look for: the first steps set global composition, so the risk is in
layout, subject count and framing rather than texture. Compare shot
composition, whether people and objects are where the prompt puts them, and
the first second of each shot.

Scoring: owner, by eye. Unscored at the time of writing.

| scene | 0.1 against 0.2 | 0.0 against 0.2 | notes |
|---|---|---|---|
| market | | | |
| post office | | | |
