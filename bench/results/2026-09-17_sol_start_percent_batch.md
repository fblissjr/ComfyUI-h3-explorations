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

Scoring: owner, by eye, 2026-09-17.

| scene | 0.2 | 0.1 | 0.0 |
|---|---|---|---|
| market, seed 730451892 | coins come from nowhere (the known flaw of this seed on this chain) | the stallholder drops the coins; in shot 3 the porter walks TOWARD the camera | the stallholder drops the coins; in shot 3 the porter walks BACKWARDS |
| post office | camera at a three-quarter angle to the window | the same angle | head-on, centred on the cat; nothing wrong with it |

Read against the prompt: shot 3 asks for a static wide as he "carries both
crates away between the stalls", so walking away is the instruction; toward
the camera at 0.1 is a staging change, walking backwards at 0.0 is a motion
defect. The coin sentence names no agent ("coins clatter one after another
into a metal tin"), and the porter's hands are on a crate, so the stallholder
dropping them is a fair reading and "from nowhere" is partly the prompt's
doing. The owner's call: this seed and scene are a poor judge, use another.
Post office against its prompt: it asks for a medium shot of the front window,
the letter slot at the left of frame, a stone step below the glass, a static
camera and the cat on the inner sill, and names no angle. All three clips have
all of that (checked on a frame at six seconds); 0.0 differs only in being
frontal and flatter.

Two scenes, one pattern worth testing rather than believing: with no dense
steps the staging drifts (a frontal composition, a porter walking backwards),
while 0.1 stays close to 0.2.
Follow-up arms: `2026-09-17_sol_start_percent_arms_2.jsonl`.

## Second batch, 2026-09-17: a second market seed and the noodle bar

Rows: `2026-09-17_sol_start_percent_arms_2.jsonl`. The 0.2 arm of each scene is
the default-chain clip the owner already judged in the block-49 reproduction
batch (`Video/block49_repro/`, same chain and recipe, rendered 2026-09-15 on
the previous kitchen wheel; the default path did not change between the two
wheels). The first row's total includes time queued behind other renders, so
read sampler seconds.

| scene | seed | start_percent | sampler s |
|---|---|---|---|
| market | 20260915 | 0.1 | 419 |
| market | 20260915 | 0.0 | 369 |
| noodle bar | 730451892 | 0.1 | 418 |
| noodle bar | 730451892 | 0.0 | 365 |

Stacks: `Video/sol_start_percent/stacks/stack_market_s20260915_start_percent.mp4`
and `stack_noodlebar_start_percent.mp4`. Unscored.

| scene | 0.2 | 0.1 | 0.0 |
|---|---|---|---|
| market, seed 20260915 | | | |
| noodle bar | | | |

## Correction, 2026-09-17 evening: the reference clip in these stacks is a different sample

Found while checking the owner's report that today's clips morph around the
four second mark. The runner scripts for the 2026-09-17 batches read the prompt
file with its trailing newline; the 2026-09-15 reproduction batch had passed it
stripped. One character changes the conditioning, so **every noodle bar and post
office clip rendered on 2026-09-17 is a different sample from the 2026-09-15
default of the same scene and seed** (about 20 dB PSNR apart, as far as two
unrelated arms are). Consequences:

- Stacks whose top row is `Video/block49_repro/noodle_bar_default_s730451892`
  compare across samples and cannot be read as "what the option changed":
  the five `stack_noodlebar_default_vs_*` stacks, the top row of
  `stack_noodlebar_sage_balanced_vs_rotated`, and the top row of
  `stack_noodlebar_start_percent`.
- Comparisons WITHIN one day's batch stand, since every arm carries the same
  prompt: the reorder stack (its own default on top), sage balanced against
  sage rotated, the post office start_percent stack, and both market stacks
  (the market prompt comes from the graph, not from a file).
- Not a regression in any repository: the market default and the market rotate
  clips rendered on 2026-09-17 at 10:56, 14:54 and 15:03 are bit-identical
  (PSNR infinite) to the 2026-09-15 renders of the same graphs and seed, and
  renders repeat exactly across server restarts.
