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
| reorder3d: tau 1.0 against plain order | BEST of the three (owner, 2026-09-17) | clean; none of the default's morphing |
| reorder3d: tau 1.3 against plain order | second | better than plain order; a man in the left stall visible moving around, and possibly a little less detail |
| sage rotated against sage balanced | | |


### Owner's verdict on the reorder stack, 2026-09-17

The 3d reorder at tau 1.0 looks best, the reorder at tau 1.3 second, and the
plain-order default last by a distance: "an artifacted morphing mess", with a
figure at the bottom centre-right near the radio around the four second mark
that starts dissolving through the five second mark and is gone by six, and
music that "sounds creepy as hell like a broken radio".

Checked the same evening on frames at 3.5, 4.5, 5.5 and 6.5 s from four clips
of this scene and seed: the phantom figure at the bottom right around 4.5 s is
in BOTH plain-order default renders, the one made with the memory compiler off
for this stack and the 2026-09-15 default render made with it on
(`Video/block49_repro/noodle_bar_default_s730451892`), and the two are the same
composition frame for frame. It is in neither reorder clip. So it belongs to
the default chain with plain token order, not to the compiler flag, and the
2026-09-15 clip that was judged to have nothing obviously wrong has it too.
This is the first case in this pack where a Sol change was visibly better and
the capture metric had said so beforehand (`2026-09-17_sol_orderings.md`).
Audio was not checked by instrument; the prompt does ask for a thin, tinny
radio, so part of that may be intended. One scene, one seed.

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

### Owner's verdict on `stack_noodlebar_sage_balanced_vs_rotated`, 2026-09-17 evening

All three rows have the clone / morphing around the four second mark: the
2026-09-15 default on the kitchen chain (top, rendered before any of this day's
changes, and with the stripped prompt), sage balanced and sage rotated. The
owner had separately seen "two of him" in the token-routing arms and the sage
chain clip, and ranked the two reorder clips clean. Taken together: the
duplicate figure is a property of this scene and seed under PLAIN token order,
on both dense chains, on both prompt variants, and it predates 2026-09-17. The
only arms without it are the two with the 3d reorder. Rotation in sage neither
causes nor cures it. One scene, one seed; a second scene is the next test.

### Owner's verdict on `stack_noodlebar_default_vs_tokens_measured`, 2026-09-17 evening

Both rows have the man morphing in at the bottom of the frame and out again:
the 2026-09-15 default (top) and token routing on the four measured blocks
(bottom). So token routing on blocks 0, 24, 32 and 40 does not remove the
artifact. It is the same artifact as in every other plain-order clip of this
scene; see the verdict above. Token routing recovers individual tokens inside
blocks that were already routed badly or well; it does not change which tokens
share a block, which is what the reorder changes.

### Owner's verdict on `stack_noodlebar_default_vs_rotate`, 2026-09-17 evening

Both rows morph. The top row, the 2026-09-15 default, is the worse of the two:
"wild", the shape turns into a person who disappears between the five and six
second marks. Sol `rotate` on (bottom) still has the artifact. So rotation in
Sol, like rotation in sage and token routing, does not remove it; only the
reorder clips are clean. Note the two rows are different samples (the prompt
newline correction above), so "worse" here is about two samples of one scene,
not about what `rotate` did.

### Owner's verdict on `stack_noodlebar_default_vs_sage_chain`, 2026-09-17 evening

Both rows have the problem; the bottom row, the generated sage chain (sage
balanced under Sol), is the worse one because it also clones the man. Same
caution as the rotate stack: the rows are different samples, so this is not a
clean kitchen-against-sage comparison. What it adds to the tally is that the
sage chain in plain token order morphs too.

Tally across the owner's verdicts this evening, noodle bar, seed 730451892:
every plain-order arm morphs (2026-09-15 default, today's default, Sol rotate,
sage balanced, sage rotated, token routing on the measured blocks, the sage
chain); the two 3d reorder arms do not. `tokens_early_middle` and
`tokens_all_rotate` were reported as "two of him" as well.

