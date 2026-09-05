# Plan: token routing (`token_aug`) from graded lever to shippable per-block policy

last updated: 2026-09-05

**This file owns the sequence for roadmap step 7 after the 2026-09-04 grade**,
from a scratch wheel that nobody runs to a per-block setting that either
survives the decision standard or is closed with a record saying why. It
states, per stage, what is done, what it costs, what it produces, what would
count, and what would stop it. It restates no measurement:
[`2026-09-04_sol_token_aug_grade.md`](2026-09-04_sol_token_aug_grade.md) owns
the numbers that put this lane on the roadmap, [`../SOLATTN.md`](../SOLATTN.md)
owns the knobs, [`../roadmap.md`](../roadmap.md) owns the decision standard
(the 2026-09-03 plan's "The decision standard" and "What would count as
finding it"), and [`../eval_comparison.md`](../eval_comparison.md) section 3
owns how a pair is judged. Nothing below starts until the owner says go; the
card is shared and every stage that touches it says so.

## What is already established, and what this plan takes as given

- Kijai's token routing (Comfy-Org/comfy-kitchen PR 156, head `1128df6` on
  2026-09-04) lowers Sol's error against exact attention on four of the five
  captured blocks at every captured step and raises it on block 49 at every
  step; the budget knob is nearly inert. Grade record and reading in the
  2026-09-04 note. So the lever is per block, and the block set is the
  question.
- Our three `blk_cnt` commits (`cc81f55`, `24908e1`, `b292c74` on the
  workspace clone's `sol-blk-cnt`) do not cherry-pick onto that branch;
  four files conflict. The probe (`sol_block_probe.py`) needs `blk_cnt`, so a
  wheel that can be probed with token routing on is a rebase, not a build.
- The owner's rules of 2026-09-04: sage is the floor every comparison is
  judged against; the routing policy is the lane; no more seeds; a kernel
  build enters the venv only through `vendor/rebuild_kernel.sh` from a
  known sha; any node change means a restart before a render counts.
- The upstream PR is open and its review flagged the fractional-budget
  handling; its head may move. Every stage below pins the sha it used, and a
  moved head means the grade is redone before anything downstream is trusted.

## Stage 1: the rebase (no card)

**Do.** In a detached worktree of the workspace clone at `1128df6`, replay
the three `blk_cnt` commits by hand, resolving the four conflicting files:
`comfy_kitchen/__init__.py`, `comfy_kitchen/backends/cuda/__init__.py`,
`comfy_kitchen/backends/eager/sol_attn.py`, `comfy_kitchen/backends/hip/__init__.py`.
Both knobs thread through `sol_attn` and `sol_attn_chunked`; the HIP side
keeps signature parity and warns for `token_aug` as upstream already does.

**One design decision, taken here rather than discovered later.** With
token routing on, the routed block count no longer describes everything a
query attended: the selected tokens outside the routed blocks are attended
exactly too. `blk_cnt` keeps its meaning (routed blocks per query block,
what the probe's replay and `bench/measure_sol_exact_variants.py`'s
forced-pairs check already assert) and gains nothing; a second optional
out-parameter for the admitted token count is the honest addition if the
probe needs a keep fraction under routing, and it is written only if Stage 4
asks for it. Do not fold the two into one number.

**Produces.** A local branch in the workspace clone, `sol-blk-cnt-token-aug`,
its tip recorded in the changelog by sha. Pushing it to the owner's fork is
the owner's call, as every push is.

**Counts as done when** the branch builds and kitchen's own
`tests/test_sol_attn.py` passes on it, our `blk_cnt` cases and upstream's
`token_aug` cases together. That test run needs the card for minutes and is
the one card touch in this stage; run it with the server down.

**Stops if** the conflicts cannot be resolved without changing the routed
count's semantics, or upstream's tests fail on the merged tree for a reason
the rebase introduced.

## Stage 2: reproduce the grade with telemetry present (minutes of card)

**Do.** Build the Stage 1 tip into a scratch wheel, version tagged
`+sol.<sha>`, unzipped into a scratch directory and selected through
`PYTHONPATH`; the venv stays untouched. Run
`bench/measure_sol_exact_variants.py --capture` on the retained Base16 cells
with `--token-aug 64` and `--kernel-source` naming the branch and sha; then
the random-mode run for `blk_cnt_forced_pairs` and the isolated timing at the
capture's shape, both budgets.

**Produces.** `bench/results/<date>_sol_exact_base16_capture_<sha>_token_aug.json`
and its timing twin, beside the 2026-09-04 records.

**Counts as done when** every aggregate reproduces the `1128df6` record to
print precision (the rebase changed no arithmetic; this is the control) and
`blk_cnt_forced_pairs` reads ok on the same build. Both together, or neither
counts.

**Stops if** any aggregate moves. Then the rebase changed numerics, and
Stage 1 is reopened before anything else; nothing is installed.

## Stage 3: the node knob (node code, restart before it counts)

**Do.** `MiniMaxH3SolAttn` gains a per-block spec input for token routing,
a string in the `block_spec.py` grammar `dense_blocks` and `tau_profile`
already use (for example `0-40=64`), empty by default, budgets validated as
multiples of 64 up to 256 at the node so a bad value fails at graph time.
The kernel call passes `token_aug` only for blocks in the spec. A string
spec rather than a numeric widget: an empty string is "off" without a
numeric sentinel, which is the rule `bench/check_literal_widgets.py`
enforces. `h3_config.SOL_CUDA_DEFAULTS` pins the new key empty, since
`bench/check_sol_kernel.py` requires every pinned key to be a declared input
and every declared knob to be pinned or listed as deliberately unpinned.
The route observer and the provenance stamp record the spec so a render
says which blocks routed tokens.

**Install path.** `SRC=<clean worktree at the Stage 1 sha> vendor/rebuild_kernel.sh 89`,
which tags the version, keeps the wheel under the clone's `dist/`, writes
`.venv/comfy_kitchen_build.json`, and runs `bench/check_sol_kernel.py --require`;
then a server restart by the port-owner recipe, and the new input read back
from `/object_info` before any render. `bench/check_attention_defaults.py`
and `bench/check_literal_widgets.py` run before the commit; the checks page
gains no new check unless one of these lets an instance through.

**Counts as done when** the shipped graphs rebuild byte-identical apart from
the new empty input (the default is off), and a graph with the spec set on
one block renders and stamps it.

**Stops if** the kitchen API for `token_aug` changes shape before the PR
merges in a way the node cannot express; then the node waits for the merge
rather than tracking a moving branch.

## Stage 4: where it helps, by the probe (hours of card, armed server)

**Do.** `sol_block_probe.py` on the shipped call with sink ranges, on the
standoff and subway scenes the lane already probes, token routing off and
then on for all fifty blocks at one budget, matched seed, each an armed
restart the operator owns. The probe's per-block Sol-versus-sage
disagreement is the ranking; the offline grade already predicts the
direction on five blocks (better on 0, 24, 32, 40; worse on 49) and the
probe either agrees on the shipped call or it does not.

**Produces.** Two probe records per scene under `bench/results/`, and a
candidate block set: the blocks whose disagreement drops with routing on,
never a block where it rises. If the probe disagrees with the offline grade
on block 49, that is a finding about the sink ranges and is recorded before
anything else proceeds.

**Counts as done when** the set is named with its per-block evidence and
its cost from the timing records, and the roadmap's step 4 (block policy)
lists it as one candidate beside `dense_blocks` and `tau_profile`.

**Stops if** no block improves on the shipped call. Then token routing is
closed here with the records, and the upstream PR is left to upstream.

## Stage 5: one blind pair per candidate set (one render pair each)

**Do.** The shipped Sol graph against itself with the Stage 4 block set,
sage on in both (the floor), matched seed, one scene where the probe
predicts the largest gain and one where Sol lost blind (the subway chase),
rendered through `bench/run_graph_arms.py`, blinded through
`bench/blind_batch.py` with a regenerated score page, scored before
unblinding, joined by `bench/score_session.py`. Singles scored for audio,
since audio is the first casualty the ladder found and the PDD lesson is
that unscored singles leave the audio verdict verbal. Kijai's own named
symptom for the defect routing addresses is a brightness pulse on a
five-latent-frame period; the pair's free-text field asks for it by name,
and a per-frame mean-luminance periodogram at that period is a cheap
instrument to add beside the loudness tool if the owner wants an objective
line next to the verdict.

**Counts as done when** the pair does not prefer the shipped call on the
scene where the probe predicted the gain, and the timing record prices the
set. One seed anchors; it does not decide.

**Stops if** the pair prefers the shipped call, or the pulse the routing is
meant to remove is not present in our renders at all, in which case the
mechanism the PR fixes may not be one we have.

## Stage 6: ship or close

**Ship** means: the block set becomes the default in `SOL_CUDA_DEFAULTS`
through the generator, the wheel is the one the build record names, the
SOLATTN knob table gains the row, `docs/evidence.md` gains a bounded row
pointing at the probe and pair records, and the changelog names the sha.
The reversal condition is written beside the default: a blind session where
the set loses to the shipped call on a scene class the owner cares about.

**Close** means: the records stay, the branch stays in the clone, the node
knob stays empty by default or is removed, and roadmap step 7 says why with
a pointer.

## Costs, in one place

| stage | card | server | node code | reversible |
|---|---|---|---|---|
| 1 rebase | minutes, tests only | down | no | yes, scratch branch |
| 2 reproduce | minutes | down | no | yes, scratch wheel |
| 3 knob | none until install; install is a rebuild and restart | restart | yes | yes, rebuild the previous sha |
| 4 probe | hours | armed restart, timings void | no | yes |
| 5 pair | one render pair per candidate | unarmed | no | yes |
| 6 ship | none | restart | defaults only | yes, the previous default |

## Not decided by this plan

- Whether a block set found on t2va transfers to Base20, PDD8, FL2VA or
  Ref2VA; each is its own population under the 2026-09-03 plan.
- What happens if upstream merges a different `token_aug` than the one
  graded; the sha pins every stage and a moved head restarts at Stage 2.
- Whether the routed-token count should feed a keep-fraction reading in the
  probe; Stage 1 leaves the seam and Stage 4 decides.
