# Historical Forward Plans (2026-09-03 to 2026-09-05)

> **Archived from [`docs/roadmap.md`](../../roadmap.md) on 2026-10-27.**
> For the current day boundary and active queue, see [`docs/wiki/handover.md`](../handover.md) and [`docs/wiki/next_steps.md`](../next_steps.md). For active closed lanes and rails, see [`docs/roadmap.md`](../../roadmap.md).

## Current forward plan — 2026-09-04

Supersedes the 2026-09-03 plan below in sequence only; that plan's decision
standard and its "what would count as finding it" stand unchanged and are
not restated here.

**Two owner decisions on 2026-09-04, after reading the ladder verdict**
(`data/results/2026-09-04_ladder_2026-09-03_verdict.json`):

- **Sage is always on.** The ladder could not tell sage alone from the true
  baseline on four scenes of five at one seed and read it narrowly behind
  on the fast-motion scene, at a fraction of the dense cost (the ladder
  record carries both). So the floor every comparison is judged against is
  the sage render, not the dense one. The dense stamped arm stays as the
  statement of what the model itself does, rendered when that question is
  asked, never as a rung of every ladder. What would reverse this: a blind
  multi-scene session where sage loses to dense on a scene class the owner
  cares about.
- **The routing policy is the core priority.** Where Sol runs, which steps,
  which blocks, which rows, is what decides quality against cost; the
  kernel's arithmetic is the smaller term through the trunk
  (`docs/SOLATTN.md`, the decomposition section), and a global tau is the
  blunt hammer the owner has already found degrades quality as it rises.
  The instrument is the online probe (`sol_block_probe.py`), which measures
  Sol against the sage fallback on identical inputs, and under the first
  decision that fallback is exactly the floor.

**What the two decisions change:**

- The PDD ladder (`bench/pdd_ladder_arms.json`) drops its stock-attention
  control and pairs against the ladder's sage clips: three rungs, all under
  sage. The stock-attention PDD8 graph stays as the vendor's reference
  configuration and is not a rung.
- The fp16-sage rung is dropped from further ladders: slower than Sol as
  shipped on every scene and no quality signal at one seed.
- The offline Sol-versus-exact records stay as kernel-fidelity work and as
  the footing for grading upstream kernel changes (token_aug is the open
  one); they are not the policy instrument.
- More seeds are not the next step; the owner has ruled them out for now.
  One seed anchors; a probe record plus one blind pair per candidate is how
  a policy change earns its place, and the decision standard below is what
  it must pass to ship.

**The sequence, each step earning the next:**

1. **Probe the subway scene** on the shipped call, the one scene where Sol
   lost blind, and set it beside the standoff record
   (`data/results/2026-09-03_sol_probe_base16_standoff.json`). If the
   disagreement is larger there, the proxy predicts the eye and its ranking
   of blocks, rows and steps is what the next three steps act on; if not,
   the proxy is missing something and token_aug's DC bias is the first
   suspect. **Done 2026-09-04**
   (`data/results/2026-09-04_sol_probe_subway_vs_standoff.json`): same
   ranking, less disagreement on subway than on standoff. Neither branch
   above applies: the ladder's dense-versus-sage pair lost on the same
   subway defect, so the slip is the sage floor's, and Sol added less there
   than on the scene the owner could not tell apart. The proxy and the eye
   agree once the floor is read correctly; one seed per scene.
2. **The PDD ladder as rebuilt**: PDD8 under sage alone against the ladder's
   sage clip, and Sol on the PDD schedule at the shipped and a narrower
   window. The same policy question on the coarse schedule.
3. **Segment policy**: the text query rows dense, a `sink_conditioning`
   option (a node change, so a restart before it counts), measured on the
   probe first, then one blind pair. The standoff record has the text rows
   disagreeing most in every block and they are a few hundred rows in a
   hundred thousand. **Built and probed 2026-09-04** (`exact_kv_and_all_rows`,
   changelog 0.99.40 and 0.99.42;
   `data/results/2026-09-04_sol_probe_subway_allrows_vs_shipped.json`):
   the text segment collapses to the audio floor, video and audio move by
   nothing the record resolves, ranking unchanged. At the pixel level the
   mode moves the subway render about as far as dense-versus-sage does
   (`data/results/2026-09-04_probe_allrows_vs_shipped_pixels.json`), so
   the blind pair is meaningful, and it costs no card time: the armed
   render's clip is a valid sample
   (`data/results/2026-09-04_probe_render_vs_unarmed_pixels.json`). Owed,
   as a one-pair session against the 2026-09-03 subway Sol clip; until it
   is judged this is free on the proxy and unproven to the eye.
   **Owner, 2026-09-10: plan the switch as the next step.** The judgement
   is now pair C of `bench/sol_core_ab_arms.json` (three new scenes, two
   seeds, ours as shipped against the all-rows candidate). If all-rows
   loses on no scene, flip `h3_config.SOL_RECOMMENDED_CUDA`'s
   `sink_conditioning` to `exact_kv_and_all_rows` for t2va and fl2va and
   rebuild; ref2va keeps `exact_kv_and_rows`, because with a video
   reference all-rows runs the reference's rows dense, until the Sana-style
   segment permutation (text and audio dense, references sparse;
   `docs/SOLATTN.md`, "What Sana's newer H3 packages offer this card") is
   built. Not flipped before that session renders, because its "ours as
   shipped" arm is defined as today's default. *(2026-10-02: flipped
   (0.185.0), for every graph, video references included, by owner decision
   on the output-level panel `bench/results/2026-10-02_sol_dense_blocks_panel.md`,
   where all-rows lost on no scene. The ref2va exception above was not kept:
   the owner asked for one general default.)* Two external corroborations
   beside the probe: Sana's Sol-H3 T2V `prefix` sink is this mode
   (`docs/sol_upstream.md`), and sglang's dropped "sink block" is a
   different mechanism, a reserved block inside a fixed budget, so it says
   nothing against it.
4. **Block policy**: a per-block tau profile (`tau_profile`, retired
   unused on 2026-09-27 with the old node) or a dense set on the band the
   probe ranks hottest, same footing.
5. **Step policy**: the window's start, since the per-step trend rises
   toward the end of the window, not the start. **A prior arrived
   2026-09-04**: sglang's cube sparse backend, co-authored by MiniMax
   engineers, publishes a per-update keep schedule for the fifty-step grid
   (dense for two updates, then a fast decay to a floor). Open experiment 27
   holds the record, the mapping onto our sigma grid, and the arm.
6. Then the other populations, legal PDD8 through the node's own sigmas and
   Ref2VA, each its own ranking, never pooled.
7. **token_aug**, graded offline first in the upstream-survey lane; if it
   holds on our capture it moves the frontier under every policy above and
   goes in through the vendoring rule. **Graded 2026-09-04**
   ([`research/2026-09-04_sol_token_aug_grade.md`](../../research/2026-09-04_sol_token_aug_grade.md)):
   it holds on four of the five captured blocks and degrades block 49 at
   every step, with a budget knob that barely moves anything, so it does
   not move the frontier under every policy; it is one more per-block
   candidate for step 4, to be ranked by the probe on the shipped call.
   **The plan from here is
   [`research/2026-09-05_token_aug_plan.md`](../../research/2026-09-05_token_aug_plan.md)**:
   six stages from the `blk_cnt` rebase to ship-or-close, each with its card
   cost, its record, what counts and what stops it; nothing in it starts
   before the owner's go.

   **Stages 1 and 3 are DONE 2026-09-08**, both off the card. Stage 1, the
   rebase, retargeted from kijai's PR head to v0.2.33 which merged it with an
   identical tree, so the 2026-09-04 grade transferred rather than being
   redone; installed as `0.2.33+sol.990ae4c`. Stage 3, the node knob, shipped
   as `token_aug_blocks` on `MiniMaxH3SolAttn`, per block, empty in both
   shared configs, budget validated at patch time, all 188 graphs regenerated.
   Records and controls in the plan page.

   **Stage 2 is the next step and it is the gate**, not stage 4: reproduce the
   2026-09-04 grade on the installed build, which is minutes of card on the
   retained Base16 cells and produces the control that says the rebase changed
   no arithmetic. Stage 4 is where the hours are, and it should not start
   until stage 2 says the numbers still reproduce.

**Not established after 2026-09-04, and must not be quoted as if it were:**
any scene on which Sol itself is visibly worse than the sage floor (the one
scene where Sol lost blind, the subway chase, lost the same way under sage
alone, so the proxy-to-perception link is untested in the direction that
matters); whether the all-rows sink mode is visible (its pixel footprint at
one seed is the size of dense-versus-sage, and the blind pair is rendered
but unjudged, step 3 above); just-Sol against the floor WAS judged 2026-09-05
(`data/results/2026-09-05_sol_nosage_2026-09-04_verdict.json`: the floor won four with one can't-tell and Sol as shipped won three
with two same, nearly all on composition, so the direction is against
Sol without sage and the evidence is weak); PDD8 under sage alone against the floor WAS judged
2026-09-05, pairs only, `data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`: the floor won every scene, so the PDD8
loss is PDD's own and not Sol's, and Sol on two of eight steps read
same as sage alone on all five); and
the size of the tau lever, which the owner reports as a quality loss without
a measured curve. Two things established on the way: arming the probe does
not perturb a render (`data/results/2026-09-04_probe_render_vs_unarmed_pixels.json`,
the armed subway render identical to the unarmed ladder clip on both
streams), so a probe render's clip is a valid blind sample of its
configuration; and the post-pull execution regime changed timing only
(`data/results/2026-09-04_stairwell_dense_retime_pixels.json`).

## Owner decisions, 2026-09-05 evening, after scoring three sessions

Recorded here the same hour; each supersedes what stood before it.

1. **Composition differences count as prompt adherence.** A half whose scene
   makes less sense for the prompt has adhered less, and that is a loss for
   its arm. `eval_comparison.md`'s judging section carries it; the three
   2026-09-05 verdict records were first read the other way and say so.
2. **PDD is parked.** PDD8 lost to the sage floor on every scene under sage
   alone, and the bake did not move it, so the look is the schedule's. No
   partition arm; "move on to something better."
3. **The turbo rung goes**, with two LoRAs: the larryvrh v4 step-600 EMA at
   six steps and the lightx2v fl2v turbo v1.2 768p at its own step count
   (`models/loras/h3/lightx2v_Minimax-h3-Turbo/`, the v1.2 file landed
   2026-09-05), each under sage alone on the five ladder scenes, blinded
   against the sage floor. Owned by the bake session, which proposed it.
4. **Token routing stage one goes** (`research/2026-09-05_token_aug_plan.md`,
   step 7 below). The survey session was offline when the go was given, so
   this line is the handoff.
5. **Two seeds**, superseding "no more seeds" of 2026-09-04: the turbo rung
   renders at the ladder's seed and one more, which means the sage floor is
   rendered at the second seed too.

Standing from the day: the shipped chain (sage on the outer steps, Sol
inside) is the leader; nothing beat it and nothing matched it faster. The
bake stays as the PDD8 checkpoint on hygiene alone.

*2026-09-15: superseded as the shipped chain.* The default is now core's
Model Attention Backend on kitchen int8 under Sol, sage off, on capture
grades and unscored renders rather than a blind verdict
([`wiki/decisions.md`](../../wiki/decisions.md), 2026-09-15). The sentence above is
what 2026-09-05 held.

## Current forward plan — 2026-09-03

The short form, for a session that only wants the next action, is
[`wiki/next_steps.md`](../../wiki/next_steps.md); this section owns the reasoning.

**The goal, in the owner's words:** personal experiments, shared with the
community, to find the best quality and performance this box can render.
Operationally: how much time Sol, Sage and PDD can take out of an H3 render
before the result visibly degrades, per scene, measured from the TRUE
baseline, with the owner's blind judgment as the top-level measure and the
numeric proxies reported beside it so we learn which proxies predict it.
The plan below replaced the 2026-08-24 one, whose encoder lane closed on
2026-08-27; that section was removed from this page on 2026-09-13.

**What 2026-09-03 established** (records under `bench/results/` with that
date prefix; `docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md`
is the design and correction record):

- kijai's block-max exact branch, the one the installed comfy-kitchen
  build carries, is the right numerical direction: on real Base16
  activations it is far closer to exact than the running-max branch and
  closer than the shipped Sage fallback, where random inputs had shown a
  wash. Sol's exact-stage arithmetic is therefore not a reason to make a
  block dense.
- The baseline is now the render you would otherwise make:
  `workflows/bench/h3_text_to_video_dense_stamped_api.json`, stock
  attention, no LoRA, and it proves it is one. Every earlier "dense
  baseline" number was Sage alone.
- Provenance is trustworthy from prompt to clip: one home for prompt text
  (`prompt_bank/`), the render's identity in every record, a capture
  manifest that says what ran and is checked by controls that can go red.

**What is NOT established, and must not be quoted as if it were:** whether
routing dominates at the shipped call (its sink ranges were not in the
capture); any dense-block set; whether a smaller kernel error is visible;
anything beyond the one scene the retained capture holds.

**The emerging architecture, as a hypothesis to test rather than a
setting to ship:** the middle of the trajectory on Sol with tuned routing;
the outer steps on Sage, possibly its fp16 mode; exceptional blocks at a
lower tau or handed to Sage; PDD on a baked checkpoint with its own legal
schedule. Per-block tau and a Sage fallback are rungs of one cost-quality
ladder per block, not alternatives.

**The sequence, each step earning the next:**

1. **The online Sol-versus-Sage instrument**, from the `sol_block_probe.py`
   scaffold: `trajectory=sol` returns Sol's output and computes Sage as the
   counterfactual on identical q/k/v, all 50 blocks, summaries only, no
   tensor dump, environment-gated with a zero-cost unarmed path. Validated
   against the retained Base16 capture (its retention note says so) before
   any new render. Its per-cell record is specified in the scaffold's
   docstring.
2. **Canonical Base16 FL2VA**: the pruned INT8 ConvRot fl2va DiT every
   graph loads, 1344x768, 345 frames, 16 steps, Sol on with production
   sink ranges and the empty `dense_blocks`: the first measurement of the
   shipped call. Done for one scene and seed on 2026-09-03
   (`data/results/2026-09-03_sol_probe_base16_standoff.json`, read through
   the research record's closing section); a second scene is owed before
   its ranking informs anything. Then **legal PDD8 FL2VA** by the node's
   own sigmas, then **Ref2VA** on the ref2va DiT; three populations, never
   pooled.
3. **A tau ladder per suspicious block**, plus the Sage fallback, on the
   direct Sol-versus-Sage delta and route cost together; one measurement
   ranks blocks, it does not choose their tau.
4. **Sage fp16 on the outer steps** as a plain render arm. Rendered in the
   2026-09-03 ladder as `solfp16`: slower than Sol as shipped on every
   scene; whether it looks different was the blind session's question, and
   at one seed it read both ways
   (`data/results/2026-09-04_ladder_2026-09-03_verdict.json`).
5. **Per-head routing** only if the shipped call shows a few heads carrying
   the error; that is kernel work, not a graph knob.
6. **A compact multi-scene blind ladder with clean timing**, base and PDD
   ladders kept apart, scenes from the bank chosen for their failure
   surfaces (a figure at distance, on-screen text, fast motion, quiet
   speech over music). This is where the numbers meet what the owner sees.
   First pass rendered and blinded 2026-09-03 (`bench/ladder_arms.json`,
   session `ladder_2026-09-03`), with PDD8 as a rung of the same ladder
   rather than a ladder of its own. Scored by the owner on 2026-09-04,
   pairs only, one seed
   (`data/results/2026-09-04_ladder_2026-09-03_verdict.json`); the
   direction and the two reading rules are in `wiki/next_steps.md`. The
   PDD8 rung turned out confounded (it carries sage and Sol like the rungs
   below it, plus the PDD-specific Sol `end_percent`), so PDD8 does need a
   ladder of its own after all: stock attention, sage alone, and the shipped
   graph, same scenes and seed. Built 2026-09-04 as
   `bench/pdd_ladder_arms.json` with a narrower-window Sol rung beside the
   three; the owner has ruled out more seeds for now. Rendered 2026-09-04,
   judged 2026-09-05 on pairs only (`data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`): PDD8 under sage alone lost
   to the floor on all five scenes, the narrow Sol window was free
   against sage alone on all five, the shipped window cost on two. The
   step is met on the pairs; the singles were not scored, so the audio
   half rests on the owner's notes and the loudness record.

**In parallel, not behind the above: the PDD bake.** It removes merge-time
requantisation at no runtime cost and is the strongest practical PDD
improvement on the table. A hypothesis for it to test, written 2026-09-05:
the defects the owner named blind on the shipped PDD8 rung (brightness,
compressed skin texture, melted on-screen text) are the kind a requantised
merge produces; the baked checkpoint rendering the same scenes at the same
seed is the test, and the PDD8-under-sage arm of the 2026-09-04 ladder says
first how much of that look is Sol's. The strongest argument for the bake is
now section 7 of [`research/merge_requantisation.md`](../../research/merge_requantisation.md):
on this checkpoint every runtime path, ours and the sister turbo node's,
leaves `mlp.fc2` requantising per cast, and the bake is the one fix that
covers every LoRA at once (`docs/h3_pdd.md`, "What that node does
differently", for what else generalises). Its contract is built and green
(`bench/check_pdd_sidecar_contract.py`; the two gaps Codex's 2026-09-03 audit
named were closed the same day in `e38655d`), so nothing precedes the bake
script itself. Until 2026-09-05 this sentence said two prerequisites remained
and pointed at `docs/h3_pdd.md`, which names none; the claim outlived its fix
by two days. The plan is
[`research/pdd/2026-09-05_bake_plan.md`](../../research/pdd/2026-09-05_bake_plan.md):
inputs, the pieces it lifts, the script step by step, the repo wiring, the
first pair, and a risk table with a control per row. **State 2026-09-05
afternoon: the bake is built and verified** (CHANGELOG 0.99.46 to 0.99.48;
`docs/pdd_artifacts.md` for the files), the graphs are wired, and the
merged-versus-baked pair is rendering as `bench/pdd_bake_arms.json`; the
owner scored it blind as `pdd_bake_2026-09-05` the same afternoon (`data/results/2026-09-05_pdd_bake_2026-09-05_verdict.json`):
the baked arm lost to the sage floor on four scenes of five on the same
properties the merged arm had lost on, so **the hypothesis did not hold at
this seed**: the PDD8 look is the schedule's, not the merge's. The bake
stays as hygiene (exact weights, nothing requantised at load) and the next
PDD quality lever is the partition. It shares
nothing with the attention lane and must not wait for it. Granular PDD strength comes after
the bake, on PDD's own ladder.

**The decision standard.** Trustworthy directional evidence promotes an
EXPERIMENT. A shared workflow default requires set-level testing, clean
timing in an uninstrumented process, and a small blind multi-scene
comparison; `dense_blocks="0-2,32"` became a default without those and was
withdrawn, and that is the precedent this sentence exists to break.

**What would count as finding it:** a setting that survives step 6 on more
than one scene and seed, with its cost measured in a clean process, and
its record naming the prompt, seed, canvas, length and build.

