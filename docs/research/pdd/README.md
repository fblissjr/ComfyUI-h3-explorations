# PDD diagrams and handoffs

**No record here is current.** The current pages are [`docs/distills/pdd.md`](../../distills/pdd.md), which owns
the contract, the traps and what is measured, and [`docs/distills/pdd_node.md`](../../distills/pdd_node.md), which
owns how the node, `pdd_math.py` and the converter work. Every file below is what was true on its date, and where it
disagrees with those two pages they are right. PDD quality work has been parked since 2026-09-05
([`docs/state/roadmap.md`](../../state/roadmap.md), "Closed lanes"), reopened only for the 2026-09-25 upstream
research and the audio-freeze lane ([`docs/state/decisions.md`](../../state/decisions.md)).

## Handoffs and dated records, newest first

- [`2026-09-25_upstream_pdd_comparison.md`](2026-09-25_upstream_pdd_comparison.md) -- **the newest record.** Seven
  PDD implementations against ours: none beats ours on anything that reaches a shipped render; core's audio carry is
  exact under Euler, which refuted the change-of-variable account of PDD's audio loss; the stale Kijai files were
  unlinked. Supersedes [`2026-08-28_pdd_implementations.md`](2026-08-28_pdd_implementations.md) as the comparison and
  [`2026-08-28_audio_under_pdd.md`](2026-08-28_audio_under_pdd.md) section 1.
- [`2026-09-05_bake_plan.md`](2026-09-05_bake_plan.md) -- **the last handoff, executed.** The backbone bake was
  planned, approved, built, wired and judged blind the same day: the baked arm lost to the sage floor as the merged
  arm had, so the PDD8 look is the schedule's, not the merge's, and the bake stands on hygiene and cost. Supersedes the
  2026-08-31 handoff's step 2 on the bake.
- [`archive/docs/pdd_handoffs/2026-08-31_handoff.md`](../../../archive/docs/pdd_handoffs/2026-08-31_handoff.md) -- superseded. What merging a PDD LoRA onto an int8 module costs,
  measured over all 200 modules and stored-weight only. **Read its section 6 before quoting a number**: the headline
  changed four times that day, and the last version is that the merge leaves stored-weight error 1.40x the base int8
  error under the shipped stochastic rounding (1.12x, the +11.6%, under round-to-nearest). Section 7 records the hold
  on the Tier 1 gate. Its `unmerged_blocks` lever was replaced by the exact branch on 2026-09-26
  ([`docs/state/decisions.md`](../../state/decisions.md)).
- [`2026-08-31_tier1_gate.md`](2026-08-31_tier1_gate.md) -- closed, never passed. The ten conditions the Tier 1 activation observer had to
  meet before a render (dated 2026-08-31, item 10 added 2026-09-01). The owner closed the experiment on 2026-09-29
  and the observer was removed ([`docs/state/open_experiments.md`](../../state/open_experiments.md) #23).
- [`archive/docs/pdd_handoffs/2026-08-30_handoff.md`](../../../archive/docs/pdd_handoffs/2026-08-30_handoff.md) -- superseded by the 2026-08-31 handoff. Per-block quantisation
  sensitivity is flat and closed; where distillation strength matters by depth stayed open; the one capture, and the
  four next steps, which did not run in that form.
- [`archive/docs/pdd_handoffs/2026-08-28_handoff.md`](../../../archive/docs/pdd_handoffs/2026-08-28_handoff.md) -- superseded in part. The paper makes variable NFE the designed
  feature; the six-evaluation `[8,8,4,4,4,4]` partition shipped; four evaluations cannot be improved; the measured
  partition-fidelity result and the prediction it refuted; five methodology traps. Its audio account was refuted on
  2026-09-25 and its partition work parked on 2026-09-05.
- [`archive/docs/pdd_handoffs/2026-08-27_handoff.md`](../../../archive/docs/pdd_handoffs/2026-08-27_handoff.md) -- superseded by the 2026-08-28 handoff, and carries a dated
  in-place correction to its item 2.

**Depth and the remaining axes**, opened 2026-08-30:
[`2026-08-30_depth_and_axes.md`](2026-08-30_depth_and_axes.md) — where in the DiT running undistilled
or over-distilled makes the most difference, what is already settled about
per-block *quantisation* (and why that is a different question), the four
measured depth profiles and the ways they disagree, and an inventory of every
PDD axis: exposed, structurally present but unreachable, and fixed. **Read its
§3 before repointing `probe_block_propagation.py`** — that probe is controlled
by a property of Sol's shape that a strength change does not share, and running
it naively returns numbers that mean nothing.

**Audio under PDD**, opened 2026-08-28 and handed to the `audioclaude` session:
[`2026-08-28_audio_under_pdd.md`](2026-08-28_audio_under_pdd.md) is the finding and the reasoning —
*(2026-09-25: refuted; see [`2026-09-25_upstream_pdd_comparison.md`](2026-09-25_upstream_pdd_comparison.md))* why a fused head's block-MEAN velocity meets an instantaneous change of variable
that only audio has, and why that predicts an audio-only error growing with
block width. [`2026-08-28_audio_plan.md`](2026-08-28_audio_plan.md) is the
execution plan: state, three experiments in order, what each outcome means, and
the traps. Read the first, work from the second.

**What makes a clip degrade, 2026-08-28:**
[`2026-08-28_scene_complexity.md`](2026-08-28_scene_complexity.md) is the shot-count ablation's record (three
accounts refuted, the owner's account of total demanded content standing, with a duration confound left open), and
[`2026-08-28_artifact_axis.md`](2026-08-28_artifact_axis.md) is the same night's viewing map and pre-registration for the arms queued
against it.

**Rendered arms, 2026-08-29:**
[`2026-08-29_market_scene_arms.md`](2026-08-29_market_scene_arms.md) — three
market-scene renders forming two single-variable pairs (Sol on/off, and
`dpmpp_2m_sde_gpu` against `euler`) at 8 steps on PDD-emitted sigmas. It
opens by saying why these are **not** comparable to group S, since steps,
length, canvas and the sound prompt all moved. Its one transferable result is
that the reference integrator is viable here, and that only the deterministic
arm is repeatable — which is the precondition for every other comparison in
this lane.

[`2026-08-27_queued_arms.md`](2026-08-27_queued_arms.md) is the session queue those handoffs draw on.
The paper itself is
[`arxiv_2607.26004v1_...md`](arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md);
`docs/distills/pdd.md` owns the contract and every file here defers to it.

## Diagrams

[`two_pdd_paths.html`](two_pdd_paths.html) — our PDD implementation against
Kijai's, traced end to end through the six production PDD graphs. Open it from
disk; it is self-contained apart from web fonts, which have local fallbacks.

Four diagrams: the shared graph spine with the accelerator slot marked, load
time side by side, one sampling step side by side, and the 32-interval grid at
8, 4 and an off-schedule step count. Plus a per-graph table and the four places
the two implementations diverge.

**Corrected 2026-08-27** for the derived-step-count rewrite. It was written
before the node stopped asking for a step count, so it described fusing at load
and refusing an off-grid count; both are gone. The same pass completed the
per-graph table, which had claimed to be all four PDD graphs while there are ten
and it listed only the two t2v and two image-ref arms.

[`three_pdd_implementations.html`](three_pdd_implementations.html) — the same
mechanism in a third pack, `silveroxides/ComfyUI-UtilsCollection`, against ours
and Kijai's. Three diagrams: where each one gets the block index, which
partitions of the 32-interval grid each will build, and who owns the fused head
tensors at runtime. Plus a guard matrix, which is where the three actually
separate — that pack fails closed on several things ours only warns about, and
on one row (a partial patch-key match) it enforces what `docs/distills/pdd.md` lists
under "Enforced by nothing".

[`thirty_two_intervals.html`](thirty_two_intervals.html) — what an NFE is, what
PDD does while a render is running, and what moves when the step count does.
Three diagrams: the fixed 32-interval grid grouped at 8 and at 4, the ragged
blocks 5 and 6 produce because they do not divide 32, and each step's share of
the sigma path with Sol's sparse steps marked on it. That last one carries the
finding worth keeping — the final evaluation is 44% of the path at 16 NFE and
80% at 4, while Sol's coverage falls from 11 of 16 steps to 2 of 4, so the two
accelerations are buying time from overlapping budgets. Every number on it was
computed from `pdd_math.py` rather than quoted, and it shows the trace lines of
a real 4-step render so the diagrams can be checked against a log. **Two of
those coverages were observed rather than computed on 2026-09-01**, by the
live Sol route record on the shipped graphs at the trained canvas: the
16-step base graph ran Sol on 11 of 16 forwards and the 8-evaluation PDD
graph on 4 of 8, each forward carrying blocks 0-49 once, with the actual
routed density per (block, step) beside it --
`data/results/2026-09-01_sol_route_base16_cold.json` and
`..._pdd8_cold.json`, plus `..._pdd8_cold_vs_warm.json`, which shows the
PDD graph's counts bitwise identical across a restart and a warm process.
The kernel figures include the forced pairs; the ordering-effect figures
are the `analyze_routing.py` number. Routing, not quality: nothing in them
says what the density cost.

[`2026-08-27_queued_arms.md`](2026-08-27_queued_arms.md) — the arms waiting on the GPU, each with
what it decides. A session queue rather than a roadmap: `docs/state/roadmap.md` and
`docs/state/open_experiments.md` own those, and nothing in this file is deliberately
unmeasured. Delete an entry when it runs.

**A teaching surface, not a source.** Every number on those pages is owned by
[`docs/distills/pdd.md`](../../distills/pdd.md), and where a page disagrees with it that
document is right. Nothing generates them and no check reads them, so they are
the copies that go stale silently — the same standing hazard `CLAUDE.md` records for
`internal/h3_resolution_explainer.html`.

Both built 2026-08-27, and every third-party claim on them describes that date,
because all three subjects were moving: the Kijai converted files changed
head-bank encoding that day, Comfy-Org/ComfyUI#15908 was open with
`comfy/ldm/minimax/model.py` alone in its diff, and the third pack had added its
PDD path the day before (`23ab5f2dd4f6`, read at HEAD `5bac35be3d61`). The
three-way page also describes our own `nfe` widget, which was being removed the
same afternoon; it says so where it matters.

Each is also published as an Artifact. Those copies and these are updated by
hand and have no mechanism keeping them in step.
