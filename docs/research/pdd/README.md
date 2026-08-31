# PDD diagrams and handoffs

## Handoffs — newest first, and read the newest before anything else here

- [`2026-08-31_handoff.md`](2026-08-31_handoff.md) — **current.** What merging
  a PDD LoRA onto an int8 module costs, measured over all 200 modules: the
  backbone requantises and the output heads do not, the update lands but the
  requantisation perturbs the weights by more than the update on 97% of
  modules, and which PDD file and which base are now closed questions. Also the
  Tier 1 observer's state and the four review defects that reproduced against
  it. **Everything in it is stored-weight; nothing was rendered.** Read **§6
  before quoting any number from it** — the headline changed twice at the end
  of the day and the third framing is the one to use: the merge injects an
  error about the size of the entire int8 quantisation the checkpoint already
  carries, which is +11.6% on total stored-weight error against the release.
  §7 records the decision to HOLD on the Tier 1 gate, and why.
- [`tier1_gate.md`](tier1_gate.md) — **the gate, and undated because it is a
  standing precondition rather than a day's state.** Nine items that must be
  true before the Tier 1 activation observer is given a render: the dedicated
  capture graph, the arming key, the disk budget, the full-input allowlist, the
  open join to `H3_CAPTURE`, the absent offline scorer, the four-arm
  decomposition, the composition test against a real model, and the canvas.
  Read it before arming anything. Restates none of the handoff above.
- [`2026-08-30_handoff.md`](2026-08-30_handoff.md) — superseded as current by
  the above, and unchanged on its own subject. The depth
  question and the method correction that reshaped it: what is settled about
  per-block quantisation, the one clean observation capture and the two things
  it does not say, the four next steps in order, and the multi-session working
  rules the day earned.
- [`2026-08-28_handoff.md`](2026-08-28_handoff.md) — What the paper
  says about variable NFE and why it overturns the previous handoff's central
  conclusion; the six-evaluation arm the node refuses but the paper permits; the
  audio-refinement path; six prioritised action items with sources; the measured
  partition-fidelity result and the prediction it refuted; and five methodology
  traps that each cost time.
- [`2026-08-27_handoff.md`](2026-08-27_handoff.md) — superseded on its central
  question by the above, and carries a dated in-place correction to its item 2.

**Depth and the remaining axes**, opened 2026-08-30:
[`depth_and_axes.md`](depth_and_axes.md) — where in the DiT running undistilled
or over-distilled makes the most difference, what is already settled about
per-block *quantisation* (and why that is a different question), the four
measured depth profiles and the ways they disagree, and an inventory of every
PDD axis: exposed, structurally present but unreachable, and fixed. **Read its
§3 before repointing `probe_block_propagation.py`** — that probe is controlled
by a property of Sol's shape that a strength change does not share, and running
it naively returns numbers that mean nothing.

**Audio under PDD**, opened 2026-08-28 and handed to the `audioclaude` session:
[`audio_under_pdd.md`](audio_under_pdd.md) is the finding and the reasoning —
why a fused head's block-MEAN velocity meets an instantaneous change of variable
that only audio has, and why that predicts an audio-only error growing with
block width. [`2026-08-28_audio_plan.md`](2026-08-28_audio_plan.md) is the
execution plan: state, three experiments in order, what each outcome means, and
the traps. Read the first, work from the second.

**Rendered arms, 2026-08-29:**
[`2026-08-29_market_scene_arms.md`](2026-08-29_market_scene_arms.md) — three
market-scene renders forming two single-variable pairs (Sol on/off, and
`dpmpp_2m_sde_gpu` against `euler`) at 8 steps on PDD-emitted sigmas. It
opens by saying why these are **not** comparable to group S, since steps,
length, canvas and the sound prompt all moved. Its one transferable result is
that the reference integrator is viable here, and that only the deterministic
arm is repeatable — which is the precondition for every other comparison in
this lane.

[`queued_arms.md`](queued_arms.md) is the session queue those handoffs draw on.
The paper itself is
[`arxiv_2607.26004v1_...md`](arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md);
`docs/h3_pdd.md` owns the contract and every file here defers to it.

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
on one row (a partial patch-key match) it enforces what `docs/h3_pdd.md` lists
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
a real 4-step render so the diagrams can be checked against a log.

[`queued_arms.md`](queued_arms.md) — the arms waiting on the GPU, each with
what it decides. A session queue rather than a roadmap: `docs/roadmap.md` and
`docs/open_experiments.md` own those, and nothing in this file is deliberately
unmeasured. Delete an entry when it runs.

**A teaching surface, not a source.** Every number on those pages is owned by
[`docs/h3_pdd.md`](../../h3_pdd.md), and where a page disagrees with it that
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
