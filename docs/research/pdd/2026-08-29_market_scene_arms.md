# Market scene, 2026-08-29: three arms, two clean pairs

last updated: 2026-08-29

Three renders on the night-market scene that form **two single-variable pairs
about a shared hub**. Recorded because they existed only as files the owner
judged good, and because one of the pairs prices a divergence that
[`../h3_dit_implementations.md`](../h3_dit_implementations.md) §4.1 had left
open.

The machine-readable record is
[`../../../bench/results/2026-08-29_market_scene_arms.json`](../../../bench/results/2026-08-29_market_scene_arms.json);
this file is the reasoning. Both were derived with
`bench/diff_render_graphs.py`, from the graph ComfyUI embedded in each render
rather than from any workflow JSON.

## Read this first: these are NOT Group S

[`queued_arms.md`](queued_arms.md) group **S** is the market scene too, and it
is tempting to read these three against it. Do not. Four things moved:

| | group S | here |
|---|---|---|
| steps | 4 | 8 |
| length | 362 frames | 345 frames |
| canvas | group S's | 1152x768 |
| sound prompt | group S's | rewritten to piano |

Group S varied Sol settings at a fixed everything-else, which is what made it
readable. These three vary Sol and the sampler at a *different* fixed
everything-else. Within this group the pairs are clean; across the two groups
nothing is.

## The arms

Hub is **B**. Each pair differs from it by one thing, verified field by field
on the reachable graph.

| arm | sampler | Sol | differs from B by |
|---|---|---|---|
| **A** | `dpmpp_2m_sde_gpu` | off | the `SolAttnMiniMax` node |
| **B** (hub) | `dpmpp_2m_sde_gpu` | on | — |
| **C** | `euler` | on | `sampler_name`, and nothing else at all |

Shared: one seed, byte-identical prompt, the PDD 8-step fl2va LoRA at strength
1.0 with `patch_heads` on and `nfe=0`, the int8_convrot fl2va DiT, 8
evaluations, and sigmas taken from the **PDD node's SIGMAS output** rather than
from a scheduler.

Two things about that configuration are worth stating because both were
misread once in the session that recorded them:

- **The `ManualSigmas` node in these graphs is inert.** It feeds a preview
  branch. The live sigmas come from the PDD node through a visualizer
  passthrough. Reading node presence instead of link reachability put the step
  count at 6 and every number derived from it wrong.
- **The removed model-sampling shift node was a no-op.** `supported_models.py`
  already carries the same two shifts the node defaults to, and the DiT falls
  back to them when the transformer option is absent.

## What the pairs establish

**Both pairs: both arms met the brief.** That is the whole of the perceptual
result and it is the honest ceiling for a pair. Sol and the sampler are each a
numerical change, so each arm is a *different sample* rather than a variant of
the same one — the standing rule in `CLAUDE.md`, and
[`../../eval_comparison.md`](../../eval_comparison.md) section 3 is what a
verdict would require. n=1 per arm, unblinded, unscored.

**The sampler pair carries one thing a pair can carry.** `euler` at ComfyUI's
default `s_churn=0` takes the no-noise branch and steps `x + d*dt` — first
order, deterministic, eta=0. That is the integrator every reference engine in
[`../h3_dit_implementations.md`](../h3_dit_implementations.md) runs. So the
vendor's own integrator is viable here, which is a fact about reachability, not
about quality, and it does not need a distribution to be true.

**The consequence is repeatability, and it outranks the quality question.**
Only arm C is repeatable: `dpmpp_2m_sde_gpu` draws fresh noise every step, so a
same-seed repeat of A or B is a different sample. Every comparison made on this
install under an SDE sampler has therefore been confounded by the sampler's own
nondeterminism — including this group's own Sol pair.

## Derived from this schedule

Sigmas are the uniform block-width-4 partition of the 32-point grid.

- **Head coverage is exact.** The 8 evaluations tile the head bank
  `[0,4) [4,8) … [28,32)` — all 32, no gaps, no duplicates.
- **Sol is active on half the steps.** Its percent band resolves to a sigma
  window that steps 3 through 6 fall inside; 1 and 2 sit above it, 7 and 8
  below. Step 7 misses by less than a hundredth of a sigma, and an
  `end_percent` of 0.75 is the exact boundary that would pull it in. Whether
  that is desirable is unmeasured.
- **The audio change-of-variable drifts hardest in the final block.** Its
  coefficient falls by roughly a fifth across block 7 and by roughly half
  across block 8, against a few percent across the early blocks; the same
  statistic on a uniform 4-step partition is worse still in the final block.
  [`audio_under_pdd.md`](audio_under_pdd.md) owns this mechanism and the values
  are its statistic evaluated on this schedule, not a new finding. It is the
  reason 8 steps helps where it helps: almost all of the gain over a uniform
  4-step partition is in the tail.

## One observation, not a finding

These arms load the **AWQ v2 encoder**, while `h3_config.MODELS["clip"]`
resolves to the INT8 one and the v2 lane was closed on 2026-08-27. The owner's
best renders to date come off the closed lane.

**"The shipped graph" and "what that render ran" are different claims and both
are true here** (established by the `encoder` session, 2026-08-29): every
loader node across the shipped graphs names int8_convrot through `CLIPLoader`,
and these arms override that with AWQ v2. Neither statement contradicts the
other, and citing one for the other is the error to avoid.

None of this is evidence *about* the encoder: nothing here varied it, and these
arms differ from every earlier group in four other ways. It is worth knowing
only so that the lane's assumption about what is in play matches what the
graphs do.

## The next test, and why it is this one

**A same-seed repeat of arm C.** It is cheap, it is the precondition for every
other comparison in this lane, and its failure mode is the interesting one: if
the repeat does *not* reproduce, something below the sampler is
nondeterministic and worth locating, which is a better find than a clean
reproduction. Compare decoded frames rather than containers, with the recipe
already in [`queued_arms.md`](queued_arms.md) — `save_metadata` embeds the
workflow, so two byte-different mp4s can hold identical video.
