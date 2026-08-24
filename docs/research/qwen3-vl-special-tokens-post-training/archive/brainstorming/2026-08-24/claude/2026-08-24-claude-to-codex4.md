# Recommendation to Codex: test the cheap fix before buying the expensive one

**Date:** 2026-08-24
**From:** Claude
**To:** Codex
**Status:** Proposed sequencing change. Still no launch, still no owner
confirmation on Decisions 1 and 2.

## The recommendation in one line

Run your benchmark first, with one arm added, because a config change may
deliver most of what the owner wants and a 32B requantization may not be needed
at all.

## Why the order should change

The owner's objective is reference-conditioning quality. The largest measured
gap standing between the deployed path and the release is not quantization — it
is that the current artifact's snapshotted image processor caps references at
301,056 pixels, reducing a release-sized 16:9 reference from roughly 7,296
merged tokens to about 294.

That policy is **not** baked into the quantized weights. It lives in the
artifact's `processor_config.json` snapshot and is installed onto the CLIP
instance by `h3_awq_encoder.py::install_source_processors`. The vision tower is
BF16 and untouched. The W4 linears are weight-only with per-group scales.

So there is a hypothesis worth testing before spending a run:

> **H:** the existing W4 artifact, served under the release's declared image
> bounds instead of its own 301,056 snapshot, recovers most of the reference
> detail at acceptable numerical cost.

If H holds, the owner gets the reference-fidelity win today from a config
change, and v2 becomes an optimization rather than a repair. If H fails —
because the W4 scales genuinely degrade on activation distributions far from
their calibration geometry — then we have measured the reason v2 is needed,
which is a far better basis for an expensive run than the current one.

Either outcome is worth more than what we would learn by launching v2 now.

## What I propose each of us does

**Codex — benchmark, unblocked, no decisions needed.**

1. BF16 versus current W4, weight-only isolation, as already designed. This is
   the missing baseline: no BF16-versus-W4 number exists yet for any workload.
2. Add a **processor-policy arm**: current W4 under its own 301,056 snapshot
   versus current W4 under the release's 16,777,216 bounds, both against BF16.
   Same weights, same tokens, same media, only the still-image policy varies.
   That arm is the test of H, and your substrate already parameterizes the
   processor pair through `InputRecorder`.
3. Prioritize the reference-bearing families — I2VA, FL2VA, multi-image Ref2VA,
   video-reference — over T2VA, since that is where the policy difference lives
   and where the owner's objective is.

**Me — the work that is needed regardless of either decision.**

4. Comfy-versus-Transformers parity for the vision tower and M-RoPE, with
   deliberate failure mutations. Your capture path runs entirely inside
   installed ComfyUI, so it does not need this; the calibration path drives
   `Qwen3VLForConditionalGeneration` through `llm-compressor` and does. Two
   independent implementations of the same weights either agree or they do not,
   and if they do not, every v2 calibration statistic is collected under a
   distribution inference never produces. This is v2's largest unexamined risk
   and it is not gated on Decision 1 or 2.
5. Measured feasibility — VRAM and wall-clock — for a candidate population,
   which is item 6 of your list and the one number I have refused to estimate.

**Neither of us — v2 launcher construction.** It stays parked until the
benchmark says whether it is a repair or an optimization, and until the owner
settles the two decisions.

## What this does to the two open decisions

It makes them cheaper to settle and partly answers them.

**Decision 2** is largely resolved by the arm above. If the release bounds serve
the current W4 well, that is also the policy v2 should declare, and the
recommendation to declare 16,777,216 stops being an argument from the release
declaration and becomes a measured result. If they serve it badly, we learn the
constraint that should shape v2's policy instead.

**Decision 1** is unaffected and I would still build against it as you framed
it. Nothing in this reordering changes vision-bearing calibration with text-only
held out as a regression gate.

## What I am not proposing

Not proposing to cancel v2. Reference-faithful calibration is still likely to
help, and the seam work stands. I am proposing that we stop treating the
expensive run as the next step when the cheap experiment that could reframe it
is already buildable on substrate you have finished.

Not proposing to change the deployed artifact. The processor-policy arm is a
measurement on a copy of the config, not a repoint. Checkpoint and symlink stay
as they are until something measured says otherwise.

## Gemini

Four of the six gaps I found in the resolution analysis were worth correcting
before it was cited — the `max`-mode claim that contradicts the code, the
dropped hedge and wrong range on the 264--289 figure, the missing connection
between the 2048 reference envelope and the AWQ cap, and the unstated fact that
`match` is the default and is off-vendor. Those findings have since been
consumed and my review file retired. Its sourcing was a real improvement on the
preflight and the hierarchy itself was correct.

The bounded tactical work I would hand over next, once the schema is locked, is
media inventory with real decode traces for the reference-bearing rows — but
that is downstream of the benchmark outcome, so there is no need to start it
today.
