# Step caching for H3, re-examined

last updated: 2026-09-25

> **Provenance, 2026-09-25.** A research subagent (Claude) read the upstream
> caches and searched the literature, using CPU only. The session that
> commissioned it wrote this note. Figures from papers are the papers'
> claims, labelled as such. Nothing here was rendered.

**The owner's question:** step caching was parked on 2026-08-20
(`docs/roadmap.md`, `CACHE_NODE` probe-only, "Dies: step caching"). Is there a
newer method worth reopening it for?

**Answer: only for the 16-step base graphs, and only one method.** Under
every upstream's own warmup and cooldown rules:
- nothing can be skipped at 4 or 6 steps;
- at most 2 or 3 steps can be skipped at 8.

So the parking reason still holds for every distilled graph. The newer method
that gets around the rest of it is **DPCache**, which LightX2V ships for H3
together with Sol. It pays only at the base step count. The decision turns on
a question for the owner: **are the 16-step base graphs still rendered, or only
PDD and the distills?**

The adopt-upstream rule does not fire. sglang and vllm-omni both default H3 to
`quality="lossless"`, which means no cache, as ours does. On TeaCache they
disagree: sglang refuses it for H3, and vllm-omni ships it for FL2VA only.

## What upstreams ship for H3

- **LightX2V DPCache**
  - What it does: skips the whole block stack at the steps it leaves out and
    predicts the stack's output by an order-2 Taylor extrapolation from the
    last computed steps. The final layer still runs, with the current step's
    modulation.
  - How it chooses: a dynamic program over a path-aware cost tensor picks a
    fixed skip set. The tensor comes from one uncached calibration render.
  - Calibration signature: steps, frames, size, shifts, model, task, attention
    type and the Sol settings. A mismatch refuses to run.
  - It is approximate. Licence: Apache-2.0, adapted from
    `argsss/DPCache`.
  - Pointers:
    `coderef/LightX2V/lightx2v/models/networks/minimax_h3/infer/feature_caching/dpcache/transformer_infer.py::MiniMaxH3TransformerInferDPCaching`,
    `coderef/LightX2V/lightx2v/models/networks/minimax_h3/infer/feature_caching/dpcache/schedule.py::select_steps`.
- **LightX2V MagCache**: can't be selected
  (`coderef/LightX2V/lightx2v/models/networks/minimax_h3/model.py::MiniMaxH3Model._init_infer_class`),
  and the calibration tool its error message names does not exist.
- **vllm-omni Cache-DiT (first-block residual)**
  - Its "high" profile warms up 4 steps, uses threshold 0.04, and allows one
    consecutive skip. sglang ships the same profile.
  - Off by default. With a 4-step warmup it can skip nothing at 4 steps.
  - `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/quality_policy.py::MiniMaxH3QualityPolicy`.
- **vllm-omni TeaCache**: its polynomial coefficients are FL2VA-only, and the
  step count they were fitted at is not stated.
- **Sana's TeaCache and FirstBlockCache**: uncalibrated (identity
  coefficients) and tuned at 50 steps.
- **Core's EasyCache and LazyCache**: output-level reuse inside a window.
  EasyCache decides on video only; LazyCache calls itself worse. Measured here
  on 2026-08-18 (`bench/results/2026-08-18_cache_arms.jsonl`): nothing skipped
  on `er_sde` at the 0.2 threshold, and 7 of 16 skipped on `res_multistep`.
  The two quality pairs from that day are still unjudged.

## Newer methods from the literature (their claims, not measured here)

- **DPCache** (arXiv 2602.22654): a globally planned skip set, then
  prediction. The one fit for H3 today, since LightX2V already runs it on the
  packed sequence.
  - **A gap no upstream closes:** its cost sample is drawn uniformly over the
    whole packed output, so it includes text and reference rows the final
    layer never reads, and it is dominated by video. A port should sample only
    the target video and audio rows and weight the two separately.
- **MagCache** (2506.09045): residual magnitude ratios from one calibration
  sample. The ComfyUI node (Apache-2.0) has no H3 support.
- **TaylorSeer**: GPL-3.0, and sglang calls it unsuitable for few-step models.
- **FoCa** (2508.16211): a predictor-corrector, available in cache-dit.
- **Spectrum** (2603.01623) and **RACER** (2608.01740): forecasting, and a
  closed-loop guard that could sit on a forecaster.
- **DisCa** (2602.05449): the only method aimed at 4-step distills, but it
  needs a trained predictor. Not feasible for a 33B model on one card.
- **EchoCache, Chorus**: the wrong workload.

## If the owner wants it: a DPCache-style node for the base graphs

- **Why it gets around the 2026-08-20 reason:**
  - The skip schedule is fixed and calibrated, so `er_sde`'s re-noise cannot
    suppress skipping.
  - It forecasts rather than reuses.
  - It forecasts the hidden state before the final layer, so the current
    step's modulation and any head patch still run.
- **The build:**
  - A DIFFUSION_MODEL wrapper plus `patches_replace["dit"]` on the blocks.
  - Clear weight prefetch on skipped steps, or they still stream weights.
    Reasoned from `comfy/model_base.py`, untested.
  - Hold the forecast state on the host.
  - Refuse to stack with the pack's other `patches_replace["dit"]` users.
  - Record the cache config and the chosen steps in the provenance stamp.
- **Calibration:** one uncached render per signature, sampling only target
  video and audio rows. Check on a few scenes that the dynamic program picks
  the same steps.
- **How it would be judged:**
  1. Teacher-forced numerics on one capture render. The output-level variant
     can use today's final tap; the hidden-level variant needs a new tap.
  2. A blind matched-seed pair through `h3-ab-session`, against its own
     no-cache control and the baseline.
  3. Timing, with cache state stated.

**Second choice:** a first-block residual cache at vllm-omni's high profile.
It needs no calibration, and its likely failure is the one EasyCache already
hit on `er_sde`.
