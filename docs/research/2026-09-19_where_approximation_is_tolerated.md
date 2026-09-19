# Where approximation is tolerated: steps, blocks and heads chosen by profiling (2026)

> **Provenance, 2026-09-19.** Written by a research subagent of the `libguy`
> session for the lead's wider sweep, read-only, no GPU. The commissioning
> session re-checked its code pointers and found them correct: the final-layer
> velocity tap (`h3_capture.py::maybe_capture_final`), the Sol override's
> fall-through to the previous override outside its window and on
> `dense_blocks` (`sol_attn_h3.py`, `dense()` and the `outside_range` route),
> the absence of any refusal when a second Sol node chains onto a first (the
> node only logs "chaining onto an existing attention override"), and the
> three 2026-08-19 per-head records it cites. Whether two chained Sol nodes
> give a per-block sigma window is the agent's reading and is untested. Paper
> figures are the papers' own; arXiv 2605.14513 is HASTE in v1 and HEART in v2,
> and the two versions differ on the dense warm-up, so cite it by version.

Provenance. Research subagent, 2026-09-19, read-only, no GPU, no git. Read in
full text (arXiv HTML converted to text and grepped, so quotes below are from
the paper text, not a summariser): Sol-Attn 2607.24027, Sol Video Inference
Engine 2606.23743, PISA 2602.01077, CalibAtt 2603.05503, ScalingAttention
2606.23019, PASA ("Ride the Wave") 2604.12219, MOD-DiT 2601.11641, DFSAttn
2605.23445, EpaCache 2608.29264, SpectralCache 2603.05315, NaviCache
2606.26795, LiteAttention 2511.11062, "Not All Tokens Need 40 Steps"
2605.06892, HEART 2605.14513v2 and its v1 PDF (then titled HASTE), EasyCache
2507.02860, MagCache 2506.09045, TeaCache 2411.19108, Sparse VideoGen
2502.01776, Sparse-vDiT 2506.03065, STA 2502.04507, DuoAttention 2410.10819,
Autonomy-of-Heads 2608.06849, HeadCast 2607.20125, the ParaAttention README
(First Block Cache). Read as abstract only: SeaCache 2602.18993, ERTACache
2508.21091, BAC 2506.13456, BWCache 2509.13789, Delta-DiT 2406.01125, BOSCH
2604.05942 (abstract plus a summariser pass), Head Forcing 2605.14487, SPADE
2608.03335, DPCache 2602.22654, AdaCorrection 2602.13357, TAP 2603.03792,
VSA 2505.13389. Local: the sglang SubBlock router notes and schedule
constants, the Sana, LightX2V and sglang H3 configs, and this pack's
`sol_attn_h3.py`, `sol_observe.py`, `workflows/h3_config.py`,
`docs/SOLATTN.md` and the named records. Out of scope by assignment (one line
each at most): LoSA, "Attention Sparsity is Input-Stable", HASTE/HEART's
method, cumulative-energy filtering, DeepSeek V4.1. Builds on
`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md` D.6-D.7 and
`internal/2026-09-19_question_review.md` C.2-C.4, D, F.4, F.9; does not
repeat them.

## The answer

- **Almost every dense warm-up in the literature is a borrowed convention, not
  a measurement.** Sol-Attn, Sparse VideoGen, Sparse-vDiT, DFSAttn and HEART
  all borrow it with "following prior work" or similar, and ablate
  nothing. Only a handful measured
  the boundary at all (table in 1.1). `start_percent` 0.2 is not a knob this
  pack failed to measure; it is a knob the field never measured. [V]
- **The measured evidence says removing warm-up can change WHICH sample you
  get without changing how good it is, for methods that approximate the
  unrouted tail.** PISA (same first author as Sol-Attn) loses most of its
  PSNR against dense without warm-up while the two VBench quality columns it
  reports hold; SpargeAttn, which drops the tail, loses both. Sol sits between
  them: a pooled zeroth-order tail, but not the first-order term PISA credits
  for its robustness. So a quality-neutral basin change is the optimistic
  end of Sol's range and SpargeAttn's collapse the pessimistic end. "Not All Tokens Need 40
  Steps" (Appendix A) states the mechanism: early perturbation moves the
  sample into a different basin, and per-sample metrics against the dense
  render stop meaning anything. [V for the tables; I for the Sol analogy]
- **So the instrument decides the answer for `start_percent`.** Cosine or
  PSNR against a same-seed dense render (sglang's H3 sweep, and this pack's
  B.4 instrument) measures basin membership. Only a quality judgement over
  several scenes, read against a decoy floor (question review C.1), can say
  whether 0.0 is worse. [I]
- **Every borrowed "20%" is a fraction of step index under that model's own
  shift, not a sigma.** At shift 12, `start_percent` 0.2 ends dense attention
  at sigma 0.98; the same 20% of steps ends at sigma 0.95 at shift 5.
  [I, labelled arithmetic in 1.6]
- **On 16 base against 8 distilled steps, convention holds the fraction and
  nobody tested it, except one paper that refit and got a different curve.**
  STA keeps 12/6/3 dense at 50/25/10 steps; CalibAtt's fitted schedule for a
  4-step distilled model has no near-dense first step, unlike its 50-step
  fit. Sol-Attn's own authors ran an 8-step LoRA stage fully dense. [V]
- **The last step is protected by step CACHING papers and not by sparse
  ATTENTION papers**, and the two approximations fail in opposite places.
  Late steps are where attention sparsity costs least (LiteAttention,
  CalibAtt, ScalingAttention, DFSAttn) and where reusing the previous step's
  output costs most (EasyCache, EpaCache, MagCache, PASA, SpectralCache).
  Upstream's `end_percent` 1.0 is consistent with the attention evidence at
  50 steps; nothing covers an 8-step shift-12 last step. [V per paper; I for
  the split]
- **Per head: the heads' patterns transfer across prompts, but the heads
  interact.** Most papers find per-head patterns stable across prompts, steps
  and resolutions. BOSCH and HEART say per-head choices are not separable, so
  scoring heads one at a time misleads. This pack's own 2026-08-19 records
  already show that making the worst heads exact bought about the same local
  error per unit of density as lowering tau globally. Do not ask for a
  per-head tau kernel yet. [V for the papers and the records; I for the
  conclusion]
- **Human studies are nearly absent.** Of everything read, only STA ran a
  real one: 200 prompts, pairwise. Sol Video Inference Engine keeps an
  informal human validator because PSNR misranks blur and jitter. Everything
  else is VBench/PSNR/SSIM/LPIPS. [V]

## What this means for H3, ranked

Ranked by how directly each item could change a default or a kernel in this
pack within a week. Item 1 costs no render; items 2 and 3 need ordinary
renders, with no capture and no panel.

1. **Read `start_percent` evidence as basin evidence, and change the
   instrument before the panel.** [I, from V sources] The one H3-specific
   measurement (sglang: step cutoff 10 to 5 "halves cosine-vs-dense ... and
   visibly re-frames the shot",
   `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:77-81`)
   is a same-seed similarity figure, so it shows a basin change, not
   necessarily a defect. This pack's own 0.0 arm on the post office scene
   re-framed without a fault (`bench/results/2026-09-17_sol_start_percent_batch.md`;
   that prompt ran past its scripted length, see
   `bench/results/2026-09-18_off_length_prompts.md`, so it is a hint, not a
   result).
   Measure first: nothing new. Score the valid unscored stack (F.4) as
   "is this good", never "does it match 0.2". Then run F.4's panel with
   several seeds per scene and the decoy pair from F.3, because a 0.0-vs-0.2
   pair at one seed is statistically a two-sample pair once the basin moves.
   Drop distance-to-dense (B.4) as evidence for this knob specifically.
2. **Get a measured boundary from the model itself: PASA's velocity curve and
   EasyCache's transformation rate.** [V for the signals; I for H3] Both are
   computed from consecutive per-step model outputs: velocity L1 between
   adjacent steps (PASA 4.1, Eq. 9-11) and
   `k_t = ||v_t - v_{t-1}|| / ||v_{t-1} * delta_s_t||` for an Euler flow step
   (EasyCache Sec. III-B). Both papers find a volatile opening phase, a flat
   middle and (PASA) a late resurgence. Measure first: one render each of
   base 16-step and PDD8, saving the DiT's velocity at every step. The
   capture's final-layer tap already does this (`h3_capture.py:588`, armed
   by `H3_CAPTURE` with `final=1` and the step list; installed by the sage
   node (`_final_tap` in this pack's `nodes.py`, not ComfyUI's), so it needs that node in the graph, and it
   also writes q/k/v for its `blocks` set, default block 0). Two armed
   renders (arming is a server restart; ask first on the shared server),
   then CPU; no panel. The step
   where `k_t` stops moving is a model-derived candidate for
   `start_percent` at each step count; it replaces "the paper uses 0.2" with
   a number this model produced. Whether the default chain's graph carries
   the sage node that hosts the tap: [?], not checked.
3. **A per-block sigma window may already be expressible by chaining two Sol
   nodes; check before building it.** [V for the code path; I that it works]
   Outside its window, and for blocks in its `dense_blocks`, the override
   calls `dense()`, which hands off to the previously installed override
   (`sol_attn_h3.py:1028-1029`, `:1063-1066`, `:1068-1078`). So node B
   (start 0.2, `dense_blocks` = S) over node A (start 0.4) should give "blocks
   in S dense until 0.4, others until 0.2". Risks: the node keeps one
   `sol_compose` dict per model, which a second node overwrites
   (`sol_attn_h3.py:1386`). That dict gates foreign forward patches, not the
   override chain, so the default chain is probably unaffected; untested
   either way. Measure first: one short render on a server armed with
   `H3_SOL_OBSERVE` (ask before restarting a shared server),
   reading the route rows. Both overrides record on the same call, so
   expect two rows per call, not one. Pass condition: for a block in S at a
   sigma between the two starts, B logs `dense_block` and A logs
   `outside_range`; past A's start, B logs `dense_block` and A logs `sol`;
   for a block outside S past B's start, B logs `sol` and A logs nothing.
   If it fails, the cheap node
   change is a per-block sigma start read in the depth gate at
   `sol_attn_h3.py:1063`. No kernel change either way.
4. **Build F.9 as a structured, multi-prompt, single-step injection:
   EpaCache's protocol with Sol as the perturbation.** [V for the protocol; I
   for the mapping] EpaCache injects the real approximation (cache reuse) at
   exactly one step, runs the rest exact, and records final-latent relative
   L1 over N=20 prompts. It reports N=1 as "highly prompt-dependent" and N=5
   as degrading fidelity (Tab. 5, "Additional Analysis"). The node can already inject Sol at
   one step (a sigma window around it) and one block (`dense_blocks` = the
   other 49), which is `bench/probe_block_propagation.py` moved off the final
   step. Record the low-frequency band of the deviation separately: HEART v2
   Table 1 finds equal-magnitude low-frequency velocity perturbations more
   harmful. At early steps, read a large deviation as a basin change (item 1),
   not as damage. Measure first: steps 1-4 at one block on at least ten
   prompts before any per-block sweep. This settles whether the early-step
   question and the per-block question are the same question.
5. **If `dense_blocks` is used at all, spread the blocks through depth.**
   [V, weak transfer] SpectralCache (Sec. 3.2, Fig. 2) finds that caching k
   consecutive blocks costs more error than k blocks spread out at the same
   rate, because each exact block re-anchors the residual stream. It is an
   image model (FLUX.1-schnell), 20 steps, one author and no venue listed on
   its abstract page, and
   caching is not Sol's error, but it is the only depth-layout evidence found.
   The front-clumped `0-2,32` never had this argument. Separately, sglang
   measured dense first layers as free to drop on H3
   (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:81-85`), contradicting ScalingAttention's
   "shallower layers ... most sensitive". Trust the H3 measurement.
6. **Per head: diagnose, do not ask for a kernel.** [V for sources; I for the
   plan] Sol's threshold is already per (head, query block) and adaptive
   (survey D.1), so part of sglang's "per-head budget" lever (a lever for
   uniform top-k routers, `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:63-68`) is already
   pulled by construction. The pack has per-head density and error records
   (`bench/results/2026-08-19_routing_density_per_head.json`,
   `2026-08-19_sol_error_per_head.json`, priced in
   `bench/results/2026-08-19_head_granularity_arms.json`). Measure first, CPU, existing
   capture: per head, Sol's retained exact mass at its routed density,
   against that head's attention entropy. If diffuse heads already get more
   blocks, the lever is spent. Add the data-free Autonomy-of-Heads statistic
   (effective rank of `W_K^T W_Q` per head, from frozen weights) as a free
   predictor to correlate. Only if both point at a few heads, and a render
   confirms, is a per-head tau worth a kitchen change.
7. **Step caching is the biggest lever and lives outside this node.** [V]
   Sana's H3 RTX 4090 record prices TeaCache far above Sol-Attn (survey D.5).
   Every caching schedule read here computes the opening steps and the last
   step exactly, and EasyCache composed with a sparse method (SVG) on
   HunyuanVideo (Sec. IV-E, Table VII). Relevant to `start_percent` only in
   that a cache would remove middle steps, where Sol is cheapest to be wrong.

## 1. Step-wise schedules

### 1.1 How the dense warm-up is set: convention against measurement

| work | warm-up | measured or borrowed | pointer |
|---|---|---|---|
| Sol-Attn | first 20% of steps + first layer (Wan, Hunyuan: 10 of 50; Bernini: 8 of 40). LTX: 8-step LoRA stage 1 **dense throughout**, stage 2 Sol with no warm-up | borrowed ("Following prior work"); ablation 4.5 covers thresholding, correction, pipeline only | 2607.24027 Sec. 4.1 Implementation Details; App. A |
| Sparse VideoGen | first 25% | borrowed ("following previous works"), no ablation | 2502.01776 Sec. 5.1 Parameters |
| Sparse-vDiT | first 10 steps for baselines, "not required for Sparse-vDiT on CogVideoX1.5" | borrowed for baselines; the claim for its own method has no table in what I read | 2506.03065 Sec. 5.1 |
| DFSAttn | first 25%; 4-step distilled: first step only | borrowed | 2605.23445 Implementation details; App. C.2 |
| Input-Stable (SVOO) | 20% Wan, 10% Hunyuan, first layer | borrowed ("unified warm-up") | 2603.18636 Sec. 5 |
| HEART v2 | first 5 of 50 | borrowed, justified as "crucial for preserving similarity to the original dense-generation outputs" | 2605.14513v2 Sec. 5.1 |
| HASTE (same id, v1) | none: "we do not apply dense warm-up steps" | a choice, no ablation | 2605.14513v1 Sec. 5, Experimental Setup |
| PASA | first 20% | borrowed ("Following prior setups"); its own curvature budget starts after | 2604.12219 Sec. 4.1 |
| MOD-DiT | 12 steps; ablated m in {8, 12, 16, 20} on CogVideoX-v1.5, VBench SubConsist and ImageQual; "m=12 ... near-optimal"; "reducing m to 8 ... marginal quality impact" | **measured** (step total not stated in the text I read [?]) | 2601.11641 Sec. 5; App. A.1.7, Fig. 10 |
| PISA | 15 (Wan 1.3B) or 10 (Wan 14B, Hunyuan) dense steps + 1 layer, against 0/0 | **measured**, the only with/without table across methods | 2602.01077 Tables 1, 2, 5 |
| STA (2025) | 12/6/3 dense steps at 50/25/10 steps | fraction held across step counts; T0 itself not ablated in what I read | 2502.04507 Sec. 4.3 |
| EasyCache (2025, caching) | R=10 of 50 on Wan | **measured**: R=2, 5, 10, 15 in Table IV; PSNR rises and speed falls with R | 2507.02860 Sec. IV, Table IV |
| ScalingAttention | none, but density graded by (layer, step) | **measured** against static-uniform with 30% warm-up | 2606.23019 Sec. 5.6, Table 5 |
| CalibAtt | none hard; energy threshold epsilon(t) decays from high to low | **fitted** by Optuna against VBench on 64 prompts | 2603.05503 Sec. 3.2 Eq. 7; App. 0.A.1 |
| sglang SubBlock (H3) | 10 of 50 steps, 0 layers | **measured on H3**: 10 to 5 halves cosine-vs-dense; 0 "essentially uncorrelated with dense"; dense layers 2 to 0 within noise | `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:77-85` |

**PISA is the closest evidence for Sol's family.** [V] PISA approximates
unselected blocks with a block-wise zeroth-order term plus a global
first-order term. Sol keeps a zeroth-order pooled tail. The two share their
first author. On Wan2.1-1.3B at 87.5% sparsity, PISA with warm-up reports
PSNR 22.62, I.Q. 65.94, A.Q. 60.03 (Table 1); without warm-up PSNR 14.16,
I.Q. 66.08, A.Q. 60.32 (Table 2). SpargeAttn, which drops the tail, goes from
PSNR 20.75 / I.Q. 64.89 to 10.92 / 60.10. The authors' reading: "unlike
competing methods that deteriorate precipitously without a warmup strategy,
PISA maintains high-fidelity generation". Their numbers support the narrower
reading that the sample moves while two VBench quality columns do not. Two
limits: Table 2 drops the Subject Consistency column that Table 1 reports,
which is the column most likely to catch temporal breakage from an early-step
perturbation; and PISA credits its robustness to "piecewise computation
covering the full attention span", that is, its global first-order term.
[I] Sol has the zeroth-order pooled tail and not the first-order term, so it
sits between PISA and SpargeAttn on exactly the axis that predicts warm-up
sensitivity. A quality-neutral basin change at `start_percent` 0.0 is the
optimistic end of the range; SpargeAttn's collapse is the pessimistic end.
F.4's panel decides which, and the owner should look first at subject and
temporal consistency in the 0.0 arm, the dimension PISA did not report.

**The basin argument, in the paper that states it.** [V] "Not All Tokens Need
40 Steps" (2605.06892, App. A): "If a schedule perturbs enough of that
early-stage computation, the trajectory commits to a different low-frequency
structure and the sample is pulled into a different basin: structurally
different, but still a valid generation of the same prompt ... two
perceptually strong, prompt-consistent samples can exhibit low PSNR and high
LPIPS simply because they settled on different plausible compositions." It
uses PSNR/LPIPS only "to diagnose whether a given schedule remains in the
reference basin" and draws quality conclusions from VBench. It also gives an
order: PSNR degrades first, SSIM next, LPIPS last. HEART v2's own reason for
adding warm-up in its revision is similarity to dense, not quality (table
above).

### 1.2 Measured criteria that set a boundary or a per-step budget

- **PASA** (2604.12219, Sec. 4.1) [V]: velocity L1 between adjacent steps,
  averaged over 10 calibration prompts into one offline curve. Three phases:
  "approximately the first 10 timesteps" with "remarkably high" variation, a
  "prolonged stable regime", and "a distinct resurgence" in the final 5. It
  sets per-step density `rho_t = rho * l_t / mean(l)` after a fixed 20%
  dense warm-up. PASA uses the signal for the budget after warm-up, not for
  the warm-up itself. Flow matching, like H3. No human study.
- **CalibAtt** (2603.05503) [V]: per (timestep, layer, head) block masks from
  cumulative block energy on 64 prompts, with a timestep threshold
  `epsilon(t) = A + (C - A) exp(-k t / T)`, t=0 the noisiest step (Eq. 7),
  "ranging from 0.99 to 0.84 in Wan". Fitted per configuration by Optuna
  against VBench (0.A.1). High-step fit: C = 0.99, k = 16, and a base A that
  grows with sequence length N (Eq. S11). Distilled 4-step fit: A = 0.763,
  C = 0.863, k = 5.64. It needs no hard warm-up and "does not require
  arbitrary exclusion of specific layers or timesteps". [I, labelled
  arithmetic] With k = 16, epsilon's excess over A falls to exp(-1.6), about
  a fifth, by t/T = 0.1 and to exp(-3.2), about 4%, by t/T = 0.2: the fitted
  "soft warm-up" is spent within the first tenth to fifth of the steps. Eq. S11 at H3's length gives a
  higher A than any configuration they fitted: extrapolation, do not use it.
- **ScalingAttention** (2606.23019) [V]: per (layer, step, head) sparsity
  chosen offline so a Hellinger-distance fidelity score meets a parametric
  target that is higher "to early denoising steps and shallower layers where
  error accumulation makes the model most sensitive" (Sec. 4.2). Topology
  from 27 prompts; fidelity profiling from one prompt, tested for stability
  across four prompt groups (Sec. 5.4, Table 2). Its "no warm-up" result
  (Table 5: PSNR 23.43 vs 17.83 at 45% density) compares its **graded**
  schedule against **static uniform plus 30% warm-up** at equal global
  density. It shows a graded schedule subsumes warm-up, not that early steps
  tolerate sparsity. Tested at 45-74% global density; Sol routes about a
  fifth to a quarter of blocks (`bench/results/2026-09-17_sol_stage_profile.md`), outside
  that range. No human study.
- **DFSAttn** (2605.23445, Sec. 4.1, Fig. 3) [V]: with block mean pooling
  (Sol's routing family), mass recall at fixed sparsity "consistently
  increases as the diffusion process proceeds", attributed to diffuse
  attention at high noise. Budget: 25% dense, then density 0.3 falling by
  0.1 every further 25% of steps.
- **LiteAttention** (2511.11062, Sec. 4.4, Table 1) [V]: a fixed sparsity
  threshold applied at one intermediate timestep of Wan2.1; the resulting
  attention error at the last step is 0.392, 0.375, 0.325, 0.318 for
  injection at steps 0, 16, 32, 48. "The earlier the timestep, the greater
  its influence". It then splits steps into three segments with tighter
  error bounds early. The metric is attention-output error at the last step,
  not the final latent.
- **HEART v2** (2605.14513v2, Sec. 3.2) [V, one line, method out of scope]:
  attention-output error and denoising-velocity error per head "are not
  equivalent", so a threshold acceptable on the first "may still produce a
  disproportionate error" on the second.

### 1.3 Step caching: what signal decides, and at what granularity

All decide per step for the whole model unless noted. None ran a human study.

| method | decision signal | calibration | forced exact steps | pointer |
|---|---|---|---|---|
| TeaCache (2024-11, CVPR 2025) | accumulated relative L1 of the timestep-modulated input, polynomial-rescaled | 70 prompts from T2V-CompBench | none stated in the paper; Sana's H3 port forces the first 5 and the last 1 | 2411.19108 Sec. 3.3, 4.1; `coderef/Sana/models/minimax_h3/RTX4090/teacache.py:45-46` |
| First Block Cache (ParaAttention, no paper) | change in the first block's residual; skip the rest of the stack if small | threshold `residual_diff_threshold` | not stated | ParaAttention README |
| TaylorSeer (2025) | forecasts features by finite-difference Taylor expansion instead of reusing them; full compute once per caching interval N | none | one full step per interval N | 2503.06923 Sec. 3 (error analysis in N), Sec. 4 |
| MagCache (NeurIPS 2025) | accumulated error from the ratio of residual magnitudes between steps | "a single random sample", robust across calibration sets (Table 3) | first 20% ("retention ratio"); ratio "drops sharply" in the last 20% | 2506.09045 Sec. 3.2, 4.1 |
| EasyCache (2025) | transformation rate k_t, runtime, no calibration | none | first R steps and the last step (Eq. 6); R ablated (Table IV) | 2507.02860 Sec. III |
| SeaCache (CVPR 2026) | redundancy on spectrally filtered inputs; "front-loads recomputation during the early denoising steps" per EpaCache's comparison | offline filter | abstract only | 2602.18993; 2608.29264 "Cache Decision Visualization" |
| NaviCache (2026) | predicted output variation via an inertial-navigation style filter, test-time | none | first N_align steps: 10 (Wan), 5 (Hunyuan, Open-Sora) | 2606.26795 Sec. 3.2.1, 4.1 |
| EpaCache (2026) | TeaCache-style local proxy against a **per-step threshold scaled by measured error propagation** | 20 prompts, one single-step injection run per step | first and last ("fully computed to prevent severe quality degradation") | 2608.29264 Sec. 3.2 |
| SpectralCache (2026) | cosine-bell threshold schedule over time, cumulative error budget over depth, frequency bands | FLUX | protects both ends | 2603.05315 Sec. 3-4 |
| ERTACache (ICLR 2026) | offline residual profiling; splits cache error into feature-shift and step-amplification terms | offline | abstract only | 2508.21091 |

One-liners, abstracts only: DPCache (2602.22654) plans which timesteps to
compute as a path; AdaCorrection (2602.13357) corrects cached offsets; TAP
(2603.03792) predicts per token. BWCache (2509.13789, 2025) reports block
feature change as "U-shaped" over timesteps. [V for the abstracts; nothing
else read]

### 1.4 Does the boundary depend on step count (16 base vs 8 PDD)?

- **Convention holds the fraction.** [V] STA: 12/6/3 of 50/25/10 (Sec. 4.3).
  Sol-Attn: 10 of 50 and 8 of 40. H3 vendors: sglang 10 of 50
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:79`),
  LightX2V 6 of 29 (`coderef/LightX2V/configs/minimax_h3/minimax_h3_sol_block_offload.json:23`)
  and 1 of 4 DMD (`coderef/LightX2V/configs/minimax_h3/dmd/minimax_h3_bf16_4step_sol.json:23`),
  Sana Sol-H3 1 of 4 forwards
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:116`).
  DFSAttn's 4-step distilled run keeps the first step dense (App. C.2). None
  of these ablates the choice. `workflows/h3_config.py` already notes that
  `start_percent` 0.2 is a flat quarter of evaluations at 16, 8 and 4 steps.
- **The one refit changes shape.** [V] CalibAtt's distilled 4-step fit starts
  at epsilon 0.863 and stays nearly flat. Its 50-step fit starts at 0.99 and
  decays fast (1.2). The fit on the distilled model did not ask for a
  near-dense first step.
- **Sol's authors did not sparsify an 8-step LoRA stage at all.** [V] LTX
  stage 1 "uses an 8-step LoRA with dense attention throughout"
  (2607.24027 Sec. 4.1). Stage 2, a refinement pass, runs Sol with no
  warm-up. That is a choice, not a measurement, and it may reflect stage
  1's short sequence rather than its step count. [I]
- **Other distilled results carry no warm-up ablation.** [V] ScalingAttention
  runs on 3-step FastWan (Sec. 5.5, Table 4); MOD-DiT on 8-step FastWan
  (App. A.4, Table 7, VBench only, warm-up count on that run not stated
  [?]); CalibAtt on 4-step LightX2V (Table 2).
- **Reading for PDD8.** [I] The literature cannot say whether 2 of 8 dense
  steps is right. Item 2's `k_t` curve on PDD8 is the cheapest model-derived
  answer. PDD's canonical sigma grid and fused heads
  (`workflows/h3_config.py` notes around `SOL_END_PERCENT_BY_STEPS`) make any
  50-step result a weak guide.

### 1.5 The last steps (`end_percent`)

- **Attention sparsity: late steps are the most tolerant.** [V] LiteAttention
  Table 1 (step 48 contributes least); CalibAtt's epsilon(t) lowest at the
  last step and late masks "highly similar" across steps (Sec. 3.1,
  Fig. S11); ScalingAttention Observation 2 ("robust later stages"); DFSAttn
  Fig. 3 (recall rises with step). Sol-Attn and sglang run to the last step,
  which is what this pack adopted on 2026-09-11 (`docs/SOLATTN.md` knob
  table).
- **Step caching: the last step is protected.** [V] EasyCache forces
  t = T-1; EpaCache computes the last step fully "to prevent severe quality
  degradation"; MagCache's ratio "drops sharply" in the last 20%; PASA sees a
  resurgence in the final 5 steps; SpectralCache's single-step injection on
  FLUX.1-schnell is U-shaped with "a moderate resurgence" at the end
  (Sec. 3.1, Fig. 1); Sol Video Inference Engine keeps "the first three and
  last three denoising steps" in high precision on Cosmos3 because "late
  steps refine high-frequency details" (App. A.2.1).
- **Why they disagree.** [I] Caching replaces the whole output with the
  previous step's, and the late output changes most, so reuse fails there.
  Sparse attention approximates attention within the current step, and late
  attention is concentrated, so sparsity costs least there. Evidence about
  one approximation does not transfer to the other. This is the "words do
  not disambiguate the stage" hazard, from a new direction.
- **Where H3 differs from every paper.** [V pointer, I reading] At 8 steps
  and shift 12 the last step spans the largest sigma interval of the
  schedule, and PDD's fused heads deviate most there (`workflows/h3_config.py`,
  the `SOL_END_PERCENT_BY_STEPS` note). No paper tests a last step that
  large. The existing propagation probe already isolates the final step
  (`bench/probe_block_propagation.py`, record
  `bench/results/2026-08-29_block_propagation.json`, base model, 4 steps).
  Running it on the PDD8 schedule is the cheapest check.

### 1.6 Step fraction is not sigma [I, labelled arithmetic]

ComfyUI's `percent_to_sigma` for a flow model is `time_snr_shift(shift, 1 -
percent)` with `time_snr_shift(a, t) = a t / (1 + (a - 1) t)`
(`comfy/model_sampling.py:289-292` in the ComfyUI checkout). For a schedule
uniform in t before shifting, a step fraction p maps the same way. So the
sigma at which a 20% warm-up ends is 0.980 at shift 12 (this pack; the PDD8
route record logs `sigma_start` 0.9796,
`bench/results/2026-09-01_sol_route_pdd8_chunked_ab.json`), 0.966 at shift 7,
0.952 at shift 5 and 0.923 at shift 3. STA's 3 of 10 steps at shift 17 ends
near 0.975. The literature's "20%" is a different point on the noise axis for
every model. `docs/SOLATTN.md` ("The sigma window is not a step fraction")
already shows the step-count side of this. Compare dense step counts and
sigmas, never percents.

## 2. Layer-wise and head-wise budgets set by profiling

### 2.1 Layers

- **Dense first layers are inherited.** [V] Sol-Attn, PISA and SVOO keep
  layer 0 dense by convention, and SVG the first two (as Sparse-vDiT
  describes it, Sec. 5.1; not found in SVG's own text). Sparse-vDiT calls it "unnecessary
  for Sparse-vDiT" (Sec. 5.1). sglang measured it free to drop on H3
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:81-85`).
- **Depth times step is one surface.** [V] ScalingAttention's target
  fidelity is a function of `l + omega t` (Eq. 5): earlier steps and
  shallower layers need more. CalibAtt calibrates a separate mask per
  (t, l, h). Delta-DiT (2406.01125, abstract, 2024) links front blocks to
  outline and rear blocks to detail. [I] Neither the node nor any paper here
  offers a per-block sigma window as a measured policy, but both papers'
  surfaces imply one. Item 3 is how to test it cheaply.
- **Depth layout of exact blocks.** [V] SpectralCache Sec. 3.2 (item 5).
  BAC (2506.13456, abstract, ICLR 2026, diffusion policy) reports caching
  errors that surge downstream through FFN blocks and updates upstream
  blocks first.

### 2.2 Heads in video DiTs

| work | classification | statistic | offline or online, how many inputs | transfers? |
|---|---|---|---|---|
| Sparse VideoGen (2025) | spatial or temporal head | MSE of each candidate mask against full attention on 1% of query rows (~3% overhead) | online, per head, per step | **no**: "a certain head can be a spatial head for one prompt while being a temporal head given another" (Sec. 4.1) |
| Sparse-vDiT (2025) | 5 modes incl. skip | MSE plus sparsity penalty | offline, "a small number of samples" | patterns "show limited dependence on the input content" (abstract); 3-6% of heads skippable, degradation at 10% (Sec. 4.1, Table 1) |
| STA (2025) | window size per head | mask-search loss averaged over 16 prompts | offline | "largely consistent across different prompts" (Sec. 3.2) |
| CalibAtt (2026) | per (t, l, h) block mask; "repetitive" heads | cumulative block energy; spatial row similarity | offline, 64 prompts | "attention patterns persist across inputs"; used at 480p and 720p; repetitive heads "cluster in the first and last layers" (App. 0.B.1) |
| ScalingAttention (2026) | per-head topology + per (l, t) sparsity | union of masks; Hellinger fidelity | offline, 27 prompts (topology), 1 (fidelity) | mask from 480p "aligns with the native 720p map with >96% recall"; stable under aspect-ratio change (Sec. 3) |
| HeadCast (2026, AR video) | sink, dummy, spatial, global | per-head output cosine of restricted-context proxies, worst-case score for spatial | online, once, at the max-noise step | ">90%" same archetype across AR steps; "similarly stable" across steps and prompts (Sec. 3.2) |
| Head Forcing (2026, AR video) | local, anchor, memory | profiling (abstract-level read) | offline | [?] |
| HEART v2 (2026) | per-head top-p threshold | frequency-weighted velocity error | offline | one line: per-head errors "non-separable" (Sec. 4.3) |

**The transfer split, reconciled.** [I] Papers that classify a head's
**shape** (which region it attends) find it stable across prompts and
resolutions. SVG classifies a **choice between two masks** by output error,
which flips with content. Stable patterns do not imply a stable per-head
**budget**. HeadCast's warning applies to any per-head statistic here:
"averaged cosine similarity can stay high even when a head fails at a few
boundary tokens ... residual ghosting". That is the local-versus-whole-call
error point of question review C.11, found independently.

### 2.3 Retrieval and streaming heads in LLMs

- **DuoAttention** (2410.10819, 2024) [V]: a retrieval head is defined by
  output, as a head that "significantly alter[s] model outputs when
  restricted to recent tokens and attention sinks" (Sec. 2.2). Found by
  optimizing one gate per KV head with the model frozen, on synthetic
  passkey data. It beats attention-score profiling, which "struggles to
  identify retrieval heads" (Sec. 3.5). Weights frozen, but it needs
  backprop through the model: on a 104k-token DiT on one 4090, not cheap.
  [I]
- **Autonomy-of-Heads** (2608.06849) [V]: data-free; low effective rank of
  `M_h = W_K^h^T W_Q^h` marks a retrieval head. Attention distance per head
  is "structurally stable across 4K-100K contexts". A RoPE-aware variant
  ranks heads "highly consistent" with the plain one (Sec. 6.4). [I] H3's
  weights are frozen and on disk, so this is a zero-data CPU statistic to
  correlate with Sol's per-head density (item 6). An LLM result; no video
  evidence.
- **BOSCH** (2604.05942, ACL 2026) [V, abstract and Sec. 1]: "a head's
  local/global behavior can change after hybridization", so independent
  per-head rankings mislead and it optimizes heads jointly.

### 2.4 The per-head budget lever against Sol's design

- sglang's note is about a **uniform top-k** router: "spending more blocks on
  diffuse heads and fewer on peaked ones lifts 5th-percentile mass recall"
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:63-68`).
  [V]
- Sol's `tau_i = mu_i + beta sigma_i` is already computed per head and query
  block (survey D.1, D.3), so counts already vary per head, by score spread
  rather than by mass. [V for the rule; I that part of the lever is spent]
- The counts are recorded per (block, step, head, query block) by
  `sol_observe.py` (module docstring, lines 10-18). The 2026-08-19 records
  measured per-head density and error and priced "top-k heads dense" arms
  against a global tau change. They found about the same local-error cut per
  unit of density, with error concentrated in a few heads per cell
  (`bench/results/2026-08-19_head_granularity_arms.json`). That was tau 1.3,
  one ref2va capture and local error only. [V pointer; I reading]

## 3. Sensitivity and propagation

**What 2026 work did, and what it found about depth and step.**

- **EpaCache** (2608.29264, Sec. 3.2) [V]: the structured single-step
  injection. For each step except the first and last, reuse the previous
  residual at that step only, run the rest exact, record final-latent
  relative L1 averaged over N=20 prompts. The profile is not monotone: "at
  some timesteps, error caused by caching propagates more rapidly and can
  lead to larger final deviations than reuse at their neighboring
  timesteps". Thresholds are then set inversely to the profile. Cost is
  about N x [T + (T-2)(T-1)/2] step-evaluations, cut by calibrating a subset
  of steps and interpolating (App. A-B). The per-step profile is plotted, not
  tabulated. No per-block decomposition.
- **SpectralCache** (2603.05315, Sec. 3.1-3.2) [V]: single-step forced reuse
  on FLUX.1-schnell, 20 steps: error 48.6 at t1, 13.5 by t5, under 7.0 across
  t6-t14, rising to 6.4 at t18. Depth: consecutive cached blocks cost more
  than spread ones (item 5).
- **LiteAttention** (Sec. 4.4, Table 1) [V]: the only single-step injection
  of **attention sparsity** found. The final attention error falls with the
  injection step, over a narrower range than the caching curves.
- **HEART v2** (Table 1) [V, one line]: equal-magnitude perturbations of the
  dense velocity in different 3D-FFT bands; "lower-frequency perturbations
  are more harmful".
- **DuoAttention** [V]: head importance as output deviation under a
  structured restriction (sink plus recent), optimized, not sampled.
- **sglang H3** [V]: an estimator that scored better on both single-layer
  proxies (mass recall, rel-L2) scored worse end to end, by 0.107 cosine
  against the dense render, paired t = -6.4 over 15 prompts
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:46-61`). This is the strongest local caution
  against any capture-only ranking.

**No paper found injects Sol-shaped error (drop the unrouted blocks, keep a
pooled tail) at one (block, step) cell of a video DiT and follows it to the
output.** [?] Searched arXiv via web search for error-propagation,
sensitivity and perturbation work on DiT caching and sparse attention in
2026; the closest are EpaCache (steps, caching) and LiteAttention (steps,
attention, attention-output metric). The node can run exactly that
experiment (item 4). The literature adds three design rules. [I]

1. Inject the real approximation, not Gaussian noise. EpaCache,
   SpectralCache and LiteAttention do; HEART's frequency-band test is the
   exception, used only to motivate a weighting. This settles question
   review F.9's caveat.
2. Use ten or more prompts. EpaCache's Table 5 shows one-prompt profiles are
   unreliable. The pack's existing propagation record is one seed.
3. Split the deviation by frequency, and read early-step deviation as basin
   change. HEART's Table 1 and "Not All Tokens" App. A are the grounds.

## 4. Validation against human judgement

| method | human study | automatic metrics | agreement reported |
|---|---|---|---|
| STA (2025) | **yes**: 200 MovieGen Bench prompts, pairwise, "higher overall quality or ... tie"; training-free arm 83.0% ties against original HunyuanVideo, win rate 7.0 points below loss rate (Sec. 4.2, Fig. 7) | VBench, SSIM, PSNR, CD-FVD | "we find existing automated metrics are often unreliable" (Sec. 4); no correlation computed |
| Sol Video Inference Engine (2026) | informal validator in the loop; no rater or clip counts | VBench | PSNR "may assign a large distance to outputs with shifted visual details, while assigning a smaller distance to outputs with blur, temporal jitter, or degraded motion coherence" (Sec. 4.1) |
| VSA (2025, trainable) | reported in abstract-level sources; not read [?] | VBench | [?] |
| Not All Tokens (2026) | no | VBench for quality, PSNR/LPIPS for basin only | argues same-seed metrics are incoherent after a basin change (App. A) |
| Sol-Attn, PISA, CalibAtt, ScalingAttention, PASA, DFSAttn, MOD-DiT, HEART, LiteAttention, SVG, Sparse-vDiT, EpaCache, NaviCache, MagCache, EasyCache, TeaCache | **no** | VBench and/or PSNR/SSIM/LPIPS | none measured |
| sglang SubBlock on H3 | no; "visibly re-frames the shot" is an informal look | cosine of decoded video against dense, paired t over 15 prompts | none measured |

[I] The field validates on VBench and same-seed similarity. Neither answers
the owner's question at `start_percent`, where the sample moves. The pack's
blind panel with decoys is stricter than anything published here. Its gap is
volume, which is why the ranked items keep the panel for the final call and
use model-derived curves (item 2) and structured injection (item 4) to
choose what reaches it.

## 5. What the node can express

From `docs/SOLATTN.md`, `sol_attn_h3.py` and question review D: per block
(`dense_blocks`, `tau_profile`, `token_aug_blocks`), one global sigma window
(`start_percent`, `end_percent`), per head recordable only.

| finding | expressible today | needs a per-block sigma window (node change) | needs per-head control (kernel change) |
|---|---|---|---|
| Measured warm-up length (PASA and EasyCache curves; MOD-DiT; PISA) | yes: `start_percent` | | |
| Graded density over steps (DFSAttn; CalibAtt; PASA budget) | coarsely: one tau per window; several windows only if chained nodes compose (item 3) | a per-sigma tau is the clean form | |
| Depth-by-step surface (ScalingAttention; CalibAtt) | per-block tau, fixed over steps | yes, unless the two-node chain works | |
| Spread-out exact blocks (SpectralCache) | yes: `dense_blocks` | | |
| Last-step protection (caching papers) | yes: `end_percent` (attention evidence says not needed) | | |
| Per-head budget, retrieval-like heads (sglang note; DuoAttention; AoH; HEART EBC) | diagnose only: `sol_observe` per-head counts, captures | | yes: a per-head tau in the kitchen kernel |
| Skippable heads (Sparse-vDiT) | no | | yes, and a quality claim no H3 evidence supports |

The gate order matters. `dense_blocks` is checked before the sigma window
(`sol_attn_h3.py:1063` before `:1068`), so a block in `dense_blocks` is dense
at every step, unless a chained node below it takes the call.

## Sources

Papers (arXiv ids verified on their abstract pages 2026-09-19):

- Sol-Attn: https://arxiv.org/abs/2607.24027
- Sol Video Inference Engine: https://arxiv.org/abs/2606.23743
- PISA: https://arxiv.org/abs/2602.01077
- CalibAtt (Accelerating Text-to-Video Generation with Calibrated Sparse Attention): https://arxiv.org/abs/2603.05503
- ScalingAttention: https://arxiv.org/abs/2606.23019
- Ride the Wave (PASA): https://arxiv.org/abs/2604.12219
- MOD-DiT (Mixture of Distributions Matters): https://arxiv.org/abs/2601.11641
- DFSAttn: https://arxiv.org/abs/2605.23445
- EpaCache: https://arxiv.org/abs/2608.29264
- SpectralCache (Frequency-Aware Error-Bounded Caching): https://arxiv.org/abs/2603.05315
- NaviCache: https://arxiv.org/abs/2606.26795
- SeaCache (abstract): https://arxiv.org/abs/2602.18993
- ERTACache (abstract): https://arxiv.org/abs/2508.21091
- LiteAttention: https://arxiv.org/abs/2511.11062
- Not All Tokens Need 40 Steps: https://arxiv.org/abs/2605.06892
- HEART (v2) / HASTE (v1): https://arxiv.org/abs/2605.14513v2, https://arxiv.org/abs/2605.14513v1
- Attention Sparsity is Input-Stable (one line): https://arxiv.org/abs/2603.18636
- Cumulative energy filtering (one line): https://arxiv.org/abs/2606.16317
- EasyCache: https://arxiv.org/abs/2507.02860
- MagCache: https://arxiv.org/abs/2506.09045
- TeaCache: https://arxiv.org/abs/2411.19108
- TaylorSeer: https://arxiv.org/abs/2503.06923
- First Block Cache: https://github.com/chengzeyi/ParaAttention (README)
- Sparse VideoGen: https://arxiv.org/abs/2502.01776
- Sparse-vDiT: https://arxiv.org/abs/2506.03065
- STA (Sliding Tile Attention): https://arxiv.org/abs/2502.04507
- DuoAttention: https://arxiv.org/abs/2410.10819
- Autonomy-of-Heads: https://arxiv.org/abs/2608.06849
- BOSCH (abstract): https://arxiv.org/abs/2604.05942
- HeadCast: https://arxiv.org/abs/2607.20125
- Head Forcing (abstract): https://arxiv.org/abs/2605.14487
- SPADE (abstract): https://arxiv.org/abs/2608.03335
- Delta-DiT (abstract): https://arxiv.org/abs/2406.01125
- BAC (abstract): https://arxiv.org/abs/2506.13456
- BWCache (abstract): https://arxiv.org/abs/2509.13789
- DPCache, AdaCorrection, TAP (abstracts): https://arxiv.org/abs/2602.22654, https://arxiv.org/abs/2602.13357, https://arxiv.org/abs/2603.03792
- VSA (abstract): https://arxiv.org/abs/2505.13389

Local code and records:

- `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:24-68`
- `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:77-85`
- `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:79-80`
- `coderef/LightX2V/configs/minimax_h3/minimax_h3_sol_block_offload.json:23`
- `coderef/LightX2V/configs/minimax_h3/dmd/minimax_h3_bf16_4step_sol.json:23`
- `coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:116`
- `coderef/Sana/models/minimax_h3/RTX4090/teacache.py:45-46`
- `sol_attn_h3.py:1028-1029, 1063-1078, 1386`; `sol_observe.py:10-18`
- `workflows/h3_config.py` (`start_percent` note in `SOL_RECOMMENDED_CUDA`; `SOL_END_PERCENT_BY_STEPS` note)
- `docs/SOLATTN.md` ("The sigma window is not a step fraction"; knob table)
- `bench/results/2026-09-17_sol_start_percent_batch.md`, `bench/results/2026-09-17_sol_stage_profile.md`, `bench/results/2026-08-29_block_propagation.json`, `bench/results/2026-09-01_sol_route_pdd8_chunked_ab.json`, `bench/results/2026-08-19_head_granularity_arms.json`, `bench/results/2026-08-19_routing_density_per_head.json`, `2026-08-19_sol_error_per_head.json`
