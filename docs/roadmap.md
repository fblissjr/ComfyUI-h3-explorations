# Roadmap: what we are trying to find out, and what would count as finding it

Last updated: 2026-08-23.

## What this is, and who it is for

A working plan for the H3 speed-and-quality investigation, written so a fresh
session can pick it up without re-deriving the constraints or re-litigating
decisions that are already made.

**The context matters and shapes everything below.** This is one person
tinkering on a single RTX 4090 with 24 GB, finding out what is possible with
MiniMax H3. It is not a research programme and it is not production R&D. There
is no team, no cluster, no deadline, and no obligation to be exhaustive.

The practical consequence: **card time is the scarce resource and rigour is
cheap.** A measurement that takes a minute of Python and settles something is
worth more than an afternoon of renders that produces a number nobody can
defend. Most of the good results in this repo came from that trade -- the
canvas cost table, the Morton geometry, the checkpoint internals, the MMA rates
all needed zero or one render.

Two companion documents, and the split matters:
`docs/open_experiments.md` is the list of things **not measured** and why.
`docs/evidence.md` is the ledger of things measured that **should not be
relied on**. This file is the forward-looking one: what we are doing next and
what would make it a real answer.

## Current forward plan — 2026-08-23

The typed-reference migration is complete. Base-guide alignment is exact, the
ordered resolver admits explicit VHS sources and traces branched audio, all 38
shipped reference API graphs now compile append chains, and both runtime and
static-consumer controls are green. The all-media smoke rendered 1024x768 at
39 frames and 10 steps in 84.51 seconds, with the server logging the expected
five presentation labels and 21,283 packed rows.

Completed this slice:

1. Repointed every generated reference workflow from native core's parallel
   sockets to this repo's typed append/compiler nodes, preserving the legacy
   presentation order, `force_rate=24`, and existing sizing policy.
2. Removed explicit reference-audio trims from those workflows. The local
   typed compiler now derives the cap from aligned frame count and normalizes
   mono; native ComfyUI remains unchanged, so these are locally handled gaps,
   not native fixes.
3. Passed prompt, guide, ordering, typed-consumer, bounds, preflight, live API,
   and UI-schema checks for the migrated population. The eight full-schema red
   rows were 768p Turbo LoRA filenames absent on this install; resolved
   2026-08-23 by moving the 768p arm to v1.1, the file that is present.
4. Fixed VHS `LazyAudioMap` compatibility found by the first all-media smoke,
   then completed the rerun. The native tokenizer already had all twenty tokens,
   so this repo's compatibility shim logged a no-op.

The next policy slice is also complete:

5. `MiniMaxH3ReferenceConditioning.video_policy=release` is one opt-in switch
   for both release-video stages: the full-rate VAE view is put on the release
   canvas, while the raw 2 fps Qwen samples go through the release's
   duration-aware video processor. Generated graphs now use `encoder`: the
   selected encoder artifact's snapshotted Qwen stage with Comfy-compatible
   no-upscale VAE sizing. The two snapshots agree today but remain separate
   authorities and are tested by deliberate disagreement.
   `comfy` remains the native preprocessing control.
6. Added `h3_probe_release_video_policy` against the otherwise-matched
   `h3_ref_video_audio` graph. Preflight marks which VAE and Qwen rows are
   active and labels the release path as a local typed policy, not a native
   fix.
7. The 39-frame live acceptance rendered in 92.73 seconds. The server logged
   the 960x544 source becoming a 1344x768 VAE reference, four raw Qwen samples,
   23,892 packed rows, Sage routing, and Sol sparse execution. The CPU control
   separately pins the long-duration 31-versus-32-sample boundary; its red
   mutations collapse Qwen back onto the VAE frames and substitute release
   settings for encoder settings.

Next, in order:

1. Later cleanup: retire `vendor_tokens` from generated workflow inputs now
   that merged ComfyUI PR 15808 supplies the tokens natively. First set and
   verify the minimum supported ComfyUI version; until then, keep the helper as
   backward compatibility for older installs. This is cleanup, not a current
   conditioning or migration blocker.
2. GPU experiments: the FL2VA base-versus-turbo pair,
   the singing-removed speaker-attribution scene, and `ncu` profiling all need
   the card alone.

The historical lists below remain evidence of how priorities arrived; this
block is the current authority when they conflict.

---

## The left rail: constraints

Hard, and not worth arguing with.

**One 4090, 24,564 MiB, shared with actual use.** The heaviest shipped config
peaks at 21,186 MiB. Nothing runs beside a render. `ncu` needs the card alone.

**Renders are the expensive thing, and there are now two regimes.** The base
model at 16 steps is roughly 12 minutes at full length, so a five-point sweep
at two runs is about two hours. A 4-step lightx2v student at the same canvas
and length was 169 s end to end on 2026-08-20
(`bench/results/2026-08-20_sla_arms.jsonl`, at a 330 W power limit), which is
what makes the distribution standard below affordable. This is the binding
constraint on every quality question, and which regime you are in decides
what a day can hold.

**Length: 362 is the ceiling** (`h3_rules.MAX_LENGTH`, the longest H3 was
trained on). Frame counts snap to a 17n+5 grid. The old "345 is the maximum"
claim was invented and is withdrawn -- do not reintroduce it. There is **no
fixed reference-video length**: OOM risk depends on reference count, size,
resolution and canvas, and no single frame count describes it.

**Canvas: the area cap means wider is free and buys nothing.** Every ratio at
or above 1.75 costs the same ~1008 tokens/frame. The only cheap direction is
toward square, on a smooth 32px ramp.

**Kernel:** Sol-Attn needs bf16, head_dim 128, sm_80+. Its eager reference is
O(T²) and refuses past 4 GiB, so it can never run at production length --
every correctness number is at a toy size unless something replaces it.

**No test suite.** Verification is a live ComfyUI plus a GPU. A code change
means a restart before it counts. `bench/smoke_h3.py` is the only thing that
submits a prompt; a green static validator on an unsubmitted graph is
unverified.

**Shared working tree.** Other sessions work here. Stage by path, never
`git add -A`, and check `git status` before committing.

---

## The right rail: what "good" means

The thing this project keeps getting wrong is not measurement, it is
**deciding in advance what would count as an answer**. So, per axis:

### Speed

**Good = a ratio between two arms that differ in one variable, at a length
above the token floor, on the workload actually run, with the build recorded.**

Every one of those clauses has failed here at least once. Ratios taken against
a baseline nobody ships (the retracted 1.611x), at a length that misrepresents
the config, on t2v when the work is reference-heavy, with an unrecorded
kernel build.

The floor matters specifically: Sol-Attn needs **roughly 60k tokens** before
it shows anything. Below that a null result reads as "this knob does nothing"
and is indistinguishable from a real negative.

### Accuracy and correctness

**Good = graded against an independent implementation, on inputs with the
right distribution, with a control that has been shown to fail.**

Synthetic `torch.randn` is not the right distribution: it gives a near-uniform
softmax, so a block-sparse router has nothing to find and the premise of the
method is absent. That single fact retired every accuracy figure this repo
used to quote.

A check is untrusted until it has gone red for the right reason. Prefer a
control the check compares against over asserting numbers computed in the test
itself.

### Quality

**Good = a person watched a clip end to end and said what they saw.**

There is no output-quality instrument here and building one is not on the
list. `docs/SOLATTN.md`'s Quality section is marked REOPENED because every
judgement in it came from grids of stills, and the failure mode that matters
-- a small persistent object dissolving over about four frames -- is temporal
and cannot be seen that way.

Numbers can rank kernels. Only watching can say whether any of it reached the
picture. **This is the axis nothing else substitutes for**, and it is the one
that costs the scarce resource.

### Provenance

**Good = the record says which artifact produced the number.**

The kernel build tag, the graph, the length, the canvas, the model path. The
fork build of `comfy_kitchen` declares a version identical to the stock wheel,
so without the local tag nothing distinguishes them.

---

## The dials, and where they sit

All in `workflows/h3_config.py`, all single switches, all regenerate with
`python workflows/build_workflows.py`.

| dial | default | range | cost effect |
|---|---|---|---|
| `CANVAS_TIER` | `full` | full / near / fast / draft | 1.00 / 0.91 / 0.73 / 0.58 attention |
| length | 362 | 17n+5 grid | linear in tokens |
| `SOL_RECOMMENDED_CUDA` / `SOL_CUDA_DEFAULTS` | Sol **on** in every video graph since 2026-08-14; `tau` 1.0 from 2026-08-20 by owner decision | tau, window, sinks | see `docs/SOLATTN.md`; 1.3 returns only if it shows no difference from 1.0 on the distilled LoRAs while buying meaningful speed |
| `TURBO_LORA` / `TURBO_768P_*` / `TURBO_SLA_*` | none shipped by default; probe graphs | the lightx2v rows `bench/check_distill_settings.py` attests | 4-8 steps against 16 |
| `CACHE_NODE` | probe graphs only, **not canonical** (owner decision 2026-08-20) | EasyCache threshold/window | 1.56-1.74x on deterministic samplers at 16 steps; a 16-step lever with nothing to skip at 4 |

**Iterate at `fast` (1152x768) and 243 frames; confirm at `full` and 362.**
`fast` is exact 3:2, 27% off attention, and only 0.25 of ratio from the
shipped canvas, so framing reads the same. At 243 frames it is 62,208 tokens
-- just above the Sol floor. `draft` is 55,296 and **below** it, so `draft` is
for "does the pipeline run", never for a Sol measurement.

---

## What is established

Evidence grade is inside each claim, because a trailing "(unverified)" is the
part that gets trimmed when a sentence is copied forward.

**Canvas cost is fully mapped.** MEASURED by enumerating `adapt_canvas` over
the legal ratio range: 95 distinct resolutions, every ratio at or above 1.75
costing an identical 1008 tokens/frame because the area cap binds and trades
width for height. Twenty legal landscape canvases sit between 1:1 and 16:9,
about 0.04 apart in ratio, and attention goes as the *square* of tokens. Three
of the four shipped tiers hit a common ratio exactly, which the default
(1.75) does not. All four verified as stable fixed points.

**The HF "hybrid" checkpoints are an adaln swap, and we can build them
locally.** MEASURED 2026-08-16 by byte-comparing
`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models` against both parents, every
probe with a control confirming the parents actually differ there. The
community framing is "hybrid models"; the reality is narrower:

- Only `blocks.{N..49}.adaln_proj.linear.{weight,bias}` are taken from ref2va.
  **Everything else is 100% fl2va** -- attention, MLP, norms, `final_layer`.
  The filename range is which blocks take ref2va's adaln (`b20-49` = blocks
  20-49, boundary confirmed at 19/20).
- They are **already `int8_convrot`**, decoded from the file's own
  `comfy_quant` tensor. Same format and key set we run, so no conversion and
  no VRAM change -- and no bf16 version exists or can be derived from them.
- Net effect on delivered modulation is **1.1-4.7%**. The adaln *weight* is
  strongly anti-correlated between parents (cos -0.71 to -0.83) but the *bias*
  is cos ~0.9995 and dominates. It is a small broad perturbation, only
  ~1.3-2x more targeted at reference rows than at ordinary video rows.

  > **2026-08-20: the anti-correlation is an artifact, not a real feature
  > the bias happens to mask.** The two checkpoints' `adaln_t_table` bases
  > agree in sign on columns 0-3 and are sign-flipped on columns 4-7
  > (per-column cosine +1, +1, +0.996, +0.996, -0.9997, -0.9997, -0.99,
  > -0.99), and the coefficient columns on the flipped basis columns are
  > the large-norm ones. Compared at the modulation output, where the
  > basis is applied, the *time-varying* part of the modulation differs
  > 5-9% per block with cosine 0.996-0.999, and the whole output 1.4-4.7%
  > -- the same 1.1-4.7% this bullet measured, now without a mechanism
  > that was wrong. `bench/results/2026-08-20_dit_internals.json`.
- **It cannot be a reference-pathway transplant.** `adaln_proj`'s input is an
  8-dim timestep coordinate from the shared `adaln_t_table`; reference rows are
  distinguished only by being pinned at `VISUAL_COND_TIMESTEP`. It carries no
  reference image content.
- It deliberately leaves `final_layer.adaln_proj` on fl2va -- ~~the single
  most-rewritten tensor between the parents (rel_delta 1.92)~~ **withdrawn
  2026-08-20**: same coefficient-level artifact as above. At the modulation
  output the final layer's time-varying part differs ~12%, the largest
  single adaln item but not "rewritten".

**So it swaps adaln in four fixed steps.**
~~Both routes independently agree that adaln is the only thing meaningfully
different between the parents.~~ **Withdrawn 2026-08-20.** The int8 linears
differ ~3.2% relative between the parents and the modulation output 1.4-4.7%;
nothing at the weight level singles adaln out, and the hybrids keep fl2va's
linears, which differ from ref2va's by as much as the adaln they swap. The
2026-08-16 byte-comparison and the 1.1-4.7% figure stand; the reading that
adaln was *the* difference does not. Evidence behind the community praise is
three comments, no samples, no numbers.

**Not worth the 84 GB**: any variant is reproducible from files already on disk
by copying those tensors between two safetensors. The discrete block-selective
blend *is* a different interpolation path from a uniform LoRA strength, so it
remains a distinct experiment if anyone wants it -- just a local one.

**`subject_definitions` overrides the reference image. Two independent
instances.** MEASURED 2026-08-16, and this is the most transferable thing
learned this session because it is about prompting, not about a model.

`_ref_prompt()` emits a generic template asserting that Subject 2's
"**architecture**, palette, and lighting are carried into the target video".
The reference used here (`2-mountain_landscape.png`) is an alpine lake with
snow peaks, conifers and a wildflower meadow -- **no buildings, no structures,
nothing architectural**. Both renders put the man inside a building looking out
through a window. Neither could have got that from the image; both got it from
the definition. Confirmed independently by Gemini, blind, from the clips and
the reference alone.

That is the second instance of the pattern. The first, from the owner's own
prompting work, is a brunette reference described as blonde in
`subject_definitions` producing a blonde output despite `retention_analysis`
saying `fully_preserved`. Different attribute type, different session, one of
them accidental. **`subject_definitions` is binding authority and
`retention_analysis` cannot correct it.** A prompt that asserts an attribute
the reference lacks will get that attribute hallucinated rather than dropped.

The same prompt also specifies "steady interior room tone" over an outdoor
scene, and carries two contradictory lighting instructions. All three defects
come from one generic template pasted onto references it does not match.

**Attention is not very sparse on this workload.** MEASURED on captured
activations: at its most concentrated a query still needs 178 key blocks of
591 to hold 90% of its mass, and 394 at the first block. That bounds what any
block-sparse method can save here.

**Tensor-core rates on this box.** MEASURED, `bench/mma_rate.cu`: int8
m16n8k32 at 334.5 TMAC/s against bf16/f16 m16n8k16 f32-accumulate at 83.8
(0.25x) and f16-accumulate at 167.3 (0.50x). f32-accumulate forms issue at
half rate on sm_89, as upstream claims for sm_120.

---

## What is not established

| question | why it matters | blocker |
|---|---|---|
| Does fl2va+LoRA render the same as ref2va? | the LoRA is canonical on 18 graphs | **Rendered and judged 2026-08-16 -- partially answered, see below. Identity: equivalent. Scene coherence: ref2va ahead in one pair. Re-run owed on a sound prompt.** |
| Is the Sol exact kernel MMA-bound or staging-bound? | decides whether a 16-bit PV costs 2.5x or much less | an idle card, `ncu` |
| Sparsity error against quantization error, on real activations | if quantization is small, a 16-bit PV buys nothing | **Done 2026-08-17/19**, item 2 below; quantization is not small, and on 2026-08-20 it was found to track Q/K magnitude rather than V (`docs/open_experiments.md` #17b) |
| Routed density at production length | nobody knows how much of Sol's work the exact branch is | **Done 2026-08-19**, `bench/sweep_routing_density.py`; flat per block and per step, structured per head, priced empty |
| `start_percent` / `end_percent` at any length | zero measurements ever | card time |
| Anything on the reference-heavy workload | every Sol number is t2v on fl2va | **Closed 2026-08-17**: four reference-heavy captures now exist (2026-08-17 fl2va+ref LoRA, its multistep sibling, 2026-08-18 ref2va, 2026-08-20 clean fl2va), and every error number since is on them |
| What the 330 W power limit costs | decides whether 330 W timings compare to the 450 W records, and the bandwidth-bound reading in `docs/hardware.md` | one `sudo`; run 2026-08-20, `bench/results/2026-08-20_power_limit_pair.jsonl` -- verdict below when written |

**Power, stated once.** No 330-vs-450 W pair was ever run before 2026-08-20; it
was a forward item three times and blocked on sudo each time. The 18th's
records are at 450 W, the 2026-08-20 morning's at 330 W, and ratios within a
day hold while absolute seconds across that boundary do not. The decision rule
written on 2026-08-17: a delta under ~2% supports the bandwidth-bound reading;
a delta near the clock delta supports L2-bound.

**Run 2026-08-20** on the 4-step 768p turbo graph (fl2va + v1.0 LoRA,
1344x768 x 362, Sol at tau 1.0), two timed runs per limit with disjoint seed
bases, `bench/results/2026-08-20_power_limit_pair.jsonl` and its verdict
sidecar `_verdict.json`: sampler 134.1 s at 330 W against 126.8 s at 450 W,
within-arm spread 0.2%, so the cap costs **5.8%** -- against a **12.5%**
core-clock delta (p50 2145 against 2452 MHz during sampling, power pegged at
328 and 446 W). Between the two readings: sampling at this workload is partly
core-clock-bound, neither cleanly bandwidth- nor L2-bound. Stated for this
4-step t2v workload; the 16-step all-refs pair remains unrun.

---

## The regime question (opened 2026-08-20)

Every speed, caching, Sol-window and capture fact in this file was measured on
the base model at 16 steps. The lightx2v 4-step students change the economics
by about 4x and the owner is moving fl2va-base work onto them. What that does
to the existing record, in three groups:

- **Survives unchanged:** canvas cost and token maths; the rotary-clock facts;
  fl2va vs ref2va differing by a few percent everywhere; the block-49
  loud-channel structure (it lives in `k_norm` gains the LoRAs do not touch);
  the different-sample rule, which gets stronger at 4 steps; every provenance
  and check rule.
- **Adjusts:** Sol's sigma window selects steps by sigma, so the "5 of 16"
  sage fraction becomes a different fraction at 4-6 steps (arithmetic from the
  shift-6 grid, not measured); the captures are at steps 3 and 11 of 16 and a
  student visits different states, so Sol's per-call error on the distilled
  trajectory is unmeasured until the 2026-08-20 captures are read; the
  step count and shift are set by the LoRA's vendor row and graded by
  `bench/check_distill_settings.py`.
- **Dies:** step caching. The 1.74x was 7 of 16 steps skipped on a
  deterministic sampler; at 4 steps there is nothing to skip. `CACHE_NODE`
  stays a probe and is not canonical (owner decision 2026-08-20), which closes
  the "record a verdict beside `CACHE_NODE`" forward item of the 2026-08-18
  and 2026-08-19 postmortems.

The day's plan and verdicts land here as they are measured (power pair; LoRA
file session; reference transfer on fl2va, the HF b30-49 hybrid, a locally
built all-adaln hybrid and ref2va; the SLA LoRA under its training router;
captures on the distilled trajectory). Scope for the first pass is t2v and
1-3 reference images; no video or audio references.

**Verdicts, 2026-08-20, as they landed:**

- *Power*: the 330 W cap costs 5.8% against a 12.5% clock delta on the 4-step
  workload (above). 450 W stays.
- *Distilled-trajectory captures*: Sol's error and routed density are the same
  on the SLA student's activations as on v1.1's, within a few percent and
  half a point respectively (`docs/open_experiments.md` #20). SLA changes
  nothing Sol can see at the call.
- *Reference transfer, single seed, briefs met*: with the v1.1 LoRA patched
  in, the three reference images (performer, jersey, loft) survive on **all
  four** checkpoints -- fl2va, HF b30-49, the local all-adaln hybrid, and
  ref2va -- at the same seed (`bench/results/2026-08-20_ref_transfer_single.jsonl`;
  stills at 7 s). The predicted failure on ref2va did not appear, so the
  community claim that an fl2v distill cannot blend references on ref2va is
  not reproduced at one seed with v1.1. Observed beside it: fl2va and b30-49
  render near-identical compositions (curtain backdrop), while the all-adaln
  hybrid and ref2va both move to a wood-panel wall with artwork -- the
  modulation swap changes the sample's composition where the linears do not.
  ref2va also ran fastest on this request (sampler 108 s against 124-125 s
  for the three fl2va-linear arms), consistent with its lower Sol sparsity
  error on reference-heavy input earlier today. One seed; a distribution is
  the next step if the question matters.
- *Session 1, LoRA file, 8 matched seeds per contest, judged blind by the
  owner on free text, tags and a coarse verdict
  (`bench/results/2026-08-20_session1_verdict.json`; rows in
  `2026-08-20_session1_lora_file.jsonl`)*: **v1.1 against SLA,
  indistinguishable** (same 3, can't tell 4, v1.1 once) -- the notes name a
  bag colour, a helmet logo, a look direction. **v1.1 against the vendor
  recipe (simple, strength 1.0), indistinguishable** (same 2, can't tell 5,
  vendor once) while the vendor recipe runs ~20% less sampler time (126 s
  against 151 s) -- the owner's beta/0.75 recipe has no blind support at this
  sample. **v1.0 against v1.1, a lean to v1.0, 4 to 2** with one same and one
  can't tell; the notes put v1.0 ahead on lighting, reflections and framing in
  most pairs and behind on motion in two (wheels without pedalling, a
  background that stops while she rides), with v1.1 carrying two scene
  defects. A preference over distributions, not a decision; a second session
  at the vendor recipe would separate look from motion.
- *SLA regime set, one seed, timing is the result
  (`bench/results/2026-08-20_sla_regime_arms.jsonl`)*: the SLA LoRA under its
  own router 155 s of sampler, under Sol at tau 1.0 126 s, under sage alone
  (no sparse attention) 201 s; v1.1 under the same three 155 / 126 / 201 s, and
  v1.1 under Sol at tau 1.3 118 s. Both router arms render coherent clips (stills
  at 2, 7, 12 s), so the vendored kernel works end to end -- after one failed
  render on output contiguity, fixed in 84ba741 and left in the record. The
  router is slower than Sol here: a 64/64 Triton kernel with three contiguous
  copies against a tuned CUDA kernel keeping about a fifth of the blocks.
  Sol at 1.0 is 1.6x over sage-only at 4 steps; 1.3 buys a further 6.4%,
  quality half unmeasured. SLA under its own router looks like v1.1 under the
  same router at this seed; nothing here makes the SLA release the one to run.

**Verdicts, 2026-08-22:**

- *Sol selection, upstream's question*: `top-k` beats `adaptive tau` at 1.0 by
  4.7% (keep 15%) and 10.3% (keep 10%), and the LoRA does not matter -- the SLA
  student and v1.1 land within 0.06s at every selection.
  `docs/SOLATTN.md` owns it. **An earlier run the same day said the opposite
  and is void**; `docs/evidence.md` carries both that row and the rule it
  produced, which is that a long-lived ComfyUI session is not a substrate to
  time on. Two independent measures moved together across a restart on
  identical inputs.
- *Speaker attribution across a cut*: **the first real quality finding the new
  scene set bought.** Asked for a busker singing in Shot 1 and two different
  commuters speaking after a cut at 3.5s, 11 of 12 clips hand the singing to a
  character introduced after the cut. One clip gets it right. It spans both
  selections and both LoRAs, so it is a model-and-prompt finding rather than an
  attention one, and no prompt shipped here before 2026-08-22 could have
  surfaced it -- none had two speakers, a cut, or singing.
  `bench/results/2026-08-22_subway_speaker_bleed.json`. The owner's second
  judgement stands beside it: all twelve look and sound poor regardless, so
  the one that is correct on this point is not thereby a good clip.
  **The cheapest next test is named in that file** -- the same scene with the
  singing removed, which scopes the failure to singing or to cuts.
- *The audio hum*: not Sol, not the kernel, not the tokenizer. Sol on, Sol off
  and the SLA router hum identically; what moves it is the PROMPT asking for
  continuous texture ("tyre hiss through standing water") and 4-step
  distillation amplifying it. `bench/results/2026-08-22_audio_hum.json`.

## CLOSED 2026-08-16: token ordering as a quality question

**Stop working on which curve to use. This is a decision, not a pause**, and it
is recorded here so the next session does not reopen it by finding an
interesting geometry result.

The reason is not that ordering was disproven. It is that **a controlled
ordering A/B cannot be built with the knobs that ship:**

- Holding `tau` fixed does not hold the operating point fixed -- block
  membership sets `kcvar`, so the curve moves the routing threshold. Measured:
  up to 1.23x difference in routed density between curves.
- Holding *density* fixed would need a per-`(block, sigma)` `tau`. The
  correction is a joint function of both and is not separable -- the sigma
  spread is 0.022 at block 24 and 0.136 at block 49 -- while `tau_profile` is
  keyed per transformer block and has no sigma axis at all.
- So the only runnable comparison is **matched wall clock**, which answers "at
  the speed I am paying, is this better" and is not a mechanism experiment.

Add the priors: the permutation is free, the whole activation spread across
every ordering ever measured here is 0.3-4%, geometry does not rank orderings,
link 6 is untouched, and `SOL_RECOMMENDED_CUDA` ships `morton=False` inside a
Sol-Attn that ships off. Two sessions spent a day on a knob nobody runs.

**What is still worth doing** is in `docs/open_experiments.md` and below: the
density-vs-wall-clock consistency check, and depth-based sparsity
(`dense_blocks` / `tau_profile`, both shipping empty) which is a lever on a
knob that ships **on**. The 1440x736 and 1952x544 captures that would have
settled the `3d` pin are **not** to be run -- that pin governs a knob that is
off, and the question is only interesting if this section is reopened.

**What would reopen it:** a matched-wall-clock render pair showing a visible
difference, or upstream giving `tau_profile` a sigma axis. Nothing else.

Full reasoning and the measurements behind each clause: `docs/morton.md`, and
`internal/postmortems/2026-08-16_session_sol-ordering-and-blind-controls.md`.

## Next, in order

> **Status of this list as of 2026-08-20.** Items 0 and 0b were done on
> 2026-08-17 (`docs/evidence.md`: the `_ref_prompt()` refactor is guarded by
> `bench/check_ref_prompt_labels.py` and the blind re-renders were run); 1 was
> done the same day; 3 is overtaken by `bench/sweep_routing_density.py`
> (2026-08-19); 5 was done on 2026-08-16. Only 4 (`ncu`) is still open, and the
> 2026-08-20 finding that quant error tracks Q/K rather than V competes with it
> for the same question. The list is kept as history; **"The regime question"
> below is the forward plan.**

**0. Fix `_ref_prompt()` before running anything else on references.** New top
priority as of 2026-08-16, and it displaces the Sol work because it is cheaper
and currently more load-bearing. The generator asserts attributes generically
-- "architecture, palette, and lighting" -- regardless of what the reference
actually contains, and that assertion is binding: both arms of the LoRA A/B
built a house that exists in no reference image. Every reference measurement in
this repo inherits the defect, and the fix needs no GPU. Either derive the
attribute list from the reference, or drop it and let the image speak.

Two further defects in the same template, same root cause: an interior
soundscape specified over an outdoor scene, and two contradictory lighting
instructions. Separately, all 20 reference prompts are single-shot with a
`detailed_description` of 46-73 words against the guide's stated 350-500, so a
ten-second clip is driven by about three seconds of instruction and the model
improvises the rest -- which is what produced the 60%-against-90% subject
presence in the A/B.

**0b. Re-run the LoRA A/B on a sound prompt.** Only after 0. Twenty-five
minutes of card time, and it is the actual answer to whether the LoRA is safe
as canonical. Today's run says equivalent on identity and leaves scene
coherence on one pair.

1. **Capture a reference-heavy render. Done 2026-08-17.** Two captures exist
   under `$H3_CAPTURE_ROOT/`: one single-step at seven blocks, one multi-step at
   four blocks across steps 3/8/14, both at 362 frames 1024x768 with three
   references and `S = 98498`. Taken with Sol bypassed, so the tensors are the
   true attention inputs rather than the output of the algorithm under test.
   Only the single-step directory carries a `manifest.json`; the multi-step one
   predates the manifest tooling and has no provenance record.
2. **Decompose Sol's error** (`bench/analyze_sol_error.py`). **Implemented and
   run 2026-08-17.** Splits total error into sparsity against quantization by
   measuring an exact fp32 Sol reference against dense and against the CUDA
   kernel. The headline answer: **quantization is not negligible** -- across all
   twelve rows the quant/sparsity ratio runs 14.43% to 62.20%, against the 5%
   threshold this script uses to retire the 16-bit PV question. So 17 does
   **not** close, and a 16-bit PV stays on the table. Block 49 climbs
   monotonically toward convergence (49.42%, 55.56%, 62.20%), which is the
   clearest trajectory signal in the set.

   > **2026-08-20: block 49 attributed, at the input level.**
   > `bench/analyze_head_magnitudes.py` on both captures: block 49's Q and K
   > are concentrated in four channels (K: 82, 34, 67, 19 carry ~93% of the
   > energy across heads; block 40's loudest carries ~1.4%), per-head K rms
   > spans ~2.2 to ~32 where every other captured block is flat, and the
   > heads with the largest K and Q rms are the heads with the largest INT8
   > error (Spearman ~0.4-0.5; ~0 against V). The channels are where
   > `attn.k_norm.weight` peaks at ~37 and ~31 against an rms of ~5, identical
   > in fl2va and ref2va to ~0.3%. A weights property of the release, present
   > in both checkpoints, and **not** a fl2va-vs-ref2va differentiator.
   > `bench/results/2026-08-20_head_magnitudes.json`. **Closed the same day
   > with the control:** a clean-fl2va capture (no LoRA, otherwise the
   > ref2va capture's graph) shows the same channels, the same head spread,
   > and per-head INT8 error ranking the heads identically to ref2va's
   > (Spearman ~0.95; quant 0.124 against 0.134 at block 49).
   > `bench/results/2026-08-20_sol_error_per_head_fl2va.json`,
   > `bench/results/2026-08-20_head_magnitudes_fl2va.json`.

   All twelve rows were re-measured after the eager reference was found to
   diverge from the vendored oracle and fixed; the numbers above are the
   post-fix ones, and a calibration gate now runs before any capture is read.
   The correction moved every row by under 1.2 points and changed no conclusion.

   **Re-run at every head on 2026-08-19**, on the first capture taken after the
   ref2va-direct switch: `bench/results/2026-08-19_sol_error_per_head.json`. The
   ratio band widens to 16.5%-87.2% and block 49 separates from every other
   block rather than merely climbing. The eight-head limit below is discharged;
   the oracle limit is not.

   **One limit still bounds the per-head columns.** The calibration gate can
   only run at small t, because the oracle materialises the full t-by-t score
   matrix -- so agreement at production S is inferred, not verified. Chunking
   the oracle is the one change that would close that.
3. **Port the block probe** (`sol_block_probe.py`, scaffolded, not
   implemented -- every entry point still raises). Produces routed density and
   says which transformer blocks the sparsity is hurting. Unblocks
   `dense_blocks`, which is currently a guess. Partly overtaken: item 2 already
   shows the per-block spread, so this is now about routed density specifically.
4. **Profile the Sol stages** (`bench/profile_sol_stages.py`, scaffolded, not
   implemented). Needs the card alone.
5. **Paired render**, fl2va+LoRA@1.0 against ref2va, same seed. The only thing
   that can close the reconstruction question.
6. **Watch a clip end to end.** Nothing above substitutes for it.

### What would make the error-decomposition line worth continuing

The captures are the expensive part and they are already on disk, so everything
below runs against tensors that exist. Roughly a day's work, of which under an
hour is card time.

- ~~**Chunk the oracle.**~~ **Refuted 2026-08-19, and the refutation is the
  useful part.** This was the highest-value remaining change here. The gap it
  named is real -- `bench/analyze_sol_error.py`'s calibration gate agrees with
  the oracle at t <= 2001 and infers agreement at production S -- but chunking
  closes it the wrong way, and the oracle was never why the gate stops at 2001.

  `bench/probe_oracle_gate_scaling.py` and
  `bench/results/2026-08-19_oracle_gate_scaling.json`: the oracle refuses on a
  score-matrix budget rather than a length and runs to about 384 blocks
  untouched, twelve times the length the gate uses. Run there it goes red -- and
  not for a defect. Agreement holds to ~3e-04 out to 192 blocks and jumps to
  ~1e-02 at 256, and the jump is a handful of whole query blocks, always an
  exact multiple of 64 rows, whose routing decision lands on opposite sides of
  the threshold in two float32 reduction orders. Reseed the input and different
  blocks flip. That is a tie broken differently, not two algorithms disagreeing.

  Chunked to production's 1539 blocks, flips become a certainty and the gate
  reports red while both implementations are correct, with the only relief on
  offer being the `--tol` the gate's own refusal text forbids raising. That is
  this repo's worst category of check, bought at the cost of rewriting the
  oracle into the shape of the thing it exists to check independently.

  **What to build instead, if this line is resumed:** compare the two routing
  MASKS rather than the two outputs. At production S that is a 1539x1539 boolean
  per head, needs no chunking, and separates a tie flip from a real divergence
  by reporting each flipped block's margin -- which output relative L2 cannot do
  at any length. And note that length was never the only axis inferred across:
  the gate runs at head dimension 64 with one head on `torch.randn` against
  production's 128, 56 and real activations, so closing length alone would have
  made it feel like a production gate without being one.
- ~~**Run the `--control` arm that already exists.**~~ Done 2026-08-19, first
  time since it was written: `bench/results/2026-08-19_sol_error_control.json`.
  At the dense limit the apparatus reports per-head sparsity error 3.1e-05 to
  1.3e-04 with no head above 1e-3, so it has no floor of its own on that side
  and a head reporting large error is that head. The >1.0 reading is reproduced
  at the *sparse* limit instead, where it is the expected regime -- a relative
  L2 of 1.0 is what emitting zeros gives.
- ~~**Measure all the heads.**~~ Done 2026-08-19, same day as the control.
- ~~**Emit JSON next to the printed table.**~~ Done 2026-08-19: `--json`.
- **Chunking the oracle** was the one open item above. It is now refuted rather
  than done; see the entry above for what to build instead. Nothing on this list
  is open.

**The per-head escape was designed, priced, and does not pay.** This section
used to end by predicting that if error non-uniformity held at 56 heads,
`dense_blocks` would be the wrong granularity and a per-head escape would be the
thing to design. The premise held -- per-head sparsity error spans 4.2x to
140.7x inside one block and step. The conclusion did not survive being priced.

`bench/price_head_arms.py` puts every per-head arm and the global tau on one
currency, routed density spent per fraction of error removed
(`bench/results/2026-08-19_head_granularity_arms.json`). A per-head dense escape
list ties the global tau at three heads and loses at five and eight, because
per-head error is only about 2.6x concentrated: the worst 5 of 56 heads carry
23.2% of summed error against 8.9% for uniform. A per-head *tau* could not be
priced at all -- tau moves a head's error by a median factor of 1.27 where heads
differ from each other by a median factor of 19, so equalising error puts most
heads outside any measured interval.

So the per-head axis joins per-block and per-step: real structure, nothing to
exploit at this granularity. **Anything proposing per-head work has to answer
this table first.** What it does not rule out is a mechanism that changes a
head's error rather than its threshold, which is a different kind of change.

### Two decisions that need no card, added 2026-08-16

Both fell out of reading Sol-Engine's H3 profiles properly. Neither is
scheduled work; both are calls the owner can make.

**Adopt `dense_blocks="0-1"`, or decide not to.** Every H3 profile NVLabs
publishes runs the first two transformer blocks dense --
`SOL_ATTN_FIRST_DENSE_LAYERS=2` on the single-card RTX 5090 cell,
`H3_SOL_DENSE_LAYERS=2` on GB200, `dense_blocks: int = 2` on GB10, and the
A100 README's prose. We ship none, which makes their tested recipe strictly
more conservative than ours in exactly one place.

The cost, by arithmetic rather than measurement: the 2026-08-16 dense/sparse
pair was 860.8 s against 454.0 s across 50 blocks, so **assuming uniform
per-block cost** two blocks is about 16 s of 454, or 3.6%. That assumption is
the weak part and it is cheap to check with one arm.

> **Rendered 2026-08-18** as the NVLabs recipe transplant:
> `bench/results/2026-08-18_erode_recipe_arms.jsonl`, about +2% on the sampler
> at 16 steps on the all-refs workload. Not re-priced at 4 steps.

What this does **not** need is the block probe at item 3. That instrument
answers "which blocks does sparsity hurt *here*", which is the question for
choosing **our own** list. Copying a list four hardware profiles already
validated needs no instrument, and conflating the two is why this sat as
`SOL_ARTIFACT_INSURANCE`, unwired, for weeks.

**Compare the NVLabs sm89 kernel against comfy-kitchen's.** PR #464 (2026-08-15)
added an official BF16 SM89 CuTe Sol-Attn kernel, so there are now two
independent 4090 implementations: comfy-kitchen's, which we build and run, and
NVLabs' own. Which is faster or more accurate on this card is unmeasured, and it
is the only external cross-check available to us since the Triton pack was
deleted.

The dependency half of the cost is spent. `vendor/build_sana_sol_sm89.sh`
installs their kernel into the ComfyUI venv and proves the CuTe path -- not the
Triton fallback -- is what runs; both implementations are now importable in one
process, which is what a head-to-head needs.

What is left is the seam, and it is the harder half: their public API has no
`sink_q`, so `exact_kv_and_rows`'s query half has to be done outside the kernel.
Nothing in this repo calls their kernel yet. Full accounting in
[`docs/sol_upstream.md`](sol_upstream.md).

**Scaffolded means every entry point raises.** Three files exist to fix the
design decisions -- metric choice, the control each one needs, the permanent
node id -- before implementation, so those get argued once.

---

## Hazards that have each cost something here

- **A check that skips reads exactly like a check that passed.** Exit 2 is the
  convention for "did not run".
- **A green static validator on an unsubmitted graph is unverified.** Only the
  smoke submits.
- **Free the GPU before CUDA checks**, or they OOM and read as regressions.
- **Restart ComfyUI by port owner**, not `pgrep | head -1`, which kills the uv
  wrapper and leaves the old server serving stale `/object_info`.
- **`import nodes` resolves to ours.** Any script importing both must put
  ComfyUI's root first.
- **A running ComfyUI holds two copies of every `comfy_extras` module.**
  Resolve by identity, and verify against the live `/object_info`.
- **A measurement stated at a scope wider than it was taken at** is the single
  most common defect in this repo's history. When a claim is copied into a new
  sentence, the scope qualifier is the part that gets dropped.
- **Three arms agreeing is not a control.** A fourth arm that isolates the
  variable is. A 3.7 GB "saving" survived until the dense control showed the
  opposite sign.
- **A deterministic arm re-run at a held seed is a node-cache hit**, and two
  arms that share a graph file need disjoint seed bases. ComfyUI returns the
  cached output for a byte-identical resubmission in 0.0 s, which reads as a
  fast render. `bench/run_graph_arms.py` refuses `--runs > 1` without `--seed`
  and flags `suspect_cache_hit` for this reason; a power-limit pair on one
  graph nearly shipped as two cache hits on 2026-08-20.
