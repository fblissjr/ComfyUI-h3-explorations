# Historical Regime Question, Token Ordering & August 2026 Queue

> **Archived from [`docs/roadmap.md`](../../roadmap.md) on 2026-10-27.**
> Preserves the 2026-08-16 to 2026-08-20 regime, token-ordering, and error-decomposition sections verbatim. For active closed lanes and rails, see [`docs/roadmap.md`](../../roadmap.md).

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
  ref2va -- at the same seed (`data/results/2026-08-20_ref_transfer_single.jsonl`;
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
  (`data/results/2026-08-20_session1_verdict.json`; rows in
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
  (`data/results/2026-08-20_sla_regime_arms.jsonl`)*: the SLA LoRA under its
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
  `data/results/2026-08-22_subway_speaker_bleed.json`. The owner's second
  judgement stands beside it: all twelve look and sound poor regardless, so
  the one that is correct on this point is not thereby a good clip.
  **The cheapest next test is named in that file** -- the same scene with the
  singing removed, which scopes the failure to singing or to cuts.
- *The audio hum*: not Sol, not the kernel, not the tokenizer. Sol on, Sol off
  and the SLA router hum identically; what moves it is the PROMPT asking for
  continuous texture ("tyre hiss through standing water") and 4-step
  distillation amplifying it. `data/results/2026-08-22_audio_hum.json`.

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
(`dense_blocks` / `tau_profile`, both shipping empty then; since 2026-09-25
`dense_blocks` ships `45,48,49` [under trial 2026-10-02 for replacement by `"39,41,42,49"` or `"39,40,41,42,49"`, see `docs/research/sparse/sol_dense_blocks_reanalysis.md`], and `tau_profile` was retired 2026-09-27) which is a lever on a
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
   > `data/results/2026-08-20_head_magnitudes.json`. **Closed the same day
   > with the control:** a clean-fl2va capture (no LoRA, otherwise the
   > ref2va capture's graph) shows the same channels, the same head spread,
   > and per-head INT8 error ranking the heads identically to ref2va's
   > (Spearman ~0.95; quant 0.124 against 0.134 at block 49).
   > `data/results/2026-08-20_sol_error_per_head_fl2va.json`,
   > `data/results/2026-08-20_head_magnitudes_fl2va.json`.

   All twelve rows were re-measured after the eager reference was found to
   diverge from the vendored oracle and fixed; the numbers above are the
   post-fix ones, and a calibration gate now runs before any capture is read.
   The correction moved every row by under 1.2 points and changed no conclusion.

   **Re-run at every head on 2026-08-19**, on the first capture taken after the
   ref2va-direct switch: `data/results/2026-08-19_sol_error_per_head.json`. The
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
4. **Profile the Sol stages.** *Done 2026-09-17 for stage shares*
   (`bench/profile_sol_stages.py`, implemented; reading and the live-render
   cross-check in `bench/results/2026-09-17_sol_stage_profile.md`): the exact
   stage is nearly the whole call, and compute rather than staging is its
   critical path (`data/results/2026-09-17_sol_exact_stage_vs_compute.json`,
   probe builds in place of `ncu`, whose counters are admin-only here).
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
  `data/results/2026-08-19_oracle_gate_scaling.json`: the oracle refuses on a
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
  time since it was written: `data/results/2026-08-19_sol_error_control.json`.
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
(`data/results/2026-08-19_head_granularity_arms.json`). A per-head dense escape
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

**Adopt `dense_blocks="0-1"`, or decide not to.** **DECIDED 2026-09-14: not.**
Two instruments that share no code agree the first blocks are where the
error is smallest on both terms: this repo's Sol split has blocks 0 and 8
carrying the lowest sparsity and INT8 error of anything measured, and the
sage fork's ten-cell grade has block 0 as the cleanest cell with INT8 error
rising ~5x to block 49. `0-1` protects the blocks that need it least for
the ~2% the transplant measured. `dense_blocks` stays empty. The block that
does need something is 49, and `dense_blocks` cannot give it: the fallback
is sage, also INT8, so `-1` removes the sparsity term and leaves the INT8
term that is half the error there. What can is a weights-side fold, see
"Fold block-49 K-channel balancing" below and `docs/SOLATTN.md`, "The
defaults, re-read against the sage-side error records, 2026-09-14". The
original reasoning is kept under this line.

Every H3 profile NVLabs
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
> `data/results/2026-08-18_erode_recipe_arms.jsonl`, about +2% on the sampler
> at 16 steps on the all-refs workload. Not re-priced at 4 steps.

What this does **not** need is the block probe at item 3. That instrument
answers "which blocks does sparsity hurt *here*", which is the question for
choosing **our own** list. Copying a list four hardware profiles already
validated needs no instrument, and conflating the two is why this sat as
`SOL_ARTIFACT_INSURANCE`, unwired, for weeks.

**Fold block-49 K-channel balancing into the norm weights (added 2026-09-14).**
The 2026-08-20 attribution above named four K channels (82, 34, 67, 19) as
~93% of block 49's K energy, at the `k_norm.weight` peaks. The sage fork
reproduced that on the 2026-09-03 capture and tested the obvious response:
rescale q and k per channel before INT8 quantization, `q . k` unchanged,
`s = rms_k^0.5 / rms_q^0.5` per head. Block 49's sage fp8++ error fell by
a fifth at both captured steps, below the fp16 kernel's unbalanced error;
every other captured block, having no loud channels, moved by a few percent
the wrong way. The four channels are two RoPE pairs (34/82, 19/67), so a
pair-equal `s` commutes with the split-half RoPE and folds into block 49's
`q_norm.weight`/`k_norm.weight` at load, zero runtime cost; Sol's threshold
is invariant under the paired rescale, so routing does not move. An
EXPERIMENT under the decision standard, four steps in `docs/SOLATTN.md`:
rank all 50 blocks from `k_norm.weight` (no card; 41-48 were never
captured), fold at block 49, grade `quant_l2` with `analyze_sol_error.py`
on the balanced q/k (the one step that says whether Sol's kernel benefits,
or only the sage steps do), then the blind comparison. Record:
sage fork `CHANGELOG.md`, workload intel, "MiniMax H3, block 49".

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
[`docs/sol_upstream.md`](../../sol_upstream.md).

**Scaffolded means every entry point raises.** Three files exist to fix the
design decisions -- metric choice, the control each one needs, the permanent
node id -- before implementation, so those get argued once.

---

