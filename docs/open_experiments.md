# Open experiments

Last updated: 2026-08-20

> **Several of these are now scheduled rather than parked.** The working plan
> and the render scenes that would settle the quality-blocked ones live in
> `internal/`, which is **gitignored and not distributed** -- if you cloned
> this repo you do not have them, by design. The entries below stand on their
> own; the cross-references are for the owner's checkout. Specifically:
> **#3** (comfy-kitchen INT8 arm) has a moved target, now 0.2.31; **#6**
> (revalidate 16 steps) is subsumed by the plan's stage B, which sweeps
> sampler against step count on the base model and reads a convergence curve
> rather than judging a single count; and **#1** (reference `short_edge`)
> stays blocked on the same owner judgment, now recorded as hole H5 with the
> decision to hold `allow_upscale` at True until its benefit is measured.
> `docs/checks.md` is the companion index of what *is* checked.

What has not been measured, what each run would settle, and what changes
depending on the answer. Ordered by value per unit of card time.

An experiment belongs here only if its result would change a decision. If the
answer would not move a default, a doc claim, or a piece of code, it is
curiosity and should be cut rather than parked.

Each entry states its **blocker** honestly. Several are blocked on a judgment
only the owner can make, not on machine time, and those will not resolve by
running them harder.


## Aging: this file is swept, not accumulated

**Next sweep: 2026-09-28.** At a sweep every entry is either **closed** --
which includes "we are living with this, reviewed `<date>`", a real and
different state from open -- or **re-dated with a reason**. An entry that
survives two sweeps with the same reason is not open, it is declined, and
should say so.

Adopted 2026-08-28. The reason is measured rather than felt: this file's
numbered sections went 7, 8, 13, 16, 19, 21, 24, 24 across sampled commits --
monotone at every sample. Nothing here prevents false belief; the machinery for
that is good. What was missing is anything that makes an open question more
expensive to leave open, and a sweep date is the cheapest version of that.
[`sustainability.md`](sustainability.md) argues the case;
[`research/pdd/queued_arms.md`](research/pdd/queued_arms.md) is where the
convention already worked in one lane.

---

## 1. Reference `short_edge`, sweeping down from 2048

**Tests:** whether identity fidelity holds as a reference arrives smaller,
and where it stops holding.

**Why it matters:** cost is quadratic in the short edge, and reference tokens
are attended at every sampling step. At the shipped default two references
cost 13,872 vision tokens, roughly a third of the sequence at 124 frames. If
fidelity holds at 1024 that is three quarters of it back. It also decides
whether `allow_upscale=True` is the right default: 2048 matches the released
pipeline, which is a good reason to offer it and a weaker reason to default
to it, because upscaling adds tokens rather than detail.

**Arms:** `short_edge` 2048, 1536, 1024, 768, 512 at `allow_upscale=True`,
fixed seed, one small reference (`krea2_ref_dog_face.png`, 500x520, where
upscaling is a real 4.1x) and one already-large (`krea2_ref_dog.png`,
1024x1024, where it is 2x).

**Cost:** 5 renders per reference. Minutes each at 124 frames.

**Decision it changes:** the shipped default, and whether `allow_upscale`
stays on.

**Blocker: owner judgment.** The machine can produce the videos and the token
counts; only a person can say whether the subject still looks like itself.
Do not expect a number to settle this.

---

## 2. `_chunked_heads` output buffer

**Tests:** whether the chunked path allocates a full-size output buffer the
single-shot path does not, and how much that costs.

**Why it matters:** it is the last live hypothesis for why head chunking
behaves worse than its per-call peak suggests. `_chunked_heads` builds
`out = torch.empty((1, s, heads, head_dim))` and writes groups into it, where
the single-shot path returns the kernel's own output. That is a concrete,
deterministic, per-call quantity.

**Cost:** microbench, seconds, no card contention.

**Decision it changes:** whether chunking is worth recommending at all, and
whether the chunked path should be restructured to write into the kernel's
output rather than a separate buffer.

**Blocker: none.** This is the cheapest remaining lead and should be run first.

**RESULT (2026-08-11): confirmed, and it is a flat 572 MiB.** Extra allocated
across the loop at S=41822: n=1 1144 MiB, n=2 858, n=4 715, n=8 643. That is
the full-size `out` buffer (572, constant) plus the kernel's per-group output
(572/n). The single-shot path returns the kernel's own output and allocates
neither. So chunking's floor is `full output + largest group transient`, which
is why its advantage narrows rather than scaling with n. Recorded in
`_chunked_heads`. Not removable by assembling differently; it would need a
kernel that writes into a caller-provided view.

---

## 3. comfy-kitchen INT8 attention arm

**Tests:** how core's own INT8 attention compares with ours on the same graph.

**Why it matters:** ComfyUI now ships `comfy_kitchen.int8_attention` behind
`--use-ck-attention` and a `ModelAttentionBackend` node, available on this
machine. The README claims ~2.1x against PyTorch attention, which was the
only alternative when it was written. Against a first-party INT8 path that
number may be much smaller, and the claim as written invites the wrong
reading.

**Arms:** `sage` against `ModelAttentionBackend=comfy kitchen attention`,
with our node removed from the second arm, since it is inert otherwise.

**Cost:** 2 arms, 2 runs each.

**Decision it changes:** the README's headline claim, and possibly whether
this pack's attention node earns its complexity.

**Blocker: none.**

---

## 4. Does any call reach the attention override

**Tests:** whether `make_sage_override` is invoked at all in a Sol-Attn graph.

**Why it matters:** the override deliberately keeps q, k and v alive so its
fallback can degrade, giving up the release that `sageattn_consume` exists
for. If no call ever reaches it, that trade is documenting a path that never
executes, and the comment should say so. If calls do reach it, core has
already cloned v and handed us sole ownership and we re-pin it, which is the
clone-without-consume arm.

**Cost:** a counter and one render.

**Decision it changes:** whether to restructure the override to consume, and
what its comment should claim.

**Blocker: none.** Instrument deliberately rather than bolting a counter on
at the end of a session.

---

## 5. `MiniMaxChunkFeedForward`

**Tests:** whether KJNodes' FFN chunking lowers the block peak.

**Why it matters:** derived arithmetic says no. `fc1`'s output at S=41822 is
2287 MiB, and attention and the MLP do not co-exist within a block, so the
block peak is the taller of the two. Attention is still taller at every
setting we run. That is reasoning from layer sizes, not a measurement.

**Cost:** one arm.

**Decision it changes:** whether to add the node to any graph. Currently
recommended against on arithmetic alone.

**Blocker: none, but low value.** Run it only before anyone leans on the
recommendation.

---

## 6. Revalidate 16 steps

> **Reframed 2026-08-20.** Under the distilled regime the owner is moving to
> (4-6 step lightx2v students on fl2va; `docs/roadmap.md`, "The regime
> question"), the step count is set by the LoRA's vendor row and graded by
> `bench/check_distill_settings.py`, so this entry's question becomes "which
> LoRA row is canonical", which that section tracks. 16 stays the base-model
> number and this entry stays as written for the base path.

**Tests:** whether 16 is still the right step count.

**Why it matters:** it was measured at 362 frames on 2026-08-06. That length
was called illegal on 2026-08-14 and the default moved to 345; both were
reverted on 2026-08-16, so **the 2026-08-06 measurement is back on the shipped
length** and this entry no longer needs a re-measure to be comparable. The
rejection of 12 steps rested on a real gate — the third shot of a three-shot
prompt silently stops happening — and that gate still stands.

**Cost:** 3 arms at 345 frames, and someone watching each to the end knowing
what the prompt asked for.

**Decision it changes:** `SAMPLING["steps"]`, which multiplies everything.

**Blocker: owner judgment**, same as #1. The failure mode is prompt
adherence, which no metric here detects.

---

## 7. Head-chunk process peak, settled

**Tests:** whether the 1186 MiB process-peak difference between
`head_chunks=1` and `4` is real.

**Why it matters:** it would close out the one measurement this repo has that
points the opposite way from its microbench.

**Cost:** more runs than the decision justifies. Process peak showed a 2265
MiB excursion across two runs of one unchanged configuration, so separating a
1186 MiB effect needs many runs or a less excursive observable — steady-state
allocated, or resident weight bytes sampled during the sampler loop.

**Decision it changes: none.** Wall-clock already answers whether to chunk
(0.992x, so no).

**Blocker: not worth the card.** Listed so nobody re-derives the idea and
spends the time.

> **Amended 2026-08-13, and the amendment matters more than the entry.** That
> 0.992x was measured in a graph that also loads Sol-Attn, and Sol's compose
> gate hands every call it TAKES to ComfyUI's stock forward
> (`__init__.py` at `842c4ea`) unless a delegate is published. So head
> chunking never ran on the taken steps at all. The wall-clock that "already
> answers whether to chunk" is a number for a configuration in which chunking
> was mostly switched off, and it does not answer the question it was retired
> for. Entry 8 is the live version.

---

## 8. Does head chunking do anything once nothing is bypassing it

**Tests:** whether `head_chunks` moves peak VRAM when it actually runs on
every step, and separately whether publishing a `sol_take_forward` delegate is
worth implementing.

**Why it matters:** the 345-frame reference arms peak at 22,735 MiB on a
24,564 MiB card. 1,829 MiB of headroom is the whole margin, and today's OOM
happened on Sol's own kernel, on the sparse path, where chunking cannot reach
it at any value.

**The trap in the obvious fix.** Sol's gate prefers a delegate over the stock
forward, so publishing one looks like a one-line win. It is not: our forward
is built around a single sage kernel (`make_minimax_attn_forward(kernel_fn,
...)`, and `attention.py` says so outright -- "it calls sage directly and
never reaches `optimized_attention`"). Hand it a sparse call and it runs
sage's DENSE kernel, silently disabling Sol-Attn inside its own sigma window,
with a render that succeeds and looks fine. The delegate contract is
specifically a forward that *reaches* `optimized_attention` so the override
chain still decides sparse-versus-dense. KJNodes' low-VRAM forward qualifies
and self-declares `_uses_optimized_attention = True`; ours does not.

So implementing this means a NEW forward -- the head slicing we already have
in `_chunked_heads`, calling `optimized_attention` per group instead of
`kernel_fn` -- not publishing the existing one.

**Cost:** two renders to answer whether it is worth writing at all, which is
why the measurement comes first. `sage only + chunks 4` against `sage only`
removes Sol entirely, so chunking runs on every step with no gate in the way.
If the peak does not move there, chunking is not the lever and no delegate
should be written.

**Do not reuse the dense/sparse step ratio without re-deriving it.** The
audit's 5 dense / 11 sparse is specific to shift 12, `simple`, and 16 steps:
`take` is gated on `min_tokens` and a sigma window resolved as a *percent*
band, so it moves with clip length, with step count, and with any window
change. The new 8-step turbo arms do not share it.

**The observable had to change before the arm could answer anything.** The
first version of this probe took a single-run `max()` of whole-GPU
`nvidia-smi` samples. Predicted effect ~715 MiB; entry 7's documented
single-config excursion is 2265 MiB. It could not have resolved the effect in
either direction, and a null would have been reportable as "chunking is
inert". Now sampling torch *allocated*
(`torch_vram_total - torch_vram_free` from `/system_stats`) at 1 Hz, which
excludes sibling processes and the allocator reserve, and comparing on **p90**
— a max is one excursion, a p90 is the level the run sat at.

**Do not turn that into "prefer p90 over max".** It is right *here* because
the observable is a noisy 33-minute sampled series. For a high-water mark over
a bounded region — `torch.cuda.max_memory_allocated()` around one call,
baseline-subtracted — max *is* the statistic and a percentile of it is
meaningless. Same principle, opposite implementation; the instrument shape
decides. Flagged because the slogan travels better than the reasoning.

**And state the question narrowly.** "Does chunking lower the peak" is too
loose to be falsified: `_chunked_heads` already measures net *per call* lower
(2645 against 2862 MiB at n=4, a 7.6% saving), and both terms are s-linear —
`out` is `(1, s, 56, 128)` bf16, exactly 571.8 MiB at S=41822 and ~2010 MiB at
these arms' size — so the ratio is scale-invariant and the absolute saving
grows. The open question is whether a per-call transient saving reaches
**process** peak at all, or whether that peak is set by what stays resident
across the whole step: weights, latents, and ~53k reference rows. A null
answers that, not "chunking does nothing".

**If two arms land close, reverse them before believing it.** The arms run
sequentially in one process with a `POST /free` between, which is
order-dependent state by construction — the same hazard class that cost an
upstream measurement ~535 MiB of a real delta once, via an allocator primed by
the preceding arm. Waiting to *notice* suspiciously-close numbers is a symptom
you have to be lucky to catch; running the pair in reversed order and checking
whether the delta changes magnitude or sign is a positive test that can be
scheduled. One extra render, and only worth it if the numbers are close.

**The wall-clock half of this is load-bearing outside this repo.** Upstream
has filed the caller-provided-view work with a trigger-to-act that names this
A/B: it proceeds only if a consumer adopts head-group chunking in production
*and* the in-pipeline A/B is acceptable on **wall-clock**. So the minutes
column here is not incidental to the VRAM question — a null on process peak
leaves that item live, while chunking being too slow retires it. Report both
numbers, and do not drop the timing because the memory answer looked
conclusive.

This also makes the sage-only arms the first clean wall-clock chunking has
ever had here: the 0.992x in entry 7 was measured with Sol taking most of the
steps, i.e. with chunking mostly not running.

**Pre-registered prediction, so a bad number cannot be explained after the
fact.** 0.992x is not merely unreliable, it is biased in a knowable
*direction*: chunking ran only on the dense steps, and every sparse step
contributed exactly 1.0x. Solving `(d/N)x + ((N-d)/N)(1) = 0.992` recovers the
per-chunked-step factor.

**The right unknown is the dense WALL-CLOCK SHARE, not the dense step count.**
Weighting by steps assumes a dense step and a sparse step cost the same, which
is false by construction — making sparse steps faster is what the sparse
kernel is *for* (measured on record: 1.15x at 124 frames, 1.39x at 362). With
`f` the dense share of wall-clock:

    ratio = f·x + (1-f)   ⟹   x = 1 − 0.008/f

Step-count weighting is just the `f = d/N` special case, and it **overstates
the cost** in a knowable direction:

| sparse speedup | f (d=5, N=16) | x | slower by |
|---|---|---|---|
| 1.00x (step-count weighting) | 0.3125 | 0.9744 | 2.56% |
| 1.15x (measured, 124f) | 0.3433 | 0.9767 | 2.33% |
| 1.39x (measured, 362f) | 0.3872 | 0.9793 | 2.07% |

An earlier revision of this entry carried a four-row table of `d` values
instead. It was correct arithmetic on the wrong model; its magnitudes are
withdrawn, its direction stands.

**The falsification branch needs neither `d` nor `f`.** For any `f < 1`,
`x < 0.992` — algebraically, not approximately. So "0.99x or better means the
per-step-overhead model is wrong" is sound no matter how the dense count
resolves. Only the *confirmation* branch's magnitude was ever input-dependent,
which leaves the pre-registration in better shape than the unresolved `d`
suggests. Expect roughly **0.96–0.98x**; treat any single value in it as
illustrative.

**`d` is the wrong thing to measure, and the planned instrument measured it.**
The intent was verbose-on plus a dense-path call count — which yields `d`, not
`f`, and would have produced a precise number for a quantity that does not
enter the corrected solve. Cancelled before the render rather than after.

If the historical 0.992x is ever wanted, `x` is directly measurable with no
model at all: `dense_phase_time_chunked / dense_phase_time_base`. The whole
solve exists only because the original figure was an aggregate over a mixed
run.

**Round 2 supersedes all of it.** Sol removed, every step chunked, so `f = 1`
and `x` is read straight off the two arms' wall-clock. The reconstruction is
then optional history — worth doing only to settle whether the audit's
"reproduces 6 dense at 20 steps" control ever held, which is a real question
about that audit's credibility but not a prerequisite for this result.

**A null here does not close the area.** The out buffer is real and constant
regardless of where process peak is set, and removing it is not something any
consumer-side arrangement can do — it needs a kernel writing into a
caller-provided view. That is scoped upstream and unscheduled, not impossible;
see the amended `_chunked_heads` docstring.

### Round 1 result, 2026-08-13 — and read the timing, not the memory

`h3_ref_video_swap_api.json`, 345 frames at 1024x768, unmodified against the
same graph with `MiniMaxLowVRAMAttention` (head_chunks 4) spliced between sage
and Sol-Attn:

| arm | whole-GPU peak | wall clock |
|---|---|---|
| control | 22,248 MiB | 33.2 min |
| + lowvram delegate x4 | 22,762 MiB | **30.7 min** |

**Memory: +514 MiB, and that is UNRESOLVED, not "unchanged".** The delta is
well inside this observable's documented 2,265 MiB single-config excursion
(entry 7). The probe script auto-printed "the peak did not move, so head
slicing is not what sets it" — that conclusion is not supported by its own
instrument and is withdrawn. All that can be said is that the effect is
smaller than whole-GPU peak can see.

**Time: 0.925x — 7.5% FASTER.** This is the axis that resolves, since
wall-clock does not carry the allocator excursion, and it is the axis the
script buried in a parenthetical because it was written to answer a memory
question. Single run, so it wants a repeat before it is quoted as a figure.

**Do not read this as a result about head chunking.** The delegate is
KJNodes' `minimax_attn_lowmem_forward`, which chunks *and* releases the
block's `h` early via a separate block-level patch. A speedup cannot be
attributed between the two from this arm. The clean test is round 2's
`sage only` against `sage only + chunks 4`, where nothing else differs.

Worth noting against the pre-registered prediction: it expected chunking to
cost wall-clock. This arm gained it. If round 2 agrees, the per-step-overhead
model is wrong — which is the falsification branch, and it holds without
needing `d` or `f`.

**The script's conclusion logic had no branch for this outcome** — memory
unresolved, timing moved — so it defaulted to a memory verdict it could not
support. Same lesson as choosing an instrument that no longer matches the
model: the interpretation has to be re-checked when the informative axis
changes, not just the measurement.

**Blocker: none, round 2 is running.**

---

## 9. A clean run with no speed-ups at all

> **More important under distillation, 2026-08-20, and still not run.** Every
> distilled arm stacks a LoRA on sage on Sol, so a disappointing 4-step result
> has three candidate causes. Note what does NOT discharge this entry: the
> "dense" arm in the SLA-router regime set (`docs/roadmap.md`) is sage on with
> Sol off, the repo's baseline convention, not stock torch attention.

**Tests:** whether the stack costs prompt adherence and reference fidelity,
not just time.

**Why it matters:** every reference graph here ships sage plus Sol-Attn, and
several add a turbo LoRA on top. There is **no arm anywhere in this repo that
renders without them**, so every quality judgment ever made here has been made
through the stack. General prompting research reports that stacking
accelerators degrades adherence and that a plain 20-25 step run is the thing
to fall back on when references are being ignored — untested here, and
untestable without the control.

**Cost:** one render per comparison, at base steps, on an existing graph.

**Decision it changes:** whether a disappointing reference result is the
model, the prompt, or the acceleration. Today that is unattributable, which
makes it the most expensive gap on this page.

**Blocker CLEARED 2026-08-13** — see `docs/checks.md`. Was: `SageChainAssert`
fails on a sage-only graph. Not because sage is broken — its call-time probe reads
Sol-Attn's counters (`sol_attn_stats`), and we publish none of our own, so
with Sol removed nothing moves and the node reports the composed path was not
taken. Both sage-only arms of round 2 died on it in 1.0 min. See
`docs/checks.md`. The control needs either that counter added or
`require_override`/`exercise` turned off, and only the first is a repair.

---

## 10. The swap arm measures likeness with likeness turned down

**Tests:** whether `allow_upscale=True` fits the reference-video budget.

**Why it matters:** `REF_VIDEO_BUDGET` sets `ref_upscale=False` across all
eight video-bearing arms, which was right for the arms that vary structure,
motion or continuation. `h3_ref_video_swap` is the one arm whose entire
subject is **identity**, and its reference is `1-man.png` at 1024x1024 —
1.0 MP, half the 2048 short edge the model accepts. General prompting research
puts the floor for face likeness at 2 MP and says a small reference or a face
far from camera is the first failure to rule out. So that arm currently
undercuts the thing it exists to measure.

Cost is known from the table in `docs/h3_references.md`: 1,024 → 4,096
reference rows, about +2% on a ~147k sequence.

**Second-order:** `h3_probe_reference_upscale` exists to A/B upscaling against
`h3_image_ref_plus_text_to_video`. Now that `ref_upscale=False` is the default
for eight arms, that probe no longer isolates its own variable.

**Blocker: none, it is priced in round 2's third arm.**

---

## 11. Does a reference video's INPUT resolution cost tokens

**RESOLVED for mechanism and cost, 2026-08-23.** Feeding a smaller clip changes
both core's no-upscale VAE rows and Comfy's Qwen rows. Preflight now reports
source, VAE-prepared, raw/padded sample count, and both Comfy and release Qwen
grids. The typed conditioner's opt-in `release` policy executes the other path:
full release canvas for the VAE, duration-budgeted sampled view for Qwen.

**Why it mattered:** general prompting research recommends downscaling a
reference video hard when it is only providing motion, and our loaders sit at
native (`custom_width: 0`). A reference video already costs rows in two
places — the DiT reference block, and vision blocks inside the *text* segment
at 2 fps, ~519 tokens per merged pair (`docs/h3_references.md`). The DiT side
uses core's source-size clamp when the source is below its canvas, and the
**Qwen vision** side tokenizes the prepared input resolution under Comfy's
per-pair policy. Downscaling is therefore a real sequence lever, not only a
load-time saving.

The remaining question is quality, not token accounting: whether the more
expensive release-sized video transfers identity or motion better. That is gap
6's controlled-benefit experiment, not this retired geometry question.

---

## 12. Aspect-ratio agreement between references and target

**Tests:** whether mismatched aspect ratios across reference video, reference
images and the target canvas degrade output.

**Why it matters:** general prompting research says line them up. Nothing here
does — `1-man.png` is 1:1, the placeholder clips are 16:9, and the arms render
4:3 or 7:4. If it matters, it is a confound sitting under every reference
result on this page.

**Cost:** a matched-ratio arm against an existing one.

**Blocker: needs a quality judgment, not a measurement.** Listed because the
mismatch is currently invisible and unacknowledged rather than accepted.

---

## 13. Preflight's token math has an independent implementation now

**Tests:** our packed-sequence arithmetic against somebody else's.

**Why it matters:** every validation of the reference cost model so far has
compared Preflight to numbers this repo derived. KJNodes shipped
`MiniMaxH3TokenCounter` on 2026-08-11, which builds the same
`[text | keyframes/refs | audio | video]` layout and reports `seq_len` plus a
breakdown, written by a different author from the same upstream source. This
repo's stated preference is an independent implementation over a self-derived
number, and one is now sitting in `custom_nodes/`.

Note the two are already known to differ elsewhere: `docs/h3_resolutions.md`
records it computing its int32 warning from the contiguous stride where our
Preflight uses the fused one. That is a reason to diff them, not a reason to
assume either is wrong.

**Cost:** wire it beside Preflight in one graph and compare, or lift its
arithmetic into a CUDA-free check.

**Blocker: none.** The strongest available control for a number this repo
leans on heavily.

---

## 14. There is no output-quality instrument for any modality

**Tests:** nothing yet. This entry exists because four entries above are filed
under "blocker: owner judgment" and that is only half true — it is the blocker
on the *verdict*, not on getting a first pass.

**The observation that prompted it.** Asked why quality work here leans on
audio, the honest answer turned out not to be "audio matters more". It is that
`bench/` has **no output-quality tooling at all**: every number in it is
kernel-level cosine or rtol against a reference tensor, and nothing reads a
rendered frame or a decoded waveform. What exists at the output level is ad
hoc, and it split by whether a number could be extracted mechanically:

| modality | what was measured | status |
|---|---|---|
| audio | dB mean/peak per render | **retracted** — did not survive 362 frames; `SOLATTN.md` says accuracy against the sage output was never checked at all |
| video | human viewing — object dissolve, prompt adherence at 12 steps | `SOLATTN.md`'s Quality section is marked **REOPENED**: judged from stills, and the failure mode is temporal |
| reference image identity | nothing | entry #1, blocked on owner judgment |

Audio is not over-tested. It is the one that got a *number*, because `ffmpeg`
prints dB and nothing prints "the subject still looks like itself". That is
instrument availability, not a claim about what matters — and the one number it
did produce was retracted, which is the argument for building the instrument
rather than against it.

**Why it matters:** #1, #6, #9 and #12 are all queued behind the owner
watching renders, and that queue does not drain by adding card time. A
mechanical screen would not settle any of them, but it would order them —
say which pairs are worth watching and which are indistinguishable — and it
would catch regressions between the sessions where somebody is watching.

**Arms:** none. The deliverable is the instrument, and the repo already knows
what shape each one has to be:

- **temporal, not per-frame.** `SOLATTN.md` is explicit that a grid of stills
  cannot catch the object-dissolve artifact, and states the form a gate takes:
  track one small persistent object frame by frame. A per-frame metric
  averaged over a clip would report the artifact as noise.
- **not a numeric diff of two renders.** Comparing finished renders
  numerically measures trajectory chaos — at 20 steps of a flow ODE any
  perturbation diverges while both outputs look fine. That trap is already in
  `SOLATTN.md`'s measurement-traps list and it is the reason the obvious
  instrument is the wrong one.
- **not through a codec.** `h3_config.py` records the first int8-VAE quality
  pass measuring an h264 round trip's noise floor (1.63/255 at 41.1 dB) and
  reporting it as the decoder. Three comparisons returning the same number was
  the tell. Whatever this reads, it reads it before encode.
- **audio has the cheapest first version**: the paired files already exist, so
  accuracy against the sage output is a comparison, not a render.

**Cost:** unknown, and deliberately not estimated here. Scoping it is the
first task, not a thing to guess at.

**Decision it changes:** whether #1, #6, #9 and #12 stay blocked on a person.

**The cheaper route, 2026-08-20:** at ~3 minutes a 4-step render, the standard
CLAUDE.md sets for a perceptual claim -- many seeds per arm, judged blind in
aggregate -- costs under an hour for 8 seeds x 3 arms, which no base-model
session could afford. That does not build the instrument; it makes the
owner's judgment affordable enough to drain the queue without one.
`bench/blind_batch.py` is the batching layer.

**Blocker: none, and that is the point.** Every other entry blocked on owner
judgment has been listed that way since 2026-08-13 without anyone asking what
would make the judgment cheaper. A screen that can go red is worth more than
another parked question, and this repo's own rule — a check is not trusted
until it has been shown red for the right reason — applies to the instrument
before it applies to anything it grades.

---

## 15. tau 1.0 against 1.3, and morton's quality axis

**Tests:** two defaults the algorithm's author reframed on 2026-08-14, both of
which we set on evidence that turns out to address a different question.

> Kijai: "tau 1.0 is where it's the default *max* quality, any higher further
> degrades it, but also speeds it up. morton may or may not increase quality,
> that's something to test."

**Why it matters — and this is the part worth reading carefully.** Neither
statement contradicts anything measured here. Both reveal that what we measured
was a different axis from what we claimed.

**tau.** `h3_config.py` presents 1.3 as the quality choice, on the grounds that
it sits below the object-dissolve artifact's onset (~1.5). True, and it is a
statement about where quality falls off a *cliff*. Kijai's is about where
quality *peaks*: 1.0, degrading monotonically above it. So 1.3 is a
speed-for-quality trade we have been describing as a quality decision, and we
sit in the gradual-degradation band between the two thresholds — precisely the
band with no dramatic tell, which a stills-based judgement cannot see. Every
tau arm ever run here compared 1.3 against 2.0, so all it establishes is that
1.3 beats something worse. **1.3 against 1.0 has never been run.**

**morton.** We set `morton=False` on a speed result — worth 1.16x alone, a net
loss stacked on int8, 94% GPU utilisation against 99% elsewhere — and **that
result is RETRACTED for the CUDA backend, 2026-08-16.** It was Triton, 362
frames, stacked on int8. Isolated against a dense control the permutation is
free -- the dense and sparse pairs disagree in sign and both sit under the
bench's run-to-run spread -- and the three curves are speed-indistinguishable.
Kijai says the quality effect is untested, and it still is.

This makes the arm **more** worth running, not less. Reordering video tokens so
each 64-token block is a compact 3D neighbourhood changes *which* blocks the
router keeps, and on captured activations it measurably tightens the per-block
centroid. With the cost at zero, any quality gain at all would justify turning
it on, so the default is no longer a trade — it is an untested knob left off.
`Canonical: docs/morton.md`.

**Arms:** `shipped` (tau 1.3, morton off), `shipped[tau=1.0]`,
`shipped+morton2d`, `shipped+morton3d`, `shipped+hilbert`, and
`shipped+reorder_only` for the cost-in-isolation control. All are named arms in
`bench_e2e_h3.py` as of 2026-08-16; the ad-hoc syntax covers any tau crossing.
Note that a fair Morton test raises `tau` rather than holding it, since the one
stated payoff is "the same quality at higher sparsity" — an arm at fixed tau
measures the cost and none of the benefit.

**Cost:** 4 arms at 345 frames. Time is the cheap half.

**Decision it changes:** `SOL_RECOMMENDED_CUDA`'s two most-quoted values.

> **2026-08-20: the default moves to tau 1.0 by owner decision, and 1.3 has to
> earn its way back.** 1.3 stays only if it shows no difference from 1.0 on the
> distilled LoRAs while buying meaningful speed. The speed half is a
> `--set SolAttnMiniMax.tau=1.3` patch arm in the day's regime set; the quality
> half is an 8-seed blind session, not yet run.

**Blocker: the same one as #1, #6, #9 and #12 — an instrument.** Speed will
answer itself in one run. The quality half is a gradual degradation with no
artifact to point at, which is the failure mode #14 exists for: stills cannot
see it, a numeric diff of two renders measures trajectory chaos, and there is
no output-quality tooling for any modality. **Run the speed half now and do not
pretend the quality half was answered by it.** Recording the timing alone would
produce exactly the reading this repo keeps having to retract — "1.0 costs N
seconds" filed as if the tradeoff had been priced.

---

## 16. The single-frame path: four questions the first render did not answer

> **PARKED 2026-08-27 with the path itself.** 16a-16g are closed as *will not
> be measured*, which is not the same as answered. The owner moved off this
> lane; the graphs are `archive/workflows/image/`, the core patch is
> `archive/single_frame.py`, and nothing generates, walks or grades any of it.
> [`h3_image_editing.md`](h3_image_editing.md) is the parking record.
>
> **Everything below is left as it was written** -- registrations, blockers
> and results in their original tense -- because it is the record of what was
> measured and what was not. Read every "blocker", "still open" and "next
> arms" line as superseded by this banner: the arms they name are archived,
> and running one means un-parking first.

Added 2026-08-15, when `length=1` became reachable and
`workflows/image/h3_image_edit.json` shipped (it was `workflows/h3_image_edit.json` until the image graphs were foldered by use case on 2026-08-16; both paths are now under `archive/`). The path worked end to end and the VAE
question is settled with ground truth (37.27 dB against 22.04 dB on a `T=1`
round trip, in `CHANGELOG.md`). These four are not.

**16a. Does the latent-slice fallback work, and what does it cost?** The
no-shim workaround everyone uses is: render 5 frames, decode all five, keep
image 0. Nobody appears to have tried the other one -- render 5 frames, slice
the LATENT to `T=1`, then decode that single temporal step with the image VAE.
The reported grid artifact comes from handing a 2-step latent to a decoder
trained on 1, so slicing first should avoid it entirely while still paying 2x
the video rows. **Why it matters:** if it works it is a shim-free path, and it
also separates "the artifact is a decode-side effect of multi-frame latents"
from "the DiT's extra temporal context changes the image". **Blocker:** needs a
node to slice a nested AV latent temporally, and this repo has deliberately not
spent a permanent `node_id` on a temporary problem. Cheap to answer with a
throwaway script before deciding.

**16b. Is the image VAE's softness ever the wrong trade on GENERATED latents?**
The 15.2 dB result is a *reconstruction* measurement -- encode a real image,
decode it, compare. That is the right test for a decoder and it is decisive
there. It is not the same question as which decoder produces the better image
from a latent the DiT invented, where there is no ground truth. On the one
generated sample compared, variance-of-Laplacian preferred the video VAE
(163 against 118) and a 1:1 crop showed why the metric was wrong: the video VAE
was adding local contrast and a colour shift, not detail. **That is n=1 with a
metric that misled once already**, which is exactly the shape of the h264
noise-floor mistake in `h3_config.py`. **Blocker:** owner judgment on a handful
of paired renders; there is no automatic instrument for this (see #14).

**16c. fl2va at one frame is unmeasured, and we refuse it rather than know.**
`MiniMaxH3ImageToVideo` pins a `last_frame` keyframe at `frame_count - 1`,
which in a one-frame video is frame 0 -- on top of `first_frame`.
`keyframe_canvas.py` therefore refuses `length=1` outright -- **still true and
now the only trace of this lane on the live tree**, and its error message says
the lane is parked -- and the image path that shipped was ref2v. Whether
fl2va with only a `first_frame` does something useful at one frame (a true
img2img) is unknown. **Why it matters:** it is a
different edit modality from reference conditioning, and the refusal is a
guess wearing an error message. **Blocker:** one render, plus deciding what a
`last_frame` should mean at one frame.

**16e. How many reference images does a single-frame edit actually hold?
Untested, and the cost curve says the limit is not the one we would guess.**
Every render on this path used ONE reference. Core caps images at 9 (and
videos at 3, audio at 3), and the sequence arithmetic says nothing breaks
before that: one 1024x1024 reference through `MiniMaxH3ReferenceFit` at
`allow_upscale=True` is 4,096 rows (measured by Preflight, and the arithmetic
agrees exactly), so nine of them is 36,864 reference rows and 42,008 total --
**51% of the 124-frame reference video graph, which already fits on the 4090.**
Turn the fit node's upscaling off and nine references cost 9,216 rows, a
sequence of 14,360.

So VRAM and sequence length are NOT the binding constraint, and the interesting
limits are the ones no number here predicts: whether identities stay separate
past two or three people, whether the model still attends to `<Picture 7>` when
it is told to, and whether prompt adherence degrades with label count. The
community write-up reached four (three people plus a location) and reported it
works; nobody has published where it stops.

**Decision it changes:** whether the shipped graph stays single-reference, and
whether `allow_upscale=True` is right on THIS path -- it is the single largest
cost here (4x per reference) and it exists for identity fidelity, which is the
one thing a multi-subject edit stresses most.

**Arms:** 1, 2, 3, 4, 6, 9 references at the shipped canvas, same seed, prompt
naming every label; then the 4-reference arm repeated with `allow_upscale=False`
to price the fit node. `internal/reference_library.md` has 19 images ready.

**Cost:** 7 renders, seconds each at one frame. This is the cheapest experiment
on the whole list.

**Blocker: none for the numbers, owner judgment for the verdict.** Token counts
and render times are mechanical; whether four faces still look like four
specific people is not.

**RESULT (2026-08-16), 13 arms via `bench/bench_image_edit_refs.py`.** Three
things settled and one still open.

*Composition is not what breaks; the card is.* Measured sequence and wall clock
at the shipped sizing, one frame each:

| refs | sequence | secs | outcome |
|---:|---:|---:|---|
| 1 | 9,135 | 42 | identity held |
| 2 | 17,352 | 78 | subject + scene composed |
| 3 | 32,093 | 198 | subject + garment + place, jersey text legible |
| 4 | 40,294 | 280 | **two distinct identities, no blending**, right garment on the right person |
| 6 | 56,710 | 491 | **three distinct identities**, all correct |
| 9 | ~94,000 | -- | **OOM on a 24 GB 4090** |

Three separate faces survive in one frame. What fails first is memory, and it
fails between 6 and 9 references at this sizing.

*The cost model was wrong, and the correction is the finding.* Reference images
are paid for TWICE -- as Qwen vision tokens in the `text` segment and again as
latent rows -- and the text half scales with count, landing 75-160 rows above
the reference half at every rung. The ladder is recorded in
`docs/h3_references.md`. The projected costs in the bench script are therefore
about half of true, and 9 references is ~94k rows: more than the 124-frame
video graph, on a single frame.

*What `allow_upscale` buys is, on this evidence, nothing.* Same seed, same two
references, three sizings:

| sizing | ref rows | secs |
|---|---:|---:|
| `max` + fit upscale (SHIPPED) | 8,192 | 84 |
| `max`, no fit upscale | 2,048 | 18 |
| `match` | 1,682 | 16 |

4.9x the rows and 5.2x the wall clock, and at 1:1 on the face all three hold
the same identity, glasses, hair and features. **One subject, one seed, one
scene -- this is not enough to move the default on its own**, and identity
drift is exactly what a person judges better than a crop comparison. But it is
the first evidence either way, and it points at the shipped default costing 5x
for nothing. `size-small-source` is the sharper version of the same point: a
662x1177 source enlarged 3.1x to reach 2048 cost 14,784 reference rows and
186s, more than the entire four-reference composition, for the least detailed
source in the library.

**RESOLVED FOR THE IMAGE PATH, 2026-08-16.** The sizing result was reproduced
on a second subject and seed -- `h3_image_style`, two references, 89.1s with
the fit upscale against 18.1s without, and the pair compared against the source
reference rather than against each other: same identity, freckle pattern, head
angle, expression and hairstyle, with the graphite medium transferring in both.
So `ref_upscale=False` became the default for every graph on that path
(`workflows/h3_config.py`'s `IMAGE_EDIT_BUDGET`, which outlived the graphs),
and the set rendered in about two minutes against about eleven.

**Two subjects and two seeds is still a small n**, and none of it transfers to
the video path, which keeps `ref_upscale=True`: a 124-frame render is minutes,
and identity there has to survive motion as well as a still frame.

**STILL OPEN, and it is 16b's question in a new place:** whether the sizing
result holds across more subjects and seeds. Two cheap follow-ups: repeat the three
sizings on 3 more subjects at 2 seeds, and re-run the 9-reference arm at native
sizes (~19k reference rows rather than ~94k), which should fit and would tell
us whether 9 identities hold when the memory wall is moved.

*Canvas, at one frame, is nearly free and nothing chose itself.* 768x1152,
1024x1536 (out of family), 1344x768 and 768x768 all rendered cleanly at
comparable cost. 16d stays open on owner judgment.

**16d. In-family 768x1152 against the community's 1024x1536.** The shipped
graph uses the in-family canvas; the write-up it follows renders 1.57 MP, 52%
over H3's area cap. At one frame the canvas costs almost nothing (the video
segment is 9% of the sequence), so the usual reason to stay small does not
apply here -- which makes this the cheapest quality question on the list and
the one most likely to change a shipped default. **Blocker:** owner judgment on
paired renders.

---

## 16f. Which prompt format a single-frame edit wants

Added 2026-08-16, when the image path moved to `workflows/image/` and its
prompts were rewritten into the guide's structure.

**Tests:** whether the official ref2va structure earns its tokens on a still
frame, and separately whether the two audio sections cost anything when
carried on a graph that has no audio decoder.

**Why it is open rather than decided.** This path shipped flat prose until
2026-08-16, on the argument that the guide is a *video* guide -- two audio
sections and a `[Shot 1]` with shot timing. Then the r/StableDiffusion author
whose write-up this path follows published a second prompt set, and **between
their two posts they switched from flat prose to the guide's structure** with
the audio sections dropped. They had rendered a couple of thousand images by
then. But neither post held the scene or the references fixed, so it is a
practitioner's revealed preference and not a measurement -- the same grade of
evidence as the Custom-GPT kit in `internal/PROMPTING.md` section 4.2.

**The arms exist and are unrendered.** `h3_image_style.json` (four sections,
the shipped default), `h3_image_probe_format_av.json` (all six, audio ones
`N/A`), `h3_image_probe_format_flat.json` (one paragraph). Same scene, same two
references, same seed. The content is generated once per scene and rendered
into all three formats, so the arms cannot differ in wording -- which is what
the two Reddit posts do differ in, and why they cannot answer this.

**What to look at:** whether the style reference brings its own cottage. That
is the scene's designed failure and it is visible at a glance, which matters
because there is still no output-quality instrument here (#14).

**Ladder, and read a result against it:** `av` -> `sections` removes only the
audio pair. `sections` -> `flat` removes the guide's formal apparatus as a unit
-- headers, shot marker, and marker vocabulary rendered as English. That last
rung is three things on purpose: a paragraph carrying `attribute_transfer -`
mid-sentence is a form nobody writes, and beating a strawman would tell us
nothing.

**Cost:** 3 renders, seconds each at one frame with 2 references.

**RESULT (2026-08-16). All three arms rendered; none failed.** Same scene, same
two references, same seed, same 16 steps, one pass so nothing drifted between
them. ~18s each.

| arm | prompt tokens | sequence | mean pixel delta vs `sections` |
|---|---:|---:|---:|
| `sections` (default) | 326 | 5,386 | -- |
| `av` (six sections) | 341 | 5,401 | **3.45** |
| `flat` (one paragraph) | 292 | 5,352 | **44.61** |

Grey-scale mean absolute difference, 0-255. For scale, an h264 round trip on
*identical* pixels measures ~1.6 (`h3_config.py`), so 3.45 is close to the
floor and 44.61 is 13x larger.

**The two audio sections cost essentially nothing**, in tokens (15) or in
output (3.45, near the noise floor). The prediction in the probe's own note --
"no visible difference, which is the useful outcome" -- held. Carrying them is
free; so is not carrying them, which is why the four-section default stands on
the honesty argument rather than a cost one.

**The scaffolding is a large lever on the image and did NOT break the role
binding.** `flat` differs from both structured arms by 44.6, which is a
materially different picture -- tighter crop, the subject larger in frame. But
a prompt text change moves the image by construction, so a large delta is
expected and is not by itself a defect. What matters is the designed failure:
**no cottage appeared in any of the three.** Identity, freckle pattern, head
angle and the graphite medium held in all three. `attribute_transfer` bound
without the section scaffolding.

**So the honest reading is a negative result: on this scene, at this seed,
format did not decide whether the roles bound.** That is worth having and it is
not "format does not matter" -- see the limits below.

**Limits, and they are real.** n=1 per arm, one scene, one seed. The `flat` arm
deliberately KEEPS the negative clause ("<Picture 2> supplies no subject, no
scene and no composition") because content is held fixed across formats, so it
carried the same protection the structured arms have -- it is not the bare
community-style prompt. And a two-reference style transfer with an explicit
negative clause may simply be too easy a case to separate the formats.

**What would actually discriminate**, in order: drop the negative clause from
one arm (that is the untested technique, not the scaffolding); run the ladder
on `h3_image_multiperson`, where three references and two identities give the
model more to confuse; and repeat at 2-3 seeds, since one seed cannot separate
a format effect from a sample.

**Blocker: the path is parked.** It was none, and the next arms are specified
above; they were never run and now sit in `archive/workflows/image/`.

**Two follow-ups it would open, not close.** If the structured arms win,
whether the `<Subject N>` indirection specifically is what did it (the flat arm
keeps it, deliberately, so content is held fixed). And whether the result holds
on a one-reference scene, where there are no roles to confuse.

---

## 16g. Step count on the single-frame path

Added and partly answered 2026-08-16.

**Tests:** whether one frame needs the 16 steps the video path uses.

**RESULT: it does, and the value is in where it breaks.** One paired render per
scene, same seed, `ref_upscale=False`, 16 against 8:

| scene | refs | 16 steps | 8 steps | verdict |
|---|---:|---:|---:|---|
| `h3_image_edit` | 1 | 13.0s | 4.0s | indistinguishable |
| `h3_image_style` | 2 | 18.0s | 7.0s | freckling and medium both hold |
| `h3_image_multiperson` | 3 | 25.0s | 10.0s | **8 loses the freckling and the pendant** |

At three references, 8 steps drops precisely the fine detail that scene's
`partially_preserved` entry names as retained. It buys ~15s on the one graph
where that detail is the point, so **`steps` stays 16** and no per-scene step
count was introduced.

**The methodological point is the durable half.** Measured only on the
one-reference portrait -- the obvious scene to try, and the fastest -- 8 steps
looks free everywhere and the default would have moved. A check whose input
already satisfies the expected outcome cannot fail, and a single studio
portrait is that input for step count.

**Still open:** a real sweep (12, 10) on the three-reference scene, and whether
a per-scene step count is worth the complexity. One paired render per condition
is consistent with the expected mechanism and is not a sweep.

---

## 18. Routed density under each curve, at fixed `tau`

**Tests:** how far a token ordering moves Sol-Attn's operating point. Every
Morton and Hilbert A/B this repo has run compared two orderings *and* two
sparsity levels, without knowing the size or the sign of the second.

> **Substantially done 2026-08-19.** `bench/sweep_routing_density.py` measures
> routed density per block, step and head at production S on the 2026-08-18
> capture (`bench/results/2026-08-19_routing_density_per_head.json`), with the
> curve as an argument; the per-block and per-step axes came out flat. What
> remains of this entry is the curve comparison at fixed tau specifically.

**Blocker: a capture at a known video geometry.** No render, no GPU, no server
once one exists, and it belongs in `bench/analyze_routing.py`, which now runs
without the `coderef/` clone. The `2026-08-15_dense_124f_1344x768` capture this
item was written against is **no longer on disk**; what is there is the
2026-08-17 reference-heavy pair at 362 frames 1024x768, whose video span differs,
so the geometry has to be re-derived rather than reused.

**Method, and one constraint that is not optional.** Score with upstream's eager
reference (`coderef/comfy-kitchen-sol/comfy_kitchen/backends/eager/sol_attn.py`,
routing at `:112-142`), not a fresh transcription of the threshold formula. Its
own docstring says it "defines the algorithm, not the CUDA kernel's arithmetic",
which is exactly the right relationship: it is upstream's statement of what
routing *means*, so using it keeps a reimplementation risk out from between the
measurement and the claim. `bench/analyze_capture.py` already builds all four
orderings and applies the `(-video_start) % 64` roll -- reuse that, do not
rebuild it. Count the conditioning rows into the block population, since `kcvar`
is a variance over *all* block centroids; exclude the forced-exact pairs
(diagonal +-1 and sink) from the density, since they never consult the
threshold.

**Those two exclusions are different things, and only one of them is an
exclusion.** Forced-exact *pairs* come out of the density -- they never consult
the threshold. Conditioning *rows* stay in the block population, because
`kcvar` is a variance over every block centroid in the sequence and dropping
530 rows would move the threshold for every query block. That is also the
reason to score the whole packed sequence rather than slicing the video span:
the slice is a different partition from the one the kernel sees.

> **A caveat stood here on 2026-08-16 saying the prototype had dropped the
> conditioning rows, and it was wrong.** Withdrawn the same day after reading
> the script rather than the report of it:
> `internal/scripts/archive/sol_curve_2026-08-16/probe_routed_density.py:41` sets
> `n = S // BLOCK` over the full 37,826-row sequence, and `kc`, `kmean` and
> `kc_var` are all computed over that population. The threshold is already
> derived the way the kernel derives it, and no reported number moves for this
> reason. Left in place because "expect the numbers to move" would have sent
> the next reader hunting a discrepancy that does not exist -- a wrong caveat
> costs more than a missing one.

### Pick which density you are reporting, and emit both

**The real defect in the prototype, and it is a labelling one.** Its docstring
claims diagonal, neighbour *and sink* blocks are excluded from numerator and
denominator. The code masks only `|i-j| <= 1`
(`internal/scripts/archive/sol_curve_2026-08-16/probe_routed_density.py:47`); there is no sink mask
anywhere. Sink pairs sit in the denominator and are judged by the threshold
like any other pair, when the kernel forces them exact regardless.

So the figures produced on 2026-08-16 are **"the fraction of non-adjacent pairs
the threshold would route"**, not "the fraction the kernel routes exact". The
kernel's number is higher. Two consequences, and they point opposite ways:

- **The ratios stand.** Same pair set under every ordering, and forced-exact
  status is ordering-invariant, so the 1.15x / 1.11x comparisons are unaffected.
- **The absolutes do not mean what their label says**, and anything sized
  against them -- `routed_cap_percent` headroom, cost estimates -- would be
  sized against the wrong quantity.

These are two different measurements and neither substitutes for the other:

| | what it answers | how |
|---|---|---|
| **ordering-effect density** | what did the permutation do | drop *every* forced-exact pair -- diagonal, neighbour, sink-KV range, sink-query range -- from numerator and denominator |
| **kernel density** | what does the kernel actually route, and what does it cost | count forced pairs in, as the kernel does |

**Emit both.** It is one extra mask, and reporting one while labelling it the
other is the mistake that already happened once here. `docs/morton.md`'s pending
figures are the first kind and are labelled as such.

**What it must report, not just a number.** Density per (curve, depth), and the
`tau` that returns each curve to raster's density. If those compensating taus
differ by depth, that is expressible as a `tau_profile` -- it is keyed per
transformer block (`vendor/sol_attn_minimax.py:55-76`). If they differ by
*sigma*, nothing in the node can compensate it, and the whole idea dies there.

**Do not build a compensation table off this alone.** The only capture is step 1
of 6, and Sol runs 11 of 16 steps at the shipped window. Three depths is also
three points to cover fifty blocks. A second capture at a late step is the
prerequisite for anything shipping, and is a separate small job:
`h3_capture.py` needs `H3_CAPTURE` set before ComfyUI starts.

**Why it matters beyond Morton.** It is the missing denominator under
`docs/morton.md`'s six-arm sweep, under `SOL_RECOMMENDED_CUDA`'s pinning of
`3d`, and under any future curve. It converts "these two orderings scored X and
Y" into "these two orderings sat at different points on the speed-quality curve,
this far apart" -- which is the difference between a comparison and a
coincidence.

---

## 17. A 16-bit PV branch for the CUDA Sol-Attn kernel

**Tests:** whether `sol_attn_exact.cu` should get a 16-bit PV matmul, keeping
INT8 QK -- the same shape sage runs -- and whether that is worth the kernel
work at all.

**The name is misleading and the scope is much narrower than it sounds.**
Sage's `fp16 (most accurate)` is not an fp16 kernel. It is `qk_int8_sv_f16`:
INT8 QK, 16-bit PV (`csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`, `MMA_QK_K 32`
int8 against `MMA_SV_K 16` fp16). So "give Sol an fp16 path" means one thing --
move the PV matmul from `mma_u8s8` to a 16-bit MMA. QK stays INT8 on both
sides. Read as "add fp16 kernels" this looks like a rewrite; it is one matmul.

### A 16-bit Sol-Attn exists on Triton -- but it does NOT price this change

**Corrected 2026-08-16, same day this entry was written.** The first draft said
the Triton numbers "already price" a 16-bit PV. They do not, and the reason is
the failure shape the 2026-08-15 postmortem names three times: a measurement
stated at a scope wider than it was taken at.

What is verified, read from `coderef/ComfyUI-SolAttn_triton/__init__.py` rather than
from our own bench comment:

- The node dispatches **two different kernels**, not one kernel with a dtype
  flag: `kernel = _sol_attn_int8_kernel if int8_qk else _sol_attn_kernel`
  (`:214`), logged as `int8` or `bf16` (`:222`).
- `int8_pv` is passed only when `int8_qk` is on (`:213`), and the node's own
  default is **`int8_pv=True`** (`:468`). Our `SOL_BASELINE_124F` pins it
  `False`, which is the only reason the bench's `sage+sol+int8qk` arm really is
  INT8 QK with 16-bit PV. **Read from the node alone, that arm is full INT8.**

So the frontier table's `sol, no int8` (827.9 s) against `sol + int8 + int8_pv`
(714.9 s) is **bf16 kernel against int8 kernel** -- it varies the PV dtype, the
QK dtype, and the implementation, all at once. It cannot be read as the price
of the PV dtype.

**The clean isolation has never been run.** `int8_qk=True` with `int8_pv` on
against off, same kernel, one variable, is the arm that would price this, and
there is no such row anywhere. The 2668 ms profile is the only measurement of
the proposed config and it has no `int8_pv=True` counterpart profiled beside it.

**And Triton cannot price a CUDA change anyway.** `docs/morton.md` retracted
exactly this move on 2026-08-16: `h3_config.py`'s "morton is worth 1.16x alone"
was Triton, 362 frames, stacked on int8, and the CUDA isolation found no
resolvable cost at all. Its verdict -- "correct for what it measured, wrong as a description
of the CUDA backend" -- applies here unchanged. The 2026-08-14 CUDA migration
postmortem is blunter: **every headline number that migration produced was
withdrawn.**

Everything above also inherits SOLATTN.md's "do not rely on" list: 362 frames
(not a legal length), an unrecorded build, pre-2026-08-14, `res_multistep`.

**Not an accuracy argument, and the first draft used it as one.** Triton bf16
grades 0.999995 against the eager reference where Triton INT8 grades 0.999885.
SOLATTN.md lists those cosines under "do not rely on": they are **implementation
fidelity, not accuracy** -- each kernel is graded against the reference *at the
same tau*, so the sparse approximation sits on both sides and cancels -- and
they are T=512 synthetic. They say the bf16 kernel tracks its own reference more
closely. They say nothing about what 16-bit buys on real H3 activations, which
is the question.

### The one measured fact that bears on the cost

From the captured-activation run written up in `docs/morton.md`, real q/k, this
model, 124 frames: **attention is not very sparse on this workload.** At its
most concentrated a query still needs 178 key blocks of 591 to hold 90% of its
mass, and 394 at block 0.

That is a bound on how small the exact branch can be, and it pushes the cost of
a 16-bit PV **up**, not down: the more work sits in the exact branch, the more
of the kernel a 2.5x PV touches. It is not the routed density -- mass
concentration and the tau routing decision are different quantities -- but it is
the closest measured thing, and it argues against the assumption that the exact
branch is a thin slice.

### The layout problem is already solved in-tree

Expected to be the hard part; it is not.
`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu::mma_bf16`
already runs bf16 PV inside the Sol codebase, and
`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh::pack_bf2`
already carries the helpers it needs.

**Cited by symbol since 2026-08-30, and the reason is a silent miss.** Both
were line ranges into those two files -- lines 509-528 of the routing kernel
and 104-116 of the layout header -- taken when that checkout sat at kijai's
pre-merge tip. Advancing it to the
merged code rewrote the routing kernel, and the two citations came apart
DIFFERENTLY: the first went out of range and `bench/check_doc_links.py` caught
it, while the second stayed in range and now points at a staging helper with
nothing to do with the claim. **The range check can only see a line that does not
exist, never a line that means something else**, so a citation into a file that
moves under you is only half-guarded by it. The claim itself survived both --
bf16 PV is still there -- which is exactly why nothing else would have noticed. The INT8 QK score tile feeds a 16-bit A
operand with no shuffle and no permutation -- two adjacent n8 tiles are exactly
the `m16n8k16` A layout. Sage does the same (`RS_32_to_16` in its
`attn_utils.cuh` is a pure convert, no lane exchange). Read from both sources,
not derived from a fragment map on paper.

Consequence: **`perm_key` exists only to make the INT8 repack free**
(`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:63-65`). A 16-bit path does not need it, and V^T stays in the
logical key order it is already stored in.

### MMA issue rates, measured on this box 2026-08-16

`bench/mma_rate.cu`, RTX 4090 sm_89, register-resident and issue-bound:

```
form                        ms      TMAC/s   vs int8
s8   m16n8k32 -> s32     0.822       334.5     1.00x
u8s8 m16n8k32 -> s32     0.822       334.6     1.00x
bf16 m16n8k16 -> f32     1.640        83.8     0.25x
f16  m16n8k16 -> f32     1.640        83.8     0.25x
f16  m16n8k16 -> f16     0.821       167.3     0.50x
```

`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:83` justifies the all-INT8 branch with "sm_120 is issue-rate
bound and f32-accumulate forms issue at half rate". **That holds on sm_89 too**
-- identical instruction count, exactly 2x the time. Verified rather than
carried over.

Exact-kernel MMA per warp per key block: QK is 8 n-tiles x 4 k-chunks = 32; PV
INT8 is 16 x 2 = 32; PV 16-bit is 16 x 4 = 64. So the arithmetic predicts
**2.5x on the exact branch** with f32 accumulate, 1.5x with f16 accumulate.

**This is now the only cost estimate that exists**, and it is arithmetic over
an instruction-rate measurement, not a kernel measurement. It is an upper bound
on the MMA term alone: it assumes the exact kernel is MMA-issue-bound, and if
the kernel is bound by cp.async staging instead -- which the 16-bit V tile also
makes worse, doubling `LDV` and taking smem from 32 KB to 48 KB per block -- the
real figure sits somewhere below 2.5x for a reason this arithmetic cannot see.
Nothing here has profiled the CUDA Sol path per stage. **17a is no longer a
tie-breaker between two numbers; it is the only way to get a second one.**

### Three gates, all cheap, before any CUDA is written

**17b and 17c are done; 17a is still scaffolded and its entry points raise.**
The scaffolds exist to fix the metric, the control and the sampling before the
measurement, because in each case a wrong design produces a plausible number
rather than an error -- which is exactly what 17b then demonstrated, by shipping
three of them past its own first run. See each item below for current state.

**17a. Profile the CUDA exact kernel per stage.**
`bench/profile_sol_stages.py`. One `ncu` run settles MMA-bound against
staging-bound, which is the whole uncertainty in the 2.5x estimate. While
there, record routed density: `sol_attn_stats()` counts dispatches, not blocks
(`vendor/sol_attn_minimax.py:102-104`), so **how much of Sol's work the exact
branch even is has never been measured**. Two hazards the scaffold already
carries: `ncu` needs the card alone, and Sol runs only inside the sigma window,
so an unfiltered capture mixes 5 dense sage steps into the average.
*Blocker: an idle GPU.*

**17b. Decompose Sol's error on captured activations.**
`bench/analyze_sol_error.py`. Split total error into sparsity error (eager Sol
against dense) and quantization error (CUDA Sol against eager Sol at the same
tau). **If quantization error is small against sparsity error, a 16-bit PV buys
nothing measurable and 17 closes without a kernel being written**, so this runs
before anything expensive.

`bench/check_solattn_correctness.py` already computes both quantities and
already says why they do not count: at T=512 on `torch.randn` it prints DOUBLY
PESSIMISTIC, DO NOT QUOTE, because a near-uniform softmax leaves a block router
nothing to find and 8 blocks is a different regime rather than a small version
of production. 17b is that same decomposition somewhere the premise holds.

**Done 2026-08-17, and it does not close 17.** Across all twelve rows the
quant/sparsity ratio runs 14.43% to 62.20%, against the 5% threshold the script
uses to retire the question. Quantization is a measurable share of total error,
so a 16-bit PV is not ruled out on these numbers. Block 49 climbs monotonically
across the trajectory (49.42%, 55.56%, 62.20%).

**Three things this gate got wrong about itself**, all now stated in
`bench/analyze_sol_error.py`'s module docstring rather than only here.

Its `rho` column was not a valid cosine, because `quant_l2` was normalised by
`‖out_eager‖` while the other two used `‖out_dense‖`. Its `cosine_sim` returned
values above 1.0 at production tensor size. And **its eager Sol diverged from
the vendored oracle**: `colmean` was normalised on the key-block axis rather
than the query-block axis, which is identical at every block-aligned length and
wrong at every ragged one -- rel_l2 0.166 against the oracle at t=1000, where
production S = 98498 = 1539*64 + 2 is ragged.

All three are fixed and a calibration gate now runs before any capture is read.
The twelve rows above are post-fix; the correction moved each by under 1.2
points and changed no conclusion, because the mis-normalised row and column are
two entries out of 1540 at production scale.

The instructive part is how it survived a day. The sibling gate in
`bench/simulate_track_b_lite.py` tests at t=320 = 5*64, block-aligned, so it
could never have caught a ragged-block divergence -- while naming ragged-block
handling as a known failure mode in its own refusal text. A control whose
fixture cannot express the defect is not a control for it.

*Blocker: none.* ~~Next action is chunking the oracle, so the gate can run at
production S instead of inferring from t <= 2001.~~ **Refuted 2026-08-19**
(`docs/roadmap.md`, the error-decomposition section): chunked to production
length the gate goes red on tie flips while both implementations are correct.

**The next action instead, 2026-08-20:** split the quantization error into
its INT8 QK half and its INT8 PV half. `bench/analyze_head_magnitudes.py`
found per-head quant error tracks Q and K magnitude (Spearman ~0.4-0.5) and not
V (~0), on both checkpoints; a 16-bit PV changes the V side. If that holds
under a proper split, 17 targets the wrong operand. **`bench/simulate_track_b_lite.py`
cannot do this split**, and its own docstring says why: its "fp16 PV" arm is
the unquantized reference, so it adds nothing beyond the quant/sparsity ratio
-- and it refuses to run. The split needs the kernel's quantization scheme
mirrored in the eager reference, which is a plausible-number trap; scoped
here, not built.

**17c. Capture a reference-heavy render.** NEW, and it gates the value of the
other two. Every Sol measurement in this repo -- including the
`2026-08-15_dense_124f_1344x768` capture this was written against, which has
since been removed -- was t2v on fl2va with zero references, while the work
actually being done is reference-heavy.
Reference rows are pinned by `sink_conditioning`, so reference-heavy is where
Sol has the **least** room: every existing ratio is an optimistic bound for the
real workload, and a bigger pinned region makes a 16-bit PV *more* expensive,
not less. `h3_capture.py` is env-driven and graph-agnostic, so this is one
render with `H3_CAPTURE` set against `h3_probe_sol_on_refs_api.json` -- no code
change. Run 17b on both captures and report both.

**Done 2026-08-17**, on `h3_probe_capture_ref3_api.json` rather than
`h3_probe_sol_on_refs_api.json`: 362 frames at 1024x768 with three references,
`S = 98498`, captured at seven blocks single-step and at four blocks across
steps 3/8/14. The prediction above held -- reference-heavy is not a smaller
version of t2v, and the multi-step arm additionally showed the ratio moving
along the trajectory, which a single-step capture cannot see. *Blocker: none.
Outstanding: the multi-step directory has no `manifest.json`, because it was
written before the manifest tooling existed.*

### The port itself, if the gates pass

| file | work |
|---|---|
| `sage_attention/sol_attn_exact.cu` | the real work. `pack4u8`/`mma_u8s8`/`__dp4a` l-sum become `pack_bf2`/`mma_bf16`/a plain float sum; `PKC` 2 to 4; `LDV` 64 to 128 bytes; the epilogue drops the `vsc` multiply and the 255. Removes the `log2(255)` exponent fold and the num/den-quantize-identically subtlety rather than adding one. |
| `sage_attention/sol_layout.cuh` | `swz_v` re-derived for a 128-byte V row. The header says to enumerate both 16-lane LDS.64 phases against 32 banks; that is not optional. |
| `sage_attention/sol_attn_vtranspose.cu` | a bf16 variant: transpose without quantize. |
| `sage_attention/sol_attn_route.cu` | **the dangerous part.** Both route kernels hand over `o_part * (255/vsc)` and `l * 255` to land in the INT8 exact kernel's units (lines 252-260, 546-551). A 16-bit branch wants plain units. Ten lines -- and `coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:19-21` warns this class of drift "is invisible to either side's own test". |
| `sage_attention/sol_attn.cu`, `dlpack_bindings.cpp`, `backends/cuda/__init__.py`, `constraints.py` | plan sizing (`vTi` doubles), the flag, validation. |
| `coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/CMakeLists.txt:135-139` | sol sources are listed explicitly; a new `.cu` needs adding, a template parameter does not. |
| `tests/test_sol_attn.py` | parametrize the existing cosine cases over the flag. The eager reference is full-precision and already the oracle for both, so **a case asserting 16-bit scores no worse than INT8 is free, and it is the one that would catch a bad handover.** |

Ours: a node input appended **last** (`vendor/sol_attn_minimax.py`, the
widget-order rule), a `SOL_CUDA_DEFAULTS` key, an arm in
`check_solattn_correctness.py`, then regenerate, restart, smoke.
`check_sol_kernel.py`'s `schema` case picks the new key up with no edit.

**Cost:** 17a and 17b are hours each. The port is 2-4 days for someone
comfortable with MMA fragment layouts, most of it verification rather than
writing. A full `comfy_kitchen` rebuild is ~4 minutes (from the `build/`
timestamps, not timed).

**Decision it changes:** whether the CUDA Sol path gets a precision knob at
all, and if so whether it ships on. Also whether the standing suspicion --
that Sol's all-INT8 branch discards what sage's 16-bit PV pays 1.58x for --
survives contact with a measurement.

**Blocker: 17b, and it may close the entry.** The motivating premise -- that
Sol's INT8 PV discards what sage's 16-bit PV pays for -- rested on an
fp8-vs-fp16 accuracy ratio that was **withdrawn on 2026-08-16 as untrusted and
removed from this repo** (`docs/evidence.md`). So the premise now has no
supporting number at all, in either direction: INT8-V is unmeasured, sage's fp8
and fp16 kernels differ in both PV operands so nothing isolates V, and the
durable half of the sage verdict is perceptual -- a verdict about *dense*
attention, which does not transfer unexamined to a kernel that is also dropping
most of the blocks.

---

## Completed, kept for the record

- **Should we load through `DiffusionModelLoaderKJ`** — no, decided
  2026-08-13, recorded so it is not re-opened on the strength of its feature
  list. It offers six inputs where stock `UNETLoader` has two, and **three of
  them mutate global state as a side effect of loading**: `sage_attention`
  patches comfy attention globally, `enable_fp16_accumulation` flips
  `torch.backends.cuda.matmul.allow_fp16_accumulation` (and its `else` branch
  turns it *off*), `patch_cublaslinear` swaps `nn.Linear`. In a repo whose
  premise is that two arms differ in one variable, that is three new ways for
  a graph to silently change numerics. `sage_attention` is specifically
  hostile here — we install our own H3 patch and compose Sol-Attn onto it in a
  pinned order that `SageChainAssert` verifies, so a second global sage path
  would either double-patch or shadow ours. Its fp8 `weight_dtype` options are
  moot on an already-int8 checkpoint, and the fp16-accumulation flag cannot
  reach sage's kernels at all (confirmed against the fork's source): it would
  change Linear numerics while telling us nothing about attention. `VAELoaderKJ`
  is a TAE selector we already get through the preview override;
  `GGUFLoaderKJ` and `CheckpointLoaderKJ` do not apply.
- **int8 VAE decode after ComfyUI `2a68ce33`** — re-measured 2026-08-11 at
  1.28x against 1.29x recorded. Unchanged; the figure stands.
- **Does per-call attention peak predict process peak** — no. See
  `attention.py`'s clone docstring. This is why several entries above are
  scoped to per-call claims.
- **Does chunking fragment the allocator** — no. `allocated` and `reserved`
  track within 8 MiB on all three arms.

## 20. The SLA LoRA under its training router

**PARTLY RETIRED 2026-08-31.** The router half cannot be run here any
more: `MiniMaxH3SLARouter` and `vendor/sla_sparse_triton.py` were removed
(the arm itself was retired 2026-08-28 by owner decision, for the reason
below -- it patched `diffusion_model.blocks`, 50 of the 52 `Attention`
modules the LoRA adapts, so it never answered the question it was named
for). **The second half is still open and still reachable**: whether the
Turbo-SLA LoRA behaves differently under Sol-Attn than under dense
attention, which the two shipped `h3_probe_turbo_768p_sla*` arms address
without the node. Sol's `top-k (SLA)` selection is NOT a substitute for
the router -- it keeps a pooled term for every unpicked block, which
`docs/SOLATTN.md` records as making it a third attention rather than a
cheaper spelling.

**Tests:** whether lightx2v's Turbo-SLA LoRA behaves differently under the
attention it was distilled with than under Sol-Attn or dense attention, and
whether a student trained to survive a top-k block cut produces attention that
is more block-sparse under Sol's router than a student that was not.

**What is reproducible, read 2026-08-20 from `coderef/LightX2V` and
`coderef/SLA` (source reads, not builds).** LightX2V's "SLA" is SLA v1's
sparse top-k branch only: mean-pooled q.(k - mean k) block scores, a hard
`topk` at `1 - sparsity_ratio`, no linear branch, no `proj_l`. The LoRA file
carries no extra weights (624 tensors, attn+mlp only; byte accounting against
the upstream PEFT file closes). The Triton kernel has no arch gate. So the
LoRA's training attention is runnable here. Two named gaps: the release ran
128/64 blocks through SageSLA's `sage2` operator (needs a SpargeAttn build;
the Triton path is 64/64), and LightX2V's inference config leaves the token
refiner on the router while ours patches it only on request.

**Arms:** the SLA LoRA under {router, Sol at the shipped tau, sage-only
dense}; the v1.1 768p LoRA under the same three as the never-SLA-trained
control. Plus two 4-step captures (v1.1, SLA) read by
`bench/sweep_routing_density.py` and `bench/analyze_sol_error.py`.

**Decision it changes:** whether SLA + Sol is the speed lever, and whether the
SLA probe graph should ship with the router rather than Sol.

**Blocker: none.** A correctness gate on captured activations comes before
any render through the kernel; `docs/roadmap.md` carries the day's plan.

**Measured 2026-08-20, the capture half.** Two 4-step t2v captures on the
fl2va base, one under the v1.1 LoRA and one under the SLA LoRA
(`2026-08-20_t2v_362f_1344x768_{v11,sla}`, Sol absent, blocks 0/24/49, steps 1
and 2). Sol's per-head error at tau 1.0 agrees between them within a few
percent at every cell (`bench/results/2026-08-20_sol_error_per_head_{v11,sla}.json`),
and Sol's routed density under its own router agrees within half a point at
every block, step and tau (`bench/results/2026-08-20_routing_density_{v11,sla}.json`:
block 0 about 22% at tau 1.0 and 16% at 1.3 on both; block 49 about 19% and
13% on both). **The SLA distillation did not make the activations Sol sees
sparser or easier.** What it changed is what the router renders, which is the
regime set's question, not the captures'.

## 21. The power-limit pair

**Tests:** what the 330 W board limit costs against the stock 450 W on one
graph and seed, which `docs/hardware.md` names as one of two things that would
settle whether sampling is core-clock-bound here.

**History:** a forward item three times (2026-08-17, twice on 2026-08-18),
blocked on sudo every time; never run, controlled or accidental. The card
was at 330 W on the morning of 2026-08-20 with no unit currently setting it.

**Decision it changes:** whether any timing taken at 330 W can be compared to
the 450 W records, and the bandwidth-bound reading in `docs/hardware.md`. The
rule written on 2026-08-17: a delta under ~2% supports bandwidth-bound; a
delta near the clock delta supports L2-bound.

**Run 2026-08-20**, on the 4-step 768p turbo graph: 5.8% sampler cost at
330 W against a 12.5% core-clock delta, within-arm spread 0.2%
(`bench/results/2026-08-20_power_limit_pair.jsonl`, verdict in
`_verdict.json`). Partly core-clock-bound. Closed for this workload; the
16-step all-refs pair is the remaining optional arm.

## 19. Does convrot's rotation reach Sol's routing or Morton's ordering?

**Tests:** whether the shipped weights' quantization changes the geometric
premise both Sol-Attn and the token orderings rest on. Sol reorders and routes at
a 64-token block, an 8x8 tile in 2d and 4x4x4 in 3d, and the ordering's whole
argument is that tokens adjacent in that tile carry similar q/k, so grouping them
yields a tight centroid and the block becomes skippable at a given `tau`.

**Three properties of the shipped checkpoints attack that premise
independently**, and none has been tested against the ordering:

- **int8** adds quantization noise to q/k, inflating within-block variance
  whatever the ordering, so the locality signal competes with noise the ordering
  cannot reduce.
- **convrot** applies a rotation: the file stores `W @ H^T` in a Hadamard
  basis rather than the weight (measured 2026-08-21,
  `bench/analyze_quant_delta.py`). So "similar in value" is judged in a rotated
  basis, and whether spatial adjacency still maps
  to proximity there is stated nowhere.
- **pruned** means the distribution being exploited is the pruned model's. A
  locality result need not transfer to unpruned weights, and pruning may itself
  have removed the structure that carried spatial coherence.

**Evidence kind: inferences from source reads, not measurements.** Together they
are a mechanism that would explain the retracted Morton speed claim, which
`docs/SOLATTN.md` records as not surviving an int8 path. That does not make them
true.

**Method, and the control is already on hand.** `fp8_scaled` exists as a matched
build of **both** models, so `int8_convrot` against `fp8_scaled` within one model
role is a one-variable comparison — the quantization moves and the role does not.
Compare the routed density and the centroid variance the router reads, not
wall-clock. A `w4a8_mixed` fl2va build gives a third point if the first two
separate. No new render is needed if the captures record which build produced
them.

**Blocker: provenance, and narrower than first written.** Corrected 2026-08-17
the same day, after a peer session checked the schema rather than reading my
summary of it.

- **For captures, the build is recoverable.** `models` in
  `docs/capture_manifest_schema.md` requires `unet`, `clip` and `video_vae`, and
  those filenames are self-describing: `_int8_convrot`, `_fp8_scaled` and
  `_w4a8_mixed` are distinguishable by name. So any conforming manifest already
  says which build ran. The first version of this entry claimed the opposite.
- **What is actually broken is the assertion.** `weight_quantization` and
  `vae_quantization` exist as properties and appear in **no** `required` list, and
  `bench/check_capture_manifest.py` never inspects either. A manifest can omit
  both and pass green. That is the tight piece of work: make them required, then
  assert them.
- **For bench runs there is no manifest at all**, so a timing carries no record
  of the weights it was measured on. That is the real hole, and it is the same
  shape as the host power-limit gap `docs/evidence.md` now records: the substrate
  is knowable at run time and nothing writes it down.

Settle the bench-run half before comparing anything whose evidence is a timing.
The centroid and density comparison this item actually calls for reads captures,
so it is not blocked on that.

## 22. Pruning sensitivity: does the same AdaLN residual move ref2va's output more than fl2va's?

**MEASURED AND REFUTED 2026-08-21.** Eleven forwards at 768x768, 124 frames,
seed 730451892, one sampler step; record in
[`bench/results/2026-08-21_pruning_sensitivity.json`](../bench/results/2026-08-21_pruning_sensitivity.json),
grader `bench/grade_pruning_sensitivity.py`, driver `bench/run_pruning_arms.py`.
ref2va moves 0.86 of what fl2va moves on the reference input and 0.81 on the
plain one -- inside the [0.5, 2] refutation band and in the opposite direction
to the hypothesis. **The prediction attached below was also wrong**: it said the
velocity would move under 1% and it moves 5.6-9.4%, so the pruning is not
invisible at the output, only equally (in)visible on both checkpoints and
smaller than the int8-vs-fp8 difference already shipped. `docs/evidence.md`
carries the consequence. The design below is left as written, including the
prediction it got wrong.

**How the "decision it changes" clause resolved.** The fl2va hedge for
reference work has no basis in *sensitivity*: both checkpoints respond to the
pruning by the same amount, so preferring the unpruned 34 GB ref2va file over
the pruned one cannot be justified on ref2va-specific grounds. What stays open
is the plain pruned-vs-unpruned preference on either checkpoint, which is now a
perceptual question rather than a numerical one and would need its own blind
session under `docs/eval_comparison.md` section 3. This entry never claimed one
and still does not.

**One clause was graded under a reading this entry did not specify.** The rule
below asks whether the profile "opens at or after the blocks where the
reference rows carry the modulation residual" without giving a threshold. The
grader implements the weakest testable reading — deepest block at least twice
block 0 — and says so in its own docstring. Under that reading the profile does
open late, but the shape is not a monotone opening: q error climbs to a peak at
block 36 and falls back at 49, on both checkpoints. The verdict did not turn on
this clause; it was refuted on the ratio.


**Tests:** whether the rank-8 AdaLN pruning, which perturbs the modulation
output identically on both checkpoints
([`bench/results/2026-08-20_adaln_pruning_residual.json`](../bench/results/2026-08-20_adaln_pruning_residual.json):
same size, same per-parameter and per-timestep shape, ratio 0.97-1.01),
moves the *network output* by a different amount on ref2va than on fl2va.
That record measured the perturbation; this measures the sensitivity to it.
Designed 2026-08-20 evening at the owner's request, not run.

**Why it is not already answered.** A perturbation of equal size can matter
more to one set of weights than another. There is a structural reason to
suspect ref2va: the residual grows with t and is largest at the 0.999
condition timestep, reference rows sit at 0.999 on every step, and the ref2va
task rides on those rows. Against it stands a magnitude argument: on the bf16
final layer the truncation is a 2e-4 perturbation, the int8 linears both
builds carry are tens of times larger on the same activations, and the pruned
AdaLN is closer to bf16 than the unpruned int8 AdaLN is. An argument is not a
measurement.

**Decision it changes.** `docs/evidence.md`'s "pruning closed" paragraph, and
whether reference work should prefer the unpruned `int8_convrot` ref2va file
(34 GB, offloaded on this card, slower) over the pruned one in
`internal/recipes/`. A perceptual consequence, if the numbers say there could
be one, would then need its own blind session under `docs/eval_comparison.md`
section 3; this entry never claims one.

**Method: a fixed-input forward, not a render.** The first sampler step of any
graph is a controlled comparison between two checkpoints: the latent is the
seed's noise, the conditioning is identical, and the forward is deterministic,
so the different-sample rule in `CLAUDE.md` does not apply. Two arms that
differ only in `unet_name`, pruned against unpruned, compared at that step.

- **Graphs:** the two capture graphs, `workflows/h3_probe_capture_ref3_api.json`
  (ref2va, three image references) and its twin
  `workflows/h3_probe_capture_ref3_fl2va_api.json` (fl2va), because they wire
  sage without Sol and so record the true attention inputs; plus
  `workflows/bench/h3_text_to_video_stamped_api.json` for the no-reference
  input. The unpruned arms patch `--set LABEL:UNETLoader.unet_name=minimax_h3_{fl2va,ref2va}_int8_convrot.safetensors`
  (the files are on disk since 2026-08-20; `comfy/model_detection.py` handles
  the time-embedder layout, and the int8 AdaLN carries its own `comfy_quant`
  marker at group 64, so no code change is expected). All arms patch
  `BasicScheduler.steps=1`: the first sigma is the schedule's maximum
  whatever the step count, so the forward is the same one a 16-step render
  would make first, and nothing after it is needed.
- **Canvas:** 768x768, the cheapest legal canvas (`docs/h3_resolutions.md`,
  0.33x attention), at 124 frames, the bottom of the trained range. The owner
  has not yet confirmed the canvas; the standing rule is to ask before the
  card is touched. The references keep the capture graphs' fit (2048 short
  edge, no upscale), so the reference rows are the ones the hypothesis is
  about. Sol's 60k-token floor is irrelevant here because Sol is off in these
  graphs by design.
- **Instrument:** `H3_CAPTURE="dir=...,blocks=0:12:24:36:49,steps=0,cycle=1"`
  at server start (step 0 is the fixed-input forward; the module's default of
  step 1 exists for trajectory statistics, which this is not), giving q/k/v
  at five depths per arm. Plus one addition to `h3_capture.py`: an optional
  `final=1` key that writes the final layer's output (the velocity) at the
  captured steps, the one number that is "the network output". Its
  deliberate violation: a run with `final=1` on a graph where the tap is
  bypassed must write nothing, and a written tensor must reproduce the
  sampler's first-step update to bf16 precision. If the tap is not built,
  the fallback is the decoded one-step frames, a monotone but nonlinear
  proxy, and the entry must say so.
- **Arms:** {t2v, ref3} x {fl2va, ref2va} x {pruned, unpruned}, eight
  forwards; one **repeat** of a pruned arm with a changed output prefix
  (defeats the node cache, which returns a cached result for a byte-identical
  resubmission) to measure the determinism floor; and two **scale
  references**: the `fp8_scaled` build of each checkpoint on the ref3 input,
  so "pruned vs unpruned" can be read against "int8 vs fp8", a quantisation-
  size difference the repo already lives with. Eleven forwards; each is
  seconds of compute, and the unpruned arms cost a model load from `Storage`
  each switch, so the whole thing is under an hour including the two server
  restarts (capture armed, then plain). Seed 730451892 throughout. Matched
  seeds across arms are what make the input identical.
- **Grader:** a new `bench/grade_pruning_sensitivity.py`, reading the
  captures by filename: per arm pair, relative L2 and cosine of q, k, v per
  block and per head, and of the velocity; records to
  `bench/results/<date>_pruning_sensitivity.json` with filenames only.

**Pre-registered prediction and decision rule.** Let `S(ckpt, input)` be the
relative L2 between the pruned and unpruned velocity on the same input, and
`floor` the repeat arm's own relative L2.

- Prediction from the magnitude argument: `S` under 1% on both checkpoints,
  and well under the int8-vs-fp8 reference on the same input.
- The hypothesis **survives** if `S(ref2va, ref3) / S(fl2va, ref3) >= 2` with
  both values at least 10x `floor`, and the per-depth profile shows the gap
  opening at or after the blocks where the reference rows carry the
  modulation residual rather than uniformly from block 0.
- It is **refuted** if the ratio sits in [0.5, 2] with both values above the
  floor, or if both values sit within 10x of the floor (then the pruning is
  invisible at the output on both, and the ratio is noise over noise).
- Any other outcome, including `S` above the int8-vs-fp8 reference on either
  checkpoint, is a finding about the pruning itself and reopens the
  `docs/evidence.md` paragraph regardless of the ratio.

**Blocker:** none technical. The unpruned file has never been loaded through
ComfyUI on this card; the first action is a one-step render with it alone,
output read end to end (a 34 GB file on a 24 GB card goes through partial
offload, and that path is the untested one). Then the canvas question to the
owner, then the tap, then the arms.


## 23. What INT8 actually costs at run time, per module kind

**Tests:** how much of a module's `int8_convrot` error is the WEIGHT rounding
and how much is the ACTIVATION rounding, per module kind, at production
geometry. Everything this repo has measured about int8 fidelity is the first
term; the second has never been looked at.

**Why it is not already answered.** `int8_convrot` is W8A8, and
[`research/comfyui_h3_t2va_trace.md`](research/comfyui_h3_t2va_trace.md)
sections 1.7-1.8 trace why: `int8_linear` rotates the activation online with
the same Hadamard, quantises it **per token**, runs an int8 GEMM whose int32
accumulation is exact, and scales in fp32 — "all the error is in the two
roundings". Every quant record here
(`2026-08-21_quant_delta_*`, `2026-08-28_quant_hotspots_ref2va`,
`2026-08-29_int8_convrot_headroom`, `2026-08-30_pdd_quant_interaction`) is a
stored-weight distance. `docs/evidence.md` states that caveat correctly on the
source measurement; two files then cited those records past it, and both were
corrected on 2026-08-31.

**Decision it changes.** Three, and none of them can move without this:

- **Whether `attn.out_proj` deserves different treatment.** It is the worst
  kind on stored weights (1.18x qkv_proj on the mean, row_rel p95 0.0128
  against 0.0101) and its input is the attention output, the most
  outlier-heavy activation in the block. Those two facts point at different
  levers and the records cannot separate them.
- **Whether `convrot_groupsize` is a live knob.** The encoder sweep found it
  flat, on weights. The rotation's whole job is to spread outliers before
  rounding, and the activation is the side with the outliers, so that result
  does not transfer. The DiT's dimensions make this specific rather than
  hypothetical: `_build_hadamard` wants a power of 4 dividing `in_features`,
  so `attn.qkv_proj` and `mlp.fc1` (5376 = 2^8·21) are **capped at the shipped
  256**, while `attn.out_proj` (7168 = 2^10·7) and `mlp.fc2` (14336 = 2^11·7)
  admit **1024**. The kind that is worst is one of the two that can take a
  wider group.
- **`docs/research/h3_dit_implementations.md` §10.5**, which until 2026-08-31
  said building a better quantised DiT "does not exist as an option" on the
  strength of the weight-only lane.

**Method: captured activations, then everything offline.** One capture, scored
many ways — the capture-broadly-first rule, and the reason not to design this
as a sweep.

- Capture `attn.out_proj` **input** (and one `qkv_proj` input as the control
  kind) at 1344x768, one or two `(block, step)` pairs. At 98k x 7168 in bf16
  that is ~1.4 GB per pair, a third of what the existing qkv captures cost.
  Every capture inventoried in `2026-08-30_capture_inventory.json` was deleted
  the day it was written, so this needs a fresh render, and
  `bench/restart_comfy.sh`'s arming rule applies to the server that runs it.
- Offline, per kind, on the same captured `x`: `int8_linear(x, W_q, s,
  convrot=True, gs=256)` against `F.linear(x.to(bf16), W_ref.to(bf16))` — the
  runtime error. Then two decompositions against the same reference:
  exact weight with quantised activation (activation term alone), and
  quantised weight with exact activation (weight term alone, which is what
  every existing record measures).
- Only then sweep `gs` in {256, 1024} on `out_proj` and `fc2` against the same
  `x`. It is a re-score of one capture, not a second render.

**Pre-registered prediction and decision rule.** Let `E_rt`, `E_act`, `E_wt` be
the three relative L2s above, per kind.

- Prediction: `E_act > E_wt` on `attn.out_proj`, and the gap between kinds in
  `E_rt` does **not** follow the stored-weight ranking.
- The stored-weight lane's implicit claim **survives** if `E_wt / E_rt >= 0.8`
  on every kind, i.e. the weight rounding dominates and the existing ranking
  transfers. Then a re-bake is the right lever after all and groupsize is
  noise.
- It is **refuted** if `E_act > E_wt` on any kind, which makes every
  kind-ranking in this repo a statement about half the error, and makes
  `convrot_groupsize` the first thing to sweep.
- If `E_rt` is within a few percent of `E_wt + E_act` on all kinds the two
  terms are independent and can be reasoned about separately; if it is not,
  they interact through the rotation and only `E_rt` is quotable.

**Blocker:** one render's worth of card time, plus the capture spec on the
server that runs it. Nothing else — the scoring is CPU and needs no server.

**Narrowed 2026-08-31 by the weight-side pass.**
`bench/analyze_weight_outliers.py` answered everything answerable without a
card ([`research/quant_levers.md`](research/quant_levers.md)), which changes
what this entry has to establish rather than closing it:

- The second bullet above is no longer speculative on the weight side.
  `convrot_groupsize` 1024 buys 10.2% on `attn.out_proj` and nothing on
  `mlp.fc2`, and out_proj's excess is explained — its outliers span wider than
  256 channels, which a 1024-wide rotation reaches and the shipped one does
  not. What this entry must now decide is whether that survives to the output,
  and whether it pays for the **fused CUDA kernel** it costs:
  `_should_use_convrot_fused_kernel` requires `group_size == 256`, so an arm
  must be TIMED and not only scored.
- The first bullet is unchanged and is still the point of the entry.
- One prediction is now cheap to score against something real: the weight-side
  kind ranking is out_proj worst, and T2 says the runtime ranking will differ
  from it.
