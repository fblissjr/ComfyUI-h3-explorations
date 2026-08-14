# Open experiments

Last updated: 2026-08-14

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

**Tests:** whether 16 is still the right step count.

**Why it matters:** it was measured at 362 frames on 2026-08-06, and 362 is
now known to be illegal — the reference applies its 15s ceiling after the
17n+5 snap, so 345 is the maximum. The rejection of 12 steps rested on a real
gate (the third shot of a three-shot prompt silently stops happening) but was
measured on a configuration that no longer exists.

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

**Tests:** whether feeding a downscaled clip reduces the sequence, or only
decode time.

**Why it matters:** general prompting research recommends downscaling a
reference video hard when it is only providing motion, and our loaders sit at
native (`custom_width: 0`). A reference video already costs rows in two
places — the DiT reference block, and vision blocks inside the *text* segment
at 2 fps, ~519 tokens per merged pair (`docs/h3_references.md`). The DiT side
is resized to canvas, so input resolution should not touch it. Whether the
**Qwen vision** side tokenizes at input resolution is the open half, and if it
does, downscaling is a real lever on the ceiling rather than a load-time
saving.

**Cost:** two Preflight reads. No render, no GPU time, no sampling.

**Blocker: none — this is the cheapest unrun item here.**

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

**Blocker: none, and that is the point.** Every other entry blocked on owner
judgment has been listed that way since 2026-08-13 without anyone asking what
would make the judgment cheaper. A screen that can go red is worth more than
another parked question, and this repo's own rule — a check is not trusted
until it has been shown red for the right reason — applies to the instrument
before it applies to anything it grades.

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
