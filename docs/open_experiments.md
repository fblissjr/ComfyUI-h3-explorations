# Open experiments

Last updated: 2026-08-13

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
> (`__init__.py:373-381` at `842c4ea`) unless a delegate is published. So head
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

**The dense count `d` is an ASSUMED INPUT and it is not verified.** It is the
same config-dependent quantity flagged above, and it was nearly hardened into
a point prediction by using it in a solve. Sensitivity at N=16:

| dense steps | per-step factor | slower by | cost under-expressed |
|---|---|---|---|
| 3 | 0.9573 | 4.3% | 5.3x |
| 4 | 0.9680 | 3.2% | 4.0x |
| 5 | 0.9744 | 2.6% | 3.2x |
| 6 | 0.9787 | 2.1% | 2.7x |

`d=5` comes from an internal audit's CPU reconstruction, which reported it
reproduced an independently recorded "6 dense at 20 steps". **That control does
not reproduce here.** Three mappings were tried — shifted sigmas against
shifted percent bounds, against plain bounds, and unshifted sigmas — and none
yields 6 at N=20; two yield 5 at both 16 and 20. So either the reconstruction
is right and this closed form is missing something about `ModelSamplingAV`'s
dual schedule, or the reproduction claim was wrong. Not resolved.

**So the prediction is directional, not numeric.** Across the whole plausible
range of `d` the sign is the same and the magnitude is 2.7-5.3x understated,
so the pre-registered discrimination survives: **near 0.96-0.98x** against
**0.99x or better**. Do not quote 0.974 as the expected value.

If it comes back at 0.99x or better, the per-step-launch-overhead model is
wrong and the extra cost is something else. If it lands in the 0.96-0.98 band,
entry 7's "wall-clock already answers whether to chunk" had the sign right and
the magnitude wrong by 3x or more — the difference between a free knob and a
real cost.

**`d` is measurable, not merely uncertain, and should be measured.** Turn
`verbose` on in SolAttnPatch for one render at the arm's own scheduler, steps
and shift, and count dense-path calls in the log. The Aug 11 log that produced
0.992x has already rotated away (all three surviving logs are Aug 13), so this
needs a fresh short render — cheap, and it converts the solve's input from an
assumption into a number.

**A null here does not close the area.** The out buffer is real and constant
regardless of where process peak is set, and removing it is not something any
consumer-side arrangement can do — it needs a kernel writing into a
caller-provided view. That is scoped upstream and unscheduled, not impossible;
see the amended `_chunked_heads` docstring.

**Blocker: none, it is running.** Round 2 of the VRAM probes.

---

## Completed, kept for the record

- **int8 VAE decode after ComfyUI `2a68ce33`** — re-measured 2026-08-11 at
  1.28x against 1.29x recorded. Unchanged; the figure stands.
- **Does per-call attention peak predict process peak** — no. See
  `attention.py`'s clone docstring. This is why several entries above are
  scoped to per-call claims.
- **Does chunking fragment the allocator** — no. `allocated` and `reserved`
  track within 8 MiB on all three arms.
