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
