# Check postmortems: what each defect taught

**This file is history. Nothing here is operative.** `docs/checks.md` is the
index and the standard; it is the file to read before adding a check. This one
holds the per-defect narrative that used to sit inside it: why particular checks
are shaped the way they are.

**Do not quote a count or a date from this file as current.** Every number below
was true when written and is frozen. The authority for what exists now is the
index in `docs/checks.md`, and for measurements `docs/evidence.md`.

Read it when you are investigating one specific check. Otherwise skip it.

Two things that used to be here have moved, because neither was an incident:

- **the run logs** — deleted 2026-08-17. Their counts had drifted three ways and
  contradicted the table in their own file, and the next full run supersedes them.
  Git history keeps them.
- **the four controls whose input could not fail** —
  [`bench/red/README.md`](../bench/red/README.md), beside the spine and fixture
  that exist because of them.

---

## A note on `check_lowvram_handoff.py`

Its name undersells it, and the name is why it looks droppable. KJNodes'
`MiniMaxLowVRAMAttention` does three things, and we already own one of them:

| what their node does | ours? |
|---|---|
| head chunking via `minimax_head_chunks` | yes, already our widget |
| block-level `h` release (the `[x]` hand-off) | no, the only additive piece |
| `sol_take_forward` so Sol-Attn keeps the low-VRAM path | no |

The division is currently clean and deliberate on their side: their
**attention** patch yields to ours (`if attn_key in m.object_patches:
continue`), while their **block** patch is unconditional and unguarded. If we
ever write our own block-level release, both packs would write
`diffusion_model.blocks.{idx}.forward` with no marker convention and
last-node-wins silently -- the collision class `reference_fit.py`'s
`_WRAP_MARKER` exists to prevent. **Decided 2026-08-13: keep the split, do not
reimplement.** The interop cases stay because that boundary has already
produced one real bug (the `clone_v` regression at `head_chunks=4`).

## A note on `check_solattn_correctness.py`: updating an oracle changes the check

Re-vendoring `bench/_sol_attn_reference.py` on 2026-08-14 (`ad9a4a8` ->
`c04ef20`) added `centroid_tail`, defaulting **True**. The Triton kernel has
no such parameter and runs the per-row mode. So the moment the oracle was
updated, every Triton case was grading the kernel against a different
algorithm than the one it implements -- and **all of them still passed**,
because the two modes differ by cos 0.9988 and the bar is 0.998. The bar was
looser than a whole-branch change to the algorithm.

Three things worth keeping from that:

- **Nobody edited a case, and the cases broke.** The defect entered through a
  dependency the check trusts. A check is only as pinned as its oracle, and
  the oracle here is deliberately something we do not control.
- **It passed, which is the bad outcome.** Had it gone red the re-vendor would
  have been examined immediately. Passing is what let it sit.
- The fix was not to tighten the bar but to **measure which mode each kernel
  is on** and grade it against that. The mode is now printed on every run,
  for both kernels, because the source does not document it and reading the
  kernel to decide would be an inference where a measurement was available.

The general form: when an oracle gains an option, every assertion against it
inherits a new case, exactly as CLAUDE.md says an "off"/"absent" state does.

## A note on the bench: an error that flatters nothing gets missed longest

`bench_e2e_h3.py` hardcoded `"mode": "auto"` on its sage node. The
`mode="fp16 (most accurate)"` flip on 2026-08-13 changed `h3_config.py` and
every shipped graph and did not change the bench, so from that day every e2e
arm was compared against a baseline nobody runs.

The instructive part is why it survived. `auto` resolves to `fp8_cuda++`, the
**fastest** kernel. A fast baseline makes every competing arm look *worse*, so
the bug produced conservative numbers. Nothing looked too good to be true,
which is the signal people actually check for. **A bug that overstates gets
caught; one that understates does not.**

`check_generator_constants.py` already enforced "read the shared constant,
don't repeat it" -- for the generator. The bench was never in scope, and the
bench is where the numbers come from.


---

## Write the evidence kind inside the claim, not beside it

Not about a check, but about how a claim in this repo stops being true.

On 2026-08-13 an upstream finding arrived explicitly labelled unverified — a
read of somebody's source, not a build and not a measurement. It was written
into `attention.py` as "on inspection, is not", with the label dropped. Nobody
asserted anything false at any point. Each hop repeated the previous hop's
confidence and left the caveat behind, because a trailing "(unverified)" reads
as the sender hedging rather than as part of the claim, and hedges are what
get trimmed when text is copied.

The wording that survives a copy-paste states what kind of evidence it is
*inside* the sentence:

> **Reported, not verified:** the sm89 kernel appears already stride-aware on
> its output … That is a source read from upstream, **not a build and not a
> measurement**.

against the version that does not:

> …which reads as out of reach and, on inspection, is not. The sm89 kernel is
> already stride-aware on its output.

Both are honest when written. Only one is still honest after somebody quotes
half of it. This matters here more than in most repos because measured
numbers, upstream source reads and analytical estimates sit in the same
paragraphs, and six months later they are indistinguishable by tone.

### `SageChainAssert`'s call-time case cannot see sage

Found 2026-08-13 by removing Sol-Attn from a graph and watching the assert
fail for a reason unrelated to what changed.

`_exercise` pushes one tensor through the composed attention and requires a
routing counter to move. The counter it reads is resolved by scanning loaded
modules for a callable named **`sol_attn_stats`** (`assert_chain.py`)
— Sol-Attn's counters. `attention.py` exposes no counter of its own; the only
state it publishes is `reset_fallback_state`.

So on a sage-only graph the probe runs, sage routes it, nothing named
`sol_attn_stats` moves, and the node reports "the composed path was not
taken". Sage is fine. The instrument cannot observe it.

**Confirmed from the log, not only from the source.** The arm that passes
prints `[h3] chain assert, call-time: routed as sparse=1` — `sparse` is
Sol-Attn's counter name. The arm that fails prints the sage patch line
(`50 attention modules patched`) and no `[sol_attn]` lines at all, then
fails. Both halves of the diagnosis are visible in one run.

The inverse is the part that matters for graphs we actually ship: when the
assert passes at call time, **what it confirmed is that Sol-Attn routed the
probe**. It says nothing at call time about sage, which is the node it is
named for. And because Sol-Attn's module is imported process-wide whenever the
pack is installed, `sol_attn_stats` resolves even in graphs that do not use
it — so the check cannot distinguish "Sol is not in this graph" from "the
composed path was not taken".

This is the same check that, per the note at `assert_chain.py`, "ran
registration-only from the day it was written until 2026-08-11, and said so in
a line nobody read, under a final `chain assert ok`". The 2026-08-11 fix
closed the registration-only gap and wired the new case to the wrong module's
counters.

**Consequences, in order:**

1. The sage-only configuration is not merely unmeasured (open experiment 9),
   it is currently **unrunnable** with the shipped assert in the graph.
2. Every "routed as …" line in this repo's logs is a statement about Sol.
3. The fix is **not** a counter of our own, which was the first plan. The
   sage fork already exports `get_last_dispatched_kernel()` and
   `KNOWN_KERNEL_NAMES` as public API, set on every sage call including the
   sm89 fp8++ path. That proves routing *and* identity in one read, so the
   assert can require "landed on fp8_cuda++" rather than "something moved" —
   the claim this node's name has always implied and never made.

   **Two preconditions, both of which would otherwise reproduce today's false
   negative.** The value is `threading.local`, so the probe and the read must
   happen on the same thread: fine while `SageChainAssert` runs as a graph
   node, *not* fine if anyone moves it to an HTTP-side check, where it would
   return `None` and read as "sage did not route". And it is last-dispatch,
   not a count, so it must be read immediately after the probe.

   It also needs a reset to be sound. Without one the check reduces to a
   before/after comparison that is conclusive in one direction only: a change
   proves routing, but an unchanged value does not disprove it, since the
   probe may route to the same kernel a previous call already recorded and the
   thread-local persists across prompts on one worker. That failure mode is a
   **false negative on graphs that route consistently** — the same defect being
   fixed, wearing a better API. `_reset_dispatch_for_test` exists but is
   explicitly not public; upstream is promoting it through their downstream
   symbol process so it acquires a removal checklist. The repair waits for
   that rather than importing an underscore symbol.

**FIXED and verified 2026-08-13**, without needing the reset and without any
   new contract surface. The probe now fires on a **fresh thread**: the
   dispatch value lives on a `threading.local`, so a thread that has never made
   a sage call returns `None` by construction. The thread-locality that was the
   hazard becomes the mechanism.

   Two things the verification itself turned up:

   * **The off-thread probe does traverse the composed forward** — the open
     question when this was designed. Confirmed by the log: at 4608 tokens it
     produced `[sol_attn] sparse (1, 4608, 56, 128)`.
   * **That first attempt still failed, and correctly.** At 4608 the sparse
     patch *takes* the call and runs its own kernel, so sage never runs and the
     new check truthfully said so. The right probe size **inverted** when the
     instrument changed: the old counter check needed a probe large enough for
     the sparse kernel to fire, the new one needs a probe small enough for the
     sparse patch to decline, so the call falls through to sage. That is the
     composition claim this node is named for — *sage handles what the sparse
     patch does not* — and it had never been the thing being tested.

   The probe now reads the gate's own `min_tokens` from `transformer_options`
   and sizes to half of it, so lowering that threshold in a graph cannot
   silently push the probe back above it.

   **One probe was still not enough, and the reason is the same shape again.**
   The sparse gate *falls through* to our patch whenever it declines
   (`take = gate is not None and ...` then `return patched_forward(...)`), so a
   call reaching sage is consistent with two different worlds: composed and
   healthy with the gate declining, or composition dead with the gate never
   engaging. A small probe reports green in both — evidence that cannot
   separate "working as designed" from "the mechanism is absent", which is
   precisely the counter bug it replaced.

   It now fires a **pair**, pinning the gate from both sides:

   | probe | requirement | proves |
   |---|---|---|
   | below `min_tokens` | must reach sage | the fall-through works |
   | above `min_tokens` | must **not** reach sage | the gate is live and taking |

   The second assertion is sound *only* because of the fresh thread. `None`
   normally means "cannot tell"; on a thread that has made exactly one call it
   cannot mean anything else, so `None` after a large probe is positive
   evidence sage did not route it. The mechanism adopted for the baseline
   turned out to license the negative too.

   It also refuses to default a missing `sol_compose`. An absent key *is* the
   dead-composition case, so substituting 4096 would size a probe against a
   gate that is not there and call it green. Present → sparse expected; absent
   → sage-only, and the message says which was verified.

   Verified live, both configurations, and they are now distinguishable:

   ```
   composed:  sage routed a 2048-token probe on fp8_cuda++ and correctly did
              NOT get the 4608-token one, so the sparse gate at 4096 is live
              and sage is taking what it declines
   sage-only: sage routed a 2048-token probe on fp8_cuda++; no sparse patch
              published `sol_compose`, so this graph is sage-only
   ```

### The same defect pointed inward

Within an hour of writing the rule above, the same failure recurred in the
other direction. A number had been flagged — correctly, and by me — as
config-dependent and needing re-derivation per config. Two messages later it
was used as a known input to a solve, and the result pre-registered as a
prediction.

Nothing careless happened in between. **A caveat accepted about someone else's
number does not attach to your own later use of that number**, and no normal
process makes it attach: the caveat is filed as a fact about the old claim,
while the new claim is being built somewhere else. That makes it structural
rather than a lapse in attention, which is why "be more careful" does not fix
it any more than it fixes caveat decay.

The counter that seems to work: **when a caveated number becomes an INPUT,
re-read the caveat as a precondition of the new claim, not as history attached
to the old one.** If the caveat says "re-derive per config", then a solve
using it is blocked until that derivation exists — the same way a missing
argument blocks a call.

Worth pairing with a second habit from the same incident: check that the
quantity you are about to measure is the one that enters the model. That solve
was reformulated from a step count to a wall-clock share, and the instrument
already planned would have returned a precise value for the abandoned
variable — a real measurement of the wrong thing, which is harder to notice
than no measurement at all.

---

## Controls whose input could not fail

**Moved 2026-08-17 to [`bench/red/README.md`](../bench/red/README.md).** The four
instances now sit beside the spine and the fixture that exist because of them,
where somebody writing a harness will actually read them. They were not an
incident, so this collection was the wrong home.
## The node_id result, #18, and the connectivity numbers

Narrative behind three rows of the uncontrolled-requirement table in
`docs/checks.md`.

   **The `node_id` result is the one worth acting on, and it is finding 2.1's
   shape again.** `CLAUDE.md` opens with "The one rule that matters: saved
   graphs address everything by position", and a rename breaks every saved
   graph silently. But this repo's graphs are *generated from the schema*: a
   rename regenerates all 89 consistently, every check stays green, and the
   only artifacts that break are the owner's live graphs outside the repo —
   which no check can see. Verified there is no recorded baseline of `node_id`
   strings anywhere outside the generated graphs themselves; the only other
   occurrences are Python *class* names, which `CLAUDE.md` says are safe to
   rename.

   So the checks' inputs were regenerated from the same source as the violation.
   **A control whose input is derived from the thing it is checking cannot
   fail** — the fourth phrasing of the same defect, and this time it guarded the
   rule the repo names as its most important.

   **CLOSED.** The fix named here — a committed manifest of `node_id` strings
   that a check diffs against — landed in `1fe598b` as
   `bench/node_id_manifest.json` and `bench/check_node_ids.py`, with a red
   harness. `docs/checks.md`'s requirement table is the live status; this
   paragraph is the state before that commit and must not be read as current.

   **A written requirement is not a control, and this repo produced a clean
   instance the same day.** `docs/open_experiments.md` #18 requires that
   conditioning rows stay in the block population, because `kcvar` is a variance
   over every centroid the kernel pools. When `bench/analyze_routing.py` was
   built against that requirement, a mutant computing `kcvar` over the video
   blocks only **passed every control the script had**. The requirement was
   stated, agreed by two readers, implemented correctly — and enforced by
   nothing, so an implementation that violated it would have shipped green. It
   now has an explicit control asserting the population is the full pooled set
   and that the block count equals `sequence // 64`.

   The generalisation is worth more than the instance: **every "must" written
   into a spec is a candidate control, and the ones most likely to lack one are
   the requirements everybody agrees with**, because agreement feels like
   coverage. When adding a requirement to a doc, ask in the same breath which
   assertion would go red if someone ignored it.

   **The connectivity numbers had the opposite problem: no instrument at all.**
   The 60%/90% connected-block figures in `sol_curves.py`, `docs/morton.md` and
   the node tooltip were not computed by anything in `bench/` --
   `analyze_morton.py` reports radius, fill and neighbour retention, and has no
   notion of connectivity. They reproduced exactly when re-derived, so this was a
   missing instrument rather than a bad number, but it was the only load-bearing
   figure in the Morton work that no committed script could regenerate.

   **CLOSED.** `analyze_canvas_geometry.py`'s `connected_frac` is that
   instrument, and it has been shown red by mutating it to 26-neighbour
   adjacency. Transcribing this paragraph's "no instrument exists" into
   `docs/checks.md`'s live table on 2026-08-17 put a false status claim in that
   table for an hour, which is why its cells must be derived from `bench/` and
   never copied from here.
