# What this repo checks, and why

Index of every check in `bench/`. One row per script: what it defends, what it
needs to run, and whether it has earned trust.

There is no test suite and no runner. Each script is standalone, prints its own
`ok` / `FAIL` lines, and returns a non-zero exit code on failure.

**Last full run: 2026-08-13 (evening)**, against ComfyUI `8f37cf8c` (v0.33.0),
KJNodes `6ab7e81`, on an RTX 4090, with ComfyUI running. There are now **15
`check_*.py`** and **62 graphs** (31 UI, 31 API). The five CUDA-free checks
that run in a second all passed, including `check_workflow_schema.py` against
every UI graph. `smoke_h3.py` passed earlier the same day; the CUDA checks
(`check_correctness`, `check_clone_v_wiring`) were **not** re-run after the
`mode="fp16 (most accurate)"` flip and should be, since that changes which
kernel they exercise.

**Partial run 2026-08-14**, same box. `check_sol_kernel.py` added and shown
red. `check_solattn_correctness.py` re-run and extended to the CUDA kernel
after `bench/_sol_attn_reference.py` was re-vendored from `ad9a4a8` to
`c04ef20`; it passes. Nothing else was re-run.

The counts in the paragraph above were wrong until this run -- it claimed
twelve checks and 24 UI graphs. A header that states a scope it no longer has
is the same defect this file exists to catch, one level up.

Nothing here is stale in the sense of failing against current upstream. The
staleness is in the documentation around them, which is what this file fixes.

## The standard

From `CLAUDE.md`, and it is the reason the last column exists:

> A check here is not trusted until it has been shown to go red for the right
> reason -- break the thing it guards, watch it fail, put it back.

This is not theoretical. Three checks written on 2026-08-10 passed for the
wrong reason on first writing, and one reported zero failures with the bug
reintroduced. **An empty "shown red" cell is a finding, not a formatting gap.**

## Running them

Most need neither CUDA nor a model and finish in about a second.

```bash
# from the repo root
python bench/check_reference_fit.py

# three of them import comfy without bootstrapping sys.path themselves
PYTHONPATH=/path/to/ComfyUI python bench/check_clone_v_wiring.py
```

**The `PYTHONPATH` requirement is undocumented at the point of failure.**
Exactly three checks need it -- `check_clone_v_wiring.py`,
`check_correctness.py` and `check_short_edge_override.py` -- and they die with
a bare `ModuleNotFoundError: No module named 'comfy_api'` without it. Every
other check runs from anywhere: those that import comfy bootstrap `sys.path`
themselves, and the rest never import it. That inconsistency is a real
papercut and is listed under Gaps.

## The index

| check | defends | needs | claims block | shown red |
|---|---|---|---|---|
| `check_correctness.py` | the patched H3 forward against the stock one; mean relative error 0.0732 on both the eager norm path and the fused RMSNorm+RoPE path | CUDA, `PYTHONPATH` | no | not recorded |
| `check_clone_v_wiring.py` | `clone_v` reaches the forward, and only on modes that earn it | CUDA, `PYTHONPATH` | no | not recorded |
| `check_override_routing.py` | which calls the attention override sends to sage and which it declines, including the fallback when the kernel raises | - | yes | not recorded |
| `check_lowvram_handoff.py` | **more than its name says.** Three cases are KJNodes interop (the `[x]` list hand-off, the fallback with a list input, `minimax_head_chunks` honoured from `transformer_options`); **two are ours regardless of KJNodes** -- the plain tensor path, and that head chunking at 1/2/3/7 groups reassembles bit-identically | - | yes | not recorded |
| `check_schema_defaults.py` | every node's `io.Schema` defaults match its `execute()` signature defaults, for all 7 nodes. ComfyUI does **not** inject a schema default for an input a prompt omits, so the two are independent and a split means the UI and the API path see different values | `PYTHONPATH` self-bootstrapped | yes | **yes**, on the real `length` split, 2026-08-13 |
| `check_ref_prompt_labels.py` | every ref graph's prompt names **exactly** the labels its graph wires. The tokenizer derives `<Picture i>` / `<Video k>` / `<Audio j>` from the wired sockets, not from the prompt, so the two drift silently -- and a video's soundtrack takes `<Audio 1>` ahead of a standalone clip, which is easy to number wrong by hand | - | yes | **yes**, both directions, 2026-08-13 |
| `check_prompt_guide_conformance.py` | every shipped ref prompt against the **official guide's own tables**, parsed at run time -- the six sections and their order, the `[...]` task-type vocabulary and its ` + ` combining rule, markers never crossing the visual/audio sets, and `<d>` only in `detailed_description`. Exists because `check_ref_prompt_labels.py` compares the generator to itself and so passed clean while all fourteen arms shipped a hardcoded `[reference generation]`. The `keyframe completion` case checks the GRAPH, not the vocabulary: the type is allowed only where the graph wires `MiniMaxH3AddGuide` (added to ComfyUI 2026-08-13, and merged with refs additively by `comfy/model_base.py`, so the mechanism now exists). Before that node it was rejected outright on the grounds that nothing could honour it -- reasoning that expired the day the node landed. Exits 2, not 0, when the guide is absent. Carries **one waiver**, `_STRUCTURE_PROBES`: `h3_probe_prompt_concise` is unstructured on purpose, so its section and prefix cases are skipped **by name and printed on every run** -- its markers, dialogue placement and label agreement are still enforced, proven by mutating it | the guide in `internal/` | yes | **yes**, six mutations incl. the fail-open guard, plus the waiver shown narrow, 2026-08-13 |
| `check_distill_settings.py` | **every** shipped graph, both forms: a turbo graph matches its LoRA's shift and steps, a base graph sits at the base checkpoint's 12/3, and the UI and API forms of each are paired and compared. Shifts *and* recommended step counts graded against the vendor README, not against itself. Exits 2, not 0, when that control is skipped | - | yes | **yes**, eight mutations, 2026-08-13 |
| `check_solattn_correctness.py` | Sol-Attn's Triton **and CUDA** kernels against the algorithm's own reference, cosine > 0.998, each graded in its own measured `centroid_tail` mode. Exits 2, not 0, when the CUDA arm is skipped for cause | CUDA, Triton, and a fork build of comfy_kitchen for the CUDA arm | yes | **partial**, 2026-08-14: the re-vendor exposed a real cross-mode defect (see below), but no case has been mutated |
| `check_sol_kernel.py` | that the installed `comfy_kitchen` still carries `sol_attn`, that it is the CUDA backend and not the eager reference alone, and that its signature still accepts the kwargs our node passes. **The first check here covering a call INTO a dependency we do not control**, and it covers exactly one contract. Presence is gated on a graph wiring `SolAttnMiniMax`, because Sol is shipped OFF and absent is the expected state; ungated it exits 2. Also pins `SOL_CUDA_DEFAULTS` against the inputs the node declares -- parsed with `ast` rather than imported, so the check stays free of ComfyUI | - | yes | **yes**, 2026-08-14: `present` with the stock PyPI wheel as the control, `schema` by simulating an upstream rename |
| `check_keyframe_canvas.py` | canvas derivation, plus the aspect and duration rules | `PYTHONPATH` self-bootstrapped | yes | not recorded |
| `check_reference_fit.py` | reference image sizing against both upstream rules, and that the stock resize becomes a no-op after our node | `PYTHONPATH` self-bootstrapped | yes | not recorded |
| `check_short_edge_override.py` | the reference short-edge override applies once and never leaks | `PYTHONPATH` | no | **yes**, documented in-file |
| `check_generator_constants.py` | the workflow generator reads upstream constants rather than repeating them | `PYTHONPATH` self-bootstrapped | no | not recorded |
| `check_workflow_schema.py` | saved UI graphs against a live ComfyUI `/object_info`, type-checking widget values positionally | **live ComfyUI**, or `--object-info` cache | no | not recorded |
| `smoke_h3.py` | the H3 chain composes and runs, after any node-pack update. **The only thing here that actually POSTs a prompt**, and on 2026-08-13 it was the only reason anyone discovered every API graph was unsubmittable -- `validate_api` was asserting the wrong shape, so no static check could have found it | live ComfyUI, GPU, model | no | n/a, it is a smoke test |

"claims block" means the file carries a `Claims, i.e. what breaks if a case is
deleted:` header enumerating what each case defends. Eight of fifteen do.

### A note on `check_lowvram_handoff.py`

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

### A note on `check_solattn_correctness.py`: updating an oracle changes the check

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

## What is deliberately not checked

`docs/open_experiments.md` is the other half of this document: seven things
this repo has decided **not** to measure, each with its cost, the decision it
would change, and the actual blocker. Read it before proposing a new check --
several obvious ideas are already there with a reason attached.

A suite of twelve **render** scenes is designed but not run. They are quality
gates for output, not code checks: each carries a pre-registered binary claim
and a predicted per-arm direction, and a human watches and listens. Nothing
executes them, and nothing here can -- judging whether a third shot happened
or two voices stayed distinct is not a `check_*.py`.

> Those live in `internal/`, which is **gitignored and not distributed**. If
> you cloned this repo you do not have them, and that is deliberate: they are
> the owner's working research notes, not shipped content. Everything this
> document describes is in `bench/` and is present in the clone.

Two things there have no code check and cannot get one until the graphs exist:

- **Two-stage split graphs do not exist yet.** When they do, the checks worth
  writing are: both stages read one schedule, stage 2 carries `DisableNoise`,
  the split point is inside the step range, and the finish-stage LoRA's shift
  matches the shared schedule. That last one is `check_distill_settings.py`
  extended, not a new file.
- **ref2v with an fl2v distill LoRA is out of distribution by construction.**
  All three turbo LoRAs are `fl2v`; the vendor lists ref2v distillation as
  future work. It is on the test matrix deliberately as an experiment, at
  varied LoRA strength and as a two-stage split. It must not be validated as
  a supported pairing, so `check_distill_settings.py` deliberately does not
  police which task type a turbo LoRA is loaded into.
  `docs/h3_ref2v_distillation.md` works out why it resists distillation, what
  to expect, and what failure to look for.

## Gaps

Ordered by how much they undermine the standard above.

1. **Ten of fourteen have no record of having been shown red.** Only
   `check_short_edge_override.py`, `check_distill_settings.py`,
   `check_schema_defaults.py` and `check_ref_prompt_labels.py` document their
   own calibration. For the rest, the repo's central trust standard is
   unverifiable from the artifacts. This does not mean they are wrong -- it
   means nobody can tell.

2. **Seven of fourteen have no claims block**, so "what breaks if this case is
   deleted" is not recoverable without reading the assertions and inferring
   backwards. The seven that have one are the model to copy.

3. **No runner.** Every script prints its own ad-hoc `ok` / `FAIL`, with no
   shared harness and no case registry. There is no way to run everything and
   get one report, and no reliable way to count cases.

4. **The `PYTHONPATH` split is invisible until it fails.** Three scripts
   require it and give a bare import error; six do not. Either all of them
   should bootstrap `sys.path`, or none should.

0. **Run `smoke_h3.py` after any change to the generator.** It is the only
   check that submits, and the class of bug it found -- a graph the validators
   pass and the server refuses -- is invisible to everything else here.

5. **`check_workflow_schema.py` and `smoke_h3.py` cannot run unattended.**
   Both need a live ComfyUI, and `smoke_h3.py` needs the models loaded too, so
   both are absent from any headless pass -- and a check that is silently
   skipped reads the same as a check that passed.
   `check_distill_settings.py` is the only one that answers this properly, by
   exiting 2 rather than 0 when one of its controls did not run. That pattern
   is worth copying.

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
modules for a callable named **`sol_attn_stats`** (`assert_chain.py:114-131`)
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

This is the same check that, per the note at `assert_chain.py:110-113`, "ran
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
