# What this repo checks, and why

Index of every check in `bench/`. One row per script: what it defends, what it
needs to run, and whether it has earned trust.

There is no test suite and no runner. Each script is standalone, prints its own
`ok` / `FAIL` lines, and returns a non-zero exit code on failure.

**Last full run: 2026-08-13**, against ComfyUI `12666983` (v0.32.0),
comfy-kitchen 0.2.31, KJNodes `6ab7e81`, on an RTX 4090, with ComfyUI running.
**All twelve `check_*.py` passed**, including `check_workflow_schema.py`
against all 24 UI graphs. `smoke_h3.py` was not run and does not count as a
pass: it needs a live server, a GPU and the models loaded.

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
| `check_distill_settings.py` | **every** shipped graph, both forms: a turbo graph matches its LoRA's shift and steps, a base graph sits at the base checkpoint's 12/3, and the UI and API forms of each are paired and compared. Shifts *and* recommended step counts graded against the vendor README, not against itself. Exits 2, not 0, when that control is skipped | - | yes | **yes**, eight mutations, 2026-08-13 |
| `check_solattn_correctness.py` | Sol-Attn's Triton kernels against the algorithm's own reference, cosine > 0.998 | CUDA, Triton | no | not recorded |
| `check_keyframe_canvas.py` | canvas derivation, plus the aspect and duration rules | `PYTHONPATH` self-bootstrapped | yes | not recorded |
| `check_reference_fit.py` | reference image sizing against both upstream rules, and that the stock resize becomes a no-op after our node | `PYTHONPATH` self-bootstrapped | yes | not recorded |
| `check_short_edge_override.py` | the reference short-edge override applies once and never leaks | `PYTHONPATH` | no | **yes**, documented in-file |
| `check_generator_constants.py` | the workflow generator reads upstream constants rather than repeating them | `PYTHONPATH` self-bootstrapped | no | not recorded |
| `check_workflow_schema.py` | saved UI graphs against a live ComfyUI `/object_info`, type-checking widget values positionally | **live ComfyUI**, or `--object-info` cache | no | not recorded |
| `smoke_h3.py` | the H3 chain composes and runs, after any node-pack update | live ComfyUI, GPU, model | no | n/a, it is a smoke test |

"claims block" means the file carries a `Claims, i.e. what breaks if a case is
deleted:` header enumerating what each case defends. Six of thirteen do.

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

1. **Ten of thirteen have no record of having been shown red.** Only
   `check_short_edge_override.py`, `check_distill_settings.py` and
   `check_schema_defaults.py` document their own calibration. For the rest, the repo's central trust standard is
   unverifiable from the artifacts. This does not mean they are wrong -- it
   means nobody can tell.

2. **Seven of thirteen have no claims block**, so "what breaks if this case is
   deleted" is not recoverable without reading the assertions and inferring
   backwards. The six that have one are the model to copy.

3. **No runner.** Every script prints its own ad-hoc `ok` / `FAIL`, with no
   shared harness and no case registry. There is no way to run everything and
   get one report, and no reliable way to count cases.

4. **The `PYTHONPATH` split is invisible until it fails.** Three scripts
   require it and give a bare import error; six do not. Either all of them
   should bootstrap `sys.path`, or none should.

5. **`check_workflow_schema.py` and `smoke_h3.py` cannot run unattended.**
   Both need a live ComfyUI, and `smoke_h3.py` needs the models loaded too, so
   both are absent from any headless pass -- and a check that is silently
   skipped reads the same as a check that passed.
   `check_distill_settings.py` is the only one that answers this properly, by
   exiting 2 rather than 0 when one of its controls did not run. That pattern
   is worth copying.
