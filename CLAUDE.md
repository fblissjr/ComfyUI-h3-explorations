# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. Start with README.md for what
ships and why. This file holds only what README.md would not.

## The one rule that matters: saved graphs address everything by position

Never rename a node's `node_id=` (in any `io.Schema`). It is baked into
every saved workflow's `type` field: this repo's `workflows/*.json` and the
owner's live workflows outside this repo. Renaming one breaks every saved
graph that uses it, silently, with no clear error in the UI. Class names,
menu `category=`, log prefixes, and package metadata are all safe to rename.

The same applies to **order**, and this cost a real bug on 2026-08-10. A
saved graph stores `widgets_values` as a bare list and wires links to
integer output slots, so both are matched by index at load time. Adding an
input or output anywhere except the **end** silently re-points every later
one in every existing graph. `head_chunks` went in after `mode`, where it
reads better, and landed an older graph's `patch_token_refiner=False` on an
INT with `min=1`. Append new inputs and outputs, and leave a comment saying
why the ordering looks wrong -- semantic grouping is exactly the instinct
that breaks this.

`bench/check_workflow_schema.py` catches it: it type-checks widget values
against their declared types positionally. Run it after any schema change,
and regenerate the workflows only against a ComfyUI that has already
reloaded the change -- the generator validates against a live
`/object_info`, so regenerating against a stale server bakes in the
mismatch it is supposed to catch.

## What is where

| | |
|---|---|
| `docs/checks.md` | **the index of every check**: what it defends, what it needs, whether it has been shown red. Read before adding one. |
| `docs/h3_references.md` | every reference type, its processing, measured cost, the label rules, and a worked prompt per relationship |
| `docs/h3_ref2v_distillation.md` | why ref2v resists step distillation |
| `docs/h3_resolutions.md` | all 95 legal canvases and what each costs |
| `docs/open_experiments.md` | what is deliberately **not** measured, and the blocker for each |
| `internal/postmortems/` | gitignored. **Start with the newest and work back** rather than re-deriving what is open; each one's forward items are phrased so they can be marked done or refuted. Newest is `2026-08-15_session_morton-sampler-and-refs.md` (eight items, and a repeat escape the 2026-08-14 one also names). |
| `docs/morton.md` | what Morton reordering does to Sol-Attn's blocks, measured on the latent grid **and on captured activations**. Read its assumption chain before quoting any of it: five of six links are now verified, and link 6 (does any of it reach the output) is not. |
| `docs/sol_engine_reference.md` | NVLabs' own Sol-Engine recipe for H3, read from `coderef/Sana` at `origin/sol-engine`. Their validated policy, how ours differs, FirstBlockCache, and why their published speedups are not comparable to ours. |
| `internal/reference_library.md` | the 19 reference images in the input `h3_refs/` folder, their DiT cost at both size modes, and which carry logos, text or IP. Read before wiring a reference arm. |
| `workflows/h3_config.py` | every shared constant. Nothing here may have a second copy anywhere. |
| `workflows/build_workflows.py` | generates all graphs. Never hand-edit a `workflows/*.json`. |
| `bench/check_*.py` | fast, mostly CUDA-free guards |
| `bench/bench_*.py`, `bench/smoke_h3.py` | need a GPU and a live server |

## Running things

No test suite. Verify changes against a live ComfyUI and GPU:
`bench/smoke_h3.py` for a fast sanity pass, the relevant `bench/*.py` script
for the specific claim you changed. Most `bench/check_*.py` scripts need
neither CUDA nor a model and run in a second. This repo runs inside
ComfyUI's own venv, not a standalone uv project. There is no uv.lock here on
purpose.

**Start ComfyUI with `~/ComfyUI/start.sh`** so its log is readable while you
work. Several findings here were only visible in that log.

**Only `bench/smoke_h3.py` submits a prompt.** Every other check reasons about
graphs. On 2026-08-13 that gap hid a bug making *every* API graph
unsubmittable for as long as the Resolution node had been wired in, because
the validator asserting correctness was itself asserting the wrong shape. A
static check cannot catch a bug whose cause is the static check. **Run the
smoke after any generator change**, and treat a green validator on an
unsubmitted graph as unverified.

Free the GPU between a render and the CUDA checks (`POST /free` with
`unload_models`), or they OOM and look like regressions.

### Changing a graph

1. Edit `workflows/build_workflows.py` or `workflows/h3_config.py`.
2. If a node schema changed, **restart ComfyUI** so it reloads the pack.
3. Confirm the reload actually happened -- read the changed default back out
   of `/object_info` before regenerating. A stale server bakes in the exact
   mismatch the validation exists to catch.
4. `python workflows/build_workflows.py`
5. `python bench/check_workflow_schema.py workflows/*.json` and the smoke.

### Prompts

Graphs carry their prompt baked in, which is what makes them editable and
what lets a hand-edit diverge from the generator. To load the right prompt
into the right arm:

```bash
python workflows/build_workflows.py --list-prompts
python workflows/build_workflows.py --print-prompt ref_video_edit
```

Every reference prompt comes from one function, `_ref_prompt()`, and
`bench/check_ref_prompt_labels.py` fails the build if a shipped graph carries
a prompt that function cannot produce. It also checks the prompt names exactly
the labels the graph wires -- the tokenizer derives `<Picture i>` /
`<Video k>` / `<Audio j>` from the **sockets**, not the prompt, so the two
drift silently.

## Two settings you will be tempted to "fix". Don't, without measuring.

**Sage runs `mode="fp16 (most accurate)"`, not `auto`.** `auto` resolves to
`fp8_cuda++` -- the *fastest* kernel -- which looks like the sensible default
and is the wrong end of this project's tradeoff. The owner judged it on video
at the same seed: "way clearer and better motion and less drift". **That
perceptual verdict is the load-bearing half and it is the half that has held.**

**There is no numeric half any more. Withdrawn 2026-08-16 by the owner, as
untrusted.** This paragraph used to carry an fp8-vs-fp16 accuracy ratio and the
`mean_rtol` figures behind it, first from a synthetic `torch.randn` sweep and
then a smaller ratio reported secondhand from the sage fork's captured
activations. The numbers are gone from this repo -- code, configs, docs and
node text -- because their provenance could not be defended: the synthetic
sweep measures an input distribution H3 does not have, the real-activation
figure was never re-derived here and its script is not committed in the fork,
and nothing in `bench/` uses captured activations at all. See
`docs/evidence.md`.

**The decision does not change, and it never rested on those numbers.** It
rests on the perceptual verdict above, which is independent of every retracted
figure. **Do not reintroduce a ratio to defend it.** If an accuracy claim is
wanted, measure it here on the captures that now exist and cite that run.

**`h3_capture.py` ran for the first time on 2026-08-15, and the data now
exists**: three files at blocks 0/24/49, step 1, `[1, 56, 37826, 128]` bf16,
from a dense 124-frame 1344x768 render with Sol-Attn bypassed, in
`~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/` with a README. **The
fp8-vs-fp16 question is still unanswered** -- those captures were made and used
for a different question (Morton's effect on block centroids, `docs/morton.md`)
and nothing has yet graded a sage kernel against them. So the caveat above
stands unchanged; what has gone away is the excuse. Anyone re-running a render
to make this data is duplicating work that is already on disk.

fp16 also holds q/k/v for the whole call (no `sageattn_consume` path), costing
roughly 1.58x **per attention call**. That is a kernel number, not a render
number: it was written here as "wall clock", and if attention were nearly all
of H3's compute a render should be ~1.5x slower, which nothing observed shows.
The heaviest shipped config still peaks at 21,186 MiB of 24,564. All three fp8
variants land within 0.0004 of each other, so the PV accumulator is not the
lever -- quantizing V to fp8 at all is.

**Sol-Attn is opt-in and shipped OFF.** Every graph carries the node bypassed
in the UI form and omits it from the API form. That is policy, not an
oversight: sage must always be on and compose with anything downstream, while
Sol changes *what the model computes* and nobody has weighed that against
what its speed buys. Two probe graphs enable it and exist so the question
stays answerable: `h3_probe_sol_on.json` (t2v) asks whether Sol earns its
influence on the output, and `h3_probe_sol_on_refs.json` puts references in
front of it, which is the only way to verify the conditioning sink — on t2v
the sink is a few hundred rows and the signal is too thin to trust. Re-enable
another with `sol_on=True` in its `GRAPHS` entry, not by hand-editing a saved
graph.

**Reference-heavy is where Sol has the LEAST room, not the most**, which is
the opposite of what this repo assumed for weeks. Reference rows are pinned
exact, so they raise the token count without adding anything Sol can
sparsify. Treat the refs probe as a mechanism check, not a speed one.

## Four traps that have each bitten more than once

**`import nodes` resolves to ours.** This repo has a `nodes.py`, and
`workflows/build_workflows.py` inserts the repo root at `sys.path[0]`, so a later bare
`import nodes` inside `comfy_extras` finds ours and dies on a relative import.
Any script importing both must get ComfyUI's root ahead of the repo, or import
`nodes` first so `sys.modules` is already populated. Cost three separate
debugging rounds.

**A running ComfyUI holds TWO copies of every `comfy_extras` module, and
patching the wrong one looks exactly like success.** `load_custom_node`
registers each file under a **file-path** module name (`nodes.py`:
`sys_module_name = os.path.splitext(module_path)[0]`), and `comfy_extras/` has
no `__init__.py`, so it is a PEP 420 namespace package. A dotted
`import comfy_extras.nodes_minimax_h3` -- which `resolution.py`, `preflight.py`
and `build_workflows.py` all do -- therefore builds a **second, independent
module object** with its own copy of every module-level function. On
2026-08-15 `single_frame.py` patched only the dotted copy: it logged success,
an in-process check agreed, and the server went on serving and executing the
unpatched one. **Resolve by identity, not by name** -- start from
`nodes.NODE_CLASS_MAPPINGS`, then collect every `sys.modules` entry whose
`__file__` matches. And verify against the live `/object_info`, which is the
only surface that can tell a patched module from a patched copy of one.

**DynamicCombo members are dotted in the API form.** `shape.wide_resolution`,
not `wide_resolution`. ComfyUI's executor rejects the flat spelling with
`required_input_missing`. The UI form is positional and unaffected.

**A ComfyUI `git pull` can break every graph here, and nothing local will say
so.** On 2026-08-13 upstream dropped `frame_count` from `PackedLayout`;
`preflight.py` still passed it, so **every graph in the repo** failed at the
Preflight node -- no render at all, not a degraded number. **No check in `bench/` covers a
call INTO a dependency we do not control**, and no amount of local validation
can: the only instrument that sees a broken contract is the contract being
used. So **run `bench/smoke_h3.py` after any ComfyUI or node-pack update**,
not only after changing the generator. It surfaced that day only because an
unrelated render happened to be queued.

A check here is not trusted until it has been shown to go red for the right
reason -- break the thing it guards, watch it fail, put it back. Three
checks written on 2026-08-10 passed for the wrong reason on first writing
and only the mutation caught it; one of them reported zero failures with the
bug reintroduced. Prefer a control the check compares against (a
frontend-written graph, the pre-fix code, an independent implementation)
over asserting against numbers computed in the test itself.

**A baseline that shares mutable state with the thing it measures is not a
baseline.** `check_single_frame.py` compared against core's *original*
`temporal_shape` function object -- but its body resolves the two grid helpers
through module globals, which are exactly what the patch replaces, so both
sides of the comparison ran the patched helpers. It could only see changes made
inside `temporal_shape` itself, and reported green with a helper deliberately
broken at n=200. Rebuild the baseline from source into its own namespace
(`ast` works and stays CUDA-free); holding a reference is not enough.

**A check whose input already satisfies the expected outcome cannot fail, and
it is most convincing when it is emptiest.** The first single-image scene asked
to age a subject to 60 against a reference of a man well past 70. It rendered,
it looked like a working edit, and it demonstrated only that the pipeline runs.
Before trusting a passing case, ask what the input would have to look like for
it to fail.

**The positive case, because these get recorded less often than failures.** On
2026-08-15 `bench_e2e_h3.py` refused an entire run because an override dict
carried a `_curve` key the Sol node does not declare. It validates every
override key against the node's real inputs precisely so a dropped knob cannot
silently turn one arm into a different configuration -- without it, a Hilbert
ordering arm would have rendered as plain `2d_frame` and been reported as a
Hilbert result. Same failure class as the two above, opposite outcome, and the
only difference was that someone had written the guard.

**And the same standard applies to claims, not only to checks -- with the
uncomfortable finding that re-reading your own work does not meet it.** On
2026-08-13, eight substantive defects were found across this repo and the sage
fork. **Not one was caught by whoever wrote it.** Every one came from a second
reader with independent access to the artifact. That is not a statement about
carelessness; the defects were things like a probe size that stayed correct
until its mechanism was replaced, and a caveat that fell off a claim between
one message and the next. Nothing about either looks wrong when you re-read
the sentence you just wrote.

So: **a claim derived from a call site, a docstring, or a plausible mechanism
is an inference, and inferences at that distance fail often enough to justify
verifying before asserting rather than after.** Say which kind of evidence you
have *inside* the claim -- "reported, not verified: a source read, not a build"
survives being quoted; a trailing "(unverified)" reads as hedging and gets
trimmed. When it matters, get the artifact in front of something that can
disagree: a control that can go red, an independent implementation, or another
reader. `docs/checks.md` has the two mechanical forms this takes -- caveat
decay outward, and a caveat you accepted about someone else's number failing to
attach when you make it your own input.

## Reference implementations

`coderef/` (gitignored) holds symlinks to diffusers, DiffSynth-Studio and
comfy-kitchen. diffusers is the one to treat as authoritative: it is where
`h3_rules.py`'s limits come from, and several node docstrings cite it by
file and line. When a claim about "the reference" matters, read it there
rather than trusting a summary -- the ordering detail that makes the
duration rule work (the ceiling applies *after* the frame-count snap) was
absent from every secondhand description of it.

## Research notes

`internal/` (gitignored) holds prompt-writing research, session logs,
upstream-change surveys, and `internal/postmortems/`. Not shipped, not for
redistribution.
