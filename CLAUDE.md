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

## Two traps that have each bitten more than once

**`import nodes` resolves to ours.** This repo has a `nodes.py`, and
`workflows/build_workflows.py` inserts the repo root at `sys.path[0]`, so a later bare
`import nodes` inside `comfy_extras` finds ours and dies on a relative import.
Any script importing both must get ComfyUI's root ahead of the repo, or import
`nodes` first so `sys.modules` is already populated. Cost three separate
debugging rounds.

**DynamicCombo members are dotted in the API form.** `shape.wide_resolution`,
not `wide_resolution`. ComfyUI's executor rejects the flat spelling with
`required_input_missing`. The UI form is positional and unaffected.

A check here is not trusted until it has been shown to go red for the right
reason -- break the thing it guards, watch it fail, put it back. Three
checks written on 2026-08-10 passed for the wrong reason on first writing
and only the mutation caught it; one of them reported zero failures with the
bug reintroduced. Prefer a control the check compares against (a
frontend-written graph, the pre-fix code, an independent implementation)
over asserting against numbers computed in the test itself.

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
