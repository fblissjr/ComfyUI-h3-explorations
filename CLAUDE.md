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

## Running things

No test suite. Verify changes against a live ComfyUI and GPU:
`bench/smoke_h3.py` for a fast sanity pass, the relevant `bench/*.py` script
for the specific claim you changed. Most `bench/check_*.py` scripts need
neither CUDA nor a model and run in a second. This repo runs inside
ComfyUI's own venv, not a standalone uv project. There is no uv.lock here on
purpose.

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
