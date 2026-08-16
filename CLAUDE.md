# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. `README.md` is what ships and why.
This file is only what would cost you a session to rediscover — the operative
rule, not the story behind it. Stories live in `docs/` and the postmortems.

## The one rule that matters: saved graphs address everything by position

Never rename a node's `node_id=` (in any `io.Schema`). It is baked into every
saved workflow's `type` field — this repo's `workflows/*.json` and the owner's
live graphs outside it. Renaming breaks every saved graph silently, with no
clear error in the UI. Class names, menu `category=`, log prefixes and package
metadata are all safe to rename.

**The same applies to order.** `widgets_values` is a bare list and links are
integer slots, both matched by index at load. Adding an input or output
anywhere but the **end** re-points every later one in every existing graph.
Append, and leave a comment saying why the ordering looks wrong — semantic
grouping is the instinct that breaks this. (Cost a real bug on 2026-08-10:
`head_chunks` inserted after `mode` landed an old graph's
`patch_token_refiner=False` on an INT with `min=1`.)

`bench/check_workflow_schema.py` catches it, positionally.

## What is where

| file | what it answers |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | what we are trying to find out next, and what would count as finding it. **Start here.** |
| [`docs/checks.md`](docs/checks.md) | the index of every check: what it defends, what it needs, whether it has been shown red. Read before adding one. |
| [`docs/evidence.md`](docs/evidence.md) | what is measured, what is retracted, and what must not be relied on. Read before quoting a number. |
| [`docs/open_experiments.md`](docs/open_experiments.md) | what is deliberately not measured, and the blocker for each |
| [`docs/SOLATTN.md`](docs/SOLATTN.md) | **the Sol-Attn entry point and the authority.** Knobs, sink, measured arms, ordering, and its own do-not-rely-on table. It owns every Sol-Attn number measured on this box; the two deep dives below are reached from it and must not be quoted against it |
| [`docs/morton.md`](docs/morton.md) | deep dive, reached from `SOLATTN.md`: what token reordering does to Sol's blocks. Read its assumption chain before quoting it — link 6 (does any of it reach the output) is unverified |
| [`docs/h3_references.md`](docs/h3_references.md) | every reference type, its processing, measured cost, label rules, and the two sizing knobs that are constantly confused |
| [`docs/h3_image_editing.md`](docs/h3_image_editing.md) | the **experimental** single-frame image gen/edit path: why its graphs live in `workflows/image/`, the prompt-format ladder, and the six scenes |
| [`docs/h3_resolutions.md`](docs/h3_resolutions.md) | all 95 legal canvases and what each costs |
| [`docs/h3_geometry_and_nodes.md`](docs/h3_geometry_and_nodes.md) | the frame grid, the token maths, and which node to use |
| [`docs/h3_ref2v_distillation.md`](docs/h3_ref2v_distillation.md) | why ref2v resists step distillation |
| [`docs/sol_upstream.md`](docs/sol_upstream.md) | deep dive, reached from `SOLATTN.md`: what upstream says — the paper, Sol-Engine's per-profile H3 recipes, the other ComfyUI packs. States their claims, asserts none of ours, and says why their speedups are not comparable |
| [`docs/bench_plan.md`](docs/bench_plan.md) | pre-registered predictions and the runs that scored them |
| [`workflows/h3_config.py`](workflows/h3_config.py) | every shared constant. Nothing here may have a second copy anywhere. |
| [`workflows/build_workflows.py`](workflows/build_workflows.py) | generates all graphs. Never hand-edit a `workflows/*.json`. |
| `workflows/image/` | the single-frame image graphs. Routing is derived from `single_frame=True`; discovery is `h3_config.GRAPH_DIRS`, and **every check that walks graphs must go through `graph_paths()`** -- a bare non-recursive glob passes green over a subset |
| `bench/check_*.py` | fast, mostly CUDA-free guards |
| [`bench/preflight_graph.py`](bench/preflight_graph.py) | **run this before you queue a reference render.** Grades the prompt against the guide's mechanical rules and prices the packed sequence, statically, on any graph path including hand-built ones. Reports, never refuses |
| `bench/bench_*.py`, `bench/smoke_h3.py` | need a GPU and a live server |
| `internal/` | gitignored: prompt research, session logs, upstream surveys, postmortems. Not shipped. |
| `internal/postmortems/` | **start with the newest and work back** rather than re-deriving what is open |

## Running things

No test suite. Verify against a live ComfyUI and GPU: `bench/smoke_h3.py` for a
fast pass, the relevant `bench/*.py` for the claim you changed. Most
`check_*.py` need neither CUDA nor a model. This repo runs inside ComfyUI's
venv, not a standalone uv project — there is no `uv.lock` here on purpose.

**Start ComfyUI with `~/ComfyUI/start.sh`** so its log is readable. Several
findings here were only ever visible in that log.

**Restart by the process that owns port 8188, not by `pgrep | head -1`.**
`start.sh` launches through a `uv` wrapper, so killing the first match takes
the wrapper and leaves the server serving. You then read a stale
`/object_info` and confirm a reload that never happened.

**No `check_*.py` submits a prompt** — they all reason about graphs.
`smoke_h3.py` is the fast one that does; `bench_e2e_h3.py` and
`bench_image_edit_refs.py` submit real renders. A static check cannot catch a
bug whose cause is the static check, so **run the smoke after any generator
change and after any ComfyUI or node-pack update**, and treat a green
validator on an unsubmitted graph as unverified.

Free the GPU between a render and the CUDA checks (`POST /free` with
`unload_models`) or they OOM and look like regressions.

### Changing a graph

1. Edit `workflows/build_workflows.py` or `workflows/h3_config.py`.
2. If a node schema changed, **restart ComfyUI** (see the port note above).
3. Read the changed default back out of `/object_info` before regenerating. A
   stale server bakes in the exact mismatch the validation exists to catch.
4. `python workflows/build_workflows.py`
5. `python bench/check_workflow_schema.py workflows/*.json workflows/image/*.json`,
   then the smoke. That glob is the one place a directory has to be typed --
   the script takes paths from the CLI, so it cannot read `GRAPH_DIRS`.

### Prompts

Graphs carry their prompt baked in, which is what lets a hand-edit diverge from
the generator.

```bash
python workflows/build_workflows.py --list-prompts
python workflows/build_workflows.py --print-prompt ref_video_edit
```

Every reference prompt comes from one function, `_ref_prompt()`.
`bench/check_ref_prompt_labels.py` fails the build if a graph carries a prompt
that function cannot produce, or if the prompt does not name exactly the labels
the graph wires — the tokenizer derives `<Picture i>` / `<Video k>` /
`<Audio j>` from the **sockets**, so the two drift silently.

## Settings not to change without measuring

**Sage runs `mode="fp16 (most accurate)"`, not `auto`.** `auto` resolves to
`fp8_cuda++`, the fastest kernel and the wrong end of this project's tradeoff.
The decision rests on the owner's perceptual verdict on video at one seed, and
on nothing else: every fp8-vs-fp16 accuracy ratio was withdrawn 2026-08-16 as
untrusted. **Do not reintroduce a ratio to defend it** — see `docs/evidence.md`.
Captured activations to measure it properly already exist at
`~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/`.

**Sol-Attn is opt-in and shipped OFF**, bypassed in every UI graph and absent
from every API graph. Policy, not oversight: sage must always be on, while Sol
changes *what the model computes*. Four probe graphs enable it —
`h3_probe_sol_on{,_refs,_all_refs,_i2v}.json`. Re-enable another with
`sol_on=True` in its `GRAPHS` entry, never by hand-editing a saved graph.

**Sol-Attn Triton is retired**, deleted 2026-08-16 (`6872dfd`). Recover from
`github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea` if an old number needs
re-deriving. `SolAttnMiniMax` (CUDA) is the only implementation now.
`SolAttnBlockProbe` went with it and has no CUDA replacement, so `dense_blocks`
cannot currently be chosen from measurement.

**Reference-heavy is where Sol has the LEAST room**, the opposite of what this
repo assumed for weeks. Reference rows are pinned — exactly for audio, to 0.999
for visual (`comfy/ldm/minimax/model.py:32-33`) — so they raise the token count
without adding anything Sol can sparsify. The refs probe is a mechanism check,
not a speed one.

**362 frames is the ceiling** (`h3_rules.MAX_LENGTH`), and `MAX_DURATION`
derives from it. 345 was never a model limit — it is the largest count
*diffusers* emits, and presenting that as legality cost a withdrawn bench run.
`reference_would_emit()` answers the portability question separately. Know what
362 rests on before quoting it: one upstream statement with no artifact, plus a
third-party config that ships it.

**Reference graphs load fl2va plus the ref LoRA** (`REF_LORA_ENABLED` in
`h3_config.py`). The ref2va checkpoint is still required and must not be
deleted from `MODELS`: the builder has one LoRA slot, so the three turbo probes
keep the checkpoint. Reference conditioning and turbo distillation do not
currently stack. **Do not call fl2va+LoRA and ref2va interchangeable** — that
needs a paired render nobody has done.

## Traps that have each bitten more than once

**`import nodes` resolves to ours.** This repo has a `nodes.py` and
`build_workflows.py` inserts the repo root at `sys.path[0]`, so a later bare
`import nodes` inside `comfy_extras` finds ours and dies on a relative import.
Put ComfyUI's root ahead of the repo, or import `nodes` first.

**A running ComfyUI holds TWO copies of every `comfy_extras` module, and
patching the wrong one looks exactly like success.** `load_custom_node`
registers each file under its own path-minus-extension
(`ComfyUI/nodes.py:2245-2250` -- **ComfyUI's**, not ours, which is the trap the
paragraph above describes and which this citation walked into),
and `comfy_extras/` has no `__init__.py`, so a dotted
`import comfy_extras.nodes_minimax_h3` builds a **second, independent module
object**. `keyframe_canvas.py` and `reference_fit.py` do this at module scope;
`resolution.py`, `preflight.py` and `build_workflows.py` do it inside
functions. On 2026-08-15 `single_frame.py` patched only the dotted copy: it
logged success, an in-process check agreed, and the server served the unpatched
one. **Resolve by identity, not by name** — start from
`nodes.NODE_CLASS_MAPPINGS`, then collect every `sys.modules` entry whose
`__file__` matches. Verify against a live `/object_info`, the only surface that
can tell a patched module from a patched copy of one.

**DynamicCombo members are dotted in the API form.** `shape.wide_resolution`,
not `wide_resolution`; the executor rejects the flat spelling with
`required_input_missing`. The UI form is positional and unaffected.

**A ComfyUI `git pull` can break every graph here and nothing local will say
so.** On 2026-08-13 upstream dropped `frame_count` from `PackedLayout` and
every graph failed at the Preflight node. **No check in `bench/` covers a call
INTO a dependency we do not control**, and none can — the only instrument that
sees a broken contract is the contract being used.

**A constant ending in `_LORA` must be a LoRA filename.**
`bench/check_lora_alpha.py` selects by that suffix and resolves each on disk, so
a bool named `REF_VIA_LORA` crashed it with a bare `TypeError`.

## How this repo decides something is true

`docs/checks.md` is the long form. The rules:

- **A check is not trusted until it has been shown red for the right reason.**
  Break the thing it guards, watch it fail, put it back. An empty "shown red"
  cell in the index is a finding, not a formatting gap.
- **A control that stays green is more often a control that never landed than
  a check that is inert. Diff the mutant against the source before concluding
  anything.** Twice on 2026-08-16: a `.replace()` whose target string was not
  in the file (the source said `partially_copy`, the patch expected
  `reference`), and a grep pattern that matched nothing. Both printed exactly
  what a broken check prints, and nothing in either check's output could tell
  the two apart.
- **A baseline that shares mutable state with the thing it measures is not a
  baseline.** Rebuild it from source into its own namespace; holding a
  reference is not enough.
- **A check whose input already satisfies the expected outcome cannot fail**,
  and it is most convincing when it is emptiest. Ask what the input would have
  to look like for it to fail.
- **Prefer a control the check compares against** — a frontend-written graph,
  the pre-fix code, an independent implementation — over asserting against
  numbers the test computed itself.
- **The same standard applies to claims, and re-reading your own work does not
  meet it.** On 2026-08-13, eight substantive defects were found here and in the
  sage fork; not one was caught by whoever wrote it. Every one came from a
  second reader.
- **A claim derived from a call site, a docstring, or a plausible mechanism is
  an inference.** Say which kind of evidence you have *inside* the claim —
  "reported, not verified: a source read, not a build" survives being quoted,
  where a trailing "(unverified)" reads as hedging and gets trimmed.
- **When you reverse a decision, update the document that argued for it.** That
  is the one you will forget, and it is how three files ended up describing a
  directory that had been deleted.

## Reference implementations

`coderef/` (gitignored) symlinks the sister checkouts — diffusers,
DiffSynth-Studio, comfy-kitchen, sage-fork, triton, LightX2V, Minimax-H3-Turbo
— plus real clones of MiniMax-H3, Sana, MiniMax-Music3 and h3-turbo-eval.

**`coderef/MiniMax-H3` is the official repo and outranks everything for the
spec.** It has no reference `generate.py`, so **diffusers stays authoritative
for implementation** — it is where `h3_rules.py`'s limits come from. Read
either directly rather than trusting a summary: the ordering detail that makes
the duration rule work (the ceiling applies *after* the frame-count snap) was
absent from every secondhand description of it.
