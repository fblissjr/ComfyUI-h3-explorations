# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. `README.md` is what ships and why.
`VISION.md` is what this repo holds itself to and rarely changes. This file is
the operative form: one line per rule, and a pointer to what owns the rest.

## A tinkering repo

This repo is the owner's own experiments, shared with the community, and it is
not research-grade. Rigour is proportional: the heavier rules below (blind
seed distributions, captures, the baseline comparison) apply to a finding we
publish, not to a practical choice. **When sglang and ComfyUI's own node or
comfy-kitchen agree on a default and ours differs, adopt theirs** in
`workflows/h3_config.py` and the generator, rebuild, and say in the CHANGELOG
what it replaced. Our own evals do not gate that; a later eval showing ours was
better reverses it. One upstream alone, upstreams that disagree, or a knob no
upstream has is an ordinary judgement call. (Owner, 2026-09-11.) **A
distill runs on its trainer's contract**, not a downstream template:
`docs/wiki/references.md`, "A distill's reference is its trainer's contract".

## Do not trust prose, including this file's

When prose and code disagree, the code is right. Every sentence here that
names a default, a shipped artifact, or what "every graph" does is a cache
with no invalidation. Cite the observable: `h3_config.MODELS["clip"]` over
"the shipped encoder is"; a node's `define_schema` over "the default is"; a
walk of `graph_paths` over "every graph". When you find prose that lost,
correct it and log what it used to claim in
`docs/wiki/decisions.md`. This file and `VISION.md` carry no history notes;
another doc may also keep a dated note in place where a reader would
otherwise trust the stale text.

## Settled about H3

Before establishing anything about the text encoder, the marker tokens,
prompt structure, PDD quality or reference sizing, read `docs/evidence.md`
"Settled about H3": each was measured or decided and has been re-derived as
news at least once. Three facts every session acts on:

- **Sol-Attn is on by default in every shipped video workflow**
  (`MiniMaxH3SolAttn`, `sol_attn_h3.py`). The exempt set is
  `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`; read the constant,
  never a sentence about it.
- **The baseline is `workflows/bench/h3_text_to_video_dense_stamped_api.json`**:
  stock attention, no sage, Sol or LoRA, the base step count. `VISION.md`
  defines it in words. The older `_stamped` graphs wire sage and are a
  different comparison.
- **1344x768 is a trained canvas.** Probes run at 1152x768 or 1344x768 with
  345 frames, which is how the owner renders; a cheaper canvas only proves a
  harness runs. `docs/h3_resolutions.md`.

## Operative rules

Rules with no other home. The tenet behind each is in `VISION.md`.

- **Closed lanes stay closed** unless the owner reopens them. `docs/roadmap.md`
  "Closed lanes" lists them.
- **Every walker goes through `workflows/h3_config.py::graph_paths`** and the
  set it walks is `GRAPH_DIRS`. Enforced by `bench/check_graph_discovery.py`.
- **Never hand-edit `workflows/*.json`.** `workflows/build_workflows.py`
  generates them, and nothing is true of a graph until it is rebuilt. Rebuild
  before you claim and before you commit.
- **Nothing in this pack patches ComfyUI core**, and the checkout is stock.
  `bench/check_vsa_core_patch.py` is the observable, because the patch is one
  command away from being back.
- **The server process is the resource, not the GPU.** An idle server can be
  an armed one: read `/proc/<pid>/environ` for `H3_*` keys and ask before
  killing it. `bench/restart_comfy.sh` is disabled; restart by hand with the
  recipe in `docs/comfy_notes.md`, finding the port owner rather than the
  first matching pid.
- **ComfyUI caches.** A wall time or VRAM figure is a statement about cache
  state; say which state you measured in. `/history` is the observable for
  which arm ran; a log line can belong to someone else's run.
- **A rendered clip cannot A/B a numerical change.** The trajectory diverges
  at frame zero under any sampler. Grade on captured activations
  (`bench/grade_sage_on_capture.py`).
- **Two models live here and the words do not disambiguate the stage.**
  "Attention" and "capture" each name something at the DiT and at the encoder.
  Check what input a node takes or what a module name starts with before
  carrying a claim across.
- **Write the provenance of a constant beside it**: measured, inherited, or
  reasoned. Three words that stop the next reader re-deriving it.
- **Prose carries pointers; records carry numbers.** `docs/prose_measurements.md`
  is the rule and the migration plan; `bench/list_prose_measurements.py` is
  the worklist.
- **Checks**: `docs/checks.md` is the standard and the index; add one only
  when something escapes the existing ones.
- **This checkout is shared with other agents, one of them not Claude.**
  `git status` before editing, and commit promptly with
  `git commit -F <msg> -- <paths>`. Never `git add`, `commit -a`, a bare
  `git commit` or `--amend`: each has swept a peer's work into another
  commit. The one exception is `git add -- <file>` for a new file you
  created. A file holding a peer's uncommitted hunk is theirs to commit.
  Every subagent prompt says the tree is shared and that the agent must not
  run git.
- **Prompt timing, shot counts and dialogue length fit each scene, never a
  template.** `docs/prompting.md` section 5.10.

## Reference implementations

`coderef/` (gitignored) holds the sister checkouts, and
[`docs/wiki/references.md`](docs/wiki/references.md) says what each is for.
**Do not import Python from it.** The Sol-Attn kernel is built by
`vendor/rebuild_kernel.sh`, whose `--check` says whether the source is
current, and `bench/check_sol_kernel.py` reports the build that is installed.
Read those, never a sentence naming a build.

## Where things are

[`docs/wiki/index.md`](docs/wiki/index.md) is the router: what to read
before you start, which document answers which question, and the rule for
each code directory. [`docs/wiki/next_steps.md`](docs/wiki/next_steps.md) is
what to work on next, and [`docs/wiki/decisions.md`](docs/wiki/decisions.md)
is what was decided or reversed, and when. Read [`VISION.md`](VISION.md)
first if you are new. Three directory rules every agent needs:
`workflows/h3_config.py` holds every shared constant, one copy each;
`vendor_config/` holds the release's own config files verbatim, read through
`vendor_config.py` and never retyped; `internal/` is gitignored and never
shipped.
