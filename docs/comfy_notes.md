# Notes on building custom nodes and workflows inside ComfyUI

Last updated: 2026-08-16.

## Running things

No test suite. Verify against a live ComfyUI and GPU: `bench/smoke_h3.py` for a
fast pass, the relevant `bench/*.py` for the claim you changed. Most
`check_*.py` need neither CUDA nor a model. This repo runs inside ComfyUI's
venv, not a standalone uv project — there is no `uv.lock` here on purpose.

**Start ComfyUI with `<comfy>/start.sh`** so its log is readable. Several
findings here were only ever visible in that log. Check the settings inside `<comfy>/start.sh` to ensure they align with what you're trying to do. 
If you need to change them, ask the owner first.

**Restart by the process that owns port 8188, not by `pgrep | head -1`.**
`start.sh` launches through a `uv` wrapper, so killing the first match takes
the wrapper and leaves the server serving. You then read a stale
`/object_info` and confirm a reload that never happened.

`pgrep -f` also matches the shell command that contains the pattern, so a
liveness check written that way reports the server up when nothing is
listening. `ss -lptn 'sport = :8188'` answers the question actually being
asked: who holds the port.

**The recipe, by hand.** `bench/restart_comfy.sh` was written to automate this
and is DISABLED since 2026-09-02 because it hung on every run; it exits 2 and
points here. Before killing, read the port owner's environment: an `H3_*`
key there means the server is ARMED for somebody's capture, and restarting
it silently disarms them (CLAUDE.md, "the server process is the resource").

    PID=$(ss -ltnp | grep ':8188' | grep -oP 'pid=\K[0-9]+' | head -1)
    tr '\0' '\n' < /proc/$PID/environ | grep '^H3_'   # armed? ask first
    kill $PID; sleep 5; ss -ltnp | grep ':8188' || echo "port free"
    (cd <comfy> && nohup ./start.sh > /dev/null 2>&1 &)
    curl -s localhost:8188/system_stats                # until it answers
    ps -o lstart= -p $(ss -ltnp | grep ':8188' | grep -oP 'pid=\K[0-9]+' | head -1)

**The three arming keys**, all read once by the server process from its
environment: `H3_CAPTURE` (q/k/v tensors, gigabytes per cell),
`H3_SOL_OBSERVE` (Sol route counts per call), `H3_SOL_PROBE` (Sol against
the shipped fallback per call, summaries only). Each takes `dir=`; put it
under `$H3_CAPTURE_ROOT`. A server carrying any of them renders slower and
its timings are void, which is the reason to read the port owner's
environment before you trust a number from it.

**Between the kill and the launch, confirm the launcher chain is gone too.**
On 2026-09-03 a `kill` of the port owner was followed within seconds by a
new listener whose parent chain led back to the PREVIOUS `start.sh` shell,
and the fresh launch then lost the port and exited 1 -- so the running
server was the old chain's, without the environment the new launch
carried. It did not reproduce on a later kill and the mechanism was not
found; the operative rule is to check `ps` for `start.sh` and `uv run
--active` survivors after the port frees, kill those too, and read the new
owner's environment rather than assuming it is the one you launched.

The last line is the point: the new process's start time must be LATER than
the file you changed, or you are measuring the old code with a fresh-looking
server. The server writes its own log to `user/comfyui_<port>.log` whatever
you do with stdout, so redirecting the launcher to `/dev/null` loses nothing.

**Captures are transient, and the collection root is `start.sh`'s.** Since
2026-09-03 `start.sh` exports `H3_CAPTURE_ROOT`, the directory every
`h3_capture.py` capture goes under (pass `dir=$H3_CAPTURE_ROOT/<date>_<what>`
when arming) and the one `bench/check_capture_manifest.py` and
`bench/recycle_captures.py` walk. The path is not written anywhere in this
repo on purpose; read it from `start.sh`, and export the same line in the
shell that runs the bench tools, or they skip. The lifecycle, because the
disk is not unlimited: take the capture; write `manifest.json` with
`bench/generate_capture_manifest.py` (it needs the graph that rendered) and a
`retention.json` beside it saying what the capture is for and a `keep_until`
date; copy the manifest and an inventory record under `bench/results/`
(`bench/record_capture_inventory.py`); grade; then
`bench/recycle_captures.py --delete <name>` removes the tensors and leaves
the manifest, the retention note and a `DELETED.json` marker. The recycler
refuses any capture the repo cannot account for, so a directory that owes
a manifest keeps its tensors until someone writes one or owns the deletion.
A capture's value ends when the analysis it was taken for is recorded; a
week is the outer bound anyone has set so far.

**If you started the server, stop it when you are done.** An idle ComfyUI still
holds a CUDA context and VRAM, and from the outside a leftover server is
indistinguishable from somebody's live run without inspecting the queue — so
leaving one up costs the next person the card and a diagnosis. This applies to a
server *you* started; one that was already running when you arrived is not yours
to stop.

**No `check_*.py` submits a prompt** — they all reason about graphs.
`smoke_h3.py` is the fast one that does, and `bench_e2e_h3.py` submits real
renders. A static check cannot catch a bug whose cause is the static check, so **run the smoke after any generator
change and after any ComfyUI or node-pack update**, and treat a green
validator on an unsubmitted graph as unverified.

Free the GPU between a render and the CUDA checks (`POST /free` with
`unload_models`) or they OOM and look like regressions.

**The host is a variable too.** A GPU board power limit changes render times and
appears in no workflow, log or manifest. `docs/hardware.md` is what about this
machine can move a result; `bench/hwinfo.py` prints its current state and says
so when the power limit is not stock.

### Changing a graph

1. Edit `workflows/build_workflows.py` or `workflows/h3_config.py`.
2. If a node schema changed, **restart ComfyUI** (see the port note above).
3. Read the changed default back out of `/object_info` before regenerating. A
   stale server bakes in the exact mismatch the validation exists to catch.
4. `uv run python workflows/build_workflows.py`
5. `uv run python bench/check_workflow_schema.py`, then the smoke. With no
   paths it walks `graph_paths(include_bench=True)`, so no directory is typed
   here; pass paths only to narrow it. It exits 2, not 0, when no server
   answered -- nothing validated is not nothing wrong.

## Settings not to change without measuring

**Sage runs `mode="fp16 (most accurate)"`, not `auto`.** `auto` resolves to
`fp8_cuda++`, the fastest kernel and the wrong end of this project's tradeoff.
The decision rests on the owner's perceptual verdict on video at one seed, and
on nothing else: every fp8-vs-fp16 accuracy ratio was withdrawn 2026-08-16 as
untrusted. **Do not reintroduce a ratio to defend it** — see `docs/evidence.md`.
Captured activations to measure it properly exist under `$H3_CAPTURE_ROOT/`, but
not the `2026-08-15_dense_124f_1344x768` one this used to name -- that is gone.
The 2026-08-17 reference-heavy pair is what is on disk.

**Sol-Attn Triton is retired**, deleted 2026-08-16 (`6872dfd`). Recover from
`github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea` if an old number needs
re-deriving. `SolAttnMiniMax` (CUDA) is the only implementation now.
`SolAttnBlockProbe` went with it and has no CUDA replacement, so `dense_blocks`
cannot currently be chosen from measurement.

**`centroid_tail`'s "~1.4x" is the operation, not end-to-end.** Measured 2.5%
e2e here, which makes it the *smallest* knob in the node against sol's 1.20x and
int8's 1.39x. Two separate readers have quoted the tooltip as an e2e figure and
built arguments on it. See `docs/evidence.md`.

**362 frames is the ceiling** (`h3_rules.MAX_LENGTH`), and `MAX_DURATION`
derives from it. 345 was never a model limit — it is the largest count
*diffusers* emits, and presenting that as legality cost a withdrawn bench run.
`reference_would_emit()` answers the portability question separately. Know what
362 rests on before quoting it: one upstream statement with no artifact, plus a
third-party config that ships it.

**Reference graphs load the ref2va checkpoint directly.** There is no
alternative path and no switch; the turbo probes carry their own turbo LoRAs
and nothing else in the model path moves.

## Traps that have each bitten more than once

**The server writes its own log, and the launcher piping stdout does not hide
it.** `user/comfyui_<port>.log` holds the current server session and
`user/comfyui_<port>.prev.log` the previous one, whatever `start.sh` does with
stdout. This is where the answer lives for "did that render silently fall back
to lowvram, novram or a partial load", which is otherwise unanswerable after the
fact and turns any timing or quality comparison into an argument. On 2026-08-28
this lane read a stale `comfyui_detail.log`, saw the live process had both fds
on a pipe, and concluded a VRAM-mode switch was *unobservable* — then proposed a
26-minute control render partly to work around the gap. It was observable the
whole time, and a grep across both files answered it in seconds. **Grep the log
before designing a control for something the server already records.**

**`free_memory` on `/free` unloads every model, even with
`unload_models: False`.** The server sets the unload flag only when it is
truthy, so the executor's `flags.get("unload_models", free_memory)` falls
through to `free_memory`. There is no way to clear cached memory without
dropping the models. Both flags are consumed *between* prompts rather than
mid-render, so calling it is safe on a queue shared with other sessions — it
just costs the next job a full reload. Worth saying out loud on a shared box.

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
functions. On 2026-08-15 `single_frame.py` -- parked since 2026-08-27 and now
`archive/single_frame.py` -- patched only the dotted copy: it
logged success, an in-process check agreed, and the server served the unpatched
one. **Resolve by identity, not by name** — start from
`nodes.NODE_CLASS_MAPPINGS`, then collect every `sys.modules` entry whose
`__file__` matches. Verify against a live `/object_info`, the only surface that
can tell a patched module from a patched copy of one.

**The same family, not yet bitten, recorded before it does: custom-node import
order is `os.listdir` order, with no sort** (`ComfyUI/nodes.py:2356`). So
whether `ComfyUI-SolAttn-cuda` imports before `ComfyUI-h3-explorations` is a
property of directory entry order, not of the names — it holds today by
accident. **Anything that patches another pack must do it at `execute()` time
and count what it patched**, which is why `sol_curves.install()` returns an int
and the node raises on zero. Moving that to `__init__.py` to catch an earlier
hook would swap a loud failure for a silent one: patch nothing, log nothing,
render with the unpatched path, look fine. That proposal has been made once.

**DynamicCombo members are dotted in the API form.** `shape.wide_resolution`,
not `wide_resolution`; the executor rejects the flat spelling with
`required_input_missing`. The UI form is positional and unaffected.

**A constant ending in `_LORA` must be a LoRA filename.**
`bench/check_lora_alpha.py` selects by that suffix and resolves each on disk, so
a bool named `REF_VIA_LORA` crashed it with a bare `TypeError`.

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

## Saved graphs address everything by position

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

`bench/check_workflow_schema.py` catches **the ordering rule**, positionally.

**`bench/check_node_ids.py` catches the rename rule**, against
`bench/node_id_manifest.json` — a committed baseline that is *not* regenerated
from the schema, which is the whole point. It covers the positional contract
too: ordered input and output names, so the "append only" half above is guarded
by the same file. Added 2026-08-16; before that nothing guarded either.

**Why nothing else can, and it is worth knowing.** Every graph here is
generated from the schema, so a rename regenerates all 91 tracked graphs
consistently: `check_workflow_schema.py` passes, the generator revalidates
against a live `/object_info` and passes, the smoke renders. Everything is
green and the artifacts that actually break — the owner's live graphs outside
this repo — are invisible to all of it. **A control whose input is regenerated
from the thing it is checking cannot fail**, which is why the guard had to be a
hand-maintained file.

Two partial catches exist and neither rescues the rule: `bench/bench_e2e_h3.py`
hardcodes **2 of the 8** `node_id` strings, so a rename of those two fails at
submit time (GPU, server, runtime — not the fast suite); and a rename shows as
~91 files changing their `type` field in `git diff`, which is human-visible and
machine-checked by nothing.
