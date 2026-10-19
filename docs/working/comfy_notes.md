# Running ComfyUI, and building nodes and workflows for it

Read before the first command. The rules that cost the most when missed:

| rule | why | section |
|---|---|---|
| Run every script with `<comfy-venv-python>`; never a `uv` project command | `uv run` / `uv sync` has emptied the ComfyUI venv twice | "Running things" |
| Restart by the process that owns port 8188, after reading its `H3_*` environment | the first `pgrep` match is a wrapper; an armed server is somebody's capture | "The recipe, by hand" |
| A restart counts only if the server started after the file you changed | otherwise you measure the old code | "The recipe, by hand" |
| Stop a server you started | an idle server holds the card and looks like a live run | "If you started the server, stop it when you are done" |
| Run `bench/smoke_h3.py` after any generator, ComfyUI or node-pack change | no static check submits a graph | "Changing a graph" |
| Read `/history` and the server log before trusting a time | ComfyUI caches node results | "What is cached, and by whom" |
| Never rename a `node_id=`; only append inputs and outputs | the owner's saved graphs bind by name and position | "Saved graphs address everything by position" |

## Running things

There is no test suite. Verify against a live ComfyUI and GPU: `bench/smoke_h3.py`
for a fast pass, the relevant `bench/*.py` for the claim you changed. Most
`check_*.py` need neither CUDA nor a model ([`checks.md`](../method/checks.md),
"Running them"). This repo runs inside ComfyUI's venv, not a standalone uv
project; there is no `uv.lock` here on purpose.

### Where this machine keeps things: three environment keys

A folder that is this machine's is an environment key, never a path in a tracked
file. Each tool that needs one says so when the key is not set.

| key | what it is | read by |
|---|---|---|
| `H3_OUTPUT_DIR` | the folder the server writes to (it is started with `--output-directory`; nothing in the HTTP API reports it back) | `workflows/h3_config.py::output_dir` |
| `H3_INPUT_DIR` | the folder the server reads stills and input videos from | `bench/masked_job.py::input_dir` |
| `H3_MODEL_STORE` | the folder that holds the pose model's own folder (`h3_config.POSE_MODEL`); `--model-store` on the tool is over it | `bench/load_chain.py land`, `bench/shot_to_load.py reference` |

A fourth, `H3_CAPTURE_ROOT`, is exported by `start.sh` ("Captures are transient"). The bench
tools that resolve the output folder through `bench/_paths.py::comfy_output` read their own
`H3_COMFY_OUTPUT` first, else the live server's `--output-directory`.

### The venv and `uv`

- **No `uv` project command from this repo while `VIRTUAL_ENV` names the ComfyUI
  venv.** Sessions inherit that variable, and `uv run` or `uv sync` treats this
  repo's `pyproject.toml` as a project and can recreate the active venv empty.
  It happened on 2026-09-11 from this checkout (the recreated venv's `pyvenv.cfg`
  carried this project's name as its prompt).
- **`start.sh` must never reach `uv run`.** On 2026-09-10 a restart whose
  `start.sh` fell back to `uv run` deleted `.venv` and recreated it empty (the
  `uv` that ComfyUI-Manager installs inside the venv is first on PATH while it is
  active); torch, the kitchen build and the editable sage and torchaudio installs
  all had to be rebuilt. The owner's `start.sh` now falls back to
  `$PWD/.venv/bin/python`; `COMFY_PYTHON` still overrides it. If a `start.sh` in
  front of you still says `uv run`, set `COMFY_PYTHON=.venv/bin/python` first.
- Before starting the server or a kitchen build, confirm the venv's
  `site-packages` is not empty.
- **Do not restore the June `known_good_venv.txt` / `restore_venv.sh`
  environment**: the owner ruled it too old on 2026-09-10. Rebuild instead.

### The recipe, by hand

**Start ComfyUI with `<comfy>/start.sh`**, so its log is readable; several
findings here were only visible in that log. Check its settings match what you
are doing; ask the owner before changing them. `bench/restart_comfy.sh` is
disabled since 2026-09-02 (it hung on every run; it exits 2 and points here).

    PID=$(ss -ltnp | grep ':8188' | grep -oP 'pid=\K[0-9]+' | head -1)
    tr '\0' '\n' < /proc/$PID/environ | grep '^H3_'   # armed? ask first
    kill $PID; sleep 5; ss -ltnp | grep ':8188' || echo "port free"
    ps -ef | grep -E 'start\.sh|uv run --active'      # kill survivors too
    (cd <comfy> && setsid nohup ./start.sh > <a log you can read> 2>&1 < /dev/null &)
    curl -s localhost:8188/system_stats                # until it answers
    ps -o lstart= -p $(ss -ltnp | grep ':8188' | grep -oP 'pid=\K[0-9]+' | head -1)

- **Go by the port owner, not `pgrep | head -1`.** `start.sh` launches through a
  wrapper, so killing the first match leaves the server serving and you read a
  stale `/object_info`. `pgrep -f` also matches the shell command holding the
  pattern; `ss -lptn 'sport = :8188'` asks who holds the port.
- **An `H3_*` key in the port owner's environment means the server is armed** for
  somebody's capture; restarting silently disarms them (root `AGENTS.md`, "The
  server process is the resource"). Ask first.
- **Between the kill and the launch, confirm the launcher chain is gone too.** On
  2026-09-03 a new listener appeared within seconds whose parent chain led back to
  the previous `start.sh`; the fresh launch lost the port and exited 1. The
  mechanism was not found. Kill `start.sh` and `uv run --active` survivors, and
  read the new owner's environment rather than assuming it is yours.
- **The last line is the point**: the new process's start time must be later than
  the file you changed.
- **`setsid` belongs on every long runner** launched from an agent's shell
  (`bench/run_graph_arms.py` too): a process in the agent's session dies with the
  agent's process group, and a lost render looks like a hang.
- **Send the launcher's output to a file.** The running server's log is also
  `GET /internal/logs/raw` ("Traps"). The server no longer writes
  `user/comfyui_<port>.log` (its last write was 2026-09-16).

### The three arming keys

Read once by the server process from its environment. Each takes `dir=`; put it
under `$H3_CAPTURE_ROOT`. A server carrying any of them renders slower and its
timings are void.

| key | records |
|---|---|
| `H3_CAPTURE` | q/k/v tensors, gigabytes per cell |
| `H3_SOL_OBSERVE` | Sol route counts per call |
| `H3_SOL_PROBE` | Sol against the shipped fallback per call, summaries only |
| `H3_TELEMETRY` (2026-09-26) | what each prompt loaded, moved and spent per node: [`pipeline_telemetry.md`](pipeline_telemetry.md) |

### Captures are transient

`start.sh` exports `H3_CAPTURE_ROOT` (since 2026-09-03), the directory every
`h3_capture.py` capture goes under (`dir=$H3_CAPTURE_ROOT/<date>_<what>`) and the
one `bench/check_capture_manifest.py` and `bench/recycle_captures.py` walk. The
path is written nowhere in this repo on purpose: read it from `start.sh` and
export the same line in the shell that runs the bench tools, or they skip.

The lifecycle, because the disk is not unlimited:

1. Take the capture.
2. Write `manifest.json` with `bench/generate_capture_manifest.py` (it needs the
   graph that rendered) and a `retention.json` saying what it is for and a
   `keep_until` date.
3. Copy the manifest and an inventory record under `bench/results/`
   (`bench/record_capture_inventory.py`).
4. Grade.
5. `bench/recycle_captures.py --delete <name>` removes the tensors and leaves the
   manifest, the retention note and a `DELETED.json` marker. It refuses any
   capture the repo cannot account for.

A capture's value ends when the analysis it was taken for is recorded; a week is
the outer bound anyone has set so far.

### If you started the server, stop it when you are done

An idle ComfyUI still holds a CUDA context and VRAM, and from outside it looks
like somebody's live run. A server that was already running when you arrived is
not yours to stop.

### The local review page has its own small server, started and stopped by hand

It is the page the owner watches renders and answers on (root `AGENTS.md`, "Where
state lives"). It needs no ComfyUI and no GPU. From the repo's root:

    python3 data/<video>/poc/serve.py
    # or, left running after the terminal closes:
    (setsid nohup python3 data/<video>/poc/serve.py > <a log you can read> 2>&1 < /dev/null &)
    # stop it by the port's owner:
    ss -ltnp | grep ':<PORT> '        # pid=N
    kill <N>; sleep 1; ss -ltnp | grep ':<PORT> ' || echo "port free"

- It prints each address it listens on; open one in a browser. It reads only the
  files beside it and appends each click as one line to `answers.jsonl` there.
- Its port is `PORT` in the script (`--port` overrides); `--bind` adds an address
  and is the owner's word to use.
- `python3 data/<video>/poc/check_page.py` says whether the page and its clips load.
- Since 2026-10-15 it also serves the owner's two generated pages:
  `data/<video>/poc/fun_share` links to the share's `Video/<video>/`, so
  `http://<the address it prints>:8765/fun_share/TO_REVIEW.html` and
  `.../fun_share/PROVENANCE.html` open with every clip link working (it serves
  files only; a link to a folder does not open).
- Whoever starts it stops it at the day's end; a session that finds it up asks
  first, since the owner may be reviewing.
- The script lives under untracked `data/` (a throwaway of 2026-10-11 in daily
  use since); a tracked tool with a check is a row in
  [`next_steps.md`](../state/next_steps.md).

### After a ComfyUI update

An update that bumps `comfy-kitchen` replaces our fork wheel with stock, so our Sol
settings (`rotate`, `qk_balance`, `tau_map`, `blk_cnt`) refuse to run; an older
fork wheel lacks modules new core imports (2026-10-18).

1. Sync the owner's `comfy-kitchen` fork (`h3-frontier`) with the new upstream tag.
2. Rebuild with `bash vendor/rebuild_kernel.sh`; never hand-copy files into `.venv`.
3. Before restarting, run `bash vendor/rebuild_kernel.sh --check`,
   `bench/check_sol_kernel.py` and `bench/check_solattn_correctness.py`.

`nanobind` 2.10.x (the venv's 2.10.2) rejects `None` for a `std::optional`
parameter bound with a bare `nb::arg`; 2.11.0 applies `.none()` implicitly
(wjakob/nanobind#1262). The fork marks them `.none()` (`997dd79`), correct under
every version; update the venv's nanobind to the latest at the next planned rebuild.

### Before trusting a run

- **No `check_*.py` submits a prompt.** `smoke_h3.py` is the fast one that does,
  `bench_e2e_h3.py` submits real renders. Run the smoke after any generator change
  and any ComfyUI or node-pack update; a green validator on an unsubmitted graph
  is unverified.
- **Free the GPU between a render and the CUDA checks** (`POST /free` with
  `unload_models`), or they OOM and look like regressions.
- **The host is a variable too.** The GPU board power limit changes render times
  and appears in no workflow, log or manifest: [`hardware.md`](hardware.md);
  `bench/hwinfo.py` prints it.

### Changing a graph

1. Edit `workflows/build_workflows.py` or `workflows/h3_config.py`.
2. If a node schema changed, restart ComfyUI ("The recipe, by hand").
3. Read the changed default back out of `/object_info` before regenerating. A
   stale server bakes in the exact mismatch the validation exists to catch.
4. `<comfy-venv-python> workflows/build_workflows.py` (the repo-local venv lacks
   `packaging`; while a render holds the card, prefix `CUDA_VISIBLE_DEVICES=`,
   [`checks.md`](../method/checks.md), "Running them"). It validates every graph
   by input name against the live `/object_info` and writes nothing if one fails;
   with `--object-info <file>` it validates against that cached schema and says it
   could not confirm the server.
5. Run `bench/smoke_h3.py`.

**Adding a probe or an arm.** Copy an existing probe row in
`workflows/build_workflows.py::main`'s `GRAPHS`. A setting no vendor row attests
runs as a `bench/run_graph_arms.py --set` patch, not as a shipped row; that
script's docstring says when. Before the card, run `bench/preflight_graph.py
<graph>` on any new or hand-built graph and `bench/check_distill_settings.py` on
any LoRA row, and give a new bench tool one throwaway run, read end to end, before
any batch.

## What is cached, and by whom

**ComfyUI caches node results; this pack caches nothing across runs.** Read
`/history` before believing a time.

**ComfyUI's node cache** is always on and lasts for the server's life. A node's
result is served again when its signature is unchanged: its class, what its change
test returned, its input values, and the same for every node upstream
(`comfy_execution/caching.py::CacheKeySetInputSignature.get_node_signature`). With
none of `--cache-classic`, `--cache-lru` or `--cache-none` it is the RAM-pressure
cache (`main.py`, where `cache_type` is chosen), which is how `start.sh` runs it.

- Two graphs in one session that share a loader, a tracker and a Masked Source
  share one track and one mask. That is wanted when arms are compared, and it is
  why the first arm after a restart is the slow one.
- A node whose result depends on something that is not an input is served stale.
  VHS's video loaders hash the path and its modified time, not its bytes
  (`calculate_file_hash` in VideoHelperSuite's `utils` module); a model loader has
  no change test, so a model file replaced under the same name needs a restart.
- An identical graph queued twice runs nothing the second time.
- `bench/record_render_substrate.py` reads `/history` and reports, per prompt, how
  many nodes were served from cache; `/history`'s `execution_cached` names them.
- **Not a fault** (2026-10-10): "VHS failed to extract audio from ..." for a mask
  video, under core's "Could not size cache entry ...; evicting it first". The
  RAM-pressure cache sizes every cached output (`comfy_execution/caching.py`), a
  VHS loader's audio output is lazy (`LazyAudioMap` in VideoHelperSuite's `utils`
  module) and sizing touches it, and a mask video has no sound. It happens once
  per mask loader each time memory is sized, on an output nothing wired, never on
  a render's path. Not worth a loader of our own to silence.

**This pack reuses nothing an earlier run made** (root `AGENTS.md`, "Operative
rules"; [`decisions.md`](../state/decisions.md), 2026-10-10). A result is made
again, or handed in as a named file or a wired input. Four opt-in stores were
removed that day with their inputs, code and checks, because none could show its
key covered everything its result depended on (a stale kept mask had already hidden
a fix on 2026-10-07): `reuse_mask` (`mask_store.py`), `reuse_windows`
(`loop_resume.py`), `window_keep.py`, and `keep_references` (`reference_encode.py`).

What remains is explicit:

- a window's files, which a person names (`keep_windows`, `continue_from`);
- a mask video a job loads from the file its capture wrote;
- the two reference nodes that encode once and wire the result
  (`MiniMaxH3EncodeReferences`, `MiniMaxH3PromptOnReferences`);
- the probe-only step cache, which keeps rows from the first model call of one run
  for that run's later steps.

## Settings not to change without measuring

- **Sage runs `mode="auto"`, which resolves to `fp8_cuda++`**
  (`workflows/h3_config.py::SAGE_NODE`). `fp16 (most accurate)` shipped on
  2026-08-13 on a verdict rendered with Sol absent; `497b421` returned to `auto`
  on 2026-08-18, because with Sol on, sage runs only the steps outside Sol's
  window, so fp16 was paid on every step and delivered on a minority. The six-arm
  comparison is `bench/results/2026-08-18_attention_defaults.json`. Every
  fp8-vs-fp16 accuracy ratio was withdrawn 2026-08-16; do not reintroduce a ratio
  to defend either mode ([`evidence.md`](../method/evidence.md)). Captures to
  measure it properly are under `$H3_CAPTURE_ROOT/`: the 2026-08-17
  reference-heavy pair, not the `2026-08-15_dense_124f_1344x768` one (gone).
- **Sol-Attn Triton is retired** (deleted 2026-08-16, `6872dfd`); recover from
  `github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea` if an old number needs
  re-deriving. `SolAttnBlockProbe` went with it and has no CUDA replacement, so
  `dense_blocks` cannot currently be chosen from measurement.
- **`centroid_tail`'s "~1.4x" is the operation, not end-to-end.** Measured 2.5%
  e2e here, the smallest knob in the node against sol's 1.20x and int8's 1.39x
  ([`evidence.md`](../method/evidence.md)).
- **362 frames is the ceiling** (`h3_rules.MAX_LENGTH`; `MAX_DURATION` derives
  from it). Nothing shorter is a model limit; what 362 rests on is in
  [`evidence.md`](../method/evidence.md).
- **Reference graphs load the ref2va checkpoint directly.** There is no
  alternative path and no switch; the turbo probes carry their own turbo LoRAs.

## Traps that have each bitten more than once

- **Grep the server log before designing a control for something it already
  records.** `GET /internal/logs/raw` returns core's in-memory log buffer for the
  current session (`app/logger.py`), timestamped. It answers "did that render fall
  back to lowvram, novram or a partial load" and what dynamic VRAM staged. The
  buffer is bounded and dies with the process, so copy what a record needs into
  the record. (On 2026-08-28 a stale `comfyui_detail.log` led to a 26-minute
  control render proposed for something the log had all along.)
- **`free_memory` on `/free` unloads every model, even with
  `unload_models: False`.** The server sets the unload flag only when truthy, so
  `flags.get("unload_models", free_memory)` falls through. There is no way to clear
  cached memory without dropping the models. Both flags are consumed between
  prompts, so calling it is safe on a shared queue; it costs the next job a full
  reload.
- **`import nodes` resolves to ours.** This repo has a `nodes.py` and
  `build_workflows.py` inserts the repo root at `sys.path[0]`, so a later bare
  `import nodes` inside `comfy_extras` finds ours and dies on a relative import.
  Put ComfyUI's root ahead of the repo, or import `nodes` first.
- **A running ComfyUI holds two copies of every `comfy_extras` module; patching
  the wrong one looks like success.** `load_custom_node` registers each file under
  its own path-minus-extension (`ComfyUI/nodes.py:2245-2250`, ComfyUI's
  `nodes.py`), and `comfy_extras/` has no `__init__.py`, so a dotted
  `import comfy_extras.nodes_minimax_h3` builds a second module object.
  `keyframe_canvas.py` does this at module scope; `resolution.py`, `preflight.py`
  and `build_workflows.py` inside functions. On 2026-08-15 `single_frame.py` (now
  `archive/single_frame.py`) patched only the dotted copy: it logged success, an
  in-process check agreed, and the server served the unpatched one. **Resolve by
  identity, not by name**: start from `nodes.NODE_CLASS_MAPPINGS`, collect every
  `sys.modules` entry whose `__file__` matches, and verify against a live
  `/object_info`.
- **Custom-node import order is `os.listdir` order, with no sort**
  (`ComfyUI/nodes.py:2356`), so which pack imports first is directory entry order
  and holds today by accident. **Anything that patches another pack does it at
  `execute()` time and counts what it patched** (`sol_curves.install()` returns an
  int and the node raises on zero). Moving it to `__init__.py` would swap a loud
  failure for a silent one; that proposal has been made once.
- **DynamicCombo members are dotted in the API form**: `shape.wide_resolution`,
  not `wide_resolution`; the flat spelling fails with `required_input_missing`. The
  UI form is positional and unaffected.
- **A constant ending in `_LORA` must be a LoRA filename.**
  `bench/check_lora_alpha.py` selects by that suffix, so a bool named
  `REF_VIA_LORA` crashed it with a bare `TypeError`.

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
the graph wires: the tokenizer derives `<Picture i>` / `<Video k>` / `<Audio j>`
from the sockets, so the two drift silently.

## Saved graphs address everything by position

**Never rename a node's `node_id=`, and add inputs and outputs only at the end.**
`node_id` is baked into every saved workflow's `type` field; `widgets_values` is a
bare list and links are integer slots, both matched by index at load. A rename
breaks every saved graph silently; an insertion re-points every later input.
Class names, menu `category=`, log prefixes and package metadata are safe to
rename. Append, and comment why the order looks wrong. (On 2026-08-10
`head_chunks` inserted after `mode` landed an old graph's
`patch_token_refiner=False` on an INT with `min=1`.)

Shipped workflows are API format and carry inputs by name, so the rule protects
the owner's editor-saved graphs. A removal or reorder is therefore the owner's
call: when the owner accepts re-making their saved graphs of a node, its inputs
may be removed and regrouped, `bench/node_id_manifest.json` is regenerated in the
same commit, and the CHANGELOG names the node. First used 2026-09-14
(`MiniMaxH3AudioFreezeSong` lost `prompt_mode` and `window_mode`; the list nodes
lost `seed` and `count`).

**`bench/check_node_ids.py` guards both halves** (since 2026-08-16) against
`bench/node_id_manifest.json`, a committed baseline that is not regenerated from
the schema. Nothing else can: every graph here is generated from the schema, so a
rename regenerates them consistently and the generator and the smoke stay green
while the owner's graphs break. A control whose input is regenerated from the
thing it checks cannot fail. Two partial catches: `bench/bench_e2e_h3.py`
hardcodes 2 of the 8 `node_id` strings (fails at submit time), and a rename shows
as every generated graph changing its `class_type` in `git diff`, which nothing
machine-checks.
