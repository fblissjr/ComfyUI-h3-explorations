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
`workflows/h3_config.py` and the generator, rebuild, and say in the commit
message what it replaced. Our own evals do not gate that; a later eval showing ours was
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
  (`MiniMaxH3Sol`, `sol_attn_h3.py`). The exempt set is
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
  "Closed lanes" lists them. **TaoMate-H3 is deprecated** (owner, 2026-09-27,
  said again 2026-09-29): cite it as history only, never as a live reference, a
  target or a source of evidence for a choice here.
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
- **Nothing an earlier run made is reused by a key.** A store that hands
  back a mask, a window, a latent or an encoding because a key matched is
  never reliable: the key cannot see everything the result depends on. A
  result is made again, or it is handed in explicitly as a named file or
  wired input that the job says it reads. A mask is preprocessing data for
  the job that reads it, temporary, and rebuilt when preprocessing runs
  again. A store that keeps rows inside one run for that run's later steps
  is not this. (Owner, 2026-10-10.)
- **A masked render is run on data, before and after.** A no-sampling
  capture and its flags before it samples; the same capture read first when
  it goes wrong; a "why" cites a row, a frame range or a measured figure;
  every flag gets its outcome. `docs/wiki/masked_v2v.md`, "The rule: data
  before a render, and the same data after"; the tool is
  `bench/capture_masked_run.py`, and captured data lives under the untracked
  `data/`.
- **Preprocessing data is verified the moment it is made, by code and by
  eyes.** Straight after a preview's data is captured, and before anything
  is planned or rendered from it: the tool's `verify`, and a look at what
  was made (the mask on the right subject on every frame, the part, the
  motion video, each file's frame offset) by someone who did not make it,
  recorded with who looked. A passing check says no rule failed, never that
  the data is right. What eyes found that no code checks is written down
  as a gap in the code, and what the code found that eyes missed as a gap
  in the look. (Owner, 2026-10-10.)
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
- **A record under `bench/results/` is its `.md`; data goes under `data/`.**
  The record holds the tables a reader needs. The file of figures it was
  made from lives under the untracked `data/`, and the record points at it.
  The one exception is a small file a tracked check or tool reads, named in
  that check. (Owner, 2026-10-10.)
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
- **The changelog is written once a day, from the commits.** A commit
  carries no changelog entry, no fragment and no version bump: its message
  is the record, so write the message to be read later. At the end of a
  working day one session reads the day's `git log`, summarises it into one
  fragment, and builds `CHANGELOG.md` once (`changelog.d/README.md`). The
  detail of what happened and why belongs in a postmortem.
- **Prompt timing, shot counts and dialogue length fit each scene, never a
  template.** `docs/prompting.md` section 5.10.
- **A scene's text is data, served by whatever plans the shots; it is never
  stored verbatim in code or in a workflow.** A text is written from the
  frames of one window and is true of that window only: kept in a constant
  or a graph it gets used again after the window has changed, and nobody
  remembers it was there. It lives with the job's data beside the frames it
  was written from, and a load takes the text of its own frames. The model
  does not know it is editing: a text is a plain caption of the finished
  shot (what is seen and heard), never a statement of what stays or is
  kept, and it says nothing about sound, action or expression that is not
  known from the frames or the track. A served text is an input like a
  stored mask and is trusted as little: know which text a load is getting,
  read it against that load's frames before it is used, and check after
  the render that it did what was expected. (Owner, 2026-10-10.)
- **When a render is wrong, read its text first.** Before a kernel, a
  setting, the audio or a mask is suspected or a test is queued: the text
  the model was given, read against the frames, for what it says and what
  it does not say about the thing that is wrong. On 2026-10-10 three test
  renders went to a kernel setting and an audio file before the text was
  read; one sentence fixed it. (Owner, 2026-10-10.)
- **A rule that depends on the picture takes its value from the input
  video, up front.** A likeness line, a size line, a cut score, a margin:
  the preview measures this clip and the rule is set from that, as
  `subject_track.auto_cuts` and `auto_match` do. A constant measured on
  another clip is a starting value, marked as one, and the tool says when
  this clip's own figures disagree with it. Sharp modern video and soft old
  film do not share thresholds.

## Reference implementations

`coderef/` (gitignored) holds the sister checkouts, and
[`docs/wiki/references.md`](docs/wiki/references.md) says what each is for.
**Do not import Python from it.** A check may run a coderef file as a
one-off numeric reference to compare against (owner, 2026-10-05: "just
dont write code that relies on it here - just one offs"); nothing shipped
depends on it. The Sol-Attn kernel is built by
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
