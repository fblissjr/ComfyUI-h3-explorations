# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. `README.md` is what ships and why.
`VISION.md` is what this repo holds itself to and rarely changes. This file is
the operative form: one line per rule, and a pointer to what owns the rest.
The dated instances that earned each rule are frozen in
[`docs/rules_history.md`](docs/rules_history.md); read that when you want to
argue with a rule, not before you follow one.

## Do not trust prose, including this file's

When prose and code disagree, the code is right. Every sentence here that
names a default, a shipped artifact, or what "every graph" does is a cache
with no invalidation. Cite the observable: `h3_config.MODELS["clip"]` over
"the shipped encoder is"; a node's `define_schema` over "the default is"; a
walk of `graph_paths` over "every graph". When you find prose that lost,
correct it and say what it used to claim.

## Settled about H3

Each of these was measured or decided and has been re-derived as news at
least once. `docs/evidence.md` owns them; the record is named on each line.
If you are about to establish one, you are repeating work.

- **The released text encoder is byte-identical to stock Qwen3-VL-32B-Instruct.**
  `bench/results/2026-08-25_released_encoder_is_stock.json`.
- **The seven marker rows are untrained in the encoder, and that discriminates
  nothing**: it follows from the line above, and both live hypotheses predict
  it. `bench/results/2026-08-21_h3_token_embeddings.json`. Do not read it as
  "the tokenizer is wrong".
- **The marker ids are fixed by the release by loading its tokenizer, and no
  JSON literal assigns them.** Both halves are true; grepping for the id finds
  nothing. `bench/results/2026-08-27_marker_tokenization_alignment.json`.
- **The DiT was trained on the vendor's prompt structure, per mode.**
  `vendor_guides/` and `docs/prompting.md`. Structure is not a free parameter.
- **1344x768 is a trained canvas.** A number meant to inform a shipped decision
  is measured there; small canvases have inverted a finding.
  `docs/h3_resolutions.md`.
- **PDD quality is governed by the sigma schedule's coarseness and where it
  lands, not by the evaluation count.** Both streams degrade together. To find
  an audio-specific mechanism, vary the transform at fixed partition, never
  the partition. `docs/research/pdd/audio_under_pdd.md`.
- **The encoder this install loads is `h3_config.MODELS["clip"]`**, and the
  v2 AWQ lane was closed. A different question from the first line.
- **Reference sizing has three knobs and one is a prior.** `docs/h3_references.md`
  is the authority. `short_edge` targets the shorter side and only shrinks
  unless `allow_upscale`; `qwen_view` on `MiniMaxH3AppendRefImage` keeps the
  encoder off the video view; `MiniMaxH3ReferenceFit` is deprecated. The
  shipped encoder short edge rests on one render at one seed.
- **Sol-Attn is on by default in every shipped video workflow.** The node is
  `MiniMaxH3SolAttn` in `sol_attn_h3.py`. The exempt set is
  `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`, which also asserts
  each exemption is necessary. Read the constant, never a sentence about it.
- **The baseline is `workflows/bench/h3_text_to_video_dense_stamped_api.json`**:
  no sage node, no Sol node, no LoRA, stock attention, the base step count,
  paired with `h3_text_to_video_stamped_api` on the same scene, seed and
  length. `VISION.md` defines it in words; every quality or speed claim is
  relative to it and says so. The older stamped graphs wire sage and are the
  dense-sage baselines the Sol arms were measured against, which is a
  different comparison.

## Operative rules

Rules with no other home. The tenet behind each is in `VISION.md`.

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
  (`bench/grade_sage_on_capture.py`); a perceptual claim needs many seeds
  judged blind through `docs/eval_comparison.md` section 3.
- **Two models live here and the words do not disambiguate the stage.**
  "Attention" and "capture" each name something at the DiT and at the encoder.
  Check what input a node takes or what a module name starts with before
  carrying a claim across.
- **A numeric input means the quantity it names; a mode gets a named input.**
  `bench/check_literal_widgets.py` enforces it, and its two allowlists carry
  the judgement: a new sentinel cannot be added without naming its kind.
- **Capture broadly first.** Prefer per-step, per-block tensors scored offline
  (`h3_capture.py`, the `grade_*_on_capture.py` family) over one number per
  arm. Where an effect is known, add granularity rather than re-deriving it.
- **Write the provenance of a constant beside it**: measured, inherited, or
  reasoned. Three words that stop the next reader re-deriving it.
- **Prose carries pointers; records carry numbers.** `docs/prose_measurements.md`
  is the rule and the migration plan; `bench/list_prose_measurements.py` is
  the worklist.
- **Checks**: `docs/checks.md` is the standard and the index. No new check
  until an instance escapes the existing ones. Prefer a control the check
  compares against over numbers it computed itself. When something gains an
  off or parked state, every assertion about it inherits a third case.
- **This checkout is shared with other agents.** `git status` before editing,
  stage by path, commit promptly.

## Reference implementations

`coderef/` (gitignored) holds the sister checkouts; `ls -l coderef/` is the
list and more than half of it is symlinks, so use `find -L` and
`grep -r --dereference-recursive` or a search answers about a minority of it.
**Do not import Python from it.** The Sol-Attn kernel is installed from
comfy-kitchen main; every build calls itself the same version, so read the
local segment of the dist-info, which `bench/check_sol_kernel.py` reports, and
never a sentence naming a build. `bench/_sol_attn_reference.py` is the vendored
reference for what can be imported; `vendor/sol_attn_minimax.py` is a
read-only reference node that is not loaded.

## What is where

### Read these before you start

| file | what it answers |
|---|---|
| [`VISION.md`](VISION.md) | the tenets behind every rule here. Read it first if you are new, and again before calling a rule ceremony |
| [`docs/wiki/index.md`](docs/wiki/index.md) | the generated router: where to start, who owns each answer, which documents nothing links to. Regenerate with `bench/build_wiki_index.py`; never hand-edit |
| [`docs/rules_history.md`](docs/rules_history.md) | this file as it stood before the 2026-09-03 cut, frozen: every dated instance behind every rule above |
| [`docs/roadmap.md`](docs/roadmap.md) | what we are trying to find out next and what would count as finding it. The `h3-experiment` skill routes here first |
| [`docs/evidence.md`](docs/evidence.md) | what is measured, what is retracted, what must not be relied on. Start here before stating a number |
| [`docs/checks.md`](docs/checks.md) | every check, the standard it is held to, and the uncontrolled-requirement audit. Start here before changing behaviour or adding a check |
| [`docs/comfyui_vendor_gaps.md`](docs/comfyui_vendor_gaps.md) | every known divergence from the release, with impact and what enforces it. A dated snapshot that defers to the owners it cites |
| [`docs/custom_node_gaps.md`](docs/custom_node_gaps.md) | what our nodes do end to end, and where that differs from sglang, LightX2V, DiffSynth, diffusers and native ComfyUI |
| [`docs/config_drift.md`](docs/config_drift.md) | why `h3_config.py` drifts and what would stop it. Read before adding prose to a config file or proposing a docs-match-code checker |
| [`docs/prose_measurements.md`](docs/prose_measurements.md) | why a number in prose is a copy, where it goes instead, and the migration plan. Read before writing a speedup, a size or a time into any doc |
| [`docs/sustainability.md`](docs/sustainability.md) | the direction argument and its do-not-do list. Read before proposing process |
| [`docs/prompting.md`](docs/prompting.md) | how to write an H3 prompt, every mode, the single source of truth. Section 14 ranks the five sources that claim to govern a prompt. Grade with `bench/grade_prompt_text.py` |
| [`docs/prompt_catalogue.md`](docs/prompt_catalogue.md) + [`docs/prompt_audit.md`](docs/prompt_audit.md) | every prompt this repo renders (generated by `bench/build_prompt_catalogue.py`, run `--check` first) and the hand-written verdict on each |
| [`docs/prompt_bank.md`](docs/prompt_bank.md) + `prompt_bank/` | the graded bank, one house prompt per named part of the structure, and since `57b3200` the one home of every shipped prompt: `workflows/prompts.py` loads them by id, and the bank's `ships` column says which graphs render each. Generated by `bench/build_prompt_bank.py` |

### Reference, when you touch the thing it covers

| file | what it answers |
|---|---|
| [`docs/comfy_notes.md`](docs/comfy_notes.md) | running and restarting ComfyUI, generating workflows, the `node_id` rule and the `import nodes` trap |
| [`docs/hardware.md`](docs/hardware.md) | what bounds this workload and which host settings invalidate a timing. Carries no values; `bench/hwinfo.py` prints those |
| [`docs/open_experiments.md`](docs/open_experiments.md) | what is deliberately not measured, and the blocker for each |
| [`docs/SOLATTN.md`](docs/SOLATTN.md) | the Sol-Attn authority: knobs, sink, measured arms, ordering, its own do-not-rely table. `morton.md`, `h3_input_impacts.md` and `sol_upstream.md` are reached only through it |
| [`docs/h3_references.md`](docs/h3_references.md) | every reference type, its processing, label rules, and the three sizing knobs |
| [`docs/h3_resolutions.md`](docs/h3_resolutions.md) | every legal canvas and what each costs |
| [`docs/h3_geometry_and_nodes.md`](docs/h3_geometry_and_nodes.md) | the frame grid, the token maths, and which node to use |
| [`docs/h3_pdd.md`](docs/h3_pdd.md) | Parallel Decoding Distillation: what the Acc LoRAs are, the converter and node contract, the two silent traps |
| [`docs/research/pdd/pdd_implementations.md`](docs/research/pdd/pdd_implementations.md) | our PDD against the four other implementations; no other engine implements it |
| [`docs/h3_ref2v_distillation.md`](docs/h3_ref2v_distillation.md) | why ref2v resists step distillation |
| [`docs/h3_image_editing.md`](docs/h3_image_editing.md) | the single-frame path, parked 2026-08-27, in the past tense |
| [`docs/eval_comparison.md`](docs/eval_comparison.md) | the A/B process: matched seeds, blinded, scored before unblinding, recorded as a distribution. The `h3-ab-session` skill routes here |
| [`docs/capture_manifest_schema.md`](docs/capture_manifest_schema.md) | the capture manifest schema. The accepted versions are `bench/check_capture_manifest.py::SCHEMA_VERSIONS` |
| [`docs/bench_plan.md`](docs/bench_plan.md) | pre-registered predictions and the runs that scored them |
| [`docs/check_postmortems.md`](docs/check_postmortems.md) | history of individual checks, not operative |
| [`docs/research/conditioning_nodes.md`](docs/research/conditioning_nodes.md) | what was built against the two conditioning defects, and the five contracts in `MiniMaxH3ReferenceToVideo` that live only in comments |
| [`docs/research/official_weights_metadata.md`](docs/research/official_weights_metadata.md) | what the release declares against what ComfyUI assumes. Read before using a marker |
| [`docs/research/h3_partition_distance.md`](docs/research/h3_partition_distance.md) | how far apart fl2va and ref2va are, per component |
| [`docs/research/merge_requantisation.md`](docs/research/merge_requantisation.md) | what happens to a LoRA merged onto an int8 module. Stored weights only |
| [`docs/research/quant_levers.md`](docs/research/quant_levers.md) | what can be changed about H3's quantisation and which levers are closed |
| [`docs/research/h3_dit_implementations.md`](docs/research/h3_dit_implementations.md) | the DiT across every implementation available here; diffusers is the reference of record |
| [`docs/research/comfyui_h3_t2va_trace.md`](docs/research/comfyui_h3_t2va_trace.md) | what ComfyUI's own code does, call by call, for one t2va render |
| [`docs/research/sglang_h3_pipeline.md`](docs/research/sglang_h3_pipeline.md) | sglang's H3 pipeline stage by stage, at source level. Compares nothing |
| [`docs/research/sglang_comparison.md`](docs/research/sglang_comparison.md) | what the vendor serving path does that we do not. Read before proposing an optimization |
| [`docs/research/technique_transfer.md`](docs/research/technique_transfer.md) | what transfers from LLM and ViT serving to H3 and what does not |

### Code and directories

| path | the rule |
|---|---|
| `vendor_config/` | the release's own config files, verbatim; `vendor_config.py` reads them. Anything the release declares is read from here, never retyped. Guarded by `bench/check_vendor_config.py` |
| [`workflows/h3_config.py`](workflows/h3_config.py) | every shared constant; nothing here may have a second copy |
| [`workflows/build_workflows.py`](workflows/build_workflows.py) | generates all graphs |
| `archive/` | parked work kept as history. Walked by no check, imported by nothing, absent from `GRAPH_DIRS` |
| `bench/check_*.py` | fast, mostly CUDA-free guards |
| [`bench/preflight_graph.py`](bench/preflight_graph.py) | run before queueing a reference render: grades the prompt and prices the sequence, statically |
| `bench/bench_*.py`, `bench/smoke_h3.py` | need a GPU and a live server |
| `bench/run_graph_arms.py` -> `bench/blind_batch.py` -> `score.html` -> `bench/score_session.py` | the path for anything judged by a person. A comparison that skipped a step is two samples, not a result |
| `internal/` | gitignored: prompt research, session logs, postmortems (start with the newest). Not shipped |
