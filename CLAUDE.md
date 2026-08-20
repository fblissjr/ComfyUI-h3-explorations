# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. `README.md` is what ships and why.
This file is only what would cost you a session to rediscover — the operative
rule, not the story behind it. Stories live in `docs/` and the postmortems.

## Guiding Principles
- **Before writing a number into prose, substitute a different plausible value. If the reader's next action is unchanged, the number is decorative — delete it.** "Sixteen rows claim calibration" and "seventeen rows claim calibration" prompt the same next step, so the count is liability carrying no information. This replaced a volatility test on 2026-08-17, which kept accurate-but-useless counts and cut useful ones.
  - A number that survives the test is one of two things. **Normative** — a limit you are setting, an exit code, a threshold — which cannot drift, because the world moves toward it. Or **descriptive**, in which case it needs an observation point: a date, a commit, an attribution, or past tense. Descriptive counts belong only in dated records.
  - **Generating a decorative number is not a lesser fix than deleting it.** It makes the claim permanently true and permanently useless, and still charges every reader a reconciliation against what they can see. Delete first; generate only what passed the test.
  - Auditing prose already written rather than prose being written: `claim-audit`.
- DO NOT write meaningless tests - if something can't be tested or be a simple red/green, then find another way to make sure you're measuring and testing what you think you are
- **Sol-Attn is ALWAYS ON by default in every shipped video workflow.** Bypassing is reserved for explicit testing / comparative experiments. Three kinds of graph legitimately do not wire it, and all of them wire `MiniMaxH3SageAttention` instead: `workflows/image/` (single-frame, Sol does not apply), `workflows/bench/*_stamped_api.json` (dense baselines — the thing Sol is measured against), and the two capture graphs `workflows/h3_probe_capture_ref3_api.json` / `workflows/h3_probe_capture_ref3_fl2va_api.json` (activation capture, which must record the true attention inputs rather than Sol's output). A fourth kind, since 2026-08-20, wires neither sage nor Sol: `workflows/h3_probe_turbo_768p_sla_router_api.json` runs `MiniMaxH3SLARouter`, the sparse top-k router the Turbo-SLA LoRA was distilled under, as a comparative arm. **Enforced by `bench/check_attention_defaults.py` since 2026-08-18**: it grades every graph's Sol and sage values against `h3_config`, by reachability from the output node rather than node presence, and asserts each exemption above is *necessary* (an exempt graph with live Sol goes red). Until then this rule was enforced on one graph by accident — `check_bench_matches_shipped.py` reads `h3_probe_sol_on_api.json` for a different purpose — and this paragraph said so until 2026-08-20, two days after the check landed. The **Uncontrolled requirements** table in [`docs/checks.md`](docs/checks.md) records the handover.

## What is where

### Read these before you start

| file | what it answers |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | what we are trying to find out next, and what would count as finding it. **Start here** if the question is what to work on. |
| [`docs/evidence.md`](docs/evidence.md) | what is measured, what is retracted, and what must not be relied on. **Start here** if you are about to state a number. |
| [`docs/checks.md`](docs/checks.md) | the index of every check, the standard it is held to, and the standing uncontrolled-requirement audit. **Start here** if you are about to change behaviour or add a check. |

### Reference, when you touch the thing it covers

| file | what it answers |
|---|---|
| [`docs/comfy_notes.md`](docs/comfy_notes.md) | running and starting comfyui, generating workflows, prompting, useful tips and tricks when working with comfyui custom nodes and workflows. Also holds the `node_id` rule and the `import nodes` trap |
| [`docs/hardware.md`](docs/hardware.md) | the box every number here was measured on: what bounds this workload, what has been ruled out as a bottleneck, and which host settings silently invalidate a timing comparison. **Read before quoting a render time.** Carries no values — `bench/hwinfo.py` prints those |
| [`docs/open_experiments.md`](docs/open_experiments.md) | what is deliberately not measured, and the blocker for each |
| [`docs/SOLATTN.md`](docs/SOLATTN.md) | **the Sol-Attn entry point and the authority.** Knobs, sink, measured arms, ordering, and its own do-not-rely-on table. It owns every Sol-Attn number measured on this box. **Three deep dives are reached only through it and must not be quoted against it:** `morton.md` (token reordering; its link 6 — does any of it reach the output — is unverified), `h3_input_impacts.md` (canvas/frames/Sol interaction, the `latent_t % 4` length effect, and why there is no 100k token budget), `sol_upstream.md` (what upstream claims; asserts nothing of ours, and their speedups are not comparable) |
| [`docs/h3_references.md`](docs/h3_references.md) | every reference type, its processing, measured cost, label rules, and the two sizing knobs that are constantly confused |
| [`docs/h3_image_editing.md`](docs/h3_image_editing.md) | the **experimental** single-frame image gen/edit path: why its graphs live in `workflows/image/`, the prompt-format ladder, and the scenes -- it owns their count |
| [`docs/h3_resolutions.md`](docs/h3_resolutions.md) | every legal canvas and what each costs -- it owns the count, do not restate it here. `h3_input_impacts.md` is its deep dive |
| [`docs/h3_geometry_and_nodes.md`](docs/h3_geometry_and_nodes.md) | the frame grid, the token maths, and which node to use |
| [`docs/h3_ref2v_distillation.md`](docs/h3_ref2v_distillation.md) | why ref2v resists step distillation |
| [`docs/eval_comparison.md`](docs/eval_comparison.md) | paired evaluation and stacking: layout rules (vertical for widescreen, horizontal for portrait), labels, and blind evals |
| [`docs/capture_manifest_schema.md`](docs/capture_manifest_schema.md) | activation capture manifest schema: generation parameters, prompt, reference metadata, token accounting, and tensor checksums. **Do not name a version here** -- the accepted set is `bench/check_capture_manifest.py::SCHEMA_VERSIONS` and what new manifests are stamped with is `bench/generate_capture_manifest.py`. A version written into prose drifted once already, which is why that constant exists |
| [`docs/bench_plan.md`](docs/bench_plan.md) | pre-registered predictions and the runs that scored them |
| [`docs/check_postmortems.md`](docs/check_postmortems.md) | **history, not operative — skip it** unless you are investigating one specific check or about to write a control. Per-defect narrative and frozen run logs; every count in it is stale by design |

### Code and directories: the operative rule

| path | the rule |
|---|---|
| [`workflows/h3_config.py`](workflows/h3_config.py) | every shared constant. Nothing here may have a second copy anywhere. |
| [`workflows/build_workflows.py`](workflows/build_workflows.py) | generates all graphs. Never hand-edit a `workflows/*.json`. |
| `workflows/image/` | the single-frame image graphs. Routing is derived from `single_frame=True`; discovery is `h3_config.GRAPH_DIRS`, and **every check that walks graphs must go through `graph_paths()`** -- a bare non-recursive glob passes green over a subset. Enforced by `bench/check_graph_discovery.py` |
| `bench/check_*.py` | fast, mostly CUDA-free guards |
| [`bench/preflight_graph.py`](bench/preflight_graph.py) | **run this before you queue a reference render.** Grades the prompt against the guide's mechanical rules and prices the packed sequence, statically, on any graph path including hand-built ones. Reports, never refuses |
| `bench/bench_*.py`, `bench/smoke_h3.py` | need a GPU and a live server |
| `internal/` | gitignored: prompt research, session logs, upstream surveys, postmortems. Not shipped. |
| `internal/postmortems/` | **start with the newest and work back** rather than re-deriving what is open |
| `internal/h3_resolution_explainer.html` | gitignored, so no link: the interactive canvas-cost companion to `h3_resolutions.md`. **A teaching surface, not a source** — every number it shows is owned by that doc, and it is the copy that goes stale silently because nothing checks it |

## How this repo decides something is true

`docs/checks.md` is the long form. The rules:

- **A baseline that shares mutable state with the thing it measures is not a
  baseline.** Rebuild it from source into its own namespace; holding a
  reference is not enough.
- **A check whose input already satisfies the expected outcome cannot fail**,
  and it is most convincing when it is emptiest. Ask what the input would have
  to look like for it to fail.
- **A check reporting red while the state is correct trains you to ignore red,
  which is worse than no check.** `bench/check_retraction_consumers.py` and
  `bench/check_doc_links.py` both cite this rule to justify an allowlist over a
  grep-and-judge.
- **When something gains an "off", "parked" or "absent" state, revisit every
  assertion about it.** Each one inherits a third case, and "correctly absent"
  is not "broken". `bench/smoke_h3.py` and `bench/check_provenance_stamp.py`
  are in their current shape because of it.
- **A requirement is not a control. When you write a "must" into a doc, name
  the assertion that goes red if it is ignored — or write "enforced by
  nothing".** `docs/open_experiments.md` #18 required conditioning rows to stay
  in the block population; a violating implementation passed every control the
  script had, and only a deliberate mutation found it. The requirements most
  likely to lack a control are the ones everybody agrees with, because
  agreement feels like coverage. The **Uncontrolled requirements** table in
  `docs/checks.md` is the standing audit.
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
- **No new check until a drift instance appears that the existing gates provably
  could not have caught.** Adopted 2026-08-17 after a day in which six proposed
  instruments dissolved on contact with something already installed: a
  quoted-attribution grammar that `check_doc_links.py` already had as
  `path::symbol`, a requirements-table checker that was refuted outright, a
  postmortem filing scheme the toolchain shipped, and a principles document for
  two principles this file already states. The gates here are good and the
  reflex to add one is usually a failure to read them. Cite the escaped instance
  before building, and if you cannot name one, the answer is to use what exists.
- **Building the replacement is not the change.** Retiring the original and
  repointing everything that cites it is the change. On 2026-08-17 three red
  harnesses were rewritten and the pre-port copies left in place, still runnable
  and still returning success unconditionally, with `docs/checks.md` still citing
  them — so the fix shipped and the defect stayed.
- **A rendered clip cannot A/B a numerical change.** The sampling trajectory
  diverges completely from any perturbation, on **any** sampler. Measured
  2026-08-18: two arms differing only in sage `mode` diverge at **frame 0**, at
  the same PSNR as two unrelated clips, under `er_sde` and under the
  deterministic `res_multistep` alike. The output of the changed arm is a
  *different sample*, not a degraded version of the same one, so "which clip
  looks better" is a draw from a distribution and answers nothing about the
  knob.
  - **This was got wrong once, the same day, in this file.** The first version
    of this rule blamed `er_sde`'s injected noise and prescribed a deterministic
    sampler. That was built, run, and refuted within the hour — the
    deterministic pair diverged no less. Keep the refutation: the tempting fix
    does not work, and the plausible mechanism was not the mechanism.
  - **So compare knobs at the call, not at the output.**
    `bench/grade_sage_on_capture.py` grades a kernel against an exact reference
    on captured activations, which is controlled by construction. That is
    currently the only controlled comparison this repo can make about a
    numerical knob.
  - **A perceptual claim about a numerical knob needs a distribution, not a
    pair** — many seeds per arm judged blind in aggregate. One clip per arm
    cannot support it however carefully it is stacked, and
    `docs/eval_comparison.md`'s blind tooling does not change that; it controls
    who knows which arm, not whether the arms are comparable.
  - **This retro-applies.** The 2026-08-13 A/B that chose
    `fp16 (most accurate)` was one clip per arm, so it compared two samples.
    Sound as a preference between two outputs; never controlled evidence about
    the kernel. Ranked against it, the call-level measurement is the stronger
    claim, which is the opposite of how the two were weighted at the time.
  - **A weight-level difference** — LoRA against checkpoint — diverges for the
    same reason and was always answering "does each arm satisfy the brief".
    Read those as briefs met, never as clips matched.
  - **Check the seed before trusting any pair at all.** The 2026-08-15 ordering
    arms carry a different seed per clip and were never paired to begin with.

## Reference implementations

`coderef/` (gitignored) symlinks the sister checkouts — diffusers,
DiffSynth-Studio, comfy-kitchen, sage-fork, triton, LightX2V, Minimax-H3-Turbo
— plus real clones of MiniMax-H3, Sana, MiniMax-Music3, h3-turbo-eval and
**`comfy-kitchen-sol`**, which is the one most cited here: `docs/morton.md` and
`docs/sol_upstream.md` quote its `.cu` files by path, and those ship in no wheel.

**Do not import Python from it.** The built branch is installed
(`comfy_kitchen_version: 0.2.31+sol.c04ef20`), so
`from comfy_kitchen.backends.eager.sol_attn import _pool` works without the
clone, and its `sol_attn`, `_pool` and `_normalize_key_bias` are structurally
identical to the vendored `bench/_sol_attn_reference.py`. Requiring the clone
and prepending it to `sys.path` is how `bench/analyze_routing.py` made itself
unrunnable on a box with the wheel and no checkout, which kept its red harness
skipped and inert. Use the clone for sources you cannot import; import the rest.
