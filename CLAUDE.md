# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. `README.md` is what ships and why.
This file is only what would cost you a session to rediscover — the operative
rule, not the story behind it. Stories live in `docs/` and the postmortems.

## Read this before anything else: do not trust prose, including this file's

**When prose and code disagree, the code is right. Verify or trace before you
act on any sentence here or in `docs/` — including sentences that name a
default, a shipped artifact, or what "every graph" does.** Those are the three
that rot fastest, because each has a machine-readable home and the sentence is a
second copy with no invalidation.

This is not caution for its own sake. On 2026-08-28 alone, prose lost to code
four times in one session: `docs/h3_references.md` said the shipped text encoder
was the v2 AWQ artifact when `h3_config.MODELS["clip"]` is `ENCODER_INT8`;
`docs/prompt_audit.md` said four prompt defects were "all still present" after a
commit had fixed them; a node tooltip stated a pixel floor 20x off; and a rule
written into THIS file — that reaching the vendor's reference floor needs a
second node — was wrong within the hour, because it was taken from a doc
sentence describing a code path the shipped graphs had left.

The generalisation, and `docs/config_drift.md` is the long form: **prose stating
a fact the code already knows is a cache, and this repo has no invalidation for
it.** A date does not help — every one of those claims carried one, and the date
told you when it was written rather than whether it is still true.

So: cite the observable, not the claim. `h3_config.MODELS["clip"]` over "the
shipped encoder is"; the node's `define_schema` over "the default is"; a walk of
`graph_paths` over "every graph". If you find prose that lost, **correct it and
say what it used to claim** — a reader who remembers the old sentence needs to
know it was withdrawn rather than quietly reworded.

## Settled, and re-derived from scratch anyway

Every item here was measured or decided, and every one has been rediscovered as
"news" in a later session. Read this before opening a marker, tokenizer, or
prompt-structure question; if you are about to establish one of these, you are
repeating work.

- **The released text encoder is byte-identical to stock Qwen3-VL-32B-Instruct**
  — all 14 shards, hub LFS SHA-256, `bench/results/2026-08-25_released_encoder_is_stock.json`.
  MiniMax shipped no encoder post-training. **Read the next bullet only through
  this one.**
- **The seven H3 marker rows are untrained, and that fact discriminates
  nothing.** Measured 2026-08-21 in four artifacts
  (`bench/results/2026-08-21_h3_token_embeddings.json`): the seven sit with the
  untrained padding tail, not the trained Qwen specials. This is about the
  ENCODER's input embedding table, not the DiT. But since the encoder is stock
  in its entirety, *every* row is untrained-by-MiniMax and the seven could not
  have been otherwise — so it is a consequence of the bullet above, not a
  finding about markers. **Both live hypotheses predict it**: that MiniMax used
  single ids against a frozen encoder, so the DiT learned fixed-but-arbitrary
  vectors as delimiters; and that MiniMax BPE'd them, so the ids never
  appeared. Row norms cannot separate those, and neither can prediction
  sensitivity. The discriminator is `release_id` against `mean_init_rows` and
  it has not been run. **Do not read "untrained" as "the fixed tokenizer is
  wrong"** — a session did exactly that on 2026-08-27 and had to retract it.
- **The seven marker ids ARE fixed by the release, and no JSON literal assigns
  them.** Both halves matter, because the second half has now caused two
  sessions to conclude the first is false. Grep the release for `151669` and you
  get nothing: they are absent from `vocab.json`, from `tokenizer.json`'s
  `model.vocab` and `added_tokens`, and from every `added_tokens_decoder`. They
  appear only as strings in `additional_special_tokens`. That is not a gap —
  loading the directory resolves them deterministically, and anyone who loads
  it, MiniMax included, gets the same numbers:

      AutoTokenizer.from_pretrained(<release>/tokenizer)  -> Qwen2Tokenizer
      len(tokenizer) 151676   =  151643 base + 26 added + 7
      <d> 151669 … <|caption_end|> 151675, and `<d>` round-trips as ONE token

  So `comfy/text_encoders/minimax.py`'s "ids fixed by the released tokenizer" is
  correct, and `release_id` addresses the rows the vendor's own pipeline
  addresses. Verified by loading, not by reading:
  `bench/results/2026-08-27_marker_tokenization_alignment.json`. **Do not
  re-raise the missing literal as a finding.**
- **The DiT was trained on MiniMax's own prompt-writing structure, and it
  differs by mode.** `internal/official_prompt_guides/` holds the base and ref
  guides. Prompt structure is not a free parameter you may restructure to chase
  an effect; deviating trades a trained-on layout for an untrained one.
- **1344x768 is a trained canvas, and small canvases have inverted a finding.**
  Anything meant to inform a shipped decision gets measured there. A cheap
  canvas is for making a harness run, never for the number you will quote.
- **PDD quality is governed by the SIGMA SCHEDULE'S COARSENESS, not by the
  evaluation count** — and this was established the long way round, so do not
  re-derive it. A 4-evaluation arm renders jagged video and scratchy audio; a
  6-evaluation arm on a tail-weighted partition renders acceptably. But the
  count is not the variable: **two arms with the SAME evaluation count and the
  SAME block-width multiset, differing only in whether the wide blocks sit at
  the front or the end, have materially different coarseness** — and the
  coarser one is worse. (No magnitude here: two sessions summed the drift
  over different index ranges and got a constant offset, ranks identical.
  `docs/research/pdd/audio_under_pdd.md` carries the definition it uses.)
  - **Where the coarseness lands is what matters, and under shift 12 that means
    the TAIL.** The uniform 4-evaluation partition spends its last Euler step on
    **80%** of the trajectory; every partition that keeps a narrow final block
    gets that to 63.2%, including at five evaluations. **Four evaluations cannot
    be fixed** — `[8,8,8,8]` is the only partition of the 32-point grid into
    four blocks that is legal under the trained envelope, so its 80% tail is
    forced rather than chosen.
  - **Both streams degrade together and for the same reason**, which is why
    "audio is the thing that is off" was a misreading that survived a whole
    session. It reproduced because raw video rel L2 is dominated by the DC term
    every arm preserves; remove the mean and video degrades monotonically with
    coarseness too. **A metric that says one stream is fine is a claim about the
    metric until you have checked what it is blind to.**
  - **Any coarseness statistic gives the same answer, which is a warning as much
    as a result.** Summed drift computed through the audio change-of-variable
    and summed drift computed in pure video time **rank all six arms
    identically**. So no partition experiment can attribute the effect to an
    audio-specific mechanism — to identify one you must vary the TRANSFORM at
    fixed partition (`shift_audio`, graded), never the partition.
- **The encoder THIS INSTALL loads is `ENCODER_INT8`, and the v2 AWQ lane was
  closed rather than adopted.** `h3_config.MODELS["clip"]`, and every
  `CLIPLoader` in the shipped graphs, resolves to
  `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`. **Do not confuse this with
  the first bullet above** — that one is about the *released* encoder's weights
  being stock; this one is about which local artifact is wired, and they are
  different questions. On 2026-08-28 `docs/h3_references.md` was found asserting
  the shipped encoder is v2 and reasoning from v2's bounds; the conclusion it
  drew survived, the mechanism did not. **The observable is
  `h3_config.MODELS["clip"]`, never a doc sentence and never a lane's name.**
- **Reference-image sizing has three knobs, one of them is a prior rather than a
  measurement, and the vendor-matching settings are already in every shipped
  graph.** [`docs/h3_references.md`](docs/h3_references.md) is the authority and
  `h3_config.py`'s `REF_QWEN_SHORT_EDGE` note is the current record of *why*.
  The operative rules, all of which have cost a session:
  - `size_policy=max` with `short_edge=2048` matches the vendor, which caps the
    short edge and applies no area cap; the node default `match` sizes to the
    generation's pixel area instead, which is several times smaller on a
    landscape canvas. Image references are deliberately exempt from the area cap
    that binds video, which is how one can legitimately exceed what the video
    itself may reach.
  - **`short_edge` targets the SHORTER side, and `max` only shrinks until you
    say otherwise.** The scale is `short_edge / min(w, h)`, clamped by
    `min(1.0, ...)` unless `allow_upscale` is on, then snapped to 32
    (`reference_geometry.py::fit_reference_image`). So `short_edge` alone sets a
    ceiling: a source already under it is untouched, and the knob does nothing
    at all for that source until `allow_upscale` is on. Both knobs live on
    `MiniMaxH3AppendRefImage` and are read ONLY under `max`.
  - **Do not reach for `MiniMaxH3ReferenceFit`.** It carries
    `is_deprecated=True` since 2026-08-28, so it is gone from the node picker
    while saved graphs that wire it still load, and **no shipped graph wires
    it**; the typed path
    (`AppendRefImage` -> `MiniMaxH3ReferenceConditioning`) carries both knobs
    itself and does one resize with the canvas in scope. `docs/h3_references.md`
    says in places that you need `max` *and* `ReferenceFit(allow_upscale=True)`
    to reach the vendor's floor -- **that describes the older native path**, and
    on the typed path `allow_upscale` on the append node is the whole of it.
    `reference_fit.py`'s own docstring is the current statement, and two of that
    node's inputs are inert.
  - **`qwen_short_edge` must not be 0 on the shipped encoder.** Reference tokens
    land in the TEXT segment ahead of the prompt, so they compete with it rather
    than merely lengthening the sequence, and unclamped they can crowd the
    prompt into a small minority of its own segment. The knob is *exactly inert*
    under the v1 snapshot, live under v2, and widest under what actually ships —
    so which regime you are in decides the answer, and reading the wrong row
    gives the opposite one.
  - **The shipped value is a PRIOR, not a measurement**, resting on a single
    render at one seed. Cite it as a default, never as measured. The A/B/C
    reference-view ablation (`workflows/h3_probe_refview_*`) is BUILT and has
    never been rendered, and the experiment that would settle the mechanism
    holds the weights fixed and varies only the bounds — so it needs no render
    at all.

## Guiding Principles
- **Before writing a number into prose, substitute a different plausible value. If the reader's next action is unchanged, the number is decorative — delete it.** "Sixteen rows claim calibration" and "seventeen rows claim calibration" prompt the same next step, so the count is liability carrying no information.
  - A number that survives the test is one of two things. **Normative** — a limit you are setting, an exit code, a threshold — which cannot drift, because the world moves toward it. Or **descriptive**, in which case it needs an observation point: a date, a commit, an attribution, or past tense. Descriptive counts belong only in dated records.
  - **Generating a decorative number is not a lesser fix than deleting it.** It makes the claim permanently true and permanently useless, and still charges every reader a reconciliation against what they can see. Delete first; generate only what passed the test.
  - Auditing prose already written rather than prose being written: `claim-audit`.
- **A default is not a decision, and shipping is not evidence.** A value that is
  the node default, sits in every graph, and is named in three docs has acquired
  standing through repetition, not measurement. Before citing one, ask what it
  would take to find out it was wrong — if the answer is "a record nobody has
  written", say so in the sentence that uses it.
  - Four instances on 2026-08-28, all found by asking the same question of
    values nobody doubted. `qwen_short_edge=512` is a prior resting on one
    render at one seed. `ref_upscale=True` cost 6,300 extra reference rows per
    step for a benefit this repo has never measured, and was flipped once asked.
    `nfe=0` was the ordinary mode wearing a falsy sentinel. Sol's `tau=1.0` DOES
    have a matched controlled measurement behind it — and is also the lowest
    value ever tested, in the direction that was still improving, which is a
    different claim from optimal.
  - **The tell is that nobody chose it.** Each was inherited — from a vendor
    default, an earlier era's tooling, or a first draft — and then read as
    settled because it was everywhere. `docs/SOLATTN.md` has the worked audit;
    `docs/config_drift.md` has the prose half of the same disease.
  - The cheap discipline: when you write a value into a constant, say in the
    same comment whether it was measured, inherited, or reasoned. Three words
    that stop the next reader re-deriving it or, worse, citing it.
- DO NOT write meaningless tests - if something can't be tested or be a simple red/green, then find another way to make sure you're measuring and testing what you think you are
- **Sol-Attn is ALWAYS ON by default in every shipped video workflow.** Bypassing is reserved for explicit testing / comparative experiments. Some graphs legitimately do not wire it, and the exempt set is `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS` plus the single-frame class it derives from `h3_config.GRAPH_DIRS` — read it there rather than counting the kinds here, which is what went stale. **That derived class has been empty since the single-frame path was parked on 2026-08-27** ([`docs/h3_image_editing.md`](docs/h3_image_editing.md)); it stays derived rather than deleted, so a directory added back to `GRAPH_DIRS` re-arms it without anyone remembering to. The kinds still exempt, and why: `workflows/bench/*_stamped_api.json` (dense baselines — the thing Sol is measured against) and the two `h3_probe_capture_ref3*` graphs (activation capture, which must record the true attention inputs rather than Sol's output) wire `MiniMaxH3SageAttention` instead. Two kinds wire neither sage nor Sol: `h3_probe_turbo_768p_sla_router` (since 2026-08-20) runs `MiniMaxH3SLARouter`, the sparse top-k router the Turbo-SLA LoRA was distilled under; and the `h3_probe_ref2v_pdd*` arms (since 2026-08-26) run stock SDPA because that is what the vendor runs, which costs about 2.4x and must not be copied outside a reference arm. **The node the graphs wire for Sol is the vendored `SolAttnMiniMax`, not this repo's `MiniMaxH3SolAttnCurve`** — that one is the Hilbert permutation node and appears in no shipped graph. **Enforced by `bench/check_attention_defaults.py` since 2026-08-18**: it grades every graph's Sol and sage values against `h3_config`, by reachability from the output node rather than node presence, and asserts each exemption above is *necessary* (an exempt graph with live Sol goes red).

## What is where

### Read these before you start

| file | what it answers |
|---|---|
| [`docs/wiki/index.md`](docs/wiki/index.md) | **the entry point, and the only generated one.** A router: where to start, who owns each answer, and which documents nothing links to. Two written pages beside it — `references.md` (what each `coderef/` checkout implements for H3 and what has been compared against it) and `stages.md` (per render stage: our code, its owner, its guard, the implementation to compare against). Regenerate with `bench/build_wiki_index.py`; never hand-edit `index.md` |
| [`docs/roadmap.md`](docs/roadmap.md) | what we are trying to find out next, and what would count as finding it. **Start here** if the question is what to work on. Designing a new probe or prompt: the `h3-experiment` skill in `.claude/skills/` routes to the files that own each step (this one first), and restates none of them. |
| [`docs/evidence.md`](docs/evidence.md) | what is measured, what is retracted, and what must not be relied on. **Start here** if you are about to state a number. |
| [`docs/checks.md`](docs/checks.md) | the index of every check, the standard it is held to, and the standing uncontrolled-requirement audit. **Start here** if you are about to change behaviour or add a check. |
| [`docs/comfyui_vendor_gaps.md`](docs/comfyui_vendor_gaps.md) | every known divergence between this install and the release, with practical impact, priority, and which are enforced by nothing. **Start here** if the question is "what is still wrong against the vendor". A dated snapshot that defers to the docs below, not a fourth authority -- where it disagrees with an owner, the owner is right |
| [`docs/config_drift.md`](docs/config_drift.md) | **why `h3_config.py` drifts and what would stop it.** Four failure classes from a 2026-08-28 audit, the six flatly-wrong claims it found (corrected), and what will NOT fix it -- review discipline is refuted outright, by a contradiction `git blame` shows one commit wrote both halves of. **Read it before adding prose to a config file, or before proposing a docs-match-code checker.** Its general claim: prose stating a fact the code already knows is a cache with no invalidation, so derive the fact or write the sentence to explain a decision instead of reporting a state |
| [`docs/sustainability.md`](docs/sustainability.md) | the direction argument: the instrumentation is healthy, the closure rate is not, and the fix is to generalise an aging rule this repo already has in one lane rather than add instruments. Carries its own **do not do** list and a record of the four claims an adversarial review broke — including a proposed check class that was already installed three times, which is the 2026-08-17 failure mode repeating. Read it before proposing process; argue with it or delete it |
| [`docs/prompting.md`](docs/prompting.md) | **how to write an H3 prompt, for every mode, and the canonical source.** A working manual, not a summary: the closed vocabularies in full, the exact Part One templates per mode, section layouts, all seven markers, and a worked example per mode that grades clean through `preflight_graph.py`. **You do not need `internal/` to use it.** Every rule carries its layer -- GUIDE (the vendor's, and off-distribution if broken), OWNER, HOUSE, OPEN -- and every GUIDE rule says whether the guide **states** it or only **shows** it in an example. That second distinction is the one that matters: two rules have been invented here by reading examples as rules, and both were retracted. §11 maps every rule to what checks it |
| [`docs/prompt_catalogue.md`](docs/prompt_catalogue.md) + [`docs/prompt_audit.md`](docs/prompt_audit.md) | **every prompt this repo renders, and whether it follows the guides.** The catalogue is GENERATED by `bench/build_prompt_catalogue.py` from the graphs themselves and judges nothing; the audit is hand-written, keyed to its scene names, and carries a keep/revise/rewrite/discard verdict each. **Three authorities are kept separate and must stay that way** -- the official guides (off-distribution if broken), `internal/PROMPTING.md` (house rules, may themselves be wrong), and the STATED RULE / NOT A RULE adjudication in the generator. Start here before writing or debugging a scene |
| [`docs/custom_node_gaps.md`](docs/custom_node_gaps.md) | the companion question: what OUR nodes do end to end, and where that differs from sglang, LightX2V, DiffSynth, diffusers and native ComfyUI. **Start here** if the question is "what does this pack actually do differently". Carries the node inventory by class (load-bearing / convenience / instrumentation), the three canonical dataflows, the reading traps in the shipped JSON, and its own enforced-by-nothing table. Same standing as the file above -- a snapshot that defers to owners |

### Reference, when you touch the thing it covers

| file | what it answers |
|---|---|
| [`docs/comfy_notes.md`](docs/comfy_notes.md) | running and starting comfyui, generating workflows, prompting, useful tips and tricks when working with comfyui custom nodes and workflows. Also holds the `node_id` rule and the `import nodes` trap |
| [`docs/hardware.md`](docs/hardware.md) | the box every number here was measured on: what bounds this workload, what has been ruled out as a bottleneck, and which host settings silently invalidate a timing comparison. **Read before quoting a render time.** Carries no values — `bench/hwinfo.py` prints those |
| [`docs/open_experiments.md`](docs/open_experiments.md) | what is deliberately not measured, and the blocker for each |
| [`docs/SOLATTN.md`](docs/SOLATTN.md) | **the Sol-Attn entry point and the authority.** Knobs, sink, measured arms, ordering, and its own do-not-rely-on table. It owns every Sol-Attn number measured on this box. **Three deep dives are reached only through it and must not be quoted against it:** `morton.md` (token reordering; its link 6 — does any of it reach the output — is unverified), `h3_input_impacts.md` (canvas/frames/Sol interaction, the `latent_t % 4` length effect, and why there is no 100k token budget), `sol_upstream.md` (what upstream claims; asserts nothing of ours, and their speedups are not comparable) |
| [`docs/research/conditioning_nodes.md`](docs/research/conditioning_nodes.md) | what was built against the two conditioning defects, what was deliberately not built, and the **five load-bearing contracts in `MiniMaxH3ReferenceToVideo` that live only in code comments**. Read it before replacing a conditioning node -- that list is the acceptance criteria and nothing asserts any of it |
| [`docs/research/official_weights_metadata.md`](docs/research/official_weights_metadata.md) | what the published release declares against what ComfyUI assumes: the tokenizer's seven unreachable H3 special tokens, the partition split that lives in the weights, and the list of things checked and found clean. **Read before writing a prompt that uses a marker** |
| [`docs/research/h3_partition_distance.md`](docs/research/h3_partition_distance.md) | **how far apart fl2va and ref2va actually are, measured per component.** The companion to the file above: that one owns what the release DECLARES about the split, this one what the weights differ by. The two checkpoints are within a few percent at the output heads with identical key sets, while the PDD LoRAs distilled on them are near ORTHOGONAL -- and the divergence runs one way with depth, perfectly unrelated through block 30 and converging toward the output. Read it before assuming a wrong-partition load degrades gracefully, or before merging anything across partitions |
| [`docs/research/sglang_h3_pipeline.md`](docs/research/sglang_h3_pipeline.md) | **what sglang's H3 pipeline does, stage by stage, at the source level**: request and admission, time grid and canvas, media ingestion, the Qwen3-VL encode, the VAE encodes and seeds, the packed sequence and positions, the DiT forward, the denoise loop, decode and output, the runtime and quality gates, and 27 numbered insights. Compares nothing; read it before `sglang_comparison.md` |
| [`docs/research/sglang_comparison.md`](docs/research/sglang_comparison.md) | **what the vendor serving path does that we do not, and where ours is the weaker version.** Owns the runtime and optimization comparison against sglang; the reference-conditioning comparison is `h3_references.md`'s and is not restated there. Read it before proposing an optimization -- it records what is already priced, and one hypothesis it killed |
| [`docs/research/technique_transfer.md`](docs/research/technique_transfer.md) | **what transfers from LLM and ViT serving to H3 and what does not**: each technique, the model property it needs, what it becomes here, and its status. Read before proposing a borrow from the LLM world; the two open ones are named there with their first measurement |
| [`docs/research/h3_dit_implementations.md`](docs/research/h3_dit_implementations.md) | **the DiT itself, across every implementation of it available here**: module tree, packed sequence, rope, attention, modulation, forward and the sampler that drives it, for ComfyUI against diffusers, sglang, DiffSynth and vllm-omni. Owns the numerical comparison; defers to `sglang_comparison.md` for runtime and to `h3_references.md` for reference conditioning. The release ships no DiT code, so diffusers is the reference of record -- its own `model_index.json` names that class. Read it before reimplementing a stage or concluding a knob is wrong |
| [`docs/research/comfyui_h3_t2va_trace.md`](docs/research/comfyui_h3_t2va_trace.md) | **what ComfyUI's own code does, call by call, for one t2va render**: the four loaders, int8/convrot representation and where the GEMM actually runs, the packed layout and position grid, one forward stage by stage, the block x50, the output heads with and without PDD, and both VAE decodes. Traces ours; compares nothing -- `h3_dit_implementations.md` owns the five-way comparison. **Start here if the question is what a dtype, a buffer or a kernel is doing on this box.** Every surprising claim in it was re-derived by execution, and section 14 is its own sharp-edge list |
| [`docs/h3_references.md`](docs/h3_references.md) | **every reference type, its processing, measured cost, label rules, and the three sizing knobs — two of which are constantly confused for each other.** Start here for "what should `AppendRefImage` be set to"; the settled answers and the one value that is only a prior are summarised in the bullet above, and the three encoder regimes that change what `qwen_short_edge` even does are tabulated there. Carries its own dated correction about which encoder ships |
| [`docs/h3_image_editing.md`](docs/h3_image_editing.md) | **PARKED 2026-08-27** — the single-frame image gen/edit path in the past tense: what it was, what moved to `archive/`, and what un-parking would take. It owns that record. Nothing in the live tree depends on it, and no graph, check or node is generated from it |
| [`docs/h3_resolutions.md`](docs/h3_resolutions.md) | every legal canvas and what each costs -- it owns the count, do not restate it here. `h3_input_impacts.md` is its deep dive |
| [`docs/h3_geometry_and_nodes.md`](docs/h3_geometry_and_nodes.md) | the frame grid, the token maths, and which node to use |
| [`docs/h3_ref2v_distillation.md`](docs/h3_ref2v_distillation.md) | why ref2v resists step distillation |
| [`docs/h3_pdd.md`](docs/h3_pdd.md) | **Parallel Decoding Distillation**: what the alibaba-pai Acc LoRAs actually are (not a step distill -- replicated output heads over a 32-point grid), the three surfaces one file reaches, why a PDD arm moves only the step count, and the two silent traps -- identical fl2va/ref2va key sets, and a pruned base with nowhere to put the adaln delta. Owns the converter and node contract |
| [`docs/research/pdd/pdd_implementations.md`](docs/research/pdd/pdd_implementations.md) | how our PDD compares to the other four implementations (the alibaba-pai source, Comfy-Org/ComfyUI#15908, UtilsCollection, and a third-party pack), and how this lane reached its current shape. **No other engine implements PDD at all** -- diffusers, LightX2V, DiffSynth and sglang were each searched and none does. Defers to the file above; §4 is a list of places it found that file stale |
| [`docs/eval_comparison.md`](docs/eval_comparison.md) | **the A/B process, and the authority for it**: how a rendered comparison is rendered (matched seeds), blinded, scored before unblinding, and recorded as a distribution. Every comparison that will be quoted goes through its section 3; the `h3-ab-session` skill in `.claude/skills/` only routes there. Also the stacking layout rules |
| [`docs/capture_manifest_schema.md`](docs/capture_manifest_schema.md) | activation capture manifest schema: generation parameters, prompt, reference metadata, token accounting, and tensor checksums. **Do not name a version here** -- the accepted set is `bench/check_capture_manifest.py::SCHEMA_VERSIONS` and what new manifests are stamped with is `bench/generate_capture_manifest.py`. A version written into prose drifted once already, which is why that constant exists |
| [`docs/bench_plan.md`](docs/bench_plan.md) | pre-registered predictions and the runs that scored them |
| [`docs/check_postmortems.md`](docs/check_postmortems.md) | **history, not operative — skip it** unless you are investigating one specific check or about to write a control. Per-defect narrative and frozen run logs; every count in it is stale by design |

### Code and directories: the operative rule

| path | the rule |
|---|---|
| `vendor_config/` | the published release's own config files, verbatim, with `vendor_config.py` as the reader. **Anything the release declares is read from here, never retyped** -- special tokens, pixel bounds, patch geometry, partition task lists. Guarded by `bench/check_vendor_config.py`, which says so loudly when it could not reach the release to compare |
| [`workflows/h3_config.py`](workflows/h3_config.py) | every shared constant. Nothing here may have a second copy anywhere. **Every check that walks graphs must go through `workflows/h3_config.py::graph_paths`** -- a bare non-recursive glob passes green over a subset, which is how `workflows/image/` went ungoverned by two prompt checks before it was parked. The directories it walks are `GRAPH_DIRS`; adding one there is what makes every walker see it at once. Enforced by `bench/check_graph_discovery.py` |
| [`workflows/build_workflows.py`](workflows/build_workflows.py) | generates all graphs. Never hand-edit a `workflows/*.json` — and the inverse: **editing the generator is half the change, and nothing is true of a graph until it is rebuilt.** On 2026-08-21 a corrected note sat in this file across two commits while the shipped graph still carried the old text; only an explicit regeneration request caught it. Rebuild before you claim, and before you commit. |
| `archive/` | parked work, kept as history rather than deleted. Generated by nothing, walked by no check, absent from `GRAPH_DIRS`, imported by no `__init__.py`. The single-frame image path moved here 2026-08-27 ([`docs/h3_image_editing.md`](docs/h3_image_editing.md)), and `archive/single_frame.py` was **the last thing in this pack that modified ComfyUI core** -- nothing here patches core now, so a symptom that looks like a core patch is not ours |
| `bench/check_*.py` | fast, mostly CUDA-free guards |
| [`bench/preflight_graph.py`](bench/preflight_graph.py) | **run this before you queue a reference render.** Grades the prompt against the guide's mechanical rules and prices the packed sequence, statically, on any graph path including hand-built ones. Reports, never refuses |
| `bench/bench_*.py`, `bench/smoke_h3.py` | need a GPU and a live server |
| `bench/run_graph_arms.py` -> `bench/blind_batch.py` -> `score.html` -> `bench/score_session.py` | **the bench path for anything judged by a person.** Arms rendered with matched seeds and recorded substrate; clips blinded with a sealed key in `internal/blind_keys/`; scored in the generated app; the key opened only by the joiner, which writes the verdict record. A comparison that skipped any of these is two samples, not a result (the different-sample rule below). `docs/eval_comparison.md` section 3 is the process; do not restate it here |
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
- **Proving a check can go red earns its cost when the outcome is genuinely
  open, and is ceremony when you have just measured it.** The rule above is
  about how a check is *designed*; it is not a demand to stage a
  demonstration for every one. Both shapes happened on 2026-08-26. The one
  that paid: `bench/convert_pdd_lora.py`'s fit assertion was *expected* to
  catch a bake solved against the wrong partition's basis, a deliberate
  violation showed it did not -- both bases span nearly the same subspace, so
  a residual is blind to which one was used -- and that moved the guard to a
  different comparison and corrected what two files claimed. The ones that did
  not pay: cases whose red-proof restated a measurement taken minutes earlier
  in the same session, and found nothing, because the answer was already on
  screen.
  - **The test is whether you can predict the outcome.** If you can, write the
    case and move on. If you cannot -- you believe a guard covers something and
    have not seen it fail -- that is the one to run, and it is where the escaped
    instances come from.
  - Judgement, not a gate. Build the control when it is load-bearing and cheap;
    skip the theatre when it is neither. Practical and reliable beats
    ritually complete, and a session spent proving things you already knew is a
    session not spent finding what you did not.
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
- **An assumption that has only ever met one implementation is not a tested
  assumption.** Adopted 2026-08-22, with the escaped instance that earns it:
  two harnesses reconstructed the pre-fix tokenizer by keying off the *name* of
  the constant holding the token list. Correct against the only version that
  existed. Upstream's fix named it differently, so under it the grader read "no
  patch present" and would have returned the CORRECTED tokenizer labelled as
  stock -- reporting near-zero deltas, which reads as a retraction of its own
  measured numbers. Nothing in the suite could have caught it; what caught it
  was running against somebody else's implementation. So when code branches on
  a detail of one implementation -- a constant's name, a private attribute, a
  message string -- branch on the *observable* instead (there, the vocabulary),
  and say what would have to change for the branch to be wrong.
- **Prefer a control the check compares against** — a frontend-written graph,
  the pre-fix code, an independent implementation — over asserting against
  numbers the test computed itself, **including through a helper the check
  defines rather than imports.** The helper is the part that escapes: on
  2026-08-28 `check_pdd_sigmas.py` graded its headline case against ComfyUI's
  own `calculate_sigmas` and still could not fail, because three other cases
  routed through an `emitted()` that restated the node's expression. Dropping
  the `1.0 -` in `pdd_lora.py` left the whole file green. `docs/checks.md` has
  the long form.
- **After a fix, ask which code paths were DEAD before it and are live after.**
  Those are unexercised by construction and no existing test covers them,
  because until the fix there was nothing to cover. **Four instances on
  2026-08-28, three of them the same day as each other.** `smoke_h3.py --steps`
  had been a no-op on PDD graphs, so nothing had ever evaluated its default
  against `resolve_emit_steps`; making it work made the default raise on every
  PDD graph. The PDD node emitting SIGMAS made a schedule reachable that
  `BasicScheduler` had always supplied, and the headfree arm consumed it with
  no guard. The shift guard made a comparison reachable that had never run, and
  was silent on the exact render it was written for. Each fix was correct in
  itself; **a correct fix moves where a constraint applies, and it moves it
  somewhere nobody is looking.** Verifying the thing you changed is not
  verifying the thing your change newly touches.
- **"It works" is not "it was trained for".** A capability that functions
  because a code path has no bounds check is not one the model was trained to
  do. Established for prompt structure above; it applies to layout, geometry and
  conditioning too, and the question to ask of any knob is **does the vendor's
  own pipeline ever produce this input?** Instances: a third-party pack pins
  audio at a fractional negative keyframe index that stock nodes cannot produce
  and the release never emits — it works because core applies no cast and no
  bounds check. Our own `qwen_short_edge` view split currently answers no to
  the same question, which is why it is priced rather than proven.
- **The same standard applies to claims, and re-reading your own work does not
  meet it.** On 2026-08-13, eight substantive defects were found here and in the
  sage fork; not one was caught by whoever wrote it. Every one came from a
  second reader.
- **A claim derived from a call site, a docstring, or a plausible mechanism is
  an inference.** Say which kind of evidence you have *inside* the claim —
  "reported, not verified: a source read, not a build" survives being quoted,
  where a trailing "(unverified)" reads as hedging and gets trimmed.
- **Two models live in this repo, and the words for their parts do not
  disambiguate the stage.** "Attention" and "capture" each name something at the
  DiT and at the Qwen3-VL encoder; a fact about one is not a fact about the
  other. Three instances on 2026-08-26, every one carrying a DiT-side fact to an
  encoder-side conclusion: a per-module quant-delta table read as encoder
  evidence (its modules are `blocks.N.`, the DiT's); `h3_capture.py` being
  DiT-only read as "no encoder capture exists"
  (`bench/capture_h3_encoder_states.py` does, tapping the layer-0 input and the
  layer-49 output); and "these graphs wire no attention patching" read as an
  encoder property (sage, SLA and Sol all take `io.Model.Input`, and the encoder
  resolves its own `optimized_attention_for_device` inside its decoder forward).
  **The tell every time was a type or a prefix — what input the node takes,
  what a module name starts with — never the vocabulary of the claim.** Check
  which stage a thing attaches to before carrying a claim across.
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
  - **A weight-level difference** — LoRA against checkpoint — diverges for the
    same reason and was always answering "does each arm satisfy the brief".
    Read those as briefs met, never as clips matched.

## Reference implementations

`coderef/` (gitignored) holds the sister checkouts, some symlinked and some
real clones; `ls -l coderef/` is the list. **`comfy-kitchen-sol`** is the one
most cited here: `docs/morton.md` and `docs/sol_upstream.md` quote its `.cu`
files by path, and those ship in no wheel.

**Do not import Python from it.** The built branch is installed
(`comfy_kitchen_version: 0.2.31+sol.23d1a66`), so
`from comfy_kitchen.backends.eager.sol_attn import _pool` works without the
clone, and its `sol_attn`, `_pool` and `_normalize_key_bias` are structurally
identical to the vendored `bench/_sol_attn_reference.py`. Requiring the clone
and prepending it to `sys.path` is how `bench/analyze_routing.py` made itself
unrunnable on a box with the wheel and no checkout, which kept its red harness
skipped and inert. Use the clone for sources you cannot import; import the rest.
