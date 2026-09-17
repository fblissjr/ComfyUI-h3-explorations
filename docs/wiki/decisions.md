# Decisions and reversals

Written by hand. One dated line per decision the owner made, or per claim in
the prose that was corrected: what changed, what it used to say, where it
lives now, and the commit. Newest first. `CLAUDE.md` and `VISION.md` carry no
history notes; this page is where those go. Another document may also keep a
dated note in place, where a reader would otherwise trust the stale text.

Older history lives elsewhere and is not copied here:

- [`CHANGELOG.md`](../../CHANGELOG.md): every change, by version.
- [`docs/rules_history.md`](../rules_history.md): `CLAUDE.md` as it stood
  before the 2026-09-03 cut, frozen.
- [`docs/roadmap.md`](../roadmap.md): "Closed lanes", and the dated "Owner
  decisions" and forward-plan sections.
- `bench/results/`: the verdict records, each with its conditions.

## 2026-09-17

- **`SageChainAssert` taken out of every generated workflow and the
  generator** (owner, 2026-09-17; `cfeeaa1`). The node stays registered, so
  saved graphs still load.
  - **Why.** On the default chain it could only confirm that no sage kernel
    ran. Its generation-time flags refused an intended editor swap to our
    sage node on `h3_candidate_t2v_pdd8_baked`.
  - **Lost with it.** The call-time probe on the sage arms, the "nothing
    patched" guard on the baselines, and the "[h3] sol window" log line.
    `bench/check_attention_defaults.py` still grades the wiring.
  - **Corrected.** Three docs still showed our sage node as the default dense
    node after the 2026-09-15 flip, which that change's prose sweep missed:
    - `docs/h3_geometry_and_nodes.md`'s chain block;
    - `docs/wiki/stages.md`'s attention row;
    - `docs/SOLATTN.md`'s Ordering diagram, which also showed the assert.

## 2026-09-15

- **Two bench manifests and one probe graph retired** (the owner deferred the
  call to the sage-fork session, which chose it; `531fbab`). Both would re-run as
  something other than their records: `bench/ladder_arms.json` (rendered
  2026-09-03, verdict `bench/results/2026-09-04_ladder_2026-09-03_verdict.json`)
  refuses on its sage-mode patch now that `h3_text_to_video` has no sage node,
  and `bench/sol_core_ab_arms.json` with `h3_probe_t2v_sol_core` (rendered
  2026-09-10, no verdict record) would set ours on kitchen against core's node
  on sage. The four manifests that pointed at the ladder for what each scene
  tests carry the descriptions inline. **Corrected:** `docs/wiki/next_steps.md`
  said the core-versus-ours A/B was "BUILT not rendered"; its 2026-09-10 records
  say it rendered.
- **Sage out of the default attention chain; Sol's `qk_balance` on** (owner,
  2026-09-15; `28d8ee5`). **Reverses the 2026-09-04 floor decision** ("sage
  always on is the floor", `docs/roadmap.md` forward plan 2026-09-04, quoted
  in `bench/sol_core_ab_arms.json` and `bench/sol_nosage_arms.json`, which stay
  as records of what those runs held). Every video graph that names no mode
  now carries core's `ModelAttentionBackend` at `h3_config.DENSE_BACKEND_NODE`
  under Sol, and `h3_config.SOL_RECOMMENDED_CUDA` carries `qk_balance=True`.
  Taken on capture grades and one render per arm, not a blind verdict: the
  community-chain arms were unscored when it landed
  (`bench/results/2026-09-15_block49_community_chain.md`,
  `bench/results/2026-09-15_ck_int8_attention_block49.json`). What stays on
  sage or stock attention, and why, is
  `bench/check_attention_defaults.py::FLOOR_STEMS`. `h3_probe_t2v_balanced`
  and `h3_probe_t2v_ck_balanced` retired. Prose that said the fallback under
  Sol is sage on every graph (`docs/SOLATTN.md` knob table and two dense-block
  sections, `docs/wiki/next_steps.md` "Now") carries a dated note in place.
  The policy page's table rows for this landed early, in `76b2378`.
- **`docs/research/2026-09-14_block49_quant_error.md` moved to
  `docs/h3_block49_quant_error.md` and revised** (owner, 2026-09-15): it is
  a reference page now, not a dated research note. Its "So what" was
  rewritten with the blast radius (a design property of INT8 attention
  meeting this model, affecting every H3 user on any quantized-attention
  kernel, not a bug in any kernel and not confined to these forks) and a
  numbered status: diagnosed and closed; two levers built, measured, both
  off; open items are the blind visibility check, the factor in Sol's
  quantizer, and sending the finding upstream. Every link was rewritten;
  `bench/build_wiki_index.py` reaches the new path.
- **Kitchen build moved to `0.2.34+sol.2aff3c5`.** ComfyUI's pin went to
  `comfy-kitchen==0.2.34`; `h3-build` was rebased onto the tag by the recipe
  in `vendor/rebuild_kernel.sh`, old tip archived as `archive/h3-build-0.2.33`,
  six `blk_cnt` commits reapplied clean. Brings the H3 VAE kernels (#167/#175),
  persistent RoPE allocations (#162), `compress-mode=size` (#165). Kijai's
  branches carried nothing unmerged on the CUDA side; open PRs assessed and
  not carried: `docs/sol_upstream.md`, 2026-09-15. `docs/SOLATTN.md` "Install
  the CUDA kernel" used to say main had moved past 0.2.33 with no tag.
- **`docs/h3_block49_quant_error.md` gained sections 7
  (the sage-side `qk_balance` kernel fix) and "So what".** No default changed.

## 2026-09-14

- **The generator runs masked, as documented, rather than needing the card**
  (owner, choosing the code fix over a doc line saying it needs the GPU). Prose
  that lost: `docs/checks.md` ("While a render is on the card") said the
  generator takes ComfyUI's CPU path when no device is visible, and the
  generator's docstring said nothing touches the GPU; a masked build raised in
  `resolution.py`'s core import before the CPU switch ran, at `1345791` and
  after. Fixed in the generator (`_core_cpu_when_no_card`) and in
  `bench/check_generator_constants.py`; `docs/checks.md` is true again and the
  docstring now says a visible card gets a CUDA context (CHANGELOG 0.108.1).
- **The frozen-audio loop simplified before extending** (owner, after the
  footguns were listed: "simplifying is the only way to avoid that"; the
  worry was a new place starting where a window ends, not where the song
  changes section). `prompt_mode`, random window lengths, `frames:` lines, the
  implicit wildcard fallback and Fill Prompt Lists' `count` went; the list
  node's `seed` became `shuffle`; the song node gained `timeline` and
  `preview`, planned by `loop_plan.py` (CHANGELOG 0.108.0). Prose that lost,
  and what it claimed: the shot-workflow bullet below said the song node's
  prompt blocks and `frames:` lines do what those workflows did (the timeline
  and `--- label` blocks do now); `prompt_lists.py` said a placeholder with no
  list node reads `name.txt` or `name.json` from the wildcards folder;
  `docs/comfy_notes.md` and `bench/check_node_ids.py` said an input may only
  ever be appended (removal is the owner's call now that shipped graphs are
  API format); `docs/h3_geometry_and_nodes.md` gave `MiniMaxH3ImageToVideo`
  as the t2v/i2v node without saying the pack wires `MiniMaxH3Conditioning`.
- **Shipped workflows are API format only** (owner: "If I dont need a layout,
  node titles, bypassed nodes (we dont even use those), or notes in workflows,
  why dont we just simplify all of them to API json workflows?"). The UI twins,
  `build_ui`, `UIGraph`, `validate_ui`, `cross_check`, the in-graph notes and
  `bench/check_workflow_schema.py` went; `validate_api` gained the value-type
  and DynamicCombo-member checks `validate_ui` had, and member order. The editor
  loads an API file by input name (the frontend's `loadApiJson`, read from the
  source of comfyui_frontend_package 1.52.7, not exercised). Prose that lost,
  and what it claimed: `UIGraph`'s docstring said the generator made no
  widget-to-input conversions (its Resolution and PDD step links were exactly
  that); `bench/smoke_h3.py` said Sol-Attn ships off; `bench/check_graph_discovery.py`
  said it had no exemptions (it has one); `docs/custom_node_gaps.md` item 3
  described a UI/API node mismatch that no longer exists. The deleted notes
  also carried stale claims (the Sol window at 0.2/0.9, a keyframe canvas node
  the graphs do not wire, sage-only twins of the Sol probes); they went with
  the notes. Facts that lived only in the notes moved to their docs, and the
  move corrected doc prose that had lost: `docs/SOLATTN.md` printed `[sol_attn]`
  log lines (the prefix is `[h3-sol]`); `docs/h3_geometry_and_nodes.md` told
  readers to wire `MiniMaxH3KeyframeCanvas` on every i2v and fl2v graph (the
  conditioner owns the canvas), showed a node order through the deleted
  `SolAttnPatch`, and said `MiniMaxH3SigmaShift` sits in every shipped graph;
  `docs/h3_references.md` said the concise swap twin stays shipped (retired
  2026-08-22); `docs/h3_ref2v_distillation.md` said no v4 ref2va graph existed;
  `docs/h3_image_editing.md` pointed at a note in a graph that now lives only
  under `archive/`. Plan: `internal/2026-09-14_api_only_workflows_plan.md`.
- **Two duplicate Sol-on probes retired** (owner: "retire the
  duplicative/redundant workflows"). `h3_probe_sol_on` and `h3_probe_sol_on_i2v`
  matched `h3_text_to_video` and `h3_first_frame_to_video` in everything but
  the output name once Sol went on by default. What prose claimed: the
  generator read `h3_probe_sol_on` "against h3_text_to_video.json, which is
  now sage-only", and `check_bench_matches_shipped.py` said the plain t2v graph
  omits Sol. This session's own note first counted `h3_probe_sol_on_all_refs`
  among the duplicates; its reference video and audio disprove that, and it
  stays.
- **Prompt lists serve every loop, not the song node** (owner: "I should be
  able to use this for any type of looping workflow we build here"; agreed
  the approach the same day). A node that loops inside itself fills through
  `prompt_lists.fill_windows`, takes `lists` and fingerprints the wildcard
  files it may read, and `bench/check_prompt_lists.py` fails one that does
  not. A graph that loops by chaining nodes, or one clip per queue, uses
  `MiniMaxH3FillPromptLists` by index; value N of a list depends only on the
  list and N, so both routes agree. The fallback stays: a placeholder with no
  list node reads a wildcard file of its name, shuffled at seed 0 (the
  option the session recommended over always raising, or a seed widget for
  it, and the owner agreed).
- **No live preview node in generated graphs** (owner: it wastes GPU).
  `ModelPreviewOverrideKJ` with taeh3 left every UI graph and the generator's
  `preview` path. What the prose used to claim: the generator's node note and
  graph comment called it "arguably the largest optimization here" and worth
  more than any kernel knob; `docs/h3_geometry_and_nodes.md` said the same.
  The owner also answered the API-only plan's open questions
  (`internal/2026-09-14_api_only_workflows_plan.md`, top): loop stage 2 goes
  before the conversion.
- **The frozen-audio loop plan, confirmed** (owner, in answers the same day).
  Encoding comes before sampling: the track and each distinct prompt once.
  Reference stills go with every window's prompt, with a workflow for it.
  The song node keeps its window files in a per-graph working folder
  (`<prefix>_windows/`, kept by default) and gains resume from the first
  changed window, with its seed held fixed between queues so a re-queue can
  reuse windows. Prompt lists fill `__name__` placeholders -- not `{name}`,
  which the frontend's dynamic prompts rewrite, and not `[name]` or `<name>`,
  which H3 prompts already use -- one list per node with its own seed held
  fixed, in order or reshuffled on each pass with no repeat before the list
  is used up, advancing only when a window uses the placeholder. Wildcard
  files live in `wildcards/` under ComfyUI's input directory (the
  `--input-directory` override included) and are selectable on the node.
  `docs/wiki/next_steps.md` carries the stages.
- **The shot-per-window workflows retired, unrendered** (owner: never used).
  `workflows/h3_text_to_video_audio_freeze_shots.json` and
  `..._shots_repeat.json` with their API twins, the generator's
  `freeze_shots` knob, and `MiniMaxH3JoinWindows` went. The song node's
  prompt blocks and `frames:` lines do what they did. The two-window seam
  graph (`workflows/h3_text_to_video_audio_freeze_2windows_api.json`) stays.
- **A loop feeds the encoder no per-window image** (owner, after discussion).
  A frame from the previous window would carry drift forward and force an
  encode between windows; the anchor for identity is a fixed reference still.
  `docs/h3_audio_freeze.md` step 5 said a first-frame keyframe was "needed
  before the loop, because the loop anchors each window on a frame"; the loop
  shipped anchored on the previous window's frozen latent tail. Corrected in
  place.
- **References work on the fl2va checkpoint** (owner: done often). Three
  places said or implied otherwise and are corrected:
  `MiniMaxH3ReferenceConditioning`'s description ("Use an H3 reference
  checkpoint"), `docs/h3_geometry_and_nodes.md`'s node table (fl2va "for
  t2v/i2v, ref2va for reference-to-video"), and `docs/h3_audio_freeze.md`
  step 6 ("the fix is the ref2va checkpoint with a per-window reference
  image").
- **The song node's seed has a control widget.** The generator's comment said
  "no control widget: the node's seed input declares none"; the frontend draws
  one by name for any INT called `seed` (`useIntWidget.ts`), so the shipped
  song UI graphs loaded shifted. The node declares it now
  (`audio_freeze_song.py`, the `seed` input).
- **The song node's docstring and graph note said its first run had not
  happened.** `bench/results/2026-09-12_audio_freeze_song_smoke.jsonl` records
  one. The docstring points there now.

## 2026-09-13

- **The AWQ lane's code is deleted** (owner). The lane closed on 2026-08-27
  (`docs/roadmap.md` "Closed lanes"); today `h3_awq_encoder.py` and its
  `MiniMaxH3AWQEncoderLoader`, the `config/` W4 snapshot directories,
  `docs/h3_awq_encoder.md` and the bench tools that served them
  (`build_h3_awq_standalone.py`, `check_h3_awq_encoder.py`,
  `convert_h3_awq_candidate.py`, `capture_h3_encoder_states.py`,
  `measure_qwen_view_under_snapshot.py`, `measure_still_policy_token_cost.py`,
  `build_native_h3_calibration_batch.py`) went with it, as did
  `h3_config.ENCODER_V1` and `ENCODER_V2`. The records under `bench/results/`
  and `docs/research/` stay as history. `h3_config.MODELS["clip"]` is
  `ENCODER_INT8`, and every generated graph loads it through
  `MiniMaxH3EncoderLoader` (`workflows/build_workflows.py`, the loader
  comment). What the prose used to claim: `docs/wiki/stages.md` named the
  AWQ loader as the alternate encoder load; `docs/custom_node_gaps.md` §5.1
  said every graph wires core's `CLIPLoader` and the adapter was live code
  read by preflight and the config; `docs/h3_geometry_and_nodes.md` called it
  "the loader used by every generated graph".
- **The `encoder` option is gone from `MiniMaxH3ReferenceConditioning`**
  (owner). `image_policy` and `video_policy` each offer `comfy` (default) and
  `release` (`reference_geometry.IMAGE_POLICIES`, the conditioner's
  `define_schema`). On the shipped core-loaded encoder `encoder` always
  resolved to `comfy`, since the CLIP carried no contract, and for stills
  `release` and `comfy` produce the same geometry at every legal short edge on
  that encoder (`bench/results/2026-08-29_qwen_view_under_snapshot.json`).
  `video_policy`'s default moved from `encoder` (which ran as `comfy`) to
  `comfy`; every graph was rebuilt. What the prose used to claim:
  `docs/comfyui_vendor_gaps.md` listed `video_policy=encoder` as the "shipped
  hybrid encoder policy" and `docs/h3_references.md` said "the encoder-aware
  hybrid is now the generated default"; both had been marked dormant on
  2026-08-29 and are now marked removed.
- **`MiniMaxH3AppendRefImage` defaults are vendor parity** (owner):
  `size_policy=max`, `dit_short_edge=2048`, `allow_upscale=True`,
  `qwen_view=shared`, which is what sglang, diffusers and DiffSynth do
  (`docs/research/sglang_h3_pipeline.md` "Reference stills"): one prepared
  still feeds both the video VAE and Qwen3-VL. Read the values from the node's
  `define_schema`. Before today `allow_upscale` was off (since 2026-08-28) and
  `qwen_view` was `separate` at 512 (since 2026-08-27, on one observation,
  CHANGELOG 0.82.0). `h3_rules.REF_QWEN_SHORT_EDGE` is now only the value
  pre-filled when a user picks `separate`; `REF_VIDEO_BUDGET` still sets
  `ref_upscale=False` on the video-bearing reference arms, as an arm setting
  for memory. What the prose used to claim: `docs/evidence.md` "Reference
  sizing" called the 512 view the shipped default; `docs/custom_node_gaps.md`
  §4.1 said "three independent implementations agree, and we differ";
  `docs/h3_conditioning_end_to_end.md` said most append nodes set
  `qwen_short_edge=512`.
- **The reference-view ablation is rebuilt** (owner). The three-arm Gate 6
  family (`h3_probe_refview_{a_source,b_qwen2048,c_parity}`,
  `bench/gate6_refview_arms.json`) was priced on 2026-08-25, never rendered,
  and is deleted. The new one is five ref2va scene graphs
  (`h3_config.REFVIEW2_SCENES`, `workflows/h3_probe_refview2_*.json`) built
  at the node defaults, with six arms per scene as widget patches in
  `bench/refview2_arms.json`. Unrendered; nothing is claimed.
- **`docs/h3_input_impacts.md` pointed at `preflight.py:28`** for the int32
  crossing; the line moved, and `preflight.py::_INT32_FUSED` is the pointer now.
  The same section gains the second ceiling, the CUDA v-side `uint32` wrap in
  the sage fork's `csrc/fused/fused.cu`, from the fork's own CHANGELOG.

## 2026-09-14

- **`dense_blocks="0-1"` decided against.** The roadmap's open call since
  2026-08-16. Two instruments that share no code agree the first blocks carry
  the least error on both terms; `dense_blocks` stays `""`. Reasoning in
  `docs/roadmap.md` under the original item and `docs/SOLATTN.md`, "The
  defaults, re-read against the sage-side error records, 2026-09-14".
- **`MiniMaxH3ChannelBalance` added, off by default, an experiment.** Folds a
  per-channel q/k rebalancing into the norm weights of the blocks whose
  K-norm is lopsided (45, 48, 49 on the shipped checkpoint). No workflow
  wires it; no default changed. Why the blocks are lopsided and what the fold
  buys: `docs/h3_block49_quant_error.md`.
- **Corrected: "K offset predicts `smooth_k`'s benefit."** The sage fork's
  real-activation spike printed that inference; graded across ten cells the
  benefit tracks block depth and error magnitude, not the offset, and the
  line is removed there. `smooth_k` stays off.

## 2026-09-12

- **PDD reopened for the audio-freeze lane only** (owner: "worth testing if
  PDD works with this since it's faster iteration"). The lane was parked on
  2026-09-05 for quality work; this is iteration speed, and the documented
  PDD weakness is its audio, which a frozen track takes out of PDD's hands.
  `workflows/h3_candidate_t2v_pdd8_baked_audio_freeze.json` is the graph;
  the parked quality work stays parked.
- **Audio-freeze lane opened** (owner): a known track frozen into the target
  audio rows with a per-stream mask, on the fl2va base first, the LTX pack's
  loop geometry on H3's grid after; reference audio parked until the loop
  runs because it regenerates the track. `docs/h3_audio_freeze.md` owns it.
  Before this, the mask path was known here only as something no shipped
  graph used (2026-08-30) and the looping packs were explicitly unread
  (`custom_node_gaps.md` section 8).

## 2026-09-11

- **The frontier table in `next_steps.md` counted a dense last step.** It said
  the leader runs five sage and eleven Sol of sixteen; since `07b903c` it runs
  four and twelve (reasoned, not rendered). A note above the table says so.
- **`docs/comfy_notes.md` said Sage runs `fp16 (most accurate)`, not `auto`.**
  The config has shipped `auto` since `497b421` (2026-08-18), which scoped the
  fp16 verdict to renders without Sol; the paragraph now says so.
- **The `shown red` column is gone from `docs/checks.md`** (owner). It
  recorded which checks had been shown failing under the retired
  red-before-green rule; git history has the cells.
- **`docs/prompting.md` 5.10 rewritten around the owner's point** (owner): too
  many words in a shot make the speaker rush. It carried a second word-budget
  formula beside §3.4's and blamed the rushed delivery on the Audio VAE; it now
  budgets from speaking time, points at §3.4, and claims no mechanism.
- **`CLAUDE.md` cut to what every session acts on** (owner). Seven "Settled
  about H3" facts, the numeric-input and capture-broadly rules, and the
  `coderef/` search advice moved to `docs/evidence.md`, `VISION.md` and
  [`references.md`](references.md); nothing was withdrawn. "A rendered clip
  cannot A/B a numerical change" stayed, because code and docs cite it as
  `CLAUDE.md`'s.
- **`bench/red/` removed** (owner). The red harnesses and their shared spine
  served only the retired red-before-green rule, and nothing imported or ran
  them; git history has them.
- **The `h3-experiment` skill removed** (owner). Its two steps nothing else
  held are in `docs/comfy_notes.md` "Adding a probe or an arm".
- **Documented commands run the ComfyUI venv's python** (owner). They said
  `uv run --active --no-sync python`; the venv's own interpreter is exactly
  what the server runs and involves no uv project step.
- **Run bench scripts with `python`, not `uv run`** (owner). `docs/eval_comparison.md`
  and the `h3-prompt` skill said `uv run python bench/...`; plain `uv run` in
  this repo builds a repo-local venv and, with `VIRTUAL_ENV` set, can recreate
  the ComfyUI one, as `docs/comfy_notes.md` records.
- **`CLAUDE.md` routes to the wiki instead of carrying its tables** (owner;
  CHANGELOG 0.99.72). The "What is where" tables moved into
  [`index.md`](index.md), which is now written by hand.
  `bench/build_wiki_index.py` used to generate that page from `CLAUDE.md`; it
  now only reports documents no link reaches.
- **History notes leave `CLAUDE.md`** (owner; CHANGELOG 0.99.72). Its rule
  "when you find prose that lost, correct it and say what it used to claim"
  now logs the old claim on this page.
- **"A perceptual claim needs a distribution of seeds judged blind, never a
  pair" is withdrawn** (owner; CHANGELOG 0.99.72) from `VISION.md`,
  `CLAUDE.md`, the `h3-experiment` skill, `docs/open_experiments.md` and
  [`prompting.md`](prompting.md), under the tinkering-repo rule.
  `docs/eval_comparison.md` still describes the blind process for when one
  is wanted.
- **Where the Sol-Attn kernel comes from.** `CLAUDE.md` said it was
  "installed from comfy-kitchen main". It is built from the owner's fork by
  `vendor/rebuild_kernel.sh`: the tag ComfyUI pins plus the `blk_cnt`
  commits, enforced since `d378479`.
- **The tinkering-repo rule** (owner). `CLAUDE.md` opens with it: not
  research-grade, rigour proportional to the claim, and a default that
  sglang and ComfyUI's own node or comfy-kitchen agree on is adopted without
  waiting on our evals.
- **No check has to be shown red before it is trusted** (owner, `48e1219`).
  `docs/checks.md` "The standard" says what it used to require.
- **Closed lanes moved into the repo** (`48e1219`). They were recorded only
  in agent memory; `docs/roadmap.md` "Closed lanes" is their home.
- **The probe canvas** (`48e1219`). The `h3-experiment` skill said to bench
  canvases cheaper than 16:9 by default; probes run at 1152x768 or 1344x768
  with 345 frames, the owner's rule of 2026-08-30.
- **Sol runs through the last step** (owner, `07b903c`). `end_percent` is 1.0
  on every graph, adopting sglang's and core's default. It used to stop short
  so the last step ran dense; the retired values are in the comments beside
  `workflows/h3_config.py::SOL_END_PERCENT_BY_STEPS` and `SOL_PDD_OVERRIDES`.
