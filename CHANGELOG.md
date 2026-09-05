# Changelog

Semantic versioning. Nothing here has been tagged or published, so every
version below describes the state of the working repo rather than a release
artifact.

## 0.99.42

### Added

- **The subway probe records, the plan's step 1.** Two armed renders of
  the shipped t2v graph on the subway scene at the ladder's seed, each in
  its own server: the shipped sink mode
  (`bench/results/2026-09-04_sol_probe_base16_subway.json`, route
  `..._sol_route_base16_subway_probe.json`) and `exact_kv_and_all_rows`
  (`..._sol_probe_base16_subway_allrows.json`, route
  `..._sol_route_base16_subway_allrows_probe.json`); render rows in the
  two `2026-09-04_probe_base16_subway*.jsonl` files, raw records under
  `internal/sol_observe/`. Every invariant holds on both. Read through
  `bench/compare_sol_probe_records.py`:
  `bench/results/2026-09-04_sol_probe_subway_vs_standoff.json` (same block
  ranking as standoff, less disagreement in every in-window step and in
  the video segment, the last block the one exception, mostly its audio
  rows) and `bench/results/2026-09-04_sol_probe_subway_allrows_vs_shipped.json`
  (the text segment's disagreement collapses to the audio floor, video
  and audio segments and the ranking unchanged; the prediction written
  before the run). Read with the ladder verdict, the subway slip the owner
  saw belongs to the sage floor, which lost the same pair against dense,
  not to Sol. One seed per scene.

### Added (the morning after)

- **Two pixel controls that change what the probe records can be used
  for.** `bench/results/2026-09-04_probe_render_vs_unarmed_pixels.json`:
  the armed subway probe render is identical on both streams to the
  unarmed 2026-09-03 ladder clip of the same graph, seed and prompt, so
  arming the probe does not perturb a render and a probe render's clip is
  a valid blind sample of its configuration.
  `bench/results/2026-09-04_probe_allrows_vs_shipped_pixels.json`: the
  all-rows sink mode moves the subway render about as far as
  dense-versus-sage does at the same seed, so its blind pair (already
  rendered, no card time) is meaningful. The roadmap's 2026-09-04 section
  gains a "not established" list on the strength of both, the PDD bake
  paragraph a hypothesis to test, open experiment 6 the base-model
  step-count arm the owner raised, and `docs/comfy_notes.md` the detached
  launch every server and runner has used since the 2026-09-04 power loss.

### Changed (the morning after)

- **Branch `sol-nosage` merged** (`7b9c0ce`, peer session's work at this
  session's request): `SageChainAssert` gains `require_no_forward_patch`,
  appended last, and its exercise takes the state as an argument, so the
  Sol-over-stock graph is proved at call time rather than merely
  permitted (a probe below the gate reaches no sage kernel; one above it
  with Sol composed reaches none either). The manifest row was appended by
  `--write`; every UI graph carries the sixth widget value; merged with no
  server running, then the tree validated against the restarted server
  (empty scratch diff, schema and widget checks green).
- **The PDD ladder is rendered and blinded** as session
  `pdd_ladder_2026-09-04`: PDD8 under sage alone and PDD8 with Sol on two
  of eight steps, five scenes at the ladder's seed, against the ladder's
  sage clips as the floor and its shipped-PDD8 clips as the shipped rung
  (the re-render of that rung was cut once the retime check proved the
  post-pull output identical). Two judged arms rendered before the
  2026-09-04 power loss and eight after the 2026-09-05 reboot, so the
  record spans two cache states and says which is which; records under
  `bench/results/2026-09-04_pdd_ladder_*`. Scoring is the owner's.

### Fixed (the morning after)

- **`bench/check_literal_widgets.py` is green for the first time since
  `caa81ef`** (peer session, commit `8abd3d8`): a third named allowlist
  kind, `REFUSED_ZERO`, for a numeric widget whose only comparison against
  zero is a raising guard, with the one entry the chunked Sol node's
  `chunk_rows` earned (minimum 64, step 64, zero refused, no branch
  selected). Red-proved both ways: the entry removed, and the guard
  removed from a scratch copy so the entry reads as stale. This session's
  first reading of that widget was wrong and the peer's correction is what
  the entry records.
- **`docs/checks.md` says how to run a check while a render holds the
  card** (peer session, commit `6d2f28b`): mask the card and read exit 2
  as the expected outcome, with the equivalence check and the generator as
  the two invocations.

### Added (later the same evening)

- **`h3_probe_t2v_sol_nosage`** (peer session, commit `7d59ec9`, at this
  session's request after the owner's "maybe it's better to try without
  sage at all"): Sol as shipped over ComfyUI's stock attention with no sage
  node, so the outer steps and Sol's fallback run stock attention. The
  generator's `dense_attn` gains the named mode `sol`, and the chain
  assert's flags are derived from the chain in one place; validated
  against the live server. A stronger assert for that state
  (`require_no_forward_patch`, exercised) waits on branch `sol-nosage`
  for a restart and a validation, since the live server predates the
  input.
- **`bench/sol_nosage_arms.json`**: just Sol on the ladder's five scenes
  at its seed, Sol widened to every step but the last by two window
  patches (one stock step, fifteen routed, by the sigma-window arithmetic
  in `docs/SOLATTN.md`), subway and standoff first at the owner's ask for
  at least two scenes tonight, plus an optional subway arm at a slightly
  higher tau listed last. Judged blind against the ladder's sage and Sol
  clips per scene.

## 0.99.41

### Added

- **Token routing graded on our capture.** Comfy-Org/comfy-kitchen PR 156
  (kijai's `token_aug`) built from its head `1128df6` into a scratch wheel,
  the venv untouched, and graded on the twenty-five Base16 cells with
  `bench/measure_sol_exact_variants.py` on the 2026-09-03 footing:
  `bench/results/2026-09-04_sol_exact_base16_capture_1128df6_token_aug.json`,
  plus the isolated call at the capture's shape in
  `bench/results/2026-09-04_sol_exact_random_1128df6_token_aug_timing.json`.
  The plain arms reproduce the 2026-09-03 installed-wheel record to every
  digit (the control); the knob lowers the error on four blocks and raises
  it on block 49 at every step, and its budget barely matters.
  `docs/research/2026-09-04_sol_token_aug_grade.md` owns the reading; the
  roadmap's step 7 now points at it and the candidate moves into the block
  policy step. The grader gained `--token-aug N` (repeatable), `--limit K`
  (first run on a new arm) and `--kernel-source TEXT` (recorded verbatim
  beside the version, since the venv's build record cannot describe a
  scratch wheel). Kernel provenance follows the vendoring pattern: the sha
  is fetched into the workspace clone as `kijai/sol_token_aug_main` and the
  wheel sits in that clone's `dist/`. Nothing is installed.
- **The vendor's step schedule as a prior for the window's start.**
  `bench/results/2026-09-04_sglang_cube_topk_schedule.json` records the
  per-update keep-ratio list sglang's cube sparse attention backend (merged
  2026-09-02, MiniMax engineers co-authoring) recommends for the fifty-step
  grid, with provenance; `bench/map_cube_schedule.py` maps it by sigma onto
  the `simple` grid our graphs sample at shift 12 (built by ComfyUI's own
  scheduler, as `bench/check_pdd_sigmas.py` builds it) and writes
  `bench/results/2026-09-04_cube_schedule_on_h3_grid.json`: the vendor's
  leading dense count on our grid against the shipped window's, and the
  `start_percent` interval that reproduces it. Open experiment 27 names the
  arm; roadmap step 5 points at it. `start_percent` has never been measured
  at any value, and this is the first prior for it from anyone with the
  training data. Nothing rendered.
- **Vendor gap 17**, `docs/comfyui_vendor_gaps.md`: core's gated residual
  rounds once (`addcmul_`) where the diffusers reference and sglang's fused
  kernel round twice; the scale-shift path matches. Read, unmeasured, and
  the measurement is named beside it.
- **One dense rung re-timed on the post-2026-09-04 core**
  (`bench/results/2026-09-04_stairwell_dense_retime.jsonl`): the ladder's
  stairwell dense arm, same graph, prompt and seed, on a fresh unarmed
  server after the pull that turned on the Comfy Compiler's malloc graph.
  The sampler time sits inside the 2026-09-03 dense rows and the clip is
  pixel-identical to the 2026-09-03 one, video and audio, by the scoring
  lane's comparison (`bench/results/2026-09-04_stairwell_dense_retime_pixels.json`),
  so the regime moved neither output nor time on this rung and the
  2026-09-03 ladder clips stay valid blind references. The server log's
  "Comfy model compiler graph breaks: 2" line is the regime's own marker.

### Changed

- `docs/wiki/next_steps.md` replaces the two upstream-survey checks with
  their results; `docs/roadmap.md` step 7 carries the pointer.
- **sglang's H3 path re-read holistically** against its tip of 2026-09-05:
  `docs/research/sglang_h3_pipeline.md` section 14 records every H3-touching
  commit since the 2026-08-30 anchor by stage (FastH3 as its own pipeline,
  the three-tier quality contract, component-backend enforcement, block-FP8
  loading, the tiered AdaLN cache with its fail-closed traps, the per-block
  compression gate, per-step attention metadata, and four attention policies:
  cube sparse, VSA-H3, SpargeAttention, SubBlock on SM120), and says which
  line citations were not re-anchored. `docs/research/sglang_comparison.md`
  closes with what each item is against what we do; no earlier verdict there
  changed.

## 0.99.40

### Added

- **The PDD ladder's missing rungs**, at the owner's ask after reading the
  ladder verdict: `h3_probe_t2v_pdd8_sage` (PDD8 with sage auto on every
  step, Sol absent) and `h3_probe_t2v_pdd8_dense` (PDD8 under stock
  attention, neither kernel), generated by `workflows/build_workflows.py`
  through the existing `dense_attn` extra; the sage-alone stem is listed in
  `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS` with its reason,
  the dense one is exempt by mechanism as every reference-replication PDD
  arm is. Built with `--no-validate` on a stopped server; validation against
  a live `/object_info` is owed before the first render.
- **`bench/pdd_ladder_arms.json`**: four PDD8 rungs on the five ladder
  scenes at the ladder's seed, the two graphs above plus the shipped PDD8
  graph as shipped and with Sol's window narrowed by two patches to two of
  the eight steps (reasoned from `docs/SOLATTN.md`'s sigma-window
  arithmetic, not measured). Names the controlled pairs, the reading rules,
  the ladder's dense rows to append as the quality reference, and the
  cross-regime caveat those rows carry after the 2026-09-04 ComfyUI pull.
  Nothing has rendered.
- **The reference pathway verdict record**,
  `bench/results/2026-09-04_ref_pathway_2026-09-03_verdict.json`: the
  owner's blind scoring of the four controlled pairs of session
  `ref_pathway_2026-09-03`, joined with the sealed key after the key was
  checked against the manifest and the render rows; pairs only, so partial.
  Open experiment 26 carries the reading: can't tell on both controlled
  ablations, no identity item named, static background people named on
  every encoder-only arm. Beside it,
  `bench/results/2026-09-04_ref_pathway_encoder_arms_pixels.json` records
  that the two encoder-only arms the owner called the same are different
  renders.
- **`bench/frontier_table.py`** (peer session, commit `0d880e1`, at this
  session's request after the owner's words "it's a whole bunch of grey in
  the middle"): per scene and arm, sampler time and speed against the sage
  floor and the dense reference beside every verdict and every verbatim
  note the arm received, nothing passing or failing. First record:
  `bench/results/2026-09-04_ladder_2026-09-03_frontier.json`, the speedup
  ladder read that way. Controls in its docstring, run green here.
- **`bench/build_outputs_record.py`**: the arm-to-clip join from the
  rendering server's `/history`, in the shape of
  `bench/results/2026-09-03_ladder_outputs.json` (which was built by hand),
  so `bench/measure_clip_loudness.py` and a blind batch have it before the
  server that rendered the rows goes down. Refuses a row the server does
  not know or one without exactly one muxed clip. Its refusal path ran
  first; the join path runs on the PDD ladder's server tonight before it
  counts.
- **`bench/compare_sol_probe_records.py`** (written by a peer session at
  this session's request): two `check_sol_probe.py --json` records side by
  side, per block, per segment and per step, with the Spearman of the block
  ranking and a builds-identical flag, so the plan's first step (the subway
  probe against the standoff record) is a command. Controls in its
  docstring: self against self, shuffled, mirrored, permuted, a dropped
  block, an absolute path; re-run green here before use.
- **The cross-regime check is closed**:
  `bench/results/2026-09-04_stairwell_dense_retime_pixels.json`. The
  stairwell dense rung re-rendered by the upstream-survey session on the
  post-pull ComfyUI (compiler malloc graph on; its row is
  `bench/results/2026-09-04_stairwell_dense_retime.jsonl`) is identical to
  its 2026-09-03 clip on every decoded frame and every audio sample. The
  regime changed nothing in the output, so the 2026-09-03 ladder clips
  stay valid blind references for anything rendered after the pull, and
  the PDD ladder's pairs against them carry no caveat. The five ladder
  prompts were also re-hashed against the judged rows before tonight's
  renders (same text, under the rstrip convention `workflows/prompts.py::describe` uses).
- **`bench/compare_clip_pixels.py`**: are two rendered clips the same
  render, by decoded frames and decoded samples, never bytes. Reuses
  `bench/verify_vsa_render.py`'s mechanism (embedded graph for seed and
  node set, container ignored) and adds per-frame statistics when the
  frames differ. Written for the cross-regime question above: a dense rung
  re-rendered after the 2026-09-04 ComfyUI pull against its 2026-09-03
  clip. Controls run before first use: a clip against itself is identical
  on both streams; the ladder's stairwell dense against its sage rung at
  the same seed differs on every frame.

- **A fourth `sink_conditioning` mode on `MiniMaxH3SolAttn`,
  `exact_kv_and_all_rows`** (written by a peer session at this session's
  request on a branch, merged before any server started tonight): every
  conditioning query row runs dense, references included, because the
  kernel takes one dense-query range and text rows sit before the
  references. On t2v the extra cost over the shipped mode is the text rows
  alone; on ref2v with a video reference it is the reference's rows. The
  roadmap's step 3, the segment policy: the standoff probe record has the
  text rows disagreeing most in every block. Shipped default unchanged
  (`h3_config.SOL_CUDA_DEFAULTS`); `_sink_blocks` now refuses a mode it
  does not know; the sink pair per mode is graded on CPU in
  `bench/check_sol_node_equivalence.py` ahead of its kernel cases, red
  proved by mutation. Chosen by a patch at render time; the subway probe
  tonight measures it beside the shipped mode.

### Fixed

- **`bench/check_node_ids.py` is green again**: `bench/node_id_manifest.json`
  records the optional `require_absent` input appended to `SageChainAssert`
  on 2026-09-03 (commit `29b8967`), the one change `--write` made (peer
  session, commit `4ea4434`). It had been red at HEAD since that append.
- **`bench/check_sol_node_equivalence.py` exits 2 on a busy card** instead
  of a raw traceback: the kernel cases run under a wrapper that catches the
  allocator's error, names the cases not graded, and returns the exit its
  docstring promised; `--oom-control` proves it with the card masked, red
  proved by mutation (peer session, commit `ff3eee5`).

### Changed

- **Two owner decisions recorded in `docs/roadmap.md` as the 2026-09-04
  forward plan: sage is always on, and the routing policy is the core
  priority.** What they change and the sequence they set live there;
  `docs/wiki/next_steps.md` points at it. `bench/pdd_ladder_arms.json` is
  cut to three rungs under sage and pairs against the ladder's sage clips;
  the stock-attention PDD8 graph stays as the vendor's reference
  configuration and its generator note says it is not a rung.
- **`bench/_paths.py::comfy_output` and `comfy_input` resolve the media
  directories from the launcher's flag on the live server's command line,
  or refuse** (written by a peer session at this session's request, commit
  `d5b9dfe`): the env var first, then the port owner's `--output-directory`
  read from its command line, then a refusal naming the env var, the port
  and the local directory the old code silently fell back to. The blind
  tool, its audio repair, the loudness tool, the stacker and the capture
  manifest generator all go through it; the resolved path is a runtime
  input and is never written into a record. Controls in the module's
  docstring, including a read-only parse of a live server; re-run green
  here. The ladder's first small fix closes.
- **`bench/grade_arm_audio_spectrum.py` refuses below three clips per
  arm** instead of warning above a table of verdicts, which is how twenty
  void ladder records came to exist on 2026-09-03 (0.99.35). Shown red on a
  ladder pair (one clip per arm, exit 2 naming both arms' counts) and green
  on two four-clip arms. The 2026-09-03 session postmortem's forward item 2
  closes.

## 0.99.39

### Changed

- **Upstream survey, 2026-09-04, recorded where each answer lives.**
  `docs/sol_upstream.md` gains a dated section on Comfy-Org/comfy-kitchen:
  PR 150 merged, PR 156 (`token_aug`) open with its contract and the fact
  that our `blk_cnt` commits conflict on it, PR 146 and PR 124 still open,
  kijai's `minimax_vae` still uncalled. `docs/research/vsa/fastvideo_vsa_checkpoint.md`
  section 6 and `docs/research/vsa/vsa_node.md` blocker 5 now point at the
  FastVideo FastH3 Preview v1 release and at the two engine constants that
  pin its four-step schedule, in place of "the artifact carries no schedule"
  standing alone; the artifact still carries none.
  `docs/research/sglang_comparison.md` gains the cube sparse attention and
  VSA-H3 backends with the knob-for-knob mapping onto `MiniMaxH3SolAttn`.
  `docs/wiki/next_steps.md` carries the two checks the owner approved from
  the survey. No code changed, no default changed, nothing was measured.

## 0.99.38

### Added

- **The speedup ladder's verdict record**,
  `bench/results/2026-09-04_ladder_2026-09-03_verdict.json`: the owner's
  blind scoring of every stacked pair in session `ladder_2026-09-03`,
  joined with the sealed key by `bench/score_session.py` with the batch
  MANIFEST cross-checked, and the key's rows and labels verified against
  the render record before the join. Pairs only: no single was scored, so
  the record is marked partial and nothing audible was judged through the
  page. The two reading rules travel with it in `docs/wiki/next_steps.md`
  and a bounded row in `docs/evidence.md`: one seed per arm, and every
  contest within a scene shares one clip per rung. On reading the result
  the owner named a third, now in the record's `owner_report_after_join`
  with its provenance: the PDD8 rung carries sage and Sol like the rungs
  below it (plus the PDD-specific Sol `end_percent`), and no arm renders
  PDD8 without either, so its losses attribute to nothing narrower than the
  shipped graph. The owner's audio report is in the same field: good across
  the board, a bit quieter on the PDD clips.
- **`bench/measure_clip_loudness.py`** and its first record,
  `bench/results/2026-09-04_ladder_audio_loudness.json`: EBU R128
  integrated loudness, loudness range and true peak per ladder clip through
  ffmpeg's `ebur128`, each rung against its scene's dense rung, with a sign
  summary per rung across scenes. Descriptive at one clip per arm and says
  so; written so the owner's "quieter on the PDD ones" has a level beside
  it. Refuses a clip without an audio stream or one the record names that
  the output root does not hold; basenames only reach the record.

### Changed

- **`bench/score_session.py` reads the pair questions from `clip_rubric`
  when `pair_rubric` is absent** and writes which key it read into the
  record (`pair_rubric_read_from`). The ladder's `score.html` on the share
  was edited by hand after it was built, renaming that key in its data,
  its item loop and its export; `bench/blind_score_app.py` never emitted
  the name and still does not. Regenerate a page rather than editing it.

### Not done

- **The reference pathway session has no export.** The owner reports
  scoring `ref_pathway_2026-09-03`; no scores file reached its batch
  folder, so the key stays sealed and open experiment 26 waits on the
  export.

## 0.99.37

### Fixed

- **`bench/blind_batch.py` copied the silent file for every single, so the
  ladder's judge could not hear the rungs apart.** The 0.99.35 fix mapped a
  muxed `-audio.mp4` history entry to its silent sibling as the row's handle
  and then copied that handle; the singles are the only place a blind session
  carries audio (stacks map video only) and every one of them was deaf. A
  single now copies the muxed sibling when the combine node wrote one
  (`blind_batch.py::single_source`); the stack is still built from the silent
  file. `docs/eval_comparison.md` said the singles carry audio throughout;
  the code lost to the prose, which is the wrong way round. A batch whose
  single would carry no audio stream now refuses before anything is written
  (`has_audio_stream` on every located source), with `--silent-ok` for a
  graph that saves none; proven red on a scratch output root holding only
  the silent clips, green on the same root with the flag.
- **`ladder_2026-09-03` repaired in place** by the new
  `bench/blind_batch_add_audio.py`: each single's source is found by byte
  identity against the share (no key opened, exactly one match required, a
  refusal before any copy otherwise), replaced with its muxed sibling, and
  probed for an audio stream after. Its deliberate-violation test (a
  truncated clip with no source) refused as designed. The sealed key's
  `source` field for that session still names the silent file; the muxed one
  is that name plus `-audio`.

### Added

- **The reference pathway ablation is blinded** as session
  `ref_pathway_2026-09-03` from `bench/results/2026-09-03_ref_pathway_arms.jsonl`
  with the four controlled contests `bench/ref_pathway_arms.json` names, the
  brief derived from that manifest, and the key sealed under
  `internal/blind_keys/`. Resolved through the mtime fallback on a stopped
  server. Scoring is the owner's.

## 0.99.36

### Added

- **The first online Sol-versus-fallback record of the shipped call.** The
  probe replayed the retained Base16 capture and reproduced the offline
  exact-branch numbers on every cell
  (`bench/results/2026-09-03_probe_replay_base16.txt`), then rendered the
  shipped t2v graph on the standoff scene armed, and the record holds every
  invariant over fifty blocks. Summary as data:
  `bench/results/2026-09-03_sol_probe_base16_standoff.json`; the route the
  same render took: `bench/results/2026-09-03_sol_route_base16_standoff_probe.json`;
  what the ranking says and cannot say is the closing section of
  `docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md`. One
  scene, one seed; the roadmap's step 2 says a second scene is owed.
- `bench/check_sol_probe.py --record --json PATH` writes the per-block table
  as data beside the text table, with the header, render rows and
  violations; `block_summary` is the one source of both.

### Changed

- The two summary writers (`check_sol_probe.py --json`,
  `sol_observe_report.py --json`) redact the home directory from every
  string, so an arming spec naming a directory outside the repo does not
  put that path into a tracked record.

## 0.99.35

### Added

- **The first speedup ladder rendered: five t2va scenes by five rungs, matched
  seed, at the trained canvas and full length.** `bench/ladder_arms.json`
  rendered end to end into `bench/results/2026-09-03_ladder_arms.jsonl`;
  `2026-09-03_ladder_outputs.json` maps every arm to its clip and carries the
  per-rung timing medians. Descriptive, dated, one seed: the timings are the
  first cold-cache wall times for the true dense baseline on this box, and
  the fp16-sage rung came out slower than Sol as shipped on every scene. The
  standoff dense row is a cache hit and its timing is void; it was retimed at
  a nudged seed for the time alone
  (`bench/results/2026-09-03_ladder_arms_retime.jsonl`), and the retime sits
  inside the other four dense arms' spread.
- **The ladder is blinded for scoring** as session `ladder_2026-09-03`:
  every rung against its scene's dense baseline plus Sol against the fp16
  rung, stacked by matched seed, with the sealed key under
  `internal/blind_keys/`. `2026-09-03_ladder_arms_judged.jsonl` is the copy
  the batch was built from, with the cache-hit flag cleared on the one row
  whose clip is the arm's own earlier render, and says so on the row.
- `bench/results/2026-09-03_ladder_subtitle_timing.json`: per-clip vocal
  burst structure for all twenty-five clips, descriptive.

### Fixed

- **`bench/blind_batch.py` could not resolve a clip from a graph that saves
  only the muxed `-audio.mp4`.** It skipped that history entry and fell
  through to the mtime fallback, which refused. It now maps the audio entry
  to the silent sibling the stacker takes.

### Withdrawn before it was written

- The pairwise audio-spectrum grader reported a strong separation on every
  feature of every ladder pair. It needs at least three clips per arm to
  estimate within-arm spread, and this ladder has one, so every pair reads
  strong by construction. Those records were deleted and the outputs record
  says why; rerun it only on a multi-seed ladder.

## 0.99.34

### Fixed

- **The Hub bundle of the PDD node failed to load with
  `No module named 'pdd_observe'`.** `bench/build_sidecar_node.py` copied
  the two files its `SOURCES` tuple named, while `pdd_lora.py` had imported
  `pdd_observe` and `block_spec` at module level since `e11afd7` and
  `vendor_config` inside `expected_population` since the bake contract;
  nothing compared the tuple with the file, so the bundle regenerated at
  `72b8d42` and uploaded on 2026-09-03 carried a current stamp and three
  missing modules. The tuple is now checked against the closure of relative
  imports walked from `pdd_lora.py` (`required_sources`), in both build and
  `--check`, and refuses in either direction; the `vendor_config/` folder
  ships beside `vendor_config.py` since the population read needs it at load
  time. Red on the old tuple and on a stale extra entry before the staged
  bundle was rebuilt; the rebuilt bundle loads through ComfyUI's own
  `spec_from_file_location` path and reads the release population. The
  bundle README's notice and the staged Hub README's install step say to
  replace the broken copy. Upload is the owner's.
- **The Hub bundle's node started on the wrong file.** `shipped_pdd_loras`
  reads the two shipped names off `workflows/h3_config.py` by path to put
  the checkpoint-matching sidecar first in the combo; the bundle has no
  `workflows/`, the read returned empty, and every Hub user's fresh node
  started on the alphabetically first file, the `adaln2688` one. The bundler
  now writes `shipped_pdd_defaults.json` beside the node from `h3_config` at
  build time, the node reads that file first, and `--check` compares its
  bytes. `pdd_lora_options` matches by file name rather than path, since the
  Hub README puts the file in `models/loras/` directly and Windows spells
  the subfolder with a backslash; the entry offered is the population's own
  spelling.
- **`schedule_knots` cast the sampler's sigmas to float64 on their own
  device.** `torch.as_tensor` keeps a tensor's device, and at run time
  `sample_sigmas` is on the model's device, which on a Mac is MPS and has no
  float64. The same hazard in `fuse_block` was closed in `72b8d42` and this
  one was named in the same review and missed. Computed on the CPU now; the
  result is `steps + 1` Python ints. A CUDA float32 and bf16 sigma tensor
  give the same knots as the list, and `check_pdd_sigmas.py` is green; MPS
  itself is not reachable from this box.

### Added

- **`build_sidecar_node.py` load-tests the bundle** in both modes: imports
  the built folder in a subprocess exactly as `nodes.load_custom_node`
  does, with the ComfyUI checkout beside the pack on the path, and compares
  the node list, the release population and the shipped defaults read back
  off the loaded module with what the source says. Exit 2 without that
  checkout. Red on a bundle missing a module and on a defaults file naming
  the wrong file, before the staged bundle was rebuilt.
- **`build_sidecar_examples.py` validates every example graph against the
  live schemas before writing**, and gained `--check`. The widget names,
  combo values and links in it are typed by hand and nothing had compared
  them with the nodes since the file was written; the owner asked whether
  they were still aligned with the PDD node and nobody could say. A
  subprocess loads core ComfyUI with its bundled extras and the STAGED node
  bundle beside `--out`, then refuses an input the loaded node lacks, a
  missing required input, a literal combo value off that combo's options
  (file combos need only a filename), a link whose source output type is not
  the input's type, and a class neither core nor the bundle registers. Red
  means nothing is written. Six deliberate breaks each red, an untouched
  graph green; the first revision recognised combos by `isinstance` against
  `io.Combo`, which `io.Combo.Input` is not, and checked no combo on any v3
  node until the red proof said so. The four staged graphs validate and
  match the generator byte for byte, so the upload of 2026-09-03 stands.

## 0.99.33

### Changed

- **`MiniMaxH3ReferenceConditioning` takes both VAEs as optional**, mirroring
  core's `MiniMaxH3ReferenceToVideo` after ComfyUI PR 16065 (commit
  `1aec3a13`, merged 2026-09-03). A reference whose VAE is unwired reaches
  the text encoder exactly as before -- same label, same Qwen view -- and
  contributes no reference-latent rows to the DiT; with no rows at all the
  `minimax_refs` key is absent rather than empty, which is the door core
  uses. Core's gates are reproduced, not improved: a sounded video loses
  its whole block without the video VAE and keeps its `<Audio>` label as a
  silent `video` block without the audio VAE; a standalone audio reference
  without the audio VAE is a bare label, since the encoder is never handed
  audio. `bench/check_reference_runtime.py::encoder_only_references_skip_the_dit_rows`
  holds the three cells, and went red on the previous node before the
  change. `docs/h3_references.md` section "Encoder-only references" is the
  authority. Every reference UI graph regenerated: the two VAE sockets now
  carry the optional marker and nothing else moved.

### Added

- **The reference pathway ablation**: `h3_probe_ref_pathway_{typed_both,
  typed_encoder, native_both, native_encoder, fl2va_encoder}` from the
  generator's new `ref_latents` and `native_ref` knobs (`native_ref` emits
  core's node with its zero-indexed autogrow sockets, stills only, API
  form only via the new `api_only` spec flag). `bench/ref_pathway_arms.json`
  is the manifest and names which pairs are controlled; the typed-versus-
  native both-pathway pair is not, since the two nodes size stills
  differently. Nothing has rendered: the server was mid-render for another
  session and predates both the core commit and this node change, so the
  graphs were validated against a cached `/object_info` with the two
  inputs moved to optional, which is what a restarted server serves.
  Later the same evening, on a server restarted onto this commit: all five
  arms rendered once at the manifest's seed, one warmup each for the two
  typed arms, and every clip decodes with its full frame and audio count.
  `bench/results/2026-09-03_ref_pathway_arms.jsonl` is the record; its two
  `typed_both` rows ran on a different server process than the rest, since
  the first run was killed by hand mid-batch (the owner took a silent first
  step for a hang) and resumed after a restart with its own warmup. The
  encoder-only rows sit below the both-pathway rows on sampler time in
  every family, as the missing reference rows predict. Nothing has been
  judged: the blind pairing in the manifest is the owner's.
- `bench/preflight_graph.py` prices an encoder-only conditioner as zero DiT
  reference rows and says why, instead of pricing rows the graph will
  never build.

## 0.99.32

### Added

- **The Sol-versus-fallback probe is implemented** (`sol_block_probe.py`,
  a stub since 2026-08-16). Armed by `H3_SOL_PROBE` on the server process,
  it runs the chained fallback -- the shipped sage override on the
  canonical graphs -- on the identical q/k/v of every call Sol takes,
  records Sol against it (whole-call relative L2 and cosine, difference and
  reference RMS, per-head numerators, denominators, relative L2 and cosine,
  per-segment aggregates over the authoritative segment table, per-head and
  per-segment row distributions as count, mean, p50, p90, p99, max; a zero
  denominator is null with both sums kept), and returns Sol's output under
  `trajectory=sol` or the fallback's under `trajectory=sage`. Calls Sol did
  not take are recorded as skips with their reason. It writes no tensors,
  streams per head so its temporaries are one head's worth, confirms the
  counterfactual went through sage by the fork's dispatch counter and
  reports `reference_not_sage` otherwise, and joins the route record by
  prompt id, schedule index and block. Unarmed it is one cached boolean at
  each of the two seams in `sol_attn_h3.py`. `bench/check_sol_probe.py`
  carries the eighteen fixture controls (all green on 2026-09-03), the
  record validator and per-block table, and the capture replay that proves
  the metric against the exact-branch records; the replay and the first
  canonical Base16 record are pending the card.

## 0.99.31

### Fixed

- **PDD head fusion ran its float64 arithmetic on the model's device, which
  Apple's MPS backend cannot do** -- a Mac user hit the failing line on
  2026-09-03. `pdd_math.py` now fuses on the CPU in float64 and moves only
  the float32 result to wherever the bank lives; the block slice is a few
  MiB, and the result is bit-identical to the previous path on CPU and
  CUDA (checked on every block of a 32-head stack). The device-split
  failure the docstring records from `2af7f0b` is closed the same way,
  since both operands are CPU by construction again. PDD checks green.
- **The Hub bundle was a month behind the node.** `bench/build_sidecar_node.py`
  regenerated `comfyui_minimax_h3_pdd/` in the staged Hub copy from this
  tree; the Hub's copy predates the converter-v3 sidecars and the node's
  backbone-identity contract, so a user pairing the Hub's node with a new
  sidecar, or the reverse, gets a refusal that reads as breakage. The
  upload is the owner's; the staged copy and its comparison are ready.

## 0.99.30

### Changed

- **The composed prompts are bank entries too, so `derived:` scenes are
  gone from the catalogue.** 0.99.28 moved every prompt LITERAL into
  `prompt_bank/` and said prompt text had one home; seventeen prompts were
  not literals and stayed outside it. `_ref_prompt()` builds a ref2va
  prompt from the role tables so it declares exactly the labels its arm
  wires, and `fl2v_prompt()`/`l2v_prompt()` resolve a duration into their
  Part One line -- so no constant held their text,
  `bench/build_prompt_catalogue.py` could only name them
  `derived:<graph>` after a graph that carried them, and a render record
  could not identify what it rendered. All seventeen are now
  `prompt_bank/<id>.txt` with manifest entries, taken VERBATIM from the
  graphs: no prompt's text changed, and every graph regenerated
  byte-identical, which is the proof of that. The catalogue names a scene
  by its bank id when no constant holds the text, so it lists no `derived:`
  scene; the prefix survives for text matching neither, which now means a
  hand-edited graph or a parametric prompt at a length its entry does not
  declare.
- **The composition stays; it now ends at the bank.** `_composed_from_bank`
  looks the composed text up by `prompts.identify` and ships the bank's
  copy, refusing the build otherwise -- so a new reference combination
  fails naming the file to write rather than emitting a graph carrying text
  nothing can identify. The role tables still decide what the text SAYS.
- **The two keyframe defaults stay parametric, and that is the one
  exception.** Their alignment sentence carries the effective duration to
  two decimals (`base_en.md:14-32`), so the text is a function of the frame
  count; typing it into the bank would be right for one length and silently
  wrong for every other. `_retimed_from_bank` takes the whole prompt from
  the bank at the frame count its entry declares, swaps in the same
  sentence resolved at the graph's length, and asserts the two agree at the
  declared count -- so editing either half alone fails the build.
  `fl2v_prompt(192)` returns the same body under an 8.00-second Part One
  line; all three refusals were fired deliberately before this was trusted.
- **`docs/prompt_audit.md`'s seventeen `derived:` keys are renamed to bank
  ids, verdicts unchanged**, with a note at the head of the table saying
  what each used to be, because a reader who remembers `derived:h3_ref_video_edit`
  needs to know it reads `ref2va_video_garment_edit` rather than that its
  verdict was dropped. The catch-all "the remaining `derived:` ref2va
  scenes" row is now the finding rather than a key; every scene resolves to
  a row of its own. `check_prompt_docs_sync.py` still binds each catalogue
  scene to a verdict.
- **Fourteen of the seventeen carry `recorded_findings`.** Ten carry the ref
  §5.2 word-budget WARN the audit already verdicts as systematic; five name
  no camera motion in `MOTION_PROSE`'s forms because they follow the
  reference video's camera or write it as handheld reframing, which is the
  bank's own reading convention rather than a guide rule (one carries both).
  None was edited to make it grade: a prompt change is a content change and
  belongs in the audit. Zero FAIL across all seventeen.

### Fixed

- **`bench/build_prompt_bank.py` suppressed a shape problem in silence.**
  An entry with `recorded_findings` whose only finding was a shape problem
  printed nothing at all -- no FAIL, no WARN, no `noted` line -- because
  that line fired only on a grader finding. The composed edit arms are
  exactly that case, so the defect arrived with them -- but it had already
  escaped: `t2va_subway_platform`'s entry names a suppressed shape problem
  in its own `recorded_findings` and the run said nothing about it. The
  `noted` line now counts suppressed shape problems too.
- **`workflows/build_workflows.py` could not run from a git worktree.**
  `_ref_short_edge` reached ComfyUI as `HERE.parent.parent.parent`, which
  is the checkout only when the pack sits directly in `custom_nodes/`; from
  a worktree it lands three levels short and the generator dies on
  `ModuleNotFoundError: comfy_extras` before writing a graph. It now finds
  the root by walking up for `comfy_extras/nodes_minimax_h3.py`, which
  returns the same directory in the ordinary layout, and falls through to
  the import when the walk finds nothing -- a caller that has already
  imported core, as `bench/check_generator_constants.py` does, needs no
  path work and must not be refused.

## 0.99.29

### Added

- **`docs/roadmap.md` has a 2026-09-03 forward plan**: the goal in the
  owner's words, what the day established and what it did not, the
  emerging architecture as a hypothesis, the seven-step sequence agreed
  with Codex, and what would count as finding it. The 2026-08-24 plan it
  replaces described an encoder lane that closed on 2026-08-27.
  `sol_block_probe.py`'s docstring carries the online instrument's full
  per-cell specification and its required controls, so the scaffold and
  the spec cannot drift apart.

### Corrected

- **Three overclaims in 0.99.27, on Codex's review, withdrawn in place
  there** (block 49 "on both builds" and "no `dense_blocks` list touches
  it"; the `tau_1.0` arm called the shipped call; "one footing" for a sage
  record that scored different rows against a different reference). The
  grader now carries `sage_<mode>` arms -- every sage mode on the same
  q/k/v, every row, the same fp32 reference -- and both Base16 records were
  regenerated with them; the tau arm is `tau_1.0_no_sinks` and says why.
  The direction the matched arms give lives in the research record's
  addendum, next to the retraction.

### Fixed

- **`bench/generate_capture_manifest.py` described a capture it had not
  read.** Its first manifest of the Base16 capture claimed a 1024x768
  canvas for a 1344x768 render, no text encoder, no seed, and a token
  total that disagreed with the files by tens of thousands of rows: four
  reads had met one implementation (the ref3 capture graphs) and none of
  this pack's current nodes, and text and audio rows were typed constants
  under a docstring promising none. Canvas now comes from
  `MiniMaxH3Resolution`'s real inputs, the encoder from
  `MiniMaxH3EncoderLoader`, the seed from `RandomNoise`, audio rows from
  the core helper preflight uses, the sequence length from the files, and
  text as the labelled remainder; the generator refuses to write when that
  remainder goes negative. Schema 1.5.0 adds the bank id and prompt hash,
  the workflow and graph hashes, and each tensor's sigma, kernel, render
  and segments (read through a memory map, which also took the run from
  minutes to seconds per file). `bench/check_capture_manifest.py` accepts
  1.5.0, requires the new keys from it, and allows an empty `references`
  when the accounting agrees there were none -- a text-to-video capture is
  a state, not a defect. `docs/capture_manifest_schema.md` has the record.
- **Three checker holes closed on Codex's second review, same day**: the
  recorded tensor sha256 is recomputed under `--verify-hashes` (and the
  checker prints on every run whether it was); `prompt_sha256` must hash
  `full_prompt_text` and `bank_id` must be what that text identifies as;
  the generator refuses mixed server stamps across a directory and each
  tensor carries its `server_pid`. Labels say their source
  (`*_quantization_source: filename`), token accounting says its method
  (text is the remainder; `segments_recorded` says whether the split is
  proven), and the capture writer's server stamp carries the parsed
  ComfyUI namespace and the sageattention build identity.
- **`bench/check_capture_manifest_controls.py`**: the seven red controls for
  the contract, on megabyte fixtures driving the real generator and
  checker -- one stamp and all-null legacy pass; mixed stamps, a stamped
  record beside a legacy one, a flipped tensor byte, edited prompt text
  and a wrong bank id are refused. Green means every violation was
  caught, not that nothing was tried.
- **Captures are transient and have a lifecycle now** (owner's call: the
  disk is not unlimited). `start.sh` exports `H3_CAPTURE_ROOT`, the
  collection every capture and both capture tools use; each capture
  carries `retention.json` (purpose, `keep_until`); and
  `bench/recycle_captures.py` lists what each capture still owes and
  deletes tensors only when the repo holds its manifest copy and inventory
  record, leaving the manifest, the retention note and a `DELETED.json`.
  The delete is a recoverable transition, on Codex's review of the first
  version (which wrote the marker before the unlinks, so a crash read as
  done and could not be retried): `DELETING.json` with the planned files
  first, a nonzero exit and a resumable state on any failed unlink, an
  atomic rename to `DELETED.json` only when nothing planned remains;
  `--self-test` forces one unlink to fail and then resumes. Then crash-safe
  as well as retryable, on the same review: the state file is written
  atomically (tmp, fsync, replace), a resumed plan is validated to bare
  tensor basenames, planned sizes are recorded so bytes freed survive a
  crash between unlink and update, and a rescan before the rename keeps a
  capture INCOMPLETE if a tensor appeared after the plan; the self-test
  covers a tampered plan and a late file.
- **One path to which comfy-kitchen is running.** `vendor/rebuild_kernel.sh`
  writes a build record beside the venv at install (version, commit,
  branch, source checkout, wheel, arch, time), and `start.sh` prints it on
  every launch and cross-checks it against the wheel the venv holds:
  a stock wheel with no local tag, a record that disagrees with the venv,
  or a source checkout that no longer has the commit each get a line.
  The record for the build installed on 2026-09-01 was reconstructed by
  hand once and says so; the next rebuild replaces it.
  The Base16 capture moved under the root with a week's retention tied to
  validating the online instrument against its cells. Two 2026-08-30
  captures already under the root carry no manifest and are reported red
  by the collection check until their owner writes one or recycles them.
- **The manifest names the clip.** `workload.render` carries the render's
  prompt id (stamped into every record by `h3_capture.py` from now on) and
  the output BASENAMES, joined from the live server's history when it can
  be and from the operator when it cannot, labelled either way; the mp4
  itself stays in the output folder `start.sh` names.
- **The Base16 capture has provenance now**:
  `bench/results/2026-09-03_capture_manifest_base16.json` (the manifest,
  names only) and `2026-09-03_capture_inventory_base16.json` (the
  inventory-then-delete record). The capture itself stays outside the repo
  for the moment; both records survive its deletion. The graph it rendered
  is the bench t2v graph as of `4b9c85f`, before the pair changed scene.

## 0.99.28

### Changed

- **Prompt text has one home: `prompt_bank/`.** Until now the generator
  held every shipped prompt as a string literal, the bank held a second
  population that had never rendered, and no record of a render said what
  was rendered beyond the workflow file -- which is how every
  real-activation Sol number came from one scene without anyone noticing.
  Owner's call, 2026-09-03: the seventeen literals moved to
  `prompt_bank/<id>.txt` with manifest entries, `workflows/build_workflows.py`
  loads them by id through the new `workflows/prompts.py`, and the constant
  NAMES stay because five bench scripts and `prompt_audit.md`'s scene keys
  bind to them. Regenerating every graph changed nothing but the bench t2v
  pair (below), which is the proof the move was a move. The bank builder
  now derives a `ships` column from the graphs (the catalogue's scanner, so
  no second copy of what ships where) and an `adapt` column, mechanical:
  a prompt naming no cut time and no duration can take another length from
  the graph alone. Two shipped prompts do not meet the bank's own bar and
  `prompt_audit.md` already says so; the manifest's `recorded_findings`
  carries that adjudication, the table shows the grade, and the check does
  not gate on them. The catalogue names scenes by resolving the bank call
  as it resolved the literal, plus a `bank id` column.
- **What was rendered goes into every record.** `workflows/prompts.py::describe`
  reads an API graph for bank id (or the full text when the prompt is not
  in the bank), prompt sha256, length, canvas and seed;
  `bench/run_graph_arms.py` rows and the Sol route record's render row
  carry it as `rendered`. `h3_capture.py` records are unchanged: they see
  tensors, not graphs, and the route record joins them by prompt id.
- **The bench t2v pair renders `t2va_frontier_standoff`**, not the covered
  market: a bank scene with a figure at distance, a painted sign, dialogue
  and a silent bystander. Records that name
  `h3_text_to_video_stamped_api` carry the graph hash, so pre-change rows
  still say which prompt they were.

### Added

- **`workflows/bench/h3_text_to_video_dense_stamped_api.json`, the true
  baseline**: the DiT and encoder every graph loads, ComfyUI's stock
  attention, the base step count, no LoRA -- no sage node, no Sol node;
  `SageChainAssert` stays in warn-only mode with nothing to require, the
  shape the PDD reference arms use. The render you would otherwise make on
  this box. Every "dense baseline" number before this was sage alone
  (`h3_text_to_video_stamped_api`, kept beside it on the same scene, seed
  and length as the matched pair). Bench graphs are outside
  `check_attention_defaults`' scope, so no exemption entry.
- **`SageChainAssert.require_absent`, the inverse control.** A graph that
  patches attention not at all -- this baseline and the PDD reference arms
  -- carried the assert node with nothing required and warn-only, and it
  logged "override installed" over an empty chain (the prose session
  caught it in review). Now `sage=False` sets `require_absent=True` and
  `warn_only=False`: the render raises if any attention override or
  per-block forward patch is installed, so the baseline proves it is one.
  Appended last in the schema so saved graphs keep their widget indices;
  every generated graph changed by that one input. The no-requirement log
  line says nothing was required instead of claiming a verdict. Proved
  red and green on the dense arm at short length before commit.

## 0.99.27

### Added

- **`bench/measure_sol_exact_variants.py --capture DIR`**: the exact-branch
  comparison on REAL activations. Grades every `qkv_*.pt` a capture wrote
  on the installed comfy-kitchen build against an fp32 dense reference the
  script computes per head in query chunks (no S x S allocation), in three
  arms: every block routed (kernel arithmetic alone), `topk_ratio` 0.10
  (kijai's "10% keep", per-head cosine mean and worst) and the shipped
  `tau=1.0`. Every arm also carries per-ROW relative L2 and cosine, the
  statistic `bench/grade_sage_on_capture.py` reports, so Sol and the shipped
  Sage fallback read on one footing. Metrics accumulate per head because the
  first run OOMed on a whole-tensor fp32 copy with the server resident.
- **A Base16 dense-trajectory capture and its two records**,
  `bench/results/2026-09-03_sol_exact_base16_capture_{d25f2e8,24908e1}.json`:
  the shipped text-to-video bench graph (pruned INT8 fl2va, 1344x768, 345
  frames, 16 steps, Sage, no Sol) captured at blocks 0, 24, 32, 40, 49 and
  steps 4, 8, 12, 14, 15, then replayed through both exact-branch variants.
  The capture is retained outside the repo for now and is NOT inventoried;
  `bench/results/2026-08-30_capture_inventory.json` is the pattern when it
  goes. **It is one scene** (the covered-market prompt every shipped t2v
  graph carries) and a dense-trajectory control, not the production
  trajectory: `docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md`
  is the design record and says what that does and does not license.
- **`sol-blk-cnt-0.2.32`** in the workspace comfy-kitchen clone: the three
  direct-`sol_attn` `blk_cnt` commits rebased onto Comfy-Org v0.2.32, wheel
  built to a scratch target (never installed), blk_cnt tests green. The
  upstream PR candidate; also cherry-picks clean onto kijai's PR #150 head.
  Not pushed.

### Measured

- **kijai's per-block P quantisation (Comfy-Org/comfy-kitchen#150, the same
  exact-branch change our installed `d25f2e8` carries) is a large win on
  real H3 activations and a wash on random ones**, which settles the
  2026-09-01 disagreement in the inputs' favour. With every block routed the
  running-max build (`24908e1`) sits several times further from fp32 than
  the block-max build on all 25 cells, mean relative L2 0.0503 against
  0.0134 (kijai's own capture: 0.0196 against 0.0140), and block 0 moves
  from under half a percent to several percent. At `tau=1.0` WITHOUT sink
  ranges (the record's `tau_1.0_no_sinks` arm; see the correction below)
  the gap is small (whole-tensor cosine 0.9893 against 0.9911) because
  routing error dominates the exact branch's arithmetic. Records above; the
  random-input records from 2026-09-01 stand as what they are.
- **Corrected the same day, on Codex's review of this entry.** Three
  claims above the line were overclaims and are withdrawn: (1) "block 49
  carries about four times any other block's error on both builds ... and
  no `dense_blocks` list touches it" -- true only on the new build and only
  under the norm-weighted whole-tensor metric (per row, block 49 is barely
  above block 40; on the old build block 40 is the worse one), and a
  `dense_blocks` entry sends the block to sage, bypassing Sol's exact stage
  entirely, so it does touch it. What stands: block 49 is a large
  exact-stage hotspot on the new build under the whole-tensor metric, in a
  few high-norm rows, and this five-block record does not say whether
  making it dense is worthwhile. (2) The `tau_1.0` arm was called "the
  shipped tau"; it is the shipped tau but not the shipped CALL, because the
  node passes sink ranges from the segment table and a capture taken with
  Sol absent has no table. Renamed `tau_1.0_no_sinks`: an unsunk
  diagnostic, not a bound on the shipped call's error, since relative L2
  and cosine are not monotone in the sink ranges the way forced-pair
  counts are (errors can cancel). (3) "Sol against the shipped
  fallback on the same cells ... on one footing" -- the first sage record
  (`2026-09-03_sage_on_base16_capture.json`, kept) scored 512 sampled rows
  against float64 while Sol was scored on every row against fp32; the
  statistic matched, the sample and reference did not. The grader now
  runs every sage mode on the same q/k/v, every row, the same fp32
  reference (`sage_<mode>` arms), and both records were regenerated with
  them; the direction those arms give is stated in the research record's
  addendum, not here, so this entry cannot go stale twice.

### Fixed

- **`vendor/rebuild_kernel.sh` broke on any checkout based on comfy-kitchen
  v0.2.32**: its version tag was a diff hardcoded against `0.2.31`
  (`vendor/patches/001-local-version-tag.patch`), found by the first build
  of the rebased branch. The script now rewrites whatever version the
  checkout declares with sed and reverts it on every exit path; the patch is
  gone and `docs/SOLATTN.md`'s provenance row says so.
- **`h3_capture.py` logged every write as `wrote v`**: the shape-check loop
  rebound the filename variable. Files were always named correctly; only
  the log lied. Found on this capture.

## 0.99.26

### Added

- **The three repo skills are pure routers now, with a review point.** Every
  sentence that restated a mechanism (what a probe row contains, the prompt
  rule layers, the base-versus-reference trap, how the blind key is handled)
  is replaced by a pointer to the file that owns it; two of those
  restatements had already drifted from `docs/prompting.md`, which names four
  layers, not the five the skill listed. Each `SKILL.md` carries
  `reviewed: <commit>`, and `bench/check_skill_routes.py` reports every named
  file that changed after that commit without the skill changing with it
  (`--strict` fails), so a home moving under a skill is visible without
  anyone remembering to look. Shown red before landing. The declared-absent
  entry for the withdrawn 2026-09-01 routing is gone with the note that named
  it.

- **The three repo skills re-pointed after the prompt bank became the one
  home of prompt text.** `h3-prompt` routed edits into the generator's
  constants; it now routes to `prompt_bank/` and `docs/prompt_bank.md`.
  `h3-experiment` names `SOL_EXEMPT_STEMS` instead of "the exceptions
  `CLAUDE.md` lists" (the cut file lists none), points at the baseline
  `VISION.md` defines, and drops the dead-pointer history note that
  `bench/check_skill_routes.py` and this changelog already record.
  `bench/list_prose_measurements.py` now governs `.claude/skills` and
  `VISION.md` as prose.

- **`CLAUDE.md` cut to its operative form**: one line per rule, the settled
  H3 facts as one-liners each naming their record, the routing tables at one
  line per row, and the operative rules that have no other home. The file it
  replaced is frozen verbatim as `docs/rules_history.md`, which owns the dated
  instances behind every rule. `VISION.md` is the tenets, condensed to nine.
  Three claims the old file made were found stale in the cut and are not
  carried: its Sol-Attn exemption list (the constant names stems it did not),
  `internal/PROMPTING.md` as "being retired" (it is deleted), and a pinned
  ComfyUI checkout hash.
- **`docs/evidence.md`** gained the settled-H3 subsection under "These hold",
  and two withdrawals: the vendored-node row (the node is a read-only
  reference since 2026-08-30) and "`LONG_LENGTH` is now 362" (the window is
  `h3_rules.MAX_LENGTH`, still 362; the shipped default `LONG_LENGTH` is 345
  by the owner's preference, and the two answer different questions).

- **Prose carries pointers, records carry numbers.** `docs/prose_measurements.md`
  states the rule (a measurement in prose is an uninvalidated copy; the
  sentence keeps the direction and points at the script, record or constant
  that owns the value), the three kinds of number and what each gets, the
  record set that is exempt, and a tiered plan for prose written before it --
  with the explicit case where no origin can be found and the fix is to mark
  the number unsupported rather than invent a pointer or re-measure.
  `CLAUDE.md`'s Guiding Principles carry the short form.
- **`bench/list_prose_measurements.py`** inventories unit-bearing numbers
  (multipliers, percentages, sizes, times, rates, measured counts) in governed
  prose, skipping code spans, identifiers and the dated-record set named in
  its `RECORD_PATTERNS`. A report that exits 0, not a gate; `--file` lists the
  hits in one doc and `--json` writes a dated record.
  `bench/results/2026-09-03_prose_measurements_baseline.json` is the backlog
  at adoption.
- **`VISION.md`**: the tenets this repo holds itself to, one bold line each
  under truth, measurement, controls and structure, with no instances, dates
  or numbers. `CLAUDE.md` is its operative form and routes to it, so the wiki
  index derives the row. `docs/prose_measurements.md` gained a section on
  what the rule means for the wiki: nothing to migrate, the index inherits
  `CLAUDE.md`'s blurbs, and a generated records page is deferred until the
  migration has produced the links it would be built from.

## 0.99.25

### Added

- **`docs/pdd_artifacts.md` is the master inventory of PDD weights.** Hand-
  written glossary (including the two meanings of "baked": the adaln update
  solved into the curve basis, which every `_comfy` file has, and the
  backbone folded into a checkpoint, which does not exist yet), the
  which-file-on-which-checkpoint decision table with what the node does on
  each wrong pairing, a converter changelog by version and commit, and a
  dated artifact changelog. `bench/pdd_artifact_inventory.py` fills the
  generated region between two markers from each file's own metadata
  (converter version, the new `h3_pdd_converted_on` and
  `h3_pdd_converter_commit` stamps, backbone kind, adaln form, probe origin,
  what it loads on) plus which `h3_config` constant names it; `--check` goes
  red when the region and the folder disagree.
  - **Same day, made inspectable rather than narrated.** `--record` writes a
    dated fingerprint record (`bench/results/2026-09-03_pdd_artifact_fingerprints.json`:
    per file, metadata, a content hash by the 2026-08-28 reproducibility
    recipe, and per tensor group dtype, shape, count and sha256, every
    `h3_pdd.*` sidecar tensor hashed individually). The page's sidecar table
    is generated from it, with the `introduced` commit and date read from
    `git log -S` over the converter and the reader functions from the node's
    AST; the version-to-version diff (archived v1 against current, and the
    two forms against each other) is computed as groups added, removed,
    changed and byte-identical. `--check` also requires a provenance heading
    for every sidecar family the files carry, every commit the page names to
    exist, and the files on disk to match the record. **Withdrawn on the
    way**: the page and this file dated the retirement of the pre-fused
    `h3_pdd.head.*` tensors to 2026-08-27; git puts it in `548629e` on
    2026-08-26. Earned by the
  owner asking which file goes with which base while `docs/h3_pdd.md`
  carried a hand-typed table whose sizes had gone stale that same day. The
  loras folder is organised to match: the two dated 2026-08-28 copies moved
  to `pdd_archive/` with `v1` in their names (one saved UI graph names the
  ref2va one; nothing in the repo does), the two `adaln2688` files were
  regenerated at converter version 3 with every prior tensor bit-identical,
  and the alibaba-pai sources and Kijai's conversions stay where the dated
  comparison records cite them. No graph changed: every shipped PDD graph
  already wires the current `_comfy` file for its partition.

### Corrected

- **The PDD bake contract from 2026-09-02 proved "not the base", not "this
  exact bake", and let a short population define its own size; both are
  fixed on Codex's 2026-09-03 audit.** The probe now stores `blocks.49.mlp.fc2`'s
  int8 codes AND fp32 row scales from the ONE checkpoint a file is paired
  with -- the base for a full sidecar, the exact bake (`--omit-backbone
  --baked <ckpt>`, now required) for a stripped one -- and `MiniMaxH3PDDLoRA`
  requires equality on both (`probe_match`, `check_backbone_identity`),
  refusing a full file on any other checkpoint, a stripped file on the base
  (named as such) or on any other bake, and a scale-only or codes-only
  difference by name. The population is asserted against the release's
  declared depth, `vendor_config.transformer_depth()` from the two
  `transformer/config.json` files vendored today, as sets of `(block, kind)`
  in both the converter (`assert_population`, `strip_backbone`) and the node
  (`check_file_population`, `check_stripped_targets`). Converter version 3;
  both shipped `_comfy` files regenerated with every prior tensor
  bit-identical; the two staged stripped files and their `h3_config`
  constants removed, since none can be cut before a bake exists.
  `bench/check_pdd_sidecar_contract.py` carries every red control the audit
  named. Also from the audit: `bench/analyze_pdd_unmerge_curve.py` stamps
  the source records' measurement date rather than the run date, so a rerun
  no longer diffs a tracked record, and labels its legacy RTN curve as the
  hypothetical it is; a stale `docs/SOLATTN.md` sentence said `SOL_PDD_CUDA`
  is "now `0-2,32`".

### Added

- **The PDD bake contract, built: stripped sidecars, a backbone probe, and
  the node-side refusals.** `bench/convert_pdd_lora.py` is converter version
  2: it stores 64 rows of `blocks.49.mlp.fc2.weight` from the checkpoint the
  file loads on as `h3_pdd.backbone_probe`, and `--omit-backbone` emits the
  stripped sidecar (every `diffusion_model.blocks.*` LoRA tensor dropped
  after the self-checks, refiner/adaln/heads/tables kept, `h3_pdd_backbone`
  and `h3_pdd_backbone_strength_baked` in metadata). `MiniMaxH3PDDLoRA`
  compares the probe with the loaded module before patching and refuses
  both mismatches that render normally: a full sidecar on a checkpoint that
  is not its base (the backbone applied twice) and a stripped sidecar on
  the unbaked base (never applied). A stripped file also refuses any
  `strength` other than its bake's and any `unmerged_blocks`, and asserts
  after `load_lora` that no backbone target resolved and the refiner's did.
  Both shipped `_comfy` files were regenerated at version 2 with every prior
  tensor bit-identical; the stripped files sit beside them as
  `PDD_*_STRIPPED_LORA`, wired by no graph because no baked checkpoint
  exists yet. `bench/check_pdd_sidecar_contract.py` grades every refusal and
  went red on its first run, on the strip's own shape assertion. The
  converter also stops emitting 50 inert `h3_pdd.adaln.blocks.N.alpha`
  tensors on the `--pruned` path.

### Changed

- **`bench/restart_comfy.sh` is disabled, owner's call: it hung on every
  run.** It now exits 2 with a pointer and carries its original body
  commented out, so `ARMING_KEYS` stays readable. Every place that told
  someone to run it (`CLAUDE.md`, `docs/comfy_notes.md`,
  `bench/run_marker_arms.sh`, `sol_observe.py`, `docs/open_experiments.md`,
  `docs/research/pdd/tier1_gate.md`) now points at the manual recipe in
  `docs/comfy_notes.md`: kill the port owner after reading its environ for
  `H3_*` arming keys, launch `start.sh`, poll, and confirm the new start time
  postdates the file you changed.

- **`dense_blocks="0-2,32"` is demoted from every shipped Sol workflow back
  to the experiment it was.** Both `SOL_RECOMMENDED_CUDA` and
  `SOL_PDD_CUDA` now inherit the node's empty default, and all 156 governed
  graphs were regenerated from that source. The 2026-08-29 propagation run
  sampled 11 of 50 blocks on one base-model trajectory at one specially
  isolated sigma; it did not measure the PDD head, canonical PDD active
  sigmas, multi-block interactions, or watched output. That record motivates
  a future arm and does not justify silently routing four blocks to Sage in
  every production graph. Explicit block lists remain supported.

### Corrected

- **The PDD `unmerged_blocks=0-49` stored-weight estimate was an RTN hybrid.**
  The dated handoff said `1.0236x` even though the shipped merge uses seeded
  stochastic rounding. The executable aggregation now reads the node's literal
  three-kind reach set and the full stochastic record: `mlp.fc2` remains
  merged, leaving `1.096307x` base error and recovering 76.2% of the merge gap.

- **A stripped PDD sidecar loads on the current node, and an earlier
  0.99.25 entry said it could not.** That entry claimed `pdd_lora.py`'s
  "matched no module" guard refuses an empty backbone and had to move to
  asserting the refiner count. The guard's input is every key under
  `diffusion_model.`, which includes the refiner's
  `diffusion_model.token_refiner.blocks.*` keys and, on the pruned base, the
  baked adaln diffs the node inserts before `load_lora`, so it cannot fire on
  a stripped sidecar. The claim was two reads of the guard's line, neither of
  which read the line building its input; the refuting tensor counts were in
  the same bullet. `docs/h3_pdd.md` and the 2026-08-31 handoff now say what
  the guard sees, keep the counts (600 tensors under `blocks.*` for 200
  modules, 24 under `token_refiner.*` for 8; the `backbone_modules: 208`
  metadata counts modules including the refiner), and name the assertion a
  stripped sidecar actually needs: no backbone target matched, and the
  checkpoint is the baked one.

### Measured

- **The no-dense PDD8 arm ran at the production 1344x768 / 345-frame
  geometry.** All 50 blocks routed through Sol on each of the four active
  evaluations; the other four evaluations carried 50 composed Sage rows each.
  The structural grader passed all 400 calls and verified all 200 raw count
  pointers. Pair-weighted ordering-effect density averaged 0.207625; the four
  formerly hidden blocks averaged 0.207809 (0), 0.200175 (1), 0.208215 (2),
  and 0.232876 (32). These are route-cost observations, not a quality ranking.
  The ignored record and provenance are under
  `internal/sol_observe/2026-09-02_pdd8_dense_none/`.

## 0.99.21

### Added

- **The marker render, built and deferred.** `bench/marker_arms.json` holds
  seven arms of one shipped t2va graph -- prompt and length patches only, base
  texts from `prompt_bank/` -- covering both spellings, both spacings and both
  split-line forms of the two markers this repo and the sister engine write
  differently, plus the arms that isolate the tag from the split and the
  spelling from the spacing. Every variant is an asserted single substitution,
  graded clean, and every patched graph passed preflight.
  `bench/marker_arms_brief.md` is the judge's brief. The owner chose the
  trained canvas, two seeds per arm and all seven arms, then deferred the run
  before anything was queued; `docs/open_experiments.md` #24 and
  `docs/prompt_audit.md` item 4 record that, with the commands, so it is not
  forgotten. Nothing has rendered.

## 0.99.24

### Added

- **`MiniMaxH3SolChunked`: Sol-Attn fed from chunks of the QKV projection,
  so Q, K and V are never built.** comfy-kitchen's `sol_attn_chunked` as
  H3's attention forward for the calls Sol takes, published through the
  `sol_take_forward` delegate the Sol gate already prefers; declined calls
  still go to Sage, dense blocks still go through the override. The
  producer applies the same fused norm-and-rope core applies, so the
  per-chunk arithmetic is core's own; it thresholds on the previous step's
  key statistics, so output and counts are close to the direct path rather
  than identical. `bench/check_sol_chunked.py` grades it against the direct
  path on the same weights: cosine 0.99995 on the second call and 0.99994
  on a ragged first call, peak allocation over a 32k-token call 87 MiB
  against 192, routing agreement on 99% of query blocks, and the gate,
  recording and non-H3 refusal paths. **The canonical-geometry A/B answered
  `docs/open_experiments.md` #25 against the node**: on the shipped PDD graph
  the process peak (11.68 GiB by the end of the first forward, which runs on
  Sage outside the Sol window) never sat in Sol's attention call, so there
  was nothing for the producer to lower, and its first call raised the peak
  by about a gigabyte (12.87 GiB); and it routes a different block set at
  the same density, agreeing with the direct path on 23 percent of (head,
  query block) pairs at the first in-window forward and 11 percent by the
  fourth. It stays as an experiment node, not a default. Records under
  `internal/sol_observe/2026-09-01_chunked_ab/`, summaries in
  `bench/results/2026-09-01_sol_route_pdd8_{direct_ab,chunked_ab,direct_vs_chunked}.json`.
- `sol_attn_chunked` gains the same optional `blk_cnt` out-parameter on the
  comfy-kitchen branch, so a chunked call records like a direct one
  (`sol_chunked` route, `path: chunked_delegate`).
- `peak_alloc_bytes` on every Sol record row: the allocator's high-water
  mark, read for free, so a memory lever can be graded from the record.
- `bench/measure_sol_exact_variants.py` and two records: the exact-branch
  metrics of the two candidate kernels at the same seeded inputs. Kijai's
  `sol_attn_continued` change (per-block probability scaling in the exact
  stage) measures as a wash against upstream main on random inputs, 0.00909
  against 0.00922 relative error at all-routed, identical cosine at tau 1.0,
  identical kernel time; both already sit under the bound his own test
  sets. The branch carrying it plus `blk_cnt` is `sol-blk-cnt-continued` in
  the workspace clone; the installed wheel is now built from it
  (`0.2.31+sol.d25f2e8`) because it is a superset, not because it measured
  better.

## 0.99.23

### Added

- **The Sol route record names the workflow it ran under.** A `render` row
  per prompt id: the running graph read from the server's queue, hashed as
  `provenance.py` hashes a graph, matched by hash to the shipped file under
  `workflows/` (null with a reason when the submission was modified), with a
  summary of PDD LoRA and step count, UNET, sampler, scheduler, canvas and
  length, and `process_render_index` so a cold first render and a warm
  repeat are told apart. Asked for by the owner on the first live capture,
  where two graphs shared one file and only their Sol windows differed.
- `bench/control_blk_cnt_public_api.py` and
  `bench/results/2026-09-01_blk_cnt_public_api_controls.json`: the positive
  and negative `blk_cnt` controls through the public API on the installed
  CUDA wheel, on a five-block hand-written fixture that neither the observer
  check nor the upstream tests use. Codex owed this pair after its review
  process lost CUDA access; run here on the owner's word.
- `bench/grade_sol_record.py`: the structural grader for a live record
  (every forward carries blocks 0-49 once; routes match the window and the
  dense list; counts, raw pointers and CRCs on `sol` rows only; one prompt
  id, known to `/history`). Expected refiner and probe row counts are
  parameters, because the first record proved both depend on the graph and
  the run.
- **First live records**, under `internal/sol_observe/` (gitignored): the
  canonical PDD graph as shipped, 400 rows over eight forwards, graded
  clean; the canonical 16-step base graph in its own file, 800 rows over
  sixteen forwards, eleven inside the window; a warm repeat of the PDD graph
  in that same process; each directory with a `provenance.md` explaining the
  header's dirty marker and what else the file holds. **Tracked summaries**
  from `bench/sol_observe_report.py --json` and
  `bench/compare_sol_records.py --json`:
  `bench/results/2026-09-01_sol_route_{pdd8_cold,base16_cold,pdd8_warm,pdd8_cold_vs_warm}.json`.
  The last one is the cache-state control: cold and warm PDD raw counts
  bitwise identical on every paired Sol call.
- `docs/research/pdd/`: the README's Sol coverage figures marked as
  observed, `queued_arms.md`'s `min_tokens` mechanism refined (no
  sampler-time refiner call on the shipped graphs), `tier1_gate.md` item 10
  (choose capture cells from the route record), and a 2026-09-01 addendum
  on the 2026-08-31 handoff: render both bake arms armed and compare their
  routing before attributing anything to the bake.

### Fixed

- **`bench/restart_comfy.sh` killed the shell that invoked it** whenever
  that shell's command line contained the literal `main.py --output`: its
  `pkill -f` matched every argv, ancestors included. Now `pkill -A`, which
  excludes the script's own ancestors; proved on a decoy process from a
  caller whose argv matched.
- **The chain assert's live probe ran on a fresh thread without ComfyUI's
  executing context**, so an armed observer recorded it with no prompt id.
  The thread now runs under a copied context. The first live record shows
  the defect: two probe rows labelled `no_executing_context`.
- `bench/grade_sol_record.py` imported the observer as a bare module; it
  goes through `bench/_live_sol.py` like every other bench script.

### Notes

- The shipped t2v graphs hand the sampler already-projected text, so a
  forward has NO sampler-time refiner attention calls; the "two refiner rows
  per forward" expectation in the review notes was a source read of a path
  these graphs do not take. The grader takes the count as a parameter.

## 0.99.22

### Fixed

- **Three record-semantics defects in the first Sol telemetry revision, all
  from Codex's review before any live capture existed.** Outside Sol's sigma
  window the canonical graph's 50 DiT calls run on Sage's per-block forward
  patch and never reach the override, so they were absent from the record;
  the composition gate now writes them as `route: composed_patch` with its
  verdict leading the reason, block label and all. `routed_density.mean` was
  presented as `bench/analyze_routing.py`'s number and is not: it weighs
  every query block equally, while that script weighs pairs; the record now
  carries the pair-weighted `ordering_effect_density` (overall, per head,
  per segment) beside the query-weighted distribution, and the docs say
  which is which. `forced.sink` was the minimum of the forced vector, which
  is an edge diagonal when there is no sink; the sink cardinality and the
  diagonal contribution are now computed from their own definitions. Three
  check cases pin each, including a five-block fixture where the two
  weightings read 0.5667 and 0.5833. No workflow, default or PDD file
  changed.
- **An undefined adaptive distribution serialised as `{"weighting": "query"}`
  rather than null** (Codex's follow-up review): the weighting stamp was
  applied to an empty dict. Stamped only on a defined distribution now, and
  a control with every query block inside `sink_q` asserts the null
  contract for the whole-call, per-head and per-segment adaptive figures.

## 0.99.20

### Added

- **`prompt_bank/`: a tracked, graded H3 prompt bank, and `bench/build_prompt_bank.py`
  to keep it honest.** One plain-text prompt per file and a manifest naming each
  one's mode, frame count and, for ref2va, the shipped donor graph whose sockets
  its labels are graded against. The builder re-grades every prompt through
  `grade_prompt_text.grade_text`, the function the CLI now also runs, and
  derives `docs/prompt_bank.md`: a table per prompt and coverage tables against
  the guides' closed sets -- the frame grid, base 4.3's motion rows and
  modifiers, 4.2's cut phrasings, 4.1's styles, ref 3's task types, ref 4's
  markers, the declared H3 markers, the language tags and the speech shapes.
  `--check` fails on any FAIL or WARN, any manifest/directory disagreement, a
  count off the grid, a missing mode, or a stale document. Every closed set is
  exercised in full except `keyframe completion`, which no shipped ref2va graph
  can carry, and the file says so rather than a list somewhere else.

  What the bank is and is not is stated at the top of the generated file:
  house-authored, mechanically conformant, unrendered, not the vendor's five
  attested examples and not a substitute for them. Two house choices every
  prompt makes are stated as arguable -- `<scenetrans>` never written,
  `<|cutoff|>` piped and tight against `</d>` -- because the sister engine makes
  the opposite call on both and neither side has rendered either.

  Twenty of the prompts are the internal bank from earlier today, ported with
  one addressee renamed from a bare name to a visible description and the one
  mouth-cue formula it repeated eighteen times varied the way the vendor's own
  example varies it. Twenty-two are new: the missing frame counts, every motion
  row the first bank left out, all five cut phrasings and the two request-only
  transitions, an off-screen speaker who is not a voiceover, three speakers in
  one scene, silence requested by the user, diegetic music kept out of the
  score field, the caption pair and the lyrics pair each once as the house
  patterns they are, six languages inside `<d>`, a storyboard `<Picture N>`
  with its own retention line, a garment transferred by `attribute_transfer`, a
  lyric that lives only in a `fully_copy` soundtrack and takes no speaker id,
  reused speech lip-synced with an `[unclear]` span, and a five-label chain
  where the soundtrack is `<Audio 1>` and the standalone clip `<Audio 2>`.

  Earned under the no-new-check rule: the first bank's hand-typed coverage
  table was wrong the day it was written, and its prompts sat where nothing
  re-graded them while the grader changed twice that day. Red-proved on a
  marker moved inside `<d>` and on a stray file, mutation confirmed by `cmp`.

### Changed

- **`bench/grade_prompt_text.py` exposes `grade_text()`.** The CLI's donor
  selection, prompt injection, length override and grading moved into one
  function that `main()` and the bank builder both call, so the bank is graded
  by exactly what an author grades by. CLI output and exit codes unchanged.
- **`docs/prompt_audit.md`** items 4, 6 and 7 name the bank prompts that serve
  them, and a coverage-map paragraph points at the derived tables instead of
  restating them. `docs/prompting.md` section 10, `docs/checks.md` and
  `CLAUDE.md`'s routing table point at the bank.

## 0.99.19

### Added

- **Live Sol-Attn route telemetry.** `sol_observe.py`, armed by
  `H3_SOL_OBSERVE="dir=...[,raw=0]"` in the server's environment, writes one
  JSONL row per attention override call -- every route, not only Sol -- with
  call-time identity (`prompt_id` from ComfyUI's executing context, the
  conditioning uuids, sigma and a justified schedule index), the block,
  shape, selection, sinks, and for Sol calls the routed-block counts the
  kernel actually produced: kernel density (forced pairs included), adaptive
  density (forced pairs removed, `sink_q` rows excluded), per head, and
  overlap-weighted per query segment. A uint16 raw sidecar keeps the full
  `(B, H, NQ)` tensor per call. The producer asserts its own shape and aborts
  the render on a violation rather than thinning the file. Until now every
  routed-density figure here was an offline approximation.
- **`blk_cnt` out-parameter on `comfy_kitchen.sol_attn`**, on the
  `sol-blk-cnt` branch of the workspace comfy-kitchen clone (two commits past
  upstream main `c1c6751`): the CUDA backend copies the plan's `cnt` slot
  from the same launch that produced the output, the eager reference fills it
  from its own route mask, HIP refuses it. The installed wheel is now
  `0.2.31+sol.24908e1`; `bench/check_sol_kernel.py` prints which build
  answers.
- `bench/check_sol_observe.py`, nine cases; `H3_SOL_OBSERVE` in
  `bench/restart_comfy.sh::ARMING_KEYS`; one more case in
  `bench/check_sol_node_equivalence.py`.
- `bench/sol_observe_report.py`, a reader: per (prompt, step) routes and
  block span, a block-by-step adaptive-density table, per-segment means, and
  `--join <server>` to ask `/history` whether each recorded prompt id is
  known -- the live acceptance the uncontrolled row waits on. Asserts
  nothing; refuses to summarise past an `error` row unless told to.

### Fixed

- **`sol_block` was never cleared after a DiT block**, so the next step's two
  token-refiner calls inherited block 49's label; with 49 in `dense_blocks`
  they were counted as `dense_block`. A paired post-hook now removes it, and
  `h3_segments` is dropped with the two spans in the outer forward's
  `finally`. Output-neutral: those calls were dense either way.
- The block-index hooks are installed whenever the observer is armed, not
  only when `dense_blocks` or a tau profile is set; a canonical graph would
  otherwise have recorded no block identity.
- `bench/_live_sol.py` re-executed a member the node had already imported,
  producing two module objects under one name. Found by the new check's
  first run.
- `provenance.py` said ComfyUI does not expose `prompt_id` to nodes. It does,
  through `comfy_execution.utils.get_executing_context()`; the sentence is
  withdrawn, the hash join stays.

### Notes

- The count includes the sink range and the diagonal; upstream tests pin
  that in closed form at both tau extremes, elementwise monotonicity in tau,
  the top-k lower bound and a tie fixture showing no `k+3` upper bound
  exists. The CUDA-vs-eager disagreement at an ordinary tau is reported, not
  gated: a bound chosen after seeing the number would be decoration.
- The live `/history` join is uncontrolled until an armed render runs;
  `docs/checks.md` carries the row.

## 0.99.18

### Fixed

- **`docs/prompting.md` §15 is back.** `a52999d` rewrote §14.5 to say the retired
  internal file's content "is §15" and, in the same commit and without saying
  so, removed the whole of §15 -- six subsections migrated and re-derived that
  morning. The manual's last heading was 14 while §14.5, the wiki page and
  `docs/portable/h3_system_prompt.md` all routed readers to §15. Restored
  verbatim from the parent commit, with a note at its head saying so. Found by
  the sister engine's session reading this tree at `b3823c5` as a wrong section
  number; it was a missing section.
- **The addressing rule was in both derived extracts and not in the manual.**
  Say who a line is spoken to, outside `<d>`, and never give a listener a
  speaker id. Now §5.9, with its layers -- the slot is base §4.4 *stated*,
  naming the addressee is ref §5.4 *shown*, the listener rule follows from base
  §4.4's no-id-for-non-vocalisers -- and a §11 ledger row saying it is checked
  by nothing. Same lag §5.8 records for singing, and the same finder.
- **Three §11 ledger rows and one wiki row said camera vocabulary is checked by
  nothing.** `bench/check_camera_vocabulary.py` has graded every shipped
  prompt's amplitude and speed red/green since 2026-08-28 and reports known-bad
  motion phrases as warnings; §13 said so and the ledger did not. The rows now
  name the check and what it cannot see: a novel out-of-table motion phrase is
  caught by neither case.

### Changed

- **`docs/prompt_audit.md` item 4 covers both marker disagreements.** The
  split-line render now also renders a truncated line with the guide's
  `<cutoff>` and the declared `<|cutoff|>`, because the sister engine writes the
  guide spellings and this repo the declared ones, neither side has rendered
  either, and neither side's gate can see the difference. Both repos proposed
  the same render independently on 2026-09-01.

### Notes

- Two of the three findings came from the sister engine reading this tree; the
  third grew when checked. The prose-lost-to-code rule at the top of `CLAUDE.md`
  applied within a single commit here: the sentence saying the content was in
  §15 and the deletion of §15 share an author and a timestamp.

## 0.99.17

### Fixed

- **`docs/portable/h3_system_prompt.md` claimed no separate lyrics tag exists,
  and contradicted itself five lines later.** The sentence was unscoped and
  tagged `[guide]`, so it asserted the guides state an absence; what is true is
  that neither guide *names* one, while the release does declare
  `<|lyrics_start|>` / `<|lyrics_end|>` -- which the same file says a few lines
  down. Now "Neither guide names a separate lyrics tag." The portable page and
  `docs/prompting.md` §5.8 already carried the scoped form and are unchanged.

### Notes

- Caught from a downstream consumer's correction to their own copy, not from
  reading ours. The distinction is the same one this repo already applies to
  guide numbers: a claim about what a document says is not a claim about what
  exists.

## 0.99.16

### Fixed

- **Both portable artifacts forbade something this repo ships.** They said not
  to use `<|lyrics_start|>` / `<|lyrics_end|>`, while four shipped graphs
  (`h3_ref2v_scene_kitchen`, `h3_ref2v_scene_subway`, both forms) emit that
  pair deliberately as marker arms -- they were the first graphs to carry a
  marker other than `<d>`. Both now record the pair as OPEN, matching
  `docs/prompting.md` §5.8: the `<d>` block is the only form the guides state
  so default to it, the pair is emitted on purpose in marker experiments, and
  nothing has been rendered and judged either way. Neither a licence nor a ban.
- The artifact was republished and the dated snapshot re-frozen a third time,
  with `snapshots.json` and the snapshot's own banner recording why.

### Notes

- Found by checking a peer's absolution rather than accepting it. The peer
  reported the proposed wording as the only defect and the shipped files as
  clean; the shipped files carried it too, in both artifacts. A second reader
  being wrong in your favour is still a claim to verify.

## 0.99.15

### Added

- **Four guide-stated rules the portable standard was missing**, in
  `docs/portable/h3_prompt_standard.html`: sung lines use the dialogue block
  rather than a lyrics tag (ref-en states it, and the `<|lyrics_start|>` pair
  the release declares is named by neither guide, so the page says not to use
  it); saying who a line is spoken to, with the listener taking no speaker id
  because ids belong to voices; the four sanctioned phrasings for a line
  crossing a cut, with the deliberate house divergence on the `<scenetrans>`
  token; and the exact-words and punctuation rules for reused or reperformed
  audio, including `[unclear]` for spans that cannot be made out.
- The same rules in `docs/portable/h3_system_prompt.md`, which is where they
  were written first.

### Changed

- Republished the artifact to the same URL and re-froze
  `2026-09-01_h3_prompt_standard.html` against the new source, with
  `snapshots.json` hashes updated and the second replacement stated in the
  snapshot's own banner rather than made silently.

### Notes

- The gaps were found by comparing the page's rule headings against the system
  prompt's, not by reading either. That is the third time on this date that a
  comparison found something a re-read had not.
- Turns-per-shot was checked and was already correct -- a first pass reported it
  stale by matching on a phrase and missing the adjacent Owner rule that
  supersedes it.

## 0.99.14

### Added

- **A `Singing` section in `docs/portable/h3_system_prompt.md`, and the marker
  rule that goes with it.** The file had no form for a sung line at all, while
  the guide's own heading is "Speakers, Dialogue, and Singing" -- a stated
  topic, missing entirely. Written from the guides: ref-en states "Write
  dialogue and lyrics as `<d>[Language] ...</d>`", so lyrics use the dialogue
  block and the `<|lyrics_start|>` / `<|lyrics_end|>` pair the release declares
  is named by neither guide and is not to be used. Also adds base-en's four
  sanctioned continuity phrasings for a line crossing a cut, and ref-en's
  punctuation rules for reused or reperformed words.
- **An addressing rule.** Naming who a line is spoken to goes in the action
  outside `<d>`, which is the slot base-en 4.4 states, and ref-en shows it once.
  A listener never takes a speaker id -- ids belong to voices, and giving one to
  someone who is only listening creates a vocal source the clip has to fill.
  How reliably a model follows an addressing cue is unmeasured and marked so.

### Changed

- **The turns-per-shot rule, which had gone stale against the owner's verdict.**
  It said "at most one dialogue turn per shot"; that is what the vendor examples
  show, not what the guide states, and a dense exchange has since been rendered
  and judged good. It now says the vendor shows one, the guide states no limit,
  more is a sanctioned capability, and past one turn you are beyond what the
  vendor demonstrates. Carries the intra-shot ordering caveat: a cut timestamp
  is the only hard temporal anchor, so turns sharing a shot are ordered by prose
  alone.
- Five outputs written to the file's rules now grade 0 FAIL through
  `bench/grade_prompt_text.py`, up from three. The scope caveat is unchanged and
  still the honest one: the grader covers the guide's STATED mechanical rules
  and is silent on everything tagged GUIDE-SHOWN or HOUSE, which is most of the
  file.

### Fixed

- A duplicated `<|cutoff|>` instruction in the same file, stated in both the
  dialogue and the continuity sections.

## 0.99.13

### Removed

- **`internal/PROMPTING.md` and `internal/official_prompt_guides/`, on the
  owner's call.** The manual was superseded 2026-08-28 and its content migrated;
  the guide copies were byte-identical duplicates of `vendor_guides/` that
  nothing read after the tracked move, and two identical copies is the drift
  this repo spends its time on. Two historical citations in code comments now
  say the file was retired rather than pointing at nothing.

### Changed

- **Turns per shot is CLOSED, by the owner, on the render.** A shipped
  base-format scene stacks four dialogue turns in one shot against a vendor
  practice of one, and the rendered result is judged good -- so stacked turns
  are a sanctioned capability, not a divergence to correct. One render settles
  it because "does the model deliver this at all" is presence/absence; it does
  NOT establish that stacking beats cutting, which needs matched seeds and a
  distribution, and nobody has run that.
- **The camera-vocabulary ruling now lives in the checker**, not three files
  away, so its three standing warnings are met with their adjudication instead
  of being re-litigated.

### Added

- **`bench/check_skill_routes.py`.** Every path a skill names must exist,
  because a skill is an entry point and a dead route sends an agent nowhere
  while it believes it is following the repo. Red-proved both ways.

## 0.99.12

### Added

- **`.claude/skills/h3-prompt`** -- the agent entry point for prompt work,
  routing by TASK (write one, edit a shipped one, convert between modes, judge
  one) to the file that owns each answer and the command that verifies the
  result. Restates no rule, so it cannot become a second authority.

### Fixed

- **`h3-experiment` routed prompt-writing to a directory that does not exist.**
  Its step 4 named `internal/2026-8-20-system-prompts/` -- gitignored and
  absent -- and neither shipped skill named `docs/prompting.md` at all. So an
  agent asking where to go to write a prompt was sent nowhere, while the
  manual sat unreferenced by any entry point.

### Removed

- **`bench/check_uncontrolled_claims.py`, built and deleted the same day.** Its
  escaped instance was real, but correcting the two documents was the fix and
  the check was not: across `docs/`, 132 lines claim nothing guards something
  and exactly one cites a numbered guide section -- the row already corrected.
  It scanned zero subjects and printed "this run proves nothing" every run.
  Widening it to the rest needs judgement about prose, which means false reds.
  A check that cannot fail is not a cheap safety net; it is a green light
  nobody earned. Retirement and reasoning recorded in `docs/checks.md`.

## 0.99.11

### Fixed

- **The derived page was checked and the document it derives FROM was not.**
  `check_portable_standard.py` guarded the published extract's quotations while
  `docs/prompting.md` -- the file this repo calls its single source of truth --
  was guarded by nothing. Deliberately corrupting the manual's FL2VA Part One
  template and its camera vocabulary was caught by no check in the repo.
  Renamed to `check_prompt_docs_sync.py` and generalised: it now grades the
  manual's quotations against the same sources, and asserts that
  `prompt_audit.md` still carries a verdict for every scene
  `prompt_catalogue.md` generates -- a hand-maintained key coupling that was
  how the audit came to cover a minority of the catalogue unnoticed.
- **Three defects in the new check, all found by red-proving it rather than by
  running it.** Its first version could not fail on two of its five coverages.
  Scene coverage used a substring test, so renaming `T2V_RAIL_LONG` to
  `T2V_RAIL_LONGG` still passed. Camera vocabulary did the same, so
  `Pedestal Up` matched inside `Pedestal Upward`. And the Part One check asked
  only whether SOME instance matched, which stays green while a corrupted
  instance sits beside a correct one -- the likely shape of a real typo. All
  three now word-bounded and per-instance.
- **And one false positive, corrected before it shipped.** Grading every Part
  One line rejected the manual's UNRESOLVED templates -- `Shot N`, `S.SS`,
  printed exactly as the guide prints them -- which are correct documentation.
  Both forms are now accepted; only a line matching neither is drift.

## 0.99.10

### Added

- **A portable prompt standard, published and checked.**
  `docs/portable/h3_prompt_standard.html` is a self-contained extract of the
  prompting rules for readers outside this repo -- every rule tagged with
  whether the vendor states it, shows it, or we inferred it; the per-mode
  structure; the operational answers on speaker IDs, dialogue markers, style
  and camera; two graded examples; nine failure patterns. It cites no
  repository path, so it survives being read where ours do not resolve.
- **`bench/check_prompt_docs_sync.py`.** That page is a second copy of rules
  owned elsewhere, and this repo's documented failure mode is a second copy
  with nothing to invalidate it. A note saying "regenerate rather than edit" is
  not invalidation, so every QUOTATION on the page is verified against its
  source: the three Part One templates against the guide-parsed constants, the
  camera vocabulary against the guide's own table, every worked example against
  `prompting.md` section 10 verbatim, each quoted guide sentence against the
  guide. Prose is not checkable and is not checked.

### Fixed

- **The portable standard had drifted before it was checked once**, which is
  the escaped instance the check above cites. Its T2VA example carried
  `non_diegetic_music: N/A` where the manual carries a real cue -- reworded in
  hand transcription, into the exact habit the same page warns against. Seven
  of the twelve camera motion types had lost half their name to abbreviation
  (`Zoom In / Out` for the guide's `Zoom In / Zoom Out`), in the section
  readers copy from most. The camera cell is now generated from the guide.
- **Two general lessons that lost their home** when a third-party pack's name
  was removed from the tracked tree are restated without naming anyone: hash
  every file you rely on rather than one representative of them, since a bundle
  can carry one file verbatim and a modified copy of another; and difflib's
  autojunk heuristic overstates a prose diff past 200 lines, which
  `unified_diff` cannot disable.

## 0.99.9

### Fixed

- **`preflight_graph.grade` could not see past the first shot of a multi-shot
  prompt, and graded keyframe alignment against the wrong shot number.** The
  shot pattern's body group was greedy to end of line, and the guides put every
  shot in ONE unbroken paragraph -- so `findall` returned a single pair however
  many shots the prompt had, `shots[-1][0]` was always `"1"`, and
  `_expected_base_alignment` demanded `from Shot 1`. **A correct two-shot FL2VA
  prompt FAILED and an incorrect one naming `Shot 1` PASSED**, an exact
  inversion. It reached no shipped graph: every shipped keyframe prompt is one
  shot and the t2va path returns before reading shots. Grading of all shipped
  graphs is byte-identical after the fix, and the newly-live branch is
  red-proved.
- **`bench/build_wiki_index.py` silently dropped any CLAUDE.md routing row
  naming more than one document**, so `docs/prompt_catalogue.md` and
  `docs/prompt_audit.md` were absent from the generated wiki -- and absent from
  its unreachable report too, since other documents link them. It was the only
  such row.
- **`docs/prompt_catalogue.md` was eight scenes stale.** Regenerated; the six
  `T2V_*` description-length and predictability arms and both `h3_ref2v_scene_*`
  marker arms now appear.
- **`docs/prompt_audit.md` labelled `I2V_PROMPT` as fl2va.** It is i2va.

### Added

- **`bench/grade_prompt_text.py`** -- grade a candidate prompt TEXT for a chosen
  mode without building a graph. Wraps the text in a shipped graph of that mode
  and runs `preflight_graph.grade`; adds no rules of its own. Donors are
  discovered from `h3_config.graph_paths` and selected by socket, and it raises
  rather than falling back to a near neighbour when a mode has none. `--length`
  grades at the duration an example is written for; `--like` names a donor whose
  reference sockets match the prompt's labels.
- **Twelve worked examples**, four each for I2VA, FL2VA and L2VA, taking every
  keyframe mode from one specimen to five. All seventeen examples in
  `docs/prompting.md` section 10 now grade clean at their declared durations,
  which until now was an assertion nothing checked.
- **`docs/wiki/prompting.md`** -- the wiki's prompting entry point: the five
  sources ranked by what a violation means, who owns which question, the
  per-mode example index, and how to grade a draft.
- **`docs/prompting.md` sections 14 and 15** -- a reconciliation across every
  source that claims to govern a prompt, and the model/presentation material
  migrated out of the superseded `internal/PROMPTING.md`.
- **Verdicts for the seventeen scenes the audit had never covered**, plus two
  misalignments recorded rather than fixed: our split on shot line breaks, and
  the shipped dialogue default carrying four turns in one shot against our own
  house rule.

### Changed

- **The vendor's own prompt-writing skill was cross-checked and adds no rule.**
  Its bundled guides are byte-identical (SHA-256) to
  `internal/official_prompt_guides/`, its two copies are identical to each
  other, and its `SKILL.md` defers entirely to them. Recorded so the next
  session that finds it does not read it as a fourth authority.

## 0.99.8

### Retracted

- **The day's `+11.6%` headline is measured in a rounding regime the shipped
  path does not use. The number is `+40%`.** `bench/measure_pdd_quant_interaction.py`
  rounds to nearest and its record says so — *"deterministic; the shipped path
  uses seeded stochastic rounding, so these are its expectation"*. **Round-to-nearest
  is the expectation of the WEIGHT under stochastic rounding, not of the
  ERROR**: `E[Q_s(x)] = x` is why the deterministic-merge lever was withdrawn,
  and `E‖Q_s(x) − x‖ > ‖Q_rtn(x) − x‖` is the √2 the same day measured. A
  distance statistic cannot borrow the first identity.
  - All 200 int8 modules, `bench/measure_merge_rounding_regimes.py`: merged
    under RTN **1.1164x** the base error, merged under the shipped stochastic
    path **1.4046x**, worse on **200 of 200** modules, the gap **3.48x** larger.
  - **The arithmetic was never in doubt.** A second implementation reproduces
    `e_shipped`, `e_patched` and `e_baked_from_release` **bit-identically**, and
    the two RTN bodies of code agree elementwise to 0.0. Only the regime label
    was wrong, and it sat in the record's own `rounding` field where it read as
    a disclosure rather than a claim.
- **"A bake pins strength AND the partition" is withdrawn. It pins strength.**
  The partition never reaches the backbone: `emit_steps`, `widths` and
  `block_w` feed only the `SIGMAS` output, `_StepTracker` and `_FusedHeads`,
  and the head bank is fused per span at run time. `bench/convert_pdd_lora.py`
  had already RETIRED its `h3_pdd.head.*` payload on 2026-08-27 because *"that
  pinned a step count into the artifact"* — so the claim described a file
  format this repo had deliberately removed four days earlier.
  - That was the whole cost of baking, and it is why `unmerged_blocks` was
    framed as the lever to reach for. One baked artifact serves every step
    count.

### Added

- **`bench/measure_merge_rounding_regimes.py`** — merged-under-RTN, merged-under-
  the-shipped-seed and baked, over all 200 int8 modules against the BF16
  release. Carries two controls: two independent RTN implementations that must
  agree elementwise, and a **cross-backend** check that meets the other
  implementation of the shipped quantiser.
- **`bench/measure_bake_realisation.py`** — `realised_along_d` for the same
  three arms, each against its OWN no-LoRA baseline. Exists because
  `e_baked ≈ e_shipped` is equally consistent with "the rounding kept the
  delta" and "the rounding threw it away and landed back on the base", and this
  lane has already had one ranking reversed by exactly that question.

### Fixed

- **`_StepTracker.check_shift`'s error named a node the reader cannot find.**
  With `MiniMaxH3SigmaShift` dropped from the PDD graphs (`44374a4`), the
  `graph_shift is None` branch went from unreachable-on-anything-shipped to the
  only path those graphs take, and its message still said *"this graph runs
  MiniMaxH3SigmaShift at ..."*. Split on which branch produced the number; both
  exercised. No behaviour change. `CLAUDE.md`'s "which code paths were dead
  before the fix and are live after", met on somebody else's fix.

### Measured

- **The offline bake wins on BOTH statistics rather than trading them**, which
  is the first thing in this lane that is not a denominator argument. It lands
  at the base checkpoint's own stored-weight error to seven digits
  (`1.0000019x`) AND realises the update at `1.0000` (min `0.9986`, all 200
  modules) — where RTN merging buys its lower distance by discarding, at
  `0.341` mean realised and `0.0043` on its worst module.
  - The mechanism: a merge starts from `W_q`, already ON the int8 grid, so a
    sub-step delta rounds back to the same codes. A bake starts from
    `W_release`, OFF the grid, so the delta shifts where the rounding lands and
    survives in the codes.
- **Only 200 weights in the DiT are int8, and they are all `blocks.*`
  linears.** `token_refiner.*` is BF16 and `blocks.N.adaln_proj.linear` is F16,
  so neither requantises. The 200-module population every record in this lane
  measures is the WHOLE affected set, not a sample of the file's 208 backbone
  modules.
- **A stochastic-rounding figure measured on CPU is not the draw a GPU load
  makes.** `comfy_kitchen`'s registry resolves `quantize_int8_convrot_weight`
  to the eager implementation on CPU and a CUDA one on GPU; on the same seed
  they draw different noise (about a third of codes differ by one step) while
  agreeing on magnitude to `1.0e-04` and both landing at √2 of RTN. So every
  stochastic MEAN in this lane is the shipped path's and every per-module
  stochastic figure is one draw from it. Checked rather than assumed.

### Changed

- **The offline bake enters `docs/research/quant_levers.md`'s inventory**, which
  had no row for it, at the top. Its 2026-08-28 deprioritisation is recorded as
  history rather than a standing no, on the owner's call that a decision that
  old does not bind this lane.
- **`docs/research/pdd/2026-08-31_handoff.md` gains what tomorrow starts from**:
  the lever ordering with `unmerged_blocks`' 150-of-200 reach beside the bake's
  200, the four steps in order, and the two things that are NOT available — no
  runnable BF16 H3 DiT on this box (the release is 62 GB of diffusers shards and
  no comfy-layout BF16 DiT exists), and no same-seed pair that can A/B a weight
  change.
- **`docs/h3_pdd.md`'s sigma-shift subsection is rewritten** for `44374a4`. Its
  argument against removing the node was answered rather than overruled: both
  branches it listed were real, the list was not exhaustive, and the third route
  reads the value from what the RUNTIME falls back to when the node is absent.

## 0.99.7

### Changed

- **Every PDD graph's widget values now match the node's own defaults**, on the
  owner's rule that a workflow's PDD values should not differ from the node's
  defaults without a reason. Auditing all nine widgets across the 20 PDD nodes
  found exactly one unintended divergence: `head_strength`, at a literal `1.0`
  in all 20, where the schema default is the `-1.0` sentinel meaning "follow
  `strength`". `resolve_head_strength` follows only on exactly `-1.0`, so a
  shipped graph and a freshly created node behaved identically at
  `strength=1.0` and diverged the moment anyone edited `strength` — backbone
  scaling while the heads stayed pinned. Both builders now emit `-1.0`. No
  shipped graph changes behaviour. Nothing graded this: `check_literal_widgets`
  and `check_pdd_head_selection` both read the node, never the graphs. The
  other two divergences are deliberate and stay — `patch_heads=False` on
  `h3_probe_ref2v_pdd_headfree`, and the per-arm `steps`.
- **The PDD node's sentinel meanings are now on the widget row**, not only in
  the tooltip: `head_strength (-1 follows strength)`, `nfe (0 = use steps)`,
  and the three `unmerged_*` labels. `display_name` only — no input `id`,
  default, or declaration order changed, so none of the saved graphs wiring
  this node are re-pointed.

### Fixed

- **`h3_config.PDD_MANUAL_SIGMAS` was ungraded, and a third copy of it was
  hiding in the check meant to grade it.** `check_pdd_sigmas.py` asserted the
  emitted six-block schedule against its own hardcoded literal, so the config
  string could have drifted freely and the check would still have passed — the
  "helper the check defines rather than imports" trap, one level out from where
  CLAUDE.md states it. It now imports the config value, and additionally
  asserts that `repr(round(v, 6))` of the derivation reproduces it exactly.
  Red-proved by a one-digit change. **The 6dp rounding is load-bearing**: the
  literal differs from the raw derivation by 2.2e-07 (3-4 ULP at float32), so
  substituting the derived value would move the emitted sigmas bitwise and
  break comparability with every `..._manual_sigmas` arm rendered so far. The
  literal stays in `h3_config` rather than becoming a derivation because that
  module imports without torch and several checks depend on that.

- **The PDD graphs no longer wire `MiniMaxH3SigmaShift`** (owner's call). At
  the checkpoint's own 12/3 it patched the model into what it already was, so
  it read as a knob while being a no-op — and the one thing it invited you to
  do, move the shift, is what `pdd_lora.py::check_shift` raises on at step 0.
  Verified redundant at every surface it touches rather than assumed: the class
  it installs (`ModelSamplingAV + CONST`) is what `ModelType.FLOW_AV` already
  selects, its values match `MiniMaxH3.sampling_settings`, and the DiT falls
  back to its own `sigma_shift_video/audio` constructor defaults when the
  `transformer_options` keys are absent — all three land on 12.0/3.0. With the
  node gone `check_shift` compares against the model's class default instead,
  which is the same pair. 20 graphs and their API twins; non-PDD graphs keep
  it, and the generator's condition is `shift == SIGMA_SHIFT` rather than
  `not pdd`, so a PDD arm fused at another shift gets the node back.

### Fixed

- **Three graph notes described a topology that had moved.** All 20 PDD graphs
  told you to change `steps` on a `BasicScheduler` none of them has — the PDD
  node has emitted SIGMAS since 2026-08-28 and `steps` arrives on a socket from
  a `PrimitiveInt`. The node-order note named `SolAttnMiniMax`, withdrawn on
  2026-08-30; the graphs wire `MiniMaxH3SolAttn`. The same note gave
  `ModelSamplingMiniMaxH3` with no hint it is the picker name for
  `MiniMaxH3SigmaShift`, which cost this session a wrong "that node is dead"
  call — `/object_info` is keyed by node_id, and the display name is not in it.
  The `nfe` widget's falsy-sentinel meaning is now stated rather than implied.
- **Three checks asserted the shift node's presence** and went red on the
  correct state once it was removed — the "gains an absent state, revisit every
  assertion" rule, met head on. `check_pdd_sigmas.py` and `check_distill_grid.py`
  now read the fallback from ComfyUI's own `MiniMaxH3.sampling_settings` rather
  than a repo-side literal, so neither can agree with itself after core moves
  the real value; `check_distill_settings.py` normalises the absent case and
  says which check grades it against the observable.

### Retracted

- **The merged-arm reading in `2026-08-31_stochastic_rounding.json` ranked the
  rounding modes backwards, and is withdrawn the same day it landed.** It said
  round-to-nearest was the better arm and stochastic ~18x worse, from
  stored-weight error alone. **That metric rewards the arm that does nothing.**
  Below one quantisation step the merged target sits close to the unmerged
  weight, so an arm that discards the update scores well on distance-to-target.
  RTN is biased and throws most of the update away; stochastic is unbiased by
  construction (`E[Q_s(x)] = x`) and lands it.
  - Fraction of the delta realised along its own direction,
    `<Q(W+d) - Q(W), d> / <d, d>`: PDD RTN **0.467** mean / 0.020 worst against
    stochastic 1.0000; turbo RTN **0.025** mean / 0.0001 worst against
    stochastic 0.9999.
  - **So ComfyUI's stochastic `set_weight` is deliberate and correct**, and the
    sqrt(2) is a real but second-order cost paid to get an unbiased update.
  - The statistic and the PDD arm are pddclaude's
    (`bench/measure_merge_realisation.py`, `b653466`, which withdrew a
    deterministic-merge lever on the same evidence). The turbo arm and an
    independent re-derivation are this lane's; two implementations agree on
    PDD's worst case to four figures.
  - CLAUDE.md already has the general form: a metric that says one arm is fine
    is a claim about the metric until you have checked what it is blind to.
    This one was blind to whether the update happened at all.

## 0.99.6

### Fixed

- **The UI half of the `qwen_view` rename was missed on the first pass**, which
  is this repo's own "editing the generator is half the change" landing on the
  person who wrote the API half. The API branch emitted the dotted form while
  the UI branch kept writing the bare number, so 42 UI graphs carried
  `qwen_view = 512`. Caught by `check_workflow_schema.py`, which grades UI
  widget values against the served node -- the same check that caught the
  identical miss on `size_policy` on 2026-08-27.
- **`check_workflow_schema.py::expand_dynamic_combo` indexed `values` by
  position in `wants`.** That is correct only while a node has at most ONE
  DynamicCombo: expanding one inserts its revealed widgets, so every later
  entry sits further along in `values` than its own index.
  `MiniMaxH3AppendRefImage` became the first node here with two, and the second
  read the first's revealed widget as its selection -- `qwen_view` graded
  against the value of `dit_short_edge`, failing 42 CORRECT graphs. Now walks a
  cursor. Same shape as this function's earlier `forceInput` defect, and the
  same "an assumption that has only ever met one implementation is not a tested
  assumption" rule, this time inside a check.

### Changed

- **`bench/results/` is archived by closed lane, not by date, and the date
  cutoff that prompted it was refuted by measurement.** The proposal was to move
  everything before 2026-08-26. Mapping every result filename against the files
  that cite it showed that axis is close to orthogonal to liveness: most
  pre-cutoff results were cited by tracked code or docs -- several by checks
  that read them at runtime -- while ten post-cutoff results were cited by
  nothing. A flat cutoff would have broken the majority of what it moved and
  stranded the dead files it left behind.
- Three closed lanes moved to `bench/results/archive/`, each with a README
  saying what closed it and what superseded it: the v2 text-encoder calibration
  lane (rejected at Gate 5, closed 2026-08-27; `h3_config.MODELS["clip"]` is the
  artifact of record), the finished LoRA and DiT file-comparison survey, and the
  parked single-frame image swap. Citations that pointed at a moved file were
  rewritten to its new path in the same commit; `bench/check_doc_links.py` is
  what proves it landed.
- Deliberately left in place, because absence of a citation is not evidence a
  record is dead: raw arm records backing verdicts `docs/evidence.md` and
  `docs/SOLATTN.md` still cite, the `_sla` and `_v11` variants (the SLA router
  graph is live), the reference-view occupancy measurement (that ablation is an
  open experiment), and everything from the currently active PDD, marker and
  audio lanes. `bench/results/archive/README.md` carries the rule and the
  exclusions.

## 0.99.6

### Changed

- **`MiniMaxH3AppendRefImage.qwen_short_edge` is now `qwen_view`, a
  DynamicCombo of `separate` / `shared`.** It was an Int whose `0` meant "no
  separate text-encoder view" -- a number selecting a mode, which is the shape
  the literal-widget rule added in 0.99.5 forbids. The size lives under
  `separate` with `min=CANVAS_MULTIPLE`, so `0` is no longer typeable and the
  size box is no longer on screen while inert. Same trade `size_policy` took
  on this node on 2026-08-27: saved-graph widget positions move, and a node
  that cannot mislead is worth it.
  - **All 163 graphs regenerated and validated against a live server**, with an
    exact 1:1 mapping and no behaviour change: 78 at 512, 3 at 2048, and the
    six deliberate shared-view arms (`refview_a`, `refview_c` and the parity
    arms) now say `shared` where they said `0`.
  - `0` survives only as the internal spelling on `RuntimeImageReference`,
    a derived field nobody types. `check_literal_widgets.py`'s allowlist entry
    stays and says so, rather than the detector growing a special case.

### Fixed

- **`bench/preflight_graph.py` could not read the new spelling and said so
  loudly**, refusing to price rather than defaulting -- which is exactly what
  its own docstring argues for, and it caught this rename the way it was
  written to. Now reads `qwen_view` and its nested size, and maps `shared` to
  the internal 0.
  - While fixing it: a `None` default would have reported an ABSENT input as
    `linked`, because `_value` returns `None` for a value wired to another
    node and `linked` is computed from exactly that. Absent and wired are
    different defects and the fallbacks are non-`None` again.
- **`check_reference_runtime.py` asserted the defect fixed in 0.99.4.** Its
  last case required that OMITTING the input yield 0, which is the schema-vs-
  signature split itself written down as an expectation. Omission is no longer
  expressible; `shared` is named instead.

## 0.99.5

### Added

- **`bench/check_literal_widgets.py`, and a standing rule in CLAUDE.md: a
  numeric input means the quantity it names, and a MODE gets its own named
  input.** `qwen_short_edge=0` is not a short edge of zero pixels, it is "no
  separate encoder view"; `nfe=0` and `steps=0` are "the file's own count";
  `reference_video_fit.short_edge=0` is "report only"; `keyframe_canvas.length=0`
  is "no length". Five widgets where a number quietly selects a branch.
  The check flags a declared numeric input whose own value is tested falsy or
  against zero, following a one-hop rename and the `x or fallback` idiom. It
  cannot distinguish a sentinel from a guard, so two allowlists carry that
  judgement and a stale entry fails. Existing sentinels are recorded as debt
  with a replacement named, because converting a widget to a combo re-points
  saved graph values by index and needs every graph regenerated.

## 0.99.4

### Fixed

- **`MiniMaxH3AppendRefImage.qwen_short_edge` rendered differently through the
  UI than through an API prompt that omitted it.** The schema declared
  `REF_QWEN_SHORT_EDGE`; `execute()`'s signature defaulted to `0`. ComfyUI does
  not inject a schema default for an input an API prompt omits, so the two are
  independent -- and `0` is the one value that leaves the reference view
  unclamped in the TEXT segment, where it competes with the prompt rather than
  merely lengthening the sequence. Every shipped API graph sets the key
  explicitly, so no shipped render moves; the exposure was hand-built prompts
  and any external consumer. `0` stays legal when asked for on purpose -- six
  graph arms do -- it is just no longer what you get by omission.

### Documentation

- **`docs/checks.md` gains rows for `check_sol_node_equivalence.py`,
  `check_vsa_core_patch.py` and `check_vsa_geometry.py`**, which were on disk
  and invisible to the index. Three of the six `check_doc_inventory.py`
  reports; the rest belong to other lanes.
- **`check_sol_kernel.py`'s row said it gates presence on a graph wiring
  `SolAttnMiniMax`.** It gates on either node id, and the vendored one has not
  been what a shipped graph wires since 2026-08-30.

## 0.99.3

### Fixed

- **A baked PDD file on an unpruned checkpoint now says what is wrong and how
  to fix it.** It installed none of its 50 adaln modules and tripped the
  generic `declared_adaln` guard, whose message ("declares 50 adaln modules but
  0 reached the model") reads as a corrupt file rather than a form mismatch.
  The new message names both sides from observables -- the file's adaln key
  prefix against the checkpoint's `use_adaln_curves` -- and names the two
  fixes. Metadata is deliberately not consulted: `h3_pdd_base` records the
  converter's `--base`, which for a baked file is the unpruned checkpoint it
  will refuse to load on, so it points at the wrong answer.
- **`docs/h3_pdd.md`'s trade table still priced `unmerged_blocks` at "+19 MB
  per block"** -- the figure the same section withdraws two paragraphs later,
  measured identical at 19995 MB across both arms of the 2026-08-30 smoke. The
  row now reads "no memory".
- **`bench/measure_pdd_quant_interaction.py` hardcoded `"measured":
  "2026-08-30"`**, so any re-run would stamp its record with the first run's
  date. It reads the clock now.

### Changed

- **`MiniMaxH3PDDLoRA` now opens on the BAKED PDD file, not the `adaln2688`
  one.** The combo listed whatever `folder_paths` returned, and ComfyUI selects
  a combo's first option — alphabetically `..._adaln2688_comfy` sorts ahead of
  `..._comfy`, so a freshly dragged node started on the portable file. Both
  work on a pruned base; only the baked one gets there without installing 50
  runtime forward patches, and every shipped graph loads a pruned
  int8_convrot checkpoint. The head of the list is now
  `h3_config.PDD_FL2VA_LORA` then `PDD_REF2VA_LORA`, read from that file rather
  than copied, and intersected with what is actually on disk. Reordering combo
  OPTIONS is safe for saved graphs — `widgets_values` stores a combo's chosen
  string, where inputs are matched positionally.
- **`MiniMaxH3PDDLoRA`'s tooltips lead with what to set.** Every one was
  rewritten to open with the operative instruction ("Leave at 1.0", "EXPERIMENT
  ONLY", "ACCURACY KNOB, off by default") before the reasoning, and the repo
  history several of them carried -- a widget-ordering defect, a withdrawn
  memory figure, which upstream node an input was borrowed from -- moved to
  code comments or to `docs/h3_pdd.md`, which already held it. The node
  `description` now states the three surfaces and says that everything below
  `steps` is an experiment knob. **No widget was added, removed or reordered**,
  so every saved graph reads the same values.
- **`lora_name`'s tooltip now answers "which file for which checkpoint"**, the
  question the node previously left to trial and error, including the
  checkpoint-side observable (`blocks.0.adaln_proj.linear.weight` is
  `[96768, 8]` pruned, `[96768, 2688]` unpruned).
- **`bench/convert_pdd_lora.py` records what its output fits.** New metadata
  `h3_pdd_adaln_form`, `h3_pdd_loads_on` and `h3_pdd_pruned_base`. Until now
  the only base a file named was `--base`, which on the `--pruned` path is the
  one checkpoint it cannot load on. Older files are still classified by key
  prefix, which needs no metadata.

### Added

- **`dit_observe.py`, `quant_observe.py` (`MiniMaxH3QuantObserve`) and
  `bench/check_quant_observe.py`** — Tier 1 of `docs/open_experiments.md` #23:
  the observer that records what each quantised linear's INPUT looks like, per
  block and per step, so the ACTIVATION half of int8_convrot's two roundings
  can be measured. Records per-input-channel absmax and RMS (the absmax vector
  IS the SmoothQuant scale), the per-token max distribution the dynamic
  quantiser acts on, and `PackedLayout.segments` as the authority for row
  modalities. Two gates: the node must be wired AND `H3_QUANT_OBSERVE` set.
- **The observer reaches `mlp.fc2` through an `MLP.forward` wrapper**, because
  `mlp.fc2.forward` is never called on the shipped INT8 path —
  `comfy.ops.linear_input_act` owns it for the SwiGLU fusion. The naive design
  records three kinds and looks complete, which is the defect `unmerged_blocks`
  shipped on 2026-08-30. `_shape_check` asserts four kinds and does not infer
  them from what reported.

### Fixed

- **Both observers armed themselves with their environment variable unset.**
  `enabled()` returned `bool(str(_dir))` with `_dir = Path("")`, and
  `str(Path(""))` is `"."`, so the disabled state evaluated TRUE.
  `pdd_observe.py` shipped that expression, so it would record and write JSON
  into the server's working directory on any render that set
  `unmerged_blocks` — the knob this repo spent today recommending. Arming is
  now its own boolean rather than a property of a path's spelling. Found by
  `check_quant_observe::inert_without_the_env_var` on its first run, a case
  that looked like a formality.

- **`bench/analyze_weight_outliers.py` and
  `bench/results/2026-08-31_dit_weight_outliers.json`** — Tier 0 of the
  quantisation-lever plan: the DiT headroom measurement that had never been
  run, plus the outlier structure the rotation does or does not flatten, over
  all 200 backbone modules. CPU only.
- **[`docs/research/quant_levers.md`](docs/research/quant_levers.md)** — the
  owner doc for what can be changed about H3's quantisation. Nothing owned
  this, which is how a withdrawn claim sat in `h3_dit_implementations.md`
  §10.5.

### Fixed

- **The "deterministic merge" lever is WITHDRAWN, hours after it was proposed,
  and it would have made things worse.** The measured √2 stands — it is a
  statement about rounding a FIXED tensor. Merging is a different problem: the
  delta is **smaller than one int8 step**, so round-to-nearest, being biased,
  simply discards most of it, while stochastic rounding is unbiased by
  construction. `bench/measure_merge_realisation.py` and
  `bench/results/2026-08-31_merge_realisation.json`, 20 modules: RTN realises
  **0.395 of the delta on average and 0.020 on the worst module**; stochastic
  realises **0.99996**. The stored-weight metric ranks them the OTHER way
  (0.004709 against 0.009370) precisely because RTN barely applies the LoRA.
  **A stored-weight metric rewards the arm that does nothing**, and this repo
  recommended that arm on the strength of one — inside the file whose own scope
  line warns about exactly that. ComfyUI's stochastic `set_weight` now looks
  deliberate and correct. The sub-step observation came from a peer session,
  whose independent implementation reproduced the worst-module figure (0.0199
  against 0.01988) on a different subset.
- **Both merge arms are bad on the turbo LoRA, for opposite reasons, and
  `realised_along_d` alone concluded too early.** Records
  `bench/results/2026-08-31_merge_realisation_{pdd,turbo}.json`. A sub-step
  delta is realised as a sparse set of FULL-step jumps, so the direction is
  right and the per-weight representation is not: the shipped stochastic merge
  applies the turbo update and injects **11.87x its magnitude in noise on
  average, 26.6x worst**, against 1.85x for PDD. Deterministic rounding instead
  discards it — 0.025 realised on turbo, 0.0001 worst. `noise_over_delta` was
  added to the producer because neither existing column shows this.
- **The shipped turbo graphs are NOT rendering with a discarded LoRA**, which
  the discard finding could be misread as. Discarding is the RTN failure and
  RTN does not ship: `comfy.utils.string_to_seed` is non-zero on all 200 module
  keys (minimum 12054335), so the stochastic branch runs everywhere and the
  update is realised. The shipped defect is noise, not absence.

### Measured

- **The DiT int8 build is at its format floor.** `e_shipped` 0.0093617314051
  against a deterministic reproduction's 0.0093617314403 — ten significant
  figures, on all 200 modules. **Corrected 2026-08-31**: that is an agreement
  of error NORMS, and the byte claim originally written from it was verified
  separately and is weaker than stated — the file reproduces to within a
  handful of ties (6, 13 and 5 differing values per module at max abs 1,
  roughly one in twenty million), not byte-identically. The reproduction must
  be computed in fp32; in bf16 the Hadamard rotation loses enough precision to
  move ~8% of values, which is a compute dtype effect and **not** the
  encoder/DiT source-precision difference this entry first claimed. There is no
  such difference — see `docs/research/quant_levers.md` §1.
- **Stochastic rounding costs exactly √2, and it is on the shipped merge
  path.** `e_stochastic / e_deterministic` = 1.4142150 over 200 modules
  against √2 = 1.4142136 — seven significant figures, predicted from the grid
  geometry (1/6 against 1/12 of a cell). `comfy/model_patcher.py:928` passes
  `seed=string_to_seed(key)` into `set_weight`, so **any LoRA merged onto an
  int8_convrot module carries √2 the requantisation error a deterministic bake
  would**, and every merge number published here was measured deterministically
  — the optimistic case.
- **`attn.out_proj`'s excess is explained**, closing a `not_measured` line in
  `2026-08-28_quant_hotspots_ref2va.json`. Its outliers span wider than 256
  channels: median row kurtosis after the shipped rotation is ≈0.00 on
  qkv_proj, fc1 and fc2, and 0.16-0.68 on out_proj. A 1024-wide rotation
  reaches them.
- **`convrot_groupsize` 1024 buys 10.2% on `attn.out_proj` and 0.17% on
  `mlp.fc2`** — strongest shallow (15.4% over blocks 0-9, 7.2% at 30-39), 49
  of 50 blocks at or above 5%. Block 0 alone reads 20.8%, so a one-block
  measurement would have overstated it twofold. **It is a trade, not a free
  win**: `_should_use_convrot_fused_kernel` requires `group_size == 256`, so
  gs 1024 loses the kernel that fuses the activation rotation, the row-wise
  quantisation and the SwiGLU, per forward.
- **Two invariant violations, both named rather than smoothed**: `gs 64` beats
  `gs 256` on one module by 3.8e-06 (a near-tie), and the rotation raised
  `group_disagreement` on two out_proj modules — a property of that metric,
  which is a ratio across groups, not the within-group spread the Hadamard
  optimises.

- **`bench/analyze_pdd_unmerge_curve.py` and
  `bench/results/2026-08-31_pdd_unmerge_recovery.json`** — what un-merging N
  blocks recovers, and whether picking N pays. It asserts the 50x4 module shape
  before aggregating, so a partial source cannot yield a ranking that looks
  complete.
- **`bench/results/2026-08-31_pdd_quant_interaction_all_blocks.json`** — the
  quant-interaction measurement over all 200 backbone modules rather than 28.

### Documentation

- **The `unmerged_blocks` worst-first ranking is corrected, in the tooltip and
  in `docs/h3_pdd.md`.** The 7-block sample said "49, 7, 24, 16"; over all 50,
  49 and 7 hold at 1st and 5th but 24 and 16 are mid-pack at 17th and 16th.
  Worst-first is `49, 15, 20, 11, 7, 10, 18, 23, 14, 17`. The sample's mean
  generalised (1.1164x against 1.1234x, correlation 0.751 against 0.78); its
  detail did not.
- **There is no hotspot to target, which changes what the knob is for.** Block
  inflation spans 1.131x, so worst-first is mildly concave rather than steep --
  the worst 5 recover 15.3% of the gap against 10% for any 5. Picking a subset
  buys compute and gives up fidelity, not the reverse.
- **Inflation and stored error rank the module kinds oppositely.**
  `attn.qkv_proj` has the highest inflation under PDD and the lowest error
  before it; `attn.out_proj` the reverse. Only the second bears on the re-bake
  question, and the two had been read as one ranking.
- **The claim that a better quantised DiT "does not exist as an option" is
  withdrawn** (`docs/research/h3_dit_implementations.md` §10.5). It carried an
  ENCODER record to a DiT conclusion, attributed to §7 things §7 does not
  establish, and cited a "534/534 release modules" figure sourced to nothing.
  What survives is that convrot is a deterministic, data-free transform with no
  calibration to improve. What does not is the conclusion, because the lane it
  rests on measures one of the **two** roundings: `int8_convrot` is W8A8, and
  `int8_linear` rotates the activation online and quantises it per token before
  the GEMM.
- **Three quant records now carry that caveat in the record itself**
  (`2026-08-28_quant_hotspots_ref2va`, `2026-08-29_int8_convrot_headroom`,
  `2026-08-30_pdd_quant_interaction`), and their producers emit it. The
  hotspots finding that made `attn.out_proj` "a candidate for different
  treatment in a re-bake" is amended: a re-bake changes the weight rounding,
  and that file cannot see whether the weight rounding is what out_proj's
  runtime error is made of.
- **`docs/open_experiments.md` #23**: what INT8 costs at run time, per module
  kind, with the weight/activation decomposition, a pre-registered decision
  rule, and the observation that the DiT's dimensions single out `attn.out_proj`
  for the one in-format lever -- `_build_hadamard` wants a power of 4 dividing
  `in_features`, so `qkv_proj` and `fc1` (5376) are capped at the shipped 256
  while `out_proj` (7168) and `fc2` (14336) admit 1024.
- **`docs/h3_pdd.md` gains a "which file for which checkpoint" table**: the
  shape observable, what each pairing does, which to prefer, and why
  `h3_pdd_base` is the wrong field to read.

## 0.99.2

### Removed

- **`MiniMaxH3SolAttnCurve` is deleted.** It existed to add a `hilbert` token
  ordering the vendored Sol node's combo could not express, by rebinding
  `morton_perm` on that module at execute time. That node stopped being loaded
  on 2026-08-30, so the rebind patched nothing and `execute()` could only
  raise. `MiniMaxH3SolAttn` owns the Morton code and offers `hilbert` directly.
  `sol_curves.install()` and `_live_modules()` go with it; `hilbert_perm`,
  `verify_adjacency` and `OURS` stay, because the Sol node and two bench
  scripts use them. The node id leaves `bench/node_id_manifest.json`, and the
  two red-harness cases that mutated its entry now mutate other nodes.

### Fixed

- **Six scripts were reading `vendor/sol_attn_minimax.py` as though it were the
  running node.** It has not been since 2026-08-30, and it cannot run on the
  installed kernel at all -- it passes `centroid_tail`, which
  comfy-kitchen#117 removed. They now go through `bench/_live_sol.py`, one
  loader that imports `sol_attn_h3.py` as a member of a synthetic package so
  its relative import resolves without executing the pack's `__init__.py`.
  This is the 2026-08-17 rule repeating: building the replacement is not the
  change, repointing everything that cites it is.
- **`provenance.py::SOL_CLOSURE_KEYS` was recording knobs that no longer exist
  and missing one that does, for the third time.** It asked for `centroid_tail`
  and `reuse_qkv_memory`, both gone from the live node, so both stamped "not
  detected" on every render; and it omitted `tail`, the knob exposed as
  `pooled_tail`, which decides whether unselected blocks contribute a pooled
  term at all. **Its guard could not catch it because the guard was loading the
  retired file** -- repointing `check_provenance_stamp.py` at the live node
  turned it red immediately. `STAMP_SCHEMA_VERSION` 3 -> 4, per the precedent
  that a key change bumps it. The control's candidate list named only knobs the
  node no longer has, so it would have reported a vacuous pass.
- **Seven bench scripts looked for the Sol node under its retired id alone.**
  `smoke_h3.py`, `preflight_graph.py`, `generate_capture_manifest.py`,
  `probe_block_propagation.py`, `time_dense_blocks.py`,
  `check_h3_awq_encoder.py` and `bench_e2e_h3.py` all keyed on
  `SolAttnMiniMax`. The readers now accept both ids, because saved graphs
  predating 2026-08-30 legitimately carry the old one; the builders emit the
  live id. The worst of these was silent: `generate_capture_manifest.py`
  recorded `sol_attn: absent` for a render that had Sol on.
- **`check_bench_matches_shipped.py` was red before this work and is green
  after.** It found the graph's Sol node by a `"SolAttn"` prefix, which
  `MiniMaxH3SolAttn` does not have, so it reported the graph as wiring `None`;
  and `bench_e2e_h3.py` built `SolAttnMiniMax`, a class ComfyUI does not load.

## 0.99.1

### Fixed

- **`bench/check_vsa_core_patch.py` no longer goes red on the state it argues
  is correct.** Its docstring holds that a missing patch is legitimate and that
  failing on it "would train a reader to ignore red" -- then its checkpoint
  case failed on exactly that, because it asked whether the gate keys find a
  slot without first asking whether anything was supposed to build one. Patch
  absent plus checkpoint present is stock ComfyUI with a file downloaded. That
  case is now gated on the patch actually being present, so the run reports
  INCOMPLETE rather than FAILED, and it keeps its teeth where they belong: with
  the patch applied, gate keys that find no slot still fail.

### Changed

- **The VSA core patch is not applied, and waiting for the merge is now the
  decision.** Comfy-Org/ComfyUI#15958 is still an open draft. It was applied to
  the ComfyUI working tree on 2026-08-30; a `git reset` and two pulls carried
  it away, and rather than re-apply a draft this box tracks stock core. The
  working-tree arrangement was chosen so a `git pull` would refuse rather than
  merge silently -- a reset is the case it does not cover, which is worth
  keeping. `MiniMaxH3VSAAttention` refuses at execute on stock core, which is
  the designed behaviour.

### Corrected

- **Three files claimed the VSA core patch was applied; all withdrawn.**
  `CLAUDE.md`, `docs/research/vsa/vsa_node.md` and `vsa_attention.py` each said
  the ComfyUI checkout carried the draft in its working tree. It does not.
- **`vsa_attention.py` said nothing had rendered end to end, and that the gate
  projection, kernel call and output reordering had never run under a real
  forward.** Both withdrawn: it rendered on 2026-08-30 against a dense control
  (`bench/results/2026-08-30_vsa_first_render.json`,
  `_vsa_length_scaling.json`). Those runs used the patched tree and are not
  reproducible here until #15958 merges. Production length, a matched sampler
  recipe and anything perceptual remain unexercised.

## 0.99.0

### Added

- **`MiniMaxH3SolAttn`: the Sol-Attn node is ours now, and `pooled_tail` is
  exposed.** A fork of the vendored upstream node, taken because
  `vendor/README.md`'s value proposition -- a disagreement with that file is a
  finding rather than a merge artifact -- was spent on 2026-08-29 when the
  merged kernel's API change was absorbed by editing it in place. Its own rules
  give the remedy: fork, rename, record. `centroid_tail` and
  `reuse_qkv_memory` are gone rather than inert; the kernel API is asserted
  once at patch time instead of probed per call, because a missing kwarg raises
  inside `override`, which catches everything and falls through to dense.
  Output-neutral at the shipped settings and measured, not argued:
  `bench/check_sol_node_equivalence.py` asserts the two dispatches produce the
  same bytes at both selections. All 149 graphs regenerated.
- **`pooled_tail` is the kernel's `tail` and is not `centroid_tail` renamed.**
  That one asked WHERE the pooled term is evaluated; this asks WHETHER there is
  one. Off, unselected blocks are dropped outright -- upstream's tests call it
  the SLA / VSA fine stage, and with top-k selection it reproduces the routing
  the Turbo-SLA LoRA was distilled under, through `optimized_attention`, which
  reaches all 52 `Attention` modules rather than the 50 an object patch sees.
  Nothing has rendered under it and the shipped config leaves it on.
- **`MiniMaxH3VSAAttention`: FastVideo VSA, blocked on core and saying so.**
  Replaces the 50 main DiT blocks, groups video tokens into 4x4x4 cubes one per
  64-row kernel block, and passes each block's learned `to_gate_compress` as
  `coarse_gate`. **Separate from the Sol node by design, not necessity** --
  corrected 2026-08-30, having first been written as "cannot be a widget on the
  Sol node". An override is handed Q/K/V already built, but a forward pre-hook
  can stash the block input into `transformer_options`, which is the route
  `MiniMaxH3SolAttn` already uses to publish the block index; verified by
  executing the pattern. The real reasons are that VSA needs the cube reorder
  and padding alongside the gate, and that the two regimes are mutually
  exclusive at the same 50 blocks. Only `sol_attn_chunked` is genuinely
  unreachable from an override: it exists to never materialise Q/K/V, and by
  the time an override runs they are built and rope is applied, so its saving
  is spent and feeding it post-rope tensors would apply rope twice.
  Accepts any prefix where video is
  last, so reference graphs are in scope, where the one other implementation
  restricts itself to plain text/audio/video.
  **It refuses on a model without a gate rather than running.** On stock
  ComfyUI a VSA checkpoint's gate keys have no slot and are dropped on load
  with a warning, and the render then succeeds as the dense base -- a silent
  dense render the user believes is VSA. Needs Comfy-Org/ComfyUI#15958, still a
  draft, and that PR is necessary and not sufficient: its own comment says the
  gate is unused by the dense forward.
- **VSA rendered, once, and its gate is genuinely consumed.** Two arms,
  `h3_probe_vsa` and `h3_probe_vsa_dense`, matched at one seed on the same
  checkpoint and differing only in whether the node is wired. The node logs
  `VSA on 50 blocks` with no fallback warning, its output DIFFERS from the
  dense control, and two VSA runs at the same seed produce identical pixels --
  so the replacement is not quietly falling through to the original block.
  `bench/results/2026-08-30_vsa_first_render.json`.
  **This answers a mechanical question and no other.** No quality claim: a
  rendered pair cannot A/B a numerical change, and this pair changes the
  attention regime outright. The shape is 22,121 packed rows against the
  31k-128k the shipped graphs run, which is close to the least favourable
  length for sparse attention. The timing (warm, VSA 22.68 s against 24.18 s,
  one run each) is NOT MEASURED in either direction -- one run per arm cannot
  separate 6% from run-to-run excursion, so it is neither evidence of a speedup
  nor of its absence.
- **`bench/verify_vsa_render.py`, and the trap it exists for.** The first
  comparison hashed the mp4 FILES and was wrong in a way that looked right: two
  renders of identical pixels produce different mp4 bytes. Two tags do it and
  only one explains the same-arm case: `format.tags.comment` carries the whole
  API prompt, so any two ARMS differ by construction, and
  `format.tags.creation_time` is a wall clock, so any two RUNS differ. The
  muxer and codec are not the cause -- remux and re-encode are both
  deterministic. Caught because the two VSA runs hashed differently while their
  file sizes matched to the byte. The one-way implication survives: a MATCHING
  hash still implies matching frames, so nothing previously concluded from one
  is withdrawn.
  It also identifies the arms from the graph embedded in each file rather than
  from filenames, because `bench/smoke_h3.py` hard-codes one `_smoketest`
  prefix and every session on this box shares that counter. Red-proved with
  another session's render as the wrong arm: the pixel comparison alone reports
  "the arms differ" and passes.
- **`bench/check_vsa_core_patch.py`, and the draft core patch it records.**
  VSA needs ComfyUI to build a `to_gate_compress` slot per block, which stock
  master does not; `comfyanonymous/ComfyUI#15958` is a DRAFT that adds it.
  Applied to this box on 2026-08-30 from head `10febb01`, as an uncommitted
  working-tree change rather than a merge, so it reverts with one `git
  checkout` and a later `git pull` refuses instead of quietly merging a draft.
  **So the H3 model this box builds is not the one stock ComfyUI builds, and
  any H3 result taken here has to say so.** `CLAUDE.md`'s "nothing here patches
  core" shorthand is corrected: still true of the pack, no longer true of the
  box.
  The check reports ABSENCE rather than failing on it -- a machine without the
  patch is the normal state. It fails on a HALF-applied patch, because the two
  halves fail in opposite directions and one is silent: with only the model
  change, every H3 model takes a `gate_compress` parameter detection never
  sets, so it stays False and looks exactly like stock. It also verifies the
  patch against the artifact rather than only the source -- all 50 gate weights
  find a slot and no weight key is orphaned, on meta tensors so nothing is
  allocated. Before the patch all 50 were dropped on load and the render
  succeeded as the dense base.
  One case in it was written vacuous and is fixed: it asked whether the running
  core takes `gate_compress` by importing core FRESH, which reads the same
  files the check had just read and so agreed by construction. It now compares
  the port owner's start time against the patched files' mtimes, which is the
  question that matters.
- **`bench/check_vsa_geometry.py`.** The cube reorder cannot fail loudly -- a
  wrong permutation yields a tensor of the right shape and a successful render
  -- so the invariants are asserted directly over five shapes ragged in every
  axis at once. Cube membership is re-derived from the source index rather than
  from the walk that built it. A red control corrupts the permutation by one
  block and confirms three invariants catch it.
- **`bench/check_sol_node_equivalence.py`**, and **`block_spec.py`**, which
  gives `parse_blocks` a home outside a node module -- four callers were
  reaching for it and two into the vendored file, which cannot stay true of a
  read-only reference.

### Changed

- **`MiniMaxH3SolAttnCurve` is superseded and now says so.** It added the
  `hilbert` ordering by rebinding `morton_perm` on the live VENDORED module,
  resolved by identity because a running ComfyUI can hold two module objects
  for one file. That node is no longer loaded, so this one can only raise --
  which it does, loudly, by its own zero-is-a-failure rule. `MiniMaxH3SolAttn`
  owns the Morton code and offers `hilbert` directly in `morton_curve`, so the
  capability is unchanged and needs one node instead of two. Marked deprecated
  and its error now names the replacement instead of telling you to install a
  pack that is deliberately disabled. Wired by no graph.

- **`vendor/sol_attn_minimax.py` is a read-only reference again.** Restored to
  the last genuine upstream drop (v3, `7805cf37`, recovered from `e18bbc0`) and
  the installed pack renamed `.disabled`, which ComfyUI skips. Nothing deleted;
  the symlink still points here, so renaming it back restores the old
  arrangement exactly. The property a vendored file exists to provide -- a
  disagreement with it is a finding rather than a merge artifact -- was spent
  on 2026-08-29 when the merged kernel's API change was absorbed by editing it
  in place. The restored file cannot run on the installed kernel (it passes
  `centroid_tail`, which #117 removed, and that raises inside the override's
  catch-all and becomes a silent dense render), which is why it is disabled
  rather than merely superseded. `check_sol_kernel.py` now asserts the hash is
  a recorded version, is NOT one of ours, and is not loaded by ComfyUI.
- **`bench/check_sol_node_equivalence.py` grades against the KERNEL, not the
  algorithm, and the first draft got that wrong.** Its baseline was the
  vendored node, which no longer runs, so it was repointed rather than left to
  skip. Written first against the eager reference, it read as a marginal
  failure at cos 0.994-0.998 depending on shape and seed, and the instinct was
  to loosen the bar. Measured instead: the dispatch is BITWISE identical to a
  direct `sol_attn` call, and every bit of that spread was the kernel's INT8
  arithmetic against fp32 -- a property `check_solattn_correctness.py` already
  owns, which a loosened tolerance here would have silently absorbed. **A
  tolerance where an equality is available is a check that cannot see small
  defects.** The kernel oracle also needs no O(T^2) score tensor, so it runs at
  16k tokens instead of a toy shape, and gains a sink case and a
  transposed-oracle red control.

- **`bench/_sol_attn_reference.py` was the PRE-MERGE algorithm.** It carried
  `centroid_tail` and had no `tail`, `block_len` or `coarse_gate`, so from the
  day the merged kernel was installed the only controlled comparison this repo
  can make about a numerical Sol knob graded that kernel against an algorithm
  it does not implement. It passed because the arms exercised only shared
  parameters. Re-vendored from `dae00a1`, byte-identical to upstream's function.
- **`bench/check_solattn_correctness.py` gains four cases and two red
  controls**, covering `tail=False`, top-k with no tail, `block_len` padding
  and `coarse_gate`. Every one was unreachable before the merge, which is the
  class no existing case could have covered: there was no argument to pass. Its
  tail-mode probe is retired -- it existed to DISCOVER which form the kernel
  implemented, and there is no longer a choice.
- **`h3_config.SOL_CUDA_DEFAULTS` now describes our node.** Three values moved
  and all three are the node's own defaults rather than re-tunings; `tau` and
  `morton_curve` now agree with `SOL_RECOMMENDED_CUDA`, which they did not
  before, with nothing asserting either.

### Fixed

- **Two doc citations into `coderef/` broke when that checkout advanced to the
  merged kernel, and only one was caught.** The other stayed in range and now
  points at an unrelated helper. `bench/check_doc_links.py` can see a line that
  does not exist, never a line that means something else, so a citation into a
  file that moves under you is only half-guarded by it. Both re-cited by symbol.

### Verified, and worth recording as a non-finding

- **kijai's `sol_attn` branch tip is content-identical to the installed
  build.** The branch gained 34 commits after the merge -- VSA support, a
  chunked QKV producer, top-k guards -- which reads like a kernel we did not
  have. Whole-tree `diff -rq` between the two checkouts reports no difference,
  and the four entry files match `site-packages` byte for byte. There was
  nothing to pull and nothing to rebuild.

### Added, earlier the same day

- **`unmerged_blocks` on `MiniMaxH3PDDLoRA`: apply a block's backbone LoRA at
  the call instead of merging it into the quantised weight.** Takes
  `dense_blocks` syntax; empty is the default and is bit-for-bit the old
  behaviour. Verified on the card at 1344x768 x 39 frames, t2v PDD 4-step: the
  arm logs `292 weight patches, 16 module(s) un-merged at blocks 7,16,24,49`
  against the control's `308 weight patches, all merged`, and both render.
  `bench/check_pdd_unmerged.py` grades the identity that makes the arms
  comparable, with four deliberate violations.
- **The bake blocker in `docs/h3_pdd.md` is superseded, in the direction it did
  not expect.** That section treated the dequantise/add/requantise round trip
  as the reason not to pre-merge the backbone. Measured: a bake from the bf16
  release quantises ONCE and lands on 0.00942 at every strength -- exactly the
  base checkpoint's own error -- while the run-time merge quantises twice and
  reaches 0.01058 at strength 1.0. So the run-time patching is the lossy
  option and the bake is not, which is what that section's own "what would
  settle it" paragraph asked for. It stays deprioritised; the blocker is what
  changed, not the priority. The section carries a superseded note saying what
  it used to claim.

- **Merging the PDD LoRA into an int8_convrot module raises that module's
  quantisation error, and the size of the rise tracks the LoRA rather than the
  base.** `ModelPatcher.patch_weight_to_device` dequantises, patches, then
  requantises with `scale="recalculate"`, so PDD moves the quantisation grid.
  Measured against the bf16 release over 28 backbone modules
  (`bench/measure_pdd_quant_interaction.py`,
  `bench/results/2026-08-30_pdd_quant_interaction.json`): the mean stored-weight
  error goes from 0.00942 unpatched to 0.01058 at strength 1.0, smooth and
  monotone through the intermediate strengths, and correlating 0.78 with the
  module's own `||BA||/||W||`. `blocks.49.mlp.fc2` is worst at 1.31x. At
  strength 0 the round trip is free to eight digits, which is the harness
  checking itself; `e_vs_unpatched` is carried as the control that the
  measurement can see a strength effect at all.
- **There is no per-block quantisation sensitivity to key a schedule off, and
  PDD's own per-block magnitude has the structure instead.**
  `bench/measure_pdd_block_magnitude.py` and
  `bench/results/2026-08-30_pdd_block_magnitude.json`: int8-vs-bf16 stored-weight
  error spans 1.086x across the 50 blocks, while PDD's update spans 6.42x, is
  smallest around blocks 29-47 and spikes at block 49. The update is orthogonal
  to the weight it patches (cosine ~1e-4), so it is new structure rather than a
  rescale. Consequence for any "scale PDD down where quantisation hurts" plan:
  the two profiles are not the same variable, and the second one is anti-aligned
  with the propagation ranking in `2026-08-29_block_propagation.json`.

- **Gap 16: the generic VAE crop narrows the reference waveform's sample axis,
  and this repo's own trim is what puts us on a length where it bites.**
  `comfy/sd.py` leaves `crop_input` at its `True` default on the H3 audio
  branch, so `vae_encode_crop_pixels` treats the sample axis as a spatial one
  and narrows it to a multiple of 800, taking half off the front. Measured
  against the real audio VAE with `crop_input = False` as the matched control
  (`bench/audit_ref_audio_crop.py`): the shipped 124-frame trim of 165,333
  samples loses 266 leading samples, 8.3 ms, and one latent step. The worst
  case over all lengths is 399 samples, 12.5 ms. Reference video is not shifted
  with it, measured in the same script: its time axis is dim 0, so the crop
  never reaches it. Prompted by the upstream PR that proposes the one-line fix.
- **Gap 5's measurement is recorded as having been blind to it.** Its arms were
  5 s and 15 s, both exact multiples of 800, which is precisely the input on
  which the crop is a no-op. Nothing in gap 5 is withdrawn.

## 0.98.0

### Fixed

- **The `centroid_tail=False` guard produced the exact failure it was written to
  prevent, and a code review caught it.** It raised from `_run`, which
  `make_override.override` wraps in `except Exception -> dense()`. So on the
  merged kernel a graph asking for `centroid_tail=False` did not fail: every
  eligible call raised, was swallowed, and fell through to sage — a fully dense
  render reporting success at roughly 1.9x the time. A guard against a silent
  change of MATH produced a silent change of KERNEL. It now lives in
  `_apply_patch`, where it propagates out of `execute` and fails the node
  before sampling; confirmed by rendering the refused configuration and getting
  an error on `SolAttnMiniMax` rather than a clean video.
  - **Nothing inside `_run` may rely on raising to reach a user.** Said in the
    code beside the dispatch, because the catch-all is several hundred lines
    away from the thing it silences.
- **The kernel-signature probe read the wrong entry.** `_kernel_kwargs()`
  introspected `comfy_kitchen.sol_attn` and the result gated a call to
  `comfy_kitchen.backends.cuda.sol_attn`. On kijai's branch build — the build
  the whole adaptation exists to keep working — `reuse_qkv_memory` is a
  parameter of the CUDA entry and NOT of the registry entry, so the reuse path
  was unreachable there and the log claimed the build "does not accept it"
  while the entry about to be called did. Now probed per entry, cached per
  callable, and exposed as `kernel_accepts()` so `check_sol_kernel.py` grades
  the same entry the render uses.
- **`bench/check_exact_blocks.py`'s docstring exclusion was a no-op**, and its
  recorded red-proof reason was wrong with it. `ast.walk` is breadth-first, so
  the Expr's Constant child is already enqueued when the Expr is skipped — a
  function body that is exactly `"_uses_optimized_attention"` satisfied the
  contract check. The control that "proved" prose could not pass went red
  because a multi-line docstring is not equal to the flag. Docstring constants
  are now subtracted, and the fix is re-proved on the case the old code could
  not catch. `docs/checks.md` records the corrected reason: **a control that
  goes red for a reason other than the recorded one is a control nobody can
  rely on.**
- **`bench/time_dense_blocks.py`'s per-block cost was only correct for its
  default `--specs`.** It divided the delta against the reference arm by the
  widest arm's ABSOLUTE block count; those coincide only when the first spec
  names zero blocks. Any custom `--specs` understated the cost silently. Now
  divided by the block-count difference, with the reference arm named in the
  header and recorded in the result.
- **Six stale rows in `docs/SOLATTN.md`**, all written earlier the same day.
  The `SOL_RECOMMENDED_CUDA`-vs-`SOL_PDD_CUDA` table still had four rows after
  three of the overrides were dropped; one paragraph said the base config was
  "deliberately left at the vendor's `0-1`" while `h3_config.py` and the top of
  the same file said `0-2,32`; and the open-questions table still listed the
  `centroid_tail` arm as live and `dense_blocks="0-1"` as unadopted. Corrected
  with what each used to claim.
- **Both `SolAttnMiniMax` widget tooltips described the old kernel** —
  `centroid_tail` invited the user to "turn OFF for a quality A/B", which on
  this build is a refused configuration, and `reuse_qkv_memory` promised a
  saving it can no longer deliver. The tooltip is the surface a user actually
  reads; `vendor/README.md` recording them as inert was not enough.

### Added

- **`bench/restart_comfy.sh`, because the same mistake landed three times in
  one day.** It stops ComfyUI **by port owner** (not by process pattern, which
  picks the `uv run` wrapper and leaves the server holding the socket), waits
  for the port, starts detached, waits for readiness, and then asserts the new
  process's start time POSTDATES every path given with `--newer-than`
  (`--kernel` resolves the installed `comfy_kitchen` dist-info). Exits 4 if
  not.
  - The three escaped instances: a "merged kernel is bit-identical" claim taken
    against a process that started 82 seconds before the wheel was installed; a
    red-proof of a new guard that failed in the wrong place because the server
    predated the guard by an hour; and a graph validation against a schema
    missing a new node. Every one produced a plausible result rather than an
    error, and all three had one cause — `pkill ... ; nohup ./start.sh &` in a
    single compound command, where the kill takes the shell down before the
    launch runs and `start.sh` logs "Port 8188 is already in use" into a file
    nobody reads.

### Changed

- **Moved to the merged upstream Sol-Attn kernel**, 0.2.31+sol.dae00a1
  (Comfy-Org/comfy-kitchen#117, merged 2026-08-29), from kijai's branch build
  0.2.31+sol.23d1a66 that every Sol figure here was measured on.
  - **The merged API is not the branch API.** `centroid_tail`,
    `reuse_qkv_memory` and `max_blocks` are gone from both entries; `tail`,
    `block_len` and `coarse_gate` arrived. Installing it against the old node
    breaks every Sol render -- and breaks it SILENTLY, because a TypeError
    inside the override is caught and converted to a dense fallback.
  - **The vendored node now reads `sol_attn`'s signature and passes what it
    accepts**, so one node drives both builds. Branching on the observable
    rather than a version is not a style choice here: both builds call
    themselves 0.2.31.
  - **`centroid_tail=False` raises rather than being dropped.** The merged
    kernel evaluates the pooled tail at the query block's centroid
    unconditionally -- which is what `True` always did, so our config is
    behaviour-preserving -- but `False` is a different computation this build
    cannot express, and swallowing it would change the math silently.
  - **Equivalence measured, not assumed**
    (`bench/results/2026-08-29_sol_kernel_merge_equivalence.json`). Two
    controls make it attributable: branch-against-branch is bit-identical
    three hours and a restart apart, and the sage-only baseline is
    bit-identical across the swap. The Sol arm differs by **rel L2 7.67e-05**
    on video and not at all on audio -- about four orders of magnitude below
    the 0.0128 effect the harness measures. Existing numbers stand; a figure
    quoted to more than three significant figures across the two builds does
    not.
  - `bench/check_sol_kernel.py` now splits REQUIRED kwargs (passed on every
    call; missing means nothing renders) from OPTIONAL ones the node adapts
    around, prints which are present, and fails if `h3_config` asks for a
    `centroid_tail` this kernel cannot express.
  - `vendor/rebuild_kernel.sh` takes `SRC=` so it can build any checkout or
    worktree, and exports `VIRTUAL_ENV` for `--no-build-isolation` -- without
    which uv reports a MISSING BUILD DEPENDENCY rather than a missing
    environment, which sends you installing setuptools where it already is.
  - **Not yet exposed**: `tail=False` (VSA), `block_len`, `coarse_gate` and the
    chunked QKV producer (~5 GB peak at 113k tokens, our exact regime). Those
    need node inputs, and adding them is a separate change.

- **`SOL_RECOMMENDED_CUDA`'s `dense_blocks` is `0-2,32`, was the vendor's
  `0-1`.** The propagation probe ran on the BASE model, so this config is the
  one it bears on most directly and `SOL_PDD_CUDA` now inherits the value
  rather than restating it -- **a PDD arm differs from the base recipe in
  exactly ONE knob**, `end_percent`, which is the one with a derivation behind
  it. Every shipped graph moves, not just the distilled ones.
  - It keeps the vendor's front and extends it: NVLabs' `0-1` survives (blocks
    0 and 1 rank first and third), and what it adds is 2 and 32. What it
    refutes is protecting the TAIL -- 45, 48 and 49 are the three lowest
    measured, under half of block 0.
  - Cost at 16 steps is priced, not measured: Sol covers 11 steps there rather
    than 2, so two extra blocks cost proportionally more and still land near
    1%. Said as a price rather than a figure.

### Fixed

- **`MiniMaxH3PDDLoRA`'s widget order disagreed with its own schema, and every
  UI graph on disk was wrong for most of 2026-08-29.** `head_strength` was
  added at input position 2 that afternoon while the generator emitted it
  third, so neither matched the other: a loaded graph read `patch_heads` as
  1.0, `nfe` as True and `steps` as 0.
  - The input is now APPENDED, which is the rule `nodes.py` and
    `MiniMaxH3SageAttention` both state and the reason for it -- saved graphs
    match `widgets_values` by INDEX, so an insertion re-points every later
    value in every graph already on disk. The tooltip's "no saved workflow
    changes" was true of the SENTINEL and never of the ORDER; two different
    claims.
  - **This is what `check_distill_settings.py` had been red about since the
    morning.** It was reporting an `nfe` of True on a node whose `nfe` is an
    Int -- the value it was really reading was `patch_heads`. Read as a graph
    defect for three sessions. It is green now, and nothing about the graph was
    ever wrong.
  - `bench/check_node_ids.py` caught the insertion the day it happened and was
    overruled by nobody; it simply went unread. Its manifest is now updated
    deliberately, recording the append plus the two genuinely new nodes.

- **The build-time validator now type-checks each widget value against the
  widget it lands on**, which is the gap that let the above validate clean: it
  compared the COUNT of values against the schema and never their types, so six
  values for six widgets passed while three of them were on the wrong widget.
  Lenient by design -- FLOAT accepts an int, linked widgets are skipped -- so
  it fires on a positional shift rather than on formatting.
  - **Proven red on the exact defect**: restoring the old order fails the build
    naming `patch_heads is BOOLEAN but got 1.0` and `nfe is INT but got True`,
    and writes nothing.

- **`SOL_PDD_CUDA` is now TWO knobs, measured, where it shipped with five that
  morning.** `end_percent` 0.74 and `dense_blocks` `"0-2,32"`. Three overrides
  were dropped for having no evidence and no effect: `min_tokens` 11776 (inert
  -- every PDD graph packs 60,972 to 113,032 rows, so it and the inherited
  12288 select identically on all of them), `morton_curve` `2d_frame` (inert
  while `morton=False`, and the inherited `3d` at least has a measurement
  behind it), and an intermediate widening to `0-5,48-49` whose blocks 3-5
  were extrapolated.
  - **`dense_blocks` rests on `bench/probe_block_propagation.py`**, which runs
    Sol at exactly ONE block with sage everywhere else and reads the output
    latent. **Block 0 -- the block Sol approximates BEST -- moves the output
    MOST (0.0306); blocks 45, 48 and 49 move it LEAST (0.0109-0.0128).** So the
    tail is the worst place in the model to spend a dense block, and the
    original `0,1,2,48,49` spent two of five there. Video and audio
    independently rank the same four highest, and block 32 -- a genuine second
    peak -- replicated at a second seed (+25% and +31% over block 16 on audio).
  - **This reverses `bench/rank_dense_blocks.py`**, added hours earlier, which
    put block 0 last and block 40 first. Propagation is the difference. That
    file stands: it is right about what it measures and simply does not decide
    this knob.
  - **Cost is measured, not derived**: `bench/time_dense_blocks.py` gives
    **1.01 s per dense block** at 4 steps, linear from 2 to 8 blocks, against a
    0.9 s noise floor. Four blocks is +4.1 s on a 150.4 s render. Per-block
    cost is UNIFORM, refuting the guess that block 0 would be cheapest.
  - **The first timing run was wrong and the harness now refuses that shape.**
    Its warm-up shared a spec with the first timed arm, so the reference came
    back from ComfyUI's cache in 3.0 s and per-block cost printed as 19.51 s.
    Arm times were also quantised to exactly 3.0 s, the poll interval. It now
    reads the server's own execution span and refuses any arm whose SAMPLER was
    cached.
  - **`SOL_RECOMMENDED_CUDA` deliberately stays at the vendor's `0-1`.** The
    measurement is a base-model one and applies there most directly, but `0-1`
    has H3-specific external validation and two seeds is not enough to overturn
    it across every non-distilled graph.

- **`check_pdd_sigmas.py` now asserts the flat `end_percent` keeps the final
  evaluation dense at every legal PDD step count**, at both shipped shifts.
  That property is the entire argument for one constant, and nothing asserted
  it. **It is newly load-bearing**: `resolve_emit_steps` refused every
  non-divisor until 2026-08-29, so 4 and 8 were the only reachable counts;
  the envelope route now admits 5, 6 and 7 as well. Verified 2/4, 2/5, 3/6,
  4/7, 4/8 sparse, identical at shift 12 and shift 6.
  - **Scoped to counts within 2x the trained block width, and the exclusion was
    found by the check going red rather than assumed**: at 2 evaluations the
    flat 0.74 genuinely does leave the last step sparse. The property is true
    over the usable range and false just outside it, which is worth knowing
    before anyone reaches for a 2-step arm.
  - Shown red at `end_percent` 0.9, naming the sigma and the band.

- **Wrote down what would replace the eyeballing, before the shipped values
  close the question.** `docs/SOLATTN.md` gains a derivation per knob, pointed
  at from `h3_config` and `docs/roadmap.md`. Three findings, none of them a new
  measurement -- all three fall out of what the repo already holds:
  - **`dense_blocks` is the only behavioural change on 11 of the 16 PDD arms.**
    They run 4 evaluations, where 0.74 is what `SOL_END_PERCENT_BY_STEPS`
    already gave and the other two knobs are inert. On the 4 arms at 8
    evaluations both are live, and `end_percent` is the larger intervention by
    compute -- one whole sparse step against three blocks.
  - **`end_percent` 0.74 is a rule, not a number.** `percent_to_sigma(0.75)` is
    0.8 exactly at shift 12, and 0.8 is index 24 of PDD's 32-point grid: the
    start of the final block at 4 evaluations. So the value chosen by eye is
    "run dense over the coarsest schedule's last block", pinned to the sigma
    path rather than to the step grid -- which is why one constant serves both
    counts. It makes an untested prediction the recipe already bets on: at 8
    evaluations 0.87 should be worse than 0.74.
  - **The measured per-block evidence does not support `0,1,2`.**
    `bench/results/2026-08-19_sol_error_per_head_tau1.0.json`, at the shipped
    tau and production sequence length, ranks block 0 as Sol's MOST accurate
    and block 40 as its worst -- and nothing keeps 40 dense. Block 49 earns its
    place on quantization error rather than sparsity; 1, 2 and 48 have never
    been measured. Stated with its counter-reading: error at a block is not
    impact on the output, and nothing here measures propagation.

### Added

- **`MiniMaxH3ExactBlocks`: run named DiT blocks on exact attention, neither
  sage nor Sol.** Nothing in this pack could do that before. `dense_blocks`
  reads as "keep these exact" and never was: `dense()` hands the call to
  `previous`, which on every shipped graph is sage.
  - **The measured case is the last block.** Sage's rel L2 at block 49 is
    ~0.031 against ~0.005 at block 0, and its `cos_min` there is NEGATIVE, so
    on some rows its output is anti-correlated with the exact answer. That is
    also the block a distilled output head reads directly.
  - Costs about 1.7x the sage time on the blocks it names, so `48,49` is
    roughly +7% of attention -- a block-level swap rather than the per-step
    blowup that dropping sage outright would be, which is the only shape that
    makes sense at 4-8 steps.
  - **No shipped graph wires it and its end-to-end benefit is unmeasured.**
    Shipped as a knob, said so in the node description and in `docs/SOLATTN.md`.
  - Appended to the node list, never inserted: saved graphs match widgets by
    index.
- **`bench/check_exact_blocks.py`, because that node branches on somebody
  else's private attribute.** It stays out of Sol's composition by setting
  `_uses_optimized_attention`, and an upstream rename would silently return its
  blocks to sage with every render still succeeding. Asserts that both compose
  sites in the vendored module still READ that flag -- from the AST, so prose
  cannot satisfy it -- and that `_exact_forward` strips the override from what
  it passes down **without mutating the caller's dict**, which would otherwise
  disable sage and Sol for the whole rest of the model while looking like a
  two-block change.
  - **Proven red three ways**, and the outcome was open: the flag renamed
    upstream (both sites go red), the flag left only as docstring prose (still
    red), the unmodified source green.

- **`bench/rank_dense_blocks.py`, and it corrects the ranking above.** Ranking
  blocks by Sol's error is the wrong ranking, because **a block in
  `dense_blocks` does not run dense attention** -- `make_override`'s `dense()`
  hands the call to `previous`, which on every shipped graph is sage. What the
  knob buys is the DIFFERENCE. The script subtracts the two records this repo
  already holds and writes
  `bench/results/2026-08-29_dense_block_ranking.json`.
  - **Block 40 removes the most error (0.209) and is in no shipped list.
    Block 0 removes the least of any block in the model (0.093). Block 49 is a
    sound mid-table 0.160.** So `SOL_PDD_CUDA`'s `0,1,2,48,49` is the weak half
    and the sound half together, and the strongest available candidate is
    absent.
  - **The correction matters most exactly at block 49**, which is why the
    subtraction was worth doing: sage is 6.6x worse there than at block 0, so
    Sol beats it by only 6.2x against 15-21x elsewhere. Ranking on Sol alone
    puts 49 second; the honest figure puts it third. Sage's own `cos_min` at
    block 49 is NEGATIVE (-0.04 to -0.11) -- on some rows the replacement is
    anti-correlated with the exact answer. Dense still wins there by six times,
    so this bounds the size of the win rather than removing it.
  - Refuses rather than prints when the two records sit at different sequence
    lengths, since Sol's error is length-dependent and two lengths are two
    operating points. Confirmed red on a doctored input.
  - **`-1` is not a sixth block.** `parse_blocks` resolves it to 49 and returns
    a frozenset, so `"0,1,2,48,49,-1"`, `"0,1,2,48,49"` and `"0-2,48-49"` are
    the same five blocks. Verified by calling the function.

### Fixed

- **The PDD head bank is back on the CPU, and out of `model_management`'s
  hands.** 2af7f0b attached it to the model patcher with
  `set_additional_models` so ComfyUI could account for it and offload it under
  pressure. A render says that trade goes the wrong way. What it bought is
  42 MiB of HOST ram per cached arm, on a box whose peaks are tens of
  gibibytes; what it cost is ~87 MiB of CARD memory for the whole render (the
  buffers are fp32 and ComfyUI loads them to `load_device`), plus the fp32
  fused masters, which now cached on the card too. The ref2va failure in
  `bench/results/2026-08-28_pdd_ref2va_memory_marginality.json` was short by
  17.5 MiB. It had also already cost the device crash fixed in 0.97.0.
  - **And it left the leak detector permanently red.**
    `ModelPatcher.clone` re-clones every `additional_models` entry, so each run
    wrapped the same `_HeadBank` in a throwaway patcher; `LoadedModel` holds
    the patcher only weakly, so once a clone chain was collected the entry
    reported `is_dead` -- module alive, patcher gone -- and every subsequent
    model load logged `WARNING, memory leak with model _HeadBank` and dragged a
    full `gc.collect()` with it. Whether the retention it points at is real is
    NOT established; what is established is that with no patcher there is no
    entry.
  - **`set_additional_models` was the wrong shape, checked against core rather
    than assumed.** Core uses it in one place, `comfy/multigpu.py`, for
    whole-model clones on other devices -- the case its per-clone copy is
    written for. For weight-sized side data riding along with a patcher, core's
    own shape is `ModelPatcher.patches`: plain CPU tensors, unmanaged, cast at
    use, shared across clones by a list slice. That is what this is now. The
    managed alternative, if it is ever wanted, is ControlNet's: ONE long-lived
    patcher every `copy()` shares by reference, reached through the
    conditioning's `get_models()` -- not available to a node that returns MODEL.
  - The 0.97.0 device fix stays, and so does its case: the module is still free
    to move and the fusion still follows the stack, it just is not moved any
    more. `_HeadBank.__init__` pins its buffers to the CPU so where it is built
    is where it stays. `nbytes` is kept as a measurement now that nothing
    declares it.

- **`docs/SOLATTN.md` claimed `dense_blocks` ships empty, in two places.** It
  had said so since before 2026-08-26, when `SOL_RECOMMENDED_CUDA` took
  NVLabs' `0-1`; one passage called the knob "unexploited headroom on the
  shipped path" and another pointed at `docs/roadmap.md` for a decision that
  had already been taken. Both corrected, both saying what they used to claim.
  The knob table's `dense_blocks` row now names what each config ships rather
  than only the node's default.
- **`docs/h3_pdd.md` described the wrong mechanism for PDD `end_percent`,
  twice.** It said the value is a build-time lookup that goes stale silently
  when `steps` is hand-edited. That is now true only of non-distilled arms. Its
  1-to-10 evaluation sweep also carried a Sol `end_percent` column, which is
  one value repeated ten times under the new recipe; the column is deleted and
  the one thing it conveyed is said once.

## 0.97.0

### Fixed

- **Fusing a head crashed whenever ComfyUI put the bank on the GPU.**
  `fuse_block` derived its plan on the default device and multiplied it against
  the bank wherever that was, which was fine for as long as the bank lived in a
  closure and never left the CPU. Handing it to ComfyUI as a managed model made
  it movable, ComfyUI moved it to cuda, and the first render after that raised
  `Expected all tensors to be on the same device` from inside the tensordot.
  The plan now follows the stack, so the fusion happens where the weights are
  and a 42 MiB round trip stays off the wire. Not specific to any step count --
  `fuse_block` runs for every block at every count.
  - The audit that followed found the rest of the run-time paths already
    device-aware: the tracker moves its table to the embedding's device, and
    the adaln injection moves all three operands to `x`'s. This was the only
    gap, precisely because it was the only place both operands used to be CPU
    by construction.
  - Graded by a new case, skipped loudly on a CPU-only box. It asserts the
    result LANDS on the bank's device -- fusing on the CPU and copying back
    would also avoid the crash, at the cost of the round trip -- and that both
    devices agree to the bit, since a silent fp64 precision change would be
    worse than the crash it replaced.

- **`head_strength`'s sentinel was documented and never implemented, so the
  node default applied the distilled head correction backwards.** The input's
  tooltip has said "-1.0, the default, means FOLLOW `strength`" since the
  strength split; no code resolved it. The literal reached
  `_FusedHeads.get`, where the master weight is `base + s * (fused - base)`, so
  every head came out `base - (fused - base)`. Any PDD render taken between the
  split and this commit ran inverted heads. Resolved in
  `resolve_head_strength`, a named function rather than two lines inside
  `execute`, so the contract is importable and testable -- it was untestable
  before precisely because it lived only in prose.
- **`build_workflows.py` could not write anything, and had not been able to
  since the `steps` widget landed.** The validator refuses the whole run rather
  than part of it, which is the right design and did mean the entire tree was
  frozen for every lane. The missing widget was `head_strength`, not `steps`:
  the generator already supplied `steps` on both forms. The UI widget list now
  carries six values in required-then-optional order, which is how the
  validator derives it from `define_schema` and is NOT declaration order.
  Reported by a peer session, which is the second reader the repo's own rule
  asks for.

### Added

- **`bench/build_sidecar_node.py`**: assembles the standalone `MiniMaxH3PDDLoRA`
  that ships beside the weights on the Hub, so nobody has to clone a research
  repo to load a LoRA. `pdd_lora.py` imports exactly one module from this pack
  and that one needs only torch, which is what makes a three-file drop-in
  possible at all.
  - Generated rather than hand-copied, because bundling means the node's source
    lives in two places and a copy with no invalidation goes stale silently.
    `--check` re-derives it and reports drift; proven to go red on both a
    modified file and a missing licence, and green after a rebuild.
  - It does NOT prove the bundle matches what is published. That needs a
    comparison against the Hub, which the check cannot do offline.

- **`bench/measure_pdd_step_ladder.py` and its record**: what each legal PDD
  step count buys on the shift-12 grid. Written because published guidance was
  about to rest on the final block alone, which is the one statistic on which
  5, 6, 7 and 8 are exactly tied. The ladder is flat where it was assumed to
  climb -- those four sit within 2% of each other on summed squared step,
  because under shift 12 the early blocks span almost none of the sigma range.
  Only 4 evaluations is materially different, at 1.506x, and its `[8,8,8,8]`
  partition is forced rather than chosen. Also records that 16 and 32 are legal
  by the divisor route while sitting OUTSIDE the trained envelope in the
  narrow direction, which nothing had stated.
- **A case in `check_pdd_head_selection.py` for the sentinel.** It reads the
  default from `define_schema` rather than retyping it, so the two sources stay
  independent, drives the real `_HeadBank` and `_FusedHeads` rather than
  restating their arithmetic, and asserts that the UNRESOLVED default lands
  somewhere else entirely -- without that last part it would pass on a resolver
  returning any constant.

### Changed

- **The shipped graphs write `head_strength` explicitly instead of relying on
  the sentinel.** Same behaviour, and no graph JSON carries a negative number
  that is not a negative scale.

## 0.96.0

### Changed

- **`comfyui_vendor_gaps.md` brought up to date with the audio work.** Gap 7's
  withdrawal is now flagged at the top rather than only in its section, because
  it changes the summary table; gap 5 carries its measurement (15 s against a
  5.167 s target is 786 excess rows) in both the table and the section; gap 15
  is added; and mono is recorded under Settled with "do not re-file this as a
  gap".
- **And the top block says what gap 7 means for this file specifically.** Its
  opening rule is "if this file disagrees with an owner, the owner is right".
  Gap 7 did not disagree with its owner -- it faithfully reproduced an owner
  that was wrong, and a gate agreed with both by verifying the claim against a
  hand-built latent instead of a real encode. Deferring to an owner is not a
  substitute for the owner having run anything, and three artifacts agreeing
  with each other is not three pieces of evidence. Recorded there because the
  file's premise is exactly what failed.

## 0.95.0

### Measured

- **Gap 5 is real, unlike gap 7.** Run at the real entry point against the real
  audio VAE: a 15-second reference soundtrack against the default 124-frame
  target (5.167 s, `audio_t` 207) encodes to `ref_audio_t` 600 -- **1,200 rows
  where 414 were wanted, 786 of them excess**, attended at every sampling step
  and expanding the reference's RoPE span. A 5-second soundtrack fits with none
  to spare. Nothing in core trims it.
- **New gap 15: the reference-audio encode CRASHES under VRAM pressure instead
  of falling back.** `comfy/sd.py:514` defaults `extra_1d_channel` to `None`;
  ACE Step (`:884`) and LTX Audio (`:966`) both set 16; the H3 audio branch
  leaves it unset. So `VAE.encode`'s OOM retry hands a `[B, L, C]` audio tensor
  to `encode_tiled_`, the 2D image tiler, which indexes `shape[3]`.
  `vae.encode_tiled(torch.randn(1, 8000, 2))` raises `IndexError: tuple index
  out of range`.

  **Found by accident, which is the only reason it is recorded.** A 15-second
  soundtrack encoding while ComfyUI held the card hit
  `expandable_segments: memory mapping failed with OOM`, logged the tiled-retry
  warning, and died in the retry. The reference encode runs at graph start, when
  the DiT and text encoder may still be resident, so the conditions are
  ordinary. `comfyui_h3_t2va_trace.md` §14 item 9 had called this "not reachable
  today" on the strength of no graph wiring `VAEDecodeAudioTiled` -- true of
  **decode**, and silent about **encode**, which needs no node because
  `VAE.encode` falls back on its own. Corrected there.

## 0.94.0

### Fixed

- **Gap 7 withdrawn: a mono reference does NOT raise on native core, and the
  gate that said so could never have noticed.** *measured* against the real
  audio VAE (`bench/results/2026-08-29_ref_audio_channels.json`): a mono
  waveform encodes to `[1,32,2,40]` and produces exactly the 80 rows
  `PackedLayout` allocates, with the two latent channels **bitwise identical**.
  `_encode_ref_audio` calls `comfy.sd.VAE.encode`, whose
  `vae_encode_crop_pixels` replicates the channel because the H3 audio VAE
  declares `output_channels = 2, pad_channel_value = "replicate"`. sglang does
  the same with `-ac 2`, so this is parity, not a divergence. Corrected in
  `comfyui_vendor_gaps.md` (summary row, section 7, both tables, the policy
  paragraph) and in `h3_references.md` (Known limitations and the typed-boundary
  paragraph).
- **Core never had the defect, so this was wrong when written rather than
  stale.** `output_channels = 2` and `pad_channel_value = "replicate"` were set
  in `57500fc5`, the commit that added H3 support on 2026-08-03, on a generic
  channel trim/pad mechanism from `6a2678ac` (2025-12-18). The gap was filed on
  2026-08-21, eighteen days later. The retirement contract both this gate and
  `check_reference_contracts.py` case 5c are written to anticipates a *stale*
  claim -- upstream fixes it, the arm flips, retire the file. It has no state
  for a claim that was never true, which is why nothing fired for a week.
- **The escaped instance, which is the part worth keeping.**
  `check_mono_ref_audio.py` asserted the current state so the record could not
  rot — the right instinct — but it verified the claim by hand-building a
  1-channel latent and reproducing `PackedLayout`'s assignment, instead of
  encoding mono audio. That assignment does fail, so the gate stayed green while
  being green about a state the live path cannot produce. It traced
  `comfy/ldm/minimax/audio_vae.py::encode` and stopped one wrapper short of the
  call the node actually makes. **A current-state assertion is only as good as
  its entry point.** Inherited by name into
  `check_reference_contracts.py`'s case 5c, the other check written to the same
  retirement contract.

### Added

- `bench/audit_ref_audio_channels.py` and
  `bench/results/2026-08-29_ref_audio_channels.json` — what core does with 1, 2
  and 6 channels, run against the real audio VAE on CPU. Deliberately an
  **audit, not a gate**: mono is no longer a defect, so there is nothing to hold
  red. It replaces the retired check and carries the finding that check could
  not have produced.

### Measured

- **More than two reference-audio channels is silently truncated to the first
  two.** Six channels also encode to `[1,32,2,40]` — `vae_encode_crop_pixels`
  slices `pixels[..., :2]` when the input is wider than the VAE wants — with no
  error and no warning, so a 5.1 reference conditions on a pair nobody chose.
  Open in core and watched by nothing there; this repo's typed
  `reference_conditioning._prepare_audio` refuses it.
- **A mono source reaches the DiT as exact dual-mono**: the audio VAE treats
  stereo as a batch axis with no cross-channel coupling, so identical input
  channels give bitwise identical latent channels. The two still receive
  different position ids (the `w` grid's two extremes), so they are
  positionally distinct and content-identical.

### Removed

- `bench/check_mono_ref_audio.py`, retired under the contract in its own
  docstring: a succeeding mono arm means upstream fixed the path, which retires
  the file rather than repairing it. The twist is that its mono arm never
  succeeded — see above.

## 0.93.0

### Fixed

- **Retracted "the release ships no fused qkv", withdrawn the same day it was
  written, and reverted the two documents it was used to "correct".** The claim
  came from grepping one of the release's **two** DiT formats.
  `coderef/MiniMax-H3/transformer/` is the diffusers copy
  (`MiniMaxH3Transformer3DModel`, 638 keys, split `to_q`/`to_k`/`to_v`);
  `FL2VA/transformer/` and `Ref2VA/transformer/` are the **native** format
  (`MiniMaxH3DiTModel`, 535 keys) and carry 52 `blocks.N.attn.qkv_proj.weight`
  under key names **identical to ComfyUI's** — `attn.out_proj`, `attn.q_norm`,
  `mlp.fc1`, `adaln_proj.linear`, `final_layer.video_out`, `rope.inv_freq`.
  ComfyUI implements the release's native checkpoint directly; diffusers is the
  one that transforms. `docs/evidence.md` and
  `docs/research/h3_dit_implementations.md` were right as written and are
  restored, each carrying a note that the correction was withdrawn.
- **`comfyui_vendor_gaps.md` carried a claim its owner had retracted a day
  earlier** — that Sol-Attn and the sage kernels have no counterpart in sglang,
  which `sglang_comparison.md` withdrew on 2026-08-28. Corrected, and the
  correction says so, since this is the snapshot-versus-owner failure that
  file's own opening rule exists to catch.

### Added

- **`comfyui_vendor_gaps.md` gap 14: the DiT's fp32 island collapses to bf16
  under the int8 load.** sglang enforces the island with a named frozenset
  (`MINIMAX_H3_FP32_PARAM_NAMES`, plus `rope.inv_freq` as a buffer) and prunes
  `time_embedder.*` from it on a curve checkpoint. ComfyUI declares the same
  intent in a comment and drops it, because `MixedPrecisionOps.Linear` discards
  the caller's `dtype=`. Open, core-side, **enforced by nothing**, and 5-20x
  below the quantization error already present.
- `sglang_comparison.md` gains that section plus a text-encoder precision
  section recording the inverse direction: sglang encodes in bf16, ComfyUI in
  fp32.

### Decided

- **The two sglang documents stay separate.** Asked whether to merge
  `sglang_comparison.md` into `sglang_h3_pipeline.md` to stop drift. They drift
  for different reasons on different clocks — the pipeline walk goes stale when
  *their* tree moves and is fixed by re-reading at a new commit; the comparison
  goes stale when *ours* moves and is fixed by re-deriving against our side. One
  merged file would need both kinds of maintenance and signal neither. Recorded
  in `sglang_comparison.md` so the question is not re-opened from scratch. The
  drift that was actually present was fixed instead, and the clone's move to
  `97781eb7f3` (2026-08-29) is now named in the header.

## 0.92.0

### Measured

- **int8_convrot on the text encoder is worth keeping over the bf16 file, and
  the reason is PCIe, not disk.** Same layer, same box, weights resident:
  int8 -> fp32 dequant with the Hadamard un-rotation is 0.78 ms against a
  bf16 -> fp32 cast's 0.87 ms (both memory-bound on the same fp32 write; int8
  reads half the input), and the fp32 GEMM is 1.87 ms either way. Host -> device
  is **10.87 ms int8 against 21.75 ms bf16**, so transfer dominates compute ~5x
  and the encoder does not fit in 24 GiB in either format. Footprint 25.28 vs
  47.97 GiB for the same 24.38 G elements. The price is 0.88% relative weight
  error. **The dequantization is not a tax and the rotation is nearly free.**
- **`full_precision_mm=True` for text encoders is deliberate, not an
  oversight.** `25022e0b` (2025-11-24) replaced
  `scaled_fp8_ops(fp8_matrix_mult=False, ...)` with it -- upstream's policy is
  that a text encoder stores quantized and computes in full precision.
- **The published release ships NO fused qkv.** 52 `to_q`, 52 `to_k`, 52 `to_v`
  and no `qkv` key in any of its 14 transformer shards. Both fused layouts in
  circulation are repacker conventions, and a split-qkv file cannot load into
  ComfyUI at all -- detection subscripts `blocks.0.attn.qkv_proj.weight`
  unguarded. Corrects `docs/evidence.md` and
  `docs/research/h3_dit_implementations.md`, which both named one of the two
  fused orders "the release order".
- **The bf16 load of the DiT's fp32 island costs 3.7e-4 to 1.7e-3 relative**,
  against the 8.8e-3 the same checkpoint's int8 blocks already carry -- so it is
  5x to 20x below the dominant error term. Two consequences that are not about
  magnitude: `adaln_proj` is a strict regression (F16's 11 mantissa bits to
  bf16's 8), and a bf16-against-int8 checkpoint A/B changes four things at once,
  not one.

### Fixed

- **Retracted a claim this repo made yesterday and could not support.**
  `comfyui_h3_t2va_trace.md` said the encoder's per-forward fp32 dequantization
  was "consistent with this repo's own recorded ~691 s encode". The arithmetic
  was never done: a full streaming pass is ~2 s, all 350 GEMMs ~0.7 s, the
  dequantizations ~0.3 s. The encode's real bottleneck is none of the three and
  is now recorded as unexplained.
- Corrected the decode output buffer for this render: 3.98 GiB, not the 1.30 GB
  a reader had computed for a different frame count.

## 0.91.0

### Added

- **`docs/research/comfyui_h3_t2va_trace.md` -- what ComfyUI's own code does,
  call by call, for one t2va render at 1344x768x345.** Four loaders through both
  VAE decodes, with and without the PDD node: detection and the eleven keys it
  sniffs, int8/convrot representation and where the GEMM actually runs, the
  packed layout and position grid, one forward stage by stage, the block x50,
  the output heads, and section 14's sharp-edge list. It compares nothing --
  `h3_dit_implementations.md` keeps the five-way comparison. Every surprising
  claim in it was re-derived by execution rather than relayed.

### Measured

- **The int8 text encoder never runs int8 arithmetic.** Two independent gates --
  `full_precision_mm=True`, hardcoded for every quantized text encoder at
  `comfy/sd1_clip.py:114`, and the `force_cast_weights` that
  `comfy/sd.py:270`'s `set_model_compute_dtype(torch.float32)` stamps on every
  module -- each make `_use_quantized` false at `comfy/ops.py:1373-1379`. So
  `comfy/ops.py:431-436` dequantizes the full weight to fp32 on **every
  forward** and runs an fp32 GEMM; `int8_linear` is called zero times. The int8
  storage buys disk, host RAM and PCIe volume, and no arithmetic speed. The DiT
  is the same scheme in the same wheel and does take the kernel, because
  `pick_operations` passes no `full_precision_mm` and nothing sets a compute
  dtype on the diffusion patcher.
- **The DiT's fp32 output-head island does not survive the int8 load.**
  `MixedPrecisionOps.Linear.__init__` (`comfy/ops.py:1300-1303`) discards the
  `dtype=` its caller passed and uses the compute dtype, so
  `final_layer.video_out`/`audio_out`, both patch projections and every
  `adaln_proj.linear` load as bf16 -- the last of those from F16 on disk, i.e. a
  loss of mantissa. Verified by executing the load path, not by reading it. The
  comment at `comfy/ldm/minimax/model.py:302` is true of the file and of the
  bf16 checkpoint, and false of the configuration this box renders.
- **Core's PDD head fusion assumes a delta-encoded bank and nothing checks
  which encoding is resident.** `_pdd_head` computes
  `rows[0] + sum(w_k * rows[k])`, correct only if rows 1.. are offsets from row
  0. The published alibaba-pai bank is verbatim -- rows differ from row 0 by
  2.6-5% of a head's own norm -- and against it that formula is 0.77 to 1.00
  relative wrong per block. Not reachable through `pdd_lora.py`, which leaves
  `video_out.weight` its original size so core takes its `n == 1` path.

## 0.90.0

### Added

- **`MiniMaxH3EncoderLoader` (`h3_encoder_loader.py`): ComfyUI's own H3
  text-encoder load, with the checks core does not perform.** Deliberately thin
  -- core's preprocessing is left exactly as it is, and everything added either
  runs after the load or is stamped on the result. It refuses a checkpoint that
  does not exactly populate the model, refuses one whose tokenizer does not
  realise the released special-token ids, and stamps a contract describing what
  core's preprocessing will actually do, so the reference nodes price against a
  named ceiling instead of silently falling back. Serves t2v, fl2va and ref2va
  alike: the guards are mode-independent, and fl2va keyframes reach the encoder
  as image embeds exactly as references do.
- `bench/check_h3_encoder_loader.py` -- its control. Proves the shipped encoder
  passes every guard, that the contract is derived from ComfyUI rather than
  declared here, and that an incomplete checkpoint cannot load quietly.

### Measured

- **Core does not reject an incomplete H3 checkpoint. It detects a DIFFERENT
  ARCHITECTURE from the tensors present.** Dropping the DeepStack vision norm
  sends the load into `comfy/text_encoders/flux.py`, which dies parsing a
  Mistral tokenizer with a `TypeError` about JSON -- nothing in that message
  names the cause. Dropping a mid-stack layernorm instead reports as an
  ordinary missing key. The two failure shapes read nothing alike, which is why
  the loader has both a construction wrapper and an inventory guard; before the
  mutations ran it was not known which tensors produced which.
- **A correct `int8_convrot` artifact reports no missing and no unexpected
  keys**, so the inventory guard costs the shipped encoder nothing. Worth
  stating because core reports unexpected keys at DEBUG level, so that half of
  the comparison is invisible on a normal server.

### Changed

- `h3_awq_encoder.expected_special_token_ids` splits the special-token id
  arithmetic out of `_validate_native_tokenizer`, so the two loaders share one
  rule against two declaration sources -- the AWQ loader checks the selected
  artifact's snapshot, the guarded loader checks the release's own. It stays in
  that module because the standalone build copies it verbatim and may import
  nothing beside it; the dependency runs one way.

## 0.89.0

### Added

- `bench/convert_h3_bf16_encoder.py` — adapts a full-depth HF Qwen3-VL-32B text
  encoder to the ComfyUI H3 encoder: drops decoder layers 50-63, `lm_head` and
  the final norm (22.8% of a bf16 release artifact), and renames
  `model.language_model.`/`model.visual.` into the naming
  `comfy/sd.py::detect_te_model` requires. Shares its drop rule with
  `h3_awq_encoder._drop_source_key` so the bf16 and W4A16 lanes cannot disagree
  about what H3 consumes. Copies tensor bytes verbatim, so surviving weights are
  bit-identical to the source; verifies by running core's detector and by
  comparing the key set against an existing working artifact.

### Measured

- **A full-depth HF-named Qwen3-VL-32B misdetects as `QWEN3VL_8B`, not as the
  H3 encoder.** `detect_te_model` tests
  `model.visual.deepstack_merger_list.0.norm.weight` before the H3 test, and the
  `QWEN3VL_32B` branch of `load_text_encoder_state_dicts` is the one qwen3vl
  branch applying no `state_dict_prefix_replace`. So the rename is load-bearing,
  not cosmetic — this is the 2026-08-23 escape `h3_awq_encoder` already guards
  against in the W4A16 lane, reproduced here on a bf16 artifact.

## 0.88.0

### Added

- `bench/measure_int8_convrot_headroom.py` and
  `bench/results/2026-08-29_int8_convrot_headroom.json` — what a rebuilt
  `int8_convrot` encoder could gain over the shipped one, asked the way the AWQ
  lane asked it and answered on weights rather than on a render.

### Measured

- **The shipped `int8_convrot` encoder is exactly ComfyUI's stock recipe applied
  to an FP32 source, and there is no calibration lever to improve.**
  `TensorWiseINT8Layout.quantize(W.float(), per_channel=True, convrot=True,
  convrot_groupsize=256)` reproduces its int8 values to within a rounding
  handful across every sampled linear. Unlike AWQ, `convrot` is a
  deterministic data-free weight transform: there is no calibration population,
  so the lever the AWQ lane spent itself on does not exist here.
- **The one exposed knob is already at its optimum.** Rotation group size is
  flat across the legal powers of four, and the shipped 256 is the best of them.
- **The rotation earns its place**, and is the only thing in the recipe that
  does: turning it off is materially worse.
- **Quantizing from BF16 instead of FP32 is a real trap**, and the one finding
  here that changes what a rebuild should do: it moves a noticeable share of the
  int8 values and lands measurably worse. Anyone rebuilding this artifact must
  upcast first.
- **What the file leaves unquantized is the vision tower and the embedding
  table**, both BF16, read from the header. The embedding table is the one that
  must not be touched: quantizing it would break the layer-0 bit-identity the
  marker work rests on.

## 0.87.0

### Added

- **`h3_awq_encoder.py` says which config files it bound and what it took from
  each**, once per bind, unconditionally. The still-image processor's file
  (`processor_config.json` or `preprocessor_config.json`, whichever the
  artifact carries, and whether the settings sit nested or flat) with the
  settings handed to the processor; `video_preprocessor_config.json` as read;
  the special-token list in `tokenizer_config.json` with the key it came from
  and the seven H3 markers; and `config.json`'s declared depth, how much of it
  H3 consumes, its storage dtype and its W4A16 fields. The generations this
  adapter accepts are shaped identically and differ only in these files, so
  until now nothing downstream of the bind could say which still-image budget a
  render had run under.
- **A "where each file comes from" section at the top of `h3_awq_encoder.py`**,
  answering the question the module previously only answered by tracing: the
  checkpoint supplies weights and a copy of its own `config.json` in its
  safetensors metadata, and every processor, tokenizer and geometry setting is
  read from a `config/` snapshot selected by matching that metadata. It states
  that the load path never sees the filename, and that `ARTIFACT_SNAPSHOTS` --
  which is keyed by filename -- serves only the static readers, so a renamed
  artifact loads under its own snapshot and is priced under the name's.
  `bench/build_h3_awq_standalone.py` carries the standalone-correct variant,
  which says the configs are embedded rather than read from a directory.

### Changed

- The still-image budget override notice folded into that record instead of
  logging separately. The record names the budget in force and, when it was
  overridden for one CLIP instance, what the artifact still declares.
- `_still_settings` and `_quant_contract` now read through `_still_config_name`
  and `_quant_declaration`, so the report cites the same file choice and the
  same field set the loader enforces rather than a second copy of either.

## 0.86.0

### Corrected by measurement

- **Inter-frame delta does not predict PDD artifact severity, and neither does
  output spatial detail.** Both were registered as accounts this session and
  both are refuted by the same table: across four arms of one scene, the clip
  judged bad sits INSIDE the range of the good ones on both statistics at once
  (second of four by each). No threshold separates it and no monotone function
  of either can, because good arms lie on both sides. The delta account was
  additionally declared confirmed on a two-arm subset and refuted by the third
  arm once it rendered. `docs/research/pdd/2026-08-28_scene_complexity.md`.
- **A specific shot is not the cause either.** The action repeatedly named as
  the failure point appears in an arm that rendered clean, so it is not
  sufficient to produce the failure.
- **A VRAM-mode switch was NOT unobservable, contrary to what this lane
  concluded.** ComfyUI writes its own log under `user/`, with a `.prev` file for
  the previous server session, independent of whether the launcher tees stdout.
  Across both files there are zero lowvram, novram or partial-load switches for
  the whole evening. A reference render proposed partly to work around the
  supposed gap was not needed.

### Added

- `bench/run_shot_count_ablation.py` — renders one scene as arms that drop a
  shot, holding seed, canvas, partition, sigmas and Sol settings. `--length`
  separates total demanded content from per-shot duration, which the original
  design confounds; `--ref` builds a 32-evaluation counterpart at block width 1.
- `bench/score_shot_ablation.py` — delta and spatial detail on one pipeline.
  Deliberately does not import the existing delta helper: that file measures at
  a different resolution and colour handling, and a spatial gradient does not
  survive a resolution change the way a temporal one does, so mixing the two
  silently produces incomparable numbers.

## 0.85.0

### Corrected by measurement

- **"Audio is the thing a low step count costs" was a metric artifact, and both
  streams degrade.** The audio-only reading rested on video staying flat across
  partitions while audio rose. Raw video rel L2 is dominated by the DC term
  every arm reproduces; remove the mean and video contrast falls monotonically
  with partition coarseness, on every frame of the two coarsest arms, while
  video's correlation with the reference moves far less. Both streams lose
  amplitude and only audio's was visible to the chosen metric.
  `bench/results/2026-08-28_pdd_stream_energy.json`, from renders already on
  disk. **A metric that says one stream is fine is a claim about the metric
  until you have checked what it is blind to.**

- **The audio failure is an ENERGY collapse, not a decorrelation**, which is a
  blunter account of the original complaint than any phase or content story.
  Best-fit gain against the 32-evaluation reference falls from 0.308 to 0.046
  across the partitions, confirmed against `ffmpeg volumedetect` independently
  of the analysis script's own decode path. The coarsest arm's audio rel L2 of
  0.992 means nearly silent rather than uncorrelated.

- **Audio rel L2 has no phase gradation and should not be read as a fidelity
  metric.** It reaches its decorrelation ceiling of sqrt(2) at a ONE-frame
  shift -- measured 1.418, against video's 0.181 at the same shift -- so across
  these arms it was mostly reporting the energy loss above. Recorded in the same
  file.

- **No partition experiment can attribute any of this to the audio
  change-of-variable.** Summed drift computed THROUGH that transform and summed
  drift computed in pure video time rank every arm identically, including arms
  not yet rendered. Identifying it needs the transform varied at a fixed
  partition -- and `shift_audio` is not that knob either, since
  `pdd_math.fuse_block` weights each block by `pdd_time_grid(shift_a, ...)`, so
  the audio fused head and the audio adaln grid move with it and the published
  heads end up at sigmas they were never distilled for.

- **Video rel L2 in `bench/results/2026-08-28_pdd_partition_fidelity_362.json`
  was computed in float32 over ~9.6e8 elements and is imprecise in the third
  decimal**, by more than the arm-to-arm gap one of its scored predictions
  turns on. **The ordering is unchanged and nothing in that record is
  withdrawn** -- the scoring survives, the margin behind it was thinner than it
  looked. `grade_pdd_partitions.py` now accumulates in float64.
### Corrected against the paper

- **4 evaluations is most likely a TRAINED configuration, not our
  extrapolation**, and the previous entry said the opposite. Read from
  `arxiv_2607.26004v1` rather than from the vendor's inference script: the
  abstract says varying block size during training is what lets PDD *"support
  sampling with different number of function evaluations at inference"*; Table 1
  lists its NFE as **Variable**; and for **LTX-2.3, the 22B joint video+audio
  model and the closest analogue to H3 in the paper**, `L_min`/`L_max` are
  chosen so the available NFEs are **4 and 8**. The H3 files' `pdd_num_steps 32`
  / `pdd_block_size 4` is exactly consistent with `L_min=4, L_max=8`. What stops
  it being settled is that `L_max` appears in no released metadata; recovering
  it is the one thing that would.

- **The paper also explains why the `[28,2,1,1]` partition lost**, which the
  measurement alone could not. §3.1: training takes block starts at multiples of
  `L_min` and samples offsets below `L_max`. That partition starts two of its
  four blocks off-multiple and makes one 28 wide, so it was outside the training
  distribution in a way fusion loss cannot see — fusion loss reads head weights
  and knows nothing about which spans were trained.

- **And the same rule implies an arm the node refuses.** With 4/8, block sizes
  compose from `{4, 8}`, so `[4,4,4,4,8,8]` is a legal **six**-evaluation
  partition. `resolve_emit_steps` rejects 6 because the SIGMAS output requires
  uniform division; the run-time path takes an uneven partition happily. The arm
  is reachable today via `ManualSigmas` and is unrun.


### Added

- **A probe that moves only the audio carry's evaluation point.**
  `MiniMaxH3AudioCarryProbe` inverts H3's audio change-of-variable, recovers the
  model's raw audio velocity, and re-applies the transform at the block's
  average sigma instead of its start -- holding heads, schedule, adaln grid,
  seed and every weight fixed. Three modes: `off` installs nothing,
  `block_start` is the round trip, `block_mean` is the ablation. **`block_start`
  is arithmetically the identity to ~1e-6 but is NOT a reproduction of `off` as
  a render**, because 1e-6 diverges a sampling trajectory completely; it is a
  second sample at the same knob, which makes it the noise floor the ablation
  has to clear. `bench/check_audio_carry_inversion.py` asserts the inversion and
  separately asserts the ablation moves the output -- neutering the ablation
  leaves the round-trip assertion green, which is the check-cannot-fail shape.

- **A reader that can follow a linked widget** -- `h3_config.resolve_link` and
  `resolve_widget`, with `bench/check_graph_values.py` as their control.

  The defect they close is not hypothetical. In API form a node input is either
  a literal (`"steps": 4`) or a link `[node_id, slot]`. Every static reader here
  read literals, and `check_distill_settings::_literal` returned `None` for a
  link, which its caller reports as a **failure**. So linking any widget a check
  reads turned a perfectly good graph red -- and the pattern is already in the
  tree: 60 UI nodes link `width` from `MiniMaxH3Resolution` today. This is the
  groundwork for constant/primitive nodes feeding values downstream, which would
  otherwise have reddened every affected graph on arrival.

  **Four states rather than a value-or-None**, and the fourth is the point.
  `RESOLVED`; `COMPUTED`, meaning the node produces it at run time and no static
  reader can ever know it (`MiniMaxH3Resolution.width`); `OPAQUE`, meaning THIS
  resolver cannot see it because the table has no row for that class -- the
  graph is probably fine; and `MALFORMED`, meaning the graph is broken. Keeping
  `COMPUTED` and `OPAQUE` apart matters because their fixes are opposite: skip
  it forever, versus add a table row. Collapsing them asserts "no reader can
  ever know this" about a node nobody has described yet.

  `graph_schedule` now reads through it. Verified inert against HEAD's version
  over all 135 graphs: 0 differ. No graph was converted to a primitive and no
  literal was turned into a link -- groundwork only.

- **`bench/check_pdd_head_selection.py::no state outlives its schedule`**, which
  makes a convention into an assertion. See the Fixed entry below.

### Fixed

- **Two PDD guards were going silent, both because state outlived its render.**
  `_StepTracker` is a mutable object held by the ModelPatcher, and ComfyUI's
  execution cache keeps that across prompts in a session. `_shift_checked` (a
  one-shot latch on the new shift guard) and `warned` (the boundary warning's
  latch, set in `__init__` and never reset) both survived into the next render:
  a passing graph set the latch and the failing graph queued next skipped the
  check and completed. Neither produced a wrong picture -- both produced silence
  where a guard was due, which `docs/checks.md` already calls worse than no
  check. Nothing in the suite could have caught either.

  All sixteen attributes were then enumerated: eight are immutable load-time
  configuration, seven are per-render and every one is reset by `_adopt`, and
  `_key` must survive because it is the schedule identity `observe` compares
  against. The new case asserts the mechanical form of that -- assigned outside
  `__init__` implies assigned in `_adopt` -- statically, so it catches the field
  somebody adds later rather than the fields that exist today.

### Recorded from two reference packs

- **Audio is the first thing a low step count costs, and there is a known fix.**
  `coderef/ComfyUI-H3-AudioRefine` states the symptom this lane opened with --
  "4-step video is acceptable but 4-step audio is not" -- and the reason is
  structural: audio is ~400 rows against video's ~37,000, about 1% of the packed
  sequence, so a distillation gives it up first and least visibly. The fix is a
  per-stream `denoise_mask` that freezes video and runs audio-only steps, taking
  the MODEL from BEFORE the LoRA so those steps use undistilled weights. It
  needs nothing from the PDD head schedule, which is what makes it applicable
  here. Not run; the composition with our patches is untested.

  **The supporting measurement is length-limited, flagged 2026-08-28.** The
  partition grading behind "audio diverges about twice as far as video" ran at
  39 frames, where the packed sequence is 12,226 rows -- 62 below Sol's
  `min_tokens` of 12,288, so Sol was inert and no arm ran the production
  attention path. The graph also carries the market prompt
  `docs/prompt_audit.md` verdicts `rewrite`. The ORDERING survives, since every
  arm shares the confound; the magnitudes and the audio-versus-video ratio do
  not. `bench/grade_pdd_partitions.py` now defaults to 362.

- **A sixth PDD implementation, and the strictest.**
  `coderef/comfyui-minimax-h3-audio-T8/pdd_advanced.py` raises on anything but
  the exact 8-evaluation schedule and separately verifies video shift 12 to
  5e-6 -- independent convergence on the run-time shift guard added this
  release. Across six implementations the authors give no step knob, T8
  refuses, UtilsCollection allows the trained width and twice it, Mamad8
  exposes `steps`, and ours allows any divisor and warns past 2x. **We are at
  the permissive end.**

### Corrected

- **`denoise < 1.0` needs no `BasicScheduler` fallback**, and the previous
  release said it did. `SIGMAS(8)` -> `SplitSigmasDenoise(0.5)` -> `low_sigmas`
  is `torch.equal` to `BasicScheduler(simple, 4, denoise=0.5)` and renders
  pixel-identically at a matched seed. The composed form is the better of the
  two, because every entry of the emitted vector IS a block boundary, so an
  index split lands on the grid by construction rather than by
  `int(steps/denoise)` happening to come out even.

- **The three-instances claim about the tracker was two.** The 2026-08-26
  head-selector defect was a quantising table lookup -- a precision bug that
  produced a wrong picture -- and does not belong with the two latches, which
  produced silence. Different defect, different fix.

## 0.84.0

### Added

- **`docs/custom_node_gaps.md`** -- the companion to `comfyui_vendor_gaps.md`.
  That file asks how native ComfyUI differs from the release; this asks what
  the nodes in this pack do end to end, and where that differs from sglang,
  LightX2V, DiffSynth-Studio, diffusers and native ComfyUI. Carries the node
  inventory by class -- load-bearing, convenience, instrumentation, which is
  the cut that matters, since six registered nodes are wired by no shipped
  graph and only two of those are merely superseded -- the three canonical
  dataflows, the reading traps in the shipped JSON, and its own
  enforced-by-nothing table. A snapshot that defers to owners, same standing
  as the file it accompanies.

- **`docs/research/pdd/pdd_implementations.md`** -- our PDD against the four
  other implementations (the alibaba-pai source we converted from, the core
  PR, `ComfyUI-UtilsCollection`, and a third-party pack), plus how the lane
  reached its current shape. **No other engine implements PDD**: diffusers,
  LightX2V, DiffSynth and sglang were each searched.

- **`docs/wiki/`, with `bench/build_wiki_index.py` generating its index.** An
  entry point that routes rather than restates. The index is DERIVED from
  CLAUDE.md's own routing tables, so a blurb edited there cannot drift from a
  copy here. Two written pages the generator never touches: `references.md`
  (what each `coderef/` checkout implements for H3, what has been compared
  against it, at which revision, and what it is not evidence of) and
  `stages.md` (per render stage: our code, its owner, its guard, the
  implementation to compare against). `--check` refuses a stale index and was
  verified in both directions.

### Found, not fixed

- **The Turbo-SLA LoRA adapts the token refiner as well as the fifty DiT
  blocks; `MiniMaxH3SLARouter` patches the blocks only.** Read off the
  artifact's own key set. `docs/open_experiments.md` #20 named this gap
  without evidence the refiner mattered to the distillation; it does.

- **The Sol node every shipped graph wires is this repo's own `vendor/`
  file** -- the sibling pack symlinks to it -- while that pack's docstring
  still explains why the file is not vendored here.

- **Two three-against-one divergences**, both reachable today: the reference
  view handed to the text encoder (sglang, DiffSynth and diffusers all feed
  one prepared tensor to both towers), and video VAE precision, where the node
  that would close it exists, defaults the way the other implementations run,
  and is wired nowhere.

- **A matched-seed A/B is not matched on audio when the arms differ in canvas
  or length.** Both streams draw from one generator consumed in order, so the
  audio noise depends on the video latent's element count. Recorded in
  `docs/wiki/stages.md`; `docs/eval_comparison.md` owns the A/B process and
  does not carry the caveat.

- **The special-tokens research subtree is reachable by no markdown link.**
  Surfaced by the wiki generator's report. Reachable by prose mentions only,
  which link-following navigation cannot use.


## 0.83.0

### Changed

- **`MiniMaxH3PDDLoRA` emits its own `SIGMAS`, and every shipped non-split PDD
  graph now samples from it instead of a `BasicScheduler`.** The dependency ran
  the wrong way: every knob that can put a distilled render off its own grid --
  `scheduler`, `steps` -- lives on `BasicScheduler`, which sits DOWNSTREAM of
  every model-patch node. So the PDD node could only ever observe the schedule
  after the fact and report on it, and three separate footguns followed. A
  scheduler that is not `simple`, a step count that does not tile the 32-point
  grid, and evaluation off the block boundaries entirely were each caught, if
  at all, by a static check over SHIPPED graphs -- leaving a hand-edited or
  hand-built graph with nothing at all.

  Emitting the schedule inverts that. The sampler steps at exactly the
  boundaries the heads were fused for, there is no scheduler widget left to get
  wrong, and off-grid is no longer expressible.
  `silveroxides/ComfyUI-UtilsCollection` reached the same design from the other
  end -- its off-grid error tells you to use its SIGMAS output; this makes that
  the wiring rather than the advice.

  **Numerically inert on every shipped PDD graph, which is why it could be
  done at all.** The node emits `1 - pdd_time_grid`, which IS the plain shifted
  schedule for the block count, and that is bit-identical to
  `BasicScheduler(simple, N)` at 2, 4 and 8 steps on shift 12 and shift 6. At
  16 the two sit ~2e-3 apart, because `simple` reads a 1,000-entry table and
  `1000 % 16 != 0`; the closed form is the more correct of the two there and no
  PDD graph runs 16. No render moves; the ways to get one wrong go away.

- **The node's new `steps` input is inert at its default of 0**, which is what
  keeps a deliberately off-grid arm working. 0 means "the file's own count" and
  never refuses; a non-zero request must tile the grid and RAISES otherwise,
  because at such a count no on-grid schedule exists and there is nothing
  honest to emit. The first version of this raised unconditionally and would
  have refused a 6-step PDD render that was in flight while it was written.

- **A step count past twice the file's distilled block width now says so.**
  `pdd_block_size` is 4 in both shipped files, so 4 sampling steps is exactly
  the 2x edge and 2 steps is past it.
  `silveroxides/ComfyUI-UtilsCollection` refuses anything wider outright; this
  warns instead, because the 2x arm is one the repo deliberately renders and
  refusing would break a live experiment. Nothing said so before, at any count.

### Verified on the card

- **The rewiring is inert end to end**, not only in the sigma arithmetic. Eight
  settled runs at 1344x768 x 39 frames, matched seed -- four on the old
  `BasicScheduler` wiring and four on the new SIGMAS wiring -- are
  pixel-identical, and the same holds at a second step count. Changing the
  node's `steps` does move the pixels, which is what shows the output is
  actually consumed rather than decorative. `steps=6` is refused at the node
  with no sampler in the executed list, and the trained-envelope warning fired
  in a real render for the first time.

  Two instrument failures on the way there, both recorded in
  [`docs/h3_pdd.md`](docs/h3_pdd.md): the comparison was first run over `.png`
  file bytes, which embed the prompt JSON and so differed on arms whose frames
  were identical -- the same trap that doc already recorded for `.mp4`
  containers; and a render here has a **warm-up transient**, so the first run
  after a state change differs from the value that configuration settles on. It
  affects both wirings equally. Read against a single matched pair, that looks
  exactly like a regression, and it was read that way here before interleaved
  repeats separated the two.

### Fixed (found while running the full suite, unrelated to the above)

- **The UI generator emitted a widget the schema had dropped.**
  `build_workflows.py` still passed a trailing `True` for
  `MiniMaxH3Conditioning`'s retired `vendor_tokens` slot -- in the same branch
  whose comment already said the slot was removed. 24 shipped UI graphs carried
  the surplus. The sibling `MiniMaxH3ReferenceConditioning` branch had its copy
  removed when the schema changed and this one had only its comment updated:
  "editing the generator is half the change", inside the generator.
  `check_workflow_schema.py` was correctly red the whole time.
- **`h3_config.py` restated a claim retracted on 2026-08-14** -- that H3's DiT
  has only a single attention site, the smaller calls being SageChainAssert's
  probes. (The retracted wording is deliberately not quoted here;
  `check_retraction_consumers.py` grades exactly that phrase and went red on an
  earlier draft of this entry, which is the check working.)
  `docs/SOLATTN.md` owns the correction: one source line but **52 modules**, and
  the refiner calls are model code, not instrumentation. Verified at source
  (`RefinerBlock` and `DiTBlock` both instantiate the same `Attention`). The
  `min_tokens` conclusion the comment was supporting is unaffected and now
  rests on the right reason.

### Documented

- **What the PDD authors actually ship, read from their source rather than
  inferred.** `apply_pdd_lora` derives `nfe = num_steps // block_size` from the
  file's own config and returns it; there is no step parameter in their API.
  Their config is 32/4, so `nfe` is 8; `predict_ref2v.py` calls the pipeline
  with `num_inference_steps=nfe + 1`; the README says "Official 8 Step Acc
  LoRA" and its comparison grid puts that against LightX2V's **4-step turbo**.
  So 4 steps is the turbo lane's number and every 4-step PDD arm here is our
  own extrapolation to twice the trained block width -- never something the
  authors shipped or measured. **Unvalidated upstream is not the same as shown
  worse**, and an earlier draft of this entry blurred them: three of the five
  other implementations allow four evaluations, the one that refuses does so on
  a stated fidelity-to-the-reference policy rather than a measurement, and the
  weight-space fusion loss measured here does not grow from width 4 to width 8.
  `C2_pdd4_nosol` against `C2_pdd8_nosol` is the only thing in this repo that
  can settle it, and it is unjudged.

- **A generated sweep of 1 to 10 evaluations** in [`docs/h3_pdd.md`](docs/h3_pdd.md):
  which counts tile the 32-point grid, the block width each gives, where that
  sits against the trained width, whether the node emits or raises, and the Sol
  `end_percent` each would take. Three things move when the step count changes
  and two are invisible: the block width (obvious), `nfe` (does **not** react --
  it is an independent override), and Sol's `end_percent` (a build-time lookup
  that goes stale if steps is edited on a loaded graph). Only divisors of 32
  have a schedule at all, so most of the sweep raises, correctly.

  One thing the sweep corrected on inspection rather than confirming: the `6`
  in `SOL_END_PERCENT_BY_STEPS` is a **turbo** count, not a PDD one. Every
  6-step arm this repo ships is a 768p turbo graph, and no PDD graph can run 6
  because 6 does not divide 32 and the node refuses. Reading that table alone
  suggests otherwise, which is why it is now said out loud.

### Known gap, stated rather than closed

- **The shift became a second place the schedule is decided.** The PDD node is
  upstream of `MiniMaxH3SigmaShift`, so it emits its FILE's shift while the
  model follows the widget. They agree on every shipped graph;
  `check_pdd_sigmas.py` asserts that, and it is a control over shipped graphs
  rather than a fix -- a hand-edited graph is not reached. Removing the widget
  from PDD graphs was measured inert (two runs pixel-identical with the node
  deleted; `comfy/supported_models.py` declares 12.0/3.0 as the H3 class
  default) and deliberately NOT done: two cheap static checks read the shift off
  that node, so deleting it would push them onto the file's metadata or onto a
  duplicated constant. That trades one duplication for a worse one.
  [`docs/h3_pdd.md`](docs/h3_pdd.md) carries the reasoning and names the clean
  fix, which is to move the node downstream of the shift.

### Added

- **`bench/check_pdd_sigmas.py`**, the control for both claims above. Grades
  the emitted vector against **ComfyUI's own `comfy.samplers.calculate_sigmas`
  over `ModelSamplingAV`** rather than a value it computes, asserts the
  exactness precondition instead of assuming it (so a graph shipping outside
  `simple`'s exact regime goes red rather than quietly widening a tolerance),
  round-trips the output through `schedule_knots` to confirm the sampler steps
  where the heads were fused, drives `resolve_emit_steps` directly for the
  refusal and inertness cases, and fails if any shipped PDD graph still carries
  a `BasicScheduler`. Its first two refusal cases restated their own condition
  and could not have failed; `resolve_emit_steps` was lifted out of `execute`
  so they could drive the real code instead.

### Fixed

- The node's load line and `docs/h3_pdd.md` claimed a cross-partition file
  "falls back to the runtime adaln injection, which is correct on any pruned
  base". **It is not, for either file this repo ships.** `--pruned` deliberately
  pops the raw `h3_pdd.adaln.*` tensors and the `h3_pdd.silu_temb_grid` (about
  40% of the file, dead on a pruned base), so the fallback path reads keys that
  are not there and raises `KeyError` -- after logging that it is doing the
  correct thing. Unreachable today, because the head guard refuses a
  cross-partition file first; it matters because the 2026-08-27 handoff used
  that same claim to argue the adaln "already takes care of itself" if the head
  guard were relaxed. It does not.

## 0.82.0

### Changed

- **Every shipped reference graph now gives Qwen3-VL a 512-short-edge view**
  (`h3_config.REF_QWEN_SHORT_EDGE`), where before it saw whatever the video VAE
  encoded. Reference tokens land in the TEXT segment ahead of the prompt, so
  they compete with it rather than merely costing sequence. Under the v1 encoder
  every reference clamped to ~290 merged tokens and this could not bite; v2
  shipped the same day declaring the release's own bounds, and two
  2048-short-edge references then cost 9,408 tokens there against a
  ~1,000-token prompt -- leaving the prompt 9.5% of its own segment. At 512 the
  view is 592 tokens and the prompt is back to 63%, while the DiT keeps every
  one of its 9,408 reference rows.

  Observed once: a two-speaker scene at the old default rendered with the
  dialogue attributed to the wrong subject, and the binding lives in those
  prompt tokens. **The constant says in as many words that it is a prior and
  not a measurement** -- 63% is v1's ratio, and v1's ratio was an accident of a
  snapshot's pixel bounds rather than a number anyone chose. The arm that would
  move it is named there.

  34 reference graphs took the new default. The three `h3_probe_refview_*`
  ablation arms were untouched without needing a carve-out, because they were
  written to pin their own values -- which is the reason that ablation still
  means what its arm names say.

### Fixed

- **`qwen_short_edge`'s tooltip and its geometry docstring described a knob that
  only grows.** `qwen_view_size` has no `min(1.0, ...)`, so it shrinks as
  readily, and under v2 shrinking is the direction that matters. Both now say
  so, with what it costs and what it buys. `allow_upscale`'s tooltip gains the
  distinction the two knobs acquired when v2 made them separable: it governs
  what the DiT sees, `qwen_short_edge` governs what the prompt competes with.

## 0.81.0

### Changed

- **`min_tokens` 4096 -> 12288 on the shipped Sol recipe**, adopting the node's
  own default. The gate is not Sol against dense torch: below the threshold Sol
  declines and the call falls through to whatever override is installed, which
  on every graph here is sage. Verified in `vendor/sol_attn_minimax.py`'s
  `dense()` and in the render log. Since sage is about 2.7x ahead of torch's
  flash backend on this shape, the crossover sits higher than it would against a
  naive baseline -- and `SOL_CUDA_DEFAULTS` had already recorded that upstream
  puts it near 12k and that 4096 "engages Sol-Attn in the regime where it costs
  time". Deference, not evidence, and the comment says so along with what would
  overturn it. Changes nothing this repo renders; it closes the one reachable
  gap at ~22 frames.

### Fixed

- **`bench/convert_pdd_lora.py` exited non-zero on every `--pruned` run.** It
  read `h3_pdd.silu_temb_grid` unconditionally when reporting, but that key is
  emitted only on the unpruned path, so the converter saved a correct artifact
  and then died with `KeyError`. Both shipped LoRAs are pruned conversions, so
  that was every real run. No render was ever affected -- the write is complete
  before the report, and both artifacts were re-verified against the published
  stack at 0.0 the same day -- but a correct conversion looked like a failed one.
- **Two leftovers from the retired precompute path in the same file**: the
  docstring documented `h3_pdd.head.*`, a key the bank replaced, and `fuse_heads`
  was imported without being called. `pdd_math.py`'s module docstring had the
  matching staleness, describing the converter as baking `fuse_heads` output and
  the node as selecting with `block_bounds`; neither is what either does now.
- **The `min_tokens` no-op claim contradicted its own arithmetic**, in
  `h3_config.py` and in `docs/SOLATTN.md` which owns it. It argued from S = 7,194
  at 22 frames being "already above 4096" -- but 7,194 is below 12288, so the two
  thresholds disagreed exactly there. The claim holds only for the lengths this
  repo renders, and now says so. Closes the question raised 2026-08-26.

### Added

- **`start_percent` is annotated with its cost**, since it remains unmeasured at
  any value ever: it forces the top of the trajectory dense at a flat 25% of
  evaluations at every step count -- 4 of 16, 2 of 8, 1 of 4 -- because it is a
  fixed fraction of a schedule uniform in base sigma.

## 0.80.0

### Fixed

- **`docs/h3_pdd.md` still described the design the 2026-08-27 rewrite
  replaced.** Re-derived against `pdd_lora.py`, `pdd_math.py`,
  `bench/convert_pdd_lora.py` and `bench/check_pdd_head_selection.py`, treating
  the code as the authority. Nine corrections, of which three were claims a
  reader would have acted on:

  - It said the converter "collapses the 32-head stack offline". It does not,
    and has not since the heads began fusing per block: it ships the bank
    verbatim, because collapsing it pins a step count into the artifact.
  - It said `pdd_math.block_bounds` is "shared by the converter and the node".
    Neither imports it any more. The shared implementations are
    `pdd_time_grid` and `fusion_plan`; `block_bounds` is the closed form the
    checks grade `schedule_knots` against.
  - Its comparison against Comfy-Org/ComfyUI#15908 claimed we "fuse at load"
    and "refuse a count that does not divide 32". Both stopped being true the
    same day, and the second contradicted a table three sections earlier.

  Also: the guard table against `silveroxides/ComfyUI-UtilsCollection` now
  records two rows adopted and one partly, rather than one; the boundary
  embeddings are described as built at every grid point and indexed by knots,
  which is the reason they can survive a mid-graph schedule change; and the
  header says the document was revised rather than implying one 2026-08-26
  reading.

### Added

- **`docs/h3_pdd.md` gains the reasons behind three decisions it previously
  only stated.** Why the node refuses to stack on the output heads rather than
  chaining (chaining is right for the observer patch and impossible for the
  swap: two things cannot both produce one tensor); why an enlarging-weight
  approach leaks state across graphs where a forward patch cannot (the shape
  change outlives the graph, because ComfyUI caches the patched model, which is
  how an enlarged bank broke the next render on 2026-08-27); and why euler is
  required by PDD rather than merely preferred by the repo's new distilled-arm
  default.
- **A note that the 2026-08-26 render table's instrument was weaker than two of
  its claims.** Those md5s are of the `.mp4` container and the graphs embed the
  workflow, so identical implies identical frames but "differs" establishes
  nothing. The bit-identical row stands; the two "differs" rows keep their
  conclusions and lose their evidence. Nothing re-run, no rows removed.

## 0.79.0

### Added

- **The two nodes that do the most invisible work now explain themselves in the
  graph.** `MarkdownNote` panels sit directly above `MiniMaxH3PDDLoRA` and
  `SolAttnMiniMax` in every UI graph that wires them, covering only what the
  widgets cannot say: that PDD reads its step count off the sampler's schedule
  rather than from the `nfe` widget, that it patches three surfaces and swaps
  the output heads every step, and that it refuses a wrong-partition file or a
  second owner of the heads; and that Sol's `end_percent` is derived per step
  count by the generator, so editing `steps` by hand leaves it stale with
  nothing at run time to say so. `MarkdownNote` is in `_UI_ONLY`, so neither
  reaches the API form.

### Fixed

- **Four pieces of generated text still described yesterday's sampler defaults.**
  `_NOTE_TURBO_OWNER` said the owner recipe moves **three** knobs from the
  vendor row; the sampler stopped being one of them when euler became the
  default for every distilled arm, so it moves two. The comment above that
  graph still called out a one-widget `er_sde -> euler` difference that no
  longer exists. `_NOTE_NODES` showed `tau=1.3` in its sample `[sol_attn]` log
  line, a value the config stopped shipping on 2026-08-20. And the PDD block
  comment framed euler as a deviation from "the repo's er_sde default" when it
  is now the default -- rewritten to say PDD required it before the policy and
  would require it without one.

  Checked and deliberately left alone: `h3_probe_euler` and its twin. That pair
  is a **base** workload, where `_distill` does not apply and `er_sde` is still
  the default, so the euler-against-`er_sde` comparison it exists for is intact.

## 0.78.0

### Changed

- **Every distilled arm samples on euler/simple.** Owner decision. `h3_config`
  already carried the argument -- a distilled model is trained so one Euler step
  from sigma_i lands at sigma_i+1 -- and had deliberately declined to apply it,
  keeping `er_sde` as the default and euler as a probe because the argument was
  an argument and not a measurement. That is reversed: at 4 and 8 evaluations the
  final step covers the largest jump in the schedule and a sampler that re-noises
  has no step left to recover. Nine turbo graphs moved off `er_sde`; every PDD
  graph already carried euler, and for PDD it is required rather than preferred,
  since a fused head IS the block's mean velocity and the paper's Algorithm 1
  names an Euler step as its consumer.

  Applied by the new `h3_config.DISTILL_SAMPLING` through
  `build_workflows._distill`, keyed on whether a graph wires a distillation LoRA,
  rather than by each call site remembering -- which is how the turbo arms ended
  up split across two samplers while every PDD arm passed `sampler_name="euler"`
  by hand. A `sampler_name=` at a call site still wins, so a deliberate deviation
  stays possible and stays visible.

  **Not changed: `h3_probe_turbo_768p_owner`**, which runs euler with `beta`.
  That graph is the owner's own recipe with a recorded sha, and the scheduler is
  the deliberate part of it.

### Removed

- **`h3_probe_turbo_euler` and `h3_text_to_video_turbo_768p_euler`.** Both existed
  to name the euler-against-`er_sde` difference; with the defaults moved they were
  duplicates of their own baselines. Owner decision to retire rather than invert.
  The two documents that cited them for being the euler arm now cite the
  baselines, which are euler.

### Fixed

- **Two checks graded a PDD graph's step count against the FILE.**
  `check_distill_settings` asserted `steps == pdd_nfe` and `check_distill_grid`
  compared sigmas against `block_bounds` for the file's count. Both were correct
  while the graph carried an `nfe` widget and both went red on correct 4-step arms
  the moment it stopped. They now take the sampler's `steps` as the evaluation
  count, require it to divide the grid for a shipped arm, and treat a non-zero
  `nfe` as an override that must agree with the sampler. Both new assertions were
  shown red by deliberate violation -- 5 steps against a 32-point grid, and an
  override of 8 against a 4-step sampler.

## 0.77.0

### Changed

- **`MiniMaxH3PDDLoRA` derives the step count instead of asking for it.** The
  block extents come from `transformer_options["sample_sigmas"]` at run time,
  mapped back to grid indices through the new `pdd_math.base_sigma` /
  `schedule_knots`. The `nfe` widget it replaces had to be kept equal to
  `BasicScheduler.steps` by hand, with only a warning behind the requirement --
  and that warning fires after sampling has started and had never fired in a
  real render. `nfe` survives as an override that forces uniform blocks; every
  shipped graph now carries 0, so the step count is entered once, in the
  scheduler.

  Verified end to end at 1344x768 / 362 frames / 4 steps: the node logs the
  file's own count at load, then `4 evaluations ... derived from the sampler's
  own sigma schedule, blocks uniform width 8`, and every step reports its block
  0.00000 from its boundary.

  Reaching that dict needs `diffusion_model.forward` patched to observe it and
  delegate. Sol-Attn composes only with `.forward` patches whose owner segment
  contains `attn`, so it leaves this one alone -- checked against
  `vendor/sol_attn_minimax.py`, not assumed.

- **Heads fuse per block on first use rather than `nfe` of them at load**, since
  which blocks a render visits is not knowable until the schedule is. Still the
  paper's one-fused-linear-per-block; a render visits at most `nfe` spans and
  each is fused once. `pdd_math.fusion_plan` takes an end index rather than a
  width, and `fuse_heads` is now a loop over the new `fuse_block`. Verified
  bit-identical to the previous arithmetic on both streams, weight and bias, at
  every step count the grid divides by.

### Added

- **`MiniMaxH3PDDLoRA` raises on a partial patch-key match.** `add_patches`
  returns only what it found in the state dict; a shortfall against what
  `load_lora` resolved now raises and names the first unmatched keys. This was a
  row in `docs/h3_pdd.md`'s **Enforced by nothing**; adopted from
  `silveroxides/ComfyUI-UtilsCollection`, which had the guard while we had the
  admission.
- **`bench/check_pdd_head_selection.py` drives the tracker through `observe`
  with a real sigma schedule**, so the step count is derived the way a render
  derives it rather than handed in. New cases: the derived knots equal what the
  widget computed at every divisor, an uneven schedule (5 steps ->
  `[0, 6, 13, 19, 26, 32]`) is taken and reported rather than refused,
  `denoise < 1.0` starts partway down the trajectory, and the `nfe` override
  overrides.
- **`docs/research/pdd/three_pdd_implementations.html`**, comparing
  `silveroxides/ComfyUI-UtilsCollection`'s PDD node against ours and Kijai's.
  A third independent agreement on the conversion arithmetic, and a guard matrix
  where they are ahead of both of us.

## 0.76.0

### Fixed

- **`MiniMaxH3PDDLoRA` would have died on the first sampling step the day core
  learns PDD.** Comfy-Org/ComfyUI#15908 widens `FinalLayer.forward` to
  `(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)`. We
  object-patch that method, which replaces it outright, so our four-parameter
  replacement would drop three arguments the stock forward now requires. The
  patch forwards extras verbatim now. Our converted file leaves
  `video_out.weight` its original size, so a merged core takes its `n == 1`
  path and our output-linear patches still own the head swap.
- **`bench/compare_pdd_conversions.py` had stopped comparing anything.** It read
  Kijai's head bank by the literal key `final_layer.video_out.set_weight` and
  died with a `KeyError` when his files were re-uploaded in a different
  encoding on 2026-08-27 -- an assumption that had only ever met one
  implementation. It now branches on which keys are present and reconstructs
  the bank through the same arithmetic core uses for the
  `lora_up`/`lora_down`/`reshape_weight` path, recording which encoding it saw.
- **`docs/h3_pdd.md` described a partition guard that had been replaced.** It
  said the node refuses a mismatched sha256; the node compares
  `h3_pdd.base_video_out` by relative Frobenius distance, because the hash
  version fired on the CORRECT checkpoint after ComfyUI cast it on load. Its
  adaln table also carried an `ours` column from a run that predated the
  conversion it described.
- **`bench/check_pdd_head_selection.py`'s docstring still described the snap
  tolerance and `step_for_t`**, both deleted when the tracker moved to matching
  the boundary embeddings directly.

### Added

- **`docs/research/pdd/` holds the PDD mechanisms drawn against Kijai's**, traced
  through the four shipped PDD graphs: the shared graph spine, load time, one
  sampling step, and the 32-interval grid at 8, 4 and an off-schedule count.
  A teaching surface -- `docs/h3_pdd.md` owns every number on it, nothing
  generates it and no check reads it, and its README says so.
- **`bench/compare_pdd_conversions.py` now covers both streams and the bias
  bank**, and records a sha256 of every input. Only the video weight was
  compared before, so the audio head -- its own shape, its own shift, its own
  entry in his file -- rested on the video result. The hashes exist because
  those artifacts are re-uploaded in place: a record naming only a filename
  cannot say what it read, and one 2026-08-26 record turned out to describe a
  converted file that was rebuilt sixteen minutes later.
- **`bench/check_pdd_head_selection.py` section 3: the `final_layer` patch is
  arity-transparent.** Asserts the patch accepts today's four arguments and the
  seven #15908 introduces, forwarding extras verbatim. Needs no checkpoint, so
  it runs when the rest SKIPs. Graded by pinning the patch back to four
  parameters, which also exposed a defect in the case itself: it raised
  `TypeError` past `check()`'s `AssertionError` handler and aborted the run
  instead of reporting a named FAIL.

## 0.75.0

### Added

- **`bench/measure_marker_epsilon.py`: is the frozen DiT sensitive to the seven
  H3 marker rows at all?** The marker record's three instruments are
  encoder-level, render-level and prefix-attention; none answer the cheap
  question that gates the expensive one. This compares the DiT's denoised
  prediction across marker arms at fixed noise, latents and sigma -- the
  conditioning analogue of `grade_sage_on_capture.py`, and the second
  controlled comparison this repo can make about a numerical knob rather than
  about a rendered sample. Carries a null row (same arm, re-encoded and re-run;
  must read exactly 0.0), a stripped-text scale row and an unrelated-scene
  ceiling, because a relative L2 with no ladder has no units a reader can act
  on. First record: `bench/results/2026-08-27_marker_epsilon.json`.

### Fixed

- **`marker_arms.mean_init_rows_clip` could not run against a card-resident
  encoder.** Its mean accumulator was built on the default device while the
  embedding table was on CUDA, so it raised "found at least two devices" --
  under the very condition its own comment anticipated. Every fixture it had
  met was CPU-resident, which is why nothing caught it until a real encoder
  did. `MiniMaxH3MarkerArm` would have hit the same wall in a live graph.

## 0.75.0

### Added

- **`bench/check_graph_discovery.py` says when it is dormant.** Parking the
  image lane took `GRAPH_DIRS` to a single directory, so `graph_paths()` began
  returning exactly what the non-recursive glob it exists to prevent returns --
  130 files either way, with only `include_bench` adding anything (3, under
  `workflows/bench/`). Nothing said so, and "49 check(s) audited" reads as
  coverage to anyone who does not go and measure it. It now prints which case
  it is in, and both branches were exercised rather than one assumed.

  It stays rather than being deleted: `graph_paths()` re-arms the moment a
  directory returns to `GRAPH_DIRS`, and deleting it would silently
  reintroduce the 2026-08-16 defect where a documented one-directory
  invocation priced 20 graphs and missed 8. Dormant is not broken. The problem
  is a green implying coverage it is not providing, which is the same disease
  as a red on correct state.

  The same check now states the axis it has never covered: it audits **which
  files** a scan sees, never **which fields** it reads. A scan can route
  through `graph_paths()` correctly and still read only `inputs`, missing every
  UI graph's `widgets_values` -- which happened on 2026-08-27 in a bench scan,
  and nothing here would have caught it. Recorded as the escaped instance
  CLAUDE.md requires before anyone builds a second instrument for it.

- **A "Settled, and re-derived from scratch anyway" section at the top of
  `CLAUDE.md`.** Owner observation, and it was correct: established H3 facts
  keep arriving as fresh discoveries. Two sessions did it on 2026-08-27 alone,
  and one of them (mine) built a confident argument that
  `bench/results/2026-08-21_h3_token_embeddings.json` had already refuted six
  days earlier -- that the seven H3 marker rows are the trained spelling. They
  are indistinguishable from the untrained padding tail in every artifact
  measured, including an upstream Qwen3-VL the H3 work never touched.

  The cost of the pattern is not repeated work. It is that a session
  re-deriving half of one of these reaches a confident wrong conclusion and
  acts on it. The section carries the four that recur -- the untrained marker
  rows, core's derived-not-declared token ids, the guide-trained prompt
  structure, and 1344x768 as the trained canvas -- each pointing at its record
  rather than restating it.

### Changed

- **The single-frame image path is parked.** Owner decision: it was an
  experimental lane resting on a temporary patch to a module ComfyUI owns, and
  it is not worth carrying for now. `workflows/image/`, `single_frame.py`,
  `bench/check_single_frame.py` and `bench/bench_image_edit_refs.py` moved to
  `archive/`; `h3_config.GRAPH_DIRS` dropped `image`; the generator still
  defines `_image_graphs()` but its one call site is commented out, so
  un-parking is that line plus reversing the moves.
  [`docs/h3_image_editing.md`](docs/h3_image_editing.md) carries the record and
  everything below its banner is now past tense.

  **A consequence worth stating on its own: this pack no longer modifies
  ComfyUI core at all**, at import or otherwise. The shim was the last thing
  that did, and it is archived rather than merely switched off. Model changes
  go through `ModelPatcher.add_object_patch`.

- **Two exemptions became vacuous when the graphs left, and both are fixed
  rather than left to pass.** This is the "off, parked or absent" rule earning
  its place a third time. `bench/check_attention_defaults.py` derives its
  single-frame class from `GRAPH_DIRS`, which now yields an empty set; it says
  so on its own line, because correctly-empty and never-computed had become
  indistinguishable. `bench/check_prompt_guide_conformance.py` was worse: its
  `_STRUCTURE_PROBES` still named an archived graph, so it waived nothing while
  reading as coverage, and the waiver line only printed when non-empty. The set
  is empty, the line always prints, and a new case asserts every structure
  probe still ships -- red-proved with a bogus entry.

  The class stays *derived* rather than deleted in both, so a directory added
  back to `GRAPH_DIRS` re-arms the exemption without anyone remembering to.

- **User-facing guidance that had quietly become wrong advice.**
  `resolution.py` told anyone asking for `length=1` to enable a shim that no
  longer exists, and `keyframe_canvas.py` pointed at "the path this repo ships
  and renders". Both rewritten. `_core_supports_single_frame()` is unchanged
  and still correct: it always probed *core*, never the shim.

- `docs/checks.md` re-keyed the archived check and dropped two counts that were
  stale independently of this change -- an audit count the run itself reports,
  and a waiver count the code contradicted.

## 0.74.0

### Changed

- **The single-frame shim is opt-in.** `single_frame.apply()` patches
  `comfy_extras.nodes_minimax_h3` in memory to lift core's 5-frame `length`
  floor, and it did that on every import. It now does nothing unless
  `H3_EXPLORATIONS_SINGLE_FRAME=1` is set, and says nothing when it does
  nothing -- patching nothing is stock behaviour with nothing to announce, and
  a banner at every startup for a path most installs never use is how a console
  stops being read. Owner decision, prompted by outside feedback that a pack
  should not monkey-patch core at startup: the patch is process-global, so the
  default charged every install for a feature it had not asked for.

  **A node cannot replace it, and that was checked rather than conceded.** The
  floor is enforced by `execution.py::validate_inputs`, which raises
  `value_smaller_than_min` *before* any node executes. A node placed in the
  graph is therefore rejected along with the graph it exists to enable, and
  could only affect later prompts -- so the first queue would always fail. An
  environment variable is read at import, which is before registration builds
  the schema. That is the whole reason this is not a node.

  Consequence: every graph in `workflows/image/` needs the variable set, at
  render time and on any server the generator validates against. Without it
  they fail validation with "Value 1 smaller than min of 5".

  `bench/check_single_frame.py` sets the variable for itself, in one place and
  after reading the default, because its subject is the patch rather than the
  deployment: none of "is it equivalent off length=1", "does the guard refuse a
  wrong rewrite", "does a second apply rewrap" is reachable through a closed
  gate, and inheriting the operator's shell would make the suite pass or fail
  on how the terminal was configured. It gained a case for the new default that
  asserts the *reason* as well as the outcome, because "nothing to patch" would
  otherwise pass it for the wrong reason.

- `docs/h3_image_editing.md` pointed at a README section that does not exist --
  that file is five lines. It now points at `single_frame.py`, which actually
  owns the mechanism, and states the opt-in.

- **The shipped text encoder is the v2 W4A16 AWQ artifact.** `MODELS["clip"]`
  now names `ENCODER_V2`, defined above it so the string still has one copy,
  and every graph is rebuilt onto it. The two artifacts differ in one setting
  that matters here: v1's snapshot declares a still-image budget far narrower
  than the release's, v2 declares the release's own. Under v1 every reference
  reached the conditioner reduced to a few hundred merged tokens whatever it
  had been prepared at, which is why `MiniMaxH3AppendRefImage.qwen_short_edge`
  could not do anything; under v2 a 2048 short edge arrives intact.

  That is a cost as well as a capability, and the graphs were not re-tuned for
  it. An unclamped Qwen view costs `(w/32)*(h/32)` merged tokens, the same
  arithmetic as the DiT reference rows, and both segments sit inside Sol-Attn's
  exact sink. A graph that keeps `allow_upscale=True` now pays for that upscale
  in two segments where it used to pay in one, so reference policy is the open
  question this change hands forward rather than something it settled.

- **`MiniMaxH3AWQEncoderLoader` selects the shipped artifact by default.** Its
  `encoder_name` combo declared no default, so ComfyUI took `options[0]` --
  whatever sorts first in `text_encoders`, which on this box was
  `clip_l.safetensors`, not an H3 encoder at all. It now reads
  `MODELS["clip"]` out of `workflows/h3_config.py` by path, so the menu default
  and the graphs cannot disagree, and falls back to the unmodified menu when
  that file is absent (the standalone distribution carries the module without
  it) or when the directory does not offer the name -- a default outside
  `options` would be the manufactured menu item the node already refuses.

- `bench/check_h3_awq_encoder.py` chooses the adapter for its large CPU load by
  asking whether that adapter's own resolver recognizes the artifact, instead
  of assuming the standalone distribution packages whatever `MODELS["clip"]`
  names. Those coincided until the change above and no longer do: the
  standalone embeds one artifact's configs, so handing it a different one
  tested only the mismatch. Its own contract case already proves the render is
  deterministic, the embedded configs match their source digests and the
  load-bearing functions are copied verbatim, so the shipped default now takes
  the large load through the repo-local adapter that scans every snapshot.

  **The branch is on the embedded config, never the filename**, and a live
  instance is why rather than a principle: pointing the v1 *name* at the v2
  *file* through a symlink made the loader read v2's release bounds while the
  name-keyed registry read v1's clamp, so `bench/preflight_graph.py` priced a
  clamp that would not happen -- confidently, because the v1 name is a known
  key and the lookup succeeded. An unknown name would have returned "no
  contract" and been safe. Name an artifact in `workflows/h3_config.py`; do not
  symlink one file into another's name.

### Added

- **The Hugging Face distribution targets v2, and ships a model card.**
  `bench/build_h3_awq_standalone.py` embeds the v2 snapshot (which spells its
  still-image settings `preprocessor_config.json`, where v1 nested them in
  `processor_config.json`) and names the v2 checkpoint. The generated loader
  accepts v2 and refuses v1, which is the mirror of what the published v1
  loader does to a v2 file -- verified by driving both modules' own
  `_validate_metadata` against both checkpoints rather than reasoning about it.

  The card carries no open-report section: what to say publicly about an
  unresolved third-party report is the owner's call, not the build's.

  The card's encoder comparison is **read from
  `bench/results/2026-08-25_four_encoders_holdout_layer50.json` at build time**,
  not retyped, and it says plainly what that record shows: v2 is not a fidelity
  improvement on v1 by that measure, and ComfyUI's INT8 ConvRot conditioner is
  about an order of magnitude closer to BF16 than either W4A16 build. It also
  says that layer-50 distance is a proxy and no blinded matched-seed clip
  comparison has been run.

- `bench/probe_audio_marker_effect.py` and its two records, for the report that
  a reference with audio produced video ignoring the prompt. It asks an
  **absolute** question on purpose -- non-finite values, collapsed rows, dead
  spans -- because the obvious comparison, how far the rows move when the audio
  marker is added, has no meaning without a reference encoder: the marker is
  real text and legitimately shifts every downstream row. Both v1 and v2 come
  back clean on both arms, so the loud failure modes are ruled out and the
  question escalates to a BF16 comparison rather than being closed.

### Documentation

- `docs/h3_awq_encoder.md` no longer states one budget as "the current
  artifact's". The budget is the setting that varies between artifacts, so the
  section names `source_image_pixel_bounds()` as the way to read it and records
  which artifact ships. The layer-49 benchmark that argued for keeping v1's
  snapshot unchanged is marked as history taken while v1 was the default.
- `docs/h3_references.md`'s caveat on `qwen_short_edge` said the knob stays
  inert until an encoder whose bounds admit the view is loaded. That encoder is
  now the shipped one, so the caveat records when it stopped binding and what
  the knob costs now that it works.

### Fixed

- Two changelog sections were both filed as 0.72.0. The newer is 0.73.0.

## 0.73.0

### Added

- `bench/measure_encoder_footprint.py` and its record: what each H3
  conditioning encoder costs to hold resident, by component. The four-encoder
  fidelity table said nothing about cost, and file size answers that wrongly
  twice -- the W4A16 files carry all 64 layers plus an `lm_head` H3 never uses,
  and equal-sized files split their bytes very differently between the decoder,
  the embedding table and the vision tower. Safetensors headers only, no tensor
  data, either virtualenv. The layer sum is exact rather than a per-layer
  average, because a mixed-precision candidate is the reason to measure this at
  all; a layout whose layers are not a contiguous run from zero, or that has
  fewer than the fifty H3 consumes, is refused with the reason rather than
  reported as a number.

- The same record's bit-allocation section, rewritten the same day after the
  owner proposed calibrating each layer on real render traces. Two of the three
  parts of that proposal already hold -- `propagate_error` defaults to true, so
  the sequential pipeline already quantizes each layer against the propagated
  outputs of the layers above it, and the v2 bundle already carried genuine H3
  presentation through real 32B weights. The part that does not is worth more
  applied to the allocation than to another calibration: rank modules by
  `||Wx - W_hat x||` on captured activations rather than by weight error alone,
  since a large perturbation in a direction the activations never excite costs
  nothing downstream. The record now says so, names `grade_sage_on_capture.py`
  as the existing instrument of that shape, and carries the second use for the
  same trace data -- weighting encoder error by what the DiT actually reads,
  which is the metric criticism the launch record already made of itself.

- The same record again, on whether the calibration measured the encoding H3
  actually uses or ordinary inference. It measured the encoding, and that was
  checked rather than assumed: ComfyUI's encode is one causal forward with no
  padding mask, Gate 1 already found the all-ones mask bit-identical at the raw
  layer-49 state and omits it from the traced graph behind a red control, and no
  `generate()` is called anywhere in the pilot. The finding underneath is that
  the objective was never the encoding's: `AWQModifier._compute_loss` compares
  one mapping's parent module against its quantized self, four mappings per
  layer across all sixty-four, with no term for the layer-49 state H3 reads.
  Compounding is handled on the input side and nothing is handled on the output
  side, which is the mechanism behind the ten-percent ceiling the overfit test
  measured -- and a second reason to spend trace data on the bit allocation,
  which can be propagated to layer 49, rather than on another population.

- Two owner decisions on 2026-08-26 recorded where they reverse things already
  written. The W4A16 artifacts are **published for the community**, not built
  for this box, so the encoder record's "no further own-calibration quants"
  recommendation is superseded in scope and the GPTQ items on the point handoff
  stand: their users cannot run a 25 GiB encoder, and an INT8 build ships
  alongside if it earns one. With it, a distinction both documents had blurred
  -- the tenth-order prior sizes a *data* change, and GPTQ is a *solver* change,
  so that prior does not bound it. And a mixed-precision encoder is not wanted
  unless INT8 can be reduced for this task at no quality cost, which it cannot:
  the only lossless reduction is the embedding table at 0.724 GiB, landing the
  H3 path at 24.553 GiB, still above what a 24 GB card reports usable and
  before any activation.

- An annotation on the v2 launch record beside the Gate 5 table: v2's medians
  moved against v1 while the means at `upscale_2048` were a wash and its worst
  row was better, so v2 cut the tail and lifted the middle where the bar
  rewarded the opposite shape. The verdict is unchanged -- criterion 1 was
  pre-registered, which is what makes it legible rather than post-hoc
  rescuable -- and the annotation is filed against the metric, which that record
  already names as this result's third suspect. It carries the forward
  instruction: decide before a run whether tail or median is what a bar buys.

- Reconciliation of that record after a peer session filed its own section 5 on
  the same subject from a session that could not see this one: three additive
  cross-references binding section 5's GPTQ consolidation by the local-objective
  finding, naming `bench/capture_h3_encoder_states.py` as the harness its first
  measurement extends rather than a hook to build, and pointing section 3
  forward so the two treatments are not found independently. A fourth records
  that a graph's attention arm cannot be a selection criterion for an encoder
  trace -- `MiniMaxH3SageAttention`, `MiniMaxH3SLARouter` and the vendored
  `SolAttnMiniMax` the graphs actually wire all take a MODEL input, and the
  encoder resolves its own `optimized_attention_for_device` inside its decoder
  forward -- so the criterion is schema coverage and the candidate set
  is every shipped graph, not the four unpatched PDD arms.

- A rule in `CLAUDE.md` under "How this repo decides something is true": two
  models live here and the words for their parts do not disambiguate the stage.
  Three instances on 2026-08-26 across two sessions, every one carrying a
  DiT-side fact to an encoder-side conclusion, and the tell was a type or a
  prefix each time rather than the vocabulary of the claim.

- `canonical/2026-08-26_encoder_choice_and_marker_measurement.md` under the
  Qwen3-VL research tree: why no AWQ calibration population could have reached
  the marker, prompt-structure or vision-encoding questions -- the holdout
  captures already record `layer0_input` relative L2 of exactly zero, so both
  the embedding table and the whole vision path are byte-identical across the
  BF16 reference and both W4 artifacts. Also the footprint table above read
  beside the fidelity one, and how the marker question is answered by selecting
  among representations that already exist rather than by a trainer. Its two
  recommendations are marked open for the owner; nothing in it supersedes a
  decision or changes the card order.

## 0.72.0

### Changed

- **The PDD head selector matches block boundaries directly.** It built a `t`
  from the 1025-row curve table and bucketed it, which is how a `t` sitting
  exactly on a boundary selected the previous block; the first fix was a snap
  tolerance that then had to be justified against the table's quantisation. It
  now matches the timestep embedding against the `nfe + 1` boundary
  embeddings, built at load from the model's own arithmetic. No intermediate
  `t` to quantise, exactly `nfe` answers to choose between, and **no tolerance
  in the selection path at all**. `pdd_math.step_for_t` and
  `boundary_residual` are deleted rather than left: two selectors for one
  question is how the wrong one gets called.

- **The converter ships the published head bank and the node fuses at load.**
  One file is every step count the 32-point grid divides by -- 8 and 4 are both
  counts the vendor's README reports rendering at, and 2 and 16 come free from
  the same divisibility. Fusing at setup rather than per forward is the paper's
  own recommendation (3.1: "we only need to hold one fused linear layer per
  block in memory"); the vendor adapter and the independent conversion both
  fuse inside the forward.

- **A converted file carries one adaln form.** `--pruned` names the base, so it
  also says which form to emit. Both together was ~40% of the file dead in the
  only configuration this repo renders, and two representations of one update
  with nothing recording which the node used. With the fp16 bake the file is
  1.12 GB against 1.82 before, and against 1.70 for the independent conversion
  at the same information for this base.

### Added

- Six canonical PDD arms -- t2v, fl2v and ref2v at 8 and 4 evaluations -- with
  sage ON and Sol ABSENT. Sol skips attention adaptively per step, which is
  incoherent against a fixed fused block schedule, and a bypassed node is an
  invitation to switch it on. The dense probes stay as the reference
  configuration, which is what the vendor's own pipeline runs.

- `bench/compare_pdd_conversions.py`: grades our conversion against the paper,
  the vendor adapter and an independent conversion, which fail differently.
  Backbone transforms are bit-identical to the independent one including the
  block-diagonal qkv fusion and the SwiGLU half-swap; both banks fuse to the
  same heads at every step count; and the adaln bake, which each arrived at
  separately, is scored against ground truth rather than against the other.

### Fixed

- `bench/check_distill_settings.py` and `bench/check_distill_grid.py` graded a
  PDD arm's step count against the FILE's `pdd_nfe` and failed a correct 4-step
  arm the hour it landed. With load-time fusion the file records a default and
  the graph records what runs, so both read the graph first.

## 0.71.1

### Fixed

Found by an xhigh `/code-review` that caught this tree mid-flight and was
relayed by a peer session. All confirmed against the code before acting; none
were taken on the relay alone.

- `bench/generate_capture_manifest.py`: `attention.sage_mode` was the hardcoded
  string `"fp16 (most accurate)"`, overwritten only when a sage node happened
  to exist. Every PDD arm runs no sage, so their manifests would have reported
  a mode for a kernel that never ran -- and the constant did not even match
  `h3_config.SAGE_NODE["mode"]`. That is the identical "a value that cannot
  fail" defect the comment directly above it records for `sol_attn`, left
  behind in the same dict literal when that one was fixed. Derived now, with
  `absent` / `orphaned` as real states.

- Same file: LoRAs were recorded by PRESENCE. An active-but-unconsumed loader
  would be written as a LoRA that ran -- the exact failure `_sol_attn_state`
  was built to stop for Sol. Latent in the shipped graphs and reachable through
  `--workflow`, which takes hand-built ones. `_sol_attn_state`'s reasoning is
  now `_class_state`, used by all three consumers.

- Same file: `bool(inputs.get("patch_heads", True))` returned True for a linked
  widget, fabricating the heads-ON arm in the one field that distinguishes a
  PDD arm from its control; and `isinstance(raw, (int, float))` recorded null
  for a string strength that `pdd_lora.py` explicitly coerces and runs.
  `_scalar` now separates four cases -- number, string, missing (the node's own
  default, a fact) and linked (unknowable, so null rather than invented).

- Same file: `MiniMaxH3TurboLoRA`'s `low_vram` was dropped, though it selects
  merge-vs-bypass. Two numerically different renders produced byte-identical
  manifests.

- The loader class list existed in three copies with nothing red on divergence,
  and the manifest's copy was already a class behind. One list now, in
  `workflows/h3_config.py`, whose stated rule is that nothing there may have a
  second copy. Its docstring names the stronger shape it is not:
  `substrate.weights()` matches on the value looking like a weight file, which
  cannot go quietly incomplete.

- `docs/capture_manifest_schema.md`: the lora record's shape changed
  (`strength` nullable, `loader`, `pdd_patch_heads`, `low_vram`) and
  `sage_mode` gained non-mode states. Schema stamped 1.3.0 and
  `SCHEMA_VERSIONS` accepts it -- two manifests at one version with different
  shapes defeats the version-gated assertion pattern the checker is built on.

- `bench/check_attention_defaults.py` said "Three kinds of graph legitimately
  ship without Sol", which the PDD arms made wrong. It points at
  `SOL_EXEMPT_STEMS` now instead of counting.

- `substrate.py::_infer_rank` cited a literal `"rank": 256` in
  `generate_capture_manifest.py:133` that has been false since 2026-08-17, at a
  line number that had become something else. Past tense, no line number.

- `docs/h3_pdd.md`: an encoder-scoping sentence inserted earlier that day
  orphaned "It costs about 2.4x" from its antecedent.

## 0.71.0

### Added

- Four PDD arms in the generator (`h3_probe_ref2v_pdd`, its `_headfree`
  control, and `_345` / `_8s` length variants), a PDD branch in both graph
  builders, and PDD grading in `bench/check_distill_settings.py`. That grading
  is not optional: `is_turbo()` is false for a PDD filename, so without it a
  PDD arm reads as a base graph -- policed for shift, which it satisfies, and
  never graded on steps. The same trap the pack loader's own comment records.

- `bench/convert_pdd_lora.py`, `pdd_math.py` and `MiniMaxH3PDDLoRA`
  (`pdd_lora.py`): end-to-end support for alibaba-pai's Parallel Decoding
  Distillation acceleration LoRAs, which are not step distillations and load
  nowhere in ComfyUI as published. One file reaches the model on three
  surfaces -- a 208-module backbone LoRA, 50 adaln modules that are a weight
  patch on an unpruned base and a runtime injection on a pruned one, and the
  per-interval output heads, which are not a delta at all. The converter does
  every transform that can be decided offline and fuses the 32-head stack to
  the entries a run can actually ask for; the node does the three runtime
  surfaces. `pdd_math.py` holds the schedule arithmetic both consume, because a
  drift between what the converter fused and what the node selects is a silent
  wrong-head.

  Two properties are worth stating because they decide how the arm is wired.
  The block boundaries are the plain 8-step shifted schedule bit for bit, so a
  PDD arm moves the step count and nothing else -- the shift stays at the
  checkpoint's own 12/3. And the step index is derived from `t_emb` rather than
  from a call counter, so it cannot desync from the schedule the way the
  vendor's forward-hook counter does on any extra evaluation.

  The partition check is a fingerprint of `final_layer.video_out.weight`, which
  is bit-identical across pruned/unpruned and across quantisation formats and
  differs between fl2va and ref2va -- an observable, not a filename, against
  the identical-key-set trap `docs/h3_ref2v_distillation.md` records.

- `docs/h3_pdd.md`: the mechanism, the mapping measurements, what was borrowed
  from `ComfyUI-MiniMax-H3-Turbo` and `ComfyUI-MiniMaxH3-PDD-Mamad8` and where
  each is credited in the source, and what is enforced by nothing. Nothing here
  has been through a sampler yet, and the doc says so.

- `workflows/h3_config.py`: `PDD_FL2VA_LORA`, `PDD_REF2VA_LORA`, `PDD_STEPS`,
  `PDD_STRENGTH`, `PDD_SHIFT`. Named with the `_LORA` suffix deliberately --
  that suffix is `bench/check_lora_alpha.py`'s selector, and the first draft
  named them `PDD_LORA_*`, which resolved and graded nothing.

### Fixed

- `pdd_math.step_for_t`: a PDD render decoded the wrong fused head at two of
  its eight steps. `t` is recovered by a nearest-row lookup against the
  1025-row curve table, which quantises to ~1e-3, so a `t` sitting exactly on
  a block boundary came back a fraction below it and interval membership
  returned the previous block. Video heads went `[0, 0, 2, 3, 4, 5, 6, 6]`,
  audio `[0, 1, 1, 3, 4, 5, 6, 7]` -- wrong at step 1 and at step 7, the
  largest jump in this schedule and where the fused heads differ most.

  It shipped four renders before anything noticed, and nothing could have: the
  recovered `t` was correct to 4e-5, so the node's own boundary-residual
  warning is silent by construction. On-schedule times now snap to the nearest
  boundary, with interval membership kept for the off-schedule fallback, and
  both regimes use one tolerance so they cannot disagree about which applies.

### Added

- `bench/check_pdd_head_selection.py`: the control for the above, driving the
  real `_StepTracker` with a real `adaln_t_table` read off a shipped
  checkpoint and the real 8-step boundaries, video and audio separately. Its
  red case reproduces the escaped selection with snapping disabled, so it
  cannot go green on a tracker that lost the fix.

## 0.70.13

### Added

- `bench/h3_gptq_recipe.py`: a W4A16 GPTQ recipe on the same decoder-only
  boundary as the AWQ one, which it imports rather than copies, plus the
  AWQ-then-GPTQ composition and an `overrides` map so a later
  W8-on-named-layers variant needs no new file. Gate 5 rejected v2 against v1
  with both artifacts sitting at the same error and the calibration data moving
  it by about a tenth either way, so the recipe is the axis nobody has varied:
  AWQ moves rounding error between channels, GPTQ compensates it against the
  layer's own input covariance. The docstring carries the field-by-field
  justification read off the pinned source, the seam in the composition (GPTQ's
  Hessian is built from the activations of forwards that precede AWQ's
  smoothing of the same layer, and nothing rescales it), and the host and
  device budget as arithmetic that is explicitly unmeasured until the card
  probe runs.
- `bench/check_h3_gptq_recipe.py`: the CPU red control for it. It drives the
  whole modifier lifecycle on a reduced-width Qwen3-VL rather than stopping at
  config application, so the per-layer Hessian free -- the property the host
  budget rests on, and the whole difference from AWQ -- is observed rather than
  read out of a docstring. Its mutations must each fire *and* name what they
  broke.

### Changed

- `bench/pilot_sequential_feasibility.py` takes `--modifier gptq` and
  `--modifier awq_gptq` beside `awq`, sharing everything that is not
  AWQ-specific. The smoothing instrumentation and the scales sidecar are
  skipped rather than emitted at zero, because a record carrying those fields
  empty reads as a measurement. GPTQ contributes the two numbers the budget
  question needs -- Hessian bytes per device, censused at the entry to the
  quantize step because that step frees them as it goes, and the time spent
  there -- plus the failure that is otherwise silent: a Hessian the Cholesky
  cannot invert degrades that one module to round-to-nearest with nothing but a
  log line to say so. The GPTQ wrapper patches the class rather than the
  instance, because these modifiers are pydantic models and refuse an attribute
  that is not a declared field; the AWQ wrapper escapes that only because all
  three of its targets are private names.

## 0.70.12

### Added

- `bench/select_v2_calibration_rows.py` gained four selection paths, each off
  unless its flag is given. **`--stratum`** replaces the role row-quota with a
  token-balanced design over `primary role | marker presence`, marker presence
  read against the markers the release appends past the stock vocabulary
  (`vendor_config.h3_markers()`, new, derived from the vendored declaration
  rather than typed). `--stratum-token-share NAME=SHARE` names one stratum's
  share of the token budget and the rest split what is left equally, with
  `--stratum-floor-share` as the minimum any occupied stratum may be given.
  The reason is that AWQ's statistics are per token and a row quota says
  nothing about where the token mass goes: the 2026-08-25 run was 29 rows and
  214k tokens with roughly nine tenths of them visual, so the H3 schema
  positions were about a tenth of what the scales were fitted on.
  **`--max-vision-tokens-per-row`** caps a row's estimated visual tokens; a row
  over it is skipped, never truncated, and the cap reaches calibration only
  because a vision cap on the holdout would choose which geometries the holdout
  grades. **`--component-map`** assigns by corrected visual family instead of
  exact media component -- every exclusion widens to the family, no two
  calibration rows may share one, and the split assertion moves with it -- and
  refuses a map carrying no `caveat`. **`--text-only-share`** admits the pool's
  text-only T2VA rows into their own `text_only` list, never merged into
  `calibration`, because one `oneshot` call traces the model once and that
  trace fixes the modality envelope for the whole run. `--pool`,
  `--text-only-pool` and `--dataset-root` exist so a control can drive a
  synthetic population. The record now carries the achieved text/visual split
  per stratum, which is the number that decides whether the schema positions
  had weight.
- `bench/check_calibration_selector.py` and its red harness
  `bench/red/show_red_check_calibration_selector.py`. The arm that matters
  holds the untouched role-quota path to the last committed revision of the
  selector before this change, run out of a scratch mirror of `bench/`, rather
  than to an expectation written beside it.
- `bench/results/2026-08-25_v3_selection_{max_no_upscale,upscale_2048}.json`: a
  v3 candidate population selected with the new flags, family-disjoint across
  the two still-policy arms, at an assumed 400k-token budget that is UNKNOWN
  until the GPTQ host probe. The union carries 93 rows in 93 corrected
  families, all twelve strata occupied, and the estimated visual share falls
  from the 2026-08-25 run's nine tenths to about three quarters.

## 0.70.11

### Changed

- The three Gate 6 reference-view arms (`workflows/h3_probe_refview_*`) load
  the ComfyUI-native INT8 ConvRot encoder (`h3_config.ENCODER_INT8`) instead
  of the W4A16 v2 artifact. On the 13-row holdout INT8 sits about fifteen
  times closer to the BF16 release at layer 50 than either W4A16 artifact
  (`bench/results/2026-08-25_four_encoders_holdout_layer50.json`), so it is
  the encoder of record for the ablation and the marker arms; the W4 lane
  continues as the small-host variant. `bench/gate6_refview_arms.json` says
  how to patch a W4 artifact back in on every arm at once. The generator
  picks the loader node from the file (`h3_config.CORE_LOADED_ENCODERS`):
  core's `CLIPLoader` with `type=minimax` for the ComfyUI-native files, the
  repo's `MiniMaxH3AWQEncoderLoader` for W4A16 artifacts, which refuses
  anything else; the first render of the switched arms hit exactly that
  refusal.

## 0.70.10

### Added

- `bench/measure_dit_prefix_attention.py`: how much of a latent-video query's
  attention mass the H3 DiT puts on the Qwen3-VL prefix rather than on the
  latents, per captured block and sampler step, and where inside the prefix it
  lands. Writes a per-prefix-position importance vector as safetensors, usable
  later as a loss weight. A capture records no segment table, so the packed
  layout is rebuilt from `PackedLayout` through
  `bench/count_packed_rows.py` and the capture's own graph; the prefix length
  is agreed by two independent routes -- the geometric residual and the
  tokenised prompt -- before any number is reported.
- `bench/check_dit_prefix_attention.py`: the controls, on a synthetic capture
  of a few hundred rows, CPU only. The class map is a total partition; the
  prefix-versus-latent split survives a shuffle of the prefix labels while the
  within-prefix split does not; a prefix boundary wrong by one row is refused.
  Each of the three carries a negative arm. The tag-derived check that would confirm
  each prefix position's modality **cannot be built from a capture** --
  `h3_capture.py` writes q/k/v and nothing else -- so the record says `UNKNOWN`
  and a case asserts that it does.
- `bench/results/2026-08-25_dit_prefix_attention_t2va.json` and its importance
  vectors, measured on the 2026-08-20 T2VA capture. The record also carries the
  finding that that capture's `manifest.json` `workload` and `token_accounting`
  blocks describe a different render from the tensors beside them, while its
  `prompt` and `captured_tensors` blocks are correct.

## 0.70.9

### Changed

- `bench/pilot_sequential_feasibility.py` captures the session's recipe as
  YAML while the session is live and writes it as the candidate's
  `recipe.yaml` at emit. The save wrapper writes that file from the *active*
  session, and the pilot emits after its session has closed, so the first v2
  candidate shipped `default_stage: {}`. The v2 candidate and its snapshot
  were repaired from the same recipe constructor (`duo_scaling: true`,
  `n_grid: 20`); the run record had carried the recipe throughout.
- `bench/compare_transformers_comfy_layer50.py`: the ComfyUI arm takes
  `--clip-path`, a ComfyUI-native encoder file through core's own
  `load_clip` (the shipped `int8_convrot` and `nvfp4_awq` variants), so all
  four encoders can be graded on one holdout against BF16. Text-only rows no
  longer break the block splitter or the presentation record; a rerun with
  `--all-rows` skips rows already captured and records the VRAM reserve per
  row.

## 0.70.8

### Changed

- `bench/convert_h3_awq_candidate.py` prints the resolved snapshot as a
  repo-relative path. The absolute form landed in a gitignored log and turned
  `bench/check_no_owner_paths.py` red on a correct tree; that line is the kind
  that gets pasted into a record.

## 0.70.7

### Fixed

- **`bench/review_pool_near_duplicates.py` hashed images only**, so the pool's
  video files were never candidates in a near-duplicate review whose record read
  as covering the pool's media. Videos are now hashed as sampled frame sets and
  compared against each other *and against every still*; because there are few
  enough, all of them were also inspected exhaustively rather than only through
  the window. Two things followed that no image-only scan could reach: a
  reference still that is a frame of a reference video, in two different exact
  components; and two videos that are the same shot list rendered twice, at more
  than twice the Hamming threshold.
- **The second of those is a limit on the method, and the map now says so.**
  Same-brief-different-render relatedness exists in this source and the window
  cannot reach it -- widening that far would swamp the candidate list, since
  unrelated pairs already reach a fifth of a percent at Hamming 12. It was found
  only because nineteen files is small enough to look at; the equivalent among
  the images is unexamined and the population is too large to eyeball. The
  emitted map's caveat states both halves rather than the image half alone.
- An adjudication entry may now carry `outside_window` and is applied to the
  component map on the strength of the ruling alone. A pair a person found
  outside the candidate window still has to be able to reach the map: the window
  generates candidates, it does not bound what is true.
- The corrected component map absorbs both new edges. The running calibration
  bundle, both holdouts and the Gate 6 population re-verify green against it.

## 0.70.6

### Changed

- **Checkpoint cadence is a parameter, defaulting to every 4 layers.** Time was
  never the constraint: every-layer cadence costs under 7 minutes across 64
  layers. Disk written is, at roughly 1.1 TB per run, so the default trades at
  most four layers of redone work -- about 18 minutes at the measured rate --
  for roughly 280 GB. `every=1` remains available for a run that has already
  failed once. The transient second copy at the rename is the same size at any
  cadence.
- `bench/check_calibration_checkpoint.py` gains the two cases the cadence
  needs. `resuming_from_an_older_checkpoint_is_still_exact` is the one that
  licenses a coarse default: it resumes from the FIRST checkpoint rather than
  the newest, so the run redoes a completed layer, and the result is still
  identical to never having failed. `cadence_is_a_parameter_and_the_default_is_coarse`
  pins the shipped default by exercising it -- on a three-layer fixture the
  default correctly writes nothing, which is the assertion rather than a defect,
  because a test quietly running at `every=1` would leave the shipped value
  unexercised.

### Fixed

- The cadence guard was validating nothing. `checkpoint_each_boundary` is a
  context manager, so its body does not run until `__enter__` and a bare call
  with `every=0` raised nothing; the case asserting the refusal passed while
  the refusal did not happen. The case now enters the context.

### Parked

- **Checkpoint and resume is parked by owner decision, 2026-08-25.** The
  module, the design note and the check stay committed and green; nothing is
  wired into `bench/pilot_sequential_feasibility.py` and the card proof on real
  weights has not run. It returns when a run longer than that evening's is
  planned. `docs/research/calibration_checkpoint_resume.md` and the
  `docs/checks.md` row both say so, so a green check cannot be read as evidence
  that any real run checkpointed.

## 0.70.5

### Added

- **`bench/h3_calibration_checkpoint.py`**: checkpoint a sequential AWQ
  calibration at a subgraph boundary and resume from it, so a mid-run failure
  costs one layer instead of the run. The resume is two seams the pipeline
  already has -- `trace_subgraphs` wrapped to return `subgraphs[start:]`, and
  `IntermediatesCache.from_dataloader` wrapped to hand back the restored cache
  -- rather than a copy of the loop with a start index, which would have been a
  second implementation drifting from the installed one on the first upstream
  change. Writes are staged and renamed, so a kill during a write leaves the
  previous checkpoint intact.
- **`bench/check_calibration_checkpoint.py`**, six cases, built around the
  proof: an uninterrupted run and a killed-then-resumed run must agree tensor
  for tensor. It runs the real pipeline with the real recipe at fixture scale on
  CPU, so the mechanism is settled before the card is free.
- `docs/research/calibration_checkpoint_resume.md`, the design and the state
  inventory it implements.

### Measured

- `AWQModifier._parent_args_cache` is pre-populated for every layer when hooks
  are installed, so at a boundary it looks like live state crossing it. Counting
  filled batches inside those caches gives zero, at every boundary, for every
  surviving parent -- structure, not data, rebuilt by a fresh process. It is
  therefore not checkpointed. `_smooth_activation_stats` is likewise empty at
  every boundary; `_error_metrics` is not, and is the one piece of modifier
  state the brief's inventory did not list.
- Subgraph `k` smooths layer `k-1`, with a prologue subgraph that smooths
  nothing, so checkpoints index by subgraph and derive the layer mapping.
- **The resumable instant is the top of a subgraph's epoch end, not after it.**
  With `propagate_error` at its default the propagation pass that writes a
  subgraph's outputs runs after the callback, so at the callback the cache
  still holds that subgraph's inputs. The first version of the design note said
  otherwise and the resume failed on exactly that.
- The recipe description is stable across builds but **not across
  `session.initialize`**, which fills in mappings inferred from the model. An
  identity taken afterwards never matches one taken by a fresh process, so
  every resume would have refused itself.
- Checkpoint cadence, the open question from the brief: a 16 GiB cache writes
  in about 6 s per boundary, under 7 minutes across 64 layers, so every-layer
  cadence is affordable. The honest figure needs an fsync -- 2.7 GB/s synced
  against 6.0 GB/s unsynced, which is the page cache rather than the device.
  The cost that is not time is endurance: overwriting one checkpoint 64 times
  still writes roughly 1.1 TB per run.
- The AWQ arm does not run on CPU at all: `_apply_smoothing` pins host memory.
  The CPU cases neuter that call, which is legitimate for arithmetic that does
  not depend on it and is why the card run is still required.

### Fixed

- A claim of my own, by mutation: moving the checkpoint snapshot to the other
  side of the epoch-end callback does not double-smooth a layer, because that
  layer is not restored either way and the weights come out identical. The real
  consequence is a checkpoint whose error metrics describe a layer
  `completed_layers` excludes, so the resumed run reports it twice. The comment
  asserting the wrong consequence is corrected and a case now owns the right one.

## 0.70.4

### Added

- `bench/compare_transformers_comfy_layer50.py`: the ComfyUI arm takes
  `--w4-path` (a W4 artifact through the capture instrument's loader, its
  stamped contract recorded as the source) and `--all-rows` (one model load,
  every bundle row to `OUT/<row_id>/`), and `--field-under-test encoder` lets
  a W4 arm compare against the BF16 ComfyUI arm on the same replayed rows.
  A guard requires each captured row to be exactly the bundle's recorded
  sequence length: it proves the `preprocess_embed` replacement held, not
  anything about the artifact's bounds. First real run is Gate 5 on the v2
  candidate; until then it has met only its syntax check.
- `bench/summarize_h3_holdout_captures.py`: folds the comparator's per-row
  reports over a `<geometry>/<row_id>.json` tree into one holdout record
  (distributions of relative L2 and cosine per arm and geometry, per-row
  values, refusals verbatim; exits red on any refusal). Computes nothing new;
  tested on a synthetic tree, first real run is Gate 5.
- `docs/research/qwen3-vl-special-tokens-post-training/2026-08-26_point_handoff.md`:
  the brief for the next point session.

## 0.70.3

### Added

- **`bench/select_gate6_ablation_rows.py`** and
  `bench/results/archive/v2_encoder/2026-08-25_gate6_upscale_ablation_rows.json`: the render
  population for the Gate 6 reference-upscale ablation. Computed from the
  shipped sizing code before selecting anything, because the population follows
  from it: **the three arms differ if and only if a still's short edge is below
  2048.** `max`/no-upscale clamps with `min(1.0, 2048/short_edge)`, so a source
  already at or past that boundary is untouched by every arm. Source *size* is
  not the discriminator and a large-source stratum would have been a guaranteed
  null, so rows are stratified by upscale factor, computed over reference
  stills only -- a keyframe takes the target canvas and no arm moves it.
  Keyframe-only rows are excluded for the same reason. One row is a deliberate
  null control past the boundary: its arms receive bit-identical conditioning,
  so any arm-labelled difference reported on it is a labelling or seeding
  error. Per-arm visual token cost is recorded per row, and every family is
  disjoint from the calibration bundle and both holdouts under the corrected
  component map.
- **`bench/select_t2va_holdout_rows.py`** and its bundle record: the
  deterministic text-only T2VA regression population `active_plan.md` names.
  **Its disjointness is not the vision holdout's**, and claiming it was would
  be the emptiest kind of green -- these rows carry no media, so media and
  visual-family disjointness are true by construction. The informative axis is
  the prompt, and the near-duplicate review is reported against the
  population's own background level rather than a threshold, because every
  `target_ir` shares a section skeleton and a bare floor is high for unrelated
  rows.
- **`bench/results/2026-08-25_violation_arm_grading_audit.md`**: every
  violation arm under `bench/` classified by how it is graded. None grades on
  the exit code of a check that could return non-zero for an unrelated reason;
  the one place an exit code is read is the red spine's own control, where the
  subject *is* a harness and both directions are asserted. `bench/red/
  harness.py` already fails **closed** on a red baseline. The precondition rule
  therefore stands for new arms and earned no rework of existing ones.

### Changed

- `bench/build_native_h3_calibration_batch.py` gains `--population text-only`,
  reading the pool's exclusion complement. The two populations cannot mix and
  that is asserted rather than assumed: the mode refuses a bundle carrying any
  vision-bearing row, and `--family` is refused outright because it names a
  vision role. The vision path is unchanged.
- `bench/review_v2_calibration_bundle.py`: **correctly absent is not broken.**
  A text-only bundle turned every row red twice on a bundle that was correct,
  because `bundle_files` demanded a media file and `family_disjointness`
  demanded a component-map entry from rows that have neither by construction.
  Both now key on whether the row carries a media tensor, so a vision row that
  lost either stays red -- two new mutations assert exactly that. The family
  arm declares itself vacuous on a media-free bundle rather than reporting a
  green that asserts nothing. Separately, three mutations selected their
  subject with a bare `next(...)`, so the violation arm *crashed* on a
  population that could not satisfy them instead of reporting that they did not
  apply; every mutation now names its precondition and refuses by name.
- `docs/checks.md` corrects the precondition rule it adopted the same day. The
  rule said a disk-tier control "reported the null result as a pass"; it did
  not -- its guard yields "unchanged" on a null read, which refuses to emit the
  candidate. Two of the three instances failed open, one failed closed. The
  shared shape is a reading whose precondition was not met producing a verdict
  anyway, and the direction differed. It also corrected what caught them: the
  per-arm baseline caught the two mutation controls, while the third was found
  by running on a tier it had never met. Gap 8 records the one real residual --
  a whole-check baseline lets a mutation aimed at one arm be satisfied by
  another firing, known and un-earned.

## 0.70.2

### Added

- **`marker_arms.py` and `MiniMaxH3MarkerArm` (bench)**: binds one arm of
  `bench/marker_corpus/compiled.json` to a CLIP. `release` binds nothing and is
  what the `release_id` and `stripped` arms use, since their difference is
  prompt bytes and the corpus owns those. `legacy_bpe` swaps in a freshly built
  pre-fix tokenizer. `mean_init_rows` replaces the seven H3 marker embedding
  rows with the table mean. Both transforms attach to a CLONE -- a fresh
  tokenizer and an offset-keyed patch on the cloned patcher -- because
  `CLIP.clone()` shares `cond_stage_model` and the tokenizer by reference and
  the encoder loader caches its patcher, so an in-place change would be
  inherited by every later render that reuses the model. Nothing is written to
  disk and the deployed artifact is untouched.
- **`MiniMaxH3ProvenanceStamp` records an `encoder_arm` block** from an
  optional CLIP input, appended last so saved graphs keep their widget
  positions. Three states: no CLIP is "not detected", a CLIP that never met the
  arm node still reports what its tokenizer and rows actually are with a null
  `declared_arm`, and an armed CLIP adds the label. The block is read off the
  live CLIP -- which markers the tokenizer resolves, and what the patcher makes
  of the marker rows through ComfyUI's own `calculate_weight` -- never from the
  arm name. Schema version 3.
- **`bench/check_marker_arms.py`**, ten cases, every one of the shape "the
  declaration and the value read back must agree". The governing case builds
  all three arms from one fixture and requires their records to differ once the
  label is stripped; the non-self-referential one requires this repo's legacy
  reconstruction to reproduce token ids recorded by a compiler that shares no
  code with it. Runs on a synthetic embedding at the real vocabulary height, no
  artifact and no CUDA, with the real key re-read off the installed encoder so
  the fixture cannot pass by construction.

### Changed

- `bench/node_id_manifest.json` records the appended provenance input and the
  new node. Written entry by entry rather than by `--write`, which regenerates
  the whole file and would have swept in another lane's pending
  `qwen_short_edge` record as though it were deliberate here.

## 0.70.2

### Added

- **`docs/research/sglang_h3_pipeline.md`**: sglang's MiniMax-H3 pipeline end
  to end at the source level, read at `coderef/sglang` commit `6569125e3a`
  by five scoped readers whose citations were verified line by line, then
  synthesised: request and admission, time grid and canvas formulas, media
  ingestion, the raw Qwen3-VL presentation and the layer-50 tap, both VAE
  encode paths and their seeds, the packed layout with its position grid and
  timestep plans, the DiT forward with its bf16 rounding points, the Euler
  loop, decode and output validation, the runtime, parallelism and the
  `high` quality gate, sglang's own ComfyUI node, and 27 numbered insights
  including four contradictions inside sglang's own tree. Nothing was run.
  `sglang_comparison.md` points at it as the walk that precedes the
  comparison; indexed from `CLAUDE.md`.

## 0.70.1

### Added

- **`docs/research/conditioning_cache.md`**, a design note and nothing built:
  a persistent conditioning cache argued for the cross-session and
  `--cache-none` bench cases only, since ComfyUI's own cache covers
  same-session reuse. The key is the capture manifest's provenance record
  (artifact digest, stamped contract, prompt and media hashes, every sizing
  knob, both policies, canvas and length), with the rule that a missing field
  refuses; the payload is what node 5 emits; the node shape is one wrapper
  that computes the key from live inputs rather than a Save/Load pair. The
  vision-tower cache is filed as the smaller second item.

## 0.70.0

### Added

- **The Gate 6 reference-view ablation arms**, `workflows/h3_probe_refview_{a_source,b_qwen2048,c_parity}.json`
  and their API twins: three ref2va arms from the capture request, differing
  only in how each still reaches the video VAE and Qwen3-VL. A leaves every
  still at source size for both; B rescales the Qwen view alone to the vendor
  short edge through `qwen_short_edge` with the VAE at source; C turns
  `allow_upscale` on so one upscaled view feeds both. Every arm names the v2
  encoder artifact (`h3_config.ENCODER_V2`) so v1/v2 swap at the combo, sets
  `video_policy=release`, and keeps Sol on. `bench/gate6_refview_arms.json`
  is the arm manifest; `bench/results/2026-08-25_gate6_refview_preflight.json`
  is each arm priced as generated (A and B share the DiT sequence, B's encoder
  sequence grows by the Qwen views, C grows both).
- `workflows/build_workflows.py`: `clip=` and `ref_qwen_short_edge=` on both
  builders. `qwen_short_edge` is written only when set, so no shipped graph
  changed on regeneration.
- `bench/run_graph_arms.py --manifest ARMS.json`: an arm set as a file
  (`{"arms": {label: repo-relative graph}, "patches": [...]}`), appended after
  any `--arm`/`--set` on the line.

## 0.69.3

### Added

- **`bench/compare_lora_files.py`**: byte comparison of H3 LoRA files across
  the diffusers, diffusers-plus-alpha and ComfyUI-native key conventions,
  local or `https://`. Maps modules between conventions, checks the fused
  `qkv_proj` and SwiGLU `fc1` structure band by band, tests the row
  permutations a SwiGLU re-layout produces before calling a mismatch a
  difference, and reads every alpha scalar beside the metadata alpha. Eleven
  records under `bench/results/2026-08-25_lora_*.json`.

### Measured

- DBM's lightx2v LoRAs are lightx2v's bytes plus alpha tensors; v0.1's alpha
  of 16 is DBM's, declared nowhere else. lightx2v's `comfyui_bf16` files are
  exact conversions. The shipped turbo pack is DBM's `larryvrh_v4_step600_ema`.
  `docs/evidence.md` carries the entry.

## 0.69.2

### Added

- **`bench/compare_dit_checkpoints.py`**: compares two H3 DiT safetensors by
  header and range-fetched samples, local or `https://`, with two tests a
  plain diff cannot settle: which fused-qkv row order each file holds, and
  whether two pruned files' AdaLN factorisations produce the same modulation.
  Records local files by basename. Eight records under
  `bench/results/2026-08-25_dit_fl2va_*.json`.

### Measured

- `DeepBeepMeep/MiniMax-H3`'s FL2VA int8 files are Comfy's bytes; its bf16
  files keep the release qkv order and are not ComfyUI-loadable as-is; its
  `pruned_rank8` is Comfy's pruning in another basis, and its `pruned` is a
  rank-64 variant Comfy does not publish. The `adaln_all` hybrid runs
  ref2va's AdaLN linears on fl2va's curve table, 0.1-0.2% from ref2va's own
  modulation. `docs/evidence.md` carries both entries.

## 0.69.1

### Added

- **`docs/research/technique_transfer.md`**: the translation table from LLM
  and ViT serving techniques to H3 (prefix caches, speculative decoding, step
  caching, sparse attention, token merging, weight quantisation, CUDA graphs,
  parallelism), each with the model property it needs, what it becomes for a
  bidirectional DiT with a prefill-only encoder, and the repo's status on it.
  Names the two borrows still open (token merging on the video rows, W4 DiT
  weights) beside the capture-graded measurement that would decide each.
  Indexed from `CLAUDE.md`.

## 0.69.0

### Changed

- `docs/checks.md` states a new standard: **a mutation control needs its own
  precondition, and must fail rather than pass when it is not met.** Three
  instances in one day, the same shape from three directions -- a mutation
  written against decoded text where the record holds `U+0120`, a mutation that
  picked a row whose mutated field is never exercised, and a control that read
  `None == None` and reported a verdict on a reading it had not taken (that
  one failed closed: it refused the emit). In each the control did
  not apply, and not applying looked exactly like applying and finding nothing.
  What catches it is grading a mutation on the arm it targets gaining a problem
  the unmutated baseline did not have, never on a process exit code.
- `bench/pilot_sequential_feasibility.py`: the modifier-entered control reads
  a disk-tier weight through the offload cache's index instead of returning
  `None` for a meta tensor, and records the weight's tier, file and whether it
  is staged before and after the modifier. Under the disk tier the old control
  compared `None` with `None` and reported the weight unchanged on a run whose
  staged files proved otherwise, which would have refused to emit the
  candidate; it had only ever met the CPU tier.
- `bench/select_v2_calibration_rows.py`: the holdout reserves small-source
  components first (`--holdout-small-source`, the locked "at least two",
  reference stills ahead of keyframes because only reference stills are
  upscaled), and a holdout can be rebuilt around a consumed calibration set
  (`--rows 0`, `--keep-holdout`, `--exclude-row`, `--exclude-component`,
  `--exclude-prompt-term`). The first rebuild dropped a holdout row that was a
  shot-for-shot match of three calibration rows from one catalogue series.

### Added

- `bench/review_v2_calibration_bundle.py`: an independent data review of a
  native-H3 calibration bundle before a long run consumes it. It recomputes
  rather than reads -- `adapt_canvas`, the still-policy scale, the 17n+5 frame
  grid and the 2 fps walk are reimplemented from the release constants and
  cross-checked against the installed nodes -- because grading a builder's own
  record against that builder is the defect the rejected preflight shipped. Two
  escaped instances earned it, both recorded in
  `bench/results/2026-08-25_v2_calibration_set_review.md`: a holdout row that
  was a shot-for-shot match of three calibration rows from one catalogue
  series, and, after that repair, a split that was pairwise green yet shared a
  visual family. The second is why the split is graded against the corrected
  component map as well as by media file.
- `bench/review_pool_near_duplicates.py`: the pool-wide near-duplicate review
  `active_plan.md` requires before a split is frozen, and the corrected
  component map it implies. A report and a human ruling, not a gate: neither a
  perceptual hash nor a background-masked correlation discriminates alone on
  this source, which was measured both ways, so the window only generates
  candidates and every one is adjudicated. The record and the map are
  `bench/results/2026-08-25_pool_near_duplicate_review.md`,
  `..._adjudication.json` and `..._component_map_corrected.json`; the map
  carries a derived caveat naming its own residual limit.
- `bench/compile_marker_corpus.py`, `bench/check_marker_corpus.py` and
  `bench/marker_corpus/`: the marker evaluation corpus of
  `canonical/owner_authored_marker_corpus.md`, as a seed set. A scene is
  serialized to prompt text exactly once and every arm is a declared
  transformation of that one string, so the semantic drift between arms that
  sank the rejected generator is unrepresentable rather than checked for. An
  arm is a triple of prompt bytes, tokenizer identity and model transform,
  which is what lets the mean-initialised-rows arm be declared without
  anything here loading weights or writing a token row.
- `bench/results/2026-08-25_gate2b_host_budget_prefix8_2layers.json` and
  `..._disk_tier.json`: the host-memory cost of the real AWQ recipe at 110k
  population tokens, weights resident and on the bridge's disk tier, the
  latter emitting a packed candidate. The first v2 launch was OOM-killed on
  the host; these set the budget the relaunch was sized from. The record is
  `docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-25_v2_launch_record.md`.

## 0.68.3

### Fixed

- **A W4 CLIP loaded through `bench/capture_h3_encoder_states.py::_load_w4`
  was rebound to the adapter's default snapshot immediately after loading.**
  The re-install called `install_source_processors(clip)` bare, which was
  correct while one artifact existed and became wrong in 0.68.0: on a CLIP the
  loader had resolved to a second generation it silently rebound the still-image
  processor to the first generation's ceiling *and* overwrote the stamp saying
  which artifact it was. An arm would then have recorded and preprocessed as v1
  while holding v2's weights, and the stamp that would have exposed it is the
  thing that got overwritten. `_rebind_source_processors` reads the stamp and
  re-installs against it, then asserts the artifact did not move. Covered in
  `--self-test` on every installed snapshot; the pre-fix line shown red, naming
  both snapshots.

  **Who it would have reached, and who it would not.** Any arm that lets the
  artifact's own processor run — the capture instrument's W4 arm — would have
  produced a delta that was mostly a processor difference wearing a
  weight-quantization label. An arm that replaces `preprocess_embed` to replay
  a bundle's recorded patches is immune by construction, because the declared
  bounds never execute; the layer-50 comparator's ComfyUI arm is built that way
  deliberately, and guards it by requiring each row to reproduce the bundle's
  recorded sequence length, which fires if the replacement did not take.

  Found by pointing the conversion path at a second candidate directory rather
  than by any check: the self-test stubs a CLIP directly and so never traversed
  the real load path. Recorded because that reachability gap is the lesson —
  the control existed and its envelope did not include the defect, and the
  thing that found it was running against an input nobody wrote the code
  against.

## 0.68.2

### Changed

- `bench/check_h3_awq_encoder.py`'s candidate arm distinguishes the two ways
  its snapshot reproduction can go red, because they need different responses.
  A *contract* file differing means the converter stopped copying verbatim or
  the snapshot was edited; only *provenance* differing means the candidate is a
  different calibration run that happens to share the contract, which the
  loader accepts and this comparison cannot. Both branches shown red.

### Measured

- A snapshot pins the contract, not the calibration. Two independently produced
  two-layer candidates, 2026-08-25: `config.json`, both processor configs and
  `tokenizer_config.json` byte-identical; run record and weights different.
  Both resolve to the same snapshot and both load, correctly — the artifact
  digest in `sha256.json` and the copied run record describe the candidate the
  snapshot was written from, not every candidate that resolves to it. This is a
  live case rather than a hypothetical: relaunching a calibration with a
  different row set produces exactly that pair. Recorded in
  `docs/h3_awq_encoder.md`, with the guidance to give a re-run its own snapshot
  name rather than `--force` over one a deployed artifact still matches.
- The conversion path runs unchanged on a second, independently produced
  candidate directory, which is the first time it has met an input it was not
  written against.

## 0.68.1

### Added

- `bench/capture_h3_encoder_states.py` records, per arm, which artifact
  snapshot the shared processor policy was applied over
  (`model.artifact_declaration_overridden`: snapshot name, its config digest,
  and the still/video bounds it declared before the override). It goes in the
  model record and not in `processor_policy_record`, which the comparator
  requires to be *equal* across arms — that equality is what "weight-only"
  means, and a per-artifact value there would make every BF16-versus-candidate
  run refuse itself. `effective_image_processor_config_sha256` is untouched.
- A guard on the same path: the shared policy rebinds the video patchifier as
  well as the still-image processor, so an artifact declaring a different video
  view is now refused rather than captured and labelled weight-only. v1 and v2
  agree about video today, which is exactly when that assumption forms silently.
- `bench/capture_h3_encoder_states.py --self-test`: CPU-only, no model, no
  card. Asserts that every installed snapshot binds one shared policy
  identically, that each arm records the declaration it overrode, that an
  unstamped CLIP declares nothing, and — the deliberate violation — that a
  divergent video view is refused.
- `bench/check_h3_awq_encoder.py`'s candidate arm asserts that every FP32
  tensor in a produced artifact round-trips through BF16 exactly, so a recipe
  that computes genuinely new FP32 values where ComfyUI will downcast goes red
  instead of losing them quietly. Shown red on a candidate with one patch-embed
  element nudged by less than a BF16 ulp.

### Measured

- The candidate's FP32 vision patch embed is the release's BF16 upcast, kept
  wide at calibration so the vision input is never downcast, so ComfyUI's cast
  back to BF16 on load is exact. On the two-layer candidate its
  `visual.patch_embed.proj.{weight,bias}` round-trip through BF16 bit for bit,
  and equal the installed release BF16 encoder's own patch embed bit for bit.
  This corrects the reading in 0.68.0, which recorded the cast as lossy in
  general and did not say that for this artifact it costs nothing.

## 0.68.0

### Added

- **`bench/convert_h3_awq_candidate.py`**: one command from a compressed-tensors
  W4A16 H3 calibration output to what `MiniMaxH3AWQEncoderLoader` loads. It
  consolidates the candidate directory's shards into a single `.safetensors`
  carrying the contract metadata and writes the versioned config snapshot under
  `config/` as byte-for-byte copies of the candidate's own files. The
  single-file form was chosen over teaching the loader to read the directory:
  one load path instead of two, indifferent to shard count, a distinctive name
  in ComfyUI's combo rather than a generic `model.safetensors`, and the
  existing full-file digest control stays meaningful. It costs one more copy of
  the weights on disk.
- **`config/qwen3vl_32b_minimax_h3_w4a16_awq_v2_smoke/`**: the snapshot of the
  two-layer smoke candidate the conversion path was exercised on, so the
  adapter carries a second artifact generation that a check can resolve, bind
  and refuse against with no external file present.
- `bench/check_h3_awq_encoder.py`: `snapshot_resolution_is_by_content`,
  `every_snapshot_binds_its_own_processors_and_tokens`, and an opt-in
  `H3_AWQ_CANDIDATE_DIR` arm that converts a real candidate directory,
  byte-compares the reproduced snapshot against the committed one, and drives
  the produced file.
- **`bench/instrument_render_occupancy.py`**: per-node GPU occupancy of one
  render (`nvidia-smi --query-gpu` at 100 ms bracketed by the websocket's
  node transitions), with a normative verdict on the sampler window:
  launch-bound signal, compute/power-bound, or mixed, printed beside the
  thresholds that decided it. Refuses beside another job. `--self-test`
  pushes synthetic sample sets through the verdict; `--replay-dmon`
  summarises the 2026-08-18 dmon log beside that record's own numbers, and
  reproduces them.

### Changed

- **`h3_awq_encoder.py` recognizes an artifact by its embedded config, not by
  a module default.** `_resolve_snapshot` matches the selected file against
  every `config.json` under `config/`, and the resolved snapshot then drives
  the quant contract, the still and video processors, the token list and the
  stamped `_h3_encoder_contract`. A file matching no snapshot is refused by
  name of what the adapter carries.
- Four v1 details the adapter had only ever met once now branch on the
  observable: the decoder dtype is read from `text_config` rather than the
  checkpoint's top-level `dtype` (a candidate keeping the vision patch embed in
  FP32 declares `float32` there); the truncation depth is the artifact's
  declaration rather than a constant, with the quantized-linear count asserted
  exactly against it; the still-image settings are read from
  `processor_config.json` or `preprocessor_config.json` and from wherever they
  sit inside it; and the 20 special tokens are read from
  `extra_special_tokens` or `additional_special_tokens`.
- The loader refuses a reduced-depth artifact by name of its declared depth
  rather than handing core a state dict it will misdetect. Core recognizes the
  H3 encoder by decoder layer 49 and builds a fixed 50-layer model, so a
  reduced-depth candidate cannot be constructed at all; its adaptation is still
  depth-parametric and can be inspected directly.
- `bench/build_h3_awq_standalone.py` follows the split declaration and stops its
  `_config` replacement at the next definition, so the snapshot readers and the
  resolver survive into the generated one-file loader. The standalone has no
  `config/` directory, which is the correct answer for a published loader that
  answers for the artifact whose configs it embeds.
- `docs/h3_awq_encoder.md` gains *Artifact generations, and how one is
  recognized*, and `docs/checks.md` a row for the converter and one in the
  uncontrolled-requirements table.

### Measured

- `safetensors` 0.8.0 does not write a byte-reproducible header: two writes of
  one identical state dict differ in the header and agree on its length, with
  and without sorted keys. A produced artifact's recorded digest is therefore
  an integrity record of the file that was deployed, not a reproducibility
  claim; the snapshot's small files do reproduce byte for byte.
- An FP32 tensor loaded into a BF16 module comes back BF16 and is not an exact
  round trip, so a candidate's FP32 vision patch embed does not survive
  ComfyUI's H3 construction. It is a property of how the scales were observed,
  not of how the artifact runs here.
- The release's flat `preprocessor_config.json` and v1's nested
  `processor_config.json` construct the same slow `Qwen2VLImageProcessor` apart
  from `size`; the release file omits `resample` and the `do_*` flags and the
  class defaults supply v1's values.

## 0.67.1

### Changed

- **`docs/research/sglang_comparison.md`, `docs/comfyui_vendor_gaps.md`**:
  the breakable-CUDA-graph entry said ComfyUI had no equivalent. It does:
  core captures per-block graphs inside the dynamic-VRAM prefetch queue,
  replayed only on a matching weight placement, and enables them for decode
  paths and MiniMax Music but not for the H3 DiT loop. Recorded with what
  bounds the gain here (sampling measured at full SM occupancy and pegged
  power on 2026-08-18), that sglang's own H3 path leaves attention eager under
  its graphs, and the single-frame small-canvas case that is still untested.

## 0.67.0

### Added

- **`MiniMaxH3AppendRefImage.qwen_short_edge`** (default 0, appended after
  `short_edge` so saved graphs keep their positional widgets): a view of the
  reference for the text encoder alone. 0 is today's path byte for byte, one
  tensor for both consumers. N scales the source so its shorter side reaches
  N, nearest 32, one Lanczos resample, for Qwen3-VL only; the video VAE keeps
  the stage-one view, and under `image_policy` `encoder` / `release` the
  stage-two bounds are pre-applied to the Qwen view alone. The B arm of the
  reference-view ablation (A no upscale, B Qwen-only 2048, C full parity);
  section 1b of `docs/h3_conditioning_end_to_end.md` records why the two
  branches need not share a geometry. Under the current W4 snapshot the view
  is clamped back to about 265 tokens, which the tooltip and preflight say.
- **`bench/preflight_graph.py` prices the two views per reference**:
  reference-latent rows from the VAE view and Qwen tokens from the Qwen view
  under the loaded encoder's contract (or Comfy's defaults), with a total for
  each and a note on the line when the Qwen view was clamped and by whom.
- `bench/check_reference_runtime.py`: `qwen_view_is_separate_from_the_vae_view`
  and `preflight_prices_the_two_views`; red harness M9 feeds the Qwen view to
  the VAE.

## 0.66.0

### Added

- **`bench/h3_awq_recipe.py`**: the W4A16 AWQ v2 candidate recipe as a
  constructible object, and `assert_decoder_only_boundary`, which grades a model
  the quantization config has already been applied to rather than grading the
  patterns that produced it. The ignore list is two anchored module-name
  patterns that also hold for a reduced-width model; the assertion does not
  consult them, recovering the vision tower, the decoder layers and the output
  head from the model itself, so an upstream rename goes red naming the modules
  instead of quietly quantizing the vision side. The input embedding needs no
  ignore entry because it is not a `Linear`, and the assertion says so rather
  than the docstring. `describe_recipe` and `resolved_awq_mappings` put the
  modifier fields and AWQ's smooth-to-balance resolution into a report.
- **`bench/probe_awq_recipe_boundary.py`** and
  **`bench/results/2026-08-25_awq_recipe_boundary.json`**: the candidate driven
  through the pinned session's own `initialize` path -- the call `oneshot`
  makes, not a reimplementation of config application -- against the
  reduced-width Qwen3-VL and against the released config instantiated on meta
  tensors. No weights, no download, no GPU, and the working directory's entries
  identical before and after. The released arm quantizes 448 Linears, seven in
  each decoder layer and nothing else; its resolved ignore list and weight
  scheme are identical to the deployed artifact's, element for element and field
  for field, compared through the compressed-tensors serializer that wrote that
  file. Three red controls ran and each failed as required: an ignore list
  without the DeepStack mergers, the rejected preflight's `scheme` alongside
  `config_groups`, and a nonexistent `AWQModifier` field.
- **AWQ resolves three of its four default mappings per decoder layer**,
  recorded in that result. Under grouped-query attention the declared
  `v_proj -> o_proj` mapping never resolves, because the resolver drops a
  `.v_proj` smooth layer whose `out_features` differ from the balance layer's
  `in_features`. `o_proj` is therefore quantized without being smoothed, on the
  released shape and on the reduced-width one alike.

## 0.65.0

### Added

- **`comfy_exact_bf16_store`** in `bench/h3_calibration_precision.py`: the
  `comfy_exact` arithmetic over BF16-stored weights, cast to FP32 per call at
  the functional layer so it reaches the computation under Accelerate's hooks,
  compressed-tensors' offload, and the sequential pipeline's traced subgraphs
  alike. `storage_dtype` / `storage_policy` keep the patch-embed conv FP32 at
  load; any activation below FP32 reaching a patched op raises. Measured
  bit-identical to FP32 storage on four released-weight fixtures.
- **`bench/h3_attention_kernel.py`** and **`bench/check_attention_kernel.py`**:
  whether SDPA sees grouped-query or expanded key/value heads, as a scoped,
  counted, revertible switch; `--attention` on the layer-49 comparison and the
  pilot.
- **`bench/check_calibration_precision_policy.py`** gained the manual-cast arm.
- Results under `bench/results/` dated 2026-08-25: the storage axis, the
  kernel matrix on four fixtures with an early-tap control, and the composed
  path running the whole Gate 2A population through the bridge.

### Changed

- **`bench/compare_transformers_comfy_layer50.py`** loads at the policy's
  storage dtype, takes `--attention`, and accepts `--field-under-test` so a
  policy-versus-policy or kernel-versus-kernel comparison between two
  Transformers arms is declared rather than refused; against the ComfyUI arm
  both are exempt by construction.

## 0.64.0

### Added

- **`ALLOW: (none)` in the `retraction-consumers` ledger**: the row shape for a
  claim deleted repo-wide rather than caveated. Such a claim has no consumer to
  enumerate, so the row becomes a tripwire -- any occurrence in any file is a
  reintroduction. A bare empty `ALLOW:` still fails the parse, so the state is
  opt-in and a truncated line is still caught.

### Fixed

- **`bench/check_retraction_consumers.py` printed advice that could not be
  followed.** Its `stale_allowlist` warning said to drop the named file from
  the row; when that file was the row's only entry, doing so emptied the row
  and turned the warning into `FAIL parses_the_ledger`. The warning now names
  the sentinel in that case. The row for the withdrawn adaln figure's second
  spelling had been warning since 2026-08-20 with no reachable remedy, and is
  now `(none)`.

### Changed

- **`docs/evidence.md`** records two retractions moved out of `CLAUDE.md`: any
  A/B of a numerical knob resting on one rendered clip per arm, including the
  2026-08-13 comparison that chose `fp16 (most accurate)`, and the 2026-08-15
  ordering arms, whose per-clip seeds differ so they were never a matched pair.
  The rule stays in `CLAUDE.md`; the retraction records now sit in the file
  that owns retractions.
- **`CLAUDE.md`** drops two passages of its own edit history and replaces the
  hand-maintained `coderef/` roster, which had drifted, with `ls -l coderef/`.

## 0.64.0

### Fixed

- **`image_policy=encoder` and `video_policy=encoder` read the current W4
  artifact's snapshot whichever CLIP the graph had loaded.** A stock-loader
  graph on `encoder` was pre-sized and priced at bounds no loaded encoder
  declared, and a v2 artifact would have been priced at v1 geometry. The
  loader now stamps its artifact's own declaration on the CLIP it builds
  (`_h3_encoder_contract`), the conditioner reads it back off the CLIP it was
  handed, and a CLIP that declares nothing resolves to the native path once,
  logged. `snapshot_contract(directory)` reads any artifact directory carrying
  the two processor files, so a candidate shipped as an HF directory needs no
  table row.
- **`bench/preflight_graph.py` priced `encoder` graphs without knowing the
  loader.** It now walks the conditioner's `clip` link to the loader node,
  resolves the same contract statically, reports which one priced the rows,
  and prices the encoder video grid from it instead of leaving that line out.

### Added

- `bench/check_reference_runtime.py`: `encoder_policy_binds_to_the_loaded_clip`
  (bare CLIP resolves to native; a stamped contract applies; a different
  stamped contract applies differently; a partial stamp is refused) and
  `preflight_resolves_encoder_from_the_loader_node`. The red harness gains
  M7 (module default whichever CLIP) and M8 (encoder kept on a bare CLIP).

## 0.63.0

### Added

- **`bench/h3_producer_provenance.py`**: every Gate 2A pilot and backend-probe
  report now names the commit, file hash and dirty state of the producer that
  wrote it. The readiness record accepts a floor only from one committed
  harness version, and nothing in a report said which.
- **Gate 2A record from one harness version**, under `bench/results/` dated
  2026-08-25: the abort control, the primary escalation, the 2048-upscale
  stress arm behind a small trace row, and the backend-selection probe at
  every real row length and vision-block size. The deliverable is
  `docs/research/qwen3-vl-special-tokens-post-training/brainstorming/claude-encoder/2026-08-25-gate2a-corrected-floor.md`.

### Changed

- **`bench/pilot_sequential_feasibility.py`** reads the all-ones-mask
  omission back from the dataloader, the intermediates cache and the traced
  subgraphs; samples cache growth after every forward instead of reading the
  end-of-run residual; times the trace as its own stage; and records host
  memory before and after the bridge load.
- **`bench/probe_sdpa_backend_selection.py`** keeps the dispatched op name when
  a call runs out of memory, and sweeps the expanded-KV text shape beside the
  grouped-query one.

## 0.62.0

### Fixed

- **`bench/count_packed_rows.py` was broken outright by 0.61.0.** It imports
  the sizing helper that moved into `reference_geometry`, so every invocation
  died with `ImportError`. Its own docstring explains why the fix is another
  import and not a local copy: it rounds where the pricing truncates, and the
  first version of that file reimplemented the arithmetic and was wrong by 8%.

- **The identity guard dropped `_resize`'s channel normalisation.** `_resize`
  does `image[..., :3]` on every other path, so an RGBA reference already at
  its target size reached `vae.encode` with four channels where every earlier
  render had been sliced to RGB.

- **Two preparation stages were merged instead of composed.** A saved graph
  wiring the retired fit node upstream of the append had the two argument sets
  combined by taking the larger `short_edge` and OR-ing the upscale flags. They
  compose: the fit sizes the source, the append sizes that result. The merge
  over-priced by the square of the ratio whenever the append was narrower --
  `Fit(2048, upscale) -> Append(1024)` really yields 1024x1024 and was reported
  as 2048x2048, a 4x error per reference in both `bench/preflight_graph.py` and
  `bench/audit_shipped_reference_bounds.py`.

- **`bench/preflight_graph.py` ignored `image_policy` entirely**, so it priced
  the role size and not the geometry the DiT receives. Under `image_policy=encoder`
  a 2048x2048 reference is pre-resized to 544x544 before the VAE, which the
  tool reported as 4,096 rows against a true 289. Its `_vision_bound_warnings`
  also asserted the opposite of what `release` and `encoder` do, and is now
  reported only under `comfy`, which is the policy it describes.

- **Widget-to-input conversion crashed or silently lied.** `short_edge` wired
  to an input socket raised `TypeError` from `int([...])` and abandoned the
  whole report; `allow_upscale` wired the same way evaluated a non-empty list
  as True and priced the reference as upscaling with no marker. Both are
  ordinary frontend actions. Unresolvable values are now named and that
  reference is reported as not priced, which is preflight's documented
  "reports, never refuses" contract.

- **`MiniMaxH3ReferenceFit.keep_towers_matched` went inert in silence.** It was
  the input that actually did something and was set on every shipped graph,
  while its retired sibling -- which never worked -- got the only warning. It
  now says so and names `image_policy`, whose default is off.

### Added

- **A runtime control that the append's sizing reaches the encoded geometry.**
  Nothing asserted that `short_edge` and `allow_upscale` are read by
  `_compile_reference_records`; replacing them with the function defaults left
  the node-id, reference-fit and typed-consumer checks green while every
  shipped graph lost its upscale. Asserted through the registered node against
  the latent grid the DiT is handed, and shown to go red under exactly that
  mutation.

### Changed

- **The still-image policy resolver moved into `reference_geometry`**, so the
  conditioner and the static readers share one implementation of stage two as
  they already do of stage one.

- **`_fitted` in the bounds audit defaults to the node's own `match`**, not to
  the `max` every shipped graph happens to set, and reports one unpriceable row
  instead of raising and abandoning the run. The audit also records that it
  models the `comfy` policy only.

## 0.61.0

### Added

- **`MiniMaxH3ReferenceConditioning.image_policy`, the still-image sibling of
  `video_policy`.** One Qwen still-image ceiling had three live values -- the
  installed ComfyUI code path's `process_qwen2vl_images` defaults, the loaded
  encoder artifact's snapshotted bounds, and the release's declared bounds --
  and nothing could select between them. The new input does, with the same
  three arms and the same shape as the video policy: `comfy` (default,
  passthrough, exactly what every graph got before), `encoder`, `release`. The
  selected policy's `smart_resize` is applied *before* the VAE encode, so the
  VAE and Qwen encode one tensor at one size rather than the DiT receiving
  latent rows at one resolution and hidden states at another for one reference.
  Both the ceiling and the **floor** apply; the mechanism this replaces only
  ever clamped a ceiling. `h3_awq_encoder.source_image_patch_geometry()` is the
  one accessor that had to be added.

- **`reference_geometry.py`, the single implementation of reference-image role
  sizing.** `fit_reference_image` is called by the conditioner (which performs
  the resize), the retired fit node, and the static readers
  `bench/preflight_graph.py` and `bench/audit_shipped_reference_bounds.py`. The
  append node records the decision and never sizes anything, because the canvas
  `match` needs is not in its scope. Importable by the post-training
  calibration builder, whose accepted plan names two strata that are exactly
  its arguments.

### Changed

- **Reference sizing folded onto `MiniMaxH3AppendRefImage`.** It gained
  `allow_upscale` and `short_edge`, appended after `references`, and the
  shipped graphs now wire `LoadImage` straight to the append. The conditioner
  performs exactly one resize with the target canvas in scope, where before the
  fit node resized and `_compile_reference_records` resized again -- a full
  second lanczos pass and a second float32/uint8/float32 quantization on every
  shipped image reference, because `comfy/utils.py::lanczos` has no identity
  short-circuit. An identity guard skips the resize entirely when the target
  already equals the source.

- **The sizing decision is now visible to static analysis.**
  `bench/preflight_graph.py` reads `allow_upscale` -- it read `size_policy` and
  never read `allow_upscale`, because that knob lived on a node the chain model
  could not see. `bench/audit_shipped_reference_bounds.py` reads the append
  node's arguments directly rather than recovering them by walking the first
  linked input for eight hops, and drives the shared sizing function instead of
  allocating and resizing a real tensor to learn two integers.

- **`MiniMaxH3AppendRefImage` warns instead of dropping frames silently.**
  Wiring a video loader's IMAGE output kept only the first frame with no
  message. It also carries the reference aspect gate now, so an unconditionable
  reference fails before the VAE is touched, and warns when `size_policy=match`
  is paired with knobs `match` does not read.

### Removed

- **The experimental downstream clamp lift, and the Qwen ceiling clamp on the
  image fit node.** Both inputs remain on `MiniMaxH3ReferenceFit` as inert,
  documented legacy so saved-graph widget positions stay valid; the machinery
  is gone. `lift_downstream_clamp` armed an override on
  `MiniMaxH3ReferenceToVideo`, which no graph in this repo wires, and which
  both consumers of the rebound constant had already bound by value at import,
  so the arm was invisible to them regardless. Its `fingerprint_inputs` hook
  returned `float("nan")` whenever the flag was set, so arming a feature that
  could never work also permanently disabled that node's cache.
  `keep_towers_matched` read Comfy's function default as though it were
  universal; that decision moved to `image_policy`, where the CLIP is in scope.
  `MiniMaxH3ReferenceVideoFit` keeps its copy and keeps reading Comfy's
  default, which is correct for a reporter on native-core paths.

- **`bench/check_short_edge_override.py`**, with the code it guarded. It was
  green because its fixture supplied synthetic prompts containing
  `MiniMaxH3ReferenceToVideo`, so its input pre-satisfied its outcome and it
  could never notice that no shipped graph contains that node.

## 0.60.0

### Added

- **A repo-local loader for compatible compressed-tensors W4A16 AWQ H3
  encoders.** `MiniMaxH3AWQEncoderLoader` offers the real `text_encoders`
  population and validates the selected checkpoint's embedded
  Qwen3-VL-32B/group-128 config, retains the 50 H3 language layers, maps the
  full Hugging Face namespace into core's native H3 model, exposes the packed
  int4 weights to comfy-kitchen without a weight-sized repack, and installs the
  source image/video preprocessing on that CLIP instance. It is not bound to a
  filename: metadata, packing, every required native tensor, and shapes are the
  acceptance boundary. FP32 H3 activations are narrowed only across the BF16
  W4A16 matmul so kitchen selects its CUDA kernel, then restored to FP32 for
  residual arithmetic. The small source
  configs, tokenizer/processor declarations, quantization recipe and digests
  are versioned under `config/qwen3vl_32b_minimax_h3_w4a16_awq/`; the model
  remains an external symlink. `check_h3_awq_encoder.py` proves the distinction
  the filenames hide: core natively recognizes the shipped NVFP4-AWQ artifact,
  while this compressed-tensors W4A16 artifact still needs the local adapter.
  This is not a claim that ComfyUI lacks AWQ support.

- **A self-contained Hugging Face distribution of the AWQ loader.**
  `bench/build_h3_awq_standalone.py` deterministically embeds the four runtime
  JSON configs in one directly discoverable custom-node `.py` and adapts three
  official native-Comfy H3 templates for text-to-video, image-reference, and
  first-frame use. The FL2VA copies explicitly carry the provisional v1.1
  owner recipe (strength 0.75, six render steps, shift 6/3); the ref2va copy
  remains a separate base recipe. `check_h3_awq_encoder.py` compares embedded
  config digests and critical implementation source, exercises direct V3
  registration, validates the generated workflow population, and performs the
  full CPU construction through the generated module. This does not bundle
  the research repo's typed conditioning or reference-video policy nodes.

- **A distilled graph is now checked against the grid it was distilled at.**
  `bench/check_distill_grid.py` grades where the scheduler actually puts the
  steps, which `check_distill_settings.py` never saw: that file grades the
  shift and the step count, and both can be right while the sampler evaluates
  somewhere else entirely. It computes no expected values of its own. The grid
  comes from the rule and the literal NFE=4 sigmas published in the vendor's
  `Minimax-H3-Turbo` README, and from ComfyUI's own `calculate_sigmas` over the
  object `MiniMaxH3SigmaShift` builds, plus `time_shift_sigma` for the audio
  schedule the DiT derives. Red harness `bench/red/show_red_distill_grid.py`:
  eight mutations red across both sources, seven near-misses green.

- **`bench/preflight_graph.py` grades `embedding:` references.** ComfyUI
  resolves them in an H3 prompt and, when the file is missing, logs a warning
  and renders anyway, so a queued job silently loses a concept with nothing in
  its output saying so. Preflight now reports each reference as resolved,
  missing or present-but-ungradeable, and prices the rows a resolved one
  contributes into the packed sequence.

- **The capture manifest records model file digests** (`models.sha256`).
  Model identity was the filename and nothing else, with `rank` regexed back
  out of it; a file replaced in place left every field unchanged. Hashing is
  what reference media and captured tensors already got. What it cannot see is
  stated in the schema rather than left implied: a runtime patch, a LoRA
  strength, a post-load quantization all hash identically.

- **The replacement AWQ v2 lane now has an executable native-H3 calibration
  seam proof.** Real H3-IR still, multi-reference, keyframe, mixed, and
  reference-video inputs are produced by the installed presentation path and
  re-identified through a preconstructed `llm-compressor` dataloader, its
  intermediate cache, and the traced graph. The same work strictly loads the
  released checkpoint, holds the BF16 vision/DeepStack/embedding boundary, and
  repairs the candidate-pool media gate so videos are opened and hashed rather
  than trusted by filename. Released-weight comparison also found that
  ComfyUI's BF16 position interpolation plus FP32 manual-cast linears is not
  reproduced by either plain Transformers dtype; a bounded hybrid-precision
  gate now precedes the one-4090 feasibility pilot.

### Changed

- **H3 special-token ownership is now exclusively native ComfyUI.** Commit
  `924743af` adds the seven release tokens in `MiniMaxH3Tokenizer`, so the
  FL2VA and ordered Ref2VA conditioners no longer import, construct, clone or
  mutate tokenizers. New API workflows omit `vendor_tokens`. The old input
  stays optional and ignored at its original schema position, and the
  deprecated standalone node is a pass-through tombstone, so saved UI graphs
  keep loading without carrying two tokenizer implementations.

- **Generated graphs and the e2e bench now load the canonical W4A16 encoder
  through its format owner**, not through core `CLIPLoader`. Core's menu did
  offer the filename, but the executed job selected Qwen3-VL-8B from the full
  HF namespace and failed width 4096 against H3's 5120. `check_model_files.py`
  now separates discovery from format ownership, and
  `check_bench_matches_shipped.py` grades the loader and filename alongside the
  attention settings.

- **Reference-video preprocessing defaults to `video_policy=encoder`.** It
  keeps ComfyUI's cheaper, no-upscale VAE view while sending the raw 2 fps Qwen
  samples through the custom encoder's source-config duration-aware processor.
  `comfy` remains the native preprocessing control; `release` remains the
  opt-in full local parity policy that also moves the VAE view to the release
  canvas. All three are repo-local choices around core's unchanged reference
  node, not native ComfyUI fixes. A red control deliberately makes `encoder`
  read the release snapshot, so today's equal config values cannot hide which
  artifact owns that policy.

- **v1.1 is the canonical 768p LoRA on every operative surface.** Moving the
  constant was not the change: sixteen graphs loaded v1.1 while their own
  generated help text still read v1.0, which is the right file under the wrong
  instructions, and nothing in the suite read a note. Generator notes, the
  config commentary, the graph help text, and the operative advice in
  `docs/h3_ref2v_distillation.md` now name v1.1. The version a note SHOWS is
  derived from the filename that owns it (`h3_config.turbo_label()`) rather
  than typed beside it, and `check_distill_settings.py::notes_match_the_lora`
  is the control -- it went red on the sixteen stale graphs before the
  regeneration that fixed them. Only claims a note makes about its own graph
  are graded, so the comparison table naming the other four LoRAs stays green.
  v1.0 is historical, not a fallback: it survives where it records genuine
  provenance (the vendor's documented row, the measured distillation table, the
  dated 2026-08-20 power-limit run) and nowhere else.

- **The 768p arm renders at 6 steps, strength 0.75** (v1.1 only; SLA and the
  8-step keep the vendor's 1.0). Owner-selected on their own trials, provisional
  and unscored. Kept structurally distinct from what the student was distilled
  to do: `TURBO_768P_DISTILLED_STEPS` holds the vendor's 4 NFE and
  `check_distill_settings.LEGAL` still grades that row against the vendor, while
  `OWNER_RECIPE` records what we actually run. A dedicated
  `TURBO_768P_STRENGTH` rather than the shared `TURBO_LORA_STRENGTH`, so moving
  it cannot move every other turbo arm. The recipe never touches the shift: 6/3
  is the training value and stays. Filename, shift, steps and strength are now
  graded as one configuration -- strength was read by nothing until now, and
  three of four fields staying right is exactly how the fourth drifts.

- **Six steps does not divide the 1,000-step training grid, so those arms are
  partitioned out of the exact-grid claim rather than tolerated inside it.**
  `check_distill_grid.py` now splits vendor-grid arms from owner-recipe arms:
  the first keep the exactness assertion, the second get a weaker claim stated
  as weaker -- the deviation must be declared, and `simple` must still be
  strictly the nearest scheduler at the arm's own step count. Loosening the
  tolerance would have destroyed the vendor claim for all 22 remaining arms at
  once; rewriting the vendor row to say 6 would have erased what the student was
  distilled to do. An arm at neither a distilled count nor a declared recipe
  fails, so a recipe cannot arrive by editing a widget.

- **The 768p turbo arm runs v1.1, not v1.0** (`TURBO_768P_LORA`, 16 graph
  references). v1.0 is no longer on this machine and every reference to it was
  red at the loader. The vendor still documents no v1.1 row, so its 6/3 shift
  and 4 steps are inherited from v1.0's row by filename family rather than
  attested, and that is declared rather than assumed:
  `check_distill_settings.UNATTESTED` names the row and its reason, fails if a
  LEGAL row is neither found in a vendor source nor declared, and fails again
  if a declared row is later carried by one. The assertion that previously
  refused to classify v1.1 is inverted rather than deleted, because
  classification is what hands a file its shift and this is the one row whose
  shift came from a filename. v1.0 remains published if the inheritance is
  ever doubted.

- **The release video policy refuses a clip it cannot size, with its own
  message.** One 2 fps sample reached the release `smart_resize`, which raised
  `t:1 must be larger than temporal_factor:2` from inside transformers, naming
  neither the reference nor the policy. Reference lengths snap to 17n+5 and the
  Qwen sampler steps by 12, so release mode's real minimum is 22 prepared
  frames; 5 samples to one. This is error handling for a release requirement,
  not a new floor: comfy mode still accepts 5 frames, and
  `check_reference_runtime.py` asserts both ends plus the boundary.

- Capture manifest `schema_version` is `1.2.0`. The new digests are gated on
  it, so the 1.1.0 manifests already on disk conform to the version they
  declare and are not failed for a field that did not exist.
- `check_distill_settings.py`'s graph readers also return the scheduler, so the
  new check reuses them instead of walking the same JSON a second time.
- `workflows/h3_config.py` names the control for its `simple`-scheduler claim,
  which was prose enforced by nothing, and records that DiffSynth's H3 pipeline
  defaults to flow shift 2.22 on video and audio alike where ComfyUI and every
  vendor row use 12/3 or 6/3. The rule is the same in all three; only the
  constant differs.

### Not adopted

**Nothing here was ever committed, so `git log` shows no deletion** -- these
arrived as working-tree files, were reviewed, and were dropped before landing.
Recorded because the two gaps they found are real and became the entries above,
and because the next person to meet the same source should not re-derive why it
was declined. The detail is in the note under `internal/gemini/`.

- An imported flow-schedule module, embedding-injection nodes, and a weight
  fingerprinting module, together with four proposed nodes and three
  checks. The schedule was keyed on a shift belonging to a different
  implementation and fed nothing; the embedding nodes counted an `[N, 5120]`
  injection as one sequence position where ComfyUI expands it to N, and formed
  a third conditioning entry point that skipped both the vendor-token
  registration and the empty-prompt refusal that `MiniMaxH3Conditioning` owns;
  the fingerprinting module was imported by nothing but its own check. Node
  registration is unchanged, `node_id` baseline included. Static
  `embedding:name` resolution is a ComfyUI feature and needs no node here.

## 0.59.0

### Added

- **A typed, ordered reference runtime now exists.** Three copy-on-append nodes
  build `MINIMAX_H3_REFERENCES` image, video(+soundtrack), and standalone-audio
  records; `MiniMaxH3ReferenceConditioning` compiles that one list into both
  Qwen's presentation items and the DiT reference blocks. List position is the
  authority, a video owns its soundtrack and VHS metadata, and arbitrary
  cross-modality order no longer has to be simulated through parallel sockets.

- **Reference media is normalized at that typed boundary.** `loaded_fps` comes
  from the frames' own `VHS_VIDEOINFO`; non-24-fps clips are resampled before
  either encoder, mono audio is duplicated to stereo, and every soundtrack and
  standalone audio reference is capped at the aligned target duration. Still
  images retain their per-record policy and video geometry defaults to
  Comfy-compatible no-upscale behaviour.

- **Release reference-video preparation is one named opt-in policy.** On
  `MiniMaxH3ReferenceConditioning`, `video_policy=release` puts the full-rate
  VAE view on the release canvas and independently runs raw 2 fps samples
  through the release's duration-aware Qwen processor. `comfy` remains the
  default. This is local parity handling over native-open gaps, not a native
  ComfyUI fix.

- **CPU acceptance and red controls for the runtime.**
  `check_reference_runtime.py` covers copy-on-add order, runtime metadata
  ownership, fps normalization, mono/duration normalization, and the sounded
  video's two-Qwen-items/one-DiT-block contract, release Qwen's raw-sample
  boundary, and the policy's distinct VAE/Qwen views. It also drives a lazy
  `Mapping` shaped like VHS's `LazyAudioMap`, after the first multimodal smoke
  found that accepting only core's concrete audio dict rejected a schema-valid
  VHS soundtrack at execution. Its red harness detects five independent
  regressions, including collapsing release Qwen back onto the VAE frames.
  `check_typed_reference_consumers.py` proves label discovery and
  preflight read the same typed chain and refuse malformed plans.

### Changed

- Prompt-label discovery and static preflight now recognize
  `MiniMaxH3ReferenceConditioning`. Preflight selects `ref-en`, prices image
  policies and discovers video/audio media through the validated chain, and
  reports that trim/mono normalization is owned by the typed compiler.

- **Every shipped reference workflow now uses the typed surface.** The
  generator emits image append records first, then the video with its owned
  soundtrack, then standalone audio, preserving every existing prompt ordinal
  while replacing suffix pairing with ownership. UI and API forms agree on
  repeated append-node counts. Explicit `TrimAudioDuration` nodes are gone;
  the compiler owns the cap and cannot drift from a patched `length`.

- The migration deliberately leaves `force_rate=24`, image fit settings, and
  no-upscale video geometry unchanged. Those remain native ComfyUI/vendor
  divergences; the typed nodes are this repo's handling, not upstream fixes.

- The ordered resolver admits only the four served VHS video-loader classes,
  traces both branches of `JoinAudioChannels`, and exposes one validated entry
  view for static consumers. Matching output slots on an unrelated node are no
  longer accepted as video plus `VHS_VIDEOINFO`.

- Schema collection, preflight, and the reference-fit acceptance check
  explicitly import Comfy in CPU mode. These checks no longer select the CUDA
  device merely to read schemas or scalar H3 geometry while a render owns the
  GPU.

### Fixed

- **Base-guide alignment is now exact, not presence-only.** Preflight parses
  I2VA/FL2VA/L2VA's literal Part One templates from the release guide, resolves
  FL2VA/L2VA's final shot and snapped `length / 24`, and rejects a plausible
  sentence for the wrong mode, wrong duration, or wrong final shot. The red
  harness covers all three.

- The shipped-reference bounds audit follows image links through typed append
  chains and refuses malformed chains instead of returning a partial clean
  answer. Its fit-path red control now explicitly disables
  `keep_towers_matched`; shipped graphs keep that safety guard enabled.

- `smoke_h3.py --log` now reports a missing launcher log as an unchecked
  diagnostic (exit 2) after a successful render, rather than throwing a
  `FileNotFoundError` that makes the render itself look failed.

### Live accepted; shipped migration complete

- ComfyUI was restarted onto commit `0b665b7` after the GPU became free, and
  `/object_info` served all four new schemas. A scratch two-image append chain
  validated against that live schema and rendered at 768x768, 39 frames and 10
  steps in 49.80 seconds. The server log records two ordered picture entries,
  the typed payload reaching preflight at 9,578 packed rows, and the native
  tokenizer already carrying all twenty tokens, so the compatibility shim was
  a no-op.

- The regenerated population contains 39 reference API graphs and 79 typed
  UI/API/stamped files, with no shipped `MiniMaxH3ReferenceToVideo` or
  `TrimAudioDuration` node. Prompt labels, guide conformance, typed consumers,
  reference ordering, generator constants, preflight, and the real-image Qwen
  bounds audit all pass.

- A second live smoke exercised every reference medium at 1024x768, 39 frames,
  and 10 steps. The first attempt exposed and fixed VHS lazy-audio compatibility;
  the rerun completed in 84.51 seconds. The server logged
  `['<Picture 1>', '<Picture 2>', '<Audio 1>', '<Video 1>', '<Audio 2>']`,
  21,283 packed rows, Sage routing, and Sol sparse execution. Models were
  unloaded afterward.

- Full live schema validation now reaches the current code. Its only red rows
  are eight existing graphs that name a removed v1.0 768p Turbo LoRA while
  this installation carries v1.1; every migrated reference UI graph is clean.

- The opt-in release-video probe also passed live at 1024x768, 39 frames, and
  10 steps in 92.73 seconds. The server logged its 960x544 source prepared at
  1344x768 for the VAE, four raw Qwen samples, 23,892 packed rows, Sage routing,
  and Sol sparse execution. Models were unloaded afterward. The long-duration
  processor boundary is pinned by the CPU check rather than an expensive
  full-length render.

## 0.58.0

### Removed

- **The denoising-trajectory pair is gone from every UI graph.**
  `GetPreviewOverrideFramesKJ` ("Preview frames (trajectory)") and its
  `PreviewImage` sink ("Denoising trajectory") were in all 59 UI graphs and no
  API graph, so nothing measured changes. `Preview (taeh3)` is all these graphs
  need and stays in all 59. `PreviewImage` stays in `_UI_ONLY` even though
  nothing emits one now: it is a stock node somebody may add by hand.

### Added

- **Two fl2va graphs ship, and one of them is the first turbo arm this repo has
  run in distribution.** `h3_first_last_frame_to_video.json` and
  `h3_first_last_frame_to_video_turbo_4step_768p.json`: two keyframes into one
  continuous shot, 1152x768 fallback canvas, 362 frames, matched seed and
  placeholders so the pair is comparable by construction.

  **Every released turbo LoRA is an fl2v distill** -- the filenames say so and
  `_NOTE_REF2V_TURBO` says so -- and until now every turbo graph here was t2v or
  ref2v. So every turbo number recorded in this repo was taken out of
  distribution, including the arms that reason carefully about how far out. The
  turbo graph is the reference point those arms have never had. It is not an
  answer to "is turbo good": a rendered pair cannot A/B a numerical knob.

  The canvas is a FALLBACK. Under `from_keyframe` the geometry comes from the
  loaded first frame, as the release resolves it on `keyframes[0]`, and the
  closing frame cover-crops to match. 1152x768 governs only under `explicit`,
  and it carries the 768 short edge the 4-step LoRA is named for.

  `last_frame` runs through both builders; `cross_check` is what asserts they
  agree. The task string stays `i2v`, because one wired frame and two share the
  geometry and canvas logic exactly and a fourth task value would fork it.

### Fixed

- **The fl2va alignment sentence is the one of three that carries no brackets,
  and two places had it wrong.** `base_en.md:14-32` gives one string per task:
  I2VA and L2VA bracket (`<Picture 1> (from [Shot 1])`), FL2VA does not
  (`Picture 1 (from Shot 1)`), T2VA has none.

  `scene_prompt()` prepended the I2VA sentence whenever `first_frame` was set,
  so a first+last call emitted the I2VA line for an fl2va task and a last-only
  call emitted no line at all -- which `preflight_graph.grade` fails outright.
  The function is still uncalled, so no graph carried the defect, but it was
  staged for exactly the task that would have hit it first.

  The default-prompt lookup keyed on the task string alone, so an fl2va graph
  built through it would have been handed `I2V_PROMPT`. It now follows the
  wired sockets in both builders.

  `fl2v_prompt()` derives the sentence's `S.SS` from the snapped frame count
  rather than carrying a typed duration, so it cannot drift from `length`.

- **`preflight_graph.py` would have failed the first correct fl2va prompt
  anybody wrote.** Its label rule demanded `<Picture N>` with brackets, which is
  right for every keyframe graph except the two-frame one, whose guide-mandated
  sentence is bare. The rule was written when i2v was the only keyframe graph in
  the repo; it was correct about that graph and untested against any other.
  Second implementation, not reasoning, is what found it.

  The waiver is narrow: the bare form satisfies the requirement only on a graph
  wiring both keyframe sockets, and only for `Picture`. Shown red on three
  mutations (a picture never named, a picture no socket wires, no alignment line
  at all), and i2v with a bare label still fails.

  **Closed in 0.59.0:** preflight now compares the exact guide-derived template
  for the resolved mode, final shot, and snapped duration; the red harness
  exercises the wrong-mode sentence that used to pass.

## 0.57.0

### Added

- **`preflight_graph.py` grades the five markers neither official guide
  documents.** `<|lyrics_start|>` / `<|lyrics_end|>`, `<|caption_start|>` /
  `<|caption_end|>` and `<|cutoff|>` are in the release's token list; the guides
  name neither the lyrics nor the caption pair, and spell the cutoff marker
  without its pipes. Nothing anywhere asserted anything about them and
  `check_prompt_guide_conformance.py` cannot -- it parses its vocabulary out of
  the guide and refuses to assert what the guide does not state. `marker_rules`
  fails an unbalanced pair, a `<d>` inside a caption pair, a marker pair inside
  a `<d>` and a lyrics pair wrapping no `<d>`; it warns on a padded caption
  string, a marker opening a line, two caption pairs with only whitespace
  between them, whitespace before `</d>`, and a full stop directly before
  `<|cutoff|>` (which BPE drags into the marker on an install lacking the
  tokens, so the marker retokenizes the sentence in front of it).

  **The escaped instance** is a prompt written outside this repo that arrived
  carrying three caption pairs on their own lines, one padded with spaces, one
  spoken line split across two adjacent pairs, and a trailing space inside a
  `<d>`. Graded by preflight as it stood, it scored exactly one WARN, for word
  count. The balance and nesting cases did not escape and exist for the other
  reason: the prompting guide gained a nesting requirement the same hour, and
  this is the assertion behind it.

  Green on all five t2v scenes, both ref-form stress scenes, and seven of the
  nine audit-harness scenes; the two that warn are stressors doing on purpose
  what the rule describes. Each rule shown red on its own mutation.

### Changed

- **The prompting guide now says which marker wraps what.** The only nesting
  that exists is `<d>` inside a lyrics pair -- lyrics wrap the same words that
  are sung, because they mark them as sung. A caption pair is a sibling of `<d>`
  in both directions, placed immediately after the `</d>` it belongs to, and its
  string may differ from the spoken line -- which is what makes speech in one
  language with subtitles in another a caption rather than a mistake.

- **`<|caption_start|>` is read as a subtitle marker, and the signage reading is
  withdrawn.** This repo asserted, in the audit harness and then in the guide,
  that the pair marks burned-in signage. Nothing sourced that. The marker
  appears in neither prompt guide, in no script, in no skill the release ships,
  and in no worked example in the release or any sister checkout -- verified by
  grep, not assumed. Its name is the only evidence available and it sits in the
  declared list beside `<d>`, `<|cutoff|>` and the lyrics pair, all of which
  concern speech. Subtitle is the reading to write to. The scenes in
  `build_workflows.py` use the pair for signage and are left alone pending an
  owner call, since they are the baseline set. Whether the marker renders
  anything at all is still unmeasured, which is why no shipped default carries
  one, and the guide's §4.5 double-quoted string stays the documented route to
  on-screen text of any kind.

- **The prompt writer is told to emit `<|cutoff|>`, with the pipes.** Both
  official guides spell it `<cutoff>`, which matches no entry in the release's declared
  token list; the scenes were moved onto the piped form on 2026-08-22 and the
  system prompt that generates prompts was not. Diverging from the guide's
  prose here is deliberate. `<scenetrans>` stays as the guide writes it, since
  no piped variant exists to prefer.

### Fixed

- **The prompting guide argued for a decision that has since been reversed,
  and still argued for it.** Its tokenizer section read the seven special
  tokens' embedding rows sitting at initialisation as settling what to send at
  serving time. It does not settle it: the rows being untrained and the
  vendor's own tokenizer emitting those ids are both true, which is a
  train/serve mismatch rather than a question with one answer. The measurement
  stands; the conclusion drawn from it, and specifically "do not build a node
  to fix it", is withdrawn in favour of `docs/comfyui_vendor_gaps.md` gap 1 and
  upstream's own fix.

## 0.57.0

### Removed

- **`h3_ref_video_swap_concise` is retired.** The owner watched the
  three-matched-seed batch at the shipped canvas and length: the concise arm is
  **broken speech, gibberish, 3 of 3**. `bench/grade_soundtrack_reuse.py`
  ordered it the same way without hearing anything -- the three lowest margins
  over control, and bad in the FIRST third on two of three seeds where every
  structured render starts at 0.589-0.704.

  **What this does NOT establish is that the six sections are the cause.** The
  two prompts differ in structure AND length AND whether the soundtrack is an
  `<Audio 1>: fully_copy` retention line or prose. Three variables moved
  together; the separating arm was never rendered. The finding is narrower and
  still useful: that prompt breaks speech reliably on a soundtrack-reuse task.

### Changed

- **The shared reference clip is trimmed from 19.56s to 14.375s**, and the trim
  is a fix rather than housekeeping. The model tops out at 362 frames /
  15.083s, so 23% of a continuous monologue was cut wherever 362 frames
  happened to land -- the reference kept talking past the end of the render and
  the last third of every render drifted. The owner heard it before any measure
  showed it. The new cut lands inside a 0.3s silence at -56 dB, so the
  utterance ENDS instead of being interrupted.

  **Still 25 fps on purpose.** Trimming to 24 would have made `force_rate=24` a
  no-op and quietly retired the fps hazard this clip exists to exercise. The
  problem was the length, so only the length changed.

- **Reference-video graphs render at `REF_VIDEO_LENGTH = 345`**, which is
  `14.375 * 24` -- the clip and the render now end together. **This is not the
  2026-08-10 global move to 345 that was reverted on 2026-08-16**: that capped
  every render at diffusers' emit limit and broke comparability with
  measurements taken at 362. `LONG_LENGTH` is untouched; only the thirteen
  graphs wired to this clip move, because only they have a clip to match.

- The clip lives in the input ROOT, not `h3_refs/`. `VHS_LoadVideo`'s `video`
  widget is a combo of root filenames and lists no subfolder paths, so a graph
  naming one fails the served-schema validation -- which is how this was found.

### Added

- **`grade_soundtrack_reuse.py` reports thirds, not just a mean.** The whole-
  window average hid the finding the instrument was built for: structured seed
  894 averages +0.426, mid-pack, for a render that is +0.688 early and +0.017
  by the end. A mean is the wrong summary for a quantity that drifts inside the
  window it is averaged over.

## 0.56.0

### Removed

- **`h3_ref_video_swap_directive`, the community-prompt arm added earlier the
  same day, is deleted.** It was rendered against the structured and concise
  arms on matched seeds at 1152x768 / 124 frames and it damaged the speech in
  the reused soundtrack. The structured arm -- the shipped default, unchanged
  throughout -- was the only one of the three that did not. Owner's call on
  playback; the arm existed to be judged and it was.

  **The finding outlived the graph**, in `docs/h3_references.md`'s swap
  section: two seeds each, one clip, one kind of dialogue, so it is a reason
  to keep writing the `<Audio 1>: fully_copy` retention line and not a
  measured rate. The candidate mechanism -- that the structured prompt is the
  only one of the three stating the soundtrack as a retention marker rather
  than as prose -- is named there and is untested, because three variables
  moved at once.

  The `_STRUCTURE_PROBES` waiver and the drift-guard entry went with it, so
  nothing is left waived for a graph that no longer exists.

## 0.55.0

### Changed

- **Every reference soundtrack now stops where the render does.** sglang caps
  each one at `frame_count / fps` into `ffmpeg -t`, diffusers does the same,
  and both apply it to a video's soundtrack and a standalone audio reference
  alike. `comfy_extras/nodes_minimax_h3.py:71` truncates neither, at 80 rows
  per second of excess attended on every sampling step. Every `ref_audio_*`
  and `ref_video_audio_*` socket in every shipped graph now reaches the node
  through `TrimAudioDuration` at `length / 24`.

  Checked before wiring rather than after: the node **caps and never pads**
  (`end_frame = min(start + duration*sr, audio_length)`,
  `comfy_extras/nodes_audio.py:473-474`), so a soundtrack shorter than the
  render passes through untouched. Padding would have been a change in the
  wrong direction and would have shipped silently.

  **This changes the conditioning of every reference-audio graph.** Renders
  from before it are different samples, not degraded ones.

- **`VHS_LoadVideo.frame_load_cap` now matches the generated length**, and the
  reason is narrower than it looks. Core already truncates a reference video
  to the generated frame count (`nodes_minimax_h3.py:321-322`), so the cap
  saves the full-clip decode and the resize at line 320 -- which runs on every
  loaded frame *before* line 321 throws most of them away -- and saves no rows
  at all on the video side.

  **It does save audio rows, and that was not the reasoning when it landed.**
  `VHS_LoadVideo` asks ffmpeg for `frame_load_cap / force_rate` seconds of
  audio, so setting the cap trims the soundtrack by itself -- measured by
  driving that call: cap 0 yields the full 19.541s, cap 124 yields 5.167s. On
  the `ref_video_audio_*` path the cap and the trim therefore close gap 5
  redundantly. The trim is still what holds if the cap goes back to 0, and it
  is the only mechanism on the standalone `ref_audio_*` path.
  `docs/h3_references.md` carries the per-path table.

### Added

- **`preflight_graph.py` grades reference audio**, and it is the named control
  for two requirements that had none. It warns when a ref-audio socket reaches
  the node untrimmed, and when the baked trim disagrees with `length` -- which
  is the state a bench creates by patching the length alone, as
  `run_graph_arms --set` routinely does. Both shown red by deliberate
  violation.

- **A mono probe, which reports and does not fix.** Preflight ffprobes the
  media each ref-audio socket resolves to and warns on a mono track: core's
  `_encode_ref_audio` does not upmix, so the packed assignment raises rather
  than degrading (gap 7). The upmix belongs in core's encoder, where diffusers
  and DiffSynth-Studio put it; a `JoinAudioChannels(a, a)` in every shipped
  graph would alter every stereo source to prevent a crash no shipped source
  hits. "Channel count unreadable" is reported as its own state, so a missing
  ffprobe cannot read as a pass. Verified against a generated mono file, a
  stereo control and a missing path: 1, 2, None.

## 0.54.0

### Added

- **A community character-swap prompt became the third arm of an experiment
  that had been sitting at two.** `h3_ref_video_swap_directive` carries a
  prompt reported on 2026-08-22 to swap characters reliably, on this repo's
  references, seed, canvas and length -- so it differs from
  `h3_ref_video_swap` (six sections) and `h3_ref_video_swap_concise` (one
  paragraph) in the prompt and nothing else. What it contributes that neither
  twin has: an imperative register, an occlusion clause, and a closing list of
  things not to add.

  **Three edits to the source prompt, and one clause dropped.** Its
  `<Image_1>`/`<Video_1>` tags became `<Picture 1>`/`<Video 1>`, because
  `comfy/text_encoders/minimax.py:164` emits the literal string
  `<Picture %d>: ` into the sequence immediately before each image's vision
  block -- `<Image_1>` names nothing that appears anywhere in the packed
  sequence, so the reported prompt binds by position and description instead,
  and shipping the underscore form would have tested the tags rather than the
  register. Two identities became one, so the arm matches its twins socket for
  socket. `<Audio 1>` was named, because the graph wires it. And "use its
  front, and close-up views as one identity reference" was dropped rather than
  adapted: one image is wired, and a clause naming something not present is an
  invitation to invent it.

  **Whether the tags matter is NOT answered by this arm** and cannot ship as a
  graph -- `check_ref_prompt_labels` rejects a label no socket wires, which was
  confirmed by mutation rather than assumed. Run it as a widget patch through
  `bench/run_graph_arms.py` if it is worth answering.

- **`h3_image_swap`: the swap generalized off the video path, and the only
  graph here that replaces TWO identities at once.** A still plate of two
  people on a loft couch, plus two face references, on the path where a render
  costs seconds instead of minutes. The two-identity case is where the reported
  failure lives -- the model blending the pair, or landing a face on the wrong
  person -- and no video graph covers it. **Chosen so a failure cannot pass for
  a success**: both people in the plate are young and dark-haired, neither
  replacement is.

  It is the only image scene not on the shared 2:3 portrait canvas. The prompt
  promises the plate's framing survives, and a portrait output cannot hold a
  16:9 plate's framing however the model tries.

### Changed

- **`h3_probe_prompt_concise` -> `h3_ref_video_swap_concise`.** The old name
  said nothing about the thing that makes the graph what it is: a video
  reference. It sorts beside its sibling now. `h3_probe_prompt_directive` was
  never released under its first name.

- **`scene()`'s `extra` now overrides rather than merges.** It was
  `dict(**IMAGE_EDIT_BUDGET, **extra)`, which raises on a collision, so the
  only way to change one image scene's canvas was to change every scene's. No
  caller relied on the old behaviour -- a collision could not have shipped, it
  would have crashed the build.

- **One argument against the six-section format is dead, and it was made
  here.** `_NOTE_PROMPT_STRICTNESS` said the sections "cost tokens in a budget
  where reference rows already dominate". Priced by `bench/preflight_graph.py`
  on 2026-08-22: 459 text tokens structured, 172 directive, 92 concise, inside
  a packed sequence whose floor is about 85,700 rows. Half a percent. The
  choice between these prompts is entirely about what comes out.

## 0.53.0

### Added

- **A baseline t2v scene set replaces the single cyclist prompt** every t2v
  measurement here had been taken on. Five scenes, two speakers or more each,
  all seven of the release's special tokens across the set, and soundscapes
  that are events rather than continuous texture. Two of them are stress
  scenes: overlapping speech, singing over the top, high motion and burned-in
  text inside 15.083s. `--list-scenes` and `--print-scene` reach them from a
  shell, and `scene_prompt()` anchors any of them for the keyframe
  permutations. `_ref_prompt(scene=...)` carries them into the six-field
  reference layout, where `detailed_description` goes from 47 words to 360-384
  and lands inside the guide's budget for the first time.
- **The answer to upstream's selection question**, replicated: top-k beats
  adaptive tau at 1.0 by 4.7% (keep 15%) and 10.3% (keep 10%), with the LoRA
  making no difference. `docs/SOLATTN.md` owns it.
- **A speaker-attribution failure the old prompt could not have shown.** Asked
  for a singer in one shot and two different speakers after a cut, 11 of 12
  clips hand the singing to a character introduced after the cut. It spans both
  selections and both LoRAs, so it is not an attention finding.

### Fixed

- **The audio hum was chased to the prompt, not to anything changed that day.**
  Sol on, Sol bypassed and the SLA router all hum identically; what moves it is
  a prompt asking for continuous texture, amplified about 10 dB by 4-step
  distillation. Two earlier readings of mine were wrong and are recorded as
  such -- one compared two variables at once and read a kernel regression out
  of it, the other held the prompt fixed at the single value that produces the
  artifact.

### Changed

- **A same-day measurement is void by replication**, and the rule it produced
  is in `docs/evidence.md`: do not time an arm set on a long-lived ComfyUI
  session. Two independent measures moved together across a restart on
  identical inputs, and the mechanism is left unknown rather than guessed --
  the two obvious candidates are ruled out by evidence, not by suspicion.
- **The ComfyUI core tokenizer patch is reverted** by owner decision pending
  upstream merge, and `vendor/UPSTREAM.md` carries it as a ledger entry with
  what the shim cannot reach. The marker path became load-bearing the same day,
  since every new scene uses `<d>`, and the audit confirms the shim supplies
  all seven markers with the core patch gone.

## 0.52.0

### Changed

- **The Sol-Attn backend moved to the current head of kijai's PR 117**,
  `comfy-kitchen` `0.2.31+sol.c04ef20` -> `0.2.31+sol.23d1a66`, five commits.
  It was already this branch; this is an update, not an adoption. What is in
  the range: a rewritten CUDA routing kernel ("Optimize routing", about a
  third of `sol_attn_route.cu`), `topk_ratio`, a nanobind change, and a move of
  the CUDA sources from `backends/cuda/ops/` to `backends/cuda/sage_attention/`.
  The rest of the diff is an upstream-main merge (HIP/AMD, flash-decode) that
  does not reach this box.
  - **The routing rewrite is the risk, and it was re-graded, not assumed.**
    `bench/check_solattn_correctness.py` puts the rebuilt kernel at cos 0.999919
    against the re-vendored oracle in its own tail mode -- a controlled
    call-level comparison, which is the only kind that answers a numerical
    knob.
  - `bench/_sol_attn_reference.py` re-vendored, `c04ef20` -> `23d1a66`, and its
    three functions confirmed AST-identical to upstream's eager module.
- **The Sol-Attn node is upstream's v3 (sha256 `7805cf37...`), and it is a
  schema change** -- unlike v1 to v2, which needed no regeneration. `tau` and
  `tau_profile` fold into a `selection` DynamicCombo whose other option is
  `top-k (SLA)` with `keep_percent`; `routed_cap_percent` is gone. Every graph
  regenerated in both forms.
  - The node passes `topk_ratio` to the kernel unconditionally, so **the node
    and the kernel have to move together**; neither works against the other's
    old half.
  - The two graph forms spell the selection DIFFERENTLY, which is why one
    module now owns both: the UI form splices the option's widgets in
    immediately after the selector (source read at ComfyUI_frontend v1.49.6),
    the API form keys them under the combo with a dot (`selection.tau`).
  - **ComfyUI's prompt validation does not gate this.** A graph carrying the
    pre-v3 inputs validates clean and dies at execute on `selection["selection"]`,
    so a stale hand-edited Sol node reaches the queue before anything complains.
- **`top-k (SLA)` is not `MiniMaxH3SLARouter`,** though the two are described
  almost identically. Read in the eager implementation: `topk_ratio` changes
  which blocks are marked exact and leaves the pooled tail alone, so Sol still
  adds a term for every block it did not pick, where the router drops them.
  A third attention, not a cheaper spelling of the arm the Turbo-SLA LoRA was
  distilled under. Nothing here renders under it and no graph wires it.

### Fixed

- **`bench/check_workflow_schema.py` counted a `force_input` nested under a
  DynamicCombo as a widget**, applying at the top level a rule it dropped one
  level down. It failed 48 correct graphs on `selection.tau_profile` -- the
  mirror image of the defect its own docstring says it exists to catch, and
  invisible until an option first carried a socket.
- **`bench/check_attention_defaults.py` would have gone green having graded
  nothing.** Its comparison is `if k in vals`, and the API form's `tau` is now
  `selection.tau`, so every Sol knob would have been skipped silently. Both
  forms are normalised to one vocabulary before comparison, and the fix was
  shown red by mutating `tau` in each form.
- **`vendor/rebuild_kernel.sh` hardcoded the commit it tagged builds with**, so
  the version-tag patch went stale on the first update that used it. The patch
  carries a placeholder and the script substitutes the checkout's own short
  sha. Two latent bugs in the same script went with it: the wheel was installed
  through a glob that matches every build ever made in `dist/`, and a guard
  written as `grep -q ... && exit 1` aborted the script under `set -e` on its
  success path.

### Added

- `bench/probe_sol_topk.py`, the only thing here that executes `topk_ratio`.
  Not a check and it asserts nothing: no graph renders under that selection, so
  there is no threshold anyone has agreed to hold it to. It does carry the
  control that separates "agrees" from "the argument was ignored", and it
  records that the top-k path sits further from its own reference than the tau
  path does at the same shape.

## 0.51.0

### Changed

- **The special-token fix belongs in ComfyUI's tokenizer**, the only place that
  reaches every consumer -- core's `MiniMaxH3ReferenceToVideo` included, and no
  custom pack can add an import to that. **Upstream PR 15808 does exactly this**
  and supersedes the local branch written here the same day; it is OPEN, not
  merged. Verified against its own diff on a clean master: nine of nine audit
  scenes reproduce the release tokenizer exactly, the reference path carries the
  marker, and a marker-free reference prompt is byte-identical before and after.
  - Neither version touches the bundled `qwen25_tokenizer` directory, which the
    Qwen3VL image models share and which must not change.
- **`MiniMaxH3VendorTokens` is deprecated** and flagged `is_deprecated` in its
  schema. It is wired into no shipped graph and does nothing on an install
  carrying the core patch. `clip_with_vendor_tokens` is NOT deprecated: a pack
  cannot assume the install it runs on carries the patch, and it already
  returns the CLIP unchanged when the tokens are present.
  - `audit_h3_marker_tokenization.py` asserts the two agree rather than
    assuming it -- with the core patch in place the shim must be a no-op, and
    the run refuses if it quietly does work instead.

### Added

- **`bench/audit_h3_marker_tokenization.py`**, auditing all seven markers the
  release declares and ComfyUI's bundled tokenizer does not, across nine prompt
  shapes plus a reference-presentation integrity section. Token level only --
  no encoder forward, no GPU. `bench/grade_h3_marker_tokens.py` remains the
  encoder-level companion and owns the hidden-state deltas.
  - The load-bearing control is that the patched tokenizer reproduces the
    RELEASE tokenizer's ids exactly on every text-path scene, which is what
    makes "patched" mean "what the model authors emit". Disagreement is
    reported as a defect in the fix, not a finding about stock.
  - **The damage is not confined to the marker.** BPE does not stop at the
    angle bracket, so fragments fuse with the text on either side: the release
    emits `<d>` then `[`, stock emits `>[` as one token, and a sentence-final
    `.` fuses forward into `.<`. `contaminated_neighbours` counts the ordinary
    prose tokens that come out different, and each scene carries a decoded
    example so the count can be checked by eye.
  - **The marker survives the reference presentation**, through the exact call
    `comfy_extras/nodes_minimax_h3.py:351` makes. This closes the prior
    session's forward item 6, which existed because the sizing of a reference
    conditioning node rested on a source read rather than a run.
  - **The patch is inert where it must be**: over a full reference set (two
    images, an odd-frame video, audio) a marker-free prompt tokenizes
    identically under stock and patched, ids and vision structure alike, and a
    marker prompt leaves the `<Picture i>` / `<Video k>` / `<Audio j>` labels
    and vision sentinels untouched.

### Added

- **`MiniMaxH3ReferenceVideoFit`**, closing the half of the reference path this
  pack could not reach. `MiniMaxH3ReferenceFit` covers images; nothing touched
  reference *video*, so its resolution could not be reported, held under Qwen's
  ceiling, or distinguished from an accidental downscale.
  - **Reporting is the deliverable, not resizing.** It defaults to changing
    nothing and saying what core will do. Reference rows are attended every
    sampling step, so a smaller reference is usually right; the node makes that
    choice visible rather than overriding it.
  - Resizing is honest about its own limits: core re-derives the size from its
    own canvas rule, so only downscales below the canvas area survive and an
    upscale is capped straight back. The node warns when it was overridden
    rather than reporting a size the user did not get.
  - Deliberately carries no fps control. `force_rate=24` owns that and
    `check_ref_prompt_labels.py` gates it; a second place to set it is a second
    place to get it wrong.
- **`bench/check_conditioning_behaviour.py`**, the first control on
  `MiniMaxH3Conditioning`, which owns 19 shipped graphs and had been asserted
  in a docstring and controlled by nothing. Carried as a forward item through
  **three** postmortems.
  - **Core is the reference and it is right in both directions.** Arms are
    AGREE (ours must match core, because this is a replacement not a rewrite)
    or DIFFER (ours must not, because a documented defect is being fixed). A
    DIFFER arm that starts agreeing is the dangerous failure: the node keeps
    running, the graphs keep rendering, and the fix is silently gone.
  - Shown red twice and independently -- deleting the empty-prompt refusal
    reddens only that arm, forcing `fit_to_canvas` reddens only the last-frame
    arm.
- **`bench/check_reference_contracts.py`**, the first assertion of any kind on
  the load-bearing contracts in core's `MiniMaxH3ReferenceToVideo`. Five of the
  seven now have a control; they had stood as "enforced by nothing" through two
  postmortems. Contracts 4 and 5 remain uncovered and the check prints which on
  every run, so an uncovered contract cannot read as a covered one.
  - **Core is the control, not a fixture.** Contract 2's arm supplies
    soundtracks in reverse order to the videos, so suffix pairing and
    positional pairing give visibly different answers. Shown red by replacing
    core's suffix pairing with positional pairing.
  - Also the prerequisite for ever replacing that node: until something asserts
    these, "did the replacement reproduce them" is unanswerable.
- **`bench/check_ref_video_prediction.py`**, which holds that node's copy of
  core's sizing rule to core's real behaviour by patching `_resize` to record
  and abort. No weights, no server, and the expectation is core's own call
  rather than a number the check computed. Shown red by deleting core's
  no-upscale override from the prediction.

### Fixed

- **`MiniMaxH3ReferenceFit` now keeps the VAE and Qwen on one size.** Core
  encodes one tensor with the VAE and hands the SAME object to the conditioner;
  Qwen then applies its own ceiling inside the text encoder, after the VAE is
  done. Above that ceiling the second resize fires for Qwen alone, so the DiT
  receives one reference at two resolutions and nothing says so. The node now
  pre-applies the shrink, so the tensor core encodes is the one both towers get.
  - New `keep_towers_matched` input, appended last and defaulting on. Off
    reproduces ComfyUI's behaviour including the split, which is what you want
    if you are measuring the split rather than avoiding it.
  - The ceiling is read from `process_qwen2vl_images`'s signature default by
    introspection, never copied. That is ComfyUI's number; the release ships a
    different one and `vendor_config.image_pixel_bounds()` owns THAT. Conflating
    the two is the whole defect.
  - Verified: every previously divergent size now matches, and at exactly the
    resolution Qwen would have chosen, so nothing is lost.
  - Reaches only very wide references -- the crossing is around 3.06:1 at a
    2048 short edge and the aspect gate refuses past 4:1. **No shipped graph
    reaches it**, so the regeneration below is schema-only and no shipped
    conditioning changes.

### Added

- **[`docs/comfyui_vendor_gaps.md`](docs/comfyui_vendor_gaps.md)**, the
  consolidated report: every known divergence between this install and the
  release in one file, with practical impact per gap, a priority by what it
  costs a working user, and links to every instrument, record and fix built
  against each one.
  - **Explicitly a dated snapshot, not a fourth authority.** It names the owner
    of each fact and states that where it disagrees with an owner, the owner is
    right. That is the only honest way to have a consolidated view in a repo
    whose rule is single ownership, and the header says so rather than leaving
    the next reader to discover it.
  - Carries the "Everything built against these gaps" inventory and the "What
    is actually enforced" table. The second is the uncomfortable one: five of
    nine open gaps are enforced by nothing, and the two highest-priority ones
    are both among them.
  - Answers whether the seven special tokens reach Qwen3-VL's vision tower.
    They do not, and the assertion that closes it already existed.

### Fixed

- **The "reference media never upscaled" gap was filed as a defect and is not
  one.** Corrected same-day after a user pointed out the cost, which the
  measurements in this repo already carried and the gap report had not
  consulted. Reference rows are attended at every sampling step, so upscaling
  is a tax on the whole render: a single 1024x1024 image upscaled to 2048 costs
  6,144 tokens, half of it in the conditioner's vision blocks. The clamp is a
  speed/fidelity knob ComfyUI exposes and the vendor's pipeline does not.
  - What remains is narrower and stated as such: the cost is measured, the
    benefit is not, and a rendered pair cannot supply it because two arms
    differing in reference size are two different samples.

### Changed

- **`docs/research/sglang_comparison.md` points at the consolidated report**
  rather than carrying its own index, so the list exists once. Every known
  ComfyUI-versus-vendor gap in one table, with practical impact, a priority by
  what it costs a working user, a dated status and the doc that owns the detail.
  - Records that `coderef/sglang` has moved past the commit the 2026-08-21
    prose was read at, so older sections are labelled as resolving but not
    re-read.
- **`docs/h3_references.md` now says what sglang does about frame rate.** Both
  implementations target 24; only sglang enforces it, with an ffmpeg `fps=`
  filter in the same decode pass. fps never enters its API surface at all --
  the caller asks for a duration.

### Fixed

- **`docs/research/sglang_comparison.md` listed a closed question as open.** Its
  VAE-encode item asked for an instrument that separates encode from decode.
  That instrument landed the previous evening and the file's own last-updated
  date preceded it by minutes. The precision half is measured; what stays open
  is the mean-versus-sample tangle and whether any of it is visible.
- **`bench/grade_h3_marker_tokens.py`'s `comfy` arm changed meaning under it**
  when the correction moved into the tokenizer, and would have reported
  near-zero deltas -- reading as a retraction of its own earlier numbers rather
  than as the arm having become a second copy of `vendor`. It now reconstructs
  the pre-patch tokenizer, importing that reconstruction from the audit rather
  than copying it, and says so when it does. `CLAUDE.md`'s "when something
  gains an off state, revisit every assertion about it", applied to an
  instrument here rather than to a node.
- **`docs/research/official_weights_metadata.md` argued from a state that has
  been reversed.** Its finding 1 is retitled to the past tense, carries the fix
  and the fact that an install without the branch still has the defect, and
  gains the measured detail that the damage extends past the marker into the
  prose beside it.
- **`bench/check_workflow_schema.py` now runs in the bare sweep.** Its argument
  default was changed to `graph_paths()` last session and deliberately not run.
  It passes bare and with an explicit path, closing forward item 5.

## 0.50.0

### Added

- **`bench/audit_shipped_reference_bounds.py`**, answering whether any shipped
  graph hands Qwen a reference image it shrinks. It does not: every reference
  input is priced end to end -- real file size, the real `MiniMaxH3ReferenceFit`
  with that graph's own arguments, the real `process_qwen2vl_images` -- and
  nothing is resized (`bench/results/2026-08-21_shipped_reference_bounds.json`).
  Two synthetic arms must trip or the run is rejected, because the expected
  answer is an empty list and so is the answer a broken detector gives.
- **`bench/red/show_red_preflight_guide_split.py`**, the red harness for the
  guide split below. Five mutations red; two near-misses that are the point of
  the file rather than padding.
- **`bench/grade_vae_encoder_precision.py`**, measuring what promoting the H3
  video VAE's encoder actually changes, at the call rather than at a rendered
  clip -- encoder precision is a numerical knob and a clip pair cannot A/B one.
  fp16 is bit-identical to itself, fp32 moves the latent 6.4e-4, bf16 moves it
  17x further; the last two are controls, not results.
- **`workflows/h3_probe_ref_vae_encoder_{fp16,fp32}_api.json`**, the first
  graphs to wire `MiniMaxH3VAEPrecision` at all. They differ in exactly the
  precision node and share a seed. They price the knob and prove it runs; they
  are labelled as unable to say which output is better.
- **`bench/run_marker_stock_tail.py`**, six more stock-arm seeds staged for the
  `rolloff85_hz` question. Built, deliberately not run.
- **`internal/briefs/` and the blind batch for the marker arms**, staged with
  the key sealed.

### Changed

- **`bench/preflight_graph.py` and `bench/check_prompt_guide_conformance.py`
  now grade against the guide that applies to each graph.** The release ships
  two prompt guides with different section lists, and both instruments knew
  only the six-section one, and only `MiniMaxH3ReferenceToVideo`. Base-format
  graphs were therefore not failing -- they were invisible, answering `nothing
  to grade`. Which guide applies is read off the graph's own sockets. Preflight
  returns no FAIL and nothing ungraded across every shipped graph, up from 21
  ungraded; the conformance check grades 55 graphs, up from 36.
- **`wired_labels` counts keyframes as the `<Picture N>` labels they are**
  (`comfy/text_encoders/minimax.py:183`). Without this, adding the node name
  above would have reported every correct keyframe prompt as naming a label no
  socket wires.
- **`bench/check_workflow_schema.py` defaults its argument to
  `graph_paths()`**, so the bare `for c in bench/check_*.py` sweep reaches it.
  `nargs="+"` made a no-argument call exit on argparse, indistinguishable from
  the exit 2 this check uses for "no server, nothing validated", and it sat out
  two rounds of graph regeneration. **Written and not run; unverified until a
  bare invocation validates against a live `/object_info`.**
- **`bench/repro_mono_ref_audio.py` is now `bench/check_mono_ref_audio.py`**, a
  gate. It guards a crash that still ships and needs no GPU, model or server,
  so running only when a person typed it was the same as not existing.

### Fixed

- **`bench/audit_shipped_reference_bounds.py` read the input directory the
  server is actually launched with.** Its first run resolved zero files and
  printed a clean empty list, which is indistinguishable from the correct
  answer; it now fails when it prices nothing.

## 0.49.0

### Added

- **`MiniMaxH3Conditioning`**, replacing core's `MiniMaxH3ImageToVideo` on the
  t2v and keyframe paths across all 19 graphs that used it. ref2va is
  untouched. It closes four seams measured here on 2026-08-21:
  the seven H3 special tokens are registered by the node itself, so `<d>`
  reaches the model as id 151669 rather than BPE debris and nobody has to
  remember to wire a second node in front; the canvas is derived from the
  anchor keyframe by the node that also encodes it, so geometry has one owner
  instead of two in series; **a lone `last_frame` is now a valid graph**, which
  core cannot express and `MiniMaxH3KeyframeCanvas` cannot reach because its
  `first_frame` input is required; and an empty prompt is refused rather than
  conditioned on a pad token.
  Smoke-rendered on both signatures: a first-frame graph and a last-frame-only
  graph, each deriving 768x768 from a 1024x1024 keyframe.

### Changed

- **`clip_with_vendor_tokens()` and `resolve_keyframe_geometry()` are module
  level**, with `MiniMaxH3VendorTokens` and `MiniMaxH3KeyframeCanvas` as thin
  wrappers over them. A node cannot call another node, and the alternative was
  a second copy of each -- which this repo forbids. Both old nodes keep their
  schemas exactly, so nothing that already ships changes.
- **`resolve_keyframe_geometry` accepts a keyframe set with no first frame.**
  The anchor is chosen by semantic frame index, sglang's rule at
  `prequeue.py:97-107`, so the lone frame anchors the canvas instead of being
  cover-cropped into one chosen elsewhere.
- **`docs/open_experiments.md`'s two dead citations now resolve.** The script
  was archived, not deleted. They had been red across every `check_doc_links`
  run for days, which is the state this repo says is worse than no check.

### Deliberately not closed

- **The Qwen processor's pixel bounds.** Inert on this path -- no legal H3
  canvas trips either bound -- and reaching them means patching
  `cond_stage_model`, which `CLIP.clone()` shares by reference, so the patch
  would leak into every other graph in the process.
- **The VAE posterior.** `VAE.encode` never returns the log-variance, so a node
  would have to reach into `first_stage_model` and skip `load_models_gpu`,
  giving a CPU-resident model whenever the VAE is offloaded. `vae_precision.py`
  is the safe shape for this and the posterior belongs in a node like it.
  Both are recorded in the node's own docstring.

## 0.48.6

### Added

- **`grade_arm_audio_spectrum.py` splits the top band by whether anyone is
  talking**, because a whole-clip figure is ambiguous between bright consonants
  and hiss and those are opposite findings. Frames in the top quartile of
  energy against frames in the bottom quartile.
  **The gaps row is a control, not a second result**: arms differing by an
  overall filter would differ in their silence too. Measured on the marker
  batch, speech separates at 3.26 and the gaps at 0.01, so the difference is on
  the voice rather than the recording. Without that row the speech number could
  not be read at all.

### Changed

- **`docs/bench_plan.md` Run 6 carries what the top band is.** Registering the
  seven tokens makes the model's consonants softer -- less sibilance, not less
  intelligibility. That also reconciles the measurement with the owner's ear:
  the earlier note claiming the numbers pointed opposite to what was heard was
  reading "tin can" as narrowband when the audible feature was consonant
  harshness.

## 0.48.5

### Added

- **`bench/grade_arm_audio_spectrum.py`**, which asks whether one render arm's
  audio differs in timbre from another's or whether that is the seed. Band
  energy fractions and brightness on the decoded audio, loudness normalised
  first so a quieter clip is not read as a boxier one. **The within-arm spread
  across seeds is the control**, and the reported quantity is between-arm
  separation in units of it -- a version reporting only per-arm means would
  find a difference every time, because two finite samples of anything differ.
  Prints no p-value: at six clips an arm that would be theatre.
  It also prints every filename it ingests, after its own first run silently
  globbed two experiments' clips together and reported a result off the mixed
  set. The script cannot see which prompt made a clip; a reader can.
- **Twelve-clip marker batch**,
  `bench/results/2026-08-21_marker_arms.jsonl` and the spectral record beside
  it. Routing `<d>` to its real token id **measurably changes the audio the
  model produces**: stock carries about 2.1x the 4-16 kHz energy and the
  per-clip ranges do not touch across six seeds an arm. So the DiT does read
  the tokenizer difference, which the encoder measurement could not tell us.
  Which arm is *better* is not established and the record says so.

### Changed

- **`docs/bench_plan.md` Run 6 carries its outcome, including that the
  pre-registered score sheet was never used.** Both arms speak the dialogue, so
  the spoken/wrong/absent sheet could only tie; the spectral measure that
  replaced it was designed after hearing the clips and is labelled post-hoc
  rather than quietly presented as the planned test. Prediction 1's conclusion
  is recorded as refuted: same spoken count, and the marker still reaches the
  DiT.

## 0.48.4

### Added

- **`bench/grade_h3_marker_tokens.py`**, which answers whether routing `<d>` to
  its real id changes what the encoder produces. Not a render: `CLAUDE.md`'s
  different-sample rule means two arms differing in conditioning give two
  different samples, so this compares at the encoder output where the arms are
  comparable by construction. Three arms per prompt on one set of weights --
  markers routed, markers as the BPE fragments ComfyUI emits, markers deleted --
  aligned by `difflib` over the id sequences rather than by position, because
  the arms have different lengths.
  **The deleted arm is the scale and the finding depends on it.** Read against
  it, ComfyUI's fragments recover about a tenth of what the marker does on the
  two strongest prompts. So "the fragments carry the delimiter anyway, routing
  them is cosmetic" is refuted. Two controls held: a marker-free prompt gives
  identical ids and identical states on both sides, and the forward is
  bit-deterministic.
  This measures the encoder, not the DiT. It establishes the representation
  changes; whether the DiT reads the change is still open.

### Changed

- **`docs/h3_references.md` no longer presents the six-section prompt format as
  the only one.** The guide ships two, and this page had described one.
  `ref-en.txt` is the six-section reference format; `base-en.txt` is three
  fields for t2va/i2va/fl2va/l2va, with `integrated_multimodal_description` in
  place of `detailed_description` and no subject or retention sections.
  `<Subject N>` labels are reference-format only.
- **Both prompt instruments encode the reference format as the only format**,
  which is correct today and is a trap for the extension this repo has been
  considering. `preflight_graph.py` never sees a t2v graph, so its `SECTIONS`
  constant is never wrong in practice; graded by hand, a correct t2v prompt
  comes back with four sections reported absent that its guide does not have.
  Recorded beside the format so the next reader meets it before the change.

## 0.48.3

### Added

- **`bench/measure_qwen_bounds_bite.py`**, which turns the pixel-bounds
  divergence from a pair of constants into a location. It calls the real
  `process_qwen2vl_images` with the arguments the H3 path passes and reports
  the grid returned, on two arms: reference images at a 2048 short edge, and
  keyframes at a legal canvas. **The ceiling starts biting between 3:1 and
  3.25:1** and the release carries the same image untouched to 4:1. The
  keyframe arm is the control and comes back untouched at every canvas, which
  is what licenses the inert claim below; if it ever does not, the script says
  so and names the doc it refutes.

### Changed

- **`docs/research/official_weights_metadata.md` gains a per-mode table** and
  loses part of its "Not done". The question that decides what to do about any
  of these divergences is which of them fire in the mode being run, and several
  do not. t2v never runs the vision tower, so only the marker gap reaches it.
  The pixel bounds and the interpolation kernel are **inert on every keyframe
  mode**, measured, because a legal canvas is a multiple of the helper's
  rounding factor and sits inside both implementations' bounds. Last-frame-only
  is the most divergent keyframe mode and the only one `KeyframeCanvas` cannot
  reach.
- **The same file stops implying ComfyUI points a Qwen2.5 tokenizer at a
  Qwen3-VL model.** Measured against a stock Qwen3-VL checkout, the bundled
  directory is byte-equivalent to Qwen3-VL's own tokenizer: same class, same
  vocabulary, same merges, same 26 `added_tokens_decoder`. Qwen2.5, Qwen3 and
  Qwen3-VL share one BPE vocabulary and the release itself declares
  `Qwen2Tokenizer`. The directory name is legacy and the whole divergence is
  seven `additional_special_tokens`. The file also now records *why* ComfyUI
  cannot load them, which is structural: a single-file text encoder travels
  with no tokenizer config, so nothing can carry a model's own additions.
- **`docs/h3_references.md`'s ceiling paragraph is measured rather than
  estimated**, and states the consequence it had been missing: past 3.0625:1
  the VAE receives the full 2048-short-edge tensor while Qwen receives a
  smaller one, so the one-image-two-towers row in the table above it stops
  holding and nothing warns.

## 0.48.2

### Added

- **`bench/audit_h3_token_embeddings.py`**, which answers the question
  `vendor_tokens.py` had left open in its own docstring: whether the seven
  marker rows carry trained values. **They do not.** Read against two controls
  -- the stock Qwen specials, trained, and the padding rows past the added
  tokens, not -- the seven land on the untrained pole in the official release
  and in both repacked encoders alike, at roughly a hundredth of the distance
  to the trained one. The controls separate cleanly, so the norm discriminates
  and a null result here is a result.
  Independently measured first in an outside review; reproduced here before it
  was believed.

### Changed

- **Making the seven markers reachable is a parity fix, not a quality fix**,
  and `vendor_tokens.py` and `docs/research/official_weights_metadata.md` now
  say so. Neither path carries learned meaning for `<d>`: the release emits one
  ID and attends init noise, ComfyUI emits several trained BPE pieces. They are
  still different sequences of different lengths, which shifts every position
  after the marker, so the node is worth wiring -- just not for fidelity. Two
  readings of the untrained rows stay open and the measurement does not
  separate them: vestigial tokens, or a text encoder frozen through H3
  training, under which no row moved and the norms say nothing about intent.
- `vendor_tokens.py` no longer states that no prompt in this repo uses one of
  the markers. `workflows/h3_ref_audio_voice.json` does, which 0.48.1 corrected
  in the doc that first made the claim.

## 0.48.1

### Added

- **`bench/compare_h3_tokenizers.py` and `bench/repro_mono_ref_audio.py`**,
  the measurements behind the two corrections below. Neither is a check --
  both report and neither gates -- but the claims they support are now in
  shipped docs, and a claim whose evidence lives in a scratch directory is a
  claim nobody can re-run. Each carries its own control: the tokenizer script
  asserts that the labels the conditioner emits on every render are
  ID-identical, so the finding stays confined to the seven markers, and the
  mono repro runs a stereo arm so the failure is attributable to the channel
  count rather than to the harness.

### Fixed

- **`docs/research/official_weights_metadata.md` said no prompt in this repo
  used the seven unreachable H3 tokens. `workflows/h3_ref_audio_voice.json`
  and its API twin both carry `<d>`**, emitted by
  `workflows/build_workflows.py:1718`, which quotes the guide rule requiring it
  in the comment directly above. The claim was written from expectation rather
  than a grep. Corrected with the measured IDs on both sides, and with the
  fact the section also got wrong in the other direction: `<d>` is not
  undocumented, it is what the official prompt guide mandates for all dialogue
  and lyrics, so every H3 prompt containing speech is affected on ComfyUI.
- **`docs/h3_references.md` carried mono reference audio as read-but-not-
  verified. It does not degrade, it raises.** ComfyUI's audio VAE preserves the
  input channel count and nothing upmixes, so `pack_audio` returns half the
  rows `PackedLayout` allocated and the masked assignment fails. Run on CPU
  against the real `pack_audio` by `bench/repro_mono_ref_audio.py`, whose
  stereo arm is the control that makes the failure attributable to the channel
  count. Moved to Known limitations;
  `docs/research/sglang_comparison.md`'s open-items bullet updated to match.
- **`docs/h3_references.md` claimed everything downstream of the reference
  resize matches between ComfyUI and sglang. The VAE encode does not.** sglang
  samples the released posterior under a seed pinned at 42 for keyframes and
  reference video; ComfyUI takes the mean and has no sampling path. That is a
  second, independent latent-boundary difference on top of the condition-noise
  realisation the doc already recorded. Added as a row in the vendor image path
  table, and tangled into the open VAE-encode precision question in
  `sglang_comparison.md`, which now has two variables rather than one.

## 0.48.0

### Changed

- **The video VAE is `minimax_h3_video_vae_fp16`, by owner decision
  (2026-08-21); the int8_convrot decoder was deleted from disk.** Measured
  before the switch: the fp16 file is the official release's fp32 weights
  (`MiniMaxAI/MiniMax-H3`, `video_vae/source/model.safetensors`) cast down --
  all 559 shared tensors match in name and shape, median relative delta
  2.07e-4, max 2.48e-4, which is fp16's rounding floor and nothing more. Only
  the fp32 original is more faithful, at twice the size; a bf16 conversion
  would be less faithful, since fp16 carries three more mantissa bits. The
  int8 decoder's own measurements (1.29x decode, 53.3 dB against fp16) stay in
  0.13.0 as history.

### Added

- **`bench/check_model_files.py`**, for an escape the existing gates could not
  catch: the VAE was deleted while `h3_config.py` still named it, and
  `build_workflows.py` reported 113 graphs validated against `/object_info`
  ok minutes later -- it grades classes, input names and link shapes, never a
  widget's value against the combo options that input offers. The new check
  grades every model name in a shipped graph and in `h3_config` against what
  the live server offers **for that node class**, so a VAE name under
  `UNETLoader` fails though both files exist. Four controls run every
  invocation, including a bench-only defect replayed to confirm the widened
  coverage fires. `docs/checks.md` gains its row and retires the matching
  uncontrolled-requirement line.
- **`h3_config.graph_paths()` takes `include_bench`**, so `workflows/bench/`
  is reachable by the checks whose property is true of any graph while staying
  out of schema grading, which is what `GRAPH_DIRS` actually exempts it from.
  The flag went on the shared discovery function rather than a glob in the
  check, because `check_graph_discovery.py` forbids the second and widening
  coverage should widen the one function every walker goes through.

## 0.47.0

### Removed

- **The extracted ref LoRA, by owner decision (2026-08-21), from every
  operative surface.** Gone: `bench/analyze_ref_lora.py`,
  `bench/bench_0b_lora_ab.py`, the `h3_image_ref_plus_text_to_video_ref_lora`
  graph pair, the `REF_LORA` / `REF_LORA_STRENGTH` / `REF_LORA_ENABLED`
  constants and `ref_base_and_lora()` in `workflows/h3_config.py`, the
  generator's entry, note and switch call, and every current-state mention in
  `docs/`. Reference graphs load the ref2va checkpoint directly and there is
  no longer a second path -- which was already true at runtime since
  2026-08-18; what goes away is the alternative that made it a question.
  All 113 graphs regenerate byte-identical apart from the two deletions.
- **`docs/roadmap.md`'s ref-LoRA grading claim goes with it**, including the
  three-way cross-check ("they agree at cos 0.040 ... 9x better"), which was
  an artifact: the int8 side was compared in its Hadamard basis because
  `analyze_ref_lora.py`'s `dequant()` never un-rotated. Un-rotated on three
  sampled modules that cosine is 0.52-0.65. Nothing now asserts it, so it is
  recorded here rather than retracted in place.
- Two docs that cited the rotation finding through the roadmap
  (`docs/drift_frontier.md` F16, `docs/open_experiments.md`) now cite
  `bench/analyze_quant_delta.py` for what the rotation is.

History is kept: CHANGELOG entries, dated records under `bench/results/`,
postmortems and session logs still name it, and so does the provenance of
captures taken on the fl2va+LoRA path.

## 0.46.0

### Added

- **`bench/analyze_quant_delta.py` and its records
  `bench/results/2026-08-21_quant_delta_{fl2va,ref2va}.json`**: the two shipped
  quantizations of the same DiT graded against the bf16 release. `int8_convrot`
  lands about three times closer to the original weights than `fp8_scaled`
  (median relative delta 0.0091 against 0.0265 per block linear), identically
  on both checkpoints. Records the formats as read from the files -- fp8's
  scalar `weight_scale` and its `input_scale` on 150 of 200 tensors, int8's
  per-row scale and Hadamard rotation at group 256 -- and states inside the
  record that stored-weight fidelity is all it measures.
- **The qkv row order differs between the release and the repack**, measured:
  interleaved per head upstream, concatenated in the repack. A probe runs
  before any reference comparison and refuses to measure if the reordering is
  not the better reading. `docs/evidence.md` carries both entries.

- **`docs/research/sglang_comparison.md`**: sglang's H3 serving path against
  ours, read at commit `a41da991c8`. What they do that we do not (an exact
  per-schedule AdaLN cache, breakable CUDA graphs over the packed sequence,
  admission-time refusals), what we both do where ours is weaker (video VAE
  encode precision, silent VAE tiling), and what looks like a gap but is
  already priced (step caching, killed here by owner decision 2026-08-20).
  Records one hypothesis it killed: the fp8-vs-int8 fidelity gap is not a qkv
  row-permutation defect, because fp8's scale is a scalar and the gap is
  uniform across module kinds.

- **`docs/research/conditioning_nodes.md`**: the conditioning work and its
  scope reduction. The first sketch was two replacement conditioning nodes; the
  node-by-node trace refuted that shape, because `MiniMaxH3ReferenceToVideo`
  carries five contracts that exist only in code comments and fail silently, and
  because neither defect needs that node replaced. Records the three tokenizer
  seams that look available and are not, why the pixel floor is node-reachable
  and the ceiling is not, and the five contracts as acceptance criteria for
  anyone who does replace it.
- **`bench/preflight_graph.py` warns when a reference will cross a pixel bound**
  the conditioner and the release disagree on, naming which side does what.
  ComfyUI's values are read from its source rather than restated; an unreadable
  source says the comparison was not made rather than printing nothing.

- **`vendor_config/` and `vendor_config.py`**: the release's own configuration,
  vendored verbatim and read rather than retyped -- the twenty
  `additional_special_tokens`, the image and video pixel bounds, the patch
  geometry, and the two partition task lists. 13 KB against 200 GB of weights,
  so a graph that needs the image floor does not depend on the weights being
  downloaded. `bench/check_vendor_config.py` hashes them against what they were
  copied as and, when the release is on disk, against the release itself; a
  skipped release comparison announces itself rather than reading as a pass.
- **`MiniMaxH3VendorTokens`** (`vendor_tokens.py`): adds the seven special
  tokens the release declares and ComfyUI's bundled tokenizer lacks, so `<d>`
  and the cutoff, lyrics and caption markers tokenize as markers instead of as
  literal text. Builds a fresh tokenizer rather than mutating the loaded one,
  because `clip.clone()` shares it by reference and an in-place edit would
  contaminate every graph in the process. What the markers are FOR is
  undocumented upstream and unmeasured here; the node makes them reachable, not
  useful.

- **`docs/open_experiments.md` #22 is measured and refuted**
  (`bench/run_pruning_arms.py`, `bench/grade_pruning_sensitivity.py`,
  `bench/results/2026-08-21_pruning_sensitivity.json`). Eleven fixed-input
  first-step forwards. ref2va is no more sensitive to the AdaLN pruning than
  fl2va -- it moves slightly less, in the opposite direction to the hypothesis.
  The pre-registered prediction was wrong in the other direction and is
  recorded as wrong: the velocity moves 5.6-9.4%, not under 1%, so the pruning
  is not invisible at the output, only equally so on both checkpoints and
  smaller than the fp8-vs-int8 difference already shipped. The determinism
  floor is exactly zero.

### Fixed

- **The capture module stopped capturing after a checkpoint swap.** Block
  indices came from first-seen call order keyed on `id(module)`, so a second
  loaded checkpoint numbered its blocks 50..99, no requested index matched, and
  the render counter jammed -- silently, with empty directories as the only
  symptom. The patching loop now stamps the index. Found on the #22 arms, which
  swap checkpoints by design.
- **The provenance stamp read any attention override as Sol's**, so every
  sage-only graph -- both capture graphs and every dense baseline -- was one
  stamp away from being declared unattributable for correctly having no Sol.
  The override now says which kernel installed it and what it wrapped.
- **`bench/run_graph_arms.py` could not patch a dynamic-combo widget**, because
  the target split took the last dot rather than the first one that names a
  node. It could therefore not express a canvas change.

- **`MiniMaxH3VAEPrecision`** (`vae_precision.py`): sets the H3 video VAE's
  encoder and decoder precision independently, which ComfyUI's single
  `--fp32-vae` flag cannot express. The released pipeline keeps this VAE fp32
  and decodes under fp16; the flag moves both halves together. The prices are
  not symmetric -- read out of the shipped checkpoint, the decoder is 93% of
  the weights and the encoder 7% (4.51 GiB against 0.34 GiB), so fp32 encode is
  cheap and fp32 decode is the part that got the flag reverted. Both module
  boundaries are wrapped to cast to whatever that half holds, so a graph that
  also wires the unmodified VAE elsewhere stays correct. Whether fp32 encode
  changes anything is unmeasured and the node says so.

- **`docs/research/official_weights_metadata.md`**: the published release read
  beside the code that consumes it. Seven H3-specific special tokens
  (`<d>`, `</d>`, the cutoff, lyrics and caption markers) are declared by the
  release's tokenizer config and absent from the one ComfyUI bundles, so a
  prompt using them is tokenized as ordinary text; the embedding rows for them
  ship in the encoders we already run. The partition split is in the weights,
  not in sglang. Records what was checked and found clean: the two transformer
  configs are identical, the layer-50 tap matches, patch geometry and scheduler
  shifts match.

### Changed

- **The vendor-side authority for reference conditioning is sglang, not
  diffusers.** `docs/h3_references.md` gains a stage-by-stage table of the
  vendor image path and now says which claims are diffusers-only. The two
  implementations agree everywhere downstream of the resize; the divergence is
  sizing, and the default `match` mode is further from the vendor than the
  known `min(1.0, ...)` clamp.
- **Three attributions corrected after re-derivation against sglang, diffusers
  and DiffSynth-Studio.** The 12-reference and audio-pairing limits are
  diffusers-only, originating in MiniMax's Open Platform README table; audio
  truncation is two of three implementations, not a consensus; reference-video
  upscaling is all three, so that claim strengthens.
- **`docs/h3_resolutions.md` no longer calls the 768 short edge a trained
  distribution enforced by the reference pipeline.** Re-derived: the area cap
  is enforced upstream with no bypass, the short edge only warns, nothing
  refuses a larger one, and no config shipped with the weights carries a canvas
  geometry.

### Fixed

- **`bench/preflight_graph.py` over-priced a reference image with no
  `MiniMaxH3ReferenceFit` in front of it**, treating it as upscaled to a 2048
  short edge where core clamps with `min(1.0, ...)` in both modes and never
  enlarges. The over-count is the square of a scale the reference never gets.
  No shipped API graph reaches the branch; a hand-built one does, which is
  exactly what the tool promises to price. Confirmed non-inert against a graph
  rewired to bypass the fit node.

## 0.45.0

### Added

- **`bench/analyze_adaln_pruning.py` and its record
  `bench/results/2026-08-20_adaln_pruning_residual.json`**: what the Comfy-Org
  "pruned" checkpoints remove and what it costs, measured against the unpruned
  `int8_convrot` files. The pruning is the rank-8 SVD of the mean-centred
  time-embedding curve folded into each block's AdaLN, every other tensor
  byte-identical; it costs ~0.02% of the modulation on the bf16 final layer
  and 0.1-0.2% per int8 block, identically for fl2va and ref2va. The
  ref2va-specific pruning-loss hypothesis is refuted. `docs/evidence.md`
  carries the summary; the script's own self-test refuses to measure if its
  two deliberate violations are not caught.
- **`docs/open_experiments.md` #22**, the sensitivity half of the pruning
  question: a fixed-input first-step forward, pruned against unpruned on both
  checkpoints, with the determinism floor and the fp8 builds as controls and
  the decision rule pre-registered. Designed, not run.

## 0.44.0

### Changed

- **Sol `tau` defaults to 1.0, by owner decision (2026-08-20).** Every video
  graph regenerates at 1.0. The reversal condition sits beside the old note in
  `workflows/h3_config.py`: 1.3 returns only if an 8-seed blind session finds
  the two indistinguishable on the distilled LoRAs while the
  `--set SolAttnMiniMax.tau=1.3` patch arm buys meaningful speed. Every Sol
  number recorded before this date was measured at 1.3.
- **The t2v prompt takes the owner's v6 conditioning layout**: field name alone
  on its line, content below, `N/A` for the empty field. Content unchanged.
- **`h3_probe_turbo_768p_owner.json`** ships the owner's working recipe for the
  768p students (euler, `beta`, 4 steps, strength 0.75) as a graph with a sha,
  so bench arms can patch LoRA files onto it. Its note prices two of the three
  knobs it moves: strength below 1.0 under-distills a 4-step schedule, and
  `beta` puts 2 of 4 steps inside Sol's window where `simple` puts 3
  (arithmetic from the shift-6 sigma grid and the 0.2/0.9 window). `cross_check`
  now pins `BasicScheduler` scheduler and steps across the two graph forms.
- **The docs catch up to the distilled regime.** `docs/roadmap.md` gains "The
  regime question" (what survives, adjusts and dies under 4-6 step
  distillation; step caching is a 16-step lever and `CACHE_NODE` is closed as
  not canonical by owner decision), corrects the dials table, closes the
  not-established rows done since the 17th, states the power question once,
  and adds the node-cache-at-a-held-seed hazard. `docs/open_experiments.md`
  reframes #6 and #9, records the tau decision on #15, replaces #17b's
  refuted next action with the QK-vs-PV split (and why
  `simulate_track_b_lite.py` cannot do it), marks #18 substantially done, and
  adds #20 (the SLA LoRA under its training router) and #21 (the power pair).
  `docs/hardware.md` replaces the continuous-streaming claim with the phase-0
  measurement.
- **`bench/build_hybrid.py`** builds an fl2va/ref2va hybrid by copying tensors
  with the adaln cut as an argument, and its control ran first: `--blocks
  30-49 --verify-against` the HF b30-49 file matches all 932 tensors
  byte-for-byte, and two wrong recipes are refused naming the differing
  tensors. The build that followed -- ref2va's adaln in all 50 blocks plus
  the final layer, which no HF variant offers -- is `unet_hybrid_adaln_all` in
  `MODELS` beside the HF `unet_hybrid_b30`. `substrate.py` tags the `-int8`
  filenames.
- **Four reference-transfer graphs** `h3_probe_ref_turbo768p_{fl2va,hybrid_b30,
  hybrid_adaln_all,ref2va}`: the capture graph's three-image request with the
  4-step 768p turbo LoRA at the vendor row, differing in the checkpoint only.
  The note states the prediction before the render (the LoRA was fitted
  against fl2va's linears; the hybrids keep them; ref2va does not).
- **`bench/blind_batch.py`** turns a `run_graph_arms` JSONL into neutral clips
  with a sealed key in `internal/blind_keys/`, refusing cache-hit, error and
  missing-clip rows. Its first self-test caught its own off-by-one (the
  warmup's clip took a counter slot); fixed and re-verified against the
  share's mtimes. `run_once` can return the server's `prompt_id` and
  `run_graph_arms` records it per row.
- **The SLA router, vendored, gated, and wired.** `vendor/sla_sparse_triton.py`
  is LightX2V's sparse top-k router and forward kernel (commit afcfe8f1), the
  attention the Turbo-SLA LoRA was distilled under in the sparse-only form
  LightX2V ships (no linear branch; none in the LoRA file).
  `bench/grade_sla_router_on_capture.py` gates it on the clean-fl2va capture
  against the float64 row reference (`bench/results/2026-08-20_sla_router_gate.json`):
  at sparsity 0 the kernel lands at rel-L2 0.0005-0.0017, inside the sage band
  by ~10x; a zeroed lut and a row-permuted lut both go red on every cell. The
  same run puts the router at 0.85 sparsity comparable to the eager Sol
  reference at tau 1.0-1.3 on ordinary heads and better on the loudest, while
  keeping 15% of key blocks -- on base-model activations, which the student
  was trained to change. `MiniMaxH3SLARouter` is the node;
  `h3_probe_turbo_768p_sla_router` and `_sla_dense` complete the three-regime
  set with the existing Sol probe, exempted from the Sol-on rule by mechanism.

- **A figure library and a postmortem renderer.** `bench/gen_figures.py`
  generalises `gen_morton_figures.py`'s inline-SVG drawing (derived captions,
  theme-following colours) into signed bars, lines, scatter with identity,
  slope, grouped log bars and a swimlane, plus the five figures of the
  2026-08-20 session postmortem, each naming the record and field it draws
  from. `bench/render_postmortem_html.py` turns a postmortem's markdown into
  the plugin's self-contained page with those figures spliced beside the
  findings they illustrate; `gen_morton_figures.py` is a shim whose output is
  byte-identical to before. Built by a subagent; the checker was shown red on
  a removed citation, a renamed section, an injected script and an external
  link before the real page passed.

### Records

- `bench/results/2026-08-20_power_limit_pair.jsonl` and `_verdict.json` -- the
  first 330-vs-450 W pair ever run here, on the 4-step 768p turbo graph with
  disjoint seed bases: 134.1 s against 126.8 s sampler, a 5.8% cost against a
  12.5% core-clock delta with power pegged at both limits. Partly
  core-clock-bound on this workload; neither of the 2026-08-17 readings holds
  cleanly.
- `bench/results/2026-08-20_sla_regime_arms.jsonl` -- the three attention
  regimes for the SLA and v1.1 LoRAs at one seed: router 155 s, Sol 126 s,
  sage-only 201 s of sampler, and Sol at tau 1.3 118 s. The first router row
  failed on output contiguity and stays in the file.
- `bench/results/2026-08-20_session1_lora_file.jsonl` and
  `2026-08-20_session1_verdict.json` -- the first multi-seed blind session
  through the new path: four arms x 8 seeds, 24 stacked contests judged on
  free text. v1.1 indistinguishable from SLA and from the vendor recipe; v1.0
  leans ahead of v1.1 4 to 2 on look, behind on motion in two pairs.
- `bench/results/2026-08-20_sol_error_per_head_{v11,sla}.json`,
  `2026-08-20_routing_density_{v11,sla}.json` -- Sol's error and routed
  density on the two student captures agree within a few percent and half a
  point; the SLA distillation changes nothing Sol sees at the call.
- `bench/results/2026-08-20_ref_transfer_single.jsonl` -- v1.1 carries the
  three reference images on fl2va, HF b30-49, the local all-adaln hybrid and
  ref2va alike at one seed; the adaln-swapped arms share a composition the
  fl2va-adaln arms do not.
- `bench/results/2026-08-20_head_magnitudes_v11.json` and `_sla.json` -- the
  per-head input structure on the two distilled-trajectory captures
  (`2026-08-20_t2v_362f_1344x768_{v11,sla}`, fl2va base, 4 steps, Sol absent):
  block 49's four loud K channels and its 2.7-to-32.6 per-head K rms spread
  are unchanged under either student, as the untouched `k_norm` gains predict.

## 0.43.0

### Changed

- **The block-49 control is run, and the SLA probe renders.** With the card
  free: `h3_probe_capture_ref3_fl2va.json` (the capture graph with the unet
  swapped to fl2va and no LoRA, Sol off, exempted in
  `bench/check_attention_defaults.py`) produced the capture
  `2026-08-20_ref3_362f_1024x768_fl2va`, manifested. On it,
  `bench/analyze_sol_error.py` at blocks 49/40/24, step 3, all heads
  (`bench/results/2026-08-20_sol_error_per_head_fl2va.json`) finds block 49's
  INT8 error at 0.124 against ref2va's 0.134 and the per-head error ranking
  the heads the same way as ref2va's at Spearman ~0.95 with the same six
  worst heads; `bench/analyze_head_magnitudes.py`
  (`bench/results/2026-08-20_head_magnitudes_fl2va.json`) finds the same four
  K channels at the same shares and the same per-head K spread. The claim
  below, "present in both checkpoints", is now observed on clean fl2va
  rather than inferred from matching gains. One lead, not a finding:
  fl2va's sparsity error runs higher than ref2va's at every captured block
  on this reference-heavy input, one capture each.

  `bench/results/2026-08-20_sla_arms.jsonl`: the 768p v1.0 graph and its SLA
  probe, one render each, 133.6 s and 133.5 s sampler, both on-brief at 2, 7
  and 12 s. The SLA LoRA runs under Sol-Attn; nothing finer is claimed.

- **"AdaLN is replaced in every block" is withdrawn; the two checkpoints'
  modulation differs by a few percent, like everything else.** The
  2026-08-18 checkpoint-internals record compared `adaln_proj.linear.weight`
  between fl2va and ref2va directly and reported rel-delta ~1.9 with negative
  cosine at every depth, a "flat" delta-energy profile, and the HF hybrids as
  a linear dial on that energy. These are curve-form checkpoints: the block
  consumes `adaln_t_table[t] @ W.T + b`, the two files were factorised
  separately, and their bases are sign-flipped on columns 4-7 (per-column
  cosine +1, +1, +0.996, +0.996, -0.9997, -0.9997, -0.99, -0.99). The
  large-norm coefficient columns sit on the flipped basis columns, so the
  coefficient comparison was measuring the factorisation, and the energy
  profile was the squared norm of the sign flip -- and was not flat even on
  its own numbers.

  `bench/analyze_checkpoint_delta.py` now compares the modulation output over
  the whole time grid, in float64, and reports the stored coefficients only
  under `basis_dependent`. `bench/results/2026-08-20_dit_internals.json`:
  per block, the whole modulation differs 1.4-4.7% (tracking the bias, which
  is 90-95% of its norm), the time-varying part 5-9% with cosine
  0.996-0.999, the final layer's time-varying part ~12% -- the same order as
  the ~3.2% int8 linears. Every linear, norm, scale and global field of the
  2026-08-18 record reproduces exactly, which is the port control; that
  record carries a `retraction_2026-08-20` key naming the withdrawn fields.
  Block 49's text-row modulation (`gate_msa`, `shift_mlp`, `scale_mlp`,
  `gate_mlp` for tag 1) is exactly zero in both checkpoints, recorded as the
  one structurally special thing about that block at the weight level.

  Consumers corrected in place: `docs/roadmap.md` (the 2026-08-16 hybrid
  section already had the right 1.1-4.7% figure under the wrong mechanism,
  and its "adaln is the only thing meaningfully different" reading goes with
  it), `docs/h3_ref2v_distillation.md`, two comments in
  `workflows/h3_config.py`, `bench/analyze_ref_lora.py`, and two
  generated-graph notes. `docs/evidence.md` carries the row and two new
  `retraction-consumers` phrases.

- **Block 49's INT8 anomaly is attributed to its inputs, and it is in both
  checkpoints.** `bench/analyze_head_magnitudes.py`, new, reads the captured
  q/k/v on CPU and joins per-head magnitude statistics to the 2026-08-19
  per-head error record, refusing a record from a different capture or a
  head-prefix record. `bench/results/2026-08-20_head_magnitudes.json`, on
  the 2026-08-17 (fl2va + ref LoRA) and 2026-08-18 (ref2va) captures at
  step 3: block 49's per-head K rms spans ~2.2 to ~32 where blocks 24 and
  40 are flat; its K energy sits ~93% in four channels (82, 34, 67, 19)
  where block 40's loudest channel carries ~1.4%; the heads with the
  largest Q and K rms are the heads with the largest INT8 error (Spearman
  ~0.4-0.5, ~0 against V). The channels are where `attn.k_norm.weight`
  peaks at ~37 and ~31 against an rms of ~5 (mid-depth blocks peak ~1.9
  against ~1.7), and the gains match between fl2va and ref2va to ~0.3%.
  Both captures show the same structure. So the anomaly is a property of
  the released weights and not a fl2va-vs-ref2va differentiator; the step
  from "a few loud channels" to "INT8 per-block error" is an inference, and
  no capture on clean fl2va exists. Closes forward item 2 of the 2026-08-19
  postmortem at the input level rather than the output level it asked for.

- **The lightx2v turbo LoRAs live in per-repo folders, and the SLA release
  has a probe.** `TURBO_LORA` and `TURBO_768P_LORA` carry the
  `h3/lightx2v_Minimax-h3-Turbo/` prefix and the 18 graphs that load them are
  regenerated. `TURBO_SLA_LORA` is new: header-identical to the 768p v1.0
  (624 tensors, attn+mlp, rank 128, alpha 128, fl2va base), distilled under a
  top-k block router at an 85% sparsity ratio per its model card and
  LightX2V's config. `h3_probe_turbo_768p_sla.json` is the 768p v1.0 graph
  with only the LoRA swapped, Sol on; its note names the three attention
  regimes none of which is measured. `bench/check_distill_settings.py` gains
  the SLA row at 6/3 and 4 steps, graded against the LightX2V config that
  loads it because the Turbo README has no SLA row, after verifying that
  config's `infer_steps = N + 1` convention on the rows both sources share.
  The v1.1 768p upload stays unclassified: no vendor row attests its shift,
  and the self-test asserts it is refused.

## 0.42.0

### Changed

- **Chunking the calibration oracle is refuted, not done.** It was the
  highest-value remaining item on the error-decomposition line in
  `docs/roadmap.md` and the one open limit named in
  `bench/analyze_sol_error.py`'s own docstrings. `bench/probe_oracle_gate_scaling.py`
  and `bench/results/2026-08-19_oracle_gate_scaling.json` priced it and found
  two things.

  The gate was never limited to t <= 2001. The oracle refuses on a score-matrix
  budget rather than a length and runs to about 384 blocks untouched, so the
  ceiling everyone read as a limit was a chosen constant.

  Run there, the gate goes red for something that is not a defect: agreement
  holds to ~3e-04 out to 192 blocks and jumps to ~1e-02 at 256, and the jump is
  a handful of whole query blocks whose routing decision lands on opposite sides
  of the threshold in two float32 reduction orders. Reseed the input and
  different blocks flip. Chunked to production's 1539 blocks, flips are certain
  and the gate would be red while both implementations are correct, with the
  only relief being the tolerance its own refusal text forbids raising.

  What would close the gap is a different instrument rather than a longer one:
  compare the routing masks, which at production S is a 1539x1539 boolean per
  head needing no chunking, and which can report a flipped block's margin where
  output relative L2 cannot. Recorded at both docstrings and the roadmap entry
  that argued for chunking.

## 0.41.0

### Added

- **`bench/price_head_arms.py`, and the answer it returns: the per-head axis is
  empty.** Density and error per head established that a head's spend does not
  predict its damage. This prices what that is worth by putting every per-head
  arm and the global tau on one currency -- routed density spent per fraction of
  error removed. Records: `bench/results/2026-08-19_sol_error_per_head_tau1.0.json`
  (the second operating point) and
  `bench/results/2026-08-19_head_granularity_arms.json`.

  Nothing beats moving tau. A per-head dense escape list ties the global tau at
  three heads and loses at five and eight, because per-head error is only about
  2.6x concentrated -- the worst 5 of 56 heads carry 23.2% of summed error
  against 8.9% for uniform.

  A per-head **tau** is deliberately not among the arms, and that is a result
  rather than an omission. Pricing one needs each head's error as a function of
  tau; every attempt to equalise error from two operating points put 45 to 52 of
  56 heads outside the measured interval, because a tau move changes one head's
  error by a median factor of 1.27 where heads differ from each other by a
  median factor of 19. The estimate would have been extrapolation reported as
  measurement.

  Its four refusals -- capture mismatch, a measured head prefix, two error
  records at one tau, a control record with no dense-limit arm -- each go red on
  a deliberate violation.

## 0.40.0

### Added

- **`bench/join_density_error.py` and the per-head error record it joins.**
  `bench/analyze_sol_error.py` now takes `--json`, which its own module
  docstring had asked for and which the 2026-08-18 review found being done by
  hand out of terminal scrollback. Run at every head over the post-ref2va
  capture, it produces `bench/results/2026-08-19_sol_error_per_head.json`; the
  join against the density record produces
  `bench/results/2026-08-19_density_error_join.json`.

  The join asks the question that decides whether a per-head tau is worth
  anything: does a head's routed density predict its error? If every head sits
  on one shared cost/damage curve, a single tau is already the efficient
  operating point. Rank correlation between per-head density and per-head
  sparsity error runs -0.38 to +0.42 with a median of -0.23 -- weak coupling,
  not a curve. Its three refusals (head-prefix rather than all heads, capture
  mismatch, tau mismatch) each go red on a deliberate violation.

- **`bench/results/2026-08-19_sol_error_control.json`** -- the `--control` arm,
  run for the first time since it was written. At the dense limit the apparatus
  reports sparsity error 7.7e-05, so it has no floor of its own on that side and
  a head reporting large sparsity error is that head rather than the
  instrument. The same arm gives the INT8 quantization floor with no sparsity
  at all.

### Changed

- **`bench/analyze_sol_error.py` gained `--json`.** Records carry the capture by
  name; the store is outside the repo and its path is per-box.

## 0.39.0

### Added

- **`bench/sweep_routing_density.py` -- routed density with the head axis kept.**
  `bench/analyze_routing.py` averages over its sampled heads by construction, so
  the axis yesterday's density record named as "the remaining structure" was the
  one axis it could not report. This sweeps the same grid per head and emits a
  record rather than a table to re-type. The routing arithmetic is imported from
  that module, not restated, and the aggregate it prints reproduces that tool's
  raster row exactly on the same capture, head set and tau -- which is the check
  that the reuse is real.

  It answers the question a spread alone cannot. A per-head density spread inside
  one cell is the router already responding to per-head content; what would make
  a per-head tau exploitable is whether the spread **persists**. So the record
  carries rank correlations of the per-head ordering, split into same-transformer-
  block-across-steps and across-blocks-at-one-step, because those two answer
  differently.

- **`bench/results/2026-08-19_routing_density_per_head.json`** -- every head, all
  seven captured blocks, both captured steps, tau 1.0 and 1.3, on the first
  capture taken after the ref2va-direct switch. Supersedes nothing: the
  2026-08-18 record measured a different capture on a different config and
  remains what it says it is.

## 0.38.0

### Added

- **`vendor/build_sana_sol_sm89.sh` -- NVLabs' own SM89 Sol-Attn kernel, built
  and proven on this card.** Installs the CuTe DSL runtime and the `sol-attn`
  wheel built out of `coderef/Sana`, then compiles and exercises the kernel.
  There is no build artifact to keep: the SM89 backend is CuTe DSL Python that
  `cute.compile()` JITs on the first eligible call, so the verify step *is* the
  compile, and a later ComfyUI process still pays its own.

  The verify does not take the run's word for it. `_backend_for_arch` returns
  `"triton"` whenever `cutlass.cute` fails to import, silently and by design, so
  sane-looking numbers are not evidence the CuTe kernel ran -- the script
  asserts the selected backend first, then grades the output against two
  controls, upstream's own Triton implementation and dense SDPA. Neither ratio
  test catches a uniformly mis-scaled output; a deliberate 5% gain error passed
  both, which is why a third assertion on the output norm exists. All three
  arms of that violation test go red, and the clean run goes green.

  Like `vendor/rebuild_kernel.sh`, it leaves the upstream checkout exactly as it
  found it -- build leftovers cleaned on every exit path including a failed
  build, and a wheel rather than an editable install, so a routine
  `update-coderef.sh` pull cannot swap the kernel under a measurement. It also
  refuses a dirty checkout before building, because otherwise the commit it
  reports having built is not the code it built, and the mismatch surfaces only
  after the wheel is installed.

### Changed

- **`docs/sol_upstream.md` and `docs/roadmap.md` no longer describe the sm89
  kernel as an unspent dependency.** Both argued for a decision that has now
  been made, and the roadmap item narrows to the half that is still open: their
  API has no `sink_q`, nothing here calls their kernel yet. Two corrections to
  what those files recorded about upstream -- their requirements list omits
  `apache-tvm-ffi`, without which the first SM89 call raises
  `ModuleNotFoundError`, and their stated compile-time failure discipline covers
  a DSL version mismatch but not an absent DSL, which falls back to Triton in
  silence.

## 0.37.0

### Added

- **The euler probe pair: `h3_probe_euler` and `h3_probe_euler_cache`.**
  Owner-requested 2026-08-18: the all-refs workload with only the sampler
  changed to euler — the deterministic arm where step caching works at the
  stock threshold and where cache-on/off is a valid numeric A/B under the
  deterministic-sampler rule. Two graphs so the pair varies exactly the
  cache node; rationale and the measured er_sde/res_multistep context at
  the entries in `workflows/build_workflows.py`.

- **A step-caching probe arm: `h3_probe_cache_easy`, paired against
  `h3_probe_sol_on_all_refs`.** ComfyUI core's EasyCache as a builder option
  (`CACHE_NODE` / `CACHE_NODE_CLASS` in `workflows/h3_config.py`), inserted
  after `SageChainAssert` in both graph formats so the assert still grades the
  attention chain and the cache skips over it whole. The twin graph is the
  control; the pair varies exactly one node. Motivation (NVLabs' 4090 H3
  runtime attributes 3.18x of its 4.44x to step caching at 50 steps), the
  16-step ceiling arithmetic, and the er_sde reuse caveat are documented at
  the constant.

### Fixed

- **`bench/run_graph_arms.py` behavior changes from the 2026-08-18 code
  review** (89b91f3): `--runs` above 1 without `--seed` is refused instead
  of silently producing node-output-cache hits; the warmup renders at
  seed−1 and no longer consumes a run index (it desynchronized seed-matched
  pairs); `sampler_untimed` is reported separately from
  `suspect_cache_hit`; submission-path failures write an error row; every
  writer with a `filename_prefix` is disambiguated per arm; rows carry a
  repo-portable graph name plus content hash. `substrate()` reads package
  versions from dist metadata and anchors git queries to this repo.

### Changed

- **`CLAUDE.md` gains the rule that a rendered clip cannot A/B a numerical
  change.** The sampling trajectory diverges completely from any perturbation, on
  **any** sampler: two arms differing only in sage `mode` diverge at frame 0, at
  the same PSNR as two unrelated clips, under `er_sde` and under deterministic
  `res_multistep` alike. The changed arm is a different *sample*, not a degraded
  version of the same one. The rule keeps its own refuted first draft — which
  blamed the stochastic sampler and prescribed a deterministic one, built and
  disproved within the hour — because the tempting fix not working is the
  transferable part. Consequences: compare knobs at the call
  (`bench/grade_sage_on_capture.py`), not at the output; a perceptual claim
  about a numerical knob needs a distribution rather than a pair; and this
  retro-applies to the 2026-08-13 A/B that chose fp16, which was one clip per
  arm.


- **`REF_LORA_ENABLED` flipped to `False`: reference graphs load the ref2va
  checkpoint directly, with no ref LoRA.** Owner decision 2026-08-18, to take
  a moving part out of the model path ahead of the bandwidth/sparsity
  experiments — the LoRA is a dequantize/add/requantize round trip at load
  whose output-equivalence to ref2va was never verified by a paired render.
  Every graph regenerated; the diff in the reference graphs is the
  `UNETLoader` name and the removed `LoraLoaderModelOnly`, nothing else. The
  named ref-LoRA A/B pair (`h3_image_ref_plus_text_to_video_ref_lora*`) keeps
  the LoRA on purpose; the turbo probes keep their own turbo LoRAs. Reasoning
  and the flip-back condition sit with the switch in
  `workflows/h3_config.py`::`REF_LORA_ENABLED`.

## 0.36.0

### Changed

- **`SAGE_NODE` ships `mode="auto"`, reversing the 2026-08-13 flip to
  `fp16 (most accurate)`.** Every graph regenerated; the diff is that key and
  nothing else. The reason is not speed, which was always the accepted cost.
  The surviving argument for fp16 was a perceptual A/B taken at 124 frames with
  **Sol-Attn absent** — one day before Sol landed — so it graded a configuration
  this repo no longer ships. With Sol on, sage runs only the steps outside the
  sigma window, so fp16 was paid for on every step and delivering on a minority.
  Measured in `bench/results/2026-08-18_attention_defaults.json`; reasoning and
  the reversal condition in `workflows/h3_config.py`::`SAGE_NODE`;
  `docs/evidence.md` corrected in place, since the row there asserted the
  decision was unaffected by the ratio withdrawal.

### Added

- **`bench/check_attention_defaults.py`, which retires a standing uncontrolled
  requirement.** `docs/checks.md` recorded *"Sol-Attn is on by default in every
  shipped video workflow"* as enforced by **one graph, by accident** — the other
  graphs could lose Sol in silence, measured by deliberate violation on
  2026-08-17. This pins every graph `graph_paths()` reaches: Sol values against
  `SOL_RECOMMENDED_CUDA`, sage values against `SAGE_NODE`, and the rule itself as
  **reachability from an output node** rather than presence, since a node nothing
  consumes never executes. Values as well as presence, because the sage mode
  changed in every graph at once the same day and nothing verified the rewrite
  took.

  It complements `check_bench_matches_shipped.py` rather than replacing it —
  that check's subject is the bench harness, this one's is the graphs, and
  neither implies the other. The objection `docs/checks.md` raised against
  building this at all, that a hand-edited exemption list rots silently, is
  answered: the single-frame class is derived from `GRAPH_DIRS`, and every
  exemption must be NECESSARY, so one that stops being true goes red instead of
  covering a graph nobody reads. Shown red four ways, each reverted.


- **`bench/results/`, and the convention that a measurement is a data file the
  prose links to.** `2026-08-18_attention_defaults.json` is six arms over sage
  mode against Sol reachability, generated by
  `make_attention_defaults_json.py` rather than typed, carrying its own
  substrate and caveats. Nothing restates its figures in prose.
- **`bench/preflight_graph.py` reads the attention chain, on UI-format graphs
  too.** It previously skipped them outright, which is why it had nothing to say
  about the hand-built graphs that are the only place this defect appears. It
  distinguishes live / bypassed / muted / absent / orphaned and calls only
  *orphaned* a defect: an ACTIVE `SolAttnMiniMax` whose MODEL output feeds
  nothing never executes, so the render is dense and looks normal. Bypass
  (`mode=4`) is how the shipped graphs disable Sol deliberately and is reported
  without complaint. No node can catch this — an orphaned node is never
  executed, so there is no runtime moment at which to raise.
- **`bench/grade_sage_on_capture.py`, and it has been run.** It grades each sage
  mode against a real captured activation instead of `torch.randn` — the run
  `h3_capture.py`'s own docstring recorded as never having happened, and the one
  `docs/evidence.md` named as what would restore an accuracy figure. Result:
  `bench/results/2026-08-18_sage_accuracy_on_capture.json`, and the
  corresponding row in `docs/evidence.md` moves from withdrawn to restored.
  The measured advantage is smaller than the withdrawn synthetic figure — the
  direction the withdrawal predicted — and it decays with depth while the
  absolute error grows, so fp16 helps least where the kernel is worst.
  It remains kernel fidelity, not perceptual quality.

  The reference is **float64, not fp32**, because fp32 could not do the job: an
  fp32 reference and torch's fp32 SDPA sit the same distance from float64 while
  differing from each other by more than that, so two fp32 paths disagree by
  more than either one's error. Attention averages over the whole key set, so
  cancellation amplifies relative error. The control also gained a negative arm
  — a 1% wrong softmax scale must be caught — after the first version could
  only ever confirm.

## 0.35.0

### Fixed

- **`bench/analyze_sol_error.py`: the eager Sol reference diverged from the
  vendored oracle, and now has a gate that says so.** `colmean` was normalised
  on the key-block axis (`lengths.view(1, 1, 1, n)`) where
  `bench/_sol_attn_reference.py:172` uses the query-block axis
  (`lengths.view(1, 1, n, 1)`). Every interior block is `BLOCK` long so the two
  agree everywhere except the ragged final block — which makes it invisible at
  any block-aligned length and wrong at every ragged one. Measured against the
  oracle: rel_l2 0.013 at t=330, 0.091 at t=514, 0.166 at t=1000, against 0.0004
  after the fix. Production S is 98498 = 1539*64 + 2, i.e. ragged.

  `calibrate_against_oracle` now runs before any capture is read and refuses to
  report on disagreement, and its default lengths deliberately mix aligned and
  ragged. The sibling gate in `bench/simulate_track_b_lite.py` tests at t=320 =
  5*64 only, so it could not have caught this class while naming ragged-block
  handling as a known failure mode in its own refusal text. Verified both ways:
  the gate passes at 0.0004 on the fixed code and goes red at 0.0912 with exit 1
  when the defect is reintroduced.

  **All twelve decomposition rows were re-measured after the fix.** The
  correction moved each by under 1.2 points and changed no conclusion, because
  the mis-normalised row and column are two entries out of 1540 at production
  scale. Post-fix, the quant/sparsity ratio runs 14.43% to 62.20% across blocks
  0, 24, 40 and 49, with block 49 climbing monotonically (49.42%, 55.56%,
  62.20%).

- **`bench/red/show_red_check_capture_manifest.py` covers the enumeration
  branch.** Its eight existing cases all call `check_manifest` on one file, so
  none of them could reach the code deciding *which* directories are captures —
  the part that was blind. Four cases added driving `main()` against synthetic
  collections: a capture with tensors and no manifest must go red, and three
  near-misses pin the false reds that the fix could have introduced (a fully
  manifested collection, a non-capture directory sitting beside a capture, and
  an empty collection, which must skip rather than fail).

- **`bench/analyze_sol_error.py`: the three errors now share one denominator.**
  `quant_l2` was normalised by `‖out_eager‖` while `sparsity_l2` and `total_l2`
  used `‖out_dense‖`, because `rel_l2_error` divides by its second argument. The
  quadrature identity the `rho` column rests on requires all three in the same
  units, so it never held, and part of what was reported as vector alignment was
  the denominator mismatch. Block-level ratios barely move; near-zero per-head
  `rho` moves a lot, which is the point — block 49 step 14 head 2 sits at
  `-0.0008` against `-0.0022` before, a sign that was never resolved to begin
  with. Added `rel_l2_against` so the sharing is explicit rather than
  incidental. (Figures here are from the fully corrected run, i.e. after the
  eager-reference fix below; the two landed the same day.)

- **`bench/analyze_sol_error.py`: `cosine_sim` returned values above 1.0.**
  It printed `1.047609` on a real run, which Cauchy-Schwarz forbids, so it was
  accumulation error over ~1e8 fp32 terms. Now accumulated in float64 in chunks.
  Verified against a synthetic control at the same element count: fp32 gives
  `1.021457` where float64 gives `0.998752`. `rel_l2` is not affected to the
  same degree (+0.053%) because the two norms' errors cancel in the ratio, which
  is why the tables reproduced while this did not — and why the figures support
  about three significant figures rather than the six printed.

- **`bench/analyze_sol_error.py`: `--heads` takes a prefix, and now says so.**
  `q[:, :n]` is heads 0..n-1 of 56, not a sample of 56, so every aggregate row
  was a first-n-heads figure while being labelled by block. The banner now
  prints `n/total` and names it a prefix.

- **`bench/check_capture_manifest.py` enumerates captures, not manifests.**
  Globbing `*/manifest.json` and validating the hits could only fail on a
  malformed manifest, never on a capture nobody recorded — the case it exists
  for. It reported ok on a collection where one of two capture directories held
  twelve tensors and no provenance at all. It now walks directories containing
  `qkv_*.pt` and fails on any without a manifest, naming them.

- **`bench/run_capture.py` no longer exits 0 on a failed render.** It returned 0
  as soon as `prompt_id` appeared in `/history`, reading `status_str` only to
  print it — but a render that errored appears in history like any other. It
  also polled in an unbounded `while True` with a bare `except Exception: pass`,
  so an unreachable server was indistinguishable from a slow one. Now: 0
  success, 1 submit failure or repeated poll failure, 2 non-success status, 3
  timeout, with a `--timeout` ceiling.

- **`bench/simulate_track_b_lite.py` shares `find_one` instead of `glob(...)[0]`.**
  The hazard was already documented in `bench/verify_multistep_capture.py`,
  which replaced the same pattern; both files landed in one commit and only one
  got the fix.

### Changed

- **`bench/simulate_track_b_lite.py` states what its `fp16_pv` arm is.** The
  flag returns the unquantized eager reference rather than widening the PV
  product, so the value it reports is the sparsity error and its "recovery %"
  reduces to `1 - sparsity/total`. That makes it a valid upper bound on what a
  perfect PV matmul could recover, and not a measurement of one — the module
  docstring now says so before any figure is quoted.

- **`docs/roadmap.md`, `docs/open_experiments.md`:** gate 17b
  (`bench/analyze_sol_error.py`) and 17c (the reference-heavy capture) are done
  and described as done; both were still listed as scaffolded with every entry
  point raising. 17a and `sol_block_probe.py` genuinely are still scaffolded and
  stay that way. The headline result is recorded: the quant/sparsity ratio runs
  roughly 15% to 62% against a 5% retirement threshold, so the 16-bit PV
  question does not close.

- **`docs/capture_manifest_schema.md`:** removed the claim that the manifest
  makes a tensor traceable to "per-head error decomposition analysis". No such
  property exists in the schema; that is a property of an analysis run over a
  capture. The document also no longer names a schema version in prose —
  `bench/check_capture_manifest.py::SCHEMA_VERSIONS` owns the accepted set, and
  a version written into prose had drifted against the code once already.

- **`CLAUDE.md`, `docs/checks.md`:** the manifest schema row points at the code
  constant rather than naming a version, and the `check_capture_manifest.py` row
  records its red harness (which existed) and its blind spot (which did not stop
  it reporting ok).

## 0.34.0

### Added

- **`docs/hardware.md`** — the box every number here was measured on: what
  bounds this workload, what has been ruled out as a bottleneck, and which host
  settings silently invalidate a timing comparison.

  **The finding that motivated it: a GPU board power limit moves render times
  and is invisible to everything this repo inspects.** Not in a workflow JSON,
  not in the capture manifest, not in a ComfyUI log line, and it persists across
  reboots once a systemd unit exists for it. A run at a changed limit yields an
  ordinary-looking s/it that is not comparable to anything on
  `docs/bench_plan.md`, and no check would notice. The limit on this box was
  changed away from stock on 2026-08-17, after every timing in `bench_plan.md`
  and `SOLATTN.md` had been recorded — those stay correct at stock, but
  reproducing one now requires resetting first.

  Recorded as an **uncontrolled requirement**, not as coverage: the requirement
  is "compare timings only at equal power state" and the control is a script
  somebody has to choose to run.

  **Corrected same day.** The doc and the `checks.md` row first said the power
  limit appears in no capture manifest. It does — `provenance.gpu_power_limit_watts`
  already existed and the one manifest on disk populates it. The claim was
  inferred from the schema's `required` list without reading its properties, and
  a peer session refuted it. The gap is real but differently shaped per
  artifact: for captures a field exists and nothing asserts it, for render
  stamps no power field exists at all, and bench runs persist nothing. Only the
  middle case is fixed by adding a field.

  The doc deliberately carries no values. Cross-linked from `CLAUDE.md`'s
  reference table, `docs/comfy_notes.md`, `docs/bench_plan.md`'s ground rules,
  and `docs/evidence.md`'s Environment section, which owns the software half of
  state-not-in-git while this owns the host half.

- **`bench/hwinfo.py`** — prints host, GPU, power state and PCIe topology.
  Not a check; it asserts nothing. It exists so `docs/hardware.md` can describe
  the machine's shape while the drifting values live in output, per the number
  rule. Flags a non-stock power limit and any PCIe device linked below its
  capability. Standard library only, so it runs on a bare interpreter rather
  than `uv run`, which would build a second venv and write the `uv.lock` that
  `docs/comfy_notes.md` says must not exist here.

- **`bench/red/`** — the home and the shared spine for red harnesses, the
  programs that prove a check can fail. `harness.py` carries four primitives
  (`subject`, `fixture`, `baseline`, `case`) and one rule.

  **The rule is derived, not authored.** A case declares a KIND rather than an
  expected verdict: `MUTATION` requires the verdict to differ from the unmutated
  baseline, `NEAR_MISS` requires it to match. One rule covers every case that
  will ever be added, which is what stops a harness suite collapsing into tests
  for tests — an authored expectation is itself a claim needing verification,
  and each new case adds another. It doubles as the needle check: a mutation
  that never reached its subject leaves the verdict unchanged, which is exactly
  what `MUTATION` already asserts. An exception is `ERROR`, never "differed".

  `fixture()` exits 2 when a required capture is absent, so "did not run" stays
  distinguishable from "passed" — the pattern `check_distill_settings.py` set.

- **`bench/red/spine_control.py`** — the control on the spine itself. Shared
  harness infrastructure fails silently across every harness at once, which is
  the defect the directory exists to remove, one level up. Two fixtures run as
  subprocesses: `_fixture_inert.py`, whose mutation does not mutate, must exit
  1; `_fixture_healthy.py` must exit 0. Red for the right reason and green for
  the right reason, both structural rather than authored per case.

- **`docs/drift_frontier.md`** — the tracking file for the doc-drift and
  control-calibration work, one entry per open decision with its dependencies,
  recommendation and status. Annotate-don't-rewrite, so a resolved entry keeps
  the reasoning that produced it.

- **`docs/check_postmortems.md`** — the per-defect narrative and frozen run logs
  moved out of `docs/checks.md`, which had grown to the point where the index it
  promises was a small fraction of it.

### Fixed

- **Every red harness in the repo exited 0 unconditionally.** All three computed
  an expected-outcome mismatch, printed it, and returned success whether every
  case came back red or every case came back green; one had no comparison at
  all. They are the cited evidence for rows of `docs/checks.md`, so those rows
  cited programs that could not fail. Ported onto the spine, which supplies the
  exit code they were missing.

- **Two requirement rows claimed "enforced by nothing" against controls that
  exist.** The `node_id` rule is enforced by `bench/check_node_ids.py` against a
  committed manifest, and the connected-block figures by
  `bench/analyze_canvas_geometry.py`'s `connected_frac`, which has been shown
  red. Both cells had been transcribed from narrative prose rather than derived
  from `bench/`, which is the failure the table exists to find. A third row had
  been stale since the control landed.

- **`scripts/experimental/` hardcoded an absolute path to this checkout**, in
  every file that resolved the repo root, and one that resolved ComfyUI's. They
  were the only such paths in the tree and they were committed. Now
  `Path(__file__).resolve().parents[3]`, verified by running the harnesses from
  outside the repo — they had worked *because* of the hardcoding, not despite
  it. The routing harness reads its capture location from an environment
  variable rather than a home-directory literal.

- **Docstrings across `bench/` attributed rules to `CLAUDE.md` that had moved.**
  The `node_id` rule and the `import nodes` trap now live in
  `docs/comfy_notes.md`; the GPU-contention warning in `docs/checks.md`. The
  Sol-Attn default is stated from the graphs rather than cited to a document
  that never carried it. Rechecked wrap-tolerantly, since single-line `grep`
  over hard-wrapped prose returns false negatives on exactly this question.

- **`docs/open_experiments.md` cited a script by an absolute path** that
  resolved nowhere, breaking `check_doc_links.py`. The file exists; the citation
  carried a leading slash.

### Changed

- **`docs/checks.md` split.** Keeps the index, the standard, the run
  instructions, the deliberately-not-checked list, a new uncontrolled-requirement
  table, and one-line gaps. The standard is now scoped: red-first calibration
  applies to mechanically specified behaviour, and explicitly not to a check
  whose expected value is the measurement itself, where demanding it invents a
  threshold rather than testing one. Gap items point at the columns instead of
  quoting tallies that had drifted from them.

- **`CLAUDE.md`'s index restructured** into read-first, reference, and code
  sections. The three deep dives no longer have top-level rows: the file says
  they are reached through their parents and must not be quoted against them,
  which the flat list contradicted by presenting them as peers. Their warnings
  moved into the parent row.

## 0.33.0

### Added

- **`bench/analyze_canvas_geometry.py`** — the source for every table in
  `docs/h3_input_impacts.md`, which previously carried them hand-transcribed
  with nothing behind them. That is the drift CLAUDE.md's guiding principle
  names, and it bit within a day: see Fixed below.

  `analyze_morton.py` answers "what does this permutation do" on one canvas;
  this answers the comparative question across all of them. `--markdown` emits
  the doc's tables ready to paste, `--lengths WxH` sweeps the length axis.

  **The canvas set is enumerated from `adapt_canvas`, not listed.** A
  hand-maintained list would agree with `docs/h3_resolutions.md` forever and
  stop agreeing with the code the first time the area cap or the rounding
  moved.

  **Two controls run before it prints anything, and both have been shown red.**
  The vendored `morton_perm` against `analyze_morton`'s independent
  implementation, borrowed rather than rewritten. And its connectivity figures
  against `docs/morton.md`'s published four worst canvases — those came from a
  different implementation, so agreement is confirmation rather than tautology.
  Mutating `connected_frac` from 6-neighbour to 26-neighbour adjacency moves
  1952x544 from 51.5% to 90.0% and the control fails with all four rows named.
  The mutation script asserts its target string is present first, because a
  `.replace()` that matches nothing prints exactly what a broken check prints.

### Fixed

- **`docs/h3_input_impacts.md`'s length table said "all lengths on the grid"
  and listed ten of fifteen.** 141, 158, 175, 192 and 226 were missing, and one
  of them matters: 175 frames is a third `latent_t % 4 == 0` length the page
  did not mention. Found by running the committed script against the page it
  was written from, one day after the page was written by hand.
- **The same section credited alignment with the wrong effect.** It said
  aligned lengths reach 100% connected; 328 and 345 reach it too. What
  alignment actually buys is radius, 1.581 against 1.598-1.645. The column that
  tracks `% 4` cleanly is radius, and the case to avoid is `% 4 == 3` (158,
  226, 294, and the shipped 362), which is where fill and connectivity dip.

## 0.32.0

### Added

- **`docs/h3_input_impacts.md`** — the page for choosing a canvas and a frame
  count *together*. Three docs already own the pieces and none of them owns the
  interaction, so the question "what should I actually type" had no home.

  It leads with **block maps** rather than statistics. `analyze_morton.py`
  already prints them and nothing in `docs/` had ever shown one, so every
  discussion of block geometry was conducted in radius and fill numbers whose
  meaning a reader had to take on faith. One frame of raster order next to one
  frame of Morton `3d` explains the entire setting in two pictures.

  **New measurements it owns**, none of which exist elsewhere: the per-canvas
  ranking of Morton `3d` over all 48 legal canvases, the
  `h/32 % 4` by `w/32 % 4` grouping behind it, and the `latent_t % 4` length
  effect. The six canvases with both token axes divisible by 4 (768x768,
  896x768, 1024x768, 1152x768, 1280x768, 1664x640, plus portrait mirrors) score
  radius 1.61 with zero spread; at 243 or 311 frames they reach 100% connected.

  **Cross-checked before it was written down.** The independently implemented
  connectivity pass reproduces `docs/morton.md`'s four worst canvases to the
  decimal (51.5 / 52.5 / 52.7 / 53.7) and its 124-frame floor to 0.1 points,
  which is why the new numbers beside them are trustworthy.

  **One claim was refuted on the way and did not survive into the doc.** The
  hypothesis that `latent_t % 4` also explains `morton.md`'s observation that
  `3d`'s floor degrades with length is wrong: at 243 frames, which is perfectly
  aligned, the floor is 46.1%, worse than 362's 51.5%. Two independent effects,
  and only the top of the range is modular. `morton.md` was right and the page
  says so.

  **It also states that there is no 100,000-token budget**, because the
  question keeps being asked. 99,864 is a Triton int32 offset crossing that
  `preflight.py` already records as fixed in every sage build able to run this
  node, and every shipped graph is past it. Three files call it "the model's
  ~100k ceiling", which reads as a property of the checkpoint; no independent
  upstream claim of one was found in `docs/sol_upstream.md` or the Sol-Engine
  tree.

### Changed

- **`docs/h3_resolutions.md`** gained a **Token grid** column beside the
  existing VAE latent, and a paragraph naming the difference. Two grids were
  both called "latent" — `W/16 x H/16` here, `W/32 x H/32` in `morton.md` and
  in every Sol-Attn discussion — and tokens per frame is the product of the
  second. Also a full on-grid length table with video tokens and attention cost
  per frame count, since length is the second-largest cost lever and the page
  previously showed only the rounding rule.
- **`docs/SOLATTN.md`** and **`docs/morton.md`** link the new page; SOLATTN's
  "two deep dives" table is now three.
- **`docs/morton.md`** records that its "pick any width" rule is first-order
  correct but not the whole structure, with the `w/32 % 4` finding and a
  pointer to the page that owns it. Written the same hour as the finding
  specifically so the two pages do not spend a day disagreeing, which is the
  failure mode that produced that page's own ownership rule.

## 0.31.0

### Added

- **`bench/check_graph_discovery.py`** — guards CLAUDE.md's `graph_paths()`
  rule, which every graph-walking check satisfies today and nothing enforced.
  A bare `workflows/*.json` glob misses `workflows/image/`, so a check written
  next month could pass green over a subset and look identical to one that
  covered everything.

  **It parses rather than greps, and that is load-bearing.**
  `check_ref_prompt_labels.py:66` contains `WORKFLOWS.glob("*.json")` inside a
  comment explaining not to do it; a regex flags that line, and the natural
  "fix" is to reword the comment — teaching the opposite of the rule. The AST
  cannot see comments or docstrings, both verified green in the harness.

  **First run found two sites and both were false positives**, which is the
  more useful result. `check_single_frame.py:163` enumerates `/proc`, not
  graphs — the rule now requires a graph-shaped receiver. And
  `check_ref_prompt_labels.py:151` walks the tree deliberately: its subject is
  discovery *coverage*, so routing it through `graph_paths()` would make it
  derive its expectation from the thing it checks. It is the one exemption, and
  `EXEMPT` records the mechanism plus the cost (the rest of that file is
  unaudited).

  Nine harness cases, four red and five green, in `internal/`. The green ones
  carry the weight: a check that fires on a comment, on `/proc`, or on
  `rglob("*.py")` gets ignored, and an ignored check is worse than none.

  Unlike the `node_id` harness, this one exercises the **collector**
  (`enumeration_sites`) on synthetic sources rather than feeding the comparator
  a mutated baseline — that gap is what let a self-referential collector slip
  through the previous check's seven mutations.

## 0.30.0

### Added

- **`_ref_prompt` takes a role per image socket, so a graph can wire three
  references.** `_REF_IMAGE_NODES` has declared 3 sockets since it was written
  while `_ref_prompt` only ever emitted 2 labels, so the third was unreachable
  — `check_ref_prompt_labels.py` requires the prompt to name exactly what the
  graph wires, and the mismatch made 3 a build failure rather than an option.

  `images=` now accepts `True` (the historical pair) or an explicit tuple of
  roles in socket order: `("character", "garment", "environment")`. **An int is
  deliberately rejected** — `images=3` would make the generator invent a
  relationship for a picture it cannot see, which is the failure `1fa5607` paid
  for when the environment template asserted "architecture" for whatever image
  happened to be wired. The caller picks the file, so the caller declares the
  role.

- **`_env_label()`**, because the establishing beat had a hidden ordinal
  dependency. The shot prose and the `structure` summary hard-coded
  `<Subject 2>` as the environment, which is only true when the roles are the
  historical pair. With a garment at socket 2 the generated prompt read "a
  medium shot establishes <the garment>". Now resolved by role, and the beat is
  dropped entirely when no socket carries `environment`. Found by reading the
  three-role output, not by any check.

### Fixed

- **`ref_image_count` above the placeholder count truncated silently.**
  `[A, B][:3]` is two files, not an error, so a graph asking for three
  placeholder references wired two and surfaced much later as a prompt/label
  mismatch naming the wrong cause. Now refuses and asks for explicit
  `ref_images=(...)`.

### Verified

- **Byte-identity, as a control rather than an assertion.** All 43 prompts were
  snapshotted via `--print-prompt` *before* the change; after it, 0 of 43
  differ and 0 of 87 regenerated graphs changed. `check_ref_prompt_labels`,
  `check_prompt_guide_conformance`, `check_generator_constants` and
  `check_workflow_schema` pass, and `smoke_h3.py` rendered.

## 0.30.0

### Added

- **`bench/check_node_ids.py` and `bench/node_id_manifest.json`** — the first
  guard on the rule CLAUDE.md opens with, checked against a baseline the schema
  cannot move.

  Nothing enforced it before, and not by omission. **Every existing guard
  derives its expectation from the thing it is checking**:
  `check_workflow_schema.py` validates saved graphs against a live
  `/object_info`, and `build_workflows.py` regenerates all 89 graphs — both
  downstream of the schema. So renaming a `node_id` and regenerating leaves
  every artifact internally consistent and every fast check green, while the
  only broken artifacts are the owner's live graphs outside the repo, which no
  check can see. That is the failure mode CLAUDE.md describes, and it was
  unguarded.

  A control whose input is derived from the thing it is checking cannot fail —
  the fourth phrasing of the family in `docs/checks.md`, and this time on the
  repo's stated most important rule. The fix is the only thing that fixes that
  family: a committed, independent baseline that a schema change cannot
  rewrite. The manifest records `node_id` plus ordered input and output names,
  which is the whole of what a saved graph addresses positionally.

  **If it goes red the default is to revert, not to `--write`.** `--write` is
  for the two changes CLAUDE.md permits: a new node, or an input/output
  appended at the END.

  Seven branches, each shown red for the right reason, harness in `internal/`:
  rename, reorder, mid-insert (the 2026-08-10 `head_chunks` bug), output
  reorder, node disappearance, permitted append, and new node. `collect()` was
  then verified independently — the schema objects and a source grep for
  `node_id="..."` return the same 8 strings — because the mutation harness
  exercises `compare()` and would not have caught a collector that returned the
  manifest back to itself.

## 0.29.0

### Added

- **`bench/analyze_routing.py`** — the routed-density instrument
  (`docs/open_experiments.md` #18). Answers the question under every curve
  comparison in this repo: a fixed-`tau` A/B does not hold the operating point
  fixed, because `kcvar` is a variance over the block centroids the permutation
  defines. Emits **two** densities, deliberately, because conflating them is
  how the prototype behind this mislabelled its own output: *ordering-effect*
  (forced-exact pairs dropped from numerator and denominator — the number to
  compare curves with) and *kernel* (what the kernel routes, the number to size
  `routed_cap_percent` against). Also reports the `tau` that reproduces
  raster's density per ordering.

  The pooling is imported from upstream's eager reference rather than
  transcribed; the threshold is transcribed and cross-checked against a second,
  deliberately naive implementation in the same file. No GPU, no model, ~5 s.

  Measured on the 2026-08-15 capture (1344x768, 124 frames, blocks 24/49, 8 of
  56 heads, float routing rule — the kernel's INT8 quantization is skipped):
  `hilbert` routes **1.171x** raster at block 24 and **1.113x** at block 49,
  while `3d` routes **0.987x** at block 49. The direction is not derivable from
  block coherence, and it is not even the same sign across curves at one depth.

  **Six controls, each shown red for the right reason** before the numbers were
  trusted; the mutation harness is in `internal/`. One of them exists because
  the first version of the identity control compared `torch.arange` to
  `torch.arange` and stayed green under a deliberate mutation — the same defect
  as `verify_adjacency` in 0.28.0, written the same day it was found. It is now
  a within-block shuffle, which is a non-trivial permutation with a forced
  answer, paired with its converse so both cannot pass by nothing happening.

### Fixed

- **`h3_capture.py` built its transposed copies on the device.** Three
  `[1,H,S,D]` buffers next to a model already near the card's limit: 5.4 GiB at
  S=124,582, against ~6.7 GiB headroom. The copy now happens on the host, which
  changes where the intermediate lives and not a byte that is saved. This is
  plausibly **why every capture on this box is 124 frames** — the failure would
  have presented as a length limit rather than a tooling one, and 124 frames is
  37,826 tokens, below the ~60k floor `docs/SOLATTN.md` warns about.

## 0.28.0

### Fixed

- **`verify_adjacency` was a check whose input could not fail.** It counts
  non-adjacent steps along the Hilbert curve, and its only call site
  (`bench/analyze_capture.py`) passed `side=64`. On a power-of-two square,
  adjacency is Hilbert's defining property — zero is what every correct
  implementation returns, so the assertion could only go red on a corrupted
  `hilbert_d`, never on the ordering being scored. The ordering that actually
  runs is a rectangle clipped out of that square, which splices the curve in
  **6 places of 1007** at latent 24x42.

  So the repo asserted "a Hilbert curve never jumps" in `sol_curves.py`, in
  `docs/morton.md` and in the node's UI tooltip while holding a green check
  structurally incapable of contradicting it. `verify_adjacency` now takes
  `height`/`width`; the square stays a gate, the rectangle is reported and
  never gated (no threshold for it has been established). Corrected in all
  three places. Indexed in `docs/checks.md`.

- **`docs/morton.md` predicted the sign of a routed-density change it could not
  derive.** The argument moved the threshold and held the scores fixed. Both
  move: `colmean` is taken against the same mean-centred pooled key centroids
  whose variance sets the threshold, so coherence raises numerator and
  denominator together and the formula does not say which wins. Replaced with
  the source reading, and marked as not-derivable rather than re-signed.

- **`docs/morton.md`: "the residue is the frame-sliding, which no curve fixes"
  was wrong.** A per-frame phase change fixes most of it — 86.1% to 93.8%
  connected at (37,24,42). The claim survived because the metric that would
  have caught it excludes frame-straddling blocks, which are exactly the blocks
  a phase change acts on.

- **`docs/morton.md`: the connectivity and radius table did not state its
  restriction.** The figures reproduce exactly (59.9%/90.0%, 4.88/4.49/11.90)
  but only over single-frame blocks; over all blocks it is 57.1%/85.8%. The
  metric is **undefined for `3d`**, whose every block spans four frames, so
  that table can never rank `3d` against `hilbert`.

- One corrupted sentence in `docs/morton.md`'s capture section.

### Added

- **`docs/morton.md`: "geometry does not rank orderings".** Two arms with
  indistinguishable block geometry (radius 4.09 vs 4.10, both 93.8% connected)
  differ consistently on centroid fidelity at all three captured depths, and the
  geometrically best arm of four is not the best on activations — the ranking
  inverts. So spatial compactness is the **wrong objective**, not a saturated
  one, which forecloses more: `analyze_morton.py`'s radius and connectivity
  answer mechanism questions and are not a basis for choosing a curve. A second
  leg from the legal-canvas sweep: geometric rankings are not stable across
  canvases either, so tuning against them picks a different winner per
  resolution.

  Stated separately from **the priority call** ("a fourth curve is not where the
  remaining quality is"), which is a judgement resting on the 0.3–0.9% total
  spread, speed-identical arms, `3d` winning by mixing frames, and link 6 being
  untouched. Kept apart deliberately so the methodology finding cannot be cited
  as if it settled the priority question.

- **`docs/open_experiments.md` #18: routed density under each curve.** The
  cheapest unrun item in the repo — captures on disk, no render, no GPU — and
  the missing denominator under every ordering A/B run here. Specifies scoring
  against upstream's eager reference rather than a fresh transcription of the
  threshold formula, and records the sigma axis as the prerequisite that kills
  any compensation scheme if it goes the wrong way.

- **Canvas sweep and the serpentine form**, in `docs/morton.md`, over **all 48
  legal landscape/square canvases**. Rotation is the wrong shape of the phase
  fix: a Hilbert curve is an open path, not a cycle, so rotating splices its two
  ends together, and that costs more than the alignment buys on 5 of the 48
  (1536x672, 1440x736, 1408x736, 1376x736, 896x768). Reversing alternate frames
  achieves the same alignment with no splice and **regresses on none**. Also
  records that only 3 of 48 canvases have tokens per frame divisible by 64, so
  the phase problem is the normal case; and that **`3d` is the most
  canvas-sensitive ordering, not the most robust** — floor 67.2%, below plain
  `hilbert`'s, against 97.9% at the one canvas everything here is measured on.

  **The first version of this sweep used four canvases H3 cannot render**
  (1152x640, 1024x576, 832x480, 1216x704), taken from general video-model habit
  rather than from `docs/h3_resolutions.md`, and drew a conclusion from one of
  them. Corrected the same day; the legal set is 48 landscape/square canvases
  plus portrait mirrors and does not contain the obvious sizes.

- **The one-canvas warning**, at the top of `docs/morton.md`. Every activation
  measurement on that page is 1344x768 — which is both the most expensive canvas
  in the legal set (1.00x; 1152x768 is 0.73x, 1024x768 0.58x, 768x768 0.33x) and
  `2d_frame`'s worst, so the shipped curve is judged where it is weakest and the
  alternatives where they flatter best. Aspect ratio is the lever, not
  resolution, and `CANVAS_TIER` makes it one edit.

- **The custom-node import-order trap**, in `CLAUDE.md` and
  `sol_curves.install()`. ComfyUI imports packs in bare `os.listdir` order with
  no sort, so anything patching another pack must do it at `execute()` time and
  count what it patched. Recorded before it bit.

- **A tripwire for `centroid_tail`'s "~1.4x"** in `docs/evidence.md`. It is the
  operation, not end-to-end; ours measured 2.5% e2e, making it the smallest
  knob in the node rather than the largest. Two readers have now quoted it as
  e2e. Deliberately not added to the `retraction-consumers` block, because the
  string is correct where it appears.

- **The operating-point warning on the curve node and in `CLAUDE.md`.** Block
  membership feeds `kcvar`, so changing the ordering moves the routing
  threshold: a fixed-`tau` curve A/B varies two things. Notes that
  `tau_profile` is keyed per transformer block and therefore cannot express a
  sigma-dependent correction.

## 0.27.0

### Fixed

- **The generic environment template asserted an attribute it could not know,
  and it built one.** Every image-reference arm said `<Subject 2> is the
  environment in <Picture 2>, whose **architecture**, palette, and lighting are
  carried into the target video` — for whatever image happened to be wired.

  Measured the same day (`docs/prompt_length_experiment.md`, `1fa5607`) against
  a mountain-lake reference containing no buildings: the arm whose
  `detailed_description` said nothing about the environment rendered the
  subject **inside a timber veranda with a chalet beside it**, and the arm
  whose description named the lake, reflection and meadow produced **no
  structure at all**. Identical `subject_definitions` in both.

  Now "setting", which is true of any environment. The generator cannot see the
  reference, so it must not assert content; naming the real content is the
  prompt author's job. Seven template strings, 87 graphs regenerated, zero
  graphs still claiming architecture. The hand-written image scenes keep theirs
  — a green-lit marble corridor genuinely has architecture.

### Added

- **`docs/h3_references.md`: "A label is a bare ordinal. You have to say what
  it is."** Records the authority order from two n=1 results, neither in any
  guide: `subject_definitions` beats `retention_analysis` (the owner's
  blonde-against-a-brunette-reference case), and a specific
  `detailed_description` beats `subject_definitions` (the architecture case
  above). Together: **a wrong word in the definitions is load-bearing exactly
  to the extent that nothing downstream contradicts it**, and silence
  downstream is not neutral.

  With what follows for writing one, including the counter-intuitive part —
  describe the environment in `detailed_description` *even when a reference
  supplies it*.

- **The verdict in `docs/prompt_length_experiment.md`**, judged against
  predictions written before either clip was watched. Four of five confirmed.

  The fifth was the control and it was **invalid by construction**: both
  prompts carried the identical camera sentence, the long arm executed the move
  and the short one did not, because the long arm elaborated it with
  consequences. You cannot hold one sentence constant inside a prompt whose
  length you are varying — a conditioning model reads it in context, not as a
  string. Consequence recorded plainly: the direction is credible and the
  magnitude has no noise bound.

## 0.26.0

### Added

- **`bench/preflight_graph.py` — grade a prompt and price a render before you
  queue it.** Static, no CUDA and no server: it grades against the guide's
  mechanical rules (ordinals derived from sockets, every defined label cited in
  `detailed_description`, one retention line per label, markers never crossing
  the visual/audio sets, no `(Sx)` in `retention_analysis`, `<d>` placement and
  language tag, cut times inside the clip) and prices the packed sequence.

  **Takes paths and never globs**, which is the point: both prompt checks only
  ever saw `workflows/*_api.json`, so hand-built graphs were ungoverned and a
  new directory was invisible. Reports, never refuses — a tool that blocks you
  in your own repo gets disabled. Names what it cannot count: a video reference
  is a floor, not a budget.

  Shown red six ways. A seventh mutation looked green and had simply failed to
  apply, caught only by diffing the mutant; that is recorded in its row rather
  than quietly fixed. It imports `wired_labels`, `_audio_sections_optional` and
  `_STRUCTURE_PROBES` rather than restating them — the first run without the
  last two reported FAIL on 7 of 8 image graphs for sections that structurally
  cannot apply.

- **`docs/prompt_length_experiment.md`**, pre-registered before either clip was
  watched (`7de7a16`): five predictions with confidence, an internal control,
  and the RoPE confound named up front. Prompt length is not a free variable —
  `text_len` is the temporal cursor for every reference block and both target
  segments (`comfy/ldm/minimax/model.py:307-318`).

### Changed

- **CLAUDE.md: 293 lines to 215, and it stopped being an archive.** An audit
  found 19 stale or wrong claims. Six were the Triton block alone, which still
  described a deleted pack as "moved, not deleted" and told the reader to
  symlink it back to run a probe that no longer exists; that block is now four
  lines. Also corrected: two probe graphs to four, the dotted-import list
  (which omitted the two files importing at *module* scope, the worse case),
  "pinned exact" to 0.999 visual / 1.0 audio, and a `nodes.py:2245-2250`
  citation that resolved to our own 194-line file — inside the paragraph
  warning that `import nodes` finds ours.

  Added what the day cost to learn: 362 as the ceiling with what it rests on,
  `REF_LORA_ENABLED` with an explicit "do not call fl2va+LoRA and ref2va
  interchangeable", restarting by port owner rather than `pgrep | head -1`, and
  "when you reverse a decision, update the document that argued for it."

- **Five reference arms stopped asking for something they did not contain.**
  `h3_ref_video_only`, `h3_ref_video_audio`, `h3_ref_video_to_video`,
  `h3_ref_image_video_audio` and `h3_probe_sol_on_all_refs` carried
  `<Video 1> (cut and pacing structure): weak_reference` inside a single-shot
  prompt whose summary says "a single continuous shot". A cut-structure
  reference on a cutless prompt asks for nothing. Narrowed to camera movement;
  the cut language can return if these arms ever gain a shot timeline.

- `bench/check_lora_alpha.py` fails with a sentence rather than a bare
  `TypeError` when a constant ends in `_LORA` and is not a filename.

### Fixed

- **`docs/evidence.md` records what the retraction ledger cannot see.** A
  `mean_rtol` spread survived the fp8/fp16 withdrawal because the sentence
  contained no listed phrase: the check defends the *spelling* of a retracted
  claim, not the claim, and derived figures are exactly that shape. Also
  records that the gap **cannot be closed by adding a row** — the allowlist
  needs every phrase to have a legitimate home, so a spelling that should
  appear nowhere emits a permanent warning. Tried, reverted the same hour.

- `bench/preflight_graph.py` priced no 1024x768 arm on first writing: it
  assumed the linked source was `MiniMaxH3Resolution` *and* assumed its `wide`
  shape, so the most OOM-prone graphs in the repo reported "cannot price"
  rather than a number. Follows the link now.

## 0.25.0

### Changed

- **The three Sol-Attn documents now own one topic each, and `SOLATTN.md` is
  the entry point.** They had been peers, all three asserting Sol-Attn numbers,
  with the link graph running upward only -- `morton.md` and the upstream doc
  both pointed at `SOLATTN.md` and it linked to neither. That is a hub with no
  spokes, and it produced live contradiction rather than the theoretical kind:
  `docs/SOLATTN.md` sold Morton as "worth 1.16x alone" in its Configuration
  findings while `docs/morton.md` carried the retraction of exactly that
  figure.

  The split: **`SOLATTN.md`** owns what we run and every number measured on
  this box. **`morton.md`** owns token order. **`sol_upstream.md`** owns what
  other people claim and asserts none of our numbers. The rule that keeps them
  apart is that a number is stated once, in the page that owns it, and
  everywhere else is a one-line verdict plus a `Canonical:` link.

- **`docs/sol_engine_reference.md` is renamed `docs/sol_upstream.md`**, because
  it stopped being about one vendor's framework. It gained the Sol-Attn paper
  and a section on the two other ComfyUI Sol-Attn packs. `CHANGELOG.md:455`
  still names the old path and is deliberately not rewritten; it records what
  0.18.0 added.

- **The Sol-Attn paper is read and recorded.** arXiv 2607.24027 was fetched on
  2026-08-16 -- abstract, ablation summary and HTML, not the full PDF, and the
  doc says so. `docs/morton.md` had listed it as unread in five separate
  places, including as the cheapest unrun item on its own to-do list. Four
  findings land: the contribution is the **correction** (unselected blocks
  reuse their proxy scores) rather than the routing, and its advantage widens
  as sparsity rises; `tau` is the paper's `beta` and **is never swept**, so
  nothing upstream adjudicates our 1.3 against Sol-Engine's 1.0; **H3 is not
  evaluated in the paper at all**; and **no token reordering appears in it**,
  which narrows the Morton attribution for the fourth time -- from "upstream
  says the payoff is at higher sparsity" to "not part of the published method".

### Fixed

- **`docs/morton.md` contradicted itself about `h3_capture.py`** in three
  places, saying it had never been run while its own results section recorded
  the first run. Also corrected: it described `SOL_RECOMMENDED_CUDA` as pinning
  `morton_curve="2d_frame"` when `h3_config.py` had moved to `"3d"`.

- **The Morton permutation cost is corrected from "0.8 s of 861, or 1.0009x"
  to "free".** The old sentence quoted one arm of a two-arm control as "the
  isolated number". The other arm moved 1.2 s the opposite way -- morton-on
  *faster*, which it cannot be -- and both deltas sit at or under this bench's
  measured run-to-run spread on one run per arm. **Opposite signs across a
  control pair means no effect**, which is the argument the same section
  already makes about a VRAM figure three paragraphs later, unapplied to time.
  Found by a second reader. The ratio column is deleted from the arm table:
  a figure printed to four decimal places reads as a measurement whatever
  caveat sits beside it.

- **Two "gaps" against Sol-Engine were miscategorised and are corrected.**
  `thresh_type` was written as a knob our node fails to expose; comfy-kitchen's
  CUDA kernel implements only the `diag` threshold math, so it is an
  unimplemented kernel path, and NVLabs' own single-consumer-card H3 profile
  runs `diag` too -- the mode we already run. `kv_splits` was written as a gap;
  Sol-Engine raises `"kv_splits=2/4 is currently available on SM90 only"`, so a
  4090 gets 1 there as well. Also corrected: "exact thresholding for H3" is
  A100/H100 only, and the A100 cell they call their validated policy runs the
  **Triton reference**, not a CuTe kernel.

- **`h3_config.py`'s Morton comment contradicted itself**, arguing three
  paragraphs below its own retraction that a quality gain would mean "the 1.16x
  it costs buys something". There is no cost to buy anything with.

### Added

- **`bench/check_doc_links.py`**, and `docs/checks.md` grows a
  `doc-link-absent` ledger for citations that are deliberately unresolvable.
  Checks both relative markdown links between docs and `path:line` code
  citations. It exists because the rename above broke `CLAUDE.md:41` within an
  hour of a CLAUDE.md rewrite whose whole purpose was removing stale
  references, and nothing in the repo noticed.

  **It found a live one on its first run.** `CLAUDE.md` cited
  `nodes.py:2245-2250` meaning ComfyUI's 2595-line file; it resolved to our
  194-line one -- inside the paragraph warning that `import nodes` finds ours.
  Citations into ComfyUI's tree now carry a `ComfyUI/` prefix. Thirteen further
  citations were bare basenames and are now paths.

  Shown red five ways before being trusted, including one attempt that appeared
  to prove it inert and was a wrong grep pattern in the control rather than a
  hole in the check.

- **Two card-free decisions in `docs/roadmap.md`**: whether to adopt
  `dense_blocks="0-1"` (every published H3 profile runs the first two blocks
  dense; ~3.6% by arithmetic, and it needs no block probe because copying a
  validated list is not the same question as choosing our own), and whether to
  build NVLabs' new sm89 CuTe kernel for a cross-check against comfy-kitchen's.

## 0.24.0

### Changed

- **Image graphs are foldered by use case: `workflows/image/`.** Video is the
  primary case and stays at the root. The routing is *derived* --
  `_graph_dir()` sends a graph there when its `GRAPHS` entry sets
  `single_frame=True`, which is exactly what makes it an image graph -- so
  there is no `image=True` flag to fall out of sync with reality.

  **The reading side needed real work, and the reason is a demonstrated
  failure rather than a predicted one.** Six checks walked graphs with a bare
  `workflows/*.json`, which is non-recursive. Run against a `GRAPH_DIRS` that
  had not learned about the new folder, `check_ref_prompt_labels` and
  `check_prompt_guide_conformance` **both exited 0 while covering 20 ref graphs
  instead of 28** -- no error, no warning, just a smaller number on a line
  nobody has a prior for. Discovery is now `h3_config.graph_paths()` in one
  place, and `check_ref_prompt_labels` carries a case that compares the
  discovered set against what is on disk, shown red by reverting `GRAPH_DIRS`.

- **The single-frame prompts are in the guide's structure, reversing a
  decision this file never recorded.** It lived only in
  `_image_edit_prompt()`'s docstring and in the waiver comment in
  `check_prompt_guide_conformance.py`, both of which argued at length that the
  guide could not apply to a still. Both are rewritten rather than deleted. What changed is
  evidence: the r/StableDiffusion author this path follows published a second
  prompt set on 2026-08-15, and **between their two posts they switched from
  flat prose to the guide's structure** with the audio sections dropped, after
  rendering a couple of thousand images. Neither post held the scene or the
  references fixed, so it is a revealed preference and not a measurement --
  which is why this ships as a ladder rather than a rewrite.

  `_image_prompt(scene, fmt)` replaces it, over six scenes drawn from both
  write-ups and three formats: `av` (all six sections, audio ones `N/A`),
  `sections` (the four visual ones, **the default**) and `flat`. Content is
  written once per scene and rendered into all three, so the arms cannot differ
  in wording -- which is exactly what the two Reddit posts do differ in, and
  why they cannot answer this. `docs/open_experiments.md` #16f states the
  question and what would settle it.

  The half of the old argument that survives: the audio sections describe a
  track a single-frame graph structurally cannot produce. Hence `sections` is
  the default and `av` is the arm.

- **`check_prompt_guide_conformance` grades image graphs properly now.** The
  blanket `h3_image_edit` waiver became a structural rule --
  `overall_soundscape` and `non_diegetic_music` are not required of a graph
  with no `VAEDecodeAudio`, read off the graph rather than off a name. The
  whole-file waiver had been buying silence on four cases to excuse two. Its
  "six sections, in order" case now **actually checks order**: it compared
  against a list built by iterating the guide's own sections, so it could only
  ever detect a missing section.

- **Both validators accept `LoadImage` paths carrying a subfolder.**
  `LoadImage` builds its combo from a non-recursive `os.listdir`
  (`ComfyUI/nodes.py::LoadImage.INPUT_TYPES`) but validates with `VALIDATE_INPUTS` ->
  `folder_paths.exists_annotated_filepath`, and the executor skips its own
  combo check for any input a node validates itself. So `h3_refs/face_x.png`
  renders, and `validate_api` plus `check_workflow_schema` were **rejecting
  graphs the server accepts** -- the mirror of the 2026-08-13 bug where the
  validator accepted graphs the server rejected. Bare filenames are still
  checked. The UI cost is real and documented: the dropdown will not offer the
  value, though the graph renders.

### Added

- **Eight graphs in `workflows/image/`**, replacing the single
  `h3_image_edit.json`. Six scenes, each exercising a different retention
  marker -- camera move, style transfer, environment composite, two-person
  composition, selective recolor, character sheet -- plus two probes rendering
  the `style` scene in the other two prompt formats. All name documented
  `h3_refs/` assets instead of the input-root placeholders, so a result is
  attributable to a subject somebody can look up.

- **`docs/h3_image_editing.md`** -- the experimental image use case in one
  place: the layout rule and why it needed a check, the format ladder, the six
  scenes and what fails first in each, and what is still unsettled.

- **Reference images up to three per graph.** `build_api`/`build_ui` take
  `ref_images=(...)`, which names the files and sets the count from its own
  length. Node ids stay fixed per slot (15/24, 16/25, 34/35) because two
  benches address the first pair by name. Three is the ceiling: the UI node's
  socket list declares `ref_image_0..2` and that list is positional in every
  saved graph, so a fourth has to be appended, never inserted.

### Measured

- **`allow_upscale=False` is now the default for every image graph**, and it
  was confirmed here rather than inherited. `h3_image_style`, two references,
  same seed: **89.1s with the fit upscale against 18.1s without**, a 4.9x
  saving. The pair was compared against the source reference rather than
  against each other -- same identity, freckle pattern, head angle, expression
  and hairstyle, graphite medium transferring in both, and in neither did the
  style reference drag its own cottage in. That reproduces #16e's 84s/18s
  ladder on a second subject and seed, which is why the default moved here
  where #16e had declined to move it. The whole eight-graph set now renders in
  about **two minutes against about eleven**. Video graphs are untouched.

- **`steps` stays at 16, and the reason is the useful part.** 16 against 8, one
  paired render per scene: `h3_image_edit` (1 ref) 13.0s vs 4.0s and
  indistinguishable; `h3_image_style` (2 refs) 18.0s vs 7.0s with freckling and
  medium both holding; **`h3_image_multiperson` (3 refs) 25.0s vs 10.0s, where
  8 steps loses the woman's freckling and her pendant** -- precisely the detail
  that scene's `partially_preserved` entry names as retained. ~15s bought on
  the one graph where the detail is the point.

  **Measured only on the one-reference portrait, 8 steps looks free
  everywhere.** That is this repo's own trap -- a check whose input already
  satisfies the expected outcome cannot fail -- and it is the reason the ladder
  was run on the hard scenes before touching a default. Recorded as #16g.

- **The `attribute_transfer` role binds.** No cottage appeared in any style-scene
  render, at any step count or sizing. First evidence here that telling a
  reference what it does *not* supply does something; still uncontrolled, since
  no arm omitted the negative clause.

- **The three prompt-format arms rendered, and none of them failed.** Same
  scene, same two references, same seed, one pass. `av` (six sections) differs
  from `sections` (four) by a grey-scale mean of **3.45**/255 -- against an
  h264 round-trip floor of ~1.6 -- for 15 extra prompt tokens. `flat` (one
  paragraph) differs from both by **~44.6**, a materially different picture.
  **No cottage appeared in any arm**: the `attribute_transfer` role bound with
  and without the section scaffolding, and identity, freckling and the graphite
  medium held in all three.

  A negative result, recorded as one. It is not "format does not matter": n=1
  per arm on one scene at one seed, and the `flat` arm keeps the negative
  clause because content is held fixed across formats, so it is not the bare
  community-style prompt. `docs/open_experiments.md` #16f names the three arms
  that would discriminate -- drop the negative clause, run the ladder on the
  three-reference scene, repeat at more seeds.

### Known wrong

- **`crop` cannot be retained on a 1:1 reference.** `h3_image_style` and
  `h3_image_recolor` both claim the crop is preserved; both references are
  1024x1024 against a 768x1152 canvas, so the model widens the frame because it
  has to. The claim is unachievable as written. Not fixed -- it needs either a
  square canvas for those scenes or the word dropped.

### Not done

- **The three format arms were not rendered.** They are the point of #16f and
  remain unjudged; the renders above were for cost, on `h3_image_style`'s
  shipped `sections` form only.
- **`smoke_h3.py` still not run.** Every graph in
  `workflows/image/` is schema-valid against a live `/object_info` and
  unsubmitted, and `smoke_h3.py` was not run -- so by this repo's own standard
  they are unverified, including the three format arms whose whole purpose is
  to be looked at.

## 0.23.0

### Removed

- **The Sol-Attn Triton pack is deleted**, not moved. `coderef/ComfyUI-SolAttn_triton/`
  is gone; `SolAttnPatch` and `SolAttnBlockProbe` are absent from a live
  `/object_info` after a restart. Recoverable from
  `github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea`, so this is a lost
  *tool*, not lost work. **This supersedes 0.21.0**, which argued for keeping
  it and was written about an hour earlier; that entry is annotated rather than
  rewritten.

  **Dependency 1 resolved before deletion, which is what made it safe.**
  `bench/check_solattn_correctness.py` grades the CUDA kernel only. The old
  coupling -- load Triton first, `return 2` on failure, before the CUDA arm --
  was an accident of control flow, not of method. Red control shown red against
  a copy; green with the pack absent.

  **Dependency 2 NOT resolved, and it is a real cost.** `SolAttnBlockProbe` had
  no CUDA equivalent and is now gone from the tree. So
  `SOL_ARTIFACT_INSURANCE = dict(tau=1.3, dense_blocks="33-35,39-42")` is a
  guess whose validating probe run is blocked on an instrument that no longer
  exists here, and `dense_blocks` -- the stated fix for the object-dissolve
  artifact -- currently cannot be chosen from measurement. `sol_block_probe.py`
  scaffolds the replacement and **implements none of it**. Caught by the peer
  session reading 0.21.0 against the tree; the two documents disagreed for
  about an hour and both told a reader to look in a directory that was not
  there.

### Fixed

- **`docs/SOLATTN.md` said the Triton pack was "kept, not retired" and pointed
  at `coderef/`** in three places -- the header, the backend table's `pack`
  row, and a source citation. All three now name the deletion and cite upstream
  at `842c4ea`. The "Its status" section is rewritten around what the deletion
  cost rather than around why it should not happen.
- **`~1,700` blocks at 362 frames was labelled as production scale** in
  `bench/analyze_sol_error.py` and `check_solattn_correctness.py`'s output. It
  is `1,626 x 362/345` rounded -- an arithmetic rescale of a measurement,
  printed where a measurement goes. Now marked DERIVED with the derivation
  inline, and the one real measurement named with the length it was taken at.
  Also caught by the peer session.

## 0.22.0

### Changed

- **The max length is 362 frames (15.083s), and every "345 is the largest
  legal count" claim is withdrawn.** Owner decision, 2026-08-16. 345 is the
  largest count the *reference pipeline* will emit -- diffusers' `max_duration`
  is a hard-coded 15.0s applied after the 17n+5 snap -- which is a fact about
  diffusers and was never a limit on the model. This repo presented it as
  legality for a week, called 362 "illegal / out of distribution" on
  2026-08-14, and withdrew a whole bench run over it.

  `h3_rules.MAX_LENGTH = 362` is now the ceiling and `MAX_DURATION` derives
  from it. Frames first, seconds derived: a seconds-first ceiling of 15.0
  excludes its own maximum by 0.083s, which is exactly how 345 became the
  answer. `duration_in_range()` is the model's window; the new
  `reference_would_emit()` is the separate portability question, and callers
  that care about diffusers must now ask for it by name.

- **`LONG_LENGTH` is 362**, reverting the 2026-08-10 move to 345. All 73
  graphs regenerated against a live `/object_info`; no graph carries 345 as a
  length any more. Measurements taken between 2026-08-10 and 2026-08-16 were
  taken at 345 and now sit one grid step below the shipped default; everything
  older is back on the length it was measured at. The 5% difference should not
  move a ratio, but that was not re-checked in either direction.

- **What 362 rests on, recorded rather than implied.** One upstream statement
  from 2026-08-14 (`6e85e48`) with no artifact attached, plus LightX2V shipping
  a 362-frame config. MiniMax's own README gives a rounded "4-15 seconds" and
  the official checkpoint configs state no frame limit at all, so the primary
  source neither confirms nor refutes it. A decision taken on thin evidence and
  labelled as one; not a measurement.

### Removed

- **`REF_VIDEO_LENGTH` is deleted and must not be reintroduced.** A safe length
  for a reference arm is not a constant: it depends on how many references are
  wired, their kinds and durations, the canvas, and whether they are upscaled,
  so one number is only right for the configuration it was measured on and
  silently wrong everywhere else. Benches and tests now pick the duration the
  test calls for. `REF_VIDEO_BUDGET` takes `LONG_LENGTH`.

  The measurement it came from is kept as a warning, because the ceiling is
  real: at 345 frames, 1024x768, references not upscaled, the arm peaked at
  **22,735 MiB of 24,564** over 34.3 minutes. That is 1,829 MiB of headroom,
  and 362 is ~5% more tokens -- **expect these arms to sit at or over the
  edge.** Read preflight first, and if one OOMs, shorten that run rather than
  reaching for a new constant.

### Fixed

- `bench/check_keyframe_canvas.py` pins the flip: 346 and 362 are accepted
  where they were refused, and 363 (which snaps to 379) is refused. Runs green.
- `docs/evidence.md` asserted "**345 legal / 362 illegal**" under `## These
  hold` -- the section reserved for claims a second reader confirmed -- eleven
  lines below the correction withdrawing exactly that claim. The 2026-08-14
  commit said it had fixed every consumer; it added the correction and left the
  bullet. Removed.

## 0.21.0

### Changed

- **The Sol-Attn Triton pack moved out of `custom_nodes/` into
  `coderef/ComfyUI-SolAttn_triton/`.** ComfyUI no longer registers
  `SolAttnPatch` or `SolAttnBlockProbe`, so neither can be wired by accident.
  `SolAttnMiniMax` (CUDA) is what every graph wires and what the owner runs in
  everything; no graph has referenced the Triton node since 2026-08-14 and
  `check_sol_kernel.py`'s `no_triton_graphs` case enforces it.

  **SUPERSEDED THE SAME DAY by 0.23.0: the pack was deleted, not kept.**
  The paragraph below is left as written because one of its two legs was
  resolved and the other was accepted as a cost, and deleting it would hide
  which. Read 0.23.0 before acting on any of it.

  **Moved, not deleted, and the distinction is the finding.** Audited before
  moving: at *runtime* the pack is dead, but as *tooling* it has two live
  dependencies. `bench/check_solattn_correctness.py` **hard-requires** it --
  it loads the Triton kernels first and returns 2 on failure, before its CUDA
  arm is reached, so deleting the pack would turn the only independent
  correctness check on the CUDA kernel into a permanent skip. And
  `SolAttnBlockProbe` has no CUDA equivalent; it is the instrument for choosing
  `dense_blocks`, and `SOL_ARTIFACT_INSURANCE` sits unwired pending a probe run.
  Only the third role -- `--sol-backend triton` for pre-2026-08-14 numbers --
  is purely historical.

### Fixed

- **`bench/check_solattn_correctness.py` now searches both locations** for the
  pack, `coderef/` first so a stale `custom_nodes/` copy cannot shadow the
  maintained one, and raises a named error pointing at `docs/SOLATTN.md` rather
  than degrading to a silent exit 2.
- **`provenance.py` stamped the wrong Sol build, and omitted three knobs that
  actually run.** `builds.sol_attn` recorded the *Triton* pack's git HEAD -- a
  pack no graph has wired since 2026-08-14 -- while saying nothing about the
  kernel that did run. It is now `builds.sol_attn_cuda`, the installed
  `comfy_kitchen` version plus whether `sol_attn` is present at all, which is
  the only field that can tell the fork build from the stock wheel (both
  declare `0.2.31`). Separately, `SOL_CLOSURE_KEYS` was still written in the
  Triton vocabulary: it asked for `int8_qk`, `use_tma` and `int8_pv`, which do
  not exist on the CUDA node and so recorded "not detected" on every render
  forever, and it **omitted `routed_cap_percent`, `centroid_tail` and
  `reuse_qkv_memory`**, which do run. `centroid_tail` is the one that stings --
  it has a live A/B with a deadline and no stamped render says how it was set.
  Corrected against `vendor/sol_attn_minimax.py:497-501`.
  `STAMP_SCHEMA_VERSION` 1 -> 2.

### Notes

- **A stale external write reverted `CHANGELOG.md` on disk** after `01db3c9`,
  dropping both this session's 0.19.0 "Retracted" section and the peer
  session's 0.20.0 entry, and reinstating a withdrawn figure.
  `check_retraction_consumers.py` caught it -- the phrase reappearing in a
  file not on its allowlist is exactly what that check is for. Recovered with
  `git checkout HEAD -- CHANGELOG.md` after confirming the working copy
  contained no new content. Same shape as the incident `docs/evidence.md`
  already records for the device poller.

## 0.20.0

### Added

- **`bench/check_lora_alpha.py`**, covering a failure this repo could not have
  seen: a LoRA whose scale ComfyUI cannot read. ComfyUI takes `alpha` only from
  a `"<module>.alpha"` tensor (`comfy/lora.py:41-45`) and falls back to 1.0
  (`comfy/weight_adapter/lora.py:248-251`); it never reads the file's
  `__metadata__`. diffusers' #14408 (2026-08-14) documents a published MiniMax-H3
  turbo LoRA that records alpha 8 against rank 128 in metadata alone, which such
  a loader applies **16x too strong**, silently, on a graph that validates and
  renders. **The first check here whose subject is a third-party binary** rather
  than our graphs, our nodes, or a dependency's API. Also asserts every `*_LORA`
  in `h3_config.py` resolves on disk -- on the morning it was written `REF_LORA`
  did not, so the ref-LoRA graph could not run and nothing said so.

  The exemption carried the work: three kijai `_resized_avg_` conversions in
  this install carry a metadata `alpha` and are **correct**, because they also
  declare `baked_scale` and fold the scale into `lora_B`. A rule keyed on the
  naive shape would fail three good files on every run. A declared bake outranks
  a declared alpha, and that precedence is asserted by a control rather than
  trusted. Both controls are synthetic and run every invocation, because every
  real file here is clean and an all-clean corpus cannot distinguish a working
  check from an inert one. It also pins its own premise by reading
  `comfy/lora.py`, so it goes red rather than quiet the day that loader grows a
  metadata channel.

  Shown red 2026-08-16, five mutations, all against copies: a dangling
  `REF_LORA`; the unsafe branch deleted; the exemption widened by presence; the
  exemption removed (three correct files misclassified); and a fake ComfyUI that
  reads `__metadata__`. A sixth mutation moved no verdict and is recorded in-file
  as proving nothing -- **the mutation that disproved a sentence in the
  docstring**, which claimed `control:baked` defends against a widened exemption.
  Measurement says `control:unsafe` does, and `control:baked` defends the
  opposite direction.

### Changed

- **`docs/checks.md`** indexes the new check and its count moves 18 -> 19.

## 0.19.0

### Added

- **`docs/open_experiments.md` #17: a 16-bit PV branch for the CUDA Sol-Attn
  kernel.** Scoping pass, no code. Three things it establishes. First, the
  scope is one matmul, not a rewrite: sage's `fp16 (most accurate)` is
  `qk_int8_sv_f16` -- INT8 QK, 16-bit PV -- so the Sol equivalent is moving
  `mma_u8s8` to `mma_bf16` and nothing else. Second, **it has already been
  measured, on Triton**: `bench_e2e_h3.py:475` records that the Triton exact
  branch is 16-bit by default, so `sol, no int8` (827.9 s, 1.20x over sage)
  already prices a stronger version of this change against the all-INT8
  714.9 s. Third, the fragment layout is a solved problem in-tree --
  `sol_attn_route.cu:446-465` already runs bf16 PV, and `perm_key` exists only
  to make the INT8 repack free.
- **`bench/mma_rate.cu`**, a tensor-core MMA issue-rate microbenchmark. Not a
  check and not wired to one; it needs `nvcc`, which nothing else in `bench/`
  does. It exists so #17's cost numbers have a reproduction path. Measured on
  this 4090: int8 `m16n8k32` 334.5 TMAC/s against bf16/f16 `m16n8k16` f32-accum
  at 83.8 (0.25x) and f16-accum at 167.3 (0.50x). **Confirms on sm_89 what
  `sol_layout.cuh:81` claims for sm_120** -- f32-accumulate forms issue at half
  rate. A cuBLAS GEMM was tried first and rejected as the instrument:
  `torch._int_mm` reaches ~142 TOPS of int8's ~660 peak while bf16 `matmul`
  sits at its own peak, which would have inverted the conclusion.

### Retracted

- **Every fp8-vs-fp16 sage accuracy ratio, withdrawn by the owner as untrusted
  and deleted rather than caveated.** Removed from `CLAUDE.md`,
  `workflows/h3_config.py`, `docs/SOLATTN.md`, `docs/evidence.md`,
  `vendor/UPSTREAM.md`, `h3_capture.py`, `README.md` and
  `docs/open_experiments.md`, along with the `mean_rtol` values behind them.
  Ruled out on provenance, not on size: the sweep is `torch.randn`, which is
  not the input distribution H3 has; the competing real-activation figure was
  never re-derived here and its script is not committed in the sage fork, so it
  was an uncommitted ad-hoc run cited across a repo boundary; and nothing in
  `bench/` uses captured activations, so re-running today would reproduce the
  synthetic instrument rather than replace it. **The decision to ship
  `fp16 (most accurate)` is unaffected** -- it rests on the owner's perceptual
  verdict, which never depended on a ratio. `docs/evidence.md` keeps one
  deliberate spelling of the phrase so `check_retraction_consumers.py` has a
  tripwire; that check now fails if the number reappears anywhere else.
- **Not swept:** `attention.py` and `README.md` keep a "2.7x" that is a *speed*
  figure against torch's flash backend, and `vendor/sol_attn_minimax.py` has a
  "2.0 ~ 2.7%" routing density. Different claims, same spelling.

### Notes

- #17's cost estimate is now MMA arithmetic alone, predicting 2.5x on the exact
  branch. The Triton figures it was first written against **do not price this
  change** -- corrected the same day. That backend dispatches two different
  kernels rather than toggling a dtype, so its bf16-vs-int8 arms vary the PV
  dtype, the QK dtype and the implementation at once; the within-kernel
  isolation has never been run. Using Triton numbers to describe the CUDA
  backend is also the move `docs/morton.md` retracted on 2026-08-16.

## 0.18.0

### Changed

- **`SOL_RECOMMENDED_CUDA`'s `morton_curve` is now `3d`**, was `2d_frame`.
  This changes nothing about what renders today, because `morton` ships off;
  it changes which curve you get if you turn it on. On captured activations
  `3d` beats `2d_frame` on per-block centroid fidelity at every depth sampled,
  and all three curves measured speed-identical, so the switch was wired to the
  weakest option for no reason. The `FRAME_PER_TOKEN` argument for `2d_frame`
  is mechanically correct and is not refuted by this; it simply does not win.

### Retracted

- **"morton is worth 1.16x alone, at 94% GPU utilisation"**, for the CUDA
  backend. Isolated properly on 2026-08-16 -- all 50 blocks dense, morton on
  against morton off, so the permutation is the only difference -- it costs
  **0.8 s of 861, or 1.0009x**. In the sparse arm, 1.2 s of 454. The
  permutation is free at 1344x768 / 294 frames. The old figure was Triton, 362
  frames, stacked on int8; correct for what it measured, wrong as a
  description of this backend. `morton` stays off, now because nothing has
  shown it changes the output rather than because it costs anything.
- **A 3.7 GB peak-VRAM saving from Morton**, before it was ever written down.
  It appeared consistently across all three curves in the sparse arms. The
  dense control killed it: Morton saves 3,706 MiB sparse and *costs* 2,144 MiB
  dense. Opposite signs, so not a Morton effect. Recorded because the control
  is the only reason it did not ship as a finding.

### Measured

- **Sol-Attn is worth 1.896x on the sampler** at 1344x768 / 294 frames, 860.8 s
  dense against 454.0 s sparse, same config with only `dense_blocks` changed.
  Cleaner than the retracted 1.611x, which compared against an fp8 sage
  baseline nobody ships. One run per arm.
- **The three token orderings are indistinguishable on speed**, 452.8 to
  454.8 s across a 2 s spread. The choice between them rests entirely on the
  activation measurements, not on cost.

### Added

- **`sol_curves.py` and `MiniMaxH3SolAttnCurve`** -- a per-frame 2D Hilbert
  token ordering for Sol-Attn, in the shape `morton_perm` already returns.
  Installed by rebinding that one name on the live node: `_perm_for` resolves
  it as a plain module global and the curve arrives as a string through
  `transformer_options`, so a new ordering needs no edit to upstream's file and
  no kernel rebuild. `install()` resolves the module **by identity, not by
  name**, because a running ComfyUI can hold two objects for one file and
  patching the wrong one looks exactly like success; zero patched is treated as
  a failure rather than a silent no-op. The node must sit after
  `SolAttnMiniMax`, which it overwrites a transformer option of.
- **`bench/analyze_capture.py`** -- answers whether Morton helps the *router*,
  from captured q/k rather than from geometry. Two tests, neither
  reimplementing the kernel: per-block centroid fidelity (the assumption
  everything else rests on) and mass concentration (upstream's stated
  mechanism). Because a dense capture is permutation-equivariant, one render
  gives every ordering exactly, at every block.
- **`docs/sol_engine_reference.md`** -- NVLabs' own Sol-Engine recipe for H3,
  read from `coderef/Sana` at `origin/sol-engine`. Their validated policy, how
  ours differs, FirstBlockCache, the `thresh_type` knob kijai's kernel does not
  have, and why their published speedups share no denominator with ours.
- **Four ordering arms plus the reorder-only control** in `bench_e2e_h3.py`.
  The control needed no code: `--arms 'shipped[morton=1,dense_blocks=0-49]'`
  already expresses it. Copied from Sol-Engine's own
  `config/wan21_t2v_14b/reorder_only.toml`; reordering on with every layer
  dense isolates what the permutation costs from what it buys, and checks that
  it really is output-neutral.

### Measured

- **Link 5 holds.** First ever run of `h3_capture.py`. On captured activations
  from a dense 124-frame 1344x768 render, Morton raises per-block centroid
  fidelity at every depth sampled, so the canvas result is about something
  real. Blocks 24 / 49, against raster 0.7442 / 0.8678: `2d_frame` 0.7665 /
  0.8804, `3d` 0.7915 / 0.9434, `hilbert` 0.7748 / 0.8978.
- **Both added curves beat the shipped `2d_frame` on both metrics, and do not
  beat each other.** `3d` leads centroid fidelity, `hilbert` leads mass
  concentration (142.1 / 226.8 blocks for 90% of mass against `2d_frame`'s
  149.3 / 228.9), and `hilbert` has the higher floor at block 49. The metrics
  disagree, so both are arms and neither is a new default.
- **Attention is not very sparse on this workload.** At its most concentrated a
  query still needs 178 of 591 blocks for 90% of its mass, and 394 at block 0.
  That bounds what any block-sparse method can save here.
- **1280x768 at 294 frames peaks at 23,192 MiB of 24,564** on plain t2v with no
  references. Tighter than expected, and it constrains every reference plan.

### Fixed

- `analyze_capture.py` restricted itself to block 0 on the grounds that later
  blocks diverge. Wrong when the capture is dense: attention is
  permutation-equivariant, so the Morton arm's q/k is the permuted capture at
  every block. The restriction was self-imposed and it mattered, because block
  0 is the worst place to ask -- early-layer attention is closest to uniform.
- `MiniMaxH3SolAttnCurve.execute()` had no default where its schema declared
  one, so an API graph omitting the widget would have raised instead of
  defaulting. Caught by `check_schema_defaults.py`, which now covers 8 nodes.

### Corrected

- **Sol-Engine does ship Morton.** This repo said three times on 2026-08-15
  that no token reordering exists upstream. It exists, is on by default for
  Wan, off for Hunyuan, absent for H3, and is only ever the 3D curve. The
  claim was made from a grep whose non-empty output was misread as empty.
- **The "same quality at higher sparsity" line is not the paper's.** It is one
  sentence in the Triton pack's *Wan* Morton file. The paper (arXiv 2607.24027)
  contains no token reordering at all, does not evaluate H3, and does not
  ablate the threshold. Sol-Engine's H3 policy states no reordering explicitly.

## 0.17.0

### Measured

- **Three distinct identities compose in one single-frame edit; the card is
  what breaks first.** 13 arms through the new `bench/bench_image_edit_refs.py`,
  drawing from the `h3_refs/` library rather than re-rendering one subject:
  1 reference 9,135 rows / 42s, 2 refs 17,352 / 78s, 3 refs 32,093 / 198s,
  4 refs 40,294 / 280s (two identities, no blending, the right garment on the
  right person), 6 refs 56,710 / 491s (three identities, all correct), and
  9 refs **OOM on a 24 GB 4090**.

- **Reference images are paid for TWICE, which the cost model here missed.**
  Each one enters as Qwen vision tokens in the `text` segment AND as latent
  rows in the reference segment, and the text half scales with reference count,
  landing 75-160 rows *above* the reference half at every rung of the ladder.
  So a "44% references / 46% text" reading of Preflight was wrong: the prompt is
  under 200 tokens and the references are ~89% of the sequence. Ladder recorded
  in `docs/h3_references.md`; nine references is ~94k rows, more than the
  124-frame video graph asks for.

- **`ref_image_size="max"` is a no-op on the shipped image graph.**
  `MiniMaxH3ReferenceFit` has already taken the reference to 2048 short edge and
  core's `max` is `min(1.0, 2048 / short_edge)`. The real lever is the fit
  node's `allow_upscale`: 8,192 reference rows and 84s with it, 2,048 and 18s
  without, 1,682 and 16s under `match`. At 1:1 on the face all three hold the
  same identity. One subject, one seed -- not enough to move the default, and
  recorded as such in `docs/open_experiments.md` #16e -- but it is the first
  evidence either way and it points at the shipped default costing 5x for
  nothing.

### Added

- **`bench/bench_image_edit_refs.py`** -- the sweep above. Refuses to run
  against a server whose length floor is 5 rather than quietly producing a table
  about 5-frame renders.

## 0.16.0

### Added

- **`single_frame.py` -- H3 as a single-image edit model, by lifting ComfyUI's
  5-frame floor in memory.** ComfyUI's H3 nodes floor `length` at 5 and
  `temporal_shape` clamps with `max(5, length)`, so one frame is unreachable;
  H3 is a capable reference-driven image editor at exactly one frame. This is
  **a temporary patch to somebody else's module and is meant to be deleted**
  when ComfyUI ships the same change (Comfy-Org/ComfyUI#15644). It logs a
  multi-line banner saying so at every startup, retires itself automatically
  the moment upstream reports a floor of 1, and can be disabled outright with
  `H3_EXPLORATIONS_NO_SINGLE_FRAME=1`.

  **Only `length=1` changes, and that is verified at load rather than argued.**
  The two grid functions delegate to upstream's own body for every count above
  1, and the one function that must be rewritten (`temporal_shape`, because the
  clamp is inside it) is compared against the original across the node's entire
  declared range before anything is installed. A single disagreement anywhere
  and nothing is patched. The interesting region is 2-4: those were never
  reachable as themselves because the clamp rewrote them to 5, and they still
  snap to 5.

  Targets are resolved through `NODE_CLASS_MAPPINGS` and by file identity, not
  by module name, because there are normally **two** copies of the module in a
  running ComfyUI: `load_custom_node` registers `comfy_extras` files under a
  file-path module name, `comfy_extras/` has no `__init__.py`, and a dotted
  `import comfy_extras.nodes_minimax_h3` (which `resolution.py` and
  `preflight.py` both do) builds a second, independent one. Patching only the
  dotted copy left the server serving and executing an unpatched floor of 5
  while the shim logged success -- found by asking the live `/object_info`,
  invisible to an in-process check.

- **`workflows/h3_image_edit.json`** (and its API form) -- one reference image
  plus text to ONE edited image. ref2va, `length=1`, the single-image H3 VAE,
  `SaveImage`, and no audio decoder. Carries a note explaining what it rests
  on, and where it deliberately departs from the community workflow it follows
  (in-family 768x1152 rather than 1024x1536, sage fp16 rather than Comfy
  Kitchen attention, base checkpoint rather than a hybrid plus two LoRAs).

- **`bench/check_single_frame.py`** -- 11 cases over the shim, including a
  CONTROL that hands `apply()` a deliberately wrong implementation and asserts
  it refuses, and the first LIVE case in `bench/`: it asks the running server
  what floor it reports. Shown red two ways on 2026-08-15.

- **`h3_rules.is_single_frame()` and `SINGLE_FRAME`** -- one frame is the only
  exception to the 17n+5 grid, and none of the duration rules apply to it.

### Changed

- **`h3_rules.snap_length()` returns 1 for a length of 1**, matching core's
  post-shim `align_frame_count`. It clamped to 5 before, exactly as core did;
  the two moved together deliberately.
- **`MiniMaxH3Resolution` accepts `length=1`** (`min` 5 -> 1) and reports it as
  single-frame mode rather than warning that 0.042s is outside the 5-15s
  window. A check going red on correct state is worse than no check.
- **`MiniMaxH3Preflight` reports 1 frame at `latent_t == 1`.** It derived
  frames from the latent and returned 5 for anything below 3 temporal steps,
  which was right for every latent that could exist before this release and is
  wrong for the one the image path uses.
- **`MiniMaxH3KeyframeCanvas` refuses `length=1` explicitly**, naming the
  reason: `MiniMaxH3ImageToVideo` pins a `last_frame` at `frame_count - 1`,
  which in a one-frame video is frame 0, and fl2va at one frame is unmeasured.
  It refused before too, with a misleading message about seconds.

### Measured

- **The single-image VAE is worth 15.2 dB, and its encoder is byte-identical
  to the stock one.** Of 562 tensors, 121 match the Comfy-Org video VAE
  exactly -- all 116 encoder tensors, `quant_conv`, and the latent statistics
  -- while the 441 that differ are 439 decoder tensors plus both
  `post_quant_conv`. So the latent space is untouched and swapping VAEs is a
  pure decoder swap. Round-tripping a source image at `T=1`, where ground truth
  exists: **37.27 dB / SSIM 0.947** for the image VAE against **22.04 dB /
  0.821** for the stock video VAE fp16. Core decodes `T=1` with either -- the
  video VAE does not fail, it returns a harsher, colour-shifted image, which is
  the trap.
- **The reported grid artifact reproduces and is now quantified.** Decoding a
  5-frame latent with the image VAE and keeping frame 0 leaves gradient energy
  aligned to the patch grid at 1.46x (16px) and 1.50x (32px) the off-grid
  average, against 1.03-1.22x for every other combination tried.
- **At one frame the canvas stops being the cost lever.** Preflight on the
  shipped image graph: 9,240 rows total, of which text is 4,276 (46.3%),
  references 4,096 (44.3%) and video only 864 (9.4%). Changing aspect ratio
  moves 3% or less, where in a 124-frame render it is the largest single lever.
- **Core has anticipated single-frame decode all along.**
  `comfy/ldm/minimax/vae.py` branches on `z.shape[2] == 1` and returns the LAST
  of the 4 frames one latent decodes to, which is exactly the
  `h3_t1_output_slice: 3` the image VAE's metadata declares.

## 0.15.0

### Changed

- **Default sampler is now `er_sde`**, replacing `res_multistep`. Owner's
  call; the old value was core's base-template default carried unquestioned.
  One eval per step either way, so it is step-cost-neutral. It is stochastic
  (`s_noise` 1.0), the first such default here, which matters for reading an
  A/B more than for speed. The scheduler stays `simple`: `beta` was tried the
  same day and reverted because it moves two of sixteen steps out of
  Sol-Attn's sparse window (11 sparse / 5 dense under `simple`, 9 / 7 under
  `beta`) with no benefit measured against that. Reasoning lives at `SAMPLING`
  in `workflows/h3_config.py` and nowhere else, same as every other default.
- **`bench/bench_e2e_h3.py` now reads `sampler`/`scheduler` from
  `h3_config.SAMPLING`** instead of hardcoding them. This is the exact shape
  of the bug `check_bench_matches_shipped.py` exists for, one field over --
  that check pins the sage and Sol nodes and says nothing about the sampler,
  so the bench could have gone on sampling a schedule no graph ships. Derived
  rather than checked, so nothing has to notice.

### Added

- **`bench/analyze_morton.py`** -- what Morton reordering does to the
  64-token blocks the Sol-Attn router operates on, with no GPU, no model and
  no clip-watching. Morton cannot affect dense attention at all (attention is
  permutation-equivariant and the permutation is undone after the last block),
  so the only thing it can change is **which tokens share a block**, and that
  is pure arithmetic on the latent grid. Reports per-block frame span,
  bounding box, RMS radius from the block centroid, fill, and the share of a
  token's grid neighbours kept in-block; `--map` prints one latent frame as
  ASCII block ids. The shipped `morton_perm` is imported from
  `vendor/sol_attn_minimax.py` and cross-checked against an independently
  written implementation before any number prints.

  Two findings, both at 294 frames:

  **`morton_curve="2d_frame"` delivers whole 8x8 tiles on only 3 of the 48
  legal landscape canvases** -- 1280x768, 1024x768 and 768x768, the ones whose
  latent dims `(h//32, w//32)` are both multiples of 8. Morton codes tile a
  padded power-of-two space, so a latent grid like 1344x768's 24x42 keeps only
  the in-range corner of each tile. Measured: at 1024x768 a block is a solid
  8x8, fill 1.00, centroid radius 3.24. At **1344x768, the repo's default
  canvas**, a block is typically two disjoint 8x4 fragments in different parts
  of the frame plus a broken right-edge column -- fill 0.60, radius 5.54, and
  4.7% of blocks end up *looser* than raster order's worst block (radius 20.3
  against 16.1). Mean radius still beats raster's 12.12, so morton is not
  simply worse there; it is partial, and its tail is worse.

  **The start-offset rotation in `_perm_for` is load-bearing and correct.**
  Block boundaries are anchored at absolute row 0, so reference rows move
  where the video span's blocks fall; rotating the permutation by
  `(-video_start) % 64` makes block geometry invariant to `video_start`,
  verified at seven offsets. Without it, 1024x768 with references falls from
  fill 1.00 to 0.42. This was nearly recorded as the opposite finding --
  grouping tokens by `j // 64` instead of `(video_start + j) // 64` measures a
  partition that exists on no graph with references, and makes the rotation
  look like the cause of the damage it prevents. `block_ids()` carries the
  correction.

## 0.14.0

### Added

- **`bench/check_sol_kernel.py`** -- the first check here covering a call INTO
  a dependency we do not control. Asserts the installed `comfy_kitchen`
  carries `sol_attn`, that it is the CUDA backend rather than the eager
  reference alone, and that the signature still accepts the kwargs our node
  passes. Presence is gated on a graph wiring `SolAttnMiniMax`, because
  Sol-Attn ships OFF and "absent" is the expected state on any machine that
  has not built the fork; ungated it exits 2, not 0. Shown red against the
  stock PyPI wheel as the control. A second case pins `SOL_CUDA_DEFAULTS`
  against the inputs the node declares, parsed with `ast` so the check needs
  no ComfyUI -- upstream is weighing making `centroid_tail` unconditional,
  and a knob that gets *renamed* rather than removed would otherwise leave
  the pin silently not reaching it while the bench arm kept printing under the
  old name. Shown red by simulating exactly that rename.
- **`custom_nodes/ComfyUI-SolAttn-cuda/`** -- the CUDA node installed
  standalone, upstream's file kept byte-identical with a two-line
  `__init__.py` shim and a README recording provenance. Deliberately NOT
  vendored into this repo: the node id is provisional, so when upstream ships
  the real node this is a directory to delete rather than a graph migration.
  Verified through ComfyUI's own loader path.
- **CUDA arms in `check_solattn_correctness.py`**, grading
  `comfy_kitchen.sol_attn` against the same eager oracle as the Triton
  kernels, with its own red control. Exits 2 when the CUDA arm is skipped for
  cause.
- **`start_percent` sweep arms** in `bench_e2e_h3.py`
  (`shipped+start0.0/0.1/0.3/0.4`), derived from `SOL_RECOMMENDED` so they
  track tau rather than pinning it. It was the only knob in that config with
  no measured rationale -- 0.2 is the paper's number, carried through
  unexamined.
- **`--sol-backend {triton,cuda}`** in `bench_e2e_h3.py`, selecting which
  Sol-Attn node the sol arms build. `triton` stays the default so every
  recorded number stays comparable. The two nodes do not share a vocabulary,
  so `SOL_CUDA_DEFAULTS` is a separate dict in `h3_config.py` and the run
  **refuses** rather than silently dropping an orphaned knob -- otherwise
  `sage+sol+int8` under the CUDA backend becomes plain `sol` and still prints
  as an int8 result.
- **A token-floor warning.** Upstream reports Sol-Attn's gains are invisible
  below ~250-300 frames at 1344x768. `bench_e2e_h3.py` defaults to
  `--length 73` and the frontier table in `docs/SOLATTN.md` was measured at
  124 -- both far under. A Sol arm below the floor now says so, because a null
  result there reads as "this knob does nothing" rather than "this run could
  not have shown anything". Counted in **tokens**, not frames: video tokens
  are `latent_t * (w/32) * (h/32)`, so 250 frames is 72,576 tokens at
  1344x768 but only 44,928 at 832x768. The first version of the guard counted
  frames and would have passed that second run.

### Changed

- **`bench/_sol_attn_reference.py` re-vendored**, `ad9a4a8` -> `c04ef20`. The
  old vendor predated `centroid_tail` (default **True**), which shares one
  pooled tail per query block instead of computing it per row -- so the oracle
  did not contain the path a real render takes, and any correctness verdict
  from it described `centroid_tail=False` only.
- **`check_solattn_correctness.py` now measures which tail mode each kernel is
  on** and grades it against the matching reference, instead of assuming.
  Re-vendoring silently made every Triton arm cross-mode and **they all still
  passed**, because the two modes differ by cos 0.9988 and the bar is 0.998 --
  the bar was looser than a whole-branch change to the algorithm. Measured,
  not read off the kernel: Triton is per-row, the CUDA kernel defaults to
  centroid. Graded in matched modes the two backends' arithmetic is
  equivalent (cuda 0.999919, triton int8 0.999885); the naive cross-mode
  comparison invents a CUDA quality win that is not there.

### Corrected

- **Kernel speed is not end-to-end speed, and this page briefly conflated
  them.** `docs/SOLATTN.md` paired upstream's "CUDA is 1.4x over Triton at the
  same tau" with the `centroid_tail` tooltip's "~1.4x faster" and proposed
  they were the same number, making the backend gap a default gap. The first
  is end to end, the second is the operation. Since e2e speedup can never
  exceed kernel speedup when only the kernel changes, a 1.4x e2e win requires
  a kernel gap well above 1.4x, so a knob worth 1.4x on the op cannot explain
  it -- and upstream puts `centroid_tail` at ~5-10% e2e, which settles it. The
  matching digits were the whole basis of the claim.
- **The 124-frame frontier table measures the wrong regime.** Upstream's floor
  for seeing anything is ~250-300 frames. That is not a weaker version of the
  result; it is a measurement taken where there was nothing to find.

### Notes

- A local build of kijai/comfy-kitchen's `sol_attn` branch is installed as
  `0.2.31+sol.c04ef20`. The branch declares plain `0.2.31`, identical to the
  wheel ComfyUI pins, so nothing could otherwise distinguish the two and a
  `--force-reinstall` would swap the kernel out with no error -- the node
  falls back to dense and the render merely gets slower.
- `SolAttnMiniMax` is **not** wired into any graph, deliberately. Upstream's
  position is that a proper node waits on global attention timestep scheduling
  landing in ComfyUI core, so the node id is provisional, and this repo's one
  rule is that a node id in a saved graph is forever.

## 0.13.0

### Changed

- **Sage runs `fp16 (most accurate)`, not `auto`.** `auto` resolves to the
  *fastest* kernel, which is the wrong end of this project's tradeoff.
  An accuracy ratio measured against an fp32 reference was recorded here and
  **its figures were withdrawn on 2026-08-16 (see 0.19.0) and removed from this
  entry.** The owner judged fp16 clearer with better motion and less drift on
  video at the same seed. Costs ~1.58x per attention call; the heaviest shipped
  config still peaks at 21,186 MiB of 24,564.

  **The 1.58x was also qualified on 2026-08-14** -- it said "wall clock" here
  and is a per-call kernel cost. The fork retracted that framing, on the
  grounds that if attention were nearly all of H3's compute a render should run
  ~1.5x slower and nothing observed shows it. The decision stands on its
  perceptual leg, which never depended on either number.
- **Sol-Attn is opt-in and ships OFF.** Bypassed in every UI graph, omitted
  from every API graph. It changes what the model computes, and that has never
  been weighed against what its speed buys. `h3_probe_sol_on.json` is the one
  graph that enables it, so the question stays answerable from an artifact.
  Flipping the default also exposed the split graphs' second model chain
  adding Sol unconditionally, which would have shipped stage 2 enabled while
  everything else was off.
- `check_prompt_guide_conformance.py` now tests the **graph** rather than the
  vocabulary for `keyframe completion`. It rejected the type outright because
  `MiniMaxH3ReferenceToVideo` has no keyframe socket; ComfyUI's new
  `MiniMaxH3AddGuide` supplies one, and `model_base.py` merges its keyframes
  with references additively. The verdict was still right for our graphs, the
  reason was false.

### Fixed

- **The reference-video graphs shipped at a length that does not fit on a
  24 GB card.** Measured, not predicted: at 345 frames the arm builds a
  **182,092-token** sequence (102,816 video, 60,212 references, 16,352 text)
  and the render reached step 4 of 16 at 123.5 s/it before Sol-Attn's kernel
  OOMed and fell back, then sage's OOMed and fell back, then ComfyUI's own
  SDPA OOMed with 21.05 GiB allocated against a 23.54 GiB limit. The fallback
  chain behaved exactly as designed; there was simply no room.

  A reference video is truncated to the **generated** frame count, so its cost
  scales with that number twice over: once for the video rows and once for the
  reference rows. That is why shortening the clip helps disproportionately.

  **The first fix was the wrong lever, and shipping it was a mistake.** Cutting
  the render to 124 frames also cut the reference to 5.2 seconds -- so the arms
  built to measure the expensive case stopped containing it. Re-measured
  best-case-first, the full 345 frames **does** fit once the two incidental
  costs are given up:

  | config | result | peak | time |
  |---|---|---|---|
  | 345f @ 1024x768, refs not upscaled | success | 22,735 MiB / 24,564 | 34.3 min |

  The video-bearing arms now ship at 345 frames on a 4:3 1024x768 canvas with
  reference images left at native size, as `REF_VIDEO_BUDGET` -- one constant
  spread into all eight, so the three numbers have a single home. Canvas and
  reference-image detail are incidental to what these arms measure; reference
  duration is the entire point. 1,829 MiB of headroom is **not** a general
  budget: a third reference image or a longer soundtrack can still exceed it.
- **`MiniMaxH3Preflight` told the user a ceiling was "unreachable at legal
  lengths".** True of length alone, false once references are in play -- which
  is exactly when anyone reads that line. 345 frames plus three reference
  videos reaches 201,246 against a 199,728 wrap, with entirely legal inputs.
  It now reports the remaining headroom and says plainly that length alone
  cannot reach it but references can. Caught by reading Preflight's own output
  on a live render.

- **All fourteen reference arms shipped the same task-type prefix.** Every
  prompt opened `[reference generation]`, hardcoded, including the two edits
  and the continuation -- so `h3_ref_video_edit` and `h3_ref_video_only`
  opened with identical words and the relationship axis those arms exist to
  vary had quietly collapsed. The prefix is now derived from the arm's role
  against the official guide's section 3.2 vocabulary, combined with ` + `
  and never repeated: the edit-plus-images arm now reads
  `[video editing + reference generation + audio reuse]`, which is the
  guide's own worked example verbatim. Motion and structure stay
  `reference generation`, because 3.2 is explicit that presence does not
  imply a type -- only a video actually edited or continued earns its own.
- **The voice arm put its only spoken line in `overall_soundscape`.** Guide
  section 6: dialogue and lyrics go only inside `<d>` in
  `detailed_description`. In the soundscape nothing anchored the line in
  time. It now sits in the shot with its `(S1)` speaker id, and the
  soundscape states only the reference relationship, which is what that
  section is for.
- **The motion arm marked the recipient of a transfer, not its source.**
  `attribute_transfer` is defined as characteristics transferred *to* a
  different subject, so on `<Subject 1>` it read as a request to move that
  person's appearance onto somebody else -- the opposite of the arm's intent.
  `<Subject 1>` is now `fully_preserved` from its image and `<Video 1>`
  carries the transfer. Independently corroborated by two outside
  character-swap examples that mark it the same way.
- **Three Sol-Attn documents quoted a `tau` nobody runs.** The note baked
  into all 26 UI graphs showed `tau=2.0` in its "check it is actually
  running" example, and `docs/SOLATTN.md` showed `tau=1.2 bf16`. The node's
  own default is **1.3** and `SOL_RECOMMENDED` pins 1.3, so anyone following
  the instructions saw a number that disagreed with their own log and had no
  way to tell which was wrong. `SOL_BASELINE_124F`'s 1.2 is untouched -- it
  reproduces old measurements on purpose.
- `docs/SOLATTN.md` located `SOL_RECOMMENDED` in `build_workflows.py` (it is
  in `h3_config.py`) and said it ships a `dense_blocks` starting set (it
  ships `""`; the set is `SOL_ARTIFACT_INSURANCE`, deliberately not wired).

### Added

- **`bench/check_prompt_guide_conformance.py`**, which takes its vocabulary
  from the official guide's own tables at run time rather than from us. This
  exists because `check_ref_prompt_labels.py` rebuilds every prompt the
  generator can produce and compares -- a real guard against hand-edits, and
  structurally unable to catch a generator that is confidently wrong. It
  passed clean through the entire prefix collapse above. The new check
  asserts the six sections and their order, the task-type vocabulary and its
  combining rule, that markers never cross the visual/audio sets, and that
  `<d>` appears only in `detailed_description`. It also rejects
  `keyframe completion`, which is legal vocabulary and inert here:
  `MiniMaxH3ReferenceToVideo` has no keyframe socket and the reference
  implementation drops `image`/`last_image` whenever references are present.
  Exits 2, not 0, when the guide is absent. Shown red six ways, including a
  guide whose tables were reformatted away -- the fail-open case, since every
  other assertion is set membership and membership in an empty set passes.
- **A character-swap arm, `h3_ref_video_swap`** -- the reference video as the
  base plate with its character replaced by one from a reference image. It is
  deliberately the twin of `h3_ref_video_image_edit`: same sockets, same
  budget, opposite request. There the person in `<Video 1>` stays and the
  garment changes; here the person is the only thing that changes.

  Its distinguishing feature is a technique **the official guide does not
  contain**: each reference is told what it does *not* supply (`<Picture 1>`
  supplies identity only, not lighting or background or framing; `<Video 1>`
  does not supply the face). Every relationship in the guide is stated
  positively, so these negative clauses come from general prompting research,
  where the reported failure is the model blending the two identities.
  **Whether they earn their tokens is untested**, which is why the arm exists
  next to its twin rather than replacing it.

  It ships `[video editing + reference generation + audio reuse]`, matching
  the guide's own worked example for this socket combination. Community
  write-ups of this scenario typically stop at a bare `[video editing]`;
  guide section 3.2 adds `audio reuse` whenever the source audio stays
  audible, which here it does at `fully_copy`.
- **`h3_probe_prompt_concise`, the first graph here that breaks the prompt
  format on purpose.** It is `h3_ref_video_swap` with one paragraph instead
  of six sections -- same clip, image, seed, canvas, length and sampler, so
  the only thing reaching the model that differs is the structure. General
  prompting research reports working character swaps from prompts far looser
  than the guides specify, some with no `retention_analysis` and one a single
  sentence. Those reports are uncontrolled, so they are not evidence the
  format is useless, only that nobody has measured it -- and the six sections
  cost tokens in a budget where reference rows already dominate.

  `check_prompt_guide_conformance.py` waives its two structural cases for
  this graph **by name**, prints the waiver on every run, and still enforces
  markers, dialogue placement and label agreement on it. Both were verified
  by mutation: the five conformance cases still go red on a normal graph with
  the waiver in place, and the probe still goes red when its prompt names a
  reference the graph does not wire.

  Fixing this exposed a bug in the day-old check: it located prompts by
  searching for `"subject_definitions:"` in any string input, so the one
  graph with no section headers was skipped **silently** and counted as a
  clean pass over a prompt it never read. It now reads the reference node's
  own `prompt` input.
- `ref_image_count`, so an arm can wire **one** reference image instead of
  always two. The swap arm takes its environment from the plate, so a second
  image was not merely redundant -- it paid reference rows on every sampling
  step to say nothing. `check_ref_prompt_labels.py` caught it as an unnamed
  wired reference before it shipped.
- `VIDEO_ROLES` / `AUDIO_ROLES` are now named once in the generator and
  imported by the drift guard, which previously kept its own hardcoded copy.
  That copy stopped covering the generator the moment `swap` was added and
  reported a freshly generated graph as a hand-edit.
- **The denoising trajectory is now recoverable, not just watchable.** The UI
  graphs gain `GetPreviewOverrideFramesKJ` and a `PreviewImage` sink, which
  return the frames `ModelPreviewOverrideKJ` already decoded through taeh3 as
  an image batch. The live widget shows the current step and forgets the
  previous one; this keeps the whole run, at no extra compute, which is what
  makes two sampler or scheduler arms comparable without paying for a full
  decode each. UI-only, like the preview node itself: in an API graph the
  frames node would not merely be useless, it would raise.
- `force_rate` is now guarded, with the hazard measured rather than argued.
  On three 6.00-second clips trimmed to differ only in frame rate:

  | source | H3 reads it as | error | last conditioner label |
  |---|---|---|---|
  | 24 fps | 5.875s | 0.0% | `<5.2 seconds>` |
  | 25 fps | 5.875s | **+4.2%** | `<5.2 seconds>` |
  | 30 fps | **7.292s** | **+25.0%** | `<7.0 seconds>` |

  At 30 fps the model is told a six-second reference is seven and a quarter
  seconds of action. A 24 fps source is unaffected either way, **which is why
  testing on one proves nothing**. `check_ref_prompt_labels.py` now fails the
  build if any loader feeding a reference socket drops off 24.

### Verified

- **The reference cost model is exact.** Preflight on a live render reported
  `references 60,212`; the model predicted 60,212 from the clip's own
  properties (345 ref frames, latent_t 102, canvas clamped to 960x544 by the
  no-upscale rule, plus two 1024x1024 images upscaled to 2048x2048).
- **The untruncated reference-audio divergence is real, and now observed.**
  Preflight reported `audio refs 1,562` rows = 781 latents = **19.52 seconds**
  of audio conditioning for a **14.375 second** generation. The reference
  pipeline truncates a soundtrack to the generated duration; ComfyUI encodes
  the whole waveform. 412 rows carried past the end of the clip.
- The Preflight duration line, fixed in 0.11.0, is live and correct on a
  reference graph -- the case where it was absent in 7 of 8 graphs before.

## 0.12.0

### Added

- **The reference-combination matrix.** Five graphs differing only in which
  reference sockets are wired: video alone, video plus its soundtrack, images
  plus a standalone audio clip, images plus video plus soundtrack, and all
  four at once. Everything else is shared by construction, so they are
  directly comparable.
- `bench/check_ref_prompt_labels.py`. A `ref2va` prompt refers to its
  references by label, and `MiniMaxH3Tokenizer` derives those labels **from
  the wired sockets, not from the prompt** -- so the two drift silently. The
  check asserts each shipped ref graph's prompt names exactly what its graph
  wires, in the tokenizer's own numbering: images, then videos with each
  soundtrack's `<Audio j>` immediately *before* its `<Video k>`, then
  standalone audio, with a separate 1-based counter per type. A video's
  soundtrack therefore takes `<Audio 1>` and a standalone clip beside it is
  `<Audio 2>`.

  It caught a real mismatch on its first run, and was shown red both ways: a
  prompt naming a label the graph does not wire, and a wired reference the
  prompt never mentions. The second matters because an unreferenced reference
  still costs its rows on every sampling step -- the most expensive way to say
  nothing.
- `_ref_prompt()` generates each arm's prompt from what it wires, following
  `internal/official_prompt_guides/...ref_en.md`: the six sections in order,
  visual markers from section 4.1 and audio markers from 4.2, which are a
  different set and do not interchange.

### Fixed

- **A silent clip cannot have its audio socket wired.** VHS raises "failed to
  extract audio" when its audio output is pulled on a video with no audio
  stream, and the render dies at execution having validated cleanly. The
  video-only arm loads a different, silent clip and leaves the socket alone.
  Found by running it.
- The placeholder input files were checked against `ComfyUI/input`, which on
  this install is **not** the input directory --
  `folder_paths.get_input_directory()` resolves elsewhere. That produced a
  confident "29 of 30 combo entries are stale" conclusion which was entirely
  an artifact of looking in the wrong place. All placeholders verified against
  the real directory; the original two images were correct all along.

## 0.11.0

### Fixed

- **Every API graph in this repo was unsubmittable, and had been since
  `93b08b1` wired the Resolution node in.** `MiniMaxH3Resolution.shape` is a
  `COMFY_DYNAMICCOMBO_V3`, and ComfyUI addresses a DynamicCombo's members by
  their **dotted** path: `shape.wide_resolution`. The generator wrote them
  flat, and ComfyUI's executor answers with `required_input_missing` naming
  `shape.wide_resolution`. The API form is the form `bench/*` drives, so every
  bench run since then was POSTing a prompt the server refuses.

  **`validate_api` had it exactly backwards**, on a belief written into its own
  comment: "the API prompt carries them flat for ComfyUI to re-nest". It does
  not. So the validator was green-lighting graphs the server rejects, and once
  the generator was fixed it briefly rejected the correct form. A validator
  that accepts what the server refuses is worse than no validator.

  Both spellings were tried against a running ComfyUI before either changed:
  dotted accepted, flat refused, for the band case and the `custom` case
  alike. **Found by `bench/smoke_h3.py`, the only thing here that actually
  submits a prompt** -- no static check could have caught it, because the
  static check was the thing that was wrong.
- The three ref `_api` graphs hardcoded `length` while wiring `width`/`height`,
  so sweeping length on `MiniMaxH3Resolution` moved the canvas and left the
  duration behind, silently, and only in the form the benches drive. It also
  skipped the node's own `snap_length()`.
- Four bugs in `reference_fit.py`'s clamp-lift machinery, all one family:
  the "nothing to lift" branch neither armed nor disarmed, so it was the
  branch that let a previous prompt's value through; a **cached** fit node
  never armed at all, so editing only the downstream prompt silently reverted
  to the 2048 clamp with the checkbox still ticked; an unconsumed arm survived
  into a later prompt; and with two fit nodes the one with the box **off**
  cancelled the other's arm, resolved by an execution order that is neither
  the graph's visual order nor settable. Arms are now per node, cleared on
  every downstream call, and `fingerprint_inputs` keeps the node out of the
  cache exactly when the arm depends on it.
- `MiniMaxH3ReferenceFit` now reads the downstream `ref_image_size` from the
  prompt and **says so when it is on `match`** -- under which the stock node
  sizes references from the video's pixel area, never reads the 2048 constant,
  and undoes this node's resize entirely. The log previously reported a 3.6x
  to 16x improvement that had not happened.
- `MiniMaxH3Preflight` printed its duration line only when
  `minimax_frame_count` was present, which core sets **only** on the keyframe
  path. It was therefore absent from 7 of the 8 shipped graphs, including every
  ref graph -- where the 345-frame ceiling matters most. Derived from
  `latent_t` when the key is absent, and marked as derived.
- `preflight.py`'s docstring said the `csrc/fused` uint32 wrap sits near
  199,729; the code computes 199,728.

### Added

- **Reference video, wired for the first time.** `h3_ref_video_to_video.json`
  loads a clip through `VHS_LoadVideo` at **`force_rate=24`** into
  `ref_videos.ref_video_0`, with its own soundtrack into the index-paired
  `ref_video_audios.ref_video_audio_0`. The rate is not optional: the stock
  node has no fps input and assumes 24 twice, for the DiT's temporal clock and
  for the `<T.T seconds>` labels the conditioner reads, so a 30 fps source at
  `force_rate=0` is conditioned 25% slow with nothing said.

  No fit node on this path, deliberately. The same no-upscale divergence
  exists as for images, but a five-second reference at full canvas is +32,256
  rows against +7,168 for a `max` image reference, so it is documented and
  left open until the cost is known to buy something.
- **The two-stage split, both orderings.** `h3_probe_split_base_last.json` and
  `h3_probe_split_base_first.json`: one `BasicScheduler` into `SplitSigmas`,
  two `SamplerCustomAdvanced` stages, `DisableNoise` on the second, and a
  second model chain at an identical shift so the halves run different models
  on one schedule. Built on the custom-sampler stack rather than
  `KSamplerAdvanced`, which was broken on nested latents until core
  `27bca654` (2026-08-12) -- and H3's AV latent is a NestedTensor.
- `bench/check_schema_defaults.py` (from 0.10.1) and six new cases in
  `check_short_edge_override.py` covering the arm fixes above.

### Changed

- `validate_api`'s model-fork invariant understands the split: two model
  sources are allowed **only** when `SplitSigmas` is present, and both stages
  must still read sigmas from one `BasicScheduler`. Verified it still catches
  a stray second source on a non-split graph, and a third source on a split
  one -- a relaxation that could not fail would be worse than the check it
  replaced.
- `MiniMaxH3ReferenceFit`'s second output is `latent_rows`, not
  `vision_tokens`. It returns the DiT's packed rows; the description and the
  module docstring already called it that.

## 0.10.1

### Fixed

- **`MiniMaxH3KeyframeCanvas.execute` still carried the old defaults**
  (`mode="fit_to_canvas"`, `length=0`) after 0.10.0 moved the schema to
  `match_keyframe` / 124. ComfyUI does **not** inject a schema default for an
  input a prompt omits -- the Python signature default is what applies -- so
  the widget fix only protected a node newly dropped in the UI. An
  API-format prompt that left `length` out still emitted 0 from slot 5 and
  rendered the 5-frame, 0.208-second clip 0.10.0 exists to kill, on exactly
  the path `bench/*` drives renders over. Found by review, not by the tests.
- The ultrawide probe's note claimed 1536 is "the widest the trained family
  allows". It is not. Sweeping `adapt_canvas` over the legal 1:4..4:1 range
  gives **eight** canvases at exactly 1008 tokens/frame -- 1344x768,
  1536x672, 1792x576 and 2016x512, plus each transposed -- so the equal-cost
  axis runs to a 3.94:1 frame, not 2.29:1. The note now says so and says this
  probe takes one step along it rather than the last one.
- `_NOTE_TURBO` still read "the trade is aspect: it saw one, where the 4-step
  v0.1 saw mixed aspect ratios" after the table was corrected to show the
  8-step v1.0 as mixed-aspect too. It named the wrong sibling.
- The 0.10.0 "Notes" block described the shipped graphs as not yet
  regenerated. They were regenerated in the same release, so all three of its
  statements had become false.

### Added

- `bench/check_schema_defaults.py`. Asserts every node's `io.Schema` defaults
  match its `execute()` signature defaults, across all 7 nodes, plus that a
  required input never acquires a signature default (which would quietly make
  it optional on the API path). This is the general form of the bug above:
  the two are independent by construction and nothing else compares them.
  Shown red on the real `length` split before being trusted.

### Changed

- `_check_geometry` now documents its own scope. Under the `match_keyframe`
  default an i2v graph derives its canvas from the loaded keyframe at run
  time, so the aspect assertion validates the configured *fallback* rather
  than what renders -- swap in a 3:4 still and the graph renders 768x1344
  having passed a check that looked at 1344x768. The guarantee is not lost,
  it moves to the node, which enforces it on the source image and raises. The
  `GRAPHS` comment's "canvas is shared by construction" claim now carries the
  i2v exception.

## 0.10.0

### Changed

- **`MiniMaxH3KeyframeCanvas.mode` now defaults to `match_keyframe`**, and the
  generator's `canvas_mode` with it. `fit_to_canvas` is the reference
  pipeline's deliberate-override branch, not its default, and it is lossy:
  it cover-crops **both** keyframes, so a 3:4 photo forced to 1344x768 keeps
  43% of its frame. The "it keeps render cost where you put it" defence does
  not hold at the default, because 1344x768 is already the largest area
  `adapt_canvas` ever returns -- it only pays once you lower width/height on
  purpose. `match_keyframe` is also the parity-faithful branch, so anything
  compared against diffusers wants it.
- **`MiniMaxH3KeyframeCanvas.length` now defaults to 124**, from 0. At 0 the
  node forwarded 0, and core's own `min=5` does not catch it: a *linked* input
  skips range validation entirely, so the shipped default rendered a 5-frame,
  0.208-second clip. 124 is the trained floor and matches both core's default
  and `MiniMaxH3Resolution`'s.

  Neither flip touches a saved graph. Widget values are stored per node, so
  only a newly dropped node moves.

### Added

- Both graph builders take `sampler_name` and `scheduler_name` overrides.
  Previously every graph inherited `SAMPLING` with no way to vary the sampler,
  which made the one comparison the vendor's own graphs invite impossible to
  ship.
- `TURBO_768P_LORA` / `_STEPS` / `_SHIFT`, `TURBO_HOME_CANVAS` and
  `TURBO_SAMPLER` in `h3_config.py`, and `check_distill_settings.py` now
  grades **every** turbo triple the config declares rather than the first one,
  with a guard that fails if a new `*_TURBO*_LORA` constant appears ungraded.
  Shown red by setting the 768p shift to 12.
- Six graph variants, from the 08-13 research. None existed before and each
  covers a use case or a theory the shipped set could not express:

  | graph | what it is for |
  |---|---|
  | `h3_text_to_video_turbo_4step_768p` | the only turbo LoRA whose shift is not 12/3, shipped correct rather than described |
  | `h3_probe_turbo_home_canvas` | the 8-step LoRA at the 544p it was distilled at, against the same LoRA at 1344x768 |
  | `h3_probe_turbo_euler` | the vendor's sampler against core's, scheduler held at `simple` |
  | `h3_probe_ref2v_turbo` | ref2v with an fl2v distill LoRA, deliberately out of distribution |
  | `h3_probe_canvas_ultrawide` | 1536x672 |
  | `h3_probe_canvas_portrait` | 768x1344 |

  The last two are the **equal-cost shape control**: 21:9, 16:9 and 9:16 are
  all 1008 tokens/frame, so all three run at an identical sequence length
  (verified: 104,478 at 345 frames for all three) while the long edge goes
  768 to 1536. Every other probe here changes cost to change shape. These
  change shape with cost held constant, which is the only way to ask whether
  the model is shape-neutral rather than just cheaper in one orientation.

### Notes

- The shipped `workflows/*.json` were regenerated against a live ComfyUI in
  the same release. `/object_info` was checked first and reported the new
  `match_keyframe` / `124` defaults, confirming the pack had reloaded --
  regenerating against a stale server bakes in exactly the mismatch
  `check_workflow_schema.py` exists to catch. 31 graphs written and
  validated, UI/API cross-check passing.

## 0.9.0

### Added

- `bench/check_distill_settings.py`. The three FL2VA turbo LoRAs do not share
  a schedule -- two were distilled at video shift 12, the 768p 4-step at 6 --
  and a LoRA inherits the sampler's shift rather than carrying its own. So
  loading the 768p one into a graph still reading 12/3 samples it off a
  schedule it never saw, and nothing errors.

  Covers **every** shipped graph, not only the two that load a LoRA: a turbo
  graph must match its LoRA's row, a base graph must sit at the base
  checkpoint's own 12/3, and the UI and API forms of each graph are paired and
  compared, since they are generated separately and have already diverged once.
  Both the shifts and the recommended step counts are graded against the vendor
  README in `coderef/` rather than against themselves; grading only the shifts
  would leave the step sets self-checked. When `coderef/` is absent that control
  is skipped and the script **exits 2, not 0**, so a runner keying on the exit
  code can tell a skipped control from a clean pass.

  LoRA filenames are parsed structurally rather than by substring, because a
  substring match reads a hypothetical `turbo_8step_v1.0_768p` as the 12/3
  `turbo_8step_v1.0` row -- the exact silent failure the file exists to catch,
  committed by the file itself.

  Shown red eight ways before being trusted: config shift wrong, config steps
  wrong, a turbo graph's shift edited, a base graph's shift edited, the UI and
  API forms disagreeing, our shifts disagreeing with the vendor, our steps
  disagreeing with the vendor, and `classify` reverted to substring matching.
  Green restored after each.
- `docs/checks.md`, an index of every check: what it defends, what it needs to
  run, and whether it has been shown red. Ten of twelve have no such record,
  which is a finding rather than a formatting gap.
- `docs/h3_ref2v_distillation.md`. lightx2v has shipped three FL2VA turbo
  LoRAs and no ref2v one, and their roadmap lists it as future work. This is
  why, from the code. Three mechanisms: fl2v conditioning is positionally
  *identical* to the target (a first-frame keyframe's rotary coordinates are
  `torch.equal` to the target's first latent frame) while a reference sits on
  its own grid and pushes the target's origin by 1 to 1206 units; ref2v is a
  separate `transformer_ref` partition measuring **4.2% relative Frobenius**
  from fl2va while the whole 8-step turbo LoRA measures **0.036%**; and the
  DMD trainer that produced those LoRAs has no ref2v path at all, rejecting
  non-text conditioning outright.

  Both headline measurements were reproduced independently before shipping:
  key sets identical (0 on either side, 1082 shared), projection deltas
  0.037-0.046, `final_layer.adaln_proj` at 1.92 i.e. essentially rewritten,
  and mean LoRA perturbation 0.00036 across 208 touched modules at
  `alpha/rank = 0.0625`. The LoRA does not touch `final_layer`, `adaln_proj`,
  the norms or the patch projections, which is where the checkpoints differ
  most -- a point in favour of the out-of-distribution experiment working at
  all.

  Also records three hypotheses that did **not** survive: re-injected
  reference rows are byte-identical in both tasks, no guidance mechanism
  exists for either, and DMD here is genuinely data-free so "reference pairs
  are scarce" does not bite.

### Changed

- `docs/h3_geometry_and_nodes.md` corrected in three places. The low-VRAM
  saving is **~3227 MiB at 4 groups**, not the ~1070 carried here and in the
  shipped graph notes -- `workflows/h3_config.py` measured three times the
  earlier estimate. `MiniMaxH3SigmaShift` was listed as an untested
  third-party node; it is core ComfyUI and sits in all eight shipped graphs.
  And the keyframe section no longer tells the reader to wire both image
  outputs: with no `last_frame` input, that slot returns the same tensor as
  `first_frame`, so wiring it silently anchors the render to return to its
  opening frame.
- `workflows/build_workflows.py` carried the same stale ~1070 MiB in its note
  template, so it was baked into all eight shipped graphs. Fixed at the
  generator. **The shipped `workflows/*.json` still carry the old number until
  they are regenerated against a live ComfyUI**, which this change does not do.
- `README.md`'s `bench/` listing was missing four scripts and understated what
  needs CUDA or `PYTHONPATH`. It now points at `docs/checks.md` as the index.
- `docs/open_experiments.md` notes which entries the 2026-08-13 plan
  schedules, and which stay blocked on owner judgment.

### Notes

- Every check was run against ComfyUI `12666983` (v0.32.0), comfy-kitchen
  0.2.31 and KJNodes `6ab7e81` on 2026-08-13. All eleven runnable ones pass;
  `check_workflow_schema.py` needs a live server and was not run. Nothing was
  stale in the sense of failing -- the staleness was in the documentation
  around them.
- **Decided: do not reimplement KJNodes' low-VRAM path.** Their node does
  three things and we already own head chunking. Their attention patch yields
  to ours; their block patch is unconditional and unguarded, so writing our
  own block-level release would collide on
  `diffusion_model.blocks.{idx}.forward` with no marker convention. The
  interop cases in `check_lowvram_handoff.py` stay, and that file's name
  undersells it -- two of its five cases guard our own head-chunk
  reassembly, not the KJNodes boundary.

## 0.8.2

### Changed

- `PublisherId` removed from `pyproject.toml`. It exists to publish this to
  ComfyUI's registry, and this repo is not asking to be installed -- the same
  reason the graphs carry no `cnr_id`. Public so the work is readable, not so
  it is distributable.

## 0.8.1

### Changed

- Graphs no longer carry `cnr_id`, reversing a change made earlier the same
  day. It exists so ComfyUI-Manager can offer "install missing custom nodes"
  to someone opening a graph without this pack -- an audience pulling from a
  public registry, which is not what this repo is. Meanwhile
  `useConflictDetection` ships in the same lazily-loaded chunk as
  `useComfyRegistryService` (baseURL `https://api.comfy.org`) and the
  consuming path was not proven to stay local. Under a local-only
  constraint, unproven beats unlikely when the benefit is near zero.
- The workflow `id` namespace seed is a bare string rather than a fabricated
  `github.com` URL naming a handle and a repository. Determinism was the only
  property needed. Every graph id changes once and is stable after.
- Recorded in the generator, so neither is re-added: never emit `models[]`
  (it carries download URLs and the local model directory layout), never
  emit `extra.info` (identity), and never derive `aux_id` automatically --
  its conventional value is the git remote's `owner/repo`, and this repo's
  remote is a LAN address, so that would write a private IP into every
  shared workflow.

### Verified

- Schema: staying on `version: 0.4`. The installed frontend (1.48.7) writes
  0.4 exclusively across every saved workflow on this machine, there is no
  schema-1 serializer in the bundle, and ComfyUI's Python side never parses
  workflow schema at all -- UI workflows move through `userdata` as opaque
  blobs. Moving to schema 1 gains nothing observable and produces a file the
  installed frontend has no writer for.
- Pre-publish scan of tracked content: no home paths, usernames, hostnames,
  LAN addresses, emails or keys. The only IPv4 matches are `127.0.0.1`
  defaults. Our graphs leak less than a frontend save, which carries
  `extra.frontendVersion` and a per-node `ver`; we emit neither and should
  not start.

## 0.8.0

### Added

- `MiniMaxH3Resolution` is wired into every graph except first-frame, where
  the keyframe decides the geometry and `MiniMaxH3KeyframeCanvas` is the node
  that does it. Width, height and length now arrive as links from a node whose
  dropdown says what the choice costs, instead of as two numbers that say
  nothing: `1344x768  7/4  1008 tok/frame  1.00x`.
- Both validators learned that a `DynamicCombo` declares its option inputs
  nested under `options` rather than as top-level inputs. `validate_api`
  called them unknown; `check_workflow_schema` counted the widgets short.
  Third and fourth instance of the same shape -- a node whose input set is
  not fully static -- after VHS's format widgets and dynamic member slots.
  The UI branch was shown red by removing it and watching the count mismatch
  return.

## 0.7.3

### Fixed

- Output prefixes were split across `Video/` and `video/`, which is two
  sibling directories on a case-sensitive filesystem for what reads as one
  destination. All eight graphs now write to `Video/`.
- Our nodes carry `cnr_id` in `properties`, which is what ComfyUI-Manager
  reads to offer "install missing custom nodes". Without it someone opening
  a shipped graph without this pack got red boxes and no way to resolve
  them. Stamped only on nodes this pack owns; claiming another pack's id
  would send Manager after the wrong repo.
- Graphs carry `extra.ds`, so they open on the nodes. Without it litegraph
  starts at its default viewport while these graphs begin at x = -2860, so
  the first thing you saw was empty canvas.

## 0.7.2

### Changed

- Every per-call VRAM figure in this repo is now scoped as per-call, in
  `attention.py` and in `bench_minimax_attn.py`'s own docstring, because an
  e2e run contradicted the inference everyone was drawing from them. At 124
  frames, head_chunks=4 measured 1186 MiB *higher* on process peak than
  head_chunks=1, the opposite direction from the per-call numbers, and a
  second run measured a 2265 MiB spread across two runs of one unchanged
  configuration. The excursion is larger than the effect, so the sign is not
  settled either way at that sample size -- what is settled is that per-call
  peak does not predict process peak.
- `bench_minimax_attn.py` reports `reserved` beside `allocated`. That was to
  test whether chunking fragments the allocator, since reserved rather than
  allocated is what a process occupies. It does not: the two track within
  8 MiB on all three arms, so a single call on a clean allocator does not
  fragment and the mechanism lives in the interaction across 50 blocks and
  every step with a model resident -- which a per-module bench excludes by
  construction.
- The clone still ships. It is free (0.7% wall-clock, bit-identical output)
  and it does lower the attention call's peak. It is simply not demonstrated
  to lower what a user's card reports, and nothing should be spent to get it.

### Fixed

- Workflow `id` is a UUID rather than a readable slug, matching what the
  frontend writes. Stale copies of three graphs in ComfyUI's own workflow
  directory, carrying the old slug ids, were the source of "Unable to find
  workflow in <file>"; they were moved aside rather than deleted.

## 0.7.1

### Fixed

- `MiniMaxH3Resolution` ignored every widget and always returned 1344x768. A
  `DynamicCombo` arrives as one nested dict -- the selected key under the
  input's own id, the chosen option's inputs alongside it -- not as flattened
  kwargs. It read `shape.get("key")` and took its values from `**kw`, so every
  selection fell through to the custom branch and picked up the literal
  defaults. Its test passed because the test called `execute` with flattened
  kwargs: the caller was invented rather than observed, so the test agreed
  with the bug.
- The armed short-edge override could outlive its prompt. Arming is per fit
  node and consumption is per downstream call, and a graph has two fit nodes
  feeding one `ReferenceToVideo`, so the second arm survived into whatever ran
  next. Each fit node now disarms on entry when the flag is off, and a
  mismatched second arm warns.
- `_install_wrapper` would install `classmethod(_make_wrapper(None))` if
  upstream ever moved `execute` to a base class, killing every reference
  render in the process including graphs that never enabled the flag. It now
  declines and says why.
- `SageChainAssert` was emitted with `warn_only=False` even for `sage=False`,
  so a control arm would always raise at the gate. `warn_only` now follows
  `sage`.
- The display renames reached the nodes but not the shipped in-graph note,
  the README, or `docs/h3_geometry_and_nodes.md`, which still told readers to
  search for names the menu no longer has. Two notes in the same graph
  disagreed.

### Changed

- `SageChainAssert` finds Sol-Attn's counters by capability rather than by
  module name. ComfyUI registers a directory custom node under a path-derived
  key, so `import_module("ComfyUI-SolAttn_triton")` resolved in a shell and
  never inside ComfyUI: the call-time probe had never executed once since the
  node was written, and every render printed `chain assert ok` with its
  strongest evidence skipped.
- The override declines an unexpected `q.ndim` instead of raising. It exists
  to catch calls another patch handed back, so the case where a safety net
  earns its keep is a caller nobody predicted -- and it killed the render
  instead of degrading. Our own probe was that caller.

## 0.7.0

### Added

- Three probe graphs, each a pair with a named twin, one variable changed, and
  a note saying what to compare and what to expect.
  `h3_probe_reference_upscale` runs references without the released
  pipeline's upscale, against roughly 13,900 fewer vision tokens.
  `h3_probe_square_canvas` renders the same prompt at 768x768, which is about
  a third of the attention cost and the largest lever anywhere in this
  pipeline. `h3_probe_head_chunks` runs the heads in 4 groups, trading kernel
  launches for 217 MiB of peak.
- `bench/check_generator_constants.py`. Sharing a constant prevents drift;
  this prevents un-sharing, which is a different failure on a different
  timeline -- someone inlining the literal back because it reads more
  directly. Agreement is not a testable property, so each case forces a
  disagreement by moving upstream's value and asserting the graph follows.
  Writing `2048` back into the builder turns it red.

### Changed

- `SEED` is 730451892 rather than 1. Fixed rather than randomised, because
  the probe graphs are pairs and a seed that moved between them would put the
  difference you are looking for underneath the difference you are not.

## 0.6.1

### Added

- `MiniMaxH3ReferenceFit` is wired into both reference graphs, one node per
  reference image, paired with `ref_image_size` on `max`. The pairing is
  load-bearing rather than tidy: under `match` the stock node sizes
  references from the video's pixel area and never reads the 2048 constant,
  so the fit nodes would be silently undone and their two resamples wasted.
  Closes the gap where the graphs named for references were the ones
  under-sizing them.
- `MiniMaxH3Preflight` sits between conditioning and the sampler in every
  render graph, with both consumers reading through it, so a graph cannot
  half-apply it.

## 0.6.0

### Added

- `MiniMaxH3Preflight`. Reads the assembled conditioning and latent through
  the model's own `PackedLayout`, so the sequence length it reports is the one
  attention will run at. Draws on the node: resolution and whether it is
  inside the trained family, sequence length broken down by segment with
  bars, what other aspect ratios would cost at the same length and
  conditioning, and where the int32 thresholds sit for the fused layout.
  Pass-through. It is the only node that can answer "what am I actually
  sending", because the total does not exist until conditioning is assembled.

### Changed

- `MiniMaxH3ReferenceFit`'s `mode` combo is now an `allow_upscale` boolean.
  The two modes differed by exactly `min(1.0, full)` versus `full`, so they
  were one strategy with the clamp on or off, and `reference`/`down_only`
  named provenance rather than behaviour. Saved graphs carrying the old
  string value will need it reset; the node is in no shipped workflow.
- Display names: "MiniMax H3 Keyframe Resolution" and "MiniMax H3 Reference
  Resolution", so the pair reads as two answers to one question. Both
  `node_id`s are unchanged.
- `lift_downstream_clamp` now announces itself as experimental in its label,
  leads its tooltip with what it will cost you, and logs a WARNING rather
  than an info line when armed. It monkeypatches a core node and pushes
  references off the distribution the checkpoint was trained on; that should
  be impossible to trip over by accident.

## 0.5.1

### Added

- `lift_downstream_clamp` on `MiniMaxH3ReferenceFit`, appended last so saved
  widget order is untouched. `MiniMaxH3ReferenceToVideo` sizes references
  with `min(1.0, 2048/short_edge)`, so anything larger this node produces is
  scaled straight back and `short_edge` above 2048 appears to do nothing.
  This lifts that clamp for exactly one downstream call by rebinding the
  module constant the stock node reads at call time, then restoring it in a
  `finally`. Off by default, and above 2048 is off-distribution: 2048 is what
  the released checkpoint conditioned image references at.
- `bench/check_short_edge_override.py` pins the scope rather than the
  arithmetic, because a global rebind that outlived its call would change
  references in graphs that never asked for it. Four properties: applies to
  exactly one call, restores on a raise, declines under `ref_image_size`
  `match` where the constant is never read, and installs idempotently behind
  a marker the way the chaining packs do. Driven through `_make_wrapper`
  with a stub, so it needs no VAE, CLIP or model. Mutated three ways --
  never clearing the arm, dropping the `finally`, removing the marker check
  -- and each turns cases red.

## 0.5.0

### Added

- `MiniMaxH3Resolution`. Pick a resolution by shape and read its cost in the
  dropdown you pick it from: each entry carries resolution, aspect ratio,
  video tokens per latent frame, and attention cost against 16:9. A
  `DynamicCombo` bands the choices so no list runs long -- 22 entries at
  worst -- while all 95 trained resolutions stay reachable, plus `custom`
  for anything the DiT can patch. The list is swept from `adapt_canvas` at
  import rather than hardcoded, so it tracks upstream.
- It answers the question no other node could: whether you are inside the
  trained family. Core's conditioning nodes never call `adapt_canvas`, so the
  768 short edge and the area cap constrain nothing you type. 1024x1024 is
  legal, renders, and is outside the family; the node says so and names the
  resolution `adapt_canvas` would have given instead.
- A README section on which sizing node to use when, built on the rule that
  separates them: a keyframe is patchified on the video's latent grid so its
  resolution must equal the video's, a reference is patchified on its own so
  its resolution only sets the vision tokens it contributes.

## 0.4.3

### Added

- `SageChainAssert` is now last in the model chain of every render graph,
  after Sol-Attn, asserting what the composition ended up as rather than what
  any one node intended. That seam is negotiated through a duck-typed
  attribute both sides rewrote within a minute of each other once already,
  and when it breaks the render still succeeds while being quietly slower or
  numerically different. `exercise` stays on, so the evidence is call-time
  rather than install-time: a fraction of a second and ~176 MiB transiently,
  against renders that peak near 3 GB in attention alone.

## 0.4.2

### Fixed

- `MiniMaxH3ReferenceFit` over-reported its cost by 4x. It counted VAE latent
  cells, where the DiT patchifies those 2x2 before attending them, so a
  reference fitted to 2048x2048 reports 4096 vision tokens and not 16384.
  The 0.3.0 entry's "1024 latent rows instead of 16384" was the same error:
  the real figures are 256 and 4096.
- `check_reference_fit.py` imported the node's own `_rows` and graded it
  against itself, so it passed while every number was 4x high. It now checks
  against `comfy.ldm.minimax.model._frame_grid`, which is what actually
  builds the reference block. Reverting the math turns 7 cases red.

### Changed

- That output is now `vision_tokens` rather than `latent_rows`, matching
  standard transformer terminology: tokens qualified by modality, sequence
  length for the total.

## 0.4.1

### Added

- `check_no_write_through_to_source`: patching a model must not reach the
  model the caller still holds. It runs against a deliberately
  shallow-cloning fake, because against the real `ModelPatcher` the
  assertion cannot fail and would be decoration. Under the real thing the
  node's defensive copy is redundant; the case pins that the node does not
  depend on that, since the failure it prevents is a contaminated A/B
  control arm that looks like a result.
- The `ModelPatcher` stub now clones the way ComfyUI does, ported from
  ComfyUI-AudioLoopHelper's `tests/_fakes.py`. It previously returned
  `self`, which cannot tell a node that mutates its source from one that
  does not.

### Fixed

- The comment explaining that copy said `clone()` shallow-copies
  `model_options`. It does not, and did not when the comment was written:
  `clone()` runs them through `comfy.utils.deepcopy_list_dict` on every
  branch, which landed 2026-02-10. No behaviour was wrong, only the stated
  reason, which is the kind of thing a later decision gets made on.

## 0.4.0

### Changed

- All graphs write video through `VHS_VideoCombine` instead of
  `CreateVideo` -> `SaveVideo`. One node, muxing the audio itself, at 24 fps,
  `video/h264-mp4` (yuv420p, crf 19), `save_metadata` on, `save_output` on,
  prefix `Video/h3_<task>`. h264 rather than h265 or nvenc: software x264 at
  crf 19 is the most portable mp4, and the nvenc paths trade quality per bit
  for encode speed on a file that writes in seconds next to a render that
  takes minutes. `trim_to_audio` stays off, since H3 generates the pair
  jointly and trimming video to the audio track can only drop frames it meant
  to keep.
- The canvas note is rebuilt around the rule rather than examples: you pick
  an aspect ratio and the canvas is derived, there being only 94 legal
  canvases in the whole 1/4 to 4 range. It now states two things the old note
  got wrong by omission. The short edge is 768 only while the area cap does
  not bind, so 21:9 is 1536x672, not 1536x768. And 1.00x is not the cost
  ceiling: rounding each axis to 32 puts 29:9 (1856x576) at 1.07x, above
  16:9, for no extra pixels.

### Fixed

- `UIGraph.add` turned a dict of widget values into a list of its keys.
  Nothing had passed a dict before `VHS_VideoCombine`, whose widget set
  depends on the selected format and so serializes keyed rather than
  positionally. The graph loaded and rendered with every setting wrong.
- Both validators treated format-dependent widgets as errors, then as
  invisible. `build_workflows.validate_api` called `pix_fmt`/`crf`/... unknown
  inputs, though VHS reads them from `**kwargs` and defaults any it does not
  find. `check_workflow_schema` only understood list-shaped `widgets_values`,
  so a keyed one fell through both branches and the node went unchecked while
  reporting ok. Both now resolve the selected format's own widget list out of
  `/object_info`. Third false-positive class of this shape, all of them nodes
  whose input set is not fully static.

## 0.3.7

### Added

- `h3_text_to_video_turbo.json`, on the 8-step v1.0 LoRA at 8 steps. It
  carries a note on what the training resolution means, which is the part
  that is not obvious: 544p and 768p are about a factor of two apart in
  tokens, and H3's own canvas rule is a 768 short edge, so a 544p-distilled
  LoRA cannot be at home at the same time as the base model. Rendering at
  1344x768 keeps the base model in its trained canvas and the LoRA off its
  distillation resolution; rendering at 544p reverses that. Which costs less
  is unmeasured, and the note says so rather than implying the LoRA wins.
  t2v deliberately, because `MiniMaxH3KeyframeCanvas` refuses sub-768 and an
  i2v graph could not demonstrate the choice its own note describes.
- `steps` and `shift` are builder parameters now, so a variant can move them
  without a second copy of the graph description.

### Changed

- The `head_chunks` tooltip said "1 disables it" four lines above saying that
  1 defers to KJNodes' published count. Both were true and together they were
  misleading. It now says 1 means this node does not chunk, that nothing
  overrides a published count, and that 1 is not the lowest-peak setting:
  4 groups peaks at 2645 MiB against 2862 at 1, because chunking rules out
  the clone that only pays unchunked.

## 0.3.6

### Added

- `MiniMaxH3SigmaShift` (ModelSamplingMiniMaxH3) in all four shipped
  workflows, at the base checkpoint's own 12/3 so it changes nothing by
  default. It is there to be edited: the turbo LoRAs inherit the sampler's
  shift rather than carrying their own, and the 4-step v1.0 768p one was
  distilled at video shift 6. That is the variant whose training resolution
  matches our 1344x768 canvas, so it is the one people will reach for, and a
  graph with no shift node gives them nowhere to set it and no hint they
  needed to. Steps move with the LoRA too: 16 is a base-model number against
  these LoRAs' 4 or 8. Specs in `h3_config.SIGMA_SHIFT`, sourced from
  `coderef/Minimax-H3-Turbo`.
- The shifts joined the UI/API cross-check, for the same reason the
  checkpoint did: they are a free builder value the two formats can now
  disagree about, and a graph sampling off the wrong schedule renders
  cleanly rather than failing.

## 0.3.5

### Added

- A `unchunked_hands_over_ownership` case. sage-fork measured (`d59b82d`)
  that a slicing loop suppresses the release as completely at one group as at
  four -- the saving tracks whether the caller still holds the parents, not
  the group count. That makes the `n <= 1` branch's direct hand-over
  load-bearing rather than a shortcut for the trivial case, and unifying the
  two paths through `_chunked_heads` would turn -286 MiB into +572 with
  nothing visible in the output. The case drops the kernel's tensors and
  asserts the fused buffer actually goes; under that mutation it reports 0
  MiB freed of 336.

### Fixed

- The canvas sequence lengths in 0.3.1 were derived by fitting `rows*41 + 494`
  to the single known 41822, which also fits `rows*38 + 3518` -- the true
  decomposition, since the keyframe conditioning rows scale with the canvas
  just as the video rows do. The anchor was right and the three extrapolated
  canvases were 432 to 1296 tokens short. Re-measured at lengths computed
  from the model's own `PackedLayout`: the peak figures move slightly and the
  9.1% holds at every canvas. Surfaced by KJNodes' new `MiniMaxH3TokenCounter`
  (`6ab7e81`), which reads the layout from the model rather than inferring it.

## 0.3.4

### Fixed

- The clone is now off whenever the heads are chunked. `_chunked_heads` holds
  q, k and v across every group, so the kernel's per-group release frees
  nothing and the clone was a flat cost: at seq=41822, chunks=4 measured 3217
  MiB with it against 2645 without -- the clone's own 572 MiB recovered by
  nothing, and 69 MiB worse than chunking nothing and cloning nothing. Worst
  for the people least able to afford it, since KJNodes'
  `MiniMaxLowVRAMAttention` publishes `minimax_head_chunks` through
  `transformer_options` and our forward honours it with our own widget at 1.
  Introduced in 0.3.1 and shipped for the length of one session.
- The gate is code motion, not a new condition: it sits below the whole `n`
  resolution including the `transformer_options` read. Written against
  `head_chunks` where the clone used to be, it would read 1 in exactly the
  KJNodes case that motivated it -- confirmed by mutating it that way and
  watching only the options-route case go red.

### Added

- `bench/bench_minimax_attn.py` gained `--head-chunks` and
  `--chunks-via-options`, the latter delivering the value the way KJNodes
  does. Different code reaching the same `n`, and a fix verified only through
  our own argument is verified for the configuration nobody affected runs.
- A `chunked_path_does_not_clone` case covering both routes. Storage aliasing
  settles it exactly at 64 rows, so the peak numbers stay a one-off here and
  the check needs no threshold and no 8 GiB.

## 0.3.3

### Changed

- The clone is now gated on sage's own
  `sageattn_consume_prefers_cloned_v(device)` as well as the mode, and asked
  in the forward where `x.device` is real rather than at graph-patch time,
  where ComfyUI may not have loaded or cast the model yet and a multi-GPU
  box would bake an answer for the wrong card. The predicate exists because
  "does the release happen" and "should I clone" are the same question only
  until sage's fused-case peak drops below what a cloning caller can reach;
  gating on it means that flip arrives on upgrade instead of needing an edit
  here. A fork without the predicate keeps today's behaviour.
- `check_clone_v_wiring.py` gained a device-gate case. sm89 answers True, so
  a forward ignoring the predicate outright was invisible to the other three
  cases; forcing sage to disagree is the only way to see the gate from this
  arch, and it is what stops the file from grading our answer against itself
  now that the node reads the same predicate.

## 0.3.2

### Added

- `bench/check_clone_v_wiring.py`. The 9.1% peak saving depends on two
  things in different files joined by one argument in `nodes.py`, and
  dropping that argument changes no output, fails no existing check, and
  renders identical frames. Three cases: the predicate agrees with the
  kernel `build_kernel` actually returns, the node passes it down, and the
  flag really does take v out of the fused buffer. Each was shown red on its
  own mutation.

### Changed

- The `smooth_k=False` kwarg is now load-bearing for a second reason, noted
  where it is set: with `smooth_k=True` the K-mean copy eats the clone's
  saving and turns -286 MiB into +286 MiB.

## 0.3.1

### Changed

- The sage forward gives v its own storage before handing q/k/v to a kernel
  that releases them, cutting peak attention VRAM 9.1% at every canvas
  measured (-286 MiB at seq=41822, -174 MiB at seq=25406) for 0.7-1.0% more
  time. q, k and v are three views of one fused qkv buffer, so releasing q
  and k frees nothing while v still pins the allocation -- the same fix
  ComfyUI made upstream in `comfy/ldm/minimax/model.py`. Tied to
  `mode_releases_qkv(mode)`, not on outright: on the `fp16` mode, whose
  kernel holds q/k/v for the whole call, the same clone costs 571 MiB.

### Fixed

- `bench/bench_minimax_attn.py` drove the forward with `rope_freqs=None`,
  which takes the eager `q_norm`/`k_norm` branch. RMSNorm returns fresh
  tensors, so q and k were separate allocations and only v pointed into the
  fused qkv buffer -- not the aliasing the real inference path has, and
  aliasing is what the bench's peak column is about. It now builds a real
  rope table, and `--probe` reports the storage q, k and v land in so a null
  peak result can be told apart from a flag that did nothing.

## 0.3.0

### Added

- `MiniMaxH3ReferenceFit` node and `bench/check_reference_fit.py`. ComfyUI
  sizes reference images with `min(1.0, 2048/min(w,h))` where the reference
  pipeline has no clamp, so it never upscales: a 512x512 reference reaches the
  DiT with 1024 latent rows instead of 16384. The node does the resize itself,
  making the stock node's own resize a bit-identical no-op, and reports
  `latent_rows` because those rows are attended at every sampling step.
- `h3_rules.py`: the reference's input limits in one place -- trained aspect
  range, the 5-15s duration window checked *after* the frame-count snap, and
  the 17n+5 grid -- each citing where in `coderef/diffusers` it comes from.
- `head_chunks` on `MiniMaxH3SageAttention`, plus honouring the
  `minimax_head_chunks` key KJNodes publishes. Off by default; it exists so
  the head-chunk A/B could be run through our node at all.
- `bench/bench_e2e_h3.py` gained a canvas axis (`--canvases`, with `cheap`
  expanding to every ratio below the 16:9 default), a VRAM-knob axis
  (`--vram-arms`), and a sampled peak-VRAM column.
- `bench/check_solattn_correctness.py` and `bench/_sol_attn_reference.py`:
  the first independent correctness check the Sol-Attn Triton kernels have
  had. The reference is kijai/comfy-kitchen's pure-PyTorch eager
  implementation, vendored from the unmerged `sol_attn` branch (`ad9a4a8`)
  because no released wheel ships it; the registry import and the
  `comfy_kitchen::sol_attn` custom-op wrapper are stripped so importing it
  cannot collide with a future real one. Kernel agrees with the reference to
  cos 0.9995-0.9999 at T=512 and T=2048, bf16 and both INT8 paths, against
  upstream's own 0.998 bar. Includes a red control (kernel at one tau vs
  reference at 20x that tau, which must disagree) and a warning if the
  shipped tau is not actually sparsifying at the tested shape. Bounded by
  the reference being O(T^2): it refuses past 4 GiB of scores, so this
  checks the kernel's arithmetic, never its behaviour at H3's real length.

- `bench/check_workflow_schema.py`: validates saved UI workflows against a
  live `/object_info`. `build_workflows.py` only ever validated the API
  graphs, which carry no widget list and no slot table, so widget/socket
  confusion and link-table corruption were invisible to it. Calibrated by
  requiring a clean report on a graph ComfyUI itself wrote; two
  false-positive classes (widget-backed input slots, dynamic member slots)
  were found that way.
- `bench/check_lowvram_handoff.py`: asserts the sage forward accepts the
  single-item list KJNodes' low-VRAM block patch hands to attention.

### Fixed

- `head_chunks` moved after `patch_token_refiner` on `MiniMaxH3SageAttention`.
  Widget values map positionally in saved graphs, so a new widget at index 1
  landed an older graph's `patch_token_refiner=False` on an INT with `min=1`.
  The same hazard had been avoided one commit earlier on
  `MiniMaxH3KeyframeCanvas`'s output slots — the rule is that widgets are
  positional too, not just outputs. `bench/check_workflow_schema.py` caught
  this on its own, which is the first time a check added in this release went
  red on a defect that was not a deliberate mutation.
- `attention.py` no longer raises `AttributeError: 'list' object has no
  attribute 'shape'` when KJNodes' `MiniMaxLowVRAMAttention` is in the same
  graph. That node patches the *block* forward unconditionally to hand `x`
  over in a list, and only skips the *attention* forward if another patch
  owns the key -- so our forward received the list in either node order, and
  again from Sol-Attn's compose gate on calls it declines. The crash was
  outside the try/except, so it killed the render instead of degrading.
- `build_workflows.py` emitted Sol-Attn's `tau_profile` as a 13th widget
  value on a node with 12 widgets. It is declared `force_input=True`, which
  makes it a socket; the graphs now declare the socket and drop the value.
  Harmless in effect -- it sat past the end of the widget list, where
  LiteGraph drops it -- but the node carried a widget count no build of
  Sol-Attn has had.
- `build_workflows.py`'s own widget check counted `force_input` inputs as
  widgets, which is why it never saw the above. It also only flagged a
  shortfall, never a surplus.

### Changed

- `LONG_LENGTH` 362 -> 345. 362 frames is 15.083s against the reference's 15s
  ceiling, which it checks *after* the 17n+5 snap; 345 (14.375s) is the
  largest legal count. Breaks comparability with every recorded 362-frame
  measurement, and `h3_config.py` says which ones. `build_api` now refuses an
  illegal length or aspect rather than emitting the graph -- a comment did not
  stop 362 shipping for a week, since it is on the grid, inside ComfyUI's own
  3600 limit, and renders fine.
- `h3_config.py` records the head-chunk A/B it had been asking for since
  August. Head chunking frees 3227 MiB (three times the earlier estimate) and
  costs 0.2%; head and FFN chunking together use 1904 MiB *more* than baseline,
  so the two are antagonistic rather than additive. Noise floor stated: the
  baseline's own run-to-run VRAM spread is 784 MiB, which puts the FFN arm's
  +674 inside it and not evidence of anything.
- Workflows decode through `minimax_h3_video_vae_int8_convrot` instead of
  the fp16 VAE. Decode 12.8s -> 9.9s (1.29x) at 124 frames, 1344x768, and
  2300 MiB less resident. Quality measured on paired arms sharing a seed, so
  the latents are identical and every pixel difference is the decoder: 40.3 /
  41.8 dB against a different-seed control at 13.8 dB, and the temporal axis
  `h3_config.py` specifically warned about is flat (per-frame error std/mean
  0.063, frame-to-frame motion energy int8/fp16 = 0.998, where flicker reads
  above 1). Measured at 124 frames, not the 250+ this config usually runs.
- `bench_e2e_h3.py` reports `VAEDecode` time alongside sampler time, and
  `--video-vae` takes a list, crossing the VAE with `--arms` so a VAE
  comparison alternates instead of running as two invocations that share no
  thermal state. A VAE swap invalidates the sampler too, since
  `MiniMaxH3ImageToVideo` takes the same VAE for keyframe encoding.
- `attention.py` documents that it mirrors the stock forward's inference
  path only. Upstream grew a `comfy.model_management.in_training` branch
  using the non-in-place `rms_rope_split_half`; mirroring it would be
  theatre, since sageattn has no backward.
- `build_kernel` records why KJNodes' "pad V to CTA_K=128 in H3 mem-eff sage
  sm90" fix does not apply here: that bug comes from reimplementing sage's
  internals, and `sageattn_consume`'s fp8 dispatch excludes sm90.

## 0.2.0

### Added

- `workflows/h3_image_ref_plus_text_to_video_ref_lora.json` (and its `_api`
  copy): the shipped image-ref graph with one thing changed, loading fl2va
  plus Kijai's extracted ref LoRA instead of loading ref2va. Seed, prompt,
  canvas, length, sampler, sage and every Sol-Attn setting are shared with
  its sibling by construction, so anything that differs between the two
  renders is the LoRA.
- `REF_LORA` and `REF_LORA_STRENGTH` in `workflows/h3_config.py`.
- `unet`, `lora`, `out_prefix` and `title` parameters on `build_api` and
  `build_ui`. Checkpoint and task had to come apart: the new graph wants r2v
  conditioning driven by the fl2va checkpoint, which no task name describes.

### Changed

- Both workflow-emitting loops now read one `GRAPHS` table, so the
  difference between the shipped ref graph and its ref-LoRA sibling is a
  single `extra` dict and the two cannot drift apart in a dimension nobody
  intended.
- `cross_check` compares `UNETLoader.unet_name` and
  `LoraLoaderModelOnly.strength_model`, not only node presence. Until `unet`
  and `lora` became free builder parameters the checkpoint was derived from
  `task` inside both builders and the two formats could not disagree about
  it. Now they can, and the node-set check cannot see it, because both
  formats carry a `UNETLoader` either way. Both assertions were confirmed to
  fail when the formats are deliberately desynced.
- `coderef/` is ignored. It holds machine-local symlinks to sister repos
  rather than repo content.

The three existing graphs regenerate byte-identical.

## 0.1.0

Initial state, covering everything before the repo took an interest in the
ref LoRA.

### Added

- MiniMax H3 SageAttention node. Replaces each DiT block's attention
  `forward` with SageAttention's INT8-QK / FP8-PV kernel on Ada (sm89), and
  registers itself as the `optimized_attention_override` fallback so a
  sparse-attention patch layered on top composes with it instead of
  replacing it.
- MiniMax H3 Keyframe Canvas. Fits a keyframe to the target canvas before
  it reaches `MiniMaxH3ImageToVideo`, whose own resize stretches
  non-uniformly.
- MiniMax H3 Provenance Stamp, bench only. Records what a render's settings
  resolved to rather than what was typed, which `/history` already has. The
  field it exists for is `n_sparse`, the sigma window intersected with the
  sampler's schedule, which is readable from nothing else.
- Assert Sage Attention Chain. Turns a silently bypassed attention patch
  into a hard failure instead of a render that succeeds and is quietly
  slower or numerically different than intended.
- Benchmarks under `bench/`: module-level and end-to-end sage A/B,
  correctness check, keyframe-canvas check, override-routing check, and a
  smoke pass.
- Three generated workflows in `workflows/`, in UI and API format, built
  from one description so the two formats cannot describe different graphs.
- `workflows/h3_config.py` as the single source for checkpoint names,
  sampler settings, canvas geometry and Sol-Attn knobs, shared by the
  workflow generator and the bench. Both previously carried their own copy
  and those copies drifted.
- Sol-Attn interop evaluation in `docs/SOLATTN.md`, and H3 geometry notes in
  `docs/h3_geometry_and_nodes.md`.

### Changed

- Renamed from `ComfyUI-sageattn-ada` to `ComfyUI-h3-explorations`. Node
  ids were deliberately left alone: they are baked into every saved
  workflow's `type` field, and renaming one breaks every saved graph that
  uses it with no clear error in the UI.
