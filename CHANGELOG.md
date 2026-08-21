# Changelog

Semantic versioning. Nothing here has been tagged or published, so every
version below describes the state of the working repo rather than a release
artifact.

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
