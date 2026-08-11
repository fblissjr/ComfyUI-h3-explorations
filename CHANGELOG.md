# Changelog

Semantic versioning. Nothing here has been tagged or published, so every
version below describes the state of the working repo rather than a release
artifact.

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
