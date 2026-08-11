# Changelog

Semantic versioning. Nothing here has been tagged or published, so every
version below describes the state of the working repo rather than a release
artifact.

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
