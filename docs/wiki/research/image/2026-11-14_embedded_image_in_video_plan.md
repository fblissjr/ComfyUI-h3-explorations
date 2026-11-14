# Embedded Image Generation & Editing in H3 Video Workflows — Plan (Phases 0–4)

Status: **Phase 0B measured ([result](../../../../bench/results/2026-11-14_embedded_image_in_video_phase0b.md)) · Phase 1 implemented & live-measured on `:8188` ([result](../../../../bench/results/2026-11-14_embedded_image_in_video_phase1.md))** — see [`../../working_with_the_owner.md`](../../working_with_the_owner.md). Every node contract, branch condition, and file path below was cross-checked directly against the `.py` implementations, generated [`../../image/workflows_matrix.md`](../../image/workflows_matrix.md) (`95aa4447`), ComfyUI core (`comfy/ldm/minimax/`, `comfy/sd.py`, `comfy/model_base.py`), and live `:8188` measurements.

---

## 1. Architectural Premise & Current Repo Baseline

H3's single-frame image models and H3's AV video pipelines live in the same latent, tokenizer, and transformer family ([`../../h3_uses.md`](../../h3_uses.md), [`../../core/h3_frame_rules.md`](../../core/h3_frame_rules.md)):

- **Shared `Qwen3-VL-32B` bf16 text/vision encoder across video and image:** Since `7b7e185a`, [`workflows/h3_config.py`](../../../../workflows/h3_config.py) (`MODELS["clip"] = ENCODER_BF16`) wires `qwen3vl_32b_minimax_h3_bf16_pruned.safetensors` across 105 of 108 shipped workflows (every video and image graph except the decode-only graph and the two `te4b` P-H3 probes). Combining image and video stages in one ComfyUI graph shares **one** loaded `MiniMaxH3EncoderLoader` instance in memory with zero encoder reload.
- **Three natural image-stage tiers inside video graphs:**
  1. **Tier 1 — Pika `single_frame=True` (`t = 1`) on the active `fl2va` / `ref2va` video DiT (zero DiT swap):** [`MiniMaxH3Conditioning`](../../../../conditioning.py), [`MiniMaxH3EncodeReferences`](../../../../reference_encode.py) + [`MiniMaxH3PromptOnReferences`](../../../../reference_encode.py), and [`MiniMaxH3GuideLatent`](../../../../guide_latent.py) (`single_frame=True`) reuse the loaded video DiT + Pika LoRA directly (`shift_video=12.0, shift_audio=3.0` on `t = 1`; see [`../../image/h3_image_editing.md`](../../image/h3_image_editing.md)).
  2. **Tier 2 — PrunaAI `P-H3-Image` (`PH3_IMAGE_LORA` or merged `p-h3-image_int8_convrot.safetensors`):** Ranked LoRA (`rank=128` across all 50 DiT blocks + `token_refiner`, [`lora_branch.py`](../../../../lora_branch.py)) over `fl2va` using the 8-step `ManualSigmas` distillation grid (`PH3_IMAGE_MANUAL_SIGMAS`, `PH3_IMAGE_SHIFT`), generating at `8 MP` via [`MiniMaxH3ImageSize`](../../../../h3_image_nodes.py) (`shape="1:1"` on T2I / `"same as the picture"` on Edits, `a61f8db7`) -> [`MiniMaxH3PH3ImageConditioning`](../../../../ph3_image_nodes.py) (`[4 MP, 8 MP]` canvas/`image_1` clamp, `[1 MP, 8 MP]` secondary refs, decoupled `canvas='explicit'`, `seed=42` encoding, 8-image cap) -> [`MiniMaxH3ImageDownscale`](../../../../h3_image_nodes.py) delivering `4 MP` Lanczos (`bb914951`), plus [`MiniMaxH3PH3ImagePromptUpsampler`](../../../../ph3_image_nodes.py). Teacher-compatible with the shared 32B [`MiniMaxH3EncoderLoader`](../../../../h3_encoder_loader.py) ([`../../image/p_h3_image.md`](../../image/p_h3_image.md), [`../../image/workflows_matrix.md`](../../image/workflows_matrix.md)).
  3. **Tier 3 — Standalone `MiniMax-H3-Image` (`minimax_h3_image_pruned_int8_convrot.safetensors`):** Separate DiT load when flagship still quality / 16-ref identity fidelity justifies a model swap, using [`MiniMaxH3ImageSize`](../../../../h3_image_nodes.py) (`4.0 MP` floor, e.g. `16 MP` generate -> `4 MP` deliver via [`MiniMaxH3ImageDownscale`](../../../../h3_image_nodes.py)), [`MiniMaxH3ImagePromptExpander`](../../../../h3_image_nodes.py), and [`MiniMaxH3EncodeImageReferences`](../../../../reference_encode.py) + [`MiniMaxH3PromptOnImageReferences`](../../../../reference_encode.py) (16-still KV-prefix cache; [`../../image/h3_image.md`](../../image/h3_image.md)).

### Core Mechanical Laws Governing Any Embedded Image <-> Video Graph

1. **Single-frame bypass vs. `17n + 5` AV windows:** `t = 1` (`SINGLE_FRAME` in [`h3_rules.py`](../../../../h3_rules.py)) bypasses `17n + 5`. Any multi-frame AV window must land on `17n + 5` (`5..362`), and [`MiniMaxH3Conditioning`](../../../../conditioning.py) (`keyframe_canvas.py:181-189`) enforces `duration_in_range(length)` (`124..362`, `5..15s` at `24 fps`), making **`141`** (`5.875s`) the shortest dual-clock `on_both_clocks` length (`39 + 51m`: `39, 90, 141, 192, 243, 294, 345`) for `i2va`/`fl2va`/`l2va` keyframe runs (`24.0 fps` video, `40 Hz` audio; see [`../../core/h3_frame_rules.md`](../../core/h3_frame_rules.md) and [`bench/results/2026-11-14_embedded_image_in_video_phase0b.md`](../../../../bench/results/2026-11-14_embedded_image_in_video_phase0b.md)).
2. **Mid-clip keyframes land on `17m`:** Video VAE encodes `[1, 4, 4, 4, 4]` temporal latent spans (`comfy/ldm/minimax/model.py`), so `frame_idx = 17m` (`0, 17, 34, 51, ...`) targets `1`-frame standalone clip starts rather than smeared `4`-frame chunks.
3. **Pixel-domain handoff across image and video stages:** Route `VAEDecode` RGB `IMAGE` tensors from the image stage into `MiniMaxH3Resolution` sizing and `MiniMaxH3Conditioning` / `MiniMaxH3EncodeReferences` / `MiniMaxH3GuideLatent`. Never pass single-frame `H3ImageVAE` latents directly into a multi-frame video decode/encode path (Phase 0A).

---

## 2. Phase 0 — Zero-Risk Baseline Plumbing & Immediate Workflows

Phase 0 requires **zero custom-node code changes** — it enforces VAE/conditioning hygiene and proves the first three embedded-image patterns on existing nodes.

### Phase 0A — Explicit Dual-VAE Split & Precision Separation

- Since `0b566413`, all 18 image graphs load `IMAGE_GRAPH_VAE` (`p-h3-image_vae_int8_convrot.safetensors`, [`workflows/h3_config.py`](../../../../workflows/h3_config.py)), while video graphs load `MODELS["video_vae"]` (`minimax_h3_video_vae_int8_convrot.safetensors`, or `VIDEO_VAE_FP16` — precision/timing trade-offs recorded in [`bench/results/2026-09-26_vae_decoders_345f.md`](../../../../bench/results/2026-09-26_vae_decoders_345f.md)).
- **`PH3ImageVAE` / `H3ImageVAE` cannot decode or encode multi-frame video:** `PH3ImageVAE` (`comfy/ldm/minimax/vae.py:1036-1042`) inherits `H3ImageVAE.decode` (`comfy/ldm/minimax/vae.py:1030-1033`), which explicitly raises `ValueError` whenever `z.shape[2] != 1`. `H3ImageVAE.encode` (`comfy/ldm/minimax/vae.py:1022-1028`) injects diagonal-Gaussian noise with fixed `PH3_IMAGE_ENCODE_SEED = 42` (`seed=42`) and performs no multi-frame chunking.
- **Combined-graph rule:**
  - Wire **`image_vae`** (`IMAGE_GRAPH_VAE`, optionally paired with [`MiniMaxH3VAEPrecision`](../../../../vae_precision.py) for `fp32` encoder separation; noting `comfy/sd.py:1090, 1170-1186` load-time `working_dtypes`) strictly to the `t = 1` image stage's `VAEEncode` / `VAEDecode`.
  - Wire **`video_vae`** (`MODELS["video_vae"]` / `VIDEO_VAE_FP16`) strictly to `MiniMaxH3Conditioning`, `MiniMaxH3EncodeReferences`, `MiniMaxH3GuideLatent`, and final video `VAEDecode`.
  - Resize decoded `IMAGE` stills (including Pika `single_frame=True` edit graphs) with [`MiniMaxH3Resolution`](../../../../resolution.py) (`shape="same as the picture"`, `picture=...`, appended in `a61f8db7`; `1344x768` trained-family cap) and budget tokens via [`MiniMaxH3Preflight`](../../../../preflight.py) (`<= 99,864` packed tokens for signed `int32` QKV safety).

### Phase 0B — Immediate Embedded-Image Workflows (Existing Nodes Only — [Measured `2026-11-14`](../../../../bench/results/2026-11-14_embedded_image_in_video_phase0b.md))

1. **In-graph start / end frame synthesis (`t = 0` + `t = frame_count - 1` -> `fl2va`):**
   Generate `first_frame` (and optionally edit/synthesize `last_frame` conditioned on `first_frame` via `P-H3-Image` or Pika `t = 1`) -> `VAEDecode` -> scale to video canvas -> `MiniMaxH3Conditioning` (avoiding core's lone-`last_frame` cover-crop bug).
2. **2D Image Edit -> Continuous Control Vector (`No Pixel-Pinning`):**
   Take a source frame (`image_a`), edit it once in 2D (`image_b`: relight, weather, lens/grade, style shift), feed `(image_a, image_b)` into [`MiniMaxH3MakeMediaControlVector`](../../../../control_vector_nodes.py), and wire the resulting `H3ControlVector` into [`MiniMaxH3Steer`](../../../../control_vector_nodes.py).
   - **Mandatory Phase 0B guards on [`MiniMaxH3MakeMediaControlVector`](../../../../control_vector_nodes.py):**
     - Set `overwrite=True` when iterating (`control_vector_nodes.py:3263-3267` defaults to `overwrite=False` and refuses to overwrite an existing output path).
     - Wire the **LoRA-stacked `model` branch** (`lora_branch.py:725-726` attaches `h3_lora_branches`, consumed by `control_vector_nodes.py:305-334, 601-625` so extraction runs in LoRA-aware post-refiner space).
     - Set `MiniMaxH3Steer(start_percent=0.0)`: `start_percent` gates **denoising step fraction**, not video timeline frames `0..T`. Under `shift_video=12.0` on 6-step Pika, delaying `start_percent` skips early flow steps where global lighting and composition form. Treat this as a **clip-wide transform/mood dial** (`strength`, `end_percent`).
3. **2D Local Edit Pixel-Composite -> Start Frame / Released Reference:**
   Compute `diff = |edited - orig|` against a dilated `CHANGE_THRESHOLD = 0.05` pattern ([`video_mask.py:301`](../../../../video_mask.py#L301)), alpha-composite edited subject/prop pixels cleanly over `orig` on frame `0` (preserving exact untouched pixels before video encoding), OR pass `edited` as `<Picture N>` (`ref_img` at `t < 0`) released mid-denoise via [`MiniMaxH3ReleaseReferences`](../../../../control_vector_nodes.py) (`video_end_percent = 0.33` on 6-step Pika).

---

## 3. Phase 1 — Temporary Mid-Clip Steering (`guide_end_percent`) — **Verified on `:8188` ([Measured `2026-11-14`](../../../../bench/results/2026-11-14_embedded_image_in_video_phase1.md))**

### Problem Statement

Using an in-graph edited frame at mid-clip `frame_idx = 17m` with [`MiniMaxH3GuideLatent`](../../../../guide_latent.py) previously pinned pixels tightly all the way to step 0, causing trajectory freezes or hard "snap-through" motion when a still didn't match moving dynamics — and blocked [`EditGuider`](../../../../edit_guidance.py) original-branch matching whenever `orig` already carried `first_frame` / `last_frame` keyframes (`edit_guidance.py:121-123` previously required `same_text and kf_orig == 0` in `_release_to_original_after`).

### Deliverable

Implemented in [`control_vector_nodes.py`](../../../../control_vector_nodes.py) and [`edit_guidance.py`](../../../../edit_guidance.py):

- Appended optional **`guide_end_percent`** (`default=1.0`, `MiniMaxH3ImageModel` refusal when `< 1.0`) to [`MiniMaxH3ReleaseReferences`](../../../../control_vector_nodes.py), allowing mid-clip `17m` `GuideLatent` guides (`0 < resolved_frame_index < total_frames - 1`) to steer coarse motion and composition early in denoising (`0.0 -> min(end_percent, guide_end_percent)`) and drop out of `PackedLayout` for clean late-step dynamics and boundary blending while preserving `first_frame` (`0`), `last_frame` (`total_frames - 1`), `<Picture N>` / `<Video N>` refs (or slicing both simultaneously when `video_end_percent` is also lowered), and timeline `cond_audio` islands.
- Generalized [`EditGuider.set_conds`](../../../../edit_guidance.py) (`_keyframes_extend_prefix`) so `same_text` conditionings where `edited` appends extra keyframes onto `original`'s identical prefix keyframes (including `i2va`/`fl2va` bookends) also set `_release_to_original_after = True` and hand back to 1-pass `[original]` after `last_step`.
- **Live 6-step Pika quantization note (`PIKA_MANUAL_SIGMAS`, `n_steps = 6`):** [`_step_in_window`](../../../../control_vector_nodes.py#L1152-L1188) tests `step_idx / n_steps < end_percent`, so step `2` lands at `2 / 6 = 0.3333...`. Set `guide_end_percent = 0.33` for a `2/6`-step (`[0, 1]`) release at unshifted midpoint `b = 0.50` (`0.34` matches `0.50` at `3/6` steps `[0, 1, 2]`). Live `:8188` captures verified both `1008`-row `PackedLayout` mid-clip release on `MiniMaxH3ReleaseReferences` (`guide_end_percent = 1.00 / 0.50 / 0.33`, zero `67->68->69` seam, `4.3x` audio lift when paired with soundscape-aligned prompting) and live `EditGuider._keyframes_extend_prefix` auto-release (`scale = 1.65`, `last_step = 2`; see [`bench/results/2026-11-14_embedded_image_in_video_phase1.md`](../../../../bench/results/2026-11-14_embedded_image_in_video_phase1.md)).

### Five Mandatory Implementation Guards (`control_vector_nodes.py:1381-1715`)

Every guard below was verified against [`control_vector_nodes.py`](../../../../control_vector_nodes.py), [`guide_latent.py`](../../../../guide_latent.py), [`sound_cues.py`](../../../../sound_cues.py), and `comfy/model_base.py` / `comfy/ldm/minimax/model.py`:

1. **Derive `total_frames` from `target_sig[1]` via `FRAME_PER_TOKEN`:** `payload` does not carry a `frame_count` key. Compute `total_frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(target_sig[1]))` matching `comfy/model_base.py:2314-2319`.
2. **Protect both endpoints (`first_frame` AND `last_frame`):** `first_frame` (`0`), `last_frame` (`total_frames - 1`), and mid-clip `GuideLatent` guides all pack into `PackedLayout.segments` with `kind == "cond"` (`comfy/ldm/minimax/model.py:395-525`). Slice strictly **`0 < resolved_frame_index < total_frames - 1`** — never `resolved_frame_index > 0`, which would silently drop `last_frame`.
3. **Filter to visual keyframes only before indexing `cond` segments:** Audio-only [`MiniMaxH3GuideLatent`](../../../../guide_latent.py) entries (`guide_latent.py:321-343`) and [`MiniMaxH3SoundCues`](../../../../sound_cues.py) entries (`sound_cues.py:467-471`) live in `payload["keyframes"]` without an image `latent`. Build the keep-mask strictly over `[kf for kf in (payload.get("keyframes") or ()) if isinstance(kf, Mapping) and kf.get("latent") is not None]` so audio keyframes never desync positional `cond` segment indices.
4. **Apply the identical keep-mask across all three parallel structures:** Apply the exact same boolean keep list to:
   - `orig_layout.segments` where `seg.kind == "cond"`,
   - `_strip_cond_entries(...)`, and
   - `orig_vid_lats[:n_kf_vis]` (`control_vector_nodes.py:1691-1715`),
   so paired `guide + mask` keyframes (`guide_latent.py:345-353`) stay element-aligned when dropped.
5. **Include `drop_guide` in the `released_layout_cache` key:** Add `drop_guide` to the tuple key at `control_vector_nodes.py:1564-1567` so cached `PackedLayout` objects cannot collide between reference-only release and reference+guide release steps.

**Runtime note for `EditGuider` work (`edit_guidance.py`):** In live sampling, [`EditGuider.predict_noise`](../../../../edit_guidance.py) executes the flattened `packed_split` 2D/3D branch (`edit_guidance.py:209-236`; scalar short-circuits at `edit_guidance.py:219-222`; exercised by [`bench/check_edit_guidance.py:215-232`](../../../../bench/check_edit_guidance.py#L215-L232)), **not** `is_nested`. Keep both `EditGuider` optional mask wiring ([`../../next_steps.md`](../../next_steps.md)) and Phase 1 guide-release changes aligned to `packed_split`.

---

## 4. Phase 2 — Scout-Edit-Fork Pipelines, Diff Masking, & Split-KV Caching

Once Phase 1 (`guide_end_percent`) passes tests, build reusable embedded-image video patterns and workflow builders:

### Phase 2A — Two-Stage "Scout Draft -> Image Edit -> Forked Render"

Generalizes the turtle-hopping counterfactual loop:
1. **Scout Pass:** Run a fast 6-step Pika draft (or reuse per-step `x0` / decoded keyframes) on base prompt + conditioning.
2. **In-Graph Image Fork (`17m`):** Pick a decisive `17m` frame (or `last_frame`), edit it via Tier 1 (`Pika single_frame=True`) or Tier 2 (`PH3_IMAGE_LORA`), keeping clean background pixels via Phase 0B-3 / Phase 2B.
3. **Guided Re-Render:** Fork into the final pass using **either**:
   - [`MiniMaxH3GuideLatent`](../../../../guide_latent.py) (`17m`) + Phase 1 `guide_end_percent` release; **or**
   - [`MiniMaxH3PromptEdit`](../../../../edit_guidance.py) + [`MiniMaxH3EditGuider`](../../../../edit_guidance.py) (`orig + scale * (edited - orig)` with `first_step` delay and `both` / `video_only` / `audio_only` / `independent` `audio_scale`) to steer the branch without snapping geometry; **paired if needed** with [`MiniMaxH3AudioRefineMask`](../../../../audio_refine.py) + [`MiniMaxH3FrozenVideoCache`](../../../../frozen_video_cache.py) when keeping/re-tuning audio independently.

### Phase 2B — Edit-Diff Auto-Masking (`|edited - orig|` -> `MiniMaxH3MaskedSource`)

- Compute `|edited - orig|` relative to the unedited scout/source frame, threshold & morph-dilate (aligned with `CHANGE_THRESHOLD = 0.05` in [`video_mask.py:301`](../../../../video_mask.py#L301)), and project across the target clip span into [`MiniMaxH3MaskedSource`](../../../../video_mask.py).
- **Why [`MiniMaxH3MaskedSource`](../../../../video_mask.py) and NOT [`MiniMaxH3GuideLatent`](../../../../guide_latent.py):** `MiniMaxH3GuideLatent.mask` (`guide_latent.py:345-353`) only encodes a second B&W RGB silhouette keyframe for paired control LoRAs (`minimax_h3_lms_v1.0`) and attaches **no `noise_mask`** (so stock `fl2va`/`ref2va` still resamples unmasked background). [`MiniMaxH3MaskedSource`](../../../../video_mask.py) properly max-pools masks over `[1, 4, 4, 4, 4]` x `32x32` latent tokens and sets `NestedTensor(video, audio=0)` audio freeze.
- Pair [`MiniMaxH3RestorePlate`](../../../../plate_restore.py) (`plate_restore.py:11-26`) **only** when running a **two-sampler** masked refinement pass to repair `inverse_noise_scaling` plate drift — never in a single sampler.

### Phase 2C — Prompt Expansion & Split-KV Reference Caching

- Pair [`MiniMaxH3ImagePromptExpander`](../../../../h3_image_nodes.py) (`heylook` SSE) or [`MiniMaxH3PH3ImagePromptUpsampler`](../../../../ph3_image_nodes.py) upstream of image generation.
- For interactive multi-prompt video + still iteration on `<Picture N>` / `<Video N>`, use [`MiniMaxH3EncodeReferences`](../../../../reference_encode.py) + [`MiniMaxH3PromptOnReferences`](../../../../reference_encode.py) (and [`MiniMaxH3EncodeImageReferences`](../../../../reference_encode.py) + [`MiniMaxH3PromptOnImageReferences`](../../../../reference_encode.py) for `MiniMax-H3-Image`) so Qwen3-VL vision/KV prefixes compute once.

---

## 5. Phase 3 — Long-Horizon Anti-Burnout Loops & Temporal Ramps

Multi-window chains accumulate contrast saturation, softness, and trajectory stagnation over repeated context handoffs.

### Phase 3A — Window-Tail Keyframe Perturbation on `MiniMaxH3FreezeAudioWindow`

- **Start on [`MiniMaxH3FreezeAudioWindow`](../../../../audio_freeze.py), NOT [`MiniMaxH3AudioFreezeSong`](../../../../audio_freeze_song.py):** Inspecting `audio_freeze_song.py:524-634` confirms [`MiniMaxH3AudioFreezeSong`](../../../../audio_freeze_song.py) currently has **no `first_frame` or external `keyframes` socket**. Prove the loop mechanism cleanly on single-window / chained [`MiniMaxH3FreezeAudioWindow`](../../../../audio_freeze.py) graphs first.
- **Anti-burnout protocol (on `on_both_clocks` grids `141..345`, contexts `39` or `90`):**
  - Extract clean tail frame(s) of Window $k$ -> apply a controlled 2D image edit (relight reset, detail restoration, camera-angle/pose nudge, or introduced `<Picture N>` subject prop) -> pass the edited still either as a released `<Picture N>` / temporary `17m` guide (`guide_end_percent`) into Window $k+1$, refreshing color/texture statistics without hard-cutting `first_frame` motion continuity.

### Phase 3B — Expose Per-Window Keyframe/Ref Hooks on `MiniMaxH3AudioFreezeSong`

- Only after Phase 3A shows clear visual lift on multi-window blind panels:
  - Append optional per-window image-edit / keyframe hook inputs to **the end** of [`MiniMaxH3AudioFreezeSong`](../../../../audio_freeze_song.py)'s schema (strictly obeying `AGENTS.md §1.5`: never insert into middle of inputs/outputs or rename `node_id=`).

### Phase 3C — True Temporal Frame-Ramp Steering (Future Hook/Guider Research)

- Because [`MiniMaxH3Steer`](../../../../control_vector_nodes.py) `start_percent`/`end_percent` gates **solver step fraction** (uniform across all video frames), achieving a true frame-index ramp ("0% at frame `0` -> 100% at frame `243`") requires scaling steering strength per temporal token index inside a dedicated custom guider/model hook. Deferred to Phase 3C behind Phase 1–3A delivery.

---

## 6. Explicit Non-Starters & Architectural Traps (Avoid List)

| Trap / Anti-Pattern | Why It Fails in Code | Correct Replacement |
| :--- | :--- | :--- |
| Decoding/encoding multi-frame video through `IMAGE_GRAPH_VAE` (`PH3ImageVAE`) | `comfy/ldm/minimax/vae.py:1030-1033` raises `ValueError` on `z.shape[2] != 1`; `encode` (`1022-1028`) injects `seed=42` noise without temporal chunking | Keep **`image_vae`** strictly on `t = 1` stills and **`video_vae`** (`MODELS["video_vae"]`) on all AV video paths; hand off decoded RGB `IMAGE` tensors |
| Feeding `|edited - orig|` into `MiniMaxH3GuideLatent.mask` on stock `fl2va`/`ref2va` | `guide_latent.py:345-353` encodes a 2nd RGB keyframe for paired control LoRAs and attaches **no `noise_mask`**; background still resamples | Feed dilated diff mask across `T` into [`MiniMaxH3MaskedSource`](../../../../video_mask.py) (`CHANGE_THRESHOLD = 0.05` at `video_mask.py:301`) |
| Wiring stock `SetLatentNoiseMask` onto H3 AV latents | Replaces `NestedTensor(video, audio)` mask with flat tensor, padding `1`s onto audio (silently regenerating soundtrack) and corrupting 4-frame causal chunks | [`MiniMaxH3MaskedSource`](../../../../video_mask.py) / [`MiniMaxH3FreezeAudio`](../../../../audio_freeze.py) (`NestedTensor(video, audio=0)`) |
| Using `MiniMaxH3RestorePlate` in a single-sampler pass | `plate_restore.py:11-26` only fixes `inverse_noise_scaling` plate drift between **two chained samplers** | Pixel-composite clean regions before single-pass encoding, or reserve [`MiniMaxH3RestorePlate`](../../../../plate_restore.py) for two-sampler chains |
| Using `MiniMaxH3ReferenceNoise(clean_level=0.70)` to soften refs | `reference_noise.py:47-54` leaves late-step noise-vs-`seg_t` mismatch (`comfy/ldm/minimax/model.py:746-750`) | Keep clean visual refs (`0.999`) and release mid-denoise with [`MiniMaxH3ReleaseReferences`](../../../../control_vector_nodes.py) |
| Using `MiniMaxH3Steer(start_percent=0.4)` as a video-frame fade | `start_percent` gates **denoising step fraction**, skipping early structural steps under `shift_video=12.0` on 6-step Pika | Keep `start_percent=0.0` for clip-wide look/mood strength; defer true temporal ramps to Phase 3C |
| Slicing `[Shot N]` multi-panel still renders into sequential video frames | Documented unreliable in [`docs/wiki/image/h3_image.md:416-426`](../../image/h3_image.md#L416-L426) | Generate keyframes sequentially / with single-frame `<Picture N>` references (`t = 1`) |
| Running `MiniMaxH3MakeMediaControlVector` on base un-LoRA `model` or with default `overwrite=False` | Overwrites fail silently/refuse (`control_vector_nodes.py:3263-3267`); base model lacks `h3_lora_branches` (`lora_branch.py:725-726`, `control_vector_nodes.py:305-334, 601-625`) | Pass `overwrite=True` and wire the [`MiniMaxH3LoRABranch`](../../../../lora_branch.py) `model` output |
| Dropping `cond` blindly via `idx > 0` or reading `payload["frame_count"]` | Drops `last_frame` (`total_frames - 1`), crashes on missing key, or desyncs audio cues & paired `mask` keyframes | Compute `total_frames` via `FRAME_PER_TOKEN` (`comfy/model_base.py:2314-2319`), slice `0 < idx < total_frames - 1`, filter `kf.get("latent") is not None`, sync all 3 structures, key cache with `drop_guide` |

---

## 7. Phase 4 — Verification Gates & Empirical Acceptance Protocol

All code and workflow phases (Phases 1–3) ship only after passing both static invariants and controlled blind render evaluations ([`../../ops/checks.md`](../../ops/checks.md), [`../../ops/evidence.md`](../../ops/evidence.md)):

1. **Schema, node-ID, & generated-graph hygiene (every Phase 1–3 change):**
   - Never hand-edit `workflows/**/*.json` or `INDEX.md` files; regenerate via [`workflows/build_workflows.py`](../../../../workflows/build_workflows.py) (`workflows/h3_config.py`) and [`bench/build_index.py`](../../../../bench/build_index.py).
   - Run static suite (`$COMFY/.venv/bin/python`):
     - `bench/check_node_ids.py` & `bench/check_literal_widgets.py` (strictly append-at-end schema discipline)
     - `bench/check_edit_guidance.py` (`packed_split` live branch verification)
     - **New Phase 1 unit tests** asserting all 5 `guide_end_percent` guards: preserves `first_frame` (`0`) and `last_frame` (`total_frames - 1`), drops mid-clip `17m` keyframes after cutoff, keeps segment alignment when audio-only `SoundCues` / `GuideLatent` entries exist, keeps paired `guide + mask` latents synchronized, and isolates `released_layout_cache` entries on `drop_guide`.
     - `bench/check_no_owner_paths.py`, `bench/check_doc_links.py`, `bench/check_doc_inventory.py`, `bench/build_index.py --check`, and `bench/check_commit_ready.py`.
2. **Single-variable render & blind A/B checks (ComfyUI `:8188` captures):**
   - Stamp every capture with [`MiniMaxH3ProvenanceStamp`](../../../../provenance.py) and test strictly against a matched-seed single-variable control (Pika baseline vs. +Phase 0B start/end image, vs. +`MakeMediaControlVector`, vs. +Phase 1 `17m` `guide_end_percent` release, vs. +Phase 3A `MiniMaxH3FreezeAudioWindow` anti-burnout chain on `141..345` `on_both_clocks` grids).
   - Inspect renders with the Native-First Gate (`read_file` on `.1fps.<first>-<last>.mp4` strips and `< 100 MB` clips first), measure wall-clock & peak VRAM (`bench/results/`), and gate workflow promotion on blind side-by-side panels confirming: zero VAE mismatch crashes, clean audio preservation, no end-frame freeze/pop artifacts around `17m` release, and reduced burnout drift over 2+ chained windows.
