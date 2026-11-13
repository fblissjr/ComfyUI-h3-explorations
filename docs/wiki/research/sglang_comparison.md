# sglang's H3 serving path against ours

last updated: 2026-11-04 (dated notes where the retired `FASTH3_*` constants are named); 2026-11-04 ("Every known divergence" now holds the list itself, re-checked, after the owner retired `../comfyui_vendor_gaps.md`); 2026-11-04 (closing section "Eleventh read" added: a month of the pipeline, stage by stage; drifted ComfyUI core line citations re-checked against core `5f415e31a` and repointed to symbols or current lines, no core claim found false); 2026-11-04 (every section before "Tenth read" re-checked against sglang `origin/main` at `8884b4b421`: dated notes where a claim moved, drifted line citations repointed to symbols, and the worker-patch pin in "Tenth read" checked by hash); 2026-11-04 (closing section "Tenth read" added, with dated notes in the seventh, eighth and ninth reads); 2026-10-08 (closing section "Ninth read" added: frozen audio
and video inputs, with a dated note in the eighth read); 2026-10-02 (closing section "Eighth read" added, with dated
notes on the seventh read's ComfyUI sentence, the `kitchen_int8` name and our
dense-blocks cell); 2026-09-25 (closing section "Seventh read" added); 2026-09-19 (closing section "Sixth read" added, and a dated
note under the Sol-Attn defaults table); 2026-09-11 (subsection "Sol-Attn
defaults: sglang against core's and ours" added under "What we do that they
do not", and closing section "Fifth read" added; "Fourth read" added 2026-09-10; one subsection under "What
they do that we do not" and the section "Third read" added 2026-09-04;
everything else is the 2026-08-29 read)

What the vendor-side serving implementation does that this install does not,
what both do where ours may be the weaker version, and what looks like a gap
until you check the record and find it already priced.

**Read from source 2026-08-21** against `coderef/sglang` at commit
`a41da991c8`, alongside ComfyUI's `comfy_extras/nodes_minimax_h3.py`,
`comfy/ldm/minimax/model.py`, `comfy/model_management.py` and this repo's own
bench records. No renders were made for any of it. Every claim below says
which file it came from; nothing here is a measurement on this card unless it
cites a record in `bench/results/`.

**The clone has moved twice since that read, and this is the file's standing
hazard.** `coderef/sglang` was at `a7ec6b97f7` on 2026-08-22 and is at
`97781eb7f3` (2026-08-29) now. The prose sections below were read at
`a41da991c8`; only the sections dated later were read at a later commit.
`bench/check_doc_links.py` confirms every cited line still exists, which is not
the same as confirming it still says the same thing. Re-read before quoting an
older section as current.
*Note 2026-11-04: the clone's checkout is `16dd97df61` (2026-11-02) and its
`origin/main` is `8884b4b421` (fetched 2026-11-04). Every section was
re-read against `origin/main` that day; where a `path:line` citation had
drifted and the claim held, the citation now names the symbol instead, and
where the claim moved, a dated note says what is true at `origin/main`.*

**Why this file is not merged into the pipeline walk, asked 2026-08-29.**
Because the two drift for different reasons and on different clocks.
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) is a source read of somebody
else's tree: it goes stale when **their** code moves, and the fix is to re-read
at a new commit. This file goes stale when **ours** moves — a ComfyUI upgrade,
a node change, a new measurement here — and the fix is to re-derive against our
side. Merging them would produce one file that needs both kinds of maintenance
and signals neither, and would put a 700-line vendor description in front of
every reader who only wanted the delta. The split is kept. What was actually
drifting is fixed below and in
[`../comfyui_vendor_gaps.md`](../archive/comfyui_vendor_gaps.md).

**The pipeline itself, before the comparison.** Since 2026-08-25
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) is the stage-by-stage walk of
sglang's H3 path: request, time grid and canvas, media ingestion, the Qwen3-VL
encode, the VAE encodes, the packed sequence, the DiT forward, the denoise
loop, decode and output, the runtime around it, and a numbered insights
section. It compares nothing; this file does. Read it first when the question
is "what does sglang actually do", and this one when it is "how does that
differ from here".

**Scope, and what the index is for.** This file owns the *optimization and
runtime* comparison. The reference-conditioning comparison — sizing, patchify,
presentation, packing, condition timestep — is owned by
[`h3_references.md`](../core/h3_references.md), section "The vendor image path,
stage by stage", and its detail must not be restated here.

That split kept every divergence documented and left none of them listed
together. The consolidated snapshot that fixes it is
[`comfyui_vendor_gaps.md`](../archive/comfyui_vendor_gaps.md), which defers to this
file and to [`h3_references.md`](../core/h3_references.md) rather than competing
with them.

---

## Every known divergence, in one place

This is the list. Until 2026-11-04 it lived in
[`../comfyui_vendor_gaps.md`](../archive/comfyui_vendor_gaps.md), which the owner
retired that day: it is kept, frozen, so that pointers to its gap numbers
still resolve, and its sections hold the detail and the measurements behind
each row. The rows below were re-checked on 2026-11-04 by reading code, not
by running anything, at ComfyUI core `5f415e31a`, sglang `c07df5a055`,
diffusers `b674fd8c2` and the release repository `199b7d0`; every figure
behind a row still rests on its original record. Gap numbers are the old
document's, so "gap 16" means the same thing in both. Where this list and
any other page disagree, this list is right.

The list exists because the ownership rule below is correct and had a cost:
the gaps were documented across several files and none of them were listed
together.

| gap | what differs | status at 2026-11-04 | what this repo does | where the detail is |
|---|---|---|---|---|
| 1 | Core's bundled Qwen tokenizer lacked the release's seven extra special tokens (`<d>`, `</d>`, `<\|cutoff\|>`, lyrics and caption markers), so dialogue prompts tokenized into fragments. | fixed in core at `924743af0` (#15808): `comfy/text_encoders/minimax.py::MINIMAX_EXTRA_TOKENS`, added by `MiniMaxQwenSDTokenizer` | nothing to patch; `bench/audit_h3_marker_tokenization.py` requires the native fix and checks it against the release tokenizer (`vendor_config.py`, guarded by `bench/check_vendor_config.py`) | `../comfyui_vendor_gaps.md` section 1; `official_weights_metadata.md` |
| 2 | sglang forces the reference video to 24 fps in its ffmpeg decode (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_decode_reference_video_frames`, the `fps=` filter); core assumes 24 (`comfy_extras/nodes_minimax_h3.py::FPS`) and has no fps input, so the 2 fps subsample and its timestamps are wrong at any other rate. | open in core; fixed here by `MiniMaxH3AppendRefVideo` (requires `VHS_VIDEOINFO.loaded_fps`) and `reference_conditioning.py::_resample_video_to_24fps` | typed path normalizes to 24; shipped graphs also keep `force_rate=24`; `bench/check_ref_prompt_labels.py` and `bench/check_reference_runtime.py` | `../comfyui_vendor_gaps.md` section 2; `../h3_references.md`, "Frame rate: the one that bites silently" |
| 2b | Core's forced-hook branch of `CLIP.encode_from_tokens_scheduled` dropped the encoder's third return value, so `minimax_token_tags` was lost when a CLIP hook was wired. | fixed in core at `6bfaacc67` (#16400): `comfy/sd.py::CLIP.encode_from_tokens_scheduled`, `if len(o) > 2: pooled_dict.update(o[2])` | nothing; no shipped graph wires a CLIP hook; `bench/check_reference_contracts.py` case 5c retired | `../comfyui_vendor_gaps.md` section 2b; `conditioning_nodes.md`; `../wiki/decisions/2026-09-25_to_2026-09-29.md` |
| 3 | Core leaves the still-image floor at the shared Qwen helper's default (`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images`, `min_pixels`) where the release enlarges anything under its own floor (`vendor_config/preprocessor_config.json`, `size.shortest_edge`). | open in core; on this repo's default still sizing the floor is not reached (reasoned from the schema, not run): `qwen_view=separate` scales the encoder's copy to `h3_rules.REF_QWEN_SHORT_EDGE` in both directions (`reference_conditioning.py::qwen_view_size`), and `shared` with `size_policy=max, allow_upscale` raises it to `dit_short_edge` | `bench/preflight_graph.py` reports when the release floor would enlarge a still (reads `vendor_config.image_pixel_bounds`); no runtime parity rule | `../comfyui_vendor_gaps.md` sections "3 and 4" and "3. The floor"; `../h3_references.md`, "The vendor image path, stage by stage" |
| 4 | Above core's helper ceiling the encoder's copy is resized while the VAE keeps the full one, so the two towers see different resolutions; the release ceiling is higher and, for video, clip-wide. | open in core. Here the towers now differ by design, not only at the ceiling: `qwen_view=separate` at `h3_rules.REF_QWEN_SHORT_EDGE` is the default again since `a36ca547` (2026-10-03) and is on most shipped append-image nodes; the ceiling itself is reachable only on `qwen_view=shared` with a still wider than the crossing in the old section 4 (reasoned) | `MiniMaxH3ReferenceConditioning.image_policy=release` opt-in, `comfy` on every shipped conditioner; `bench/measure_qwen_bounds_bite.py`, `bench/audit_shipped_reference_bounds.py` | `../comfyui_vendor_gaps.md` section "4. The ceiling"; `../custom_node_gaps.md` section 4.1 item 1 (the `qwen_view` divergence); `../h3_references.md`, "A third knob, 2026-08-25: a Qwen view of its own" |
| 5 | Core encodes a reference soundtrack whole (`comfy_extras/nodes_minimax_h3.py::_encode_ref_audio`); sglang cuts it to the generated duration with `ffmpeg -t` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py`, `max_duration_seconds`), and diffusers does the same. | open in core; fixed here by `reference_conditioning.py::_prepare_audio` (caps at the aligned target duration) on every shipped graph | `bench/check_reference_runtime.py`; `bench/preflight_graph.py` warns on hand-built native socket graphs | `../comfyui_vendor_gaps.md` section "5, 6, 7", item 5; `../h3_references.md`, "Audio references" and "Known limitations, collected" |
| 6 | Core never upscales a reference and does not report the size it reached; sglang puts reference video on the canvas rule (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py`, `minimax_h3_resolve_spatial_shape` at `MINIMAX_H3_BASE_SHORT_EDGE`) and by default (`reference_resize_mode="diffusers"`) scales stills to a 2048 short edge. | open in core, for video and stills. sglang now also offers no-upscale still modes `match` and `max` with core's formulas (`5727f08932`, #35839), default still `diffusers`. Here stills follow sglang's default: `MiniMaxH3AppendRefImage` defaults to `size_policy=max`, `dit_short_edge`=`REF_IMAGE_SHORT_EDGE`, `allow_upscale=True` (`6231a456`); video follows core (`video_policy=comfy` on every shipped conditioner but the `release` probe arm) | `MiniMaxH3ReferenceConditioning` preview reports each reference's size and cost (`reference_report.py::format_report`); `video_policy=release` is the opt-in full vendor path, checked by `bench/check_reference_runtime.py` | `../comfyui_vendor_gaps.md` section "5, 6, 7", item 6; `sglang_comparison.md` "Tenth read" (reference image resize modes) and "Eleventh read", "Request and references"; `../custom_node_gaps.md` section 4.1 |
| 7 | Filed as "mono reference audio raises"; it never did, since core replicates mono to stereo (`comfy/sd.py`, H3 audio branch `pad_channel_value = "replicate"`). What remains: core keeps the first two channels of a wider track silently (`comfy/sd.py::VAE.vae_encode_crop_pixels`), where sglang downmixes with `-ac 2`. | mono: withdrawn (not a gap). Multichannel: open in core; fixed here by `reference_conditioning.py::_prepare_audio`, which refuses more than two channels | `bench/audit_ref_audio_channels.py` (audit, not a gate); `bench/check_reference_runtime.py` covers the refusal; `bench/preflight_graph.py::audio_channels` reports | `../comfyui_vendor_gaps.md` section "5, 6, 7", item 7, and "Settled"; `../h3_references.md`, "Known limitations, collected" |
| 8 | sglang keeps the video VAE fp32 for encode and decodes in fp16 autocast with an fp32 residual stream, and samples the encode posterior at a pinned seed (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py::MINIMAX_H3_KEYFRAME_ENCODE_SEED`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED`); core runs one dtype per VAE object, fp16 on this card, and takes the posterior mean (`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE.encode`). | open. Core added `VAEComputeDtype` (`fbca05a62`, #17450, 2026-10-30), a per-VAE-object dtype on a clone, which may let a graph give the encode branch its own dtype natively (read, not run) | `MiniMaxH3VAEPrecision` (`vae_precision.py`, since `9f300ca1`) splits encode from decode and refuses to cast a quantized half; wired only in `workflows/h3_probe_ref_vae_encoder_fp32_api.json`. The shipped video VAE is `h3_config.MODELS["video_vae"]`, whose provenance comment says its encoder encodes as the fp16 file's does. `bench/grade_vae_encoder_precision.py` measures; nothing enforces a choice | `../comfyui_vendor_gaps.md` section 8; `sglang_comparison.md` "Video VAE precision, and why our existing note does not settle it" |
| 9 | Filed as "H3 VAE tiling is an unrecorded runtime fallback". Core's H3 video VAE owns its tiling (`comfy/sd.py`, `handles_tiling = True` on the H3 branch; `comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE`, `comfy_has_chunked_io`), so an OOM retry does not switch algorithm. | not a gap (removed from the old table). The provenance side, that nothing records whether a decode was tiled or how tiles were batched, is a separate divergence (list below) | nothing | `../comfyui_vendor_gaps.md` section 8, "Removed gap: H3 VAE tiling"; `sglang_comparison.md` "VAE tiling is silent here" |
| 10 | sglang reads `partition` from `model_index.json` and refuses a task the checkpoint does not serve (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3ReleaseMetadata`); core has no partition concept. | open (architectural) | the generator picks the partition by graph kind (`workflows/build_workflows.py`, `MODELS["unet_ref2va" if ref else "unet_fl2va"]`); `vendor_config.py::partition_tasks` is read only by `bench/check_vendor_config.py`; no runtime gate | `../comfyui_vendor_gaps.md` section "10-13"; `sglang_comparison.md` "Refusals at admission rather than degraded service" |
| 11 | sglang can precompute the AdaLN modulation for the visited timesteps and drop `adaln_proj` (exact AdaLN cache). | not a gap on the files this repo loads: sglang refuses the cache on quantized weights and on pruned curve checkpoints (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel`, "only compatible with unquantized weights", "pruned curve checkpoints cannot use a separate AdaLN cache"); still a difference on an unpruned bf16 DiT | nothing; researched, not built | `../comfyui_vendor_gaps.md` section "10-13"; `sglang_comparison.md` "An exact AdaLN cache" |
| 12 | sglang captures breakable CUDA graphs over a padded packed sequence; core's per-block graph capture in `comfy/model_prefetch.py` is not enabled by the H3 DiT loop (`comfy/ldm/minimax/model.py`, `make_prefetch_queue` called with the caller's `transformer_options` only). | open (architectural); declined, and sglang's own quality gate requires it off | nothing | `../comfyui_vendor_gaps.md` section "10-13"; `sglang_comparison.md` "Breakable CUDA graphs over a packed sequence" |
| 13 | sglang ships step caching (Spectrum skip-step, opt-in: `SamplingParams.enable_spectrum` is False); core has none for H3. | open (architectural); declined, with nothing to skip at distilled step counts | nothing | `../comfyui_vendor_gaps.md` section "10-13"; `sglang_comparison.md` "Eighth read" (Spectrum); `2026-09-25_step_caching_survey.md` |
| 14 | sglang enforces a named fp32 set (`MINIMAX_H3_FP32_PARAM_NAMES`, `MINIMAX_H3_FP32_BUFFER_NAMES`) and, on curve-AdaLN checkpoints, also builds and requires `adaln_proj` and the time tables in fp32; core declares fp32 for those layers but `comfy/ops.py::mixed_precision_ops`'s `Linear.__init__` uses the compute dtype whatever the caller passed, and `comfy/ldm/minimax/model.py::_mod_row` casts table rows to the stream dtype. | open in core; wider than filed, since on the pruned INT8 files this repo runs sglang keeps the AdaLN path fp32 at every tier, `exact` included | nothing; magnitudes measured once (`comfyui_h3_t2va_trace.md` section 1.5); no check compares our fp32 set with the vendor's | `../comfyui_vendor_gaps.md` section 14; `sglang_comparison.md` "An enforced fp32 island, which our int8 load silently collapses" and "Eleventh read", "DiT forward"; `comfyui_h3_t2va_trace.md` section 1.5 |
| 15 | On an OOM retry core sends the H3 reference-audio encode to the 2D image tiler, because the H3 audio branch sets `latent_dim = 2` and leaves `extra_1d_channel` at `None` (`comfy/sd.py::VAE.encode`, the `do_tile` branch; `encode_tiled_` indexes `shape[3]`); peer audio VAEs set it. | open in core (read at core HEAD, not re-run) | nothing; trimming the soundtrack (gap 5) shrinks the encode but does not fix the fallback | `../comfyui_vendor_gaps.md` section 15; `comfyui_h3_t2va_trace.md` section 14 |
| 16 | Core's generic `VAE.vae_encode_crop_pixels` narrows the reference waveform's sample axis to a multiple of `spacial_compression_encode()` and drops leading samples, because the H3 audio branch leaves `crop_input = True`; the release and sglang right-pad to a whole hop instead (`coderef/MiniMax-H3/Ref2VA/audio_vae/dac_audio_vae.py::preprocess`; sglang `coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_audio_vae/audio_vae.py::preprocess`). | open in core: core HEAD has no commit for Comfy-Org/ComfyUI#15972 (`git log --grep=15972` is empty) and the H3 audio branch still sets no `crop_input`. Fixed here by `reference_conditioning.py::_encode_ref_audio_aligned` since `8870f8e0` (2026-09-10), which end-pads to the hop so the crop is a no-op | `bench/check_reference_runtime.py::ref_audio_end_padded_to_the_hop`; `bench/audit_ref_audio_crop.py` measures core's own path, which core's `MiniMaxH3ReferenceToVideo` still takes | `../comfyui_vendor_gaps.md` section 16 (its "Handling in this repo: none" is stale) |
| 17 | Core folds the gated residual into one in-place `addcmul_` in bf16 (`comfy/ldm/minimax/model.py::_mod_gate`), one rounding; diffusers computes product and add as two bf16 ops and sglang's kernel rounds the product to bf16 before the add (`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py::_indexed_gate_bf16_kernel`). At sglang's default `lossless` tier its RMSNorm and scale-shift also run as one fp32 pass (`121aa8dcc7`, #43327) where core rounds three times (`_mod_scale_shift`). | open in core; gate half at every sglang tier, scale-shift half at `lossless` and `high` (and on pruned curve checkpoints at every tier, see gap 14); unmeasured | nothing; the attention mirror never touches the residual | `../comfyui_vendor_gaps.md` summary row 17; `sglang_comparison.md` "Eighth read" (dated note) and "Tenth read" |

Changed since that document's last full update (2026-09-13): gap 2b was
fixed in core (`6bfaacc67`, #16400); gap 16 has been handled here since
`8870f8e0` (2026-09-10), which the old document missed; the encoder's
copy of a still has its own size again by default since `a36ca547`
(2026-10-03), which moves gaps 3 and 4 on this side; sglang gained still
modes that never upscale (gap 6); core gained `VAEComputeDtype` and the
shipped video VAE is the INT8 file (gap 8); sglang no longer offers its
AdaLN cache on the files this repo loads (gap 11); and gaps 14 and 17
widened on sglang's side. Not re-verified: gap 15's crash (read, not
reproduced), Comfy-Org/ComfyUI#15972's state on GitHub, and
`VAEComputeDtype` in use.

**Recorded elsewhere in this file, and never in the old document:**

From "What they do that we do not":

- Two more sparse-attention backends (cube sparse attention, and VSA-H3, whose capability set `coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::VSA_H3_CAPABILITIES` still excludes this card): section "Two more sparse-attention backends, read 2026-09-04".
- sglang refuses VAE decode modes outside its quality contract (`MiniMaxH3VideoVAEConfig.resolved_parallel_decode_mode`): section "Refusals at admission rather than degraded service".
- Text-encoder precision, where core runs the encode in fp32 and sglang in bf16 (ours the more conservative): section "Text-encoder precision, where ours is the more conservative one". The vendor-gaps doc had it only as "Adjacent, not a gap" under section 14.

From "What we both do, where ours may be the weaker version":

- sglang's fp16 decode carries an fp32 residual stream (`284cda01fb`, #41906); core's adds the residual at fp16: section "Video VAE precision, and why our existing note does not settle it" (dated note).
- Nothing here records whether a decode was tiled; sglang tiles by default as its quality contract and, at `lossless`, batches tiles per forward: section "VAE tiling is silent here".

From the eighth to eleventh reads:

- sglang's integrated mode runs the H3 DiT under a ComfyUI graph and refuses ComfyUI model patches, so this pack's DiT patches cannot run through it (`5730345717`, #43495): "Eighth read".
- sglang's ComfyUI adapter's velocity masking and audio scaling, fixed to follow core (`b4822bae03`, #43504): "Eighth read" and "Tenth read".
- A sounded reference video always brings its soundtrack as an audio block in sglang; here only when wired: "Ninth read".
- sglang pins a target frame only by a first or last still keyframe; core places a guide at any frame index (`comfy/ldm/minimax/model.py::PackedLayout`): "Ninth read".
- sglang can take a request's duration from its one audio-bearing reference; length is set by hand here: "Ninth read".
- `num_inference_steps` now counts DiT forwards in sglang (`32f34e898b`, #38671): "Tenth read".
- sglang's worker patch profile (`45de4adde9`, #44841, `coderef/sglang/python/sglang/multimodal_gen/apps/comfyui_worker_patches/h3.py`) lists op by op where its DiT differs from core's: time embedding cast to bf16, the gate through `addcmul`, the fused norm, RoPE through kitchen's `rms_rope_split_half_`, and the final layer on output rows only. Only the gate and the norm are in the vendor-gaps doc (row 17): "Tenth read".
- Pika H3 Flex's in-loop re-noise, and that its RNG draws differ across sglang, Pika's code and core's `sample_lcm`, so a same-seed comparison across engines is not an A/B: "Tenth read" and "Eleventh read", "Sampling, LoRAs and Pika".
- A reference video shorter than the target: sglang refuses under one VAE chunk, core and `video_policy=comfy` walk down and accept a much shorter clip: "Eleventh read", "Request and references" (also `../h3_references.md`, "Known limitations, collected").
- sglang pins transformers 5.19.0, the ComfyUI venv here has 5.18.0, and this pack imports two processors from it: "Eleventh read", "Text encoder".
- LoRA merging: sglang keeps an adapter unmerged on quantized layers and adds the delta with one rounding at its default tier; Comfy-Org's Pika template merges into the INT8 checkpoint: "Eleventh read", "Sampling, LoRAs and Pika".
- Video decode tile batching depends on free VRAM in core (`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE._decode_tile_row`) and on a config value in sglang, so re-decodes here can differ at the rounding level with memory state: "Eleventh read", "VAEs".
- sglang pins its ffmpeg binary; this pack's videos are written by whatever ffmpeg VideoHelperSuite finds, unrecorded, and crf, tags and rounding differ from both sglang's and core's: "Eleventh read", "Output".

Adjacent, not in `sglang_comparison.md` but missing from the old table too (`../h3_references.md`, "Known limitations, collected"): core pairs a reference video's soundtrack by socket number where sglang routes the material itself; core has no trim offset where sglang takes `start_time_seconds`; and the twelve-reference total and audio-never-alone limits are enforced only by diffusers.

---

## The filter: most of sglang's speed is four cards

sglang's audited deployment for `quality="high"` is a 4xH200 fl2va server with
`sp_degree=4` and `ulysses_degree=4`, and `validate_quality_deployment`
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig.validate_quality_deployment`)
raises unless the resident server matches it exactly — down to asserting the
device is an H200 at compute capability 9.0. That same gate requires
`enable_torch_compile`, `enable_breakable_cuda_graph`,
`is_dit_layerwise_offload_selected` and `quantization` all to be **off**.
*Note 2026-11-04: the citation named lines 98-186; it names the method now.
At `origin/main` the gate still expects every value above, and also no
LoRA, no substituted transformer weights and no text-encoder quantization;
the substituted-weights rule is `aa79087017` (#34978).*

Two consequences worth holding on to. Their headline performance is sequence
parallelism we cannot copy on one card. And the knobs they ship but require
off for their quality claim are knobs they do not stand behind for quality
either, so "sglang has it and we don't" is not on its own an argument.

---

## What they do that we do not

### An exact AdaLN cache

`coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_adaln_cache.py`
precomputes the modulation parameters for every timestep the schedule will
actually visit, and the DiT then drops `adaln_proj` entirely
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTBlock`,
where `use_adaln_cache` leaves `adaln_proj` as `None`).
There is an in-process prepass that builds the same thing by reading all the
`adaln_proj` layers once
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py::MiniMaxH3AdalnCache`,
its `_project_plans_from_checkpoint`),
sized by a plan-width knob whose per-task requirement is stated in the
comment on `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py::MINIMAX_H3_ADALN_MAX_PLAN_WIDTH` — t2va, fl2va and ref2va need different numbers of distinct timesteps
per step, which is the same fact our packing already encodes.
*Note 2026-11-04: the three citations named DiT line ranges; the cache now
lives in its own module and they name symbols. At `origin/main` the DiT
refuses a separate AdaLN cache on quantized weights and on pruned curve
checkpoints
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel`,
"only compatible with unquantized weights" and "pruned curve checkpoints
cannot use a separate AdaLN cache"), so sglang serves the cache only on an
unpruned, unquantized DiT, never on the pruned INT8 kind of file this repo
loads.*

**Why this is interesting here.** It is an exact answer to the problem the
Comfy-Org "pruned" checkpoints answer approximately. The pruned file replaces
those weights with a rank-8 SVD of the time curve
([`evidence.md`](../ops/evidence.md) owns that measurement); the cache keeps full
accuracy and pays a cache tensor sized by the step count instead of by the
weight matrix.

**What it does and does not buy on this box.** No speed. On the pruned file we
run, the AdaLN projection is already tiny, so there is no matmul left worth
eliminating. What it buys is *unpruned accuracy at pruned memory*.

**#22 has since reported, and it half-opened the gate**
([`data/results/2026-08-21_pruning_sensitivity.json`](../../../data/results/2026-08-21_pruning_sensitivity.json)).
The pruning is **not** invisible at the output: the first-step velocity moves
5.6-9.4% against a determinism floor of exactly zero. So "the residual does not
matter" — the outcome that would have killed this outright — did not happen.
What did happen is that the effect is the same size on both checkpoints and
**smaller than the `fp8_scaled`-vs-`int8_convrot` difference this repo already
ships**, which is why it is still not urgent: an exact AdaLN would remove an
error that is not the largest one in the stack.

**What would make it worth building** is evidence that the difference is
*visible*, which no measurement here can supply — the arms are one forward, not
a render. That is a blind session on pruned against unpruned under
[`eval_comparison.md`](../ops/eval_comparison.md) section 3, and it has not been run
or scheduled.


### Two more sparse-attention backends, read 2026-09-04

Neither needs the FastH3 weights to run on base H3, and neither is a serving
feature; both are attention policies, which is the axis this repo works on.

**Cube sparse attention** (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/cube_sparse_attn/`,
merged 2026-09-02 with two MiniMax engineers as co-authors). FlexAttention
over `(T, H, W)` cubes of latent tokens, so a block is a spatial-temporal
neighbourhood rather than a run of Morton-ordered rows; a per-step
`topk_ratio_list` with one entry per denoise update, where a ratio of one
means that step runs dense; text, audio, standalone reference images and the
token refiner stay dense; reference videos and the target compete in one
global top-k pool. No pooled correction term, so it is the SLA shape of the
idea rather than the Sol shape. Their own cookbook calls it approximate and
says FlexAttention's routing overhead can outweigh the saving on short
sequences. It is the vendor's engineers choosing cube geometry for this
model, which [`../morton.md`](../archive/morton.md) has no measurement against.

**VSA-H3** (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py`, merged 2026-09-02): the
trained sparse policy for the FastH3 weights, an in-tree Triton kernel gated
to SM90 and above, so it does not run on this card at all. Its knobs are
what to read: `vsa_mode` exempt/compete (whether non-video keys are always
kept or compete in the top-k), `vsa_dense_first_n_steps`, `vsa_dense_layers`.
Ours: sink ranges derived from the segment table play the role of `exempt`,
`dense_blocks` is `vsa_dense_layers`, and the `sigma_start`/`sigma_end`
window on `MiniMaxH3SolAttn` is the per-step knob in sigma rather than in
step index. [`../SOLATTN.md`](../attention/SOLATTN.md) owns those.
*Note 2026-11-04: "gated to SM90 and above" is stale. Since `465ddf9758`
(#43823) the gate is the set
`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::VSA_H3_CAPABILITIES`,
which adds capability 8.0 and still leaves out this card's 8.9 ("Tenth
read"). The three knobs are still read at `origin/main`, in
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py`.*

### Breakable CUDA graphs over a packed sequence

`coderef/sglang/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py`
pads the packed prompt to the model's sequence alignment and rewrites
`cu_seqlens_q`, `max_seqlen_q` and the position ids so one captured graph
serves varying lengths. That padding layer is the part that makes graph
capture possible for a packed-sequence model at all. Note again that their own
quality gate requires this off, and that under it H3's attention core and
`_embed` are marked `eager_on_graph(True)`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`,
read 2026-08-25): the captured region excludes the phase that dominates
sampling time on this box.

**Corrected 2026-08-25: ComfyUI does have an equivalent, and it is off for H3
by construction.** Source read, not run: `comfy/model_prefetch.py` captures one
`torch.cuda.CUDAGraph` per block inside the dynamic-VRAM prefetch queue and
replays it only when the allocator's placement signature for that block's
weights matches the capture (`vbar_signature_compare`), which is the
streaming-weights problem answered per block rather than avoided. It is wired
for LLaMA-family decode (`fixed_kv_decode` only, so never for a prompt encode),
Gemma4 decode and MiniMax Music; the H3 DiT loop
(`comfy/ldm/minimax/model.py`, the `prefetch_queue_pop` calls) passes neither
`core` nor `enable_graph`, so no H3 block is captured. `TorchCompileModel`'s
`cudagraphs` backend is the other route and it clones the model with
`disable_dynamic=True`, which on a card the DiT does not fit
([`hardware.md`](../ops/hardware.md), measured 2026-08-17) is not a route.
What is left to gain is bounded by
`data/results/2026-08-18_phase0_instrument.json`: sampling at 1024x768 with
three references ran at 100% SM occupancy with power pegged at the limit, and
graph replay removes launch gaps only. Untested and the one place it could
still pay: a single-frame `workflows/image/` render at a small canvas, where
the same instrument reading SM occupancy well under 100% would be the signal.
The technique is also available outside sglang as `meta-pytorch/breakable-cuda-graphs`
(BSD-3; README read 2026-08-25: `@no_graph` regions may not return CUDA
tensors, and it says nothing about weights that move between replays).

### An enforced fp32 island, which our int8 load silently collapses

**Found 2026-08-29 from the ComfyUI side**, and it is the sharpest new entry in
this file because the vendor names the exact set we lose.

sglang keeps a **named, enforced** list of tensors that stay fp32 while the rest
of the DiT is bf16 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MINIMAX_H3_FP32_PARAM_NAMES`):
`MINIMAX_H3_FP32_PARAM_NAMES` covers both patch projections, the time embedder
and both output heads, and `MINIMAX_H3_FP32_BUFFER_NAMES` covers
`rope.inv_freq`. It even handles the pruned case, dropping `time_embedder.*`
from the list when `adaln_t_table` is present
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel.post_load_weights`).
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) §7 records that the island is
never quantised.
*Note 2026-11-04: both citations named line ranges and name symbols now; the
lists are unchanged at `origin/main`. Not recorded before for the pruned
(curve AdaLN) checkpoints, the kind this repo runs: there the island is
wider. `post_load_weights` adds `adaln_t_table` (and the basis and mean
when present) to the fp32 list and refuses any `adaln_proj.linear` weight
that is not fp32 ("must stay fp32 with curve AdaLN"), and
`MiniMaxH3AdalnProj` builds those projections in fp32 when
`adaln_curve_grid` is set. The modulation then reads the fp32 tables at
every tier: at `exact` through `indexed_scale_shift_fp32_` and
`indexed_gate_fp32_`
(`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py`,
added by `1fd46ddd56`, #44180, and asserted bit-equal to the eager fp32
expression by its test), and at `lossless` and `high` through the fused
norm-modulate pass of the tenth read. Either way the affine is computed in
fp32 and rounds to bf16 once. Core casts each table row to the stream's
bf16 before three in-place bf16 ops
(`comfy/ldm/minimax/model.py::_mod_scale_shift`, its `_mod_row`), so on
this path the two differ at every sglang tier, not only at `lossless`
(read, not run). Core's side, re-read the same day: the "fp32 island"
comment now sits in `FinalLayer.__init__` and names only the output heads,
and `comfy/ops.py::MixedPrecisionOps` still builds its `Linear` at the
compute dtype whatever `dtype=` the caller passes.*

ComfyUI declares the same intent and does not keep it on a quantized
checkpoint. `comfy/ldm/minimax/model.py` constructs those layers with an
explicit `dtype=torch.float32` and calls them "the checkpoint's fp32 island" in
a comment in `comfy/ldm/minimax/model.py::FinalLayer.__init__` — but
`comfy/ops.py::MixedPrecisionOps.Linear.__init__` discards the `dtype=` its caller passed and uses the
compute dtype, so on `int8_convrot` every one of them loads bf16.
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) §1.5 has the mechanism
and the verification-by-execution; the magnitudes are 3.7e-4 to 1.7e-3
relative, against the 8.8e-3 the same checkpoint's int8 blocks already carry.

**Priced, not urgent, and the reason is the same as the AdaLN cache above**: it
removes an error that is not the largest one in the stack. What makes it worth
recording anyway is that it is a *stated intention this install does not meet*,
it is a **strict** regression for `adaln_proj` (F16 on disk to bf16 in memory),
and it silently confounds any bf16-against-int8 checkpoint comparison, which
changes four things at once rather than one. **Enforced by nothing** — no check
asserts our DiT's fp32 set against the vendor's named list, and nothing would
notice if core changed it again.

### Text-encoder precision, where ours is the more conservative one

Recorded so this file is not read as a list of places we are behind. sglang runs
the Qwen3-VL encode in **bf16**
([`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) §4). ComfyUI upcasts the
embedding to fp32 (`comfy/sd1_clip.py:213`) and never comes back down, and
`comfy/sd.py::CLIP.__init__` sets the patcher's compute dtype to match with the comment
"Match torch.float32 hardcode upcast in TE implemention". So the whole 50-layer
stack runs fp32 activations here and bf16 there.

That is also why an int8 encoder never reaches an int8 GEMM in ComfyUI, which is
upstream policy rather than an oversight — `25022e0b` (2025-11-24) replaced an
explicit `fp8_matrix_mult=False` with today's `full_precision_mm=True`. Measured
consequence on this box: int8 costs no more time than bf16 per resident layer
(0.78 ms dequant against a 0.87 ms cast) and halves the PCIe transfer that
actually dominates (10.87 ms against 21.75 ms), for 0.88% weight error.
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) §2.4 owns those numbers.

### Refusals at admission rather than degraded service

Not speed, but the design difference that shows up most often:

- The release partition gate
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3ReleaseMetadata`,
  its `from_model_index` and `canonical_task`; the citation named lines
  60-135 until 2026-11-04)
  reads `partition` from `model_index.json._minimax_h3` and raises when the
  requested task does not belong to it. A `t2va` request on the `ref2va`
  partition is refused, not served badly.
- `resolved_parallel_decode_mode()`
  (`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py::MiniMaxH3VideoVAEConfig.resolved_parallel_decode_mode`;
  the citation named lines 42-56 until 2026-11-04)
  refuses `spatial`, `spatial_shard` and `patch` VAE decode as outside the
  released quality contract, before any large component downloads.

ComfyUI has no partition concept: a plain t2v graph loads `ref2va` and
renders. **This is the frame for the checkpoint-swap arm** — a `ref2va` loss at
plain t2v reads as *not a t2v model by its own release metadata*, not as a
defect in the checkpoint.

---

## What we both do, where ours may be the weaker version

### Video VAE precision, and why our existing note does not settle it

sglang keeps the video VAE **fp32-resident and decodes in fp16 autocast**, and
the config says why in as many words: the VAE also encodes keyframes
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig`,
`vae_precision: "fp32"`, `vae_decode_precision: "fp16"`).
*Note 2026-11-04: the citation named lines 59-62 and names the class now;
both values and the comment are unchanged at `origin/main`. What "decodes
in fp16" covers there was not recorded: under fp16 autocast sglang's
decoder blocks carry an fp32 residual stream with fp16 attention and FFN
updates, by the docstring of
`coderef/sglang/python/sglang/kernels/ops/diffusion/norm/h3_vae_fused_norm.py`,
which since `284cda01fb` (#41906) fuses each scaled residual add into the
next RMSNorm and keeps that fp32 residual; it is on by default
(`MINIMAX_H3_VAE_DECODER_FUSED_NORM`) and not tied to a quality tier. Core's
decoder blocks add the residual at the VAE's own dtype, fp16 on this card
(`comfy/ldm/minimax/vae.py::TransformerBlock`, the residual scale cast to
the input's dtype). So the two fp16 decodes are not the same arithmetic
(read, not run).*

Ours runs fp16 throughout. `comfy/model_management.py::vae_dtype` returns fp16
on this card unless `--fp32-vae` is passed, and the default launch mode does
not pass it.

`<comfy>/start.sh` already carries a considered decision against `--fp32-vae`,
added and reverted 2026-08-10: measured 2-3x decode cost, no established
benefit, and the reference's own fp32 evidence was an *audio* VAE measurement
that ComfyUI already honours. That reasoning stands for what it addressed.

**What the sglang read adds is that the decision was scoped to decode, and the
vendor does not decode in fp32 either.** The vendor splits the two: fp32 for
residency and encode, fp16 for decode. So the 2-3x decode cost we measured and
rejected was never the vendor's behaviour, and the open half of the question is
the *encode* side — reference images, reference videos and keyframes, whose
whole job is identity fidelity, computed once per render rather than per step.

`--fp32-vae` cannot express that split; it forces both. So the flag is the
wrong instrument for the remaining question, and the start.sh note should not
be read as having closed it. **Enforced by nothing** — no check asserts our VAE
encode precision against the vendor's, and nothing would notice if it changed.

### VAE tiling is silent here

ComfyUI tiles under memory pressure and records nothing about having done so;
sglang treats decode mode as part of the quality contract and refuses the modes
it considers inexact (cited above). We cannot currently distinguish a tiled
decode from an untiled one after the fact.
*Note 2026-11-04, at `origin/main`: sglang tiles by default, with overlapping
latent tiles as its quality contract (`use_tiling` and the comment on
`parallel_decode_mode` in
`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py::MiniMaxH3VideoVAEConfig`).
At `lossless` and `high` on CUDA the decode also takes a fast path: each
decoder attention block fuses its per-head QK RMSNorm and NeoX RoPE into
one launch and runs on cuDNN SDPA, and since `0498ab7b6e` (#45179) up to
`decoder_tile_batch` tiles and `window_batch` windows share one decoder
forward
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_vae_cuda_opt.py::maybe_optimize_minimax_h3_vae`,
gated by `quality_allows(quality, "lossless")` in
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::MiniMaxH3DecodingStage`).
`exact`, `--performance-mode memory` and non-CUDA platforms decode one tile
and one window per forward. The config's comment says batching "moves
rounding, not the math". The decode here is core's, so none of this runs
in this repo.*

---

## Refuted here, and worth keeping refuted

**Hypothesis, raised and killed on 2026-08-21: that the fp8-vs-int8 fidelity
gap is a qkv row-permutation defect in the repack.**

The hypothesis had a real source behind it. sglang treats the H3 qkv layout as
a load-time hazard: the release interleaves each head's Q, K and V rows, sglang
wants them concatenated, so it permutes on load
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3Attention._install_qkv_weight_loader`)
and then permutes **row-indexed quantization metadata the same way** via
`_install_qkv_row_reorder`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::_install_qkv_row_reorder`;
both citations named line ranges until 2026-11-04, and the code they point
at still does this at `origin/main`),
using the row count as the gate so swizzled and per-tensor scales pass through
untouched. Its unit test states the failure mode: get it wrong and the model
"loads and runs, and renders noise"
(`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_qkv_scale_reorder.py:22-26`).

It dies on two independent grounds, both already in this repo:

1. **The fp8 file has no row-indexed scale to misalign.** Its `weight_scale` is
   a scalar (`bench/analyze_quant_delta.py`, the format description at the top
   of the file). No permutation of output rows can put a per-tensor scalar on
   the wrong row.
2. **The gap is uniform across module kinds, so it is not qkv-specific.**
   [`data/results/2026-08-21_quant_delta_fl2va.json`](../../../data/results/2026-08-21_quant_delta_fl2va.json):
   `fp8_vs_bf16` reads the same for `attn.qkv_proj` as for `attn.out_proj`,
   `mlp.fc1` and `mlp.fc2`. A layout defect confined to the fused qkv weight
   would single that module out and it does not.

The reordering itself was never taken on faith: `probe_qkv_layout()` in
`bench/analyze_quant_delta.py` refuses to measure unless reordering is the
better reading, and the same record carries its verdict.

So the remaining explanation for the gap is the one the script was built to
measure: a per-tensor scalar scale against a per-output-row scale, which differ
exactly in the per-channel error distribution and nowhere in a whole-tensor
norm. **No new work is owed here.**

---

## Looks like a gap, already priced

**Step caching.** sglang ships a Cache-DiT configuration for `quality="high"`
with a measured SSIM and PSNR against lossless
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_PHASES`),
so it reads as an obvious gap. It is not. [`roadmap.md`](../ops/roadmap.md) records
the owner decision of 2026-08-20 that step caching dies here: the speedup came
from skipping steps on a 16-step schedule, and at 4 steps there is nothing to
skip. Their number is at 50 steps on four H200s. Different regime.
*Note 2026-11-04: the citation named lines 57-64 and names the constant
now. "Against lossless" meant the bit-exact reference tier, which
`7a4d6dfd2f` (#42370) renamed `exact`; the old `extra-high` is today's
`lossless` and the default
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`).
The constant's comment now measures against `quality="exact"` and still
calls `exact` the default, which the sampling params contradict.*

**Quantization, torch.compile, layerwise offload.** All present in sglang and
all required *off* by its own quality gate, so none of them is evidence for
anything. We already run quantized weights by necessity on 24 GB.

---

## What we do that they do not

Recorded so the comparison is not read one-directionally: the SLA router has no
counterpart in sglang's H3 path. [`SOLATTN.md`](../attention/SOLATTN.md) owns those
numbers. Their speedups and ours are not comparable and must not be put in the
same table.

**Sol-Attn and the sage kernels no longer belong in that sentence, corrected
2026-08-28 against `803b4fb31c`.** This section read "Sol-Attn, the sage kernels
and the SLA router have no counterpart ... which runs dense FlashAttention",
which was true when written and is not now: sglang ships a `sol_attn` attention
backend (`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_sol_attn_backend.py`)
and a `kitchen_int8` linear path dispatching the same `comfy_kitchen.int8_linear`
our checkpoints load through. *2026-10-02: `kitchen_int8` is now a deprecated
alias of `convrot_int8` (`e667ab10d5`), which still dispatches to kitchen on
this card; see "Eighth read".*
*Note 2026-11-04: `3472e16f96` (#45706) removed the alias, so `kitchen_int8`
is now refused as an unknown quantization method and only `convrot_int8`
remains; on this card it still dispatches to `comfy_kitchen.int8_linear`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/quantization/configs/convrot_int8_config.py`).*

**What that convergence is, and is not.** Their published consumer-card table's
fastest row is an int8 DiT with a sage-to-Sol hybrid — which is the stack our
graphs already wire. That is two projects arriving at one answer, and it is
worth more as corroboration than as an action item, because there is nothing
here to adopt. **Do not import their numbers.** Their PSNR column compares
different samples by our own 2026-08-18 measurement, and their own quality
section says as much; their seconds came off a host that page-caches the whole
checkpoint, so the mechanism carries and the figures do not.

### Sol-Attn defaults: sglang against core's and ours (2026-09-11)

Read from source: sglang at `593c7a900d`
([`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) section 14.9 has the walk
and the lines); core's `BlockSparseAttention` at ComfyUI `1d48d9cf`
(`comfy_extras/nodes_sparse_attention.py::BlockSparseAttention.define_schema` for the schema,
`comfy_extras/nodes_sparse_attention.py::make_h3_block_patch` for what an H3 block does outside the window). Ours is
`workflows/h3_config.py::SOL_RECOMMENDED_CUDA`, and core's defaults are
copied into `workflows/h3_config.py::SOL_CORE_DEFAULTS`.

| knob | sglang `sol_attn` | core `BlockSparseAttention` | ours, before this change |
|---|---|---|---|
| on by default | no; the DiT default is `fa` and Sol is opt-in | only in a graph that adds the node | on in every shipped video graph, except the stems in `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS` |
| dense at the start | the first 10 steps (`dense_steps`), a count | the first fifth of the sigma schedule (`start_percent` 0.2) | same as core |
| dense at the end | none: sparse through the last step | none at its default `end_percent` 1.0 | the last step, through `end_percent` 0.9, `SOL_END_PERCENT_BY_STEPS` for the distilled step counts, and `SOL_PDD_OVERRIDES` for PDD |
| tau | 1.0 | 1.3 | 1.0 |
| dense blocks | 0 and 1 (and, by the name match, both token-refiner blocks) | none | none |
| sink | none by default (`sink_tokens` 0) | `exact_kv_and_rows` | `exact_kv_and_rows` |
| token routing | none | `extra_tokens` 256 on every block | off |
| outside the window | FlashAttention, or Sage in the documented recipe | the block's own attention, whatever the model already runs | Sage, which Sol chains onto |

*2026-09-19: the last cell is no longer true. Since 2026-09-15 the shipped
graphs chain Sol onto kitchen's dense attention through core's Model
Attention Backend, not onto Sage; `workflows/h3_config.py::DEFAULT_DENSE_CHAIN`
names the chain and its comment block says what each chain runs. The other
cells of our column still match `SOL_RECOMMENDED_CUDA`, the knobs added to it
since (`qk_balance`, `rotate`, `morton`) aside.*

*2026-10-02: the dense-blocks cell of our column is no longer "none". The Sol
node's `dense_blocks` default is `sol_attn_h3.py::SOL_DENSE_TAIL`, mirrored by
`workflows/h3_config.py::SOL_DENSE_TAIL`, so on that row sglang, core and ours
now all differ. sglang's and core's cells were re-read at `89f21671bb` and
`65787d66` and have not moved.*

*Note 2026-11-04: the sglang column was re-read row by row at `origin/main`
and holds.
`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py`
has no commit since `593c7a900d`; its defaults are still `dense_steps` 10,
`dense_layers` "0,1", `tau` 1.0, `sink_tokens` 0, no end of window and no
token routing, with `dense_backend` "fa"
(`_get_sol_attn_runtime_config`, `SolAttnImpl._should_use_dense`). Sol is
still opt-in: the cookbook's 4090 table says "Default remains
`convrot_int8` + `fa`", and its recipe (the paragraph after "`fa` keeps
exact attention" under "Feature contracts and advanced recipes") still
sets `dense_backend=sage_attn,dense_steps=10`. The SGLD app's preset
matches ("Tenth read"). Core's column was re-read in
`comfy_extras/nodes_sparse_attention.py` at ComfyUI `5f415e31a` and holds.
Our column is the 2026-09-11 state and has moved on the sink row as well:
`workflows/h3_config.py::SOL_RECOMMENDED_CUDA` now sets `sink_conditioning`
to `exact_kv_and_all_rows`, so the 2026-09-19 note's "the other cells still
match" no longer holds for that row.*

The start row agrees only on a fifty-step grid, where 10 steps is a fifth.
At our sixteen-step base, sglang's count would keep 10 steps dense against
the 4 that `start_percent` 0.2 keeps (the `start_percent` comment in
`workflows/h3_config.py::SOL_RECOMMENDED_CUDA` has that arithmetic).

**Decision, owner, 2026-09-11.** This is a tinkering repo: where sglang and
ComfyUI's own node agree on a default and ours differs, ours takes theirs,
without waiting on an eval of our own. The one knob where both agree against
us is the end of the window, so ours moves to sparse through the last step,
`end_percent` 1.0, for base, distilled and PDD graphs alike. It lands with the
next graph rebuild, and `CHANGELOG.md` carries it. The other rows are not
adopted: on tau, dense blocks, the sink and token routing, sglang and core
disagree with each other, and on the start they agree only at a step count
we do not run.

---

## The 768 cap, and what it did to our code

The open release is 768p on the short edge. Establishing what *kind* of limit
that is changed one doc claim and found one defect; the finding itself is owned
by [`h3_resolutions.md`](../core/h3_resolutions.md) and is not restated here. The
short version for anyone arriving from a performance question: the area cap is
hard upstream, the short edge only warns, and neither is a property of the
weights.

Auditing this repo against it (2026-08-21, source read) came out mostly clean.
`resolution.py` already grades a canvas by whether `adapt_canvas` is a fixed
point on it and labels anything outside the trained family rather than refusing,
which is the right posture given what the cap turns out to be. `reference_fit.py`
keeps the 2048 reference short edge and the 768 target cap properly separate — the two knobs this repo has confused
before — and `h3_config.py` has no target short-edge constant to confuse.

**One real defect, found and fixed.** `bench/preflight_graph.py` priced a
reference image that is *not* fed through `MiniMaxH3ReferenceFit` as if it were
upscaled to a 2048 short edge. Core does the opposite: it clamps with
`min(1.0, ...)` in both sizing modes (`comfy_extras/nodes_minimax_h3.py:312-316`)
and never enlarges. The over-count is the square of a scale the reference never
gets — a 1024x1024 reference priced at four times its real row count. No shipped
API graph reaches that branch, because they all wire the fit node; the exposure
is exactly the hand-built graph that `CLAUDE.md` promises preflight can price.
Fixed by defaulting the no-fit case to no upscale, and confirmed non-inert
against a graph rewired to bypass the fit node.

---

## Open after this read

- The VAE **encode** precision question, above. **Half of this closed on
  2026-08-21, after the sentence below was written**: the instrument exists.
  `bench/grade_vae_encoder_precision.py` grades the encoder at the call rather
  than at a rendered clip, and
  [`data/results/2026-08-21_vae_encoder_precision.json`](../../../data/results/2026-08-21_vae_encoder_precision.json)
  records fp16 bit-identical to itself against fp32 moving the latent, with
  bf16 as the far control. What stays open is whether that delta is *visible*,
  and the variable below, which the instrument does not separate.
  `--fp32-vae` remains the wrong flag because it forces both halves. **It now has a second
  variable tangled with it, found 2026-08-21**: sglang *samples* the released
  posterior under a seed pinned at 42 for keyframes and reference video
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py::MINIMAX_H3_KEYFRAME_ENCODE_SEED`,
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED`) where ComfyUI takes the mean
  (`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE.encode`). Any encode instrument has to separate
  precision from mean-versus-sample or it measures neither.
  [`h3_references.md`](../core/h3_references.md) carries it in the vendor image
  path table.
  *Note 2026-11-04: the two sglang citations named lines 30 and 613 and
  name the constants now. Still open at `origin/main`: both seeds are 42,
  the keyframe encode still samples the posterior (its module docstring,
  `use_mean=False`), and the video VAE still encodes in fp32. Core's mean
  is now in `comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE`, its `encode`.*
- Whether an AdaLN cache is worth building, which is downstream of
  [`open_experiments.md`](../ops/open_experiments.md) #22 and should not be
  started before it.
  *Note 2026-11-04: sglang's own cache is refused on quantized weights and
  on pruned curve checkpoints (see "An exact AdaLN cache" above), so the
  vendor has no form of it for the pruned INT8 file this repo loads. On
  that file sglang instead keeps the curve tables and projections in fp32
  (see "An enforced fp32 island" above).*
- Nothing left on the reference path: video, audio, frame rate, soundtrack
  pairing and condition noise were all re-derived on 2026-08-21 against sglang,
  diffusers and DiffSynth-Studio, and [`h3_references.md`](../core/h3_references.md)
  carries the results. One item there is marked unverified and stays that way:
  what sglang does when a reference video is shorter than its aligned target.
  The mono-upmix question that sat beside it was closed on 2026-08-21 — no
  upmix happens and the packed layout raises — and `h3_references.md` carries
  it under Known limitations.
  *Note 2026-11-04: the shorter-reference-video item is answered from
  source at `origin/main` (code unchanged across the tenth read's range,
  read, not run). The ffmpeg pass caps the decode at the target's frame
  count with `-frames:v` and neither pads nor loops
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_decode_reference_video_frames`),
  so a short source yields fewer frames. The VAE encode then trims down to
  the chunk grid
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/processor.py::get_suitable_video_length`,
  through `align_video_length` in trim mode), which with the release's chunk
  of 17 frames and overlap of 5 is the `17n+5` grid with n at least one; a
  source under that floor raises "Cannot trim". In the packed sequence the
  reference then takes only its own length in time: the cursor advances by
  the longer of its soundtrack's span and its trimmed video's span
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py`,
  `t_cursor += max(float(ref_t), _video_t_span(vt))`).
  Core walks down to the same grid and accepts shorter sources
  ([`h3_references.md`](../core/h3_references.md), "Known limitations,
  collected"). The overlap of 5 is core's derivation from the same
  defaults (`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE`, from
  `clip_length` and `token_drop`); sglang reads its value from the
  release's config, which was not opened here.*

## Third read, 2026-09-04

[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) section 14 records what
landed in sglang between 2026-08-31 and 2026-09-05. This section says what
each item is against what we do, in the order a reader deciding what to borrow
would want, and it changes no earlier verdict on this page.

**FastH3 is a different model, and sglang treats it as one.** Its pipeline
config refuses every mode, task, step count and quality tier the student was
not distilled for. That is the discipline any FastH3 rung here would need to
inherit: a t2va-only arm at five grid points, its own dense pair, never pooled
with base-H3 rungs. [`vsa/vsa_node.md`](vsa/vsa_node.md) holds the blockers.

**Their quality tiers are our ladder, stated as a contract instead of
measured.** `lossless`, `extra-high` and `high` are request-time promises
enforced at admission; our dense, sage and Sol rungs are renders judged blind.
For H3 the middle tier is empty on their side, which says the same thing the
ladder verdict says on ours: on this model the kernel arithmetic is not where
the quality goes. Their definition of "lossless" also names the exception
explicitly, `torch.compile`, and refuses it as ground truth; ours names the
Comfy Compiler's malloc graph as the regime and checked one rung against it
(`data/results/2026-09-04_stairwell_dense_retime.jsonl`).
*Note 2026-11-04: the tier names and the "empty middle tier" are stale.
`7a4d6dfd2f` (#42370) renamed the tiers after what they guarantee: the old
`lossless` is now `exact` (bit-identical to the reference), the old
`extra-high` is now `lossless` (same math, rounding may move; the default,
with `extra-high` kept as an alias) and `high` is unchanged
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`).
For H3 the middle tier is no longer empty: at `lossless` the DiT fuses
RMSNorm and AdaLN into one rounding (`121aa8dcc7`, #43327), a dynamic LoRA
delta is added inside the second GEMM (`65386cb95a`, #43501), and the video
VAE decode takes its fast path (see "VAE tiling is silent here"). The
`torch.compile` exception is still named, now in the cookbook's feature
contracts table ("Do not use it for consistency ground truth").*

**Silent fallback is now refused on their side too.** An explicit
per-component attention backend must be consumed or the server does not
start, and the SubBlock README's warning that `transformer=subblock_sparse_attn`
"appears to work and silently does nothing" is the failure this repo's chain
assert and `bench/check_attention_defaults.py` exist for. Two projects
arriving at the same guard; nothing to import.

**The AdaLN cache verdict stands, with a new hazard beside it.** The tiered
host cache saves a checkpoint re-read on a plan miss, which a single-graph
ComfyUI render never pays, so "no speed on the pruned file, unpruned
accuracy at pruned memory" is unchanged. What is new is that they now fail
closed on a LoRA that touches `adaln_proj` under either cache mode, after
finding such deltas were silently dropped. Whether any LoRA this repo loads
touches those weights is a question for `bench/check_pdd_head_selection.py`'s
owner, not settled here.

**The cube schedule is a step policy from the vendor's own engineers.** The
recommended `topk_ratio_list` for the fifty-step grid keeps the first two
updates dense and decays from there; roadmap step 5 (the window's start) has
so far had only our own probe trend to reason from. It is a prior, not a
measurement on our stack, and cube geometry is not Morton order.

**SpargeAttention is adaptable and not urgent.** It runs on this card's
architecture with one knob and no per-model tuning, so it could be an arm
against Sol; their own measurement on another model found no single-GPU
speed or memory win and a large perceptual change, and they call it
approximate even at full keep. Below Sol's block policy in priority.

**Block-FP8 is the qkv reorder hazard again.** A standard block-quantized
export loaded clean and rendered blank until the scale-row permutation was
made block-aware. Our refuted-hypothesis section above already established
that our fp8 file has no row-indexed scale to misalign; this is corroboration
that the hazard is real for formats that do, not evidence about ours.

**Everything else is serving or other silicon**: warmup at the served shape
(our runner's `--warmup` row is the same idea), profiler spans, SM120 paths,
the SM12.x decoder workaround, key masks under Ulysses, third-party bundle
loading. Read, priced, no action.

## Fourth read, 2026-09-10

What landed in `coderef/sglang` between `320bdd1ee2` (the third read) and
`887c401e15`, against what we do. It changes no earlier verdict on this page.
The two Sol-side items, SubBlock's new Sage compute and its router's note on
reserved sink blocks and the forced diagonal, are recorded in
[`../sol_upstream.md`](../attention/sol_upstream.md) and not repeated here.

**"Singularity" is a third-party checkpoint, not a vendor variant.**
`65400bb420` (#38455) serves `WarmBloodAban/Minimax-h3_Singularity`, an FL/Ref
fusion fine-tune shipped full or pruned INT8, through `--model-variant hybrid`
plus an explicit transformer-weights path: one pipeline for t2va, fl2va and
ref2va, with the partition gate relaxed (sglang's H3 cookbook page, its
Singularity section). ComfyUI has no partition gate
([`../comfyui_vendor_gaps.md`](../archive/comfyui_vendor_gaps.md) gap 10), so such a
file loads here as it is. Nothing to borrow.

**Two fixes that do not apply to core.** `b83f1bdd21` (#38225) removed
`@torch.jit.script` from the audio VAE's snake activation, because the
profiling JIT changed its rounding after the first call and a repeated
reference-audio request stopped matching the first. Core's `snake` is plain
eager code (`comfy/ldm/minimax/audio_vae.py`). `a8e45f16cc` (#38506) adds a
shared INT8 embedding lookup for INT8, W4A8 and NVFP4 encoders; the encoder
this install loads keeps `model.embed_tokens.weight` in BF16 (read from the
header of `h3_config.MODELS["clip"]`, 2026-09-10).

**Skip-softmax is other silicon.** `0ea8378085` (#37959) adds a
request-scoped skip-softmax attention through FlashInfer's TRTLLM kernels,
dispatched only for compute capability 9.0 and 10.x
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/skip_softmax.py`).
It is one global threshold with a dense warm-up, the same blunt shape as a
single tau, and nothing of it runs on this card.

**Everything else is serving or other silicon**: XPU (#33366) and Xeon CPU
(#35147) enablement, GB300/GB200 and DGX Spark recipes (#38296, #37456),
per-phase warmup memory for residency calibration (#37916), and a docs sync
(#38784). Read, priced, no action.

## Fifth read, 2026-09-11

What landed in `coderef/sglang` between `887c401e15` and `593c7a900d`. It
changes no earlier verdict on this page, and none of it touches an H3 file.

**The RoPE refactor does not reach H3.** `d0035da34e` (#33555) moves DiT
RoPE onto a shared `RotaryEmbedding` custom op and edits the shared qk-norm
helpers (`runtime/layers/rotary_embedding/base.py`,
`runtime/layers/layernorm.py`) for Flux, Qwen-Image, GLM-Image and others.
H3's DiT imports neither: it keeps its own `MiniMaxH3Rope` and
`_apply_qk_norm` and calls `sgl_kernel`'s rotary directly
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`).

**Everything else is serving**: in-place pinning for layerwise-offload host
stores (#39021), IPC JIT recovery after an interrupted build (#39034), and
LLM-side work. Read, priced, no action.

## Sixth read, 2026-09-19

What landed in `coderef/sglang` between `593c7a900d` and `993d1fccba`. It
changes no earlier verdict on this page. Of the commits touching H3 files,
three matter here.

**The RTX 5090 recipe (`a25f213bc4`, #39373) is documentation, and it sets
memory placement only.** The command in
`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` (section on
the RTX 5090) picks the allocator, the fl2va variant, the `memory`
performance preset, what is offloaded and how many DiT layers stay resident.
The preset sets offload and residency defaults only
(`coderef/sglang/python/sglang/multimodal_gen/runtime/server_args/auto_tune.py`);
it chooses no attention backend, quantisation, compile or step cache.
*Note 2026-11-04: still so for those four at `origin/main`; for H3 the
`memory` preset now also sets the video VAE to one tile and one window
per decoder forward
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig`,
`validate_server_args`), a rounding-level change; see "VAE tiling is
silent here".* The
sampler settings on that page describe the benchmark run on both engines, not
a new sglang default. So the adopt-upstream rule has nothing new to act on:
the Sol table above still holds (sglang's `sol_attn` defaults and core's node
schema have not moved since 2026-09-11), and on canvas, steps, sampler and
text-encoder precision sglang and core still disagree with each other. Two
things on that page are easy to misread:

- **"kitchen int8" in its 24 GB row means INT8 for the DiT's Linear layers**
  (the page says it changes Linear numerics only). This repo's "kitchen" chain
  is an attention kernel. Same word, different stage.
- **The recipe's SDPA attention is a property of SM120**, where sglang
  defaults to SDPA (`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py`);
  the same command on SM89 gets FlashAttention. Its resident-layer count and
  bf16 DiT are for the 32 GB card.

The same page withdraws its earlier physical-4090 step time as an artifact of
an old pinning path and replaces it with a derived figure for a machine it
has not re-measured. Nothing in this repo quoted the withdrawn figure.

**VDN-H3 (`ff1ce11348`, #37903) is a different checkpoint, not a backend for
ours.** OpenVDN's `vdn-minimax-h3` adds a trained linear-attention branch and
an 8-step DMD2 LoRA to the H3 backbone, t2va and fl2va only. Its
`hybrid_window_attn_h3` backend replaces video-to-video self-attention with
an exact gated softmax over a window of neighbouring frame chunks (first and
last frames, text and audio rows dense) plus a bidirectional linear branch
for the rest
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/hybrid_window_attn_h3.py`,
`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3_vdn.py`).
The linear branch is trained, so it does not transfer to the base weights.
The 2026-09-17 rotation survey already lists it. `42b5af8c62` refreshes its
cookbook numbers.

**Everything else is serving, other models or docs**: out-of-tree platform
support, CFG and tracing docs, DSV4 and Qwen work, ROCm and XPU. Read at the
title, priced, no action.

## Seventh read, 2026-09-25

What landed in `coderef/sglang` between `993d1fccba` and `2f5c9ac43d`. Four
commits touch H3 paths. One of them is news.

**sglang now implements PDD (`973fb44471`, #40568).** Until this commit no
engine did ([`pdd/pdd_implementations.md`](pdd/pdd_implementations.md), section
1, corrected in place). How it works:

- Two offline tools, then an environment variable at serve time. The first
  merges the alibaba-pai adapter into the release's unpruned native
  transformer and copies the raw head stacks out
  (`coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_pdd_weights.py`).
  The second fuses each block of heads into one head per step with weights
  proportional to each sub-interval's `dsigma` on the shifted grid, computing
  in fp32 and storing bf16
  (`coderef/sglang/python/sglang/multimodal_gen/tools/fuse_minimax_h3_pdd_heads.py::fuse`).
  `SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS` points the final layer at the fused
  file (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3FinalLayer.load_pdd_fused_heads`).
- The step index is the denoise loop's counter, set per step, not recovered
  from `t` or sigma. The timestep stage raises unless the request's sigma grid
  equals the grid the heads were fused for
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py`,
  `_apply_pdd_schedule`).
- The cookbook recipe
  (`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`, "PDD
  acceleration LoRAs") counts sigma points, so its step count is eight
  evaluations. *Note 2026-11-04: since `32f34e898b` (#38671) sglang counts
  forwards and the recipe says `num_inference_steps: 8`; "Tenth read".* It uses the shifts the fusion tool defaults to and the default
  attention backend, which is not Sol. The fl2va adapter serves t2va and
  fl2va; ref2va has its own.

**Where it agrees with ours:** the fusion formula (`pdd_math.fusion_plan`),
the shifts (`h3_config.PDD_SHIFT`), the evaluation count (`h3_config.PDD_STEPS`), Euler at eta 0, reading alibaba-pai's raw
head stack as absolute heads rather than deltas, and failing closed on an
off-grid request. **Where it differs:** one uniform partition fixed when the
heads are fused, where ours fuses lazily from the sampler's sigmas and takes
any schedule. Also a loop counter for the step index where ours matches
`t_emb`, no strength or head-off control, the backbone merged offline into
unpruned BF16 weights where ours patches the pruned INT8 checkpoint or uses
the bake, and dense attention where our PDD graphs run Sol. That last one is
one upstream alone, so the adopt rule does not fire.

**A probable defect in its offline builder: the fc1 merge does not swap gate
and value.** The builder maps diffusers `ff.net.0.proj` onto native
`mlp.fc1` as a plain rename. The reader compared the two layouts in the
local release. Native `fc1` equals the diffusers weight with its halves
swapped, and does not equal it in the same order. sglang's own runtime LoRA
path swaps (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py::_swap_peft_swiglu_fc1_lora_b`),
and so do its checkpoint loader and our converter. The builder's only test
merges `to_out` alone. The layouts were measured; the merge was not run. So
the fc1 delta landing swapped is an inference, not an observation.
*Note 2026-11-04: `8776a94f29` (#46907) fixed it: the builder now swaps the
fc1 halves, with a unit test; "Tenth read".*

**The other three:**

- `abef3efb64` (#40378) fuses SwiGLU for quantized H3 MLPs with a kernel that
  rounds SiLU to bf16 before the multiply, the same rounding as its eager path
  and as core's eager `comfy/ops.py::_swiglu_eager`. It is a fusion, not a new
  convention. Whether kitchen's fused INT8 SwiGLU rounds SiLU internally is
  not established here: its CUDA source is not in the wheel.
- `2a0cb2f04e` (#40116) extends SubBlock's `sage_fp8` to SM120; the note is
  under "sglang's SubBlock router" in
  [`../sol_upstream.md`](../attention/sol_upstream.md). No SM89 path.
- `e6931ca889`, reverted by `f702a0be29`, re-landed as `ce06a14444`: each
  model's pipeline config now registers from its own file
  (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::register`).
  Every `coderef/sglang/...` path and `::symbol` cited under `docs/` still
  resolves. The FastH3 sentence in
  [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) that says "Registered under
  `registry.py`" is half stale and has a dated note.

**Sol-Attn defaults have not moved**: no new commit touches `sol_attn.py`,
and it is still opt-in, so the table under "Sol-Attn defaults: sglang against
core's and ours" holds. `50ec9702d0` adds a "Run in ComfyUI" section to the H3
cookbook naming a server-mode node, which agrees with
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) that sglang has no in-process
ComfyUI path for H3. The rest is serving, MiniMax-M3, HiSparse, ROCm and XPU,
read at the title.

*Corrected 2026-10-02: since `f1e62e3a2e` sglang has a ComfyUI path for H3
that runs the DiT under a ComfyUI graph, one step at a time in a separate
worker process. "No in-process path" is still literally true; "server mode
only" is not. See "Eighth read".*

## Eighth read, 2026-10-02

What landed in `coderef/sglang` between `2f5c9ac43d` and `89f21671bb`. Six
commits touch H3 paths. One retires a sentence in the seventh read, and the
re-read retires a sentence in [`../wiki/references.md`](../references.md)
that was already wrong when written.

**No default moved, and the adopt-upstream rule does not fire.**
`sol_attn.py` is unchanged, so the table under "Sol-Attn defaults" holds
(with the dated note on our dense-blocks cell). The H3 sample config still
runs fifty steps with guidance pinned at 1.0. The integrated mode's shifts
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py`,
`DEFAULT_SIGMA_SHIFT_VIDEO` and `DEFAULT_SIGMA_SHIFT_AUDIO`)
equal core's `comfy/supported_models.py::MiniMaxH3.sampling_settings` and
`h3_config.SIGMA_SHIFT`, which already agree. Its example graphs use
`res_multistep` on `simple`. They copy core's templates, so they echo one
upstream rather than adding a second one.

**sglang now runs the H3 DiT under a ComfyUI graph (`f1e62e3a2e`, #35990).**
The loader builds core's model object from the safetensors header, then swaps
its diffusion model for an executor that ships each sampler step over local
ZMQ, with CUDA IPC handles, to an sglang worker
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/generator.py::load_model`,
`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Executor`).
The worker runs one forward per call
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py::MiniMaxH3ComfyUIStepStage`).
ComfyUI keeps the encoder, both VAEs, core's H3 conditioning nodes, the packed
layout and the sampler. Two consequences:

- **None of our DiT-side patches would reach it** (reasoned). The diffusion
  model is replaced, so the Sol node, the dense attention chain, PDD heads in
  core's `final_layer` and core's LoRA loader all patch a model that no longer
  runs. sglang ships its own LoRA loader node for this mode.
  *Note 2026-11-04: since `5730345717` (#43495) the integrated mode refuses
  a graph that carries ComfyUI block patches, model patches or an attention
  override
  (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/base.py::_reject_model_patches`),
  so these patches are now refused at the first step instead of dropped.*
- **Two places read as departures from core's `MiniMaxH3Model.forward`**
  (reasoned, not run). `MiniMaxH3Adapter.unpack` scales the audio velocity by
  `time_shift_slope` on top of the carry correction, which core does not, and
  it does not multiply the output by the denoise mask, which core does. Its
  only audio test checks shapes. If anyone compares this mode against a stock
  graph, these are the first two places to look.
  *Note 2026-11-04: both are fixed by `b4822bae03` (#43504), which masks the
  velocities and drops the extra audio scaling; "Tenth read".*
  *2026-10-08: this bullet says what the adapter leaves out and not what the
  worker does with a mask, which is in the ninth read below. Both
  departures were read again at `214347891a` and stand.*

**Spectrum skip-step (`ae47bcd4da`, #35684) is opt-in**
(`SamplingParams.enable_spectrum` is False). It forecasts the
pre-final-layer hidden state of the target audio and video rows with a
Chebyshev ridge fit, holds audio at its last real value, blends video toward
the forecast, and runs the embed step and the final layer every step
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel._h3_spectrum_predict_targets`,
`coderef/sglang/python/sglang/multimodal_gen/runtime/cache/spectrum.py::SpectrumMixin.begin_spectrum_step`).
The skip schedule is a rule: real warm-up forwards, then a widening window
with no cooldown. That is DPCache's insertion point with a rule where DPCache
has a calibration. It already forecasts only the target rows, which
[`2026-09-25_step_caching_survey.md`](../archive/2026-09-25_step_caching_survey.md)
said no upstream did, and holds audio rather than weighting the two streams.
The survey's verdict stands (reasoned): nothing to skip at distilled step
counts, a second candidate at the base step count. Its code comment credits
"Comfy / Wan2GP", probably xmarre's Spectrum pack
([`../sol_upstream.md`](../attention/sol_upstream.md), "Other ComfyUI Sol-Attn
packs"); not verified.

**The same commit's "fused RMSNorm/AdaLN" landed unfused.**
`_modulate_rmsnorm_scale_shift` is the norm followed by `_modulate_scale_shift`,
and its docstring says the fused kernel was removed for drift. On the block
path, bf16 on CUDA,
`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py::_indexed_scale_shift_bf16_kernel`
rounds to bf16 at `1+scale`, at the product and at the stored sum. Core's
`comfy/ldm/minimax/model.py::_mod_scale_shift` is three in-place bf16 ops at
the same three points. That kernel has not changed since 2026-08-18
(`git log` on the file), so the seventh-read-era claim in
[`../wiki/references.md`](../references.md) that sglang keeps the affine
in fp32 and core rounds at more points was wrong when written; it is corrected
there. vllm-omni's fp32 affine is the outlier of the three.
*Note 2026-11-04: since `121aa8dcc7` (#43327) this holds at sglang's
`exact` tier only. At its default `lossless` the norm and modulation run as
one fp32 pass with one rounding, so core is the one of the three that
rounds more; "Tenth read".*

**`kitchen_int8` is now a deprecated alias of `convrot_int8` (`e667ab10d5`,
#38040).** Same quantization as kitchen's, with interchangeable weight and
scale tensors
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/quantization/configs/convrot_int8_config.py`).
`auto` picks its own JIT CUTLASS kernels on compute capability 9.0, 10.0, 12.0
and 12.1 and `comfy_kitchen.int8_linear` everywhere else, so on this card it is
still kitchen, and the `int8_convrot` files in `h3_config.MODELS` load
auto-detected. Nothing to adopt.
*Note 2026-11-04: `3472e16f96` (#45706) removed the `kitchen_int8` alias;
`convrot_int8` and its dispatch on this card are unchanged.*

**The rest:**

- `dc1bd46802` (#40470): an exact, bounded conditioning cache, on by default,
  that also stores the H3 VAE encode's moments before posterior sampling, so
  seeded sampling stays exact
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/cache/conditioning.py`).
  ComfyUI's node cache covers the same ground here.
- `0931a72eb2` (#34365): RL rollout for t2va only, a stochastic video-target
  update that returns log-probs, audio deterministic, off by default.
- `84622ce9d5` (#41272) is a crash fix for LoRA-wrapped linears and
  `bf8adf9602` puts int64 offsets in the SubBlock router kernels.
- Everything else is Qwen-Image, Flux 3, other models, NPU, MiniMax-M3 and
  docs, read at the title.

## Ninth read, 2026-10-08

A scoped read, not a walk of commits. The owner asked whether sglang does
anything with frozen audio, or anything special with a video that is edited,
continued or rendered in windows. Read at `214347891a`, the clone's head on
the day, in the H3 stage folder and the files named below; what was not read
is listed at the end. It is not a read of what landed since the eighth
read's `89f21671bb`. No renders.

**The answer is no, in sglang's own pipeline.** Each of these was already on
record and holds at this head:

- **A source track that is reused is a reference audio block and nothing
  else.** The target's audio starts as noise
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py::MiniMaxH3LatentPreparationStage`),
  is updated like any target row while the reference rows stay pinned
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::minimax_h3_denoise_loop`),
  and the file's track is the decoded latent, with no mux of the source's
  audio (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::MiniMaxH3DecodingStage`).
  [`../h3_audio_freeze.md`](../core/h3_audio_freeze.md) section 8 owns this; its
  dated note says which of its items were checked again.
- **There is no edit, continuation, window or partial-strength path.** One
  request is one clip inside the duration limits. The newest file in the
  folder with a promising name, `minimax_h3_rollout.py`, is the t2va RL
  rollout of the eighth read.
- **Sampling does not differ by task.** Every task row carries the same
  shifts
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`),
  and guidance and the negative prompt are refused at the door
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py::MiniMaxH3VideoModelAdapter`).
  [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) sections 1 and 8 have the
  detail.

**What moved: the stage that runs a DiT step for ComfyUI handles denoise
masks.** The eighth read described that stage and left this out. ComfyUI's
side forwards the video mask and the audio mask with each step
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Adapter`),
and the worker turns them into per-row timesteps with core's own formula, on
target rows only
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py::comfyui_payload_to_branch_inputs`,
`::_overlay_per_row_timesteps`). An audio mask that is one everywhere is
ignored (`::_audio_mask_values`), as in core. So a row frozen by a mask is
labelled clean there as it is under core's
`comfy/ldm/minimax/model.py::MiniMaxH3Model`. What the adapter still does not
do is scale the returned velocity by the mask, which core does; the eighth
read's bullet on that stands, so the labels agree and the output scaling
does not (reasoned, not run). *Note 2026-11-04: `b4822bae03` (#43504) adds
the mask scaling, so labels and output scaling both follow core at
`origin/main`; "Tenth read".* This is core's rule carried over so a ComfyUI
sampler can drive sglang's DiT. Nothing in sglang's request, task table or
native loop reaches it, so it is no evidence about training.
[`../h3_audio_freeze.md`](../core/h3_audio_freeze.md) section 8 said no mask path
existed anywhere in the serving code; it is corrected there.

**How its edit-shaped request differs from the whole-frame edit rendered
here** (a reference video at the canvas size, the source's audio frozen in
the target rows by a mask, a long clip in windows whose head frames are
frozen):

- **A sounded reference video always brings its soundtrack as an audio
  block, with its own label ahead of the video's.** The task table routes
  the video into the audio encoder with no switch
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`),
  and the label is emitted whenever the file has an audio stream
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py::MiniMaxH3TextEncodingStage`).
  Here the soundtrack is a block only when it is wired. A reference video
  with no audio block beside a frozen target track is therefore a state
  sglang cannot produce. The arm that carries both is the hybrid row of
  [`../h3_audio_freeze.md`](../core/h3_audio_freeze.md) section 2.
- **Its only way to pin a frame of the target is a still keyframe at the
  first frame, the last, or both**, and ref2va admits those beside its
  references
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES`).
  Core's layout places a keyframe or guide at any frame index
  (`comfy/ldm/minimax/model.py::PackedLayout`), and the window carry here
  freezes frames by mask, which is ComfyUI's mechanism and not the
  vendor's.
- **A reference video is resized to its own shape at the vendor's short
  edge, whatever the target's, and cut to the target's frame count.** The
  shape comes from the same resolver the canvas uses, called with the
  constant and not the request's short edge
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py::minimax_h3_prepare_for_queue`,
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py::minimax_h3_resolve_spatial_shape`);
  one ffmpeg pass sets the rate, the scale, the start offset and the frame
  cap
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_decode_reference_video_frames`).
  A source with the canvas's aspect ratio, handed in at the canvas size and
  the window's own length, is what that path would have made of it.
- **A request may leave the duration out and take it from its one
  audio-bearing reference**, snapped to the frame grid
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py::_resolve_deferred_temporal_shape`).
  Length is set by hand here.

The rest belongs to the reference comparison and is written up where that
lives, [`../h3_references.md`](../core/h3_references.md): where a reference
video's rows sit against the target's in time and space ("Edit a source
video"), and the condition noise level as a setting that both sides have,
set here by one node since `c7fcae2f` ("The vendor image path, stage by
stage").

**No default moved, and the adopt-upstream rule does not fire.** Nothing in
this read is a default of ours that sglang and core agree against.

**Not read:** `stages/visual_encoding.py`, `stages/replica_broadcast.py`,
`keyframe_encoding.py`, `packed_tokens.py`, `release_metadata.py`, most of
`video_adapter.py` and `resolved_plan.py`, the DiT forward beyond its mask
lines, the scheduler, the release's video processor config, and the ComfyUI
app beyond the H3 executor's pack and unpack. What happens to a reference
video shorter than the target is still untraced
([`../h3_references.md`](../core/h3_references.md), "Known limitations,
collected"). Line-number citations in the older sections of this file and
of [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) were not re-verified;
several have moved.
*Note 2026-11-04: the short reference video is now traced from source, in
the dated note on the last item of "Open after this read" above, and every
line-number citation into sglang in this file's older sections was checked
and repointed to a symbol where it had drifted.*

## Tenth read, 2026-11-04

A walk of what landed from the 2026-10-09 read's `f48ed2127e` to
`origin/main` as fetched on 2026-11-04. The working checkout sat behind that
and was not moved, so files were read with `git show origin/main:<path>`.
Of the commits in the range that touch `python/sglang/multimodal_gen/`,
the H3 ones were found by path and by message and read; the rest are other
models, multi-GPU, ROCm, NPU and Blackwell plumbing, and CI. A read-only
subagent did the walk; the claims below were then checked by hand against
the files cited. No renders.
[`../wiki/references.md`](../references.md), "What moved by
2026-11-04", has the other upstreams of the same day.

**No default moved for base H3, and the adopt-upstream rule does not
fire.** The shifts are still 12 and 3 (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py`,
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py::DEFAULT_SIGMA_SHIFT_VIDEO`), the visual condition
level is still 0.999 (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::MINIMAX_H3_IMGVID_COND_TIMESTEP`,
core's `VISUAL_COND_TIMESTEP`), the video VAE still encodes in fp32 and
decodes in fp16 (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig`),
`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py` is unchanged, and the text encoder only lost a cache
decorator (`dee613b283`, #47466). The one default where sglang and core
now differ, reference image sizing, has ours on sglang's side (below).
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py` and `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py` are unchanged, and the ninth
read's answer stands: sglang has no edit, continuation or window path, and
sampling does not differ by task. The new releases below differ by release.

**`num_inference_steps` now counts DiT forwards** (`32f34e898b`, #38671).
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py::minimax_h3_time_shift_sigmas` builds
`linspace(1, 0, N + 1)`, where it built `linspace(1, 0, N)`, and the
denoise stage's workload is N. Base H3 keeps the default of 50 and now runs
50 forwards where it ran 49; the cookbook says the trajectory changed
(`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`, section
4). The distills changed their label and not their behavior: FastH3 and
VDN-H3 went from 9 to 8 (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams`,
`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3_vdn.py::VDNH3SamplingParams`), and the PDD check is now that the step count
equals the head count (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py`). sglang now
counts steps the way ComfyUI's `simple` scheduler does. Our 16 steps
(`h3_config.SAMPLING`) were never tied to the vendor's 50, so nothing is
adopted. What this makes stale:

- The seventh read's "the cookbook recipe counts sigma points", above; the
  recipe now says eight. Dated note there.
- [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md), insight 20, and
  [`h3_dit_implementations.md`](h3_dit_implementations.md) section 4.2,
  each corrected with a dated note the same day.
- The docstring of `bench/map_cube_schedule.py::vendor_grid` says it
  replicates `minimax_h3_time_shift_sigmas` and reproduces the old
  `linspace(1, 0, N)`. It still agrees with the record it reads
  (`data/results/2026-09-04_sglang_cube_topk_schedule.json`,
  `recommended_50_steps`, whose `semantics` fields describe the old
  convention); at `origin/main` that list is what a request of 49 gets. A
  code file, left for a build.
- The Sol table's dense-steps row, "Sol-Attn defaults" above: sglang's 10
  dense steps are now exactly 10 of 50 updates where they were 10 of 49.
  Nothing to do.

**At the default `lossless` tier, the DiT's RMSNorm and AdaLN modulation
round once** (`121aa8dcc7`, #43327). `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::_modulate_rmsnorm_scale_shift`
takes `fused=True` at `lossless` and `high` and runs
`rmsnorm_indexed_scale_shift`, one fp32 pass with one bf16 rounding;
`exact` keeps the reference chain bit for bit
(`coderef/sglang/python/sglang/kernels/ops/diffusion/sites/minimax_h3_norm_modulate_site.py`,
its docstring). `lossless` is still the default
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py`, `quality`). So the eighth read's "the
same commit's fused RMSNorm/AdaLN landed unfused" now holds at `exact`
only: at its default, sglang keeps the affine in fp32 as vllm-omni does,
and core, which still rounds at three points
(`comfy/ldm/minimax/model.py::_mod_scale_shift`), is the one of the three
that rounds more. sglang and core disagree, so the rule does not fire.
Dated notes in the eighth read and in
[`../wiki/references.md`](../references.md), "What moved by
2026-10-09".

**sglang's ComfyUI adapter now follows core on both counts the eighth read
flagged** (`b4822bae03`, #43504).
`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Adapter.unpack`
multiplies the video and audio velocities by their denoise masks before
the carried-audio transform, with the comment "Match ComfyUI
MiniMaxH3._forward", and the extra `time_shift_slope` scaling of audio is
gone. With the ninth read's per-row labels, the adapter now agrees with
core on labels and on output scaling (read, not run). The same commit adds
FastH3 and VDN-H3 executors and an `SGLDH3DistilledSigmas` node
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/nodes.py`) whose FastH3 grid is the trained
rungs at shift 10, the values of `h3_config.FASTH3_CONTRACT_POSITIONS` (*retired 2026-11-04 with the FastH3 lane; in git at `358e6199`*).
`45de4adde9` (#44841) adds a worker-side patch profile,
`coderef/sglang/python/sglang/multimodal_gen/apps/comfyui_worker_patches/h3.py`, that swaps core's numerics into
sglang's DiT where they differ: the time embedding cast to bf16, the gate
through `addcmul`, the fused norm off, RoPE through kitchen's
`rms_rope_split_half_`, and the final layer on output rows only. It is a
vendor-written list of where the two DiTs differ op by op, and worth
reading as that. It pins `SOURCE_SHA256` of the DiT file and refuses a
changed one; whether the pin matches the file at `origin/main` was
reasoned by the subagent (it does not), not checked here.
*Note 2026-11-04, checked by hash the same day: the pin equals the SHA-256
of `runtime/models/dits/minimax_h3.py` at `b931b2c8bd` (#43634, 2026-10-11)
and at no other revision of that file in the range, including `45de4adde9`,
the commit that added the profile, and `origin/main`. So `install` raises
"H3 upstream source differs from the reviewed h3_comfyui_v1 source" at
`origin/main`. One of its overrides also targets
`minimax_h3_norm_modulate_active`, a DiT global that `329072d99a` (#44690)
removed. Apart from the pin, the profile refuses curve (pruned) AdaLN,
quantized layers and any card but Hopper, so it never covered the file
or card this repo runs.*

**Reference images have a resize mode** (`5727f08932`, #35839).
`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams.reference_resize_mode`
takes `match`, `max` or `diffusers`, default `diffusers`; the formulas are
in the comment above `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_REFERENCE_RESIZE_MODES`.
`diffusers` scales the short edge to 2048, upscaling a smaller still;
`match` matches the canvas's pixel area without upscaling, as core's
`MiniMaxH3ReferenceToVideo` does by default (`ref_image_size`); sglang's
`max` only shrinks a short edge past 2048. The comment says ModelTC's
Ref2VA Turbo LoRA was trained with `match`. This pack's
`reference_conditioning.py` defaults to `size_policy` `max` with
`allow_upscale` on, which is sglang's `diffusers` mode, so ours is still
the vendor's default; core's differs, and the rule does not fire. The
turbo LoRA is a closed lane.

**FastH3 Omni and Pika H3 Flex are served** (`d9ce511646`, #45530;
`37fc17b8d6`, #47378). Their contracts, and what each would take in
ComfyUI, are sections of their own in
[`../wiki/references.md`](../references.md): "FastH3 Omni" and "Pika
H3 Flex". Two things specific to sglang: VSA-H3 now tiles Ref2VA with each
reference video its own region and its own top-k, on sglang's native
layout only (a ComfyUI-serialized ref2va layout is refused;
`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py::_h3_segment_tile_geometry`,
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::_ref2va_vsa_segments`); and Pika's re-noise update
draws fresh noise inside the loop
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::_minimax_h3_renoise_target_rows_`), the first H3
release here that does.

**The PDD fc1 half-swap is fixed** (`8776a94f29`, #46907): the weight
builder now swaps diffusers' [value, gate] halves into the native [gate,
value] order, with a unit test. The record that measured the defect,
`../../bench/results/2026-09-25_sglang_pdd_fc1_merge.md`, was right for
the commit it read and carries a dated note. PDD quality work is a closed
lane.

**VSA-H3 runs on capability 8.0** (`465ddf9758`, #43823), through a
raw-pointer Triton load path, and is picked for FastH3 when no backend is
named, on a supported card only (`97bfa5cc53`, #43690;
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::FastH3PipelineConfig._default_to_trained_sparse_attention`).
`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::VSA_H3_CAPABILITIES` is 8.0, 9.0, 10.0 and
10.3, so an explicit request on this card's 8.9 still raises and an
unnamed one falls back to dense. Whether the pointer path would run on 8.9
was not tested. FastH3's sparsity is still the value of
`h3_config.FASTH3_CONTRACT_VSA` (*retired 2026-11-04 with the FastH3 lane;
in git at `358e6199`*).

**Smaller, with no effect here.** The video VAE decodes tiles in batches
at `lossless` (`0498ab7b6e`, #45179; `decoder_tile_batch`), which by its
comment "moves rounding, not the math"; the decode here is core's.
`114675a0d1` (#47852) fuses SiLU and padding in the VAE encoder, matched to
aten. `aa79087017` (#34978) makes `quality="high"` fail closed on
substituted weights. `329072d99a` (#44690) deletes the Euler-ancestral
scheduler file two of our docs cited; the loop's update is
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::_minimax_h3_update_target_rows_`, and both citations are
corrected. The SGLD app's Sol preset (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/web/options_state.mjs`) matches the
sglang column of "Sol-Attn defaults" above.

**Not read:** the configs FastH3 Omni and Pika H3 Flex ship on Hugging
Face; most of the VSA-H3 backend diff past its region tiling; the VAE fast
paths; the DiT diff past the fused norm and the PDD error text; the SGLD
app's server, preflight and generator; the cookbook past its H3 sections.
Many line-number citations in the older sections point into files this
range changed and were not re-verified; three were spot-checked and have
drifted: the pipeline config's quality validator, cited at lines 98-186, is
now `validate_quality_deployment`; the reference-video seed, cited at line
613 of the reference encoder, is
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED`
(corrected in [`../h3_references.md`](../core/h3_references.md) the same day);
and the cookbook's Sol recipe has moved down the file.
*Note 2026-11-04, a second pass the same day: every line-number citation
into sglang in the older sections was then checked against `origin/main`;
the drifted ones name symbols now, each with a dated note where it stands.
The VAE fast paths listed under "Not read" were read for the two VAE
sections above.*

## Eleventh read, 2026-11-04: a month of the pipeline, stage by stage

A second read the same day, at the owner's ask: what changed in sglang's H3
pipeline over a month, from the model code out to the related models (the
text encoder, the VAEs, references), judged against what the owner runs now.
That is base H3 with Sol-Attn through core on one RTX 4090, the masked lane
on ref2va, and Pika H3 Flex as the one distill they would test; they use no
sage, no PDD and no fast or turbo distill. The range is `1315d171a0`
(2026-10-03) to `origin/main` at `8884b4b421`, a week wider than the tenth
read's, so it also covers what landed between the eighth read and
2026-10-09. Four read-only subagents each read one stage, diffing every file
of it across the range and not only the commits that name H3; the claims
below were then checked by hand against the files cited. No renders and no
checks were run for it.

**No default moved that this pack ships, and the adopt-upstream rule does not
fire.** Wherever sglang moved, core disagrees with it or this pack already
matches it. The tenth read's findings stand; this one adds what a walk by
stage found beyond them.

**What did not move at all.** Across the whole range these have no diff:
`runtime/layers/attention/backends/sol_attn.py`, the DiT weight loader and
its ComfyUI checkpoint loader, and in the H3 stage folder `task_profiles.py`,
`canvas.py`, `resolved_plan.py`, `keyframe_encoding.py`,
`condition_noise.py`, `packed_sequence.py`, `packed_tokens.py`,
`presentation.py`, and the text, visual, audio and replica-broadcast stages;
also the encoder configs and the Qwen-VL RoPE. So canvas and length rules,
keyframes, condition noise levels, label order, time and space placement,
the reference-video VAE seed, initial noise, the text presentation, vision
token bounds, the encoder's layer tap and its bf16 precision are as the
earlier reads have them.

### Request and references

- **Reference images have three resize modes** (`5727f08932`, #35839), in
  the tenth read. The detail: `match` and `max` use core's formulas exactly,
  with the same `min(1.0, ...)`, Python rounding and 32-pixel floor
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_resolve_reference_image_shape`;
  core's `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo`). Only
  stills are affected. sglang ships two defaults of its own: native serving
  uses `diffusers`, and its ComfyUI example graph wires core's node with
  `match`
  (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/workflows/minimax_h3_r2v_sgld.json`).
  Its server-mode H3 node has no resize input. Pika H3 Flex is served at the
  2048 short edge at 768p, which is Pika's own rule and this pack's default.
- **A reference video shorter than the target, answered.** The code did not
  change; it was untraced. sglang decodes at most the target's frame count
  and pads nothing; the VAE encode then trims down to the 17n+5 grid with at
  least one step
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/processor.py::get_suitable_video_length`),
  so a reference under 22 frames raises "Cannot trim". The shorter latent
  advances the packed timeline by less
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py::minimax_h3_packed_sequence_ref2va_blocks`).
  Admission does not compare a reference's length with the target's. Core
  walks a short reference down to 17n+5 and accepts as few as 5 frames.
  Window references in the masked lane are far longer than 22 frames.
  [`../h3_references.md`](../core/h3_references.md), "Known limitations,
  collected", is corrected.

### Text encoder

Nothing reached what the encoder computes. `dee613b283` (#47466) drops the
cross-request cache of Qwen3-VL vision features; whole-presentation outputs
can still be cached. The ComfyUI app's conditioning cache used to key on a
tensor's shape and its first and last elements, so two conditionings that
matched there could be swapped; it is a hash of the embeddings since
`3458c94fad` (#43492). sglang now pins transformers 5.19.0 (`11972e5207`,
#42784); the ComfyUI venv here has 5.18.0, and this pack imports two of its
processors (`smart_resize` and `Qwen3VLVideoProcessor`, in
`reference_conditioning.py` and `reference_geometry.py`). Whether 5.19
changed them was not read. Core's `Llama2_.forward`, which turned
`bench/check_reference_encode.py` red, changed on its decode side only; the
encode path `reference_encode.py::_lm_pass` copies computes `freqs_cis`
itself, as core now does when none is passed (read, not run). The check's
own instruction stands.

### DiT forward

- **The norm and modulation fusion at the default tier** is the tenth
  read's. Its kernel computes the reciprocal RMS, the weight, `1 + scale` and
  the shift in fp32 with one store, and by its own docstring does not
  reproduce aten's reduction order
  (`coderef/sglang/python/sglang/kernels/ops/diffusion/norm/rmsnorm_scale_shift_triton.py::_rmsnorm_indexed_scale_shift_kernel`). Since `b4822bae03` it is mounted in the
  ComfyUI app's step stage too.
- **On pruned checkpoints the AdaLN path is fp32 at every tier, `exact`
  included.** This predates the range and was not on record for this path.
  sglang builds the projection in fp32 when the checkpoint has a curve AdaLN
  and refuses one that is not fp32 after load
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`,
  the `params_dtype` line and `post_load_weights`), and modulates with fp32
  tables (`1fd46ddd56`, #44180, fused that bit for bit). Core casts each
  table row to bf16 and rounds three times
  (`comfy/ldm/minimax/model.py::_mod_scale_shift`). The pruned INT8
  checkpoints this pack runs have that time table
  ([`2026-11-04_pika_h3_flex_weights_and_loras.md`](2026-11-04_pika_h3_flex_weights_and_loras.md),
  section 4), so the gap to sglang there does not close at its `exact` tier.
  It is a second face of "An enforced fp32 island" above, and only a capture
  could size it.
- The rest is speed with the bytes unchanged by the commits' own words, or
  multi-GPU, ROCm, NPU and Blackwell plumbing: the indexed modulation tiled
  by 2048 columns, QK-norm and RoPE two rows per warp, the attention section
  eager under torch.compile, outputs written into caller buffers, host syncs
  removed, and the Ulysses exchange pipelined.

### Sampling, LoRAs and Pika

- **Step counts** are the tenth read's: a request of N is N forwards.
- **LoRAs.** Under `merge_mode` "auto" an adapter stays unmerged when a merge
  would round away more than
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py::AUTO_MERGE_MAX_ROUNDING_LOSS`
  of it (`d8c5eca3ac`, #43385), and a quantized layer is never merged into:
  "auto" falls back to the unmerged delta and "merge" raises. At the default
  tier the unmerged delta is added inside the second GEMM with one rounding
  (`65386cb95a`, #43501); at `exact` it rounds twice. `MiniMaxH3LoRABranch`
  adds its delta with one rounding (`lora_branch.py`, `addmm_`).
- **Pika H3 Flex, as sglang serves it** (`37fc17b8d6`, #47378). The update
  is core's `lcm` with audio re-noised at its own shift-3 sigma, as
  [`../wiki/references.md`](../references.md), "Pika H3 Flex", and the
  day's weights record have it. Beyond them:
  - The re-noise draws come from a CUDA generator seeded with the request
    seed, video target rows then audio, in packed shape
    (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py`, `renoise_generator`;
    `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::_minimax_h3_renoise_target_rows_`).
    Pika's own code seeds the next seed and draws in latent shape; core's
    `sample_lcm` draws once over the whole latent. The same seed does not
    give the same clip across the three, so a same-seed comparison between
    engines is not an A/B (reasoned).
  - The adapter stays unmerged with its fp32 factors, never merged into
    quantized weights; Comfy-Org's template merges it into the INT8
    checkpoint, which the weights record finds keeps little of it.
  - Its default tier is `lossless` and `high` is refused, so sglang's Pika
    output is not bit-equal to Pika's reference forward.
  - Under sglang's Sol all six steps run dense, because `sol_attn.py`'s
    default `dense_steps` is 10 by step index. sglang is no evidence about
    Sol on Pika; how this pack's Sol window lands on six sigmas is a
    question of its own.
- **The ComfyUI app now passes the step index to the DiT**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py`), so Sol's dense steps count
  ComfyUI's steps there; before 2026-10-13 they would have been dense on
  every step (reasoned). The app refuses any ComfyUI model patch
  (`5730345717`, #43495), so this pack's Sol and masking patches cannot run
  through it, and its core-parity patch profile cannot install at
  `origin/main`: its pinned hash is the DiT file at `b931b2c8bd`, checked by
  hash (the tenth read had this as reasoned).

### VAEs

- **The decoder's fast path is on at the default tier**: fused QK RMSNorm
  and RoPE, cuDNN attention, and since `0498ab7b6e` (#45179) up to
  `decoder_tile_batch` tiles per forward, which by its comment "moves
  rounding, not the math"
  (`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py::MiniMaxH3VideoVAEConfig`).
  Core batches too, up to four tiles chosen from free VRAM at call time
  (`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE._decode_tile_row`), so a
  re-decode here can differ at the rounding level with memory state
  (reasoned). That matters for bit-identical rebuilds, not for anything
  visible.
- **sglang's decoder carries an fp32 residual; core's is fp16**
  (`284cda01fb`, #41906; `coderef/sglang/python/sglang/kernels/ops/diffusion/norm/h3_vae_fused_norm.py`,
  its docstring). The fused norm is switched by an environment flag that
  defaults on, not by the tier, and is tested to a tolerance, so sglang's own
  `exact` decode is not bit-equal to its eager module. diffusers was the
  first upstream on the fp32 side ("What moved by 2026-10-02" in
  [`../wiki/references.md`](../references.md)); unsized.
- **The encoder's SiLU and padding are fused** (`114675a0d1`, #47852),
  tested equal to aten; core already fuses the same step. Reference and
  source encodes are unaffected.
- **The audio VAE** changed in no behavior: a shared alias-free module, dead
  state removed, and audio decoded before video so the MP4 can start early
  (`9101e895ea`, #41819).

### Output

- **sglang pins its ffmpeg binary**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py::_resolve_ffmpeg_exe`,
  "keep the pinned encoder binary: x264 upgrades change lossy video
  pixels"). This pack's videos are written by VideoHelperSuite with whatever
  ffmpeg it picks, `/usr/local/bin/ffmpeg` on this machine, and
  `bench/sidecar.py` says the binary that wrote the bytes is not recorded.
  An ffmpeg upgrade here would change delivered pixels with nothing to show
  why, which touches comparisons of pieces across days and the re-composite
  tolerance. Recording the encoder's `-version` in the sidecar is the cheap
  fix; it is a build for the lead.
- **Encode settings**: a per-request preset was added (default `fast`,
  `ba68306256`, #41797); crf, pixel format and the absent bt709 tags are as
  before. This pack's crf, tags and rounding differ from both sglang's and
  core's; no rule fires.

### Not for the owner's use

FastH3's 8-step V2 and its batched decode, FastH3 Omni, VDN-H3, the PDD fc1
fix, and VSA-H3 on capability 8.0 (still not 8.9) are recorded in the tenth
read and in [`../wiki/references.md`](../references.md) as references
only.

**Not read:** the CUDA sources of the fused kernels, the Hugging Face configs
of Pika and Omni, the transformers 5.17 to 5.19 processor diff, the app's
server, generator and preflight beyond the lines cited, and the cookbook
beyond its H3 sections.
