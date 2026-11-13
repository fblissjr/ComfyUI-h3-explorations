# One render, stage by stage: code, owner, guard, reference

> **Current live implementation & rules (2026-10-27):** For the masked video-to-video lane's 7-stage pipeline, AST jump index, and stage contracts, see [`../masked_v2v/CODE_MAP_INDEX.md`](masked_v2v/CODE_MAP_INDEX.md), [`../masked_v2v/code_map_stages.md`](masked_v2v/code_map_stages.md), [`../masked_v2v/code_map_contracts.md`](masked_v2v/code_map_contracts.md), and [`../masked_v2v/chain_trace_and_remakes.md`](masked_v2v/chain_trace_and_remakes.md).

last updated: 2026-11-12 (added the `MiniMax-H3-Image` stage table and corrected the `P-H3-Image` text-encoder row for the 32B default vs. 4B `te4b` probes); 2026-11-11 (updated control-vector rows for `MiniMaxH3Steer` / `MiniMaxH3ReleaseReferences` and added `MiniMaxH3EditGuider` + `MiniMaxH3Preflight(edited_conditioning=...)`); 2026-11-07 (added the single-frame `P-H3-Image` stage table and updated the step-distillation and sigma-schedule rows); 2026-11-06 (PDD retired completely: the step-distillation, sigma-schedule, and sampler rows updated); 2026-11-04 (three owner cells point at the current divergence list, the retired vendor-gaps doc's successor; the video-decode cell corrected on diffusers; the attention, reference-sizing and sampler compare-against cells brought to sglang origin/main); 2026-10-27 (added router callout to `docs/wiki/masked_v2v/CODE_MAP_INDEX.md` and updated `CLAUDE.md` to `AGENTS.md`); 2026-10-04 (the masked video to video rows); 2026-09-11 (one reference cell renamed after the kitchen clone move; otherwise the 2026-08-28 page)

The cross-index. For each stage of a render: **our code**, the **document that
owns** it, the **check that would go red** if it broke, and the
**implementation to compare against** when you need to know what it should do.

**Written by a person. Not generated** — the generator that builds
[`index.md`](index.md) never touches this file.

**"nothing" in the guard column is the useful entry.** It is not a to-do list;
[`../../AGENTS.md`](../../AGENTS.md)'s rule is that a new check needs a drift instance the existing gates
provably could not have caught. It is there so nobody mistakes an unguarded
stage for a guarded one. The standing audit is
[`../checks.md`](ops/checks.md)'s uncontrolled-requirements table, which is the
authority; this column routes to it.

Compare-against names the implementation whose reading is most useful at that
stage, not the one that is right. [`references.md`](references.md) says what each
checkout is and is not evidence of.

---

## The model branch

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| checkpoint load | core `UNETLoader` | [`../research/sglang_comparison.md`](research/sglang_comparison.md) "Every known divergence" | `check_model_files.py` — a graph naming a file its loader cannot open goes red | LightX2V, for what the same int8 path does there |
| sigma shift | core `MiniMaxH3SigmaShift` | [`../../workflows/h3_config.py`](../../workflows/h3_config.py)'s `SAMPLING` note | `check_distill_settings.py` (shift and step count) | diffusers' named H3 scheduler |
| attention patch | core's `ModelAttentionBackend` on kitchen int8, the default since 2026-09-15; `attention.py`'s `MiniMaxH3SageAttention` on the sage arms | [`../SOLATTN.md`](attention/SOLATTN.md) | `check_attention_defaults.py` — by reachability, and values not presence | sglang (dense packed-varlen attention by default: FlashAttention when its package is installed and torch SDPA when not, cuDNN SDPA on Blackwell; `coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::CudaPlatformBase`, `get_attn_backend_cls_str`); LightX2V for the kernel choice on this card. *Corrected 2026-11-04: this cell used to say "sglang (dense FA varlen)", which leaves out sglang's Blackwell default (`ba12a16a62`, #33655).* |
| sparse attention | the vendored Sol node | [`../SOLATTN.md`](attention/SOLATTN.md) | same | `comfy-kitchen-kijai` (read-only; `comfy-kitchen-sol` until 2026-09-08), for sources that ship in no wheel |
| channel balance (off by default) | `channel_balance.py`, `MiniMaxH3ChannelBalance` | [`../h3_block49_quant_error.md`](attention/h3_block49_quant_error.md) | `check_channel_balance.py` — the fold is exact, RoPE-safe, off by default, and selects 45/48/49 on the shipped weights | the sage fork's `spike_h3_k_channel_balance.py`, the same fold graded on sage's kernel; `grade_channel_balance.py` on Sol's |
| chain order | the generator's wiring, `workflows/build_workflows.py::build_api` (`SageChainAssert` left every generated graph on 2026-09-17 and was deleted in 0.174.0) | [`../custom_node_gaps.md`](core/custom_node_gaps.md) | `check_attention_defaults.py`, by reachability and the dense floor; Sol's own composition log at render time | — |
| step distillation | `MiniMaxH3LoRABranch` (`lora_branch.py`) with `pika_h3_flex_{fl2va,ref2va}_svd995_bf16.safetensors` + `ManualSigmas` (`PIKA_MANUAL_SIGMAS`) + `lcm` for 6-step Pika H3 Flex (`MiniMaxH3PDDLoRA` and the five PDD8 graphs were retired on 2026-11-06) | [`../research/2026-11-04_pika_h3_flex_weights_and_loras.md`](research/2026-11-04_pika_h3_flex_weights_and_loras.md); [`../h3_pdd.md`](archive/h3_pdd.md) | `check_distill_grid.py`, `check_lora_branch.py` | `coderef/Pika-H3-Flex` |
| post-refiner control vectors & reference release | `MiniMaxH3Steer`, `MiniMaxH3WordVector`, `MiniMaxH3LoadControlVector`, `MiniMaxH3ReleaseReferences`, `MiniMaxH3MakeControlVector`, `MiniMaxH3MakeMediaControlVector` (plus legacy `MiniMaxH3ApplyControlVectors`; `control_vector_nodes.py`, `control_vectors.py`, `h3_spans.py`, `h3_pair_compiler.py`, `j_lens.py`) | [`control_vectors.md`](control_vectors.md); [`control_vectors/comfyui_nodes_and_controls.md`](control_vectors/comfyui_nodes_and_controls.md) | `check_control_vectors.py` — `diff_span` multi-span masks, shared `encode_pair_set` / `pool_pair_set`, float64 math, `mode="add_norm_matched"`, `target="scene_only"` / `fields` / `non_vision` bitwise preservation, `MiniMaxH3ReleaseReferences` (`end_percent`, `clean_level`), 5D `MiniMaxH3ImageModel` guards, `prefer_lora` stem resolution | `coderef/ComfyUI-MAINodes` (`dials/core.py`) |

## The conditioning branch

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| encoder load | `MiniMaxH3EncoderLoader` (`h3_encoder_loader.py`): core's own load plus the guards core lacks, on every generated graph. Graphs ship `h3_config.ENCODER_INT8`; the owner runs the bf16 pruned file, built from the release by `bench/convert_h3_bf16_encoder.py`. The AWQ adapter that was the alternate was deleted 2026-09-13 ([`decisions.md`](decisions.md)) | [`../h3_conditioning_end_to_end.md`](core/h3_conditioning_end_to_end.md) section 0 | `check_h3_encoder_loader.py` | — |
| prompt structure | `prompt_bank/<id>.txt`, loaded by id through `workflows/prompts.py` (the generator's constants were literals until 2026-09-03) | `vendor_guides/` | `preflight_graph.py` grades sections, markers, labels — **reports, never refuses**; motion vocabulary is **nothing** | sglang's presentation stage |
| text encode | core `comfy/text_encoders/minimax.py` | [`../research/official_weights_metadata.md`](research/official_weights_metadata.md); int8 against bf16 by token type: `../../data/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`; at the DiT: `bench/measure_encoder_quant_dit.py` | — | all four; they agree here |
| keyframes | `MiniMaxH3Conditioning`, `keyframe_canvas.py` (the geometry function only; its node was retired 0.173.0) | [`../h3_conditioning_end_to_end.md`](core/h3_conditioning_end_to_end.md) | `check_conditioning_behaviour.py`, against core as reference | sglang |
| reference ingestion | the three `AppendRef*` nodes | [`../h3_references.md`](core/h3_references.md) | `check_reference_runtime.py` — order, fps, duration, mono, policy | sglang, DiffSynth, diffusers |
| reference sizing | `reference_geometry.py`, `reference_report.py` (its pricing feeds `MiniMaxH3ReferenceConditioning`'s preview of both copies of every reference; the `MiniMaxH3ReferenceReport` node was deleted in 0.174.0) | [`../h3_references.md`](core/h3_references.md) | `check_reference_runtime.py`; the append node's defaults are vendor parity since 2026-09-13 and the **encoder view split is an option, unmeasured** (`bench/refview2_arms.json`) | diffusers, DiffSynth and sglang's default `diffusers` resize mode agree with the defaults; sglang's `match` and `max` modes are core's two modes, and its own ComfyUI example graph uses core's `match` ([`../h3_references.md`](core/h3_references.md), "Image references"). *Corrected 2026-11-04: this cell used to say "all three agree with the defaults"; sglang `5727f08932` (#35839) added the two modes beside its default.* |
| packing and positions | core `comfy/ldm/minimax/model.py` | [`../h3_geometry_and_nodes.md`](core/h3_geometry_and_nodes.md) | `check_reference_contracts.py` (all five contracts) | sglang; LightX2V matches core exactly |
| canvas and length | `MiniMaxH3Resolution`, `resolution.py` | [`../h3_resolutions.md`](core/h3_resolutions.md) | — | — |
| static pricing | `MiniMaxH3Preflight` (`edited_conditioning` optional), `bench/preflight_graph.py` | [`../checks.md`](ops/checks.md) | `check_edit_guidance.py` verifies peak `max(seq_orig, seq_edit)` pricing; reports, never refuses | — |

## Sampling, decode, output

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| sigma schedule | core `BasicScheduler`, or `ManualSigmas` (`PIKA_MANUAL_SIGMAS`) on Pika H3 Flex | [`../../workflows/h3_config.py`](../../workflows/h3_config.py) | `check_distill_grid.py` against the **vendor's published grid**, not a computed one | LightX2V's closed form |
| sampler & edit guider | core `SamplerCustomAdvanced` (`BasicGuider` or `MiniMaxH3EditGuider` in `edit_guidance.py`, with `streams` + `audio_scale` per-stream targeting) | [`../../workflows/h3_config.py`](../../workflows/h3_config.py); [`edit_guidance.md`](edit_guidance.md) | `check_distill_settings.py` checks `lcm` on Pika H3 Flex; `check_edit_guidance.py` checks `MiniMaxH3EditGuider` / `MiniMaxH3PromptEdit` (including 2-stream `NestedTensor` `streams` / `audio_scale`); **nothing** reads the sampler name off a base graph | sglang runs eta-0 Euler for base H3; we run a stochastic SDE. *Corrected 2026-11-04: since sglang `37fc17b8d6` (#47378) its Pika H3 Flex release re-noises inside the loop; this cell used to say "sglang runs eta-0 Euler" for every release.* |
| audio carry | core `comfy/model_sampling.py` | — | **nothing.** One schedule carries both streams; the other three run two | sglang, DiffSynth, diffusers |
| video decode | core `VAEDecode` | [`../research/sglang_comparison.md`](research/sglang_comparison.md) "Every known divergence" | — | diffusers decodes in the pipeline dtype with the encoder, norms and LayerScale scales kept fp32 (`coderef/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py::AutoencoderKLMiniMaxH3._keep_in_fp32_modules`); sglang decodes fp16 with an fp32 residual; we ship fp16. *Corrected 2026-11-04: this cell used to say "diffusers pins fp32", untrue since diffusers `51a454be9` (#14754).* |
| VAE precision | `MiniMaxH3VAEPrecision` (`vae_precision.py`, resetting `cloned.size = None` after casting) | [`../research/sglang_comparison.md`](research/sglang_comparison.md) "Every known divergence" | wired in `h3_probe_ref_vae_encoder_fp32_api.json` and `h3_image_edit*_api.json`; **three implementations run more precisely than the shipped video default** | diffusers, DiffSynth |
| audio decode | core `VAEDecodeAudio` | [`../h3_references.md`](core/h3_references.md) | — | agreement is broad here |
| mux and write | `VHS_VideoCombine` | — | **nothing.** sglang re-probes the written file and fails on drift; we write and stop | sglang's output adapter |
| provenance | `MiniMaxH3ProvenanceStamp`, `substrate.py` | [`../capture_manifest_schema.md`](ops/capture_manifest_schema.md) | `check_provenance_stamp.py`, `check_capture_manifest.py` | — |

---

## Two facts that cut across every stage

**A rendered clip cannot A/B a numerical change.** The trajectory diverges
completely from any perturbation, on any sampler, at frame 0. The output of a
changed arm is a *different sample*, not a degraded one.
[`../eval_comparison.md`](ops/eval_comparison.md) owns the process; [`../../AGENTS.md`](../../AGENTS.md)
owns the rule. Compare knobs at the call, not at the output.

Two riders on that were established 2026-08-28 and now live with the process
they constrain, under *What a matched seed does not match* in
[`../eval_comparison.md`](ops/eval_comparison.md): the audio stream is not
matched by a shared seed when arms differ in canvas or length, and the first
run after a state change is not the arm's settled behaviour. That file owns
them; this row exists so nobody rediscovers them from the stage table.

## Masked video to video

The lane's own page is [`masked_v2v.md`](masked_v2v.md). These rows are the
stages a masked render adds to the song chain.

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| cuts, the pick, the match across shots | `subject_track.py` (`cut_scores`, `main_subject`, `follow`) | the module docstring; [`masked_v2v.md`](masked_v2v.md) for the limits | `check_subject_track.py`, with stand-ins for SAM 3: it cannot check that the features tell real people apart | core's `SAM3_VideoTrack` with a text prompt, for what it does across a cut; `coderef/sam3` for the caller-mask request |
| tracking inside a shot | core `SAM3_VideoTrack` on its `initial_mask` path, called from `subject_track.py` | core | nothing | `coderef/sam3` |
| the region and what is encoded | `video_mask.py` (`grow`, `token_mask`, `window`) | the module docstring; [`../h3_audio_freeze.md`](core/h3_audio_freeze.md) section 4 | `check_video_mask.py` | core's `SetLatentNoiseMask` and mask resize, which the docstring says are not enough |
| the composite | `video_mask.py` (`changed_alpha`, `composite`) | [`../h3_audio_freeze.md`](core/h3_audio_freeze.md) section 4 | `check_video_mask.py` | nothing |
| the mask through two samplers | `plate_restore.py`, in no shipped graph | the module docstring | `check_plate_restore.py`, on a stub model: not run on H3 | core's `KSamplerX0Inpaint` |

## Single-frame image generation and editing (`P-H3-Image`)

The lane's wiki page is [`p_h3_image.md`](image/p_h3_image.md), and its prompting guide is [`../prompting/17_p_h3_image_prompting.md`](prompting/17_p_h3_image_prompting.md).

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| DiT load / LoRA branch | core `UNETLoader` (`p-h3-image_int8_convrot.safetensors`), or `MiniMaxH3LoRABranch` (`lora_branch.py`) with `h3/p-h3-image_lora_bf16.safetensors` (`rank=128`) on `unet_fl2va` | [`p_h3_image.md`](image/p_h3_image.md); [`../h3_image_editing.md`](image/h3_image_editing.md) | `check_model_files.py`, `check_distill_settings.py` | `diffusers` `src/diffusers/modular_pipelines/p_h3_image/` |
| text encoder load | `MiniMaxH3EncoderLoader` (`h3_encoder_loader.py`, 32B teacher default since `842a4108`), or core `CLIPLoader(type="minimax")` (`PH3ImageTEModel` in `comfy/text_encoders/minimax.py:172`, `Qwen3-VL-4B` + 5-layer `TextConnector`) on the two `te4b` probes | [`p_h3_image.md`](image/p_h3_image.md); [`../prompting/17_p_h3_image_prompting.md`](prompting/17_p_h3_image_prompting.md) | `check_model_files.py` (`PH3_IMAGE_CLIPLOADER_ENCODERS`) | `diffusers` `p_h3_image/encoders.py` |
| prompt upsampler (standalone node, not wired in generated workflows) | `MiniMaxH3PH3ImagePromptUpsampler` (`ph3_image_nodes.py`, `enabled=False`, `lazy=True` on `clip`) | [`p_h3_image.md`](image/p_h3_image.md); [`../prompting/17_p_h3_image_prompting.md`](prompting/17_p_h3_image_prompting.md) §17.4 | `check_schema_defaults.py`, `check_literal_widgets.py` | `diffusers` `p_h3_image/prompt_upsampler.py` |
| single-frame conditioning & reference streams | `MiniMaxH3PH3ImageConditioning` (`ph3_image_nodes.py`) fed by `prompt_bank/p_h3_image/*.txt` via `workflows/prompts.py` | [`p_h3_image.md`](image/p_h3_image.md); [`../prompting/17_p_h3_image_prompting.md`](prompting/17_p_h3_image_prompting.md) | `check_prompt_guide_conformance.py` (`ph3_image_prompts_conform`, `ph3_image_conditioning_geometry_conforms`) | `diffusers` `p_h3_image/{before_encoder,encoders}.py` and `comfy_extras/nodes_minimax_h3.py::PH3ImageConditioning` |
| post-refiner control vector steering (`P-H3-Image`) | `MiniMaxH3Steer` + `MiniMaxH3ReleaseReferences` (`control_vector_nodes.py`, `workflows/image/h3_p_h3_image_{t2i,edit}_control_vector_api.json`) | [`control_vectors.md`](control_vectors.md); [`p_h3_image.md`](image/p_h3_image.md) | `check_control_vectors.py` (`_is_same_family_h3_recipe_compatible`, `build_ph3_image_api(control_vector=...)`) | — |
| 8-step non-uniform schedule & decode | `MiniMaxH3SigmaShift(1.0, 1.0)` + `ManualSigmas(PH3_IMAGE_MANUAL_SIGMAS)` + `KSamplerSelect("euler")` + `PH3ImageVAE` (`comfy/ldm/minimax/vae.py:1014`) | [`../../workflows/h3_config.py`](../../workflows/h3_config.py) (`PH3_IMAGE_MANUAL_SIGMAS`) | `check_distill_settings.py`, `check_distill_grid.py` | `diffusers` `p_h3_image/before_denoise.py` |

## Official `MiniMax-H3-Image` (`workflows/image/h3_image_*`)

The owning page is [`../h3_image.md`](image/h3_image.md), and its standalone prompt-expansion suite is [`system_prompts/h3_image/README.md`](system_prompts/h3_image/README.md).

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| DiT, text encoder & optional LoRA / steering | core `UNETLoader` (`minimax_h3_image_pruned_int8_convrot.safetensors`, 1-stream `MiniMaxH3ImageModel` with 6 per-step PDD output heads) + `MiniMaxH3EncoderLoader` (`h3_encoder_loader.py`) + `ModelAttentionBackend("comfy kitchen attention")`; optional 1-stream `MiniMaxH3LoRABranch` (`lora_branch.py`) and optional `"h3_image"` `MiniMaxH3Steer` / J-Lens (`control_vector_nodes.py`, `j_lens_h3.safetensors`) | [`../h3_image.md`](image/h3_image.md); [`../../bench/results/2026-11-11_h3_image_weights.md`](../../bench/results/2026-11-11_h3_image_weights.md) | `check_model_files.py`, `check_attention_defaults.py`, `check_h3_image.py`, `check_control_vectors.py` | `coderef/sglang` (`minimax_h3_image.py`), `coderef/vllm-omni` (`pipeline_minimax_h3_image.py`), `coderef/diffusers` (`modular_pipelines/minimax_h3_image/`) |
| prompt expansion (`heylook` graphs) | `MiniMaxH3ImagePromptExpander` + `parse_expander_reply` (`h3_image_nodes.py`, `workflows/image/h3_image_{t2i,edit}_heylook_api.json`) | [`system_prompts/h3_image/README.md`](system_prompts/h3_image/README.md); [`../h3_image.md`](image/h3_image.md) | `check_h3_image.py`, `check_prompt_docs_sync.py` | `vendor_guides/h3_image/` (`MiniMax-H3-Image-PE`) |
| canvas size, reference conditioning & VAE precision | `MiniMaxH3ImageSize` (`h3_image_nodes.py`, `enforce_4mp_floor = True` + `target_width`/`target_height`) + core `MiniMaxH3ImageConditioning` (`comfy_extras/nodes_minimax_h3.py:711`, `0`-based `images.image_0..15`, `"Picture %d: "`) or experimental KV-prefix `MiniMaxH3EncodeImageReferences` + `MiniMaxH3PromptOnImageReferences` (`reference_encode.py`) + `MiniMaxH3VAEPrecision(encoder="fp32", decoder="unchanged")` on edit graphs | [`../h3_image.md`](image/h3_image.md) | `check_h3_image.py`, `check_reference_encode.py`, `check_literal_widgets.py` | `coderef/sglang` (`image_canvas.py`), `coderef/vllm-omni`, `coderef/diffusers` |
| 6-step PDD grid, decode & sub-4MP downscale | `ManualSigmas(H3_IMAGE_MANUAL_SIGMAS)` (`0.0` float32 error vs. `pdd_config.json`) + `BasicGuider` + `KSamplerSelect("euler")` + `PH3_IMAGE_VAE` (`p-h3-image_vae_bf16.safetensors` default since `5f180b87`; `H3_IMAGE_VAE` optional) + `MiniMaxH3ImageDownscale` (`h3_image_nodes.py`, 8-bit PIL Lanczos on sub-`4.0 MP` targets) | [`../../workflows/h3_config.py`](../../workflows/h3_config.py) (`H3_IMAGE_MANUAL_SIGMAS`, `PH3_IMAGE_VAE`); [`../h3_image.md`](image/h3_image.md) | `check_h3_image.py`, `check_model_files.py`, `check_literal_widgets.py` | `coderef/sglang`, `coderef/vllm-omni`, `pdd_config.json` |
