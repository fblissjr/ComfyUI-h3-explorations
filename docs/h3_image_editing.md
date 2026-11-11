# H3 as an image gen/edit model

Last updated: 2026-11-11 (the two one-frame guide edits on Pika load no edit LoRA and their prompt has no `vfx_edit:` opening, with two same-seed renders of the change; on 2026-11-10: added `h3_pika_image_t2i_control_vector_api.json` and `h3_pika_image_edit_control_vector_api.json`, the two plain Pika stills with a control vector node; retired `h3_p_h3_image_guide_edit_api.json`; added the two plain Pika stills, `h3_pika_image_t2i_api.json` and `h3_pika_image_edit_api.json`, prompted in the vendor guides' structure, and the `single_frame` input on `MiniMaxH3Conditioning` and `MiniMaxH3ReferenceConditioning` that gives them their one frame; added `h3_probe_pika_image_guide_edit_api.json` and `h3_probe_pika_image_guide_edit_control_vector_api.json`, one-frame edits on Pika with no reference picture, guide-branch edit-guidance wiring, and the section on them); 2026-11-09 (the `P-H3-Image` graphs default to the 32B encoder; the two probes now wire the distilled 4B encoder); 2026-11-08 (added `h3_p_h3_image_t2i_control_vector_api.json` and `h3_p_h3_image_edit_control_vector_api.json` bringing `workflows/image/` to 9 workflows); 2026-11-07 (reopened `workflows/image/` with PrunaAI `P-H3-Image` and guide-latent single-frame T2I and image-edit workflows using `MiniMaxH3PH3ImageConditioning` in [`ph3_image_nodes.py`](../ph3_image_nodes.py) plus `MiniMaxH3GuideLatent` in [`guide_latent.py`](../guide_latent.py), with `prompt` wired directly into `MiniMaxH3PH3ImageConditioning` and `ModelAttentionBackend(attention="comfy kitchen attention")`; no core patch). Previously updated: 2026-08-27, 2026-08-16.

## Reopened 2026-11-07: PrunaAI `P-H3-Image` & Guide-Latent Image Edit (`workflows/image/`)

ComfyUI core natively supports `PH3ImageTEModel` (`comfy/text_encoders/minimax.py:172`) and `PH3ImageVAE` (`comfy/ldm/minimax/vae.py:1014`), and this pack provides [`MiniMaxH3PH3ImageConditioning`](../ph3_image_nodes.py) (plus the standalone [`MiniMaxH3PH3ImagePromptUpsampler`](../ph3_image_nodes.py) node) and [`MiniMaxH3GuideLatent`](../guide_latent.py) for 1:1 parity with HuggingFace `diffusers` (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/`) and `MiniMax-H3`. Nothing in this pack patches ComfyUI core (`bench/check_vsa_core_patch.py` remains green; `archive/single_frame.py` stays archived).

`workflows/h3_config.py::GRAPH_DIRS` once again includes `SINGLE_FRAME_DIR = "image"`, and `workflows/build_workflows.py` generates the single-frame API workflows in [`workflows/image/`](../workflows/image/):

| Workflow | Mode & Default Canvas | DiT + LoRA | Text Encoder | Steps & Sigmas |
|---|---|---|---|---|
| [`workflows/image/h3_p_h3_image_t2i_api.json`](../workflows/image/h3_p_h3_image_t2i_api.json) | T2I (`2912x2912`, `~8.48 MP`, `t=1`) | `p-h3-image_int8_convrot.safetensors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`: `1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0`) |
| [`workflows/image/h3_p_h3_image_edit_api.json`](../workflows/image/h3_p_h3_image_edit_api.json) | 1–8 Image Edit (`canvas="auto"`, clamping `image_1` and output canvas to `4..8 MP` at `image_1`'s aspect ratio, `t=1`) | `p-h3-image_int8_convrot.safetensors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_p_h3_image_lora_t2i_api.json`](../workflows/image/h3_p_h3_image_lora_t2i_api.json) | T2I (`2912x2912`, `t=1`) via LoRA on `unet_fl2va` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `h3/p-h3-image_lora_bf16.safetensors` (`MiniMaxH3LoRABranch`, `strength=1.0`) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_p_h3_image_lora_edit_api.json`](../workflows/image/h3_p_h3_image_lora_edit_api.json) | 1–8 Image Edit (`canvas="auto"`, clamping `image_1` and output canvas to `4..8 MP`, `t=1`) via LoRA on `unet_fl2va` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `h3/p-h3-image_lora_bf16.safetensors` (`MiniMaxH3LoRABranch`, `strength=1.0`) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_probe_p_h3_image_t2i_te4b_api.json`](../workflows/image/h3_probe_p_h3_image_t2i_te4b_api.json) | T2I Distilled-Encoder Probe (`2912x2912`, `t=1`) | `p-h3-image_int8_convrot.safetensors` | `p-h3-image_text_encoder_int8_convrot.safetensors` (`CLIPLoader`, `type="minimax"`; `Qwen3-VL-4B` + 5-layer `TextConnector`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_probe_p_h3_image_edit_te4b_api.json`](../workflows/image/h3_probe_p_h3_image_edit_te4b_api.json) | Edit Distilled-Encoder Probe (`canvas="auto"`, clamping `image_1` and output canvas to `4..8 MP`, `t=1`) | `p-h3-image_int8_convrot.safetensors` | `p-h3-image_text_encoder_int8_convrot.safetensors` (`CLIPLoader`, `type="minimax"`; `Qwen3-VL-4B` + 5-layer `TextConnector`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_p_h3_image_t2i_control_vector_api.json`](../workflows/image/h3_p_h3_image_t2i_control_vector_api.json) | T2I + Post-Refiner Control Vector (`2912x2912`, `t=1`, `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` at `strength=1.15`) | `p-h3-image_int8_convrot.safetensors` + `MiniMaxH3ApplyControlVectors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_p_h3_image_edit_control_vector_api.json`](../workflows/image/h3_p_h3_image_edit_control_vector_api.json) | 1–8 Image Edit + Post-Refiner Control Vector (`canvas="auto"`, `t=1`, `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` at `strength=1.15, ref_vae_end_percent=1.0`) | `p-h3-image_int8_convrot.safetensors` + `MiniMaxH3ApplyControlVectors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | 8 Euler steps (`PH3_IMAGE_MANUAL_SIGMAS`) |
| [`workflows/image/h3_pika_image_t2i_api.json`](../workflows/image/h3_pika_image_t2i_api.json) | Text to image on Pika, one frame (`MiniMaxH3Conditioning`, `single_frame=True`; the prompt in the base guide's three fields; the family's square canvas) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `PIKA_FL2VA_SVD995_LORA` (`MiniMaxH3LoRABranch`) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | `PIKA_STEPS` steps on `PIKA_SAMPLER` (`PIKA_MANUAL_SIGMAS`) |
| [`workflows/image/h3_pika_image_edit_api.json`](../workflows/image/h3_pika_image_edit_api.json) | A picture edited at one frame on Pika (`MiniMaxH3ReferenceConditioning`, `single_frame=True`; the picture as `<Picture 1>`, sized to the output; the prompt in the reference guide's six sections; no guide node, no edit LoRA) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `PIKA_REF2VA_SVD995_LORA` (`MiniMaxH3LoRABranch`) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | `PIKA_STEPS` steps on `PIKA_SAMPLER` (`PIKA_MANUAL_SIGMAS`) |
| [`workflows/image/h3_pika_image_t2i_control_vector_api.json`](../workflows/image/h3_pika_image_t2i_control_vector_api.json) | The text-to-image still above with a control vector node (the vector, strength and mask are node `115`'s inputs in the graph; every section mask runs, since `MiniMaxH3Conditioning` attaches the section map) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `PIKA_FL2VA_SVD995_LORA` (`MiniMaxH3LoRABranch`) + `MiniMaxH3ApplyControlVectors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | `PIKA_STEPS` steps on `PIKA_SAMPLER` (`PIKA_MANUAL_SIGMAS`) |
| [`workflows/image/h3_pika_image_edit_control_vector_api.json`](../workflows/image/h3_pika_image_edit_control_vector_api.json) | The edited still above with a control vector node (the vector, strength and mask are node `115`'s inputs in the graph; the reference inputs stay at the node's defaults, which hold the picture on every step) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `PIKA_REF2VA_SVD995_LORA` (`MiniMaxH3LoRABranch`) + `MiniMaxH3ApplyControlVectors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | `PIKA_STEPS` steps on `PIKA_SAMPLER` (`PIKA_MANUAL_SIGMAS`) |
| [`workflows/image/h3_probe_pika_image_guide_edit_api.json`](../workflows/image/h3_probe_pika_image_guide_edit_api.json) | One-frame edit on Pika, no reference picture (the picture to edit is the guide: core's `MiniMaxH3ReferenceToVideo` + `MiniMaxH3GuideLatent`, `single_frame=True`; the family's square canvas) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `PIKA_REF2VA_SVD995_LORA` (`MiniMaxH3LoRABranch`; no edit LoRA since 2026-11-11) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | `PIKA_STEPS` steps on `PIKA_SAMPLER` (`PIKA_MANUAL_SIGMAS`) |
| [`workflows/image/h3_probe_pika_image_guide_edit_control_vector_api.json`](../workflows/image/h3_probe_pika_image_guide_edit_control_vector_api.json) | One-frame edit on Pika + Post-Refiner Control Vector (`768x768`, `t=1`, `film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors` at `strength=1.15, mask="non_vision", ref_vae_end_percent=1.0`) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `PIKA_REF2VA_SVD995_LORA` + `MiniMaxH3ApplyControlVectors` (no edit LoRA since 2026-11-11) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`) | `PIKA_STEPS` steps on `PIKA_SAMPLER` (`PIKA_MANUAL_SIGMAS`) |

### What Was Fixed vs. PrunaAI's Reference ComfyUI Workflows & ComfyUI Core's `PH3ImageConditioning`

1. **Replaced `ModelSamplingSD3(shift=1.0)` + `BasicScheduler(simple, 8)` with `MiniMaxH3SigmaShift(shift_video=1.0, shift_audio=1.0)` + `ManualSigmas(PH3_IMAGE_MANUAL_SIGMAS)`**:
   - PrunaAI's ComfyUI example workflows wired `ModelSamplingSD3(shift=1.0)` + `BasicScheduler(scheduler="simple", steps=8)`, which produces a uniform linear schedule `[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]` instead of the non-uniform 8-step distillation schedule `1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0` in `PrunaAI/P-H3-Image`'s `modular_model_index.json` (`diffusers` `src/diffusers/modular_pipelines/p_h3_image/before_denoise.py`).
   - Also wires `ModelAttentionBackend(attention="comfy kitchen attention")` for single-frame dense attention (matching the dense attention backend across H3 video workflows; Sol-Attn is a 3D spatiotemporal video kernel and is exempt on single-frame graphs).
2. **Removed the Prompt Upsampler from Generated `workflows/image/` Graphs (`prompt` Wired Directly into Node `5` `MiniMaxH3PH3ImageConditioning`)**:
   - PrunaAI's T2I workflow wired `ChatTemplate` (not a core node) or `TextGenerateLTX2Prompt` (which injects Gemma `<start_of_turn>` tags and LTX-2's video system prompt), and its Edit workflow wired `TextEncodeQwenImageEditPlus` (which accepts at most 3 images, downscales them to `384x384`, injects Qwen-Image-Edit's system prompt, and outputs `CONDITIONING` rather than calling `clip.generate`).
   - Per owner decision (2026-11-07), the generated `workflows/image/` graphs omit the prompt upsampler entirely and wire `prompt` directly into node `5` (`MiniMaxH3PH3ImageConditioning`). `MiniMaxH3PH3ImagePromptUpsampler` (`ph3_image_nodes.py`) remains registered in the node pack as a standalone node if ever needed manually.
3. **Replaced ComfyUI Core's `PH3ImageConditioning` with `MiniMaxH3PH3ImageConditioning` (Node `5`, `ph3_image_nodes.py`)**:
   - ComfyUI core's `PH3ImageConditioning` (`comfy_extras/nodes_minimax_h3.py:668-731`) only downscales reference images (`_size_within_area`), couples explicit `width, height` to the VAE reference area (`area = width * height`), omits float16 conditioning-latent rounding, and exposes 9 image slots instead of 8.
   - `MiniMaxH3PH3ImageConditioning` implements 1:1 parity with `diffusers` (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py`, `coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/encoders.py`) and `MiniMax-H3` (`coderef/MiniMax-H3/Ref2VA/video_vae/klvae.py:910`):
     - Under `canvas="auto"` (default), clamps `image_1` and the output canvas to `[4 MP, 8 MP]` (`4*1024^2..8*1024^2`) at `image_1`'s aspect ratio (scaling a `1024x1024` `image_1` up to `2048x2048`), clamps `image_2..image_8` to `[1 MP, 8 MP]`, and resizes Qwen3-VL `vision_images` via `size_at_area` (`~1 MP`, both up and down).
     - Under `canvas="explicit"`, uses `round_canvas(width, height)` for the output canvas while keeping edit reference images clamped to `[1 MP, 8 MP]`.
     - Calls `z = vae.encode(image_tensor)` without mutating `vae.first_stage_model.latents_mean` or `latents_std` (so `ModelPatcherDynamic.backup_buffers` is never poisoned before `VAEDecode`), recovers unnormalized `z * std + mean`, rounds to `torch.float16` before normalizing `(rounded - mean) / std`, and attaches `_h3_span_masks` (`h3_spans.py`).
4. **LoRA-on-`unet_fl2va` (`MiniMaxH3LoRABranch`), the 32B Text Encoder Default and the 4B Distilled-Encoder Probes**:
   - `h3_p_h3_image_lora_t2i_api.json` and `h3_p_h3_image_lora_edit_api.json` apply `h3/p-h3-image_lora_bf16.safetensors` (`rank=128` across all 50 DiT blocks + `token_refiner`) onto our existing `minimax_h3_fl2va_pruned_int8_convrot.safetensors` via `MiniMaxH3LoRABranch` (full bf16 rank without int8 merge quantization).
   - Every `h3_p_h3_image_*` graph wires our `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`MiniMaxH3EncoderLoader`, node `2`) into `MiniMaxH3PH3ImageConditioning` (`workflows/build_workflows.py::build_ph3_image_api`, `teacher_te=True`). The owner chose 32B as the default on 2026-11-09 after reporting better quality from it: the image DiT's text path was trained against 32B-style rows that the 4B `TextConnector` only approximates, and the pack's control vectors are built from 32B rows.
   - `h3_probe_p_h3_image_t2i_te4b_api.json` and `h3_probe_p_h3_image_edit_te4b_api.json` wire the release's distilled 4B encoder (core `CLIPLoader`, `type="minimax"`, `p-h3-image_text_encoder_int8_convrot.safetensors`, `Qwen3-VL-4B` + `TextConnector`) for same-seed A/B comparison against the 32B default.
5. **Prompting Documentation**:
   - Full `P-H3-Image` prompting rules (`p_h3_image_t2i` and `p_h3_image_edit`) and upsampler documentation live in [`docs/prompting/17_p_h3_image_prompting.md`](prompting/17_p_h3_image_prompting.md) and [`docs/wiki/system_prompts/h3_p_h3_image_system_prompt.md`](wiki/system_prompts/h3_p_h3_image_system_prompt.md).

### A one-frame edit on Pika (`h3_probe_pika_image_guide_edit_api.json` and `h3_probe_pika_image_guide_edit_control_vector_api.json`, 2026-11-10)

The two graphs in `workflows/image/` that take nothing from `P-H3-Image` but its image VAE. The picture to edit goes in
as the guide ([`MiniMaxH3GuideLatent`](../guide_latent.py), `single_frame=True`) and Pika H3 Flex runs on its own
sampler and sigmas (`workflows/h3_config.py::PIKA_SAMPLER`, `PIKA_MANUAL_SIGMAS`). No reference picture is wired,
so the conditioning node is core's `MiniMaxH3ReferenceToVideo`: `MiniMaxH3ReferenceConditioning` refuses an empty
reference chain (`h3_probe_pika_image_guide_edit_control_vector_api.json` therefore sets `mask="non_vision"` on
`MiniMaxH3ApplyControlVectors`, since core's node attaches no `h3_span_masks` and a zero-reference one-sentence prompt
carries no vision or `<d>` spans). Both rows are in [`workflows/build_workflows.py`](../workflows/build_workflows.py)
(`native_ref=True` with `guide_image=True` and no references).

**No edit LoRA and no `vfx_edit:` opening since 2026-11-11 (the owner's word).** Until that day both graphs chained a
guide edit LoRA (`workflows/h3_config.py::GUIDE_VFX_EDIT_FFP_LORA`, node `72`) after Pika's and their prompt opened
with `vfx_edit:`, that LoRA's trigger. Both are out: the graphs load Pika's LoRA alone
(`bench/check_guide_latent.py` holds the one-frame graph to that) and the prompt is the same sentence without the
opening, [`prompt_bank/guide_edit/man_red_jacket.txt`](../prompt_bank/guide_edit/man_red_jacket.txt).
`h3_guide_video_edit_api.json` was not part of that word and still loads the edit LoRA.

**The prompt is one plain sentence, not the house sections.** It says the edit first and then what stays, and names
no `<Picture 1>` because no reference picture is wired. The sentence came from the edit LoRA author's form (the notes
in [Alissonerdx/Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI), its VFX edit page and the
prompt template inside its workflow; this repo carries neither), which also asks for the `vfx_edit:` opening.
`bench/check_prompt_guide_conformance.py::_STRUCTURE_PROBES` still waives the section cases for both graphs. Whether
this graph should be prompted in the reference guide's sections now that no LoRA asks for the sentence has not been
tried.

**What was rendered on 2026-11-11, without the edit LoRA and the opening.** Two of the 2026-11-10 arms below
(`c2_man_cap_pika`, `c2_woman_jacket_pika`) queued again with node `72` taken out and `vfx_edit: ` cut from the
prompt, every other input as it was: the same pictures, seed and sizes. One seed and two pictures, so a first look and
not a ranking. The base model's reading of a guide with no edit LoRA had not been rendered before this.

- The edit landed on both pictures: the cap is gone with short dark hair in its place, and the sweater is a red
  leather jacket. Face, hair, pose, framing and background are kept in both.
- Beside the 2026-11-10 render of the same arm, the jacket differs in kind: zipped to the neck with the edit LoRA,
  open over a white top with lapels and snaps without it, so more of the picture below the neck is new. On the cap
  picture the two renders are close.
- Renders `n1_man_cap_no_edit_lora` and `n1_woman_jacket_no_edit_lora`, the sheets
  `sheet_guide_edit_without_edit_lora_*` (source, with, without) and the queued graphs are in the same
  `pika_still_trial` folder.

**What was rendered on 2026-11-10, with the edit LoRA and the `vfx_edit:` opening.** Everything in this list, and the
vector and release sweeps further down that name the guide edit graphs, was rendered on the graphs as they stood that
day, with the edit LoRA on node `72`. One seed and two pictures of people, so a first look and not a ranking. The
renders, the sheets that lay them side by side and the graph each was queued with are on the output share under the
folder `pika_still_trial` in `Image`.

- With no reference picture the edit landed on Pika on both pictures (a garment replaced, a cap removed) and the rest
  of each picture was kept, on the family's canvas and on larger ones up to a picture's own size.
- Beside the `P-H3-Image` edit graph on the same pictures, Pika's result was softer in fine detail such as skin
  texture. Beside the same guide edit run on the `P-H3-Image` LoRA, it stayed closer to the source in hair, color and
  expression.
- With a reference picture wired, both arms failed. At the generator's default reference size (`size_policy` `max`)
  the reference took over the one-frame target and the picture to edit was lost; sized to the output (`match`) the
  picture was kept, but the reference's background came across where its head was asked for. The author's notes ask
  for a reference that is one thing cut out on its own. That was tried later the same day, on one picture and one
  seed: a head on a plain backdrop as the reference, sized to the output and at the small size the author's own
  workflow feeds, with a head replacement asked for. The picture came back unchanged in both arms, at both sizes.
- A control vector (`MiniMaxH3ApplyControlVectors`, `mask` `non_vision`, `ref_vae_end_percent=1.0`, `ref_vae_aug=0.999`,
  `GUIDE_VFX_EDIT_FFP_LORA` `end_percent=1.0`) changed the render at the same seed; neither of the two vectors tried
  gave the look it is named for outside the edited garment. In code (`comfy/model_base.py:2302-2306`,
  `comfy/ldm/minimax/model.py:375-389, 535-549`), `MiniMaxH3GuideLatent(single_frame=True, frame_idx=0)` places the
  source still in `payload["cond_video_latents"]` at `cond_t = cursor + 0`, the target frame's own `(t=0, h, w)` RoPE
  coordinates; `control_vector_nodes.py:1244-1274` already gates `cond_video_latents` when `ref_vae_end_percent < 1.0`
  or `ref_vae_aug < 0.999`, and `node 72` (`MiniMaxH3LoRABranch`) gates `GUIDE_VFX_EDIT_FFP_LORA` via `end_percent`.
- The graph as generated was rendered once with a picture of a man in place of the placeholder, which was not on the
  machine: the clothing was replaced and the rest kept.

**Plain H3 prompting at one frame, with no guide and no edit LoRA (later the same day).** The owner's direction,
2026-11-10: Pika is a distill of the video model and was not trained on prose or on the edit LoRA's sentence, so a
Pika still is prompted as the video is, in the vendor guides' structure (`vendor_guides/base_en.md` on the fl2va
checkpoint, `vendor_guides/ref_en.md` on ref2va). One seed each:

- Text to image on the fl2va checkpoint with the Pika fl2va LoRA and `MiniMaxH3Conditioning`: a clean picture of the
  prompt's scene, with the prompt as one paragraph and with the same paragraph inside the base guide's three fields.
  The two differ in composition; neither failed.
- An edit on the ref2va checkpoint with the Pika ref2va LoRA and `MiniMaxH3ReferenceConditioning`: the picture as
  `<Picture 1>`, sized to the output, a six-section prompt, no guide node and no edit LoRA. The edit landed and the
  rest of the picture was kept. This is the reference-only shape of the 2026-08 graphs below, on Pika. The reference
  recipe that failed above is a different graph: a guide, the edit LoRA and a reference together.
- The same forms on `P-H3-Image`: both rendered the scene for text to image, and the six-section prompt did the edit.
  The image model's own upsampler (`MiniMaxH3PH3ImagePromptUpsampler`), run once for each, wrote one paragraph for
  text to image and the six-section brief for the edit.

Those renders took their one-frame latent from the second output of `MiniMaxH3PH3ImageConditioning` and used none
of its conditioning, because no H3 conditioning node could hand the sampler one frame. The sheets are
`sheet_h3_forms_t2i` and `sheet_h3_forms_edit` in the same folder on the output share.

**The two graphs that ship from it.** `h3_pika_image_t2i_api.json` and `h3_pika_image_edit_api.json`, the plain
Pika stills. Each has a twin with a control vector node, `h3_pika_image_t2i_control_vector_api.json` and
`h3_pika_image_edit_control_vector_api.json`; both rendered as shipped beside the plain graphs
(`sheet_vector_twins_as_shipped` in the trial folder). The phone studio drives both, and the guide twin, with the vector's
mask, mode, step window and typed words set on its tab; four such renders went through its own API the same day
(`sheet_through_the_studio_vector_controls`). `MiniMaxH3Conditioning` and `MiniMaxH3ReferenceConditioning` take a `single_frame` input: on, the node
hands the sampler one frame and `length` is not used, so `MiniMaxH3Resolution` stays on the video grid and nothing
in core is patched. It is text to image on the first node (a keyframe is refused with it) and still references only
on the second. The prompts are `prompt_bank/pika_image/t2i_clockmaker.txt`, the three fields of
`vendor_guides/base_en.md`, and `prompt_bank/pika_image/edit_man_red_jacket.txt`, the six sections of
`vendor_guides/ref_en.md` with `[keyframe completion]`: that guide's task type for an edited frame, which
`bench/check_prompt_guide_conformance.py` accepts on a reference node with `single_frame` on. The edit graph sizes
the picture to the output (`ref_size_policy="match"` in the generator), as the render above did; `max` was not
rendered on this form. A still's two audio fields read `N/A`, as the image model's upsampler writes them; the base
guide gives `N/A` for music only.

Each was rendered once as shipped on 2026-11-10, after ComfyUI restarted with the new input: the text-to-image
graph gave the same picture as the earlier render to the eye, and the edit graph, with a square picture of a man in
place of the placeholder (which was not on the machine), replaced the clothing and kept the face, pose, framing and
backdrop. The edit graph's canvas is square. A picture of another shape wants the canvas at its own shape: the same
form on a square canvas with a portrait picture still made the edit, but recomposed the frame and drew surroundings
that are not in the picture (`sheet_edit_canvas_shape` and `sheet_shipped_pika_image_graphs` in the same folder).

**Beside the `P-H3-Image` edit at the same size (2026-11-10).** The edit graph at the picture's own size, with the
six-section brief the image model's upsampler wrote, made the edit and kept the picture, with no breakup. Next to
the `P-H3-Image` edit of the same picture, prompt and size, both kept the face's freckles and skin texture close to
the source; the jackets differ in design. The owner looked at the three renders that day (the `P-H3-Image` edit,
and the Pika edit at that size and at the family's) and could not pick between them. One picture and one seed, so
this says the Pika edit is not visibly behind on that picture, and nothing about speed, other pictures, or text to
image, which has not been rendered on Pika at the image model's size. The sheets are
`sheet_plain_pika_edit_against_ph3_same_size` and the two `crops_plain_pika_edit_against_ph3_*` in the same folder.

**Control vector inputs on the plain Pika edit, tried once each (2026-11-10).** `MiniMaxH3ApplyControlVectors`
added to the edit graph: one picture, one seed, the family's portrait canvas, the `claymation_stopmotion` vector
unless said. The sheets are `sheet_knobs_*` in the same folder and every graph is under its `how_it_was_made/graphs/`.
What was seen, for the owner's eye to judge:

- **Strength.** At 1.5 the hair and the jacket's surface changed and the face and background stayed. At 3.0 the frame
  became a clay figure and the person and the framing were gone.
- **Step window.** The vector on the first half of the schedule alone gave about the picture it gives on all of it;
  on the second half alone the picture stayed close to the edit with no vector. The window counts whole steps
  (`control_vector_nodes.py::_step_in_window`) and a still's schedule is short (the `ManualSigmas` node of each
  graph), so the window has only as many positions as the schedule has steps.
- **Mask.** `scene_only`, `look_only` and `description_only` gave three different pictures; the hair changed only
  under `scene_only`. The node logs how many rows each mask steered.
- **Mode.** `amplify` stayed close to the edit with no vector; `project_out` changed the jacket. `hold` does nothing
  on one vector node: it removes a direction from the shift earlier vector nodes added
  (`control_vectors.py::_apply_holds_to_shift64`).
- **Typed `words`, no file.** At 1.5 the jacket and hair changed without the look the words name; at 3.0 a cartoon
  clay doll replaced the picture. The node logs which words its list matched.
- **The picture's weight and its release (`ref_vae_aug`, `ref_vae_end_percent`): read these as the fault, not the
  inputs.** These tries ran before `85aba935`. Until that commit a released picture was not let go: the node zeroed
  its latents and core filled the rows with noise and labeled them with the stream's own time, so the model was
  handed noise as its reference. That commit drops the rows on released steps
  (`control_vector_nodes.py::_drop_visual_cond_from_layout`). The release stills were rendered again on it after the
  server's restart, same graphs and seed (`sheet_release_before_and_after_the_fix`): the stray objects and streaks
  are gone, the held still came out pixel for pixel as before, and the node logs the rows it released. With the
  release clean, letting the picture go for the second half of the schedule left this picture close to the held
  one, with the vector and without it. Then the hold was swept (`sheet_release_sweep_how_many_steps_the_picture_is_held`):
  the picture held for none, one, two, three or all of the steps, with no vector and with `claymation_stopmotion`,
  `balloon_head_inflate` and `sunken_coral_reef`. Every still came out clean, the one that never held the picture
  included, and the person stayed recognizable in all of them: the picture's vision rows in the text carry her, the
  held copy carries fine texture (the hair's frizz went smooth once it was let go). An earlier release changed the
  hair and the jacket's cut a little; it did not make the reef or the swollen head appear at these strengths.
  - Before the commit: released for the second half of the schedule, stray colored objects and streaks appeared,
    with the vector and without it. Released for the last step or the last two, the picture stayed about as it is held.
  - Weight lowered to 0.30, the source's long hair and black top came back, with the vector and without it; at 0 the
    picture was a third variant. Core mixes the rows with noise at this weight on every step and labels them with
    the larger of the weight and the stream's time (`comfy/ldm/minimax/model.py`, `_cond_video_rows` and `seg_t` in
    `_forward`), so on the later steps of a low weight the rows are noisier than their label says. Read, not run.
    `dd054d6b` then changed the input: the node sends core the larger of the weight and the stream's time, so the
    rows' noise matches their label on every step (`control_vector_nodes.py::_stream_time`, held by a case in
    `bench/check_control_vectors.py`). No still has been rendered on that commit; these are its same-seed before.
- **On the guide edit graphs** (`h3_probe_pika_image_guide_edit*`), where the guide is the only copy of the picture:
  `ref_vae_aug` at 0.30 lost the edit and the person. A section mask is refused there, because those graphs use
  core's conditioning node and the section masks need the map our H3 conditioning nodes attach. With no reference
  there are no vision rows, so `non_vision` and `all_rows` steer the same rows (the node's log line shows every row
  steered).

**Not known.** Whether the reference form can do an edit at one frame at all: it did not in any of the tries above, and
a garment or an object as the cut-out was not among them. Whether releasing `ref_vae_end_percent` (with
a matched no-vector guide-release control) lets a style control vector restyle the whole frame while keeping the
named edit (the edit LoRA's `end_percent` was the other lever named here until 2026-11-11, when the LoRA left these
graphs). Edit guidance on a still:
`workflows/build_workflows.py::_wire_edit_guidance` and `bench/edit_guidance_arms.py::_with_second_prompt` now carry
`MiniMaxH3GuideLatent` / `MiniMaxH3AddGuide` (and `MiniMaxH3ReferenceToVideo`) onto the edited prompt's branch, not yet
graded on renders. `h3_guide_video_edit_api.json` carries a house scene prompt with no `vfx_edit:` opening, still
loads the edit LoRA, and is left as it was. Whether the one-frame guide edit holds up without the edit LoRA beyond the
two pictures above, and on the control vector twin, which was not rendered after the change.

**Retired 2026-11-10, on the owner's word:** `h3_p_h3_image_guide_edit_api.json`, the one-frame guide edit on the
`P-H3-Image` LoRA with a reference picture. It carried the same house scene prompt, and the form never made the asked
edit in the tries above. A picture edited from one or more pictures is `h3_p_h3_image_edit_api.json` on `P-H3-Image`
and `h3_pika_image_edit_api.json` on Pika.

---

> ## Historical Record (2026-08-16 to 2026-08-27): Base H3 Single-Frame Shim (`archive/workflows/image/`)
>
> **The 2026-08 single-frame base-H3 shim path below remains archived in `archive/workflows/image/` and `archive/single_frame.py`.** The notes below are kept as the historical record of that earlier path.
>
> What moved on 2026-08-27, and where:
>
> | was | now |
> |---|---|
> | `workflows/image/` (18 base-H3 graphs) | `archive/workflows/image/`, no longer generated |
> | `single_frame.py` (the core patch) | `archive/single_frame.py`, no longer imported |
> | `bench/check_single_frame.py` | `archive/bench/check_single_frame.py` |
> | `bench/bench_image_edit_refs.py` | `archive/bench/bench_image_edit_refs.py` |
>
> Everything below describes that 2026-08 base-H3 single-frame path as it stood when it was live.

Video is this repo's primary use case. This one is **experimental**: H3 renders
a single frame if you ask it for one, and at one frame it behaves like a
capable reference-driven image editor. It rests on a temporary patch, it is
moving fast upstream, and it gets its own folder so the two cases do not blur
into each other.

**The shim is OFF by default, and every graph in this folder needs it.**
Start ComfyUI with `H3_EXPLORATIONS_SINGLE_FRAME=1` or these graphs fail
validation with "Value 1 smaller than min of 5" before any node runs. Opt-in
since 2026-08-27, by owner decision: the patch is process-global, so a default
charged every install for a path most never use. Without the variable the pack
patches nothing and says nothing.

**A node cannot replace it**, which was checked rather than assumed:
`execution.py::validate_inputs` raises `value_smaller_than_min` before
execution, so a node placed in the graph is rejected along with the graph it
exists to enable.

**Everything else about the mechanism** -- the 5-frame floor, what the patch
does and refuses to do, the single-image VAE, and how the shim retires itself
-- is the module docstring of [`archive/single_frame.py`](../archive/single_frame.py), which
owns it, plus the note drawn on
[`archive/workflows/image/h3_image_edit.json`](../archive/workflows/image/h3_image_edit.json)
itself, which also carries the single-image VAE round-trip against the video
VAE and core's `t == 1` decode branch (`comfy/ldm/minimax/vae.py`). The
generator no longer draws notes, so that archived file is the only copy. (This
pointed at `workflows/image/h3_image_edit.json` until 2026-09-14; that folder
is gone.) This file is about the **prompts** and the **layout**.

---

## Layout: `workflows/image/`

Graphs are foldered by use case. Video sits at the root of `workflows/`; every
single-frame graph is written to `workflows/image/`.

**The routing is derived, not declared.** `build_workflows._graph_dir()` sends
a graph to `image/` when its `GRAPHS` entry sets `single_frame=True`, which is
exactly what makes it an image graph. A separate `image=True` flag would be a
second source of truth for one fact, and the two would eventually disagree.

**The reading side is `h3_config.GRAPH_DIRS`**, and every check that walks the
shipped graphs goes through `h3_config.graph_paths()`. This is not tidiness.
Six checks used a bare `workflows/*.json`, which is non-recursive, and on
2026-08-16 that was **demonstrated** rather than reasoned about: pointed at a
stale `GRAPH_DIRS`, `check_ref_prompt_labels` and
`check_prompt_guide_conformance` both **exited 0 while covering 20 ref graphs
instead of 28**. No error. No warning. Just a smaller number on a line nobody
has a prior for.

So `check_ref_prompt_labels` now carries a case -- *no graph directory is
invisible to discovery* -- that compares the discovered set against what is on
disk. `bench/` and `archive/` are excluded there by name, deliberately.

Naming inside the folder follows the repo's existing convention rather than a
new one: `h3_image_*.json` is a scene worth rendering, `h3_image_probe_*.json`
exists to answer a question and says which one in the note drawn on its canvas.

---

## The prompts, and what changed

**Until 2026-08-16 this path shipped one prompt in flat prose, and the
docstring argued the guide format could not apply to a still.** Two of the
guide's six sections are audio, and `detailed_description` is specified as
`[Shot 1]` with camera movement and shot timing -- none of which a one-frame
render has.

What changed is evidence. The r/StableDiffusion write-up this path follows
published a second set of prompts on 2026-08-15, and **between their two posts
the author switched formats**: post 1 is flat `Task: Reference-guided
generation. ...` prose, post 2 is the guide's structure with the audio sections
dropped. That is the direction the old docstring argued against, from someone
who had rendered a couple of thousand images this way.

**It is a practitioner's revealed preference, not a measurement.** Neither post
held the scene or the references fixed, so nothing there isolates the format.
That is a reason to test, not a reason to believe, and it is why the ladder
below exists instead of a rewrite.

The half of the old argument that survives: the audio sections describe
something a single-frame graph structurally cannot produce. It has no
`VAEDecodeAudio` at all. That is why four sections is the default and six is
the arm.

### The format ladder

Three formats, each rung removing exactly one thing, so a difference between
two arms has one candidate cause.

| format | what it is | shipped as |
|---|---|---|
| `av` | all six guide sections, audio ones present as `N/A` | `h3_image_probe_format_av.json` |
| `sections` | the four visual sections | **the default**, every `h3_image_*.json` |
| `flat` | one paragraph, no headers, no `[Shot 1]`, markers in English | `h3_image_probe_format_flat.json` |

Both probes render the **same scene as `h3_image_style.json`** at the same
seed, with the same references. Read all three together; each is worthless
alone.

**Rendered 2026-08-16, and the short version is that none of them failed.**
`av` differs from `sections` by 3.45 (grey-scale mean absolute, 0-255) against
an h264 noise floor of ~1.6; `flat` differs from both by ~44.6. So the audio
sections are free and the scaffolding is a large lever on the picture -- but
the designed failure, the style reference bringing its cottage, did not fire in
any arm. Numbers, limits and the arms that would discriminate: `#16f`.

**Content is written once per scene and rendered into all three formats.**
Hand-writing a flat variant would let the arms differ in wording as well as in
structure, and the comparison would measure the writing. Two consequences worth
knowing before reading a result:

- **`flat` keeps `<Subject N>`** even though the community's post-1 prompts do
  not. The subject labels are the only place the reference roles are stated, so
  dropping them would change what the arm says. If the structured arms win,
  whether the subject indirection specifically is what did it is a separate
  follow-up.
- **`flat` renders the retention markers as English** (`<Subject 2> supplies an
  attribute transfer:` rather than `<Subject 2>: attribute_transfer -`). A
  paragraph carrying raw marker vocabulary mid-sentence is a form nobody
  writes, and beating a strawman would tell us nothing. So that rung removes
  the guide's formal apparatus as a unit -- headers, shot marker, and marker
  vocabulary.

### What every format guarantees

These are the parts that are not stylistic, and they are checked:

- **Every `<Picture N>` the graph wires gets a job, and no prompt names one it
  does not wire.** `check_ref_prompt_labels` enforces this in every format,
  unwaived. Naming an absent reference is the failure that reads as a model
  problem: the render succeeds and quietly ignores the instruction.
- **A reference supplying technique says what it does *not* supply.** The
  official guide never writes a negative clause -- every relationship there is
  stated as what a reference provides. This comes from general prompting
  research and from the community write-ups, where the reported failure is a
  style reference dragging its own content along. **Untested here**, same as
  the identical technique in `_ref_prompt`'s swap arm.
- **Retention markers stay inside the guide's visual set.**
  `check_prompt_guide_conformance` grades image graphs on markers, task types,
  section order and dialogue placement exactly like video graphs. Only the two
  audio sections are excused, and only for graphs with no audio decoder.

---

## The scenes

Chosen so each exercises a different retention marker rather than a different
subject. Most are drawn from the two write-ups; `h3_image_swap` is not, and is
the image-path twin of `h3_ref_video_swap`. Every one names an `h3_refs/` asset
from `internal/reference_library.md`, so a result is attributable to a
documented subject rather than to whatever was in the input root that day.

| graph | refs | marker under test | what fails first |
|---|---|---|---|
| `h3_image_edit` | 1 | `partially_preserved` | the camera move, or the far side of the face |
| `h3_image_style` | 2 | `attribute_transfer` | the style reference bringing its cottage |
| `h3_image_composite` | 2 | `partially_preserved` + `fully_preserved` | the cutout: no contact shadow, studio light still on |
| `h3_image_multiperson` | 3 | two `partially_preserved` | the two faces blending into one person |
| `h3_image_recolor` | 1 | `partially_preserved`, strict | everything that was *not* asked to change |
| `h3_image_sheet` | 1 | `fully_preserved` | the rear view, which has no source pixels |
| `h3_image_swap` | 3 | two `attribute_transfer` + `fully_preserved` | a face landing on the wrong one of the two people |

**Two rules the scenes are written against**, both learned the hard way here:

- **A scene the reference already satisfies cannot fail.** The first version of
  the camera scene asked to age a subject to 60 against a reference of a man
  well past 70. It rendered, it looked like a working edit, and it demonstrated
  only that the pipeline runs. Before trusting a passing case, ask what the
  input would have to look like for it to fail.
- **One primary change per beat.** Two changes collapse into whichever is
  easier to render, so `recolor` names exactly two attributes and spends the
  rest of its words on what must not move.

### Reference files, and one UI wrinkle

The scenes name subfolder paths (`h3_refs/face_young_man_glasses_1024x1024.png`)
because `internal/reference_library.md` documents those assets and the input
root is the owner's own media.

`LoadImage` builds its combo from a **non-recursive** `os.listdir` of the input
directory (`ComfyUI/nodes.py::LoadImage.INPUT_TYPES`), so nothing in a subfolder ever
reaches `/object_info`. It also defines `VALIDATE_INPUTS` ->
`folder_paths.exists_annotated_filepath`, and ComfyUI's executor skips its own
combo check for any input the node validates itself. **So these render
correctly**, and both of this repo's validators were rejecting them -- being
stricter than the server, which is the same class of defect as being looser.
Both now exempt values carrying a subfolder, and only those; a typo in a bare
filename still fails.

**The cost, which is real and is not a bug:** the frontend populates that
dropdown from the same list, so opening one of these graphs shows the subfolder
value in the widget but will not offer it in the menu. Re-picking it from the
dropdown is the thing you cannot do.

---

## What is not settled

- **Which format is right -- partly answered 2026-08-16, and the answer is "no
  arm failed".** All three rendered at one seed: the audio sections move the
  image by 3.45 (grey-scale mean, near the ~1.6 h264 noise floor) and cost 15
  tokens; `flat` moves it by 44.61, a materially different picture. But **no
  cottage appeared in any arm** -- the `attribute_transfer` role bound without
  the scaffolding, and identity, freckling and medium held in all three. n=1
  per arm on one scene, and `flat` keeps the negative clause, so this does not
  say format is irrelevant. `open_experiments` #16f has the numbers and the
  three arms that would actually discriminate.
- **Whether the negative clauses earn their tokens.** Same open question as on
  the video path.
- **The canvas**, `docs/open_experiments.md` #16d. Every image graph ships the
  in-family 768x1152; the write-up uses 1024x1536, 52% over H3's area cap. At
  one frame the canvas is nearly free either way, which makes this cheap and
  unresolved rather than expensive and unresolved.
- **Whether `allow_upscale=False` holds across subjects.** It is now the
  default here (see below), on two paired renders rather than one. That is
  still a small n.
- **Whether a lower step count is safe on the simple scenes.** Measured as
  unsafe on the complex one, which is enough to leave the default alone, but
  a per-scene step count was not explored.


---

## What it costs, measured 2026-08-16

Both arms rendered on this box, same seed, nothing else changed. These are
wall clock including queue and any model load, so read the ratios rather than
the absolute seconds.

### `allow_upscale` -- taken, 4.9x

`h3_image_style`, two references:

| | secs |
|---|---:|
| `allow_upscale=True` (the old default) | 89.1 |
| `allow_upscale=False` (**ships now**) | 18.1 |

The two images hold the same identity, freckle pattern, head angle, expression
and hairstyle, checked against the source reference rather than against each
other. The graphite medium transferred in both, and in neither did the style
reference bring its own cottage. This reproduces the 84s/18s ladder in #16e on
a second subject and seed, which is why the default moved here.

### `steps` -- NOT taken, and the reason is the interesting part

| scene | refs | 16 steps | 8 steps | verdict |
|---|---:|---:|---:|---|
| `h3_image_edit` | 1 | 13.0s | 4.0s | indistinguishable |
| `h3_image_style` | 2 | 18.0s | 7.0s | freckling and medium both hold |
| `h3_image_multiperson` | 3 | 25.0s | 10.0s | **8 loses the freckling and the pendant** |

At three references, 8 steps drops exactly the fine detail that scene's
`partially_preserved` entry names as the thing being retained. The saving is
~15s on the one graph where the detail is the whole point, so `steps` stays 16.

**Measured only on the one-reference portrait, 8 steps looks free everywhere.**
That is this repo's own trap -- a check whose input already satisfies the
expected outcome cannot fail -- and it is why the ladder was run on the hard
scenes before touching the default.

One paired render per scene. Consistent with the expected mechanism, not a
sweep.

### The whole set

With `allow_upscale=False`, all eight graphs render in roughly **two minutes
total**, against about eleven before.

---

## Two defects another session's tool found in these prompts

`bench/preflight_graph.py`, written in a parallel session, was pointed at
`workflows/image/` on 2026-08-16 and immediately found something every check
here had passed.

**1. Six of the eight scenes defined `<Subject N>` and then never cited it in
`detailed_description`.** Guide 5.3 asks for each label at its first real
appearance *and where its role applies*, not merely once in
`subject_definitions`. The bodies said "he", "the likeness", "the same man" --
so the binding between a reference and the action it drives was left to
inference, on the exact path whose whole point is saying what each reference
does. Fixed: every scene body now cites its labels.

**`check_ref_prompt_labels` could not see it, and that was the real gap.** Its
`subjects_resolve` case checks that every subject USED is defined -- the
dangling direction -- and passes clean on a prompt that defines two subjects
and cites neither. A new case, *every defined subject is cited where it acts*,
closes it, and was shown red by reverting one scene's body to its pre-fix
wording.

**2. A row count in the generator's own note was 58% high.** It said "~5,200
rows" for the single-reference graph from arithmetic; measured, it is 3,282.
Now a measured table rather than an estimate.

Both were caught by a second reader with independent access to the artifact,
which is the finding `CLAUDE.md` records and this is another instance of it.

### The one warning left, and why it is not being fixed

Every scene trips `detailed_description is N words; the guide asks 350-500 for
generation tasks`, at 99-146 words.

**Deliberate.** That figure is the guide's for a *video* generation task, where
the body has to carry shot timing, camera movement over time, and action in
chronological order. A still frame has none of those. The community's own
working image prompts run 150-200 words, and the first-post set is shorter
still. Padding these to 350 would mean writing filler to satisfy a checker.

**But it is NOT the same case as the audio sections, and an earlier version of
this paragraph said it was.** The distinction is load-bearing and it came from
the session that wrote `preflight_graph.py`:

| | audio sections | word budget |
|---|---|---|
| status | **structurally impossible** | **possible, and judged wrong** |
| why | no `VAEDecodeAudio`, so there is no track to describe | a still frame can carry 350 words; they would be padding |
| exemption | real, and read off the graph | would be invented -- the guide exempts editing tasks explicitly and says nothing about single frames |

So the audio pair is genuinely waived by
`check_prompt_guide_conformance._audio_sections_optional`, which reads the
graph for a decoder. The word budget is **not waived anywhere**, and should not
be: scoping it by latent-frame count would mean inventing an exemption the
guide does not contain, which is the specific thing this repo criticised the
Custom-GPT pack for.

Left as a visible warning: it is correct about the guide, and the deviation is
ours to defend. `preflight_graph.py` prints a note beside it pointing here, so
a reader sees an argued deviation rather than an unexplained red. If the format
arms ever show longer bodies help, this is the number to revisit.

---

## Two things the renders turned up that are not about speed

- **`crop` cannot be retained on a 1:1 reference.** `h3_image_style` and
  `h3_image_recolor` both say the crop is preserved, and both references are
  1024x1024 while the canvas is 768x1152. The model widens the frame -- it has
  to. The claim is unachievable as written and should either drop `crop` or the
  scenes should render on a square canvas. Still unfixed for those two, but the
  **mechanism now exists**: since 2026-08-22 a scene can override the shared
  canvas through `scene()`'s `extra`, which `h3_image_swap` uses to render its
  16:9 plate on a 16:9 canvas rather than promise a framing a portrait output
  cannot hold. The same edit is available to these two and has not been made.
- **The style scene works.** No cottage, in any arm, at any step count. The
  `attribute_transfer` role bound, which is the one thing that scene exists to
  find out, and it is the first evidence here that naming what a reference does
  *not* supply does something.
