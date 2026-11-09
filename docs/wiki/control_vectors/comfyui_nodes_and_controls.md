# 3. ComfyUI Nodes & Runtime Controls

[← Back to Control Vectors Overview](../control_vectors.md)

This guide covers the three ComfyUI nodes in [`control_vector_nodes.py`](../../../control_vector_nodes.py) (`MiniMaxH3ApplyControlVectors`, `MiniMaxH3MakeControlVector`, and `MiniMaxH3MakeMediaControlVector`) and how each runtime knob interacts with MiniMax H3 video (`t2va`, `fl2va`, `ref2va`) and single-frame `P-H3-Image` (`t2i`, `edit`, `guide_edit`) generation.

---

## 1. Node Summary & Placement in the Graph

| Node (`node_id` / Display Name) | Inputs $\rightarrow$ Outputs | Role in Workflow |
|---|---|---|
| **`MiniMaxH3ApplyControlVectors`**<br>*MiniMax H3 Apply Control Vectors* | `model: MODEL`, `vector_file` (or inline `words`) $\rightarrow$ `MODEL` | **Runtime Steering Node**: Sits on the `MODEL` wire immediately before `BasicGuider` / `SamplerCustomAdvanced`. Multiple `MiniMaxH3ApplyControlVectors` nodes can be chained in series to stack or orthogonalize (`mode = "hold"`) multiple vectors in `float64`. |
| **`MiniMaxH3MakeControlVector`**<br>*MiniMax H3 Make Control Vector (Pairs)* | `model: MODEL`, `clip: CLIP`, `pair_file` or `target_prompt` + `words` (+ optional `encoded_references: H3_REF_ENCODED`) $\rightarrow$ `vector_name: STRING, report: STRING` | **In-Graph Pair Builder**: Builds and saves a `.safetensors` control vector inside ComfyUI from either a `.toml` bank (`pair_file`) or an in-place contrast on `target_prompt` (`words`), with optional `encoded_references` vision conditioning. |
| **`MiniMaxH3MakeMediaControlVector`**<br>*MiniMax H3 Make Media Control Vector (Images/Video)* | `model: MODEL`, `clip: CLIP`, `image_a: IMAGE`, `image_b: IMAGE` $\rightarrow$ `vector_name: STRING, report: STRING` | **In-Graph Image Look Builder**: Extracts and saves a visual look-transfer vector (`Image A -> Image B`) from two reference images inside ComfyUI. |

---

## 2. `MiniMaxH3ApplyControlVectors` Inputs Explained (`control_vector_nodes.py::MiniMaxH3ApplyControlVectors`)

![Runtime Token Masks (scene_only), Step Windows & ref2va 3D VAE Gate](assets/runtime_steering_and_masks.svg)

| Input | Default | Options / Range | What It Does |
|---|---|---|---|
| **`vector_file`** | `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` | Any `.safetensors` in `models/control_vectors/` or `"none"` | Selects the saved control vector file. [`resolve_vector_file_path`](../../../control_vector_nodes.py) automatically resolves the matching `__fl2va__` or `__ref2va__` and LoRA/base sibling for the connected `model: MODEL`. |
| **`strength`** | `1.15` (workflows) / `1.0` (schema) | `-10.0 .. 10.0` | Multiplier on the control vector. Under `mode = "add_norm_matched"`, `strength = 1.0` equals a post-refiner shift norm of `67.5`. Negative values (`-1.0`) invert the concept direction. |
| **`mode`** | **`add_norm_matched`** | `add_norm_matched`, `add`, `hold`, `project_out`, `amplify` | • **`add_norm_matched` (Default)**: Normalizes `v` to `NORM_MATCHED_TARGET = 67.5` before multiplying by `strength`.<br>• **`add`**: Uses the vector's raw unnormalized norm (`raw_norm`).<br>• **`hold`**: Gram-Schmidt orthogonal projection (`v_clean = v - (v · u) u`) that scrubs this vector's direction out of the conditioning tokens and out of any earlier vectors in the chain.<br>• **`project_out`**: Removes each token's component along $\hat{\mathbf{v}}$.<br>• **`amplify`**: Scales each token's existing projection along $\hat{\mathbf{v}}$ by `1 + strength`. |
| **`mask`** | **`scene_only`** | `scene_only`, `non_vision`, `description_only`, `soundscape_only`, `after_dialogue`, `all_rows` | Controls which token positions receive the vector ([`h3_spans.py`](../../../h3_spans.py)):<br>• **`scene_only` (Default)**: Steers scene prose while leaving `tag == 0` `<Picture N>` vision patches, `<d>[Language] ...</d>` spoken dialogue, `(S1)` speaker tags, `<Picture 1>:` labels, and `<0.0 seconds>` timestamps bitwise untouched.<br>• **`description_only`**: Steers only `integrated_multimodal_description:` / `detailed_description:` (visuals only).<br>• **`soundscape_only`**: Steers only `overall_soundscape:` (ambient audio/foley only).<br>• **`after_dialogue`**: Steers only scene tokens after the final `</d>` dialogue closing tag. |
| **`start_percent` / `end_percent`** | `0.00` / `1.00` | `0.00 .. 1.00` | Fraction of the denoising step schedule where steering is active:<br>• **`0.00 .. 1.00`**: Full schedule (required for genre, physics, camera rig motion, and layout changes).<br>• **`0.33 .. 1.00`**: Pose-locked optical grade (lets steps `0..1` lock exact baseline framing and pose, then applies film stock/lighting across steps `2..7`).<br>• **`0.00 .. 0.50`**: Early layout window (steers blocking/wardrobe in early steps and lets fine detail settle unsteered). |
| **`video_scale` / `audio_scale` / `split_end_percent`** | `1.0` / `1.0` / `1.0` | `-5.0 .. 5.0` | Independent multipliers for the video stream vs. audio stream. When `video_scale != audio_scale`, the wrapper runs a dual-conditioning pass during `0.0 .. split_end_percent` and splices `[out_vid[0], out_aud[1]]`. Set `audio_scale = 0.0` for visual-only steering or `video_scale = 0.0` for soundscape-only steering. |
| **`ref_vae_end_percent` / `ref_vae_aug`** | **`1.00`** / **`0.999`** | `0.00 .. 1.00` / `0.000 .. 1.000` | For `ref2va` and `P-H3-Image` Edit: keeps the 3D Video VAE / `PH3ImageVAE` reference latents (`cond_video_latents` / `minimax_refs`) pinned at `ref_vae_aug = 0.999` through `ref_vae_end_percent = 1.00` so frames `0` and `1` never inherit late-step noise. |
| **`words` / `against_words` / `hold_words` / `hold_alpha`** | `""` / `"@neutral"` / `""` / `1.0` | Comma-separated words | **Inline `< 2 ms` CPU J-Lens controls**:<br>• If **`words`** is non-empty (and `vector_file = "none"`), synthesizes a Method 3 (`from-words`) control vector on the fly in `< 2 ms` on CPU.<br>• If **`hold_words`** is non-empty alongside a loaded `vector_file`, projects those nuisance words (`"coat, parka"`) out of the loaded vector on CPU before applying it. |
| **`allow_recipe_mismatch`** | `False` | `False` / `True` | Verifies that the vector's `model_family` (`fl2va` vs `ref2va`) and `token_refiner` weights are compatible with the connected DiT (`fl2va` vectors automatically work with `P-H3-Image` T2I and 1–8 ref Image Edit; `ref2va` vectors automatically work with `ref2va` and `P-H3-Image` Guide Edit). |

---

## 3. Using `MiniMaxH3MakeControlVector` Inside ComfyUI

`MiniMaxH3MakeControlVector` supports two authoring modes directly inside a ComfyUI graph:
1. **From a `.toml` Bank (`pair_file != "none"`)**:
   Select any `.toml` file from `prompt_bank/control_vectors/` in the `pair_file` dropdown.
2. **In-Place Scene-Paired Contrast (`pair_file = "none"`, `target_prompt` + `words`)**:
   Paste your scene prompt into `target_prompt` and write your in-place `pole_a -> pole_b` replacement(s) in `words` (e.g., `sunlit diner -> neon-lit cyberpunk diner; soundscape: faint murmurs -> rain and synth hum`). If your scene is `ref2va` or `i2va`, connect `encoded_references: H3_REF_ENCODED` so extraction conditions on the reference image's vision tokens.
