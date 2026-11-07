# 3. ComfyUI Nodes & Runtime Controls

**Parent guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
**Source files**: [`control_vector_nodes.py`](../../../control_vector_nodes.py), [`control_vectors.py`](../../../control_vectors.py), [`h3_spans.py`](../../../h3_spans.py)

---

## 1. The Three Control Vector Nodes in ComfyUI

All three nodes register under the **`model/conditioning/minimax`** category in ComfyUI (`control_vector_nodes.py`):

| Node Class | Purpose | Outputs & Typical Placement |
|---|---|---|
| **`MiniMaxH3ApplyControlVectors`** | Applies a `.safetensors` control vector (or inline J-Lens `words`) to the refined text conditioning via a `DIFFUSION_MODEL` wrapper on `MODEL` (`ModelPatcher`). Stack multiple vectors by chaining `MiniMaxH3ApplyControlVectors` nodes in series (shifts sum in `float64`). | Outputs `(model: MODEL)`. Place between the DiT/LoRA loader (`MiniMaxH3LoraLoader`) and the sampler (`KSampler` / `SamplerCustom`). |
| **`MiniMaxH3MakeControlVector`** | Measures and saves `<name>__<family>__*.safetensors` under `models/control_vectors/` from a `pair_file` (`prompt_bank/control_vectors/*.toml`) or inline `words` (`"pole_a -> pole_b"`, compiled across 10 canonical scenes via `h3_pair_compiler.py`). | Outputs `(vector_name: STRING, report: STRING)`. Optional in-graph authoring node (`is_output_node=True`). |
| **`MiniMaxH3MakeMediaControlVector`** | Measures and saves `<output_name>__<family>__*.safetensors` (`B - A`) under `models/control_vectors/` from contrasting `IMAGE` batches (`image_a` baseline vs. `image_b` positive/steered) resized to an identical 32-px-snapped M-RoPE grid. | Outputs `(vector_name: STRING, report: STRING)`. Connect two `LoadImage` nodes (`image_a`, `image_b`) plus `model` and `clip`. |

---

## 2. `MiniMaxH3ApplyControlVectors` Inputs Explained (`control_vector_nodes.py::MiniMaxH3ApplyControlVectors`)

- **`model`** (`MODEL`): The MiniMax H3 `ModelPatcher` (`fl2va` or `ref2va`). Chain multiple `MiniMaxH3ApplyControlVectors` nodes in series to combine multiple vectors (`add`, `add_norm_matched`, `hold`, `project_out`, `amplify`).
- **`vector_name`** (`COMBO`): Select any `.safetensors` file under `models/control_vectors/` (or `"none"` when using inline `words`).
- **`strength`** (`-20.0` to `+20.0`, default `1.0`, step `0.05`):
  - `+1.0` applies one mean pair difference (`Prompt A -> Prompt B`) for `mode = "add"`, or `67.5` post-refiner shift norm (`NORM_MATCHED_TARGET`) for `mode = "add_norm_matched"`.
  - `-1.0` **inverts** the vector (`Prompt B -> Prompt A`, e.g., flipping `blizzard` toward warm/dry air).
  - `0.0` is an exact no-op.
- **`mode`** (`COMBO`, `control_vectors.py::APPLY_MODES`):
  - **`add_norm_matched`** *(Default)*: Normalizes `v` to `67.5` post-refiner shift norm at `strength = 1.0` across all pooling modes (`diff_span`, `scene_text`, etc.).
  - **`add`**: Adds `strength * v` in raw mean-pair-difference units.
  - **`hold`**: Treats this vector as a protected direction $\hat{\mathbf{u}}$ and projects out $\text{strength} \cdot (\Delta\mathbf{r} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}$ from the accumulated additive steering shift without modifying the base prompt.
  - **`project_out`**: Removes $\text{strength} \cdot \frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v}$ directly from the prompt token states.
  - **`amplify`**: Adds $\text{strength} \cdot \frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v}$ to scale existing alignment in the prompt.
- **`mask`** (`COMBO`, `control_vector_nodes.py::MASK_MODES`):
  - **`scene_only`** *(Default)*: Steers scene text while leaving tag-0 `<Picture N>` vision rows, `<d>...</d>` dialogue, `(S1)` speaker tags, `<Picture 1>:` labels, and `<0.0 seconds>` timestamps bitwise untouched.
  - **`non_vision`**: Steers all tag-1 text rows (excluding row 0 sink) while leaving tag-0 `<Picture N>` vision rows bitwise untouched.
  - **`description_only`**: Steers only scene tokens inside `integrated_multimodal_description:` (`fl2va`) or `detailed_description:` (`ref2va`).
  - **`soundscape_only`**: Steers only scene tokens inside `overall_soundscape:`.
  - **`after_dialogue`**: Steers only scene tokens after the last `</d>`.
  - **`all_rows`**: Unmasked ablation across all rows.

---

## 3. Schedule Window, Stream Split & Inline Word Inputs

- **`start_percent` / `end_percent`** (`0.0` to `1.0`, default `0.0` to `1.0`):
  - Active schedule window (`step_idx / N_steps`). Setting `start_percent = 0.33, end_percent = 1.0` on a 6-step render lets steps 0–1 lock camera geometry and pose unsteered before injecting the style/weather vector on steps 2–5.
- **`allow_recipe_mismatch`** (`BOOLEAN`, default `False`):
  - Refuses a vector whose checkpoint family (`fl2va` vs. `ref2va`) or LoRA recipe differs from the loaded `model` unless set to `True`.
- **`video_scale` / `audio_scale` / `split_end_percent`** (defaults `1.0`, `1.0`, `1.0`):
  - Stream multipliers applied within `[0.0, split_end_percent)`. At `(1.0, 1.0)` runs a single DiT pass; at `(1.0, 0.0)` steers video while keeping audio unsteered; at `(0.0, 1.0)` steers audio while keeping video unsteered.
- **`ref_vae_end_percent` / `ref_vae_aug`** (defaults `1.0`, `0.999`):
  - For `ref2va` renders: controls the fraction of steps `[0.0, ref_vae_end_percent)` where clean 3D Video VAE reference latents stay active at weight `ref_vae_aug`.
  - **`ref_vae_end_percent = 1.00`** *(Default)*: Keeps `cond_video_latents` pinned at `ref_vae_aug = 0.999` across all steps so frames `0` and `1` never inherit late-step noise from `seg_t["ref_img"] = max(t_v, vis_aug)` (`comfy/ldm/minimax/model.py:538-547, 679-683`).
- **`words` / `against_words` / `hold_words` / `hold_alpha`**:
  - Optional inline J-Lens (`< 2 ms` CPU) unigram word inputs and soft-hold word scrubbing (`hold_alpha` in `[0.0, 2.0]`, default `1.0`).

---

## 4. `MiniMaxH3MakeControlVector` & `MiniMaxH3MakeMediaControlVector` Inputs

- **`MiniMaxH3MakeControlVector`** (`control_vector_nodes.py:1795`):
  - Inputs: `model`, `clip`, `pair_file` (default `"film_16mm_1974.toml"`), `pooling` (`"diff_span"`, `"from_first_diff"`, `"scene_text"`, `"all_tag1"`), `apply_holds` (`True`), `output_name` (`""`), optional `encoded_references`, `overwrite` (`False`), `words` (`""`), and `slot` (`"auto"`, `"opening_framing"`, `"subject_behavior"`, `"interpersonal_dynamics"`, `"camera_motion"`, `"speech_delivery"`, `"soundscape"`).
  - Outputs: `(vector_name: STRING, report: STRING)`.
- **`MiniMaxH3MakeMediaControlVector`** (`control_vector_nodes.py:2308`):
  - Inputs: `model`, `clip`, `image_a` (`IMAGE`), `image_b` (`IMAGE`), `output_name` (`"media_contrast"`), `pooling` (`"scene_text"`, `"all_tag1"`, `"vision_tokens"`, `"all_rows"`), `short_edge` (`512`, snapped to `32`), `scene_prompt` (`""`), `overwrite` (`False`).
  - Outputs: `(vector_name: STRING, report: STRING)`.
