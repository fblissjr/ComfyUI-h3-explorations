# 6. Practical Examples & Recipes (End-to-End Walkthroughs)

**Parent guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
**Related guides**: [`3. ComfyUI Nodes & Controls`](comfyui_nodes_and_controls.md), [`7. Prompting Generative LLMs for Control Vectors`](llm_prompting_guide.md), [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md)

---

## Overview: Choosing the Right Creation Path & Runtime Settings

Different creative goals require different authoring paths and ComfyUI node settings on `MiniMaxH3ApplyControlVectors` ([`control_vector_nodes.py`](../../../control_vector_nodes.py)):

| Creative Goal | Recommended Authoring Path | Recommended `MiniMaxH3ApplyControlVectors` Settings | Why |
|---|---|---|---|
| **1. Film Stock, Color Grade & Optical Texture** (`t2va` or `ref2va`) | `build --words` (`opening_framing` slot) or `.toml` bank (`film_16mm_1974`) | `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `ref_vae_end_percent = 1.00` | Steers optical texture, film grain, and color grade across `scene` tokens while keeping `<Picture 1>`'s 3D Video VAE latents clean (`ref_vae_end_percent = 1.00`). |
| **2. Changing `<Subject 1>`'s Clothing in `ref2va`** | `build --words` (`opening_framing` / `subject_behavior` slot) with explicit garment nouns in Pole A (`sweater, shirt, jacket, coat`) | `strength = 2.00–2.50`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `start_percent = 0.0`, `end_percent = 0.50`, `ref_vae_end_percent = 1.00` | Naming the garment nouns to subtract in Pole A (`sweater, shirt, jacket`) + steering the early layout window (`0.0..0.50`) at `strength = 2.0–2.5` overrides `<Subject 1>`'s outfit while `ref_vae_end_percent = 1.00` keeps frames `0..1` free of late-step noise. |
| **3. Weather, Heatwave, or Atmosphere (`t2va` & `ref2va`)** | `build --words` with coupled `soundscape:` clause | `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"` | Coupled `description + overall_soundscape` clauses steer both the optical atmosphere (heat haze / blowing snow) and the ambient audio bed while `mask = "scene_only"` leaves `<d>...</d>` spoken dialogue untouched. |
| **4. Bespoke Multi-Scene Physical / World Transform** | LLM 10-Pair Markdown (`generate-pairs` or external LLM with [`llm_prompting_guide.md`](llm_prompting_guide.md)) | `strength = 1.00–1.25`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `start_percent = 0.00` | Physical/world transforms need scene-specific nouns in each of the 10 pairs and must start at `start_percent = 0.00` because H3 locks 3D layout on steps `0..1`. |
| **5. Visual Look Transfer from Two Stills (`Image A -> Image B`)** | `make-media` (or Studio **Screen 4**) | `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"` | Extracts the post-refiner difference between a baseline still (`Image A`) and a color-graded/stylized still (`Image B`) on an identical 32-px M-RoPE grid. |
| **6. Stacking Multiple Vectors + Holding a Nuisance Axis** | `sculpt` (`< 5 ms` CPU) or chained `MiniMaxH3ApplyControlVectors` nodes | Node 1 (`add_norm_matched`, `1.0`) $\rightarrow$ Node 2 (`add_norm_matched`, `0.8`) $\rightarrow$ Hold (`hold_words` or `mode = "hold"`) | Summing in `float64` with an orthogonal `hold` removes unwanted collateral drift (e.g., preventing `"blizzard"` from putting winter parkas on indoor actors). |

---

## Example 1: Changing `<Subject 1>`'s Clothing in `ref2va`

Suppose `<Picture 1>` shows `<Subject 1>` wearing indoor or winter clothing (e.g., a black knit sweater), and you want a control vector to dress `<Subject 1>` in summer resort / beachwear.

### Step 1 — Build the Wardrobe Control Vector (`~12s`)
Name the concrete garment nouns you want to subtract in **Pole A** (`sweater, shirt, jacket, coat, trousers`) rather than vague setting words like `"indoor setting"` (which cause $B - A$ to subtract `"indoors"` instead of clothing articles), and put the target attire in **Pole B**:

```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"

"$PY" bench/control_vectors.py build \
  --words "opening_framing: wearing a heavy knit sweater, buttoned shirt, and long jacket -> dressed in sun-drenched summer beachwear and tropical resort swimwear" \
  --name summer_beachwear \
  --families ref2va \
  --overwrite
```

### Step 2 — Apply in ComfyUI (`ref2va` Workflow)
In [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) (node `115`, `MiniMaxH3ApplyControlVectors`):
- **`vector_name`**: `summer_beachwear__ref2va__diff_span__lora_865ee189.safetensors`
- **`strength`**: `2.00` to `2.50`
- **`mode`**: `add_norm_matched` *(Default)*
- **`mask`**: `scene_only` *(Default — protects `<Picture 1>:` labels, `(S1)` speaker tags, and `<d>[English] ...</d>` dialogue)*
- **`start_percent` / `end_percent`**: `0.00` to `0.50`
- **`ref_vae_end_percent`**: **`1.00`** *(Default — keeps `ref_vae_aug = 0.999` across all steps so frames `0` and `1` never inherit late-step Gaussian noise)*

> [!IMPORTANT]
> **Two keys for `ref2va` wardrobe overrides**:
> 1. **Subtract concrete garment nouns in Pole A (`sweater, shirt, jacket, coat`) and omit the original garment noun from `subject_definitions:` / `retention_analysis:` in the prompt**: If `subject_definitions:` explicitly says `"wearing a black chunky ribbed knit sweater"`, those unsteered tokens re-assert the sweater. Describing `<Subject 1>`'s face and hair in `subject_definitions:` (without naming the sweater) lets the control vector replace `<Subject 1>`'s outfit cleanly on `start_percent = 0.0, end_percent = 0.50` at `strength = 2.0–2.5`.
> 2. **Keep `ref_vae_end_percent = 1.00` (`ref_vae_aug = 0.999`)**: In `comfy/ldm/minimax/model.py:538-547, 679-683`, `seg_t["ref_img"] = max(t_v, vis_aug)` is always $\ge t_v$ (`0.95` on the final step). Lowering `ref_vae_end_percent < 1.0` sets `vis_aug = 0.0` on late steps, which mixes unit Gaussian noise into `ref_img` while `seg_t["ref_img"] = 0.95` marks it clean, stamping a rectangle of static noise onto frames `0` and `1`.

---

## Example 2: Applying 1974 16mm Celluloid Film Stock to `t2va` and `ref2va` (Canonical Default)

The canonical workflows [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_text_to_video_pika_control_vector_api.json) and [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) come pre-configured with `film_16mm_1974`:

### How It Was Built
```bash
"$PY" bench/control_vectors.py build \
  prompt_bank/control_vectors/film_16mm_1974.toml \
  --families both \
  --pooling diff_span \
  --overwrite
```

### Runtime Settings in ComfyUI
- **`t2va` (`workflows/h3_text_to_video_pika_control_vector_api.json`)**:
  - `vector_name = "film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors"`
  - `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`
- **`ref2va` (`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`)**:
  - `vector_name = "film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors"`
  - `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `ref_vae_end_percent = 1.00`

---

## Example 3: Coupled Sight + Sound Atmosphere (`scorching_heatwave`)

When a concept has both a visual signature and an acoustic signature, include a `; soundscape: <Pole A> -> <Pole B>` clause so `h3_pair_compiler.py` edits both `integrated_multimodal_description:` and `overall_soundscape:` in each of the 10 scenes:

```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: bathed in cool overcast daylight with crisp neutral air -> drenched in scorching desert heatwave glare with shimmering heat-haze mirages; soundscape: quiet mild afternoon air -> dry desert wind and high-pitched cicada heat drone" \
  --name scorching_heatwave \
  --families both \
  --save-toml \
  --overwrite
```

- **Want to steer ONLY the visuals while keeping your original audio track?**
  Set `video_scale = 1.0, audio_scale = 0.0` on `MiniMaxH3ApplyControlVectors`.
- **Want to steer ONLY the soundscape while keeping baseline visuals?**
  Set `video_scale = 0.0, audio_scale = 1.0` (or `mask = "soundscape_only"`).

---

## Example 4: Building a Look Vector from Two Reference Images (`make-media`)

When you have two reference frames—for example, a flat ungraded frame (`flat_daylight.png`) and a stylized target frame (`cyberpunk_neon.png`)—you can extract the visual transformation directly without writing 10 text pairs:

```bash
"$PY" bench/control_vectors.py make-media \
  --image-a input/flat_daylight.png \
  --image-b input/cyberpunk_neon.png \
  --output-name neon_grade_from_stills \
  --families both \
  --pooling scene_text \
  --short-edge 512 \
  --overwrite
```

Or in **Studio Screen 4 (`Reference Image Contrast`)**: drag-and-drop **Image A** and **Image B** into the browser dropzones and click **Extract Multimodal Image Control Vector (GPU)**.

---

## Example 5: Combining Vectors & Scrubbing Nuisance Drift (`sculpt` in `< 5 ms`)

Suppose you want to combine `neon` lighting (`+1.0`) with `kinetic_energy` camera motion (`+0.8`), while ensuring the vector never adds rain/wet pavement (`hold_words = "rain, wet, puddle, umbrella"`):

```bash
"$PY" bench/control_vectors.py sculpt neon kinetic_energy \
  --modes add_norm_matched,add_norm_matched \
  --strengths 1.0,0.8 \
  --hold-words "rain, wet, puddle, umbrella" \
  --hold-alpha 1.0 \
  --name neon_kinetic_dry \
  --families both \
  --overwrite
```

This runs on CPU in `< 5 ms` (`0 GB` GPU VRAM) and writes both `neon_kinetic_dry__fl2va__*.safetensors` and `neon_kinetic_dry__ref2va__*.safetensors`.

---

## Example 6: Instant `< 2 ms` CPU Word Prototyping (`words` / Inline Node `words`)

When you want to test a lexical concept in `< 2 ms` on CPU before building a 10-pair Qwen3-VL vector:

1. **Directly inside ComfyUI (`MiniMaxH3ApplyControlVectors`)**:
   - Leave `vector_name = "none"`, and type into the node's widgets:
     - `words`: `bioluminescent, glowing, cyan, spores, cavern`
     - `against_words`: `daylight, fluorescent, office, dry`
     - `hold_words`: `text, watermark, subtitle`
     - `strength`: `1.15`, `mode`: `add_norm_matched`, `mask`: `scene_only`
2. **Or save a `.safetensors` file from the CLI (`< 2 ms` CPU)**:
   ```bash
   "$PY" bench/control_vectors.py words \
     "bioluminescent, glowing, cyan, spores, cavern" \
     --against-words "daylight, fluorescent, office, dry" \
     --hold-words "text, watermark, subtitle" \
     --name biolum_quick \
     --overwrite
   ```
