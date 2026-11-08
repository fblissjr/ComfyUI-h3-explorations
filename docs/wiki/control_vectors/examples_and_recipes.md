# 6. Practical Examples & Recipes (By Control Vector Type)

**Parent guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
**Related guides**: [`2. Building Pairs & Vectors`](building_pairs_and_vectors.md), [`3. ComfyUI Nodes & Controls`](comfyui_nodes_and_controls.md), [`7. Prompting Generative LLMs for Control Vectors`](llm_prompting_guide.md), [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md)

---

## Overview: Choosing the Right Vector Type, Authoring Path & Runtime Settings

Every control vector type acts on a specific combination of **token spans** (`mask`), **denoising step windows** (`start_percent..end_percent`), and **audio/video streams** (`video_scale / audio_scale`) in `MiniMaxH3ApplyControlVectors` ([`control_vector_nodes.py`](../../../control_vector_nodes.py)):

| Vector Type | Recommended Authoring Path | Recommended `MiniMaxH3ApplyControlVectors` Settings | Why It Works |
|---|---|---|---|
| **[Type 1: Film Stock, Optical Lenses & Color Grade](#type-1-film-stock-optical-lenses--color-grade)** | `build --words` (`opening_framing` slot) or `.toml` bank (`film_16mm_1974`, `vhs_1991_camcorder`, `noir`) | `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `0.00–1.00` *(or `0.33–1.00` to lock exact baseline pose)* | Steers emulsion grain, halation, and color grade across `scene` tokens while keeping `<Picture 1>`'s 3D Video VAE latents clean (`ref_vae_end_percent = 1.00`). |
| **[Type 2: Coupled Sight + Sound Atmosphere & Weather](#type-2-coupled-sight--sound-atmosphere--weather)** | `build --words` with coupled `opening_framing:` + `; soundscape:` clauses | `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `video_scale = 1.0, audio_scale = 1.0` | Coupled `description + overall_soundscape` clauses steer both optical weather (heat haze, snow, bioluminescence) and ambient audio while `mask = "scene_only"` leaves `<d>...</d>` spoken dialogue untouched. |
| **[Type 3: Physics, World Rules & Scene-Adaptive `{props}` Transforms](#type-3-physics-world-rules--scene-adaptive-props-transforms)** | `build --words` (`subject_behavior:` with `{props}` / `{surface}`) or 10 LLM pairs | `strength = 1.10–1.25`, `mode = "add_norm_matched"`, `start_percent = 0.00`, `mask = "scene_only"` *(or `mask = "after_dialogue"`)* | Physical/world rules must be active at `start_percent = 0.00` because H3 commits 3D object trajectories on steps `0..1`. |
| **[Type 4: Camera Rig, Motion & Framing Energy](#type-4-camera-rig-motion--framing-energy)** | `build --words` (`camera_motion:` slot) or 10 LLM pairs | `strength = 1.00–1.20`, `mode = "add_norm_matched"`, `start_percent = 0.00`, `mask = "scene_only"`, `audio_scale = 0.0` | Isolates camera rig dynamics (handheld shake, whip-pans, floating drift) at `start_percent = 0.00` without altering lighting or audio. |
| **[Type 5: Vocal Delivery & Acoustic Space](#type-5-vocal-delivery--acoustic-space)** | `build --words` (`speech_delivery:` or `soundscape:` slot) | `mask = "scene_only"` for `(S1)` vocal delivery; `mask = "soundscape_only"` or `video_scale = 0.0, audio_scale = 1.0` for audio-only | Steers the delivery clause framing `<d>...</d>` (whispered, panicked) without corrupting the spoken transcript, or steers room acoustics alone. |
| **[Type 6: Visual Look Transfer from Two Reference Images (`make-media`)](#type-6-visual-look-transfer-from-two-reference-images-make-media)** | `make-media` CLI or Studio **Screen 4** (`Image A -> Image B`) | `strength = 1.10–1.25`, `mode = "add_norm_matched"`, `mask = "scene_only"` | Extracts the post-refiner difference between a baseline still (`Image A`) and a color-graded/stylized still (`Image B`) on an identical 32-px M-RoPE grid. |
| **[Type 7: Instant `< 5 ms` CPU Sculpting: Flipping (`--invert`), Blending & Orthogonal `hold`](#type-7-instant--5-ms-cpu-sculpting-flipping---invert-blending--orthogonal-hold)** | `sculpt` (`< 5 ms` CPU) or chained `MiniMaxH3ApplyControlVectors` nodes | Outputs a single pre-cleaned `.safetensors` ready for `strength = 1.15, mode = "add_norm_matched"` | Flips (`-1.0x`), blends on the `67.5` norm circle, and projects out nuisance axes (`--hold` / `--hold-words`) in `float64` on CPU. |
| **[Type 8: Subject Material, Era Prop, or Attire Overrides (`ref2va` & `t2va`)](#type-8-subject-material-era-prop-or-attire-overrides-ref2va--t2va)** | `build --words` (`opening_framing` / `subject_behavior`) with concrete baseline nouns in Pole A | `strength = 1.25–2.50`, `mode = "add_norm_matched"`, `start_percent = 0.00, end_percent = 0.50`, `ref_vae_end_percent = 1.00` | Naming the baseline props/garments in Pole A + steering the early layout window (`0.00..0.50`) overrides materials, era props, or `<Subject 1>` attire while keeping frames `0..1` clean. |

---

## Type 1: Film Stock, Optical Lenses & Color Grade

The canonical workflows [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_text_to_video_pika_control_vector_api.json), [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json), [`workflows/image/h3_p_h3_image_t2i_control_vector_api.json`](../../../workflows/image/h3_p_h3_image_t2i_control_vector_api.json), and [`workflows/image/h3_p_h3_image_edit_control_vector_api.json`](../../../workflows/image/h3_p_h3_image_edit_control_vector_api.json) ship pre-configured with `film_16mm_1974`. You can also build your own optical look (such as Kodak Aerochrome infrared, bleach-bypass silver retention, or 1993 VHS camcorder tape) in ~12 seconds.

### Option A — Build from a Shipped `.toml` Bank (`film_16mm_1974`, `vhs_1991_camcorder`, `noir`)
```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"

"$PY" bench/control_vectors.py build \
  prompt_bank/control_vectors/film_16mm_1974.toml \
  --families both \
  --pooling diff_span \
  --overwrite
```

### Option B — Build a Custom Optical / Film Look from `--words` (`opening_framing` Slot)
```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: captured on a clean modern digital cinema sensor with neutral daylight color balance -> shot on Kodak Aerochrome false-color infrared film stock with crimson-magenta foliage, deep inky skies, and glowing halation" \
  --name aerochrome_infrared \
  --families both \
  --save-toml \
  --overwrite
```

### How to Use It in ComfyUI (`MiniMaxH3ApplyControlVectors`)
- **Full-schedule optical + lighting steer (`start_percent = 0.00, end_percent = 1.00`)**:
  - `vector_name`: `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` (for `t2va`, `fl2va`, and `P-H3-Image` T2I / Edit) or `film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors` (for `ref2va` video and `P-H3-Image` Guide Edit; [`resolve_vector_file_path`](../../../control_vector_nodes.py#L1535-L1605) auto-swaps `__fl2va__` $\leftrightarrow$ `__ref2va__` and LoRA/base siblings to match the loaded model)
  - `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `ref_vae_end_percent = 1.00`
- **Pose-locked optical grade (`start_percent = 0.33, end_percent = 1.00`)**:
  - On an 8-step Pika workflow, setting `start_percent = 0.33` lets steps `0` and `1` run unsteered—locking the exact actor pose, camera framing, and blocking of your unsteered baseline seed—while steps `2..7` apply the 16mm grain, halation, and color grade on top.

---

## Type 2: Coupled Sight + Sound Atmosphere & Weather

Because MiniMax H3 generates video and audio jointly inside a single 50-block DiT, a control vector can shift both the **visual atmosphere** (`integrated_multimodal_description:`) and the **acoustic bed** (`overall_soundscape:`) at the same time—while `mask = "scene_only"` leaves `<d>[Language] ...</d>` spoken dialogue untouched.

### Step 1 — Build a Coupled Visual + Soundscape Vector (`~12s`)
Separate the visual clause and the audio clause with `; soundscape:`:
```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: bathed in cool overcast daylight with crisp neutral air -> drenched in scorching desert heatwave glare with shimmering heat-haze mirages; soundscape: quiet mild afternoon air -> dry desert wind and high-pitched cicada heat drone" \
  --name scorching_heatwave \
  --families both \
  --save-toml \
  --overwrite
```

Another example—turning any room or exterior into a **bioluminescent subterranean cavern**:
```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: lit by ordinary neutral daylight with clear air -> illuminated by glowing cyan-and-violet bioluminescent cavern flora, drifting luminous spores, and caustic water ripples; soundscape: dry everyday room tone -> resonant subterranean cavern water drips and low crystalline hum" \
  --name bioluminescent_cavern \
  --families both \
  --save-toml \
  --overwrite
```

### Step 2 — Choose How Video and Audio Streams Are Steered in ComfyUI
On `MiniMaxH3ApplyControlVectors`:
1. **Steer BOTH sight and sound together *(Default)***:
   - `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `video_scale = 1.0`, `audio_scale = 1.0`
2. **Steer ONLY the visuals while keeping your original audio bed**:
   - Set `video_scale = 1.0, audio_scale = 0.0`.
3. **Steer ONLY the soundscape while keeping baseline visuals**:
   - Set `video_scale = 0.0, audio_scale = 1.0` (or set `mask = "soundscape_only"`).

---

## Type 3: Physics, World Rules & Scene-Adaptive `{props}` Transforms

When you want to change the **physical rules of the world**—such as **cryogenic flash-freeze**, **sudden zero-gravity levitation**, or **time-freeze**—the transformation works best when it acts on the native objects inside each scene rather than inserting a foreign object.

### Option A — Using Scene-Adaptive `{props}` and `{surface}` Placeholders (`--words`)
In `h3_pair_compiler.py`, the `subject_behavior:` and `interpersonal_dynamics:` slots automatically expand `{props}` and `{surface}` to match each of the 10 canonical scenes (e.g., ceramic mugs on the kitchen counter, brass calipers on the workbench, chess pieces on the board):

```bash
# 1. Cryogenic Flash-Freeze (with coupled ice-cracking audio + hold(wardrobe) so indoor clothes don't turn into parkas):
"$PY" bench/control_vectors.py build \
  --words "subject_behavior: while {props} rest undisturbed on {surface} at normal room temperature -> while {props} across {surface} flash-freeze under creeping blue rime frost and branching ice crystals; soundscape: quiet ambient room tone -> sharp crystalline ice cracking and sub-zero frost hiss; hold(wardrobe): dressed in indoor casual shirt sleeves -> bundled in a heavy insulated arctic parka and fur hood" \
  --name cryogenic_freeze \
  --families both \
  --save-toml \
  --overwrite

# 2. Sudden Zero-Gravity Levitation:
"$PY" bench/control_vectors.py build \
  --words "subject_behavior: while {props} rest firmly on {surface} under normal gravity -> while {props} lift smoothly off {surface} and drift weightlessly in mid-air under zero gravity" \
  --name zero_gravity \
  --families both \
  --save-toml \
  --overwrite
```

### Option B — How to Use `mask = "after_dialogue"` for Post-Speech Physical Triggers
- **Full-shot physics (`mask = "scene_only"`, `start_percent = 0.00`)**: Applies the zero-gravity or flash-freeze effect across the entire scene description from step `0` (`start_percent = 0.00, end_percent = 1.00, strength = 1.15`).
- **Post-speech trigger (`mask = "after_dialogue"`, `start_percent = 0.00`)**: If your prompt has `<Subject 1>` speak a short line (`<d>[English] ...</d>`) and then describes what happens in the room after `</d>`, setting `mask = "after_dialogue"` applies the vector **only** to the scene prose tokens after the closing `</d>` tag (plus `overall_soundscape:`), keeping the opening speaking beat calm before the effect triggers.

---

## Type 4: Camera Rig, Motion & Framing Energy

Use the `camera_motion:` slot (or Category C in [`7. Prompting Generative LLMs`](llm_prompting_guide.md)) when you want to change **how the camera moves**—such as frantic handheld camcorder urgency, whip-pans, or slow-motion floating drift—without changing lighting, props, or audio.

```bash
"$PY" bench/control_vectors.py build \
  --words "camera_motion: The camera holds a steady, composed frame on the action -> The camera shakes with frantic handheld camcorder urgency, abrupt whip-pans, and erratic framing jolts" \
  --name urgent_handheld \
  --families both \
  --save-toml \
  --overwrite
```

### Runtime Settings in ComfyUI
- `strength = 1.05–1.20`, `mode = "add_norm_matched"`, `mask = "scene_only"`
- **`start_percent = 0.00, end_percent = 1.00`**: Always start camera-motion vectors at `0.00` because global camera motion is established in the first 2 Pika denoising steps.
- Optional: set `audio_scale = 0.0` if you want zero change to the audio stream.

---

## Type 5: Vocal Delivery & Acoustic Space

Even though `mask = "scene_only"` never modifies the words inside `<d>[Language] ...</d>` (so the spoken words remain intact), you can steer **how `(S1)` delivers the line** or **the acoustic space around the voice**:

### Option A — Steering Vocal Delivery (`speech_delivery:` Slot)
The `speech_delivery:` slot targets the framing clause immediately before `<d>[English] ...</d>` on the 5 speaking canonical scenes:
```bash
"$PY" bench/control_vectors.py build \
  --words "speech_delivery: speaks in a calm, measured, conversational tone -> whispers in a hushed, breathless, urgent tone; soundscape: ordinary room tone -> close-mic hushed breath presence in a dead-quiet room" \
  --name urgent_whisper \
  --families both \
  --save-toml \
  --overwrite
```

### Option B — Steering Only the Acoustic Space (`soundscape:` Slot + `mask = "soundscape_only"`)
```bash
"$PY" bench/control_vectors.py build \
  --words "soundscape: dry indoor room tone with close natural acoustics -> vast vaulted stone cathedral acoustics with long echoing reverberation" \
  --name cathedral_reverb \
  --families both \
  --save-toml \
  --overwrite
```
In ComfyUI, set `mask = "soundscape_only"` (or `video_scale = 0.0, audio_scale = 1.0`) so the video stays 100% unchanged while the audio track gains the target acoustic space.

---

## Type 6: Visual Look Transfer from Two Reference Images (`make-media`)

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

## Type 7: Instant `< 5 ms` CPU Sculpting: Flipping (`--invert`), Blending & Orthogonal `hold`

You can flip, blend, or scrub existing vectors in `< 5 ms` on CPU (`0 GB` GPU VRAM) using `sculpt` (or **Studio Screen 5**), or prototype a quick lexical concept in `< 2 ms` using `words`:

### A. Flip Any Vector (`--invert`) or Combine Multiple Vectors with `hold` (`< 5 ms` CPU)
```bash
# 1. Flip blizzard (-1.0x) into a new saved vector:
"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard

# 2. Combine neon lighting (+1.0) + kinetic_energy (+0.8) while scrubbing out wet/rain pavement drift:
"$PY" bench/control_vectors.py sculpt neon kinetic_energy \
  --modes add_norm_matched,add_norm_matched \
  --strengths 1.0,0.8 \
  --hold-words "rain, wet, puddle, umbrella" \
  --hold-alpha 1.0 \
  --name neon_kinetic_dry \
  --families both \
  --overwrite
```

### B. Instant `< 2 ms` CPU Word Prototyping (`words` CLI or Inline Node Widget)
When you want to test a quick unigram lexical direction before running a 12-second Qwen3-VL 10-pair build:
- **Directly inside ComfyUI (`MiniMaxH3ApplyControlVectors`)**: leave `vector_name = "none"`, and set `words = "bioluminescent, glowing, cyan, spores, cavern"`, `against_words = "daylight, fluorescent, office, dry"`, `hold_words = "text, watermark, subtitle"`.
- **Or from the CLI**:
  ```bash
  "$PY" bench/control_vectors.py words \
    "bioluminescent, glowing, cyan, spores, cavern" \
    --against-words "daylight, fluorescent, office, dry" \
    --hold-words "text, watermark, subtitle" \
    --name biolum_quick \
    --overwrite
  ```

---

## Type 8: Subject Material, Era Prop, or Attire Overrides (`ref2va` & `t2va`)

When you want to replace **specific physical props, surface materials, or subject attire**—including overriding what `<Subject 1>` is wearing in `ref2va` while preserving `<Subject 1>`'s face and room geometry—name the concrete baseline nouns you want to subtract in **Pole A** and apply the vector on the early layout window (`start_percent = 0.00, end_percent = 0.50`).

### Example 8A — Replacing Modern Props with 1970s Analog Brass & Bakelite (`t2va` & `ref2va`)
```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: equipped with sleek modern smartphones, flatscreen glass monitors, and aluminum laptops -> outfitted with 1970s corded Bakelite rotary telephones, curved CRT television tubes, and brushed-brass analog toggle switches" \
  --name analog_1970s_props \
  --families both \
  --save-toml \
  --overwrite
```
Apply in ComfyUI at `strength = 1.25–1.50`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `start_percent = 0.00, end_percent = 0.50`.

### Example 8B — Overriding `<Subject 1>`'s Clothing in `ref2va`
Suppose `<Picture 1>` shows `<Subject 1>` wearing indoor or cool-weather clothing (e.g., a knit sweater), and you want a control vector to dress `<Subject 1>` in summer resort wear while keeping `<Subject 1>`'s identity and the room intact:

```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: wearing a heavy knit sweater, buttoned shirt, and long jacket -> dressed in sun-drenched summer beachwear and tropical resort swimwear" \
  --name summer_beachwear \
  --families ref2va \
  --overwrite
```

In [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) (node `115`, `MiniMaxH3ApplyControlVectors`):
- **`vector_name`**: `summer_beachwear__ref2va__diff_span__lora_865ee189.safetensors`
- **`strength`**: `2.00` to `2.50`
- **`mode`**: `add_norm_matched` *(Default)*
- **`mask`**: `scene_only` *(Default — protects `<Picture 1>:` labels, `(S1)` speaker tags, and `<d>[English] ...</d>` dialogue)*
- **`start_percent` / `end_percent`**: `0.00` to `0.50`
- **`ref_vae_end_percent`**: **`1.00`** *(Default — keeps `ref_vae_aug = 0.999` across all steps so frames `0` and `1` never inherit late-step Gaussian noise)*

> [!IMPORTANT]
> **Two keys for `ref2va` attire/material overrides**:
> 1. **Subtract concrete baseline nouns in Pole A (`sweater, shirt, jacket, coat`) and omit the original garment noun from `subject_definitions:` / `retention_analysis:` in the prompt**: If `subject_definitions:` explicitly says `"wearing a black chunky ribbed knit sweater"`, those unsteered tokens re-assert the sweater. Describing `<Subject 1>`'s face and hair in `subject_definitions:` (without naming the sweater) lets the control vector replace `<Subject 1>`'s outfit cleanly on `start_percent = 0.0, end_percent = 0.50` at `strength = 2.0–2.5`.
> 2. **Keep `ref_vae_end_percent = 1.00` (`ref_vae_aug = 0.999`)**: In `comfy/ldm/minimax/model.py:538-547, 679-683`, `seg_t["ref_img"] = max(t_v, vis_aug)` is always $\ge t_v$ (`0.95` on the final step). Lowering `ref_vae_end_percent < 1.0` sets `vis_aug = 0.0` on late steps, which mixes unit Gaussian noise into `ref_img` while `seg_t["ref_img"] = 0.95` marks it clean, stamping a rectangle of static noise onto frames `0` and `1`.
