# Worked Examples and Copy-Paste Recipes

Last updated: 2026-11-11.

Every recipe below uses tested settings from our 6-step Pika `t2va` and `ref2va` runs. Replace `/path/to/comfy-venv/bin/python` with your ComfyUI environment's Python interpreter (never `uv run`).

---

## Recipe 1 — Exact `<Subject 1>` clothing or prop swap on `ref2va` (`MiniMaxH3EditGuider`)

**Use when**: You have a `<Picture 1>` reference portrait and want `<Subject 1>` in a **specific new outfit** (exact color, collar, cut, fabric) while keeping her face, hair, voice, and the room around her.

- **Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance.json`
- **`MiniMaxH3PromptEdit` `edits`** (or second `MiniMaxH3ReferenceTextEncode`):
  ```text
  subject_definitions: from <Picture 1>. -> from <Picture 1>, wearing a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans.
  ```
- **`MiniMaxH3EditGuider` settings**:
  - `scale = 1.5`
  - `first_step = 0`
  - `last_step = 2` (`9` DiT passes across 6 Pika steps instead of `12`, keeping the outfit locked on steps `3..5`)
- **Why not a control vector here**: Pooling a 22-token multi-garment description into one `5376`-d vector loses which color belongs to which garment (giving a cream blouse on `s730451892`), whereas `MiniMaxH3EditGuider` keeps every token unpooled across all 50 DiT blocks (`$H3_OUTPUT_DIR/Video/edit_guidance/stacked/diner16f_edit_guidance_style_vs_subject_s730451892_stacked.mp4`).

---

## Recipe 2 — Film look / lighting grade on `ref2va` *without* touching `<Subject 1>`

**Use when**: You want 1974 16mm celluloid grain (with or without amber cast), noir contrast, or VHS texture across the room and background, while keeping `<Subject 1>`'s skin tone and `<Picture 1>` likeness clean.

- **Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_control_vector.json`
- **`MiniMaxH3ApplyControlVectors` settings**:
  - `vector_name`: `film_16mm_1974_neutral__ref2va__diff_span__lora_865ee189.safetensors` (neutral daylight grain; pick `film_16mm_1974__ref2va...` if you want warm amber tungsten cast too)
  - `strength = 1.25`
  - `mode = "add_norm_matched"`
  - `mask = "description_only"` *(or tick `description` in `fields`)*
  - `end_percent = 1.0` (if pushing `strength >= 2.0`, drop `end_percent` to `0.50` so high-strength text steering never burns red into facial shadows)
  - `ref_vae_end_percent = 1.0`, `ref_vae_aug = 0.999`

---

## Recipe 3 — Whole-scene material / world conversion on `ref2va` (`claymation_stopmotion` / `felt_muppet`)

**Use when**: You want the **entire world and `<Subject 1>`** converted into sculpted plasticine clay, felt puppetry, or exaggerated cartoon proportions—where `<Picture 1>`'s clean live-action VAE latents would otherwise pull late steps back toward photoreal skin.

- **Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_control_vector.json`
- **`MiniMaxH3ApplyControlVectors` settings**:
  - `vector_name`: `claymation_stopmotion__ref2va__diff_span__lora_865ee189.safetensors` (or `felt_muppet__ref2va...`)
  - `strength = 2.0`
  - `mode = "add_norm_matched"`
  - `mask = "scene_only"`
  - `end_percent = 0.50` *(steers steps `0, 1, 2` of 6, avoiding late-step color burn)*
  - `ref_vae_end_percent = 0.33` *(holds clean `<Picture 1>` VAE latents on steps `0, 1` to lock composition/identity, then releases visual VAE rows on steps `2..5` while keeping Qwen tag-`0` vision tokens and 3D RoPE positions intact)*

---

## Recipe 4 — Zero-file `< 2 ms` weather shift *without* wardrobe drift (`words` + `hold_words`)

**Use when**: You want an instant environmental transformation (e.g., turning a sunny summer market into a snowstorm) without building a `.safetensors` file, and **without** the DiT's world prior putting winter coats on the characters.

- **Workflow**: `workflows/h3_text_to_video_pika_control_vector.json` (`prompt_bank/04_t2va_market.txt`)
- **`MiniMaxH3ApplyControlVectors` settings** (verified in `$H3_OUTPUT_DIR/Video/cv_e2e_verify/10_t2va_jlens_words_00001-audio.mp4`):
  - `vector_name = "none"`
  - `words = "blizzard, snowdrift, freezing, snow, frost"`
  - `against_words = "warm, sunny, indoor, summer"`
  - `hold_words = "coat, parka"`
  - `hold_alpha = 0.5`
  - `strength = 1.65`
  - `mode = "add_norm_matched"`
  - `mask = "scene_only"`
- **Tip**: Stick to single-token English words (`snow, frost, blizzard, neon, clay, smoke, flame`); multi-token words (`bioluminescent`, `plasticine`, `pyrotechnics`) are dropped by J-Lens's single-token vocabulary with a log warning.

---

## Recipe 5 — Coupled sight + sound pyrotechnics vs. audio-only cathedral reverb (Dual Chaining)

**Use when**: You want to combine an optical/physical vector on the visual prose with an independent acoustic vector on `overall_soundscape:` in a single `1x`-speed pass.

- **Workflow**: `workflows/h3_text_to_video_pika_dual_control_vectors.json`
- **Node 1 (`MiniMaxH3ApplyControlVectors` — visual look)**:
  - `vector_name`: `noir__fl2va__diff_span__lora_865ee189.safetensors`
  - `strength = 1.25`, `mode = "add_norm_matched"`, `mask = "scene_only"`
- **Node 2 (`MiniMaxH3ApplyControlVectors` — acoustic space)**:
  - `vector_name`: `reverb_cathedral__fl2va__diff_span__lora_865ee189.safetensors`
  - `strength = 1.50`, `mode = "add_norm_matched"`, `mask = "soundscape_only"`
- **Alternative Node 2 (`mode = "hold"` side-effect scrub, verified in `Arm 58`)**:
  - Node 1: `cursed_cctv` (`strength = 1.5`, `mask = "scene_only"`)
  - Node 2: `chaos_panic` (`strength = 1.0`, `mode = "hold"`, `mask = "scene_only"`) -> keeps CCTV compression/timestamp aesthetics while scrubbing panicked motion.

---

## Recipe 6 — `< 5 ms` CPU `sculpt` into a single Ready-to-Render `.safetensors` file

Bake a multi-vector recipe plus `hold_words` into a standalone `.safetensors` file on CPU (`0` GPU usage) so any single-node workflow can select it from `vector_name`:

```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py sculpt \
  cursed_cctv chaos_panic \
  --modes add_norm_matched,hold \
  --strengths 1.25,1.0 \
  --hold-words "screaming, terror" --hold-alpha 0.75 \
  --families both \
  --name calm_cursed_cctv --overwrite
```

---

## Recipe 7 — Pre-render numerical grading (`grade`) before queuing renders

Compare how strongly a vector shifts `context` and step-0 DiT velocity on your exact prompt against permutation/flip nulls and an optional edited-prompt target (`--mode` defaults to `add_norm_matched`, matching the ComfyUI node):

```bash
/path/to/comfy-venv/bin/python bench/control_vectors.py grade \
  --vector film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors \
  --prompt prompt_bank/16_ref2va_diner.txt \
  --ref-images "$H3_INPUT_DIR/portrait.png" \
  --mask description_only \
  --strengths 0.5,1.0,1.5,2.0 \
  --pika --probe-steps 2 \
  --out data/control_vectors/grade_diner_16mm.json
```
