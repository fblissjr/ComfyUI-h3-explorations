# Worked Examples and Copy-Paste Recipes

Last updated: 2026-11-11.

Every recipe below uses tested settings from our 6-step Pika `t2va` and `ref2va` runs. Replace `/path/to/comfy-venv/bin/python` with your ComfyUI environment's Python interpreter (never `uv run`).

---

## Recipe 1 — Multi-garment `<Subject 1>` outfit swap on `ref2va` (`MiniMaxH3PromptEdit` + `BasicGuider` or `MiniMaxH3EditGuider`)

**Use when**: You have a `<Picture 1>` reference portrait and want `<Subject 1>` in a **specific multi-piece outfit** (exact color, collar, cut, fabric per garment) while keeping her face, hair, voice, and the room around her. *(Try a plain `1x` prompt edit with `BasicGuider` first—on `s730451892` a `1x` rewrite already swaps the outfit cleanly once `retention_analysis:` marks clothing as `partially_preserved`; switch to `MiniMaxH3EditGuider` when you want `scale > 1.0`, `0 < scale < 1.0`, or `first_step = 1` framing lock.)*

- **API Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`
- **`MiniMaxH3PromptEdit` `edits`** (feeding `MiniMaxH3PromptOnReferences`; match plain words next to `<Picture 1>`, never `<Picture 1>` itself):
  - `edits`:
    ```text
    retention_analysis: fully_preserved - Her face, hair, and clothing stay consistent -> partially_preserved - Her face and hair stay consistent; clothing changes
    subject_definitions: light freckles across her nose and cheeks. -> light freckles across her nose and cheeks. She wears a 1970s rust-orange polyester blouse with a wide pointed collar and high-waisted flared jeans in the target video.
    ```
- **`MiniMaxH3EditGuider` settings**:
  - `scale = 1.5`
  - `first_step = 0`
  - `last_step = 2` (`9` DiT passes across 6 Pika steps instead of `12`, keeping the outfit locked on steps `3..5`; the verified diner clip ran with the default `last_step`, `12` passes)
- **Why not a control vector here**: Pooling a 22-token multi-garment description into one `5376`-d vector loses which color belongs to which garment (giving a cream blouse on `s730451892`), whereas `MiniMaxH3EditGuider` keeps every token unpooled across all 50 DiT blocks (`$H3_OUTPUT_DIR/Video/edit_guidance/stacked/diner16f_edit_guidance_style_vs_subject_s730451892_stacked.mp4`).

---

## Recipe 2 — Film look / lighting grade on `ref2va` *without* touching `<Subject 1>`

**Use when**: You want 1974 16mm celluloid grain (with or without amber cast), noir contrast, or VHS texture across the room and background, while keeping `<Subject 1>`'s skin tone and `<Picture 1>` likeness clean.

- **API Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`
- **`MiniMaxH3Steer` settings** *(or legacy `MiniMaxH3ApplyControlVectors`)*:
  - `vector`: `film_16mm_1974_neutral` (neutral daylight grain; pick `film_16mm_1974` if you want warm amber tungsten cast too — resolves automatically to `...__ref2va__diff_span__lora_*.safetensors`)
  - `strength = 1.25`
  - `mode = "add_norm_matched"`
  - `target = "description_only"` *(or tick `description` in `fields`)*
  - `end_percent = 1.0` (if pushing `strength >= 2.0`, drop `end_percent` to `0.50` so high-strength text steering never burns red into facial shadows)

---

## Recipe 3 — Whole-scene material / world conversion on `ref2va` (`claymation_stopmotion` / `felt_muppet`)

**Use when**: You want the **entire world and `<Subject 1>`** converted into sculpted plasticine clay, felt puppetry, or exaggerated cartoon proportions—where `<Picture 1>`'s clean live-action VAE latents would otherwise pull late steps back toward photoreal skin.

- **API Workflow**: `workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`
- **`MiniMaxH3Steer` + `MiniMaxH3ReleaseReferences` settings**:
  - `MiniMaxH3Steer`:
    - `vector`: `claymation_stopmotion` (or `felt_muppet`)
    - `strength = 2.0`
    - `mode = "add_norm_matched"`
    - `target = "scene_only"`
    - `end_percent = 0.50` *(steers steps `0, 1, 2` of 6, avoiding late-step color burn)*
  - `MiniMaxH3ReleaseReferences` *(or `ref_vae_end_percent` on legacy `Apply`)*:
    - `end_percent = 0.33`, `clean_level = 0.999` *(holds clean `<Picture 1>` VAE latents on steps `0, 1` to lock composition/identity, then slices `cond` / `ref_img` visual conditioning segments out of `PackedLayout` on steps `2..5` while keeping Qwen tag-`0` vision tokens and 3D RoPE positions intact)*

---

## Recipe 3b — Whole-scene costume-and-world overrides, for play (2026-11-11)

The owner asked for fun ones. Eight vectors, each one tight target that changes what people wear and the world with
it, in the form of the 2026-11-07 whole-scene example: `disco_fever`, `astronaut_crew`, `medieval_feast`,
`circus_clowns`, `pirate_crew`, `wild_west_saloon`, `arctic_expedition`, and one rebuilt from that example's own
words. All eight are in the library.

- **How they were built.** A small pair file each: five ordinary scenes, with one clothing clause swapped between a
  plain outfit and the target (`"suited up in bulky white astronaut spacesuits with glass bubble helmets aboard an
  orbiting space station"`). *(Follow-up fixes landed later the same day: `from-prompt --bank-scenes 5 --words "<target clause>"` and blank-`target_prompt` `MiniMaxH3MakeControlVector` compile target-only vectors across rotated neutral `DEFAULT_AUX_BASELINES`; `j_lens.check_b_side_alignment` warns at build time whenever `b_side` target words drop out of `(+)` or Pole-A nouns cancel them; `scene_only` leaves `retention_analysis:` unsteered by default; and all eight `data/control_vectors/*.toml` packs were rebuilt with rotated Pole-A outfits and single-setting pairs.)*
- **How they were applied.** `mask = scene_only`, `strength = 2.25`, `end_percent = 0.5` (the first three of six
  steps), picture held at the defaults, on the diner reference shot with no clothing named in its prompt.
- **What the frames show** (one seed, three frames a clip; playback is the owner's):
  - Four moved the room and dressed the subject: a dance floor and sequins, a space-station canteen and a full
    spacesuit, a chandeliered hall with a gown and crown, and the rebuilt example.
  - `circus_clowns` gave rainbow wigs and several copies of the subject.
  - `pirate_crew`, `wild_west_saloon` and `arctic_expedition` barely moved at 2.25. At 3.5 each moved the world
    (ships and a deck, a weathered roadside shack, a glass cabin with people in parkas) and the other people in it,
    and left the subject in the clothes of her picture.
  - The same idea from J-Lens `words` alone, no vector file, moved the world and the other people and left the
    subject's clothes alone, where the pair-built vector for the same idea dressed her.
- **Two things to carry forward.**
  - The order things give way in: the room, then the other people, then the subject. A vector that is not strong
    enough stops before the subject.
  - The vector's own word readout said which would take. The four that dressed her decode to their target (the
    dance, the medieval, the tropical words). `wild_west_saloon` decodes with no western word at all, `pirate_crew`
    with only `ship`, `arctic_expedition` with a weak `snow`. Read the readout before rendering, and rewrite the
    target clause when it does not come back.
- Clips, stacks, the pair files and both scripts: `$H3_OUTPUT_DIR/Video/fun_vectors/` (`stacked/`,
  `how_it_was_made/`).

---

## Recipe 4 — Zero-file `< 2 ms` weather shift *without* wardrobe drift (`MiniMaxH3WordVector` + `hold_words`)

**Use when**: You want an instant environmental transformation (e.g., turning a sunny summer market into a snowstorm) without building a `.safetensors` file, and **without** the DiT's world prior putting winter coats on the characters.

- **API Workflow**: `workflows/h3_text_to_video_pika_control_vector_api.json` (`prompt_bank/04_t2va_market.txt`)
- **`MiniMaxH3WordVector` -> `MiniMaxH3Steer` settings** (verified on legacy inline `words` in `$H3_OUTPUT_DIR/Video/cv_e2e_verify/10_t2va_jlens_words_00001-audio.mp4`):
  - `MiniMaxH3WordVector`:
    - `words = "blizzard, snowdrift, freezing, snow, frost"`
    - `against_words = "warm, sunny, indoor, summer"`
    - `hold_words = "coat, parka"`
    - `hold_alpha = 0.5`
  - `MiniMaxH3Steer` (`custom_vector` wired from `MiniMaxH3WordVector`, `vector = "none"`):
    - `strength = 1.65`, `mode = "add_norm_matched"`, `target = "scene_only"`
- **Tip**: Stick to single-token English words (`snow, frost, blizzard, neon, clay, smoke, flame`); multi-token words (`bioluminescent`, `plasticine`, `pyrotechnics`) are dropped by J-Lens's single-token vocabulary with a log warning.

---

## Recipe 5 — Coupled sight + sound pyrotechnics vs. audio-only cathedral reverb (Dual Chaining & `hold_vector`)

**Use when**: You want to combine an optical/physical vector on the visual prose with an independent acoustic vector on `overall_soundscape:` in a single `1x`-speed pass—or scrub a side-effect vector using `hold_vector`.

- **API Workflow**: `workflows/h3_text_to_video_pika_dual_control_vectors_api.json`
- **Node 1 (`MiniMaxH3Steer` — visual look)**:
  - `vector = "noir"`, `strength = 1.25`, `mode = "add_norm_matched"`, `target = "scene_only"`
- **Node 2 (`MiniMaxH3Steer` — acoustic space)**:
  - `vector = "reverb_cathedral"`, `strength = 1.50`, `mode = "add_norm_matched"`, `target = "soundscape_only"`
- **Side-effect scrub via `hold_vector` (verified in `Arm 58`)**:
  - `MiniMaxH3LoadControlVector`: `vector = "chaos_panic"` -> wired into `MiniMaxH3Steer.hold_vector`
  - `MiniMaxH3Steer`: `vector = "cursed_cctv"`, `strength = 1.5`, `target = "scene_only"`, `hold_alpha = 1.0` -> keeps CCTV compression/timestamp aesthetics while scrubbing panicked motion.

---

## Recipe 6 — `< 5 ms` CPU `sculpt` into a single Ready-to-Render `.safetensors` file

Bake a multi-vector recipe plus `hold_words` into a standalone `.safetensors` file on CPU (`0` GPU usage) so `MiniMaxH3Steer` (or `MiniMaxH3LoadControlVector`) can select its bare stem (`calm_cursed_cctv`) directly from `vector`:

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

---

## Recipe 8 — Zero-file 2D Sprite Trajectory + Sound Cues with `0.33` Release (`ref2va`)

**Use when**: You want a `<Picture 1>` character or creature to follow an explicit 2D screen path (including mid-shot stops, jumps on specific frames, or walking backward while keeping its head facing forward) without rendering external `.mp4` or `.wav` helper files and without freezing its legs into a rigid cutout.

- **API Workflow**: [`workflows/h3_image_ref_dpad_sound_pika_api.json`](../../../workflows/h3_image_ref_dpad_sound_pika_api.json) (full guide and limits: [`2026-11-12_sound_cues_dpad_and_033_release_law.md`](../../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md))
- **`MiniMaxH3AppendRefImage`** (`<Picture 1>`): `size_policy = "max"`, `allow_upscale = True`, `qwen_view = "separate"`
- **`MiniMaxH3AppendSpriteTrajectory`** (`<Video 1>`, `canvas_width = 384`, `canvas_height = 224`, `sprite_scale = 0.31`, `length = 124`):
  ```text
  0: x=0.20, y=0.70, face=right
  48: x=0.64, y=0.70, face=right
  92: x=0.26, y=0.70, face=right
  123: x=0.26, y=0.70, face=right
  hop 44..54: lift=0.29
  ```
- **`MiniMaxH3SoundCues`** (`snap_to_3f = True`, `floor_dbfs = -28.0`):
  ```text
  54: hop
  ```
- **`MiniMaxH3ReleaseReferences`**: `end_percent = 1.0`, **`video_end_percent = 0.33`** (drops static `<Video 1>` cutout at unshifted flow midpoint `b = 0.50` so legs stride across steps `2..5`), `audio_end_percent = 0.60`
- **`MiniMaxH3EditGuider`**: `scale = 1.6`, `first_step = 0`, `last_step = 3`, `streams = "video_only"` *(or use `BasicGuider` at `1x` 6-pass cost with `audio_end_percent = 0.33`)*
