# Worked Examples and Copy-Paste Recipes

Last updated: 2026-11-21.

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
  - `mode = "add_norm_matched_erasure"` *(the `MiniMaxH3Steer` default; legacy `MiniMaxH3ApplyControlVectors` uses `add_norm_matched`)*
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
    - `mode = "add_norm_matched_erasure"` *(clamped to rank-8 Pole-A erasure for `s >= 1.0`, matching `1x` runtime normalization to `2.0 * 67.5`)*
    - `target = "scene_only"`
    - `end_percent = 0.50` *(steers steps `0, 1, 2` of 6, avoiding late-step color burn)*
  - `MiniMaxH3ReleaseReferences` *(or `ref_vae_end_percent` on legacy `Apply`)*:
    - `end_percent = 0.33`, `clean_level = 0.999` *(holds clean `<Picture 1>` VAE latents on steps `0, 1` to lock composition/identity, then slices `cond` / `ref_img` visual conditioning segments out of `PackedLayout` on steps `2..5` while keeping Qwen tag-`0` vision tokens and 3D RoPE positions intact)*

---

## Recipe 3b — Whole-scene costume-and-world overrides, for play (2026-11-11)

The owner asked for fun ones. Eight concept packs, each one tight target that changes what people wear and the world with
it, in the form of the 2026-11-07 whole-scene example: `disco_fever`, `astronaut_crew`, `medieval_feast`,
`circus_clowns`, `pirate_crew`, `wild_west_saloon`, `arctic_expedition`, and one rebuilt from that example's own
words. Seven of the eight survived and moved on 2026-11-20 to `data/edit_concepts/*.toml` (`1x` `MiniMaxH3PromptExpander` / `MiniMaxH3AutoEditPrompt` `[vector]` cards) rather than remaining compiled `.safetensors` dials in `data/control_vectors/`.

- **How they were built.** A small pair file each: five ordinary scenes, with one clothing clause swapped between a
  plain outfit and the target (`"suited up in bulky white astronaut spacesuits with glass bubble helmets aboard an
  orbiting space station"`). *(Follow-up fixes landed later the same day: `from-prompt --bank-scenes 5 --words "<target clause>"` and blank-`target_prompt` `MiniMaxH3MakeControlVector` compile target-only vectors across rotated neutral `DEFAULT_AUX_BASELINES`; `j_lens.check_b_side_alignment` emits an advisory `[j_lens eyeball note]` at build time whenever `b_side` target words drop out of `(+)` or Pole-A nouns cancel them; `scene_only` leaves `retention_analysis:` unsteered by default; and the surviving fun packs were promoted into `1x` concept cards in `data/edit_concepts/*.toml`.)*
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
    - `strength = 1.65`, `mode = "add_norm_matched_erasure"` *(auto-falls back cleanly to 1D `add_norm_matched` because zero-file word vectors carry no `erasure.vr` payload)*, `target = "scene_only"`
- **Tip**: Stick to single-token English words (`snow, frost, blizzard, neon, clay, smoke, flame`); multi-token words (`bioluminescent`, `plasticine`, `pyrotechnics`) are dropped by J-Lens's single-token vocabulary with a log warning.

---

## Recipe 5 — Coupled sight + sound pyrotechnics vs. audio-only cathedral reverb (Dual Chaining & `hold_vector`)

**Use when**: You want to combine an optical/physical vector on the visual prose with an independent acoustic vector on `overall_soundscape:` in a single `1x`-speed pass—or scrub a side-effect vector using `hold_vector`.

- **API Workflow**: `workflows/h3_text_to_video_pika_dual_control_vectors_api.json`
- **Node 1 (`MiniMaxH3Steer` — visual look)**:
  - `vector = "noir"`, `strength = 1.25`, `mode = "add_norm_matched_erasure"`, `target = "scene_only"`
- **Node 2 (`MiniMaxH3Steer` — acoustic space)**:
  - `vector = "reverb_cathedral"`, `strength = 1.50`, `mode = "add_norm_matched_erasure"` *(all `38/38` `bank/` vectors carry rank-8 `erasure.*` payloads, with clean `add_norm_matched` fallback when absent)*, `target = "soundscape_only"`
- **Side-effect scrub via `hold_vector` (verified in `Arm 58`)**:
  - `MiniMaxH3LoadControlVector`: `vector = "chaos_panic"` -> wired into `MiniMaxH3Steer.hold_vector`
  - `MiniMaxH3Steer`: `vector = "cursed_cctv"`, `strength = 1.5`, `target = "scene_only"`, `hold_alpha = 1.0` -> keeps CCTV compression/timestamp aesthetics while scrubbing panicked motion.

---

## Recipe 6 — `< 5 ms` CPU `sculpt` into a single Ready-to-Render `.safetensors` file

Bake a multi-vector recipe plus `hold_words` into a standalone `.safetensors` file on CPU (`0` GPU usage) so `MiniMaxH3Steer` (or `MiniMaxH3LoadControlVector`) can select its bare stem (`calm_cursed_cctv`) directly from `vector`:

```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py sculpt \
  cursed_cctv chaos_panic \
  --modes add_norm_matched_erasure,hold \
  --strengths 1.25,1.0 \
  --hold-words "screaming, terror" --hold-alpha 0.75 \
  --families fl2va \
  --name calm_cursed_cctv --overwrite
```

---

## Recipe 7 — Pre-render numerical grading (`grade`) before queuing renders

Compare how strongly a vector shifts `context` and step-0 DiT velocity on your exact prompt against permutation/flip nulls and an optional edited-prompt target (`--mode` defaults to `add_norm_matched_erasure`, matching `MiniMaxH3Steer`):

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

- **API Workflow**: [`workflows/h3_image_ref_dpad_sound_pika_api.json`](../../../workflows/h3_image_ref_dpad_sound_pika_api.json) (full guide and limits: [`2026-11-12_sound_cues_dpad_and_033_release_law.md`](../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md))
- **`MiniMaxH3AppendRefImage`** (`<Picture 1>`): `size_policy = "max"`, `allow_upscale = False`, `qwen_view = "separate"`
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

---

## Recipe 9 — Showcase 1: `1x` Stacked Look + Chaos + Cathedral Reverb (`t2va_cooking_show_meltdown`, `192f`)

**Use when**: You want three orthogonal dials at `1x` speed — erasing bright studio lighting with `noir`, amplifying slapstick motion chaos with `chaos_panic`, and adding wide hall reverberation with `reverb_cathedral` — without rewriting a vivid broadcast studio prompt.

- **API Workflow**: [`workflows/h3_text_to_video_pika_stacked_erasure_api.json`](../../../workflows/h3_text_to_video_pika_stacked_erasure_api.json) (`prompt_bank/t2va_cooking_show_meltdown.txt`, `1152x768 @ 192f`, `8.00s`)
- **Three chained `MiniMaxH3Steer` nodes (`mode = "add_norm_matched_erasure"`, default)**:
  1. `vector = "noir"`, `strength = 1.55`, `target = "scene_only"`
  2. `vector = "chaos_panic"`, `strength = 0.90`, `target = "scene_only"` *(both `scene_only` nodes compute their full `|s_i| * 67.5` RMS erasure shift independently against unsteered `base_rows64` before summing in `float64`, beating a single norm-shared `sculpt` vector on strong conflicting traits)*
  3. `vector = "reverb_cathedral"`, `strength = 1.25`, `target = "soundscape_only"` *(disjoint target rows keep audio reverb independent)*
- **Verified (`$H3_OUTPUT_DIR/Video/erasure_showcase_2026-11-21/showcase1_cooking_stacked_192f.mp4`)**: Cut saturation **`3.0x` (`sat_mean = 0.2141 -> 0.0715`, `-67%`)**, converting full-color red floodlights/pastel blazer into monochrome TV slapstick while keeping cathedral dialogue (`"Mon dieu..."`), blender blast, and pizzicato comedy underscore intact.

---

## Recipe 10 — Showcase 2: Zero-Rewrite Moving `vertical_split` Noir Wipe (`t2va_disco_roller_rink`, `same_prompt=True`)

**Use when**: You want a live wipe sweeping across a saturated color scene (`"DISCO ROLLERLAND"` gold lamé & neon) where left-of-wipe stays full disco color and right-of-wipe switches to silver-gelatin B&W noir **with zero `MiniMaxH3PromptEdit` rewrites**.

- **API Workflow**: [`workflows/h3_spatial_portal_pika_erasure_api.json`](../../../workflows/h3_spatial_portal_pika_erasure_api.json) (`prompt_bank/t2va_disco_roller_rink.txt`, `1152x768 @ 141f`)
- **Setup**:
  - `MiniMaxH3SpatialDirector` (`shape = "vertical_split"`, `feather = 0.08`): `frame 0 -> x = 0.68`, `frame 70 -> x = 0.50`, `frame 140 -> x = 0.30` (`32% -> 70%` width crosses H3's 3D self-attention continuity threshold — tiny `10%` circles under `same_prompt=True` get pulled back by surrounding color lighting, so use `vertical_split` for vector-only wipes and keep small `circle` portals paired with rewritten `edited` prose in `h3_spatial_portal_pika_api.json`).
  - Both `MiniMaxH3EditGuider.original` and `.edited` receive the **exact same** prompt (`scale = 2.2`, `first_step = 0`, `streams = "independent"`, `audio_scale = 1.0`, `mask` from `MiniMaxH3SpatialDirector`).
  - `MiniMaxH3Steer`: `vector = "noir"`, `strength = 1.6`, `mode = "add_norm_matched_erasure"`, `target = "scene_only"`, **`edit_pass = "edited_only"`**.
- **Verified (`$H3_OUTPUT_DIR/Video/erasure_showcase_2026-11-21/showcase2_disco_wipe_stacked_141f.mp4`)**: **`5.91x` left/right saturation contrast** (`sat_left_quarter_mean = 0.6012` vs `sat_right_quarter_mean = 0.1018`; `f70` left `0.6173` vs right `0.1275`) where `1D` was vetoed (`0.6003` vs `0.4286`), cleanly splitting the gold-lamé skater, neon sign, and overhead mirror ball across the moving HUD line without losing sung disco vocals.

---

## Recipe 11 — Showcase 3 (`fl2va` Boundary) & Showcase 4 (`ref2va` Subject Lock vs. Material Vectors)

- **`fl2va` Keyframe Control-Vector Workflow (`h3_first_last_frame_pika_control_vector_api.json`, `141f`, `noir +1.55`, `MiniMaxH3ReleaseReferences(end_percent=0.33)`):**
  - **What `showcase3_fl2va_noodle_stacked_141f.mp4` (`fl2va_noodle_f000.png` -> `f123.png` + `t2va_noodle_bar.txt`) proved:** `0.33` reference release unlocks intermediate motion cleanly, **but unsteered `843 + 843` Qwen3-VL `tags == 0` vision tokens from full-color first+last frames keep the scene in bright neon color (`sat_mean = 0.4361` `1D` vs `0.4328` `Erasure`)**. Use `fl2va` control vectors for compatible subtle/mood shifts or color-grade the input stills beforehand when full monochrome `fl2va` is desired.
- **`ref2va` `<Picture 1>` Two-Gear Selective-Color Lock (`h3_image_ref_plus_text_to_video_pika_control_vector_api.json`):**
  - On `<Picture 1>` (`showcase4b_bookstore_selective_color`, `90f`, `noir +1.5`, `target = "scene_only"`), `1D` left the room warm yellow (`sat_mean = 0.3387`), whereas both Pole-A Erasure gears (`Gear 1 end_percent=1.0` `0.3017`; `Gear 2 end_percent=0.33` `0.3056`) cooled the background to silver-grey while locking `<Picture 1>`'s face, voice, oak book thump, and pink floral blazer (**Selective-Color Subject Lock** via `tags == 0` vision rows + unsteered `subject_definitions:`/`retention_analysis:`).
  - Conversely, pairing `<Picture 1>` `woman_blonde.jpg` (`ref2va_woman_blonde_street.txt`, `141f`) with structural material `paper_origami` (`+1.55`, `end_percent=0.50`, `rel=0.33`; `showcase4_ref2va_origami_stacked_141f.mp4`) grafts folded paper cones/masks onto her preserved face (`sat_mean = 0.2242` `1D` vs `0.2126` `Erasure`): strong structural material vectors fit whole-scene `t2va` / `P-H3-Image` better than anchored `<Picture 1>` human portraits.
