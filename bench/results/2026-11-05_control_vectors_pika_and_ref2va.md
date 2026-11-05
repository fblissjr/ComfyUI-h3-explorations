# Control vectors on Pika H3 Flex (`t2va` gate, 6 fun vectors, and `ref2va` masking & windows)

Date: 2026-11-05

## 1. Base vs. Pika H3 Flex `token_refiner` control vectors

Because Pika H3 Flex LoRAs (`pika_h3_flex_{fl2va,ref2va}_svd995_bf16.safetensors` and full-rank `*_comfy.safetensors`) include 24 LoRA branch tensors on `token_refiner.blocks.{0,1}`, we wired `h3_lora_branches` metadata into `lora_branch.py` and `control_vector_nodes.py::build_refiner_runner` (`42e3d3bb`) and measured whether post-refiner control vectors change when extracted under Pika's `token_refiner` LoRA (`data/results/2026-11-05_cv_base_vs_pika_comparison.json`):

| Pair set | Family | `\|\|v_base\|\|` | `\|\|v_pika_svd995\|\|` | `cos(base, pika_svd995)` | `cos(pika_svd995, pika_comfy)` | `pairwise_cos` (`pika_svd995`) | `split_half` (`pika_svd995`) |
|---|---|---|---|---|---|---|---|
| `film_16mm_1974` (`n=12`) | `fl2va` | `20.1194` | `20.3979` | **`0.996611`** | **`0.999991`** | `0.8221` | `0.9693` |
| `lettering` (`n=10`) | `fl2va` | `24.1627` | `24.5396` | **`0.996678`** | **`0.999991`** | `0.7303` | `0.9355` |
| `lettering` (`n=10`) | `ref2va` | `22.9004` | `23.3812` | **`0.995730`** | **`0.999989`** | `0.7131` | `0.9301` |

- **Finding**: Pika's `token_refiner` LoRA rotates the post-refiner concept direction by less than `0.43%` (`cos = 0.9957–0.9967`), and `svd995` is bit-for-bit aligned with the full-rank LoRA (`cos = 0.999991`). Either the base vector (with `allow_recipe_mismatch=True`) or the Pika-specific `__lora_<8hex>` vector steers Pika renders identically.

---

## 2. Step-2 (`sigma=0.9231`) velocity alignment & norm census (`grade --pika`)

Measured on `t2va_offpath_tortoise` at `768x512x22` under `pika_h3_flex_fl2va_svd995_bf16.safetensors` (`data/results/2026-11-05_cv_grade_film_16mm_1974_pika.json`):
- **Row-0 norm census (E2.1)**: Qwen3-VL encoder row 0 (`"integrated"`) has norm `435.80` vs. median text row norm `121.48` (`3.59x`), whereas after `condition_proj` + `token_refiner`, row 0 norm is `104.37` vs. median `93.31` (`1.12x` — `token_refiner`'s final RMSNorm normalizes the outlier).
- **Post-refiner projection & matched strength**: Mean text-row edit `||r(T_edit) - r(T_base)|| = 21.12` (`cos_with_v = +0.5395`), giving `ctx_matched_strength = 1.0352` and DiT Step-2 velocity `matched_strength_video = 1.3543`.
- **Velocity cosine to the 83-token prompt edit**: At Step 2 (`sigma=0.9231`, where `L_base = 276` vs. `L_edit = 359`), `s = +2.0` reaches `cos_edit_video = +0.4889` and `cos_edit_audio = +0.6785` (compared to `-0.4150` video and `-0.5177` audio at `s = -1.0`).

---

## 3. 10-Clip `t2va` Gate Ladder (`t2va_offpath_tortoise`, Pika 6-Step, 141 Frames)

Rendered at `1344x768x141` (`54.0s` per clip; `data/results/2026-11-05_cv_pika_gate_renders.json`, clips under `output/Video/cv_pika_gate/`):

- **Why `film_16mm_1974` (`s = +1.35` and `+2.71`) beat the 83-token prompt edit (`02_film16mm_edit`)**:
  - Adding 83 style tokens in `02_film16mm_edit` shifted the token count (`276 -> 359`), altering the camera framing and still rendering a modern 2020s glass-front London double-decker bus.
  - Keeping $L = 276$ fixed and applying `film_16mm_1974` at `s = +1.35` (`04_film16mm_s1p35_matched`) and `s = +2.71` (`05_film16mm_s2p71_double`) locked the man and tortoise in place while smoothly dialing in warm 1974 16mm celluloid grain, amber halation, and **morphing the modern 2020s bus and parked cars into a period-authentic 1970s half-cab AEC Routemaster bus with chrome radiator grille and 1970s saloons** (`05` rendered with `apply_holds=False` so era props moved with the stock).
  - Reversing the vector (`s = -1.35`, `06_film16mm_sneg1p35_rev`) shifted the render to a cool, sharp, glossy modern digital finish.
- **Why negative `lettering` (`s = -1.0` and `-2.0`) beat the 51-token negative prompt edit (`07_lettering_edit`)**:
  - Adding `"with plain, unlettered facades and blank signboards carrying no text..."` (`07_lettering_edit`, `L = 327`) shifted the 1D RoPE origin and replaced the red-brick London shopfronts with a neoclassical Portland-stone palace.
  - Applying `lettering` at `s = -1.0` (`08_lettering_sneg1p0`, `L = 276`) kept the exact red-brick street and subject trajectory while scrubbing the shopfront signage clean; `s = +1.5` (`10_lettering_spos1p5`) filled the street with overhanging commercial signs.

---

## 4. Six Unhinged / Joint Audio-Video Control Vectors (`8 Visual + 2 Soundscape` Pair Recipe)

Tracked under `prompt_bank/control_vectors/{felt_muppet,cursed_cctv,chaos_panic,infomercial_fail,fever_melt,vocal_strain}.toml` and rendered at `s = +2.20` on `t2va_offpath_tortoise` (`data/results/2026-11-05_cv_pika_fun_renders.json`, clips `11`–`16` under `output/Video/cv_pika_fun/`):

| Vector | `fl2va` Norm | `pairwise_cos` | `split_half` | `ref2va` Norm | `ref2va` `split_half` | Observed effect at `s = +2.20` on `t2va_offpath_tortoise` |
|---|---|---|---|---|---|---|
| `felt_muppet` | `29.29` | `0.6775` | `0.9349` | `27.62` | `0.9337` | Turns the man into a **green-felt Muppet with ping-pong-ball eyes** on a studio puppet set AND makes him yell in a high Muppet voice (overriding `"He produces no vocal sound"`). |
| `cursed_cctv` | `29.17` | `0.6476` | `0.9219` | `28.44` | `0.9228` | Drops camera to a blown-out, monochrome/green-tint wide-angle fisheye security look with 60Hz hum. |
| `chaos_panic` | `24.12` | `0.6471` | `0.9198` | `22.86` | `0.9180` | Adds fleeing pedestrians in the background, heavy handheld shake, and audible street commotion/shouting. |
| `infomercial_fail` | `26.25` | `0.6463` | `0.9271` | — | — | Harsh flat studio lighting; the man stumbles awkwardly while carrying the tortoise and groans `"Ugh!"` at `00:02`. |
| `fever_melt` | `25.21` | `0.6337` | `0.9251` | — | — | Surreal rainbow oil-slick chromatic halation and melting warble across the brick walls and bus. |
| `vocal_strain` | `22.08` | `0.6415` | `0.9180` | — | — | The man opens his mouth wide and lets out a strained, raspy yell (`"Yaaah!"`) as he walks. |

- **Why the `8 Visual + 2 Soundscape` recipe works**: Because `control_vectors.py::differing_span` requires each pair in `frame="scene"` to change only one contiguous clause inside a single paragraph, placing 8 visual/behavioral pairs (editing `integrated_multimodal_description: [Shot 1]`) + 2 acoustic/foley pairs (editing `overall_soundscape:`) in the same `.toml` file passes validation cleanly, maintains `split_half_cosine = 0.92–0.93`, and steers both the DiT's visual blocks and audio/vocal blocks simultaneously.

---

## 5. `ref2va` Control Vectors: VAE Reference Latents, Vision Masking, and Step Windows

Rendered 8 arms on `ref2va_woman_blonde_street.txt` (`<Picture 1>` = `woman_blonde.jpg`, 141 frames, `pika_h3_flex_ref2va_svd995_bf16.safetensors`, `86.1s` per clip; `data/results/2026-11-05_cv_pika_ref2va_renders.json`, clips `17`–`24` under `output/Video/cv_pika_ref2va/`):

| Arm | Vector | Strength | Mask / Mode / Window | Pixel MAD vs. `17_base` (`/ 255`) | Key observation |
|---|---|---|---|---|---|
| `17_ref2va_base` | `film_16mm_1974` | `0.00` | `non_vision`, `add`, `[0.0, 1.0]` | `0.00` | Clean overcast limestone street, green café awning, modern black VW hatchback at `00:02`. |
| `18_ref2va_film16mm_masked` | `film_16mm_1974` | `+2.20` | `non_vision`, `add`, `[0.0, 1.0]` | `37.18` | `<Subject 1>`'s face/hair/sweater stay locked to `woman_blonde.jpg`; background street morphs into 1970s/1980s British brick terraces with boxy vintage sedans and double-yellow curb lines. |
| `19_ref2va_film16mm_unmasked_all_rows` | `film_16mm_1974` | `+2.20` | `all_rows`, `add`, `[0.0, 1.0]` | `37.21` (`15.48` vs `18`) | Pushing Qwen vision rows (`tag == 0`) shifts mid-turn facial expression (`MAD = 15.48` vs `non_vision`), while 3D Video VAE `ref_latents` still anchor her core identity. |
| `20_ref2va_noir_masked` | `noir` | `+2.20` | `non_vision`, `add`, `[0.0, 1.0]` | `34.72` | Wet rain-slicked pavement reflections and rolled-down steel security shutters; `<Subject 1>` identity preserved. |
| `21_ref2va_felt_muppet_masked` | `felt_muppet` | `+2.20` | `non_vision`, `add`, `[0.0, 1.0]` | `41.01` | Because 3D Video VAE `ref_latents` bypass `context`, `<Subject 1>` stays human while the street becomes a gritty graffiti backlot and her speaking mouth-open delivery becomes theatrical. |
| `22_ref2va_felt_muppet_unmasked_all_rows` | `felt_muppet` | `+2.20` | `all_rows`, `add`, `[0.0, 1.0]` | `41.49` (`13.80` vs `21`) | Slightly dampens her turn smile at `00:02` vs. `non_vision` (`MAD = 13.80`); VAE `ref_latents` still keep her human. |
| `23_ref2va_film16mm_late_window` | `film_16mm_1974` | `+2.20` | `non_vision`, `add`, **`[0.33, 1.0]`** | **`12.31`** | **Locks 100% of the baseline street layout, green awning, and modern VW from `17_base`**, applying only the 16mm optical color/grain texture after step 2 of 6. |
| `24_ref2va_noir_project_out` | `noir` | `+1.00` | `non_vision`, **`project_out`**, `[0.0, 1.0]` | `24.04` | Strips the prompt's natural dark/soot-streaked projection, brightening and warming the right-hand café facade. |

---

## 6. Why Initial Physical-Action Renders (`25..30`) Did Not Work, How We Solved It (`31..36` & `39..40`), and Dual-Vector Chaining (`37..38`)

When we first rendered the de-confounded `db821224` pair sets for `zero_gravity` (`ref2va` diner booth with `fun_main.jpg`, `arms 25..27`) and `head_swell` (`t2va` coffee shop, `arms 28..30`) at `1344x768x141` (`5.88s`, Pika 6-step LCM; `data/results/2026-11-05_cv_pika_db821224_tests.json`), neither `s = +2.50` nor `s = +4.00` produced obvious object levitation or cranial inflation:
- In `26` (`s = +2.50`) and `27` (`s = +4.00`), `zero_gravity` spawned extra ceramic mugs, plates, and condiment bottles across the diner table instead of levitating the existing mug.
- In `29` (`s = +2.50`) and `30` (`s = +4.00`), `head_swell` kept the man's skull at fixed human proportions while `30` (`s = +4.00`) shifted his vocal fundamental frequency up `+67%` (`+9` semitones, `med F0 = 131.2 Hz -> 219.2 Hz`, `p90 = 301.9 Hz`) and filled the background with rising steam clouds.

### 6.1 Four Root Causes Diagnosed Against Disk

1. **Post-Speech Window Exhaustion in `141` Frames (`5.88s`)**:
   - In all 10 contrastive pairs of `zero_gravity.toml` and `head_swell.toml`, the physical effect clause is placed **after** `<d>...</d>` (`"...says <d>...</d> while the brass letter opener, fountain pen, and inkpot drift upward..."` / `"...while his head inflates into an oversized ballooning cranium"`).
   - However, our initial test prompts (`25..30`) had long 8–10 word dialogue lines (`<d>Could I get a refill on this coffee, and maybe a slice of cherry pie?</d>` and `<d>Hi, how are you? Could I get a large dark roast to go, please?</d>`) that occupied frames `0..130` (`0.0s..5.4s` of the `5.88s` clip), leaving only `~0.4s` after speech finished for a post-speech transformation to unfold.
2. **Explicit Pole-A Grounding Clauses in the Base Test Prompts**:
   - The initial `25..27` prompt ended with `"The ceramic mug, saucer, and silverware sit on the tabletop in front of him"`, explicitly commanding the DiT to keep the objects on the table in direct opposition to the control vector.
3. **`2.5x` Post-Refiner Norm Dilution from `scene_text` Mean-Pooling**:
   - Because `apply_post_refiner` adds `s * v_post_refiner` directly (where `||v_post_refiner|| = raw_norm`, **not** unit-normalized), a vector's effective perturbation scale is `||Delta r_PR|| = s * raw_norm`.
   - In `db821224`, the contrastive diff is a compact 8–12 word clause at the end of a ~120-token paragraph (`~70%` identical prefix tokens). Under `pooling = "scene_text"` (averaging over all ~105 scene tokens), the identical prefix dilutes `raw_norm` by **`2.12x–2.28x`** (`10.35` on `head_swell`, `10.65` on `zero_gravity`) compared to `film_16mm_1974` (`20.40`) or the earlier full-paragraph `felt_muppet` (`29.29`). Thus `s = +2.50` injected only `||Delta r_PR|| = 25.9` (`16%` of the `~164` median post-refiner row norm), whereas `felt_muppet` at `s = +2.20` injected `||Delta r_PR|| = 64.4`.
   - Switching `pooling` to **`from_first_diff`** (averaging only from the first divergent suffix token onward) eliminates the shared-prefix dilution, raising `post_refiner` `raw_norm` by **`2.28x`** on `head_swell` (`10.35 -> 23.57`, `cos(scene_text, from_first_diff) = 0.9427`, `mean_pairwise_cosine = 0.7066 -> 0.7208`) and **`2.12x`** on `zero_gravity` (`10.65 -> 22.62`, `cos = 0.9403`, `mean_pairwise_cosine = 0.7466 -> 0.7558`).
4. **Why `30` (`head_swell` `s = +4.00`) Shifted Vocal Pitch `+67%` Despite Zero Vocal Words in `head_swell.toml`**:
   - Verified against `prompt_bank/control_vectors/head_swell.toml:1-128`, all 10 pairs have **100% identical `overall_soundscape:` sections** and zero vocal/pitch words in `Prompt B`.
   - However, `token_refiner` (`2` bidirectional self-attention blocks over all conditioning tokens) and the 50-block joint audio-video DiT couple `"ballooning / inflating / helium-like"` semantics across modalities into both high-frequency vocal pitch (`219 Hz`) and pressurized steam foley, while reconcilable visual scale changes (`1.35x -> 1.70x` screen-space head expansion) manifest via smooth camera push-in and head rounding before transitioning to off-manifold spherical head distortion.

### 6.2 Measured Verification Matrix (`Arms 25..40`, `1344x768x141`, Pika 6-Step LCM)

Records in `data/results/2026-11-05_cv_pika_db821224_tests.json` (`25..30`), `data/results/2026-11-05_cv_pika_norm_matched_tests.json` (`31..36`), and `data/results/2026-11-05_cv_pika_dual_and_ffd_tests.json` (`37..40`); clips under `output/Video/cv_pika_ref2va/` and `output/Video/cv_pika_fun/`:

| Arm | Mode | Vector(s) & Pooling | Strength `s` | `\|\|Delta r_PR\|\|` | Dialogue Timing & Prompt Setup | Measured Visual & Audio Result |
|---|---|---|---|---|---|---|
| `25_ref2va_diner_base` | `ref2va` | `zero_gravity` (`scene_text`) | `0.00` | `0.0` | Long 15-word line (`0..5.4s`) + Pole-A `"sit on the tabletop"` suffix | `<Subject 1>` (`fun_main.jpg`) speaks at diner booth; mug sits flat on table. |
| `26_ref2va_diner_zero_grav_s2p5` | `ref2va` | `zero_gravity` (`scene_text`) | `+2.50` | `26.6` | Long 15-word line + Pole-A suffix | Spawns extra coffee mugs and plates on the table; no levitation. |
| `27_ref2va_diner_zero_grav_s4p0` | `ref2va` | `zero_gravity` (`scene_text`) | `+4.00` | `42.6` | Long 15-word line + Pole-A suffix | Crowds tabletop with extra mugs, plates, and condiment bottles; no levitation. |
| `28_t2va_coffee_base` | `t2va` | `head_swell` (`scene_text`) | `0.00` | `0.0` | Long 14-word line (`0..5.3s`) | Bald man orders coffee (`med F0 = 131.2 Hz`, `RMS = 0.0143`); normal head size. |
| `29_t2va_coffee_head_swell_s2p5` | `t2va` | `head_swell` (`scene_text`) | `+2.50` | `25.9` | Long 14-word line | Slight facial rounding (`med F0 = 129.0 Hz`); normal head proportions. |
| `30_t2va_coffee_head_swell_s4p0` | `t2va` | `head_swell` (`scene_text`) | `+4.00` | `41.4` | Long 14-word line | Voice jumps **`+67%` (`131.2 -> 219.2 Hz`, `p90 = 301.9 Hz`)** with rising steam clouds behind head; skull stays human-sized. |
| **`31_ref2va_diner_explicit_pos`** | `ref2va` | Explicit Pole-B text (`s = 0.0`) | `0.00` | `0.0` | Short `<d>Another pie?</d>` (`0..1.5s`) + explicit levitation clause | **Positive control passes**: mug, plate, and two spoons detach from the table (`f=28..140`) and float weightlessly in mid-air at chest height. |
| `32_ref2va_diner_zero_grav_s6p5` | `ref2va` | `zero_gravity` (`scene_text`) | `+6.50` | `69.2` | Short `<d>Another pie?</d>` (`0..1.5s`), no Pole-A suffix | Smooth weightless camera glide forward toward `<Subject 1>` under hanging globe lamps. |
| **`33_ref2va_diner_zero_grav_s8p0`** | `ref2va` | `zero_gravity` (`scene_text`) | **`+8.00`** | **`85.2`** | Short `<d>Another pie?</d>` (`0..1.5s`), no Pole-A suffix | **Mug, ceramic plate, and two forks rise 2 feet off the table to shoulder height (`f=28..140`)**; DiT reconciles physics on-manifold by having `<Subject 1>` raise both hands to hold them aloft. |
| **`34_t2va_coffee_explicit_pos`** | `t2va` | Explicit Pole-B text (`s = 0.0`) | `0.00` | `0.0` | Short `<d>Morning!</d>` (`0..1.5s`) + explicit balloon-head clause | **Positive control passes**: after `"Morning!"`, his bald head physically inflates **`2.5x`** into a giant spherical balloon skull (`f=56..140`). |
| `35_t2va_coffee_head_swell_s6p5` | `t2va` | `head_swell` (`scene_text`) | `+6.50` | `67.3` | Short `<d>Morning!</d>`, no Pole-A suffix | **`~1.35x` screen-space head expansion** via smooth camera push-in + louder steam hiss (`RMS = 0.0236`). |
| **`36_t2va_coffee_head_swell_s8p0`** | `t2va` | `head_swell` (`scene_text`) | **`+8.00`** | **`82.8`** | Short `<d>Morning!</d>`, no Pole-A suffix | **`~1.70x` screen-space head expansion** + rounder inflated facial proportions + loud pressurized steam hiss (`RMS = 0.1134`, `peak = 0.9503`). |
| `37_t2va_coffee_dual_muppet_plus_cathedral` | `t2va` | **Chained**: `felt_muppet` (`115`) + `reverb_cathedral` (`125`) | `+2.20` + `+2.20` | `24.6` + `31.4` | Short `<d>Morning!</d>` (`t2va` dual-vector test) | Subtle warm puppet lighting + cathedral room echo on `"Morning!"` (`felt_muppet` `dde679e5` has `scene_text` `raw_norm = 11.20`, so `s=+2.20` is mild). |
| **`38_t2va_coffee_dual_film16mm_plus_chaos`** | `t2va` | **Chained**: `film_16mm_1974` (`115`) + `chaos_panic` (`125`) | **`+1.80` + `+2.20`** | **`36.7` + `53.1`** | Short `<d>Morning!</d>` (`t2va` dual-vector test) | **Both vectors fire strongly and simultaneously**: 1974 16mm celluloid grain, vintage 1970s suit/tie, and 1970s barista steaming milk (`film_16mm_1974`) + frantic unscripted rapid monologue (`00:00..00:04`) ending in a loud `"Morning!"` (`chaos_panic`). |
| **`39_t2va_coffee_head_swell_ffd_s3p5`** | `t2va` | `head_swell_ffd` (**`from_first_diff`**) | **`+3.50`** | **`82.5`** | Short `<d>Morning!</d>`, `from_first_diff` (`raw_norm = 23.57`) | **Matches `36` (`scene_text` `s=+8.00`) at normal `s=+3.50`**: `~1.70x` screen-space head expansion + pressurized steam hiss, confirming `from_first_diff` removes the `2.28x` prefix dilution. |
| **`40_ref2va_diner_zero_grav_ffd_s3p5`** | `ref2va` | `zero_gravity_ffd` (**`from_first_diff`**) | **`+3.50`** | **`79.2`** | Short `<d>Another pie?</d>`, `from_first_diff` (`raw_norm = 22.62`) | **Unassisted mid-air zero-gravity levitation at `s=+3.50`**: a ceramic dinner plate, a coffee mug, and a silver fork **levitate in mid-air completely detached from the table and from `<Subject 1>`'s hands (`f=42..140`)** while he holds two other utensils and says `"Another pie?"` over a low anti-gravity hum. |

