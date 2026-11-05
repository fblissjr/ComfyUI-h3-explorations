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
