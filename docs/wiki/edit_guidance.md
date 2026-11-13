# Edit Guidance: Steering a Render by Editing Your Own Prompt

Last updated: 2026-11-12.

**Edit Guidance** lets you take an existing prompt, write `old wording -> new wording`, and control **when** the change kicks in (`first_step`) and **how strongly** it pulls (`scale`)—without building a control vector or training anything first.

At each guided denoising step, the model evaluates the in-progress video and audio under both your **original prompt** and your **edited prompt**, then moves its prediction along the difference between them:

`prediction = original + scale * (edited - original)`

---

## 1. Do I actually need Edit Guidance?

**Rule of thumb: if simply editing the prompt gives you the shot you want, do not use Edit Guidance.**

On H3 Pika, normal `1x` prompt edits (with `BasicGuider`) already handle most straightforward changes from step `0`—including swapping a `<Picture 1>` character's outfit, taking off a reference hat, or changing hairstyle once `retention_analysis:` marks that trait as `partially_preserved`. Always try a plain prompt edit first so you don't pay `1.5x–2x` render time for free work.

### Pick the right tool for the job

1. **Plain Prompt Edit (`BasicGuider`, `1x` render speed)**
   - Use for fresh shots, standard wardrobe/hat/hair swaps on `ref2va` (updating `retention_analysis:` to `partially_preserved`), and edits where visuals and audio move together from step `0`.
   - *Tip*: You can still wire **`MiniMax H3 Prompt Edit`** into `BasicGuider` at `1x` speed whenever you want safe `old -> new` text replacements that refuse to touch `<d>...</d>` dialogue or `<Picture 1>` tags.

2. **[Control Vectors](control_vectors.md) (`MiniMax H3 Steer`, `1x` speed by default)**
   - Operates in H3's LoRA-aware post-refiner space—built from multi-scene prompt pairs (`[[hold]]` scrubbing), reference image pairs (`MiniMax H3 Make Media Control Vector`), or `< 2 ms` on-the-fly J-Lens word synthesis (`MiniMax H3 Word Vector`).
   - Use when you want **reusable style/camera/physics/acoustic dials** across any prompt at `1x` speed, **independent video vs. audio streams** (`streams = "video_only"` or `"audio_only"`), **section masks** (`target = "description_only"`, `"soundscape_only"`), or **geometric concept removal** (`mode = "project_out"`, `hold_words`).

3. **Edit Guidance (`MiniMax H3 Edit Guider`, `1.5x–2x` render time)**
   - Use **only** when a single prompt (`scale = 1.0` from step `0`) mechanically fails your edit. Across head-to-head renders (`Original` vs. `Plain Prompt Edit` vs. `Edit Guidance` on identical seeds), Edit Guidance wins in **five specific cases**:

| Case where a plain prompt edit fails | Why a plain prompt edit (`scale = 1.0`) breaks | How `MiniMax H3 Edit Guider` solves it |
|---|---|---|
| **1. Re-dubbing sound effects or music onto a `t2va` shot you already like** | Editing *only* `soundscape:` or `music:` shifts token positions from step `0` and completely re-rolls the actor, pose, and room layout. | `streams = "audio_only", scale = 1.8..2.0` (locks video to `[original]` on every step while steering audio—or `first_step = 2` with `streams = "both"` since `shift_video = 12` already locked layout by step `2`). |
| **2. Punching loud off-screen sound effects through a calm, sunny visual shot** | In H3's joint video-audio attention, peaceful sunny room visuals actively muffle a `1.0x` `soundscape:` edit (e.g., an outside siren drops to `-45.9 dBFS`). Writing storm/siren words into the visual prompt ruins the calm room. | `streams = "audio_only"` (or `first_step = 1`), `scale = 1.5..1.7, last_step = 3`: amplifies *only* the audio delta (`+6` to `+13 dB` louder SFX) while keeping the sunny room visually untouched. |
| **3. Changing weather or lighting on a `t2va` take without losing your actor or set** | Rewriting `"golden-hour sunset"` to `"midnight rainstorm"` at step `0` throws away the actor, hallucinates new hats/clothes, and deletes background props/boats. | `first_step = 1, scale = 1.8, streams = "independent", audio_scale = 1.0..1.2, last_step = 3`: step `0` seeds the original actor and blocking before steps `1..3` pour storm light onto video without clipping audio above `0 dBFS`. |
| **4. Continuous "in-between" dialing (`0 < scale < 1`) where words overshoot** | Prompt words act like on/off switches: aging a 25-year-old `<Picture 1>` portrait in text either gets ignored by `<Picture 1>`'s young pixels or snaps straight to 75 years old and changes the room. | `scale = 0.45, first_step = 1`: interpolates 45% of the way toward the aged prompt on the exact same hair strands, lip sync, and background. |
| **5. Frame-accurate `t2va` action timing from short sound islands without synthetic `.wav` bleed** | Prompt prose cannot pin impacts to exact frames, while running `MiniMax H3 Guide Latent` audio islands on `BasicGuider` copies the synthetic `.wav` waveform into the output soundtrack (`r = +0.86`) or mutes inter-island ambience (`-94 dBFS`). | Wire one prompt (`1x` Qwen encode) as `0`-keyframe `original` vs. `K`-island `edited`, with `streams = "video_only", scale = 1.6, last_step = 3` + `MiniMax H3 Release References(audio_end_percent = 0.60)`: locks exact visual impacts (`[24, 57, 96]`) in `10` DiT calls while `x_audio` stays `100%` native (`r = -0.08`, `-30.8 dBFS` street ambience). |

*(Bonus vs. 1D Control Vectors: when swapping a multi-garment outfit with distinct colors per piece—such as a rust-orange blouse plus blue denim jeans—Edit Guidance keeps every word unpooled through cross-attention so colors never blur across garments, whereas a single pooled vector loses color-to-garment binding.)*

---

## 2. Quick start in ComfyUI

Open any ready-to-run workflow:

- **Text-to-video (`t2va`)**: [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json)
- **Reference image + text-to-video (`ref2va`)**: [`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json)
- **Timeline sound-island action choreography (`t2va` `0`-vs-`K` `cond_audio`, `streams = "video_only"`)**: [`workflows/h3_text_to_video_pika_sound_guide_api.json`](../../workflows/h3_text_to_video_pika_sound_guide_api.json) *(run `<comfy-venv-python> bench/make_sound_guide_wavs.py` once if `$H3_INPUT_DIR` does not yet have `h3_guide_hop_200ms.wav` and `h3_guide_yank_slam_300ms.wav`; full design in [`docs/research/2026-11-12_guide_sound_islands_and_kinematic_dpad.md`](../research/2026-11-12_guide_sound_islands_and_kinematic_dpad.md))*

You only touch three places:

1. **The Prompt text box** (`PrimitiveStringMultiline`): Write your base prompt once here. It feeds both the original branch and `MiniMax H3 Prompt Edit` so they never fall out of sync.
2. **`MiniMax H3 Prompt Edit`**: Write one `old wording -> new wording` line per edit (use `retention_analysis: old -> new` if you also need to relax a `ref2va` preservation clause).
3. **`MiniMax H3 Edit Guider`** (replaces `BasicGuider` on `SamplerCustomAdvanced`): Set `scale`, `first_step`, `last_step`, and—when steering video and audio differently—`streams` and `audio_scale`.

---

## 3. Dials and edit syntax

### `MiniMax H3 Edit Guider` dials

| Dial | Default | Plain-English guide |
|---|---|---|
| `scale` | `1.5` (`0.0 .. 4.0`) | **How strongly to apply the edit.** `0.0` = original prompt; `0.3..0.6` = subtle partial dial; `1.0` = plain edited prompt; `1.5..2.0` = extra push to overcome visual damping or step-`0` layout locks. Above `~2.2`, high-gain extrapolation can add unwanted color burn or audio static/clipping. |
| `first_step` | `0` | **Which sampling step turns the edit on** (`0`-indexed; 6-step Pika runs steps `0..5`). Use `0` when changing silhouettes from the start; `1` to keep step `0`'s actor/framing while changing weather, age, or off-screen SFX; `2` to lock the **entire visual take** on `t2va` while re-dubbing foley and music. |
| `last_step` | `1000` (all steps) | **Which step stops the 2-pass boost.** Steps after `last_step` run a fast 1-pass `[edited]` step (`scale = 1.0`) on active streams while keeping any `0.0`-locked stream on `[original]`, cutting pass count (e.g., `last_step = 2` or `3` runs `9–10` passes instead of `12`) and preventing late high-SNR steps from clipping audio above `0 dBFS`. |
| `streams` | `"both"` | **Which streams receive edit guidance (`0` extra DiT passes).** `"both"` steers video and audio at `scale`; `"audio_only"` locks video to `[original]` (`0.0`) and steers audio at `scale`; `"video_only"` steers video at `scale` and locks audio to `[original]` (`0.0`); `"independent"` steers video at `scale` and audio at `audio_scale`. |
| `audio_scale` | `1.0` (`0.0 .. 4.0`) | **Audio edit scale when `streams = "independent"`** (ignored in other modes). Keep at `1.0` when pushing visual `scale >= 1.65` so audio adopts your edited `soundscape:` cleanly without `+2.7..+3.0 dBFS` transient clipping, or raise to `1.5..2.0` to punch off-screen SFX past calm visuals. |
| `align_positions` *(Advanced)* | `False` | Advanced RoPE origin toggle for prompts of unequal token length. Leave `False`. |

### `MiniMax H3 Prompt Edit` rules & shortcuts

- **Exact unique match (`old wording -> new wording`)**: Put one edit per line. `old wording` must appear **once** in the prompt (or inside the named section). Add two or three neighboring words if a word appears multiple times.
- **Section prefixes**: Aim an edit at a single section when needed:
  - `description: old -> new`
  - `soundscape: old -> new` (aliases `audio:`, `foley:`, `sound:`)
  - `music: old -> new` (aliases `score:`, `non_diegetic_music:`)
  - `subject_definitions: old -> new` (alias `subjects:`)
  - `summary: old -> new`
  - `retention_analysis: old -> new` (alias `retention:` — e.g. `retention_analysis: fully_preserved - face, hair, and clothing are retained -> partially_preserved - face and hair are retained; clothing changes`)
- **Protected tags**: `old -> new` intentionally refuses to edit `<d>...</d>` spoken dialogue, `(S1)` speaker markers, `<Subject 1>` / `<Picture 1>` tags, `[Shot 1]` headers, or section titles. Match the plain words *adjacent* to tags instead:
  ```text
  subject_definitions: shoulder-length blonde hair -> shoulder-length blonde hair, wearing a dark navy wool blazer
  ```

---

## 4. Verified examples (where a plain prompt edit fails)

Every example below was rendered 3 ways (`Original` vs. `Plain Prompt Edit` vs. `Edit Guidance`) on seed `20261112` under `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/` (with 3-take watch-and-listen reels in `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/stacked/` and frame sheets in `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/sheets/`).

### Example 1 — Re-dubbing foley and music onto a `t2va` shot without losing the take
- **What happens if you just edit the prompt (`scale = 1.0`)?** Editing *only* `overall_soundscape:` and `non_diegetic_music:` (touching zero visual words) shifts token positions at step `0` and **scrambles the entire video** (`L1 = 54.3` vs. `Original`): you get a different actor standing sideways in profile, a rotated espresso machine, and no chalkboard menu—plus clipped `+2.5 dBFS` audio.
- **What Edit Guidance does (`first_step = 2, scale = 2.0, last_step = 5`)**: Because Pika locks video layout much earlier (`shift_video = 12.0`) than audio (`shift_audio = 3.0`), steps `0..1` preserve the **exact same barista, olive shirt, chalkboard, espresso machine, and cup slide** (`L1 = 18.2`), while steps `2..5` layer crisp steam/cup foley (`+6.3 dB` in `1.2–7 kHz`) and an upbeat acoustic swing-jazz bass/brush-snare groove (`+5.2 dB` overall RMS over `Original`, clean `0.19 dBFS` peak) underneath his intact spoken line.
- **Renders**: `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/stacked/reel_1_t2va_audio_redub_espresso.mp4` | `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/sheets/sheet_exp5_espresso_foley_lock.jpg`
- **Workflow**: [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json)
- **Settings**: `scale = 2.0`, `first_step = 2`, `last_step = 5` *(or set `streams = "audio_only"` from `first_step = 0` or `1` to hold the video prediction strictly on `[original]` across every step; use `last_step = 3` or `4` if a louder foley edit approaches `0 dBFS`)*.
- <details><summary><strong>Base prompt</strong> (paste into the Prompt text box)</summary>

  ```text
  integrated_multimodal_description:
  [Shot 1] Live-action, cinematic, shallow depth of field. A medium close-up frames an Italian barista in his thirties in a rolled-sleeve olive work shirt standing behind a gleaming stainless-steel espresso machine in a warm neighborhood cafe. He sets a small white porcelain espresso cup onto a saucer on the wooden counter, slides it slightly forward, looks up at the camera with a confident grin, and in a clear, friendly tenor (S1) says: <d>[English] Try one sip before you put any sugar in it.</d> His lips close and he rests both palms on the edge of the counter as the camera holds a steady medium close-up.
  overall_soundscape:
  Quiet, hushed early-morning room tone and one tiny soft clink of porcelain.

  non_diegetic_music:
  N/A
  ```
  </details>
- **`edits`**:
  ```text
  soundscape: Quiet, hushed early-morning room tone and one tiny soft clink of porcelain. -> Loud high-pressure espresso steam wand screeching and frothing milk in a steel pitcher, sharp metallic portafilter knock-box bang, rattling saucers, and lively bustling cafe clatter.
  music: N/A -> Upbeat acoustic swing jazz trio with a prominent plucked upright walking bassline, crisp brushed snare groove, and warm acoustic guitar comping.
  ```

### Example 2 — Punching off-screen sound effects through a calm, sunny visual scene (`ref2va` & `t2va`)
- **What happens if you just edit the prompt (`scale = 1.0`)?** In a peaceful, sun-drenched apartment (`woman_blonde.jpg` holding a book and whispering *"Wait, someone is right outside"*), the calm sunlit visual tokens dampen a plain `soundscape:` edit: the passing police siren in the tail drops to **`-45.9 dBFS`** (barely a faint distant murmur) while her head turn drifts (`L1 = 25.0`). Writing `"police sirens"` into the visual description instead would add flashing lights and ruin the sunny room.
- **What Edit Guidance does (`first_step = 1, scale = 1.65, last_step = 3`)**: Locks the warm sunbeam and `<Picture 1>` pose (`L1 = 14.1`) while punching urgent heavy door-pounding and a loud wailing two-tone siren right past the sunny visual damping—**`+7.3 dB` louder overall (`-17.3` vs. `-24.5 dBFS`)** and **`+13.3 dB` louder in the siren tail (`-32.6` vs. `-45.9 dBFS`)**. *(Same pattern verified on [`prompt_bank/t2va_night_porter.txt`](../../prompt_bank/t2va_night_porter.txt), where `scale = 1.5, last_step = 2` boosts an off-screen elevator klaxon and sub-bass rumble by `+6.2 dB` RMS and `+8.5 dB` `< 250 Hz` over `scale = 1.0` without touching the calm lift visuals.)*
- **Renders**: `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/stacked/reel_2_ref2va_offscreen_door_siren_sfx.mp4` | `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/sheets/sheet_exp8_sunny_room_door_siren.jpg`
- **Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json) (`LoadImage = woman_blonde.jpg`)
- **Settings**: `scale = 1.65`, `first_step = 1`, `last_step = 3`, `streams = "audio_only"` *(or `streams = "both"`; stick to structured/pitched SFX like sirens, bells, knocks, horns, or alarms around `scale = 1.5..1.7`—boosting broadband rain/wind past `2.0` sounds like static hiss)*.
- <details><summary><strong>Base prompt</strong> (paste into the Prompt text box)</summary>

  ```text
  subject_definitions:
  <Subject 1> is the young blonde woman in <Picture 1>, with long wavy center-parted blonde hair, blue eyes, light freckles, and a black chunky knit sweater.

  summary:
  [reference generation] <Subject 1> stands quietly in a peaceful sunlit living room, pauses to listen, and whispers one short line in a single continuous shot.

  retention_analysis:
  <Subject 1> (appears in [Shot 1]): fully_preserved - her face, long wavy blonde hair, blue eyes, freckles, and black chunky knit sweater are retained.

  detailed_description:
  The target video is live-action and cinematic, bathed in calm, warm golden afternoon sunlight with soft shallow depth of field.
  [Shot 1] A medium close-up in a serene, sun-drenched apartment living room finds <Subject 1> standing still beside a warm oak bookshelf, holding a closed hardcover book against her sweater. Bright golden sunbeams rest steadily across the cream wall behind her. Her eyes shift alertly toward the hallway off frame, and <Subject 1> in a quiet, hushed voice (S1) says: <d>[English] Wait, someone is right outside.</d> Her lips close and she stays completely still, listening intently in the calm sunlit room until the final frame.

  overall_soundscape:
  Very quiet, still afternoon apartment room tone and a faint soft rustle of cloth.

  non_diegetic_music:
  N/A
  ```
  </details>
- **`edits`**:
  ```text
  soundscape: Very quiet, still afternoon apartment room tone and a faint soft rustle of cloth. -> Loud urgent heavy fist knocking pounding repeatedly on a solid wooden door, accompanied by a loud wailing two-tone police siren passing on the street outside.
  ```

### Example 3 — Changing weather, lighting, and storm audio on a `t2va` shot without losing your actor or set
- **What happens if you just edit the prompt (`scale = 1.0`)?** Changing `"calm golden-hour sunset"` to `"fierce midnight rainstorm"` from step `0` completely re-rolls the scene (`L1 = 67.9`): it replaces the actor, hallucinates a shiny black fedora hat onto his head, and erases the white cabin cruiser boat and harbor hills behind him.
- **What Edit Guidance does (`first_step = 1, scale = 1.85, last_step = 3`)**: Step `0` runs only the original golden-hour prompt, seeding the exact bareheaded bearded harbor watchman, his navy peacoat, the wooden pier bollard, and the white cabin cruiser boat (`L1 = 37.3`). Steps `1..3` then plaster wet hair across his forehead, pelt the pier with falling rain, light up the boat's warm cabin windows against overcast storm grey, and add rolling thunder and downpour (`+25.0 dB` `< 250 Hz` thunder over `Original`, `+14.9 dB` over `Plain Prompt Edit`) under his spoken line. *(Note: `first_step = 2` is too late for sky/weather changes—by step `2` the golden-hour sky is already locked in.)*
- **Renders**: `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/stacked/reel_3_t2va_weather_lock_harbor.mp4` | `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/sheets/sheet_exp3_harbor_weather_lock_f1.jpg`
- **Workflow**: [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json)
- **Settings**: `scale = 1.85`, `first_step = 1`, `last_step = 3` *(add `streams = "independent", audio_scale = 1.0` whenever a high visual `scale >= 1.65` pushes storm/thunder transients near `0 dBFS`)*.
- <details><summary><strong>Base prompt</strong> (paste into the Prompt text box)</summary>

  ```text
  integrated_multimodal_description:
  [Shot 1] Live-action, cinematic 35mm, shallow depth of field. In calm golden-hour sunset with warm amber sunlight glinting off glassy harbour water, a medium close-up frames a weathered harbourmaster in his fifties with a salt-and-pepper beard and a dry navy wool peacoat standing beside a wooden pier bollard. He rests one gloved hand on the timber rail, turns his head toward the camera, and with a low, gravelly baritone (S1) says: <d>[English] The last ferry cleared the channel twenty minutes ago.</d> His lips close and he keeps a steady gaze toward the lens as the camera holds a gentle handheld medium close-up.
  overall_soundscape:
  Calm harbour water lapping softly against wooden pilings, distant seagull calls, and a light warm evening breeze.

  non_diegetic_music:
  N/A
  ```
  </details>
- **`edits`**:
  ```text
  description: calm golden-hour sunset with warm amber sunlight glinting off glassy harbour water -> a fierce midnight rainstorm with sheets of heavy pouring rain, cold blue storm light, and rainwater streaming down his face and soaked coat
  soundscape: Calm harbour water lapping softly against wooden pilings, distant seagull calls, and a light warm evening breeze. -> Torrential rain hammering the wooden pier, howling gale gusts, splashing downpour, and a deep rolling thunderclap across the water.
  ```

### Example 4 — Continuous "in-between" subtlety dial (`scale = 0.45` vs. `1.0` prompt overshoot)
- **What happens if you just edit the prompt (`scale = 1.0`)?** Prompt words are discrete switches: writing an aged character description on a `ref2va` portrait (`woman_blonde.jpg`) snaps straight to a **75-year-old elderly woman with white hair** and even swaps out the background window (`L1 = 60.8`), whereas milder phrasing (`"late thirties"`) gets overpowered by `<Picture 1>`'s smooth 25-year-old reference pixels.
- **What Edit Guidance does (`scale = 0.45, first_step = 1, last_step = 4`)**: Acts like a continuous fader—dialing `<Picture 1>` cleanly to **~40 years old** (natural forehead creases, subtle crow's-feet, and mid-life facial lines) while keeping every blonde hair strand, head tilt, lip-sync frame, and background bokeh circle locked to the original take (`L1 = 14.4`).
- **Renders**: `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/stacked/reel_4_ref2va_subtle_aging_dial.mp4` | `$H3_OUTPUT_DIR/Video/edit_guidance_showcase/sheets/zoom_exp7_subtle_aging_dial_f92.jpg`
- **Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json) (`LoadImage = woman_blonde.jpg`)
- **Settings**: `scale = 0.45`, `first_step = 1`, `last_step = 4` *(sweep `scale` between `0.25` and `0.70` to pick the exact age, expression intensity, or style blend you want)*.
- <details><summary><strong>Base prompt</strong> (paste into the Prompt text box)</summary>

  ```text
  subject_definitions:
  <Subject 1> is the young twenty-five-year-old woman in <Picture 1>, with smooth youthful skin, long wavy blonde hair, bright blue eyes, light freckles, and a black chunky knit sweater.

  summary:
  [reference generation] <Subject 1> looks into the camera in soft natural window light and speaks one reflective line in a single continuous shot.

  retention_analysis:
  <Subject 1> (appears in [Shot 1]): fully_preserved - her facial structure, smooth youthful skin, wavy hair, blue eyes, and black chunky knit sweater are retained.

  detailed_description:
  The target video is live-action and cinematic, shot on 35mm portrait lens in soft natural window light.
  [Shot 1] A steady medium close-up frames <Subject 1>, the young twenty-five-year-old woman with smooth youthful skin and long wavy blonde hair in a black chunky knit sweater, seated beside a softly blurred studio window. She looks directly into the lens with a calm, thoughtful expression, and <Subject 1> in a warm, steady voice (S1) says: <d>[English] Funny how fast ten years goes by when you are not looking.</d> Her lips close into a gentle, knowing smile as the camera holds steady.

  overall_soundscape:
  Quiet indoor room tone and a soft breath.

  non_diegetic_music:
  N/A
  ```
  </details>
- **`edits`**:
  ```text
  subject_definitions: young twenty-five-year-old woman -> seventy-five-year-old elderly woman
  subject_definitions: smooth youthful skin, long wavy blonde hair -> deeply wrinkled weathered skin, creased forehead and crow's-feet, and long wavy silver-grey hair
  retention_analysis: fully_preserved - her facial structure, smooth youthful skin, wavy hair -> partially_preserved - her core bone structure and blue eyes are recognizable, aged into deeply wrinkled seventy-five-year-old skin and silver-grey hair
  description: young twenty-five-year-old woman with smooth youthful skin and long wavy blonde hair -> seventy-five-year-old elderly woman with deeply wrinkled weathered skin, pronounced crow's-feet, and long wavy silver-grey hair
  ```

### Combining Edit Guidance with a Control Vector
`MiniMax H3 Edit Guider` (on `SamplerCustomAdvanced`) and `MiniMax H3 Steer` (on the `MODEL` wire upstream) compose on one graph: `MiniMax H3 Steer` applies its `1x` post-refiner shift to both passes so the control vector cancels out of `(edited - original)` and stays at `1.0x`, while `scale` amplifies only your prompt edit.
- *Cost & `@prompt_subject` notes*: Keep `MiniMax H3 Steer` on `streams = "both"` when pairing with `EditGuider`—setting `streams = "video_only"` or `"audio_only"` splits each guider pass in two (`4` DiT passes per guided step). Also note that a `MiniMax H3 Word Vector` using `against_words = "@prompt_subject"` resolves against `original` on pass `0` and `edited` on pass `1` separately.

---

## 5. Under the hood (technical reference)

For readers inspecting [`edit_guidance.py`](../../edit_guidance.py), [`h3_prompt_edit.py`](../../h3_prompt_edit.py), or [`h3_rules.py`](../../h3_rules.py):

- **`CFGGuider` subclass (`EditGuider`)**: Subclasses `comfy.samplers.CFGGuider` without patching ComfyUI core (following `Guider_DualCFG` in `comfy_extras/nodes_custom_sampler.py`). Inside `comfy/samplers.py` -> `CONST.calculate_denoised`, `calc_cond_batch` returns predicted clean latents `x0_hat = x - sigma * v`, so `out[0] + scale * (out[1] - out[0])` extrapolates `(x0_orig, x0_edit)` linearly (algebraically identical to extrapolating velocity `v` at matched `(x, sigma)`, with H3's shifted audio input carry cancelling cleanly in `out[1] - out[0]`). When `streams` is `"audio_only"`, `"video_only"`, or `"independent"` (`eff_v != eff_a`), `EditGuider.predict_noise` splits `(v, a)` on both `NestedTensor` mocks and live `CFGGuider.sample` `pack_latents` 3D `[B, 1, C_flat]` tensors (`comfy.utils.unpack_latents` / `pack_latents` via `inner_model.latent_shapes`, `e0a5bb35`) and recombines `(v0 + eff_v * (v1 - v0), a0 + eff_a * (a1 - a0))` from the **same two DiT passes (`0` extra passes)**.
- **Keyframe-differential `0`-vs-`K` guidance (`_release_to_original_after`)**: `check_same_but_text` (`h3_rules.py`) permits `original` and `edited` to share identical text tokens when their `minimax_keyframes` differ (e.g., `0` keyframes vs. `K` timeline `cond_audio` islands from `MiniMaxH3GuideLatent`). When `original` has `0` keyframes and `edited` has `K > 0`, `_release_to_original_after = True` routes steps `> last_step` to 1-pass `[original]` once `MiniMaxH3ReleaseReferences(audio_end_percent)` has already dropped `cond_audio` (`10` DiT calls on 6-step Pika at `audio_end_percent = 0.60, last_step = 3`).
- **Why `first_step = 1` vs. `first_step = 2` splits weather from audio on Pika `sample_lcm`**:
  - Under `sample_lcm` (`comfy/k_diffusion/sampling.py`) on `sigmas = [1.0, 0.972973, 0.923077, 0.80, 0.571429, 0.333333, 0.0]`, step `0` deposits `2.70%` `x0_hat_0` + `97.30%` fresh noise (`first_step = 1` nudges actor/boat/room layout while leaving `97.3%` variance open for sky/weather re-lighting).
  - By step `2` (`t = 0.50`), `shift_video = 12.0` is at `sigma_video = 0.923` (locking visual layout and sky color), whereas `shift_audio = 3.0` is at `sigma_audio = 0.75`—which is why `first_step = 2` locks the video take while still completely re-dubbing foley and music (`Example 1`).
  - Conversely, `last_step = 2` or `3` keeps the edit locked because steps `> last_step` run **1-pass `[edited]`** (`scale = 1.0`), avoiding high-SNR color burn and `> 0 dBFS` audio clipping on steps `4..5`.
- **Token-length RoPE offset (`PackedLayout`)**: `PackedLayout` starts media rows at 3D RoPE `t = text_len`. Unchanged text tokens *after* the edit keep identical relative RoPE distance to media (`align_positions = False`), while tokens *before* the edit shift relative distance by `L_edit - L_orig`—contributing to collateral drift above `scale ~ 2.2`.
- **Single-pass media encoding & `h3_edit_pass`**:
  - Reference workflows encode vision tokens and VAE latents **once** via [`MiniMaxH3EncodeReferences`](../../reference_encode.py) and feed two [`MiniMaxH3PromptOnReferences`](../../reference_encode.py) nodes; guide-latent workflows encode guide media **once** via [`MiniMaxH3GuideLatent`](../../guide_latent.py); [`MiniMaxH3Preflight`](../../preflight.py) prices peak sequence length across both branches.
  - `_prepare` tags each branch's `minimax_payload` with `h3_edit_pass = "original"` or `"edited"`, keeping layout-less `P-H3-Image` T2I passes at `B = 1` (video `extra_conds` already carry distinct `PackedLayout` objects) and letting `control_vector_nodes.py` identify each pass.
- **Verification & comparison tools**:
  ```bash
  <comfy-venv-python> bench/check_edit_guidance.py
  <comfy-venv-python> bench/edit_guidance_arms.py --help
  ```
