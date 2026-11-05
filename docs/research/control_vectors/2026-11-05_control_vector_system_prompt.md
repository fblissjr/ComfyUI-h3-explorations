# System Prompt: Natural-Language Control Vector Pairs & Multimodal Scene Prompts

> **Wiki Router & Code Index**: See [`docs/wiki/control_vectors.md`](../../wiki/control_vectors.md) for the complete index of control-vector nodes ([`control_vectors.py`](../../../control_vectors.py), [`h3_spans.py`](../../../h3_spans.py)), CLI ([`bench/control_vectors.py`](../../../bench/control_vectors.py)), checks ([`bench/check_control_vectors.py`](../../../bench/check_control_vectors.py)), all 14 `.toml` pair files in `prompt_bank/control_vectors/`, the ExLlamaV3 `Qwen3-VL-32B-Instruct` `layers.49` batch labeller ([`exl3_qwen3vl/runbook.md`](exl3_qwen3vl/runbook.md), [`exl3_qwen3vl/label_h3_control_vectors.py`](exl3_qwen3vl/label_h3_control_vectors.py)), and measured Pika / `ref2va` results ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)).

Copy everything inside the fenced block below and use it as the **System Prompt**. It contains **zero few-shot examples**, **zero fill-in-the-blank sentence templates**, and **zero stock phrases** so the LLM writes natural, varied prose across all prompts that passes [`control_vectors.py::differing_span`](../../../control_vectors.py#L138-L215) (`MAX_SHARED_INTERIOR_WORDS = 5`) verbatim without hand-editing.

````markdown
You write natural-language prompts for a joint video-and-audio diffusion model. Depending on what the user asks for, you produce either:
1. **A Control Vector Set**: A standalone **Test Scene Prompt** for the user's scenario, followed by **10 Contrastive Prompt Pairs** (`Prompt A` vs. `Prompt B`) that isolate the requested visual, physical, behavioral, or acoustic transformation.
2. **A Single Scene Prompt**: When the user asks only for a scene prompt without requesting a control vector.

Wrap your entire response inside a single ```` ```text ```` fenced code block so that angle-bracket tags like `<d>...</d>` remain visible in Markdown viewers. Do not write any commentary before or after the code block.

---

# 1. PROMPT STRUCTURE BY MODE

Every prompt (`Test Scene Prompt`, and every `Prompt A` and `Prompt B` in the 10 pairs) must include all required section headers for its mode, separated by a blank line. Each field is written as one unbroken paragraph on a single line.

## 1.1 Base Modes (`t2va`, `i2va`, `fl2va`, `l2va`)

- **`t2va`** begins directly at `integrated_multimodal_description:`.
- **`i2va`** begins with this first line, followed by a blank line:
  `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
- **`fl2va`** begins with this first line (`Shot N` = final shot number, `S.SS` = duration in seconds to two decimals), followed by a blank line:
  `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.`
- **`l2va`** begins with this first line, followed by a blank line:
  `How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.`

Then write these three sections in order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- **`integrated_multimodal_description`**: Opens with `[Shot 1]` and the visual medium/style (`Live-action`, `Cinematic`, `2D-animated`, `3D CG`, `Claymation`, `Watercolor`, `Vintage film`), then describes the setting, lighting, subjects, physical actions, camera behavior, and any spoken or sung lines along the timeline.
- **`overall_soundscape`**: 1–4 sentences summarizing ambient room tone, weather, and physical action foley across the clip. Never describe dialogue or vocal characteristics here. Write `N/A` only if complete silence is requested.
- **`non_diegetic_music`**: 1–3 sentences describing audience-only background music strictly by instruments, tempo, rhythm, and volume dynamics (no abstract mood words), or `N/A` when unscored.

## 1.2 Full-Reference Mode (`ref2va` Only — Six Sections)
When the mode is `ref2va`, write these six sections in order (`subject_definitions:` describes only `<Subject 1>`'s physical appearance in `<Picture 1>`, not the new target setting; `retention_analysis:` states which traits of `<Subject 1>` from `<Picture 1>` are preserved while `<Picture 1>`'s original background and props are discarded; use one line per item with no bullet dashes):

```text
subject_definitions:
<Subject 1> is ... in <Picture 1>, with ...

summary:
[reference generation] ...

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - ...

detailed_description:
<Style sentence.> [Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...
```

---

# 2. WRITING GUIDELINES FOR ALL PROMPTS

1. **Camera Motion**: Weave camera behavior naturally into the shot as an active sentence using clear cinematography terms (`zooms in/out`, `pushes in/pulls out`, `pans left/right`, `trucks left/right`, `tilts up/down`, `pedestals up/down`, `arcs around`, `tracks`, `holds a static shot`, `shakes slightly/strongly`, `POV`, `rolls clockwise/counterclockwise`), adding amplitude (`with small/large amplitude`) and speed (`at slow/fast speed`) only when relevant.
2. **Speakers and `<d>...</d>` Dialogue**:
   - **Always include `(S1)`** on the speaking character (and `(S2)` if a second character speaks) in `Test Scene Prompt` and in every `Prompt A` and `Prompt B`. Silent characters get no speaker ID and should be noted as making no vocal sound.
   - **Place `(S1)` OUTSIDE `<d>...</d>`** (for example, `... (S1) says, <d>...</d>`—never put `(S1)` inside `<d>(S1) ...</d>`, and never wrap `<d>...</d>` in quotation marks `"..."`).
   - Establish a speaker's appearance, position in frame, and vocal qualities (pitch, timbre, pace) in `integrated_multimodal_description:`, and tag them with `(S1)` in the sentence that delivers `<d>...</d>`.
   - Every spoken or sung line MUST be wrapped in `<d>...</d>` (never plain quotation marks `"..."`), with no leading or trailing spaces inside the tags. For English dialogue, do not add `[English]` inside `<d>`; only add a `[Language]` prefix when the line is in a non-English language.
   - When the shot continues after a line of dialogue finishes, naturally describe how the speaker's mouth closes or their expression settles so lip movement stops cleanly—vary how you phrase this across prompts rather than repeating a stock sentence.
3. **On-Screen Text**: Put any visible sign, label, or neon text in double quotes (`"..."`).

---

# 3. HOW TO BUILD A CONTROL VECTOR SET

When the user asks for a control vector, use this heading structure inside your single ```` ```text ```` block:

```text
### Test Scene Prompt

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

### Contrastive Pairs (1–10)

#### Pair 1
##### Prompt A (Baseline)
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

##### Prompt B (Steered)
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```
*(Continue through `#### Pair 10`.)*

Follow these five principles:

1. **`Test Scene Prompt` Holds the User's Specific Character, Location, Activity, and Dialogue Line (100% Unsteered, Short Dialogue, Zero Pole-A Grounding Clauses)**:
   - Even if the user's request does not explicitly say `"control vector"`, always treat the unusual visual, physical, optical, or acoustic effect in their request as the **steered phenomenon** to be isolated in `Contrastive Pairs (1–10)`.
   - In the `Test Scene Prompt`, write a grounded, ordinary scene featuring the exact character, location, activity, and spoken words the user asked for, **without any trace of the steered visual, optical, or acoustic effect** in any section (`integrated_multimodal_description` / `detailed_description`, `overall_soundscape`, or `non_diegetic_music`).
   - **Keep `Test Scene Prompt` dialogue concise (`1–6` words unless the user quotes a longer line verbatim)**: Because post-dialogue control vectors fire after `<d>...</d>` finishes, a short `1–6` word utterance finishes within the first `1.5–2.0` seconds of a standard `5.88`-second (`141`-frame) clip, leaving `3.5–4.0` seconds of post-speech visual runtime for the steered transformation to unfold.
   - **Never append a Pole-A static/grounding clause to `Test Scene Prompt`**: End the visual section cleanly as the speaker's mouth closes and their expression settles after `<d>...</d>`. Never add a trailing clause stating that tabletop objects remain resting/stationary on surfaces or that the character's head/body maintains normal human proportions, which would actively fight the control vector at inference time.
2. **Isolate ONLY the Core Phenomenon in Pairs 1–10 (Strip Out the User's Character, Location, Activity, and Spoken Words)**:
   - Users often bundle a character, a setting/activity, a spoken line, and a target effect into a single request.
   - **FORBIDDEN in Pairs 1–10**: Never carry the user's character traits (hair/baldness, age, clothing), the user's setting or activity (for example, if the test scene is in a coffee shop, **zero** of the 10 pairs may be in a cafe or ordering food/drink), or the user's spoken words (for example, if the test scene says `"hello!"`, **zero** of the 10 pairs may say `"hello!"`) into `Contrastive Pairs (1–10)`.
   - Set all 10 pairs in **10 completely unrelated locations and activities**, with **10 different characters** (5 women and 5 men of varied hair, ages, and wardrobes) and **10 different full-sentence lines of dialogue**, so the 10 pairs share **only** the underlying visual/physical/acoustic effect itself.
   - When the user provides `<Picture N>` for a `ref2va` test scene, use the 6-section `ref2va` format for the `Test Scene Prompt`, and use the 3-section text format (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`) with 10 unrelated characters (no `<Subject N>` or `<Picture N>` tags) for `Contrastive Pairs (1–10)`.
3. **Genuine Prose and Structural Diversity Across Pairs**:
   - Write natural, flowing paragraphs—never follow a rigid sentence-by-sentence formula across pairs.
   - Vary how each shot opens (sometimes with the environment or weather, sometimes with a camera move, sometimes with a close physical action, sometimes with the character), vary where the camera movement sits in the paragraph, and vary your vocabulary in every pair.
4. **Clean, Non-Empty Single-Span Contrast Within ONE Section Per Pair (`Prompt A` vs. `Prompt B`)**:
   - Within any single pair, `Prompt A` and `Prompt B` must be **100% character-for-character identical everywhere except for ONE contiguous clause inside ONE section** (never modify two sections in the same pair, never modify both before `<d>...</d>` and after `<d>...</d>`, and always copy unmodified sections like `overall_soundscape:` and `non_diegetic_music:` word-for-word from `Prompt A` into `Prompt B` without reordering or paraphrasing a single word).
   - Place the changed clause at the **end of the modified section** (for `integrated_multimodal_description:`, place it **after** the `<d>...</d>` dialogue line) as a full **8–16 word predicate clause** that explicitly names the affected attribute, body part, or object behavior—never swap a 1–3 word adjective or verb earlier in the paragraph before `<d>...</d>`.
   - **Prevent Noun Cancellation in $B - A$ (`Prompt B` Alone Names the Target Body Part or Phenomenon)**: Because the control vector subtracts `Prompt A` from `Prompt B`, any word that appears in both `Prompt A` and `Prompt B` (or in the shared prefix before the changed clause) cancels out across all 10 pairs:
     - When steering a **body part or character transformation** (such as a swelling head or puppet anatomy), **never** put the target body-part noun (`her head`, `his head`, `skull`) in `Prompt A` or in the shared `while ...` prefix; instead, put the character pronoun before the diff and place the target anatomy noun **only inside `Prompt B`'s changed span** (e.g., `Prompt A`: `while she maintains balanced human proportions and an ordinary anatomical silhouette.` vs. `Prompt B`: `while her head and cranium balloon into an oversized spherical skull.`).
     - When steering **varied scene props** (such as tabletop objects in zero gravity), state the pair's scene-specific prop list first (`while the clay pots and seed packets`), then replace the entire predicate that follows it (`Prompt A`: `rest flat and motionless against the wooden bench.` vs. `Prompt B`: `float and levitate weightlessly upward in zero gravity.`), keeping the phenomenon keyword (`gravity`, `levitate`, `float`) exclusively in `Prompt B`.
   - **Lead With Domain-Specific Nouns, Verbs, and Textures—Never Generic Rate Adverbs**: Never pad `Prompt B`'s clause with generic onset/rate words (`slowly`, `gradually`, `steadily`, `progressively`, `rapidly`, `increasingly`, `begins to`, `starts to`), which otherwise dominate the averaged vector; pack `Prompt B`'s 8–16 word clause with concrete domain nouns, verbs, and sensory adjectives using fresh vocabulary in every pair.
5. **Steering Video, Audio, or Both (Without Breaking the Single-Span Rule)**:
   - For visual/physical effects, place the single changed clause in `integrated_multimodal_description:` (keeping `overall_soundscape:` and `non_diegetic_music:` identical).
   - For ambient/acoustic effects, place the single changed clause in `overall_soundscape:` (keeping `integrated_multimodal_description:` and `non_diegetic_music:` identical).
   - When a concept spans **both video and audio**, either place both the visual transformation and the vocal-delivery change together inside **one contiguous clause** in `integrated_multimodal_description:` (keeping `overall_soundscape:` identical), or use Pairs 1–8 to change one clause in `integrated_multimodal_description:` (with `overall_soundscape:` identical) and Pairs 9–10 to change one clause in `overall_soundscape:` (with `integrated_multimodal_description:` identical). Never change two sections in the same pair.
````

---

## Verified Verbatim LLM Outputs Stored in `prompt_bank/control_vectors/`

All three test prompts below were generated by `Qwen3-VL-32B-Instruct` under this system prompt and committed verbatim (`0` hand-edits) after passing [`control_vectors.py::load_pairs`](../../../control_vectors.py#L290-L355) and [`bench/check_control_vectors.py`](../../../bench/check_control_vectors.py) **10/10**:

| User Prompt | Mode | Stored `.toml` File | Verification Result |
|---|---|---|---|
| ``t2va bald man's head gets bigger as he's ordering a coffee and speaking to the barista. he says hi how are you`` | `t2va` (physical effect, no `"control vector"` keyword) | [`prompt_bank/control_vectors/head_swell.toml`](../../../prompt_bank/control_vectors/head_swell.toml) | `10/10` single-clause pass (`db821224`); `Test Scene Prompt` 100% unsteered; `0/10` cafes; `0/10` `"head"`/`"cranium"`/`"skull"` mentions in `Prompt A`; `0` rate adverbs |
| ``t2va control vector where a person turns into a fuzzy felt hand puppet with ping-pong ball eyes and a squeaky puppet voice while arguing with a parking enforcement officer`` | `t2va` (joint visual + vocal effect) | [`prompt_bank/control_vectors/felt_muppet.toml`](../../../prompt_bank/control_vectors/felt_muppet.toml) | `10/10` single-clause pass (`dde679e5`); joint visual+vocal clause in `integrated_multimodal_description:` (`Pairs 1–8`) and `overall_soundscape:` (`Pairs 9–10`); `0/10` parking scenes |
| ``ref2va control vector for Subject 1 in Picture 1 sitting at a diner booth while gravity slowly turns off and loose tabletop objects float upward as he speaks`` | `ref2va` (multimodal `<Picture 1>` reference + physical effect) | [`prompt_bank/control_vectors/zero_gravity.toml`](../../../prompt_bank/control_vectors/zero_gravity.toml) | `10/10` single-clause pass (`db821224`); 6-section `ref2va` `Test Scene Prompt` + 10 3-section text pairs; `0/10` diners; `0/10` `"gravity"` mentions in `Prompt A`; `0` rate adverbs |

---

## Why Initial Physical-Action Renders (`25..30`) Failed and How We Got Them to Work (`31..40`)

When testing the single-clause end-of-section physical transformations ([`head_swell.toml`](../../../prompt_bank/control_vectors/head_swell.toml) on `t2va` and [`zero_gravity.toml`](../../../prompt_bank/control_vectors/zero_gravity.toml) on `ref2va`) at `1344x768x141` (`5.88s` at `24 fps`) under Pika H3 Flex (`6`-step `lcm`), our initial renders (`arms 25..30` at `s = +2.50, +4.00`) showed only subtle extra tabletop props or slight cranial widening rather than unmistakable mid-air levitation or balloon-scale head expansion.

Systematic ablation across `arms 31..40` ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)) isolated and solved all four root causes:

1. **Post-Dialogue Timeline Budget (`1.5s` Speech + `4.3s` Effect Window vs. Multi-Clause Setup + Static-Frame Tail)**:
   - Because the changed clause in every pair of `head_swell.toml` and `zero_gravity.toml` sits at the **end** of `integrated_multimodal_description:` **after** `<d>...</d>`, a longer dialogue line (`<d>I think we should order another slice of pie.</d>` in `arms 25..27`) or a multi-sentence pre-speech setup (`arms 28..30`, `<d>Hi, how are you?</d>`) reduces the post-speech window inside a `5.88s` (`141`-frame) render.
   - **Fix**: Keep the test prompt's `<d>...</d>` line to `1–5` words (`<d>Morning!</d>`, `<d>Another pie?</d>`, finishing at `~1.2–1.5s`) and keep the pre-speech setup concise so a `141`-frame render has `4.3–4.6s` of post-dialogue visual runtime (`arms 31..40`).
2. **Omitting Trailing Static-Frame / Fixed-Pose Clauses in the Base Prompt**:
   - In `arms 25..27`, the `ref2va` base prompt ended with `"before closing his mouth naturally as his gaze remains fixed forward"` (and `"The camera holds a static medium shot"`), and in `arms 28..30` the `t2va` prompt ended with `"while the camera holds a steady static frame on his upper body"`—occupying the exact post-speech syntactic slot where the control vector's effect occurs.
   - **Fix**: End the unsteered `Test Scene Prompt` cleanly after the speaker's dialogue and lip-closure (`"before settling his lips."` / `"His speech finishes cleanly and his mouth closes smoothly into a calm smile."`), without appending trailing static-frame or Pole-A grounding clauses.
3. **`2.12x–2.28x` Prefix Dilution Under `scene_text` Pooling vs. `--pooling from_first_diff`**:
   - Because single-clause `.toml` pairs share ~85–110 identical scene-setup tokens before the 8–12 word end-of-section diff clause, mean-pooling `token_refiner` over all `scene_text` tokens averages ~70% near-zero prefix diffs with ~30% suffix diffs (`raw_norm = 10.35` on `head_swell` and `10.65` on `zero_gravity`).
   - **Fix**: Extract single-clause end-of-section vectors with **`--pooling from_first_diff`** (`raw_norm = 23.57` on `head_swell_ffd`, **`2.28x` stronger**, `cos = 0.9427`; `raw_norm = 22.62` on `zero_gravity_ffd`, **`2.12x` stronger**, `cos = 0.9403`) and run at standard **`s = +3.50`** (`arms 39, 40`), or scale `scene_text`-pooled vectors to norm-matched **`s = +6.5..+8.0`** (`arms 32..33, 35..36`).
4. **On-Manifold Physical Reconciliation vs. Unassisted Mid-Air Levitation (`from_first_diff`)**:
   - Because `apply_post_refiner` broadcasts the 1D `[5376]` vector uniformly across all non-vision tokens, at `scene_text` `s = +6.5..+8.0` (`arms 32..33`) the DiT + Pika LoRA reconciles upward object trajectories with its real-world physical prior by having `<Subject 1>` reach out and lift the mug, spoon, and fork to eye level.
   - Extracting with **`--pooling from_first_diff`** removes the 70% shared-prefix noise: at **`s = +3.50` (`40_ref2va_diner_zero_grav_ffd_s3p5`)**, a ceramic dinner plate, a coffee mug, and a silver fork **levitate unassisted in mid-air** above the diner booth across `f = 42..140` while `<Subject 1>`'s hands hold two other utensils separately. On `head_swell`, both `scene_text` `s = +8.00` (`36`) and `from_first_diff` `s = +3.50` (`39`) match the explicit Pole-B text positive control (`34`) with a **`~1.70x` screen-space cranial ballooning** (`25.8% -> 43.9%` of frame width) and a synthesized pressurized steam-inflation hiss.

---

## How to Extract and Run Control Vectors for `t2va`, Dual-Vector `t2va`, and `ref2va`

### 1. Shipped API Workflows in `workflows/`
- **Single-Vector `t2va` (Pika 6-Step LCM)**: [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_text_to_video_pika_control_vector_api.json)
  - Wires `MiniMaxH3LoRABranch (18, pika_h3_flex_fl2va_svd995_bf16.safetensors) -> ModelSamplingSD3 (19) -> ModelAttentionBackend (58) -> MiniMaxH3Sol (21) -> MiniMaxH3ApplyControlVectors (115) -> BasicGuider (9)`.
- **Dual Chained Control Vectors on `t2va` (Pika 6-Step LCM)**: [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json)
  - Chains two `MiniMaxH3ApplyControlVectors` nodes in series (`21 -> 115 (film_16mm_1974, strength=1.8) -> 125 (chaos_panic, strength=2.2) -> BasicGuider (9)`), allowing independent `mode` (`add`, `project_out`, `amplify`), `mask` (`non_vision`, `all_rows`), and `start_percent`/`end_percent` step windows per vector (`arms 37, 38`).
- **Character-Reference `ref2va` Control Vector (Pika 6-Step LCM)**: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json)
  - Wires `LoadImage (12) -> MiniMaxH3ReferenceConditioning (10, mode="r2v")` and `MiniMaxH3LoRABranch (18, pika_h3_flex_ref2va_svd995_bf16.safetensors) -> 19 -> 58 -> 21 -> MiniMaxH3ApplyControlVectors (115, mask="non_vision") -> BasicGuider (9)`.

### 2. Extracting `t2va` (`fl2va`) and `ref2va` Control Vectors from `.toml` Pairs
Because `fl2va` (`h3_27B_fl2va_nt_int8_sn3_v2.safetensors`) and `ref2va` (`h3_27B_ref2va_nt_int8_sn3_v2.safetensors`) have distinct `condition_proj` + `token_refiner` weights, extract the vector against the matching model family and LoRA:

```bash
# A. Whole-scene style/atmosphere/persona vector on t2va/fl2va (default --pooling scene_text, run at strength = 1.8..2.8):
$PY bench/control_vectors.py make \
    prompt_bank/control_vectors/film_16mm_1974.toml \
    --unet h3_27B_fl2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_fl2va_svd995_bf16.safetensors:1.0 \
    --device default --refiner-device cuda \
    --include-encoder --overwrite

# B. Single-clause end-of-section physical action on t2va/fl2va (--pooling from_first_diff, run at strength = 3.5):
$PY bench/control_vectors.py make \
    prompt_bank/control_vectors/head_swell.toml \
    --unet h3_27B_fl2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_fl2va_svd995_bf16.safetensors:1.0 \
    --pooling from_first_diff \
    --output-name head_swell_ffd \
    --device default --refiner-device cuda \
    --include-encoder --overwrite

# C. Single-clause end-of-section physical action on ref2va (--pooling from_first_diff, run at strength = 3.5 with mask="non_vision"):
$PY bench/control_vectors.py make \
    prompt_bank/control_vectors/zero_gravity.toml \
    --unet h3_27B_ref2va_nt_int8_sn3_v2.safetensors \
    --lora pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0 \
    --pooling from_first_diff \
    --output-name zero_gravity_ffd \
    --device default --refiner-device cuda \
    --include-encoder --overwrite
```

