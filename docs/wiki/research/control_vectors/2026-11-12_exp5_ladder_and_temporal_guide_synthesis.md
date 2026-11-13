# 2026-11-12: The `exp2`–`exp5` Steering Ladder, Temporal/Audio Guides, and Next Directions

This memo synthesizes two threads investigated on `2026-11-12` that converge on the same underlying mechanism—how conditioning and velocity guidance act across time, tokens, and streams inside MiniMax H3:

1. **The `exp2`–`exp5` Control Vector vs. Plain Rewrite vs. Edit Guidance Ladder** (`$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/`): judging the unjudged `exp2`, `exp3`, and `exp4` runs from [`2026-11-12_new_vectors_three_ways.md`](2026-11-12_new_vectors_three_ways.md) and [`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../../bench/results/2026-11-12_control_vector_word_share.md), isolating the four confounds in `exp3`/`exp4`, and running the 13-clip controlled **`exp5` 1:1 ladder** on `prompt_bank/t2va_offpath_tortoise.txt` (`seed = 730451892`, `1344x768`, `124` frames, `6` Pika H3 Flex steps).
2. **Temporal & Audio Event Guides (`NKD_v153` & `NKD_v173`) -> `masked_v2v` & Multimodal `EditGuider`**: forensic frame-and-audio analysis of community guide-driven H3 clips (`141` frames, `17k + 5` grid), why co-located `cond` / `cond_audio` keyframes (`comfy/ldm/minimax/model.py::PackedLayout`) drive frame-exact inverse physics, and how event-gated envelopes plus multimodal differential velocity subtraction resolve long-standing `masked_v2v` face/mouth/gaze failure modes.

For day-to-day recipes and parameter selection, see [`../../wiki/edit_guidance.md`](../../edit_guidance.md), [`../../wiki/control_vectors/examples_and_recipes.md`](../../control_vectors/examples_and_recipes.md), and [`../../wiki/control_vectors/which_tool_and_what_to_call_it.md`](../../control_vectors/which_tool_and_what_to_call_it.md).

---

## 1. Context: Why Today's Four Batches Needed a Controlled Synthesis

Earlier on `2026-11-12`, two separate questions arose:

- **How much of a post-refiner control vector is actually captured by its top J-Lens words, and how do fresh single-phrase vectors compare against plain prompt rewrites and 2-pass `MiniMaxH3EditGuider`?**
  - `exp2` (`$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/exp2/`) decomposed `film_16mm_1974` and `head_swell` into orthogonal parts `v = m (shared) + n (named 16 J-Lens words orth shared) + r (rest)`.
  - `exp3` built eight new single-phrase control vectors via `control_vector_nodes.py::MiniMaxH3MakeControlVector` with empty `target_prompt` and rendered all eight at node defaults (`mode = add_norm_matched`, `strength = 1.15`, `target = scene_only`, `start_percent = 0.0`, `end_percent = 1.0`).
  - `exp4` rendered the same eight phrases as plain prompt prepends (`exp4_<concept>_rewrite`) and through `edit_guidance.py::MiniMaxH3EditGuider` at `scale = 1.5, first_step = 0, last_step = 1000` (`exp4_<concept>_node_s1.5`), leaving the clips unjudged at handoff.
- **How do external temporal/audio guides (`NKD_v153`, `NKD_v173`) achieve frame-exact physical choreography (dribbling crossovers, mid-air tortoise jumps with landing thuds) without destroying continuous scene foley, and how does that translate to `masked_v2v` (mesh/eyes/gaze/mouth) and `MiniMaxH3EditGuider`?**

Inspecting both the video frames and the decoded audio waveforms across `exp2`–`exp4` alongside `NKD_v153`/`NKD_v173` revealed that **conditioning timing (which step, which token span, and which frame interval gets pushed)** governs almost every win and failure across both problems.

---

## 2. Forensic Audit of `exp2`, `exp3`, and `exp4`

### 2.1 What `exp2` proved (and one normalization trap)

On `prompt_bank/t2va_offpath_tortoise.txt:1` (`seed = 730451892`), `film_16mm_1974` splits its squared norm into `16.0%` `shared` (`m`), `30.2%` `named` (`n`, the 16 J-Lens readout words orthogonalized against `m`), and `53.8%` `rest` (`r`).

- **The 16 J-Lens words carry genuine visual semantics, not just lexical echoes:** Both `WHOLE` (`v`) and `NAMED` (`n`) physically replace the modern rounded 2020s London double-decker (`62`) in `exp2_base` with a 1950s–1970s open-radiator AEC Routemaster on warm, halated 1974 16mm stock. `REST` (`r`, despite holding `53.8%` of raw energy) loses both the vintage Routemaster body and the warm 1974 color grade, leaving only flatter midtone contrast.
- **Normalization caveat in `exp2`:** `exp2` rendered `m`, `n`, and `r` individually using `mode = add_norm_matched` at `strength = 1.15` (`77.6` post-refiner norm each; `control_vectors.py:2588-2591`). For `head_swell`, where `n` holds only `15%` of `v`'s squared norm, `add_norm_matched` amplified `n` by `1 / sqrt(0.15) = 2.58x` relative to its actual weight inside `WHOLE`. Future part-ablation renders must use `mode = add` at `strength = 1.15 * (||v_hat_norm_matched|| / ||v||)` so `m + n + r` sums linearly to the `WHOLE` push.

### 2.2 The `shared` / `shared_new` "anti-mundane" Pole-A artifact (`cos = 0.72`)

Across the eight `exp3` vectors, median cosine with their shared batch mean (`shared_new`) was **`0.72` (~`52%` of squared length)**, and five of the eight fell below the healthy pairwise-agreement floor (`0.49` on `tv_1950s_broadcast`, `0.57–0.62` on four others; [`2026-11-12_new_vectors_three_ways.md`](2026-11-12_new_vectors_three_ways.md)). Two compounding causes produced this:

1. **Lopsided Pole-A vs. Pole-B compilation (`h3_pair_compiler.py:59-98`):** With `target_prompt = ""` and a bare 12–19 token Pole-B phrase, the compiler paired vivid concept clauses against 3–6 token `DEFAULT_AUX_BASELINES` (`"Everyday contemporary realism,"`, `"Plain matter-of-fact framing,"`). Over half of every vector's energy therefore pointed away from "quiet/ordinary/everyday" rather than uniquely toward its own concept.
2. **Colliding prompt-bank scene headers:** The five bank scenes (`t2va_appalachian_folk_cabin`, `t2va_arena_rock_video`, `t2va_box_office`, `t2va_cable_car`, `t2va_cafe_kids`) open with baked-in style/weather headers (`"Vintage film, 16mm grain"`, `"2D-animated, flat cel shading... snowed valley"`, `"golden hour"`), fighting the prepended clauses during `TokenRefiner` self-attention.

**Observable render consequence:** `prompt_bank/t2va_offpath_tortoise.txt:1` specifies `"a green knitted cardigan"` and `"a single lit white birthday candle... its small flame flickering"`. In `exp2_base` (`-29.0 dBFS`), the candle flame is tiny. Pushing `+1.15` along `exp2_shared` (`-23.1 dBFS`, **`+5.9 dB`** louder) or `exp3_shared_new` (`-28.4 dBFS`) turns the tiny birthday candle into a **roaring open torch** with moody street haze, while reversing `shared_new` (`-1.15`, `-36.5 dBFS`, **`-7.5 dB`**) snuffs the candle flame completely and strips the background to a sterile overcast street.

### 2.3 Four confounds in `exp3` and `exp4` (including audio blind spots)

Checking both video and decoded PCM audio across `exp3` and `exp4` surfaced four confounds that prevented `exp3` vs. `exp4` from being a fair comparison of Control Vectors vs. Edit Guidance:

1. **Under-driven CV settings across all 6 steps:** `strength = 1.15` (`77.6` norm, of which ~`52%` was `shared_new` energy -> only ~`54` effective concept norm) spread across `0.0..1.0` with `target = scene_only` left all `41` `soundscape` tokens untouched and barely frosted the asphalt on `heavy_blizzard`.
2. **Prompt self-contradiction on all four medium/style concepts:** `exp4` prepended `<phrase>,` directly before `"Live-action, cinematic, a charming whimsical urban scene..."`. Keeping `"Live-action, cinematic"` vetoed `watercolor_wash`, `toy_brick_world`, `tv_1950s_broadcast`, and `knitted_wool_world` across **every** method (`knitted_wool_world` only thickened the old man's sweater because `"green knitted cardigan"` was already written in `prompt_bank/t2va_offpath_tortoise.txt:1`).
3. **Silent-stack audio blind spots (`-10 dB` dead storm audio vs. `+7.3 dB` 1950s TV hiss):**
   - In `exp4`, `heavy_blizzard` and `ankle_deep_flood` succeeded visually, yet **audio RMS collapsed by `-7.9` to `-10.6 dB`** (`-36.9 dBFS` rewrite / `-39.3 dBFS` guider on `heavy_blizzard`; `-37.1 dBFS` / `-39.6 dBFS` on `ankle_deep_flood`, vs. `-29.0 dBFS` `exp2_base`) with zero howling wind or sloshing water, because `overall_soundscape:` still instructed `"the tortoise's claws scrape softly on the asphalt, with light traffic in the distance"`.
   - Conversely, on `tv_1950s_broadcast`, video stayed full-color live-action (vetoed by `"Live-action, cinematic"` and `"red double-decker bus"`), while **`> 2.5 kHz` audio hiss jumped `+7.3 dB` (rewrite) and `+6.9 dB` (guider)**—real 1950s television static/air hiss emerged in the audio stream where no contradictory prose blocked it.
4. **`EditGuider` tested only at `first_step = 0` (`last_step = 1000`):** Running `EditGuider` across all 6 steps (`first_step = 0`) re-rolls step-0 scene layout just like a plain rewrite (`12` passes = `2x` DiT cost) and never tests `EditGuider`'s primary advantage: locking the unedited shot's exact composition on step `0` (`first_step = 1`) before steering weather, lighting, and foley on steps `1..3`.

---

## 3. The `exp5` 1:1 Controlled Ladder (`heavy_blizzard` & `watercolor_wash`)

To separate **vector quality**, **inference knobs**, **prompt hygiene**, and **step-0 layout locking**, we ran 13 new renders on the exact same base graph, prompt (`prompt_bank/t2va_offpath_tortoise.txt`), resolution (`1344x768`), length (`124` frames), and seed (`730451892`) under `$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/exp5/`, choosing one world/weather+foley concept (`heavy_blizzard`) and one medium conversion (`watercolor_wash`) that had failed completely across `exp3` and `exp4`.

### 3.1 Clean non-contradictory edits and in-place vector builds

Using `edit_guidance.py::apply_edits` (`h3_prompt_edit.py`), we replaced the conflicting clauses instead of prepending onto them:

- **`heavy_blizzard` (2 clauses — visual + coupled soundscape):**
  ```text
  on a quiet city street on an overcast morning -> on a wind-lashed city street in a heavy blizzard with thick snow piled on every surface and swirling flakes
  soundscape: the tortoise's claws scrape softly on the asphalt, with light traffic in the distance -> freezing gale wind howls across deep snowdrifts as thick snow crunches softly underfoot
  ```
- **`watercolor_wash` (1 clause — replaces `"Live-action, cinematic"`):**
  ```text
  Live-action, cinematic -> Loose hand-painted watercolor animation on rough cold-press paper with soft bleeding pigment washes and visible brush strokes
  ```

Passing those exact `old -> new` edits into `control_vector_nodes.py::MiniMaxH3MakeControlVector` with `target_prompt` set to `prompt_bank/t2va_offpath_tortoise.txt` (`pooling = diff_span`, `apply_holds = true`) built balanced in-place vectors (`trial_exp5_*`, archived after rendering so `<comfy>/models/control_vectors/` stayed clean):

| Vector | Pooled tokens (Pole A vs. Pole B) | Raw norm | Cosine vs. `shared_new` (`%` energy in batch bias) | J-Lens `+` (toward) | J-Lens `-` (away from) |
|---|---|---:|---:|---|---|
| `heavy_blizzard` (`exp3` generic) | `3–6` vs `16` | `83.1` | **`+0.765` (`58.5%`)** | `snow, dust, storm, cloud, rain, layers, thick, frost` | `shot, documentary, straightforward, unnamed, minimalist...` |
| `trial_exp5_blizzard_inplace` | `25` vs `40` | `66.4` | **`+0.244` (`6.0%`)** | `snow, snowy, extreme, frost, freezing, storm, arctic, overwhelming` | `quiet, modest, pleasant, casual, mildly, casually, quietly, walking` |
| `watercolor_wash` (`exp3` generic) | `3–6` vs `13` | `77.9` | **`+0.735` (`54.0%`)** | `water, sketch, texture, layers, paper, textured, delicate, shimmer` | `nighttime, daytime, unidentified, protagonist, unnamed...` |
| `trial_exp5_watercolor_inplace` | `4` vs `23` | `96.3` | **`+0.389` (`15.1%`)** | `sketch, drawn, texture, fuzzy, paper, traced, irregular, loosely` | `shot, cinematic, scene, view, nighttime, indoor, shots, camera` |

Matching Pole A to real scene clauses collapsed the unwanted `shared_new` variance share by nearly **`10x` on `heavy_blizzard` (`58.5% -> 6.0%`)** and **`3.6x` on `watercolor_wash` (`54.0% -> 15.1%`)**.

### 3.2 Quantitative & visual/audio results across the full ladder

In the tables below, `Mean L1` and `Frame-0 L1` are mean absolute RGB pixel differences (`0..255`) against `exp2_base` (`seed = 730451892`), `Audio dBFS` is full-clip RMS (`exp2_base = -29.0 dBFS`), and `Hi >2.5k` is high-band spectral change relative to `exp2_base` (wind/brush/hiss energy). Labeled comparison reels with per-segment audio live at `$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/exp5_heavy_blizzard_ladder_with_audio.mp4` and `$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/exp5_watercolor_wash_ladder_with_audio.mp4`.

#### Table A — `heavy_blizzard` (Weather + Foley Concept)

| Arm | Render time (passes) | Mean L1 (`f0` L1) | Audio dBFS ($\Delta$ base / $\Delta$ `>2.5k`) | Visual & audio outcome |
|---|---:|---:|---:|---|
| `exp2_base` (unsteered) | `52s` (`6`) | `0.0` (`0.0`) | `-29.0` (`0.0` / `0.0 dB`) | Dry London street, bus `62`, yellow bollard, old man tipping cap, tortoise with small lit candle, diesel + street ambience. |
| `exp3` CV generic (`s=1.15, scene_only, 0..1`) | `52s` (`6`) | `49.9` (`52.7`) | `-34.0` (`-5.0` / `-3.7 dB`) | Faint frost dust on asphalt, zero falling snow, quieter audio (`soundscape` unsteered). |
| `exp5` CV generic (`s=2.25, non_vision, 0..0.5`) | `55s` (`6`) | `70.8` (`64.3`) | **`-19.1` (`+9.8` / `+14.5 dB`)** | **Ferocious whiteout blizzard + roaring gale wind** (`exp3`'s vector was under-driven, NOT empty), **but** `151.9` step-0 shift overwrites `"tortoise"`/`"candle"` nouns into a 3-foot snow-volcano cone. |
| `exp5` CV in-place (`s=2.25, non_vision, 0..0.5`) | `52.5s` (`6`) | `75.1` (`81.4`) | **`-22.0` (`+7.0` / `+15.4 dB`)** | Even deeper snowdrifts + howling wind, completely burying the tortoise into a moving snow mound at `s=2.25` from step `0`. |
| **`exp5` CV generic (`s=1.85, non_vision, [0.16, 0.67)`)** | **`52s` (`6`)** | **`45.5` (`46.4`)** | **`-28.3` (`+0.7` / `+11.8 dB`)** | **1-pass step-matched (`steps 1..3`) CV:** locks the `base` London street, bus (`02`), and old man at `1x` cost while adding falling snow + `+11.8 dB` wind hiss; still frosts/cakes the tortoise shell because CV pushes all 276 tokens uniformly. |
| `exp5` CV in-place (`s=1.85, non_vision, [0.16, 0.67)`) | `52s` (`6`) | `51.9` (`52.9`) | `-33.1` (`-4.1` / `+4.4 dB`) | Locks the street and man, lighter snow/frost, softer wind than generic at `1.85` (`raw_norm = 66.4`). |
| `exp4` Rewrite (conflicting prepend) | `57s` (`6`) | `49.5` (`51.9`) | `-36.9` (`-7.9` / `-8.4 dB`) | Snow on road, re-rolled street/bus, **dead muffled audio** (dry-asphalt `soundscape:` clashes with snow visuals). |
| `exp5` Rewrite CLEAN (visual + audio edit) | `57.5s` (`6`) | `67.6` (`66.1`) | `-34.4` (`-5.4` / `-3.4 dB`) | Clean snowy street, intact tortoise + lit candle, natural winter wind + footsteps, **but** re-rolls street geometry (`bus 12`, different shops). |
| `exp5` `EditGuider` (`f0, s=1.65, last=3`) | `87.6s` (`10`) | `74.4` (`70.6`) | `-30.7` (`-1.7` / `+1.5 dB`) | Heavier swirling snow + `+3.7 dB` stronger wind than clean rewrite, intact tortoise + candle, re-rolled street (`bus 26`). |
| **`exp5` `EditGuider` (`f1, s=1.85, last=3`)** | **`72.6s` (`9`)** | **`44.5` (`46.9`)** | **`-30.7` (`-1.7` / `+1.7 dB`)** | **Best overall result:** preserves `exp2_base`'s **exact** Georgian facade, bus `62`, yellow bollard, cap-tipping choreography, and clean un-mutated tortoise + flickering candle, while piling deep snowdrifts across the road with falling flakes and winter wind + bus rumble. |

#### Table B — `watercolor_wash` (Whole-Medium Style Conversion)

| Arm | Render time (passes) | Mean L1 (`f0` L1) | Audio dBFS ($\Delta$ base) | Visual & audio outcome |
|---|---:|---:|---:|---|
| `exp3` CV generic (`s=1.15, scene_only, 0..1`) | `52s` (`6`) | `59.7` (`59.2`) | `-35.4` (`-6.4 dB`) | 100% photorealistic live-action street (slight contrast warm-up, zero watercolor). |
| `exp5` CV generic (`s=2.25, scene_only, 0..0.5`) | `52.5s` (`6`) | `91.9` (`96.2`) | `-32.0` (`-3.0 dB`) | **Total 3D perspective collapse:** camera flattens into abstract 2D red-and-white painted stripes on paper (`151.9` shift norm overwrites `"bus"` and `"zebra crossing"` nouns at step `0`). |
| `exp5` CV in-place (`s=2.25, scene_only, 0..0.5`) | `52.5s` (`6`) | `73.0` (`67.9`) | `-28.7` (`+0.3 dB`) | Hand-drawn ink-and-watercolor man and tortoise, but `s=2.25` still flattens background bus/crosswalk into 2D checkerboard squares. |
| **`exp5` CV in-place (`s=1.60, scene_only, 0..0.5`)** | **`52s` (`6`)** | **`66.0` (`62.8`)** | **`-30.3` (`-1.3 dB`)** | **1-pass CV sweet spot (`108.0` norm `< 110` card ceiling):** preserves full 3D London street, red double-decker bus, crosswalk, man, and tortoise while rendering them in a stylized hand-illustrated storybook/watercolor look at `1x` DiT cost. |
| `exp4` Rewrite (`"Live-action, cinematic"` kept) | `57s` (`6`) | `49.6` (`47.7`) | `-32.6` (`-3.6 dB`) | 100% live-action photograph (vetoed by `"Live-action, cinematic"`). |
| **`exp5` Rewrite CLEAN (`"Live-action"` replaced)** | **`57.6s` (`6`)** | **`70.0` (`68.5`)** | **`-27.9` (`+1.1 dB`)** | **Full hand-painted British picture-book watercolor animation** across the entire street, bus + driver, old man, and tortoise + candle (`1x` cost). |
| **`exp5` `EditGuider` (`f0, s=1.65, last=3`)** | **`87.6s` (`10`)** | **`75.4` (`70.8`)** | **`-29.8` (`-0.8 dB`)** | **Strongest painterly medium aesthetic:** amplifies rough cold-press paper vignette borders, wet-on-wet cobalt/crimson pigment bleeds, and loose ink linework while keeping every character coherent. |
| `exp5` `EditGuider` (`f1, s=1.85, last=3`) | `72.6s` (`9`) | `34.2` (`38.2`) | `-30.2` (`-1.2 dB`) | Locks `exp2_base` live-action street (`bus 66/62`) with subtle painterly wall washes—proving **whole-medium conversions belong at `first_step = 0`**, whereas **weather/lighting/foley edits on a live-action take belong at `first_step = 1`**. |

---

## 4. What `exp5` Settled: When to Use Plain Rewrite vs. Control Vector vs. Edit Guidance

### 4.1 Why `t2va` step `0` behaves differently from `ref2va` under strong Control Vectors

On `ref2va` (`<Picture 1>`), clean VAE reference latents sit in `minimax_refs` pinned at `max(t_v, VISUAL_COND_TIMESTEP)` (`comfy/ldm/minimax/model.py:746-750`), anchoring 3D object identity and perspective even when text rows receive a `120–150` norm shift at step `0`.

On `t2va`, **text rows (`median post-refiner row norm ≈ 164.4`) are the sole blueprint of 3D geometry on step `0` (`sigma = 1.0`)**. Adding a uniform vector of length `2.25 * 67.5 = 151.9` (`92%` of every noun's own embedding norm!) across all `212` (`scene_only`) or `276` (`non_vision`) rows on step `0` drowns out individual subject nouns (`"tortoise"`, `"candle"`, `"double-decker bus"`), turning the tortoise into a snow cone (`heavy_blizzard`) or collapsing the street into flat red-and-white stripes (`watercolor_wash`).

Two distinct levers avoid `t2va` noun overwrite at `1x` single-pass speed:
1. **If steering from step `0` (`start_percent = 0.0`, e.g., style/medium vectors):** keep the post-refiner shift norm at or below the build card's `110` ceiling (`strength <= 1.60` in `add_norm_matched`, `end_percent = 0.50`), as verified by `watercolor_wash` at `s = 1.60`.
2. **If pushing strong weather/atmosphere (`strength = 1.80–2.20`):** set **`start_percent = 0.16, end_percent = 0.67`** (`steps 1..3` of 6), letting unsteered step `0` lock 3D scene and subject geometry before the vector drives weather and wind foley.

### 4.2 Why `EditGuider` is the subtle, clause-aware tool and Control Vectors are the blunt broadcast dial

This ladder directly tests—and confirms—the owner's reading recorded in [`2026-11-12_chain_lens_digest_and_experiment_plan.md`](2026-11-12_chain_lens_digest_and_experiment_plan.md) (§15.6: *"a control vector is the blunt tool and edit guidance the subtle one, and a change of style or look is where edit guidance shines"*):

1. **Rank-1 broadcast vs. clause-resolved spatiotemporal velocity:** Even when `MiniMaxH3Steer` skips step `0` (`[0.16, 0.67)`), a control vector adds **one rank-1 vector uniformly across every steered text row**, so `"tortoise"`, `"birthday candle"`, and `"double-decker bus"` receive the exact same shift as `"city street"`. On `watercolor_wash`, pushing the vector hard (`s = 2.25`) flattened 3D perspective into 2D stripes, leaving a narrow `1x`-pass sweet spot around `s = 1.60` on the clean in-place vector. By contrast, `MiniMaxH3EditGuider` runs the DiT on the actual edited grammar and extrapolates **per-patch spatiotemporal velocity** `v_orig + scale * (v_edit - v_orig)` (`edit_guidance.py:196-202`):
   - At `first_step = 0, scale = 1.65, last_step = 3`, `EditGuider` delivered the richest cold-press paper vignette, bleeding cobalt/crimson washes, and ink linework on `watercolor_wash` while keeping the 3D bus, driver, old man, and tortoise intact.
   - At `first_step = 1, last_step = 3` (or fractional `scale < 1.0`), `EditGuider` holds the original shot's exact geometry and dials in subtle painterly washes (`watercolor_wash`, `L1 = 34.2`) or full take-locked blizzard weather + winter foley (`heavy_blizzard`, `L1 = 44.5`) in `9` passes (`1.5x` DiT cost, `72.6s` vs. `52.0s` 1-pass and `104s` naive 12-pass).
2. **Where Control Vectors still earn their keep:** zero-prompt-edit `1.0x`-cost reuse across any prompt, signed negation (`-strength` / `project_out`), and gentle stock/era grades within the `<= 108` step-0 shift ceiling (`strength <= 1.60`) or step-1 window (`[0.16, 0.67)`).

### 4.3 Practical Decision Matrix

| Goal | Best tool | Recommended settings (6-step Pika H3 Flex) | Cost |
|---|---|---|---|
| **New shot or whole-medium conversion** (`watercolor`, `claymation`, `2D cel`) where base take layout need not be locked | **Plain clean rewrite** (replacing conflicting `"Live-action"` clauses) — or **`EditGuider` (`first_step = 0`)** if `"100%"` of prose still under-shoots | Clean `MiniMaxH3PromptEdit` (`EditGuider`: `first_step = 0, last_step = 3, scale = 1.5–1.7`) | `1.0x` (`1.67x` w/ guider) |
| **Lock an existing take's exact building, actors, camera, and subject anatomy** while changing weather, time-of-day, lighting, or foley | **`MiniMaxH3EditGuider` (`first_step = 1`)** + coupled `soundscape:` edit | `first_step = 1, last_step = 3, scale = 1.65–1.90` (`streams = independent` when audio needs stronger/softer push than video) | **`1.5x`** (`9` passes) |
| **Reusable zero-prompt-edit aesthetic dial across arbitrary prompts at `1x` speed**, or continuous slider blends (`0.3x`, `-0.8x`, cross-vector holds) | **`MiniMaxH3Steer` (Control Vector)** built from balanced Pole-A/Pole-B pairs | Style/stock from step `0`: `strength = 1.2–1.6, scene_only, [0.0, 0.50)`. Weather/foley preserving `t2va` layout: `strength = 1.7–2.0, non_vision, [0.16, 0.67)`. | **`1.0x`** (`6` passes) |

---

## 5. Temporal & Audio Event Guides (`NKD_v153` & `NKD_v173`): How They Work and Why

We inspected two `141`-frame (`5.875s` @ `24 fps`, `17 * 8 + 5` H3 temporal grid) clips shared in Discord (`NKD_v153.mp4` and `NKD_v173.mp4`), each rendered with a bottom telemetry HUD plotting frame index `0..141` and a temporal guide envelope:

### 5.1 Observable frame-and-audio telemetry

1. **`NKD_v153.mp4` (`1216x704` canvas + `196px` post-render telemetry HUD, `141` frames):**
   - **Envelope:** 12 narrow square-wave pulses at `f5–6, 17–18, 29, 40–41, 52–53, 64–65, 76, 87–88, 99–100, 111–112, 123, 135` (period = `11–12` frames ≈ `0.48s` ≈ `125 BPM`), zero everywhere else.
   - **Audio waveform:** Literal digital silence (`RMS = 0.0000`) on inter-pulse frames (`f10..15`, `f25..26`, `f35`, etc.) and sharp `0.15–0.36` RMS basketball-bounce transients at `[5, 17, 29, 41, 64, 76, 88, 100, 111, 123, 135]`.
   - **Generated video:** A player on an **outdoor painted asphalt court** (chain-link fence, trees, neighborhood houses) dribbles left-to-right crossovers and between-the-legs switches such that **the ball strikes the ground on the exact pulse frames**. Between pulses, H3's DiT solves the inverse kinematics (hand switch, knee bend, ball apex mid-way between pulses) needed to satisfy the boundary conditions.
2. **`NKD_v173.mp4` (`1216x704` canvas + `196px` post-render telemetry HUD, `141` frames, London tortoise-on-leash scene — static camera, harness strap with no birthday candle):**
   - **Envelope:** Three shaped, fast-attack / exponential-decay impulse curves (`~9–11` frames wide) starting at **`f20..29`**, **`f70..80`**, and **`f105..113`**, flat zero elsewhere.
   - **Generated video & audio:** On `f0–17`, `f30–68`, `f81–103`, and `f114–140`, the tortoise walks normally along the London street with **continuous street ambience (`min RMS = 0.0181`, `median = 0.0361` at `24 kHz` 1-frame windows, never zeroed)**. At **`f18–26`**, **`f69–76`**, and **`f104–110`**—lifting off **1–2 frames *before* each HUD curve onset** because `f20`, `f70`, `f105` sit inside 4-frame `(1, 4, 4, 4, 4)` latent buckets (`f18..21`, `f69..72`, `f103..106`) and bidirectional self-attention anticipates the pinned event—the tortoise launches **1.5–2 feet into the air** while the red leash goes slack mid-air and pulls taut on landing (`24 kHz` 1-frame RMS bumps to `0.0558` at `f27`, `0.0529` at `f76`, and `0.0632` at `f108` / `0.0605` at `f111` vs. `0.0361` median).

### 5.2 Why H3 responds so strongly: `keyframes` (`cond` / `cond_audio`) vs. `refs` in `PackedLayout`

In `comfy/ldm/minimax/model.py:400-472`, H3 has two distinct conditioning mechanisms with very different RoPE geometries:

- **`minimax_refs` (`<Picture 1>`, `<Video 1>`, `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo`):** Packed **before** `cursor` (`[text_len .. cursor)`), while the generated video+audio timeline starts **after** `cursor` (`[cursor .. end)`). Multiple `<Video k>` references pack sequentially (`[text_len, text_len + T1)`, `[text_len + T1, text_len + T1 + T2)`) rather than sharing a common RoPE time axis.
- **`minimax_keyframes` (`cond` and `cond_audio`, `comfy_extras/nodes_minimax_h3.py::MiniMaxH3AddGuide` and `guide_latent.py::MiniMaxH3GuideLatent`):** Placed at **`cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]`** (`comfy/ldm/minimax/model.py:408-426`) on the **exact same `(t, h, w)` 3D RoPE coordinates** as the noisy target tokens, while pinned at clean timestep `max(t_v, VISUAL_COND_TIMESTEP)` / `max(t_a, AUDIO_COND_TIMESTEP)` (`comfy/ldm/minimax/model.py:748-750`) with `img_update = False` / `audio_update = False`. Crucially, on stock H3 without a specialized Control LoRA, `cond_video_latents` pass through the exact same `video_patch_proj` (`comfy/ldm/minimax/model.py:608`) as `first_frame`/`last_frame` in `fl2va`—so feeding drawn HUD bars as a full-clip visual guide paints literal bars unless released early in the step schedule or replaced by localized audio/image event keyframes.
- Furthermore, when `audio_denoise_mask` or `denoise_mask` is supplied (`comfy/ldm/minimax/model.py:758-774`), per-frame mask values $m(t) \in [0, 1]$ continuously modulate each token's effective noise label `1.0 - m(t) * sigma`.

### 5.3 Community Cross-Check: Scheduled Attention Bias vs. `until`-Released Guides vs. Velocity Curves

Auditing ComfyUI core, sister checkouts (`coderef/ComfyUI-MAINodes`, `coderef/ComfyUI-UtilsCollection`, `coderef/MaskVidExperiments`), and Nekodificador's public repos clarifies how existing "scheduled attention bias" and guide tools map onto H3:

1. **Logit-space attention bias in ComfyUI (`LTXV`, `MOSS-Image`, `NKD-Klein-Tools`) — and why dense 2D bias fails on H3:**
   - Core LTX-Video (`comfy/ldm/lightricks/model.py:371-424`, `GuideAttentionMask`) and MOSS-Image (`comfy/ldm/moss_image/model.py:246-250`, `context_attention_bias`) add $\ln w_g$ to attention logits before softmax; Nekodificador built a step-scheduled 2D `[canvas_Q × ref_K]` regional attention bias for Flux Klein (`ComfyUI-NKD-Klein-Tools`, `S ≈ 4k`).
   - On H3 (`comfy/ldm/minimax/model.py:207` hardcodes `mask=None`, `S ≈ 30,000`, `56` heads), passing a frame-varying 2D `[1, 1, S, S]` mask into `optimized_attention` forces `MiniMaxH3Sol` (`comfy/ldm/modules/attention.py:700-701`) and `attention.py::make_sage_override` (`attention.py:366-367`) off their fast paths, allocates `1.8–3.6 GB` (`100.8 GB` OOM if head-expanded), and streams `~100.8 GB` of mask reads per block (`5 TB/step`). Moreover, softmax reweighting stays strictly inside the convex hull $\text{conv}(V)$ and cannot subtract a ground-bound base trajectory. *(If 1-pass per-frame attention bias onto small key segments is ever needed on H3, the right implementation is an exact `return_lse=True` online-softmax merge against `coderef/sage-fork/sageattention/core.py:282`, keeping the `~30,000`-token main self-attention 100% unmasked.)*
2. **What Nekodificador actually built for H3 (`ComfyUI-NKD-Basic-Tools`, `nkd_minimax_guides.py`):**
   - Multi-slot `MiniMaxH3AddGuide` (`cond` + `cond_audio` keyframes) paired with a per-slot **`until` step-cutoff widget** (`ConditioningSetTimestepRange` over `[0.0, until]`). Because `comfy/model_base.py:2292-2300` builds a separate `PackedLayout` per timestep-ranged conditioning entry, dropping sparse event guides after early steps (`until ≈ 0.50–0.80`) locks exact frame timing on steps `0..3` and then lets H3's unguided joint AV prior synthesize clean leg push, leash slack, and continuous street ambience on the final steps with zero attention mask.
3. **2-pass velocity-curve limit on text dials (`coderef/ComfyUI-MAINodes/dials/timecurve.py:137-177`, `coderef/ComfyUI-MAINodes/dials/LESSONS.md:137-140`):**
   - Blending a **prompt-side control vector** per latent frame (`v0 + M(t)*(v_dial - v0)`) showed `~20%` bleed onto `M(t) = 0` frames and `1.5–2 s` transition ramps (missing `0.33 s` beats) because text dials lack RoPE frame coordinates and step $s+1$ attends bidirectionally across the shared latent. Combining **time-invariant state edits + `[T, H, W]` spatial/temporal velocity masks in `MiniMaxH3EditGuider`** (kept active through `last_step = 4` on `sample_lcm` so step `3–4` noise re-injection does not erase the impulse) with **`until`-released `MiniMaxH3GuideLatent` / `MiniMaxH3AddGuide` keyframes** gives us both sharp sub-second triggers and artifact-free semantic/spatial control.

---

## 6. Actionable Recommendations & Prioritized Roadmap

Combining the `exp5` ladder findings with the `NKD` temporal-guide mechanics points to **four concrete, high-leverage engineering steps**, ordered by return on effort:

### Priority 1 — Spatiotemporal Curve & Mask Modulation in `MiniMaxH3EditGuider` (`~40` lines in `edit_guidance.py`)

Right now, `EditGuider.predict_noise` (`edit_guidance.py:196-202`) multiplies the velocity delta `(v1 - v0)` and `(a1 - a0)` by scalar floats `eff_v` and `eff_a`. Because `v1 - v0` already has shape `[1, 24, latent_t, lat_h, lat_w]` (`latent_t` spans `1, 4, 4, ...` pixel frames) and `a1 - a0` has shape `[1, 32, 2, audio_t]`:

- **What to build:** Accept an optional **`mask` (`MASK` `[T, H, W]`)** socket appended to `MiniMaxH3EditGuider`, reduced onto `[1, 1, latent_t, lat_h, lat_w]` for video (using `comfy/model_base.py:2314-2325` `MiniMaxH3._latent_pixel_starts` with temporal `mean` over `(1, 4, 4, 4, 4)` buckets so smooth envelopes do not clip to `1.0`) and spatially `amax`-pooled over `(H, W)` onto `[1, 1, 1, audio_t]` for audio:
  $$v_{\text{out}}(t, h, w) = v_0(t, h, w) + s_{\text{video}} \cdot M(t, h, w) \cdot \bigl(v_1(t, h, w) - v_0(t, h, w)\bigr)$$
  Under 6-step `sample_lcm` (`sigmas = [1.0, 0.973, 0.923, 0.80, 0.571, 0.333, 0.0]`), keep masked 2-pass guidance active through `last_step = 4` (`10` passes when `first_step = 1`, or apply `mask` at scale `1.0` after `last_step`) so `57%`/`33%` noise re-injection on steps `3–4` does not erase sparse impulses.
- **Why this is huge:**
  1. **Temporal choreography from time-invariant state edits:** Phrase `edited` as a state true on every frame (*"the tortoise is airborne two feet above the asphalt, all four legs off the ground"*) with smooth impulse peaks in `mask(t)` (`first_step = 1, last_step = 4`), or extend to 2-channel additive steering `v0 + s1 * M1 * (v1 - v0) + s2 * M2 * (v2 - v0)` to test multi-actor cause-and-effect (e.g., tortoise hop + mid-hop leash yank).
  2. **Spatial subject protection (fixing the `t2va` snow-cone / subject bleed problem):** Pass a simple box or SAM mask $M(t, h, w)$ so `heavy_blizzard` or `head_swell` pushes `2.2x` on its target region and `0.0x` elsewhere.

### Priority 2 — Multimodal Differential `EditGuider` (`refs` / `keyframes`) -> Differential Grey-Mesh Subtraction for `masked_v2v`

Currently, `check_same_but_text` (`edit_guidance.py:75-82`) only checks that `original` and `edited` have the same **count** of `minimax_refs` and `minimax_keyframes`, while `_prepare` (`edit_guidance.py:156-170`) aligns RoPE origins using `layout.signature[0]` (`text_len`) alone.

- **What works today vs. what to build:**
  1. **Works today with zero code changes (`2` refs vs. `2` refs):** When `original` (`<Picture 1> + plain_cut`) and `edited` (`<Picture 1> + drawn_cut`) have the same reference count and duration, `check_same_but_text` already passes and `cursor_orig == cursor_edit`!
  2. **Segment-aware RoPE alignment for differing `refs` / `keyframes` counts:** When lifting `check_same_but_text` for `0` guides vs. `N` guides or `<Picture 1>` vs. `<Picture 1> + <Video 1>`, walk `layout.segments` (`comfy/ldm/minimax/model.py:414-482`) rather than shifting `[text_len:, 0]` uniformly: `ref_img`/`ref_audio` segments start at `text_len` (shift by `max_text_len - text_len`), whereas target-timeline segments (`cond`, `cond_audio`, `audio`, `video`) start at `cursor` (shift by `max_cursor - cursor`), keeping shared `<Picture 1>` coordinates aligned.
  3. **Apply directly to `masked_v2v` (`bench/head_reference.py`, `bench/draw_face_points.py`, [`../masking/2026-10-26_code_map_and_simplification_plan.md`](../masking/2026-10-26_code_map_and_simplification_plan.md)):**
     - **`original` pass:** `<Picture 1>` + **plain neutral grey mesh** (`<stem>_plain_cut.mp4`, already written automatically by `bench/head_reference.py:40`).
     - **`edited` pass:** `<Picture 1>` + **drawn eyes/gaze/mouth mesh** (`<stem>.mp4`, identical grey head + white/black eye/iris/lip rings).
     - In `v_orig + scale * (v_edit - v_orig)`, **every shared artifact of the grey 3D mesh (grey clay shading, head-shape discrepancy, neck seam) cancels out in `v_edit - v_orig`**, isolating the pure differential velocity of **iris gaze, eyelid blink, and lip articulation** on top of `<Picture 1>`'s identity.

### Priority 3 — Event-Gated Facial Envelopes + Co-Located Guide Latents in `masked_v2v`

Even before running 2-pass differential guidance, `NKD_v173` directly diagnoses a classic `masked_v2v` failure mode: drawing static lip or brow lines on **every** frame (`0..T`) continuously tugs H3 away from `<Picture 1>` during neutral/closed-mouth stretches, causing steady-state parted-lip drift or chin/neck artifacts when keypoints slip off-mesh.

- **What to build in `bench/draw_face_points.py` / `bench/head_reference.py`:**
  - Compute a 1D per-frame facial-activity envelope from `SHOT_points.npz`—specifically **mouth-opening velocity / inner-lip gap** and **eyelid-closure blink transients**—and fade or gate drawn mouth features to `0` on closed-mouth resting intervals so `<Picture 1>` governs neutral expression undisturbed, firing the drawn guide only during actual syllable/smile/blink events.
  - Test routing the cropped/aligned face-event guide (and speech audio when lip-syncing) through co-located `minimax_keyframes` (`guide_latent.py::MiniMaxH3GuideLatent`, $\Delta t_{\text{RoPE}} = 0$) vs. `<Video 1>` (`minimax_refs`) on hard fast-blink and head-turn spans where cross-cursor attention lags by 1–2 frames.

### Priority 4 — Control Vector Hygiene & Guardrail Polish (`control_vector_nodes.py` / `h3_pair_compiler.py`)

1. **Fix bare-phrase `DEFAULT_AUX_BASELINES` length & domain balance (`h3_pair_compiler.py:59-98`):** Replace 3–5 word abstract meta-phrases (`"Everyday contemporary realism,"`) with matched-length concrete plain-scene clauses (or offer `project_out_bank_mean = True` at compile time) so single-phrase vectors without a `target_prompt` do not dump `50%+` of their energy into the `"anti-mundane / torch-candle"` direction.
2. **Surface the `t2va` vs. `ref2va` step-0 rule in `MiniMaxH3Steer` tooltips & recipes:** Document that on `t2va`, step-0 `add_norm_matched` pushes above `strength = 1.60` (`> 108` shift norm) overwrite subject nouns; recommend `start_percent = 0.16, end_percent = 0.67` whenever pushing `strength > 1.65` on `t2va`.
3. **Optional `.toml` pair refresh for the kept `2026-11-12` medium vectors:** Per [`2026-11-12_new_vectors_three_ways.md`](2026-11-12_new_vectors_three_ways.md) (§12), all eight `2026-11-12` vectors are intentionally kept in `<comfy>/models/control_vectors/` alongside generated `data/control_vectors/<name>.toml` pair files and `ref2va` siblings. If the four medium/style vectors (`watercolor_wash`, `tv_1950s_broadcast`, `toy_brick_world`, `knitted_wool_world`) are later refreshed in place, replacing contradictory `"Live-action, cinematic"` / `"Vintage film, 16mm grain"` headers on Pole B and matching Pole A clause length will lift their pairwise agreement (`0.49–0.65`) into the healthy range while stripping the `52%` `shared_new` bias.
