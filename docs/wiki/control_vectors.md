# Control Vectors (Activation Steering on MiniMax H3)

Control vectors steer MiniMax H3 video+audio generation by shifting the 5376-d post-refiner conditioning stream (`context` after `condition_proj` + 2-layer bidirectional `token_refiner`, wired via `WrappersMP.DIFFUSION_MODEL` around `MiniMaxH3Model._forward` right after `preprocess_text_embeds`) along a learned concept direction $\mathbf{v}_{\text{PR}} \in \mathbb{R}^{5376}$ extracted from 8–16 contrastive prompt pairs $(A_i, B_i)$.

- **Zero training, zero DiT slowdown**: A control vector is a single `~24 KB` `.safetensors` file containing a `5376`-float vector (`vector.post_refiner`). Applying one (or chaining several) adds `< 0.1 ms` per diffusion step.
- **No running ComfyUI server required to build**: Building a control vector only runs the text encoder (`Qwen3-VL-32B-Instruct` `layers.0..49`, `~5.5s` load + `~2.0s` for 10 pairs) and the tiny 2-block `token_refiner` (`~0.18s` on CPU). You do **not** need ComfyUI (`:8188`) running or the 22 GB DiT loaded to create vectors.

---

## 1. Progressive Quickstart Guide: 3 Ways to Build & Use Control Vectors

### Path A (`< 2 ms` on CPU, 0 VRAM, Zero `.toml` / `.md` Files): Instant Word Synthesis & Inline Steering

With the precomputed **Analytical Jacobian Lens (`J-Lens`)** bundle (`models/j_lens/j_lens_h3.safetensors`, [`j_lens.py`](../../j_lens.py)), you can create and screen `5376`-d post-refiner control vectors from **3–5 plain English words in `< 2 ms` on CPU** without loading `Qwen3-VL-32B-Instruct` or authoring any `.toml`/`.md` files:

#### Option A1 — Type Words Directly on `MiniMaxH3ApplyControlVectors` in ComfyUI (Zero CLI)
On any **`MiniMaxH3ApplyControlVectors`** node in ComfyUI:
1. Leave `vector_name` as `(none)` (or select an existing vector to stack with your words) and type 3–5 words into **`words`**, e.g.:
   - `words = "blizzard, freezing, snow"`
   - `words = "medieval, knight, armor, castle"`
   - `words = "disco, glitter, dancing"`
   - `words = "8-bit, led, steampunk"`
2. **Replacing a human subject or live-action medium?** (e.g., `skeleton`, `balloon`, `muppet`, `claymation`, `robot`): set **`against_words = "@prompt_subject"`** (which automatically extracts your actual scene prompt's medium and character clauses on Step 0 via `_h3_tail_text`) or **`"@replace_human"`**, and **`strength = 1.35 .. 1.60`** so the vector actively suppresses the base prompt's human/live-action anchor tokens.
3. **Strip unwanted side-effects on any vector with `hold_words` (`hold_alpha = 0.0 .. 1.0`)**: Whether you are using inline `words` **or** a pre-built `.safetensors` file in `vector_name` (`cursed_cctv`, `rubber_hose_1930s`, `film_16mm_1974`), typing **`hold_words = "panic, alarm, shaky"`** on the same node projects out that nuisance direction in `< 2 ms` on CPU and re-normalizes `vector.post_refiner` to its pre-hold L2 norm.
4. **Note on `vector_name`**: If you select a pre-built vector in `vector_name`, you can pass a bare stem (`"cursed_cctv"`) or either family's file—[`control_vector_nodes.py::resolve_vector_file_path`](../../control_vector_nodes.py#L97-L162) automatically resolves the matching `__fl2va__` or `__ref2va__` sibling for the connected UNet!

#### Option A2 — One CLI Command (`words`) or One Click in the Local Studio UI (`< 2 ms`)
```bash
# Build both fl2va + ref2va .safetensors files + J-Lens readiness check in < 2 ms on CPU:
python bench/control_vectors.py words "blizzard, freezing, snow"

# Pass --prompt to run the Token-Wise Prompt Receptor Scan (< 2 ms CPU) and auto-contrast against @prompt_subject:
python bench/control_vectors.py words "inflatable, balloon, glossy, latex" \
    --against-words @prompt_subject \
    --prompt prompt_bank/t2va_noodle_bar.txt

# Or use the fixed @replace_human preset and soft holds:
python bench/control_vectors.py words "skeleton, translucent, bones, skull" \
    --against-words @replace_human \
    --hold-words "shaky, blur"

# Or launch the local browser Studio UI (live < 2 ms J-Lens Readiness + Prompt Receptor Scan + 1-click Instant Build):
python bench/control_vectors.py ui
```

---

### Path B (`~8s` Encoder Pass, Zero `.md` / `.toml` Authoring): 10-Pair Canonical Scene Compiler (`--words`)

When you want a full `Qwen3-VL-32B-Instruct` (`layers.0..49`) + `token_refiner` extraction across 10 diverse H3 scenes (6 speaking + 4 silent, 5 interior + 5 exterior) without writing a Markdown file, [`h3_pair_compiler.py`](../../h3_pair_compiler.py) compiles a short contrast phrase into 10 validated H3 pairs automatically:

```bash
python bench/control_vectors.py build --words "steady tripod -> erratic handheld camera shake"
```
> **When to use Path A (`words`) vs. Path B (`build --words`)**: For concrete visual nouns, weather, lighting, and period styles (`blizzard`, `medieval`, `disco`, `8-bit`), **Path A (`words` / `--from-words`) produces a `2.1x–2.6x` stronger post-refiner J-Lens gain in `< 2 ms`** (`+3.44 .. +5.17` vs. `+1.47 .. +1.60`) because embedding a short phrase inside a 15-word template sentence in Path B dilutes the noun direction across grammar/scaffold tokens (`Arms 53..56`, [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md#L148-L173)). Use Path B (`--words`) or Path C (`.md`) when steering **multi-word camera choreography or vocal delivery trajectories**.

---

### Path C (`~8s`, Maximum Precision for Complex Physical Trajectories & Media Contrast Pairs)

For complex physical trajectories (`head_swell`, `zero_gravity`, `smoke_dissolve`) where each of the 10 scenes needs custom object-specific physics and foley—or when extracting a style/identity vector directly from reference images or videos (`MiniMaxH3MakeMediaControlVector` / `--ref-images-a` vs. `--ref-images-b`):

1. **Generate a 10-Pair Markdown File with Any LLM**:
   - Open [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md), copy the fenced system prompt in **Section 1**, paste it into Claude/Gemini/ChatGPT, and save the output to `my_effect.md`.
2. **Run One-Shot `build`**:
   ```bash
   python bench/control_vectors.py build my_effect.md
   # Or batch-build all 34 shipped .toml banks (68 .safetensors files + 68 .md sidecars):
   python bench/control_vectors.py build --all --overwrite
   ```
   In one `~8s` pass, `build` validates the pairs (`lint_pair_set`), writes `prompt_bank/control_vectors/my_effect.toml` and `my_effect_test_prompt.txt`, encodes all pairs once through `Qwen3-VL-32B-Instruct`, runs `token_refiner` for **both** `fl2va` and `ref2va` (`--pooling diff_span`), writes `.safetensors` + `.md` sidecars into `models/control_vectors/`, and prints the **J-Lens Pre-Render Readiness Report**.

---

### Pre-Render J-Lens Diagnostics (`j_lens.py::assess_steering_readiness` & `analyze_prompt_coupling`)

Before you spend `~40s` rendering a video on GPU, [`j_lens.py::assess_steering_readiness`](../../j_lens.py#L419-L539) and [`j_lens.py::analyze_prompt_coupling`](../../j_lens.py#L1308-L1430) report objective pre-render J-Lens diagnostics in `< 2 ms` on CPU (measured on the 24,912-word 7-prefix cloze `M_union` basis and `J_ref^{(family)}`):

| Pre-Render Metric | Word Synthesis (`basis="word_synthesis"`) | 10-Pair `Qwen3-VL` (`basis="post_refiner"`) | What It Measures & What It Cannot See |
|---|---|---|---|
| **`top1_gain` & `top3_mean_gain`** | `+2.60 .. +5.20` (`>= +3.0` on strong word hits) | `+1.00 .. +2.63` (median `+1.64` across 37 `diff_span` banks) | Log-odds gain at `scale = 90` on `M_union = W_lm_head[valid] @ J_49_to_63`. Word-synthesis vectors are constructed directly in `row(M_centered)` so they project onto `M_union` with `~2x` higher `top1_gain` than full contextual `Qwen3-VL` pair vectors. |
| **`refiner_transmission`** | `0.44 .. 0.59` (`>= 0.50` on concrete visual/material nouns) | `0.49 .. 0.65` (measured when `vector.encoder` is saved; `None` if absent) | $\|\mathbf{J}_{\text{ref}} \hat{\mathbf{v}}_{\text{enc}}\|_2$: fraction of the `5120`-d `layers.49` direction (`vector.encoder`) that survives the 2-layer bidirectional `TokenRefiner` bottleneck (`s_1 = 1.665` down to `s_{5120} = 0.089`). Never estimated from `J_wiener @ v_pr`. |
| **Token-Wise Prompt Receptor Scan (`--prompt`)** | `cos(w_i, mu_steer)` per content word in target prompt | `cos(w_i, mu_steer)` from `pairs.b_side` + top-4 `positive_words` | Predicts which specific nouns/props in your scene prompt will absorb the steer in the DiT's 32 cross-attention blocks (`balloon` on `t2va_noodle_bar.txt` couples to `plastic (+0.223), bulb (+0.190)` while `customer (-0.043)` is orthogonal; `medieval` couples to `steel (+0.133), male (+0.112)`). |
| **Concept Type (`against_words`)** | Additive (`@neutral`, `0.35 * mu_neg`) vs. Replacement (`@prompt_subject` or `@replace_human`, `0.60 * mu_neg`) | Explicit `Prompt A` vs. `Prompt B` | **Additive concepts** (`blizzard`, `medieval`, `disco`, `8-bit`) fill an unoccupied attribute slot (`weather`, `lighting`, `era`) and work cleanly at `strength = 1.0 .. 1.25`. **Subject/medium replacement concepts** (`skeleton`, `balloon`, `muppet`, `claymation`) fight `"Live-action ... human / male customer"` tokens in the base prompt and benefit from `against_words="@prompt_subject"` (auto-extracting `[Shot 1]` medium + `<Subject 1>` + `on-screen`/`(S1)` character clauses via [`j_lens.py::extract_prompt_subject_words`](../../j_lens.py#L1228-L1305)) or `"@replace_human"` + `strength = 1.25 .. 1.55`. |

#### Measured Pre-Render Numbers vs. Actual Rendered Pixels (`Arms 53..65`)

| Arm & Concept | Build Method (`basis`) | `top1_gain` | `top3_mean` | `ref_pass` | Observed Visual & Audio Effect in Rendered Video (`141f`, `1152x768`) |
|---|---|---|---|---|---|
| **`63_noodle_blizzard_words`** | `words "blizzard, freezing, snow"` (`word_synthesis`) | **`+5.17`** | **`+4.49`** | **`0.572`** | Red neon sign turns icy cyan, snow covers street & cars, frost coats stall frame, raincoat becomes fur-hooded winter parka + howling wind. |
| **`64_lift_disco_words`** | `words "disco, glitter, dancing"` (`word_synthesis`) | **`+4.79`** | **`+4.57`** | **`0.585`** | Elevator ceiling becomes a radial disco light fixture with warm gold cove lighting and sparkly disco wardrobe. |
| **`65_noodle_medieval_words`** | `words "medieval, knight, armor, castle"` (`word_synthesis`) | **`+4.75`** | **`+4.23`** | **`0.585`** | Tiled stall becomes a cut-stone medieval archway, neon sign becomes an arched crest, raincoat becomes a hooded medieval cloak. |
| **`54_noodle_8bit_words`** | `words "8-bit, led, steampunk"` (`word_synthesis`) | **`+3.27`** | **`+2.93`** | **`0.538`** | Sign becomes a blue/pink pixel LED matrix, wall becomes a brass/mosaic grid with filament bulbs, translucent cyberpunk raincoat. |
| **`61_lift_skeleton_words`** | `words "skeleton, translucent, bones, skull"` (`word_synthesis`) | `+3.50` (`@replace_human`) | `+3.26` | `0.508` | Default `@neutral` at `s=1.15` left the human porter intact because `"Live-action ... porter"` anchored human skin; `@replace_human` shifts `-ve` pole to `person (-2.38), realistic (-2.12), people (-2.02), real (-1.90), human (-1.51)`. |
| **`53_noodle_balloon_words`** | `words "inflatable, balloon, glossy, latex"` (`word_synthesis`) | `+3.09` | `+2.90` | `0.497` | Lower `TokenRefiner` pass-through (`0.497 < 0.50`) + replacement concept at `s=1.15`: slight surface gloss change only. |
| **`55_balloon_comp` / `56_8bit_comp`** | `build --words` (`post_refiner`, 10-pair template compiler) | `+1.60` / `+1.47` | `+1.42` / `+1.38` | `0.369` / `0.340` | Template-clause dilution (`ref_pass = 0.34..0.37`); `55` curves the overhead sign into a bulb-lined neon arch while `56` stays visually near baseline (`43`). |

---

### Surgical Steering Controls on `MiniMaxH3ApplyControlVectors` (Wave 1 + Wave 2)

```text
[UNETLoader] ──► [MiniMaxH3LoRABranch (Pika)] ──► [MiniMaxH3ApplyControlVectors] ──► [BasicGuider]
```

In addition to `vector_name`, `strength`, `mode`, and `mask`, `MiniMaxH3ApplyControlVectors` provides five surgical steering controls verified in Wave 1 + Wave 2 (`Arms 57..66`):
1. **Inline `< 2 ms` Word Synthesis (`words`, `against_words`, `hold_words`, `hold_alpha`)**: Synthesizes and applies a word vector on the fly (or stacks it after `vector_name` when both are set).
2. **Inference-Time Cross-Vector Orthogonal Hold (`mode = "hold"`, `hold_alpha = 0.0 .. 1.0`)**: Chain a second `MiniMaxH3ApplyControlVectors` node with `mode = "hold"` to project out an unwanted nuisance vector from all preceding `add` vectors on the model (`Arm 59`: `cursed_cctv` + `hold(handheld_documentary_zoom)` kept 100% of the green fisheye CCTV look while scrubbing handheld camera shake).
3. **Sub-Section Prompt Masks (`mask = "description_only"` / `"soundscape_only"`)**: Steer only the visual `integrated_multimodal_description:` / `detailed_description:` section while leaving `overall_soundscape:` untouched (`Arm 57`), or steer only `overall_soundscape:` while keeping the visual scene locked (`Arm 58`).
4. **Velocity-Space Audio/Video Modality Split (`video_scale`, `audio_scale`, `split_end_percent`)**: Scales the steered DiT velocity delta $\Delta \mathbf{u}_t = \mathbf{u}_t^{\text{steered}} - \mathbf{u}_t^{\text{base}}$ separately on `video` (`video_scale`) and `audio` (`audio_scale`) up to `split_end_percent` (`default = 0.34`, steps 0–1 of 6-step Pika), letting you suppress visual drift on vocal/acoustic vectors (`video_scale = 0.0, audio_scale = 1.5` in `Arm 60`) at only `+2` extra DiT forwards per render.
5. **`ref2va` 3D VAE Reference-Latent Gate (`ref_vae_end_percent`, `ref_vae_aug`)**: On `ref2va`, `<Picture 1>` is injected both as Qwen vision tokens (`tag == 0`) and as 3D Video VAE `ref_latents`. Setting `ref_vae_end_percent = 0.50` (`Arm 66`) drops the 3D VAE reference anchor in the second half of denoising so radical style/material vectors (`felt_muppet`, `claymation_stopmotion`, `rubber_hose_1930s`) can reshape surface geometry after `<Subject 1>`'s identity and staging lock in on early steps.

---

## 2. Practical FAQ: Prompting, Pika LoRAs, `.toml` Fields & Pooling Modes

### Q1: Should Pole B describe a passive static state ("is floating") or an active physical trajectory ("inflates and balloons outward" / "drift upward and float in mid-air")?
**Write Pole B as a direct, active physical transformation and trajectory** (e.g., *"while his skull and cranium inflate and balloon outward to twice their normal size like an over-pumped helium sphere"* or *"while the brass letter opener, fountain pen, and inkpot drift weightlessly upward off the mahogany desk and float in mid-air"*), and **avoid generic onset/rate filler adverbs** (`slowly`, `gradually`, `steadily`, `begins to`, `starts to`, `rapidly`, `increasingly`):
- **Why active trajectory beats passive static state (`fl2va` / `i2v` / `t2va`)**: In `fl2va` and `i2v`, Frame 0 is anchored by an input image with normal proportions/gravity. Describing the active physical action and secondary motion (inflating, rising off the desk, rotating in mid-air, plus the matching sound cue) drives H3's 3D spatiotemporal attention blocks to animate the transition out of Frame 0.
- **Why you must avoid `"begins to"` / `"slowly"` / `"gradually"` filler ([`h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md#L141-L149))**: Repeating generic onset adverbs across all 10 Pole-B clauses causes `Qwen3-VL` `layers.49` to extract a generic *"slow/gradual onset"* direction instead of the physical effect itself. Use concrete domain verbs and nouns directly, and use `10/10` distinct diff clauses across the 10 pairs (`control_vectors.py::lint_pair_set` automatically checks for rate adverbs, negative target cancellation in `a_span`, cross-section diffs, and verbatim span cloning).

### Q2: How closely aligned do the subject and gender in the 10 pairs need to be to my final render prompt?
**They do NOT need to match your specific render subject—in fact, the 10 pairs SHOULD vary subjects, settings, and speaking/silent states across pairs 1–10** (while staying identical between Prompt A and Prompt B *within* each pair):
- Within each pair $i$, Prompt A and Prompt B share the exact same subject and setting, so when we compute $\Delta_i = \mathbf{r}(B_i) - \mathbf{r}(A_i)$, the subject/gender/setting tokens cancel out.
- Stratifying across 5 interior + 5 exterior settings and 6 speaking (`(S1)` + `<d>[English] ...</d>`) + 4 silent (`produces no vocal sound.` or `No people are present.`) pairs cancels subject-specific and speech-specific curvature in `Qwen3-VL` and `token_refiner`, leaving a pure effect direction that transfers cleanly to men, women, groups, or unpeopled shots.
- **Rule for `ref2va` ([`h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md#L49-L70))**: Even when building a control vector intended for `ref2va` (such as [`zero_gravity.toml`](../../prompt_bank/control_vectors/zero_gravity.toml)), the **10 Contrastive Pairs (`Pair 1..10`) always use the 3-section text format (`integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:`) with 10 unrelated characters/objects and NO `<Subject N>` or `<Picture N>` tags**. Only the `### Test Scene Prompt` at the top of the Markdown uses the 6-section `ref2va` format (`subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, `overall_soundscape:`, `non_diegetic_music:`) with `<Subject 1>` when testing on `ref2va`.

### Q3: Does it matter *which* Pika LoRA variant I use (`svd995`, `svd99`, `comfy_bf16`, `kijai`)?
**No—all four Pika LoRA variants within a family (`fl2va` or `ref2va`) produce virtually identical control vectors (`cosine = 0.999964 .. 1.000000`, `0.00% .. 0.05%` norm difference):**
- Why: Extracting a control vector only runs `condition_proj` + the 2-layer `token_refiner` (not the 50 DiT blocks), and `svd995`/`svd99` are 99.5%/99% SVD truncations of the exact same `comfy_bf16` weights.
- **What DOES matter**:
  1. **Model family (`fl2va` vs. `ref2va`)**: `fl2va` and `ref2va` checkpoints have completely different `condition_proj` + `token_refiner` weights (cross-family cosine is `~0.19`). Always use the `__fl2va__` vector on `t2v`/`fl2v`/`i2v` graphs and the `__ref2va__` vector on `ref2va` graphs (`python bench/control_vectors.py build` builds both automatically).
  2. **Pika LoRA vs. base UNet ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md#L9-L15))**: The Pika LoRA rotates `token_refiner` output by `< 0.43%` (`cos = 0.995730 .. 0.996678` vs. base UNet, with a `1.38%–2.10%` norm increase), and records the LoRA SHA256 in `recipe.loras`.
  3. **Same-family Pika LoRA variant tolerance (`control_vectors.py::_lora_family_signature`)**: Since `65d76c5d`, switching between Pika H3 Flex LoRA files of the same model family (`svd995`, `svd99`, `comfy_bf16`, `kijai_bf16`) is accepted automatically with an info log (`allow_recipe_mismatch` can stay `False`). *(Before `65d76c5d`, this paragraph said switching between `svd995` and `comfy_bf16`/`svd99` required `allow_recipe_mismatch = True`; you now only need `allow_recipe_mismatch = True` when mixing a base-UNet vector with a Pika LoRA or across unrelated LoRAs.)*

### Q4: What data in the `.toml` file is actually used during extraction vs. inference?
- **At inference time in ComfyUI (`MiniMaxH3ApplyControlVectors`)**: The `.toml` file is **not read at all**. ComfyUI only loads the `.safetensors` file and reads `vector.post_refiner` (`float32[5376]`) plus the `family` and `recipe` hashes in its JSON header.
- **At extraction time (`build` / `make`)**:

| `.toml` Field | Used in Vector Math? | What It Actually Does |
|---|---|---|
| `[[pair]].a`, `[[pair]].b` | **Yes (Core math)** | Encoded through `Qwen3-VL` (`layers.0..49`) -> `condition_proj` -> `token_refiner`, masked by `--pooling` (default `diff_span`), and subtracted ($\mathbf{r}_B - \mathbf{r}_A$). Validated for single-span diff (`differing_span`) and semantic quality (`lint_pair_set`). |
| `[[pair]].weight` | **Yes (Core math)** | Weights each pair in the weighted mean difference (`default = 1.0`). |
| `[[hold]].a`, `[[hold]].b`, `group` | **Yes (Optional math)** | If present (and `--no-holds` is not set), builds an orthonormal basis of unwanted nuisance directions and projects them out of `vector.post_refiner`. |
| `[vector].frame` | **Yes (Validation)** | `"scene"` (default) enforces the 3-section H3 prompt grammar (`integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:`). |
| `[vector].mode` | **Default UNet selector** | `"t2va"` or `"ref2va"` (auto-detected from `"subject_definitions:"` in [`control_vectors.py::parse_markdown_pairs`](../../control_vectors.py#L649-L654) when converting `.md`); used when `--families auto` or `make` without `--unet` is run. |
| `[vector].name` | **Filename stem** | Default output filename stem (`<name>__<family>__...safetensors`). |
| `[vector].quality`, `a_side`, `b_side` | **No (Metadata only)** | Human-readable descriptions copied into the `.safetensors` header and companion `.md` sidecar report. Changing these strings does **not** change a single float in the vector. |

---

## 3. How to Read Control Vector Scores & Choose the Right Variation

Every time you run `python bench/control_vectors.py build` or `make`, it writes a companion **`.md` sidecar** beside each `.safetensors` vector (via [`control_vectors.py::format_vector_sidecar_md`](../../control_vectors.py#L1649-L1850)) containing the vector's norms, recommended ComfyUI strengths, agreement cosines, and per-pair breakdown.

### 3.1 What Each Score Tells Us

| Score / Metric | Where It Appears | Healthy Target Range | What It Tells Us & How to Diagnose Outliers |
|---|---|---|---|
| **`raw_norm` (`post_refiner`)** | CLI output & `.md` sidecar (`Norms & Recommended Strength`) | `49.45 .. 87.03` (`diff_span`, mean `~62.5`), `~20–28` (`from_first_diff`), `~10–25` (`scene_text`) | The L2 length $\|\mathbf{v}_{\text{PR}}\|_2$ of the unnormalized post-refiner mean difference vector (`5376`-d). Because `apply_post_refiner` in `mode="add"` adds $\text{strength} \times \mathbf{v}_{\text{PR}}$ directly (`control_vectors.py:988-994`), `raw_norm` determines what `strength` knob value is needed to reach the effective steering regime ($\|\Delta \mathbf{r}_{\text{PR}}\| \approx 50\text{–}85$). |
| **`median_row_norm` (`post_refiner`)** | `.md` sidecar (`Norms & Recommended Strength`) | `~93–164` | The median L2 norm of individual post-refiner text rows across the pair set. Visible steering begins when $\text{strength} \times \text{raw\_norm}$ reaches `~30%–55%` of `median_row_norm` (`50..85`). |
| **`Recommended ComfyUI strength (50/norm .. 85/norm)`** | CLI output & `.md` sidecar | Auto-computed per file (`0.57 .. 1.72` across all 68 `diff_span` files) | The exact ComfyUI `strength` window that injects $\|\Delta \mathbf{r}_{\text{PR}}\| \in [50, 85]$ regardless of how many shared tokens diluted `raw_norm`. Start near the lower bound (`50 / raw_norm`) for subtle atmospheric/style shifts and near the upper bound (`85 / raw_norm`) for strong physical transformations (`head_swell`, `zero_gravity`). |
| **`mean_pairwise_cosine`** | CLI output & `.md` sidecar (`Agreement Summary`) | **`0.62 .. 0.85`** (with `10/10` unique diff spans) | Mean cosine similarity $\frac{2}{N(N-1)}\sum_{i<j}\cos(\Delta_i, \Delta_j)$ across all pair differences. **Two critical failure regimes**:<br>1. **`> 0.90` (Lexical-cloning warning)**: Almost always indicates that multiple pairs pasted the exact same wording in `a_span`/`b_span` (e.g., pre-fix `vocal_strain.toml` had `0.930` because all 10 pairs repeated the same 17-word string verbatim), baking lexical quirks into the vector.<br>2. **`< 0.55` (Cancellation or cross-section warning)**: Indicates either **target-noun cancellation** (e.g., pre-fix `lettering.toml` had `0.5415` because `Prompt A` said `"with no lettering, text, or painted signs"`, subtracting the target nouns out of $B - A$) or conflicting concepts across pairs. |
| **`split_half_cosine`** | CLI output & `.md` sidecar (`Agreement Summary`) | **`>= 0.90`** (`0.90 .. 0.97` typical) | Cosine similarity between the average of the odd-numbered pairs and the average of the even-numbered pairs. Measures whether `N = 10..14` pairs are sufficient for the mean vector to converge (`>= 0.90` means adding more pairs will barely rotate the direction). |
| **`per_pair_cosine` (`min` / `max` & Per-Pair Table)** | `.md` sidecar (`Agreement Summary` & `Per-Pair Breakdown`) | Every pair **`>= 0.50`** | Cosine of each individual pair $\Delta_i$ with the overall mean vector $\mathbf{v}_{\text{PR}}$. If 8 pairs have `0.70+` and 2 pairs drop to `0.29–0.45`, those 2 pairs almost certainly edited a **different prompt section** (e.g., editing `overall_soundscape:` while `P01–P08` edited `integrated_multimodal_description:`) or inverted `Prompt A` and `Prompt B`. |
| **`PC1 variance` / `flip_null` / `random_null`** | `grade` / `exl3_qwen3vl` report | `PC1 >= 60%`, `raw_norm > 2.5x flip_null` | Fraction of variance explained by the top principal component of $\{\Delta_i\}$, and comparison against sign-flipped (`flip_null`, which cancels the concept and measures residual scene noise) and random Gaussian (`random_null`) baselines. |
| **`hold_removed_fraction`** | `.md` sidecar (when `[[hold]]` pairs exist) | `0.02 .. 0.35` (`2%–35%`) | Fraction of the raw vector's energy projected out by the `[[hold]]` nuisance subspace (`control_vectors.py::hold_basis`). If `hold_removed_fraction > 0.60`, your hold pairs are too close to the target concept itself and are scrubbing out the main effect. |
| **`layers.49 -> lm_head` Contrastive Log-Odds** | [`h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md) | Domain nouns & verbs in top-15 `+ve` tokens | Passing `vector.encoder` (`5120`-d at `layers.49`) through `Qwen3-VL-32B-Instruct`'s remaining `layers.50..63` + `lm_head` reveals the exact vocabulary tokens the vector boosts before the DiT ever runs. If all 10 pairs share a single setting (e.g., stone arches) or repeat a filler adverb (`slowly`, `gradually`), those spurious words show up immediately in the top `+ve` tokens. |

---

### 3.2 Which Variation Works Best in Which Scenario?

#### A. Pooling Modes (`--pooling diff_span` vs. `from_first_diff` vs. `scene_text`)

Suppose Pair 1 changes **one clause** near the end of `integrated_multimodal_description:` (say tokens `104..122` out of `155` total tokens):

```text
Token idx:  [0 ................. 103] [104 .............. 122] [123 ....................... 154]
Content:    Shared scene setup prefix  CHANGED CLAUSE (A vs B)  Shared soundscape + music suffix
            (identical in A and B!)    (differs in A and B)     (identical text, but attended to
                                                                 104..122 via causal Qwen3-VL!)
```

Because `Qwen3-VL-32B-Instruct` (`layers.0..49`) is **strictly causal**, tokens `0..103` before `first_diff` (`104`) have **identical** `layers.49` hidden states in Prompt A and Prompt B (and near-zero difference after the 2-layer bidirectional `token_refiner`). Averaging over those `104` unchanged prefix tokens dilutes the vector's norm (`raw_norm`) by `2.1x–8.2x`!

| Pooling Mode | Exact Token Mask Pooled | Diluted by Shared Prefix/Suffix? | Measured `raw_norm` (`head_swell` / `zero_gravity`) | Recommended ComfyUI `strength` (`50/norm .. 85/norm`) | Scenario & When to Use One Over Another |
|---|---|---|---|---|---|
| **`diff_span`** *(Default in `build`, `make`, & node)* | `[first_diff:end_diff]` within `span_masks["scene"]` (falls back to `~span_masks["vision"]` if diff is inside `<d>...</d>`) ([`h3_spans.py::differing_token_bounds`](../../h3_spans.py#L361-L381)) | **0% shared tokens** (strips both shared prefix and shared suffix) | **`76.04`–`85.01` / `84.32`–`87.03`** (`~7.3x–8.2x` stronger than `scene_text`) | **`0.57 .. 1.13`** (`s = 0.97` in `arm 41`, `s = 0.93` in `arm 42`) | **Default across all 34 banks**: isolates the exact changed clause (`a_span` vs. `b_span`) and places all 68 built vectors in a tight `49.45 .. 87.03` norm band so unit-scale ComfyUI strengths (`0.60 .. 1.50`) work across every vector in the repo. |
| **`from_first_diff`** | `[first_diff:]` within `~span_masks["vision"]` (strips shared prefix `[:first_diff]`, keeps changed clause + trailing soundscape/music tokens) ([`h3_spans.py::first_differing_index`](../../h3_spans.py#L343-L358)) | Strips prefix; keeps trailing suffix tokens that causally attended to the diff | **`23.57` / `22.62`** (`~2.12x–2.28x` stronger than `scene_text`) | **`2.10 .. 3.75`** (`s = 3.50` in `arms 39..40`) | **Best when you want extra cross-modal visual + ambient soundscape coupling**: because trailing `overall_soundscape:` tokens in `Qwen3-VL` causally attended to the changed visual clause, keeping `[first_diff:]` blends the exact diff span with the scene-level summary accumulated on the trailing tokens (`cos(ffd, ds) = 0.86–0.88`). |
| **`scene_text`** | All `span_masks["scene"]` rows across the entire prompt (excludes vision, `<Picture N>` labels, timestamps, `(S1)`, `<d>...</d>`) | **Yes (`70%–85%` shared tokens)** if only one clause changed | **`10.35` / `10.65`** on single-clause; **`20.40`–`24.54`** on multi-clause | **`4.70 .. 8.20`** (single-clause) or **`1.80 .. 3.00`** (multi-clause) | Legacy full-scene mean-pooling; use `mode="add_norm_matched"` (`strength = 1.0`) if comparing a `scene_text` vector directly against a `diff_span` vector. |
| **`all_tag1`** | Every non-vision (`tag == 1`) row, including `<d>...</d>` dialogue and `(S1)` speaker tags | **Yes** | Diagnostic only | Diagnostic only | Only for testing whether dialogue tokens carry the concept (`E2.6` ablation). |

> **Measured GPU Verification (`diff_span` vs. `from_first_diff` vs. `scene_text`, [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md#L113-L120))**: At matched post-refiner perturbation norm $\|\Delta \mathbf{r}_{\text{PR}}\| = \text{strength} \times \text{raw\_norm} \approx 79\text{–}85$, `head_swell` produces the same `~1.70x` cranial expansion + pressurized steam hiss across `scene_text` (`s = 8.00`, `arm 36`), `from_first_diff` (`s = 3.50`, `arm 39`), and `diff_span` (`s = 0.97`, `arm 41`), while on `zero_gravity` (`ref2va`), both `from_first_diff` (`s = 3.50`, `arm 40`) and `diff_span` (`s = 0.93`, `arm 42`) produce **unassisted mid-air levitation** of tabletop utensils/plates without `<Subject 1>` having to hold them all in his hands (`arm 33`).

#### B. Application Mode (`mode`), Row Mask (`mask`), and Step Schedule (`start_percent` / `end_percent`)

| Knob on `MiniMaxH3ApplyControlVectors` | Options | When to Use Each Option |
|---|---|---|
| **`mode`** | **`add`** *(default)* | Shifts every selected conditioning row by $+s \cdot \mathbf{v}_{\text{PR}}$ (or $-s \cdot \mathbf{v}_{\text{PR}}$ when $s < 0$). Use for standard steering—both adding an effect ($s > 0$) and pushing toward the opposite pole ($s < 0$, e.g., `lettering` at `s = -1.0` in `arm 08` to scrub street signage clean without altering building architecture). |
| | **`add_norm_matched`** | Shifts every selected conditioning row by $+s \cdot (\text{NORM\_MATCHED\_TARGET} / \|\mathbf{v}_{\text{PR}}\|_2) \cdot \mathbf{v}_{\text{PR}}$, where `NORM_MATCHED_TARGET = 67.5` ([`control_vectors.py:926`](../../control_vectors.py#L926), the midpoint of the `[50, 85]` effective post-refiner perturbation regime). In `add_norm_matched` mode, `strength = 1.0` always injects a `67.5` post-refiner shift norm regardless of whether the vector was built with `diff_span`, `from_first_diff`, or `scene_text`. |
| | **`hold`** | Projects out the vector's direction (with soft ridge attenuation `hold_alpha in [0, 1]`, default `1.0`) from all preceding `add` / `add_norm_matched` control vectors attached to the model (`arm 59`). |
| | **`project_out`** | Removes the component of each conditioning row parallel to $\hat{\mathbf{v}}_{\text{PR}}$: $\mathbf{r} \leftarrow \mathbf{r} - s (\mathbf{r} \cdot \hat{\mathbf{v}}_{\text{PR}})\hat{\mathbf{v}}_{\text{PR}}$ (`s = 1.0` for full orthogonal projection). Use when a prompt naturally triggers an unwanted quality (e.g., `noir` `project_out` at `s = 1.0` in `arm 24` strips dark/sooty contrast from a shadowed street prompt without injecting an artificial opposite style). |
| | **`amplify`** | Scales the existing projection of each conditioning row along $\hat{\mathbf{v}}_{\text{PR}}$: $\mathbf{r} \leftarrow \mathbf{r} + s (\mathbf{r} \cdot \hat{\mathbf{v}}_{\text{PR}})\hat{\mathbf{v}}_{\text{PR}}$. Use when you only want to boost tokens that *already* align with the vector's concept rather than shifting every token uniformly. |
| **`mask`** | **`non_vision`** *(default)* | Applies the vector to all text conditioning tokens (`tag == 1`) while leaving reference-image vision tokens (`tag == 0`) bitwise untouched (`0.0` diff). **Standard choice for both `t2va` and `ref2va`** so `<Subject N>` facial identity and wardrobe stay locked (`arm 18` vs. `arm 19`). |
| | **`scene_only`** | Applies the vector only to scene-description text rows (`span_masks["scene"]` attached via `_h3_span_masks` on `minimax_token_tags`), leaving vision tokens (`tag == 0`), `<d>...</d>` dialogue tokens, and `(S1)` speaker tags untouched. Use when a strong physical/material vector would otherwise distort spoken vowel articulation or voice timbre. |
| | **`description_only`** | Applies the vector only to the visual description section (`span_masks["description"]`: `integrated_multimodal_description:` or `detailed_description:`), leaving `overall_soundscape:`, `non_diegetic_music:`, `<d>...</d>` dialogue, and vision tokens untouched (`arm 57`). |
| | **`soundscape_only`** | Applies the vector only to the `overall_soundscape:` section (`span_masks["soundscape"]`), leaving the visual scene description, dialogue, and vision tokens untouched (`arm 58`). |
| | **`after_dialogue`** | Applies the vector only to non-vision tokens strictly **after** the last `<d>...</d>` dialogue span (`span_masks["after_dialogue"]`). Use for post-speech physical transformations (`head_swell`, `zero_gravity`, `smoke_dissolve`) so the character's pre-speech setup and spoken line stay unperturbed while the post-speech tokens carry the transformation. |
| | **`all_rows`** | Shifts both text tokens (`tag == 1`) and Qwen reference-image vision tokens (`tag == 0`). Only use on `ref2va` if you intentionally want to alter the reference character's facial expression or texture (note: 3D Video VAE `ref_latents` still anchor core identity via the latent stream). |
| | **`vision_only`** | Shifts only reference-image vision tokens (`tag == 0`). Diagnostic use for isolating Qwen vision-token contributions on `ref2va`. |
| **`start_percent` .. `end_percent`** | **`0.0 .. 1.0`** *(default)* | Applies the control vector across all diffusion steps. Necessary when the vector needs to alter **3D geometry, object trajectory, or period architecture** (e.g., `zero_gravity` levitation, `head_swell` inflation, or `film_16mm_1974` morphing a modern bus into a 1970s Routemaster in `arm 04`). |
| | **`0.33 .. 1.0`** *(late window)* | Skips the first third of denoising steps (e.g., steps 1–2 of a 6-step Pika render) so **100% of the baseline scene layout, camera framing, and object trajectories lock in first**, then applies the vector's color grading, film grain, or surface texture on steps 3–6 (`arm 23`, where `film_16mm_1974` kept the exact baseline street and car while adding 16mm color/grain). |
| | **`0.0 .. 0.50`** *(early window)* | Applies the vector only during coarse layout/motion formation and lets the unsteered model refine sharp high-frequency details in the second half of denoising. |

---

## 4. Code Architecture & File Map

| File | Responsibility | Key Entry Points |
|---|---|---|
| [`j_lens.py`](../../j_lens.py) | Analytical Jacobian Lens (`J-Lens`), `< 2 ms` CPU word-to-vector synthesis (`synthesize_from_words`), pre-render steerability & readiness predictor (`assess_steering_readiness`), Token-Wise Prompt Receptor Scan (`analyze_prompt_coupling`), `@prompt_subject` subject/medium extractor (`extract_prompt_subject_words`), `AGAINST_PRESETS` (`@replace_human`), soft ridge holds (`apply_soft_holds`), and multi-vector recipe decoder | `load_j_lens_bundle`, `decode_vector`, `decode_safetensors`, `assess_steering_readiness`, `analyze_prompt_coupling`, `extract_prompt_subject_words`, `synthesize_from_words`, `save_synthesized_vector`, `decode_recipe`, `AGAINST_PRESETS` |
| [`h3_pair_compiler.py`](../../h3_pair_compiler.py) | Deterministic 10-scene H3 contrastive pair compiler (6 speaking + 4 silent, 5 interior + 5 exterior) for turning short `Pole A -> Pole B` phrases into validated `PairSet`s | `compile_words_to_pairset`, `compile_short_contrast_to_toml`, `compile_placebo_pairset`, `infer_slot` |
| [`h3_spans.py`](../../h3_spans.py) | Deterministic token-span partition of H3 conditioning sequences (`vision`, `label`, `timestamp`, `dialogue`, `speaker`, `scene`, `description`, `soundscape`, `music`, `after_dialogue`) and pooling masks | `spans`, `span_masks_from_prompt`, `differing_token_bounds`, `first_differing_index`, `pooling_mask` (`POOLING_MODES = ("scene_text", "all_tag1", "from_first_diff", "diff_span")`) |
| [`control_vectors.py`](../../control_vectors.py) | Core float64 math, `.toml` pair loader, semantic pair-set linter (`lint_pair_set`), Markdown-to-`.toml` converter (including `#### Hold N`), `.safetensors` + `.md` sidecar I/O, model family & recipe verification (`check_vector_compatibility`, `_lora_family_signature`), and `apply_post_refiner` | `load_pairs`, `differing_span` (`MAX_SHARED_INTERIOR_WORDS = 5`), `lint_pair_set`, `parse_markdown_pairs`, `write_markdown_to_toml`, `mean_difference` (`unit_mean`), `flip_null`, `random_null`, `hold_basis`, `project_out`, `apply_post_refiner` (`APPLY_MODES = ("add", "add_norm_matched", "project_out", "amplify", "hold")`, `NORM_MATCHED_TARGET = 67.5`), `save_vector`, `format_vector_sidecar_md`, `load_vector`, `check_vector_compatibility` |
| [`control_vector_nodes.py`](../../control_vector_nodes.py) | ComfyUI V3 nodes, runtime `WrappersMP.DIFFUSION_MODEL` hook, bare-stem & cross-family sibling resolution (`resolve_vector_file_path`), inline word synthesis (`words`, `against_words`, universal `hold_words`), Step-0 `_h3_tail_text` `@prompt_subject` resolution, velocity-space audio/video split, `ref2va` 3D VAE gate, and media contrast extraction | `MiniMaxH3MakeControlVector`, `MiniMaxH3MakeMediaControlVector`, `MiniMaxH3ApplyControlVectors`, `MiniMaxH3ControlVector`, `MiniMaxH3LayerNormMatch`, `attach_control_vector`, `resolve_vector_file_path`, `make_control_vector_from_pairs`, `make_control_vector_from_media`, `build_refiner_runner` |
| [`bench/control_vectors.py`](../../bench/control_vectors.py) | Standalone CLI and on-demand local UI server (`words`, `ui` with `/api/j-lens` & cached encoder/refiners, one-shot `build` / `build --from-words` / `build --words` / `build --all`, `from-md`, `make`, `inspect`, and `grade`) | `cmd_words`, `cmd_ui`, `ControlVectorUIState`, `handle_ui_request`, `cmd_build`, `cmd_from_md`, `cmd_make`, `cmd_inspect`, `cmd_grade` |
| [`bench/control_vector_ui.html`](../../bench/control_vector_ui.html) | Self-contained dark-mode browser UI served by `python bench/control_vectors.py ui` with `< 2 ms` J-Lens Pre-Render Readiness, Token-Wise Prompt Receptor Scan, 1-click Instant Build from Words, pair diff & BPE token-span inspector, and 24fps A/B video player | Served at `GET /` by `bench/control_vectors.py::handle_ui_request` |
| [`workflows/build_workflows.py`](../../workflows/build_workflows.py) | Programmatic workflow builders for single/chained control-vector arms, 6-arm gate ladders, and E4.2/E4.3 probes | `build_control_vector_arm`, `build_control_vector_gate_arms`, `build_control_vector_probe_arm`, `build_api(..., control_vector=...)` |
| [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py) | Tracked CPU check (`C1`–`C7`) defending tokenizer constants, span masks, `.toml`/`.md` refusals, `lint_pair_set`, float64 math, `.safetensors` + `.md` sidecar I/O, runtime hooks, J-Lens readiness & word synthesis, and CLI subcommands | `main` (`python bench/check_control_vectors.py`) |

---

## 5. Shipped Contrastive Pair Bank (`prompt_bank/control_vectors/*.toml`, 32 Vectors / 64 Built `diff_span` Files)

All **32 shipped `.toml` files** follow the stratified authoring standard in [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md): `100%` unique `a_span` and `b_span` clauses (`0` verbatim span cloning), `<d>[English] ...</d>` language tags on all dialogue, explicit `produces no vocal sound` on silent subjects, `non_diegetic_music: N/A` (`10/10` per bank), balanced interior/exterior and speaking/silent scene regimes, and zero rate adverbs (`slowly`, `gradually`, `begins to`). All 64 `diff_span` `.safetensors` files (`32` `fl2va` + `32` `ref2va`, plus 64 `.md` sidecars) are pre-built via `python bench/control_vectors.py build --all --overwrite`. *(Until 2026-11-06 this bank also included two early `E2.3` bare-frame ablation twins, `film_16mm_1974_bare.toml` and `lettering_bare.toml`, which were retired on 2026-11-06.)*

### 5.1 Physical, Anatomical & Object-Physics Transformations (6 Files)

| Pair File | Concept / Pole A -> Pole B | Pairs / Holds | `fl2va` `diff_span` (`norm`, `sh`, `rec s`) | `ref2va` `diff_span` (`norm`, `sh`, `rec s`) |
|---|---|---|---|---|
| [`prompt_bank/control_vectors/head_swell.toml`](../../prompt_bank/control_vectors/head_swell.toml) | Normal head proportions -> Cranium inflates and balloons to 2x size like an over-pumped sphere + latex creak & steam hiss | 10 pairs + 2 holds (`pressurized_steam_acoustics`) | `76.04`, `sh=0.9180`, **`0.66..1.12`** | `75.21`, `sh=0.9148`, **`0.66..1.13`** |
| [`prompt_bank/control_vectors/zero_gravity.toml`](../../prompt_bank/control_vectors/zero_gravity.toml) | Normal 1g gravity -> Weightlessness as loose tabletop objects detach and float into mid-air + low hum | 10 pairs | `87.03`, `sh=0.9379`, **`0.57..0.98`** | `84.32`, `sh=0.9362`, **`0.59..1.01`** |
| [`prompt_bank/control_vectors/elastic_neck_stretch.toml`](../../prompt_bank/control_vectors/elastic_neck_stretch.toml) | Standard anatomical neck height -> Neck extends vertically upward like a telescoping periscope above stationary shoulders | 10 pairs | `63.07`, `sh=0.9123`, **`0.79..1.35`** | `59.97`, `sh=0.9038`, **`0.83..1.42`** |
| [`prompt_bank/control_vectors/magnetic_attraction.toml`](../../prompt_bank/control_vectors/magnetic_attraction.toml) | Metal tools/cutlery resting undisturbed -> Electromagnetic pull yanking metal tools, cutlery, and keys inward to snap/orbit around subject + transformer hum | 10 pairs | `65.38`, `sh=0.9377`, **`0.76..1.30`** | `62.00`, `sh=0.9359`, **`0.81..1.37`** |
| [`prompt_bank/control_vectors/fever_melt.toml`](../../prompt_bank/control_vectors/fever_melt.toml) | Solid rigid architectural geometry -> Walls, furniture, and props softening and sagging like melting wax with oil-slick halation and pitch-bending drone | 10 pairs | `65.09`, `sh=0.9266`, **`0.77..1.31`** | `62.68`, `sh=0.9245`, **`0.80..1.36`** |
| [`prompt_bank/control_vectors/smoke_dissolve.toml`](../../prompt_bank/control_vectors/smoke_dissolve.toml) | Intact solid shoulders and limbs -> Billowing charcoal smoke and glowing orange embers shedding from shoulders and limbs into swirling plumes + crackling hearth foley | 10 pairs | `67.38`, `sh=0.9326`, **`0.74..1.26`** | `65.47`, `sh=0.9310`, **`0.76..1.30`** |

### 5.2 Film Stocks, Optical Formats & Camera Rig Styles (8 Files)

| Pair File | Concept / Pole A -> Pole B | Pairs / Holds | `fl2va` `diff_span` (`norm`, `sh`, `rec s`) | `ref2va` `diff_span` (`norm`, `sh`, `rec s`) |
|---|---|---|---|---|
| [`prompt_bank/control_vectors/film_16mm_1974.toml`](../../prompt_bank/control_vectors/film_16mm_1974.toml) | Clean modern digital color -> 1974 16mm reversal film stock, halation, gate weave, optical hiss | 14 pairs + 6 holds (`warm_colour_cast`, `soft_focus`) | `54.07`, `sh=0.9668`, **`0.92..1.57`** | `52.86`, `sh=0.9657`, **`0.95..1.61`** |
| [`prompt_bank/control_vectors/noir.toml`](../../prompt_bank/control_vectors/noir.toml) | Balanced daylight/tungsten color -> High-contrast 1940s 35mm silver-nitrate B&W film noir with deep inky shadows and stark silver highlights | 12 pairs | `52.72`, `sh=0.9171`, **`0.95..1.61`** | `51.35`, `sh=0.9187`, **`0.97..1.66`** |
| [`prompt_bank/control_vectors/cursed_cctv.toml`](../../prompt_bank/control_vectors/cursed_cctv.toml) | Clean rectilinear eye-level cinema camera -> Ceiling-mounted 2004 fisheye CCTV barrel distortion, green phosphor monochrome tint, macroblocks, and 60Hz hum | 10 pairs | `63.26`, `sh=0.9251`, **`0.79..1.34`** | `62.98`, `sh=0.9243`, **`0.79..1.35`** |
| [`prompt_bank/control_vectors/vhs_1991_camcorder.toml`](../../prompt_bank/control_vectors/vhs_1991_camcorder.toml) | Clean progressive digital sensor -> 1991 home-video VHS camcorder tape degradation with chroma bleed, head-switching snow, CRT scanlines, and warbling tape hiss | 10 pairs | `66.25`, `sh=0.9488`, **`0.75..1.28`** | `63.62`, `sh=0.9441`, **`0.79..1.34`** |
| [`prompt_bank/control_vectors/thermal_infrared.toml`](../../prompt_bank/control_vectors/thermal_infrared.toml) | Natural visible-spectrum photography -> False-color FLIR thermographic infrared imaging mapping warm bodies to white-yellow-orange-magenta and cold surfaces to indigo | 10 pairs | `57.09`, `sh=0.9336`, **`0.88..1.49`** | `56.79`, `sh=0.9360`, **`0.88..1.50`** |
| [`prompt_bank/control_vectors/miniature_tiltshift.toml`](../../prompt_bank/control_vectors/miniature_tiltshift.toml) | Full-scale deep-focus cinema optics -> Macro tilt-shift diorama optics with razor-thin focal plane, painted hobby-train miniature textures, and tiny clockwork acoustics | 10 pairs | `59.80`, `sh=0.9162`, **`0.84..1.42`** | `58.44`, `sh=0.9175`, **`0.86..1.45`** |
| [`prompt_bank/control_vectors/wes_anderson_symmetry.toml`](../../prompt_bank/control_vectors/wes_anderson_symmetry.toml) | Naturalistic off-center 3/4 framing -> Deadpan planimetric center-framed pastel symmetry, flat doll-house orthographic staging, and harpsichord/plucked-string cues | 10 pairs | `52.06`, `sh=0.8839`, **`0.96..1.63`** | `51.35`, `sh=0.8866`, **`0.97..1.66`** |
| [`prompt_bank/control_vectors/handheld_documentary_zoom.toml`](../../prompt_bank/control_vectors/handheld_documentary_zoom.toml) | Locked-off fluid-head tripod stability -> Raw verite shoulder-rig handheld urgency with sudden snap-zoom/whip-pan reframing and brief lens focus-hunt breathing | 10 pairs | `54.39`, `sh=0.9141`, **`0.92..1.56`** | `53.97`, `sh=0.9113`, **`0.93..1.58`** |

### 5.3 Material, Puppet, Animation & Surface Replacement Styles (6 Files)

| Pair File | Concept / Pole A -> Pole B | Pairs / Holds | `fl2va` `diff_span` (`norm`, `sh`, `rec s`) | `ref2va` `diff_span` (`norm`, `sh`, `rec s`) |
|---|---|---|---|---|
| [`prompt_bank/control_vectors/felt_muppet.toml`](../../prompt_bank/control_vectors/felt_muppet.toml) | Live-action human anatomy & voice -> Hand-stitched fuzzy felt puppet with bulging ping-pong-ball eyes, hinged felt mouth, and squeaky puppet delivery | 10 pairs | `67.94`, `sh=0.9278`, **`0.74..1.25`** | `64.97`, `sh=0.9247`, **`0.77..1.31`** |
| [`prompt_bank/control_vectors/rubber_hose_1930s.toml`](../../prompt_bank/control_vectors/rubber_hose_1930s.toml) | Live-action skeletal articulation -> 1930s pie-eyed ink-and-cel rubber-hose cartoon physics with boneless noodle-limb bounce and slapstick xylophone/slide-whistle foley | 10 pairs | `77.43`, `sh=0.9348`, **`0.65..1.10`** | `74.55`, `sh=0.9329`, **`0.67..1.14`** |
| [`prompt_bank/control_vectors/claymation_stopmotion.toml`](../../prompt_bank/control_vectors/claymation_stopmotion.toml) | Fluid 24fps live-action motion & skin -> Hand-sculpted thumbprinted plasticine claymation stop-motion at stepped 12fps armature movement + clay-squish foley | 10 pairs | `66.77`, `sh=0.9240`, **`0.75..1.27`** | `65.69`, `sh=0.9252`, **`0.76..1.29`** |
| [`prompt_bank/control_vectors/paper_origami.toml`](../../prompt_bank/control_vectors/paper_origami.toml) | Rounded organic anatomy & woven cloth -> Crisp folded geometric origami paper construction with visible crease lines, matte kraft/washi planes, and paper-folding rustle | 10 pairs | `58.31`, `sh=0.8619`, **`0.86..1.46`** | `56.63`, `sh=0.8618`, **`0.88..1.50`** |
| [`prompt_bank/control_vectors/bioluminescent_glow.toml`](../../prompt_bank/control_vectors/bioluminescent_glow.toml) | Matte non-emissive skin & surfaces -> Deep-sea neon cyan and ultraviolet bioluminescent vein networks and drifting spore luminescence pulsing in darkened ambiance | 10 pairs | `60.80`, `sh=0.9282`, **`0.82..1.40`** | `59.21`, `sh=0.9301`, **`0.84..1.44`** |
| [`prompt_bank/control_vectors/lettering.toml`](../../prompt_bank/control_vectors/lettering.toml) | Plain unadorned material surfaces (canvas, enamel, timber, slate, brick) -> Surfaces covered in printed lettering, painted signs, and bold text | 14 pairs | `55.37`, `sh=0.8709`, **`0.90..1.54`** | `54.07`, `sh=0.8724`, **`0.92..1.57`** |

### 5.4 Environmental, Weather & Lighting Regimes (5 Files)

| Pair File | Concept / Pole A -> Pole B | Pairs / Holds | `fl2va` `diff_span` (`norm`, `sh`, `rec s`) | `ref2va` `diff_span` (`norm`, `sh`, `rec s`) |
|---|---|---|---|---|
| [`prompt_bank/control_vectors/underwater_submerged.toml`](../../prompt_bank/control_vectors/underwater_submerged.toml) | Dry terrestrial air & gravity-draped hair/cloth -> Deep-water submersion with shimmering caustic light ripples, buoyant hair/cloth suspension, rising bubble streams, and muffled hydrophone acoustics | 10 pairs | `56.79`, `sh=0.9101`, **`0.88..1.50`** | `55.72`, `sh=0.9094`, **`0.90..1.53`** |
| [`prompt_bank/control_vectors/wind_tunnel_gale.toml`](../../prompt_bank/control_vectors/wind_tunnel_gale.toml) | Calm still air & vertically draped garments -> Hurricane-force horizontal gale blasting hair, clothing flaps, and flying debris backward against forward lean + roaring wind-shear acoustics | 10 pairs | `53.78`, `sh=0.9193`, **`0.93..1.58`** | `51.66`, `sh=0.9198`, **`0.97..1.65`** |
| [`prompt_bank/control_vectors/crystallize_ice.toml`](../../prompt_bank/control_vectors/crystallize_ice.toml) | Temperate dry skin, fabric, and surfaces -> Translucent blue frost and jagged ice-crystal encrustation spreading across subject and surfaces + brittle glacial cracking sounds | 10 pairs | `62.38`, `sh=0.9115`, **`0.80..1.36`** | `60.81`, `sh=0.9124`, **`0.82..1.40`** |
| [`prompt_bank/control_vectors/cyberpunk_neon_rain.toml`](../../prompt_bank/control_vectors/cyberpunk_neon_rain.toml) | Dry balanced natural illumination -> Neon-drenched cyberpunk night cinematography with glowing magenta/cyan tube signage, mirror-wet asphalt reflections, and slashing rain streaks | 10 pairs | `58.08`, `sh=0.9195`, **`0.86..1.46`** | `55.87`, `sh=0.9161`, **`0.89..1.52`** |
| [`prompt_bank/control_vectors/strobe_club_pulse.toml`](../../prompt_bank/control_vectors/strobe_club_pulse.toml) | Continuous steady ambient lighting -> High-contrast rhythmic Xenon strobe-light flashes alternating stark white illumination and pitch-black shadow with pounding sub-bass club thump | 10 pairs | `69.13`, `sh=0.9358`, **`0.72..1.23`** | `68.91`, `sh=0.9364`, **`0.73..1.23`** |

### 5.5 Behavioral, Vocal & Acoustic Delivery (7 Files)

| Pair File | Concept / Pole A -> Pole B | Pairs / Holds | `fl2va` `diff_span` (`norm`, `sh`, `rec s`) | `ref2va` `diff_span` (`norm`, `sh`, `rec s`) |
|---|---|---|---|---|
| [`prompt_bank/control_vectors/chaos_panic.toml`](../../prompt_bank/control_vectors/chaos_panic.toml) | Calm unhurried posture & peaceful room tone -> Frantic wide-eyed panic, trembling urgency, violent handheld camera shake, and chaotic emergency commotion | 10 pairs | `64.23`, `sh=0.9266`, **`0.78..1.32`** | `61.88`, `sh=0.9234`, **`0.81..1.37`** |
| [`prompt_bank/control_vectors/infomercial_fail.toml`](../../prompt_bank/control_vectors/infomercial_fail.toml) | Smooth effortless physical coordination -> Exaggerated infomercial-style clumsy fumbling, losing grip, dropping objects, and messy spills | 10 pairs | `55.94`, `sh=0.8612`, **`0.89..1.52`** | `53.84`, `sh=0.8580`, **`0.93..1.58`** |
| [`prompt_bank/control_vectors/reverb_cathedral.toml`](../../prompt_bank/control_vectors/reverb_cathedral.toml) | Dry close-miked acoustics with immediate decay -> Vast echoing stone cathedral acoustics with a long cavernous reverb tail | 12 pairs | `67.50`, `sh=0.9540`, **`0.74..1.26`** | `67.17`, `sh=0.9533`, **`0.74..1.27`** |
| [`prompt_bank/control_vectors/vocal_strain.toml`](../../prompt_bank/control_vectors/vocal_strain.toml) | Calm conversational delivery with relaxed jaw -> Strained, raspy, vein-bulging full-lung screaming with jaw stretched wide | 10 pairs | `65.47`, `sh=0.9071`, **`0.76..1.30`** | `63.99`, `sh=0.9058`, **`0.78..1.33`** |
| [`prompt_bank/control_vectors/whisper_asmr.toml`](../../prompt_bank/control_vectors/whisper_asmr.toml) | Normal full-voiced room projection -> Ultra-intimate close-mic breathy ASMR whisper delivery with delicate lip/consonant transients and hushed airflow | 10 pairs | `51.09`, `sh=0.9015`, **`0.98..1.66`** | `49.45`, `sh=0.8946`, **`1.01..1.72`** |
| [`prompt_bank/control_vectors/auctioneer_rapid_patter.toml`](../../prompt_bank/control_vectors/auctioneer_rapid_patter.toml) | Unhurried everyday speech cadence -> Breathless high-velocity auctioneer rapid-fire vocal patter with machine-gun syllable articulation and brisk rhythmic pointing | 10 pairs | `60.40`, `sh=0.9165`, **`0.83..1.41`** | `57.74`, `sh=0.9126`, **`0.87..1.47`** |
| [`prompt_bank/control_vectors/operatic_bellow.toml`](../../prompt_bank/control_vectors/operatic_bellow.toml) | Plain spoken conversational vocal delivery -> Full-chest vibrating operatic bellow with wide resonant jaw opening, dramatic diaphragm posture, rich vibrato, and ringing concert-hall acoustics | 10 pairs | `65.49`, `sh=0.9354`, **`0.76..1.30`** | `65.16`, `sh=0.9381`, **`0.77..1.30`** |

### Output Filename Convention (`models/control_vectors/`)
[`control_vectors.py::default_vector_filename`](../../control_vectors.py#L1605-L1638) names every vector so different model families, pooling modes, and LoRA recipes never overwrite each other:
```text
<stem>__<family>[__<pooling>][__ref_<8hex>][__noholds][__lora_<8hex>].safetensors
```
Every `.safetensors` file is accompanied by an auto-generated **`.md` sidecar** (`<same_stem>.md`) containing its recipe table, recommended ComfyUI `strength` range, agreement cosines (`mean_pairwise_cosine`, `split_half_cosine`), **"How to Read These Scores"** guide, and per-pair token spans.

---

## 6. Workflows & Physical-Action Best Practices (`t2va`, Dual-Vector, and `ref2va`)

### 6.1 Single & Chained Dual Control Vectors on `t2va` / `fl2va`
- **Single Control Vector Workflow**: [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json)
  - Wiring: `MiniMaxH3LoRABranch (18, pika_h3_flex_fl2va_svd995_bf16.safetensors) -> MiniMaxH3SigmaShift (19, shift_video=12.0, shift_audio=3.0) -> ModelAttentionBackend (58, comfy kitchen attention) -> MiniMaxH3Sol (21) -> MiniMaxH3ApplyControlVectors (115, film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors, strength=1.25) -> BasicGuider (9)`. *(Until 2026-11-06 this bullet described nodes 19 and 58 as `ModelSamplingSD3 (19, shift=5.0) -> ModelAttentionBackend (58, triton)`.)*
- **Chained Dual Control Vectors Workflow**: [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json)
  - Chains two `MiniMaxH3ApplyControlVectors` nodes in series (`"115"` `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` at `1.0` $\to$ `"125"` `chaos_panic__fl2va__diff_span__lora_2a8c2b52.safetensors` at `1.0` $\to$ `BasicGuider (9)`). Consecutive same-mask `add` vectors accumulate in `float64` while allowing independent `strength`, `mode` (`add`, `add_norm_matched`, `project_out`, `amplify`), `mask` (`non_vision`, `scene_only`, `after_dialogue`, `all_rows`, `vision_only`), and step schedules (`start_percent`, `end_percent`).

### 6.2 Character-Reference Control Vectors on `ref2va`
- **Shipped `ref2va` Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json)
  - Uses `zero_gravity__ref2va__diff_span__lora_865ee189.safetensors` (`strength = 1.0`) and `mask="non_vision"` (default) so `<Picture 1>`'s vision patch embeddings (`tag == 0`) remain bitwise identical (`0.0` diff), locking `<Subject 1>`'s face and wardrobe while steering the scene via text tokens (`tag == 1`).

### 6.3 Four Rules That Make Physical-Action Vectors (`head_swell`, `zero_gravity`) Fire Cleanly
Systematic ablation across 52 Pika H3 Flex arms ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)) established four rules for physical transformations:
1. **Match `strength` to the pooling mode's `raw_norm` (`50 / raw_norm .. 85 / raw_norm`)**: Single-clause pairs share ~70%–85% identical setup tokens before the changed clause. Use `--pooling diff_span` (`raw_norm ≈ 50..87`, `strength = 0.60 .. 1.50`) or `--pooling from_first_diff` (`raw_norm ≈ 22..24`, `strength = 3.0 .. 3.5`), scale `scene_text` (`raw_norm ≈ 10.5`) to `strength = 6.5 .. 8.0`, or set `mode = "add_norm_matched"` (`strength = 1.0`).
2. **Keep `<d>[English] ...</d>` dialogue short (`1–5` words, finishing by `~1.5s`) in your Test Scene Prompt**: In `141`-frame (`5.88s`) clips, a short line (`<d>[English] Morning!</d>` or `<d>[English] Another pie?</d>`) leaves `4.3s+` of post-speech runtime for the physical transformation to unfold.
3. **Never end your Test Scene Prompt with a static-pose or grounding clause**: Avoid ending with *"while the camera holds a steady static frame on his normal upper body"*—end cleanly as the character's mouth closes after `<d>...</d>` (*"before settling his lips into a calm smile."*).
4. **Always match the vector's family (`fl2va` vs. `ref2va`) to the UNet**.

---

## 7. ExLlamaV3 `Qwen3-VL-32B-Instruct` `layers.49` Readout & Research Records

Because H3's text encoder is `Qwen3-VL-32B-Instruct` truncated after `layers.49`, loading the full 64-layer model in ExLlamaV3 lets us inject `vector.encoder` (`[5120]`) at `layers.49` and read out what `layers.50..63` + `lm_head` decode in natural language:

| Document / Tool | What It Covers |
|---|---|
| [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md) | Canonical copy-pasteable LLM system prompt for authoring 10-pair contrastive `.md` files + test scene prompts + optional hold pairs |
| [`docs/wiki/system_prompts/h3_system_prompt.md`](system_prompts/h3_system_prompt.md) | Canonical general H3 scene-prompt system prompt (`t2va`, `fl2va`, `ref2va`) |
| [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) | Historical 2026-11-05 research record on `diff_span` vs. `from_first_diff` vs. `scene_text` pooling, post-speech timeline budgets, and `layers.49` readout findings |
| [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md) | Full technical specification (`E2.1`–`E2.8`, `E4.1`–`E4.3`), math (`mean_difference`, `flip_null`, `random_null`, `hold_basis`), and 3 text-path weight families (`fl2va`, `ref2va`, `fasth3`) |
| [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md) | 52 rendered Pika H3 Flex (`6`-step `lcm`) and `ref2va` arms across 5 waves (`film_16mm_1974` gate ladder, audio+video steering, `ref2va` `mask="non_vision"`, `head_swell` & `zero_gravity` `scene_text` / `from_first_diff` / `diff_span` pooling and timeline ablations, and diverse `t2va`/`ref2va` scenes across the 34-bank `diff_span` suite) |
| [`docs/research/control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md`](../research/control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md) | Architectural comparison of `coderef/ComfyUI-MAINodes` (`dials/`, `h3_cond_strength.py`, `concept_lab/`) vs. our stack, what adopting `MAINodes` methods would break, and the 6-item Pika H3 Flex priority roadmap (single-clause pair compiler, inference-time cross-vector holds, velocity-space audio/video split, `ref2va` 3D VAE reference gate, picture/video media contrast vectors, and `1,4,4,4,4`-aligned temporal curves) |
| [`docs/research/control_vectors/2026-11-06_detangle_comparison_and_borrowable_concepts.md`](../research/control_vectors/2026-11-06_detangle_comparison_and_borrowable_concepts.md) | Comparison of `coderef/DeTangle` (`matlowai/DeTangle`) vs. H3 control vectors: soft ridge hold attenuation ($\alpha \in [0, 1]$), inference-time cross-vector orthogonalization, paired two-number evaluation, and why 5376-d covariance whitening is a trap |
| [`docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md) | Analytical Jacobian Lens (`J-Lens`) and empirical-covariance Wiener `TokenRefiner` readout for `5376`-d post-refiner control vectors: token-0 sink handling, broadcast attention value path ($\beta$), `TokenRefiner` singular value spectrum, 7-prefix cloze-gated CPU readout (`< 15 ms`), $\alpha \in \{0.0, 0.5, 1.0\}$ hold sweeps, and why inverse BPE word pullback was dropped |
| [`docs/research/control_vectors/exl3_qwen3vl/runbook.md`](../research/control_vectors/exl3_qwen3vl/runbook.md) | End-to-end ExLlamaV3 quantization & `layers.49` steering readout runbook on a 24 GB RTX 4090 |
| [`docs/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py`](../research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py) | Batch labeller combining blind pair-diff synthesis, `layers.49 -> lm_head` log-odds shifts, coupled-Gumbel steered rollouts, and VLM video critique |
| [`docs/research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md) | Generated batch-labelling and `layers.49` readout report across the initial 14 `.toml` control vectors |

---

## 8. CLI Reference (`bench/control_vectors.py`)

```bash
# 0. INSTANT < 2 ms WORD SYNTHESIS (0 VRAM, zero .toml): Build fl2va + ref2va .safetensors + J-Lens report:
python bench/control_vectors.py words "blizzard, freezing, snow"
python bench/control_vectors.py words "inflatable, balloon, glossy, latex" \
    --against-words @prompt_subject --prompt prompt_bank/t2va_noodle_bar.txt
python bench/control_vectors.py words "skeleton, translucent, bones, skull" --against-words @replace_human

# 1. INTERACTIVE STUDIO UI: Launch the local browser studio (exits when you click 'Quit Server'):
python bench/control_vectors.py ui

# 2. ONE-SHOT BUILD: Markdown (.md), TOML (.toml), or --words / --from-words -> fl2va & ref2va .safetensors
python bench/control_vectors.py build my_effect.md
python bench/control_vectors.py build --from-words "medieval, knight, armor, castle" --name medieval_words
python bench/control_vectors.py build --words "steady tripod -> erratic handheld camera shake"

#    Batch-build all 32 shipped .toml banks in prompt_bank/control_vectors/ in one cached pass:
python bench/control_vectors.py build --all --overwrite

#    Build multiple pooling modes (diff_span, from_first_diff, scene_text) at once:
python bench/control_vectors.py build my_effect.md --pooling all --overwrite

#    Build from an existing .toml file for both fl2va and ref2va with diff_span pooling:
python bench/control_vectors.py build prompt_bank/control_vectors/head_swell.toml --overwrite

# 3. Convert LLM Markdown (.md) to .toml only (without running the encoder):
python bench/control_vectors.py from-md my_effect.md \
    --test-prompt-out prompt_bank/control_vectors/my_effect_test_prompt.txt

# 4. Single-target make (explicit single UNet + LoRA + pooling, optional --unit-mean):
python bench/control_vectors.py make prompt_bank/control_vectors/zero_gravity.toml \
    --pooling diff_span \
    --unet minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    --lora pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0

# 5. Inspect a saved .safetensors vector (with optional --prompt receptor scan, --hold-words, and --meter):
python bench/control_vectors.py inspect \
    models/control_vectors/head_swell__fl2va__diff_span__lora_2a8c2b52.safetensors \
    --prompt prompt_bank/control_vectors/head_swell_test_prompt.txt \
    --meter prompt_bank/control_vectors/head_swell_test_prompt.txt

# 6. Run the CPU unit & integration check suite (C1-C7):
python bench/check_control_vectors.py
```
