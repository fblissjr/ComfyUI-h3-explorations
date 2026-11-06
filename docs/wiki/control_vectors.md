# Control Vectors (Activation Steering on MiniMax H3)

Control vectors steer MiniMax H3 video+audio generation by shifting the 5376-d post-refiner conditioning stream (`context` after `condition_proj` + 2-layer bidirectional `token_refiner`, wired via `WrappersMP.DIFFUSION_MODEL` around `MiniMaxH3Model._forward` right after `preprocess_text_embeds`) along a learned concept direction $\mathbf{v}_{\text{PR}} \in \mathbb{R}^{5376}$ extracted from 8–16 contrastive prompt pairs $(A_i, B_i)$.

- **Zero training, zero DiT slowdown**: A control vector is a single `~24 KB` `.safetensors` file containing a `5376`-float vector (`vector.post_refiner`). Applying one (or chaining several) adds `< 0.1 ms` per diffusion step.
- **No running ComfyUI server required to build**: Building a control vector only runs the text encoder (`Qwen3-VL-32B-Instruct` `layers.0..49`, `~5.5s` load + `~2.0s` for 10 pairs) and the tiny 2-block `token_refiner` (`~0.18s` on CPU). You do **not** need ComfyUI (`:8188`) running or the 22 GB DiT loaded to create vectors.

---

## 1. Quickstart Guide (For Humans): End-to-End in 3 Steps

### Step 1: Generate a Markdown Pair File with Any LLM
1. Open [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) and copy the fenced system prompt block in **Section 1**.
2. Paste it into Claude, Gemini, or ChatGPT, and ask for the effect you want, e.g.:
   > *"Create a control vector for `head_swell`: a person's head inflates and balloons to twice its normal size like a helium balloon, with a stretched latex creak and pressurized air hiss."*
3. Save the LLM's Markdown response (which contains `### Test Scene Prompt` followed by `#### Pair 1` .. `#### Pair 10`) to a `.md` file, e.g. `my_effect.md`.

### Step 2: Run the One-Shot `build` Command
From the repo root, run **one command**:

```bash
python bench/control_vectors.py build my_effect.md
```

*(Note: `bench/control_vectors.py` automatically locates `../../.venv/bin/python` if your ComfyUI virtualenv is not currently active.)*

In **one pass (`~8 seconds` total)**, `build` automatically:
1. Validates every pair in `my_effect.md` (single contiguous changed clause, identical section headers) and writes **`prompt_bank/control_vectors/my_effect.toml`**.
2. Extracts the `### Test Scene Prompt` from the Markdown into **`prompt_bank/control_vectors/my_effect_test_prompt.txt`** so you can paste it straight into ComfyUI.
3. Loads `Qwen3-VL-32B-Instruct` **once** and encodes all 10 contrastive pairs through `layers.49`.
4. Runs the lightweight 2-layer `token_refiner` for **both** `fl2va` (using `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `pika_h3_flex_fl2va_svd995_bf16.safetensors:1.0`) and `ref2va` (using `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0`) with **`--pooling diff_span`**, writing the ready-to-use `.safetensors` vectors and human-readable `.md` sidecars into `<comfy>/models/control_vectors/` and printing a summary table with the exact filenames and recommended ComfyUI `strength` (`50 / raw_norm .. 85 / raw_norm`).

Want multiple pooling modes at once (`diff_span`, `from_first_diff`, `scene_text`) or to overwrite an existing build?
```bash
# Build diff_span, from_first_diff, and scene_text for both fl2va and ref2va in one ~8s pass:
python bench/control_vectors.py build my_effect.md --pooling all --overwrite
```

### Step 3: Render in ComfyUI
1. Open one of the shipped control-vector workflows in ComfyUI (or add **`MiniMaxH3ApplyControlVectors`** to your own workflow):
   - **Single control vector (`t2va` / `fl2va`)**: [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json)
   - **Chained dual control vectors (`t2va` / `fl2va`)**: [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json)
   - **Character-reference control vector (`ref2va`)**: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json)
2. If wiring manually, place **`MiniMaxH3ApplyControlVectors`** on the **`MODEL`** wire **after** `UNETLoader` + `MiniMaxH3LoRABranch` (your Pika LoRA) and **before** `BasicGuider` / `SamplerCustomAdvanced`:
   ```text
   [UNETLoader] ──► [MiniMaxH3LoRABranch (Pika)] ──► [MiniMaxH3ApplyControlVectors] ──► [BasicGuider]
   ```
3. On **`MiniMaxH3ApplyControlVectors`**, set:
   - **`vector_name`**: pick the matching family file printed by `build`:
     - For `t2v` / `fl2v` / `i2v`: `<name>__fl2va__diff_span__lora_2a8c2b52.safetensors`
     - For `ref2va`: `<name>__ref2va__diff_span__lora_865ee189.safetensors`
   - **`strength`**: use the recommended range printed by `build` (and recorded in the companion `.md` sidecar):
     - **`diff_span` vectors (`raw_norm ≈ 85`)**: **`0.60 .. 1.00`**
     - **`from_first_diff` vectors (`raw_norm ≈ 22..24`)**: **`3.00 .. 3.50`**
     - **`scene_text` vectors**: **`1.80 .. 2.80`** for multi-clause style vectors (`raw_norm ≈ 20..25`), or **`6.50 .. 8.00`** for single-clause action vectors (`raw_norm ≈ 10..11`)
     - Use `0.0` for an exact bitwise baseline check or negative values to invert the direction.
   - **`mask`**: **`non_vision`** (default; shifts text conditioning rows while leaving tag-0 reference-image vision tokens untouched so character identity is preserved in `ref2va`).
   - **`mode`**: **`add`** (default).
   - **`allow_recipe_mismatch`**: leave `False` if using the standard `svd995` Pika LoRA, or set `True` if you switched to a different Pika LoRA file (`comfy_bf16`, `svd99`, `kijai`) or base checkpoint.
4. Paste the contents of **`prompt_bank/control_vectors/my_effect_test_prompt.txt`** (a neutral baseline scene that does **not** mention the effect in text) into your prompt node and queue!

---

## 2. Practical FAQ: Prompting, Pika LoRAs, `.toml` Fields & Pooling Modes

### Q1: Should Pole B describe a passive static state ("is floating") or an active physical trajectory ("inflates and balloons outward" / "drift upward and float in mid-air")?
**Write Pole B as a direct, active physical transformation and trajectory** (e.g., *"while his skull and cranium inflate and balloon outward to twice their normal size like an over-pumped helium sphere"* or *"while the brass letter opener, fountain pen, and inkpot drift weightlessly upward off the mahogany desk and float in mid-air"*), and **avoid generic onset/rate filler adverbs** (`slowly`, `gradually`, `steadily`, `begins to`, `starts to`):
- **Why active trajectory beats passive static state (`fl2va` / `i2v` / `t2va`)**: In `fl2va` and `i2v`, Frame 0 is anchored by an input image with normal proportions/gravity. Describing the active physical action and secondary motion (inflating, rising off the desk, rotating in mid-air, plus the matching sound cue) drives H3's 3D spatiotemporal attention blocks to animate the transition out of Frame 0.
- **Why you must avoid `"begins to"` / `"slowly"` / `"gradually"` filler ([`2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md#L135))**: Repeating generic onset adverbs across all 10 Pole-B clauses causes `Qwen3-VL` `layers.49` to extract a generic *"slow/gradual onset"* direction instead of the physical effect itself. Use concrete domain verbs and nouns directly.

### Q2: How closely aligned do the subject and gender in the 10 pairs need to be to my final render prompt?
**They do NOT need to match your specific render subject—in fact, the 10 pairs SHOULD vary subjects and settings across pairs 1–10** (while staying identical between Prompt A and Prompt B *within* each pair):
- Within each pair $i$, Prompt A and Prompt B share the exact same subject and setting, so when we compute $\Delta_i = \mathbf{r}(B_i) - \mathbf{r}(A_i)$, the subject/gender/setting tokens cancel out.
- Averaging across 10 diverse subjects (`a man`, `a woman`, `an elderly watchmaker`, `a chef`, tabletop objects) cancels subject-specific curvature in `Qwen3-VL` and `token_refiner`, leaving a pure effect direction that transfers to men, women, groups, or objects.
- **Rule for `ref2va` ([`2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md#L125))**: Even when building a control vector intended for `ref2va` (such as [`zero_gravity.toml`](../../prompt_bank/control_vectors/zero_gravity.toml)), the **10 Contrastive Pairs (`Pair 1..10`) always use the 3-section text format (`integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:`) with 10 unrelated characters/objects and NO `<Subject N>` or `<Picture N>` tags**. Only the `### Test Scene Prompt` at the top of the Markdown uses the 6-section `ref2va` format (`subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, `overall_soundscape:`, `non_diegetic_music:`) with `<Subject 1>` when testing on `ref2va`.

### Q3: Does it matter *which* Pika LoRA variant I use (`svd995`, `svd99`, `comfy_bf16`, `kijai`)?
**No—all four Pika LoRA variants within a family (`fl2va` or `ref2va`) produce virtually identical control vectors (`cosine = 0.999964 .. 1.000000`, `0.00% .. 0.05%` norm difference):**
- Why: Extracting a control vector only runs `condition_proj` + the 2-layer `token_refiner` (not the 50 DiT blocks), and `svd995`/`svd99` are 99.5%/99% SVD truncations of the exact same `comfy_bf16` weights.
- **What DOES matter**:
  1. **Model family (`fl2va` vs. `ref2va`)**: `fl2va` and `ref2va` checkpoints have completely different `condition_proj` + `token_refiner` weights (cross-family cosine is `~0.19`). Always use the `__fl2va__` vector on `t2v`/`fl2v`/`i2v` graphs and the `__ref2va__` vector on `ref2va` graphs (`python bench/control_vectors.py build` builds both automatically).
  2. **Pika LoRA vs. base UNet ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md#L9-L15))**: The Pika LoRA rotates `token_refiner` output by `< 0.43%` (`cos = 0.995730 .. 0.996678` vs. base UNet, with a `1.38%–2.10%` norm increase), and records the LoRA SHA256 in `recipe.loras`.
  3. **If you switch between Pika LoRA files in ComfyUI** (e.g., the vector was built with `svd995` and your graph loads `comfy_bf16` or `svd99`): just set **`allow_recipe_mismatch = True`** on `MiniMaxH3ApplyControlVectors` so the SHA256 recipe guard logs an info warning instead of refusing.

### Q4: What data in the `.toml` file is actually used during extraction vs. inference?
- **At inference time in ComfyUI (`MiniMaxH3ApplyControlVectors`)**: The `.toml` file is **not read at all**. ComfyUI only loads the `.safetensors` file and reads `vector.post_refiner` (`float32[5376]`) plus the `family` and `recipe` hashes in its JSON header.
- **At extraction time (`build` / `make`)**:

| `.toml` Field | Used in Vector Math? | What It Actually Does |
|---|---|---|
| `[[pair]].a`, `[[pair]].b` | **Yes (Core math)** | Encoded through `Qwen3-VL` (`layers.0..49`) -> `condition_proj` -> `token_refiner`, masked by `--pooling`, and subtracted ($\mathbf{r}_B - \mathbf{r}_A$). Also validated to ensure exactly one contiguous span differs. |
| `[[pair]].weight` | **Yes (Core math)** | Weights each pair in the weighted mean difference (`default = 1.0`). |
| `[[hold]].a`, `[[hold]].b`, `group` | **Yes (Optional math)** | If present (and `--no-holds` is not set), builds an orthonormal basis of unwanted nuisance directions and projects them out of `vector.post_refiner`. |
| `[vector].frame` | **Yes (Validation)** | `"scene"` (default) enforces the 3-section H3 prompt grammar (`integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:`). |
| `[vector].mode` | **Default UNet selector** | `"t2va"` or `"ref2va"` (auto-detected from `"subject_definitions:"` in [`control_vectors.py::parse_markdown_pairs`](../../control_vectors.py#L531-L534) when converting `.md`); used when `--families auto` or `make` without `--unet` is run. |
| `[vector].name` | **Filename stem** | Default output filename stem (`<name>__<family>__...safetensors`). |
| `[vector].quality`, `a_side`, `b_side` | **No (Metadata only)** | Human-readable descriptions copied into the `.safetensors` header and companion `.md` sidecar report. Changing these strings does **not** change a single float in the vector. |

### Q5: How do the `--pooling` modes work, and which one should I use?
Suppose Pair 1 changes **one clause** near the end of `integrated_multimodal_description:` (say tokens `104..122` out of `155` total tokens):

```text
Token idx:  [0 ................. 103] [104 .............. 122] [123 ....................... 154]
Content:    Shared scene setup prefix  CHANGED CLAUSE (A vs B)  Shared soundscape + music suffix
            (identical in A and B!)    (differs in A and B)     (identical text, but attended to
                                                                 104..122 via causal Qwen3-VL!)
```

Because `Qwen3-VL-32B-Instruct` (`layers.0..49`) is **strictly causal**, tokens `0..103` before `first_diff` (`104`) have **identical** `layers.49` hidden states in Prompt A and Prompt B (and near-zero difference after the 2-layer bidirectional `token_refiner`). Averaging over those `104` unchanged prefix tokens dilutes the vector's norm (`raw_norm`) by `2.1x–8.2x`!

| Pooling Mode | Exact Token Mask Pooled | Diluted by Shared Prefix/Suffix? | Measured `raw_norm` (`head_swell` / `zero_gravity`) | Recommended ComfyUI `strength` (`50/norm .. 85/norm`) | When to Use |
|---|---|---|---|---|---|
| **`diff_span`** *(Default in `build`)* | `[first_diff:end_diff]` within `span_masks["scene"]` (falls back to `~span_masks["vision"]` if diff is inside `<d>...</d>`) ([`h3_spans.py::differing_token_bounds`](../../h3_spans.py#L361-L381)) | **0% shared tokens** (strips both shared prefix and shared suffix) | **`85.01` / `85.13`** (`~8.0x–8.2x` stronger than `scene_text`) | **`0.60 .. 1.00`** (`~0.59 .. 1.00`) | **Best default for single-clause vectors**—isolates the exact changed clause (`a_span` vs. `b_span`) with zero shared-token dilution. |
| **`from_first_diff`** | `[first_diff:]` within `~span_masks["vision"]` (strips shared prefix `[:first_diff]`, keeps changed clause + trailing soundscape/music tokens) ([`h3_spans.py::first_differing_index`](../../h3_spans.py#L343-L358)) | Strips prefix; keeps trailing suffix tokens that causally attended to the diff | **`23.57` / `22.62`** (`~2.12x–2.28x` stronger than `scene_text`) | **`2.10 .. 3.75`** (`3.0 .. 3.5` in `arms 39..40`) | Great when the changed clause is at the end of Section 1 and you want the trailing `overall_soundscape:` tokens included in the average. |
| **`scene_text`** | All `span_masks["scene"]` rows across the entire prompt (excludes vision, `<Picture N>` labels, timestamps, `(S1)`, `<d>...</d>`) | **Yes (`70%–85%` shared tokens)** if only one clause changed | **`10.35` / `10.65`** on single-clause; **`20.40`–`24.54`** on multi-clause | **`4.70 .. 8.20`** (`6.5 .. 8.0` single-clause) or **`2.00 .. 3.50`** (multi-clause) | Use when multiple clauses across the prompt change (like `film_16mm_1974.toml`), or scale `strength` up to `6.5 .. 8.0` to compensate for dilution. |
| **`all_tag1`** | Every non-vision (`tag == 1`) row, including `<d>...</d>` dialogue and `(S1)` speaker tags | **Yes** | Diagnostic only | Diagnostic only | Only for testing whether dialogue tokens carry the concept (`E2.6` ablation). |

> **Why `diff_span` at `strength = 0.6..1.0`, `from_first_diff` at `strength = 3.0..3.5`, and `scene_text` at `strength = 6.5..8.0` produce the same physical effect**: What matters at inference is the total post-refiner injection magnitude $\|\Delta \mathbf{r}_{\text{PR}}\| = \text{strength} \times \text{raw\_norm} \approx 50\text{–}85$ (compared to a median post-refiner token norm of `~93–164`). Both `bench/control_vectors.py build` and the `.md` sidecar automatically compute `50 / raw_norm .. 85 / raw_norm` so you always see the exact recommended `strength` range for each vector file.

---

## 3. Code Architecture & File Map

| File | Responsibility | Key Entry Points |
|---|---|---|
| [`h3_spans.py`](../../h3_spans.py) | Deterministic token-span partition of H3 conditioning sequences (`vision`, `label`, `timestamp`, `dialogue`, `speaker`, `scene`) and pooling masks | `spans`, `differing_token_bounds`, `first_differing_index`, `pooling_mask` (`POOLING_MODES = ("scene_text", "all_tag1", "from_first_diff", "diff_span")`) |
| [`control_vectors.py`](../../control_vectors.py) | Core float64 math, `.toml` pair loader, Markdown-to-`.toml` converter, `.safetensors` + `.md` sidecar I/O, model family & recipe verification, and `apply_post_refiner` | `load_pairs`, `differing_span` (`MAX_SHARED_INTERIOR_WORDS = 5`), `parse_markdown_pairs`, `write_markdown_to_toml`, `mean_difference`, `flip_null`, `random_null`, `hold_basis`, `project_out`, `apply_post_refiner` (`APPLY_MODES = ("add", "project_out", "amplify")`), `save_vector`, `format_vector_sidecar_md`, `load_vector` |
| [`control_vector_nodes.py`](../../control_vector_nodes.py) | ComfyUI V3 nodes, runtime `WrappersMP.DIFFUSION_MODEL` hook, recipe extraction, and CPU/GPU `token_refiner` runner (with `lora_branch.py` support) | `MiniMaxH3MakeControlVector`, `MiniMaxH3ApplyControlVectors`, `MiniMaxH3LayerNormMatch`, `attach_control_vector`, `make_control_vector_from_pairs`, `build_refiner_runner`, `extract_model_recipe` |
| [`bench/control_vectors.py`](../../bench/control_vectors.py) | Standalone CLI for one-shot `build` (`.md`/`.toml` -> `.safetensors`), `from-md`, `make`, `inspect` (`--meter`), and `grade` | `cmd_build`, `cmd_from_md`, `cmd_make`, `cmd_inspect`, `cmd_grade` |
| [`workflows/build_workflows.py`](../../workflows/build_workflows.py) | Programmatic workflow builders for single/chained control-vector arms, 6-arm gate ladders, and E4.2/E4.3 probes | `build_control_vector_arm`, `build_control_vector_gate_arms`, `build_control_vector_probe_arm`, `build_api(..., control_vector=...)` |
| [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py) | Tracked CPU check (`C1`–`C7`) defending tokenizer constants, span masks, `.toml`/`.md` refusals, float64 math, `.safetensors` + `.md` sidecar I/O, runtime hooks, and CLI subcommands | `main` (`python bench/check_control_vectors.py`) |

---

## 4. Shipped Contrastive Pair Files & Vector Naming Convention

### Shipped Pair Files (`prompt_bank/control_vectors/*.toml`)

| Pair File | Concept / Pole A -> Pole B | Pair / Hold Count & Default Mode |
|---|---|---|
| [`prompt_bank/control_vectors/film_16mm_1974.toml`](../../prompt_bank/control_vectors/film_16mm_1974.toml) | Clean modern digital color -> 1974 16mm reversal film stock, halation, gate weave, optical hiss (bare twin: [`film_16mm_1974_bare.toml`](../../prompt_bank/control_vectors/film_16mm_1974_bare.toml)) | 16 pairs + 6 holds (`warm_tint`, `soft_focus`), `t2va` |
| [`prompt_bank/control_vectors/chaos_panic.toml`](../../prompt_bank/control_vectors/chaos_panic.toml) | Calm, orderly scene -> Sudden chaotic panic, papers flying, frantic shouting, urgent alarms | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/noir.toml`](../../prompt_bank/control_vectors/noir.toml) | Balanced daylight color + ambient sound -> High-contrast 1940s silver-nitrate B&W noir + smoky jazz atmosphere | 12 pairs, `t2va` |
| [`prompt_bank/control_vectors/felt_muppet.toml`](../../prompt_bank/control_vectors/felt_muppet.toml) | Live-action human skin/fabric -> Hand-stitched fuzzy felt puppet texture, button eyes, practical studio stage lighting | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/cursed_cctv.toml`](../../prompt_bank/control_vectors/cursed_cctv.toml) | Clean cinema camera -> Degraded low-bitrate night-vision CCTV security feed with harsh IR blowouts and static hum | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/reverb_cathedral.toml`](../../prompt_bank/control_vectors/reverb_cathedral.toml) | Dry close-mic room acoustics -> Cavernous stone cathedral acoustics with long echoing reverb tails | 12 pairs, `t2va` |
| [`prompt_bank/control_vectors/vocal_strain.toml`](../../prompt_bank/control_vectors/vocal_strain.toml) | Calm measured delivery -> Breathless, cracking, emotionally strained vocal delivery | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/fever_melt.toml`](../../prompt_bank/control_vectors/fever_melt.toml) | Stable solid geometry -> Surreal fever-dream melting and warping surfaces with distorted drone | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/infomercial_fail.toml`](../../prompt_bank/control_vectors/infomercial_fail.toml) | Calm everyday task -> Over-the-top exasperated black-and-white infomercial clumsy spill/fumble | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/lettering.toml`](../../prompt_bank/control_vectors/lettering.toml) | Unlettered surface -> Bold physical signage/lettering integrated into the scene (bare twin: [`lettering_bare.toml`](../../prompt_bank/control_vectors/lettering_bare.toml)) | 16 pairs, `t2va` |
| [`prompt_bank/control_vectors/head_swell.toml`](../../prompt_bank/control_vectors/head_swell.toml) | Normal head proportions -> Cranium inflates and balloons to 2x size like an over-pumped sphere + latex creak & steam hiss | 10 pairs, `t2va` |
| [`prompt_bank/control_vectors/zero_gravity.toml`](../../prompt_bank/control_vectors/zero_gravity.toml) | Normal 1g gravity -> Sudden zero-gravity weightlessness as loose tabletop objects float into mid-air + low hum | 10 pairs, `ref2va` |

### Output Filename Convention (`models/control_vectors/`)
[`control_vectors.py::default_vector_filename`](../../control_vectors.py#L1382-L1415) names every vector so different model families, pooling modes, and LoRA recipes never overwrite each other:
```text
<stem>__<family>[__<pooling>][__ref_<8hex>][__noholds][__lora_<8hex>].safetensors
```
Every `.safetensors` file is accompanied by an auto-generated **`.md` sidecar** (`<same_stem>.md`) containing its recipe table, recommended ComfyUI `strength` range, agreement cosines (`mean_pairwise_cosine`, `split_half_cosine`), and per-pair token spans.

---

## 5. Workflows & Physical-Action Best Practices (`t2va`, Dual-Vector, and `ref2va`)

### 5.1 Single & Chained Dual Control Vectors on `t2va` / `fl2va`
- **Single Control Vector Workflow**: [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json)
  - Wiring: `MiniMaxH3LoRABranch (18, pika_h3_flex_fl2va_svd995_bf16.safetensors) -> ModelSamplingSD3 (19, shift=5.0) -> ModelAttentionBackend (58, triton) -> MiniMaxH3Sol (21) -> MiniMaxH3ApplyControlVectors (115) -> BasicGuider (9)`.
- **Chained Dual Control Vectors Workflow**: [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json)
  - Chains two `MiniMaxH3ApplyControlVectors` nodes in series (`"115"` $\to$ `"125"` $\to$ `BasicGuider (9)`). Consecutive same-mask `add` vectors accumulate in `float64` while allowing independent `strength`, `mode` (`add`, `project_out`, `amplify`), `mask`, and step schedules (`start_percent`, `end_percent`).

### 5.2 Character-Reference Control Vectors on `ref2va`
- **Shipped `ref2va` Workflow**: [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json)
  - Uses a `ref2va`-family vector (`*__ref2va__*.safetensors`) and `mask="non_vision"` (default) so `<Picture 1>`'s vision patch embeddings (`tag == 0`) remain bitwise identical (`0.0` diff), locking `<Subject 1>`'s face and wardrobe while steering the scene via text tokens (`tag == 1`).

### 5.3 Four Rules That Make Physical-Action Vectors (`head_swell`, `zero_gravity`) Fire Cleanly
Systematic ablation across 40 Pika H3 Flex arms ([`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)) established four rules for physical transformations:
1. **Match `strength` to the pooling mode's `raw_norm` (`50 / raw_norm .. 85 / raw_norm`)**: Single-clause pairs share ~70%–85% identical setup tokens before the changed clause. Use `--pooling diff_span` (`raw_norm ≈ 85`, `strength = 0.60 .. 1.00`) or `--pooling from_first_diff` (`raw_norm ≈ 22..24`, `strength = 3.0 .. 3.5`), or scale `scene_text` (`raw_norm ≈ 10.5`) to `strength = 6.5 .. 8.0`.
2. **Keep `<d>...</d>` dialogue short (`1–5` words, finishing by `~1.5s`) in your Test Scene Prompt**: In `141`-frame (`5.88s`) clips, a short line (`<d>Morning!</d>` or `<d>Another pie?</d>`) leaves `4.3s+` of post-speech runtime for the physical transformation to unfold.
3. **Never end your Test Scene Prompt with a static-pose or grounding clause**: Avoid ending with *"while the camera holds a steady static frame on his normal upper body"*—end cleanly as the character's mouth closes after `<d>...</d>` (*"before settling his lips into a calm smile."*).
4. **Always match the vector's family (`fl2va` vs. `ref2va`) to the UNet**.

---

## 6. ExLlamaV3 `Qwen3-VL-32B-Instruct` `layers.49` Readout & Research Records

Because H3's text encoder is `Qwen3-VL-32B-Instruct` truncated after `layers.49`, loading the full 64-layer model in ExLlamaV3 lets us inject `vector.encoder` (`[5120]`) at `layers.49` and read out what `layers.50..63` + `lm_head` decode in natural language:

| Document / Tool | What It Covers |
|---|---|
| [`docs/research/control_vectors/2026-11-05_control_vector_system_prompt.md`](../research/control_vectors/2026-11-05_control_vector_system_prompt.md) | Copy-pasteable LLM system prompt for authoring 10-pair contrastive `.md` files + test scene prompts |
| [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md) | Full technical specification (`E2.1`–`E2.8`, `E4.1`–`E4.3`), math (`mean_difference`, `flip_null`, `random_null`, `hold_basis`), and 3 text-path weight families (`fl2va`, `ref2va`, `fasth3`) |
| [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md) | 40 rendered Pika H3 Flex (`6`-step `lcm`) and `ref2va` arms across 4 waves (`film_16mm_1974` gate ladder, audio+video steering, `ref2va` `mask="non_vision"`, `head_swell` & `zero_gravity` pooling and timeline ablations) |
| [`docs/research/control_vectors/exl3_qwen3vl/runbook.md`](../research/control_vectors/exl3_qwen3vl/runbook.md) | End-to-end ExLlamaV3 quantization & `layers.49` steering readout runbook on a 24 GB RTX 4090 |
| [`docs/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py`](../research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py) | Batch labeller combining blind pair-diff synthesis, `layers.49 -> lm_head` log-odds shifts, coupled-Gumbel steered rollouts, and VLM video critique |
| [`docs/research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md) | Generated batch-labelling and `layers.49` readout report across all 14 `.toml` control vectors |

---

## 7. CLI Reference (`bench/control_vectors.py`)

```bash
# 1. ONE-SHOT BUILD (Recommended): Markdown (.md) or TOML (.toml) -> .toml + _test_prompt.txt + fl2va & ref2va .safetensors
python bench/control_vectors.py build my_effect.md

#    Build multiple pooling modes (diff_span, from_first_diff, scene_text) at once:
python bench/control_vectors.py build my_effect.md --pooling all --overwrite

#    Build from an existing .toml file for both fl2va and ref2va with diff_span pooling:
python bench/control_vectors.py build prompt_bank/control_vectors/head_swell.toml --overwrite

# 2. Convert LLM Markdown (.md) to .toml only (without running the encoder):
python bench/control_vectors.py from-md my_effect.md \
    --test-prompt-out prompt_bank/control_vectors/my_effect_test_prompt.txt

# 3. Single-target make (explicit single UNet + LoRA + pooling):
python bench/control_vectors.py make prompt_bank/control_vectors/zero_gravity.toml \
    --pooling diff_span \
    --unet minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    --lora pika_h3_flex_ref2va_svd995_bf16.safetensors:1.0

# 4. Inspect a saved .safetensors vector and meter prompts along it (no DiT load):
python bench/control_vectors.py inspect \
    models/control_vectors/head_swell_ds__fl2va__diff_span__lora_2a8c2b52.safetensors \
    --meter prompt_bank/control_vectors/head_swell_test_prompt.txt

# 5. Run the CPU unit & integration check suite (C1-C7):
python bench/check_control_vectors.py
```
