# Control Vectors for MiniMax H3 — Start Here

Activation control vectors steer MiniMax H3 video + audio generation (`t2va`, `fl2va`, `ref2va`) and single-frame `P-H3-Image` generation & editing (`p_h3_image_t2i`, `p_h3_image_edit`, `p_h3_image_guide_edit`) by adding, projecting out, or amplifying a **5376-dimensional direction** in the post-`TokenRefiner` text-conditioning space — **without changing your prompt text or token length $L$**.

Because Qwen3-VL causal attention changes every downstream token whenever you edit a single word in a prompt, ordinary prompt edits reshuffle camera framing, actor blocking, and lip-sync timing on the same seed. A control vector leaves your prompt tokens untouched and shifts only the semantic concept direction (genre, atmosphere, lighting, film stock, era props, motion energy, or acoustic space) across the tokens you choose.

![Animated explainer of MiniMax H3 Control Vector extraction and runtime steering](control_vectors/assets/control_vector_pipeline.svg)

> **More Animated Visual Explainers (Lightweight Inline SVGs)**:
> - [Surgical `diff_span` Token Pooling (`B - A`)](control_vectors/assets/diff_span_extraction.svg) — in [`2. Building Pairs & Control Vectors`](control_vectors/building_pairs_and_vectors.md#3-token-pooling-modes-pooling)
> - [Runtime Token Masks (`scene_only`), Step Windows & `ref2va` 3D VAE Gate](control_vectors/assets/runtime_steering_and_masks.svg) — in [`3. ComfyUI Nodes & Runtime Controls`](control_vectors/comfyui_nodes_and_controls.md#2-minimaxh3applycontrolvectors-inputs-explained-control_vector_nodespyminimaxh3applycontrolvectors)
> - [Gram-Schmidt Orthogonal `hold`, `< 5 ms` CPU Inversion (`--invert`) & `67.5` Norm Matching](control_vectors/assets/sculpt_and_orthogonal_hold.svg) — in [`2. Building Pairs & Control Vectors`](control_vectors/building_pairs_and_vectors.md#4-orthogonal-holds-hold---hold---hold-words) and [`4. J-Lens Diagnostics & Word Lab`](control_vectors/j_lens_diagnostics_and_word_lab.md)

---

## 1. What Control Vectors CAN and CANNOT Do

| Capability | What Control Vectors **CAN** Do Well | What Control Vectors **CANNOT** Do (Use Another Tool Instead) |
|---|---|---|
| **Genre, Atmosphere, Weather & Lighting** | Shift a scene into a 90s multi-camera sitcom, indie arthouse drama, psychological horror dread, blizzard, scorching heatwave, golden hour, noir chiaroscuro, or 16mm film grain while keeping the same shot composition. | Cannot place a specific object at exact pixel coordinates $(x, y)$ or enforce a frame-accurate silhouette mask (use the [masked V2V lane](../masked_v2v/AGENTS.md) or reference images). |
| **Invert or Combine Any Concept** | Invert any existing vector (`strength = -1.0`, e.g., flipping `blizzard` toward warm/dry air) or build its true opposite (`"calm clear air -> scorching desert heatwave"`), and stack up to 6 vectors with orthogonal holds. | Multiplying `blizzard` by `-1.0` removes snow/cold toward neutral dry air, whereas authoring `"clear mild day -> scorching desert heatwave with shimmering heat haze"` actively injects heat haze and sun glare. |
| **Independent Video vs. Audio Steering** | Steer video and audio together (`video_scale = 1.0, audio_scale = 1.0`), steer **only video** while keeping baseline dialogue/foley (`audio_scale = 0.0`), or steer **only the soundscape** (`video_scale = 0.0`). | Cannot transcribe or rewrite spoken dialogue words — dialogue words are locked by the `<|speech_start|>...<|speech_end|>` tokens in your prompt. |
| **Lock Early Composition & Geometry** | Delay steering until 33% of denoising (`start_percent = 0.33`) so the first 2 Pika steps lock identical actor pose and camera framing, applying the vector only to optical texture and lighting. | Cannot turn a single-character `ref2va` portrait into a multi-character crowd if the reference image conditioning strongly anchors a single subject at `start_percent = 0.0`. |
| **Strip Unwanted Concept Drift (`hold`)** | Project out nuisance directions (`mode = "hold"`, `--hold-words`, or `[[hold]]` pairs) such as unwanted wardrobe shifts, gender/age drift, or indoor/outdoor background bleed. | Cannot invent a brand-new character identity or face likeness from a vector alone (use `<Picture 1>` reference conditioning in `ref2va`). |

---

## 2. The Four Creation Methods & Their Mathematical Objectives

No training or weight fine-tuning ever happens when creating a control vector. Instead, each method computes a closed-form activation difference $\mathbf{v}_{\text{post}} \in \mathbb{R}^{5376}$ (and encoder direction $\hat{\mathbf{v}}_{\text{enc}} \in \mathbb{R}^{5120}$) under a distinct mathematical objective:

| Method & CLI Subcommand | Mathematical / Extraction Objective | Does `--ref-images` Apply? | Strengths | Weaknesses / Trade-offs |
|---|---|---|---|---|
| **Method 1: `from-prompt`**<br>*(In-Place Scene-Paired Contrast)*<br>`bench/control_vectors.py from-prompt` | **Conditioned Local Activation Difference on Your Exact Scene ($P$):**<br>Substitutes each `pole_a -> pole_b` clause directly inside your `--target-prompt` $P$ ($P \to P_A, P_B$), encodes both through frozen `Qwen3-VL` (`layers.49`) + `condition_proj` + 2-block `TokenRefiner`, and pools the exact changed span (`diff_span`):<br>$$\mathbf{v}_{\text{post}} = \operatorname{Pool}_{m_B}(H_{\text{post}}(P_B)) - \operatorname{Pool}_{m_A}(H_{\text{post}}(P_A))$$ | **YES (`--ref-images img1.png ...`)**: If $P$ is a `ref2va`, `i2va`, or `fl2va` prompt citing `<Picture 1>`, passing `--ref-images` prepends the real vision patch tokens (`tag == 0`) ahead of the text tokens (`tag == 1`). Because `Qwen3-VL` causal attention lets text tokens attend back to earlier `<Picture N>` patches and `TokenRefiner` runs bidirectional self-attention across all tokens, $H_{\text{post}}(P_B) - H_{\text{post}}(P_A)$ is conditioned on the actual visual appearance of `<Picture 1>`. | • **Zero cross-scene dilution**: exact prefix/suffix attention context of your shot.<br>• **Supports multi-clause & coupled `soundscape:` edits** (`;`-separated) and inline `hold:` clauses.<br>• **`~2–3 s` GPU build time** (1 pair instead of 10). | • **Scene-specific**: tailored to your target prompt $P$ rather than a multi-scene library preset.<br>• Every `pole_a` must exist verbatim in `--target-prompt`. |
| **Method 2: `from-toml`**<br>*(Multi-Scene Library Vector)*<br>`bench/control_vectors.py from-toml`<br>*(alias: `build`)* | **Expected Cross-Scene Concept Direction ($\mathbb{E}_{s \sim \mathcal{S}}[\Delta H(s)]$) with Optional Subspace Orthogonalization (`[[hold]]`):**<br>Averages per-pair span-pooled differences across $N=10$ stratified scenes (`5` interior / `5` exterior; speaking + silent + environment-only) so scene-specific nouns cancel out, then projects out the `[[hold]]` nuisance subspace $U_{\text{hold}}$:<br>$$\bar{\mathbf{v}} = \frac{1}{N}\sum_{i=1}^N \Delta \mathbf{v}_i, \quad \mathbf{v}_{\text{post}} = \left(I - U_{\text{hold}} U_{\text{hold}}^\top\right)\bar{\mathbf{v}}$$ | **Optional** (rarely needed: library `.toml` pairs are self-contained 3-section `integrated_multimodal_description:` scenes without `<Picture N>` dependencies so they compile into both `fl2va` and `ref2va` cleanly). | • **Reusable across any prompt**: nuisance scene nouns cancel across the 10 diverse pairs.<br>• **Built-in `split_half_cosine` quality metric** measures whether the 10 pairs agree on one clean axis (`>= 0.70`). | • Requires authoring 10 diverse Prompt A / Prompt B pairs (via LLM system prompt or `.toml` bank).<br>• Takes `~12 s` on GPU for 10 pairs. |
| **Method 3: `from-words`**<br>*(Instant CPU J-Lens Synthesis)*<br>`bench/control_vectors.py from-words`<br>*(alias: `words`, or inline `words` widget on `MiniMaxH3ApplyControlVectors`)* | **Linear Unigram Atlas Contrast + Linearized Jacobian Transport ($\mathbb{R}^{5120} \to \mathbb{R}^{5376}$):**<br>Looks up positive words $W_+$ and baseline words $W_-$ in the pre-extracted `24,912`-word cloze atlas $M_{\text{union}} \in \mathbb{R}^{27407 \times 5120}$, scrubs `hold_words` in $\mathbb{R}^{5120}$, and multiplies by the pre-computed `TokenRefiner` Jacobian $J_{\text{ref},\text{family}} \in \mathbb{R}^{5376 \times 5120}$:<br>$$\mathbf{v}_{\text{post}} = \alpha_{\text{cal}} \cdot J_{\text{ref},\text{family}}\,\hat{\mathbf{v}}_{\text{enc}}$$ | **No** (pure CPU lexical lookup table; zero image or prompt encoding). | • **`< 2 ms` on CPU with `0 GB` GPU VRAM**.<br>• Can be typed directly into the `words` field on `MiniMaxH3ApplyControlVectors` in ComfyUI with no `.safetensors` build step. | • **Bag-of-words linear approximation**: captures single words in isolation (`~25%–35%` of contextual variance), not multi-word syntax or coupled visual + audio scene interactions. |
| **Method 4: `from-images`**<br>*(Visual Look Transfer `Image A -> Image B`)*<br>`bench/control_vectors.py from-images`<br>*(alias: `make-media`)* | **Vision-Conditioned Cross-Attention Shift on Shared Neutral Text (`pooling = "scene_text"`):**<br>Snaps `Image A` (baseline look) and `Image B` (target graded/stylized look) to an identical 32-px M-RoPE grid (`short_edge = 512`) so both produce the exact same number of `<\|image_pad\|>` tokens ($T_{\text{vis},A} = T_{\text{vis},B}$), encodes both with the same neutral prompt $P_0$, and subtracts how `Image B` vs `Image A` shifts the downstream text conditioning tokens:<br>$$\mathbf{v}_{\text{post}} = \operatorname{Pool}_{\text{scene\_text}}(H_{\text{post}}(I_B, P_0)) - \operatorname{Pool}_{\text{scene\_text}}(H_{\text{post}}(I_A, P_0))$$ | **Required (`--image-a` and `--image-b`)**: here the two reference images *are* the contrast poles ($I_A \to I_B$). | • Extracts an optical grade, lighting mood, or texture shift directly from two visual reference frames when a look is easier to show in pictures than describe in prose. | • Best when `Image A` and `Image B` share similar subject layout and differ primarily in color grade, lighting, or surface treatment (otherwise subject differences leak into the vector). |

---

## 3. Control Vectors vs. J-Lens (`models/j_lens/j_lens_h3.safetensors`)

A **Control Vector** (`models/control_vectors/*.safetensors`) is a single `5,376`-dimensional steering direction applied at render time. **J-Lens** ([`j_lens.py`](../../j_lens.py), [`bench/control_vector/j_lens_builder.py`](../../bench/control_vector/j_lens_builder.py)) is the pre-computed `24,912 × 5,120` lexical vocabulary atlas + `TokenRefiner` Jacobian bundle (`models/j_lens/j_lens_h3.safetensors`) that powers:
1. **Decoding any Control Vector into plain English words** (`j_lens (+)` / `j_lens (-)` readouts via Wiener-regularized pseudoinverse $J_{\text{wiener}} \in \mathbb{R}^{5120 \times 5376}$ and layer-49-to-63 transport $J_{49 \to 63} \in \mathbb{R}^{5120 \times 5120}$).
2. **Pre-render readiness diagnostics** (`top1_gain`, `refiner_transmission`, `echo_ratio`, `off_target_words`).
3. **Orthogonal word scrubbing (`--hold-words`)** on CPU in `< 1 ms`.
4. **Method 3 (`from-words`) `< 2 ms` CPU unigram synthesis**.

### What Builds `models/j_lens/j_lens_h3.safetensors`?
The J-Lens bundle is built and verified by [`bench/control_vector/j_lens_builder.py`](../../bench/control_vector/j_lens_builder.py) via the `build-j-lens` CLI subcommand:
- **Verify the installed bundle (`< 50 ms`, CPU)**:
  ```bash
  "$PY" bench/control_vectors.py build-j-lens --verify-only
  ```
- **Rebuild `Sigma_49_p030`, `J_ref_{fl2va,ref2va}`, and `J_wiener_{fl2va,ref2va}` from the shipped 50-layer ComfyUI encoder + H3 `TokenRefiner` checkpoints**:
  ```bash
  "$PY" bench/control_vectors.py build-j-lens --num-prompts 24
  ```

---

## 4. Shipped Runtime Defaults (`MiniMaxH3ApplyControlVectors` & Canonical Workflows)

Every `MiniMaxH3ApplyControlVectors` node ([`h3_rules.py:266-288`](../../h3_rules.py#L266-L288), [`control_vector_nodes.py`](../../control_vector_nodes.py)) and the five canonical control-vector workflows ([`workflows/build_workflows.py`](../../workflows/build_workflows.py)) ship with these defaults:

| Input / Setting | Shipped Default | Why This Is the Default |
|---|---|---|
| **`mode`** | **`add_norm_matched`** | Normalizes the vector to `NORM_MATCHED_TARGET = 67.5` post-refiner shift norm at `strength = 1.0`, so `strength = 1.0–1.25` behaves consistently across `from-prompt`, `from-toml`, `from-words`, and `from-images` vectors without manual norm math. |
| **`mask`** | **`scene_only`** | Steers scene prose while leaving tag-0 `<Picture N>` vision tokens, `<d>[Language] ...</d>` spoken dialogue, `(S1)` speaker tags, `<Picture 1>:` labels, and `<0.0 seconds>` timestamps bitwise untouched. |
| **`ref_vae_end_percent`** (`ref2va` & `P-H3-Image` Edit) | **`1.00`** (`ref_vae_aug = 0.999`) | Keeps the 3D Video VAE / `PH3ImageVAE` reference latents (`cond_video_latents` / `minimax_refs`) pinned at `0.999` across all sampling steps so frames `0` and `1` (or edited single frames) never inherit late-step noise from `seg_t["ref_img"] = max(t_v, vis_aug)`. |
| **Same-Family Video/Image Recipe Compatibility** | **`fl2va` $\leftrightarrow$ `P-H3-Image` T2I/Edit**<br>**`ref2va` $\leftrightarrow$ Guide Edit** | [`control_vectors.py::_is_same_family_h3_recipe_compatible`](../../control_vectors.py) accepts same-family H3 checkpoints (`minimax_h3_{fl2va,ref2va}_{pruned_int8_convrot,int8_convrot,bf16}.safetensors` and `p-h3-image_int8_convrot.safetensors`) and full-refiner unit-strength LoRAs out of the box (`allow_recipe_mismatch = False`), and [`control_vector_nodes.py::resolve_vector_file_path`](../../control_vector_nodes.py) auto-selects the matching `__fl2va__`/`__ref2va__` and LoRA/base sibling. |
| **Canonical `t2va` Workflows (2)** | `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` (`strength = 1.15`) | [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json) and dual-vector [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json) (chaining `film_16mm_1974` + `chaos_panic` at `1.0` each). |
| **Canonical `ref2va` Workflow (1)** | `film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors` (`strength = 1.15`, `ref_vae_end_percent = 1.00`) | [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) on the 4-shot kitchen scene. |
| **Canonical `P-H3-Image` T2I & Edit Workflows (2)** | `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` (`strength = 1.15`) | [`workflows/image/h3_p_h3_image_t2i_control_vector_api.json`](../../workflows/image/h3_p_h3_image_t2i_control_vector_api.json) (`2912x2912` single-frame T2I) and [`workflows/image/h3_p_h3_image_edit_control_vector_api.json`](../../workflows/image/h3_p_h3_image_edit_control_vector_api.json) (multi-reference single-frame edit with `ref_vae_end_percent = 1.00`). Both T2I and 1–8 reference Image Edit use `__fl2va__` vectors because `p-h3-image_int8_convrot.safetensors` is distilled from the `fl2va` DiT backbone (`in_channels = 128`) and shares `fl2va`'s exact `condition_proj` and `token_refiner` text-path weights. |

---

## 5. Quick CLI Cheat Sheet

| Task | Command (`PY="$(cd ../.. && pwd)/.venv/bin/python"`) |
|---|---|
| **Method 1 (`from-prompt`)**: Build a scene-paired vector from in-place `pole_a -> pole_b` replacements on your prompt | `"$PY" bench/control_vectors.py from-prompt --name diner_sitcom --target-prompt my_scene.txt --words "Cinematic. -> A brightly lit 1990s multi-camera studio sitcom.; warm, amused voice -> punchy sitcom delivery" --overwrite` |
| **Method 1 (`from-prompt`) with `<Picture 1>` vision conditioning (`--ref-images`)** | `"$PY" bench/control_vectors.py from-prompt --name diner_sitcom --target-prompt my_ref2va.txt --ref-images subject1.png --words "inside a sunlit diner -> inside a 1990s sitcom diner set" --families ref2va --overwrite` |
| **Method 2 (`from-toml`)**: Build a reusable 10-pair library `.toml` or `.md` | `"$PY" bench/control_vectors.py from-toml prompt_bank/control_vectors/film_16mm_1974.toml --families both --overwrite` |
| **Method 2 (`from-toml --all`)**: Build all 32 library `.toml` banks into `models/control_vectors/` | `"$PY" bench/control_vectors.py from-toml --all --families both --overwrite` |
| **Method 3 (`from-words`)**: Synthesize a vector on CPU in `< 2 ms` via J-Lens | `"$PY" bench/control_vectors.py from-words "blizzard, snowstorm, frost" --against-words "sunlight, clear, mild" --hold-words "coat, parka" --name fast_blizzard` |
| **Method 4 (`from-images`)**: Build a look-transfer vector from `Image A -> Image B` | `"$PY" bench/control_vectors.py from-images --image-a flat.png --image-b graded.png --name look_grade --overwrite` |
| **Sculpt (`< 5 ms` CPU)**: Invert (`-1.0x`), blend, or orthogonalize (`hold`) existing vectors | `"$PY" bench/control_vectors.py sculpt film_16mm_1974 golden_hour --modes add_norm_matched,add_norm_matched --strengths 1.0,0.5 --hold-words "coat, parka" --name warm_16mm` |
| **Rebuild**: Re-extract all `.safetensors` in `models/control_vectors/` in one shared model pass | `"$PY" bench/control_vectors.py rebuild` |
| **Inspect & Prompt Meter**: Decode a vector with J-Lens & check coupling on a prompt | `"$PY" bench/control_vectors.py inspect film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors --prompt my_scene.txt` |
| **J-Lens Bundle (`build-j-lens`)**: Verify or refresh `models/j_lens/j_lens_h3.safetensors` | `"$PY" bench/control_vectors.py build-j-lens --verify-only` |

---

## 6. Sub-Guides & Reference Index

| Guide | What It Covers |
|---|---|
| **[1. Studio UI Guide](control_vectors/studio_ui_guide.md)** | Overview of the browser Studio UI (`bench/control_vectors.py ui`). |
| **[2. Building Pairs & Vectors](control_vectors/building_pairs_and_vectors.md)** | Deep dive into the 4 creation methods (`from-prompt`, `from-toml`, `from-words`, `from-images`), their mathematical objectives, when `--ref-images` applies, pooling modes (`diff_span` vs. `scene_text`), and `[[hold]]` de-biasing. |
| **[3. ComfyUI Nodes & Controls](control_vectors/comfyui_nodes_and_controls.md)** | Using `MiniMaxH3MakeControlVector`, `MiniMaxH3MakeMediaControlVector`, and `MiniMaxH3ApplyControlVectors` in ComfyUI (`strength`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `ref_vae_end_percent = 1.00`, `video_scale`, `audio_scale`, inline `words`). |
| **[4. J-Lens Diagnostics & Word Lab](control_vectors/j_lens_diagnostics_and_word_lab.md)** | How `models/j_lens/j_lens_h3.safetensors` is constructed (`build-j-lens`), `< 2 ms` CPU `from-words` synthesis, `assess_steering_readiness` observables (`top1_gain`, `refiner_transmission`, `echo_ratio`), and lexical readouts. |
| **[5. Preset Catalog & CLI Reference](control_vectors/preset_catalog_and_cli_reference.md)** | Complete CLI command & flag reference (`from-prompt`, `from-toml`, `from-words`, `from-images`, `build-j-lens`, `sculpt`, `rebuild`, `inspect`, `grade`), file naming & JSON metadata contract, and the 32 generalizable `.toml` banks in `prompt_bank/control_vectors/`. |
| **[6. Practical Examples & Recipes](control_vectors/examples_and_recipes.md)** | End-to-end recipes across `from-prompt` (including `ref2va` diner genre/atmosphere shifts), `from-toml` library presets, `from-words` CPU synthesis, `from-images` visual look transfer, and `< 5 ms` CPU sculpting. |
| **[7. Prompting Generative LLMs for Control Vectors](control_vectors/llm_prompting_guide.md)** | Copy-paste input prompts for local LLMs (`transformers`, `exllamav3`) or external LLM services using [`h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md) to author 10-pair reusable `.toml`/`.md` banks or `from-prompt` specs. |

---

## 7. Code & Research Map

- **Core math & `.safetensors` I/O**: [`control_vectors.py`](../../control_vectors.py) (`NORM_MATCHED_TARGET = 67.5`, `mean_difference`, `hold_basis`, `project_out`, `apply_post_refiner`)
- **ComfyUI nodes & runtime hooks**: [`control_vector_nodes.py`](../../control_vector_nodes.py) (`resolve_vector_file_path`, per-stream `[out_vid[0], out_aud[1]]` velocity splice when `video_scale != audio_scale`)
- **Token-span partitioner**: [`h3_spans.py`](../../h3_spans.py) (`differing_token_mask`, `scene_only` mask)
- **In-place `--target-prompt` + `--words` compiler**: [`h3_pair_compiler.py`](../../h3_pair_compiler.py) (`compile_words_to_pairset`, `compile_short_contrast_pairs`)
- **J-Lens CPU engine & bundle builder**: [`j_lens.py`](../../j_lens.py) & [`bench/control_vector/j_lens_builder.py`](../../bench/control_vector/j_lens_builder.py)
- **CLI & Studio UI server**: [`bench/control_vectors.py`](../../bench/control_vectors.py) & [`bench/control_vector/`](../../bench/control_vector/__init__.py)
- **Contract check (`C1`–`C7`)**: [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py)
- **Research memos & measured results**: [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md), [`docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md), [`docs/research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md), [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)
