# Control Vectors for MiniMax H3 — Start Here

Activation control vectors steer MiniMax H3 video + audio generation by adding, projecting out, or amplifying a **5376-dimensional direction** in the post-`TokenRefiner` text-conditioning space — **without changing your prompt text or token length $L$**.

Because Qwen3-VL causal attention changes every downstream token whenever you edit a single word in a prompt, ordinary prompt edits reshuffle camera framing, actor blocking, and lip-sync timing on the same seed. A control vector leaves your prompt tokens untouched and shifts only the semantic concept direction (weather, lighting, film stock, era props, motion energy, or acoustic space) across the tokens you choose.

---

## 1. What Control Vectors CAN and CANNOT Do

| Capability | What Control Vectors **CAN** Do Well | What Control Vectors **CANNOT** Do (Use Another Tool Instead) |
|---|---|---|
| **Atmosphere, Weather & Lighting** | Shift a scene into a blizzard, scorching heatwave, golden hour, sodium-vapor rain, noir chiaroscuro, or 16mm film grain while keeping the same shot composition. | Cannot place a specific object at exact pixel coordinates $(x, y)$ or enforce a frame-accurate silhouette mask (use the [masked V2V lane](../masked_v2v/AGENTS.md) or reference images). |
| **Invert or Combine Any Concept** | Invert any existing vector (`strength = -1.0`, e.g., flipping `blizzard` toward warm/dry air) or build its true opposite (`"calm clear air -> scorching desert heatwave"`), and stack up to 6 vectors with orthogonal holds. | Multiplying `blizzard` by `-1.0` removes snow/cold toward neutral dry air, whereas authoring `"clear mild day -> scorching desert heatwave with shimmering heat haze"` actively injects heat haze and sun glare. |
| **Independent Video vs. Audio Steering** | Steer video and audio together (`video_scale = 1.0, audio_scale = 1.0`), steer **only video** while keeping baseline dialogue/foley (`audio_scale = 0.0`), or steer **only the soundscape** (`video_scale = 0.0`). | Cannot transcribe or rewrite spoken dialogue words — dialogue words are locked by the `<|speech_start|>...<|speech_end|>` tokens in your prompt. |
| **Lock Early Composition & Geometry** | Delay steering until 33% of denoising (`start_percent = 0.33`) so the first 2 Pika steps lock identical actor pose and camera framing, applying the vector only to optical texture and lighting. | Cannot turn a single-character `ref2va` portrait into a multi-character crowd if the reference image conditioning strongly anchors a single subject at `start_percent = 0.0`. |
| **Strip Unwanted Concept Drift (`hold`)** | Project out nuisance directions (`mode = "hold"`, `--hold-words`, or `[[hold]]` pairs) such as unwanted wardrobe shifts, gender/age drift, or indoor/outdoor background bleed. | Cannot invent a brand-new character identity or face likeness from a vector alone (use `<Picture 1>` reference conditioning in `ref2va`). |

---

## 2. Shipped Runtime Defaults (`MiniMaxH3ApplyControlVectors` & Canonical Workflows)

Every `MiniMaxH3ApplyControlVectors` node ([`h3_rules.py:266-288`](../../h3_rules.py#L266-L288), [`control_vector_nodes.py`](../../control_vector_nodes.py)) and the three canonical Pika control-vector workflows ([`workflows/build_workflows.py:4606-4642`](../../workflows/build_workflows.py#L4606-L4642)) ship with these defaults:

| Input / Setting | Shipped Default | Why This Is the Default |
|---|---|---|
| **`mode`** | **`add_norm_matched`** | Normalizes the vector to `NORM_MATCHED_TARGET = 67.5` post-refiner shift norm at `strength = 1.0`, so `strength = 1.0–1.25` behaves consistently across `diff_span`, `scene_text`, `from_first_diff`, and J-Lens vectors without manual norm math. |
| **`mask`** | **`scene_only`** | Steers scene prose while leaving tag-0 `<Picture N>` vision tokens, `<d>[Language] ...</d>` spoken dialogue, `(S1)` speaker tags, `<Picture 1>:` labels, and `<0.0 seconds>` timestamps bitwise untouched. |
| **`ref_vae_end_percent`** (`ref2va`) | **`0.50`** | On 6-step `ref2va` renders, keeps the 3D Video VAE reference latents (`cond_video_latents`) active on steps `0..2` to lock `<Subject 1>`'s face, hair, and clothing, then gates them on steps `3..5` so lighting/film-stock vectors harmonize across `<Subject 1>`. *(Set to **`0.00`** when overriding `<Subject 1>`'s clothing/wardrobe without editing the prompt; see [Practical Examples & Recipes](control_vectors/examples_and_recipes.md#example-1-changing-subject-1s-clothing-in-ref2va-without-editing-the-prompt)).* |
| **Canonical `t2va` Workflow** | `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` (`strength = 1.15`) | [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json) (and dual-vector [`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`](../../workflows/h3_text_to_video_pika_dual_control_vectors_api.json) chaining `film_16mm_1974` + `chaos_panic` at `1.0` each). |
| **Canonical `ref2va` Workflow** | `film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors` (`strength = 1.15`, `ref_vae_end_percent = 0.50`) | [`workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json) on the 4-shot kitchen scene. |

---

## 3. How to Create Your Own Control Vector (Three Ways)

You do **not** need pre-canned scenes from `prompt_bank/` to use control vectors. The Studio UI and CLI are built for authoring **your own** vectors:

### Way A — From a Short Clause Contrast (`Pole A -> Pole B`) *(Recommended, ~12s)*
Write the baseline phrase (`Pole A`) and the target phrase (`Pole B`), for example:
- **Pole A (Baseline)**: `bathed in cool overcast daylight with crisp neutral air`
- **Pole B (Target)**: `drenched in scorching desert heatwave glare with shimmering heat-haze mirages`

The pair compiler ([`h3_pair_compiler.py`](../../h3_pair_compiler.py)) automatically embeds your contrast into **10 diverse, structurally matched H3 scene templates** (and optionally your own target prompt as Pair 0), runs them through the actual contextual **Qwen3-VL-32B (`layers.49`) + 2-block `TokenRefiner`**, and saves both `fl2va` and `ref2va` `.safetensors` vectors (with default Pika LoRA branches automatically attached unless `--no-lora` is passed).

### Way B — From Full Custom or LLM-Authored Prompt Pairs (`.md` / `.toml`) *(Highest Precision, ~12s)*
When you want complete control over all 10 Prompt A / Prompt B scenes (for example, complex multi-clause physical transformations, wardrobe overrides with `[[hold]]` pairs, or custom domain scenes):
1. Follow **[7. Prompting Generative LLMs for Control Vectors](control_vectors/llm_prompting_guide.md)** (which provides copy-paste input prompt templates for [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md)) using either our built-in local LLM generator (`transformers` or `exllamav3` in **Studio Screen 3** / `generate-pairs`) or any external LLM service.
2. Each pair holds all non-target sections byte-identical between Prompt A and Prompt B so `pooling = "diff_span"` isolates only your changed clause.

### Way C — From Browser-Uploaded Reference Images (`Image A -> Image B`) *(Visual Look Extraction, ~15s)*
Upload or drag-and-drop two contrasting reference images directly in the **Studio UI → Create from Reference Images** screen (or pass `--image-a` and `--image-b` to `make-media`):
- **Image A**: Baseline look (e.g., flat daylight frame)
- **Image B**: Target look (e.g., graded film frame or stylized lighting reference)
Both images are snapped to an identical 32-px M-RoPE vision token grid (`short_edge = 512`) and encoded through `Qwen3-VL` + `TokenRefiner` to extract the visual difference direction.

---

## 4. End-to-End Walkthrough Examples (See [Full Examples & Recipes](control_vectors/examples_and_recipes.md))

Below are three common end-to-end recipes. For all 6 detailed recipes (including media look transfer, multi-vector sculpting, and `< 2 ms` CPU word synthesis), see **[6. Practical Examples & Recipes](control_vectors/examples_and_recipes.md)**.

### Example A: Changing `<Subject 1>`'s Clothing in `ref2va` Without Editing the Prompt
To override `<Subject 1>`'s outfit from `<Picture 1>` (for example, shifting everyday indoor/winter attire to summer resort beachwear) while keeping `<Subject 1>`'s face and the kitchen/room scene intact:
```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"

"$PY" bench/control_vectors.py build \
  --words "opening_framing: dressed in everyday indoor casual layers and street attire -> dressed in sun-drenched summer beachwear and tropical resort swimwear" \
  --name summer_beachwear \
  --families ref2va \
  --overwrite
```
In ComfyUI (`MiniMaxH3ApplyControlVectors` on `ref2va`), set:
- `vector_name = "summer_beachwear__ref2va__diff_span__lora_865ee189.safetensors"`
- `strength = 1.35`, `mode = "add_norm_matched"`, `mask = "scene_only"`
- **`ref_vae_end_percent = 0.00`** *(releases the 3D Video VAE pixel copy-lock on `<Picture 1>`'s original clothing from Step 0 while Qwen3-VL vision tokens preserve `<Subject 1>`'s facial identity)*.

### Example B: Coupled Visual + Audio Atmosphere (`scorching_heatwave`)
```bash
"$PY" bench/control_vectors.py build \
  --words "opening_framing: bathed in cool overcast daylight with crisp neutral air -> drenched in scorching desert heatwave glare with shimmering heat-haze mirages; soundscape: quiet mild afternoon air -> dry desert wind and high-pitched cicada heat drone" \
  --name scorching_heatwave \
  --families both \
  --save-toml \
  --overwrite
```
This writes `prompt_bank/control_vectors/scorching_heatwave.toml` plus both `fl2va` (`__lora_2a8c2b52`) and `ref2va` (`__lora_865ee189`) `.safetensors` files. In ComfyUI, apply at `strength = 1.15, mode = "add_norm_matched", mask = "scene_only"`.

### Example C: Inverting or Combining Vectors in `< 5 ms` on CPU (`sculpt`)
```bash
# Invert blizzard (-1.0x) into a new saved vector:
"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard

# Combine scorching_heatwave (+1.0) + golden_hour (+0.5) while scrubbing out winter coat drift:
"$PY" bench/control_vectors.py sculpt scorching_heatwave golden_hour \
  --modes add_norm_matched,add_norm_matched \
  --strengths 1.0,0.5 \
  --hold-words "coat, parka, heavy clothing" \
  --name scorching_golden
```

---

## 5. Launching the Control Vector Studio UI

```bash
COMFY="$(cd ../.. && pwd)"
"$COMFY/.venv/bin/python" bench/control_vectors.py ui
```

The Studio UI opens in your browser and is organized into **6 focused task screens** so you only see the controls you need when you need them:
1. **Vector Library & Quick Actions** — Browse built `.safetensors` and `.toml` banks, inspect quality metrics (`split_half_cosine`, `raw_norm`, recommended strength), or one-click **Invert (`-1x`)** / **Open in Sculpt**.
2. **Quick Clause Builder (`Pole A -> Pole B`)** — Build a new vector from a `Pole A -> Pole B` contrast (with optional `soundscape:` and `hold:` clauses and `target_prompt` Pair 0 injection).
3. **Full Pair & LLM Studio** — Edit all 10 Prompt A / Prompt B pairs side by side, generate pairs with a local LLM (`transformers` or `exllamav3`), or paste Markdown from any external LLM with live token-diff highlighting and `< 1 ms` CPU pair re-pooling.
4. **Media Contrast (`Image A -> Image B`)** — Drag-and-drop **Image A (Baseline)** and **Image B (Target)** in the browser to build an image-contrast vector.
5. **Subspace Sculptor (`< 5 ms` CPU)** — Stack, invert, or orthogonalize (`hold`) existing vectors in `< 5 ms` on CPU.
6. **J-Lens & Receptor Meter** — Paste your own H3 scene prompt to see per-token cosine alignment before rendering, or run `< 2 ms` CPU unigram word synthesis and lexical readouts (`models/j_lens/j_lens_h3.safetensors`).

---

## 6. Sub-Guides & Reference Index

| Guide | What It Covers |
|---|---|
| **[1. Studio UI Guide](control_vectors/studio_ui_guide.md)** | Screen-by-screen walkthrough of the 6 Studio screens, progressive disclosure, tooltips, and interactive `< 1 ms` pair re-pooling. |
| **[2. Building Pairs & Vectors](control_vectors/building_pairs_and_vectors.md)** | How pairs are built: `Pole A -> Pole B` template compilation (`h3_pair_compiler.py`), the LLM system prompt (`h3_control_vector_system_prompt.md`), why Qwen3-VL + `TokenRefiner` is used, pooling modes (`diff_span` vs. `scene_text`), and `[[hold]]` de-biasing. |
| **[3. ComfyUI Nodes & Controls](control_vectors/comfyui_nodes_and_controls.md)** | Using `MiniMaxH3MakeControlVector`, `MiniMaxH3MakeMediaControlVector`, and `MiniMaxH3ApplyControlVectors` in ComfyUI (`strength`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `ref_vae_end_percent = 0.50`, `video_scale`, `audio_scale`, `words`). |
| **[4. J-Lens Diagnostics & Word Lab](control_vectors/j_lens_diagnostics_and_word_lab.md)** | How J-Lens (`j_lens.py`) differs from full 10-pair control vectors, when to use `< 2 ms` CPU word synthesis vs. 10-pair Qwen builds, and how to read `assess_steering_readiness` observables (`top1_gain`, `refiner_transmission`, `echo_ratio`) and lexical readouts. |
| **[5. Preset Catalog & CLI Reference](control_vectors/preset_catalog_and_cli_reference.md)** | Complete CLI command & flag reference (`--dit`, `--words`, `--save-toml`, `generate-pairs`, `make-media`, `sculpt --invert`, `inspect`, `grade`), file naming & JSON metadata contract, and the starter `.toml` banks in `prompt_bank/control_vectors/`. |
| **[6. Practical Examples & Recipes](control_vectors/examples_and_recipes.md)** | 6 end-to-end recipes: `ref2va` clothing override (`ref_vae_end_percent = 0.00`), 1974 16mm film stock (`ref_vae_end_percent = 0.50`), coupled visual + soundscape weather, reference-image look transfer, multi-vector sculpting with holds, and `< 2 ms` word synthesis. |
| **[7. Prompting Generative LLMs for Control Vectors](control_vectors/llm_prompting_guide.md)** | Copy-paste input prompts for local LLMs (`transformers`, `exllamav3`) or external LLM services using [`h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md) across atmosphere, wardrobe/material overrides, physics, camera motion, and `--words` specs. |

---

## 6. Common Questions (FAQ)

- **Do I have to use pre-canned `prompt_bank/` scenes like the violin busker scene?**
  **No.** The Studio UI and CLI are designed to build control vectors for **your own** concepts and scenes. In **Studio Screen 2 (Create from Text Pairs)**, you can paste your own target scene prompt as Pair 0 (`target_prompt`), write your own `Pole A -> Pole B` contrast, or edit all 10 Prompt A / Prompt B pairs directly. In **Studio Screen 5 (Test Against Your Prompt)**, you test any vector against your own H3 prompt.
- **How do I make the opposite of an existing preset (e.g., `"scorching"` vs. `"blizzard"`)?**
  - To **flip** `blizzard` immediately (`< 5 ms` CPU), click **Invert (`-1x`)** in the Studio Library or run `"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard` (or set `strength = -1.0` in ComfyUI).
  - To **author a true opposite concept** with its own positive optical signature (shimmering heat haze, sun glare, parched ground), build `"calm mild air -> scorching desert heatwave"` via **Studio Screen 2** or `bench/control_vectors.py build --words ... --name scorching --save-toml`.
- **Are J-Lens and Control Vectors separate?**
  **Yes.** Core control vectors encode 10 full contrasting H3 scenes through the 32B contextual `Qwen3-VL` (`layers.49`) + `TokenRefiner`. **J-Lens (`j_lens.py`)** is an optional, separate `< 2 ms` CPU unigram lookup table (`24,912` single words) and linear readout (`models/j_lens/j_lens_h3.safetensors`). It lives in its own separate Studio screen (**Screen 6**) and sub-guide ([`j_lens_diagnostics_and_word_lab.md`](control_vectors/j_lens_diagnostics_and_word_lab.md)).
- **Does pair compilation (`--words`) call an LLM?**
  **No.** `--words "pole_a -> pole_b"` uses the deterministic template compiler ([`h3_pair_compiler.py`](../../h3_pair_compiler.py)) in `< 1 ms` on CPU, and then encodes the 10 compiled scenes through `Qwen3-VL-32B` + `TokenRefiner`. When you want an LLM to author 10 bespoke multi-clause scenes, use [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md).
- **Why does the CLI use `--dit` and `--check-dit` instead of `--unet`?**
  MiniMax H3 is a 50-block joint audio-video **Diffusion Transformer (DiT)** (`MiniMaxH3Model`), even though ComfyUI's internal folder is named `diffusion_models` / `UNETLoader`. The CLI uses `--dit` and `--check-dit` (while keeping `--unet` and `--check-unet` as hidden backwards-compatible aliases).

---

## 7. Quick CLI Cheat Sheet

| Task | Command (`PY="$(cd ../.. && pwd)/.venv/bin/python"`) |
|---|---|
| Launch the 6-screen Studio UI | `"$PY" bench/control_vectors.py ui` |
| Build a vector from `Pole A -> Pole B` | `"$PY" bench/control_vectors.py build --words "calm air -> scorching heatwave" --name scorching --save-toml --overwrite` |
| Build a vector from `.toml` or `.md` | `"$PY" bench/control_vectors.py build prompt_bank/control_vectors/blizzard.toml --families both --overwrite` |
| Build a vector from two reference images | `"$PY" bench/control_vectors.py make-media --image-a flat.png --image-b graded.png --output-name look_grade --overwrite` |
| Invert an existing vector (`-1.0x`) | `"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard` |
| Combine vectors + project out hold words | `"$PY" bench/control_vectors.py sculpt scorching golden_hour --modes add,add --strengths 1.0,0.5 --hold-words "coat, parka" --name desert_gold` |
| Inspect a vector & meter a target prompt | `"$PY" bench/control_vectors.py inspect scorching__fl2va__pika_v2_2__diff_span.safetensors --prompt "A wide shot..."` |

---

## 8. Code & Research Map

- **Core math & `.safetensors` I/O**: [`control_vectors.py`](../../control_vectors.py) (`NORM_MATCHED_TARGET = 67.5`, `mean_difference`, `hold_basis`, `project_out`, `apply_vector`)
- **ComfyUI nodes & runtime hooks**: [`control_vector_nodes.py`](../../control_vector_nodes.py) (`resolve_vector_file_path`, per-stream `[out_vid[0], out_aud[1]]` velocity splice when `video_scale != audio_scale`)
- **Token-span partitioner**: [`h3_spans.py`](../../h3_spans.py)
- **Deterministic 10-pair compiler**: [`h3_pair_compiler.py`](../../h3_pair_compiler.py) & [`prompt_bank/control_vectors/canonical_frames.json`](../../prompt_bank/control_vectors/canonical_frames.json)
- **J-Lens CPU engine**: [`j_lens.py`](../../j_lens.py)
- **CLI & Studio UI server**: [`bench/control_vectors.py`](../../bench/control_vectors.py) & [`bench/control_vector/`](../../bench/control_vector/__init__.py) ([`web/index.html`](../../bench/control_vector/web/index.html), [`web/style.css`](../../bench/control_vector/web/style.css), [`web/app.js`](../../bench/control_vector/web/app.js))
- **Contract check (`C7`)**: [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py)
- **Research memos & measured results**: [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md), [`docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md), [`docs/research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md), [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)
