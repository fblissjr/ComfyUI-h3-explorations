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

## 2. How to Create Your Own Control Vector (Three Ways)

You do **not** need pre-canned scenes from `prompt_bank/` to use control vectors. The Studio UI and CLI are built for authoring **your own** vectors:

### Way A — From a Short Clause Contrast (`Pole A -> Pole B`) *(Recommended, ~12s)*
Write the baseline phrase (`Pole A`) and the target phrase (`Pole B`), for example:
- **Pole A (Baseline)**: `calm mild afternoon air with neutral daylight`
- **Pole B (Target)**: `scorching desert heatwave with shimmering heat-haze mirages and blinding sun glare`

The pair compiler ([`h3_pair_compiler.py`](../../h3_pair_compiler.py)) automatically embeds your contrast into **10 diverse, structurally matched 6-part H3 scene templates** (and optionally your own target prompt as Pair 0), runs them through the actual contextual **Qwen3-VL-32B (`layers.49`) + 2-block `TokenRefiner`**, and saves both `fl2va` and `ref2va` `.safetensors` vectors.

### Way B — From Full Custom or LLM-Authored Prompt Pairs (`.md` / `.toml`) *(Highest Precision, ~12s)*
When you want complete control over all 10 Prompt A / Prompt B scenes (for example, complex multi-clause physical transformations or custom `[[hold]]` pairs):
1. Copy the canonical system prompt ([`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](system_prompts/h3_control_vector_system_prompt.md)) into any LLM — or write/edit the 10 Prompt A / Prompt B pairs directly in the **Studio UI → Create from Text Pairs → Full Scene Pairs** editor.
2. Each pair holds 5 of the 6 H3 prompt sections byte-identical between Prompt A and Prompt B so `pooling = "diff_span"` isolates only your changed clause.

### Way C — From Browser-Uploaded Reference Images (`Image A -> Image B`) *(Visual Look Extraction, ~15s)*
Upload or drag-and-drop two contrasting reference images directly in the **Studio UI → Create from Reference Images** screen (or pass `--image-a` and `--image-b` to `make-media`):
- **Image A**: Baseline look (e.g., flat daylight frame)
- **Image B**: Target look (e.g., graded film frame or stylized lighting reference)
Both images are snapped to an identical 32-px M-RoPE vision token grid (`short_edge = 512`) and encoded through `Qwen3-VL` + `TokenRefiner` to extract the visual difference direction.

---

## 3. Quick Example: Creating `"scorching"` (or Inverting `"blizzard"`)

### Option 1: Author a dedicated `"scorching"` vector (Studio UI or CLI)
```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"

"$PY" bench/control_vectors.py build \
  --words "calm clear mild air -> scorching desert heatwave with shimmering heat haze and harsh sun glare; soundscape: quiet outdoor air -> dry desert wind and cicada drone" \
  --name scorching \
  --save-toml \
  --overwrite
```
This writes:
- `prompt_bank/control_vectors/scorching.toml` (re-editable 10-pair bank)
- `models/control_vectors/scorching__fl2va__pika_v2_2__diff_span.safetensors`
- `models/control_vectors/scorching__ref2va__pika_r2v_v1_1__diff_span.safetensors`

### Option 2: Invert or blend existing vectors in `< 5 ms` (`sculpt`)
If you want the exact mathematical inverse of `blizzard` (or want to blend `scorching` with `golden_hour` while holding `wardrobe` constant):
```bash
# Invert blizzard (-1.0x) into a new saved vector:
"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard

# Combine scorching (+1.0) + golden_hour (+0.5) while projecting out wardrobe drift:
"$PY" bench/control_vectors.py sculpt scorching golden_hour \
  --modes add,add \
  --strengths 1.0,0.5 \
  --hold-words "coat, parka, heavy clothing" \
  --name scorching_golden
```

---

## 4. Launching the Control Vector Studio UI

```bash
COMFY="$(cd ../.. && pwd)"
"$COMFY/.venv/bin/python" bench/control_vectors.py ui
```

The Studio UI opens in your browser and is organized into **6 focused task screens** so you only see the controls you need when you need them:
1. **Vector Library & Quick Actions** — Browse built `.safetensors` and `.toml` banks, inspect quality metrics (`split_half_cosine`, `raw_norm`, recommended strength), or one-click **Invert (`-1x`)** / **Open in Sculpt**.
2. **Create from Text Pairs** — Build a new vector from a `Pole A -> Pole B` contrast or full multi-line Prompt A / Prompt B pairs (`rows="10"` editors), with live token-diff highlighting and `< 1 ms` CPU pair re-weighting.
3. **Create from Reference Images** — Drag-and-drop **Image A (Baseline)** and **Image B (Target)** in the browser to build an image-contrast vector.
4. **Sculpt, Invert & Combine** — Stack, invert, or orthogonalize (`hold`) existing vectors in `< 5 ms` on CPU.
5. **Test Against Your Prompt (Receptor Meter)** — Paste your own H3 scene prompt to see per-token cosine alignment before rendering.
6. **J-Lens Word Lab & Diagnostics *(Advanced / Separate Tool)*** — Optional `< 2 ms` CPU unigram word synthesizer (`24,912`-word vocabulary) and lexical decoder (`models/j_lens/j_lens_h3.safetensors`). Kept completely separate from core Qwen3-VL pair building.

---

## 5. Sub-Guides & Reference Index

| Guide | What It Covers |
|---|---|
| **[1. Studio UI Guide](control_vectors/studio_ui_guide.md)** | Screen-by-screen walkthrough of the 6 Studio screens, progressive disclosure, tooltips, and interactive `< 1 ms` pair re-pooling. |
| **[2. Building Pairs & Vectors](control_vectors/building_pairs_and_vectors.md)** | How pairs are built: `Pole A -> Pole B` template compilation (`h3_pair_compiler.py`), the LLM system prompt (`h3_control_vector_system_prompt.md`), why Qwen3-VL + `TokenRefiner` is used, pooling modes (`diff_span` vs. `scene_text`), and `[[hold]]` de-biasing. |
| **[3. ComfyUI Nodes & Controls](control_vectors/comfyui_nodes_and_controls.md)** | Using `MiniMaxH3MakeControlVector`, `MiniMaxH3MakeMediaControlVector`, and `MiniMaxH3ApplyControlVectors` in ComfyUI (`strength`, `mode`, `mask`, `start_percent`, `video_scale`, `audio_scale`, `words`). |
| **[4. J-Lens Diagnostics & Word Lab](control_vectors/j_lens_diagnostics_and_word_lab.md)** | How J-Lens (`j_lens.py`) differs from full 10-pair control vectors, when to use `< 2 ms` CPU word synthesis vs. 10-pair Qwen builds, and how to read `assess_steering_readiness` observables (`top1_gain`, `refiner_transmission`, `echo_ratio`) and lexical readouts. |
| **[5. Preset Catalog & CLI Reference](control_vectors/preset_catalog_and_cli_reference.md)** | Complete CLI command & flag reference (`--dit`, `--words`, `--save-toml`, `make-media`, `sculpt --invert`, `inspect`, `grade`), file naming & JSON metadata contract, and the 32 starter `.toml` banks in `prompt_bank/control_vectors/`. |

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
- **CLI & Studio UI server**: [`bench/control_vectors.py`](../../bench/control_vectors.py) & [`bench/control_vector_ui.html`](../../bench/control_vector_ui.html)
- **Contract check (`C7`)**: [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py)
- **Research memos & measured results**: [`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md), [`docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md), [`docs/research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md), [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)
