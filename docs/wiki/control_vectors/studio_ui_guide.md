# 1. Control Vector Studio UI Guide

**Parent guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
**Source files**: [`bench/control_vectors.py`](../../../bench/control_vectors.py), [`bench/control_vector/ui_server.py`](../../../bench/control_vector/ui_server.py), [`bench/control_vector/web/index.html`](../../../bench/control_vector/web/index.html), [`bench/control_vector/web/style.css`](../../../bench/control_vector/web/style.css), [`bench/control_vector/web/app.js`](../../../bench/control_vector/web/app.js)

---

## Launching the Studio

Run the Studio server with the ComfyUI virtual environment's Python:

```bash
COMFY="$(cd ../.. && pwd)"
"$COMFY/.venv/bin/python" bench/control_vectors.py ui
```

Optional flags:
- `--port 8780` — bind a specific port (default `0` picks an unused local port and prints the URL).
- `--no-browser` — start the HTTP server without automatically opening a browser tab.

---

## Design Philosophy: One Screen Per Task

The Control Vector Studio is organized around **creating and testing your own control vectors**, with progressive disclosure so advanced parameters stay tucked inside collapsible **Advanced Options** drawers until you need them. Every metric and input control includes a plain-English `?` hover tooltip.

---

## Screen 1 — Vector Library & Quick Actions

Shows your saved `.safetensors` vectors (`models/control_vectors/`) and `.toml` pair banks (`prompt_bank/control_vectors/`).
- **Search & Family Filter**: Filter by name, `fl2va` (used by `t2va`/`fl2va` video and PrunaAI `P-H3-Image` T2I + 1–8 reference image editing), or `ref2va` (used by `ref2va` video and `h3_p_h3_image_guide_edit_api.json`).
- **Plain-English Quality Badges**:
  - **Pair Consistency (`MPC`)**: Mean pairwise cosine across your pairs (`>= 0.50` is strong).
  - **Split-Half Stability**: Cosine similarity between two random halves of your pairs (`>= 0.70` means all pairs agree on one clean direction).
  - **Raw Norm (`||v||`) & Recommended Strength**: Shows the vector's natural 1-pair shift magnitude and the suggested `strength` range in ComfyUI (`~1.0` for `diff_span`, or `add_norm_matched` for normalized `67.5`-norm units).
- **Quick Actions on Any Vector or Pair Pack**:
  - **Invert (`-1x`)** — Opens **Sculpt, Invert & Combine** pre-populated with `strength = -1.0` so you can immediately save the opposite direction (e.g., turning `blizzard` into `anti_blizzard`).
  - **Sculpt** — Adds the vector to a multi-vector recipe on Screen 5.
  - **Test on Prompt** — Sends the vector to **Prompt Meter & J-Lens Lab** (Contextual Prompt Meter).
  - **J-Lens Readout** — Opens the **Separate J-Lens Word Lab** sub-tab to inspect the vector's top positive/negative dictionary concepts.
  - **Edit Pairs in Studio →** (on `.toml` rows) — Loads the `.toml` pair bank into **Full Pair & LLM Studio** with the full multi-line Markdown editor.

---

## Screen 2 — Clause Contrast Builder (`Pole A -> Pole B`)

Write a short `Pole A -> Pole B` contrast to compile 10 contextual `Qwen3-VL-32B` + `TokenRefiner` scene pairs in `< 1 ms` on CPU:
- Enter your **Vector Name** (e.g., `scorching`), **Pole A (Baseline Clause)**, and **Pole B (Target Clause)**.
- Optional: open **Advanced Options** to add a **Soundscape Contrast (`Sound A -> Sound B`)** or **Hold Constant (`Hold A -> Hold B`)** clause, choose a structural slot (`auto`, `opening_framing`, `subject_behavior`, `interpersonal_dynamics`, `camera_motion`, `speech_delivery`, `soundscape`), or paste your own **Target Scene Prompt (Pair 0)** so Pair 0 is tailored to your exact shot.
- Click **Generate 10 Contrast Pairs & Inspect →** to open the **Step 2 — Review & Edit the 10 Expanded Scene Pairs** workbench directly below (with side-by-side `rows="7"` Prompt A / Prompt B textareas, per-pair weights, and **Instant CPU Re-Pool (`< 1 ms`)** via `/api/repool`), or click **Encode & Build Control Vector Now (GPU)** to compile and encode in one step (~12s).

---

## Screen 3 — Full Pair & LLM Studio (10 Contrasting H3 Prompts & Generative LLM)

Provides full control over all 10 Prompt A / Prompt B scene pairs and integrates directly with local or external generative LLMs:
1. **Generate 10 Surgical Pairs with a Local LLM (`transformers` or `exllamav3`)**:
   - Select your local backend (`transformers` or `exllamav3`), model path / HuggingFace ID, and type your concept prompt (see [`7. Prompting Generative LLMs for Control Vectors`](llm_prompting_guide.md)), then click **Generate 10 Pairs with LLM**.
2. **Import / Paste LLM Markdown (`.md`) or Load Existing `.toml`**:
   - Click **Copy LLM System Prompt** to copy [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md) (or **Insert Pair Template**), paste your LLM's 10-pair Markdown output into the large `rows="16"` editor, or select any `.toml` pack and click **Load Selected .toml**.
3. **Validate, Save `.toml`, & Encode on GPU**:
   - Click **1. Validate & Preview Token Diffs (CPU)** (`POST /api/preview_pairs`) to inspect per-pair token diffs (`n_diff_a` vs. `n_diff_b`, passing `other_ids` to `h3_spans.pooling_mask` so coupled `description + overall_soundscape` pairs highlight both spans accurately) and verify all 3 H3 sections are present.
   - Click **2. Save to `prompt_bank/control_vectors/<name>.toml`** or **3. Encode & Build Control Vector (GPU)** to save and build both `fl2va` and `ref2va` `.safetensors` vectors.

---

## Screen 4 — Reference Image Contrast (`Image A -> Image B`)

Extracts a control vector directly from two visual reference frames:
1. Drag-and-drop or click to upload **Image A (Baseline / Subtracted Look)** and **Image B (Target / Added Look)** directly in the browser (sent as base64 data URLs to `/api/make-media`, or provide server paths).
2. Choose **Output Vector Name**, **Target Families** (`ref2va` default, or `both` to emit both `__ref2va__` and `__fl2va__` siblings for `t2va` and `P-H3-Image`), **Pooling Mode** (`scene_text`, `vision_tokens`, or `all_rows`), and **Resolution (`short_edge`, default `512` px snapped to the 32-px M-RoPE grid)**.
3. Click **Extract Multimodal Image Control Vector (GPU)** (~15s). Both images are resized to the exact same token grid so `vision_tokens` and `scene_text` pooling remain aligned.

---

## Screen 5 — Sculpt, Invert & Combine Vectors (`< 5 ms` CPU)

Algebraically combine or invert existing `.safetensors` vectors on CPU without loading Qwen3-VL:
- **Quick One-Click Inversion**: Pick any source vector and click **Invert (`-1.0x`) & Save Vector** to flip its polarity immediately.
- **Multi-Vector Sculpt & Gram-Schmidt Hold Recipe**: Add any number of recipe rows with mode `add_norm_matched` (default, normalizes each input vector to `67.5` norm before weighting), `add`, or `hold` (projects out a nuisance vector with soft attenuation $\alpha \in [0, 2]$). When you select an explicit `__fl2va__` or `__ref2va__` filename and write `both` families, `_resolve_vector_for_family` (`bench/control_vector/helpers.py:249-280`) automatically swaps to each family's sibling file and stamps the matching family recipe.
- Optionally enter comma-separated **Hold Words** (e.g., `parka, snowsuit, coat`) to project out lexical drift on CPU.
- Click **Compute & Save Sculpted Vector (`< 5 ms` CPU)** to write `fl2va` and `ref2va` `.safetensors` files + `.md` reports in `< 5 ms`.

---

## Screen 6 — Prompt Receptor Meter & Separate J-Lens Word Lab

Combines contextual prompt token metering with the separate J-Lens CPU diagnostic and word synthesis lab:
- **Sub-tab A — Contextual Prompt Receptor Meter (GPU, `/api/prompt-meter`)**:
  1. Select any `.safetensors` vector from your library.
  2. Paste your full H3 prompt into the multi-line prompt editor (`rows="8"`).
  3. Click **Scan Token Alignment on Prompt** to inspect the color-coded per-token cosine & projection heatmap across your prompt tokens.
- **Sub-tab B — Separate J-Lens Word Lab (`< 2 ms` CPU)**:
  1. **J-Lens Lexical Readout**: Inspect any existing `.safetensors` vector's top `positive_words` (`+gain`) / `negative_words` (`-gain`) and `assess_steering_readiness` observables (`top1_gain`, `top3_mean_gain`, `refiner_transmission`).
  2. **J-Lens Word-Bag Synthesizer (`< 2 ms` CPU)**: Synthesize a fast exploratory vector from 3–6 English words (`pos_words`, optional `against_words`, optional `hold_words`) using the precomputed `24,912`-word J-Lens vocabulary (`models/j_lens/j_lens_h3.safetensors`).
