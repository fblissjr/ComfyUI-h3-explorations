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
- **Search & Family Filter**: Filter by name, `fl2va`, or `ref2va`.
- **Plain-English Quality Badges**:
  - **Pair Consistency (`MPC`)**: Mean pairwise cosine across your pairs (`>= 0.50` is strong).
  - **Split-Half Stability**: Cosine similarity between two random halves of your pairs (`>= 0.70` means all pairs agree on one clean direction).
  - **Raw Norm (`||v||`) & Recommended Strength**: Shows the vector's natural 1-pair shift magnitude and the suggested `strength` range in ComfyUI (`~1.0` for `diff_span`, or `add_norm_matched` for normalized `67.5`-norm units).
- **Quick Actions on Any Vector**:
  - **Invert (`-1x`)** — Opens **Sculpt, Invert & Combine** pre-populated with `strength = -1.0` so you can immediately save the opposite direction (e.g., turning `blizzard` into `anti_blizzard`).
  - **Combine / Sculpt** — Adds the vector to a multi-vector recipe.
  - **Test on Prompt** — Sends the vector to **Test Against Your Prompt (Receptor Meter)**.
  - **Edit Pairs (`.toml`)** — Loads the vector's `.toml` pair bank into **Create from Text Pairs** with full multi-line editors.

---

## Screen 2 — Create from Text Pairs (Primary Creation Flow)

Provides three authoring modes for building contextual `Qwen3-VL-32B` + `TokenRefiner` vectors:

1. **Mode A — Quick Clause Contrast (`Pole A -> Pole B`)**:
   - Enter your **Vector Name** (e.g., `scorching`), **Pole A (Baseline Clause)**, and **Pole B (Target Clause)**.
   - Optional: add a **Soundscape Contrast (`Sound A -> Sound B`)** or **Hold Constant (`Hold A -> Hold B`)** clause, or paste your own **Target Scene Prompt (Pair 0)** so Pair 0 is tailored to your exact shot.
   - Click **Expand into 10 Editable Scene Pairs** to inspect or tweak all 10 full H3 prompts in large multi-line textareas (`rows="10"`), or click **Build Vector (`.safetensors`)** to compile and encode in one step (~12s).
2. **Mode B — Full Scene Pairs Editor (10 Contrasting H3 Prompts)**:
   - Edit each pair's **Prompt A (Baseline)** and **Prompt B (Steered)** side by side in large multi-line textareas (`rows="10"`), adjust per-pair weights, toggle individual pairs on/off, or mark a pair as a `Hold` (nuisance direction to project out).
   - **Live `< 1 ms` CPU Re-Pooling**: After a build finishes, changing pair weights, disabling an outlier pair, or moving a pair to `Hold` recomputes the `.safetensors` vector in `< 1 ms` on CPU via `/api/repool` without re-running Qwen3-VL.
3. **Mode C — Import / Paste LLM Markdown (`.md`)**:
   - Click **Copy System Prompt for LLM** to copy [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md), paste your LLM's 10-pair Markdown output into the large `rows="14"` editor, click **Validate & Preview Token Diffs** to verify single-clause surgical diffs, and click **Build Vector**.

---

## Screen 3 — Create from Reference Images (`Image A -> Image B`)

Extracts a control vector directly from two visual reference frames:
1. Drag-and-drop or click to upload **Image A (Baseline Look)** and **Image B (Target / Steered Look)** directly in the browser (sent as base64 data URLs to `/api/make-media`, or provide server paths).
2. Choose **Output Vector Name**, **Target Families** (`ref2va` or `both`), **Pooling Mode** (`scene_text`, `vision_only`, or `all_rows`), and **Resolution (`short_edge`, default `512` px snapped to the 32-px M-RoPE grid)**.
3. Click **Build Media Control Vector** (~15s). Both images are resized to the exact same token grid so `vision_only` and `scene_text` pooling remain aligned.

---

## Screen 4 — Sculpt, Invert & Combine Vectors (`< 5 ms` CPU)

Algebraically combine or invert existing `.safetensors` vectors on CPU without loading Qwen3-VL:
- Add any number of recipe rows with mode `add`, `add_norm_matched` (normalizes each input vector to `67.5` norm before weighting), or `hold` (projects out a nuisance vector with soft attenuation $\alpha \in [0, 2]$).
- Use the **Invert (`-1.0`)** shortcut on any row to flip a vector's polarity.
- Optionally enter comma-separated **Hold Words** (e.g., `parka, snowsuit, coat`) to project out lexical drift on CPU.
- Click **Save Sculpted Vector** to write `fl2va` and `ref2va` `.safetensors` files + `.md` reports in `< 5 ms`.

---

## Screen 5 — Test Against Your Prompt (Receptor Meter)

Before spending GPU time on a video render, check how strongly a control vector couples to your own H3 scene prompt:
1. Select any `.safetensors` vector from your library.
2. Paste your full 6-part H3 prompt into the multi-line prompt editor (`rows="10"`).
3. Click **Run Token Receptor Scan** (`/api/prompt-meter`).
4. Inspect the color-coded per-token cosine & projection heatmap across your prompt tokens (`scene_text`, `non_vision`, or `all_rows`).

---

## Screen 6 — J-Lens Word Lab & Diagnostics *(Separate Advanced Tool)*

Kept intentionally separate from core pair-based vector creation:
- **Sub-tab 6A — Unigram Word Synthesizer (`< 2 ms` CPU)**: Synthesize a fast exploratory vector from 3–6 English words (`pos_words`, optional `against_words` or `@prompt_subject`, optional `hold_words`) using the precomputed `24,912`-word J-Lens vocabulary (`models/j_lens/j_lens_h3.safetensors`).
- **Sub-tab 6B — Vector Lexical Readout & Readiness Inspector**: Inspect any existing `.safetensors` vector's top `positive_words` (`+gain`) / `negative_words` (`-gain`) and `assess_steering_readiness` observables (`top1_gain`, `top3_mean_gain`, `refiner_transmission`).
