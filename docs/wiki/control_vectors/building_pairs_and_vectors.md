# 2. Building Pairs & Control Vectors

[← Back to Control Vectors Overview](../control_vectors.md)

This guide explains how MiniMax H3 control vectors are constructed, the mathematical objective of each of the **4 creation methods** (`from-prompt`, `from-toml`, `from-words`, `from-images`), how `--ref-images` vision conditioning works, how token-span pooling (`diff_span`) isolates your change, and how orthogonal `hold` projections remove unwanted concept drift.

---

## 1. Why Control Vectors Are Extracted After `TokenRefiner`

In MiniMax H3 (`fl2va`, `ref2va`, and single-frame `P-H3-Image`), text and reference-image conditioning passes through two stages before entering the 50-block joint audio-video Diffusion Transformer (DiT):
1. **Stage 1 — Frozen `Qwen3-VL-32B` (`layers.49`, $d_{\text{enc}} = 5120$)**:
   Encodes any `<Picture N>` vision patches (`tag == 0`) followed by the prompt text (`tag == 1`) under causal self-attention. Because attention is causal (`i >= j`), every text token after a `<Picture N>` vision block or an edited clause changes its `5,120`-d representation.
2. **Stage 2 — Per-Family `condition_proj` + 2-Block Bidirectional `TokenRefiner` ($d_{\text{post}} = 5376$)**:
   Projects the `5,120`-d states to `5,376`-d (plus optional Pika / Image-Edit LoRA deltas on `condition_proj` and `token_refiner`) and runs 2 blocks of **bidirectional** self-attention across the sequence.

Every `.safetensors` control vector stores both:
- **`vector.post_refiner` (`float32[5376]`)**: The primary steering vector injected at runtime right after `token_refiner` (before the 50 DiT blocks), in natural mean-pair-difference units (`raw_norm ≈ 49..87` for `diff_span`, normalized to `NORM_MATCHED_TARGET = 67.5` at runtime under `mode = "add_norm_matched"`).
- **`vector.encoder` (`float32[5120]`, unit-normalized, `row 0` excluded)**: The pre-refiner direction used by [`j_lens.py`](../../../j_lens.py) for lexical readouts, readiness scoring, and cross-family diagnostics.

---

## 2. The Four Creation Methods & Their Mathematical Objectives

### Method 1: `from-prompt` — In-Place Scene-Paired Contrast (`--target-prompt` + `--words`)

```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"

"$PY" bench/control_vectors.py from-prompt \
  --name diner_sitcom \
  --target-prompt my_diner_ref2va.txt \
  --ref-images subject1.png \
  --words "The target video uses a live-action cinematic style with warm natural sunlight. -> The target video uses a brightly lit 1990s multi-camera studio sitcom style with flat three-point stage lighting.; warm, amused voice -> punchy comedic sitcom delivery; soundscape: Atmospheric diner sounds and faint murmurs of distant patrons. -> Warm studio room tone with a live studio audience laughing uproariously after her punchline." \
  --families ref2va --overwrite
```

- **How it works ([`h3_pair_compiler.py::compile_words_to_pairset`](../../../h3_pair_compiler.py))**:
  1. Takes your exact `--target-prompt` $P$ as `Prompt A` ($P_A = P$).
  2. Substitutes each `pole_a -> pole_b` clause in place inside $P$ to form `Prompt B` ($P_B$). Multi-clause edits separated by `;` or newlines are applied together; `soundscape: pole_a -> pole_b` substitutes directly inside `overall_soundscape:`; and `hold(group): pole_a -> pole_b` compiles an in-place hold pair $(P, P_{\text{hold}})$ on the same target prompt.
  3. Encodes $(P_A, P_B)$ through `Qwen3-VL` + `TokenRefiner` and pools only the changed token span(s) (`pooling = "diff_span"`).
- **Mathematical Objective**:
  $$\mathbf{v}_{\text{post}} = \operatorname{Pool}_{m_B}\!\left(H_{\text{post}}(P_B, I_{\text{ref}})\right) - \operatorname{Pool}_{m_A}\!\left(H_{\text{post}}(P_A, I_{\text{ref}})\right)$$
  By holding 100% of your scene's surrounding prefix and suffix tokens identical between $P_A$ and $P_B$, `from-prompt` isolates the exact directional shift of the replacement clause **conditioned on your scene's own attention context**—and then applies that shift back onto the unedited prompt $P_A$ at render time without changing token count $L$ or reshuffling seed geometry.
- **When `--ref-images` Applies**:
  When your `--target-prompt` is a `ref2va`, `i2va`, or `fl2va` prompt referencing `<Picture 1>` (or `Picture 1` / `Picture 2`), passing `--ref-images img1.png [img2.png ...]` encodes the actual reference images as `tag == 0` vision patch tokens ahead of the `tag == 1` text tokens during extraction. Because `Qwen3-VL` causal attention lets text tokens attend to earlier `<Picture N>` patches and `TokenRefiner` runs bidirectional self-attention across all tokens, $H_{\text{post}}(P_B, I_{\text{ref}}) - H_{\text{post}}(P_A, I_{\text{ref}})$ is conditioned on the real visual appearance of `<Picture 1>`.

---

### Method 2: `from-toml` — Multi-Scene Reusable Library Vector (`10` Stratified Pairs)

```bash
# Build a single reusable 10-pair library .toml (or .md from the LLM system prompt)
"$PY" bench/control_vectors.py from-toml prompt_bank/control_vectors/film_16mm_1974.toml --families both --overwrite

# Or build all 32 generalizable library .toml files in prompt_bank/control_vectors/
"$PY" bench/control_vectors.py from-toml --all --families both --overwrite
```

- **How it works ([`control_vectors.py::mean_difference`](../../../control_vectors.py))**:
  Encodes $N = 10$ stratified contrastive scene pairs (`Pair 1`..`Pair 10`: 5 interior / 5 exterior, mixing speaking `(S1)`, silent human, and environment-only shots) where each pair shares the same abstract concept shift ($B - A$) in a completely different setting with zero verbatim span cloning.
- **Mathematical Objective**:
  $$\bar{\mathbf{v}}_{\text{post}} = \frac{1}{\sum_{i=1}^N w_i} \sum_{i=1}^N w_i \left(\operatorname{Pool}_{m_{B,i}}(H_{\text{post}}(P_{B,i})) - \operatorname{Pool}_{m_{A,i}}(H_{\text{post}}(P_{A,i}))\right)$$
  $$\mathbf{v}_{\text{post}} = \left(I - U_{\text{hold}} U_{\text{hold}}^\top\right) \bar{\mathbf{v}}_{\text{post}}$$
  Averaging across 10 unrelated scenes cancels out scene-specific nouns (mugs, trees, rooms, clothing) and isolates the **expected cross-scene concept direction** $\mathbb{E}_{s \sim \mathcal{S}}[\Delta H(s)]$. The built-in **`split_half_cosine`** diagnostic measures the cosine similarity between the average of odd pairs (`1, 3, 5, 7, 9`) and even pairs (`2, 4, 6, 8, 10`)—high values (`>= 0.70`) confirm that all 10 pairs converge on a single reusable direction.

---

### Method 3: `from-words` — Instant `< 2 ms` CPU Unigram Synthesis via J-Lens

```bash
"$PY" bench/control_vectors.py from-words "blizzard, snowstorm, frost" \
  --against-words "sunlight, clear, mild" \
  --hold-words "coat, parka" \
  --name fast_blizzard
```

- **How it works ([`j_lens.py::synthesize_from_words`](../../../j_lens.py))**:
  Looks up positive words $W_+$ and baseline words $W_-$ in the pre-computed `24,912`-word cloze vocabulary atlas (`M_union` in `models/j_lens/j_lens_h3.safetensors`), subtracts the baseline mean, projects out `--hold-words` in $\mathbb{R}^{5120}$, and transports the unit encoder direction $\hat{\mathbf{v}}_{\text{enc}}$ through the pre-computed `TokenRefiner` Jacobian $J_{\text{ref},\text{family}} \in \mathbb{R}^{5376 \times 5120}$ scaled to `NORM_MATCHED_TARGET = 67.5`:
  $$\mathbf{v}_{\text{raw}} = \frac{1}{|W_+|}\sum_{w \in W_+} M[w] - \frac{1}{|W_-|}\sum_{w \in W_-} M[w], \quad \mathbf{v}_{\text{post}} = \alpha_{\text{cal}} \cdot J_{\text{ref},\text{family}}\,\hat{\mathbf{v}}_{\text{enc}}$$
- **When to use it**:
  Ideal for instant (`< 2 ms`, `0 GB` VRAM) lexical exploration or typing words directly into the `words` field on `MiniMaxH3ApplyControlVectors` in ComfyUI. Because it is a linear bag-of-words superposition without multi-word attention context, upgrade to Method 1 (`from-prompt`) or Method 2 (`from-toml`) when steering complex multi-clause styles, physics, or coupled sight + sound.

---

### Method 4: `from-images` — Visual Look Transfer from Two Reference Stills (`Image A -> Image B`)

```bash
"$PY" bench/control_vectors.py from-images \
  --name warm_kodak_grade \
  --image-a flat_daylight.png \
  --image-b kodak_2383_grade.png \
  --pooling scene_text --overwrite
```

- **How it works ([`control_vector_nodes.py::match_media_pair_geometry`](../../../control_vector_nodes.py))**:
  1. Center-crops and resizes both `Image A` (baseline look) and `Image B` (target look) to the exact same 32-pixel-aligned `(H, W)` grid (`--short-edge 512`) so `Qwen3-VL` emits the exact same number of `<|image_pad|>` tokens and identical 3D M-RoPE coordinates ($T_{\text{vis},A} = T_{\text{vis},B}$).
  2. Encodes both images with a shared neutral scene prompt $P_0$ through `Qwen3-VL` + `TokenRefiner`.
  3. Pools the difference across `pooling = "scene_text"` (default):
     $$\mathbf{v}_{\text{post}} = \operatorname{Pool}_{\text{scene\_text}}\!\left(H_{\text{post}}(I_B, P_0)\right) - \operatorname{Pool}_{\text{scene\_text}}\!\left(H_{\text{post}}(I_A, P_0)\right)$$
- **Why `scene_text` is the default pooling mode for `from-images`**:
  While `Image A` and `Image B` enter as `tag == 0` vision patches, causal self-attention in `Qwen3-VL` and bidirectional self-attention in `TokenRefiner` propagate the visual look of `<Picture 1>` into the downstream `tag == 1` `scene_text` tokens. Pooling `scene_text` extracts the representation shift in the exact subspace that `mask = "scene_only"` steers at runtime.

---

## 3. Token Pooling Modes (`--pooling`)

![Surgical diff_span Token Pooling (B - A)](assets/diff_span_extraction.svg)

When extracting a text-pair control vector (`from-prompt` or `from-toml`), `--pooling` controls which token positions are averaged in `Prompt A` and `Prompt B` ([`h3_spans.py:38-43`](../../../h3_spans.py#L38-L43)):

| Pooling Mode | What Tokens It Pools | Typical `raw_norm` | When to Use |
|---|---|---|---|
| **`diff_span` (Default)** | Only the exact differing token span(s) between `Prompt A` and `Prompt B` via `SequenceMatcher(..., autojunk=False)` (`min_shared_run = 8` for multi-clause or coupled `description + overall_soundscape` edits). | `49 .. 87` | **Recommended for all text-pair vectors (`from-prompt` & `from-toml`)**. Eliminates shared-tail dilution (`~2.9x` higher signal norm than `from_first_diff`) and lets you place edits anywhere in the prompt. |
| **`from_first_diff`** | From the first differing token through the end of the `tag == 1` text sequence. | `20 .. 28` | Captures causal downstream accommodation in later tokens after an early edit. |
| **`scene_text`** | All `scene`-category text tokens (`tag == 1` excluding `<Picture/Audio/Video>` labels, `<T.T seconds>` timestamps, `(S1)` speaker tags, and `<d>...</d>` dialogue). | `10 .. 25` | **Default for `from-images` (`make-media`)** where the text prompt is identical between `Image A` and `Image B`. |
| **`all_tag1`** | Every `tag == 1` text token including speaker tags and dialogue. | `10 .. 22` | Whole-sequence diagnostic comparison only. |

---

## 4. Orthogonal Holds (`[[hold]]`, `--hold`, `--hold-words`)

![Gram-Schmidt Orthogonal hold, < 5 ms CPU Inversion & 67.5 Norm Matching](assets/sculpt_and_orthogonal_hold.svg)

A concept contrast often carries unintended collateral associations—for example, `"blizzard"` naturally pulls toward `"heavy winter coat, parka"`, or `"90s sitcom"` might pull toward a specific room prop. You can strip any nuisance subspace in three ways:

1. **In `from-prompt` (`hold:` or `hold(group):` clause)**:
   ```bash
   --words "sunlit diner -> blizzard-swept diner; hold(wardrobe): wearing a light t-shirt -> wearing a heavy down parka"
   ```
2. **In `.toml` / `.md` pair files (`[[hold]]` blocks or `### Hold Pairs`)**:
   Group hold pairs by `group = "wardrobe"` or `group = "location"`. [`control_vectors.py::hold_basis`](../../../control_vectors.py) averages each group and runs two-pass modified Gram-Schmidt orthonormalization in `float64` to project $U_{\text{hold}}$ out of $\bar{\mathbf{v}}$.
3. **On CPU in `< 1 ms` with `--hold-words` (via J-Lens) or `sculpt --hold`**:
   ```bash
   "$PY" bench/control_vectors.py sculpt blizzard --hold-words "coat, parka, jacket" --name blizzard_indoor_attire
   ```
