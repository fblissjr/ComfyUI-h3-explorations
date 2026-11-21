# Unpooled Multi-Vector `1x` Control Sliders in MiniMax H3 (`2026-11-20`)

Dated research study and empirical benchmark (`Phase 1` spectral extraction, `Phase 2` 5-architecture head-to-head, `Round 2A` November-2026 retrieval-SOTA operators + stress tests, and `M4` causal ablation) on replacing rank-1 pooled control vectors with **in-place, zero-RoPE-shift, low-rank conditional transport** in H3's `5376`-D post-refiner text space.

- **Status:** Research-verified on single-frame Pika FL2VA SVD995 (`280+` renders at `1152x768`, 6-step `lcm`, `1x` DiT pass). **Not yet wired into production `MiniMaxH3Steer` or validated on multi-second `141f`/`192f` AV video and `ref2va` `<Picture 1>` identity** (see Section 8 open gates).
- **Related active wiki pages:** [`../../control_vectors.md`](../../control_vectors.md), [`../../edit_guidance.md`](../../edit_guidance.md), [`../../control_vectors/building_pairs_and_vectors.md`](../../control_vectors/building_pairs_and_vectors.md).
- **Evidence on disk:**
  - Quantitative JSON & cached `.pt` banks/latents (gitignored): `data/exp_unpooled_sliders/phase1_spectra.json`, `data/exp_unpooled_sliders/phase2/metrics.json`, `data/exp_unpooled_sliders/round2a/metrics_r2a.json`, `data/exp_unpooled_sliders/round2a/metrics_m4_ablation.json`.
  - Rendered contact sheets (all media outside `data/`): `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/phase2/sheet_*.png` (`7` sheets) and `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/round2a/sheet_r2a_*.png` (`10` sheets).

---

## 1. The "So What" (Executive Summary)

1. **1D mean pooling throws away `52%–77%` of the post-refiner concept signal** that is already sitting inside our existing 10-pair contrastive `.toml` files — peaking at **`76.5%` discarded energy** on compositional styles (`noir`, `wes_anderson_symmetry`) and **`52%–68%` discarded energy** even on our best 1D winners (`watercolor_wash`, `molten_lava_golem`, `paper_origami`, `chaos_panic`).
2. **H3 concepts live in a sharp `rank 4..8` conditional subspace — not a rank-1 line.** In a direct rank sweep (`r = 2, 4, 8, 16`) at identical seed and identical `67.5` mean per-row shift norm, **`rank = 2` fails just like `1D`** (stays full color on `noir`, stays photorealistic on `watercolor_wash`), and the semantic phase transition snaps sharply between **`rank = 2` and `rank = 4..8`**.
3. **In-place conditional transport (`M3_subspace` / `M4_minish_subspace`) and whitened Ward+WARP routing (`M5_ward_warp_hybrid`) unlock broken 1D concepts at `1x` pass cost, `< 0.2 ms` inference, `~190 KB` storage, exact `0.0000` identity at `s = 0`, and zero 3D RoPE cursor shift:**
   - On `noir` (`indoor_neutral`, `s = 1.5`), `M0_1d` stays warm color (`MAE 64.7`), while `M3`, `M4`, and `M5` all render crisp high-contrast black-and-white 35mm film noir (`MAE 72.2–86.7`).
   - On `watercolor_wash` (`covered_market`, `s = 1.5`, from a 4-pair `.toml`), `M0_1d` stays photorealistic (`MAE 77.0`), while `M3`, `M4`, and `M5` turn the scene into hand-painted watercolor illustration (`MAE 97.1–113.3`).
   - On live 2-slider stacking (`noir 1.3 + chaos_panic 1.0`, `1x` pass), `M0_1d` fails monochrome (renders screaming porter in full color), while `M3`, `M4`, and `M5` render a pure black-and-white 1940s film-noir market riot.
4. **Appending unpooled concept tokens (`M1` / `M1b`, `[K=12, 5376]`) fails as a continuous slider** even with H3's 3D RoPE cursor frozen at `orig_L`: changing sequence length `L -> L + 12` perturbs the attention softmax denominator immediately at `s = 0.0` (`MAE 22.7–54.5` at zero strength) and dilutes `12` tokens against `250–400` prompt tokens.
5. **H3's post-refiner token space is strongly anisotropic (`mean pairwise cosine = 0.61–0.74`).** Raw-cosine routers (`M2_routed`) collapse to near-uniform weights (`row_div = 0.004–0.013`). Stripping the top-2 common-mode corpus PCs (Minish Lab `Model2Vec` / `reach` recipe) collapses anisotropy `14.5x` (`0.6088 -> 0.0419`), and allocating the fixed `67.5` norm budget via **geometric SIF salience** (downweighting stopwords like `"and"`, `"a"`, `"The"`, `"with"`) drives a major, causally verified gain on `noir`, `wes_anderson_symmetry`, and `look_direct_1990s_flash`.
6. **This flips how contrastive `.toml` pairs should be judged:** naturally varied multi-facet pairs (`noir.toml`, `wes_anderson_symmetry.toml`, `mpc = 0.45–0.57`) looked "weak" under 1D pooling only because mean pooling destructively canceled their orthogonal facets; under multi-vector transport they are ideal training data. Conversely, verbatim copy-paste prefix templates (`look_direct_1990s_flash.toml`, `mpc = 0.936`) starve multi-vector of contextual diversity.

---

## 2. H3 Internals Verification (Prerequisites Checked Against Code & Weights)

Before prototyping any operator, we verified the underlying H3 architecture directly on disk:

1. **Text conditioning path & `int8_convrot` invariance:**
   - All `19` `condition_proj.*` and `token_refiner.*` tensors across all 4 shipped checkpoints (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, `minimax_h3_image_pruned_int8_convrot.safetensors`, `p-h3-image_int8_convrot.safetensors`) are unquantized `BF16` (zero `convrot`, zero `comfy_quant`).
   - Across the 200 DiT `blocks.0..49` linears, `convrot` (`comfy_kitchen/tensor/int8_utils.py:11-70`) is a per-linear self-cancelling online Hadamard rotation ($x H_{256} (W H_{256}^\top)^\top = x W^\top$), **not** a residual-stream rotation. Post-refiner `5376`-D text coordinates are completely unrotated.
   - `control_vector_nodes.py::build_refiner_runner` matches ComfyUI's live `UNETLoader` + `MiniMaxH3LoRABranch` bit-for-bit (`max_abs_diff = 0.0`, `torch.equal = True`).
   - *(Minor separate backlog finding: `bench/control_vector/j_lens_builder.py:412` probes `blk.mlp.fc2` directly rather than `blk.mlp`, bypassing Pika's `0.35%–0.45%` Frobenius `token_refiner.*.mlp.fc2` LoRA; `cos(J_ref_fl2va, J_ref_fl2va_base) = 0.999434`, negligible vs. J-Lens's linearization residual and zero impact on GPU-built vectors or renders.)*
2. **Packed single-stream 3D RoPE cursor (`comfy/ldm/minimax/model.py:728-803`):**
   - H3 packs `[text | refs | audio | video]` into one sequence `h` with a `PackedLayout` whose non-text 3D RoPE temporal cursor starts at `text_len`.
   - Any **in-place** operator that transforms `context[:, scene_mask, :]` without changing `text_len` preserves exact `PackedLayout` positions across `<Picture N>`, `<Video N>`, `<Audio N>`, `MiniMaxH3ReleaseReferences`, `MiniMaxH3SoundCues`, and `MiniMaxH3Sol`, and guarantees bitwise-identical `0.0000` output at `strength = 0.0`.

---

## 3. The Core Math Formula — And Why Each Term Works

### 3.1 Why 1D Pooling Fails Mathematically

Given a contrastive pair set $\{(A_p, B_p)\}_{p=1}^P$ where $A_p \in \mathbb{R}^{n_{A,p} \times D}$ and $B_p \in \mathbb{R}^{n_{B,p} \times D}$ ($D = 5376$) are the post-refiner `diff_span` token rows of Pole A and Pole B, current 1D control vectors (`control_vectors.py::pool_pair_set`) compute a single direction $v_{\text{1D}} \in \mathbb{R}^D$ and broadcast it identically to every selected prompt row $h_i \in \mathbb{R}^D$:

$$h_i'(s) = h_i + s \cdot \frac{\tau}{\|v_{\text{1D}}\|_2} v_{\text{1D}}, \qquad \tau = 67.5$$

This makes two false assumptions:
1. **Rank-1 collapse:** Every semantic facet of Pole B (e.g., monochrome color grading, chiaroscuro lighting, film grain, tonal mood) points in the same `5376`-D direction. When facets are orthogonal, averaging them causes destructive interference: $\|\mu_d\|_2 / \mathbb{E}[\|d_i\|_2]$ drops to `0.45–0.49` on `noir` and `wes_anderson_symmetry`, discarding `76.5%` of shift energy.
2. **Uniform token weighting:** Every token in the target prompt (`"the"`, `"and"`, `"with"`, `"oak table"`, `"plaster walls"`) receives the exact same shift vector and the exact same fraction of the `67.5` norm budget.

### 3.2 Unpooled Low-Rank Conditional Transport (`M3` / `M4`)

Instead of collapsing the pair bank to a single vector, we fit a **token-conditional affine transport map** $\Delta: \mathbb{R}^D \to \mathbb{R}^D$ from Pole-A token rows to their optimal-transport-matched Pole-B deltas, factored at rank $r \in \{4, 8\}$:

#### Offline Build (from existing `.toml` pairs, `< 0.1 s` CPU after refiner pass)

1. **Per-pair Balanced Sinkhorn-Knopp Optimal Transport (`A_p -> B_p`):**
   Let $\tilde{x} = \mathcal{W}(x) = (x - \mu_{\text{ref}}) - (x - \mu_{\text{ref}}) U_c^\top U_c$ denote centered / common-mode-removed coordinates (where $U_c \in \mathbb{R}^{k_c \times D}$ holds the top $k_c \in \{0, 2\}$ corpus common-mode PCs). For each pair $p$, compute unit-normalized whitened similarities $S_p = \hat{\tilde{A}}_p \hat{\tilde{B}}_p^\top / \varepsilon$ ($\varepsilon = 0.12$) and run 12 log-domain Sinkhorn-Knopp iterations to obtain a doubly stochastic coupling $P_p \in \mathbb{R}^{n_{A,p} \times n_{B,p}}$ (marginals $1/n_{A,p}$ and $1/n_{B,p}$). Row-normalize $P_{\text{cond}, p}$ to get per-token deltas without hub collapse:

   $$D_p = P_{\text{cond}, p} B_p - A_p \in \mathbb{R}^{n_{A,p} \times D}$$

   Stack across all pairs into $A \in \mathbb{R}^{N \times D}$ and $D_{\text{ot}} \in \mathbb{R}^{N \times D}$.

2. **Geometric SIF Salience Weighting (Minish Lab `Model2Vec` / `reach` principle):**
   Without needing external word-frequency tables, measure each token's angular distance from the corpus common-mode mean $\hat{\mu}_{\text{ref}}$:

   $$w_{\text{sif}}(h_i) = \text{clip}\!\left(\frac{\left(1 - \cos(h_i, \mu_{\text{ref}})\right)^{\alpha}}{\text{mean}_j \left(1 - \cos(h_j, \mu_{\text{ref}})\right)^{\alpha}},\; 0.35,\; 2.2\right), \qquad \alpha \in [0.5, 0.75]$$

   with non-alphanumeric punctuation/whitespace tokens clamped to $0.5$ (`clean SIF`, see Section 6.3). Tokens hugging the common-mode cone (`"and"`, `"a"`, `"The"`, `"with"`) receive `~0.65–0.75x` weight; semantically specific content tokens receive `~1.2–1.8x` weight.

3. **Weighted Rank-$r$ Ridge Regression (Thin SVD, Closed Form):**
   Let $\tilde{A} = \mathcal{W}(A)$, weighted means $\mu_{\tilde{A}}, \mu_d \in \mathbb{R}^D$, and weighted centered matrices $X = \text{diag}(\sqrt{w}) (\tilde{A} - \mathbf{1}\mu_{\tilde{A}}^\top)$ and $Y = \text{diag}(\sqrt{w}) (D_{\text{ot}} - \mathbf{1}\mu_d^\top)$. Take the thin SVD $X = U \Sigma V^\top$ and keep top $r$ right singular vectors $V_r \in \mathbb{R}^{r \times D}$, singular values $\sigma_1, \dots, \sigma_r$, and ridge shrinkage $\lambda = 0.15\,\sigma_1^2$:

   $$B_{\text{map}} = \text{diag}\!\left(\frac{\sigma_k}{\sigma_k^2 + \lambda}\right) U_r^\top Y \in \mathbb{R}^{r \times D}$$

   **Total stored payload (`bf16`, $r = 8$):** `mu_A [5376]`, `mu_d [5376]`, `Vr [8, 5376]`, `B_map [8, 5376]` = **`18 \times 5376 \times 2` bytes = `193.5 KB`** (backward-compatible alongside existing `vector.post_refiner [5376]`).

#### Online Inference (`1x` DiT Pass, `< 0.2 ms` for $M = 266$ scene rows)

For the selected prompt rows $H \in \mathbb{R}^{M \times D}$ (e.g., `target = "scene_only"`):

$$\Delta_{\text{raw}}(H) = \mathbf{1}\mu_d^\top + \underbrace{\left(\mathcal{W}(H) - \mathbf{1}\mu_{\tilde{A}}^\top\right) V_r^\top}_{[M,\, r]} \underbrace{B_{\text{map}}}_{[r,\, D]}$$

$$\Delta_{\text{weighted}}(H) = \text{diag}\!\left(w_{\text{sif}}(H)\right)\,\Delta_{\text{raw}}(H)$$

$$H'(s) = H + s \cdot \frac{\tau}{\frac{1}{M}\sum_{i=1}^M \|\Delta_{\text{weighted}}(H)_i\|_2}\;\Delta_{\text{weighted}}(H), \qquad \tau = 67.5$$

#### Why Every Piece of This Formula Works
- **Base anchor $\mu_d$:** Carries the coherent global shift (`cos(v_1d, mu_d) = 0.970–0.999`), giving clean monotonic behavior and negative-polarity inversion (`s < 0`).
- **Low-rank conditional residual $(\mathcal{W}(H) - \mu_{\tilde{A}}) V_r^\top B_{\text{map}}$:** Supplies `40.5%–61.9%` of total shift magnitude on real prompts (`61.9%` on `wes_anderson_symmetry`), giving lighting tokens lighting deltas, color tokens palette deltas, and framing tokens geometry deltas in $O(M r D)$ time (`r = 4..8`, zero overfitting).
- **Ridge shrinkage $\lambda = 0.15\,\sigma_1^2$:** Keeps out-of-distribution prompt tokens bounded so unfamiliar nouns degrade smoothly to $\mu_d$ rather than exploding.
- **Clean SIF salience scaling $w_{\text{sif}}(H)$:** Reallocates the fixed `67.5` mean norm budget off common-mode stopwords (`"and"`, `"a"`, `"The"`, `"with"`) and onto content tokens without increasing total shift norm.
- **Mean-norm normalization to $\tau = 67.5$:** Keeps slider strength `s = 1.0` directly calibrated to existing `MiniMaxH3Steer` norms, and guarantees $H'(0) \equiv H$ bitwise.

### 3.3 Alternative Strong Operator: Whitened Ward + `WARP_SELECT` Hybrid (`M5_ward_warp_hybrid`)

Drawing from **LightOn / Benjamin Clavié** (Hierarchical Ward Token Pooling, `LateOn`), **Minish Lab** (PCA common-mode removal + SIF), and **Omar Khattab** (`ColBERTv2` centroid+residual + `WARP_SELECT` dynamic similarity imputation, SIGIR 2025):

1. Cluster $\mathcal{W}(A)$ into $K = 8$ semantic slots via **Ward agglomerative clustering** (`scipy.cluster.hierarchy.linkage(method="ward")`) with unit centroids $C_8 \in \mathbb{R}^{8 \times D}$ and residual shifts $R_8 = \bar{D}_{\text{slot}} - \mu_d$.
2. At inference, compute whitened cosine similarities $S = \hat{\mathcal{W}}(H) C_8^\top$, keep only the **top-2** slots above threshold $\theta = 0.05$ via `ReLU(s_k - 0.05)` scaled by confidence $\text{clip}(\max_k s_k, 0, 1)$, and impute $\mu_d$ for unmatched rows:

$$\Delta_{\text{M5}}(H) = \text{diag}\!\left(w_{\text{sif}}(H)\right)\left(\mathbf{1}\mu_d^\top + 1.5\,W_{\text{top-2}}(S)\,R_8\right)$$

This cured Phase 2 `M2_routed`'s anisotropy collapse (`watercolor_wash` jumped from photorealistic `MAE 76.7` to full watercolor `MAE 97.1`) while preserving human structure tightly on `paper_origami` and multi-slider stacking.

---

## 4. November 2026 Multi-Vector Retrieval SOTA — What Transferred vs. What Did Not

We surveyed multi-vector retrieval and lightweight embedding methods through November 2026 across **LightOn / PyLate / LateOn**, **Mixedbread (`mxbai-edge-colbert-v0`, *Latent Terms* `arXiv:2605.29384`)**, **Omar Khattab (`ColBERTv2` / `PLAID` / `WARP` SIGIR 2025)**, **Minish Lab (Stéphan Tulkens & Thomas van Dongen: `reach`, `Model2Vec`, `POTION`, `StaRSE`)**, and **Google Research (`MUVERA` / FDE)**:

| Research Source (as of Nov 2026) | Technique Borrowed | Where Tested | Verdict in H3 `5376`-D Space |
| :--- | :--- | :--- | :--- |
| **Minish Lab** (`reach`, `Model2Vec`, `POTION`) | **PCA common-mode removal** (`-top-2 PCs`) | `M4`, `M5`, `M6` | **Essential for cosine routers:** dropped `covered_market` anisotropy `14.5x` (`0.6088 -> 0.0419`), unlocking `M5` on `watercolor_wash`. *Caveat on ridge (`M4`):* global `-2PC` without SIF slightly weakened `noir` vs. concept-local centering `h - mu_A` (Section 6.3). |
| **Minish Lab** (`Model2Vec`, SIF / Zipf) | **Geometric SIF token salience** (`clean SIF`) | `M4`, `M5`, `M6` | **Big causal winner:** concentrated the `67.5` norm budget off stopwords (`' and'`, `' a'`, `' The'`, `' with'`) onto content tokens, driving `noir` B&W recovery, `wes_anderson` (`76.7 -> 90.0`), and `1990s_flash` (`78.7 -> 87.2`). Punctuation must be clamped (`0.5`). |
| **LightOn / B. Clavié** (`PyLate`, `LateOn`) | **Ward hierarchical token pooling** | `M5`, `M6` | **Winner over k-means:** preserved distinct semantic facets inside the anisotropic space cleanly. |
| **Omar Khattab** (`ColBERTv2`, `WARP` SIGIR 2025) | **Base + residual + `WARP_SELECT` top-2 imputation** | `M5` | **Strong winner (`#2` overall, best structure retention):** eliminated tail-routing noise (`M2_routed` failure) and held clean structure even on high-norm 2-slider stacks. |
| **Balanced Optimal Transport** | **12-step Sinkhorn-Knopp `A_p <-> B_p` pairing** | `M4`, `M5`, `M6` | **Clean hygiene upgrade:** prevents Pole-B hub collapse during pair-span matching. |
| **ColBERT** (appended token bag) | Appending `[K=12, 5376]` tokens with frozen RoPE | `M1`, `M1b` | **Rejected for continuous sliders:** breaks `s = 0` identity (`MAE 22.7–54.5`) via attention softmax denominator shift and dilutes against long prompts. *(Curiosity: `M1b` exhibited "entity insertion" on `molten_lava_golem`, summoning a golem behind an unchanged porter.)* |
| **Mixedbread** (*Latent Terms*, `arXiv:2605.29384`) | **Sparse top-3 non-negative IDF atoms** (`K=16`) | `M6` | **Mixed / too fragmented for global grades:** produced the highest token diversity (`row_div = 0.22–0.43`) and best literal origami people on `paper_origami`, plus crisp `1990s_flash`, but **failed monochrome on `noir` and failed painting on `watercolor_wash`** (palette/medium transforms need a dense global base shift `mu_d`). |

---

## 5. Phase 1 & 2 Discovery: Why Some Concepts Worked in 1D and Others Failed

### 5.1 Spectral & Pair-Geometry Correlation Across All 7 Tested Concepts

Measured from `data/exp_unpooled_sliders/phase1_spectra.json` and `data/exp_unpooled_sliders/banks/*.pt`:

| Concept | Bucket | `1D_E%` *(rank-1 energy)* | `B_r8` / `B_r16` | `eff_rank` | `\|mu_d\| / E\|d\|` *(coherence)* | `mpc` *(pair cos)* | `B_aniso` *(cone cos)* | `M3_ctx%` *(% non-1D)* | `M3_row_div` | `M2_row_div` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `watercolor_wash` *(4 pairs)* | `winner_1d` | **48.4%** | 74.3% / 94.8% | 15.6 | **0.604** | 0.600 | **0.744** | 40.5% | 0.0480 | *0.0049* |
| `molten_lava_golem` | `winner_1d` | **46.1%** | 49.5% / 64.8% | 44.6 | **0.633** | 0.709 | 0.708 | 48.9% | 0.0704 | *0.0039* |
| `paper_origami` | `winner_1d` | **38.6%** | 43.6% / 59.2% | 58.0 | **0.559** | 0.699 | 0.679 | 42.2% | 0.0688 | *0.0093* |
| `chaos_panic` | `winner_1d` | **31.9%** | 39.8% / 54.2% | 73.2 | **0.540** | 0.711 | 0.636 | 50.5% | **0.1690** | *0.0131* |
| `wes_anderson_symmetry` | `weak_1d` | **23.5%** | 45.8% / 61.8% | 49.9 | **0.451** | 0.453 | 0.646 | **61.9%** | **0.1214** | *0.0088* |
| `noir` | `weak_1d` | **23.4%** | 52.0% / 66.5% | 47.0 | **0.490** | 0.574 | 0.609 | 41.6% | **0.1592** | *0.0132* |
| `look_direct_1990s_flash` | `weak_1d` | *37.8%* | 58.3% / 77.5% | 29.0 | *0.700* | **0.936** | 0.629 | *35.7%* | 0.0557 | *0.0052* |

### 5.2 Three Regimes Identified (Zero-Render Triage Diagnostic)

1. **Monolithic global transforms (`1D_E% = 32%–48%`, coherence `0.54–0.63`):** `watercolor_wash`, `molten_lava_golem`, `paper_origami`, `chaos_panic`. Every Pole-B token pushes in roughly the same direction, so 1D pooling survives — yet multi-vector still produces substantially richer transforms because `52%–68%` of shift variance was being discarded.
2. **Multi-facet compositional styles (`1D_E% = 23.4%–23.5%`, coherence `0.45–0.49`):** `noir`, `wes_anderson_symmetry`. Verified on disk in `prompt_bank/control_vectors/noir.toml` and `wes_anderson_symmetry.toml`: pairs naturally vary both Pole A (`warm streetlamp`, `tungsten office`, `cafe illumination`) and orthogonal Pole B facets (`pitch-black voids`, `Venetian-blind bars`, `silver grain`, `pastel geometry`). 1D averaging destructively cancels over half the token shift norm; multi-vector preserves it.
3. **Copy-paste prefix template + test-prompt conflict (`mpc > 0.90`, `M3_ctx% = 35.7%`):** Verified on disk in `internal/internal_prompt_bank/control_vectors_v2/look_direct_1990s_flash.toml:12-90`: all 10 pairs substitute the identical 18-word copy-paste string `"blasted by harsh direct 1990s on-camera flash with blown white specular highlights and hard black drop-shadows"` in the exact same prefix slot. On `indoor_neutral` (which explicitly specifies `"quiet sunlit apartment ... near a tall window"`), all methods fought the prompt. Switching to a non-contradictory `indoor_evening` prompt immediately unlocked crisp 1990s direct flash across all methods — with `M4` reaching full flash at `s = 1.0` where `M0_1d` and `M3` were still warm amber.

---

## 6. Empirical Results Across Phase 2, Round 2A, and Causal Ablation

All renders used identical seed `730451892`, canvas `1152x768`, `single_frame` (`t = 1`), `Pika FL2VA SVD995` via `MiniMaxH3LoRABranch`, `MiniMaxH3SigmaShift(12.0, 3.0)`, `comfy kitchen attention`, `6`-step `lcm` (`PIKA_MANUAL_SIGMAS`), `PH3_IMAGE_VAE` decoded in `bf16` via `MiniMaxH3VAEPrecision`, and mean per-row shift norm `67.5` at `s = 1.0`.

### 6.1 Round 2A Head-to-Head (`s = 1.5` pixel MAE vs. baseline + visual reading)

Source: `data/exp_unpooled_sliders/round2a/metrics_r2a.json` and `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/round2a/sheet_r2a_*.png`.

| Concept (Prompt) | `M0_1d` | `M3_subspace` | `M4_minish_subspace` | `M5_ward_warp_hybrid` | `M6_latent_terms` | Visual Reading at `s = 1.5` (and onset) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `noir` (`indoor_neutral`) | `64.7` *(color)* | `72.2` *(B&W)* | **`84.7`** *(crisp B&W)* | **`86.7`** *(rich B&W)* | `90.0` *(color)* | `M0_1d` and `M6` never leave warm color; `M3`, `M4`, `M5` all achieve crisp 35mm monochrome noir. |
| `watercolor_wash` (`covered_market`) | `77.0` *(photo)* | `97.6` *(paint)* | **`113.3`** *(bleed paint)* | **`97.1`** *(paint)* | `74.7` *(photo)* | `M0_1d` and `M6` stay photorealistic; `M3`, `M4`, `M5` produce full hand-painted watercolor on paper. |
| `wes_anderson_symmetry` (`covered_market`) | `78.3` | `66.9` | **`89.4`** | `84.1` | `73.9` | `M3` hits bilateral symmetry earliest (`s = 0.5`); `M4` reaches dead-center orthographic pastel wall of symmetrical shelves + twin pastel clerks at `s = 1.5`. |
| `paper_origami` (`covered_market`) | `71.8` *(tunnel)* | `78.9` *(sculpt)* | `72.3` *(sculpt)* | **`76.7`** *(booth)* | **`77.0`** *(figures)* | `M0_1d` collapses into an abstract paper tunnel; `M5` keeps a folded-paper booth & vendor; `M6` produces two recognizable folded-paper people behind orange crates. |
| `look_direct_1990s_flash` (`indoor_evening`) | `63.7` *(flash@1.5)* | `68.1` *(flash@1.5)* | **`85.2`** *(flash@1.0!)* | `75.0` *(flash@1.5)* | `72.8` *(flash@1.0!)* | Removing `"sunlit"` unlocked stark direct xenon flash + tight black drop shadows on all 5 methods; `M4` and `M6` cross to direct flash already at `s = 1.0`. |

### 6.2 Stress Tests (`sheet_r2a_rank_sweep.png`, `sheet_r2a_stacking_2sliders.png`, `sheet_r2a_polarity_*.png`)

1. **Rank-Sweep Phase Transition (`M4_minish_subspace`, `s = 1.5`, `r = 2, 4, 8, 16`):**
   - `noir`: `r = 2` (`MAE 81.5`) **stays full warm color** → `r = 4` (`80.7`) **snaps to pure B&W film noir** → `r = 8` (`84.7`) & `r = 16` (`90.9`) deepen silver-gelatin contrast.
   - `watercolor_wash`: `r = 2` (`MAE 77.8`) **stays photorealistic** → `r = 4` (`104.4`) **snaps to pure hand-painted watercolor on white paper** → `r = 8` (`113.3`) & `r = 16` (`103.8`) hold vivid painted washes.
   - **Takeaway:** `rank = 4..8` is the empirical sweet spot (`~110–190 KB` in `bf16`); `rank = 2` is below the phase-transition threshold.
2. **Negative Polarity (`s = -1.0, -0.5, 0.0, +0.5, +1.0, +1.5`):**
   - `chaos_panic` smoothly inverts at `s = -1.0..-0.5` into calm, smiling clerks holding neat crates, then progresses monotonically through `s = +0.5` (tilting crates) → `+1.0` (shouting) → `+1.5` (airborne chaos).
   - `noir` smoothly inverts at `s = -1.0..-0.5` into bright, warm sunlit natural color with open shadows.
3. **Live 2-Slider Simultaneous Stacking (`1x` pass):**
   - `noir (1.3) + chaos_panic (1.0)` on `covered_market`: `M0_1d` and `M6` stay color; **`M3`, `M4`, and `M5` all render a pure monochrome film-noir market riot** in one pass.
   - **Failure mode found (`watercolor_wash 1.3 + wes_anderson_symmetry 1.0`):** Stacking two high-strength medium/composition sliders (`~155` combined shift norm) over-drove `M3` and `M4` into surreal fruit-collage closeups (while `M5` retained twin symmetrical clerks). Multi-slider stacking requires combined-norm budgeting/clamping (`~100–115` cap) or lower per-slider strengths (`~0.7 + 0.7`).

### 6.3 Causal Ablation of `M4` (`metrics_m4_ablation.json`, `sheet_r2a_m4_ablation.png`)

To isolate which component of `M4_minish_subspace` drove the gain over `M3_subspace` (and test whether punctuation getting `1.88–2.08` raw geometric SIF weight was noise), we ran a 4-arm ablation at `s = 1.5`:

| Concept (Prompt) | `M3_subspace` *(raw `A - mu_A`, uniform)* | `M4_ot_pca_nosif` *(Sinkhorn + `-2PC`, uniform)* | `M4_ot_pca_sif_clean` *(Sinkhorn + `-2PC` + clean SIF)* | `M4_full` *(Sinkhorn + `-2PC` + raw SIF)* |
| :--- | :---: | :---: | :---: | :---: |
| `noir` (`indoor_neutral`) | `72.25` *(B&W)* | `86.75` *(slipped back to color!)* | **`86.61`** *(crisp B&W)* | `84.65` *(crisp B&W)* |
| `watercolor_wash` (`covered_market`) | `97.58` *(clean full paint)* | **`95.34`** *(clean full paint)* | `102.84` *(over-zoomed paint)* | `113.31` *(vivid bleed paint)* |
| `wes_anderson_symmetry` (`covered_market`) | `66.91` | `76.70` | **`90.01`** *(strongest full-wall symmetry)* | `89.36` *(twin pastel clerks)* |
| `look_direct_1990s_flash` (`indoor_evening`) | `68.13` | `78.68` | **`87.18`** *(sharpest drop-shadow flash)* | `85.19` *(strong flash)* |

**What this ablation proved (two critical corrections):**
1. **Clean SIF norm-budget concentration (`w_sif` with punctuation clamped to `0.5`) is the largest single lever inside `M4`:** without SIF (`M4_ot_pca_nosif`), `noir` actually regressed to color while `wes_anderson` (`76.7 -> 90.0`) and `1990s_flash` (`78.7 -> 87.2`) jumped sharply once clean SIF was added.
2. **Global `-2PC` whitening should NOT be applied blindly to ridge inputs when concept-local centering `h - mu_A` already works:** on `noir`, projecting out the global top-2 corpus PCs removed part of the color-temperature axis (explaining why `M4_ot_pca_nosif` lost B&W while raw `M3_subspace` kept B&W). Furthermore, `power = 0.75` SIF slightly over-drove `watercolor_wash` at `s = 1.5` into a close-up. The cleanest candidate formula for production is therefore **concept-local `M3` (`h - mu_A`, Sinkhorn OT, rank `4..8`) + gentle clean SIF (`alpha = 0.5`, punctuation clamped, no external corpus PCA file needed)** — to be verified directly against `M4_ot_pca_sif_clean` in Round 2B.

---

## 7. What This Informs Across the Repo

1. **Zero-cost `.safetensors` upgrade path (100% backward-compatible):** Existing control vector files (`models/control_vectors/bank/*.safetensors`) keep `vector.post_refiner [5376]` untouched and can optionally append `subspace.mu_a [5376]`, `subspace.mu_d [5376]`, `subspace.vr [8, 5376]`, and `subspace.b_map [8, 5376]` (`~193 KB` in `bf16`). Zero re-authoring of `.toml` files is required.
2. **Contrastive `.toml` authoring guidance (`docs/wiki/control_vectors/llm_prompting_guide.md`):**
   - Keep natural contextual variation on *both* Pole A and Pole B across pairs (`noir.toml` style) rather than copy-pasting one rigid Pole-B clause (`look_direct_1990s_flash.toml`, `mpc = 0.936`).
   - Log `1D_E%` and `|mu_d| / E|d|` alongside `mpc` during `bench/control_vectors.py build` to automatically flag monolithic vs. multi-facet vs. over-templated pair sets before spending GPU time.
3. **Steering boundary (`docs/wiki/control_vectors.md`):** Multi-vector `1x` sliders expand what `MiniMaxH3Steer` can handle well — rescuing compositional looks (`noir`, `wes_anderson_symmetry`) and dramatically strengthening medium styles (`watercolor_wash`) at `1x` speed — **without** replacing `MiniMaxH3PromptExpander` (contradictory physical lighting/multi-shot blocking) or `MiniMaxH3EditGuider` (spatial `[F, H, W]` masks, `lock_take`, exact garment swaps).
4. **Multi-slider stacking guard:** When chaining multiple `MiniMaxH3Steer` nodes in a single pass, total accumulated delta norm should support a soft ceiling (e.g., `~105–120`) so stacked high-strength medium + composition sliders do not saturate.

---

## 8. Open Verification Gates (What Is NOT Yet Proven)

A verifier auditing this study should treat the following as **explicitly unproven until Round 2B completes**:

1. **Multi-second AV video coherence (`141f` / `192f` @ `1152x768`, 6-step Pika FL2VA & Ref2VA):** All `280+` renders above are `t = 1` stills (`PH3_IMAGE_VAE`). Multi-frame temporal stability, motion continuity, and **audio cleanliness** (no muffled soundtrack or speech bleed when steering `scene_only` with `M3`/`M4`/`M5`) must be verified on full `141f` AV clips.
2. **`ref2va` `<Picture 1>` identity lock & `retention_analysis:` preservation:** Must verify that per-token conditional deltas on `description_only` / `scene_only` do not degrade `<Picture 1>` facial likeness or trigger late-step red burn.
3. **Self-contained formula check (`M3 + Sinkhorn + gentle clean SIF alpha=0.5` vs. `M4_ot_pca_sif_clean`):** Needs a 1-sheet head-to-head to confirm whether dropping the global `-2PC` corpus dependency keeps `noir` B&W, `wes_anderson` symmetry, `1990s_flash`, *and* full-frame `watercolor_wash` simultaneously.
4. **H3 DiT block-band routing (`blocks 0..16` vs. `17..35` vs. `36..49`):** Untested; hook point verified at `comfy/ldm/minimax/model.py:660-669` (`blocks_replace[("double_block", i)]`), deferred to Round 2B.
5. **Cross-seed & cross-prompt generalization:** All Phase 2 / Round 2A runs used `seed = 730451892` across 3 test prompts (`covered_market`, `indoor_neutral`, `indoor_evening`). Production promotion requires multi-seed verification across at least 3 seeds and `t2va` + `ref2va` prompts.
