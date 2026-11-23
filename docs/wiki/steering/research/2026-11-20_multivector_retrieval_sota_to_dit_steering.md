# Multi-Vector Retrieval SOTA (Oct–Nov 2026) -> H3 `5376`-D DiT Steering

Date: 2026-11-20 (archived from the primary investigation session into `docs/wiki/steering/research/`, companion to [`2026-11-20_unpooled_multivector_sliders.md`](2026-11-20_unpooled_multivector_sliders.md))  
Math & invariant proof script: [`bench/verify_pole_a_erasure.py`](../../../../bench/verify_pole_a_erasure.py)  
Production runtime & UI: [`control_vectors.py`](../../../../control_vectors.py), [`control_vector_nodes.py`](../../../../control_vector_nodes.py), [`director/compiler.py`](../../../../director/compiler.py)

---

## 1. Why Retrieval Research Mattered for Image/Video Steering

Single-vector steering (`mean(B) - mean(A)`) suffers from the **exact same bottleneck** that modern late-interaction Information Retrieval (IR) spent 2021–2026 trying to escape: compressing a whole multi-token sequence down to one `5376`-d average destroys within-prompt structure, lets repetitive boilerplate/punctuation drown content words, and fails whenever a prompt already states an opposing attribute.

On **2026-11-20**, rather than guessing heuristics, we surveyed state-of-the-art multi-vector, sparse-latent, static-embedding, and optimal-transport retrieval architectures as of **October–November 2026**, translated six distinct mathematical mechanisms directly into H3's post-refiner `[L, 5376]` space, and subjected every single one to zero-leakage ablations, 5-seed image checks, `192f` (`8.0s`) joint audio-video trials, and a double-dissociation specificity control.

**Headline outcome:** Borrowing from retrieval SOTA revealed *why* naive multi-vector port attempts failed (anisotropy, Sinkhorn uniform-column collapse, appended-token denominator drift) and uncovered the unexpected winning mechanism: **asymmetric ColBERT-style row-softmax transport implicitly isolates and erases the conflicting Pole-A subspace**, which we distilled into **Clamped Prompt-Centered Pole-A Subspace Erasure (`add_norm_matched_erasure`)**, with Minish-style **Clean SIF** and LightOn+Khattab **Ward + `WARP_SELECT`** ranking cleanly behind it as distinct tools.

---

## 2. Six Retrieval SOTA Lineages & Their Exact Translation into H3 `[L, 5376]`

### 2.1 Minish Lab (`reach`, `Model2Vec`, `POTION`, `pyNIFE`, `SemHash`, `Semble`, `Vicinity`, `StaRSE`)
- **Researchers & ecosystem:** Stéphan Tulkens & Thomas van Dongen (`github.com/stephantul/reach`), `Model2Vec`, `POTION` (`8M`/`32M`), `pyNIFE`, `SemHash`, `Semble`, `Vicinity`, `StaRSE`.
- **Retrieval insight:** Static & distilled token embeddings rely heavily on **All-But-The-Top PCA common-mode removal** (de-anisotropizing raw representation cones) and **Smooth Inverse Frequency (SIF) token weighting** so high-frequency functional tokens do not swamp semantic tokens when pooling.
- **What we observed directly in H3:**
  1. **Severe post-refiner anisotropy (`covered_market` = `0.6088`):** Without whitening, raw H3 token vectors sit inside a tight shared directional cone, causing cosine clustering/routing to treat almost all tokens as neighbors. Fitting **-2 Principal Components strictly on Pole-A training banks** (zero test-prompt leakage!) dropped `covered_market` anisotropy **`14.5x` (`0.6088 -> 0.0419`)**, immediately unblocking Ward clustering and `WARP_SELECT` routing.
  2. **Punctuation / whitespace SIF trap & Clean SIF fix:** Under naïve IDF/SIF, punctuation and whitespace (`'.'` = `1.888`, `' '` = `2.083`) sat furthest from the shared mean and stole the largest steering weights. **Clean SIF** computes `w_sif = clip((1 - cos(h_i, mu_ref))^alpha / mean, 0.35, 2.2)` while clamping non-alphanumeric tokens to weight `0.5`.
  3. **Controlled verdict (`A2_1d_clean_sif`):** Acts as a genuine **1D-compatible token-salience booster** raising peak content-token shifts `~1.6x–2.3x` at `s = 1.0`. It flipped `watercolor_wash` (`83.2` paint) in 1D without any erasure matrix, **but** still failed `noir` (`0/5` seeds, `S = 0.132–0.175`, staying bright color), over-drove `watercolor_wash` into collapsed lips/tabletop close-ups past total norm `~110`, and remained gated out of `192f` video integration because the experimental prototype fell back to `pieces = ['tok'] * L` in Gate 2B-2.

### 2.2 LightOn & Benjamin Clavié (`PyLate`, `LateOn`, `FastPLAID`, Ward Hierarchical Token Pooling)
- **Researchers & ecosystem:** LightOn (Antoine Chaffin et al.), Benjamin Clavié (`RAGatouille`, *LIR @ ECIR 2026* `arXiv:2511.00444`, *"Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling"*, `PyLate`, `pylate-rs` Rust/Candle engine, `FastPLAID`, `LateOn-hpool-regularized` trained with `MultiVectorTokenPoolingLoss` / Straight-Through Estimator retaining `>99%` retrieval performance).
- **Retrieval insight:** ColBERT token vectors inside a single document are redundant due to anisotropy; **Ward hierarchical agglomerative clustering** over tokens outperforms spherical k-means when compressing token bags down to `K` representatives.
- **What happened in H3 (`M2_routed` vs `M5_ward_warp_hybrid`):**
  - Spherical k-means (`M2_routed`) collapsed centroids onto prompt boilerplate (`row_div = 0.0039–0.015`).
  - Replacing spherical k-means with **Ward agglomerative clustering (`K = 8`)** over PCA-whitened tokens produced distinct semantic token clusters (`M5`, tested with Khattab `WARP_SELECT` below).

### 2.3 Omar Khattab — MIT OASYS Lab (`ColBERTv2`, `PLAID`, `WARP` SIGIR 2025)
- **Researchers & papers:** Omar Khattab, Scheerer, Zaharia, Potts, Alonso (*`ColBERTv2`*, *`PLAID`*, and *`WARP`* SIGIR 2025 dynamic similarity imputation + centroid/residual decomposition).
- **Retrieval insight:** Represent multi-vector keys via cluster **centroids + compact residuals** and use selective top-`k` centroid routing (`WARP_SELECT`) with calibrated fallback imputation so sparse queries don't miss unrouted tokens.
- **What happened in H3 (`M5_ward_warp_hybrid` -> `A6` & `A7`):**
  - Combining Minish `-2PC` whitening + LightOn/Clavié Ward `K=8` + Khattab `WARP_SELECT` top-2 gating (`A6` without SIF, `A7` with Clean SIF) ranked **#2 overall**:
    - Without SIF (`A6`): preserved clean multi-entity structure (`paper_origami` vendor+crate `78.4`, `stack: water 1.3 + wes 1.0` clean standing hall `82.4`), but remained too mild to crack photorealism on `noir` (`86.9` still color) or `watercolor_wash` (`76.9` still photo).
    - With Clean SIF (`A7`): flipped both `noir` (`88.4` crisp B&W) and `watercolor_wash` (`84.1` painted), though `stack` zoomed to `96.2`.
  - Why it lost `#1` to Pole-A erasure: requires Clean SIF token spans + PCA corpus statistics to win `noir`/`watercolor`, and delivers softer entity preservation on `paper_origami` than direct subspace erasure (`A3`/`S1`).

### 2.4 Mixedbread AI (`mxbai-edge-colbert-v0` & *Latent Terms* `arXiv:2605.29384`, June 2026)
- **Researchers & papers:** Benjamin Clavié, Sean Lee, Aamir Shakir, Makoto P. Kato — `mxbai-edge-colbert-v0-17m` (`48`-D projections, *Ettin* ModernBERT backbone, beating `ColBERTv2` and competing on `LongEmbed`) / `32m`, Mixedbread Search (~80ms billion-scale), and *Latent Terms* (`arXiv:2605.29384`, *"Latent Terms: Dense Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies"*, May/June 2026) showing dense/multi-vector spaces encode Zipfian sparse latent vocabulary atoms competitive with **LIMIT** & **BEIR** without extra supervision.
- **What happened in H3 (`M6_latent_terms`, `K = 16`, top-3 non-negative ReLU atoms):**
  - Sparse entity routing generated the **sharpest literal folded figures** on `paper_origami` and the fastest `look_direct_1990s_flash` onset.
  - **Where it hit a hard limit:** On `watercolor_wash` (`5` pairs -> `N_A = 22` tokens), splitting 22 tokens across `K = 16` sparse atoms leaves `~1.3` tokens per cluster, starving global medium overrides; and `noir` stayed full color because diffuse global color/lighting grades are spread across dense subspaces rather than concentrated on single entity nouns.

### 2.5 Google Research `MUVERA` (Fixed Dimensional Encodings) & Appended Token Bags (`ColBERT`)
- **Hypothesis tested (`M1` / `M1b`):** Concatenate `K = 12` routed `[K, 5376]` virtual ColBERT/FDE tokens onto the DiT prompt sequence with H3's 3D RoPE positions frozen at `orig_L`.
- **Why it failed for sliders:** Even with zero shift amplitude at `s = 0.0`, merely appending `K = 12` keys alters the DiT cross/self-attention softmax normalization denominator (`MAE = 22.7–54.5` vs unsteered baseline). An operator that cannot stay exact identity at `s = 0` cannot be a zero-cost neutral slider (though `M1b` remains an interesting curiosity for discrete "entity insertion" on concepts like `molten_lava_golem`).

### 2.6 Balanced Sinkhorn Optimal Transport vs. Asymmetric Row-Softmax `ColBERT` (`tau = 0.15`)
This comparison supplied the **decisive algebraic discovery** of the entire investigation:

| Transport Formulation | Column Constraints | Algebra on Mean Shift $\overline{D_p}$ | Empirical `noir` B&W Rate | Proven Max Error (`verify_pole_a_erasure.py`) |
| :--- | :--- | :--- | :---: | :--- |
| **Balanced Sinkhorn OT** | Doubly-stochastic: uniform column sums $= n_A / n_B$ | $\frac{1}{n_A}\sum_i \big((\sum_j T_{ij} B_j) - A_i\big) \equiv \bar{B} - \bar{A} = v_{1\text{d}}$ | Regressed `3/3 -> 1/3` (`cos = 1.000000` with `v_1d`) | `max_err = 8.882e-16` vs `v_1d` |
| **Row-Softmax (`tau = 0.15`, `ColBERT` soft-MaxSim)** | Right-stochastic only: each Pole-A token matches its nearest Pole-B neighbors unconstrained | Aligns per-token semantics cleanly; linear ridge $D = \mu_d + A W + \varepsilon$ recovers **rank-`r` Pole-A erasure** $-P_{V_r,\lambda}$ | **`3/3` `192f` AV + `5/5` Stills** (`min cos(D_ctx, -A) = +0.9686`) | `max_err <= 5.862e-14` across all `7` `.pt` banks |

---

## 3. From Transport Discovery to Clamped Pole-A Subspace Erasure (`add_norm_matched_erasure`)

Once row-softmax transport (`tau = 0.15`) revealed that $97\%\text{–}99.6\%$ of its advantage came from erasing conflicting Pole-A variance along a compact rank-$r$ (`r = 8`) subspace $V_r$, we bypassed runtime retrieval lookup entirely:

1. **Zero external corpus or Sinkhorn runtime loop:** Precompute `erasure.mu_d [5376]`, `erasure.vr [r, 5376]` (`r = 8`, `r = 16` max rank when multi-vector composing via `sculpt`), and `erasure.diag [r]` in `float32` (`~193.5 KB` total) at build time inside [`steering/extract.py::pool_pair_set`](../../../../control_vectors.py).
2. **Exact algebraic falsification ([`bench/verify_pole_a_erasure.py`](../../../../bench/verify_pole_a_erasure.py)):**
   - Synthetic identity test proves row-wise projection onto $V_r$ recovers exact `-A_parallel` (`max_err = 7.994e-15`, `cos(ctx_D, -A) = +0.9999`) while leaving orthogonal content `B_perp` nearly untouched (`||ctx_B|| = 1.01` vs `||ctx_-A|| = 43.74`).
   - All `7` on-disk `data/exp_unpooled_sliders/banks/*.pt` files pass with `max_err = 5.862e-14` and `min_cos(ctx_D, -A) = +0.9686`.
   - Runtime invariants (`s = 0` bit-exact identity, `s < 0` exact `add_norm_matched` fallback, exact `total_rms = |s| * 67.5` clamping, rank-`16` union preservation via `COMPOSED_ERASURE_MAX_RANK = 16`) all pass at `CUDA_VISIBLE_DEVICES=`.
3. **Double-dissociation causal proof (`S1`..`S5` Specificity Control):**
   - Keeping `total_rms = 101.25` and `erase_rms` **byte-for-byte identical** across all arms:
     - On `noir`, **`S1_own_Vr` = `5/5` B&W**, whereas **`S2_swapped_Vr` (`paper_origami` `V_r`), `S3_random_Vr`, and `S5_prompt_top8` all stay bright color**.
     - On `paper_origami`, **`S1_own_Vr` = `5/5` folded-paper porter + crate**, whereas **`S2_swapped_Vr` (`noir` `V_r`), `S3`, `S4_iso_shrink`, and `S5` destroy the porter into abstract geometry/tunnels**.
   - **Bonus discovery (`S4_iso_shrink`):** Zero-byte uniform prompt dampening (`(1 - gamma)(H - mean) + s * mu_d`) also cracks `noir` (`5/5` B&W) and pushes `watercolor_wash` painterly, **at the cost of erasing specific scene figures**. Per-concept `erasure.vr` surgically erases only the opposing Pole-A cue (`natural warm daylight color`, `live-action cloth/skin`) while sparing unrelated scene tokens.

---

## 4. What Remained Unshipped vs. What Was Shipped (`2026-11-21`)

| Concept / SOTA Mechanism | Shipped Status | Where It Lives / Why Deferred |
| :--- | :--- | :--- |
| **Clamped Pole-A Subspace Erasure + `mu_d`** | **Shipped default on `MiniMaxH3Steer` & Spatial Director (`add_norm_matched_erasure`)** | [`control_vectors.py`](../../../../control_vectors.py), [`director/compiler.py`](../../../../director/compiler.py), [`bench/verify_pole_a_erasure.py`](../../../../bench/verify_pole_a_erasure.py), [`how_pole_a_erasure_works.md`](../how_pole_a_erasure_works.md) |
| **Composed Multi-Vector Union Erasure (`sculpt` / `grade`)** | **Upgraded to `COMPOSED_ERASURE_MAX_RANK = 16`** (previously clipped at rank `8`, dropping `comp16 diag[8..15]`) | [`control_vectors.py`](../../../../control_vectors.py) (`COMPOSED_ERASURE_MAX_RANK`), [`bench/check_control_vectors.py`](../../../../bench/check_control_vectors.py) `[C7]` |
| **Dedicated Showcase Workflows (`t2va` stacked + Spatial Portal erasure)** | **Shipped** | [`h3_text_to_video_pika_stacked_erasure_api.json`](../../../../workflows/h3_text_to_video_pika_stacked_erasure_api.json), [`h3_spatial_portal_pika_erasure_api.json`](../../../../workflows/h3_spatial_portal_pika_erasure_api.json) |
| **Minish Clean SIF Token Weighting** | **Research prototype only (`verify_controls`) — not shipped in runtime** | Requires reliable tokenizer word-span alignment (`_extract_span_info` fell back to `['tok']*L` in Gate 2B-2) and risks over-driving `N_A=22` concepts (`watercolor_wash` lips collapse past norm `110`) |
| **LightOn Ward `K=8` + Khattab `WARP_SELECT`** | **Research benchmark (`#2` with Clean SIF) — not needed at runtime** | Needs Clean SIF + `-2PC` corpus whitening; `add_norm_matched_erasure` beats it on `paper_origami` and `noir` with zero external corpus |
| **Mixedbread *Latent Terms* SAE Atoms & Appended ColBERT/MUVERA Bags** | **Rejected for continuous sliders** | Fragmented global grades (`K=16` on `N_A=22`) / broke `s=0` identity via attention softmax denominator change |
