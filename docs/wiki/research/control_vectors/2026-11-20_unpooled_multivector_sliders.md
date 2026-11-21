# Clamped Pole-A Subspace Erasure & Unpooled Multi-Vector `1x` Sliders in MiniMax H3 (`2026-11-20`)

Dated research study, peer-verified mathematical correction, and controlled disentanglement benchmark (`Phase 1` spectral extraction, `Phase 2` 5-architecture head-to-head, `Round 2A` November-2026 retrieval-SOTA operators, and `Verifier Controls` 8-arm disentanglement with held-out PCA) on moving beyond rank-1 additive control vectors in H3's `5376`-D post-refiner text space.

- **Status:** Verified across single-frame (`5` seeds), real **`192f` (`8.0s`) joint AV video** on both `fl2va` and `ref2va` `<Picture 1>` (`3` seeds, native `.mp4` speech/lip-sync/foley audit), and **`P-H3-Image` (`4.1 MP`, 8-step `shift = 1.0`)**. Ready for opt-in production wiring as `STEER_MODES` entry `add_norm_matched_erasure` (`~193.5 KB` `float32`: `erasure.mu_d` + `erasure.vr` + `erasure.diag`).
- **Peer-review correction logged:** Our initial `M3`/`M4` ridge interpretation attributed the gain to "multi-facet Pole-B transport." Peer verification proved algebraically (`max_err <= 5.5e-14`) that regressing `D = B_matched - A` onto `A` is **`97%–99.6%` automatic rank-`r` Pole-A subspace erasure (`-P_{V_r, \lambda}`)**, caught a test-prompt PCA leakage bug in Round 2A, and requested three missing controls (`1D @ s=2.0`, `1D * clean_sif`, and pure Pole-B vs. clamped Pole-A erasure) plus a swapped-$V_r$ specificity check, 5-seed Gate 2B-1, `192f` AV Gate 2B-2, and `v_1d - e` vs. `mu_d - e` check. All passed cleanly.
- **Related active wiki pages:** [`../../control_vectors.md`](../../control_vectors.md), [`../../edit_guidance.md`](../../edit_guidance.md), [`../../control_vectors/building_pairs_and_vectors.md`](../../control_vectors/building_pairs_and_vectors.md).
- **Evidence on disk:**
  - Quantitative JSON & cached `.pt` banks/latents (gitignored): `data/exp_unpooled_sliders/phase1_spectra.json`, `data/exp_unpooled_sliders/phase2/metrics.json`, `data/exp_unpooled_sliders/round2a/metrics_r2a.json`, `data/exp_unpooled_sliders/round2a/metrics_m4_ablation.json`, `data/exp_unpooled_sliders/verifier_controls/metrics_controls.json`, `data/exp_unpooled_sliders/specificity_control/metrics_specificity.json`, `data/exp_unpooled_sliders/gate2b1/metrics_gate2b1.json`, `data/exp_unpooled_sliders/gate2b2/metrics_gate2b2.json`, `data/exp_unpooled_sliders/pre_integration/metrics_pre_int.json`.
  - Rendered media (all outside `data/`): `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/{phase2,round2a,verifier_controls,specificity_control,gate2b1,pre_integration}/sheet_*.png` and `$H3_OUTPUT_DIR/Video/exp_unpooled_round2b/{*.mp4,strips/strip_*.png}` (`29` muxed `192f` `.mp4` clips + `9` strips).

---

## 1. The "So What" (Verified Mechanisms After Disentanglement)

1. **Why plain 1D additive vectors hit a wall (`M0_1d`):**
   - 1D mean pooling discards **`52%–77%`** of post-refiner `B - \mu_A` token shift energy (`76.5%` on `noir` and `wes_anderson_symmetry`; `52%–68%` even on 1D winners).
   - More fundamentally, an additive 1D shift $h + s \cdot v_{\text{1D}}$ **never removes contradictory Pole-A directions** already present in `h` (e.g., natural color / daylight / live-action realism). On `noir` (`indoor_neutral`), `M0_1d` stays warm color on `5/5` seeds at `s = 1.5`, stays warm color at **`s = 2.0`** (`MAE 70.5`), stays warm color with **`1D * clean_sif`** (`0/5` seeds), and stays bright yellow-orange color on `3/3` seeds of `192f` `fl2va` AV video.
2. **Mechanism 1 (Verified Multi-Vector Upgrade) — Explicit Clamped Prompt-Centered Pole-A Subspace Erasure (`S1_own_Vr` / `add_norm_matched_erasure`, `~193.5 KB` `float32`, zero external PCA):**
   - Peer verification proved that `M3_subspace`'s conditional term was **`97%–99.6%` subtracting Pole A's top-$r$ subspace** (`cos(ctx_D, -A) = +0.969..+0.996`; pure Pole-B transport `W_B` alone failed on `noir`, staying full color at `MAE 58.0`).
   - Writing that mechanism **explicitly** — adding row-softmax cross-attention `erasure.mu_d` ($\tau_{\text{attn}} = 0.15$) while subtracting a **prompt-centered, `[0, 1]`-clamped** rank-8 Pole-A projector `-clip(s, 0, 1) * P_{V_r, \lambda}(h - \bar{h})`, RMS-normalized to `abs(s) * 67.5` — requires **zero corpus PCA** and **zero SIF norm inflation**, and passed every single gate:
     - **Single-frame double dissociation + `5/5` seeds (Gate 2B-1):** Flips `noir` to crisp monochrome B&W on `5/5` seeds (`S = 0.017–0.033` vs. `S0_1d` `0/5` and `A2_1d_clean_sif` `0/5`), and preserves a recognizable folded-paper market aisle + porter on `paper_origami` across `5/5` seeds (vs. `S4_iso_shrink` `0/5`, `A2_1d_clean_sif` `0/5`, `S0_1d` `2/5`).
     - **`192f` (`8.0s`) joint AV video (Gate 2B-2):** Flips `fl2va` speaking market `noir` to crisp B&W across `3/3` seeds (`sat_mean = 0.078–0.140`, `3.6x–5.5x` less color flicker than `S0_1d`), preserves all 3 shots and full speech/lip-sync/foley clarity (verified by watching/listening to the `.mp4` clips natively), holds `temp_l1_mean = 5.85` on `paper_origami` (half the jitter of `S4_iso_shrink`'s `11.30`), and holds `100%` `<Picture 1>` face/hair/wardrobe lock + crystal-clear speech across `3/3` seeds on `ref2va`.
     - **Why `erasure.mu_d` (via row-softmax `tau = 0.15`, NOT balanced Sinkhorn OT) must be kept (`Check A`):** Substituting `vector.post_refiner` (`v_1d - e`) instead of row-softmax `erasure.mu_d` (`mu_d - e`) regressed `noir` from `3/3` B&W (`S = 0.023–0.033`) down to `1/3` (`S = 0.121, 0.019, 0.199`), proving both `erasure.mu_d` (`+21.5 KB` `float32`) and per-concept `erasure.vr` are jointly required on multi-facet concepts where `cos(v_1d, mu_d) ≈ 0.97`. Balanced Sinkhorn OT enforces uniform column sums ($n_A / n_B$), which algebraically collapses $\operatorname{mean}(D_p)$ back to `mean(B_p) - mean(A_p)` (`cos ≈ 0.9999` with `v_1d`) and reproduces the `1/3` regression.
     - **`P-H3-Image` (`4.1 MP`, 8-step `shift = 1.0`, `Check C`):** Even `v1d_erase` flipped `noir` to crisp `4.1 MP` B&W (`S = 0.04` vs. `1D` `0.20`) and turned `paper_origami` into two complete full-body origami figures in an intact paper market aisle.
3. **Mechanism 2 (Orthogonal 1D-Compatible Lever) — Minish Lab Clean SIF Token Salience (`A2_1d_clean_sif` / `A7_M5_ward_clean_sif`):**
   - Reallocating the fixed `67.5` mean shift norm off common-mode stopwords (`"and"`, `"a"`, `"The"`, `"with"`) onto semantically specific content tokens (clamping punctuation to `0.5`) raises RMS shift energy slightly (`+6%` $L_2^2$) and boosts peak content-token shifts by `~1.6x–2.3x` at `s = 1.0`.
   - On `watercolor_wash`, **both** mechanisms matter cleanly: `M3_subspace` (`97.6`) and `M4_ot_pca_nosif` (`95.3`) rendered full, clean watercolor paintings with **zero SIF** (`sheet_r2a_m4_ablation.png` Row 2 Cols 2–3), while `A2_1d_clean_sif` (plain 1D × clean SIF, zero multi-vector!) also flipped photorealistic live-action into watercolor (`83.2`, though with darker vignette / surreal orange-slice bouquet), and `A3`/`A4` stayed milder (`72.0–72.8`) because they did not renormalize their combined shift norm (`98.0` vs. `M3`'s `101.25` after partial cancellation between `mu_d` and `-erasure`). Meanwhile, plain `A1_1d_s2.0` (`111.3`) reproduced the exact giant-lips / orange-slice saturation collapse all by itself when pushed to norm `135`.
4. **Appending unpooled concept tokens (`M1` / `M1b`, `[K=12, 5376]`) fails as a continuous slider** even with H3's 3D RoPE cursor frozen at `orig_L`: changing sequence length `L -> L + 12` perturbs the attention softmax denominator immediately at `s = 0.0` (`MAE 22.7–54.5` at zero strength).

---

## 2. H3 Internals Verification (Prerequisites Checked Against Code & Weights)

1. **Text conditioning path & `int8_convrot` invariance:**
   - All `19` `condition_proj.*` and `token_refiner.*` tensors across all 4 shipped checkpoints (`fl2va`, `ref2va`, `h3_image`, `p-h3-image`) are unquantized `BF16` (zero `convrot`, zero `comfy_quant`).
   - Across the 200 DiT `blocks.0..49` linears, `convrot` (implemented in the installed `comfy-kitchen` package's `int8_utils` Hadamard helper) is a per-linear self-cancelling online Hadamard rotation ($x H_{256} (W H_{256}^\top)^\top = x W^\top$), **not** a residual-stream rotation. Post-refiner `5376`-D text coordinates are completely unrotated.
   - `control_vector_nodes.py::build_refiner_runner` matches ComfyUI's live `UNETLoader` + `MiniMaxH3LoRABranch` bit-for-bit (`max_abs_diff = 0.0`, `torch.equal = True`).
   - *(Minor separate backlog finding: `bench/control_vector/j_lens_builder.py:412` probes `blk.mlp.fc2` directly rather than `blk.mlp`, bypassing Pika's `0.35%–0.45%` Frobenius `token_refiner.*.mlp.fc2` LoRA; `cos(J_ref_fl2va, J_ref_fl2va_base) = 0.999434`, zero impact on GPU-built vectors or renders.)*
2. **Packed single-stream 3D RoPE cursor (`comfy/ldm/minimax/model.py:728-803`):**
   - H3 packs `[text | refs | audio | video]` into one sequence `h` with a `PackedLayout` whose non-text 3D RoPE temporal cursor starts at `text_len`.
   - Any **in-place** operator modifying `context[:, scene_mask, :]` leaves `text_len` and all `<Picture N>` / `<Video N>` / `<Audio N>` 3D RoPE coordinates untouched and guarantees exact `0.0000` identity at `s = 0.0`.

---

## 3. The Exact Math — Why `M3` Worked, Why It Melted, and the Clean Clamped Formula

### 3.1 Algebraic Proof: Why Regressing `D = B - A` onto `A` Is Automatic Pole-A Erasure

Given stacked post-refiner `diff_span` token banks $A \in \mathbb{R}^{N \times D}$ and matched Pole-B targets $B_m \in \mathbb{R}^{N \times D}$ ($D = 5376$), let $D = B_m - A$, with centered matrices $A_c = A - \mathbf{1}\mu_A^\top$, $B_c = B_m - \mathbf{1}\mu_B^\top$, and $D_c = B_c - A_c$. With thin SVD $A_c = U \Sigma V^\top$ truncated to rank $r$ and ridge shrinkage $\lambda = 0.15\,\sigma_1^2$, the rank-$r$ ridge map $W_D = V_r^\top \operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U_r^\top D_c$ splits **algebraically exact** (`max_abs_err <= 5.5e-14`):

$$(h - \mu_A) W_D = \underbrace{(h - \mu_A) V_r^\top \operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U_r^\top B_c}_{\text{Pole-B transport } W_B} \;-\; \underbrace{(h - \mu_A) V_r^\top \operatorname{diag}\!\left(\frac{\sigma_i^2}{\sigma_i^2 + \lambda}\right) V_r}_{\text{rank-}r\text{ Pole-A erasure } (-P_{V_r,\lambda})}$$

Verified in `float64` across all 7 saved `.pt` banks on the actual test prompts:

| Concept (`N_A`) | `cos(mu_d, v_1d)` | `mean \|\|ctx_{-A}\|\|` | `mean \|\|ctx_B\|\|` | `cos(ctx_D, -A)` | `cos(ctx_D, B)` | Per-token `var_frac` | Identity `max_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `noir` (`170`) | `0.970` | **`26.6`** | `5.4` | **`+0.982`** | `-0.299` | `0.85` | `3.7e-14` |
| `watercolor_wash` (`22`) | `0.993` | **`30.5`** | `7.6` | **`+0.969`** | `-0.129` | **`0.29`** | `1.3e-14` |
| `wes_anderson_symmetry` (`101`) | `0.995` | **`35.1`** | `8.3` | **`+0.972`** | `-0.146` | **`0.32`** | `4.1e-14` |
| `molten_lava_golem` (`99`) | `0.999` | **`38.4`** | `7.7` | **`+0.979`** | `-0.012` | **`0.29`** | `1.6e-14` |
| `paper_origami` (`126`) | `0.998` | **`29.3`** | `5.2` | **`+0.984`** | `-0.093` | `0.44` | `3.6e-14` |
| `chaos_panic` (`214`) | `0.992` | **`34.1`** | `8.1` | **`+0.970`** | `-0.197` | `0.66` | `3.0e-14` |
| `look_direct_1990s_flash` (`113`) | `0.984` | **`27.9`** | `2.7` | **`+0.996`** | `-0.455` | `0.42` | `5.5e-14` |

This immediately explains **both** the wins and the failures of raw `M3` / `M4`:
- **Why it beat `M0_1d` on `noir` and `paper_origami`:** Subtracting $P_{V_r, \lambda}$ actively erases Pole-A's natural-color / live-action subspace from `h` while adding $\mu_d$.
- **Why raw `M3`/`M4` failed at `s > 1.0` and on small `N_A`:** (1) Scaling $-s \cdot P_{V_r,\lambda}$ past $s = 1.0$ **sign-flips** erased coordinates past zero; (2) centering at $\mu_A$ instead of the prompt's own mean $\bar{h}$ causes `71%` (`1 - 0.29`) of `ctx_D` on `watercolor_wash` (`N_A = 22`) to be a constant prompt-mean subtraction $(\bar{h} - \mu_A)$ that erases unseen scene backgrounds.

### 3.2 Clean Formula: Explicit Clamped Prompt-Centered Pole-A Erasure (`A3_erase_prompt_clp`)

Separate the two operations cleanly so erasure never sign-flips and never subtracts unseen prompt means:

#### Offline Build (`< 0.05 s` CPU from existing `.toml` pairs, zero external corpus file)
1. Compute per-pair **row-softmax cosine cross-attention** deltas at temperature `0.15` (verified bit-exact `max_err = 0.0` against all 7 saved `.pt` banks — **do NOT use balanced Sinkhorn OT here**, whose uniform column sums algebraically collapse $\operatorname{mean}(D_p)$ back to `v_1d`):
   $$P_p = \operatorname{softmax}\!\left(\frac{\hat{A}_p \hat{B}_p^\top}{0.15},\; \text{dim}=-1\right), \qquad D_p = P_p B_p - A_p, \qquad \mu_d = \operatorname{mean}(D)$$
   and scale $\hat{\mu}_d = \tau\,\mu_d / \|\mu_d\|_2$, $\tau = 67.5$.
2. Take the thin SVD of centered Pole-A rows $A_c = A - \mathbf{1}\mu_A^\top = U \Sigma V^\top$ and keep top $r = 8$ right singular vectors $V_r \in \mathbb{R}^{r \times D}$ and ridge erasure weights $g_i = \frac{\sigma_i^2}{\sigma_i^2 + \lambda}$ ($\lambda = 0.15\,\sigma_1^2$).

**Total stored payload (`float32`, $r = 8$):** `erasure.mu_d [5376]`, `erasure.vr [8, 5376]`, `erasure.diag [8]` = **`193.5 KB`** (stored in `float32` to match `_validate_vector_payload`'s `float32` contract and `atol = 1e-4` orthonormality check; zero external corpus file, 100% backward-compatible alongside `vector.post_refiner`).

#### Online Inference (`1x` DiT Pass, `< 0.1 ms` for $M = 266$ scene rows)

For selected prompt rows $H \in \mathbb{R}^{M \times D}$ ($M > 1$) with prompt-local row mean $\bar{h} = \frac{1}{M}\sum_{i=1}^M H_i$ and stream-effective strength $s_j$:

$$E(H) = \left(H - \mathbf{1}\bar{h}^\top\right) V_r^\top \operatorname{diag}(g)\, V_r, \qquad \beta_{\text{cap}} = \min\!\left(1.0,\; \frac{\tau}{\operatorname{RMS}(E(H))}\right)$$

$$\Delta_{\text{raw}}(s_j) = s_j \cdot \mathbf{1}\hat{\mu}_d^\top \;-\; \operatorname{clip}(s_j, 0, 1)\cdot \beta_{\text{cap}}\, E(H), \qquad H'(s_j) = H + \frac{|s_j|\,\tau}{\operatorname{RMS}(\Delta_{\text{raw}}(s_j))}\,\Delta_{\text{raw}}(s_j)$$

*(When $s_j \le 0$ or $M \le 1$, $\operatorname{clip}(s_j, 0, 1) = 0$ shuts erasure off cleanly so negative polarity subtracts $\hat{\mu}_d$ without amplifying Pole A past its natural manifold; RMS renormalization keeps total shift energy strictly matched to $|s_j|\tau$.)*

### 3.3 Orthogonal Knob: Minish Lab Clean SIF Token Salience (`w_sif`)

Independently of multi-vector erasure, a held-out corpus mean $\mu_{\text{ref}}$ lets us measure each token's angular distance from the common-mode cone (`"and"`, `"a"`, `"The"`, `"with"` sit at `cos ≈ 0.80–0.88`, while content nouns sit further out):

$$w_{\text{sif}}(h_i) = \text{clip}\!\left(\frac{(1 - \cos(h_i, \mu_{\text{ref}}))^{\alpha}}{\text{mean}_j (1 - \cos(h_j, \mu_{\text{ref}}))^{\alpha}},\; 0.35,\; 2.2\right)$$

with non-alphanumeric punctuation/whitespace tokens clamped to `0.5` (because raw geometric distance assigned `'.'` and `' '` weights `1.88–2.08`). Applying $w_{\text{sif}}$ to the additive term concentrates the fixed `67.5` mean norm onto content tokens — **but it must pass the `192f` AV video gate** because `token_focus = 0.30` lost `3/3` on video when skewing per-token shift norms diluted cross-modal attention mass.

---

## 4. November 2026 Multi-Vector Retrieval SOTA — What Actually Held Up Under Controls

We tested ideas from **LightOn / Benjamin Clavié** (`PyLate`, `LateOn`, Ward hierarchical token pooling), **Mixedbread** (`mxbai-edge-colbert-v0`, *Latent Terms* `arXiv:2605.29384`), **Omar Khattab** (`ColBERTv2` / `PLAID` / `WARP` SIGIR 2025), **Minish Lab** (Stéphan Tulkens & Thomas van Dongen: `reach`, `Model2Vec`, `POTION`), **Google Research** (`MUVERA` / appended bags), and **Balanced Sinkhorn OT** *(full Oct–Nov 2026 retrieval-to-steering lineage write-up in [`2026-11-20_multivector_retrieval_sota_to_dit_steering.md`](2026-11-20_multivector_retrieval_sota_to_dit_steering.md))*:

| Research Source (as of Nov 2026) | Mechanism Tested | Controlled Verdict in H3 `5376`-D Space |
| :--- | :--- | :--- |
| **Subspace Projection / Erasure** *(emergent in `M3`, isolated in `A3` / `S1`)* | **Clamped prompt-centered rank-8 Pole-A erasure + row-softmax `mu_d`** | **#1 verified multi-vector winner:** only non-SIF method passing `noir` (`5/5` stills, `3/3` `192f` AV, `P-H3-Image`) and `paper_origami` (`5/5` aisle+porter), with zero corpus PCA and `193.5 KB` `float32` storage. |
| **Minish Lab** (`reach`, `Model2Vec`, `POTION`) | **PCA common-mode removal (`-2PC`) + Clean SIF salience** | **Verified anisotropic geometry + orthogonal lever:** stripping 2 corpus PCs dropped `covered_market` anisotropy `14.5x` (`0.6088 -> 0.0419`), unblocking Ward/WARP cosines. Clean SIF is a strong **1D-compatible** booster (`A2_1d_clean_sif` flipped `watercolor_wash` alone), though it can over-drive saturated concepts (`watercolor` at `s=1.5`) and needs video gating. |
| **LightOn / B. Clavié + Omar Khattab** (`LateOn` Ward + `WARP_SELECT`) | **Whitened Ward (`K=8`) + base+residual + top-2 `WARP_SELECT` (`M5` / `A6` / `A7`)** | **Solid #2 when paired with clean SIF (`A7`), mild alone (`A6`):** without SIF (`A6`), `M5` held clean structure on `paper_origami` and `stack_water_wes` but stayed color on `noir` and photorealistic on `watercolor_wash`. Adding clean SIF (`A7`) flipped both cleanly (`noir 88.4` B&W, `watercolor 84.1` paint). |
| **Mixedbread** (*Latent Terms*, `arXiv:2605.29384`) | **Sparse top-3 non-negative IDF atoms (`K=16`, `M6`)** | **Too fragmented for global grades:** failed `noir` (stayed color) and `watercolor_wash` (stayed photo; `N_A=22` split into 16 clusters is ~1.3 tokens/cluster). Best literal origami figures on `paper_origami` and fast `1990s_flash` onset. |
| **ColBERT** (appended token bag) | **Appending `[K=12, 5376]` tokens with frozen RoPE (`M1`/`M1b`)** | **Rejected for continuous sliders:** breaks `s = 0` identity (`MAE 22.7–54.5`) via attention softmax denominator change. *(Curiosity: `M1b` showed "entity insertion" on `molten_lava_golem`.)* |

---

## 5. Pair-Geometry Correlation: Three Regimes Across the 7 Tested Concepts

Measured from `data/exp_unpooled_sliders/phase1_spectra.json` and `data/exp_unpooled_sliders/banks/*.pt`:

| Concept | Bucket | `1D_E%` *(rank-1)* | `B_r8` / `B_r16` | `eff_rank` | `\|mu_d\| / E\|d\|` | `mpc` | `B_aniso` | Dominant Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `watercolor_wash` *(5 pairs)* | `winner_1d` | **48.4%** | 74.3% / 94.8% | 15.6 | **0.604** | 0.600 | **0.744** | Monolithic medium (`data/control_vectors/watercolor_wash.toml`, 5 pairs, `A_all = 22` tokens); clean full paint via `M3`/`M4_nosif` (zero SIF) or `A2_1d_clean_sif`; saturates into a lips/orange-slice collapse past norm `~110` (reproduced even by plain `A1_1d_s2.0`). |
| `molten_lava_golem` | `winner_1d` | **46.1%** | 49.5% / 64.8% | 44.6 | **0.633** | 0.709 | 0.708 | Monolithic material override; works in 1D, richer with erasure. |
| `paper_origami` | `winner_1d` | **38.6%** | 43.6% / 59.2% | 58.0 | **0.559** | 0.699 | 0.679 | 1D collapses into abstract paper tunnel; Pole-A erasure (`A3`/`A4`) and `M5` preserve human porter + crates. |
| `chaos_panic` | `winner_1d` | **31.9%** | 39.8% / 54.2% | 73.2 | **0.540** | 0.711 | 0.636 | Multi-person action state; clean negative polarity (`-1.0` calm → `+1.5` chaos). |
| `wes_anderson_symmetry` | `weak_1d` | **23.5%** | 45.8% / 61.8% | 49.9 | **0.451** | 0.453 | 0.646 | Compositional style; `M0_1d` already achieves symmetry by `s = 0.5–1.5`, erasure/SIF sharpen pastel orthographic walls. |
| `noir` | `weak_1d` | **23.4%** | 52.0% / 66.5% | 47.0 | **0.490** | 0.574 | 0.609 | **True multi-facet 1D wall:** `M0_1d` stays color even at `s = 2.0` or `* clean_sif`; requires Pole-A erasure (`A3`) or `M5 + clean_sif` (`A7`). |
| `look_direct_1990s_flash` | `weak_1d` | *37.8%* | 58.3% / 77.5% | 29.0 | *0.700* | **0.936** | 0.629 | **Test-prompt conflict + copy-paste template:** verbatim 18-word Pole-B string across all 10 pairs (`internal/internal_prompt_bank/control_vectors_v2/look_direct_1990s_flash.toml:12-90`). On `indoor_evening` (no `"sunlit"`), even plain `M0_1d` hits stark direct flash at `s = 1.5`. |

---

## 6. Verifier Disentanglement Results (8 Controlled Arms, Zero Test-Prompt Leakage)

All renders used identical seed `730451892`, canvas `1152x768`, `t = 1`, `Pika FL2VA SVD995` via `MiniMaxH3LoRABranch`, `MiniMaxH3SigmaShift(12.0, 3.0)`, `comfy kitchen attention`, `6`-step `lcm`, `PH3_IMAGE_VAE` (`bf16`), and **held-out PCA fit strictly on Pole-A training banks (zero test prompts)**.

Source: `data/exp_unpooled_sliders/verifier_controls/metrics_controls.json` and `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/verifier_controls/sheet_vc_part{1,2}_*.png`.

| Concept / Stack | `A0_1d` *(s=1.5)* | `A1_1d` *(s=2.0)* | `A2_1d * CleanSIF` | `A3_erase_prompt_clp` *(no SIF)* | `A4_erase_muA_clp` *(no SIF)* | `A5_pure_poleB` *(no `-A`)* | `A6_M5_ward` *(no SIF)* | `A7_M5_ward` *(+ CleanSIF)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`noir`** (`indoor_neutral`) | `64.7` *(color)* | `70.5` *(color)* | `79.0` *(color)* | **`87.0` *(crisp B&W!)*** | **`83.1` *(crisp B&W!)*** | `58.0` *(color)* | `86.9` *(color)* | **`88.4` *(crisp B&W!)*** |
| **`watercolor_wash`** (`market`) | `77.0` *(photo)* | `111.3` *(lips collapse!)* | **`83.2` *(paint!)*** | `72.8` *(mild soft; norm `98`)* | `72.0` *(mild soft; norm `98`)* | `73.1` *(photo)* | `76.9` *(photo)* | **`84.1` *(paint!)*** |
| **`wes_anderson`** (`market`) | `78.3` *(symm)* | `80.2` *(symm)* | `75.2` *(symm)* | `72.7` *(symm)* | `71.7` *(symm)* | `81.3` *(symm)* | `80.3` *(symm)* | `80.8` *(symm)* |
| **`paper_origami`** (`market`) | `71.8` *(tunnel)* | `81.7` *(tunnel)* | `78.2` *(tent)* | **`74.7` *(porter+crate!)*** | **`82.7` *(porter+crate!)*** | `74.2` *(tunnel)* | **`78.4` *(vendor+crate!)*** | `78.8` *(tent)* |
| **`stack: water 1.3 + wes 1.0`** | `81.7` *(clerks)* | `91.5` *(lips collapse!)* | `83.1` *(stylized)* | `83.5` *(doll busts / red lips)* | `94.4` *(twin busts)* | **`84.6` *(standing hall)*** | **`82.4` *(standing hall)*** | `96.2` *(close-up)* |

*(Note on `A3`/`A4` norm & clamp hygiene: in the 8-arm pass above, `A3` and `A4` did not renormalize their combined `mu_d - erasure` vector back to `s * 67.5`, so partial cancellation left `watercolor_wash` slightly under-driven at `98.0` vs. `M3`'s `101.25`; and the scratch script initially coded `min(s, 1.0)` rather than `max(0.0, min(s, 1.0))` — both fixed in Section 6.2 below.)*

### 6.2 Specificity Control: Double Dissociation Proving `V_r` Is Concept-Specific (Not Generic Variance Dampening)

To rule out the alternative hypothesis that subtracting *any* rank-8 projector from $H - \bar{h}$ merely dampens prompt variance to boost the SNR of $\hat{\mu}_d$, we ran a 6-arm specificity control (`data/exp_unpooled_sliders/specificity_control/metrics_specificity.json`, `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/specificity_control/sheet_specificity_swapped_vr_and_shrink.png`) with **every arm strictly RMS-matched to `total_rms = 101.25` at `s = 1.5`** and **exact matched erasure RMS** (`26.21` on `noir`, `20.08` on `paper_origami`, `16.82` on `watercolor_wash`), using `max(0.0, min(s, 1.0))`:

| Concept (Prompt) | `S0_1d` *(s=1.5)* | `S1_own_Vr` *(concept's own `V_r`)* | `S2_swapped_Vr` *(wrong concept's `V_r`)* | `S3_random_Vr` *(random orth `r=8`)* | `S4_iso_shrink` *(0-byte $-\gamma(H-\bar{h})$)* | `S5_prompt_top8` *(prompt's own top-8 PCs)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`noir`** (`indoor_neutral`, `swap = paper_origami`) | `64.7` *(color)* | **`73.2` *(crisp B&W!)*** | `78.0` *(color!)* | `56.1` *(color!)* | **`81.0` *(crisp B&W!)*** | `76.0` *(color!)* |
| **`paper_origami`** (`covered_market`, `swap = noir`) | `71.8` *(tunnel)* | **`78.1` *(crouching porter+crate!)*** | `85.3` *(mangled tent)* | `76.3` *(empty tunnel)* | `82.1` *(abstract diamond, 0 people!)* | `83.1` *(empty tent, 0 people!)* |
| **`watercolor_wash`** (`covered_market`, `swap = noir`) | `77.0` *(photo)* | `74.8` *(soft photo)* | `76.0` *(soft photo)* | `73.4` *(photo)* | **`79.9` *(painterly aprons+wash!)*** | `78.7` *(soft photo)* |

**What this double dissociation proves cleanly:**
1. **Per-concept `V_r` (`193.5 KB` `float32` with `erasure.mu_d` + `diag`) IS causally concept-specific:** At the exact same `total_rms = 101.25` and exact same `erase_rms`, **`noir`'s `V_r` flips `noir` to B&W while `paper_origami`'s `V_r` (`S2`), random `r=8` (`S3`), and prompt top-8 PCs (`S5`) all stay color**; symmetrically, **`paper_origami`'s `V_r` is the ONLY arm out of all 6 that preserves a recognizable folded-paper porter lifting a crate**, while `noir`'s `V_r` (`S2`), random `r=8` (`S3`), isotropic shrinkage (`S4`), and prompt top-8 (`S5`) all destroy the human figure.
2. **Bonus `0`-byte discovery (`S4_iso_shrink` — Blunt Style SNR Dial vs. Surgical Erasure):** Uniformly shrinking prompt variance $\bar{h} + (1 - \gamma)(H - \bar{h}) + s\hat{\mu}_d$ (`0` extra bytes stored!) also flips `noir` to crisp B&W (`81.0`) and pushes `watercolor_wash` painterly (`79.9`) at exact `101.25` RMS — **but because it dampens all prompt directions indiscriminately, it erases specific scene entities** (turning `paper_origami`'s porter into an empty abstract diamond of triangles). Per-concept `S1_own_Vr` erases *only* the conflicting Pole-A subspace while leaving orthogonal subject tokens intact.

### 6.3 Gate 2B-1 (5-Seed Multi-Seed Check), Gate 2B-2 (`192f` Joint AV Video), & Pre-Integration Checks

1. **Gate 2B-1 (`5` seeds `[730451892, 101, 2026, 424242, 99991]`, strictly RMS = `101.25` @ `s = 1.5`, `metrics_gate2b1.json`, `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/gate2b1/sheet_2b1_*.png`):**
   - **`noir` (`indoor_neutral`, true B&W `< 0.06` HSV saturation):** **`S1_own_Vr` = `5/5` B&W** (`0.028, 0.023, 0.033, 0.017, 0.027`) and `S4_iso_shrink` = `5/5` (`0.023–0.032`), vs. **`S0_1d` = `0/5`** (`0.166–0.197`) and **`A2_1d_clean_sif` = `0/5`** (`0.132–0.175`).
   - **`paper_origami` (`covered_market`):** **`S1_own_Vr` = `5/5`** preserving a folded-paper market corridor + recognizable origami porter carrying a crate, vs. **`S4_iso_shrink` = `0/5`** (flat wooden shadow-boxes), **`A2_1d_clean_sif` = `0/5`** (hollow paper tents), and **`S0_1d` = `2/5`**.
   - **`chaos_panic` (`5/5` clean)** & **`watercolor_wash` (`5/5` zero breakage; confirms `N_A = 22` boundary where `S1_own_Vr` stays mild while `S4`/`A2` go painterly by collapsing market depth into tabletop close-ups on `5/5` seeds)**.

2. **Gate 2B-2 (`29` real `192f` / `8.0s` `1152x768` AV clips @ `24 fps` + `32 kHz` stereo via custom pack nodes + `MiniMaxH3Sol`, `metrics_gate2b2.json`, `$H3_OUTPUT_DIR/Video/exp_unpooled_round2b/`):**
   - **`fl2va` speaking market (`prompt_bank/t2va_covered_market.txt`, 3 shots, 2 speakers, `target = "scene_only"`):**
     - `noir` (`3` seeds `[730451892, 101, 2026]`): `S0_1d` stayed bright color on `3/3` seeds (`sat_mean = 0.251–0.349`, `sat_std = 0.033–0.040`). **`S1_own_Vr` flipped `3/3` seeds to crisp B&W noir (`sat_mean = 0.078–0.140`) with `3.6x–5.5x` lower color flicker (`sat_std = 0.007–0.009`)**, kept all 3 shots cleanly (whereas `S4_iso_shrink` lost Shot 2's close-up cut on `s101`/`s2026`), and native `.mp4` playback confirmed **100% intelligible speech (`"Last of the good ones..."` / `"Then I will take two."`), tight lip-sync, hollow crate thuds, coin clinks, and paper rustles** (zero `ab_token_focus` muffling).
     - `paper_origami` (`s730451892`): **`S1_own_Vr`** delivered a temporally smooth animated folded-paper stallholder/porter (`temp_l1_mean = 5.85` vs. `S0_1d` `6.57` and `S4_iso_shrink` `11.30`) with clear synced dialogue; `S4_iso_shrink` broke into a live-action man in a suit holding paper pyramids.
   - **`ref2va` `<Picture 1>` speaking bookstore (`t2i_29_edit_reference_portrait_seed7.png`, `fl2va`-built `V_r` transferred cross-family, `target = "scene_only"` protecting `retention_analysis:` and `843` vision rows; note `detailed_description` is `126w`, judged relative to `Base`/`S0_1d`):**
     - `S1_own_Vr` preserved **100% `<Picture 1>` facial likeness, dark bob with grey front streak, pink floral blazer, and smooth motion (`temp_l1_mean = 1.42–1.62` vs. `Base` `1.80`) across `3/3` seeds**, and native `.mp4` playback confirmed crisp oak book thump, clear `"We saved the last first-edition copy behind the desk for you."`, clean lip closure, and wood-slide foley. With `<Picture 1>`'s bright pink blazer pinned at `end_percent = 1.0, clean_level = 0.999`, `noir` at `s = 1.5` cooled/desaturated the scene on `3/3` seeds (`sat_mean = 0.238 / 0.327 / 0.296` vs. `S0_1d` `0.287 / 0.386 / 0.362`) without overriding `<Picture 1>`'s locked color reference into full monochrome.
   - *(Audit note: the `6` `A2_1d_clean_sif` + `M5_ward_warp_hybrid` `192f` clips ran with `pieces = ['tok'] * L` fallback in `_extract_span_info`, i.e., uncleaned SIF; `S0_1d`, `S1_own_Vr`, `S4_iso_shrink`, and `Base` never read `pieces` and are 100% clean.)*

3. **Pre-Integration Checks A, B, C (`metrics_pre_int.json`, `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/pre_integration/sheet_check*.png`):**
   - **Check A (`v_1d - e` vs. `mu_d - e`, 3 seeds):** On `noir`, `mu_d - e` passed **`3/3`** B&W (`S = 0.028, 0.023, 0.033`), whereas substituting `vector.post_refiner` (`v_1d - e`) dropped to **`1/3`** (`0.121, 0.019, 0.199`). On `paper_origami` (`cos(v_1d, mu_d) = 0.998`), both passed `3/3`. **Conclusion: keep `erasure.mu_d [5376]` in the `.safetensors` payload (`+21.5 KB` `float32`).**
   - **Check B (`s = [-1.0, -0.5, 0.0, +0.5, +1.0, +1.5]` ramp with `clip(s, 0, 1)`):** Clean bit-exact negative polarity (`s < 0` matches `1D` identically), exact `s = 0` identity, and smooth monotonic progression through default `s = 1.0` to `s = 1.5` with zero artifacts.
   - **Check C (`P-H3-Image` `2304x1792`, `4.1 MP`, `PH3_IMAGE_TE`, `PH3_IMAGE_UNET`, 8-step `euler`, `shift = 1.0`):** Even `v1d_erase` flipped `noir` to crisp `4.1 MP` B&W (`S = 0.04` vs. `1D` `0.20`) and turned `paper_origami` into two complete sculptural full-body origami figures in an intact paper corridor (vs. `1D`'s half-human in blue jeans).

### 6.4 Additional Verified Findings from Round 2A (Still Valid After Controls)
- **Rank-sweep phase transition (`sheet_r2a_rank_sweep.png`):** On `M4_minish_subspace` (`s = 1.5`), `rank = 2` fails on both `noir` (stays warm color, `81.5`) and `watercolor_wash` (stays photorealistic, `77.8`), while `rank = 4..8` crosses sharply into crisp B&W noir (`80.7–84.7`) and painted watercolor (`104.4–113.3`). This reflects that the erased Pole-A subspace requires **`r = 4..8` dimensions** (not 1–2).
- **Round 2A `M4` ablation visual caution (`sheet_r2a_m4_ablation.png`):** On `watercolor_wash`, `M3_subspace` (`97.6`) and `M4_ot_pca_nosif` (`95.3`) kept full figures, crates, hanging scale, and market background intact with zero SIF, whereas `M4_ot_pca_sif_clean` (`102.8`) suffered severe compositional collapse (cropped giant mouths/chins with eyes cut off at top edge) and `M4_full` (`113.3`) erased the background to a white void.
- **Live 2-slider stacking `noir (1.3) + chaos_panic (1.0)` (`sheet_r2a_stacking_2sliders.png`):** `M0_1d` and `M6` stay full color; `M3`, `M4`, and `M5` all render a pure monochrome 1940s film-noir market riot in one `1x` pass.
- **Negative polarity (`sheet_r2a_polarity_{noir,chaos_panic}.png`):** Clean inversion at `s = -1.0..-0.5` (`chaos_panic` → calm serene order; `noir` → warm open daylight).
- **Test-prompt conflict (`sheet_r2a_look_direct_1990s_flash__indoor_evening.png`):** Removing `"sunlit ... near a tall window"` unlocked stark 1990s direct xenon flash + tight black drop shadows across all methods, including plain `M0_1d` at `s = 1.5`.

---

## 7. Locked Production Integration Spec (Ready to Wire)

1. **Backward-compatible `.safetensors` payload (`~193.5 KB` `float32`):**
   - Keep `vector.post_refiner [5376]` untouched; optionally append `erasure.mu_d [5376]`, `erasure.vr [r, 5376]` (`r = 8`), and `erasure.diag [r]` in **`float32`** (satisfying `_validate_vector_payload`'s `float32` contract and `atol = 1e-4` orthonormality check).
   - Computed in `control_vectors.py::pool_pair_set` from the same `diff_span` bank with zero external corpus file; when `[[hold]]` groups exist, orthogonalize both `erasure.mu_d` and `erasure.vr` against `holds.post_refiner`. Leave `H3VectorSpec` untouched.
2. **`MiniMaxH3Steer` rollout (`add_norm_matched_erasure` — promoted to `MiniMaxH3Steer` default on `2026-11-21`, commit `6d96b09a`):**
   - Added `add_norm_matched_erasure` to `STEER_MODES` and promoted `h3_rules.STEER_DEFAULTS["mode"] = "add_norm_matched_erasure"` (`MiniMaxH3Steer` default; legacy `MiniMaxH3ApplyControlVectors` and `CONTROL_VECTOR_DEFAULTS` retain `"add_norm_matched"`). Composite payloads preserve rank-capped `erasure.{mu_d,vr,diag}` through `control_vectors.py::compose_erasure_payload`, `j_lens.py::save_composed_recipe_vector`, and `bench/control_vector/sculpt.py`, and `apply_post_refiner` emits `STACK_SHIFT_WARN_NORM` on high-shift mixed stacks.
   - Runtime guards: clamp `max(0.0, min(sj, 1.0))` on **stream-effective `sj`**, scrub `hold_vector`/`hold_words` **per-row `[M, 5376]`**, skip erasure on `h3_image` `vector.encoder` auto-projection (automatically falling back to 1D `add_norm_matched`), guard `M <= 1` (`e_own_rms == 0`), keep `erasure.vr`/`erasure.diag` on CPU (no `id(ctx)` GPU cache), and RMS-normalize the combined shift to `abs(sj) * 67.5`.
3. **Contrastive `.toml` hygiene:**
   - Note on `data/control_vectors/watercolor_wash.toml` (5 pairs): `"lips"` / `"mouth"` appears identically in the shared `a` and `b` dialogue-closing suffix (not in `diff_span`), while `diff_b` repeats the same 11-word prefix across all 5 pairs (`len(diff_b) > len(diff_a)`). Its norm-`135` lips/fruit-slice attractor (`A1_1d_s2.0`) likely reflects bidirectional `token_refiner` length-offset coupling or H3's prior at extreme shift norm rather than an extraneous Pole-B diff clause; either way, high-norm stacking requires a combined-norm cap.
   - Keep natural contextual variation on both Pole A and Pole B (`prompt_bank/control_vectors/noir.toml` style) rather than verbatim copy-paste prefix clauses (`internal/internal_prompt_bank/control_vectors_v2/look_direct_1990s_flash.toml`, `mpc = 0.936`).
   - Log `1D_E%` and `|mu_d| / E|d|` alongside `mpc` in `bench/control_vectors.py build`.

---

## 8. Remaining Minor Open Items

1. **In-repo unit coverage & rollout (`6d96b09a` landed):** Production implementation and tests (`bench/check_control_vectors.py`, `check_widget_deviations.py`, regenerated workflows) landed on `2026-11-21`.
2. **Native `MiniMax-H3-Image` (`--families h3_image`) & base 50-step H3 check:** Explicit `--families h3_image` landed in `bench/control_vector/{helpers,build_cmd}.py` (`6d96b09a`); base 50-step full H3 schedule remains optional.
3. **Optional `M5_ward_warp_hybrid` / `A2_1d_clean_sif` 6-clip re-decode:** Only needed if anyone later explores SIF-weighted routing on video (`S1_own_Vr` / `add_norm_matched_erasure` does not use SIF or `pieces`).
4. **H3 DiT block-band routing (`comfy/ldm/minimax/model.py:660-669`):** Deferred as a separate follow-up study.

---

## 9. Follow-Up Rollout Addendum (2026-11-21)

- **Default promotion & composite erasure (`6d96b09a` -> `COMPOSED_ERASURE_MAX_RANK = 16`):** `MiniMaxH3Steer.mode` and [`director/compiler.py`](../../../../director/compiler.py) both default to `"add_norm_matched_erasure"`, and `compose_erasure_payload()` in [`control_vectors.py`](../../../../control_vectors.py) now retains up to `COMPOSED_ERASURE_MAX_RANK = 16` union singular directions (`comp16`) across [`j_lens.py`](../../../../j_lens.py) and [`bench/control_vector/sculpt.py`](../../../../bench/control_vector/sculpt.py) (previously clipped at `8`, dropping `diag[8..15] = [0.694, 0.626, 0.569, 0.510, 0.386, 0.328, 0.294, 0.264]`).
- **Reproducible CPU math falsifier & varied non-market showcases (`bench/verify_pole_a_erasure.py` & `$H3_OUTPUT_DIR/Video/erasure_showcase_2026-11-21/`):**
  - **`bench/verify_pole_a_erasure.py`** (`CUDA_VISIBLE_DEVICES=`): Proves `float64` algebraic ridge split (`max_err = 7.994e-15` synthetic, `5.862e-14` on `7` `.pt` banks), Balanced Sinkhorn OT uniform-column collapse back to `v_1d` (`max_err = 8.882e-16`, `cos = 1.000000`), `s = 0`/`s < 0`/RMS `101.25`/per-row hold/rank-16 union invariants.
  - **Showcase 1 (`t2va_cooking_show_meltdown`, [`workflows/h3_text_to_video_pika_stacked_erasure_api.json`](../../../../workflows/h3_text_to_video_pika_stacked_erasure_api.json), `192f`, `noir +1.55` + `chaos_panic +0.90` on `scene_only` + `reverb_cathedral +1.25` on `soundscape_only`):** `Pole-A Erasure` cut saturation **`3.0x` (`sat_mean = 0.0715` vs `1D` `0.2141`)**, turning the color TV studio into classic B&W comedy with booming cathedral speech (`showcase1_cooking_stacked_192f.mp4`).
  - **Showcase 2 (`t2va_disco_roller_rink`, [`workflows/h3_spatial_portal_pika_erasure_api.json`](../../../../workflows/h3_spatial_portal_pika_erasure_api.json), `141f`, `vertical_split` `x = 0.68 -> 0.50 -> 0.30`, `same_prompt = True`, `scale = 2.2`, `noir +1.6`, `edit_pass = "edited_only"`):** `Pole-A Erasure` achieved **`5.91x` left/right saturation contrast** (`sat_left_quarter_mean = 0.6012` vs `sat_right_quarter_mean = 0.1018`; `f70` left `0.6173` gold disco vs right `0.1275` silver-gelatin B&W) where `1D` was vetoed (`0.6003` vs `0.4286`, `f70` right `0.4521`), preserving sung disco vocals (`showcase2_disco_wipe_stacked_141f.mp4`). Note: a small `rx = 0.24` (`10.3%`-area) circular portal under `same_prompt = True` at `scale = 1.5` stayed warm (`sat_mean = 0.4278 -> 0.4134`) under global 3D self-attention; `vertical_split` (`32%..70%`) crosses the spatial continuity threshold cleanly.
  - **Showcase 3 (`fl2va` Beef-Noodle Counter keyframes `fl2va_noodle_f000.png -> f123.png`, [`workflows/h3_first_last_frame_pika_control_vector_api.json`](../../../../workflows/h3_first_last_frame_pika_control_vector_api.json), `141f`, `noir +1.55`, `MiniMaxH3ReleaseReferences(end_percent=0.33)`):** Measured boundary — both `1D` (`0.4361`) and `Pole-A Erasure` (`0.4328`) stayed bright neon color because `843 + 843` unsteered `tags == 0` Qwen3-VL vision tokens from full-color first+last frames dominate the palette (`showcase3_fl2va_noodle_stacked_141f.mp4`).
  - **Showcase 4 (`ref2va` `<Picture 1>` `woman_blonde.jpg` `paper_origami +1.55` vs. Bookstore `noir +1.5` Selective-Color Lock):** On `woman_blonde_street` (`141f`, `scene_only`, `end_percent = 0.50`, `rel = 0.33`), `paper_origami` (`sat_mean = 0.2242` `1D` vs `0.2126` `Erasure`) placed a triangular folded-paper hood/mask over `<Subject 1>`'s face (`showcase4_ref2va_origami_stacked_141f.mp4`), confirming structural material overrides suit whole-scene `t2va`/`P-H3-Image` better than anchored `<Picture 1>` portraits; on `<Picture 1>` bookstore (`90f`, `noir +1.5`), `Gear 1` (`Pin = 1.0`, `0.3017`) and `Gear 2` (`Rel = 0.33`, `0.3056`) cooled the bookstore to silver-grey vs `1D` (`0.3387`) while holding `<Picture 1>`'s face, oak-thump audio, and pink floral blazer (Selective-Color Subject Lock).
- **Sculpted composite GPU check (`verify_2026-11-21`, Pika 6-step `fl2va` `90f` on `prompt_bank/t2va_covered_market.txt`, `strength = 1.5`, `target = "scene_only"`):** `noir_chaos_composed` (rank-8 pre-upgrade) delivered frantic covered-market chaos cleanly (`sat_mean = 0.422` vs `add_norm_matched` `0.4275`), without artifacts or NaNs. Because a single norm-matched composite at `1.5` splits one shared `101.25` RMS shift budget across both traits, chaining separate `MiniMaxH3Steer` nodes (Showcase 1 above) evaluates each vector's full `|s_i| * 67.5` shift independently against `base_rows64` (`control_vectors.py:3454-3492`) and yields much stronger noir color suppression.
- **Native `MiniMax-H3-Image` GPU check (`--families h3_image`, `4.0 MP` `4:3`, `strength = 1.5`, `target = "scene_only"`):**
  - `noir` (`raw_norm = 51.27`, `vr = (8, 5376)`, `mu_d_norm = 52.59`): `add_norm_matched` stayed warm full color (`sat_mean = 0.2355`, `red_excess = 5.85`), whereas **`add_norm_matched_erasure` flipped cleanly to crisp pure B&W silver-gelatin noir (`sat_mean = 0.033`, `red_excess = -0.04`)** while preserving the exact room layout.
  - `paper_origami` (`raw_norm = 68.63`, build warning `only 2/11 b_side target words survived` from Pole-A noun cancellation): folded orange crates into geometric paper boxes while keeping photographic market shoppers/aisle (`add_norm_matched_erasure` `sat_mean = 0.558` vs `add_norm_matched` `0.5186`); kept experimental in scratch rather than promoting to `models/control_vectors/bank/`.
- **Legacy `2026-11-12` `data/control_vectors/*.toml` lint audit:** All 6 pair sets (`golden_hour_glow`, `heavy_blizzard`, `knitted_wool_world`, `toy_brick_world`, `tv_1950s_broadcast`, `watercolor_wash`; 5 pairs each) fail `lint_pair_set` Gate B (`duplicate b_span` + `Causal prefix collapse: median word first_diff=3 with 1/5 unique prefixes before diff`) and were intentionally not built into `models/control_vectors/`.
