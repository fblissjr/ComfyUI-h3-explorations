# Clamped Pole-A Subspace Erasure & Unpooled Multi-Vector `1x` Sliders in MiniMax H3 (`2026-11-20`)

Dated research study, peer-verified mathematical correction, and controlled disentanglement benchmark (`Phase 1` spectral extraction, `Phase 2` 5-architecture head-to-head, `Round 2A` November-2026 retrieval-SOTA operators, and `Verifier Controls` 8-arm disentanglement with held-out PCA) on moving beyond rank-1 additive control vectors in H3's `5376`-D post-refiner text space.

- **Status:** Mechanistically disentangled and visually verified on single-frame Pika FL2VA SVD995 (`320+` renders at `1152x768`, 6-step `lcm`, `1x` DiT pass). **Not yet wired into production `MiniMaxH3Steer` or validated on multi-second `141f`/`192f` AV video and `ref2va` `<Picture 1>` identity** (see Section 8 open gates).
- **Peer-review correction logged:** Our initial `M3`/`M4` ridge interpretation attributed the gain to "multi-facet Pole-B transport." Peer verification proved algebraically (`max_err <= 5.5e-14`) that regressing `D = B_matched - A` onto `A` is **`97%–99.6%` automatic rank-`r` Pole-A subspace erasure (`-P_{V_r, \lambda}`)**, caught a test-prompt PCA leakage bug in Round 2A, and requested three missing controls (`1D @ s=2.0`, `1D * clean_sif`, and pure Pole-B vs. clamped Pole-A erasure). All three controls were run with strictly held-out PCA (Section 6) and cleanly isolated **two distinct, orthogonal `1x` levers**.
- **Related active wiki pages:** [`../../control_vectors.md`](../../control_vectors.md), [`../../edit_guidance.md`](../../edit_guidance.md), [`../../control_vectors/building_pairs_and_vectors.md`](../../control_vectors/building_pairs_and_vectors.md).
- **Evidence on disk:**
  - Quantitative JSON & cached `.pt` banks/latents (gitignored): `data/exp_unpooled_sliders/phase1_spectra.json`, `data/exp_unpooled_sliders/phase2/metrics.json`, `data/exp_unpooled_sliders/round2a/metrics_r2a.json`, `data/exp_unpooled_sliders/round2a/metrics_m4_ablation.json`, `data/exp_unpooled_sliders/verifier_controls/metrics_controls.json`.
  - Rendered contact sheets (all media outside `data/`): `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/phase2/sheet_*.png` (`7` sheets), `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/round2a/sheet_r2a_*.png` (`10` sheets), and `$H3_OUTPUT_DIR/Image/exp_unpooled_sliders/verifier_controls/sheet_vc_part{1,2}_*.png` (`2` sheets).

---

## 1. The "So What" (Verified Mechanisms After Disentanglement)

1. **Why plain 1D additive vectors hit a wall (`M0_1d`):**
   - 1D mean pooling discards **`52%–77%`** of post-refiner `B - \mu_A` token shift energy (`76.5%` on `noir` and `wes_anderson_symmetry`; `52%–68%` even on 1D winners).
   - More fundamentally, an additive 1D shift $h + s \cdot v_{\text{1D}}$ **never removes contradictory Pole-A directions** already present in `h` (e.g., natural color / daylight / live-action realism). On `noir` (`indoor_neutral`), `M0_1d` stays warm color at `s = 1.5` (`MAE 64.7`), stays warm color at **`s = 2.0`** (`MAE 70.5`), and stays warm color even with **`1D * clean_sif`** (`MAE 79.0`).
2. **Mechanism 1 (True Multi-Vector Win) — Explicit Clamped Prompt-Centered Pole-A Subspace Erasure (`A3_erase_prompt_clp`, `~96.8 KB`, zero external PCA):**
   - Peer verification proved that `M3_subspace`'s conditional term was **`97%–99.6%` subtracting Pole A's top-$r$ subspace** (`cos(ctx_D, -A) = +0.969..+0.996`; pure Pole-B transport `W_B` alone completely failed on `noir`, staying full color at `MAE 58.0`).
   - Writing that mechanism **explicitly** — adding `mu_d` while subtracting a **prompt-centered, clamped-at-1.0** rank-8 Pole-A projector `-min(s, 1.0) * P_{V_r, \lambda}(h - \bar{h})` — requires **zero corpus PCA**, **zero SIF norm inflation**, and **only `96.8 KB` (`bf16`)**, yet:
     - Is the **only non-SIF operator** that flips `noir` to crisp 35mm black-and-white film noir (`MAE 87.0`, beating `1D @ 1.5`, `1D @ 2.0`, `1D * clean_sif`, `pure_poleB`, and `M5_ward_nosif`, ALL of which stayed color).
     - Preserves a full, recognizable crouching folded-paper porter lifting a crate on `paper_origami` (`MAE 74.7`) where `1D @ 1.5`, `1D @ 2.0`, and `1D * clean_sif` all collapsed into empty geometric paper tunnels.
     - Eliminates `M3`/`M4`'s sign-flip overshoot (`-1.5 * P_{V_r}`) and constant prompt-mean wipeout (`(mean(h) - \mu_A)`).
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
1. Compute per-pair optimal-transport matched deltas $D_p = P_{\text{cond}, p} B_p - A_p$ and mean direction $\mu_d = \operatorname{mean}(D)$ (unit-scaled to $\hat{\mu}_d = \tau\,\mu_d / \|\mu_d\|_2$, $\tau = 67.5$).
2. Take the thin SVD of centered Pole-A rows $A_c = A - \mathbf{1}\mu_A^\top = U \Sigma V^\top$ and keep top $r = 8$ right singular vectors $V_r \in \mathbb{R}^{r \times D}$ and ridge erasure weights $g_i = \frac{\sigma_i^2}{\sigma_i^2 + \lambda}$ ($\lambda = 0.15\,\sigma_1^2$).

**Total stored payload (`bf16`, $r = 8$):** `mu_d [5376]`, `Vr [8, 5376]`, `erasure_diag [8]` = **`96.8 KB`** (half the size of `M3`, zero `B_map` or `mu_A` needed, zero external corpus file, 100% backward-compatible alongside `vector.post_refiner`).

#### Online Inference (`1x` DiT Pass, `< 0.1 ms` for $M = 266$ scene rows)

For selected prompt rows $H \in \mathbb{R}^{M \times D}$ with prompt-local row mean $\bar{h} = \frac{1}{M}\sum_{i=1}^M H_i$:

$$E(H) = \left(H - \mathbf{1}\bar{h}^\top\right) V_r^\top \operatorname{diag}(g)\, V_r, \qquad \beta_{\text{cap}} = \min\!\left(1.0,\; \frac{\tau}{\frac{1}{M}\sum_{i=1}^M \|E(H)_i\|_2}\right)$$

$$H'(s) = H + s \cdot \mathbf{1}\hat{\mu}_d^\top \;-\; \operatorname{clip}(s, 0, 1)\cdot \beta_{\text{cap}}\, E(H)$$

*(For negative polarity $s < 0$, the erasure term naturally turns off via $\operatorname{clip}(s, 0, 1) = 0$ so $s < 0$ cleanly subtracts $\hat{\mu}_d$ without amplifying Pole A past its natural manifold!)*

### 3.3 Orthogonal Knob: Minish Lab Clean SIF Token Salience (`w_sif`)

Independently of multi-vector erasure, a held-out corpus mean $\mu_{\text{ref}}$ lets us measure each token's angular distance from the common-mode cone (`"and"`, `"a"`, `"The"`, `"with"` sit at `cos ≈ 0.80–0.88`, while content nouns sit further out):

$$w_{\text{sif}}(h_i) = \text{clip}\!\left(\frac{(1 - \cos(h_i, \mu_{\text{ref}}))^{\alpha}}{\text{mean}_j (1 - \cos(h_j, \mu_{\text{ref}}))^{\alpha}},\; 0.35,\; 2.2\right)$$

with non-alphanumeric punctuation/whitespace tokens clamped to `0.5` (because raw geometric distance assigned `'.'` and `' '` weights `1.88–2.08`). Applying $w_{\text{sif}}$ to the additive term concentrates the fixed `67.5` mean norm onto content tokens — **but it must pass the `192f` AV video gate** because `token_focus = 0.30` lost `3/3` on video when skewing per-token shift norms diluted cross-modal attention mass.

---

## 4. November 2026 Multi-Vector Retrieval SOTA — What Actually Held Up Under Controls

We tested ideas from **LightOn / Benjamin Clavié** (`PyLate`, `LateOn`, Ward hierarchical token pooling), **Mixedbread** (`mxbai-edge-colbert-v0`, *Latent Terms* `arXiv:2605.29384`), **Omar Khattab** (`ColBERTv2` / `PLAID` / `WARP` SIGIR 2025), **Minish Lab** (Stéphan Tulkens & Thomas van Dongen: `reach`, `Model2Vec`, `POTION`), and **Balanced Sinkhorn OT**:

| Research Source (as of Nov 2026) | Mechanism Tested | Controlled Verdict in H3 `5376`-D Space |
| :--- | :--- | :--- |
| **Subspace Projection / Erasure** *(emergent in `M3`, isolated in `A3`)* | **Clamped prompt-centered rank-8 Pole-A erasure (`A3`)** | **#1 verified multi-vector winner:** only non-SIF method that turns `noir` into crisp B&W (`87.0`) and preserves a crouching origami porter on `paper_origami` (`74.7`), with zero corpus PCA and `96.8 KB` storage. |
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

*(Note on `A3`/`A4` norm & clamp hygiene: in this control pass, `A3` and `A4` did not renormalize their combined `mu_d - erasure` vector back to `s * 67.5`, so partial cancellation left `watercolor_wash` slightly under-driven at `98.0` vs. `M3`'s `101.25`; and the scratch script coded `min(s, 1.0)` rather than `max(0.0, min(s, 1.0))` — fixed for Round 2B, where all arms will use matched **RMS** total-shift normalization.)*

### Additional Verified Findings from Round 2A (Still Valid After Controls)
- **Rank-sweep phase transition (`sheet_r2a_rank_sweep.png`):** On `M4_minish_subspace` (`s = 1.5`), `rank = 2` fails on both `noir` (stays warm color, `81.5`) and `watercolor_wash` (stays photorealistic, `77.8`), while `rank = 4..8` crosses sharply into crisp B&W noir (`80.7–84.7`) and painted watercolor (`104.4–113.3`). This reflects that the erased Pole-A subspace requires **`r = 4..8` dimensions** (not 1–2).
- **Round 2A `M4` ablation visual caution (`sheet_r2a_m4_ablation.png`):** On `watercolor_wash`, `M3_subspace` (`97.6`) and `M4_ot_pca_nosif` (`95.3`) kept full figures, crates, hanging scale, and market background intact with zero SIF, whereas `M4_ot_pca_sif_clean` (`102.8`) suffered severe compositional collapse (cropped giant mouths/chins with eyes cut off at top edge) and `M4_full` (`113.3`) erased the background to a white void.
- **Live 2-slider stacking `noir (1.3) + chaos_panic (1.0)` (`sheet_r2a_stacking_2sliders.png`):** `M0_1d` and `M6` stay full color; `M3`, `M4`, and `M5` all render a pure monochrome 1940s film-noir market riot in one `1x` pass.
- **Negative polarity (`sheet_r2a_polarity_{noir,chaos_panic}.png`):** Clean inversion at `s = -1.0..-0.5` (`chaos_panic` → calm serene order; `noir` → warm open daylight).
- **Test-prompt conflict (`sheet_r2a_look_direct_1990s_flash__indoor_evening.png`):** Removing `"sunlit ... near a tall window"` unlocked stark 1990s direct xenon flash + tight black drop shadows across all methods, including plain `M0_1d` at `s = 1.5`.

---

## 7. What This Informs Across the Repo

1. **Leanest, cleanest multi-vector `.safetensors` upgrade (`~96.8 KB`, backward-compatible):** Add optional `erasure.mu_d [5376]`, `erasure.vr [8, 5376]`, and `erasure.diag [8]` alongside `vector.post_refiner [5376]` inside existing `.safetensors` files. Zero external corpus file, zero test-prompt leakage, zero RoPE shift, exact `0.0000` at `s = 0`, and never sign-flips past `s = 1.0`.
2. **Keep Pole-A Erasure (`A3`), Ward+`WARP_SELECT` (`M5`), and Clean SIF Salience (`w_sif`) as *separate* experimental dials:**
   - Pole-A Erasure (`A3`) solves contradictory base attributes (`noir` overcoming color, `paper_origami` retaining the human figure) without skewing per-token norms.
   - `M5_ward_warp_hybrid` (`A6`/`A7`) is the strongest bounded-residual alternative for preserving standing figures and full market corridors under heavy stacking (`stack_water_wes`).
   - Clean SIF (`w_sif`) concentrates additive norm onto content words (helping `watercolor_wash` and `1990s_flash` onset) — **but increases RMS/peak content-token energy (`1.6x–2.3x`) and carries a real attention-dilution risk on video**, so it must be RMS-matched and gated independently on `192f` AV clips.
3. **Contrastive `.toml` hygiene:**
   - Note on `data/control_vectors/watercolor_wash.toml` (5 pairs): `"lips"` / `"mouth"` appears identically in the shared `a` and `b` dialogue-closing suffix (not in `diff_span`), while `diff_b` repeats the same 11-word prefix across all 5 pairs (`len(diff_b) > len(diff_a)`). Its norm-`135` lips/fruit-slice attractor (`A1_1d_s2.0`) likely reflects bidirectional `token_refiner` length-offset coupling or H3's prior at extreme shift norm rather than an extraneous Pole-B diff clause; either way, high-norm stacking requires a combined-norm cap.
   - Keep natural contextual variation on both Pole A and Pole B (`prompt_bank/control_vectors/noir.toml` style) rather than verbatim copy-paste prefix clauses (`internal/internal_prompt_bank/control_vectors_v2/look_direct_1990s_flash.toml`, `mpc = 0.936`).
   - Log `1D_E%` and `|mu_d| / E|d|` alongside `mpc` in `bench/control_vectors.py build`.

---

## 8. Open Verification Gates (Must Pass Before Any Production Integration)

Nothing below is proven yet — treat all four as mandatory **Round 2B** gates before touching `control_vector_nodes.py` or `MiniMaxH3Steer`:

1. **Real `192f` AV Video Gate (`fl2va` & `ref2va` `<Picture 1>`, RMS-matched arms):** Test `A0_1d` vs. **RMS-normalized** `A3_erase_prompt_clp` (`max(0.0, min(s, 1.0))`) vs. `A2_1d_clean_sif` vs. `M5_ward_warp_hybrid` (`alpha = 0.5` clean SIF) on multi-second `192f` `1152x768` Pika AV clips (`noir`, `watercolor_wash`, `paper_origami`, `chaos_panic`). Verify temporal coherence, **speech / soundscape clarity** (explicitly testing against the `ab_token_focus_2026-11-20` audio/attention-mass warning), and `<Picture 1>` face/identity preservation + `retention_analysis:` integrity on `ref2va`.
2. **Rank sweep directly on RMS-matched `A3_erase_prompt_clp` (`r = 2, 4, 8, 16`)** to confirm whether clean clamped erasure shares the same `r = 4..8` knee without SIF.
3. **Held-out `prompt_bank/` corpus fit** for any production `clean_sif` / `M5` basis (never fit on test prompts).
4. **Multi-seed & multi-prompt check** (at least 3 seeds across `t2va` + `ref2va` prompts) + optional H3 DiT block-band hook (`comfy/ldm/minimax/model.py:660-669`) testing if needed.
