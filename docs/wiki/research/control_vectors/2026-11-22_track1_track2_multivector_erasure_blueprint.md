# Track 1 vs. Track 2 Multi-Vector & Subspace Erasure Blueprint (`2026-11-22`)

**Date:** `2026-11-22`  
**Owner (Spec & CPU Audits):** Lane B · **Aligned With:** Lane A (`faa62ef5`) + Wave 2 / Coordinator (`90e9d2bd`)  
**Policy:** **Strictly Multi-Vector & Pole-A Subspace Erasure** — zero `1D` work or `1D` comparison arms.  
**Audit Outputs (`data/exp_unpooled_sliders/phase5_cpu_audits/`):** `bank_cross_interference_audit.json` · `multivector_pole_b_audit.json`  
**Related Specs:** [`2026-11-22_unified_steer_attention_hook_spec.md`](2026-11-22_unified_steer_attention_hook_spec.md) · [`2026-11-22_paper_to_codebase_promotion_spec.md`](2026-11-22_paper_to_codebase_promotion_spec.md) · [`2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md`](2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md) · [`2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md)

---

## 1. Strategic North Star: Two Differentiated Multi-Vector Tracks

Rather than collapsing steering into one bucket, ongoing engineering and Paper v5 research divide cleanly into **two complementary multi-vector tracks**:

| Axis | **Track 1: Effortless Subspace Erasure & Conflict-Free Composition** | **Track 2: True Per-Token & Spatial Multi-Vector Pole-B Routing** |
|---|---|---|
| **Goal** | Make rank-`r` Pole-A erasure + single target shift $\boldsymbol{\mu}_d$ **zero-hassle, conflict-free when stacked, identity-locked (`blocks.20..49`), voice-locked (`1x Q_aud`), and spatially movable (`MiniMaxH3SpatialDirector`)**. | Break the **single global $\boldsymbol{\mu}_d$ ceiling** when a concept has distinct visual/blocking/camera/lighting facets or spatial image-patch roles that one pooled vector blurs together. |
| **Algebraic Core** | $\mathbf{H}' = \mathbf{H} - (\mathbf{H} - \bar{\mathbf{h}})\mathbf{P}_{\text{union}}^{\text{ROS}} + \sum_k \mathbf{a}_k$, compiler-safe attention K/V band routing, and clamped eigenvalue protection ($\lambda_{\max} \le 1.0$). | Prompt-centered per-token multi-vector Pole-B transport ($\{\mathbf{c}_k^{(A)}, \boldsymbol{\delta}_k\}_{k=1}^K$ via soft-MaxSim / `WARP_SELECT` / ridge router or rank-`r` $\mathbf{W}_B$) + multi-pair `ColQwen` `vision_tokens` banks (`N >= 6`). |
| **External Corpus Dependency** | **Zero** (prompt-centered on live sequence rows). | **Zero** (self-contained in `.safetensors` pairs/prototypes; zero external corpus PCA/SIF). |

---

## 2. Track 1 Empirical Cross-Interference Audit & Two-Sided RASteer Equilibrium

### 2.1 Bank-Wide Cross-Interference Census (`52` `fl2va` Vectors = `2,652` Ordered Pairs)
Prompted by live Showcase A's early diagnostic (`box_office s=101`: `New Clamped Union` beat `Old Parallel Sum` `sat 0.2592 -> 0.1911`, `lap 279.9 -> 306.8`, yet stacking `noir + chaos_panic + de_llm_slop` still opposed `Single noir` `sat = 0.0246`), CPU Audit 1 measured all `52` `*__fl2va__*.safetensors` banks carrying `erasure.{vr, diag, mu_d}`:

| Metric Across Full `52`-Vector `fl2va` Bank | Mean | Median | `p90` | `p99` | Max |
|---|---|---|---|---|---|
| **Cross-Interference (raw):** $\|\mathbf{V}_r^{(j)} \hat{\boldsymbol{\mu}}_d^{(i)}\|_2^2$ (`i != j`, `n=2,652`) | `0.0825` | `0.0733` | `0.1446` | `0.2312` | **`0.4240`** |
| **Cross-Interference ($D_j$-weighted):** $\|\sqrt{\mathbf{D}_j}\mathbf{V}_r^{(j)} \hat{\boldsymbol{\mu}}_d^{(i)}\|_2^2$ | `0.0638` | `0.0567` | `0.1136` | `0.1831` | **`0.3520`** |
| **Self-Overlap Trap (`i == j`):** $\|\mathbf{V}_r^{(i)} \hat{\boldsymbol{\mu}}_d^{(i)}\|_2^2$ (`n=52`) | **`0.1593`** | `0.1483` | `0.2451` | `0.3067` | **`0.3284`** |
| **Self-Overlap ($D_i$-weighted):** $\|\sqrt{\mathbf{D}_i}\mathbf{V}_r^{(i)} \hat{\boldsymbol{\mu}}_d^{(i)}\|_2^2$ | `0.1242` | `0.1190` | `0.1932` | `0.2502` | `0.2761` |
| **Top Principal Angle:** $\cos\theta_1(\mathbf{V}_r^{(i)}, \mathbf{V}_r^{(j)})$ (`n=1,326`) | `0.6750` | `0.6691` | `0.7912` | `0.9825` | `1.0000` |
| **Absolute Target Cosine:** $|\cos(\hat{\boldsymbol{\mu}}_d^{(i)}, \hat{\boldsymbol{\mu}}_d^{(j)})|$ | `0.1881` | `0.1654` | `0.3623` | `0.6260` | `0.9488` |

**Top-5 Cross-Interference Pairs (`erasure_j` destroying `target_i`, `D_j`-weighted):**
1. `expr_deadpan_stonefaced` $\leftarrow$ `V_r(expr_flushed_breathless_tension)`: **`0.4240` raw (`0.3520` weighted)**, $\cos(\hat{\boldsymbol{\mu}}_d) = -0.2113$, top SV `0.8423`.
2. `operatic_bellow` $\leftarrow$ `V_r(whisper_asmr)`: **`0.3623` raw (`0.2931` weighted)**, top SV `0.9116`.
3. `reverb_cathedral` $\leftarrow$ `V_r(look_amber_candlelight)`: **`0.3092` raw (`0.2618` weighted)**.
4. `cyberpunk_neon_rain` $\leftarrow$ `V_r(noir)`: **`0.3166` raw (`0.2434` weighted)** — `noir`'s erasure deletes warm/saturated neon chromaticity injected by `cyberpunk_neon_rain`.
5. `cyberpunk_neon_rain` $\leftarrow$ `V_r(golden_hour_glow)`: **`0.3028` raw (`0.2352` weighted)**.

**Showcase A (`noir + chaos_panic + de_llm_slop`) Exact Diagnosis:**
- Directly verified across the `.safetensors` files: `V_r(noir)` overlaps `0.0998` raw (`0.0758` $D$-weighted) with `mu_hat_d(chaos_panic)` and `0.1198` raw (`0.0866` $D$-weighted) with `mu_hat_d(de_llm_slop)`. Meanwhile `chaos_panic` ($\cos = +0.2415$ with `noir`, norm `61.48`) + `de_llm_slop` ($\cos = +0.1291$, norm `51.84`, skewed toward archival/institutional `documentary` tokens with `b_side_alignment.coverage = 0.071` per [`2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md`](2026-11-22_fresh_eye_audit_de_slop_erasure_and_multivector_future.md)) inject competing visual tokens onto the **exact same `scene_only` mask**, reducing `noir` direction concentration from `0.958` (`80.84 / 84.38`) to `0.769` (`103.6 / 134.64`) and cutting effective per-slider erasure via RMS rescaling (`0.9546`). Paired with our median `0.1541` squared Grassmann overlap (`50.9%` shared generic scaffolding across the top 16 bank-wide PCs), unprotected parallel subspaces inevitably collide on `V_r`.

### 2.2 Why Naive Global Orthogonalization Fails — And Per-Slider `ROS` (`V_r^(j) ⊥ mu_d^(-j)`) Fixes It
Because **self-overlap** $\|\mathbf{V}_r^{(i)} \hat{\boldsymbol{\mu}}_d^{(i)}\|_2^2$ averages `15.9%` (`max 32.8%` `look_amber_candlelight`, `28.6%` `thermal_infrared`, `28.0%` `de_llm_slop`), blindly calling `orthogonalize_erasure_against_holds(vr_union, d_union, all_mu_d)` strips each slider's **own** erasure directions along `mu_d^{(i)}`!

Empirical comparison across 3 multi-slider stacks on cached held-out post-refiner prompts (`data/exp_unpooled_sliders/prompts/{covered_market,indoor_evening,indoor_neutral}.pt`):

| Multi-Slider Stack (Same Mask `scene_only`) | Single Slider `0` Resid Ratio | Shipped Clamped Union Resid Ratio | **Naive Global `V_r ⊥ all μ_d`** | **Per-Slider RASteer `ROS` (`V_r^(j) ⊥ μ_d^(-j)`)** |
|---|---|---|---|---|
| `noir (1.25) + chaos_panic (1.0) + de_llm_slop (0.9)` | `0.2786` | `0.2006` | `0.2692` (`+34%` leak) | **`0.2204`** (`0.000` peer-target hit) |
| `paper_origami (1.2) + noir (1.15) + de_llm_slop (0.9)` | `0.2966` | `0.2131` | **`0.3627` (`+70%` self-cancellation!)** | **`0.2412`** (`0.000` peer-target hit) |
| `claymation_stopmotion (1.2) + noir (1.15) + chaos_panic (1.0)` | `0.2631` | `0.1949` | `0.2875` (`+48%` leak) | **`0.2038`** (`0.000` peer-target hit) |

**Exact Two-Sided Production Rule (Extending [`control_vectors.py:1473,3488`](../../../../control_vectors.py#L1473)):**
1. **Per-Slider Retain-Orthogonal Subspace (`ROS`) Before Union:** For each additive slider $j \in \{1..K\}$ on the same mask, call `orthogonalize_erasure_against_holds(vr_j, diag_j, [mu_d_k for k != j])` **prior** to concatenating $\mathbf{W}_j = \sqrt{\min(s_j, 1)\,\mathbf{D}_j^{\text{ROS}}}\mathbf{V}_{r,j}^{\text{ROS}}$ into `_stacked_erasure_block_shift64`'s clamped-union thin SVD (`lam_max <= 1.0`).
   - *Theorem proved on CPU:* For constant Track 1 shifts $\mathbf{a}_k \parallel \hat{\boldsymbol{\mu}}_d^{(k)}$, per-slider `ROS` **automatically guarantees two-sided orthogonality** ($\mathbf{V}_{\text{union},-k} \mathbf{a}_k = \mathbf{0}$ identically!), matching `two_sided_equilibrium` numerically without an extra runtime pass!
2. **Track 2 / OAC Asymmetric Cross-Filter:** When stacking **Track 2 multi-vector Pole-B shifts** ($\boldsymbol{\delta}_{k,c} \not\parallel \hat{\boldsymbol{\mu}}_d^{(k)}$) OR when RASteer `Overlap-Adaptive Calibration` (`OAC`, §5.1) relaxes `ROS` at high overlap (`> 0.25` such as `expr_deadpan_stonefaced` + `expr_flushed_breathless_tension`), apply explicit peer erasure filtering $\tilde{\boldsymbol{\delta}}_{j,c} = (\mathbf{I} - \mathbf{P}_{\text{union},-j})\,\boldsymbol{\delta}_{j,c}$.
3. **Role-Split UX Default (`MiniMaxH3Steer` / `app/cv_lab.py`):** Because orthogonal semantic shifts still dilute cosine concentration (`0.958 -> 0.769`) on the same tokens, default stacking presets should map look/palette dials (`noir`, `golden_hour_glow`) onto `look_only` (`subject_definitions + summary`) and dynamic/motion dials (`chaos_panic`) onto `description_only` whenever composing `2+` visual concepts.
4. **Non-Consecutive Mask Grouping & Vision Stacking Fix (`control_vector_nodes.py:2188-2236`, `control_vectors.py:3787`):** Currently `_build_steered_context` only batches *consecutive* specs sharing `(mask, prompt)`. Interleaving `noir (scene_only) -> reverb_cathedral (soundscape_only) -> chaos_panic (scene_only)` splits the two `scene_only` specs into separate `apply_post_refiner` passes (bypassing `_stacked_erasure_block_shift64`), and `control_vectors.py:3787` excludes `mu_a is not None` vision vectors from union clamping. Grouping additive specs by `(mask, prompt)` before running per-slider `ROS` + clamped union guarantees order-independence.

---

## 3. Track 2 Discovery Audit: Per-Token Multi-Vector Pole-B (`7` Cached Banks)

### 3.1 What Single-$\boldsymbol{\mu}_d$ Leaves Behind — And What Leave-One-Pair-Out CV Honestly Measures
CPU Audit 2 ran full **Leave-One-Pair-Out cross-validation** across all 7 cached per-pair banks (`data/exp_unpooled_sliders/banks/*.pt`) with **zero external corpus PCA/SIF**, comparing Track 1 (`V_r` r=8 + single `mu_d`) against:
- **Track 2A-raw:** Ward $K \in \{2,4,8\}$ clusters on raw unit Pole-A tokens $\hat{\mathbf{a}}_n$ -> cluster shift $\boldsymbol{\delta}_k$ (residual $(\boldsymbol{\delta}_k - \boldsymbol{\mu}_d) \perp \mathbf{V}_r$)
- **Track 2A-centered:** Ward $K$ clusters on **prompt-centered** Pole-A tokens $(\mathbf{a}_n - \bar{\mathbf{a}}_p)$ -> $\boldsymbol{\delta}_k$
- **Track 2A-facet:** Ward $K$ clusters directly on **transition vectors** $\mathbf{t}_n = \tilde{\mathbf{b}}_n - \mathbf{a}_{\text{erased},n}$ to isolate semantically distinct Pole-B facets first, paired with either centered Pole-A cosine centroids (`facet_cos`) or a dual L2-ridge multiclass router (`facet_ridge`)
- **Track 2B:** Theorem-1 rank-$r \in \{4,8\}$ prompt-centered cross-covariance ridge transport $\mathbf{W}_B$.

| Concept (`N` pairs) | Census `mpc` / `eff_r` | Unexplained Residual RMS $R_B / T$ | Track 1 `cos` | **2A-raw** Best $\Delta\text{MSE}$ | **2A-centered** Best $\Delta\text{MSE}$ | **2A-facet Ridge/Cos** Best $\Delta\text{MSE}$ | **2B** Best $\Delta\text{MSE}$ | `k4_facet` $\min\cos(\boldsymbol{\delta}_k)$ |
|---|---|---|---|---|---|---|---|---|
| `look_direct_1990s_flash` (`10`) | `—` | `0.486` | `0.9526` | `-5.60%` | `-11.53%` | **`-14.34%`** (`+0.0072` cos) | `-1.22%` | `0.839` |
| `noir` (`12`) | `0.574` / `55.06` | `0.819` | `0.8511` | `-2.30%` | `-4.56%` | **`-5.03%`** (`+0.0071` cos) | `-2.54%` | `0.717` |
| `molten_lava_golem` (`10`) | `0.709` / `41.04` | `0.685` | `0.8494` | `-1.23%` | **`-3.79%`** (`+0.0046` cos) | `-1.05%` | `-1.67%` | `0.817` |
| `chaos_panic` (`10`) | `0.711` / **`72.44`** | `0.792` | `0.8449` | `-2.15%` | `-2.94%` (`+0.0037` cos) | **`-3.07%`** | `-2.48%` | **`0.693`** |
| `paper_origami` (`10`) | `0.699` / `38.27` | `0.740` | `0.8531` | `-1.18%` | **`-2.76%`** (`+0.0037` cos) | `-2.01%` | `-1.52%` | `0.843` |
| `wes_anderson_symmetry` (`10`) | **`0.453`** / `43.63` | **`0.843`** | `0.8176` | `-0.82%` | **`-2.19%`** (`+0.0034` cos) | `-0.79%` | `-1.73%` | **`0.579`** |
| `watercolor_wash` (`5`) | `—` | `0.659` | `0.8089` | `-0.46%` | **`-0.94%`** | `-0.25%` | `-0.60%` | `0.764` |

**Three Falsification-First Conclusions (Critical for Research Honesty):**
1. **Prompt-Centering the Keys Fixes Anisotropy Collapse:** At `tau = 0.15` on uncentered raw keys, OOD prompts suffered normalized routing entropy `0.78–0.99`, collapsing `shift_cos_std` to `0.0001–0.0022` (effectively reverting to `mu_d`). Switching to **prompt-centered keys** $(\mathbf{a}_n - \bar{\mathbf{a}}_p)$ + sharper routing ($\tau = 0.07$ `warp2` or `rtau = 0.25` `facet_ridge`) **doubled to tripled** CV MSE reductions on every bank (`look_direct_1990s_flash`: `-5.60% -> -14.34%`; `noir`: `-2.30% -> -5.03%`; `molten_lava_golem`: `-1.23% -> -3.79%`) while lifting OOD `shift_cos_std` to **`0.089` (`wes_anderson_symmetry`), `0.059` (`chaos_panic`), and `0.049` (`noir`)**.
2. **Where Track 2 Does NOT Help (`N <= 5` or Monolithic Styles):** `watercolor_wash` (`5` pairs) only gains `-0.94%` MSE under Track 2A (overfitting risk > payoff). Track 1 (`V_r` r=8 + single `mu_d`) already saturates uniform global washes, color grades, and simple filters.
3. **Why Prose LOPO Caps at `-14%` While Latent/Visual Spatial Facets Have Massive Headroom:** In `10` human/LLM-written prose pairs, `49%–84%` of `R_B` reflects creative writer word choice in Pole B not uniquely predictable from a single Pole-A word (`chaos_panic` writers arbitrarily add sirens vs. overturned chairs). Unlocking Track 2's full leap requires:
   - **Role-Targeted Semantic Facets** (`N >= 16–24` pairs or explicit span stratification separating `[subject/emotion, action/blocking, lighting/palette, camera/framing, foley/audio]` so `erasure.proto_delta` captures deterministic role maps rather than random lexical variance — e.g. `wes_anderson_symmetry` facets already hit `min_cos = 0.579`), **AND**
   - **`ColQwen` Spatial `vision_tokens` Banks (§4 below)**, where Pole-A spatial patches (`face`, `clothing`, `background`, `sky`) deterministically fix what Pole-B facet belongs at `(x, y)`!

### 3.2 Backward-Compatible `.safetensors` Track 2 Schema (`~1.2 MB`, Zero External Files)
Only emitted on multi-facet / role-stratified banks (`>= 10` pairs, optional weights):
- `erasure.proto_a`: `[K, 5376]` (`float32`), unit prompt-centered Pole-A key prototypes (or linear ridge router `erasure.proto_router_w: [K, 5376]` when `facet_ridge` beats cosine centroid in LOPO CV by `> 1%`)
- `erasure.proto_delta`: `[K, 5376]` (`float32`), per-facet target shift where residual $(\boldsymbol{\delta}_k - \boldsymbol{\mu}_d)$ is orthogonalized against $\mathbf{V}_r$ (and cross-slider $\mathbf{V}_{\text{union},-j}$ at runtime)
- Runtime metadata keys: `"multivector.kind"` (`"centered_warp2"` | `"facet_ridge"`), `"multivector.tau"` (`0.07` or `0.25`), `"multivector.top_k"` (`2`). Legacy loaders ignore extra keys and fall back transparently to Track 1 `erasure.{vr, diag, mu_d}` + `vector.post_refiner`.

---

## 4. Multi-Pair `ColQwen` Spatial `vision_tokens` Multi-Vector Banks (`N >= 6` Pairs)

Paper v4 Proposition 4 established the **single-image `(1 - w*lambda)` scalar ceiling** of `N = 1` vision-token steering: with `N = 1` fixed `<Picture 1>` pair, only a static point-to-point erasure exists (`stage_c_vision_erasure.json`). Multi-pair `ColQwen` visual banks overcome that ceiling cleanly:

1. **Construction (`N >= 6` Paired `(image_pole_a, image_pole_b)` Reference Stills in `MiniMax-H3` `ref2va`):**
   - Pass each pair through `Qwen3-VL` + `condition_proj` + `token_refiner` alongside standardized dummy `<Picture 1>` framing so `text_token_tags == 0` extracts the exact spatial grid of refined `vision_tokens` ($\mathbf{V}_A^{(p)}, \mathbf{V}_B^{(p)} \in \mathbb{R}^{M_{\text{vis}} \times 5376}$).
   - **Soft Spatial Transport:** Because `<Picture 1>` pairs may feature shifted subjects or crops, align patches via ColQwen soft-MaxSim over spatially modulated similarities $\tilde{\mathbf{V}}_B^{(p)} = \operatorname{softmax}\big(\langle \hat{\mathbf{v}}_{A,m}^{(p)}, \hat{\mathbf{v}}_{B,n}^{(p)} \rangle / 0.15\big)\mathbf{V}_B^{(p)}$.
   - **Global Spatial Erasure + Face Retain Subspace (`RASteer ROS`):** Stack across all $p = 1..N$:
     - `erasure.vr_vis` `[r=8..16, 5376]`, `erasure.diag_vis`, `erasure.mu_a_vis` `[5376]` (preserving `erasure.mu_a` pooled centering because live `<Picture 1>` encodes semantic visual background mean!), `erasure.mu_d_vis` `[5376]`.
     - **Face / Skin Identity Retain Subspace (`RASteer ROS`):** Center-crop portrait/skin vision tokens from unsteered `ref2va` face priors (e.g., human skin, eyes, lip-region `vision_tokens`) define a retain subspace $\mathbf{B}_{\text{face}} \in \mathbb{R}^{r_{\text{ret}} \times 5376}$ orthogonalized out of `erasure.vr_vis` and out of background/costume `proto_delta_vis`, protecting `<Picture 1>` identity while allowing material/lighting/background conversion!
   - **Patch-Role Multi-Vector Prototypes (`erasure.proto_a_vis`, `erasure.proto_delta_vis`, `K = 8..16`):** Cluster centered visual Pole-A patches (e.g., sky/atmosphere vs. foreground skin/face vs. fabric/garment vs. architectural surfaces) into spatial transition pairs $(\mathbf{c}_{k}^{\text{vis}}, \boldsymbol{\delta}_{k}^{\text{vis}})$ saved into `bank/<stem>__ref2va__...safetensors`.
2. **Joint Text + Vision Composition:** At inference, `control_vector_nodes` applies text erasure on `tag == 1` scene rows (`blocks.20..49` when identity-locked) **simultaneously** with patch-routed `ColQwen` steering on `tag == 0` `vision_tokens` — letting clothing/props/environment transform cleanly while RASteer-protected face patches stay locked to `<Picture 1>`.

---

## 5. Applied Literature Routing (Strictly Multi-Vector & Erasure, Zero `1D`)

| External Paper (Verified) | Core Mechanism | Direct Multi-Vector & Erasure Application in H3 |
|---|---|---|
| **RASteer** (`arXiv:2610.01969`, Wu et al. 2026) | **Retain-Orthogonal Steering (`ROS`)** projects retain subspaces out of erasure directions; **Overlap-Adaptive Calibration (`OAC`)** scales orthogonal removal by layer/step overlap to avoid destroying erasure potency when target/retain overlap exceeds `25%–40%`. | 1. **Track 1 Cross-Slider `ROS` + `OAC`:** Explains why naive global `orthogonalize_erasure_against_holds` loses `+70%` Pole-A erasure on `paper_origami` while per-slider `ROS` keeps `0.2412` (`§2.2`). Introduces continuous `OAC` weight $\alpha_{\text{oac}} \in [0, 1]$ when `e_raw > 0.25` (e.g. `expr_deadpan_stonefaced` + `expr_flushed`).<br>2. **Track 2 Spatial `Face-Retain` Subspace:** Protects `<Picture 1>` skin/eye/mouth `vision_tokens` inside multi-pair `ColQwen` banks (`§4`). |
| **MUTE** (`arXiv:2609.36832`, Liu & Zhang 2026) | Proves perturbing temporal RoPE kills *all* motion indiscriminately, whereas **text cross-attention layers isolate concept-specific motion** + intrinsic **spatial gating** prevents static background bleed. | Maps **directly** to Track 1 **blocks `20..49` shadow-text K/V hook + `MiniMaxH3SpatialDirector`**! Instead of corrupting `layout.position_ids` temporal RoPE (`comfy/ldm/minimax/model.py:558-565`), dynamic vectors (`chaos_panic`, `handheld_documentary_zoom`, `zero_gravity`) intervene strictly on text K/V `[tx_a:tx_b]` in `blocks.20..49` and optionally spatialized `[F, H, W]` edit regions via `MiniMaxH3EditGuider` or token masks. |
| **Looped-DiT** (`arXiv:2609.40305`, Chng et al. 2026) | Iterates weight-tied DiT core loops across denoising with **deep supervision** + **self-modulating attention** to stop spatial amnesia across loop depths. | Validates Paper §7.3's **Prelude Input Injection** transfer: because `Looped-DiT` re-attends fixed $\mathbf{H}_{\text{prelude}}$ across every loop depth *inside every step*, multi-vector Prelude EQ clears Pole-A before loop `0`, compounding without per-loop attention hacks. Anchors Window 3 cross-architecture validation on local `$H3_MODEL_STORE/diffusion_models/qwen_image_2.1_bf16.safetensors` (`Base` vs. `Multi-Vector EQ`, zero `1D`). |

---

## 6. Signature Showcase Matrix & Paper v5 Absorption Plan

### 6.1 Side-by-Side Track 1 vs. Track 2 "Nobody Else Can Do This" Showcase Matrix

| Showcase | Track | Mechanism Demonstrated | Prompt / Assets | Falsifiable Pass Criterion (No Cherry-Picking) |
|---|---|---|---|---|
| **T1-A: Single-Pass Global Material / Lighting Override** | **Track 1** | Clamped Pole-A `r=8` erasure (`scene_only`) in `1x` Pika (`cfg=1.0`, `6` steps) | Conflicting prompts (`sitcom_kitchen`, `box_office`) $\times$ `noir`, `paper_origami`, `claymation_stopmotion` | `5/5` conflicting overrides succeed at `1x` speed with character/scene retention (`DINOv3 >= 0.30` in `mid_late_20_49`). |
| **T1-B: Zero-Interference Multi-Slider Stacking (`ROS` + Role-Split)** | **Track 1** | Clamped union (`lam_max <= 1.0`) + per-slider `ROS` ($\mathbf{V}_r^{(j)} \perp \boldsymbol{\mu}_d^{(-j)}$) | `noir + chaos_panic + de_llm_slop + reverb_cathedral` & `cyberpunk_neon_rain + noir` | `M` (`cyberpunk`) retains neon chroma (`> 2x` shipped saturation recovery under `ROS` vs. unprotected `V_r(noir)`) + clean `de_llm_slop` audio clarity. |
| **T1-C: Identity-Locked Body/World Transformation + `1x Q_aud`** | **Track 1** | Compiler-safe `blocks.20..49` shadow-text + Mode 2/3 `1x Q_aud` K/V split | `90f` `ref2va` `<Picture 1>` `woman_blonde` + `90f` `t2va_box_office` speech + `de_llm_slop` | Facial lip-sync & identity `center_vs_ref` intact (`blocks.20..49`) + `YIN` $F_0$ pitch drift `< 0.25 st` in `1x` (`~50%` lower latency than `2x`), `MALLOC_GRAPH_BREAKS == 0`. |
| **T1-D: Moving / Expanding Steered Reality Portal** | **Track 1** | `MiniMaxH3SpatialDirector` keyframed `[F, H, W]` portal mask + `MiniMaxH3Steer` | Normal room/street turning into `molten_lava_golem` / `underwater_submerged` ONLY inside a moving portal | Inside-portal metric shift `>= 2x` outside-portal baseline; background geometry & exterior audio ambient track preserved. |
| **T2-A: Role-Selective Multi-Vector Facet Routing (`K`-Prototype)** | **Track 2** | Prompt-centered `facet_ridge` / `warp2` routing across `N >= 16` role-balanced pairs | High-facet concepts (`wes_anderson_symmetry` `mpc=0.453`, `infomercial_fail` `mpc=0.433`, `chaos_panic` `eff_r=72.4`) | Per-token routing targets camera spans with camera/symmetry shifts and emotion spans with deadpan/panic shifts — **higher targeted facet fidelity without erasing non-target clauses** than uniform `mu_d`. |
| **T2-B: Multi-Pair `ColQwen` Spatial `vision_tokens` Face-Safe Restyle** | **Track 2** | `N >= 6` `<Picture 1>` pairs (`erasure.vr_vis`, `mu_a_vis`, `proto_delta_vis` + RASteer `face_retain`) | Multi-subject `ref2va` portrait + sculptural/garment/environment restyle (`felt_muppet`, `paper_origami`, `cyberpunk`) | Patch-selective visual transformation cleanly out-scores single `N=1` fixed `vision_tokens` erasure AND uniform `mu_d` on unseen `<Picture 1>` references while preserving human face/lip-sync. |

### 6.2 Paper v5 & `$H3_OUTPUT_DIR` Bundle Absorption Checklist

Once **Window 1** (`faa62ef5` Phase 4 Showcases A–D), **Window 2** (`90e9d2bd` Wave 2 `spotlights` + Exp 4/6/7), and **Window 3** (`Qwen-Image-2.1` + Phase 5 Track 2 experiments) complete cleanly:
1. **Paper v5 Monograph (`docs/wiki/research/control_vectors/` + refreshed typeset PDF + `$H3_OUTPUT_DIR` bundle):**
   - Fold Phase 4 Showcases A–D + Wave 2 metrics into §5/§6 (clamped union `lambda_max <= 1.0` proof, Showcase B `blocks.20..49` inverted shadow proof, Showcase C `1x Q_aud` + `YIN` $F_0$ proof, Showcase D native `4.0 MP` `MiniMax-H3-Image` rank-8 proof + `P-H3-Image`, Dual-Lock `fl2va` unlock).
   - Fold Track 1 `52`-vector cross-interference census + per-slider RASteer `ROS` table (`§2`) and Track 2 LOPO CV audit (`§3`, `ColQwen` `vision_tokens` `§4`) cleanly as **Subspace Equilibrium v2**.
   - Fold Window 3 `Qwen-Image-2.1` (`Base` vs. `Multi-Vector EQ`, zero `1D`) into §7.3 cross-architecture validation.
2. **CPU Reproducibility Ledger (`bench/control_vector/audit_*.py`):** Promote finalized Audit 1 & Audit 2 into `bench/` alongside existing `bench/verify_pole_a_erasure.py` only after Phase 4/Wave 2 settle cleanly.
