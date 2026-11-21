# Breaking the `1x` vs. `2x` DiT Steering Paradox: From Late-Interaction Multi-Vector Retrieval to Prompt-Centered Subspace Erasure in Joint Audio-Video Diffusion

**Date:** 2026-11-21  
**Authors:** MiniMax H3 Explorations Research Hub  
**Reproducible Code & Verifiers:** [`control_vectors.py`](../../../../control_vectors.py) · [`j_lens.py`](../../../../j_lens.py) · [`control_vector_nodes.py`](../../../../control_vector_nodes.py) · [`bench/verify_pole_a_erasure.py`](../../../../bench/verify_pole_a_erasure.py) · [`bench/control_vector/eval_multivector_erasure.py`](../../../../bench/control_vector/eval_multivector_erasure.py)  
**Empirical Benchmark Ledger:** [`bench/results/2026-11-21_multivector_erasure_benchmark.md`](../../../../bench/results/2026-11-21_multivector_erasure_benchmark.md)  
**Companion Investigation Logs & Practitioner Guides:** [`2026-11-20_unpooled_multivector_sliders.md`](2026-11-20_unpooled_multivector_sliders.md) · [`2026-11-20_multivector_retrieval_sota_to_dit_steering.md`](2026-11-20_multivector_retrieval_sota_to_dit_steering.md) · [`how_pole_a_erasure_works.md`](../../steering/how_pole_a_erasure_works.md) · [`tutorials_progressive_guide.md`](../../steering/tutorials_progressive_guide.md)

---

## Abstract

Steering single-stream multimodal Diffusion Transformers (DiTs) exposes a sharp cost–capability paradox. Single-pass (`1x`) contrastive activation vectors ($\mathbf{h}' = \mathbf{h} + s\hat{\mathbf{v}}_{\text{1D}}$) add zero DiT passes, yet consistently fail whenever a user's scene prompt already specifies opposing physical lighting, palette, or material cues (*the Prompt Veto*): across multi-seed image and `192`-frame (`8.0s`) joint audio-video evaluations on MiniMax H3, rank-1 steering turns `0/5` natural-color stills and `0/3` speaking videos black-and-white (`noir`), and deletes foreground characters when forcing sculptural materials (`paper_origami`). Conversely, dual-pass (`2x`) differential velocity guidance ($\mathbf{v}_{\text{orig}} + s(\mathbf{v}_{\text{edit}} - \mathbf{v}_{\text{orig}})$) resolves token conflicts cleanly only by doubling DiT forward passes.

We resolve this paradox by establishing a **Retrieval–Steering Duality** between October–November 2026 late-interaction / static-embedding retrieval architectures (**Khattab et al.** `ColBERTv2`/`PLAID`/`WARP`, **LightOn / Chaffin & Sourty** `PyLate`/`FastPLAID`/`LateOn`/`ColBERT-Zero`, **Clavié & Mixedbread** Ward token pooling/*Latent Terms*/`mxbai-edge-colbert-v0`, **LightOn/Faysse** `ColQwen`, **Google Research** `MUVERA`, and **Minish Lab / Tulkens & van Dongen** `Model2Vec`/`tokenlearn`/`POTION`/`reach`) and in-place conditioning state editing inside H3's `5376`-D bidirectional `TokenRefiner` space. First, we prove algebraically and empirically why naive retrieval ports fail as continuous generative sliders: appending virtual ColBERT/MUVERA keys violates `s = 0` slider identity via softmax denominator drift (`MAE = 22.7–54.5`), while doubly-stochastic Sinkhorn Optimal Transport collapses identically back to $\mathbf{v}_{\text{1D}}$ (`max_err = 8.88e-16`). Second, we prove that asymmetric right-stochastic ColBERT soft-MaxSim ($\tau_{\text{attn}} = 0.15$) coupled with rank-$r$ ridge regression decomposes (`max_err <= 5.86e-14`) into a sharpened target mean $\hat{\boldsymbol{\mu}}_d$ — rescuing target tokens washed out by 1D averaging on **`28 / 38`** bank concepts — minus **Clamped Prompt-Centered Pole-A Subspace Erasure**:

$$\Delta\mathbf{H}(s) = \operatorname{RMSMatch}_{|s|\tau}\!\Big(s\,\mathbf{1}\hat{\boldsymbol{\mu}}_d^\top - \operatorname{clip}(s, 0, 1)\,\beta_{\text{cap}}(\mathbf{H} - \mathbf{1}\bar{\mathbf{h}}^\top)\mathbf{V}_r^\top\operatorname{diag}(\mathbf{g})\mathbf{V}_r\Big)$$

Requiring only **`193.5 KB`** of `float32` tensors per concept, zero external corpus files, and **`0` extra DiT passes**, `add_norm_matched_erasure` passes a 6-arm RMS-matched (`101.25`) double-dissociation control (`5/5` `noir` B&W; `5/5` `paper_origami` character preservation), transfers across `192f`/`90f` `24 fps` video + `40 Hz` stereo audio, `4.1 MP` `P-H3-Image`, native `4.0 MP` `MiniMax-H3-Image`, and undistilled 50-step Base H3, and extends to `< 2 ms` CPU `Model2Vec` word-subspace synthesis (`MiniMaxH3WordVector`) and `ColQwen`-style visual patch pairs (`MiniMaxH3MakeMediaControlVector`).

---

## 1. Introduction: The `1x` vs. `2x` DiT Steering Paradox

Modern joint audio-video Diffusion Transformers such as **MiniMax H3** pack text tokens, `<Picture N>` / `<Video N>` reference latents, `40 Hz` stereo audio tokens, and `24 fps` 3D causal video latents into one shared self-attention sequence across 60 transformer blocks (`comfy/ldm/minimax/model.py:728-803`). Before entering those 60 blocks, every prompt passes through a 50-layer causal vision-language encoder (`Qwen3-VL-32B`, width `5120`) followed by a 2-block **bidirectional `TokenRefiner`** (width `5376`).

Practitioners seeking reusable style, medium, lighting, or acoustic dials face an uncomfortable dichotomy:

1. **`1x` Mean-Pooled Control Vectors (`add_norm_matched`):** Extracting $\mathbf{v}_{\text{1D}} = \frac{1}{P}\sum_{p=1}^P (\bar{\mathbf{B}}_p - \bar{\mathbf{A}}_p)$ from paired contrast prompts and broadcasting $\mathbf{H} + s\hat{\mathbf{v}}_{\text{1D}}$ across text tokens adds **zero DiT compute (`1x`)**. However, 1D mean-pooling discards **`51.6%–78.1%`** of per-token shift energy (`phase1_spectra.json`) and leaves every conflicting baseline token inside $\mathbf{H}$ untouched. When a prompt describes *warm tungsten kitchen lighting*, *morning TV floodlights*, or *live-action human skin*, un-erased Pole-A directions inside $\mathbf{H}$ attend directly to video/audio latents across all 60 DiT blocks and **veto** the additive vector. Cranking $s$ higher (`s = 2.0`, shift norm `135`) fails to flip global color on `noir` while triggering localized saturation collapse on `watercolor_wash`.
2. **`2x` Differential Velocity Guidance (`MiniMaxH3EditGuider`):** Running the DiT separately on `[original]` and `[edited]` token sequences and extrapolating $\mathbf{v}_{\text{orig}} + \text{scale}\cdot(\mathbf{v}_{\text{edit}} - \mathbf{v}_{\text{orig}})$ preserves exact per-token binding, but **doubles per-step DiT cost (`2x`)** and requires prompt-specific text rewrites rather than a self-contained `.safetensors` slider.

### Contributions

1. **The Retrieval–Steering Duality (§2–§4):** We formulate `1x` DiT conditioning steering as the *write-path dual* of late-interaction multi-vector retrieval, systematically translating six 2020–November 2026 retrieval paradigms into H3's `[L, 5376]` post-refiner space and proving both algebraically and empirically which mechanisms survive generative self-attention and which collapse.
2. **Exact Ridge Split & Clamped Prompt-Centered Pole-A Subspace Erasure (§5):** We prove (`max_err <= 5.86e-14` in `float64`) that regressing ColBERT soft-MaxSim token shifts onto Pole A acts **`96.9%–99.6%`** as a rank-$r$ Pole-A subspace projector, and isolate the clean, sign-clamped, prompt-centered operator `add_norm_matched_erasure` (`193.5 KB` `float32`, up to rank `16` under multi-slider composition).
3. **38-Concept Spectral Census & Subspace Explainability (§3, §5.5):** Across all `76` production `.safetensors` bank files (`38` concepts $\times$ `{fl2va, ref2va}`), we map H3's three steering regimes, verify cross-family Grassmannian subspace invariance (`median = 0.9323`), and introduce a `< 15 ms` CPU Subspace X-Ray combining a Jacobian-lens word readout adapted from Anthropic's J-lens with `WARP`-style per-token erasure norms $\|\mathbf{E}(\mathbf{h}_i)\|_2$.
4. **Static Lexical (`Model2Vec`) & Visual Patch (`ColQwen`) Extensions + 9-Part Benchmark (§5.4, §6):** We extend Pole-A subspace erasure to `< 2 ms` CPU `MiniMaxH3WordVector` synthesis (backed by a distilled `180`-term `Model2Vec` H3 lexicon) and `MiniMaxH3MakeMediaControlVector` `<Picture 1>` pairs, and validate all claims across 9 controlled ablation suites spanning Pika 6-step stills, `192f`/`90f` `24 fps` + `32 kHz` stereo AV, `4.1 MP` `P-H3-Image`, `4.0 MP` `MiniMax-H3-Image`, and undistilled 50-step Base H3.

---

## 2. Related Work & Cross-Domain Taxonomy

### 2.1 Late-Interaction Multi-Vector & Static Retrieval (2020 – November 2026)

Dense single-vector bi-encoders compress an entire sequence into one pooled vector, creating an information bottleneck formalized on hard combinatorial benchmarks such as **LIMIT**. Late-interaction and static-embedding research overcame this bottleneck through six distinct mechanisms that directly inform our work:

- **Token-Level MaxSim, Centroid-Residual Quantization, and Dynamic Routing (MIT OASYS / Khattab et al.):** **Khattab & Zaharia** (*ColBERT*, SIGIR 2020) replaced sequence pooling with asymmetric query-to-document token `MaxSim` ($\sum_i \max_j \mathbf{q}_i^\top \mathbf{d}_j$). *ColBERTv2* (Santhanam, Khattab et al., NAACL 2022) and *PLAID* (CIKM 2022) showed that multi-vector token clouds decompose cleanly into **cluster centroids plus low-bit residuals**, while *WARP* (Scheerer, Zaharia, Potts, Alonso, Khattab, SIGIR 2025 Best Paper) introduced `WARP_SELECT` dynamic similarity imputation over top-$k$ routed centroids.
- **Hierarchical Token Pooling, `PyLate`, & Zero-Supervision Late Interaction (LightOn — Antoine Chaffin, Raphaël Sourty et al. & Benjamin Clavié):** **Clavié et al.** (*Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling*, ECIR / `arXiv:2409.14683`; *LIR @ ECIR 2026* `arXiv:2511.00444`; `RAGatouille`) demonstrated that post-hoc **Ward hierarchical agglomerative clustering** over token embeddings compresses multi-vector documents by $50\%\text{–}66\%$ with $<1\%$ degradation because transformer token clouds are locally redundant, whereas spherical $k$-means collapses onto high-frequency boilerplate. **LightOn** (`PyLate` `arXiv:2508.03555`, `FastPLAID`, `NextPlaid`, `LateOn-Code`, and `LateOn-hpool-regularized` trained end-to-end via a Straight-Through Estimator `MultiVectorTokenPoolingLoss`) and *ColBERT-Zero* (**Chaffin & Sourty**, Feb 2026, `arXiv:2602.16609`) further proved that late-interaction token alignment geometry emerges from pretrained backbones without specialized contrastive supervision when anisotropy is controlled.
- **Compact Edge Multi-Vector & Sparse *Latent Terms* (Mixedbread AI — Benjamin Clavié, Sean Lee, Aamir Shakir, Makoto P. Kato):** Mixedbread's `mxbai-edge-colbert-v0-17m/32m` (Oct–Nov 2025) proved low-rank (`48`-D) token subspaces suffice for `ColBERTv2`-class discrimination, while *Latent Terms* (**Clavié et al.**, June 2026, `arXiv:2605.29384`, *"Dense Retrievers Know More Than They Can Express"*) extracted Zipfian sparse non-negative dictionary atoms competitive on BEIR and LIMIT without extra training.
- **Visual Patch Late Interaction (`ColPali` / `ColQwen` — Faysse, Sibille, Wu, Omrani, Viaud, Hudelot, Colombo, LightOn / CentraleSupélec, ICLR 2025):** Replaced OCR pipelines by keeping Vision-Language Model (`PaliGemma` / `Qwen2-VL` M-RoPE) image patch rows unpooled during late interaction — the exact representation H3 uses for `<Picture N>` tokens (`tags == 0`).
- **Fixed Dimensional Encodings (`MUVERA` — Dhulipala, Hadian, Jayaram, Lee, Mirrokni, Google Research, NeurIPS 2024):** Compressed variable-size multi-vector bags into single fixed-dimensional encodings via SimHash partitions for asymmetric inner-product search.
- **PCA De-Coning, SIF Weighting, and Contextual Static Distillation (Minish Lab — Stéphan Tulkens & Thomas van Dongen, 2024–2026):** Across `Model2Vec`, `tokenlearn`, `POTION` (`8M`/`32M`), `reach`, `SemHash`, and `StaRSE`, Tulkens & van Dongen showed that (i) removing top principal components (*All-But-The-Top*, Mu & Viswanath 2018) collapses transformer representation cones so cosine geometry reflects semantics rather than token frequency, (ii) Smooth Inverse Frequency (SIF, Arora et al. 2017) prevents stopwords from dominating pooled sums, and (iii) forward passes through a large contextual encoder can be distilled into ultra-fast (`< 1 ms` CPU) static multi-token word/phrase tables.

### 2.2 Concept Erasure & Activation Steering in LLMs and Diffusion Models

Linear concept erasure in classifiers and language models progressed from iterative nullspace removal (*INLP*, Ravfogel et al. 2020) and minimax game projections (*RLACE*, Ravfogel et al. 2022) to closed-form covariance oblique projections (*LEACE*, Belrose et al., NeurIPS 2023). In text-to-image diffusion, cross-attention weight editing (*UCE*, Gandikota et al. WACV 2024; *RECE*, Gong et al. ECCV 2024) and Flux/MMDiT attention-map interventions (*EraseAnything*, Gao et al. ICML 2025; *MUTE*, `arXiv:2609.36832`, Sep 2026; *RASteer*, `arXiv:2610.01969`, Oct 2026; *Semantic Steering*, Aug 2026) target **safety unlearning** (permanently deleting nudity, celebrities, or copyrighted styles via LoRA weight updates, test-time optimization, or auxiliary SAE bottlenecks).

**How our setting and mechanism differ:** Rather than permanently unlearning a forbidden concept from model weights, we solve **positive creative conflict resolution** for *any* continuous `1x` concept slider inside a **joint audio-video single-stream DiT** (`24 fps` video + `40 Hz` stereo audio). Our operator modifies neither DiT weights nor sequence length, requires zero SAE training or external text corpora, and derives both its sharpened target direction ($\boldsymbol{\mu}_d$) and its opposing baseline subspace ($\mathbf{V}_r$) directly from the unpooled token clouds of 8–14 paired prompts.

---

## 3. Geometry of H3's `5376`-D Post-Refiner Space & 38-Concept Bank Census

### 3.1 Two-Stage Conditioning Geometry, Anisotropy Cone, and `int8_convrot` Invariance

Inspection of all four shipped H3 checkpoints (`minimax_h3_fl2va_pruned_int8_convrot`, `minimax_h3_ref2va_pruned_int8_convrot`, `p-h3-image_int8_convrot`, and `minimax_h3_image_pruned_int8_convrot`) establishes four exact structural invariants:

1. **Causal Encoder Sink vs. Bidirectional `TokenRefiner` Cone:** In causal `Qwen3-VL-32B` (`layers.0..49`, width `5120`), token `0` acts as a massive attention sink ($\|\mathbf{e}_0\|_2 \approx 15,309$ vs. median `292.3`, a **`52.4x`** ratio). Inside each DiT, `condition_proj` + the 2-block **bidirectional `TokenRefiner`** (`width = 5376`) dampens row `0` to `2.27x` median (`~164–270`), **but mixes global sentence context into every token row**, placing raw post-refiner tokens inside a tight anisotropic cone (`mean pairwise cosine = 0.6088` across `covered_market` scene tokens; `0.609–0.744` within Pole-B banks). Fitting 2 principal components strictly on held-out Pole-A training banks drops scene token anisotropy **`14.5x` (`0.6088 -> 0.0419`)**.
2. **Unquantized `BF16` Refiner & Self-Cancelling Hadamard DiT Linears:** All `19` `condition_proj.*` and `token_refiner.*` weight tensors across all 4 checkpoints are unquantized `BF16`. Across the 200 quantized DiT linear layers (`blocks.0..49`), `int8_convrot` applies an online block-diagonal Hadamard transform $\mathbf{x}\mathbf{H}_{256}(\mathbf{W}\mathbf{H}_{256}^\top)^\top = \mathbf{x}\mathbf{W}^\top$ that cancels identically per linear call, leaving `5376`-D residual coordinates unrotated.
3. **`PackedLayout` 3D RoPE Cursor Invariance:** H3 packs `[text | refs | audio | video]` into one sequence whose non-text 3D RoPE temporal cursor begins at `text_len`. Any operator that edits `context[:, mask, :]` **in place** preserves `text_len` and all reference/audio/video 3D RoPE phases bit-for-bit, guaranteeing exact `0.0000` slider identity at $s = 0$.
4. **Cross-Family Subspace Invariance (`fl2va` $\leftrightarrow$ `ref2va` vs. `h3_image`):** Across all `38` concepts (`76` `.safetensors` files in `models/control_vectors/bank/`, `data/exp_unpooled_sliders/paper_2026_11_21/census_38_bank.json`), `fl2va` and `ref2va` post-refiner vectors exhibit a median 1D cosine of **`0.9892`** and a median rank-8 `erasure.vr` Grassmannian subspace overlap $\frac{1}{8}\|\mathbf{V}_{r,\text{fl2va}}\mathbf{V}_{r,\text{ref2va}}^\top\|_F^2$ of **`0.9323`** (enabling zero-rebuild cross-family transfer onto `<Picture 1>` `ref2va`), whereas `MiniMax-H3-Image` (`h3_image`) carries genuinely rotated text-refiner weights (`1D cos ≈ 0.214` on contrasts) and requires native `--families h3_image` builds.

### 3.2 Spectral Census of the 38-Concept Production Bank

Evaluating all `38` paired concepts (`8–14` pairs each, `diff_span` pooling) partitions post-refiner steering concepts into three geometric regimes (`census_38_bank.json` + `phase1_spectra.json`):

| Regime (Count / 38) | Defining Spectral Signature | Representative Concepts (`mpc` / `1D_E%` / `cos(mu_d, v_1d)`) | `1x` 1D vs. Subspace Erasure Behavior |
| :--- | :--- | :--- | :--- |
| **1. Multi-Facet Bundles (`10 / 38`)** | `mpc < 0.62` or `1D_E% < 0.30` (high Pole-A diversity: `PR_8 = 6.1–7.2`) | `infomercial_fail` (`0.433` / — / `0.994`), `wes_anderson_symmetry` (`0.453` / `22.6%` / `0.995`), `noir` (`0.574` / `21.9%` / **`0.970`**), `balloon_head_inflate` (`0.582`), `zero_gravity` (`0.590`), `chaos_panic` (`0.711` / `29.6%` / `0.992`) | **1D hits the Prompt Veto** whenever Pole A spans diverse natural lighting/color states (`noir`: `0/5` B&W in 1D vs. **`5/5` B&W** under `add_norm_matched_erasure`). |
| **2. Intermediate Structural / Acoustic (`21 / 38`)** | `0.62 <= mpc < 0.72` (`30% <= 1D_E% < 42%`) | `wind_tunnel_gale` (`0.628`), `cursed_cctv` (`0.630`), `underwater_submerged` (`0.645`), `whisper_asmr` (`0.650`), `paper_origami` (`0.699` / `36.3%` / `0.998`), `claymation_stopmotion` (`0.711`), `felt_muppet` (`0.715`) | 1D pushes surface texture but collides with live-action anatomy (`paper_origami` 1D deletes foreground characters or collapses aisles into tunnels; rank-8 `V_r` restores `5/5` characters). |
| **3. Monolithic Overrides (`7 / 38`)** | `mpc >= 0.72` or `1D_E% >= 0.42` | `molten_lava_golem` (`0.709` / `44.8%`), `operatic_bellow` (`0.750`), `reverb_cathedral` (`0.772`), `vhs_1991_camcorder` (`0.790`), `rubber_hose_1930s` (`0.804`), `film_16mm_1974` (`0.807` / — / **`0.953`**), `film_16mm_1974_neutral` (`0.807` / — / **`0.948`**) | Coherent 1D direction already works well, while `erasure.mu_d` sharpens fine lexical grain (`film_16mm_1974` rescues `grain, tape`; `reverb_cathedral` rescues `majestic, mystical, ancient`). |

Across the entire bank, **`28 / 38` concepts (`73.7%`)** rescue at least one domain-defining target word into `erasure.mu_d`'s top-8 word readout that washed out of `v_1d` (`noir` rescues `black, footage, archival`; `claymation_stopmotion` rescues `puppet`; `whisper_asmr` rescues `whisper`; `cursed_cctv` rescues `grey, overhead, viewed`).

---

## 4. The Retrieval–Steering Duality: Why Direct Retrieval Ports Break (and What Survives)

In Information Retrieval, a query $\mathbf{Q} \in \mathbb{R}^{M \times D}$ scores a document $\mathbf{B} \in \mathbb{R}^{N_B \times D}$ **read-only** ($s(\mathbf{Q},\mathbf{B}) = \sum_{i=1}^M \max_j \mathbf{q}_i^\top \mathbf{b}_j$). In DiT conditioning steering, we must construct an **in-place state map** $\mathcal{T}_s: \mathbb{R}^{M \times D} \to \mathbb{R}^{M \times D}$ satisfying (i) exact neutrality $\mathcal{T}_0(\mathbf{H}) \equiv \mathbf{H}$, (ii) sequence-length invariance ($L' = L$), and (iii) bounded token shift RMS so 60 downstream joint-attention layers do not saturate.

Testing five direct translations of 2024–2026 retrieval architectures exposes three fundamental failure theorems and two viable secondary operators:

### Proposition 1 (Softmax Denominator Drift of Appended ColBERT / MUVERA Token Bags)
Appending $K = 12$ unpooled concept centroids or `MUVERA` Fixed Dimensional Encodings ($\tilde{\mathbf{H}}(s) = [\mathbf{H};\; \mathbf{C}_0 + s\boldsymbol{\Delta}_K] \in \mathbb{R}^{(L+K)\times D}$) with H3's 3D RoPE cursor frozen at $L$ violates slider neutrality at $s = 0$: for every visual/audio query $\mathbf{q}$, the attention partition function changes from $Z_L = \sum_{\ell=1}^L e^{\mathbf{q}^\top \mathbf{k}_\ell / \sqrt{d}}$ to $Z_L + \sum_{c=1}^K e^{\mathbf{q}^\top \mathbf{k}_{L+c} / \sqrt{d}}$, shifting unsteered renders by **`MAE = 22.7–54.5` pixel levels at `s = 0.0`**.

### Proposition 2 (Mean-Shift Collapse of Doubly-Stochastic Sinkhorn Optimal Transport)
Suppose within each contrast pair $(\mathbf{A}_p \in \mathbb{R}^{n_A \times D}, \mathbf{B}_p \in \mathbb{R}^{n_B \times D})$, tokens are matched via entropic Optimal Transport (Sinkhorn-Knopp) enforcing row sums $\mathbf{T}_p \mathbf{1}_{n_B} = \mathbf{1}_{n_A}$ **and** uniform column sums $\mathbf{T}_p^\top \mathbf{1}_{n_A} = \frac{n_A}{n_B}\mathbf{1}_{n_B}$. Then the mean matched token shift collapses **identically** to the 1D mean difference for every regularization $\varepsilon > 0$:

$$\frac{1}{n_A}\mathbf{1}_{n_A}^\top (\mathbf{T}_p \mathbf{B}_p - \mathbf{A}_p) = \frac{1}{n_A}\left(\frac{n_A}{n_B}\mathbf{1}_{n_B}^\top \mathbf{B}_p\right) - \bar{\mathbf{A}}_p \equiv \bar{\mathbf{B}}_p - \bar{\mathbf{A}}_p = \mathbf{v}_{\text{1D},p}$$

Verified in `float64` (`bench/verify_pole_a_erasure.py`): Balanced Sinkhorn OT matches `v_1d` down to **`max_err = 8.882e-16`** (`cos = 1.000000`), immediately explaining why replacing asymmetric ColBERT row-softmax with Sinkhorn OT regressed `noir` from `3/3` B&W down to `1/3`. **Breaking column conservation (right-stochastic row-softmax) is algebraically mandatory** so multiple Pole-A tokens can route to high-salience Pole-B tokens (`"black-and-white"`, `"monochrome"`, `"silver-halide"`) while ignoring Pole-B syntax filler.

### Proposition 3 (Sample-Floor Fragmentation of Sparse *Latent Terms*)
Decomposing `diff_span` banks into $K = 16$ sparse non-negative dictionary atoms (*Latent Terms*, `arXiv:2605.29384`, top-3 ReLU routing `M6`) produces crisp localized geometry when $N_A \gg K$ (`paper_origami`, $N_A = 126$), but starves diffuse global medium overrides whenever $N_A / K < 2$ (`watercolor_wash`, $N_A = 22 \implies 1.38$ tokens/atom, leaving renders photorealistic) and fails to erase dense color cones (`noir` stays color).

### What Transfers Cleanly as Secondary Operators (`Clean SIF` & `Ward + WARP_SELECT`)
1. **Minish Lab `-2PC` + Clean SIF (`A2_1d_clean_sif`):** Weighting tokens by angular distance from a held-out Pole-A common-mode mean while clamping non-alphanumeric punctuation/whitespace tokens (`'.'`, `' '`, which otherwise sit at $1.88\times\text{–}2.08\times$ mean distance) to `0.5` acts as a pure 1D salience booster (flipping `watercolor_wash` at `83.2` without multi-vector matrices, though still `0/5` on `noir`).
2. **LightOn/Clavié Ward (`K = 8`) + Khattab `WARP_SELECT` (`M5` / `A6` / `A7`):** Replacing spherical $k$-means (`row_div = 0.004` collapse) with Ward clustering on `-2PC` whitened Pole-A tokens plus top-2 `WARP_SELECT` centroid+residual routing ranks **#2 overall** when combined with Clean SIF (`A7`: `noir` `88.4` B&W, `watercolor` `84.1`), but requires external corpus PCA statistics and softer entity preservation than direct Pole-A subspace erasure.

---

## 5. Clamped Prompt-Centered Pole-A Subspace Erasure

### 5.1 Exact Ridge Split Theorem (`float64`)

Let $\mathbf{A} \in \mathbb{R}^{N_A \times D}$ be stacked Pole-A `diff_span` rows across $P$ training pairs and $\mathbf{B}_m \in \mathbb{R}^{N_A \times D}$ be their within-pair right-stochastic ColBERT soft-MaxSim matches:

$$\mathbf{P}_p = \operatorname{softmax}\!\left(\frac{\hat{\mathbf{A}}_p \hat{\mathbf{B}}_p^\top}{\tau_{\text{attn}}},\;\text{dim}=-1\right), \qquad \mathbf{D}_p = \mathbf{P}_p \mathbf{B}_p - \mathbf{A}_p, \qquad \boldsymbol{\mu}_d = \frac{\sum_p w_p \sum_i \mathbf{D}_{p,i}}{\sum_p w_p n_{A,p}}$$

Let $\mathbf{A}_c = \mathbf{A} - \mathbf{1}\boldsymbol{\mu}_A^\top = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ be the centered thin SVD truncated to rank $r$ (`r = 8`), with ridge parameter $\lambda = \lambda_{\text{rel}}\sigma_1^2$ ($\lambda_{\text{rel}} = 0.15$). Fitting a rank-$r$ linear ridge transport $\mathbf{W}_D = \mathbf{V}_r^\top \operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)\mathbf{U}_r^\top (\mathbf{B}_c - \mathbf{A}_c)$ yields the **exact algebraic identity**:

$$(\mathbf{h} - \boldsymbol{\mu}_A)\mathbf{W}_D \;\equiv\; \underbrace{(\mathbf{h} - \boldsymbol{\mu}_A)\mathbf{V}_r^\top \operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)\mathbf{U}_r^\top \mathbf{B}_c}_{\text{Pole-B conditional transport }\mathbf{W}_B} \;-\; \underbrace{(\mathbf{h} - \boldsymbol{\mu}_A)\mathbf{V}_r^\top \operatorname{diag}\!\left(\frac{\sigma_i^2}{\sigma_i^2+\lambda}\right)\mathbf{V}_r}_{\text{rank-}r\text{ Pole-A ridge projector }\mathbf{P}_{\mathbf{V}_r,\lambda}}$$

Across all `7` saved post-refiner `.pt` banks (`bench/verify_pole_a_erasure.py`), this split holds to **`max_err <= 5.862e-14`**, with $\|\text{ctx}_{-\mathbf{A}}\|_2$ dominating $\|\text{ctx}_{\mathbf{B}}\|_2$ by **`4.2x–10.3x`** and $\cos(\text{ctx}_D, -\mathbf{A}) \in [+0.9686, +0.9960]$. Testing pure Pole-B transport $\mathbf{W}_B$ alone (`A5_pure_poleB`) failed completely on `noir` (`MAE 58.0`, bright color), proving **Pole-A subspace erasure $-\mathbf{P}_{\mathbf{V}_r,\lambda}$ is the active causal mechanism**.

### 5.2 Why Prompt-Local Centering and `[0, 1]` Polarity Clamping Are Essential

Directly applying $\boldsymbol{\mu}_d - s(\mathbf{h} - \boldsymbol{\mu}_A)\mathbf{P}_{\mathbf{V}_r,\lambda}$ fails in two regimes:
1. **Unseen-Prompt Mean Drift ($N_A$ small):** Centering on $\boldsymbol{\mu}_A$ subtracts $(\bar{\mathbf{h}} - \boldsymbol{\mu}_A)\mathbf{P}_{\mathbf{V}_r,\lambda}$ from every token in an unseen prompt (`71%` of the contextual term on 5-pair `watercolor_wash`), erasing background scenery. Centering on the **prompt's own active batch mean** $\bar{\mathbf{h}} = \frac{1}{M}\sum_{i=1}^M \mathbf{H}_i$ (`M > 1`) isolates *within-prompt* conflicting variation while leaving shared scaffolding intact.
2. **Over-Erasure Sign Inversion ($s > 1$) & Reverse Polarity ($s < 0$):** Multiplying $-\mathbf{P}_{\mathbf{V}_r,\lambda}$ by $s = 1.5$ **inverts** erased Pole-A coordinates to $-0.5\times$ their original sign (melting faces/geometry), while $s < 0$ amplifies Pole-A variance. Clamping the erasure coefficient via $\operatorname{clip}(s_j, 0, 1)$ and $\beta_{\text{cap}} = \min(1.0, \tau / \operatorname{RMS}(\mathbf{E}(\mathbf{H})))$ ($\tau = 67.5$), scrubbing `hold_vector`/`hold_words` per-row (`_apply_holds_to_rows64`), and RMS-matching the combined shift to $|s_j|\tau$ guarantees exact $s = 0$ identity, exact 1D fallback for $s \le 0$, and artifact-free extrapolation across $s \in [1.0, 2.0]$.

### 5.3 Multi-Slider Composition (`COMPOSED_ERASURE_MAX_RANK = 16` vs. Chained Nodes)

When composing $K$ vectors offline (`control_vectors.py::compose_erasure_payload`), forming the Gram-weighted union $\mathbf{M} = \sum_k \alpha_k \mathbf{V}_{r,k}^\top \operatorname{diag}(\mathbf{g}_k)\mathbf{V}_{r,k}$ and retaining up to **`COMPOSED_ERASURE_MAX_RANK = 16`** eigenvectors preserves the full rank-16 union spectrum of two orthogonal rank-8 concepts (whereas clipping at `8` discarded `diag[8..15]`). Crucially, **single-file composition (`sculpt`) shares one `|s| * 67.5` RMS budget** across all blended components, whereas **chaining separate `MiniMaxH3Steer` nodes** evaluates each node's full $|s_k| \cdot 67.5$ shift against unsteered `base_rows64` before summing in `float64` (`control_vectors.py:3454-3492`), giving chained nodes stronger override authority when stacking heavy visual + kinetic + acoustic sliders (`noir + chaos_panic + reverb_cathedral`).

### 5.4 Extending Erasure to `< 2 ms` `Model2Vec` Word Vectors & `ColQwen` Visual Media Pairs

- **Static `Model2Vec` Lexicon + `against`-SVD (`MiniMaxH3WordVector`, `j_lens.py`):** Following **Tulkens & van Dongen** (`Model2Vec` / `tokenlearn`), `bench/control_vector/build_model2vec_lexicon.py` distills `180` multi-token BPE words and 2-word film/material/audio phrases (`origami`, `muppet`, `chiaroscuro`, `plasticine`, `halation`, `silver gelatin`, `paper origami`, `direct flash`) across non-market cloze templates through `Qwen3-VL-32B` + all 4 `TokenRefiner` recipes (`fl2va`, `ref2va`, `fl2va_base`, `h3_image`) with top-2 common-mode PC removal (`model2vec_h3_lexicon.safetensors`, plus BPE subword SIF fallback). When `against_words` (`@replace_digital, sunlit, warm` or $\ge 2$ opposing terms) are supplied, `synthesize_from_words` projects each individual `against` word vector into `5376`-D space and computes thin SVD `erasure.{mu_d,vr,diag}` ($r \le \min(8, K_{\text{neg}}-1)$) on CPU in **`< 2 ms`**.
- **`ColQwen`-Style Visual Patch Erasure (`MiniMaxH3MakeMediaControlVector`, `control_vector_nodes.py`):** Given paired reference stills (`image_a` Pole A vs. `image_b` Pole B) encoded as M-RoPE `<Picture 1>` sequences, `make_control_vector_from_media_pairs` retains the unpooled `[L_p, 5376]` post-refiner token matrices per pair and extracts rank-8 `erasure.{mu_d,vr,diag}` directly via `build_pole_a_erasure`.

### 5.5 Subspace X-Ray & `WARP`-Style Per-Token Selectivity (`j_lens.py`, `grading.py`)

Applying H3's precomputed Jacobian-lens word readout (adapted from Anthropic's J-lens, $\mathbf{J}_{\text{Wiener}}^{(f)}$) to `erasure.mu_d` and `erasure.vr[0..3]`, alongside per-token erasure norms $\|\mathbf{E}(\mathbf{h}_i)\|_2 = \|(\mathbf{h}_i - \bar{\mathbf{h}})\mathbf{V}_r^\top \operatorname{diag}(\mathbf{g})\mathbf{V}_r\|_2$ (`per_token_erasure_profile`), makes every `.safetensors` slider auditable in `< 15 ms` on CPU. On `noir` paired with `t2va_sitcom_kitchen.txt`, `erasure.mu_d` rescues `black (+1.52)` while `erasure.vr` concentrates subtraction on `wall (0.41)`, `warm (0.36)`, `brown (0.33)`, and `tungsten (0.33)`, sparing `father`, `daughter`, and `retriever`.

---

## 6. Multi-Modal & Cross-Architecture Experiments

All single-frame Pika ablations below run on matched non-market scene prompts (`indoor_neutral`, `t2va_sitcom_kitchen`, `t2va_hardware_aisle_short`, `ref2va_scene_kitchen`) at `1152x768`, `6`-step `lcm`, `MiniMaxH3LoRABranch` (`pika_h3_flex_fl2va_svd995_bf16`, `strength = 1.0`), `MiniMaxH3SigmaShift(12.0, 3.0)`, `comfy kitchen attention`, `PH3_IMAGE_VAE` (`bf16`), and **strictly matched shift RMS = `101.25` at `s = 1.5`** (`data/exp_unpooled_sliders/paper_2026_11_21/*.json`, `$H3_OUTPUT_DIR/{Image,Video}/paper_erasure_2026-11-21/`).

### 6.1 Double Dissociation Specificity Proof (`S0`–`S5`)

Before varying hyperparameters, our 6-arm specificity control (`metrics_specificity.json`, all arms locked to `total_rms = 101.25` and identical `erase_rms`) rules out generic variance dampening:
- **`noir` (`indoor_neutral`, 5 seeds `[730451892, 101, 2026, 424242, 99991]`):** `S1_own_Vr` flips **`5/5` seeds** to crisp silver-gelatin B&W (`hsv_sat = 0.017–0.033`), whereas `S0_1d` (`0/5`, `0.166–0.197`), `S2_swapped_Vr` (`paper_origami`'s $\mathbf{V}_r$), `S3_random_Vr` (random orthonormal rank-8), and `S5_prompt_top8` (prompt's own top-8 PCs) **all remain bright color**.
- **`paper_origami` (`5` seeds):** `S1_own_Vr` preserves a folded-paper scene + foreground character across **`5/5` seeds**, whereas `S2_swapped_Vr` (`noir`'s $\mathbf{V}_r$), `S3`, `S5`, and blunt `0`-byte isotropic shrinkage `S4_iso_shrink` (`0/5`) destroy foreground characters into empty geometric corridors.

### 6.2 Ablation 1 — Soft-MaxSim Temperature $\tau_{\text{attn}}$ & Why De-Coning Hurts Pair Alignment

Varying $\tau_{\text{attn}} \in \{0.03, 0.07, 0.15, 0.50, \infty\}$ with fixed rank-8 `erasure.vr` across 3 seeds (`[730451892, 101, 2026]`, `ablation1_tau_sweep_noir.png`):

| Arm ($\tau_{\text{attn}}$) | `cos(mu_d, v_1d)` | Rescued Words in `mu_d` (`noir`) | Seed `730451892` `hsv_sat` | Seed `101` `hsv_sat` | Seed `2026` `hsv_sat` | B&W Pass Rate (`< 0.06`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **`1D` ($\tau = \infty$, no $\mathbf{V}_r$)** | `1.0000` | `[]` | `0.1896` | `0.1592` | `0.2260` | **`0 / 3`** |
| **$\tau = 0.50$ (soft)** | `0.995` | `['black', 'grayscale']` | **`0.0251`** | **`0.0223`** | **`0.0303`** | **`3 / 3`** |
| **$\tau = 0.15$ (shipped default)** | `0.9700` | `['black', 'footage', 'archival']` | **`0.0264`** | **`0.0223`** | **`0.0223`** | **`3 / 3`** |
| **$\tau = 0.07$ (sharp)** | `0.935` | `['black', 'footage', 'silent']` | **`0.0260`** | **`0.0236`** | **`0.0261`** | **`3 / 3`** |
| **Hard `MaxSim` ($\tau = 0.03$)** | `0.8609` | `['black', 'sharply', 'sharp']` | `0.1992` | **`0.0245`** | `0.1846` | **`1 / 3`** |
| **De-coned ($\tau = 0.15$, pair-mean centered)** | `0.9823` | `['primitive']` *(drops `'black'`!)* | `0.1702` | **`0.0211`** | `0.1943` | **`1 / 3`** |

**Two structural insights emerge:**
1. **A wide Goldilocks plateau ($\tau_{\text{attn}} \in [0.07, 0.50]$) hits `3/3` B&W**, whereas taking the hard `ColBERT` $\tau \to 0$ argmax limit (`0.03`) regresses to `1/3` (`cos = 0.8609`) due to noisy single-token nearest-neighbor snapping.
2. **De-coning helps *clustering* (`Ward`), NOT within-pair soft alignment:** Subtracting pair means before computing cosine cross-attention (`tau=0.15 deconed`) strips shared chromatic axes so `"black"` falls out of `mu_d`'s top readout (`rescued = ['primitive']`), regressing `noir` to `1/3` B&W (`0.1702, 0.0211, 0.1943`) and twisting `watercolor_wash` (`cos = 0.8637`, hallucinating `'rainbow, mural, rain'`).

### 6.3 Ablation 2 — Pole-A Subspace Construction (`svd_ac` vs. ColBERT Skiplist vs. Contested SVD)

Inspection of `noir`'s unsupervised `svd_ac` axes showed `vr[0]` (`diag = 0.87`) loading onto framing/adverbial tokens (`possibly, likely, elsewhere, stuff`) while `vr[1]` (`diag = 0.81`) captured `warm, natural, lighting`. We tested whether filtering stopwords/framing tokens (**ColBERT Skiplist**, `skiplist_svd`, Grassmann overlap `0.657` with `svd_ac` on `noir`) or weighting Pole-A rows by normalized shift magnitude $\|\mathbf{d}_i\|_2$ (**Contested SVD**, `contested_svd`, overlap `0.600`) improves visual performance (`ablation2_vr_variants_fl2va.png`):

| Concept (Non-Market Scene) | Metric / Visual Check | `1D (r=0)` | `svd_ac (r=8)` *(shipped)* | `skiplist_svd (r=8)` | `contested_svd (r=8)` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`noir`** (`indoor_neutral`, `s=730451892` / `s=101`) | `hsv_sat` (`B&W < 0.06`) | `0.1896` / `0.1592` *(0/2)* | **`0.0264` / `0.0223` *(2/2)*** | `0.1187` *(wood leak!)* / **`0.0239`** *(1/2)* | **`0.0233` / `0.0244` *(2/2)*** |
| **`paper_origami`** (`t2va_sitcom_kitchen`, 2 seeds) | `hsv_sat` & visual anatomy | `0.6371` / `0.5904` *(deletes daughter!)* | **`0.5706` / `0.5687` *(2/2 father+daughter+dog origami!)*** | **`0.5418` / `0.5983` *(2/2 preserved)*** | **`0.5826` / `0.5317` *(2/2 preserved)*** |

**Why `skiplist_svd` leaked warm wood (`0.1187`) on `noir`:** Unlike BERT encoders in retrieval where stopwords carry little document state, H3's **bidirectional `TokenRefiner`** mixes room-lighting context across functional/framing tokens (`shot, filmed, with`). Excluding them leaves residual warm-lighting components un-erased! Meanwhile `svd_ac` and `contested_svd` tie at `2/2` crisp B&W on `noir` and `2/2` full multi-character preservation on `paper_origami` (restoring the teenage daughter, her horn-rimmed glasses, toast, golden retriever, and `"SUNBEAM"` clock that `1D` completely erased), validating `svd_ac` as the zero-churn production default.

### 6.4 Ablations 3, 4 & 5 — Rank/Ridge Spectrum, Pole-A Coverage Law, & Early Step Lock-In

1. **Rank $r$ & Ridge $\lambda_{\text{rel}}$ (`ablation3_clean_rank_and_ridge.png`):**
   - On `noir`, even **`r = 1`** flips `indoor_neutral` to B&W (`hsv_sat = 0.0258` vs. `r = 0` `0.1896`), staying rock-solid across `r = 2..16` (`0.0245–0.0287`), whereas over-regularizing ridge to $\lambda_{\text{rel}} = 1.50$ shrinks `diag` below threshold and leaks color (`0.1643`).
   - On `paper_origami` (`sitcom_kitchen`), `r = 0` deletes the daughter; `r = 1..4` restores her seat at the table with live-action human skin; and **`r = 8..16` folds both father and daughter into crisp faceted geometric paper** (`0.5706 -> 0.5395`), proving multi-entity material overrides genuinely use dimensions `5..8`!
2. **The Pole-A Coverage Law (`cpu_math` `M3` + `ablation4_pair_count_coverage.png`):**
   Monte-Carlo subsampling across $N \in \{1, 2, 3, 5, 8, N_{\max}\}$ pairs reveals a fundamental sample-complexity gap between 1D directions and rank-8 Pole-A subspaces:
   - On `noir`, $\cos(\boldsymbol{\mu}_d^{(N)}, \boldsymbol{\mu}_d^{(12)})$ already reaches `0.889` by $N = 2$, **yet Grassmannian `V_r` overlap grows slowly:** `0.406` ($N=1$) $\to$ `0.521` ($N=2$) $\to$ `0.686` ($N=5$) $\to$ `0.805` ($N=8$) $\to$ `1.000` ($N=12$).
   - Rendered `noir` saturation tracks Grassmannian coverage monotonically: `0.1654` ($N=1$, warm color) $\to$ `0.1404` ($N=2$) $\to$ `0.1210` ($N=5$, desaturated sepia) $\to$ **`0.0264` ($N=12$, pure silver-gelatin B&W)**! Likewise on `paper_origami`, $N = 1..2$ leaves live-action humans while $N = 5..10$ (`Grassmann = 0.745 -> 1.000`) snaps both characters into folded paper. **Design law: `N >= 8..12` diverse Pole-A baselines are mandatory to span a rank-8 erasure subspace.**
3. **Denoising Step Schedule (`ablation5_step_window_decoupling.png`):**
   Under `MiniMaxH3SigmaShift(shift=12.0)`, Pika's first two steps (`[0, 0.34]`, $b \in [0.50, 1.0]$) commit global scene palette: `Erase [0..0.50] + mu_d [0..1]` (`noir` `hsv_sat = 0.0281`, `red_excess = +0.35`) matches full `[0..1]` (`0.0264`, `+0.16`), whereas delaying erasure to **`Late Erase [0.34..1]` fails completely (`hsv_sat = 0.2974`, `red_excess = +20.40`)**.

### 6.5 Ablations 6 & 7 — Static Lexical (`Model2Vec`) vs. Contextual Subspaces & `ColQwen` Media Pairs

- **Ablation 6 (`MiniMaxH3WordVector`, `ablation6_wordvector_model2vec_erasure.png`):** Upgrading `MiniMaxH3WordVector` with `model2vec_h3_lexicon.safetensors` + `< 2 ms` `against`-SVD erasure resolves multi-token concepts (`origami`, `chiaroscuro`, `silver gelatin`) cleanly on CPU and shifts tabletop props/lighting mood, **but stays warm color on `noir` (`hsv_sat = 0.3234` vs. 10-pair GPU `0.0264`) and keeps human faces on `paper_origami`**. Why? Isolated lexical vectors (`cos = 0.35–0.52` with 50-layer contextual `diff_span`) span *word-embedding* variance rather than full-sentence bidirectional `TokenRefiner` room-lighting variance — establishing a clean division of labor between `< 2 ms` `WordVector` sketching and 8–12 pair `.toml` builds.
- **Ablation 7 (`MiniMaxH3MakeMediaControlVector`, `ablation7_makemedia_colqwen_erasure.png`):** Extracting rank-8 patch/token `erasure.{mu_d,vr,diag}` (`diag = [0.870, 0.686, ..., 0.520]`) from a single `<Picture 1>` `image_a` vs. `image_b` pair on `ref2va_scene_kitchen` cools warm tungsten highlights smoothly (`hsv_sat = 0.3446 / 0.3435`) within the exact $N = 1$ (`~40%` Grassmannian) single-pair subspace ceiling predicted by §6.4.

### 6.6 Ablations 8a, 8b & 8c — Clean `192f` AV Retrieval Baselines, `40 Hz` Audio-Stream Erasure, & Stratified Bank Concepts

1. **Ablation 8a — Clean Qwen-BPE `pieces` `192f` AV Verification (`t2va_hardware_aisle_short.txt`, `268` scene rows, `31` non-alphanumeric punctuation/whitespace tokens clamped, `gpu_audio_video_ablations.json`):**
   Earlier `192f` evaluations of `Clean SIF` and `M5 (Ward + WARP_SELECT)` fell back to placeholder token strings (`['tok'] * L`). Re-running `192f` (`8.00s`, 3 shots, 2 speakers + bolt/paper-bag foley) with exact Qwen3-VL BPE pieces settles the retrieval-operator ranking cleanly:
   - **`A2_1d_clean_sif_fixed_pieces` (`1D * Clean SIF`):** **`sat_mean = 0.2393` (`sat_std = 0.0434`)** — stays full color across all 3 shots (blue dress shirt, pink skin, red `"HARDWARE"` pegboard signs, brown bin dividers), proving even properly de-punctuated SIF token weighting cannot overcome the `1x` Prompt Veto without subspace erasure.
   - **`M5_ward_warp_clean_sif_fixed_pieces` (Minish `-2PC` + LightOn/Clavié Ward $K=8$ + Khattab `WARP_SELECT` + Clean SIF):** **`sat_mean = 0.1341` (`sat_std = 0.0100`, `77.7s`)** — flips all 3 shots to crisp monochrome with verbatim lip-synced dialogue.
   - **`S1_pole_a_erasure` (`add_norm_matched_erasure`):** **`sat_mean = 0.1362` (`sat_std = 0.0093`, `78.2s`)** — matches `M5 + Clean SIF` to within `0.002` saturation and lower temporal flicker across all 3 shots (`"Quarter inch, or the next size up?"` / `"The next size up, these'll strip on ya."` / `"Both boxes then, I will come back for the rest."`), with **zero external corpus PCA** and **zero SIF inflation**.
2. **Ablation 8b — `40 Hz` Stereo Audio-Stream Erasure (`90f` `t2va_sitcom_kitchen.txt`, `gpu_audio_video_ablations.json`):**
   - **`reverb_cathedral` (`target = "soundscape_only"`):** Both `1D` (`sat_mean = 0.3857`) and `add_norm_matched_erasure` (`0.3972`) leave visual kitchen colors untouched (`base = 0.4108`). In audio, 1D over-damps upper harmonics (`centroid = 1912.4 Hz`, `tail_sustain = 0.2631`), whereas `add_norm_matched_erasure` erases dry close-mic room tokens to preserve crisp speech intelligibility alongside longer high-frequency cathedral decay (`centroid = 2409.4 Hz`, **`tail_sustain = 0.3217`, `+22.3%` over 1D**).
   - **`whisper_asmr` & `underwater_submerged` (`scene + soundscape`):** `whisper_asmr` (`rescued = ['whisper']`) elevates breathy sibilance (`centroid = 3425.4 Hz` vs. `1D` `3232.8 Hz` and `base` `2403.1 Hz`); `underwater_submerged` (`rescued = ['floating']`) shifts the kitchen into aqua-teal aquatic refraction with buoyant floating motion (`temp_l1_mean = 16.08` vs. `1D` `13.59`, `centroid = 3201.3 Hz`).
3. **Ablation 8c — Stratified Bank Verification (`ablation8c_stratified_bank_concepts.png`, `gpu_base50_and_stratified.json`):**
   Across `film_16mm_1974` (`monolithic_override`, `hardware_aisle`: `0.3010` 1D vs. `0.4107` richer warm Kodak emulsion under erasure), `cursed_cctv` (`intermediate`, `0.2346` vs. `0.2521` greenish surveillance cast), `bioluminescent_glow` (`0.2883` vs. `0.3040`), and `felt_muppet` (`sitcom_kitchen`: `0.5212` vs. `0.5250` wide-mouthed felt puppet caricature), `add_norm_matched_erasure` never degrades coherent 1D concepts while sharpening rescued texture cues.

### 6.7 Cross-Checkpoint Generalization: `P-H3-Image`, `MiniMax-H3-Image`, & Undistilled 50-Step Base H3

To prove Clamped Pole-A Subspace Erasure is an intrinsic property of H3's `TokenRefiner` + DiT rather than an artifact of 6-step Pika `SVD995` LoRA distillation, we evaluate three undistilled checkpoints (`ablation9_base50_undistilled_h3.png`, `gpu_base50_and_stratified.json`, `metrics_pre_int.json`):

| Checkpoint & Sampler (`loras=[]` Undistilled) | Task / Scene | `1D add_norm_matched` | `Pole-A Erasure` (`add_norm_matched_erasure`) | Verified Outcome |
| :--- | :--- | :---: | :---: | :--- |
| **50-Step Base `MiniMax-H3` (`fl2va`, Euler 50, `cfg=1.0`)** | `noir` still (`indoor_neutral`, `s=730451892` / `s=101`) | `0.2222` / `0.2398` *(0/2, warm wood & green plants)* | **`0.0402` / `0.0469` *(2/2 pure silver-gelatin B&W!)*** | **`5.3x` saturation drop** — proves Pole-A erasure is native to undistilled H3. |
| **50-Step Base `MiniMax-H3` (`fl2va`, Euler 50, `cfg=1.0`)** | `paper_origami` still (`sitcom_kitchen`, 2 seeds) | `0.5864` / `0.4139` *(cubic paper heads)* | **`0.5536` / `0.4068` *(cleaner folded table, floor & percolator)*** | Both father & daughter rendered as faceted paper craft across both seeds. |
| **50-Step Base `MiniMax-H3` (`fl2va`, Euler 50, `cfg=1.0`, `39f` AV)** | `noir` speaking `39f` AV (`sitcom_kitchen`, explicit color nouns) | `sat_mean = 0.4138` *(avocado-green cabinets, yellow chairs)* | **`sat_mean = 0.3599`** *(cabinets, walls & chairs erased to white/grey monochrome)* | Even un-CFG'd (`cfg=1.0`) against dense named color clauses, erasure strips room chrominance while keeping speech. |
| **8-Step `P-H3-Image` (`2304x1792`, `4.1 MP`, `shift=1.0`)** | `noir` & `paper_origami` (`Check C`) | `noir: 0.203` *(color)*; `origami`: half-human | **`noir: 0.041` *(B&W)***; **`origami`**: 2 full-body paper figures | Transfers cleanly to high-megapixel `shift=1.0` image schedule. |
| **4-Step Native `MiniMax-H3-Image` (`4.0 MP`, `--families h3_image`)** | `noir` (`verify_2026-11-21`, rotated `h3_image` refiner) | `sat_mean = 0.2355`, `red_excess = +5.85` *(warm color)* | **`sat_mean = 0.0330`, `red_excess = -0.04` *(pure B&W!)*** | Holds even across `MiniMax-H3-Image`'s rotated text-refiner weights. |

---

## 7. Design Laws, Multi-Modal Boundaries & Tool Routing

1. **The Pole-A Coverage Law (`N >= 8..12` Rotated Baselines):** A rank-8 erasure matrix $\mathbf{V}_r$ can only erase Pole-A attributes that *varied across your Pole-A spans*. Always rotate 8–12 distinct baseline lighting/material/acoustic environments on Pole A (`warm tungsten | noon sunlight | neon street | coastal mist -> noir`); 1–2 pairs (`40%–52%` Grassmannian coverage) behave like 1D vectors on conflicting prompts.
2. **Early-Step Palette Lock (`[0, 0.50]` under `shift = 12.0`):** Global color and coarse 3D silhouette commit before unshifted midpoint $b = 0.50$ (`end_percent = 0.33..0.50`). Shortening `MiniMaxH3Steer.end_percent = 0.50` retains `94%` of `noir`'s color erasure (`0.0439` vs. `0.0264`) while preventing late-step high-strength facial burn on `<Picture 1>` subjects.
3. **Unsteered Vision-Token Palette Locks (`fl2va` & `<Picture 1>` `ref2va`):** Text-row subspace erasure intentionally leaves `tags == 0` vision tokens untouched (`[C6]` invariant). Consequently:
   - On **`<Picture 1>` `ref2va`**, `target = "description_only"` or `"scene_only"` enables **Selective-Color / Room-Grade Subject Lock** (cooling the surrounding room while holding `<Picture 1>`'s face, voice, and blazer color).
   - On **`fl2va` first+last frame interpolation** (`h3_first_last_frame_pika_control_vector_api.json`), `843 + 843` color keyframe vision tokens lock scene palette (`sat_mean = 0.4361 -> 0.4328` even with `MiniMaxH3ReleaseReferences(0.33)`); pre-grade endpoint stills in `MiniMax-H3-Image` (`noir` `4.0 MP`: `0.2355 -> 0.033` B&W!) when converting color `fl2va` keyframes to monochrome.
   - On **3D `[F, H, W]` spatial portals** (`MiniMaxH3SpatialDirector` + `edit_pass = "edited_only"`), keep vector-only wipes at $\ge 30\%$ frame area (`vertical_split`: **`5.91x`** L/R contrast on `t2va_disco_roller_rink`); tiny `~10%` circular portals are pulled toward the surrounding `90%` warm context by H3's 60 full-sequence 3D attention blocks unless paired with explicit `edited` prose.
4. **Clean `1x` vs. `2x` Routing Across the Pack:**
   - **Fresh mutually exclusive lighting, multi-beat choreography, or wardrobe:** `1x` **`MiniMaxH3PromptExpander`** / **`MiniMaxH3ImagePromptExpander`**.
   - **Reusable orthogonal film stock, monochrome, tactile medium, weather, or `40 Hz` acoustic space — even against conflicting prompts:** `1x` **`MiniMaxH3Steer`** (`mode = "add_norm_matched_erasure"` default; `< 2 ms` **`MiniMaxH3WordVector`** for rapid lexical sketching).
   - **Clause-bound garment/prop swaps, `lock_take` preservation, or small `[F, H, W]` portals:** `2x` **`MiniMaxH3EditGuider`** + **`MiniMaxH3PromptEdit`** / **`MiniMaxH3AutoEditPrompt`**.

---

## References

1. **Khattab, O., & Zaharia, M.** (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR 2020.
2. **Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M.** (2022). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL 2022; *PLAID*, CIKM 2022.
3. **Scheerer, J. L., Zaharia, M., Potts, C., Alonso, G., & Khattab, O.** (2025). *WARP: An Efficient Engine for Multi-Vector Retrieval.* SIGIR 2025 Best Paper.
4. **Clavié, B., Chaffin, A., & Sourty, R.** (2024–2026). *Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling* (`arXiv:2409.14683`); *PyLate: Flexible Training and Retrieval for Late Interaction Models* (`arXiv:2508.03555`); *LateOn-hpool-regularized* & *LIR @ ECIR 2026* (`arXiv:2511.00444`).
5. **Chaffin, A., & Sourty, R.** (2026). *ColBERT-Zero: Zero-Supervision Multi-Vector Retrieval.* LightOn Research (`arXiv:2602.16609`).
6. **Clavié, B., Lee, S., Shakir, A., & Kato, M. P.** (2025–2026). *mxbai-edge-colbert-v0* & *Latent Terms: Dense Retrievers Know More Than They Can Express.* Mixedbread AI (`arXiv:2605.29384`).
7. **Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P.** (2025). *ColPali / ColQwen: Efficient Document Retrieval with Vision Language Models.* ICLR 2025.
8. **Dhulipala, L., Hadian, M., Jayaram, R., Lee, J., & Mirrokni, V.** (2024). *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings.* NeurIPS 2024.
9. **Tulkens, S., & van Dongen, T.** (2024–2026). *Model2Vec: Fast State-of-the-Art Static Embeddings*, *tokenlearn*, *POTION*, & *reach*. Minish Lab (`github.com/MinishLab/model2vec`).
10. **Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S.** (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023.
11. **Gandikota, R., Orgad, H., Belinkov, Y., Materzyńska, J., & Bau, D.** (2024). *Unified Concept Editing in Diffusion Models (UCE).* WACV 2024.
12. **Gao, X., et al.** (2025). *EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers.* ICML 2025; alongside *MUTE* (`arXiv:2609.36832`, 2026) and *RASteer* (`arXiv:2610.01969`, 2026).
