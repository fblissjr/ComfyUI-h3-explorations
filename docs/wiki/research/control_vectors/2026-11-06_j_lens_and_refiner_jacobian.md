# 2026-11-06: The Jacobian Lens (`J-Lens`) and `TokenRefiner` Wiener Readout for H3 Control Vectors

This note records the mathematical derivation, empirical verification, and implementation of an analytical **Jacobian Lens (`J-Lens`)** for MiniMax H3 post-refiner control vectors (`vector.post_refiner` $\in \mathbb{R}^{5376}$ and `vector.encoder` $\in \mathbb{R}^{5120}$), adapting Gurnee, Lindsey et al. (*"Verbalizable Representations Form a Global Workspace in Language Models"*, Anthropic, July 2026; `anthropics/jacobian-lens`) to H3's two-stage text conditioning stack.

For the control-vector math and runtime contract, see [`2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md) and [`../../wiki/control_vectors.md`](../../control_vectors.md). For soft ridge hold attenuation ($\alpha \in [0, 1]$) and inference-time cross-vector holds (`mode = "hold"`), see [`2026-11-06_detangle_comparison_and_borrowable_concepts.md`](2026-11-06_detangle_comparison_and_borrowable_concepts.md) and [`2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md).

---

## 1. Motivation: Two Observability Gaps in `5376`-d Post-Refiner Space

MiniMax H3 extracts text conditioning from `Qwen3-VL-32B-Instruct` truncated after `layers.49` (`hidden_size = 5120`), projects it through the DiT's `condition_proj` (`5120 -> 5376`), and passes it through the 2-block bidirectional `TokenRefiner` (`5376 -> 5376`) before feeding the refined text tokens into the 50-block joint audio-video DiT ([`2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md) §1).

Previously, verbalizing a control vector in natural language via [`exl3_qwen3vl/label_h3_control_vectors.py`](exl3_qwen3vl/label_h3_control_vectors.py) had two blind spots:

1. **Saved `.safetensors` vectors could not be read back from `vector.post_refiner`**:
   - [`docs/wiki/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py::load_encoder_vector_from_safetensors`](exl3_qwen3vl/label_h3_control_vectors.py#L613-L636) checks `if "vector.encoder" in f.keys():` and returns `None` whenever `vector.encoder` (`5120`-d) was not saved alongside `vector.post_refiner` (`5376`-d).
   - Even when `vector.encoder` is saved ([`bench/control_vector/build_cmd.py:160-185`](../../../../bench/control_vector/build_cmd.py#L160-L185)), running [`compute_top_logit_shifts`](exl3_qwen3vl/label_h3_control_vectors.py#L640-L754) requires loading the 64-layer `Qwen3-VL-32B-Instruct` EXL3 checkpoint on GPU (`~17 GB` VRAM, `~12s` per vector).
2. **`holds.post_refiner` and inference-time `mode = "hold"` operate in `5376`-d post-refiner space, which `layers.49` cannot see**:
   - Build-time hold orthogonalization ([`control_vector_nodes.py:914-923`](../../../../control_vector_nodes.py#L914-L923), [`bench/control_vector/build_cmd.py:160-185`](../../../../bench/control_vector/build_cmd.py#L160-L185)) and inference-time cross-vector hold projection (`mode = "hold"` / `project_out` with soft or over-projected $\alpha \in [0, 2]$, [`2026-11-06_detangle_comparison_and_borrowable_concepts.md`](2026-11-06_detangle_comparison_and_borrowable_concepts.md) §3) modify `vector.post_refiner` (`5376`-d):
     $$\mathbf{v}_{\text{PR}}(\alpha) = \mathbf{v}_{\text{PR, raw}} - \alpha U_{\text{hold}}^\top \left(U_{\text{hold}} \mathbf{v}_{\text{PR, raw}}\right)$$
   - A `layers.49`-only readout inspects `vector.encoder` (`5120`-d) **before** `condition_proj`, before the 2-block `TokenRefiner` (and any Pika `token_refiner` LoRA branch), and before `5376`-d hold projection. It therefore cannot tell what semantic directions a `TokenRefiner` LoRA amplifies, what concepts a `5376`-d hold projection $\alpha \in [0, 1]$ strips out vs. preserves, what a composite multi-vector recipe $\Delta\mathbf{r} = (I - \alpha U^\top U)\sum_i s_i \mathbf{v}_i$ pushes toward, or what concepts a **textless media-contrast vector** (extracted from `<Picture>` / `<Video>` reference pairs where no text prompts exist at all) encodes in `5376`-d space.

---

## 2. Analytical Expected Jacobians Across `Qwen3-VL` (`layers.50..63`) and `TokenRefiner`

In Anthropic's Jacobian Lens (`anthropics/jacobian-lens`), the local linear transport operator from layer $l$ to the final pre-unembedding representation $\mathbf{h}_L$ is averaged across a calibration corpus of token activations $\mathbf{x} \sim \mathcal{D}$ so input-specific attention routing and MLP gate noise cancel while the shared "global workspace" transport matrix survives:

$$\bar{J}_{l \to L} = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\left[\frac{\partial \mathbf{h}_L(\mathbf{x})}{\partial \mathbf{h}_l(\mathbf{x})}\right]$$

For MiniMax H3, we need two coupled expected Jacobians computed over `24` stratified multimodal prompts $\times 64$ tokens (`1,536` tokens total) sampled across [`prompt_bank/bank.json`](../../../../prompt_bank/bank.json):
1. **$\bar{J}_{49 \to 63} \in \mathbb{R}^{5120 \times 5120}$**: the expected Jacobian from `Qwen3-VL-32B-Instruct` `layers.49` (`5120`-d, where H3 truncates the encoder) through `layers.50..63` and the final `model.language_model.norm` (`RMSNorm`) to `lm_head`.
2. **$\bar{J}_{\text{ref}}^{(f)} \in \mathbb{R}^{5376 \times 5120}$**: the expected Jacobian from `layers.49` (`5120`-d) through `condition_proj` (`5120 -> 5376`) and the 2-block bidirectional `TokenRefiner` (`5376 -> 5376`) for each weight family $f \in \lbrace \text{fl2va}, \text{ref2va} \rbrace$ (with its default Pika H3 Flex `svd995` LoRA applied to `diffusion_model.token_refiner.*`).

### 2.1 Rule 1: The Token-0 Attention Sink (`52.4x` Norm Outlier at `layers.49`)

Measuring the exact BF16 forward pass of `Qwen3-VL-32B-Instruct` (`layers.0..49`) across the 24 `prompt_bank/` sequences reveals a massive attention-sink norm disparity at `layers.49` (consistent with [`2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md) §2.1 / §3.2 `E2.1`):

| Token Slice at `layers.49` | Median Row L2 Norm | Ratio vs. `tok1+` | Role in Forward Pass vs. Jacobian Expectation |
|---|---:|---:|---|
| `H_49[:, 0, :]` (token 0) | `15,309.0` | `52.37x` | **Keep in sequence during forward attention** so causal `layers.50..63` and bidirectional `TokenRefiner` heads dump unused attention mass onto token 0 as trained |
| `H_49[:, 1:, :]` (`tok1+`, `1,512` tokens) | `292.3` | `1.00x` | **Use exclusively (`[:, 1:, :]`)** when averaging `RMSNorm` Jacobians, SwiGLU MLP gate derivatives, and empirical activation covariance $\widehat{\text{Cov}}(H_{49, 1:})$ |

Excluding token 0 during the forward pass breaks attention softmax normalization across the remaining tokens; including token 0 when averaging `RMSNorm` or covariance matrices corrupts the scale factor $s(\mathbf{x}) = (\lVert\mathbf{x}\rVert_2^2 / D + \varepsilon)^{-1/2}$ and rank-1 outer product $\hat{\mathbf{x}}\hat{\mathbf{x}}^\top / D$. Keeping token 0 in the forward pass while restricting expectations to `[:, 1:, :]` avoids both errors.

### 2.2 Rule 2: Broadcast Sequence Perturbations and the Attention Value Path ($\beta$)

Both [`docs/wiki/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py:702-705`](exl3_qwen3vl/label_h3_control_vectors.py#L702-L705) (`h_pos[:, 1:, :] = h_pos[:, 1:, :] + d`) and [`control_vector_nodes.py`](../../../../control_vector_nodes.py) broadcast the steering vector $\mathbf{\delta}$ across a span of token positions $j \in \mathcal{S} \subseteq \lbrace 1, \dots, S-1 \rbrace$ rather than perturbing a single token in isolation.

Under a broadcast perturbation $\mathbf{h}_j \mapsto \mathbf{h}_j + \mathbf{\delta}$ for $j \in \mathcal{S}$, the output of an attention block at token $i \in \mathcal{S}$ with frozen routing weights $A_{i,j}$ differentiates with respect to $\mathbf{\delta}$ as:

$$\frac{\partial \text{Attn}_i}{\partial \mathbf{\delta}} = \left(\sum_{j \in \mathcal{S}} A_{i,j}\right) W_O W_V J_{\text{RMSNorm}_1}(\mathbf{h}_i) \approx \beta W_O W_V J_{\text{RMSNorm}_1}(\mathbf{h}_i)$$

where $\beta = \mathbb{E}_{i \in \mathcal{S}}\left[\sum_{j \in \mathcal{S}} A_{i,j}\right] \in [0, 1]$ is the average non-sink attention mass that remains inside the steered span $\mathcal{S}$ (`1 - A_{i,0}` when $\mathcal{S} = \lbrace 1, \dots, S-1 \rbrace$, or the in-span fraction for `diff_span`), while each per-token `RMSNorm` and SwiGLU FFN block has exact analytical Jacobian:

$$J_{\text{RMSNorm}}(\mathbf{x}) = s(\mathbf{x}) \text{diag}(\mathbf{w})\left(I_D - \frac{\hat{\mathbf{x}}\hat{\mathbf{x}}^\top}{D}\right), \qquad J_{\text{FFN}}(\mathbf{y}) = W_{\text{down}}\Big(\text{diag}(\text{silu}'(\mathbf{g})\odot \mathbf{u}) W_{\text{gate}} + \text{diag}(\text{silu}(\mathbf{g})) W_{\text{up}}\Big)$$

In `comfy/ldm/minimax/model.py:158-169, 250-276, 477-484, 509-511`, `MiniMaxH3Model`'s 2-block `TokenRefiner` has `hidden_size = 5376`, `num_attention_heads = 56`, `attention_head_dim = 128` (`inner_dim = 7168`, not `5376`), `ffn_hidden_size = 14336`, and `norm_eps = 1e-5`. Testing $\bar{J}_{\text{ref}}$ across the initial **35 dual `.safetensors` control vectors** (`19` `ref2va`, `16` `fl2va`, each holding both `vector.encoder` $\in \mathbb{R}^{5120}$ and `vector.post_refiner` $\in \mathbb{R}^{5376}$; disk now holds `90` dual `.safetensors` files = `45` `fl2va` + `45` `ref2va`) shows why the broadcast attention value path is essential:

| `TokenRefiner` Attention Value Weight $\beta$ | Mean $\cos(\bar{J}_{\text{ref}}\mathbf{v}_{\text{enc}}, \mathbf{v}_{\text{PR}})$ (All 35 Vectors) | Best Tikhonov $\cos(J_{\text{pinv}}\mathbf{v}_{\text{PR}}, \mathbf{v}_{\text{enc}})$ | Best Tikhonov $\cos(\bar{J}_{49 \to 63}J_{\text{pinv}}\mathbf{v}_{\text{PR}}, \bar{J}_{49 \to 63}\mathbf{v}_{\text{enc}})$ | Physical Interpretation |
|---|---:|---:|---:|---|
| $\beta = 0.00$ (MLP + residual only; attention omitted) | `0.5091` | `0.4040` | `0.7217` | Misses cross-token `W_O W_V` value accumulation across the steered span |
| $\beta = 0.25$ (partial span attention + MLP) | `0.7665` | `0.4692` | `0.7713` | Under-weights multi-token span attention |
| **$\beta = 0.50$ (`diff_span` broadcast attention + MLP)** | **`0.8290`** | **`0.4870`** | **`0.7838`** | Matches multi-token `diff_span` pooling (`6–25` tokens inside a `~60`-token prompt) |
| **$\beta = 1.00$ (full-sequence `tok1+` broadcast + MLP)** | **`0.8355`** (`0.8359` saved: `0.8324` `ref2va`, `0.8400` `fl2va`) | `0.4613` | `0.7683` | Matches full `scene_text` / `non_vision` broadcast steering |

Including the broadcast attention value path $\beta W_O W_V J_{\text{RMSNorm}_1}$ jumps forward alignment between $\bar{J}_{\text{ref}}\mathbf{v}_{\text{enc}}$ and the true nonlinear `TokenRefiner` contrastive difference $\mathbf{v}_{\text{PR}}$ from `0.5091` to **`0.8290–0.8359`**.

---

## 3. `TokenRefiner` Singular Value Spectrum and Wiener vs. Isotropic Pseudoinversion

### 3.1 Spectral Concentration of `TokenRefiner` (`81.4%` of Signal in Top `10%` of Singular Vectors)

Taking the singular value decomposition $\bar{J}_{\text{ref}}^{\text{ref2va}} = U \operatorname{diag}(\sigma_1, \dots, \sigma_{5120}) V^\top \in \mathbb{R}^{5376 \times 5120}$ reveals that the 2-block `TokenRefiner` acts as a strongly anisotropic low-pass filter on `layers.49`:

| Singular Value Rank $k$ (of `5120`) | Singular Value $\sigma_k(\bar{J}_{\text{ref}}^{\text{ref2va}})$ | Cumulative `vector.post_refiner` Energy $\lVert U_{:, :k}^\top \mathbf{v}_{\text{PR}}\rVert_2^2 / \lVert\mathbf{v}_{\text{PR}}\rVert_2^2$ (`n = 35`) | Isotropic Random Baseline ($k / 5120$) |
|---:|---:|---:|---:|
| `1` (`max`) | `3.0118` | `1.7%` | `0.02%` |
| `51` (`p99`, top `1%`) | `1.3597` | `33.4%` | `1.0%` |
| `128` (top `2.5%`) | `0.9995` | **`58.5%`** | `2.5%` |
| `256` (`p95`, top `5%`) | `0.6620` | **`73.1%`** | `5.0%` |
| `512` (`p90`, top `10%`) | `0.3589` | **`81.4%`** | `10.0%` |
| `1024` (top `20%`) | `0.1605` | **`86.8%`** | `20.0%` |
| `1280` (`p75`) | `0.1220` | `88.3%` | `25.0%` |
| `2560` (`median`) | `0.0474` | `92.9%` | `50.0%` |
| `3840` (`p25`) | `0.0198` | `96.3%` | `75.0%` |
| `4608` (`p10`) | `0.0083` | `98.1%` | `90.0%` |
| `5120` (`min`) | `0.0011` | `99.3%` (`100.0%` of $5120$-d column space of $U$) | `100.0%` |

**`81.4%` of the norm squared of our 35 real `vector.post_refiner` directions lives inside the top `512` (`10%`) left singular vectors of $\bar{J}_{\text{ref}}$** (`99.3%` in the full $5120$-dimensional column space of $U \in \mathbb{R}^{5376 \times 5120}$, with only `0.7%` in the $256$-dimensional orthogonal complement), while the bottom half of the spectrum ($\sigma_{2560..5120} \le 0.0474$) is attenuated by `64x–2,700x` relative to $\sigma_1$.

Because $\bar{J}_{\text{ref}}$, $\Sigma_{49}$, and $\bar{J}_{49 \to 63}$ are derived purely from the `Qwen3-VL-32B-Instruct` and `TokenRefiner` weights over `prompt_bank/bank.json` with **zero `.toml` control vectors used in their construction**, the J-Lens can also probe the **unsupervised left singular vectors** $\mathbf{u}_k \in \mathbb{R}^{5376}$ of $\bar{J}_{\text{ref}}^{\text{fl2va}}$ directly to see which semantic axes `TokenRefiner` amplifies most strongly on its own:

| Left Singular Vector $\mathbf{u}_k$ of $\bar{J}_{\text{ref}}^{\text{fl2va}}$ | Singular Value $\sigma_k$ | Top Positive Words (`+u_k`) | Top Negative Words (`-u_k`) |
|---|---:|---|---|
| `SV 1` | `2.872` | `complex`, `chaotic`, `rigid`, `intricate`, `acidic`, `cautious` | `footage`, `runner`, `music`, `cooked`, `shot`, `cinema` |
| `SV 2` | `1.897` | `chaotic`, `excessively`, `glitch`, `shaky`, `messy`, `extremely` | `photographs`, `vintage`, `archival`, `photograph`, `retrod`, `autobiographies` |
| `SV 4` | `1.770` | `approximately`, `beginning`, `repetitive`, `fairly`, `relatively`, `frequently` | `cinematic`, `mood`, `atmosphere`, `lens`, `cinema`, `film` |
| `SV 5` | `1.747` | `film`, `movie`, `layered`, `cinematic`, `cinematography`, `documentary` | `identical`, `unchanged`, `intact`, `same`, `unilateral`, `consistent` |
| `SV 7` | `1.701` | `close`, `frontal`, `front`, `characters`, `above`, `detailed` | `document`, `amateur`, `orient`, `tentative`, `homemade`, `procedural` |
| `SV 8` | `1.674` | `brick`, `cozy`, `amateur`, `crude`, `cosy`, `maybe` | `elongate`, `slender`, `transition`, `vertical`, `elegant`, `transitions` |

### 3.2 Why Covariance-Weighted Wiener Inversion Beats Isotropic Tikhonov Pseudoinversion

To read a `5376`-d `vector.post_refiner` tensor $\mathbf{v}_{\text{PR}}$ through `layers.50..63` (`5120`-d), we must invert $\mathbf{v}_{\text{PR}} \approx \bar{J}_{\text{ref}}\mathbf{v}_{\text{enc}}$.

A standard isotropic Tikhonov pseudoinverse
$$J_{\text{Tikhonov}}^\dagger = \left(\bar{J}_{\text{ref}}^\top \bar{J}_{\text{ref}} + \lambda^2 I_{5120}\right)^{-1}\bar{J}_{\text{ref}}^\top$$
assumes an isotropic prior $\mathbf{v}_{\text{enc}} \sim \mathcal{N}(\mathbf{0}, I_{5120})$. Because Qwen `layers.49` activations `H_49[:, 1:, :]` occupy an anisotropic low-rank cone, an isotropic prior maps small post-refiner residuals into dead `layers.49` directions that `layers.50..63` never see during pretraining.

Replacing $I_{5120}$ with the shrinkage-regularized empirical covariance of `tok1+` activations at `layers.49`,
$$\Sigma_{49} = (1 - \gamma) \widehat{\text{Cov}}\left(H_{49, 1:}\right) + \gamma \frac{\operatorname{tr}(\widehat{\text{Cov}})}{5120} I_{5120} \qquad (\gamma = 0.20),$$
yields the closed-form **Wiener inverse**:

$$J_{\text{Wiener}}^{(f)} = \Sigma_{49}\left(\bar{J}_{\text{ref}}^{(f)}\right)^\top\left(\bar{J}_{\text{ref}}^{(f)}\Sigma_{49}\left(\bar{J}_{\text{ref}}^{(f)}\right)^\top + \lambda^2 I_{5376}\right)^{-1} \in \mathbb{R}^{5120 \times 5376}$$

with $\lambda = \lambda_{\text{rel}} \sigma_{\max}(\bar{J}_{\text{ref}}\Sigma_{49}^{1/2})$ ($\lambda_{\text{rel}} = 0.03$). Across all 35 dual `.safetensors` vectors on disk, Wiener inversion improves both `layers.49` recovery and downstream `layers.63` transport fidelity:

| Inverse Estimator $\hat{\mathbf{v}}_{\text{enc}} = J^{-1}\mathbf{v}_{\text{PR}}$ | Mean $\cos(\hat{\mathbf{v}}_{\text{enc}}, \mathbf{v}_{\text{enc}})$ (`5120`-d `layers.49`) | Mean $\cos(\bar{J}_{49 \to 63}\hat{\mathbf{v}}_{\text{enc}}, \bar{J}_{49 \to 63}\mathbf{v}_{\text{enc}})$ (`5120`-d `layers.63`) |
|---|---:|---:|
| Isotropic Tikhonov $(\bar{J}_{\text{ref}}^\top \bar{J}_{\text{ref}} + \lambda^2 I)^{-1}\bar{J}_{\text{ref}}^\top$ | `0.4832` | `0.7820` |
| **Empirical-Covariance Wiener $J_{\text{Wiener}}$ ($\gamma = 0.20, \lambda_{\text{rel}} = 0.03$)** | **`0.6663` (`0.6690` saved)** | **`0.8769` (`0.8778` saved)** |

After transport through $\bar{J}_{49 \to 63}$, the Wiener-recovered post-refiner direction $\bar{J}_{49 \to 63} J_{\text{Wiener}} \mathbf{v}_{\text{PR}}$ achieves **`0.8769–0.8778` mean cosine similarity** with the true `layers.49` direction $\bar{J}_{49 \to 63}\mathbf{v}_{\text{enc}}$ across all 35 vectors.

---

## 4. Combining 1 Shared Jacobian with 7 Cloze Baseline Logit Vectors (`14.5 ms` on CPU)

### 4.1 Why Uncentered Linear Projection $W_{\text{head}} \bar{J}_{49 \to 63} \mathbf{v}$ Is Not Enough

Directly sorting raw dot products $\Delta \mathbf{z} = W_{\text{head}} \bar{J}_{49 \to 63} J_{\text{Wiener}} \mathbf{v}_{\text{PR}}$ (where $W_{\text{head}}$ is `lm_head.weight`) over a regex vocabulary filter has two failure modes:
1. **BPE prefix stem leakage**: In Qwen's `151,936`-piece BPE vocabulary, multi-token words split into leading space pieces (`" explos"`, `" pne"`, `" fauc"`, `" Buccane"`, `" commande"`). Without [`docs/wiki/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py::_get_bpe_vocab_map`](exl3_qwen3vl/label_h3_control_vectors.py#L233-L324) (which maps `24,912` complete words and unambiguous domain/dictionary stems to their full English words while excluding stopwords and ambiguous short prefixes), raw BPE subwords pollute the top ranks.
2. **Off-context high-norm unembedding rows**: Without a syntactic cloze baseline, high-norm proper nouns and technical tokens (`"USSR"`, `"Yugoslavia"`, `"openid"`, `"concatenate"`) that are grammatically impossible after *"The visual medium, camera look, lighting, and image fidelity look ___"* can score high raw dot products.

### 4.2 1 Shared Jacobian + 7 Cloze Baseline Logit Vectors + Exact Partition-Function Union

In [`docs/wiki/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py:665-721`](exl3_qwen3vl/label_h3_control_vectors.py#L665-L721), `compute_top_logit_shifts` evaluates 7 syntactic cloze prefixes (`141` tokens total across visual medium, lighting/film stock, head/body proportions, room motion, signage, surface texture, and `overall_soundscape`) and applies three nonlinear filters:
1. **Grammatical admissibility gate (`lp_pos > -11.5`)**: Only tokens whose steered log-probability $\log P(w \mid \text{prefix}, +\mathbf{d}) > -11.5$ on at least one of the 7 cloze prefixes can appear.
2. **One-sided peak-flattening cancellation**:
   $$\text{shift}_{+}(w) = \min\Big(\log P(w \mid +\mathbf{d}) - \log P(w \mid \text{base}), \quad \log P(w \mid +\mathbf{d}) - \log P(w \mid -\mathbf{d})\Big)$$
   which preserves 100% of directional boosts when $\log P(w \mid -\mathbf{d}) \le \log P(w \mid \text{base})$ while subtracting isotropic off-manifold entropy flattening when both $+\mathbf{d}$ and $-\mathbf{d}$ raise a token above baseline.
3. **Max-pooling across the 7 syntactic slots**: $\text{score}(w) = \max_{p \in \lbrace 0..6 \rbrace} \text{gain}_p(w)$, allowing visual adjectives, motion verbs, nouns, and acoustic descriptors to each surface on the prefix where they are grammatically valid.

Critically, we do **not** need 7 separate $5120 \times 5120$ Jacobians (`~367 MB`). Instead, we combine:
- **One** `1,536`-token prompt-bank-averaged transport operator $\bar{J}_{49 \to 63}$, and
- **Seven** unsteered baseline logit vectors $\mathbf{z}_{\text{base}}^{(0..6)}$ precomputed once from the exact 64-layer `Qwen3-VL-32B-Instruct` forward pass on the 7 cloze prefixes:
  $$\mathbf{z}_{\pm}^{(p)} = \mathbf{z}_{\text{base}}^{(p)} \pm \left(W_{\text{head}} \bar{J}_{49 \to 63}\right) \mathbf{d}$$

#### The Full-Vocabulary `logsumexp` Partition-Function Invariant
In [`docs/wiki/research/control_vectors/exl3_qwen3vl/label_h3_control_vectors.py:707-709`](exl3_qwen3vl/label_h3_control_vectors.py#L707-L709), `F.log_softmax(tail_logits(...), dim=-1)[valid_idx]` normalizes over all `151,936` vocabulary tokens **before** indexing `valid_idx` (`24,912` content words). Because function words, articles, and punctuation (`" the"`, `" a"`, `","`, `"."`) dominate the next-token probability mass on cloze prefixes, taking `log_softmax` over `valid_idx` alone omits the dominant terms in the partition function $\log \sum_{v=1}^{151936} \exp(z_v)$.

Taking `union_idx` ($27,407$ tokens) as the union of `valid_idx` ($24,912$ words) with the top-$2,048$ highest-logit background tokens across the 7 cloze prefixes reproduces the exact `151,936`-token `logsumexp` denominator to **`0.0098` nats max error** across all 7 prefixes while requiring only a single $[27407, 5120] \times [5120]$ matrix-vector product (`14.5 ms` on CPU).

### 4.3 Storage Location Outside `models/control_vectors/`
Because ComfyUI's `folder_paths.get_filename_list("control_vectors")` (`folder_paths.recursive_search`) walks subdirectories via `os.walk` ([`control_vector_nodes.py:82-97`](../../../../control_vector_nodes.py#L82-L97)), any `.safetensors` file placed inside a subdirectory of `<comfy>/models/control_vectors/` would still appear in `MiniMaxH3ApplyControlVectors`'s `vector_name` dropdown. The precomputed J-Lens operator bundle is therefore stored in the sibling directory **`<comfy>/models/j_lens/j_lens_h3.safetensors`** (untracked by git and outside `models/control_vectors/`).

---

## 5. Empirical Verification on `5376`-d Control Vectors and Hold Sweeps

### 5.1 Cloze-Gated `5376`-d `vector.post_refiner` Readouts (`14.5 ms` on CPU)

Running the 7-prefix cloze-gated J-Lens readout ($\text{scale} = 90.0$, $\min\log P = -11.5$) directly on the `5376`-d `vector.post_refiner` tensors of our `diff_span` control vectors yields clean, part-of-speech-coherent lexical readouts across visual style, physical geometry, camera motion, lighting, and soundscape banks:

| Control Vector (`fl2va`, `diff_span`) | Top Positive `5376`-d J-Lens Words (`+v_PR`) | Top Negative `5376`-d J-Lens Words (`-v_PR`) |
|---|---|---|
| `film_16mm_1974` (`apply_holds=False`) | `vintage`, `faded`, `dated`, `outdated`, `taped`, `tape`, `older`, `decades` | `cinematic`, `detailed`, `focus`, `dynamic`, `lighting`, `realistic`, `meticulous`, `professional` |
| `vhs_1991_camcorder` | `tape`, `outdated`, `unstable`, `shaky`, `taped`, `dated` | `natural`, `detailed`, `focus`, `lighting`, `realistic`, `crisp` |
| `head_swell` | `oversized`, `dome`, `spherical`, `cavern`, `cone`, `engulf`, `overcrowd`, `cozy` | `consistent`, `action`, `established`, `motion`, `chore`, `shots`, `standard`, `accurate` |
| `felt_muppet` | `cartoon`, `puppet`, `animated`, `fuzzy`, `psychedelic`, `fake`, `tiny`, `oversized` | `natural`, `quiet`, `professional`, `steady`, `nature`, `presence`, `lighting`, `strong` |
| `claymation_stopmotion` | `cartoon`, `tiny`, `exaggerated`, `puppet`, `whimsical`, `grotesque` | `natural`, `indoor`, `shot`, `outdoor`, `lighting`, `naturally` |
| `underwater_submerged` | `underwater`, `submerged`, `submarine`, `psychedelic`, `floating`, `mesmerism`, `swimming`, `fragmented` | `tidy`, `quiet`, `natural`, `clean`, `dusty`, `outdoor`, `town`, `rustic` |
| `crystallize_ice` | `frost`, `frozen`, `glitch`, `shattered`, `snow`, `fractal` | `indoor`, `quiet`, `overall`, `environment`, `studio`, `natural` |
| `thermal_infrared` | `alarm`, `vortex`, `fused`, `colder`, `zone`, `crypt`, `envelop`, `thermos` | `natural`, `shot`, `realistic`, `sunlight`, `photography`, `realism`, `sunny`, `typical` |
| `chaos_panic` | `chaotic`, `frantic`, `panicked`, `desperate`, `panic`, `chaos` | `steady`, `gentle`, `gently`, `centered`, `natural`, `softly` |
| `handheld_documentary_zoom` | `gritty`, `frantic`, `erratic`, `glimpsed`, `shaky`, `messy` | `neutral`, `consistent`, `stationary`, `perfectly`, `unchanged`, `timeless` |
| `cursed_cctv` | `unstable`, `erratic`, `outdated`, `corrupted`, `jittery`, `alarm`, `shaky`, `twitch` | `natural`, `shot`, `realism`, `realistic`, `detailed`, `shots`, `photography`, `rich` |
| `auctioneer_rapid_patter` | `frantic`, `dazzling`, `exhilarate`, `jittery`, `rapid`, `rapidly` | `quiet`, `quietly`, `gentle`, `gently`, `relaxed`, `casual` |
| `operatic_bellow` | `crescendi`, `cathedral`, `soaring`, `intensified`, `gorge`, `exhilarate` | `quiet`, `casual`, `simple`, `casually`, `quietly`, `straightforward` |
| `wind_tunnel_gale` | `aggressive`, `storm`, `blasting`, `exhilarate`, `urgent`, `explosive` | `quiet`, `quietly`, `gently`, `gentle`, `timeless`, `stationary` |

### 5.2 Build-Time Hold Sweep ($\alpha \in \lbrace 0.0, 0.5, 1.0 \rbrace$ on `diff_span` `holds.post_refiner`)

Evaluating $\mathbf{v}_{\text{PR}}(\alpha) = \mathbf{v}_{\text{raw}} - \alpha U_{\text{hold}}^\top (U_{\text{hold}}\mathbf{v}_{\text{raw}})$ across $\alpha \in \lbrace 0.0, 0.5, 1.0 \rbrace$ directly in `5376`-d space shows what `holds.post_refiner` removes and preserves:

| Vector (`fl2va`, `diff_span`) | Hold Strength $\alpha$ | $\lVert\mathbf{v}_{\text{PR}}(\alpha)\rVert_2$ | Top Positive Words (`+v_PR`, with log-prob gain) | Top Negative Words (`-v_PR`, with log-prob gain) | What Changes Across $\alpha$ |
|---|---:|---:|---|---|---|
| `head_swell` | `0.0` (unprojected `apply_holds=False`) | `85.01` | `oversized` (`1.27`), `cone` (`1.00`), **`psychedelic` (`0.99`)**, `cavern` (`0.96`), `enormous` (`0.95`), `colossal` (`0.95`), **`futuristic` (`0.94`)**, `smaller` (`0.94`) | `consistent` (`1.54`), `shot` (`1.16`), `shots` (`1.12`), `consistently` (`1.06`), `overall` (`1.05`), `unchanged` (`1.03`) | Unprojected vector leaks style/genre terms (`psychedelic`, `futuristic`) |
| `head_swell` | `0.5` (soft ridge hold) | `79.55` | `oversized` (`1.33`), **`dome` (`1.09`)**, `cone` (`1.01`), `cavern` (`1.00`), **`spherical` (`0.99`)**, `psychedelic` (`0.97`), `engulf` (`0.92`), `futuristic` (`0.92`) | `consistent` (`1.55`), `shots` (`1.24`), `shot` (`1.21`), `motion` (`1.19`), `same` (`1.12`), `establishing` (`1.09`) | Geometric terms (`dome`, `spherical`) rise into top 5 while retaining `93.6%` of vector norm |
| `head_swell` | `1.0` (full orthogonal hold) | `77.65` | **`oversized` (`1.34`)**, **`dome` (`1.14`)**, **`spherical` (`1.05`)**, `cavern` (`1.01`), `cone` (`0.98`), `engulf` (`0.97`), `overcrowd` (`0.95`), `cozy` (`0.94`) | `consistent` (`1.57`), `action` (`1.45`), `established` (`1.44`), `motion` (`1.37`), `chore` (`1.35`), `shots` (`1.33`) | **`psychedelic` and `futuristic` are completely removed**; pure spherical head expansion dominates |
| `film_16mm_1974` (`apply_holds=False` vs. `[[hold]]`) | `0.0` (unprojected `apply_holds=False`) | `63.89` | `vintage` (`1.77`), `faded` (`1.60`), `dated` (`1.59`), `outdated` (`1.55`), `taped` (`1.38`), `tape` (`1.33`), `older` (`1.31`), `decades` (`1.26`) | `cinematic` (`1.85`), `detailed` (`1.69`), `focus` (`1.63`), `dynamic` (`1.61`), `lighting` (`1.56`), `realistic` (`1.54`) | Raw 16mm direction slightly mixes magnetic `tape`/`taped` with celluloid |
| `film_16mm_1974` (`apply_holds=False` vs. `[[hold]]`) | `0.5` (soft ridge hold) | `61.22` | `vintage` (`1.75`), `dated` (`1.68`), `outdated` (`1.58`), `faded` (`1.50`), `older` (`1.31`), `decades` (`1.29`), `tape` (`1.26`), `originally` (`1.21`) | `cinematic` (`2.04`), `dynamic` (`1.70`), `lighting` (`1.69`), `focus` (`1.67`), `detailed` (`1.60`), `cinematography` (`1.53`) | Demotes `taped` while sharpening optical opposition (`cinematography`) |
| `film_16mm_1974` (`apply_holds=False` vs. `[[hold]]`) | `1.0` (full orthogonal hold) | `60.30` | `dated` (`1.74`), `vintage` (`1.68`), `outdated` (`1.57`), `faded` (`1.34`), `original` (`1.30`), `decades` (`1.29`), `older` (`1.27`), `originally` (`1.27`) | `cinematic` (`2.18`), `lighting` (`1.78`), `dynamic` (`1.75`), `focus` (`1.67`), `cinematography` (`1.59`), `lens` (`1.51`) | Removes `tape`/`taped` completely and brings `lens` (`1.51`) and `cinematography` (`1.59`) into the top 6 negative pole |

### 5.3 Inference-Time Cross-Vector Hold Sweep (`mode = "hold"`: `cursed_cctv` Holding Out `chaos_panic` + `handheld_documentary_zoom`)

In [`2026-11-06_detangle_comparison_and_borrowable_concepts.md`](2026-11-06_detangle_comparison_and_borrowable_concepts.md) §2.1 and [`2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md) §4 Priority #2, we designed inference-time cross-vector holds (`mode = "hold"`, $\alpha \in [0, 1]$) so a user can project existing 10-pair control vectors out of a primary style or action vector without authoring bespoke `[[hold]]` pairs.

Testing `cursed_cctv` (`fl2va`, `diff_span`) while holding out the 2-vector orthonormal basis spanned by `chaos_panic` and `handheld_documentary_zoom` demonstrates why inference-time 10-pair holds work so well:

| Inference-Time Hold Configuration | $\lVert\mathbf{v}_{\text{PR}}(\alpha)\rVert_2$ | Top Positive Words (`+v_PR`, with log-prob gain) | Physical Interpretation |
|---|---:|---|---|
| $\alpha = 0.0$ (raw `cursed_cctv`) | `63.26` | `unstable` (`1.47`), **`erratic` (`1.38`)**, `outdated` (`1.27`), `corrupted` (`1.25`), **`jittery` (`1.22`)**, **`alarm` (`1.21`)**, **`shaky` (`1.20`)**, `twitch` (`1.17`) | Mixes low-bitrate CCTV degradation with camera shake (`shaky`, `erratic`, `jittery`) and panic (`alarm`) |
| $\alpha = 0.5$ (`cursed_cctv` $-$ $0.5 P_{\text{panic, shaky}}$) | `58.48` | `unstable` (`1.30`), `outdated` (`1.21`), `erratic` (`1.14`), `jittery` (`1.11`), `corrupted` (`1.10`), `twitch` (`1.07`), `warped` (`1.07`), `shaky` (`1.07`) | Softens camera shake while preserving CCTV signal corruption (`corrupted`, `warped`) |
| $\alpha = 1.0$ (`cursed_cctv` $-$ $1.0 P_{\text{panic, shaky}}$) | `56.80` | `outdated` (`1.09`), `fluctuate` (`1.06`), `unstable` (`1.05`), `intermittently` (`1.04`), `intermittent` (`1.01`), **`fixed` (`0.99`)**, **`timestamp` (`0.99`)** | **Completely removes `shaky`, `erratic`, `jittery`, and `alarm`**, exposing **`fixed`** and **`timestamp`** (a stationary wall-mounted CCTV feed with timestamp overlay) |

---

## 6. Instant (`< 2 ms` CPU) Word-to-`5376`-d Control Vector Synthesis (`j_lens.py::synthesize_from_words`)

### 6.1 Why Naive $\Sigma_{49}^{1.0} M_{\text{valid}}^\top$ Pullback Failed (`cos = 0.27–0.38`) vs. Why Spectral-Filtered $\Sigma_{49, \text{filt}}^{0.30}$ Works (`cos = 0.52–0.64`)

Our first attempt to synthesize control vectors directly from simple English seed words pulled raw rows of $M_{\text{valid}} = W_{\text{head}}[\text{valid}] \bar{J}_{49 \to 63}$ backward through the full empirical covariance $\Sigma_{49}^{1.0}$ (`cos_5120 = 0.27–0.38` against 10-pair `vector.encoder`). Inspecting the spectrum of $\Sigma_{49}$ and the row geometry of $M_{\text{valid}}$ revealed two distinct failure causes:
1. **Top-1 Common-Mode Eigenvalue Dominance in $\Sigma_{49}^{1.0}$**: The largest eigenvalue of $\widehat{\text{Cov}}(H_{49, 1:})$ is $\sim 100\times$ larger than the semantic eigenvalues, so multiplying by $\Sigma_{49}^{1.0}$ collapses every word difference onto a single common-mode activation direction.
2. **Shared Vocabulary Mean in $M_{\text{valid}}$ and Missing Forward Refiner Push**: Raw rows of $M_{\text{valid}}$ share a strong common token-frequency bias, and evaluating similarity in `5120`-d `layers.49` omits the `5376`-d `TokenRefiner` subspace filter $\bar{J}_{\text{ref}}^{(f)}$.

When we fix both inside [`j_lens.py`](../../../../j_lens.py) (`synthesize_from_words` / `--from-words`):
1. **Mean-center and row-normalize $M_{\text{valid}}$** across the `24,912`-word vocabulary:
   $$\widehat{M}_i = \frac{M_{\text{valid}, i} - \bar{\mathbf{m}}}{\lVert M_{\text{valid}, i} - \bar{\mathbf{m}}\rVert_2}, \qquad \Delta\mathbf{\mu} = \frac{\frac{1}{|P|}\sum_{i \in P} \widehat{M}_i}{\left\lVert\frac{1}{|P|}\sum_{i \in P} \widehat{M}_i\right\rVert_2} - \frac{\frac{1}{|N|}\sum_{j \in N} \widehat{M}_j}{\left\lVert\frac{1}{|N|}\sum_{j \in N} \widehat{M}_j\right\rVert_2}$$
2. **Precondition with spectral-filtered fractional covariance $\Sigma_{49, \text{filt}}^{0.30}$** (`Sigma_49_p030` in `j_lens_h3.safetensors`, where the top-1 common-mode eigenvalue $\lambda_{\max}$ is zeroed out and the remaining eigenvalues are raised to $p = 0.30$), and
3. **Push forward through $\bar{J}_{\text{ref}}^{(f)}$ (`5376`-d)**:
   $$\hat{\mathbf{v}}_{\text{enc}} = \text{normalize}\left(\Sigma_{49, \text{filt}}^{0.30} \Delta\mathbf{\mu}\right), \qquad \hat{\mathbf{v}}_{\text{PR}}^{(f)} = 67.5 \cdot \text{normalize}\left(\bar{J}_{\text{ref}}^{(f)} \Sigma_{49, \text{filt}}^{0.30} \Delta\mathbf{\mu}\right)$$

alignment with the true 10-pair `diff_span` `vector.post_refiner` jumps from `0.27–0.38` to **`mean cos_5376 = 0.5724` (`0.5238–0.6368` on §5.1 top-8 words, `0.5236` on ad-hoc user seed words)** in **`< 2 ms` on CPU with zero `Qwen3-VL` load and zero `.toml` files**:

| Target Concept (`fl2va`, `diff_span`) | Ad-Hoc Positive / Negative Seed Words (`--from-words` / `--against-words`) | Naive $\Sigma_{49}^{1.0}$ Pullback (`cos_5120`) | $\bar{J}_{\text{ref}} \Sigma_{49, \text{filt}}^{0.30}$ on §5.1 Top-8 Words (`cos_5376`) | $\bar{J}_{\text{ref}} \Sigma_{49, \text{filt}}^{0.30}$ on Ad-Hoc Pos+Neg Words (`cos_5376`) | 4 Positive Words Only vs. Default Neutral Anchor (`cos_5376`) |
|---|---|---:|---:|---:|---:|
| `chaos_panic` | `+chaotic, frantic, panicked, desperate, erratic, panic, chaos` / `-steady, gentle, gently, centered, natural, softly, stationary` | `0.3755` | **`0.6368`** | **`0.6370`** | **`0.5188`** (`+chaotic, frantic, panicked, desperate`) |
| `underwater_submerged` | `+underwater, submerged, aquatic, murky, floating, swimming, bubbles` / `-dry, indoor, outdoor, tidy, quiet, natural, clean` | — | **`0.6181`** | **`0.5608`** | **`0.4943`** (`+underwater, submerged, aquatic, murky`) |
| `cursed_cctv` | `+unstable, erratic, outdated, corrupted, surveillance, cctv, jittery, footage` / `-natural, realistic, detailed, crisp, cinematic, studio, sharp` | `0.2705` | **`0.5746`** | **`0.4998`** | **`0.3121`** (`+unstable, cctv, surveillance, footage`) |
| `film_16mm_1974` (`apply_holds=False`) | `+vintage, faded, dated, outdated, celluloid, grainy, older, decades` / `-cinematic, detailed, digital, modern, crisp, realistic, sharp` | `0.2741` | **`0.5626`** | **`0.5587`** | **`0.3553`** (`+vintage, faded, celluloid, grainy`) |
| `felt_muppet` | `+cartoon, puppet, animated, fuzzy, felt, plush, textile` / `-natural, human, realistic, skin, professional, steady` | `0.3224` | **`0.5463`** | **`0.4688`** | **`0.4478`** (`+puppet, felt, fuzzy, cartoon`) |
| `head_swell` | `+oversized, dome, spherical, inflated, swollen, balloon, enormous` / `-normal, proportional, consistent, natural, ordinary, standard` | — | **`0.5446`** | **`0.4806`** | **`0.4570`** (`+oversized, spherical, swollen, inflated`) |
| `crystallize_ice` | `+frost, frozen, crystal, crystalline, ice, shattered, snow, glacial` / `-warm, indoor, natural, soft, normal, studio` | — | **`0.5238`** | **`0.4591`** | **`0.3829`** (`+frost, frozen, crystal, ice`) |
| **Mean across 7 concepts** | — | `0.3106` | **`0.5724`** | **`0.5236`** | **`0.4240`** |

### 6.2 When to Use Contextual 10-Pair Compilation vs. `< 2 ms` Unigram Word Synthesis

1. **Primary Path — 10-Pair Contextual `Qwen3-VL` + `TokenRefiner` Compiler ([`h3_pair_compiler.py`](../../../../h3_pair_compiler.py), `bench/control_vectors.py build --words`, `~8 s` GPU, `basis = "post_refiner"`)**:
   - Slots short user contrasts (including multi-clause `;`-separated packs, `{props}`/`{surface}` bindings, and coupled `soundscape:` clauses) into 10 canonical H3 frames and runs full 50-layer `Qwen3-VL` causal attention + 2-block `TokenRefiner` self-attention (`diff_span`), preserving syntax, relational binding, trajectory direction, and cross-modal coupling.
2. **Exploratory / Surface-Attribute Path — Instant CPU Unigram Synthesis ([`j_lens.py`](../../../../j_lens.py) `synthesize_from_words` / `--from-words`, `< 2 ms` CPU, `0 GB` VRAM, `basis = "unigram_synthesis"`, `cos_5376 = 0.46–0.64`)**:
   - Synthesizes `vector.post_refiner` (`norm = 67.5`) and `vector.encoder` directly from a commutative unigram average in `M_centered` (plus optional `--against-words` and `--hold-words`). Because `synthesize_from_words` constructs vectors inside `row(M_centered)` and `decode_vector` projects onto the same `M_union` basis, its `top1_gain` (`+3.27 .. +5.17`) is an in-subspace self-projection and must not be compared directly against 50-layer contextual `Qwen3-VL` pair vectors (`+1.00 .. +2.63`; recorded in `assess_steering_readiness["basis_note"]`). Use it for quick `< 2 ms` surface-attribute overlays (`blizzard`, `disco`, `medieval`, `8-bit`) or inline `hold_words` scrubbing.
