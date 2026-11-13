# Assessment of Meta FAIR `e-means` (`arXiv:2611.09501v1`) for H3 and Sol-Attn

- **Date:** 2026-11-13
- **Verdict:** **Declined** for live attention (`MiniMaxH3Sol` and core `BlockSparseAttention`), linear/weight quantization (`int8_convrot`), VAE/tracking models, and current control-vector caches (`~131k` rows). **Revisit only if** an offline workload grows past `N >= 10^7` vectors per fit (for example multi-million-row corpus indexing or large-scale vector-quantized weight codebooks) where full-dataset Lloyd passes exceed memory or take minutes.
- **Primary sources reviewed:** Paper PDF (`arXiv:2611.09501v1`, 2026-11), project page (`https://e-means.github.io/`), and Faiss contrib code (`https://github.com/facebookresearch/faiss/tree/main/contrib/e_means`).

---

## 1. What `e-means` Actually Is (Mechanics, Scaling Laws, & Caveats)

`e-means` is an online GPU mini-batch $k$-means optimizer built to fit very large vector collections ($N \approx 10^6\text{--}10^8$, e.g., neural audio/video VQ-VAE codebooks and billion-scale IVF search indices) in fewer than one to a handful of passes.

### Core algorithm (`Algorithm 1`, pp. 5, 26–32)

1. **Dual-accumulator EMA state:** For $j \in \{1, \dots, k\}$, maintain scalar mass $N_j \in \mathbb{R}_+$ and weighted vector sum $S_j \in \mathbb{R}^d$ in FP32, initialized from random dataset points with $N_j^{(0)} = 10^{-10}$ and $S_j^{(0)} = N_j^{(0)} c_j^{(0)}$.
2. **Assignment and update per mini-batch $B$ ($M = T \lfloor n / B \rfloor$ steps):**
   - Assign points to the current centroids $c_j = S_j / N_j$ via $a(x) = \arg\min_j \|x - c_j\|_2^2$.
   - Accumulate per-cluster batch counts $n_j$ and batch sums $s_j$ in FP32.
   - Update both accumulators with a cosine-decayed step size $\alpha_t$ running from $\alpha_0$ down to $0.1 \alpha_0$ (Eq. 5):
     $$N_j \leftarrow (1 - \alpha_t) N_j + \alpha_t n_j, \qquad S_j \leftarrow (1 - \alpha_t) S_j + \alpha_t s_j$$
   - Because dividing $S_j / N_j$ always forms a convex combination of cluster means regardless of how empty or unbalanced a cluster was in batch $B$, $\alpha_t$ can stay stable all the way up to $\alpha_t = 1$ at large batch sizes where mini-batch SGD and Adam diverge past $1.0$ (Appendix E, pp. 26–27).
3. **Proportional donor-split revival (Appendix H.1, Eq. 8, p. 30):** Every $5$ batches, any cluster with $N_j < 0.1 \cdot \text{mean}(N)$ is revived by splitting the largest donor cluster $j'$ as $c_{j'} \odot [1 \pm \delta, 1 \mp \delta, \dots]$ (with $\delta =$ BF16 machine $\varepsilon$) and halving $N_{j'}$. On `DEEP 1M` ($k = 16,384$), this drops dead clusters from `26.3%` (`+2.97%` train / `+2.24%` test MSE without revival) and `14.3%` (VQ-VAE random replacement) down to `0.3%` (Table 1 p. 5, Table 14 p. 30).
4. **Casted-norm low-precision assignment (Appendix J.1, Eq. 9–11, p. 32):** Distances are scored as $\|\hat{c}_j\|_2^2 - 2 \hat{x}_i^\top \hat{c}_j$ using the *casted* centroid norm $\|\hat{c}_j\|_2^2$ rather than FP32 $\|c_j\|_2^2$. That eliminates quantization bias and keeps `FP8` / `INT8` within `0.13%` MSE of BF16 while $(N_j, S_j)$ remain in FP32.
5. **Finalizer choice (Table 15):** The paper switches finalizer by objective — `last-epoch-mean` for reconstruction/train MSE vs. `slow-EMA` for generalization/test MSE.
6. **Empirical scaling laws (Table 2, p. 6):**
   $$\alpha_0 = \min\!\left(0.824\, k^{-0.46} B^{0.90} n^{-0.45} T^{-0.11},\, 1\right), \qquad B = 0.035\, k^{0.13} n^{0.88} T^{0.81}$$
   ($B$ divided by $8$ when $n > 2\text{M}$, rounded to a power of $2$, capped at $n$).

### Caveats that govern fit to this repo

- **Empirical hyperparameter domain:** The scaling fits in Table 2 were calibrated on `DEEP` (`d = 96`) over $10^5 \le n \le 10^8$, $256 \le k \le 65,536$, and $n / k \ge 8$. Attention head dimensions here (`HEAD_DIM = 128`) and control-vector hidden dimensions (`d = 5120` / `5376`) sit well outside that calibration grid, and typical attention block counts have small $n / k$ (`64` per spatial-temporal block in Sol).
- **Cluster-size imbalance (`1.16–1.19x`, Table 14):** `e-means` intentionally lets cluster populations follow density rather than forcing equal sizes. As Appendix M.2 notes, ragged/variable-length cluster GEMMs require padding or splitting to avoid tensor-core waste.
- **Unwhitened directional data:** Euclidean $L_2$ centroids shrink toward the origin when matching queries/keys or cosine-like embeddings without explicit whitening or spherical normalization.
- **Kernel and dependency footprint on local hardware (`RTX 4090`, `sm_89`):** The repo venv runs on an NVIDIA RTX 4090 (`24564 MiB` VRAM, [`hardware.md`](../ops/hardware.md)) with `triton` available but `faiss`, `sklearn`, and `cutlass` absent. The paper's custom CuTe kernel targets H100 (`sm_90`); on RTX 4090 it falls back to Triton or a ~65-line pure PyTorch EMA + assignment loop.

---

## 2. Live H3 Attention and `MiniMaxH3Sol` Evaluation

The primary motivation for inspecting `e-means` was whether semantic token clustering (in the style of SVG2 / Tactic) could cut attention compute on long H3 renders. Measured repo profiles show the opposite: **clustering tokens adds work to the cheapest `0.5%` stage of Sol while colliding with Sol's fixed block layout, memory budget, and exact-GEMM kernel design.**

### Where sampler time actually goes

| Run profile | Total sampler time | Attention vs. non-attention breakdown | Inside `sol_attn` stage split | Source |
|---|---|---|---|---|
| Base 16-step `t2v` (`1344x768`, `345f`, `L = 104,361`, `56` heads) | `469 s` | `200` dense kitchen int8 calls = `143.4 s` (`719.6 ms` median); `600` Sol calls = `133.4 s` (`219.5 ms` median); `192 s` non-attention | `exact`: `0.92–0.94`; `preprocess`: `0.05–0.06`; `vtranspose`: `0.01`; `route` + tail: `< 0.005` | [`2026-09-17_sol_stage_profile.md`](../../../bench/results/2026-09-17_sol_stage_profile.md) |
| 6-step Pika `t2va` (`300` calls total) | `dense`: `302.3 s`; `shipped`: `192.8 s`; `s0_b49`: `180.3 s` (Pass 1) / `182.7 s` (Pass 2) | Linear fit vs. dense call count: `153.0 + 0.498 * N_dense` (Pass 1), `154.2 + 0.492 * N_dense` (Pass 2) | `s0_b49` runs `55` dense + `245` Sol calls | [`2026-11-11_sol_pika_panel.md`](../../../bench/results/2026-11-11_sol_pika_panel.md) |
| 6-step Pika `ref2va` (`300` calls total, `2048px` `<Picture 1>`) | `dense`: `358.9 s`; `shipped`: `260.9 s`; `s0_b49`: `249.7 s` | Linear fit: `225.2 + 0.446 * N_dense`; `+72.2 s` sparse shift from `2048px` ref rows under `exact_kv_and_all_rows` | Same `55` dense + `245` Sol split on `s0_b49` | [`2026-11-11_sol_pika_panel.md`](../../../bench/results/2026-11-11_sol_pika_panel.md) |

Two numbers settle the bottleneck question immediately:
- In [`SOLATTN.md`](../attention/SOLATTN.md), a representative `345f` capture has `N = 104,277` tokens (`102,816` video tokens = `98.6%`, `1,607` video blocks at `BLOCK_SIZE = 64`, `HEAD_DIM = 128`, `56` heads in [`sol_attn_h3.py`](../../../sol_attn_h3.py)).
- Sol's existing block router (`route` + tail) already costs `< 0.5%` of `sol_attn` (`< 1 ms` per call), while `exact` (the sparse block-GEMM itself) takes `92–94%` (`~205 ms` of the `219.5 ms` median). Even a `0 ms` router saves virtually nothing unless it reduces the number of selected blocks in `exact` without degrading video quality.

### Five structural blockers for `e-means` inside `MiniMaxH3Sol`

1. **Clustering FLOPs go the wrong way per call (and amortization breaks on short/Pika schedules):**
   - A single full distance pass assigning `N = 104,277` vectors per head (`56` heads, `D = 128`) against $k \approx 1,607$ centroids costs $2 \cdot H \cdot N \cdot k \cdot D \approx 2.40\text{ TFLOPs}$ (`~4.80 TFLOPs` for both $Q$ and $K$), before counting EMA updates or multiple mini-batch passes. Running full clustering inside every block costs orders of magnitude more than Sol's `< 0.5%` mean-pooled block router.
   - **Why cross-step amortization does not rescue it:** Pika only has `5` sparse steps after Step 0, and [`2026-09-19_sol_block_grouping.md`](../../../bench/results/2026-09-19_sol_block_grouping.md) (§G) already measured that frozen cross-step sparsity patterns (`LoSA`) lose to fresh per-step Sol routing at matched block cardinality (`p5` heads drift fast across denoising steps).
2. **Global token permutation collides with `24 GB` VRAM (`RTX 4090`), early-free, and the memory compiler:**
   - A single `[1, 104277, 56, 128]` tensor in BF16 occupies `1.49 GB` (`1.39 GiB`), so $Q, K, V, O$ together total `~6.0 GB`. Reordering $Q, K, V$ into cluster order out-of-place and scattering $O$ back requires duplicate full-sequence buffers (`~12 GB` live footprint on `24564 MiB` total VRAM), breaking `AttentionTensorContainer` early-free ([`SOLATTN.md`](../attention/SOLATTN.md)) unless pushed outside the block loop.
   - Moving token permutation out to the embedder and final layer is the exact architectural workaround [`2026-09-18_sol_reorder_under_memory_compiler.md`](../../../bench/results/2026-09-18_sol_reorder_under_memory_compiler.md) needed just to test static `3d` Morton ordering under the memory compiler—a global static permutation that cannot track per-block, per-head dynamic semantic clusters without allocating inside every layer.
3. **Unbalanced clusters (`1.16–1.19x`) break fixed `BLOCK_SIZE = 64` block-sparse tiling:**
   - Sol's fast CUDA kernels (`BLOCK_SIZE = 64`, `exact` `92–94%` of call time) rely on uniform contiguous blocks aligned with tensor-core MMA tiles. Variable-size `e-means` clusters (Table 14 shows `1.16–1.19x` population imbalance even on isotropic `DEEP` vectors) either leave ragged partial tiles or force padded/split batched GEMMs (Appendix M.2), inflating the block count inside `exact`.
   - [`2026-09-19_sol_block_grouping.md`](../../../bench/results/2026-09-19_sol_block_grouping.md) (§B, §D, §F) measured this exact dynamic on query/key splitting (`split4`): splitting queries without changing layout inflated the CTA tile union by `1.20–1.48x` and eroded sparsity gains.
4. **Centroid-washout fixes already exist without re-clustering or permuting tokens:**
   - Reordering tokens globally via `3d` Morton already failed a 6-scene blind panel and stays OFF (`token_routing="off"` in [`workflows/h3_config.py`](../../../workflows/h3_config.py) `SOL_RECOMMENDED_CUDA`; [`2026-09-18_sol_reorder_panel.md`](../../../bench/results/2026-09-18_sol_reorder_panel.md); closed in [`morton.md`](../archive/morton.md)).
   - Meanwhile, [`2026-09-19_sol_block_grouping.md`](../../../bench/results/2026-09-19_sol_block_grouping.md) (§B, §D, §F) showed that `max4` / `lse4` sub-mean key summaries inside existing `64`-token blocks cure centroid washout and beat the shipped centroid router at equal sparse budget with **zero** token layout change (analysis in [`analyze_sol_block_grouping.py`](../../../bench/analyze_sol_block_grouping.py)).
5. **Quantization accuracy does not need `e-means`:**
   - `quantizer="rotated"` (`int8_convrot`) already reduces quantization error by `-11.7%` vs. plain and `-10.6%` vs. balanced at `< 1%` (`x1.008–x1.010`) kernel overhead ([`2026-09-27_sol_redesign_test2.md`](../../../bench/results/2026-09-27_sol_redesign_test2.md); policy in [`h3_quant_policy.md`](../attention/h3_quant_policy.md)).

### Higher-leverage Sol speed levers already on disk

Instead of dynamic clustering, three local levers target the `92–94%` `exact` compute and Step-0 dense budget directly:

1. **Step-0 sparse sweep (`start_percent = 0.0` vs. `0.2` on 6-step Pika / FlashGen):** In [`2026-11-11_sol_pika_panel.md`](../../../bench/results/2026-11-11_sol_pika_panel.md), `s0_b49` still spends `55` of `300` calls in dense attention (Step 0's `49` non-block-49 layers plus dense block `49` across all `6` steps). Linear extrapolation from `N_dense = 55` down to `6` (dropping Step 0's `49` dense blocks while keeping block `49`) predicts `~24.4 s` savings on `t2va` Pass 1 (`49 * 0.498 s`) and `~21.9 s` on `ref2va` (`49 * 0.446 s`). An earlier PDD8-FlashGen finish blind test ([`2026-10-01_start_percent_panel.md`](../../../bench/results/2026-10-01_start_percent_panel.md)) found `5/5` `"can't tell"` between `0.0` and `0.2` with `~1/5` faster sampling, while noting that 1-dense-step FlashGen/Pika and `ref2va` remain untested by blind panel.
2. **Segment-aware `tau_table` (`rows="per segment"`, `min_tokens=12288`):** Per-segment `tau` tuning via [`sol_tau_sweep.py`](../../../sol_tau_sweep.py) and [`calibrate_sparse_table.py`](../../../bench/calibrate_sparse_table.py) ([`SOLATTN.md`](../attention/SOLATTN.md)) shrinks `exact` block cardinality directly on reference/video sub-segments.
3. **Drop-in `max4` / `lse4` routing (`dense_blocks="49"` retained):** Reached from [`2026-09-19_sol_block_grouping.md`](../../../bench/results/2026-09-19_sol_block_grouping.md) without modifying data layout or breaking memory-compiler / early-free invariants.

---

## 3. Offline Repo Touchpoints & When to Revisit

We also checked every offline or non-attention component in the repository. None needs `e-means` today:

| Subsystem | Observed scale / bottleneck | Why `e-means` adds nothing (or breaks assumptions) | Owning file / doc |
|---|---|---|---|
| **Control-vector activation row caches** | ~`131k` rows (e.g. `130,963` rows across `765` prompts in the `2026-11-09` row cache), `d = 5120` / `5376` | Tiny relative to `N >= 10^7` (even `k = 64` full-dataset BF16 assignment is `~0.09 TFLOPs`); the real precondition is excluding token-0's `52.37x` sink norm outlier (`15,309.0` vs. `292.3` `tok1+`) and whitening before Euclidean clustering | [`bench/control_vector_rows.py`](../../../bench/control_vector_rows.py), [`bench/control_vector_lexicon.py`](../../../bench/control_vector_lexicon.py), [`2026-11-06_j_lens_and_refiner_jacobian.md`](control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md) |
| **Grouped readout scoring & `J++` clustered Jacobians** | Word-level readout scoring (`GROUP_SIZE`) and sparse Jacobian layer groupings (`j_lens.py`) | Bottleneck is token-refiner semantic alignment (`cos ~0.8355/0.8359` at `beta = 1.00`) and late-layer signal capture, not mini-batch clustering throughput | [`bench/control_vector_readout_score.py`](../../../bench/control_vector_readout_score.py), [`j_lens.py`](../../../j_lens.py), [`2026-11-12_chain_lens_digest_and_experiment_plan.md`](control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md) |
| **Masked V2V subject tracking (`SAM 3.1`)** | ViTDet trunk signature collapse in crowds / wardrobe shifts (`0.72–0.79` cosine scores across cuts) | This is representation collapse in the frozen trunk features across scene boundaries, not an optimizer or $k$-means initialization problem | [`subject_track.py`](../../../subject_track.py), [`sam3_prompting.md`](../masked_v2v/sam3_prompting.md) |
| **Model & VAE quantization** | Scalar `int8_convrot` linear/attention quant (`Tier 3` 4-bit linear bake) and continuous video/audio VAEs | Continuous VAEs use no VQ codebook, and current quantization uses per-channel / rotated scalar quantization rather than VQ codebooks | [`h3_quant_policy.md`](../attention/h3_quant_policy.md) |

### When to pull `e-means` off the shelf

Revisit `e-means` (written directly as a ~65-line pure PyTorch EMA + donor-split + casted-norm assignment loop—no `faiss` wheel needed) only if a future repo task enters the $N \ge 10^7$ regime where full-dataset Lloyd stalls:
- Multi-million-row offline prompt/activation/patch corpus indexing or IVF partitioning.
- An offline `Tier 3` AQLM/VQ weight-codebook bake (`2–4`-bit multi-dimensional weight codebooks across 50+ transformer blocks) if scalar 4-bit quantization hits a quality floor.
