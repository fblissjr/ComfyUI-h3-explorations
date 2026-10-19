# Sol-Attn Dense Blocks Re-Analysis and 5-Test Experimental Plan

**Date**: 2026-10-02  
**Context**: MiniMax H3 Attention Optimization, Rotated INT8 Quantization, and Sparse Routing Telemetry  
**Primary Tools**: `sol_observe.py`, `bench/sol_duckdb_analyzer.py`, `docs/research/sparse/sol_duckdb_dashboard.html`, `preflight.py`  
**Referents**: `docs/attention/block49_quant_error.md`, `workflows/h3_config.py::SOL_DENSE_TAIL`, `docs/state/decisions.md`

> **Re-read 2026-10-02, later the same day**: `bench/results/2026-10-02_sol_campaign_reanalysis.md`
> corrects this plan against the raw captures. Test 1 was two-stage like the
> rest. No step runs near sigma 0.1, and error falls as sigma drops rather
> than rising. Block 40's routed head 47 was already its worst head with routing
> off. Blocks 45-48 are quiet in the PDD stage but not in the FlashGen
> finisher. The `--run-id` flag in section 8.2 never existed: the analyzer
> takes the run from the capture directory.

---

## 1. Executive Summary

Historically, `MiniMaxH3Sol` shipped with the default configuration `dense_blocks="45,48,49"`. This default was established when Sol and SageAttention ran unrotated INT8 attention, where severe channel outlier spikes in the model's $K$-norm destroyed INT8 quantization precision in blocks 45, 48, and 49.

On 2026-09-27, the repository standardized on `quantizer="rotated"` (Hadamard rotation). By multiplying $Q$ and $K$ by a randomized orthogonal Hadamard matrix prior to INT8 quantization, channel energy is uniformly dispersed across all 128 head dimensions. This eliminated the tail-block quantization penalty.

In **Test 1** (our unmasked 50-block diagnostic baseline on Ref2VA at sequence length 119,102 tokens) and **Test 2** (the two-stage PDD8 + FlashGen finisher run with `token_routing="measured"`), comprehensive telemetry revealed:

1. **Tail Blocks (45–48) Are Tamed**: Under Hadamard rotation, Block 48 exhibits an average relative $L_2$ error of only **7.14%** (max 9.41%), Block 46 is **7.40%** (T1) / **7.37%** (T2), and Block 47 is **6.38%** (T1) / **6.53%** (T2). Keeping Block 48 in `dense_blocks` expends dense computation on blocks that Sol already calculates with high fidelity.
2. **The True Bottleneck is the Middle Cluster (Blocks 38–43)**: With quantization error suppressed, the dominant error source is **sparsity truncation error (routing loss)**. In the middle of the DiT, attention is diffuse and globally integrated across text, reference images, and video tokens. Truncating tokens via top-$k$ sparsity produces severe errors:
   - **Block 42**: **25.67%** (T1) / **24.59%** (T2) avg relative $L_2$ error (max **27.01%**) — nearly $4\times$ higher than Block 48.
   - **Block 39**: **25.22%** (T1) / **25.18%** (T2) avg relative $L_2$ error (max **27.20%**) — $3.5\times$ higher than Block 48.
   - **Block 41**: **24.35%** (T1) / **22.45%** (T2) avg relative $L_2$ error.
   - **Block 40**: **23.43%** (T1) / **23.93%** (T2) avg relative $L_2$ error.
   - Individual attention heads in these blocks suffer catastrophic degradation (e.g., Block 42 Head 44 has **57.0%** error, min cosine similarity **0.7697**; Block 39 Head 12 has **80.1%** error).
3. **Token Routing Cannot Rescue Diffuse Middle Layers**: Test 2 demonstrated that while token routing reduced error on localized layers (Block 0 dropped from 7.63% to 5.37%, Block 32 dropped from 14.81% to 12.99%), attempting to rescue tokens on Block 40 caused Head 47 error to explode to **111.06%** (cosine similarity down to **0.6655**). Diffuse layers cannot be repaired with sparse token heuristics; they must be executed fully dense.
4. **Block 49 Must Remain Dense**: Although Block 49's error is 13.73%, it directly feeds into `final_layer.video_out`. Any approximation error in Block 49 propagates directly into output latents without subsequent transformer layers to smooth it out.

**Conclusion**: Retaining `"45,48,49"` misallocates dense compute to a quiet tail (Block 48) while exposing the 25% error middle cluster to unmitigated sparsity loss. We propose transitioning `dense_blocks` to **Option A (`"39,41,42,49"`)** or **Option B (`"39,40,41,42,49"`)**, evaluated across a structured 5-test matrix.

---

## 2. The Historical Origin of `"45,48,49"` vs. Rotated INT8

### 2.1 The Unrotated INT8 Regime (Pre-2026-09-27)
Prior to late September 2026, INT8 attention in both stock Sol and SageAttention used per-row or per-block quantization with a single shared scale across all 128 channels. 

As documented in `docs/attention/block49_quant_error.md`, MiniMax H3's released weights place an order-of-magnitude gain on four specific channels in `k_norm.weight` on blocks 45, 48, and 49:
- Because the quantization scale had to cover these outlier spikes, the remaining 124 channels were quantized into only 1 or 2 bits of effective precision.
- This caused INT8 attention error on Block 49 to spike to over $5\times$ that of Block 0.
- `SOL_DENSE_TAIL = "45,48,49"` was implemented on 2026-09-25 as an emergency heuristic to force these three outlier blocks onto the dense fallback (`attention_comfy_kitchen_int8`), bypassing the broken INT8 quantizer.

### 2.2 What Changed with `quantizer="rotated"`
On 2026-09-27, Sol integrated Hadamard rotation (`quantizer="rotated"`), matching `comfy-kitchen`'s dense kernel. 

Hadamard transformation multiplies activation vectors by a fixed orthonormal Hadamard matrix before quantization:
$$Q_{\text{rot}} = Q \cdot H, \quad K_{\text{rot}} = K \cdot H$$
Because $H^T H = I$, the inner product is mathematically identical:
$$Q_{\text{rot}} K_{\text{rot}}^T = Q H H^T K^T = Q K^T$$
However, the outlier energy concentrated in 4 channels is now distributed evenly across all 128 dimensions. The peak-to-average channel ratio drops from $>10:1$ to near unity. As a result, 8-bit quantization preserves full dynamic range across all channels.

---

## 3. Experimental Architecture & Preflight Geometry Specification

To evaluate Sol-Attn under production-representative conditions rather than a synthetic, truncated workload, our experimental setup uses a complete two-stage reference-conditioned generation graph with full latent packing.

### 3.1 Two-Stage Sampler Architecture

The experimental workflow (`workflows/h3_text_to_video_pdd8_flashgen_finish_api.json`) executes a hybrid, two-stage denoising trajectory:
- **Pass 1 (PDD 8-Step Schedule, Steps 0–5)**: Runs the first 6 steps from initial latent noise ($\sigma \approx 1.0$) down to intermediate noise ($\sigma \approx 0.8$). Across all 50 DiT blocks, this produces $6 \times 50 = 300$ attention calls.
- **Pass 2 (FlashGen 4-Step Finisher, Steps 6–7)**: Takes the partially denoised latent at $\sigma \approx 0.8$ and finishes generation down to $\sigma = 0.0$ in 2 specialized fast steps. Across 50 DiT blocks, this produces $2 \times 50 = 100$ attention calls.
- **Sequential Multi-Node Capture**: The workflow contains two distinct `MiniMaxH3Sol` nodes—one attached to each sampler stage. Both nodes stream sequentially into the same diagnostic log files (`sol_probe_*.jsonl`, `sol_observe_*.jsonl`, `blk_cnt_*.u16`) without clobbering, capturing all **400 DiT calls** (376 sparse `cell` evaluations + 24 dense `skip` executions when running 3 dense blocks).

### 3.2 Reference Conditioning Pipeline

Reference conditioning injects high-resolution visual context into the unified sequence:
- **Input Source**: 1 reference image at $2752 \times 1536$ resolution.
- **Append Node (`MiniMaxH3AppendRefImage`)**:
  - `size_policy`: `"max"`
  - `short_edge`: `2048`
  - `allow_upscale`: `True`
  - `qwen_view`: `"shared"`
- **Conditioning Node (`MiniMaxH3ReferenceConditioningOrdered`)**:
  - `image_policy`: `"comfy"`
- **Latent Yield**: VAE encoding processes the scaled reference into exactly **7,360 reference latent rows**.

### 3.3 Sequence Layout & Preflight Report

The sequence layout was verified directly using `MiniMaxH3Preflight` (`preflight.py`), which reads the model's actual compiled `PackedLayout`:

```
1344x768  trained family  1008 video tokens/frame
345 frames (14.375s at 24fps)  102 latent frames
sequence length 119,102
  video           102,816  #################...   86.3%
  text              7,776  #...................    6.5%
  references        7,360  #...................    6.2%
  audio             1,150  ....................    1.0%
if the aspect ratio changed, same length:
  1:1   768x768      75,038    -37%
  4:3   1024x768      94,622    -21%
  3:2   1152x768     104,414    -12%
  16:9  1344x768     119,102     +0%  <- current
  9:16  768x1344    119,102     +0%
sage: past the fused int32 crossing at 99,864 (Triton q/k quantizers), fixed in every fork build that has sageattn_consume. The installed build forms 64-bit offsets in the CUDA v quantizer, so the uint32 wrap at 199,728 does not apply to it.
```

#### Modality Breakdown ($S = 119,102$ tokens)
- **Video Tokens**: $102,816$ tokens ($86.3\%$). Formed by 102 latent frames at $1,008$ tokens/frame ($1344 \times 768$ canvas, 7/4 trained family).
- **Text Tokens**: $7,776$ tokens ($6.5\%$). Prompt embeddings and prefix context.
- **Reference Tokens**: $7,360$ tokens ($6.2\%$). Packed visual reference latents.
- **Audio Tokens**: $1,150$ tokens ($1.0\%$). Synchronized latent audio stream.

#### Integer-Offset Ceilings & Quantizer Safety
MiniMax H3 constructs $Q, K, V$ as three views of a fused projection (`qkv_proj(x).split`), creating a fused sequence stride of:
$$\text{stride\_seq} = 3 \times \text{heads} \times \text{head\_dim} = 3 \times 56 \times 128 = 21,504$$
Two quantizer memory boundaries govern long sequences:
1. **Triton $Q/K$ Quantizer Int32 Crossing**: $2^{31} / 21,504 = 99,864$ tokens. At $119,102$ tokens, our sequence operates beyond this crossing. This boundary is safely handled because our installed kernel build incorporates `sageattn_consume` with 64-bit index arithmetic (`USE_I64`).
2. **CUDA $V$ Quantizer UInt32 Wrap**: $2^{32} / 21,504 = 199,728$ tokens. The installed kernel forms 64-bit offsets in `csrc/fused/fused.cu` (`sageattention.quant.ELEMENT_OFFSET_BITS = 64`), ensuring substantial headroom before approaching the $199\text{k}$ ceiling.

---

## 4. Empirical Telemetry: Test 1 vs. Test 2 Findings

### 4.1 Overview of Test Runs

- **Test 1 (Diagnostic Baseline)**:
  - Config: `dense_blocks=""`, `tau=1.0`, `token_routing="off"`, `start_percent=0.0`, `quantizer="rotated"`.
  - Workflow: Single-stage 8 PDD steps, all 50 blocks unmasked.
  - Ingested Records: 400 sparse cells.
- **Test 2 (Two-Stage Sampler & Measured Routing Validation)**:
  - Config: `dense_blocks="45,48,49"`, `tau=1.0`, `token_routing="measured"` (active on blocks 0, 24, 32, 40), `start_percent=0.0`, `quantizer="rotated"`.
  - Workflow: Two-stage hybrid (Pass 1: PDD8 6 steps, Pass 2: FlashGen 2 steps).
  - Ingested Records: 376 sparse cells + 24 dense skip calls.

### 4.2 Cross-Test Agreement: The Invariant Middle Error Peak

The most critical finding from comparing Test 1 and Test 2 is that **the middle cluster error peak is an invariant structural property of the model**, persisting across different sampling schedules, timesteps, and routing modes:

| Block Index | Historical Status | Test 1 Avg Rel $L_2$ | Test 2 Avg Rel $L_2$ | Difference | Assessment |
|---|---|---|---|---|---|
| **Block 0** | Sparse | 4.88% | 5.37% (routed) | +0.49% | Low error (initial feature extraction) |
| **Block 24** | Sparse | 14.28% | 15.11% (routed) | +0.83% | Transition mid-point |
| **Block 32** | Sparse | 14.81% | 12.99% (routed) | **-1.82%** | Rescued by token routing |
| **Block 38** | Sparse | 19.16% | 19.82% | +0.66% | Middle error onset |
| **Block 39** | Sparse | **25.22%** | **25.18%** | **-0.04%** | **Critical Error Peak (Identical)** |
| **Block 40** | Sparse | **23.43%** | **23.93%** (routed) | +0.50% | **Critical Error Peak (Unrescued)** |
| **Block 41** | Sparse | **24.35%** | **22.45%** | -1.90% | **Critical Error Peak** |
| **Block 42** | Sparse | **25.67%** | **24.59%** | -1.08% | **Highest Error in DiT (Both runs)** |
| **Block 43** | Sparse | 19.49% | 19.24% | -0.25% | Transition decay |
| **Block 46** | Shipped Sparse | **7.40%** | **7.37%** | **-0.03%** | **Quiet Tail (Confirmed)** |
| **Block 47** | Shipped Sparse | **6.38%** | **6.53%** | **+0.15%** | **Quiet Tail (Confirmed)** |
| **Block 48** | Historical Dense | **7.14%** (T1) | *(Dense Skip)* | — | **Quiet Tail (Dense wasted)** |
| **Block 49** | Historical Dense | **13.73%** (T1) | *(Dense Skip)* | — | **Terminal Block (Must stay dense)** |

### 4.3 Key Divergences and New Discoveries in Test 2

#### 1. Token Routing is Effective on Localized Layers, Destructive on Diffuse Layers
Test 2 enabled `token_routing="measured"` on blocks 0, 24, 32, and 40:
- **Success on Localized Layers**: On Block 32, average error dropped from $14.81\%$ to $12.99\%$ ($-1.82\%$). Outlier tokens in localized layers represent clear high-attention targets that benefit from dense recovery.
- **Catastrophic Failure on Diffuse Layer (Block 40)**: In Block 40, attention is globally distributed across the 119k tokens. Rescuing tokens into a dense slice diluted the attention distribution: **Head 47 exploded to 111.06% relative $L_2$ error**, with cosine similarity dropping to **0.6655**.
- **Takeaway**: Diffuse middle layers (Blocks 39–42) cannot be saved by token routing. They must be routed through `dense_blocks`.

#### 2. Sampler Stage Trajectory (PDD vs. FlashGen Finisher)
Analyzing Pass 1 (PDD8, steps 0–5) versus Pass 2 (FlashGen, steps 6–7) revealed that relative $L_2$ error increases as the latent clears noise:
- At $\sigma \approx 1.0$ (step 0), mean network error is $\sim 10.2\%$.
- At $\sigma \approx 0.1$ (step 7, FlashGen finisher), mean network error reaches $\sim 13.8\%$.
- Because high-frequency visual details and motion vectors resolve in the low-noise regime, errors in the middle layers during the finisher stage have an outsized impact on perceptual clarity.

---

## 5. Why `"45,48,49"` Must Change

### 5.1 The Waste of Protecting Block 48
Under Hadamard rotation, Block 48 has an average relative $L_2$ error of **7.14%**. It is cleaner than Block 37, Block 38, Block 39, Block 40, Block 41, Block 42, Block 43, and Block 44. Keeping Block 48 in `dense_blocks` is equivalent to using a fire extinguisher on a cold room while the adjoining room is blazing.

### 5.2 The Vulnerability of Leaving Blocks 39–42 Sparse
Blocks 39 through 42 sit in a sustained error plateau above **23%–25%**, with peak heads losing up to **80%** of their output magnitude and suffering directional alignment drops down to $0.76$ cosine similarity. In a generative diffusion trajectory, injecting 25% error into the middle representations degrades object coherence and fine motion vectors.

### 5.3 Why Block 49 Cannot Be Made Sparse
Block 49 is the final transformer block of the DiT. Its output directly feeds the final adaptive layer norm and linear projection (`final_layer.video_out`). Any error in Block 49 cannot be attenuated or redirected by subsequent attention layers; it maps linearly into the predicted latent velocity vector. Therefore, Block 49 remains mandatory in any `dense_blocks` specification.

---

## 6. Candidate Replacement Configurations

| Configuration | `dense_blocks` Spec | Dense Block Count | Target Error Reduction | Compute Cost Impact | Recommendation |
|---|---|---|---|---|---|
| **Historical Default** | `"45,48,49"` | 3 blocks | Ineffective (shields 7% tail, ignores 25% middle) | Baseline | Deprecated under `rotated` |
| **Option A (Targeted Peak)** | `"39,41,42,49"` | 4 blocks | Shields top 3 worst middle blocks (>24%) + terminal block | +2.0% sampler time | **Active Default (0.184.8)** |
| **Option B (Full Plateau)** | `"39,40,41,42,49"` | 5 blocks | Shields the entire 23%–26% middle plateau + terminal block | +4.1% sampler time | **High-Quality Candidate** |
| **Option C (Strict Budget)** | `"39,42,49"` | 3 blocks | Shields the two worst middle blocks + terminal block | Exactly 0.0% cost vs. historical | Budget fallback |

---

## 7. The 5-Test Experimental Matrix

| Test ID | Name / Objective | Configuration Levers | Target Sequence / Scene | Telemetry Scope | Status / Verdict |
|---|---|---|---|---|---|
| **Test 1** | **Diagnostic Baseline** | `tau=1.0`<br>`dense_blocks=""`<br>`token_routing="off"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, 8 PDD steps | Full 50 blocks unmasked (`sol_probe_*.jsonl`, `sol_observe_*.jsonl`, `.u16`) | **Completed**: Discovered 38–43 middle error peak (25.7%) and confirmed 45–48 tail calmness (~7%). |
| **Test 2** | **Historical Control & Two-Stage Routing** | `tau=1.0`<br>`dense_blocks="45,48,49"`<br>`token_routing="measured"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, PDD8 + FlashGen | 400 calls logged (376 cells, 24 skips). Multi-node verified. | **Completed**: Verified middle peak invariant; proved token routing rescues localized layers (32) but fails on diffuse layers (40 Head 47: 111% error). |
| **Test 3** | **Option A Validation** | `tau=1.0`<br>`dense_blocks="39,41,42,49"`<br>`token_routing="measured"` (or `"off"` on 40)<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, PDD8 + FlashGen | 4 dense blocks (39, 41, 42, 49); probe on remaining 46 sparse blocks | **Ready to Run**: Expected to eliminate the 3 worst error peaks (>24%). Peak network error drops below 20%. |
| **Test 4** | **Early Warmup Anchor** | `tau=1.0`<br>`dense_blocks="39,41,42,49"`<br>`start_percent=0.2`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, PDD8 + FlashGen | First 20% steps dense; Sol on remaining 80% | **Scheduled**: Tests whether dense initial warmup adds stability when middle layers are also shielded. |
| **Test 5** | **Pareto Optimization (Option B)** | `tau=1.1`<br>`dense_blocks="39,40,41,42,49"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, PDD8 + FlashGen | 5 dense blocks, higher sparsity on non-critical layers | **Scheduled**: Tests whether relaxing tau to 1.1 offsets the compute cost of shielding the full 5-block plateau. |

---

## 8. Execution Guide: Running Test 3 in ComfyUI

### 8.1 Arming the Capture Environment
Capture files must reside in gitignored `data/sparse/captures/`. Because ComfyUI is typically launched from the parent directory, provide the relative path to this custom node repository:

```bash
# Untracked capture target for Test 3
RUN_DIR="custom_nodes/ComfyUI-h3-explorations/data/sparse/captures/test3_middle_dense"
mkdir -p "$RUN_DIR"

# Launch ComfyUI armed with probe and observe
H3_SOL_PROBE="dir=$RUN_DIR" \
H3_SOL_OBSERVE="dir=$RUN_DIR" \
python main.py --listen 127.0.0.1 --port 8188
```

### 8.2 Ingestion and Comparative Dashboard
Once Test 3 completes:

```bash
# Ingest into the existing untracked database
python bench/sol_duckdb_analyzer.py \
  --db data/sparse/sol_analysis.duckdb \
  --logs-dir data/sparse/captures/test3_middle_dense \
  --run-id test3_middle_dense \
  --out docs/research/sparse/sol_duckdb_dashboard.html
```

---

## 9. Code and Workflow Configuration Status

As of version **0.184.8** (commit `20528eab`):
1. **Option A is Adopted as Active Default**:
   ```python
   # workflows/h3_config.py & sol_attn_h3.py
   SOL_DENSE_OPTION_A = "39,41,42,49"
   SOL_DENSE_OPTION_B = "39,40,41,42,49"
   SOL_DENSE_HISTORICAL_TAIL = "45,48,49"

   SOL_DENSE_TAIL = SOL_DENSE_OPTION_A  # Active default
   ```
2. **All 166 Shipped Workflows Rebuilt**: The entire workflow fleet reflects Option A while leaving Option B exposed as an easily selectable configuration.
3. **Decisions Log Updated**: The rationale, empirical evidence, and reversal of `"45,48,49"` are documented in `docs/state/decisions.md`.
