# MiniMax H3 Special Tokens, Multimodal Prompting, Dataset Architecture & Post-Training Proposal

**Date:** 2026-08-24
**Author:** Antigravity (Gemini) — *Working Draft for Technical Lead & Codex Review*
**Location:** [`docs/research/qwen3-vl-special-tokens-post-training/brainstorming/master_post_training_blueprint.md`](master_post_training_blueprint.md)
**Status:** Working Research Proposal (Not an Authoritative Specification)

> **Authority notice:** Shared facts and accepted work live in
> [`../canonical/`](../canonical/README.md). This proposal must not override a
> conflicting canonical statement.

---

## 1. Executive Summary & The Initialization Dilemma

The official MiniMax H3 release introduces seven custom tokens into the Qwen3-VL tokenizer (`tokenizer_config.json`, token IDs `151669`–`151675`):

```json
{
  "<d>": 151669,
  "</d>": 151670,
  "<|cutoff|>": 151671,
  "<|lyrics_start|>": 151672,
  "<|lyrics_end|>": 151673,
  "<|caption_start|>": 151674,
  "<|caption_end|>": 151675
}
```

### 1.1 Established Weight Facts
Our empirical weight audit ([`official_weights_metadata.md`](../../official_weights_metadata.md)) confirmed that in the shipped Qwen3-VL 32B weights:
* **Rows `151669`–`151675` in `embed_tokens.weight` are 1:1 identical to base Qwen3-VL unallocated padding rows.**
* They carry identical vector norms ($\approx 0.50498$) and base random initializations.
* They were not updated by MiniMax during base language model pre-training.

### 1.2 The Downstream Grounding Hypotheses
How downstream video generation interacts with these tokens remains an open research question:
* **Hypothesis 1 (Direct DiT Grounding):** MiniMax froze Qwen3-VL 32B during DiT training and fed prompts containing token IDs `151669`–`151675`. The DiT's cross-attention layers learned what those specific Layer 50 activation vectors mean, even though the embedding rows began as arbitrary unallocated vectors.
* **Hypothesis 2 (BPE Discrepancy):** MiniMax's internal training pipeline used an unpatched tokenizer emitting standard BPE fragments (e.g. `<, d, >`), leaving dedicated token IDs unsupported by the DiT.
* **Hypothesis 3 (Delimiter Inertness):** The DiT conditioning relies primarily on surrounding prose semantics, rendering boundary markers largely redundant.

*(Note: The owner's observation that caption-tagged prompts work behaviorally confirms a functional effect for caption markers in practice, but does not establish the training history or utility of all seven tokens).*

> [!NOTE]
> **Objective Mismatch of Generic Language Model (LM) Loss:**
> Training Qwen with standard causal next-token cross-entropy (`CrossEntropy(next_token)`) optimizes language modeling perplexity without a direct video/audio diffusion loss signal. This carries an unmeasured risk of shifting Layer 50 representations away from the distribution expected by the downstream DiT.

---

## 2. Inferred & Intended Token Semantics

The following table reflects inferred downstream conditioning roles based on token naming and prompt guide conventions:

| Token ID | Token String | Inferred Conditioning Role | Intended Cross-Attention Modulation (Hypothesized) |
| :--- | :--- | :--- | :--- |
| `151669` | `<d>` | Spoken dialogue stream initiator | Guides character mouth articulation and speaker-specific attention. |
| `151670` | `</d>` | Spoken dialogue stream terminator | Marks transition from speech back to general ambient narrative. |
| `151671` | `<|cutoff|>` | Temporal boundary marker | Intended to limit attention bleed on truncated or variable-length utterances. |
| `151672` | `<|lyrics_start|>` | Musical lyric marker start | Intended to signal sung vocal synthesis in audio cross-attention. |
| `151673` | `<|lyrics_end|>` | Musical lyric marker end | Intended to transition audio attention back to background instrumentation. |
| `151674` | `<|caption_start|>` | Onscreen title / caption marker | Inferred to bind text tokens to subtitle / graphic rendering. |
| `151675` | `<|caption_end|>` | Onscreen caption closure | Inferred to separate title formatting from scene narrative. |

---

## 3. Verified Corpus Inventory & Paths (2,670 Prompts)

The candidate prompt corpus consists of **2,670 production prompts** across two primary sources:

```
                            VERIFIED DATASET TOPOLOGY
┌────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 1. StellarVoyager/H3-IR (1,110 Prompts)                │ 2. local_extracted/malcolmrey_various (1,560 Prompts)  │
├────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ • Focus: Full-Reference Ref2VA, Multi-Image Anchors    │ • Focus: Production Dialogue, Rich Foley Soundscapes   │
│ • HF: datasets/StellarVoyager/H3-IR                    │ • Source: Extracted ComfyUI Workflows & MP4 Containers │
│ • Local JSONL: data/train.jsonl                        │ • Local JSONL: h3_extracted_metadata.jsonl             │
│ • Local Images: media/images/                          │ • Extractor Script: extract_h3_metadata.py             │
└────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

### Exact Dataset Paths on Disk:

1. **`StellarVoyager/H3-IR` (1,110 Rows):**
   * **Hugging Face Hub:** [`https://huggingface.co/datasets/StellarVoyager/H3-IR`](https://huggingface.co/datasets/StellarVoyager/H3-IR)
   * **Local JSONL Cache:** [`<HOME>/.cache/huggingface/hub/datasets--StellarVoyager--H3-IR/snapshots/460db3256f19dc70d0def2068a22e6e0dca87e8e/data/train.jsonl`](<HOME>/.cache/huggingface/hub/datasets--StellarVoyager--H3-IR/snapshots/460db3256f19dc70d0def2068a22e6e0dca87e8e/data/train.jsonl)
   * **Local Reference Images:** [`<HOME>/.cache/huggingface/hub/datasets--StellarVoyager--H3-IR/snapshots/460db3256f19dc70d0def2068a22e6e0dca87e8e/media/images/`](<HOME>/.cache/huggingface/hub/datasets--StellarVoyager--H3-IR/snapshots/460db3256f19dc70d0def2068a22e6e0dca87e8e/media/images/)

2. **`local_extracted/malcolmrey_various` (1,560 Rows):**
   * **Local Root Directory:** [`/mnt/hub/ai/data/malcolmrey_various/`](/mnt/hub/ai/data/malcolmrey_various/)
   * **Extracted JSONL:** [`/mnt/hub/ai/data/malcolmrey_various/h3_extracted_metadata.jsonl`](/mnt/hub/ai/data/malcolmrey_various/h3_extracted_metadata.jsonl) (1,560 rows, 3.68 MB)
   * **Extractor Script:** [`/mnt/hub/ai/data/malcolmrey_various/extract_h3_metadata.py`](/mnt/hub/ai/data/malcolmrey_various/extract_h3_metadata.py)

3. **`oakmindai/minimax_h3_avatar_500` (Deprecated Reference Corpus):**
   * **Status:** Deprecated from calibration due to narrow portrait distributions and lack of multi-shot / retention metadata.

---

## 4. Official MiniMax H3 Prompt Specifications & Syntax Rules

Conditioning presentation is defined by two governing vendor guides: **Full-Reference Mode (`Ref2VA`)** ([`ref_en.md`](../../../../internal/official_prompt_guides/minimax-h3-official-VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)) and **Base Mode (`T2VA` / `I2VA` / `FL2VA` / `L2VA`)** ([`base_en.md`](../../../../internal/official_prompt_guides/minimax-h3-official-VIDEO_PROMPT_WRITING_GUIDE_base_en.md)).

### 4.1 Canonical Ref2VA Structure (6 Sections)
1. `subject_definitions:` Defines all referenced entities (`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`).
2. `summary:` Summarizes generation task and reference bindings using `[reference generation]`.
3. `retention_analysis:` Specifies visual feature preservation rules.
4. `detailed_description:` Chronological shot description with dialogue in `<d>[Language] Text</d>`.
5. `overall_soundscape:` Ambient acoustics and Foley.
6. `non_diegetic_music:` Background score.

### 4.2 Base Mode Tasks (Header + 3 Core Fields)
* **T2VA:** Raw prompt text directly in 3 fields (`integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:`).
* **I2VA:** Mandatory header `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
* **FL2VA:** Mandatory header `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.`
* **L2VA:** Mandatory header `How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.`

---

## 5. Empirical Corpus Audit (2,670 Prompts)

```
                            EMPIRICAL CORPUS AUDIT (2,670 PROMPTS)
┌──────────────────────────────────────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Feature / Syntax Element                                 │ StellarVoyager/H3-IR     │ malcolmrey_various (MP4) │
├──────────────────────────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ **Total Available Rows**                                 │ **1,110**                │ **1,560**                │
│ `Ref2VA` (6 Canonical Sections)                          │ 938 (84.5%)              │ 0 (0.0%)                 │
│ `T2VA` (Text + Subject Definition)                       │ 0 (0.0%)                 │ 1,507 (96.6%)            │
│ `Pure T2VA` (3 Core Fields, No Header)                   │ 82 (7.4%)                │ 0 (0.0%)                 │
│ `I2VA` (First Frame Keyframe)                            │ 81 (7.3%)                │ ~50 (3.2%)               │
│ `FL2VA` (First & Last Frame Interpolation)               │ 9 (0.8%)                 │ 3 (0.2%)                 │
│ `subject_definitions:` Header                            │ 1,028 (92.6%)            │ 1,507 (96.6%)            │
│ `<Subject 1>` Token Binding                              │ 1,026 (92.4%)            │ 1,552 (99.5%)            │
│ `<Picture 1>` Token Binding                              │ 1,009 (90.9%)            │ 120 (7.7%)               │
│ `retention_analysis:` Header                             │ 1,028 (92.6%)            │ 0 (0.0%)                 │
│ `summary:` / `[reference generation]`                    │ 1,028 (92.6%)            │ 0 (0.0%)                 │
│ `integrated_multimodal_description:`                     │ 82 (7.4%)                │ 1,507 (96.6%)            │
│ Spoken Dialogue (`<d>[Language] ... </d>`)               │ 536 (48.3%)              │ 1,552 (99.5%)            │
│ `overall_soundscape:`                                    │ 1,110 (100.0%)           │ 1,507 (96.6%)            │
│ `non_diegetic_music:`                                    │ 1,110 (100.0%)           │ 1,477 (94.7%)            │
│ `[Shot 1]` Multi-Shot Structuring                        │ 1,110 (100.0%)           │ 1,477 (94.7%)            │
└──────────────────────────────────────────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 6. Preprocessing Policies & Geometry

Distinct preprocessing policies operate across native serving and quantization artifacts:

1. **Native Still-Image Policy:** Derived from release configs ($65,536$ to $16,777,216$ pixels).
2. **Current AWQ Still-Image Policy:** Constrained bounds of $200,704$ to $301,056$ pixels (`min_pixels=256*28*28`, `max_pixels=384*28*28`). Under this constrained policy, a $2 \times 2$ spatial merge results in **264–289 `<|image_pad|>` tokens** for common still images.
3. **Release Video Policy:** Owned by the released video processor configuration ($4,096$ to $25,165,824$ pixels across the clip), sampled at 2 fps and sliced into two-frame temporal blocks.
4. **Encoder-Artifact Video Policy:** Owned by the candidate encoder's snapshotted video configuration (`video_policy="encoder"`).

---

## 7. Native-H3 Presentation vs. Calibration Gap

### 7.1 Production Presentation Contract ([`minimax.py`](../../../../../../comfy/text_encoders/minimax.py))
Production H3 conditioning never uses chat templates:
* **T2VA:** Raw prompt text only.
* **FL2VA / Image:** `"<Picture i>: "` followed by vision block.
* **Audio:** `"<Audio j>: "` in text; no audio tensor enters Qwen.
* **Video:** `"<Video k>: "`, followed by two-frame blocks, each preceded by `"<T.T seconds>"`.
* **Modality Tags:** Tag `0` for all vision tokens (including flanking vision start/end); Tag `1` for text.

### 7.2 The Offline Calibration Gap
The original AWQ run wrapped all samples in Hugging Face chat templates (`<|im_start|>user ... <|im_end|><|im_start|>assistant`) and evaluated only 1-image samples. Because visual features feed directly into the residual stream of quantized language linears, this represents an **unmeasured distribution gap** for multi-image, video-reference, and raw prompt conditioning.

---

## 8. Proposed Differentiable Autograd Wrapper

The installed `comfy_kitchen.gemv_awq_w4a16` CUDA operation currently lacks a registered PyTorch autograd formula. Any backpropagation through the frozen W4 encoder requires a differentiable alternative. The following wrapper is unvalidated pseudocode: packed-weight dequantization, matrix orientation, gradients, memory, and numerical agreement must be tested against a small BF16 reference before it can become an implementation plan.

```python
class DifferentiableAWQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qweight, scales, zeros):
        ctx.save_for_backward(qweight, scales, zeros)
        return comfy_kitchen.gemv_awq_w4a16(x, qweight, scales, zeros)

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, zeros = ctx.saved_tensors
        # Dequantize W on-the-fly for active layer
        W_dequant = dequantize_awq_w4a16(qweight, scales, zeros)
        grad_x = torch.matmul(grad_output, W_dequant.t())
        return grad_x, None, None, None
```

*(Note: In ComfyUI, the text encoder and 33B DiT are sequentially offloaded between execution stages. Staged encoder weights occupy ~14.97 GB in memory).*

---

## 9. Proposed Research & Validation Roadmap

```
                             RESEARCH & VALIDATION ROADMAP

    [ Step 0: No-Training Multi-Seed Screen ]     [ Step 1: BPE Proxy Representation Study ]
 ┌─────────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
 │ • Render 3 arms (Release vs BPE vs Stripped)│ │ • Proxy distillation of BPE Layer 50 states│
 │ • Initial screen across controlled seeds    │ │ • Trainable delta (10,240 parameters)     │
 │ • Characterize behavioral response          │ │ • Optimize aligned proxy objective        │
 └─────────────────────────────────────────────┘ └───────────────────────────────────────────┘
                       │                                                │
                       ▼                                                ▼
    [ Step 2: Proposed 3-Pass DiT Recomputation ]  [ Step 3: Isolated Layer 50 Benchmark ]
 ┌─────────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
 │ • Proposed 3-pass recomputation design      │ │ • Weight-only numerical comparison        │
 │ • Decoupled Qwen/DiT gradient caching       │ │ • Deployed-path comparison                │
 │ • Memory and gradient correctness unmeasured│ │ • Independent evaluation across 5 buckets │
 └─────────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

### Step 0: No-Training Multi-Seed A/B Screen
Screen whether Release IDs (`151669`/`151670`), true unpatched BPE, or stripped prompts produce detectable differences in mouth motion across controlled seeds.

### Step 1: BPE Representation Proxy Optimization
If BPE demonstrates superior behavioral alignment, minimize the proxy distance between Layer 50 activations produced by $\Delta$ versus unpatched BPE on shared context tokens:
$$\mathcal{L}_{\text{proxy}} = \frac{1}{|S_{\text{context}}|} \sum_{i \in S_{\text{context}}} \left\| \mathbf{h}_i^{(50)}(\mathbf{p}, \mathbf{w}_{\text{base}} + \Delta) - \mathbf{h}_i^{(50)}(\mathbf{p}_{\text{BPE}}, \mathbf{w}_{\text{base}}) \right\|_2^2 + \lambda \|\Delta\|_2^2$$

### Step 2: Proposed 3-Pass DiT Recomputation Design
A proposed decoupled optimization design:
1. Pass 1: Forward Qwen $\to$ Cache Layer 50 hidden states.
2. Pass 2: Forward/Backward DiT $\to$ Compute $\nabla_{\mathbf{h}_{50}} \mathcal{L}$.
3. Pass 3: Backward Qwen $\to$ Compute $\nabla_{\Delta} \mathcal{L}$.

---

## 10. Immediate Empirical Action: Independent Layer 50 Benchmark

Codex is implementing the independent Layer 50 benchmark following the mandatory isolation split in [`canonical/native_h3_contract.md`](../canonical/native_h3_contract.md):
1. **Weight-Only Isolation:** Forces identical native-H3 token IDs and identical preprocessing into BF16 and W4 to measure pure weight quantization drift.
2. **Deployed-Path Evaluation:** Evaluates each artifact under its declared serving processor policy.
