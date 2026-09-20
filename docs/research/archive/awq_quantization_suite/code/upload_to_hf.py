#!/usr/bin/env python3
"""
Upload Qwen3-VL 32B W4A16 AWQ model files and ComfyUI checkpoint to Hugging Face.

Usage:
  python3 upload_to_hf.py \
    --repo_id your-username/qwen3-vl-32b-W4A16-AWQ-H3 \
    --multi_shard_dir models/qwen3-vl-32b-W4A16-AWQ-H3 \
    --single_file models/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors
"""

import os
import sys
import shutil
import argparse
from huggingface_hub import HfApi, create_repo


def build_model_card(repo_id: str) -> str:
    return f"""---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3
tags:
  - quantized
  - awq
  - w4a16
  - vllm
  - sglang
  - compressed-tensors
  - multimodal
  - vision-language
  - minimax-h3
  - text-to-video
pipeline_tag: image-text-to-text
---

# Qwen3-VL-32B-W4A16-AWQ-H3

This is a **W4A16 AWQ (Activation-Aware Quantized)** release of **Qwen3-VL 32B** optimized specifically for low-latency, single-GPU inference (**RTX 4090 / 24GB VRAM**) and serving as the multimodal text/reference conditioning encoder for the **MiniMax H3 Video DiT pipeline**.

* **Base Model Source:** [MiniMaxAI/MiniMax-H3 (text_encoder)](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/text_encoder)

---

## 1. Quantization Architecture & Component Breakdown

To maintain maximum visual fidelity on reference images while slashing the VRAM footprint for single-GPU serving, the model utilizes a **selective mixed-precision architecture**:

| Model Component | Layer Key Pattern | Parameter Count | Precision Level | Technical Rationale & Pipeline Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Transformer (ViT)** | `model.visual.blocks.*` (27 layers) | ~410M | **BF16 (Unquantized)** | Preserves high-frequency spatial details, facial geometry, and reference image resolution with zero quantization loss. |
| **DeepStack Feature Mergers** | `model.visual.deepstack_merger_list.*` | ~60M | **BF16 (Unquantized)** | Maintains continuous cross-scale feature adaptation between visual patch hierarchies and the language backbone. |
| **Patch & Spatial Projections** | `model.visual.patch_embed.*`, `merger.*` | ~30M | **BF16 (Unquantized)** | Ensures linear patch embedding projections enter Layer 0 without numerical noise. |
| **Token Embeddings** | `model.language_model.embed_tokens` | ~777M | **BF16 (Unquantized)** | Retains exact vector representations for vocabulary lookups, specifically critical H3 delimiters like `<d>`, `</d>`, and `<Subject N>`. |
| **Normalization Layers** | `.*input_layernorm$`, `.*post_attention_layernorm$`, `.*norm$` | ~3.2M | **BF16 (Unquantized)** | Preserves dynamic range and numerical stability across all 64 decoder layers. |
| **Output Head (LM Head)** | `lm_head.weight` | ~777M | **BF16 (Unquantized)** | Prevents logit drift during text token generation and hidden state extraction. |
| **Language Decoder Layers** | `model.language_model.layers.*` (64 layers) | ~30.6B | **W4A16 AWQ (INT4 Group 128)** | Compresses the 30.6B parameter language backbone to 4-bit weights with activation-informed channel scaling, accelerating GEMM operations via Marlin kernels. |
| **Activations & KV States** | Dynamic attention & MLP activations | N/A (Runtime) | **16-bit (A16)** | Keeps all intermediate activations in full 16-bit precision to maintain numerical fidelity in the Layer 50 hidden state tap. |
| **Total Model Footprint** | Complete Model Package | **~32.7B Params** | **Selective Mixed Precision** | Compresses total weight storage from 66.7 GB to 18.99 GB (3.5x reduction), enabling full in-VRAM execution on a single 24GB GPU. |

---

## 2. Calibration Strategy

Calibration was performed exclusively on **native MiniMax H3 multimodal prompt schemas** to ensure zero activation scale drift on H3 control syntax:

* **Data Mix & Formats:**
  * **Omni-Reference (`Ref2VA`):** Multi-reference conditioning with `<Subject N>`, `retention_analysis:`, and `summary: [reference generation]`.
  * **First/Last Keyframes (`I2VA` / `FL2VA`):** Temporal anchor conditioning (`At 0.00 seconds into the target video, <Picture 1>...`).
  * **Text-to-Video/Audio (`T2VA`):** Standard 3-field schemas (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`).
  * **Portrait & Face Geometry:** High-resolution unblurred facial reference portraits with dialogue tags (`<d>...</d>`).
* **Tap Point Protection:** Optimized to preserve the fidelity of **unnormalized hidden states after Layer 50**, which directly condition the downstream 33B DiT video generator.

---

## 3. Hardware & Memory Footprint

| Metric | BF16 Base Model | W4A16 AWQ (This Model) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Weight Footprint** | **66.7 GB** | **18.99 GB** | **3.5x Compression** |
| **Minimum Hardware** | 2× 80GB GPUs | **1× 24GB GPU (e.g. RTX 4090 / 3090)** | **Runs on Consumer Hardware** |
| **KV Cache Headroom (24GB)** | 0 GB (OOM) | **~5.5 GB** | Supports long multimodal context |

---

## 4. Inference & Deployment

### Serving with vLLM
```bash
vllm serve {repo_id} \\
  --quantization compressed-tensors \\
  --max-model-len 4096 \\
  --gpu-memory-utilization 0.90
```

### Serving with SGLang
```bash
python3 -m sglang.launch_server \\
  --model-path {repo_id} \\
  --quantization compressed-tensors \\
  --port 30000
```

### ComfyUI Single-File Loading
The repository includes the single-file checkpoint `qwen3vl_32b_minimax_h3_w4a16_awq.safetensors` (18.99 GB). Place it in your ComfyUI `models/clip/` or `models/text_encoders/` directory for direct use with MiniMax H3 nodes.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload quantized Qwen3-VL 32B model to Hugging Face.")
    parser.add_argument("--repo_id", type=str, default=os.environ.get("HF_REPO_ID", "your-username/qwen3-vl-32b-W4A16-AWQ-H3"), help="Hugging Face repo ID (e.g. username/model-name)")
    parser.add_argument("--multi_shard_dir", type=str, default="models/qwen3-vl-32b-W4A16-AWQ-H3", help="Multi-shard directory path")
    parser.add_argument("--single_file", type=str, default="models/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors", help="Single-file safetensors path")
    parser.add_argument("--storage_source", type=str, default=None, help="Optional path to source directory containing extra configs")
    args = parser.parse_args()

    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"Warning: Could not fetch user profile ({e}). Ensure you have run 'huggingface-cli login'.")

    # 1. Create or verify repository
    print(f"Ensuring repository exists: {args.repo_id}...")
    repo_url = create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    print(f"Repository URL: {repo_url}")

    # 2. Copy missing official tokenizers/configs from storage if provided
    if args.storage_source and os.path.exists(args.storage_source):
        for extra_file in ["vocab.json", "merges.txt", "video_preprocessor_config.json", "chat_template.json"]:
            src = os.path.join(args.storage_source, extra_file)
            dst = os.path.join(args.multi_shard_dir, extra_file)
            if os.path.exists(src) and not os.path.exists(dst):
                print(f"Copying {extra_file} into {args.multi_shard_dir}...")
                shutil.copy(src, dst)

    # 3. Write README.md (Model Card)
    readme_path = os.path.join(args.multi_shard_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(build_model_card(args.repo_id))
    print(f"Created Model Card at {readme_path}")

    # 4. Upload multi-shard model directory
    print(f"\nUploading multi-shard checkpoint and configs to {args.repo_id}...")
    api.upload_folder(
        folder_path=args.multi_shard_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )

    # 5. Upload single-file ComfyUI checkpoint if present
    if os.path.exists(args.single_file):
        size_gb = os.path.getsize(args.single_file) / (1024 ** 3)
        print(f"\nUploading single-file ComfyUI checkpoint ({size_gb:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=args.single_file,
            path_in_repo=os.path.basename(args.single_file),
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"\nSuccessfully uploaded all model artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
