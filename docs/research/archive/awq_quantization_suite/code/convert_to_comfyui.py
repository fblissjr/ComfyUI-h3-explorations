#!/usr/bin/env python3
"""
Convert multi-shard HuggingFace W4A16 AWQ checkpoint into a single, consolidated
.safetensors file formatted for ComfyUI's MiniMax H3 CLIP/Text Encoder loader.

Usage:
  python3 convert_to_comfyui.py \
    --input_dir models/qwen3-vl-32b-W4A16-AWQ-H3 \
    --output_file models/clip/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors
"""

import os
import sys
import glob
import json
import argparse
from safetensors.torch import load_file, save_file
import torch


def convert_multi_shard_to_single_file(input_dir: str, output_file: str):
    print(f"Starting conversion from multi-shard checkpoint at: {input_dir}")
    
    # 1. Discover all safetensors shards
    shard_files = sorted(glob.glob(os.path.join(input_dir, "model-*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
        shard_files = [f for f in shard_files if not f.endswith("index.safetensors")]

    if not shard_files:
        print(f"Error: No .safetensors shards found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(shard_files)} shard files to consolidate.")

    # 2. Extract configuration metadata for header
    metadata = {"format": "pt", "quantization": "awq", "scheme": "w4a16"}
    config_json = os.path.join(input_dir, "config.json")
    if os.path.exists(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            metadata["config"] = json.dumps(cfg)

    # 3. Consolidate state dicts in CPU memory
    consolidated_state_dict = {}
    total_tensors = 0

    for i, shard_path in enumerate(shard_files, 1):
        print(f"Loading shard {i}/{len(shard_files)}: {os.path.basename(shard_path)}...")
        shard_dict = load_file(shard_path, device="cpu")
        for key, tensor in shard_dict.items():
            consolidated_state_dict[key] = tensor
            total_tensors += 1

    print(f"\nAll shards merged! Total tensors loaded: {total_tensors}")

    # 4. Ensure output directory exists
    parent_dir = os.path.dirname(os.path.abspath(output_file))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # 5. Save consolidated single safetensors file
    print(f"Writing single consolidated checkpoint to: {output_file}...")
    save_file(consolidated_state_dict, output_file, metadata=metadata)

    file_size_gb = os.path.getsize(output_file) / (1024 ** 3)
    print(f"Conversion complete! Single file size: {file_size_gb:.2f} GB")
    print("Ready for direct loading in ComfyUI CLIPLoader nodes.")


def main():
    parser = argparse.ArgumentParser(description="Convert multi-shard AWQ checkpoint to single ComfyUI safetensors.")
    parser.add_argument("--input_dir", type=str, default="models/qwen3-vl-32b-W4A16-AWQ-H3", help="Path to quantized multi-shard directory")
    parser.add_argument("--output_file", type=str, default="models/clip/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors", help="Target single file path")
    args = parser.parse_args()

    convert_multi_shard_to_single_file(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
