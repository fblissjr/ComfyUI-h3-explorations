#!/usr/bin/env python3
"""
Standalone multi-threaded MP4 container extractor for MiniMax H3 ComfyUI workflows.
Parses embedded prompt metadata, generation parameters, and video specifications.

Usage:
  python3 extract_metadata.py \
    --dataset_root ./data/video_dataset \
    --output_json ./data/h3_extracted_metadata.json \
    --output_jsonl ./data/h3_extracted_metadata.jsonl \
    --workers 16
"""

import os
import glob
import json
import argparse
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_prompt_from_workflow(workflow_json_str: str) -> dict:
    """Parse ComfyUI workflow JSON and extract H3 prompt fields and generation parameters."""
    result = {
        "full_prompt": None,
        "subject_definitions": None,
        "integrated_multimodal_description": None,
        "overall_soundscape": None,
        "non_diegetic_music": None,
        "width": None,
        "height": None,
        "duration_seconds": None,
        "seed": None,
        "steps": None,
        "sampler_name": None,
        "scheduler": None,
        "unet_name": None,
        "clip_name": None,
        "raw_workflow": None,
    }

    try:
        graph = json.loads(workflow_json_str)
        result["raw_workflow"] = graph

        for node_id, node in graph.items():
            inputs = node.get("inputs", {})
            class_type = node.get("class_type", "")

            # Extract prompt from MiniMaxH3 nodes
            if "prompt" in inputs and isinstance(inputs["prompt"], str):
                prompt_text = inputs["prompt"]
                if "integrated_multimodal_description" in prompt_text or "subject_definitions" in prompt_text:
                    result["full_prompt"] = prompt_text
                    if "width" in inputs and isinstance(inputs["width"], (int, float)):
                        result["width"] = int(inputs["width"])
                    if "height" in inputs and isinstance(inputs["height"], (int, float)):
                        result["height"] = int(inputs["height"])

            # Extract Seed
            if "noise_seed" in inputs and isinstance(inputs["noise_seed"], int):
                result["seed"] = inputs["noise_seed"]
            elif "seed" in inputs and isinstance(inputs["seed"], int):
                result["seed"] = inputs["seed"]

            # Extract Scheduler & Steps
            if class_type == "BasicScheduler":
                result["steps"] = inputs.get("steps")
                result["scheduler"] = inputs.get("scheduler")
            elif class_type == "KSamplerSelect":
                result["sampler_name"] = inputs.get("sampler_name")

            # Extract Models
            if class_type == "CLIPLoader":
                result["clip_name"] = inputs.get("clip_name")
            elif class_type == "UNETLoader":
                result["unet_name"] = inputs.get("unet_name")

            # Extract duration if set via float node
            if class_type == "PrimitiveFloat" and "_meta" in node:
                if "duration" in node["_meta"].get("title", "").lower():
                    result["duration_seconds"] = inputs.get("value")

        # Parse sections from prompt text
        if result["full_prompt"]:
            text = result["full_prompt"]

            subj_match = re.search(r"subject_definitions:\s*(.*?)(?=\n\n|\n[a-z_]+:|$)", text, re.DOTALL)
            if subj_match:
                result["subject_definitions"] = subj_match.group(1).strip()

            desc_match = re.search(r"integrated_multimodal_description:\s*(.*?)(?=\n\noverall_soundscape:|\n\nnon_diegetic_music:|$)", text, re.DOTALL)
            if desc_match:
                result["integrated_multimodal_description"] = desc_match.group(1).strip()

            sound_match = re.search(r"overall_soundscape:\s*(.*?)(?=\n\nnon_diegetic_music:|$)", text, re.DOTALL)
            if sound_match:
                result["overall_soundscape"] = sound_match.group(1).strip()

            music_match = re.search(r"non_diegetic_music:\s*(.*?)$", text, re.DOTALL)
            if music_match:
                result["non_diegetic_music"] = music_match.group(1).strip()

    except Exception as e:
        result["parse_error"] = str(e)

    return result


def process_video(video_path: str, dataset_root: Path) -> dict | None:
    """Run ffprobe on an MP4 file and extract metadata."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return None

        data = json.loads(res.stdout)
        fmt = data.get("format", {})
        tags = fmt.get("tags", {})

        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})

        video_info = {
            "file_path": video_path,
            "filename": os.path.basename(video_path),
            "relative_path": os.path.relpath(video_path, dataset_root),
            "file_size_bytes": os.path.getsize(video_path),
            "duration_seconds": float(fmt.get("duration", 0.0)) if fmt.get("duration") else None,
            "video_codec": video_stream.get("codec_name"),
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in video_stream.get("r_frame_rate", "") else None,
            "has_audio": bool(audio_stream),
            "audio_codec": audio_stream.get("codec_name"),
        }

        prompt_tag = tags.get("prompt") or tags.get("comment") or tags.get("description")
        if prompt_tag:
            workflow_data = extract_prompt_from_workflow(prompt_tag)
            video_info.update(workflow_data)
        else:
            video_info["full_prompt"] = None

        return video_info
    except Exception as e:
        return {"file_path": video_path, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Extract MiniMax H3 metadata from ComfyUI video MP4s.")
    parser.add_argument("--dataset_root", type=str, default="./data", help="Root directory containing MP4 files")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent worker threads")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_json = Path(args.output_json) if args.output_json else dataset_root / "h3_extracted_metadata.json"
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else dataset_root / "h3_extracted_metadata.jsonl"

    print(f"Scanning for MP4 files in {dataset_root}...")
    mp4_files = sorted(glob.glob(os.path.join(dataset_root, "**/*.mp4"), recursive=True))
    total_files = len(mp4_files)
    print(f"Found {total_files} MP4 files. Extracting metadata with {args.workers} threads...")

    results = []
    with_prompts_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, f, dataset_root): f for f in mp4_files}
        for idx, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                results.append(res)
                if res.get("full_prompt"):
                    with_prompts_count += 1
            if idx % 100 == 0 or idx == total_files:
                print(f"Progress: {idx}/{total_files} processed ({with_prompts_count} prompts extracted)...")

    results.sort(key=lambda x: x.get("relative_path", ""))

    print(f"\nExtraction complete! Processed {len(results)} videos ({with_prompts_count} valid H3 prompts).")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            item_clean = {k: v for k, v in item.items() if k != "raw_workflow"}
            f.write(json.dumps(item_clean, ensure_ascii=False) + "\n")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_jsonl}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
