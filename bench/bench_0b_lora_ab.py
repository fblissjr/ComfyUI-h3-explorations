#!/usr/bin/env python3
"""Priority 0b: Paired fl2va + LoRA vs ref2va Checkpoint A/B Test.

Runs the two arms on the cleaned P0 prompt with the sealed blind evaluation protocol.

Arm A: unet_fl2va + minimax_h3_ref_lora_rank_256_bf16.safetensors @ 1.0
Arm B: unet_ref2va base checkpoint (no LoRA)

Both arms share:
- Seed: 42
- Resolution: 1152x768 (fast tier)
- Length: 243 frames (10.125 s at 24 fps)
- Steps: 16 (er_sde / simple)
- Reference 1: h3_refs/face_young_man_glasses_1024x1024.png
- Reference 2: h3_refs/scene_alpine_lake_meadow_1024x1024.png
- Prompt: Verbatim P0 continuous 10s temporal blocking with neutral environment

Blind protocol:
- Randomly maps Arm A and Arm B to eval_render_01 and eval_render_02.
- Writes sealed key to internal/0b_blind_key_2026-08-17.json.
- Leaves eval_render_01.mp4 and eval_render_02.mp4 in ComfyUI's output directory, under Video/.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "workflows"
INTERNAL = REPO / "internal"

# Reference images live in ComfyUI's input directory.
REF_1 = "h3_refs/face_young_man_glasses_1024x1024.png"
REF_2 = "h3_refs/scene_alpine_lake_meadow_1024x1024.png"

sys.path.insert(0, str(WORKFLOWS))
import build_workflows as bw
import h3_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402


def build_graphs(prompt: str, seed: int = 42, length: int = 243) -> tuple[dict, dict]:
    """Build exact API graphs for Arm A (LoRA) and Arm B (Checkpoint)."""
    template_path = WORKFLOWS / "h3_image_ref_plus_text_to_video_ref_lora_api.json"
    raw = json.loads(template_path.read_text(encoding="utf-8"))

    # Arm A: fl2va + LoRA
    arm_a = copy.deepcopy(raw)
    arm_a["1"]["inputs"]["unet_name"] = cfg.MODELS["unet_fl2va"]
    arm_a["18"]["inputs"]["lora_name"] = cfg.REF_LORA
    arm_a["18"]["inputs"]["strength_model"] = cfg.REF_LORA_STRENGTH
    arm_a["19"]["inputs"]["model"] = ["18", 0]
    arm_a["6"]["inputs"]["noise_seed"] = seed
    arm_a["27"]["inputs"]["length"] = length
    arm_a["5"]["inputs"]["prompt"] = prompt
    arm_a["15"]["inputs"]["image"] = REF_1
    arm_a["16"]["inputs"]["image"] = REF_2
    arm_a["13"]["inputs"]["filename_prefix"] = "Video/eval_render_A_temp"

    # Arm B: ref2va Checkpoint (no LoRA)
    arm_b = copy.deepcopy(raw)
    arm_b["1"]["inputs"]["unet_name"] = cfg.MODELS["unet_ref2va"]
    # Bypass LoraLoader (node 18) -> SigmaShift (node 19) connects directly to UNETLoader (node 1)
    arm_b["19"]["inputs"]["model"] = ["1", 0]
    arm_b["6"]["inputs"]["noise_seed"] = seed
    arm_b["27"]["inputs"]["length"] = length
    arm_b["5"]["inputs"]["prompt"] = prompt
    arm_b["15"]["inputs"]["image"] = REF_1
    arm_b["16"]["inputs"]["image"] = REF_2
    arm_b["13"]["inputs"]["filename_prefix"] = "Video/eval_render_B_temp"

    return arm_a, arm_b


def submit_prompt(host: str, graph: dict[str, Any]) -> str:
    """Submit prompt to ComfyUI and return prompt_id."""
    data = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode("utf-8")
    req = urllib.request.Request(f"http://{host}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))["prompt_id"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP Error {e.code} submitting prompt: {err_body}", file=sys.stderr)
        raise


def wait_for_prompt(host: str, prompt_id: str, poll_s: float = 3.0) -> dict[str, Any]:
    """Wait for prompt execution to complete and return history entry."""
    print(f"Waiting for prompt {prompt_id} on {host}...")
    start_t = time.time()
    while True:
        try:
            with urllib.request.urlopen(f"http://{host}/history/{prompt_id}") as resp:
                hist = json.loads(resp.read().decode("utf-8"))
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False) or status.get("status_str") == "success":
                        elapsed = time.time() - start_t
                        print(f"Prompt {prompt_id} completed successfully in {elapsed:.1f}s")
                        return entry
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"Prompt failed: {status.get('messages')}")
        except urllib.error.URLError:
            pass
        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1:8188", help="ComfyUI host:port")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--length", type=int, default=243, help="Frame count (243 = 10.125s)")
    args = parser.parse_args()

    # Verify ComfyUI is reachable
    try:
        with urllib.request.urlopen(f"http://{args.host}/system_stats", timeout=5) as r:
            stats = json.loads(r.read())
            print(f"Connected to ComfyUI on {args.host} ({stats['devices'][0]['name']})")
    except Exception as e:
        print(f"Error connecting to ComfyUI on {args.host}: {e}", file=sys.stderr)
        return 1

    prompt = bw.R2V_PROMPT
    arm_a_graph, arm_b_graph = build_graphs(prompt, seed=args.seed, length=args.length)

    # Sealed Blind Randomization
    mapping = ["arm_a_lora", "arm_b_checkpoint"]
    random.shuffle(mapping)
    blind_labels = {
        "eval_render_01": mapping[0],
        "eval_render_02": mapping[1],
    }

    # Set output filenames according to blind mapping
    prefix_01 = f"Video/eval_render_seed{args.seed}_01"
    prefix_02 = f"Video/eval_render_seed{args.seed}_02"
    if mapping[0] == "arm_a_lora":
        arm_a_graph["13"]["inputs"]["filename_prefix"] = prefix_01
        arm_b_graph["13"]["inputs"]["filename_prefix"] = prefix_02
    else:
        arm_b_graph["13"]["inputs"]["filename_prefix"] = prefix_01
        arm_a_graph["13"]["inputs"]["filename_prefix"] = prefix_02

    # Write sealed keyfile
    INTERNAL.mkdir(parents=True, exist_ok=True)
    keyfile = INTERNAL / f"0b_blind_key_seed{args.seed}_2026-08-17.json"
    keyfile.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "length": args.length,
        "mapping": {
            f"eval_render_seed{args.seed}_01": mapping[0],
            f"eval_render_seed{args.seed}_02": mapping[1],
        },
        "prompt": prompt,
    }, indent=2), encoding="utf-8")
    print(f"Sealed randomization key written to {keyfile.name} (DO NOT OPEN BEFORE SCORING)")

    # Execute renders sequentially
    print("\n--- Queueing Render 1 (eval_render_01) ---")
    first_graph = arm_a_graph if mapping[0] == "arm_a_lora" else arm_b_graph
    id1 = submit_prompt(args.host, first_graph)
    wait_for_prompt(args.host, id1)

    print("\n--- Queueing Render 2 (eval_render_02) ---")
    second_graph = arm_b_graph if mapping[0] == "arm_a_lora" else arm_a_graph
    id2 = submit_prompt(args.host, second_graph)
    wait_for_prompt(args.host, id2)

    print("\n" + "=" * 60)
    print("0b RENDERS COMPLETE — BLIND EVALUATION INSTRUCTIONS")
    print("=" * 60)
    print(f"Output video files (in {_paths.comfy_output() or 'ComfyUI output'}/Video/):")
    print("  1. eval_render_01.mp4")
    print("  2. eval_render_02.mp4")
    print("\nReference Images:")
    print(f"  <Picture 1>: {REF_1}")
    print(f"  <Picture 2>: {REF_2}")
    print("\nEvaluation Rubric:")
    print("  1. Identity Retention (1-5): Facial geometry, hairstyle, wardrobe match <Picture 1>")
    print("  2. 3D Parallax & Depth (1-5): Background 3D parallax during lateral camera truck vs flat 2D collapse")
    print("\nLog your scores and frame notes into the session ledger BEFORE unblinding.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
