#!/usr/bin/env python3
"""Prepare controlled I2VA/FL2VA encoder arms from the shipped HF workflow.

ComfyUI expands frontend subgraphs before POSTing them. This tool starts from
one captured expanded prompt, then takes the public settings from the HF
first-frame workflow and applies the requested prompt/image. The emitted arms
differ only in their encoder loader and output prefix.

Example (always run through uv from the repository root):

  uv run --active --no-sync python bench/prepare_hf_first_frame_encoder_arms.py \
      --history /tmp/h3_history.json --prompt-id PROMPT_ID \
      --workflow /path/to/hf-model-workspace/\
comfyui_minimax_h3_awq_first_frame.json \
      --input /mnt/hub/ai/img/input/example.jpg \
      --out-dir /tmp/h3-first-frame-encoder-ab
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ENCODERS = {
    "w4a16": (
        "MiniMaxH3AWQEncoderLoader",
        {
            "encoder_name": "qwen3vl_32b_minimax_h3_w4a16_awq.safetensors",
            "device": "default",
        },
    ),
    "int8_convrot": (
        "CLIPLoader",
        {
            "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "type": "minimax",
            "device": "default",
        },
    ),
    "nvfp4": (
        "CLIPLoader",
        {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax",
            "device": "default",
        },
    ),
}
ENCODER_SHA256 = {
    "w4a16": "4d34831ddb7a5b9c8660fbf8eb6740b3cccace518fea883e4a41a76cc20b4dad",
    "int8_convrot": "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
    "nvfp4": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
}


def _one(graph: dict, class_type: str) -> tuple[str, dict]:
    found = [(nid, node) for nid, node in graph.items()
             if node.get("class_type") == class_type]
    if len(found) != 1:
        raise SystemExit(
            f"expected one {class_type}, found {len(found)}: "
            f"{[nid for nid, _ in found]}"
        )
    return found[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _encoder_filename(inputs: dict) -> str:
    names = [inputs[key] for key in ("encoder_name", "clip_name") if key in inputs]
    if len(names) != 1 or not isinstance(names[0], str):
        raise AssertionError(f"encoder arm has no unique filename: {inputs}")
    return names[0]


def _source_settings(workflow: dict) -> tuple[dict, dict]:
    resolution = [n for n in workflow["nodes"]
                  if n.get("type") == "ResolutionSelector"]
    subgraph_ids = {s["id"] for s in workflow.get("definitions", {})
                    .get("subgraphs", [])}
    instance = [n for n in workflow["nodes"] if n.get("type") in subgraph_ids]
    if len(resolution) != 1 or len(instance) != 1:
        raise SystemExit(
            "workflow must contain exactly one ResolutionSelector and one "
            "subgraph instance"
        )
    return resolution[0]["widgets_values_named"], instance[0]["widgets_values_named"]


def _normalized(graph: dict) -> dict:
    value = copy.deepcopy(graph)
    loader_id = next(
        nid for nid, node in value.items()
        if node.get("class_type") in {"MiniMaxH3AWQEncoderLoader", "CLIPLoader"}
    )
    value[loader_id] = {"class_type": "ENCODER", "inputs": {"encoder": "ARM"}}
    _, writer = _one(value, "SaveVideo")
    writer["inputs"]["filename_prefix"] = "ARM"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--last-frame",
        type=Path,
        default=None,
        help="optional closing anchor; when present the arm is FL2VA",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).with_name("prompts") /
        "hf_first_frame_sushi_cat.txt",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--aspect-ratio",
        default=None,
        help="override the HF workflow ResolutionSelector aspect label",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="override the HF workflow duration while retaining its H3 snap expression",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    history = json.loads(args.history.read_text())
    if args.prompt_id not in history:
        raise SystemExit(f"prompt id {args.prompt_id!r} is absent from history")
    base = copy.deepcopy(history[args.prompt_id]["prompt"][2])
    captured_base_sha256 = _json_sha256(base)
    workflow = json.loads(args.workflow.read_text())
    resolution, settings = _source_settings(workflow)

    loader_id, _ = _one(base, "MiniMaxH3AWQEncoderLoader")
    _, selector = _one(base, "ResolutionSelector")
    _, conditioner = _one(base, "MiniMaxH3ImageToVideo")
    _, noise = _one(base, "RandomNoise")
    _, duration = _one(base, "PrimitiveFloat")
    _, writer = _one(base, "SaveVideo")

    seed = args.seed if args.seed is not None else int(settings["noise_seed"])
    duration_seconds = (
        args.duration_seconds
        if args.duration_seconds is not None
        else float(settings["value_1"])
    )
    requested_frames = max(5, round(duration_seconds * 24))
    snapped_frames = requested_frames + (5 - requested_frames % 17) % 17

    prompt = args.prompt_file.read_text().strip()
    fields = ("integrated_multimodal_description:", "overall_soundscape:",
              "non_diegetic_music:")
    positions = [prompt.find(field) for field in fields]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        raise SystemExit("base prompt must contain the three official fields in order")
    if args.last_frame is None:
        expected_start = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    else:
        expected_start = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of "
            "the target video; Picture 2 (from Shot 1) aligns with the "
            f"{snapped_frames / 24:.2f}-second mark of the target video."
        )
    if not prompt.startswith(expected_start + "\n\n"):
        raise SystemExit("prompt does not start with the exact task alignment line")
    input_name = args.input.name

    selector["inputs"].update({
        "aspect_ratio": args.aspect_ratio or resolution["aspect_ratio"],
        "megapixels": resolution["megapixels"],
        "multiple": resolution["multiple"],
    })
    conditioner["inputs"]["prompt"] = prompt
    conditioner["inputs"]["first_frame"] = ["hf:first_frame", 0]
    if args.last_frame is None:
        conditioner["inputs"].pop("last_frame", None)
    else:
        conditioner["inputs"]["last_frame"] = ["hf:last_frame", 0]
    noise["inputs"]["noise_seed"] = seed
    duration["inputs"]["value"] = duration_seconds
    task = "fl2va" if args.last_frame is not None else "first_frame"
    writer["inputs"]["filename_prefix"] = f"Video/hf_{task}_encoder_ab"
    base["hf:first_frame"] = {
        "class_type": "LoadImage",
        "inputs": {"image": input_name},
    }
    if args.last_frame is not None:
        base["hf:last_frame"] = {
            "class_type": "LoadImage",
            "inputs": {"image": args.last_frame.name},
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = {}
    canonical = None
    for label, (class_type, inputs) in ENCODERS.items():
        graph = copy.deepcopy(base)
        graph[loader_id] = {
            "class_type": class_type,
            "inputs": copy.deepcopy(inputs),
            "_meta": {"title": f"encoder arm: {label}"},
        }
        _, arm_writer = _one(graph, "SaveVideo")
        arm_writer["inputs"]["filename_prefix"] += f"_{label}"
        comparable = _normalized(graph)
        if canonical is None:
            canonical = comparable
        elif comparable != canonical:
            raise SystemExit(f"{label} differs outside loader/output label")
        path = args.out_dir / f"{task}_{label}.json"
        path.write_text(json.dumps(graph, indent=2) + "\n")
        arms[label] = {
            "path": path.name,
            "sha256": _sha256(path),
            "encoder_filename": _encoder_filename(inputs),
            "encoder_sha256": ENCODER_SHA256[label],
        }

    manifest = {
        "source_workflow": args.workflow.name,
        "source_workflow_sha256": _sha256(args.workflow),
        "source_expansion": {
            "history_file": args.history.name,
            "history_sha256": _sha256(args.history),
            "prompt_id": args.prompt_id,
            "captured_base_graph_sha256": captured_base_sha256,
        },
        "inputs": [
            {"name": args.input.name, "sha256": _sha256(args.input)},
            *([] if args.last_frame is None else [{
                "name": args.last_frame.name,
                "sha256": _sha256(args.last_frame),
            }]),
        ],
        "prompt_file": args.prompt_file.name,
        "prompt_sha256": _sha256(args.prompt_file),
        "seed": seed,
        "resolution": dict(resolution, **({} if args.aspect_ratio is None else {
            "aspect_ratio": args.aspect_ratio,
        })),
        "task": task,
        "duration_seconds": duration_seconds,
        "arms": arms,
        "controlled_difference": "encoder loader and output prefix only",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(arms)} controlled arms and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
