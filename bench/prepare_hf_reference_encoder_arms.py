#!/usr/bin/env python3
"""Convert the flat HF Ref2VA workflow into controlled encoder API arms."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from prepare_hf_first_frame_encoder_arms import (
    ENCODERS,
    ENCODER_SHA256,
    _encoder_filename,
    _normalized,
    _one,
    _sha256,
)


DROP_WIDGETS = {"control_after_generate", "fixed", "upload"}
DROP_CLASSES = {"MarkdownNote"}
SECTIONS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def _assert_prompt(prompt: str) -> None:
    positions = [prompt.find(section) for section in SECTIONS]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        raise SystemExit(
            "Ref2VA prompt must contain all six official sections in order"
        )
    for label in ("<Picture 1>", "<Picture 2>", "<Subject 1>", "<Subject 2>"):
        if label not in prompt:
            raise SystemExit(f"Ref2VA prompt does not cite {label}")


def _ui_to_api(workflow: dict) -> dict:
    nodes = {str(node["id"]): node for node in workflow["nodes"]
             if node.get("mode", 0) == 0 and node["type"] not in DROP_CLASSES}
    graph = {}
    for node_id, node in nodes.items():
        inputs = copy.deepcopy(node.get("widgets_values_named") or {})
        for key in DROP_WIDGETS:
            inputs.pop(key, None)
        graph[node_id] = {
            "class_type": node["type"],
            "inputs": inputs,
            "_meta": {"title": node.get("title", node["type"])},
        }

    for link in workflow["links"]:
        _link_id, origin_id, origin_slot, target_id, target_slot, _type = link
        target = nodes[str(target_id)]
        input_name = target["inputs"][target_slot]["name"]
        graph[str(target_id)]["inputs"][input_name] = [str(origin_id), origin_slot]

    # Frontend dynamic widgets are not fully represented in
    # widgets_values_named. Mirror the expanded form emitted by ComfyUI.
    _, writer = _one(graph, "SaveVideo")
    writer["inputs"].update({"format": "auto", "format.codec": "auto",
                             "codec": "auto"})
    return graph


def _benchmark_ui(workflow: dict, images: list[Path], prompt: str,
                  duration_seconds: float, seed: int) -> dict:
    """A loadable UI copy matching the submitted W4 arm's public widgets."""
    ui = copy.deepcopy(workflow)
    loads = sorted(
        (node for node in ui["nodes"] if node.get("type") == "LoadImage"),
        key=lambda node: int(node["id"]),
    )
    if len(loads) != 2:
        raise SystemExit(f"expected two UI LoadImage nodes, found {len(loads)}")
    for node, image in zip(loads, images, strict=True):
        node["widgets_values"] = [image.name, "image"]
        node["widgets_values_named"] = {"image": image.name, "upload": "image"}

    anchors = {}
    for class_type in ("PrimitiveStringMultiline", "PrimitiveFloat", "RandomNoise",
                       "SaveVideo"):
        found = [node for node in ui["nodes"] if node.get("type") == class_type]
        if len(found) != 1:
            raise SystemExit(f"expected one UI {class_type}, found {len(found)}")
        anchors[class_type] = found[0]
    anchors["PrimitiveStringMultiline"]["widgets_values"] = [prompt]
    anchors["PrimitiveStringMultiline"]["widgets_values_named"] = {"value": prompt}
    anchors["PrimitiveFloat"]["widgets_values"] = [duration_seconds]
    anchors["PrimitiveFloat"]["widgets_values_named"] = {"value": duration_seconds}
    anchors["RandomNoise"]["widgets_values"] = [seed, "fixed"]
    anchors["RandomNoise"]["widgets_values_named"] = {
        "noise_seed": seed,
        "control_after_generate": "fixed",
    }
    prefix = "Video/hf_reference_encoder_ab_w4a16"
    anchors["SaveVideo"]["widgets_values"][0] = prefix
    anchors["SaveVideo"]["widgets_values_named"]["filename_prefix"] = prefix
    ui.setdefault("extra", {})["h3_encoder_ab"] = {
        "arm": "w4a16",
        "controlled_source": workflow.get("id"),
        "duration_seconds": duration_seconds,
        "seed": seed,
    }
    return ui


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=261662374822964)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--ui-out",
        type=Path,
        default=None,
        help="optional loadable UI copy of the exact W4 benchmark settings",
    )
    args = parser.parse_args()
    if len(args.image) != 2:
        raise SystemExit("this benchmark requires exactly two --image arguments")

    workflow = json.loads(args.workflow.read_text())
    graph = _ui_to_api(workflow)
    prompt = args.prompt_file.read_text().strip()
    _assert_prompt(prompt)

    if args.ui_out is not None:
        ui = _benchmark_ui(
            workflow, list(args.image), prompt, args.duration_seconds, args.seed
        )
        args.ui_out.parent.mkdir(parents=True, exist_ok=True)
        args.ui_out.write_text(json.dumps(ui, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote loadable W4 benchmark workflow {args.ui_out}")

    load_ids = sorted(
        (nid for nid, node in graph.items() if node["class_type"] == "LoadImage"),
        key=int,
    )
    if len(load_ids) != 2:
        raise SystemExit(f"expected two LoadImage nodes, found {load_ids}")
    for node_id, image in zip(load_ids, args.image, strict=True):
        graph[node_id]["inputs"] = {"image": image.name}

    loader_id, _ = _one(graph, "MiniMaxH3AWQEncoderLoader")
    _, prompt_node = _one(graph, "PrimitiveStringMultiline")
    _, duration = _one(graph, "PrimitiveFloat")
    _, noise = _one(graph, "RandomNoise")
    _, writer = _one(graph, "SaveVideo")
    prompt_node["inputs"] = {"value": prompt}
    duration["inputs"] = {"value": args.duration_seconds}
    noise["inputs"] = {"noise_seed": args.seed}
    writer["inputs"]["filename_prefix"] = "Video/hf_reference_encoder_ab"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical = None
    arms = {}
    for label, (class_type, inputs) in ENCODERS.items():
        arm = copy.deepcopy(graph)
        arm[loader_id] = {
            "class_type": class_type,
            "inputs": copy.deepcopy(inputs),
            "_meta": {"title": f"encoder arm: {label}"},
        }
        _, arm_writer = _one(arm, "SaveVideo")
        arm_writer["inputs"]["filename_prefix"] += f"_{label}"
        comparable = _normalized(arm)
        if canonical is None:
            canonical = comparable
        elif comparable != canonical:
            raise SystemExit(f"{label} differs outside loader/output label")
        path = args.out_dir / f"reference_{label}.json"
        path.write_text(json.dumps(arm, indent=2) + "\n")
        arms[label] = {
            "path": path.name,
            "sha256": _sha256(path),
            "encoder_filename": _encoder_filename(inputs),
            "encoder_sha256": ENCODER_SHA256[label],
        }

    manifest = {
        "source_workflow": args.workflow.name,
        "source_workflow_sha256": _sha256(args.workflow),
        "images": [
            {"name": image.name, "sha256": _sha256(image)}
            for image in args.image
        ],
        "prompt_file": args.prompt_file.name,
        "prompt_sha256": _sha256(args.prompt_file),
        "duration_seconds": args.duration_seconds,
        "seed": args.seed,
        "arms": arms,
        "controlled_difference": "encoder loader and output prefix only",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(arms)} controlled arms and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
