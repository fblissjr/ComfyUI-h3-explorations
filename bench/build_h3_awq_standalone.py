#!/usr/bin/env python3
"""Build the self-contained HF loader and native-Comfy example workflows.

The repo's ``h3_awq_encoder.py`` and four versioned runtime JSON files remain
authoritative.  This build embeds those JSON files in one directly installable
custom-node module, then adapts ComfyUI's installed official MiniMax H3
templates to select that node.  It does not copy any of this repo's other
custom nodes into the standalone distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
from h3_config import (  # noqa: E402
    TURBO_768P_LORA,
    TURBO_768P_SHIFT,
    TURBO_768P_STEPS,
    TURBO_768P_STRENGTH,
)

LOADER_SOURCE = REPO / "h3_awq_encoder.py"
CONFIG_DIR = REPO / "config" / "qwen3vl_32b_minimax_h3_w4a16_awq"
RUNTIME_CONFIGS = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)

HF_REPO_ID = "fbjr/qwen3-vl-32b-W4A16-AWQ-H3"
MODEL_FILENAME = "qwen3vl_32b_minimax_h3_w4a16_awq_v1-comfy.safetensors"
STANDALONE_FILENAME = "comfyui_minimax_h3_awq_loader.py"
COMPARE_WORKFLOW_FILENAME = "comfyui_minimax_h3_encoder_ab_compare.json"
# Repo-relative directory the generated workflows are emitted into. The loader
# and the Hugging Face checkpoint files stay at the root; see build().
WORKFLOW_SUBDIR = "comfyui_sample_workflows"
NODE_ID = "MiniMaxH3AWQEncoderLoader"
OLD_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
OLD_ENCODER_URL = (
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/"
    + OLD_ENCODER
)
NEW_ENCODER_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{MODEL_FILENAME}"
OLD_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TURBO_LORA_URL = (
    "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/"
    + Path(TURBO_768P_LORA).name
)
OLD_TURBO_LORA_URL = (
    "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/"
    + OLD_TURBO_LORA
)
REF2V_TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REF2V_TURBO_LORA_PATH = f"h3/lightx2v_Minimax-h3-Turbo/{REF2V_TURBO_LORA}"

WORKFLOWS = {
    "comfyui_minimax_h3_awq_text_to_video.json": "video_minimax_h3_t2v.json",
    "comfyui_minimax_h3_awq_image_reference.json": "video_minimax_h3_r2v.json",
    "comfyui_minimax_h3_awq_first_frame.json": "video_minimax_h3_i2v.json",
    "comfyui_minimax_h3_awq_first_last_frame.json": "video_minimax_h3_i2v.json",
}
FIRST_LAST_WORKFLOW = "comfyui_minimax_h3_awq_first_last_frame.json"

_FIRST_LAST_PROMPT = """How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.17-second mark of the target video.

integrated_multimodal_description: [Shot 1] Cinematic live-action realism in one continuous shot with no cuts. The target video begins exactly from Picture 1, preserving its subjects, faces, wardrobe, objects, lighting, camera side, and spatial composition. Natural motion develops continuously between the two supplied moments while the camera performs a slow Push In with small amplitude. Every subject moves with coherent anatomy and stable identity; fixed objects remain fixed, materials retain their textures, and lighting changes only where Picture 2 requires it. Motion gradually settles as framing, pose, expression, object placement, illumination, shadows, and the complete composition converge on Picture 2 exactly at the final frame. No additional subject, duplicate face, wardrobe change, unrelated object, scene change, text, subtitles, logo, watermark, or camera reversal appears.

overall_soundscape: Quiet location ambience matched to the reference scene, with restrained movement sounds synchronized to the visible action and no unexplained voice.

non_diegetic_music: A minimal sustained cinematic underscore at low volume, resolving gently at the final frame."""

_CONFIG_DECLARATION = '''CONFIG_DIR = (Path(__file__).resolve().parent / "config" /
              "qwen3vl_32b_minimax_h3_w4a16_awq")
CONFIG_SOURCE = str(CONFIG_DIR)
'''

_SOURCE_DOC = '''This is deliberately a custom loader instead of a patch to ``CLIPLoader``.
ComfyUI natively supplies the H3 architecture and tokenizer (including the
seven H3 tokens), and comfy-kitchen supplies the CUDA W4A16 operator.  Core
does not currently recognize compressed-tensors' Hugging Face namespace,
packing, or metadata.  This module is the repo-local adapter for that gap.

The source stays symlinked under ``models/text_encoders``.  Adaptation is
in-memory and view-based for the 4-bit weights; it does not write a second
multi-gigabyte checkpoint.  The authoritative small source configs are
snapshotted under ``config/qwen3vl_32b_minimax_h3_w4a16_awq``.
'''

_STANDALONE_DOC = '''This is deliberately a custom loader instead of a patch to ``CLIPLoader``.
ComfyUI natively supplies the H3 architecture and tokenizer (including the
seven H3 tokens), and comfy-kitchen supplies the CUDA W4A16 operator.  Core
does not currently recognize compressed-tensors' Hugging Face namespace,
packing, or metadata.  This generated module is the standalone adapter for
that gap.

Adaptation is in-memory and view-based for the 4-bit weights; it does not
write a second multi-gigabyte checkpoint.  Its four small runtime configs are
embedded from the versioned ComfyUI-h3-explorations snapshot.
'''

_STANDALONE_FOOTER = f'''


class MiniMaxH3AWQStandaloneExtension(ComfyExtension):
    async def get_node_list(self):
        return [{NODE_ID}]


async def comfy_entrypoint() -> MiniMaxH3AWQStandaloneExtension:
    return MiniMaxH3AWQStandaloneExtension()
'''

_WORKFLOW_NOTE = f"""## Standalone W4A16 AWQ encoder copy

This workflow is derived from ComfyUI's official MiniMax H3 template. Its
native `CLIPLoader` was replaced with `{NODE_ID}` from
[`{STANDALONE_FILENAME}`](https://huggingface.co/{HF_REPO_ID}/blob/main/{STANDALONE_FILENAME}).
All other H3 conditioning, sampling, and decode nodes in this graph are native
ComfyUI nodes. Select your own input image(s) where applicable.

"""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} anchor, found {count}")
    return text.replace(old, new)


def _runtime_config_text() -> tuple[dict[str, str], dict[str, str]]:
    texts = {}
    digests = {}
    for name in RUNTIME_CONFIGS:
        path = CONFIG_DIR / name
        text = path.read_text()
        # Fail at build time rather than shipping an embedded malformed file.
        json.loads(text)
        texts[name] = text
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return texts, digests


def render_standalone_loader() -> str:
    source = LOADER_SOURCE.read_text()
    texts, digests = _runtime_config_text()
    source_digest = hashlib.sha256(source.encode()).hexdigest()
    combined = hashlib.sha256()
    for name in RUNTIME_CONFIGS:
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(texts[name].encode())
        combined.update(b"\0")

    embedded = ["_EMBEDDED_CONFIG_TEXT = {"]
    for name in RUNTIME_CONFIGS:
        embedded.append(f"    {name!r}: {texts[name]!r},")
    embedded.extend(["}", "", "_EMBEDDED_CONFIG_SHA256 = {"])
    for name in RUNTIME_CONFIGS:
        embedded.append(f"    {name!r}: {digests[name]!r},")
    embedded.extend([
        "}",
        "CONFIG_SOURCE = (",
        "    'four runtime configs embedded by build_h3_awq_standalone.py '",
        f"    '({combined.hexdigest()})'",
        ")",
        "",
    ])
    config_decl = "\n".join(embedded)

    config_function_start = source.index("@functools.lru_cache(maxsize=None)\ndef _config")
    config_function_end = source.index("\n\ndef _quant_contract()", config_function_start)
    source = (
        source[:config_function_start]
        + "@functools.lru_cache(maxsize=None)\n"
          "def _config(name: str) -> dict:\n"
          "    try:\n"
          "        text = _EMBEDDED_CONFIG_TEXT[name]\n"
          "    except KeyError as exc:\n"
          "        raise FileNotFoundError(\n"
          "            f\"{name} is not embedded in this standalone loader\"\n"
          "        ) from exc\n"
          "    return json.loads(text)"
        + source[config_function_end:]
    )
    source = _replace_once(
        source, _CONFIG_DECLARATION, config_decl, "config declaration"
    )
    source = _replace_once(source, _SOURCE_DOC, _STANDALONE_DOC, "module doc")
    source = _replace_once(
        source,
        "from comfy_api.latest import io",
        "from comfy_api.latest import ComfyExtension, io",
        "ComfyExtension import",
    )
    source = _replace_once(
        source,
        "Repo-local adapter for Qwen3-VL-32B H3 checkpoints",
        "Standalone adapter for Qwen3-VL-32B H3 checkpoints",
        "node description",
    )
    source = _replace_once(
        source,
        "through repo-local compressed-tensors adapter",
        "through standalone compressed-tensors adapter",
        "loader log",
    )
    header = (
        "# GENERATED FILE - DO NOT EDIT BY HAND.\n"
        "# Source: ComfyUI-h3-explorations/h3_awq_encoder.py\n"
        f"# Source SHA-256: {source_digest}\n"
        "# Runtime config SHA-256 values are recorded in "
        "_EMBEDDED_CONFIG_SHA256.\n\n"
    )
    return header + source.rstrip() + _STANDALONE_FOOTER


def _template_dir() -> Path:
    spec = importlib.util.find_spec("comfyui_workflow_templates_json")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "comfyui-workflow-templates-json is not installed in the active uv environment"
        )
    path = Path(spec.origin).parent / "templates"
    if not path.is_dir():
        raise FileNotFoundError(f"official template directory is missing: {path}")
    return path


def _nodes(workflow: dict):
    yield from workflow.get("nodes", [])
    for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def _replace_strings(value):
    if isinstance(value, str):
        return (
            value.replace(OLD_ENCODER_URL, NEW_ENCODER_URL)
            .replace(OLD_ENCODER, MODEL_FILENAME)
            .replace(OLD_TURBO_LORA_URL, TURBO_LORA_URL)
            .replace(OLD_TURBO_LORA, TURBO_768P_LORA)
        )
    if isinstance(value, list):
        return [_replace_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item) for key, item in value.items()}
    return value


def _validate_link_table(nodes: list[dict], links: list, label: str) -> None:
    """Validate LiteGraph list links and V3 subgraph object links alike."""
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError(f"{label}: duplicate node id")
    records = {}
    for raw in links:
        if isinstance(raw, dict):
            record = raw
        else:
            record = {
                "id": raw[0], "origin_id": raw[1], "origin_slot": raw[2],
                "target_id": raw[3], "target_slot": raw[4], "type": raw[5],
            }
        link_id = record["id"]
        if link_id in records:
            raise ValueError(f"{label}: duplicate link id {link_id}")
        records[link_id] = record
        source_id = record["origin_id"]
        target_id = record["target_id"]
        if source_id >= 0:
            source = by_id.get(source_id)
            if source is None:
                raise ValueError(f"{label}: link {link_id} source {source_id} missing")
            slot = record["origin_slot"]
            outputs = source.get("outputs", [])
            if slot >= len(outputs) or link_id not in (outputs[slot].get("links") or []):
                raise ValueError(f"{label}: link {link_id} disagrees with source slot")
        if target_id >= 0:
            target = by_id.get(target_id)
            if target is None:
                raise ValueError(f"{label}: link {link_id} target {target_id} missing")
            slot = record["target_slot"]
            inputs = target.get("inputs", [])
            if slot >= len(inputs) or inputs[slot].get("link") != link_id:
                raise ValueError(f"{label}: link {link_id} disagrees with target slot")

    for node in nodes:
        for input_ in node.get("inputs", []):
            link_id = input_.get("link")
            if link_id is not None and link_id not in records:
                raise ValueError(f"{label}: node {node['id']} input link {link_id} missing")
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                if link_id not in records:
                    raise ValueError(
                        f"{label}: node {node['id']} output link {link_id} missing"
                    )


def _validate_workflow_structure(workflow: dict, label: str) -> None:
    _validate_link_table(workflow.get("nodes", []), workflow.get("links", []), label)
    subgraphs = (workflow.get("definitions") or {}).get("subgraphs", [])
    definitions = {subgraph["id"] for subgraph in subgraphs}
    instances = {
        node.get("type") for node in workflow.get("nodes", [])
        if node.get("type") in definitions
    }
    if definitions != instances:
        raise ValueError(
            f"{label}: subgraph definitions/instances differ: "
            f"{definitions ^ instances}"
        )
    for subgraph in subgraphs:
        _validate_link_table(
            subgraph.get("nodes", []), subgraph.get("links", []),
            f"{label}:{subgraph['id']}",
        )
    numeric_nodes = [
        node["id"] for node in workflow.get("nodes", [])
        if isinstance(node.get("id"), int)
    ]
    numeric_nodes.extend(
        node["id"] for subgraph in subgraphs for node in subgraph.get("nodes", [])
        if isinstance(node.get("id"), int)
    )
    numeric_links = [
        link[0] for link in workflow.get("links", [])
        if isinstance(link, list) and isinstance(link[0], int)
    ]
    numeric_links.extend(
        link["id"] for subgraph in subgraphs for link in subgraph.get("links", [])
        if isinstance(link, dict) and isinstance(link.get("id"), int)
    )
    if numeric_nodes and workflow.get("last_node_id", 0) < max(numeric_nodes):
        raise ValueError(f"{label}: last_node_id is stale")
    if numeric_links and workflow.get("last_link_id", 0) < max(numeric_links):
        raise ValueError(f"{label}: last_link_id is stale")


def _apply_owner_turbo_recipe(workflow: dict, template_name: str) -> None:
    """Make FL2VA examples use the repo's current v1.1 owner recipe."""
    if template_name not in {"video_minimax_h3_t2v.json", "video_minimax_h3_i2v.json"}:
        return
    subgraphs = (workflow.get("definitions") or {}).get("subgraphs", [])
    if len(subgraphs) != 1:
        raise ValueError(f"{template_name}: expected one implementation subgraph")
    subgraph = subgraphs[0]
    nodes = subgraph.get("nodes", [])
    loras = [node for node in nodes if node.get("type") == "LoraLoaderModelOnly"]
    model_switches = [
        node for node in nodes
        if node.get("type") == "ComfySwitchNode"
        and any(output.get("type") == "MODEL" for output in node.get("outputs", []))
    ]
    booleans = [node for node in nodes if node.get("type") == "PrimitiveBoolean"]
    if len(loras) != 1 or len(model_switches) != 1 or len(booleans) != 1:
        raise ValueError(
            f"{template_name}: owner recipe anchors changed "
            f"(lora={len(loras)}, model_switch={len(model_switches)}, "
            f"boolean={len(booleans)})"
        )

    lora = loras[0]
    lora["widgets_values"] = [TURBO_768P_LORA, TURBO_768P_STRENGTH]
    lora["widgets_values_named"] = {
        "lora_name": TURBO_768P_LORA,
        "strength_model": TURBO_768P_STRENGTH,
    }
    lora["properties"]["models"] = [{
        "name": Path(TURBO_768P_LORA).name,
        "url": TURBO_LORA_URL,
        "directory": "loras",
    }]
    booleans[0]["widgets_values"] = [True]
    booleans[0]["widgets_values_named"] = {"value": True}

    turbo_steps = [
        node for node in nodes
        if node.get("type") == "PrimitiveInt"
        and node.get("widgets_values", [None])[0] == 6
    ]
    if len(turbo_steps) != 1:
        raise ValueError(f"{template_name}: expected one turbo-step primitive")
    turbo_steps[0]["widgets_values"] = [TURBO_768P_STEPS, "fixed"]
    turbo_steps[0]["widgets_values_named"] = {"value": TURBO_768P_STEPS}

    outer = [
        node for node in workflow.get("nodes", [])
        if isinstance(node.get("type"), str) and len(node["type"]) == 36
        and TURBO_768P_LORA in node.get("widgets_values", [])
    ]
    if len(outer) != 1:
        raise ValueError(f"{template_name}: expected one exposed subgraph instance")
    values = outer[0]["widgets_values"]
    index = values.index(TURBO_768P_LORA)
    if index < 1 or len(values) <= index + 2:
        raise ValueError(f"{template_name}: exposed turbo widgets changed order")
    values[index - 1:index + 3] = [
        True, TURBO_768P_LORA, TURBO_768P_STRENGTH, TURBO_768P_STEPS,
    ]
    named = outer[0].get("widgets_values_named")
    if isinstance(named, dict):
        named.update({
            "value": True,
            "lora_name": TURBO_768P_LORA,
            "strength_model_1": TURBO_768P_STRENGTH,
            "value_2": TURBO_768P_STEPS,
        })

    model_switch = model_switches[0]
    output = next(o for o in model_switch["outputs"] if o.get("type") == "MODEL")
    downstream = list(output.get("links") or [])
    links = subgraph.get("links", [])
    downstream_records = [link for link in links if link.get("id") in downstream]
    if len(downstream_records) != len(downstream) or not downstream:
        raise ValueError(f"{template_name}: model switch downstream links changed")

    used_node_ids = {
        node.get("id")
        for node in workflow.get("nodes", [])
        if isinstance(node.get("id"), int)
    } | {node.get("id") for node in nodes if isinstance(node.get("id"), int)}
    used_link_ids = {
        link.get("id") for link in links if isinstance(link.get("id"), int)
    }
    node_id = max(used_node_ids | {int(workflow.get("last_node_id") or 0)}) + 1
    link_id = max(used_link_ids | {int(workflow.get("last_link_id") or 0)}) + 1
    for link in downstream_records:
        link["origin_id"] = node_id
    output["links"] = [link_id]
    links.append({
        "id": link_id,
        "origin_id": model_switch["id"],
        "origin_slot": 0,
        "target_id": node_id,
        "target_slot": 0,
        "type": "MODEL",
    })
    sx, sy = model_switch.get("pos", [-400, 4790])
    nodes.append({
        "id": node_id,
        "type": "MiniMaxH3SigmaShift",
        "pos": [sx + 380, sy + 150],
        "size": [360, 110],
        "flags": {},
        "order": max(int(node.get("order") or 0) for node in nodes) + 1,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": link_id}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": downstream}],
        "properties": {"Node name for S&R": "MiniMaxH3SigmaShift"},
        "widgets_values": [
            float(TURBO_768P_SHIFT["shift_video"]),
            float(TURBO_768P_SHIFT["shift_audio"]),
        ],
        "widgets_values_named": {
            "shift_video": float(TURBO_768P_SHIFT["shift_video"]),
            "shift_audio": float(TURBO_768P_SHIFT["shift_audio"]),
        },
        "title": (
            f"Owner recipe: v1.1, {TURBO_768P_STEPS} steps, "
            f"strength {TURBO_768P_STRENGTH:g}, shift "
            f"{TURBO_768P_SHIFT['shift_video']:g}/"
            f"{TURBO_768P_SHIFT['shift_audio']:g}"
        ),
    })
    workflow["last_node_id"] = node_id
    workflow["last_link_id"] = link_id


def _add_last_frame_input(workflow: dict, label: str) -> None:
    """Turn the official I2VA template into an explicit two-anchor example."""
    nodes = workflow.get("nodes", [])
    loads = [node for node in nodes if node.get("type") == "LoadImage"]
    subgraph_ids = {
        subgraph["id"]
        for subgraph in (workflow.get("definitions") or {}).get("subgraphs", [])
    }
    instances = [node for node in nodes if node.get("type") in subgraph_ids]
    if len(loads) != 1 or len(instances) != 1:
        raise ValueError(
            f"{label}: first/last variant expected one LoadImage and one instance"
        )
    instance = instances[0]
    target_slot = next(
        (index for index, input_ in enumerate(instance.get("inputs", []))
         if input_.get("name") == "last_frame"),
        None,
    )
    if target_slot is None or instance["inputs"][target_slot].get("link") is not None:
        raise ValueError(f"{label}: last_frame input is absent or already connected")

    numeric_node_ids = [
        node["id"] for node in nodes if isinstance(node.get("id"), int)
    ]
    numeric_link_ids = [
        link[0] for link in workflow.get("links", [])
        if isinstance(link, list) and isinstance(link[0], int)
    ]
    node_id = max(numeric_node_ids + [int(workflow.get("last_node_id") or 0)]) + 1
    link_id = max(numeric_link_ids + [int(workflow.get("last_link_id") or 0)]) + 1

    last = json.loads(json.dumps(loads[0]))
    last["id"] = node_id
    last["pos"] = [last["pos"][0] + 500, last["pos"][1]]
    last["title"] = "Last frame"
    last["outputs"][0]["links"] = [link_id]
    nodes.append(last)
    instance["inputs"][target_slot]["link"] = link_id
    workflow.setdefault("links", []).append(
        [link_id, node_id, 0, instance["id"], target_slot, "IMAGE"]
    )
    named = instance.get("widgets_values_named")
    values = instance.get("widgets_values")
    if not isinstance(named, dict) or not isinstance(named.get("prompt"), str):
        raise ValueError(f"{label}: exposed prompt widget is missing")
    old_prompt = named["prompt"]
    matches = [index for index, value in enumerate(values or [])
               if value == old_prompt]
    if len(matches) != 1:
        raise ValueError(
            f"{label}: could not identify one positional prompt widget"
        )
    values[matches[0]] = _FIRST_LAST_PROMPT
    named["prompt"] = _FIRST_LAST_PROMPT
    workflow["last_node_id"] = node_id
    workflow["last_link_id"] = link_id


def render_workflow(template: Path, output_name: str | None = None) -> str:
    workflow = _replace_strings(json.loads(template.read_text()))
    _apply_owner_turbo_recipe(workflow, template.name)
    replaced = 0
    for node in _nodes(workflow):
        if node.get("type") == "CreateVideo":
            values = node.setdefault("widgets_values", [])
            if len(values) == 2:
                values.append("sRGB")
            named = node.setdefault("widgets_values_named", {})
            named.setdefault("color_space", "sRGB")
        if node.get("type") == "CLIPLoader":
            replaced += 1
            node["type"] = NODE_ID
            for input_ in node.get("inputs", []):
                if input_.get("name") == "clip_name":
                    input_["name"] = "encoder_name"
                    input_["localized_name"] = "encoder_name"
                    if isinstance(input_.get("widget"), dict):
                        input_["widget"]["name"] = "encoder_name"
            node["properties"] = {
                "Node name for S&R": NODE_ID,
                "standalone_source": (
                    f"https://huggingface.co/{HF_REPO_ID}/blob/main/"
                    f"{STANDALONE_FILENAME}"
                ),
            }
            node["widgets_values"] = [MODEL_FILENAME, "default"]
            node["widgets_values_named"] = {
                "encoder_name": MODEL_FILENAME,
                "device": "default",
            }
        if (template.name == "video_minimax_h3_r2v.json"
                and node.get("type") == "LoraLoaderModelOnly"
                and node.get("widgets_values", [None])[0] == REF2V_TURBO_LORA):
            node["widgets_values"][0] = REF2V_TURBO_LORA_PATH
            named = node.setdefault("widgets_values_named", {})
            named["lora_name"] = REF2V_TURBO_LORA_PATH
        if node.get("type") == "LoadImage":
            values = node.get("widgets_values")
            if isinstance(values, list) and values:
                values[0] = ""
            named = node.get("widgets_values_named")
            if isinstance(named, dict) and "image" in named:
                named["image"] = ""

    for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
        for input_ in subgraph.get("inputs", []):
            if input_.get("name") == "clip_name":
                input_["name"] = "encoder_name"
                input_["label"] = "AWQ encoder"
    for node in workflow.get("nodes", []):
        named = node.get("widgets_values_named")
        if isinstance(named, dict) and "clip_name" in named:
            named["encoder_name"] = named.pop("clip_name")

    if replaced != 1:
        raise ValueError(f"{template.name}: expected one CLIPLoader, found {replaced}")
    if any(node.get("type") == "CLIPLoader" for node in _nodes(workflow)):
        raise AssertionError(f"{template.name}: native CLIPLoader escaped transformation")
    if sum(node.get("type") == NODE_ID for node in _nodes(workflow)) != 1:
        raise AssertionError(f"{template.name}: standalone loader population is not one")

    if output_name == FIRST_LAST_WORKFLOW:
        _add_last_frame_input(workflow, output_name)

    note = next(
        (node for node in workflow.get("nodes", []) if node.get("type") == "MarkdownNote"),
        None,
    )
    if note is None or not isinstance(note.get("widgets_values"), list):
        raise ValueError(f"{template.name}: no top-level MarkdownNote for provenance")
    variant_note = ""
    if output_name == FIRST_LAST_WORKFLOW:
        variant_note = (
            "**First/last alignment:** the default 5-second duration snaps to "
            "124 frames, so Picture 2 is named at 5.17 seconds in the prompt. "
            "If you change duration, update that final timestamp to the "
            "workflow's snapped frame count divided by 24.\n\n"
        )
    note["widgets_values"][0] = (
        _WORKFLOW_NOTE + variant_note + note["widgets_values"][0]
    )
    if template.name in {"video_minimax_h3_t2v.json", "video_minimax_h3_i2v.json"}:
        note["widgets_values"][0] = (
            _WORKFLOW_NOTE
            + f"**Owner recipe in this copy:** `{TURBO_768P_LORA}` at strength "
              f"{TURBO_768P_STRENGTH:g}, {TURBO_768P_STEPS} render steps, and "
              f"video/audio shift {TURBO_768P_SHIFT['shift_video']:g}/"
              f"{TURBO_768P_SHIFT['shift_audio']:g}. "
              "This is a repo-owned working recipe, not a vendor-attested v1.1 "
              "schedule.\n\n"
            + note["widgets_values"][0].removeprefix(_WORKFLOW_NOTE)
        )
    workflow.setdefault("extra", {})["h3_awq_standalone"] = {
        "source_template": template.name,
        "source_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "loader_node": NODE_ID,
    }
    _validate_workflow_structure(workflow, template.name)
    return json.dumps(workflow, indent=2, ensure_ascii=False) + "\n"


def render_compare_workflow() -> str:
    """Two-clip, side-by-side review graph matching the owner's UI block."""
    load_outputs = [
        {"name": "IMAGE", "type": "IMAGE", "links": None},
        {"name": "frame_count", "type": "INT", "links": None},
        {"name": "audio", "type": "AUDIO", "links": None},
        {"name": "video_info", "type": "VHS_VIDEOINFO", "links": None},
    ]

    def load_node(node_id: int, x: int, link_id: int, title: str) -> dict:
        outputs = json.loads(json.dumps(load_outputs))
        outputs[0]["links"] = [link_id]
        return {
            "id": node_id,
            "type": "VHS_LoadVideo",
            "pos": [x, 120],
            "size": [340, 500],
            "flags": {},
            "order": node_id - 1,
            "mode": 0,
            "inputs": [],
            "outputs": outputs,
            "properties": {"Node name for S&R": "VHS_LoadVideo"},
            "widgets_values": {
                "video": "",
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "AnimateDiff",
            },
            "title": title,
        }

    workflow = {
        "id": "5ed7ef44-7d93-438b-a429-58c78cc6ac37",
        "revision": 0,
        "last_node_id": 5,
        "last_link_id": 3,
        "nodes": [
            load_node(1, 0, 1, "A — choose first encoder clip"),
            load_node(2, 380, 2, "B — choose second encoder clip"),
            {
                "id": 3,
                "type": "ImageConcatMulti",
                "pos": [780, 230],
                "size": [300, 220],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "image_1", "type": "COMFY_MATCHTYPE_V3", "link": 1},
                    {"name": "image_2", "type": "IMAGE,MASK", "link": 2},
                ],
                "outputs": [
                    {"name": "output", "type": "COMFY_MATCHTYPE_V3", "links": [3]},
                ],
                "properties": {"Node name for S&R": "ImageConcatMulti"},
                "widgets_values": [2, "right", True],
            },
            {
                "id": 4,
                "type": "VHS_VideoCombine",
                "pos": [1140, 100],
                "size": [600, 520],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 3},
                    {"name": "audio", "type": "AUDIO", "link": None, "shape": 7},
                    {"name": "meta_batch", "type": "VHS_BatchManager", "link": None,
                     "shape": 7},
                    {"name": "vae", "type": "VAE", "link": None, "shape": 7},
                ],
                "outputs": [
                    {"name": "Filenames", "type": "VHS_FILENAMES", "links": None},
                ],
                "properties": {"Node name for S&R": "VHS_VideoCombine"},
                "widgets_values": {
                    "frame_rate": 24.0,
                    "loop_count": 0,
                    "filename_prefix": "Video/h3_encoder_ab",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 13,
                    "save_metadata": False,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
            {
                "id": 5,
                "type": "MarkdownNote",
                "pos": [780, 500],
                "size": [300, 210],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [
                    "## Encoder A/B viewer\n\nChoose two equal-length 24 fps clips. "
                    "Frames are concatenated left-to-right with matching size.\n\n"
                    "Requires VideoHelperSuite and KJNodes. Audio is intentionally "
                    "not connected so the visual comparison has one unambiguous clock."
                ],
            },
        ],
        "links": [
            [1, 1, 0, 3, 0, "IMAGE"],
            [2, 2, 0, 3, 1, "IMAGE"],
            [3, 3, 0, 4, 0, "IMAGE"],
        ],
        "groups": [],
        "config": {},
        "extra": {
            "h3_awq_standalone": {
                "purpose": "side-by-side encoder A/B review",
                "dependencies": ["VideoHelperSuite", "ComfyUI-KJNodes"],
            },
        },
        "version": 0.4,
    }
    return json.dumps(workflow, indent=2, ensure_ascii=False) + "\n"


def build(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # The loader stays at the repo root deliberately. ComfyUI's custom-node scan
    # walks only the top level of `custom_nodes/`, and `hf download --local-dir`
    # preserves repo-relative paths, so a loader emitted into a subdirectory
    # would land at `custom_nodes/<dir>/loader.py` and never be imported --
    # silently, with no error. Verified against `nodes.py::init_external_custom_nodes`.
    loader = output_dir / STANDALONE_FILENAME
    loader.write_text(render_standalone_loader())
    written.append(loader)

    # Workflows nest under WORKFLOW_SUBDIR so a ComfyUI user can fetch the whole
    # set with one `--include`, and so the workflow browser renders them as a
    # folder. Safe to nest: they are downloaded into `user/default/workflows`,
    # which ComfyUI walks recursively -- unlike custom_nodes/, which it does not.
    workflow_dir = output_dir / WORKFLOW_SUBDIR
    workflow_dir.mkdir(parents=True, exist_ok=True)

    templates = _template_dir()
    for output_name, template_name in WORKFLOWS.items():
        output = workflow_dir / output_name
        output.write_text(render_workflow(templates / template_name, output_name))
        written.append(output)
    compare = workflow_dir / COMPARE_WORKFLOW_FILENAME
    compare.write_text(render_compare_workflow())
    written.append(compare)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in build(args.output_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
