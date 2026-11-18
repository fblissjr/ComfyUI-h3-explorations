"""Structured ComfyUI graph builder and pipeline wiring helpers for Spatial Director.

Replaces fragile ad-hoc dictionary mutation across ``workflows/*.json`` templates with
explicit handle-based node wiring, reachability pruning, and lean counterfactual
baseline compilation. Keeps top-level imports strictly stdlib.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

HUD_COMBINE_NODE_ID = "13"
CLEAN_DECODE_NODE_ID = "162"
PREVIEW_NODE_ID = "165"
PREVIEW_NODE_CLASS = "ModelPreviewOverrideKJ"


@dataclass(frozen=True)
class NodeHandle:
    """Typed handle referencing a ComfyUI graph node ID and its output slots."""

    node_id: str

    def out(self, slot: int = 0) -> list[Any]:
        return [self.node_id, int(slot)]


class GraphBuilder:
    """Safe builder over ComfyUI prompt graph dictionaries."""

    def __init__(self, graph: dict[str, Any] | None = None) -> None:
        self._graph: dict[str, Any] = copy.deepcopy(graph) if isinstance(graph, dict) else {}

    @classmethod
    def from_template(cls, template_graph: dict[str, Any]) -> GraphBuilder:
        return cls(template_graph)

    def has(self, node_id: str) -> bool:
        return str(node_id) in self._graph and isinstance(self._graph[str(node_id)], dict)

    def get(self, node_id: str) -> dict[str, Any] | None:
        node = self._graph.get(str(node_id))
        return node if isinstance(node, dict) else None

    def inputs(self, node_id: str) -> dict[str, Any]:
        node = self._graph.get(str(node_id))
        if not isinstance(node, dict):
            raise KeyError(f"Graph node {node_id!r} missing")
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            ins = {}
            node["inputs"] = ins
        return ins

    def set(self, node_id: str, class_type: str, inputs: dict[str, Any]) -> NodeHandle:
        nid = str(node_id)
        self._graph[nid] = {"class_type": str(class_type), "inputs": dict(inputs)}
        return NodeHandle(nid)

    def ensure(self, node_id: str, class_type: str, defaults: dict[str, Any] | None = None) -> NodeHandle:
        nid = str(node_id)
        if not self.has(nid):
            self._graph[nid] = {"class_type": str(class_type), "inputs": dict(defaults or {})}
        return NodeHandle(nid)

    def remove(self, *node_ids: str) -> None:
        for nid in node_ids:
            self._graph.pop(str(nid), None)

    def remove_matching(self, class_types: Iterable[str]) -> list[str]:
        targets = set(class_types)
        removed: list[str] = []
        for nid in list(self._graph):
            node = self._graph.get(nid)
            cls = node.get("class_type") if isinstance(node, dict) else ""
            if cls in targets:
                del self._graph[nid]
                removed.append(nid)
        return removed

    def prune_unreachable(self, roots: Iterable[str]) -> list[str]:
        """Drop nodes not transitively referenced by any ID in ``roots``."""
        visited: set[str] = set()
        stack: list[str] = [str(r) for r in roots if self.has(str(r))]

        def _collect_refs(val: Any) -> None:
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], int):
                ref_id = val[0]
                if ref_id not in visited and self.has(ref_id):
                    stack.append(ref_id)
            elif isinstance(val, dict):
                for v in val.values():
                    _collect_refs(v)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    _collect_refs(item)

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            node = self.get(cur)
            if isinstance(node, dict):
                _collect_refs(node.get("inputs") or {})

        dropped: list[str] = []
        for nid in list(self._graph):
            if nid not in visited:
                del self._graph[nid]
                dropped.append(nid)
        return dropped

    def to_dict(self) -> dict[str, Any]:
        return self._graph


def wire_audio_refine_pass(
    graph: dict[str, Any],
    *,
    edited_ref: list[Any],
    steps: int,
    denoise: float,
) -> None:
    """Attach or strip optional 2nd-pass ``MiniMaxH3FrozenVideoCache`` + ``MiniMaxH3AudioRefineMask``."""
    b = GraphBuilder(graph) if not isinstance(graph, GraphBuilder) else graph
    # Caller may pass raw dict; keep mutation in place so existing dict references stay valid.
    raw = graph
    for stale_ref_id in ("80", "81", "82", "83", "84", "85", "86", "87", "88"):
        raw.pop(stale_ref_id, None)

    rsrc: list[Any] = ["1", 0]
    raw["80"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": rsrc, "shift_video": 12.0, "shift_audio": 3.0},
    }
    rsrc = ["80", 0]
    if "58" in raw and isinstance(raw["58"].get("inputs"), dict):
        raw["81"] = {
            "class_type": "ModelAttentionBackend",
            "inputs": {"model": rsrc, "attention": raw["58"]["inputs"].get("attention", "comfy kitchen attention")},
        }
        rsrc = ["81", 0]
    if "21" in raw and isinstance(raw["21"], dict) and raw["21"].get("class_type") == "MiniMaxH3Sol":
        sol_inputs = {k: copy.deepcopy(v) for k, v in (raw["21"].get("inputs") or {}).items() if k != "model"}
        sol_inputs["model"] = rsrc
        raw["82"] = {"class_type": "MiniMaxH3Sol", "inputs": sol_inputs}
        rsrc = ["82", 0]
    raw["88"] = {
        "class_type": "MiniMaxH3FrozenVideoCache",
        "inputs": {
            "model": rsrc,
            "precision": "int4",
            "refresh": False,
            "refresh_every": 2,
            "verify": False,
            "halo": 0,
        },
    }
    rsrc = ["88", 0]
    raw["83"] = {
        "class_type": "MiniMaxH3AudioRefineMask",
        "inputs": {"latent": ["10", 0], "video_mask": 0.0, "audio_mask": 1.0},
    }
    raw["84"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": rsrc,
            "scheduler": "simple",
            "steps": max(1, min(30, int(steps))),
            "denoise": max(0.05, min(1.0, float(denoise))),
        },
    }
    raw["85"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    raw["86"] = {"class_type": "BasicGuider", "inputs": {"model": rsrc, "conditioning": copy.deepcopy(edited_ref)}}
    raw["87"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["86", 0],
            "sampler": ["85", 0],
            "sigmas": ["84", 0],
            "latent_image": ["83", 0],
        },
    }
    raw["11"]["inputs"]["samples"] = ["87", 0]
    raw["12"]["inputs"]["samples"] = ["87", 0]
    del b


def finalize_lean_baseline(graph: dict[str, Any]) -> dict[str, Any]:
    """Strip edited-prompt encoding, HUD compositing, and double-pass audio steering from baseline graphs.

    Keeps the exact noise seed, Reference/GuideLatent chain, clean VHS output (``"162"``),
    TAE preview override (``"165"``), and ``MiniMaxH3EditGuider`` (``"9"`` with ``scale=0.0``
    and ``audio_scale=0.0``) so EditGuider executes its 1-pass ``[original]`` fast path cleanly.
    """
    b = GraphBuilder(graph)

    # Point all downstream `edited` inputs straight to `original` so Qwen3-VL skips the edited prompt node.
    if b.has("169"):
        ins169 = b.inputs("169")
        if "positive" in ins169:
            ins169["edited_positive"] = copy.deepcopy(ins169["positive"])
    if b.has("70"):
        ins70 = b.inputs("70")
        if "positive" in ins70:
            ins70["edited_positive"] = copy.deepcopy(ins70["positive"])
    if b.has("26"):
        ins26 = b.inputs("26")
        if "conditioning" in ins26:
            ins26["edited_conditioning"] = copy.deepcopy(ins26["conditioning"])
    if b.has("86"):
        ins86 = b.inputs("86")
        if b.has("26"):
            ins86["conditioning"] = ["26", 0]
        elif b.has("5"):
            ins86["conditioning"] = ["5", 0]
    if b.has("9"):
        ins9 = b.inputs("9")
        if "original" in ins9:
            ins9["edited"] = copy.deepcopy(ins9["original"])
        ins9["scale"] = 0.0
        ins9["audio_scale"] = 0.0
        ins9.pop("mask", None)

    # Strip unused sprite trajectory references ("51".."54") when the [original] prompt
    # references only <Picture 1> and never invokes <Video N>.
    if b.has("133") and b.has("132") and b.has("50"):
        orig_val = str(b.inputs("133").get("value") or "")
        if "<Video " not in orig_val and b.inputs("132").get("references") == ["51", 0]:
            b.inputs("132")["references"] = ["50", 0]

    b.remove(HUD_COMBINE_NODE_ID, "161")
    b.prune_unreachable([CLEAN_DECODE_NODE_ID])
    return b.to_dict()
