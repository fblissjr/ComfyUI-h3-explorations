#!/usr/bin/env python3
"""Index ComfyUI steering & edit-guidance renders from embedded PNG workflow graphs
into a queryable SQLite database and DuckDB-ready JSONL file.

Every `VHS_VideoCombine` output saved with `save_metadata=True` writes a companion
`*_00001.png` whose PNG text chunk `prompt` contains the exact executed ComfyUI API
graph JSON. This tool walks one or more render directories (default root:
`$H3_OUTPUT_DIR/Video`), parses each embedded graph, joins control-vector
`.safetensors` header metadata (pure Python, zero GPU/torch), tags deterministic
configuration and prompt failure modes, and writes:

  - `--out-db`: SQLite database (`renders`, `render_vectors`, `render_problems`
    tables + `v_problem_summary`, `v_renders_flat`, `v_vector_health` views)
  - `--out-jsonl`: flat one-JSON-object-per-render file (DuckDB / jq ready)

Examples:
    "$PY" bench/index_steering_renders.py --dirs dir_a dir_b \\
        --out-db data/steering_audit.sqlite --out-jsonl data/steering_audit.jsonl
    "$PY" bench/index_steering_renders.py --since 2026-11-19 \\
        --out-db data/steering_audit.sqlite
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import re
import sqlite3
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows import h3_env  # noqa: E402

SCHEMA_VERSION = "h3_steering_render_index/1"

_SECTION_RE = re.compile(r"^([a-z][a-z0-9_]*)\s*:\s*(.*)$")
_KNOWN_H3_SECTIONS = frozenset({
    "summary",
    "subject_definitions",
    "retention_analysis",
    "detailed_description",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
    "technical_parameters",
})

_STALE_BASELINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("neat_casual_clothes", re.compile(r"\bneat casual clothes\b", re.IGNORECASE)),
    ("softly_lit_room", re.compile(r"\bsoftly lit room\b", re.IGNORECASE)),
    ("steady_medium_frame", re.compile(r"\bsteady medium frame\b", re.IGNORECASE)),
    ("cinematic_live_action_style", re.compile(r"\bcinematic live-action style\b", re.IGNORECASE)),
    ("warmly_lit_apartment", re.compile(r"\bwarmly lit apartment\b", re.IGNORECASE)),
)


def _is_link(val: Any) -> bool:
    return isinstance(val, list) and len(val) == 2 and isinstance(val[0], (str, int)) and isinstance(val[1], int)


def _node_by_id(graph: dict[str, Any], nid: Any) -> dict[str, Any] | None:
    if nid is None:
        return None
    return graph.get(str(nid))


def _resolve_scalar(graph: dict[str, Any], val: Any, *, depth: int = 0) -> Any:
    """Resolve scalar values through Primitive/String/Int/Float passthrough nodes."""
    if depth > 12 or not _is_link(val):
        return val
    node = _node_by_id(graph, val[0])
    if not node:
        return None
    ct = str(node.get("class_type") or "")
    inp = node.get("inputs") or {}
    for key in ("value", "string", "text", "prompt", "number", "int", "float"):
        if key in inp:
            return _resolve_scalar(graph, inp[key], depth=depth + 1)
    if ct == "MiniMaxH3Resolution":
        # Output 0=width, 1=height, 2=length
        slot = int(val[1])
        if slot == 2:
            return _resolve_scalar(graph, inp.get("length"), depth=depth + 1)
        shape = str(inp.get("shape") or "")
        for res_key in ("shape.wide_resolution", "shape.tall_resolution", "shape.square_resolution", "resolution"):
            if res_key in inp and isinstance(inp[res_key], str) and "x" in inp[res_key]:
                w_s, _, h_s = inp[res_key].partition("x")
                try:
                    return int(w_s) if slot == 0 else int(h_s)
                except ValueError:
                    pass
        if "x" in shape:
            w_s, _, h_s = shape.partition("x")
            try:
                return int(w_s) if slot == 0 else int(h_s)
            except ValueError:
                pass
    if ct == "MiniMaxH3AutoEditPrompt":
        slot = int(val[1])
        if slot == 0:
            return _resolve_scalar(graph, inp.get("original"), depth=depth + 1)
    return None


def parse_sections(prompt: str) -> dict[str, str]:
    """Parse structured H3 `section_name: ...` blocks into `{name: body}`."""
    if not prompt:
        return {}
    sections: dict[str, list[str]] = {}
    cur: str | None = None
    for line in prompt.splitlines():
        m = _SECTION_RE.match(line.strip())
        if m and m.group(1) in _KNOWN_H3_SECTIONS:
            cur = m.group(1)
            sections.setdefault(cur, [])
            if m.group(2).strip():
                sections[cur].append(m.group(2).strip())
        elif cur is not None:
            sections[cur].append(line.rstrip())
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def read_safetensors_meta(path: Path) -> dict[str, Any]:
    """Read JSON `__metadata__` from a `.safetensors` header in pure Python."""
    try:
        with path.open("rb") as f:
            hdr_len_bytes = f.read(8)
            if len(hdr_len_bytes) < 8:
                return {}
            hdr_len = struct.unpack("<Q", hdr_len_bytes)[0]
            if hdr_len <= 0 or hdr_len > 64 * 1024 * 1024:
                return {}
            hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        raw_meta = hdr.get("__metadata__") or {}
        out: dict[str, Any] = {}
        for k, v in raw_meta.items():
            if isinstance(v, str):
                try:
                    out[k] = json.loads(v)
                except json.JSONDecodeError:
                    out[k] = v
            else:
                out[k] = v
        return out
    except Exception:
        return {}


def build_vector_catalog(vector_roots: list[Path]) -> dict[str, dict[str, Any]]:
    """Index available `.safetensors` control vectors by relative name and basename."""
    catalog: dict[str, dict[str, Any]] = {}
    for root in vector_roots:
        if not root.is_dir():
            continue
        for sf in sorted(root.rglob("*.safetensors")):
            meta = read_safetensors_meta(sf)
            if not meta:
                continue
            qrep = meta.get("quality_report") if isinstance(meta.get("quality_report"), dict) else {}
            points = meta.get("points") if isinstance(meta.get("points"), dict) else {}
            pr = points.get("post_refiner") if isinstance(points.get("post_refiner"), dict) else {}
            pairs_meta = meta.get("pairs") if isinstance(meta.get("pairs"), dict) else {}
            jl = qrep.get("j_lens_readout") if isinstance(qrep.get("j_lens_readout"), dict) else {}
            bs = jl.get("b_side_alignment") if isinstance(jl.get("b_side_alignment"), dict) else {}
            ppc_list = qrep.get("per_pair_cosine") if isinstance(qrep.get("per_pair_cosine"), list) else []
            min_ppc = _opt_float(qrep.get("min_pairwise_cosine"))
            if min_ppc is None and ppc_list:
                try:
                    min_ppc = float(min(ppc_list))
                except (TypeError, ValueError):
                    min_ppc = None
            n_pairs = _opt_int(pairs_meta.get("count") or pairs_meta.get("n_pairs")) or (len(ppc_list) or None)
            bs_warn_raw = bs.get("warning")
            bs_warn = bool(bs_warn_raw) if "warning" in bs else None
            bs_status = str(bs.get("status") or ("warn" if bs_warn else ("ok" if bs else "")))
            rec = {
                "vector_file": sf.relative_to(root).as_posix(),
                "vector_name": str(meta.get("name") or sf.stem),
                "vector_family": str(meta.get("family") or ""),
                "raw_norm": _opt_float(pr.get("raw_norm")),
                "mpc": _opt_float(qrep.get("mean_pairwise_cosine")),
                "min_ppc": min_ppc,
                "split_half": _opt_float(qrep.get("split_half_cosine")),
                "n_pairs": n_pairs,
                "n_holds": _opt_int(pairs_meta.get("n_holds")) or 0,
                "b_side_status": bs_status,
                "b_side_warning": bs_warn,
            }
            rel_key = sf.relative_to(root).as_posix()
            catalog[rel_key] = rec
            catalog.setdefault(sf.name, rec)
            vname = rec["vector_name"]
            vfam = rec["vector_family"]
            if vname:
                if vfam:
                    catalog[f"{vname}::{vfam}"] = rec
                    # Prefer newer build if duplicate stem+family exists across folders
                    if sf.parent.name == "v2":
                        catalog[f"{vname}::{vfam}"] = rec
                catalog.setdefault(vname, rec)
                if sf.parent.name == "v2":
                    catalog[vname] = rec
    return catalog


def _opt_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _extract_ref_images(graph: dict[str, Any], ref_link: Any) -> list[str]:
    """Walk chained `MiniMaxH3AppendRefImage` nodes to collect ordered `<Picture 1..N>` basenames."""
    out: list[str] = []
    seen: set[str] = set()
    cur = ref_link
    while _is_link(cur):
        nid = str(cur[0])
        if nid in seen:
            break
        seen.add(nid)
        node = _node_by_id(graph, nid)
        if not node:
            break
        ct = str(node.get("class_type") or "")
        inp = node.get("inputs") or {}
        if ct == "MiniMaxH3AppendRefImage":
            img_link = inp.get("image")
            img_name = _resolve_image_name(graph, img_link)
            if img_name:
                out.append(img_name)
            cur = inp.get("references")
        else:
            break
    out.reverse()
    return out


def _resolve_image_name(graph: dict[str, Any], val: Any, *, depth: int = 0) -> str | None:
    if depth > 8:
        return None
    if isinstance(val, str):
        return Path(val).name
    if not _is_link(val):
        return None
    node = _node_by_id(graph, val[0])
    if not node:
        return None
    ct = str(node.get("class_type") or "")
    inp = node.get("inputs") or {}
    if ct == "LoadImage":
        raw = inp.get("image")
        return Path(str(raw)).name if raw else None
    for key in ("image", "images", "pixels"):
        if key in inp:
            found = _resolve_image_name(graph, inp[key], depth=depth + 1)
            if found:
                return found
    return None


def _extract_conditioning_prompt(graph: dict[str, Any], cond_link: Any, *, depth: int = 0) -> dict[str, Any]:
    """Extract prompt, family, length, width, height, and ref images from a conditioning link."""
    info: dict[str, Any] = {
        "prompt": None,
        "family_hint": None,
        "length": None,
        "width": None,
        "height": None,
        "ref_images": [],
        "auto_edit_node": None,
    }
    if not _is_link(cond_link) or depth > 10:
        return info
    node = _node_by_id(graph, cond_link[0])
    if not node:
        return info
    ct = str(node.get("class_type") or "")
    inp = node.get("inputs") or {}

    # Step through passthrough conditioning wrappers (e.g. MiniMaxH3Preflight, MiniMaxH3GuideLatent, ConditioningSet*)
    if ct == "MiniMaxH3Preflight":
        slot = int(cond_link[1])
        next_key = "edited_conditioning" if (slot == 1 and "edited_conditioning" in inp) else "conditioning"
        return _extract_conditioning_prompt(graph, inp.get(next_key), depth=depth + 1)
    if ct in ("MiniMaxH3GuideLatent", "MiniMaxH3SoundCues", "MiniMaxH3ReferenceNoise", "ConditioningSetTimestepRange"):
        for k in ("conditioning", "positive"):
            if k in inp:
                return _extract_conditioning_prompt(graph, inp[k], depth=depth + 1)

    if "prompt" in inp:
        p_raw = inp["prompt"]
        if _is_link(p_raw):
            p_node = _node_by_id(graph, p_raw[0])
            if p_node and p_node.get("class_type") == "MiniMaxH3AutoEditPrompt":
                info["auto_edit_node"] = p_node
        p_val = _resolve_scalar(graph, p_raw)
        if isinstance(p_val, str):
            info["prompt"] = p_val

    if ct == "MiniMaxH3Conditioning":
        info["family_hint"] = "fl2va"
        info["length"] = _opt_int(_resolve_scalar(graph, inp.get("length")))
        info["width"] = _opt_int(_resolve_scalar(graph, inp.get("width")))
        info["height"] = _opt_int(_resolve_scalar(graph, inp.get("height")))
    elif ct == "MiniMaxH3ReferenceConditioning":
        info["family_hint"] = "ref2va"
        info["length"] = _opt_int(_resolve_scalar(graph, inp.get("length")))
        info["width"] = _opt_int(_resolve_scalar(graph, inp.get("width")))
        info["height"] = _opt_int(_resolve_scalar(graph, inp.get("height")))
        info["ref_images"] = _extract_ref_images(graph, inp.get("references"))
    elif ct == "MiniMaxH3PromptOnReferences":
        info["family_hint"] = "ref2va"
        enc_link = inp.get("encoded_references")
        if _is_link(enc_link):
            enc_node = _node_by_id(graph, enc_link[0])
            if enc_node:
                e_inp = enc_node.get("inputs") or {}
                info["length"] = _opt_int(_resolve_scalar(graph, e_inp.get("length")))
                info["width"] = _opt_int(_resolve_scalar(graph, e_inp.get("width")))
                info["height"] = _opt_int(_resolve_scalar(graph, e_inp.get("height")))
                info["ref_images"] = _extract_ref_images(graph, e_inp.get("references"))
    elif ct in ("MiniMaxH3ImageConditioning", "MiniMaxH3PromptOnImageReferences", "MiniMaxH3PH3ImageConditioning"):
        info["family_hint"] = "h3_image"
        info["length"] = 1
    return info


def _analyze_mask_subgraph(graph: dict[str, Any], mask_link: Any, *, depth: int = 0) -> dict[str, Any]:
    """Inspect a `MiniMaxH3EditGuider.mask` subgraph for temporal curve patterns (e.g. SolidMask zero-prefix)."""
    if not _is_link(mask_link) or depth > 12:
        return {"present": False, "curve": None, "zero_prefix_frames": 0, "segments": []}
    node = _node_by_id(graph, mask_link[0])
    if not node:
        return {"present": True, "curve": "unknown", "zero_prefix_frames": 0, "segments": []}
    ct = str(node.get("class_type") or "")
    inp = node.get("inputs") or {}

    if ct == "SolidMask":
        val = _opt_float(_resolve_scalar(graph, inp.get("value")))
        return {
            "present": True,
            "curve": f"solid_{val}" if val is not None else "solid",
            "zero_prefix_frames": 1 if val == 0.0 else 0,
            "segments": [(val, 1)],
        }
    if ct in ("RepeatImageBatch", "RepeatMaskBatch"):
        sub = _analyze_mask_subgraph(graph, inp.get("image") or inp.get("mask"), depth=depth + 1)
        amt = _opt_int(_resolve_scalar(graph, inp.get("amount"))) or 1
        segs = [(v, n * amt) for (v, n) in sub["segments"]]
        zp = segs[0][1] if (segs and segs[0][0] == 0.0) else 0
        return {"present": True, "curve": sub["curve"], "zero_prefix_frames": zp, "segments": segs}
    if ct in ("MaskToImage", "ImageToMask"):
        sub = _analyze_mask_subgraph(graph, inp.get("mask") or inp.get("image"), depth=depth + 1)
        return sub
    if ct in ("ImageBatch", "MaskBatch", "BatchImagesNode"):
        s1 = _analyze_mask_subgraph(graph, inp.get("image1") or inp.get("mask1"), depth=depth + 1)
        s2 = _analyze_mask_subgraph(graph, inp.get("image2") or inp.get("mask2"), depth=depth + 1)
        segs = list(s1["segments"]) + list(s2["segments"])
        zp = 0
        for v, n in segs:
            if v == 0.0:
                zp += n
            else:
                break
        if len(segs) == 2 and segs[0][0] == 0.0 and (segs[1][0] or 0.0) > 0.0:
            curve = f"sudden_midclip_{segs[0][1]}_{segs[1][1]}"
        elif len(segs) == 2 and (segs[0][0] or 0.0) > 0.0 and segs[1][0] == 0.0:
            curve = f"sudden_cutout_{segs[0][1]}_{segs[1][1]}"
        else:
            curve = "batched_mask"
        return {"present": True, "curve": curve, "zero_prefix_frames": zp, "segments": segs}
    return {"present": True, "curve": ct, "zero_prefix_frames": 0, "segments": []}


def _walk_model_chain(graph: dict[str, Any], model_link: Any) -> dict[str, Any]:
    """Walk the `model` chain back from the guider to extract UNET, LoRAs, Steer stack, and ReleaseReferences."""
    checkpoint: str | None = None
    loras: list[dict[str, Any]] = []
    steers: list[dict[str, Any]] = []
    release_refs: dict[str, Any] | None = None
    seen: set[str] = set()
    cur = model_link

    while _is_link(cur):
        nid = str(cur[0])
        if nid in seen:
            break
        seen.add(nid)
        node = _node_by_id(graph, nid)
        if not node:
            break
        ct = str(node.get("class_type") or "")
        inp = node.get("inputs") or {}

        if ct == "UNETLoader":
            checkpoint = Path(str(inp.get("unet_name") or "")).name or None
            break
        if ct in ("MiniMaxH3LoRABranch", "LoraLoaderModelOnly", "LoraLoader"):
            loras.append({
                "lora_name": Path(str(inp.get("lora_name") or "")).name,
                "strength": _opt_float(inp.get("strength", inp.get("strength_model"))),
                "start_percent": _opt_float(inp.get("start_percent")),
                "end_percent": _opt_float(inp.get("end_percent")),
            })
        elif ct == "MiniMaxH3Steer":
            vec_val = inp.get("vector")
            vec_file = None
            if isinstance(vec_val, str):
                vec_file = vec_val
            elif _is_link(vec_val):
                v_node = _node_by_id(graph, vec_val[0])
                if v_node:
                    v_inp = v_node.get("inputs") or {}
                    vec_file = str(v_inp.get("vector") or v_inp.get("vector_name") or v_inp.get("path") or "")
            steers.append({
                "node_id": nid,
                "vector_file": vec_file or "",
                "strength": _opt_float(_resolve_scalar(graph, inp.get("strength"))),
                "mode": str(_resolve_scalar(graph, inp.get("mode")) or ""),
                "target": str(_resolve_scalar(graph, inp.get("target")) or ""),
                "start_percent": _opt_float(_resolve_scalar(graph, inp.get("start_percent"))),
                "end_percent": _opt_float(_resolve_scalar(graph, inp.get("end_percent"))),
                "streams": str(_resolve_scalar(graph, inp.get("streams")) or "both"),
                "edit_pass": str(_resolve_scalar(graph, inp.get("edit_pass")) or "both"),
                "hold_words": str(_resolve_scalar(graph, inp.get("hold_words")) or ""),
                "hold_alpha": _opt_float(_resolve_scalar(graph, inp.get("hold_alpha"))),
            })
        elif ct == "MiniMaxH3ReleaseReferences":
            release_refs = {
                "end_percent": _opt_float(_resolve_scalar(graph, inp.get("end_percent"))),
                "video_end_percent": _opt_float(_resolve_scalar(graph, inp.get("video_end_percent"))),
                "audio_end_percent": _opt_float(_resolve_scalar(graph, inp.get("audio_end_percent"))),
                "clean_level": _opt_float(_resolve_scalar(graph, inp.get("clean_level"))),
                "edit_pass": str(_resolve_scalar(graph, inp.get("edit_pass")) or "both"),
            }
        cur = inp.get("model")

    loras.reverse()
    steers.reverse()
    return {
        "checkpoint": checkpoint,
        "loras": loras,
        "steers": steers,
        "release_refs": release_refs,
    }


def _probe_audio_rms(mp4_path: Path) -> tuple[float | None, float | None]:
    """Return `(mean_volume_db, max_volume_db)` via ffmpeg volumedetect if available."""
    if not mp4_path.is_file():
        return None, None
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(mp4_path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        mean_m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", proc.stderr)
        max_m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", proc.stderr)
        return (
            float(mean_m.group(1)) if mean_m else None,
            float(max_m.group(1)) if max_m else None,
        )
    except Exception:
        return None, None


def extract_render_record(
    png_path: Path,
    root: Path,
    vector_catalog: dict[str, dict[str, Any]],
    compiled_by_id: dict[str, dict[str, Any]],
    *,
    with_audio: bool = False,
) -> dict[str, Any] | None:
    """Parse one PNG's embedded ComfyUI `prompt` graph into a normalized audit record."""
    try:
        with Image.open(png_path) as im:
            raw_prompt = im.info.get("prompt")
    except Exception:
        return None
    if not raw_prompt:
        return None
    try:
        graph = json.loads(raw_prompt)
    except json.JSONDecodeError:
        return None
    if not isinstance(graph, dict):
        return None

    rel_png = png_path.relative_to(root).as_posix()
    folder = rel_png.split("/", 1)[0] if "/" in rel_png else "."
    subfolder = str(png_path.parent.relative_to(root).as_posix())
    stem = re.sub(r"_\d{5}$", "", png_path.stem)
    stat = png_path.stat()
    mtime_utc = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Find companion mp4 (-audio.mp4 preferred)
    mp4_audio = png_path.with_name(f"{png_path.stem}-audio.mp4")
    mp4_plain = png_path.with_suffix(".mp4")
    mp4_path = mp4_audio if mp4_audio.is_file() else (mp4_plain if mp4_plain.is_file() else None)

    # Locate sampler -> guider, noise, sigmas
    sampler_node = None
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") in ("SamplerCustomAdvanced", "SamplerCustom", "KSampler"):
            sampler_node = node
            break

    seed: int | None = None
    steps: int | None = None
    guider_node = None
    if sampler_node:
        s_inp = sampler_node.get("inputs") or {}
        if _is_link(s_inp.get("guider")):
            guider_node = _node_by_id(graph, s_inp["guider"][0])
        if _is_link(s_inp.get("noise")):
            n_node = _node_by_id(graph, s_inp["noise"][0])
            if n_node:
                seed = _opt_int(_resolve_scalar(graph, (n_node.get("inputs") or {}).get("noise_seed")))
        elif "seed" in s_inp:
            seed = _opt_int(_resolve_scalar(graph, s_inp.get("seed")))
        if _is_link(s_inp.get("sigmas")):
            sig_node = _node_by_id(graph, s_inp["sigmas"][0])
            if sig_node and sig_node.get("class_type") == "ManualSigmas":
                sig_str = str((sig_node.get("inputs") or {}).get("sigmas") or "")
                vals = [x.strip() for x in sig_str.split(",") if x.strip()]
                if len(vals) >= 2:
                    steps = len(vals) - 1
            elif sig_node:
                steps = _opt_int(_resolve_scalar(graph, (sig_node.get("inputs") or {}).get("steps")))

    # Fallback guider lookup if sampler not found
    if guider_node is None:
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type") in ("MiniMaxH3EditGuider", "BasicGuider", "CFGGuider"):
                guider_node = node
                break

    guider_ct = str(guider_node.get("class_type") or "") if guider_node else ""
    g_inp = (guider_node.get("inputs") or {}) if guider_node else {}
    model_chain = _walk_model_chain(graph, g_inp.get("model"))

    original_prompt: str | None = None
    edited_prompt: str | None = None
    family: str | None = None
    length: int | None = None
    width: int | None = None
    height: int | None = None
    ref_images: list[str] = []
    auto_edit_live = False

    edit_scale: float | None = None
    edit_audio_scale: float | None = None
    first_step: int | None = None
    last_step: int | None = None
    edit_streams: str | None = None
    mask_info: dict[str, Any] = {"present": False, "curve": None, "zero_prefix_frames": 0, "segments": []}

    if guider_ct == "MiniMaxH3EditGuider":
        orig_c = _extract_conditioning_prompt(graph, g_inp.get("original"))
        edit_c = _extract_conditioning_prompt(graph, g_inp.get("edited"))
        original_prompt = orig_c["prompt"]
        edited_prompt = edit_c["prompt"]
        family = edit_c["family_hint"] or orig_c["family_hint"]
        length = edit_c["length"] or orig_c["length"]
        width = edit_c["width"] or orig_c["width"]
        height = edit_c["height"] or orig_c["height"]
        ref_images = edit_c["ref_images"] or orig_c["ref_images"]
        auto_edit_live = bool(edit_c["auto_edit_node"] or orig_c["auto_edit_node"])
        edit_scale = _opt_float(_resolve_scalar(graph, g_inp.get("scale")))
        edit_audio_scale = _opt_float(_resolve_scalar(graph, g_inp.get("audio_scale")))
        first_step = _opt_int(_resolve_scalar(graph, g_inp.get("first_step")))
        last_step = _opt_int(_resolve_scalar(graph, g_inp.get("last_step")))
        edit_streams = str(_resolve_scalar(graph, g_inp.get("streams")) or "")
        mask_info = _analyze_mask_subgraph(graph, g_inp.get("mask"))
    elif guider_ct in ("BasicGuider", "CFGGuider"):
        c_key = "conditioning" if guider_ct == "BasicGuider" else "positive"
        c_info = _extract_conditioning_prompt(graph, g_inp.get(c_key))
        original_prompt = c_info["prompt"]
        family = c_info["family_hint"]
        length = c_info["length"]
        width = c_info["width"]
        height = c_info["height"]
        ref_images = c_info["ref_images"]

    # Infer family fallback from checkpoint name
    ckpt = model_chain["checkpoint"] or ""
    if not family:
        low_ckpt = ckpt.lower()
        if "ref2va" in low_ckpt or "r2v" in low_ckpt:
            family = "ref2va"
        elif "image" in low_ckpt:
            family = "h3_image"
        elif ckpt:
            family = "fl2va"

    # Determine method (ignore inert strength==0.0 steers so unsteered baselines classify as 'plain')
    active_steers = [st for st in model_chain["steers"] if (st.get("strength") is None or abs(float(st["strength"])) > 1e-9)]
    steers = active_steers
    has_steers = len(steers) > 0
    is_edit_guider = guider_ct == "MiniMaxH3EditGuider"
    is_fast_1x = (
        is_edit_guider
        and edit_scale == 1.0
        and (edit_audio_scale in (None, 1.0))
        and (first_step in (None, 0))
        and (edit_streams in (None, "", "both"))
        and not mask_info["present"]
    )
    if is_edit_guider and has_steers:
        method = "hybrid"
    elif is_fast_1x:
        method = "fast_1x"
    elif is_edit_guider:
        method = "edit_guidance"
    elif has_steers:
        method = "control_vector"
    else:
        method = "plain"

    # Enrich steers with vector catalog metadata (prefer exact family match BEFORE bare stem!)
    enriched_steers: list[dict[str, Any]] = []
    for idx, st in enumerate(steers):
        vf = st["vector_file"]
        fam_key = f"{vf}::{family}" if (vf and family) else ""
        cat = (
            (vector_catalog.get(fam_key) if fam_key else None)
            or vector_catalog.get(vf)
            or vector_catalog.get(Path(vf).name)
            or vector_catalog.get(Path(vf).stem)
            or {}
        )
        row = dict(st)
        row["order_idx"] = idx
        row["vector_file"] = cat.get("vector_file") or vf
        row["vector_name"] = cat.get("vector_name") or Path(vf).stem
        row["raw_norm"] = cat.get("raw_norm")
        row["mpc"] = cat.get("mpc")
        row["min_ppc"] = cat.get("min_ppc")
        row["split_half"] = cat.get("split_half")
        row["n_pairs"] = cat.get("n_pairs")
        row["n_holds"] = cat.get("n_holds")
        row["b_side_status"] = cat.get("b_side_status")
        row["b_side_warning"] = cat.get("b_side_warning")
        enriched_steers.append(row)

    # Prompt diff metrics
    orig_sections = parse_sections(original_prompt or "")
    edit_sections = parse_sections(edited_prompt or "") if edited_prompt is not None else {}
    orig_words = len((original_prompt or "").split())
    edit_words = len((edited_prompt or "").split()) if edited_prompt is not None else None
    prompt_similarity: float | None = None
    soundscape_changed: bool | None = None
    retention_changed: bool | None = None
    retained_phrases: list[str] = []

    if original_prompt is not None and edited_prompt is not None:
        prompt_similarity = round(difflib.SequenceMatcher(None, original_prompt, edited_prompt).ratio(), 4)
        if "overall_soundscape" in orig_sections or "overall_soundscape" in edit_sections:
            soundscape_changed = orig_sections.get("overall_soundscape", "") != edit_sections.get("overall_soundscape", "")
        if "retention_analysis" in orig_sections or "retention_analysis" in edit_sections:
            retention_changed = orig_sections.get("retention_analysis", "") != edit_sections.get("retention_analysis", "")
        for tag, pat in _STALE_BASELINE_PATTERNS:
            if pat.search(original_prompt) and pat.search(edited_prompt):
                retained_phrases.append(tag)
    elif original_prompt is not None:
        for tag, pat in _STALE_BASELINE_PATTERNS:
            if pat.search(original_prompt):
                retained_phrases.append(tag)

    # Optional join with compiled JSONL
    comp = compiled_by_id.get(stem) or compiled_by_id.get(png_path.stem) or {}

    # Optional audio volume check
    audio_mean_db, audio_max_db = _probe_audio_rms(mp4_path) if (with_audio and mp4_path) else (None, None)

    rel = model_chain["release_refs"] or {}
    ref_end_percent = rel.get("end_percent")
    video_end_percent = rel.get("video_end_percent")
    audio_end_percent = rel.get("audio_end_percent")
    ref_clean_level = rel.get("clean_level")

    # Deterministic problem flags
    problems: list[tuple[str, str]] = []

    if family == "ref2va":
        # Flag ref_pinned_1_0 only when it risks fighting an edit/deformation (e.g. baked EditGuider
        # with layout lock first_step>=1, or a control vector steering physique/wardrobe/action), NOT
        # on live MiniMaxH3AutoEditPrompt graphs (which resolve edit_plan at runtime), plain baselines,
        # or live-action fast_1x where rel=1.0 preserves crisper face likeness.
        pin_risk = (
            (is_edit_guider and not is_fast_1x and not auto_edit_live and (first_step is not None and first_step >= 1))
            or any(
                any(k in str(st.get("vector_name", "")).lower() for k in ("wardrobe", "subj_", "act_", "motion_", "golem", "clay", "origami", "muppet", "inflate"))
                for st in enriched_steers
            )
        )
        if pin_risk and (ref_end_percent is None or ref_end_percent >= 0.99):
            problems.append(("ref_pinned_1_0", f"ref_end_percent={ref_end_percent}"))
        if len(ref_images) == 0:
            problems.append(("ref2va_missing_ref_image", "no LoadImage found in AppendRefImage chain"))
        if original_prompt and ("<Picture 2>" in original_prompt or "beside a partner" in original_prompt.lower()) and len(ref_images) < 2:
            problems.append(("ref_count_mismatch", f"prompt implies >=2 subjects but only {len(ref_images)} ref image(s) attached"))

    if is_edit_guider and not is_fast_1x:
        if first_step is not None and first_step >= 1:
            problems.append(("first_step_1_layout_lock", f"first_step={first_step}, last_step={last_step}"))
        if mask_info["zero_prefix_frames"] > 0:
            problems.append(("sudden_midclip_zero_prefix", f"curve={mask_info['curve']} zeroes leading {mask_info['zero_prefix_frames']} frames"))
        if soundscape_changed is False:
            problems.append(("soundscape_locked_dry", "overall_soundscape identical between original and edited"))
        if family == "ref2va" and retention_changed is False:
            problems.append(("retention_cancelled_zero", "retention_analysis identical between original and edited (cancels in v1-v0)"))
        if prompt_similarity is not None and prompt_similarity >= 0.70:
            problems.append(("high_prompt_similarity", f"SequenceMatcher similarity={prompt_similarity:.3f}"))
        if any(p in retained_phrases for p in ("neat_casual_clothes", "softly_lit_room", "warmly_lit_apartment")):
            problems.append(("stale_baseline_phrase_leak", f"retained={','.join(retained_phrases)}"))

    if orig_words > 0 and orig_words < 100 and family in ("fl2va", "ref2va"):
        problems.append(("short_stub_prompt", f"original_prompt has only {orig_words} words (<100)"))

    for st in enriched_steers:
        mpc = st.get("mpc")
        sh = st.get("split_half")
        if (mpc is not None and mpc < 0.62) or (sh is not None and sh < 0.88):
            problems.append((
                "low_mpc_vector_inflated" if st.get("mode") == "add_norm_matched" else "low_mpc_vector",
                f"{st['vector_name']} mpc={mpc} sh={sh} raw_norm={st.get('raw_norm')} mode={st.get('mode')}",
            ))
        if family == "ref2va" and st.get("target") == "look_only":
            problems.append(("look_only_on_ref2va", f"{st['vector_name']} targets look_only on ref2va"))
        if (st.get("n_holds") or 0) > 0:
            problems.append(("baked_holds_present", f"{st['vector_name']} has n_holds={st['n_holds']}"))
        if st.get("b_side_warning") is True:
            problems.append(("j_lens_b_side_warning", f"{st['vector_name']} b_side_status={st.get('b_side_status')}"))

    eff_prompt_for_audio = (edited_prompt or original_prompt or "").lower()
    explicitly_silent = bool(
        re.search(r"\b(?:neither\b.*\bspeaks|produces no vocal sound|remains quiet|none of the .* speaks)\b", eff_prompt_for_audio)
    )
    if audio_mean_db is not None and audio_mean_db < -35.0 and not explicitly_silent:
        problems.append(("audio_very_quiet", f"mean_volume={audio_mean_db:.1f} dB"))

    # Deduplicate problem codes preserving order
    dedup_problems: list[tuple[str, str]] = []
    seen_codes: set[str] = set()
    for code, detail in problems:
        if code not in seen_codes:
            seen_codes.add(code)
            dedup_problems.append((code, detail))

    return {
        "schema_version": SCHEMA_VERSION,
        "render_id": rel_png,
        "folder": folder,
        "subfolder": subfolder,
        "stem": stem,
        "png_relpath": rel_png,
        "mp4_relpath": mp4_path.relative_to(root).as_posix() if mp4_path else None,
        "mp4_bytes": mp4_path.stat().st_size if mp4_path else None,
        "mtime_utc": mtime_utc,
        "mtime_epoch": stat.st_mtime,
        "method": method,
        "family": family,
        "checkpoint": ckpt,
        "loras": model_chain["loras"],
        "width": width,
        "height": height,
        "length": length,
        "seed": seed,
        "steps": steps,
        "ref_count": len(ref_images),
        "ref_images": ref_images,
        "auto_edit_live": auto_edit_live,
        "original_prompt": original_prompt,
        "edited_prompt": edited_prompt,
        "orig_words": orig_words,
        "edit_words": edit_words,
        "prompt_similarity": prompt_similarity,
        "soundscape_changed": soundscape_changed,
        "retention_changed": retention_changed,
        "retained_phrases": retained_phrases,
        "edit_scale": edit_scale,
        "edit_audio_scale": edit_audio_scale,
        "first_step": first_step,
        "last_step": last_step,
        "edit_streams": edit_streams,
        "mask_present": mask_info["present"],
        "temporal_curve": mask_info["curve"],
        "zero_prefix_frames": mask_info["zero_prefix_frames"],
        "ref_end_percent": ref_end_percent,
        "video_end_percent": video_end_percent,
        "audio_end_percent": audio_end_percent,
        "ref_clean_level": ref_clean_level,
        "audio_mean_db": audio_mean_db,
        "audio_max_db": audio_max_db,
        "compiled_repair_used": comp.get("repair_used") if comp else None,
        "compiled_status": comp.get("status") if comp else None,
        "vectors": enriched_steers,
        "problems": [{"code": c, "detail": d} for c, d in dedup_problems],
        "problem_codes": [c for c, _ in dedup_problems],
    }


def _load_compiled_jsonl(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for p in paths:
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                for key in ("id", "concept_id", "stem", "name"):
                    val = rec.get(key)
                    if isinstance(val, str) and val:
                        by_id[val] = rec
                        break
    return by_id


def write_sqlite(db_path: Path, records: list[dict[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE renders (
                render_id TEXT PRIMARY KEY,
                folder TEXT NOT NULL,
                subfolder TEXT NOT NULL,
                stem TEXT NOT NULL,
                png_relpath TEXT NOT NULL,
                mp4_relpath TEXT,
                mp4_bytes INTEGER,
                mtime_utc TEXT NOT NULL,
                mtime_epoch REAL NOT NULL,
                method TEXT NOT NULL,
                family TEXT,
                checkpoint TEXT,
                loras_json TEXT,
                width INTEGER,
                height INTEGER,
                length INTEGER,
                seed INTEGER,
                steps INTEGER,
                ref_count INTEGER NOT NULL,
                ref_images_json TEXT,
                auto_edit_live INTEGER NOT NULL,
                orig_words INTEGER,
                edit_words INTEGER,
                prompt_similarity REAL,
                soundscape_changed INTEGER,
                retention_changed INTEGER,
                retained_phrases_json TEXT,
                edit_scale REAL,
                edit_audio_scale REAL,
                first_step INTEGER,
                last_step INTEGER,
                edit_streams TEXT,
                mask_present INTEGER NOT NULL,
                temporal_curve TEXT,
                zero_prefix_frames INTEGER NOT NULL,
                ref_end_percent REAL,
                video_end_percent REAL,
                audio_end_percent REAL,
                ref_clean_level REAL,
                audio_mean_db REAL,
                audio_max_db REAL,
                compiled_repair_used INTEGER,
                compiled_status TEXT,
                problem_count INTEGER NOT NULL,
                problem_codes_json TEXT NOT NULL,
                original_prompt TEXT,
                edited_prompt TEXT
            );

            CREATE TABLE render_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                render_id TEXT NOT NULL REFERENCES renders(render_id),
                order_idx INTEGER NOT NULL,
                vector_file TEXT NOT NULL,
                vector_name TEXT NOT NULL,
                strength REAL,
                mode TEXT,
                target TEXT,
                start_percent REAL,
                end_percent REAL,
                streams TEXT,
                edit_pass TEXT,
                hold_words TEXT,
                hold_alpha REAL,
                raw_norm REAL,
                mpc REAL,
                min_ppc REAL,
                split_half REAL,
                n_pairs INTEGER,
                n_holds INTEGER,
                b_side_status TEXT,
                b_side_warning INTEGER
            );

            CREATE TABLE render_problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                render_id TEXT NOT NULL REFERENCES renders(render_id),
                code TEXT NOT NULL,
                detail TEXT
            );

            CREATE INDEX idx_renders_folder_method ON renders(folder, method);
            CREATE INDEX idx_renders_family ON renders(family);
            CREATE INDEX idx_render_vectors_name ON render_vectors(vector_name);
            CREATE INDEX idx_render_problems_code ON render_problems(code);

            CREATE VIEW v_problem_summary AS
            WITH totals AS (
                SELECT folder, method, COUNT(*) AS n_renders
                FROM renders
                GROUP BY folder, method
            )
            SELECT
                r.folder,
                r.method,
                t.n_renders,
                p.code AS problem_code,
                COUNT(DISTINCT p.render_id) AS n_affected,
                ROUND(100.0 * COUNT(DISTINCT p.render_id) / t.n_renders, 1) AS pct_affected
            FROM render_problems p
            JOIN renders r ON r.render_id = p.render_id
            JOIN totals t ON t.folder = r.folder AND t.method = r.method
            GROUP BY r.folder, r.method, t.n_renders, p.code
            ORDER BY r.folder, r.method, n_affected DESC, p.code;

            CREATE VIEW v_renders_flat AS
            SELECT
                r.render_id,
                r.folder,
                r.subfolder,
                r.stem,
                r.mtime_utc,
                r.method,
                r.family,
                r.length,
                r.seed,
                r.ref_count,
                r.ref_images_json,
                r.orig_words,
                r.edit_words,
                r.prompt_similarity,
                r.soundscape_changed,
                r.retention_changed,
                r.edit_scale,
                r.first_step,
                r.last_step,
                r.temporal_curve,
                r.zero_prefix_frames,
                r.ref_end_percent,
                r.audio_mean_db,
                r.problem_count,
                r.problem_codes_json,
                (SELECT GROUP_CONCAT(rv.vector_name || '@' || COALESCE(rv.strength, ''), '+')
                 FROM render_vectors rv WHERE rv.render_id = r.render_id) AS vectors_summary,
                (SELECT MIN(rv.mpc) FROM render_vectors rv WHERE rv.render_id = r.render_id) AS min_vector_mpc
            FROM renders r;

            CREATE VIEW v_vector_health AS
            SELECT
                rv.vector_name,
                rv.vector_file,
                COUNT(*) AS n_renders,
                ROUND(AVG(rv.strength), 3) AS avg_strength,
                rv.mode,
                rv.target,
                rv.raw_norm,
                rv.mpc,
                rv.split_half,
                rv.n_holds,
                rv.b_side_status
            FROM render_vectors rv
            GROUP BY rv.vector_name, rv.vector_file, rv.mode, rv.target,
                     rv.raw_norm, rv.mpc, rv.split_half, rv.n_holds, rv.b_side_status
            ORDER BY rv.mpc ASC NULLS LAST, n_renders DESC;
            """
        )

        for r in records:
            conn.execute(
                """
                INSERT INTO renders VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    r["render_id"],
                    r["folder"],
                    r["subfolder"],
                    r["stem"],
                    r["png_relpath"],
                    r["mp4_relpath"],
                    r["mp4_bytes"],
                    r["mtime_utc"],
                    r["mtime_epoch"],
                    r["method"],
                    r["family"],
                    r["checkpoint"],
                    json.dumps(r["loras"]),
                    r["width"],
                    r["height"],
                    r["length"],
                    r["seed"],
                    r["steps"],
                    r["ref_count"],
                    json.dumps(r["ref_images"]),
                    int(bool(r["auto_edit_live"])),
                    r["orig_words"],
                    r["edit_words"],
                    r["prompt_similarity"],
                    None if r["soundscape_changed"] is None else int(bool(r["soundscape_changed"])),
                    None if r["retention_changed"] is None else int(bool(r["retention_changed"])),
                    json.dumps(r["retained_phrases"]),
                    r["edit_scale"],
                    r["edit_audio_scale"],
                    r["first_step"],
                    r["last_step"],
                    r["edit_streams"],
                    int(bool(r["mask_present"])),
                    r["temporal_curve"],
                    r["zero_prefix_frames"],
                    r["ref_end_percent"],
                    r["video_end_percent"],
                    r["audio_end_percent"],
                    r["ref_clean_level"],
                    r["audio_mean_db"],
                    r["audio_max_db"],
                    None if r["compiled_repair_used"] is None else int(bool(r["compiled_repair_used"])),
                    r["compiled_status"],
                    len(r["problems"]),
                    json.dumps(r["problem_codes"]),
                    r["original_prompt"],
                    r["edited_prompt"],
                ),
            )
            for v in r["vectors"]:
                conn.execute(
                    """
                    INSERT INTO render_vectors (
                        render_id, order_idx, vector_file, vector_name, strength, mode, target,
                        start_percent, end_percent, streams, edit_pass, hold_words, hold_alpha,
                        raw_norm, mpc, min_ppc, split_half, n_pairs, n_holds, b_side_status, b_side_warning
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        r["render_id"],
                        v["order_idx"],
                        v["vector_file"],
                        v["vector_name"],
                        v["strength"],
                        v["mode"],
                        v["target"],
                        v["start_percent"],
                        v["end_percent"],
                        v["streams"],
                        v["edit_pass"],
                        v["hold_words"],
                        v["hold_alpha"],
                        v["raw_norm"],
                        v["mpc"],
                        v["min_ppc"],
                        v["split_half"],
                        v["n_pairs"],
                        v["n_holds"],
                        v["b_side_status"],
                        None if v["b_side_warning"] is None else int(bool(v["b_side_warning"])),
                    ),
                )
            for p in r["problems"]:
                conn.execute(
                    "INSERT INTO render_problems (render_id, code, detail) VALUES (?,?,?)",
                    (r["render_id"], p["code"], p["detail"]),
                )
        conn.commit()
    finally:
        conn.close()


def write_jsonl(jsonl_path: Path, records: list[dict[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_summary(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No PNG graphs indexed.")
        return
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[(r["folder"], r["method"])].append(r)

    print(f"Indexed {len(records)} renders across {len({r['folder'] for r in records})} folder(s):")
    for (folder, method), rows in sorted(by_group.items()):
        fams = Counter(r["family"] or "?" for r in rows)
        fam_str = ", ".join(f"{k}:{v}" for k, v in sorted(fams.items()))
        probs = Counter(code for r in rows for code in r["problem_codes"])
        top_probs = ", ".join(f"{c}={n}/{len(rows)}" for c, n in probs.most_common(6)) or "none"
        print(f"  - {folder} [{method}] n={len(rows)} ({fam_str}) | flags: {top_probs}")


def discover_pngs(root: Path, dirs: list[str] | None, since_epoch: float | None) -> list[Path]:
    targets = [root / d for d in dirs] if dirs else [root]
    found: list[Path] = []
    for t in targets:
        if not t.exists():
            continue
        for p in sorted(t.rglob("*.png")):
            if since_epoch is not None and p.stat().st_mtime < since_epoch:
                continue
            found.append(p)
    return found


def main(argv: list[str] | None = None) -> int:
    h3_env.load()
    comfy_root = next((p for p in REPO_ROOT.parents if (p / "comfy").is_dir()), REPO_ROOT.parents[1])
    default_root = Path(os.environ["H3_OUTPUT_DIR"]) / "Video" if os.environ.get("H3_OUTPUT_DIR") else (comfy_root / "output" / "Video")
    default_vec_root = comfy_root / "models" / "control_vectors"

    ap = argparse.ArgumentParser(description="Index H3 steering & edit-guidance PNG workflow graphs into SQLite + JSONL.")
    ap.add_argument("--root", type=Path, default=default_root, help="Base render directory (default: $H3_OUTPUT_DIR/Video)")
    ap.add_argument("--dirs", nargs="*", default=None, help="Subdirectories under --root to scan (default: all under --root)")
    ap.add_argument("--since", type=str, default=None, help="Only include PNGs modified on/after YYYY-MM-DD (UTC)")
    ap.add_argument("--vector-roots", nargs="*", type=Path, default=[default_vec_root], help="Control-vector directories for .safetensors metadata join")
    ap.add_argument("--compiled-jsonl", nargs="*", type=Path, default=[], help="Optional compiled JSONL file(s) to join by stem/id")
    ap.add_argument("--with-audio", action="store_true", help="Measure companion -audio.mp4 RMS volume via ffmpeg")
    ap.add_argument("--out-db", type=Path, default=REPO_ROOT / "data" / "steering_renders_audit.sqlite", help="Output SQLite database path")
    ap.add_argument("--out-jsonl", type=Path, default=None, help="Optional output DuckDB-ready JSONL path")
    args = ap.parse_args(argv)

    since_epoch: float | None = None
    if args.since:
        since_dt = dt.datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        since_epoch = since_dt.timestamp()

    vec_catalog = build_vector_catalog(args.vector_roots)
    compiled_map = _load_compiled_jsonl(args.compiled_jsonl)
    pngs = discover_pngs(args.root, args.dirs, since_epoch)

    records: list[dict[str, Any]] = []
    for p in pngs:
        rec = extract_render_record(p, args.root, vec_catalog, compiled_map, with_audio=args.with_audio)
        if rec is not None:
            records.append(rec)

    if args.out_db:
        write_sqlite(args.out_db, records)
    if args.out_jsonl:
        write_jsonl(args.out_jsonl, records)

    print_summary(records)
    if args.out_db:
        print(f"SQLite written: {args.out_db}")
    if args.out_jsonl:
        print(f"JSONL written:  {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
