#!/usr/bin/env python3
"""
Build the self-contained interactive HTML Control Vector Explorer (`control_vector_viewer.html`)
and optionally serve it with live reload (`--serve --port 8788`).

Combines:
  1. Input `.toml` contrastive prompt banks (`prompt_bank/control_vectors/*.toml`)
     with word-level prefix/diff/suffix segmentation and `scene_text` vs. `from_first_diff`
     pooling mask diagnostics (`<d>...</d>`, `(S1)`, shared prefix, and unchanged
     `overall_soundscape:` trailing sections).
  2. Output `Qwen3-VL-32B-Instruct` batch-labelling JSON (`h3_control_vector_labels.json`)
     including `layers.49 -> lm_head` lexical log-odds shifts (`delta_log_prob`),
     coupled Gumbel rollouts across `alpha in {-0.15, +0.00, +0.15, +0.22}` on Anchor A
     and Anchor B, Blind Pair-Diff labels, and `layers.49` Steered Self-Reports.
  3. Pairwise `14 x 14` `layers.49` cosine similarity matrix from `h3_control_vector_report.md`.
"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import tomllib
from typing import Any


DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.DOTALL)
SPEAKER_RE = re.compile(r"\(S\d+(?:,\s*S\d+)*\)")
TOKEN_SPLIT_RE = re.compile(r"\S+|\s+")
WORD_SCORE_RE = re.compile(r"^(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)$")


def _resolve_h3_root() -> Path:
    env_h3 = os.environ.get("H3_REPO_ROOT")
    if env_h3 and (Path(env_h3) / "prompt_bank" / "control_vectors").is_dir():
        return Path(env_h3).resolve()
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for parent in [here.parent, *here.parents, cwd, *cwd.parents]:
        if (parent / "prompt_bank" / "control_vectors").is_dir():
            return parent
        candidate = parent / "ComfyUI" / "custom_nodes" / "ComfyUI-h3-explorations"
        if (candidate / "prompt_bank" / "control_vectors").is_dir():
            return candidate
    return here.parent


def _resolve_exl3_model_script_dir(h3_root: Path) -> Path | None:
    env_root = os.environ.get("EXLLAMAV3_ROOT")
    if env_root:
        cand = Path(env_root) / "models" / "Qwen3-VL-32B-Instruct"
        if cand.is_dir():
            return cand.resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if parent.name == "Qwen3-VL-32B-Instruct" and (parent.parent.parent / "exllamav3").is_dir():
            return parent
    if len(h3_root.parents) >= 3:
        base = h3_root.parents[2]
        for candidate in base.glob("*/exllamav3/models/Qwen3-VL-32B-Instruct"):
            if candidate.is_dir():
                return candidate.resolve()
    return None


def segment_dialogue_and_speakers(text: str) -> list[dict[str, str]]:
    """
    Split a text string into ordered spans of kind:
      - 'normal': included in `scene_text`
      - 'dialogue': `<d>...</d>` (excluded by `scene_text` and `from_first_diff`)
      - 'speaker': `(S1)` / `(S1, S2)` (excluded by `scene_text` and `from_first_diff`)
    """
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    for m in DIALOGUE_RE.finditer(text):
        spans.append((m.start(), m.end(), "dialogue"))
    for m in SPEAKER_RE.finditer(text):
        # Avoid overlapping with dialogue if a tag ever appeared inside <d>
        if not any(max(s0, m.start()) < min(s1, m.end()) for s0, s1, _ in spans):
            spans.append((m.start(), m.end(), "speaker"))
    spans.sort(key=lambda x: x[0])

    out: list[dict[str, str]] = []
    pos = 0
    for s0, s1, kind in spans:
        if s0 > pos:
            out.append({"text": text[pos:s0], "kind": "normal"})
        out.append({"text": text[s0:s1], "kind": kind})
        pos = s1
    if pos < len(text):
        out.append({"text": text[pos:], "kind": "normal"})
    return out


def count_active_and_masked_words(text: str) -> tuple[int, int]:
    """Return `(active_words, dialogue_or_speaker_words)` in `text`."""
    if not text:
        return 0, 0
    all_words = len(text.split())
    cleaned = SPEAKER_RE.sub(" ", DIALOGUE_RE.sub(" ", text))
    active_words = len(cleaned.split())
    return active_words, max(0, all_words - active_words)


def analyze_toml_pair(idx: int, pair: dict[str, Any], frame: str) -> dict[str, Any]:
    a_txt = str(pair.get("a", "")).strip()
    b_txt = str(pair.get("b", "")).strip()
    weight = float(pair.get("weight", 1.0))

    a_toks = TOKEN_SPLIT_RE.findall(a_txt)
    b_toks = TOKEN_SPLIT_RE.findall(b_txt)

    px = 0
    min_len = min(len(a_toks), len(b_toks))
    while px < min_len and a_toks[px] == b_toks[px]:
        px += 1

    sx = 0
    while sx < (min_len - px) and a_toks[len(a_toks) - 1 - sx] == b_toks[len(b_toks) - 1 - sx]:
        sx += 1

    shared_prefix = "".join(a_toks[:px])
    a_span = "".join(a_toks[px : len(a_toks) - sx])
    b_span = "".join(b_toks[px : len(b_toks) - sx])
    shared_suffix = "".join(a_toks[len(a_toks) - sx :]) if sx > 0 else ""

    kept_suffix = shared_suffix
    trimmed_soundscape_suffix = ""
    soundscape_identical = False
    edit_section = "bare" if frame == "bare" else "integrated_multimodal_description"

    if frame != "bare":
        for marker in ("\n\noverall_soundscape:", "\noverall_soundscape:"):
            if marker in a_txt and marker in b_txt:
                a_pre, a_tail = a_txt.split(marker, 1)
                b_pre, b_tail = b_txt.split(marker, 1)
                if a_tail == b_tail and a_pre != b_pre:
                    soundscape_identical = True
                    edit_section = "integrated_multimodal_description"
                    if marker in shared_suffix:
                        m_idx = shared_suffix.index(marker)
                        kept_suffix = shared_suffix[:m_idx]
                        trimmed_soundscape_suffix = shared_suffix[m_idx:]
                elif a_pre == b_pre and a_tail != b_tail:
                    soundscape_identical = False
                    edit_section = "overall_soundscape"
                elif a_pre != b_pre and a_tail != b_tail:
                    soundscape_identical = False
                    edit_section = "both"
                break

    pre_act, pre_dlg = count_active_and_masked_words(shared_prefix)
    a_act, a_dlg = count_active_and_masked_words(a_span)
    b_act, b_dlg = count_active_and_masked_words(b_span)
    diff_act_avg = 0.5 * (a_act + b_act)
    diff_dlg_avg = 0.5 * (a_dlg + b_dlg)
    ksuf_act, ksuf_dlg = count_active_and_masked_words(kept_suffix)
    tsuf_act, tsuf_dlg = count_active_and_masked_words(trimmed_soundscape_suffix)

    total_dlg = pre_dlg + diff_dlg_avg + ksuf_dlg + tsuf_dlg
    total_all_words = pre_act + diff_act_avg + ksuf_act + tsuf_act + total_dlg

    scene_text_active = pre_act + diff_act_avg + ksuf_act + tsuf_act
    ffd_active = (diff_act_avg + ksuf_act) if frame != "bare" else scene_text_active

    scene_text_diff_pct = round(100.0 * diff_act_avg / max(scene_text_active, 1e-6), 1)
    ffd_diff_pct = round(100.0 * diff_act_avg / max(ffd_active, 1e-6), 1)

    return {
        "index": idx,
        "weight": weight,
        "a": a_txt,
        "b": b_txt,
        "shared_prefix": shared_prefix,
        "a_span": a_span,
        "b_span": b_span,
        "shared_suffix": shared_suffix,
        "kept_suffix": kept_suffix,
        "trimmed_soundscape_suffix": trimmed_soundscape_suffix,
        "soundscape_identical": soundscape_identical,
        "edit_section": edit_section,
        "segments": {
            "shared_prefix": segment_dialogue_and_speakers(shared_prefix),
            "a_span": segment_dialogue_and_speakers(a_span),
            "b_span": segment_dialogue_and_speakers(b_span),
            "kept_suffix": segment_dialogue_and_speakers(kept_suffix),
            "trimmed_soundscape_suffix": segment_dialogue_and_speakers(trimmed_soundscape_suffix),
        },
        "stats": {
            "total_words_avg": round(total_all_words, 1),
            "prefix_active_words": pre_act,
            "diff_words_a": a_act,
            "diff_words_b": b_act,
            "diff_active_words_avg": round(diff_act_avg, 1),
            "kept_suffix_active_words": ksuf_act,
            "trimmed_soundscape_active_words": tsuf_act,
            "excluded_dialogue_words": round(total_dlg, 1),
            "scene_text_active_words": round(scene_text_active, 1),
            "from_first_diff_active_words": round(ffd_active, 1),
            "scene_text_diff_pct": scene_text_diff_pct,
            "from_first_diff_diff_pct": ffd_diff_pct,
            "prefix_pct_of_total": round(100.0 * pre_act / max(total_all_words, 1e-6), 1),
            "diff_pct_of_total": round(100.0 * diff_act_avg / max(total_all_words, 1e-6), 1),
            "kept_suffix_pct_of_total": round(100.0 * ksuf_act / max(total_all_words, 1e-6), 1),
            "trimmed_soundscape_pct_of_total": round(100.0 * tsuf_act / max(total_all_words, 1e-6), 1),
            "dialogue_pct_of_total": round(100.0 * total_dlg / max(total_all_words, 1e-6), 1),
        },
    }


def parse_word_shifts(items: list[Any]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict) and "word" in item:
            parsed.append(item)
            continue
        raw = str(item).strip()
        m = WORD_SCORE_RE.match(raw)
        if m:
            parsed.append({
                "word": m.group(1).strip(),
                "delta_log_prob": float(m.group(2)),
                "raw": raw,
            })
        else:
            parsed.append({
                "word": raw,
                "delta_log_prob": 0.0,
                "raw": raw,
            })
    return parsed


def parse_cosine_matrix_from_report(report_path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not report_path.is_file():
        return [], {}
    md = report_path.read_text(encoding="utf-8", errors="ignore")
    in_cos = False
    raw_rows: list[tuple[str, list[float]]] = []
    for line in md.splitlines():
        if line.startswith("## 2. Pairwise Cosine Similarity"):
            in_cos = True
            continue
        if in_cos and line.startswith("## "):
            break
        if in_cos and line.startswith("| `") and not line.startswith("| Vector"):
            parts = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            if len(parts) >= 2:
                row_name = parts[0]
                try:
                    vals = [float(x) for x in parts[1:]]
                    raw_rows.append((row_name, vals))
                except ValueError:
                    pass
    names = [r[0] for r in raw_rows]
    matrix: dict[str, dict[str, float]] = {}
    for row_name, vals in raw_rows:
        row_dict: dict[str, float] = {}
        for col_idx, val in enumerate(vals):
            if col_idx < len(names):
                row_dict[names[col_idx]] = val
        matrix[row_name] = row_dict
    return names, matrix


def build_viewer_payload(h3_root: Path, exl3_dir: Path) -> dict[str, Any]:
    toml_dir = h3_root / "prompt_bank" / "control_vectors"
    labels_path = exl3_dir / "h3_control_vector_labels.json"
    report_path = exl3_dir / "h3_control_vector_report.md"

    labels_data: dict[str, Any] = {}
    if labels_path.is_file():
        labels_data = json.loads(labels_path.read_text(encoding="utf-8"))

    cos_order, cos_matrix = parse_cosine_matrix_from_report(report_path)

    toml_files = sorted(toml_dir.glob("*.toml")) if toml_dir.is_dir() else []
    toml_by_stem: dict[str, Path] = {p.stem: p for p in toml_files}
    all_keys = sorted(set(labels_data.keys()) | set(toml_by_stem.keys()))

    vectors: dict[str, Any] = {}
    for key in all_keys:
        entry = labels_data.get(key, {})
        toml_path = toml_by_stem.get(key)
        toml_header: dict[str, Any] = {}
        pair_records: list[dict[str, Any]] = []

        if toml_path and toml_path.is_file():
            raw_toml = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            toml_header = dict(raw_toml.get("vector", {}))
            frame = str(toml_header.get("frame", "scene"))
            raw_pairs = raw_toml.get("pair", [])
            for idx, pr in enumerate(raw_pairs):
                pair_records.append(analyze_toml_pair(idx, pr, frame))
        else:
            gt = entry.get("ground_truth_header", {})
            toml_header = {
                "name": gt.get("name", key),
                "quality": gt.get("quality", ""),
                "a_side": gt.get("a_side", ""),
                "b_side": gt.get("b_side", ""),
                "mode": "t2va",
                "frame": "bare" if key.endswith("_bare") else "scene",
                "exclude_shots": [],
            }

        n_pairs = len(pair_records)
        if n_pairs > 0:
            avg_st_diff = round(sum(p["stats"]["scene_text_diff_pct"] for p in pair_records) / n_pairs, 1)
            avg_ffd_diff = round(sum(p["stats"]["from_first_diff_diff_pct"] for p in pair_records) / n_pairs, 1)
            avg_pre_pct = round(sum(p["stats"]["prefix_pct_of_total"] for p in pair_records) / n_pairs, 1)
            avg_diff_pct = round(sum(p["stats"]["diff_pct_of_total"] for p in pair_records) / n_pairs, 1)
            avg_ksuf_pct = round(sum(p["stats"]["kept_suffix_pct_of_total"] for p in pair_records) / n_pairs, 1)
            avg_tsuf_pct = round(sum(p["stats"]["trimmed_soundscape_pct_of_total"] for p in pair_records) / n_pairs, 1)
            avg_dlg_pct = round(sum(p["stats"]["dialogue_pct_of_total"] for p in pair_records) / n_pairs, 1)
            avg_st_words = round(sum(p["stats"]["scene_text_active_words"] for p in pair_records) / n_pairs, 1)
            avg_ffd_words = round(sum(p["stats"]["from_first_diff_active_words"] for p in pair_records) / n_pairs, 1)
            soundscape_trimmed_pairs = sum(1 for p in pair_records if p["soundscape_identical"])
            sound_only_pairs = sum(1 for p in pair_records if p["edit_section"] == "overall_soundscape")
        else:
            avg_st_diff = avg_ffd_diff = avg_pre_pct = avg_diff_pct = avg_ksuf_pct = avg_tsuf_pct = avg_dlg_pct = 0.0
            avg_st_words = avg_ffd_words = 0.0
            soundscape_trimmed_pairs = sound_only_pairs = 0

        sr = entry.get("layer49_self_report", {})
        lex = sr.get("lexical_logit_shifts", {})
        pos_words = parse_word_shifts(lex.get("positive_boosted_words", []))
        neg_words = parse_word_shifts(lex.get("negative_boosted_words", []))

        blind = entry.get("blind_pair_label", {})
        synth = sr.get("synthesis", {})
        modality = blind.get("modality") or synth.get("dominant_modality") or "visual"

        vectors[key] = {
            "key": key,
            "toml_file": entry.get("toml_file", f"{key}.toml"),
            "toml_sha256": entry.get("toml_sha256", ""),
            "safetensors_files": entry.get("safetensors_files", []),
            "encoder_vector_source": entry.get("encoder_vector_source", "computed_exl3_layer49"),
            "layer49_raw_norm": float(entry.get("layer49_raw_norm", 0.0)),
            "layer49_median_row_norm_ex_row0": float(entry.get("layer49_median_row_norm_ex_row0", 0.0)),
            "elapsed_sec": entry.get("elapsed_sec", None),
            "modality": modality,
            "vector_header": {
                "name": toml_header.get("name", key),
                "quality": toml_header.get("quality", entry.get("ground_truth_header", {}).get("quality", "")),
                "a_side": toml_header.get("a_side", entry.get("ground_truth_header", {}).get("a_side", "")),
                "b_side": toml_header.get("b_side", entry.get("ground_truth_header", {}).get("b_side", "")),
                "mode": toml_header.get("mode", "t2va"),
                "frame": toml_header.get("frame", "bare" if key.endswith("_bare") else "scene"),
                "exclude_shots": toml_header.get("exclude_shots", []),
            },
            "pair_summary": {
                "num_pairs": n_pairs,
                "avg_scene_text_diff_pct": avg_st_diff,
                "avg_from_first_diff_diff_pct": avg_ffd_diff,
                "avg_scene_text_active_words": avg_st_words,
                "avg_from_first_diff_active_words": avg_ffd_words,
                "avg_prefix_pct": avg_pre_pct,
                "avg_diff_pct": avg_diff_pct,
                "avg_kept_suffix_pct": avg_ksuf_pct,
                "avg_trimmed_soundscape_pct": avg_tsuf_pct,
                "avg_dialogue_pct": avg_dlg_pct,
                "soundscape_trimmed_pairs": soundscape_trimmed_pairs,
                "sound_only_pairs": sound_only_pairs,
            },
            "pairs": pair_records,
            "blind_pair_label": blind,
            "layer49_self_report": {
                "median_row_norm": sr.get("median_row_norm", entry.get("layer49_median_row_norm_ex_row0", 0.0)),
                "sampling": sr.get("sampling", {}),
                "positive_boosted_words": pos_words,
                "negative_boosted_words": neg_words,
                "sweep_completions": sr.get("sweep_completions", {}),
                "synthesis": synth,
            },
        }

    ordered_keys = cos_order if len(cos_order) == len(vectors) else all_keys
    return {
        "schema_version": "1.0",
        "model_name": "Qwen3-VL-32B-Instruct (EXL3 5.37 bpw, 6 bpw layers.50..63, 8 bpw lm_head)",
        "tap_point": "model.language_model.layers.49 (hidden_states[-15], 5120-dim fp32)",
        "vector_order": ordered_keys,
        "vectors": vectors,
        "cosine_matrix": cos_matrix,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>H3 Control Vector Explorer — Inputs (.toml) ↔ Qwen3-VL layers.49 Readouts</title>
<style>
  :root {
    --bg-0: #090c12;
    --bg-1: #0f141d;
    --bg-2: #151c28;
    --bg-3: #1c2536;
    --bg-hover: #232e42;
    --border: #263247;
    --border-bright: #384967;
    --text-0: #f0f4fc;
    --text-1: #c3cee4;
    --text-2: #8896b3;
    --text-muted: #586580;
    --accent: #38bdf8;
    --accent-dim: rgba(56, 189, 248, 0.14);
    --neg: #38bdf8;
    --neg-bg: rgba(56, 189, 248, 0.14);
    --neg-border: rgba(56, 189, 248, 0.42);
    --pos: #f59e0b;
    --pos-bg: rgba(245, 158, 11, 0.15);
    --pos-border: rgba(245, 158, 11, 0.45);
    --emerald: #10b981;
    --emerald-bg: rgba(16, 185, 129, 0.14);
    --purple: #a855f7;
    --purple-bg: rgba(168, 85, 247, 0.14);
    --rose: #f43f5e;
    --rose-bg: rgba(244, 63, 94, 0.14);
    --font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    background-color: var(--bg-0);
    color: var(--text-0);
    font-family: var(--font-sans);
    font-size: 13.5px;
    line-height: 1.5;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Top Bar */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 18px;
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    gap: 14px;
    flex-shrink: 0;
    z-index: 10;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .brand-badge {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 11.5px;
    padding: 4px 8px;
    border-radius: 5px;
    letter-spacing: 0.04em;
  }

  .brand-title {
    font-weight: 700;
    font-size: 15px;
    color: var(--text-0);
  }

  .brand-sub {
    font-size: 11.5px;
    color: var(--text-2);
    font-family: var(--font-mono);
  }

  .nav-tabs {
    display: flex;
    gap: 6px;
    background: var(--bg-0);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }

  .nav-tab {
    background: transparent;
    color: var(--text-1);
    border: none;
    padding: 6px 13px;
    border-radius: 5px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .nav-tab:hover {
    color: var(--text-0);
    background: var(--bg-2);
  }

  .nav-tab.active {
    background: var(--bg-3);
    color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--neg-border);
  }

  .topbar-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-2);
    color: var(--text-1);
    border: 1px solid var(--border);
    padding: 6px 11px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }

  .btn:hover {
    background: var(--bg-hover);
    color: var(--text-0);
    border-color: var(--border-bright);
  }

  .btn.active, .btn-primary {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: var(--neg-border);
  }

  /* Main Workspace */
  .workspace {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* Left Sidebar */
  .sidebar {
    width: 310px;
    background: var(--bg-1);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }

  .sidebar-controls {
    padding: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .search-input {
    width: 100%;
    background: var(--bg-0);
    color: var(--text-0);
    border: 1px solid var(--border);
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 12.5px;
    outline: none;
  }

  .search-input:focus {
    border-color: var(--accent);
  }

  .filter-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .filter-pill {
    background: var(--bg-0);
    color: var(--text-2);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }

  .filter-pill:hover {
    color: var(--text-0);
  }

  .filter-pill.active {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: var(--neg-border);
  }

  .vector-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .vec-item {
    padding: 9px 11px;
    border-radius: 7px;
    border: 1px solid transparent;
    background: var(--bg-2);
    cursor: pointer;
    transition: all 0.12s ease;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .vec-item:hover {
    background: var(--bg-hover);
    border-color: var(--border);
  }

  .vec-item.active {
    background: var(--bg-3);
    border-color: var(--accent);
    box-shadow: inset 3px 0 0 var(--accent);
  }

  .vec-item-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .vec-name {
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 12.5px;
    color: var(--text-0);
  }

  .vec-badges {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .pill {
    font-size: 10px;
    font-weight: 700;
    font-family: var(--font-mono);
    padding: 1.5px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .pill-visual { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.3); }
  .pill-audiovisual { background: rgba(168, 85, 247, 0.16); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.32); }
  .pill-acoustic { background: rgba(16, 185, 129, 0.16); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.32); }
  .pill-typography { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.32); }
  .pill-bare { background: rgba(244, 63, 94, 0.18); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.35); }
  .pill-scene { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.25); }
  .pill-high { background: rgba(16, 185, 129, 0.15); color: #34d399; }
  .pill-medium { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }

  .vec-item-sub {
    font-size: 11px;
    color: var(--text-2);
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
  }

  /* Content Pane */
  .main-pane {
    flex: 1;
    overflow-y: auto;
    padding: 18px 22px 48px 22px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .tab-panel {
    display: none;
    flex-direction: column;
    gap: 18px;
  }

  .tab-panel.active {
    display: flex;
  }

  /* Section Cards */
  .card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
  }

  .card-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-0);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .card-subtitle {
    font-size: 12px;
    color: var(--text-2);
  }

  /* 3-Way Alignment Grid */
  .tri-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
  }

  @media (max-width: 1280px) {
    .tri-grid {
      grid-template-columns: 1fr;
    }
  }

  .stage-box {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 13px 15px;
    display: flex;
    flex-direction: column;
    gap: 9px;
  }

  .stage-box-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 7px;
  }

  .stage-label {
    font-size: 11px;
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    color: var(--text-2);
  }

  .stage-slug {
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 700;
    color: var(--accent);
  }

  .stage-quality {
    font-size: 12.5px;
    color: var(--text-0);
    font-weight: 500;
  }

  .pole-box {
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 12px;
    line-height: 1.45;
  }

  .pole-neg {
    background: var(--neg-bg);
    border-left: 3px solid var(--neg);
    color: #e0f2fe;
  }

  .pole-pos {
    background: var(--pos-bg);
    border-left: 3px solid var(--pos);
    color: #fef3c7;
  }

  .pole-tag {
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 11px;
    margin-right: 6px;
  }

  .meta-kv-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 11.5px;
    color: var(--text-2);
    margin-top: auto;
    padding-top: 6px;
  }

  .meta-kv {
    background: var(--bg-0);
    padding: 3px 7px;
    border-radius: 4px;
    border: 1px solid var(--border);
    font-family: var(--font-mono);
  }

  /* Word Budget Progress Bar */
  .budget-wrap {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .budget-bar {
    height: 12px;
    border-radius: 6px;
    background: var(--bg-0);
    overflow: hidden;
    display: flex;
    border: 1px solid var(--border);
  }

  .budget-seg {
    height: 100%;
    transition: width 0.25s ease, opacity 0.2s ease;
  }

  .seg-prefix { background: #475569; }
  .seg-diff { background: linear-gradient(90deg, #0284c7, #f59e0b); }
  .seg-kept-suffix { background: #64748b; }
  .seg-trimmed-sound { background: #334155; }
  .seg-dialogue { background: #7e22ce; }

  .budget-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 11.5px;
    color: var(--text-1);
    font-family: var(--font-mono);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 2px;
  }

  /* Pair List & Diff Formatting */
  .pairs-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .pair-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .pair-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-2);
    border-bottom: 1px dashed var(--border);
    padding-bottom: 6px;
  }

  .pair-body-inline {
    font-size: 13px;
    line-height: 1.65;
    color: var(--text-0);
    white-space: pre-wrap;
  }

  .pair-body-sbs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  @media (max-width: 1000px) {
    .pair-body-sbs {
      grid-template-columns: 1fr;
    }
  }

  .sbs-col {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 12.8px;
    line-height: 1.6;
    white-space: pre-wrap;
  }

  .sbs-col-a { border-top: 2px solid var(--neg); }
  .sbs-col-b { border-top: 2px solid var(--pos); }

  .diff-a-inline {
    background: var(--neg-bg);
    border: 1px solid var(--neg-border);
    color: #bae6fd;
    padding: 1px 5px;
    border-radius: 4px;
    font-weight: 600;
  }

  .diff-b-inline {
    background: var(--pos-bg);
    border: 1px solid var(--pos-border);
    color: #fde68a;
    padding: 1px 5px;
    border-radius: 4px;
    font-weight: 600;
  }

  .diff-arrow {
    color: var(--text-2);
    font-family: var(--font-mono);
    font-weight: 700;
    margin: 0 4px;
  }

  .tok-masked {
    opacity: 0.28;
    text-decoration: line-through;
    text-decoration-color: rgba(148, 163, 184, 0.5);
    filter: grayscale(80%);
    transition: opacity 0.2s ease;
  }

  .tok-dialogue {
    background: rgba(168, 85, 247, 0.14);
    border-bottom: 1px dashed #a855f7;
    color: #e9d5ff;
    padding: 0 3px;
    border-radius: 3px;
  }

  .tok-speaker {
    background: rgba(244, 63, 94, 0.15);
    border-bottom: 1px dashed #f43f5e;
    color: #fecdd3;
    padding: 0 3px;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 11.5px;
  }

  /* Diverging Log-Odds Bar Chart */
  .logodds-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }

  @media (max-width: 900px) {
    .logodds-grid {
      grid-template-columns: 1fr;
    }
  }

  .logodds-col {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .logodds-col-title {
    font-size: 12px;
    font-family: var(--font-mono);
    font-weight: 700;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
  }

  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-mono);
    font-size: 12.5px;
  }

  .bar-word {
    width: 125px;
    font-weight: 700;
    color: var(--text-0);
    flex-shrink: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .bar-track {
    flex: 1;
    height: 16px;
    background: var(--bg-0);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
  }

  .bar-fill-neg {
    height: 100%;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.35), #38bdf8);
    border-radius: 3px;
  }

  .bar-fill-pos {
    height: 100%;
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.35), #f59e0b);
    border-radius: 3px;
  }

  .bar-val {
    width: 56px;
    text-align: right;
    font-size: 11.5px;
    color: var(--text-1);
    flex-shrink: 0;
  }

  /* Coupled Rollouts Inspector */
  .anchor-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  @media (max-width: 1150px) {
    .anchor-grid {
      grid-template-columns: 1fr;
    }
  }

  .anchor-box {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 13px 15px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .anchor-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-0);
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 7px;
  }

  .alpha-cards-stack {
    display: flex;
    flex-direction: column;
    gap: 9px;
  }

  .alpha-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .alpha-card.alpha-neg { border-left: 3px solid var(--neg); }
  .alpha-card.alpha-zero { border-left: 3px solid #94a3b8; }
  .alpha-card.alpha-pos1 { border-left: 3px solid #fbbf24; }
  .alpha-card.alpha-pos2 { border-left: 3px solid #f97316; }

  .alpha-badge {
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    color: var(--text-1);
  }

  .rollout-text {
    font-size: 12.8px;
    line-height: 1.6;
    color: var(--text-0);
    white-space: pre-wrap;
  }

  .hl-changed-neg {
    background: rgba(56, 189, 248, 0.18);
    color: #e0f2fe;
    padding: 0 2px;
    border-radius: 3px;
  }

  .hl-changed-pos {
    background: rgba(245, 158, 11, 0.22);
    color: #fef3c7;
    padding: 0 2px;
    border-radius: 3px;
  }

  .field-tag {
    font-family: var(--font-mono);
    font-weight: 700;
    color: var(--accent);
    font-size: 11.5px;
  }

  /* Overview Comparison Table */
  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }

  table.comp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    background: var(--bg-1);
  }

  table.comp-table th, table.comp-table td {
    padding: 9px 11px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }

  table.comp-table th {
    background: var(--bg-2);
    color: var(--text-1);
    font-weight: 700;
    font-family: var(--font-mono);
    font-size: 11.5px;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }

  table.comp-table th:hover {
    color: var(--accent);
  }

  table.comp-table tr:hover td {
    background: var(--bg-hover);
  }

  /* Cosine Heatmap */
  .heatmap-wrap {
    overflow-x: auto;
  }

  table.heatmap-table {
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 11.5px;
  }

  table.heatmap-table th {
    padding: 6px 8px;
    color: var(--text-1);
    font-weight: 600;
    text-align: right;
    white-space: nowrap;
  }

  table.heatmap-table th.col-hdr {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    text-align: left;
    height: 135px;
    padding: 8px 5px;
  }

  table.heatmap-table td.hm-cell {
    width: 52px;
    height: 36px;
    text-align: center;
    border: 1px solid var(--bg-0);
    cursor: pointer;
    font-weight: 600;
    transition: transform 0.1s ease;
  }

  table.heatmap-table td.hm-cell:hover {
    outline: 2px solid #fff;
    z-index: 2;
    position: relative;
  }

  table.heatmap-table td.hm-cell.selected {
    outline: 2px solid var(--pos);
    z-index: 3;
    position: relative;
  }

  .shared-word-chip {
    display: inline-block;
    background: rgba(245, 158, 11, 0.24);
    border: 1px solid var(--pos);
    color: #fde68a;
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    margin: 2px;
  }

  .word-chip {
    display: inline-block;
    background: var(--bg-1);
    border: 1px solid var(--border);
    color: var(--text-1);
    font-family: var(--font-mono);
    font-size: 11.5px;
    padding: 2px 7px;
    border-radius: 4px;
    margin: 2px;
  }

  .status-toast {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--emerald);
  }
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <span class="brand-badge">H3 × EXL3</span>
    <div>
      <div class="brand-title">Control Vector Input / Output Explorer</div>
      <div class="brand-sub" id="modelSubtitle">Qwen3-VL-32B-Instruct • layers.49 residual tap [5120] → layers.50..63 + lm_head</div>
    </div>
  </div>

  <nav class="nav-tabs">
    <button class="nav-tab active" data-tab="deep-dive">1. Vector Deep-Dive (Input ↔ Output)</button>
    <button class="nav-tab" data-tab="overview-table">2. All-14 Overview Table</button>
    <button class="nav-tab" data-tab="cosine-matrix">3. 14×14 Cosine Matrix & Comparator</button>
  </nav>

  <div class="topbar-actions">
    <span class="status-toast" id="statusToast"></span>
    <label class="btn" title="Drop or select updated h3_control_vector_labels.json or .toml files">
      <span>📂 Load JSON / TOML</span>
      <input type="file" id="fileLoaderInput" multiple accept=".json,.toml,.md" style="display:none">
    </label>
    <button class="btn" id="refreshLiveBtn" title="Fetch fresh data from build_control_vector_viewer.py --serve">
      🔄 Refresh Live
    </button>
  </div>
</header>

<div class="workspace">
  <!-- Left Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-controls">
      <input type="text" id="vecSearchInput" class="search-input" placeholder="Filter 14 vectors by name, word, pole...">
      <div class="filter-pills" id="modalityFilterGroup">
        <button class="filter-pill active" data-filter="all">All (14)</button>
        <button class="filter-pill" data-filter="visual">Visual</button>
        <button class="filter-pill" data-filter="audiovisual">Audiovisual</button>
        <button class="filter-pill" data-filter="acoustic">Acoustic</button>
        <button class="filter-pill" data-filter="typography">Typography</button>
        <button class="filter-pill" data-filter="bare">Bare Frame</button>
      </div>
    </div>
    <div class="vector-list" id="vectorListContainer"></div>
  </aside>

  <!-- Main Content -->
  <main class="main-pane">
    <!-- TAB 1: DEEP DIVE -->
    <div class="tab-panel active" id="tab-deep-dive">
      <!-- 3-Way Alignment Card -->
      <section class="card">
        <div class="card-header">
          <div>
            <div class="card-title" id="ddHeaderTitle">Vector Deep-Dive</div>
            <div class="card-subtitle" id="ddHeaderMeta"></div>
          </div>
          <div id="ddHeaderBadges" style="display:flex; gap:6px; align-items:center;"></div>
        </div>
        <div class="tri-grid" id="ddTriGrid"></div>
      </section>

      <!-- Direct layers.49 -> lm_head Lexical Log-Odds Chart & Coupled Rollouts -->
      <section class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Direct <code>layers.49 → lm_head</code> Lexical Log-Odds Shift (<code>Δlog P</code> at α = ±0.18)</div>
            <div class="card-subtitle">Zero-sampling first-token log-probability boost over space-prefixed English content words across 6 H3 probe prefixes</div>
          </div>
        </div>
        <div class="logodds-grid" id="ddLogOddsGrid"></div>
      </section>

      <!-- Coupled Activation Steering Rollouts -->
      <section class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Coupled <code>layers.49</code> Activation-Steered Rollouts (<code>ComboSampler temp=0.7</code>, Coupled Gumbel Seed 101)</div>
            <div class="card-subtitle">Assistant-span steering (<code>x[:, steer_start:, :] += α · median_row_norm · v_encoder</code>) on two fixed neutral H3 scene anchors</div>
          </div>
          <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
            <div id="alphaFilterGroup" style="display:flex; gap:4px;">
              <button class="btn active" data-alpha="all">All 4 α Side-by-Side</button>
              <button class="btn" data-alpha="-0.15">α = -0.15</button>
              <button class="btn" data-alpha="+0.00">α = +0.00</button>
              <button class="btn" data-alpha="+0.15">α = +0.15</button>
              <button class="btn" data-alpha="+0.22">α = +0.22</button>
            </div>
            <button class="btn active" id="toggleRolloutDiffBtn">✨ Highlight Changes vs α = +0.00</button>
          </div>
        </div>
        <div class="anchor-grid" id="ddAnchorGrid"></div>
      </section>

      <!-- Input .toml Contrastive Pair Explorer -->
      <section class="card">
        <div class="card-header">
          <div>
            <div class="card-title" id="ddPairsTitle">Input <code>.toml</code> Contrastive Prompt Pairs</div>
            <div class="card-subtitle">Inspect how pooling masks (<code>scene_text</code> vs <code>from_first_diff</code>) weight or trim shared context, dialogue, and trailing soundscapes</div>
          </div>
          <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
            <div id="poolingModeGroup" style="display:flex; gap:4px;">
              <button class="btn active" data-pooling="from_first_diff" title="Masks out shared prefix before first_diff, <d>...</d>, (S1), and unchanged overall_soundscape: suffix">🎯 Pooling: from_first_diff</button>
              <button class="btn" data-pooling="scene_text" title="Pools all tokens except row 0, <d>...</d> dialogue, and (S1) tags">🌐 Pooling: scene_text</button>
              <button class="btn" data-pooling="raw" title="Show raw unmasked prompt text">📄 Unmasked Raw</button>
            </div>
            <div id="pairViewModeGroup" style="display:flex; gap:4px;">
              <button class="btn active" data-view="inline">Inline [A → B] Diff</button>
              <button class="btn" data-view="sbs">Side-by-Side (A vs B)</button>
            </div>
          </div>
        </div>
        <div class="budget-wrap" id="ddBudgetWrap"></div>
        <div class="pairs-grid" id="ddPairsContainer"></div>
      </section>
    </div>

    <!-- TAB 2: OVERVIEW TABLE -->
    <div class="tab-panel" id="tab-overview-table">
      <section class="card">
        <div class="card-header">
          <div>
            <div class="card-title">All-14 Control Vectors — Input Pooling Budget vs. <code>layers.49</code> Readout Summary</div>
            <div class="card-subtitle">Click any column header to sort; click any row to jump to its Vector Deep-Dive</div>
          </div>
        </div>
        <div class="table-wrap">
          <table class="comp-table" id="overviewCompTable">
            <thead>
              <tr>
                <th data-sort="key">Vector (.toml)</th>
                <th data-sort="modality">Modality</th>
                <th data-sort="frame">Frame</th>
                <th data-sort="raw_norm">L49 Raw Norm</th>
                <th data-sort="st_diff">scene_text Diff %</th>
                <th data-sort="ffd_diff">from_first_diff Diff %</th>
                <th data-sort="blind_slug">Blind Pair-Diff Slug</th>
                <th data-sort="self_slug">L49 Self-Report Slug</th>
                <th>Top +v_encoder Words (Δlog P)</th>
                <th>Top -v_encoder Words (Δlog P)</th>
              </tr>
            </thead>
            <tbody id="overviewTableBody"></tbody>
          </table>
        </div>
      </section>
    </div>

    <!-- TAB 3: COSINE MATRIX & PAIR COMPARATOR -->
    <div class="tab-panel" id="tab-cosine-matrix">
      <section class="card">
        <div class="card-header">
          <div>
            <div class="card-title">14 × 14 Pairwise Cosine Similarity Matrix at <code>layers.49</code> (<code>vector.encoder [5120]</code>)</div>
            <div class="card-subtitle">Click any cell <code>(row, col)</code> to compare the two vectors' <code>.toml</code> poles, shared boosted words, and differing clauses below</div>
          </div>
        </div>
        <div class="heatmap-wrap" id="cosineHeatmapContainer"></div>
      </section>

      <section class="card" id="pairComparatorDrawer">
        <div class="card-header">
          <div>
            <div class="card-title" id="compDrawerTitle">Vector-vs-Vector Comparison Drawer</div>
            <div class="card-subtitle" id="compDrawerSub">Select any cell in the 14×14 matrix above to inspect why two vectors align or diverge at layers.49</div>
          </div>
        </div>
        <div id="compDrawerBody"></div>
      </section>
    </div>
  </main>
</div>

<script>
window.__INITIAL_CV_DATA__ = __EMBEDDED_JSON_PAYLOAD__;
</script>
<script>
(function() {
  "use strict";

  let appData = window.__INITIAL_CV_DATA__ || { vector_order: [], vectors: {}, cosine_matrix: {} };
  let state = {
    selectedVector: (appData.vector_order && appData.vector_order[0]) || Object.keys(appData.vectors)[0] || "chaos_panic",
    activeTab: "deep-dive",
    searchQuery: "",
    modalityFilter: "all",
    poolingMode: "from_first_diff",
    pairViewMode: "inline",
    alphaFilter: "all",
    highlightRolloutDiff: true,
    sortCol: "key",
    sortAsc: true,
    compRow: "chaos_panic",
    compCol: "vocal_strain"
  };

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showToast(msg) {
    const el = document.getElementById("statusToast");
    if (!el) return;
    el.textContent = msg;
    setTimeout(() => {
      if (el.textContent === msg) el.textContent = "";
    }, 4000);
  }

  function renderSegmentedSpan(segments, isMaskedByPooling) {
    if (!segments || !segments.length) return "";
    return segments.map(seg => {
      const txt = escapeHtml(seg.text);
      if (seg.kind === "dialogue") {
        const cls = state.poolingMode === "raw" ? "tok-dialogue" : "tok-dialogue tok-masked";
        return `<span class="${cls}" title="<d>...</d> dialogue span (excluded by scene_text and from_first_diff)">${txt}</span>`;
      }
      if (seg.kind === "speaker") {
        const cls = state.poolingMode === "raw" ? "tok-speaker" : "tok-speaker tok-masked";
        return `<span class="${cls}" title="Speaker tag (excluded by scene_text and from_first_diff)">${txt}</span>`;
      }
      if (isMaskedByPooling) {
        return `<span class="tok-masked">${txt}</span>`;
      }
      return txt;
    }).join("");
  }

  function highlightRolloutAgainstBaseline(text, baseText, isNegative) {
    if (!text) return "";
    if (!state.highlightRolloutDiff || !baseText || text === baseText) {
      return formatRolloutSections(escapeHtml(text));
    }
    const baseWords = new Set(
      baseText.toLowerCase().replace(/[^\w\s-]/g, " ").split(/\s+/).filter(w => w.length >= 4)
    );
    const tokens = text.split(/(\s+)/);
    const cls = isNegative ? "hl-changed-neg" : "hl-changed-pos";
    const html = tokens.map(tok => {
      if (/^\s+$/.test(tok)) return tok;
      const clean = tok.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, "");
      if (
        clean.length >= 4 &&
        !baseWords.has(clean) &&
        clean !== "integrated_multimodal_description" &&
        clean !== "overall_soundscape" &&
        clean !== "shot"
      ) {
        return `<span class="${cls}">${escapeHtml(tok)}</span>`;
      }
      return escapeHtml(tok);
    }).join("");
    return formatRolloutSections(html);
  }

  function formatRolloutSections(escapedHtml) {
    return escapedHtml
      .replace(/integrated_multimodal_description:/g, '<span class="field-tag">integrated_multimodal_description:</span>')
      .replace(/overall_soundscape:/g, '<span class="field-tag">overall_soundscape:</span>');
  }

  function renderSidebar() {
    const container = document.getElementById("vectorListContainer");
    if (!container) return;
    const keys = appData.vector_order || Object.keys(appData.vectors);
    const q = state.searchQuery.trim().toLowerCase();

    const filtered = keys.filter(k => {
      const v = appData.vectors[k];
      if (!v) return false;
      const frame = (v.vector_header && v.vector_header.frame) || "scene";
      const mod = v.modality || "visual";
      if (state.modalityFilter === "bare" && frame !== "bare") return false;
      if (state.modalityFilter !== "all" && state.modalityFilter !== "bare" && mod !== state.modalityFilter) return false;
      if (!q) return true;
      const hay = [
        k,
        v.vector_header?.quality,
        v.vector_header?.a_side,
        v.vector_header?.b_side,
        v.blind_pair_label?.inferred_slug,
        v.layer49_self_report?.synthesis?.self_report_slug,
        ...(v.layer49_self_report?.positive_boosted_words || []).map(x => x.word),
        ...(v.layer49_self_report?.negative_boosted_words || []).map(x => x.word)
      ].join(" ").toLowerCase();
      return hay.includes(q);
    });

    container.innerHTML = filtered.map(k => {
      const v = appData.vectors[k];
      const activeCls = k === state.selectedVector ? "vec-item active" : "vec-item";
      const mod = v.modality || "visual";
      const frame = v.vector_header?.frame || "scene";
      const clarity = v.layer49_self_report?.synthesis?.readout_clarity || "high";
      const stDiff = v.pair_summary?.avg_scene_text_diff_pct ?? 0;
      const ffdDiff = v.pair_summary?.avg_from_first_diff_diff_pct ?? 0;
      return `
        <div class="${activeCls}" data-vec="${escapeHtml(k)}">
          <div class="vec-item-top">
            <span class="vec-name">${escapeHtml(k)}</span>
            <div class="vec-badges">
              <span class="pill pill-${escapeHtml(mod)}">${escapeHtml(mod)}</span>
              ${frame === "bare" ? '<span class="pill pill-bare">bare</span>' : ''}
            </div>
          </div>
          <div class="vec-item-sub">
            <span>‖v‖=${v.layer49_raw_norm.toFixed(1)}</span>
            <span>diff: ${stDiff.toFixed(0)}%→${ffdDiff.toFixed(0)}%</span>
            <span class="pill pill-${escapeHtml(clarity)}">${escapeHtml(clarity)}</span>
          </div>
        </div>
      `;
    }).join("");

    container.querySelectorAll(".vec-item").forEach(el => {
      el.addEventListener("click", () => {
        state.selectedVector = el.getAttribute("data-vec");
        renderSidebar();
        renderDeepDive();
      });
    });
  }

  function renderDeepDive() {
    const v = appData.vectors[state.selectedVector];
    if (!v) return;

    const vh = v.vector_header || {};
    const blind = v.blind_pair_label || {};
    const sr = v.layer49_self_report || {};
    const synth = sr.synthesis || {};
    const ps = v.pair_summary || {};

    document.getElementById("ddHeaderTitle").innerHTML = `
      <span style="font-family:var(--font-mono); color:var(--accent);">${escapeHtml(v.key)}</span>
      <span style="color:var(--text-2); font-weight:400;">— ${escapeHtml(vh.quality || "")}</span>
    `;
    document.getElementById("ddHeaderMeta").innerHTML = `
      <code>${escapeHtml(v.toml_file)}</code> (sha256: <code>${escapeHtml(v.toml_sha256 || "n/a")}</code>) •
      <code>L49 raw_norm = ${v.layer49_raw_norm.toFixed(2)}</code> •
      <code>median_row_norm = ${v.layer49_median_row_norm_ex_row0.toFixed(1)}</code> •
      <code>${ps.num_pairs || 0} pairs</code>
    `;

    const clarity = synth.readout_clarity || "high";
    document.getElementById("ddHeaderBadges").innerHTML = `
      <span class="pill pill-${escapeHtml(v.modality)}">${escapeHtml(v.modality)}</span>
      <span class="pill pill-${escapeHtml(vh.frame || "scene")}">frame: ${escapeHtml(vh.frame || "scene")}</span>
      <span class="pill pill-scene">mode: ${escapeHtml(vh.mode || "t2va")}</span>
      <span class="pill pill-${escapeHtml(clarity)}">clarity: ${escapeHtml(clarity)}</span>
    `;

    // 3-Way Alignment Grid
    document.getElementById("ddTriGrid").innerHTML = `
      <div class="stage-box">
        <div class="stage-box-header">
          <span class="stage-label">1. Input .toml Target ([vector])</span>
          <span class="stage-slug">${escapeHtml(vh.name || v.key)}</span>
        </div>
        <div class="stage-quality">${escapeHtml(vh.quality || "")}</div>
        <div class="pole-box pole-neg"><span class="pole-tag">a_side (-v):</span>${escapeHtml(vh.a_side || "")}</div>
        <div class="pole-box pole-pos"><span class="pole-tag">b_side (+v):</span>${escapeHtml(vh.b_side || "")}</div>
        <div class="meta-kv-row">
          <span class="meta-kv">pairs: ${ps.num_pairs || 0}</span>
          <span class="meta-kv">scene_text diff: ${ps.avg_scene_text_diff_pct}%</span>
          <span class="meta-kv">from_first_diff: ${ps.avg_from_first_diff_diff_pct}%</span>
        </div>
      </div>

      <div class="stage-box">
        <div class="stage-box-header">
          <span class="stage-label">2. Blind Pair-Diff Judge (Text Only)</span>
          <span class="stage-slug">${escapeHtml(blind.inferred_slug || "")}</span>
        </div>
        <div class="stage-quality">${escapeHtml(blind.inferred_quality || "")}</div>
        <div class="pole-box pole-neg"><span class="pole-tag">a_pole (-v):</span>${escapeHtml(blind.a_pole || "")}</div>
        <div class="pole-box pole-pos"><span class="pole-tag">b_pole (+v):</span>${escapeHtml(blind.b_pole || "")}</div>
        <div class="meta-kv-row">
          <span class="meta-kv" title="Visual signature">👁️ ${escapeHtml(blind.visual_signature || "n/a")}</span>
          <span class="meta-kv" title="Acoustic signature">🔊 ${escapeHtml(blind.acoustic_signature || "n/a")}</span>
          <span class="meta-kv">notes: ${escapeHtml(blind.confounds_or_notes || "clean")}</span>
        </div>
      </div>

      <div class="stage-box">
        <div class="stage-box-header">
          <span class="stage-label">3. layers.49 Steered Self-Report</span>
          <span class="stage-slug">${escapeHtml(synth.self_report_slug || "")}</span>
        </div>
        <div class="stage-quality">${escapeHtml(synth.self_report_quality || "")}</div>
        <div class="pole-box pole-neg"><span class="pole-tag">-v_encoder:</span>${escapeHtml(synth.negative_pole || "")}</div>
        <div class="pole-box pole-pos"><span class="pole-tag">+v_encoder:</span>${escapeHtml(synth.positive_pole || "")}</div>
        <div class="meta-kv-row">
          <span class="meta-kv">${escapeHtml(synth.clarity_reason || "")}</span>
        </div>
      </div>
    `;

    // Lexical Log-Odds Chart
    const posWords = sr.positive_boosted_words || [];
    const negWords = sr.negative_boosted_words || [];
    const maxShift = Math.max(
      5.0,
      ...posWords.map(w => w.delta_log_prob || 0),
      ...negWords.map(w => w.delta_log_prob || 0)
    );

    const renderBarCol = (words, isPos) => {
      const fillCls = isPos ? "bar-fill-pos" : "bar-fill-neg";
      const headerColor = isPos ? "var(--pos)" : "var(--neg)";
      const title = isPos
        ? `Top +v_encoder Boosted Words (b_side / +α)`
        : `Top -v_encoder Boosted Words (a_side / -α)`;
      return `
        <div class="logodds-col">
          <div class="logodds-col-title" style="color:${headerColor}">
            <span>${title}</span>
            <span>Δlog P</span>
          </div>
          ${words.map(item => {
            const pct = Math.min(100, Math.max(4, (item.delta_log_prob / maxShift) * 100));
            return `
              <div class="bar-row">
                <span class="bar-word">${escapeHtml(item.word)}</span>
                <div class="bar-track">
                  <div class="${fillCls}" style="width:${pct.toFixed(1)}%"></div>
                </div>
                <span class="bar-val">+${item.delta_log_prob.toFixed(2)}</span>
              </div>
            `;
          }).join("")}
        </div>
      `;
    };

    document.getElementById("ddLogOddsGrid").innerHTML =
      renderBarCol(negWords, false) + renderBarCol(posWords, true);

    // Coupled Rollouts Inspector
    const sweeps = sr.sweep_completions || {};
    const anchors = [
      {
        key: "anchor_human_action_room",
        title: "Anchor A — Person + Kitchen Counter + Pitcher/Mug + Spoken Greeting",
        sub: "Probes human anatomy/material, hand coordination, vocal timbre, and room acoustics"
      },
      {
        key: "anchor_surface_street_camera",
        title: "Anchor B — Storefront Canvas Awning + Wooden Crate + White Van + Street",
        sub: "Probes film stock/color/grain, lens optics, surface lettering/signage, and camera motion"
      }
    ];
    const alphaOrder = [
      { key: "-0.15", label: "α = -0.15 (Toward a_side / -v_encoder)", cls: "alpha-neg", isNeg: true },
      { key: "+0.00", label: "α = +0.00 (Unsteered Baseline Reference)", cls: "alpha-zero", isNeg: false },
      { key: "+0.15", label: "α = +0.15 (Moderate +v_encoder Steering)", cls: "alpha-pos1", isNeg: false },
      { key: "+0.22", label: "α = +0.22 (Strong +v_encoder Steering)", cls: "alpha-pos2", isNeg: false }
    ];

    document.getElementById("ddAnchorGrid").innerHTML = anchors.map(anc => {
      const compMap = sweeps[anc.key] || {};
      const baseText = compMap["+0.00"] || "";
      const visibleAlphas = state.alphaFilter === "all"
        ? alphaOrder
        : alphaOrder.filter(a => a.key === state.alphaFilter);

      return `
        <div class="anchor-box">
          <div class="anchor-title">
            <div>
              <div>${escapeHtml(anc.title)}</div>
              <div style="font-size:11px; color:var(--text-2); font-weight:400;">${escapeHtml(anc.sub)}</div>
            </div>
          </div>
          <div class="alpha-cards-stack">
            ${visibleAlphas.map(a => {
              const txt = compMap[a.key] || "";
              const scale = (parseFloat(a.key) * v.layer49_median_row_norm_ex_row0).toFixed(1);
              const rendered = a.key === "+0.00"
                ? formatRolloutSections(escapeHtml(txt))
                : highlightRolloutAgainstBaseline(txt, baseText, a.isNeg);
              return `
                <div class="alpha-card ${a.cls}">
                  <div class="alpha-badge">
                    <span>${escapeHtml(a.label)}</span>
                    <span>‖Δh₄₉‖ = ${scale}</span>
                  </div>
                  <div class="rollout-text">${rendered}</div>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    }).join("");

    // Input .toml Pair Explorer & Pooling Budget
    renderPairsSection(v);
  }

  function renderPairsSection(v) {
    const ps = v.pair_summary || {};
    const vh = v.vector_header || {};
    const isBare = vh.frame === "bare";
    const pairs = v.pairs || [];

    document.getElementById("ddPairsTitle").innerHTML = `
      Input <code>.toml</code> Contrastive Prompt Pairs (${pairs.length} pairs in <code>${escapeHtml(v.toml_file)}</code>)
    `;

    const preDim = (state.poolingMode === "from_first_diff" && !isBare) ? 0.25 : 1.0;
    const tsufDim = (state.poolingMode === "from_first_diff" && !isBare) ? 0.25 : 1.0;
    const dlgDim = state.poolingMode === "raw" ? 1.0 : 0.25;

    document.getElementById("ddBudgetWrap").innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div style="font-size:12px; font-weight:600; color:var(--text-0);">
          Average Prompt Word Budget & Active Pooling Concentration:
          <span style="font-family:var(--font-mono); color:var(--accent); margin-left:6px;">
            scene_text: ${ps.avg_scene_text_diff_pct}% diff words (${ps.avg_scene_text_active_words}w pooled)
            → from_first_diff: ${ps.avg_from_first_diff_diff_pct}% diff words (${ps.avg_from_first_diff_active_words}w pooled)
          </span>
        </div>
        <div style="font-size:11.5px; font-family:var(--font-mono); color:var(--text-2);">
          soundscape trimmed in ${ps.soundscape_trimmed_pairs}/${pairs.length} pairs • soundscape-only edit in ${ps.sound_only_pairs}/${pairs.length} pairs
        </div>
      </div>
      <div class="budget-bar">
        <div class="budget-seg seg-prefix" style="width:${ps.avg_prefix_pct}%; opacity:${preDim}" title="Shared Prefix: ${ps.avg_prefix_pct}%"></div>
        <div class="budget-seg seg-diff" style="width:${ps.avg_diff_pct}%" title="Differing Clause (A vs B): ${ps.avg_diff_pct}%"></div>
        <div class="budget-seg seg-kept-suffix" style="width:${ps.avg_kept_suffix_pct}%" title="Kept In-Section Suffix: ${ps.avg_kept_suffix_pct}%"></div>
        <div class="budget-seg seg-trimmed-sound" style="width:${ps.avg_trimmed_soundscape_pct}%; opacity:${tsufDim}" title="Unchanged Trailing overall_soundscape: ${ps.avg_trimmed_soundscape_pct}%"></div>
        <div class="budget-seg seg-dialogue" style="width:${ps.avg_dialogue_pct}%; opacity:${dlgDim}" title="Excluded <d>...</d> & (S1): ${ps.avg_dialogue_pct}%"></div>
      </div>
      <div class="budget-legend">
        <span class="legend-item"><span class="legend-dot seg-prefix"></span>Shared Prefix (${ps.avg_prefix_pct}%)${state.poolingMode === "from_first_diff" && !isBare ? " [masked]" : ""}</span>
        <span class="legend-item"><span class="legend-dot seg-diff"></span>Differing Clause A↔B (${ps.avg_diff_pct}%) [active]</span>
        <span class="legend-item"><span class="legend-dot seg-kept-suffix"></span>Kept Section Suffix (${ps.avg_kept_suffix_pct}%) [active]</span>
        <span class="legend-item"><span class="legend-dot seg-trimmed-sound"></span>Unchanged Soundscape Tail (${ps.avg_trimmed_soundscape_pct}%)${state.poolingMode === "from_first_diff" && !isBare ? " [trimmed]" : ""}</span>
        <span class="legend-item"><span class="legend-dot seg-dialogue"></span>Dialogue <code>&lt;d&gt;</code> &amp; <code>(S1)</code> (${ps.avg_dialogue_pct}%)${state.poolingMode !== "raw" ? " [masked]" : ""}</span>
      </div>
    `;

    const maskPrefix = state.poolingMode === "from_first_diff" && !isBare;
    const maskTrimmedSuffix = state.poolingMode === "from_first_diff" && !isBare;

    document.getElementById("ddPairsContainer").innerHTML = pairs.map(pr => {
      const st = pr.stats || {};
      const segs = pr.segments || {};
      const preHtml = renderSegmentedSpan(segs.shared_prefix, maskPrefix);
      const aSpanHtml = renderSegmentedSpan(segs.a_span, false);
      const bSpanHtml = renderSegmentedSpan(segs.b_span, false);
      const ksufHtml = renderSegmentedSpan(segs.kept_suffix, false);
      const tsufHtml = renderSegmentedSpan(segs.trimmed_soundscape_suffix, maskTrimmedSuffix);

      const headerHtml = `
        <div class="pair-header">
          <span><strong>Pair #${pr.index + 1}</strong> • edit in <code>${escapeHtml(pr.edit_section)}</code>${pr.soundscape_identical ? " • unchanged soundscape trimmed under from_first_diff" : ""}</span>
          <span>
            prefix: ${st.prefix_active_words}w |
            diff A/B: ${st.diff_words_a}w / ${st.diff_words_b}w |
            kept suffix: ${st.kept_suffix_active_words}w |
            trimmed soundscape: ${st.trimmed_soundscape_active_words}w |
            <strong>scene_text diff: ${st.scene_text_diff_pct}% → from_first_diff: ${st.from_first_diff_diff_pct}%</strong>
          </span>
        </div>
      `;

      if (state.pairViewMode === "inline") {
        return `
          <div class="pair-card">
            ${headerHtml}
            <div class="pair-body-inline">${preHtml}<span class="diff-a-inline" title="Prompt A (-v)">[A: ${aSpanHtml}]</span><span class="diff-arrow">→</span><span class="diff-b-inline" title="Prompt B (+v)">[B: ${bSpanHtml}]</span>${ksufHtml}${tsufHtml}</div>
          </div>
        `;
      } else {
        return `
          <div class="pair-card">
            ${headerHtml}
            <div class="pair-body-sbs">
              <div class="sbs-col sbs-col-a"><div style="font-family:var(--font-mono); font-size:11px; color:var(--neg); margin-bottom:4px; font-weight:700;">PROMPT A (-v_encoder / a_side)</div>${preHtml}<span class="diff-a-inline">${aSpanHtml}</span>${ksufHtml}${tsufHtml}</div>
              <div class="sbs-col sbs-col-b"><div style="font-family:var(--font-mono); font-size:11px; color:var(--pos); margin-bottom:4px; font-weight:700;">PROMPT B (+v_encoder / b_side)</div>${preHtml}<span class="diff-b-inline">${bSpanHtml}</span>${ksufHtml}${tsufHtml}</div>
            </div>
          </div>
        `;
      }
    }).join("");
  }

  function renderOverviewTable() {
    const tbody = document.getElementById("overviewTableBody");
    if (!tbody) return;
    const keys = [...(appData.vector_order || Object.keys(appData.vectors))];

    keys.sort((a, b) => {
      const va = appData.vectors[a];
      const vb = appData.vectors[b];
      let cmp = 0;
      switch (state.sortCol) {
        case "raw_norm":
          cmp = va.layer49_raw_norm - vb.layer49_raw_norm;
          break;
        case "st_diff":
          cmp = (va.pair_summary?.avg_scene_text_diff_pct || 0) - (vb.pair_summary?.avg_scene_text_diff_pct || 0);
          break;
        case "ffd_diff":
          cmp = (va.pair_summary?.avg_from_first_diff_diff_pct || 0) - (vb.pair_summary?.avg_from_first_diff_diff_pct || 0);
          break;
        case "modality":
          cmp = (va.modality || "").localeCompare(vb.modality || "");
          break;
        case "frame":
          cmp = (va.vector_header?.frame || "").localeCompare(vb.vector_header?.frame || "");
          break;
        case "blind_slug":
          cmp = (va.blind_pair_label?.inferred_slug || "").localeCompare(vb.blind_pair_label?.inferred_slug || "");
          break;
        case "self_slug":
          cmp = (va.layer49_self_report?.synthesis?.self_report_slug || "").localeCompare(vb.layer49_self_report?.synthesis?.self_report_slug || "");
          break;
        default:
          cmp = a.localeCompare(b);
      }
      return state.sortAsc ? cmp : -cmp;
    });

    tbody.innerHTML = keys.map(k => {
      const v = appData.vectors[k];
      const ps = v.pair_summary || {};
      const posTop = (v.layer49_self_report?.positive_boosted_words || []).slice(0, 6).map(w => w.raw).join(", ");
      const negTop = (v.layer49_self_report?.negative_boosted_words || []).slice(0, 6).map(w => w.raw).join(", ");
      const frame = v.vector_header?.frame || "scene";
      return `
        <tr data-vec="${escapeHtml(k)}" style="cursor:pointer;">
          <td style="font-family:var(--font-mono); font-weight:700; color:var(--accent);">${escapeHtml(k)}</td>
          <td><span class="pill pill-${escapeHtml(v.modality)}">${escapeHtml(v.modality)}</span></td>
          <td><span class="pill pill-${escapeHtml(frame)}">${escapeHtml(frame)}</span></td>
          <td style="font-family:var(--font-mono); font-weight:700;">${v.layer49_raw_norm.toFixed(2)}</td>
          <td style="font-family:var(--font-mono);">${(ps.avg_scene_text_diff_pct || 0).toFixed(1)}%</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:var(--emerald);">${(ps.avg_from_first_diff_diff_pct || 0).toFixed(1)}%</td>
          <td style="font-family:var(--font-mono);">${escapeHtml(v.blind_pair_label?.inferred_slug || "")}</td>
          <td style="font-family:var(--font-mono);">${escapeHtml(v.layer49_self_report?.synthesis?.self_report_slug || "")}</td>
          <td style="font-family:var(--font-mono); font-size:11.5px; color:#fde68a;">${escapeHtml(posTop)}</td>
          <td style="font-family:var(--font-mono); font-size:11.5px; color:#bae6fd;">${escapeHtml(negTop)}</td>
        </tr>
      `;
    }).join("");

    tbody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        state.selectedVector = tr.getAttribute("data-vec");
        switchTab("deep-dive");
        renderSidebar();
        renderDeepDive();
      });
    });
  }

  function cosineColor(val, isDiag) {
    if (isDiag) return "background: rgba(56, 189, 248, 0.25); color: #e0f2fe;";
    if (val >= 0) {
      const alpha = Math.min(0.85, Math.pow(val, 1.15) * 0.9);
      const textCol = val > 0.45 ? "#fff" : "#cbd5e1";
      return `background: rgba(245, 158, 11, ${alpha.toFixed(3)}); color: ${textCol};`;
    } else {
      const alpha = Math.min(0.85, Math.abs(val) * 1.4);
      return `background: rgba(56, 189, 248, ${alpha.toFixed(3)}); color: #bae6fd;`;
    }
  }

  function renderCosineMatrix() {
    const container = document.getElementById("cosineHeatmapContainer");
    if (!container) return;
    const keys = appData.vector_order || Object.keys(appData.vectors);
    const mat = appData.cosine_matrix || {};

    let html = `<table class="heatmap-table"><thead><tr><th>Vector</th>`;
    keys.forEach(col => {
      html += `<th class="col-hdr">${escapeHtml(col)}</th>`;
    });
    html += `</tr></thead><tbody>`;

    keys.forEach(row => {
      html += `<tr><th>${escapeHtml(row)}</th>`;
      keys.forEach(col => {
        const val = (mat[row] && typeof mat[row][col] === "number") ? mat[row][col] : (row === col ? 1.0 : 0.0);
        const isDiag = row === col;
        const selCls = (row === state.compRow && col === state.compCol) ? "hm-cell selected" : "hm-cell";
        const sign = val >= 0 ? "+" : "";
        html += `<td class="${selCls}" data-row="${escapeHtml(row)}" data-col="${escapeHtml(col)}" style="${cosineColor(val, isDiag)}" title="${escapeHtml(row)} × ${escapeHtml(col)}: ${sign}${val.toFixed(2)}">${sign}${val.toFixed(2)}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;

    container.querySelectorAll(".hm-cell").forEach(cell => {
      cell.addEventListener("click", () => {
        state.compRow = cell.getAttribute("data-row");
        state.compCol = cell.getAttribute("data-col");
        renderCosineMatrix();
        renderPairComparator();
      });
    });

    renderPairComparator();
  }

  function renderPairComparator() {
    const v1 = appData.vectors[state.compRow];
    const v2 = appData.vectors[state.compCol];
    if (!v1 || !v2) return;

    const mat = appData.cosine_matrix || {};
    const cosVal = (mat[state.compRow] && typeof mat[state.compRow][state.compCol] === "number")
      ? mat[state.compRow][state.compCol]
      : (state.compRow === state.compCol ? 1.0 : 0.0);
    const sign = cosVal >= 0 ? "+" : "";

    document.getElementById("compDrawerTitle").innerHTML = `
      Comparing <code style="color:var(--accent)">${escapeHtml(v1.key)}</code> ×
      <code style="color:var(--pos)">${escapeHtml(v2.key)}</code> —
      <code>cos(layers.49) = ${sign}${cosVal.toFixed(2)}</code>
    `;

    const pos1 = v1.layer49_self_report?.positive_boosted_words || [];
    const pos2 = v2.layer49_self_report?.positive_boosted_words || [];
    const neg1 = v1.layer49_self_report?.negative_boosted_words || [];
    const neg2 = v2.layer49_self_report?.negative_boosted_words || [];

    const setPos2 = new Set(pos2.map(w => w.word));
    const setPos1 = new Set(pos1.map(w => w.word));
    const setNeg2 = new Set(neg2.map(w => w.word));
    const setNeg1 = new Set(neg1.map(w => w.word));

    const sharedPos = pos1.filter(w => setPos2.has(w.word)).map(w => w.word);
    const sharedNeg = neg1.filter(w => setNeg2.has(w.word)).map(w => w.word);

    const renderVecCol = (v, otherPosSet, otherNegSet) => {
      const vh = v.vector_header || {};
      const synth = v.layer49_self_report?.synthesis || {};
      const posList = v.layer49_self_report?.positive_boosted_words || [];
      const negList = v.layer49_self_report?.negative_boosted_words || [];
      const p0 = (v.pairs && v.pairs[0]) || null;

      return `
        <div class="stage-box">
          <div class="stage-box-header">
            <span class="stage-slug">${escapeHtml(v.key)}</span>
            <span class="meta-kv">‖v‖ = ${v.layer49_raw_norm.toFixed(2)} • ${escapeHtml(v.modality)}</span>
          </div>
          <div class="stage-quality">${escapeHtml(vh.quality || "")}</div>
          <div class="pole-box pole-neg"><span class="pole-tag">a_side (-v):</span>${escapeHtml(vh.a_side || "")}</div>
          <div class="pole-box pole-pos"><span class="pole-tag">b_side (+v):</span>${escapeHtml(vh.b_side || "")}</div>
          <div style="font-size:12px; color:var(--text-1);">
            <strong>L49 Self-Report (<code>${escapeHtml(synth.self_report_slug || "")}</code>):</strong>
            ${escapeHtml(synth.self_report_quality || "")}
          </div>
          <div>
            <div style="font-size:11px; font-family:var(--font-mono); color:var(--pos); margin-bottom:4px;">Top +v_encoder Words (shared highlighted):</div>
            ${posList.map(w => `<span class="${otherPosSet.has(w.word) ? 'shared-word-chip' : 'word-chip'}">${escapeHtml(w.raw)}</span>`).join("")}
          </div>
          <div>
            <div style="font-size:11px; font-family:var(--font-mono); color:var(--neg); margin-bottom:4px;">Top -v_encoder Words (shared highlighted):</div>
            ${negList.map(w => `<span class="${otherNegSet.has(w.word) ? 'shared-word-chip' : 'word-chip'}">${escapeHtml(w.raw)}</span>`).join("")}
          </div>
          ${p0 ? `
            <div style="border-top:1px dashed var(--border); padding-top:8px; font-size:12px;">
              <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-2); margin-bottom:4px;">Sample Pair #1 Differing Clause:</div>
              <div><span class="diff-a-inline">[A: ${escapeHtml(p0.a_span)}]</span> <span class="diff-arrow">→</span> <span class="diff-b-inline">[B: ${escapeHtml(p0.b_span)}]</span></div>
            </div>
          ` : ""}
        </div>
      `;
    };

    document.getElementById("compDrawerBody").innerHTML = `
      <div style="margin-bottom:12px; font-size:12.5px; color:var(--text-1); font-family:var(--font-mono);">
        Shared +v_encoder top words (${sharedPos.length}): ${sharedPos.length ? sharedPos.map(w => `<span class="shared-word-chip">${escapeHtml(w)}</span>`).join("") : "none"}
        &nbsp;•&nbsp;
        Shared -v_encoder top words (${sharedNeg.length}): ${sharedNeg.length ? sharedNeg.map(w => `<span class="shared-word-chip">${escapeHtml(w)}</span>`).join("") : "none"}
      </div>
      <div class="anchor-grid">
        ${renderVecCol(v1, setPos2, setNeg2)}
        ${renderVecCol(v2, setPos1, setNeg1)}
      </div>
    `;
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    document.querySelectorAll(".nav-tab").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-panel").forEach(panel => {
      panel.classList.toggle("active", panel.id === `tab-${tabId}`);
    });
  }

  // Client-side lightweight .toml and .json loader for live browser drops
  function parseTomlControlVectorClient(tomlText, fileName) {
    const stem = fileName.replace(/\.toml$/i, "");
    const vecHeader = { name: stem, quality: "", a_side: "", b_side: "", mode: "t2va", frame: "scene" };
    const headerBlock = tomlText.split("[[pair]]")[0] || "";
    for (const key of ["name", "quality", "a_side", "b_side", "mode", "frame"]) {
      const m = headerBlock.match(new RegExp(`${key}\\s*=\\s*"([^"]*)"`));
      if (m) vecHeader[key] = m[1];
    }
    const pairBlocks = tomlText.split("[[pair]]").slice(1);
    const pairs = [];
    pairBlocks.forEach((blk, idx) => {
      const ma = blk.match(/a\s*=\s*(?:'''([\s\S]*?)'''|"([^"]*)")/);
      const mb = blk.match(/b\s*=\s*(?:'''([\s\S]*?)'''|"([^"]*)")/);
      const aTxt = (ma ? (ma[1] ?? ma[2] ?? "") : "").trim();
      const bTxt = (mb ? (mb[1] ?? mb[2] ?? "") : "").trim();
      if (aTxt && bTxt) {
        pairs.push(analyzePairClient(idx, aTxt, bTxt, vecHeader.frame));
      }
    });
    return { stem, vecHeader, pairs };
  }

  function segmentTextClient(text) {
    if (!text) return [];
    const re = /(<d>[\s\S]*?<\/d>|\(S\d+(?:,\s*S\d+)*\))/g;
    const out = [];
    let last = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push({ text: text.slice(last, m.index), kind: "normal" });
      const tok = m[0];
      out.push({ text: tok, kind: tok.startsWith("<d>") ? "dialogue" : "speaker" });
      last = re.lastIndex;
    }
    if (last < text.length) out.push({ text: text.slice(last), kind: "normal" });
    return out;
  }

  function countWordsClient(text) {
    if (!text) return [0, 0];
    const allW = text.trim() ? text.trim().split(/\s+/).length : 0;
    const cleaned = text.replace(/<d>[\s\S]*?<\/d>/g, " ").replace(/\(S\d+(?:,\s*S\d+)*\)/g, " ").trim();
    const actW = cleaned ? cleaned.split(/\s+/).length : 0;
    return [actW, Math.max(0, allW - actW)];
  }

  function analyzePairClient(idx, aTxt, bTxt, frame) {
    const aToks = aTxt.match(/\S+|\s+/g) || [];
    const bToks = bTxt.match(/\S+|\s+/g) || [];
    let px = 0;
    const minL = Math.min(aToks.length, bToks.length);
    while (px < minL && aToks[px] === bToks[px]) px++;
    let sx = 0;
    while (sx < (minL - px) && aToks[aToks.length - 1 - sx] === bToks[bToks.length - 1 - sx]) sx++;
    const sharedPrefix = aToks.slice(0, px).join("");
    const aSpan = aToks.slice(px, aToks.length - sx).join("");
    const bSpan = bToks.slice(px, bToks.length - sx).join("");
    const sharedSuffix = sx > 0 ? aToks.slice(aToks.length - sx).join("") : "";

    let keptSuffix = sharedSuffix;
    let trimmedSound = "";
    let soundIdentical = false;
    let editSec = frame === "bare" ? "bare" : "integrated_multimodal_description";
    if (frame !== "bare") {
      for (const marker of ["\n\noverall_soundscape:", "\noverall_soundscape:"]) {
        if (aTxt.includes(marker) && bTxt.includes(marker)) {
          const [ap, at] = aTxt.split(marker);
          const [bp, bt] = bTxt.split(marker);
          if (at === bt && ap !== bp) {
            soundIdentical = true;
            const mIdx = sharedSuffix.indexOf(marker);
            if (mIdx !== -1) {
              keptSuffix = sharedSuffix.slice(0, mIdx);
              trimmedSound = sharedSuffix.slice(mIdx);
            }
          } else if (ap === bp && at !== bt) {
            editSec = "overall_soundscape";
          }
          break;
        }
      }
    }
    const [preAct, preDlg] = countWordsClient(sharedPrefix);
    const [aAct, aDlg] = countWordsClient(aSpan);
    const [bAct, bDlg] = countWordsClient(bSpan);
    const diffAvg = 0.5 * (aAct + bAct);
    const [ksufAct, ksufDlg] = countWordsClient(keptSuffix);
    const [tsufAct, tsufDlg] = countWordsClient(trimmedSound);
    const totDlg = preDlg + 0.5 * (aDlg + bDlg) + ksufDlg + tsufDlg;
    const totAll = preAct + diffAvg + ksufAct + tsufAct + totDlg;
    const stAct = preAct + diffAvg + ksufAct + tsufAct;
    const ffdAct = frame !== "bare" ? (diffAvg + ksufAct) : stAct;

    return {
      index: idx,
      a: aTxt,
      b: bTxt,
      shared_prefix: sharedPrefix,
      a_span: aSpan,
      b_span: bSpan,
      shared_suffix: sharedSuffix,
      kept_suffix: keptSuffix,
      trimmed_soundscape_suffix: trimmedSound,
      soundscape_identical: soundIdentical,
      edit_section: editSec,
      segments: {
        shared_prefix: segmentTextClient(sharedPrefix),
        a_span: segmentTextClient(aSpan),
        b_span: segmentTextClient(bSpan),
        kept_suffix: segmentTextClient(keptSuffix),
        trimmed_soundscape_suffix: segmentTextClient(trimmedSound)
      },
      stats: {
        total_words_avg: totAll,
        prefix_active_words: preAct,
        diff_words_a: aAct,
        diff_words_b: bAct,
        diff_active_words_avg: diffAvg,
        kept_suffix_active_words: ksufAct,
        trimmed_soundscape_active_words: tsufAct,
        excluded_dialogue_words: totDlg,
        scene_text_active_words: stAct,
        from_first_diff_active_words: ffdAct,
        scene_text_diff_pct: Number((100 * diffAvg / Math.max(stAct, 1)).toFixed(1)),
        from_first_diff_diff_pct: Number((100 * diffAvg / Math.max(ffdAct, 1)).toFixed(1)),
        prefix_pct_of_total: Number((100 * preAct / Math.max(totAll, 1)).toFixed(1)),
        diff_pct_of_total: Number((100 * diffAvg / Math.max(totAll, 1)).toFixed(1)),
        kept_suffix_pct_of_total: Number((100 * ksufAct / Math.max(totAll, 1)).toFixed(1)),
        trimmed_soundscape_pct_of_total: Number((100 * tsufAct / Math.max(totAll, 1)).toFixed(1)),
        dialogue_pct_of_total: Number((100 * totDlg / Math.max(totAll, 1)).toFixed(1))
      }
    };
  }

  function parseWordListClient(arr) {
    return (arr || []).map(item => {
      if (item && typeof item === "object" && "word" in item) return item;
      const raw = String(item);
      const m = raw.match(/^(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)$/);
      return m
        ? { word: m[1].trim(), delta_log_prob: parseFloat(m[2]), raw }
        : { word: raw, delta_log_prob: 0, raw };
    });
  }

  async function handleFilesSelected(fileList) {
    let loadedCount = 0;
    for (const file of fileList) {
      const text = await file.text();
      if (file.name.endsWith(".json")) {
        const parsed = JSON.parse(text);
        if (parsed.vectors && parsed.vector_order) {
          appData = parsed;
        } else {
          Object.keys(parsed).forEach(k => {
            const entry = parsed[k];
            const existing = appData.vectors[k] || { key: k, pairs: [], pair_summary: {} };
            const sr = entry.layer49_self_report || {};
            const lex = sr.lexical_logit_shifts || {};
            existing.toml_file = entry.toml_file || `${k}.toml`;
            existing.toml_sha256 = entry.toml_sha256 || "";
            existing.layer49_raw_norm = Number(entry.layer49_raw_norm || 0);
            existing.layer49_median_row_norm_ex_row0 = Number(entry.layer49_median_row_norm_ex_row0 || 0);
            existing.blind_pair_label = entry.blind_pair_label || {};
            existing.modality = existing.blind_pair_label.modality || sr.synthesis?.dominant_modality || "visual";
            existing.layer49_self_report = {
              median_row_norm: sr.median_row_norm || existing.layer49_median_row_norm_ex_row0,
              sampling: sr.sampling || {},
              positive_boosted_words: parseWordListClient(lex.positive_boosted_words || sr.positive_boosted_words),
              negative_boosted_words: parseWordListClient(lex.negative_boosted_words || sr.negative_boosted_words),
              sweep_completions: sr.sweep_completions || {},
              synthesis: sr.synthesis || {}
            };
            appData.vectors[k] = existing;
            if (!appData.vector_order.includes(k)) appData.vector_order.push(k);
          });
        }
        loadedCount++;
      } else if (file.name.endsWith(".toml")) {
        const { stem, vecHeader, pairs } = parseTomlControlVectorClient(text, file.name);
        const existing = appData.vectors[stem] || {
          key: stem,
          toml_file: file.name,
          layer49_raw_norm: 0,
          layer49_median_row_norm_ex_row0: 296.0,
          modality: "visual",
          blind_pair_label: {},
          layer49_self_report: { positive_boosted_words: [], negative_boosted_words: [], sweep_completions: {}, synthesis: {} }
        };
        existing.vector_header = vecHeader;
        existing.pairs = pairs;
        const n = Math.max(pairs.length, 1);
        existing.pair_summary = {
          num_pairs: pairs.length,
          avg_scene_text_diff_pct: Number((pairs.reduce((s, p) => s + p.stats.scene_text_diff_pct, 0) / n).toFixed(1)),
          avg_from_first_diff_diff_pct: Number((pairs.reduce((s, p) => s + p.stats.from_first_diff_diff_pct, 0) / n).toFixed(1)),
          avg_scene_text_active_words: Number((pairs.reduce((s, p) => s + p.stats.scene_text_active_words, 0) / n).toFixed(1)),
          avg_from_first_diff_active_words: Number((pairs.reduce((s, p) => s + p.stats.from_first_diff_active_words, 0) / n).toFixed(1)),
          avg_prefix_pct: Number((pairs.reduce((s, p) => s + p.stats.prefix_pct_of_total, 0) / n).toFixed(1)),
          avg_diff_pct: Number((pairs.reduce((s, p) => s + p.stats.diff_pct_of_total, 0) / n).toFixed(1)),
          avg_kept_suffix_pct: Number((pairs.reduce((s, p) => s + p.stats.kept_suffix_pct_of_total, 0) / n).toFixed(1)),
          avg_trimmed_soundscape_pct: Number((pairs.reduce((s, p) => s + p.stats.trimmed_soundscape_pct_of_total, 0) / n).toFixed(1)),
          avg_dialogue_pct: Number((pairs.reduce((s, p) => s + p.stats.dialogue_pct_of_total, 0) / n).toFixed(1)),
          soundscape_trimmed_pairs: pairs.filter(p => p.soundscape_identical).length,
          sound_only_pairs: pairs.filter(p => p.edit_section === "overall_soundscape").length
        };
        appData.vectors[stem] = existing;
        if (!appData.vector_order.includes(stem)) appData.vector_order.push(stem);
        state.selectedVector = stem;
        loadedCount++;
      }
    }
    if (loadedCount > 0) {
      showToast(`Loaded ${loadedCount} file(s)`);
      renderAll();
    }
  }

  function bindEvents() {
    document.querySelectorAll(".nav-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        switchTab(btn.getAttribute("data-tab"));
      });
    });

    document.getElementById("vecSearchInput")?.addEventListener("input", e => {
      state.searchQuery = e.target.value;
      renderSidebar();
    });

    document.querySelectorAll("#modalityFilterGroup .filter-pill").forEach(btn => {
      btn.addEventListener("click", () => {
        state.modalityFilter = btn.getAttribute("data-filter");
        document.querySelectorAll("#modalityFilterGroup .filter-pill").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderSidebar();
      });
    });

    document.querySelectorAll("#poolingModeGroup .btn").forEach(btn => {
      btn.addEventListener("click", () => {
        state.poolingMode = btn.getAttribute("data-pooling");
        document.querySelectorAll("#poolingModeGroup .btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const v = appData.vectors[state.selectedVector];
        if (v) renderPairsSection(v);
      });
    });

    document.querySelectorAll("#pairViewModeGroup .btn").forEach(btn => {
      btn.addEventListener("click", () => {
        state.pairViewMode = btn.getAttribute("data-view");
        document.querySelectorAll("#pairViewModeGroup .btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const v = appData.vectors[state.selectedVector];
        if (v) renderPairsSection(v);
      });
    });

    document.querySelectorAll("#alphaFilterGroup .btn").forEach(btn => {
      btn.addEventListener("click", () => {
        state.alphaFilter = btn.getAttribute("data-alpha");
        document.querySelectorAll("#alphaFilterGroup .btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderDeepDive();
      });
    });

    document.getElementById("toggleRolloutDiffBtn")?.addEventListener("click", e => {
      state.highlightRolloutDiff = !state.highlightRolloutDiff;
      e.currentTarget.classList.toggle("active", state.highlightRolloutDiff);
      renderDeepDive();
    });

    document.querySelectorAll("#overviewCompTable th[data-sort]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.getAttribute("data-sort");
        if (state.sortCol === col) {
          state.sortAsc = !state.sortAsc;
        } else {
          state.sortCol = col;
          state.sortAsc = col === "key" || col === "modality" || col === "frame";
        }
        renderOverviewTable();
      });
    });

    document.getElementById("fileLoaderInput")?.addEventListener("change", e => {
      if (e.target.files && e.target.files.length) {
        handleFilesSelected(e.target.files);
      }
    });

    document.getElementById("refreshLiveBtn")?.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/data", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        appData = await res.json();
        showToast("Refreshed live data from disk");
        renderAll();
      } catch (err) {
        showToast("Offline / static file mode (use 📂 Load JSON / TOML)");
      }
    });
  }

  function renderAll() {
    renderSidebar();
    renderDeepDive();
    renderOverviewTable();
    renderCosineMatrix();
  }

  bindEvents();
  renderAll();
})();
</script>
</body>
</html>
"""


def generate_html(payload: dict[str, Any]) -> str:
    json_str = json.dumps(payload, ensure_ascii=False)
    # Escape closing script tags if any ever appeared inside strings
    json_str = json_str.replace("</script>", "<\\/script>")
    return HTML_TEMPLATE.replace("__EMBEDDED_JSON_PAYLOAD__", json_str)


def run_server(h3_root: Path, exl3_dir: Path, out_html: Path, host: str, port: int) -> None:
    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(out_html.parent), **kwargs)

        def do_GET(self) -> None:
            if self.path.startswith("/api/data"):
                payload = build_viewer_payload(h3_root, exl3_dir)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in ("/", "/index.html"):
                self.path = f"/{out_html.name}"
            super().do_GET()

    server = ThreadingHTTPServer((host, port), ViewerHandler)
    print(f"Serving H3 Control Vector Explorer at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and optionally serve the H3 Control Vector HTML Explorer."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (defaults to control_vector_viewer.html beside this script).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not copy the script and HTML to the sibling exllamav3 Qwen3-VL-32B-Instruct folder.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server with live /api/data reload.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8788, help="HTTP bind port.")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    h3_root = _resolve_h3_root()
    exl3_dir = h3_root / "docs" / "research" / "control_vectors" / "exl3_qwen3vl"
    if not exl3_dir.is_dir():
        exl3_dir = here.parent

    out_html = (args.output or (exl3_dir / "control_vector_viewer.html")).resolve()
    payload = build_viewer_payload(h3_root, exl3_dir)
    html_text = generate_html(payload)
    out_html.write_text(html_text, encoding="utf-8")

    n_vecs = len(payload.get("vectors", {}))
    print(f"Built {out_html.name} ({len(html_text):,} bytes, {n_vecs} control vectors).")

    if not args.no_sync:
        ext_dir = _resolve_exl3_model_script_dir(h3_root)
        if ext_dir and ext_dir.is_dir():
            for src in (here, out_html):
                dst = ext_dir / src.name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                    print(f"Synced {src.name} -> {dst.name} in exllamav3 model dir.")

    if args.serve:
        run_server(h3_root, exl3_dir, out_html, args.host, args.port)


if __name__ == "__main__":
    main()
