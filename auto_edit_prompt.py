"""`heylook`-Driven Auto-Edit Guidance (`MiniMaxH3AutoEditPrompt`) and 4-Axis `EditPlan`.

Translates `(original, concept | edit_intent, optional images)` via a local or LAN
`heylookitsanllm` (`POST /v1/messages`) server into a deterministically validated
`edited_prompt` plus a 4-axis `EditPlan` wire (`MINIMAX_H3_EDIT_PLAN`) that auto-configures
`MiniMaxH3EditGuider` (`first_step`, `last_step`, `streams`, `audio_scale`, `[F, 1, 1]`
temporal curve, `fast_1x` 1-pass override) and `MiniMaxH3ReleaseReferences`
(`end_percent`) while keeping `MiniMaxH3EditGuider.scale` cached as the master intensity dial.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Mapping, Sequence

import httpx

_COMFY_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "comfy_api").is_dir()),
    None,
)
if _COMFY_ROOT is not None and str(_COMFY_ROOT) not in sys.path:
    sys.path.append(str(_COMFY_ROOT))

from comfy_api.latest import io  # noqa: E402

try:
    from . import control_vectors as _cv  # noqa: E402
    from . import h3_prompt_edit as _pe  # noqa: E402
    from . import h3_rules as _rules  # noqa: E402
    from . import h3_spans as _spans  # noqa: E402
    from .heylook_client import (  # noqa: E402
        SAMPLER_MODES,
        THINKING_MODES,
        HeylookStreamResult,
        _comfy_interrupt_check,
        _extract_inline_thinking,
        build_heylook_request_body,
        discover_heylook_model,
        expand_heylook_preset,
        format_heylook_report,
        prepare_heylook_tensor,
        probe_heylook_server_info,
        resolve_heylook_base_url,
        stream_heylook_message,
        with_heylook_backoff,
    )
except ImportError:  # run from the repo root by a check or script
    import control_vectors as _cv  # noqa: E402
    import h3_prompt_edit as _pe  # noqa: E402
    import h3_rules as _rules  # noqa: E402
    import h3_spans as _spans  # noqa: E402
    from heylook_client import (  # noqa: E402
        SAMPLER_MODES,
        THINKING_MODES,
        HeylookStreamResult,
        _comfy_interrupt_check,
        _extract_inline_thinking,
        build_heylook_request_body,
        discover_heylook_model,
        expand_heylook_preset,
        format_heylook_report,
        prepare_heylook_tensor,
        probe_heylook_server_info,
        resolve_heylook_base_url,
        stream_heylook_message,
        with_heylook_backoff,
    )


_HERE = Path(__file__).resolve().parent
_SYSTEM_PROMPTS_DIR = _HERE / "docs" / "wiki" / "system_prompts" / "h3_auto_edit"
PROMPT_BANK_CV_DIR = _HERE / "prompt_bank" / "control_vectors"
DATA_CV_DIR = _HERE / "data" / "control_vectors"
DATA_CONCEPTS_DIR = _HERE / "data" / "edit_concepts"

H3EditPlan = io.Custom("MINIMAX_H3_EDIT_PLAN")

TARGET_SCOPES: tuple[str, ...] = (
    "auto",
    "scene_only",
    "description_only",
    "look_only",
    "soundscape_only",
    "scene_and_soundscape",
    "all_sections",
)
REWRITE_MODES: tuple[str, ...] = ("section_edits", "edit_lines")
TAKE_MODES: tuple[str, ...] = ("auto", "fast_1x", "lock_take", "allow_reshoot", "manual_widgets")
TIMING_MODES: tuple[str, ...] = ("auto", "whole_clip", "ramp_in", "sudden_midclip", "second_half")
TEMPORAL_CURVES: tuple[str, ...] = ("whole_clip", "ramp_in", "sudden_midclip", "second_half")
PROMPT_LAYOUTS: tuple[str, ...] = ("system_and_user", "user_only")
MAX_AUTO_EDIT_IMAGES = 4
_CACHE_MAXSIZE = 32

_SECTION_LINE_RE = re.compile(
    r"(?m)^[ \t]*(integrated_multimodal_description|subject_definitions|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):"
)
_DIALOGUE_SPAN_RE = re.compile(r"<d>.*?</d>", re.S)
_SHOT_HEADER_RE = re.compile(r"\[Shot\s+\d+\]")
_MEDIA_REF_TAG_RE = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")
_TASK_BRACKET_RE = re.compile(r"^\s*(\[[^\]]+\])")
_RETENTION_LINE_RE = re.compile(
    r"^([ \t]*<(Subject|Picture|Video|Audio)\s+\d+>(?:\s*\([^)\n]*\))?\s*:\s*)([a-z_]+)(.*)$"
)
_RETENTION_MODES = frozenset({
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
    "fully_copy",
    "partially_copy",
    "reference",
})
_UNSECTIONED_ALIASES = frozenset({"_unsectioned", "unsectioned", "prompt", "text", "caption", "body"})
_SECTION_ALIAS_MAP = {
    "subject_definitions": "subject_definitions",
    "subjects": "subject_definitions",
    "summary": "summary",
    "retention_analysis": "retention_analysis",
    "retention": "retention_analysis",
    "detailed_description": "detailed_description",
    "integrated_multimodal_description": "integrated_multimodal_description",
    "overall_soundscape": "overall_soundscape",
    "soundscape": "overall_soundscape",
    "audio": "overall_soundscape",
    "foley": "overall_soundscape",
    "sound": "overall_soundscape",
    "non_diegetic_music": "non_diegetic_music",
    "music": "non_diegetic_music",
    "score": "non_diegetic_music",
}
_VISUAL_SECTIONS = frozenset({
    "_unsectioned",
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "integrated_multimodal_description",
})
_AUDIO_SECTIONS = frozenset({"overall_soundscape", "non_diegetic_music"})
_RETENTION_LOCKED_SCOPES = frozenset({"soundscape_only", "description_only"})


class ValidationRepairError(ValueError):
    """Deterministic post-validation failure eligible for a 1-shot APC-cached repair turn."""


@dataclass(frozen=True)
class EditPlan:
    """4-axis physical edit plan carried on `MINIMAX_H3_EDIT_PLAN` wires."""

    first_step: int
    last_step: int
    streams: str
    audio_scale: float
    ref_end_percent: float
    temporal_curve: str
    force_scale: float | None
    manual_widgets: bool
    visual_changed: bool
    audio_changed: bool
    preserve_step0_layout: bool
    deforms_ref_silhouette: bool
    reason: str = ""


@dataclass(frozen=True)
class SectionChunk:
    name: str
    header_raw: str
    leading_ws: str
    body: str
    trailing_ws: str


@dataclass(frozen=True)
class ConceptBrief:
    stem: str
    source_kind: str  # "edit_concept" | "control_vector"
    path: Path
    file_stat: tuple[str, int, int]
    title: str
    summary: str
    a_side: str
    b_side: str
    avoid: str
    notes: str
    modified_sections: tuple[str, ...]
    exemplars: tuple[tuple[str, str, str], ...]  # (section, a_span, b_span)
    target_scope_hint: str | None = None
    timing_hint: str | None = None
    axis_hints: Mapping[str, Any] = None  # type: ignore[assignment]


_REWRITE_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()


def clear_rewrite_cache() -> None:
    """Clear the in-process `MiniMaxH3AutoEditPrompt` rewrite cache."""
    _REWRITE_CACHE.clear()


def _read_system_prompt_file(rewrite_mode: str) -> str:
    fname = (
        "section_edits_system_prompt.md"
        if rewrite_mode == "section_edits"
        else "edit_lines_system_prompt.md"
    )
    return (_SYSTEM_PROMPTS_DIR / fname).read_text(encoding="utf-8").strip()


def _concept_search_dirs(
    *,
    concepts_dir: Path | None = None,
    data_cv_dir: Path | None = None,
    bank_cv_dir: Path | None = None,
) -> tuple[tuple[str, Path], ...]:
    return (
        ("edit_concept", concepts_dir or DATA_CONCEPTS_DIR),
        ("control_vector", data_cv_dir or DATA_CV_DIR),
        ("control_vector", bank_cv_dir or PROMPT_BANK_CV_DIR),
    )


def list_concept_options(
    *,
    concepts_dir: Path | None = None,
    data_cv_dir: Path | None = None,
    bank_cv_dir: Path | None = None,
) -> list[str]:
    """Return `['none', *sorted_stems]` across `data/edit_concepts/`, `data/control_vectors/`, and `prompt_bank/control_vectors/`."""
    stems: set[str] = set()
    for _kind, d in _concept_search_dirs(
        concepts_dir=concepts_dir, data_cv_dir=data_cv_dir, bank_cv_dir=bank_cv_dir
    ):
        if d.is_dir():
            for p in d.glob("*.toml"):
                if p.is_file():
                    stems.add(p.stem)
    return ["none", *sorted(stems)]


def resolve_concept_path(
    name: str,
    *,
    concepts_dir: Path | None = None,
    data_cv_dir: Path | None = None,
    bank_cv_dir: Path | None = None,
) -> tuple[str, Path]:
    """Resolve a concept stem or `.toml` filename in priority order."""
    raw = str(name or "").strip()
    if not raw or raw.lower() == "none":
        raise ValueError("concept is 'none'")
    cand = Path(raw)
    if cand.is_file():
        resolved = cand.resolve()
        try:
            doc = tomllib.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"{resolved}: invalid TOML syntax: {exc}") from exc
        kind = "edit_concept" if "concept" in doc else "control_vector"
        return kind, resolved
    fname = raw if raw.endswith(".toml") else f"{raw}.toml"
    for kind, d in _concept_search_dirs(
        concepts_dir=concepts_dir, data_cv_dir=data_cv_dir, bank_cv_dir=bank_cv_dir
    ):
        p = d / fname
        if p.is_file():
            return kind, p.resolve()
    raise ValueError(
        f"concept {name!r} not found under data/edit_concepts/, data/control_vectors/, or prompt_bank/control_vectors/"
    )


def _sample_exemplars(
    pairs: Sequence[_cv.PairItem],
    *,
    frame: str,
    count: int,
) -> tuple[tuple[str, str, str], ...]:
    n = min(max(1, int(count)), len(pairs))
    if n <= 0:
        return ()
    if n >= len(pairs):
        indices = list(range(len(pairs)))
    else:
        step = (len(pairs) - 1) / max(1, n - 1) if n > 1 else 0.0
        indices = []
        for k in range(n):
            idx = round(k * step)
            if idx not in indices:
                indices.append(idx)
        for idx in range(len(pairs)):
            if len(indices) >= n:
                break
            if idx not in indices:
                indices.append(idx)
    out: list[tuple[str, str, str]] = []
    for idx in indices:
        item = pairs[idx]
        _, sec_str = _cv.differing_section_info(item.a, item.b, frame=frame)
        out.append((sec_str, item.a_span, item.b_span))
    return tuple(out)


def load_concept_brief(
    name: str,
    *,
    exemplar_count: int = 3,
    concepts_dir: Path | None = None,
    data_cv_dir: Path | None = None,
    bank_cv_dir: Path | None = None,
) -> ConceptBrief:
    """Load a `[concept]` card (`data/edit_concepts/*.toml`) or a contrastive `[vector]` TOML."""
    kind, resolved = resolve_concept_path(
        name, concepts_dir=concepts_dir, data_cv_dir=data_cv_dir, bank_cv_dir=bank_cv_dir
    )
    st = resolved.stat()
    file_stat = (str(resolved), int(st.st_mtime_ns), int(st.st_size))

    if kind == "edit_concept":
        try:
            doc = tomllib.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"{resolved}: invalid TOML syntax: {exc}") from exc
        c = doc.get("concept")
        if not isinstance(c, dict):
            raise ValueError(f"{resolved}: missing required [concept] table")
        intent = str(c.get("intent") or "").strip()
        if not intent:
            raise ValueError(f"{resolved}: concept.intent must be a non-empty string")
        cname = str(c.get("name") or resolved.stem).strip() or resolved.stem
        notes = str(c.get("notes") or "").strip()
        avoid = str(c.get("avoid") or "").strip()
        t_scope = str(c.get("target_scope") or "").strip() or None
        if t_scope is not None and t_scope not in TARGET_SCOPES:
            raise ValueError(f"{resolved}: invalid target_scope {t_scope!r}; expected one of {TARGET_SCOPES}")
        timing_hint = str(c.get("timing") or "").strip() or None
        if timing_hint is not None and timing_hint not in TIMING_MODES:
            raise ValueError(f"{resolved}: invalid timing {timing_hint!r}; expected one of {TIMING_MODES}")

        raw_examples = doc.get("example") or doc.get("pair") or []
        exemplars: list[tuple[str, str, str]] = []
        if isinstance(raw_examples, list):
            for ex in raw_examples[: max(1, int(exemplar_count))]:
                if isinstance(ex, dict) and str(ex.get("a") or "").strip() and str(ex.get("b") or "").strip():
                    sec = str(ex.get("section") or "description").strip()
                    exemplars.append((sec, str(ex["a"]).strip(), str(ex["b"]).strip()))

        axis_hints: dict[str, Any] = {}
        for k in ("visual_changed", "audio_changed", "preserve_step0_layout", "deforms_ref_silhouette"):
            if k in c and isinstance(c[k], bool):
                axis_hints[k] = c[k]
        mod_secs: list[str] = []
        if axis_hints.get("visual_changed") is True:
            mod_secs.append("detailed_description")
        if axis_hints.get("audio_changed") is True:
            mod_secs.append("overall_soundscape")
        return ConceptBrief(
            stem=resolved.stem,
            source_kind="edit_concept",
            path=resolved,
            file_stat=file_stat,
            title=cname,
            summary=intent,
            a_side="",
            b_side="",
            avoid=avoid,
            notes=notes,
            modified_sections=tuple(mod_secs),
            exemplars=tuple(exemplars),
            target_scope_hint=t_scope,
            timing_hint=timing_hint,
            axis_hints=axis_hints,
        )

    ps = _cv.load_pairs(resolved)
    mod_secs = tuple(_cv.pair_set_differing_sections(ps.pairs, frame=ps.frame))
    exemplars = _sample_exemplars(ps.pairs, frame=ps.frame, count=exemplar_count)
    return ConceptBrief(
        stem=resolved.stem,
        source_kind="control_vector",
        path=resolved,
        file_stat=file_stat,
        title=ps.name,
        summary=ps.quality,
        a_side=ps.a_side,
        b_side=ps.b_side,
        avoid="",
        notes="",
        modified_sections=mod_secs,
        exemplars=exemplars,
        target_scope_hint=None,
        timing_hint=None,
        axis_hints={},
    )


def parse_h3_sections(text: str) -> tuple[str, list[SectionChunk]]:
    """Split an H3 prompt losslessly into `(leading_prefix, chunks)` preserving exact whitespace."""
    matches = list(_SECTION_LINE_RE.finditer(text))
    if not matches:
        return "", [SectionChunk(name="_unsectioned", header_raw="", leading_ws="", body=text, trailing_ws="")]

    leading_prefix = text[: matches[0].start()]
    chunks: list[SectionChunk] = []
    for idx, m in enumerate(matches):
        name = m.group(1)
        header_raw = text[m.start() : m.end()]
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        after_colon = text[m.end() : next_start]
        stripped = after_colon.strip()
        if not stripped:
            leading_ws = after_colon
            body = ""
            trailing_ws = ""
        else:
            b0 = after_colon.find(stripped)
            leading_ws = after_colon[:b0]
            body = stripped
            trailing_ws = after_colon[b0 + len(stripped) :]
        chunks.append(
            SectionChunk(
                name=name,
                header_raw=header_raw,
                leading_ws=leading_ws,
                body=body,
                trailing_ws=trailing_ws,
            )
        )
    return leading_prefix, chunks


def reassemble_h3_sections(
    leading_prefix: str,
    chunks: Sequence[SectionChunk],
    bodies: Mapping[str, str],
) -> str:
    """Reassemble sections losslessly; untouched sections remain byte-for-byte identical."""
    parts = [leading_prefix]
    for c in chunks:
        body = bodies.get(c.name, c.body)
        parts.append(f"{c.header_raw}{c.leading_ws}{body}{c.trailing_ws}")
    return "".join(parts)


def resolve_allowed_sections(
    original: str,
    target_scope: str,
    *,
    allow_retention_relax: bool,
    concept_brief: ConceptBrief | None,
    has_edit_intent: bool,
) -> tuple[str, tuple[str, ...]]:
    """Return `(resolved_scope, allowed_section_names_in_prompt_order)`."""
    if target_scope not in TARGET_SCOPES:
        raise ValueError(f"unknown target_scope {target_scope!r}; expected one of {TARGET_SCOPES}")

    _pfx, chunks = parse_h3_sections(original)
    present = [c.name for c in chunks]
    present_set = set(present)

    if present == ["_unsectioned"]:
        resolved_scope = (
            concept_brief.target_scope_hint
            if target_scope == "auto" and concept_brief and concept_brief.target_scope_hint
            else ("scene_only" if target_scope == "auto" else target_scope)
        )
        return resolved_scope, ("_unsectioned",)

    eff_scope = target_scope
    if eff_scope == "auto":
        if concept_brief and concept_brief.target_scope_hint and not has_edit_intent:
            eff_scope = concept_brief.target_scope_hint
        elif concept_brief and concept_brief.modified_sections and not has_edit_intent:
            mods = set(concept_brief.modified_sections)
            if mods <= _AUDIO_SECTIONS:
                eff_scope = "soundscape_only"
            else:
                eff_scope = "scene_and_soundscape"
        else:
            eff_scope = "scene_and_soundscape"

    desc_names = [s for s in ("detailed_description", "integrated_multimodal_description") if s in present_set]
    look_names = [s for s in ("subject_definitions", "summary") if s in present_set]
    sound_names = [s for s in ("overall_soundscape", "non_diegetic_music") if s in present_set]

    allowed: list[str] = []
    if eff_scope == "look_only":
        allowed.extend(look_names)
    elif eff_scope == "description_only":
        allowed.extend(desc_names)
    elif eff_scope == "scene_only":
        allowed.extend(look_names)
        allowed.extend(desc_names)
    elif eff_scope == "soundscape_only":
        allowed.extend(sound_names)
    elif eff_scope == "scene_and_soundscape":
        allowed.extend(look_names)
        allowed.extend(desc_names)
        allowed.extend(sound_names)
    elif eff_scope == "all_sections":
        allowed.extend(s for s in present if s != "retention_analysis")

    if (
        allow_retention_relax
        and "retention_analysis" in present_set
        and eff_scope not in _RETENTION_LOCKED_SCOPES
        and "retention_analysis" not in allowed
    ):
        allowed.append("retention_analysis")

    ordered = tuple(s for s in present if s in set(allowed))
    if not ordered:
        raise ValueError(
            f"target_scope={eff_scope!r} matched no editable sections in original prompt (sections present: {present})"
        )
    return eff_scope, ordered


def _strip_think_and_fence(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raise ValueError(f"expected str reply, got {type(raw_text).__name__}")
    s = raw_text.strip()
    while s.startswith("<think>"):
        end = s.find("</think>")
        if end < 0:
            raise ValidationRepairError("reply contains an unclosed <think> block")
        s = s[end + len("</think>") :].strip()
    if "</think>" in s and "<think>" not in s.split("</think>", 1)[0]:
        s = s.split("</think>", 1)[1].strip()
    if "<think>" in s:
        raise ValidationRepairError("reply contains an unclosed or mid-output <think> tag")

    if s.startswith("```") and s.endswith("```"):
        first_nl = s.find("\n")
        if first_nl > 0:
            inner = s[first_nl + 1 : -3].strip()
            if "```" not in inner:
                s = inner
    if not s:
        raise ValidationRepairError("model returned an empty reply")
    return s


def _extract_balanced_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj

    start = text.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
                    break
    raise ValidationRepairError(
        "reply must be a single valid JSON object containing 'section_edits'/'edit_lines' and 'axes'"
    )


def _normalize_axes(raw_axes: Any, *, concept_brief: ConceptBrief | None = None) -> dict[str, Any]:
    if not isinstance(raw_axes, Mapping):
        raise ValidationRepairError("JSON reply is missing required 'axes' object")

    def _as_bool(key: str, default: bool) -> bool:
        val = raw_axes.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
        raise ValidationRepairError(f"axes.{key} must be a boolean, got {val!r}")

    curve_raw = str(raw_axes.get("temporal_curve") or "whole_clip").strip().lower()
    if curve_raw in ("none", "full", "all"):
        curve_raw = "whole_clip"
    if curve_raw not in TEMPORAL_CURVES:
        raise ValidationRepairError(
            f"axes.temporal_curve must be one of {TEMPORAL_CURVES}, got {curve_raw!r}"
        )

    hints = dict(concept_brief.axis_hints) if (concept_brief and concept_brief.axis_hints) else {}
    vis = _as_bool("visual_changed", bool(hints.get("visual_changed", True)))
    aud = _as_bool("audio_changed", bool(hints.get("audio_changed", False)))
    deforms = _as_bool("deforms_ref_silhouette", bool(hints.get("deforms_ref_silhouette", False)))
    pres_step0 = _as_bool("preserve_step0_layout", bool(hints.get("preserve_step0_layout", not deforms)))
    if deforms and pres_step0:
        # Silhouette-deforming edits cannot keep step-0 layout locked while releasing references early.
        pres_step0 = False
    reason = " ".join(str(raw_axes.get("reason") or "").split()).strip()
    return {
        "visual_changed": vis,
        "audio_changed": aud,
        "preserve_step0_layout": pres_step0,
        "deforms_ref_silhouette": deforms,
        "temporal_curve": curve_raw,
        "reason": reason,
    }


def parse_auto_edit_json(raw_text: str, *, concept_brief: ConceptBrief | None = None) -> dict[str, Any]:
    """Parse and normalize `{section_edits | edit_lines, axes}` from a `heylook` reply."""
    cleaned = _strip_think_and_fence(raw_text)
    obj = _extract_balanced_json(cleaned)
    axes = _normalize_axes(obj.get("axes"), concept_brief=concept_brief)
    return {"obj": obj, "axes": axes}


def _resolve_section_key(raw_key: str, original: str, present_names: set[str]) -> str:
    k = str(raw_key or "").strip().rstrip(":").strip().lower()
    if present_names == {"_unsectioned"}:
        if k in _UNSECTIONED_ALIASES:
            return "_unsectioned"
        raise ValidationRepairError(
            f"section {raw_key!r} does not exist in unsectioned prompt; use key '_unsectioned'"
        )
    if k == "description":
        resolved = _pe.resolve_section_name(original, "description")
        if resolved in present_names:
            return resolved
        raise ValidationRepairError(
            f"section {raw_key!r} does not exist in original prompt; valid sections are {sorted(present_names)}"
        )
    mapped = _SECTION_ALIAS_MAP.get(k, k)
    if mapped in present_names:
        return mapped
    raise ValidationRepairError(
        f"section {raw_key!r} does not exist in original prompt; valid sections are {sorted(present_names)}"
    )


def _strip_redundant_section_header(body: str, sec_name: str) -> str:
    s = str(body or "").strip()
    if sec_name != "_unsectioned":
        m = re.match(rf"^\s*{re.escape(sec_name)}\s*:\s*", s, flags=re.IGNORECASE)
        if m:
            s = s[m.end() :].strip()
    return s


def _validate_modified_section_invariants(
    sec_name: str,
    orig_body: str,
    new_body: str,
    *,
    allow_audio_retention_edit: bool,
) -> None:
    if not new_body.strip():
        raise ValidationRepairError(f"section {sec_name!r} was rewritten to an empty string")

    # 1. Exact ordered <d>...</d> dialogue spans
    orig_dlg = _DIALOGUE_SPAN_RE.findall(orig_body)
    new_dlg = _DIALOGUE_SPAN_RE.findall(new_body)
    if orig_dlg != new_dlg:
        raise ValidationRepairError(
            f"section {sec_name!r} altered spoken dialogue <d>...</d>: expected {orig_dlg!r}, got {new_dlg!r}"
        )

    # 2. Exact ordered [Shot N] headers
    orig_shots = _SHOT_HEADER_RE.findall(orig_body)
    new_shots = _SHOT_HEADER_RE.findall(new_body)
    if orig_shots != new_shots:
        raise ValidationRepairError(
            f"section {sec_name!r} altered [Shot N] headers: expected {orig_shots!r}, got {new_shots!r}"
        )

    # 3. Exact ordered (S1)/(S2) speaker IDs
    orig_spk = _spans.SPEAKER_RE.findall(orig_body)
    new_spk = _spans.SPEAKER_RE.findall(new_body)
    if orig_spk != new_spk:
        raise ValidationRepairError(
            f"section {sec_name!r} altered speaker tags: expected {orig_spk!r}, got {new_spk!r}"
        )

    # 4. Exact multiset of <Subject N>, <Picture N>, <Video N>, <Audio N> tags
    orig_tags = Counter(_MEDIA_REF_TAG_RE.findall(orig_body))
    new_tags = Counter(_MEDIA_REF_TAG_RE.findall(new_body))
    if orig_tags != new_tags:
        raise ValidationRepairError(
            f"section {sec_name!r} altered media/subject tags: expected {dict(orig_tags)}, got {dict(new_tags)}"
        )

    # 4b. Exact ordered <|...|> control tags (<|caption_start|>, <|lyrics_start|>, <|cutoff|>, etc.)
    orig_spec = _SPECIAL_TOKEN_RE.findall(orig_body)
    new_spec = _SPECIAL_TOKEN_RE.findall(new_body)
    if orig_spec != new_spec:
        raise ValidationRepairError(
            f"section {sec_name!r} altered special tokens: expected {orig_spec!r}, got {new_spec!r}"
        )

    # 5. Summary leading [task] bracket
    if sec_name == "summary":
        m_o = _TASK_BRACKET_RE.match(orig_body)
        if m_o is not None:
            want = m_o.group(1)
            m_n = _TASK_BRACKET_RE.match(new_body)
            if m_n is None or m_n.group(1) != want:
                raise ValidationRepairError(
                    f"summary must preserve leading task bracket {want!r} verbatim"
                )

    # 6. Retention analysis anchor prefixes, <Audio N> protection, and valid modes
    if sec_name == "retention_analysis":
        orig_lines = [ln.strip() for ln in orig_body.splitlines() if _RETENTION_LINE_RE.match(ln.strip())]
        new_lines = [ln.strip() for ln in new_body.splitlines() if _RETENTION_LINE_RE.match(ln.strip())]
        if len(orig_lines) != len(new_lines):
            raise ValidationRepairError(
                f"retention_analysis line count changed ({len(orig_lines)} -> {len(new_lines)})"
            )
        for o_ln, n_ln in zip(orig_lines, new_lines):
            mo = _RETENTION_LINE_RE.match(o_ln)
            mn = _RETENTION_LINE_RE.match(n_ln)
            assert mo is not None and mn is not None
            if mo.group(1).strip() != mn.group(1).strip():
                raise ValidationRepairError(
                    f"retention_analysis changed anchor prefix {mo.group(1).strip()!r} -> {mn.group(1).strip()!r}"
                )
            if mo.group(2) == "Audio" and not allow_audio_retention_edit and o_ln != n_ln:
                raise ValidationRepairError(
                    f"retention_analysis modified <Audio N> line {o_ln!r}; <Audio N> retention must stay untouched"
                )
            if mn.group(3) not in _RETENTION_MODES:
                raise ValidationRepairError(
                    f"retention_analysis used invalid retention mode {mn.group(3)!r}; "
                    f"expected one of {sorted(_RETENTION_MODES)}"
                )


def _summarize_section_diffs(
    chunks: Sequence[SectionChunk],
    orig_bodies: Mapping[str, str],
    new_bodies: Mapping[str, str],
) -> tuple[list[str], list[str], int, int]:
    """Return `(changed_sections, diff_lines, orig_words, edit_words)`."""
    changed: list[str] = []
    diff_lines: list[str] = []
    orig_w = 0
    edit_w = 0
    for c in chunks:
        ob = orig_bodies[c.name]
        nb = new_bodies[c.name]
        ow = len(ob.split())
        nw = len(nb.split())
        orig_w += ow
        edit_w += nw
        if ob != nb:
            changed.append(c.name)
            diff_lines.append(f"{c.name}: ({ow} -> {nw} words) {ob!r} -> {nb!r}")
    return changed, diff_lines, orig_w, edit_w


def apply_section_edits_validated(
    original: str,
    raw_reply: str,
    *,
    allowed_sections: Sequence[str],
    allow_audio_retention_edit: bool,
    concept_brief: ConceptBrief | None = None,
) -> dict[str, Any]:
    """Validate JSON `section_edits` + `axes`, copy locked sections byte-for-byte, and return result dict."""
    parsed = parse_auto_edit_json(raw_reply, concept_brief=concept_brief)
    obj = parsed["obj"]
    axes = parsed["axes"]
    raw_edits = obj.get("section_edits")
    if not isinstance(raw_edits, Mapping) or not raw_edits:
        raise ValidationRepairError("JSON reply must include a non-empty 'section_edits' object")

    leading, chunks = parse_h3_sections(original)
    present_names = {c.name for c in chunks}
    allowed_set = set(allowed_sections)
    orig_bodies = {c.name: c.body for c in chunks}
    new_bodies = dict(orig_bodies)

    reverted_locked: list[str] = []
    for raw_key, raw_val in raw_edits.items():
        if not isinstance(raw_val, str):
            raise ValidationRepairError(f"section_edits[{raw_key!r}] must be a string")
        sec_name = _resolve_section_key(str(raw_key), original, present_names)
        cleaned_body = _strip_redundant_section_header(raw_val, sec_name)
        if sec_name not in allowed_set:
            if cleaned_body != orig_bodies[sec_name]:
                reverted_locked.append(sec_name)
            continue
        if cleaned_body == orig_bodies[sec_name]:
            continue
        _validate_modified_section_invariants(
            sec_name,
            orig_bodies[sec_name],
            cleaned_body,
            allow_audio_retention_edit=allow_audio_retention_edit,
        )
        new_bodies[sec_name] = cleaned_body

    changed_sections, diff_lines, orig_w, edit_w = _summarize_section_diffs(chunks, orig_bodies, new_bodies)
    if not changed_sections:
        if reverted_locked:
            raise ValidationRepairError(
                f"only locked section(s) {reverted_locked} were changed; edit inside allowed sections {list(allowed_sections)}"
            )
        raise ValidationRepairError("section_edits produced no changes compared to original prompt")

    edited_prompt = reassemble_h3_sections(leading, chunks, new_bodies)
    return {
        "edited_prompt": edited_prompt,
        "axes": axes,
        "changed_sections": tuple(changed_sections),
        "edit_lines_text": "\n".join(diff_lines),
        "reverted_locked": tuple(reverted_locked),
        "orig_words": orig_w,
        "edit_words": edit_w,
    }


def _clean_edit_line(raw: str) -> str:
    s = str(raw or "").strip()
    s = re.sub(r"^(?:[-*•]+|\d+[.)])\s+", "", s).strip()
    return s


def _section_at_offset(original: str, offset: int) -> str:
    matches = list(_SECTION_LINE_RE.finditer(original))
    if not matches:
        return "_unsectioned"
    cur = "_unsectioned"
    for m in matches:
        if m.start() <= offset:
            cur = m.group(1)
        else:
            break
    return cur


def apply_edit_lines_validated(
    original: str,
    raw_reply: str,
    *,
    allowed_sections: Sequence[str],
    allow_audio_retention_edit: bool,
    concept_brief: ConceptBrief | None = None,
) -> dict[str, Any]:
    """Validate JSON `edit_lines` + `axes` through `h3_prompt_edit.apply_edits` with strict scope guards."""
    parsed = parse_auto_edit_json(raw_reply, concept_brief=concept_brief)
    obj = parsed["obj"]
    axes = parsed["axes"]
    raw_lines = obj.get("edit_lines")
    if isinstance(raw_lines, list):
        cleaned = [_clean_edit_line(ln) for ln in raw_lines if _clean_edit_line(ln)]
    elif isinstance(raw_lines, str):
        cleaned = [_clean_edit_line(ln) for ln in raw_lines.splitlines() if _clean_edit_line(ln)]
    else:
        raise ValidationRepairError("JSON reply must include 'edit_lines' as a non-empty list of 'old -> new' strings")

    if not cleaned:
        raise ValidationRepairError("edit_lines list is empty; emit at least one 'old -> new' line")

    leading, chunks = parse_h3_sections(original)
    present_names = {c.name for c in chunks}
    allowed_set = set(allowed_sections)

    for line in cleaned:
        _pfx, sec_name, body = _pe.parse_section_prefix(line)
        try:
            old_part, _new_part = _pe.split_pole_pair(body)
        except ValueError as exc:
            raise ValidationRepairError(str(exc)) from exc
        if sec_name is not None:
            if present_names == {"_unsectioned"}:
                raise ValidationRepairError(
                    f"edit {line!r} specifies section {sec_name!r}, but original prompt has no section headers"
                )
            resolved_sec = _pe.resolve_section_name(original, sec_name)
            if resolved_sec not in present_names:
                raise ValidationRepairError(
                    f"edit {line!r} targets missing section {resolved_sec!r}; valid sections are {sorted(present_names)}"
                )
            if resolved_sec not in allowed_set:
                raise ValidationRepairError(
                    f"edit {line!r} targets locked section {resolved_sec!r}; allowed sections are {list(allowed_sections)}"
                )
        elif present_names != {"_unsectioned"} and old_part and original.count(old_part) == 1:
            idx = original.index(old_part)
            hit_sec = _section_at_offset(original, idx)
            if hit_sec not in allowed_set:
                raise ValidationRepairError(
                    f"edit {line!r} hits locked section {hit_sec!r}; allowed sections are {list(allowed_sections)}"
                )

    try:
        edited_prompt, _records = _pe.apply_edits(original, "\n".join(cleaned))
    except ValueError as exc:
        raise ValidationRepairError(str(exc)) from exc

    if edited_prompt.strip() == original.strip():
        raise ValidationRepairError("edit_lines produced no changes compared to original prompt")

    _lead2, edited_chunks = parse_h3_sections(edited_prompt)
    orig_bodies = {c.name: c.body for c in chunks}
    new_bodies = {c.name: c.body for c in edited_chunks}
    for c in chunks:
        if orig_bodies[c.name] != new_bodies[c.name]:
            if c.name not in allowed_set:
                raise ValidationRepairError(
                    f"edit modified locked section {c.name!r}; allowed sections are {list(allowed_sections)}"
                )
            _validate_modified_section_invariants(
                c.name,
                orig_bodies[c.name],
                new_bodies[c.name],
                allow_audio_retention_edit=allow_audio_retention_edit,
            )

    changed_sections, _diff_lines, orig_w, edit_w = _summarize_section_diffs(chunks, orig_bodies, new_bodies)
    return {
        "edited_prompt": edited_prompt,
        "axes": axes,
        "changed_sections": tuple(changed_sections),
        "edit_lines_text": "\n".join(cleaned),
        "reverted_locked": (),
        "orig_words": orig_w,
        "edit_words": edit_w,
    }


def compile_edit_plan(
    *,
    axes: Mapping[str, Any],
    changed_sections: Sequence[str],
    take_mode: str,
    timing: str,
    has_references_or_ref2va: bool,
    is_sectioned: bool,
    concept_brief: ConceptBrief | None = None,
) -> EditPlan:
    """Compile validated 4-axis classification + user `take_mode`/`timing` into `EditPlan`."""
    if take_mode not in TAKE_MODES:
        raise ValueError(f"unknown take_mode {take_mode!r}; expected one of {TAKE_MODES}")
    if timing not in TIMING_MODES:
        raise ValueError(f"unknown timing {timing!r}; expected one of {TIMING_MODES}")

    changed_set = set(changed_sections)
    if is_sectioned:
        visual_changed = bool(changed_set & _VISUAL_SECTIONS)
        audio_changed = bool(changed_set & _AUDIO_SECTIONS)
    else:
        visual_changed = bool(axes.get("visual_changed", True))
        audio_changed = bool(axes.get("audio_changed", False))

    deforms = bool(axes.get("deforms_ref_silhouette", False))
    preserve_step0 = bool(axes.get("preserve_step0_layout", not deforms))
    reason = str(axes.get("reason") or "").strip()

    eff_timing = timing
    if eff_timing == "auto":
        if concept_brief and concept_brief.timing_hint and concept_brief.timing_hint != "auto":
            eff_timing = concept_brief.timing_hint
        else:
            eff_timing = str(axes.get("temporal_curve") or "whole_clip")
    if eff_timing not in TEMPORAL_CURVES:
        eff_timing = "whole_clip"

    ref_end_percent = 0.33 if (has_references_or_ref2va and deforms) else 1.0

    if take_mode == "manual_widgets":
        return EditPlan(
            first_step=0,
            last_step=1000,
            streams="both",
            audio_scale=1.0,
            ref_end_percent=ref_end_percent,
            temporal_curve=eff_timing,
            force_scale=None,
            manual_widgets=True,
            visual_changed=visual_changed,
            audio_changed=audio_changed,
            preserve_step0_layout=preserve_step0,
            deforms_ref_silhouette=deforms,
            reason=reason or "manual widgets override",
        )

    if take_mode == "fast_1x":
        return EditPlan(
            first_step=0,
            last_step=3,
            streams="both",
            audio_scale=1.0,
            ref_end_percent=ref_end_percent,
            temporal_curve="whole_clip",
            force_scale=1.0,
            manual_widgets=False,
            visual_changed=visual_changed,
            audio_changed=audio_changed,
            preserve_step0_layout=False,
            deforms_ref_silhouette=deforms,
            reason=reason or "fast 1x single-pass rewrite",
        )

    if not visual_changed and audio_changed:
        streams = "audio_only"
        last_step = 4
    else:
        streams = "independent"
        last_step = 3

    if take_mode == "lock_take":
        first_step = 2 if (streams == "audio_only" and not has_references_or_ref2va) else 1
    elif take_mode == "allow_reshoot":
        first_step = 0
    else:  # "auto"
        if deforms or not preserve_step0:
            first_step = 0
        elif streams == "audio_only" and not has_references_or_ref2va:
            first_step = 2
        else:
            first_step = 1

    return EditPlan(
        first_step=first_step,
        last_step=last_step,
        streams=streams,
        audio_scale=1.0,
        ref_end_percent=ref_end_percent,
        temporal_curve=eff_timing,
        force_scale=None,
        manual_widgets=False,
        visual_changed=visual_changed,
        audio_changed=audio_changed,
        preserve_step0_layout=preserve_step0,
        deforms_ref_silhouette=deforms,
        reason=reason,
    )


def build_auto_edit_user_prompt(
    *,
    original: str,
    allowed_sections: Sequence[str],
    allow_retention_relax: bool,
    rewrite_mode: str,
    concept_brief: ConceptBrief | None,
    edit_intent: str,
    has_images: bool,
) -> str:
    """Assemble the APC-friendly user turn (`Original prompt` first, then concept/intent)."""
    _lead, chunks = parse_h3_sections(original)
    present = [c.name for c in chunks]
    parts: list[str] = [
        "Original prompt:\n<<<\n" + original + "\n>>>",
        f"Present sections in original: {', '.join(present)}",
        f"Allowed sections to edit: {', '.join(allowed_sections)}",
        f"Retention relaxation allowed: {'yes' if ('retention_analysis' in allowed_sections and allow_retention_relax) else 'no'}",
    ]
    if has_images:
        parts.append("Reference images attached: inspect <Picture 1>.. accurately for retained vs. replaced details.")

    if concept_brief is not None:
        c_lines = [f"Selected concept ({concept_brief.title}): {concept_brief.summary}"]
        if concept_brief.a_side and concept_brief.b_side:
            c_lines.append(f"Transition: {concept_brief.a_side} -> {concept_brief.b_side}")
        if concept_brief.notes:
            c_lines.append(f"Notes: {concept_brief.notes}")
        if concept_brief.avoid:
            c_lines.append(f"Avoid: {concept_brief.avoid}")
        if concept_brief.axis_hints:
            hint_parts = [f"{k}={json.dumps(concept_brief.axis_hints[k])}" for k in sorted(concept_brief.axis_hints)]
            c_lines.append(
                "Concept default axis hints (follow unless user edit instruction contradicts): "
                + ", ".join(hint_parts)
            )
            if concept_brief.axis_hints.get("deforms_ref_silhouette") is True:
                c_lines.append(
                    "Silhouette/medium overhaul active: translate photorealistic subject surfaces, "
                    "props, environment textures, and motion phrasing across allowed visual sections "
                    "into the target medium's physical materials and cadence while keeping <Picture N> identity."
                )
        if concept_brief.timing_hint and concept_brief.timing_hint != "auto":
            c_lines.append(f"Concept default temporal_curve: {concept_brief.timing_hint}")
        if concept_brief.exemplars:
            c_lines.append("Reference examples:")
            for idx, (sec, a_ex, b_ex) in enumerate(concept_brief.exemplars, start=1):
                c_lines.append(f"  Example {idx} [{sec}]: {a_ex!r} -> {b_ex!r}")
        parts.append("\n".join(c_lines))

    intent_clean = (edit_intent or "").strip()
    if intent_clean:
        parts.append(f"User edit instruction:\n{intent_clean}")

    if rewrite_mode == "section_edits":
        parts.append(
            "Task: Return ONLY the JSON object `{\"section_edits\": {...}, \"axes\": {...}}`. "
            "Include ONLY allowed sections that actually change, keeping all tags, dialogue <d>...</d>, "
            "speaker IDs, and [Shot N] headers verbatim."
        )
    else:
        parts.append(
            "Task: Return ONLY the JSON object `{\"edit_lines\": [...], \"axes\": {...}}`. "
            "Each line must match an exact unique phrase inside an allowed section of original."
        )
    return "\n\n".join(parts)


def _detect_has_references(original: str, prepared_images: Sequence[tuple[str, str]]) -> bool:
    if prepared_images:
        return True
    if _MEDIA_REF_TAG_RE.search(original):
        return True
    _lead, chunks = parse_h3_sections(original)
    names = {c.name for c in chunks}
    return bool(names & {"subject_definitions", "retention_analysis"})


def run_auto_edit_rewrite(
    *,
    original: str,
    concept: str = "none",
    edit_intent: str = "",
    take_mode: str = "auto",
    timing: str = "auto",
    enabled: bool = True,
    base_url: str = "",
    model: str = "",
    thinking: str = "on",
    reasoning_effort: str = "medium",
    seed: int = 0,
    images: Any = None,
    target_scope: str = "auto",
    rewrite_mode: str = "section_edits",
    allow_retention_relax: bool = True,
    exemplar_count: int = 3,
    preset: str = "",
    sampler_mode: str = "heylook / preset default",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    custom_system_prompt: str = "",
    prompt_layout: str = "system_and_user",
    concepts_dir: Path | None = None,
    data_cv_dir: Path | None = None,
    bank_cv_dir: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Execute or cache-hit one `MiniMaxH3AutoEditPrompt` rewrite and compile its `EditPlan`."""
    orig_norm = _rules.normalize_prompt(original)
    if not orig_norm:
        raise ValueError("MiniMaxH3AutoEditPrompt: 'original' prompt is empty")
    if take_mode not in TAKE_MODES:
        raise ValueError(f"unknown take_mode {take_mode!r}; expected one of {TAKE_MODES}")
    if timing not in TIMING_MODES:
        raise ValueError(f"unknown timing {timing!r}; expected one of {TIMING_MODES}")
    if rewrite_mode not in REWRITE_MODES:
        raise ValueError(f"unknown rewrite_mode {rewrite_mode!r}; expected one of {REWRITE_MODES}")
    if prompt_layout not in PROMPT_LAYOUTS:
        raise ValueError(f"unknown prompt_layout {prompt_layout!r}; expected one of {PROMPT_LAYOUTS}")

    if not enabled:
        disabled_plan = EditPlan(
            first_step=0,
            last_step=1000,
            streams="both",
            audio_scale=1.0,
            ref_end_percent=1.0,
            temporal_curve="whole_clip",
            force_scale=None,
            manual_widgets=True,
            visual_changed=False,
            audio_changed=False,
            preserve_step0_layout=True,
            deforms_ref_silhouette=False,
            reason="disabled passthrough",
        )
        return {
            "original_prompt": orig_norm,
            "edited_prompt": orig_norm,
            "edit_plan": disabled_plan,
            "edit_lines": "",
            "thinking": "",
            "report": "auto-edit disabled (passthrough)",
            "cache_hit": False,
            "repair_used": False,
            "resolved_scope": "passthrough",
            "stream_result": None,
        }

    concept_name = (concept or "none").strip() or "none"
    intent_clean = (edit_intent or "").strip()
    if concept_name.lower() == "none" and not intent_clean:
        raise ValueError("MiniMaxH3AutoEditPrompt: choose a 'concept' or enter an 'edit_intent'")

    eff_exemplars = max(1, int(exemplar_count))
    concept_brief = (
        None
        if concept_name.lower() == "none"
        else load_concept_brief(
            concept_name,
            exemplar_count=eff_exemplars,
            concepts_dir=concepts_dir,
            data_cv_dir=data_cv_dir,
            bank_cv_dir=bank_cv_dir,
        )
    )

    resolved_scope, allowed_sections = resolve_allowed_sections(
        orig_norm,
        target_scope,
        allow_retention_relax=bool(allow_retention_relax),
        concept_brief=concept_brief,
        has_edit_intent=bool(intent_clean),
    )

    names = sorted(images or {}, key=lambda nm: int(nm.rsplit("_", 1)[-1]))
    raw_images = [images[nm] for nm in names if images[nm] is not None]
    if len(raw_images) > MAX_AUTO_EDIT_IMAGES:
        raise ValueError(
            f"MiniMaxH3AutoEditPrompt accepts at most {MAX_AUTO_EDIT_IMAGES} images, got {len(raw_images)}"
        )
    prepared_images = [prepare_heylook_tensor(t, index=i) for i, t in enumerate(raw_images, start=1)]
    image_hashes = tuple(hashlib.sha256(b64.encode("ascii")).hexdigest()[:16] for b64, _ in prepared_images)

    has_refs = _detect_has_references(orig_norm, prepared_images)
    _lead, orig_chunks = parse_h3_sections(orig_norm)
    is_sectioned = not (len(orig_chunks) == 1 and orig_chunks[0].name == "_unsectioned")

    base = resolve_heylook_base_url(base_url)
    preset_name = (preset or "").strip()
    req_seed = int(seed) & 0x7FFFFFFF
    eff_effort = (reasoning_effort or "").strip()
    sampler_tuple = (
        (float(temperature), int(max_tokens))
        if sampler_mode == "custom"
        else ()
    )
    cache_key = (
        base,
        (model or "").strip(),
        orig_norm,
        concept_brief.file_stat if concept_brief else None,
        intent_clean,
        resolved_scope,
        allowed_sections,
        rewrite_mode,
        bool(allow_retention_relax),
        eff_exemplars,
        image_hashes,
        thinking,
        eff_effort,
        req_seed,
        preset_name,
        sampler_mode,
        sampler_tuple,
        (custom_system_prompt or "").strip(),
        prompt_layout,
    )

    cached = _REWRITE_CACHE.get(cache_key) if use_cache else None
    if cached is not None:
        _REWRITE_CACHE.move_to_end(cache_key)
        plan = compile_edit_plan(
            axes=cached["axes"],
            changed_sections=cached["changed_sections"],
            take_mode=take_mode,
            timing=timing,
            has_references_or_ref2va=has_refs,
            is_sectioned=is_sectioned,
            concept_brief=concept_brief,
        )
        return {
            **cached,
            "edit_plan": plan,
            "cache_hit": True,
            "report": f"{cached['report_base']} | take={take_mode}(step0={plan.first_step},streams={plan.streams},rel={plan.ref_end_percent:g},timing={plan.temporal_curve}) [cached]",
        }

    base_sys_prompt = (custom_system_prompt or "").strip() or _read_system_prompt_file(rewrite_mode)
    user_prompt = build_auto_edit_user_prompt(
        original=orig_norm,
        allowed_sections=allowed_sections,
        allow_retention_relax=bool(allow_retention_relax),
        rewrite_mode=rewrite_mode,
        concept_brief=concept_brief,
        edit_intent=intent_clean,
        has_images=bool(prepared_images),
    )
    if prompt_layout == "user_only" and not (custom_system_prompt or "").strip():
        sys_prompt = ""
        final_user_prompt = f"{base_sys_prompt}\n\n---\n\n{user_prompt}"
    else:
        sys_prompt = base_sys_prompt
        final_user_prompt = user_prompt

    need_caps = frozenset({"chat", "vision"} if prepared_images else {"chat"})
    allow_audio_ret = target_scope == "all_sections"
    validator_fn = (
        apply_section_edits_validated if rewrite_mode == "section_edits" else apply_edit_lines_validated
    )

    repair_used = False
    thinking_parts: list[str] = []
    req_effort = "" if thinking == "off" else eff_effort
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(None, connect=10.0)) as client:
        model_row = discover_heylook_model(client, base, model, need=need_caps)
        preset_fields = (
            expand_heylook_preset(client, base, preset_name, model_row)
            if preset_name
            else None
        )
        sampler_overrides = (
            {"temperature": float(temperature), "max_tokens": int(max_tokens)}
            if sampler_mode == "custom"
            else None
        )
        body = build_heylook_request_body(
            model_row=model_row,
            system_prompt=sys_prompt,
            user_prompt=final_user_prompt,
            prepared_images=prepared_images,
            thinking=thinking,
            request_seed=req_seed,
            preset_fields=preset_fields,
            sampler_mode=sampler_mode,
            sampler_overrides=sampler_overrides,
            reasoning_effort_override=req_effort,
            default_reasoning_effort=req_effort,
        )
        res: HeylookStreamResult = with_heylook_backoff(
            lambda: stream_heylook_message(base, body, client=client, interrupt_check=_comfy_interrupt_check),
            interrupt_check=_comfy_interrupt_check,
        )
        t1 = res.thinking or _extract_inline_thinking(res.text)
        if t1:
            thinking_parts.append(t1)

        try:
            validated = validator_fn(
                orig_norm,
                res.text,
                allowed_sections=allowed_sections,
                allow_audio_retention_edit=allow_audio_ret,
                concept_brief=concept_brief,
            )
        except ValidationRepairError as first_err:
            if res.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"heylook stopped at max_tokens before completing a valid rewrite ({first_err}). "
                    "Raise max_tokens or switch thinking to 'off' / lower reasoning_effort."
                ) from first_err
            repair_used = True
            valid_headers_note = ", ".join(c.name for c in orig_chunks)
            repair_msg = (
                f"Your JSON output failed validation: {first_err}\n"
                f"Valid section names in original: {valid_headers_note}\n"
                f"Allowed sections to edit: {', '.join(allowed_sections)}\n"
                "Fix ONLY this issue and output ONLY the corrected single valid JSON object."
            )
            repair_body = build_heylook_request_body(
                model_row=model_row,
                system_prompt=sys_prompt,
                user_prompt=final_user_prompt,
                prepared_images=prepared_images,
                thinking=thinking,
                request_seed=req_seed,
                preset_fields=preset_fields,
                sampler_mode=sampler_mode,
                sampler_overrides=sampler_overrides,
                reasoning_effort_override=req_effort,
                default_reasoning_effort=req_effort,
                extra_messages=[
                    {"role": "assistant", "content": res.text},
                    {"role": "user", "content": repair_msg},
                ],
            )
            res = with_heylook_backoff(
                lambda: stream_heylook_message(
                    base, repair_body, client=client, interrupt_check=_comfy_interrupt_check
                ),
                interrupt_check=_comfy_interrupt_check,
            )
            t2 = res.thinking or _extract_inline_thinking(res.text)
            if t2:
                thinking_parts.append(t2)
            try:
                validated = validator_fn(
                    orig_norm,
                    res.text,
                    allowed_sections=allowed_sections,
                    allow_audio_retention_edit=allow_audio_ret,
                    concept_brief=concept_brief,
                )
            except ValidationRepairError as second_err:
                raise ValueError(
                    f"MiniMaxH3AutoEditPrompt validation failed after 1 repair attempt: {second_err}"
                ) from second_err

    telem = format_heylook_report(res, resolved_mode=f"{rewrite_mode}/{resolved_scope}")
    delta_words = validated["edit_words"] - validated["orig_words"]
    delta_sign = "+" if delta_words >= 0 else ""
    report_base = (
        f"{telem} words={validated['orig_words']}->{validated['edit_words']}({delta_sign}{delta_words}) "
        f"changed={','.join(validated['changed_sections'])}"
        + (" [repaired]" if repair_used else "")
        + (f" reverted={','.join(validated['reverted_locked'])}" if validated["reverted_locked"] else "")
    )
    think_trace = "\n\n---\n\n".join(thinking_parts)

    cache_entry = {
        "original_prompt": orig_norm,
        "edited_prompt": validated["edited_prompt"],
        "axes": validated["axes"],
        "changed_sections": validated["changed_sections"],
        "edit_lines": validated["edit_lines_text"],
        "thinking": think_trace,
        "report_base": report_base,
        "repair_used": repair_used,
        "resolved_scope": resolved_scope,
        "stream_result": res,
        "orig_words": validated["orig_words"],
        "edit_words": validated["edit_words"],
    }
    if use_cache:
        _REWRITE_CACHE[cache_key] = cache_entry
        while len(_REWRITE_CACHE) > _CACHE_MAXSIZE:
            _REWRITE_CACHE.popitem(last=False)

    plan = compile_edit_plan(
        axes=validated["axes"],
        changed_sections=validated["changed_sections"],
        take_mode=take_mode,
        timing=timing,
        has_references_or_ref2va=has_refs,
        is_sectioned=is_sectioned,
        concept_brief=concept_brief,
    )
    full_report = (
        f"{report_base} | take={take_mode}(step0={plan.first_step},streams={plan.streams},"
        f"rel={plan.ref_end_percent:g},timing={plan.temporal_curve})"
    )
    return {
        **cache_entry,
        "edit_plan": plan,
        "cache_hit": False,
        "report": full_report,
    }


class MiniMaxH3AutoEditPrompt(io.ComfyNode):
    """Auto-rewrite an H3 prompt via local/LAN `heylook` and emit a 4-axis `edit_plan` for `EditGuider`."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AutoEditPrompt",
            display_name="MiniMax H3 Auto Edit Prompt (heylook)",
            category="conditioning/minimax",
            description=(
                "Rewrites an H3 video or image prompt from a pre-created concept card (.toml) and/or "
                "free-text edit_intent via a local or LAN heylookitsanllm server (/v1/messages), "
                "enforcing deterministic tag/dialogue/section guardrails and emitting a 4-axis edit_plan "
                "wire for MiniMaxH3EditGuider and MiniMaxH3ReleaseReferences."
            ),
            inputs=[
                io.String.Input(
                    "original",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Base prompt as written. Wire from the same text node feeding the original "
                        "conditioning node so original and edited branches stay synchronized."
                    ),
                ),
                io.Combo.Input(
                    "concept",
                    options=list_concept_options(),
                    default="none",
                    tooltip=(
                        "Optional pre-created concept (.toml under data/edit_concepts/, "
                        "data/control_vectors/, or prompt_bank/control_vectors/). Can be combined "
                        "with edit_intent below, or left 'none' when using edit_intent alone."
                    ),
                ),
                io.String.Input(
                    "edit_intent",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Optional free-text editing instruction (used alone or on top of 'concept')."
                    ),
                ),
                io.Combo.Input(
                    "take_mode",
                    options=list(TAKE_MODES),
                    default="auto",
                    tooltip=(
                        "'auto' lets the 4-axis classifier set first_step, streams, and reference "
                        "release; 'fast_1x' forces a 1-pass (scale=1.0) EditGuider shortcut; "
                        "'lock_take' pins step-0 actor/layout (first_step=1/2); 'allow_reshoot' "
                        "allows full step-0 restaging; 'manual_widgets' ignores edit_plan overrides."
                    ),
                ),
                io.Combo.Input(
                    "timing",
                    options=list(TIMING_MODES),
                    default="auto",
                    tooltip=(
                        "Temporal envelope ([F, 1, 1]) applied inside MiniMaxH3EditGuider: 'auto' uses "
                        "the classified curve ('whole_clip', 'ramp_in', 'sudden_midclip', 'second_half'); "
                        "auto-collapses to no-op on single-frame stills."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Call heylook to rewrite 'original'. Turn off for an offline passthrough with "
                        "zero network I/O."
                    ),
                ),
                io.String.Input(
                    "base_url",
                    default="",
                    tooltip=(
                        "Optional heylook server URL override. Leave blank to read H3_HEYLOOK_URL from "
                        "your environment / gitignored h3.env, falling back to http://127.0.0.1:8080."
                    ),
                ),
                io.String.Input(
                    "model",
                    default="",
                    tooltip=(
                        "Optional model id from GET /v1/models. Leave blank to auto-select a served "
                        "'chat' model (plus 'vision' when images are wired), preferring resident models."
                    ),
                ),
                io.Combo.Input(
                    "thinking",
                    options=list(THINKING_MODES),
                    default="on",
                    tooltip=(
                        "Thinking switch on models advertising 'thinking'. Thinking blocks are routed "
                        "to the 'thinking' output and never reach H3 conditioning."
                    ),
                ),
                io.String.Input(
                    "reasoning_effort",
                    default="medium",
                    tooltip=(
                        "Reasoning depth vocabulary word (e.g. 'medium' on Qwen3.8). Automatically "
                        "capability-gated and omitted on models without depth support (e.g. Gemma-4)."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Seed forwarded to heylook (when supported) and ComfyUI node caching.",
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, MAX_AUTO_EDIT_IMAGES + 1)],
                        min=0,
                    ),
                    tooltip=(
                        "Optional reference stills (<Picture 1..4>) for vision-grounded rewrites on "
                        "ref2va/i2va (downscaled client-side to <= 2048 px)."
                    ),
                ),
                io.Combo.Input(
                    "target_scope",
                    options=list(TARGET_SCOPES),
                    default="auto",
                    optional=True,
                    tooltip=(
                        "Which sections heylook may modify; unselected sections are copied "
                        "byte-for-byte from original."
                    ),
                ),
                io.Combo.Input(
                    "rewrite_mode",
                    options=list(REWRITE_MODES),
                    default="section_edits",
                    optional=True,
                    tooltip=(
                        "'section_edits' (default, handles sparse and dense prompts) rewrites only "
                        "changed sections in JSON; 'edit_lines' emits old -> new lines validated via "
                        "MiniMaxH3PromptEdit."
                    ),
                ),
                io.Boolean.Input(
                    "allow_retention_relax",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Allow relaxing retention_analysis from fully_preserved to partially_preserved "
                        "when visual appearance/wardrobe/medium changes on ref2va."
                    ),
                ),
                io.Int.Input(
                    "exemplar_count",
                    default=3,
                    min=1,
                    max=6,
                    optional=True,
                    tooltip="Number of few-shot [[pair]] / [[example]] items included from 'concept'.",
                ),
                io.String.Input(
                    "preset",
                    default="",
                    optional=True,
                    tooltip="Optional preset name or id from GET /v1/presets on heylook.",
                ),
                io.Combo.Input(
                    "sampler_mode",
                    options=list(SAMPLER_MODES),
                    default="heylook / preset default",
                    optional=True,
                    tooltip="'heylook / preset default' uses server cascade; 'custom' sends temperature and max_tokens.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip="Sampling temperature sent when sampler_mode is 'custom'.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=4096,
                    min=64,
                    max=32768,
                    step=64,
                    optional=True,
                    tooltip="Maximum output tokens sent when sampler_mode is 'custom'.",
                ),
                io.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Optional editable override for the auto-edit system prompt.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="original_prompt"),
                io.String.Output(display_name="edited_prompt"),
                H3EditPlan.Output(display_name="edit_plan"),
                io.String.Output(display_name="edit_lines"),
                io.String.Output(display_name="thinking"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original: str = "",
        concept: str = "none",
        edit_intent: str = "",
        take_mode: str = "auto",
        timing: str = "auto",
        enabled: bool = True,
        base_url: str = "",
        model: str = "",
        thinking: str = "on",
        reasoning_effort: str = "medium",
        seed: int = 0,
        images: io.Autogrow.Type = None,
        target_scope: str = "auto",
        rewrite_mode: str = "section_edits",
        allow_retention_relax: bool = True,
        exemplar_count: int = 3,
        preset: str = "",
        sampler_mode: str = "heylook / preset default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        custom_system_prompt: str = "",
    ) -> io.NodeOutput:
        res = run_auto_edit_rewrite(
            original=original,
            concept=concept,
            edit_intent=edit_intent,
            take_mode=take_mode,
            timing=timing,
            enabled=enabled,
            base_url=base_url,
            model=model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            seed=seed,
            images=images,
            target_scope=target_scope,
            rewrite_mode=rewrite_mode,
            allow_retention_relax=allow_retention_relax,
            exemplar_count=exemplar_count,
            preset=preset,
            sampler_mode=sampler_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_system_prompt=custom_system_prompt,
        )
        return io.NodeOutput(
            res["original_prompt"],
            res["edited_prompt"],
            res["edit_plan"],
            res["edit_lines"],
            res["thinking"],
            res["report"],
        )


def build_auto_edit_catalog(base_url: str = "") -> dict[str, Any]:
    """Return concept stems plus best-effort live `heylook` models & presets for UI autofill."""
    return {
        "concepts": list_concept_options(),
        "heylook": probe_heylook_server_info(base_url, fallback_env=True),
    }


def register_auto_edit_routes() -> None:
    """Register `GET /h3_auto_edit/catalog` on ComfyUI's `PromptServer` when running inside ComfyUI."""
    try:
        import asyncio
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    async def auto_edit_catalog(request: web.Request) -> web.Response:
        base_url = request.query.get("base_url", "")
        payload = await asyncio.to_thread(build_auto_edit_catalog, base_url)
        return web.json_response(payload)

    instance.routes.get("/h3_auto_edit/catalog")(auto_edit_catalog)
