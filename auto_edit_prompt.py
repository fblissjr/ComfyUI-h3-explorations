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
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Callable, Mapping, Sequence

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
    from . import h3_prompt_contract as _pc  # noqa: E402
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
        compose_preset_system_prompt,
        discover_heylook_model,
        expand_heylook_preset,
        format_heylook_report,
        make_heylook_client,
        parse_extend_directive,
        prepare_heylook_tensor,
        probe_heylook_server_info,
        resolve_heylook_base_url,
        stream_heylook_message,
        with_heylook_backoff,
    )
except ImportError:  # run from the repo root by a check or script
    import control_vectors as _cv  # noqa: E402
    import h3_prompt_contract as _pc  # noqa: E402
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
        compose_preset_system_prompt,
        discover_heylook_model,
        expand_heylook_preset,
        format_heylook_report,
        make_heylook_client,
        parse_extend_directive,
        prepare_heylook_tensor,
        probe_heylook_server_info,
        resolve_heylook_base_url,
        stream_heylook_message,
        with_heylook_backoff,
    )


_HERE = Path(__file__).resolve().parent
_SYSTEM_PROMPTS_DIR = _HERE / "docs" / "wiki" / "system_prompts" / "h3_auto_edit"
_VIDEO_SYSTEM_PROMPTS_DIR = _HERE / "docs" / "wiki" / "system_prompts" / "h3_video"
_IMAGE_SYSTEM_PROMPTS_DIR = _HERE / "docs" / "wiki" / "system_prompts" / "h3_image"
PROMPT_BANK_CV_DIR = _HERE / "prompt_bank" / "control_vectors"
DATA_CV_DIR = _HERE / "data" / "control_vectors"
DATA_CONCEPTS_DIR = _HERE / "data" / "edit_concepts"

H3EditPlan = io.Custom("MINIMAX_H3_EDIT_PLAN")
H3References = io.Custom("MINIMAX_H3_REFERENCES")

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
    auto_fast_1x: bool = False
    video_end_percent: float = 1.0


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
    if rewrite_mode == "fresh":
        fname = "fresh_prompt_system_prompt.md"
    elif rewrite_mode == "section_edits":
        fname = "section_edits_system_prompt.md"
    else:
        fname = "edit_lines_system_prompt.md"
    return (_SYSTEM_PROMPTS_DIR / fname).read_text(encoding="utf-8").strip()


def resolve_fresh_system_prompt(resolved_mode: str, *, has_images: bool = False) -> str:
    """Return the standalone zero-lore system prompt for a resolved fresh-generation mode."""
    m = str(resolved_mode or "").strip().lower()
    if m in ("ref2va", "t2va", "i2va", "fl2va", "l2va"):
        return (_VIDEO_SYSTEM_PROMPTS_DIR / f"{m}_system_prompt.md").read_text(encoding="utf-8").strip()
    if m == "single_frame":
        fname = "edit_system_prompt.md" if has_images else "t2i_system_prompt.md"
        return (_IMAGE_SYSTEM_PROMPTS_DIR / fname).read_text(encoding="utf-8").strip()
    return _read_system_prompt_file("fresh")


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
    for _dir_kind, d in _concept_search_dirs(
        concepts_dir=concepts_dir, data_cv_dir=data_cv_dir, bank_cv_dir=bank_cv_dir
    ):
        p = d / fname
        if p.is_file():
            resolved = p.resolve()
            try:
                doc = tomllib.loads(resolved.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"{resolved}: invalid TOML syntax: {exc}") from exc
            kind = "edit_concept" if "concept" in doc else "control_vector"
            return kind, resolved
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
    """Parse and normalize `{section_edits | edit_lines, axes}` from a JSON `heylook` reply."""
    cleaned = _strip_think_and_fence(raw_text)
    obj = _extract_balanced_json(cleaned)
    axes = _normalize_axes(obj.get("axes"), concept_brief=concept_brief)
    return {"obj": obj, "axes": axes}


_MD_CONTAINER_RE = re.compile(r"^#{1,4}\s*section_edits\s*:?\s*$", re.IGNORECASE)
_MD_EDIT_LINES_RE = re.compile(r"^#{1,4}\s*edit_lines\s*:?\s*$", re.IGNORECASE)
_MD_AXES_RE = re.compile(r"^#{1,4}\s*axes\s*:?\s*$", re.IGNORECASE)
_MD_SECTION_RE = re.compile(
    r"^#{1,4}\s*(?:section\s*:\s*(?P<named>[a-zA-Z0-9_ :]+?)|"
    r"(?P<bare>subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|overall_soundscape|non_diegetic_music|_unsectioned))\s*:?\s*$",
    re.IGNORECASE,
)


def _parse_markdown_axes_block(lines: Sequence[str]) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^(?:[-*•]+|\d+[.)])\s+", "", s).strip()
        if ":" not in s:
            continue
        raw_k, raw_v = s.split(":", 1)
        key = raw_k.strip().strip("`*_ ").lower()
        val_s = raw_v.strip()
        if len(val_s) >= 2 and val_s[0] == val_s[-1] and val_s[0] in ("`", '"', "'"):
            val_s = val_s[1:-1].strip()
        if key in ("visual_changed", "audio_changed", "preserve_step0_layout", "deforms_ref_silhouette"):
            low = val_s.lower()
            if low in ("true", "yes", "1"):
                axes[key] = True
            elif low in ("false", "no", "0"):
                axes[key] = False
            else:
                axes[key] = val_s
        elif key in ("temporal_curve", "reason"):
            axes[key] = val_s
    return axes


def _extract_markdown_reply(text: str) -> dict[str, Any] | None:
    """Parse `## section: <name>` / `## edit_lines` + `## axes` Markdown blocks, or return `None`."""
    section_edits: dict[str, str] = {}
    edit_lines: list[str] = []
    axes_lines: list[str] = []
    cur_kind: str | None = None
    cur_name: str = ""
    cur_buf: list[str] = []
    saw_heading = False

    def _flush() -> None:
        nonlocal cur_kind, cur_name, cur_buf
        if cur_kind == "section" and cur_name:
            body = "\n".join(cur_buf).strip()
            if body:
                section_edits[cur_name] = body
        elif cur_kind == "edit_lines":
            for ln in cur_buf:
                cln = _clean_edit_line(ln)
                if cln:
                    edit_lines.append(cln)
        elif cur_kind == "axes":
            axes_lines.extend(cur_buf)
        cur_kind = None
        cur_name = ""
        cur_buf = []

    for line in text.splitlines():
        st = line.strip()
        if _MD_CONTAINER_RE.match(st):
            _flush()
            saw_heading = True
            continue
        if _MD_EDIT_LINES_RE.match(st):
            _flush()
            saw_heading = True
            cur_kind = "edit_lines"
            continue
        if _MD_AXES_RE.match(st):
            _flush()
            saw_heading = True
            cur_kind = "axes"
            continue
        msec = _MD_SECTION_RE.match(st)
        if msec:
            _flush()
            saw_heading = True
            cur_kind = "section"
            cur_name = (msec.group("named") or msec.group("bare") or "").strip()
            continue
        if cur_kind is not None:
            cur_buf.append(line)

    _flush()
    if not saw_heading:
        return None
    obj: dict[str, Any] = {}
    if section_edits:
        obj["section_edits"] = section_edits
    if edit_lines:
        obj["edit_lines"] = edit_lines
    if axes_lines:
        obj["axes"] = _parse_markdown_axes_block(axes_lines)
    return obj


def parse_auto_edit_reply(raw_text: str, *, concept_brief: ConceptBrief | None = None) -> dict[str, Any]:
    """Parse Markdown (`## section:` / `## edit_lines` + `## axes`) or JSON `{..., axes}` from `heylook`."""
    cleaned = _strip_think_and_fence(raw_text)
    if cleaned.lstrip().startswith("{"):
        obj = _extract_balanced_json(cleaned)
        axes = _normalize_axes(obj.get("axes"), concept_brief=concept_brief)
        return {"obj": obj, "axes": axes}
    md_obj = _extract_markdown_reply(cleaned)
    if md_obj is not None:
        if "axes" not in md_obj:
            raise ValidationRepairError("reply is missing required '## axes' block")
        axes = _normalize_axes(md_obj.get("axes"), concept_brief=concept_brief)
        return {"obj": md_obj, "axes": axes}
    if "{" in cleaned and "}" in cleaned:
        obj = _extract_balanced_json(cleaned)
        axes = _normalize_axes(obj.get("axes"), concept_brief=concept_brief)
        return {"obj": obj, "axes": axes}
    raise ValidationRepairError(
        "reply must contain Markdown '## section: <name>' (or '## edit_lines') and '## axes' blocks"
    )


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

    # 4. Media/subject tag set preservation (<Subject N>, <Picture N>, <Video N>, <Audio N>):
    # Never invent nonexistent tags or drop bound subjects/media; allow natural repeated mentions
    # of existing <Subject N> tags in prose sections (summary / detailed_description).
    orig_tag_set = set(_MEDIA_REF_TAG_RE.findall(orig_body))
    new_tag_set = set(_MEDIA_REF_TAG_RE.findall(new_body))
    invented = sorted(new_tag_set - orig_tag_set)
    if invented:
        raise ValidationRepairError(
            f"section {sec_name!r} invented media/subject tag(s): {invented} (allowed: {sorted(orig_tag_set)})"
        )
    if sec_name in ("subject_definitions", "retention_analysis"):
        missing = sorted(orig_tag_set - new_tag_set)
    else:
        # In narrative sections, every <Subject N> / <Video N> / <Audio N> must remain present,
        # while a <Picture N> reference tied to a replaced wardrobe phrase may be omitted.
        required = {t for t in orig_tag_set if not t.startswith("<Picture ")}
        missing = sorted(required - new_tag_set)
    if missing:
        raise ValidationRepairError(
            f"section {sec_name!r} dropped required media/subject tag(s): {missing}"
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
    """Validate `section_edits` + `axes` (Markdown or JSON), copy locked sections byte-for-byte, and return result dict."""
    parsed = parse_auto_edit_reply(raw_reply, concept_brief=concept_brief)
    obj = parsed["obj"]
    axes = parsed["axes"]
    raw_edits = obj.get("section_edits")
    if not isinstance(raw_edits, Mapping) or not raw_edits:
        raise ValidationRepairError("reply must include at least one changed '## section: <name>' block")

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
    """Validate `edit_lines` + `axes` (Markdown or JSON) through `h3_prompt_edit.apply_edits` with strict scope guards."""
    parsed = parse_auto_edit_reply(raw_reply, concept_brief=concept_brief)
    obj = parsed["obj"]
    axes = parsed["axes"]
    raw_lines = obj.get("edit_lines")
    if isinstance(raw_lines, list):
        cleaned = [_clean_edit_line(ln) for ln in raw_lines if _clean_edit_line(ln)]
    elif isinstance(raw_lines, str):
        cleaned = [_clean_edit_line(ln) for ln in raw_lines.splitlines() if _clean_edit_line(ln)]
    else:
        raise ValidationRepairError("reply must include a non-empty '## edit_lines' block of 'old -> new' lines")

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
    has_synthetic_video: bool = False,
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
    video_end_percent = 0.33 if has_synthetic_video else 1.0

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
            video_end_percent=video_end_percent,
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
            video_end_percent=video_end_percent,
        )

    if not visual_changed and audio_changed:
        streams = "audio_only"
        last_step = 4
    else:
        streams = "independent"
        last_step = 3

    auto_fast_1x = False
    if take_mode == "lock_take":
        first_step = 2 if (streams == "audio_only" and not has_references_or_ref2va) else 1
    elif take_mode == "allow_reshoot":
        first_step = 0
    else:  # "auto"
        if deforms or not preserve_step0:
            first_step = 0
            # Whole-clip visual overhauls (setting/pose/wardrobe/medium) run cleanly in 1 pass
            # when unmasked (EditGuider transparently falls back to 2-pass first_step=0 if a spatial mask is wired).
            if visual_changed and eff_timing == "whole_clip":
                auto_fast_1x = True
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
        auto_fast_1x=auto_fast_1x,
        video_end_percent=video_end_percent,
    )


_INTERNAL_NOTE_CLAUSE_RE = re.compile(
    r"^(?:category\s*=|pair\s+|replaced(?:\s+cleanly)?\s+by\s+|stack(?:able)?\s+with\s+)",
    re.IGNORECASE,
)
_PRESET_NOTE_PREFIX_RE = re.compile(r"^1x\s+prompt\s+preset\s*[—–-]\s*", re.IGNORECASE)
_IMAGE_AUDIO_SKIP_SECTIONS = frozenset({"overall_soundscape", "non_diegetic_music", "soundscape", "music", "audio"})


def clean_concept_notes(notes: str) -> str:
    """Strip internal catalog/pairing metadata from concept notes while preserving actionable cues."""
    kept: list[str] = []
    for chunk in str(notes or "").split(";"):
        part = chunk.strip()
        if not part or _INTERNAL_NOTE_CLAUSE_RE.match(part):
            continue
        part = _PRESET_NOTE_PREFIX_RE.sub("", part).strip()
        if part:
            kept.append(part)
    return "; ".join(kept)


def format_concept_brief_block(
    concept_brief: ConceptBrief,
    *,
    mode: str = "fresh",
    native_format: bool = True,
) -> str:
    """Format a `ConceptBrief` consistently across `fresh` video, `edit` (2x), and `image` modes."""
    eff_mode = (mode or "fresh").strip().lower()
    if eff_mode not in ("fresh", "edit", "image"):
        raise ValueError(f"unknown concept formatting mode {mode!r}; expected 'fresh', 'edit', or 'image'")

    notes_clean = clean_concept_notes(concept_brief.notes)
    c_lines: list[str] = [f"MANDATORY CREATIVE CONCEPT ({concept_brief.title}): {concept_brief.summary}"]

    if eff_mode == "image":
        c_lines.append(
            "Translate this concept thoroughly into the finished single still frame (lighting, atmosphere, "
            "setting, wardrobe, materials, and visual styling), overriding contradicting baseline defaults "
            "while keeping the user's subjects and any attached <Picture N> identities intact."
        )
        if concept_brief.a_side and concept_brief.b_side:
            c_lines.append(f"Target visual state: {concept_brief.b_side}")
    elif eff_mode == "fresh":
        c_lines.append(
            "Apply this concept thoroughly across the scene — visibly transforming setting, lighting, "
            "atmosphere, wardrobe, action, and matching soundscape (overriding contradicting baseline "
            "details) while keeping the user's subjects and attached references intact."
        )
        if concept_brief.a_side and concept_brief.b_side:
            c_lines.append(f"Target state: {concept_brief.b_side}")
    else:  # "edit"
        c_lines.append(
            "Apply this concept thoroughly across allowed sections (replacing contradicting lighting, "
            "setting, wardrobe, action, atmosphere, and soundscape details while keeping user subjects "
            "and <Picture N>/<Subject N> identities intact)."
        )
        if concept_brief.a_side and concept_brief.b_side:
            c_lines.append(f"Transition: {concept_brief.a_side} -> {concept_brief.b_side}")

    if notes_clean:
        c_lines.append(f"Notes: {notes_clean}")
    if concept_brief.avoid:
        c_lines.append(f"Avoid: {concept_brief.avoid}")

    if eff_mode == "edit":
        if concept_brief.axis_hints:
            hint_parts = [f"{k}={json.dumps(concept_brief.axis_hints[k])}" for k in sorted(concept_brief.axis_hints)]
            c_lines.append(
                "Concept default axis hints (advisory tie-breaker only — override whenever the concept or "
                "edit instruction changes pose, setting, attire, medium, or silhouette): "
                + ", ".join(hint_parts)
            )
            if concept_brief.axis_hints.get("deforms_ref_silhouette") is True:
                c_lines.append(
                    "Silhouette/medium overhaul active: boldly rewrite contradicting setting, pose, "
                    "wardrobe, lighting, props, and (when a non-photographic medium is requested) "
                    "surface materials and motion cadence across allowed visual sections while keeping "
                    "<Picture N> facial identity."
                )
            elif concept_brief.axis_hints.get("preserve_step0_layout") is False:
                c_lines.append(
                    "Scene/pose/wardrobe overhaul active: boldly rewrite contradicting setting, pose, "
                    "wardrobe, lighting, props, and soundscape across allowed sections while keeping "
                    "<Picture N> facial identity."
                )
        if concept_brief.timing_hint and concept_brief.timing_hint != "auto":
            c_lines.append(f"Concept default temporal_curve: {concept_brief.timing_hint}")
        if concept_brief.exemplars:
            c_lines.append("Reference examples:")
            for idx, (sec, a_ex, b_ex) in enumerate(concept_brief.exemplars, start=1):
                c_lines.append(f"  Example {idx} [{sec}]: {a_ex!r} -> {b_ex!r}")
        return "\n".join(c_lines)

    # Fresh video or still image authoring: retain both >=25w flavor examples and <25w sensory cues
    # (labeled explicitly so short cues expand richly rather than priming stubby drafts).
    if eff_mode == "fresh" and concept_brief.axis_hints and concept_brief.axis_hints.get("deforms_ref_silhouette") is True:
        if native_format:
            c_lines.append("Note: this concept substantially alters medium, age, or silhouette.")
        else:
            c_lines.append(
                "Note: this concept substantially alters medium, age, or silhouette (set deforms_ref_silhouette: true)."
            )

    filtered_ex = [
        (sec, a_ex, b_ex)
        for sec, a_ex, b_ex in (concept_brief.exemplars or ())
        if str(b_ex).strip()
        and not (eff_mode == "image" and str(sec or "").strip().lower() in _IMAGE_AUDIO_SKIP_SECTIONS)
    ]
    long_exemplars = [(sec, b_ex) for sec, _a_ex, b_ex in filtered_ex if len(str(b_ex).split()) >= 25]
    short_cues = [(sec, b_ex) for sec, _a_ex, b_ex in filtered_ex if len(str(b_ex).split()) < 25]
    if long_exemplars:
        c_lines.append("Concept visual examples:" if eff_mode == "image" else "Concept flavor examples:")
        for idx, (sec, b_ex) in enumerate(long_exemplars, start=1):
            c_lines.append(f"  Example {idx} [{sec}]: {b_ex!r}")
    if short_cues:
        if eff_mode == "image":
            c_lines.append(
                "Concept visual cues (expand naturally into cohesive single-frame prose — do NOT copy as a short fragment):"
            )
        else:
            c_lines.append(
                "Concept visual/action cues (weave into full multi-sentence cinematic prose — do NOT copy as a short stub):"
            )
        for idx, (sec, b_ex) in enumerate(short_cues, start=1):
            c_lines.append(f"  Cue {idx} [{sec}]: {b_ex!r}")
    return "\n".join(c_lines)


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
        parts.append(format_concept_brief_block(concept_brief, mode="edit"))

    intent_clean = (edit_intent or "").strip()
    if intent_clean:
        parts.append(f"User edit instruction:\n{intent_clean}")

    if rewrite_mode == "section_edits":
        parts.append(
            "Task: Return ONLY Markdown `## section: <exact_section_name>` blocks for changed allowed sections "
            "followed by `## axes`, keeping all tags, dialogue <d>...</d>, speaker IDs, and [Shot N] headers verbatim."
        )
    else:
        parts.append(
            "Task: Return ONLY a Markdown `## edit_lines` bullet list (`- <section_prefix>: <old> -> <new>`) "
            "followed by `## axes`. Each line must match an exact unique phrase inside an allowed section of original."
        )
    return "\n\n".join(parts)


def build_fresh_prompt_user_prompt(
    *,
    concept_brief: ConceptBrief | None,
    edit_intent: str,
    n_images: int,
    resolved_mode: str | None = None,
    manifest: _pc.MediaManifest | None = None,
    length: int = 141,
    native_format: bool = True,
) -> str:
    """Assemble the user turn for direct fresh H3 prompt generation across all H3 modes."""
    eff_mode = (
        resolved_mode
        if resolved_mode is not None
        else ("ref2va" if (n_images > 0 or (manifest is not None and manifest.has_any)) else "t2va")
    )
    eff_manifest = manifest if manifest is not None else _pc.describe_references(n_extra_images=n_images)
    parts: list[str] = []

    snapped = _rules.snap_length(max(5, int(length)))
    dur_s = _rules.duration_of(snapped)
    if eff_mode == "fl2va":
        pacing_note = (
            f"Target duration: {dur_s:.2f}s ({snapped} frames) — bridge cleanly from start frame to end "
            "frame in one continuous [Shot 1] (or add a justified cut only if the transition clearly "
            "calls for it); never write meta-duration filler like 'across the six-second take'."
        )
    elif dur_s >= 7.5:
        pacing_note = (
            f"Target duration: {dur_s:.2f}s ({snapped} frames) — structure 2-3 concrete cinematic shots "
            "([Shot 1] establishing action/dialogue -> [Shot 2] close-up reaction/payoff) with real motion "
            "unless the user explicitly asks for a single unbroken shot; never write meta-duration filler "
            "like 'across the six-second take'."
        )
    else:
        pacing_note = (
            f"Target duration: {dur_s:.2f}s ({snapped} frames) — concrete continuous action and camera "
            "motion; never write meta-duration filler like 'across the six-second take'."
        )

    if eff_mode == "ref2va":
        lines = [
            "Mode: ref2va (6 required sections; strictly 350-500 words in detailed_description).",
            pacing_note,
        ]
        if eff_manifest.summary_lines:
            lines.append("Attached media references in exact order:")
            lines.extend(eff_manifest.summary_lines)
        if eff_manifest.pictures:
            subjs = ", ".join(f"<Subject {i}>" for i in range(1, eff_manifest.n_pictures + 1))
            lines.append(
                f"Bind {', '.join(eff_manifest.pictures)} to {subjs} in subject_definitions; "
                "inspect every attached image carefully for facial identity, hair, and features."
            )
        if eff_manifest.audios:
            lines.append(
                "In retention_analysis, give every <Audio N> an audio retention marker "
                "(fully_copy/partially_copy/reference/weak_reference) with NO '(appears in ...)' shot scope, "
                "and never write speaker IDs (Sx) inside retention_analysis."
            )
        parts.append("\n".join(lines))
    elif eff_mode in _pc.KEYFRAME_BASE_MODES:
        kf_note = (
            "Bare `Picture 1` (0.00s start) and bare `Picture 2` (end) ONLY — never `<Picture 1>`!"
            if eff_mode == "fl2va"
            else (
                "Start anchored on `<Picture 1>` at 0.00s."
                if eff_mode == "i2va"
                else "End converging onto `<Picture 1>` at the final beat."
            )
        )
        parts.append(
            f"Mode: {eff_mode} (3 base sections; duration {dur_s:.2f}s / {snapped} frames; "
            f"Do not write a leading alignment header line — begin directly with integrated_multimodal_description:).\n{kf_note}\n{pacing_note}"
        )
    elif eff_mode == "single_frame":
        pic_note = (
            f" Attached pictures: {', '.join(eff_manifest.pictures)}."
            if eff_manifest.pictures
            else ""
        )
        parts.append(
            f"Mode: single_frame (sectionless visual prose only; zero soundscape/music sections).{pic_note}"
        )
    else:
        parts.append(f"Mode: t2va (3 base sections; no reference images attached).\n{pacing_note}")

    if concept_brief is not None:
        parts.append(
            format_concept_brief_block(
                concept_brief,
                mode="image" if eff_mode == "single_frame" else "fresh",
                native_format=native_format,
            )
        )

    intent_clean = (edit_intent or "").strip()
    if intent_clean:
        parts.append(f"User scene instruction:\n{intent_clean}")

    if native_format:
        if eff_mode == "ref2va":
            parts.append(
                "Task: Return ONLY the 6 plain-text sections in order: "
                "subject_definitions:, summary:, retention_analysis:, detailed_description:, "
                "overall_soundscape:, non_diegetic_music: (no markdown fences, no commentary). "
                "If anyone speaks via <d>[English] ...</d>, bind <Subject N> (S1) earlier in the shot."
            )
        elif eff_mode == "single_frame":
            parts.append(
                "Task: Return ONLY a single continuous plain-text visual paragraph "
                "(no section headers, no markdown fences, no commentary)."
            )
        else:
            parts.append(
                "Task: Return ONLY the 3 plain-text sections in order: "
                "integrated_multimodal_description:, overall_soundscape:, non_diegetic_music: "
                "(no leading alignment line, no markdown fences, no commentary)."
            )
    elif eff_mode == "ref2va":
        parts.append(
            "Task: Return ONLY Markdown `## section: <name>` blocks for "
            "`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, "
            "`overall_soundscape`, `non_diegetic_music`, followed by `## axes`. "
            "If anyone speaks via `<d>[English] ...</d>`, bind `<Subject N> (S1)` earlier in the shot."
        )
    elif eff_mode == "single_frame":
        parts.append(
            "Task: Return ONLY Markdown `## section: _unsectioned` (one concise paragraph of visual prose) "
            "followed by `## axes` (with `audio_changed: false`)."
        )
    else:
        parts.append(
            "Task: Return ONLY Markdown `## section: <name>` blocks for "
            "`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`, "
            "followed by `## axes`."
        )
    return "\n\n".join(parts)


def validate_fresh_h3_prompt(
    raw_reply: str,
    *,
    n_images: int,
    concept_brief: ConceptBrief | None = None,
    resolved_mode: str | None = None,
    manifest: _pc.MediaManifest | None = None,
    length: int = 141,
    enforce_min_words: bool = True,
) -> dict[str, Any]:
    """Parse and validate a fresh Markdown or native plain-text H3 prompt across all H3 modes."""
    eff_manifest = manifest if manifest is not None else _pc.describe_references(n_extra_images=n_images)
    eff_mode = (
        resolved_mode
        if resolved_mode is not None
        else ("ref2va" if (n_images > 0 or eff_manifest.has_any) else "t2va")
    )
    cleaned = _strip_think_and_fence(raw_reply)
    is_legacy = bool(
        re.search(
            r"(?m)^#{1,4}\s*(?:section\s*:|(?:subject_definitions|summary|retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music)\b)",
            cleaned,
            re.I,
        )
        or cleaned.lstrip().startswith("{")
    )
    if is_legacy:
        parsed = parse_auto_edit_reply(raw_reply, concept_brief=concept_brief)
        obj = parsed["obj"]
        sec_map = obj.get("section_edits")
        if not isinstance(sec_map, Mapping) or not sec_map:
            raise ValidationRepairError("fresh prompt reply contained no `## section:` blocks")
    else:
        m_axes = re.search(r"(?m)^#{1,4}\s*axes\s*:?\s*$", cleaned, re.I)
        if m_axes:
            body_text = cleaned[: m_axes.start()].strip()
            axes_tail = cleaned[m_axes.start() :]
            md_tail = _extract_markdown_reply(axes_tail)
            raw_axes = (md_tail or {}).get("axes") if isinstance(md_tail, Mapping) else None
            parsed_axes = _normalize_axes(raw_axes, concept_brief=concept_brief)
        else:
            body_text = cleaned.strip()
            parsed_axes = _normalize_axes({}, concept_brief=concept_brief)
        if eff_mode == "single_frame":
            body_no_ar = re.sub(r"(?im)\n+\s*(?:aspect_ratio|target_aspect_ratio)\s*:.*$", "", body_text).strip()
            if not body_no_ar:
                raise ValidationRepairError("single_frame fresh prompt is empty")
            sec_map = {"_unsectioned": body_no_ar}
        else:
            sec_map = _pc._extract_sections(body_text)
            if not sec_map:
                want_hdrs = (
                    "subject_definitions:, summary:, retention_analysis:, detailed_description:, "
                    "overall_soundscape:, non_diegetic_music:"
                    if eff_mode == "ref2va"
                    else "integrated_multimodal_description:, overall_soundscape:, non_diegetic_music:"
                )
                raise ValidationRepairError(
                    f"fresh prompt missing plain-text section headers ({want_hdrs})"
                )
        parsed = {"obj": {"section_edits": sec_map}, "axes": parsed_axes}

    if eff_mode == "single_frame":
        body_raw = sec_map.get("_unsectioned")
        if body_raw is None and len(sec_map) == 1:
            body_raw = next(iter(sec_map.values()))
        if not body_raw or not str(body_raw).strip():
            raise ValidationRepairError("single_frame fresh prompt missing `## section: _unsectioned` visual prose")
        assembled = _pc.strip_part_one(str(body_raw).strip())
        required: tuple[str, ...] = ("_unsectioned",)
    else:
        required = _pc.REF2VA_SECTIONS if eff_mode == "ref2va" else _pc.BASE_SECTIONS
        cleaned: dict[str, str] = {}
        missing = [s for s in required if s not in sec_map or not str(sec_map[s]).strip()]
        if missing:
            raise ValidationRepairError(f"fresh prompt missing required section(s): {missing}")
        for s in required:
            cleaned[s] = _strip_redundant_section_header(str(sec_map[s]).strip(), s)
        if eff_mode == "ref2va" and not _TASK_BRACKET_RE.match(cleaned["summary"]):
            cleaned["summary"] = f"[reference generation] {cleaned['summary']}"
        if eff_mode in _pc.KEYFRAME_BASE_MODES:
            cleaned["integrated_multimodal_description"] = _pc.strip_part_one(
                cleaned["integrated_multimodal_description"]
            )
        assembled = "\n\n".join(f"{s}:\n{cleaned[s]}" for s in required)
        if eff_mode in _pc.KEYFRAME_BASE_MODES:
            assembled = _pc.stamp_part_one(assembled, mode=eff_mode, length=length)

    mech_fails, _mech_warns = _pc.validate_h3_prompt_mechanics(
        assembled,
        mode=eff_mode,
        manifest=eff_manifest if eff_mode in ("ref2va", "single_frame") else None,
        length=length,
        enforce_min_words=bool(enforce_min_words and eff_mode == "ref2va"),
    )
    if mech_fails:
        raise ValidationRepairError("; ".join(mech_fails))

    axes = dict(parsed.get("axes") or {})
    axes.setdefault("visual_changed", True)
    if eff_mode == "single_frame":
        axes["audio_changed"] = False
    else:
        axes.setdefault("audio_changed", True)
    axes["preserve_step0_layout"] = False
    if "deforms_ref_silhouette" not in axes and concept_brief and concept_brief.axis_hints:
        axes["deforms_ref_silhouette"] = bool(concept_brief.axis_hints.get("deforms_ref_silhouette", False))
    axes.setdefault("deforms_ref_silhouette", False)
    axes.setdefault("temporal_curve", "whole_clip")

    edit_words = len(assembled.split())
    return {
        "edited_prompt": assembled,
        "changed_sections": required,
        "reverted_locked": (),
        "axes": axes,
        "edit_lines_text": "",
        "orig_words": 0,
        "edit_words": edit_words,
    }


def _extract_vision_tensors_from_references(
    references: Any,
    *,
    max_total: int = 8,
) -> list[Any]:
    """Collect still images + up to 3 evenly spaced keyframes per `<Video N>` from `H3References`."""
    if not references or not isinstance(references, (tuple, list)):
        return []
    tensors: list[Any] = []
    for item in references:
        if len(tensors) >= max_total:
            break
        if hasattr(item, "image") and getattr(item, "image", None) is not None:
            tensors.append(item.image)
        elif hasattr(item, "frames") and getattr(item, "frames", None) is not None:
            frames = item.frames
            if hasattr(frames, "shape") and len(frames.shape) == 4 and int(frames.shape[0]) > 0:
                n_f = int(frames.shape[0])
                sample_idxs = sorted({0, n_f // 2, n_f - 1})
                for s_idx in sample_idxs:
                    if len(tensors) >= max_total:
                        break
                    tensors.append(frames[s_idx : s_idx + 1])
    return tensors


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
    references: Any = None,
    length: int = 141,
    prompt_mode: str = "auto",
    client: httpx.Client | None = None,
    interrupt_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute or cache-hit one `MiniMaxH3AutoEditPrompt` rewrite (or fresh generation when `original` is blank) and compile its `EditPlan`."""
    orig_norm = _rules.normalize_prompt(original)
    is_fresh_mode = not bool(orig_norm)
    if take_mode not in TAKE_MODES:
        raise ValueError(f"unknown take_mode {take_mode!r}; expected one of {TAKE_MODES}")
    if timing not in TIMING_MODES:
        raise ValueError(f"unknown timing {timing!r}; expected one of {TIMING_MODES}")
    if rewrite_mode not in REWRITE_MODES:
        raise ValueError(f"unknown rewrite_mode {rewrite_mode!r}; expected one of {REWRITE_MODES}")
    if prompt_layout not in PROMPT_LAYOUTS:
        raise ValueError(f"unknown prompt_layout {prompt_layout!r}; expected one of {PROMPT_LAYOUTS}")

    if not enabled:
        if is_fresh_mode:
            raise ValueError("MiniMaxH3AutoEditPrompt: 'original' prompt is empty while enabled=False")
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

    names = sorted(images or {}, key=lambda nm: int(nm.rsplit("_", 1)[-1]))
    raw_images = [images[nm] for nm in names if images[nm] is not None]
    if len(raw_images) > MAX_AUTO_EDIT_IMAGES:
        raise ValueError(
            f"MiniMaxH3AutoEditPrompt accepts at most {MAX_AUTO_EDIT_IMAGES} images, got {len(raw_images)}"
        )
    ref_tensors = _extract_vision_tensors_from_references(references)
    vision_tensors = ref_tensors if ref_tensors else raw_images
    prepared_images = [prepare_heylook_tensor(t, index=i) for i, t in enumerate(vision_tensors, start=1)]
    image_hashes = tuple(hashlib.sha256(b64.encode("ascii")).hexdigest()[:16] for b64, _ in prepared_images)

    eff_length = int(length)
    manifest = _pc.describe_references(references, n_extra_images=len(raw_images))
    resolved_mode = _pc.resolve_prompt_mode(
        prompt_mode,
        original=orig_norm,
        manifest=manifest,
        length=eff_length,
        has_references_wire=bool(references),
    )

    if is_fresh_mode:
        resolved_scope = f"fresh_{resolved_mode}"
        if resolved_mode == "ref2va":
            allowed_sections = _pc.REF2VA_SECTIONS
            is_sectioned = True
        elif resolved_mode == "single_frame":
            allowed_sections = ("_unsectioned",)
            is_sectioned = False
        else:
            allowed_sections = _pc.BASE_SECTIONS
            is_sectioned = True
        has_refs = bool(manifest.has_any or prepared_images or resolved_mode in ("ref2va", "i2va", "fl2va", "l2va"))
        orig_chunks = ()
    else:
        resolved_scope, allowed_sections = resolve_allowed_sections(
            orig_norm,
            target_scope,
            allow_retention_relax=bool(allow_retention_relax),
            concept_brief=concept_brief,
            has_edit_intent=bool(intent_clean),
        )
        has_refs = _detect_has_references(orig_norm, prepared_images) or manifest.has_any
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
    base_cache_key = (
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
        resolved_mode,
        eff_length,
        manifest.labels,
    )

    def _hit_cached(ck: tuple[Any, ...]) -> dict[str, Any] | None:
        cached = _REWRITE_CACHE.get(ck) if use_cache else None
        if cached is None:
            return None
        _REWRITE_CACHE.move_to_end(ck)
        plan = compile_edit_plan(
            axes=cached["axes"],
            changed_sections=cached["changed_sections"],
            take_mode=take_mode,
            timing=timing,
            has_references_or_ref2va=has_refs,
            is_sectioned=is_sectioned,
            concept_brief=concept_brief,
            has_synthetic_video=manifest.has_synthetic_video,
        )
        if plan.auto_fast_1x:
            cached_report = (
                f"{cached['report_base']} | take={take_mode}(auto_fast_1x,step0={plan.first_step},streams=both,"
                f"rel={plan.ref_end_percent:g},timing={plan.temporal_curve}) [cached]"
            )
        else:
            cached_report = (
                f"{cached['report_base']} | take={take_mode}(step0={plan.first_step},streams={plan.streams},"
                f"rel={plan.ref_end_percent:g},timing={plan.temporal_curve}) [cached]"
            )
        return {
            **cached,
            "edit_plan": plan,
            "cache_hit": True,
            "report": cached_report,
        }

    custom_sys = (custom_system_prompt or "").strip()
    custom_extend_mode, _custom_extend_body = parse_extend_directive(custom_sys) if custom_sys else (None, "")

    def _resolve_prompts_for_request(
        preset_fields: dict[str, Any] | None,
    ) -> tuple[str, str, str, bool, dict[str, Any] | None]:
        preset_sys = str((preset_fields or {}).get("system") or "").strip()
        extend_mode, _extend_body = parse_extend_directive(preset_sys) if preset_sys else (None, "")
        eff_preset_fields = preset_fields
        if is_fresh_mode:
            local_sys = resolve_fresh_system_prompt(
                resolved_mode,
                has_images=bool(prepared_images),
            )
            if custom_sys and custom_extend_mode is not None:
                base_sys_prompt, sys_source, _ = compose_preset_system_prompt(
                    local_sys,
                    custom_sys,
                    preset_name=preset_name or "custom",
                )
                eff_preset_fields = (
                    {k: v for k, v in (preset_fields or {}).items() if k != "system"}
                    if preset_fields
                    else None
                )
                use_native = "## section:" not in base_sys_prompt
            elif custom_sys:
                base_sys_prompt = custom_sys
                sys_source = "custom"
                use_native = "## section:" not in custom_sys
            elif preset_sys and extend_mode is not None:
                base_sys_prompt, sys_source, _ = compose_preset_system_prompt(
                    local_sys,
                    preset_sys,
                    preset_name=preset_name,
                )
                eff_preset_fields = {k: v for k, v in (preset_fields or {}).items() if k != "system"}
                use_native = True
            elif preset_sys:
                base_sys_prompt = ""
                sys_source = f"preset:{preset_name}"
                use_native = "## section:" not in preset_sys
            else:
                base_sys_prompt = local_sys
                sys_source = "local"
                use_native = True
            user_prompt = build_fresh_prompt_user_prompt(
                concept_brief=concept_brief,
                edit_intent=intent_clean,
                n_images=len(prepared_images),
                resolved_mode=resolved_mode,
                manifest=manifest,
                length=eff_length,
                native_format=use_native,
            )
        else:
            local_sys = _read_system_prompt_file(rewrite_mode)
            if custom_sys and custom_extend_mode is not None:
                base_sys_prompt, sys_source, _ = compose_preset_system_prompt(
                    local_sys,
                    custom_sys,
                    preset_name=preset_name or "custom",
                )
                eff_preset_fields = (
                    {k: v for k, v in (preset_fields or {}).items() if k != "system"}
                    if preset_fields
                    else None
                )
            elif custom_sys:
                base_sys_prompt = custom_sys
                sys_source = "custom"
            elif preset_sys and extend_mode is not None:
                base_sys_prompt, sys_source, _ = compose_preset_system_prompt(
                    local_sys,
                    preset_sys,
                    preset_name=preset_name,
                )
                eff_preset_fields = {k: v for k, v in (preset_fields or {}).items() if k != "system"}
            elif preset_sys:
                req_marker = "## edit_lines" if rewrite_mode == "edit_lines" else "## section:"
                if req_marker in preset_sys and "## axes" in preset_sys:
                    base_sys_prompt = ""
                    sys_source = f"preset:{preset_name}"
                else:
                    raise ValueError(
                        f"MiniMaxH3AutoEditPrompt: heylook preset {preset_name!r} has a non-2x system_prompt "
                        f"(missing {req_marker!r} and '## axes'); clear 'preset', choose a sampler-only preset "
                        "(blank system_prompt), or choose a 2x rewrite preset."
                    )
            else:
                base_sys_prompt = local_sys
                sys_source = "local"
            user_prompt = build_auto_edit_user_prompt(
                original=orig_norm,
                allowed_sections=allowed_sections,
                allow_retention_relax=bool(allow_retention_relax),
                rewrite_mode=rewrite_mode,
                concept_brief=concept_brief,
                edit_intent=intent_clean,
                has_images=bool(prepared_images),
            )
            use_native = False
        if prompt_layout == "user_only" and not custom_sys and base_sys_prompt:
            eff_sys = ""
            eff_user = f"{base_sys_prompt}\n\n---\n\n{user_prompt}"
        else:
            eff_sys = base_sys_prompt
            eff_user = user_prompt
        return eff_sys, eff_user, sys_source, use_native, eff_preset_fields

    if not preset_name:
        pre_sys, pre_user, _pre_src, _pre_native, _pre_pf = _resolve_prompts_for_request(None)
        cache_key = base_cache_key + (
            hashlib.sha256(pre_sys.encode("utf-8")).hexdigest()[:16],
            hashlib.sha256(pre_user.encode("utf-8")).hexdigest()[:16],
            "",
        )
        hit = _hit_cached(cache_key)
        if hit is not None:
            return hit

    sys_prompt = ""
    final_user_prompt = ""
    sys_source = "local"
    use_native = False

    need_caps = frozenset({"chat", "vision"} if prepared_images else {"chat"})
    allow_audio_ret = target_scope == "all_sections"

    def _run_validator(raw_text: str) -> dict[str, Any]:
        if is_fresh_mode:
            return validate_fresh_h3_prompt(
                raw_text,
                n_images=len(prepared_images),
                concept_brief=concept_brief,
                resolved_mode=resolved_mode,
                manifest=manifest,
                length=eff_length,
            )
        fn = apply_section_edits_validated if rewrite_mode == "section_edits" else apply_edit_lines_validated
        out_val = fn(
            orig_norm,
            raw_text,
            allowed_sections=allowed_sections,
            allow_audio_retention_edit=allow_audio_ret,
            concept_brief=concept_brief,
        )
        if resolved_mode in _pc.KEYFRAME_BASE_MODES:
            out_val["edited_prompt"] = _pc.stamp_part_one(
                out_val["edited_prompt"],
                mode=resolved_mode,
                length=eff_length,
            )
        orig_mech_fails, _ = _pc.validate_h3_prompt_mechanics(
            orig_norm,
            mode=resolved_mode,
            manifest=manifest if manifest.has_any else None,
            length=eff_length,
        )
        mech_fails, _mech_warns = _pc.validate_h3_prompt_mechanics(
            out_val["edited_prompt"],
            mode=resolved_mode,
            manifest=manifest if manifest.has_any else None,
            length=eff_length,
        )
        new_mech_fails = [f for f in mech_fails if f not in set(orig_mech_fails)]
        if new_mech_fails:
            raise ValidationRepairError(
                f"H3 prompt contract violation introduced by edit: {'; '.join(new_mech_fails)}"
            )
        return out_val

    repair_used = False
    thinking_parts: list[str] = []
    req_effort = "" if thinking == "off" else eff_effort
    eff_interrupt = interrupt_check if interrupt_check is not None else _comfy_interrupt_check
    client_cm = (
        contextlib.nullcontext(client)
        if client is not None
        else make_heylook_client(base, timeout=httpx.Timeout(None, connect=10.0))
    )
    with client_cm as req_client:
        model_row = discover_heylook_model(req_client, base, model, need=need_caps)
        raw_preset_fields = (
            expand_heylook_preset(req_client, base, preset_name, model_row)
            if preset_name
            else None
        )
        sys_prompt, final_user_prompt, sys_source, use_native, preset_fields = _resolve_prompts_for_request(
            raw_preset_fields
        )
        resolved_preset_sig = (
            json.dumps(raw_preset_fields or {}, sort_keys=True, default=str)
            if preset_name
            else ""
        )
        preset_sig_hash = (
            hashlib.sha256(f"{model_row.get('id', '')}|{resolved_preset_sig}".encode("utf-8")).hexdigest()[:16]
            if preset_name
            else ""
        )
        cache_key = base_cache_key + (
            hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()[:16],
            hashlib.sha256(final_user_prompt.encode("utf-8")).hexdigest()[:16],
            preset_sig_hash,
        )
        if preset_name:
            hit = _hit_cached(cache_key)
            if hit is not None:
                return hit

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
            lambda: stream_heylook_message(base, body, client=req_client, interrupt_check=eff_interrupt),
            interrupt_check=eff_interrupt,
        )
        t1 = res.thinking or _extract_inline_thinking(res.text)
        if t1:
            thinking_parts.append(t1)

        try:
            validated = _run_validator(res.text)
        except ValidationRepairError as first_err:
            if res.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"heylook stopped at max_tokens before completing a valid rewrite ({first_err}). "
                    "Raise max_tokens or switch thinking to 'off' / lower reasoning_effort."
                ) from first_err
            repair_used = True
            err_s = str(first_err)
            band_hint = (
                " Aim squarely for ~410-440 words inside detailed_description so it sits comfortably "
                "inside the 350-500 word band (do not stop near the 350 edge)."
                if ("350-500 words" in err_s or "> 500 ceiling" in err_s)
                else ""
            )
            valid_headers_note = ", ".join(allowed_sections) if is_fresh_mode else ", ".join(c.name for c in orig_chunks)
            if is_fresh_mode and use_native:
                if resolved_mode == "single_frame":
                    repair_msg = (
                        f"Your output failed validation: {first_err}\n"
                        "Fix ONLY this validation error and output ONE plain-text visual paragraph "
                        "with no Markdown headings, no code fences, and no commentary."
                    )
                else:
                    headers_csv = ", ".join(f"{s}:" for s in allowed_sections)
                    repair_msg = (
                        f"Your output failed validation: {first_err}\n"
                        f"Required plain-text section headers (exact order): {headers_csv}\n"
                        "Fix ONLY this validation error — do NOT drop or revert passing sections."
                        f"{band_hint} "
                        "Output ALL required plain-text sections cleanly with no Markdown headings, "
                        "no code fences, and no commentary."
                    )
            else:
                repair_msg = (
                    f"Your output failed validation: {first_err}\n"
                    f"Required/valid section names: {valid_headers_note}\n"
                    f"Allowed sections to edit: {', '.join(allowed_sections)}\n"
                    "Fix ONLY this validation error — do NOT drop or revert previously written sections."
                    f"{band_hint} "
                    "Output ALL required Markdown sections and the `## axes` block."
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
                    base, repair_body, client=req_client, interrupt_check=eff_interrupt
                ),
                interrupt_check=eff_interrupt,
            )
            t2 = res.thinking or _extract_inline_thinking(res.text)
            if t2:
                thinking_parts.append(t2)
            try:
                validated = _run_validator(res.text)
            except ValidationRepairError as second_err:
                raise ValueError(
                    f"MiniMaxH3AutoEditPrompt validation failed after 1 repair attempt: {second_err}"
                ) from second_err

    telem = format_heylook_report(
        res,
        resolved_mode=f"{rewrite_mode}/{resolved_scope}",
        system_source=sys_source,
        concept=concept_brief.stem if concept_brief else None,
    )
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
        "original_prompt": validated["edited_prompt"] if is_fresh_mode else orig_norm,
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
        has_synthetic_video=manifest.has_synthetic_video,
    )
    if plan.auto_fast_1x:
        full_report = (
            f"{report_base} | take={take_mode}(auto_fast_1x,step0={plan.first_step},streams=both,"
            f"rel={plan.ref_end_percent:g},timing={plan.temporal_curve})"
        )
    else:
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
                        "changed sections in Markdown (`## section: ...`); 'edit_lines' emits old -> new "
                        "lines validated via MiniMaxH3PromptEdit."
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
