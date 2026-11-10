"""In-place target-prompt contrast compiler for MiniMax H3 control vectors.

Compiles explicit `(pole_a -> pole_b)` clause contrasts, `soundscape:` audio
contrasts, and `hold:` / `hold(group):` orthogonal anchors directly in-place
on a user's `target_prompt` (or multi-scene `target_prompt` separated by
`\\n---\\n`), with zero static template banks, zero Mad-Libs slot wrappers, and
zero canonical-frame dilution.

Why in-place `target_prompt` compilation works mathematically:
Under `pooling = "diff_span"` (`h3_spans.py::pooling_mask`), only the differing
token span `[first_diff:end_diff]` (or multi-span union when multiple clauses
or `soundscape:` are coupled) is mean-pooled per pair side. Encoding the
contrast in-place inside the actual target prompt provides:
  1. The exact causal Stage-1 prefix context (`0..first_diff`) in `Qwen3-VL-32B`,
  2. The exact bidirectional Stage-2 self-attention context (`0..L`) in the
     2-block `TokenRefiner` (with Pika's 24 `token_refiner` LoRA branches), and
  3. The exact 3D M-RoPE / sequence positions of the target render.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import control_vectors as _cv
except ImportError:
    import control_vectors as _cv

SUPPORTED_FRAME_MODES: tuple[str, ...] = ("t2va", "fl2va", "ref2va", "p_h3_image")

try:
    from . import h3_prompt_edit as _pe
except ImportError:
    import h3_prompt_edit as _pe

# The clause grammar has one copy, in `h3_prompt_edit.py`, which loads no model library so the static
# graph walkers can read an edit node's text. These names are kept for the callers here and in
# `bench/control_vector/ui_state.py`.
_PAIR_SPLIT_RE = _pe.PAIR_SPLIT_RE
_SOUNDSCAPE_PREFIX_RE = _pe.SOUNDSCAPE_PREFIX_RE
_split_pole_pair = _pe.split_pole_pair
_split_clause_statements = _pe.split_clause_statements
_HOLD_PREFIX_RE = re.compile(
    r"^(?:hold|holds)(?:\s*\(\s*([a-zA-Z0-9_-]+)\s*\))?\s*:\s*(.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContrastClause:
    """A single contrastive clause (`a` baseline vs. `b` steered)."""

    a: str
    b: str
    is_soundscape: bool = False


@dataclass(frozen=True)
class HoldClause:
    """A nuisance contrast (`a` vs. `b`) to project out via Gram-Schmidt (`[[hold]]`)."""

    a: str
    b: str
    group: str = "nuisance"
    weight: float = 1.0
    is_soundscape: bool = False


def normalize_compile_mode(mode: str | None) -> str:
    """Normalize and validate an H3 prompt mode (`'t2va'`, `'fl2va'`, `'ref2va'`, `'p_h3_image'`)."""
    m = str(mode or "t2va").strip().lower()
    if m in ("t2v", "t2va"):
        return "t2va"
    if m in ("fl2v", "fl2va", "i2v", "i2va"):
        return "fl2va"
    if m in ("r2v", "ref2v", "ref2va"):
        return "ref2va"
    if m in ("image", "t2i", "i2i", "ph3", "p_h3_image", "p-h3-image"):
        return "p_h3_image"
    raise ValueError(
        f"unsupported mode {mode!r}; expected one of {SUPPORTED_FRAME_MODES}"
    )


def infer_compile_mode(
    explicit_mode: str | None = None,
    *,
    ref_images: Any | None = None,
    target_prompt: str | None = None,
    words: str | None = None,
) -> str:
    """Infer the H3 prompt mode from `explicit_mode`, `ref_images`, `target_prompt`, or `words`."""
    if explicit_mode is not None and str(explicit_mode).strip().lower() not in ("", "auto"):
        return normalize_compile_mode(explicit_mode)
    if ref_images:
        return "ref2va"
    for text in (target_prompt, words):
        if not text or not str(text).strip():
            continue
        s = str(text)
        if (
            "subject_definitions:" in s
            or "detailed_description:" in s
            or "retention_analysis:" in s
            or "<Subject " in s
        ):
            return "ref2va"
        if (
            "How the reference pictures align with the target video" in s
            or "Alignment between keyframes and video timeline:" in s
            or "<0.0 seconds>" in s
            or "<last frame>" in s
        ):
            return "fl2va"
        if "integrated_multimodal_description:" in s or "overall_soundscape:" in s:
            return "t2va"
    if target_prompt and str(target_prompt).strip():
        tp_s = str(target_prompt).strip()
        if (
            "[Shot " not in tp_s
            and "integrated_multimodal_description:" not in tp_s
            and "detailed_description:" not in tp_s
            and "overall_soundscape:" not in tp_s
        ):
            return "p_h3_image"
    return "t2va"


def _slugify_name(text: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(text or "").lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    tokens = [t for t in s.split("_") if t]
    stop = {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "with",
        "and",
        "or",
        "of",
        "to",
        "from",
        "into",
        "by",
        "as",
        "is",
        "are",
        "shot",
        "camera",
        "scene",
        "filmed",
        "captured",
    }
    kept = [t for t in tokens if t not in stop][:4] or tokens[:4]
    slug = "_".join(kept)[:48].strip("_")
    return slug or "custom_vector"


def _inject_soundscape_into_target(tp_text: str, sound_sentence: str) -> str:
    """Replace the `overall_soundscape:` body inside `tp_text` with `sound_sentence`."""
    snd = sound_sentence.strip()
    if snd and not snd.endswith((".", "!", "?")):
        snd = f"{snd}."
    return re.sub(
        r"(overall_soundscape:\s*\n?)([^\n]+)",
        lambda m: f"{m.group(1)}{snd}",
        tp_text,
        count=1,
    )


def _extract_existing_soundscape(tp_text: str) -> str:
    m = re.search(r"overall_soundscape:\s*\n?([^\n]+)", tp_text)
    return m.group(1).strip() if m else ""


def _parse_clauses_and_holds(
    words_or_spec: str,
    *,
    extra_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    soundscape_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    holds: Sequence[HoldClause | Mapping[str, Any] | tuple[str, ...] | str] | None = None,
) -> tuple[list[ContrastClause], list[ContrastClause], list[HoldClause]]:
    visual_clauses: list[ContrastClause] = []
    sound_clauses: list[ContrastClause] = []
    hold_clauses: list[HoldClause] = []

    for stmt in _split_clause_statements(words_or_spec):
        m_hold = _HOLD_PREFIX_RE.match(stmt)
        if m_hold:
            grp = (m_hold.group(1) or "nuisance").strip() or "nuisance"
            body = m_hold.group(2).strip()
            m_snd = _SOUNDSCAPE_PREFIX_RE.match(body)
            is_snd = False
            if m_snd:
                is_snd = True
                body = m_snd.group(1).strip()
            ha, hb = _split_pole_pair(body)
            hold_clauses.append(
                HoldClause(a=ha, b=hb, group=grp, weight=1.0, is_soundscape=is_snd)
            )
            continue

        m_snd = _SOUNDSCAPE_PREFIX_RE.match(stmt)
        if m_snd:
            sa, sb = _split_pole_pair(m_snd.group(1))
            sound_clauses.append(ContrastClause(a=sa, b=sb, is_soundscape=True))
            continue

        ca, cb = _split_pole_pair(stmt)
        visual_clauses.append(ContrastClause(a=ca, b=cb, is_soundscape=False))

    if extra_clauses:
        for item in extra_clauses:
            if isinstance(item, ContrastClause):
                if item.is_soundscape:
                    sound_clauses.append(item)
                else:
                    visual_clauses.append(item)
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                visual_clauses.append(
                    ContrastClause(a=str(item[0]).strip(), b=str(item[1]).strip())
                )
            elif isinstance(item, str) and item.strip():
                ca, cb = _split_pole_pair(item)
                visual_clauses.append(ContrastClause(a=ca, b=cb))

    if soundscape_clauses:
        for item in soundscape_clauses:
            if isinstance(item, ContrastClause):
                sound_clauses.append(
                    ContrastClause(a=item.a, b=item.b, is_soundscape=True)
                )
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                sound_clauses.append(
                    ContrastClause(
                        a=str(item[0]).strip(),
                        b=str(item[1]).strip(),
                        is_soundscape=True,
                    )
                )
            elif isinstance(item, str) and item.strip():
                sa, sb = _split_pole_pair(item)
                sound_clauses.append(ContrastClause(a=sa, b=sb, is_soundscape=True))

    if holds:
        for item in holds:
            if isinstance(item, HoldClause):
                hold_clauses.append(item)
            elif isinstance(item, Mapping):
                ha = str(item.get("a", "") or "").strip()
                hb = str(item.get("b", "") or item.get("words", "") or "").strip()
                if not ha and hb:
                    ha, hb = _split_pole_pair(hb)
                grp = str(item.get("group", "nuisance") or "nuisance").strip()
                w = float(item.get("weight", 1.0))
                is_snd = bool(item.get("is_soundscape", False)) or str(item.get("slot", "")).lower() == "soundscape"
                if hb:
                    hold_clauses.append(
                        HoldClause(a=ha, b=hb, group=grp, weight=w, is_soundscape=is_snd)
                    )
            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    hold_clauses.append(
                        HoldClause(a=str(item[0]).strip(), b=str(item[1]).strip())
                    )
                elif len(item) >= 3:
                    hold_clauses.append(
                        HoldClause(
                            group=str(item[0]).strip() or "nuisance",
                            a=str(item[1]).strip(),
                            b=str(item[2]).strip(),
                        )
                    )
            elif isinstance(item, str) and item.strip():
                m_h = _HOLD_PREFIX_RE.match(item.strip())
                if m_h:
                    grp = (m_h.group(1) or "nuisance").strip() or "nuisance"
                    ha, hb = _split_pole_pair(m_h.group(2))
                else:
                    grp = "nuisance"
                    ha, hb = _split_pole_pair(item)
                hold_clauses.append(HoldClause(a=ha, b=hb, group=grp))

    return visual_clauses, sound_clauses, hold_clauses


def _split_target_prompts(target_prompt: str | None) -> list[str]:
    tp_clean = str(target_prompt or "").strip()
    if not tp_clean:
        raise ValueError(
            "target_prompt is required when compiling contrast clauses "
            "(pass --target-prompt <prompt>, --pairs <file.toml>, or use inline J-Lens words on MiniMaxH3ApplyControlVectors)"
        )
    parts = [
        blk.strip()
        for blk in re.split(r"\n\s*---\s*\n", tp_clean)
        if blk.strip()
    ]
    return parts or [tp_clean]


def _apply_clause_inplace(tp_a: str, tp_b: str, clause: ContrastClause) -> tuple[str, str]:
    a = clause.a.strip()
    b = clause.b.strip()
    if clause.is_soundscape:
        if "overall_soundscape:" not in tp_a:
            raise ValueError(
                "soundscape contrast requires an H3 prompt with an 'overall_soundscape:' section "
                "(not supported on single-paragraph p_h3_image prompts)"
            )
        if a and a in tp_a:
            if tp_a.count(a) > 1:
                raise ValueError(
                    f"soundscape pole_a {a!r} occurs {tp_a.count(a)} times in target_prompt; "
                    "include surrounding words so the clause matches a unique substring"
                )
            return tp_a, tp_b.replace(a, b, 1)
        if a and b in tp_b:
            if tp_b.count(b) > 1:
                raise ValueError(
                    f"soundscape pole_b {b!r} occurs {tp_b.count(b)} times in target_prompt; "
                    "include surrounding words so the clause matches a unique substring"
                )
            return tp_a.replace(b, a, 1), tp_b
        if a and b:
            return (
                _inject_soundscape_into_target(tp_a, a),
                _inject_soundscape_into_target(tp_b, b),
            )
        if b:
            return tp_a, _inject_soundscape_into_target(tp_b, b)
        raise ValueError("soundscape clause must specify a target soundscape description")

    if not a or not b:
        raise ValueError(
            f"in-place compilation requires an explicit 'pole_a -> pole_b' contrast "
            f"(got {b!r} without '->'); either pole_a or pole_b must match a substring in target_prompt"
        )
    if a == b:
        raise ValueError(f"pole_a and pole_b must differ, got {a!r}")
    if a in tp_a:
        if tp_a.count(a) > 1:
            raise ValueError(
                f"pole_a {a!r} occurs {tp_a.count(a)} times in target_prompt; "
                "include surrounding words so the clause matches a unique substring"
            )
        return tp_a, tp_b.replace(a, b, 1)
    if b in tp_b:
        if tp_b.count(b) > 1:
            raise ValueError(
                f"pole_b {b!r} occurs {tp_b.count(b)} times in target_prompt; "
                "include surrounding words so the clause matches a unique substring"
            )
        return tp_a.replace(b, a, 1), tp_b
    raise ValueError(
        f"neither pole_a ({a!r}) nor pole_b ({b!r}) was found as a substring in target_prompt; "
        "in-place compilation requires pole_a or pole_b to match text inside target_prompt"
    )


def _compile_hold_items(
    holds: Sequence[HoldClause],
    *,
    tp_scenes: Sequence[str],
    anchor_clauses: Sequence[ContrastClause],
    frame: str = "scene",
) -> tuple[_cv.HoldItem, ...]:
    if not holds:
        return ()
    out: list[_cv.HoldItem] = []
    idx = 0
    anchor = anchor_clauses[0] if anchor_clauses else None
    for hc in holds:
        ha = hc.a.strip()
        hb = hc.b.strip()
        if not hb:
            continue
        for tp in tp_scenes:
            if hc.is_soundscape and "overall_soundscape:" in tp:
                if ha and ha in tp:
                    h_tp_a = tp
                    h_tp_b = tp.replace(ha, hb, 1)
                elif ha and hb in tp:
                    h_tp_a = tp.replace(hb, ha, 1)
                    h_tp_b = tp
                elif ha:
                    h_tp_a = _inject_soundscape_into_target(tp, ha)
                    h_tp_b = _inject_soundscape_into_target(tp, hb)
                else:
                    h_tp_a = tp
                    h_tp_b = _inject_soundscape_into_target(tp, hb)
            elif ha and ha in tp:
                h_tp_a = tp
                h_tp_b = tp.replace(ha, hb, 1)
            elif ha and hb in tp:
                h_tp_a = tp.replace(hb, ha, 1)
                h_tp_b = tp
            elif anchor is not None and not anchor.is_soundscape:
                base_sub = anchor.a if anchor.a in tp else (anchor.b if anchor.b in tp else "")
                if not base_sub:
                    raise ValueError(
                        f"hold clause {ha!r} -> {hb!r} could not be anchored in target_prompt"
                    )
                eff_ha = ha if ha else base_sub
                h_tp_a = tp.replace(base_sub, eff_ha, 1)
                h_tp_b = tp.replace(base_sub, hb, 1)
            else:
                raise ValueError(
                    f"hold clause {ha!r} -> {hb!r} not found in target_prompt; "
                    "specify 'hold: <substring_in_prompt> -> <nuisance_variant>'"
                )
            h_a_span, h_b_span = _cv.differing_span(h_tp_a, h_tp_b, frame=frame)
            out.append(
                _cv.HoldItem(
                    index=idx,
                    group=hc.group or "nuisance",
                    weight=float(hc.weight),
                    a=h_tp_a,
                    b=h_tp_b,
                    a_span=h_a_span,
                    b_span=h_b_span,
                )
            )
            idx += 1
    return tuple(out)


def compile_short_contrast_pairs(
    pole_a: str,
    pole_b: str,
    *,
    mode: str = "t2va",
    target_prompt: str | None = None,
    target_weight: float = 1.0,
    extra_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    soundscape_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    **_unused_kwargs: Any,
) -> list[_cv.PairItem]:
    """Compile `(pole_a, pole_b)` (plus optional `extra_clauses` and `soundscape_clauses`)
    in-place across the scene prompt(s) in `target_prompt`."""
    norm_mode = normalize_compile_mode(mode)
    tp_scenes = _split_target_prompts(target_prompt)

    main_spec = f"{pole_a} ||| {pole_b}" if str(pole_a or "").strip() else str(pole_b or "").strip()
    vis_clauses, snd_clauses, _ = _parse_clauses_and_holds(
        main_spec,
        extra_clauses=extra_clauses,
        soundscape_clauses=soundscape_clauses,
    )
    if norm_mode == "p_h3_image" and snd_clauses:
        raise ValueError(
            "mode='p_h3_image' is single-frame image generation and has no overall_soundscape section"
        )
    all_clauses = [*vis_clauses, *snd_clauses]
    if not all_clauses:
        raise ValueError("at least one contrast clause is required")

    frame_kind = "scene"
    multi_span_ok = len(all_clauses) > 1
    pairs: list[_cv.PairItem] = []
    for idx, tp in enumerate(tp_scenes):
        tp_a = tp
        tp_b = tp
        for cl in all_clauses:
            tp_a, tp_b = _apply_clause_inplace(tp_a, tp_b, cl)
        a_span, b_span = _cv.differing_span(
            tp_a, tp_b, frame=frame_kind, allow_multi_span=multi_span_ok
        )
        w = float(target_weight) if idx == 0 else 1.0
        pairs.append(
            _cv.PairItem(
                index=idx,
                weight=w,
                a=tp_a,
                b=tp_b,
                a_span=a_span,
                b_span=b_span,
            )
        )
    return pairs


def compile_words_to_pairset(
    words: str,
    *,
    name: str | None = None,
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    mode: str = "t2va",
    holds: Sequence[HoldClause | Mapping[str, Any] | tuple[str, ...] | str] | None = None,
    extra_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    soundscape_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    target_prompt: str | None = None,
    target_weight: float = 1.0,
    **_unused_kwargs: Any,
) -> _cv.PairSet:
    """Compile a contrast specification (`'pole_a -> pole_b'`, semicolon/newline
    multi-clause spec, `soundscape:`, and `hold:`) in-place on `target_prompt`."""
    raw = str(words or "").strip()
    if not raw:
        raise ValueError("words must be a non-empty string")

    norm_mode = infer_compile_mode(
        mode if mode and mode != "auto" else None,
        target_prompt=target_prompt,
        words=raw,
    )
    tp_scenes = _split_target_prompts(target_prompt)
    vis_clauses, snd_clauses, parsed_holds = _parse_clauses_and_holds(
        raw,
        extra_clauses=extra_clauses,
        soundscape_clauses=soundscape_clauses,
        holds=holds,
    )
    if norm_mode == "p_h3_image" and snd_clauses:
        raise ValueError(
            "mode='p_h3_image' is single-frame image generation and has no overall_soundscape section"
        )
    all_clauses = [*vis_clauses, *snd_clauses]
    if not all_clauses:
        raise ValueError("at least one contrast clause is required")

    frame_kind = "scene"
    multi_span_ok = len(all_clauses) > 1
    pairs: list[_cv.PairItem] = []
    for idx, tp in enumerate(tp_scenes):
        tp_a = tp
        tp_b = tp
        for cl in all_clauses:
            tp_a, tp_b = _apply_clause_inplace(tp_a, tp_b, cl)
        a_span, b_span = _cv.differing_span(
            tp_a, tp_b, frame=frame_kind, allow_multi_span=multi_span_ok
        )
        w = float(target_weight) if idx == 0 else 1.0
        pairs.append(
            _cv.PairItem(
                index=idx,
                weight=w,
                a=tp_a,
                b=tp_b,
                a_span=a_span,
                b_span=b_span,
            )
        )

    hold_items = _compile_hold_items(
        parsed_holds,
        tp_scenes=tp_scenes,
        anchor_clauses=all_clauses,
        frame=frame_kind,
    )

    first_b = all_clauses[0].b
    resolved_name = (name or "").strip() or _slugify_name(first_b)
    if not _cv.NAME_RE.match(resolved_name):
        resolved_name = _slugify_name(resolved_name)

    a_parts: list[str] = []
    b_parts: list[str] = []
    for cl in all_clauses:
        if cl.is_soundscape and not cl.a:
            a_parts.append(_extract_existing_soundscape(tp_scenes[0]) or "baseline soundscape")
        else:
            a_parts.append(cl.a)
        b_parts.append(cl.b)
    resolved_a_side = (a_side or "").strip() or " | ".join(p for p in a_parts if p)
    resolved_b_side = (b_side or "").strip() or " | ".join(p for p in b_parts if p)
    resolved_quality = (quality or "").strip() or (
        f"In-place target-prompt contrast ({norm_mode}): {resolved_a_side} -> {resolved_b_side}"
    )

    digest_payload = json.dumps(
        {
            "name": resolved_name,
            "mode": norm_mode,
            "pairs": [(p.weight, p.a, p.b) for p in pairs],
            "holds": [(h.group, h.weight, h.a, h.b) for h in hold_items],
        },
        sort_keys=True,
    ).encode("utf-8")

    return _cv.PairSet(
        path=Path(f"<compiled:{resolved_name}.toml>"),
        sha256=hashlib.sha256(digest_payload).hexdigest(),
        name=resolved_name,
        quality=resolved_quality,
        a_side=resolved_a_side,
        b_side=resolved_b_side,
        mode=norm_mode,
        frame=frame_kind,
        exclude_shots=(),
        pairs=tuple(pairs),
        holds=hold_items,
    )


def parse_short_contrast_spec(
    spec_text: str,
    *,
    target_prompt: str | None = None,
    target_weight: float = 1.0,
    **_unused_kwargs: Any,
) -> dict[str, Any]:
    """Parse a multi-line short contrast specification (with optional `name:`,
    `quality:`, `a_side:`, `b_side:`, `mode:`, `target_prompt:`, `target_weight:`,
    `soundscape:`, and `hold:` lines) and compile it in-place on `target_prompt`."""
    raw = str(spec_text or "").strip()
    if not raw:
        raise ValueError("spec_text must be non-empty")

    meta: dict[str, str] = {}
    body_lines: list[str] = []
    header_keys = {
        "name",
        "quality",
        "a_side",
        "b_side",
        "mode",
        "target_prompt",
        "scene_prompt",
        "target_weight",
    }
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m_kv = re.match(r"^([a-zA-Z_]+)\s*:\s*(.+)$", s)
        if m_kv and m_kv.group(1).lower() in header_keys:
            meta[m_kv.group(1).lower()] = m_kv.group(2).strip()
        else:
            body_lines.append(s)

    eff_tp = target_prompt or meta.get("target_prompt") or meta.get("scene_prompt")
    eff_tw = float(meta["target_weight"]) if "target_weight" in meta else float(target_weight)
    eff_mode = infer_compile_mode(
        meta.get("mode"),
        target_prompt=eff_tp,
        words="\n".join(body_lines),
    )
    pair_set = compile_words_to_pairset(
        "\n".join(body_lines),
        name=meta.get("name"),
        quality=meta.get("quality"),
        a_side=meta.get("a_side"),
        b_side=meta.get("b_side"),
        mode=eff_mode,
        target_prompt=eff_tp,
        target_weight=eff_tw,
    )
    return {
        "name": pair_set.name,
        "quality": pair_set.quality,
        "a_side": pair_set.a_side,
        "b_side": pair_set.b_side,
        "mode": pair_set.mode,
        "pole_a": pair_set.a_side,
        "pole_b": pair_set.b_side,
        "pair_set": pair_set,
    }


def write_compiled_toml(
    ps: _cv.PairSet,
    out_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a compiled `PairSet` to a `.toml` pair file with the compiler header."""
    out = Path(out_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"compiled TOML output already exists (pass overwrite=True): {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _toml_str(val: str) -> str:
        return json.dumps(str(val), ensure_ascii=False)

    def _toml_multiline(val: str) -> str:
        s = str(val).replace('"""', '\\"\\"\\"')
        return f'"""\n{s}\n"""'

    has_multi_span = any(" | " in p.a_span or " | " in p.b_span for p in ps.pairs)
    lines = [
        f"# Compiled in-place via h3_pair_compiler (mode={ps.mode!r}, n_pairs={len(ps.pairs)})",
        "[vector]",
        f"name = {_toml_str(ps.name)}",
        f"quality = {_toml_str(ps.quality)}",
        f"a_side = {_toml_str(ps.a_side)}",
        f"b_side = {_toml_str(ps.b_side)}",
        f"mode = {_toml_str(ps.mode)}",
        f"frame = {_toml_str(ps.frame)}",
        f"exclude_shots = {json.dumps(list(ps.exclude_shots))}",
        *([ "allow_multi_span = true" ] if has_multi_span else []),
        "",
    ]
    for p in ps.pairs:
        lines.append("[[pair]]")
        if abs(float(p.weight) - 1.0) > 1e-6:
            lines.append(f"weight = {float(p.weight):.4g}")
        lines.append(f"a = {_toml_multiline(p.a)}")
        lines.append(f"b = {_toml_multiline(p.b)}")
        lines.append("")
    for h in ps.holds:
        lines.append("[[hold]]")
        lines.append(f"group = {_toml_str(h.group)}")
        if abs(float(h.weight) - 1.0) > 1e-6:
            lines.append(f"weight = {float(h.weight):.4g}")
        lines.append(f"a = {_toml_multiline(h.a)}")
        lines.append(f"b = {_toml_multiline(h.b)}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def compile_short_contrast_to_toml(
    pole_a: str,
    pole_b: str,
    out_path: str | Path,
    *,
    name: str | None = None,
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    mode: str = "t2va",
    holds: Sequence[HoldClause | Mapping[str, Any] | tuple[str, ...] | str] | None = None,
    extra_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    soundscape_clauses: Sequence[ContrastClause | tuple[str, str] | str] | None = None,
    target_prompt: str | None = None,
    target_weight: float = 1.0,
    overwrite: bool = True,
    **_unused_kwargs: Any,
) -> Path:
    """Compile `(pole_a, pole_b)` in-place on `target_prompt` and write a valid `.toml` pair file."""
    main_spec = f"{pole_a} ||| {pole_b}" if str(pole_a or "").strip() else str(pole_b or "").strip()
    ps = compile_words_to_pairset(
        main_spec,
        name=name or Path(out_path).stem,
        quality=quality,
        a_side=a_side,
        b_side=b_side,
        mode=mode,
        holds=holds,
        extra_clauses=extra_clauses,
        soundscape_clauses=soundscape_clauses,
        target_prompt=target_prompt,
        target_weight=target_weight,
    )
    return write_compiled_toml(ps, out_path, overwrite=overwrite)
