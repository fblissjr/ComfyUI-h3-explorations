"""Apply `old wording -> new wording` edits to an H3 prompt: plain text code, no model libraries.

One copy of the edit grammar, for three readers that must agree on it:

- `edit_guidance.py::MiniMaxH3PromptEdit`, which applies an edit when a graph runs;
- `h3_pair_compiler.py`, which compiles the same `a -> b` clauses into control-vector pairs;
- the static graph walkers (`workflows/prompts.py`, `bench/preflight_graph.py`), which must know the text
  an edit node hands to the encoder without running the graph.

It imports nothing but `re`, so a walker that only reads graphs does not load torch or ComfyUI to learn
what a prompt says. `control_vectors.py` takes `HEADER_TOKEN_RE` from here for the same reason.
"""

from __future__ import annotations

import re
from typing import Any

#: The arrow between the two sides of an edit or a contrast clause.
PAIR_SPLIT_RE = re.compile(r"\s*(?:\|\|\||-->|->|=>|::)\s*")
#: A line that aims its edit at the soundscape section.
SOUNDSCAPE_PREFIX_RE = re.compile(r"^(?:soundscape|audio|foley|sound)\s*:\s*(.+)$", re.IGNORECASE)
#: A line that aims its edit at the non-diegetic music section.
MUSIC_PREFIX_RE = re.compile(r"^(?:music|non_diegetic_music|score)\s*:\s*(.+)$", re.IGNORECASE)
#: A line that aims its edit at the summary section.
SUMMARY_PREFIX_RE = re.compile(r"^summary\s*:\s*(.+)$", re.IGNORECASE)
#: A line that aims its edit at the subject_definitions section.
SUBJECT_DEFS_PREFIX_RE = re.compile(r"^(?:subject_definitions|subjects)\s*:\s*(.+)$", re.IGNORECASE)
#: A line that aims its edit at the retention_analysis section.
RETENTION_PREFIX_RE = re.compile(r"^(?:retention_analysis|retention)\s*:\s*(.+)$", re.IGNORECASE)
#: A line that aims its edit at the main visual description section.
DESCRIPTION_PREFIX_RE = re.compile(
    r"^(?:description|detailed_description|integrated_multimodal_description)\s*:\s*(.+)$",
    re.IGNORECASE,
)
#: A directive line that rewrites `<Subject N>`'s `retention_analysis:` marker from `fully_preserved`
#: to `partially_preserved` so reference clothing is unlocked without hand-editing tags.
RELEASE_ATTIRE_RE = re.compile(
    r"^(?:release_attire|unlock_attire|release_clothing)(?:\s*:\s*(.+))?$",
    re.IGNORECASE,
)
#: Any section-scoped edit prefix (`<section>: ...`) or attire-release directive (`release_attire[: ...]`).
SECTION_EDIT_PREFIX_RE = re.compile(
    r"^(?:(?:soundscape|audio|foley|sound|music|non_diegetic_music|score|summary|"
    r"subject_definitions|subjects|retention_analysis|retention|"
    r"description|detailed_description|integrated_multimodal_description)\s*:\s*(.+)|"
    r"(?:release_attire|unlock_attire|release_clothing)(?:\s*:\s*(.+))?)$",
    re.IGNORECASE,
)
#: Section header line boundaries for locating a named H3 section body.
_SECTION_BOUNDARY_RE = re.compile(
    r"(?:^|\n)[ \t]*(integrated_multimodal_description|subject_definitions|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):"
)
#: A section header or a shot header of the sectioned prompt formats.
HEADER_TOKEN_RE = re.compile(
    r"\b(?:integrated_multimodal_description|subject_definitions|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):|"
    r"\[Shot\s+\d+\]"
)
_DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.S)
_TAG_RE = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>|\(S\d+(?:,\s*S\d+)*\)|</?d>|<\|[a-z_]+\|>")
#: Lines of a pair-authoring block that are metadata, not clauses, so a `;` in them splits nothing.
_METADATA_PREFIXES = ("name:", "quality:", "a_side:", "b_side:", "mode:", "target_prompt:", "scene_prompt:")


def section_span(text: str, section_name: str) -> tuple[int, int] | None:
    """Return `(body_start, body_end)` strictly after `<section_name>:` up to the next H3 section header."""
    matches = list(_SECTION_BOUNDARY_RE.finditer(text))
    for idx, m in enumerate(matches):
        if m.group(1) == section_name:
            b0 = m.end()
            b1 = matches[idx + 1].start() if (idx + 1) < len(matches) else len(text)
            return b0, b1
    return None


def resolve_section_name(text: str, section_name: str) -> str:
    """Resolve generic `'description'` to `'detailed_description'` or `'integrated_multimodal_description'`."""
    if section_name != "description":
        return section_name
    if section_span(text, "detailed_description") is not None:
        return "detailed_description"
    if section_span(text, "integrated_multimodal_description") is not None:
        return "integrated_multimodal_description"
    return "detailed_description"


def parse_section_prefix(raw: str) -> tuple[str | None, str | None, str]:
    """Return `(prefix_label, section_header_name, body)` for any section-scoped edit prefix."""
    m_snd = SOUNDSCAPE_PREFIX_RE.match(raw)
    if m_snd:
        return "soundscape:", "overall_soundscape", m_snd.group(1).strip()
    m_mus = MUSIC_PREFIX_RE.match(raw)
    if m_mus:
        return "music:", "non_diegetic_music", m_mus.group(1).strip()
    m_sum = SUMMARY_PREFIX_RE.match(raw)
    if m_sum:
        return "summary:", "summary", m_sum.group(1).strip()
    m_sub = SUBJECT_DEFS_PREFIX_RE.match(raw)
    if m_sub:
        return "subject_definitions:", "subject_definitions", m_sub.group(1).strip()
    m_ret = RETENTION_PREFIX_RE.match(raw)
    if m_ret:
        return "retention_analysis:", "retention_analysis", m_ret.group(1).strip()
    m_desc = DESCRIPTION_PREFIX_RE.match(raw)
    if m_desc:
        raw_head = raw.split(":", 1)[0].strip().lower()
        sec = raw_head if raw_head in ("detailed_description", "integrated_multimodal_description") else "description"
        return f"{raw_head}:", sec, m_desc.group(1).strip()
    return None, None, raw


def split_pole_pair(raw: str) -> tuple[str, str]:
    """`"a -> b"` as `(a, b)`, whitespace collapsed; `("", text)` when there is no arrow."""
    clean = " ".join(str(raw or "").strip().split())
    if not clean:
        raise ValueError("contrast clause must be non-empty")
    parts = PAIR_SPLIT_RE.split(clean, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", clean


def split_clause_statements(raw: str) -> list[str]:
    """One clause a line; a line of several `a -> b; c -> d` clauses is split, prose with a `;` is not."""
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[str] = []
    for ln in (ln.strip() for ln in text.splitlines() if ln.strip()):
        if ";" in ln and not ln.lower().startswith(_METADATA_PREFIXES):
            parts = [sub.strip() for sub in ln.split(";") if sub.strip()]
            n_delims = len(PAIR_SPLIT_RE.findall(ln))
            if all(bool(PAIR_SPLIT_RE.search(p)) or bool(SECTION_EDIT_PREFIX_RE.match(p)) for p in parts):
                out.extend(parts)
            elif n_delims <= 1:
                out.append(ln)
            else:
                raise ValueError(
                    "ambiguous ';' inside clause; put each 'pole_a -> pole_b' clause on its own line when prose contains semicolons"
                )
        else:
            out.append(ln)
    return out


def _inside_dialogue(text: str, start: int, end: int) -> bool:
    return any(m.start() < end and start < m.end() for m in _DIALOGUE_RE.finditer(text))


def _parse_release_attire_targets(arg: str | None) -> tuple[str, ...]:
    """Normalize `release_attire` target argument (`None`/`''` -> `('<Subject 1>',)`, `'all'` -> `('all',)`)."""
    raw = (arg or "").strip()
    if not raw:
        return ("<Subject 1>",)
    if raw.lower() in ("all", "all_subjects", "*"):
        return ("all",)
    out: list[str] = []
    for part in re.split(r"[,+]", raw):
        tok = part.strip()
        if not tok:
            continue
        m = re.match(r"^<?Subject\s+(\d+)>?$", tok, re.IGNORECASE)
        if not m:
            raise ValueError(
                f"release_attire target {tok!r} is invalid; expected '<Subject N>' (for example '<Subject 1>') or 'all'"
            )
        lbl = f"<Subject {int(m.group(1))}>"
        if lbl not in out:
            out.append(lbl)
    if not out:
        raise ValueError("release_attire requires at least one '<Subject N>' target or 'all'")
    return tuple(out)


def _rewrite_subject_attire_line(line: str, label: str) -> tuple[str, str, str]:
    """Rewrite `<Subject N> (...): fully_preserved - ...` to `partially_preserved` with clothing released."""
    m = re.match(
        rf"^([ \t]*{re.escape(label)}(?:\s*\([^)\n]*\))?\s*:\s*)(\w+)(\s*-\s*)(.+?)([ \t]*)$",
        line,
    )
    if not m:
        raise ValueError(
            f"release_attire on {label}: retention_analysis line does not match '<Subject N>: <marker> - ...': {line.strip()!r}"
        )
    prefix, marker, dash, body, trailing_ws = m.groups()
    if marker not in ("fully_preserved", "partially_preserved"):
        raise ValueError(
            f"release_attire on {label}: expected 'fully_preserved' or 'partially_preserved' in retention_analysis:, got {marker!r}"
        )
    core = body[:-1].rstrip() if body.endswith(".") else body.rstrip()
    already_released = bool(
        re.search(
            r"\b(?:clothing|wardrobe|attire|garments?)\b[^.;]*\b(?:not|change|changes|changed|replaces|replaced)\b"
            r"|\bonly the (?:wardrobe|clothing|attire|garments?)\b",
            core,
            re.IGNORECASE,
        )
    )
    if not already_released:
        core = re.sub(
            r",\s+([^,;]+?),?\s+and\s+(?:her\s+|his\s+|their\s+|its\s+|the\s+)?(?:clothing|wardrobe|attire)\b",
            r" and \1",
            core,
        )
        core = re.sub(
            r",?\s+and\s+(?:her\s+|his\s+|their\s+|its\s+|the\s+)?(?:clothing|wardrobe|attire)\b",
            "",
            core,
        )
        core = f"{core}; clothing is not retained and follows the setting of the scene"
    new_line = f"{prefix}partially_preserved{dash}{core}.{trailing_ws}"
    return new_line, line.strip(), new_line.strip()


def release_subject_attire_in_prompt(
    text: str,
    targets: tuple[str, ...] = ("<Subject 1>",),
) -> tuple[str, list[dict[str, Any]]]:
    """Rewrite the requested `<Subject N>` line(s) in `retention_analysis:` from `fully_preserved`
    to `partially_preserved` so reference attire is unlocked cleanly without touching structural tags."""
    span = section_span(text, "retention_analysis")
    if span is None:
        raise ValueError(
            "release_attire requires a ref2va prompt with a 'retention_analysis:' section"
        )
    b0, b1 = span
    sec_body = text[b0:b1]
    raw_lines = sec_body.splitlines(keepends=True)
    if targets == ("all",):
        wanted = [
            m.group(1)
            for ln in raw_lines
            for m in (re.match(r"^[ \t]*(<Subject\s+\d+>)", ln),)
            if m is not None
        ]
        if not wanted:
            raise ValueError("release_attire: retention_analysis: contains no '<Subject N>' lines")
    else:
        wanted = list(targets)

    records: list[dict[str, Any]] = []
    for lbl in wanted:
        matched_idxs = [
            idx
            for idx, ln in enumerate(raw_lines)
            if re.match(rf"^[ \t]*{re.escape(lbl)}(?:\s|:)", ln)
        ]
        if not matched_idxs:
            raise ValueError(
                f"release_attire: {lbl} was not found in retention_analysis:"
            )
        if len(matched_idxs) > 1:
            raise ValueError(
                f"release_attire: {lbl} appears {len(matched_idxs)} times in retention_analysis:"
            )
        li = matched_idxs[0]
        raw_ln = raw_lines[li]
        nl = "\n" if raw_ln.endswith("\n") else ""
        stripped_nl = raw_ln[:-1] if nl else raw_ln
        if stripped_nl.endswith("\r"):
            nl = "\r" + nl
            stripped_nl = stripped_nl[:-1]
        rewritten, old_clean, new_clean = _rewrite_subject_attire_line(stripped_nl, lbl)
        raw_lines[li] = rewritten + nl
        records.append(
            {
                "old": old_clean,
                "new": new_clean,
                "soundscape": False,
                "section": "retention_analysis",
                "old_words": len(old_clean.split()),
                "new_words": len(new_clean.split()),
            }
        )
    new_sec_body = "".join(raw_lines)
    return text[:b0] + new_sec_body + text[b1:], records


def apply_edits(
    prompt: str,
    edits: str,
    *,
    release_subject_attire: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply every `old -> new` line (and any `release_attire` directive or `release_subject_attire=True`
    toggle) to `prompt`. Returns the edited prompt and one record per applied change.

    Refuses, with the line named: a line with no arrow or an empty side; old wording that is absent or
    occurs more than once; an edit that touches dialogue, a tag, a speaker id, a section header or a shot
    header; a section-prefixed line whose old wording is not in that section."""
    text = str(prompt or "")
    if not text.strip():
        raise ValueError("the prompt is empty")
    lines = split_clause_statements(edits)
    if not lines and not release_subject_attire:
        raise ValueError("no edit given; write one 'old wording -> new wording' per line")
    records: list[dict[str, Any]] = []
    released_labels: set[str] = set()
    if release_subject_attire:
        text, rel_recs = release_subject_attire_in_prompt(text, ("<Subject 1>",))
        records.extend(rel_recs)
        released_labels.add("<Subject 1>")

    for raw in lines:
        m_rel = RELEASE_ATTIRE_RE.match(raw)
        if m_rel:
            targets = _parse_release_attire_targets(m_rel.group(1))
            if targets != ("all",):
                targets = tuple(t for t in targets if t not in released_labels)
            if targets:
                text, rel_recs = release_subject_attire_in_prompt(text, targets)
                records.extend(rel_recs)
                if targets == ("all",):
                    released_labels.add("all")
                else:
                    released_labels.update(targets)
            continue
        prefix_label, sec_name, body = parse_section_prefix(raw)
        old, new = split_pole_pair(body)
        if not old or not new:
            raise ValueError(f"edit {raw!r}: write it as 'old wording -> new wording'")
        if old == new:
            raise ValueError(f"edit {raw!r}: both sides are the same")
        for side, wording in (("old", old), ("new", new)):
            if _TAG_RE.search(wording) or HEADER_TOKEN_RE.search(wording):
                raise ValueError(f"edit {raw!r}: the {side} wording contains a tag, a speaker id or a header; "
                                 "edit only the prose around them")
        if sec_name is not None:
            resolved_sec = resolve_section_name(text, sec_name)
            span = section_span(text, resolved_sec)
            if span is None:
                if old in text:
                    raise ValueError(
                        f"edit {raw!r}: a {prefix_label!r} edit must replace wording inside the {resolved_sec} section"
                    )
                raise ValueError(
                    f"edit {raw!r}: the prompt has no {resolved_sec}: section for a {prefix_label!r} edit"
                )
            b0, b1 = span
            sec_body = text[b0:b1]
            sec_count = sec_body.count(old)
            if sec_count == 0:
                if old in text:
                    raise ValueError(
                        f"edit {raw!r}: a {prefix_label!r} edit must replace wording inside the {resolved_sec} section"
                    )
                raise ValueError(
                    f"edit {raw!r}: the old wording occurs 0 time(s) in the prompt; it must "
                    "occur exactly once (add the words that follow it to make it unique)"
                )
            if sec_count > 1:
                raise ValueError(
                    f"edit {raw!r}: the old wording occurs {sec_count} time(s) in the {resolved_sec} section; it must "
                    "occur exactly once (add the words that follow it to make it unique)"
                )
            at = b0 + sec_body.index(old)
            sec_name = resolved_sec
        else:
            count = text.count(old)
            if count != 1:
                raise ValueError(
                    f"edit {raw!r}: the old wording occurs {count} time(s) in the prompt; it must "
                    "occur exactly once (add the words that follow it to make it unique)"
                )
            at = text.index(old)
        if _inside_dialogue(text, at, at + len(old)):
            raise ValueError(f"edit {raw!r}: the old wording is inside <d>...</d>; dialogue is not edited here")
        text = text[:at] + new + text[at + len(old):]
        records.append({
            "old": old,
            "new": new,
            "soundscape": sec_name == "overall_soundscape",
            "section": sec_name,
            "old_words": len(old.split()),
            "new_words": len(new.split()),
        })
    return text, records
