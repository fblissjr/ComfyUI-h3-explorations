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
            if all(bool(PAIR_SPLIT_RE.search(p)) or bool(SOUNDSCAPE_PREFIX_RE.match(p)) for p in parts):
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


def apply_edits(prompt: str, edits: str) -> tuple[str, list[dict[str, Any]]]:
    """Apply every `old -> new` line of `edits` to `prompt`. Returns the edited prompt and one record a line.

    Refuses, with the line named: a line with no arrow or an empty side; old wording that is absent or
    occurs more than once; an edit that touches dialogue, a tag, a speaker id, a section header or a shot
    header; a `soundscape:` line whose old wording is not in the soundscape section."""
    text = str(prompt or "")
    if not text.strip():
        raise ValueError("the prompt is empty")
    lines = split_clause_statements(edits)
    if not lines:
        raise ValueError("no edit given; write one 'old wording -> new wording' per line")
    records: list[dict[str, Any]] = []
    for raw in lines:
        sound = SOUNDSCAPE_PREFIX_RE.match(raw)
        body = sound.group(1) if sound else raw
        old, new = split_pole_pair(body)
        if not old or not new:
            raise ValueError(f"edit {raw!r}: write it as 'old wording -> new wording'")
        if old == new:
            raise ValueError(f"edit {raw!r}: both sides are the same")
        for side, wording in (("old", old), ("new", new)):
            if _TAG_RE.search(wording) or HEADER_TOKEN_RE.search(wording):
                raise ValueError(f"edit {raw!r}: the {side} wording contains a tag, a speaker id or a header; "
                                 "edit only the prose around them")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"edit {raw!r}: the old wording occurs {count} time(s) in the prompt; it must "
                             "occur exactly once (add the words that follow it to make it unique)")
        at = text.index(old)
        if _inside_dialogue(text, at, at + len(old)):
            raise ValueError(f"edit {raw!r}: the old wording is inside <d>...</d>; dialogue is not edited here")
        if sound:
            marker = text.find("overall_soundscape:")
            music = text.find("non_diegetic_music:")
            if marker < 0 or at < marker or (0 <= music < at):
                raise ValueError(f"edit {raw!r}: a 'soundscape:' edit must replace wording inside the "
                                 "overall_soundscape section")
        text = text[:at] + new + text[at + len(old):]
        records.append({"old": old, "new": new, "soundscape": bool(sound),
                        "old_words": len(old.split()), "new_words": len(new.split())})
    return text, records
