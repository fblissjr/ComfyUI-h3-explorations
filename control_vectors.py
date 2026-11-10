"""Pure-PyTorch math, pair-file reader, and `.safetensors` I/O for H3 control vectors.

Implements the specification in `docs/research/control_vectors/2026-11-04_h3_control_vectors.md`:
- `load_pairs(path)`: parses and validates `prompt_bank/control_vectors/<name>.toml`,
  extracting and verifying the single contiguous differing clause per pair.
- `mean_difference`, `flip_null`, `random_null`, `hold_basis`, `project_out`,
  `mean_pairwise_cosine`, `per_pair_cosine`, `split_half_cosine`: computed in
  `torch.float64` and returned as `torch.float32`.
- `detect_family`, `safetensors_recipe_digest`, `save_vector`, `load_vector`,
  `check_vector_compatibility`, and `apply_post_refiner`: file format, family
  and recipe verification, and post-refiner masked addition.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import struct
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save as save_safetensors_bytes

FORMAT_VERSION = "h3-control-vector/1"
POST_REFINER_DIM = 5376
ENCODER_DIM = 5120

NAME_RE = re.compile(r"^[a-z0-9_]+$")
CUSTOM_FAMILY_RE = re.compile(r"^custom:[0-9a-f]{12}$")
LLM_VECTOR_KEY_RE = re.compile(r"^vector\.llm\.\d+$")

VALID_MODES = ("t2va", "fl2va", "i2va", "l2va", "ref2va", "p_h3_image")
VALID_FRAMES = ("scene", "bare")
VALID_POOLING = ("scene_text", "all_tag1", "from_first_diff", "diff_span")
VALID_MEDIA_POOLING = ("scene_text", "all_tag1", "vision_tokens", "all_rows")
VALID_ALL_POOLING = (
    "scene_text",
    "all_tag1",
    "from_first_diff",
    "diff_span",
    "vision_tokens",
    "all_rows",
)
DEFAULT_EXCLUDE_SHOTS: tuple[str, ...] = ("t2va_offpath_tortoise", "ref2va_look_anchor")

ALLOWED_TOP_KEYS = frozenset({"vector", "pair", "hold"})
ALLOWED_VECTOR_KEYS = frozenset(
    {"name", "quality", "a_side", "b_side", "mode", "frame", "exclude_shots", "allow_multi_span"}
)
ALLOWED_PAIR_KEYS = frozenset({"weight", "a", "b"})
ALLOWED_HOLD_KEYS = frozenset({"group", "weight", "a", "b"})

# Every H3 prompt section header (base_en.md and ref_en.md) plus the shot-header prefix.
SECTION_AND_SHOT_MARKERS: tuple[str, ...] = (
    "integrated_multimodal_description:",
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
    "[Shot ",
)
HEADER_TOKEN_RE = re.compile(
    r"\b(?:integrated_multimodal_description|subject_definitions|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):|"
    r"\[Shot\s+\d+\]"
)

#: Measured/reasoned ceiling on consecutive identical words inside a single modified
#: clause: two poles of one quality or action-predicate clause may share a short
#: preposition or compound noun phrase of up to 4 words (e.g. "with soft, warm" or
#: "the polished mahogany desk" in zero_gravity.toml), whereas 5 or more unchanged
#: words between edits indicates two separate clauses were modified.
MAX_SHARED_INTERIOR_WORDS = 5

FAMILY_TENSOR_KEYS: tuple[str, str, str] = (
    "condition_proj.weight",
    "condition_proj.bias",
    "token_refiner.final_norm.weight",
)

# Measured across all 7 local H3 diffusion checkpoints on disk (BF16 tensor sha256[:12]).
KNOWN_FAMILIES: dict[tuple[str, str, str], str] = {
    ("af3c58d7f2ad", "701e0673282f", "cf10a56a1137"): "fl2va",
    ("7c5485465531", "529e19c04005", "6ca615e4e3f8"): "ref2va",
    ("d6b17a52eb10", "2d8bca2b367d", "cf10a56a1137"): "fasth3",
}

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "format",
    "name",
    "quality",
    "pairs",
    "recipe",
    "family",
    "measured_under",
    "points",
    "quality_report",
    "provenance",
)

_REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PairItem:
    index: int
    weight: float
    a: str
    b: str
    a_span: str
    b_span: str


@dataclass(frozen=True)
class HoldItem:
    index: int
    group: str
    weight: float
    a: str
    b: str
    a_span: str
    b_span: str


@dataclass(frozen=True)
class PairSet:
    path: Path
    sha256: str
    name: str
    quality: str
    a_side: str
    b_side: str
    mode: str
    frame: str
    exclude_shots: tuple[str, ...]
    pairs: tuple[PairItem, ...]
    holds: tuple[HoldItem, ...]


def _multi_span_section_diffs(
    sec_a: str,
    sec_b: str,
    *,
    label: str = "pair",
) -> tuple[list[str], list[str]]:
    """Extract one or more contiguous differing clauses `(a_spans, b_spans)` inside a single section."""
    if HEADER_TOKEN_RE.findall(sec_a) != HEADER_TOKEN_RE.findall(sec_b):
        raise ValueError(
            f"{label}: differing span modifies a prompt section or [Shot N] header; "
            "scene-frame pairs must keep all section and shot headers identical"
        )
    words_a = sec_a.split()
    words_b = sec_b.split()
    if words_a == words_b:
        raise ValueError(f"{label}: 'a' and 'b' differ only in whitespace")

    matcher = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    opcodes = matcher.get_opcodes()
    clusters: list[list[tuple[str, int, int, int, int]]] = []
    cur: list[tuple[str, int, int, int, int]] = []
    for idx, op in enumerate(opcodes):
        tag, i1, i2, _j1, _j2 = op
        if tag != "equal":
            cur.append(op)
        elif cur:
            has_more_ne = any(o[0] != "equal" for o in opcodes[idx + 1 :])
            eq_str = " ".join(words_a[i1:i2])
            has_marker = any(m in eq_str for m in SECTION_AND_SHOT_MARKERS)
            if has_more_ne and (i2 - i1) < MAX_SHARED_INTERIOR_WORDS and not has_marker:
                cur.append(op)
            else:
                clusters.append(cur)
                cur = []
    if cur:
        clusters.append(cur)

    a_spans: list[str] = []
    b_spans: list[str] = []
    for cl in clusters:
        a0, a1 = cl[0][1], cl[-1][2]
        b0, b1 = cl[0][3], cl[-1][4]
        sub_a = " ".join(words_a[a0:a1])
        sub_b = " ".join(words_b[b0:b1])
        if not sub_a or not sub_b:
            raise ValueError(
                f"{label}: one side of a differing span is empty ({sub_a!r} vs {sub_b!r}); "
                "both 'a' and 'b' must state an explicit clause"
            )
        for boundary in SECTION_AND_SHOT_MARKERS:
            if boundary in sub_a or boundary in sub_b:
                raise ValueError(
                    f"{label}: differing span crosses or modifies section/shot header {boundary!r}; "
                    "scene-frame pairs must not modify section or shot headers"
                )
        a_spans.append(sub_a)
        b_spans.append(sub_b)
    return a_spans, b_spans


def differing_span(
    a: str,
    b: str,
    *,
    frame: str = "scene",
    label: str = "pair",
    allow_multi_span: bool = False,
) -> tuple[str, str]:
    """Return the single contiguous differing clause `(a_span, b_span)` between `a` and `b`
    (or `" | "`-joined clauses when `allow_multi_span=True` for declared in-place compiler edits).

    In `frame == 'scene'`, refuses if `a` and `b` differ only in whitespace, differ
    outside a single contiguous clause (when `allow_multi_span=False`), cross or modify
    a prompt section or `[Shot N]` boundary, or delete/insert without a non-empty clause
    on both sides."""
    if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
        raise ValueError(f"{label}: both 'a' and 'b' must be non-empty strings")
    if a == b:
        raise ValueError(f"{label}: 'a' and 'b' are identical; every pair must differ")

    words_a = a.split()
    words_b = b.split()
    if words_a == words_b:
        raise ValueError(f"{label}: 'a' and 'b' differ only in whitespace")

    if frame == "bare":
        return a.strip(), b.strip()

    if allow_multi_span:
        if HEADER_TOKEN_RE.findall(a) != HEADER_TOKEN_RE.findall(b):
            raise ValueError(
                f"{label}: differing span modifies a prompt section or [Shot N] header; "
                "scene-frame pairs must keep all section and shot headers identical"
            )
        blocks_a = a.split("\n\n")
        blocks_b = b.split("\n\n")
        if len(blocks_a) != len(blocks_b):
            raise ValueError(
                f"{label}: 'a' and 'b' have different paragraph counts ({len(blocks_a)} vs {len(blocks_b)})"
            )
        all_a_spans: list[str] = []
        all_b_spans: list[str] = []
        for idx_b, (ba, bb) in enumerate(zip(blocks_a, blocks_b)):
            if ba == bb:
                continue
            sa, sb = _multi_span_section_diffs(ba, bb, label=f"{label} (section {idx_b})")
            all_a_spans.extend(sa)
            all_b_spans.extend(sb)
        if not all_a_spans:
            raise ValueError(f"{label}: 'a' and 'b' differ only in whitespace")
        return " | ".join(all_a_spans), " | ".join(all_b_spans)

    # 1. Character-level common prefix and suffix check: ensures no paragraph break,
    #    section/shot boundary, or outside-clause whitespace edits exist.
    min_len = min(len(a), len(b))
    c_prefix = 0
    while c_prefix < min_len and a[c_prefix] == b[c_prefix]:
        c_prefix += 1
    c_suffix = 0
    while (
        c_suffix < (min_len - c_prefix)
        and a[len(a) - 1 - c_suffix] == b[len(b) - 1 - c_suffix]
    ):
        c_suffix += 1

    char_diff_a = a[c_prefix : len(a) - c_suffix]
    char_diff_b = b[c_prefix : len(b) - c_suffix]
    if "\n\n" in char_diff_a or "\n\n" in char_diff_b:
        if HEADER_TOKEN_RE.findall(a) == HEADER_TOKEN_RE.findall(b):
            blocks_a = a.split("\n\n")
            blocks_b = b.split("\n\n")
            if len(blocks_a) == len(blocks_b):
                diff_idxs = [idx_b for idx_b in range(len(blocks_a)) if blocks_a[idx_b] != blocks_b[idx_b]]
                if len(diff_idxs) == 2:
                    i0, i1 = diff_idxs
                    b0_a = blocks_a[i0].lstrip()
                    b1_a = blocks_a[i1].lstrip()
                    if (
                        b0_a.startswith(("integrated_multimodal_description:", "detailed_description:"))
                        and b1_a.startswith("overall_soundscape:")
                    ):
                        a_d, b_d = differing_span(
                            blocks_a[i0], blocks_b[i0], frame="scene", label=f"{label} (description)"
                        )
                        a_s, b_s = differing_span(
                            blocks_a[i1], blocks_b[i1], frame="scene", label=f"{label} (soundscape)"
                        )
                        return f"{a_d} | {a_s}", f"{b_d} | {b_s}"
        raise ValueError(
            f"{label}: differing span crosses a paragraph break; "
            "scene-frame pairs must change only one contiguous clause within a single section"
        )
    if HEADER_TOKEN_RE.findall(a) != HEADER_TOKEN_RE.findall(b):
        raise ValueError(
            f"{label}: differing span modifies a prompt section or [Shot N] header; "
            "scene-frame pairs must keep all section and shot headers identical"
        )
    for boundary in SECTION_AND_SHOT_MARKERS:
        if boundary in char_diff_a or boundary in char_diff_b:
            raise ValueError(
                f"{label}: differing span crosses or modifies section/shot header {boundary!r}; "
                "scene-frame pairs must change only one contiguous clause"
            )

    matcher = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    opcodes = matcher.get_opcodes()
    non_equal = [idx for idx, (tag, *_rest) in enumerate(opcodes) if tag != "equal"]
    first_ne, last_ne = non_equal[0], non_equal[-1]

    for idx in range(first_ne + 1, last_ne):
        tag, i1, i2, _j1, _j2 = opcodes[idx]
        if tag != "equal":
            continue
        shared_words = words_a[i1:i2]
        shared_str = " ".join(shared_words)
        if len(shared_words) >= MAX_SHARED_INTERIOR_WORDS:
            first_edit_a = " ".join(words_a[opcodes[first_ne][1] : opcodes[first_ne][2]])
            last_edit_a = " ".join(words_a[opcodes[last_ne][1] : opcodes[last_ne][2]])
            raise ValueError(
                f"{label}: 'a' and 'b' differ in multiple separate spans "
                f"({first_edit_a!r} ... {last_edit_a!r}) separated by {len(shared_words)} "
                f"unchanged words ({shared_str!r}); scene-frame pairs must change only one contiguous clause"
            )

    a_start, a_end = opcodes[first_ne][1], opcodes[last_ne][2]
    b_start, b_end = opcodes[first_ne][3], opcodes[last_ne][4]
    a_span = " ".join(words_a[a_start:a_end])
    b_span = " ".join(words_b[b_start:b_end])
    if not a_span or not b_span:
        raise ValueError(
            f"{label}: one side of the differing span is empty ({a_span!r} vs {b_span!r}); "
            "both 'a' and 'b' must state an explicit clause"
        )
    for boundary in SECTION_AND_SHOT_MARKERS:
        if boundary in a_span or boundary in b_span:
            raise ValueError(
                f"{label}: differing span crosses or modifies section/shot header {boundary!r}; "
                "scene-frame pairs must change only one contiguous clause"
            )

    # Check that the character-level diff sits strictly inside the word-level diff span
    # and introduces no newline changes.
    if (
        len(_normalize_ws(char_diff_a)) > len(a_span)
        or len(_normalize_ws(char_diff_b)) > len(b_span)
        or "\n" in char_diff_a
        or "\n" in char_diff_b
    ):
        raise ValueError(
            f"{label}: 'a' and 'b' have whitespace differences outside the differing clause "
            f"({a_span!r} -> {b_span!r})"
        )

    return a_span, b_span


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def load_pairs(path: str | Path, *, bank_dir: str | Path | None = None) -> PairSet:
    """Parse and validate a control-vector pair TOML file."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"pair file not found: {p}")
    raw_bytes = p.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = tomllib.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{p}: invalid TOML syntax: {exc}") from exc

    unknown_top = sorted(set(doc.keys()) - ALLOWED_TOP_KEYS)
    if unknown_top:
        raise ValueError(
            f"{p}: unknown top-level TOML key(s) {unknown_top}; expected only {sorted(ALLOWED_TOP_KEYS)}"
        )

    vec = doc.get("vector")
    if not isinstance(vec, dict):
        raise ValueError(f"{p}: missing required [vector] table")

    unknown_vec = sorted(set(vec.keys()) - ALLOWED_VECTOR_KEYS)
    if unknown_vec:
        raise ValueError(
            f"{p}: unknown key(s) in [vector]: {unknown_vec}; expected only {sorted(ALLOWED_VECTOR_KEYS)}"
        )

    name = vec.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"{p}: vector.name must match [a-z0-9_]+, got {name!r}")
    if p.stem != name:
        raise ValueError(f"{p}: vector.name {name!r} does not match file stem {p.stem!r}")

    for field_name in ("quality", "a_side", "b_side"):
        val = vec.get(field_name)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{p}: vector.{field_name} must be a non-empty string")

    mode = vec.get("mode")
    if mode not in VALID_MODES:
        raise ValueError(f"{p}: vector.mode must be one of {VALID_MODES}, got {mode!r}")

    frame = vec.get("frame")
    if frame not in VALID_FRAMES:
        raise ValueError(f"{p}: vector.frame must be one of {VALID_FRAMES}, got {frame!r}")

    excl_raw = vec.get("exclude_shots", [])
    if not isinstance(excl_raw, list) or not all(isinstance(x, str) and x.strip() for x in excl_raw):
        raise ValueError(f"{p}: vector.exclude_shots must be a list of non-empty shot ID strings")
    exclude_shots = tuple(excl_raw)

    resolved_bank_dir = Path(bank_dir) if bank_dir is not None else (_REPO_ROOT / "prompt_bank")
    excluded_texts: dict[str, str] = {}
    for shot_id in exclude_shots:
        shot_file = resolved_bank_dir / f"{shot_id}.txt"
        if not shot_file.is_file():
            raise ValueError(
                f"{p}: vector.exclude_shots names {shot_id!r}, which does not exist at {shot_file}"
            )
        excluded_texts[shot_id] = _normalize_ws(shot_file.read_text(encoding="utf-8"))

    raw_pairs = doc.get("pair")
    if not isinstance(raw_pairs, list) or len(raw_pairs) == 0:
        raise ValueError(f"{p}: must contain at least one [[pair]] entry")

    def _check_excluded(text: str, entry_label: str) -> None:
        norm = _normalize_ws(text)
        for shot_id, excl_norm in excluded_texts.items():
            if norm == excl_norm or excl_norm in norm:
                raise ValueError(
                    f"{p} {entry_label}: prompt equals or contains excluded shot {shot_id!r}"
                )

    allow_multi_span_raw = vec.get("allow_multi_span", False)
    if not isinstance(allow_multi_span_raw, bool):
        raise ValueError(f"{p}: vector.allow_multi_span must be a boolean, got {allow_multi_span_raw!r}")
    is_compiled_toml = bool(allow_multi_span_raw)
    pairs: list[PairItem] = []
    for idx, item in enumerate(raw_pairs):
        lbl = f"pair[{idx}]"
        if not isinstance(item, dict):
            raise ValueError(f"{p} {lbl}: must be a table")
        unknown_pair = sorted(set(item.keys()) - ALLOWED_PAIR_KEYS)
        if unknown_pair:
            raise ValueError(
                f"{p} {lbl}: unknown key(s) {unknown_pair}; expected only {sorted(ALLOWED_PAIR_KEYS)}"
            )
        w_raw = item.get("weight", 1.0)
        if isinstance(w_raw, bool) or not isinstance(w_raw, (int, float)):
            raise ValueError(f"{p} {lbl}: weight must be a finite non-zero number, got {w_raw!r}")
        w = float(w_raw)
        if not math.isfinite(w) or w == 0.0:
            raise ValueError(f"{p} {lbl}: weight must be finite and non-zero, got {w_raw!r}")
        a = item.get("a")
        b = item.get("b")
        a_span, b_span = differing_span(
            a, b, frame=frame, label=f"{p} {lbl}", allow_multi_span=is_compiled_toml
        )
        _check_excluded(a, f"{lbl}.a")
        _check_excluded(b, f"{lbl}.b")
        pairs.append(PairItem(index=idx, weight=w, a=a, b=b, a_span=a_span, b_span=b_span))

    raw_holds = doc.get("hold", [])
    if not isinstance(raw_holds, list):
        raise ValueError(f"{p}: 'hold' must be a list of [[hold]] tables")
    holds: list[HoldItem] = []
    for idx, item in enumerate(raw_holds):
        lbl = f"hold[{idx}]"
        if not isinstance(item, dict):
            raise ValueError(f"{p} {lbl}: must be a table")
        unknown_hold = sorted(set(item.keys()) - ALLOWED_HOLD_KEYS)
        if unknown_hold:
            raise ValueError(
                f"{p} {lbl}: unknown key(s) {unknown_hold}; expected only {sorted(ALLOWED_HOLD_KEYS)}"
            )
        grp = item.get("group")
        if not isinstance(grp, str) or not NAME_RE.match(grp):
            raise ValueError(f"{p} {lbl}: group must match [a-z0-9_]+, got {grp!r}")
        w_raw = item.get("weight", 1.0)
        if isinstance(w_raw, bool) or not isinstance(w_raw, (int, float)):
            raise ValueError(f"{p} {lbl}: weight must be a finite non-zero number, got {w_raw!r}")
        w = float(w_raw)
        if not math.isfinite(w) or w == 0.0:
            raise ValueError(f"{p} {lbl}: weight must be finite and non-zero, got {w_raw!r}")
        a = item.get("a")
        b = item.get("b")
        a_span, b_span = differing_span(a, b, frame=frame, label=f"{p} {lbl}")
        _check_excluded(a, f"{lbl}.a")
        _check_excluded(b, f"{lbl}.b")
        holds.append(
            HoldItem(index=idx, group=grp, weight=w, a=a, b=b, a_span=a_span, b_span=b_span)
        )

    return PairSet(
        path=p,
        sha256=file_sha256,
        name=name,
        quality=vec["quality"].strip(),
        a_side=vec["a_side"].strip(),
        b_side=vec["b_side"].strip(),
        mode=mode,
        frame=frame,
        exclude_shots=exclude_shots,
        pairs=tuple(pairs),
        holds=tuple(holds),
    )


_FENCE_OUTER_RE = re.compile(
    r"^[ \t]*(`{3,})[A-Za-z0-9_-]*[ \t]*\r?\n(.*)\r?\n[ \t]*\1[ \t]*$",
    flags=re.DOTALL,
)
_TEST_SCENE_HEADING_RE = re.compile(
    r"^[ \t]*###[ \t]+Test[ \t]+Scene[ \t]+Prompt\b[^\n]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_PAIR_HEADING_RE = re.compile(
    r"^[ \t]*####[ \t]+Pair[ \t]+(\d+)\b[^\n]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_HOLD_HEADING_RE = re.compile(
    r"^[ \t]*####[ \t]+Hold[ \t]+(\d+)(?:[ \t]*\(([^)\n]+)\))?[^\n]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_WEIGHT_TAG_RE = re.compile(
    r"\bweight\s*=\s*([+-]?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)
_PROMPT_A_HEADING_RE = re.compile(
    r"^[ \t]*#####[ \t]+Prompt[ \t]+A\b[^\n]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_PROMPT_B_HEADING_RE = re.compile(
    r"^[ \t]*#####[ \t]+Prompt[ \t]+B\b[^\n]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_SECTION_ONLY_RE = re.compile(
    r"\b(integrated_multimodal_description|subject_definitions|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):"
)


def _unwrap_markdown_fence(md_text: str) -> str:
    text = md_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    m_outer = _FENCE_OUTER_RE.match(text)
    if m_outer is not None:
        return m_outer.group(2).strip()
    if _PAIR_HEADING_RE.search(text) is None:
        for m_inner in re.finditer(
            r"^[ \t]*(`{3,})[A-Za-z0-9_-]*[ \t]*\n(.*?)\n[ \t]*\1[ \t]*$",
            text,
            flags=re.MULTILINE | re.DOTALL,
        ):
            candidate = m_inner.group(2).strip()
            if _PAIR_HEADING_RE.search(candidate) is not None:
                return candidate
    return text


def differing_section_info(a: str, b: str, *, frame: str = "scene") -> tuple[int, str]:
    """Return `(sec_idx, section_name)` for the prompt section where `a` and `b` differ."""
    if frame == "bare":
        return 0, "bare"
    min_len = min(len(a), len(b))
    c_prefix = 0
    while c_prefix < min_len and a[c_prefix] == b[c_prefix]:
        c_prefix += 1
    sec_matches = list(_SECTION_ONLY_RE.finditer(a))
    if not sec_matches:
        return 0, "text"
    chosen_idx = 0
    chosen_name = sec_matches[0].group(1)
    for idx, m in enumerate(sec_matches):
        if m.start() <= c_prefix:
            chosen_idx = idx
            chosen_name = m.group(1)
        else:
            break
    blocks_a = a.split("\n\n")
    blocks_b = b.split("\n\n")
    if len(blocks_a) == len(blocks_b):
        diff_idxs = [idx_b for idx_b in range(len(blocks_a)) if blocks_a[idx_b] != blocks_b[idx_b]]
        if len(diff_idxs) == 2:
            b0_a = blocks_a[diff_idxs[0]].lstrip()
            b1_a = blocks_a[diff_idxs[1]].lstrip()
            if (
                b0_a.startswith(("integrated_multimodal_description:", "detailed_description:"))
                and b1_a.startswith("overall_soundscape:")
            ):
                return chosen_idx, f"{chosen_name}+overall_soundscape"
    return chosen_idx, chosen_name


def parse_markdown_pairs(md_text: str) -> dict[str, Any]:
    """Parse an LLM-authored control-vector Markdown document into structured data.

    Strips an outer fenced code block (e.g. ```` ```text ```` or ```` ```markdown ````)
    when present, extracts the optional `### Test Scene Prompt` section and all
    `#### Pair N` (`##### Prompt A` / `##### Prompt B`) blocks, and auto-detects
    `mode` (`'ref2va'` when `subject_definitions:` appears in `test_scene_prompt`
    or any pair prompt, else `'t2va'`)."""
    if not isinstance(md_text, str) or not md_text.strip():
        raise ValueError("markdown text must be a non-empty string")

    body = _unwrap_markdown_fence(md_text)

    test_scene_prompt = ""
    m_test = _TEST_SCENE_HEADING_RE.search(body)
    if m_test is not None:
        after_test = body[m_test.end() :]
        m_next = re.search(
            r"^[ \t]*(?:###[ \t]+|####[ \t]+Pair\b)",
            after_test,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        raw_test = after_test[: m_next.start()] if m_next is not None else after_test
        test_scene_prompt = raw_test.strip()

    pair_matches = list(_PAIR_HEADING_RE.finditer(body))
    if not pair_matches:
        raise ValueError("no '#### Pair N' blocks found in markdown")

    pairs: list[dict[str, Any]] = []
    for idx, m_pair in enumerate(pair_matches):
        pair_num = int(m_pair.group(1))
        start = m_pair.end()
        end = pair_matches[idx + 1].start() if idx + 1 < len(pair_matches) else len(body)
        block = body[start:end]

        # Trim any trailing top/mid-level heading, Hold heading, or horizontal rule after the final pair.
        m_tail = re.search(
            r"^[ \t]*(?:###?[ \t]+|####[ \t]+Hold\b|---+[ \t]*$)",
            block,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if m_tail is not None:
            block = block[: m_tail.start()]

        ma_list = list(_PROMPT_A_HEADING_RE.finditer(block))
        mb_list = list(_PROMPT_B_HEADING_RE.finditer(block))
        if not ma_list:
            raise ValueError(f"Pair {pair_num}: missing '##### Prompt A' heading")
        if len(ma_list) > 1:
            raise ValueError(f"Pair {pair_num}: duplicate '##### Prompt A' heading")
        if not mb_list:
            raise ValueError(f"Pair {pair_num}: missing '##### Prompt B' heading")
        if len(mb_list) > 1:
            raise ValueError(f"Pair {pair_num}: duplicate '##### Prompt B' heading")

        ma, mb = ma_list[0], mb_list[0]
        if mb.start() <= ma.start():
            raise ValueError(
                f"Pair {pair_num}: '##### Prompt B' appears before '##### Prompt A'"
            )

        a_txt = block[ma.end() : mb.start()].strip()
        b_txt = block[mb.end() :].strip()
        if not a_txt:
            raise ValueError(f"Pair {pair_num}: '##### Prompt A' body is empty")
        if not b_txt:
            raise ValueError(f"Pair {pair_num}: '##### Prompt B' body is empty")

        m_w = _WEIGHT_TAG_RE.search(m_pair.group(0))
        p_weight = float(m_w.group(1)) if m_w is not None else 1.0
        pairs.append(
            {
                "index": idx,
                "number": pair_num,
                "weight": p_weight,
                "a": a_txt,
                "b": b_txt,
            }
        )

    hold_matches = list(_HOLD_HEADING_RE.finditer(body))
    holds: list[dict[str, Any]] = []
    for idx, m_hold in enumerate(hold_matches):
        hold_num = int(m_hold.group(1))
        raw_group = (m_hold.group(2) or "").strip().lower()
        group_name = re.sub(r"[^a-z0-9_]+", "_", raw_group).strip("_") or "default_hold"
        start = m_hold.end()
        end = hold_matches[idx + 1].start() if idx + 1 < len(hold_matches) else len(body)
        block = body[start:end]

        m_tail = re.search(
            r"^[ \t]*(?:###?[ \t]+|####[ \t]+Pair\b|---+[ \t]*$)",
            block,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if m_tail is not None:
            block = block[: m_tail.start()]

        ma_list = list(_PROMPT_A_HEADING_RE.finditer(block))
        mb_list = list(_PROMPT_B_HEADING_RE.finditer(block))
        if not ma_list:
            raise ValueError(f"Hold {hold_num}: missing '##### Prompt A' heading")
        if len(ma_list) > 1:
            raise ValueError(f"Hold {hold_num}: duplicate '##### Prompt A' heading")
        if not mb_list:
            raise ValueError(f"Hold {hold_num}: missing '##### Prompt B' heading")
        if len(mb_list) > 1:
            raise ValueError(f"Hold {hold_num}: duplicate '##### Prompt B' heading")

        ma, mb = ma_list[0], mb_list[0]
        if mb.start() <= ma.start():
            raise ValueError(
                f"Hold {hold_num}: '##### Prompt B' appears before '##### Prompt A'"
            )

        a_txt = block[ma.end() : mb.start()].strip()
        b_txt = block[mb.end() :].strip()
        if not a_txt:
            raise ValueError(f"Hold {hold_num}: '##### Prompt A' body is empty")
        if not b_txt:
            raise ValueError(f"Hold {hold_num}: '##### Prompt B' body is empty")

        holds.append(
            {
                "index": idx,
                "number": hold_num,
                "group": group_name,
                "a": a_txt,
                "b": b_txt,
            }
        )

    has_ref2va_headers = "subject_definitions:" in test_scene_prompt or any(
        "subject_definitions:" in p["a"] or "subject_definitions:" in p["b"] for p in pairs
    )
    detected_mode = "ref2va" if has_ref2va_headers else "t2va"

    return {
        "test_scene_prompt": test_scene_prompt,
        "mode": detected_mode,
        "pairs": pairs,
        "holds": holds,
    }


def _format_toml_multiline(text: str) -> str:
    """Format `text` as a TOML multiline string (`'''...'''`, falling back to
    escaped `\"\"\"...\"\"\"` if `'''` or a trailing `'` is present)."""
    if "'''" not in text and not text.endswith("'"):
        return f"'''{text}'''"
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"""', '\\"\\"\\"')
    )
    if escaped.endswith('"'):
        escaped = escaped[:-1] + '\\"'
    return f'"""{escaped}"""'


def write_markdown_to_toml(
    md_source: str | Path,
    out_path: str | Path,
    *,
    name: str | None = None,
    mode: str | None = None,
    frame: str = "scene",
    quality: str | None = None,
    a_side: str | None = None,
    b_side: str | None = None,
    exclude_shots: Sequence[str] | None = None,
    test_prompt_out: str | Path | None = None,
    overwrite: bool = False,
    bank_dir: str | Path | None = None,
) -> Path:
    """Convert an LLM-authored control-vector Markdown file or string to a validated `.toml` file."""
    source_path: Path | None = None
    if isinstance(md_source, Path):
        if not md_source.is_file():
            raise ValueError(f"markdown file not found: {md_source}")
        source_path = md_source
        md_text = md_source.read_text(encoding="utf-8")
    elif isinstance(md_source, str):
        if "\n" not in md_source and "####" not in md_source:
            cand = Path(md_source)
            if not cand.is_file():
                raise ValueError(f"markdown file not found: {cand}")
            source_path = cand
            md_text = cand.read_text(encoding="utf-8")
        else:
            md_text = md_source
    else:
        raise ValueError("md_source must be a Markdown string or Path")

    if frame not in VALID_FRAMES:
        raise ValueError(f"frame must be one of {VALID_FRAMES}, got {frame!r}")

    parsed = parse_markdown_pairs(md_text)
    pairs = parsed["pairs"]
    holds = list(parsed.get("holds") or ())
    test_scene_prompt: str = parsed["test_scene_prompt"]

    resolved_mode = (mode.strip() if mode is not None and mode.strip() else parsed["mode"])
    if resolved_mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {resolved_mode!r}")

    out_p = Path(out_path)
    if out_p.suffix == ".toml":
        target_toml = out_p
        resolved_name = (name.strip() if name is not None and name.strip() else target_toml.stem)
    else:
        if name is not None and name.strip():
            resolved_name = name.strip()
        elif source_path is not None:
            resolved_name = source_path.stem
        else:
            raise ValueError(
                "name is required when out_path does not end with '.toml' and md_source is raw text"
            )
        target_toml = out_p / f"{resolved_name}.toml"

    if not NAME_RE.match(resolved_name):
        raise ValueError(f"vector.name must match [a-z0-9_]+, got {resolved_name!r}")
    if target_toml.stem != resolved_name:
        raise ValueError(
            f"vector.name {resolved_name!r} does not match output file stem {target_toml.stem!r}"
        )
    if not overwrite and target_toml.exists():
        raise ValueError(
            f"pair TOML file {target_toml.name!r} already exists at {target_toml} "
            "(pass overwrite=True / --overwrite to replace it)"
        )

    # Validate Pair 1 first and auto-derive default a_side, b_side, and quality when omitted.
    p1_a_span, p1_b_span = differing_span(
        pairs[0]["a"], pairs[0]["b"], frame=frame, label=f"pair[0] (Pair {pairs[0]['number']})"
    )
    resolved_a_side = (a_side.strip() if a_side is not None and a_side.strip() else p1_a_span)
    resolved_b_side = (b_side.strip() if b_side is not None and b_side.strip() else p1_b_span)
    resolved_quality = (
        quality.strip()
        if quality is not None and quality.strip()
        else f"{resolved_b_side} vs. {resolved_a_side}"
    )

    # Validate all remaining pairs and holds before touching disk.
    for p_item in pairs[1:]:
        differing_span(
            p_item["a"],
            p_item["b"],
            frame=frame,
            label=f"pair[{p_item['index']}] (Pair {p_item['number']})",
        )
    for h_item in holds:
        differing_span(
            h_item["a"],
            h_item["b"],
            frame=frame,
            label=f"hold[{h_item['index']}] (Hold {h_item['number']})",
        )

    resolved_bank_dir = Path(bank_dir) if bank_dir is not None else (_REPO_ROOT / "prompt_bank")
    if exclude_shots is not None:
        resolved_exclude_shots = tuple(exclude_shots)
    else:
        resolved_exclude_shots = tuple(
            shot_id
            for shot_id in DEFAULT_EXCLUDE_SHOTS
            if (resolved_bank_dir / f"{shot_id}.txt").is_file()
        )

    if test_prompt_out is not None:
        if not test_scene_prompt:
            raise ValueError(
                "cannot write --test-prompt-out: markdown input has no '### Test Scene Prompt' section"
            )

    lines: list[str] = []
    if test_scene_prompt:
        lines.append("# Test Scene Prompt:")
        for raw_line in test_scene_prompt.splitlines():
            lines.append(f"# {raw_line}" if raw_line.strip() else "#")
        lines.append("")

    excl_toml = "[" + ", ".join(json.dumps(s, ensure_ascii=False) for s in resolved_exclude_shots) + "]"
    lines.extend(
        [
            "[vector]",
            f"name = {json.dumps(resolved_name, ensure_ascii=False)}",
            f"quality = {json.dumps(resolved_quality, ensure_ascii=False)}",
            f"a_side = {json.dumps(resolved_a_side, ensure_ascii=False)}",
            f"b_side = {json.dumps(resolved_b_side, ensure_ascii=False)}",
            f"mode = {json.dumps(resolved_mode, ensure_ascii=False)}",
            f"frame = {json.dumps(frame, ensure_ascii=False)}",
            f"exclude_shots = {excl_toml}",
            "",
        ]
    )

    for idx, p_item in enumerate(pairs):
        lines.append("[[pair]]")
        p_w = float(p_item.get("weight", 1.0))
        if round(p_w, 6) != 1.0:
            lines.append(f"weight = {p_w:g}")
        lines.append(f"a = {_format_toml_multiline(p_item['a'])}")
        lines.append(f"b = {_format_toml_multiline(p_item['b'])}")
        if idx + 1 < len(pairs) or holds:
            lines.append("")

    for idx, h_item in enumerate(holds):
        lines.append("[[hold]]")
        lines.append(f"group = {json.dumps(h_item['group'], ensure_ascii=False)}")
        lines.append(f"a = {_format_toml_multiline(h_item['a'])}")
        lines.append(f"b = {_format_toml_multiline(h_item['b'])}")
        if idx + 1 < len(holds):
            lines.append("")

    toml_content = "\n".join(lines) + "\n"
    target_toml.parent.mkdir(parents=True, exist_ok=True)
    had_existing = target_toml.exists()
    target_toml.write_text(toml_content, encoding="utf-8")
    try:
        load_pairs(target_toml, bank_dir=bank_dir)
    except Exception:
        if not had_existing and target_toml.is_file():
            target_toml.unlink(missing_ok=True)
        raise

    if test_prompt_out is not None:
        tp_path = Path(test_prompt_out)
        tp_path.parent.mkdir(parents=True, exist_ok=True)
        tp_path.write_text(test_scene_prompt + "\n", encoding="utf-8")

    return target_toml


_BANNED_FILLER_TERMS: tuple[str, ...] = (
    "slowly",
    "gradually",
    "steadily",
    "progressively",
    "rapidly",
    "increasingly",
    "begins to",
    "starts to",
)

_CANCEL_STOPLIST: frozenset[str] = frozenset({
    "with", "from", "into", "onto", "over", "under", "through", "across",
    "along", "around", "behind", "between", "beneath", "beside", "against",
    "toward", "towards", "while", "where", "which", "their", "there", "these",
    "those", "this", "that", "both", "each", "every", "some", "more", "most",
    "very", "just", "only", "still", "even", "back", "down", "away", "open",
    "shut", "stay", "stays", "keep", "keeps", "hold", "holds", "remain",
    "remains", "rest", "rests", "before", "after", "above", "below", "underneath",
    "inside", "outside", "without", "within", "about", "until", "upon",
    "shot", "filmed", "captured", "recorded", "photographed", "color", "colour",
    "light", "lighting", "shadow", "shadows", "focus", "optics", "camera",
    "motion", "acoustics", "vocal", "voice", "reverb", "tones", "surfaces",
    "surface",
})

_DIALOGUE_TAG_CHECK_RE = re.compile(r"<d>(?!\s*\[[A-Za-z][A-Za-z -]*\])")


def lint_pair_set(
    pairs: Any,
    *,
    frame: str = "scene",
    allow_repeated_spans: bool = False,
) -> list[str]:
    """Check semantic pair-quality rules across a `PairSet` or sequence of pair items/dicts.

    Returns a list of human-readable advisory strings (empty when clean):
    1. 100% unique `a_span` and `b_span` across all pairs (unless `allow_repeated_spans=True`
       or `pairs` is an in-memory `<compiled:...>` `PairSet` where a single target phrase is
       slotted across distinct canonical scene frames with shared lead-in wrappers).
    2. Every `<d>...</d>` dialogue block starts with a `[Language]` tag.
    3. Zero rate/onset filler adverbs (`slowly`, `gradually`, `begins to`, ...) in `b_span`.
    4. Zero content-noun cancellation (`>=4`-char content words appearing in both `a_span` and `b_span`).
    5. `non_diegetic_music: N/A` instead of `'No non-diegetic music.'` on scene-framed pairs."""
    if isinstance(pairs, PairSet):
        frame = pairs.frame
        items = list(pairs.pairs)
        if str(pairs.path).startswith("<compiled:"):
            allow_repeated_spans = True
    elif isinstance(pairs, Sequence):
        items = list(pairs)
    else:
        raise ValueError("pairs must be a PairSet or a sequence of pair items/dicts")

    warnings: list[str] = []
    seen_a_spans: dict[tuple[str, str], int] = {}
    seen_b_spans: dict[tuple[str, str], int] = {}

    for idx, item in enumerate(items):
        if isinstance(item, Mapping):
            num = int(item.get("number", item.get("index", idx) + 1))
            a_txt = str(item.get("a", ""))
            b_txt = str(item.get("b", ""))
            a_sp = item.get("a_span")
            b_sp = item.get("b_span")
            sec_name = item.get("section")
        else:
            num = int(getattr(item, "index", idx)) + 1
            a_txt = str(getattr(item, "a", ""))
            b_txt = str(getattr(item, "b", ""))
            a_sp = getattr(item, "a_span", None)
            b_sp = getattr(item, "b_span", None)
            sec_name = None

        if a_sp is None or b_sp is None:
            a_sp, b_sp = differing_span(a_txt, b_txt, frame=frame, label=f"Pair {num}")
        if sec_name is None:
            _sec_idx, sec_name = differing_section_info(a_txt, b_txt, frame=frame)

        norm_a = " ".join(str(a_sp).lower().split())
        norm_b = " ".join(str(b_sp).lower().split())
        if allow_repeated_spans:
            ctx_a = a_txt
            for part_a in str(a_sp).split(" | "):
                ctx_a = ctx_a.replace(part_a, "", 1)
            ctx_b = b_txt
            for part_b in str(b_sp).split(" | "):
                ctx_b = ctx_b.replace(part_b, "", 1)
            norm_ctx_a = " ".join(ctx_a.lower().split())
            norm_ctx_b = " ".join(ctx_b.lower().split())
            key_a = (norm_a, norm_ctx_a)
            key_b = (norm_b, norm_ctx_b)
        else:
            key_a = (norm_a, "")
            key_b = (norm_b, "")
        if key_a in seen_a_spans:
            warnings.append(
                f"Pair {num}: duplicate a_span (matches Pair {seen_a_spans[key_a]}): {a_sp!r}"
            )
        else:
            seen_a_spans[key_a] = num

        if key_b in seen_b_spans:
            warnings.append(
                f"Pair {num}: duplicate b_span (matches Pair {seen_b_spans[key_b]}): {b_sp!r}"
            )
        else:
            seen_b_spans[key_b] = num

        if _DIALOGUE_TAG_CHECK_RE.search(a_txt) or _DIALOGUE_TAG_CHECK_RE.search(b_txt):
            warnings.append(
                f"Pair {num}: <d> dialogue span is missing a [Language] tag (expected <d>[English] ...</d>)"
            )

        for side_lbl, side_txt in (("a", a_txt), ("b", b_txt)):
            sp_nums = [int(m) for m in re.findall(r"\(S(\d+)\)", side_txt)]
            n_dlg = len(re.findall(r"<d>", side_txt))
            if sp_nums and max(sp_nums) > max(1, n_dlg) and not re.search(r"\(S\d+\s*,\s*S\d+\)", side_txt):
                warnings.append(
                    f"Pair {num} ({side_lbl}): assigns speaker ID (S{max(sp_nums)}) when only {n_dlg} <d>...</d> vocalization span(s) exist; silent characters must not carry an (Sn) tag"
                )
                break

        b_lower = str(b_sp).lower()
        a_lower = str(a_sp).lower()
        for term in _BANNED_FILLER_TERMS:
            pat = r"\b" + re.escape(term) + r"\b"
            if re.search(pat, b_lower) and not re.search(pat, a_lower):
                warnings.append(
                    f"Pair {num}: b_span contains rate/onset filler {term!r}; state the target physical action directly"
                )

        words_a = {
            w for w in re.findall(r"[a-z]{4,}", a_lower) if w not in _CANCEL_STOPLIST
        }
        words_b = {
            w for w in re.findall(r"[a-z]{4,}", b_lower) if w not in _CANCEL_STOPLIST
        }
        shared_words = sorted(words_a & words_b)
        if shared_words:
            warnings.append(
                f"Pair {num}: content word(s) {shared_words} appear in both a_span and b_span and cancel in B - A; move shared nouns outside the diff span"
            )

        if frame == "scene" and sec_name != "non_diegetic_music":
            if "no non-diegetic music" in a_txt.lower() or "no non-diegetic music" in b_txt.lower():
                warnings.append(
                    f"Pair {num}: uses 'No non-diegetic music.' instead of 'N/A' under non_diegetic_music:"
                )

    return warnings


def _check_pair_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            f"a and b must be 2D tensors of identical shape [n, d], got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    n, d = a.shape
    if n < 1 or d < 1:
        raise ValueError(f"a and b must be non-empty [n, d], got {tuple(a.shape)}")
    a64 = a.detach().to(dtype=torch.float64)
    b64 = b.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(a64).all()) or not bool(torch.isfinite(b64).all()):
        raise ValueError("a and b must contain only finite values")

    if weights is None:
        w64 = torch.ones(n, dtype=torch.float64, device=a64.device)
    else:
        w_t = weights if isinstance(weights, torch.Tensor) else torch.as_tensor(weights, dtype=torch.float64, device=a64.device)
        if w_t.ndim != 1 or w_t.shape[0] != n:
            raise ValueError(f"weights must have shape [{n}], got {tuple(w_t.shape)}")
        w64 = w_t.detach().to(device=a64.device, dtype=torch.float64)
        if not bool(torch.isfinite(w64).all()) or bool((w64 == 0).any()):
            raise ValueError("weights must be finite and non-zero")
    return a64, b64, w64


def mean_difference(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    unit_mean: bool = False,
) -> torch.Tensor:
    """Weighted mean difference `sum(w_i * (B_i - A_i)) / sum(|w_i|)` in float64,
    returned as float32 `Tensor[d]`.

    When `unit_mean=True`, each pair's difference `B_i - A_i` is normalized to unit
    L2 length before taking the weighted average direction, then scaled by the
    weighted mean pair L2 norm so every pair contributes equal directional weight
    while preserving `mean_pair_difference` magnitude units."""
    a64, b64, w64 = _check_pair_tensors(a, b, weights)
    diffs = b64 - a64
    denom = w64.abs().sum()
    if not unit_mean:
        vec = (w64[:, None] * diffs).sum(dim=0) / denom
        return vec.to(dtype=torch.float32)

    norms = torch.linalg.vector_norm(diffs, dim=-1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("cannot compute unit_mean: at least one pair difference has zero norm")
    unit_diffs = diffs / norms[:, None]
    mean_dir = (w64[:, None] * unit_diffs).sum(dim=0) / denom
    mean_norm = (w64.abs() * norms).sum() / denom
    vec = mean_dir * mean_norm
    return vec.to(dtype=torch.float32)


def flip_null(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Partition the pairs into two disjoint random halves (`n - n // 2` and `n // 2`),
    take `0.5 * (half_pos - half_neg)` where each half is normalized by its own
    `sum(|w_i|)` over the signed differences `w_i * (B_i - A_i)`, and rescale to
    `mean_difference(a, b, weights)`'s Euclidean norm.

    Cancels any shared direction identically for even or odd `n` and equal,
    positive-unequal, or mixed-sign weights without dropping a pair."""
    a64, b64, w64 = _check_pair_tensors(a, b, weights)
    n, _d = a64.shape
    if n < 2:
        raise ValueError(f"flip_null requires at least 2 pairs, got n={n}")

    diffs = b64 - a64
    denom = w64.abs().sum()
    true_vec = (w64[:, None] * diffs).sum(dim=0) / denom
    target_norm = torch.linalg.vector_norm(true_vec)
    if float(target_norm) == 0.0:
        raise ValueError("cannot construct flip_null: mean_difference has zero norm")

    gen_dev = generator.device if generator is not None else torch.device("cpu")
    perm = torch.randperm(n, generator=generator, device=gen_dev).to(a64.device)
    signs = torch.ones(n, dtype=torch.float64, device=a64.device)
    signs[perm[: n // 2]] = -1.0
    # Signed differences aligned with the pair weight direction (`w_i * (B_i - A_i)`):
    # each random half (`signs > 0` of size n - n//2, `signs < 0` of size n//2) is
    # normalized by its own weight sum W_+ and W_-, so E[half_+] = mu and E[half_-] = mu
    # and 0.5 * (half_+ - half_-) cancels the shared direction mu identically for all n
    # (even or odd) and all weights without dropping a pair or subtracting the sample mean.
    abs_w = w64.abs()
    signed_diffs = (w64[:, None] * diffs)
    pos_mask = signs > 0
    neg_mask = signs < 0
    w_pos = abs_w[pos_mask].sum()
    w_neg = abs_w[neg_mask].sum()
    if float(w_pos.item()) <= 1e-12 or float(w_neg.item()) <= 1e-12:
        raise ValueError("flip_null encountered a half with zero weight sum")
    half_pos = signed_diffs[pos_mask].sum(dim=0) / w_pos
    half_neg = signed_diffs[neg_mask].sum(dim=0) / w_neg
    flipped_vec = 0.5 * (half_pos - half_neg)

    flipped_norm = torch.linalg.vector_norm(flipped_vec)
    if float(flipped_norm) == 0.0:
        raise ValueError("flip_null produced a zero vector before rescaling")
    rescaled = flipped_vec * (target_norm / flipped_norm)
    return rescaled.to(dtype=torch.float32)


def random_null(
    norm: float,
    d: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Isotropic random direction in `R^d` scaled to Euclidean length `norm`."""
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"norm must be finite and positive, got {norm!r}")
    if d < 1:
        raise ValueError(f"dimension d must be >= 1, got {d}")
    gen_dev = generator.device if generator is not None else torch.device("cpu")
    raw = torch.randn(d, dtype=torch.float64, generator=generator, device=gen_dev)
    raw_norm = torch.linalg.vector_norm(raw)
    if float(raw_norm) == 0.0:
        raise ValueError("random_null drew a zero vector")
    out = raw * (float(norm) / raw_norm)
    return out.to(dtype=torch.float32)


def hold_basis(
    hold_a: torch.Tensor,
    hold_b: torch.Tensor,
    groups: Sequence[str],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Orthonormal basis `[k, d]` of group mean-difference directions."""
    a64, b64, w64 = _check_pair_tensors(hold_a, hold_b, weights)
    m, _d = a64.shape
    if len(groups) != m:
        raise ValueError(f"groups length ({len(groups)}) must match hold_a rows ({m})")

    ordered_groups: list[str] = []
    group_indices: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        if not isinstance(g, str) or not g.strip():
            raise ValueError(f"groups[{idx}] must be a non-empty string")
        if g not in group_indices:
            ordered_groups.append(g)
            group_indices[g] = []
        group_indices[g].append(idx)

    basis_rows: list[torch.Tensor] = []
    for g in ordered_groups:
        idxs = group_indices[g]
        ga = a64[idxs]
        gb = b64[idxs]
        gw = w64[idxs]
        vec = (gw[:, None] * (gb - ga)).sum(dim=0) / gw.abs().sum()
        raw_norm = float(torch.linalg.vector_norm(vec).item())
        if raw_norm <= 1e-12:
            raise ValueError(
                f"hold group {g!r} direction is zero or colinear with earlier hold groups"
            )
        # Two-pass modified Gram-Schmidt in float64 for numerical orthogonality.
        for _pass in range(2):
            for u in basis_rows:
                vec = vec - torch.dot(vec, u) * u
        norm = torch.linalg.vector_norm(vec)
        if float(norm) <= max(1e-12, 1e-5 * raw_norm):
            raise ValueError(
                f"hold group {g!r} direction is zero or colinear with earlier hold groups"
            )
        basis_rows.append(vec / norm)

    return torch.stack(basis_rows, dim=0).to(dtype=torch.float32)


def project_out(v: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Project `basis` (`[k, d]`) out of `v` (`[d]`) in float64 with ridge factor `alpha` in `[0, 1]`.

    At `alpha = 1.0` (default), removes 100% of the component in `span(basis)`
    (`v - U^T (U v)`). At `0.0 <= alpha <= 2.0`, applies soft ridge attenuation or
    over-projection (`v - alpha * U^T (U v)`)."""
    if v.ndim != 1:
        raise ValueError(f"v must be a 1D tensor [d], got shape {tuple(v.shape)}")
    d = v.shape[0]
    if basis.ndim != 2 or basis.shape[1] != d:
        raise ValueError(f"basis must have shape [k, {d}], got {tuple(basis.shape)}")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not (0.0 <= alpha_f <= 2.0):
        raise ValueError(f"alpha must be finite and in [0.0, 2.0], got {alpha!r}")

    v64 = v.detach().to(dtype=torch.float64)
    b64 = basis.detach().to(device=v64.device, dtype=torch.float64)
    if not bool(torch.isfinite(v64).all()) or not bool(torch.isfinite(b64).all()):
        raise ValueError("v and basis must contain only finite values")

    if basis.shape[0] == 0 or alpha_f == 0.0:
        return v64.to(dtype=torch.float32)

    ortho_rows: list[torch.Tensor] = []
    for row in b64:
        u = row.clone()
        for _pass in range(2):
            for prev in ortho_rows:
                u = u - torch.dot(u, prev) * prev
        nrm = torch.linalg.vector_norm(u)
        if float(nrm) > 1e-12:
            ortho_rows.append(u / nrm)

    for u in ortho_rows:
        v64 = v64 - alpha_f * torch.dot(v64, u) * u
    return v64.to(dtype=torch.float32)


def _unit_rows64(diffs: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if diffs.ndim != 2 or diffs.shape[0] < 2:
        raise ValueError(f"diffs must be 2D [n, d] with n >= 2, got shape {tuple(diffs.shape)}")
    d64 = diffs.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(d64).all()):
        raise ValueError("diffs contains non-finite values")
    if weights is not None:
        w_t = weights if isinstance(weights, torch.Tensor) else torch.as_tensor(weights, dtype=torch.float64, device=d64.device)
        if w_t.ndim != 1 or w_t.shape[0] != d64.shape[0]:
            raise ValueError(f"weights must have shape [{d64.shape[0]}], got {tuple(w_t.shape)}")
        w64 = w_t.detach().to(device=d64.device, dtype=torch.float64)
        if not bool(torch.isfinite(w64).all()) or bool((w64 == 0).any()):
            raise ValueError("weights must be finite and non-zero")
        d64 = d64 * torch.sign(w64)[:, None]
    norms = torch.linalg.vector_norm(d64, dim=1, keepdim=True)
    if bool((norms == 0).any()):
        raise ValueError("diffs contains a zero row; pairwise cosine is undefined")
    return d64 / norms


def mean_pairwise_cosine(
    diffs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    """Mean cosine similarity over off-diagonal pairs `0 <= i < j < n` in float64.

    When `weights` is provided, each row `diffs[i]` is oriented by `sign(weights[i])`."""
    unit = _unit_rows64(diffs, weights=weights)
    n = unit.shape[0]
    gram = unit @ unit.T
    tri = torch.triu_indices(n, n, offset=1, device=gram.device)
    return float(gram[tri[0], tri[1]].mean().item())


def per_pair_cosine(
    diffs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> list[float]:
    """For each pair `i`, its mean cosine similarity against all other pairs `j != i`.

    When `weights` is provided, each row `diffs[i]` is oriented by `sign(weights[i])`."""
    unit = _unit_rows64(diffs, weights=weights)
    n = unit.shape[0]
    gram = unit @ unit.T
    out: list[float] = []
    for i in range(n):
        row_sum = float(gram[i].sum().item()) - float(gram[i, i].item())
        out.append(row_sum / float(n - 1))
    return out


def split_half_cosine(
    diffs: torch.Tensor,
    generator: torch.Generator | None = None,
    n_splits: int = 64,
    weights: torch.Tensor | None = None,
) -> float:
    """Mean cosine between disjoint random halves of `diffs` (`[n, d]`) across `n_splits`."""
    if diffs.ndim != 2 or diffs.shape[0] < 2:
        raise ValueError(f"split_half_cosine requires diffs [n, d] with n >= 2, got {tuple(diffs.shape)}")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    d64 = diffs.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(d64).all()):
        raise ValueError("diffs contains non-finite values")
    n = d64.shape[0]
    if weights is None:
        w64 = torch.ones(n, dtype=torch.float64, device=d64.device)
    else:
        w_t = weights if isinstance(weights, torch.Tensor) else torch.as_tensor(weights, dtype=torch.float64, device=d64.device)
        if w_t.ndim != 1 or w_t.shape[0] != n:
            raise ValueError(f"weights must have shape [{n}], got {tuple(w_t.shape)}")
        w64 = w_t.detach().to(device=d64.device, dtype=torch.float64)
        if not bool(torch.isfinite(w64).all()) or bool((w64 == 0).any()):
            raise ValueError("weights must be finite and non-zero")

    weighted_diffs = w64[:, None] * d64
    half = n // 2
    gen_dev = generator.device if generator is not None else torch.device("cpu")
    cosines: list[float] = []
    for _ in range(n_splits):
        perm = torch.randperm(n, generator=generator, device=gen_dev).to(d64.device)
        idx1, idx2 = perm[:half], perm[half:]
        h1 = weighted_diffs[idx1].sum(dim=0) / w64[idx1].abs().sum()
        h2 = weighted_diffs[idx2].sum(dim=0) / w64[idx2].abs().sum()
        n1 = torch.linalg.vector_norm(h1)
        n2 = torch.linalg.vector_norm(h2)
        if float(n1) == 0.0 or float(n2) == 0.0:
            raise ValueError("split_half_cosine encountered a half with zero mean norm")
        cosines.append(float(torch.dot(h1, h2).item() / (float(n1.item()) * float(n2.item()))))
    return float(sum(cosines) / len(cosines))


def quality_report(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: Sequence[float] | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    n_splits: int = 64,
) -> dict[str, Any]:
    """Compute the `quality_report` metadata dict (`CONTRACT.md` section 3) from
    per-pair pooled activations `a` and `b` of shape `[n, d]`."""
    a64, b64, w64 = _check_pair_tensors(a, b, weights)
    diffs = b64 - a64
    if diffs.shape[0] == 1:
        norm0 = float(torch.linalg.vector_norm(diffs[0]).item())
        if norm0 == 0.0:
            raise ValueError("diffs contains a zero row; pairwise cosine is undefined")
        return {
            "mean_pairwise_cosine": None,
            "split_half_cosine": None,
            "per_pair_cosine": [],
        }
    return {
        "mean_pairwise_cosine": mean_pairwise_cosine(diffs, weights=w64),
        "split_half_cosine": split_half_cosine(
            diffs, generator=generator, n_splits=n_splits, weights=w64
        ),
        "per_pair_cosine": per_pair_cosine(diffs, weights=w64),
    }


def _tensor_bf16_bytes(t: torch.Tensor) -> memoryview:
    bf16 = t.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return bf16.view(torch.uint8).numpy().data


def _tensor_bf16_hash12(t: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bf16_bytes(t)).hexdigest()[:12]


def family_from_tensor_hashes(hashes: tuple[str, str, str]) -> str:
    """Resolve `(condition_proj.weight, condition_proj.bias, token_refiner.final_norm.weight)`
    12-hex-char BF16 hashes to `'fl2va'`, `'ref2va'`, `'fasth3'`, or `'custom:<hash12>'`."""
    if hashes in KNOWN_FAMILIES:
        return KNOWN_FAMILIES[hashes]
    combined = hashlib.sha256(":".join(hashes).encode("utf-8")).hexdigest()[:12]
    return f"custom:{combined}"


_FAMILY_PATH_CACHE: dict[tuple[str, int, int], str] = {}
_DIGEST_PATH_CACHE: dict[tuple[str, int, int, tuple[str, ...]], str] = {}


def _file_stat_key(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    st = resolved.stat()
    return (str(resolved), int(st.st_mtime_ns), int(st.st_size))


def detect_family(source: str | Path | Mapping[str, torch.Tensor] | Any) -> str:
    """Determine the H3 text-path family (`'fl2va'`, `'ref2va'`, `'fasth3'`, or
    `'custom:<hash12>'`) from a checkpoint `.safetensors` path, state dict, or
    ComfyUI `ModelPatcher`.

    Refuses in plain words if any of the three defining text-path tensors is
    missing or cannot be read."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"checkpoint file not found: {path}")
        stat_key = _file_stat_key(path)
        cached = _FAMILY_PATH_CACHE.get(stat_key)
        if cached is not None:
            return cached
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            hashes: list[str] = []
            for k in FAMILY_TENSOR_KEYS:
                matched = next(
                    (
                        cand
                        for cand in (k, f"diffusion_model.{k}", f"model.diffusion_model.{k}")
                        if cand in keys
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"checkpoint {path.name} is missing H3 text-path tensor {k!r}; "
                        "not a supported MiniMax H3 diffusion model"
                    )
                hashes.append(_tensor_bf16_hash12(f.get_tensor(matched)))
        fam = family_from_tensor_hashes((hashes[0], hashes[1], hashes[2]))
        _FAMILY_PATH_CACHE[stat_key] = fam
        return fam

    if isinstance(source, Mapping):
        hashes = []
        for k in FAMILY_TENSOR_KEYS:
            matched_t = None
            for cand in (k, f"diffusion_model.{k}", f"model.diffusion_model.{k}"):
                if cand in source and isinstance(source[cand], torch.Tensor):
                    matched_t = source[cand]
                    break
            if matched_t is None:
                raise ValueError(
                    f"state_dict is missing H3 text-path tensor {k!r}; "
                    "not a supported MiniMax H3 diffusion model"
                )
            hashes.append(_tensor_bf16_hash12(matched_t))
        return family_from_tensor_hashes((hashes[0], hashes[1], hashes[2]))

    init = getattr(source, "cached_patcher_init", None)
    if isinstance(init, tuple) and len(init) >= 2 and isinstance(init[1], tuple) and len(init[1]) >= 1:
        cand_path = init[1][0]
        if isinstance(cand_path, (str, Path)) and Path(cand_path).is_file():
            return detect_family(Path(cand_path))

    backup = getattr(source, "backup", None) or {}
    inner_model = getattr(source, "model", source)
    dm = getattr(inner_model, "diffusion_model", inner_model)
    hashes = []
    for k in FAMILY_TENSOR_KEYS:
        bk_key = f"diffusion_model.{k}"
        t: torch.Tensor | None = None
        if isinstance(backup, Mapping) and bk_key in backup:
            entry = backup[bk_key]
            t = getattr(entry, "weight", entry) if not isinstance(entry, torch.Tensor) else entry
        if t is None:
            obj = dm
            for part in k.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, torch.Tensor):
                t = obj
        if t is None or not isinstance(t, torch.Tensor):
            raise ValueError(
                f"model ({type(source).__name__}) has no checkpoint path and is missing H3 "
                f"text-path tensor {k!r}; cannot determine H3 text-path family"
            )
        hashes.append(_tensor_bf16_hash12(t))
    return family_from_tensor_hashes((hashes[0], hashes[1], hashes[2]))


def safetensors_recipe_digest(
    path: str | Path,
    *,
    extra_tensor_keys: Sequence[str] = (),
) -> str:
    """Fast deterministic SHA-256 over a `.safetensors` file header plus any
    named tensors (`FAMILY_TENSOR_KEYS` for diffusion checkpoints)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"model file not found: {p}")
    stat_key = (*_file_stat_key(p), tuple(extra_tensor_keys))
    cached = _DIGEST_PATH_CACHE.get(stat_key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with p.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{p}: invalid safetensors file (shorter than 8 bytes)")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = fp.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{p}: truncated safetensors header")
        h.update(raw_len)
        h.update(header_bytes)

    if extra_tensor_keys:
        with safe_open(str(p), framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            for k in extra_tensor_keys:
                matched = next(
                    (
                        cand
                        for cand in (k, f"diffusion_model.{k}", f"model.diffusion_model.{k}")
                        if cand in keys
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"{p.name}: missing required tensor key {k!r} for recipe digest"
                    )
                h.update(k.encode("utf-8"))
                h.update(_tensor_bf16_bytes(sf.get_tensor(matched)))
    digest = h.hexdigest()
    _DIGEST_PATH_CACHE[stat_key] = digest
    return digest


def _validate_metadata_dict(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("control vector metadata must be a mapping")
    for req in REQUIRED_METADATA_KEYS:
        if req not in metadata:
            raise ValueError(f"control vector metadata is missing required key {req!r}")

    fmt = metadata["format"]
    if fmt != FORMAT_VERSION:
        raise ValueError(
            f"unsupported control vector format {fmt!r}; expected {FORMAT_VERSION!r}"
        )

    name = metadata["name"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"control vector metadata 'name' must match [a-z0-9_]+, got {name!r}")

    quality = metadata["quality"]
    if not isinstance(quality, str) or not quality.strip():
        raise ValueError("control vector metadata 'quality' must be a non-empty string")

    family = metadata["family"]
    if not isinstance(family, str) or (
        family not in ("fl2va", "ref2va", "fasth3") and not CUSTOM_FAMILY_RE.match(family)
    ):
        raise ValueError(
            f"control vector metadata 'family' must be 'fl2va', 'ref2va', 'fasth3', "
            f"or 'custom:<12 hex chars>', got {family!r}"
        )

    pairs = metadata["pairs"]
    if not isinstance(pairs, dict) or not {"path", "sha256", "n_pairs", "n_holds"}.issubset(pairs.keys()):
        raise ValueError(
            "control vector metadata 'pairs' must be a dict with keys {'path', 'sha256', 'n_pairs', 'n_holds'}"
        )

    recipe = metadata["recipe"]
    if (
        not isinstance(recipe, dict)
        or not {"checkpoint", "loras", "encoder"}.issubset(recipe.keys())
        or not isinstance(recipe["checkpoint"], dict)
        or not isinstance(recipe["loras"], list)
        or not isinstance(recipe["encoder"], dict)
    ):
        raise ValueError(
            "control vector metadata 'recipe' must be a dict with 'checkpoint' (dict), 'loras' (list), and 'encoder' (dict)"
        )

    measured = metadata["measured_under"]
    if (
        not isinstance(measured, dict)
        or not {"mode", "frame", "prefix", "pooling"}.issubset(measured.keys())
    ):
        raise ValueError(
            "control vector metadata 'measured_under' must be a dict with keys {'mode', 'frame', 'prefix', 'pooling'}"
        )
    if measured["mode"] not in VALID_MODES:
        raise ValueError(
            f"measured_under.mode must be one of {VALID_MODES}, got {measured['mode']!r}"
        )
    if measured["frame"] not in VALID_FRAMES:
        raise ValueError(
            f"measured_under.frame must be one of {VALID_FRAMES}, got {measured['frame']!r}"
        )
    if measured["pooling"] not in VALID_ALL_POOLING:
        raise ValueError(
            f"measured_under.pooling must be one of {VALID_ALL_POOLING}, got {measured['pooling']!r}"
        )

    points = metadata["points"]
    if not isinstance(points, dict) or not points:
        raise ValueError("control vector metadata 'points' must be a non-empty dict")
    for pt_key, pt_val in points.items():
        if not isinstance(pt_val, dict) or "unit" not in pt_val or "median_row_norm" not in pt_val:
            raise ValueError(
                f"points[{pt_key!r}] must be a dict with at least 'unit' and 'median_row_norm'"
            )
        if pt_key == "post_refiner":
            if "raw_norm" not in pt_val or pt_val["unit"] != "mean_pair_difference":
                raise ValueError(
                    "points['post_refiner'] must be a dict with unit='mean_pair_difference', 'raw_norm', and 'median_row_norm'"
                )
            num_keys: tuple[str, ...] = ("raw_norm", "median_row_norm")
        else:
            if pt_val["unit"] not in ("median_row_norm", "median_tag1_row_norm"):
                raise ValueError(
                    f"points[{pt_key!r}]['unit'] must be 'median_row_norm', got {pt_val['unit']!r}"
                )
            if "row0_excluded" in pt_val and not isinstance(pt_val["row0_excluded"], bool):
                raise ValueError(f"points[{pt_key!r}]['row0_excluded'] must be a bool")
            num_keys = ("raw_norm", "median_row_norm") if "raw_norm" in pt_val else ("median_row_norm",)
        for nk in num_keys:
            nv = pt_val[nk]
            if (
                isinstance(nv, bool)
                or not isinstance(nv, (int, float))
                or not math.isfinite(float(nv))
                or float(nv) <= 0.0
            ):
                raise ValueError(
                    f"points[{pt_key!r}][{nk!r}] must be a positive finite float, got {nv!r}"
                )

    qrep = metadata["quality_report"]
    if (
        not isinstance(qrep, dict)
        or not {"mean_pairwise_cosine", "split_half_cosine", "per_pair_cosine"}.issubset(qrep.keys())
    ):
        raise ValueError(
            "control vector metadata 'quality_report' must be a dict with keys "
            "{'mean_pairwise_cosine', 'split_half_cosine', 'per_pair_cosine'}"
        )
    n_pairs_q = int((metadata.get("pairs") or {}).get("n_pairs", 2))
    for k_float in ("mean_pairwise_cosine", "split_half_cosine"):
        v_f = qrep[k_float]
        if v_f is None and n_pairs_q < 2:
            continue
        if isinstance(v_f, bool) or not isinstance(v_f, (int, float)) or not math.isfinite(float(v_f)):
            raise ValueError(
                f"quality_report[{k_float!r}] must be a finite float when n_pairs >= 2, got {v_f!r}"
            )
    ppc = qrep["per_pair_cosine"]
    if (
        not isinstance(ppc, list)
        or (not ppc and n_pairs_q >= 2)
        or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in ppc)
    ):
        raise ValueError(
            "quality_report['per_pair_cosine'] must be a list of finite floats (non-empty when n_pairs >= 2)"
        )

    prov = metadata["provenance"]
    if (
        not isinstance(prov, dict)
        or not {"core_revision", "pack_commit", "created_utc", "made_by"}.issubset(prov.keys())
    ):
        raise ValueError(
            "control vector metadata 'provenance' must be a dict with keys "
            "{'core_revision', 'pack_commit', 'created_utc', 'made_by'}"
        )


def _validate_vector_payload(
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    _validate_metadata_dict(metadata)
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("control vector tensors dict is empty")

    has_known_vector = False
    for key, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"tensor {key!r} is not a torch.Tensor")
        if t.dtype != torch.float32:
            raise ValueError(f"tensor {key!r} must have dtype float32, got {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"tensor {key!r} contains non-finite values")

        if key == "vector.post_refiner":
            has_known_vector = True
            if tuple(t.shape) != (POST_REFINER_DIM,):
                raise ValueError(
                    f"tensor 'vector.post_refiner' must have shape ({POST_REFINER_DIM},), got {tuple(t.shape)}"
                )
            if float(torch.linalg.vector_norm(t).item()) == 0.0:
                raise ValueError("tensor 'vector.post_refiner' has zero norm")
        elif key == "holds.post_refiner":
            if t.ndim != 2 or t.shape[1] != POST_REFINER_DIM or t.shape[0] < 1:
                raise ValueError(
                    f"tensor 'holds.post_refiner' must have shape [k, {POST_REFINER_DIM}], got {tuple(t.shape)}"
                )
            k_rows = t.shape[0]
            b64 = t.detach().to(dtype=torch.float64)
            gram = b64 @ b64.T
            eye = torch.eye(k_rows, dtype=torch.float64, device=b64.device)
            if not torch.allclose(gram, eye, atol=1e-4):
                raise ValueError("tensor 'holds.post_refiner' rows are not orthonormal")
        elif key == "vector.encoder" or LLM_VECTOR_KEY_RE.match(key):
            has_known_vector = True
            if tuple(t.shape) != (ENCODER_DIM,):
                raise ValueError(
                    f"tensor {key!r} must have shape ({ENCODER_DIM},), got {tuple(t.shape)}"
                )
            if float(torch.linalg.vector_norm(t).item()) == 0.0:
                raise ValueError(f"tensor {key!r} has zero norm")
        else:
            raise ValueError(f"unknown control vector tensor key {key!r}")

    if not has_known_vector:
        raise ValueError("control vector must contain at least one 'vector.*' tensor")

    if "vector.post_refiner" in tensors and "holds.post_refiner" in tensors:
        v64 = tensors["vector.post_refiner"].detach().to(dtype=torch.float64)
        b64 = tensors["holds.post_refiner"].detach().to(dtype=torch.float64)
        v_norm = float(torch.linalg.vector_norm(v64).item())
        dots = b64 @ v64
        if float(dots.abs().max().item()) > 1e-4 * max(1.0, v_norm):
            raise ValueError(
                "tensor 'vector.post_refiner' is not orthogonal to 'holds.post_refiner'"
            )


def default_vector_filename(
    stem: str,
    family: str,
    *,
    pooling: str = "scene_text",
    prefix: Any = "none",
    apply_holds: bool = True,
    n_holds: int = 0,
    loras: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Build an overwrite-safe `.safetensors` filename for a control vector.

    Keeps `<stem>__<family>.safetensors` for the default `pooling='scene_text'`,
    `prefix='none'`, `apply_holds=True`, `loras=None` recipe, and appends
    disambiguating suffixes whenever pooling, reference prefix, hold projection,
    or LoRA recipe differs."""
    safe_family = str(family).replace(":", "_")
    parts = [str(stem), safe_family]
    if pooling != "scene_text":
        parts.append(str(pooling))
    if prefix != "none" and prefix is not None:
        if isinstance(prefix, Mapping) and "references_sha256" in prefix:
            parts.append(f"ref_{str(prefix['references_sha256'])[:8]}")
        else:
            digest8 = hashlib.sha256(str(prefix).encode("utf-8")).hexdigest()[:8]
            parts.append(f"prefix_{digest8}")
    if n_holds > 0 and not apply_holds:
        parts.append("noholds")
    if loras:
        lora_digest8 = hashlib.sha256(
            json.dumps([dict(item) for item in loras], sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        parts.append(f"lora_{lora_digest8}")
    return "__".join(parts) + ".safetensors"


_POOLING_DESCRIPTIONS: dict[str, str] = {
    "scene_text": "all tag-1 scene-text rows (excluding reference labels, timestamps, speaker tags, and <d>...</d> dialogue)",
    "from_first_diff": "all tag-1 non-vision rows from the first differing token onward (idx >= first_diff)",
    "diff_span": "only the differing token span [first_diff:end_diff] (stripping both shared prefix and shared suffix tokens)",
    "all_tag1": "all tag-1 non-vision rows across the sequence",
    "vision_tokens": "all tag-0 reference vision rows across the sequence (media-contrast mode)",
    "all_rows": "all tag-0 vision and tag-1 text rows across the sequence (excluding row 0 for encoder)",
}


def format_vector_sidecar_md(
    safetensors_filename: str,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    pair_spans: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Format a human-readable companion `.md` sidecar documenting how a control
    vector `.safetensors` file was constructed, its recipe, norms, recommended
    ComfyUI strength range, quality report, and per-pair token spans."""
    name = str(metadata.get("name", Path(safetensors_filename).stem))
    quality = str(metadata.get("quality", ""))
    family = str(metadata.get("family", ""))
    measured = dict(metadata.get("measured_under") or {})
    pooling = str(measured.get("pooling", "scene_text"))
    pooling_desc = _POOLING_DESCRIPTIONS.get(pooling, pooling)
    mode = str(measured.get("mode", ""))
    frame = str(measured.get("frame", ""))
    prefix = measured.get("prefix", "none")
    prefix_str = json.dumps(prefix, sort_keys=True) if isinstance(prefix, Mapping) else str(prefix)

    prov = dict(metadata.get("provenance") or {})
    pairs_meta = dict(metadata.get("pairs") or {})
    a_side = str(pairs_meta.get("a_side") or metadata.get("a_side") or "")
    b_side = str(pairs_meta.get("b_side") or metadata.get("b_side") or "")
    resolved_spans = (
        list(pair_spans)
        if pair_spans is not None
        else list(pairs_meta.get("pair_spans") or metadata.get("pair_spans") or ())
    )

    recipe = dict(metadata.get("recipe") or {})
    ckpt = dict(recipe.get("checkpoint") or {})
    enc = dict(recipe.get("encoder") or {})
    loras = list(recipe.get("loras") or ())

    points = dict(metadata.get("points") or {})
    qrep = dict(metadata.get("quality_report") or {})

    lines: list[str] = [
        f"# Control Vector: `{name}` (`{safetensors_filename}`)",
        "",
        "## How It Was Built (`measured_under` & `provenance`)",
        f"- **Safetensors file**: `{safetensors_filename}`",
        f"- **Pooling**: `{pooling}` — {pooling_desc}",
        f"- **Family**: `{family}`",
        f"- **Mode**: `{mode}` | **Frame**: `{frame}` | **Prefix**: `{prefix_str}`",
        f"- **Created (UTC)**: `{prov.get('created_utc', 'unknown')}`",
        f"- **Made by**: `{prov.get('made_by', 'unknown')}`",
        f"- **Pack commit**: `{prov.get('pack_commit', 'unknown')}` | **Core revision**: `{prov.get('core_revision', 'unknown')}`",
        "",
        "## Target Quality & Sides",
        f"- **Quality**: {quality}",
    ]
    if a_side:
        lines.append(f"- **Pole A (`a_side`, baseline)**: {a_side}")
    if b_side:
        lines.append(f"- **Pole B (`b_side`, steered)**: {b_side}")

    ckpt_sha12 = str(ckpt.get("sha256", ""))[:12]
    enc_sha12 = str(enc.get("sha256", ""))[:12]
    lines.extend(
        [
            "",
            "## Model Recipe (`recipe`)",
            f"- **Checkpoint**: `{ckpt.get('file', 'unspecified')}` (`{ckpt_sha12}`)",
            f"- **Encoder**: `{enc.get('file', 'unspecified')}` (`{enc_sha12}`)",
        ]
    )
    if loras:
        for l_item in loras:
            l_sha12 = str(l_item.get("sha256", ""))[:12]
            l_str = l_item.get("strength", 1.0)
            l_app = l_item.get("applied", "branch")
            lines.append(
                f"- **LoRA**: `{l_item.get('file', 'unknown')}` (strength=`{l_str:g}`, sha256=`{l_sha12}`, applied=`{l_app}`)"
            )
    else:
        lines.append("- **LoRAs**: none")

    pairs_sha = str(pairs_meta.get("sha256", ""))
    lines.extend(
        [
            "",
            "## Source Pair Set (`pairs`)",
            f"- **Pair TOML path**: `{pairs_meta.get('path', 'unknown')}`",
            f"- **Pair TOML SHA-256**: `{pairs_sha[:12]}` (`{pairs_sha}`)",
            f"- **Pairs**: `{pairs_meta.get('n_pairs', 0)}` | **Holds**: `{pairs_meta.get('n_holds', 0)}`",
            "",
            "## Norms, Recommended Strength & Quality Report (`points` & `quality_report`)",
        ]
    )

    if "post_refiner" in points:
        pr = dict(points["post_refiner"])
        pr_norm = float(pr.get("raw_norm", 0.0))
        pr_med = float(pr.get("median_row_norm", 0.0))
        pr_shape = (
            list(tensors["vector.post_refiner"].shape)
            if "vector.post_refiner" in tensors
            else [POST_REFINER_DIM]
        )
        (matched_lo, matched_hi), add_range = strength_ranges(pr_norm)
        lines.append(
            f"- **`vector.post_refiner`**: shape=`{pr_shape}`, unit=`{pr.get('unit', 'mean_pair_difference')}`, "
            f"`raw_norm={pr_norm:.6f}`, `median_row_norm={pr_med:.4f}`"
        )
        if add_range is not None:
            window = f"`{NORM_SHIFT_WINDOW[0]:g}..{NORM_SHIFT_WINDOW[1]:g}`"
            lines.append(
                f"- **Recommended ComfyUI `strength` range** (targeting post-refiner shift norm {window}): "
                f"**`~{matched_lo:.2f} .. {matched_hi:.2f}`** in mode `add_norm_matched`, the node's default, "
                f"where every vector is rescaled to one length and this range is the same for all of them; "
                f"`~{add_range[0]:.2f} .. {add_range[1]:.2f}` in mode `add`, where this vector is used as stored"
            )

    if "encoder" in points:
        en = dict(points["encoder"])
        en_norm = float(en.get("raw_norm", 0.0))
        en_med = float(en.get("median_row_norm", 0.0))
        en_shape = (
            list(tensors["vector.encoder"].shape)
            if "vector.encoder" in tensors
            else [ENCODER_DIM]
        )
        lines.append(
            f"- **`vector.encoder`**: shape=`{en_shape}`, unit=`{en.get('unit', 'median_row_norm')}`, "
            f"`raw_norm={en_norm:.6f}`, `median_row_norm={en_med:.4f}`, "
            f"`row0_excluded={en.get('row0_excluded', True)}`"
        )

    for t_key, t_val in tensors.items():
        if t_key not in ("vector.post_refiner", "vector.encoder"):
            lines.append(f"- **`{t_key}`**: shape=`{list(t_val.shape)}`, dtype=`{t_val.dtype}`")

    mpc_raw = qrep.get("mean_pairwise_cosine")
    shc_raw = qrep.get("split_half_cosine")
    mpc_str = f"{float(mpc_raw):.4f}" if mpc_raw is not None else "n/a (1 pair)"
    shc_str = f"{float(shc_raw):.4f}" if shc_raw is not None else "n/a (1 pair)"
    ppc = [float(x) for x in (qrep.get("per_pair_cosine") or ())]
    ppc_min = min(ppc) if ppc else 0.0
    ppc_max = max(ppc) if ppc else 0.0
    ppc_formatted = ", ".join(f"{x:.4f}" for x in ppc)
    lines.extend(
        [
            f"- **`mean_pairwise_cosine`**: `{mpc_str}`",
            f"- **`split_half_cosine`**: `{shc_str}`",
            f"- **`per_pair_cosine`**: min=`{ppc_min:.4f}`, max=`{ppc_max:.4f}` (`[{ppc_formatted}]`)",
        ]
    )
    for extra_qk in sorted(set(qrep.keys()) - {"mean_pairwise_cosine", "split_half_cosine", "per_pair_cosine"}):
        lines.append(f"- **`{extra_qk}`**: `{json.dumps(qrep[extra_qk], sort_keys=True)}`")

    lines.extend(
        [
            "",
            "## How to Read These Scores",
            "",
            "| Metric | What It Measures | Healthy Target Range | Warning Signs |",
            "|---|---|---|---|",
            "| `raw_norm` (`vector.post_refiner`) | Unnormalized L2 magnitude of the mean $B - A$ vector in 5376-d post-refiner space | `65..95` (`diff_span`), `20..28` (`from_first_diff`), `18..28` (multi-clause `scene_text`) | `< 14` on `scene_text` indicates prefix/suffix dilution (`~80–90%` shared tokens); switch to `--pooling diff_span` or scale `strength` so $\\text{strength} \\times \\lVert\\mathbf{v}_{\\text{PR}}\\rVert_2 \\in [65, 85]$ (`strength * raw_norm`) |",
            "| `Recommended ComfyUI strength` | Multiplier $s$ on `MiniMaxH3ApplyControlVectors` targeting $\\lVert\\Delta \\mathbf{r}_{\\text{PR}}\\rVert_2 = s \\cdot \\lVert\\mathbf{v}_{\\text{PR}}\\rVert_2 \\in [50, 85]$ (`s * raw_norm`) | `0.6..1.1` (`diff_span`), `2.5..3.5` (`from_first_diff` / multi-clause `scene_text`) | $\\lVert\\Delta \\mathbf{r}_{\\text{PR}}\\rVert_2 < 35$ is subtle; $\\lVert\\Delta \\mathbf{r}_{\\text{PR}}\\rVert_2 > 110$ can push activations off-manifold |",
            "| `mean_pairwise_cosine` | Mean cosine similarity across all $\\binom{N}{2}$ pairs of individual $(B_i - A_i)$ directions | **`0.62 .. 0.85`** on diverse, non-cloned pairs | `< 0.55` indicates noisy/confounded pairs or target-noun cancellation in $B - A$; `> 0.90` indicates verbatim lexical cloning across pairs (overfitting to one phrase) |",
            "| `split_half_cosine` | Mean cosine between two random disjoint 5-pair halves across 64 splits (stability of the 10-pair average) | **`0.88 .. 0.96`** | `< 0.82` means 10 pairs are not yet converging to a stable direction; `> 0.98` usually reflects verbatim span cloning |",
            "| `per_pair_cosine` (`min` / `max`) | Each pair's mean cosine agreement with the other $N-1$ pairs | `min >= 0.55` | Any pair with `per_pair_cosine < 0.50` is an outlier (often a cross-section diff or mismatched metaphor) that should be rewritten |",
        ]
    )

    if resolved_spans:
        lines.extend(
            [
                "",
                "## Per-Pair Token Spans",
                "",
                "| Pair | Weight | Bounds (`A` / `B`) | Pooled Toks (`A` / `B`) | Seq Len (`A` / `B`) | Differing Span (`A` -> `B`) |",
                "|---:|---:|---|---|---|---|",
            ]
        )
        for sp in resolved_spans:
            idx = int(sp.get("index", 0))
            w = float(sp.get("weight", 1.0))
            la = int(sp.get("len_a", 0))
            lb = int(sp.get("len_b", 0))
            pa = int(sp.get("pooled_rows_a", 0))
            pb = int(sp.get("pooled_rows_b", 0))
            if "first_diff" in sp and ("end_a" in sp or "end_b" in sp):
                fd = int(sp["first_diff"])
                ea = int(sp.get("end_a", la))
                eb = int(sp.get("end_b", lb))
                bounds_str = f"`[{fd}:{ea}]` / `[{fd}:{eb}]`"
            elif "first_diff" in sp:
                fd = int(sp["first_diff"])
                bounds_str = f"`[{fd}:{la}]` / `[{fd}:{lb}]`"
            else:
                bounds_str = "`all` / `all`"
            a_sp = str(sp.get("a_span", "")).replace("|", "\\|").replace("\n", " ")
            b_sp = str(sp.get("b_span", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{idx:02d}` | `{w:.2f}` | {bounds_str} | `{pa}` / `{pb}` | `{la}` / `{lb}` | `-{a_sp}` -> `+{b_sp}` |"
            )

    return "\n".join(lines) + "\n"


def save_vector(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = True,
    write_sidecar_md: bool = True,
) -> Path:
    """Validate and write a `.safetensors` control vector file and its companion `.md` sidecar."""
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("control vector tensors dict is empty")
    clean_tensors: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"tensor {k!r} is not a torch.Tensor")
        if v.dtype != torch.float32:
            raise ValueError(f"tensor {k!r} must have dtype float32, got {v.dtype}")
        clean_tensors[k] = v.detach().to(device="cpu", dtype=torch.float32).contiguous()

    _validate_vector_payload(clean_tensors, metadata)

    encoded_meta: dict[str, str] = {}
    for k, v in metadata.items():
        encoded_meta[str(k)] = json.dumps(v, sort_keys=True)

    out_path = Path(path)
    if not overwrite and out_path.exists():
        raise ValueError(
            f"control vector file {out_path.name!r} already exists at {out_path} "
            "(pass overwrite=True / --overwrite to replace it, or set a distinct output_name)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(save_safetensors_bytes(clean_tensors, metadata=encoded_meta))
    if write_sidecar_md:
        sidecar_path = out_path.with_suffix(".md")
        sidecar_md = format_vector_sidecar_md(out_path.name, clean_tensors, metadata)
        sidecar_path.write_text(sidecar_md, encoding="utf-8")
    return out_path


def load_vector(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read and validate a `.safetensors` control vector file."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"control vector file not found: {p}")

    tensors: dict[str, torch.Tensor] = {}
    raw_meta: dict[str, str] = {}
    with safe_open(str(p), framework="pt", device="cpu") as sf:
        raw_meta = dict(sf.metadata() or {})
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k).clone()

    metadata: dict[str, Any] = {}
    for k, v in raw_meta.items():
        try:
            metadata[k] = json.loads(v)
        except Exception:
            metadata[k] = v

    _validate_vector_payload(tensors, metadata)
    return tensors, metadata


def check_vector_compatibility(
    metadata: Mapping[str, Any],
    *,
    loaded_family: str,
    loaded_recipe: Mapping[str, Any] | None = None,
    loaded_prefix: Any | None = None,
    allow_recipe_mismatch: bool = False,
) -> list[str]:
    """Verify `metadata` against the loaded model's `family`, `recipe`, and `prefix`.

    - Unknown `format`, malformed metadata, or mismatched `family` always raises
      `ValueError`.
    - Mismatched `recipe` (checkpoint or LoRA list) raises `ValueError` unless
      `allow_recipe_mismatch=True`, in which case it is returned in the advisory
      warnings list.
    - Mismatched `measured_under.prefix` is always reported in the warnings list
      and never refused."""
    _validate_metadata_dict(metadata)

    vec_name = metadata["name"]
    vec_family = metadata["family"]
    if vec_family != loaded_family:
        raise ValueError(
            f"control vector {vec_name!r} was made for H3 text-path family {vec_family!r}, "
            f"cannot apply to loaded model of family {loaded_family!r}"
        )

    warnings: list[str] = []
    if loaded_recipe is None:
        if not allow_recipe_mismatch:
            raise ValueError(
                "loaded_recipe is required unless allow_recipe_mismatch=True"
            )
    else:
        if not isinstance(loaded_recipe, Mapping):
            raise ValueError("loaded_recipe must be a mapping")
        vec_recipe = metadata["recipe"]
        vec_ckpt = vec_recipe.get("checkpoint")
        cur_ckpt = loaded_recipe.get("checkpoint")
        prov_made_by = str(metadata.get("provenance", {}).get("made_by", ""))
        jlens_same_family = prov_made_by.startswith("j_lens.") and vec_family == loaded_family
        h3_family_compatible = _is_same_family_h3_recipe_compatible(
            vec_recipe, loaded_recipe, vec_family=vec_family, loaded_family=loaded_family
        )
        if vec_ckpt != cur_ckpt:
            msg = (
                f"control vector {vec_name!r} recipe checkpoint {vec_ckpt!r} differs "
                f"from loaded model checkpoint {cur_ckpt!r}"
            )
            if not allow_recipe_mismatch and not jlens_same_family and not h3_family_compatible:
                raise ValueError(
                    f"{msg} (pass allow_recipe_mismatch=True to apply across checkpoints)"
                )
            warnings.append(msg)

        vec_loras = vec_recipe.get("loras", [])
        cur_loras = loaded_recipe.get("loras", [])
        if vec_loras != cur_loras:
            msg = (
                f"control vector {vec_name!r} recipe LoRA set {vec_loras!r} differs "
                f"from loaded model LoRA set {cur_loras!r}"
            )
            vec_sig = _lora_family_signature(vec_loras) if vec_loras else None
            cur_sig = _lora_family_signature(cur_loras) if cur_loras else None
            same_pika_family = (
                vec_sig is not None
                and cur_sig is not None
                and vec_sig == cur_sig
            )
            if (
                not allow_recipe_mismatch
                and not same_pika_family
                and not jlens_same_family
                and not h3_family_compatible
            ):
                raise ValueError(
                    f"{msg} (pass allow_recipe_mismatch=True to apply across LoRA sets)"
                )
            warnings.append(msg)

    if loaded_prefix is not None:
        vec_prefix = metadata["measured_under"].get("prefix")
        if vec_prefix != loaded_prefix:
            warnings.append(
                f"control vector {vec_name!r} was measured under prefix {vec_prefix!r}, "
                f"render has prefix {loaded_prefix!r}"
            )

    return warnings


_PIKA_FLEX_VARIANT_RE = re.compile(
    r"^(?:.*/)?(pika_h3_flex_(?:fl2va|ref2va))_(?:svd\d+_bf16|comfy_bf16|v1_avg_rank_\d+_bf16|kijai_bf16)\.safetensors$"
)


def _applied_refiner_coverage(applied: str) -> str:
    applied_norm = str(applied or "branch").strip()
    if applied_norm in ("branch", "merged"):
        return "full_refiner"
    if applied_norm.startswith("branch:"):
        parts: dict[str, str] = {}
        last_key: str | None = None
        for chunk in applied_norm[len("branch:") :].split(","):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                last_key = k.strip().lower()
                parts[last_key] = v.strip()
            elif last_key is not None:
                parts[last_key] = f"{parts[last_key]},{chunk.strip()}"
        modules_val = parts.get("modules", "all").strip().lower()
        blocks_val = parts.get("blocks", "all").strip().lower()
        block_tokens = {t.strip() for t in blocks_val.split(",") if t.strip()}
        covers_blocks = blocks_val == "all" or bool(block_tokens & {"refiner", "token_refiner"})
        if modules_val in ("all", "no adaln") and covers_blocks:
            return "full_refiner"
        return f"partial_refiner:modules={modules_val},blocks={blocks_val}"
    return f"other:{applied_norm}"


def _lora_family_signature(loras: Sequence[Any]) -> list[tuple[str, float, str]] | None:
    sig: list[tuple[str, float, str]] = []
    for item in loras:
        if not isinstance(item, Mapping):
            return None
        raw_file = str(item.get("file", item.get("name", "")))
        m = _PIKA_FLEX_VARIANT_RE.match(raw_file)
        if m is None:
            return None
        refiner_cov = _applied_refiner_coverage(str(item.get("applied", "branch")))
        sig.append((m.group(1), round(float(item.get("strength", 1.0)), 4), refiner_cov))
    return sig


_KNOWN_H3_CHECKPOINTS: dict[str, frozenset[str]] = {
    "fl2va": frozenset(
        {
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "minimax_h3_fl2va_int8_convrot.safetensors",
            "minimax_h3_fl2va_bf16.safetensors",
            "p-h3-image_int8_convrot.safetensors",
        }
    ),
    "ref2va": frozenset(
        {
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "minimax_h3_ref2va_int8_convrot.safetensors",
            "minimax_h3_ref2va_bf16.safetensors",
        }
    ),
}

_KNOWN_REF2VA_GUIDE_LORAS: frozenset[str] = frozenset(
    {
        "minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors",
        "minimax_h3_inpainting_v1.0_r128_ffp.safetensors",
        "minimax_h3_camera_control_v1.0_r128_ffp.safetensors",
        "minimax_h3_motion_transfer_v1.0_r128_ffp.safetensors",
    }
)


def _is_known_h3_family_lora(item: Any, family: str) -> bool:
    if not isinstance(item, Mapping):
        return False
    if _applied_refiner_coverage(str(item.get("applied", "branch"))) != "full_refiner":
        return False
    if round(float(item.get("strength", 1.0)), 4) != 1.0:
        return False
    raw_file = Path(str(item.get("file", item.get("name", "")))).name
    m = _PIKA_FLEX_VARIANT_RE.match(raw_file)
    if m is not None and m.group(1) == f"pika_h3_flex_{family}":
        return True
    if raw_file == "p-h3-image_lora_bf16.safetensors":
        return True
    if family == "ref2va" and raw_file in _KNOWN_REF2VA_GUIDE_LORAS:
        return True
    return False


def _is_same_family_h3_recipe_compatible(
    vec_recipe: Mapping[str, Any],
    cur_recipe: Mapping[str, Any],
    *,
    vec_family: str,
    loaded_family: str,
) -> bool:
    """Whether `vec_recipe` and `cur_recipe` are standard H3 `fl2va` or `ref2va` checkpoints
    and full-refiner distillation/task LoRA recipes (base `loras=[]`, Pika H3 Flex, PrunaAI
    `P-H3-Image`, or `ref2va` Guide Video Edit) of the same H3 text-path family."""
    if vec_family != loaded_family or vec_family not in ("fl2va", "ref2va"):
        return False
    cur_ckpt = cur_recipe.get("checkpoint")
    vec_ckpt = vec_recipe.get("checkpoint")
    cur_ckpt_file = Path(str(cur_ckpt.get("file", ""))).name if isinstance(cur_ckpt, Mapping) else ""
    vec_ckpt_file = Path(str(vec_ckpt.get("file", ""))).name if isinstance(vec_ckpt, Mapping) else ""
    known_ckpts = _KNOWN_H3_CHECKPOINTS[vec_family]
    if vec_ckpt != cur_ckpt and not (vec_ckpt_file in known_ckpts and cur_ckpt_file in known_ckpts):
        return False
    cur_loras = cur_recipe.get("loras", [])
    vec_loras = vec_recipe.get("loras", [])
    if not isinstance(cur_loras, Sequence) or not isinstance(vec_loras, Sequence):
        return False
    if not all(_is_known_h3_family_lora(item, vec_family) for item in vec_loras):
        return False
    if not all(_is_known_h3_family_lora(item, loaded_family) for item in cur_loras):
        return False
    return True


_is_ph3_image_compatible = _is_same_family_h3_recipe_compatible


APPLY_MODES: tuple[str, ...] = ("add", "add_norm_matched", "hold", "project_out", "amplify")
SCALE_MODES: tuple[str, ...] = ("raw", "norm_matched")
#: The post-refiner shift norms a push is meant to land in. **Measured**: the band of `raw_norm` across the
#: bank's `diff_span` builds (`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`), rounded.
NORM_SHIFT_WINDOW: tuple[float, float] = (50.0, 85.0)
#: Midpoint of `NORM_SHIFT_WINDOW`: the length `add_norm_matched` rescales every vector to.
NORM_MATCHED_TARGET: float = 67.5
assert NORM_MATCHED_TARGET == sum(NORM_SHIFT_WINDOW) / 2.0


def strength_ranges(raw_norm: float) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """The strength range that lands a push inside `NORM_SHIFT_WINDOW`, per apply mode.

    Returns `(norm_matched, add)`. Under `add_norm_matched`, the node's default mode, every vector is
    rescaled to `NORM_MATCHED_TARGET` first, so the range is the same for every vector. Under `add` the
    vector is used as stored, so the range depends on its own `raw_norm`; it is `None` for a zero vector.
    """
    lo, hi = NORM_SHIFT_WINDOW
    matched = (lo / NORM_MATCHED_TARGET, hi / NORM_MATCHED_TARGET)
    add = (lo / raw_norm, hi / raw_norm) if raw_norm > 1e-9 else None
    return matched, add


def _apply_holds_to_shift64(
    total_shift64: torch.Tensor,
    hold_entries: Sequence[tuple[torch.Tensor, float]],
    device: torch.device,
) -> torch.Tensor:
    """Project `hold_entries` (`[(v_hold, alpha), ...]`) out of `total_shift64` in float64."""
    if not hold_entries or float(torch.linalg.vector_norm(total_shift64).item()) <= 1e-24:
        return total_shift64
    ortho_holds: list[tuple[torch.Tensor, float]] = []
    for h_vec, h_alpha in hold_entries:
        h_alpha_f = float(h_alpha)
        if not math.isfinite(h_alpha_f) or not (0.0 <= h_alpha_f <= 2.0):
            raise ValueError(f"hold alpha must be finite and in [0.0, 2.0], got {h_alpha!r}")
        u = h_vec.detach().to(device=device, dtype=torch.float64).clone()
        for _pass in range(2):
            for prev_u, _ in ortho_holds:
                u = u - torch.dot(u, prev_u) * prev_u
        nrm = torch.linalg.vector_norm(u)
        if float(nrm) > 1e-12:
            ortho_holds.append((u / nrm, h_alpha_f))
    out_shift = total_shift64
    for u_hat, h_alpha in ortho_holds:
        out_shift = out_shift - h_alpha * torch.dot(out_shift, u_hat) * u_hat
    return out_shift


def vector_interactions(
    entries: Sequence[tuple[str, torch.Tensor, float]],
) -> dict[str, Any]:
    """Compute exact float64 norm decomposition and pairwise quadratic interaction
    terms `2 * s_i * s_j * (v_i . v_j)` across stacked additive control vectors.

    Each element of `entries` is `(name, vector_1d, effective_strength)`."""
    vecs64: list[tuple[str, torch.Tensor, float]] = []
    for name, vec, strength in entries:
        s_f = float(strength)
        if s_f == 0.0:
            continue
        v64 = vec.detach().to(device="cpu", dtype=torch.float64).flatten()
        vecs64.append((str(name), v64, s_f))

    if not vecs64:
        return {
            "combined_norm": 0.0,
            "sum_individual_norms": 0.0,
            "combined_norm_sq": 0.0,
            "sum_individual_norm_sq": 0.0,
            "total_interaction_sq": 0.0,
            "pairs": [],
        }

    d = vecs64[0][1].shape[0]
    total_vec = torch.zeros(d, dtype=torch.float64)
    sum_ind_norms = 0.0
    sum_ind_norm_sq = 0.0
    for _name, v64, s_f in vecs64:
        scaled = s_f * v64
        total_vec = total_vec + scaled
        nrm = float(torch.linalg.vector_norm(scaled).item())
        sum_ind_norms += nrm
        sum_ind_norm_sq += nrm * nrm

    pairs: list[dict[str, Any]] = []
    total_inter_sq = 0.0
    for i in range(len(vecs64)):
        name_i, v_i, s_i = vecs64[i]
        norm_i = float(torch.linalg.vector_norm(v_i).item())
        for j in range(i + 1, len(vecs64)):
            name_j, v_j, s_j = vecs64[j]
            norm_j = float(torch.linalg.vector_norm(v_j).item())
            dot_ij = float(torch.dot(v_i, v_j).item())
            cos_ij = dot_ij / (norm_i * norm_j) if (norm_i > 1e-12 and norm_j > 1e-12) else 0.0
            sign_ij = 1.0 if (s_i * s_j) >= 0.0 else -1.0
            signed_cos_ij = sign_ij * cos_ij
            inter_sq = 2.0 * s_i * s_j * dot_ij
            total_inter_sq += inter_sq
            pairs.append(
                {
                    "a": name_i,
                    "b": name_j,
                    "strength_a": s_i,
                    "strength_b": s_j,
                    "cosine": cos_ij,
                    "signed_cosine": signed_cos_ij,
                    "dot": dot_ij,
                    "interaction_sq": inter_sq,
                }
            )

    comb_norm_sq = float(torch.dot(total_vec, total_vec).item())
    return {
        "combined_norm": math.sqrt(max(0.0, comb_norm_sq)),
        "sum_individual_norms": sum_ind_norms,
        "combined_norm_sq": comb_norm_sq,
        "sum_individual_norm_sq": sum_ind_norm_sq,
        "total_interaction_sq": total_inter_sq,
        "pairs": pairs,
    }


def apply_post_refiner(
    context: torch.Tensor,
    tags: torch.Tensor,
    vectors: Sequence[tuple[torch.Tensor, float] | tuple[torch.Tensor, float, str]],
    *,
    mode: str = "add",
    scale_mode: str = "raw",
    mask: torch.Tensor | None = None,
    mask_vision: bool = True,
) -> torch.Tensor:
    """Apply one or more `(vector_post_refiner, strength[, mode])` specs to refined
    text conditioning `context` (`[B, L, 5376]`).

    Modes (`mode` or per-entry third tuple element):
    - `'add'`: additive shift `r -> r + s * v` (consecutive `'add'` vectors sum in
      `float64` before one cast to `context.dtype`). When `scale_mode='norm_matched'`
      (or `mode='add_norm_matched'`), `v` is normalized to length `NORM_MATCHED_TARGET`
      (`67.5`) so `strength = 1.0` produces a `67.5` post-refiner shift norm regardless
      of pooling mode (`diff_span`, `from_first_diff`, `scene_text`).
    - `'add_norm_matched'`: additive shift with `v` scaled to `NORM_MATCHED_TARGET` (`67.5`).
    - `'hold'`: projects out fraction `s` (`alpha in [0, 1]`) of `v` from the accumulated
      additive steering shift `delta_r` (`delta_r -> delta_r - s * (delta_r . u_hat) * u_hat`)
      before `delta_r` is added to `context`, without scrubbing the base prompt's own
      tokens (`r`).
    - `'project_out'`: removes fraction `s` of the existing component along `v` from the
      token activations themselves, `r -> r - s * ((r . v) / ||v||^2) * v` (at `s = 1.0`,
      leaves zero component along `v`; if `r . v == 0`, leaves `r` unchanged).
    - `'amplify'`: scales the existing component along `v` by `(1 + s)`,
      `r -> r + s * ((r . v) / ||v||^2) * v` (only moves rows that already carry a
      component along `v`).

    Invariants guaranteed:
    1. When all strengths are `0.0` (or `vectors` is empty), returns `context`
       bitwise unchanged.
    2. When `mask_vision=True` (default), rows where `tags == 0` (vision rows) or
       where `mask` is `False` are bitwise unchanged; when `mask_vision=False`
       (the unmasked E2.3 / MAINodes-style ablation), all rows matching `mask`
       (or all `L` rows if `mask is None`) are steered.
    3. Sequence length `L` and shape `[B, L, 5376]` are unchanged."""
    if mode not in APPLY_MODES:
        raise ValueError(f"mode must be one of {APPLY_MODES}, got {mode!r}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
    if context.ndim != 3 or context.shape[-1] != POST_REFINER_DIM:
        raise ValueError(
            f"context must have shape [B, L, {POST_REFINER_DIM}], got {tuple(context.shape)}"
        )
    B, L, D = context.shape

    if tags.ndim == 1:
        tag_2d = tags.unsqueeze(0).expand(B, -1)
    elif tags.ndim == 2:
        tag_2d = tags
    else:
        raise ValueError(f"tags must be 1D [L] or 2D [B, L], got shape {tuple(tags.shape)}")
    if tag_2d.shape != (B, L):
        raise ValueError(
            f"tags shape {tuple(tag_2d.shape)} does not match context batch/length {(B, L)}"
        )

    base_mask = (
        (tag_2d != 0).to(device=context.device)
        if mask_vision
        else torch.ones((B, L), dtype=torch.bool, device=context.device)
    )
    if mask is None:
        row_mask = base_mask
    else:
        if mask.ndim == 1:
            row_mask = mask.to(device=context.device, dtype=torch.bool).unsqueeze(0).expand(B, -1)
        elif mask.ndim == 2:
            row_mask = mask.to(device=context.device, dtype=torch.bool)
        else:
            raise ValueError(f"mask must be 1D [L] or 2D [B, L], got shape {tuple(mask.shape)}")
        if row_mask.shape != (B, L):
            raise ValueError(
                f"mask shape {tuple(row_mask.shape)} does not match context batch/length {(B, L)}"
            )
        row_mask = row_mask & base_mask

    active: list[tuple[torch.Tensor, float, str]] = []
    for entry in vectors:
        if len(entry) == 2:
            vec, strength = entry
            entry_mode = mode
        elif len(entry) == 3:
            vec, strength, entry_mode = entry
        else:
            raise ValueError(f"vector entry must be (vec, strength[, mode]), got {entry!r}")
        if entry_mode not in APPLY_MODES:
            raise ValueError(f"mode must be one of {APPLY_MODES}, got {entry_mode!r}")
        if not math.isfinite(strength):
            raise ValueError(f"control vector strength must be finite, got {strength!r}")
        if tuple(vec.shape) != (D,):
            raise ValueError(f"vector must have shape ({D},), got {tuple(vec.shape)}")
        if not bool(torch.isfinite(vec).all()):
            raise ValueError("control vector contains non-finite values")
        v_norm_sq = float(torch.dot(vec.detach().to(dtype=torch.float64), vec.detach().to(dtype=torch.float64)).item())
        use_norm_match = entry_mode == "add_norm_matched" or (
            entry_mode == "add" and scale_mode == "norm_matched"
        )
        if (entry_mode in ("hold", "project_out", "amplify") or use_norm_match) and v_norm_sq <= 1e-24:
            raise ValueError(f"control vector has zero norm; cannot apply mode={entry_mode!r}")
        if strength != 0.0:
            if use_norm_match:
                eff_strength = float(strength) * (NORM_MATCHED_TARGET / math.sqrt(v_norm_sq))
                active.append((vec, eff_strength, "add"))
            else:
                active.append((vec, float(strength), entry_mode))

    if not active or not bool(row_mask.any()):
        return context

    if not bool(torch.isfinite(context[row_mask]).all()):
        raise ValueError("context contains non-finite values in steered rows")

    out = context.clone()
    if all(em in ("add", "hold") for _, _, em in active):
        total_shift64 = torch.zeros(D, dtype=torch.float64, device=context.device)
        hold_entries: list[tuple[torch.Tensor, float]] = []
        for vec, strength, em in active:
            if em == "add":
                total_shift64 = total_shift64 + strength * vec.detach().to(
                    device=context.device, dtype=torch.float64
                )
            else:
                hold_entries.append((vec, strength))
        if hold_entries:
            total_shift64 = _apply_holds_to_shift64(total_shift64, hold_entries, context.device)
        if float(torch.linalg.vector_norm(total_shift64).item()) > 0.0:
            shift = total_shift64.to(dtype=context.dtype)
            out[row_mask] = out[row_mask] + shift
    else:
        rows64 = out[row_mask].to(dtype=torch.float64)
        idx = 0
        while idx < len(active):
            vec, strength, em = active[idx]
            if em in ("add", "hold"):
                total_shift64 = torch.zeros(D, dtype=torch.float64, device=context.device)
                hold_entries_blk: list[tuple[torch.Tensor, float]] = []
                while idx < len(active) and active[idx][2] in ("add", "hold"):
                    v_i, s_i, em_i = active[idx]
                    if em_i == "add":
                        total_shift64 = total_shift64 + s_i * v_i.detach().to(
                            device=context.device, dtype=torch.float64
                        )
                    else:
                        hold_entries_blk.append((v_i, s_i))
                    idx += 1
                if hold_entries_blk:
                    total_shift64 = _apply_holds_to_shift64(
                        total_shift64, hold_entries_blk, context.device
                    )
                rows64 = rows64 + total_shift64.unsqueeze(0)
            else:
                v64 = vec.detach().to(device=context.device, dtype=torch.float64)
                v_norm_sq = float(torch.dot(v64, v64).item())
                coeff = (rows64 @ v64) / v_norm_sq
                delta = (strength * coeff).unsqueeze(-1) * v64.unsqueeze(0)
                if em == "project_out":
                    rows64 = rows64 - delta
                else:
                    rows64 = rows64 + delta
                idx += 1
        out[row_mask] = rows64.to(dtype=context.dtype)

    if out.shape != (B, L, D):
        raise RuntimeError(f"apply_post_refiner changed context shape from {(B, L, D)} to {tuple(out.shape)}")
    return out


def row_norm_census(rows: torch.Tensor, tags: torch.Tensor | None = None) -> dict[str, float]:
    """Compute E2.1 per-row norm census on `[L, D]` or `[1, L, D]` activation rows.

    Returns `row0_norm`, `median_row_norm` (over tag-1 rows), `median_row_norm_ex_row0`,
    and `row0_to_median_ratio`."""
    r = rows[0] if rows.ndim == 3 else rows
    if r.ndim != 2 or r.shape[0] < 1:
        raise ValueError(f"rows must have shape [L, D] or [1, L, D] with L >= 1, got {tuple(rows.shape)}")
    r64 = r.detach().to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(r64, dim=-1)
    row0 = float(norms[0].item())
    if tags is not None:
        t_flat = tags.flatten().to(device=r64.device)
        tag1 = norms[t_flat != 0]
        tag1_ex0 = norms[1:][t_flat[1:] != 0] if norms.shape[0] > 1 else tag1
    else:
        tag1 = norms
        tag1_ex0 = norms[1:] if norms.shape[0] > 1 else norms
    med_all = float(tag1.median().item()) if tag1.numel() > 0 else row0
    med_ex0 = float(tag1_ex0.median().item()) if tag1_ex0.numel() > 0 else med_all
    return {
        "row0_norm": row0,
        "median_row_norm": med_all,
        "median_row_norm_ex_row0": med_ex0,
        "row0_to_median_ratio": row0 / med_ex0 if med_ex0 > 0.0 else 1.0,
    }


def channel_census(vec: torch.Tensor) -> dict[str, float]:
    """Compute E2.1 channel-concentration statistics on a 1D vector `[D]`.

    Returns `top1_energy_share`, `top5_energy_share`, `top10_energy_share`, and
    `max_abs_over_rms`."""
    if vec.ndim != 1 or vec.shape[0] < 1:
        raise ValueError(f"vec must be 1D [D], got {tuple(vec.shape)}")
    v64 = vec.detach().to(dtype=torch.float64)
    sq = v64 * v64
    total_energy = float(sq.sum().item())
    if total_energy <= 0.0:
        raise ValueError("vec has zero energy")
    sorted_sq, _ = torch.sort(sq, descending=True)
    d = v64.shape[0]
    rms = math.sqrt(total_energy / d)
    max_abs = float(v64.abs().max().item())
    return {
        "top1_energy_share": float(sorted_sq[:1].sum().item() / total_energy),
        "top5_energy_share": float(sorted_sq[: min(5, d)].sum().item() / total_energy),
        "top10_energy_share": float(sorted_sq[: min(10, d)].sum().item() / total_energy),
        "max_abs_over_rms": max_abs / rms if rms > 0.0 else 0.0,
    }


def meter_score(
    rows: torch.Tensor,
    vec: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Project pooled activation rows (`[L, D]` or `[1, L, D]`) onto `vec` (`[D]`).

    Returns:
    - `projection_units`: scalar projection `(pooled . v) / ||v||^2`, i.e. how many
      mean-pair-difference units (`strength=1`) the prompt sits along `vec`,
    - `signed_projection`: `(pooled . v) / ||v||` in Euclidean norm units,
    - `cosine`: cosine similarity between the pooled prompt row and `vec`,
    - `max_token_cosine`: maximum per-token cosine across active rows,
    - `p90_token_cosine`: 90th-percentile per-token cosine across active rows,
    - `per_token_cosines`: list of per-row cosines across all `L` rows,
    - `top_coupled_tokens`: top-8 `(index, token, cosine)` dicts when `tokens` is passed."""
    r = rows[0] if rows.ndim == 3 else rows
    if r.ndim != 2 or r.shape[1] != vec.shape[0]:
        raise ValueError(
            f"rows shape {tuple(rows.shape)} incompatible with vec shape {tuple(vec.shape)}"
        )
    r64 = r.detach().to(dtype=torch.float64)
    if mask is not None:
        m = mask.flatten().to(device=r64.device, dtype=torch.bool)
        if m.shape[0] != r64.shape[0] or not bool(m.any()):
            raise ValueError("meter_score mask must match row count L and select at least one row")
        pooled = r64[m].mean(dim=0)
        active_r64 = r64[m]
        active_indices = [i for i, flag in enumerate(m.tolist()) if flag]
    else:
        pooled = r64.mean(dim=0)
        active_r64 = r64
        active_indices = list(range(int(r64.shape[0])))
    v64 = vec.detach().to(device=r64.device, dtype=torch.float64)
    v_norm_sq = float(torch.dot(v64, v64).item())
    if v_norm_sq <= 0.0:
        raise ValueError("meter_score vector has zero norm")
    v_norm = math.sqrt(v_norm_sq)
    p_norm = float(torch.linalg.vector_norm(pooled).item())
    dot = float(torch.dot(pooled, v64).item())

    v_unit = v64 / v_norm
    row_norms = torch.linalg.vector_norm(r64, dim=-1).clamp(min=1e-12)
    per_tok_cos_t = (r64 @ v_unit) / row_norms
    per_tok_cos_list = [float(x) for x in per_tok_cos_t.tolist()]
    active_cos_t = per_tok_cos_t[active_indices] if active_indices else per_tok_cos_t
    max_tok_cos = float(active_cos_t.max().item()) if active_cos_t.numel() > 0 else 0.0
    p90_tok_cos = (
        float(torch.quantile(active_cos_t, 0.90).item()) if active_cos_t.numel() > 0 else 0.0
    )
    top_coupled: list[dict[str, Any]] = []
    if tokens is not None:
        for idx_i in active_indices:
            tok_s = str(tokens[idx_i]).strip() if idx_i < len(tokens) else ""
            if tok_s:
                top_coupled.append(
                    {"index": idx_i, "token": tok_s, "cosine": float(per_tok_cos_list[idx_i])}
                )
        top_coupled.sort(key=lambda d: d["cosine"], reverse=True)
        top_coupled = top_coupled[:8]

    cos_val = dot / (p_norm * v_norm) if p_norm > 0.0 else 0.0
    proj_units = dot / v_norm_sq
    return {
        "projection_units": proj_units,
        "z_score": proj_units,
        "signed_projection": dot / v_norm,
        "cosine": cos_val,
        "mean_cosine": cos_val,
        "max_token_cosine": max_tok_cos,
        "p90_token_cosine": p90_tok_cos,
        "p90_cosine": p90_tok_cos,
        "per_token_cosines": per_tok_cos_list,
        "top_coupled_tokens": top_coupled,
    }
