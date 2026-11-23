"""Contrastive pair TOML loader (`[vector]`, `[[pair]]`, `[[hold]]`, `[routing]`, `[provenance]`),
differing-span validator, Markdown converter, and Gates A/B/C/D + setting-leak linter."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import (
    CANONICAL_PROVENANCE_KEYS,
    DEFAULT_EXCLUDE_SHOTS,
    FADER_MAX,
    FADER_MIN,
    MAX_SHARED_INTERIOR_WORDS,
    NAME_RE,
    ROUTING_FAMILIES,
    SECTION_AND_SHOT_MARKERS,
    SECTION_HEADER_PREFIXES,
    STEER_BLOCK_BANDS,
    STEER_BUSES,
    STEER_REGIMES,
    STEER_STREAMS,
    STEER_TARGETS,
    VALID_FRAMES,
    VALID_MODES,
    import_repo_module,
)

HEADER_TOKEN_RE = import_repo_module("h3_prompt_edit").HEADER_TOKEN_RE
_h3_spans = import_repo_module("h3_spans")

_REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_TOP_KEYS = frozenset({"vector", "pair", "hold", "routing", "provenance"})
ALLOWED_VECTOR_KEYS = frozenset(
    {"name", "quality", "a_side", "b_side", "mode", "frame", "exclude_shots", "allow_multi_span"}
)
ALLOWED_PAIR_KEYS = frozenset({"weight", "a", "b"})
ALLOWED_HOLD_KEYS = frozenset({"group", "weight", "a", "b"})
ALLOWED_ROUTING_KEYS = frozenset(
    {
        "bus",
        "regime",
        "target",
        "fields",
        "streams",
        "block_band",
        "needs_ref_release",
        "families",
        "default_strength",
    }
)
ALLOWED_PROVENANCE_KEYS = frozenset(
    {
        *CANONICAL_PROVENANCE_KEYS,
        "core_revision",
        "pack_commit",
        "made_by",
    }
)

_BAND_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


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
class RoutingSpec:
    bus: str = "look"
    regime: str = "a"
    target: str = "scene_only"
    fields: tuple[str, ...] = ()
    streams: str = "both"
    block_band: str = "all"
    needs_ref_release: bool = False
    families: tuple[str, ...] = ("fl2va", "ref2va", "h3_image")
    default_strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bus": self.bus,
            "regime": self.regime,
            "target": self.target,
            "fields": list(self.fields),
            "streams": self.streams,
            "block_band": self.block_band,
            "needs_ref_release": bool(self.needs_ref_release),
            "families": list(self.families),
            "default_strength": float(self.default_strength),
        }


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
    routing: RoutingSpec | None = None
    provenance: dict[str, Any] | None = None


def _validate_routing_block_band(band_spec: Any, source: str) -> str:
    raw = str(band_spec if band_spec is not None else "all").strip() or "all"
    if raw in (*STEER_BLOCK_BANDS, "none"):
        return raw
    m = _BAND_RANGE_RE.match(raw)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo < 0 or hi < lo:
            raise ValueError(f"{source}: block_band range 'A-B' requires 0 <= A <= B, got {raw!r}")
        return f"{lo}-{hi}"
    raise ValueError(
        f"{source}: routing.block_band must be one of {STEER_BLOCK_BANDS} (or 'none' / 'A-B'), got {band_spec!r}"
    )


def parse_routing_table(raw: Mapping[str, Any], *, source: str = "<routing>") -> RoutingSpec:
    """Validate and normalize a `[routing]` table or metadata dict."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: [routing] must be a mapping/table")
    unknown = sorted(set(raw.keys()) - ALLOWED_ROUTING_KEYS)
    if unknown:
        raise ValueError(
            f"{source}: unknown key(s) in [routing]: {unknown}; expected only {sorted(ALLOWED_ROUTING_KEYS)}"
        )

    bus = str(raw.get("bus", "look")).strip()
    if bus not in STEER_BUSES:
        raise ValueError(f"{source}: routing.bus must be one of {STEER_BUSES}, got {bus!r}")

    regime = str(raw.get("regime", "a")).strip()
    if regime not in STEER_REGIMES:
        raise ValueError(
            f"{source}: routing.regime must be one of {STEER_REGIMES} in v1 core, got {regime!r}"
        )

    target = str(raw.get("target", "scene_only")).strip()
    if target not in STEER_TARGETS:
        raise ValueError(f"{source}: routing.target must be one of {STEER_TARGETS}, got {target!r}")

    fields_raw = raw.get("fields", ())
    if not isinstance(fields_raw, (list, tuple)) or not all(isinstance(f, str) for f in fields_raw):
        raise ValueError(f"{source}: routing.fields must be a list of prompt field strings")
    fields = tuple(f.strip() for f in fields_raw if f.strip())
    if fields:
        _h3_spans.fields_mask(fields)

    streams = str(raw.get("streams", "both")).strip()
    if streams not in STEER_STREAMS:
        raise ValueError(f"{source}: routing.streams must be one of {STEER_STREAMS}, got {streams!r}")

    block_band = _validate_routing_block_band(raw.get("block_band", "all"), source)

    nrr = raw.get("needs_ref_release", False)
    if not isinstance(nrr, bool):
        raise ValueError(f"{source}: routing.needs_ref_release must be a boolean, got {nrr!r}")

    fams_raw = raw.get("families", ("fl2va", "ref2va", "h3_image"))
    if (
        not isinstance(fams_raw, (list, tuple))
        or not fams_raw
        or not all(isinstance(x, str) and x in ROUTING_FAMILIES for x in fams_raw)
    ):
        raise ValueError(
            f"{source}: routing.families must be a non-empty list from {ROUTING_FAMILIES}, got {fams_raw!r}"
        )
    families = tuple(dict.fromkeys(str(x) for x in fams_raw))

    ds_raw = raw.get("default_strength", 1.0)
    if isinstance(ds_raw, bool) or not isinstance(ds_raw, (int, float)):
        raise ValueError(f"{source}: routing.default_strength must be a number, got {ds_raw!r}")
    ds = float(ds_raw)
    if not math.isfinite(ds) or not (FADER_MIN <= ds <= FADER_MAX):
        raise ValueError(
            f"{source}: routing.default_strength must lie in [{FADER_MIN}, {FADER_MAX}], got {ds_raw!r}"
        )

    return RoutingSpec(
        bus=bus,
        regime=regime,
        target=target,
        fields=fields,
        streams=streams,
        block_band=block_band,
        needs_ref_release=nrr,
        families=families,
        default_strength=ds,
    )


def parse_provenance_table(raw: Mapping[str, Any], *, source: str = "<provenance>") -> dict[str, Any]:
    """Validate and normalize an optional `[provenance]` table."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: [provenance] must be a mapping/table")
    unknown = sorted(set(raw.keys()) - ALLOWED_PROVENANCE_KEYS)
    if unknown:
        raise ValueError(
            f"{source}: unknown key(s) in [provenance]: {unknown}; "
            f"expected canonical keys {list(CANONICAL_PROVENANCE_KEYS)}"
        )
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None or isinstance(v, (str, bool, int)):
            out[str(k)] = v
        elif isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(f"{source}: provenance.{k} must be finite, got {v!r}")
            out[str(k)] = v
        else:
            raise ValueError(
                f"{source}: provenance.{k} must be a scalar (str, int, float, bool), got {type(v).__name__}"
            )
    return out


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def _multi_span_section_diffs(
    sec_a: str,
    sec_b: str,
    *,
    label: str = "pair",
) -> tuple[list[str], list[str]]:
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
    """Return the single contiguous differing clause `(a_span, b_span)` between `a` and `b`."""
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
                if len(diff_idxs) >= 2:
                    matched_hdrs: list[str] = []
                    for i_sec in diff_idxs:
                        ba_s = blocks_a[i_sec].lstrip()
                        bb_s = blocks_b[i_sec].lstrip()
                        hdr = next(
                            (h for h in SECTION_HEADER_PREFIXES if ba_s.startswith(h) and bb_s.startswith(h)),
                            None,
                        )
                        if hdr is None:
                            matched_hdrs = []
                            break
                        matched_hdrs.append(hdr[:-1])
                    if len(matched_hdrs) == len(diff_idxs):
                        spans_a: list[str] = []
                        spans_b: list[str] = []
                        for i_sec, sec_lbl in zip(diff_idxs, matched_hdrs):
                            sa_i, sb_i = differing_span(
                                blocks_a[i_sec],
                                blocks_b[i_sec],
                                frame="scene",
                                label=f"{label} ({sec_lbl})",
                            )
                            spans_a.append(sa_i)
                            spans_b.append(sb_i)
                        return " | ".join(spans_a), " | ".join(spans_b)
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


def load_pairs(path: str | Path, *, bank_dir: str | Path | None = None) -> PairSet:
    """Parse and validate a control-vector pair TOML file (`[vector]`, `[[pair]]`, `[[hold]]`,
    optional `[routing]`, and optional `[provenance]`)."""
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

    routing = (
        parse_routing_table(doc["routing"], source=str(p))
        if "routing" in doc
        else None
    )
    provenance = (
        parse_provenance_table(doc["provenance"], source=str(p))
        if "provenance" in doc
        else None
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
        routing=routing,
        provenance=provenance,
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
    """Return `(sec_idx, section_name)` for the prompt section(s) where `a` and `b` differ."""
    if frame == "bare":
        return 0, "bare"
    min_len = min(len(a), len(b))
    c_prefix = 0
    while c_prefix < min_len and a[c_prefix] == b[c_prefix]:
        c_prefix += 1
    sec_matches_a = list(_SECTION_ONLY_RE.finditer(a))
    if not sec_matches_a:
        return 0, "text"
    chosen_idx = 0
    chosen_name = sec_matches_a[0].group(1)
    for idx, m in enumerate(sec_matches_a):
        if m.start() <= c_prefix:
            chosen_idx = idx
            chosen_name = m.group(1)
        else:
            break
    sec_matches_b = list(_SECTION_ONLY_RE.finditer(b))
    if (
        len(sec_matches_a) == len(sec_matches_b)
        and [m.group(1) for m in sec_matches_a] == [m.group(1) for m in sec_matches_b]
    ):
        diff_sec_names: list[str] = []
        for idx_s, ma in enumerate(sec_matches_a):
            mb = sec_matches_b[idx_s]
            end_a = sec_matches_a[idx_s + 1].start() if idx_s + 1 < len(sec_matches_a) else len(a)
            end_b = sec_matches_b[idx_s + 1].start() if idx_s + 1 < len(sec_matches_b) else len(b)
            if a[ma.start() : end_a].rstrip() != b[mb.start() : end_b].rstrip():
                diff_sec_names.append(ma.group(1))
        if len(diff_sec_names) >= 2:
            return chosen_idx, "+".join(diff_sec_names)
        if len(diff_sec_names) == 1:
            first_name = diff_sec_names[0]
            first_idx = next(
                (i for i, m in enumerate(sec_matches_a) if m.group(1) == first_name),
                chosen_idx,
            )
            return first_idx, first_name
    return chosen_idx, chosen_name


def pair_set_differing_sections(
    pairs: Sequence[PairItem],
    *,
    frame: str = "scene",
) -> list[str]:
    """Return the ordered list of distinct prompt section names modified across `pairs`."""
    out: list[str] = []
    for item in pairs:
        _, sec_str = differing_section_info(item.a, item.b, frame=frame)
        for part in sec_str.split("+"):
            clean = part.strip()
            if clean and clean not in out:
                out.append(clean)
    return out


def parse_markdown_pairs(md_text: str) -> dict[str, Any]:
    """Parse an LLM-authored control-vector Markdown document into structured data."""
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
    if "'''" not in text and not text.endswith("'"):
        return f"'''{text}'''"
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if escaped.endswith('"'):
        escaped = escaped[:-1] + '\\"'
    return f'"""{escaped}"""'


def format_routing_toml_lines(routing: RoutingSpec | Mapping[str, Any]) -> list[str]:
    spec = routing if isinstance(routing, RoutingSpec) else parse_routing_table(routing)
    d = spec.to_dict()
    lines = [
        "[routing]",
        f"bus = {json.dumps(d['bus'], ensure_ascii=False)}",
        f"regime = {json.dumps(d['regime'], ensure_ascii=False)}",
        f"target = {json.dumps(d['target'], ensure_ascii=False)}",
    ]
    if d["fields"]:
        lines.append(f"fields = {json.dumps(d['fields'], ensure_ascii=False)}")
    lines.extend(
        [
            f"streams = {json.dumps(d['streams'], ensure_ascii=False)}",
            f"block_band = {json.dumps(d['block_band'], ensure_ascii=False)}",
            f"needs_ref_release = {'true' if d['needs_ref_release'] else 'false'}",
            f"families = {json.dumps(d['families'], ensure_ascii=False)}",
            f"default_strength = {float(d['default_strength']):g}",
        ]
    )
    return lines


def format_provenance_toml_lines(provenance: Mapping[str, Any]) -> list[str]:
    clean = parse_provenance_table(provenance)
    lines = ["[provenance]"]
    ordered_keys = [k for k in CANONICAL_PROVENANCE_KEYS if k in clean] + [
        k for k in sorted(clean.keys()) if k not in CANONICAL_PROVENANCE_KEYS
    ]
    for k in ordered_keys:
        v = clean[k]
        if v is None:
            continue
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, int):
            lines.append(f"{k} = {v}")
        elif isinstance(v, float):
            lines.append(f"{k} = {v:g}")
        else:
            lines.append(f"{k} = {json.dumps(str(v), ensure_ascii=False)}")
    return lines


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
    routing: RoutingSpec | Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
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

    resolved_mode = mode.strip() if mode is not None and mode.strip() else parsed["mode"]
    if resolved_mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {resolved_mode!r}")

    out_p = Path(out_path)
    if out_p.suffix == ".toml":
        target_toml = out_p
        resolved_name = name.strip() if name is not None and name.strip() else target_toml.stem
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

    p1_a_span, p1_b_span = differing_span(
        pairs[0]["a"], pairs[0]["b"], frame=frame, label=f"pair[0] (Pair {pairs[0]['number']})"
    )
    resolved_a_side = a_side.strip() if a_side is not None and a_side.strip() else p1_a_span
    resolved_b_side = b_side.strip() if b_side is not None and b_side.strip() else p1_b_span
    resolved_quality = (
        quality.strip()
        if quality is not None and quality.strip()
        else f"{resolved_b_side} vs. {resolved_a_side}"
    )

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

    if test_prompt_out is not None and not test_scene_prompt:
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

    if routing is not None:
        lines.extend(format_routing_toml_lines(routing))
        lines.append("")
    if provenance is not None:
        prov_lines = format_provenance_toml_lines(provenance)
        if len(prov_lines) > 1:
            lines.extend(prov_lines)
            lines.append("")

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

_CANCEL_STOPLIST: frozenset[str] = frozenset(
    {
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
    }
)

_BASELINE_STATE_WORDS: frozenset[str] = frozenset(
    {
        "sunlit", "sunlight", "daylight", "overcast", "tungsten", "fluorescent",
        "moonlight", "golden", "pastel", "vibrant", "colorful", "colourful",
        "vivid", "colorless", "monochrome", "grayscale", "greyscale", "matte",
        "glossy", "parched", "motionless", "static", "silent", "silence",
        "anechoic", "photorealistic", "naturalistic", "bright", "dry",
    }
)

_SETTING_LEAK_WORDS: frozenset[str] = frozenset(
    {
        "market", "bazaar", "stall", "kitchen", "diner", "subway", "station",
        "alley", "courtyard", "bakery", "harbor", "pier", "library", "hospital",
        "laboratory", "warehouse", "classroom", "office", "beach", "forest",
    }
)

_DIALOGUE_TAG_CHECK_RE = re.compile(r"<d>(?!\s*\[[A-Za-z][A-Za-z -]*\])")
_DIALOGUE_SPAN_STRIP_RE = re.compile(r"<d>.*?</d>", re.DOTALL | re.IGNORECASE)


def lint_pair_set(
    pairs: Any,
    *,
    frame: str = "scene",
    allow_repeated_spans: bool = False,
) -> list[str]:
    """Check semantic pair-quality rules (Gates A/B/C/D + setting-leak check) across a `PairSet`."""
    b_side_declared = ""
    if isinstance(pairs, PairSet):
        frame = pairs.frame
        items = list(pairs.pairs)
        b_side_declared = pairs.b_side.lower()
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

        words_a = {w for w in re.findall(r"[a-z]{4,}", a_lower) if w not in _CANCEL_STOPLIST}
        words_b = {w for w in re.findall(r"[a-z]{4,}", b_lower) if w not in _CANCEL_STOPLIST}
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

        # Gate A: boundary-append diff
        a_toks_raw = str(a_sp).split()
        b_toks_raw = str(b_sp).split()
        if frame == "scene" and 1 <= len(a_toks_raw) <= 2 and len(b_toks_raw) >= len(a_toks_raw) + 4:
            a_first_stem = re.sub(r"[^\w]+", "", a_toks_raw[0].lower())
            b_first_stem = re.sub(r"[^\w]+", "", b_toks_raw[0].lower())
            a_last_stem = re.sub(r"[^\w]+", "", a_toks_raw[-1].lower())
            b_last_stem = re.sub(r"[^\w]+", "", b_toks_raw[-1].lower())
            if (a_first_stem and a_first_stem == b_first_stem) or (a_last_stem and a_last_stem == b_last_stem):
                warnings.append(
                    f"Pair {num}: boundary-append diff ({a_sp!r} -> {b_sp[:48]!r}...); "
                    "replace a concrete baseline clause in-place instead of appending ', while ...' after punctuation"
                )

        # Gate D: shared prefix retains negated baseline state words
        if frame == "scene":
            neg_state_a = {
                w
                for w in re.findall(r"[a-z]{3,}", a_lower)
                if w in _BASELINE_STATE_WORDS and w not in set(re.findall(r"[a-z]{3,}", b_lower))
            }
            if neg_state_a:
                leaked_set: set[str] = set()
                for ba, bb in zip(a_txt.split("\n\n"), b_txt.split("\n\n")):
                    if ba == bb:
                        continue
                    wa_s, wb_s = ba.split(), bb.split()
                    fd_sec = next(
                        (k for k, (x, y) in enumerate(zip(wa_s, wb_s)) if x != y),
                        min(len(wa_s), len(wb_s)),
                    )
                    prefix_no_dlg = _DIALOGUE_SPAN_STRIP_RE.sub(" ", " ".join(wa_s[:fd_sec]))
                    pref_words = set(re.findall(r"[a-z]{3,}", prefix_no_dlg.lower()))
                    leaked_set.update(neg_state_a & pref_words)
                if leaked_set:
                    leaked = sorted(leaked_set)
                    warnings.append(
                        f"Pair {num}: shared prefix retains Pole-A baseline word(s) {leaked[:4]} negated by "
                        "a_span -> b_span; rewrite or move contradicting baseline adjectives inside the diff span"
                    )

    # Gate B (causal prefix diversity), Gate C (rotated Pole-A diversity), & setting-leak check
    if frame == "scene" and len(items) >= 4 and not allow_repeated_spans:
        first_diff_idxs: list[int] = []
        prefix_sigs: set[str] = set()
        pole_a_word_counts: dict[str, int] = {}
        pole_b_setting_counts: dict[str, int] = {}
        b_side_words = set(re.findall(r"[a-z]{4,}", b_side_declared))
        for item in items:
            a_txt = str(item.get("a", "")) if isinstance(item, Mapping) else str(getattr(item, "a", ""))
            b_txt = str(item.get("b", "")) if isinstance(item, Mapping) else str(getattr(item, "b", ""))
            a_sp = item.get("a_span") if isinstance(item, Mapping) else getattr(item, "a_span", None)
            b_sp = item.get("b_span") if isinstance(item, Mapping) else getattr(item, "b_span", None)
            if a_sp is None or b_sp is None:
                a_sp, b_sp = differing_span(a_txt, b_txt, frame=frame)
            wa, wb = a_txt.split(), b_txt.split()
            fd = next((i for i, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb)))
            first_diff_idxs.append(fd)
            prefix_sigs.add(" ".join(wa[:fd]).lower())
            for w in {t for t in re.findall(r"[a-z]{4,}", str(a_sp).lower()) if t not in _CANCEL_STOPLIST}:
                pole_a_word_counts[w] = pole_a_word_counts.get(w, 0) + 1
            for w in {
                t
                for t in re.findall(r"[a-z]{4,}", str(b_sp).lower())
                if t in _SETTING_LEAK_WORDS and t not in b_side_words
            }:
                pole_b_setting_counts[w] = pole_b_setting_counts.get(w, 0) + 1
        sorted_fd = sorted(first_diff_idxs)
        med_fd = sorted_fd[len(sorted_fd) // 2]
        min_distinct = min(len(items), max(3, len(items) // 2))
        if med_fd < 10 or len(prefix_sigs) < min_distinct:
            warnings.append(
                f"Causal prefix collapse: median word first_diff={med_fd} with {len(prefix_sigs)}/{len(items)} "
                "unique prefixes before diff; place a distinct location + subject before the diff in each pair "
                "so causal Qwen3-VL computes genuinely distinct contexts"
            )
        thresh_60 = max(4, int(math.ceil(len(items) * 0.6)))
        repeated_a = sorted(w for w, cnt in pole_a_word_counts.items() if cnt >= thresh_60)
        if repeated_a:
            warnings.append(
                f"Fixed Pole-A wording: words {repeated_a[:6]} repeat across >=60% of a_spans; "
                "rotate diverse Pole-A baselines so -E[A] does not subtract generic scene/lighting words"
            )
        leaked_settings = sorted(w for w, cnt in pole_b_setting_counts.items() if cnt >= thresh_60)
        if leaked_settings:
            warnings.append(
                f"Setting leak in b_span: venue noun(s) {leaked_settings[:4]} repeat across >=60% of b_spans "
                "outside declared b_side; diversify pair venues so B - A does not bake in one location"
            )

    return warnings
