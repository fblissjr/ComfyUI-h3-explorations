"""Partition an H3 conditioning token sequence into mutually exclusive span masks.

`comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer` builds an H3 conditioning
sequence by placing reference labels (`<Picture i>: `, `<Audio j>: `,
`<Video k>: `), video block timestamps (`<T.T seconds>`), and `<|vision_start|>`
.. `<|vision_end|>` vision blocks ahead of the raw prompt text, with no chat
template. Vision blocks carry `minimax_token_tags == 0` (including the flanking
start and end tokens); every text token carries `minimax_token_tags == 1`.

`spans(token_ids, tags, decode)` partitions a sequence of length `L` into six
mutually exclusive boolean masks (`vision`, `label`, `timestamp`, `dialogue`,
`speaker`, `scene`) whose sum is all-ones across `0 .. L-1`.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Callable, Sequence

import torch

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
DIALOGUE_START = 151669
DIALOGUE_END = 151670

CATEGORIES: tuple[str, ...] = (
    "vision",
    "label",
    "timestamp",
    "dialogue",
    "speaker",
    "scene",
)

POOLING_MODES: tuple[str, ...] = (
    "scene_text",
    "all_tag1",
    "from_first_diff",
    "diff_span",
)

#: Masks that select one prompt field by its own header name, so a vector can be aimed at any single
#: part of the prompt structure (owner, 2026-11-11). The description field is the one field that had
#: its own mask already (`description_only`, for `detailed_description:` or
#: `integrated_multimodal_description:`), so with these five every field has one. A field the prompt
#: does not have, or one whose body is `N/A`, selects no rows: these never fall back to another
#: field. Two fields at once is two chained nodes.
FIELD_MASK_MODES: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "overall_soundscape",
    "non_diegetic_music",
)

#: The fields of an H3 prompt in prompt order, as `MiniMaxH3ApplyControlVectors.fields` lists them.
#: `description` is `detailed_description:` or `integrated_multimodal_description:`, whichever the
#: prompt has. A selection of one or more is written `a+b` (`fields_mask`) and is itself a mask.
PROMPT_FIELDS: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "description",
    "overall_soundscape",
    "non_diegetic_music",
)
FIELDS_JOIN = "+"

INFERENCE_MASK_MODES: tuple[str, ...] = (
    "non_vision",
    "look_only",
    "scene_only",
    "description_only",
    "soundscape_only",
    "after_dialogue",
    "all_rows",
    *FIELD_MASK_MODES,
)

LABEL_RE = re.compile(r"<(?:Picture|Video|Audio) \d+>: ")
AUDIO_LABEL_PREFIX_RE = re.compile(r"^(?:<Audio \d+>: )+")
TIMESTAMP_RE = re.compile(r"<\d+(?:\.\d+)? seconds>")
SPEAKER_RE = re.compile(r"\(S\d+(?:,\s*S\d+)*\)")
SECTION_HEADER_RE = re.compile(
    r"(?:^|\n)[ \t]*(summary|subject_definitions|retention_analysis|"
    r"integrated_multimodal_description|detailed_description|"
    r"overall_soundscape|non_diegetic_music):"
)
_SUMMARY_TASK_TYPES = (
    r"(?:keyframe completion|reference generation|video editing|"
    r"video continuation|audio reuse|audio reference)"
)
_RETENTION_MODES = (
    r"(?:fully_preserved|partially_preserved|attribute_transfer|"
    r"weak_reference|fully_copy|partially_copy|reference)"
)
STRUCTURAL_TAG_RE = re.compile(
    rf"\[[ \t]*{_SUMMARY_TASK_TYPES}(?:[ \t]*\+[ \t]*{_SUMMARY_TASK_TYPES})*[ \t]*\]"
    rf"|<(?:\bSubject|\bPicture|\bVideo|\bAudio)\s+\d+>(?:\s*\([^)\n]*\))?\s*:\s*{_RETENTION_MODES}\s*-"
    r"|\[Shot\s+\d+\]|<(?:Subject|Picture|Video|Audio)\s+\d+>|(?:^|\n)[ \t]*Edit\s+<Picture\s+\d+>\."
)
AUDIO_RETENTION_LINE_RE = re.compile(r"^[ \t]*<Audio\s+\d+>")
LOOK_SECTIONS: frozenset[str] = frozenset({"summary", "subject_definitions"})
DESCRIPTION_SECTIONS: frozenset[str] = frozenset(
    {"integrated_multimodal_description", "detailed_description"}
)
SOUNDSCAPE_SECTIONS: frozenset[str] = frozenset(
    {"overall_soundscape", "non_diegetic_music"}
)
EMPTY_SECTION_BODIES: frozenset[str] = frozenset({"", "n/a", "na", "none"})


def fields_mask(fields: Sequence[str]) -> str:
    """The mask name for a selection of prompt fields: the chosen `PROMPT_FIELDS`, in prompt order,
    joined by `+` (`"subject_definitions+summary"`). Raises on an empty selection or an unknown field."""
    chosen = [str(f).strip() for f in (fields or ()) if str(f).strip()]
    unknown = sorted(set(chosen) - set(PROMPT_FIELDS))
    if unknown:
        raise ValueError(f"unknown prompt field(s) {unknown}; expected any of {PROMPT_FIELDS}")
    if not chosen:
        raise ValueError(f"no prompt field selected; expected one or more of {PROMPT_FIELDS}")
    return FIELDS_JOIN.join(f for f in PROMPT_FIELDS if f in chosen)


def split_fields_mask(mode: str) -> tuple[str, ...] | None:
    """The fields a mask name selects, or `None` when it is not a field selection. `"summary"` and
    `"summary+description"` are selections; `"scene_only"` is not."""
    parts = [p.strip() for p in str(mode or "").split(FIELDS_JOIN)]
    if not parts or any(p not in PROMPT_FIELDS for p in parts) or len(set(parts)) != len(parts):
        return None
    return tuple(f for f in PROMPT_FIELDS if f in parts)


def is_inference_mask(mode: str) -> bool:
    """True for a name in `INFERENCE_MASK_MODES` and for any selection of `PROMPT_FIELDS`."""
    return mode in INFERENCE_MASK_MODES or split_fields_mask(mode) is not None


def unwrap_hf_tokenizer(clip_or_tokenizer: Any) -> Any | None:
    """Extract the inner HuggingFace Qwen3-VL tokenizer from a ComfyUI CLIP
    wrapper, `MiniMaxH3Tokenizer` (`qwen3vl_32b`), `PH3ImageTokenizer`
    (`qwen3vl_4b`), or raw tokenizer object."""
    if clip_or_tokenizer is None:
        return None
    if (
        callable(getattr(clip_or_tokenizer, "decode", None))
        and callable(getattr(clip_or_tokenizer, "encode", None))
        and not hasattr(clip_or_tokenizer, "tokenize_with_weights")
    ):
        return clip_or_tokenizer
    tok_root = getattr(clip_or_tokenizer, "tokenizer", clip_or_tokenizer)
    mm_tok = getattr(tok_root, "minimax", tok_root)
    clip_attr = getattr(mm_tok, "clip", None)
    candidates: list[str] = []
    if isinstance(clip_attr, str) and clip_attr:
        candidates.append(clip_attr)
    for name in ("qwen3vl_32b", "qwen3vl_4b", "qwen3vl_2b"):
        if name not in candidates:
            candidates.append(name)
    qwen_tok = mm_tok
    for name in candidates:
        sub = getattr(mm_tok, name, None)
        if sub is not None:
            qwen_tok = sub
            break
    hf_tok = getattr(qwen_tok, "tokenizer", None)
    if hf_tok is not None:
        return hf_tok
    if callable(getattr(qwen_tok, "decode", None)):
        return qwen_tok
    return None


def verify_tokenizer_constants(tokenizer: Any | None = None) -> dict[str, int]:
    """Assert that `comfy/text_encoders/minimax.py` and the Qwen3-VL tokenizer
    agree on the special token IDs and label trailing-space tokenization."""
    import comfy.text_encoders.minimax as mm

    if mm.VISION_START != VISION_START:
        raise ValueError(
            f"comfy.text_encoders.minimax.VISION_START is {mm.VISION_START}, expected {VISION_START}"
        )
    if mm.VISION_END != VISION_END:
        raise ValueError(
            f"comfy.text_encoders.minimax.VISION_END is {mm.VISION_END}, expected {VISION_END}"
        )
    extra = getattr(mm, "MINIMAX_EXTRA_TOKENS", {})
    if extra.get("<d>") != DIALOGUE_START or extra.get("</d>") != DIALOGUE_END:
        raise ValueError(
            f"MINIMAX_EXTRA_TOKENS dialogue IDs are {extra.get('<d>')}/{extra.get('</d>')}, "
            f"expected {DIALOGUE_START}/{DIALOGUE_END}"
        )

    if tokenizer is None:
        tok = mm.MiniMaxH3Tokenizer()
        hf_tok = unwrap_hf_tokenizer(tok)
    else:
        hf_tok = unwrap_hf_tokenizer(tokenizer)
    if hf_tok is None:
        raise ValueError("tokenizer does not expose the MiniMax Qwen3-VL tokenizer")

    expected = {
        "<|vision_start|>": VISION_START,
        "<|vision_end|>": VISION_END,
        "<|image_pad|>": IMAGE_PAD,
        "<d>": DIALOGUE_START,
        "</d>": DIALOGUE_END,
    }
    observed: dict[str, int] = {}
    for token_str, expected_id in expected.items():
        ids = hf_tok.encode(token_str, add_special_tokens=False)
        if list(ids) != [expected_id]:
            raise ValueError(
                f"tokenizer encoded {token_str!r} to {ids!r}, expected [{expected_id}]"
            )
        observed[token_str] = expected_id

    for label_str in ("<Picture 1>: ", "<Video 1>: ", "<Audio 1>: "):
        label_ids = hf_tok.encode(label_str, add_special_tokens=False)
        if not label_ids or hf_tok.decode([label_ids[-1]]) != " ":
            raise ValueError(
                f"label {label_str!r} did not end in a lone space token: {label_ids!r}"
            )
    return observed


def _flat_int_list(seq: Sequence[int] | torch.Tensor, name: str) -> list[int]:
    if isinstance(seq, torch.Tensor):
        t = seq.detach().cpu()
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.ndim != 1:
            raise ValueError(f"{name} must be 1D (or [1, L]), got shape {tuple(seq.shape)}")
        return [int(x) for x in t.tolist()]
    return [int(x) for x in seq]


def expand_token_ids(
    entries: Sequence[Any],
    tags: Sequence[int] | torch.Tensor,
) -> list[int]:
    """Expand raw `MiniMaxH3Tokenizer` entries `(token_or_dict, weight, ...)` to
    length `L == len(tags)` by filling each `tags == 0` vision block with
    `[VISION_START, IMAGE_PAD, ..., IMAGE_PAD, VISION_END]`."""
    tag_list = _flat_int_list(tags, "tags")
    L = len(tag_list)
    raw = [e[0] if isinstance(e, (tuple, list)) else e for e in entries]

    vision_runs: list[tuple[int, int]] = []
    idx = 0
    while idx < L:
        if tag_list[idx] == 0:
            end = idx
            while end < L and tag_list[end] == 0:
                end += 1
            vision_runs.append((idx, end))
            idx = end
        else:
            idx += 1

    expanded: list[int] = []
    run_idx = 0
    i = 0
    while i < len(raw):
        item = raw[i]
        if (
            item == VISION_START
            and i + 2 < len(raw)
            and isinstance(raw[i + 1], dict)
            and raw[i + 2] == VISION_END
        ):
            if run_idx >= len(vision_runs):
                raise ValueError(
                    "entries contain more vision blocks than contiguous tag==0 runs in tags"
                )
            start, end = vision_runs[run_idx]
            run_len = end - start
            if run_len < 3:
                raise ValueError(f"vision block at {start}:{end} has length {run_len} < 3")
            expanded.append(VISION_START)
            expanded.extend([IMAGE_PAD] * (run_len - 2))
            expanded.append(VISION_END)
            run_idx += 1
            i += 3
        elif isinstance(item, dict):
            raise ValueError(f"vision dict at entry {i} is not flanked by VISION_START/VISION_END")
        else:
            expanded.append(int(item))
            i += 1

    if run_idx != len(vision_runs):
        raise ValueError(
            f"entries expanded {run_idx} vision block(s), but tags has {len(vision_runs)} tag==0 run(s)"
        )
    if len(expanded) != L:
        raise ValueError(
            f"expanded token_ids length ({len(expanded)}) does not match tags length ({L})"
        )
    return expanded


def spans(
    token_ids: Sequence[int] | torch.Tensor,
    tags: Sequence[int] | torch.Tensor,
    decode: Callable[[Sequence[int]], str],
) -> dict[str, torch.BoolTensor]:
    """Partition `0 .. L-1` into six mutually exclusive boolean masks:
    `vision`, `label`, `timestamp`, `dialogue`, `speaker`, and `scene`."""
    ids = _flat_int_list(token_ids, "token_ids")
    tag_list = _flat_int_list(tags, "tags")
    if len(ids) != len(tag_list):
        raise ValueError(
            f"token_ids length ({len(ids)}) does not match tags length ({len(tag_list)})"
        )
    L = len(ids)
    for idx, t in enumerate(tag_list):
        if t not in (0, 1):
            raise ValueError(f"tags[{idx}] is {t}; expected 0 (vision) or 1 (text)")

    out: dict[str, torch.BoolTensor] = {
        cat: torch.zeros(L, dtype=torch.bool) for cat in CATEGORIES
    }

    in_dialogue = False
    dialogue_open_idx = -1
    last_vision_idx = -1
    for idx in range(L):
        if tag_list[idx] == 0:
            if in_dialogue:
                raise ValueError(
                    f"vision row at index {idx} sits inside an unclosed <d> block opened at {dialogue_open_idx}"
                )
            out["vision"][idx] = True
            last_vision_idx = idx
            continue

        tid = ids[idx]
        if tid == DIALOGUE_START:
            if in_dialogue:
                raise ValueError(
                    f"nested <d> token at index {idx} inside <d> block opened at {dialogue_open_idx}"
                )
            in_dialogue = True
            dialogue_open_idx = idx
            out["dialogue"][idx] = True
        elif tid == DIALOGUE_END:
            if not in_dialogue:
                raise ValueError(f"unmatched </d> token at index {idx} with no preceding <d>")
            out["dialogue"][idx] = True
            in_dialogue = False
            dialogue_open_idx = -1
        elif in_dialogue:
            out["dialogue"][idx] = True

    if in_dialogue:
        raise ValueError(f"unclosed <d> token opened at index {dialogue_open_idx}")

    # Partition remaining tag==1 non-dialogue runs into label, timestamp, speaker, scene.
    # Reference labels (`<Picture i>: `, `<Video k>: `, `<Audio j>: `) and video block
    # timestamps (`<T.T seconds>`) exist only in the reference prefix placed by
    # `MiniMaxH3Tokenizer.tokenize_with_weights`, never inside the prompt body:
    # - Runs before a vision block (`end <= last_vision_idx` and `out["vision"][end]`):
    #   contain only reference prefix labels (each ending on a lone `" "` token) and
    #   a `<T.T seconds>` timestamp immediately before the vision block.
    # - The first run after all vision blocks (`idx == last_vision_idx + 1`, which is
    #   `idx == 0` when there are no vision blocks): may start with trailing `<Audio j>: `
    #   prefix labels at offset 0 (each ending on a lone `" "` token), followed by the
    #   prompt body where only `SPEAKER_RE` is matched against `scene` prose.
    idx = 0
    while idx < L:
        if out["vision"][idx] or out["dialogue"][idx]:
            idx += 1
            continue
        end = idx
        while end < L and not out["vision"][end] and not out["dialogue"][end]:
            end += 1

        pieces = [decode([ids[k]]) for k in range(idx, end)]
        seg_text = "".join(pieces)

        # Build token character spans in `seg_text` to verify lone-space token boundaries.
        char_ends_with_lone_space: set[int] = set()
        pos_scan = 0
        for piece in pieces:
            pos_scan += len(piece)
            if piece == " ":
                char_ends_with_lone_space.add(pos_scan)

        matches: list[tuple[str, int, int]] = []
        before_vision = end < L and bool(out["vision"][end].item())
        at_prefix_tail = idx == (last_vision_idx + 1)

        if before_vision:
            for m in LABEL_RE.finditer(seg_text):
                if m.end() in char_ends_with_lone_space:
                    matches.append(("label", m.start(), m.end()))
            for m in TIMESTAMP_RE.finditer(seg_text):
                if m.end() == len(seg_text):
                    matches.append(("timestamp", m.start(), m.end()))
        elif at_prefix_tail:
            # Only leading `<Audio j>: ` labels anchored contiguously from offset 0
            # that end on a lone space token belong to the reference prefix.
            cursor = 0
            while True:
                m = LABEL_RE.match(seg_text, cursor)
                if m is None or not m.group(0).startswith("<Audio ") or m.end() not in char_ends_with_lone_space:
                    break
                matches.append(("label", m.start(), m.end()))
                cursor = m.end()

        for m in SPEAKER_RE.finditer(seg_text):
            matches.append(("speaker", m.start(), m.end()))

        pos = 0
        for k, piece in zip(range(idx, end), pieces):
            c0, c1 = pos, pos + len(piece)
            pos = c1
            lstripped = piece.lstrip()
            rstripped = piece.rstrip()
            nc0 = c0 + (len(piece) - len(lstripped))
            nc1 = c0 + len(rstripped)
            assigned = "scene"
            for cat, m0, m1 in matches:
                if nc0 < nc1:
                    if max(nc0, m0) < min(nc1, m1):
                        assigned = cat
                        break
                else:
                    if c0 >= m0 and c1 <= m1:
                        assigned = cat
                        break
            out[assigned][k] = True

        idx = end

    cover = sum(m.to(torch.int64) for m in out.values())
    if not bool(torch.all(cover == 1)):
        bad = (cover != 1).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"span partition is not mutually exclusive at indices {bad}")

    # Attach 3-lever structural section submasks (`_h3_look_mask`,
    # `_h3_description_mask`, `_h3_soundscape_mask`, and `_h3_steerable_scene_mask`)
    # onto `out["scene"]` so `inference_mask` can select:
    #   - `look_only` (`subject_definitions:` + `summary:` + visual lines of
    #     `retention_analysis:`, excluding `ref_en.md` task/retention markers),
    #   - `scene_only` (`_h3_look_mask | _h3_description_mask`: all visual prose,
    #     excluding audio sections, `"N/A"`, headers, tags, and Part One alignment),
    #   - `description_only` (`integrated_multimodal_description:` / `detailed_description:`),
    #   - `soundscape_only` (`overall_soundscape:` + non-`"N/A"` `non_diegetic_music:`
    #     + `<Audio N>:` lines inside `retention_analysis:`),
    # while keeping `spans()`'s 6 mutually exclusive `CATEGORIES` unchanged.
    look_mask = torch.zeros(L, dtype=torch.bool)
    desc_mask = torch.zeros(L, dtype=torch.bool)
    sound_mask = torch.zeros(L, dtype=torch.bool)
    # One mask per section header found, keyed by the header's own name, for `FIELD_MASK_MODES`.
    # The three lever masks above are unions of these (`retention_analysis` split by line).
    section_masks: dict[str, torch.Tensor] = {}
    steerable_scene_mask = out["scene"].clone()
    has_sections = False
    tail_text = ""
    prompt_start = last_vision_idx + 1
    while prompt_start < L and bool(out["label"][prompt_start].item()):
        prompt_start += 1
    if prompt_start < L:
        tail_pieces: list[str] = []
        tail_char_spans: list[tuple[int, int, int, int, int]] = []
        pos_cursor = 0
        for k in range(prompt_start, L):
            if out["dialogue"][k]:
                piece_k = "<d>" if ids[k] == DIALOGUE_START else ("</d>" if ids[k] == DIALOGUE_END else decode([ids[k]]))
            else:
                piece_k = decode([ids[k]])
            c0 = pos_cursor
            c1 = c0 + len(piece_k)
            pos_cursor = c1
            lstripped = piece_k.lstrip()
            rstripped = piece_k.rstrip()
            nc0 = c0 + (len(piece_k) - len(lstripped))
            nc1 = c0 + len(rstripped)
            tail_pieces.append(piece_k)
            tail_char_spans.append((k, c0, c1, nc0, nc1))

        tail_text = "".join(tail_pieces)
        structural_tag_spans = [(m.start(), m.end()) for m in STRUCTURAL_TAG_RE.finditer(tail_text)]
        for k, _c0, _c1, nc0, nc1 in tail_char_spans:
            if not bool(out["scene"][k].item()):
                continue
            if nc0 < nc1 and any(max(nc0, t0) < min(nc1, t1) for t0, t1 in structural_tag_spans):
                steerable_scene_mask[k] = False

        sec_matches = list(SECTION_HEADER_RE.finditer(tail_text))
        if sec_matches:
            has_sections = True
            sec_intervals: list[tuple[str, int, int, list[tuple[int, int]]]] = []
            for s_idx, sm in enumerate(sec_matches):
                # Start strictly after the section header and colon (`sm.end()`) so header tokens are never steered.
                s_start = sm.end()
                s_end = sec_matches[s_idx + 1].start() if (s_idx + 1) < len(sec_matches) else len(tail_text)
                sec_name = sm.group(1)
                sec_body = tail_text[s_start:s_end].strip().rstrip(".").strip().lower()
                if sec_body in EMPTY_SECTION_BODIES:
                    continue
                audio_ret_spans: list[tuple[int, int]] = []
                if sec_name == "retention_analysis":
                    line_cursor = s_start
                    for raw_line in tail_text[s_start:s_end].splitlines(keepends=True):
                        line_end = line_cursor + len(raw_line)
                        if AUDIO_RETENTION_LINE_RE.match(raw_line):
                            audio_ret_spans.append((line_cursor, line_end))
                        line_cursor = line_end
                sec_intervals.append((sec_name, s_start, s_end, audio_ret_spans))
            for k, c0, _c1, nc0, nc1 in tail_char_spans:
                if not bool(steerable_scene_mask[k].item()):
                    continue
                probe_pos = nc0 if nc0 < nc1 else c0
                for sec_name, s_start, s_end, audio_ret_spans in sec_intervals:
                    if s_start <= probe_pos < s_end:
                        if sec_name not in section_masks:
                            section_masks[sec_name] = torch.zeros(L, dtype=torch.bool)
                        section_masks[sec_name][k] = True
                        if sec_name in LOOK_SECTIONS:
                            look_mask[k] = True
                        elif sec_name == "retention_analysis":
                            if any(a0 <= probe_pos < a1 for a0, a1 in audio_ret_spans):
                                sound_mask[k] = True
                            else:
                                look_mask[k] = True
                        elif sec_name in DESCRIPTION_SECTIONS:
                            desc_mask[k] = True
                        elif sec_name in SOUNDSCAPE_SECTIONS:
                            sound_mask[k] = True
                        break

    setattr(out["scene"], "_h3_look_mask", look_mask)
    setattr(out["scene"], "_h3_description_mask", desc_mask)
    setattr(out["scene"], "_h3_soundscape_mask", sound_mask)
    setattr(out["scene"], "_h3_section_masks", section_masks)
    setattr(out["scene"], "_h3_steerable_scene_mask", steerable_scene_mask)
    setattr(out["scene"], "_h3_has_sections", has_sections)
    setattr(out["scene"], "_h3_tail_text", tail_text)
    setattr(out["scene"], "_h3_token_ids", tuple(ids))
    return out


def apply_mask(
    span_masks: dict[str, torch.BoolTensor],
    categories: Sequence[str] | None = None,
) -> torch.BoolTensor:
    """Boolean mask of rows to steer at apply time.

    Defaults to v1's tag mask (`~vision`, i.e., every `tag == 1` row) when
    `categories is None`. When `categories` is provided, returns the union of
    the named categories."""
    if not isinstance(span_masks, dict) or "vision" not in span_masks:
        raise ValueError("span_masks is missing required 'vision' key")
    if categories is None:
        return ~span_masks["vision"]
    L = int(span_masks["vision"].numel())
    mask = torch.zeros(L, dtype=torch.bool, device=span_masks["vision"].device)
    for cat in categories:
        if cat not in span_masks:
            raise ValueError(f"unknown span category {cat!r}; expected one of {CATEGORIES}")
        mask = mask | span_masks[cat]
    return mask


def inference_mask(
    span_masks: dict[str, torch.BoolTensor],
    mode: str = "non_vision",
) -> torch.BoolTensor:
    """Boolean mask of conditioning rows to steer at inference time.

    Modes (`INFERENCE_MASK_MODES`):
    - `'non_vision'`: `~span_masks['vision']` (every `tag == 1` text row).
    - `'look_only'`: Visual Look sections (`subject_definitions:`, `summary:`, and
      visual lines of `retention_analysis:`, excluding `[reference generation]` /
      `ref_en.md` task brackets, `<Subject N> (...): fully_preserved -` headers,
      `<Audio N>:` lines, and `"N/A"`). Falls back to `'description_only'` on
      3-field `t2va` / `fl2va` or sectionless prompts.
    - `'scene_only'`: all steerable visual prose (`look_only | description_only`)
      on sectioned H3 video prompts (`t2va`, `fl2va`, `ref2va`), excluding audio
      sections (`overall_soundscape:`, `non_diegetic_music:`, `<Audio N>:` retention
      lines), `"N/A"`, vision, reference labels, timestamps, speaker tags `(S1)`,
      `<d>...</d>` dialogue, section headers, `[Shot N]`, `<Subject N>`/`<Picture N>`
      tags, and Part One keyframe alignment lines. On sectionless prompts (such as
      `P-H3-Image`), steers all non-tag `scene` tokens (`_h3_steerable_scene_mask`).
    - `'description_only'`: non-dialogue `scene` prose tokens inside the Events /
      Actions visual description section (`integrated_multimodal_description:` on
      `fl2va`/`t2va` or `detailed_description:` on `ref2va`), excluding section
      headers, `[Shot N]`, and `<Subject N>`/`<Picture N>` tags (falls back to
      `scene_only` if the prompt has no description section header).
    - `'soundscape_only'`: `scene` prose tokens inside `overall_soundscape:` plus
      non-`"N/A"` `non_diegetic_music:` (and `<Audio N>:` retention lines),
      excluding section headers and `"N/A"` / `"None"` bodies (falls back to
      `scene_only` on sectionless prompts).
    - `'after_dialogue'`: `span_masks['scene']` strictly after the last
      `dialogue` token (or all `span_masks['scene']` rows when the prompt has
      no `<d>...</d>` span or no trailing scene tokens after `</d>`).
    - `'all_rows'`: all `L` rows (`vision` and `tag == 1`).
    - a field's own name (`FIELD_MASK_MODES`: `'subject_definitions'`, `'summary'`,
      `'retention_analysis'`, `'overall_soundscape'`, `'non_diegetic_music'`): the steerable prose
      tokens inside that one section, under the same exclusions as the lever masks (headers, tags,
      task and retention markers, dialogue). `'retention_analysis'` takes every line of the section,
      its `<Audio N>:` lines too. No rows when the prompt has no such section or its body is `N/A`;
      there is no fallback to another field.
    - a selection of `PROMPT_FIELDS` joined by `+` (`fields_mask`): the union of those fields' rows,
      each as above. `description` is the description section the prompt has, with no fallback."""
    if not isinstance(span_masks, dict) or "vision" not in span_masks or "scene" not in span_masks:
        raise ValueError("span_masks must contain 'vision' and 'scene' keys")
    selection = split_fields_mask(mode)
    if mode not in INFERENCE_MASK_MODES and selection is None:
        raise ValueError(
            f"unknown inference mask mode {mode!r}; expected one of {INFERENCE_MASK_MODES} "
            f"or prompt fields joined by {FIELDS_JOIN!r} ({PROMPT_FIELDS})"
        )
    if selection is not None:
        scene_sel = span_masks["scene"]
        by_section = getattr(scene_sel, "_h3_section_masks", None)
        rows = torch.zeros_like(scene_sel, dtype=torch.bool)
        for field in selection:
            headers = ("detailed_description", "integrated_multimodal_description") if field == "description" else (field,)
            for header in headers:
                sub = by_section.get(header) if isinstance(by_section, dict) else None
                if isinstance(sub, torch.Tensor) and sub.shape == scene_sel.shape:
                    rows = rows | sub.to(device=scene_sel.device)
        return (rows & scene_sel).clone()
    if mode == "non_vision":
        return (~span_masks["vision"]).clone()
    if mode == "all_rows":
        return torch.ones_like(span_masks["vision"], dtype=torch.bool)
    scene_t = span_masks["scene"]
    has_sections = bool(getattr(scene_t, "_h3_has_sections", False))
    if mode == "look_only":
        look_sub = getattr(scene_t, "_h3_look_mask", None)
        if isinstance(look_sub, torch.Tensor) and look_sub.shape == scene_t.shape and bool(look_sub.any()):
            return (look_sub.to(device=scene_t.device) & scene_t).clone()
        return inference_mask(span_masks, "description_only")
    if mode == "scene_only":
        look_sub = getattr(scene_t, "_h3_look_mask", None)
        desc_sub = getattr(scene_t, "_h3_description_mask", None)
        combined_vis: torch.Tensor | None = None
        if isinstance(look_sub, torch.Tensor) and look_sub.shape == scene_t.shape:
            combined_vis = look_sub
        if isinstance(desc_sub, torch.Tensor) and desc_sub.shape == scene_t.shape:
            combined_vis = desc_sub if combined_vis is None else (combined_vis | desc_sub)
        if combined_vis is not None and bool(combined_vis.any()):
            return (combined_vis.to(device=scene_t.device) & scene_t).clone()
        if has_sections and combined_vis is not None:
            return torch.zeros_like(scene_t, dtype=torch.bool)
        steerable = getattr(scene_t, "_h3_steerable_scene_mask", None)
        if isinstance(steerable, torch.Tensor) and steerable.shape == scene_t.shape and bool(steerable.any()):
            return (steerable.to(device=scene_t.device) & scene_t).clone()
        return scene_t.clone()
    if mode == "description_only":
        sub = getattr(scene_t, "_h3_description_mask", None)
        if isinstance(sub, torch.Tensor) and sub.shape == scene_t.shape and bool(sub.any()):
            return (sub.to(device=scene_t.device) & scene_t).clone()
        if has_sections and isinstance(sub, torch.Tensor) and sub.shape == scene_t.shape:
            return torch.zeros_like(scene_t, dtype=torch.bool)
        return inference_mask(span_masks, "scene_only")
    if mode == "soundscape_only":
        sub = getattr(scene_t, "_h3_soundscape_mask", None)
        if isinstance(sub, torch.Tensor) and sub.shape == scene_t.shape and bool(sub.any()):
            return (sub.to(device=scene_t.device) & scene_t).clone()
        if has_sections and isinstance(sub, torch.Tensor) and sub.shape == scene_t.shape:
            return torch.zeros_like(scene_t, dtype=torch.bool)
        return inference_mask(span_masks, "scene_only")
    if mode == "after_dialogue":
        mask = span_masks["scene"].clone()
        dlg = span_masks.get("dialogue")
        if dlg is not None and bool(dlg.any()):
            last_dlg = int(dlg.nonzero(as_tuple=False).flatten()[-1].item())
            trailing = mask.clone()
            trailing[: last_dlg + 1] = False
            if bool(trailing.any()):
                return trailing
        return mask
    raise ValueError(f"unknown inference mask mode {mode!r}; expected one of {INFERENCE_MASK_MODES}")


_CACHED_CPU_TOKENIZER: Any | None = None


def _get_cpu_tokenizer() -> Any:
    global _CACHED_CPU_TOKENIZER
    if _CACHED_CPU_TOKENIZER is None:
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

        _CACHED_CPU_TOKENIZER = MiniMaxH3Tokenizer()
    return _CACHED_CPU_TOKENIZER


def _extract_batch_entries(tokens: Any) -> list[Any]:
    if isinstance(tokens, dict):
        if not tokens:
            return []
        batches = tokens.get("qwen3vl_32b")
        if batches is None:
            batches = next(iter(tokens.values()))
    else:
        batches = tokens
    if not isinstance(batches, (list, tuple)) or not batches:
        return []
    first = batches[0]
    if isinstance(first, (list, tuple)) and (
        not first or isinstance(first[0], (list, tuple, dict))
    ):
        return list(first)
    return list(batches)


def attach_span_masks(
    conditioning: list[Any],
    clip: Any,
    tokens: Any,
) -> list[Any]:
    """Attach `h3_span_masks` onto each conditioning entry's metadata dict,
    `model_conds['h3_span_masks'] = CONDConstant(span_masks)` (so ComfyUI core's
    `convert_cond` -> `get_area_and_mult` -> `cond_cat` passes `h3_span_masks`
    as a per-prompt keyword argument to the diffusion model wrapper), and
    `_h3_span_masks` onto its `minimax_token_tags` tensor in-place without
    modifying its `{0, 1}` integer elements."""
    if not isinstance(conditioning, list) or not conditioning:
        return conditioning
    hf_tok = unwrap_hf_tokenizer(clip)
    if hf_tok is None:
        return conditioning
    entries = _extract_batch_entries(tokens)
    if not entries:
        return conditioning

    import comfy.conds

    def decode_fn(ids: Sequence[int]) -> str:
        return hf_tok.decode(list(ids), skip_special_tokens=False)

    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], dict):
            continue
        tags_raw = item[1].get("minimax_token_tags")
        if not isinstance(tags_raw, torch.Tensor):
            continue
        tags_1d = tags_raw.detach().to(device="cpu", dtype=torch.long).flatten()
        token_ids = expand_token_ids(entries, tags_1d)
        span_masks = spans(token_ids, tags_1d.tolist(), decode_fn)
        item[1]["h3_span_masks"] = span_masks
        mc = dict(item[1].get("model_conds") or {})
        mc["h3_span_masks"] = comfy.conds.CONDConstant(span_masks)
        item[1]["model_conds"] = mc
        setattr(tags_raw, "_h3_span_masks", span_masks)
    return conditioning


def span_masks_from_prompt(
    prompt: str,
    total_length: int,
    *,
    tags: Sequence[int] | torch.Tensor | None = None,
    tokenizer: Any | None = None,
) -> dict[str, torch.BoolTensor]:
    """Build `span_masks` of length `total_length` by tokenizing `prompt` and
    aligning its `L_prompt` tokens to the suffix `[total_length - L_prompt : total_length]`.

    In `comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer.tokenize_with_weights`,
    the prompt text is tokenized in an isolated `add_text(text)` call after all
    reference/keyframe prefix blocks with no cross-boundary BPE merging, so the
    prompt tokens always occupy the exact suffix `[total_length - L_prompt : total_length]`."""
    clean_text = str(prompt).strip()
    if not clean_text:
        raise ValueError("prompt must be a non-empty string to resolve span masks")
    if total_length < 1:
        raise ValueError(f"total_length must be >= 1, got {total_length}")

    tok = tokenizer if tokenizer is not None else _get_cpu_tokenizer()
    hf_tok = unwrap_hf_tokenizer(tok)
    if hf_tok is None:
        raise ValueError("tokenizer does not expose the MiniMax Qwen3-VL tokenizer")

    raw_tokens = tok.tokenize_with_weights(clean_text) if hasattr(tok, "tokenize_with_weights") else tok.tokenize(clean_text)
    entries = _extract_batch_entries(raw_tokens)
    prompt_tags = [1] * len(entries)
    prompt_ids = expand_token_ids(entries, prompt_tags)
    L_prompt = len(prompt_ids)
    if L_prompt > total_length:
        raise ValueError(
            f"prompt token length ({L_prompt}) exceeds conditioning sequence length ({total_length})"
        )

    def decode_fn(ids: Sequence[int]) -> str:
        return hf_tok.decode(list(ids), skip_special_tokens=False)

    prompt_spans = spans(prompt_ids, prompt_tags, decode_fn)
    if L_prompt == total_length:
        return prompt_spans

    prefix_len = total_length - L_prompt
    if tags is not None:
        tags_list = _flat_int_list(tags, "tags")
        if len(tags_list) != total_length:
            raise ValueError(f"tags length ({len(tags_list)}) != total_length ({total_length})")
        prefix_vis = torch.tensor([t == 0 for t in tags_list[:prefix_len]], dtype=torch.bool)
    else:
        prefix_vis = torch.ones(prefix_len, dtype=torch.bool)
    prefix_lbl = ~prefix_vis
    prefix_false = torch.zeros(prefix_len, dtype=torch.bool)

    scene_cat = torch.cat([prefix_false, prompt_spans["scene"]], dim=0)
    look_sub = getattr(prompt_spans["scene"], "_h3_look_mask", None)
    if isinstance(look_sub, torch.Tensor):
        setattr(scene_cat, "_h3_look_mask", torch.cat([prefix_false, look_sub], dim=0))
    desc_sub = getattr(prompt_spans["scene"], "_h3_description_mask", None)
    if isinstance(desc_sub, torch.Tensor):
        setattr(scene_cat, "_h3_description_mask", torch.cat([prefix_false, desc_sub], dim=0))
    sound_sub = getattr(prompt_spans["scene"], "_h3_soundscape_mask", None)
    if isinstance(sound_sub, torch.Tensor):
        setattr(scene_cat, "_h3_soundscape_mask", torch.cat([prefix_false, sound_sub], dim=0))
    sections_sub = getattr(prompt_spans["scene"], "_h3_section_masks", None)
    if isinstance(sections_sub, dict):
        setattr(
            scene_cat,
            "_h3_section_masks",
            {name: torch.cat([prefix_false, sub], dim=0) for name, sub in sections_sub.items()},
        )
    steerable_sub = getattr(prompt_spans["scene"], "_h3_steerable_scene_mask", None)
    if isinstance(steerable_sub, torch.Tensor):
        setattr(scene_cat, "_h3_steerable_scene_mask", torch.cat([prefix_false, steerable_sub], dim=0))
    has_sec_sub = getattr(prompt_spans["scene"], "_h3_has_sections", None)
    if isinstance(has_sec_sub, bool):
        setattr(scene_cat, "_h3_has_sections", has_sec_sub)
    tail_sub = getattr(prompt_spans["scene"], "_h3_tail_text", None)
    if isinstance(tail_sub, str):
        setattr(scene_cat, "_h3_tail_text", tail_sub)
    tok_sub = getattr(prompt_spans["scene"], "_h3_token_ids", None)
    if isinstance(tok_sub, tuple):
        setattr(scene_cat, "_h3_token_ids", tuple([0] * prefix_len) + tok_sub)

    return {
        "vision": torch.cat([prefix_vis, prompt_spans["vision"]], dim=0),
        "label": torch.cat([prefix_lbl, prompt_spans["label"]], dim=0),
        "timestamp": torch.cat([prefix_false, prompt_spans["timestamp"]], dim=0),
        "dialogue": torch.cat([prefix_false, prompt_spans["dialogue"]], dim=0),
        "speaker": torch.cat([prefix_false, prompt_spans["speaker"]], dim=0),
        "scene": scene_cat,
    }


def first_differing_index(
    token_ids_a: Sequence[int] | torch.Tensor,
    token_ids_b: Sequence[int] | torch.Tensor,
) -> int:
    """Index of the first token where `token_ids_a` and `token_ids_b` differ.

    Raises `ValueError` if the two sequences are identical."""
    a = _flat_int_list(token_ids_a, "token_ids_a")
    b = _flat_int_list(token_ids_b, "token_ids_b")
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    raise ValueError("token_ids_a and token_ids_b are identical; no differing token exists")


_DIFF_MASK_CACHE: dict[tuple[Any, ...], torch.BoolTensor] = {}


def _raw_differing_token_bounds(
    a: list[int],
    b: list[int],
) -> tuple[int, int, int]:
    first_diff = first_differing_index(a, b)
    max_suffix = min(len(a) - first_diff, len(b) - first_diff)
    shared_suffix = 0
    while (
        shared_suffix < max_suffix
        and a[len(a) - 1 - shared_suffix] == b[len(b) - 1 - shared_suffix]
    ):
        shared_suffix += 1
    end_a = max(first_diff + 1, len(a) - shared_suffix)
    end_b = max(first_diff + 1, len(b) - shared_suffix)
    return first_diff, end_a, end_b


def differing_token_mask(
    token_ids_a: Sequence[int] | torch.Tensor,
    token_ids_b: Sequence[int] | torch.Tensor,
    *,
    min_shared_run: int = 8,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Return `(mask_a, mask_b)` marking differing token positions between `a` and `b`.

    Uses `difflib.SequenceMatcher(None, a, b, autojunk=False)` so that:
      - On single-span diffs (where any interior equal block has `< min_shared_run`
        tokens), `(mask_a, mask_b)` are `True` on `[first_diff:end_a]` and
        `[first_diff:end_b]`, bit-for-bit identical to `differing_token_bounds`.
      - On multi-span or cross-section diffs (e.g. `description + soundscape` or
        `opening_framing + camera_motion`), interior unchanged blocks of length
        `>= min_shared_run` are set to `False` on both sides while accommodating
        `Delta L != 0` token-length shifts between `a` and `b`.

    Raises `ValueError` if `token_ids_a` and `token_ids_b` are identical."""
    a = _flat_int_list(token_ids_a, "token_ids_a")
    b = _flat_int_list(token_ids_b, "token_ids_b")
    first_diff, end_a, end_b = _raw_differing_token_bounds(a, b)
    mask_a = torch.zeros(len(a), dtype=torch.bool)
    mask_b = torch.zeros(len(b), dtype=torch.bool)
    mask_a[first_diff:end_a] = True
    mask_b[first_diff:end_b] = True
    if min_shared_run > 0:
        sub_a = a[first_diff:end_a]
        sub_b = b[first_diff:end_b]
        matcher = difflib.SequenceMatcher(None, sub_a, sub_b, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal" and (i2 - i1) >= min_shared_run:
                mask_a[first_diff + i1 : first_diff + i2] = False
                mask_b[first_diff + j1 : first_diff + j2] = False
        if not bool(mask_a.any()):
            mask_a[first_diff:end_a] = True
        if not bool(mask_b.any()):
            mask_b[first_diff:end_b] = True
    return mask_a, mask_b


def differing_token_bounds(
    token_ids_a: Sequence[int] | torch.Tensor,
    token_ids_b: Sequence[int] | torch.Tensor,
) -> tuple[int, int, int]:
    """Return `(first_diff, end_a, end_b)` bounding the differing token span
    `[first_diff:end_a]` in `token_ids_a` and `[first_diff:end_b]` in `token_ids_b`.

    Also caches multi-span `differing_token_mask` when interior shared runs of
    `>= 4` tokens exist so `pooling_mask(..., mode='diff_span')` automatically
    excludes unchanged interior bridge tokens.

    Raises `ValueError` if `token_ids_a` and `token_ids_b` are identical."""
    a = _flat_int_list(token_ids_a, "token_ids_a")
    b = _flat_int_list(token_ids_b, "token_ids_b")
    first_diff, end_a, end_b = _raw_differing_token_bounds(a, b)
    mask_a, mask_b = differing_token_mask(a, b, min_shared_run=8)
    ta = tuple(a)
    tb = tuple(b)
    if not bool(mask_a[first_diff:end_a].all()) or not bool(mask_b[first_diff:end_b].all()):
        while len(_DIFF_MASK_CACHE) >= 512:
            _DIFF_MASK_CACHE.pop(next(iter(_DIFF_MASK_CACHE)))
        _DIFF_MASK_CACHE[(ta, tb, first_diff, end_a)] = mask_a
        _DIFF_MASK_CACHE[(tb, ta, first_diff, end_b)] = mask_b
        _DIFF_MASK_CACHE[(ta, first_diff, end_a)] = mask_a
        _DIFF_MASK_CACHE[(tb, first_diff, end_b)] = mask_b
    else:
        _DIFF_MASK_CACHE.pop((ta, tb, first_diff, end_a), None)
        _DIFF_MASK_CACHE.pop((tb, ta, first_diff, end_b), None)
        _DIFF_MASK_CACHE.pop((ta, first_diff, end_a), None)
        _DIFF_MASK_CACHE.pop((tb, first_diff, end_b), None)
    return first_diff, end_a, end_b


def pooling_mask(
    span_masks: dict[str, torch.BoolTensor],
    mode: str = "scene_text",
    *,
    first_diff: int | None = None,
    end_diff: int | None = None,
    diff_mask: torch.BoolTensor | None = None,
    other_ids: Sequence[int] | torch.Tensor | None = None,
) -> torch.BoolTensor:
    """Boolean mask of rows to mean-pool when constructing a pair endpoint (`E2.6`).

    - `'scene_text'` (or `'scene'`): `span_masks['scene']` (tag-1 rows minus
      reference labels, timestamps, speaker tags, and dialogue).
    - `'all_tag1'`: `~span_masks['vision']` (every tag-1 row).
    - `'from_first_diff'`: `~span_masks['vision']` at indices `>= first_diff`.
    - `'diff_span'`: `span_masks['scene']` (falling back to `~span_masks['vision']`
      when the diff sits entirely inside `<d>...</d>` dialogue/speaker tokens)
      restricted to `[first_diff:end_diff]` (and excluding any interior shared
      token runs of length `>= 8` via `differing_token_mask`)."""
    if not isinstance(span_masks, dict) or "vision" not in span_masks or "scene" not in span_masks:
        raise ValueError("span_masks must contain 'vision' and 'scene' keys")
    if mode in ("scene_text", "scene"):
        mask = span_masks["scene"].clone()
    elif mode == "all_tag1":
        mask = (~span_masks["vision"]).clone()
    elif mode == "from_first_diff":
        if first_diff is None or first_diff < 0:
            raise ValueError("pooling mode 'from_first_diff' requires a non-negative first_diff index")
        mask = (~span_masks["vision"]).clone()
        mask[:first_diff] = False
    elif mode == "diff_span":
        if diff_mask is None and other_ids is not None:
            self_ids = getattr(span_masks["scene"], "_h3_token_ids", None)
            if self_ids is not None:
                diff_mask, _ = differing_token_mask(self_ids, other_ids, min_shared_run=8)
        if diff_mask is None and first_diff is not None and end_diff is not None:
            self_ids = getattr(span_masks["scene"], "_h3_token_ids", None)
            if isinstance(self_ids, tuple):
                cached = _DIFF_MASK_CACHE.get((self_ids, int(first_diff), int(end_diff)))
                if cached is not None and cached.numel() == span_masks["scene"].numel():
                    diff_mask = cached

        if diff_mask is not None:
            dm = diff_mask.to(device=span_masks["scene"].device, dtype=torch.bool)
            mask = span_masks["scene"] & dm
            if not bool(mask.any()):
                mask = (~span_masks["vision"]) & dm
        else:
            if first_diff is None or first_diff < 0:
                raise ValueError("pooling mode 'diff_span' requires a non-negative first_diff index")
            if end_diff is not None and end_diff <= first_diff:
                raise ValueError("pooling mode 'diff_span' requires end_diff > first_diff")
            mask = span_masks["scene"].clone()
            mask[:first_diff] = False
            if end_diff is not None:
                mask[end_diff:] = False
            if not bool(mask.any()):
                mask = (~span_masks["vision"]).clone()
                mask[:first_diff] = False
                if end_diff is not None:
                    mask[end_diff:] = False
    else:
        raise ValueError(f"unknown pooling mode {mode!r}; expected one of {POOLING_MODES}")

    if not bool(mask.any()):
        raise ValueError(f"pooling mode {mode!r} selected 0 rows")
    return mask
