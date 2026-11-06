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

INFERENCE_MASK_MODES: tuple[str, ...] = (
    "non_vision",
    "scene_only",
    "after_dialogue",
    "all_rows",
)

LABEL_RE = re.compile(r"<(?:Picture|Video|Audio) \d+>: ")
AUDIO_LABEL_PREFIX_RE = re.compile(r"^(?:<Audio \d+>: )+")
TIMESTAMP_RE = re.compile(r"<\d+(?:\.\d+)? seconds>")
SPEAKER_RE = re.compile(r"\(S\d+(?:,\s*S\d+)*\)")


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
        hf_tok = tok.qwen3vl_32b.tokenizer
    elif hasattr(tokenizer, "qwen3vl_32b"):
        hf_tok = tokenizer.qwen3vl_32b.tokenizer
    elif hasattr(tokenizer, "tokenizer"):
        hf_tok = tokenizer.tokenizer
    else:
        hf_tok = tokenizer

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
    - `'scene_only'`: `span_masks['scene']` (leaves vision, reference labels,
      timestamps, speaker tags `(S1)`, and `<d>...</d>` dialogue rows untouched).
    - `'after_dialogue'`: `span_masks['scene']` strictly after the last
      `dialogue` token (or all `span_masks['scene']` rows when the prompt has
      no `<d>...</d>` span or no trailing scene tokens after `</d>`).
    - `'all_rows'`: all `L` rows (`vision` and `tag == 1`)."""
    if not isinstance(span_masks, dict) or "vision" not in span_masks or "scene" not in span_masks:
        raise ValueError("span_masks must contain 'vision' and 'scene' keys")
    if mode == "non_vision":
        return (~span_masks["vision"]).clone()
    if mode == "all_rows":
        return torch.ones_like(span_masks["vision"], dtype=torch.bool)
    if mode == "scene_only":
        return span_masks["scene"].clone()
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
    batches = tokens.get("qwen3vl_32b", tokens) if isinstance(tokens, dict) else tokens
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
    """Attach `_h3_span_masks` onto each conditioning entry's `minimax_token_tags`
    tensor in-place without modifying its `{0, 1}` integer elements."""
    if not isinstance(conditioning, list) or not conditioning:
        return conditioning
    tok_root = getattr(clip, "tokenizer", None)
    mm_tok = getattr(tok_root, "minimax", tok_root)
    qwen_tok = getattr(mm_tok, "qwen3vl_32b", mm_tok)
    hf_tok = getattr(qwen_tok, "tokenizer", None)
    if hf_tok is None:
        return conditioning
    entries = _extract_batch_entries(tokens)
    if not entries:
        return conditioning

    def decode_fn(ids: Sequence[int]) -> str:
        return hf_tok.decode(list(ids), skip_special_tokens=False)

    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], dict):
            continue
        tags_raw = item[1].get("minimax_token_tags")
        if not isinstance(tags_raw, torch.Tensor):
            continue
        try:
            tags_1d = tags_raw.detach().to(device="cpu", dtype=torch.long).flatten()
            token_ids = expand_token_ids(entries, tags_1d)
            span_masks = spans(token_ids, tags_1d.tolist(), decode_fn)
            setattr(tags_raw, "_h3_span_masks", span_masks)
        except Exception:
            continue
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
    mm_tok = getattr(tok, "minimax", tok)
    qwen_tok = getattr(mm_tok, "qwen3vl_32b", mm_tok)
    hf_tok = getattr(qwen_tok, "tokenizer", None)
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

    return {
        "vision": torch.cat([prefix_vis, prompt_spans["vision"]], dim=0),
        "label": torch.cat([prefix_lbl, prompt_spans["label"]], dim=0),
        "timestamp": torch.cat([prefix_false, prompt_spans["timestamp"]], dim=0),
        "dialogue": torch.cat([prefix_false, prompt_spans["dialogue"]], dim=0),
        "speaker": torch.cat([prefix_false, prompt_spans["speaker"]], dim=0),
        "scene": torch.cat([prefix_false, prompt_spans["scene"]], dim=0),
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


def differing_token_bounds(
    token_ids_a: Sequence[int] | torch.Tensor,
    token_ids_b: Sequence[int] | torch.Tensor,
) -> tuple[int, int, int]:
    """Return `(first_diff, end_a, end_b)` bounding the differing token span
    `[first_diff:end_a]` in `token_ids_a` and `[first_diff:end_b]` in `token_ids_b`.

    Raises `ValueError` if `token_ids_a` and `token_ids_b` are identical."""
    a = _flat_int_list(token_ids_a, "token_ids_a")
    b = _flat_int_list(token_ids_b, "token_ids_b")
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


def pooling_mask(
    span_masks: dict[str, torch.BoolTensor],
    mode: str = "scene_text",
    *,
    first_diff: int | None = None,
    end_diff: int | None = None,
) -> torch.BoolTensor:
    """Boolean mask of rows to mean-pool when constructing a pair endpoint (`E2.6`).

    - `'scene_text'` (or `'scene'`): `span_masks['scene']` (tag-1 rows minus
      reference labels, timestamps, speaker tags, and dialogue).
    - `'all_tag1'`: `~span_masks['vision']` (every tag-1 row).
    - `'from_first_diff'`: `~span_masks['vision']` at indices `>= first_diff`.
    - `'diff_span'`: `span_masks['scene']` (falling back to `~span_masks['vision']`
      when the diff sits entirely inside `<d>...</d>` dialogue/speaker tokens)
      restricted to `[first_diff:end_diff]` (stripping both shared prefix and
      shared suffix tokens)."""
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
