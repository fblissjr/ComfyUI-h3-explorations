#!/usr/bin/env python3
"""
Build a hybrid 250x2048 calibration dataset (safetensors) for Qwen3-VL-32B-Instruct
that combines:
  - 200 rows (80%) of ExLlamaV3's standard diverse calibration corpus
    (Wikipedia, C4, code, random full-vocab tokens, multilingual, technical, TinyStories)
  - 50 rows (20%) of H3 conditioning and control-vector data, balanced across:
      * 25 rows of raw H3 scene prompts (`prompt_bank/*.txt`, deterministically shuffled
        across fl2va, i2va, l2va, ref2va, t2va)
      * 25 rows of control-vector contrastive pairs (`prompt_bank/control_vectors/*.toml`,
        both raw pole prompts and chat-templated self-report / labelling prompts across
        all 12 control-vector concepts)

IMPORTANT — Why MiniMax's 7 extra tokens (151669..151675: <d>, </d>, <|cutoff|>,
<|lyrics_start|>, <|lyrics_end|>, <|caption_start|>, <|caption_end|>) are NOT enabled
by default during Qwen calibration or chat/VLM inference:
  - The released H3 text_encoder checkpoint is 100% byte-identical to stock
    Qwen3-VL-32B-Instruct. MiniMax added those 7 strings to `tokenizer_config.json`'s
    `additional_special_tokens` without training Qwen on them.
  - In `embed_tokens.weight` ([151936, 5120]), rows 151669..151675 sit in Qwen's
    untrained padding tail (random init, norm ~0.50, min/max +-0.0236).
  - In `lm_head.weight` ([151936, 5120]), rows 151669..151675 (and 151676..151935)
    are untrained dead rows that are all identical to within 1 bf16 ULP
    (norm = 0.317757, pairwise max diff = 3.05e-5).
  - Consequently, Qwen itself (layers 0..63 + lm_head) neither understands IDs
    151669..151675 as XML/marker tags when reading chat prompts nor can ever emit
    them from `lm_head` when generating text. By default, we encode H3 rows with
    `encode_special_tokens=True, add_bos=False` on stock Qwen's trained vocabulary
    (151643..151668) so `<|im_start|>` (151644) and `<|im_end|>` (151645) are real
    special tokens while `<d>`, `</d>`, `<|cutoff|>` are tokenized as trained BPE
    subwords (`['<', 'd', '>']`).
  - For offline scripts that strictly emulate ComfyUI's `MiniMaxH3Tokenizer` on
    `layers.0..49`, `register_minimax_tokens(tokenizer)` is provided as an opt-in
    in-memory helper.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import tomllib
import torch
from safetensors.torch import save_file
from tokenizers import AddedToken

_CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
if not (_CANDIDATE_ROOT / "exllamav3").is_dir():
    _CANDIDATE_ROOT = Path(os.environ.get("EXLLAMAV3_ROOT", Path.home() / "workspace" / "exllamav3"))
REPO_ROOT = _CANDIDATE_ROOT
sys.path.insert(0, str(REPO_ROOT))

from exllamav3 import Config, Tokenizer
from exllamav3.conversion.calibration_data import get_default_calibration

MINIMAX_EXTRA_TOKENS: list[tuple[str, int]] = [
    ("<d>", 151669),
    ("</d>", 151670),
    ("<|cutoff|>", 151671),
    ("<|lyrics_start|>", 151672),
    ("<|lyrics_end|>", 151673),
    ("<|caption_start|>", 151674),
    ("<|caption_end|>", 151675),
]


def register_minimax_tokens(tokenizer: Tokenizer) -> None:
    """
    Register the 7 MiniMax H3 extra tokens (151669..151675) on an in-memory ExLlamaV3
    Tokenizer instance. In the upstream checkpoint, these 7 tokens appear only in
    `tokenizer_config.json`'s `additional_special_tokens` and are missing from
    `tokenizer.json` (`added_tokens`) and `added_tokens_decoder`.

    Adding them to the underlying HF `Tokenizer` with `special=False, normalized=False`
    (matching `<think>` 151667 and `</think>` 151668) AND populating `extended_piece_to_id` /
    `missing_special_piece_to_id` raises `raw_vocab_size` and `actual_vocab_size` from
    151669 to 151676 so `Tokenizer.encode()` (with either `encode_special_tokens=True` or
    `False`) and `Tokenizer.decode()` preserve them instead of splitting them into BPE
    pieces or filtering them out at `t < raw_vocab_size`.
    """
    to_add = [
        AddedToken(
            content,
            lstrip=False,
            rstrip=False,
            single_word=False,
            normalized=False,
            special=False,
        )
        for content, tid in MINIMAX_EXTRA_TOKENS
        if tokenizer.tokenizer.token_to_id(content) != tid
    ]
    if to_add:
        tokenizer.tokenizer.add_tokens(to_add)

    for content, tid in MINIMAX_EXTRA_TOKENS:
        actual_id = tokenizer.tokenizer.token_to_id(content)
        if actual_id != tid:
            raise RuntimeError(f"Expected token {content!r} to have ID {tid}, got {actual_id}")
        tokenizer.extended_piece_to_id[content] = tid
        tokenizer.extended_id_to_piece[tid] = content
        tokenizer.missing_special_piece_to_id[content] = tid

    tokenizer.missing_special_delimiters = None
    for cached_key in (
        "raw_vocab_size",
        "_get_id_to_ord_list",
        "_get_fixed_vocab",
        "_get_id_to_piece_list_spc",
        "_get_id_to_piece_list_nonspc",
        "_get_piece_to_id_dict",
    ):
        tokenizer.__dict__.pop(cached_key, None)
    tokenizer.actual_vocab_size = 1 + max(
        list(tokenizer.extended_id_to_piece.keys())
        + list(tokenizer.unspecial_id_to_piece.keys())
        + [tokenizer.raw_vocab_size - 1]
    )
    tokenizer.get_id_to_ord_list()
    tokenizer.get_id_to_piece_list(True)
    tokenizer.get_id_to_piece_list(False)
    tokenizer.get_piece_to_id_dict()


# Alias for cross-script compatibility
register_minimax_extra_tokens = register_minimax_tokens


def patch_model_dir_tokenizer(model_dir: Path) -> None:
    """
    Patch `tokenizer.json` and `tokenizer_config.json` in a compiled ExLlamaV3 model
    directory so that IDs 151669..151675 are recognized natively by `exllamav3.Tokenizer`.
    """
    tok_json_path = model_dir / "tokenizer.json"
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if not tok_json_path.is_file() or not tok_cfg_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json or tokenizer_config.json in {model_dir}")

    tok_json = json.loads(tok_json_path.read_text(encoding="utf-8"))
    added_list = tok_json.setdefault("added_tokens", [])
    existing_ids = {int(entry["id"]) for entry in added_list if "id" in entry}
    for content, tid in MINIMAX_EXTRA_TOKENS:
        if tid not in existing_ids:
            added_list.append(
                {
                    "id": tid,
                    "content": content,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": False,
                }
            )
    tok_json_path.write_text(json.dumps(tok_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    atd = tok_cfg.setdefault("added_tokens_decoder", {})
    for content, tid in MINIMAX_EXTRA_TOKENS:
        key = str(tid)
        if key not in atd:
            atd[key] = {
                "content": content,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": False,
            }
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {tok_json_path} and {tok_cfg_path} with MiniMax H3 tokens 151669..151675")


def split_h3_art(
    articles: list[str],
    rows: int,
    columns: int,
    tokenizer: Tokenizer,
    add_eos: bool = True,
) -> list[torch.Tensor]:
    """
    Pack H3 prompt strings and chat-templated self-report examples into (1, columns) rows
    with `encode_special_tokens=True` and `add_bos=False` so `<|im_start|>` (151644),
    `<|im_end|>` (151645), and `<d>`..`<|caption_end|>` (151669..151675) are encoded as
    single special tokens without injecting `<|endoftext|>` (151643) as a spurious BOS.
    """
    t_rows: list[torch.Tensor] = []
    idx = 0
    empty = torch.empty((1, 0), dtype=torch.long)
    t_row = empty
    while len(t_rows) < rows:
        t_art = tokenizer.encode(
            articles[idx % len(articles)],
            add_bos=False,
            add_eos=add_eos,
            encode_special_tokens=True,
        )
        t_row = torch.cat((t_row, t_art), dim=-1)
        while t_row.shape[-1] >= columns and len(t_rows) < rows:
            t_rows.append(t_row[:, :columns])
            t_row = t_row[:, columns:]
        idx += 1
    return t_rows


def collect_h3_texts(prompt_bank_dir: Path) -> tuple[list[str], list[str]]:
    scene_texts: list[str] = []
    cv_texts: list[str] = []

    # 1. Raw H3 scene prompts (t2va, i2va, fl2va, l2va, ref2va) — no chat template, matching H3 encoder usage
    txt_paths = sorted(prompt_bank_dir.glob("*.txt"))
    rng = random.Random(0)
    rng.shuffle(txt_paths)
    for txt_path in txt_paths:
        content = txt_path.read_text(encoding="utf-8").strip()
        if content:
            scene_texts.append(content)

    # 2. Control vector contrastive pairs (.toml) — round-robin interleaved across all .toml files
    cv_dir = prompt_bank_dir / "control_vectors"
    if cv_dir.is_dir():
        per_toml_lists: list[list[str]] = []
        for toml_path in sorted(cv_dir.glob("*.toml")):
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            vec_meta = data.get("vector", {})
            quality = vec_meta.get("quality", toml_path.stem)
            a_side = vec_meta.get("a_side", "neutral baseline")
            b_side = vec_meta.get("b_side", "steered concept")
            items: list[str] = []
            for pair in data.get("pair", []):
                a_txt = pair.get("a", "").strip()
                b_txt = pair.get("b", "").strip()
                if a_txt and b_txt:
                    items.append(a_txt)
                    items.append(b_txt)
                    # Chat-templated self-report / labelling format for readout layers 50..63
                    chat_example = (
                        "<|im_start|>system\n"
                        "You label the visual and acoustic concept direction between two H3 conditioning states "
                        "or describe the steered activation state concisely.<|im_end|>\n"
                        "<|im_start|>user\n"
                        f"Compare baseline pole A against steered pole B and state the control vector quality:\n\n"
                        f"Pole A:\n{a_txt}\n\nPole B:\n{b_txt}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                        f"Control vector quality: {quality}\n"
                        f"- Baseline (A): {a_side}\n"
                        f"- Steered (B): {b_side}<|im_end|>"
                    )
                    items.append(chat_example)
            if items:
                per_toml_lists.append(items)
        max_len = max((len(lst) for lst in per_toml_lists), default=0)
        for i in range(max_len):
            for lst in per_toml_lists:
                if i < len(lst):
                    cv_texts.append(lst[i])

    return scene_texts, cv_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H3 hybrid calibration safetensors for ExLlamaV3")
    parser.add_argument(
        "-i",
        "--in_dir",
        type=Path,
        default=Path.home() / "Storage" / "MiniMaxAI_MiniMax-H3" / "text_encoder",
        help="Path to Qwen3-VL-32B-Instruct checkpoint directory",
    )
    parser.add_argument(
        "-p",
        "--prompt_bank",
        type=Path,
        default=Path.home() / "ComfyUI" / "custom_nodes" / "ComfyUI-h3-explorations" / "prompt_bank",
        help="Path to H3 prompt_bank directory",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        type=Path,
        default=Path(__file__).resolve().parent / "h3_cal_data.safetensors",
        help="Output safetensors path",
    )
    parser.add_argument(
        "--include-untrained-minimax-tokens",
        action="store_true",
        help=(
            "Opt-in: map <d>..<|caption_end|> to untrained padding IDs 151669..151675. "
            "Disabled by default because rows 151669..151675 are untrained in Qwen3-VL-32B-Instruct "
            "(random init in embed_tokens, dead identical rows in lm_head)."
        ),
    )
    parser.add_argument(
        "--patch-model-dir",
        type=Path,
        default=None,
        help="Optional (not recommended for chat/VLM): patch tokenizer.json and tokenizer_config.json with IDs 151669..151675 and exit",
    )
    parser.add_argument("--total_rows", type=int, default=250)
    parser.add_argument("--h3_rows", type=int, default=50)
    parser.add_argument("--cols", type=int, default=2048)
    args = parser.parse_args()

    if args.patch_model_dir is not None:
        patch_model_dir_tokenizer(args.patch_model_dir)
        return

    std_rows_target = args.total_rows - args.h3_rows
    assert std_rows_target > 0 and args.h3_rows >= 0

    config = Config.from_directory(str(args.in_dir))
    tokenizer = Tokenizer.from_config(config)
    if args.include_untrained_minimax_tokens:
        register_minimax_tokens(tokenizer)

    # 1. Standard ExLlamaV3 corpus rows (200 rows)
    std_rows = get_default_calibration({"cal_rows": std_rows_target, "cal_cols": args.cols}, tokenizer)

    # 2. H3 prompt_bank + control_vector rows (50 rows, split 25 scene + 25 control-vector)
    scene_texts, cv_texts = collect_h3_texts(args.prompt_bank)
    if not scene_texts and not cv_texts:
        raise RuntimeError(f"No H3 prompt texts found under {args.prompt_bank}")
    scene_row_count = args.h3_rows // 2
    cv_row_count = args.h3_rows - scene_row_count
    h3_scene_rows = split_h3_art(scene_texts, scene_row_count, args.cols, tokenizer, add_eos=True)
    h3_cv_rows = split_h3_art(cv_texts, cv_row_count, args.cols, tokenizer, add_eos=True)
    h3_rows = h3_scene_rows + h3_cv_rows

    all_rows = std_rows + h3_rows
    assert len(all_rows) == args.total_rows, f"Expected {args.total_rows} rows, got {len(all_rows)}"
    packed = torch.cat(all_rows, dim=0).to(torch.long).contiguous()
    assert packed.shape == (args.total_rows, args.cols)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": packed}, str(args.out_file))
    print(
        f"Saved {packed.shape[0]} x {packed.shape[1]} hybrid calibration tensor "
        f"({std_rows_target} standard + {scene_row_count} H3 scene + {cv_row_count} H3 control-vector rows) to {args.out_file}"
    )


if __name__ == "__main__":
    main()
