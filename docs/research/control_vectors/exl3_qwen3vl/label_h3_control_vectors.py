#!/usr/bin/env python3
"""
Batch-label MiniMax H3 control vectors using the quantized ExLlamaV3
`Qwen3-VL-32B-Instruct` model (`5.37 bpw` or `4.65 bpw`).

Key Interpretability & Calibration Features:
  1. **Official Qwen3-VL-32B-Instruct Non-Thinking Sampling (`ComboSampler`) + Coupled Gumbel Draws**:
     - Uses Qwen3-VL's recommended non-thinking sampling parameters (`temperature=0.7`,
       `top_p=0.8`, `top_k=20`, `min_p=0.0`, `rep_p=1.05`) for steered sensory rollouts
       with matched per-step Gumbel `rand_u32(seed, step)` across `alpha`, eliminating
       greedy `temp=0` mode collapse while keeping low-temperature (`temperature=0.1`)
       decoding for the unsteered JSON synthesizer calls.
  2. **User-Scene + Assistant Span-Targeted `layers.49` Steering**:
     - Advances `steer_start_idx` past `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n`
       and masks out all ChatML special tokens (`tid >= 151643`) so system instructions and
       turn-boundary control tokens (`<|im_end|>`, `<|im_start|>`, `<think>`, `</think>`)
       remain unperturbed while the user scene nouns, the assistant's
       `integrated_multimodal_description: [Shot 1] ` prefix, and all generated tokens
       receive `alpha * median_row_norm * v_encoder`.
  3. **Direct `layers.49 -> lm_head` Word-Initial Log-Odds Readout (`delta_log_prob`)**:
     - Computes exact first-token log-probability shifts `log P(tok | steered) - log P(tok | base)`
       over genuine space-prefixed English content words (`piece.startswith(" ")`, `tid < 151643`)
       across four H3-formatted probe prefixes (visual/film look, human anatomy/material,
       surface/typography, and acoustics/voice).
  4. **Two H3-Formatted Neutral Anchors (`integrated_multimodal_description:` + `overall_soundscape:`)**:
     - `Anchor A (anchor_human_action_room)`: A person at a counter pouring water from a pitcher
       into a ceramic mug and speaking a short greeting aloud (probing human material/anatomy,
       hand-task coordination, vocal delivery, and room acoustics).
     - `Anchor B (anchor_surface_street_camera)`: A daytime shot moving past a plain brick
       storefront window, an awning, and a parked delivery van (probing film stock/lighting,
       surface signage/lettering, camera motion, and street soundscape).
  5. **Baseline-Subtracting Synthesizer + Markdown Report (`h3_control_vector_report.md`)**:
     - Explicitly instructs the synthesizer to subtract `+0.00` baseline prompt objects and
       writes both `h3_control_vector_labels.json` and `h3_control_vector_report.md` (including
       the `12x12` `layers.49` cosine similarity matrix).
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib
import types
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


def _resolve_h3_root() -> Path:
    env_h3 = os.environ.get("H3_REPO_ROOT")
    if env_h3 and (Path(env_h3) / "prompt_bank" / "control_vectors").is_dir():
        return Path(env_h3)
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for parent in [here.parent, *here.parents, cwd, *cwd.parents]:
        if (parent / "prompt_bank" / "control_vectors").is_dir():
            return parent
        candidate = parent / "ComfyUI" / "custom_nodes" / "ComfyUI-h3-explorations"
        if (candidate / "prompt_bank" / "control_vectors").is_dir():
            return candidate
    return here.parent


def _resolve_start_sh_var(h3_root: Path, key: str) -> str | None:
    start_sh = h3_root.parents[1] / "start.sh"
    if not start_sh.is_file():
        return None
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}=['\"]?([^'\"\n#]+)")
    for line in start_sh.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            return m.group(1).strip()
    return None


def _resolve_exl3_root(h3_root: Path) -> Path:
    env_root = os.environ.get("EXLLAMAV3_ROOT")
    if env_root and (Path(env_root) / "exllamav3").is_dir():
        return Path(env_root)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "exllamav3").is_dir():
            return parent
    base = h3_root.parents[2]
    for candidate in base.glob("*/*"):
        if candidate.name == "exllamav3" and (candidate / "exllamav3").is_dir():
            return candidate
    raise RuntimeError("Could not locate exllamav3 root; set EXLLAMAV3_ROOT.")


def _default_model_dir(h3_root: Path) -> Path:
    env_dir = os.environ.get("H3_EXL3_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    for cand in h3_root.parents[2].glob("*/Qwen3-VL-32B-Instruct-exl3-h3-5.37bpw"):
        if cand.is_dir():
            return cand
    return Path("Qwen3-VL-32B-Instruct-exl3-h3-5.37bpw")


_H3_ROOT = _resolve_h3_root()
_EXL3_ROOT = _resolve_exl3_root(_H3_ROOT)
if str(_EXL3_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXL3_ROOT))

from exllamav3 import Cache, CacheLayer_quant, ComboSampler, Config, Model, Tokenizer

QWEN_FIRST_SPECIAL_ID = 151643
DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.DOTALL)
SPEAKER_RE = re.compile(r"\(S\d+(?:,\s*S\d+)*\)")
CLEAN_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z\-]{2,}$")

STOPWORDS = frozenset({
    "the", "and", "with", "from", "that", "this", "into", "over", "like", "very",
    "more", "most", "some", "such", "while", "where", "when", "which", "their",
    "there", "have", "been", "were", "they", "them", "then", "than", "only",
    "also", "just", "even", "both", "each", "upon", "onto", "through", "across",
    "under", "between", "around", "behind", "beside", "within", "without", "about",
    "after", "before", "during", "toward", "towards", "against", "along", "amid",
    "among", "beneath", "beyond", "inside", "outside", "throughout", "underneath",
    "what", "whose", "whom", "these", "those", "other", "another", "every", "either",
    "neither", "whether", "though", "although", "because", "since", "unless", "until",
    "while", "whereby", "wherein", "whereupon", "almost", "nearly", "quite", "rather",
    "somewhat", "slightly", "highly", "fully", "mostly", "mainly", "largely", "partly",
    "entirely", "totally", "wholly", "truly", "really", "actually", "simply", "merely",
    "barely", "hardly", "scarcely", "often", "always", "never", "sometimes", "usually",
    "featured", "features", "featuring", "includes", "including", "shows", "showing",
    "appears", "appearing", "seems", "seeming", "looks", "looking", "sounds", "sounding",
    "makes", "making", "gives", "giving", "takes", "taking", "holds", "holding",
    "stands", "standing", "sits", "sitting", "moves", "moving", "turns", "turning",
    "comes", "coming", "goes", "going", "remains", "remaining", "stays", "staying",
    "keeps", "keeping", "leaves", "leaving", "brings", "bringing", "carries", "carrying",
})

AMBIGUOUS_SHORT_STEMS = frozenset({
    "com", "cor", "con", "pro", "pre", "per", "sub", "sup", "inter", "intra",
    "trans", "super", "hyper", "ultra", "anti", "auto", "semi", "multi", "poly",
    "mono", "micro", "macro", "tele", "photo", "bio", "geo", "neo", "pseudo",
    "str", "cro", "min", "inc", "exc", "imp", "exp", "ind", "inf", "ins", "int",
    "abs", "acc", "add", "adj", "adm", "adv", "aff", "agg", "all", "alt", "amb",
    "app", "arr", "ass", "att", "auth", "aux", "cha", "chi", "cho", "chr", "cir",
    "cla", "cli", "clo", "clu", "coa", "col", "comp", "conc", "cond", "conf",
    "cons", "cont", "conv", "corr", "cou", "cra", "cre", "cri", "cru", "cul",
    "cur", "cus", "dec", "def", "deg", "del", "dem", "den", "dep", "der", "des",
    "det", "dev", "dia", "dif", "dig", "dil", "dim", "dir", "dis", "div", "doc",
    "dom", "dou", "dra", "dre", "dri", "dro", "ele", "emb", "emp", "enc", "end",
    "eng", "enh", "enl", "enr", "ens", "ent", "env", "epi", "equ", "err", "esc",
    "ess", "est", "eva", "eve", "evi", "evo", "exa", "exc", "exe", "exh", "exi",
    "exo", "exp", "ext", "fac", "fai", "fal", "fam", "fan", "far", "fas", "fat",
    "fav", "fea", "fed", "fee", "fel", "fer", "fes", "fic", "fie", "fig", "fil",
    "fin", "fir", "fis", "fit", "fix", "fla", "fle", "fli", "flo", "flu", "fly",
    "foc", "fol", "fon", "foo", "for", "fou", "fra", "fre", "fri", "fro", "fru",
    "ful", "fun", "fur", "fus", "fut", "gal", "gam", "gar", "gas", "gat", "gau",
    "gen", "ger", "ges", "gig", "gla", "gle", "gli", "glo", "glu", "gly", "goa",
    "god", "gol", "goo", "gor", "gos", "gov", "gra", "gre", "gri", "gro", "gru",
    "gua", "gue", "gui", "gul", "gum", "gun", "gur", "gus", "gut", "gym", "hab",
    "hac", "hai", "hal", "ham", "han", "hap", "har", "has", "hat", "hau", "hav",
    "haz", "hea", "hee", "hei", "hel", "hem", "hen", "her", "hes", "hex", "hib",
    "hid", "hig", "hil", "him", "hin", "hip", "hir", "his", "hit", "hoa", "hob",
    "hoc", "hoe", "hog", "hol", "hom", "hon", "hoo", "hop", "hor", "hos", "hot",
    "hou", "how", "hub", "hue", "hug", "hum", "hun", "hur", "hus", "hut", "hyb",
    "hyd", "hyg", "hym", "hyp", "ide", "idi", "idl", "ign", "ill", "ima", "imb",
    "imi", "imm", "imp", "ina", "inc", "ind", "ine", "inf", "ing", "inh", "ini",
    "inj", "ink", "inl", "inm", "inn", "ino", "inp", "inq", "ins", "int", "inv",
    "inw", "ion", "ira", "ire", "iri", "iro", "irr", "isl", "iso", "iss", "ite",
    "writ", "crud", "scram", "whims",
})


DOMAIN_WORDS: frozenset[str] = frozenset({
    "noir", "grayscale", "monochrome", "fisheye", "vignette", "levitate", "levitation",
    "cranial", "cranium", "reverberation", "reverberate", "reverberant", "celluloid",
    "stenciled", "stencil", "signage", "archival", "soundscape", "diegetic",
    "anamorphic", "bokeh", "sepia", "halftone", "camcorder", "cctv", "vhs",
    "muppet", "puppet", "weightless", "weightlessness", "chiaroscuro", "halation",
    "phosphor", "scanlines", "warble", "subwoofer", "handheld", "rectilinear",
    "telephoto", "macroblock", "bitcrushed", "liquefy", "liquefied", "globular",
    "bulbous", "inflatable", "inflated", "swollen", "ballooning", "buoyant", "buoyancy",
    "midair", "airborne", "skyward", "upward", "downward", "glissando", "fumble",
    "fumbling", "hilarious", "hilariously", "harmonious", "clumsy", "clumsily",
    "disastrous", "disastrously", "catastrophic", "grotesque", "monstrous",
    "monstrously", "cavernous", "whimsical", "writhing", "crudely", "scrambling",
    "morphing", "enlarged", "enlarging", "expansive", "capitalized", "antiquated",
    "quintessential",
})


def _load_english_dictionary() -> tuple[frozenset[str], dict[str, str]]:
    """
    Load `/usr/share/dict/words` plus `DOMAIN_WORDS` to validate whole English words.
    """
    words: set[str] = set(DOMAIN_WORDS)
    dict_path = Path("/usr/share/dict/words")
    if dict_path.is_file():
        for line in dict_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if not raw or not raw[0].islower():
                continue
            w = raw.lower()
            if len(w) >= 3 and w.isalpha():
                words.add(w)
    stem_to_word: dict[str, str] = {}
    for w in sorted(words, key=lambda x: (len(x), x)):
        for k in range(5, min(len(w), 10)):
            prefix = w[:k]
            if prefix not in words and prefix not in stem_to_word:
                stem_to_word[prefix] = w
    stem_to_word.update({
        "hilar": "hilarious",
        "stenc": "stenciled",
        "cellul": "celluloid",
        "rever": "reverberation",
        "clums": "clumsily",
        "disastr": "disastrously",
        "liquef": "liquefied",
        "chiar": "chiaroscuro",
        "phosph": "phosphor",
    })
    return frozenset(words), stem_to_word


_DICT_WORDS, _STEM_COMPLETIONS = _load_english_dictionary()
_BPE_VOCAB_CACHE: tuple[list[int], dict[int, str]] | None = None


def _get_bpe_vocab_map(tokenizer: Tokenizer) -> tuple[list[int], dict[int, str]]:
    """
    Build and cache `(valid_word_ids, word_map)` using actual Qwen BPE tokenization:
      1. Any space-prefixed token whose stripped lowercase form is a whole word in
         `DOMAIN_WORDS` or `_DICT_WORDS` (`len >= 4`, not in `STOPWORDS` or
         `AMBIGUOUS_SHORT_STEMS`) maps directly to itself (`noir`, `grayscale`,
         `signage`, `archival`, `cellular`, `float`, `inflate`, `balloon`).
      2. Remaining space-prefixed subword tokens are matched by checking whether
         `tokenizer.encode(" " + w)[0] == tid` for multi-token words `w` (checking
         `DOMAIN_WORDS` first so `' fis'` (`41623`) -> `'fisheye'`, `' vign'` (`92792`)
         -> `'vignette'`, `' cran'` (`69537`) -> `'cranial'`, `' cellul'` (`68902`)
         -> `'celluloid'`, `' rever'` (`35137`) -> `'reverberation'`, and never
         confusing `' cellul'` with single-token `' cellular'` `34593`).
    """
    global _BPE_VOCAB_CACHE
    if _BPE_VOCAB_CACHE is not None:
        return _BPE_VOCAB_CACHE

    id_to_piece = tokenizer.get_id_to_piece_list(include_special_tokens=False)
    vocab_limit = min(len(id_to_piece), tokenizer.raw_vocab_size, QWEN_FIRST_SPECIAL_ID)

    space_pieces: dict[str, list[int]] = {}
    for tid in range(vocab_limit):
        piece = id_to_piece[tid]
        if not piece.startswith(" "):
            continue
        cleaned = piece.strip()
        if not CLEAN_WORD_RE.match(cleaned):
            continue
        space_pieces.setdefault(cleaned.lower(), []).append(tid)

    # Verify which dictionary words are genuine single-token BPE words
    single_token_words: set[str] = set()
    for low in space_pieces:
        if low in DOMAIN_WORDS or (len(low) >= 4 and low in _DICT_WORDS and low not in AMBIGUOUS_SHORT_STEMS):
            single_token_words.add(low)

    # Map multi-token words to their actual first BPE token ID (`DOMAIN_WORDS` prioritized first)
    tid_to_stem_word: dict[int, str] = {}
    domain_First = sorted(
        (w for w in DOMAIN_WORDS if w not in single_token_words),
        key=lambda x: (len(x), x),
    )
    dict_Rest = sorted(
        (w for w in _DICT_WORDS if w not in single_token_words and w not in DOMAIN_WORDS),
        key=lambda x: (len(x), x),
    )
    for is_domain, word_list in ((True, domain_First), (False, dict_Rest)):
        min_k = 3 if is_domain else 5
        for w in word_list:
            for k in range(min_k, min(len(w), 10)):
                pref = w[:k]
                if k == 3 and pref != "fis":
                    continue
                if not is_domain and pref in AMBIGUOUS_SHORT_STEMS:
                    continue
                cand_tids = space_pieces.get(pref)
                if not cand_tids or pref in single_token_words:
                    continue
                if all(t in tid_to_stem_word for t in cand_tids):
                    break
                enc = tokenizer.encode(" " + w, add_bos=False, add_eos=False)
                first_tid = int(enc[0, 0].item())
                if first_tid in cand_tids and first_tid not in tid_to_stem_word:
                    tid_to_stem_word[first_tid] = w
                    break

    valid_word_ids: list[int] = []
    word_map: dict[int, str] = {}
    for tid in range(vocab_limit):
        piece = id_to_piece[tid]
        if not piece.startswith(" "):
            continue
        cleaned = piece.strip()
        low = cleaned.lower()
        if not CLEAN_WORD_RE.match(cleaned) or low in STOPWORDS:
            continue
        if low in single_token_words:
            resolved = low
        elif tid in tid_to_stem_word:
            resolved = tid_to_stem_word[tid]
        elif len(low) >= 5 and low not in AMBIGUOUS_SHORT_STEMS and low in _STEM_COMPLETIONS:
            resolved = _STEM_COMPLETIONS[low]
        else:
            continue
        if resolved in STOPWORDS:
            continue
        valid_word_ids.append(tid)
        word_map[tid] = resolved

    _BPE_VOCAB_CACHE = (valid_word_ids, word_map)
    return _BPE_VOCAB_CACHE


def format_chatml(system_prompt: str, user_prompt: str, assistant_prefix: str = "") -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefix}"
    )


def chatml_user_start_idx(tokenizer: Tokenizer, system_prompt: str) -> int:
    header = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
    return int(
        tokenizer.encode(
            header, add_bos=False, add_eos=False, encode_special_tokens=True
        ).shape[-1]
    )


def chatml_assistant_start_idx(tokenizer: Tokenizer, system_prompt: str, user_prompt: str) -> int:
    header = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return int(
        tokenizer.encode(
            header, add_bos=False, add_eos=False, encode_special_tokens=True
        ).shape[-1]
    )


def get_chatml_stop_ids(tokenizer: Tokenizer) -> set[int]:
    stops: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.single_id(tok_str)
        if tid is not None:
            stops.add(int(tid))
    return stops


def step_rand_u32(seed: int, step_idx: int) -> int:
    """Deterministic per-step uint32 Gumbel seed for Coupled Random Numbers across alphas."""
    h = hashlib.blake2b(f"{seed}:{step_idx}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little")


def scene_text_mask_from_ids(input_ids: torch.Tensor, tokenizer: Tokenizer) -> torch.BoolTensor:
    """
    Build a boolean mask [L] selecting `scene_text` tokens (excluding row 0,
    `<d>...</d>` dialogue spans, and `(S1)` speaker tags) on Qwen's vocabulary.
    """
    ids = input_ids[0].tolist()
    L = len(ids)
    id_to_piece = tokenizer.get_id_to_piece_list(include_special_tokens=True)
    pieces = [id_to_piece[i] if 0 <= i < len(id_to_piece) else "" for i in ids]
    full_text = "".join(pieces)

    excluded_spans: list[tuple[int, int]] = []
    for m in DIALOGUE_RE.finditer(full_text):
        excluded_spans.append((m.start(), m.end()))
    for m in SPEAKER_RE.finditer(full_text):
        excluded_spans.append((m.start(), m.end()))

    mask = torch.ones(L, dtype=torch.bool)
    if L > 1:
        mask[0] = False  # row0_excluded=True

    pos = 0
    for idx, piece in enumerate(pieces):
        c0, c1 = pos, pos + len(piece)
        pos = c1
        for m0, m1 in excluded_spans:
            if max(c0, m0) < min(c1, m1):
                mask[idx] = False
                break

    if not bool(mask.any()):
        mask[:] = True
        if L > 1:
            mask[0] = False
    return mask


@torch.inference_mode()
def forward_to_layer49(model: Model, layer_49_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Run `embed_tokens` + `layers.0..49` (up to `model.modules[layer_49_idx]`) using
    non-cached FlashAttention (`flash_attn_nc`) and return the materialized fp32
    residual stream `[1, L, 5120]`.
    """
    params: dict[str, Any] = {
        "attn_mode": "flash_attn_nc",
        "input_ids": input_ids,
    }
    x: torch.Tensor = input_ids
    for module in model.modules[: layer_49_idx + 1]:
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
    pending = params.pop("resid_pending", None)
    if pending is not None:
        x = x.float() + pending.float()
    return x.float()


def _compute_pair_quality(
    a_mat: torch.Tensor,
    b_mat: torch.Tensor,
    w_vec: torch.Tensor,
) -> dict[str, Any]:
    """
    Compute per-pair consistency metrics (`mean_pairwise_cosine`, `split_half_cosine`,
    `per_pair_cosine`, `min_pair_cosine`, `weakest_pair_1based`) on `[N, D]` fp64 matrices.
    """
    diffs = (b_mat - a_mat).to(torch.float64)
    n = int(diffs.shape[0])
    if n < 2:
        return {
            "mean_pairwise_cosine": 1.0,
            "split_half_cosine": 1.0,
            "min_pair_cosine": 1.0,
            "weakest_pair_1based": 1,
            "per_pair_cosine": [1.0] * n,
        }
    norms = torch.linalg.vector_norm(diffs, dim=-1, keepdim=True).clamp(min=1e-12)
    unit_diffs = diffs / norms
    cos_mat = unit_diffs @ unit_diffs.T

    # Off-diagonal mean pairwise cosine and leave-one-out per-pair cosine
    per_pair: list[float] = []
    off_diag_vals: list[float] = []
    for i in range(n):
        others = [float(cos_mat[i, j].item()) for j in range(n) if j != i]
        per_pair.append(round(sum(others) / len(others), 4))
        for j in range(i + 1, n):
            off_diag_vals.append(float(cos_mat[i, j].item()))
    mean_pw = round(sum(off_diag_vals) / max(len(off_diag_vals), 1), 4)

    # Split-half cosine (first half vs second half)
    mid = n // 2
    w1 = w_vec[:mid] / w_vec[:mid].sum().clamp(min=1e-12)
    w2 = w_vec[mid:] / w_vec[mid:].sum().clamp(min=1e-12)
    v1 = (diffs[:mid] * w1.unsqueeze(-1)).sum(dim=0)
    v2 = (diffs[mid:] * w2.unsqueeze(-1)).sum(dim=0)
    n1 = float(torch.linalg.vector_norm(v1).item())
    n2 = float(torch.linalg.vector_norm(v2).item())
    split_cos = round(float(torch.dot(v1, v2).item() / max(n1 * n2, 1e-12)), 4)

    min_idx = min(range(n), key=lambda i: per_pair[i])
    return {
        "mean_pairwise_cosine": mean_pw,
        "split_half_cosine": split_cos,
        "min_pair_cosine": per_pair[min_idx],
        "weakest_pair_1based": min_idx + 1,
        "per_pair_cosine": per_pair,
    }


@torch.inference_mode()
def compute_encoder_vector_from_toml(
    model: Model,
    tokenizer: Tokenizer,
    toml_path: Path,
    pooling: str = "scene_text",
) -> tuple[torch.Tensor, float, float, list[dict[str, str]], dict[str, Any]]:
    """
    Compute unit-normalized `vector.encoder` ([5120], fp32), `median_row_norm`
    (with `row0_excluded=True`), `raw_norm`, `diff_records`, and `l49_quality`
    directly from a `.toml` pair file using `layers.0..49` of the ExLlamaV3 model.

    Supported `pooling` modes:
      - `"scene_text"` (default): all `scene_text` rows (excluding row 0, `<d>...</d>`, and `(S1)`).
      - `"from_first_diff"`: encodes the full untruncated prompt and masks `scene_text` rows to
        `idx >= first_diff` within the modified section (masking out unchanged trailing
        `overall_soundscape:` / `non_diegetic_music:` tokens in the boolean mask while keeping
        `<d>...</d>` and `(S1)` excluded).
    """
    layer_49 = model.find_module("model.language_model.layers.49")
    layer_49_idx = model.modules.index(layer_49)

    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    frame = data.get("vector", {}).get("frame", "scene")
    pairs = data.get("pair", [])
    if not pairs:
        raise ValueError(f"No [[pair]] entries in {toml_path}")

    a_pooled: list[torch.Tensor] = []
    b_pooled: list[torch.Tensor] = []
    weights: list[float] = []
    median_norms: list[float] = []
    diff_records: list[dict[str, str]] = []

    for p in pairs:
        a_txt = p.get("a", "").strip()
        b_txt = p.get("b", "").strip()
        if not a_txt or not b_txt:
            continue
        w = float(p.get("weight", 1.0))

        a_span, b_span = extract_diff_span(a_txt, b_txt)
        diff_records.append({"a_span": a_span, "b_span": b_span})

        ids_a = tokenizer.encode(a_txt, add_bos=False, add_eos=False, encode_special_tokens=True)
        ids_b = tokenizer.encode(b_txt, add_bos=False, add_eos=False, encode_special_tokens=True)

        h_a = forward_to_layer49(model, layer_49_idx, ids_a)[0]  # [L_a, 5120]
        h_b = forward_to_layer49(model, layer_49_idx, ids_b)[0]  # [L_b, 5120]

        mask_a = scene_text_mask_from_ids(ids_a, tokenizer).to(h_a.device)
        mask_b = scene_text_mask_from_ids(ids_b, tokenizer).to(h_b.device)

        if pooling == "from_first_diff" and frame != "bare":
            list_a = ids_a[0].tolist()
            list_b = ids_b[0].tolist()
            fdiff = 0
            min_tok_len = min(len(list_a), len(list_b))
            while fdiff < min_tok_len and list_a[fdiff] == list_b[fdiff]:
                fdiff += 1
            if fdiff > 0:
                mask_a[:fdiff] = False
                mask_b[:fdiff] = False

            # Mask out unchanged trailing overall_soundscape: tokens in the boolean mask
            # (without truncating the prompt before tokenization).
            for marker in ("\n\noverall_soundscape:", "\noverall_soundscape:"):
                if marker in a_txt and marker in b_txt:
                    a_pre, a_tail = a_txt.split(marker, 1)
                    b_pre, b_tail = b_txt.split(marker, 1)
                    if a_tail == b_tail and a_pre != b_pre:
                        len_pre_a = int(
                            tokenizer.encode(
                                a_pre, add_bos=False, add_eos=False, encode_special_tokens=True
                            ).shape[-1]
                        )
                        len_pre_b = int(
                            tokenizer.encode(
                                b_pre, add_bos=False, add_eos=False, encode_special_tokens=True
                            ).shape[-1]
                        )
                        mask_a[len_pre_a:] = False
                        mask_b[len_pre_b:] = False
                        break

            if not bool(mask_a.any()):
                mask_a[-1] = True
            if not bool(mask_b.any()):
                mask_b[-1] = True

        norms_a = torch.linalg.vector_norm(h_a[1:] if h_a.shape[0] > 1 else h_a, dim=-1)
        norms_b = torch.linalg.vector_norm(h_b[1:] if h_b.shape[0] > 1 else h_b, dim=-1)
        median_norms.append(float(norms_a.median().item()))
        median_norms.append(float(norms_b.median().item()))

        a_pooled.append(h_a[mask_a].to(torch.float64).mean(dim=0))
        b_pooled.append(h_b[mask_b].to(torch.float64).mean(dim=0))
        weights.append(w)

    a_mat = torch.stack(a_pooled, dim=0)
    b_mat = torch.stack(b_pooled, dim=0)
    w_vec = torch.tensor(weights, dtype=torch.float64, device=a_mat.device)
    w_vec = w_vec / w_vec.sum()

    raw_vec = ((b_mat - a_mat) * w_vec.unsqueeze(-1)).sum(dim=0)
    raw_norm = float(torch.linalg.vector_norm(raw_vec).item())
    unit_vec = (raw_vec / max(raw_norm, 1e-12)).to(dtype=torch.float32).cpu()
    median_row_norm = float(sum(median_norms) / len(median_norms))
    l49_quality = _compute_pair_quality(a_mat.cpu(), b_mat.cpu(), w_vec.cpu())
    return unit_vec, median_row_norm, raw_norm, diff_records, l49_quality


def extract_diff_span(a_txt: str, b_txt: str) -> tuple[str, str]:
    p = 0
    min_len = min(len(a_txt), len(b_txt))
    while p < min_len and a_txt[p] == b_txt[p]:
        p += 1
    s = 0
    while (
        s < (min_len - p)
        and a_txt[len(a_txt) - 1 - s] == b_txt[len(b_txt) - 1 - s]
    ):
        s += 1
    a_mid = a_txt[p : len(a_txt) - s].strip()
    b_mid = b_txt[p : len(b_txt) - s].strip()
    return a_mid, b_mid


def load_encoder_vector_from_safetensors(sf_path: Path) -> tuple[torch.Tensor | None, float | None, dict[str, Any]]:
    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        meta_raw = f.metadata() or {}
        if "control_vector" in meta_raw:
            meta = json.loads(meta_raw["control_vector"])
        else:
            meta = {}
            for k, v in meta_raw.items():
                try:
                    meta[k] = json.loads(v)
                except Exception:
                    meta[k] = v
        if "vector.encoder" in f.keys():
            vec = f.get_tensor("vector.encoder").to(torch.float32)
            norm = float(torch.linalg.vector_norm(vec.to(torch.float64)).item())
            if norm > 1e-12:
                vec = vec / norm
            med_norm = (
                meta.get("points", {})
                .get("encoder", {})
                .get("median_row_norm")
            )
            return vec, (float(med_norm) if med_norm is not None else None), meta
        return None, None, meta


@torch.inference_mode()
def compute_top_logit_shifts(
    model: Model,
    tokenizer: Tokenizer,
    vec_encoder: torch.Tensor,
    median_row_norm: float,
    alpha: float = 0.18,
    top_k: int = 12,
) -> dict[str, Any]:
    """
    Zero-sampling lexical lens: run 6 domain-neutral H3-formatted prefixes through
    `layers.0..49`, add `+/- alpha * median_row_norm * v_encoder` at `layers.49`
    (positions `1:`), and continue through `layers.50..63` + `norm` + `lm_head`
    to score space-prefixed English content words (`piece.startswith(" ")`,
    `tid < 151643`) by symmetric contrastive log-odds:
        `delta_sym(w) = 0.5 * (log P(w | +v) - log P(w | -v))`
    cancelling out symmetric off-manifold entropy/temperature flattening.
    """
    layer_49 = model.find_module("model.language_model.layers.49")
    layer_49_idx = model.modules.index(layer_49)
    delta = (
        vec_encoder.to(device=layer_49.device, dtype=torch.float32) * float(alpha * median_row_norm)
    ).view(1, 1, -1)

    prefixes = [
        "integrated_multimodal_description: [Shot 1] The visual medium, camera look, lighting, and image fidelity look",
        "integrated_multimodal_description: [Shot 1] The lighting, color palette, contrast, and film stock look",
        "integrated_multimodal_description: [Shot 1] The character's head proportions, facial features, hands, and body texture look",
        "integrated_multimodal_description: [Shot 1] As the character speaks in the room, the loose objects and the character's head gradually",
        "integrated_multimodal_description: [Shot 1] Across the storefront wall, signboard, and door panel, the surface is covered in bold",
        "integrated_multimodal_description: [Shot 1] Across the wall, signboard, and door panel, the surface shows",
        "overall_soundscape: The room acoustics, vocal delivery, reverberation, and background foley sound",
    ]

    id_to_piece = tokenizer.get_id_to_piece_list(include_special_tokens=False)
    vocab_limit = min(len(id_to_piece), tokenizer.raw_vocab_size, QWEN_FIRST_SPECIAL_ID)
    valid_word_ids, word_map = _get_bpe_vocab_map(tokenizer)
    valid_idx = torch.tensor(valid_word_ids, dtype=torch.long)

    pos_scores = torch.full((len(valid_word_ids),), -1e9, dtype=torch.float32)
    neg_scores = torch.full((len(valid_word_ids),), -1e9, dtype=torch.float32)

    def tail_logits(h49: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        params: dict[str, Any] = {
            "attn_mode": "flash_attn_nc",
            "input_ids": input_ids,
        }
        x = h49
        for module in model.modules[layer_49_idx + 1 :]:
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        pending = params.pop("resid_pending", None)
        if pending is not None:
            x = x.float() + pending.float()
        return x[0, -1, :vocab_limit].float().cpu()

    for prefix in prefixes:
        ids = tokenizer.encode(prefix, add_bos=False, add_eos=False, encode_special_tokens=True)
        h_base = forward_to_layer49(model, layer_49_idx, ids)
        d = delta.to(device=h_base.device, dtype=h_base.dtype)

        h_pos = h_base.clone()
        h_pos[:, 1:, :] = h_pos[:, 1:, :] + d
        h_neg = h_base.clone()
        h_neg[:, 1:, :] = h_neg[:, 1:, :] - d

        lp_base = F.log_softmax(tail_logits(h_base, ids), dim=-1)[valid_idx]
        lp_pos = F.log_softmax(tail_logits(h_pos, ids), dim=-1)[valid_idx]
        lp_neg = F.log_softmax(tail_logits(h_neg, ids), dim=-1)[valid_idx]

        # Median-centered directional log-odds shift cancels isotropic entropy flattening
        # without mixing -v suppression of generic words into +v domain activation
        raw_pos = lp_pos - lp_base
        raw_neg = lp_neg - lp_base
        pos_shift = raw_pos - torch.median(raw_pos)
        neg_shift = raw_neg - torch.median(raw_neg)
        pos_gain = torch.where(lp_pos > -11.5, pos_shift, torch.full_like(lp_pos, -1e9))
        neg_gain = torch.where(lp_neg > -11.5, neg_shift, torch.full_like(lp_neg, -1e9))
        pos_scores = torch.maximum(pos_scores, pos_gain)
        neg_scores = torch.maximum(neg_scores, neg_gain)

    def pick_unique(scores: torch.Tensor) -> tuple[list[str], list[float]]:
        order = torch.argsort(scores, descending=True).tolist()
        out: list[str] = []
        vals: list[float] = []
        seen: set[str] = set()
        for idx_i in order:
            w = word_map[valid_word_ids[idx_i]]
            if w not in seen and len(w) >= 3:
                seen.add(w)
                val = float(scores[idx_i].item())
                out.append(f"{w} ({val:+.2f})")
                vals.append(val)
                if len(out) >= top_k:
                    break
        return out, vals

    pos_words, pos_vals = pick_unique(pos_scores)
    neg_words, _ = pick_unique(neg_scores)
    top5_mean = sum(pos_vals[:5]) / max(len(pos_vals[:5]), 1)
    if top5_mean >= 3.2:
        lex_clarity = "high"
    elif top5_mean >= 2.2:
        lex_clarity = "medium"
    else:
        lex_clarity = "low"

    return {
        "positive_boosted_words": pos_words,
        "negative_boosted_words": neg_words,
        "top5_mean_delta_log_prob": round(top5_mean, 3),
        "lexical_clarity": lex_clarity,
    }


@torch.inference_mode()
def generate_cached(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    rep_p: float = 1.05,
    seed: int = 42,
    embeddings: list[Any] | None = None,
    layer_49_vec: torch.Tensor | None = None,
    layer_49_scale: float = 0.0,
    steer_start_idx: int = 1,
) -> str:
    """
    Run a single-sequence prefill + decode pass with:
      - `ComboSampler(temperature=..., top_p=..., top_k=..., min_p=..., rep_p=...)`
        (defaults match Qwen3-VL-32B-Instruct's official non-thinking sampling parameters)
      - Coupled per-step `rand_u32 = step_rand_u32(seed, step)` so rollouts share
        exact Gumbel draws across steering alphas
      - Span-targeted `layers.49` steering starting at `steer_start_idx` while leaving
        ChatML special tokens (`tid >= 151643`) unsteered.
    """
    layer_49 = model.find_module("model.language_model.layers.49")
    orig_forward = layer_49.forward

    context_ids = tokenizer.encode(
        prompt_text,
        add_bos=False,
        add_eos=False,
        encode_special_tokens=True,
        embeddings=embeddings,
    )

    if layer_49_vec is not None and abs(layer_49_scale) > 1e-9:
        delta = (
            layer_49_vec.to(device=layer_49.device, dtype=torch.float32) * float(layer_49_scale)
        ).view(1, 1, -1)

        # Build prefill mask: steer tokens >= steer_start_idx EXCEPT ChatML special tokens (>= 151643)
        prefill_ids = context_ids[0, :-1]
        prefill_mask = torch.zeros_like(prefill_ids, dtype=torch.bool)
        start_pos = min(max(1, steer_start_idx), prefill_ids.shape[0])
        if start_pos < prefill_ids.shape[0]:
            prefill_mask[start_pos:] = prefill_ids[start_pos:] < QWEN_FIRST_SPECIAL_ID

        def steered_forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
            x = orig_forward(x, params, out_dtype)
            pending = params.pop("resid_pending", None)
            if pending is not None:
                x = x.float() + pending.float()
            d = delta.to(device=x.device, dtype=x.dtype)
            if x.shape[1] > 1:
                m = prefill_mask[: x.shape[1]].to(device=x.device).view(1, -1, 1)
                x = torch.where(m, x + d, x)
            else:
                x = x + d
            return x

        layer_49.forward = types.MethodType(steered_forward, layer_49)

    try:
        sampler = ComboSampler(
            rep_p=rep_p,
            temperature=temperature,
            min_p=min_p,
            top_k=top_k,
            top_p=top_p,
        )
        stop_ids = get_chatml_stop_ids(tokenizer)

        prefill_params: dict[str, Any] = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (1, cache.max_num_tokens),
        }
        if embeddings:
            prefill_params["embeddings"] = embeddings
            if model.caps.get("mrope"):
                freqs, offset = model.g_rope.get_mrope_freqs(
                    context_ids,
                    embeddings,
                    context_ids.shape[-1] + max_new_tokens + 16,
                )
                prefill_params["alt_rope_freqs"] = freqs
                prefill_params["alt_rope_offset"] = offset - context_ids.shape[-1]

        if context_ids.shape[-1] > 1:
            model.prefill(input_ids=context_ids[:, :-1], params=prefill_params)

        alt_rope_freqs = prefill_params.get("alt_rope_freqs")
        generated_ids: list[int] = []

        for step_idx in range(max_new_tokens):
            step_params: dict[str, Any] = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": context_ids.shape[-1] - 1,
                "batch_shape": (1, cache.max_num_tokens),
            }
            if alt_rope_freqs is not None:
                step_params["alt_rope_freqs"] = alt_rope_freqs

            logits = model.forward(input_ids=context_ids[:, -1:], params=step_params)
            sample = sampler.forward(
                logits,
                sequence_ids=context_ids,
                rand_u32=step_rand_u32(seed, step_idx),
                tokenizer=tokenizer,
            )
            tid = int(sample.item())
            if tid in stop_ids:
                break
            generated_ids.append(tid)
            context_ids = torch.cat((context_ids, sample.cpu()), dim=-1)

        if not generated_ids:
            return ""
        out_tensor = torch.tensor([generated_ids], dtype=torch.long)
        return tokenizer.decode(out_tensor, decode_special_tokens=False)[0].strip()
    finally:
        layer_49.forward = orig_forward


def run_self_report_probe(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    vec_encoder: torch.Tensor,
    median_row_norm: float,
    alphas: list[float],
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    rep_p: float = 1.05,
    seed: int = 101,
) -> dict[str, Any]:
    """
    Combine:
      1. Direct `layers.49 -> lm_head` top boosted words (`delta_log_prob`) on space-prefixed words
      2. Coupled Qwen3-VL non-thinking rollouts (`temp=0.7, top_p=0.8, top_k=20, rep_p=1.05`)
         on two H3-formatted neutral anchors (`Anchor A`: Character + Hand Action + Spoken Line +
         Room Acoustics; `Anchor B`: Awning + Crate + Delivery Van Surface + Street + Soundscape)
         with steering applied to user scene + assistant tokens (skipping system & ChatML special tokens).
      3. Low-temperature (`temp=0.1`) structured JSON synthesis that explicitly subtracts
         the `alpha = +0.00` baseline elements.
    """
    lexical_shifts = compute_top_logit_shifts(
        model=model,
        tokenizer=tokenizer,
        vec_encoder=vec_encoder,
        median_row_norm=median_row_norm,
        alpha=max((a for a in alphas if a > 0), default=0.18),
    )

    sys_msg = (
        "You write vivid, concise H3 video-and-audio scene descriptions. "
        "For the user's scene brief, write a 2-sentence `integrated_multimodal_description: [Shot 1]` "
        "followed by a 1-sentence `overall_soundscape:`. Keep the entire response under 90 words."
    )
    assistant_prefix = "integrated_multimodal_description: [Shot 1] "
    steer_start_idx = chatml_user_start_idx(tokenizer, sys_msg)

    anchor_Specs = {
        "anchor_human_action_room": (
            "Scene brief: A medium shot of a person standing at a kitchen counter with a spoon and napkin "
            "resting beside a ceramic mug, pouring water from a pitcher and speaking a short greeting aloud. "
            "Include the person's head, face, and hands, the water and tabletop items, the vocal delivery, "
            "and the room acoustics."
        ),
        "anchor_surface_street_camera": (
            "Scene brief: A daytime shot framing a canvas shop awning, a wooden crate on the sidewalk, "
            "and the side panel of a parked white delivery van. Include the camera look and lighting, "
            "the surfaces of the awning, crate, and van panel, the camera motion, and the street acoustics."
        ),
    }

    sweep_outputs: dict[str, dict[str, str]] = {k: {} for k in anchor_Specs}
    for anchor_idx, (anchor_key, user_msg) in enumerate(anchor_Specs.items()):
        prompt_text = format_chatml(sys_msg, user_msg, assistant_prefix=assistant_prefix)
        anchor_seed = seed + (anchor_idx * 1000)

        for alpha in alphas:
            scale = float(alpha) * float(median_row_norm)
            text = generate_cached(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=140,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                rep_p=rep_p,
                seed=anchor_seed,
                layer_49_vec=vec_encoder,
                layer_49_scale=scale,
                steer_start_idx=steer_start_idx,
            )
            sweep_outputs[anchor_key][f"{alpha:+.2f}"] = assistant_prefix + text

    anchor_a_block = "\n".join(
        f"  alpha={k}: {v}" for k, v in sweep_outputs["anchor_human_action_room"].items()
    )
    anchor_b_block = "\n".join(
        f"  alpha={k}: {v}" for k, v in sweep_outputs["anchor_surface_street_camera"].items()
    )

    synth_prompt = format_chatml(
        "You analyze activation-steering readouts from layer 49 of a video-and-audio conditioning model and output only valid JSON.",
        "A hidden control vector was injected at layer 49 and read out using coupled rollouts across steering strengths "
        "(without any lexical token hints):\n\n"
        f"1. Coupled rollouts (temp={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}) on Anchor A "
        f"(Person pouring water into a mug at a counter + speaking + room sound):\n{anchor_a_block}\n\n"
        f"2. Coupled rollouts (temp={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}) on Anchor B "
        f"(Canvas shop awning + wooden crate + white delivery van panel + street sound):\n{anchor_b_block}\n\n"
        "IMPORTANT: Ignore baseline objects and actions that were already requested by the anchor prompts "
        "or already present at `alpha=+0.00` (such as the counter, pitcher, mug, canvas awning, wooden crate, "
        "or delivery van). Isolate ONLY the specific visual, physical, material, camera, lighting, vocal, or "
        "acoustic transformation that systematically emerges from negative (-0.15) through baseline (+0.00) "
        "to positive (+0.15, +0.22) across the coupled rollouts.\n\n"
        "Return a single valid JSON object with keys:\n"
        '  "self_report_slug": concise snake_case label (2-4 words) deduced strictly from the rollouts,\n'
        '  "self_report_quality": 1-sentence summary of the steered transformation,\n'
        '  "negative_pole": what negative steering (-0.15) produces,\n'
        '  "positive_pole": what positive steering (+0.15, +0.22) produces,\n'
        '  "dominant_modality": "visual" | "acoustic" | "audiovisual" | "typography",\n'
        '  "readout_clarity": "high" | "medium" | "low",\n'
        '  "clarity_reason": 1-sentence explanation of why rollout clarity is high, medium, or low',
        assistant_prefix="{\n",
    )
    raw_json = "{\n" + generate_cached(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        prompt_text=synth_prompt,
        max_new_tokens=360,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        rep_p=1.0,
        seed=seed + 500,
    )
    parsed = try_parse_json(raw_json)
    return {
        "median_row_norm": round(median_row_norm, 3),
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "rep_p": rep_p,
            "coupled_seed": seed,
        },
        "lexical_logit_shifts": lexical_shifts,
        "sweep_completions": sweep_outputs,
        "synthesis": parsed if parsed is not None else raw_json,
    }


def run_blind_pair_synthesis(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    diff_records: list[dict[str, str]],
    seed: int = 202,
) -> dict[str, Any] | str:
    lines = []
    for idx, rec in enumerate(diff_records[:10], start=1):
        lines.append(f"Pair {idx:02d}:\n  - Baseline (A): {rec['a_span']}\n  + Steered  (B): {rec['b_span']}")
    diffs_text = "\n".join(lines)

    prompt = format_chatml(
        "You are an expert researcher labelling contrastive control vectors for a joint video-and-audio diffusion model. Output only valid JSON.",
        "Below are the differing clauses extracted from 10 contrastive prompt pairs (Baseline Pole A vs. Steered Pole B). "
        "Without knowing the file name, deduce the exact latent concept being isolated:\n\n"
        f"{diffs_text}\n\n"
        "Return a single JSON object with keys:\n"
        '  "inferred_slug": concise snake_case identifier (2-4 words),\n'
        '  "inferred_quality": concise description of the vector quality,\n'
        '  "a_pole": baseline state in Pole A,\n'
        '  "b_pole": steered state in Pole B,\n'
        '  "modality": "visual" | "acoustic" | "audiovisual" | "typography",\n'
        '  "visual_signature": specific visual/lighting/camera/material traits shifted,\n'
        '  "acoustic_signature": specific foley/room-tone/vocal traits shifted (or "none"),\n'
        '  "confounds_or_notes": any unintended correlations across the pairs (or "clean")',
        assistant_prefix="{\n",
    )
    raw = "{\n" + generate_cached(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        prompt_text=prompt,
        max_new_tokens=450,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        rep_p=1.0,
        seed=seed,
    )
    parsed = try_parse_json(raw)
    return parsed if parsed is not None else raw


def try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def write_markdown_report(
    results: dict[str, Any],
    vectors_map: dict[str, torch.Tensor],
    out_md: Path,
) -> None:
    lines: list[str] = [
        "# H3 Control Vector Batch-Labelling & `layers.49` Readout Report",
        "",
        "Generated with `Qwen3-VL-32B-Instruct` (`5.37 bpw` EXL3, `6 bpw` readout layers `50..63`, `8 bpw` `lm_head`) using:",
        "- **Official Qwen3-VL Non-Thinking Sampling + Coupled Gumbel Draws** (`ComboSampler(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, rep_p=1.05)` with matched per-step `rand_u32` across $\\alpha$)",
        "- **User-Scene + Assistant-Span `layers.49` Steering** (leaving `<|im_start|>system`, formatting instructions, and `<think></think>` framing tokens unsteered)",
        "- **Direct `layers.49 -> lm_head` Symmetric Contrastive Log-Odds (`0.5 * (logP(+v) - logP(-v))`)** across 6 domain-neutral prefixes (with true BPE token verification)",
        "- **Blind Self-Report Synthesis** (`synthesis` judges the coupled `Anchor A` & `Anchor B` rollouts 100% blind to `lm_head` lexical tokens)",
        "- **Dual-Site (`layers.49` + `post_refiner`) Pair-Coherence Diagnostics** (`mean_pairwise_cosine`, `split_half_cosine`, `min_pair_cosine`, and `per_pair_cosine`)",
        "",
        "## 1. Summary Table",
        "",
        "| Vector (`.toml`) | `L49` Norm | `PR` Norm (`fl2va`) | `L49` Split1/2 | `PR` Split1/2 | `PR` MinCos (Weakest) | Lexical Clarity | Blind Rollout Label (`temp=0.7`) | Rollout Clarity | Top `+v_encoder` Words (`lm_head` $\\Delta\\log P$) |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]

    for stem, entry in sorted(results.items()):
        sr = entry.get("layer49_self_report", {})
        sr_synth = sr.get("synthesis", {}) if isinstance(sr.get("synthesis"), dict) else {}
        sr_slug = sr_synth.get("self_report_slug", "—")
        sr_clar = sr_synth.get("readout_clarity", "—")
        lex = sr.get("lexical_logit_shifts", {})
        lex_clar = lex.get("lexical_clarity", "—")
        pos_words = ", ".join(lex.get("positive_boosted_words", [])[:6])

        l49_q = entry.get("layer49_quality", {})
        l49_sh = f"`{l49_q['split_half_cosine']:.3f}`" if "split_half_cosine" in l49_q else "—"

        pr_sf = entry.get("post_refiner_safetensors", {})
        fl2va_info = pr_sf.get("fl2va") or next(iter(pr_sf.values()), {})
        pr_norm = f"`{fl2va_info['post_refiner_raw_norm']:.1f}`" if "post_refiner_raw_norm" in fl2va_info else "—"
        pr_q = fl2va_info.get("post_refiner_quality", {})
        pr_sh = f"`{pr_q['split_half_cosine']:.3f}`" if "split_half_cosine" in pr_q else "—"
        if "min_pair_cosine" in pr_q:
            wp = pr_q.get("weakest_pair_1based")
            wp_s = f"P{int(wp):02d}" if isinstance(wp, int) else "P?"
            pr_min = f"`{pr_q['min_pair_cosine']:.3f}` (`{wp_s}`)"
        elif "min_pair_cosine" in l49_q:
            wp = l49_q.get("weakest_pair_1based")
            wp_s = f"P{int(wp):02d}" if isinstance(wp, int) else "P?"
            pr_min = f"`{l49_q['min_pair_cosine']:.3f}` (`{wp_s}`)"
        else:
            pr_min = "—"

        lines.append(
            f"| `{stem}` | `{entry.get('layer49_raw_norm', 0.0):.2f}` | {pr_norm} | {l49_sh} | {pr_sh} | {pr_min} | `{lex_clar}` | `{sr_slug}` | `{sr_clar}` | {pos_words} |"
        )

    if len(vectors_map) >= 2:
        names = sorted(vectors_map.keys())
        mat = torch.stack([vectors_map[n] for n in names], dim=0)
        cos_mat = mat @ mat.T
        lines += [
            "",
            "## 2. Pairwise Cosine Similarity at `layers.49` (`vector.encoder` `[5120]`)",
            "",
            "| Vector | " + " | ".join(f"`{n[:10]}`" for n in names) + " |",
            "|---|" + "|".join("---:" for _ in names) + "|",
        ]
        for i, n_i in enumerate(names):
            row_str = " | ".join(f"`{cos_mat[i, j].item():+.2f}`" for j in range(len(names)))
            lines.append(f"| `{n_i}` | {row_str} |")

    lines += ["", "## 3. Per-Vector Diagnostic Cards", ""]
    for stem, entry in sorted(results.items()):
        gt = entry.get("ground_truth_header", {})
        bp = entry.get("blind_pair_label", {})
        sr = entry.get("layer49_self_report", {})
        sr_synth = sr.get("synthesis", {}) if isinstance(sr.get("synthesis"), dict) else {}
        lex = sr.get("lexical_logit_shifts", {})
        sweeps = sr.get("sweep_completions", {})
        l49_q = entry.get("layer49_quality", {})
        pr_sf = entry.get("post_refiner_safetensors", {})
        fl2va_info = pr_sf.get("fl2va") or next(iter(pr_sf.values()), {})
        pr_q = fl2va_info.get("post_refiner_quality", {})

        lines += [
            f"### `{stem}` (`L49 raw_norm = {entry.get('layer49_raw_norm', 0.0):.2f}`, `median_row_norm = {entry.get('layer49_median_row_norm_ex_row0', 0.0):.1f}`)",
            f"- **Intended (`.toml` header)**: {gt.get('quality', '')}",
            f"- **Blind Pair-Diff Label**: `{bp.get('inferred_slug', '') if isinstance(bp, dict) else ''}` — {bp.get('inferred_quality', '') if isinstance(bp, dict) else ''}",
            f"- **Blind `layers.49` Rollout Self-Report (`temp=0.7`, clarity=`{sr_synth.get('readout_clarity', '—')}`)**: `{sr_synth.get('self_report_slug', '')}` — {sr_synth.get('self_report_quality', '')}",
        ]
        if l49_q:
            lines.append(
                f"- **`layers.49` Pair Coherence (`[5120]`)**: `MeanPairCos = {l49_q.get('mean_pairwise_cosine', 0.0):.4f}`, "
                f"`SplitHalf = {l49_q.get('split_half_cosine', 0.0):.4f}`, "
                f"`MinPairCos = {l49_q.get('min_pair_cosine', 0.0):.4f}` (Pair `{l49_q.get('weakest_pair_1based', '?')}`)"
            )
        if pr_q:
            lines.append(
                f"- **`post_refiner` Pair Coherence (`[5376]`, `raw_norm = {fl2va_info.get('post_refiner_raw_norm', 0.0):.2f}`)**: "
                f"`MeanPairCos = {pr_q.get('mean_pairwise_cosine', 0.0):.4f}`, "
                f"`SplitHalf = {pr_q.get('split_half_cosine', 0.0):.4f}`, "
                f"`MinPairCos = {pr_q.get('min_pair_cosine', 0.0):.4f}` (Pair `{pr_q.get('weakest_pair_1based', '?')}`)"
            )
        lines += [
            f"- **Top Boosted Words (`+v_encoder`, lexical_clarity=`{lex.get('lexical_clarity', '—')}`, top5_mean=`{lex.get('top5_mean_delta_log_prob', 0.0):+.2f}`)**: {', '.join(lex.get('positive_boosted_words', [])[:10])}",
            f"- **Top Boosted Words (`-v_encoder`)**: {', '.join(lex.get('negative_boosted_words', [])[:10])}",
        ]
        if isinstance(sweeps, dict):
            for anchor_label, anchor_key in (
                ("Anchor A (Person + Counter + Mug + Room)", "anchor_human_action_room"),
                ("Anchor B (Storefront + Awning + Van + Street)", "anchor_surface_street_camera"),
            ):
                if anchor_key in sweeps:
                    lines.append(f"- **Coupled `temp=0.7` Rollouts on {anchor_label}**:")
                    for a_k, a_v in sweeps[anchor_key].items():
                        one_line = " ".join(a_v.split())
                        lines.append(f"  - `alpha={a_k}`: {one_line}")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-label H3 control vectors with ExLlamaV3 Qwen3-VL-32B-Instruct")
    parser.add_argument(
        "-m",
        "--model_dir",
        type=Path,
        default=_default_model_dir(_H3_ROOT),
        help="Path to compiled ExLlamaV3 model directory",
    )
    parser.add_argument(
        "--toml_dir",
        type=Path,
        default=_H3_ROOT / "prompt_bank" / "control_vectors",
        help="Directory containing control-vector .toml files",
    )
    parser.add_argument(
        "--vectors_dir",
        type=Path,
        default=_H3_ROOT.parents[1] / "models" / "control_vectors",
        help="Directory containing compiled control-vector .safetensors files",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Optional comma-separated substring filter for vector names (e.g. 'noir,film_16mm,reverb')",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "self_report", "blind_pairs", "vlm"],
        default="all",
        help="Labelling mode to run",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["scene_text", "from_first_diff"],
        default="scene_text",
        help="Pooling mode for `layers.49` `vector.encoder` extraction (`scene_text` or `from_first_diff`)",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-0.15,0.00,0.15,0.22",
        help="Comma-separated alpha strengths (in median_row_norm units) for layer-49 self-report steering",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Coupled Gumbel rollouts (Qwen3-VL non-thinking default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p nucleus sampling threshold (Qwen3-VL non-thinking default: 0.8)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling threshold (Qwen3-VL non-thinking default: 20)",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Min-p sampling threshold (Qwen3-VL non-thinking default: 0.0)",
    )
    parser.add_argument(
        "--rep_p",
        type=float,
        default=1.05,
        help="Repetition penalty (default: 1.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Base seed for Coupled Gumbel sampling across alphas",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=None,
        help="Image / contact-sheet paths when running `--mode vlm`",
    )
    parser.add_argument(
        "-o",
        "--out_json",
        type=Path,
        default=Path(__file__).resolve().parent / "h3_control_vector_labels.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--out_md",
        type=Path,
        default=Path(__file__).resolve().parent / "h3_control_vector_report.md",
        help="Output Markdown diagnostic report path",
    )
    parser.add_argument("--cache_tokens", type=int, default=4096)
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    filters = [f.strip() for f in args.filter.split(",") if f.strip()]

    t0 = time.time()
    print(f"Loading ExLlamaV3 config from {args.model_dir} ...")
    config = Config.from_directory(str(args.model_dir))
    tokenizer = Tokenizer.from_config(config)

    vision_model = None
    if args.mode == "vlm" or args.images:
        vision_model = Model.from_config(config, component="vision")
        vision_model.load(progressbar=True)

    model = Model.from_config(config)
    cache = Cache(
        model,
        max_num_tokens=args.cache_tokens,
        layer_type=CacheLayer_quant,
        k_bits=8,
        v_bits=8,
    )
    model.load(progressbar=True)
    print(f"Loaded model + Q8 KV cache in {time.time() - t0:.2f}s")

    if args.mode == "vlm" and args.images:
        from PIL import Image

        assert vision_model is not None
        img_embs = [
            vision_model.get_image_embeddings(tokenizer=tokenizer, image=Image.open(p))
            for p in args.images
        ]
        placeholders = "\n".join(ie.text_alias for ie in img_embs) + "\n"
        vlm_prompt = format_chatml(
            "You analyze control-vector gate-ladder frames and contact sheets for a video diffusion model. Output only valid JSON.",
            placeholders
            + "Compare the progression across the provided image(s)/frames and return a JSON object with keys: "
            '"inferred_slug", "visual_progression", "artifacts_or_saturation", "recommended_strength_notes".',
            assistant_prefix="{\n",
        )
        out = "{\n" + generate_cached(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            prompt_text=vlm_prompt,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.8,
            top_k=20,
            rep_p=1.0,
            seed=args.seed,
            embeddings=img_embs,
        )
        print(out)
        return

    all_toml_files = sorted(args.toml_dir.glob("*.toml"))
    if filters:
        toml_files = [p for p in all_toml_files if any(f in p.stem for f in filters)]
    else:
        toml_files = all_toml_files
    active_stems = {p.stem for p in toml_files}

    results: dict[str, Any] = {}
    if (filters or args.mode != "all") and args.out_json.is_file():
        try:
            results = json.loads(args.out_json.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    vectors_map: dict[str, torch.Tensor] = {}
    for toml_path in all_toml_files:
        stem = toml_path.stem
        t_vec = time.time()
        toml_text = toml_path.read_text(encoding="utf-8")
        toml_sha256 = hashlib.sha256(toml_text.encode("utf-8")).hexdigest()[:16]
        toml_data = tomllib.loads(toml_text)
        vec_hdr = toml_data.get("vector", {})

        sf_matches = sorted(args.vectors_dir.glob(f"{stem}__*.safetensors"))
        vec_enc = None
        med_norm = None
        pr_safetensors_meta: dict[str, Any] = {}
        for sf_path in sf_matches:
            v_cand, m_cand, sf_meta = load_encoder_vector_from_safetensors(sf_path)
            if v_cand is not None and vec_enc is None:
                vec_enc, med_norm = v_cand, m_cand
            fam = str(sf_meta.get("family", sf_path.stem.split("__")[1] if "__" in sf_path.stem else "default"))
            pts = sf_meta.get("points", {}) if isinstance(sf_meta.get("points"), dict) else {}
            qr = sf_meta.get("quality_report", {}) if isinstance(sf_meta.get("quality_report"), dict) else {}
            pr_pt = pts.get("post_refiner", {}) if isinstance(pts.get("post_refiner"), dict) else {}
            pr_q_raw = qr.get("post_refiner") if isinstance(qr.get("post_refiner"), dict) else (qr if "per_pair_cosine" in qr else {})
            if pr_pt or pr_q_raw:
                pr_ppc = [round(float(x), 4) for x in pr_q_raw.get("per_pair_cosine", [])]
                weakest_idx = (
                    int(min(range(len(pr_ppc)), key=lambda i: pr_ppc[i]) + 1)
                    if pr_ppc
                    else None
                )
                min_p_cos = pr_ppc[weakest_idx - 1] if weakest_idx is not None else None
                pr_safetensors_meta[fam] = {
                    "file": sf_path.name,
                    "post_refiner_raw_norm": round(float(pr_pt["raw_norm"]), 4) if "raw_norm" in pr_pt else None,
                    "post_refiner_median_row_norm": round(
                        float(pr_pt.get("median_row_norm") or pr_pt.get("median_row_norm_ex_row0") or 0.0), 4
                    ),
                    "post_refiner_quality": {
                        "mean_pairwise_cosine": round(float(pr_q_raw.get("mean_pairwise_cosine", 0.0)), 4),
                        "split_half_cosine": round(float(pr_q_raw.get("split_half_cosine", 0.0)), 4),
                        **({"min_pair_cosine": min_p_cos, "weakest_pair_1based": weakest_idx} if min_p_cos is not None else {}),
                        "per_pair_cosine": pr_ppc,
                    },
                }

        computed_vec, computed_med_norm, raw_norm, diff_records, l49_quality = compute_encoder_vector_from_toml(
            model=model,
            tokenizer=tokenizer,
            toml_path=toml_path,
            pooling=args.pooling,
        )
        sf_exl3_cos: float | None = None
        if vec_enc is not None:
            sf_exl3_cos = round(float(torch.dot(vec_enc.float(), computed_vec.float()).item()), 6)
        if args.pooling != "scene_text":
            vec_enc = computed_vec
            med_norm = computed_med_norm
            vec_source = f"computed_exl3_layer49 ({args.pooling})"
        elif vec_enc is None or med_norm is None:
            vec_enc = computed_vec
            med_norm = computed_med_norm
            vec_source = "computed_exl3_layer49"
        elif sf_exl3_cos is not None and sf_exl3_cos < 0.985:
            if stem in active_stems:
                print(
                    f"  [notice] [{stem}] safetensors vector.encoder disagrees with current .toml "
                    f"(cos={sf_exl3_cos:+.4f} < 0.985; stale .safetensors); using computed_exl3_layer49"
                )
            vec_enc = computed_vec
            med_norm = computed_med_norm
            vec_source = f"computed_exl3_layer49 (stale_safetensors_cos={sf_exl3_cos:+.4f})"
        else:
            vec_source = "safetensors_vector.encoder"
        vectors_map[stem] = vec_enc

        entry: dict[str, Any] = results.get(stem, {})
        toml_changed = entry.get("toml_sha256") not in (None, toml_sha256)
        entry.update({
            "toml_file": toml_path.name,
            "toml_sha256": toml_sha256,
            "pooling": args.pooling,
            "safetensors_files": [p.name for p in sf_matches],
            "safetensors_vs_exl3_toml_cos": sf_exl3_cos,
            "ground_truth_header": {
                "name": vec_hdr.get("name", stem),
                "quality": vec_hdr.get("quality", ""),
                "a_side": vec_hdr.get("a_side", ""),
                "b_side": vec_hdr.get("b_side", ""),
            },
            "encoder_vector_source": vec_source,
            "layer49_raw_norm": round(raw_norm, 4),
            "layer49_median_row_norm_ex_row0": round(float(med_norm), 4),
            "layer49_quality": l49_quality,
            "post_refiner_safetensors": pr_safetensors_meta,
        })
        results[stem] = entry

        if stem not in active_stems:
            continue

        print(f"\n=== [{stem}] (safetensors files: {len(sf_matches)}, pooling={args.pooling}) ===")

        if (
            args.mode in ("all", "blind_pairs")
            or toml_changed
            or not isinstance(entry.get("blind_pair_label"), dict)
        ):
            entry["blind_pair_label"] = run_blind_pair_synthesis(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                diff_records=diff_records,
                seed=args.seed + 200,
            )
            print(f"  [blind_pairs] -> {json.dumps(entry['blind_pair_label'], ensure_ascii=False)}")

        if args.mode in ("all", "self_report"):
            entry["layer49_self_report"] = run_self_report_probe(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                vec_encoder=vec_enc,
                median_row_norm=float(med_norm),
                alphas=alphas,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                rep_p=args.rep_p,
                seed=args.seed,
            )
            lex_pos = ", ".join(entry["layer49_self_report"]["lexical_logit_shifts"]["positive_boosted_words"][:6])
            print(f"  [lm_head +words] -> {lex_pos}")
            print(f"  [self_report] -> {json.dumps(entry['layer49_self_report']['synthesis'], ensure_ascii=False)}")

        entry["elapsed_sec"] = round(time.time() - t_vec, 2)
        results[stem] = entry

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_markdown_report(results, vectors_map, args.out_md)

    write_markdown_report(results, vectors_map, args.out_md)
    print(f"\nSaved batch labels for {len(results)} control vector(s) to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
