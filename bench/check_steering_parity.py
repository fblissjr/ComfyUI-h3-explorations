#!/usr/bin/env python3
"""Fast CPU parity gate across tracked control-vector TOMLs, retired packs, active bank, and lexicon.

What it checks (always, < 1 s on CPU, zero CUDA):
1. Every active `prompt_bank/control_vectors/*.toml` parses cleanly via `control_vectors.load_pairs`.
2. Zero stem collisions between active `prompt_bank/control_vectors/*.toml` and `_retired/*.toml`.
3. Every stem in `NON_H3_IMAGE_STEMS` exists in active `prompt_bank/control_vectors/`.

When `<comfy>/models/control_vectors/bank/` is present:
4. 1:1 parity between active tracked TOMLs and built `fl2va` / `ref2va` bank stems.
5. Zero `_retired/` stems remain present in active `bank/`.
6. Every active stem outside `NON_H3_IMAGE_STEMS` has a built `h3_image` vector in `bank/`.
7. No active bank stem has a stranded/duplicate `.toml` in `data/control_vectors/`
   (unbuilt WIP drafts saved in `data/control_vectors/` by `/lab` are allowed).

When `model2vec_h3_lexicon.safetensors` is present:
8. `j_lens.load_model2vec_lexicon()` returns non-empty `terms` and `categories`, and
   `>= 90%` of active tracked stems match at least one lexicon entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _lib import bootstrap

bootstrap(cpu=True)

import control_vectors as cv  # noqa: E402
import folder_paths  # noqa: E402
import j_lens  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_BANK_CV_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"
RETIRED_CV_DIR = PROMPT_BANK_CV_DIR / "_retired"
DATA_CV_DIR = REPO_ROOT / "data" / "control_vectors"

NON_H3_IMAGE_STEMS: frozenset[str] = frozenset(
    {
        "auctioneer_rapid_patter",
        "handheld_documentary_zoom",
        "operatic_bellow",
        "pyro_klaxon_audio",
        "reverb_cathedral",
        "vocal_strain",
        "whisper_asmr",
    }
)

LEXICON_COVERAGE_FLOOR: float = 0.90
_BANK_FILE_RE = re.compile(r"^([a-z0-9_]+)__(fl2va|ref2va|h3_image|fasth3)__.+\.safetensors$")


def _stem_matches_lexicon(stem: str, term_to_idx: dict[str, int]) -> bool:
    q_tokens = [tok for tok in stem.split("_") if len(tok) > 1]
    if not q_tokens:
        return False
    if " ".join(q_tokens) in term_to_idx:
        return True
    if any(tok in term_to_idx for tok in q_tokens):
        return True
    for i in range(len(q_tokens) - 1):
        if f"{q_tokens[i]} {q_tokens[i + 1]}" in term_to_idx:
            return True
    for term in term_to_idx:
        t_words = set(re.split(r"[^a-z0-9]+", term))
        if any(len(qt) >= 3 and qt in t_words for qt in q_tokens):
            return True
    return False


def _bank_dir() -> Path:
    models_root = Path(getattr(folder_paths, "models_dir", REPO_ROOT.parent.parent / "models"))
    return models_root / "control_vectors" / "bank"


def main() -> int:
    errs: list[str] = []

    if not PROMPT_BANK_CV_DIR.is_dir():
        print(f"FAIL  missing {PROMPT_BANK_CV_DIR.relative_to(REPO_ROOT)}")
        return 1

    active_tomls = sorted(PROMPT_BANK_CV_DIR.glob("*.toml"))
    if not active_tomls:
        errs.append("prompt_bank/control_vectors/ contains zero active .toml packs")
    active_stems: set[str] = {p.stem for p in active_tomls}

    for p in active_tomls:
        try:
            ps = cv.load_pairs(p)
            if not ps.pairs:
                errs.append(f"{p.relative_to(REPO_ROOT)} parsed with zero [[pair]] entries")
        except Exception as exc:
            errs.append(f"{p.relative_to(REPO_ROOT)} failed load_pairs: {exc}")

    retired_stems: set[str] = (
        {p.stem for p in RETIRED_CV_DIR.glob("*.toml")} if RETIRED_CV_DIR.is_dir() else set()
    )
    overlap = sorted(active_stems & retired_stems)
    if overlap:
        errs.append(f"stems present in both prompt_bank/control_vectors/ and _retired/: {overlap}")

    stale_allowlist = sorted(NON_H3_IMAGE_STEMS - active_stems)
    if stale_allowlist:
        errs.append(f"NON_H3_IMAGE_STEMS contains stems missing from active prompt_bank: {stale_allowlist}")

    print(
        f"  ok    repo_tomls         {len(active_stems)} active TOML(s), "
        f"{len(retired_stems)} retired, 0 collisions"
        if not errs
        else f"  FAIL  repo_tomls         {len(errs)} error(s)"
    )

    # Bank parity checks (when active bank is present on disk)
    bdir = _bank_dir()
    bank_files = sorted(bdir.glob("*.safetensors")) if bdir.is_dir() else []
    if bank_files:
        by_family: dict[str, set[str]] = {"fl2va": set(), "ref2va": set(), "h3_image": set()}
        all_bank_stems: set[str] = set()
        for bf in bank_files:
            m = _BANK_FILE_RE.match(bf.name)
            if not m:
                continue
            stem, fam = m.group(1), m.group(2)
            all_bank_stems.add(stem)
            if fam in by_family:
                by_family[fam].add(stem)

        bank_errs: list[str] = []
        for fam in ("fl2va", "ref2va"):
            missing_in_bank = sorted(active_stems - by_family[fam])
            extra_in_bank = sorted(by_family[fam] - active_stems)
            if missing_in_bank:
                bank_errs.append(f"bank missing {fam} build(s) for active TOMLs: {missing_in_bank}")
            if extra_in_bank:
                bank_errs.append(f"bank has untracked {fam} stem(s) missing from prompt_bank/: {extra_in_bank}")

        want_h3_img = active_stems - NON_H3_IMAGE_STEMS
        missing_img = sorted(want_h3_img - by_family["h3_image"])
        if missing_img:
            bank_errs.append(f"bank missing h3_image build(s) for visual stems: {missing_img}")

        undead = sorted(all_bank_stems & retired_stems)
        if undead:
            bank_errs.append(f"retired stem(s) still present in active bank/: {undead}")

        if DATA_CV_DIR.is_dir():
            data_stems = {p.stem for p in DATA_CV_DIR.glob("*.toml")}
            stranded = sorted(data_stems & all_bank_stems)
            if stranded:
                bank_errs.append(
                    f"active bank stem(s) still have stranded/duplicate TOMLs in data/control_vectors/: {stranded}"
                )

        if bank_errs:
            errs.extend(bank_errs)
            print(f"  FAIL  bank_parity        {len(bank_errs)} error(s)")
        else:
            print(
                f"  ok    bank_parity        {len(by_family['fl2va'])} fl2va, "
                f"{len(by_family['ref2va'])} ref2va, {len(by_family['h3_image'])} h3_image (0 retired/stranded)"
            )
    else:
        print("  skip  bank_parity        active bank directory not populated")

    # Lexicon coverage check (when model2vec_h3_lexicon.safetensors is present)
    lex_path = j_lens.default_model2vec_lexicon_path()
    if lex_path.is_file():
        lex = j_lens.load_model2vec_lexicon(lex_path)
        lex_errs: list[str] = []
        if not lex or not lex.get("terms"):
            lex_errs.append("model2vec_h3_lexicon.safetensors loaded empty or invalid")
        else:
            if not lex.get("categories"):
                lex_errs.append("model2vec_h3_lexicon.safetensors loaded with empty categories list")
            t2i = lex.get("term_to_idx") or {}
            matched = [s for s in sorted(active_stems) if _stem_matches_lexicon(s, t2i)]
            cov = (len(matched) / len(active_stems)) if active_stems else 1.0
            if cov < LEXICON_COVERAGE_FLOOR:
                missing_lex = sorted(active_stems - set(matched))
                lex_errs.append(
                    f"lexicon stem coverage {len(matched)}/{len(active_stems)} ({cov:.1%}) "
                    f"< floor {LEXICON_COVERAGE_FLOOR:.0%}; unmatched={missing_lex}"
                )
            else:
                print(
                    f"  ok    lexicon_coverage   {len(matched)}/{len(active_stems)} active stems matched "
                    f"({cov:.1%}, {len(lex['terms'])} terms, {len(set(lex['categories']))} categories)"
                )
        if lex_errs:
            errs.extend(lex_errs)
            print(f"  FAIL  lexicon_coverage   {len(lex_errs)} error(s)")
    else:
        print("  skip  lexicon_coverage   model2vec_h3_lexicon.safetensors not present")

    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
        return 1
    print("\nok: steering parity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
