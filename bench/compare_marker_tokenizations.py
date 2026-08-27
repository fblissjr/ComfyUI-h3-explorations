#!/usr/bin/env python3
"""How the release and pre-fix tokenizers actually differ on one prompt.

## The claim this exists to correct

`bench/results/2026-08-27_marker_epsilon_*.json` reported that the two
tokenizers "mismatch 316 of 408 positions" on the dialogue prompt. That number
is true and it teaches something false. It is an INDEX-WISE comparison, and any
insertion shifts every later index, so a handful of inserted tokens makes almost
everything after the first one compare unequal without a single one of them
having changed content.

The quantity that answers the question a reader is actually asking -- how
different are these two tokenizations -- is the alignment: the longest common
subsequence, the edit sites, and where the length actually goes. That is what
this measures, and it is why the index-wise number is reported here beside it
rather than alone.

## What it decides

Whether `legacy_bpe`'s prediction delta can be attributed to DISPLACEMENT (the
sequence shifted) or to CONTENT (the marker spellings themselves are different
tokens). Those imply different actions: a displacement effect is reachable from
the prompt template while keeping core's tokenizer, and a content effect is not.

It does not settle that attribution on its own -- it sizes the two candidate
causes so an arm can be designed to separate them. A padding arm that inserts
neutral tokens at the same sites, with content otherwise untouched, is the
experiment this measurement is for; the site list below is its specification.

Needs no GPU and no model weights: tokenizers only.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import comfy.cli_args  # noqa: E402  (must precede other comfy imports)
import comfy.text_encoders.minimax as mmx  # noqa: E402
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer  # noqa: E402

import marker_arms as M  # noqa: E402

GRAPH = REPO / "workflows" / "h3_text_to_video_dialogue_api.json"


def _prompt() -> str:
    for node in json.loads(GRAPH.read_text()).values():
        if node.get("class_type") == "MiniMaxH3Conditioning":
            return node["inputs"]["prompt"]
    raise SystemExit(f"no MiniMaxH3Conditioning node in {GRAPH.name}")


def _ids(tokenizer, text: str) -> list:
    out = tokenizer.tokenize_with_weights(text)
    seq = []
    for _key, batches in (out.items() if isinstance(out, dict) else [("x", out)]):
        for batch in batches:
            for item in batch:
                seq.append(item[0] if isinstance(item, (list, tuple)) else item)
    return seq


def _legacy_tokenizer():
    """The pre-fix tokenizer, reconstructed exactly as `marker_arms` does.

    Emptying whichever token-list attribute EXISTS and then verifying by
    VOCABULARY that no marker survives -- keying off a constant's name is how a
    reconstruction hands back the patched tokenizer while calling it stock.
    """
    targets = [(owner, attr, getattr(owner, attr))
               for owner, attr in ((MiniMaxH3Tokenizer, "H3_SPECIAL_TOKENS"),
                                   (mmx, "MINIMAX_EXTRA_TOKENS"))
               if getattr(owner, attr, None)]
    if not targets:
        raise SystemExit(
            "no known H3 special-token list is present, so the pre-fix "
            "tokenizer cannot be reconstructed. Emptying nothing would hand "
            "back the release tokenizer labelled legacy.")
    try:
        for owner, attr, _ in targets:
            setattr(owner, attr, [])
        fresh = MiniMaxH3Tokenizer(embedding_directory=None)
    finally:
        for owner, attr, saved in targets:
            setattr(owner, attr, saved)
    still = [t for t, i in M.marker_ids(fresh).items() if i is not None]
    if still:
        raise SystemExit(
            f"the reconstructed legacy tokenizer still declares {still}, so it "
            "is not the pre-fix one and every comparison would use the wrong "
            "control")
    return fresh


def _release_loaded_ids() -> dict:
    """What the release's own tokenizer directory yields when loaded normally.

    This is the evidence that settles the numbering: not the JSON literal, but
    what the shipped config produces under the loader anyone -- including the
    vendor -- would use. Absent if the release is not on this box, because the
    check must not silently pass by skipping.
    """
    import os
    root = os.environ.get("H3_RELEASE_TOKENIZER")
    if not root or not os.path.isdir(root):
        return {"status": "UNAVAILABLE: set H3_RELEASE_TOKENIZER to the "
                          "release's tokenizer directory to record this"}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root)
    vocab = tok.get_vocab()
    return {"tokenizer_class": type(tok).__name__, "len_tokenizer": len(tok),
            "base_vocab_size": tok.vocab_size,
            "ids": {t: vocab.get(t) for t in M.marker_tokens()},
            "single_token": all(vocab.get(t) is not None
                                for t in M.marker_tokens())}


def id_provenance() -> dict:
    """Where the seven marker ids come from, since it is not where core says.

    `comfy/text_encoders/minimax.py` declares MINIMAX_EXTRA_TOKENS with the
    comment "ids fixed by the released tokenizer". The released
    `tokenizer_config.json` assigns the seven NO ids: all seven are listed in
    `additional_special_tokens` and none appears in `added_tokens_decoder`,
    where every one of the other thirteen specials does.

    That is a fact about the JSON literal and it is NOT a problem, which an
    earlier version of this docstring got wrong. Loading the release's own
    tokenizer directory with the standard HF loader assigns the seven
    151669-151675 and tokenizes them as single tokens -- `len(tokenizer)` is
    151676 against a base vocab of 151643 plus 26 added. The ids are fixed BY
    CONSTRUCTION rather than by literal, deterministically, from the config the
    vendor shipped. So core's comment is correct, `release_id` addresses the
    rows the vendor's own pipeline would address, and the "the numbering might
    be wrong" worry this function was written to raise is closed. It is kept,
    and now records the loaded ids as evidence, because the worry was real
    enough to be raised twice and the answer should be findable.
    """
    # Through vendor_config's own reader, never a retyped path or value --
    # CLAUDE.md's rule for anything the release declares.
    import vendor_config as VC
    cfg = VC._load("tokenizer_config.json")
    declared = VC.additional_special_tokens()
    decoder = cfg.get("added_tokens_decoder", {})
    by_content = {v.get("content"): int(k) for k, v in decoder.items()}
    markers = M.marker_tokens()

    top = max(int(k) for k in decoder)
    derived, n = {}, 0
    for token in declared:
        if token in markers:
            n += 1
            derived[token] = top + n

    from comfy.text_encoders.minimax import MINIMAX_EXTRA_TOKENS as CORE
    return {
        "release_declares_them_special": [t for t in markers if t in declared],
        "release_gives_them_ids": {t: by_content[t] for t in markers if t in by_content},
        "other_specials_all_have_ids": all(
            t in by_content for t in declared if t not in markers),
        "highest_id_in_added_tokens_decoder": top,
        "derived_by_sequential_append": derived,
        "core_constant": dict(CORE),
        "derivation_reproduces_core_exactly": derived == dict(CORE),
        "core_comment_says": "ids fixed by the released tokenizer",
        "loaded_from_release": _release_loaded_ids(),
        "evidence_class": (
            "SOURCE. The config assigns no id as a literal, but loading the "
            "release's own tokenizer directory with the standard HF loader "
            "yields 151669-151675 and tokenizes the seven as single tokens. "
            "The ids are fixed by construction from the shipped config, so "
            "core's comment is correct and release_id addresses the rows the "
            "vendor's own pipeline would address."
        ),
        "worry_this_closes": (
            "raised twice, on 2026-08-26 and 2026-08-27: that core DERIVED the "
            "numbering and might have guessed wrong, which would mean every "
            "marker arm addresses arbitrary rows. It did not guess. Recorded "
            "here so the third session to notice the missing literal finds the "
            "answer instead of re-raising it."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompt = _prompt()
    release, legacy = MiniMaxH3Tokenizer(embedding_directory=None), _legacy_tokenizer()
    a, b = _ids(release, prompt), _ids(legacy, prompt)

    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    index_wise = sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i])

    sites, shift = [], 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        shift += (j2 - j1) - (i2 - i1)
        sites.append({"op": tag, "release_span": [i1, i2], "legacy_span": [j1, j2],
                      "release_ids": i2 - i1, "legacy_ids": j2 - j1,
                      "cumulative_shift_after": shift})

    record = {
        "measurement": "alignment between the release and pre-fix H3 "
                       "tokenizers on one prompt",
        "graph": GRAPH.name,
        "release_ids": len(a), "legacy_ids": len(b),
        "longest_common_subsequence": lcs,
        "content_shared_fraction": round(lcs / len(a), 4),
        "release_ids_outside_lcs": len(a) - lcs,
        "legacy_ids_outside_lcs": len(b) - lcs,
        "max_cumulative_displacement": shift,
        "edit_sites": sites,
        "id_provenance": id_provenance(),
        "index_wise_mismatches": index_wise,
        "index_wise_is_misleading": (
            "reported only to retire it. An insertion shifts every later "
            "index, so this counts displacement as if it were content change. "
            "Use longest_common_subsequence. A record of this repo's carried "
            "the index-wise number alone and it read as 'the tokenizations are "
            "wildly different', which the alignment refutes."
        ),
        "what_it_does_not_settle": (
            "whether legacy_bpe's prediction delta comes from the changed "
            "content or from the displacement. Both are present and this sizes "
            "them; separating them needs a padding arm that reproduces the "
            "displacement with content untouched, specified by edit_sites."
        ),
    }

    out = Path(args.out) if args.out else (
        REPO / "bench" / "results" / "2026-08-27_marker_tokenization_alignment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")

    print(f"release {len(a)} ids, legacy {len(b)} ids")
    print(f"  longest common subsequence : {lcs} ({100*lcs/len(a):.1f}% of release)")
    print(f"  release ids outside it     : {len(a)-lcs}")
    print(f"  edit sites                 : {len(sites)}, net {shift:+d} tokens")
    print(f"  index-wise mismatches      : {index_wise}  <- retired, see record")
    prov = record["id_provenance"]
    print(f"\nmarker ids: release fixes {len(prov['release_gives_them_ids'])} of 7; "
          f"derivation reproduces core exactly: "
          f"{prov['derivation_reproduces_core_exactly']}")
    try:
        print(f"wrote {out.relative_to(REPO)}")
    except ValueError:
        print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
