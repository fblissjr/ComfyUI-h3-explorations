#!/usr/bin/env python3
"""What ComfyUI's tokenizer does to the seven markers the release declares, scene by scene.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Loads the H3 text
encoder for its tokenizer only -- **no encoder forward, no sampler, no VAE, no
GPU work**. `bench/grade_h3_marker_tokens.py` is the encoder-level companion and
owns the hidden-state deltas; this file owns the token sequences underneath them
and says nothing about states.

**The question.** `vendor_tokens.py` establishes that ComfyUI's bundled
tokenizer declares thirteen `additional_special_tokens` where the release
declares twenty, so `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`,
`<|lyrics_end|>`, `<|caption_start|>` and `<|caption_end|>` tokenize as ordinary
text. That much was known. What was never shown is **what the damage looks like
across the prompt shapes people actually write**, which is what a reader needs
to decide whether it matters to them.

**The controls, and the run is void without them.**

1. *Marker-free prompt.* Stock, patched and release tokenizers must produce
   identical ids. If they do not, this harness is measuring something other
   than the markers.
2. *Patched equals the release.* On every text-path scene the patched
   tokenizer's ids must equal the release tokenizer's, exactly. **This is the
   load-bearing one**: it is what makes "patched" mean "what the model authors
   emit" rather than "different from stock". A scene where they disagree is
   reported as a failure of the fix, not as a finding about stock.

Both are asserted per scene and the exit code follows control 2.

**Why `contaminated_neighbours` is the number worth reading.** The obvious cost
of a missing special token is that one token becomes several. The expensive cost
is that BPE has no reason to stop at the marker: the fragments merge with the
text on either side, so tokens that are not part of the marker at all come out
different too. That is measured here by aligning the two id sequences and
counting positions that differ while lying outside any marker span. A reader who
believes the defect is localised to the marker should look at that column.

**Scene coverage is the point.** The seven markers are not interchangeable --
they sit in different prompt shapes (dialogue, lyrics, captions, a hard cut) and
the guides mandate them in different places. Each scene below names the shape it
represents and why a user would hit it. Two are deliberate stressors rather than
prompts anyone would write, and are labelled as such.

**What this cannot answer.** Whether the DiT reads any of it. The encoder-level
question is `grade_h3_marker_tokens.py`'s and the embedding rows behind these
ids are untrained (`bench/audit_h3_token_embeddings.py`). This file establishes
only what reaches the encoder, which is the part that is unambiguous.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# insert(0): `comfy_extras.nodes_minimax_h3` does a bare `import nodes`, and
# this repo's own nodes.py would win from position 0. Same trap preflight
# documents.
_COMFY = Path.home() / "ComfyUI"
sys.path.insert(0, str(_COMFY))
sys.path.insert(0, str(_REPO.parent))
# The package directory carries hyphens, so it is not an identifier and a plain
# `import` cannot name it. `importlib.import_module` resolves it by string, the
# same idiom `bench/check_schema_defaults.py` uses for the same reason.
_PKG = _REPO.name

RELEASE_TOKENIZER = _REPO / "coderef" / "MiniMax-H3" / "tokenizer"

# Same checkpoint as `grade_h3_marker_tokens.py`, so the two files' scenes stay
# comparable. Nothing here reads the weights -- only the tokenizer that rides
# with them -- but keeping one constant means one thing to change.
ENCODER = (Path.home() / "ComfyUI" / "models" / "text_encoders"
           / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")

VISION_START = 151652
VISION_END = 151653

# The seven ComfyUI's bundled tokenizer does not declare. Read from the vendored
# release config rather than retyped: `vendor_config/` is the authority and
# CLAUDE.md forbids a second copy. The thirteen ComfyUI already has are
# subtracted at runtime by checking the stock vocabulary, so this list cannot
# drift out of agreement with what the fix actually adds.
def _missing_markers() -> list[str]:
    import importlib
    vendor_config = importlib.import_module(f"{_PKG}.vendor_config")
    bundled = _bundled_declared()
    return [t for t in vendor_config.additional_special_tokens()
            if t not in bundled]


# --------------------------------------------------------------------------
# Scenes. Each is (key, why_it_matters, text). `ref_images` marks the ones that
# go through the reference presentation instead of the plain text path.
# --------------------------------------------------------------------------

CONTROL = ("A medium shot establishes the room, then the camera trucks right "
           "with small amplitude at slow speed.")

SCENES = [
    dict(
        key="two_hander_argument",
        shape="t2v dialogue, ordinary density",
        why="The commonest way a user meets <d>: two people talking. This is "
            "the baseline the denser scenes should be read against.",
        text=(
            "Live-action, cinematic, handheld on 35mm. A woman in a charcoal "
            "coat faces a man on a concrete stairwell landing, lit hard from a "
            "caged bulb overhead. She (S1) says, <d>[English] You said "
            "tomorrow.</d> He (S2) answers, <d>[English] It moved.</d> She "
            "says, <d>[English] Moved to when?</d> He looks away and adjusts "
            "the satchel strap on his shoulder."
        ),
    ),
    dict(
        key="staccato_exchange",
        shape="t2v dialogue, maximum density an on-format prompt can reach",
        why="Short alternating lines put a marker pair every few words. If the "
            "per-marker cost compounds, this is where it shows without the "
            "prompt stopping being something a person would write.",
        text=(
            "Live-action, close two-shot in a stairwell, handheld. "
            "She (S1) says, <d>[English] Go.</d> He (S2) says, <d>[English] "
            "Not yet.</d> She says, <d>[English] Now.</d> He says, <d>[English] "
            "Why?</d> She says, <d>[English] Move.</d> He says, <d>[English] "
            "Fine.</d> They descend the stairs out of frame."
        ),
    ),
    dict(
        key="musical_number",
        shape="lyrics markers around sung dialogue",
        why="<|lyrics_start|> and <|lyrics_end|> exist for this and nothing "
            "else. No prompt in this repo has ever used them, so their "
            "behaviour under ComfyUI was entirely unmeasured before this run.",
        text=(
            "Live-action musical number on a rain-slick street at night, "
            "neon signage, a slow dolly-in on a singer at a doorway. "
            "<|lyrics_start|><d>[English] I walked the long way home again.</d>"
            "<d>[English] The lights were out on Seventh Street.</d>"
            "<d>[English] I told myself I did not mind.</d><|lyrics_end|> "
            "She steps off the kerb and the camera holds on the empty doorway."
        ),
    ),
    dict(
        key="captioned_documentary",
        shape="caption markers around on-screen text",
        why="<|caption_start|> / <|caption_end|> are the release's markers for "
            "burned-in text. Also unmeasured before this run. A documentary or "
            "subtitled scene is the shape that reaches for them.",
        text=(
            "Documentary footage, 16mm grain, a fixed tripod shot of a fishing "
            "boat unloading at dawn. <|caption_start|>Port of Vigo, 1974"
            "<|caption_end|> A man in oilskins hauls a crate across the deck "
            "while gulls circle the mast. He (S1) says, <d>[Spanish] The catch "
            "is smaller every year.</d> The camera holds as the crate is "
            "swung ashore."
        ),
    ),
    dict(
        key="hard_cut",
        shape="the cutoff marker",
        why="<|cutoff|> is the seventh and the least documented. Included so "
            "the audit covers all seven rather than the six that appear in "
            "prompt guidance.",
        text=(
            "Live-action, a static wide of a kitchen at night. A kettle begins "
            "to whistle on the hob and steam rises past the window."
            "<|cutoff|> A close-up of the same kettle, whistle cut to silence, "
            "the burner ring going dark."
        ),
    ),
    dict(
        key="multilingual_dialogue",
        shape="language tags inside the dialogue markers",
        why="The guide's [Language] tag sits immediately after <d>, so the "
            "marker's fragments land against a bracket rather than a letter. "
            "Whether the damage differs by language tag is not obvious and "
            "nothing has looked.",
        text=(
            "Live-action, a crowded night market, handheld. A vendor (S1) "
            "calls out, <d>[Mandarin] Two for the price of one, come look.</d> "
            "A tourist (S2) replies, <d>[English] How much for three?</d> An "
            "older woman (S3) laughs and says, <d>[Cantonese] He will not go "
            "lower.</d> Steam rises from a wok behind them."
        ),
    ),
    dict(
        key="reference_prompt_with_dialogue",
        shape="ref2va presentation: vision blocks then a prompt carrying <d>",
        why="**The case the postmortem's forward item 6 was written for.** "
            "Nobody had put a marker prompt through the reference presentation "
            "with the ref items wired, so whether the marker survives beside "
            "the vision blocks was an inference from a source read. This scene "
            "is the run. It goes through the exact call "
            "`comfy_extras/nodes_minimax_h3.py:351` makes.",
        text=(
            "<Picture 1> stands at the window of a wood-panelled office in the "
            "late afternoon, keeping the face, hair and clothing of the "
            "reference exactly. She (S1) turns toward the camera and says, "
            "<d>[English] I thought you would have gone by now.</d> She sets a "
            "folder down on the desk and looks back out the window."
        ),
        ref_images=1,
    ),
    dict(
        key="all_seven_together",
        shape="STRESSOR -- every one of the seven in one prompt",
        why="Not a prompt anyone would write. It exists so a reader can see "
            "all seven failure modes side by side in one token sequence, and "
            "so no marker is audited only in isolation.",
        text=(
            "Live-action, a rain-slick street at night, slow dolly-in. "
            "<|caption_start|>Seventh Street, 2:14 AM<|caption_end|> A singer "
            "(S1) at a lit doorway begins, <|lyrics_start|><d>[English] I "
            "walked the long way home.</d><d>[English] The lights were out "
            "again.</d><|lyrics_end|> She turns to a passer-by (S2) and says, "
            "<d>[English] Do you have the time?</d> He says, <d>[English] "
            "Almost two.</d><|cutoff|> A close-up of the empty doorway, the "
            "neon buzzing, rain running off the awning."
        ),
    ),
    dict(
        key="marker_saturated",
        shape="STRESSOR -- over half the prompt text is marker",
        why="The density limit. Built to answer 'how bad can this get' rather "
            "than to represent anything real: the marker share of the raw "
            "text is reported per scene and this one is built to clear half. "
            "Read it as an upper bound, not as a scene.",
        text=(
            "<|caption_start|>Alley<|caption_end|><|lyrics_start|><d>Go.</d>"
            "<d>Now.</d><d>Run.</d><d>Why?</d><d>Wait.</d><d>Move.</d>"
            "<d>Left.</d><d>Down.</d><|lyrics_end|><|cutoff|>"
            "<|caption_start|>End<|caption_end|>"
        ),
    ),
]


# --------------------------------------------------------------------------


def _bundled_declared() -> set[str]:
    """What ComfyUI's bundled tokenizer DIRECTORY declares.

    Read from the directory rather than from a live vocabulary on purpose: the
    core patch adds tokens in code, so a live vocabulary stops being able to
    tell you what the shipped directory said. This is the thing the patch
    compensates for and it does not move when the patch lands.
    """
    cfg = (_COMFY / "comfy" / "text_encoders" / "qwen25_tokenizer"
           / "tokenizer_config.json")
    return set(json.loads(cfg.read_text()).get("additional_special_tokens", []))


def _unpatched_clip(clip):
    """A CLIP whose tokenizer is what ComfyUI shipped BEFORE the core patch.

    Once the patch is applied there is no other way to obtain the broken arm,
    and without the broken arm every contamination column loses the thing it is
    measured against. Builds a fresh tokenizer with the class's token list
    emptied, which reproduces the pre-patch constructor exactly rather than
    approximating it.

    A FRESH tokenizer, not a copy: `clone()` shares the tokenizer by reference
    and a shallow copy would still share the HuggingFace object underneath.
    """
    import comfy.text_encoders.minimax as mmx
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

    # Upstream may name the token list differently than this repo's own branch
    # did -- PR 15808 puts it at module level as `MINIMAX_EXTRA_TOKENS`, the
    # local branch put it on the class as `H3_SPECIAL_TOKENS`. Empty whichever
    # exists. Guessing one name and silently finding nothing is how this
    # reconstruction would return the PATCHED tokenizer while calling it stock.
    targets = []
    for owner, attr in ((MiniMaxH3Tokenizer, "H3_SPECIAL_TOKENS"),
                        (mmx, "MINIMAX_EXTRA_TOKENS")):
        if getattr(owner, attr, None):
            targets.append((owner, attr, getattr(owner, attr)))
    try:
        for owner, attr, _ in targets:
            setattr(owner, attr, [])
        fresh = MiniMaxH3Tokenizer(
            embedding_directory=getattr(
                getattr(clip.tokenizer, "qwen3vl_32b", None),
                "embedding_directory", None),
        )
    finally:
        for owner, attr, saved in targets:
            setattr(owner, attr, saved)

    n = clip.clone()
    n.tokenizer = fresh
    return n


def _load():
    """(release tokenizer, unpatched arm, corrected arm, how it was corrected).

    **Runs identically before and after the core patch**, which is what makes
    this file the verification harness for that patch rather than only a
    demonstration of the defect. Three ways the corrected arm can arise:

      core   `MiniMaxH3Tokenizer` already declares all twenty, so the loaded
             CLIP is correct as it stands. `clip_with_vendor_tokens` must then
             be a no-op, and that is asserted rather than assumed -- a shim
             that silently did work here would mean the two disagree.
      shim   the core patch is absent and `clip_with_vendor_tokens` supplies
             the tokens, which is this pack's behaviour today.
      none   neither, which is a broken run and refuses.
    """
    import comfy.sd
    from transformers import AutoTokenizer
    if not ENCODER.exists():
        raise SystemExit(f"encoder absent: {ENCODER.name}")
    if not RELEASE_TOKENIZER.exists():
        raise SystemExit("coderef/MiniMax-H3/tokenizer is absent (coderef is "
                         "gitignored; this needs the clone)")
    release = AutoTokenizer.from_pretrained(str(RELEASE_TOKENIZER))
    loaded = comfy.sd.load_clip(ckpt_paths=[str(ENCODER)])

    import importlib
    vc = importlib.import_module(f"{_PKG}.vendor_config")
    vt = importlib.import_module(f"{_PKG}.vendor_tokens")
    declared = vc.additional_special_tokens()

    live = loaded.tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    core_patched = all(t in live for t in declared)

    if core_patched:
        mode = "core"
        corrected = loaded
        # The shim must recognise it has nothing to do. If it returns a new
        # object it did work, which would mean core and shim disagree about
        # what "already correct" means.
        passthrough = vt.clip_with_vendor_tokens(loaded, strict=True)
        if passthrough is not loaded:
            raise SystemExit(
                "the core tokenizer declares all twenty, but "
                "clip_with_vendor_tokens still rebuilt it. The shim and the "
                "core patch disagree; fix that before trusting this run.")
        unpatched = _unpatched_clip(loaded)
        still = [t for t in declared
                 if t in unpatched.tokenizer.qwen3vl_32b.tokenizer.get_vocab()
                 and t not in _bundled_declared()]
        if still:
            raise SystemExit(
                f"the unpatched arm still declares {still}, so it is not the "
                "pre-patch tokenizer and every comparison below would be "
                "against the wrong control.")
    else:
        mode = "shim"
        unpatched = loaded
        corrected = vt.clip_with_vendor_tokens(loaded, strict=True)
        if corrected is loaded:
            raise SystemExit(
                "neither the core tokenizer nor clip_with_vendor_tokens "
                "supplied the release's special tokens. There is nothing "
                "correct to compare against and this run would be green for "
                "the wrong reason.")

    print(f"=== corrected arm supplied by: {mode}")
    return release, unpatched, corrected, mode


def _ids(clip, text, ref_images=0, ref_items=None):
    """The ids ComfyUI's own path emits. Not a reimplementation of it.

    With `ref_images`, goes through `minimax_ref_items` -- the same keyword
    `MiniMaxH3ReferenceToVideo` passes at nodes_minimax_h3.py:351. The image
    payload is never touched at tokenize time (it is stashed for the encoder to
    patchify later), so a synthetic tensor exercises the real branch.
    """
    kwargs = {}
    if ref_items is not None:
        kwargs["minimax_ref_items"] = ref_items
    elif ref_images:
        import torch
        kwargs["minimax_ref_items"] = [
            {"type": "image", "data": torch.zeros(1, 64, 64, 3)}
            for _ in range(ref_images)
        ]
    entries = clip.tokenize(text, **kwargs)["qwen3vl_32b"][0]
    out = []
    for tok, *_ in entries:
        # A vision block is a dict stashed in the id slot, not an integer.
        out.append(int(tok) if isinstance(tok, int) else "<vision_block>")
    return out


def _ref_items(n_images=2, video_frames=5, n_audio=1):
    """A full reference set: images, a video with an odd frame count, and audio.

    Odd on purpose -- `tokenize_with_weights` repeat-pads to the temporal patch
    of 2, so 5 frames exercises the pad branch and produces three `<T.T
    seconds>` labels rather than a clean split. The pixel payload is never read
    at tokenize time (it is stashed for the encoder to patchify later), so
    synthetic tensors exercise the real branch.
    """
    import torch
    items = [{"type": "image", "data": torch.zeros(1, 64, 64, 3)}
             for _ in range(n_images)]
    if video_frames:
        items.append({"type": "video",
                      "data": torch.zeros(video_frames, 64, 64, 3)})
    items += [{"type": "audio", "data": None} for _ in range(n_audio)]
    return items


def _skeleton(ids):
    """The reference presentation's structure, with text runs collapsed.

    Two arms that tokenize prose differently still have to agree on this: the
    number and order of vision blocks and their flanking sentinels. A patch that
    changed it would be changing the presentation, not the prompt.
    """
    out, run = [], 0
    for t in ids:
        if t in (VISION_START, VISION_END) or t == "<vision_block>":
            if run:
                out.append(("text", run))
                run = 0
            out.append(("vision", t))
        else:
            run += 1
    if run:
        out.append(("text", run))
    return out


def _structure_only(skel):
    return [t for kind, t in skel if kind == "vision"]


def _run_reference_integrity(release, stock, patched, marker_ids, record):
    """Does the patch disturb the reference presentation itself?

    **The composability question the markers alone cannot answer.** Adding
    tokens to the tokenizer must not change how `<Picture i>` / `<Video k>` /
    `<Audio j>` labels, the `<T.T seconds>` video-block stamps, or the vision
    sentinels are emitted. Two arms:

      no_marker    a marker-free prompt over a full reference set. Stock and
                   patched must be IDENTICAL, ids and structure alike. This is
                   the assertion that the patch is inert where it should be.
      with_marker  the same reference set, a prompt carrying <d>. The vision
                   structure must still be identical and every label token run
                   before the first vision block must be unchanged; only the
                   prompt may differ.

    **Scope, stated rather than implied.** This checks the TOKEN-side
    presentation. The pixel-side preparation -- `process_qwen2vl_images` and
    `process_video_block` -- holds no vocabulary at all and is not reached at
    tokenize time, so it is out of scope here and cannot be affected by a
    tokenizer change.
    """
    items = _ref_items()
    plain = ("She stands at the window of a wood-panelled office and sets a "
             "folder down on the desk.")
    marked = ("She stands at the window and says, <d>[English] I thought you "
              "would have gone by now.</d> She sets a folder down.")

    print("\n=== reference presentation integrity "
          "[2 images + 5-frame video + audio]")
    row = {}
    ok = True

    for label, text in (("no_marker", plain), ("with_marker", marked)):
        s_ids = _ids(stock, text, ref_items=items)
        p_ids = _ids(patched, text, ref_items=items)
        s_struct = _structure_only(_skeleton(s_ids))
        p_struct = _structure_only(_skeleton(p_ids))
        same_struct = (s_struct == p_struct)
        blocks = s_struct.count("<vision_block>")

        entry = {"ids": {"stock": len(s_ids), "patched": len(p_ids)},
                 "vision_blocks": blocks,
                 "structure_identical": same_struct}

        if label == "no_marker":
            identical = (s_ids == p_ids)
            entry["fully_identical"] = identical
            print(f"  {label:<12} {blocks} vision block(s); "
                  f"stock and patched identical: {identical}")
            if not identical:
                print("    FAILED: the patch changed a marker-free reference "
                      "prompt, so it is not inert where it must be")
            ok &= identical and same_struct
        else:
            # the run of label tokens before the first vision block
            first = next(i for i, t in enumerate(p_ids) if t == VISION_START)
            labels_same = (s_ids[:first] == p_ids[:first])
            entry["label_prefix_identical"] = labels_same
            entry["marker_present"] = any(t in marker_ids for t in p_ids)
            print(f"  {label:<12} {blocks} vision block(s); structure "
                  f"identical: {same_struct}; labels identical: {labels_same}; "
                  f"marker routed: {entry['marker_present']}")
            if not (same_struct and labels_same):
                print("    FAILED: the patch disturbed the reference "
                      "presentation, not just the prompt")
            ok &= same_struct and labels_same and entry["marker_present"]

        row[label] = entry

    # What the labels actually are, recorded so the next reader does not have
    # to rerun this to see them.
    ids_plain = _ids(patched, plain, ref_items=items)
    text_ids = [t for t in ids_plain if isinstance(t, int)]
    row["decoded_presentation"] = release.decode(text_ids)
    print(f"  labels         {row['decoded_presentation'][:110]}...")

    record["reference_integrity"] = row
    return ok


def _marker_spans(ids, marker_ids):
    """Index set of positions holding a marker id."""
    return {i for i, t in enumerate(ids) if t in marker_ids}


def _contamination(patched_ids, stock_ids, marker_ids):
    """Positions that differ WITHOUT being a marker, i.e. collateral damage.

    Aligned by difflib over the id sequences rather than by position: the arms
    have different lengths, so a positional diff would compare unrelated tokens
    and attribute the offset to the markers.

    Counts, from the patched arm's point of view, tokens that are not markers
    and that the stock arm did not reproduce. That is the number that answers
    "is the damage confined to the marker".
    """
    sm = difflib.SequenceMatcher(a=patched_ids, b=stock_ids, autojunk=False)
    matched_a = set()
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            matched_a.add(blk.a + k)
    spans = _marker_spans(patched_ids, marker_ids)
    non_marker = [i for i in range(len(patched_ids)) if i not in spans]
    contaminated = [i for i in non_marker if i not in matched_a]
    return len(contaminated), len(non_marker), contaminated


def _marker_char_share(text, markers):
    total = len(text)
    marked = sum(len(m) * text.count(m) for m in markers)
    return marked, total, (marked / total if total else 0.0)


def _run_scene(scene, release, stock, patched, marker_ids, markers, record):
    key = scene["key"]
    text = scene["text"]
    refs = scene.get("ref_images", 0)
    print(f"\n=== {key}   [{scene['shape']}]")

    s_ids = _ids(stock, text, refs)
    p_ids = _ids(patched, text, refs)

    marked_chars, total_chars, char_share = _marker_char_share(text, markers)
    n_marker_tokens = len(_marker_spans(p_ids, marker_ids))
    contaminated, non_marker, where = _contamination(p_ids, s_ids, marker_ids)

    row = {
        "shape": scene["shape"],
        "why": scene["why"],
        "reference_path": bool(refs),
        "ids": {"stock": len(s_ids), "patched": len(p_ids)},
        "id_inflation": len(s_ids) - len(p_ids),
        "marker_tokens_in_patched": n_marker_tokens,
        "marker_char_share": round(char_share, 4),
        "marker_chars": marked_chars,
        "total_chars": total_chars,
        "contaminated_neighbours": contaminated,
        "non_marker_positions": non_marker,
    }

    print(f"  ids            stock {len(s_ids):>4}   patched {len(p_ids):>4}"
          f"   (+{len(s_ids) - len(p_ids)} on stock)")
    print(f"  marker tokens  {n_marker_tokens} in patched; "
          f"marker text is {char_share:.0%} of the prompt")
    print(f"  contaminated   {contaminated} of {non_marker} non-marker "
          f"positions differ on stock")

    # Control 2: patched must equal the release exactly. Text path only --
    # the release tokenizer does no vision splicing, so a reference scene has
    # no release counterpart to compare against and is checked differently.
    if refs:
        has_marker = any(t in marker_ids for t in p_ids)
        vision = sum(1 for t in p_ids if t == "<vision_block>")
        starts = sum(1 for t in p_ids if t == VISION_START)
        row["release_match"] = None
        row["reference_check"] = {
            "marker_present_beside_vision": has_marker,
            "vision_blocks": vision,
            "vision_start_tokens": starts,
        }
        ok = has_marker and vision == refs and starts == refs
        print(f"  reference path {vision} vision block(s), "
              f"marker as a real id: {has_marker}")
        if not ok:
            print("  FAILED: the reference presentation did not carry both a "
                  "vision block and a marker id")
        row["ok"] = ok
    else:
        r_ids = release(text, add_special_tokens=False)["input_ids"]
        match = (r_ids == p_ids)
        row["release_match"] = match
        row["ids"]["release"] = len(r_ids)
        print(f"  release match  {match}")
        if not match:
            print("  FAILED: patched does not reproduce the release "
                  "tokenizer; the fix is what is wrong here, not stock")
            sm = difflib.SequenceMatcher(a=r_ids, b=p_ids, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"    {tag}: release[{i1}:{i2}]={r_ids[i1:i2]} "
                          f"patched[{j1}:{j2}]={p_ids[j1:j2]}")
                    break
        row["ok"] = match

    # A decoded example of the collateral damage, so `contaminated_neighbours`
    # can be checked by eye rather than believed. Without this the count is a
    # number the harness computed about itself.
    if where:
        i = where[0]
        lo, hi = max(0, i - 3), min(len(p_ids), i + 4)
        window = [t for t in p_ids[lo:hi] if isinstance(t, int)]
        sm = difflib.SequenceMatcher(a=p_ids, b=s_ids, autojunk=False)
        # the stock arm's rendering of the same stretch of prompt text
        s_lo = None
        for blk in sm.get_matching_blocks():
            if blk.a <= lo < blk.a + blk.size:
                s_lo = blk.b + (lo - blk.a)
                break
        s_window = []
        if s_lo is not None:
            s_window = [t for t in s_ids[s_lo:s_lo + (hi - lo) + 4]
                        if isinstance(t, int)]
        # An unresolvable window means difflib found no anchor block covering
        # the position -- possible when almost every token differs, as in the
        # saturated stressor. Report null rather than an empty list, which
        # would read as "stock emitted nothing here".
        row["example"] = {
            "release_text": release.decode(window),
            "stock_text": release.decode(s_window) if s_window else None,
            "release_pieces": [release.decode([t]) for t in window],
            "stock_pieces": ([release.decode([t]) for t in s_window]
                             if s_window else None),
        }
        print(f"  example        release {row['example']['release_pieces']}")
        print(f"                 stock   {row['example']['stock_pieces']}")

    record["scenes"][key] = row
    return row["ok"]


def main() -> int:
    release, stock, patched, mode = _load()

    patched_vocab = patched.tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    markers = _missing_markers()
    marker_ids = {patched_vocab[m] for m in markers}
    print("=== the markers ComfyUI's bundled tokenizer does not declare")
    for m in markers:
        print(f"  {m:<18} -> id {patched_vocab[m]}")
    if not markers:
        print("  none: nothing to audit")
        return 1

    record = {"corrected_by": mode,
              "markers": {m: patched_vocab[m] for m in markers},
              "scenes": {}}

    # Control 1, first, because a failure here voids everything after it.
    print("\n=== control: a prompt with no markers")
    c_rel = release(CONTROL, add_special_tokens=False)["input_ids"]
    c_stk = _ids(stock, CONTROL)
    c_pat = _ids(patched, CONTROL)
    all_agree = (c_rel == c_stk == c_pat)
    print(f"  all three tokenizers identical: {all_agree}")
    record["control"] = {"all_three_identical": all_agree,
                         "ids": len(c_rel)}
    if not all_agree:
        print("  CONTROL FAILED: the tokenizers disagree on a marker-free "
              "prompt, so this harness is not measuring the markers")
        return 1

    ok = True
    for scene in SCENES:
        ok &= _run_scene(scene, release, stock, patched, marker_ids,
                         markers, record)

    ok &= _run_reference_integrity(release, stock, patched, marker_ids, record)

    out = _REPO / "bench" / "results" / "2026-08-22_h3_marker_tokenization.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(_REPO)}")

    if not ok:
        print("\nFAILED: at least one scene did not reproduce the release "
              "tokenizer or did not carry the marker through the reference "
              "path. That is a defect in the fix, not a finding about stock.")
        return 1
    print("\nEvery text-path scene reproduces the release tokenizer exactly, "
          "and the reference scene carries the marker beside its vision block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
