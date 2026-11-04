#!/usr/bin/env python3
"""Check the pure math, span masks, pair-file refusals, and vector I/O of H3 control vectors.

Claims, i.e. what breaks if a case is deleted:
- C1 (`check_tokenizer_constants`): `h3_spans.py`'s special token IDs (`VISION_START`,
  `VISION_END`, `IMAGE_PAD`, `<d>`, `</d>`) or the lone-space trailing token of
  `<Picture 1>: `, `<Video 1>: `, and `<Audio 1>: ` could drift from
  `comfy/text_encoders/minimax.py` and Qwen3-VL's tokenizer without failing.
- C2 (`check_spans_partition_and_refusals`): `h3_spans.spans` could leave a token
  uncovered or double-covered, misclassify `<Picture k>:`, `<Video k>:`, or
  `<Audio k>:` in prompt prose (`subject_definitions`, `retention_analysis`,
  `detailed_description`) as a prefix `label`, miss `(S1,S2)` or `(S1, S2)` speaker
  tags, or fail to refuse malformed `<d>...</d>` / `tags` inputs.
- C3 (`check_pair_file_loader_and_refusals`): `control_vectors.load_pairs` could
  accept a pair or hold that changes a second span alongside the quality clause,
  changes whitespace outside the clause, crosses a section or `[Shot N]` header,
  deletes a clause with an empty span on one side, contains an excluded shot,
  ignores an unknown/misspelled TOML key, refuses a short `frame = "bare"` phrase
  that happens to appear inside an excluded shot, or accepts zero/NaN/Inf weights.
- C4 (`check_math_and_synthetic_statistics`): `mean_difference`, `flip_null`,
  `random_null`, `hold_basis`, `project_out`, `mean_pairwise_cosine`,
  `per_pair_cosine`, and `split_half_cosine` could accumulate in float32, leak the
  planted direction into `flip_null` when `n` is odd or weights are unequal,
  return a non-isotropic `random_null`, include the diagonal in `mean_pairwise_cosine`
  or `per_pair_cosine`, overlap halves in `split_half_cosine`, accept near-colinear
  hold groups, or leave residual projection along a held-out direction.
- C5 (`check_vector_io_family_and_compatibility`): `save_vector`/`load_vector`
  could drop or accept malformed metadata or non-float32/non-orthonormal tensors,
  `detect_family` could fail or silently pass a model without `cached_patcher_init`
  or H3 text-path tensors, or `check_vector_compatibility` could allow a cross-family
  or mismatched-checkpoint/LoRA vector or refuse a prefix difference.
- C6 (`check_apply_post_refiner_invariants`): `apply_post_refiner` could alter
  `context` at strength 0, mutate `tags == 0` vision rows under a non-zero
  strength or custom mask, accept non-finite inputs, or change sequence length `L`.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

from _lib import bootstrap  # noqa: E402

bootstrap()

import control_vectors as cv  # noqa: E402
import h3_spans  # noqa: E402
from comfy.text_encoders.minimax import (  # noqa: E402
    MiniMaxH3Tokenizer,
    VISION_END,
    VISION_START,
    token_tags_from_embeds_info,
)


def _expect_refusal(fn, must_contain: str, label: str) -> list[str]:
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        if must_contain.lower() not in msg.lower():
            return [
                f"{label}: refused with ValueError({msg!r}), expected substring {must_contain!r}"
            ]
        return []
    except Exception as exc:
        return [
            f"{label}: raised unexpected {type(exc).__name__}({exc!r}) instead of ValueError"
        ]
    return [f"{label}: expected ValueError containing {must_contain!r}, but call succeeded"]


def check_tokenizer_constants() -> tuple[list[str], MiniMaxH3Tokenizer]:
    errs: list[str] = []
    tok = MiniMaxH3Tokenizer()
    try:
        observed = h3_spans.verify_tokenizer_constants(tok)
        if observed.get("<d>") != 151669 or observed.get("</d>") != 151670:
            errs.append(f"unexpected dialogue token IDs: {observed}")
    except Exception as exc:
        errs.append(f"verify_tokenizer_constants failed: {exc}")
    return errs, tok


def check_spans_partition_and_refusals(tok: MiniMaxH3Tokenizer) -> list[str]:
    errs: list[str] = []
    hf_tok = tok.qwen3vl_32b.tokenizer
    decode = lambda ids: hf_tok.decode(list(ids))

    ref_items = [
        {"type": "image", "data": torch.zeros((1, 64, 64, 3))},
        {"type": "audio", "data": None},
        {
            "type": "video",
            "data": torch.zeros((4, 64, 64, 3)),
            "timestamps": [0.0, 0.5, 1.0, 1.5],
        },
        {"type": "audio", "data": None},
    ]
    # Includes `<Picture 1>:`, `<Video 1>:`, `<Audio 1>:`, and `<2.5 seconds>` inside prose
    # (`subject_definitions`, `retention_analysis`, `detailed_description`) to verify that
    # only reference-prefix tokens are classed as `label` / `timestamp`.
    prompt = (
        "subject_definitions:\n"
        "<Subject 1> is the baker from <Picture 1>: a woman in a white linen apron.\n"
        "<Subject 2> is the table from <Video 1>: a scarred oak workbench.\n"
        "summary:\n"
        "[reference generation] <Subject 1> kneads dough at <Subject 2> and speaks.\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - apron retained.\n"
        "<Audio 1>: reference - room tone retained.\n"
        "detailed_description:\n"
        "[Shot 1] <Subject 1> (S1), standing at the bench from <Video 1>: oak top dusted with flour, "
        "at <2.5 seconds> in <Picture 1>: <d>[English] Almost ready.</d> "
        "Then (S1,S2) and (S1, S2) laugh together.\n"
        "overall_soundscape:\n"
        "Quiet room tone.\n"
        "non_diegetic_music:\n"
        "N/A"
    )
    entries = tok.tokenize_with_weights(prompt, minimax_ref_items=ref_items)["qwen3vl_32b"][0]

    # Derive `tags` via core's `token_tags_from_embeds_info`, expanding each vision embed to 3 patches.
    embeds_info: list[dict] = []
    expanded_pos = 0
    for item, _w in entries:
        if isinstance(item, dict):
            embeds_info.append({"type": "image", "index": expanded_pos, "size": 3})
            expanded_pos += 3
        else:
            expanded_pos += 1
    tag_tensor = token_tags_from_embeds_info(expanded_pos, embeds_info)
    tags = tag_tensor.tolist()

    token_ids = h3_spans.expand_token_ids(entries, tags)
    masks = h3_spans.spans(token_ids, tags, decode)

    cover = sum(m.to(torch.int64) for m in masks.values())
    if not bool(torch.all(cover == 1)):
        errs.append("spans masks are not mutually exclusive and exhaustive")

    if not torch.equal(masks["vision"], tag_tensor == 0):
        errs.append("vision mask does not match tags == 0")

    def _decoded(cat: str) -> str:
        idxs = masks[cat].nonzero(as_tuple=False).flatten().tolist()
        return "".join(decode([token_ids[k]]) for k in idxs)

    expected_labels = "<Picture 1>: <Audio 1>: <Video 1>: <Audio 2>: "
    if _decoded("label") != expected_labels:
        errs.append(f"label span decoded to {_decoded('label')!r}, expected {expected_labels!r}")
    if _decoded("timestamp") != "<0.2 seconds><1.2 seconds>":
        errs.append(f"timestamp span decoded to {_decoded('timestamp')!r}")
    if _decoded("dialogue") != "<d>[English] Almost ready.</d>":
        errs.append(f"dialogue span decoded to {_decoded('dialogue')!r}")
    speaker_txt = _decoded("speaker")
    if "(S1)" not in speaker_txt or "(S1,S2)" not in speaker_txt or "(S1, S2)" not in speaker_txt:
        errs.append(f"speaker span missed (S1), (S1,S2), or (S1, S2): {speaker_txt!r}")
    scene_txt = _decoded("scene")
    for prose_fragment in (
        "from <Picture 1>: a woman",
        "from <Video 1>: a scarred",
        "<Audio 1>: reference",
        "from <Video 1>: oak top",
        "at <2.5 seconds> in <Picture 1>: ",
    ):
        if prose_fragment not in scene_txt:
            errs.append(f"prose mention {prose_fragment!r} was not kept in scene mask")

    # Apply and pooling mask helpers.
    am = h3_spans.apply_mask(masks)
    if not torch.equal(am, ~masks["vision"]):
        errs.append("default apply_mask does not equal ~vision")
    pm_scene = h3_spans.pooling_mask(masks, "scene_text")
    if not torch.equal(pm_scene, masks["scene"]):
        errs.append("pooling_mask('scene_text') does not equal masks['scene']")
    pm_tag1 = h3_spans.pooling_mask(masks, "all_tag1")
    if not torch.equal(pm_tag1, ~masks["vision"]):
        errs.append("pooling_mask('all_tag1') does not equal ~vision")

    ids_b = list(token_ids)
    ids_b[-3] = ids_b[-3] + 1
    fdiff = h3_spans.first_differing_index(token_ids, ids_b)
    if fdiff != len(token_ids) - 3:
        errs.append(f"first_differing_index returned {fdiff}, expected {len(token_ids) - 3}")
    pm_fdiff = h3_spans.pooling_mask(masks, "from_first_diff", first_diff=fdiff)
    if bool(pm_fdiff[:fdiff].any()) or not bool(pm_fdiff[fdiff:].all()):
        errs.append("pooling_mask('from_first_diff') did not mask rows before first_diff")

    # Refusal controls on spans and pooling_mask.
    errs += _expect_refusal(
        lambda: h3_spans.spans(token_ids[:-1], tags, decode),
        "does not match tags length",
        "spans length mismatch",
    )
    errs += _expect_refusal(
        lambda: h3_spans.spans([10, 11], [1, 2], decode),
        "expected 0 (vision) or 1 (text)",
        "spans invalid tag value",
    )
    errs += _expect_refusal(
        lambda: h3_spans.spans([h3_spans.DIALOGUE_START, 10], [1, 1], decode),
        "unclosed <d>",
        "spans unclosed <d>",
    )
    errs += _expect_refusal(
        lambda: h3_spans.spans([10, h3_spans.DIALOGUE_END], [1, 1], decode),
        "unmatched </d>",
        "spans unmatched </d>",
    )
    errs += _expect_refusal(
        lambda: h3_spans.spans(
            [h3_spans.DIALOGUE_START, h3_spans.DIALOGUE_START, h3_spans.DIALOGUE_END],
            [1, 1, 1],
            decode,
        ),
        "nested <d>",
        "spans nested <d>",
    )
    errs += _expect_refusal(
        lambda: h3_spans.spans(
            [h3_spans.DIALOGUE_START, VISION_START, VISION_END, h3_spans.DIALOGUE_END],
            [1, 0, 0, 1],
            decode,
        ),
        "inside an unclosed <d>",
        "spans vision inside <d>",
    )
    errs += _expect_refusal(
        lambda: h3_spans.pooling_mask({"vision": masks["vision"]}, "scene_text"),
        "must contain 'vision' and 'scene'",
        "pooling_mask missing 'scene' key",
    )
    return errs


def check_pair_file_loader_and_refusals() -> list[str]:
    errs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h3_cv_pairs_") as tmp:
        tmp_dir = Path(tmp)
        bank_dir = tmp_dir / "prompt_bank"
        bank_dir.mkdir()
        excl_text = (
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot "
            "frames a zebra crossing on a quiet city street on an overcast morning across the "
            "five-second take. An elderly man in his seventies walks slowly across the crossing "
            "with a large grey tortoise beside his feet.\n\n"
            "overall_soundscape: A bus engine idles softly.\n\n"
            "non_diegetic_music: N/A"
        )
        (bank_dir / "t2va_offpath_tortoise.txt").write_text(excl_text, encoding="utf-8")

        base_a = (
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, shot on a modern "
            "digital cinema camera with clean neutral colour and crisp detail, a medium shot frames "
            "a potter shaping a clay bowl on a spinning wheel across the five-second take. Wet clay "
            "glistens under his hands as the rim rises smoothly. The camera holds a static shot. "
            "He produces no vocal sound.\n\n"
            "overall_soundscape: The wheel hums with a steady low whir and wet hands slide over clay.\n\n"
            "non_diegetic_music: N/A"
        )
        base_b = base_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "shot on 16mm colour film in 1974 with heavy grain, faded warm colour and soft blooming highlights,",
        )
        hold_a = base_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "lit by neutral white daylight,",
        )
        hold_b = base_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "lit by warm amber light,",
        )

        good_toml = tmp_dir / "film_16mm_1974.toml"
        good_toml.write_text(
            f"""[vector]
name = "film_16mm_1974"
quality = "16mm colour film in 1974 against modern digital"
a_side = "modern digital: clean neutral colour, crisp detail"
b_side = "16mm 1974: heavy grain, faded warm colour, soft blooming highlights"
mode = "t2va"
frame = "scene"
exclude_shots = ["t2va_offpath_tortoise"]

[[pair]]
weight = 1.5
a = '''{base_a}'''
b = '''{base_b}'''

[[hold]]
group = "warm_tint"
a = '''{hold_a}'''
b = '''{hold_b}'''
""",
            encoding="utf-8",
        )

        pset = cv.load_pairs(good_toml, bank_dir=bank_dir)
        if pset.name != "film_16mm_1974" or len(pset.pairs) != 1 or len(pset.holds) != 1:
            errs.append(f"load_pairs returned unexpected PairSet: {pset}")
        if (
            pset.pairs[0].a_span
            != "a modern digital cinema camera with clean neutral colour and crisp detail,"
        ):
            errs.append(f"unexpected pair[0].a_span: {pset.pairs[0].a_span!r}")
        if (
            pset.pairs[0].b_span
            != "16mm colour film in 1974 with heavy grain, faded warm colour and soft blooming highlights,"
        ):
            errs.append(f"unexpected pair[0].b_span: {pset.pairs[0].b_span!r}")

        def _write_and_load(stem: str, body: str):
            p = tmp_dir / f"{stem}.toml"
            p.write_text(body, encoding="utf-8")
            return cv.load_pairs(p, bank_dir=bank_dir)

        # Bare frame with short phrase that happens to appear inside excluded shot must succeed.
        bare_set = _write_and_load(
            "bare_ok",
            """[vector]
name = "bare_ok"
quality = "colour"
a_side = "grey"
b_side = "green"
mode = "t2va"
frame = "bare"
exclude_shots = ["t2va_offpath_tortoise"]

[[pair]]
a = "a quiet city street"
b = "a busy city street"
""",
        )
        if len(bare_set.pairs) != 1:
            errs.append("bare frame pair was not loaded")

        # 1. Empty `a` and empty `b`
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_empty_a",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_empty_a"')
                .replace(f"a = '''{base_a}'''", "a = '''   '''"),
            ),
            "non-empty strings",
            "pair-file refusal: empty a",
        )
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_empty_b",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_empty_b"')
                .replace(f"b = '''{base_b}'''", "b = '''   '''"),
            ),
            "non-empty strings",
            "pair-file refusal: empty b",
        )
        # 2. Identical `a == b`
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_same",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_same"')
                .replace(f"b = '''{base_b}'''", f"b = '''{base_a}'''"),
            ),
            "identical",
            "pair-file refusal: a == b",
        )
        # 3. Whitespace-only difference in `bare` frame and outside-clause whitespace in `scene` frame
        errs += _expect_refusal(
            lambda: cv.differing_span("heavy grain", "heavy  grain", frame="bare"),
            "differ only in whitespace",
            "differing_span refusal: bare whitespace-only",
        )
        errs += _expect_refusal(
            lambda: cv.differing_span(
                base_a,
                base_b.replace("\n\noverall_soundscape:", "\noverall_soundscape:"),
                frame="scene",
            ),
            "whitespace differences outside the differing clause",
            "differing_span refusal: newline edited outside clause",
        )
        # 4. Multi-span leak (changes quality clause AND camera movement later in the prompt)
        leaked_b = base_b.replace(
            "The camera holds a static shot.", "The camera pans rapidly to the left."
        )
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_multispan",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_multispan"')
                .replace(f"b = '''{base_b}'''", f"b = '''{leaked_b}'''"),
            ),
            "multiple separate spans",
            "pair-file refusal: multi-span camera leak",
        )
        # 5. Section/shot header boundary crossing (with only 2 shared words between edits)
        sec_a = "summary: warm light on bench. retention_analysis: <Subject 1> retained."
        sec_b = "summary: cold light on bench. retention_analysis: <Subject 2> retained."
        errs += _expect_refusal(
            lambda: cv.differing_span(sec_a, sec_b, frame="scene"),
            "section/shot header",
            "differing_span refusal: section header crossed",
        )
        # 6. One-sided empty span (pure deletion without opposite clause)
        errs += _expect_refusal(
            lambda: cv.differing_span(
                base_a,
                base_a.replace(
                    "shot on a modern digital cinema camera with clean neutral colour and crisp detail, ",
                    "",
                ),
                frame="scene",
            ),
            "empty",
            "differing_span refusal: one-sided empty span",
        )
        # 7. Excluded shot exact match AND containment (`excl_norm in norm`)
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_excl_exact",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_excl_exact"')
                .replace(f"a = '''{base_a}'''", f"a = '''{excl_text}'''")
                .replace(
                    f"b = '''{base_b}'''",
                    f"b = '''{excl_text.replace('overcast morning', 'bright afternoon')}'''",
                ),
            ),
            "equals or contains excluded shot",
            "pair-file refusal: excluded shot exact match",
        )
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_excl_contains",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_excl_contains"')
                .replace(f"a = '''{base_a}'''", f"a = '''Prefix text. {excl_text}'''")
                .replace(f"b = '''{base_b}'''", f"b = '''Other text. {excl_text}'''"),
            ),
            "equals or contains excluded shot",
            "pair-file refusal: excluded shot containment",
        )
        # 8. Zero, NaN, and Inf weights
        for bad_w_label, bad_w_lit in (("zero", "0.0"), ("nan", "nan"), ("inf", "inf")):
            errs += _expect_refusal(
                lambda bw=bad_w_lit, bl=bad_w_label: _write_and_load(
                    f"bad_w_{bl}",
                    good_toml.read_text(encoding="utf-8")
                    .replace('name = "film_16mm_1974"', f'name = "bad_w_{bl}"')
                    .replace("weight = 1.5", f"weight = {bw}"),
                ),
                "weight must be finite and non-zero",
                f"pair-file refusal: {bad_w_label} weight",
            )
        # 9. Unknown TOML keys in [vector], [[pair]], and [[hold]]
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_key_vec",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_key_vec"')
                .replace("exclude_shots =", "exclude_shot ="),
            ),
            "unknown key",
            "pair-file refusal: misspelled exclude_shot in [vector]",
        )
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_key_pair",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_key_pair"')
                .replace("weight = 1.5", "wieght = 1.5"),
            ),
            "unknown key",
            "pair-file refusal: misspelled wieght in [[pair]]",
        )
        # 10. Hold refusal (multi-span leak inside [[hold]])
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_hold_span",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_hold_span"')
                .replace(f"b = '''{hold_b}'''", f"b = '''{leaked_b}'''"),
            ),
            "multiple separate spans",
            "pair-file refusal: multi-span leak in [[hold]]",
        )
        # 11. Stem mismatch
        errs += _expect_refusal(
            lambda: _write_and_load("wrong_stem", good_toml.read_text(encoding="utf-8")),
            "does not match file stem",
            "pair-file refusal: stem mismatch",
        )
    return errs


def check_math_and_synthetic_statistics() -> list[str]:
    errs: list[str] = []
    d = 128
    # Float64 accumulation precision test: large cancelling pair differences (+1e7 and -1e7)
    # flanking 16 small pair differences (0.0625 each). Float32 sequential accumulation
    # loses the small differences into 1e7; float64 preserves (16 * 0.0625) / 18 = 1.0 / 18.
    n = 18
    a = torch.zeros((n, d), dtype=torch.float32)
    b = torch.full((n, d), 0.0625, dtype=torch.float32)
    b[0] = 1e7
    b[-1] = -1e7
    weights = torch.ones((n,), dtype=torch.float32)
    vec = cv.mean_difference(a, b, weights)
    if vec.dtype != torch.float32 or tuple(vec.shape) != (d,):
        errs.append(f"mean_difference shape/dtype wrong: {vec.shape} {vec.dtype}")
    expected_val = 1.0 / float(n)
    if not torch.allclose(vec, torch.full((d,), expected_val, dtype=torch.float32), atol=1e-7):
        errs.append(
            f"mean_difference failed float64 precision check: {vec[0].item()} vs {expected_val}"
        )

    # Also test float64 difference precision when a and b have a large common offset (16384.0)
    # and small differences in float64 inputs: downcasting a and b to float32 before b - a
    # loses bits below 2^-10.
    a_f64 = torch.full((4, d), 16384.0, dtype=torch.float64)
    b_f64 = a_f64 + 1e-4
    vec_f64 = cv.mean_difference(a_f64, b_f64)
    if abs(float(vec_f64[0].item()) - 1e-4) > 1e-7:
        errs.append(f"mean_difference downcast a/b before subtraction: {vec_f64[0].item()}")

    # Negative weights test: sum(w * (B - A)) / sum(|w|)
    w_signed = torch.tensor([1.0, -1.0], dtype=torch.float32)
    a2 = torch.zeros((2, 4), dtype=torch.float32)
    b2 = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]], dtype=torch.float32)
    v_signed = cv.mean_difference(a2, b2, w_signed)
    expected_signed = torch.tensor([1.0, -2.0, 0.0, 0.0], dtype=torch.float32)
    if not torch.allclose(v_signed, expected_signed, atol=1e-6):
        errs.append(f"signed-weight mean_difference wrong: {v_signed} vs {expected_signed}")

    # Synthetic signal-plus-noise pairs: diffs_i = mu + eps_i, ||mu|| = 1, eps_i ~ N(0, sigma^2 / d I).
    gen = torch.Generator().manual_seed(20261104)
    mu = torch.zeros(d, dtype=torch.float64)
    mu[0] = 1.0
    sigma = 0.5
    noise = torch.randn((32, d), generator=gen, dtype=torch.float64) * (sigma / math.sqrt(d))
    diffs32 = (mu.unsqueeze(0) + noise).to(torch.float32)
    zeros32 = torch.zeros_like(diffs32)

    # Test flip_null across even n (12), odd n (9 and 11), and unequal weights ([3, 1, 1, ...]):
    # in every case, the rescaled flip_null must match the true vector's norm AND cancel the
    # shared direction mu[0] (mean |cos(fnull, mu)| < 0.20 across draws).
    for case_label, n_sub, w_sub in (
        ("even n=12", 12, None),
        ("odd n=9", 9, None),
        ("odd n=11", 11, None),
        ("unequal weights n=8", 8, torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)),
    ):
        sub_diffs = diffs32[:n_sub]
        sub_zeros = zeros32[:n_sub]
        v_sub = cv.mean_difference(sub_zeros, sub_diffs, w_sub)
        sub_norm = float(torch.linalg.vector_norm(v_sub.to(torch.float64)).item())
        cos_draws: list[float] = []
        for draw_idx in range(16):
            fn_vec = cv.flip_null(
                sub_zeros,
                sub_diffs,
                w_sub,
                generator=torch.Generator().manual_seed(1000 + draw_idx),
            )
            fn_norm = float(torch.linalg.vector_norm(fn_vec.to(torch.float64)).item())
            if abs(fn_norm - sub_norm) > 1e-5 * sub_norm:
                errs.append(f"flip_null ({case_label}) norm {fn_norm} != {sub_norm}")
                break
            cos_draws.append(abs(float(fn_vec[0].item()) / fn_norm))
        mean_abs_cos = sum(cos_draws) / len(cos_draws) if cos_draws else 1.0
        if mean_abs_cos > 0.20:
            errs.append(
                f"flip_null ({case_label}) leaked planted signal: mean |cos(null, mu)| = {mean_abs_cos:.4f} > 0.20"
            )

    # random_null norm and isotropy check across multiple draws (rejects constant vectors like ones(d)).
    v_true = cv.mean_difference(zeros32, diffs32)
    true_norm = float(torch.linalg.vector_norm(v_true.to(torch.float64)).item())
    rnull_a = cv.random_null(true_norm, d, generator=torch.Generator().manual_seed(1))
    rnull_b = cv.random_null(true_norm, d, generator=torch.Generator().manual_seed(2))
    for rn in (rnull_a, rnull_b):
        rn_norm = float(torch.linalg.vector_norm(rn.to(torch.float64)).item())
        if abs(rn_norm - true_norm) > 1e-5 * true_norm:
            errs.append(f"random_null norm {rn_norm} != {true_norm}")
    cos_ab = abs(float(torch.dot(rnull_a, rnull_b).item()) / (true_norm * true_norm))
    if cos_ab > 0.35 or float(rnull_a.std().item()) < 1e-4:
        errs.append(f"random_null is not isotropic: cos_ab={cos_ab:.4f}, std={rnull_a.std().item():.6f}")

    # Off-diagonal mean_pairwise_cosine and per_pair_cosine on orthogonal rows:
    # must both be 0.0 (including the diagonal would make them 1/n or 1/(n-1) > 0).
    ortho4 = torch.eye(4, d, dtype=torch.float32)
    mpc_ortho = cv.mean_pairwise_cosine(ortho4)
    if abs(mpc_ortho) > 1e-7:
        errs.append(f"mean_pairwise_cosine on orthogonal rows is {mpc_ortho}, expected 0.0")
    ppc_ortho = cv.per_pair_cosine(ortho4)
    if max(abs(x) for x in ppc_ortho) > 1e-7:
        errs.append(f"per_pair_cosine on orthogonal rows is {ppc_ortho}, expected all 0.0")

    # Disjointness of split_half_cosine: on 4 orthogonal basis rows, disjoint halves
    # have exact cosine 0.0, whereas overlapping halves have positive cosine.
    sh_ortho = cv.split_half_cosine(ortho4, generator=torch.Generator().manual_seed(7), n_splits=16)
    if abs(sh_ortho) > 1e-7:
        errs.append(f"split_half_cosine on 4 orthogonal rows is {sh_ortho}, expected 0.0 (halves overlapped)")

    # Split-half reliability rises with n on planted-direction pairs.
    sh4 = cv.split_half_cosine(diffs32[:4], generator=torch.Generator().manual_seed(7), n_splits=64)
    sh32 = cv.split_half_cosine(diffs32, generator=torch.Generator().manual_seed(7), n_splits=64)
    if not (0.5 < sh4 < sh32 < 1.0):
        errs.append(
            f"split_half_cosine failed expected ordering 0.5 < sh4 ({sh4:.4f}) < sh32 ({sh32:.4f}) < 1.0"
        )

    # Outlier detection via per_pair_cosine: flip pair 0 and verify its per_pair_cosine < 0.
    diffs_outlier = diffs32[:8].clone()
    diffs_outlier[0] = -diffs_outlier[0]
    ppc = cv.per_pair_cosine(diffs_outlier)
    if not (ppc[0] < -0.5 and min(ppc[1:]) > 0.4):
        errs.append(f"per_pair_cosine did not isolate inverted pair 0: {ppc}")

    # hold_basis and project_out: two non-orthogonal hold groups in R^d.
    ha = torch.zeros((3, d), dtype=torch.float32)
    hb = torch.zeros((3, d), dtype=torch.float32)
    hb[0, 1] = 2.0
    hb[1, 1] = 4.0
    hb[2, 1] = 1.0
    hb[2, 2] = 3.0
    basis = cv.hold_basis(ha, hb, ["g1", "g1", "g2"])
    if tuple(basis.shape) != (2, d):
        errs.append(f"hold_basis shape {tuple(basis.shape)} != (2, {d})")
    gram = basis.to(torch.float64) @ basis.to(torch.float64).T
    if not torch.allclose(gram, torch.eye(2, dtype=torch.float64), atol=1e-6):
        errs.append(f"hold_basis is not orthonormal: gram={gram}")

    v_in = torch.zeros(d, dtype=torch.float32)
    v_in[0] = 5.0
    v_in[1] = -3.0
    v_in[2] = 7.0
    v_proj = cv.project_out(v_in, basis)
    residual_dots = basis.to(torch.float64) @ v_proj.to(torch.float64)
    if float(residual_dots.abs().max().item()) > 1e-6:
        errs.append(f"project_out left residual along basis: {residual_dots.tolist()}")
    if abs(float(v_proj[0].item()) - 5.0) > 1e-6:
        errs.append(f"project_out altered orthogonal component v[0]: {v_proj[0].item()}")

    # Colinear and near-colinear hold group refusals, plus project_out shape/NaN refusals.
    errs += _expect_refusal(
        lambda: cv.hold_basis(ha[:2], hb[:2], ["g1", "g2"]),
        "zero or colinear",
        "hold_basis exact colinear group refusal",
    )
    hb_near = torch.zeros((2, d), dtype=torch.float32)
    hb_near[0, :8] = torch.linspace(1.0, 2.0, 8, dtype=torch.float32)
    hb_near[1, :8] = hb_near[0, :8] * 1.0000001
    errs += _expect_refusal(
        lambda: cv.hold_basis(torch.zeros_like(hb_near), hb_near, ["g1", "g2"]),
        "zero or colinear",
        "hold_basis near-colinear float32 group refusal",
    )
    errs += _expect_refusal(
        lambda: cv.project_out(v_in, torch.zeros((0, d + 1), dtype=torch.float32)),
        "basis must have shape",
        "project_out wrong basis width refusal",
    )
    return errs


def check_vector_io_family_and_compatibility() -> list[str]:
    errs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h3_cv_io_") as tmp:
        tmp_dir = Path(tmp)

        ckpt_path = tmp_dir / "mock_h3_ckpt.safetensors"
        ckpt_tensors = {
            "condition_proj.weight": torch.ones((16, 16), dtype=torch.bfloat16),
            "condition_proj.bias": torch.zeros((16,), dtype=torch.bfloat16),
            "token_refiner.final_norm.weight": torch.ones((16,), dtype=torch.bfloat16),
        }
        save_file(ckpt_tensors, str(ckpt_path))

        fam_file = cv.detect_family(ckpt_path)
        fam_sd = cv.detect_family(ckpt_tensors)
        if not fam_file.startswith("custom:") or fam_file != fam_sd:
            errs.append(f"detect_family mismatch between file ({fam_file!r}) and state_dict ({fam_sd!r})")

        mock_dm = SimpleNamespace(
            condition_proj=SimpleNamespace(
                weight=ckpt_tensors["condition_proj.weight"],
                bias=ckpt_tensors["condition_proj.bias"],
            ),
            token_refiner=SimpleNamespace(
                final_norm=SimpleNamespace(weight=ckpt_tensors["token_refiner.final_norm.weight"])
            ),
        )
        mock_patcher = SimpleNamespace(
            cached_patcher_init=None,
            backup={},
            model=SimpleNamespace(diffusion_model=mock_dm),
        )
        fam_patcher = cv.detect_family(mock_patcher)
        if fam_patcher != fam_file:
            errs.append(
                f"detect_family on patcher without cached_patcher_init returned {fam_patcher!r}, expected {fam_file!r}"
            )

        broken_patcher = SimpleNamespace(
            cached_patcher_init=None,
            backup={},
            model=SimpleNamespace(diffusion_model=SimpleNamespace()),
        )
        errs += _expect_refusal(
            lambda: cv.detect_family(broken_patcher),
            "missing h3 text-path tensor",
            "detect_family refusal on non-H3 model without cached_patcher_init",
        )

        if (
            cv.family_from_tensor_hashes(("af3c58d7f2ad", "701e0673282f", "cf10a56a1137"))
            != "fl2va"
            or cv.family_from_tensor_hashes(("7c5485465531", "529e19c04005", "6ca615e4e3f8"))
            != "ref2va"
            or cv.family_from_tensor_hashes(("d6b17a52eb10", "2d8bca2b367d", "cf10a56a1137"))
            != "fasth3"
        ):
            errs.append("KNOWN_FAMILIES table did not map measured hashes to fl2va/ref2va/fasth3")

        digest = cv.safetensors_recipe_digest(ckpt_path, extra_tensor_keys=cv.FAMILY_TENSOR_KEYS)
        if len(digest) != 64:
            errs.append(f"safetensors_recipe_digest returned invalid sha256: {digest!r}")

        # Vector save/load round-trip (with v_post orthogonal to h_post).
        v_post = torch.linspace(-1.0, 1.0, cv.POST_REFINER_DIM, dtype=torch.float32)
        v_post[0] = 0.0
        h_post = torch.zeros((1, cv.POST_REFINER_DIM), dtype=torch.float32)
        h_post[0, 0] = 1.0
        tensors_in = {
            "vector.post_refiner": v_post,
            "holds.post_refiner": h_post,
        }
        recipe_a = {
            "checkpoint": {"file": "minimax_h3_fl2va_int8_convrot.safetensors", "sha256": digest},
            "loras": [],
            "encoder": {"file": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "sha256": "abc123"},
        }
        meta_in = {
            "format": cv.FORMAT_VERSION,
            "name": "film_16mm_1974",
            "quality": "16mm colour film in 1974 against modern digital",
            "pairs": {
                "path": "prompt_bank/control_vectors/film_16mm_1974.toml",
                "sha256": "deadbeef",
                "n_pairs": 12,
                "n_holds": 1,
            },
            "recipe": recipe_a,
            "family": "fl2va",
            "measured_under": {"mode": "t2va", "frame": "scene", "prefix": "none", "pooling": "scene_text"},
            "points": {
                "post_refiner": {
                    "unit": "mean_pair_difference",
                    "raw_norm": float(torch.linalg.vector_norm(v_post).item()),
                    "median_row_norm": 42.5,
                }
            },
            "quality_report": {
                "mean_pairwise_cosine": 0.62,
                "split_half_cosine": 0.88,
                "per_pair_cosine": [0.61, 0.63],
            },
            "provenance": {
                "core_revision": "0123456",
                "pack_commit": "6bb1eaad",
                "created_utc": "2026-11-04T18:00:00Z",
                "made_by": "check_control_vectors",
            },
        }
        vec_path = tmp_dir / "film_16mm_1974__fl2va.safetensors"
        cv.save_vector(vec_path, tensors_in, meta_in)
        tensors_out, meta_out = cv.load_vector(vec_path)

        if set(tensors_out.keys()) != set(tensors_in.keys()):
            errs.append(f"round-trip tensor keys {set(tensors_out.keys())} != {set(tensors_in.keys())}")
        for k in tensors_in:
            if not torch.equal(tensors_in[k], tensors_out[k]):
                errs.append(f"round-trip tensor {k!r} not bitwise equal")
        if meta_out != meta_in:
            errs.append(f"round-trip metadata mismatch: {meta_out} != {meta_in}")

        # Save/load validation refusals:
        # - non-float32 tensor at save and load
        bf16_path = tmp_dir / "bad_bf16.safetensors"
        errs += _expect_refusal(
            lambda: cv.save_vector(
                bf16_path, {"vector.post_refiner": v_post.to(torch.bfloat16)}, meta_in
            ),
            "dtype float32",
            "save_vector refusal: bf16 tensor",
        )
        save_file(
            {"vector.post_refiner": v_post.to(torch.bfloat16)},
            str(bf16_path),
            metadata={k: cv.json.dumps(v) for k, v in meta_in.items()},
        )
        errs += _expect_refusal(
            lambda: cv.load_vector(bf16_path),
            "dtype float32",
            "load_vector refusal: bf16 tensor on disk",
        )
        # - unknown format on load_vector
        bad_fmt_path = tmp_dir / "bad_fmt.safetensors"
        save_file(
            {"vector.post_refiner": v_post},
            str(bad_fmt_path),
            metadata={k: cv.json.dumps(v) for k, v in {**meta_in, "format": "h3-control-vector/99"}.items()},
        )
        errs += _expect_refusal(
            lambda: cv.load_vector(bad_fmt_path),
            "expected 'h3-control-vector/1'",
            "load_vector refusal: unknown format",
        )

        # Compatibility checks:
        # 1. Exact match -> 0 warnings
        warns = cv.check_vector_compatibility(
            meta_out, loaded_family="fl2va", loaded_recipe=recipe_a, loaded_prefix="none"
        )
        if warns:
            errs.append(f"unexpected compatibility warnings on exact match: {warns}")

        # 2. Family mismatch -> hard refusal even with allow_recipe_mismatch=True
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_out,
                loaded_family="ref2va",
                loaded_recipe=recipe_a,
                allow_recipe_mismatch=True,
            ),
            "cannot apply to loaded model of family 'ref2va'",
            "compatibility refusal: wrong family",
        )

        # 3. Checkpoint mismatch -> refuses by default, warns when allow_recipe_mismatch=True
        recipe_ckpt_b = {
            **recipe_a,
            "checkpoint": {"file": "other_fl2va.safetensors", "sha256": "0000"},
        }
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_out, loaded_family="fl2va", loaded_recipe=recipe_ckpt_b, allow_recipe_mismatch=False
            ),
            "recipe checkpoint",
            "compatibility refusal: different checkpoint",
        )
        warns_ckpt = cv.check_vector_compatibility(
            meta_out, loaded_family="fl2va", loaded_recipe=recipe_ckpt_b, allow_recipe_mismatch=True
        )
        if len(warns_ckpt) != 1 or "recipe checkpoint" not in warns_ckpt[0].lower():
            errs.append(f"expected 1 checkpoint warning when allow_recipe_mismatch=True, got {warns_ckpt}")

        # 4. LoRA mismatch -> refuses by default, warns when allow_recipe_mismatch=True
        recipe_lora_b = {
            **recipe_a,
            "loras": [{"file": "pika.safetensors", "sha256": "111", "strength": 1.0, "applied": "branch"}],
        }
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_out, loaded_family="fl2va", loaded_recipe=recipe_lora_b, allow_recipe_mismatch=False
            ),
            "recipe lora set",
            "compatibility refusal: different LoRA set",
        )
        warns_lora = cv.check_vector_compatibility(
            meta_out, loaded_family="fl2va", loaded_recipe=recipe_lora_b, allow_recipe_mismatch=True
        )
        if len(warns_lora) != 1 or "lora set" not in warns_lora[0].lower():
            errs.append(f"expected 1 LoRA warning when allow_recipe_mismatch=True, got {warns_lora}")

        # 5. Prefix mismatch -> reported in warnings, never refused
        warns_prefix = cv.check_vector_compatibility(
            meta_out,
            loaded_family="fl2va",
            loaded_recipe=recipe_a,
            loaded_prefix={"references_sha256": "9999"},
        )
        if len(warns_prefix) != 1 or "measured under prefix" not in warns_prefix[0].lower():
            errs.append(f"expected 1 prefix warning, got {warns_prefix}")

        # 6. Unknown format refusal
        bad_meta = {**meta_out, "format": "h3-control-vector/99"}
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(bad_meta, loaded_family="fl2va"),
            "expected 'h3-control-vector/1'",
            "compatibility refusal: unknown format",
        )
    return errs


def check_apply_post_refiner_invariants() -> list[str]:
    errs: list[str] = []
    B, L, D = 1, 12, cv.POST_REFINER_DIM
    gen = torch.Generator().manual_seed(42)
    context = torch.randn((B, L, D), generator=gen, dtype=torch.bfloat16)
    tags = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    vec = torch.ones((D,), dtype=torch.float32) * 0.5

    # 1. Strength 0 reproduces unhooked context bitwise.
    out_zero = cv.apply_post_refiner(context, tags, [(vec, 0.0)])
    if not torch.equal(out_zero, context):
        errs.append("apply_post_refiner at strength 0 did not reproduce context bitwise")

    # 2. Non-zero strength leaves masked-off (tag 0 vision) rows bitwise unchanged and preserves L.
    out_pos = cv.apply_post_refiner(context, tags, [(vec, 1.5), (vec, -0.5)])
    if out_pos.shape != (B, L, D):
        errs.append(f"apply_post_refiner changed shape from {(B, L, D)} to {tuple(out_pos.shape)}")
    if not torch.equal(out_pos[:, :4, :], context[:, :4, :]):
        errs.append("apply_post_refiner mutated tag==0 vision rows")
    expected_text = context[:, 4:, :] + vec.to(dtype=context.dtype)
    if not torch.equal(out_pos[:, 4:, :], expected_text):
        errs.append("apply_post_refiner did not add expected shift on tag==1 rows")

    # 3. Custom span mask (even if True on vision index 0) leaves vision rows and unmasked text rows bitwise unchanged.
    custom_mask = torch.zeros(L, dtype=torch.bool)
    custom_mask[0] = True
    custom_mask[6:9] = True
    out_custom = cv.apply_post_refiner(context, tags, [(vec, 2.0)], mask=custom_mask)
    expected_unmodified = torch.ones(L, dtype=torch.bool)
    expected_unmodified[6:9] = False
    if not torch.equal(out_custom[:, expected_unmodified, :], context[:, expected_unmodified, :]):
        errs.append("apply_post_refiner mutated vision row 0 or unmasked rows under custom_mask")

    # 4. Refusals on wrong mask shape and non-finite vector.
    errs += _expect_refusal(
        lambda: cv.apply_post_refiner(context, tags, [(vec, 1.0)], mask=torch.ones(L - 1, dtype=torch.bool)),
        "mask shape",
        "apply_post_refiner refusal: wrong mask shape",
    )
    bad_vec = vec.clone()
    bad_vec[0] = float("nan")
    errs += _expect_refusal(
        lambda: cv.apply_post_refiner(context, tags, [(bad_vec, 1.0)]),
        "non-finite",
        "apply_post_refiner refusal: NaN vector",
    )

    return errs


def main() -> int:
    all_errs: list[str] = []

    c1_errs, tok = check_tokenizer_constants()
    if c1_errs:
        for e in c1_errs:
            print(f"  FAIL  [C1] {e}")
        all_errs.extend(c1_errs)
    else:
        print("  ok    [C1] h3_spans special token IDs and label space tokens match comfy.text_encoders.minimax")

    for code, name, fn in (
        (
            "C2",
            "h3_spans partition, prose-colon immunity, pooling masks, and malformed-input refusals",
            lambda: check_spans_partition_and_refusals(tok),
        ),
        (
            "C3",
            "load_pairs single-clause diff extraction, unknown-key refusals, and section-1 refusals",
            check_pair_file_loader_and_refusals,
        ),
        (
            "C4",
            "float64 control-vector math, odd-n/weighted flip_null, orthonormal holds, and reliability statistics",
            check_math_and_synthetic_statistics,
        ),
        (
            "C5",
            "safetensors round-trip, family detection (with and without cached_patcher_init), and recipe refusals",
            check_vector_io_family_and_compatibility,
        ),
        (
            "C6",
            "apply_post_refiner strength-0 exactness, bitwise vision-row preservation, and L invariance",
            check_apply_post_refiner_invariants,
        ),
    ):
        errs = fn()
        if errs:
            for e in errs:
                print(f"  FAIL  [{code}] {e}")
            all_errs.extend(errs)
        else:
            print(f"  ok    [{code}] {name}")

    if all_errs:
        print(f"\nFAIL: {len(all_errs)} control-vector check error(s)")
        return 1
    print("\nok: all 6 control-vector check groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
