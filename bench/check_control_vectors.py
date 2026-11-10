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
- C7 (`check_runtime_nodes_arm_and_cli`): `control_vector_nodes.py` could fail to
  register `MiniMaxH3MakeControlVector` or `MiniMaxH3ApplyControlVectors`, fail to
  fall back or refuse cleanly on models without `cached_patcher_init`, fail to
  accumulate stacked `MiniMaxH3ApplyControlVectors` wrappers in `float64` while
  keeping `tag == 0` vision rows bitwise unchanged, fail to wire node `"115"` in
  `workflows/build_workflows.py::build_control_vector_arm`, or fail `bench/control_vectors.py inspect`.
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
sys.path = [p for p in sys.path if Path(p or ".").resolve() != (REPO / "bench").resolve()]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO))

from _lib import bootstrap  # noqa: E402

bootstrap(cpu=True)

import control_vectors as cv  # noqa: E402
import h3_pair_compiler as pair_comp  # noqa: E402
import h3_spans  # noqa: E402
import j_lens  # noqa: E402
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

    # Refusal when a mock tokenizer does not end a label with a lone space token
    hf_real = tok.qwen3vl_32b.tokenizer

    class _BadLabelTok:
        def encode(self, s: str, add_special_tokens: bool = False):
            return hf_real.encode(s, add_special_tokens=add_special_tokens)

        def decode(self, ids):
            out = hf_real.decode(ids)
            return "_no_space" if out == " " else out

    errs += _expect_refusal(
        lambda: h3_spans.verify_tokenizer_constants(_BadLabelTok()),
        "did not end in a lone space token",
        "verify_tokenizer_constants refusal when label does not end in lone space",
    )
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
    im_non_vis = h3_spans.inference_mask(masks, "non_vision")
    if not torch.equal(im_non_vis, ~masks["vision"]):
        errs.append("inference_mask('non_vision') does not equal ~vision")
    im_all = h3_spans.inference_mask(masks, "all_rows")
    if not bool(im_all.all()):
        errs.append("inference_mask('all_rows') is not all True")
    im_look = h3_spans.inference_mask(masks, "look_only")
    im_scene = h3_spans.inference_mask(masks, "scene_only")
    im_desc_main = h3_spans.inference_mask(masks, "description_only")
    im_sound_main = h3_spans.inference_mask(masks, "soundscape_only")
    if not torch.equal(im_scene, im_look | im_desc_main):
        errs.append("inference_mask('scene_only') on sectioned prompt does not equal look_only | description_only")
    if bool((im_look & im_desc_main).any()) or bool((im_look & im_sound_main).any()) or bool((im_desc_main & im_sound_main).any()):
        errs.append("look_only, description_only, and soundscape_only are not pairwise disjoint on sectioned prompt")
    if not bool((im_scene <= (masks["scene"] & (~masks["vision"]))).all()):
        errs.append("inference_mask('scene_only') leaked outside masks['scene'] & ~vision")
    if int(im_scene.sum().item()) >= int(masks["scene"].sum().item()):
        errs.append("inference_mask('scene_only') did not mask out non-steerable sections/headers/tags on sectioned prompt")
    im_scene_decoded = "".join(
        decode([token_ids[k]]) for k in im_scene.nonzero(as_tuple=False).flatten().tolist()
    )
    im_sound_decoded = "".join(
        decode([token_ids[k]]) for k in im_sound_main.nonzero(as_tuple=False).flatten().tolist()
    )
    for forbidden_sub in (
        "subject_definitions",
        "summary",
        "retention_analysis",
        "non_diegetic_music",
        "detailed_description",
        "overall_soundscape",
        "[reference generation]",
        "fully_preserved",
        "Quiet room tone",
        "room tone retained",
        "N/A",
        "[Shot 1]",
        "<Subject 1>",
        "<Picture 1>",
        "<Video 1>",
        "<Audio 1>",
    ):
        if forbidden_sub in im_scene_decoded:
            errs.append(f"inference_mask('scene_only') leaked structural/audio substring {forbidden_sub!r}: {im_scene_decoded!r}")
    for expected_sub in ("white linen apron", "scarred oak workbench", "kneads dough", "apron retained", "standing at the bench"):
        if expected_sub not in im_scene_decoded:
            errs.append(f"inference_mask('scene_only') omitted steerable visual prose {expected_sub!r}: {im_scene_decoded!r}")
    for expected_snd in ("Quiet room tone", "room tone retained"):
        if expected_snd not in im_sound_decoded:
            errs.append(f"inference_mask('soundscape_only') omitted audio prose {expected_snd!r}: {im_sound_decoded!r}")
    if "N/A" in im_sound_decoded or "<Audio 1>" in im_sound_decoded:
        errs.append(f"inference_mask('soundscape_only') leaked 'N/A' or '<Audio 1>': {im_sound_decoded!r}")
    if bool((im_scene & masks["dialogue"]).any()):
        errs.append("inference_mask('scene_only') did not mask out <d>...</d> dialogue rows")
    im_after_dlg = h3_spans.inference_mask(masks, "after_dialogue")
    dlg_idx_list = masks["dialogue"].nonzero(as_tuple=False).flatten().tolist()
    if not dlg_idx_list:
        errs.append("expected dialogue rows in masks for after_dialogue test")
    else:
        last_dlg_idx = dlg_idx_list[-1]
        if (
            bool(im_after_dlg[: last_dlg_idx + 1].any())
            or not torch.equal(im_after_dlg[last_dlg_idx + 1 :], masks["scene"][last_dlg_idx + 1 :])
            or not torch.equal(im_after_dlg, (~masks["vision"]) & masks["scene"] & (torch.arange(len(token_ids)) > last_dlg_idx))
        ):
            errs.append("inference_mask('after_dialogue') did not mask rows up through the final </d> token")
    # Also test inference_mask('after_dialogue') when a scene row appears BEFORE dialogue as well as AFTER dialogue
    synth_masks = {
        "vision": torch.tensor([True, False, False, False, False]),
        "scene": torch.tensor([False, True, False, False, True]),
        "dialogue": torch.tensor([False, False, True, True, False]),
    }
    im_synth_after = h3_spans.inference_mask(synth_masks, "after_dialogue")
    if not torch.equal(im_synth_after, torch.tensor([False, False, False, False, True])):
        errs.append(f"inference_mask('after_dialogue') did not mask pre-dialogue scene row: {im_synth_after}")
    errs += _expect_refusal(
        lambda: h3_spans.inference_mask(masks, "bogus_mask_mode"),
        "unknown inference mask mode",
        "inference_mask unknown mode refusal",
    )

    # Verify span_masks_from_prompt with text-only and with prefix offset (using bank prompts)
    noodle_bank_prompt = (REPO / "prompt_bank" / "t2va_noodle_bar.txt").read_text(encoding="utf-8").strip()
    errs += _expect_refusal(
        lambda: h3_spans.span_masks_from_prompt(
            noodle_bank_prompt,
            0,
            tokenizer=tok,
        ),
        "total_length must be >= 1",
        "span_masks_from_prompt total_length=0 refusal",
    )
    raw_entries = tok.tokenize_with_weights(noodle_bank_prompt)["qwen3vl_32b"][0]
    n_text_tok = len(raw_entries)
    prefix_tags = torch.tensor([0, 0, 1] + [1] * n_text_tok, dtype=torch.long)
    sm_prefixed = h3_spans.span_masks_from_prompt(
        noodle_bank_prompt,
        n_text_tok + 3,
        tags=prefix_tags,
        tokenizer=tok,
    )
    if (
        sm_prefixed is None
        or int(sm_prefixed["vision"].sum().item()) != 2
        or not bool(sm_prefixed["dialogue"].any())
        or not bool(sm_prefixed["scene"].any())
    ):
        errs.append(f"span_masks_from_prompt failed on prefixed sequence: {sm_prefixed}")
    else:
        im_look_pref = h3_spans.inference_mask(sm_prefixed, "look_only")
        im_scene_pref = h3_spans.inference_mask(sm_prefixed, "scene_only")
        im_desc_pref = h3_spans.inference_mask(sm_prefixed, "description_only")
        im_sound_pref = h3_spans.inference_mask(sm_prefixed, "soundscape_only")
        if not bool(im_desc_pref.any()) or not bool(im_sound_pref.any()):
            errs.append("description_only or soundscape_only empty on prefixed fl2va prompt")
        if not torch.equal(im_look_pref, im_desc_pref) or not torch.equal(im_scene_pref, im_desc_pref):
            errs.append("look_only or scene_only did not match description_only on 3-field t2va/fl2va prompt")
        if bool((im_desc_pref & im_sound_pref).any()):
            errs.append("description_only and soundscape_only overlapped on prefixed fl2va prompt")
        if bool((im_desc_pref & sm_prefixed["dialogue"]).any()) or bool((im_desc_pref & sm_prefixed["vision"]).any()):
            errs.append("description_only leaked into dialogue or vision rows")

    # Verify span_masks_from_prompt on a prefixed sectionless P-H3-Image-Edit prompt preserves _h3_steerable_scene_mask:
    edit_img_prompt = (REPO / "prompt_bank" / "p_h3_image" / "edit_alpine_terrace.txt").read_text(encoding="utf-8").strip()
    edit_entries = tok.tokenize_with_weights(edit_img_prompt)["qwen3vl_32b"][0]
    n_edit_tok = len(edit_entries)
    edit_prefix_tags = torch.tensor([0, 0, 1] + [1] * n_edit_tok, dtype=torch.long)
    sm_edit_pref = h3_spans.span_masks_from_prompt(
        edit_img_prompt,
        n_edit_tok + 3,
        tags=edit_prefix_tags,
        tokenizer=tok,
    )
    if sm_edit_pref is None:
        errs.append("span_masks_from_prompt returned None on prefixed P-H3-Image-Edit prompt")
    else:
        im_edit_scene = h3_spans.inference_mask(sm_edit_pref, "scene_only")
        if int(im_edit_scene.sum().item()) >= int(sm_edit_pref["scene"].sum().item()):
            errs.append("inference_mask('scene_only') on prefixed P-H3-Image-Edit prompt did not mask out '<Picture N>'")
        if bool((im_edit_scene & sm_edit_pref["vision"]).any()):
            errs.append("inference_mask('scene_only') on prefixed P-H3-Image-Edit prompt leaked into vision rows")

    # Verify attach_span_masks works with a 4B-keyed PH3ImageTokenizer wrapper and sets both cond[i][1]['h3_span_masks'] and tags._h3_span_masks:
    class _Mock4BTokenizer:
        def __init__(self, base_tok):
            self.qwen3vl_4b = base_tok.qwen3vl_32b

        def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
            return {"qwen3vl_4b": self.qwen3vl_4b.tokenize_with_weights(text, return_word_ids, **kwargs)}

    mock_4b_tok = _Mock4BTokenizer(tok)
    cond_4b_tags = torch.ones((n_edit_tok,), dtype=torch.long)
    cond_4b = [[torch.zeros((1, n_edit_tok, 8)), {"minimax_token_tags": cond_4b_tags}]]
    h3_spans.attach_span_masks(cond_4b, mock_4b_tok, mock_4b_tok.tokenize_with_weights(edit_img_prompt))
    if (
        not isinstance(cond_4b[0][1].get("h3_span_masks"), dict)
        or "h3_span_masks" not in (cond_4b[0][1].get("model_conds") or {})
        or getattr(cond_4b_tags, "_h3_span_masks", None) is None
    ):
        errs.append("attach_span_masks with qwen3vl_4b tokenizer did not set cond[0][1]['h3_span_masks'], model_conds['h3_span_masks'], and tags._h3_span_masks")

    # Test 3-lever masks (look_only, scene_only, description_only, soundscape_only) on ref2va prompt structure (including prefixed span_masks_from_prompt)
    ref2va_sample_prompt = (REPO / "prompt_bank" / "ref2va_look_anchor.txt").read_text(encoding="utf-8").strip()
    r2v_entries = tok.tokenize_with_weights(ref2va_sample_prompt)["qwen3vl_32b"][0]
    r2v_ids = [int(e[0]) for e in r2v_entries]
    sm_ref2va = h3_spans.span_masks_from_prompt(
        ref2va_sample_prompt,
        len(r2v_entries),
        tokenizer=tok,
    )
    im_look_r2v = h3_spans.inference_mask(sm_ref2va, "look_only")
    im_scene_r2v = h3_spans.inference_mask(sm_ref2va, "scene_only")
    im_desc_r2v = h3_spans.inference_mask(sm_ref2va, "description_only")
    im_sound_r2v = h3_spans.inference_mask(sm_ref2va, "soundscape_only")
    if not bool(im_look_r2v.any()) or not bool(im_desc_r2v.any()) or not bool(im_sound_r2v.any()):
        errs.append("look_only, description_only, or soundscape_only empty on ref2va prompt")
    if not torch.equal(im_scene_r2v, im_look_r2v | im_desc_r2v):
        errs.append("scene_only != look_only | description_only on ref2va prompt")
    if bool((im_look_r2v & im_desc_r2v).any()) or bool((im_scene_r2v & im_sound_r2v).any()):
        errs.append("look_only, description_only, and soundscape_only overlapped on ref2va prompt")
    sound_r2v_txt = "".join(decode([r2v_ids[k]]) for k in im_sound_r2v.nonzero(as_tuple=False).flatten().tolist())
    if "solo cello" not in sound_r2v_txt or "room tone" not in sound_r2v_txt:
        errs.append(f"soundscape_only did not include both overall_soundscape and non-'N/A' non_diegetic_music on ref2va prompt: {sound_r2v_txt!r}")
    if int(im_scene_r2v.sum().item()) + int(im_sound_r2v.sum().item()) >= int(sm_ref2va["scene"].sum().item()):
        errs.append("scene_only + soundscape_only did not exclude section headers / structural tags on ref2va prompt")
    sm_ref2va_pref = h3_spans.span_masks_from_prompt(
        ref2va_sample_prompt,
        len(r2v_entries) + 3,
        tags=torch.tensor([0, 0, 1] + [1] * len(r2v_entries), dtype=torch.long),
        tokenizer=tok,
    )
    im_look_r2v_pref = h3_spans.inference_mask(sm_ref2va_pref, "look_only")
    im_scene_r2v_pref = h3_spans.inference_mask(sm_ref2va_pref, "scene_only")
    if int(im_look_r2v_pref.sum().item()) != int(im_look_r2v.sum().item()) or bool(im_look_r2v_pref[:3].any()):
        errs.append("span_masks_from_prompt did not prefix-pad _h3_look_mask cleanly on prefixed ref2va prompt")
    if not torch.equal(im_scene_r2v_pref[3:], im_scene_r2v):
        errs.append("span_masks_from_prompt prefixed scene_only suffix mismatch on ref2va prompt")
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

    # differing_token_bounds and pooling_mask(mode="diff_span"):
    # Mutate two scene tokens mid-prompt while keeping the trailing overall_soundscape / non_diegetic_music suffix identical.
    scene_indices = masks["scene"].nonzero(as_tuple=False).flatten().tolist()
    mid_start = scene_indices[5]
    mid_end_a = mid_start + 2
    ids_b_mid = (
        list(token_ids[:mid_start])
        + [token_ids[mid_start] + 1, token_ids[mid_start + 1] + 1, token_ids[mid_start + 1] + 2]
        + list(token_ids[mid_end_a:])
    )
    fb_start, fb_end_a, fb_end_b = h3_spans.differing_token_bounds(token_ids, ids_b_mid)
    if (fb_start, fb_end_a, fb_end_b) != (mid_start, mid_end_a, mid_start + 3):
        errs.append(
            f"differing_token_bounds returned {(fb_start, fb_end_a, fb_end_b)}, "
            f"expected {(mid_start, mid_end_a, mid_start + 3)}"
        )
    pm_diff_span = h3_spans.pooling_mask(masks, "diff_span", first_diff=fb_start, end_diff=fb_end_a)
    if bool(pm_diff_span[:fb_start].any()) or bool(pm_diff_span[fb_end_a:].any()):
        errs.append("pooling_mask('diff_span') did not mask out shared prefix and trailing shared suffix tokens")
    if not torch.equal(pm_diff_span[fb_start:fb_end_a], masks["scene"][fb_start:fb_end_a]):
        errs.append("pooling_mask('diff_span') did not match masks['scene'] inside [first_diff:end_diff]")

    # Dialogue-only diff fallback in pooling_mask(mode="diff_span"):
    dlg_indices = masks["dialogue"].nonzero(as_tuple=False).flatten().tolist()
    dlg_first, dlg_last = dlg_indices[0], dlg_indices[-1] + 1
    pm_dlg_diff = h3_spans.pooling_mask(masks, "diff_span", first_diff=dlg_first, end_diff=dlg_last)
    if bool(pm_dlg_diff[:dlg_first].any()) or bool(pm_dlg_diff[dlg_last:].any()):
        errs.append("pooling_mask('diff_span') dialogue fallback leaked outside [dlg_first:dlg_last]")
    if not bool(pm_dlg_diff[dlg_first:dlg_last].all()):
        errs.append("pooling_mask('diff_span') dialogue fallback did not select differing dialogue tokens")

    # Multi-span SequenceMatcher differing_token_mask(default min_shared_run=8) and pooling_mask(mode="diff_span") bridge exclusion:
    # Construct two edits separated by 9 unchanged scene tokens (>= 8) so default min_shared_run=8 strips the interior bridge.
    if len(scene_indices) >= 18:
        s_i0 = scene_indices[2]
        s_i1 = s_i0 + 11  # 2 edited tokens + 9 unchanged bridge tokens
        ids_b_multi = list(token_ids)
        ids_b_multi[s_i0] = ids_b_multi[s_i0] + 1
        ids_b_multi[s_i0 + 1] = ids_b_multi[s_i0 + 1] + 1
        ids_b_multi[s_i1] = ids_b_multi[s_i1] + 1
        ids_b_multi[s_i1 + 1] = ids_b_multi[s_i1 + 1] + 1
        dm_a, dm_b = h3_spans.differing_token_mask(token_ids, ids_b_multi)
        if not (bool(dm_a[s_i0 : s_i0 + 2].all()) and bool(dm_a[s_i1 : s_i1 + 2].all())):
            errs.append("differing_token_mask did not mark both edited spans as True")
        if bool(dm_a[s_i0 + 2 : s_i1].any()) or bool(dm_b[s_i0 + 2 : s_i1].any()):
            errs.append("differing_token_mask failed to exclude unchanged interior bridge of 9 tokens (min_shared_run=8)")
        mb_s, mb_ea, _mb_eb = h3_spans.differing_token_bounds(token_ids, ids_b_multi)
        pm_multi = h3_spans.pooling_mask(masks, "diff_span", first_diff=mb_s, end_diff=mb_ea)
        if bool(pm_multi[s_i0 + 2 : s_i1].any()):
            errs.append("pooling_mask('diff_span') did not exclude unchanged interior bridge cached by differing_token_bounds")
        if not (bool(pm_multi[s_i0 : s_i0 + 2].all()) and bool(pm_multi[s_i1 : s_i1 + 2].all())):
            errs.append("pooling_mask('diff_span') lost edited spans when excluding interior bridge")
        # Verify single-span pair with identical outer bounds evicts/replaces the cached multi-span mask:
        ids_b_single_same_bounds = list(token_ids)
        for idx_k in range(s_i0, s_i1 + 2):
            ids_b_single_same_bounds[idx_k] = ids_b_single_same_bounds[idx_k] + 1
        sb_s, sb_ea, _sb_eb = h3_spans.differing_token_bounds(token_ids, ids_b_single_same_bounds)
        pm_single_after = h3_spans.pooling_mask(masks, "diff_span", first_diff=sb_s, end_diff=sb_ea)
        if not torch.equal(pm_single_after[sb_s:sb_ea], masks["scene"][sb_s:sb_ea]):
            errs.append("pooling_mask('diff_span') retained stale multi-span cache after single-span pair with same outer bounds")

    errs += _expect_refusal(
        lambda: h3_spans.differing_token_mask(token_ids, list(token_ids)),
        "are identical; no differing token exists",
        "differing_token_mask identical sequences",
    )
    errs += _expect_refusal(
        lambda: h3_spans.differing_token_bounds(token_ids, list(token_ids)),
        "are identical; no differing token exists",
        "differing_token_bounds identical sequences",
    )
    errs += _expect_refusal(
        lambda: h3_spans.pooling_mask(masks, "diff_span", first_diff=None),
        "requires a non-negative first_diff",
        "pooling_mask('diff_span') missing first_diff",
    )
    errs += _expect_refusal(
        lambda: h3_spans.pooling_mask(masks, "diff_span", first_diff=10, end_diff=10),
        "requires end_diff > first_diff",
        "pooling_mask('diff_span') end_diff <= first_diff",
    )

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

    # Verify that after the last vision block, only `<Audio j>: ` is accepted as a prefix-tail
    # label, whereas `<Picture 2>: ` at the start of the prompt prose stays in `scene`.
    tail_prompt = "<Picture 2>: the painting on the wall."
    tail_entries = tok.tokenize_with_weights(
        tail_prompt,
        minimax_ref_items=[{"type": "image", "data": torch.zeros((1, 64, 64, 3))}],
    )["qwen3vl_32b"][0]
    tail_einfo: list[dict] = []
    tail_pos = 0
    for it, _w in tail_entries:
        if isinstance(it, dict):
            tail_einfo.append({"type": "image", "index": tail_pos, "size": 3})
            tail_pos += 3
        else:
            tail_pos += 1
    tail_tags = token_tags_from_embeds_info(tail_pos, tail_einfo).tolist()
    tail_ids = h3_spans.expand_token_ids(tail_entries, tail_tags)
    tail_masks = h3_spans.spans(tail_ids, tail_tags, decode)
    tail_label_str = "".join(
        decode([tail_ids[k]]) for k in tail_masks["label"].nonzero(as_tuple=False).flatten().tolist()
    )
    if tail_label_str != "<Picture 1>: ":
        errs.append(f"tail `<Picture 2>: ` in prose was misclassified as label: {tail_label_str!r}")

    # Verify before_vision lone-space boundary check (`<Picture 1>: Word` where `" Word"` is a single token,
    # not a lone `" "` token) and before_vision timestamp end-of-segment check (`<0.5 seconds> extra` before vision).
    bv_pieces = [
        "<Picture 1>:",
        " Word ",
        "<0.5 seconds>",
        " extra",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
    ]
    bv_ids = list(range(len(bv_pieces)))
    bv_tags = [1, 1, 1, 1, 0, 0, 0]
    bv_masks = h3_spans.spans(
        bv_ids, bv_tags, lambda s_ids: "".join(bv_pieces[i] for i in s_ids)
    )
    if bool(bv_masks["label"].any()):
        errs.append("before_vision `<Picture 1>: Word ` without lone space token was misclassified as label")
    if bool(bv_masks["timestamp"].any()):
        errs.append("before_vision `<0.5 seconds> extra` not at end of segment was misclassified as timestamp")
    return errs


def check_pair_file_loader_and_refusals() -> list[str]:
    errs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h3_cv_pairs_") as tmp:
        tmp_dir = Path(tmp)
        bank_dir = tmp_dir / "prompt_bank"
        bank_dir.mkdir()
        excl_text = (REPO / "prompt_bank" / "t2va_offpath_tortoise.txt").read_text(encoding="utf-8").strip()
        (bank_dir / "t2va_offpath_tortoise.txt").write_text(excl_text, encoding="utf-8")

        spec_typical_txt = (REPO / "prompt_bank" / "t2va_spec_typical.txt").read_text(encoding="utf-8").strip()
        base_a = spec_typical_txt.replace(
            "Live-action, cinematic,",
            "Live-action, cinematic, shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
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
            "The camera trucks left with small amplitude at slow speed,",
            "The camera pans rapidly to the right,",
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
        # 12. Paragraph break (\n\n) inside differing span, and section/[Shot N] header rename refusals
        errs += _expect_refusal(
            lambda: cv.differing_span(
                base_a,
                base_a.replace("clean neutral colour", "vintage 16mm,\n\nextra paragraph"),
            ),
            "paragraph break",
            "differing_span refusal: paragraph break inside clause",
        )
        for hdr_old, hdr_new in (
            ("overall_soundscape:", "overall_soundscapes:"),
            ("[Shot 1]", "[Shot 2]"),
        ):
            errs += _expect_refusal(
                lambda ho=hdr_old, hn=hdr_new: cv.differing_span(
                    base_a, base_a.replace(ho, hn, 1)
                ),
                "header",
                f"differing_span refusal: header rename {hdr_old!r} -> {hdr_new!r}",
            )
        # 13. Unknown top-level key, unknown [[hold]] key, and non-finite [[hold]] weight
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_top_key",
                'extra_top = 1\n'
                + good_toml.read_text(encoding="utf-8").replace(
                    'name = "film_16mm_1974"', 'name = "bad_top_key"'
                ),
            ),
            "unknown top-level toml key",
            "pair-file refusal: unknown top-level key",
        )
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_hold_key",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_hold_key"')
                .replace('group = "warm_tint"', 'group = "warm_tint"\nextra_hold = 1'),
            ),
            "unknown key",
            "pair-file refusal: unknown [[hold]] key",
        )
        errs += _expect_refusal(
            lambda: _write_and_load(
                "bad_hold_weight",
                good_toml.read_text(encoding="utf-8")
                .replace('name = "film_16mm_1974"', 'name = "bad_hold_weight"')
                .replace('group = "warm_tint"', 'group = "warm_tint"\nweight = nan'),
            ),
            "weight must be finite and non-zero",
            "pair-file refusal: NaN [[hold]] weight",
        )

        # 14. Markdown-to-TOML parser and writer (fenced + raw, auto-mode, auto-sides/quality, and refusals)
        md_test_scene = spec_typical_txt
        md_raw = (
            f"### Test Scene Prompt\n\n{md_test_scene}\n\n"
            f"### Contrastive Pairs (1–2)\n\n"
            f"#### Pair 1\n##### Prompt A (Baseline)\n{base_a}\n\n##### Prompt B (Steered)\n{base_b}\n\n"
            f"#### Pair 2\n##### Prompt A (Baseline)\n{hold_a}\n\n##### Prompt B (Steered)\n{hold_b}\n\n"
            f"### Hold Pairs (Optional)\n\n"
            f"#### Hold 1 (warm_tint)\n##### Prompt A (Baseline)\n{hold_a}\n\n##### Prompt B (Nuisance)\n{hold_b}\n"
        )
        md_fenced = f"```text\n{md_raw}\n```\n"
        parsed_raw = cv.parse_markdown_pairs(md_raw)
        parsed_fenced = cv.parse_markdown_pairs(md_fenced)
        if parsed_raw != parsed_fenced:
            errs.append("parse_markdown_pairs differed between fenced and unfenced markdown")
        if (
            parsed_raw["mode"] != "t2va"
            or len(parsed_raw["pairs"]) != 2
            or len(parsed_raw.get("holds", [])) != 1
            or parsed_raw["holds"][0]["group"] != "warm_tint"
        ):
            errs.append(f"parse_markdown_pairs unexpected mode/pairs/holds: {parsed_raw}")
        if parsed_raw["test_scene_prompt"] != md_test_scene:
            errs.append(f"parse_markdown_pairs test_scene_prompt mismatch: {parsed_raw['test_scene_prompt']!r}")

        ref_md = md_raw.replace(
            "### Test Scene Prompt\n\n",
            "### Test Scene Prompt\n\nsubject_definitions:\n<Subject 1> is the woman in <Picture 1>.\n\n",
        )
        if cv.parse_markdown_pairs(ref_md)["mode"] != "ref2va":
            errs.append("parse_markdown_pairs did not auto-detect ref2va from subject_definitions:")

        md_file = tmp_dir / "from_md_sample.md"
        md_file.write_text(md_fenced, encoding="utf-8")
        tp_out = tmp_dir / "test_scene_out.txt"
        out_toml = cv.write_markdown_to_toml(
            md_file,
            tmp_dir / "from_md_sample.toml",
            test_prompt_out=tp_out,
            bank_dir=bank_dir,
        )
        if not out_toml.is_file() or not tp_out.is_file():
            errs.append("write_markdown_to_toml did not write .toml or test_prompt_out")
        else:
            if tp_out.read_text(encoding="utf-8").strip() != md_test_scene:
                errs.append("write_markdown_to_toml test_prompt_out content mismatch")
            if "# Test Scene Prompt:" not in out_toml.read_text(encoding="utf-8"):
                errs.append("write_markdown_to_toml omitted '# Test Scene Prompt:' comment header")
            loaded_md_ps = cv.load_pairs(out_toml, bank_dir=bank_dir)
            if (
                loaded_md_ps.name != "from_md_sample"
                or loaded_md_ps.mode != "t2va"
                or len(loaded_md_ps.pairs) != 2
                or len(loaded_md_ps.holds) != 1
                or loaded_md_ps.holds[0].group != "warm_tint"
                or loaded_md_ps.a_side != pset.pairs[0].a_span
                or loaded_md_ps.b_side != pset.pairs[0].b_span
                or loaded_md_ps.quality != f"{pset.pairs[0].b_span} vs. {pset.pairs[0].a_span}"
            ):
                errs.append(f"write_markdown_to_toml loaded PairSet mismatch: {loaded_md_ps}")

        errs += _expect_refusal(
            lambda: cv.write_markdown_to_toml(
                md_file, tmp_dir / "from_md_sample.toml", overwrite=False, bank_dir=bank_dir
            ),
            "already exists",
            "write_markdown_to_toml refusal when output .toml already exists and overwrite=False",
        )
        errs += _expect_refusal(
            lambda: cv.parse_markdown_pairs("### Test Scene Prompt\n\nJust a scene prompt without pairs."),
            "no '#### pair n' blocks",
            "parse_markdown_pairs refusal when no Pair blocks exist",
        )
        errs += _expect_refusal(
            lambda: cv.parse_markdown_pairs(
                "#### Pair 1\n##### Prompt A (Baseline)\n" + base_a
            ),
            "missing '##### prompt b'",
            "parse_markdown_pairs refusal when Prompt B is missing",
        )
        errs += _expect_refusal(
            lambda: cv.parse_markdown_pairs(
                "#### Pair 1\n##### Prompt B (Steered)\n" + base_b
            ),
            "missing '##### prompt a'",
            "parse_markdown_pairs refusal when Prompt A is missing",
        )
        errs += _expect_refusal(
            lambda: cv.write_markdown_to_toml(
                f"#### Pair 1\n##### Prompt A\n{base_a}\n\n##### Prompt B\n{leaked_b}\n",
                tmp_dir / "bad_md_multispan.toml",
                bank_dir=bank_dir,
            ),
            "multiple separate spans",
            "write_markdown_to_toml refusal when markdown pair has multiple separate spans",
        )
        if (tmp_dir / "bad_md_multispan.toml").exists():
            errs.append("write_markdown_to_toml left invalid .toml on disk after validation failure")

        # 15. Semantic pair-quality linter (lint_pair_set) checks:
        # Clean pair set with unique spans and N/A music -> 0 warnings
        clean_p1_a = base_a
        clean_p1_b = base_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "recorded on vintage 16mm celluloid stock with heavy grain and halation,",
        )
        clean_lint = cv.lint_pair_set([
            {"a": clean_p1_a, "b": clean_p1_b},
            {"a": hold_a, "b": hold_b},
        ])
        if clean_lint:
            errs.append(f"lint_pair_set produced unexpected warnings on clean pairs: {clean_lint}")

        # Duplicate span, unbracketed <d>, banned rate adverb in b_span, content-noun cancellation, and non-N/A music
        bad_dlg_a = base_a.replace("She produces no vocal sound.", 'She says, <d>Hello there.</d>')
        bad_dlg_b = bad_dlg_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "rapidly recorded on 16mm film with a lantern on the bench,",
        )
        bad_music_a = base_a.replace("non_diegetic_music: N/A", "non_diegetic_music: No non-diegetic music.")
        bad_music_b = clean_p1_b.replace("non_diegetic_music: N/A", "non_diegetic_music: No non-diegetic music.")
        noun_cancel_a = base_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "with a brass lantern glowing on the oak table,",
        )
        noun_cancel_b = base_a.replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "with a silver lantern gleaming on the stone shelf,",
        )
        diff_scene_a = base_a.replace("a woman walking along a beach", "a watchmaker seated at a bench").replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "filmed under neutral studio lighting,",
        )
        diff_scene_dup_b = base_a.replace("a woman walking along a beach", "a watchmaker seated at a bench").replace(
            "shot on a modern digital cinema camera with clean neutral colour and crisp detail,",
            "recorded on vintage 16mm celluloid stock with heavy grain and halation,",
        )
        lint_warns = cv.lint_pair_set([
            {"a": clean_p1_a, "b": clean_p1_b},
            {"a": clean_p1_a, "b": clean_p1_b},  # duplicate a_span / b_span in same scene
            {"a": diff_scene_a, "b": diff_scene_dup_b},  # duplicate b_span across a DIFFERENT scene
            {"a": bad_dlg_a, "b": bad_dlg_b},  # missing [Language] tag + banned rate adverb 'rapidly'
            {"a": bad_music_a, "b": bad_music_b},  # non-N/A music
            {"a": noun_cancel_a, "b": noun_cancel_b},  # content-noun cancellation ('lantern')
        ])
        for expected_substr in (
            "duplicate a_span",
            "Pair 3: duplicate b_span",
            "missing a [Language] tag",
            "rate/onset filler",
            "non_diegetic_music",
            "cancel in B - A",
        ):
            if not any(expected_substr in w for w in lint_warns):
                errs.append(f"lint_pair_set did not flag {expected_substr!r}; got {lint_warns}")

        # 16. h3_pair_compiler in-place target_prompt contrast compiler checks (canonical_frames.json retired)
        if (REPO / "prompt_bank" / "control_vectors" / "canonical_frames.json").exists():
            errs.append("prompt_bank/control_vectors/canonical_frames.json still exists (must be retired)")
        for retired_sym in ("load_canonical_frames", "CANONICAL_SLOTS", "_PLACEBO_CONTRASTS", "infer_slot_from_words"):
            if hasattr(pair_comp, retired_sym):
                errs.append(f"h3_pair_compiler still defines retired symbol {retired_sym!r}")

        origami_fl2va_txt = (REPO / "prompt_bank" / "fl2va_origami_crane.txt").read_text(encoding="utf-8").strip()
        origami_fl2va_na = origami_fl2va_txt.replace(
            "non_diegetic_music: A music-box figure at a moderate tempo, a bright repeating melody over a plucked bass note, slowing slightly and stopping on the final fold.",
            "non_diegetic_music: N/A",
        )
        diner_ref2va_txt = (REPO / "prompt_bank" / "ref2va_diner_breakup_refs.txt").read_text(encoding="utf-8").strip()
        diner_ref2va_na = diner_ref2va_txt.replace(
            "non_diegetic_music:\nA bittersweet, nostalgic slow dance melody at a slow tempo, featuring a warm hollow-body electric jazz guitar, lush upright bass accompaniment, and a muted vibraphone that gently decrescendos to silence.",
            "non_diegetic_music:\nN/A",
        )
        clockmaker_img_txt = (REPO / "prompt_bank" / "p_h3_image" / "t2i_clockmaker.txt").read_text(encoding="utf-8").strip()
        mode_target_prompts = {
            "t2va": (spec_typical_txt, "a woman walking along a beach at sunset -> a traveler trudging across a snowfield at dusk"),
            "fl2va": (origami_fl2va_na, "an overhead shot straight down onto a pale wooden tabletop -> a low-angle shot across a dark slate tabletop"),
            "ref2va": (diner_ref2va_na, "Photorealistic live-action, 16:9, cinematic, shallow depth of field -> 1970s 16mm celluloid film with heavy grain and soft halation"),
            "p_h3_image": (clockmaker_img_txt, "an elderly clockmaker seated at a worn walnut workbench -> a young botanist standing beside a mossy stone bench"),
        }
        for mode_k, (tp_mode, clause_mode) in mode_target_prompts.items():
            ps_m = pair_comp.compile_words_to_pairset(
                clause_mode,
                mode=mode_k,
                target_prompt=tp_mode,
            )
            if len(ps_m.pairs) != 1 or ps_m.mode != mode_k or ps_m.frame != "scene":
                errs.append(
                    f"compile_words_to_pairset(mode={mode_k!r}) unexpected pairs/mode/frame: len={len(ps_m.pairs)}, mode={ps_m.mode!r}, frame={ps_m.frame!r}"
                )
            m_lint = cv.lint_pair_set(ps_m)
            if m_lint:
                errs.append(f"compile_words_to_pairset(mode={mode_k!r}) produced lint warnings: {m_lint}")
        errs += _expect_refusal(
            lambda: pair_comp.compile_words_to_pairset(
                "an elderly clockmaker seated at a worn walnut workbench -> an elderly botanist seated at a mossy stone bench; soundscape: quiet room tone -> muffled through-the-wall acoustics",
                mode="p_h3_image",
                target_prompt=clockmaker_img_txt,
            ),
            "single-frame image generation",
            "compile_words_to_pairset refusal on soundscape clause in p_h3_image mode",
        )
        errs += _expect_refusal(
            lambda: pair_comp.compile_words_to_pairset("a woman walking along a beach -> a traveler walking"),
            "target_prompt is required",
            "compile_words_to_pairset refusal when target_prompt is omitted",
        )
        errs += _expect_refusal(
            lambda: pair_comp.compile_words_to_pairset(
                "nonexistent clause in prompt -> another clause",
                target_prompt=spec_typical_txt,
            ),
            "was found as a substring in target_prompt",
            "compile_words_to_pairset refusal when neither pole matches target_prompt",
        )
        # Duplicate pole_a refusal when clause appears more than once in target_prompt:
        errs += _expect_refusal(
            lambda: pair_comp.compile_words_to_pairset(
                "<Subject 1> -> <Subject 9>",
                target_prompt=diner_ref2va_txt,
            ),
            "occurs ",
            "compile_words_to_pairset refusal when pole_a appears multiple times in target_prompt",
        )
        # Semicolon inside prose must not split a single clause:
        ps_semi = pair_comp.compile_words_to_pairset(
            "a woman walking along a beach at sunset -> a woman walking; then pausing along a beach at sunset",
            target_prompt=spec_typical_txt,
        )
        if len(ps_semi.pairs) != 1 or "walking; then pausing" not in ps_semi.pairs[0].b:
            errs.append(f"compile_words_to_pairset split semicolon inside prose: {ps_semi.pairs}")
        # 3-clause Diner ref2va recipe (2 description clauses across detailed_description + 1 soundscape clause) compiles cleanly:
        ps_diner_3c = pair_comp.compile_words_to_pairset(
            "Photorealistic live-action, 16:9, cinematic, shallow depth of field -> 2D hand-drawn cel anime with crisp black ink outlines; "
            "A cup of black coffee sits in front of him -> A glowing hologram sits in front of him; "
            "soundscape: Steady rain taps against glass somewhere behind the camera -> Stylized anime rain patter with dramatic wind",
            name="compiled_diner_3c",
            target_prompt=diner_ref2va_txt,
        )
        if len(ps_diner_3c.pairs) != 1 or ps_diner_3c.pairs[0].b_span.count(" | ") != 2:
            errs.append(f"3-clause Diner ref2va recipe failed multi-span compilation: {ps_diner_3c.pairs[0].b_span!r}")
        compiled_multi_toml = tmp_dir / "compiled_diner_3c.toml"
        pair_comp.write_compiled_toml(ps_diner_3c, compiled_multi_toml, overwrite=True)
        loaded_diner_3c = cv.load_pairs(compiled_multi_toml)
        if len(loaded_diner_3c.pairs) != 1 or loaded_diner_3c.pairs[0].b_span.count(" | ") != 2:
            errs.append("load_pairs failed to round-trip compiled multi-span .toml")

        if pair_comp.infer_compile_mode("auto", target_prompt=diner_ref2va_txt) != "ref2va":
            errs.append("infer_compile_mode did not detect ref2va from subject_definitions:")
        if pair_comp.infer_compile_mode("auto", target_prompt=origami_fl2va_txt) != "fl2va":
            errs.append("infer_compile_mode did not detect fl2va from official base_en.md Part One header")
        if pair_comp.infer_compile_mode("auto", target_prompt=clockmaker_img_txt) != "p_h3_image":
            errs.append("infer_compile_mode did not detect p_h3_image from sectionless image prompt")
        import control_vector_nodes as cv_nodes
        if hasattr(cv_nodes, "DEFAULT_MEDIA_SCENES"):
            errs.append("control_vector_nodes still defines DEFAULT_MEDIA_SCENES (must be removed)")
        # Direct in-place --target-prompt compilation (both when Pole A is in target_prompt and when Pole B is in target_prompt):
        inplace_tp = spec_typical_txt
        ps_inplace_a = pair_comp.compile_words_to_pairset(
            "a medium shot frames a woman walking along a beach at sunset -> a shaky handheld VHS camcorder shot frames a woman walking along a beach at sunset",
            target_prompt=inplace_tp,
        )
        ps_inplace_b = pair_comp.compile_words_to_pairset(
            "a shaky handheld VHS camcorder shot frames a woman walking along a beach at sunset -> a medium shot frames a woman walking along a beach at sunset",
            target_prompt=inplace_tp,
        )
        if len(ps_inplace_a.pairs) != 1 or len(ps_inplace_b.pairs) != 1:
            errs.append(f"direct in-place target_prompt compilation expected 1 pair, got {len(ps_inplace_a.pairs)} and {len(ps_inplace_b.pairs)}")
        if "a shaky handheld VHS camcorder shot" not in ps_inplace_a.pairs[0].b or "a shaky handheld VHS camcorder shot" not in ps_inplace_b.pairs[0].a:
            errs.append("direct in-place target_prompt compilation did not substitute exact clause in place")
        if not ps_inplace_a.name.startswith("shaky_handheld_vhs_camcorder"):
            errs.append(f"compile_words_to_pairset('a -> b') did not derive default name from steered pole B: got {ps_inplace_a.name!r}")

        # Coupled description + overall_soundscape differing_span and inline hold: compilation on target_prompt:
        coupled_b = base_b.replace(
            "overall_soundscape: Gentle waves wash onto the sand and draw back under a soft sea breeze, with the distant cry of gulls.",
            "overall_soundscape: A 16mm film projector clatters with optical crackle and warm motor hum.",
        )
        c_span_a, c_span_b = cv.differing_span(base_a, coupled_b, frame="scene")
        if " | " not in c_span_a or " | " not in c_span_b:
            errs.append(f"differing_span did not join coupled description + soundscape spans with ' | ': {c_span_a!r} vs {c_span_b!r}")
        _, c_sec_name = cv.differing_section_info(base_a, coupled_b, frame="scene")
        if c_sec_name != "integrated_multimodal_description+overall_soundscape":
            errs.append(f"differing_section_info returned {c_sec_name!r}, expected 'integrated_multimodal_description+overall_soundscape'")

        spec_multi = pair_comp.parse_short_contrast_spec(
            "a medium shot frames a woman -> a shaky handheld VHS camcorder shot frames a woman; "
            "soundscape: Gentle waves wash onto the sand -> A 16mm film projector clatters with optical crackle; "
            "hold(camera_shake): walking along a beach -> running along a beach",
            target_prompt=inplace_tp,
            target_weight=2.5,
        )
        ps_multi = spec_multi["pair_set"]
        if len(ps_multi.pairs) != 1 or ps_multi.pairs[0].weight != 2.5 or len(ps_multi.holds) != 1 or ps_multi.holds[0].group != "camera_shake":
            errs.append(f"parse_short_contrast_spec multi-clause/coupled/hold unexpected result: pairs={len(ps_multi.pairs)}, holds={len(ps_multi.holds)}")
        if " | " not in ps_multi.pairs[0].b_span:
            errs.append("parse_short_contrast_spec with coupled soundscape did not produce 'description | soundscape' b_span")
        if cv.lint_pair_set(ps_multi):
            errs.append(f"parse_short_contrast_spec multi-clause PairSet failed lint_pair_set: {cv.lint_pair_set(ps_multi)}")
        # Multi-section summary: + detailed_description: and soundscape: + music: compilation & differing_span:
        ps_r2v_summary_music = pair_comp.compile_words_to_pairset(
            "summary: crossings and rain -> neon cyberpunk rain; "
            "Photorealistic live-action -> Stylized cel-shaded anime; "
            "soundscape: Steady rain taps against glass -> Heavy storm rain hammers glass; "
            "music: N/A -> A tense analog synth pulse underscores the scene",
            target_prompt=diner_ref2va_txt,
        )
        if len(ps_r2v_summary_music.pairs) != 1 or ps_r2v_summary_music.pairs[0].b_span.count(" | ") != 3:
            errs.append(f"4-section summary+desc+soundscape+music compilation unexpected spans: {ps_r2v_summary_music.pairs[0].b_span!r}")
        _, sec_4way = cv.differing_section_info(ps_r2v_summary_music.pairs[0].a, ps_r2v_summary_music.pairs[0].b, frame="scene")
        if sec_4way != "summary+detailed_description+overall_soundscape+non_diegetic_music":
            errs.append(f"differing_section_info 4-way returned {sec_4way!r}")
        # --bank-scenes <K> multi-scene auxiliary prompt_bank expansion (direct + auxiliary splice + holds):
        ps_bank_t2va = pair_comp.compile_words_to_pairset(
            "a medium shot frames a woman -> a shaky handheld VHS camcorder shot frames a woman; hold(camera_shake): walking along a beach -> running along a beach",
            target_prompt=inplace_tp,
            target_weight=2.5,
            bank_scenes=3,
        )
        if len(ps_bank_t2va.pairs) != 4 or ps_bank_t2va.pairs[0].weight != 2.5 or ps_bank_t2va.pairs[1].weight != 1.0:
            errs.append(f"bank_scenes=3 t2va unexpected pairs/weights: len={len(ps_bank_t2va.pairs)}")
        if len(ps_bank_t2va.holds) != 4:
            errs.append(f"bank_scenes=3 t2va holds did not expand across all 4 scenes: got {len(ps_bank_t2va.holds)}")
        if cv.lint_pair_set(ps_bank_t2va):
            errs.append(f"bank_scenes=3 t2va PairSet produced lint warnings: {cv.lint_pair_set(ps_bank_t2va)}")
        ps_bank_r2v = pair_comp.compile_words_to_pairset(
            "Photorealistic live-action -> Stylized cel-shaded anime",
            target_prompt=diner_ref2va_txt,
            bank_scenes=2,
        )
        if len(ps_bank_r2v.pairs) != 3 or ps_bank_r2v.mode != "ref2va":
            errs.append(f"bank_scenes=2 ref2va unexpected result: len={len(ps_bank_r2v.pairs)}, mode={ps_bank_r2v.mode!r}")
        # Verify _select_bank_scenes calls differing_span on both literal (is_direct) and
        # auxiliary-splice (ok_aux) candidates and rejects >= MAX_SHARED_INTERIOR_WORDS (5):
        bad_bank_dir = tmp_dir / "bad_bank_scenes"
        bad_bank_dir.mkdir(parents=True, exist_ok=True)
        (bad_bank_dir / "t2va_01_direct_bad.txt").write_text(
            "integrated_multimodal_description: [Shot 1] A red brick lane in crisp bright noon.\n"
            "overall_soundscape: Quiet street ambience.\n"
            "non_diegetic_music: N/A\n",
            encoding="utf-8",
        )
        (bad_bank_dir / "t2va_02_splice_bad.txt").write_text(
            "integrated_multimodal_description: [Shot 1] A harbour pier at dawn.\n"
            "overall_soundscape: Gentle courtyard breeze.\n"
            "non_diegetic_music: N/A\n",
            encoding="utf-8",
        )
        sel_skip = pair_comp._select_bank_scenes(
            mode="t2va",
            bank_scenes=2,
            existing_scenes=(),
            clauses=(
                pair_comp.ContrastClause(
                    a="red brick lane in crisp bright noon",
                    b="blue brick lane in crisp bright dusk",
                ),
            ),
            bank_dir=bad_bank_dir,
        )
        if sel_skip != []:
            errs.append(
                f"_select_bank_scenes accepted candidate(s) violating MAX_SHARED_INTERIOR_WORDS=5: got {sel_skip!r}"
            )
        # Verify h3_pair_compiler stays pure prompt text (never reading files from disk),
        # while CLI --target-prompt-file resolves file paths and prompt_bank stems explicitly (checked in [C7]).
        porter_path = REPO / "prompt_bank" / "t2va_night_porter.txt"
        errs += _expect_refusal(
            lambda: pair_comp.compile_words_to_pairset(
                "A night porter in scrubs and a fleece -> A NASA astronaut in a bulky white EVA spacesuit",
                target_prompt=str(porter_path),
            ),
            "was found as a substring in target_prompt",
            "compile_words_to_pairset refuses file paths (compiler is pure text)",
        )
        # Literal --target-prompt matching a file stem in prompt_bank/ stays literal text (no silent file substitution):
        ps_literal_collision = pair_comp.compile_words_to_pairset(
            "t2va_night_porter -> bright sunny courtyard",
            target_prompt="t2va_night_porter",
        )
        if ps_literal_collision.pairs[0].a != "t2va_night_porter" or ps_literal_collision.pairs[0].b != "bright sunny courtyard":
            errs.append(f"literal target_prompt matching prompt_bank stem was altered: {ps_literal_collision.pairs[0]!r}")
        errs += _expect_refusal(
            lambda: pair_comp.compile_words_to_pairset("   ", target_prompt=inplace_tp),
            "non-empty string",
            "compile_words_to_pairset refusal on empty words",
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

    # unit_mean=True test: pair 0 has norm 2 along e0, pair 1 has norm 10 along e1.
    # Unnormalized mean_difference points mostly along e1 ([1.0, 5.0]); unit_mean=True gives equal
    # directional weight (0.5 * e0 + 0.5 * e1) scaled by mean norm (2 + 10)/2 = 6 -> [3.0, 3.0].
    a_um = torch.zeros((2, 4), dtype=torch.float32)
    b_um = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]], dtype=torch.float32)
    v_um = cv.mean_difference(a_um, b_um, unit_mean=True)
    expected_um = torch.tensor([3.0, 3.0, 0.0, 0.0], dtype=torch.float32)
    if not torch.allclose(v_um, expected_um, atol=1e-5):
        errs.append(f"mean_difference(unit_mean=True) wrong: {v_um} vs {expected_um}")

    # Synthetic signal-plus-noise pairs: diffs_i = mu + eps_i, ||mu|| = 1, eps_i ~ N(0, sigma^2 / d I).
    gen = torch.Generator().manual_seed(20261104)
    mu = torch.zeros(d, dtype=torch.float64)
    mu[0] = 1.0
    sigma = 0.5
    noise = torch.randn((32, d), generator=gen, dtype=torch.float64) * (sigma / math.sqrt(d))
    diffs32 = (mu.unsqueeze(0) + noise).to(torch.float32)
    zeros32 = torch.zeros_like(diffs32)

    # Exact algebraic cancellation check for flip_null across even n, odd n, positive unequal weights,
    # and mixed-sign weights: for the same generator seed, adding 5 * sign(w_i) * mu to every pair
    # must leave the direction of flip_null unchanged (cosine == 1.0 within float32 tolerance),
    # and flip_null's norm must equal the norm of mean_difference(a, b, w).
    for case_label, n_sub, w_sub in (
        ("even n=12", 12, None),
        ("odd n=9", 9, None),
        ("odd n=11", 11, None),
        ("unequal weights n=8", 8, torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)),
        ("unequal weights n=9", 9, torch.tensor([3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0], dtype=torch.float32)),
        ("mixed-sign weights n=8", 8, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)),
        ("mixed-sign weights n=9", 9, torch.tensor([2.0, -1.0, 3.0, -2.0, 1.0, -1.0, 2.0, -3.0, 1.0], dtype=torch.float32)),
    ):
        sub_noise = noise[:n_sub].to(torch.float32)
        sub_zeros = zeros32[:n_sub]
        signs_w = (
            torch.sign(w_sub).to(dtype=torch.float32)[:, None]
            if w_sub is not None
            else torch.ones((n_sub, 1), dtype=torch.float32)
        )
        sub_planted = sub_noise + 5.0 * signs_w * mu.to(dtype=torch.float32).unsqueeze(0)
        v_sub = cv.mean_difference(sub_zeros, sub_planted, w_sub)
        sub_norm = float(torch.linalg.vector_norm(v_sub.to(torch.float64)).item())
        for draw_idx in range(8):
            seed_val = 1000 + draw_idx
            fn_noise = cv.flip_null(
                sub_zeros,
                sub_noise,
                w_sub,
                generator=torch.Generator().manual_seed(seed_val),
            )
            fn_planted = cv.flip_null(
                sub_zeros,
                sub_planted,
                w_sub,
                generator=torch.Generator().manual_seed(seed_val),
            )
            fn_norm = float(torch.linalg.vector_norm(fn_planted.to(torch.float64)).item())
            if abs(fn_norm - sub_norm) > 1e-5 * sub_norm:
                errs.append(f"flip_null ({case_label}) norm {fn_norm} != {sub_norm}")
                break
            u_noise = fn_noise.to(torch.float64) / torch.linalg.vector_norm(fn_noise.to(torch.float64))
            u_planted = fn_planted.to(torch.float64) / torch.linalg.vector_norm(fn_planted.to(torch.float64))
            cos_invariance = float(torch.dot(u_noise, u_planted).item())
            if abs(cos_invariance - 1.0) > 1e-5:
                errs.append(
                    f"flip_null ({case_label}) leaked planted signal: same-seed invariance cosine = {cos_invariance:.6f} != 1.0"
                )
                break

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

    # split_half_cosine must honor weights: 2 aligned rows with weight 100 vs 2 orthogonal rows with weight 1.
    w_rows = torch.zeros((4, d), dtype=torch.float32)
    w_rows[0, 0] = 1.0
    w_rows[1, 0] = 1.0
    w_rows[2, 1] = 1.0
    w_rows[3, 2] = 1.0
    sh_weighted = cv.split_half_cosine(
        w_rows, weights=[100.0, 100.0, 1.0, 1.0], generator=torch.Generator().manual_seed(7), n_splits=64
    )
    sh_unweighted = cv.split_half_cosine(
        w_rows, generator=torch.Generator().manual_seed(7), n_splits=64
    )
    if abs(sh_weighted - sh_unweighted) < 0.05:
        errs.append(
            f"split_half_cosine ignored weights: sh_weighted={sh_weighted:.4f}, sh_unweighted={sh_unweighted:.4f}"
        )

    # mean_pairwise_cosine and per_pair_cosine must orient rows by sign(weights).
    # Pairs 0, 2 point +e0 with weight +1; pairs 1, 3 point -e0 with weight -1 -> all 4 orient to +e0 (cos = 1.0).
    signed_rows = torch.zeros((4, d), dtype=torch.float32)
    signed_rows[0, 0] = 1.0
    signed_rows[1, 0] = -1.0
    signed_rows[2, 0] = 1.0
    signed_rows[3, 0] = -1.0
    mpc_signed = cv.mean_pairwise_cosine(signed_rows, weights=[1.0, -1.0, 1.0, -1.0])
    if abs(mpc_signed - 1.0) > 1e-6:
        errs.append(f"mean_pairwise_cosine ignored negative weight orientation: {mpc_signed}, expected 1.0")
    ppc_signed = cv.per_pair_cosine(signed_rows, weights=[1.0, -1.0, 1.0, -1.0])
    if any(abs(x - 1.0) > 1e-6 for x in ppc_signed):
        errs.append(f"per_pair_cosine ignored negative weight orientation: {ppc_signed}, expected all 1.0")

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
    v_nan = v_in.clone()
    v_nan[0] = float("nan")
    errs += _expect_refusal(
        lambda: cv.project_out(v_nan, basis),
        "v and basis must contain only finite values",
        "project_out non-finite v refusal",
    )
    basis_nan = basis.clone()
    basis_nan[0, 0] = float("nan")
    errs += _expect_refusal(
        lambda: cv.project_out(v_in, basis_nan),
        "v and basis must contain only finite values",
        "project_out non-finite basis refusal",
    )
    # 1-pair quality_report must return None for mean_pairwise_cosine and split_half_cosine and [] for per_pair_cosine
    qr_1pair = cv.quality_report(a[:1], b[:1])
    if (
        qr_1pair["mean_pairwise_cosine"] is not None
        or qr_1pair["split_half_cosine"] is not None
        or qr_1pair["per_pair_cosine"] != []
    ):
        errs.append(f"quality_report(n_pairs=1) expected None cosines and [], got {qr_1pair}")
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
        errs += _expect_refusal(
            lambda: cv.safetensors_recipe_digest(ckpt_path, extra_tensor_keys=("nonexistent.weight",)),
            "missing required tensor key 'nonexistent.weight'",
            "safetensors_recipe_digest missing tensor key refusal",
        )

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

        if "diff_span" not in cv.VALID_POOLING:
            errs.append(f"'diff_span' missing from cv.VALID_POOLING: {cv.VALID_POOLING}")
        diff_span_fname = cv.default_vector_filename("film_16mm_1974", "fl2va", pooling="diff_span")
        if diff_span_fname != "film_16mm_1974__fl2va__diff_span.safetensors":
            errs.append(f"unexpected default_vector_filename for diff_span: {diff_span_fname!r}")

        sidecar_md_path = vec_path.with_suffix(".md")
        if not sidecar_md_path.is_file():
            errs.append(f"save_vector did not write companion .md sidecar at {sidecar_md_path}")
        else:
            sidecar_md_text = sidecar_md_path.read_text(encoding="utf-8")
            for required_snippet in (
                "film_16mm_1974__fl2va.safetensors",
                "scene_text",
                "minimax_h3_fl2va_int8_convrot.safetensors",
                "Recommended ComfyUI `strength` range",
                "mean_pairwise_cosine",
                "split_half_cosine",
                "## How to Read These Scores",
            ):
                if required_snippet not in sidecar_md_text:
                    errs.append(f"sidecar .md missing {required_snippet!r}")

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

        # - metadata sub-dict inner keys, values, and family format validation
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_pairs.safetensors",
                tensors_in,
                {**meta_in, "pairs": {"path": "x.toml"}},
            ),
            "metadata 'pairs' must be a dict with keys",
            "save_vector refusal: missing inner key in pairs",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_measured_keys.safetensors",
                tensors_in,
                {**meta_in, "measured_under": {"mode": "t2va"}},
            ),
            "metadata 'measured_under' must be a dict with keys",
            "save_vector refusal: missing inner key in measured_under",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_pooling.safetensors",
                tensors_in,
                {**meta_in, "measured_under": {**meta_in["measured_under"], "pooling": "bogus"}},
            ),
            "measured_under.pooling",
            "save_vector refusal: invalid measured_under.pooling",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_mode.safetensors",
                tensors_in,
                {**meta_in, "measured_under": {**meta_in["measured_under"], "mode": "zzz"}},
            ),
            "measured_under.mode",
            "save_vector refusal: invalid measured_under.mode",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_points.safetensors",
                tensors_in,
                {**meta_in, "points": {"post_refiner": {}}},
            ),
            "points['post_refiner'] must be a dict with",
            "save_vector refusal: missing inner keys in points.post_refiner",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_qr_nan.safetensors",
                tensors_in,
                {**meta_in, "quality_report": {**meta_in["quality_report"], "mean_pairwise_cosine": float("nan")}},
            ),
            "finite float",
            "save_vector refusal: NaN mean_pairwise_cosine",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_qr_ppc_nan.safetensors",
                tensors_in,
                {**meta_in, "quality_report": {**meta_in["quality_report"], "per_pair_cosine": [float("nan")]}},
            ),
            "finite floats",
            "save_vector refusal: NaN in per_pair_cosine list",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_fam.safetensors",
                tensors_in,
                {**meta_in, "family": "custom:not_hex_12"},
            ),
            "metadata 'family' must be",
            "save_vector refusal: invalid custom family tag",
        )
        # 1-pair vector save/load round-trip with None mean_pairwise_cosine and split_half_cosine
        meta_1pair = {
            **meta_in,
            "name": "single_pair_sample",
            "pairs": {**meta_in["pairs"], "n_pairs": 1, "n_holds": 0},
            "quality_report": {
                "mean_pairwise_cosine": None,
                "split_half_cosine": None,
                "per_pair_cosine": [1.0],
            },
        }
        vec_1pair_path = tmp_dir / "single_pair_sample__fl2va.safetensors"
        cv.save_vector(vec_1pair_path, {"vector.post_refiner": v_post}, meta_1pair)
        _, meta_1pair_out = cv.load_vector(vec_1pair_path)
        if (
            meta_1pair_out["quality_report"]["mean_pairwise_cosine"] is not None
            or meta_1pair_out["quality_report"]["split_half_cosine"] is not None
        ):
            errs.append(f"1-pair save/load round-trip lost None cosines: {meta_1pair_out['quality_report']}")
        if "n/a (1 pair)" not in vec_1pair_path.with_suffix(".md").read_text(encoding="utf-8"):
            errs.append("1-pair sidecar .md did not format None cosines as 'n/a (1 pair)'")
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_2pair_none_cos.safetensors",
                {"vector.post_refiner": v_post},
                {
                    **meta_in,
                    "pairs": {**meta_in["pairs"], "n_pairs": 2, "n_holds": 0},
                    "quality_report": {
                        "mean_pairwise_cosine": None,
                        "split_half_cosine": 0.5,
                        "per_pair_cosine": [1.0, 1.0],
                    },
                },
            ),
            "when n_pairs >= 2",
            "save_vector refusal: None mean_pairwise_cosine when n_pairs >= 2",
        )

        # - tensor validation: non-orthonormal holds, unprojected vector vs holds, unknown key, valid llm.<layer>
        h_not_ortho = torch.zeros((2, cv.POST_REFINER_DIM), dtype=torch.float32)
        h_not_ortho[0, 0] = 2.0
        h_not_ortho[1, 1] = 1.0
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_holds_ortho.safetensors",
                {"vector.post_refiner": v_post, "holds.post_refiner": h_not_ortho},
                meta_in,
            ),
            "not orthonormal",
            "save_vector refusal: non-orthonormal holds.post_refiner",
        )
        v_not_proj = v_post.clone()
        v_not_proj[0] = 1.0  # h_post[0, 0] == 1.0 -> non-zero dot product
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_vec_proj.safetensors",
                {"vector.post_refiner": v_not_proj, "holds.post_refiner": h_post},
                meta_in,
            ),
            "not orthogonal to 'holds.post_refiner'",
            "save_vector refusal: vector.post_refiner not orthogonal to holds.post_refiner",
        )
        errs += _expect_refusal(
            lambda: cv.save_vector(
                tmp_dir / "bad_holds_enc.safetensors",
                {"vector.post_refiner": v_post, "holds.encoder": torch.zeros((1, cv.ENCODER_DIM), dtype=torch.float32)},
                meta_in,
            ),
            "unknown control vector tensor key 'holds.encoder'",
            "save_vector refusal: unknown tensor key holds.encoder",
        )
        llm_path = tmp_dir / "valid_llm.safetensors"
        cv.save_vector(
            llm_path,
            {"vector.post_refiner": v_post, "vector.llm.25": torch.ones((cv.ENCODER_DIM,), dtype=torch.float32)},
            meta_in,
        )
        t_llm, _ = cv.load_vector(llm_path)
        if "vector.llm.25" not in t_llm or tuple(t_llm["vector.llm.25"].shape) != (cv.ENCODER_DIM,):
            errs.append("save_vector/load_vector failed on valid vector.llm.25 (5120-d) key")

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

        # 4b. Same-family Pika H3 Flex LoRA variants (e.g. svd995_bf16 vs comfy_bf16 at same strength)
        # emit an advisory warning instead of a hard refusal when allow_recipe_mismatch=False.
        meta_pika_svd = {
            **meta_out,
            "recipe": {
                **recipe_a,
                "loras": [
                    {
                        "file": "pika_h3_flex_fl2va_svd995_bf16.safetensors",
                        "sha256": "aaaa",
                        "strength": 1.0,
                        "applied": "branch",
                    }
                ],
            },
        }
        recipe_pika_comfy = {
            **recipe_a,
            "loras": [
                {
                    "file": "pika_h3_flex_fl2va_comfy_bf16.safetensors",
                    "sha256": "bbbb",
                    "strength": 1.0,
                    "applied": "branch",
                }
            ],
        }
        warns_pika_variant = cv.check_vector_compatibility(
            meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_pika_comfy, allow_recipe_mismatch=False
        )
        if len(warns_pika_variant) != 1 or "lora set" not in warns_pika_variant[0].lower():
            errs.append(
                f"expected 1 same-family Pika variant advisory warning when allow_recipe_mismatch=False, got {warns_pika_variant}"
            )
        # Verify Kijai's actual v1_avg_rank_*_bf16 filename and full-refiner branch:... applied string pass,
        # while partial applied strings that exclude token_refiner (blocks=0-24 or modules=mlp only) refuse.
        recipe_pika_kijai_windowed = {
            **recipe_a,
            "loras": [
                {
                    "file": "pika_h3_flex_fl2va_v1_avg_rank_56_bf16.safetensors",
                    "sha256": "cccc",
                    "strength": 1.0,
                    "applied": "branch:modules=no adaln,blocks=refiner,0-24,window=0-0.8",
                }
            ],
        }
        warns_pika_kijai = cv.check_vector_compatibility(
            meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_pika_kijai_windowed, allow_recipe_mismatch=False
        )
        if len(warns_pika_kijai) != 1 or "lora set" not in warns_pika_kijai[0].lower():
            errs.append(
                f"expected Kijai v1_avg_rank_56_bf16 with blocks=refiner,0-24 to pass as same-family variant, got {warns_pika_kijai}"
            )
        recipe_pika_no_refiner = {
            **recipe_a,
            "loras": [
                {
                    "file": "pika_h3_flex_fl2va_svd995_bf16.safetensors",
                    "sha256": "aaaa",
                    "strength": 1.0,
                    "applied": "branch:modules=all,blocks=0-24,window=0-1",
                }
            ],
        }
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_pika_no_refiner, allow_recipe_mismatch=False
            ),
            "recipe lora set",
            "compatibility refusal: Pika LoRA applied with blocks=0-24 (excluding token_refiner)",
        )
        recipe_pika_mlp_only = {
            **recipe_a,
            "loras": [
                {
                    "file": "pika_h3_flex_fl2va_svd995_bf16.safetensors",
                    "sha256": "aaaa",
                    "strength": 1.0,
                    "applied": "branch:modules=mlp only,blocks=all,window=0-1",
                }
            ],
        }
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_pika_mlp_only, allow_recipe_mismatch=False
            ),
            "recipe lora set",
            "compatibility refusal: Pika LoRA applied with modules=mlp only (partial token_refiner)",
        )

        # 4c. Same-family J-Lens synthesized vector (provenance.made_by == "j_lens.words") applied to a
        # different LoRA set emits an advisory warning (and never raises NameError on cur_family).
        meta_jlens = {
            **meta_out,
            "provenance": {
                **meta_out.get("provenance", {}),
                "made_by": "j_lens.words",
            },
        }
        warns_jlens_lora = cv.check_vector_compatibility(
            meta_jlens, loaded_family="fl2va", loaded_recipe=recipe_lora_b, allow_recipe_mismatch=False
        )
        if len(warns_jlens_lora) != 1 or "lora set" not in warns_jlens_lora[0].lower():
            errs.append(
                f"expected 1 advisory warning for same-family j_lens.words vector on different LoRA set, got {warns_jlens_lora}"
            )

        # 4d. Same-family fl2va vector applied to PrunaAI P-H3-Image (standalone p-h3-image_int8_convrot.safetensors
        # or unet_fl2va + p-h3-image_lora_bf16.safetensors) emits an advisory warning when allow_recipe_mismatch=False,
        # while a partial P-H3-Image LoRA that excludes token_refiner still refuses.
        recipe_ph3_standalone = {
            **recipe_a,
            "checkpoint": {"file": "p-h3-image_int8_convrot.safetensors", "sha256": "dddd"},
            "loras": [],
        }
        warns_ph3_ckpt = cv.check_vector_compatibility(
            meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_ph3_standalone, allow_recipe_mismatch=False
        )
        if not warns_ph3_ckpt or not any("recipe checkpoint" in w.lower() for w in warns_ph3_ckpt):
            errs.append(
                f"expected advisory warning when applying fl2va vector to p-h3-image_int8_convrot.safetensors, got {warns_ph3_ckpt}"
            )
        recipe_ph3_lora = {
            **recipe_a,
            "loras": [
                {
                    "file": "p-h3-image_lora_bf16.safetensors",
                    "sha256": "eeee",
                    "strength": 1.0,
                    "applied": "branch:modules=all,blocks=all,window=0-1",
                }
            ],
        }
        warns_ph3_lora = cv.check_vector_compatibility(
            meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_ph3_lora, allow_recipe_mismatch=False
        )
        if len(warns_ph3_lora) != 1 or "lora set" not in warns_ph3_lora[0].lower():
            errs.append(
                f"expected 1 advisory warning when applying fl2va vector to p-h3-image_lora_bf16.safetensors, got {warns_ph3_lora}"
            )
        recipe_ph3_lora_no_refiner = {
            **recipe_a,
            "loras": [
                {
                    "file": "p-h3-image_lora_bf16.safetensors",
                    "sha256": "eeee",
                    "strength": 1.0,
                    "applied": "branch:modules=all,blocks=0-24,window=0-1",
                }
            ],
        }
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_ph3_lora_no_refiner, allow_recipe_mismatch=False
            ),
            "recipe lora set",
            "compatibility refusal: p-h3-image_lora_bf16 applied with blocks=0-24 (excluding token_refiner)",
        )

        # 4e. Same-family video recipe compatibility: base (loras=[]) <-> Pika H3 Flex (fl2va & ref2va)
        # <-> ref2va Guide Video Edit (pika_ref2va + minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors),
        # and same-family checkpoint quantization variants (pruned_int8_convrot vs bf16), while
        # non-unit LoRA strength (0.5) still refuses when allow_recipe_mismatch=False.
        warns_pika_on_base = cv.check_vector_compatibility(
            meta_pika_svd, loaded_family="fl2va", loaded_recipe=recipe_a, allow_recipe_mismatch=False
        )
        if len(warns_pika_on_base) != 1 or "lora set" not in warns_pika_on_base[0].lower():
            errs.append(
                f"expected 1 advisory warning when applying Pika fl2va vector to base fl2va model (loras=[]), got {warns_pika_on_base}"
            )
        warns_base_on_pika = cv.check_vector_compatibility(
            meta_out, loaded_family="fl2va", loaded_recipe=recipe_pika_comfy, allow_recipe_mismatch=False
        )
        if len(warns_base_on_pika) != 1 or "lora set" not in warns_base_on_pika[0].lower():
            errs.append(
                f"expected 1 advisory warning when applying base fl2va vector (loras=[]) to Pika fl2va model, got {warns_base_on_pika}"
            )
        meta_ref2va_pika = {
            **meta_out,
            "family": "ref2va",
            "recipe": {
                **recipe_a,
                "checkpoint": {"file": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "sha256": "r111"},
                "loras": [
                    {
                        "file": "pika_h3_flex_ref2va_svd995_bf16.safetensors",
                        "sha256": "r222",
                        "strength": 1.0,
                        "applied": "branch",
                    }
                ],
            },
        }
        recipe_ref2va_guide_edit = {
            **recipe_a,
            "checkpoint": {"file": "minimax_h3_ref2va_bf16.safetensors", "sha256": "r333"},
            "loras": [
                {
                    "file": "pika_h3_flex_ref2va_svd995_bf16.safetensors",
                    "sha256": "r222",
                    "strength": 1.0,
                    "applied": "branch:modules=all,blocks=all,window=0-1",
                },
                {
                    "file": "minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors",
                    "sha256": "r444",
                    "strength": 1.0,
                    "applied": "branch:modules=all,blocks=all,window=0-1",
                },
            ],
        }
        warns_ref2va_guide = cv.check_vector_compatibility(
            meta_ref2va_pika, loaded_family="ref2va", loaded_recipe=recipe_ref2va_guide_edit, allow_recipe_mismatch=False
        )
        if len(warns_ref2va_guide) != 2 or not any("checkpoint" in w.lower() for w in warns_ref2va_guide) or not any("lora set" in w.lower() for w in warns_ref2va_guide):
            errs.append(
                f"expected 2 advisory warnings (checkpoint + LoRA set) for ref2va Pika vector on ref2va Guide Video Edit model, got {warns_ref2va_guide}"
            )
        recipe_pika_half_strength = {
            **recipe_a,
            "loras": [
                {
                    "file": "pika_h3_flex_fl2va_svd995_bf16.safetensors",
                    "sha256": "aaaa",
                    "strength": 0.5,
                    "applied": "branch",
                }
            ],
        }
        warns_half_strength = cv.check_vector_compatibility(
            meta_out, loaded_family="fl2va", loaded_recipe=recipe_pika_half_strength, allow_recipe_mismatch=False
        )
        if len(warns_half_strength) != 1 or "lora set" not in warns_half_strength[0].lower():
            errs.append(
                f"expected 1 advisory warning for non-unit strength (0.5) Pika LoRA vs base model, got {warns_half_strength}"
            )

        # 5. Prefix mismatch -> reported in warnings, never refused
        warns_prefix = cv.check_vector_compatibility(
            meta_out,
            loaded_family="fl2va",
            loaded_recipe=recipe_a,
            loaded_prefix={"references_sha256": "9999"},
        )
        if len(warns_prefix) != 1 or "measured under prefix" not in warns_prefix[0].lower():
            errs.append(f"expected 1 prefix warning, got {warns_prefix}")

        # 6. Unknown format refusal, non-Mapping loaded_recipe refusal, and loaded_recipe=None refusal
        bad_meta = {**meta_out, "format": "h3-control-vector/99"}
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(bad_meta, loaded_family="fl2va"),
            "expected 'h3-control-vector/1'",
            "compatibility refusal: unknown format",
        )
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(meta_out, loaded_family="fl2va", loaded_recipe=["bad"]),  # type: ignore[arg-type]
            "loaded_recipe must be a mapping",
            "compatibility refusal: non-Mapping loaded_recipe",
        )
        errs += _expect_refusal(
            lambda: cv.check_vector_compatibility(
                meta_out, loaded_family="fl2va", loaded_recipe=None, allow_recipe_mismatch=False
            ),
            "loaded_recipe is required",
            "compatibility refusal: loaded_recipe=None when allow_recipe_mismatch=False",
        )
    return errs


def check_apply_post_refiner_invariants() -> list[str]:
    errs: list[str] = []
    B, L, D = 1, 12, cv.POST_REFINER_DIM
    gen = torch.Generator().manual_seed(42)
    context = torch.randn((B, L, D), generator=gen, dtype=torch.bfloat16)
    tags = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    vec = torch.ones((D,), dtype=torch.float32) * 0.5

    # 1. Strength 0 reproduces unhooked context bitwise (even when context contains NaN).
    out_zero = cv.apply_post_refiner(context, tags, [(vec, 0.0)])
    if not torch.equal(out_zero, context):
        errs.append("apply_post_refiner at strength 0 did not reproduce context bitwise")
    ctx_nan = context.clone()
    ctx_nan[0, 5, 0] = float("nan")
    out_zero_nan = cv.apply_post_refiner(ctx_nan, tags, [(vec, 0.0)])
    if out_zero_nan.data_ptr() != ctx_nan.data_ptr():
        errs.append("apply_post_refiner at strength 0 did not return context unchanged when context had NaN")

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

    # 4. Refusals on wrong mask shape, non-finite vector, and non-finite steered context rows (while unsteered vision NaN is allowed).
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
    errs += _expect_refusal(
        lambda: cv.apply_post_refiner(ctx_nan, tags, [(vec, 1.0)]),
        "context contains non-finite values in steered rows",
        "apply_post_refiner refusal: NaN in steered text row at strength 1.0",
    )
    ctx_vis_nan = context.clone()
    ctx_vis_nan[0, 0, 0] = float("nan")
    out_vis_nan = cv.apply_post_refiner(ctx_vis_nan, tags, [(vec, 1.0)])
    if not torch.isnan(out_vis_nan[0, 0, 0]) or not torch.equal(out_vis_nan[:, 4:, :], context[:, 4:, :] + vec.to(dtype=context.dtype)):
        errs.append("apply_post_refiner failed when NaN was only in unsteered tag==0 vision row")

    # 5. mask_vision=False (MAINodes-style unmasked ablation) shifts vision rows too.
    out_unmasked = cv.apply_post_refiner(context, tags, [(vec, 1.0)], mask_vision=False)
    expected_all = context + vec.to(dtype=context.dtype)
    if not torch.equal(out_unmasked, expected_all):
        errs.append("apply_post_refiner(mask_vision=False) did not shift all rows")

    # 6. row_norm_census, channel_census, and meter_score invariants.
    rc = cv.row_norm_census(context[0], tags[0])
    if rc["row0_to_median_ratio"] <= 0 or rc["median_row_norm"] <= 0:
        errs.append(f"unexpected row_norm_census: {rc}")
    cc = cv.channel_census(vec)
    if abs(cc["top1_energy_share"] - (1.0 / D)) > 1e-5:
        errs.append(f"unexpected uniform channel_census top1_energy_share: {cc}")
    ms = cv.meter_score(context[0], vec, mask=(tags[0] == 1))
    req_ms_keys = {
        "projection_units",
        "z_score",
        "signed_projection",
        "cosine",
        "mean_cosine",
        "max_token_cosine",
        "p90_token_cosine",
        "p90_cosine",
        "per_token_cosines",
        "top_coupled_tokens",
    }
    if not req_ms_keys.issubset(ms.keys()) or len(ms["per_token_cosines"]) != L:
        errs.append(f"unexpected meter_score output or missing per-token keys: {ms.keys()}")
    errs += _expect_refusal(
        lambda: cv._apply_holds_to_shift64(
            vec.to(dtype=torch.float64),
            [(vec, 2.5)],
            device=vec.device,
        ),
        "hold alpha must be finite and in [0.0, 2.0]",
        "_apply_holds_to_shift64 refusal on hold_alpha > 2.0",
    )

    # 7. Apply-time modes: project_out and amplify exact algebra on float32 synthetic tensors.
    v_dir = torch.zeros(D, dtype=torch.float32)
    v_dir[0] = 3.0
    v_dir[1] = 4.0  # ||v_dir|| = 5.0, v_hat = (0.6, 0.8, 0, ...)
    v_hat = v_dir / 5.0
    ortho = torch.zeros(D, dtype=torch.float32)
    ortho[2] = 7.0
    ctx_f32 = torch.zeros((1, L, D), dtype=torch.float32)
    ctx_f32[0, :4, :] = 9.0  # vision rows
    ctx_f32[0, 4:, :] = 2.0 * v_dir + ortho  # projection along v_hat is 2.0 * 5.0 = 10.0

    # project_out at s=1.0 removes the component along v_dir completely and leaves ortho untouched
    out_proj1 = cv.apply_post_refiner(ctx_f32, tags, [(v_dir, 1.0, "project_out")])
    if not torch.equal(out_proj1[:, :4, :], ctx_f32[:, :4, :]):
        errs.append("apply_post_refiner(mode='project_out') mutated vision rows")
    residual_dot = float((out_proj1[0, 4:, :].to(torch.float64) @ v_hat.to(torch.float64)).abs().max().item())
    if residual_dot > 1e-6:
        errs.append(f"apply_post_refiner(mode='project_out', s=1.0) left component along v: max_dot={residual_dot}")
    if not torch.allclose(out_proj1[0, 4:, :], ortho.unsqueeze(0).expand(L - 4, D), atol=1e-6):
        errs.append("apply_post_refiner(mode='project_out', s=1.0) altered orthogonal component")

    # amplify at s=0.5 scales only the existing component along v_dir by (1 + 0.5) = 1.5
    out_amp = cv.apply_post_refiner(ctx_f32, tags, [(v_dir, 0.5, "amplify")])
    if not torch.equal(out_amp[:, :4, :], ctx_f32[:, :4, :]):
        errs.append("apply_post_refiner(mode='amplify') mutated vision rows")
    expected_amp = 3.0 * v_dir + ortho
    if not torch.allclose(out_amp[0, 4:, :], expected_amp.unsqueeze(0).expand(L - 4, D), atol=1e-6):
        errs.append("apply_post_refiner(mode='amplify', s=0.5) did not scale component along v by 1.5 while preserving ortho")

    # strength 0 is still bitwise identity for project_out and amplify
    for m_name in ("project_out", "amplify", "add_norm_matched"):
        out_m0 = cv.apply_post_refiner(ctx_f32, tags, [(v_dir, 0.0, m_name)])
        if out_m0.data_ptr() != ctx_f32.data_ptr():
            errs.append(f"apply_post_refiner(mode={m_name!r}, s=0.0) did not return context unchanged")

    # 8. Norm-matched mode ('add_norm_matched' or scale_mode='norm_matched'):
    # v_dir has norm 5.0; in norm_matched mode, s=1.0 adds (67.5 / 5.0) * v_dir = 13.5 * v_dir.
    ctx_zeros = torch.zeros((1, L, D), dtype=torch.float32)
    out_nm_mode = cv.apply_post_refiner(ctx_zeros, tags, [(v_dir, 1.0, "add_norm_matched")])
    out_nm_scale = cv.apply_post_refiner(ctx_zeros, tags, [(v_dir, 1.0, "add")], scale_mode="norm_matched")
    expected_nm_row = (67.5 / 5.0) * v_dir
    if not torch.allclose(out_nm_mode, out_nm_scale, atol=1e-6):
        errs.append("apply_post_refiner mode='add_norm_matched' and scale_mode='norm_matched' differed")
    if not torch.allclose(out_nm_mode[0, 4:, :], expected_nm_row.unsqueeze(0).expand(L - 4, D), atol=1e-5):
        errs.append("apply_post_refiner norm_matched did not scale vector to NORM_MATCHED_TARGET (67.5)")

    # 9. Soft ridge hold projection project_out(..., alpha) and apply_post_refiner(mode='hold') + vector_interactions:
    v_steer = 2.0 * v_dir + ortho  # component along v_hat is 10.0, orthogonal component is ortho
    v_proj_half = cv.project_out(v_steer, v_dir.unsqueeze(0), alpha=0.5)
    if not torch.allclose(v_proj_half, 1.0 * v_dir + ortho, atol=1e-6):
        errs.append(f"project_out(alpha=0.5) did not attenuate component along basis by 50%: {v_proj_half[:4]}")
    v_proj_over = cv.project_out(v_steer, v_dir.unsqueeze(0), alpha=1.5)
    if not torch.allclose(v_proj_over, -1.0 * v_dir + ortho, atol=1e-6):
        errs.append(f"project_out(alpha=1.5) did not over-project component along basis: {v_proj_over[:4]}")
    errs += _expect_refusal(
        lambda: cv.project_out(v_steer, v_dir.unsqueeze(0), alpha=2.5),
        "alpha must be finite and in [0.0, 2.0]",
        "project_out refusal: alpha > 2.0",
    )

    # Standalone mode='hold' with no additive vector must leave context bitwise untouched (unlike 'project_out')
    out_hold_only = cv.apply_post_refiner(ctx_f32, tags, [(v_dir, 1.0, "hold")])
    if not torch.equal(out_hold_only, ctx_f32):
        errs.append("apply_post_refiner(mode='hold') with no 'add' vector modified base context")

    # Stacked ('add', v_steer, 1.0) + ('hold', v_dir, 1.0) adds only ortho to context, leaving ctx_f32's own v_dir component intact!
    out_add_plus_hold = cv.apply_post_refiner(
        ctx_f32,
        tags,
        [(v_steer, 1.0, "add"), (v_dir, 1.0, "hold")],
    )
    expected_add_hold = ctx_f32.clone()
    expected_add_hold[0, 4:, :] = ctx_f32[0, 4:, :] + ortho
    if not torch.allclose(out_add_plus_hold, expected_add_hold, atol=1e-6):
        errs.append("apply_post_refiner([('add', v_steer, 1.0), ('hold', v_dir, 1.0)]) did not project v_dir out of shift only")

    # Soft hold ('hold', v_dir, 0.5) removes 50% of v_dir from v_steer shift
    out_add_soft_hold = cv.apply_post_refiner(
        ctx_zeros,
        tags,
        [(v_steer, 1.0, "add"), (v_dir, 0.5, "hold")],
    )
    if not torch.allclose(
        out_add_soft_hold[0, 4:, :],
        (1.0 * v_dir + ortho).unsqueeze(0).expand(L - 4, D),
        atol=1e-6,
    ):
        errs.append("apply_post_refiner soft hold (s=0.5) did not remove 50% of hold component from additive shift")

    inter_stats = cv.vector_interactions([("v_steer", v_steer, 1.0), ("v_dir", v_dir, 1.0)])
    expected_inter_sq = 2.0 * float(torch.dot(v_steer, v_dir).item())
    if abs(inter_stats["total_interaction_sq"] - expected_inter_sq) > 1e-4 or len(inter_stats["pairs"]) != 1:
        errs.append(f"vector_interactions mismatch: {inter_stats} vs expected {expected_inter_sq}")

    errs += _expect_refusal(
        lambda: cv.apply_post_refiner(ctx_f32, tags, [(v_dir, 1.0, "bogus_mode")]),
        "mode must be one of",
        "apply_post_refiner refusal: unknown mode",
    )

    return errs


def check_runtime_nodes_arm_and_cli() -> list[str]:
    errs: list[str] = []
    import importlib.util
    import comfy.ldm.minimax.model as mm_mod
    from comfy.patcher_extension import WrappersMP
    import json
    import control_vector_nodes as cv_nodes
    if str(REPO / "workflows") not in sys.path:
        sys.path.insert(0, str(REPO / "workflows"))
    from workflows import build_workflows as bw

    spec_bcv = importlib.util.spec_from_file_location(
        "bench_control_vectors", REPO / "bench" / "control_vectors.py"
    )
    assert spec_bcv is not None and spec_bcv.loader is not None
    bench_cv = importlib.util.module_from_spec(spec_bcv)
    spec_bcv.loader.exec_module(bench_cv)

    # 1. Node schemas and registration in nodes.py + bench/node_id_manifest.json
    nodes_src = (REPO / "nodes.py").read_text(encoding="utf-8")
    manifest_data = json.loads((REPO / "bench" / "node_id_manifest.json").read_text(encoding="utf-8"))
    for cls, expected_id in (
        (cv_nodes.MiniMaxH3MakeControlVector, "MiniMaxH3MakeControlVector"),
        (cv_nodes.MiniMaxH3MakeMediaControlVector, "MiniMaxH3MakeMediaControlVector"),
        (cv_nodes.MiniMaxH3ApplyControlVectors, "MiniMaxH3ApplyControlVectors"),
        (cv_nodes.MiniMaxH3LayerNormMatch, "MiniMaxH3LayerNormMatch"),
    ):
        schema = cls.define_schema()
        if schema.node_id != expected_id:
            errs.append(f"schema.node_id={schema.node_id!r}, expected {expected_id!r}")
        if expected_id not in nodes_src or expected_id not in manifest_data:
            errs.append(f"{expected_id} missing from nodes.py or bench/node_id_manifest.json")

    # 2. Shipped prompt_bank/control_vectors/*.toml validation (when present)
    cv_bank_dir = REPO / "prompt_bank" / "control_vectors"
    if cv_bank_dir.is_dir():
        for toml_path in sorted(cv_bank_dir.glob("*.toml")):
            try:
                pf = cv.load_pairs(toml_path, bank_dir=REPO / "prompt_bank")
                if len(pf.pairs) < 4:
                    errs.append(f"{toml_path.name}: expected >= 4 pairs, got {len(pf.pairs)}")
            except Exception as exc:
                errs.append(f"{toml_path.name} failed load_pairs: {exc}")

    # 3. Mock ModelPatcher & DIFFUSION_MODEL runtime wrapper + recipe extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt_path = tmp / "mock_fl2va.safetensors"
        gen = torch.Generator().manual_seed(7)
        sd_tensors = {
            "condition_proj.weight": torch.randn(
                (cv.POST_REFINER_DIM, cv.ENCODER_DIM), generator=gen, dtype=torch.bfloat16
            ),
            "condition_proj.bias": torch.randn(
                (cv.POST_REFINER_DIM,), generator=gen, dtype=torch.bfloat16
            ),
            "token_refiner.final_norm.weight": torch.randn(
                (cv.POST_REFINER_DIM,), generator=gen, dtype=torch.bfloat16
            ),
        }
        save_file(sd_tensors, str(ckpt_path))
        fam = cv.detect_family(ckpt_path)

        class _MockDiffusionModel(mm_mod.MiniMaxH3Model):
            def __init__(self, with_tensors: bool = True):
                torch.nn.Module.__init__(self)
                if with_tensors:
                    self.condition_proj = SimpleNamespace(
                        weight=sd_tensors["condition_proj.weight"],
                        bias=sd_tensors["condition_proj.bias"],
                    )
                    self.token_refiner = SimpleNamespace(
                        final_norm=SimpleNamespace(
                            weight=sd_tensors["token_refiner.final_norm.weight"]
                        )
                    )
                self.last_context: torch.Tensor | None = None

            def run_step(self, x, timestep, context, transformer_options={}, **kwargs):
                self.last_context = context.clone()
                return x + context.mean()

        class _MockPatcher:
            def __init__(self, dm: _MockDiffusionModel, ckpt: Path | None):
                self.model = SimpleNamespace(diffusion_model=dm)
                self.cached_patcher_init = (
                    (None, (str(ckpt),), {}) if ckpt is not None else None
                )
                self.backup: dict[str, object] = {}
                self.model_options: dict[str, Any] = {"transformer_options": {}}
                self.wrappers: dict[Any, dict[str, list[Any]]] = {}

            def clone(self) -> "_MockPatcher":
                c = _MockPatcher(self.model.diffusion_model, None)
                c.cached_patcher_init = self.cached_patcher_init
                c.backup = dict(self.backup)
                c.model_options = {
                    "transformer_options": dict(
                        self.model_options.get("transformer_options", {})
                    )
                }
                c.wrappers = {
                    k: {wk: list(wv) for wk, wv in v.items()}
                    for k, v in self.wrappers.items()
                }
                return c

            def get_model_object(self, name: str) -> object:
                if name == "diffusion_model":
                    return self.model.diffusion_model
                raise KeyError(name)

            def invoke_diffusion_model(
                self, x, timestep, context, transformer_options={}, **kwargs
            ):
                dm_obj = self.model.diffusion_model
                w_list = [
                    fn
                    for fns in self.wrappers.get(WrappersMP.DIFFUSION_MODEL, {}).values()
                    for fn in fns
                ]
                if not w_list:
                    return dm_obj.run_step(x, timestep, context, transformer_options, **kwargs)

                def _call(idx: int, cx, ct, cctx, ctopts, **ckw):
                    if idx >= len(w_list):
                        return dm_obj.run_step(cx, ct, cctx, ctopts, **ckw)
                    return w_list[idx](
                        lambda nx, nt, nctx, ntopts, **nkw: _call(
                            idx + 1, nx, nt, nctx, ntopts, **nkw
                        ),
                        cx,
                        ct,
                        cctx,
                        ctopts,
                        **ckw,
                    )

                return _call(0, x, timestep, context, transformer_options, **kwargs)

        dm = _MockDiffusionModel(with_tensors=True)
        patcher_with_init = _MockPatcher(dm, ckpt_path)
        rec1 = cv_nodes.extract_model_recipe(patcher_with_init)
        if rec1["checkpoint"]["file"] != "mock_fl2va.safetensors":
            errs.append(f"unexpected extract_model_recipe with cached_patcher_init: {rec1}")

        # Fallback when cached_patcher_init is None (in-memory tensor hash fallback)
        patcher_no_init = _MockPatcher(dm, None)
        rec2 = cv_nodes.extract_model_recipe(patcher_no_init)
        if rec2["checkpoint"]["file"] != f"in_memory:{fam}":
            errs.append(f"unexpected extract_model_recipe fallback without cached_patcher_init: {rec2}")
        dm_alt = _MockDiffusionModel(with_tensors=True)
        dm_alt.condition_proj = SimpleNamespace(
            weight=dm.condition_proj.weight + 1.0,
            bias=dm.condition_proj.bias,
        )
        patcher_no_init_alt = _MockPatcher(dm_alt, None)
        rec2_alt = cv_nodes.extract_model_recipe(patcher_no_init_alt)
        if rec2_alt["checkpoint"]["sha256"] == rec2["checkpoint"]["sha256"]:
            errs.append("extract_model_recipe in-memory hash ignored tensor bytes")

        # Refusal when cached_patcher_init is None AND model lacks the 3 defining tensors
        empty_dm = _MockDiffusionModel(with_tensors=False)
        patcher_empty = _MockPatcher(empty_dm, None)
        errs += _expect_refusal(
            lambda: cv_nodes.extract_model_recipe(patcher_empty),
            "has no checkpoint path and is missing H3 text-path tensor",
            "extract_model_recipe refusal when both cached_patcher_init and tensors are absent",
        )

        # Save two valid post-refiner control vectors matching rec1 (0.75 + 0.75 = 1.5 tests bf16 rounding at 256.0)
        v_dir1 = torch.ones((cv.POST_REFINER_DIM,), dtype=torch.float32) * 0.75
        v_dir2 = torch.ones((cv.POST_REFINER_DIM,), dtype=torch.float32) * 0.75
        meta1: dict[str, Any] = {
            "format": cv.FORMAT_VERSION,
            "name": "test_v1",
            "quality": "test vector 1",
            "family": fam,
            "pairs": {
                "path": "prompt_bank/control_vectors/film_16mm_1974.toml",
                "sha256": "a" * 64,
                "n_pairs": 4,
                "n_holds": 0,
            },
            "recipe": rec1,
            "measured_under": {
                "mode": "t2va",
                "frame": "scene",
                "prefix": "none",
                "pooling": "scene_text",
            },
            "points": {
                "post_refiner": {
                    "unit": "mean_pair_difference",
                    "raw_norm": float(torch.linalg.vector_norm(v_dir1).item()),
                    "median_row_norm": 1.0,
                }
            },
            "quality_report": {
                "mean_pairwise_cosine": 0.9,
                "split_half_cosine": 0.92,
                "per_pair_cosine": [0.9, 0.9, 0.9, 0.9],
            },
            "provenance": {
                "core_revision": "test",
                "pack_commit": "test",
                "created_utc": "2026-11-04T00:00:00Z",
                "made_by": "check_control_vectors.py",
            },
        }
        v1_path = cv.save_vector(
            tmp / "test_v1.safetensors", {"vector.post_refiner": v_dir1}, meta1
        )
        meta2 = {**meta1, "name": "test_v2"}
        v2_path = cv.save_vector(
            tmp / "test_v2.safetensors", {"vector.post_refiner": v_dir2}, meta2
        )

        # Attach with strength 0 -> exact reproduction
        p_zero = cv_nodes.attach_control_vector(patcher_with_init, v1_path, strength=0.0)
        ctx_in = torch.randn((1, 6, cv.POST_REFINER_DIM), generator=gen, dtype=torch.bfloat16)
        ctx_in[0, 2, 0] = torch.tensor(256.0, dtype=torch.bfloat16)
        tags = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
        x_in = torch.zeros((1, 4), dtype=torch.bfloat16)
        t_in = torch.ones((1,), dtype=torch.bfloat16)
        import logging as _logging

        orig_propagate = cv_nodes.logger.propagate
        orig_level = cv_nodes.logger.level
        cv_nodes.logger.propagate = False
        cv_nodes.logger.setLevel(_logging.ERROR)
        p_zero.invoke_diffusion_model(
            x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags}
        )
        if not torch.equal(dm.last_context, ctx_in):
            errs.append("attach_control_vector at strength 0 altered refined context")

        # Stacking two attach_control_vector calls accumulates in float64 and preserves tag==0 vision rows
        p_stack1 = cv_nodes.attach_control_vector(
            patcher_with_init, v1_path, strength=1.0, mask="non_vision"
        )
        p_stack2 = cv_nodes.attach_control_vector(
            p_stack1, v2_path, strength=1.0, mask="non_vision"
        )
        p_stack2.invoke_diffusion_model(
            x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags}
        )
        got_ctx = dm.last_context
        if got_ctx is None or not torch.equal(got_ctx[:, :2, :], ctx_in[:, :2, :]):
            errs.append("stacked attach_control_vector mutated tag==0 vision rows")
        expected_shifted = cv.apply_post_refiner(
            ctx_in, tags, [(v_dir1, 1.0), (v_dir2, 1.0)], mask_vision=True
        )
        if got_ctx is None or not torch.equal(got_ctx, expected_shifted):
            errs.append("stacked attach_control_vector did not match single float64 accumulation")

        # Unmasked mode ('all_rows') shifts tag==0 vision rows too (E2.3 / MAINodes ablation),
        # and when minimax_payload has no text_token_tags, fallback tags are 1s (no false prefix warning).
        p_all = cv_nodes.attach_control_vector(
            patcher_with_init, v1_path, strength=1.0, mask="all_rows"
        )
        p_all.invoke_diffusion_model(
            x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags}
        )
        expected_all_rows = cv.apply_post_refiner(
            ctx_in, tags, [(v_dir1, 1.0)], mask_vision=False
        )
        if dm.last_context is None or not torch.equal(dm.last_context, expected_all_rows):
            errs.append("attach_control_vector(mask='all_rows') did not shift all rows")

        # Dialogue-aware inference masks ('scene_only' and 'after_dialogue') via _h3_span_masks on text_token_tags
        # rows 0..1 = vision, 2..3 = scene, 4..5 = dialogue (<d>...</d>) -> scene_only shifts only rows 2..3;
        # if row 5 is scene after dialogue (rows 0..1 vision, 2 scene, 3..4 dialogue, 5 scene), after_dialogue shifts only row 5.
        tags_dlg = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
        tags_dlg._h3_span_masks = {
            "vision": torch.tensor([True, True, False, False, False, False]),
            "label": torch.zeros(6, dtype=torch.bool),
            "timestamp": torch.zeros(6, dtype=torch.bool),
            "dialogue": torch.tensor([False, False, False, True, True, False]),
            "scene": torch.tensor([False, False, True, False, False, True]),
        }
        p_scene_only = cv_nodes.attach_control_vector(
            patcher_with_init, v1_path, strength=1.0, mask="scene_only"
        )
        p_scene_only.invoke_diffusion_model(
            x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags_dlg}
        )
        if (
            dm.last_context is None
            or not torch.equal(dm.last_context[:, [0, 1, 3, 4], :], ctx_in[:, [0, 1, 3, 4], :])
            or torch.equal(dm.last_context[:, [2, 5], :], ctx_in[:, [2, 5], :])
        ):
            errs.append("attach_control_vector(mask='scene_only') did not protect <d>...</d> dialogue rows 3..4 while steering scene rows 2, 5")

        # Verify mask='scene_only' works via explicit minimax_payload['h3_span_masks'] or model_conds kwarg h3_span_masks (even without tags._h3_span_masks)
        tags_bare = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
        p_scene_only.invoke_diffusion_model(
            x_in,
            t_in,
            ctx_in,
            minimax_payload={"text_token_tags": tags_bare},
            h3_span_masks=tags_dlg._h3_span_masks,
        )
        if (
            dm.last_context is None
            or not torch.equal(dm.last_context[:, [0, 1, 3, 4], :], ctx_in[:, [0, 1, 3, 4], :])
            or torch.equal(dm.last_context[:, [2, 5], :], ctx_in[:, [2, 5], :])
        ):
            errs.append("attach_control_vector(mask='scene_only') did not read model_conds kwarg h3_span_masks")

        # Verify mask='scene_only' refuses when neither h3_span_masks/_h3_span_masks nor prompt is present
        errs += _expect_refusal(
            lambda: p_scene_only.invoke_diffusion_model(
                x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags_bare}
            ),
            "uses mask='scene_only', which requires conditioning",
            "attach_control_vector(mask='scene_only') refusal when span masks and prompt are absent",
        )

        p_after_dlg = cv_nodes.attach_control_vector(
            patcher_with_init, v1_path, strength=1.0, mask="after_dialogue"
        )
        p_after_dlg.invoke_diffusion_model(
            x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags_dlg}
        )
        if (
            dm.last_context is None
            or not torch.equal(dm.last_context[:, :5, :], ctx_in[:, :5, :])
            or torch.equal(dm.last_context[:, 5:, :], ctx_in[:, 5:, :])
        ):
            errs.append("attach_control_vector(mask='after_dialogue') did not protect rows 0..4 up through final </d> while steering row 5")

        warn_msgs: list[str] = []

        class _WarnCapture(_logging.Handler):
            def emit(self, record):
                if record.levelno >= _logging.WARNING:
                    warn_msgs.append(record.getMessage())

        p_all_notag = cv_nodes.attach_control_vector(
            patcher_with_init, v1_path, strength=1.0, mask="all_rows"
        )
        wh = _WarnCapture()
        cv_nodes.logger.setLevel(_logging.WARNING)
        cv_nodes.logger.addHandler(wh)
        try:
            p_all_notag.invoke_diffusion_model(x_in, t_in, ctx_in, minimax_payload={})
        finally:
            cv_nodes.logger.removeHandler(wh)
            cv_nodes.logger.setLevel(_logging.ERROR)
        if warn_msgs:
            errs.append(f"mask='all_rows' no-tag fallback logged unexpected warning(s): {warn_msgs}")

        # Attach refusals: non-H3 model, bad mask, non-finite strength, missing post_refiner tensor,
        # 'none' returns a clone, validate_inputs rejects missing file, fingerprint_inputs & is_output_node
        non_h3_patcher = _MockPatcher(SimpleNamespace(), ckpt_path)
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(non_h3_patcher, "none"),
            "MiniMaxH3ApplyControlVectors requires a MiniMax H3 diffusion model",
            "attach_control_vector refusal on non-H3 diffusion model",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(patcher_with_init, v1_path, mask="bogus"),
            "mask must be one of",
            "attach_control_vector refusal on invalid mask",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(patcher_with_init, v1_path, strength=float("nan")),
            "strength must be finite",
            "attach_control_vector refusal on non-finite strength",
        )
        enc_only_dir = torch.zeros((cv.ENCODER_DIM,), dtype=torch.float32)
        enc_only_dir[0] = 1.0
        enc_only_meta = {
            **meta1,
            "name": "enc_only",
            "points": {
                "encoder": {
                    "unit": "median_row_norm",
                    "raw_norm": 1.0,
                    "median_row_norm": 1.0,
                    "row0_excluded": True,
                }
            },
        }
        enc_only_path = cv.save_vector(
            tmp / "enc_only.safetensors", {"vector.encoder": enc_only_dir}, enc_only_meta
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(patcher_with_init, enc_only_path),
            "has no 'vector.post_refiner' tensor",
            "attach_control_vector refusal when vector.post_refiner is absent",
        )
        p_none = cv_nodes.attach_control_vector(patcher_with_init, "none")
        if p_none is patcher_with_init:
            errs.append("attach_control_vector('none') returned input model instead of clone")
        val_missing = cv_nodes.MiniMaxH3ApplyControlVectors.validate_inputs(
            vector_name="missing_does_not_exist.safetensors"
        )
        if val_missing is True or "not found" not in str(val_missing):
            errs.append(f"MiniMaxH3ApplyControlVectors.validate_inputs accepted missing file: {val_missing!r}")
        if not cv_nodes.MiniMaxH3MakeControlVector.define_schema().is_output_node:
            errs.append("MiniMaxH3MakeControlVector schema must set is_output_node=True")
        fp_apply = cv_nodes.MiniMaxH3ApplyControlVectors.fingerprint_inputs(vector_name=str(v1_path))
        if not isinstance(fp_apply, tuple) or len(fp_apply) != 3:
            errs.append(f"MiniMaxH3ApplyControlVectors.fingerprint_inputs unexpected return: {fp_apply!r}")

        # Refusal on family mismatch in attach_control_vector
        bad_meta = {**meta1, "family": "ref2va"}
        bad_v_path = cv.save_vector(
            tmp / "bad_fam.safetensors", {"vector.post_refiner": v_dir1}, bad_meta
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(patcher_with_init, bad_v_path, strength=1.0),
            "cannot apply to loaded model of family",
            "attach_control_vector refusal on family mismatch",
        )

        # MiniMaxH3ApplyControlVectors.execute chained (explicit mode='add' matches expected_shifted)
        (p_node_1,) = cv_nodes.MiniMaxH3ApplyControlVectors.execute(
            model=patcher_with_init,
            vector_name=str(v1_path),
            strength=1.0,
            mode="add",
            mask="non_vision",
            allow_recipe_mismatch=False,
        )
        (p_node_2,) = cv_nodes.MiniMaxH3ApplyControlVectors.execute(
            model=p_node_1,
            vector_name=str(v2_path),
            strength=1.0,
            mode="add",
            mask="non_vision",
            allow_recipe_mismatch=False,
        )
        p_node_2.invoke_diffusion_model(
            x_in, t_in, ctx_in, minimax_payload={"text_token_tags": tags}
        )
        if dm.last_context is None or not torch.equal(dm.last_context, expected_shifted):
            errs.append("chained MiniMaxH3ApplyControlVectors.execute did not produce expected shifted context")

        # Refusal when minimax_payload lacks text_token_tags and mask == 'non_vision'
        errs += _expect_refusal(
            lambda: p_node_1.invoke_diffusion_model(
                x_in, t_in, ctx_in, minimax_payload={}
            ),
            "missing 'text_token_tags'",
            "wrapper refusal when minimax_payload is missing text_token_tags under mask='non_vision'",
        )
        # Refusal when hold_words is passed with vector_name='none' and empty words:
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_init, "none", hold_words="warm, golden"
            ),
            "hold_words requires",
            "attach_control_vector refusal when hold_words is set with no vector_name or words",
        )
        val_hold_only = cv_nodes.MiniMaxH3ApplyControlVectors.validate_inputs(
            vector_name="none", words="", hold_words="warm, golden"
        )
        if not isinstance(val_hold_only, str) or "hold_words requires" not in val_hold_only:
            errs.append(f"validate_inputs did not refuse hold_words alone: {val_hold_only!r}")
        # Refusal when a mode='hold' node has a different mask than the preceding additive node:
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                p_node_1, str(v2_path), strength=1.0, mode="hold", mask="all_rows"
            ),
            "must share the same mask",
            "attach_control_vector refusal when mode='hold' has a different mask from preceding add node",
        )

        # In-memory model.patches (LoraLoaderModelOnly) and object_patches (MiniMaxH3LoRABranch) recipe detection
        patcher_with_lora = patcher_with_init.clone()
        patcher_with_lora.patches = {
            "diffusion_model.blocks.0.attn.weight": [(0.75, torch.ones((2, 2)))],
            "diffusion_model.blocks.1.attn.weight": [(0.50, torch.ones((2, 2)))],
        }
        rec_lora = cv_nodes.extract_model_recipe(patcher_with_lora)
        if not rec_lora["loras"] or rec_lora["loras"][0]["applied"] != "merged":
            errs.append(f"extract_model_recipe did not record model.patches LoRAs as 'merged': {rec_lora}")
        patcher_with_lora_s2 = patcher_with_init.clone()
        patcher_with_lora_s2.patches = {
            "diffusion_model.blocks.0.attn.weight": [(0.75, torch.ones((2, 2)))],
            "diffusion_model.blocks.1.attn.weight": [(0.25, torch.ones((2, 2)))],
        }
        rec_lora_s2 = cv_nodes.extract_model_recipe(patcher_with_lora_s2)
        if rec_lora_s2["loras"][0]["sha256"] == rec_lora["loras"][0]["sha256"]:
            errs.append("extract_model_recipe LoRA hash ignored patch entry strength")
        patcher_with_branch = patcher_with_init.clone()
        patcher_with_branch.object_patches = {
            "diffusion_model.blocks.0.attn1.to_q": SimpleNamespace(
                a=torch.ones((4, 8), dtype=torch.float32),
                b=torch.ones((8, 4), dtype=torch.float32),
                scale=1.0,
            )
        }
        rec_branch = cv_nodes.extract_model_recipe(patcher_with_branch)
        if not rec_branch["loras"] or rec_branch["loras"][0]["applied"] != "branch":
            errs.append(f"extract_model_recipe did not record object_patches LoRABranch as 'branch': {rec_branch}")
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_lora, v1_path, strength=1.0, allow_recipe_mismatch=False
            ),
            "recipe lora set",
            "attach_control_vector refusal when model.patches has an unrecorded LoRA",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_branch, v1_path, strength=1.0, allow_recipe_mismatch=False
            ),
            "recipe lora set",
            "attach_control_vector refusal when model.object_patches has an unrecorded LoRABranch",
        )

        # End-to-end encode_prompt_for_vector (t2va + ref2va), make_control_vector_from_pairs,
        # MiniMaxH3MakeControlVector.execute, and bench/control_vectors.py (make, inspect --meter, grade)
        import comfy.text_encoders.minimax as mm_te

        real_mm_tok = mm_te.MiniMaxH3Tokenizer()

        class _MockClip:
            def __init__(self, ckpt: Path):
                self.tokenizer = SimpleNamespace(minimax=real_mm_tok)
                self.cond_stage_model = SimpleNamespace(clip_name="qwen3vl_32b")
                self.patcher = SimpleNamespace(
                    cached_patcher_init=(None, (str(ckpt),), {})
                )

            def tokenize(self, text: str, **kwargs):
                return real_mm_tok.tokenize_with_weights(text, **kwargs)

            def encode_from_tokens_scheduled(self, tokens):
                entries = tokens["qwen3vl_32b"][0]
                row_list: list[torch.Tensor] = []
                tag_list: list[int] = []
                vis_shift = 0.0
                i_ent = 0
                while i_ent < len(entries):
                    item0 = entries[i_ent][0]
                    if (
                        item0 == h3_spans.VISION_START
                        and i_ent + 2 < len(entries)
                        and isinstance(entries[i_ent + 1][0], dict)
                        and entries[i_ent + 2][0] == h3_spans.VISION_END
                    ):
                        img_t = entries[i_ent + 1][0]["data"]
                        n_vis = max(4, (int(img_t.shape[1]) // 32) + (int(img_t.shape[2]) // 32))
                        vis_shift = float(img_t.mean().item())
                        for v_i in range(n_vis):
                            g = torch.Generator().manual_seed(2000 + v_i)
                            row_list.append(torch.randn((cv.ENCODER_DIM,), generator=g) + vis_shift)
                            tag_list.append(0)
                        i_ent += 3
                    else:
                        tid = int(item0) if not isinstance(item0, dict) else 0
                        g = torch.Generator().manual_seed(1000 + tid + i_ent)
                        row_list.append(torch.randn((cv.ENCODER_DIM,), generator=g) + vis_shift * 0.5)
                        tag_list.append(1)
                        i_ent += 1
                rows = torch.stack(row_list, dim=0).unsqueeze(0)
                tags_1d = torch.tensor(tag_list, dtype=torch.long)
                return [[rows, {"pooled_output": None, "minimax_token_tags": tags_1d}]]

        mock_clip = _MockClip(ckpt_path)
        rec_with_clip = cv_nodes.extract_model_recipe(patcher_with_init, clip=mock_clip)
        if rec_with_clip["encoder"]["file"] != ckpt_path.name or len(rec_with_clip["encoder"]["sha256"]) != 64:
            errs.append(f"extract_model_recipe ignored encoder path from clip.patcher: {rec_with_clip['encoder']}")

        # Modify dm so build_refiner_runner(patcher_with_init) checks bf16 cast and float64 pooling precision
        proj_w = sd_tensors["condition_proj.weight"].to(dtype=torch.float32)
        proj_b = sd_tensors["condition_proj.bias"].to(dtype=torch.float32)
        dm.condition_proj = torch.nn.Linear(cv.ENCODER_DIM, cv.POST_REFINER_DIM, dtype=torch.bfloat16)
        with torch.no_grad():
            dm.condition_proj.weight.copy_(sd_tensors["condition_proj.weight"])
            dm.condition_proj.bias.copy_(sd_tensors["condition_proj.bias"])

        def _mock_preprocess(x_3d):
            if x_3d.dtype != dm.condition_proj.weight.dtype:
                raise TypeError(
                    f"preprocess_text_embeds expected {dm.condition_proj.weight.dtype}, got {x_3d.dtype}"
                )
            out = torch.nn.functional.linear(x_3d.to(dtype=torch.float32), proj_w, proj_b)
            if out.shape[1] >= 4:
                out = out.clone()
                # Cancelling +/- 2^24 at the first and last row swallows intermediate rows in float32 .mean(dim=0)
                # while float64 .mean(dim=0) preserves them.
                out[:, 0, 0] = 16777216.0
                out[:, -1, 0] = -16777216.0
            return out

        dm.preprocess_text_embeds = _mock_preprocess
        patcher_for_make = patcher_with_init.clone()
        patcher_for_make.cached_patcher_init = None

        # Verify build_refiner_runner(patcher_with_lora) uses the in-memory model rather than reloading ckpt_path
        runner_lora = cv_nodes.build_refiner_runner(patcher_with_lora)
        lora_ref_out = runner_lora(torch.ones((6, cv.ENCODER_DIM), dtype=torch.float32))
        if float(lora_ref_out[0, 0].item()) != 16777216.0:
            errs.append("build_refiner_runner with LoRA patches used unpatched file runner instead of in-memory model")

        # Test encode_prompt_for_vector on t2va
        sample_prompt = (REPO / "prompt_bank" / "t2va_spec_typical.txt").read_text(encoding="utf-8").strip()
        enc_rows, enc_tags, enc_ids, dec_fn = cv_nodes.encode_prompt_for_vector(
            mock_clip, sample_prompt
        )
        if enc_rows.shape != (len(enc_ids), cv.ENCODER_DIM) or int(enc_tags.shape[0]) != len(enc_ids):
            errs.append(
                f"encode_prompt_for_vector shape mismatch: rows={tuple(enc_rows.shape)}, len(ids)={len(enc_ids)}"
            )

        # Test encode_prompt_for_vector on ref2va with mock EncodedReferences
        ref_enc_mod = cv_nodes._repo("reference_encode")

        orig_cond_prompt = ref_enc_mod.condition_prompt
        try:
            dummy_img = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
            ref_items = [{"type": "image", "data": dummy_img}]
            prefix_entries = mock_clip.tokenize("", minimax_ref_items=ref_items)["qwen3vl_32b"][0]
            # Prefix has `<Picture 1>: ` (5 tokens) + VISION_START + dict + VISION_END -> expand dict to 4 image pads (total 6 vision rows)
            n_label_toks = len(prefix_entries) - 3
            n_vis_rows = 6
            prefix_len = n_label_toks + n_vis_rows
            prefix_rows = torch.ones((1, prefix_len, cv.ENCODER_DIM), dtype=torch.float32) * 0.5

            def _mock_condition_prompt(encoded_obj, clean_txt):
                p_cond = mock_clip.encode_from_tokens_scheduled(mock_clip.tokenize(clean_txt))
                p_rows = p_cond[0][0]
                full_rows = torch.cat([prefix_rows, p_rows], dim=1)
                full_tags = torch.ones((full_rows.shape[1],), dtype=torch.long)
                full_tags[n_label_toks : n_label_toks + n_vis_rows] = 0
                return [[full_rows, {"pooled_output": None, "minimax_token_tags": full_tags}]]

            ref_enc_mod.condition_prompt = _mock_condition_prompt
            mock_encoded_refs = SimpleNamespace(
                ref_items=ref_items,
                ref_blocks=[],
                spans=[SimpleNamespace(rows=prefix_rows)],
            )
            r_rows, r_tags, r_ids, r_dec = cv_nodes.encode_prompt_for_vector(
                mock_clip, sample_prompt, encoded_references=mock_encoded_refs
            )
            r_spans = h3_spans.spans(r_ids, r_tags.tolist(), r_dec)
            if int(r_spans["vision"].sum().item()) != n_vis_rows or int(r_spans["label"].sum().item()) != n_label_toks:
                errs.append(
                    f"encode_prompt_for_vector ref2va span counts wrong: vision={int(r_spans['vision'].sum())}, label={int(r_spans['label'].sum())}"
                )

            # Write a 4-pair + 2-hold TOML file with unequal weights and test make_control_vector_from_pairs
            pair_a_clauses = (
                "clean neutral modern digital color",
                "flat modern digital video render",
                "crisp modern digital camera capture",
                "clinical modern digital sensor output",
            )
            pair_b_clauses = (
                "1974 16mm color reversal stock with halation",
                "1974 16mm celluloid film with heavy grain",
                "1974 16mm vintage reversal emulsion",
                "1974 16mm analog stock with gate weave",
            )
            pair_toml = tmp / "unit_pair.toml"
            pair_toml.write_text(
                '[vector]\n'
                'name = "unit_pair"\n'
                'quality = "16mm grain vs modern digital"\n'
                'a_side = "clean modern digital"\n'
                'b_side = "16mm 1974 grain"\n'
                'mode = "t2va"\n'
                'frame = "scene"\n'
                'exclude_shots = []\n\n'
                + "\n".join(
                    f'[[pair]]\nweight = {float(i * i * i)}\n'
                    f'a = """integrated_multimodal_description: [Shot 1] Scene {i} filmed with {pair_a_clauses[i - 1]}.\n\noverall_soundscape: Quiet room tone."""\n'
                    f'b = """integrated_multimodal_description: [Shot 1] Scene {i} filmed on {pair_b_clauses[i - 1]}.\n\noverall_soundscape: Quiet room tone."""\n'
                    for i in range(1, 5)
                )
                + "\n"
                + "\n".join(
                    f'[[hold]]\ngroup = "warm_tint"\nweight = 1.0\n'
                    f'a = """integrated_multimodal_description: [Shot 1] Hold {i} under neutral daylight balance.\n\noverall_soundscape: Quiet room tone."""\n'
                    f'b = """integrated_multimodal_description: [Shot 1] Hold {i} under warm amber tungsten cast.\n\noverall_soundscape: Quiet room tone."""\n'
                    for i in range(1, 3)
                ),
                encoding="utf-8",
            )

            for pool_mode in ("scene_text", "all_tag1", "from_first_diff", "diff_span"):
                saved_v, rep_v = cv_nodes.make_control_vector_from_pairs(
                    pair_path=pair_toml,
                    model=patcher_for_make,
                    clip=mock_clip,
                    pooling=pool_mode,
                    apply_holds=True,
                    output_name=f"unit_{pool_mode}",
                    output_dir=tmp,
                    encoded_references=mock_encoded_refs if pool_mode == "scene_text" else None,
                )
                loaded_t, loaded_m = cv.load_vector(saved_v)
                if "vector.post_refiner" not in loaded_t or "holds.post_refiner" not in loaded_t:
                    errs.append(f"make_control_vector_from_pairs({pool_mode}) missing expected tensors")
                if loaded_m["measured_under"]["pooling"] != pool_mode:
                    errs.append(f"make_control_vector_from_pairs pooling metadata mismatch: {loaded_m}")
                if not saved_v.with_suffix(".md").is_file():
                    errs.append(f"make_control_vector_from_pairs({pool_mode}) did not write companion .md sidecar")
                if not loaded_m.get("pairs", {}).get("pair_spans"):
                    errs.append(f"make_control_vector_from_pairs({pool_mode}) metadata['pairs'] missing pair_spans")

            # Verify float64 pooling precision on channel 0 and weighted quality_report on all_tag1 (noholds)
            saved_nh, rep_nh = cv_nodes.make_control_vector_from_pairs(
                pair_path=pair_toml,
                model=patcher_for_make,
                clip=mock_clip,
                pooling="all_tag1",
                apply_holds=False,
                output_name="unit_all_tag1_noholds",
                output_dir=tmp,
            )
            t_nh, m_nh = cv.load_vector(saved_nh)
            ps_check = cv.load_pairs(pair_toml)
            a_f64_list, b_f64_list, a_f32_list, b_f32_list = [], [], [], []
            for p_it in ps_check.pairs:
                ea, _, _, _ = cv_nodes.encode_prompt_for_vector(mock_clip, p_it.a)
                eb, _, _, _ = cv_nodes.encode_prompt_for_vector(mock_clip, p_it.b)
                ra = _mock_preprocess(ea.unsqueeze(0).to(dtype=torch.bfloat16))[0]
                rb = _mock_preprocess(eb.unsqueeze(0).to(dtype=torch.bfloat16))[0]
                a_f64_list.append(ra.to(dtype=torch.float64).mean(dim=0).to(dtype=torch.float32))
                b_f64_list.append(rb.to(dtype=torch.float64).mean(dim=0).to(dtype=torch.float32))
                a_f32_list.append(ra.mean(dim=0))
                b_f32_list.append(rb.mean(dim=0))
            w_check = torch.tensor([p.weight for p in ps_check.pairs], dtype=torch.float64)
            expected_f64_vec = cv.mean_difference(
                torch.stack(a_f64_list), torch.stack(b_f64_list), w_check
            )
            wrong_f32_vec = cv.mean_difference(
                torch.stack(a_f32_list), torch.stack(b_f32_list), w_check
            )
            if torch.allclose(expected_f64_vec, wrong_f32_vec, atol=1e-5) or not torch.allclose(
                t_nh["vector.post_refiner"], expected_f64_vec, atol=1e-6
            ):
                errs.append("make_control_vector_from_pairs pooled in float32 instead of float64")
            unweighted_qr = cv.quality_report(
                torch.stack(a_f64_list),
                torch.stack(b_f64_list),
                None,
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
            weighted_qr = cv.quality_report(
                torch.stack(a_f64_list),
                torch.stack(b_f64_list),
                w_check,
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
            if (
                m_nh["quality_report"]["split_half_cosine"] == unweighted_qr["split_half_cosine"]
                or m_nh["quality_report"]["split_half_cosine"] != weighted_qr["split_half_cosine"]
            ):
                errs.append("make_control_vector_from_pairs quality_report ignored pair weights")

            orig_cv_paths = cv_nodes.folder_paths.folder_names_and_paths.get(
                cv_nodes.CONTROL_VECTORS_FOLDER
            )
            try:
                cv_nodes.folder_paths.folder_names_and_paths[cv_nodes.CONTROL_VECTORS_FOLDER] = (
                    [str(tmp)],
                    {".safetensors"},
                )
                node_out = cv_nodes.MiniMaxH3MakeControlVector.execute(
                    pair_file=str(pair_toml),
                    model=patcher_for_make,
                    clip=mock_clip,
                    pooling="scene_text",
                    apply_holds=True,
                    output_name="unit_from_node",
                    overwrite=False,
                    encoded_references=None,
                )
                # Overwrite refusal check: running again with identical recipe and overwrite=False must refuse
                errs += _expect_refusal(
                    lambda: cv_nodes.MiniMaxH3MakeControlVector.execute(
                        pair_file=str(pair_toml),
                        model=patcher_for_make,
                        clip=mock_clip,
                        pooling="scene_text",
                        apply_holds=True,
                        output_name="unit_from_node",
                        overwrite=False,
                        encoded_references=None,
                    ),
                    "already exists",
                    "MiniMaxH3MakeControlVector refusal when file exists and overwrite=False",
                )
                # Different pooling writes a distinct filename without overwriting the first
                node_out_diff = cv_nodes.MiniMaxH3MakeControlVector.execute(
                    pair_file=str(pair_toml),
                    model=patcher_for_make,
                    clip=mock_clip,
                    pooling="from_first_diff",
                    apply_holds=True,
                    output_name="unit_from_node",
                    overwrite=False,
                    encoded_references=None,
                )
                if node_out_diff[0] == node_out[0] or not node_out_diff[0].endswith("__from_first_diff.safetensors"):
                    errs.append(
                        f"different pooling did not disambiguate filename: {node_out[0]!r} vs {node_out_diff[0]!r}"
                    )
                if not (tmp / node_out[0]).is_file() or not (tmp / node_out_diff[0]).is_file():
                    errs.append("both scene_text and from_first_diff vector files should coexist in tmp")
                # Zero-.toml in-place target_prompt node execution check
                node_out_words = cv_nodes.MiniMaxH3MakeControlVector.execute(
                    pair_file="",
                    model=patcher_for_make,
                    clip=mock_clip,
                    pooling="diff_span",
                    apply_holds=True,
                    output_name="",
                    overwrite=False,
                    encoded_references=None,
                    words="a medium shot frames a woman -> an erratic handheld camera shake frames a woman",
                    mode="auto",
                    target_prompt=sample_prompt,
                )
                if not node_out_words[0].startswith("erratic_handheld_shake") or not (tmp / node_out_words[0]).is_file():
                    errs.append(f"MiniMaxH3MakeControlVector(words=...) failed: {node_out_words[0]!r}")
            finally:
                if orig_cv_paths is not None:
                    cv_nodes.folder_paths.folder_names_and_paths[
                        cv_nodes.CONTROL_VECTORS_FOLDER
                    ] = orig_cv_paths
            if (
                not isinstance(node_out[0], str)
                or not node_out[0].startswith("unit_from_node__")
                or ":" in node_out[0]
            ):
                errs.append(f"MiniMaxH3MakeControlVector.execute returned unexpected output: {node_out[0]!r}")
            if not (tmp / node_out[0]).is_file():
                errs.append(f"MiniMaxH3MakeControlVector.execute did not write to redirected folder: {node_out[0]!r}")

            # Test _build_padding_prompt and cmd_grade (with mock _dit_runner)
            edit_prompt_txt = tmp / "unit_edit.txt"
            edit_prompt_txt.write_text(
                sample_prompt.replace(
                    "Live-action, cinematic,",
                    "Live-action, cinematic, recorded on 1974 16mm color reversal film stock with heavy celluloid grain and halation,",
                ),
                encoding="utf-8",
            )
            base_prompt_txt = tmp / "unit_base.txt"
            base_prompt_txt.write_text(sample_prompt, encoding="utf-8")

            pad_built = bench_cv._build_padding_prompt(sample_prompt, 12, mock_clip)
            len_base_tok = bench_cv._prompt_token_count(mock_clip, sample_prompt)
            len_pad_tok = bench_cv._prompt_token_count(mock_clip, pad_built)
            if len_pad_tok != len_base_tok + 12:
                errs.append(f"_build_padding_prompt expected {len_base_tok + 12} tokens, got {len_pad_tok}")

            import contextlib
            import io
            import os
            import subprocess

            # Standalone CLI --help check (verifies sys.path bootstrap when run directly as a script)
            proc_help = subprocess.run(
                [sys.executable, "-B", str(REPO / "bench" / "control_vectors.py"), "--help"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc_help.returncode != 0:
                errs.append(
                    f"bench/control_vectors.py --help failed with rc={proc_help.returncode}: {proc_help.stderr!r}"
                )

            orig_load_clip = bench_cv._load_clip
            orig_build_ref = cv_nodes.build_refiner_runner
            orig_get_full = bench_cv.folder_paths.get_full_path_or_raise
            try:
                bench_cv._load_clip = lambda enc_name, device="default": mock_clip
                cv_nodes.build_refiner_runner = lambda src, **kw: (
                    lambda x_2d: torch.nn.functional.linear(x_2d.to(dtype=torch.float32), proj_w, proj_b)
                )
                bench_cv.folder_paths.get_full_path_or_raise = lambda folder, name: str(ckpt_path)

                # Test cmd_make (--include-encoder)
                cli_made_v = tmp / "cli_made.safetensors"
                buf_mk_out, buf_mk_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_mk_out), contextlib.redirect_stderr(buf_mk_err):
                    rc_mk = bench_cv.main(
                        [
                            "make",
                            str(pair_toml),
                            "--include-encoder",
                            "--out",
                            str(cli_made_v),
                        ]
                    )
                if rc_mk != 0 or not cli_made_v.is_file():
                    errs.append(f"cmd_make failed: rc={rc_mk}, err={buf_mk_err.getvalue()!r}")
                else:
                    mk_t, mk_m = cv.load_vector(cli_made_v)
                    if "vector.encoder" not in mk_t or mk_m["points"]["encoder"]["unit"] != "median_row_norm":
                        errs.append(f"cmd_make --include-encoder missing vector.encoder or wrong unit: {mk_m['points']}")
                    enc_v_norm = float(torch.linalg.vector_norm(mk_t["vector.encoder"].to(torch.float64)).item())
                    if abs(enc_v_norm - 1.0) > 1e-4:
                        errs.append(f"cmd_make vector.encoder is not unit-normalized: norm={enc_v_norm}")

                # Verify cmd_make refuses when --target-prompt is combined with a .toml file without --words
                buf_bad_tp_out, buf_bad_tp_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_bad_tp_out), contextlib.redirect_stderr(buf_bad_tp_err):
                    rc_bad_tp = bench_cv.main(
                        [
                            "make",
                            str(pair_toml),
                            "--target-prompt",
                            sample_prompt,
                            "--out",
                            str(tmp / "bad_splice.safetensors"),
                        ]
                    )
                if rc_bad_tp != 2 or "--target-prompt requires --words" not in buf_bad_tp_err.getvalue():
                    errs.append(
                        f"cmd_make with .toml + --target-prompt expected rc=2 refusal, got rc={rc_bad_tp}, err={buf_bad_tp_err.getvalue()!r}"
                    )

                # Test cmd_make with --lora, default_vector_filename(loras=...), and h3_lora_branches recipe parity
                mock_lora_path = tmp / "mock_pika_lora.safetensors"
                save_file(
                    {
                        "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones((2, 8), dtype=torch.bfloat16),
                        "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones((8, 2), dtype=torch.bfloat16),
                        "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
                    },
                    str(mock_lora_path),
                )
                cli_made_lora_v = tmp / "cli_made_lora.safetensors"
                buf_mkl_out, buf_mkl_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_mkl_out), contextlib.redirect_stderr(buf_mkl_err):
                    rc_mkl = bench_cv.main(
                        [
                            "make",
                            str(pair_toml),
                            "--pooling",
                            "diff_span",
                            "--lora",
                            f"{mock_lora_path}:1.0",
                            "--out",
                            str(cli_made_lora_v),
                        ]
                    )
                if rc_mkl != 0 or not cli_made_lora_v.is_file():
                    errs.append(f"cmd_make --lora failed: rc={rc_mkl}, err={buf_mkl_err.getvalue()!r}")
                else:
                    _, mkl_m = cv.load_vector(cli_made_lora_v)
                    if mkl_m["measured_under"]["pooling"] != "diff_span":
                        errs.append(f"cmd_make --pooling diff_span metadata mismatch: {mkl_m['measured_under']}")
                    if not cli_made_lora_v.with_suffix(".md").is_file():
                        errs.append("cmd_make did not write companion .md sidecar")
                    elif "Per-Pair Token Spans" not in cli_made_lora_v.with_suffix(".md").read_text(encoding="utf-8"):
                        errs.append("cmd_make companion .md sidecar missing Per-Pair Token Spans section")
                    patcher_with_rec_branch = patcher_with_init.clone()
                    patcher_with_rec_branch.model_options["transformer_options"]["h3_lora_branches"] = (
                        {
                            "path": str(mock_lora_path),
                            "file": mock_lora_path.name,
                            "strength": 1.0,
                            "modules": "all",
                            "blocks": "all",
                            "start_percent": 0.0,
                            "end_percent": 1.0,
                        },
                    )
                    rec_from_node_branch = cv_nodes.extract_model_recipe(patcher_with_rec_branch)
                    if mkl_m["recipe"]["loras"] != rec_from_node_branch["loras"]:
                        errs.append(
                            f"cmd_make --lora recipe {mkl_m['recipe']['loras']} != h3_lora_branches recipe {rec_from_node_branch['loras']}"
                        )
                    fn_base = cv.default_vector_filename("film", "fl2va")
                    fn_lora = cv.default_vector_filename("film", "fl2va", loras=mkl_m["recipe"]["loras"])
                    if fn_base == fn_lora or "__lora_" not in fn_lora:
                        errs.append(f"default_vector_filename did not disambiguate LoRA recipe: {fn_base!r} vs {fn_lora!r}")

                # Test one-shot cmd_build from a Markdown (.md) file (generating .toml, _test_prompt.txt, and multi-pooling .safetensors)
                build_md_path = tmp / "oneshot_build.md"
                build_md_path.write_text(
                    "### Test Scene Prompt\n\n"
                    "integrated_multimodal_description: [Shot 1] Wide shot as (S1) says, <d>Hello.</d>\n\n"
                    "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A\n\n"
                    "### Contrastive Pairs (1–2)\n\n"
                    "#### Pair 1\n##### Prompt A (Baseline)\n"
                    "integrated_multimodal_description: [Shot 1] Scene 1 filmed with clean neutral modern digital color.\n\noverall_soundscape: Quiet room tone.\n\n"
                    "##### Prompt B (Steered)\n"
                    "integrated_multimodal_description: [Shot 1] Scene 1 filmed with 1974 16mm color reversal stock.\n\noverall_soundscape: Quiet room tone.\n\n"
                    "#### Pair 2\n##### Prompt A (Baseline)\n"
                    "integrated_multimodal_description: [Shot 1] Scene 2 filmed with flat modern digital video render.\n\noverall_soundscape: Quiet room tone.\n\n"
                    "##### Prompt B (Steered)\n"
                    "integrated_multimodal_description: [Shot 1] Scene 2 filmed with 1974 16mm celluloid film grain.\n\noverall_soundscape: Quiet room tone.\n",
                    encoding="utf-8",
                )
                build_toml_dir = tmp / "build_toml_dir"
                build_vec_dir = tmp / "build_vec_dir"
                buf_bld_out, buf_bld_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_bld_out), contextlib.redirect_stderr(buf_bld_err):
                    rc_bld = bench_cv.main(
                        [
                            "build",
                            str(build_md_path),
                            "--families",
                            "fl2va",
                            "--pooling",
                            "diff_span,from_first_diff",
                            "--lora",
                            f"{mock_lora_path}:1.0",
                            "--toml-out",
                            str(build_toml_dir),
                            "--out-dir",
                            str(build_vec_dir),
                        ]
                    )
                if rc_bld != 0:
                    errs.append(f"cmd_build failed: rc={rc_bld}, err={buf_bld_err.getvalue()!r}")
                else:
                    if not (build_toml_dir / "oneshot_build.toml").is_file():
                        errs.append("cmd_build did not write oneshot_build.toml")
                    if not (build_toml_dir / "oneshot_build_test_prompt.txt").is_file():
                        errs.append("cmd_build did not write oneshot_build_test_prompt.txt")
                    built_sfs = sorted(build_vec_dir.glob("*.safetensors"))
                    if len(built_sfs) != 2:
                        errs.append(f"cmd_build expected 2 .safetensors files, got {[p.name for p in built_sfs]}")
                    # Running build again with identical .md and overwrite=False must refuse on existing .safetensors
                    buf_bld2_out, buf_bld2_err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(buf_bld2_out), contextlib.redirect_stderr(buf_bld2_err):
                        rc_bld2 = bench_cv.main(
                            [
                                "build",
                                str(build_md_path),
                                "--families",
                                "fl2va",
                                "--pooling",
                                "diff_span",
                                "--lora",
                                f"{mock_lora_path}:1.0",
                                "--toml-out",
                                str(build_toml_dir),
                                "--out-dir",
                                str(build_vec_dir),
                            ]
                        )
                    if rc_bld2 != 2 or "already exists" not in buf_bld2_err.getvalue():
                        errs.append(
                            f"cmd_build duplicate without --overwrite expected rc=2 and 'already exists', got rc={rc_bld2}, err={buf_bld2_err.getvalue()!r}"
                        )

                # Verify CLI parser and family resolution
                for img_alias in ("image", "p-h3-image", "t2i", "image_edit"):
                    if bench_cv._resolve_requested_families(img_alias) != ["fl2va"]:
                        errs.append(f"_resolve_requested_families({img_alias!r}) did not resolve to ['fl2va']")

                # Verify explicit CLI method subcommands (from-prompt, from-toml, from-words, from-images, build-j-lens)
                cli_parser = bench_cv.build_parser()
                for sub_argv, exp_cmd in (
                    (["from-prompt", "--target-prompt", "p", "--words", "a -> b"], "from-prompt"),
                    (["from-prompt", "--target-prompt-file", "t2va_night_porter", "--words", "a -> b"], "from-prompt"),
                    (["from-toml", "film_16mm_1974"], "from-toml"),
                    (["from-words", "blizzard, snow"], "from-words"),
                    (["from-images", "--image-a", "a.png", "--image-b", "b.png"], "from-images"),
                    (["build-j-lens", "--verify-only"], "build-j-lens"),
                ):
                    ns = cli_parser.parse_args(sub_argv)
                    if ns.cmd != exp_cmd:
                        errs.append(f"CLI subcommand {sub_argv[0]!r} parsed to {ns.cmd!r}, expected {exp_cmd!r}")
                porter_cli_path = REPO / "prompt_bank" / "t2va_night_porter.txt"
                ns_tp_file = cli_parser.parse_args(
                    ["from-prompt", "--target-prompt-file", str(porter_cli_path), "--words", "a -> b"]
                )
                resolved_tp_file = bench_cv._resolve_cli_target_prompt(ns_tp_file)
                if pair_comp.infer_compile_mode(None, target_prompt=resolved_tp_file) != "t2va":
                    errs.append("infer_compile_mode on resolved --target-prompt-file failed to infer 't2va'")
                ns_tp_stem = cli_parser.parse_args(
                    ["from-prompt", "--target-prompt-file", "ref2va_woman_blonde_street", "--words", "a -> b"]
                )
                resolved_tp_stem = bench_cv._resolve_cli_target_prompt(ns_tp_stem)
                if pair_comp.infer_compile_mode(None, target_prompt=resolved_tp_stem) != "ref2va":
                    errs.append("infer_compile_mode on resolved prompt_bank --target-prompt-file stem failed to infer 'ref2va'")
                ps_from_path = bench_cv._compile_words_or_spec_to_pairset(
                    "A night porter in scrubs and a fleece -> A NASA astronaut in a bulky white EVA spacesuit",
                    mode=pair_comp.infer_compile_mode(None, target_prompt=resolved_tp_file),
                    target_prompt=resolved_tp_file,
                    bank_scenes=2,
                )
                if ps_from_path.mode != "t2va" or len(ps_from_path.pairs) != 3:
                    errs.append(
                        f"CLI --target-prompt-file + bank_scenes=2 failed: "
                        f"mode={ps_from_path.mode!r}, n_pairs={len(ps_from_path.pairs)}"
                    )
                if j_lens.bundle_available():
                    buf_jl_out, buf_jl_err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(buf_jl_out), contextlib.redirect_stderr(buf_jl_err):
                        rc_jl = bench_cv.main(["build-j-lens", "--verify-only"])
                    if rc_jl != 0:
                        errs.append(f"build-j-lens --verify-only failed with rc={rc_jl}: {buf_jl_err.getvalue()!r}")

                words_vec_dir = tmp / "words_vec_dir"
                tortoise_txt = (REPO / "prompt_bank" / "t2va_offpath_tortoise.txt").read_text(encoding="utf-8").strip()
                tp_two_scenes = (
                    sample_prompt.replace("five-second take", "short take")
                    + "\n---\n"
                    + tortoise_txt.replace("five-second take", "short take")
                )
                buf_wb_out, buf_wb_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_wb_out), contextlib.redirect_stderr(buf_wb_err):
                    rc_wb = bench_cv.main(
                        [
                            "from-prompt",
                            "--target-prompt",
                            tp_two_scenes,
                            "--words",
                            "short take -> blizzard snowdrift indoors",
                            "--families",
                            "fl2va",
                            "--pooling",
                            "diff_span",
                            "--dit",
                            ckpt_path.name,
                            "--no-lora",
                            "--out-dir",
                            str(words_vec_dir),
                        ]
                    )
                if rc_wb != 0:
                    errs.append(f"cmd_build --target-prompt --words failed: rc_wb={rc_wb}, err={buf_wb_err.getvalue()!r}")
                elif len(list(words_vec_dir.glob("*.safetensors"))) != 1:
                    errs.append(f"cmd_build --target-prompt --words expected 1 .safetensors file, got {list(words_vec_dir.glob('*.safetensors'))}")
                else:
                    # Verify non-TOML compiled --words vector stores pairs_ab/words_spec and rebuilds identically
                    wb_st = next(words_vec_dir.glob("*.safetensors"))
                    wb_stem = wb_st.name.split("__")[0]
                    wb_t_orig, wb_m_orig = cv.load_vector(wb_st)
                    wb_pairs_meta = wb_m_orig.get("pairs") or {}
                    if wb_pairs_meta.get("differing_sections") != ["integrated_multimodal_description"]:
                        errs.append(
                            f"saved vector metadata['pairs']['differing_sections'] expected ['integrated_multimodal_description'], got {wb_pairs_meta.get('differing_sections')!r}"
                        )
                    reverb_ps = cv.load_pairs(REPO / "prompt_bank" / "control_vectors" / "reverb_cathedral.toml")
                    if cv.pair_set_differing_sections(reverb_ps.pairs, frame=reverb_ps.frame) != ["overall_soundscape"]:
                        errs.append(
                            f"pair_set_differing_sections(reverb_cathedral) expected ['overall_soundscape'], got {cv.pair_set_differing_sections(reverb_ps.pairs, frame=reverb_ps.frame)!r}"
                        )
                    if not isinstance(wb_pairs_meta.get("pairs_ab"), list) or len(wb_pairs_meta["pairs_ab"]) != 2:
                        errs.append(f"cmd_build --words did not persist 2 pairs_ab in metadata['pairs']: {wb_pairs_meta.keys()}")
                    with contextlib.redirect_stdout(buf_wb_out), contextlib.redirect_stderr(buf_wb_err):
                        rc_reb_wb = bench_cv.main(
                            [
                                "rebuild",
                                wb_stem,
                                "--dir",
                                str(words_vec_dir),
                                "--toml-dir",
                                str(words_vec_dir),
                                "--families",
                                "fl2va",
                                "--dit",
                                ckpt_path.name,
                                "--no-lora",
                            ]
                        )
                    if rc_reb_wb != 0:
                        errs.append(f"cmd_rebuild on compiled --words vector failed with rc={rc_reb_wb}")
                    else:
                        wb_t_reb, wb_m_reb = cv.load_vector(wb_st)
                        if not torch.allclose(wb_t_orig["vector.post_refiner"], wb_t_reb["vector.post_refiner"], atol=1e-5):
                            errs.append("cmd_rebuild on compiled --words vector did not reproduce identical vector.post_refiner")
                        if wb_m_reb.get("pairs", {}).get("words_spec") != "short take -> blizzard snowdrift indoors":
                            errs.append(f"cmd_rebuild on compiled --words vector lost words_spec: {wb_m_reb.get('pairs', {}).get('words_spec')!r}")
                        # Also verify that mutating pairs_ab[0]['weight'] on disk changes the rebuilt vector (proving pairs_ab reconstruction is used over words_spec recompilation)
                        wb_m_mut = json.loads(json.dumps(wb_m_reb))
                        wb_m_mut["pairs"]["pairs_ab"][0]["weight"] = 9.0
                        cv.save_vector(wb_st, wb_t_reb, wb_m_mut, overwrite=True)
                        with contextlib.redirect_stdout(buf_wb_out), contextlib.redirect_stderr(buf_wb_err):
                            bench_cv.main(
                                [
                                    "rebuild",
                                    wb_stem,
                                    "--dir",
                                    str(words_vec_dir),
                                    "--toml-dir",
                                    str(words_vec_dir),
                                    "--families",
                                    "fl2va",
                                    "--dit",
                                    ckpt_path.name,
                                    "--no-lora",
                                ]
                            )
                        _, wb_m_after_mut = cv.load_vector(wb_st)
                        if float(wb_m_after_mut["pairs"]["pair_spans"][0]["weight"]) != 9.0:
                            errs.append(f"cmd_rebuild did not reconstruct PairSet from pairs_ab (expected pair 0 weight 9.0, got {wb_m_after_mut['pairs']['pair_spans'][0]['weight']})")

                # Verify _attach_refiner_lora_branches matches float32 merged TokenRefiner and refuses duplicate patches
                import copy as _copy
                import comfy.ops as _cops

                tiny_ref = mm_mod.TokenRefiner(
                    2, 32, 2, 16, 48, 1e-6, 1e-6, 1e-6,
                    dtype=torch.float32, device="cpu", operations=_cops.manual_cast,
                )
                with torch.no_grad():
                    g_tr = torch.Generator().manual_seed(19)
                    for p_name, p_val in tiny_ref.named_parameters():
                        p_val.fill_(1.0) if "norm" in p_name else p_val.normal_(0.0, 0.05, generator=g_tr)
                tiny_ref.eval()
                ref_lora_sd: dict[str, torch.Tensor] = {}
                for b_idx in range(2):
                    for kind in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                        sub_p = f"blocks.{b_idx}.{kind}"
                        w_t = tiny_ref.get_submodule(sub_p).weight
                        k_full = f"diffusion_model.token_refiner.{sub_p}"
                        ref_lora_sd[f"{k_full}.lora_A.weight"] = torch.randn((2, w_t.shape[1]), generator=g_tr) * 0.1
                        ref_lora_sd[f"{k_full}.lora_B.weight"] = torch.randn((w_t.shape[0], 2), generator=g_tr) * 0.1
                        ref_lora_sd[f"{k_full}.alpha"] = torch.tensor(3.0)
                ref_lora_sd["h3_pdd.adaln_baked.blocks.0"] = torch.tensor(1.0)
                ref_lora_sd["diffusion_model.blocks.0.attn1.to_q.lora_A.weight"] = torch.ones((2, 8))
                ref_lora_sd["diffusion_model.blocks.0.attn1.to_q.lora_B.weight"] = torch.ones((8, 2))
                tiny_lora_file = tmp / "tiny_ref_lora.safetensors"
                save_file(ref_lora_sd, str(tiny_lora_file))
                tiny_ref_merged = _copy.deepcopy(tiny_ref)
                with torch.no_grad():
                    for b_idx in range(2):
                        for kind in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                            sub_p = f"blocks.{b_idx}.{kind}"
                            k_full = f"diffusion_model.token_refiner.{sub_p}"
                            tiny_ref_merged.get_submodule(sub_p).weight += (
                                0.75 * (3.0 / 2.0) * (ref_lora_sd[f"{k_full}.lora_B.weight"] @ ref_lora_sd[f"{k_full}.lora_A.weight"])
                            )
                tiny_x = torch.randn((5, 32), generator=g_tr, dtype=torch.float32)
                with torch.no_grad():
                    out_unpatched = tiny_ref(tiny_x).clone()
                    out_merged_ref = tiny_ref_merged(tiny_x).clone()
                cv_nodes._attach_refiner_lora_branches(
                    tiny_ref, [{"path": str(tiny_lora_file), "file": tiny_lora_file.name, "strength": 0.75}]
                )
                with torch.no_grad():
                    out_branched_ref = tiny_ref(tiny_x).clone()
                ref_branch_err = float((out_branched_ref - out_merged_ref).norm() / out_merged_ref.norm())
                ref_move = float((out_merged_ref - out_unpatched).norm() / out_unpatched.norm())
                if ref_move < 1e-3 or ref_branch_err > 1e-5:
                    errs.append(
                        f"_attach_refiner_lora_branches mismatch vs merged TokenRefiner: move={ref_move:.3g}, err={ref_branch_err:.3g}"
                    )
                errs += _expect_refusal(
                    lambda: cv_nodes._attach_refiner_lora_branches(
                        tiny_ref, [{"path": str(tiny_lora_file), "file": tiny_lora_file.name, "strength": 0.75}]
                    ),
                    "already taken",
                    "_attach_refiner_lora_branches refusal when patching already-patched TokenRefiner",
                )
                # Verify _attach_refiner_lora_branches chains two LoRAs within a single loras list (Trap 1A)
                tiny_lora_file2 = tmp / "tiny_ref_lora2.safetensors"
                ref_lora_sd2 = {
                    k: (v * 0.4 if k.endswith("lora_B.weight") else v)
                    for k, v in ref_lora_sd.items()
                }
                save_file(ref_lora_sd2, str(tiny_lora_file2))
                tiny_ref_dual_merged = _copy.deepcopy(tiny_ref_merged)
                with torch.no_grad():
                    for b_idx in range(2):
                        for kind in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                            sub_p = f"blocks.{b_idx}.{kind}"
                            k_full = f"diffusion_model.token_refiner.{sub_p}"
                            tiny_ref_dual_merged.get_submodule(sub_p).weight += (
                                0.5 * (3.0 / 2.0) * (ref_lora_sd2[f"{k_full}.lora_B.weight"] @ ref_lora_sd2[f"{k_full}.lora_A.weight"])
                            )
                    out_dual_merged_ref = tiny_ref_dual_merged(tiny_x).clone()
                tiny_ref_fresh = mm_mod.TokenRefiner(
                    2, 32, 2, 16, 48, 1e-6, 1e-6, 1e-6,
                    dtype=torch.float32, device="cpu", operations=_cops.manual_cast,
                )
                tiny_ref_fresh.load_state_dict(
                    {k: v for k, v in tiny_ref_merged.state_dict().items()}
                )
                tiny_ref_fresh.eval()
                # Reset weights to unpatched tiny_ref base weights before attaching both branches
                with torch.no_grad():
                    for b_idx in range(2):
                        for kind in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                            sub_p = f"blocks.{b_idx}.{kind}"
                            k_full = f"diffusion_model.token_refiner.{sub_p}"
                            tiny_ref_fresh.get_submodule(sub_p).weight -= (
                                0.75 * (3.0 / 2.0) * (ref_lora_sd[f"{k_full}.lora_B.weight"] @ ref_lora_sd[f"{k_full}.lora_A.weight"])
                            )
                cv_nodes._attach_refiner_lora_branches(
                    tiny_ref_fresh,
                    [
                        {"path": str(tiny_lora_file), "file": tiny_lora_file.name, "strength": 0.75},
                        {"path": str(tiny_lora_file2), "file": tiny_lora_file2.name, "strength": 0.5},
                    ],
                )
                with torch.no_grad():
                    out_dual_branched_ref = tiny_ref_fresh(tiny_x).clone()
                dual_branch_err = float((out_dual_branched_ref - out_dual_merged_ref).norm() / out_dual_merged_ref.norm())
                dual_vs_single = float((out_dual_branched_ref - out_merged_ref).norm() / out_merged_ref.norm())
                if dual_vs_single < 1e-4 or dual_branch_err > 1e-5:
                    errs.append(
                        f"_attach_refiner_lora_branches dual-LoRA chain mismatch: vs_single={dual_vs_single:.3g}, err={dual_branch_err:.3g}"
                    )

                # Test inspect --check-dit (and legacy --check-unet alias) compatibility check and --meter
                buf_ic_out, buf_ic_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_ic_out), contextlib.redirect_stderr(buf_ic_err):
                    rc_ic = bench_cv.main(["inspect", str(v1_path), "--check-dit", ckpt_path.name])
                    rc_ic_legacy = bench_cv.main(["inspect", str(v1_path), "--check-unet", ckpt_path.name])
                if rc_ic != 0 or rc_ic_legacy != 0 or "compatibility   : ok" not in buf_ic_out.getvalue():
                    errs.append(
                        f"inspect --check-dit/--check-unet compatibility failed: rc={rc_ic}/{rc_ic_legacy}, out={buf_ic_out.getvalue()!r}, err={buf_ic_err.getvalue()!r}"
                    )

                buf_m_out, buf_m_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_m_out), contextlib.redirect_stderr(buf_m_err):
                    rc_meter = bench_cv.main(
                        ["inspect", str(v1_path), "--meter", str(base_prompt_txt), "--meter", str(edit_prompt_txt)]
                    )
                if rc_meter != 0 or "Meter Projection" not in buf_m_out.getvalue():
                    errs.append(f"inspect --meter failed: rc={rc_meter}, out={buf_m_out.getvalue()!r}, err={buf_m_err.getvalue()!r}")

                # Test cmd_grade with mock DiT runner (including vision tags on rows 0..1 and distinct channel weights)
                ch_weights = torch.linspace(0.2, 1.8, cv.POST_REFINER_DIM, dtype=torch.float32)

                def _mock_dit_runner(m_obj, cond_obj, step_idx):
                    c_rows = cond_obj[0][0].to(dtype=torch.float32)
                    c_tags = cond_obj[0][1]["minimax_token_tags"].clone()
                    if c_tags.numel() >= 2:
                        c_tags[:2] = 0
                    ref_ctx = torch.nn.functional.linear(c_rows, proj_w, proj_b)
                    m_obj.invoke_diffusion_model(
                        torch.zeros((1, 4), dtype=torch.float32),
                        torch.tensor([float(step_idx)]),
                        ref_ctx,
                        minimax_payload={"text_token_tags": c_tags},
                    )
                    steered = dm.last_context[0].to(dtype=torch.float32) * ch_weights.unsqueeze(0)
                    vid_pred = steered.mean(dim=0)[:64] + 1.0
                    aud_pred = steered.mean(dim=0)[64:128] + 2.0
                    return 0.5, {"video": vid_pred, "audio": aud_pred}

                grade_json = tmp / "grade_out.json"
                grade_args = bench_cv.build_parser().parse_args(
                    [
                        "grade",
                        "--vector",
                        str(v1_path),
                        "--prompt",
                        str(base_prompt_txt),
                        "--edit-prompt",
                        str(edit_prompt_txt),
                        "--pairs",
                        str(pair_toml),
                        "--strengths",
                        "0,1.0,-1.0",
                        "--probe-steps",
                        "0",
                        "--compare-unmasked",
                        "--out",
                        str(grade_json),
                    ]
                )
                grade_args._dit_runner = _mock_dit_runner
                grade_args._base_model = patcher_with_init
                buf_g_out = io.StringIO()
                with contextlib.redirect_stdout(buf_g_out):
                    rc_grade = bench_cv.cmd_grade(grade_args)
                if rc_grade != 0 or not grade_json.is_file():
                    errs.append(f"cmd_grade failed: rc={rc_grade}")
                else:
                    g_data = json.loads(grade_json.read_text(encoding="utf-8"))
                    dit_step0 = g_data["dit_prediction_movement"][0]
                    if (
                        "flip_null" not in dit_step0["arms"]
                        or "random_null" not in dit_step0["arms"]
                        or "vector_s1" not in dit_step0["arms"]
                        or "masked_vs_unmasked_s1" not in dit_step0["arms"]
                        or "delta_cosine_with_edit" not in dit_step0["arms"]["vector_s1"]["video"]
                        or dit_step0.get("matched_strength_video") is None
                        or "provenance" not in g_data
                    ):
                        errs.append(f"cmd_grade JSON missing expected DiT fields: {dit_step0}")
                    else:
                        if float(dit_step0["arms"]["masked_vs_unmasked_s1"]["video"]["relative_l2"]) <= 0.0:
                            errs.append(
                                f"cmd_grade masked_vs_unmasked_s1 did not differ despite vision rows: {dit_step0['arms']['masked_vs_unmasked_s1']}"
                            )
                        flip_cos = float(dit_step0["arms"]["flip_null"]["video"]["delta_cosine_with_edit"])
                        vec_cos = float(dit_step0["arms"]["vector_s1"]["video"]["delta_cosine_with_edit"])
                        if abs(flip_cos - vec_cos) < 1e-5:
                            errs.append(
                                f"cmd_grade flip_null arm produced identical cosine to vector_s1 ({flip_cos} == {vec_cos})"
                            )
                        ed_v = float(dit_step0["arms"]["edit_prompt"]["video"]["relative_l2"])
                        s1_v = float(dit_step0["arms"]["vector_s1"]["video"]["relative_l2"])
                        if abs(ed_v - s1_v) < 1e-4 or abs(float(dit_step0["matched_strength_video"]) - (ed_v / s1_v)) > 1e-5:
                            errs.append(
                                f"cmd_grade matched_strength_video={dit_step0['matched_strength_video']} != ed_v/s1_v ({ed_v}/{s1_v})"
                            )
                        if g_data.get("provenance", {}).get("mask") != "scene_only":
                            errs.append(f"cmd_grade default provenance.mask expected 'scene_only', got {g_data.get('provenance')}")

                # Also test cmd_grade --mask look_only
                grade_look_json = tmp / "grade_look_out.json"
                grade_look_args = bench_cv.build_parser().parse_args(
                    [
                        "grade",
                        "--vector",
                        str(v1_path),
                        "--prompt",
                        str(base_prompt_txt),
                        "--mask",
                        "look_only",
                        "--strengths",
                        "0,1.0",
                        "--probe-steps",
                        "0",
                        "--out",
                        str(grade_look_json),
                    ]
                )
                grade_look_args._dit_runner = _mock_dit_runner
                grade_look_args._base_model = patcher_with_init
                with contextlib.redirect_stdout(io.StringIO()):
                    rc_grade_look = bench_cv.cmd_grade(grade_look_args)
                if rc_grade_look != 0 or not grade_look_json.is_file():
                    errs.append(f"cmd_grade --mask look_only failed: rc={rc_grade_look}")

                # Verify cmd_grade refuses if strength=0.0 null check fails
                call_counter = [0]

                def _broken_zero_dit_runner(m_obj, cond_obj, step_idx):
                    call_counter[0] += 1
                    sig_val, preds = _mock_dit_runner(m_obj, cond_obj, step_idx)
                    if call_counter[0] == 2:
                        preds = {"video": preds["video"] + 0.5, "audio": preds["audio"]}
                    return sig_val, preds

                bad_grade_args = bench_cv.build_parser().parse_args(
                    ["grade", "--vector", str(v1_path), "--prompt", str(base_prompt_txt), "--probe-steps", "0"]
                )
                bad_grade_args._dit_runner = _broken_zero_dit_runner
                bad_grade_args._base_model = patcher_with_init
                def _run_bad_grade():
                    with contextlib.redirect_stdout(io.StringIO()):
                        return bench_cv.cmd_grade(bad_grade_args)

                errs += _expect_refusal(
                    _run_bad_grade,
                    "strength=0.0 null check failed",
                    "cmd_grade refusal when strength=0.0 output differs from base",
                )
            finally:
                bench_cv._load_clip = orig_load_clip
                cv_nodes.build_refiner_runner = orig_build_ref
                bench_cv.folder_paths.get_full_path_or_raise = orig_get_full
        finally:
            ref_enc_mod.condition_prompt = orig_cond_prompt

        # CLI inspect subcommand check
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            rc_ok = bench_cv.main(["inspect", str(v1_path)])
        if rc_ok != 0 or "channel_census" not in buf_out.getvalue():
            errs.append(f"bench/control_vectors.py inspect returned {rc_ok}, out={buf_out.getvalue()!r}")
        bad_safetensors = tmp / "bad.safetensors"
        save_file({"wrong": torch.zeros((4,), dtype=torch.float32)}, str(bad_safetensors))
        buf_out2, buf_err2 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out2), contextlib.redirect_stderr(buf_err2):
            rc_bad = bench_cv.main(["inspect", str(bad_safetensors)])
        if rc_bad != 2 or "missing required key 'format'" not in buf_err2.getvalue():
            errs.append(f"bench/control_vectors.py inspect on bad file returned {rc_bad}, err={buf_err2.getvalue()!r}")

        # CLI from-md / import-md subcommand check (file path --out, directory --out, and malformed refusal)
        cli_md = tmp / "cli_md_vec.md"
        cli_md.write_text(
            "### Test Scene Prompt\n\n"
            "integrated_multimodal_description: [Shot 1] Wide shot as (S1) says, <d>Hello.</d>\n\n"
            "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A\n\n"
            "### Contrastive Pairs (1–2)\n\n"
            "#### Pair 1\n##### Prompt A (Baseline)\n"
            "integrated_multimodal_description: [Shot 1] Scene 1 filmed with clean neutral modern digital color.\n\noverall_soundscape: Quiet room tone.\n\n"
            "##### Prompt B (Steered)\n"
            "integrated_multimodal_description: [Shot 1] Scene 1 filmed with 1974 16mm color reversal stock.\n\noverall_soundscape: Quiet room tone.\n\n"
            "#### Pair 2\n##### Prompt A (Baseline)\n"
            "integrated_multimodal_description: [Shot 1] Scene 2 filmed with flat modern digital video render.\n\noverall_soundscape: Quiet room tone.\n\n"
            "##### Prompt B (Steered)\n"
            "integrated_multimodal_description: [Shot 1] Scene 2 filmed with 1974 16mm celluloid film grain.\n\noverall_soundscape: Quiet room tone.\n",
            encoding="utf-8",
        )
        cli_toml_out = tmp / "cli_md_vec.toml"
        buf_fm_out, buf_fm_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_fm_out), contextlib.redirect_stderr(buf_fm_err):
            rc_fm = bench_cv.main(["from-md", str(cli_md), "--out", str(cli_toml_out)])
        if rc_fm != 0 or not cli_toml_out.is_file() or "Control Vector From-MD: cli_md_vec" not in buf_fm_out.getvalue():
            errs.append(
                f"bench/control_vectors.py from-md failed: rc={rc_fm}, out={buf_fm_out.getvalue()!r}, err={buf_fm_err.getvalue()!r}"
            )
        cli_dir_out = tmp / "md_out_dir"
        buf_im_out, buf_im_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_im_out), contextlib.redirect_stderr(buf_im_err):
            rc_im = bench_cv.main(
                ["import-md", str(cli_md), "--out", str(cli_dir_out), "--name", "custom_stem"]
            )
        if rc_im != 0 or not (cli_dir_out / "custom_stem.toml").is_file():
            errs.append(
                f"bench/control_vectors.py import-md --out <dir> --name failed: rc={rc_im}, err={buf_im_err.getvalue()!r}"
            )
        bad_cli_md = tmp / "bad_cli.md"
        bad_cli_md.write_text("#### Pair 1\n##### Prompt A\nOnly side A.\n", encoding="utf-8")
        buf_bfm_out, buf_bfm_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_bfm_out), contextlib.redirect_stderr(buf_bfm_err):
            rc_bfm = bench_cv.main(["from-md", str(bad_cli_md), "--out", str(tmp / "bad_cli.toml")])
        if rc_bfm != 2 or "missing '##### prompt b'" not in buf_bfm_err.getvalue().lower():
            errs.append(
                f"bench/control_vectors.py from-md on malformed markdown returned {rc_bfm}, err={buf_bfm_err.getvalue()!r}"
            )

        # Step schedule window check on _make_control_vector_wrapper ([0.25, 0.75) on a 4-step schedule)
        sched_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0], dtype=torch.float32)
        dm_sched = _MockDiffusionModel(with_tensors=False)
        patcher_sched = _MockPatcher(dm_sched, None)
        spec_sched = cv_nodes._AppliedVectorSpec(
            vector_name="sched_test",
            vector=torch.ones((cv.POST_REFINER_DIM,), dtype=torch.float32),
            strength=1.0,
            mask="non_vision",
            mode="add",
            start_percent=0.25,
            end_percent=0.75,
            metadata={"measured_under": {"prefix": "none"}},
            advisories=(),
        )
        patcher_sched.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})[cv_nodes.WRAPPER_KEY] = [
            cv_nodes._make_control_vector_wrapper((spec_sched,))
        ]
        ctx_sched = torch.zeros((1, 4, cv.POST_REFINER_DIM), dtype=torch.float32)
        tags_sched = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
        for step_i, expected_active in ((0, False), (1, True), (2, True), (3, False)):
            patcher_sched.invoke_diffusion_model(
                torch.zeros((1, 4)),
                sched_sigmas[step_i : step_i + 1],
                ctx_sched,
                minimax_payload={"text_token_tags": tags_sched},
                transformer_options={"sample_sigmas": sched_sigmas},
            )
            got_shift = float(dm_sched.last_context[0, 1, 0].item())
            if expected_active and abs(got_shift - 1.0) > 1e-6:
                errs.append(f"step {step_i} inside [0.25, 0.75) was not steered: got {got_shift}")
            if not expected_active and got_shift != 0.0:
                errs.append(f"step {step_i} outside [0.25, 0.75) was steered: got {got_shift}")

        # The reference-gate default is core's own value, so the node leaves that road alone at its
        # defaults. It is written as a number in `h3_rules`; this is what notices core moving it.
        import comfy.ldm.minimax.model as _core_model
        import h3_rules as _h3_rules

        _aug_default = _h3_rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS["ref_vae_aug"]
        if _aug_default != _core_model.VISUAL_COND_TIMESTEP:
            errs.append(
                f"h3_rules ref_vae_aug default {_aug_default} is not core's VISUAL_COND_TIMESTEP "
                f"{_core_model.VISUAL_COND_TIMESTEP}"
            )

        # Wave 2: video_scale / audio_scale / split_end_percent + ref_vae_end_percent / ref_vae_aug checks
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_init, v1_path, strength=1.0, video_scale=float("nan")
            ),
            "video_scale must be finite",
            "attach_control_vector refusal on non-finite video_scale",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_init, v1_path, strength=1.0, audio_scale=float("inf")
            ),
            "audio_scale must be finite",
            "attach_control_vector refusal on non-finite audio_scale",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_init, v1_path, strength=1.0, split_end_percent=1.5
            ),
            "split_end_percent must be <= 1.0",
            "attach_control_vector refusal on split_end_percent > 1.0",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_init, v1_path, strength=1.0, ref_vae_end_percent=-0.1
            ),
            "ref_vae_end_percent must be >= 0.0",
            "attach_control_vector refusal on ref_vae_end_percent < 0.0",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.attach_control_vector(
                patcher_with_init, v1_path, strength=1.0, ref_vae_aug=1.2
            ),
            "ref_vae_aug must be <= 1.0",
            "attach_control_vector refusal on ref_vae_aug > 1.0",
        )

        class _MockAVDiffusionModel(mm_mod.MiniMaxH3Model):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.condition_proj = SimpleNamespace(
                    weight=sd_tensors["condition_proj.weight"],
                    bias=sd_tensors["condition_proj.bias"],
                )
                self.token_refiner = SimpleNamespace(
                    final_norm=SimpleNamespace(
                        weight=sd_tensors["token_refiner.final_norm.weight"]
                    )
                )
                self.contexts_seen: list[torch.Tensor] = []
                self.payloads_seen: list[dict[str, Any]] = []

            def run_step(self, x, timestep, context, transformer_options={}, **kwargs):
                self.contexts_seen.append(context.clone())
                self.payloads_seen.append(dict(kwargs.get("minimax_payload") or {}))
                c_val = context[:, 1:, 0].to(dtype=torch.float32).mean()
                return [x[0] + c_val, x[1] + c_val * 2.0]

        dm_av = _MockAVDiffusionModel()
        patcher_av = _MockPatcher(dm_av, ckpt_path)
        p_split = cv_nodes.attach_control_vector(
            patcher_av,
            v1_path,
            strength=1.0,
            mask="non_vision",
            video_scale=1.0,
            audio_scale=0.0,
            split_end_percent=0.5,
            ref_vae_end_percent=0.5,
            ref_vae_aug=0.875,
        )
        av_x = [torch.zeros((1, 2), dtype=torch.float32), torch.zeros((1, 2), dtype=torch.float32)]
        av_ctx = torch.zeros((1, 4, cv.POST_REFINER_DIM), dtype=torch.float32)
        av_tags = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
        orig_cond_lat = torch.ones((1, 16, 2, 4, 4), dtype=torch.float32)
        sentinel_refs = [{"type": "image"}]
        sentinel_layout = {"seq_len": 128}
        caller_payload = {
            "text_token_tags": av_tags,
            "cond_video_latents": [orig_cond_lat],
            "visual_cond_noise_aug": 0.999,
            "refs": sentinel_refs,
            "layout": sentinel_layout,
        }
        # Step 0 (pct = 0.0 < 0.5): inside split_end_percent and inside ref_vae_end_percent -> 2 calls, ref latents intact
        dm_av.contexts_seen.clear()
        dm_av.payloads_seen.clear()
        out_step0 = p_split.invoke_diffusion_model(
            av_x,
            sched_sigmas[0:1],
            av_ctx,
            minimax_payload=caller_payload,
            transformer_options={"sample_sigmas": sched_sigmas},
        )
        if len(dm_av.contexts_seen) != 2:
            errs.append(f"split video_scale=1.0, audio_scale=0.0 at step 0 expected 2 DiT calls, got {len(dm_av.contexts_seen)}")
        if abs(float(out_step0[0].mean().item()) - 0.75) > 1e-5 or abs(float(out_step0[1].mean().item()) - 0.0) > 1e-5:
            errs.append(f"split video_scale=1.0, audio_scale=0.0 at step 0 wrong output streams: {out_step0}")
        if (
            float(dm_av.payloads_seen[0]["visual_cond_noise_aug"]) != 0.875
            or float(dm_av.payloads_seen[0]["cond_video_latents"][0].abs().sum().item()) == 0.0
        ):
            errs.append(f"ref_vae_end_percent at step 0 (inside window) did not preserve cond_video_latents or set ref_vae_aug: {dm_av.payloads_seen[0]}")

        # Step 2 (pct = 0.5 >= 0.5): outside split_end_percent and outside ref_vae_end_percent -> 1 call, cond_video_latents zeroed, refs/layout untouched
        dm_av.contexts_seen.clear()
        dm_av.payloads_seen.clear()
        out_step2 = p_split.invoke_diffusion_model(
            av_x,
            sched_sigmas[2:3],
            av_ctx,
            minimax_payload=caller_payload,
            transformer_options={"sample_sigmas": sched_sigmas},
        )
        if len(dm_av.contexts_seen) != 1:
            errs.append(f"outside split_end_percent at step 2 expected 1 DiT call, got {len(dm_av.contexts_seen)}")
        if (
            float(dm_av.payloads_seen[0]["visual_cond_noise_aug"]) != 0.0
            or float(dm_av.payloads_seen[0]["cond_video_latents"][0].abs().sum().item()) != 0.0
            or dm_av.payloads_seen[0]["refs"] is not sentinel_refs
            or dm_av.payloads_seen[0]["layout"] is not sentinel_layout
        ):
            errs.append(f"ref_vae_end_percent at step 2 (outside window) did not zero cond_video_latents while preserving refs/layout: {dm_av.payloads_seen[0]}")
        if float(caller_payload["cond_video_latents"][0].abs().sum().item()) == 0.0 or caller_payload["visual_cond_noise_aug"] != 0.999:
            errs.append("ref_vae_end_percent mutated caller's minimax_payload dict in place")
        if cv_nodes._rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS["ref_vae_end_percent"] != 1.0:
            errs.append(
                f"CONTROL_VECTOR_OPTIONAL_DEFAULTS['ref_vae_end_percent'] must be 1.0, got {cv_nodes._rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS['ref_vae_end_percent']}"
            )
        p_disabled = cv_nodes.attach_control_vector(
            patcher_av,
            v1_path,
            strength=0.0,
            ref_vae_end_percent=0.5,
            ref_vae_aug=0.5,
        )
        dm_av.contexts_seen.clear()
        dm_av.payloads_seen.clear()
        p_disabled.invoke_diffusion_model(
            av_x,
            sched_sigmas[2:3],
            av_ctx,
            minimax_payload=caller_payload,
            transformer_options={"sample_sigmas": sched_sigmas},
        )
        if (
            float(dm_av.payloads_seen[0]["visual_cond_noise_aug"]) != 0.999
            or float(dm_av.payloads_seen[0]["cond_video_latents"][0].abs().sum().item()) == 0.0
        ):
            errs.append(
                f"strength=0.0 named vector still gated cond_video_latents or modified visual_cond_noise_aug: {dm_av.payloads_seen[0]}"
            )

        # Wave 2: match_media_pair_geometry, make_control_vector_from_media_pairs, MiniMaxH3MakeMediaControlVector, and CLI make-media
        img_a_raw = torch.rand((2, 90, 160, 3), generator=gen, dtype=torch.float32) * 0.4 + 0.6
        img_b_raw = torch.rand((2, 140, 100, 3), generator=gen, dtype=torch.float32) * 0.4
        matched_a, matched_b, (tw, th) = cv_nodes.match_media_pair_geometry(
            img_a_raw, img_b_raw, short_edge=224
        )
        if (
            matched_a.shape != matched_b.shape
            or matched_a.shape != (2, th, tw, 3)
            or tw % 32 != 0
            or th % 32 != 0
            or min(tw, th) != 224
        ):
            errs.append(
                f"match_media_pair_geometry failed: a={tuple(matched_a.shape)}, b={tuple(matched_b.shape)}, target=({tw}, {th})"
            )
        errs += _expect_refusal(
            lambda: cv_nodes.match_media_pair_geometry(
                img_a_raw[:1], img_b_raw, short_edge=224
            ),
            "same frame/batch count",
            "match_media_pair_geometry refusal on mismatched batch count",
        )
        errs += _expect_refusal(
            lambda: cv_nodes.match_media_pair_geometry(
                img_a_raw, img_b_raw, short_edge=16
            ),
            "short_edge must be >= 32",
            "match_media_pair_geometry refusal on short_edge < 32",
        )

        # End-to-end make_control_vector_from_media_pairs & MiniMaxH3MakeMediaControlVector.execute
        orig_cv_paths_m = cv_nodes.folder_paths.folder_names_and_paths.get(
            cv_nodes.CONTROL_VECTORS_FOLDER
        )
        try:
            cv_nodes.folder_paths.folder_names_and_paths[cv_nodes.CONTROL_VECTORS_FOLDER] = (
                [str(tmp)],
                {".safetensors"},
            )
            media_scene_txt = (REPO / "prompt_bank" / "p_h3_image" / "edit_alpine_terrace.txt").read_text(encoding="utf-8").strip()
            for m_pool in ("scene_text", "vision_tokens", "all_rows"):
                m_out = cv_nodes.MiniMaxH3MakeMediaControlVector.execute(
                    model=patcher_for_make,
                    clip=mock_clip,
                    image_a=img_a_raw[:1] if m_pool == "scene_text" else img_a_raw,
                    image_b=img_b_raw[:1] if m_pool == "scene_text" else img_b_raw,
                    output_name=f"unit_media_{m_pool}",
                    pooling=m_pool,
                    short_edge=224,
                    scene_prompt=media_scene_txt,
                    overwrite=True,
                )
                m_sf = tmp / m_out[0]
                if not m_sf.is_file() or not m_sf.with_suffix(".md").is_file():
                    errs.append(f"MiniMaxH3MakeMediaControlVector({m_pool}) did not write .safetensors and .md")
                else:
                    mt, mm = cv.load_vector(m_sf)
                    if "vector.post_refiner" not in mt or mm["measured_under"]["pooling"] != m_pool:
                        errs.append(f"MiniMaxH3MakeMediaControlVector({m_pool}) metadata/tensor mismatch: {mm}")
                    if (
                        mm["measured_under"].get("prefix") != {"media_contrast_grid": f"{tw}x{th}"}
                        or mm["pairs"]["pair_spans"][0].get("qwen_view_size") != [tw, th]
                    ):
                        errs.append(f"MiniMaxH3MakeMediaControlVector({m_pool}) missing media_contrast_grid provenance: {mm}")
            errs += _expect_refusal(
                lambda: cv_nodes.make_control_vector_from_media_pairs(
                    image_a=img_a_raw,
                    image_b=img_b_raw,
                    model=patcher_for_make,
                    clip=mock_clip,
                    output_name="bad_media_empty_scene",
                    pooling="vision_tokens",
                    scene_prompt="",
                    output_dir=tmp,
                ),
                "scene_prompt is required",
                "make_control_vector_from_media_pairs refusal on empty scene_prompt",
            )
            errs += _expect_refusal(
                lambda: cv_nodes.make_control_vector_from_media_pairs(
                    image_a=img_a_raw,
                    image_b=img_b_raw,
                    model=patcher_for_make,
                    clip=mock_clip,
                    output_name="bad_media_pool",
                    pooling="diff_span",
                    scene_prompt=media_scene_txt,
                    output_dir=tmp,
                ),
                "pooling must be one of",
                "make_control_vector_from_media_pairs refusal on invalid pooling",
            )
        finally:
            if orig_cv_paths_m is not None:
                cv_nodes.folder_paths.folder_names_and_paths[
                    cv_nodes.CONTROL_VECTORS_FOLDER
                ] = orig_cv_paths_m

    # 4. Generated workflow arm check (build_control_vector_arm, gate arms, probe arms, LayerNormMatch)
    arm = bw.build_control_vector_arm(
        "t2v", vector_name="film_16mm_1974__fl2va.safetensors", strength=0.75
    )
    arm_wave2 = bw.build_api(
        "t2v",
        control_vector=dict(
            vector_name="film_16mm_1974__fl2va.safetensors",
            strength=1.5,
            video_scale=1.0,
            audio_scale=0.0,
            split_end_percent=0.75,
            ref_vae_end_percent=0.5,
            ref_vae_aug=0.95,
        ),
    )
    if (
        arm_wave2["115"]["inputs"].get("video_scale") != 1.0
        or arm_wave2["115"]["inputs"].get("audio_scale") != 0.0
        or arm_wave2["115"]["inputs"].get("split_end_percent") != 0.75
        or arm_wave2["115"]["inputs"].get("ref_vae_end_percent") != 0.5
        or arm_wave2["115"]["inputs"].get("ref_vae_aug") != 0.95
    ):
        errs.append(f"build_api did not pass Wave 2 optional control_vector keys: {arm_wave2['115']['inputs']}")
    if "115" not in arm or arm["115"]["class_type"] != "MiniMaxH3ApplyControlVectors":
        errs.append("build_control_vector_arm did not emit node 115 MiniMaxH3ApplyControlVectors")
    if arm["9"]["inputs"]["model"] != ["115", 0]:
        errs.append(f"BasicGuider model input is {arm['9']['inputs']['model']}, expected ['115', 0]")
    if (
        arm["115"]["inputs"]["strength"] != 0.75
        or arm["115"]["inputs"]["mask"] != "scene_only"
        or arm["115"]["inputs"]["mode"] != "add_norm_matched"
        or arm["115"]["inputs"]["start_percent"] != 0.0
        or arm["115"]["inputs"]["end_percent"] != 1.0
    ):
        errs.append(f"unexpected node 115 inputs: {arm['115']['inputs']}")
    if "MiniMaxH3SageAttention" in {n["class_type"] for n in arm.values()}:
        errs.append("build_control_vector_arm enabled SageAttention in the base arm")

    dual_arm = bw.build_api(
        "t2v",
        control_vector=[
            ("film_16mm_1974__fl2va.safetensors", 1.8),
            ("chaos_panic__fl2va.safetensors", 2.2),
        ],
    )
    if (
        dual_arm.get("115", {}).get("class_type") != "MiniMaxH3ApplyControlVectors"
        or dual_arm.get("125", {}).get("class_type") != "MiniMaxH3ApplyControlVectors"
        or dual_arm["125"]["inputs"]["model"] != ["115", 0]
        or dual_arm["9"]["inputs"]["model"] != ["125", 0]
        or dual_arm["115"]["inputs"]["strength"] != 1.8
        or dual_arm["125"]["inputs"]["strength"] != 2.2
    ):
        errs.append("build_api with 2-element control_vector list did not chain nodes 115 -> 125 -> BasicGuider (9)")

    ph3_cv_arm = bw.build_ph3_image_api(
        "ph3_edit",
        control_vector=dict(
            vector_name="film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors",
            strength=1.15,
            ref_vae_end_percent=1.0,
            ref_vae_aug=0.999,
        ),
    )
    if (
        ph3_cv_arm.get("115", {}).get("class_type") != "MiniMaxH3ApplyControlVectors"
        or ph3_cv_arm["115"]["inputs"].get("model") != ["58", 0]
        or ph3_cv_arm["9"]["inputs"].get("model") != ["115", 0]
        or ph3_cv_arm["115"]["inputs"].get("strength") != 1.15
    ):
        errs.append(f"build_ph3_image_api(control_vector=...) did not wire node 115 between 58 and BasicGuider (9): {ph3_cv_arm.get('115')}")

    gate_arms = bw.build_control_vector_gate_arms(
        "t2v",
        vector_name="film_16mm_1974__fl2va.safetensors",
    )
    if set(gate_arms) != {"base", "edit", "s0p5", "s1", "s2", "sneg1"}:
        errs.append(f"build_control_vector_gate_arms keys mismatch: {sorted(gate_arms)}")
    if gate_arms["base"]["115"]["inputs"]["strength"] != 0.0 or gate_arms["s2"]["115"]["inputs"]["strength"] != 2.0:
        errs.append("build_control_vector_gate_arms strengths mismatch")

    probe_e42 = bw.build_control_vector_probe_arm(
        "t2v",
        late_prompt="integrated_multimodal_description: [Shot 1] Close shot.\n\noverall_soundscape: Wind.",
        switch_percent=0.5,
    )
    if (
        probe_e42.get("117", {}).get("class_type") != "ConditioningSetTimestepRange"
        or probe_e42.get("118", {}).get("class_type") != "ConditioningSetTimestepRange"
        or probe_e42.get("119", {}).get("class_type") != "ConditioningCombine"
        or probe_e42["9"]["inputs"]["conditioning"] != ["119", 0]
    ):
        errs.append("build_control_vector_probe_arm (E4.2) did not wire ConditioningSetTimestepRange + ConditioningCombine")

    probe_e43 = bw.build_control_vector_probe_arm(
        "t2v",
        stop_at_clip_layer=-6,
        norm_match=True,
    )
    if (
        probe_e43.get("120", {}).get("class_type") != "CLIPSetLastLayer"
        or probe_e43["120"]["inputs"]["stop_at_clip_layer"] != -6
        or probe_e43.get("122", {}).get("class_type") != "MiniMaxH3LayerNormMatch"
        or probe_e43["26"]["inputs"]["conditioning"] != ["122", 0]
    ):
        errs.append("build_control_vector_probe_arm (E4.3) did not wire CLIPSetLastLayer + MiniMaxH3LayerNormMatch")

    # MiniMaxH3LayerNormMatch unit check
    cond_early_t = torch.ones((1, 5, cv.ENCODER_DIM), dtype=torch.float32) * 2.0
    cond_ref_t = torch.ones((1, 5, cv.ENCODER_DIM), dtype=torch.float32) * 6.0
    cond_tags = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    cond_early = [[cond_early_t, {"minimax_token_tags": cond_tags}]]
    cond_ref = [[cond_ref_t, {"minimax_token_tags": cond_tags}]]
    matched_cond = cv_nodes.MiniMaxH3LayerNormMatch.execute(
        conditioning=cond_early,
        reference_conditioning=cond_ref,
    )[0]
    matched_t = matched_cond[0][0]
    if not torch.equal(matched_t[:, :2, :], cond_early_t[:, :2, :]):
        errs.append("MiniMaxH3LayerNormMatch modified tag==0 vision rows")
    if not torch.allclose(matched_t[:, 2:, :], cond_ref_t[:, 2:, :], atol=1e-5):
        errs.append("MiniMaxH3LayerNormMatch did not scale tag==1 rows to reference median norm")

    # 5. j_lens unit and synthetic-bundle round-trip checks
    v_test = torch.zeros(cv.POST_REFINER_DIM, dtype=torch.float32)
    v_test[0] = 3.0
    v_test[1] = 4.0
    h_test = torch.zeros(cv.POST_REFINER_DIM, dtype=torch.float32)
    h_test[0] = 1.0
    v_h0 = j_lens.apply_soft_holds(v_test, [h_test], alpha=0.0)
    v_h05 = j_lens.apply_soft_holds(v_test, [h_test], alpha=0.5)
    v_h1 = j_lens.apply_soft_holds(v_test, [h_test], alpha=1.0)
    v_h15 = j_lens.apply_soft_holds(v_test, [h_test], alpha=1.5)
    if not torch.equal(v_h0, v_test):
        errs.append("j_lens.apply_soft_holds(alpha=0.0) altered vector")
    if abs(float(v_h05[0].item()) - 1.5) > 1e-6 or abs(float(v_h1[0].item())) > 1e-6 or abs(float(v_h15[0].item()) - (-1.5)) > 1e-6:
        errs.append(f"j_lens.apply_soft_holds incorrect projection across alpha: {v_h05[0]}, {v_h1[0]}, {v_h15[0]}")
    errs += _expect_refusal(
        lambda: j_lens.apply_soft_holds(v_test, [h_test], alpha=2.5),
        "alpha must be in [0.0, 2.0]",
        "j_lens.apply_soft_holds refusal on alpha > 2.0",
    )

    # Synthetic mini bundle check (fast, deterministic, zero disk dependency)
    synth_words = ["underwater", "submerged", "aquatic", "natural", "normal", "standard", "realistic", "consistent", "ordinary", "typical", "steady", "shaky"]
    g_jl = torch.Generator().manual_seed(1042)
    m_u = torch.randn((len(synth_words), cv.ENCODER_DIM), generator=g_jl, dtype=torch.float32)
    m_c = torch.nn.functional.normalize(m_u - m_u.mean(dim=0, keepdim=True), dim=-1)
    j_ref_mini = torch.randn((cv.POST_REFINER_DIM, cv.ENCODER_DIM), generator=g_jl, dtype=torch.float32) * 0.02
    j_w_mini = j_ref_mini.T
    mini_bundle = {
        "path": "synthetic",
        "words": synth_words,
        "word_to_idxs": {w: [i] for i, w in enumerate(synth_words)},
        "M_union": m_u,
        "M_centered": m_c,
        "Sigma_49_p030": torch.eye(cv.ENCODER_DIM, dtype=torch.float32),
        "cloze_base_union": torch.zeros((7, len(synth_words)), dtype=torch.float32),
        "valid_pos_in_union": torch.arange(len(synth_words), dtype=torch.long),
        "valid_idx": torch.arange(len(synth_words), dtype=torch.long),
        "union_idx": torch.arange(len(synth_words), dtype=torch.long),
        "J_49_to_63": torch.eye(cv.ENCODER_DIM, dtype=torch.float32),
        "J_ref_ref2va": j_ref_mini,
        "J_ref_fl2va": j_ref_mini,
        "J_wiener_ref2va": j_w_mini,
        "J_wiener_fl2va": j_w_mini,
    }
    synth_res = j_lens.synthesize_from_words(
        "underwater, submerged, aquatic",
        family="fl2va",
        hold_words="shaky",
        alpha=1.0,
        bundle=mini_bundle,
    )
    if tuple(synth_res["vector.post_refiner"].shape) != (cv.POST_REFINER_DIM,) or tuple(synth_res["vector.encoder"].shape) != (cv.ENCODER_DIM,):
        errs.append("j_lens.synthesize_from_words returned wrong tensor shapes")
    with tempfile.TemporaryDirectory() as jl_tmp:
        jl_out = j_lens.save_synthesized_vector(Path(jl_tmp) / "underwater_test.safetensors", synth_res)
        loaded_t, loaded_m = cv.load_vector(jl_out)
        if "vector.post_refiner" not in loaded_t or "vector.encoder" not in loaded_t or loaded_m["name"] != "underwater_test":
            errs.append("j_lens.save_synthesized_vector output failed cv.load_vector round-trip")
    errs += _expect_refusal(
        lambda: j_lens.synthesize_from_words("nonexistent_word_xyz", bundle=mini_bundle),
        "matched the 24,912-word j-lens vocabulary",
        "j_lens.synthesize_from_words refusal on unmatched word",
    )
    errs += _expect_refusal(
        lambda: j_lens.synthesize_from_words("underwater", hold_words="shaky", alpha=-0.5, bundle=mini_bundle),
        "alpha must be in [0.0, 2.0]",
        "j_lens.synthesize_from_words refusal on negative alpha",
    )
    errs += _expect_refusal(
        lambda: j_lens.synthesize_from_words("underwater", hold_words="underwater", alpha=1.0, bundle=mini_bundle),
        "completely cancelled pos_words",
        "j_lens.synthesize_from_words refusal when hold_words cancels pos_words",
    )

    # Compound, acronym, hyphenated, and stem-stripped word resolution check
    comp_w2i = {w: [i] for i, w in enumerate(["pixel", "arcade", "console", "cartridge", "neon", "glowing", "matrix", "laser", "steam", "punk", "clock", "work", "brass", "gear", "gears", "copper", "mechanical", "antique", "pitched", "whistle", "piercing", "rubbing", "rubber"])}
    comp_ids, comp_matched, comp_unmatched = j_lens._resolve_word_indices(
        ["8-bit", "led", "steampunk", "clockwork", "squeaking", "pixelated"],
        comp_w2i,
    )
    if comp_unmatched or len(comp_matched) != 6 or not comp_ids:
        errs.append(f"j_lens._resolve_word_indices failed compound/acronym/stem check: matched={comp_matched}, unmatched={comp_unmatched}")

    # Verify assess_steering_readiness (objective metrics, no fuzzy 'tier', refiner_transmission None without v_enc)
    arrow_synth = j_lens.synthesize_from_words("shaky -> underwater, submerged", bundle=mini_bundle)
    if arrow_synth["pos_words_matched"] != ["underwater", "submerged"] or arrow_synth["neg_words_matched"] != ["shaky"]:
        errs.append(f"j_lens.synthesize_from_words '->' contrast split failed: {arrow_synth}")
    rd_pr_only = j_lens.assess_steering_readiness(synth_res["vector.post_refiner"], v_enc=None, bundle=mini_bundle)
    if (
        "tier" in rd_pr_only
        or rd_pr_only.get("refiner_transmission") is not None
        or rd_pr_only.get("basis") != "post_refiner"
        or "Contextual 50-layer" not in str(rd_pr_only.get("basis_note", ""))
    ):
        errs.append(f"assess_steering_readiness with v_enc=None should have no 'tier', basis='post_refiner', and refiner_transmission=None: {rd_pr_only}")
    rd_with_enc = synth_res["readiness"]
    if (
        "tier" in rd_with_enc
        or rd_with_enc.get("basis") != "unigram_synthesis"
        or "In-subspace M_union" not in str(rd_with_enc.get("basis_note", ""))
        or rd_with_enc.get("refiner_transmission") is None
        or not isinstance(rd_with_enc.get("top1_gain"), float)
    ):
        errs.append(f"assess_steering_readiness with v_enc failed: {rd_with_enc}")

    # Verify analyze_prompt_coupling token_heatmap and save_composed_recipe_vector
    pc_hm = j_lens.analyze_prompt_coupling(
        synth_res["vector.post_refiner"],
        "integrated_multimodal_description:\n[Shot 1] Live-action medium shot of a male customer under underwater coral.\noverall_soundscape:\nQuiet hum.",
        pos_words=["underwater", "coral"],
        bundle=mini_bundle,
    )
    hm_list = pc_hm.get("token_heatmap")
    if not isinstance(hm_list, list) or not hm_list or not any(x.get("role") in ("receptor", "subject") for x in hm_list):
        errs.append(f"j_lens.analyze_prompt_coupling token_heatmap failed: {pc_hm}")

    with tempfile.TemporaryDirectory() as comp_tmp:
        comp_res = j_lens.save_composed_recipe_vector(
            Path(comp_tmp) / "composed_test.safetensors",
            [
                {"vector": synth_res["vector.post_refiner"], "mode": "add_norm_matched", "strength": 1.15},
            ],
            family="fl2va",
            name="composed_test",
            hold_words="shaky",
            alpha=0.8,
            bundle=mini_bundle,
            write_sidecar_md=False,
        )
        comp_path = comp_res.get("path")
        if not isinstance(comp_path, Path) or not comp_path.is_file():
            errs.append(f"j_lens.save_composed_recipe_vector failed to write file: {comp_res}")
        else:
            c_t, c_m = cv.load_vector(comp_path)
            if "vector.post_refiner" not in c_t or "vector.encoder" not in c_t or c_m.get("name") != "composed_test":
                errs.append(f"j_lens.save_composed_recipe_vector round-trip failed: {c_m}")

    # Verify real J-Lens bundle _SHORT_WORD_FALLBACKS & AGAINST_PRESETS vocabulary coverage if bundle exists on disk
    if j_lens.bundle_available():
        real_b = j_lens.load_j_lens_bundle()
        real_w2i = real_b["word_to_idxs"]
        dead_fbs = [
            (k, fb)
            for k, vals in j_lens._SHORT_WORD_FALLBACKS.items()
            for fb in vals
            if fb not in real_w2i
        ]
        if dead_fbs:
            errs.append(f"j_lens._SHORT_WORD_FALLBACKS has words missing from 24,912-word bundle: {dead_fbs}")
        for p_name, p_words in j_lens.AGAINST_PRESETS.items():
            _, _, p_unm = j_lens._resolve_word_indices(p_words, real_w2i)
            if p_unm:
                errs.append(f"j_lens.AGAINST_PRESETS[{p_name!r}] has unmatched words: {p_unm}")
        rep_synth = j_lens.synthesize_from_words(
            "skeleton, translucent, bones, skull",
            neg_words="@replace_human",
            family="fl2va",
            bundle=real_b,
        )
        if not rep_synth["readiness"]["is_replacement_concept"] or rep_synth["readiness"]["refiner_transmission"] < 0.48:
            errs.append(f"@replace_human skeleton synthesis readiness unexpected: {rep_synth['readiness']}")

    # Verify resolve_vector_file_path bare-stem, cross-family sibling swap, and __lora_ preference
    with tempfile.TemporaryDirectory() as sib_tmp:
        sib_dir = Path(sib_tmp)
        base_fl = sib_dir / "test_sib__fl2va__diff_span.safetensors"
        lora_fl = sib_dir / "test_sib__fl2va__diff_span__lora_2a8c2b52.safetensors"
        base_rf = sib_dir / "test_sib__ref2va__diff_span.safetensors"
        lora_rf = sib_dir / "test_sib__ref2va__diff_span__lora_865ee189.safetensors"
        for p_f in (base_fl, lora_fl, base_rf, lora_rf):
            p_f.write_bytes(b"stub")
        # 1. Cross-family swap from __fl2va__lora -> __ref2va__lora (must prefer __lora_ over base .safetensors)
        swapped_rf = cv_nodes.resolve_vector_file_path(lora_fl, preferred_family="ref2va")
        if swapped_rf.name != lora_rf.name:
            errs.append(f"resolve_vector_file_path cross-family lora swap picked {swapped_rf.name}, expected {lora_rf.name}")
        # 2. Cross-family swap from base __fl2va -> base __ref2va
        swapped_base = cv_nodes.resolve_vector_file_path(base_fl, preferred_family="ref2va")
        if swapped_base.name != base_rf.name:
            errs.append(f"resolve_vector_file_path cross-family base swap picked {swapped_base.name}, expected {base_rf.name}")
        # 3. Bare stem lookup prefers __lora_ by default
        matched_stem = cv_nodes._match_sibling_or_stem_in_dir(sib_dir, "test_sib", "fl2va", "diff_span")
        if matched_stem is None or matched_stem.name != lora_fl.name:
            errs.append(f"_match_sibling_or_stem_in_dir bare stem picked {matched_stem}, expected {lora_fl.name}")

    # Verify inline words on MiniMaxH3ApplyControlVectors when bundle is available
    if j_lens.bundle_available():
        m_inline = cv_nodes.MiniMaxH3ApplyControlVectors.execute(
            model=patcher_with_init,
            vector_name="none",
            strength=1.15,
            mode="add_norm_matched",
            mask="non_vision",
            words="blizzard, freezing, snow",
            against_words="@neutral",
        )[0]
        inline_specs = m_inline.model_options.get("transformer_options", {}).get(cv_nodes.WRAPPER_KEY, ())
        if len(inline_specs) != 1 or not inline_specs[0].vector_name.startswith("words:"):
            errs.append(f"MiniMaxH3ApplyControlVectors inline words did not attach spec: {inline_specs}")
        errs += _expect_refusal(
            lambda: cv_nodes.MiniMaxH3ApplyControlVectors.execute(
                model=patcher_with_init,
                vector_name="none",
                words="nonexistent_word_xyz",
            ),
            "matched the 24,912-word j-lens vocabulary",
            "MiniMaxH3ApplyControlVectors refusal on unmatched inline words",
        )

        # Verify extract_prompt_subject_words and analyze_prompt_coupling (Token-Wise Prompt Receptor Scan)
        noodle_subj = j_lens.extract_prompt_subject_words(REPO / "prompt_bank" / "t2va_noodle_bar.txt")
        if noodle_subj != ["live-action", "male", "customer", "cook"]:
            errs.append(f"extract_prompt_subject_words(t2va_noodle_bar.txt) returned {noodle_subj}, expected ['live-action', 'male', 'customer', 'cook']")
        subway_subj = j_lens.extract_prompt_subject_words(REPO / "prompt_bank" / "t2va_subway_platform.txt")
        if "woman" not in subway_subj or "busker" not in subway_subj:
            errs.append(f"extract_prompt_subject_words(t2va_subway_platform.txt) missed (S2) speaker clause: {subway_subj}")
        balloon_synth = j_lens.synthesize_from_words(
            "inflatable, balloon, glossy, latex",
            neg_words="@prompt_subject",
            prompt=REPO / "prompt_bank" / "t2va_noodle_bar.txt",
            family="fl2va",
            bundle=real_b,
        )
        pc = balloon_synth.get("prompt_coupling") or {}
        top_rec_words = [r["word"] for r in pc.get("top_receptors", ())[:2]]
        if top_rec_words != ["plastic", "bulb"]:
            errs.append(f"analyze_prompt_coupling top receptors for balloon on noodle_bar got {top_rec_words}, expected ['plastic', 'bulb']")
        bliz_synth = j_lens.synthesize_from_words(
            "blizzard, snowdrift, freezing, snow",
            family="fl2va",
            bundle=real_b,
        )
        bliz_rd = (bliz_synth.get("readout") or {}).get("readiness") or {}
        if bliz_rd.get("off_target_words") != [] or float(bliz_rd.get("echo_ratio") or 0.0) < 0.99:
            errs.append(f"semantic echo_ratio failed on blizzard synonyms: {bliz_rd}")
        op_ids, _op_m, _op_u = j_lens._resolve_word_indices(["operatic"], real_b["word_to_idxs"])
        if op_ids != real_b["word_to_idxs"]["opera"]:
            errs.append(f"_resolve_word_indices(['operatic']) did not resolve to 'opera' indices: {op_ids}")
        pr_ids, _pr_m, _pr_u = j_lens._resolve_word_indices(["pressurized"], real_b["word_to_idxs"])
        if pr_ids != real_b["word_to_idxs"]["pressure"]:
            errs.append(f"_resolve_word_indices(['pressurized']) did not resolve to 'pressure' indices: {pr_ids}")

        # Verify universal hold_words on a pre-built .safetensors vector (with words="")
        with tempfile.TemporaryDirectory() as uhold_tmp:
            uhold_vec_path = Path(uhold_tmp) / "uhold_test.safetensors"
            uhold_meta = {**meta1, "name": "uhold_test"}
            cv.save_vector(uhold_vec_path, {"vector.post_refiner": balloon_synth["vector.post_refiner"]}, uhold_meta)
            m_no_hold = cv_nodes.MiniMaxH3ApplyControlVectors.execute(
                model=patcher_with_init,
                vector_name=str(uhold_vec_path),
                strength=1.0,
                words="",
                hold_words="",
                allow_recipe_mismatch=True,
            )[0]
            m_with_hold = cv_nodes.MiniMaxH3ApplyControlVectors.execute(
                model=patcher_with_init,
                vector_name=str(uhold_vec_path),
                strength=1.0,
                words="",
                hold_words="shaky, blur",
                hold_alpha=1.0,
                allow_recipe_mismatch=True,
            )[0]
            v_nh = m_no_hold.model_options["transformer_options"][cv_nodes.WRAPPER_KEY][0].vector
            v_wh = m_with_hold.model_options["transformer_options"][cv_nodes.WRAPPER_KEY][0].vector
            if torch.allclose(v_nh, v_wh, atol=1e-4):
                errs.append("Universal hold_words on pre-built .safetensors vector (words='') did not modify vector")
            if float(torch.linalg.vector_norm(v_wh)) >= float(torch.linalg.vector_norm(v_nh)):
                errs.append("Universal hold_words on pre-built .safetensors vector did not reduce L2 norm after orthogonal scrub (Invariant 2)")

        # Verify Step-0 _h3_tail_text resolution when against_words='@prompt_subject' is attached without prompt at node time
        m_deferred = cv_nodes.MiniMaxH3ApplyControlVectors.execute(
            model=patcher_with_init,
            vector_name="none",
            strength=1.25,
            mode="add_norm_matched",
            mask="non_vision",
            words="inflatable, balloon, glossy, latex",
            against_words="@prompt_subject",
        )[0]
        def_spec = m_deferred.model_options["transformer_options"][cv_nodes.WRAPPER_KEY][0]
        if not def_spec.metadata.get("deferred_against_prompt"):
            errs.append("MiniMaxH3ApplyControlVectors with against_words='@prompt_subject' did not set deferred_against_prompt=True")
        vec_before_step0 = def_spec.vector.clone()
        noodle_txt = (REPO / "prompt_bank" / "t2va_noodle_bar.txt").read_text(encoding="utf-8")
        fake_ctx = torch.zeros((1, 8, cv.POST_REFINER_DIM), dtype=torch.float32)
        fake_tags = torch.ones((1, 8), dtype=torch.long)
        scene_m = torch.ones((8,), dtype=torch.bool)
        setattr(scene_m, "_h3_tail_text", noodle_txt)
        setattr(fake_tags, "_h3_span_masks", {"scene": scene_m})
        m_deferred.invoke_diffusion_model(
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([1.0]),
            fake_ctx,
            minimax_payload={"text_token_tags": fake_tags},
        )
        if dm.last_context is None:
            errs.append("Step-0 _h3_tail_text wrapper test did not populate dm.last_context")
        else:
            applied_shift = dm.last_context[0, 0] / 1.25
            if torch.allclose(applied_shift, vec_before_step0, atol=1e-3):
                errs.append("Step-0 _h3_tail_text did not re-synthesize @prompt_subject vector from scene prompt")
            if not torch.allclose(applied_shift, balloon_synth["vector.post_refiner"], atol=1e-4):
                errs.append("Step-0 _h3_tail_text @prompt_subject vector did not match direct synthesize_from_words(prompt=t2va_noodle_bar.txt)")

        # Verify `rebuild` CLI command in both single-vector mode and folder mode (and refusal on unknown stem)
        reb_dir = sib_dir / "rebuild_test_dir"
        reb_dir.mkdir(parents=True, exist_ok=True)
        buf_reb_out, buf_reb_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_reb_out), contextlib.redirect_stderr(buf_reb_err):
            rc_w = bench_cv.main(
                [
                    "words",
                    "blizzard, snowdrift, frost",
                    "--name",
                    "reb_word_sample",
                    "--families",
                    "fl2va",
                    "--out-dir",
                    str(reb_dir),
                ]
            )
            if rc_w != 0:
                errs.append(f"CLI words setup for rebuild test failed with exit code {rc_w}")
            else:
                reb_st_file = next(reb_dir.glob("reb_word_sample__fl2va__*.safetensors"))
                t_orig, _ = cv.load_vector(reb_st_file)
                rc_reb_single = bench_cv.main(
                    [
                        "rebuild",
                        "reb_word_sample",
                        "--dir",
                        str(reb_dir),
                        "--toml-dir",
                        str(reb_dir),
                        "--families",
                        "fl2va",
                    ]
                )
                if rc_reb_single != 0:
                    errs.append(f"CLI rebuild single vector failed with exit code {rc_reb_single}")
                t_reb1, _ = cv.load_vector(reb_st_file)
                if not torch.allclose(t_orig["vector.post_refiner"], t_reb1["vector.post_refiner"], atol=1e-5):
                    errs.append("CLI rebuild single vector altered J-Lens synthesized vector.post_refiner")
                rc_reb_folder = bench_cv.main(
                    [
                        "rebuild",
                        str(reb_dir),
                        "--toml-dir",
                        str(reb_dir),
                    ]
                )
                if rc_reb_folder != 0:
                    errs.append(f"CLI rebuild folder mode failed with exit code {rc_reb_folder}")
                rc_reb_missing = bench_cv.main(
                    [
                        "rebuild",
                        "nonexistent_vector_stem_xyz",
                        "--dir",
                        str(reb_dir),
                        "--toml-dir",
                        str(reb_dir),
                    ]
                )
                if rc_reb_missing != 2:
                    errs.append(f"CLI rebuild on nonexistent stem expected exit code 2, got {rc_reb_missing}")

    return errs



def main() -> int:
    all_errs: list[str] = []

    try:
        c1_errs, tok = check_tokenizer_constants()
    except Exception as exc:
        c1_errs, tok = [f"unexpected {type(exc).__name__}: {exc}"], None
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
        (
            "C7",
            "runtime DIFFUSION_MODEL wrapper, stacked float64 accumulation, state_dict fallback/refusal, arm wiring, and CLI inspect",
            check_runtime_nodes_arm_and_cli,
        ),
    ):
        try:
            errs = fn()
        except (Exception, SystemExit) as exc:
            errs = [f"unexpected {type(exc).__name__}: {exc}"]
        if errs:
            for e in errs:
                print(f"  FAIL  [{code}] {e}")
            all_errs.extend(errs)
        else:
            print(f"  ok    [{code}] {name}")

    if all_errs:
        print(f"\nFAIL: {len(all_errs)} control-vector check error(s)")
        return 1
    print("\nok: all 7 control-vector check groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
