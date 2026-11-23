#!/usr/bin/env python3
"""Holistic zero-lore, exemplar-diversity, vendor-vocabulary, prefix, & HeylookProvenance check suite.

Verifies:
1. Zero internal codebase/model lore or unexplained internal mode slugs across all 21
   `PRESET_CATALOG` runtime prompts, `fresh_prompt_system_prompt.md`, AND rendered Python
   user-turn / repair-turn envelopes (`auto_edit_prompt`, `h3_prompt_expander`,
   `h3_image_nodes`, `director/server.py`, `bench/control_vector/llm_pairs.py`).
2. Zero crutch examples (`noir`, `film_16mm`, `covered_market`, `woman_blonde`) and
   positive-only optical restraint (`f/4` to `f/5.6`) across runtime system prompts.
3. Closed cinematography camera motions (`vendor_guides/base_en.md` §4.3) and closed
   reference task/retention vocabularies (`vendor_guides/ref_en.md` §3–§4).
4. `resolve_heylook_preset_prefix()` precedence (`explicit > H3_HEYLOOK_PRESET_PREFIX > ""`)
   and `try_inherit_heylook_preset()` exact matching.
5. Canonical 14-key `HeylookProvenance` (`build_heylook_provenance`), SSE `message_start`
   model/usage capture, caller attachment across all 5 callers, `[provenance]` TOML
   serialization, and `llm_pairs.py` single-clause / soundscape / music repair fidelity.
"""

from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import tomllib
from typing import Any

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap, case, finish  # noqa: E402

bootstrap(cpu=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import auto_edit_prompt as aep  # noqa: E402
from bench.control_vector import llm_pairs  # noqa: E402
import bench.sync_heylook_presets as shp  # noqa: E402
import h3_image_nodes as hin  # noqa: E402
import h3_prompt_contract as pc  # noqa: E402
import h3_prompt_expander as vexp  # noqa: E402
import heylook_client as hlc  # noqa: E402


BANNED_LORE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "model_or_repo_jargon",
        re.compile(
            r"\b(?:H3|MiniMax|Pika|Pruna|ComfyUI|DiT|Qwen|Qwen3-VL|LoRA|EditGuider|BasicGuider|"
            r"safetensors|heylook|heylookitsanllm|sglang|vllm|TextConnector)\b|MiniMaxH3\w*|P-H3-Image",
            re.IGNORECASE,
        ),
    ),
    (
        "code_or_toml_filename",
        re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.(?:py|toml)\b", re.IGNORECASE),
    ),
    (
        "internal_mode_slug",
        re.compile(
            r"\b(?:t2va|i2va|fl2va|l2va|ref2va|h3_image|ph3_image|fast_1x|guided_2x)\b",
            re.IGNORECASE,
        ),
    ),
)

BANNED_CRUTCH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("noir_crutch", re.compile(r"\bnoir\b", re.IGNORECASE)),
    ("film_16mm_crutch", re.compile(r"film_16mm\w*", re.IGNORECASE)),
    ("covered_market_crutch", re.compile(r"covered[_ ]market\w*", re.IGNORECASE)),
    ("woman_blonde_crutch", re.compile(r"woman_blonde\w*", re.IGNORECASE)),
)

NEGATIVE_OPTICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"without unrequested commercial HDR", re.IGNORECASE),
    re.compile(r"avoiding unrequested neon", re.IGNORECASE),
)


def scan_banned_lore(text: str) -> list[str]:
    hits: list[str] = []
    for category, pat in BANNED_LORE_PATTERNS:
        found = sorted({m.group(0) for m in pat.finditer(text)})
        for tok in found:
            hits.append(f"{category}:{tok}")
    return hits


def scan_banned_crutches(text: str) -> list[str]:
    hits: list[str] = []
    for category, pat in BANNED_CRUTCH_PATTERNS:
        found = sorted({m.group(0) for m in pat.finditer(text)})
        for tok in found:
            hits.append(f"{category}:{tok}")
    return hits


def check_zero_lore_prompts_and_envelopes() -> str:
    # 0. Red-control self-test proving the scanner catches every banned family
    for sample, expected_cat in (
        ("Write a MiniMax H3 prompt for Qwen3-VL.", "model_or_repo_jargon"),
        ("Wire MiniMaxH3EditGuider into ComfyUI.", "model_or_repo_jargon"),
        ("Inspect auto_edit_prompt.py and camera.toml.", "code_or_toml_filename"),
        ("Mode: ref2va or fl2va with guided_2x.", "internal_mode_slug"),
    ):
        red_hits = scan_banned_lore(sample)
        assert any(h.startswith(expected_cat + ":") for h in red_hits), (sample, red_hits)

    # 1. Runtime system prompts (`PRESET_CATALOG` + `fresh_prompt_system_prompt.md`)
    specs = list(shp.PRESET_CATALOG)
    assert len(specs) == 21, f"expected 21 presets in PRESET_CATALOG, got {len(specs)}"
    rel_paths = {s.rel_path for s in specs}
    assert "docs/wiki/system_prompts/h3_masked_edit_and_steering_system_prompt.md" not in rel_paths, (
        "developer manual h3_masked_edit_and_steering_system_prompt.md must not sit in PRESET_CATALOG"
    )
    assert "docs/wiki/system_prompts/h3_masked_region_edit_system_prompt.md" in rel_paths

    prompt_surfaces: list[tuple[str, str]] = [
        (f"preset:{s.name}", shp.read_spec_text(s, repo=REPO)) for s in specs
    ]
    fresh_md = (REPO / "docs/wiki/system_prompts/h3_auto_edit/fresh_prompt_system_prompt.md").read_text(encoding="utf-8")
    prompt_surfaces.append(("fresh_prompt_system_prompt.md", fresh_md))

    # 2. Rendered Python user-turn and repair-turn envelopes across all 5 callers
    sample_brief = aep.ConceptBrief(
        stem="internal_stem_should_not_leak",
        source_kind="edit_concept",
        path=REPO / "README.md",
        file_stat=("README.md", 0, 0),
        title="Warm Amber Lantern Glow",
        summary="Warm practical lantern key with gentle brass reflections.",
        a_side="cool overcast daylight",
        b_side="warm practical amber lantern light",
        avoid="blown highlights",
        notes="category=look; keep natural skin grain",
        modified_sections=("detailed_description", "overall_soundscape"),
        exemplars=(
            ("detailed_description", "cool daylight across the oak bench", "warm amber lantern light across the oak bench"),
            ("overall_soundscape", "quiet room tone", "soft oil-wick crackle under quiet room tone"),
        ),
        axis_hints={"deforms_ref_silhouette": True, "preserve_step0_layout": False},
    )

    class _FakeImg:
        image = object()

    class _FakeSynVid:
        frames = object()
        soundtrack = object()
        allow_upscale = False

    class _FakeAud:
        audio = object()

    mm_manifest = pc.describe_references([_FakeImg(), _FakeSynVid(), _FakeAud()])
    envelope_surfaces: list[tuple[str, str]] = []
    for mode in ("ref2va", "t2va", "i2va", "fl2va", "l2va", "single_frame"):
        for native_fmt in (True, False):
            u_txt = aep.build_fresh_prompt_user_prompt(
                concept_brief=sample_brief,
                edit_intent="A baker sets warm loaves on a marble counter.",
                n_images=2 if mode == "fl2va" else (1 if mode != "t2va" else 0),
                resolved_mode=mode,
                manifest=mm_manifest if mode == "ref2va" else None,
                length=1 if mode == "single_frame" else 192,
                native_format=native_fmt,
            )
            envelope_surfaces.append((f"build_fresh_prompt_user_prompt[{mode},native={native_fmt}]", u_txt))

    orig_sample = (
        "integrated_multimodal_description:\n"
        "[Shot 1] Live-action, documentary, a potter turns a clay bowl on a kick wheel in cool daylight.\n\n"
        "overall_soundscape:\nWet clay hums softly on the wheel.\n\n"
        "non_diegetic_music:\nN/A"
    )
    for rw_mode in ("section_edits", "edit_lines"):
        envelope_surfaces.append(
            (
                f"build_auto_edit_user_prompt[{rw_mode}]",
                aep.build_auto_edit_user_prompt(
                    original=orig_sample,
                    allowed_sections=("integrated_multimodal_description", "overall_soundscape"),
                    allow_retention_relax=True,
                    rewrite_mode=rw_mode,
                    concept_brief=sample_brief,
                    edit_intent="Warm lantern light fills the workshop.",
                    has_images=True,
                ),
            )
        )

    noisy_err = (
        "H3 prompt contract violation introduced by edit: "
        "fl2va prose must use bare 'Picture 1' / 'Picture 2'; "
        "ref2va prompt missing required section(s); "
        "single_frame image prompt must be sectionless; "
        "t2va / i2va / l2va Part One mismatch in MiniMaxH3AutoEditPrompt"
    )
    for mode in ("ref2va", "fl2va", "single_frame"):
        for native_fmt in (True, False):
            rep_txt = aep.build_repair_user_prompt(
                noisy_err,
                is_fresh_mode=True,
                use_native=native_fmt,
                resolved_mode=mode,
                allowed_sections=pc.REF2VA_SECTIONS if mode == "ref2va" else pc.BASE_SECTIONS,
                orig_section_names=pc.BASE_SECTIONS,
            )
            envelope_surfaces.append((f"build_repair_user_prompt[{mode},native={native_fmt}]", rep_txt))

    # Image expander concept user block
    img_block = aep.format_concept_brief_block(sample_brief, mode="image")
    envelope_surfaces.append(("h3_image_nodes:concept_user_envelope", f"{img_block}\n\nUser image request:\nA quiet harbor."))

    # Control-vector llm_pairs system + user + repair envelopes
    envelope_surfaces.append(("llm_pairs:_extract_system_prompt_text", llm_pairs._extract_system_prompt_text()))
    envelope_surfaces.append(
        ("llm_pairs:build_control_vector_user_prompt", llm_pairs.build_control_vector_user_prompt("crisp winter frost on stone"))
    )
    envelope_surfaces.append(
        (
            "llm_pairs:build_control_vector_repair_prompt",
            llm_pairs.build_control_vector_repair_prompt(
                "crisp winter frost on stone",
                ["pair 2: a_span duplicates pair 1", "pair 4: b_span uses rate word"],
                "### Test Scene Prompt\nintegrated_multimodal_description: [Shot 1] Live-action...",
            ),
        )
    )
    # Verify _resolve_concept_prompt_for_generation does NOT leak internal stem slugs like film_16mm
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        fake_toml = td_p / "film_16mm.toml"
        fake_toml.write_text(
            '[concept]\ntitle = "Warm Celluloid Grain"\nintent = "Soft organic grain and warm halation."\n'
            'a_side = "clean digital"\nb_side = "warm organic grain"\n',
            encoding="utf-8",
        )
        ns = argparse.Namespace(concept="", concept_flag="", from_concept=str(fake_toml))
        c_txt, c_meta = llm_pairs._resolve_concept_prompt_for_generation(ns)
        assert c_meta["stem"] == "film_16mm"
        assert "film_16mm" not in c_txt, f"concept stem leaked into LLM prompt: {c_txt}"
        envelope_surfaces.append(("llm_pairs:_resolve_concept_prompt_for_generation", c_txt))

    # Director fallback & continuation prompts
    from director import server as dir_srv

    envelope_surfaces.append(("director:_RETICLE_SUGGEST_FALLBACK", dir_srv._RETICLE_SUGGEST_FALLBACK))
    envelope_surfaces.append(("director:_SCENE_SUGGEST_FALLBACK", dir_srv._SCENE_SUGGEST_FALLBACK))
    envelope_surfaces.append(("director:load_shot_continuation_system_prompt", dir_srv.load_shot_continuation_system_prompt()))

    failures: list[str] = []
    for label, txt in [*prompt_surfaces, *envelope_surfaces]:
        hits = scan_banned_lore(txt)
        if hits:
            failures.append(f"{label} -> {hits}")
    assert not failures, "Banned lore/jargon found:\n  " + "\n  ".join(failures)
    return f"{len(prompt_surfaces)} runtime prompts + {len(envelope_surfaces)} Python user/repair envelopes verified 0 banned lore tokens"


def check_exemplar_diversity_and_optical_restraint() -> str:
    surfaces: list[tuple[str, str]] = [
        (s.name, shp.read_spec_text(s, repo=REPO)) for s in shp.PRESET_CATALOG
    ]
    surfaces.append(
        (
            "fresh_prompt_system_prompt.md",
            (REPO / "docs/wiki/system_prompts/h3_auto_edit/fresh_prompt_system_prompt.md").read_text(encoding="utf-8"),
        )
    )

    crutch_failures: list[str] = []
    neg_optical_failures: list[str] = []
    for name, txt in surfaces:
        chits = scan_banned_crutches(txt)
        if chits:
            crutch_failures.append(f"{name}: {chits}")
        for npat in NEGATIVE_OPTICAL_PATTERNS:
            if npat.search(txt):
                neg_optical_failures.append(f"{name}: matched {npat.pattern!r}")
    assert not crutch_failures, f"Crutch examples found: {crutch_failures}"
    assert not neg_optical_failures, f"Negative-form optical restraint phrasing found: {neg_optical_failures}"

    # Verify optical restraint (`f/4` to `f/5.6`) is propagated into h3_auto_edit, h3_video, h3_image, and masked edit
    for must_have_fstop in (
        "docs/wiki/system_prompts/h3_auto_edit/fresh_prompt_system_prompt.md",
        "docs/wiki/system_prompts/h3_auto_edit/section_edits_system_prompt.md",
        "docs/wiki/system_prompts/h3_auto_edit/edit_lines_system_prompt.md",
        "docs/wiki/system_prompts/h3_masked_region_edit_system_prompt.md",
        "docs/wiki/system_prompts/h3_video/t2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_video/ref2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_image/t2i_system_prompt.md",
        "docs/wiki/system_prompts/h3_image/edit_system_prompt.md",
    ):
        raw = (REPO / must_have_fstop).read_text(encoding="utf-8")
        assert "f/4" in raw and "f/5.6" in raw, f"{must_have_fstop} missing f/4..f/5.6 optical restraint"

    # Verify diverse exemplars across video & image suites
    t2va_txt = (REPO / "docs/wiki/system_prompts/h3_video/t2va_system_prompt.md").read_text(encoding="utf-8")
    assert "<|lyrics_start|>" in t2va_txt and "<|caption_start|>" in t2va_txt and "(S3)" in t2va_txt
    fl2va_txt = (REPO / "docs/wiki/system_prompts/h3_video/fl2va_system_prompt.md").read_text(encoding="utf-8")
    assert "established by Picture 1" in fl2va_txt and "settle exactly into Picture 2" in fl2va_txt
    ref2va_txt = (REPO / "docs/wiki/system_prompts/h3_video/ref2va_system_prompt.md").read_text(encoding="utf-8")
    assert "<Video 1>" in ref2va_txt and "<Audio 1>" in ref2va_txt and "<Audio 2>" in ref2va_txt
    return "0 crutch examples, positive-only f/4-f/5.6 optical restraint, and diverse exemplars verified"


def check_closed_vendor_vocabularies() -> str:
    base_guide = (REPO / "vendor_guides/base_en.md").read_text(encoding="utf-8")
    ref_guide = (REPO / "vendor_guides/ref_en.md").read_text(encoding="utf-8")
    motion_types = re.findall(r"\| Motion type \| `([^`]+)`", base_guide)
    assert len(motion_types) == 12, f"expected 12 camera motion rows in base_en.md, got {len(motion_types)}"

    video_docs = (
        "docs/wiki/system_prompts/h3_video/t2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_video/i2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_video/fl2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_video/l2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_video/ref2va_system_prompt.md",
        "docs/wiki/system_prompts/h3_control_vector_standalone_system_prompt.md",
        "docs/wiki/system_prompts/h3_director/shot_continuation_system_prompt.md",
        "docs/wiki/system_prompts/h3_auto_edit/fresh_prompt_system_prompt.md",
    )
    for rel in video_docs:
        txt = (REPO / rel).read_text(encoding="utf-8")
        missing = [
            mtype
            for mtype in motion_types
            if any(
                not re.search(r"(?<!\w)" + re.escape(part.strip()) + r"(?!\w)", txt, re.IGNORECASE)
                for part in mtype.split("/")
            )
        ]
        assert not missing, f"{rel} missing vendor camera motion type(s): {missing}"

    for tok in (*pc.LEGAL_TASK_TAGS, *pc.VISUAL_RETENTION_MARKERS, *pc.AUDIO_RETENTION_MARKERS):
        assert tok in ref_guide, f"{tok!r} missing from vendor_guides/ref_en.md"
        for rel in (
            "docs/wiki/system_prompts/h3_video/ref2va_system_prompt.md",
            "docs/wiki/system_prompts/h3_auto_edit/fresh_prompt_system_prompt.md",
        ):
            txt = (REPO / rel).read_text(encoding="utf-8")
            assert tok in txt, f"{rel} missing vendor ref_en.md token {tok!r}"
    return f"12 camera motion types + {len(pc.LEGAL_TASK_TAGS)} task tags + 8 visual/audio retention markers verified"


def check_preset_prefix_resolution() -> str:
    prev_env = os.environ.get("H3_HEYLOOK_PRESET_PREFIX")
    try:
        os.environ.pop("H3_HEYLOOK_PRESET_PREFIX", None)
        assert hlc.resolve_heylook_preset_prefix("") == ""
        assert hlc.resolve_heylook_preset_prefix("custom_") == "custom_"

        # Explicit env var sets prefix unless overridden by explicit arg (and empty "" clears it)
        os.environ["H3_HEYLOOK_PRESET_PREFIX"] = "envpfx_"
        assert hlc.resolve_heylook_preset_prefix("") == "envpfx_"
        assert hlc.resolve_heylook_preset_prefix("custom_") == "custom_"
        os.environ["H3_HEYLOOK_PRESET_PREFIX"] = ""
        assert hlc.resolve_heylook_preset_prefix("") == ""

        # try_inherit_heylook_preset requires exact target match (no implicit unprefixed fallback)
        model_row = {"id": "gemma-4", "capabilities": ["chat", "vision"]}
        presets_both = [
            {"id": "legacy", "name": "h3_image_t2i", "system_prompt": "LEGACY WRONG", "params": {}},
            {"id": "app", "name": "h3app_image_t2i", "system_prompt": "SYNCED APP", "params": {"temperature": 0.5}},
            {"id": "canon", "name": "h3-video-t2va", "system_prompt": "CANONICAL T2VA", "params": {}},
        ]

        def _handler_both(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"presets": presets_both})

        with httpx.Client(transport=httpx.MockTransport(_handler_both)) as c:
            # With H3_HEYLOOK_PRESET_PREFIX="h3app_": matches h3app_image_t2i, never h3_image_t2i or unprefixed h3-video-t2va
            os.environ["H3_HEYLOOK_PRESET_PREFIX"] = "h3app_"
            fields, rname = hlc.try_inherit_heylook_preset(
                c, "http://127.0.0.1:18080", "h3-image-t2i", model_row
            )
            assert rname == "h3app_image_t2i" and fields and fields["system"] == "SYNCED APP"
            fields_no_fb, rname_no_fb = hlc.try_inherit_heylook_preset(
                c, "http://127.0.0.1:18080", "h3-video-t2va", model_row
            )
            assert fields_no_fb is None and rname_no_fb == ""

            # With env unset: matches canonical h3-video-t2va directly, never legacy h3_image_t2i
            os.environ.pop("H3_HEYLOOK_PRESET_PREFIX", None)
            fields_canon, rname_canon = hlc.try_inherit_heylook_preset(
                c, "http://127.0.0.1:18080", "h3-video-t2va", model_row
            )
            assert rname_canon == "h3-video-t2va" and fields_canon and fields_canon["system"] == "CANONICAL T2VA"
            fields_legacy, rname_legacy = hlc.try_inherit_heylook_preset(
                c, "http://127.0.0.1:18080", "h3-image-t2i", model_row
            )
            assert fields_legacy is None and rname_legacy == ""

            # Explicit prefix arg overrides env and requires prefixed match
            fields_exp, rname_exp = hlc.try_inherit_heylook_preset(
                c, "http://127.0.0.1:18080", "h3-video-t2va", model_row, prefix="h3app_"
            )
            assert fields_exp is None and rname_exp == ""
    finally:
        if prev_env is None:
            os.environ.pop("H3_HEYLOOK_PRESET_PREFIX", None)
        else:
            os.environ["H3_HEYLOOK_PRESET_PREFIX"] = prev_env
    return "resolve_heylook_preset_prefix() & try_inherit_heylook_preset() exact matching verified"


def _make_valid_pair_markdown() -> str:
    scenes = [
        ("inside a sunlit spruce carpentry shop", "an older cabinetmaker in a canvas apron planes a walnut board", "warm north-window daylight"),
        ("along a wet harbor boardwalk at dawn", "a deckhand in yellow oilskins coils a thick mooring rope", "cool silver morning mist"),
        ("inside a tiled neighborhood bakery", "a baker in a white cap dusts rye loaves on a marble counter", "soft early morning window light"),
        ("across a gravel botanical courtyard", "a gardener in denim overalls trims rosemary beside terracotta pots", "clear mid-morning sunlight"),
        ("inside a quiet university map room", "a cartographer in wire glasses traces a coastal chart with brass dividers", "even overhead library lamps"),
        ("on a windy coastal stone jetty", "a lighthouse keeper in a navy wool coat steadies brass binoculars", "crisp late-afternoon sun"),
        ("inside a brick pottery studio", "a ceramicist centers red stoneware clay on a spinning wheel", "slanted skylight sunbeams"),
        ("along a cobbled hillside tram stop", "two commuters in raincoats wait beside an enamel route sign", "bright overcast afternoon light"),
        ("inside a timber boathouse slipway", "a shipwright tightens an iron clamp along a cedar skiff rib", "reflected water light through open doors"),
        ("across an alpine meadow trail", "a hiker in a green canvas pack checks a brass compass on a boulder", "clean high-altitude daylight"),
    ]
    lines = [
        "### Test Scene Prompt",
        "",
        "integrated_multimodal_description: [Shot 1] Live-action, documentary, a watchmaker in a green apron sits at an oak workbench in motivated daylight. The camera holds a static shot. She produces no vocal sound.",
        "",
        "overall_soundscape: Quiet room tone with steady brass pendulum ticking.",
        "",
        "non_diegetic_music: N/A",
        "",
        "### Contrastive Pairs (1–10)",
        "",
    ]
    base_clauses = (
        "warm north-window daylight falling across shaved bench planks",
        "cool silver morning mist drifting past damp pilings",
        "soft early morning window light resting over wooden shelves",
        "clear mid-morning sunlight brightening courtyard planters",
        "even overhead library lamps washing open chart drawers",
        "crisp late-afternoon sun glinting off iron railings",
        "slanted skylight sunbeams touching brick kiln walls",
        "bright overcast afternoon light diffused around glass shelters",
        "reflected harbor water light shimmering beneath timber gunwales",
        "clean high-altitude daylight bathing rocky ridge grass",
    )
    for idx, (loc, subj, _base_light) in enumerate(scenes, start=1):
        steered_clause = (
            "amber tungsten lantern light casting copper glow across "
            + (" oak", " hemp", " tile", " clay", " brass", " slate", " glaze", " iron", " hull", " scree")[idx - 1].strip()
        )
        base_clause = base_clauses[idx - 1]
        a_desc = (
            f"integrated_multimodal_description: [Shot 1] Live-action, documentary, a medium shot {loc} frames {subj} under "
            f"{base_clause}. The camera holds a static shot. No vocal sound is made and the figure produces no vocal sound."
        )
        b_desc = (
            f"integrated_multimodal_description: [Shot 1] Live-action, documentary, a medium shot {loc} frames {subj} under "
            f"{steered_clause}. The camera holds a static shot. No vocal sound is made and the figure produces no vocal sound."
        )
        lines.extend(
            [
                f"#### Pair {idx}",
                "##### Prompt A (Baseline)",
                a_desc,
                "",
                "overall_soundscape: Quiet ambient room and gentle movement sounds.",
                "",
                "non_diegetic_music: N/A",
                "",
                "##### Prompt B (Steered)",
                b_desc,
                "",
                "overall_soundscape: Quiet ambient room and gentle movement sounds.",
                "",
                "non_diegetic_music: N/A",
                "",
            ]
        )
    return "\n".join(lines)


def check_provenance_and_llm_pairs_contracts() -> str:
    # 1. Verify exact 14-key tuple & TOML round-trip
    expected_keys = (
        "author_kind",
        "requested_model",
        "resolved_model",
        "preset",
        "system_source",
        "system_prompt_sha256",
        "thinking",
        "reasoning_effort",
        "temperature",
        "seed",
        "repair_used",
        "input_tokens",
        "output_tokens",
        "created_utc",
    )
    assert hlc.HEYLOOK_PROVENANCE_KEYS == expected_keys

    # 2. Verify llm_pairs._single_clause_section_pair never injects canned filler & _repair_pair_prompts keeps soundscape/music
    pure_ins_a = "integrated_multimodal_description: [Shot 1] Live-action, a baker kneads dough on a wooden table."
    pure_ins_b = "integrated_multimodal_description: [Shot 1] Live-action, a baker kneads dough on a wooden table while flour drifts."
    ra, rb = llm_pairs._single_clause_section_pair(pure_ins_a, pure_ins_b)
    assert "remain steady throughout" not in ra and "remain steady throughout" not in rb
    assert "remains steady throughout" not in ra and "remains steady throughout" not in rb

    coupled_a = (
        "integrated_multimodal_description: [Shot 1] Live-action, a potter shapes clay under cool morning skylight beside a shelf and wipes her hands.\n\n"
        "overall_soundscape: Quiet wheel hum in a dry room and soft footsteps.\n\n"
        "non_diegetic_music: N/A"
    )
    coupled_b = (
        "integrated_multimodal_description: [Shot 1] Live-action, a potter shapes clay under amber lantern glow beside a shelf and dries her hands.\n\n"
        "overall_soundscape: Brisk rain pattering on skylight glass and soft footsteps.\n\n"
        "non_diegetic_music: N/A"
    )
    ca_rep, cb_rep = llm_pairs._repair_pair_prompts(coupled_a, coupled_b)
    span_a, span_b = llm_pairs.cv.differing_span(ca_rep, cb_rep, frame="scene")
    assert "|" in span_a and "Brisk rain pattering" in span_b, (span_a, span_b)

    music_a = (
        "integrated_multimodal_description: [Shot 1] Live-action, a runner jogs along a canal towpath at dawn.\n\n"
        "overall_soundscape: Gravel footsteps and light wind.\n\n"
        "non_diegetic_music: Slow solo cello notes at a quiet dynamic."
    )
    music_b = (
        "integrated_multimodal_description: [Shot 1] Live-action, a runner jogs along a canal towpath at dawn.\n\n"
        "overall_soundscape: Gravel footsteps and light wind.\n\n"
        "non_diegetic_music: Brisk acoustic guitar strumming at a fast tempo."
    )
    ma_rep, mb_rep = llm_pairs._repair_pair_prompts(music_a, music_b)
    ms_a, ms_b = llm_pairs.cv.differing_span(ma_rep, mb_rep, frame="scene")
    assert "cello" in ms_a and "guitar" in ms_b

    # 3. Mock SSE + JSON heylook server testing all 5 callers end-to-end
    call_state = {"auto_edit_turn": 0, "posted": []}
    pair_md = _make_valid_pair_markdown()

    class _MockHeylookHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._json(
                    200,
                    {
                        "data": [
                            {
                                "id": "Swift-1.5-Qwen3.8-27B-MLX-8bit",
                                "capabilities": ["chat", "vision", "thinking"],
                                "sampler_defaults": {"temperature": 0.65},
                                "engine": {
                                    "thinking": {"switch": True, "depth": {"values": ["low", "medium", "high"]}},
                                    "decoding": {"request_fields": ["max_tokens", "temperature", "top_p", "seed"]},
                                },
                            }
                        ]
                    },
                )
                return
            if self.path == "/v1/system/metrics":
                self._json(200, {"models": {"Swift-1.5-Qwen3.8-27B-MLX-8bit": {"resident": True}}})
                return
            if self.path == "/v1/presets":
                self._json(
                    200,
                    {
                        "presets": [
                            {
                                "id": "p-cv",
                                "name": "h3app_control_vector",
                                "system_prompt": "@extend:append\nKeep baseline scenes crisp.",
                                "params": {"temperature": 0.55},
                            }
                        ]
                    },
                )
                return
            self._json(404, {"detail": "not found"})

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            call_state["posted"].append(body)
            sys_txt = str(body.get("system") or "")
            if not body.get("stream", True):
                # Non-streaming director context_suggest call
                self._json(
                    200,
                    {
                        "model": "gemma-4-27b-it-mlx",
                        "usage": {"input_tokens": 210, "output_tokens": 64},
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "scene_caption: Sunlit timber boatshop beside calm water\n"
                                    "subject_label: brass bevel gauge\n\n"
                                    "## suggestions\n"
                                    "- title: Tighten Wingnut | target_prompt: brass wingnut turns clockwise and clicks tight | preset: click\n"
                                ),
                            }
                        ],
                    },
                )
                return

            if "### Contrastive Pairs" in sys_txt:
                reply_text = f"```\n{pair_md}\n```"
                in_t, out_t = 420, 680
            elif "## section:" in sys_txt:
                call_state["auto_edit_turn"] += 1
                if call_state["auto_edit_turn"] == 1:
                    # Deliberate 1st-turn failure (missing ## axes) to exercise repair_used=True + usage sum
                    reply_text = "## section: overall_soundscape\nSteady rain patters against the workshop window."
                    in_t, out_t = 180, 35
                else:
                    reply_text = (
                        "## section: overall_soundscape\n"
                        "Steady rain patters against the workshop window.\n\n"
                        "## axes\n"
                        "visual_changed: false\n"
                        "audio_changed: true\n"
                        "preserve_step0_layout: true\n"
                        "deforms_ref_silhouette: false\n"
                        "temporal_curve: whole_clip\n"
                        "reason: Updates rain foley only."
                    )
                    in_t, out_t = 240, 65
            elif "aspect_ratio:" in sys_txt:
                reply_text = (
                    "A natural-light medium photograph shows a stone Harbor pier at dawn with coiled hemp rope on wet planks.\n\n"
                    "aspect_ratio: 3:2"
                )
                in_t, out_t = 150, 42
            else:
                reply_text = (
                    "integrated_multimodal_description:\n"
                    "[Shot 1] Live-action, documentary, a sunlit medium shot frames a carpenter planing a walnut board at an oak bench in crisp f/5.6 focus. "
                    "The camera pushes in with small amplitude at slow speed. He produces no vocal sound.\n\n"
                    "overall_soundscape:\nCrisp wood shavings curl off the iron plane blade in a quiet timber shop.\n\n"
                    "non_diegetic_music:\nN/A"
                )
                in_t, out_t = 312, 148

            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "model": "Swift-1.5-Qwen3.8-27B-MLX-8bit",
                            "usage": {"input_tokens": in_t},
                        },
                    },
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": reply_text}},
                ),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": out_t},
                    },
                ),
                ("message_stop", {"type": "message_stop", "performance": {"generation_duration_ms": 80.0}}),
            ]
            sse = "".join(f"event: {ev}\ndata: {json.dumps(pay)}\n\n" for ev, pay in events).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockHeylookHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    prev_url = os.environ.get("H3_HEYLOOK_URL")
    prev_model = os.environ.get("H3_HEYLOOK_MODEL")
    prev_pfx = os.environ.get("H3_HEYLOOK_PRESET_PREFIX")
    try:
        base_url = f"http://127.0.0.1:{srv.server_port}"
        os.environ["H3_HEYLOOK_URL"] = base_url
        os.environ.pop("H3_HEYLOOK_MODEL", None)
        os.environ["H3_HEYLOOK_PRESET_PREFIX"] = "h3app_"

        # Caller 1: h3_prompt_expander.expand_video_prompt
        vres = vexp.expand_video_prompt(
            prompt="Carpenter planing walnut board.",
            mode="t2va",
            length=141,
            base_url=base_url,
            model="",
            use_cache=False,
        )
        vp = vres["provenance"]
        assert isinstance(vp, dict) and tuple(vp.keys()) == hlc.HEYLOOK_PROVENANCE_KEYS
        assert vp["requested_model"] == "" and vp["resolved_model"] == "Swift-1.5-Qwen3.8-27B-MLX-8bit"
        assert vp["input_tokens"] == 312 and vp["output_tokens"] == 148 and vp["repair_used"] is False
        assert len(vp["system_prompt_sha256"]) == 64

        # Caller 2: auto_edit_prompt.run_auto_edit_rewrite (with 1-shot repair summing tokens)
        ares = aep.run_auto_edit_rewrite(
            original=(
                "integrated_multimodal_description:\n"
                "[Shot 1] Live-action, a carpenter planes a walnut board.\n\n"
                "overall_soundscape:\nQuiet dry workshop room tone.\n\n"
                "non_diegetic_music:\nN/A"
            ),
            edit_intent="Add steady rain pattering on the workshop window.",
            base_url=base_url,
            model="",
            use_cache=False,
        )
        ap = ares["provenance"]
        assert isinstance(ap, dict) and tuple(ap.keys()) == hlc.HEYLOOK_PROVENANCE_KEYS
        assert ap["repair_used"] is True
        assert ap["input_tokens"] == 180 + 240 and ap["output_tokens"] == 35 + 65
        # Verify the repair turn sent to heylook contained zero banned lore!
        repair_turn_body = call_state["posted"][-1]
        repair_user_txt = repair_turn_body["messages"][-1]["content"]
        assert not scan_banned_lore(repair_user_txt), repair_user_txt

        # Caller 3: h3_image_nodes.expand_image_prompt
        ires = hin.expand_image_prompt(
            prompt="Stone harbor pier at dawn.",
            base_url=base_url,
            mode="auto",
            model="",
        )
        ip = ires["provenance"]
        assert isinstance(ip, dict) and tuple(ip.keys()) == hlc.HEYLOOK_PROVENANCE_KEYS
        assert ip["resolved_model"] == "Swift-1.5-Qwen3.8-27B-MLX-8bit"
        assert ip["input_tokens"] == 150 and ip["output_tokens"] == 42

        # Caller 4: bench/control_vector/llm_pairs.py + [provenance] TOML table
        with tempfile.TemporaryDirectory() as td:
            out_toml = Path(td) / "amber_lantern.toml"
            ns_cv = argparse.Namespace(
                llm_provider="heylook",
                llm_model="",
                concept="amber tungsten lantern light",
                concept_flag="",
                from_concept="",
                no_4bit=False,
                max_new_tokens=2048,
                temperature=0.6,
                exl3_root=None,
                heylook_url=base_url,
                thinking="model default",
                reasoning_effort="",
                thinking_budget=None,
                repair=False,
                out_md=str(Path(td) / "amber_lantern.md"),
                out_toml=str(out_toml),
                save_toml=False,
                name="amber_lantern",
                a_side=None,
                b_side=None,
                quality=None,
                overwrite=True,
            )
            rc = llm_pairs.cmd_generate_pairs(ns_cv)
            assert rc == 0 and out_toml.is_file()
            parsed_toml = tomllib.loads(out_toml.read_text(encoding="utf-8"))
            tprov = parsed_toml.get("provenance")
            assert isinstance(tprov, dict), parsed_toml.keys()
            assert tprov["author_kind"] == "heylook"
            assert tprov["resolved_model"] == "Swift-1.5-Qwen3.8-27B-MLX-8bit"
            assert tprov["preset"] == "h3app_control_vector"
            assert tprov["system_source"] == "local+extend:h3app_control_vector"
            assert tprov["input_tokens"] == 420 and tprov["output_tokens"] == 680

        # Caller 5: director/server.py::handle_context_suggest (max_tokens=2048, no forced 0.6 temp, provenance)
        from aiohttp.test_utils import TestClient, TestServer
        from director import server as dir_srv

        async def _check_director_provenance() -> None:
            with tempfile.TemporaryDirectory() as d_tmp:
                in_dir = Path(d_tmp) / "input"
                in_dir.mkdir(parents=True, exist_ok=True)
                img_arr = np.zeros((96, 128, 3), dtype=np.uint8)
                img_arr[:, :] = (80, 110, 140)
                buf = io.BytesIO()
                Image.fromarray(img_arr, mode="RGB").save(buf, format="PNG")
                img_bytes = buf.getvalue()

                from aiohttp import web

                comfy_stub = web.Application()
                comfy_stub.router.add_get(
                    "/view",
                    lambda _r: web.Response(body=img_bytes, status=200, content_type="image/png"),
                )
                comfy_stub.router.add_get("/object_info", lambda _r: web.json_response({}))
                async with TestServer(comfy_stub) as comfy_srv:
                    app = dir_srv.create_app(comfy_url=f"http://127.0.0.1:{comfy_srv.port}")
                    async with TestClient(TestServer(app)) as client:
                        resp = await client.post(
                            "/api/context_suggest",
                            json={"source_ref": "boatshop.png", "x": 0.4, "y": 0.5, "rx": 0.2, "ry": 0.2},
                        )
                        assert resp.status == 200
                        doc = await resp.json()
                        assert doc["ok"] is True and doc["enabled"] is True
                        dprov = doc.get("provenance")
                        assert isinstance(dprov, dict) and tuple(dprov.keys()) == hlc.HEYLOOK_PROVENANCE_KEYS
                        assert dprov["resolved_model"] == "gemma-4-27b-it-mlx"
                        assert dprov["input_tokens"] == 210 and dprov["output_tokens"] == 64
                        last_dir_body = call_state["posted"][-1]
                        assert last_dir_body.get("max_tokens") == 2048, last_dir_body
                        assert "temperature" not in last_dir_body, (
                            f"director should not hardcode temperature=0.6 without preset: {last_dir_body}"
                        )
                        # Verify zero lore on live captured director system + user content
                        assert not scan_banned_lore(str(last_dir_body.get("system") or ""))
                        for blk in last_dir_body["messages"][0]["content"]:
                            if isinstance(blk, dict) and blk.get("type") == "text":
                                assert not scan_banned_lore(str(blk.get("text") or "")), blk

        asyncio.run(_check_director_provenance())
    finally:
        srv.shutdown()
        srv.server_close()
        if prev_url is None:
            os.environ.pop("H3_HEYLOOK_URL", None)
        else:
            os.environ["H3_HEYLOOK_URL"] = prev_url
        if prev_model is None:
            os.environ.pop("H3_HEYLOOK_MODEL", None)
        else:
            os.environ["H3_HEYLOOK_MODEL"] = prev_model
        if prev_pfx is None:
            os.environ.pop("H3_HEYLOOK_PRESET_PREFIX", None)
        else:
            os.environ["H3_HEYLOOK_PRESET_PREFIX"] = prev_pfx

    return "HeylookProvenance (14 keys + SSE message_start + repair token sum + [provenance] TOML) & all 5 callers verified"


def main() -> int:
    case("zero_lore_prompts_and_envelopes", check_zero_lore_prompts_and_envelopes)
    case("exemplar_diversity_and_optical_restraint", check_exemplar_diversity_and_optical_restraint)
    case("closed_vendor_vocabularies", check_closed_vendor_vocabularies)
    case("preset_prefix_resolution", check_preset_prefix_resolution)
    case("provenance_and_llm_pairs_contracts", check_provenance_and_llm_pairs_contracts)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
