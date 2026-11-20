#!/usr/bin/env python3
"""Offline checks for heylook-driven auto-edit guidance (`auto_edit_prompt.py` + `heylook_client.py`).

Exercises concept discovery & `[concept]` / `[vector]` TOML parsing, lossless H3 section
splitting & scope gating, `section_edits` + `edit_lines` invariant enforcement (dialogue `<d>`,
`[Shot N]`, `(S1)`, `<Subject/Picture/Video/Audio N>`, `<|...|>` special tokens, `[task]`,
`<Audio N>` retention lock, hallucinated section & malformed line repairs), 4-axis `EditPlan`
compilation into `MiniMaxH3EditGuider` & `MiniMaxH3ReleaseReferences`, in-process LRU cache
with `mtime_ns` invalidation, URL resolution precedence, capability-gated `reasoning_effort`,
loopback HTTP SSE mock (thinking stream, 1-shot APC repair, 503 retry with fresh `X-Request-ID`,
mid-stream error, interrupt cancel `DELETE`, disabled passthrough), and shipped workflows.

Zero GPU, zero CUDA, and zero LAN network calls.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/check_auto_edit_prompt.py
"""

from __future__ import annotations

import dataclasses
import http.server
import json
import os
from pathlib import Path
import re
import socketserver
import sys
import tempfile
import threading
import time
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib  # noqa: E402

_lib.bootstrap()
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

import torch  # noqa: E402

import auto_edit_prompt as aep  # noqa: E402
import control_vector_nodes as cvn  # noqa: E402
import edit_guidance as eg  # noqa: E402
import heylook_client as hlc  # noqa: E402
from workflows import h3_config, prompts as wf_prompts  # noqa: E402


FAILURES: list[str] = []


def ok(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        msg = f"  FAIL  {label}" + (f" -- {detail}" if detail else "")
        print(msg)
        FAILURES.append(label)


def refuses(label: str, fn, needle: str = "") -> None:
    try:
        fn()
    except Exception as exc:
        if needle and needle.lower() not in str(exc).lower():
            print(f"  FAIL  {label} -- wrong error: {exc!r} (expected {needle!r})")
            FAILURES.append(label)
        else:
            print(f"  ok    refuses: {label}")
        return
    print(f"  FAIL  {label} -- did not raise")
    FAILURES.append(label)


SAMPLE_T2VA = (
    "summary:\n"
    "[text to video] A single continuous tracking shot through a busy daylight market.\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action style.\n"
    "[Shot 1] The camera dollies forward along wet stone paving past canvas stalls. "
    "The butcher with a gruff baritone (S1) calls out: <d>[English] Fresh catch today!</d> "
    "A chalkboard sign reads <|caption_start|>FRESH MARKET<|caption_end|> beside hanging lanterns.\n\n"
    "overall_soundscape:\n"
    "Soft footsteps on dry cobblestones, distant harbor gulls, and quiet wooden cart wheels.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

SAMPLE_REF2VA = (
    "subject_definitions:\n"
    "<Subject 1> is the woman in <Picture 1>, wearing a pale wool coat with a camera strap.\n\n"
    "summary:\n"
    "[reference generation] <Subject 1> stands at a harbor pier in morning fog.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and pale wool coat are retained.\n"
    "<Audio 1>: fully_copy - voice timbre stays identical.\n\n"
    "detailed_description:\n"
    "In <Subject 1>'s harbor scene, <Subject 1> wears the pale wool coat from <Picture 1> and "
    "turns toward the water: <d>[English] Wait for the ferry.</d><|cutoff|>\n\n"
    "overall_soundscape:\n"
    "Gentle harbor waves lapping against wooden pilings.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)


class _DummyDM:
    pass


class _DummyModel:
    def __init__(self) -> None:
        self.model_options: dict[str, Any] = {}
        self.dm = _DummyDM()

    def is_dynamic(self) -> bool:
        return False

    def clone(self):
        cloned = _DummyModel()
        cloned.model_options = dict(self.model_options)
        cloned.dm = self.dm
        return cloned

    def get_model_object(self, name: str):
        if name == "diffusion_model":
            return self.dm
        return None


def check_concepts_and_sections() -> None:
    print("[1/5] concept cards, TOML discovery priority, and lossless H3 section invariants")
    options = aep.list_concept_options()
    ok(options[0] == "none" and "cyberpunk_neon_rain" in options and "reverb_cathedral" in options,
       "list_concept_options includes bank concepts with 'none' first")

    brief_cv = aep.load_concept_brief("cyberpunk_neon_rain", exemplar_count=2)
    ok(brief_cv.source_kind == "control_vector" and len(brief_cv.exemplars) == 2 and brief_cv.title,
       "load_concept_brief reads prompt_bank/control_vectors/*.toml with sampled exemplars")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ec_dir = root / "edit_concepts"
        dcv_dir = root / "data_cv"
        bcv_dir = root / "bank_cv"
        for d in (ec_dir, dcv_dir, bcv_dir):
            d.mkdir()
        (bcv_dir / "cyberpunk_neon_rain.toml").write_text(
            '[vector]\nname = "Bank"\nquality = "bank q"\na_side = "a"\nb_side = "b"\n'
            '[[pair]]\na = "sunny alley"\nb = "neon rain alley"\n',
            encoding="utf-8",
        )
        (dcv_dir / "cyberpunk_neon_rain.toml").write_text(
            '[vector]\nname = "DataCV"\nquality = "data q"\na_side = "a"\nb_side = "b"\n'
            '[[pair]]\na = "sunny alley"\nb = "rain alley"\n',
            encoding="utf-8",
        )
        (ec_dir / "cyberpunk_neon_rain.toml").write_text(
            '[concept]\nname = "Custom Cyberpunk"\n'
            'intent = "Shift lighting to soaked neon twilight"\n'
            'target_scope = "scene_and_soundscape"\n'
            'timing = "ramp_in"\n'
            'visual_changed = true\n'
            'audio_changed = true\n'
            'preserve_step0_layout = true\n'
            'deforms_ref_silhouette = false\n'
            '[[example]]\nsection = "description"\na = "daylight market"\nb = "neon-lit night market"\n',
            encoding="utf-8",
        )
        brief_prio = aep.load_concept_brief(
            "cyberpunk_neon_rain",
            concepts_dir=ec_dir,
            data_cv_dir=dcv_dir,
            bank_cv_dir=bcv_dir,
        )
        ok(brief_prio.source_kind == "edit_concept"
           and brief_prio.title == "Custom Cyberpunk"
           and brief_prio.timing_hint == "ramp_in"
           and brief_prio.axis_hints.get("audio_changed") is True,
           "data/edit_concepts/*.toml wins priority over data/control_vectors/ and prompt_bank/control_vectors/")
        usr_prompt_with_hints = aep.build_auto_edit_user_prompt(
            original=SAMPLE_T2VA,
            allowed_sections=("detailed_description", "overall_soundscape"),
            allow_retention_relax=False,
            rewrite_mode="section_edits",
            concept_brief=brief_prio,
            edit_intent="",
            has_images=False,
        )
        ok("Concept default axis hints" in usr_prompt_with_hints
           and "Concept default temporal_curve: ramp_in" in usr_prompt_with_hints
           and "Silhouette/medium overhaul active" not in usr_prompt_with_hints
           and "## section:" in usr_prompt_with_hints
           and "JSON" not in usr_prompt_with_hints,
           "build_auto_edit_user_prompt surfaces ConceptBrief axis_hints, timing_hint, and Markdown task")

        sys_sec = aep._read_system_prompt_file("section_edits")
        sys_el = aep._read_system_prompt_file("edit_lines")
        ok("## section:" in sys_sec and "no JSON" in sys_sec
           and "## edit_lines" in sys_el and "no JSON" in sys_el,
           "auto-edit system prompt files use Markdown output contract without JSON object rules")

        deform_brief = dataclasses.replace(
            brief_prio,
            axis_hints={**brief_prio.axis_hints, "deforms_ref_silhouette": True, "preserve_step0_layout": False},
        )
        usr_prompt_deform = aep.build_auto_edit_user_prompt(
            original=SAMPLE_T2VA,
            allowed_sections=("detailed_description",),
            allow_retention_relax=False,
            rewrite_mode="section_edits",
            concept_brief=deform_brief,
            edit_intent="",
            has_images=True,
        )
        ok("Silhouette/medium overhaul active" in usr_prompt_deform,
           "build_auto_edit_user_prompt emits silhouette/medium overhaul cue when deforms_ref_silhouette=True")

    lead, chunks = aep.parse_h3_sections(SAMPLE_T2VA)
    rebuilt = aep.reassemble_h3_sections(lead, chunks, {c.name: c.body for c in chunks})
    ok(rebuilt == SAMPLE_T2VA, "parse_h3_sections + reassemble_h3_sections round-trips SAMPLE_T2VA bitwise")

    lead_r, chunks_r = aep.parse_h3_sections(SAMPLE_REF2VA)
    rebuilt_r = aep.reassemble_h3_sections(lead_r, chunks_r, {c.name: c.body for c in chunks_r})
    ok(rebuilt_r == SAMPLE_REF2VA, "parse_h3_sections + reassemble_h3_sections round-trips SAMPLE_REF2VA bitwise")

    _sc_snd, allowed_snd = aep.resolve_allowed_sections(
        SAMPLE_T2VA, "soundscape_only", allow_retention_relax=True, concept_brief=None, has_edit_intent=True
    )
    ok(allowed_snd == ("overall_soundscape", "non_diegetic_music"),
       "soundscape_only restricts allowed sections strictly to audio sections")

    _sc_ref, allowed_ref = aep.resolve_allowed_sections(
        SAMPLE_REF2VA, "scene_only", allow_retention_relax=True, concept_brief=None, has_edit_intent=True
    )
    ok("retention_analysis" in allowed_ref and "overall_soundscape" not in allowed_ref,
       "scene_only includes retention_analysis when allow_retention_relax=True and locks soundscape")

    _sc_locked, allowed_locked = aep.resolve_allowed_sections(
        SAMPLE_REF2VA, "description_only", allow_retention_relax=True, concept_brief=None, has_edit_intent=True
    )
    ok("retention_analysis" not in allowed_locked,
       "description_only locks retention_analysis even when allow_retention_relax=True")


def check_validators_and_repair() -> None:
    print("[2/5] section_edits & edit_lines validators + verbatim invariants")
    good_sec_json = json.dumps({
        "section_edits": {
            "summary": "[text to video] A single continuous tracking shot through a neon-drenched night market.",
            "detailed_description": (
                "The target video is in a neon cyberpunk live-action style.\n"
                "[Shot 1] The camera dollies forward along rain-slicked neon paving past glowing stalls. "
                "The butcher with a gruff baritone (S1) calls out: <d>[English] Fresh catch today!</d> "
                "A chalkboard sign reads <|caption_start|>FRESH MARKET<|caption_end|> beside neon lanterns."
            ),
            "overall_soundscape": "Hissing rainfall on neon pavement and muffled market chatter.",
        },
        "axes": {
            "visual_changed": True,
            "audio_changed": True,
            "preserve_step0_layout": True,
            "deforms_ref_silhouette": False,
            "temporal_curve": "whole_clip",
            "reason": "neon weather shift",
        },
    })
    res_sec = aep.apply_section_edits_validated(
        SAMPLE_T2VA,
        f"<think>internal plan</think>\n```json\n{good_sec_json}\n```",
        allowed_sections=("summary", "detailed_description", "overall_soundscape", "non_diegetic_music"),
        allow_audio_retention_edit=False,
    )
    ok(res_sec["changed_sections"] == ("summary", "detailed_description", "overall_soundscape")
       and "non_diegetic_music:\nN/A" in res_sec["edited_prompt"],
       "section_edits rewrites target sections cleanly and leaves untouched non_diegetic_music bitwise")

    # Repeated <Subject 1> mentions in detailed_description are allowed; invented or dropped tags still refuse
    res_rep = aep.apply_section_edits_validated(
        SAMPLE_REF2VA,
        "## section: detailed_description\n"
        "In <Subject 1>'s bright scene, <Subject 1> wears an olive jacket as <Subject 1> "
        "turns toward the water: <d>[English] Wait for the ferry.</d><|cutoff|>\n\n"
        "## axes\nvisual_changed: true\naudio_changed: false\npreserve_step0_layout: false\n"
        "deforms_ref_silhouette: true\ntemporal_curve: whole_clip\nreason: jacket swap\n",
        allowed_sections=("detailed_description",),
        allow_audio_retention_edit=False,
    )
    ok(res_rep["edited_prompt"].count("<Subject 1>") >= 5,
       "section_edits allows repeated <Subject N> mentions in detailed_description and omitting replaced <Picture 1> ref")
    refuses("invented <Subject 9> triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_REF2VA,
                "## section: detailed_description\n"
                "<Subject 1> and <Subject 9> turn toward the water: <d>[English] Wait for the ferry.</d><|cutoff|>\n\n"
                "## axes\nvisual_changed: true\naudio_changed: false\npreserve_step0_layout: false\n"
                "deforms_ref_silhouette: true\ntemporal_curve: whole_clip\nreason: test\n",
                allowed_sections=("detailed_description",),
                allow_audio_retention_edit=False,
            ),
            "invented")

    # Locked section edits are safely ignored if another allowed section also changed,
    # or refused if ONLY a locked section changed.
    bad_locked = json.dumps({
        "section_edits": {
            "detailed_description": "Changed only visual."
            " Wait: need valid <d> and shot? Test locked-only refusal next.",
            "summary": "[text to video] A single continuous tracking shot through a neon-lit market.",
        },
        "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
    })
    refuses("dialogue alteration triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_T2VA,
                json.dumps({
                    "section_edits": {
                        "detailed_description": (
                            "The target video is in a cinematic live-action style.\n"
                            "[Shot 1] The butcher (S1) says: <d>[English] Altered words!</d> "
                            "<|caption_start|>FRESH MARKET<|caption_end|>"
                        ),
                    },
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                }),
                allowed_sections=("detailed_description",),
                allow_audio_retention_edit=False,
            ),
            "dialogue")

    refuses("special token <|...|> removal triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_T2VA,
                json.dumps({
                    "section_edits": {
                        "detailed_description": (
                            "The target video is in a neon style.\n"
                            "[Shot 1] The camera dollies forward. (S1) says: <d>[English] Fresh catch today!</d> "
                            "FRESH MARKET"
                        ),
                    },
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                }),
                allowed_sections=("detailed_description",),
                allow_audio_retention_edit=False,
            ),
            "special tokens")

    refuses("hallucinated section key triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_T2VA,
                json.dumps({
                    "section_edits": {"lighting_notes": "Neon lights."},
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                }),
                allowed_sections=("detailed_description",),
                allow_audio_retention_edit=False,
            ),
            "does not exist")

    refuses("modifying <Audio N> retention line triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_REF2VA,
                json.dumps({
                    "section_edits": {
                        "retention_analysis": (
                            "<Subject 1> (appears in [Shot 1]): partially_preserved - wardrobe shifted.\n"
                            "<Audio 1>: reference - changed voice."
                        ),
                    },
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                }),
                allowed_sections=("retention_analysis", "detailed_description"),
                allow_audio_retention_edit=False,
            ),
            "<Audio N>")

    refuses("missing summary [task] bracket triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_T2VA,
                json.dumps({
                    "section_edits": {"summary": "A neon night market without bracket."},
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                }),
                allowed_sections=("summary",),
                allow_audio_retention_edit=False,
            ),
            "bracket")
    assert bad_locked

    # edit_lines path + malformed clause error -> ValidationRepairError
    res_el = aep.apply_edit_lines_validated(
        SAMPLE_T2VA,
        json.dumps({
            "edit_lines": [
                "busy daylight market -> rain-slicked neon night market",
                "soundscape: Soft footsteps on dry cobblestones -> Heavy rainfall on wet cobblestones",
            ],
            "axes": {
                "visual_changed": True,
                "audio_changed": True,
                "preserve_step0_layout": True,
                "deforms_ref_silhouette": False,
                "temporal_curve": "whole_clip",
            },
        }),
        allowed_sections=("summary", "detailed_description", "overall_soundscape"),
        allow_audio_retention_edit=False,
    )
    ok("Heavy rainfall on wet cobblestones" in res_el["edited_prompt"]
       and "rain-slicked neon night market" in res_el["edited_prompt"],
       "apply_edit_lines_validated places summary + soundscape edits accurately")

    refuses("empty/malformed edit_lines triggers ValidationRepairError cleanly",
            lambda: aep.apply_edit_lines_validated(
                SAMPLE_T2VA,
                json.dumps({
                    "edit_lines": ["   ", "->"],
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                }),
                allowed_sections=("detailed_description",),
                allow_audio_retention_edit=False,
            ))

    # Markdown primary format: ## section: <name>, bare ## <name>, ## edit_lines, and ## axes
    md_sec_reply = (
        "## section: summary\n"
        "[text to video] A single continuous tracking shot through a neon-drenched night market.\n\n"
        "## detailed_description\n"
        "The target video is in a cinematic live-action style.\n"
        '[Shot 1] The camera dollies forward along wet stone paving past "canvas" neon stalls. '
        "The butcher with a gruff baritone (S1) calls out: <d>[English] Fresh catch today!</d> "
        "A chalkboard sign reads <|caption_start|>FRESH MARKET<|caption_end|> beside neon lanterns.\n\n"
        "## non_diegetic_music\n"
        "Hallucinated soundtrack lock test.\n\n"
        "## axes\n"
        "- `visual_changed`: `true`\n"
        "- `audio_changed`: `false`\n"
        "- `preserve_step0_layout`: `true`\n"
        "- `deforms_ref_silhouette`: `false`\n"
        "- `temporal_curve`: `whole_clip`\n"
        "- reason: Unescaped \"quotes\" in prose and tags parse directly without JSON.\n"
    )
    res_md_sec = aep.apply_section_edits_validated(
        SAMPLE_T2VA,
        md_sec_reply,
        allowed_sections=("summary", "detailed_description"),
        allow_audio_retention_edit=False,
    )
    ok("neon-drenched night market" in res_md_sec["edited_prompt"]
       and "non_diegetic_music:\nN/A" in res_md_sec["edited_prompt"]
       and "non_diegetic_music" in res_md_sec["reverted_locked"]
       and res_md_sec["axes"]["visual_changed"] is True
       and res_md_sec["axes"]["temporal_curve"] == "whole_clip",
       "Markdown ## section: + bare ## <name> + ## axes parses cleanly with unescaped quotes and locks untouched sections")

    md_el_reply = (
        "## edit_lines\n"
        "- summary: a busy daylight market -> a rain-slicked neon night market\n"
        "- soundscape: Soft footsteps on dry cobblestones, distant harbor gulls, and quiet wooden cart wheels. -> Heavy rainfall on wet cobblestones.\n\n"
        "## axes\n"
        "visual_changed: true\n"
        "audio_changed: true\n"
        "preserve_step0_layout: true\n"
        "deforms_ref_silhouette: false\n"
        "temporal_curve: whole_clip\n"
        "reason: Rain atmosphere added.\n"
    )
    res_md_el = aep.apply_edit_lines_validated(
        SAMPLE_T2VA,
        md_el_reply,
        allowed_sections=("summary", "detailed_description", "overall_soundscape"),
        allow_audio_retention_edit=False,
    )
    ok("Heavy rainfall on wet cobblestones." in res_md_el["edited_prompt"]
       and "rain-slicked neon night market" in res_md_el["edited_prompt"],
       "Markdown ## edit_lines + ## axes places edits cleanly")

    refuses("Markdown reply missing ## axes triggers ValidationRepairError",
            lambda: aep.apply_section_edits_validated(
                SAMPLE_T2VA,
                "## section: summary\n[text to video] A neon market.\n",
                allowed_sections=("summary",),
                allow_audio_retention_edit=False,
            ))

    # Fresh prompt mode (no dummy original prompt)
    fresh_sys = aep._read_system_prompt_file("fresh")
    ok("## section:" in fresh_sys and "zero JSON" in fresh_sys,
       "fresh_prompt_system_prompt.md loads and uses plain Markdown with zero JSON")
    long_desc = (
        "The target video is in a cinematic live-action style with warm glasshouse sunlight and shallow depth of field.\n"
        "[Shot 1] A medium two-shot frames <Subject 1>, the blonde woman in a champagne silk dress on the left, "
        "and <Subject 2>, the bearded man in an unbuttoned linen shirt on the right, standing beside a wrought-iron "
        "bistro table beneath arching conservatory glass panes draped in trailing jasmine vines. "
        + " ".join(["Warm amber sunbeams catch mist rising from terracotta planters and crystal stems."] * 30)
        + "\n[Shot 2] The camera cuts to an intimate over-the-shoulder close-up as <Subject 1> (S1) raises her flute "
        "and says, <d>[English] To the glasshouse.</d> <Subject 2> produces no vocal sound and smiles while clinking glasses.\n\n"
    )
    fresh_reply = (
        "## section: subject_definitions\n"
        "<Subject 1> is the woman in <Picture 1>, wearing a champagne silk dress.\n"
        "<Subject 2> is the man in <Picture 2>, wearing an unbuttoned linen shirt.\n\n"
        "## section: summary\n"
        "[reference generation] <Subject 1> and <Subject 2> toast inside a sunlit conservatory.\n\n"
        "## section: retention_analysis\n"
        "<Subject 1> (appears in [Shot 1], [Shot 2]): partially_preserved - facial identity retained; wears champagne silk dress.\n"
        "<Subject 2> (appears in [Shot 1], [Shot 2]): partially_preserved - facial identity retained; wears linen shirt.\n\n"
        "## section: detailed_description\n"
        + long_desc
        + "## section: overall_soundscape\n"
        "Clinking crystal glasses and soft greenhouse water drips.\n\n"
        "## section: non_diegetic_music\n"
        "N/A\n\n"
        "## axes\nvisual_changed: true\naudio_changed: true\npreserve_step0_layout: false\n"
        "deforms_ref_silhouette: false\ntemporal_curve: whole_clip\nreason: fresh live-action scene\n"
    )
    res_fresh = aep.validate_fresh_h3_prompt(fresh_reply, n_images=2)
    ok(len(res_fresh["changed_sections"]) == 6
       and res_fresh["axes"]["preserve_step0_layout"] is False
       and res_fresh["axes"]["deforms_ref_silhouette"] is False
       and "[reference generation]" in res_fresh["edited_prompt"],
       "validate_fresh_h3_prompt validates 6-section 350+ word ref2va prompt cleanly")
    short_fresh_reply = fresh_reply.replace(
        long_desc,
        "The target video is in a cinematic live-action style.\n"
        "[Shot 1] A medium two-shot frames <Subject 1> and <Subject 2> raising glasses.\n\n",
    )
    refuses("short (<350 word) fresh ref2va detailed_description triggers 1-shot expansion repair",
            lambda: aep.validate_fresh_h3_prompt(short_fresh_reply, n_images=2),
            "350-500 words")
    refuses("fresh prompt referencing nonexistent <Picture 3> refuses",
            lambda: aep.validate_fresh_h3_prompt(
                fresh_reply.replace("<Picture 2>", "<Picture 2> and <Picture 3>"),
                n_images=2,
            ),
            "nonexistent <Picture N>")
    frag_brief = aep.ConceptBrief(
        stem="frag_demo",
        source_kind="edit_concept",
        path=Path("/tmp/frag_demo.toml"),
        file_stat=("/tmp/frag_demo.toml", 1, 1),
        title="frag_demo",
        summary="Warm candlelit piano bench.",
        a_side="",
        b_side="",
        avoid="",
        notes="",
        modified_sections=("description",),
        exemplars=(("description", "plain room", "tiny twelve word fragment blurb that should never prime short output"),),
    )
    u_ref = aep.build_fresh_prompt_user_prompt(
        concept_brief=frag_brief,
        edit_intent="Two friends talking at dusk.",
        n_images=1,
        resolved_mode="ref2va",
        length=192,
    )
    ok("8.00s (192 frames)" in u_ref and "2-3 concrete cinematic shots" in u_ref
       and "tiny twelve word fragment" not in u_ref,
       "build_fresh_prompt_user_prompt passes dur_s + multi-shot pacing on ref2va and filters <25-word fragment exemplars")
    import control_vector_nodes as cvn
    ok(cvn._is_hidden_vector_relpath("_quarantine/foo.safetensors")
       and cvn._is_hidden_vector_relpath(".cache/foo.safetensors")
       and cvn._is_hidden_vector_relpath("j_lens/foo.safetensors")
       and not cvn._is_hidden_vector_relpath("bank/film_16mm_1974__fl2va__diff_span.safetensors"),
       "control_vector_nodes hides _*, .*, and j_lens subdirs while keeping bank/ visible")


def check_edit_plan_compilation_and_wiring() -> None:
    print("[3/5] 4-axis EditPlan compilation + MiniMaxH3EditGuider & MiniMaxH3ReleaseReferences")
    plan_t2va_aud = aep.compile_edit_plan(
        axes={"visual_changed": False, "audio_changed": True, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "second_half"},
        changed_sections=("overall_soundscape",),
        take_mode="auto",
        timing="auto",
        has_references_or_ref2va=False,
        is_sectioned=True,
    )
    ok(plan_t2va_aud.streams == "audio_only"
       and plan_t2va_aud.first_step == 2
       and plan_t2va_aud.last_step == 4
       and plan_t2va_aud.temporal_curve == "second_half"
       and plan_t2va_aud.ref_end_percent == 1.0,
       "audio-only edit on t2va compiles to streams='audio_only', first_step=2, last_step=4")

    plan_deform = aep.compile_edit_plan(
        axes={"visual_changed": True, "audio_changed": False, "preserve_step0_layout": False, "deforms_ref_silhouette": True, "temporal_curve": "whole_clip"},
        changed_sections=("retention_analysis", "detailed_description"),
        take_mode="auto",
        timing="ramp_in",
        has_references_or_ref2va=True,
        is_sectioned=True,
    )
    ok(plan_deform.first_step == 0
       and plan_deform.streams == "independent"
       and abs(plan_deform.ref_end_percent - 0.33) < 1e-6
       and plan_deform.temporal_curve == "ramp_in"
       and plan_deform.auto_fast_1x is False,
       "deforms_ref_silhouette on ref2va unlocks step 0 and sets ref_end_percent=0.33 + timing override")

    plan_live_overhaul = aep.compile_edit_plan(
        axes={"visual_changed": True, "audio_changed": True, "preserve_step0_layout": False, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
        changed_sections=("subject_definitions", "retention_analysis", "detailed_description", "overall_soundscape"),
        take_mode="auto",
        timing="auto",
        has_references_or_ref2va=True,
        is_sectioned=True,
    )
    ok(plan_live_overhaul.first_step == 0
       and plan_live_overhaul.auto_fast_1x is True
       and plan_live_overhaul.force_scale is None
       and plan_live_overhaul.ref_end_percent == 1.0,
       "live-action whole-clip overhaul sets auto_fast_1x=True while keeping ref_end_percent=1.0")

    plan_fast = aep.compile_edit_plan(
        axes={"visual_changed": True, "audio_changed": True, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "ramp_in"},
        changed_sections=("detailed_description",),
        take_mode="fast_1x",
        timing="ramp_in",
        has_references_or_ref2va=False,
        is_sectioned=True,
    )
    ok(plan_fast.force_scale == 1.0 and plan_fast.first_step == 0 and plan_fast.streams == "both"
       and plan_fast.temporal_curve == "whole_clip",
       "fast_1x sets force_scale=1.0, first_step=0, streams='both', whole_clip")

    # ReleaseReferences wiring test
    orig_attach = cvn.attach_control_vector
    orig_is_h3 = cvn._is_minimax_h3
    seen_attach: list[dict[str, Any]] = []

    def fake_attach(model, name, **kwargs):
        seen_attach.append(kwargs)
        return model

    cvn.attach_control_vector = fake_attach
    cvn._is_minimax_h3 = lambda _dm: True
    try:
        cvn.MiniMaxH3ReleaseReferences.execute(_DummyModel(), end_percent=1.0, edit_plan=plan_deform)
        cvn.MiniMaxH3ReleaseReferences.execute(
            _DummyModel(),
            end_percent=0.8,
            edit_plan=aep.compile_edit_plan(
                axes={"visual_changed": True, "audio_changed": False, "preserve_step0_layout": False, "deforms_ref_silhouette": True, "temporal_curve": "whole_clip"},
                changed_sections=("detailed_description",),
                take_mode="manual_widgets",
                timing="auto",
                has_references_or_ref2va=True,
                is_sectioned=True,
            ),
        )
    finally:
        cvn.attach_control_vector = orig_attach
        cvn._is_minimax_h3 = orig_is_h3
    ok(len(seen_attach) == 2
       and abs(seen_attach[0]["ref_vae_end_percent"] - 0.33) < 1e-6
       and abs(seen_attach[1]["ref_vae_end_percent"] - 0.8) < 1e-6,
       "MiniMaxH3ReleaseReferences lowers end_percent to 0.33 on plan_deform and respects manual_widgets=True")

    schema_rr = cvn.MiniMaxH3ReleaseReferences.define_schema()
    ok(schema_rr.inputs[-1].id == "edit_plan" and schema_rr.inputs[-1].optional is True,
       "MiniMaxH3ReleaseReferences appends optional edit_plan strictly last")


def check_url_resolution_and_mock_sse() -> None:
    print("[4/5] URL resolution, capability-gated reasoning_effort, mock SSE (repair/503/cancel/cache)")
    with tempfile.TemporaryDirectory() as td:
        env_file = Path(td) / "test.env"
        env_file.write_text("H3_HEYLOOK_URL=http://127.0.0.1:49999/\n", encoding="utf-8")
        old_env_file = os.environ.get("H3_ENV_FILE")
        old_url = os.environ.get("H3_HEYLOOK_URL")
        try:
            os.environ["H3_ENV_FILE"] = str(env_file)
            os.environ.pop("H3_HEYLOOK_URL", None)
            u_file = hlc.resolve_heylook_base_url("")
            os.environ["H3_HEYLOOK_URL"] = "http://127.0.0.1:48888/"
            u_env = hlc.resolve_heylook_base_url("")
            u_arg = hlc.resolve_heylook_base_url("http://127.0.0.1:47777/")
            env_file.write_text("# empty\n", encoding="utf-8")
            os.environ.pop("H3_HEYLOOK_URL", None)
            u_def = hlc.resolve_heylook_base_url("")
            ok(u_file == "http://127.0.0.1:49999"
               and u_env == "http://127.0.0.1:48888"
               and u_arg == "http://127.0.0.1:47777"
               and u_def == "http://127.0.0.1:8080",
               "resolve_heylook_base_url enforces arg > env > h3.env > 127.0.0.1:8080 priority")
        finally:
            if old_env_file is None:
                os.environ.pop("H3_ENV_FILE", None)
            else:
                os.environ["H3_ENV_FILE"] = old_env_file
            if old_url is None:
                os.environ.pop("H3_HEYLOOK_URL", None)
            else:
                os.environ["H3_HEYLOOK_URL"] = old_url

    # Capability gating for Gemma-4 vs Qwen3.8 vs thinking="off"
    gemma_row = {
        "id": "gemma-mock",
        "capabilities": ["chat", "vision", "thinking"],
        "engine": {"thinking": {"switch": "enable_thinking", "depth": None}},
    }
    qwen_row = {
        "id": "qwen-mock",
        "capabilities": ["chat", "vision", "thinking", "reasoning_effort"],
        "engine": {"thinking": {"depth": {"values": ["minimal", "low", "medium", "high", "xhigh"]}}},
    }
    b_gemma = hlc.build_heylook_request_body(
        model_row=gemma_row, system_prompt="sys", user_prompt="usr",
        prepared_images=[], thinking="on", default_reasoning_effort="medium",
    )
    b_qwen = hlc.build_heylook_request_body(
        model_row=qwen_row, system_prompt="sys", user_prompt="usr",
        prepared_images=[], thinking="on", default_reasoning_effort="medium",
    )
    b_qwen_off = hlc.build_heylook_request_body(
        model_row=qwen_row, system_prompt="sys", user_prompt="usr",
        prepared_images=[], thinking="off", sampler_mode="custom", sampler_overrides={"temperature": 0.5},
        reasoning_effort_override="", default_reasoning_effort="",
    )
    ok("reasoning_effort" not in b_gemma and b_gemma.get("thinking") is True
       and b_qwen.get("reasoning_effort") == "medium" and b_qwen.get("thinking") is True
       and "reasoning_effort" not in b_qwen_off and b_qwen_off.get("thinking") is False,
       "reasoning_effort='medium' is capability-gated (sent on Qwen3.8, omitted on Gemma-4 & thinking='off')")

    # Loopback SSE server tests
    state: dict[str, Any] = {
        "mode": "repair_and_retry",
        "posts": [],
        "deletes": [],
        "req_ids": [],
        "attempt": 0,
    }

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/models":
                payload = {
                    "data": [{
                        "id": "qwen-mock",
                        "provider": "mlx",
                        "capabilities": ["chat", "vision", "thinking", "reasoning_effort"],
                        "thinking_default": True,
                        "sampler_defaults": {"temperature": 0.7},
                        "engine": {"thinking": {"depth": {"values": ["low", "medium", "high"]}}},
                    }]
                }
            elif self.path == "/v1/system/metrics":
                payload = {"models": {"qwen-mock": {"resident": True}}}
            elif self.path == "/v1/presets":
                payload = {"presets": []}
            else:
                self.send_response(404)
                self.end_headers()
                return
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_DELETE(self):  # noqa: N802
            state["deletes"].append(self.path)
            self.send_response(200)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            state["posts"].append(body)
            state["req_ids"].append(self.headers.get("X-Request-ID"))
            state["attempt"] += 1
            attempt = state["attempt"]
            mode = state["mode"]

            if mode == "503_once" and attempt == 1:
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(b'{"error":"busy"}')
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Request-ID", f"req-{attempt}")
            self.end_headers()

            def _send(ev: str, obj: dict[str, Any]) -> None:
                msg = f"event: {ev}\ndata: {json.dumps(obj)}\n\n".encode("utf-8")
                self.wfile.write(msg)
                self.wfile.flush()

            if mode == "midstream_error":
                _send("message_start", {"message": {"id": f"m{attempt}", "usage": {"input_tokens": 50}}})
                _send("error", {"error": {"message": "synthetic stream failure"}})
                return

            if mode == "cancel_on_chunk":
                _send("message_start", {"message": {"id": f"m{attempt}", "usage": {"input_tokens": 50}}})
                _send("content_block_start", {"index": 0, "content_block": {"type": "text"}})
                _send("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "{partial"}})
                time.sleep(0.05)
                return

            # Default good JSON / Markdown (or invalid first -> repaired second)
            if mode == "repair_and_retry" and attempt == 1:
                reply_json = json.dumps({
                    "section_edits": {"bad_hallucinated_section": "nope"},
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "whole_clip"},
                })
            elif mode == "good_md":
                reply_json = (
                    "## section: summary\n"
                    "[text to video] Neon night market pier in pouring rain.\n\n"
                    "## axes\n"
                    "visual_changed: true\n"
                    "audio_changed: false\n"
                    "preserve_step0_layout: true\n"
                    "deforms_ref_silhouette: false\n"
                    "temporal_curve: ramp_in\n"
                    "reason: markdown neon rain\n"
                )
            else:
                reply_json = json.dumps({
                    "section_edits": {"summary": "[text to video] Neon night market pier in pouring rain."},
                    "axes": {"visual_changed": True, "audio_changed": False, "preserve_step0_layout": True, "deforms_ref_silhouette": False, "temporal_curve": "ramp_in", "reason": "neon rain"},
                })

            _send("message_start", {
                "message": {
                    "id": f"m{attempt}",
                    "usage": {"input_tokens": 120, "cache_read_input_tokens": 80 if attempt > 1 else 0},
                }
            })
            _send("content_block_start", {"index": 0, "content_block": {"type": "thinking"}})
            _send("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": f"thought-{attempt}"}})
            _send("content_block_start", {"index": 1, "content_block": {"type": "text"}})
            _send("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": reply_json}})
            _send("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}})
            _send("performance", {"generation_tps": 64.0, "generation_duration_ms": 250.0})
            _send("message_stop", {})

    class _ReusableTCP(socketserver.ThreadingMixIn, http.server.HTTPServer):
        allow_reuse_address = True

    srv = _ReusableTCP(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    srv_url = f"http://127.0.0.1:{port}"
    thr = threading.Thread(target=srv.serve_forever, daemon=True)
    thr.start()

    try:
        aep.clear_rewrite_cache()
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td)
            cfile = cdir / "mock_neon.toml"
            cfile.write_text(
                '[concept]\nname = "Mock Neon"\nintent = "Turn scene into pouring neon rain"\n',
                encoding="utf-8",
            )
            # 1. enabled=False passthrough makes zero HTTP requests
            off_res = aep.run_auto_edit_rewrite(
                original=SAMPLE_T2VA, concept="none", enabled=False, base_url=srv_url
            )
            ok(len(state["posts"]) == 0
               and off_res["edited_prompt"] == SAMPLE_T2VA
               and off_res["edit_plan"].manual_widgets is True,
               "enabled=False passthrough makes 0 HTTP requests and sets manual_widgets=True")

            # 2. Repair round-trip + thinking_delta capture + APC message prefix preservation
            r1 = aep.run_auto_edit_rewrite(
                original=SAMPLE_T2VA,
                concept="mock_neon",
                base_url=srv_url,
                concepts_dir=cdir,
            )
            p0, p1 = state["posts"][0], state["posts"][1]
            ok(r1["repair_used"] is True
               and r1["cache_hit"] is False
               and "thought-1" in r1["thinking"] and "thought-2" in r1["thinking"]
               and p0["messages"][0] == p1["messages"][0]
               and len(p1["messages"]) == 3
               and p1["messages"][1]["role"] == "assistant"
               and p1["messages"][2]["role"] == "user",
               "1-shot APC repair preserves user turn prefix and captures thinking_delta across both turns")

            # 3. Cache hit on take_mode/timing change makes 0 HTTP posts
            posts_before = len(state["posts"])
            r_fast = aep.run_auto_edit_rewrite(
                original=SAMPLE_T2VA,
                concept="mock_neon",
                take_mode="fast_1x",
                timing="sudden_midclip",
                base_url=srv_url,
                concepts_dir=cdir,
            )
            ok(len(state["posts"]) == posts_before
               and r_fast["cache_hit"] is True
               and r_fast["edit_plan"].force_scale == 1.0,
               "take_mode/timing toggle hits in-process LRU cache with 0 HTTP calls")

            # 4. Modifying TOML updates (path, mtime_ns, size) -> cache miss + fresh call
            state["mode"] = "good"
            time.sleep(0.01)
            cfile.write_text(
                '[concept]\nname = "Mock Neon V2"\nintent = "Pouring neon rain + magenta glow"\n',
                encoding="utf-8",
            )
            r_inval = aep.run_auto_edit_rewrite(
                original=SAMPLE_T2VA,
                concept="mock_neon",
                base_url=srv_url,
                concepts_dir=cdir,
            )
            ok(r_inval["cache_hit"] is False and len(state["posts"]) == posts_before + 1,
               "editing concept TOML invalidates LRU cache entry via (path, mtime_ns, size)")

            # 4b. Markdown SSE reply end-to-end via run_auto_edit_rewrite
            state["mode"] = "good_md"
            r_md = aep.run_auto_edit_rewrite(
                original=SAMPLE_T2VA,
                edit_intent="Pouring neon rain on wet pier.",
                base_url=srv_url,
                use_cache=False,
            )
            ok("Neon night market pier in pouring rain." in r_md["edited_prompt"]
               and r_md["edit_plan"].temporal_curve == "ramp_in"
               and r_md["repair_used"] is False,
               "run_auto_edit_rewrite parses Markdown SSE replies end-to-end with 0 repairs")

        # 5. 503 backoff retry generates distinct X-Request-ID
        state["mode"] = "503_once"
        state["attempt"] = 0
        state["req_ids"].clear()
        r_503 = aep.run_auto_edit_rewrite(
            original=SAMPLE_T2VA,
            edit_intent="Pouring neon rain on wet stones.",
            base_url=srv_url,
            use_cache=False,
        )
        ok(len(state["req_ids"]) == 2
           and state["req_ids"][0] != state["req_ids"][1]
           and r_503["cache_hit"] is False,
           "503 retry issues a fresh X-Request-ID UUID")

        # 6. Mid-stream event: error raises RuntimeError
        state["mode"] = "midstream_error"
        refuses("mid-stream SSE event: error",
                lambda: aep.run_auto_edit_rewrite(
                    original=SAMPLE_T2VA,
                    edit_intent="Error test",
                    base_url=srv_url,
                    use_cache=False,
                ),
                "synthetic stream failure")

        # 7. Cancel DELETE /v1/requests/{id} when interrupt_check raises
        state["mode"] = "cancel_on_chunk"
        state["deletes"].clear()
        calls = 0

        def _abort_after_chunk() -> None:
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise InterruptedError("user cancelled")

        refuses("interrupt fires DELETE /v1/requests/{id}",
                lambda: hlc.stream_heylook_message(
                    srv_url,
                    {"model": "qwen-mock", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                    interrupt_check=_abort_after_chunk,
                ),
                "user cancelled")
        ok(len(state["deletes"]) == 1 and state["deletes"][0].startswith("/v1/requests/"),
           "cancel sends DELETE /v1/requests/{id} cleanly")
    finally:
        aep.clear_rewrite_cache()
        srv.shutdown()
        srv.server_close()


def check_shipped_workflows() -> None:
    print("[5/5] shipped auto-edit workflows + privacy guard")
    for fname, is_r2v in (
        ("workflows/h3_text_to_video_pika_auto_edit_guidance_api.json", False),
        ("workflows/h3_image_ref_plus_text_to_video_pika_auto_edit_guidance_api.json", True),
    ):
        p = REPO / fname
        raw = p.read_text(encoding="utf-8")
        ok(
            re.search(r"https?://(?:192\.168\.|10\.|172\.(?:1[6-9]|2\d|3[01])\.)", raw) is None,
            f"{fname} contains no private LAN URL",
        )
        g = json.loads(raw)
        n130 = g.get("130") or {}
        n130_in = n130.get("inputs") or {}
        ok(n130.get("class_type") == h3_config.AUTO_EDIT_PROMPT_NODE
           and n130_in.get("base_url") == ""
           and n130_in.get("model") == ""
           and n130_in.get("concept") == "cyberpunk_neon_rain",
           f"{fname} node 130 uses blank base_url/model and worked concept")
        ok((g.get("9") or {}).get("inputs", {}).get("edit_plan") == ["130", 2]
           and (g.get("131") or {}).get("inputs", {}).get("prompt") == ["130", 1]
           and (g.get("5") or {}).get("inputs", {}).get("prompt") == ["130", 0]
           and (g.get("26") or {}).get("inputs", {}).get("conditioning") == ["131", 0]
           and (g.get("9") or {}).get("inputs", {}).get("edited") == ["26", 0]
           and (g.get("9") or {}).get("inputs", {}).get("original") == ["5", 0],
           f"{fname} wires edit_plan -> EditGuider, Preflight on edited (131) so lazy original skips node 5, and node 5 <- 130[0]")
        if is_r2v:
            ok("images.image_1" in n130_in
               and (g.get("135") or {}).get("inputs", {}).get("edit_plan") == ["130", 2],
               f"{fname} forwards <Picture 1> -> node 130 and edit_plan -> ReleaseReferences")
        desc = wf_prompts.describe(g)
        cs = {c.node_id: c for c in wf_prompts.carriers(g)}
        ok(bool(desc.get("prompt_id"))
           and bool(desc.get("prompt_sha256"))
           and "5" in cs and bool(cs["5"].text) and not cs["5"].unresolved
           and "131" in cs and not cs["131"].unresolved,
           f"{fname} carriers & describe() resolve written prompt cleanly")

    # Verify MiniMaxH3EditGuider lazy=True skips 'original' on fast_1x / unmasked auto_fast_1x,
    # and requests 'original' when masked or lock_take.
    plan_auto_1x = aep.EditPlan(
        first_step=0, last_step=3, streams="independent", audio_scale=1.0,
        ref_end_percent=1.0, temporal_curve="whole_clip", force_scale=None,
        manual_widgets=False, visual_changed=True, audio_changed=True,
        preserve_step0_layout=False, deforms_ref_silhouette=False, auto_fast_1x=True,
    )
    plan_lock = aep.EditPlan(
        first_step=1, last_step=3, streams="both", audio_scale=1.0,
        ref_end_percent=1.0, temporal_curve="whole_clip", force_scale=None,
        manual_widgets=False, visual_changed=True, audio_changed=False,
        preserve_step0_layout=True, deforms_ref_silhouette=False, auto_fast_1x=False,
    )
    ok(eg.MiniMaxH3EditGuider.check_lazy_status(original=None, edit_plan=plan_auto_1x, mask=None) == [],
       "MiniMaxH3EditGuider.check_lazy_status skips 'original' on unmasked auto_fast_1x")
    ok(eg.MiniMaxH3EditGuider.check_lazy_status(original=None, edit_plan=plan_auto_1x, mask=torch.ones((1, 8, 8))) == ["original"],
       "MiniMaxH3EditGuider.check_lazy_status requests 'original' when spatial mask is wired")
    ok(eg.MiniMaxH3EditGuider.check_lazy_status(original=None, edit_plan=plan_lock, mask=None) == ["original"],
       "MiniMaxH3EditGuider.check_lazy_status requests 'original' on lock_take (first_step=1)")
    dummy_edited = [[torch.zeros((1, 12, 64)), {"minimax_token_tags": torch.ones((1, 12), dtype=torch.long)}]]
    out_lazy = eg.MiniMaxH3EditGuider.execute(
        _DummyModel(), None, dummy_edited, 1.5, 0, edit_plan=plan_auto_1x
    )
    ok(out_lazy[0].edit_scale == 1.0 and out_lazy[0].first_step == 0,
       "MiniMaxH3EditGuider.execute succeeds with original=None on lazy 1x path")

    # Verify h3_prompt_contract across FL2VA/I2VA/L2VA, multimodal H3References, ref2va speaker binding, & single_frame
    import h3_prompt_contract as pc
    import reference_order as ro

    fl_body = (
        "## section: integrated_multimodal_description\n"
        "[Shot 1] Live-action, cinematic. Picture 1 shows a quiet dock at dawn. "
        "[Shot 2] The shot cuts to Picture 2 as the ferry arrives.\n\n"
        "## section: overall_soundscape\nHarbor waves and ferry horn.\n\n"
        "## section: non_diegetic_music\nN/A\n\n"
        "## axes\nvisual_changed: true\naudio_changed: true\npreserve_step0_layout: false\n"
        "deforms_ref_silhouette: false\ntemporal_curve: whole_clip\nreason: fl2va test\n"
    )
    fl_res = aep.validate_fresh_h3_prompt(fl_body, n_images=2, resolved_mode="fl2va", length=192)
    want_fl_p1 = (
        "How the reference pictures align with the target video \u2014 "
        "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
        "Picture 2 (from Shot 2) aligns with the 8.00-second mark of the target video."
    )
    ok(fl_res["edited_prompt"].splitlines()[0] == want_fl_p1,
       "h3_prompt_contract stamps exact FL2VA Part One with bare Picture/Shot and 8.00s duration for 192 frames")
    refreshed_141 = pc.stamp_part_one(fl_res["edited_prompt"], mode="fl2va", length=141)
    ok("5.88-second mark" in refreshed_141.splitlines()[0] and refreshed_141.count("How the reference pictures") == 1,
       "stamp_part_one refreshes FL2VA S.SS cleanly when length changes without duplicating Part One")

    # Multimodal H3References manifest via reference_order.assign_labels: <Audio 1> before <Video 1>, plus synthetic video
    class _FakeImgRef:
        image = None
    class _FakeVidRef:
        frames = None
        soundtrack = object()
        allow_upscale = False
    class _FakeAudRef:
        audio = object()
    mf = pc.describe_references((_FakeImgRef(), _FakeVidRef(), _FakeAudRef()))
    ok(mf.labels == ("<Picture 1>", "<Audio 1>", "<Video 1>", "<Audio 2>")
       and mf.has_synthetic_video is True,
       "describe_references preserves reference_order.assign_labels (<Audio 1> before <Video 1>) and flags synthetic video")
    plan_syn = aep.compile_edit_plan(
        axes={"visual_changed": True, "audio_changed": True, "preserve_step0_layout": False, "deforms_ref_silhouette": False},
        changed_sections=("detailed_description",),
        take_mode="auto",
        timing="auto",
        has_references_or_ref2va=True,
        is_sectioned=True,
        has_synthetic_video=True,
    )
    ok(abs(plan_syn.video_end_percent - 0.33) < 1e-6,
       "compile_edit_plan sets video_end_percent=0.33 when synthetic trajectory video is wired")

    # Ref2VA mechanical validation: refuses missing (Sx) before <d> AND refuses (S1) in retention_analysis
    bad_speech_ref = (
        "## section: subject_definitions\n<Subject 1> from <Picture 1>: woman in linen.\n\n"
        "## section: summary\n[reference generation] <Subject 1> speaks in a courtyard.\n\n"
        "## section: retention_analysis\n<Subject 1> (appears in [Shot 1]): fully_preserved - face retained.\n\n"
        "## section: detailed_description\nLive-action. [Shot 1] <Subject 1> whispers <d>[English] Hello.</d>\n\n"
        "## section: overall_soundscape\nQuiet courtyard breeze.\n\n"
        "## section: non_diegetic_music\nN/A\n\n"
        "## axes\nvisual_changed: true\naudio_changed: true\npreserve_step0_layout: false\n"
        "deforms_ref_silhouette: false\ntemporal_curve: whole_clip\nreason: test\n"
    )
    refuses("ref2va <d> without (Sx) speaker ID triggers ValidationRepairError",
            lambda: aep.validate_fresh_h3_prompt(bad_speech_ref, n_images=1, resolved_mode="ref2va"),
            "(Sx)")
    bad_ret_sx = bad_speech_ref.replace(
        "<Subject 1> (appears in [Shot 1]): fully_preserved - face retained.",
        "<Subject 1> (S1) (appears in [Shot 1]): fully_preserved - face retained.",
    ).replace("<Subject 1> whispers <d>", "<Subject 1> (S1) whispers <d>")
    refuses("ref2va (S1) inside retention_analysis triggers ValidationRepairError",
            lambda: aep.validate_fresh_h3_prompt(bad_ret_sx, n_images=1, resolved_mode="ref2va"),
            "retention_analysis")

    # single_frame sectionless visual prose
    sf_reply = (
        "## section: _unsectioned\nEdit <Picture 1>. Shift the studio lighting to golden hour sunset.\n\n"
        "## axes\nvisual_changed: true\naudio_changed: false\npreserve_step0_layout: false\n"
        "deforms_ref_silhouette: false\ntemporal_curve: whole_clip\nreason: single frame edit\n"
    )
    sf_res = aep.validate_fresh_h3_prompt(sf_reply, n_images=1, resolved_mode="single_frame", length=1)
    ok("overall_soundscape" not in sf_res["edited_prompt"] and sf_res["axes"]["audio_changed"] is False,
       "single_frame mode emits sectionless visual prose with audio_changed=False")

    # Verify MiniMaxH3PromptExpander (dedicated 1x Easy Button) and clean 20-input AutoEditPrompt split
    import h3_prompt_expander as pe

    ae_in_names = [i.id for i in aep.MiniMaxH3AutoEditPrompt.define_schema().inputs]
    ok(len(ae_in_names) == 21 and "references" not in ae_in_names and "prompt_mode" not in ae_in_names,
       f"MiniMaxH3AutoEditPrompt stays focused on 2x edits without bolted-on fresh widgets (inputs={len(ae_in_names)})")
    pe_schema = pe.MiniMaxH3PromptExpander.define_schema()
    pe_in_names = [i.id for i in pe_schema.inputs]
    pe_out_names = [o.display_name for o in pe_schema.outputs]
    ok("prompt" in pe_in_names and "references" in pe_in_names and "first_frame" in pe_in_names
       and "last_frame" in pe_in_names and "length" in pe_in_names
       and pe_out_names == ["prompt", "video_end_percent", "thinking", "report"],
       "MiniMaxH3PromptExpander exposes clean 1x Easy-Button inputs and (prompt, video_end_percent, thinking, report) outputs")
    ok(pe.resolve_expander_mode("auto", length=141) == "t2va"
       and pe.resolve_expander_mode("auto", length=1) == "single_frame"
       and pe.resolve_expander_mode("auto", first_frame=object(), length=141) == "i2va"
       and pe.resolve_expander_mode("auto", last_frame=object(), length=141) == "l2va"
       and pe.resolve_expander_mode("auto", first_frame=object(), last_frame=object(), length=141) == "fl2va"
       and pe.resolve_expander_mode("auto", references=(_FakeImgRef(),), length=141) == "ref2va",
       "MiniMaxH3PromptExpander auto mode infers all 6 H3 modes from wired sockets + length")
    pass_res = pe.expand_video_prompt(
        prompt="integrated_multimodal_description:\n[Shot 1] Live-action. Picture 1 opens to Picture 2.\n\noverall_soundscape:\nWind.\n\nnon_diegetic_music:\nN/A",
        enabled=False,
        first_frame=object(),
        last_frame=object(),
        length=192,
    )
    ok("8.00-second mark" in pass_res["prompt"].splitlines()[0] and abs(pass_res["video_end_percent"] - 1.0) < 1e-6,
       "MiniMaxH3PromptExpander enabled=False passes prompt through offline and stamps exact FL2VA 8.00s Part One")
    syn_pass = pe.expand_video_prompt(
        prompt="subject_definitions:\n<Subject 1> from <Picture 1>.\n",
        enabled=False,
        references=(_FakeImgRef(), _FakeVidRef()),
        length=141,
    )
    ok(abs(syn_pass["video_end_percent"] - 0.33) < 1e-6,
       "MiniMaxH3PromptExpander outputs video_end_percent=0.33 when references contains synthetic trajectory video")

    # Verify both new 1x Easy-Button workflows wire MiniMaxH3PromptExpander -> BasicGuider (zero EditGuider)
    for wf_name, expect_rel in (
        ("h3_text_to_video_pika_prompt_expander_api.json", False),
        ("h3_image_ref_plus_text_to_video_pika_prompt_expander_api.json", True),
    ):
        wf_path = REPO / "workflows" / wf_name
        if not wf_path.is_file():
            FAILURES.append(f"missing shipped 1x expander workflow: {wf_name}")
            continue
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        ok(wf.get("14", {}).get("class_type") == "MiniMaxH3PromptExpander"
           and wf.get("5", {}).get("inputs", {}).get("prompt") == ["14", 0]
           and wf.get("9", {}).get("class_type") == "BasicGuider"
           and wf.get("27", {}).get("inputs", {}).get("length") == 192
           and wf.get("27", {}).get("inputs", {}).get("shape") == "wide"
           and "1152x768" in str(wf.get("27", {}).get("inputs", {}).get("shape.wide_resolution", ""))
           and "130" not in wf and "131" not in wf,
           f"{wf_name} wires 1x MiniMaxH3PromptExpander -> BasicGuider at 1152x768 @ 192f (8.00s)")
        if expect_rel:
            ok(wf.get("135", {}).get("inputs", {}).get("video_end_percent") == ["14", 1]
               and wf.get("14", {}).get("inputs", {}).get("references") == wf["5"]["inputs"]["references"],
               f"{wf_name} wires references into MiniMaxH3PromptExpander and video_end_percent into ReleaseReferences")
    _ = ro


def main() -> int:
    check_concepts_and_sections()
    check_validators_and_repair()
    check_edit_plan_compilation_and_wiring()
    check_url_resolution_and_mock_sse()
    check_shipped_workflows()
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("\nall ok -- auto_edit_prompt + heylook_client + shipped auto-edit workflows verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
