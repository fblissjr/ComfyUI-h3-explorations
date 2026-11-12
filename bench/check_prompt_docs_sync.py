#!/usr/bin/env python
"""Fail when a prompt DOCUMENT drifts from the source that owns what it quotes.

## Renamed 2026-09-01, and updated 2026-11-06 (`docs/wiki/system_prompts/`)

This was `check_portable_standard.py` and guarded only the published extract.
That was **backwards**: the extract was checked and `docs/prompting.md` -- the
document the extract is derived FROM, and the one this repo calls its single
source of truth -- was guarded by nothing. Deliberately corrupting the manual's
FL2VA Part One template and its camera vocabulary was caught by no check in the
repo. Same class of quotation, same sources, so one check owns both.

On 2026-11-06 `docs/portable/` was retired and the canonical system prompts
moved into `docs/wiki/system_prompts/` (`h3_system_prompt.md` and
`h3_control_vector_system_prompt.md`).

## What is checkable, and what is deliberately not

Only what is DERIVABLE. A document's prose -- the explanations, the layer
assignments, the "why it works" notes -- is written by a person and cannot be
diffed against anything. This checks the parts that are quotations:

1. **The two vendor guides**, and the MiniMax-H3-Image prompt files under
   `vendor_guides/h3_image/`, against `vendor_guides/sha256.json`; and the six
   standalone MiniMax-H3-Image prompts in `docs/wiki/system_prompts/h3_image/`
   against their structural pins, `H3_IMAGE_ASPECTS`, and zero-lore rules.
2. **The three Part One templates** against `BASE_ALIGNMENT`, which
   `preflight_graph` parses out of the vendor guide rather than retyping, and
   against `INDEPENDENT_ALIGNMENT` (transcribed by hand from `base_en.md`).
3. **The camera motion vocabulary** in the manual and `h3_system_prompt.md`
   against the guide's own 4.3 table.
4. **Every worked example** in `docs/prompting.md` section 10 re-graded through
   `preflight_graph.grade` at the duration its heading names.
5. **System prompts in `docs/wiki/system_prompts/`** against `BASE_ALIGNMENT`
   and `RULE_PINS`.
6. **Every scene in `docs/prompt_catalogue.md`** resolving to a verdict in
   `docs/prompt_audit.md`.
7. **Quoted guide sentences** against the guide text.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSTEM_PROMPTS_DIR = REPO / "docs" / "wiki" / "system_prompts"
SYSTEM_PROMPT = SYSTEM_PROMPTS_DIR / "h3_system_prompt.md"
PH3_IMAGE_SYSTEM_PROMPT = SYSTEM_PROMPTS_DIR / "h3_p_h3_image_system_prompt.md"
GENERAL_IMAGE_EDIT_SYSTEM_PROMPT = SYSTEM_PROMPTS_DIR / "general_image_edit_system_prompt.md"
CV_SYSTEM_PROMPT = SYSTEM_PROMPTS_DIR / "h3_control_vector_system_prompt.md"
MANUAL = REPO / "docs" / "prompting.md"
MANUAL_SHARDS = REPO / "docs" / "prompting"
GUIDE = REPO / "vendor_guides" / "base_en.md"


def read_manual() -> str:
    """Read docs/prompting.md and its modular section shards in section order."""
    parts = [MANUAL.read_text(encoding="utf-8")]
    if MANUAL_SHARDS.is_dir():
        for shard in sorted(MANUAL_SHARDS.glob("*.md")):
            parts.append(shard.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def section_10_slice(manual: str) -> str | None:
    """Return the text between '## 10. Worked examples' and '\\n## 11.', or None."""
    if "## 10. Worked examples" not in manual:
        return None
    tail = manual.split("## 10. Worked examples", 1)[1]
    if "\n## 11." not in tail:
        return None
    return tail.split("\n## 11.", 1)[0]

sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "workflows"))

import preflight_graph as pf  # noqa: E402

# Sentences quoted as vendor text. Each must appear in the guide.
QUOTED = [
    "Place the speaker's identifying phrase, ID, action, and delivery outside",
    "At the beginning of `[Shot 1]`, state the overall style and initial "
    "composition",
    "Use `N/A` when there is no non-diegetic music",
]


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# THE ONE PLACE A SECOND COPY IS THE POINT.
#
# Everything else in this file DERIVES its expected value from the guide,
# which kills transcription drift. But a derived expectation cannot contradict
# the thing it derives from: `preflight_graph._base_alignment_templates()`
# parses these three strings out of `vendor_guides/base_en.md` at import, and
# if that parse were subtly wrong -- a regex that clipped a trailing clause, a
# fence that moved -- every downstream comparison would agree with the wrong
# parse and stay green forever. Derivation cannot catch a MISREADING.
#
# So these three are transcribed BY HAND from base_en.md lines 19, 25 and 31 (the fenced strings under each "always uses"),
# read directly rather than copied from the parser's output, and asserted equal
# to what the parser produces. Two independent readings that can disagree.
INDEPENDENT_ALIGNMENT = {
    "I2VA": "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
    "FL2VA": "How the reference pictures align with the target video \u2014 "
             "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
             "target video; Picture 2 (from Shot N) aligns with the S.SS-second "
             "mark of the target video.",
    "L2VA": "How the reference pictures align with the target video \u2014 "
            "<Picture 1> (from [Shot N]) aligns with the S.SS-second mark of "
            "the target video.",
}


def parser_agrees_with_hand_reading() -> int:
    """Does the guide parser produce what a human reading the guide produces?"""
    from preflight_graph import BASE_ALIGNMENT
    bad = 0
    for mode, expected in sorted(INDEPENDENT_ALIGNMENT.items()):
        got = BASE_ALIGNMENT.get(mode)
        if got is None:
            print(f"  FAIL  the guide parser produced no {mode} template")
            bad += 1
        elif flat(got) != flat(expected):
            print(f"  FAIL  {mode}: the guide PARSER and a hand reading of the "
                  f"guide disagree. One of them is wrong -- read base_en.md and "
                  f"decide which.")
            print(f"        parser: {flat(got)[:100]}")
            print(f"        hand  : {flat(expected)[:100]}")
            bad += 1
    extra = sorted(set(BASE_ALIGNMENT) - set(INDEPENDENT_ALIGNMENT))
    if extra:
        print(f"  FAIL  the parser produced {', '.join(extra)}, which no hand "
              f"reading covers; add it here or explain why it is not checked")
        bad += 1
    if not bad:
        print(f"  ok    parser and hand reading agree on all "
              f"{len(INDEPENDENT_ALIGNMENT)} Part One templates")
    return bad


def guides_unchanged() -> int:
    """The two vendor guides must hash to what `vendor_guides/sha256.json` says."""
    import hashlib
    rec = REPO / "vendor_guides" / "sha256.json"
    if not rec.exists():
        print(f"  FAIL  {rec.relative_to(REPO)} is missing; the guides are "
              f"unpinned and every check that parses them would trust a "
              f"corrupted file")
        return 1
    recorded = json.loads(rec.read_text(encoding="utf-8"))
    bad = 0
    for name, want in sorted(recorded.items()):
        path = rec.parent / name
        if not path.exists():
            print(f"  FAIL  vendor guide {name} is recorded but missing")
            bad += 1
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            print(f"  FAIL  vendor guide {name} has CHANGED since it was "
                  f"pinned. Every check in this lane parses it, so they would "
                  f"all agree with the edit. If the change is deliberate, "
                  f"update sha256.json and re-read what moved.")
            bad += 1
    on_disk = {p.name for p in rec.parent.glob("*_en.md")}
    for extra in sorted(on_disk - set(recorded)):
        print(f"  FAIL  {extra} sits in vendor_guides/ unpinned")
        bad += 1
    if not bad:
        print(f"  ok    the vendor guides and the {len(recorded) - 2} other pinned vendor files match their hashes")
    return bad


#: Retired top-level vendor prompt copies (replaced by `docs/wiki/system_prompts/h3_image/`).
H3_IMAGE_RETIRED_TOP_LEVEL = (
    "h3_image_t2i_system_prompt.md",
    "h3_image_edit_system_prompt.md",
)

H3_IMAGE_SUITE_DIR = SYSTEM_PROMPTS_DIR / "h3_image"
H3_IMAGE_T2I_PINS = (
    "camera-relative orientation",
    "front-facing",
    "back-facing",
    "single frozen phase of motion",
    'double quotes (`"..."`)',
    "single continuous English caption paragraph",
)
H3_IMAGE_DESIGN_PINS = (
    "Pre-solve all content, math, formulas",
    "reading order",
    "zero vague fillers",
    "non-crossing",
    "shared object inventory",
    'double quotes (`"..."`)',
)
H3_IMAGE_EDIT_PINS = (
    "<Picture 1>",
    "<Subject 1>",
    "The target image",
    "<Subject N> is fully preserved.",
    "<Subject N> is partially preserved;",
    "<Subject N> supplies the",
    "<Subject N> serves as a weak",
    "single quotes `'...'",
    "BARE `<Subject N>` tokens only",
)
H3_IMAGE_SUITE_SPECS = (
    ("t2i_system_prompt.md", False, H3_IMAGE_T2I_PINS),
    ("t2i_aspect_ratio_system_prompt.md", True, H3_IMAGE_T2I_PINS),
    ("t2i_design_system_prompt.md", False, H3_IMAGE_DESIGN_PINS),
    ("t2i_design_aspect_ratio_system_prompt.md", True, H3_IMAGE_DESIGN_PINS),
    ("edit_system_prompt.md", False, H3_IMAGE_EDIT_PINS),
    ("edit_aspect_ratio_system_prompt.md", True, H3_IMAGE_EDIT_PINS),
)
H3_IMAGE_FORBIDDEN_SUBSTRINGS = (
    "```",
    "qwen",
    "minimax",
    "p-h3-image",
    "sglang",
    "vllm",
    "comfyui",
    "safetensors",
    "heylookitsanllm",
    "expanded_caption",
    "target_aspect_ratio",
    "docs/",
    "bench/",
    "workflows/",
    "vendor_guides/",
    "http://",
    "https://",
)


def h3_image_system_prompts() -> int:
    """Verify the standalone `docs/wiki/system_prompts/h3_image/` suite and retirement of the top-level vendor duplicates."""
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config

    bad = 0
    for retired in H3_IMAGE_RETIRED_TOP_LEVEL:
        if (SYSTEM_PROMPTS_DIR / retired).exists():
            bad += 1
            print(f"  FAIL  {retired} still exists in docs/wiki/system_prompts/; "
                  f"vendor files live in vendor_guides/h3_image/prompt/ and standalone prompts live in h3_image/")
    if not bad:
        print("  ok    retired duplicate vendor prompt copies in docs/wiki/system_prompts/ remain absent")

    readme = H3_IMAGE_SUITE_DIR / "README.md"
    if not readme.exists():
        print("  FAIL  docs/wiki/system_prompts/h3_image/README.md is missing")
        return bad + 1

    expected_names = {"README.md", *(name for name, _, _ in H3_IMAGE_SUITE_SPECS)}
    actual_names = {p.name for p in H3_IMAGE_SUITE_DIR.glob("*.md")}
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        bad += 1
        print(f"  FAIL  unexpected .md file(s) in docs/wiki/system_prompts/h3_image/ without check coverage: "
              f"{', '.join(unexpected)}")

    cjk_re = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
    for name, has_ratio, pins in H3_IMAGE_SUITE_SPECS:
        path = H3_IMAGE_SUITE_DIR / name
        if not path.exists():
            bad += 1
            print(f"  FAIL  missing docs/wiki/system_prompts/h3_image/{name}")
            continue
        raw = path.read_text(encoding="utf-8")
        norm = flat(raw)
        low = norm.lower()

        leaks = [tok for tok in H3_IMAGE_FORBIDDEN_SUBSTRINGS if tok in low]
        if cjk_re.search(raw):
            leaks.append("<CJK characters>")
        miss = [pin for pin in pins if flat(pin).lower() not in low]

        ratio_err = None
        if has_ratio:
            if "aspect_ratio:" not in raw:
                ratio_err = "missing 'aspect_ratio:' output line"
            else:
                missing_ratios = [r for r in h3_config.H3_IMAGE_ASPECTS if r not in raw]
                if missing_ratios:
                    ratio_err = f"missing H3_IMAGE_ASPECTS ratio(s): {', '.join(missing_ratios)}"
                elif name.startswith("edit_") and h3_config.H3_IMAGE_ASPECT_OF_PICTURE not in raw:
                    ratio_err = f"missing {h3_config.H3_IMAGE_ASPECT_OF_PICTURE!r}"
        elif "aspect_ratio:" in low:
            ratio_err = "plain prompt variant must not emit 'aspect_ratio:'"

        if leaks or miss or ratio_err:
            bad += 1
            if leaks:
                print(f"  FAIL  h3_image/{name} contains forbidden lore/fences/paths: {', '.join(repr(x) for x in leaks)}")
            if miss:
                print(f"  FAIL  h3_image/{name} missing pinned rule(s): {', '.join(repr(x) for x in miss)}")
            if ratio_err:
                print(f"  FAIL  h3_image/{name}: {ratio_err}")
        else:
            print(f"  ok    h3_image/{name} carries all {len(pins)} rule pin(s), ratio contract, and zero lore/fences")
    return bad


GENERAL_IMAGE_EDIT_PINS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "<Picture 1>",
    "<Subject 1>",
    "[keyframe completion]",
    "[reference generation]",
    "[keyframe completion + reference generation]",
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
    "[Shot 1]",
    'double quotes (`"..."`)',
)
GENERAL_IMAGE_EDIT_FORBIDDEN = (
    *H3_IMAGE_FORBIDDEN_SUBSTRINGS,
    "pika",
    "pruna",
    "textconnector",
    "overall_soundscape",
    "non_diegetic_music",
    "<audio",
    "<video",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
    "fully_copy",
    "partially_copy",
    "<d>",
    "</d>",
    "<|cutoff|>",
    "(s1)",
)


def general_image_edit_system_prompt(text: str) -> int:
    """Verify `docs/wiki/system_prompts/general_image_edit_system_prompt.md` carries the
    4-section visual `ref2va` schema pins and zero audio/video schema tokens, outer fences,
    repo paths, or internal model lore."""
    low = flat(text).lower()
    miss = [pin for pin in GENERAL_IMAGE_EDIT_PINS if flat(pin).lower() not in low]
    leaks = [tok for tok in GENERAL_IMAGE_EDIT_FORBIDDEN if tok in low]
    if re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text):
        leaks.append("<CJK characters>")
    if miss or leaks:
        if miss:
            print(f"  FAIL  general_image_edit_system_prompt.md missing pinned rule(s): "
                  f"{', '.join(repr(x) for x in miss)}")
        if leaks:
            print(f"  FAIL  general_image_edit_system_prompt.md contains forbidden audio/video/lore/fences: "
                  f"{', '.join(repr(x) for x in leaks)}")
        return 1
    print(f"  ok    general_image_edit_system_prompt.md carries all "
          f"{len(GENERAL_IMAGE_EDIT_PINS)} visual ref2va pin(s) and zero audio/video/lore/fences")
    return 0


def manual_quotations(manual: str, guide: str, system: str) -> int:
    """The manual and `h3_system_prompt.md` retype closed vocabularies. Grade against source."""
    bad = 0
    CANDIDATE = re.compile(
        r"^(How the reference pictures align[^\n]*|"
        r"For the target video, at [^\n]*)$", re.M)
    from preflight_graph import BASE_ALIGNMENT
    shapes = {}
    for mode, template in BASE_ALIGNMENT.items():
        parts = re.split(r"(Shot N|S\.SS)", flat(template))
        shapes[mode] = "".join(
            r"Shot (?:N|\d+)" if part == "Shot N"
            else r"(?:S\.SS|\d+\.\d\d)" if part == "S.SS"
            else re.escape(part)
            for part in parts)
    seen = {m: 0 for m in shapes}
    for line in CANDIDATE.findall(manual):
        f = flat(line)
        hit = [m for m, pat in shapes.items() if re.fullmatch(pat, f)]
        if hit:
            seen[hit[0]] += 1
        else:
            bad += 1
            print(f"  FAIL  manual has a Part One line matching no guide "
                  f"template: {f[:88]}...")
    for mode, n in sorted(seen.items()):
        if n:
            print(f"  ok    manual's {mode} Part One: {n} instance(s), all "
                  f"matching the guide")
        else:
            bad += 1
            print(f"  FAIL  manual shows no {mode} Part One line at all; it is "
                  f"supposed to document every mode")

    types = re.findall(r"\| Motion type \| `([^`]+)`", guide)
    for label, doc_text in (("manual", manual), ("h3_system_prompt.md", flat(system))):
        missing = []
        for t in types:
            for part in (p.strip() for p in t.split("/")):
                if not re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", doc_text):
                    missing.append(t)
                    break
        if missing:
            bad += 1
            print(f"  FAIL  {label}'s camera vocabulary altered or missing: "
                  f"{', '.join(missing)}")
        else:
            print(f"  ok    {label} quotes all {len(types)} camera motion types")
    return bad


def examples_still_grade(manual: str) -> int:
    """Section 10's examples must still PASS the grader, not merely exist."""
    import grade_prompt_text as g
    section = section_10_slice(manual)
    if section is None:
        print("  FAIL  section 10 ('## 10. Worked examples' .. '## 11.') missing "
              "from manual / docs/prompting/ shards")
        return 1
    mode = cur = None
    rows = []
    pattern = r"^(###|####) (10[\.\d]*) ([^\n]*)$|```text\n(.*?)\n```"
    for m in re.finditer(pattern, section, re.S | re.M):
        if m.group(1):
            head = m.group(3)
            named = re.match(r"(\w+) —", head)
            if m.group(1) == "###" and named:
                mode = named.group(1).lower()
            frames = re.search(r"(\d+) frames", head)
            cur = (m.group(2), mode, int(frames.group(1)) if frames else None)
        elif cur:
            rows.append((*cur, m.group(4)))
    if not rows:
        print("  FAIL  no worked example parsed from section 10; the section "
              "changed shape and a silent zero here would read as a pass")
        return 1
    bad = 0
    for num, md, frames, body in rows:
        if md is None or frames is None:
            bad += 1
            print(f"  FAIL  example {num} has no mode or no frame count in its "
                  f"heading, so it cannot be graded at a duration")
            continue
        try:
            like = "h3_ref_image_audio_api" if md == "ref2va" else None
            path, nid = g.pick(md if like is None else None, like)
            graph = json.loads(path.read_text(encoding="utf-8"))
            node = graph[nid]
            node["inputs"]["prompt"] = body.strip()
            node["inputs"]["length"] = frames
            findings = pf.grade(node, graph, path.stem)
        except Exception as exc:
            bad += 1
            print(f"  FAIL  example {num} could not be graded: {exc}")
            continue
        fails = [f for f in findings if f[0] == "FAIL"]
        if fails:
            bad += 1
            print(f"  FAIL  example {num} ({md}, {frames}f) no longer grades "
                  f"clean: {fails[0][1]}")
    if not bad:
        print(f"  ok    all {len(rows)} section-10 example(s) still grade clean")
    return bad


# Backward-compatible symbol alias cited in bench/results/2026-10-01_prompting_docs_audit.md
page_examples_grade = examples_still_grade


CV_RULE_PINS = (
    "<d>[English]",
    "10/10",
    "Zero Verbatim Span Cloning Across Pairs",
    "### Hold Pairs",
    "slowly",
    "gradually",
    "begins to",
    "non_diegetic_music: N/A",
)


PH3_T2I_RULE_PINS = (
    "OUTPUT FORMAT",
    "STRUCTURE AND ORDER",
    "CONSTRAINTS",
    "anchor the main subject's scale",
)
PH3_EDIT_RULE_PINS = (
    "OUTPUT FORMAT",
    "TEMPLATE",
    "Edit <Picture 1>.",
    "Keep <specific unchanged elements> unchanged.",
)
PH3_EDIT_STRUCTURED_RULE_PINS = (
    "subject_definitions:",
    "summary:",
    "[keyframe completion]",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
PH3_FORBIDDEN_LORE = (
    "qwen",
    "minimax",
    "p-h3-image",
    "textconnector",
    "comfyui",
    "safetensors",
)


def cv_system_prompt_and_portable_absent(cv_text: str, ph3_image_text: str) -> int:
    """Verify `h3_control_vector_system_prompt.md` and the three copyable fenced blocks
    in `h3_p_h3_image_system_prompt.md` carry their mandatory authoring pins, contain no
    internal model lore, and that `docs/portable/` remains retired."""
    bad = 0
    portable_dir = REPO / "docs" / "portable"
    if portable_dir.exists():
        bad += 1
        print("  FAIL  docs/portable/ exists on disk; it was retired on 2026-11-06 "
              "in favor of docs/wiki/system_prompts/")
    else:
        print("  ok    docs/portable/ is absent (retired in favor of docs/wiki/system_prompts/)")

    missing = [pin for pin in CV_RULE_PINS if flat(pin).lower() not in flat(cv_text).lower()]
    if missing:
        bad += 1
        print(f"  FAIL  h3_control_vector_system_prompt.md missing pinned rule(s): "
              f"{', '.join(repr(m) for m in missing)}")
    else:
        print(f"  ok    h3_control_vector_system_prompt.md carries all "
              f"{len(CV_RULE_PINS)} control-vector authoring rule pin(s)")

    blocks = re.findall(r"```text\n(.*?)\n```", ph3_image_text, re.S)
    if len(blocks) != 3:
        bad += 1
        print(f"  FAIL  h3_p_h3_image_system_prompt.md has {len(blocks)} ```text fenced block(s), expected 3")
        return bad
    for label, text, pins in zip(
        ("T2I block (§1)", "Edit direct block (§2)", "Edit structured block (§3)"),
        blocks,
        (PH3_T2I_RULE_PINS, PH3_EDIT_RULE_PINS, PH3_EDIT_STRUCTURED_RULE_PINS),
    ):
        low = flat(text).lower()
        miss = [pin for pin in pins if flat(pin).lower() not in low]
        lore = [w for w in PH3_FORBIDDEN_LORE if w in low]
        if miss or lore:
            bad += 1
            if miss:
                print(f"  FAIL  h3_p_h3_image_system_prompt.md {label} missing pinned rule(s): "
                      f"{', '.join(repr(m) for m in miss)}")
            if lore:
                print(f"  FAIL  h3_p_h3_image_system_prompt.md {label} leaks internal model lore: "
                      f"{', '.join(repr(w) for w in lore)}")
        else:
            print(f"  ok    h3_p_h3_image_system_prompt.md {label} carries all {len(pins)} rule pin(s) and zero internal model lore")
    return bad


def derived_system_prompts(guide: str) -> int:
    """Every system prompt in `docs/wiki/system_prompts/` that quotes the Part One
    templates must quote all three verbatim from the guide."""
    from preflight_graph import BASE_ALIGNMENT
    found = sorted(
        p for p in SYSTEM_PROMPTS_DIR.glob("*.md")
        if p.name != "README.md"
    )
    if not found:
        print("  FAIL  no system prompts found in docs/wiki/system_prompts/")
        return 1
    bad = 0
    for path in found:
        text = flat(path.read_text(encoding="utf-8"))
        quoted = [m for m, t in BASE_ALIGNMENT.items() if flat(t) in text]
        if not quoted:
            print(f"  note  {path.name} quotes no Part One template; not graded")
            continue
        missing = [m for m in BASE_ALIGNMENT if m not in quoted]
        if missing:
            bad += 1
            print(f"  FAIL  {path.name} quotes {', '.join(quoted)} verbatim but "
                  f"not {', '.join(missing)} -- a partial set is how a mode gets "
                  f"a reworded template")
        else:
            print(f"  ok    {path.name} quotes all three Part One templates "
                  f"verbatim from the guide")
    return bad


def audit_covers_catalogue() -> int:
    """Every generated scene name must resolve to a hand-written verdict."""
    cat = REPO / "docs" / "prompt_catalogue.md"
    aud = REPO / "docs" / "prompt_audit.md"
    if not (cat.exists() and aud.exists()):
        print("  FAIL  catalogue or audit missing; coverage cannot be checked")
        return 1
    scenes = re.findall(r"^## (.+)$", cat.read_text(encoding="utf-8"), re.M)
    audit = aud.read_text(encoding="utf-8")
    VERDICT = re.compile(
        r"\b(keep|revise|rewrite|discard|rewritten|revised|closed|withdrawn)\b",
        re.I)
    missing = []
    for s in scenes:
        mentions = [ln for ln in audit.splitlines()
                    if re.search(r"(?<![\w:])" + re.escape(s) + r"(?![\w])", ln)]
        if not any(VERDICT.search(ln) for ln in mentions):
            missing.append(s)
    if missing:
        print(f"  FAIL  {len(missing)} catalogue scene(s) have no verdict in "
              f"prompt_audit.md:")
        for s in missing[:8]:
            print(f"        {s}")
        if len(missing) > 8:
            print(f"        ... and {len(missing) - 8} more")
        return 1
    print(f"  ok    all {len(scenes)} catalogue scene(s) resolve to a verdict")
    return 0


# THE RULES A COPY CAN INVERT WITHOUT GOING RED.
RULE_PINS = [
    ("no timestamp on a shot header",
     {"manual": "shot headers carry no timestamps",
      "system": "no shot header carries a timestamp"}),
    ("that rule is the owner's, dated",
     {"manual": "owner rule since 2026-09-18",
      "system": "[owner, 2026-09-18]"}),
    ("a compound id is a line said in unison",
     {"manual": "say the line in unison",
      "system": "speak the line in unison"}),
    ("a character who never vocalises gets no id",
     {"manual": "characters who never vocalise receive no speaker id",
      "system": "characters who never vocalise get no id"}),
    ("the silent-character phrase is for one who never vocalises in the clip",
     {"manual": "never vocalises in the clip",
      "system": "never vocalises in the clip"}),
    ("the mouth-closing cue is positional",
     {"manual": "the pattern is positional",
      "system": "do not do this when the line is the last thing in the shot"}),
    ("N/A is a decision about the scene, not a default",
     {"manual": "is a decision about that scene, not a default",
      "system": "that is a judgement about this video, not a habit"}),
    ("ordinary cinematography wording is acceptable (the owner's ruling)",
     {"manual": "camera wording that aligns with common cinematography is acceptable",
      "system": "ordinary cinematography vocabulary outside that list is acceptable"}),
    ("a listener takes no speaker id",
     {"manual": "a listener never takes a speaker id",
      "system": "a listener never takes a speaker id"}),
    ("every action has an agent",
     {"manual": "an action with no agent",
      "system": "give every action an agent"}),
    ("pin what the scene turns on and leave the rest to the model",
     {"manual": "pick the most important ones and let the model guide the rest",
      "system": "pin what the scene turns on"}),
    ("who is on screen and how long the script runs are not left open",
     {"manual": "who is on screen, and how long the script runs",
      "system": "do not leave open who is on screen or how long the script runs"}),
]


def pin_failures(texts: dict[str, str], pins=RULE_PINS) -> list[str]:
    """Which pinned sentences are missing from which document."""
    low = {k: flat(v).lower() for k, v in texts.items()}
    out = []
    for label, phrases in pins:
        for doc, phrase in phrases.items():
            if doc not in low:
                continue
            if flat(phrase).lower() not in low[doc]:
                out.append(f"{label}: the {doc} no longer says {phrase!r}")
    return out


def rule_pins(manual: str, system: str) -> int:
    missing = pin_failures({"manual": manual, "system": system})
    for m in missing:
        print(f"  FAIL  pinned rule: {m}")
    if not missing:
        print(f"  ok    {len(RULE_PINS)} pinned rule sentence(s) present in the manual and system prompt")
    return len(missing)


def main() -> int:
    for p in (SYSTEM_PROMPT, PH3_IMAGE_SYSTEM_PROMPT, GENERAL_IMAGE_EDIT_SYSTEM_PROMPT, CV_SYSTEM_PROMPT, MANUAL, GUIDE):
        if not p.exists():
            print(f"FAIL  missing {p.relative_to(REPO)}; this check compares "
                  f"nothing without it and would report a misleading green")
            return 2
    if not MANUAL_SHARDS.is_dir() or not list(MANUAL_SHARDS.glob("*.md")):
        print(f"FAIL  missing or empty {MANUAL_SHARDS.relative_to(REPO)}/; "
              f"sections 5.8-16 live in modular shards under docs/prompting/")
        return 2
    manual = read_manual()
    guide = GUIDE.read_text(encoding="utf-8")
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    ph3_image_system = PH3_IMAGE_SYSTEM_PROMPT.read_text(encoding="utf-8")
    general_image_edit_system = GENERAL_IMAGE_EDIT_SYSTEM_PROMPT.read_text(encoding="utf-8")
    cv_system = CV_SYSTEM_PROMPT.read_text(encoding="utf-8")
    fails = 0

    fails += guides_unchanged()
    fails += h3_image_system_prompts()
    fails += general_image_edit_system_prompt(general_image_edit_system)
    fails += parser_agrees_with_hand_reading()
    fails += manual_quotations(manual, guide, system)
    fails += examples_still_grade(manual)
    fails += derived_system_prompts(guide)
    fails += cv_system_prompt_and_portable_absent(cv_system, ph3_image_system)
    fails += rule_pins(manual, system)
    fails += audit_covers_catalogue()

    for q in QUOTED:
        if flat(q) in flat(guide):
            print(f"  ok    quoted guide sentence found: {q[:46]}...")
        else:
            fails += 1
            print(f"  FAIL  quoted sentence absent from the guide: {q}")

    print("")
    if fails:
        print(f"  {fails} drift(s) between a prompt document and its sources")
        return 1
    print("  ok    every prompt document agrees with the source it quotes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
