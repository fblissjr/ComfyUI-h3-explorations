#!/usr/bin/env python
"""Fail when a prompt DOCUMENT drifts from the source that owns what it quotes.

## Renamed 2026-09-01, and why

This was `check_portable_standard.py` and guarded only the published extract.
That was **backwards**: the extract was checked and `docs/prompting.md` -- the
document the extract is derived FROM, and the one this repo calls its single
source of truth -- was guarded by nothing. Deliberately corrupting the manual's
FL2VA Part One template and its camera vocabulary was caught by no check in the
repo. Same class of quotation, same sources, so one check owns both.

## Why this is a check and not a promise

`docs/portable/h3_prompt_standard.html` is published as an artifact and handed
to readers outside this repo. It is, by construction, a **second copy** of rules
that are owned elsewhere -- and this repo's whole documented failure mode is a
second copy with nothing to invalidate it. A note saying "regenerate rather than
edit" is not invalidation. This is.

**It had already drifted before it was checked once**, which is the escaped
instance: the T2VA example on the page carried `non_diegetic_music: N/A` where
`docs/prompting.md` section 10.1 carries a real cue -- transcribed by hand and
changed in transcription, into the very habit the same page warns against. The
camera table had also lost half of seven motion types to abbreviation
(`Zoom In / Out` for the guide's `Zoom In / Zoom Out`), in the one section
readers copy from most.

## What is checkable, and what is deliberately not

Only what is DERIVABLE. The page's prose -- the explanations, the layer
assignments, the "why it works" notes -- is written by a person and cannot be
diffed against anything. This checks the parts that are quotations:

1. **The three Part One templates** against `BASE_ALIGNMENT`, which
   `preflight_graph` parses out of the vendor guide rather than retyping.
2. **The camera motion vocabulary** against the guide's own 4.3 table.
3. **Every worked example** against `docs/prompting.md` section 10, exactly,
   after whitespace normalisation -- so an example cannot be reworded, and
   cannot silently stop being one that grades clean.
4. **Quoted guide sentences** against the guide text.

A green run means the quotations still match their sources. It says nothing
about whether the prose around them is right.

## Comparison is whitespace-normalised, and that is deliberate

The page hard-wraps its `<pre>` blocks to a narrower column than the manual
does, so a literal comparison would fail on every example for a reason nobody
cares about. Collapsing runs of whitespace compares the words, which is the
thing that must not drift. It does mean a change purely in line breaking is
invisible here -- acceptable, since the model receives the prompt text and not
the page's wrapping.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "docs" / "portable" / "h3_prompt_standard.html"
MANUAL = REPO / "docs" / "prompting.md"
SNAPSHOTS = REPO / "docs" / "portable" / "snapshots.json"
GUIDE = REPO / "vendor_guides" / "base_en.md"

sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "workflows"))

import preflight_graph as pf  # noqa: E402

# Sentences the page quotes as vendor text. Each must appear in the guide.
QUOTED = [
    "Place the speaker's identifying phrase, ID, action, and delivery outside",
    "At the beginning of `[Shot 1]`, state the overall style and initial "
    "composition",
    "Use `N/A` when there is no non-diegetic music",
]


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def text_of(page: str) -> str:
    return flat(html.unescape(re.sub(r"<[^>]+>", " ", page)))


def manual_quotations(manual: str, guide: str) -> int:
    """The manual retypes two closed vocabularies. Grade both against source.

    `docs/prompting.md` restates the three Part One templates and base_en 4.3's
    motion table in prose, for readers. Those are the two things in it a reader
    COPIES rather than reads, so a typo there ships into a prompt -- and
    `preflight_graph` parses its own copy from the guide, so the two can
    disagree silently while every other check stays green. Found 2026-09-01 by
    mutation: nothing caught either.

    **Every instance is graded, not just one.** The manual prints the templates
    many times, resolved to different durations. A first version asked only
    whether SOME instance matched, which is green while a corrupted one sits
    beside a correct one -- and that is the likely shape of a real typo. So
    every line that looks like a Part One line must match its mode's template.
    """
    bad = 0
    # Any line opening the way a Part One line opens, whatever follows.
    CANDIDATE = re.compile(
        r"^(How the reference pictures align[^\n]*|"
        r"For the target video, at [^\n]*)$", re.M)
    from preflight_graph import BASE_ALIGNMENT
    shapes = {}
    for mode, template in BASE_ALIGNMENT.items():
        # TWO legal forms, and rejecting the second was a false positive on
        # this check's first run. The manual documents each template in its
        # UNRESOLVED form -- `Shot N`, `S.SS`, exactly as the guide prints it --
        # and then shows RESOLVED instances in the worked examples. Both are
        # correct; only a line matching neither is drift.
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
    missing = []
    for t in types:
        for part in (p.strip() for p in t.split("/")):
            # Word-bounded: `Pedestal Up` is a substring of `Pedestal Upward`,
            # so a plain `in` test passes a vocabulary that has been reworded.
            if not re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", manual):
                missing.append(t)
                break
    if missing:
        bad += 1
        print(f"  FAIL  manual's camera vocabulary altered or missing: "
              f"{', '.join(missing)}")
    else:
        print(f"  ok    manual quotes all {len(types)} camera motion types")
    return bad


def examples_still_grade(manual: str) -> int:
    """Section 10's examples must still PASS the grader, not merely exist.

    The examples are what "good" means here -- they are copied, and the
    portable standard quotes them. Checking that the page matches the manual
    keeps the two in step but says nothing about whether either is CORRECT: a
    bad example faithfully copied is still a bad example, and both would go
    green.

    Until 2026-09-01 the section claimed its examples "grade clean" and nothing
    verified it; the claim was true when checked by hand three times that day,
    and by hand is not a control. This runs the same grader
    `bench/grade_prompt_text.py` exposes, against a shipped graph of each
    mode at the duration the example's own heading names.
    """
    import grade_prompt_text as g
    section = manual.split("## 10. Worked examples", 1)[1].split("\n## 11.", 1)[0]
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
            # ref2va labels are graded against the donor's sockets, so pick a
            # donor wiring what the example declares rather than the default.
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


def audit_covers_catalogue() -> int:
    """Every generated scene name must resolve to a hand-written verdict.

    `prompt_audit.md` is keyed BY HAND to scene names `prompt_catalogue.md`
    generates from the graphs. Renaming a prompt constant silently orphans its
    verdict, and a scene added to the generator has no verdict at all -- which
    is exactly how the audit came to cover a minority of the catalogue before
    2026-09-01, unnoticed because both files looked fine on their own.
    """
    cat = REPO / "docs" / "prompt_catalogue.md"
    aud = REPO / "docs" / "prompt_audit.md"
    if not (cat.exists() and aud.exists()):
        print("  FAIL  catalogue or audit missing; coverage cannot be checked")
        return 1
    scenes = re.findall(r"^## (.+)$", cat.read_text(encoding="utf-8"), re.M)
    audit = aud.read_text(encoding="utf-8")
    # WORD-BOUNDED, not substring. `T2V_RAIL_LONG` is a substring of
    # `T2V_RAIL_LONGG`, so a plain `in` test passes a scene whose verdict was
    # renamed out from under it -- which is precisely the drift this is for.
    # Caught 2026-09-01 while red-proving: the proof did not go red, and the
    # check was the reason, not the proof.
    missing = [s for s in scenes
               if not re.search(r"(?<![\w:])" + re.escape(s) + r"(?![\w])", audit)]
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


def snapshots() -> int:
    """Dated snapshots are checked for STAYING PUT, never against the sources.

    A frozen record of what was shared on a date will fall behind the manual by
    design -- that is what makes it a record. Grading it against today's
    sources would go red for the one reason that is correct, which is the
    cry-wolf failure this repo refuses. So the only question asked of a
    snapshot is whether anyone has edited it since it was frozen.
    """
    if not SNAPSHOTS.exists():
        return 0
    recorded = json.loads(SNAPSHOTS.read_text(encoding="utf-8"))
    bad = 0
    for name, meta in sorted(recorded.items()):
        path = SNAPSHOTS.parent / name
        if not path.exists():
            print(f"  FAIL  snapshot {name} is recorded but missing")
            bad += 1
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != meta["sha256"]:
            print(f"  FAIL  snapshot {name} has been EDITED since it was "
                  f"frozen on {meta['frozen']}. A dated record that changes is "
                  f"not a record -- restore it, or freeze a new one.")
            bad += 1
        else:
            print(f"  ok    snapshot {name} unmodified since {meta['frozen']}")
    # A snapshot on disk that nothing records is the shape that rots: it looks
    # authoritative and nothing pins it.
    for path in sorted(SNAPSHOTS.parent.glob("2*_*.html")):
        if path.name not in recorded:
            print(f"  FAIL  {path.name} looks like a snapshot but is not in "
                  f"{SNAPSHOTS.name}; nothing pins it")
            bad += 1
    return bad


def main() -> int:
    for p in (PAGE, MANUAL, GUIDE):
        if not p.exists():
            print(f"FAIL  missing {p.relative_to(REPO)}; this check compares "
                  f"nothing without it and would report a misleading green")
            return 2
    page = PAGE.read_text(encoding="utf-8")
    manual = MANUAL.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    prose = text_of(page)
    fails = 0

    from preflight_graph import BASE_ALIGNMENT
    for mode, template in sorted(BASE_ALIGNMENT.items()):
        if flat(template) in prose:
            print(f"  ok    {mode} Part One template matches the guide")
        else:
            fails += 1
            print(f"  FAIL  {mode} Part One template on the page does not match "
                  f"the one parsed from the guide")

    types = re.findall(r"\| Motion type \| `([^`]+)`", guide)
    missing = [t for t in types if f"<code>{t}</code>" not in page]
    if missing:
        fails += 1
        print(f"  FAIL  camera motion types altered or missing: "
              f"{', '.join(missing)}")
    else:
        print(f"  ok    all {len(types)} camera motion types quoted verbatim")

    section = manual.split("## 10. Worked examples", 1)[1].split("\n## 11.", 1)[0]
    examples = {flat(b) for b in re.findall(r"```text\n(.*?)\n```", section, re.S)}
    blocks = [flat(html.unescape(b))
              for b in re.findall(r"<pre>(.*?)</pre>", page, re.S)]
    shown = [b for b in blocks
             if ("integrated_multimodal_description:" in b
                 or "detailed_description:" in b) and "..." not in b]
    if not shown:
        fails += 1
        print("  FAIL  no worked example found on the page; the extractor or "
              "the page changed shape, and a silent zero here would pass")
    for b in shown:
        if b in examples:
            print(f"  ok    example matches the manual: {b[:52]}...")
        else:
            fails += 1
            print(f"  FAIL  an example on the page is not in the manual's "
                  f"section 10 verbatim: {b[:60]}...")

    fails += manual_quotations(manual, guide)
    fails += examples_still_grade(manual)
    fails += audit_covers_catalogue()
    fails += snapshots()

    for q in QUOTED:
        if flat(q) in flat(guide):
            print(f"  ok    quoted guide sentence found: {q[:46]}...")
        else:
            fails += 1
            print(f"  FAIL  page quotes a sentence absent from the guide: {q}")

    print("")
    if fails:
        print(f"  {fails} drift(s) between a prompt document and its sources")
        return 1
    print("  ok    every prompt document agrees with the source it quotes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
