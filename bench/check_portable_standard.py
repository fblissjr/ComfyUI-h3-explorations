#!/usr/bin/env python
"""Fail when the portable prompt standard drifts from the sources it quotes.

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

import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "docs" / "portable" / "h3_prompt_standard.html"
MANUAL = REPO / "docs" / "prompting.md"
GUIDE = REPO / "vendor_guides" / "base_en.md"

sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "workflows"))

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

    for q in QUOTED:
        if flat(q) in flat(guide):
            print(f"  ok    quoted guide sentence found: {q[:46]}...")
        else:
            fails += 1
            print(f"  FAIL  page quotes a sentence absent from the guide: {q}")

    print("")
    if fails:
        print(f"  {fails} drift(s) between the portable standard and its sources")
        return 1
    print("  ok    every quotation on the portable standard matches its source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
