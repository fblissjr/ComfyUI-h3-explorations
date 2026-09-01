#!/usr/bin/env python
"""Report every prompt FEATURE where the vendor is unanimous and we diverge.

## The class of defect this exists to find

Two misalignments were found by hand on 2026-09-01, and neither was reachable
by any check here:

- **Shot line breaks.** Every vendor multi-shot specimen runs shots inline in
  one paragraph. Five of our six multi-shot t2va prompts break a line per shot.
- **Mouth closing.** The vendor closes a mouth at the end of every dialogue
  line. Our two dialogue-heaviest prompts carry almost none.

**Neither is a stated guide rule, so neither `preflight_graph.py` nor
`check_prompt_guide_conformance.py` could ever have caught them** -- that file
correctly refuses to assert anything the guide does not state, and preflight
encodes the stated rules. The gap is the whole space of things the vendor DOES
consistently and never says. Nothing was looking there.

That is the escaped instance this file cites (CLAUDE.md: name one before
building a new instrument). It is a REPORT, not a gate: it cannot know whether
a divergence is a defect, a deliberate house choice, or noise from a corpus of
four.

## Why "unanimous" is the bar, and why n is printed everywhere

A feature is only interesting when the vendor never varies it. Anything the
vendor itself is split on tells us nothing about what the model was trained on.
**But the vendor corpus is tiny** -- single digits, and for multi-shot base-mode
prompts it is ONE specimen. A unanimous n=1 is a coincidence with a p-value, not
a rule, so every row prints its n and the reader does the discounting. Rows are
sorted by n descending for exactly that reason.

## Base and reference formats are never pooled

They genuinely differ: ref-en STATES one line per item for
`subject_definitions` and `retention_analysis`, while base-mode shots run
inline. Pooling them manufactures a fake divergence in both directions. This
repo's standing warning is that ref2va takes OPPOSITE corrections to the base
three, so the two corpora are reported separately and never compared.

## Mode is read from conditions, never from a name

`reproducible-768p-fl2va-request.sh` carries `"task": "fl2va"` and is an I2VA
request: one condition at `frame_index` 0, and base-en's I2VA Part One string.
`task` names the CHECKPOINT (the fl2va partition covers t2va/i2va/fl2va/l2va),
not the request. Classify by conditions; a filename and a task field are both
wrong here.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))

GUIDES = REPO / "internal" / "official_prompt_guides"
PAYLOADS = REPO / "coderef" / "MiniMax-H3" / "scripts" / "readme"

MAIN_BASE = "integrated_multimodal_description"
MAIN_REF = "detailed_description"

CLOSE_CUE = re.compile(
    r"lips (?:close|closed|settle|meet|remain)|jaw (?:stops|ceases)|"
    r"ceases speaking|closes (?:her|his|their|the) (?:lips|mouth)|mouth clos",
    re.I)
CAMERA = re.compile(
    r"camera (?:pushes|pulls|pans|trucks|tilts|holds|shakes|arcs|rolls|moves)|"
    r"static shot|tracking shot|zoom(?:s)? (?:in|out)|pedestal", re.I)


def shots_of(body: str) -> list[str]:
    """Each shot's text, split at headers rather than at newlines."""
    parts = re.split(r"(?=\[Shot \d+\])", body)
    return [p for p in parts if p.strip().startswith("[Shot")]


def features(prompt: str, main_field: str) -> dict[str, object]:
    """Mechanical features only. Nothing here is a judgement."""
    m = re.search(re.escape(main_field) + r":(.*?)(?=\n[a-z_]+:|\n```|\Z)",
                  prompt, re.S)
    body = m.group(1) if m else ""
    shots = shots_of(body)
    f: dict[str, object] = {}

    if len(shots) > 1:
        # Do the headers share one line? Newlines BETWEEN shots is the question.
        f["multi_shot_inline"] = len(
            [ln for ln in body.strip().split("\n") if ln.strip()]) == 1
    if shots:
        f["shot1_has_timestamp"] = bool(re.match(r"\[Shot 1\]\s*At \d", shots[0]))
        f["every_shot_names_camera"] = all(CAMERA.search(s) for s in shots)
        # ONLY over shots that actually carry dialogue. Taking the max across
        # all shots mixes in 0 from silent shots, which made the vendor look
        # SPLIT on {0, 1} and suppressed the row -- hiding the single
        # strongest divergence in the corpus (2026-09-01).
        speaking = [s for s in shots if "<d>" in s]
        if speaking:
            f["max_d_per_speaking_shot"] = max(s.count("<d>") for s in speaking)

    d_total = prompt.count("<d>")
    if d_total:
        # POSITIONAL, not per-line. The raw cue rate is the wrong statistic:
        # every corpus checked closes a mouth when the shot CONTINUES past the
        # line and never when the line ENDS the shot, so a per-line ratio
        # reports a corpus that follows the rule perfectly as ~50% compliant.
        # Cross-tab established by a peer session over the dagthomas corpus
        # (0 cues on 16 shot-final lines, 17/24 when the shot continues) and
        # reproduced on ours (0/1 and 32/44).
        fin_cued = cont_uncued = 0
        for shot in shots:
            for m in re.finditer(r"</d>", shot):
                tail = shot[m.end():]
                cued = bool(CLOSE_CUE.search(tail[:200]))
                if not tail.strip():
                    fin_cued += cued
                else:
                    cont_uncued += not cued
        f["cues_after_a_shot_final_line"] = bool(fin_cued)
        f["ever_skips_cue_mid_shot"] = bool(cont_uncued)
        f["every_d_has_language_tag"] = (
            len(re.findall(r"<d>\s*\[[A-Z][a-z]+\]", prompt)) == d_total)
        f["speaker_ids_used"] = bool(re.search(r"\(S\d", prompt))

    sound = re.search(r"overall_soundscape:(.*?)(?=\n[a-z_]+:|\n```|\Z)",
                       prompt, re.S)
    if sound:
        f["dialogue_in_soundscape"] = "<d>" in sound.group(1)
    music = re.search(r"non_diegetic_music:\s*(.*?)(?=\n[a-z_]+:|\n```|\Z)",
                       prompt, re.S)
    if music:
        f["music_is_na"] = music.group(1).strip() == "N/A"
    return f


def vendor_specimens() -> tuple[list[dict], list[dict]]:
    """(base-format, ref-format) vendor prompts, from guides and payloads."""
    base, ref = [], []
    for path, field, bucket in (
            (sorted(GUIDES.glob("*base_en.md")), MAIN_BASE, base),
            (sorted(GUIDES.glob("*ref_en.md")), MAIN_REF, ref)):
        if not path:
            continue
        text = path[0].read_text(encoding="utf-8")
        for blk in re.split(r"(?=" + re.escape(field) + r":)", text)[1:]:
            bucket.append({"src": f"{path[0].name}#{len(bucket)+1}",
                           **features(blk, field)})
    for p in sorted(PAYLOADS.glob("*.sh")) if PAYLOADS.exists() else []:
        raw = p.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'"prompt":\s*"(.*?)"\s*,\s*\n', raw, re.S)
        if not m:
            continue
        prompt = m.group(1).encode().decode("unicode_escape", "replace")
        field = MAIN_REF if MAIN_REF + ":" in prompt else MAIN_BASE
        (ref if field == MAIN_REF else base).append(
            {"src": p.name, **features(prompt, field)})
    return base, ref


def ours() -> tuple[list[dict], list[dict]]:
    import h3_config
    base, ref = [], []
    seen: set[str] = set()
    for path in h3_config.graph_paths(REPO / "workflows"):
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(graph, dict) or isinstance(graph.get("nodes"), list):
            continue
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            prompt = node.get("inputs", {}).get("prompt")
            if not isinstance(prompt, str) or prompt in seen:
                continue
            if MAIN_BASE not in prompt and MAIN_REF not in prompt:
                continue
            seen.add(prompt)
            field = MAIN_REF if MAIN_REF + ":" in prompt else MAIN_BASE
            (ref if field == MAIN_REF else base).append(
                {"src": path.stem, **features(prompt, field)})
    return base, ref


def compare(label: str, vendor: list[dict], mine: list[dict]) -> None:
    print(f"\n=== {label} — vendor n={len(vendor)}, ours n={len(mine)} ===")
    keys = {k for row in vendor + mine for k in row if k != "src"}
    rows = []
    for key in sorted(keys):
        vv = [r[key] for r in vendor if key in r]
        mv = [r[key] for r in mine if key in r]
        if not vv or not mv:
            continue
        if len(set(map(str, vv))) != 1:
            continue                       # vendor is split: says nothing
        want = vv[0]
        off = [r["src"] for r in mine if key in r and r[key] != want]
        if off:
            rows.append((len(vv), key, want, off, len(mv)))
    if not rows:
        print("  no divergence on any feature where the vendor is unanimous")
        return
    for n, key, want, off, tot in sorted(rows, key=lambda r: -r[0]):
        strength = "WEAK — one specimen" if n == 1 else f"vendor unanimous over {n}"
        print(f"\n  {key}: vendor always {want!r}   [{strength}]")
        print(f"      {len(off)}/{tot} of ours differ:")
        for s in off[:8]:
            print(f"        {s}")
        if len(off) > 8:
            print(f"        ... and {len(off)-8} more")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    if not GUIDES.exists():
        print(f"FAIL  no guides at {GUIDES.relative_to(REPO)}; nothing to "
              f"compare against. This is a loud failure rather than an empty "
              f"green report on purpose.")
        return 2
    vb, vr = vendor_specimens()
    ob, orf = ours()
    print("Features where the vendor never varies and we do.")
    print("A REPORT, not a gate: a divergence may be a defect, a deliberate")
    print("house choice, or noise. Check n before believing any row.")
    compare("BASE format (t2va / i2va / fl2va / l2va)", vb, ob)
    compare("REFERENCE format (ref2va)", vr, orf)
    print("\nBase and reference are never pooled: they genuinely differ, and")
    print("ref2va has taken opposite corrections to the base three before.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
