#!/usr/bin/env python
"""Camera motion in every shipped prompt comes from base_en 4.3's closed sets.

## The escaped instance that earns this

`docs/checks.md` recorded this rule as decidable and unbuilt for weeks. On
2026-08-28 it acquired its instance: the shipped market t2v prompt -- the
DEFAULT, carried by seventeen graph files -- rendered badly and was found by
reading the guide against it BY HAND. Every gate passed it. It carried
`tracks left` (which conflates 4.3's `Truck Left` row with its separate
`Tracking Shot` row), `at medium amplitude and moderate speed` (neither value
is in either modifier set), and `whip pan` (in no row at all).

## Why the split, and which half this is

`docs/checks.md` also states the split correctly and this file honours it.

**The modifier axes are DECIDABLE and are gated.** Amplitude and speed are
closed at two phrases each. Any `with <word> amplitude` or `at <word> speed`
outside them is out-of-table by construction -- no judgement, no list of bad
terms to maintain. This is where `at medium amplitude and moderate speed`
would have died.

**Motion type is NOT decidable and is REPORTED.** Proving a phrase is in
vocabulary needs a parser for English; a denylist of known-bad terms only ever
catches what somebody already thought of. So known-bad terms warn, and this
file does not pretend to prove the absence of the rest.

## The allowlist, and where it comes from

`VOCAB` is vendored rather than parsed, because the guide lives in `internal/`
which is gitignored -- a check that needs it cannot run on a fresh checkout.
But a vendored copy of somebody else's table is exactly the drift this repo
keeps paying for, so **when the guide IS present it is parsed and the vendored
copy is graded against it**. That case fails on any divergence, which makes the
constant a cache rather than a second source.

A third encoding exists and agrees: a sibling project of the owner's
independently encoded the same table from the same guide, verified
spelling-and-casing on 2026-08-28. Two independent derivations agreeing is why
this is a cache worth trusting between guide checks.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
GUIDE = REPO / "internal" / "official_prompt_guides" / \
    "minimax-h3-official-VIDEO_PROMPT_WRITING_GUIDE_base_en.md"

CONDITIONERS = ("MiniMaxH3Conditioning", "MiniMaxH3ReferenceConditioning",
                "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")

#: base_en 4.3, verbatim. Cross-checked against the guide below when present.
MOTION = (
    "Zoom In", "Zoom Out", "Push In", "Pull Out", "Pan Left", "Pan Right",
    "Truck Left", "Truck Right", "Tilt Up", "Tilt Down", "Pedestal Up",
    "Pedestal Down", "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV", "Roll Clockwise",
    "Roll Counterclockwise",
)
AMPLITUDE = ("with small amplitude", "with large amplitude")
SPEED = ("at slow speed", "at fast speed")

#: Known-bad motion phrases. A DENYLIST, and it only catches what somebody has
#: already been bitten by -- every entry here is a phrase that actually shipped
#: or was proposed in this repo. It is not a proof of absence and must never be
#: described as one.
DENIED = {
    "whip pan": "in no 4.3 row; also usually re-describes a cut the shot header made",
    "tracks left": "conflates `Truck Left` with the separate `Tracking Shot` row",
    "tracks right": "conflates `Truck Right` with the separate `Tracking Shot` row",
    "dolly": "not a 4.3 row; `Push In` / `Pull Out` or `Truck` is the vocabulary",
    "drifts a few degrees": "not a 4.3 row; `Shake Slightly` is the row that means it",
    "crash zoom": "in no 4.3 row",
    "snap zoom": "in no 4.3 row",
}

# Match the MODIFIER WORD, not the preposition. The first version of this
# anchored on `with ... amplitude` and `at ... speed`, and the red proof showed
# it could not catch its own motivating instance: the shipped defect read
# `at medium amplitude and moderate speed`, where the amplitude carries `at`
# rather than `with`, and `speed` is separated from its `at` by four words. A
# check that misses the escaped instance it was written for is worse than none,
# because the green reads as coverage.
AMP_RE = re.compile(r"\b([a-z\-]+)\s+amplitude\b", re.I)
SPD_RE = re.compile(r"\b([a-z\-]+)\s+speed\b", re.I)


def prompts() -> dict[str, set[str]]:
    import h3_config
    out: dict[str, set[str]] = {}
    for path in h3_config.graph_paths(REPO / "workflows", include_bench=True):
        graph = json.loads(Path(path).read_text(encoding="utf-8"))
        for node in graph.values() if isinstance(graph, dict) else []:
            if not isinstance(node, dict):
                continue
            if node.get("class_type") in CONDITIONERS:
                p = (node.get("inputs") or {}).get("prompt")
                if isinstance(p, str) and p.strip():
                    out.setdefault(p, set()).add(Path(path).stem)
    return out


def case_vocab_matches_guide() -> list[str]:
    """The vendored table IS the guide's, when the guide is on disk."""
    if not GUIDE.exists():
        print("  skip  vocab_matches_guide  guide not on disk (internal/ is "
              "gitignored); the vendored table is unverified on this checkout")
        return []
    text = GUIDE.read_text(encoding="utf-8")
    found = set()
    for m in re.finditer(r"\|\s*Motion type\s*\|\s*`([^`]+)`\s*\|", text):
        for part in m.group(1).split(" / "):
            found.add(part.strip())
    fails = []
    missing = found - set(MOTION)
    extra = set(MOTION) - found
    if missing:
        fails.append(f"  FAIL  vocab_matches_guide  guide has motion types this "
                     f"file does not: {sorted(missing)}")
    if extra:
        fails.append(f"  FAIL  vocab_matches_guide  this file has motion types "
                     f"the guide does not: {sorted(extra)}")
    if not fails:
        print(f"  ok    vocab_matches_guide  {len(found)} motion type(s) agree "
              "with base_en 4.3")
    return fails


def case_modifiers_are_in_set(corpus, quiet: bool = False) -> list[str]:
    """DECIDABLE: amplitude and speed are closed at two phrases each."""
    fails = []
    ok_amp = {a.split()[1] for a in AMPLITUDE}
    ok_spd = {s.split()[1] for s in SPEED}
    for text, graphs in sorted(corpus.items(), key=lambda kv: sorted(kv[1])[0]):
        where = sorted(graphs)[0]
        for word in AMP_RE.findall(text):
            if word.lower() not in ok_amp:
                fails.append(
                    f"  FAIL  modifiers_in_set  {where}: `with {word} amplitude` "
                    f"-- base_en 4.3 closes amplitude at {list(AMPLITUDE)}, and "
                    "medium is written by OMITTING the phrase")
        for word in SPD_RE.findall(text):
            if word.lower() not in ok_spd:
                fails.append(
                    f"  FAIL  modifiers_in_set  {where}: `at {word} speed` -- "
                    f"base_en 4.3 closes speed at {list(SPEED)}, and normal is "
                    "written by OMITTING the phrase")
    if not fails and not quiet:
        print(f"  ok    modifiers_in_set    every amplitude and speed phrase "
              f"across {len(corpus)} prompt(s) is in 4.3's closed set")
    return fails


def case_quiet_on_guide_examples() -> list[str]:
    """CRY-WOLF: the rules stay silent on the guide's OWN prose.

    Suggested by the sibling project, whose suite asserts its validator emits
    an empty diagnostic list on the guides' worked examples. The idea is better
    than either of our encodings of the table, because the corpus is the
    vendor's rather than anybody's reading of it: base 4.3 writes motion as
    natural English action ("The camera pushes in with small amplitude at slow
    speed ..."), so a grep aimed at out-of-table phrasing can very easily fire
    on the exact sentences the guide holds up as correct.

    Extracted rather than pinned by line number: every sentence in the guide
    beginning "The camera ", plus every worked
    `integrated_multimodal_description`. A red proof shows a check CAN fail;
    this shows it does not fail on known-good text, and the two are different
    properties. `docs/checks.md`: a check reporting red while the state is
    correct trains you to ignore red.

    **What it does NOT prove**, stated because the sibling project stated it
    first and was right to: the corpus is the guide's examples, not the space
    of legitimate prose. Silence here means the rules do not fire on the
    vendor's own sentences. It does not mean they are correct.
    """
    if not GUIDE.exists():
        print("  skip  quiet_on_examples  guide not on disk; the cry-wolf "
              "corpus is the guide's own prose and cannot be built without it")
        return []
    text = GUIDE.read_text(encoding="utf-8")
    corpus = {}
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("The camera ") or \
                stripped.startswith("integrated_multimodal_description:"):
            corpus[f"base_en:{i}"] = stripped
    if not corpus:
        return ["  FAIL  quiet_on_examples  found no worked motion prose in the "
                "guide; the extraction broke and silence here would be vacuous"]

    fails = [f for f in case_modifiers_are_in_set(
        {t: {k} for k, t in corpus.items()}, quiet=True)]
    for k, t in corpus.items():
        for bad in DENIED:
            if bad in t.lower():
                fails.append(f"  FAIL  quiet_on_examples  fired `{bad}` on the "
                             f"guide's own prose at {k} -- the rule is wrong, "
                             "not the text")
    if fails:
        return [f.replace("modifiers_in_set", "quiet_on_examples") for f in fails]

    # WHAT THE CORPUS DOES NOT CONTAIN. Silence is only evidence for a rule the
    # corpus actually EXERCISES -- one whose inputs appear in it at all. A rule
    # the corpus never feeds is silent for a structural reason, and reporting
    # that as a pass is how a green cry-wolf control hides a wrong rule.
    #
    # So per rule: did the corpus contain anything this rule LOOKS AT? Not
    # anything it fires on -- that would be the same vacuity one level up.
    #
    # **And not anything that would EXPOSE its bug**, which is a third and
    # stronger property this does not have. The distinction was worth a
    # correction: an earlier version of this comment offered a sibling
    # project's punctuation bug as the motivating instance. That bug sat green
    # through their worked-example control -- true -- but their corpus DID
    # contain the input class the rule inspects, a correctly-punctuated
    # user-supplied line. The bug lived in a VALUE inside a covered class, so
    # this predicate would have stayed silent on it too. Stated because the
    # limit is easy to write down and then contradict in the next sentence,
    # which is what happened.
    #
    # What the predicate is actually good for, on their evidence rather than
    # mine: run across their five fixtures it finds four rules whose inputs are
    # absent outright -- no fixture sets the flag or fills the array those rules
    # read. Each has a passing green test asserting the code is absent, and each
    # of those greens is structural. That is the class this catches.
    blob = " ".join(corpus.values())
    exercised = {
        "modifiers_in_set": bool(AMP_RE.search(blob)) and bool(SPD_RE.search(blob)),
        # A denylist is inputs-by-enumeration, so a known-good corpus can never
        # exercise it: if it did, the corpus would not be known-good. This is
        # ALWAYS False and is reported rather than hidden.
        "denied_motion": any(bad in blob.lower() for bad in DENIED),
    }
    print(f"  ok    quiet_on_examples  silent on {len(corpus)} passage(s) of the "
          "guide's own prose (cry-wolf corpus)")
    for rule, ran in exercised.items():
        if ran:
            print(f"        exercised  {rule}: the corpus contains inputs this "
                  "rule inspects, so its silence is evidence")
        else:
            print(f"        NOT exercised  {rule}: the corpus contains nothing "
                  "this rule inspects, so its silence here is structural and "
                  "proves nothing about it")
    return []


def case_no_denied_motion(corpus) -> list[str]:
    """NOT decidable: a denylist, reported. Absence here proves nothing."""
    warns = []
    for text, graphs in sorted(corpus.items(), key=lambda kv: sorted(kv[1])[0]):
        where = sorted(graphs)[0]
        low = text.lower()
        for bad, why in DENIED.items():
            if bad in low:
                warns.append(f"  warn  denied_motion      {where}: `{bad}` -- {why}")
    for line in warns:
        print(line)
    if not warns:
        print(f"  ok    denied_motion      no known-bad motion phrase in "
              f"{len(corpus)} prompt(s). **This is a denylist: it proves only "
              "that the phrases somebody already got wrong are absent**")
    return []


def main() -> int:
    corpus = prompts()
    if not corpus:
        print("FAIL  no prompts found -- the scan or the graph shape changed; "
              "this check must not pass by looking at nothing")
        return 2
    fails = case_vocab_matches_guide()
    fails += case_quiet_on_guide_examples()
    fails += case_modifiers_are_in_set(corpus)
    case_no_denied_motion(corpus)
    if fails:
        print()
        for line in fails:
            print(line)
        print(f"\n{len(fails)} camera-vocabulary failure(s). A phrase outside "
              "4.3's table is off-distribution: the model was trained on that "
              "vocabulary and not on this one.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
