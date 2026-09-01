#!/usr/bin/env python
"""Grade a candidate prompt TEXT, for a chosen mode, without building a graph.

## Why this exists

`bench/preflight_graph.py` grades prompts, but only ones already baked into a
graph. That left the prompts in `docs/prompting.md` section 10 ungraded by
anything: the section states its examples "grade clean through
`preflight_graph.py`", and until 2026-09-01 nothing checked that -- the claim
and the artifact had no connection. Writing a new example meant either shipping
a graph for it or asserting conformance by eye, and this repo has retracted two
prompt rules that were asserted by eye.

So this wraps a bare prompt in a real shipped graph of the requested mode and
runs the existing grader against it. It adds no rules of its own. Every finding
it prints comes from `preflight_graph.grade`, which is the thing that already
owns the guides' mechanical rules; if the two ever disagree, preflight is right
because this is a thin caller of it.

## Mode is read from the DONOR'S SOCKETS, never from the prompt text

The prompt text cannot name its own mode -- that is the whole point, since a
mode-mismatched alignment sentence is exactly the defect
`_expected_base_alignment` exists to catch, and classifying by text would
inherit the bug. `preflight_graph` decides mode from whether the conditioner
has `first_frame` / `last_frame` wired, so this selects a donor by the same
observable and lets preflight re-derive it:

    MiniMaxH3Conditioning            no frame sockets  -> t2va
                                     first_frame       -> i2va
                                     first_frame+last  -> fl2va
                                     last_frame        -> l2va
    MiniMaxH3ReferenceConditioning                     -> ref2va

Donors are DISCOVERED by walking `h3_config.graph_paths`, not listed here. A
hardcoded donor list is a second copy of which graphs exist, and this repo has
watched that shape rot more than once. When a mode has no donor the tool RAISES
rather than falling back to a near neighbour: grading an l2va prompt against an
fl2va graph would compare it to the wrong alignment template and pass it.

## The duration is part of the answer

`S.SS` and every `[Shot N] At MM:SS.mmm` in a prompt are resolved against the
graph's snapped length, so a prompt is only conformant AT A DURATION. The donor
supplies one, and it is printed with the expected Part One line so an author can
see what the text has to match. Pass `--like` to grade against a different
graph's length.

## What the exit code means

Nonzero on any FAIL. WARN is advisory and does not fail, which matches
preflight's "reports, never refuses" posture -- the standing ref2va word-budget
WARN would otherwise make every reference example unfixable here.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO / "bench"))

import h3_config  # noqa: E402
import preflight_graph as pf  # noqa: E402

MODES = ("t2va", "i2va", "fl2va", "l2va", "ref2va")


def mode_of(node: dict) -> str | None:
    """The mode a conditioner node expresses, by socket presence."""
    cls = node.get("class_type")
    ins = node.get("inputs", {})
    if cls in ("MiniMaxH3ReferenceConditioning", "MiniMaxH3ReferenceToVideo"):
        return "ref2va"
    if cls not in ("MiniMaxH3Conditioning", "MiniMaxH3ImageToVideo"):
        return None
    first = ins.get("first_frame") is not None
    last = ins.get("last_frame") is not None
    return {(False, False): "t2va", (True, False): "i2va",
            (True, True): "fl2va", (False, True): "l2va"}[(first, last)]


def donors() -> dict[str, list[tuple[Path, str]]]:
    """Every API-format shipped graph that can host a prompt, by mode."""
    found: dict[str, list[tuple[Path, str]]] = {m: [] for m in MODES}
    for path in h3_config.graph_paths(REPO / "workflows"):
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(graph, dict) or isinstance(graph.get("nodes"), list):
            continue  # UI-format graph; preflight grades only the API form
        for nid, node in graph.items():
            if not isinstance(node, dict):
                continue
            mode = mode_of(node)
            if mode and isinstance(node.get("inputs", {}).get("prompt"), str):
                found[mode].append((path, nid))
    return found


def pick(mode: str, like: str | None) -> tuple[Path, str]:
    table = donors()
    if like:
        for candidates in table.values():
            for path, nid in candidates:
                if path.stem == like:
                    return path, nid
        raise SystemExit(f"FAIL  --like {like}: no shipped graph with that stem "
                         f"carries a literal prompt")
    candidates = table[mode]
    if not candidates:
        raise SystemExit(
            f"FAIL  no shipped graph expresses {mode}, so a {mode} prompt cannot "
            f"be graded at a real duration. Ship a {mode} graph, or pass --like "
            f"to name one explicitly.")
    # Prefer a PLAIN graph over a probe. A probe exists to vary one axis and
    # its canvas, length or sampler is deliberately not the ordinary one, so
    # grading an example at a probe's duration would write the probe's timing
    # into the example. Sorting by stem length alone picked `h3_probe_vsa_api`
    # over `h3_text_to_video_api`, which is why this is explicit.
    def rank(c: tuple[Path, str]) -> tuple[int, int, str]:
        stem = c[0].stem
        return (1 if "probe" in stem else 0, len(stem), stem)
    return sorted(candidates, key=rank)[0]


def grade_text(text: str, mode: str | None, like: str | None,
               length: int | None) -> dict:
    """Grade loose prompt text through a donor graph. The ONE code path.

    `main()` and `bench/build_prompt_bank.py` both call this, so the bank is
    graded by exactly what the CLI grades by. Returns the findings plus what
    the CLI prints: donor path and node, the mode preflight derived from the
    donor's sockets, the resolved length, and the Part One line it expects.
    Raises SystemExit on a mode/donor mismatch, as the CLI always did.
    """
    path, nid = pick(mode, like)
    graph = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    node = graph[nid]
    graded_mode = mode_of(node)
    if mode and graded_mode != mode:
        raise SystemExit(f"FAIL  --like {like} is {graded_mode}, not {mode}")
    node["inputs"]["prompt"] = text
    if length is not None:
        # Overwrite the LINK with a literal. `_resolved_length` snaps it and
        # `grade`'s cut-past-the-clip check reads the same field, so both halves
        # see the requested duration rather than the donor's.
        node["inputs"]["length"] = length
    resolved = pf._resolved_length(node, graph)
    # Same extraction `grade` uses -- imported rather than copied, because a
    # copy is what drifted: this held the pre-fix greedy pattern while its
    # comment claimed the two matched, so the advisory printed `from Shot 1`
    # for a multi-shot keyframe prompt and told the author to write a
    # guide-violating line.
    shots = re.findall(pf.SHOT_HEADER_RE, text)
    expected, label = pf._expected_base_alignment(node, graph, shots)
    return {"path": path, "nid": nid, "mode": graded_mode, "length": resolved,
            "expected": expected, "label": label,
            "findings": pf.grade(node, graph, path.stem)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt", nargs="?", type=Path,
                    help="file holding the prompt text; omit to read stdin")
    ap.add_argument("--mode", choices=MODES,
                    help="which mode to grade as; picks a donor graph")
    ap.add_argument("--like", metavar="STEM",
                    help="grade against this graph's sockets and length instead")
    ap.add_argument("--length", type=int, metavar="FRAMES",
                    help="grade at this length instead of the donor's. A prompt "
                         "is conformant AT A DURATION -- `S.SS` and every cut "
                         "timestamp resolve against it -- so an example written "
                         "for 192 frames must be graded at 192, not at whatever "
                         "the donor happens to be. Snapped the way the node "
                         "snaps it.")
    ap.add_argument("--list-donors", action="store_true",
                    help="print the donor graph discovered for each mode and exit")
    args = ap.parse_args()

    if args.list_donors:
        table = donors()
        for mode in MODES:
            if table[mode]:
                path, _ = pick(mode, None)
                print(f"  {mode:7s} {path.relative_to(REPO)}  "
                      f"({len(table[mode])} candidate(s))")
            else:
                print(f"  {mode:7s} -- none ship --")
        return 0

    if not args.mode and not args.like:
        ap.error("one of --mode or --like is required")

    text = (args.prompt.read_text(encoding="utf-8") if args.prompt
            else sys.stdin.read()).strip()
    if not text:
        raise SystemExit("FAIL  empty prompt")

    r = grade_text(text, args.mode, args.like, args.length)
    path, nid, graded_mode, length, expected, label, findings = (
        r["path"], r["nid"], r["mode"], r["length"], r["expected"],
        r["label"], r["findings"])
    print(f"  mode      {graded_mode}")
    print(f"  donor     {path.relative_to(REPO)}  (node {nid})")
    print(f"  length    {length if length is not None else 'unresolved'} frames"
          + (f"  = {h3_config_duration(length)}" if length else ""))
    if expected:
        print(f"  expects   Part One for {label}:\n              {expected}")
    print("")

    if not findings:
        print("  ok    every mechanical rule passes")
    for level, msg in findings:
        print(f"  {level:<4}  {msg}")
    fails = [f for f in findings if f[0] == "FAIL"]

    # A REFERENCE LABEL IS GRADED AGAINST THE DONOR'S SOCKETS, so a prompt that
    # legitimately names two pictures and an audio clip FAILS against a donor
    # wiring one picture -- and the failure names the prompt, not the donor.
    # That reading nearly got `docs/prompting.md` section 10.5 recorded as
    # defective on 2026-09-01 when it is clean against a socket-matched donor.
    # Say so here rather than leaving the next reader to re-derive it.
    unwired = [m for lvl, m in fails if "which no socket wires" in m]
    if unwired and not args.like:
        print("")
        print("  hint  every FAIL above is a label with no matching socket, "
              "which is a")
        print("        property of the DONOR, not necessarily of the prompt. "
              "Re-run with")
        print("        --like <stem> naming a graph that wires the references "
              "this prompt")
        print("        declares before believing it.")

    print("")
    warns = [f for f in findings if f[0] == "WARN"]
    notes = [f for f in findings if f[0] not in ("FAIL", "WARN")]
    print(f"  {len(fails)} FAIL, {len(warns)} WARN, {len(notes)} note")
    return 1 if fails else 0


def h3_config_duration(frames: int) -> str:
    import h3_rules
    return f"{h3_rules.duration_of(frames):.3f} s"


if __name__ == "__main__":
    sys.exit(main())
