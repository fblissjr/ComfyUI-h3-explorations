#!/usr/bin/env python3
"""Verify a VSA render actually ran VSA, by its decoded pixels.

## The question, and why the obvious check does not answer it

`MiniMaxH3VSAAttention` replaces the DiT block forward, and its replacement
falls back to the original block when the packed layout is missing or the cube
geometry does not resolve. That fallback logs once, on block 0 only -- so
"there was no warning in the log" is weak evidence, and a render that quietly
ran dense looks exactly like a render that ran VSA.

The decisive test is the dense CONTROL: the same checkpoint, the same seed, the
same step count, differing only in whether the node is wired. If VSA fell
through, the two outputs are the same computation and must match.

## The trap this file exists to avoid, and the exact field

**Hashing the mp4 answers the wrong question.** Two renders of identical pixels
produce different mp4 bytes. Comparing `md5sum` on the files therefore reports
a difference for any two runs, including two runs of the same arm -- which
reads as "the arms differ" and is not.

**Two container tags did it, and only one explained the same-arm case.** Named
rather than waved at, because "encoder metadata" is not checkable and a field
name is, in one `ffprobe -show_entries format_tags`:

  `format.tags.comment`        the whole API prompt, under VHS's
                               `save_metadata: true`. So any two ARMS differed
                               by construction -- the thing that makes them
                               different arms was serialised into the file.
  `format.tags.creation_time`  a wall-clock timestamp. So any two RUNS
                               differed, including two runs of one arm.

Only the second explained two runs of one graph, whose comment tags were
identical. The muxer and the codec are NOT the cause and that was checked
rather than assumed: remuxing one file twice, and re-encoding it twice at the
same settings, each produce identical bytes. Credit to a peer session for
pinning the field and for running that control.

**Since 2026-09-23 neither tag is written.** The owner's VHS fork writes no
metadata into video files, and the graph lives only in the first-frame PNG
beside each clip. So a same-arm pair from after that date may well hash equal
as files. Nothing here relies on that: the pixel comparison below never looked
at the container, and it is still the comparison that decides.

Caught here rather than in review: the two VSA runs hashed differently, which
would have been reported as non-determinism, and the tell was that their file
SIZES matched to the byte. This compares the DECODED RGB STREAM instead, which
carries no container at all.

**The one-way implication survives both mechanisms**, so nothing previously
concluded from a MATCHING file hash is withdrawn: identical container still
implies identical frames.

## Do not delete the arm-identity case as redundant

It looks like belt-and-braces beside the pixel comparison. It is not: without
it, **the pixel comparison cannot fail in the way its name implies.**

That check asks "do these two files differ". A wrong pair differs. A pair from
two different sessions differs. Two unrelated renders differ. So on any input
except the exact right one it returns the same verdict it returns on success --
which is CLAUDE.md's oldest rule, a check whose input already satisfies the
expected outcome cannot fail, wearing a new coat.

Demonstrated rather than argued: run this with another session's PDD render as
the `vsa` argument and the pixel case still prints `ok VSA differs from its
dense control`. The whole file passes on a comparison that never involved VSA.
The identity case is what converts "these differ" into "VSA differs from ITS
control", and it is the only part that makes the name true.

## Why it checks the arms rather than trusting the filenames

`bench/smoke_h3.py` hard-codes one `_smoketest` prefix, so every session
rendering on this box shares a single output counter. Renders from different
sessions interleave, and the FILENAME carries no arm information whatsoever --
only the embedded graph does, which is this same trap wearing a
different hat. A peer session went looking for its own two arms, found a
consecutive pair, and they were this session's.

So the arms are identified from the graph each file was rendered from: the VSA
file must wire `MiniMaxH3VSAAttention` and the control must not, and both must
carry the same seed. Passing the wrong pair is otherwise indistinguishable from
a result. The graph comes from the clip's sidecar PNG
(`diff_clip_graphs.graph_of` says how it is found, and why finding it by name
does not reintroduce the filename trap: one save writes both files under one
counter, so the PNG names the render and not just a slot). A clip with no
graph to read FAILS the identity case rather than falling back to its name.

    python bench/verify_vsa_render.py <vsa.mp4> <dense.mp4> [<vsa-repeat.mp4>]

Needs ffmpeg. Exit 0 if the arms differ (and, when given, the repeat matches),
1 if not.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diff_clip_graphs import graph_of  # noqa: E402


def embedded_graph(path):
    """The API graph `path` was rendered from, or None when there is none to read.

    `diff_clip_graphs.graph_of` owns where it is looked for: the sidecar PNG
    first, then the container tags older clips carry.
    """
    try:
        return graph_of(str(path))
    except (ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        return None


def has_node(graph, class_type):
    if graph is None:
        return None
    return any(isinstance(n, dict) and n.get("class_type") == class_type
               for n in graph.values())


def seed_of(graph):
    if graph is None:
        return None
    for node in graph.values():
        if isinstance(node, dict) and "noise_seed" in node.get("inputs", {}):
            return node["inputs"]["noise_seed"]
    return None


def pixel_digest(path):
    """sha256 of the decoded RGB frames, ignoring the container entirely."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"could not decode {path}: "
                           f"{proc.stderr.decode()[:200]}")
    return hashlib.sha256(proc.stdout).hexdigest()


def main(argv):
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 1
    paths = [Path(a) for a in argv]
    labels = ["vsa", "dense control", "vsa repeat"][:len(paths)]
    digests = {}
    for label, path in zip(labels, paths):
        digests[label] = pixel_digest(path)
        print(f"  {label:<14} {digests[label][:16]}  {path.name}")

    # Identify the arms from what is IN each file, never from its name.
    ok = True
    print()
    arms = {}
    for label, path in zip(labels, paths):
        arms[label] = embedded_graph(path)
    vsa_wired = has_node(arms["vsa"], "MiniMaxH3VSAAttention")
    control_wired = has_node(arms["dense control"], "MiniMaxH3VSAAttention")
    seeds = {label: seed_of(g) for label, g in arms.items() if g is not None}
    matched = len(set(v for v in seeds.values() if v is not None)) <= 1

    if arms["vsa"] is None or arms["dense control"] is None:
        print("  FAIL  the arms are what they are labelled   no graph to read: "
              "each clip needs its first-frame PNG beside it (or, for an older "
              "clip, the graph in its container). A filename says nothing "
              "about the arm.")
        ok = False
    else:
        right = vsa_wired and not control_wired and matched
        print(f"  {'ok  ' if right else 'FAIL'} the arms are what they are "
              f"labelled")
        if not right:
            print(f"       vsa wires MiniMaxH3VSAAttention: {vsa_wired}; "
                  f"control wires it: {control_wired}; seeds {seeds}.")
            print("       Every session on this box shares one _smoketest "
                  "counter, so consecutive\n       files are not necessarily "
                  "one session's, let alone one pair.")
            ok = False

    differ = digests["vsa"] != digests["dense control"]
    print(f"  {'ok  ' if differ else 'FAIL'} VSA differs from its dense control")
    if not differ:
        print("       The two arms computed the same thing. Either the node "
              "fell back to the\n       original block, or it is not reaching "
              "the kernel. A render that quietly\n       runs dense is the "
              "failure this check exists for.")
        ok = False

    if "vsa repeat" in digests:
        same = digests["vsa"] == digests["vsa repeat"]
        print(f"  {'ok  ' if same else 'FAIL'} the VSA arm reproduces itself "
              f"at the same seed")
        if not same:
            print("       Two runs of one arm disagree, so the difference "
                  "above cannot be\n       attributed to the attention regime.")
            ok = False

    print()
    print("all cases passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
