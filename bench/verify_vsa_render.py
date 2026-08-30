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

## The trap this file exists to avoid

**Hashing the mp4 answers the wrong question.** Two renders of identical pixels
produce different mp4 bytes, because the container carries encoder metadata
that varies per run. Comparing `md5sum` on the files therefore reports a
difference for any two runs, including two runs of the same arm -- which reads
as "the arms differ" and is not.

That was caught here rather than in review: the two VSA runs hashed
differently, which would have been reported as non-determinism, and the tell
was that their file SIZES were identical to the byte. So this compares the
DECODED RGB STREAM, which carries no container metadata.

    python bench/verify_vsa_render.py <vsa.mp4> <dense.mp4> [<vsa-repeat.mp4>]

Needs ffmpeg. Exit 0 if the arms differ (and, when given, the repeat matches),
1 if not.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


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

    ok = True
    print()
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
