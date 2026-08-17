"""Pre-flight verification for multi-step activation captures.

TWO gates and one report. The split matters and was wrong until 2026-08-17,
when all three were called invariants and the run ended with "100% PROVEN".

Gates -- these fail the run:
1. Determinism: the new `_s3.pt` is bit-identical to the baseline capture's.
2. Distinctness: the captured steps are not the same tensor as each other. A
   step counter that silently reset would produce duplicates, which is exactly
   the defect `h3_capture.py` carried until the same day.

Report -- printed, never fails the run:
3. Trajectory distance. `dist(s3,s14) > max(dist(s3,s8), dist(s8,s14))` was
   asserted as an invariant and it is not one: nothing in diffusion requires the
   endpoints of a trajectory to be further apart than the intermediate hops, and
   a schedule that turns near the tail gives a smaller end-to-end distance with
   every tensor correct. As a gate it was red on correct state, which CLAUDE.md
   rates worse than no check. It is worth LOOKING at -- a collapsed trajectory
   shows up here -- so it prints, with no threshold pretending to be a law.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
import torch

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def find_one(directory: Path, block: int, step: int) -> Path:
    """The single capture file for a (block, step), or a clean failure.

    `list(dir.glob(...))[0]` was the original, and it had two faults. It raised
    a bare IndexError on a partial capture, where section 1 in the same file
    printed a clean FAIL -- so the same missing-file condition reported two
    different ways depending on which section reached it first. And `Path.glob`
    returns `os.scandir` order, not sorted order, while the pattern matches
    every sequence length in the directory: a directory holding captures at two
    lengths made the comparison pick whichever file the filesystem happened to
    yield first, so a SHA-256 "determinism" verdict could be comparing two
    different runs.

    Ambiguity is an error here rather than a tie broken by sorting. If two files
    match, the caller does not know which run it is measuring, and picking the
    lexicographically first one silently answers a question nobody asked.
    """
    matches = sorted(directory.glob(f"*_b{block}_s{step}.pt"))
    if not matches:
        raise FileNotFoundError(
            f"no capture for block {block} step {step} in {directory}")
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise ValueError(
            f"{len(matches)} files match block {block} step {step} in "
            f"{directory}: {names}. The filename encodes L{{length}}_S{{seq}}, so "
            f"this directory holds more than one run -- pass a directory with a "
            f"single run rather than letting the comparison pick one.")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Required rather than defaulted. A capture directory is per-machine, so a
    # baked-in default only ever runs on one box and silently points at the
    # wrong place everywhere else.
    parser.add_argument("--multistep-dir", required=True,
                        help="capture directory holding the multi-step tensors")
    parser.add_argument("--baseline-dir", required=True,
                        help="capture directory holding the single-step baseline")
    parser.add_argument("--blocks", default="0,24,40,49")
    parser.add_argument("--steps", default="3,8,14",
                        help="captured step indices to compare; the first is "
                             "the one checked against the baseline")
    args = parser.parse_args()

    mdir = Path(os.path.expanduser(args.multistep_dir))
    bdir = Path(os.path.expanduser(args.baseline_dir))
    blocks = [int(b) for b in args.blocks.split(",") if b.strip()]

    print("=" * 80)
    print(" Multi-Step Activation Pre-Flight Verification Gate")
    print(f" Multi-Step Dir: {mdir}")
    print(f" Baseline Dir:   {bdir}")
    print(f" Blocks:         {blocks}")
    print("=" * 80)

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    first = steps[0]

    try:
        # 1. Determinism: the first captured step must be bit-identical to the
        #    baseline's same step.
        print(f"\n[gate 1/2] Determinism (new s{first} vs baseline s{first})...")
        for b in blocks:
            new_f, old_f = find_one(mdir, b, first), find_one(bdir, b, first)
            new_hash, old_hash = file_sha256(new_f), file_sha256(old_f)
            if new_hash != old_hash:
                print(f"  FAIL: SHA-256 mismatch for block {b} step {first}\n"
                      f"    new: {new_hash}\n    old: {old_hash}", file=sys.stderr)
                return 1
            print(f"  PASS: block {b} s{first} matches baseline: {new_hash[:16]}...")

        # 2. Distinctness, and 3. trajectory distance, share ONE load pass.
        #    They were two, each loading the same three multi-GiB tensors, which
        #    doubled a very large read for no reason and could swap or OOM a box
        #    with a render in flight.
        print(f"\n[gate 2/2] Distinctness across steps {steps}...")
        distances = {}
        for b in blocks:
            loaded = {s: torch.load(find_one(mdir, b, s), map_location="cpu")
                      for s in steps}
            for i, s_a in enumerate(steps):
                for s_b in steps[i + 1:]:
                    qa, qb = loaded[s_a]["q"], loaded[s_b]["q"]
                    if torch.equal(qa, qb):
                        print(f"  FAIL: block {b} s{s_a} == s{s_b} "
                              f"(identical tensor)", file=sys.stderr)
                        return 1
                    distances[(b, s_a, s_b)] = (
                        qa.float() - qb.float()).norm().item()
            print(f"  PASS: block {b} steps {steps} are pairwise distinct")
            del loaded

        # REPORT, not a gate. See the module docstring: end-to-end distance
        # exceeding every intermediate hop is not guaranteed by anything, so
        # failing on it was red on correct state.
        print("\n[report] Activation-space trajectory distance (no verdict)...")
        for b in blocks:
            pairs = " ".join(f"s{a}->s{c}={distances[(b, a, c)]:.2f}"
                             for (bb, a, c) in distances if bb == b)
            print(f"  block {b}: {pairs}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 80)
    print(" BOTH GATES PASS. The trajectory report above carries no verdict.")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
