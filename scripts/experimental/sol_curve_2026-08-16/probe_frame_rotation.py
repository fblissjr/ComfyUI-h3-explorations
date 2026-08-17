"""Test the external analysis's headline proposal: per-frame Hilbert start rotation.

Its claim: rotate each frame's Hilbert walk by `(frame_idx * area) % 64` and
block boundaries align with frame boundaries, converting 90% connected toward
the 100% clean canvases achieve.

Implemented exactly as specified, then scored against the unrotated curve on
geometry and on the real capture.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import sol_curves  # noqa: E402
from probe_hilbert import connected_fraction, gilbert_within, shipped_within  # noqa: E402

BLOCK = 64


def rotate(seq, shift):
    """The proposal's rotation, on the within-frame curve order."""
    shift %= len(seq)
    return seq[shift:] + seq[:shift]


def perm_rotated(within, frames, area):
    out = []
    for f in range(frames):
        out.extend(f * area + i for i in rotate(within, (f * area) % BLOCK))
    return torch.tensor(out, dtype=torch.int64)


def perm_plain(within, frames, area):
    return torch.tensor([f * area + i for f in range(frames) for i in within],
                        dtype=torch.int64)


def nonadjacent(seq, width):
    pts = [(i // width, i % width) for i in seq]
    return sum(1 for a, b in zip(pts, pts[1:])
               if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1)


def main():
    frames, height, width = 37, 24, 42
    area = height * width
    grid = (frames, height, width)
    start = 530
    pad = (-start) % BLOCK

    hil = shipped_within(height, width)
    gil = gilbert_within(height, width)

    print(f"grid {grid}, area {area}, area % 64 = {area % BLOCK}\n")

    print("Does rotating an open curve preserve adjacency? "
          "(non-adjacent steps within one frame)")
    for name, seq in (("hilbert", hil), ("gilbert", gil)):
        base = nonadjacent(seq, width)
        worst = max(nonadjacent(rotate(seq, s), width) for s in range(1, BLOCK))
        print(f"  {name:<10} unrotated {base:>3}   worst rotation {worst:>3}")
    print()

    arms = {
        "hilbert (shipped)": perm_plain(hil, frames, area),
        "hilbert + frame rotation": perm_rotated(hil, frames, area),
        "gilbert": perm_plain(gil, frames, area),
        "gilbert + frame rotation": perm_rotated(gil, frames, area),
    }
    print("Geometry at real block alignment (video_start=530):")
    for name, perm in arms.items():
        p = torch.roll(perm, pad) if pad else perm
        frac, rad, n = connected_fraction(p, grid, start)
        print(f"  {name:<26} connected {frac:6.1%}   mean radius {rad:5.2f}")

    # Does any within-frame permutation move where a frame sits? No: check it.
    print("\nDoes the rotation move a frame's span in the global sequence?")
    a = set(perm_plain(hil, frames, area)[:area].tolist())
    b = set(perm_rotated(hil, frames, area)[:area].tolist())
    print(f"  frame 0 occupies the same {len(a)} global positions: {a == b}")
    print(f"  block boundaries inside frame 0: "
          f"{[i for i in range(1, area) if (start + i) % BLOCK == 0][:3]} ...")


if __name__ == "__main__":
    main()
