#!/usr/bin/env python3
"""What Morton reordering actually does to H3's 64-token attention blocks.

Morton is the one Sol-Attn knob nobody here can predict. It reads as "a
different seed that saves ten seconds", and every attempt to judge it has gone
through clips, which `docs/SOLATTN.md` says is the instrument that cannot
answer this class of question. This script answers the half that needs no clip
and no GPU: **what the permutation does to the 64-token blocks the router
actually operates on.**

## The mechanism, so the numbers below have somewhere to land

Sol-Attn partitions the sequence into 64-token blocks and, per query block,
routes a subset of key blocks to the exact branch while covering the rest with
one pooled term per block. Which blocks get routed is decided from a per-block
summary. So the *only* thing that decides quality, at a fixed tau, is **which
tokens share a block** -- because that sets how well one pooled vector stands
in for 64 of them.

Morton does not change the model, the weights, the sampler, or the number of
tokens. It permutes the video span (and its rope rows identically) so that
64 consecutive tokens are a compact 3D or 2D neighbourhood instead of a
raster-order run. It is undone after the last block. For DENSE attention it is
exactly neutral -- attention is permutation-equivariant, and everything else in
the block is per-token. It can only ever matter through block membership.

That is why this is measurable without rendering: block membership is pure
arithmetic on the grid.

## What "raster" means here

H3's video span is `(latent_t, h//32, w//32)` in t-major, then row, then
column order. At 1344x768 that is 24 rows of 42 columns per frame, so 64
consecutive tokens are **1.52 rows of one frame** -- a thin horizontal strip
about 42 cells wide and 2 tall. Morton's claim is that a compact ~8x8 tile is
a better unit to summarise. This script measures whether it delivers one.

## Three things this found that the tooltip does not say

Run it to see them at your own grid; they are printed with the evidence.

1. **Frame boundaries do not align with block boundaries at 1344x768.**
   1008 tokens per frame / 64 = 15.75 blocks. So three of every four frame
   boundaries land mid-block, and `2d_frame` -- whose entire premise is that
   frames never mix -- still produces blocks holding two frames. At 1024x768
   it is 768/64 = 12.0 and the premise holds exactly.
2. **The latent grid is not a power of two, so Z-order cells are ragged.**
   Morton codes tile a padded 64x64 space; a 24x42 frame keeps only the
   in-range corner of each tile, so consecutive tokens in Morton order are
   sometimes a compact 8x8 and sometimes two disjoint fragments from opposite
   sides of the frame.
3. **The start-offset rotation in `_perm_for` is load-bearing and correct.**
   The block grid is anchored at absolute row 0, so reference rows move where
   the video span's blocks fall. Rotating the permutation by
   `(-video_start) % 64` makes block geometry **invariant to `video_start`** --
   verified here at seven offsets. Without it, 1024x768 with references falls
   from fill 1.00 to 0.42, worse than the ragged canvas. Pass `--video-start`
   to see both, including the `_noroll` control.

   That row was nearly written as the opposite finding. Grouping tokens by
   `j // 64` instead of `(video_start + j) // 64` measures a partition that
   exists on no graph with references, and it makes the rotation look like the
   cause of the damage it prevents. See `block_ids`.

## Running it

    python bench/analyze_morton.py                       # every shipped canvas
    python bench/analyze_morton.py --canvas 1344x768 --length 294
    python bench/analyze_morton.py --canvas 1024x768 --length 362 \
        --video-start 16821 --map          # what a reference graph really gets

`--map` prints one frame of one block partition as ASCII, which is the fastest
way to stop arguing about it.

No CUDA, no model, no server. Runs in about a second.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent

# ComfyUI's root FIRST. This repo has a `nodes.py` and a bare `import nodes`
# anywhere downstream finds ours and dies on a relative import -- three
# separate debugging rounds' worth, per CLAUDE.md.
for p in (str(COMFY), str(REPO / "workflows")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

BLOCK_SIZE = 64


def load_shipped_morton():
    """The permutation the node actually installs, from the vendored source.

    Imported rather than reimplemented on purpose: a reimplementation would
    measure this script's idea of Morton, which is the failure mode the repo
    keeps re-learning. `_independent_perm` below is the cross-check, and they
    are asserted equal before any number is printed.
    """
    path = REPO / "vendor" / "sol_attn_minimax.py"
    spec = importlib.util.spec_from_file_location("_sol_vendor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _independent_perm(grid, curve):
    """A second implementation, deliberately written a different way.

    The shipped one builds interleaved bit codes with a magic-constant bit
    twiddle and argsorts. This one sorts tuples of interleaved bit strings in
    Python. If the two ever disagree, every number this script prints is about
    a permutation nobody runs.
    """
    frames, height, width = grid

    def bits(v, n=21):
        return format(v, f"0{n}b")

    keys = []
    for index in range(frames * height * width):
        z, rem = divmod(index, height * width)
        y, x = divmod(rem, width)
        bx, by, bz = bits(x), bits(y), bits(z)
        if curve == "2d_frame":
            code = (z, "".join(by[i] + bx[i] for i in range(21)))
        else:
            code = ("".join(bz[i] + by[i] + bx[i] for i in range(21)),)
        keys.append((code, index))
    keys.sort()
    return torch.tensor([i for _, i in keys], dtype=torch.int64)


def coords(perm, grid):
    """(t, y, x) of every token, in permuted order."""
    frames, height, width = grid
    t = perm // (height * width)
    rem = perm - t * (height * width)
    y = rem // width
    x = rem - y * width
    return t, y, x


def block_ids(grid, video_start):
    """Absolute 64-token block id of each position in the video span.

    **Group by `(video_start + j) // 64`, never by `j // 64`.** The kernel
    blocks from absolute row 0, so the partition of the video span depends on
    how many conditioning rows precede it. Slicing from the span's own index 0
    measures a partition that does not exist on any graph with references --
    and it makes the shipped rotation in `_perm_for` look like it *breaks*
    alignment when in fact it is what preserves it. That wrong reading was
    produced once while writing this file; the correction is the reason this
    function exists separately.
    """
    total = grid[0] * grid[1] * grid[2]
    return (int(video_start) + torch.arange(total, dtype=torch.int64)) // BLOCK_SIZE


def block_stats(perm, grid, video_start=0):
    """Per-64-token-block geometry, for whichever ordering `perm` encodes.

    Partial blocks at either end are dropped: the leading one is shared with
    the conditioning rows, which the exact-KV sink already keeps exact, so its
    geometry is not what the router acts on.
    """
    frames, height, width = grid
    t, y, x = coords(perm, grid)
    blk = block_ids(grid, video_start)
    order = torch.argsort(blk, stable=True)
    blk, t, y, x = blk[order], t[order], y[order], x[order]

    counts = torch.bincount(blk - blk.min())
    full = (counts == BLOCK_SIZE)
    # Offset of each full block's first element in the sorted arrays.
    starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64),
                                     counts[:-1]]), 0)[full]
    idx = starts[:, None] + torch.arange(BLOCK_SIZE)
    t = t[idx].float()
    y = y[idx].float()
    x = x[idx].float()

    n_frames = torch.tensor(
        [len(set(row.tolist())) for row in t.long()], dtype=torch.float32)
    t_span = t.max(1).values - t.min(1).values + 1
    h_span = y.max(1).values - y.min(1).values + 1
    w_span = x.max(1).values - x.min(1).values + 1
    # RMS distance from the block's own spatial centroid, in latent cells.
    # This is the geometric stand-in for "how well does one pooled vector
    # represent these 64 tokens", which is the quantity the router uses.
    radius = torch.sqrt(((y - y.mean(1, keepdim=True)) ** 2
                         + (x - x.mean(1, keepdim=True)) ** 2).mean(1))
    fill = BLOCK_SIZE / (t_span * h_span * w_span)
    return dict(blocks=int(full.sum()), n_frames=n_frames, t_span=t_span,
                h_span=h_span, w_span=w_span, radius=radius, fill=fill)


def neighbour_retention(perm, grid, video_start=0):
    """Fraction of a token's 6 grid neighbours that share its 64-token block.

    The single most interpretable number here. Raster order keeps the two
    left/right neighbours and nothing else; a compact tile should keep most of
    the in-frame four. Computed over the whole span, not sampled.
    """
    frames, height, width = grid
    total = frames * height * width
    block_of = torch.empty(total, dtype=torch.int64)
    block_of[perm] = block_ids(grid, video_start)

    index = torch.arange(total, dtype=torch.int64)
    t = index // (height * width)
    rem = index - t * (height * width)
    y = rem // width
    x = rem - y * width

    same = 0
    count = 0
    for dt, dy, dx in ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0),
                       (1, 0, 0), (-1, 0, 0)):
        nt, ny, nx = t + dt, y + dy, x + dx
        ok = ((nt >= 0) & (nt < frames) & (ny >= 0) & (ny < height)
              & (nx >= 0) & (nx < width))
        n_index = (nt * height * width + ny * width + nx).clamp(0, total - 1)
        same += int(((block_of[n_index] == block_of[index]) & ok).sum())
        count += int(ok.sum())
    return same / count


def in_frame_retention(perm, grid, video_start=0):
    """Same, restricted to the four in-frame neighbours.

    Separated because `2d_frame` cannot by construction keep a temporal
    neighbour, so a combined figure penalises it for doing what it says.
    """
    frames, height, width = grid
    total = frames * height * width
    block_of = torch.empty(total, dtype=torch.int64)
    block_of[perm] = block_ids(grid, video_start)
    index = torch.arange(total, dtype=torch.int64)
    t = index // (height * width)
    rem = index - t * (height * width)
    y = rem // width
    x = rem - y * width
    same = count = 0
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = y + dy, x + dx
        ok = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        n_index = (t * height * width + ny * width + nx).clamp(0, total - 1)
        same += int(((block_of[n_index] == block_of[index]) & ok).sum())
        count += int(ok.sum())
    return same / count


def orderings(grid, video_start, vendor):
    """Every ordering worth comparing, as the node actually applies them.

    `_perm_for` rotates the permutation by `(-video_start) % 64` because the
    kernels block from absolute row 0: without the rotation, a video span
    starting off a 64 boundary splits every Z-order cell across two blocks.
    **Measured here, the rotation does exactly its job** -- with it, block
    geometry is invariant to `video_start`; without it, a reference-laden graph
    at 1024x768 falls from fill 1.00 to 0.42. So the rows named `_noroll` are a
    control showing what the shipped code buys, not an alternative to run.
    """
    total = grid[0] * grid[1] * grid[2]
    pad = (-int(video_start)) % BLOCK_SIZE
    out = {"raster": torch.arange(total, dtype=torch.int64)}
    for curve in ("3d", "2d_frame"):
        perm, _ = vendor.morton_perm(grid, "cpu", curve)
        out[f"morton_{curve}"] = torch.roll(perm, pad) if pad else perm
        if pad:
            out[f"morton_{curve}_noroll"] = perm
    return out


def latent_t(length):
    """H3's frame-count -> latent-frame rule, the inverse of preflight.py."""
    n = int(length)
    return ((n - 5) // 17) * 5 + 2 if n > 5 else 2


def describe(grid, video_start, vendor, show_map=False):
    frames, height, width = grid
    per_frame = height * width
    total = frames * per_frame
    print(f"\n{'=' * 78}")
    print(f"grid (t,h,w) = {grid}   {total:,} video tokens   "
          f"{per_frame} per frame")
    print(f"blocks of {BLOCK_SIZE}: {total / BLOCK_SIZE:.2f} total, "
          f"{per_frame / BLOCK_SIZE:.4f} per frame", end="")
    if per_frame % BLOCK_SIZE:
        print("   <- NOT an integer: frame boundaries fall inside blocks")
    else:
        print("   <- integer: every frame boundary is a block boundary")
    clean = grid[1] % 8 == 0 and grid[2] % 8 == 0
    print(f"latent dims {grid[1]} x {grid[2]}: "
          f"{'both multiples of 8 -> Z-order tiles are whole 8x8' if clean else 'NOT both multiples of 8 -> Z-order tiles are clipped ragged'}")
    if video_start:
        print(f"video_start = {video_start:,}  "
              f"-> block grid is offset by {video_start % BLOCK_SIZE} rows, "
              f"perm rolled by {(-video_start) % BLOCK_SIZE}")
    print(f"{'=' * 78}")

    header = (f"{'ordering':<24}{'frames/blk':>11}{'h x w extent':>15}"
              f"{'radius':>9}{'fill':>7}{'nbr':>7}{'in-frame nbr':>14}")
    print(header)
    print("-" * len(header))
    for name, perm in orderings(grid, video_start, vendor).items():
        s = block_stats(perm, grid, video_start)
        extent = f"{s['h_span'].mean():.1f} x {s['w_span'].mean():.1f}"
        print(f"{name:<24}{s['n_frames'].mean():>11.2f}{extent:>15}"
              f"{s['radius'].mean():>9.2f}{s['fill'].mean():>7.2f}"
              f"{neighbour_retention(perm, grid, video_start):>7.1%}"
              f"{in_frame_retention(perm, grid, video_start):>14.1%}")

    print("\n  frames/blk    distinct latent frames inside one 64-token block")
    print("  h x w extent  mean bounding box of a block, in latent cells")
    print("  radius        RMS distance from the block's spatial centroid; "
          "this is\n                the geometric proxy for how well one "
          "pooled vector\n                stands in for the block")
    print("  fill          64 / bounding-box volume; 1.00 is a solid brick")
    print("  nbr           share of a token's 6 grid neighbours in the same "
          "block")

    # The mean hides the tail, and on a ragged canvas the tail is the finding:
    # most blocks tighten a lot while a minority end up looser than raster's
    # worst. Those are the blocks one pooled vector cannot represent.
    print()
    raster_worst = float(block_stats(
        orderings(grid, video_start, vendor)["raster"], grid,
        video_start)["radius"].max())
    for name, perm in orderings(grid, video_start, vendor).items():
        s = block_stats(perm, grid, video_start)
        mixed = float((s["n_frames"] > 1).float().mean())
        worst = float(s["radius"].max())
        loose = float((s["radius"] > raster_worst).float().mean())
        print(f"  {name:<24} {mixed:>6.1%} of blocks hold >1 frame; "
              f"worst radius {worst:>5.1f}; {loose:>5.1%} of blocks looser "
              f"than raster's worst block")

    if show_map:
        print_map(grid, video_start, vendor)


def print_map(grid, video_start, vendor):
    """One frame, coloured by which 64-token block each cell belongs to.

    Reading it: each character is one latent cell. Cells sharing a character
    share a block. Raster order gives horizontal bands; a working Morton gives
    tiles. Ragged tiles are the thing to look for -- the same character
    appearing in two places that are not adjacent.
    """
    frames, height, width = grid
    total = frames * height * width
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    frame = min(2, frames - 1)
    for name, perm in orderings(grid, video_start, vendor).items():
        block_of = torch.empty(total, dtype=torch.int64)
        block_of[perm] = block_ids(grid, video_start)
        print(f"\n  --- {name}, latent frame {frame} "
              f"({height} rows x {width} cols) ---")
        base = frame * height * width
        for row in range(height):
            line = "".join(
                alphabet[int(block_of[base + row * width + col]) % len(alphabet)]
                for col in range(width))
            print(f"  {line}")


SHIPPED = [
    ("1344x768 t2v / image-ref canvas", 1344, 768),
    ("1024x768 video-ref canvas (REF_VIDEO_CANVAS)", 1024, 768),
    ("768x768 square probe", 768, 768),
    ("960x544 turbo home canvas", 960, 544),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--canvas", help="WIDTHxHEIGHT, e.g. 1344x768")
    ap.add_argument("--length", type=int, default=362,
                    help="pixel frame count (17n+5). 362 = LONG_LENGTH, the ceiling")
    ap.add_argument("--video-start", type=int, default=0,
                    help="absolute row the video span starts at. Nonzero on "
                         "any graph with references; see Preflight's output")
    ap.add_argument("--map", action="store_true",
                    help="print one latent frame as ASCII block ids")
    args = ap.parse_args()

    vendor = load_shipped_morton()

    # Cross-check the shipped permutation against an independent one before
    # printing anything derived from it. Small grid: this is O(n log n) in
    # Python and only has to prove the two agree, not scale.
    for curve in ("3d", "2d_frame"):
        small = (3, 6, 10)
        got, _ = vendor.morton_perm(small, "cpu", curve)
        want = _independent_perm(small, curve)
        if not torch.equal(got, want):
            raise SystemExit(
                f"FAIL: shipped morton_perm disagrees with the independent "
                f"implementation at grid {small}, curve {curve}. Every number "
                f"below would describe a permutation nobody runs.")
    print("ok  shipped morton_perm agrees with an independent implementation "
          "on both curves")

    if args.canvas:
        w, h = (int(v) for v in args.canvas.lower().split("x"))
        cases = [(f"{w}x{h}", w, h)]
    else:
        cases = SHIPPED

    for label, w, h in cases:
        grid = (latent_t(args.length), h // 32, w // 32)
        print(f"\n\n### {label}   {args.length} frames "
              f"({args.length / 24:.2f}s)")
        describe(grid, args.video_start, vendor, show_map=args.map)


if __name__ == "__main__":
    main()
