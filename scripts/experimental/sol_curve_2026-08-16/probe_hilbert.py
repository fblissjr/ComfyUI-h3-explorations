"""Does the shipped Hilbert actually keep its defining property on a 24x42 grid?

verify_adjacency() only ever runs at side=64 (a full power-of-two square), which
is the case that cannot fail. The grid that runs is 24x42 clipped out of 64x64.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import sol_curves  # noqa: E402

BLOCK = 64


def order_within(height, width, fn):
    """(y, x) sequence for one frame under an ordering function."""
    idx = fn(height, width)
    return [(i // width, i % width) for i in idx]


def shipped_within(height, width):
    side = 1
    while side < max(height, width):
        side <<= 1
    return sorted(range(height * width),
                  key=lambda i: sol_curves.hilbert_d(i % width, i // width, side))


# --- generalized Hilbert ("gilbert"), works on any rectangle ---------------
def _sgn(x):
    return (x > 0) - (x < 0)


def _gilbert(x, y, ax, ay, bx, by, out):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = _sgn(ax), _sgn(ay)
    dbx, dby = _sgn(bx), _sgn(by)
    if h == 1:
        for _ in range(w):
            out.append((x, y))
            x, y = x + dax, y + day
        return
    if w == 1:
        for _ in range(h):
            out.append((x, y))
            x, y = x + dbx, y + dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)
    if 2 * w > 3 * h:
        if w2 % 2 and w > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        _gilbert(x, y, ax2, ay2, bx, by, out)
        _gilbert(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, out)
    else:
        if h2 % 2 and h > 2:
            bx2, by2 = bx2 + dbx, by2 + dby
        _gilbert(x, y, bx2, by2, ax2, ay2, out)
        _gilbert(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, out)
        _gilbert(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                 -bx2, -by2, -(ax - ax2), -(ay - ay2), out)


def gilbert_pts(height, width):
    out = []
    if width >= height:
        _gilbert(0, 0, width, 0, 0, height, out)
    else:
        _gilbert(0, 0, 0, height, width, 0, out)
    return out


def gilbert_within(height, width):
    return [y * width + x for x, y in gilbert_pts(height, width)]


def raster_within(height, width):
    return list(range(height * width))


# --- metrics ---------------------------------------------------------------
def nonadjacent_steps(pts):
    return sum(1 for a, b in zip(pts, pts[1:])
               if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1)


def full_perm(within, frames, area):
    return torch.tensor([f * area + i for f in range(frames) for i in within],
                        dtype=torch.int64)


def full_perm_serpentine(within, frames, area, height, width):
    """Reverse the curve on odd frames so frame f's exit touches f+1's entry."""
    rev = list(reversed(within))
    out = []
    for f in range(frames):
        seq = within if f % 2 == 0 else rev
        out.extend(f * area + i for i in seq)
    return torch.tensor(out, dtype=torch.int64)


def connected_fraction(perm, grid, video_start):
    """Fraction of whole 64-token blocks whose cells form one 6-connected region."""
    frames, height, width = grid
    total = frames * height * width
    blk = (video_start + torch.arange(total)) // BLOCK
    order = torch.argsort(blk, stable=True)
    blk_s, perm_s = blk[order], perm[order]
    counts = torch.bincount(blk_s - blk_s.min())
    starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), counts[:-1]]), 0)
    ok = whole = 0
    radii = []
    for b in range(len(counts)):
        if counts[b] != BLOCK:
            continue
        whole += 1
        cells = perm_s[starts[b]:starts[b] + BLOCK].tolist()
        pts = set()
        for c in cells:
            t, rem = divmod(c, height * width)
            y, x = divmod(rem, width)
            pts.add((t, y, x))
        ys = torch.tensor([p[1] for p in pts], dtype=torch.float32)
        xs = torch.tensor([p[2] for p in pts], dtype=torch.float32)
        radii.append(float(torch.sqrt(((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2).mean())))
        seen = {next(iter(pts))}
        stack = [next(iter(pts))]
        while stack:
            t, y, x = stack.pop()
            for dt, dy, dx in ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)):
                n = (t + dt, y + dy, x + dx)
                if n in pts and n not in seen:
                    seen.add(n)
                    stack.append(n)
        ok += len(seen) == len(pts)
    return ok / whole, sum(radii) / len(radii), whole


def main():
    height, width, frames = 24, 42, 37
    area = height * width
    print(f"grid {frames}x{height}x{width}, {area} tokens/frame, "
          f"{area / BLOCK:.2f} blocks/frame\n")

    ship = order_within(height, width, shipped_within)
    gil = order_within(height, width, gilbert_within)
    ras = order_within(height, width, raster_within)
    print("non-adjacent consecutive steps within one frame "
          "(0 is the defining Hilbert property):")
    print(f"  shipped hilbert (clip 64x64 -> 24x42): {nonadjacent_steps(ship)} of {area - 1}")
    print(f"  generalized hilbert (gilbert):         {nonadjacent_steps(gil)} of {area - 1}")
    print(f"  raster:                                {nonadjacent_steps(ras)} of {area - 1}")
    print(f"  shipped, at side=64 full square:       {sol_curves.verify_adjacency(64)} "
          f"of {64 * 64 - 1}   <- the case verify_adjacency tests\n")

    grid = (frames, height, width)
    arms = {
        "raster": full_perm(raster_within(height, width), frames, area),
        "hilbert (shipped)": full_perm(shipped_within(height, width), frames, area),
        "gilbert": full_perm(gilbert_within(height, width), frames, area),
        "gilbert + serpentine frames":
            full_perm_serpentine(gilbert_within(height, width), frames, area, height, width),
    }
    for start in (0, 530):
        print(f"video_start={start} (pad {(-start) % BLOCK}):")
        for name, perm in arms.items():
            pad = (-start) % BLOCK
            p = torch.roll(perm, pad) if pad else perm
            frac, rad, whole = connected_fraction(p, grid, start)
            print(f"  {name:30s} connected {frac:6.1%}  mean radius {rad:5.2f}  "
                  f"({whole} whole blocks)")
        print()


if __name__ == "__main__":
    main()
