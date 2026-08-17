#!/usr/bin/env python3
"""Morton block geometry for every legal H3 canvas, ranked, and for every legal length.

`bench/analyze_morton.py` answers "what does this permutation do" on **one**
canvas at a time, and its four `SHIPPED` cases are the ones this repo happens to
render. That is the right tool for understanding the mechanism and the wrong one
for choosing a canvas, because the interesting question is comparative: of the
48 legal landscape canvases, which give Sol-Attn's router the tightest blocks,
and does the answer move with clip length.

## Why this exists as a script rather than a table someone typed

`docs/h3_input_impacts.md` carried the full ranking as hand-transcribed tables
with no committed source, which is exactly the drift CLAUDE.md's guiding
principle names: a number that can change as the project evolves does not belong
in prose, it belongs in a file the prose links to. `adapt_canvas` changing, the
vendored `morton_perm` changing, or the connectivity definition changing would
all silently invalidate those tables and nothing would say so. `--markdown`
regenerates them.

## What is exact here and what is not

**Exact**: the canvas set, which is enumerated from `adapt_canvas` itself rather
than read from a list; the permutation, which comes from the vendored node via
`analyze_morton.load_shipped_morton`; and every geometry figure, which is
computed from that permutation.

**Not a quality measurement, at all.** Everything here describes the *summary*
Sol-Attn's router reads, not the picture that comes out. Whether tighter blocks
reach the output is link 6 in `docs/morton.md`'s assumption chain and is
unverified at every canvas. Do not read the ranking as a quality ranking of
canvases.

## The two controls it runs before printing anything

1. **The shipped permutation against an independent implementation**, borrowed
   from `analyze_morton`. If they disagree, every number below describes a
   permutation nobody runs.
2. **Connectivity against `docs/morton.md`'s published four worst canvases.**
   That page's figures were produced by a different implementation than the one
   here, so reproducing them is independent confirmation rather than a tautology.
   This is the control that fires if the connectivity definition drifts, and it
   has been shown red by mutating `connected_frac` to use 26-neighbour adjacency.

## Cost

The connectivity pass is a per-block flood fill in Python: roughly two minutes
for 48 canvases at 362 frames. `--no-connectivity` skips it and leaves radius
and fill, which are tensor ops and near-instant.

Needs `PYTHONPATH` to reach ComfyUI, for `adapt_canvas`. No CUDA, no model, no
server.

    PYTHONPATH=/path/to/ComfyUI python bench/analyze_canvas_geometry.py
    PYTHONPATH=/path/to/ComfyUI python bench/analyze_canvas_geometry.py --markdown
    PYTHONPATH=/path/to/ComfyUI python bench/analyze_canvas_geometry.py --lengths 768x768
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import deque
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent

# Import the sibling by path rather than by name. `analyze_morton` is
# unambiguous today, but `docs/comfy_notes.md`'s `import nodes` trap is the same shape and
# the cost of being explicit here is one line.
_spec = importlib.util.spec_from_file_location(
    "_analyze_morton", BENCH / "analyze_morton.py")
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)

BLOCK_SIZE = am.BLOCK_SIZE

#: `docs/morton.md`'s four worst canvases on connectivity, Morton `3d`, 362
#: frames. Hand-copied from that page on purpose: the point of the control is
#: that these came from a different implementation. Percentages.
PUBLISHED_WORST_362 = {
    (1952, 544): 51.5,
    (1888, 544): 52.5,
    (1568, 672): 52.7,
    (1440, 736): 53.7,
}


def legal_canvases():
    """Every legal landscape/square canvas, from `adapt_canvas` itself.

    Enumerated rather than listed. A hand-maintained list would agree with
    `docs/h3_resolutions.md` forever and stop agreeing with the code the first
    time the area cap or the rounding moves.
    """
    from comfy_extras.nodes_minimax_h3 import adapt_canvas
    seen = set()
    steps = 30000
    for i in range(1, steps + 1):
        ratio = 1.0 + 3.0 * i / steps          # 1:1 through 4:1
        w, h = adapt_canvas(ratio, 1.0)
        if w >= h:
            seen.add((w, h))
    return sorted(seen, key=lambda c: (-c[1], -c[0]))


def connected_frac(t, y, x):
    """Share of blocks whose 64 tokens form one 6-neighbour-connected region.

    "Connected" is the property `docs/morton.md` reports and `block_stats` does
    not compute: radius and fill can both look healthy while a block is two
    tidy halves sitting in different parts of the frame, which is precisely the
    `2d_frame` failure. Face adjacency only -- a block touching at a corner is
    two pieces, because the pooled summary is an average over a region and a
    corner touch does not make one region.
    """
    ok = 0
    rows = list(zip(t.long().tolist(), y.long().tolist(), x.long().tolist()))
    for bt, by, bx in rows:
        cells = set(zip(bt, by, bx))
        start = next(iter(cells))
        seen = {start}
        queue = deque([start])
        while queue:
            ct, cy, cx = queue.popleft()
            for dt, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nb = (ct + dt, cy + dy, cx + dx)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(seen) == len(cells):
            ok += 1
    return ok / len(rows)


def full_blocks(perm, grid, video_start=0):
    """Per-block (t, y, x) coordinates for every complete 64-token block."""
    t, y, x = am.coords(perm, grid)
    blk = am.block_ids(grid, video_start)
    order = torch.argsort(blk, stable=True)
    blk, t, y, x = blk[order], t[order], y[order], x[order]
    counts = torch.bincount(blk - blk.min())
    full = counts == BLOCK_SIZE
    starts = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int64), counts[:-1]]), 0)[full]
    idx = starts[:, None] + torch.arange(BLOCK_SIZE)
    return t[idx], y[idx], x[idx]


def measure(vendor, w, h, length, curve="3d", connectivity=True):
    grid = (am.latent_t(length), h // 32, w // 32)
    perm, _ = vendor.morton_perm(grid, "cpu", curve)
    stats = am.block_stats(perm, grid, 0)
    row = dict(
        w=w, h=h, tok_w=w // 32, tok_h=h // 32,
        w_mod4=(w // 32) % 4, h_mod4=(h // 32) % 4,
        tok_per_frame=(w // 32) * (h // 32),
        radius=float(stats["radius"].mean()),
        fill=float(stats["fill"].mean()),
        conn=None,
    )
    if connectivity:
        row["conn"] = connected_frac(*full_blocks(perm, grid))
    return row


def run_controls(vendor):
    """Both controls. Raises rather than warning: a wrong number is worse than
    no number, and this script exists to be quoted."""
    for curve in ("3d", "2d_frame"):
        small = (3, 6, 10)
        got, _ = vendor.morton_perm(small, "cpu", curve)
        if not torch.equal(got, am._independent_perm(small, curve)):
            raise SystemExit(
                f"FAIL: shipped morton_perm disagrees with the independent "
                f"implementation at grid {small}, curve {curve}. Every number "
                f"below would describe a permutation nobody runs.")
    print("ok  shipped morton_perm agrees with an independent implementation")

    bad = []
    for (w, h), want in PUBLISHED_WORST_362.items():
        got = measure(vendor, w, h, 362)["conn"] * 100
        if abs(got - want) > 0.15:
            bad.append(f"{w}x{h}: got {got:.1f}%, docs/morton.md says {want}%")
    if bad:
        raise SystemExit(
            "FAIL: connectivity disagrees with docs/morton.md's published "
            "figures:\n  " + "\n  ".join(bad) +
            "\nEither this script's definition of connected drifted, or that "
            "page's numbers are stale. Do not quote either until it is "
            "resolved.")
    print(f"ok  connectivity reproduces docs/morton.md's {len(PUBLISHED_WORST_362)} "
          f"worst canvases at 362 frames\n")


def group_key(row):
    return (row["h_mod4"], row["w_mod4"])


def print_plain(rows, length, curve):
    print(f"### Morton `{curve}` over {len(rows)} legal canvases, "
          f"{length} frames (latent_t {am.latent_t(length)})\n")
    head = f"  {'canvas':<11} {'tok grid':<9} {'h%4':>3} {'w%4':>3} " \
           f"{'radius':>7} {'fill':>6} {'connected':>10} {'tok/frame':>10}"
    print(head)
    for r in sorted(rows, key=lambda r: r["radius"]):
        conn = f"{r['conn']:9.1%}" if r["conn"] is not None else f"{'n/a':>10}"
        print(f"  {f'{r['w']}x{r['h']}':<11} "
              f"{f'{r["tok_w"]}x{r["tok_h"]}':<9} "
              f"{r['h_mod4']:>3} {r['w_mod4']:>3} "
              f"{r['radius']:>7.3f} {r['fill']:>6.3f} {conn} "
              f"{r['tok_per_frame']:>10}")

    print("\n### Grouped by whether each token axis divides by 4\n")
    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    for key in sorted(groups):
        g = groups[key]
        conns = [x["conn"] for x in g if x["conn"] is not None]
        conn = (f"{min(conns):.1%}-{max(conns):.1%}" if conns else "n/a")
        print(f"  h%4={key[0]} w%4={key[1]}  n={len(g):>2}  "
              f"radius {min(x['radius'] for x in g):.2f}-"
              f"{max(x['radius'] for x in g):.2f}  conn {conn}")


def print_markdown(rows, length, curve):
    """The tables `docs/h3_input_impacts.md` carries, regenerated."""
    print(f"<!-- generated: analyze_canvas_geometry.py --length {length} "
          f"--curve {curve} -->\n")

    buckets = {"0,0": [], "0,x": [], "x,0": [], "x,x": []}
    for r in rows:
        h0, w0 = r["h_mod4"] == 0, r["w_mod4"] == 0
        buckets["0,0" if (h0 and w0) else
                "0,x" if h0 else
                "x,0" if w0 else "x,x"].append(r)

    print("| `h/32 % 4` | `w/32 % 4` | canvases | radius | connected |")
    print("|---|---|---|---|---|")
    labels = [("0,0", "**0**", "**0**"), ("0,x", "0", "1, 2, 3"),
              ("x,0", "1, 2, 3", "0"), ("x,x", "1, 2, 3", "1, 2, 3")]
    for key, hl, wl in labels:
        g = buckets[key]
        if not g:
            continue
        conns = [x["conn"] for x in g if x["conn"] is not None]
        rad = f"{min(x['radius'] for x in g):.2f} - {max(x['radius'] for x in g):.2f}"
        con = f"{min(conns):.1%} - {max(conns):.1%}" if conns else "n/a"
        if key == "0,0":
            print(f"| {hl} | {wl} | **{len(g)}** | **{rad}** | **{con}** |")
        else:
            print(f"| {hl} | {wl} | {len(g)} | {rad} | {con} |")

    ref = max(r["tok_per_frame"] for r in rows)
    print("\n| canvas | token grid | radius | fill | connected | tok/frame "
          "| attention vs 16:9 |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(buckets["0,0"], key=lambda r: r["radius"]):
        conn = f"{r['conn']:.1%}" if r["conn"] is not None else "n/a"
        attn = (r["tok_per_frame"] / 1008) ** 2
        print(f"| {r['w']}x{r['h']} | {r['tok_w']}x{r['tok_h']} "
              f"| {r['radius']:.3f} | {r['fill']:.3f} | {conn} "
              f"| {r['tok_per_frame']} | {attn:.2f}x |")
    print(f"\n<!-- widest canvas here is {ref} tok/frame; the attention column "
          f"is quoted against 16:9's 1008 -->")


def print_lengths(vendor, canvas, curve, connectivity):
    w, h = (int(v) for v in canvas.lower().split("x"))
    print(f"### Morton `{curve}` at {w}x{h}, every on-grid length\n")
    print("| frames | latent frames | `% 4` | radius | fill | connected |")
    print("|---|---|---|---|---|---|")
    for n in range(7, 22):
        length = 17 * n + 5
        lt = am.latent_t(length)
        row = measure(vendor, w, h, length, curve, connectivity)
        conn = f"{row['conn']:.1%}" if row["conn"] is not None else "n/a"
        mark = "**" if lt % 4 == 0 else ""
        print(f"| {mark}{length}{mark} | {mark}{lt}{mark} | {mark}{lt % 4}{mark} "
              f"| {row['radius']:.3f} | {row['fill']:.3f} | {conn} |")
    print("\nBold rows have the latent frame count divisible by 4, which is "
          "what a 4x4x4 brick wants on the time axis.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=int, default=362,
                    help="pixel frame count (17n+5). 362 = LONG_LENGTH")
    ap.add_argument("--curve", default="3d", choices=("3d", "2d_frame"),
                    help="which ordering to rank canvases under")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the tables docs/h3_input_impacts.md carries")
    ap.add_argument("--lengths", metavar="WxH",
                    help="sweep the length axis on one canvas instead")
    ap.add_argument("--no-connectivity", action="store_true",
                    help="skip the flood fill; radius and fill only, much faster")
    ap.add_argument("--skip-controls", action="store_true",
                    help="not for reporting. Only for iterating on this script")
    args = ap.parse_args()

    vendor = am.load_shipped_morton()
    connectivity = not args.no_connectivity

    if not args.skip_controls:
        if not connectivity:
            print("note: --no-connectivity also disables the published-figure "
                  "control, so the run is unvalidated\n")
            for curve in ("3d", "2d_frame"):
                small = (3, 6, 10)
                got, _ = vendor.morton_perm(small, "cpu", curve)
                if not torch.equal(got, am._independent_perm(small, curve)):
                    raise SystemExit("FAIL: morton_perm cross-check")
        else:
            run_controls(vendor)

    if args.lengths:
        print_lengths(vendor, args.lengths, args.curve, connectivity)
        return 0

    canvases = legal_canvases()
    rows = [measure(vendor, w, h, args.length, args.curve, connectivity)
            for w, h in canvases]
    if args.markdown:
        print_markdown(rows, args.length, args.curve)
    else:
        print_plain(rows, args.length, args.curve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
