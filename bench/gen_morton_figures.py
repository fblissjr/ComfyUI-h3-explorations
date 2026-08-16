"""SVG block maps for the Morton explainer page, from the shipped permutation.

`analyze_morton.py` answers the question in numbers and ASCII. This draws the
same thing for a page someone else reads: one latent frame, every patch, shaded
by which 64-token block it belongs to.

Every label is derived from the drawn geometry rather than typed, because a
hand-written caption on a generated figure is exactly how a picture and its
description drift apart. Cell fills use `currentColor` with varying opacity and
the highlight uses `var(--signal)`, so both figures follow the host page's
theme instead of baking one in.

Writes `fig1.svg` and `fig2.svg` next to this file's `--out` directory; splice
them into the page yourself. Needs torch, no CUDA and no model.
"""
import sys, importlib.util
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent.parent))
spec = importlib.util.spec_from_file_location("_v", REPO / "vendor" / "sol_attn_minimax.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
import torch

BS = 64
CELL = 9
SIGNAL = "var(--signal)"
FRAME = 2          # not frame 0: at 1344x768 a frame is 15.75 blocks, so frame 0
                   # is the one frame whose blocks happen to start aligned.


def block_map(grid, curve, frame):
    T, H, W = grid
    total = T * H * W
    if curve == "raster":
        perm = torch.arange(total, dtype=torch.int64)
    else:
        perm, _ = v.morton_perm(grid, "cpu", curve)
    block_of = torch.empty(total, dtype=torch.int64)
    block_of[perm] = torch.arange(total, dtype=torch.int64) // BS
    base = frame * H * W
    return {(r, c): int(block_of[base + r * W + c]) for r in range(H) for c in range(W)}


def runs(cells):
    """Maximal rectangles covering a block: contiguous column runs per row,
    merged vertically where a run repeats."""
    by_row = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    spans = []
    for r, cols in by_row.items():
        cols.sort()
        start = prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
                continue
            spans.append((r, start, prev))
            start = prev = c
        spans.append((r, start, prev))
    remaining, out = set(spans), []
    for r, c0, c1 in sorted(spans):
        if (r, c0, c1) not in remaining:
            continue
        h = 1
        while (r + h, c0, c1) in remaining:
            h += 1
        for k in range(h):
            remaining.discard((r + k, c0, c1))
        out.append((r, c0, c1, h))
    return out


def describe(cells):
    """Plain-language shape of the highlighted block, from its own cells."""
    pieces = runs(cells)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    bw, bh = max(cs) - min(cs) + 1, max(rs) - min(rs) + 1
    if len(pieces) == 1:
        _, c0, c1, h = pieces[0]
        return f"one solid {c1 - c0 + 1} x {h} block"
    sizes = {(c1 - c0 + 1, h) for _, c0, c1, h in pieces}
    if len(sizes) == 1:
        w, h = sizes.pop()
        return (f"{len(pieces)} separate {w} x {h} pieces, "
                f"{bw} x {bh} apart")
    return f"{len(pieces)} separate pieces, {bw} x {bh} apart"


def panel(grid, curve, highlight_at=(0, 10)):
    _, H, W = grid
    m = block_map(grid, curve, FRAME)
    hi = m[highlight_at]
    by_block = {}
    for (r, c), b in m.items():
        by_block.setdefault(b, []).append((r, c))
    w_px, h_px = W * CELL, H * CELL
    parts = [f'<rect x="0" y="0" width="{w_px}" height="{h_px}" fill="none" '
             f'stroke="currentColor" stroke-opacity=".3" stroke-width="1"/>']
    for b, cells in sorted(by_block.items()):
        for (r, c0, c1, h) in runs(cells):
            x, y = c0 * CELL, r * CELL
            wd, ht = (c1 - c0 + 1) * CELL, h * CELL
            if b == hi:
                parts.append(f'<rect x="{x}" y="{y}" width="{wd}" height="{ht}" '
                             f'fill="{SIGNAL}" fill-opacity=".88"/>')
            else:
                op = 0.05 + 0.055 * (b % 4)
                parts.append(f'<rect x="{x}" y="{y}" width="{wd}" height="{ht}" '
                             f'fill="currentColor" fill-opacity="{op:.3f}" '
                             f'stroke="currentColor" stroke-opacity=".22" '
                             f'stroke-width=".6"/>')
    return "".join(parts), w_px, h_px, describe(by_block[hi])


def figure(panels, gap=46, pad_top=36, pad_bot=28):
    xs, total_w, max_h = [], 0, 0
    for _, w, h, _, _, _ in panels:
        xs.append(total_w)
        total_w += w + gap
        max_h = max(max_h, h)
    total_w -= gap
    total_h = pad_top + max_h + pad_bot
    mono = "ui-monospace,SFMono-Regular,Menlo,monospace"
    out = [f'<svg viewBox="0 0 {total_w} {total_h}" role="img" '
           f'xmlns="http://www.w3.org/2000/svg">']
    for (body, w, h, shape, label, sub), x in zip(panels, xs):
        out.append(f'<text x="{x}" y="13" font-size="13" font-weight="700" '
                   f'fill="currentColor" font-family="{mono}">{label}</text>')
        out.append(f'<text x="{x}" y="28" font-size="11" fill="currentColor" '
                   f'fill-opacity=".6" font-family="{mono}">{sub}</text>')
        out.append(f'<g transform="translate({x},{pad_top})">{body}</g>')
        out.append(f'<text x="{x}" y="{pad_top + h + 19}" font-size="11.5" '
                   f'fill="{SIGNAL}" font-weight="700" font-family="{mono}">'
                   f'&#9632; {shape}</text>')
    out.append('</svg>')
    return "".join(out)


G1344 = (87, 24, 42)
G1024 = (87, 24, 32)

b1, w1, h1, s1 = panel(G1344, "raster")
b2, w2, h2, s2 = panel(G1344, "2d_frame")
b4, w4, h4, s4 = panel(G1024, "2d_frame")

fig1 = figure([
    (b1, w1, h1, s1, "raster order", "1344x768 &#183; 24 x 42 patches"),
    (b2, w2, h2, s2, "morton 2d_frame", "1344x768 &#183; 24 x 42 patches"),
])
fig2 = figure([
    (b2, w2, h2, s2, "1344x768", "24 x 42 &#183; 42 is not a multiple of 8"),
    (b4, w4, h4, s4, "1024x768", "24 x 32 &#183; both are multiples of 8"),
])

import argparse
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out", default=".", help="directory to write fig1.svg / fig2.svg")
SP = Path(_ap.parse_args().out)
(SP / 'fig1.svg').write_text(fig1)
(SP / 'fig2.svg').write_text(fig2)
print(f"raster 1344   {s1}")
print(f"morton 1344   {s2}")
print(f"morton 1024   {s4}")
print(f"bytes: fig1 {len(fig1)}, fig2 {len(fig2)}")
