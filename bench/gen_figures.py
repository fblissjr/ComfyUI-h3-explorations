"""Inline-SVG figure primitives, plus the figures for the 2026-08-20 postmortem.

Every figure here is drawn from a shipped record and captions itself from the
numbers it drew, because a hand-typed caption on a generated figure is exactly
how a picture and its description drift apart. Nothing takes a literal number as
an argument that it could read from the record instead.

Colours are CSS variables the host page defines (`--signal`, `--good`, `--warn`,
`--alt`) plus `currentColor` at varying opacity, so a figure follows the page's
light/dark theme rather than baking one in. The four hues pass the dataviz
validator's five checks against this page's light and dark panel surfaces; where
two series share a hue they are separated by a hatch texture and direct labels,
never by hue alone.

Two entry points:

- `python bench/gen_figures.py morton --out DIR` writes the Morton block maps.
  `bench/gen_morton_figures.py` is a shim over this and prints the same lines.
- `python bench/gen_figures.py postmortem --out FILE.html` writes the five
  postmortem figures as a standalone HTML fragment, for looking at them without
  building the page. The page itself is built by
  `bench/render_postmortem_html.py`, which imports `postmortem_figures()`.

CPU only. The Morton path needs torch (no CUDA, no model); the postmortem path
needs nothing but the standard library.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"
SIGNAL = "var(--signal)"
GOOD = "var(--good)"
WARN = "var(--warn)"
ALT = "var(--alt)"
INK = "currentColor"


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------------
# a figure and its provenance
# --------------------------------------------------------------------------

@dataclass
class Figure:
    """An SVG plus the caption derived from what it drew."""

    fid: str
    title: str
    svg: str
    caption: str
    sources: list[str] = field(default_factory=list)
    anchor: str = ""      # substring of the finding this figure sits beside
    notes: list[str] = field(default_factory=list)

    def to_html(self) -> str:
        src = ""
        if self.sources:
            items = "".join(f"<li><code>{esc(s)}</code></li>" for s in self.sources)
            src = f'<div class="src">Drawn from<ul>{items}</ul></div>'
        note = ""
        if self.notes:
            note = "".join(f'<div class="fignote">{n}</div>' for n in self.notes)
        return (f'<figure class="viz" id="{esc(self.fid)}">'
                f'<div class="figtitle">{self.title}</div>'
                f'{self.svg}'
                f'<figcaption class="figcap"><p>{self.caption}</p>{note}{src}'
                f'</figcaption></figure>')


# --------------------------------------------------------------------------
# svg primitives
# --------------------------------------------------------------------------

def _n(v: float) -> str:
    """Short fixed-point, so the SVG text stays diffable."""
    return f"{v:.2f}".rstrip("0").rstrip(".") if v == v else "0"


def text(x, y, s, size=11.5, fill=INK, op=1.0, anchor="start", weight=None,
         rotate=None, family=MONO) -> str:
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight else ""
    o = f' fill-opacity="{op}"' if op != 1.0 else ""
    r = f' transform="rotate({rotate},{_n(x)},{_n(y)})"' if rotate else ""
    return (f'<text x="{_n(x)}" y="{_n(y)}" font-size="{size}" fill="{fill}"{o}'
            f' font-family="{family}"{a}{w}{r}>{s}</text>')


def line(x1, y1, x2, y2, color=INK, op=0.35, w=1.0, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" '
            f'stroke="{color}" stroke-opacity="{op}" stroke-width="{w}"{d}/>')


def rect(x, y, w, h, fill, op=1.0, rx=0.0, stroke=None, sw=1.0) -> str:
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    r = f' rx="{_n(rx)}"' if rx else ""
    return (f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(max(w, 0))}" '
            f'height="{_n(max(h, 0))}" fill="{fill}" fill-opacity="{op}"{r}{s}/>')


def path(d, stroke, w=2.0, op=1.0, dash=None, fill="none") -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
            f'stroke-opacity="{op}" stroke-linejoin="round" '
            f'stroke-linecap="round"{da}/>')


HATCH_ID = "h3hatch"
HATCH_DEF = (f'<defs><pattern id="{HATCH_ID}" width="6" height="6" '
             f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
             f'<rect width="6" height="6" fill="currentColor" fill-opacity="0.06"/>'
             f'<line x1="0" y1="0" x2="0" y2="6" stroke="currentColor" '
             f'stroke-width="3" stroke-opacity="0.55"/></pattern></defs>')


def svg_doc(w: float, h: float, body: str, title: str, desc: str,
            defs: str = "") -> str:
    return (f'<svg viewBox="0 0 {_n(w)} {_n(h)}" role="img" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<title>{esc(title)}</title><desc>{esc(desc)}</desc>'
            f'{defs}{body}</svg>')


@dataclass
class Panel:
    body: str
    w: float
    h: float
    title: str = ""
    sub: str = ""


def compose(panels: list[Panel], title: str, desc: str, gap: float = 54,
            pad_top: float = 34, pad_bot: float = 10, defs: str = "") -> str:
    """Lay panels out in a row, each with its own heading."""
    xs, total_w, max_h = [], 0.0, 0.0
    for p in panels:
        xs.append(total_w)
        total_w += p.w + gap
        max_h = max(max_h, p.h)
    total_w -= gap
    out = []
    for p, x in zip(panels, xs):
        if p.title:
            out.append(text(x, 12, esc(p.title), size=12.5, weight=700))
        if p.sub:
            out.append(text(x, 26, esc(p.sub), size=10.5, op=0.62))
        out.append(f'<g transform="translate({_n(x)},{_n(pad_top)})">{p.body}</g>')
    return svg_doc(total_w, pad_top + max_h + pad_bot, "".join(out), title, desc,
                   defs)


def legend(entries, x=0.0, y=0.0, gap=None, size=10.5) -> str:
    """entries: (label, colour, hatched) laid out left to right."""
    out, cx = [], x
    for label, color, hatched in entries:
        out.append(rect(cx, y - 8, 10, 10, color, rx=2))
        if hatched:
            out.append(rect(cx, y - 8, 10, 10, f"url(#{HATCH_ID})", rx=2))
        out.append(text(cx + 15, y, esc(label), size=size, op=0.85))
        cx += (gap if gap else 22 + 6.3 * len(label))
    return "".join(out)


# --------------------------------------------------------------------------
# figure forms
# --------------------------------------------------------------------------

def signed_bars(labels, values, w=400, h=232, ymin=-1.0, ymax=1.0,
                pos=GOOD, neg=WARN, fmt="{:+.3f}") -> Panel:
    """Magnitude with polarity: bars from a zero line, hue by sign."""
    pad_l, pad_t, pad_b = 30, 12, 34
    plot_w, plot_h = w - pad_l, h - pad_t - pad_b
    y0 = pad_t + plot_h * (ymax / (ymax - ymin))
    step = plot_w / len(values)
    bw = step * 0.58
    out = [line(pad_l, pad_t, pad_l, pad_t + plot_h, op=0.3)]
    for t in (ymax, (ymax + ymin) / 2, ymin):
        yy = pad_t + plot_h * (ymax - t) / (ymax - ymin)
        out.append(line(pad_l, yy, w, yy, op=0.14 if t else 0.42))
        out.append(text(pad_l - 5, yy + 3.5, f"{t:+.0f}" if t else "0",
                        size=10, op=0.6, anchor="end"))
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = pad_l + step * i + (step - bw) / 2
        yv = pad_t + plot_h * (ymax - v) / (ymax - ymin)
        top, hh = min(y0, yv), abs(yv - y0)
        out.append(rect(cx, top, bw, hh, pos if v >= 0 else neg, rx=2))
        out.append(text(cx + bw / 2, top - 4 if v >= 0 else top + hh + 10,
                        esc(fmt.format(v)), size=9, op=0.85, anchor="middle"))
        out.append(text(cx + bw / 2, pad_t + plot_h + 26, esc(lab), size=10,
                        op=0.72, anchor="middle"))
    return Panel("".join(out), w, h)


def line_chart(xs, series, w=470, h=210, ymin=-1.0, ymax=1.0, xlabel="",
               yticks=(-1.0, -0.5, 0.0, 0.5, 1.0), pad_r=16) -> Panel:
    """series: list of dict(name, values, color, dash=None)."""
    pad_l, pad_b = 46, 34
    plot_w, plot_h = w - pad_l - pad_r, h - pad_b

    def px(i):
        return pad_l + plot_w * (xs[i] - xs[0]) / max(xs[-1] - xs[0], 1e-9)

    def py(v):
        return plot_h * (ymax - v) / (ymax - ymin)

    out = [line(pad_l, 0, pad_l, plot_h, op=0.3)]
    for t in yticks:
        out.append(line(pad_l, py(t), pad_l + plot_w, py(t),
                        op=0.42 if t == 0 else 0.13))
        out.append(text(pad_l - 6, py(t) + 3.5, f"{t:+.1f}" if t else "0",
                        size=10, op=0.6, anchor="end"))
    for tick in (xs[0], xs[len(xs) // 2], xs[-1]):
        i = xs.index(tick)
        out.append(text(px(i), plot_h + 14, esc(str(tick)), size=10, op=0.7,
                        anchor="middle"))
    if xlabel:
        out.append(text(pad_l + plot_w / 2, plot_h + 28, esc(xlabel), size=10,
                        op=0.6, anchor="middle"))
    for s_ in series:
        d = "M" + " L".join(f"{_n(px(i))},{_n(py(v))}"
                            for i, v in enumerate(s_["values"]))
        out.append(path(d, s_["color"], w=2.0, dash=s_.get("dash")))
    out.append(legend([(s_["name"], s_["color"], False) for s_ in series],
                      x=pad_l, y=plot_h + 46))
    return Panel("".join(out), w, h + 46)


def scatter_identity(x, y, w=330, h=300, xlabel="", ylabel="", color=SIGNAL,
                     highlight=None) -> Panel:
    """Agreement between two runs of the same measure: identity line, equal axes."""
    pad_l, pad_b = 46, 40
    plot_w, plot_h = w - pad_l, h - pad_b
    hi = max(max(x), max(y))
    top = hi * 1.08

    def sx(v):
        return pad_l + plot_w * v / top

    def sy(v):
        return plot_h - plot_h * v / top

    out = [line(pad_l, 0, pad_l, plot_h, op=0.3),
           line(pad_l, plot_h, w, plot_h, op=0.3)]
    ticks = [0, top / 2, top]
    for t in ticks:
        out.append(text(pad_l - 6, sy(t) + 3.5, f"{t:.2f}", size=9.5, op=0.6,
                        anchor="end"))
        out.append(text(sx(t), plot_h + 14, f"{t:.2f}", size=9.5, op=0.6,
                        anchor="middle"))
        out.append(line(pad_l, sy(t), w, sy(t), op=0.1))
    out.append(line(sx(0), sy(0), sx(top), sy(top), op=0.5, dash="4 3"))
    out.append(text(sx(top) - 6, sy(top) + 22, "y = x", size=9.5, op=0.6,
                    anchor="end"))
    for i, (a, b) in enumerate(zip(x, y)):
        big = highlight is not None and i == highlight
        out.append(f'<circle cx="{_n(sx(a))}" cy="{_n(sy(b))}" '
                   f'r="{4.6 if big else 3.4}" fill="{ALT if big else color}" '
                   f'fill-opacity="{1.0 if big else 0.72}" '
                   f'stroke="var(--panel)" stroke-width="1"/>')
        if big:
            out.append(text(sx(a) + 8, sy(b) - 6, f"head {i}", size=9.5, op=0.85,
                            fill=ALT))
    out.append(text(pad_l + plot_w / 2, plot_h + 32, esc(xlabel), size=10,
                    op=0.7, anchor="middle"))
    out.append(text(11, plot_h / 2, esc(ylabel), size=10, op=0.7,
                    anchor="middle", rotate=-90))
    return Panel("".join(out), w, h)


def slope_chart(left, right, left_label, right_label, w=250, h=300,
                color=SIGNAL, highlight=None) -> Panel:
    """Paired values, one segment per item: shows whether the pairing holds."""
    pad_t, pad_b = 14, 40
    plot_h = h - pad_b - pad_t
    lo = min(min(left), min(right))
    hi = max(max(left), max(right))
    span = max(hi - lo, 1e-9)
    x1, x2 = 58.0, w - 58.0

    def sy(v):
        return pad_t + plot_h - plot_h * (v - lo) / span

    out = [line(x1, pad_t, x1, pad_t + plot_h, op=0.3),
           line(x2, pad_t, x2, pad_t + plot_h, op=0.3)]
    for t in (lo, (lo + hi) / 2, hi):
        out.append(text(x1 - 7, sy(t) + 3.5, f"{t:.2f}", size=9.5, op=0.6,
                        anchor="end"))
    for i, (a, b) in enumerate(zip(left, right)):
        big = highlight is not None and i == highlight
        out.append(line(x1, sy(a), x2, sy(b), color=ALT if big else color,
                        op=0.95 if big else 0.32, w=2.0 if big else 1.1))
        if big:
            out.append(text(x2 + 6, sy(b) + 3.5, f"head {i}", size=9.5, op=0.9,
                            fill=ALT))
    out.append(text(x1, pad_t + plot_h + 18, esc(left_label), size=10, op=0.8,
                    anchor="middle"))
    out.append(text(x2, pad_t + plot_h + 18, esc(right_label), size=10, op=0.8,
                    anchor="middle"))
    return Panel("".join(out), w, h)


def grouped_bars_log(groups, arms, w=820, row_h=17, group_gap=34, label_w=196,
                     floor=3e-4, top=3.0, markers=None, legend_entries=None,
                     axis_title="") -> Panel:
    """Horizontal grouped bars on a log axis.

    groups: list of (group_label, {arm_name: value})
    arms:   list of (arm_name, colour, hatched) in draw order
    markers: {group_label: (value, marker_label)} drawn as a per-group rule.
    """
    plot_w = w - label_w
    lo, hi = math.log10(floor), math.log10(top)
    head, block = 16, len(arms) * row_h
    content_h = len(groups) * (head + block + group_gap)
    axis_y = content_h - group_gap + 20

    def sx(v):
        v = max(v, floor)
        return label_w + plot_w * (math.log10(v) - lo) / (hi - lo)

    out = []
    for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
        d = 10 ** e
        if not floor <= d <= top:
            continue
        out.append(line(sx(d), 0, sx(d), axis_y, op=0.16))
        out.append(text(sx(d), axis_y + 15, f"{d:g}", size=9.5, op=0.6,
                        anchor="middle"))
    out.append(line(label_w, axis_y, w, axis_y, op=0.3))
    if axis_title:
        out.append(text(label_w + plot_w / 2, axis_y + 30, esc(axis_title),
                        size=10, op=0.62, anchor="middle"))

    y = 0.0
    for glabel, vals in groups:
        out.append(text(0, y + 10, esc(glabel), size=11, weight=700))
        y += head
        gy0 = y
        for name, color, hatched in arms:
            v = vals[name]
            bx = sx(v)
            out.append(rect(label_w, y + 2, bx - label_w, row_h - 5, color, rx=2))
            if hatched:
                out.append(rect(label_w, y + 2, bx - label_w, row_h - 5,
                                f"url(#{HATCH_ID})", rx=2))
            out.append(text(6, y + row_h - 5, esc(name), size=10, op=0.8))
            out.append(text(bx + 5, y + row_h - 5, f"{v:.4f}", size=9.5, op=0.7))
            y += row_h
        if markers and glabel in markers:
            mv, mlab = markers[glabel]
            mx = sx(mv)
            out.append(line(mx, gy0 - 12, mx, y + 2, color=INK, op=0.8, w=1.4,
                            dash="3 3"))
            out.append(text(mx, gy0 - 15, esc(mlab), size=9, op=0.85,
                            anchor="middle"))
        y += group_gap
    h_total = axis_y + 34
    if legend_entries:
        out.append(legend(legend_entries, x=6, y=h_total + 10))
        h_total += 20
    return Panel("".join(out), w, h_total)


def bars(labels, values, w=300, h=210, colors=None, fmt="{:.1f}", ylabel="",
         annotate=None, ymax=None) -> Panel:
    """Plain magnitude bars from a zero baseline."""
    pad_l, pad_b = 52, 34
    plot_w, plot_h = w - pad_l, h - pad_b
    top = ymax if ymax else max(values) * 1.16
    step = plot_w / len(values)
    bw = min(step * 0.5, 54)
    out = [line(pad_l, plot_h, w, plot_h, op=0.35)]
    for t in (0, top / 2, top):
        yy = plot_h - plot_h * t / top
        out.append(line(pad_l, yy, w, yy, op=0.12))
        out.append(text(pad_l - 6, yy + 3.5, f"{t:.0f}", size=9.5, op=0.6,
                        anchor="end"))
    for i, (lab, v) in enumerate(zip(labels, values)):
        c = (colors or [SIGNAL] * len(values))[i]
        x = pad_l + step * i + (step - bw) / 2
        hh = plot_h * v / top
        out.append(rect(x, plot_h - hh, bw, hh, c, rx=3))
        out.append(text(x + bw / 2, plot_h - hh - 5, esc(fmt.format(v)),
                        size=10, op=0.85, anchor="middle"))
        out.append(text(x + bw / 2, plot_h + 14, esc(lab), size=10, op=0.75,
                        anchor="middle"))
    if ylabel:
        out.append(text(11, plot_h / 2, esc(ylabel), size=10, op=0.7,
                        anchor="middle", rotate=-90))
    if annotate:
        out.append(text(pad_l + plot_w / 2, plot_h + 30, annotate, size=10.5,
                        op=0.9, anchor="middle"))
    return Panel("".join(out), w, h + 8)


def swimlane(lanes, t0, t1, markers, w=880, lane_h=42, pad_l=86, pad_t=52,
             tick_minutes=30) -> Panel:
    """lanes: list of (lane_label, [ (start_min, end_min, colour, label) ]).

    Times are minutes since midnight; a zero-length span is drawn as a tick.
    markers: list of (start_min, end_min, label, approximate, label_row). A
    zero-width marker is a dashed rule, a wider one a shaded band, and both are
    drawn across every lane.
    """
    plot_w = w - pad_l
    span = max(t1 - t0, 1)

    def sx(m):
        return pad_l + plot_w * (m - t0) / span

    out, y = [], pad_t
    lane_tops = {}
    for label, spans in lanes:
        lane_tops[label] = y
        out.append(rect(pad_l, y, plot_w, lane_h - 8, INK, op=0.035, rx=3))
        out.append(text(0, y + lane_h / 2 - 1, esc(label), size=10.5, op=0.8,
                        weight=700))
        y += lane_h
    bottom = y - 8

    for (ma, mb, mlabel, approx, row) in markers:
        xa, xb = sx(ma), sx(mb)
        ly = pad_t - 10 - 13 * row
        if xb - xa < 1.5:
            out.append(line(xa, ly + 3, xa, bottom + 4, color=WARN, op=0.7,
                            w=1.3, dash="4 3"))
        else:
            out.append(rect(xa, ly + 3, xb - xa, bottom + 1 - ly, WARN, op=0.13,
                            rx=2))
            out.append(line(xa, ly + 3, xa, bottom + 4, color=WARN, op=0.6,
                            w=1.1, dash="4 3"))
            out.append(line(xb, ly + 3, xb, bottom + 4, color=WARN, op=0.6,
                            w=1.1, dash="4 3"))
        if mlabel:
            out.append(text((xa + xb) / 2, ly,
                            esc(("~" if approx else "") + mlabel),
                            size=9.5, op=0.95, fill=WARN, anchor="middle"))

    for label, spans in lanes:
        ty = lane_tops[label]
        for (a, b, color, slabel) in spans:
            xa, xb = sx(a), sx(b)
            if xb - xa < 1.6:
                out.append(line(xa, ty + 3, xa, ty + lane_h - 11, color=color,
                                op=0.95, w=1.8))
            else:
                out.append(rect(xa, ty + 4, xb - xa, lane_h - 16, color, op=0.9,
                                rx=2))
            if slabel:
                out.append(text(xa, ty + lane_h - 13, esc(slabel), size=9,
                                op=0.85))

    for m in range(int(t0), int(t1) + 1):
        if m % tick_minutes:
            continue
        x = sx(m)
        out.append(line(x, bottom + 4, x, bottom + 9, op=0.4))
        out.append(text(x, bottom + 22, f"{m // 60:02d}:{m % 60:02d}", size=9.5,
                        op=0.65, anchor="middle"))
    out.append(line(pad_l, bottom + 4, w, bottom + 4, op=0.3))
    return Panel("".join(out), w, bottom + 34)


# --------------------------------------------------------------------------
# small statistics, computed here rather than quoted
# --------------------------------------------------------------------------

def spearman(a, b) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((r - ma) ** 2 for r in ra) *
                    sum((r - mb) ** 2 for r in rb))
    return num / den


def load(name: str):
    return json.loads((RESULTS / name).read_text())


def load_lines(name: str):
    return [json.loads(l) for l in (RESULTS / name).read_text().splitlines() if l.strip()]


def _mins(ts: str) -> float:
    d = datetime.fromisoformat(ts)
    return d.hour * 60 + d.minute + d.second / 60.0


# --------------------------------------------------------------------------
# the five postmortem figures
# --------------------------------------------------------------------------

def fig_basis_sign() -> Figure:
    rec = "2026-08-20_dit_internals.json"
    d = load(rec)
    cols = d["adaln_t_table"]["column_cos"]
    blocks = [b["block"] for b in d["per_block"]]
    wcos = [b["adaln"]["basis_dependent"]["w_cos"] for b in d["per_block"]]
    modt = [b["adaln"]["mod_t_cos"] for b in d["per_block"]]
    neg = [i for i, c in enumerate(cols) if c < 0]

    p1 = signed_bars([f"c{i}" for i in range(len(cols))], cols, w=400,
                     fmt="{:+.4f}")
    p1.title = "the shared time basis, column by column"
    p1.sub = "adaln_t_table.column_cos"
    p2 = line_chart(blocks, [
        {"name": "basis_dependent.w_cos, withdrawn", "values": wcos,
         "color": WARN},
        {"name": "adaln.mod_t_cos, kept", "values": modt, "color": GOOD},
    ], w=430, xlabel="block")
    p2.title = "the same 50 blocks, compared two ways"
    p2.sub = "per_block[].adaln"

    svg = compose([p1, p2],
                  "AdaLN cosine, coefficients against modulation output",
                  "Left: cosine between the two checkpoints' time-basis columns, "
                  "four of eight near minus one. Right: per-block cosine under "
                  "the withdrawn coefficient comparison and under the "
                  "modulation-output comparison.")
    cap = (f"{len(neg)} of the {len(cols)} <code>adaln_t_table</code> columns "
           f"(columns {neg[0]}&#8211;{neg[-1]}) come back at cosine near "
           f"&#8722;1, so the two checkpoints store the same time basis with "
           f"those columns sign-flipped. A coefficient matrix written against a "
           f"flipped basis inherits the flip: "
           f"<code>basis_dependent.w_cos</code> runs "
           f"{min(wcos):+.3f} to {max(wcos):+.3f} across every block, which is "
           f"the negative cosine the 18th read as \"AdaLN is replaced\". "
           f"Compared where the sign cancels, at the modulation output "
           f"over the same time grid, <code>mod_t_cos</code> runs "
           f"{min(modt):.4f} to {max(modt):.4f}. Same blocks, same files, "
           f"opposite conclusion.")
    return Figure("fig-basis-sign", "The basis sign, and what it did to the "
                  "comparison", svg, cap,
                  [f"bench/results/{rec} :: adaln_t_table.column_cos",
                   f"bench/results/{rec} :: per_block[].adaln.basis_dependent.w_cos",
                   f"bench/results/{rec} :: per_block[].adaln.mod_t_cos"],
                  anchor="AdaLN is replaced in every block")


def fig_block49() -> Figure:
    err_ref = "2026-08-19_sol_error_per_head.json"
    err_fl = "2026-08-20_sol_error_per_head_fl2va.json"
    mag_ref = "2026-08-20_head_magnitudes.json"
    mag_fl = "2026-08-20_head_magnitudes_fl2va.json"
    a, b = load(err_ref), load(err_fl)

    def row(rec, block, step):
        return [r for r in rec["rows"] if r["block"] == block and r["step"] == step][0]

    ra, rb = row(a, 49, 3), row(b, 49, 3)
    x, y = ra["per_head_quant"], rb["per_head_quant"]
    rho = spearman(x, y)

    ma, mb = load(mag_ref), load(mag_fl)
    ref_cap = [c for c in ma["captures"] if c.endswith("ref2va")][0]
    kr = [r for r in ma["rows"]
          if r["block"] == 49 and r["step"] == 3 and r["capture"] == ref_cap][0]["k"]["rms"]
    kf = [r for r in mb["rows"] if r["block"] == 49 and r["step"] == 3][0]["k"]["rms"]
    loud = max(range(len(kr)), key=lambda i: kr[i])
    ratio = [abs(kf[i] / kr[i] - 1) for i in range(len(kr))]
    worst = max(range(len(ratio)), key=lambda i: ratio[i])

    p1 = scatter_identity(x, y, w=316, h=300,
                          xlabel=f"ref2va  ({ra['heads_measured']} heads)",
                          ylabel="clean fl2va", highlight=loud)
    p1.title = "per-head INT8 quantization error"
    p1.sub = "rows[block 49, step 3].per_head_quant"
    p2 = slope_chart(kr, kf, "ref2va", "fl2va", w=236, h=300, highlight=loud)
    p2.title = "per-head K rms"
    p2.sub = "rows[block 49, step 3].k.rms"

    svg = compose([p1, p2], "Block 49's loud heads on both checkpoints",
                  "Left: each of the heads plotted with its ref2va error on x "
                  "and its clean-fl2va error on y, against the identity line. "
                  "Right: the same heads' key rms on the two captures, one "
                  "segment per head.")
    cap = (f"The morning's attribution rested on gains matching between "
           f"checkpoints; this is the control it lacked. Block 49 step 3, "
           f"seq_len {ra['seq_len']}, tau {a['tau']} both sides: the per-head "
           f"INT8 error on the ref2va capture ranks the heads the same way as "
           f"on a clean fl2va capture taken on the ref2va capture's own graph, "
           f"Spearman {rho:.3f} recomputed here from the two lists rather than "
           f"quoted. Head {loud} is the loudest key on both captures, and no "
           f"head's K rms moves by more than {max(ratio) * 100:.1f}% "
           f"(head {worst}). Present in both checkpoints went from inferred to "
           f"observed.")
    note = ('Both error records were produced by '
            '<code>bench/analyze_sol_error.py</code>; the fl2va one carries a '
            '<code>measured_note</code> because the producing script stamped a '
            'literal date, which is the defect section 2 records.')
    return Figure("fig-block49", "Block 49 is in both checkpoints", svg, cap,
                  [f"bench/results/{err_ref} :: rows[block 49, step 3].per_head_quant",
                   f"bench/results/{err_fl} :: rows[block 49, step 3].per_head_quant",
                   f"bench/results/{mag_ref} :: rows[].k.rms (ref2va capture)",
                   f"bench/results/{mag_fl} :: rows[].k.rms"],
                  anchor="The block-49 question got its missing control",
                  notes=[note])


def fig_power_pair() -> Figure:
    rec = "2026-08-20_power_limit_pair_verdict.json"
    d = load(rec)
    s = d["sampler_s"]
    tel = d["telemetry_during_sampling"]
    arms = ["p330", "p450"]
    secs = [v for a in arms for v in s[a]]
    labels = [f"{a} #{i + 1}" for a in arms for i in range(len(s[a]))]
    colors = [WARN, WARN, SIGNAL, SIGNAL]

    p1 = bars(labels, secs, w=340, colors=colors, fmt="{:.1f}",
              ylabel="sampler seconds",
              annotate=f"{d['cost_of_330W_frac'] * 100:.1f}% apart "
                       f"(cost_of_330W_frac)")
    p1.title = "sampler seconds, two timed runs per arm"
    p1.sub = "sampler_s"
    p2 = bars(["330 W", "450 W"], [tel["p330"]["pclk_p50"], tel["p450"]["pclk_p50"]],
              w=330, colors=[WARN, SIGNAL], fmt="{:.0f}", ylabel="MHz",
              annotate=f"{d['core_clock_delta_frac'] * 100:.1f}% apart "
                       f"(core_clock_delta_frac)")
    p2.title = "median core clock while sampling"
    p2.sub = "telemetry_during_sampling[].pclk_p50"

    svg = compose([p1, p2], "The 330 W cap against stock 450 W",
                  "Left: four timed sampler runs, two per power arm. Right: "
                  "the median core clock sampled alongside each arm.")
    cap = (f"One graph, seeds disjoint across arms so the node cache cannot "
           f"answer the question. Sampler mean {d['sampler_mean_s']['p330']} s "
           f"at 330 W against {d['sampler_mean_s']['p450']} s at 450 W, "
           f"within-arm spread {d['within_arm_spread_frac'] * 100:.2f}% "
           f"(<code>within_arm_spread_frac</code>), so the "
           f"{d['cost_of_330W_frac'] * 100:.1f}% gap is far outside the noise. "
           f"Median board power reads {tel['p330']['pwr_p50']:.0f} W and "
           f"{tel['p450']['pwr_p50']:.0f} W, and memory clock is identical on "
           f"both arms. The decision rule written on 2026-08-17 wanted either "
           f"~2% (bandwidth-bound) or the full clock delta (L2-bound); "
           f"{d['cost_of_330W_frac'] * 100:.1f}% against "
           f"{d['core_clock_delta_frac'] * 100:.1f}% is neither, so this "
           f"4-step workload is partly core-clock-bound.")
    return Figure("fig-power", "The power pair", svg, cap,
                  [f"bench/results/{rec} :: sampler_s, sampler_mean_s, "
                   f"within_arm_spread_frac, cost_of_330W_frac",
                   f"bench/results/{rec} :: telemetry_during_sampling[].pclk_p50, "
                   f".pwr_p50, .mclk_p50",
                   f"bench/results/{rec} :: core_clock_delta_frac, decision_rule"],
                  anchor="The plan critique found the defect before the card did")


CARD_SOURCES = [
    ("2026-08-20_sla_arms.jsonl", ALT, "SLA pair"),
    ("2026-08-20_power_limit_pair.jsonl", None, "power pair"),
    ("2026-08-20_session1_lora_file.jsonl", GOOD, "session 1"),
]


def fig_day_order(commit_range: str) -> Figure:
    spans, per_file = [], {}
    for name, color, label in CARD_SOURCES:
        rows = load_lines(name)
        per_file[name] = rows
        for r in rows:
            end = _mins(r["ts"])
            start = end - r["total_s"] / 60.0
            c = color
            if c is None:
                c = WARN if r["label"] == "p330" else SIGNAL
            spans.append((start, end, c, ""))

    log = subprocess.run(
        ["git", "log", "--format=%h %ad %s", "--date=format:%H:%M", commit_range],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    commits = []
    for lineval in log:
        h, hhmm, subject = lineval.split(" ", 2)
        hh, mm = hhmm.split(":")
        commits.append((int(hh) * 60 + int(mm), h, subject))
    commits.sort()
    commit_spans = [(m, m, INK, "") for m, _, _ in commits]

    # Owner moments. Only the plan approval is carried by a record (the
    # postmortem's own deviations table); the rest are approximate.
    markers = [(11 * 60 + 30, 11 * 60 + 30, "11:30 plan approved", True, 0),
               (12 * 60 + 23, 12 * 60 + 32,
                "12:23 to 12:32 restarts, sudo 330 to 450 W at 12:29", True, 1),
               (12 * 60 + 50, 12 * 60 + 50, "12:50 router-node restart", True, 0)]
    owner = [(a, b, WARN, "") for a, b, _, _, _ in markers]

    t0 = math.floor(min(s for s, _, _, _ in spans + commit_spans) / 30) * 30
    t1 = math.ceil(max(e for _, e, _, _ in spans + commit_spans) / 30) * 30
    p = swimlane([("card", spans), ("repo", commit_spans), ("owner", owner)],
                 t0, t1, markers)
    p.body += legend([("SLA pair", ALT, False), ("power pair, 330 W", WARN, False),
                      ("power pair, 450 W", SIGNAL, False),
                      ("session 1 renders", GOOD, False),
                      ("commit", INK, False)], x=86, y=p.h + 12)
    p.h += 22
    rows_html = "".join(
        f"<li><code>{h}</code> <span class=\"t\">{m // 60:02d}:{m % 60:02d}</span> "
        f"{esc(subject)}</li>" for m, h, subject in commits)

    pw = per_file["2026-08-20_power_limit_pair.jsonl"]
    gap_a = max(_mins(r["ts"]) for r in pw if r["label"] == "p330")
    gap_b = min(_mins(r["ts"]) - r["total_s"] / 60 for r in pw if r["label"] == "p450")

    svg = compose([p], "The day's order",
                  "Three lanes: what the card was doing, when the repo gained a "
                  "commit, and the owner's moments.")
    cap = (f"Card spans come from the render records' <code>ts</code> and "
           f"<code>total_s</code>: each bar runs from <code>ts</code> minus "
           f"<code>total_s</code> to <code>ts</code>, and consecutive rows abut "
           f"exactly, which is what identifies <code>ts</code> as the "
           f"completion time rather than the start. Repo ticks are the "
           f"{len(commits)} commits of <code>{esc(commit_range)}</code> at "
           f"their author times. The card idles between "
           f"{int(gap_a) // 60:02d}:{int(gap_a) % 60:02d} and "
           f"{int(gap_b) // 60:02d}:{int(gap_b) % 60:02d}, the gap the "
           f"sudo moment and the restarts sit in.")
    note = ("The owner lane is approximate. No record carries the sudo moment "
            "or the restarts; the times shown are the session's own account, "
            "bracketed by the p330-to-p450 gap the card lane makes visible. "
            "The plan approval time is the one the deviations table states. "
            "Read them as ordering, not as measurements.")
    return Figure("fig-day", "The day's order", svg, cap,
                  ["bench/results/2026-08-20_sla_arms.jsonl :: ts, total_s, label",
                   "bench/results/2026-08-20_power_limit_pair.jsonl :: ts, total_s, label",
                   "bench/results/2026-08-20_session1_lora_file.jsonl :: ts, total_s",
                   f"git log --format='%h %ad %s' --date=format:%H:%M {commit_range}"],
                  notes=[note, f'<ol class="commits">{rows_html}</ol>'])


def postmortem_figures(commit_range: str) -> list[Figure]:
    """The 2026-08-20 postmortem's figures, in reading order."""
    return [fig_day_order(commit_range), fig_basis_sign(), fig_block49(),
            fig_power_pair()]


# --------------------------------------------------------------------------
# the Morton block maps, moved here verbatim from gen_morton_figures.py
# --------------------------------------------------------------------------

BS = 64
CELL = 9
FRAME = 2          # not frame 0: at 1344x768 a frame is 15.75 blocks, so frame 0
                   # is the one frame whose blocks happen to start aligned.


def _vendor():
    """The LIVE Sol node. Named `_vendor` when that WAS the vendored file;
    repointed 2026-08-31, since the vendored copy is a pristine reference
    ComfyUI does not load and cannot run on the installed kernel."""
    from _live_sol import live_sol
    return live_sol()


def block_map(grid, curve, frame):
    import torch
    T, H, W = grid
    total = T * H * W
    if curve == "raster":
        perm = torch.arange(total, dtype=torch.int64)
    else:
        perm, _ = _vendor().morton_perm(grid, "cpu", curve)
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


def morton_panel(grid, curve, highlight_at=(0, 10)):
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


def morton_figure(panels, gap=46, pad_top=36, pad_bot=28):
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


def morton_figures():
    """(fig1, fig2, [shape lines]) for the Morton explainer page."""
    b1, w1, h1, s1 = morton_panel(G1344, "raster")
    b2, w2, h2, s2 = morton_panel(G1344, "2d_frame")
    b4, w4, h4, s4 = morton_panel(G1024, "2d_frame")
    fig1 = morton_figure([
        (b1, w1, h1, s1, "raster order", "1344x768 &#183; 24 x 42 patches"),
        (b2, w2, h2, s2, "morton 2d_frame", "1344x768 &#183; 24 x 42 patches"),
    ])
    fig2 = morton_figure([
        (b2, w2, h2, s2, "1344x768", "24 x 42 &#183; 42 is not a multiple of 8"),
        (b4, w4, h4, s4, "1024x768", "24 x 32 &#183; both are multiples of 8"),
    ])
    return fig1, fig2, [f"raster 1344   {s1}", f"morton 1344   {s2}",
                        f"morton 1024   {s4}"]


def write_morton(out_dir: Path) -> list[str]:
    fig1, fig2, lines = morton_figures()
    (out_dir / "fig1.svg").write_text(fig1)
    (out_dir / "fig2.svg").write_text(fig2)
    return lines + [f"bytes: fig1 {len(fig1)}, fig2 {len(fig2)}"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

FRAGMENT_CSS = """
:root { --bg:#fdfdfc; --fg:#1c1c1a; --panel:#f5f4f0; --rule:#e2e2dd;
        --signal:#2b6cb0; --good:#0f9488; --warn:#c2410c; --alt:#9333ea; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16161a; --fg:#e6e6e1; --panel:#1f1f25; --rule:#2e2e34;
          --signal:#4b90d6; --good:#17ab9a; --warn:#e2683c; --alt:#b070f0; }
}
body { background:var(--bg); color:var(--fg); font:15px/1.6 ui-sans-serif,
       system-ui, sans-serif; margin:0; padding:2rem; }
figure { margin:0 0 3rem; }
svg { width:100%; height:auto; display:block; }
code { font-family:ui-monospace,Menlo,monospace; font-size:.9em; }
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("morton", help="write fig1.svg / fig2.svg")
    m.add_argument("--out", default=".", help="directory to write into")
    p = sub.add_parser("postmortem", help="write the postmortem figures alone")
    p.add_argument("--out", default="figures.html")
    p.add_argument("--range", default="5264c66..878e8f9",
                   help="commit range for the timeline lane")
    args = ap.parse_args()

    if args.cmd == "morton":
        for lineval in write_morton(Path(args.out)):
            print(lineval)
        return

    figs = postmortem_figures(args.range)
    body = "".join(f.to_html() for f in figs)
    Path(args.out).write_text(
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>figures</title><style>{FRAGMENT_CSS}</style></head><body>"
        f"{body}</body></html>")
    for f in figs:
        print(f"{f.fid:16s} {len(f.svg):7d} bytes  {f.title}")


if __name__ == "__main__":
    main()
