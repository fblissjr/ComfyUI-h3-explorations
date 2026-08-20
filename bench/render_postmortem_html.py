"""Render a postmortem markdown file to the self-contained HTML the plugin specifies.

The markdown is the postmortem; this is a transform of it, not a second
analysis. Findings and wording come from the file and nothing here re-derives
one. What it adds is the visual guide: the figures in `bench/gen_figures.py`,
each drawn from a shipped record and each spliced in directly after the finding
it illustrates, with an index of them after the header.

    python bench/render_postmortem_html.py \\
        internal/postmortems/2026-08-20_session_distilled-regime-day.md

writes the `.html` beside the `.md` and is safe to re-run: the whole page is
regenerated from the markdown and the records every time, so an edit to either
shows up on the next run. Record paths resolve from this file, not the working
directory.

Figures attach by `Figure.anchor`, a substring of the finding they belong to. An
anchor that matches nothing is a hard error rather than a figure silently
dropped to the end of the page, because a figure that quietly stops sitting
beside its finding is exactly the drift this file exists to avoid.

Constraints the plugin's `references/html-render.md` sets, and this file keeps:
no script, no external `src`/`href`/`@import`, empty sections stay visible,
citations stay visible, annotations stay distinguishable from original findings.
CPU only, no GPU, no server.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_figures import Figure, postmortem_figures  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

STYLE = """
:root {
  --bg: #fdfdfc; --fg: #1c1c1a; --muted: #6b6b66; --rule: #e2e2dd;
  --accent: #7a5c2e; --panel: #f5f4f0;
  --signal: #2b6cb0; --good: #0f9488; --warn: #c2410c; --alt: #9333ea;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16161a; --fg: #e6e6e1; --muted: #9a9a94; --rule: #2e2e34;
    --accent: #d9b978; --panel: #1f1f25;
    --signal: #4b90d6; --good: #17ab9a; --warn: #e2683c; --alt: #b070f0;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 3rem 1.25rem 6rem; background: var(--bg); color: var(--fg);
  font: 16px/1.65 ui-serif, Georgia, "Times New Roman", serif;
  -webkit-text-size-adjust: 100%;
}
main { max-width: 44rem; margin: 0 auto; }
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 1.5rem; letter-spacing: -0.01em; }
h2 {
  font-size: 1.15rem; margin: 2.75rem 0 0.85rem; padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--rule); letter-spacing: -0.005em;
}
p { margin: 0 0 1rem; }
code {
  font: 0.86em/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--panel); padding: 0.12em 0.34em; border-radius: 3px;
  word-break: break-word;
}
.meta {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
  padding: 1rem 1.25rem; margin: 0 0 2.5rem;
  font: 0.86rem/1.6 ui-sans-serif, system-ui, sans-serif;
}
.meta .summary {
  font: 1rem/1.55 ui-serif, Georgia, serif; margin: 0 0 0.9rem;
  padding-bottom: 0.85rem; border-bottom: 1px solid var(--rule);
}
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1rem; margin: 0; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; }
.meta .artifacts { margin: 0.85rem 0 0; padding-top: 0.85rem; border-top: 1px solid var(--rule); }
.meta .artifacts ul { margin: 0.35rem 0 0; padding-left: 1.1rem; }
.meta .artifacts li { margin: 0.15rem 0; }
.nothing { color: var(--muted); font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 0 0 1rem; font-size: 0.94rem; }
th, td { text-align: left; vertical-align: top; padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--rule); }
th { font: 600 0.82rem/1.4 ui-sans-serif, system-ui, sans-serif; color: var(--muted);
     text-transform: uppercase; letter-spacing: 0.04em; }
.annotation {
  border-left: 3px solid var(--accent); background: var(--panel);
  padding: 0.75rem 1rem; margin: 0 0 1rem; font-size: 0.94rem;
}
.annotation strong { color: var(--accent); }
ul, ol { margin: 0 0 1rem; padding-left: 1.35rem; }
li { margin: 0.3rem 0; }

/* the visual guide */
.guide {
  border: 1px solid var(--rule); border-radius: 6px; padding: 0.9rem 1.25rem;
  margin: 0 0 2.5rem; font: 0.86rem/1.6 ui-sans-serif, system-ui, sans-serif;
}
.guide h2 { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted); border: 0; margin: 0 0 0.4rem; padding: 0;
  font-family: ui-sans-serif, system-ui, sans-serif; }
.guide ol { margin: 0; padding-left: 1.2rem; }
.guide a { color: var(--signal); }
figure.viz {
  margin: 1.6rem 0 2rem; padding: 1rem 1.1rem 0.6rem; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 6px;
}
@media (min-width: 62rem) {
  figure.viz { width: 54rem; margin-left: calc(50% - 27rem); }
}
figure.viz svg { width: 100%; height: auto; display: block; color: var(--fg); }
.figtitle {
  font: 600 0.78rem/1.4 ui-sans-serif, system-ui, sans-serif; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.6rem;
}
.figcap { font: 0.85rem/1.6 ui-sans-serif, system-ui, sans-serif; margin: 0.5rem 0 0;
  border-top: 1px solid var(--rule); padding-top: 0.7rem; }
.figcap p { margin: 0 0 0.6rem; }
.fignote { color: var(--muted); margin: 0 0 0.6rem; }
.fignote ol { margin: 0.3rem 0 0; padding-left: 1.4rem; }
.fignote li { margin: 0.05rem 0; }
.fignote .t { color: var(--fg); }
.src { color: var(--muted); font-size: 0.8rem; margin-top: 0.5rem; }
.src ul { margin: 0.2rem 0 0; padding-left: 1.1rem; list-style: square; }
.src li { margin: 0.05rem 0; }
@media (max-width: 34rem) {
  body { padding: 2rem 1rem 4rem; }
  .meta dl { grid-template-columns: 1fr; gap: 0.1rem; }
  .meta dt { margin-top: 0.5rem; }
}
"""

META_ROWS = [("mode", "Mode"), ("scope", "Scope"), ("date", "Written"),
             ("range", "Range"), ("supersedes", "Supersedes")]


# --------------------------------------------------------------------------
# markdown, only as far as this house style goes
# --------------------------------------------------------------------------

def esc(s: str) -> str:
    return html.escape(s, quote=False)


def inline(s: str) -> str:
    """The three inline forms this house style uses.

    Code spans come out first and go back last, so a `**bold**` that wraps one
    still pairs; splitting on code first left those asterisks literal, which is
    how the defect this replaced looked on the page.
    """
    codes = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    t = esc(re.sub(r"`([^`]+)`", stash, s))
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?![\w*])", r"<em>\1</em>", t)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: f"<code>{esc(codes[int(m.group(1))])}</code>", t)


def parse_front(text: str):
    assert text.startswith("---\n"), "no frontmatter"
    end = text.index("\n---\n", 3)
    body = text[end + 5:]
    meta, artifacts, key = {}, [], None
    for lineval in text[4:end].splitlines():
        if lineval.startswith("  - "):
            artifacts.append(lineval[4:].strip())
            continue
        if lineval.startswith("    ") and key:
            meta[key] += " " + lineval.strip()
            continue
        if ":" in lineval:
            key, _, val = lineval.partition(":")
            key = key.strip()
            meta[key] = val.strip()
    meta.pop("artifacts", None)
    return meta, artifacts, body


def split_blocks(md: str):
    """(kind, payload) blocks: p, ul, ol, table, quote, empty."""
    blocks, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        lineval = lines[i]
        if not lineval.strip():
            i += 1
            continue
        if lineval.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            blocks.append(("table", rows))
            continue
        if lineval.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i][1:].lstrip() if len(lines[i]) > 1 else "")
                i += 1
            blocks.append(("quote", "\n".join(quote)))
            continue
        m = re.match(r"^(-|\d+\.)\s+(.*)$", lineval)
        if m:
            kind = "ul" if m.group(1) == "-" else "ol"
            items, cur = [], m.group(2)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    i += 1
                    if i < len(lines) and re.match(r"^(-|\d+\.)\s+", lines[i]):
                        continue
                    break
                m2 = re.match(r"^(-|\d+\.)\s+(.*)$", nxt)
                if m2:
                    items.append(cur)
                    cur = m2.group(2)
                elif nxt.startswith("  "):
                    cur += " " + nxt.strip()
                else:
                    break
                i += 1
            items.append(cur)
            blocks.append((kind, items))
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(-|\d+\.|\||>|#)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(para)))
    return blocks


def parse_body(body: str):
    """(title, [(heading, blocks)]) with the lead paragraphs under heading ''."""
    lines = body.splitlines()
    title = next(l[2:].strip() for l in lines if l.startswith("# "))
    sections, cur_head, cur = [], "", []
    for lineval in lines:
        if lineval.startswith("# "):
            continue
        if lineval.startswith("## "):
            sections.append((cur_head, "\n".join(cur)))
            cur_head, cur = lineval[3:].strip(), []
            continue
        cur.append(lineval)
    sections.append((cur_head, "\n".join(cur)))
    return title, [(h, split_blocks(md)) for h, md in sections]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_block(kind, payload) -> str:
    if kind == "p":
        if payload.strip() == "Nothing.":
            return '<p class="nothing">Nothing.</p>'
        return f"<p>{inline(payload)}</p>"
    if kind == "table":
        head, rows = payload[0], payload[2:]
        th = "".join(f"<th>{inline(c)}</th>" for c in head)
        body = "".join(
            "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            for r in rows)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"
    if kind == "quote":
        return (f'<aside class="annotation">'
                f'{"".join(render_block(k, v) for k, v in split_blocks(payload))}'
                f'</aside>')
    return ""


def render_list(kind, items, figures_at) -> str:
    """A list, split open wherever a figure attaches to one of its items."""
    out, run = [], []

    def flush():
        if run:
            out.append(f"<{kind}>" +
                       "".join(f"<li>{inline(t)}</li>" for t in run) +
                       f"</{kind}>")
            run.clear()

    for idx, item in enumerate(items):
        run.append(item)
        if idx in figures_at:
            flush()
            for fig in figures_at[idx]:
                out.append(fig.to_html())
    flush()
    return "".join(out)


def render(md_path: Path, figures: list[Figure]) -> str:
    text = md_path.read_text()
    meta, artifacts, body = parse_front(text)
    title, sections = parse_body(body)

    unplaced = {f.fid: f for f in figures if f.anchor}
    placed: dict[tuple[int, int], dict[int, list[Figure]]] = {}
    for si, (_, blocks) in enumerate(sections):
        for bi, (kind, payload) in enumerate(blocks):
            if kind not in ("ul", "ol"):
                continue
            for ii, item in enumerate(payload):
                for fig in list(unplaced.values()):
                    if fig.anchor in item:
                        placed.setdefault((si, bi), {}).setdefault(ii, []).append(fig)
                        unplaced.pop(fig.fid)
    if unplaced:
        raise SystemExit("figure anchors matched no finding: " +
                         ", ".join(f"{f.fid} -> {f.anchor!r}"
                                   for f in unplaced.values()))

    parts = [f"<h1>{inline(title)}</h1>", '<header class="meta">']
    if "summary" in meta:
        parts.append(f'<p class="summary">{inline(meta["summary"])}</p>')
    parts.append("<dl>")
    for key, label in META_ROWS:
        if key in meta:
            val = inline(meta[key])
            if key == "range":
                val = f"<code>{esc(meta[key])}</code>"
            parts.append(f"<dt>{label}</dt><dd>{val}</dd>")
    parts.append("</dl>")
    items = "".join(f"<li><code>{esc(a)}</code></li>" for a in artifacts)
    parts.append(f'<div class="artifacts">Artifacts examined:<ul>{items}</ul></div>')
    parts.append("</header>")

    guide = "".join(f'<li><a href="#{esc(f.fid)}">{f.title}</a></li>'
                    for f in figures)
    parts.append('<nav class="guide"><h2>Visual guide</h2><ol>' + guide +
                 "</ol></nav>")

    free = [f for f in figures if not f.anchor]
    for si, (heading, blocks) in enumerate(sections):
        chunk = []
        if heading:
            chunk.append(f"<h2>{inline(heading)}</h2>")
        for bi, (kind, payload) in enumerate(blocks):
            if kind in ("ul", "ol"):
                chunk.append(render_list(kind, payload,
                                         placed.get((si, bi), {})))
            else:
                chunk.append(render_block(kind, payload))
        if si == 0:
            chunk.extend(f.to_html() for f in free)
            parts.append("".join(chunk))
        else:
            parts.append("<section>" + "".join(chunk) + "</section>")

    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{esc(title)}</title>\n<style>{STYLE}</style>\n</head>\n"
            "<body>\n<main>\n" + "\n".join(parts) + "\n</main>\n</body>\n</html>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("markdown", help="the postmortem .md")
    ap.add_argument("--out", default=None,
                    help="output path (default: the .md with .html)")
    args = ap.parse_args()

    md_path = Path(args.markdown)
    if not md_path.is_absolute():
        md_path = (REPO / md_path) if not md_path.exists() else md_path
    meta, _, _ = parse_front(md_path.read_text())
    commit_range = meta.get("range", "").split()[0]
    figures = postmortem_figures(commit_range)
    out = Path(args.out) if args.out else md_path.with_suffix(".html")
    out.write_text(render(md_path, figures))
    print(f"{out.relative_to(REPO) if out.is_relative_to(REPO) else out}  "
          f"{len(out.read_text())} bytes, {len(figures)} figures")


if __name__ == "__main__":
    main()
