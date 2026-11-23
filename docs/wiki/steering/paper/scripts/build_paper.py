#!/usr/bin/env python3
"""Build Paper v5 LaTeX (`main.tex`) and optional typeset HTML / PDF from `manuscript.md`.

Uses strictly repo-relative paths (zero hardcoded user/home paths):
- Default `--tex-only` (when neither `--html-out` nor `--pdf-out` is passed) uses Python stdlib only.
- Passing `--html-out` and/or `--pdf-out` renders crisp inline SVG math via `matplotlib.mathtext`
  and prints the camera-ready PDF via headless Chromium/Chrome.
"""
from __future__ import annotations

import argparse
import html
import io
from pathlib import Path
import re
import shutil
import subprocess


def _find_paper_dir() -> Path:
    return Path(__file__).resolve().parent.parent


PAPER_DIR = _find_paper_dir()

_MATH_CACHE: dict[tuple[str, bool], str] = {}
_FALLBACKS: list[tuple[str, str]] = []


def normalize_tex_for_mathtext(tex: str) -> str:
    """Normalize LaTeX macros not natively known to matplotlib.mathtext."""
    s = tex.strip()
    s = re.sub(r"\\operatorname\*?\{([^{}]+)\}", r"\\mathrm{\1}", s)
    s = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", s)
    s = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", s)
    s = re.sub(r"\\ne(?![a-zA-Z])", r"\\neq", s)
    s = s.replace(r"\arg\min", r"\mathrm{argmin}")
    s = s.replace(r"\bigoplus", r"\oplus")

    def _repl_text(m: re.Match[str]) -> str:
        inner = m.group(1).strip().replace("-", "{-}").replace(" ", r"\ ")
        return rf"\mathrm{{{inner}}}"

    s = re.sub(r"\\text\{([^{}]+)\}", _repl_text, s)
    replacements = (
        (r"\triangleq", r"\equiv"),
        (r"\succeq", r"\geq"),
        (r"\preceq", r"\leq"),
        (r"\implies", r"\Rightarrow"),
        (r"\iff", r"\Leftrightarrow"),
        (r"\mid", "|"),
        (r"\;", " "),
        (r"\,", " "),
        (r"\!", ""),
        (r"\qquad", r"\ \ \ \ "),
        (r"\quad", r"\ \ "),
        (r"\left", ""),
        (r"\right", ""),
        (r"\square", r"\blacksquare"),
    )
    for src, dst in replacements:
        s = s.replace(src, dst)
    s = re.sub(r"\\[bB]igg?[lr]?", "", s)
    while r"\underbrace{" in s:
        idx = s.find(r"\underbrace{")
        pos = idx + len(r"\underbrace{")
        depth = 1
        while pos < len(s) and depth > 0:
            if s[pos] == "{":
                depth += 1
            elif s[pos] == "}":
                depth -= 1
            pos += 1
        body = s[idx + len(r"\underbrace{") : pos - 1]
        rest = s[pos:]
        if rest.startswith("_{"):
            p2 = 2
            d2 = 1
            while p2 < len(rest) and d2 > 0:
                if rest[p2] == "{":
                    d2 += 1
                elif rest[p2] == "}":
                    d2 -= 1
                p2 += 1
            sub = rest[2 : p2 - 1]
            s = s[:idx] + rf"({body})_{{{sub}}}" + rest[p2:]
        else:
            s = s[:idx] + body + rest
    return s


def render_math_svg(tex: str, display: bool = False) -> str:
    """Render a LaTeX math string to an inline SVG via `matplotlib.mathtext`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = (tex, display)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    norm = normalize_tex_for_mathtext(tex)
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fontsize = 11.2 if display else 9.6
        fig.text(0, 0, f"${norm}$", fontsize=fontsize, color="#0f172a")
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.015, transparent=True)
        plt.close(fig)
        svg_txt = buf.getvalue().decode("utf-8")
        svg_start = svg_txt.find("<svg")
        if svg_start != -1:
            svg_txt = svg_txt[svg_start:]
        cls = "math-display" if display else "math-inline"
        res = f'<span class="{cls}">{svg_txt}</span>'
    except Exception as exc:
        plt.close("all")
        _FALLBACKS.append((tex, str(exc)))
        esc = html.escape(tex)
        cls = "math-display-fallback" if display else "math-inline-fallback"
        res = f'<code class="{cls}">{esc}</code>'
    _MATH_CACHE[key] = res
    return res


def _split_md_table_row(line: str) -> list[str]:
    """Split a Markdown table row on unescaped `|` outside `$...$` and `` `...` `` spans."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells: list[str] = []
    cur: list[str] = []
    in_code = False
    in_math = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            cur.append(ch)
            cur.append(s[i + 1])
            i += 2
            continue
        if ch == "`" and not in_math:
            in_code = not in_code
            cur.append(ch)
            i += 1
            continue
        if ch == "$" and not in_code:
            in_math = not in_math
            cur.append(ch)
            i += 1
            continue
        if ch == "|" and not in_code and not in_math:
            cells.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _extract_doc_meta(lines: list[str]) -> tuple[str, str, str, str]:
    """Extract title, subtitle, authors, and date from top lines of `manuscript.md`."""
    title = "Causal 4D Prelude Steering in a Single-Stream Joint Audio-Video Diffusion Transformer"
    subtitle = ""
    authors = "MiniMax H3 Explorations Research Hub"
    date = "2026-11-22"
    for ln in lines[:25]:
        if ln.startswith("# "):
            title = ln[2:].strip()
        elif ln.startswith("**Subtitle:**"):
            subtitle = ln.split("**Subtitle:**", 1)[1].strip().strip("*")
        elif ln.startswith("**Authors:**"):
            authors = ln.split("**Authors:**", 1)[1].strip()
        elif ln.startswith("**Date:**"):
            date = ln.split("**Date:**", 1)[1].strip()
    return title, subtitle, authors, date


def _sanitize_code_for_tex(inner: str) -> str:
    unicode_ascii = (
        ("×", "x"),
        ("→", "->"),
        ("←", "<-"),
        ("↔", "<->"),
        ("≥", ">="),
        ("≤", "<="),
        ("±", "+/-"),
        ("≈", "~="),
        ("≠", "!="),
        ("⊥", "perp"),
        ("∈", "in"),
        ("⊆", "<="),
        ("—", "--"),
        ("–", "-"),
        ("λ", "lam"),
        ("μ", "mu"),
        ("σ", "sigma"),
        ("τ", "tau"),
        ("θ", "theta"),
        ("Δ", "Delta"),
        ("Σ", "Sigma"),
    )
    for src, dst in unicode_ascii:
        inner = inner.replace(src, dst)
    for ch, rep in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\^{}"),
        ("~", r"\textasciitilde{}"),
    ):
        inner = inner.replace(ch, rep)
    return inner


def _plain_unicode_to_tex(txt: str) -> str:
    """Escape LaTeX specials and translate Unicode symbols in plain prose."""
    for ch, rep in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\^{}"),
    ):
        txt = txt.replace(ch, rep)
    unicode_tex = (
        ("×", r"$\times$"),
        ("→", r"$\to$"),
        ("←", r"$\leftarrow$"),
        ("↔", r"$\leftrightarrow$"),
        ("≥", r"$\ge$"),
        ("≤", r"$\le$"),
        ("±", r"$\pm$"),
        ("≈", r"$\approx$"),
        ("≠", r"$\ne$"),
        ("⊥", r"$\perp$"),
        ("∈", r"$\in$"),
        ("⊆", r"$\subseteq$"),
        ("·", r"$\cdot$"),
        ("§", r"\S{}"),
        ("—", "---"),
        ("–", "--"),
        ("−", "-"),
        ("“", "``"),
        ("”", "''"),
        ("’", "'"),
        ("μ", r"$\mu$"),
        ("λ", r"$\lambda$"),
        ("θ", r"$\theta$"),
        ("σ", r"$\sigma$"),
        ("τ", r"$\tau$"),
        ("Δ", r"$\Delta$"),
        ("Σ", r"$\Sigma$"),
    )
    for src, dst in unicode_tex:
        txt = txt.replace(src, dst)
    txt = txt.replace(" <-> ", r" $\leftrightarrow$ ")
    txt = txt.replace(" -> ", r" $\to$ ")
    return txt


def _inline_md_to_tex(s: str, *, in_amsthm: bool = False) -> str:
    """Convert inline Markdown (`links`, ``code``, `$math$`, **bold**, *em*) to LaTeX."""
    s = re.sub(r"(?<!!)\[([^\[\]]+)\]\([^()]+\)", r"\1", s)
    stashed: dict[str, str] = {}
    counter = 0

    def _stash(m: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        tok = m.group(0)
        key = f"INLINETOKEN{counter}END"
        if tok.startswith("$") and tok.endswith("$"):
            stashed[key] = tok
        else:
            stashed[key] = rf"\texttt{{{_sanitize_code_for_tex(tok[1:-1])}}}"
        return key

    s = re.sub(r"`[^`\n]+`|\$[^$\n]+\$", _stash, s)
    txt = _plain_unicode_to_tex(s)
    txt = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", txt)
    if in_amsthm:
        txt = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", txt)
    else:
        txt = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", txt)
    for key, val in stashed.items():
        txt = txt.replace(key, val)
    return txt


def build_full_latex(md_text: str, tex_out: Path | None = None) -> str:
    """Convert a Paper v5 Markdown manuscript into clean `pdflatex`/arXiv-ready LaTeX."""
    lines = md_text.splitlines()
    title, subtitle, authors, date = _extract_doc_meta(lines)
    title_tex = _inline_md_to_tex(title)
    if subtitle:
        title_tex += rf" \\ \vspace{{4pt}} \large\emph{{{_inline_md_to_tex(subtitle)}}}"

    tex_lines = [
        "% Complete LaTeX Manuscript Scaffold (Generated by docs/wiki/steering/paper/scripts/build_paper.py)",
        r"\documentclass[10pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=0.8in]{geometry}",
        r"\usepackage{amsmath,amssymb,amsthm,mathtools}",
        r"\usepackage{booktabs,longtable,array,tabularx}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{microtype}",
        r"\usepackage{enumitem}",
        r"\newtheorem{theorem}{Theorem}",
        r"\newtheorem{lemma}{Lemma}",
        r"\newtheorem{proposition}{Proposition}",
        rf"\title{{\textbf{{{title_tex}}}}}",
        rf"\author{{{_inline_md_to_tex(authors)}}}",
        rf"\date{{{_inline_md_to_tex(date)}}}",
        r"\begin{document}",
        r"\maketitle",
    ]

    meta_prefixes = (
        "**Subtitle:**",
        "**Date:**",
        "**Status:**",
        "**Authors:**",
        "**Home",
        "**Predecessor",
        "**Primary",
        "**Canonical",
        "**Supersedes",
        "**Reference",
        "**Empirical",
        "**Companion",
    )

    i = 0
    in_abstract = False
    in_list = False
    list_kind = "itemize"
    table_idx = 0
    cur_section = ""
    pending_table_caption = ""

    def _close_blocks() -> None:
        nonlocal in_list
        if in_list:
            tex_lines.append(rf"\end{{{list_kind}}}")
            in_list = False

    def _emit_inner_md_blocks(sub_lines: list[str], *, strip_outer_em: bool = False) -> None:
        """Emit prose, display math (`$$...$$`), and lists from a nested line sequence."""
        nonlocal in_list, list_kind
        j = 0
        while j < len(sub_lines):
            sln = sub_lines[j]
            sstrip = sln.strip()
            if sstrip.startswith("$$"):
                _close_blocks()
                mb = [sstrip]
                if not (sstrip.endswith("$$") and len(sstrip) > 4):
                    j += 1
                    while j < len(sub_lines):
                        mb.append(sub_lines[j].strip())
                        if sub_lines[j].strip().endswith("$$"):
                            break
                        j += 1
                raw_m = "\n".join(mb).strip()[2:-2].strip()
                tex_lines.extend([r"\[", raw_m, r"\]"])
                j += 1
                continue
            mul = re.match(r"^(\s*)-\s+(.*)$", sln)
            mol = re.match(r"^(\s*)\d+\.\s+(.*)$", sln)
            if mul or mol:
                want = "itemize" if mul else "enumerate"
                itxt = (mul.group(2) if mul else mol.group(2)).strip()
                if not in_list:
                    tex_lines.append(rf"\begin{{{want}}}[leftmargin=*]")
                    in_list = True
                    list_kind = want
                elif list_kind != want:
                    tex_lines.append(rf"\end{{{list_kind}}}")
                    tex_lines.append(rf"\begin{{{want}}}[leftmargin=*]")
                    list_kind = want
                tex_lines.append(rf"\item {_inline_md_to_tex(itxt, in_amsthm=strip_outer_em)}")
                j += 1
                continue
            if not sstrip:
                _close_blocks()
                tex_lines.append("")
                j += 1
                continue
            tex_lines.append(_inline_md_to_tex(sstrip, in_amsthm=strip_outer_em))
            j += 1
        _close_blocks()

    while i < len(lines):
        ln = lines[i]
        if i < 20 and ln.startswith("# "):
            i += 1
            continue
        if i < 25 and any(ln.startswith(p) for p in meta_prefixes):
            i += 1
            continue
        if ln.strip() == "---":
            if in_abstract:
                tex_lines.append(r"\end{abstract}")
                in_abstract = False
            _close_blocks()
            i += 1
            continue

        # Fenced code or algorithm box
        if ln.startswith("```"):
            _close_blocks()
            i += 1
            c_buf: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                c_buf.append(lines[i])
                i += 1
            tex_lines.append(r"\begin{verbatim}")
            tex_lines.extend(c_buf)
            tex_lines.append(r"\end{verbatim}")
            i += 1
            continue

        # Display math $$...$$
        if ln.strip().startswith("$$"):
            _close_blocks()
            m_buf = [ln.strip()]
            if not (ln.strip().endswith("$$") and len(ln.strip()) > 4):
                i += 1
                while i < len(lines):
                    m_buf.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        break
                    i += 1
            raw_m = "\n".join(m_buf).strip()[2:-2].strip()
            tex_lines.append(r"\[")
            tex_lines.append(raw_m)
            tex_lines.append(r"\]")
            i += 1
            continue

        # HTML table-caption directive (used for LaTeX \caption without leaking into .tex)
        m_cap = re.match(r"^\s*<!--\s*table-caption:\s*(.*?)\s*-->\s*$", ln)
        if m_cap:
            pending_table_caption = m_cap.group(1).strip()
            i += 1
            continue

        # Markdown Table
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-\|]+\|\s*$", lines[i + 1]):
            _close_blocks()
            headers = _split_md_table_row(ln)
            ncol = len(headers)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _split_md_table_row(lines[i])
                while len(cells) < ncol:
                    cells.append("")
                rows.append(cells[:ncol])
                i += 1
            col_spec = "@{}" + "X" * ncol + "@{}"
            table_idx += 1
            cap_text = pending_table_caption or (cur_section if cur_section else f"Summary Table {table_idx}")
            pending_table_caption = ""
            tex_lines.append(r"\begin{table}[htbp]")
            tex_lines.append(r"\centering")
            tex_lines.append(rf"\caption{{{_inline_md_to_tex(cap_text)}}}")
            tex_lines.append(r"\scriptsize")
            tex_lines.append(rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}")
            tex_lines.append(r"\toprule")
            tex_lines.append(" & ".join(_inline_md_to_tex(h) for h in headers) + r" \\")
            tex_lines.append(r"\midrule")
            for r_cells in rows:
                tex_lines.append(" & ".join(_inline_md_to_tex(c) for c in r_cells) + r" \\")
            tex_lines.append(r"\bottomrule")
            tex_lines.append(r"\end{tabularx}")
            tex_lines.append(r"\end{table}")
            continue

        # Image ![alt](src)
        m_img = re.match(r"^!\[([^\[\]]*)\]\(([^()]+)\)\s*$", ln.strip())
        if m_img:
            _close_blocks()
            raw_cap = re.sub(r"^Figure\s+\d+\s*:\s*", "", m_img.group(1).strip())
            cap = _inline_md_to_tex(raw_cap)
            src_name = Path(m_img.group(2)).name
            src_pdf = Path(m_img.group(2)).with_suffix(".pdf").name
            tex_lines.append(r"\begin{figure}[htbp]\centering")
            tex_lines.append(
                rf"% Convert figures/{src_name} -> figures/{src_pdf} via rsvg-convert/inkscape before pdflatex"
            )
            tex_lines.append(
                rf"\IfFileExists{{figures/{src_pdf}}}{{\includegraphics[width=0.96\textwidth]{{figures/{src_pdf}}}}}"
                rf"{{\fbox{{\parbox{{0.92\textwidth}}{{\centering Vector figure asset: \texttt{{figures/{_sanitize_code_for_tex(src_name)}}}}}}}}}"
            )
            tex_lines.append(rf"\caption{{{cap}}}")
            tex_lines.append(r"\end{figure}")
            i += 1
            continue

        # Headings
        if ln.startswith("## Abstract"):
            _close_blocks()
            tex_lines.append(r"\begin{abstract}")
            in_abstract = True
            i += 1
            continue
        if ln.startswith("## "):
            if in_abstract:
                tex_lines.append(r"\end{abstract}")
                in_abstract = False
            _close_blocks()
            sec_title = re.sub(r"^(?:\d+\.|§\d+(?:\.\d+)?)\s*", "", ln[3:].strip())
            cur_section = sec_title
            pending_table_caption = ""
            tex_lines.append(rf"\section{{{_inline_md_to_tex(sec_title)}}}")
            i += 1
            continue
        if ln.startswith("### "):
            _close_blocks()
            sub_title = re.sub(r"^(?:\d+\.\d+|§\d+\.\d+)\s*", "", ln[4:].strip())
            cur_section = sub_title
            pending_table_caption = ""
            tex_lines.append(rf"\subsection{{{_inline_md_to_tex(sub_title)}}}")
            i += 1
            continue
        if ln.startswith("#### "):
            _close_blocks()
            tex_lines.append(rf"\subsubsection*{{{_inline_md_to_tex(ln[5:].strip())}}}")
            i += 1
            continue

        # Blockquote or Theorem / Lemma / Proposition environment
        if ln.startswith(">"):
            _close_blocks()
            q_lines: list[str] = []
            while i < len(lines) and lines[i].startswith(">"):
                q_lines.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            while q_lines and not q_lines[0].strip():
                q_lines.pop(0)
            first_q = q_lines[0].strip() if q_lines else ""
            first_q_plain = re.sub(r"(?<!!)\[([^\[\]]+)\]\([^()]+\)", r"\1", first_q)
            m_env = re.match(
                r"^\*\*(Theorem|Lemma|Proposition)\s+\d+(?:\s*\(([^)]+)\))?[\.:]?\*\*\s*(.*)$",
                first_q_plain,
            )
            if m_env:
                env_name = m_env.group(1).lower()
                env_opt = (m_env.group(2) or "").strip()
                rest0 = m_env.group(3).strip()
                if rest0:
                    q_lines[0] = rest0
                else:
                    q_lines.pop(0)
                while q_lines and not q_lines[0].strip():
                    q_lines.pop(0)
                if env_opt:
                    tex_lines.append(rf"\begin{{{env_name}}}[{_inline_md_to_tex(env_opt)}]")
                else:
                    tex_lines.append(rf"\begin{{{env_name}}}")
                _emit_inner_md_blocks(q_lines, strip_outer_em=True)
                tex_lines.append(rf"\end{{{env_name}}}")
            else:
                tex_lines.append(r"\begin{quote}")
                _emit_inner_md_blocks(q_lines)
                tex_lines.append(r"\end{quote}")
            continue

        # Lists
        m_ul = re.match(r"^(\s*)-\s+(.*)$", ln)
        m_ol = re.match(r"^(\s*)\d+\.\s+(.*)$", ln)
        if m_ul or m_ol:
            want_kind = "itemize" if m_ul else "enumerate"
            item_txt = (m_ul.group(2) if m_ul else m_ol.group(2)).strip()
            if not in_list:
                tex_lines.append(rf"\begin{{{want_kind}}}[leftmargin=*]")
                in_list = True
                list_kind = want_kind
            elif list_kind != want_kind:
                tex_lines.append(rf"\end{{{list_kind}}}")
                tex_lines.append(rf"\begin{{{want_kind}}}[leftmargin=*]")
                list_kind = want_kind
            tex_lines.append(rf"\item {_inline_md_to_tex(item_txt)}")
            i += 1
            continue

        if not ln.strip():
            _close_blocks()
            tex_lines.append("")
            i += 1
            continue

        tex_lines.append(_inline_md_to_tex(ln))
        i += 1

    _close_blocks()
    if in_abstract:
        tex_lines.append(r"\end{abstract}")
    tex_lines.append(r"\end{document}")
    tex_src = "\n".join(tex_lines) + "\n"
    if tex_out is not None:
        tex_out.parent.mkdir(parents=True, exist_ok=True)
        tex_out.write_text(tex_src, encoding="utf-8")
    return tex_src


def build_html_source(
    md_text: str,
    md_path: Path,
    figures_dir: Path,
    relative_asset_urls: bool = False,
) -> str:
    """Render Markdown + vector SVG math + SVG figures into standalone HTML."""
    from markdown_it import MarkdownIt

    _MATH_CACHE.clear()
    _FALLBACKS.clear()

    lines = md_text.splitlines()
    title, _subtitle, _authors, _date = _extract_doc_meta(lines)
    body_md = "\n".join(lines)
    body_md = re.sub(r"(?<!!)\[([^\[\]]+)\]\([^()]+\)", r"\1", body_md)

    placeholders: dict[str, str] = {}
    p_counter = 0

    def _stash_display(m: re.Match[str]) -> str:
        nonlocal p_counter
        p_counter += 1
        tok = f"MATHDISPLAYTOKEN{p_counter}ENDTOKEN"
        placeholders[tok] = f'<div class="eq-block">{render_math_svg(m.group(1), display=True)}</div>'
        return f"\n\n{tok}\n\n"

    code_blocks: dict[str, str] = {}
    c_counter = 0

    def _stash_code(m: re.Match[str]) -> str:
        nonlocal c_counter
        c_counter += 1
        tok = f"CODEBLOCKTOKEN{c_counter}ENDTOKEN"
        content = html.escape(m.group(2).strip("\n"))
        cls = "algo-box" if "Algorithm " in content else "code-box"
        code_blocks[tok] = f'<pre class="{cls}"><code>{content}</code></pre>'
        return f"\n\n{tok}\n\n"

    body_md = re.sub(r"```([^\n]*)\n(.*?)```", _stash_code, body_md, flags=re.DOTALL)
    body_md = re.sub(r"\$\$(.*?)\$\$", _stash_display, body_md, flags=re.DOTALL)

    inline_codes: dict[str, str] = {}
    ic_counter = 0

    def _stash_inline_code(m: re.Match[str]) -> str:
        nonlocal ic_counter
        ic_counter += 1
        tok = f"INLINECODETOKEN{ic_counter}ENDTOKEN"
        inline_codes[tok] = f"<code>{html.escape(m.group(1))}</code>"
        return tok

    body_md = re.sub(r"`([^`\n]+)`", _stash_inline_code, body_md)

    def _stash_inline_math(m: re.Match[str]) -> str:
        nonlocal p_counter
        p_counter += 1
        tok = f"MATHINLINETOKEN{p_counter}ENDTOKEN"
        placeholders[tok] = render_math_svg(m.group(1), display=False)
        return tok

    body_md = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", _stash_inline_math, body_md)
    body_md = body_md.replace(" -> ", " → ")

    table_captions: dict[str, str] = {}
    cap_counter = 0

    def _stash_table_caption(m: re.Match[str]) -> str:
        nonlocal cap_counter
        cap_counter += 1
        tok = f"TABLECAPTIONTOKEN{cap_counter}ENDTOKEN"
        table_captions[tok] = m.group(1).strip()
        return f"\n\n{tok}\n\n"

    body_md = re.sub(
        r"^\s*<!--\s*table-caption:\s*(.*?)\s*-->\s*$",
        _stash_table_caption,
        body_md,
        flags=re.MULTILINE,
    )

    def _rewrite_img(m: re.Match[str]) -> str:
        alt = html.escape(m.group(1))
        src = m.group(2).strip()
        cand = (md_path.parent / src).resolve()
        if not cand.is_file():
            cand = (figures_dir / Path(src).name).resolve()
        url = f"figures/{Path(src).name}" if relative_asset_urls else cand.as_uri()
        return (
            f'<figure class="paper-fig">'
            f'<img src="{url}" alt="{alt}"/>'
            f'<figcaption>{alt}</figcaption>'
            f"</figure>"
        )

    body_md = re.sub(r"!\[([^\[\]]*)\]\(([^()]+)\)", _rewrite_img, body_md)
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    rendered_html = md.render(body_md)

    for tok, cap_raw in table_captions.items():
        cap_html = md.renderInline(cap_raw)
        p_tok = f"<p>{tok}</p>"
        rendered_html, n_sub = re.subn(
            re.escape(p_tok) + r"\s*<table>",
            f"<table>\n<caption>{cap_html}</caption>",
            rendered_html,
            count=1,
        )
        if n_sub == 0:
            rendered_html = rendered_html.replace(p_tok, f'<p class="table-caption">{cap_html}</p>')

    for tok, val in code_blocks.items():
        rendered_html = rendered_html.replace(f"<p>{tok}</p>", val).replace(tok, val)
    for tok, val in placeholders.items():
        rendered_html = rendered_html.replace(f"<p>{tok}</p>", val).replace(tok, val)
    for tok, val in inline_codes.items():
        rendered_html = rendered_html.replace(tok, val)

    css = """
    @page { size: letter; margin: 0.62in 0.68in 0.68in 0.68in; }
    * { box-sizing: border-box; }
    body { font-family: Georgia, "Times New Roman", serif; font-size: 9.8pt; line-height: 1.44; color: #111827; margin: 0; padding: 0; background: #ffffff; }
    h1 { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 16.5pt; line-height: 1.24; font-weight: 800; color: #0f172a; text-align: center; margin: 0 0 8px 0; padding-bottom: 10px; border-bottom: 2px solid #0f172a; }
    h1 + p { text-align: center; font-family: Inter, sans-serif; font-size: 8.8pt; color: #475569; margin: 4px 0 14px 0; }
    h2 { font-family: Inter, sans-serif; font-size: 12pt; font-weight: 700; color: #0f172a; margin: 18px 0 7px 0; padding-bottom: 3px; border-bottom: 1px solid #cbd5e1; break-after: avoid; }
    h3 { font-family: Inter, sans-serif; font-size: 10.3pt; font-weight: 700; color: #1e293b; margin: 13px 0 5px 0; break-after: avoid; }
    h4 { font-family: Inter, sans-serif; font-size: 9.4pt; font-weight: 700; color: #334155; margin: 10px 0 4px 0; break-after: avoid; }
    p { margin: 0 0 7.5px 0; text-align: justify; }
    ul, ol { margin: 4px 0 9px 0; padding-left: 20px; }
    li { margin-bottom: 3.5px; text-align: justify; }
    blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid #0284c7; background: #f8fafc; color: #1e293b; }
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 12px 0; }
    code { font-family: "JetBrains Mono", "Fira Code", Consolas, monospace; font-size: 8.3pt; background: #f1f5f9; color: #0f172a; padding: 0.5px 3.5px; border-radius: 3px; }
    pre.algo-box, pre.code-box { background: #f8fafc; border: 1.2px solid #94a3b8; border-left: 4px solid #0284c7; border-radius: 5px; padding: 8px 10px; margin: 9px 0 11px 0; overflow-x: hidden; white-space: pre-wrap; break-inside: avoid; }
    pre.algo-box code, pre.code-box code { background: transparent; padding: 0; font-size: 7.9pt; line-height: 1.34; }
    table { width: 100%; border-collapse: collapse; margin: 8px 0 12px 0; font-family: Inter, sans-serif; font-size: 7.8pt; line-height: 1.28; break-inside: avoid; }
    table caption { font-family: Inter, sans-serif; font-size: 8.3pt; font-weight: 700; color: #0f172a; text-align: left; margin-bottom: 5px; caption-side: top; }
    thead th { background: #f1f5f9; color: #0f172a; font-weight: 700; text-align: left; padding: 4.5px 5.5px; border-top: 1.5px solid #0f172a; border-bottom: 1px solid #475569; }
    tbody td { padding: 4px 5.5px; border-bottom: 0.6px solid #cbd5e1; vertical-align: top; }
    tbody tr:last-child td { border-bottom: 1.5px solid #0f172a; }
    figure.paper-fig { margin: 10px 0 12px 0; padding: 7px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; break-inside: avoid; text-align: center; }
    figure.paper-fig img { max-width: 100%; max-height: 3.6in; height: auto; border-radius: 4px; display: block; margin: 0 auto; }
    figcaption { font-family: Inter, sans-serif; font-size: 8.0pt; line-height: 1.34; color: #334155; text-align: left; margin-top: 6px; }
    .eq-block { margin: 8px 0 9px 0; text-align: center; break-inside: avoid; }
    .math-display svg { max-width: 98%; height: auto; vertical-align: middle; }
    .math-inline svg { height: auto; max-width: 100%; zoom: 0.84; vertical-align: middle; }
    """
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n"
        f"<title>{html.escape(title)}</title>\n<style>{css}</style>\n</head>\n"
        f"<body>\n{rendered_html}\n</body>\n</html>\n"
    )


def print_pdf_from_html(html_path: Path, pdf_out: Path) -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise RuntimeError("Chrome or Chromium binary is required on PATH to print PDF from HTML")
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_out}",
            html_path.resolve().as_uri(),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build Paper v5 LaTeX (`main.tex`) and optional HTML/PDF.")
    ap.add_argument("--md", type=Path, default=PAPER_DIR / "manuscript.md", help="Input Markdown manuscript path")
    ap.add_argument("--figures-dir", type=Path, default=PAPER_DIR / "figures", help="Vector SVG figures directory")
    ap.add_argument("--tex-out", type=Path, default=PAPER_DIR / "main.tex", help="Output LaTeX path")
    ap.add_argument("--html-out", type=Path, default=None, help="Optional HTML output path")
    ap.add_argument("--pdf-out", type=Path, default=None, help="Optional PDF output path (requires --html-out or writes temp sibling)")
    ap.add_argument("--tex-only", action="store_true", help="Write `main.tex` only and exit")
    args = ap.parse_args(argv)

    md_path = args.md.resolve()
    md_text = md_path.read_text(encoding="utf-8")
    tex_src = build_full_latex(md_text, tex_out=args.tex_out.resolve())
    print(f"Wrote {args.tex_out} ({len(tex_src.splitlines())} lines, {len(tex_src.encode('utf-8'))} B)")

    if args.tex_only or (args.html_out is None and args.pdf_out is None):
        return 0

    html_out = (args.html_out or args.pdf_out.with_suffix(".html")).resolve()
    html_src = build_html_source(md_text, md_path=md_path, figures_dir=args.figures_dir.resolve())
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html_src, encoding="utf-8")
    print(f"Wrote {html_out} ({len(html_src.encode('utf-8'))} B; math cached={len(_MATH_CACHE)}, fallbacks={len(_FALLBACKS)})")

    if args.pdf_out is not None:
        pdf_out = args.pdf_out.resolve()
        print_pdf_from_html(html_out, pdf_out)
        print(f"Wrote {pdf_out} ({pdf_out.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
