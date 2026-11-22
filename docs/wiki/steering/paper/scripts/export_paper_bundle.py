#!/usr/bin/env python3
"""Export a self-contained portable Paper v5 bundle under `$H3_OUTPUT_DIR/<slug>/`.

Copies tracked `manuscript.md`, `README.md`, `main.tex`, `figures/*.svg`, and tracked
Paper v5 scripts (`build_paper.py`, `make_v5_figures.py`), builds relative-URL HTML
and optional PDF, and skips missing untracked metrics/plates cleanly unless requested.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "control_vectors.py").is_file():
            return p
    return Path.cwd()


PAPER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(PAPER_DIR / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import build_paper  # noqa: E402


def _resolve_output_dir() -> Path:
    try:
        from workflows import h3_env  # noqa: E402

        h3_env.load_env()
    except Exception:
        pass
    out_base = os.environ.get("H3_OUTPUT_DIR", "").strip()
    if not out_base:
        raise RuntimeError("H3_OUTPUT_DIR is not set; source h3.env first")
    return Path(out_base)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export portable Paper v5 bundle to $H3_OUTPUT_DIR.")
    ap.add_argument("--slug", default="paper_4d_prelude_steering_v5", help="Output folder name under $H3_OUTPUT_DIR")
    ap.add_argument("--include-metrics", action="store_true", help="Copy available untracked JSON ledgers if present")
    ap.add_argument("--skip-pdf", action="store_true", help="Export HTML/LaTeX/figures without invoking Chrome PDF print")
    args = ap.parse_args(argv)

    out_root = _resolve_output_dir()
    bundle_dir = out_root / args.slug
    figs_out = bundle_dir / "figures"
    scripts_out = bundle_dir / "scripts"
    for d in (bundle_dir, figs_out, scripts_out):
        d.mkdir(parents=True, exist_ok=True)

    md_src = PAPER_DIR / "manuscript.md"
    readme_src = PAPER_DIR / "README.md"
    md_text = md_src.read_text(encoding="utf-8")

    shutil.copy2(md_src, bundle_dir / "manuscript.md")
    shutil.copy2(readme_src, bundle_dir / "README.md")
    build_paper.build_full_latex(md_text, tex_out=bundle_dir / "main.tex")

    for svg_p in sorted((PAPER_DIR / "figures").glob("*.svg")):
        shutil.copy2(svg_p, figs_out / svg_p.name)

    for script_name in ("build_paper.py", "make_v5_figures.py"):
        src = PAPER_DIR / "scripts" / script_name
        if src.is_file():
            shutil.copy2(src, scripts_out / script_name)

    html_src = build_paper.build_html_source(
        md_text,
        md_path=md_src,
        figures_dir=PAPER_DIR / "figures",
        relative_asset_urls=True,
    )
    html_out = bundle_dir / f"{args.slug}.html"
    html_out.write_text(html_src, encoding="utf-8")

    if not args.skip_pdf:
        pdf_out = bundle_dir / f"{args.slug}.pdf"
        build_paper.print_pdf_from_html(html_out, pdf_out)
        print(f"Wrote {pdf_out} ({pdf_out.stat().st_size} B)")

    if args.include_metrics:
        data_exp = REPO_ROOT / "data" / "exp_unpooled_sliders"
        if data_exp.is_dir():
            metrics_out = bundle_dir / "metrics"
            metrics_out.mkdir(parents=True, exist_ok=True)
            copied = 0
            for jp in sorted(data_exp.rglob("*.json")):
                if jp.stat().st_size < 10_000_000:
                    dst_name = f"{jp.parent.name}__{jp.name}" if jp.parent != data_exp else jp.name
                    shutil.copy2(jp, metrics_out / dst_name)
                    copied += 1
            print(f"Copied {copied} JSON metric files into {metrics_out}")
        else:
            print("Note: data/exp_unpooled_sliders/ not found locally; skipped metrics copy.")

    print(f"Exported Paper v5 bundle to $H3_OUTPUT_DIR/{args.slug}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
