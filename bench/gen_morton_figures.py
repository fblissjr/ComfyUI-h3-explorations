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

**The drawing lives in `bench/gen_figures.py`** as of 2026-08-20, alongside the
other figure forms; this file is the entry point named by `docs/morton.md` and
does nothing but call it. `python bench/gen_figures.py morton --out DIR` is the
same run.
"""
import argparse
from pathlib import Path

from gen_figures import write_morton

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out", default=".", help="directory to write fig1.svg / fig2.svg")
for _line in write_morton(Path(_ap.parse_args().out)):
    print(_line)
