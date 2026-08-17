#!/usr/bin/env python3
"""Exact packed-sequence rows, per segment, from the real `PackedLayout`.

    python bench/count_packed_rows.py --length 362 --canvas 1024x768 \
        --refs 662x1177,1600x1600,2752x1536 --text 7737

Everything in this repo that prices a sequence does it by **arithmetic over
row counts recorded elsewhere** -- `bench/preflight_graph.py`,
`docs/SOLATTN.md`'s sink table, `docs/h3_references.md`'s reference costs. This
builds the layout ComfyUI will actually build and reads the segments off it, so
those numbers have something to be right or wrong against.

## Why it exists

Three consumers, all of which were derived rather than counted:

  - `docs/SOLATTN.md`'s **sink share**, quoted as 16.6% of all (query block,
    key block) pairs. Derived from `video_start`, and `video_start` was itself
    derived.
  - `docs/evidence.md`'s v1-vs-v2 sink table, which still says "derived, not
    measured" because each row's `S` came from a published percentage.
  - `bench/preflight_graph.py`'s total, which **under-reported the 2026-08-17
    capture by 1,136 rows**. See below -- the cause is a missing segment, not
    an estimation error.

## What is exact here and what is not

**Exact**: video, target audio, and every reference segment. These come from
`PackedLayout.segments`, which is the same object the model builds.

**Not exact**: `text`. Its length is whatever the tokenizer produced for the
prompt plus the vision blocks, and getting it honestly means running the text
encoder. `--text` takes a measured value; without it the tool reports the
sequence *minus* text and says so, which is still enough to price references
and to compute `video_start` if you have the real sequence length from a log.

Refusing to guess is the point. A tool that invented a text length would put
this repo back where preflight is.

Needs `PYTHONPATH` to reach ComfyUI. No CUDA, no model, no server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BLOCK = 64


def _layout(text_len, latent_t, latent_h, latent_w, audio_t, refs):
    from comfy.ldm.minimax.model import PackedLayout
    return PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t,
                        refs=refs or None)


def _ref_blocks(specs):
    """`refs=` blocks for image references given as WxH strings.

    **Sizes go through `reference_fit._fit`, not through arithmetic here.** A
    reference is snapped to a multiple of 32 by *rounding* before it is priced
    by *truncation*, and doing only the truncation under-prices it: 662x1177 is
    777 rows (snapped to 672x1184), not the 720 that `(662//32) * (1177//32)`
    gives. The first version of this file reimplemented it and was wrong by 8%
    on exactly the reference the 2026-08-17 capture used to probe fine detail.

    Importing rather than restating is the whole point of a counter that exists
    to check other people's arithmetic.
    """
    from reference_fit import _fit
    out = []
    for spec in specs:
        w, h = (int(v) for v in spec.lower().split("x"))
        fw, fh = _fit(w, h, 1.0)                 # scale 1.0 == native, no upscale
        out.append({"kind": "image", "latent_h": fh // 16, "latent_w": fw // 16,
                    "ref_audio_t": 0})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=int, required=True, help="frame count")
    ap.add_argument("--canvas", required=True, help="WIDTHxHEIGHT, e.g. 1024x768")
    ap.add_argument("--refs", default="", help="comma-separated WxH image references")
    ap.add_argument("--text", type=int, default=None,
                    help="measured text rows; omitted means report without them")
    ap.add_argument("--against", type=int, default=None,
                    help="a sequence length to compare with, e.g. from a capture log")
    args = ap.parse_args()

    # ComfyUI's root must sit AHEAD of this repo's, and the order below is
    # load-bearing: `comfy_extras.nodes_minimax_h3` does a bare `import nodes`,
    # and this repo has a `nodes.py` of its own that dies on a relative import
    # when found first. Inserting the repo root second puts ComfyUI first. This
    # is the trap in CLAUDE.md and it bit this file on its first run.
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))     # repo, for reference_fit
    sys.path.insert(0, str(here.parents[3]))     # ComfyUI, ahead of it
    from comfy_extras.nodes_minimax_h3 import temporal_shape

    w, h = (int(v) for v in args.canvas.lower().split("x"))
    _fc, latent_t, audio_t = temporal_shape(args.length)
    refs = _ref_blocks([s for s in args.refs.split(",") if s])

    text = args.text if args.text is not None else 0
    layout = _layout(text, latent_t, h // 16, w // 16, audio_t, refs)

    print(f"{args.canvas}, {args.length} frames -> latent_t {latent_t}, "
          f"audio_t {audio_t}, {len(refs)} image reference(s)\n")
    print(f"  {'segment':<14}{'rows':>10}   span")
    total = 0
    video_start = None
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "video" and video_start is None:
            video_start = a
        print(f"  {kind:<14}{n:>10,}   [{a:,}..{b:,})")
        total = b
    if args.text is None:
        print(f"\n  {'TOTAL':<14}{total:>10,}   ** text counted as 0; pass "
              f"--text to complete it **")
    else:
        print(f"\n  {'TOTAL':<14}{total:>10,}")

    if video_start is not None:
        blocks = total // BLOCK
        sink = -(-video_start // BLOCK)          # ceil
        print(f"\n  block grid: {blocks:,} whole blocks of {BLOCK}")
        print(f"  video_start {video_start:,} -> {sink:,} sink blocks "
              f"({sink / blocks:.1%} of all query/key block pairs)")
        print("  (this is the number docs/SOLATTN.md's depth table calls the "
              "sink floor)")

    if args.against is not None:
        gap = args.against - total
        print(f"\n  against {args.against:,}: {gap:+,} rows "
              f"({abs(gap) / args.against:.2%})")
        if args.text is None and gap:
            print(f"  with text unset, that gap IS the text length: {gap:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
