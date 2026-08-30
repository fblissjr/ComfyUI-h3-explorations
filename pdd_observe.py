"""Record what PDD's backbone delta DOES, at every block and every step.

## Why this is an observation and not a probe

`CLAUDE.md`'s capture rule, added 2026-08-30: a probe that varies one axis has
already assumed the answer is on that axis. The per-block strength sweep this
replaces varied block and strength while pinning the sigma schedule -- pinning
the one axis the owner already knew moved PDD -- and recorded one number per
arm, so it could not be re-asked anything.

This changes nothing about the render. `MiniMaxH3PDDLoRA`'s un-merged path
already computes a block's base output and its PDD delta as separate terms, so
the delta's size is observable for free. One ordinary PDD render yields
`block x step x position`, and the grouping is decided offline.

## Positional bins, and why not segments

The obvious grouping is H3's packed segments -- `[text | cond | ref | audio |
video]` -- and that is the field `bench/grade_sage_on_capture.py` cannot group
by, because captures never recorded it. Two other lanes are blocked on the same
gap.

Rather than add a capture-time dependency on `PackedLayout`, this records a
fixed number of equal-width POSITIONAL bins over the packed sequence. With
enough bins any boundary can be located afterwards, so a segment grouping
becomes an analysis pass rather than a re-render -- and a capture taken before
anyone knows where the boundaries are is still answerable once they do. That is
the rule applied to itself: record finely, decide later.

## Inert unless `H3_PDD_OBSERVE` is set

    H3_PDD_OBSERVE=/path/to/dir <comfy>/start.sh

No node input and no graph change, for the reason `h3_capture.py` states: this
must not be reachable by opening a workflow. Unlike that file it writes
kilobytes, not gigabytes -- one float per bin per (block, step).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

#: Equal-width bins over the packed sequence. 256 puts every H3 segment
#: boundary within ~0.4% of the sequence, which is finer than any grouping
#: question asked so far, and costs 1 KB per (block, step).
N_BINS = 256

_rows: list[dict] = []
_dir: Path | None = None


def enabled() -> bool:
    global _dir
    if _dir is None:
        d = os.environ.get("H3_PDD_OBSERVE", "").strip()
        _dir = Path(d) if d else Path("")
    return bool(str(_dir))


def record(block: int, kind: str, sigma, out: torch.Tensor,
           delta: torch.Tensor) -> None:
    """One (block, module, step) observation. Never raises into a render."""
    if not enabled():
        return
    try:
        with torch.no_grad():
            o = out.detach().float()
            d = delta.detach().float()
            if o.ndim != 2:
                o = o.reshape(-1, o.shape[-1])
                d = d.reshape(-1, d.shape[-1])
            rows = o.shape[0]
            # Per-row norms first, then binned: a bin's value is the relative
            # perturbation of the rows in it, not the perturbation of a pooled
            # vector. Pooling first would let opposite-signed rows cancel.
            on = torch.linalg.vector_norm(o, dim=1)
            dn = torch.linalg.vector_norm(d, dim=1)
            edges = torch.linspace(0, rows, N_BINS + 1).long()
            bins = []
            for i in range(N_BINS):
                a, b = int(edges[i]), int(edges[i + 1])
                if b <= a:
                    bins.append(float("nan"))
                    continue
                num = float(torch.linalg.vector_norm(dn[a:b]))
                den = float(torch.linalg.vector_norm(on[a:b]))
                bins.append(num / den if den else float("nan"))
            _rows.append({
                "block": int(block), "module": kind,
                "sigma": None if sigma is None else float(sigma),
                "rows": int(rows),
                "rel_overall": float(torch.linalg.vector_norm(dn)
                                     / torch.linalg.vector_norm(on)),
                "bins": bins,
            })
    except Exception:                      # observation must never break a render
        pass


_path: Path | None = None


def flush(meta: dict | None = None) -> str | None:
    """Write everything observed so far. Idempotent, and does NOT clear.

    Called once per model forward rather than once per render, because nothing
    in this pack sees a render end -- the node runs at patch time and the
    capture patch runs per step. Rewriting a few hundred kilobytes each step is
    cheaper than adding a lifecycle hook, and a render interrupted part-way
    still leaves the steps it completed on disk.
    """
    global _path
    if not enabled() or not _rows:
        return None
    _dir.mkdir(parents=True, exist_ok=True)
    if _path is None:
        _path = _dir / f"pdd_observe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path = _path
    path.write_text(json.dumps({
        "produced_by": "pdd_observe.py",
        "what": ("per (block, module, step): the PDD backbone delta's size "
                 "relative to the block's own output, in equal-width "
                 "positional bins over the packed sequence"),
        "n_bins": N_BINS,
        "grouping": ("POSITIONAL, not per segment. Segment boundaries were not "
                     "recorded; with 256 bins they can be applied offline once "
                     "known. See this module's docstring."),
        "meta": meta or {},
        "observations": _rows,
    }, indent=2) + "\n")
    return str(path)
