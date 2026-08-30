#!/usr/bin/env python3
"""Write down what a capture CONTAINED, so the capture can be deleted.

## Why this exists

Captures are the largest thing on this box -- 178 GiB across four sets on
2026-08-30 -- and they are intermediate: the results derived from them are what
anyone reads. So they get deleted, and they should be.

**The problem is that deleting one silently changes what its results mean.**
`bench/results/2026-08-18_sage_accuracy_on_capture.json` cites
`2026-08-17_ref3_362f_1024x768`, which is already gone. Three files still name
that capture and not one of them says it no longer exists, so a reader cannot
tell "this is reproducible" from "this is the only surviving copy of a
measurement nobody can re-run". Those are very different claims about the same
number.

This records, per capture, what it held -- tensor inventory, shapes, blocks,
steps, sequence length, and the manifest if there is one -- into a JSON that
outlives it. Run it BEFORE deleting a capture, not after.

## What it is not

Not a substitute for the capture. A re-derivation needs the tensors. This makes
the record honest about that: it says what was there and that it is gone, so a
result citing it reads as a frozen measurement rather than a live one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"qkv_L(\d+)_S(\d+)_b(\d+)_s(\d+)(?:_r(\d+))?\.pt$")


def inventory(root: Path) -> dict:
    # Recursive, and that was not the first version. `2026-08-21_pruning` is
    # 70 GiB of per-ARM subdirectories (ref3_fl2va_pruned, t2v_fl2va_unpruned,
    # ...) rather than a flat set of qkv files, so a non-recursive scan
    # reported it as 0 files and 0 bytes -- a capture recorded as empty is
    # worse than one not recorded at all, because the record then licenses
    # deleting something nobody knows the size of.
    files = sorted(p for p in root.rglob("*.pt"))
    rows, unparsed = [], []
    for p in files:
        m = NAME.search(p.name)
        if not m:
            unparsed.append({"file": str(p.relative_to(root)),
                             "bytes": p.stat().st_size})
            continue
        rows.append({"file": str(p.relative_to(root)),
                     "length": int(m.group(1)),
                     "sequence": int(m.group(2)), "block": int(m.group(3)),
                     "step": int(m.group(4)),
                     "render": int(m.group(5) or 0),
                     "bytes": p.stat().st_size})
    man = root / "manifest.json"
    out = {
        "capture": root.name,
        "existed_at_record_time": True,
        "n_tensor_files": len(rows),
        "arms": sorted({str(p.relative_to(root)).split("/")[0]
                        for p in files if "/" in str(p.relative_to(root))}),
        "total_bytes": (sum(r["bytes"] for r in rows)
                        + sum(u["bytes"] for u in unparsed)),
        "total_bytes_parsed_only": sum(r["bytes"] for r in rows),
        "blocks": sorted({r["block"] for r in rows}),
        "steps": sorted({r["step"] for r in rows}),
        "sequence_lengths": sorted({r["sequence"] for r in rows}),
        "renders": sorted({r["render"] for r in rows}),
        "unparsed_files": unparsed,
        "files": rows,
        "manifest": json.loads(man.read_text()) if man.is_file() else None,
        "note": ("Written so this capture can be DELETED without its results "
                 "becoming unreadable. A result citing a capture that no "
                 "longer exists is a frozen measurement, not a reproducible "
                 "one, and should say so."),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+", type=Path)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    got, absent = [], []
    for c in args.captures:
        if c.is_dir():
            got.append(inventory(c))
        else:
            # NAME only, never the path. Captures live outside the repo and an
            # absolute path here is a leak the pre-commit hook refuses -- which
            # it did, on the first run of this script.
            absent.append(c.name)
            print(f"  ABSENT (recording as already deleted): {c.name}")

    payload = {
        "measured": "2026-08-30",
        "produced_by": "bench/record_capture_inventory.py",
        "what": ("what each H3 capture contained, recorded so the capture "
                 "itself can be deleted without orphaning the results that "
                 "cite it"),
        "captures": got,
        "already_absent": absent,
        "capture_root": ("names only; captures live outside the repo and their "
                         "location is not recorded here by policy"),
        "total_bytes_recorded": sum(c["total_bytes"] for c in got),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    for c in got:
        print(f"  {c['capture']:44s} {c['n_tensor_files']:3d} files  "
              f"{c['total_bytes'] / 2**30:6.1f} GiB  "
              f"blocks {c['blocks']}  steps {c['steps']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
