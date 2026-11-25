#!/usr/bin/env python3
"""Does the video-and-audio row selector in `grade_dense_kernels_on_captures.py` select and grade the right rows?

The selector reads a capture manifest's segment table and grades only the
segments the final layer reads. It has no kernel in it, so its logic can be
checked on the CPU with synthetic predictions whose error is placed by hand:

- error only in the text and reference rows: the all-row grade is nonzero, the
  video-and-audio grade is zero;
- error only in the video rows: both are nonzero and the video-and-audio one is
  larger (a smaller denominator);
- the selected indices are exactly the video and audio spans of the table, and a
  manifest without the cell (or a missing manifest) selects nothing;
- against the real ref2va capture's manifest, when `--capture` names it, the
  selected row count is the audio plus video rows of the table.

Each expectation is one that a selector taking the wrong rows (all rows, the text
rows, or an off-by-one span) would fail.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/probe_grade_row_selection.py [--capture <capture dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sol_error import rel_l2_against  # noqa: E402
from grade_dense_kernels_on_captures import read_rows, rel_l2_rows  # noqa: E402


def check(name: str, ok: bool) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--capture", type=Path, default=None, help="a capture dir with a manifest, for the real-table check")
    args = ap.parse_args()

    torch.manual_seed(0)
    segs = [[0, 30, "text"], [30, 50, "ref_img"], [50, 60, "audio"], [60, 100, "video"]]
    heads, dim = 2, 8
    dense = torch.randn(1, heads, 100, dim)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cell = d / "qkv_L100_S100_b49_s2_ksol_r1.pt"
        (d / "manifest.json").write_text(json.dumps({"captured_tensors": [{"filename": cell.name, "segments": segs}]}))
        rows = read_rows(d, cell)
        expect = torch.cat([torch.arange(50, 60), torch.arange(60, 100)])
        ok &= check("selects exactly the audio and video spans", rows is not None and torch.equal(rows, expect))

        text_err = dense.clone()
        text_err[:, :, :50] += 1.0  # text and reference rows only
        ok &= check("error only in text and reference rows: all-row nonzero",
                    rel_l2_against(text_err, dense, dense.norm().item()) > 0)
        ok &= check("error only in text and reference rows: video+audio grade is zero",
                    rel_l2_rows(text_err, dense, rows) == 0.0)

        video_err = dense.clone()
        video_err[:, :, 60:] += 0.5
        allrow = rel_l2_against(video_err, dense, dense.norm().item())
        va = rel_l2_rows(video_err, dense, rows)
        manual = ((video_err[:, :, 50:] - dense[:, :, 50:]).norm() / dense[:, :, 50:].norm()).item()
        ok &= check("error only in video rows: both nonzero, video+audio larger (smaller denominator)",
                    allrow > 0 and va is not None and va > allrow)
        ok &= check("video+audio grade equals the manual norm ratio over rows 50 to 100",
                    va is not None and abs(va - manual) < 1e-6)

        other = d / "qkv_L100_S100_b0_s2_ksol_r1.pt"
        ok &= check("a cell absent from the manifest selects nothing", read_rows(d, other) is None)
        ok &= check("a missing manifest selects nothing", read_rows(d / "nope", cell) is None)
        ok &= check("no rows: no grade", rel_l2_rows(video_err, dense, None) is None)

    if args.capture:
        man = json.loads((args.capture / "manifest.json").read_text())
        entry = next(t for t in man["captured_tensors"] if t.get("segments"))
        real = read_rows(args.capture, args.capture / entry["filename"])
        want = sum(b - a for a, b, kind in entry["segments"] if kind in ("video", "audio"))
        ok &= check(f"real manifest ({args.capture.name}): selected count equals the table's audio plus video rows",
                    real is not None and int(real.numel()) == want)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
