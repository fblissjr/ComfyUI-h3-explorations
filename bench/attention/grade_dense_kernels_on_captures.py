#!/usr/bin/env python3
"""The three dense-step kernels on every captured cell: error against fp32 attention, per block and step.

Every block-49 grade so far is at the last step. Composition is decided on
the early steps, and the coins controls (2026-09-15) put a visible flaw on
the kitchen dense kernel's early steps that its last-step grade could not
see. This runs the three kernels a graph can put on the dense steps over
every cell of a capture set, same inputs, same fp32 referent:

    python bench/grade_dense_kernels_on_captures.py <capture dir> [--heads 8] [--json out]

Each cell is graded two ways. The first line is the error over EVERY query row
(the grade every earlier record uses, so they stay comparable). The second, under
it, is the error over the video and audio rows only, from the capture manifest's
segment table (`rows_video_audio` in the JSON; `rows` is unchanged): the final layer reads only those two segments, so text and reference rows
are never read out of the last block, and at block 49 a cell with many of them
(ref2va) reads far worse over all rows than over the rows that reach the picture
(measured on the kernels, all heads: `bench/results/2026-09-29_grade_video_audio_real_kernels.md`).
A capture without a manifest segment table gets no second line.

**`--heads` defaults to 8, the first eight, and that is a quick look, not a grade.**
The earlier records used every head (`heads_measured` in their JSON). On the
ref2va capture's block 49 the first eight heads read about half the all-head error
and hide the row effect entirely; pass `--heads 56` for anything quoted.

Kernels: comfy_kitchen.int8_attention (Model Attention Backend), sage fp8++
(sageattn_qk_int8_pv_fp8_cuda, pv fp32+fp16), the same with `qk_rotate` (the
Sage node's "fp8++ rotated" and "auto" modes),
and bf16 SDPA on the same bf16 inputs (the "dense" arm; its error is the
bf16 output rounding plus torch's own accumulation, the floor). Sol is not
here: it is the routed-step kernel, graded elsewhere.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_sol_error import dense_reference, load_capture, rel_l2_against  # noqa: E402


def kitchen_int8(q, k, v):
    import comfy_kitchen as ck
    return ck.int8_attention(q.cuda(), k.cuda(), v.cuda()).float().cpu()


def sage_fp8pp(q, k, v):
    import sageattention
    return sageattention.sageattn_qk_int8_pv_fp8_cuda(
        q.cuda(), k.cuda(), v.cuda(), tensor_layout="HND", is_causal=False,
        pv_accum_dtype="fp32+fp16", smooth_k=False).float().cpu()


def sage_fp8pp_rotated(q, k, v):
    # The Sage node's "fp8++ rotated" (and "auto") mode, `attention.py`: the
    # same call with the fork's fixed q/k rotation (added 2026-09-27, for the
    # Sol redesign's dense-tail question, audit section 9b).
    import sageattention
    return sageattention.sageattn_qk_int8_pv_fp8_cuda(
        q.cuda(), k.cuda(), v.cuda(), tensor_layout="HND", is_causal=False,
        pv_accum_dtype="fp32+fp16", smooth_k=False, qk_rotate=True).float().cpu()


def bf16_sdpa(q, k, v):
    outs = []
    for hh in range(q.shape[1]):
        outs.append(torch.nn.functional.scaled_dot_product_attention(
            q[:, hh:hh + 1].cuda(), k[:, hh:hh + 1].cuda(), v[:, hh:hh + 1].cuda()).float().cpu())
    return torch.cat(outs, dim=1)


KERNELS = {"kitchen_int8": kitchen_int8, "sage_fp8pp": sage_fp8pp,
           "sage_fp8pp_rotated": sage_fp8pp_rotated, "bf16_sdpa": bf16_sdpa}

READ_KINDS = ("video", "audio")  # the segments FinalLayer reads (comfy/ldm/minimax/model.py)


def read_rows(capture_dir: Path, cell: Path) -> torch.Tensor | None:
    """Row indices of the segments the final layer reads, from the manifest; None if the capture has no table."""
    manifest = capture_dir / "manifest.json"
    if not manifest.exists():
        return None
    for entry in json.loads(manifest.read_text()).get("captured_tensors", []):
        if entry.get("filename") == cell.name and entry.get("segments"):
            spans = [(a, b) for a, b, kind in entry["segments"] if kind in READ_KINDS]
            return torch.cat([torch.arange(a, b) for a, b in spans]) if spans else None
    return None


def rel_l2_rows(pred: torch.Tensor, dense: torch.Tensor, rows: torch.Tensor | None) -> float | None:
    """`rel_l2_against` over the given query rows (dim 2), each side normalised by that subset's own dense norm."""
    if rows is None:
        return None
    d = dense[:, :, rows]
    return rel_l2_against(pred[:, :, rows], d, d.float().norm().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--steps", default="", help="comma-separated steps to grade (default all)")
    ap.add_argument("--blocks", default="", help="comma-separated blocks to grade (default all)")
    args = ap.parse_args()
    want_steps = {int(x) for x in args.steps.split(",") if x}
    want_blocks = {int(x) for x in args.blocks.split(",") if x}
    cells = sorted(Path(args.capture_dir).glob("qkv_*_b*_s*.pt"),
                   key=lambda p: tuple(int(x) for x in re.search(r"_b(\d+)_s(\d+)", p.name).groups()))
    rows = []
    rows_va = []  # same shape as rows, over the video and audio rows only; `rows` stays as it was for readers of the JSON
    capture_dir = Path(args.capture_dir)
    print(f"heads measured: the first {args.heads}" + ("  (a quick look; the records use every head)" if args.heads < 56 else ""))
    print(f"{'cell':14s}" + "".join(f"{k:>14s}" for k in KERNELS))
    for cell in cells:
        m = re.search(r"_b(\d+)_s(\d+)", cell.name)
        block, step = int(m.group(1)), int(m.group(2))
        if (want_steps and step not in want_steps) or (want_blocks and block not in want_blocks):
            continue
        q, k, v = load_capture(str(cell))
        if 0 < args.heads < q.shape[1]:
            q, k, v = q[:, :args.heads], k[:, :args.heads], v[:, :args.heads]
        dense = dense_reference(q, k, v); dn = dense.float().norm().item()
        read = read_rows(capture_dir, cell)
        row = {"block": block, "step": step}
        row_va = {"block": block, "step": step}
        for name, fn in KERNELS.items():
            try:
                pred = fn(q, k, v)
                row[name] = rel_l2_against(pred, dense, dn)
                row_va[name] = rel_l2_rows(pred, dense, read)
            except Exception as exc:  # a kernel absent from this venv is a hole, not a failure
                row[name] = None
                row_va[name] = None
                row[name + "_error"] = f"{type(exc).__name__}: {exc}"[:120]
        rows.append(row)
        if read is not None:
            rows_va.append(row_va)
        print(f"b{block:<3d}s{step:<9d}" + "".join(f"{row[k]:>14.4f}" if row[k] is not None else f"{'-':>14s}" for k in KERNELS))
        if read is not None:
            print(f"{'  video+audio':14s}" + "".join(
                f"{row_va[k]:>14.4f}" if row_va[k] is not None else f"{'-':>14s}" for k in KERNELS))
        del q, k, v, dense
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "capture_dir": Path(args.capture_dir).name, "heads_measured": args.heads,
            "referent": "fp32 dense attention on the same bf16 inputs (analyze_sol_error.dense_reference)",
            "rows": rows,
            "rows_video_audio": rows_va,
            "rows_video_audio_note": "the same cells over the video and audio query rows only (the segments the final layer "
                                     "reads), from the capture manifest's segment table; empty without one"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
