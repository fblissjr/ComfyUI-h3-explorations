#!/usr/bin/env python
"""Map sglang's cube-sparse keep-ratio schedule onto the sigma grid our graphs sample.

The vendor's engineers published a per-update keep-ratio list for MiniMax-H3
at fifty steps (`bench/results/2026-09-04_sglang_cube_topk_schedule.json`,
provenance inside). It is the first step policy for H3 stated by anyone with
the training data, and roadmap step 5 (the window's start) has so far had
only our own probe trend to reason from. This script says what that schedule
would mean on OUR grid, which is shorter and built by ComfyUI's `simple`
scheduler rather than by sglang's linspace, so a reader can set the Sol
window from it instead of from a hunch.

Both grids are the same curve, `shift * t / (1 + (shift - 1) * t)` over
uniform t, so the map is by sigma: each of our denoise updates starts at a
sigma, and the vendor ratio active at that sigma is the one whose update
interval contains it. Nothing is interpolated.

What comes out, per step count:
  - each of our updates with its starting sigma and the vendor ratio there;
  - how many leading updates the vendor keeps dense on our grid, against how
    many the shipped `start_percent` keeps dense (the node runs Sol only when
    `sigma_end <= sigma <= sigma_start`, `sol_attn_h3.py`, the window gate);
  - the `start_percent` interval that would reproduce the vendor's leading
    dense count on our grid, through `percent_to_sigma`, which for this model
    is `time_snr_shift(shift, 1 - percent)` (`comfy/model_sampling.py`).

Our sigmas come from ComfyUI's own `calculate_sigmas` over `ModelSamplingAV`,
the two lines `MiniMaxH3SigmaShift.execute` runs, exactly as
`bench/check_pdd_sigmas.py::comfy_simple` builds them, and not from a copy of
the closed form; the vendor grid is the closed form its own function states.

    <comfy venv python> bench/map_cube_schedule.py --steps 16 --steps 8 --out bench/results/<date>_cube_schedule_on_h3_grid.json

Needs the ComfyUI venv (imports comfy). No GPU, no server, no model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The repo sits at <comfy>/custom_nodes/<pack>; `comfy` is importable from
# the ComfyUI root, the same derivation bench/check_pdd_sigmas.py uses.
sys.path.insert(0, str(REPO.parents[1]))
RECORD = REPO / "bench" / "results" / "2026-09-04_sglang_cube_topk_schedule.json"


def time_snr_shift(shift: float, t: float) -> float:
    return t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)


def inverse_time_snr_shift(shift: float, sigma: float) -> float:
    """t such that time_snr_shift(shift, t) == sigma."""
    return sigma if shift == 1.0 else sigma / (shift - (shift - 1) * sigma)


def vendor_grid(num_steps: int, shift: float) -> list[float]:
    """`minimax_h3_time_shift_sigmas`, replicated: linspace(1, 0, N) shifted,
    consecutive duplicates dropped, a terminal zero appended if missing."""
    import torch
    base = torch.linspace(1.0, 0.0, int(num_steps), dtype=torch.float32)
    shifted = shift * base / (1 + (shift - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(v) for v in shifted.tolist()]


def comfy_simple(shift: float, steps: int) -> list[float]:
    """The sigmas a graph samples through: `BasicScheduler(simple, steps)` on a
    shift-`shift` H3 model, built the way `check_pdd_sigmas.comfy_simple` does."""
    import torch
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.model_sampling as model_sampling
    import comfy.samplers

    class _Config:
        sampling_settings = {"shift": shift, "audio_shift": 3.0}

    class _ModelSamplingAdvanced(model_sampling.ModelSamplingAV, model_sampling.CONST):
        pass

    ms = _ModelSamplingAdvanced(_Config())
    ms.set_parameters(shift=shift, audio_shift=3.0)
    return [float(v) for v in comfy.samplers.calculate_sigmas(ms, "simple", steps).to(torch.float32).tolist()]


def ratio_at(sigma: float, grid: list[float], ratios: list[float]) -> tuple[int, float]:
    """The vendor update whose interval (grid[i+1], grid[i]] holds `sigma`."""
    for i in range(len(ratios)):
        hi, lo = grid[i], grid[i + 1]
        if lo < sigma <= hi or (i == 0 and sigma >= hi):
            return i, ratios[i]
    return len(ratios) - 1, ratios[-1]


def leading_dense(sigmas: list[float], sigma_start: float) -> int:
    """Updates from the top that the window gate sends dense: sigma > sigma_start."""
    n = 0
    for s in sigmas[:-1]:
        if s > sigma_start:
            n += 1
        else:
            break
    return n


def start_percent_interval(sigmas: list[float], k: int, shift: float) -> dict:
    """The `start_percent` values that leave exactly the first k updates dense.
    Sol runs an update when its starting sigma <= sigma_start; so sigma_start
    must sit in [sigma_k, sigma_{k-1}) in sigma, i.e. update k sparse, update
    k-1 dense. Returned as a percent interval through the inverse shift."""
    updates = sigmas[:-1]
    if k < 0 or k > len(updates):
        raise ValueError(k)
    lo_sigma = updates[k] if k < len(updates) else 0.0          # must be <= sigma_start
    hi_sigma = updates[k - 1] if k > 0 else float("inf")         # must be > sigma_start
    def pct(sig: float) -> float:
        return 1.0 - inverse_time_snr_shift(shift, sig)
    return {
        "leading_dense_updates": k,
        "sigma_start_in": [lo_sigma, hi_sigma if hi_sigma != float("inf") else None],
        "start_percent_in": [pct(hi_sigma) if hi_sigma != float("inf") else 0.0, pct(lo_sigma)],
        "note": "closed interval on the right, open on the left; any value inside reproduces k",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--steps", type=int, action="append", default=[], help="our step count(s); default 16 and 8")
    ap.add_argument("--shift", type=float, default=12.0, help="video flow shift, the grid both sides use")
    ap.add_argument("--schedule", default="recommended_50_steps", help="which list in the record to map")
    ap.add_argument("--shipped-start-percent", type=float, default=0.2)
    ap.add_argument("--shipped-end-percent", type=float, default=0.9)
    ap.add_argument("--out")
    args = ap.parse_args()
    steps_list = args.steps or [16, 8]

    rec = json.loads(RECORD.read_text())
    sched = rec["schedules"][args.schedule]
    ratios = sched["topk_ratio_list"]
    vgrid = vendor_grid(sched["num_inference_steps"], args.shift)
    if len(vgrid) - 1 != len(ratios):
        print(f"vendor grid has {len(vgrid) - 1} updates but the list has {len(ratios)} entries")
        return 1

    try:
        comfy_sha = subprocess.run(["git", "-C", str(REPO.parents[1]), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        comfy_sha = None

    out = {
        "produced_by": "bench/map_cube_schedule.py",
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_record": str(RECORD.relative_to(REPO)),
        "schedule": args.schedule,
        "shift": args.shift,
        "comfy_commit": comfy_sha,
        "vendor_grid_points": len(vgrid),
        "vendor_leading_dense_updates": next((i for i, r in enumerate(ratios) if r < 1.0), len(ratios)),
        "vendor_last_dense_sigma": vgrid[next((i for i, r in enumerate(ratios) if r < 1.0), len(ratios))],
        "per_steps": {},
    }
    sigma_start_shipped = time_snr_shift(args.shift, 1.0 - args.shipped_start_percent)
    sigma_end_shipped = time_snr_shift(args.shift, 1.0 - args.shipped_end_percent)

    for steps in steps_list:
        ours = comfy_simple(args.shift, steps)
        rows = []
        for j in range(len(ours) - 1):
            s = ours[j]
            i, r = ratio_at(s, vgrid, ratios)
            shipped_sol = sigma_end_shipped <= s <= sigma_start_shipped
            rows.append({"update": j, "sigma": s, "vendor_update": i, "vendor_keep_ratio": r,
                         "vendor_dense": r >= 1.0, "shipped_window_runs_sol": shipped_sol})
        vendor_lead = sum(1 for row in rows if row["vendor_dense"]) if all(
            rows[t]["vendor_dense"] for t in range(sum(1 for row in rows if row["vendor_dense"]))) else None
        shipped_lead = leading_dense(ours, sigma_start_shipped)
        shipped_trail = sum(1 for s in ours[:-1] if s < sigma_end_shipped)
        per = {
            "sigmas": ours,
            "updates": rows,
            "vendor_leading_dense_on_our_grid": vendor_lead,
            "shipped_leading_dense": shipped_lead,
            "shipped_trailing_dense": shipped_trail,
            "shipped_sigma_window": [sigma_end_shipped, sigma_start_shipped],
            "start_percent_for_vendor_leading_dense": start_percent_interval(ours, vendor_lead or 0, args.shift),
            "start_percent_for_one_leading_dense": start_percent_interval(ours, 1, args.shift),
            "start_percent_for_two_leading_dense": start_percent_interval(ours, 2, args.shift),
            "vendor_keep_ratio_mean_over_sparse_updates": (
                sum(row["vendor_keep_ratio"] for row in rows if not row["vendor_dense"])
                / max(1, sum(1 for row in rows if not row["vendor_dense"]))),
        }
        out["per_steps"][str(steps)] = per
        print(f"\n== {steps} steps, shift {args.shift} (simple), vendor schedule '{args.schedule}' ==")
        print(f"   update  sigma     vendor#  keep   vendor    shipped window ({args.shipped_start_percent}/{args.shipped_end_percent})")
        for row in rows:
            print(f"   {row['update']:>4d}   {row['sigma']:.4f}   {row['vendor_update']:>4d}   {row['vendor_keep_ratio']:.2f}   "
                  f"{'dense' if row['vendor_dense'] else 'sparse':6s}    {'sol' if row['shipped_window_runs_sol'] else 'dense'}")
        print(f"   vendor keeps the first {vendor_lead} update(s) dense here; shipped keeps {shipped_lead} leading + {shipped_trail} trailing dense")
        sp = per["start_percent_for_vendor_leading_dense"]
        print(f"   start_percent reproducing the vendor's leading count: ({sp['start_percent_in'][0]:.3f}, {sp['start_percent_in'][1]:.3f}]")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nrecord written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
