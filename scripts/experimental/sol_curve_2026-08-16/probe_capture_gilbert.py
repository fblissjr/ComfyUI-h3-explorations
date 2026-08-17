"""Score a generalized-Hilbert ordering on the real capture, beside the shipped arms.

Reuses bench/analyze_capture.py's metrics verbatim -- nothing is reimplemented
here except the extra permutations under test.
"""
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

spec = importlib.util.spec_from_file_location("ac", REPO / "bench" / "analyze_capture.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

import sol_curves  # noqa: E402
from probe_hilbert import gilbert_within, shipped_within  # noqa: E402

BLOCK = 64


def frame_perm(within, frames, area, serpentine=False):
    rev = list(reversed(within))
    out = []
    for f in range(frames):
        out.extend(f * area + i for i in (rev if serpentine and f % 2 else within))
    return torch.tensor(out, dtype=torch.int64)


def main(capture, canvas="1344x768", length=124, queries=48, heads=4):
    d = torch.load(capture, map_location="cpu", weights_only=True)
    q, k = d["q"][0], d["k"][0]
    H, S, D = k.shape
    w, h = (int(v) for v in canvas.lower().split("x"))
    start, stop, grid = ac.video_span(S, (w, h), length)
    frames, height, width = grid
    area = height * width
    pad = (-start) % BLOCK
    print(f"{Path(capture).name}: heads {H}, seq {S:,}, grid {grid}, "
          f"video_start {start:,}, pad {pad}\n")

    vendor = ac.load_shipped_morton()
    orders = {"raster": torch.arange(stop - start)}
    for curve in ("2d_frame", "3d"):
        p, _ = vendor.morton_perm(grid, "cpu", curve)
        orders[f"morton_{curve}"] = torch.roll(p, pad) if pad else p
    hp, _ = sol_curves.hilbert_perm(grid, "cpu")
    orders["hilbert (shipped)"] = torch.roll(hp, pad) if pad else hp

    gil = frame_perm(gilbert_within(height, width), frames, area)
    orders["gilbert"] = torch.roll(gil, pad) if pad else gil
    ser = frame_perm(gilbert_within(height, width), frames, area, serpentine=True)
    orders["gilbert+serpentine"] = torch.roll(ser, pad) if pad else ser

    # sanity: every arm must be a permutation of the same span
    for label, o in orders.items():
        assert sorted(o.tolist()) == list(range(stop - start)), label

    print("TEST 1  centroid fidelity (higher better)")
    print(f"  {'ordering':<22}{'mean cos':>10}{'p10':>9}{'min':>9}")
    for label, order in orders.items():
        cos = ac.centroid_fidelity(k, order, start, stop)
        print(f"  {label:<22}{cos.mean():>10.4f}{cos.quantile(0.10):>9.4f}{cos.min():>9.4f}")

    print("\nTEST 2  mass concentration: key blocks for 90% of mass (lower better)")
    print(f"  {'ordering':<22}{'mean':>10}{'median':>9}{'p90':>9}")
    for label, order in orders.items():
        need = ac.mass_concentration(q, k, order, start, stop, queries, heads)
        print(f"  {label:<22}{need.mean():>10.1f}{need.median():>9.0f}"
              f"{need.quantile(0.90):>9.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
