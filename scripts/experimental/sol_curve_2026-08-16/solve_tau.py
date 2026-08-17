"""tau that reproduces raster's routed density under each ordering."""
import sys
import probe_routed_density as p
import torch
from pathlib import Path

def run(capture, canvas="1344x768", length=124, base_tau=1.3, heads=8):
    d = torch.load(capture, map_location="cpu", weights_only=True)
    q, k = d["q"][0], d["k"][0]
    H, S, D = k.shape
    w, h = (int(v) for v in canvas.lower().split("x"))
    start, stop, grid = p.ac.video_span(S, (w, h), length)
    frames, height, width = grid
    area = height * width
    pad = (-start) % p.BLOCK
    vendor = p.ac.load_shipped_morton()
    orders = {"raster": torch.arange(stop - start)}
    for curve in ("2d_frame", "3d"):
        pm, _ = vendor.morton_perm(grid, "cpu", curve)
        orders[f"morton_{curve}"] = torch.roll(pm, pad) if pad else pm
    hp, _ = p.sol_curves.hilbert_perm(grid, "cpu")
    orders["hilbert (shipped)"] = torch.roll(hp, pad) if pad else hp
    gw = p.gilbert_within(height, width)
    ser = torch.tensor([f * area + i for f in range(frames)
                        for i in (list(reversed(gw)) if f % 2 else gw)], dtype=torch.int64)
    orders["gilbert+serpentine"] = torch.roll(ser, pad) if pad else ser

    target, _ = p.routed_density(q, k, orders["raster"], start, stop, base_tau, heads)
    print(f"{Path(capture).name}: raster @ tau={base_tau} routes {target*100:.2f}%\n")
    print(f"  {'ordering':<22}{'tau for same density':>22}")
    for label, order in orders.items():
        lo, hi = 0.5, 4.0
        for _ in range(18):
            mid = (lo + hi) / 2
            f, _ = p.routed_density(q, k, order, start, stop, mid, heads)
            if f > target: lo = mid
            else: hi = mid
        print(f"  {label:<22}{(lo+hi)/2:>22.3f}")

run(sys.argv[1])
