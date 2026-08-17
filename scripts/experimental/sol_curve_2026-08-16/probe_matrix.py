"""{hilbert, gilbert} x {plain, serpentine, rotation, both}, geometry then capture."""
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "bench"))
from probe_hilbert import connected_fraction, gilbert_within, shipped_within
import importlib.util
spec = importlib.util.spec_from_file_location("ac", REPO / "bench" / "analyze_capture.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
BLOCK = 64

def build(within, frames, area, serp, rot):
    rev = list(reversed(within)); out = []
    for f in range(frames):
        seq = rev if (serp and f % 2) else within
        if rot:
            s = (f * area) % BLOCK
            seq = seq[s:] + seq[:s]
        out.extend(f * area + i for i in seq)
    return torch.tensor(out, dtype=torch.int64)

def main(capture=None):
    frames, height, width = 37, 24, 42
    area = height * width; grid = (frames, height, width)
    start = 530; pad = (-start) % BLOCK
    bases = {"hilbert": shipped_within(height, width), "gilbert": gilbert_within(height, width)}
    arms = {}
    for bname, within in bases.items():
        for label, (serp, rot) in {"": (0,0), "+serp": (1,0), "+rot": (0,1), "+serp+rot": (1,1)}.items():
            arms[bname + label] = build(within, frames, area, serp, rot)
    print("Geometry at real alignment (video_start=530):")
    print(f"  {'arm':<20}{'connected':>11}{'radius':>9}")
    for name, perm in arms.items():
        p = torch.roll(perm, pad) if pad else perm
        frac, rad, _ = connected_fraction(p, grid, start)
        print(f"  {name:<20}{frac:>10.1%}{rad:>9.2f}")
    if not capture: return
    d = torch.load(capture, map_location="cpu", weights_only=True)
    q, k = d["q"][0], d["k"][0]; S = k.shape[1]
    st, sp, g2 = ac.video_span(S, (1344, 768), 124)
    assert g2 == grid and st == start
    vendor = ac.load_shipped_morton()
    ref = {"raster": torch.arange(sp - st)}
    for c in ("2d_frame", "3d"):
        pm, _ = vendor.morton_perm(grid, "cpu", c); ref[f"morton_{c}"] = torch.roll(pm, pad)
    print(f"\nCapture {Path(capture).name}:")
    print(f"  {'arm':<20}{'cos':>9}{'p10':>9}{'mass':>9}")
    for name, order in {**ref, **{k2: torch.roll(v, pad) for k2, v in arms.items()}}.items():
        cos = ac.centroid_fidelity(k, order, st, sp)
        need = ac.mass_concentration(q, k, order, st, sp, 48, 4)
        print(f"  {name:<20}{cos.mean():>9.4f}{cos.quantile(0.10):>9.4f}{need.mean():>9.1f}")

main(sys.argv[1] if len(sys.argv) > 1 else None)
