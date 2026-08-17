"""Does the rotation hold across canvases? Peer claims it reverses at 832x480."""
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent.parent))  # ComfyUI root
from probe_hilbert import connected_fraction, shipped_within
import importlib.util
spec = importlib.util.spec_from_file_location("_v", REPO / "vendor" / "sol_attn_minimax.py")
vendor = importlib.util.module_from_spec(spec); spec.loader.exec_module(vendor)
BLOCK = 64

def build(within, frames, area, serp=0, rot=0):
    rev = list(reversed(within)); out = []
    for f in range(frames):
        seq = rev if (serp and f % 2) else within
        if rot:
            s = (f * area) % BLOCK; seq = seq[s:] + seq[:s]
        out.extend(f * area + i for i in seq)
    return torch.tensor(out, dtype=torch.int64)

FRAMES, START = 37, 530
print(f"{'canvas':<12}{'grid':>8}{'tok/f':>7}{'%64':>5}"
      f"{'hilb':>8}{'+rot':>8}{'+bou':>8}{'3d':>8}")
for cw, ch in [(1344,768),(1280,768),(1152,640),(1024,576),(832,480),(1216,704),(960,544)]:
    h, w = ch // 32, cw // 32
    area = h * w; grid = (FRAMES, h, w); pad = (-START) % BLOCK
    hil = shipped_within(h, w)
    arms = {"hilb": build(hil, FRAMES, area),
            "+rot": build(hil, FRAMES, area, rot=1),
            "+bou": build(hil, FRAMES, area, serp=1)}
    p3, _ = vendor.morton_perm(grid, "cpu", "3d")
    arms["3d"] = p3
    row = f"{str(cw)+'x'+str(ch):<12}{str(h)+'x'+str(w):>8}{area:>7}{area % BLOCK:>5}"
    for name, perm in arms.items():
        pp = torch.roll(perm, pad) if pad else perm
        frac, _, _ = connected_fraction(pp, grid, START)
        row += f"{frac:>8.1%}"
    print(row)
