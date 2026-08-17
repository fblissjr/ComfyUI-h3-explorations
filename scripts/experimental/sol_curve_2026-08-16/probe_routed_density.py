"""Does the curve move the router's operating point at fixed tau?

The routing arithmetic is lifted from upstream's own eager reference,
`coderef/comfy-kitchen-sol/comfy_kitchen/backends/eager/sol_attn.py:100-140`,
rather than reimplemented -- the CUDA kernel and that file agree by
construction, and docs/morton.md warns specifically against writing a third
version of the threshold formula.

Skipped relative to the kernel: the INT8 quantization of the pooled keys and
query centroids. So this is the float routing rule, not the shipped kernel's.
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
from probe_hilbert import gilbert_within  # noqa: E402

BLOCK = 64
LOG2E = 1.4426950408889634


def routed_density(q, k, order, start, stop, tau, heads, seed=0):
    """Fraction of NON-ADJACENT (query block, key block) pairs the threshold
    would route exact.

    **This is not the fraction the kernel routes.** Only `|i-j| <= 1` is
    excluded. Sink pairs stay in the denominator and their forced-exact status
    is never applied, so the kernel's number is higher than this one.

    The ratios between orderings are unaffected -- the same pairs are counted
    under every ordering, and forced-exact is ordering-invariant -- so this is
    a sound instrument for "what did the permutation do" and an unsound one for
    "how much work does the kernel do". `bench/analyze_routing.py` should emit
    both and say which is which.

    Corrected 2026-08-16: an earlier docstring here claimed sink blocks were
    excluded from both numerator and denominator. They are not, and that claim
    was relayed to another session before it was checked.

    Conditioning rows ARE in the block population: `n = S // BLOCK` blocks the
    whole packed sequence, so `kcvar` is a variance over every centroid, which
    is how the kernel derives it (`sol_attn_preprocess.cu:107-123`).
    """
    H, S, D = k.shape
    rows = torch.arange(S)
    rows[start:stop] = order + start
    n = S // BLOCK                      # whole blocks only
    t = n * BLOCK
    scale = D ** -0.5
    log2s = scale * LOG2E

    g = torch.Generator().manual_seed(seed)
    hsel = torch.randperm(H, generator=g)[:heads].tolist()

    idx = torch.arange(n)
    offdiag = (idx.view(1, -1) - idx.view(-1, 1)).abs() > 1

    fracs, taus = [], []
    for hh in hsel:
        fk = k[hh].index_select(0, rows)[:t].float()
        fq = q[hh].index_select(0, rows)[:t].float()
        kc = fk.view(n, BLOCK, D).mean(1)                 # (N, D)
        kmean = kc.mean(0, keepdim=True)
        kcc = kc - kmean
        kc_var = kcc.pow(2).mean(0)                       # (D,)
        centroid = fq.view(n, BLOCK, D).mean(1)           # (N, D)
        var = (centroid.pow(2) * kc_var).sum(-1)          # (N,)
        thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)

        colmean = (centroid @ kcc.T) * log2s              # (NQ, N)
        exact = colmean > thr.unsqueeze(-1)
        fracs.append(float((exact & offdiag).sum()) / float(offdiag.sum()))

        # tau that would reproduce a target density is monotone in tau, so
        # record the per-query-block score distribution scale for a direct read
        taus.append(float((colmean / torch.sqrt(var * log2s * log2s + 1e-6)
                           .unsqueeze(-1))[offdiag].std()))
    return sum(fracs) / len(fracs), sum(taus) / len(taus)


def main(capture, canvas="1344x768", length=124, tau=1.3, heads=8):
    d = torch.load(capture, map_location="cpu", weights_only=True)
    q, k = d["q"][0], d["k"][0]
    H, S, D = k.shape
    w, h = (int(v) for v in canvas.lower().split("x"))
    start, stop, grid = ac.video_span(S, (w, h), length)
    frames, height, width = grid
    area = height * width
    pad = (-start) % BLOCK

    vendor = ac.load_shipped_morton()
    orders = {"raster": torch.arange(stop - start)}
    for curve in ("2d_frame", "3d"):
        p, _ = vendor.morton_perm(grid, "cpu", curve)
        orders[f"morton_{curve}"] = torch.roll(p, pad) if pad else p
    hp, _ = sol_curves.hilbert_perm(grid, "cpu")
    orders["hilbert (shipped)"] = torch.roll(hp, pad) if pad else hp
    gw = gilbert_within(height, width)
    ser = torch.tensor([f * area + i for f in range(frames)
                        for i in (list(reversed(gw)) if f % 2 else gw)],
                       dtype=torch.int64)
    orders["gilbert+serpentine"] = torch.roll(ser, pad) if pad else ser

    print(f"{Path(capture).name}  tau={tau}, {heads} of {H} heads, "
          f"{S // BLOCK} blocks\n")
    print(f"  {'ordering':<22}{'routed %':>10}{'vs raster':>12}")
    base = None
    for label, order in orders.items():
        frac, _ = routed_density(q, k, order, start, stop, tau, heads)
        if base is None:
            base = frac
        print(f"  {label:<22}{frac * 100:>9.2f}%{frac / base:>11.3f}x")


if __name__ == "__main__":
    main(sys.argv[1], tau=float(sys.argv[2]) if len(sys.argv) > 2 else 1.3)
