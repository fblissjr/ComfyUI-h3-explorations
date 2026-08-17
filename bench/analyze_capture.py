#!/usr/bin/env python3
"""Does Morton actually help the router? Answered from captured q/k, not geometry.

`bench/analyze_morton.py` measures which tokens share a block. That is exact and
it is not the question anyone cares about. The question is whether a compact
block is *better for Sol-Attn* than a scattered one, and that depends on the
activations, not on the grid.

`docs/morton.md` calls this link 5 of a six-link chain. Links 1-4 are verified;
link 5 carries everything downstream and has never been measured. This script
measures it.

## The two things it tests

**1. Centroid fidelity (link 5 itself).** Sol-Attn summarises each 64-token key
block with its mean, and uses that mean both to route and as the stand-in for
the whole block. So the question is: how well does the mean represent its
members? Reported as the mean cosine of each key to its own block centroid,
under raster order and under Morton. **If Morton does not raise this, the whole
canvas argument in `docs/morton.md` is decoration.**

**2. Mass concentration (the stated mechanism).** Upstream's `_morton.py` says
Z-ordering "concentrates the mass into fewer blocks", which is why the payoff
should appear at higher `tau` rather than at fixed `tau`. Tested by computing
real attention weights for sampled query rows, then grouping the same weights
two ways and asking how many key blocks hold 90% of the mass. **The attention
weights are identical under both orderings** -- only the grouping changes -- so
this is an exact regrouping measurement with no approximation anywhere.

## Why one capture covers both orderings, at every block

Morton permutes the video span before the first transformer block, permutes the
rope rows to match, and inverts after the last. **If the captured run used dense
attention** -- that is, Sol-Attn bypassed, which a capture run requires anyway --
then attention is permutation-equivariant and every later block's hidden states
are also exactly the permuted version. So permuting a captured q/k gives exactly
what the Morton arm would have seen, **at any block, not only block 0**.

An earlier version of this file restricted that to block 0 and warned otherwise.
That was too conservative, and it mattered: block 0 turns out to be the worst
block to ask this question at, because early-layer attention is close to
uniform and leaves a block-sparse router almost nothing to exploit. Capture deep
blocks too.

The restriction *would* apply to a capture taken with Sol-Attn on, since the
sparse arm's trajectory diverges from the dense one after the first sparse
call. `h3_capture.py`'s docstring already requires Sol off for a different
reason (the sigma window makes the sample unrepresentative), so this is one more
reason for the same rule.

## What it does NOT do

It does not reimplement Sol-Attn's router. The threshold formula lives in
`sol_attn_preprocess.cu` and replicating it would put a fidelity risk between
the measurement and the claim. Both tests above are properties of the
activations and the partition, so neither needs the kernel.

## Running it

Capture first. This needs a ComfyUI restart, because `h3_capture.py` reads its
environment at import:

    H3_CAPTURE="dir=/some/scratch,blocks=0,steps=1" <comfy>/start.sh

Render with Sol-Attn bypassed so sage takes every call, then:

    python bench/analyze_capture.py /some/scratch/qkv_L124_S38328_b0_s1.pt \\
        --canvas 1344x768 --length 124

Needs torch and the capture. No model, no server.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

BLOCK = 64
REPO = Path(__file__).resolve().parent.parent


def load_shipped_morton():
    """The permutation the node installs. Imported, never reimplemented."""
    import importlib.util
    sys.path.insert(0, str(REPO.parent.parent))
    path = REPO / "vendor" / "sol_attn_minimax.py"
    spec = importlib.util.spec_from_file_location("_sol_vendor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latent_t(length):
    n = int(length)
    return ((n - 5) // 17) * 5 + 2 if n > 5 else 2


def video_span(seq, canvas, length):
    """(start, stop, grid). Video is the last segment of the packed sequence.

    `PackedLayout` appends target audio then target video and they are always
    the last two segments, so the video span can be found by subtraction
    without plumbing the layout out of the model. If the arithmetic overshoots
    the sequence, the canvas or length passed in is wrong and it says so rather
    than analysing the wrong rows.
    """
    w, h = canvas
    grid = (latent_t(length), h // 32, w // 32)
    n = grid[0] * grid[1] * grid[2]
    if n > seq:
        raise SystemExit(
            f"video span {n:,} exceeds captured sequence {seq:,}. The --canvas "
            f"or --length does not match the capture.")
    return seq - n, seq, grid


def centroid_fidelity(k, order, start, stop):
    """Mean cosine of each key to its own block's centroid, video rows only.

    `k` is [H, S, D]. `order` maps video-span position -> original row, so
    applying it reorders the video rows exactly as the node does.
    """
    rows = torch.arange(k.shape[1])
    rows[start:stop] = order + start
    kk = k.index_select(1, rows).float()

    # keep only blocks that lie wholly inside the video span; the boundary
    # block is shared with the conditioning rows, which the sink keeps exact
    first = (start + BLOCK - 1) // BLOCK
    last = stop // BLOCK
    out = []
    for b in range(first, last):
        seg = kk[:, b * BLOCK:(b + 1) * BLOCK, :]          # [H, 64, D]
        c = seg.mean(dim=1, keepdim=True)                   # [H, 1, D]
        cos = torch.nn.functional.cosine_similarity(seg, c, dim=-1)  # [H, 64]
        out.append(cos.mean().item())
    return torch.tensor(out)


def mass_concentration(q, k, order, start, stop, n_queries, heads, seed=0):
    """How many key blocks hold 90% of a query's attention mass.

    The attention weights themselves do not depend on the ordering: same q,
    same k, same softmax. Only which keys are grouped together changes. So this
    compares two groupings of one fixed set of weights, and any difference is
    the grouping alone.
    """
    g = torch.Generator().manual_seed(seed)
    H, S, D = k.shape
    hsel = torch.randperm(H, generator=g)[:heads]
    qsel = start + torch.randperm(stop - start, generator=g)[:n_queries]

    rows = torch.arange(S)
    rows[start:stop] = order + start
    inv = torch.empty_like(rows)
    inv[rows] = torch.arange(S)          # original row -> new position

    kk = k.index_select(1, rows).float()
    scale = D ** -0.5
    need = []
    for h in hsel.tolist():
        qh = q[h].float()
        kh = kk[h]
        for qi in qsel.tolist():
            w = torch.softmax(qh[inv[qi]] @ kh.T * scale, dim=-1)   # [S]
            nb = S // BLOCK
            per = w[: nb * BLOCK].view(nb, BLOCK).sum(1)
            srt, _ = torch.sort(per, descending=True)
            c = torch.cumsum(srt, 0)
            need.append(int((c < 0.90).sum()) + 1)
    return torch.tensor(need, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture", help="a qkv_*.pt written by h3_capture.py")
    ap.add_argument("--canvas", required=True, help="WIDTHxHEIGHT of the render")
    ap.add_argument("--length", type=int, required=True, help="pixel frame count")
    ap.add_argument("--queries", type=int, default=64,
                    help="query rows sampled per head for the mass test")
    ap.add_argument("--heads", type=int, default=4,
                    help="heads sampled for the mass test")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    name = Path(args.capture).name
    m = re.search(r"_b(\d+)_s(\d+)\.pt$", name)
    if m and m.group(1) == "0":
        print("NOTE: block 0. Early-layer attention is close to uniform, so this\n"
              "  block understates what a router has to work with. Read it\n"
              "  alongside a deep block before concluding anything.\n")

    # weights_only=True: h3_capture.py writes a bare {"q","k","v"} dict of
    # tensors, so nothing here needs the pickle machinery, and these files are
    # gigabytes of scratch that a later reader may not have produced themselves.
    d = torch.load(args.capture, map_location=args.device, weights_only=True)
    q, k = d["q"][0], d["k"][0]                    # [H, S, D]
    H, S, D = k.shape
    w, h = (int(v) for v in args.canvas.lower().split("x"))
    start, stop, grid = video_span(S, (w, h), args.length)

    print(f"capture {name}")
    print(f"  heads {H}, sequence {S:,}, head_dim {D}")
    print(f"  video span [{start:,}, {stop:,}) = {stop-start:,} rows, grid {grid}")
    print(f"  conditioning rows before video: {start:,}\n")

    vendor = load_shipped_morton()
    n = stop - start
    pad = (-start) % BLOCK
    orders = {"raster": torch.arange(n)}
    for curve in ("2d_frame", "3d"):
        perm, _ = vendor.morton_perm(grid, "cpu", curve)
        orders[f"morton_{curve}"] = torch.roll(perm, pad) if pad else perm

    # Our added curve, from the same module the runtime patch uses, so the
    # thing measured here is the thing that would run.
    sys.path.insert(0, str(REPO))
    import sol_curves
    # Two calls, and only the first is a gate. The square is the correctness
    # check on hilbert_d -- zero there is the defining property, so a non-zero
    # means the implementation is broken and nothing below is worth reading.
    # The rectangle is the DESCRIPTION of the ordering this run actually scores:
    # hilbert_perm clips the grid out of the next power of two, which splices
    # the curve, and that is expected rather than a failure. Reported, never
    # gated -- no threshold for it has been established. Until 2026-08-16 only
    # the square was called here, which asserted "never jumps" against the one
    # input on which it cannot be false. See sol_curves.verify_adjacency.
    bad = sol_curves.verify_adjacency(64)
    if bad:
        raise SystemExit(f"hilbert curve is broken: {bad} non-adjacent steps")
    splices = sol_curves.verify_adjacency(height=grid[1], width=grid[2])
    print(f"  hilbert on {grid[1]}x{grid[2]}: {splices} non-adjacent steps of "
          f"{grid[1] * grid[2] - 1} within a frame "
          f"(0 of 4095 on the 64x64 square; clipping is why)\n")
    hperm, _ = sol_curves.hilbert_perm(grid, "cpu")
    orders["hilbert"] = torch.roll(hperm, pad) if pad else hperm

    print("TEST 1  centroid fidelity: mean cosine of a key to its block centroid")
    print("        higher is better. this IS link 5.\n")
    print(f"  {'ordering':<18}{'mean cos':>10}{'p10':>9}{'min':>9}{'blocks':>9}")
    base = None
    for label, order in orders.items():
        cos = centroid_fidelity(k, order, start, stop)
        if base is None:
            base = cos.mean().item()
        print(f"  {label:<18}{cos.mean():>10.4f}{cos.quantile(0.10):>9.4f}"
              f"{cos.min():>9.4f}{len(cos):>9}")
    print()

    print("TEST 2  mass concentration: key blocks holding 90% of a query's mass")
    print("        lower is better. this is upstream's stated mechanism.\n")
    print(f"  {'ordering':<18}{'mean blocks':>13}{'median':>9}{'p90':>9}")
    for label, order in orders.items():
        need = mass_concentration(q, k, order, start, stop,
                                  args.queries, args.heads)
        print(f"  {label:<18}{need.mean():>13.1f}{need.median():>9.0f}"
              f"{need.quantile(0.90):>9.0f}")

    print("\nReading it. If TEST 1 does not improve under Morton, link 5 is false")
    print("and the canvas result in docs/morton.md predicts nothing about output.")
    print("If TEST 2 does not improve, the 'concentrates the mass' mechanism does")
    print("not operate on H3, and the tau-by-Morton experiment loses its premise.")


if __name__ == "__main__":
    main()
