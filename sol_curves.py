"""Alternative token orderings for Sol-Attn, in the shape its node already expects.

Sol-Attn's block router summarises each 64-token block with one centroid, so the
only thing a token ordering can change is which 64 tokens share a block. kijai's
node ships two orderings, both Morton (Z-order): `3d` and `2d_frame`. This adds
Hilbert, and it exists because of one measured weakness in Z-order.

**Z-order jumps.** Consecutive points on a Z-order curve are often far apart in
space, because the curve crosses quadrant boundaries: at side 64, 2047 of 4095
consecutive steps are non-adjacent. That is what fragments a 64-token block on a
grid whose dimensions are not multiples of 8, and H3's default 1344x768 canvas
(latent 24x42) is exactly such a grid. A Hilbert curve never jumps; consecutive
points are always neighbours. Measured on the 1344x768 latent grid, blocks that
form a single connected region: Z-order 60%, Hilbert 90%.

That is a geometry result. Whether it survives contact with real activations is
what `bench/analyze_capture.py` is for.

## Why this is not a fork

`_perm_for` in the vendored node calls `morton_perm(...)` as a plain module
global, and the curve name arrives as a string through
`transformer_options["sol_morton_curve"]`. So an ordering can be added by
rebinding that one name and passing a different string, with no edit to
upstream's file and no kernel rebuild. `vendor/README.md`'s preference order
asks for exactly that: upstream it, else wrap it, and only fork with the
divergence recorded.

`install()` does the rebinding. It resolves the live module **by identity**
rather than by name, because a running ComfyUI can hold two module objects for
one file and patching the wrong one looks exactly like success -- see the
`comfy_extras` trap in CLAUDE.md, which cost a day on 2026-08-15's predecessor.

## The curves

  `hilbert`     2D Hilbert within each latent frame, frames left in original
                order. The direct replacement for `2d_frame`, and the one with
                measured geometry behind it.

Deliberately not implemented yet: a 3D Hilbert over the whole latent volume.
That is the natural counterpart to Morton `3d`, which scores best of the
shipped curves on captured activations, but a 3D Hilbert is materially more
code than a 2D one and nothing yet says the curve family matters more than the
dimensionality. Measure first.
"""

from __future__ import annotations

import logging
import sys

import torch

_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

#: Curve names this module adds. Anything else is delegated to upstream.
OURS = ("hilbert",)


def hilbert_d(x: int, y: int, side: int) -> int:
    """Distance along a Hilbert curve of order log2(side), for a point in a
    side x side square.

    The standard xy2d rotation algorithm. `side` must be a power of two; a
    non-power-of-two grid is handled by computing on the next power of two and
    dropping the out-of-range points, which preserves the curve's ordering on
    the points that remain.
    """
    rx = ry = 0
    d = 0
    s = side >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def verify_adjacency(side: int = 64) -> int:
    """Non-adjacent steps along the curve. Zero is the defining property.

    Called before the permutation is trusted, because a subtly wrong Hilbert is
    still a valid permutation and would silently produce a worse ordering that
    looks like a real result.
    """
    pts = sorted(((x, y) for x in range(side) for y in range(side)),
                 key=lambda p: hilbert_d(p[0], p[1], side))
    if len(set(pts)) != side * side:
        raise RuntimeError("hilbert_d is not injective; the permutation is invalid")
    return sum(1 for a, b in zip(pts, pts[1:])
               if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1)


def hilbert_perm(grid, device="cpu"):
    """Per-frame 2D Hilbert permutation and its inverse, matching `morton_perm`.

    Frames keep their original order and never mix, which is the `2d_frame`
    convention and is right for H3: `FRAME_PER_TOKEN` is (1, 4, 4, 4, 4), so
    index-adjacent latent frames are 1 or 4 real frames apart.
    """
    key = (tuple(int(x) for x in grid), "hilbert")
    hit = _CACHE.get(key)
    if hit is None:
        frames, height, width = key[0]
        side = 1
        while side < max(height, width):
            side <<= 1
        within = sorted(range(height * width),
                        key=lambda i: hilbert_d(i % width, i // width, side))
        area = height * width
        perm = torch.tensor([f * area + i for f in range(frames) for i in within],
                            dtype=torch.int64)
        hit = (perm, torch.argsort(perm))
        _CACHE[key] = hit
    return hit[0].to(device), hit[1].to(device)


def _live_modules(path):
    """Every loaded module object whose file is `path`, resolved by identity.

    A running ComfyUI can hold more than one module object for one file: the
    custom-node loader registers by file path, and a dotted import of the same
    file builds a second, independent object with its own copy of every
    module-level function. Patching one and not the other logs success while the
    server goes on running the unpatched copy.
    """
    import os
    want = os.path.realpath(path)
    out = []
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f:
            continue
        try:
            if os.path.realpath(f) == want and hasattr(module, "morton_perm"):
                out.append(module)
        except OSError:
            continue
    return out


def install(vendor_path) -> int:
    """Rebind `morton_perm` on every live copy of the node so it knows our curves.

    Idempotent, and returns how many module objects were patched. **Zero is a
    failure**, not a no-op: it means the node is not loaded, or is loaded from a
    different file than the one passed, and the caller should say so rather than
    report success.
    """
    patched = 0
    for module in _live_modules(vendor_path):
        original = getattr(module, "_curve_original_morton_perm", None)
        if original is None:
            original = module.morton_perm
            module._curve_original_morton_perm = original

        def dispatch(grid, device, curve="3d", _original=original):
            if curve in OURS:
                return hilbert_perm(grid, device)
            return _original(grid, device, curve)

        module.morton_perm = dispatch
        patched += 1
    if patched:
        logging.info(f"[sol_curves] added {OURS} to {patched} live copy(ies) of "
                     f"the Sol-Attn node")
    return patched
