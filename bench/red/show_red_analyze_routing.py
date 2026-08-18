"""Show every control in `bench/analyze_routing.py` red, for the right reason.

Mutates by monkeypatching the imported module, never by editing the file, so
there is no way to leave a mutation behind.

This is the sole evidence behind `docs/checks.md`'s conditioning-rows row
(`open_experiments.md` #18). Until 2026-08-17 it carried NO expected-outcome
comparison at all -- it printed RED or GREEN and exited 0 either way -- so that
row cited a program which returned success whatever happened.

**Needs a capture, so it cannot run in a clone.** The path comes from the
`H3_CAPTURE` environment variable rather than a hardcoded location: a
home-directory path written into repo content is a leak by `path-privacy`'s
rule, and it would also name a machine this file has no business naming. Absent
-> exit 2, so "did not run" stays distinguishable from "passed".

    H3_CAPTURE=<qkv capture .pt> python bench/red/show_red_analyze_routing.py
"""
import os

import torch

from harness import Harness, MUTATION, NEAR_MISS, main, subject

AR = subject("bench/analyze_routing.py")

CAP = os.environ.get("H3_CAPTURE", "")
_H = Harness(subject="bench/analyze_routing.py", verdict=lambda ok: not ok)
_H.fixture(CAP or "/nonexistent",
           "set H3_CAPTURE to a qkv capture .pt. The eager reference comes from "
           "the installed comfy_kitchen, so a capture is the only thing this "
           "needs that a clone does not already have.")

pool = AR.load_eager()
ac = AR.load_capture_tools()
vendor = ac.load_shipped_morton()
d = torch.load(CAP, map_location="cpu", weights_only=True)
q, k = d["q"][0], d["k"][0]
start, stop, grid = ac.video_span(k.shape[1], (1344, 768), 124)
ORDERS = AR.orderings(grid, start, vendor, ["3d"])
HEADS = [0, 1]
SINK_KV = (start + 63) // 64

_REAL_BS = AR.block_stats
_REAL_EM = AR.exact_mask


def controls(orders_):
    return AR.run_controls(q, k, orders_, start, stop, 1.3, HEADS, pool, SINK_KV, 0)


def unmutated():
    return controls(ORDERS)


def patched(attr, fn, orders_=None):
    """Apply a monkeypatch for one case, always restoring it."""

    def run():
        real = getattr(AR, attr)
        setattr(AR, attr, fn)
        try:
            return controls(ORDERS if orders_ is None else orders_)
        finally:
            setattr(AR, attr, real)

    return run


def rolled_rows(q_, k_, order, st, sp, head, pool_):
    """Shift every block boundary by one row: same rows, different membership."""
    rows = torch.arange(k_.shape[1])
    rows[st:sp] = order + st
    rows = torch.roll(rows, 1)
    return _REAL_BS(q_, k_, rows[st:sp] - st, st, sp, head, pool_)


def ignore_order(q_, k_, order, st, sp, head, pool_):
    return _REAL_BS(q_, k_, torch.arange(sp - st), st, sp, head, pool_)


def video_only_kcvar(q_, k_, order, st, sp, head, pool_):
    """kcvar over the VIDEO span only, dropping conditioning centroids.

    `open_experiments.md` #18 requires the population to be every centroid the
    kernel pools. This models getting that wrong.
    """
    c, kcc, kcv, n, D = _REAL_BS(q_, k_, order, st, sp, head, pool_)
    keep = kcc[(st + 63) // 64:]
    return c, kcc, (keep - keep.mean(0, keepdim=True)).pow(2).mean(0), n, D


def build():
    bad = dict(ORDERS)
    bad["3d"] = ORDERS["3d"].clone()
    bad["3d"][0] = bad["3d"][1]

    h = _H
    h.baseline(unmutated)
    h.case("M1 ordering is not a permutation (duplicated an index)", MUTATION,
           lambda: controls(bad))
    h.case("M2 block boundaries shifted by one row (membership changes)", MUTATION,
           patched("block_stats", rolled_rows))
    h.case("M2b ordering ignored entirely (every call uses raster)", MUTATION,
           patched("block_stats", ignore_order))
    h.case("M3 batched threshold drifts from the naive one (tau*1.02)", MUTATION,
           patched("exact_mask", lambda c, kcc, kcv, tau, sc: _REAL_EM(c, kcc, kcv, tau * 1.02, sc)))
    h.case("M4 threshold ignores tau (pinned to 1.3, density cannot move)", MUTATION,
           patched("exact_mask", lambda c, kcc, kcv, tau, sc: _REAL_EM(c, kcc, kcv, 1.3, sc)))
    h.case("M5 kcvar over video blocks only (conditioning dropped)", MUTATION,
           patched("block_stats", video_only_kcvar))
    h.case("restored: unmutated again", NEAR_MISS, unmutated)
    return h


if __name__ == "__main__":
    main(build)
