#!/usr/bin/env python3
"""Check VSA's cube reordering, and the refusal that stops it running dense.

Two things here can fail silently, and neither raises on its own.

**The reordering.** `vsa_attention._geometry` moves every row of the packed
sequence into a padded, cube-major order, and the output is moved back by the
inverse. A permutation that is subtly wrong -- a row landing in a block's dead
tail, two rows on one destination, the prefix leaking into the video blocks --
produces a tensor of the right shape and dtype, and the render succeeds. There
is no exception to catch. So the invariants are asserted directly, on shapes
chosen to be ragged in every axis at once, because a cube walk is exactly the
kind of code that is correct on a grid whose dimensions divide by four.

**The refusal.** `_gate_modules` is the only thing standing between a user and
a render that looks like VSA and is not. On stock ComfyUI a VSA checkpoint's
gate weights have no slot on the constructed model and are DROPPED on load with
a warning, so the model runs as the dense base and renders successfully. If
that refusal ever stops refusing, nothing downstream notices.

Claims, i.e. what breaks if a case is deleted:

  bijection          every source row reaches exactly one destination. A
                     collision silently overwrites a token.
  live rows only     no source row lands past its block's `block_len`. A row in
                     the dead tail is excluded from keys and its output is
                     unspecified, so a token would vanish from attention while
                     the render succeeded.
  block_len sums     the live-row counts account for every source row and
                     nothing more.
  prefix separation  prefix rows occupy exactly the blocks declared as sinks.
                     If a video row landed in the sink range it would be forced
                     exact; if a prefix row landed outside it, the conditioning
                     would be routed sparsely, which is the thing the sink
                     exists to prevent.
  round trip         scatter then gather is the identity on the source rows.
  cube membership    every block holds rows from ONE cube. This is the claim
                     that makes the tiling VSA rather than an arbitrary
                     regrouping, and it is the one an off-by-one in the walk
                     would break while leaving every other case green.
  a broken walk is   RED CONTROL. A deliberately corrupted permutation must
    caught           fail the invariants. Without it, the cases above prove
                     only that they agree with themselves.
  refuses without    `_gate_modules` returns a refusal, naming the draft PR,
    a gate           when the model has no `to_gate_compress`.
  refuses on a       a checkpoint with gates on some blocks and not others is
    partial set      refused rather than run half-VSA.

Needs neither CUDA nor a model nor a server; the geometry is integer arithmetic
and the refusals are exercised against stubs.

    python bench/check_vsa_geometry.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def load_vsa():
    sys.path.insert(0, str(REPO.parent.parent))
    spec = importlib.util.spec_from_file_location(
        "h3x", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["h3x"] = pkg
    spec.loader.exec_module(pkg)
    import importlib as il
    return il.import_module("h3x.vsa_attention")


# Ragged in every axis at once, because a cube walk is the kind of code that is
# correct whenever the grid divides by four. The first row is a realistic
# packed shape; the rest exist to break it.
SHAPES = [
    ((311, 96), (31, 48, 84)),      # text + target audio, a shipped canvas
    ((311,), (7, 6, 6)),            # ragged t, h and w together
    ((100, 64, 33), (5, 10, 7)),    # several prefix segments, one exactly 64
    ((64,), (4, 4, 4)),             # one cube, no padding anywhere
    ((1,), (1, 1, 1)),              # degenerate: a single video row
]

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def invariants(vsa, prefix, grid, destination=None):
    """Every geometry invariant for one shape. Returns a dict of name -> bool.

    `destination` overrides the computed permutation, which is how the red
    control feeds a corrupted one through the same assertions.
    """
    dest, block_len, prefix_blocks, padded = vsa._geometry(prefix, grid)
    if destination is not None:
        dest = destination
    total = sum(prefix) + math.prod(grid)
    rows = dest.tolist()
    prefix_rows = sum(prefix)

    out = {
        "bijection": len(rows) == total and len(set(rows)) == total,
        "block_len sums": int(block_len.sum()) == total,
        "round trip": None,
        "live rows only": all(0 <= d < padded and (d % 64) < int(block_len[d // 64])
                              for d in rows),
        "prefix separation": (all(rows[i] < prefix_blocks * 64 for i in range(prefix_rows))
                              and all(rows[i] >= prefix_blocks * 64
                                      for i in range(prefix_rows, total))),
    }
    x = torch.arange(total, dtype=torch.float32).unsqueeze(1)
    try:
        y = vsa._scatter_rows(x, dest, padded)
        out["round trip"] = bool(torch.equal(y[dest], x))
    except (IndexError, RuntimeError):
        out["round trip"] = False

    # Cube membership: each video block holds rows of one 4x4x4 cube. Recovered
    # from the source index rather than from the walk that built it, so this is
    # a second derivation and not a restatement.
    frames, height, width = grid
    ct, ch, cw = vsa.CUBE
    ok_cubes = True
    per_block = {}
    for source in range(prefix_rows, total):
        rel = source - prefix_rows
        t, rem = divmod(rel, height * width)
        h, w = divmod(rem, width)
        per_block.setdefault(rows[source] // 64, set()).add(
            (t // ct, h // ch, w // cw))
    ok_cubes = all(len(v) == 1 for v in per_block.values())
    out["cube membership"] = ok_cubes
    return out


def main():
    vsa = load_vsa()
    print("VSA cube geometry:\n")
    names = ["bijection", "block_len sums", "live rows only", "prefix separation",
             "round trip", "cube membership"]
    results = {n: True for n in names}
    for prefix, grid in SHAPES:
        got = invariants(vsa, prefix, grid)
        for n in names:
            results[n] = results[n] and got[n]
    _d, block_len, _p, padded = vsa._geometry(*SHAPES[0])
    total = sum(SHAPES[0][0]) + math.prod(SHAPES[0][1])
    for n in names:
        check(n, results[n], f"over {len(SHAPES)} shapes")
    print(f"\n  padding cost at {SHAPES[0][1]}: {padded - total} rows of "
          f"{padded} ({100 * (padded - total) / padded:.1f}%), which the kernel "
          f"skips via block_len but still stages.")

    print("\nred control -- a corrupted walk must not pass:")
    prefix, grid = SHAPES[0]
    dest, _bl, _pb, _pad = vsa._geometry(prefix, grid)
    broken = dest.clone()
    # Move one video row into a neighbouring block. It stays in range, stays
    # unique unless it collides, and is exactly the kind of off-by-one a cube
    # walk produces.
    broken[-1] = broken[-1] - 64
    got = invariants(vsa, prefix, grid, destination=broken)
    caught = [n for n in names if not got[n]]
    check("a broken walk is caught", bool(caught),
          f"failed {caught}" if caught else
          "the corrupted permutation passed EVERY invariant; these cases "
          "cannot see a misplaced row")

    print("\nthe refusal that stops a dense render passing for VSA:")

    class _Stub:
        """Model exposing `to_gate_compress` on the first `n` blocks only."""

        def __init__(self, n):
            self.n = n

        def get_model_object(self, path):
            index = int(path.split(".")[2])
            if index >= self.n:
                raise AttributeError(path)
            module = torch.nn.Linear(4, 4, bias=False)
            return module

    gates, why = vsa._gate_modules(_Stub(0), 50)
    check("refuses without a gate", gates is None and "15958" in (why or ""),
          "names the draft PR, so the message says what to DO"
          if gates is None else "a gateless model was accepted")

    gates, why = vsa._gate_modules(_Stub(30), 50)
    check("refuses on a partial set", gates is None and "block 30" in (why or ""),
          why.split(".")[0] if why else "a half-gated model was accepted")

    gates, why = vsa._gate_modules(_Stub(50), 50)
    check("accepts a full set", gates is not None and len(gates or []) == 50,
          f"{len(gates or [])} gates" if gates else f"refused: {why}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
