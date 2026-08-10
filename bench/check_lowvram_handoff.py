#!/usr/bin/env python3
"""Check that the sage forward survives KJNodes' low-VRAM block patch.

KJNodes' `MiniMaxLowVRAMAttention` replaces the *block* forward with one
that hands `x` to attention inside a single-item list, so the block's normed
`h` can be freed right after the qkv GEMM:

    x = _mod_gate(x, gate_msa, self.attn([h], rope_freqs=...), mod_segments)

That block patch is installed unconditionally. Its *attention* patch is not:
it skips the key if another patch already owns it. So in a graph with both
that node and ours, our forward receives the list in either node order --
and Sol-Attn's compose gate passes the list through untouched on calls it
declines, which is a second route to the same place.

Claims, i.e. what breaks if a case is deleted:
  tensor still works    the ordinary path, with no KJNodes node in the graph
  list is unwrapped     the case this file exists for. Before the fix this
                        raised AttributeError: 'list' object has no attribute
                        'shape' -- outside the try/except, so it killed the
                        render rather than falling back
  list is emptied       the block hands over its only reference expecting it
                        to be taken; leaving it in place defeats the point of
                        the hand-off even though we then hold it ourselves
  fallback still works  the kernel raising must reach _stock_forward with a
                        real tensor, not a list. Unwrapping only on the happy
                        path would move the crash rather than fix it

No CUDA, no model, no sage: the module, the kernel and the stock forward are
all stubs, because the routing is the whole claim. It does need ComfyUI
importable, since the forward imports comfy.model_management for the rope
cast -- run it from this repo inside a ComfyUI tree.

    python bench/check_lowvram_handoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# .../ComfyUI/custom_nodes/<this repo>/bench -> the ComfyUI root
sys.path.insert(0, str(HERE.parents[2]))

import attention  # noqa: E402
from attention import make_minimax_attn_forward  # noqa: E402

S, HEADS, DIM = 8, 2, 4


class FakeAttn:
    """Just enough of comfy.ldm.minimax.model.Attention to drive the forward."""

    def __init__(self):
        self.heads, self.head_dim = HEADS, DIM
        self.out_proj = lambda t: t
        self.q_norm = self.k_norm = lambda t: t

    def qkv_proj(self, x):
        return torch.zeros(x.shape[0], HEADS * DIM * 3)


def run(x, kernel):
    forward = make_minimax_attn_forward(kernel, {})
    return forward(FakeAttn(), x, rope_freqs=None)


def main():
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    def good_kernel(qkv, **kw):
        q, _k, _v = qkv
        qkv.clear()
        return torch.zeros(1, S, HEADS, DIM)

    print("sage forward against the low-VRAM block hand-off")

    def tensor_path():
        out = run(torch.zeros(S, HEADS * DIM), good_kernel)
        assert out.shape == (S, HEADS * DIM), out.shape

    def list_path():
        handed = [torch.zeros(S, HEADS * DIM)]
        out = run(handed, good_kernel)
        assert out.shape == (S, HEADS * DIM), out.shape
        assert handed == [], "the block's list was not emptied; the hand-off is ignored"

    def fallback_path():
        # The kernel raising must reach _stock_forward with a real tensor.
        # Spy on it rather than letting it run: the real one re-enters
        # ComfyUI's Attention.forward, which a stub module cannot satisfy,
        # and a failure in there would be indistinguishable from the bug
        # this case is about.
        def angry_kernel(qkv, **kw):
            qkv.clear()
            raise RuntimeError("kernel says no")

        seen = []
        real = attention._stock_forward
        attention._stock_forward = lambda self, x, rf, to: seen.append(x)
        try:
            run([torch.zeros(S, HEADS * DIM)], angry_kernel)
        finally:
            attention._stock_forward = real
        assert seen, "the kernel raised but the fallback never ran"
        assert torch.is_tensor(seen[0]), f"fallback got {type(seen[0]).__name__}, not a tensor"

    check("plain tensor still works", tensor_path)
    check("single-item list is unwrapped", list_path)
    check("kernel failure falls back with a tensor", fallback_path)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
