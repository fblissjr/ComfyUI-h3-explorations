#!/usr/bin/env python3
"""Sol's token reorder must be invisible: a forward with it on equals the forward with it off.

The reorder (`sol_attn_h3.install_h3_morton`) permutes the video rows of the
hidden states before the first DiT block, permutes the RoPE table and the
per-token modulation indices to match, and restores the order after the last
block. Everything in between is either per-token or attention, and attention is
permutation-equivariant when its positions move with its tokens, so the output
must be the same tensor. Nothing checked that until 2026-09-17: every check that
touched the reorder tested the permutation in isolation, and a code review that
day found the per-token modulation indices were NOT being permuted, which no
existing check could have seen because none ran a forward with the reorder on.

This runs the real hooks on a small stub of the model: blocks that do what an H3
block does to a token (modulate it by its segment's row, attend with a
position-dependent score, gate the residual), on CPU, in float64 so the
comparison is about the wiring and not about summation order. Cases:

  uniform_modulation     every segment modulated by one integer row
  per_token_modulation   the video segment carries a LongTensor of per-token
                         rows, as a non-uniform denoise mask produces; the case
                         that was broken
  every_curve            each name in `MORTON_CURVES`
  misaligned_mask        a per-token row that does not cover the video span
                         exactly: the reorder must DECLINE (output still equal,
                         because nothing was permuted) rather than guess
  unknown_curve_refused  `morton_perm` raises on a name it does not know
  can_fail               the same comparison with the modulation fix bypassed
                         must NOT be equal, or this check proves nothing

No GPU, no model, no server.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _live_sol import live_sol  # noqa: E402

GRID = (3, 4, 6)                     # latent T, H, W of the stub's video segment
TEXT, AUDIO = 7, 9                   # rows before it; deliberately not multiples of 64
NV = GRID[0] * GRID[1] * GRID[2]
START, STOP = TEXT + AUDIO, TEXT + AUDIO + NV
DIM = 8


class Block(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(DIM, DIM, generator=g, dtype=torch.float64) / DIM ** 0.5)

    def forward(self, h, t_emb, mod_segments, rope):
        x = h.clone()
        for a, b, row in mod_segments:                     # per-segment or per-token modulation
            x[a:b] = x[a:b] * (1.0 + t_emb[row]).reshape(-1, DIM) if torch.is_tensor(row) else x[a:b] * (1.0 + t_emb[row])
        pos = rope[0, :, 0]                                # [S, DIM]: position features, one row per token
        qk = (x @ self.w + pos)
        att = torch.softmax(qk @ qk.T / DIM ** 0.5, dim=-1) @ x
        return h + att


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block(s) for s in range(3))

    def rope_freqs(self, position_ids, device):
        g = torch.Generator().manual_seed(99)
        table = torch.randn(position_ids.shape[-1], DIM, generator=g, dtype=torch.float64)
        return table[None, :, None, :]                     # [1, S, 1, DIM]: rows move with tokens on dim 1

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        rope = self.rope_freqs(kwargs["position_ids"], x.device)
        h = x
        for block in self.blocks:
            h = block(h, timestep, kwargs["mod_segments"], rope)
        return h


def _run(node, morton, curve, mod_segments, x, t_emb, position_ids):
    model = Model().double()
    # The stub's module needs a PackedLayout for the install to patch; the span
    # itself is registered by hand, which is what that patch does for real.
    mod = sys.modules[type(model).__module__]
    if not hasattr(mod, "PackedLayout"):
        mod.PackedLayout = type("PackedLayout", (), {"__init__": lambda self, *a, **k: None})
    node.install_h3_morton(model)
    layout = types.SimpleNamespace(segments=[(0, TEXT, "text"), (TEXT, START, "audio"), (START, STOP, "video")])
    node._SPANS[id(position_ids)] = (layout, (START, STOP), (TEXT, START), (START, STOP, GRID))
    opts = {"sol_morton": morton, "sol_morton_curve": curve}
    with torch.no_grad():
        return model._forward(x, t_emb, None, transformer_options=opts,
                              position_ids=position_ids, mod_segments=mod_segments)


def main() -> int:
    node = live_sol()
    torch.manual_seed(0)
    x = torch.randn(STOP, DIM, dtype=torch.float64)
    t_emb = torch.randn(6, DIM, dtype=torch.float64) * 0.3
    position_ids = torch.arange(STOP)[None]
    uniform = [(0, TEXT, 0), (TEXT, START, 1), (START, STOP, 2)]
    per_token = [(0, TEXT, 0), (TEXT, START, 1), (START, STOP, torch.randint(2, 6, (NV,)))]
    misaligned = [(0, TEXT, 0), (TEXT, START + 5, torch.randint(1, 3, (START + 5 - TEXT,))), (START + 5, STOP, 2)]

    problems = []

    def same(label, segs, curve="3d", want_equal=True):
        off = _run(node, False, curve, segs, x, t_emb, position_ids)
        on = _run(node, True, curve, segs, x, t_emb, position_ids)
        err = float((on - off).abs().max())
        ok = (err < 1e-9) == want_equal
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}: max |on - off| = {err:.2e}")
        if not ok:
            problems.append(label)

    print("Sol's token reorder leaves the forward unchanged:")
    same("uniform_modulation", uniform)
    same("per_token_modulation", per_token)
    for curve in node.MORTON_CURVES:
        same(f"every_curve[{curve}]", per_token, curve)
    same("misaligned_mask (declines)", misaligned)

    try:
        node.morton_perm(GRID, "cpu", "zigzag")
        print("  FAIL  unknown_curve_refused: accepted 'zigzag'"); problems.append("unknown_curve_refused")
    except ValueError:
        print("  ok    unknown_curve_refused")

    # The control: put the pre-fix behaviour back (modulation rows left in
    # raster order) and the per-token case has to come out DIFFERENT.
    fixed = node._permute_mod_segments
    node._permute_mod_segments = lambda segs, start, stop, perm: segs
    try:
        same("can_fail (fix bypassed, must differ)", per_token, want_equal=False)
    finally:
        node._permute_mod_segments = fixed

    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s): {problems}")
        return 1
    print("\nthe reorder is invisible to the model, per-token modulation included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
