#!/usr/bin/env python3
"""Sol's token reorder must be invisible: a forward with it on equals the forward with it off.

The reorder (`sol_attn_h3.install_h3_morton`) permutes the target video rows on
their way into the video embedder, permutes the rows of `position_ids` to match,
and restores the order on the final layer's video output. Everything in between
is either per-token or attention, and attention is permutation-equivariant when
its positions move with its tokens, so the output must be the same tensor.

Until 2026-09-18 the same work was done in a pre-hook on block 0 and a hook on
the last block. Core now records one allocation plan per block and replays it
for all of them, so a block that allocates differently fails the forward; the
reorder therefore does nothing inside a block, and this check holds it to that.

This runs the real hooks on a stub with the shape of core's forward (publish
the layout, embed rows through `video_patch_proj`, assemble by segment, RoPE
from `position_ids`, a block stack, a per-token `final_layer` returning
`(video, audio)`), on CPU, in float64 so the comparison is about the wiring and
not about summation order. Cases:

  text_to_video          the video rows are the embedder's whole input
  with_conditioning      a `cond` segment shares the embedder, so the target
                         rows are a slice at a non-zero offset
  every_curve            each name in `MORTON_CURVES`
  masked (declines)      a non-uniform denoise mask makes modulation per token;
                         the reorder must DECLINE, and block 0 must see the
                         same hidden states as with the reorder off
  no_block_hooks         the install leaves no hook on any block
  unknown_curve_refused  `morton_perm` raises on a name it does not know
  can_fail               the same comparison with the restore removed, and with
                         the positions left in raster order, must NOT be equal,
                         or this check proves nothing

An "equal" verdict is only accepted where block 0 demonstrably saw DIFFERENT
hidden states with the reorder on; without that, a reorder that silently never
ran would pass every case.

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

GRID = (3, 4, 6)                     # token grid of the stub's video segment
LATENT = (GRID[0], GRID[1] * 2, GRID[2] * 2)   # what PackedLayout is given: 2x2 patches
TEXT, AUDIO, COND = 7, 9, 5          # deliberately not multiples of 64
NV = GRID[0] * GRID[1] * GRID[2]
DIM, PATCH = 8, 6


def _layout(cond):
    segments, at = [], 0
    for kind, n in (("text", TEXT), ("cond", COND if cond else 0), ("audio", AUDIO), ("video", NV)):
        if n:
            segments.append((at, at + n, kind))
            at += n
    g = torch.Generator().manual_seed(7)
    position_ids = torch.rand(at, 3, generator=g, dtype=torch.float64) * 10
    return types.SimpleNamespace(segments=segments, position_ids=position_ids, seq_len=at)


class Block(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(DIM, DIM, generator=g, dtype=torch.float64) / DIM ** 0.5)
        self.seen = None

    def forward(self, h, t_emb, mod_segments, rope):
        self.seen = h.clone()
        x = h.clone()
        for a, b, row in mod_segments:                     # per-segment or per-token modulation
            x[a:b] = x[a:b] * (1.0 + t_emb[row])
        qk = x @ self.w + rope                             # position features, one row per token
        att = torch.softmax(qk @ qk.T / DIM ** 0.5, dim=-1) @ x
        return h + att


class FinalLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_out = torch.nn.Linear(DIM, PATCH, dtype=torch.float64)
        self.audio_out = torch.nn.Linear(DIM, 3, dtype=torch.float64)

    def forward(self, x, t_emb, video_seg, audio_seg):
        def mod(seg):
            a, b, row = seg
            return x[a:b] * (1.0 + t_emb[row])
        return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.video_patch_proj = torch.nn.Linear(PATCH, DIM, dtype=torch.float64)
        self.blocks = torch.nn.ModuleList(Block(s) for s in range(3))
        self.final_layer = FinalLayer()
        g = torch.Generator().manual_seed(99)
        self.rope_w = torch.randn(3, DIM, generator=g, dtype=torch.float64)

    def rope_freqs(self, position_ids, device):
        return torch.sin(position_ids @ self.rope_w)       # elementwise per row, as core's is

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None,
                 denoise_mask=None, **kwargs):
        layout = minimax_payload["layout"]
        transformer_options["minimax_h3_layout"] = layout  # core publishes it before embedding
        t_emb = timestep
        rows_t = None
        if denoise_mask is not None and bool(denoise_mask.min() != denoise_mask.max()):
            rows_t = (denoise_mask.reshape(-1) > 0.5).long() + 3
        mod_segments = [(a, b, rows_t if kind == "video" and rows_t is not None else i % 3)
                        for i, (a, b, kind) in enumerate(layout.segments)]
        embed = self.video_patch_proj(torch.cat([context["cond_rows"], x]) if context["cond_rows"] is not None else x)
        h = torch.empty(layout.seq_len, DIM, dtype=torch.float64)
        voff = 0
        for a, b, kind in layout.segments:
            if kind in ("cond", "video"):
                h[a:b] = embed[voff:voff + b - a]
                voff += b - a
            else:
                h[a:b] = context[kind]
        rope = self.rope_freqs(layout.position_ids, x.device)
        for block in self.blocks:
            h = block(h, t_emb, mod_segments, rope)
        video_seg = next(s for s in mod_segments if layout.segments[mod_segments.index(s)][2] == "video")
        audio_seg = next(s for s in mod_segments if layout.segments[mod_segments.index(s)][2] == "audio")
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg)
        return torch.cat([v.reshape(-1), a.reshape(-1)])


def _run(node, morton, curve, cond, mask=None, sabotage=None):
    """One forward; returns (output, hidden states block 0 saw, block hook count)."""
    model = Model()
    # The stub's module needs a PackedLayout for the install to patch; the span
    # itself is registered by hand, which is what that patch does for real.
    mod = sys.modules[type(model).__module__]
    if not hasattr(mod, "PackedLayout"):
        mod.PackedLayout = type("PackedLayout", (), {"__init__": lambda self, *a, **k: None})
    stub_rope = model.rope_freqs
    node.install_h3_morton(model)
    if sabotage == "no_restore":
        model.final_layer._forward_hooks.clear()
    elif sabotage == "raster_positions":
        model.rope_freqs = stub_rope
    layout = _layout(cond)
    video = next((a, b) for a, b, k in layout.segments if k == "video")
    audio = next((a, b) for a, b, k in layout.segments if k == "audio")
    node._SPANS[id(layout.position_ids)] = (layout, video, audio, node._video_span(layout, *LATENT))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(NV, PATCH, generator=g, dtype=torch.float64)
    context = {"text": torch.randn(TEXT, DIM, generator=g, dtype=torch.float64),
               "audio": torch.randn(AUDIO, DIM, generator=g, dtype=torch.float64),
               "cond_rows": torch.randn(COND, PATCH, generator=g, dtype=torch.float64) if cond else None}
    t_emb = torch.randn(6, DIM, generator=g, dtype=torch.float64) * 0.3
    opts = {"sol_morton": morton, "sol_morton_curve": curve}
    with torch.no_grad():
        out = model._forward(x, t_emb, context, transformer_options=opts,
                             minimax_payload={"layout": layout}, denoise_mask=mask)
    hooks = sum(len(b._forward_pre_hooks) + len(b._forward_hooks) for b in model.blocks)
    return out, model.blocks[0].seen, hooks


def main() -> int:
    node = live_sol()
    problems = []
    mask = (torch.arange(NV).reshape(GRID) % 5 == 0).double()[None, None]

    def same(label, *, curve="3d", cond=False, mask=None, sabotage=None,
             want_equal=True, want_active=True):
        off, seen_off, _ = _run(node, False, curve, cond, mask)
        on, seen_on, hooks = _run(node, True, curve, cond, mask, sabotage)
        err = float((on - off).abs().max())
        active = not torch.equal(seen_on, seen_off)
        ok = (err < 1e-9) == want_equal and active == want_active and hooks == 0
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}: max |on - off| = {err:.2e}, "
              f"reorder {'ran' if active else 'did not run'}, {hooks} hook(s) on blocks")
        if not ok:
            problems.append(label)

    print("Sol's token reorder leaves the forward unchanged, and stays out of the blocks:")
    same("text_to_video")
    same("with_conditioning", cond=True)
    for curve in node.MORTON_CURVES:
        same(f"every_curve[{curve}]", curve=curve, cond=True)
    same("masked (declines)", cond=True, mask=mask, want_active=False)
    same("uniform mask (does not decline)", mask=torch.ones_like(mask))

    try:
        node.morton_perm(GRID, "cpu", "zigzag")
        print("  FAIL  unknown_curve_refused: accepted 'zigzag'"); problems.append("unknown_curve_refused")
    except ValueError:
        print("  ok    unknown_curve_refused")

    # The controls: take one step of the reorder away and the output has to
    # come out DIFFERENT, or an "equal" above means nothing.
    same("can_fail (no restore, must differ)", cond=True, sabotage="no_restore", want_equal=False)
    same("can_fail (positions left in raster order, must differ)", cond=True,
         sabotage="raster_positions", want_equal=False)

    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s): {problems}")
        return 1
    print("\nthe reorder is invisible to the model and does nothing inside a block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
