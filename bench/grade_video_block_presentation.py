#!/usr/bin/env python3
"""Is ComfyUI's per-pair vision call equivalent to the vendor's one-clip call?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
checkpoint, no server, no CUDA.

**The claim under test, and who made it.**
`internal/codex/2026-08-21_h3-conditioning-qwen-independent-review.md` section 2
calls this "the important non-gap": ComfyUI presents a reference video to
Qwen3-VL as separate two-frame vision calls, each with `grid_thw = [1, H, W]`
(`comfy/text_encoders/minimax.py:176-180` driving `:85`), where LightX and
sglang process the whole sampled clip in one call at `[T, H, W]`. It argues the
two are structurally equivalent -- and then says, in its own words, "this should
be tensor-tested, but it is not a reason to redesign the presentation."

It was never tensor-tested. This is that test. Reading the source supports the
claim three ways, and reading is not testing:

  `qwen35.py:660-663`  cu_seqlens splits attention at every h*w, so attention
                       never crosses a frame even in a one-call presentation
  `qwen35.py:585-586`  rot_pos_emb repeats identical spatial coords per frame;
                       there is no temporal term
  `qwen35.py:594+`     fast_pos_embed_interpolate is a function of h and w

**Random weights are the right instrument here, not a shortcut.** The claim is
about how the tower ROUTES tokens -- attention scoping and position
construction -- which is a property of the architecture, not of what it
learned. Real weights would test the same graph and cost a 32B load. What
random weights cannot test is anything weight-dependent, and nothing in the
claim is.

**Two controls, and the file is worthless without them.** A harness that
reports "identical" proves nothing unless it can report "different". So the
same comparison runs against two presentations that MUST differ: the clip with
its pairs reordered, and the clip with one pair perturbed. If either matches,
the comparison is blind and the run says so and exits non-zero instead of
reporting a pass.

Exit 0 only when the equivalence holds AND both controls separate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ComfyUI"))

import torch  # noqa: E402

import comfy.ops  # noqa: E402
from comfy.text_encoders.qwen3vl import (  # noqa: E402
    QWEN3VL_VISION, QWEN3VL_VISION_COMMON, Qwen3VLVisionModel,
)

GRID_H, GRID_W = 8, 8          # merged-unit multiples; small enough for CPU
N_PAIRS = 6                     # what 124 frames subsampled at 2 fps becomes
SEED = 730451892
# Bit-exactness is not the bar and claiming it would be wrong: the two arms
# reduce different-length tensors, so fp32 reassociation alone moves the last
# bits. The bar is "same to well inside fp32 noise", and the controls below
# establish what a real difference looks like on this same scale.
TOL = 5e-5


def build_tower():
    cfg = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION["qwen3vl_32b"],
           "out_hidden_size": 5120}
    torch.manual_seed(SEED)
    tower = Qwen3VLVisionModel(cfg, device="cpu", dtype=torch.float32,
                               ops=comfy.ops.manual_cast)
    for p in tower.parameters():
        torch.nn.init.normal_(p, std=0.02)
    for m in tower.modules():
        if isinstance(m, torch.nn.LayerNorm) and m.weight is not None:
            torch.nn.init.ones_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    return tower.eval()


def make_pairs(cfg_patch=16, tps=2):
    """One `flatten` per two-frame pair, the shape process_video_block emits."""
    torch.manual_seed(SEED + 1)
    dim = 3 * tps * cfg_patch * cfg_patch
    return [torch.randn(GRID_H * GRID_W, dim) for _ in range(N_PAIRS)]


def per_pair(tower, pairs):
    """ComfyUI: one call per pair at grid [1, H, W], results concatenated."""
    merged, deep = [], None
    for f in pairs:
        g = torch.tensor([[1, GRID_H, GRID_W]], dtype=torch.long)
        with torch.no_grad():
            m, d = tower(f, g)
        merged.append(m)
        deep = [d[i] for i in range(len(d))] if deep is None else \
               [torch.cat([deep[i], d[i]], dim=0) for i in range(len(d))]
    return torch.cat(merged, dim=0), deep


def one_clip(tower, pairs):
    """Vendor: one call for the whole clip at grid [T, H, W]."""
    g = torch.tensor([[len(pairs), GRID_H, GRID_W]], dtype=torch.long)
    with torch.no_grad():
        return tower(torch.cat(pairs, dim=0), g)


def worst(a, b):
    d = max((x - y).abs().max().item() for x, y in zip(a[1], b[1]))
    return max((a[0] - b[0]).abs().max().item(), d)


def main() -> int:
    tower = build_tower()
    pairs = make_pairs()

    ours = per_pair(tower, pairs)
    vendor = one_clip(tower, pairs)
    delta = worst(ours, vendor)

    reordered = one_clip(tower, pairs[3:] + pairs[:3])
    nudged = list(pairs)
    torch.manual_seed(SEED + 2)
    nudged[2] = nudged[2] + torch.randn_like(nudged[2]) * 1e-3
    perturbed = one_clip(tower, nudged)

    d_reorder = worst(ours, reordered)
    d_perturb = worst(ours, perturbed)

    print(f"grid {GRID_H}x{GRID_W}, {N_PAIRS} two-frame pairs, "
          f"qwen3vl_32b vision config, random weights, fp32 CPU\n")
    print(f"  per-pair vs one-clip        max |delta| = {delta:.3e}   "
          f"(tolerance {TOL:.0e})")
    print(f"  CONTROL pairs reordered     max |delta| = {d_reorder:.3e}")
    print(f"  CONTROL one pair perturbed  max |delta| = {d_perturb:.3e}")

    blind = [n for n, d in (("reordered", d_reorder), ("perturbed", d_perturb))
             if d <= TOL]
    if blind:
        print(f"\nFAIL  the comparison is BLIND: control(s) {', '.join(blind)} "
              f"did not separate, so 'identical' above measures nothing.")
        return 1
    if delta > TOL:
        print(f"\nFAIL  the two presentations DIVERGE. ComfyUI's per-pair call "
              f"is not what the vendor's one-clip call computes, on the "
              f"reference-video path.")
        return 1
    print(f"\nok    equivalent, and the comparison can see a difference: both "
          f"controls separated by >= {min(d_reorder, d_perturb) / max(delta, 1e-12):.0f}x "
          f"the observed delta. The codex review's structural argument holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
