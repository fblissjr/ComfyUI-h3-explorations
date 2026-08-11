#!/usr/bin/env python3
"""Reference-image sizing: does ComfyUI give a reference the rows it should?

The gap. ComfyUI sizes reference images with `min(1.0, 2048 / min(w, h))`
(`comfy_extras/nodes_minimax_h3.py:226`); the reference pipeline uses
`2048 / min(w, h)` with no clamp (diffusers
`modular_pipelines/minimax_h3/before_encoder.py:490`). Same constant, same
round-to-32. ComfyUI simply never upscales, so a reference below 2048 on its
short edge reaches the DiT smaller than the released pipeline would send it,
and reference tokens are latent rows.

Claims, i.e. what breaks if a case is deleted:

  matches the reference     `reference` mode reproduces the unclamped rule
                            for sizes above AND below 2048. Getting only the
                            downscale right is the bug we already have.
  down_only reproduces      the `down_only` arm must be exactly ComfyUI's
  ComfyUI                   current behaviour, or the A/B compares our new
                            code against something nobody runs.
  the gap is real           a small reference must come out with strictly
                            more rows in `reference` than in `down_only`.
                            If these ever agree at every size, ComfyUI has
                            been fixed upstream and this node can retire.
  stock resize is a no-op   the composition claim. After this node, the
                            stock node's own scale is min(1.0, 2048/2048) =
                            1.0, so its resize must be bit-identical. If it
                            is not, we are resizing twice and the second one
                            is lossy.
  aspect refused            the reference rejects reference images outside
                            1:4..4:1 and ComfyUI does not
  cost is reported          vision_tokens must match what the DiT will
                            actually attend, since it is the only signal
                            that upscaling is not free

No CUDA and no model -- this is all geometry.

**The weak point, stated rather than hidden.** `reference_rule` and
`comfy_rule` below are hand transcriptions of the two sources. If a
transcription is wrong, every case built on it is confidently wrong in the
same direction, and cases 1, 2 and 4 all rest on them. Driving ComfyUI's
actual sizing would need a VAE and a loaded model, which is what this file
exists to avoid. Case 2b is the partial mitigation: it compares the node's
two modes against each other and needs no transcription at all, so it still
fires if both transcriptions drift together.

    python bench/check_reference_fit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # ComfyUI root

# ComfyUI's own imports must resolve BEFORE this repo goes on the path: the
# repo has its own top-level `nodes.py`, and comfy_extras does `import nodes`.
# Adding the repo first shadows ComfyUI's and the import dies somewhere
# confusing.
from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE,  # noqa: E402
                                           REF_IMAGE_SHORT_EDGE, _resize)

sys.path.insert(0, str(HERE.parent))              # this repo
from reference_fit import MiniMaxH3ReferenceFit, _tokens  # noqa: E402

# below 2048, exactly 2048, above, non-square, and a portrait
SOURCES = [(512, 512), (768, 512), (2048, 2048), (4096, 2304), (512, 1024),
           (1000, 700)]

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def reference_rule(w, h, short_edge=REF_IMAGE_SHORT_EDGE):
    """diffusers before_encoder.py:490-492, transcribed."""
    scale = short_edge / min(w, h)
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def comfy_rule(w, h, short_edge=REF_IMAGE_SHORT_EDGE):
    """nodes_minimax_h3.py:226-228, transcribed."""
    scale = min(1.0, short_edge / min(w, h))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def fitted_size(w, h, allow_upscale):
    """Drive the NODE, not the helper.

    An earlier version of this file called `_fit` with a scale computed here,
    which meant the mode logic -- the only thing this node changes -- was
    never executed. Reintroducing ComfyUI's clamp into `reference` mode
    produced zero failures. Every case below now goes through `execute`.
    """
    out, _rows_out = MiniMaxH3ReferenceFit.execute(
        torch.rand(1, h, w, 3), allow_upscale=allow_upscale)
    return int(out.shape[2]), int(out.shape[1])


print("--- 1. 'reference' mode reproduces the reference's unclamped rule ---")
for w, h in SOURCES:
    got = fitted_size(w, h, True)
    check(f"{w}x{h} -> {got[0]}x{got[1]}", got == reference_rule(w, h),
          f"reference says {reference_rule(w, h)}")

print("\n--- 2. 'down_only' reproduces ComfyUI's current behaviour ---")
for w, h in SOURCES:
    got = fitted_size(w, h, False)
    check(f"{w}x{h} -> {got[0]}x{got[1]}", got == comfy_rule(w, h),
          f"ComfyUI says {comfy_rule(w, h)}")

print("\n--- 2b. the two modes must actually differ below 2048 ---")
# The load-bearing one. If `reference` and `down_only` ever agree on a small
# source, the mode switch is doing nothing and everything above is measuring
# one code path twice.
for w, h in SOURCES:
    ref_size, down_size = fitted_size(w, h, True), fitted_size(w, h, False)
    smaller = min(w, h) < REF_IMAGE_SHORT_EDGE
    check(f"{w}x{h}: modes {'differ' if smaller else 'agree'}",
          (ref_size != down_size) == smaller,
          f"reference {ref_size} vs down_only {down_size}")

print("\n--- 3. the gap: small references lose rows under ComfyUI's rule ---")
for w, h in SOURCES:
    ref_tokens = _tokens(*reference_rule(w, h))
    comfy_tokens = _tokens(*comfy_rule(w, h))
    smaller = min(w, h) < REF_IMAGE_SHORT_EDGE
    check(f"{w}x{h}: reference {ref_tokens} rows vs ComfyUI {comfy_tokens}",
          (ref_tokens > comfy_tokens) == smaller,
          f"({'expected a gap' if smaller else 'expected none'}"
          + (f", {ref_tokens / comfy_tokens:.0f}x)" if comfy_tokens and smaller else ")"))

print("\n--- 3b. the token count matches the model's own patchify ---")
# Graded against comfy's `_frame_grid`, which is what actually builds the
# reference block's rows, rather than against `_tokens` on both sides. Until
# 2026-08-11 this section compared the node's arithmetic with itself and
# passed while over-reporting every figure by 4x: it counted VAE latent
# cells and the DiT patchifies those 2x2 before attending them.
try:
    from comfy.ldm.minimax.model import _frame_grid
    for w, h in SOURCES:
        tw, th = reference_rule(w, h)
        want = _frame_grid(th // 16, tw // 16)[0].shape[0]
        check(f"{w}x{h}: {_tokens(tw, th)} tokens vs model {want}",
              _tokens(tw, th) == want,
              f"node says {_tokens(tw, th)}, _frame_grid says {want}")
except ImportError:
    print("  (skipped: comfy not importable)")

print("\n--- 4. after this node, the stock node's own resize is a no-op ---")
for w, h in SOURCES:
    out, rows = MiniMaxH3ReferenceFit.execute(torch.rand(1, h, w, 3),
                                              allow_upscale=True)
    fh, fw = int(out.shape[1]), int(out.shape[2])
    # exactly what nodes_minimax_h3.py then does to it
    tw, th = comfy_rule(fw, fh)
    again = _resize(out, tw, th, "disabled")
    check(f"{w}x{h}: stock path is identity", torch.equal(again, out),
          f"fitted {fw}x{fh}, stock would use {tw}x{th}")
    check(f"{w}x{h}: vision_tokens matches the fitted size", rows == _tokens(fw, fh))

print("\n--- 5. the reference's 1:4..4:1 limit on reference images ---")
for (w, h), want_ok in [((2048, 2048), True), ((4096, 1024), True),
                        ((1024, 4096), True), ((4200, 1024), False),
                        ((1024, 4200), False)]:
    try:
        MiniMaxH3ReferenceFit.execute(torch.rand(1, h, w, 3))
        raised = False
    except RuntimeError:
        raised = True
    check(f"{w}x{h} (aspect {w/h:.3g}): "
          f"{'accepted' if want_ok else 'refused'}", raised != want_ok)

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
