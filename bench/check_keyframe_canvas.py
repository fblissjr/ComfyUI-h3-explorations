"""Keyframe geometry: does the canvas we hand H3 avoid distorting the keyframe?

Claims, one per case. Delete a case and you stop noticing the corresponding
failure:

1. `adapt_canvas` reproduces the reference's `resolve_canvas_size` -- 768 short
   edge, 768*1344 area cap, both axes rounded to 32 -- so deriving the canvas
   from a keyframe puts ComfyUI on the reference's default path.
2. A canvas derived from an image preserves that image's aspect to within the
   round-to-32 quantisation. This is the property the whole node exists for.
3. The stock node's first-frame resize is a NON-UNIFORM stretch whenever the
   keyframe aspect differs from the canvas. This is the defect; if this case
   ever goes green the defect is gone and the node can be retired.
4. Feeding a derived canvas makes that stretch a no-op, because the keyframe
   already has exactly the canvas dimensions.

Reference: coderef/diffusers .../modular_pipelines/minimax_h3/before_encoder.py
::MiniMaxH3ResizeStep, and modular_pipeline.py::resolve_canvas_size.

Run: python bench/check_keyframe_canvas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ComfyUI root: custom_nodes/<this repo>/bench -> up three
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, MAX_PIXELS, adapt_canvas, _resize

# (w, h) worth covering: square, portrait, ultrawide, a TRUE 16:9 (1.7778), the
# canvas's own 7:4 (1.75), and an odd size that is not a multiple of 32.
# 1920x1080 and 1344x768 are deliberately both here: the default canvas is 7:4,
# NOT 16:9, so a real 16:9 source is not a no-op. Round-to-32 means no canvas is
# exactly 16:9, which is the model's rule and not a ComfyUI choice.
SOURCES = [(1024, 1024), (768, 1024), (2560, 1080), (1920, 1080), (1344, 768), (1000, 700)]

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("--- 1. adapt_canvas matches the reference's canvas rule ---")
for w, h in SOURCES:
    cw, ch = adapt_canvas(w, h)
    check(f"{w}x{h} -> {cw}x{ch}: axes are multiples of {CANVAS_MULTIPLE}",
          cw % CANVAS_MULTIPLE == 0 and ch % CANVAS_MULTIPLE == 0)
    check(f"{w}x{h}: area within cap (+round-32 slack)",
          cw * ch <= MAX_PIXELS * 1.15, f"{cw * ch} vs {MAX_PIXELS}")

print("\n--- 2. derived canvas preserves source aspect ---")
for w, h in SOURCES:
    cw, ch = adapt_canvas(w, h)
    src, got = w / h, cw / ch
    # round-to-32 on both axes is the only permitted error
    tol = max(CANVAS_MULTIPLE / ch, CANVAS_MULTIPLE * cw / (ch * ch))
    check(f"{w}x{h}: aspect {src:.4f} -> {got:.4f}", abs(src - got) <= tol,
          f"delta={abs(src - got):.4f} tol={tol:.4f}")

print("\n--- 3. the defect: stock first-frame path stretches non-uniformly ---")
DEFAULT_W, DEFAULT_H = 1344, 768
for w, h in SOURCES:
    img = torch.zeros(1, h, w, 3)
    out = _resize(img, DEFAULT_W, DEFAULT_H, "disabled")
    sx, sy = DEFAULT_W / w, DEFAULT_H / h
    distortion = max(sx, sy) / min(sx, sy)
    mismatched = abs(w / h - DEFAULT_W / DEFAULT_H) > 1e-6
    check(f"{w}x{h} at {DEFAULT_W}x{DEFAULT_H}: distortion {distortion:.3f}x",
          (distortion > 1.01) == mismatched,
          "(expected >1 only when aspect differs)")
    check(f"{w}x{h}: output is exactly the canvas",
          tuple(out.shape[1:3]) == (DEFAULT_H, DEFAULT_W))

print("\n--- 4. derived canvas makes the stock stretch a no-op ---")
for w, h in SOURCES:
    cw, ch = adapt_canvas(w, h)
    fitted = _resize(torch.rand(1, h, w, 3), cw, ch, "disabled")
    # what the stock node then does to a keyframe already at canvas size
    first = _resize(fitted, cw, ch, "disabled")
    last = _resize(fitted, cw, ch, "center")
    check(f"{w}x{h}: first-frame path is identity", torch.equal(first, fitted),
          f"max|delta|={(first - fitted).abs().max():.3e}")
    check(f"{w}x{h}: last-frame path is identity", torch.equal(last, fitted),
          f"max|delta|={(last - fitted).abs().max():.3e}")

print("\n--- 5. the trained aspect range is enforced, as the reference does ---")
# diffusers' resolve_canvas_size raises outside 1:4..4:1 (modular_pipeline.py:
# 32-33, 76-80); ComfyUI's adapt_canvas has no such check and resolves a
# plausible canvas for any ratio. This node closes that gap in match_keyframe,
# where the aspect comes from the image and nobody has chosen it.
#
# The pairs below straddle the boundary on purpose. If the in-range cases ever
# start raising, the guard is too tight and will reject ordinary work; if the
# out-of-range ones stop raising, the guard is gone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3_rules import MAX_ASPECT_RATIO, MIN_ASPECT_RATIO  # noqa: E402
from keyframe_canvas import MiniMaxH3KeyframeCanvas  # noqa: E402

ASPECT_CASES = [
    ((1024, 1024), True), ((2560, 1080), True),     # 1.0, 2.37 -- ordinary
    ((1600, 400), True), ((400, 1600), True),       # exactly 4.0 and 0.25
    ((1640, 400), False), ((400, 1640), False),     # 4.1 and 0.244 -- outside
    ((3000, 500), False),                           # 6.0 -- clearly outside
]
for (w, h), want_ok in ASPECT_CASES:
    img = torch.rand(1, h, w, 3)
    try:
        MiniMaxH3KeyframeCanvas.execute(img, mode="match_keyframe")
        raised = False
    except RuntimeError:
        raised = True
    check(f"{w}x{h} (aspect {w/h:.3g}): "
          f"{'accepted' if want_ok else 'refused'}",
          raised != want_ok,
          f"in range [{MIN_ASPECT_RATIO:g}, {MAX_ASPECT_RATIO:g}]")

# fit_to_canvas must NOT raise: there the user typed the geometry and owns it.
try:
    MiniMaxH3KeyframeCanvas.execute(torch.rand(1, 500, 3000, 3),
                                    mode="fit_to_canvas", width=3008, height=512)
    check("fit_to_canvas warns rather than refusing an out-of-range aspect", True)
except RuntimeError as exc:
    check("fit_to_canvas warns rather than refusing an out-of-range aspect",
          False, str(exc))

print("\n--- 6. the duration window is enforced, checked AFTER the grid snap ---")
from h3_rules import (duration_in_range, duration_of,  # noqa: E402
                      max_legal_length, min_legal_length, snap_length)

# The order is the whole point. 346 passes any check written against the
# request and then rounds to 362 = 15.083s, which is over the ceiling -- the
# reference names this exact case in a comment. A check that snapped after
# validating would pass every one of these and still render an illegal clip.
LENGTH_CASES = [
    (124, True), (260, True), (328, True), (345, True),   # inside
    (346, False),                                          # snaps to 362
    (362, False), (400, False),                            # over
    (20, False), (100, False),                             # under 5s
]
for n, want_ok in LENGTH_CASES:
    img = torch.rand(1, 768, 1024, 3)
    try:
        MiniMaxH3KeyframeCanvas.execute(img, mode="fit_to_canvas",
                                        width=1024, height=768, length=n)
        raised = False
    except RuntimeError:
        raised = True
    check(f"length {n} -> {snap_length(n)} ({duration_of(snap_length(n)):.3f}s): "
          f"{'accepted' if want_ok else 'refused'}", raised != want_ok)

check(f"largest legal count is {max_legal_length()}",
      max_legal_length() == 345 and duration_in_range(345)
      and not duration_in_range(snap_length(346)),
      f"345 in range, 346 snaps to {snap_length(346)} which is not")
check(f"smallest legal count is {min_legal_length()}",
      min_legal_length() == 124, "matches the node default and the trained floor")

# length=0 opts out entirely; the node must not invent a constraint.
try:
    out = MiniMaxH3KeyframeCanvas.execute(torch.rand(1, 768, 1024, 3),
                                          mode="fit_to_canvas",
                                          width=1024, height=768, length=0)
    check("length 0 skips the check and passes through", out[5] == 0)
except RuntimeError as exc:
    check("length 0 skips the check and passes through", False, str(exc))

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
