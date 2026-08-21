#!/usr/bin/env python3
"""Where does ComfyUI's Qwen pixel ceiling actually shrink an H3 input?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only; it
allocates one blank image per row and frees it.

`docs/h3_references.md` records that ComfyUI leaves the shared Qwen2-VL
helper's `min_pixels`/`max_pixels` on their signature defaults while the
release declares its own. This measures the consequence rather than deriving
it: it calls the real `process_qwen2vl_images` with the arguments
`comfy/text_encoders/qwen3vl.py:65` passes, and reports the grid that comes
back. A grid smaller than the input means the helper resized, which is the
whole question.

**Two arms, and the second is the control.** Reference images are prepared to a
2048 short edge and are the only thing in the pipeline near either bound.
Keyframes are prepared to a legal H3 canvas. Running both is what turns "the
bounds differ" into "the bounds differ *here* and are inert *there*" -- and the
keyframe arm is the one that should come back untouched at every row. If it
ever does not, the claim in `official_weights_metadata.md` that the pixel
bounds are inert on the keyframe modes is wrong and this script is how you
would find out.

The release bounds are read from `vendor_config`, never retyped, so a change on
their side shows up here rather than being silently reproduced from memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path.home() / "ComfyUI"))

# The short edge `MiniMaxH3ReferenceFit` and core's `max` mode both target, and
# the ratios the reference resize accepts, inclusive.
REF_SHORT_EDGE = 2048
REF_RATIOS = (1.0, 2.0, 3.0, 3.0625, 3.25, 3.5, 4.0)

# Every canvas is a multiple of 32 and this is the control arm: a keyframe is
# always prepared to one of these, so the helper should never resize it.
CANVASES = ((1344, 768), (768, 768), (1024, 768), (768, 1344), (1536, 672))


def _release_bounds():
    """(min_pixels, max_pixels) for the image branch, from the vendored config."""
    try:
        import vendor_config
        return vendor_config.image_pixel_bounds()
    except Exception:
        # No vendored release config on this box. The measurement of what
        # ComfyUI does still stands; only the comparison column is lost, and
        # printing that is better than substituting a remembered number.
        return None, None


def _grid(w: int, h: int):
    """(grid_h, grid_w) the Qwen helper returns for a w x h input."""
    import torch
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    img = torch.zeros(1, h, w, 3)
    try:
        _, grid = process_qwen2vl_images(img, patch_size=16,
                                         image_mean=[0.5] * 3, image_std=[0.5] * 3)
    finally:
        del img
    return int(grid[0][1]), int(grid[0][2])


def _run(label: str, sizes, rel_min, rel_max) -> list[dict]:
    rows = []
    print(f"\n=== {label}")
    print(f"  {'input':<14}{'qwen grid':<12}{'qwen sees':<14}"
          f"{'resized':<9}{'release':<10}")
    for w, h in sizes:
        gh, gw = _grid(w, h)
        out_w, out_h = gw * 16, gh * 16
        resized = (out_w, out_h) != (w, h)
        px = w * h
        rel = ("shrinks" if rel_max and px > rel_max else
               "enlarges" if rel_min and px < rel_min else "untouched")
        rows.append({"input": [w, h], "grid": [gh, gw],
                     "qwen_sees": [out_w, out_h], "resized": resized,
                     "release": rel})
        print(f"  {f'{w}x{h}':<14}{f'{gh}x{gw}':<12}{f'{out_w}x{out_h}':<14}"
              f"{('SHRUNK' if resized else 'no'):<9}{rel:<10}")
    return rows


def main() -> int:
    try:
        import torch  # noqa: F401
        import comfy.text_encoders.qwen_vl  # noqa: F401
    except ImportError as exc:
        print(f"ComfyUI is not importable from here ({exc}); run this with the "
              f"ComfyUI venv python (see docs/comfy_notes.md)")
        return 2

    rel_min, rel_max = _release_bounds()
    print(f"release image bounds: min {rel_min}  max {rel_max}"
          + ("" if rel_max else "   (vendored config unreadable; "
                                "the release column is blank)"))

    refs = [(int(round(REF_SHORT_EDGE * r / 32) * 32), REF_SHORT_EDGE)
            for r in REF_RATIOS]
    record = {
        "release_image_bounds": {"min_pixels": rel_min, "max_pixels": rel_max},
        "reference_images_at_2048_short_edge": _run(
            "reference images, prepared to a 2048 short edge",
            refs, rel_min, rel_max),
        "keyframes_at_legal_canvas": _run(
            "keyframes, prepared to a legal H3 canvas (the control)",
            CANVASES, rel_min, rel_max),
    }

    first_bite = next((r for r in record["reference_images_at_2048_short_edge"]
                       if r["resized"]), None)
    kf_bitten = [r for r in record["keyframes_at_legal_canvas"] if r["resized"]]
    print()
    if first_bite:
        w, h = first_bite["input"]
        print(f"  reference ceiling first bites at {w}x{h} "
              f"(ratio {w / h:.4f}), where the release leaves it "
              f"{first_bite['release']}")
    else:
        print("  the reference ceiling never bit; the doc's threshold is wrong")
    if kf_bitten:
        print(f"  CONTROL FAILED: {len(kf_bitten)} canvas(es) were resized. "
              f"The pixel bounds are NOT inert on the keyframe modes and "
              f"docs/research/official_weights_metadata.md says they are.")
    else:
        print("  control holds: no legal canvas is resized, so the bounds are "
              "inert on every keyframe mode")
    record["control_holds"] = not kf_bitten

    out = _REPO / "bench" / "results" / "2026-08-21_qwen_bounds_bite.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
