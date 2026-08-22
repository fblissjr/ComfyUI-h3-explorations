#!/usr/bin/env python3
"""Does `core_video_size` still predict what core actually does to a reference video?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no model weights: it drives core's node far enough to observe the
resize and stops.

**Why this check exists at all.** `reference_video_fit.py` reports the
resolution a reference video will be conditioned at, and it computes that with
a COPY of core's rule, because core derives it inline inside `execute` and
exposes no function to call. This repo forbids a second copy of anything
without a reason. The reason here is that the copy is a *prediction*, and the
useful failure is precisely that it stops matching -- a node that confidently
reports the wrong resolution is worse than one that reports nothing.

**It compares against core, not against numbers this file computed.** The
expected value is whatever `_resize` is actually called with when core's own
`MiniMaxH3ReferenceToVideo.execute` runs. `_resize` is patched to record its
arguments and the VAE is patched to abort immediately afterwards, so the node
is driven exactly as far as the sizing decision and no further. Nothing is
encoded and no weights are loaded.

**What would make this fail:** core changing `adapt_canvas`, changing the
no-upscale override, or moving the round-to-32. Any of those and the node's
report silently diverges from reality, which is the whole thing being guarded.

Exit 0 every prediction matches, 1 a prediction is wrong, 2 the harness could
not reach core's sizing at all -- distinguished because "could not look" and
"looked and disagreed" must not read the same.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# ComfyUI first, and the repo dir NOT on the path: `nodes_minimax_h3` does a
# bare `import nodes` and this repo's own nodes.py wins from position 0. The
# pack is reached as a package instead, the same way the other bench files do.
_COMFY = Path.home() / "ComfyUI"
sys.path.insert(0, str(_COMFY))
sys.path.insert(0, str(_REPO.parent))
_PKG = _REPO.name

# Sources chosen to straddle every branch of core's rule: below the canvas
# area (source size wins), above it (canvas wins), non-multiples of 32 (the
# rounding fires), portrait, square, and a wide one near the aspect gate.
SOURCES = [
    (960, 544), (640, 360), (1920, 1080), (1280, 720), (3840, 2160),
    (720, 1280), (768, 768), (1000, 562), (543, 961), (2048, 640),
]

FRAMES = 22  # 17n+5 lands on 22; enough to be a legal reference video


class _Reached(Exception):
    """Raised once the sizing decision has been observed."""


def main() -> int:
    try:
        import torch
        import comfy_extras.nodes_minimax_h3 as core
        import importlib
        rvf = importlib.import_module(f"{_PKG}.reference_video_fit")
    except Exception as exc:
        print(f"could not import core or the pack: {exc}")
        print("nothing was checked")
        return 2

    recorded: list[tuple[int, int]] = []
    real_resize = core._resize

    def spy_resize(_img, w, h, _mode):
        recorded.append((int(w), int(h)))
        raise _Reached

    class _StubVae:
        def encode(self, *_a, **_k):
            raise AssertionError("the VAE should never be reached")

    rows = []
    ok = True
    for (vw, vh) in SOURCES:
        predicted = rvf.core_video_size(vw, vh)
        recorded.clear()
        core._resize = spy_resize
        try:
            core.MiniMaxH3ReferenceToVideo.execute(
                clip=None, vae=_StubVae(), audio_vae=None,
                prompt="a reference video", width=1344, height=768, length=124,
                ref_videos={"ref_video_0": torch.zeros(FRAMES, vh, vw, 3)},
            )
        except _Reached:
            pass
        except Exception as exc:
            core._resize = real_resize
            print(f"could not drive core's node to its resize: "
                  f"{type(exc).__name__}: {exc}")
            print("nothing was checked")
            return 2
        finally:
            core._resize = real_resize

        if not recorded:
            print("core's node ran without resizing a reference video, so "
                  "this harness observed nothing")
            return 2
        actual = recorded[0]
        agree = actual == predicted
        ok &= agree
        rows.append((vw, vh, predicted, actual, agree))

    print(f"{'source':>12} {'predicted':>12} {'core does':>12}   verdict")
    for vw, vh, pred, act, agree in rows:
        print(f"{vw}x{vh:<6} {f'{pred[0]}x{pred[1]}':>12} "
              f"{f'{act[0]}x{act[1]}':>12}   {'ok' if agree else 'MISMATCH'}")

    if not ok:
        print("\nFAIL: reference_video_fit.core_video_size no longer predicts "
              "core. The node's reported resolution is wrong, which is worse "
              "than reporting nothing. Re-derive it from "
              "comfy_extras/nodes_minimax_h3.py rather than adjusting the "
              "expectations here.")
        return 1
    print(f"\nok    {len(rows)} source size(s) predicted correctly, against "
          f"core's own resize call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
