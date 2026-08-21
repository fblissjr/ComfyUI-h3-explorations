#!/usr/bin/env python3
"""A mono reference waveform does not degrade H3 conditioning. It raises.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
model, no server -- it imports ComfyUI's real `pack_audio` and reproduces the
assignment `PackedLayout` makes.

The evidence behind `docs/h3_references.md`'s mono entry under Known
limitations, which carried this as read-but-not-verified until 2026-08-21.

The chain, all read at ComfyUI `76135e557d`:

  `comfy/ldm/minimax/audio_vae.py:427`   encode preserves the input channel
                                         count as `s`; there is no upmix
  `comfy_extras/nodes_minimax_h3.py:71`  `_encode_ref_audio` does not add one,
                                         and returns `z.shape[-1]` as
                                         `ref_audio_t`
  `comfy/ldm/minimax/model.py:381-386`   the layout allocates `ref_audio_t * 2`
                                         rows for the block, stereo assumed
  `comfy/ldm/minimax/model.py:659`       the masked assignment that fails

**Imports `pack_audio` rather than restating it.** A reimplementation here
would be a baseline sharing nothing with the thing it measures, and it would
keep passing after the real function changed.

**The stereo arm is the control.** Without it this script proves only that some
assignment somewhere fails; with it, the failure is attributable to the channel
count and nothing else. If the mono arm ever succeeds, the doc entry is stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ComfyUI"))

import torch  # noqa: E402

from comfy.ldm.minimax.model import pack_audio  # noqa: E402

# Any latent length works; 40 is a second of reference audio at the VAE's 40
# latent frames per second, which keeps the printed row counts readable.
T = 40


def main() -> int:
    for ch in (1, 2):
        rows = pack_audio(torch.zeros(1, 32, ch, T))
        print(f"latent [1,32,{ch},{T}]  ->  pack_audio {tuple(rows.shape)}")

    # What PackedLayout does with the block: `ref_audio_t * 2` slots, none of
    # them "update" rows, filled from the packed condition latents.
    slots = T * 2
    update = torch.zeros(slots, dtype=torch.bool)
    dest = torch.empty(slots, 32)

    print(f"\nlayout allocates {slots} rows for ref_audio_t={T}")
    ok = True
    for ch, label in ((1, "mono"), (2, "stereo")):
        try:
            dest[~update] = pack_audio(torch.zeros(1, 32, ch, T))
            print(f"  {label:<7} assignment ok")
            if ch == 1:
                ok = False
                print("  -> the mono arm SUCCEEDED. docs/h3_references.md "
                      "says it raises; one of the two is now wrong.")
        except RuntimeError as exc:
            print(f"  {label:<7} RuntimeError: {exc}")
            if ch == 2:
                ok = False
                print("  -> the stereo control FAILED, so this script is not "
                      "measuring the channel count.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
