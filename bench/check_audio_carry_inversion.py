#!/usr/bin/env python3
"""The audio carry probe's inversion is exact, and its ablation actually moves.

`MiniMaxH3AudioCarryProbe` recovers the model's raw audio velocity from an
output the model has already transformed, then re-applies the transform at a
different sigma. Every number that node's `block_mean` arm produces rests on
that inversion being right, and nothing else in the repo asserts it.

## What is transcribed and what is imported

The stub forward here writes the transform the way
`comfy/ldm/minimax/model.py` writes it, in `MiniMaxH3Model.forward`:

    out[1] = (1 - scale) * (audio_src * carry) + (1 + (scale - 1) * sigma_a) * out[1]

**That line is a transcription, and it is this check's one assumption.** It is
stated rather than hidden because a transcription can go stale: if upstream
changes the expression, this check keeps passing while the node silently
inverts something ComfyUI no longer does. `carry` and `sigma_a` are NOT
transcribed -- they come from comfy's own `time_shift_sigma`, which is the part
most likely to move and the part an import can track.

## Why the second assertion is here

`CLAUDE.md`: a check whose input already satisfies the expected outcome cannot
fail. An inversion that returned its input unchanged would pass the round-trip
assertion perfectly and make `block_mean` a no-op arm measuring nothing. So the
ablation is required to MOVE the output materially, and the zero-width case is
required to collapse back onto `block_start` -- together those say the knob is
live and correctly parameterised, which the round trip alone does not.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent.parent))          # ComfyUI root, for comfy.*

from comfy.ldm.minimax.model import time_shift_sigma  # noqa: E402

import audio_carry_probe as acp  # noqa: E402

SCALE, SHIFT_V, SHIFT_A = 4.0, 12.0, 3.0
#: A real 4-evaluation PDD schedule: knots 0, 8, 16, 24, 32 of the 32-point
#: grid at shift 12. Blocks are 8 wide, which is where the frozen coefficient
#: costs most among the partitions that actually ship.
SIGMAS = torch.tensor([1.0, 0.923077, 0.8, 0.631579, 0.0], dtype=torch.float32)


def make_stub(velocity):
    """The transform as `comfy/ldm/minimax/model.py` applies it. Transcribed."""
    def stub(x, timestep, _context, _opts, minimax_payload=None, **_kw):
        scale = float((minimax_payload or {}).get("audio_scale", 1.0))
        audio_src = x[1]
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sigma_a = time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A)
        carry = (sigma_a / sigma_v).to(audio_src.dtype)
        return [x[0].clone(),
                (1.0 - scale) * (audio_src * carry)
                + (1.0 + (scale - 1.0) * sigma_a).to(velocity.dtype) * velocity]
    return stub


def run(mode, sigma_v, sigmas=SIGMAS):
    torch.manual_seed(0)
    x = [torch.randn(1, 4, 2, 8, 8), torch.randn(1, 8, 32)]
    velocity = torch.randn_like(x[1])
    state = {"warned": False, "seen": {}, "shift_v": SHIFT_V, "shift_a": SHIFT_A}
    fwd = acp._make_probe_forward(make_stub(velocity), mode, state)
    opts = {"sample_sigmas": sigmas,
            "minimax_h3_sigma_shift_video": SHIFT_V,
            "minimax_h3_sigma_shift_audio": SHIFT_A}
    t = torch.tensor([sigma_v * 1000.0])
    return fwd(x, t, None, opts, minimax_payload={"audio_scale": SCALE})[1]


def main() -> int:
    errs = []
    knot = 0.923077                       # a block start: [0.923077 -> 0.8]

    base = run("off", knot)
    start = run("block_start", knot)
    err = float((start - base).abs().max())
    print(f"round trip  |block_start - off|max = {err:.3e}")
    if err > 1e-5:
        errs.append(f"block_start does not recover the model's output "
                    f"({err:.3e} > 1e-5). The inversion is wrong and every "
                    f"block_mean number would be meaningless.")

    mean = run("block_mean", knot)
    moved = float((mean - base).norm() / base.norm())
    print(f"ablation    |block_mean - off|rel  = {moved:.4f}")
    if moved < 1e-3:
        errs.append(f"block_mean barely moves the output ({moved:.2e}). The "
                    f"ablation arm would measure nothing, and the round-trip "
                    f"assertion above would still pass -- which is exactly the "
                    f"check-cannot-fail shape CLAUDE.md warns about.")

    # A block of zero width has no interior, so averaging over it must give the
    # block start back. This is what says the averaging is parameterised by the
    # block rather than by something incidental.
    degenerate = torch.tensor([1.0, knot, knot, 0.0], dtype=torch.float32)
    z = run("block_mean", knot, degenerate)
    s2 = run("block_start", knot, degenerate)
    gap = float((z - s2).abs().max())
    print(f"zero width  |block_mean - block_start|max = {gap:.3e}")
    if gap > 1e-5:
        errs.append(f"at zero block width block_mean ({gap:.3e}) does not "
                    f"collapse onto block_start, so the average is not taken "
                    f"over the block the sampler steps.")

    # Off-grid sigma: the node must go inert rather than invent a block width.
    if acp._block_end(SIGMAS, 0.5) is not None:
        errs.append("_block_end invented a block for a sigma that is not a "
                    "knot of the schedule; the node would silently ablate a "
                    "width the sampler never steps.")
    if acp._block_end(SIGMAS, 0.0) is not None:
        errs.append("_block_end returned a block for the FINAL sigma, which "
                    "begins no step.")
    print("off-grid and final sigma both return None: OK")

    if errs:
        for e in errs:
            print(f"\nFAIL  {e}")
        return 1
    print("\nPASS  inversion is exact, the ablation is live, and the average "
          "is taken over the sampler's own block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
