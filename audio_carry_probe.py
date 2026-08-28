"""Re-apply H3's audio change-of-variable at a different point in the block.

## What this is for

`docs/research/pdd/audio_under_pdd.md` blames PDD's audio penalty on one thing:
ComfyUI converts the audio velocity with a change of variable evaluated at the
step's OWN sigma (`comfy/ldm/minimax/model.py`, in `MiniMaxH3Model.forward`),

    out[1] = (1 - s) * (x_a * carry) + (1 + (s - 1) * sigma_a) * out[1]

while a PDD fused head returns the block's MEAN velocity over a span that can
be 28 grid points wide. Both coefficients are frozen at the block's START,
where sigma is largest, and both fall across the block.

`bench/analyze_pdd_stream_energy.py` showed that no experiment varying the
PARTITION can test that, because every partition-derived statistic -- including
one computed through this very transform -- ranks the arms identically to plain
partition coarseness. And `shift_audio` is not a clean knob either: it moves the
transform strength, the audio schedule, the head fusion plan
(`pdd_math.fuse_block` weights by `pdd_time_grid(shift_a, ...)`) and the audio
adaln grid together, taking the heads off the schedule they were distilled on.

**This node changes exactly one thing.** The transform is invertible, so the
wrapper recovers the model's raw audio velocity and re-applies the transform
with sigma taken at the block's average instead of its start. Heads, schedule,
adaln grid, seed, prompt, sampler and every weight stay fixed.

## The modes are a ladder, and the middle rung is the NOISE FLOOR

    off           the stock path, untouched. Nothing is installed.
    block_start   invert and re-apply at the SAME sigma -- a round trip.
    block_mean    the ablation: sigma at the block's average.

`block_start` is arithmetically the identity and recovers the model's output to
about 1e-6, which `bench/check_audio_carry_inversion.py` asserts. **It does not
reproduce `off` as a RENDER, and expecting it to would be the mistake.** A 1e-6
perturbation diverges a sampling trajectory completely -- `CLAUDE.md` measured
exactly that, at frame 0, under a deterministic sampler -- so `block_start` is a
different SAMPLE drawn with the same knob.

That is what makes it worth rendering. It bounds how much the audio energy moves
for a change that is numerically nothing, which is the only thing that says
whether `block_mean`'s effect is a result or a draw. Render it, and read
`block_mean` against it rather than against `off`.

## What a result would mean

If `block_mean` recovers a large part of the audio energy that
`bench/results/2026-08-28_pdd_stream_energy.json` measured as lost (4.5 dB at
`u8`, 11.2 dB at `opt4`), the transform is the mechanism. If it changes little,
the loss is generic to coarse blocks and the transform is not the story --
which is the outcome the partition evidence currently favours, since video
loses contrast in the same ordering with no such transform anywhere near it.

## The one asymmetry, stated rather than buried

`carry` is re-evaluated in the OUTPUT only. The network already ran on
`x_a * carry(sigma_start)`, and no post-hoc correction can change what it saw
without a second forward. So `block_mean` is a full correction of the velocity
coefficient `B` and a PARTIAL one of the latent coefficient `A`. If the two
terms disagree about the sign of the fix -- and at `s = 4` they pull opposite
ways -- this cannot separate them, and no single render can.
"""
import logging

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)

#: Sample count for averaging a coefficient across one block. The sampler takes
#: ONE Euler step of length `d(sigma_v)` per block, so the average that matches
#: what it does is uniform in sigma_v -- not in sigma_a, and not in the index.
BLOCK_SAMPLES = 64

MODES = ("off", "block_start", "block_mean")


def _block_end(sample_sigmas, sigma_v):
    """The sigma_v this block steps TO, from the sampler's own schedule.

    Returns None when the schedule is absent or does not contain this sigma --
    both of which mean the block extent is unknown, and the caller must then
    leave the render alone rather than guess a width.
    """
    if sample_sigmas is None:
        return None
    s = torch.as_tensor(sample_sigmas).detach().flatten().to(torch.float64)
    if s.numel() < 2:
        return None
    d = (s - float(sigma_v)).abs()
    i = int(d.argmin())
    # Not "nearest wins": a sigma that is not actually one of the knots means
    # this forward is not a step of this schedule, and the width would be
    # invented. The tolerance is loose enough for the fp32 timestep round trip
    # and far tighter than the gap between adjacent knots.
    if float(d[i]) > 1e-4 or i + 1 >= s.numel():
        return None
    return float(s[i + 1])


def _averaged(sigma_v, sigma_next, shift_v, shift_a, time_shift_sigma):
    """Mean of `sigma_a` and of `carry` across the block, uniform in sigma_v."""
    gv = torch.linspace(float(sigma_v), float(sigma_next), BLOCK_SAMPLES,
                        dtype=torch.float64).clamp(min=1e-6)
    ga = time_shift_sigma(gv, shift_v, shift_a)
    return float(ga.mean()), float((ga / gv).mean())


def _make_probe_forward(base_forward, mode, state):
    """Invert the audio transform and re-apply it at the block's average sigma.

    Chains `base_forward`, so a pack that has already patched the model's
    forward keeps working -- same reason `pdd_lora._make_capture_forward` does.
    """
    from comfy.ldm.minimax.model import time_shift_sigma

    def forward(*args, **kwargs):
        out = base_forward(*args, **kwargs)
        if mode == "off":
            return out

        def pick(name, pos, default=None):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if len(args) > pos else default

        payload = pick("minimax_payload", 4) or {}
        scale = float(payload.get("audio_scale", 1.0))
        if scale == 1.0:
            # `MiniMaxH3Model.forward` guards the whole carry block on this, so
            # there is no transform to re-apply and nothing here is an ablation
            # of it. Silent because it is a legitimate configuration, not a
            # failure -- it is simply not this experiment.
            return out

        x = pick("x", 0)
        timestep = pick("timestep", 1)
        opts = pick("transformer_options", 3) or {}
        if x is None or timestep is None:
            return out

        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        end = _block_end(opts.get("sample_sigmas"), float(sigma_v))
        if end is None:
            if not state["warned"]:
                state["warned"] = True
                logger.warning(
                    "[h3-carry] no usable `sample_sigmas` in "
                    "transformer_options, so the block extent is unknown and "
                    "this node is INERT for this render. It is not a control "
                    "arm -- it is a missing measurement.")
            return out

        shift_v = float(opts.get("minimax_h3_sigma_shift_video",
                                 state["shift_v"]))
        shift_a = float(opts.get("minimax_h3_sigma_shift_audio",
                                 state["shift_a"]))
        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
        carry = (sigma_a / sigma_v)

        if mode == "block_mean":
            sigma_a_eff, carry_eff = _averaged(
                float(sigma_v), end, shift_v, shift_a, time_shift_sigma)
        else:                                   # block_start: the round trip
            sigma_a_eff, carry_eff = float(sigma_a), float(carry)

        audio_src = x[1]
        a_term = (1.0 - scale) * (audio_src * carry.to(audio_src.dtype))
        coef = 1.0 + (scale - 1.0) * float(sigma_a)
        # Invert what the model just applied, then re-apply at the effective
        # sigma. `coef` is 1 + 3*sigma_a at s=4 and so is bounded below by 1 on
        # sigma_a in [0, 1]; it cannot vanish, and no guard is pretending to
        # cover a case that does not arise.
        v = (out[1] - a_term.to(out[1].dtype)) / coef
        out[1] = ((1.0 - scale) * (audio_src * carry_eff).to(out[1].dtype)
                  + (1.0 + (scale - 1.0) * sigma_a_eff) * v)

        key = round(float(sigma_v), 6)
        if key not in state["seen"]:
            state["seen"][key] = True
            logger.info(
                "[h3-carry] %s: sigma_v %.4f -> %.4f, sigma_a %.4f -> %.4f "
                "(mean over block), coefficient %.4f -> %.4f, %+.2f%%",
                mode, float(sigma_v), end, float(sigma_a), sigma_a_eff, coef,
                1.0 + (scale - 1.0) * sigma_a_eff,
                100.0 * ((1.0 + (scale - 1.0) * sigma_a_eff) / coef - 1.0))
        return out

    return forward


class MiniMaxH3AudioCarryProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioCarryProbe",
            display_name="MiniMax H3 Audio Carry Probe (diagnostic)",
            category="model_patches/minimax",
            description=(
                "DIAGNOSTIC, not a quality setting. Re-applies H3's audio "
                "change-of-variable at the block's average sigma instead of "
                "its start, to test whether that freeze is what costs PDD "
                "arms their audio energy. Changes exactly one quantity: "
                "heads, schedule, adaln grid, seed and every weight stay "
                "fixed. Place it AFTER the PDD LoRA node so it wraps that "
                "node's forward rather than being wrapped by it."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode", options=list(MODES), default="off",
                    tooltip=(
                        "off: the stock path, untouched. block_start: invert "
                        "and re-apply at the SAME sigma. Arithmetically the "
                        "identity, but 1e-6 of float noise diverges a "
                        "trajectory, so as a render it is a different SAMPLE "
                        "at the same knob -- which makes it the noise floor "
                        "block_mean must clear. block_mean: the ablation, "
                        "sigma averaged across the block. Render block_start "
                        "first and read block_mean against it, never against "
                        "off."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode="off"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        m = model.clone()
        if mode == "off":
            # Deliberately installs NOTHING. An installed no-op wrapper would
            # still be a code path this arm does not intend to exercise, and
            # `off` is meant to be the stock render rather than a claim that
            # the wrapper is harmless.
            logger.info("[h3-carry] mode=off: no patch installed, stock path")
            return io.NodeOutput(m)

        ms = m.get_model_object("model_sampling")
        state = {
            "warned": False, "seen": {},
            # Read from the model, so a graph with no MiniMaxH3SigmaShift is
            # compared against what it will actually run rather than against a
            # constant typed here. Same reason the PDD node keeps
            # `default_shift_v/a`: an ABSENT key is not agreement.
            "shift_v": float(getattr(ms, "shift", 12.0)),
            "shift_a": float(getattr(ms, "audio_shift", None) or 3.0),
        }
        base = m.get_model_object("diffusion_model").forward
        m.add_object_patch("diffusion_model.forward",
                           _make_probe_forward(base, mode, state))
        logger.info(
            "[h3-carry] mode=%s installed, shifts %g/%g. %s",
            mode, state["shift_v"], state["shift_a"],
            "NOISE FLOOR arm: arithmetically the identity, but a different "
            "sample. Read block_mean against this, not against mode=off."
            if mode == "block_start" else
            "ABLATION arm: read it only against a block_start render.")
        return io.NodeOutput(m)
