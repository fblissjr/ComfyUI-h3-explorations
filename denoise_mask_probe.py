"""Log the denoise mask each masked sampling step actually uses, and change nothing.

Written 2026-09-26 for the frozen-row question in
`docs/wiki/archive/2026-09-26_distill_routing.md`: the audio-refine pass freezes the
video at mask 0, yet its video decoded about 46 dB from the base arm's
(`bench/results/2026-09-25_distill_audio_s1.md`). One of the checks is
whether the mask reaches the sampler's inpaint step as exactly 0.

Core calls `model_options["denoise_mask_function"](sigma, denoise_mask,
extra_options)` at the top of every masked step (`comfy/samplers.py`,
`KSamplerX0Inpaint.__call__`) and uses what it returns. This node installs one
through `ModelPatcher.set_model_denoise_mask_function`, chains to any function
already there, and returns the mask it was given, unchanged. Per call it logs,
for each stream the packed mask unpacks into (`comfy.utils.unpack_latents` on
the model's `latent_shapes`): min, max and the fraction of values exactly 0
and exactly 1. The line starts `[h3 mask probe]`, which
`pipeline_telemetry.py` records as a `mask_probe` event on an armed server.

Nothing here patches core; the hook is the model patcher's own.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)
TAG = "[h3 mask probe]"


def _stats(t: torch.Tensor) -> str:
    t = t.detach().float()
    n = max(t.numel(), 1)
    return (f"min={float(t.min()):.6g} max={float(t.max()):.6g} "
            f"zero={float((t == 0).sum()) / n:.4f} one={float((t == 1).sum()) / n:.4f}")


def describe(sigma, mask, shapes) -> str:
    """One log line for one call. Pure, so the check can drive it."""
    parts = []
    streams = None
    if shapes is not None and len(shapes) > 1:
        try:
            import comfy.utils
            streams = comfy.utils.unpack_latents(mask, shapes)
        except Exception:
            streams = None
    if streams is None:
        parts.append(f"all {_stats(mask)}")
    else:
        for name, s in zip(("video", "audio", "s2", "s3"), streams):
            parts.append(f"{name} {_stats(s)}")
    sig = float(torch.as_tensor(sigma).flatten()[0])
    return f"{TAG} sigma={sig:.6g} " + " | ".join(parts)


def make_probe(previous=None):
    def probe(sigma, denoise_mask, extra_options):
        if previous is not None:
            denoise_mask = previous(sigma, denoise_mask, extra_options)
        model = (extra_options or {}).get("model")
        inner = getattr(model, "inner_model", None)
        shapes = getattr(inner, "latent_shapes", None)
        try:
            logger.info(describe(sigma, denoise_mask, shapes))
        except Exception as exc:
            logger.info(f"{TAG} could not describe the mask: {exc}")
        return denoise_mask
    return probe


class MiniMaxH3DenoiseMaskProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DenoiseMaskProbe",
            display_name="MiniMax H3 Denoise Mask Probe",
            category="MiniMax H3/debug",
            description=(
                "Logs the denoise mask each masked sampling step uses, per stream "
                "(min, max, fraction exactly 0 and 1), and returns it unchanged. "
                "A diagnostic: it changes no output."),
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        previous = m.model_options.get("denoise_mask_function")
        m.set_model_denoise_mask_function(make_probe(previous))
        return io.NodeOutput(m)
