"""Freeze a sampled H3 latent's video and reopen its audio, for an audio-only refinement pass.

The design is Adudeguyman's ComfyUI-H3-AudioRefine (MIT,
`coderef/ComfyUI-H3-AudioRefine`, `H3AudioRefineMask` / `H3AudioRefineSampler`):
take a distilled pass's output, keep the video exactly, and run a few more
steps on the audio with the model from BEFORE the distill LoRA, so those steps
run undistilled. Core already has everything the pass needs. A nested
`noise_mask` (video, audio) goes to core's H3 inpaint path
(`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint` and
`_denoise_mask_conds`). Frozen video rows are injected at the visual cond
timestep each step, and the sampler's final masked blend returns them
bit-identical. So this node only writes the mask, and the pass itself is
core's `SamplerCustomAdvanced` with a partial-denoise schedule.

Why it is worth a pass here: PDD's audio loses energy at coarse partitions
(`docs/research/pdd/audio_under_pdd.md`, "What actually holds"), and that loss
is PDD's own, not a ComfyUI artefact
(`docs/research/pdd/2026-09-25_upstream_pdd_comparison.md`).

Both masks are real values in [0, 1], and neither selects a mode by being 0:
0 preserves the stream, 1 regenerates it.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

import comfy.nested_tensor


class MiniMaxH3AudioRefineMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineMask",
            display_name="MiniMax H3 Audio Refine Mask",
            category="MiniMax H3/audio",
            description=(
                "Writes a per-stream noise_mask onto a sampled H3 AV latent: video "
                "kept, audio reopened. Feed the result to a partial-denoise "
                "SamplerCustomAdvanced whose model comes from before the distill "
                "LoRA, so the extra audio steps run undistilled. Design from "
                "ComfyUI-H3-AudioRefine (MIT)."),
            inputs=[
                io.Latent.Input("latent", tooltip="A sampled MiniMax H3 AV latent (nested video, audio)."),
                io.Float.Input("video_mask", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="How far the video is reopened: 0 keeps it exactly, 1 regenerates it."),
                io.Float.Input("audio_mask", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="How far the audio is reopened: 1 regenerates it within the pass's denoise, 0 keeps it."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, video_mask=0.0, audio_mask=1.0) -> io.NodeOutput:
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError(
                "MiniMaxH3AudioRefineMask needs a MiniMax H3 AV latent (a nested "
                "video, audio tensor); this one is a single tensor.")
        video, audio = samples.unbind()
        out = dict(latent)
        out["samples"] = samples
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((
            torch.full((1, 1) + tuple(video.shape[2:]), float(video_mask), dtype=torch.float32),
            torch.full((1, 1) + tuple(audio.shape[2:]), float(audio_mask), dtype=torch.float32),
        ))
        return io.NodeOutput(out)
