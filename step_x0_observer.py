"""Save each sampling step's x0 prediction to disk, and change nothing.

Written 2026-09-26 for the PDD schedule tests in `docs/distills/which_distill.md`
("Tests that would move this section", vaedude's note): to find the step, and
so the fused block, where a moving person first appears twice, by setting each
step's prediction against the Euler-32 base's at the same sigma.

Core calls every `model_options["sampler_post_cfg_function"]` with that step's
`denoised` (`comfy/samplers.py::cfg_function`, reached through
`sampling_function` by `BasicGuider` as by any CFG guider). This node appends
one through `ModelPatcher.set_model_sampler_post_cfg_function`, which keeps
any function already there, and returns `denoised` unchanged.

Per call it writes the video half (or both halves) of the packed prediction,
unpacked with the model's own `latent_shapes`, as `SaveLatent`-format files:
`<output>/<filename_prefix>_<render stamp>_<step>_video.latent`, with the
step's sigma in the file's metadata. The render stamp is taken when a call's
sigma is not below the previous one, which is how a new sampling run starts,
so two renders under one prefix never overwrite each other. A prediction is
about 40 MB at 1344x768 x 345, so arm this on one or two renders, not a batch.

The prediction is in the model's latent space (`process_latent_in` applied).
For H3 that is the identity on video. On audio it carries the model's audio
scale (`comfy/model_base.py::MiniMaxH3.process_latent_in`).

Nothing here patches core; the hook is the model patcher's own.
"""

from __future__ import annotations

import logging
import os
import time

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)
TAG = "[h3 step x0]"


def _streams(denoised, model):
    shapes = getattr(model, "latent_shapes", None)
    if shapes is not None and len(shapes) > 1:
        import comfy.utils
        return comfy.utils.unpack_latents(denoised, shapes)
    return [denoised]


def make_observer(out_dir: str, prefix: str, with_audio: bool):
    """The post-CFG function. Pure apart from the files it writes."""
    state = {"stamp": None, "last_sigma": None, "step": 0, "renders": 0}

    def observe(args):
        denoised = args["denoised"]
        try:
            sigma = float(torch.as_tensor(args["sigma"]).flatten()[0])
            if state["last_sigma"] is None or sigma >= state["last_sigma"]:
                state["renders"] += 1
                state["stamp"] = f"{time.strftime('%Y%m%d_%H%M%S')}r{state['renders']}"
                state["step"] = 0
            state["last_sigma"] = sigma
            streams = _streams(denoised, args.get("model"))
            names = ["video", "audio"][:len(streams)] if with_audio else ["video"]
            import comfy.utils
            os.makedirs(out_dir, exist_ok=True)
            for name, t in zip(names, streams):
                path = os.path.join(out_dir, f"{prefix}_{state['stamp']}_{state['step']:02d}_{name}.latent")
                comfy.utils.save_torch_file(
                    {"latent_tensor": t.detach().float().cpu().contiguous(),
                     "latent_format_version_0": torch.tensor([])},
                    path, metadata={"sigma": f"{sigma:.9g}", "step": str(state["step"])})
            logger.info(f"{TAG} step {state['step']} sigma={sigma:.6g} -> {prefix}_{state['stamp']}_{state['step']:02d}")
            state["step"] += 1
        except Exception as exc:
            logger.warning(f"{TAG} could not save this step: {exc}")
        return denoised
    return observe


class MiniMaxH3StepX0Observer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3StepX0Observer",
            display_name="MiniMax H3 Step x0 Observer",
            category="MiniMax H3/debug",
            description=(
                "Saves each sampling step's x0 prediction as a latent file (about 40 MB "
                "a step at 1344x768 x 345), and returns it unchanged. A diagnostic: it "
                "changes no output."),
            inputs=[
                io.Model.Input("model"),
                io.String.Input("filename_prefix", default="latents/h3_step_x0",
                                tooltip="Under the output directory; a subfolder is allowed."),
                io.Boolean.Input("save_audio", default=False,
                                 tooltip="Also save the audio half of each prediction."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, filename_prefix="latents/h3_step_x0", save_audio=False) -> io.NodeOutput:
        import folder_paths
        full = os.path.join(folder_paths.get_output_directory(), filename_prefix)
        out_dir, prefix = os.path.dirname(full), os.path.basename(full)
        m = model.clone()
        m.set_model_sampler_post_cfg_function(make_observer(out_dir, prefix, bool(save_audio)))
        return io.NodeOutput(m)
