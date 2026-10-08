"""Set how clean the model is shown its picture and video references.

Core gives every visual condition latent (a keyframe, a reference still, a
reference video's own copy) a fixed, small amount of noise and labels its rows
as that clean: `comfy/ldm/minimax/model.py`, `_cond_video_rows`
(`aug * rows + (1 - aug) * noise`) and, in `_forward`, the `cond` and `ref_img`
timesteps (`max(t_video, aug)`). The level is `VISUAL_COND_TIMESTEP` unless the
payload carries `visual_cond_noise_aug`, which core fills from the conditioning
key `minimax_visual_cond_noise_aug` (`comfy/model_base.py`,
`MiniMaxH3.extra_conds`). No node of core's sets that key and none of this
pack's did. The release exposes the same level as a field of the request
(`imgvid_cond_noise_aug_for_inference` in coderef sglang's
`configs/sample/minimax_h3.py`), so lowering it is the release's own way to
make a visual reference weigh less without a word of the prompt changing.

This node sets the level on the model: a diffusion-model wrapper hands core a
copy of the payload with the level in it. It is on the model and not on the
conditioning so that it works whatever built the conditioning; the song node
builds its own inside itself (`audio_freeze_song.py`) and has no input for it.

One level for every visual reference alike: core has one, not one per
reference, so a still that carries an identity is weakened with the video that
carries a look. Reference audio has its own level and is not touched.

Written 2026-10-08 for the whole-frame video-to-video lane: a reference video
shown at the canvas's size carried the source's look along with its movement
(`internal/` session notes of that day). When written, nothing had been
rendered at any level but core's.

Nothing here patches core; the wrapper is the model patcher's own.
"""

from __future__ import annotations

import logging

from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

logger = logging.getLogger(__name__)

WRAPPER_KEY = "h3_reference_noise"
PAYLOAD_KEY = "visual_cond_noise_aug"      # the key core reads (`comfy/ldm/minimax/model.py`)


def _level_wrapper(level: float):
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        # a copy: the payload is the conditioning's own object, shared by every step and every model on it
        payload = dict(kwargs.get("minimax_payload") or {})
        payload[PAYLOAD_KEY] = level
        kwargs["minimax_payload"] = payload
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper


class MiniMaxH3ReferenceNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceNoise",
            display_name="MiniMax H3 Reference Noise",
            category="model/conditioning/minimax",
            description=(
                "How closely the model is shown its picture and video references. Lower the level and every "
                "reference still and reference video is mixed with more noise and labelled less clean, so the "
                "render follows them less closely; the prompt and the references themselves do not change."),
            inputs=[
                io.Model.Input("model"),
                # The default is core's own constant, inherited: with it this node changes nothing.
                io.Float.Input("clean_level", default=VISUAL_COND_TIMESTEP, min=0.0, max=1.0, step=0.001,
                               tooltip=("How clean every picture and video reference is when the model sees "
                                        "it, from 0 (all noise) to 1 (untouched).\n\n"
                                        "The default is the model's own level. Lower it when the render "
                                        "copies more of a reference than you asked for, for example the "
                                        "look of a reference video when you only want its movement.\n\n"
                                        "It applies to every still and every video reference alike, so "
                                        "a still that gives a face is weakened too. Reference audio is "
                                        "not changed.")),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, clean_level=VISUAL_COND_TIMESTEP) -> io.NodeOutput:
        level = float(clean_level)
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"clean_level {level} is outside 0 to 1")
        m = model.clone()
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _level_wrapper(level))
        logger.info("[h3] reference noise: picture and video references shown at clean level %g (core's is %g)",
                    level, VISUAL_COND_TIMESTEP)
        return io.NodeOutput(m)
