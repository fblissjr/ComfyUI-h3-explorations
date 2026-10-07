"""TEST CODE, in no node list of this pack: a node that calls ComfyUI's SAM 3 detect node and says what reached the model.

The SAM 3.1 Corrections node (`sam31_corrections.py`) is accepted on a server's queue, where nothing outside the
server process can look at the model. This node is the look: it runs `SAM3_Detect` on a frame it makes itself (a ramp
with one block of exact black and one of exact white, so the range at the first layer is exact whatever the resize
does), with a hook on the trunk's first convolution and one on a text MLP, and returns what they saw as JSON, as a
string output and as the node's text in `/history`.

It is registered only in a scratch copy of the pack served by a scratch server
(`bench/sam31_corrections_queue_test.py` says how). `after` and `nonce` exist so a test graph can fix the order the
probes run in and make a later prompt run them again while the loader's outputs stay cached.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui

#: read: the same path `sam31_corrections.PATCH_EMBED` names, spelled here so this file imports alone.
PATCH_EMBED = "diffusion_model.detector.backbone.vision_backbone.trunk.patch_embed"


def frame() -> torch.Tensor:
    """[1, 96, 128, 3] in 0..1: a ramp with a block of exact 0 and a block of exact 1."""
    ramp = torch.linspace(0.2, 0.8, 128)[None, :, None].expand(96, 128, 3).clone()
    ramp[8:40, 8:40] = 0.0
    ramp[56:88, 88:120] = 1.0
    return ramp[None]


def look(model, clip) -> dict:
    """Run core's detect node on `frame()` and report the first layer's input range and the text activation it ran."""
    import comfy.clip_model
    import comfy.model_management as mm
    import comfy.utils
    from comfy_extras.nodes_sam3 import SAM3_Detect

    loaded_before = any(getattr(lm, "model", None) is not None and getattr(lm.model, "model", None) is model.model
                        for lm in mm.current_loaded_models)
    seen, ran = [], []
    conv = comfy.utils.get_attr(model.model, PATCH_EMBED).proj
    mlp = next(m for m in clip.cond_stage_model.modules() if isinstance(m, comfy.clip_model.CLIPMLP))
    h1 = conv.register_forward_pre_hook(lambda m, i: seen.append((float(i[0].amin()), float(i[0].amax()))))
    h2 = mlp.fc1.register_forward_hook(lambda m, i, o: ran.append(mlp.activation is F.gelu))
    try:
        with torch.inference_mode():
            cond = clip.encode_from_tokens_scheduled(clip.tokenize("person:4"))
            SAM3_Detect.execute(model, frame(), conditioning=cond, threshold=0.5, individual_masks=True)
    finally:
        h1.remove()
        h2.remove()
    return {"patcher": type(model).__name__, "on_the_card_before_the_call": bool(loaded_before),
            "first_layer": [round(min(s[0] for s in seen), 4), round(max(s[1] for s in seen), 4)] if seen else None,
            "text_ran_exact_gelu": (all(ran) if ran else None), "text_calls": len(ran),
            "object_patches": sorted(k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1] for k in model.object_patches)[:4],
            "clip_object_patches": len(clip.patcher.object_patches)}


class H3TestSAM31FirstLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3TestSAM31FirstLayer",
            display_name="TEST: what reaches SAM 3.1 (not a shipped node)",
            category="model/latent/minimax/test",
            description="Test code. Runs ComfyUI's SAM 3 detect node on a made frame and reports the range that reached "
                        "the model's first layer and the activation the text encoder ran.",
            inputs=[
                io.Model.Input("segmenter"),
                io.Clip.Input("segmenter_clip"),
                io.String.Input("label", default="", tooltip="Written into the result, to tell the probes of a graph apart."),
                io.Int.Input("nonce", default=1, min=1, max=2 ** 31 - 1, tooltip="Change it to make a later prompt run this probe again."),
                io.String.Input("after", default="", optional=True, force_input=True, tooltip="Wire another probe's result here to run after it."),
            ],
            outputs=[io.String.Output(display_name="result")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, segmenter, segmenter_clip, label="", nonce=1, after="") -> io.NodeOutput:
        got = {"label": label, "nonce": int(nonce), **look(segmenter, segmenter_clip)}
        text = json.dumps(got)
        return io.NodeOutput(text, ui=ui.PreviewText(text).as_dict())
