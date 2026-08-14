"""MiniMax H3 SageAttention node.

Drop it between the model loader and the sampler. Defaults are the ones
you want; the rest is there for when something goes wrong.
"""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io

from .assert_chain import SageChainAssert
from .keyframe_canvas import MiniMaxH3KeyframeCanvas
from .preflight import MiniMaxH3Preflight
from .provenance import MiniMaxH3ProvenanceStamp
from .reference_fit import MiniMaxH3ReferenceFit
from .resolution import MiniMaxH3Resolution

from .attention import (
    MODES,
    build_kernel,
    make_minimax_attn_forward,
    make_sage_override,
    mode_releases_qkv,
    reset_fallback_state,
)

logger = logging.getLogger(__name__)


def _is_minimax_h3(diffusion_model):
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


class MiniMaxH3SageAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SageAttention",
            display_name="MiniMax H3 SageAttention",
            category="model/attention/minimax",
            description=(
                "Runs MiniMax H3's self-attention on SageAttention's sm89 "
                "INT8/FP8 kernel instead of torch attention. Connect between "
                "the model loader and the sampler; the defaults are the "
                "intended configuration."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode", options=list(MODES), default="auto",
                    tooltip=(
                        "Which kernel to run. 'auto' lets SageAttention pick "
                        "and is right on every supported card -- on a 4090 it "
                        "resolves to fp8++, so picking that explicitly changes "
                        "nothing. The explicit entries are for bisecting a "
                        "suspected accuracy problem. 'fp16' is the most "
                        "accurate and the slowest, and is the one mode that "
                        "gives up the per-call memory saving, because there is "
                        "no consuming entry point for that kernel."
                    ),
                ),
                io.Boolean.Input(
                    "patch_token_refiner", default=False, optional=True,
                    tooltip=(
                        "Also patch the 2 text token-refiner blocks. They run "
                        "over the text span only (~2k tokens vs ~40k for the "
                        "DiT blocks), so this is worth well under 1% of "
                        "attention time. Off by default."
                    ),
                ),
                # APPENDED, not inserted after `mode` where it reads better.
                # Widget values map positionally in every saved graph, so a
                # new widget ahead of `patch_token_refiner` would land an old
                # graph's boolean on this INT: ["auto", False] would silently
                # become head_chunks=False. Same rule as the output slots on
                # MiniMaxH3KeyframeCanvas.
                io.Int.Input(
                    "head_chunks", default=1, min=1, max=56, optional=True,
                    tooltip=(
                        "Run the heads in this many groups, shrinking the "
                        "kernel's internal transients by roughly the group "
                        "count at the cost of that many launches per call. "
                        "1 means this node does not chunk, and is the "
                        "measured default: on a 24 GB 4090 the headroom "
                        "chunking buys converts to wall-clock at a ~2.6% "
                        "ceiling, so it is for fitting a render that "
                        "otherwise will not fit, not for speed. "
                        "1 is not the lowest-peak setting, though. Measured "
                        "at the default canvas and 124 frames: 4 groups peaks "
                        "at 2645 MiB against 2862 at 1. Chunking wins on peak "
                        "by ~217 MiB because it rules out the v clone, which "
                        "only pays on the unchunked path. "
                        "1 also does not mean 'off' when KJNodes' MiniMax H3 "
                        "Low VRAM Attention is in the graph: that node "
                        "publishes a group count, and 1 here means this node "
                        "defers to it. There is no value that overrides it. "
                        "The log line at patch time says which count was used."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode="auto", head_chunks=1,
                patch_token_refiner=False) -> io.NodeOutput:
        diffusion_model = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(diffusion_model):
            raise RuntimeError(
                f"This node only patches MiniMax H3; got "
                f"{type(diffusion_model).__name__}. Remove it from the graph "
                f"or feed it an H3 model."
            )

        kernel_fn, kernel_kwargs = build_kernel(mode)
        forward = make_minimax_attn_forward(kernel_fn, kernel_kwargs,
                                            head_chunks=head_chunks,
                                            clone_v=mode_releases_qkv(mode))
        reset_fallback_state()

        m = model.clone()
        blocks = list(diffusion_model.blocks)
        targets = [(f"diffusion_model.blocks.{i}", b.attn) for i, b in enumerate(blocks)]
        if patch_token_refiner:
            targets += [
                (f"diffusion_model.token_refiner.blocks.{i}", b.attn)
                for i, b in enumerate(diffusion_model.token_refiner.blocks)
            ]

        for path, attn in targets:
            m.add_object_patch(
                f"{path}.attn.forward", forward.__get__(attn, attn.__class__)
            )

        # Also register an optimized_attention_override. The forward patch
        # above handles every call on its own, so this never fires when our
        # node runs alone. It matters when another patch (Sol-Attn) runs
        # ComfyUI's stock forward to reach its own override: anything that
        # override declines would otherwise land on ComfyUI's default
        # attention instead of sage. Chained onto whatever was already
        # there, and left in place for a later patch to chain onto in turn.
        # Copy before mutating. The reason is not the one this comment gave
        # until 2026-08-11: `clone()` does NOT leave transformer_options
        # shared. It runs model_options through `comfy.utils.deepcopy_list_dict`
        # (`comfy/model_patcher.py`, on every branch), which recurses into
        # dicts and lists and passes callables through by reference, so the
        # clone already owns its own dict. The copy is redundant today.
        #
        # Kept anyway, because what it guards against is severe and silent:
        # writing into a dict the source model still holds would install sage
        # on a model the user did not patch, which in an A/B contaminates the
        # control arm and looks like a result. One dict copy per patch is
        # nothing next to that, and it means this node does not depend on an
        # upstream guarantee it cannot enforce. `check_clone_v_wiring.py`
        # pins it against a deliberately shallow-cloning fake, since against
        # the real ModelPatcher the assertion cannot fail.
        to = m.model_options["transformer_options"] = \
            m.model_options.get("transformer_options", {}).copy()
        to["optimized_attention_override"] = make_sage_override(
            kernel_fn, kernel_kwargs,
            previous=to.get("optimized_attention_override"),
        )

        logger.info(
            "[h3] MiniMax H3 self-attention on sage (mode=%s, head_chunks=%s, "
            "%d attention modules patched, sage registered as the "
            "attention-override fallback)",
            mode,
            head_chunks if head_chunks > 1 else "1 (off; KJNodes' value used "
                                               "if its node publishes one)",
            len(targets),
        )
        return io.NodeOutput(m)


class H3ExplorationsExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SageAttention, SageChainAssert, MiniMaxH3KeyframeCanvas,
                MiniMaxH3ReferenceFit, MiniMaxH3Resolution, MiniMaxH3Preflight,
                MiniMaxH3ProvenanceStamp]


async def comfy_entrypoint() -> H3ExplorationsExtension:
    return H3ExplorationsExtension()
