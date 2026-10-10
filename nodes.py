"""MiniMax H3 SageAttention node.

Drop it between the model loader and the sampler. Defaults are the ones
you want; the rest is there for when something goes wrong.
"""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io

from .conditioning import MiniMaxH3Conditioning
from .exact_blocks import MiniMaxH3ExactBlocks
# The channel-balance node is published on its own as ComfyUI-H3-Quant
# (2026-09-15). ComfyUI keeps one class per node id and the last pack loaded
# wins with no warning, so if that pack is installed beside this one, this
# pack registers THAT pack's class: one object under the id, whatever the load
# order. Otherwise the bundled copy (kept in step with the published one).
def _channel_balance_class():
    import importlib.util
    import logging
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "ComfyUI-H3-Quant" / "channel_balance.py"
    if src.is_file():
        spec = importlib.util.spec_from_file_location("comfyui_h3_channelbalance.channel_balance", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        logging.info("[h3] MiniMaxH3ChannelBalance served by ComfyUI-H3-Quant")
        return mod.MiniMaxH3ChannelBalance
    from .channel_balance import MiniMaxH3ChannelBalance
    return MiniMaxH3ChannelBalance


MiniMaxH3ChannelBalance = _channel_balance_class()
from .h3_encoder_loader import MiniMaxH3EncoderLoader
from .pdd_lora import MiniMaxH3PDDLoRA
from .audio_carry_probe import MiniMaxH3AudioCarryProbe
from .audio_freeze import (MiniMaxH3FreezeAudio, MiniMaxH3FreezeAudioWindow,
                           MiniMaxH3EncodeTrack, MiniMaxH3AudioAttentionGain)
from .audio_freeze_song import MiniMaxH3AudioFreezeSong
from .audio_refine import MiniMaxH3AudioRefineMask
from .frozen_video_cache import MiniMaxH3FrozenVideoCache
from .lora_branch import MiniMaxH3LoRABranch
from .overlay_loader import MiniMaxH3OverlayLoader
from .denoise_mask_probe import MiniMaxH3DenoiseMaskProbe
from .video_mask import MiniMaxH3MaskedSource
from .plate_restore import MiniMaxH3RestorePlate
from .subject_track import MiniMaxH3SubjectTrack
from .sapiens2_parts import MiniMaxH3Sapiens2Loader, MiniMaxH3SubjectParts
from .subject_boxes import MiniMaxH3SubjectBoxes
from .sam3d_body_vith import MiniMaxH3SAM3DBodyViTHLoader
from .shot_table import MiniMaxH3SaveShotTable
from .sam31_corrections import MiniMaxH3SAM31Corrections
from .masked_prompt import MiniMaxH3MaskedPrompt
from .step_x0_observer import MiniMaxH3StepX0Observer
from .reference_noise import MiniMaxH3ReferenceNoise
from .core_sparse_capture import MiniMaxH3CoreSparseCapture
from .preflight import MiniMaxH3Preflight
from .provenance import MiniMaxH3ProvenanceStamp
from .resolution import MiniMaxH3Resolution
from .sol_attn_h3 import MiniMaxH3Sol
from .sol_chunked_h3 import MiniMaxH3SolChunked
from .vae_precision import MiniMaxH3VAEPrecision
from .reference_conditioning import (
    MiniMaxH3AppendRefAudio,
    MiniMaxH3AppendRefImage,
    MiniMaxH3AppendRefVideo,
    MiniMaxH3ReferenceConditioning,
)
from .reference_encode import MiniMaxH3EncodeReferences, MiniMaxH3PromptOnReferences
from .prompt_lists import MiniMaxH3FillPromptLists, MiniMaxH3PromptList, register_wildcards_folder
from . import h3_capture

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
                        "no consuming entry point for that kernel. 'fp8++ "
                        "balanced' is fp8++ with the fork's per-head q/k channel "
                        "rebalancing in the INT8 quantizer: less error on the "
                        "blocks whose K channels are lopsided (H3's last ones), "
                        "same elsewhere, under 1% slower. An experiment; "
                        "docs/h3_block49_quant_error.md."
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

        for i, (path, attn) in enumerate(targets):
            m.add_object_patch(
                f"{path}.attn.forward", forward.__get__(attn, attn.__class__)
            )
            # Stamp the block index for `h3_capture`. The capture module used
            # to derive it from first-seen call order, which is correct within
            # one loaded model and silently wrong the moment a graph swaps
            # checkpoints between renders -- see the note there. `targets` is
            # already in block order, and the token-refiner blocks (appended
            # after, only when patched) continue the numbering rather than
            # colliding with block 0.
            attn._h3_block_index = i

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
        # The final-layer tap for `docs/open_experiments.md` #22. Installed
        # here rather than in its own node so that BYPASSING THIS NODE TURNS IT
        # OFF, which is only true if the tap rides on a patch. Inert unless
        # `H3_CAPTURE` carries `final=1`, which is every normal render.
        if h3_capture.wants_final():
            # Through `get_model_object`, so the tap chains onto an upstream
            # patch on this clone and never onto an earlier render's wrapper
            # still applied to the shared module.
            _original_forward = m.get_model_object("diffusion_model.forward")

            def _final_tap(*args, **kwargs):
                out = _original_forward(*args, **kwargs)
                h3_capture.maybe_capture_final(out)
                return out

            m.add_object_patch("diffusion_model.forward", _final_tap)
            logger.info("[h3] capture: final-layer tap installed (final=1)")

        to = m.model_options["transformer_options"] = \
            m.model_options.get("transformer_options", {}).copy()
        to["optimized_attention_override"] = make_sage_override(
            kernel_fn, kernel_kwargs,
            previous=to.get("optimized_attention_override"),
        )

        # **The seam that lets a capture see Sol's calls.** Sol composes with a
        # foreign attention forward by gating it: the calls Sol takes run
        # ComfyUI's STOCK forward to reach `optimized_attention`, and only the
        # ones it declines run ours. So `h3_capture`, which lives inside our
        # forward, recorded exactly the COMPLEMENT of Sol's work on every
        # shipped graph -- the token-refiner calls, the dense blocks, and the
        # steps outside Sol's sigma band -- and said nothing about it. A
        # sage capture wearing a general label.
        #
        # Sol's composition already prefers `sol_take_forward` over the stock
        # forward when one is published, describing it as "a cooperating
        # patch's forward that reaches optimized_attention while keeping its
        # own low-VRAM behavior". Nothing published it until 2026-08-30. This
        # does: same projection, same fused rope, same capture, same memory
        # handling, then `optimized_attention` instead of sage's kernel, so
        # Sol still does the attention and the capture still fires -- tagged
        # `sol` rather than `sage`.
        #
        # Inert when Sol is absent: nothing reads the key. Inert when capture
        # is off: the delegate is the same forward, and the only difference at
        # run time is which function the attention goes to.
        to["sol_take_forward"] = make_minimax_attn_forward(
            kernel_fn, kernel_kwargs, head_chunks=head_chunks,
            clone_v=mode_releases_qkv(mode), via_optimized_attention=True,
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
        # Append only. A saved graph stores widget values as a bare list matched
        # by index and wires links to integer slots, so inserting anywhere but
        # the end silently re-points every later entry in every existing graph.
        # Retired in 0.173.0 (owner, 2026-09-29), each removed from this list:
        # `MiniMaxH3KeyframeCanvas`, `MiniMaxH3ReferenceFit`,
        # `MiniMaxH3ReferenceVideoFit`, `MiniMaxH3MarkerArm`, `SageChainAssert` and
        # `MiniMaxH3ReferenceReport` (no graph used them), `MiniMaxH3QuantObserve` (open
        # experiment #23, closed by the owner) and `MiniMaxH3VSAAttention` (parked and
        # refusing; core's `BlockSparseAttention` in `vsa` mode replaces it).
        return [MiniMaxH3SageAttention,
                MiniMaxH3Resolution, MiniMaxH3Preflight,
                MiniMaxH3ProvenanceStamp,
                MiniMaxH3VAEPrecision,
                MiniMaxH3Conditioning,
                MiniMaxH3AppendRefImage, MiniMaxH3AppendRefVideo,
                MiniMaxH3AppendRefAudio, MiniMaxH3ReferenceConditioning,
                # `MiniMaxH3AWQEncoderLoader` sat here until 2026-09-13; the
                # AWQ lane is closed and the node is gone. Removal, unlike
                # insertion, moves nothing that follows.
                MiniMaxH3EncoderLoader,
                MiniMaxH3PDDLoRA, MiniMaxH3AudioCarryProbe,
                MiniMaxH3ExactBlocks,
                # MiniMaxH3SolAttn sat here until 2026-09-27; replaced by MiniMaxH3Sol
                # (appended below). Removal, unlike insertion, moves nothing that follows.
                MiniMaxH3SolChunked,
                MiniMaxH3FreezeAudio, MiniMaxH3FreezeAudioWindow,
                MiniMaxH3EncodeTrack,
                MiniMaxH3AudioAttentionGain, MiniMaxH3AudioFreezeSong,
                MiniMaxH3PromptList,
                MiniMaxH3FillPromptLists,
                # appended 2026-09-14; insertion anywhere earlier would move what follows.
                # Since 2026-09-15 the node also ships on its own as the
                # ComfyUI-H3-Quant pack; when that is installed beside this
                # one, the name above IS that pack's class (see the import), so both
                # registrations hold one object and load order decides nothing.
                MiniMaxH3ChannelBalance,
                # appended 2026-09-25, the audio-only refinement mask (audio_refine.py)
                MiniMaxH3AudioRefineMask,
                # appended 2026-09-25, the refine pass's frozen-video cache (frozen_video_cache.py)
                MiniMaxH3FrozenVideoCache,
                # appended 2026-09-26, a LoRA applied at the call (lora_branch.py)
                MiniMaxH3LoRABranch,
                # appended 2026-09-26, the denoise-mask observer (denoise_mask_probe.py)
                MiniMaxH3DenoiseMaskProbe,
                # appended 2026-09-26, each step's x0 saved (step_x0_observer.py)
                MiniMaxH3StepX0Observer,
                # appended 2026-09-27, the redesigned Sol node (sol_attn_h3.py;
                # docs/research/2026-09-27_sol_node_redesign.md)
                MiniMaxH3Sol,
                # appended 2026-09-27, capture on core's sparse producer path
                # (core_sparse_capture.py; open_experiments #45)
                MiniMaxH3CoreSparseCapture,
                # appended 2026-09-29, a research checkpoint as an overlay on the released
                # one, by piece (overlay_loader.py, checkpoint_overlay.py)
                MiniMaxH3OverlayLoader,
                # appended 2026-10-03, references encoded apart from the prompt
                # (reference_encode.py)
                MiniMaxH3EncodeReferences, MiniMaxH3PromptOnReferences,
                # appended 2026-10-04, a source video and subject mask for the song loop
                # (video_mask.py)
                MiniMaxH3MaskedSource,
                # appended 2026-10-04, the kept rows put back between two samplers
                # (plate_restore.py)
                MiniMaxH3RestorePlate,
                # appended 2026-10-04, one subject's mask across a clip's cuts
                # (subject_track.py)
                MiniMaxH3SubjectTrack,
                # appended 2026-10-05, body parts and a matte on the tracked subject
                # (sapiens2_parts.py)
                MiniMaxH3Sapiens2Loader, MiniMaxH3SubjectParts,
                # appended 2026-10-05, SAM 3D Body's ViT-H release behind core's
                # predict and render nodes (sam3d_body_vith.py)
                # `MiniMaxH3VoidConditioning` followed here for part of 2026-10-05 and is
                # gone with the VOID code (owner, the same day). Removal moves nothing.
                MiniMaxH3SAM3DBodyViTHLoader,
                # appended 2026-10-05, the Subject Track's shot table written out for
                # review (shot_table.py)
                MiniMaxH3SaveShotTable,
                # appended 2026-10-06, the masked lane's prompt written from a few choices
                # (masked_prompt.py, masked_prompt_text.py)
                MiniMaxH3MaskedPrompt,
                # appended 2026-10-07, two of ComfyUI's departures from Meta's SAM 3.1 code
                # corrected as patches on a loaded model and text encoder (sam31_corrections.py)
                MiniMaxH3SAM31Corrections,
                # appended 2026-10-08, how clean the model is shown its visual references
                # (reference_noise.py)
                MiniMaxH3ReferenceNoise,
                # appended 2026-10-10, a tracked mask as the per-frame boxes a body-pose node takes
                # (subject_boxes.py)
                MiniMaxH3SubjectBoxes]


async def comfy_entrypoint() -> H3ExplorationsExtension:
    # before any schema is read: the Prompt List's file dropdown lists this folder
    register_wildcards_folder()
    return H3ExplorationsExtension()
