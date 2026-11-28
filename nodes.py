"""ComfyUI node registry for ComfyUI-h3-explorations (`H3ExplorationsExtension`).

ComfyUI registers nodes by `schema.node_id` (`NODE_CLASS_MAPPINGS[schema.node_id]`),
so `get_node_list()` may be organized cleanly by pipeline stage. Only each node's
`node_id=` string and per-node `inputs`/`outputs` order in `define_schema()` must
remain append-only for saved-graph compatibility (`bench/check_node_ids.py`).
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension

from .audio_freeze import (
    MiniMaxH3AudioAttentionGain,
    MiniMaxH3EncodeTrack,
    MiniMaxH3FreezeAudio,
    MiniMaxH3FreezeAudioWindow,
)
from .audio_freeze_song import MiniMaxH3AudioFreezeSong
from .audio_ref_router import MiniMaxH3AudioRefRouter
from .audio_refine import MiniMaxH3AudioRefineMask
from .auto_edit_prompt import MiniMaxH3AutoEditPrompt
from .body_pose import MiniMaxH3BodyMeshVideo, MiniMaxH3BodyModelLoader, MiniMaxH3BodyPose
from .conditioning import MiniMaxH3Conditioning
# Keep `control_vector_nodes` imported before `steering.query_router` so
# `_ensure_steering_pkg()` registers `steering` under synthetic `_h3pack` loads.
from .control_vector_nodes import (
    MiniMaxH3ApplyControlVectors,
    MiniMaxH3LayerNormMatch,
    MiniMaxH3LoadControlVector,
    MiniMaxH3MakeControlVector,
    MiniMaxH3MakeMediaControlVector,
    MiniMaxH3ReleaseReferences,
    MiniMaxH3Steer,
    MiniMaxH3WordVector,
)
from .core_sparse_capture import MiniMaxH3CoreSparseCapture
from .echo_edit import MiniMaxH3EchoEdit
from .edit_guidance import MiniMaxH3EditGuider, MiniMaxH3PromptEdit
from .exact_blocks import MiniMaxH3ExactBlocks
from .frozen_video_cache import MiniMaxH3FrozenVideoCache
from .guide_latent import MiniMaxH3GuideLatent
from .h3_encoder_loader import MiniMaxH3EncoderLoader
from .h3_image_nodes import (
    MiniMaxH3ImageDownscale,
    MiniMaxH3ImagePromptExpander,
    MiniMaxH3ImageSize,
)
from .h3_prompt_expander import MiniMaxH3PromptExpander
from .lora_branch import MiniMaxH3LoRABranch
from .masked_prompt import MiniMaxH3MaskedPrompt
from .ph3_image_nodes import (
    MiniMaxH3PH3ImageConditioning,
    MiniMaxH3PH3ImagePromptUpsampler,
)
from .plate_restore import MiniMaxH3RestorePlate
from .preflight import MiniMaxH3Preflight
from .prompt_lists import MiniMaxH3FillPromptLists, MiniMaxH3PromptList, register_wildcards_folder
from .provenance import MiniMaxH3ProvenanceStamp
from .reference_conditioning import (
    MiniMaxH3AppendRefAudio,
    MiniMaxH3AppendRefImage,
    MiniMaxH3AppendRefVideo,
    MiniMaxH3ReferenceConditioning,
)
from .reference_encode import (
    MiniMaxH3EncodeImageReferences,
    MiniMaxH3EncodeReferences,
    MiniMaxH3PromptOnImageReferences,
    MiniMaxH3PromptOnReferences,
)
from .reference_noise import MiniMaxH3ReferenceNoise
from .resolution import MiniMaxH3Resolution
from .sage_attention import MiniMaxH3SageAttention
from .sam31_corrections import MiniMaxH3SAM31Corrections
from .sam3d_body_vith import MiniMaxH3SAM3DBodyViTHLoader
from .sapiens2_parts import MiniMaxH3Sapiens2Loader, MiniMaxH3SubjectParts
from .shot_table import MiniMaxH3SaveShotTable
from .sol_attn_h3 import MiniMaxH3Sol
from .sound_cues import MiniMaxH3SoundCues
from .spatial_director import MiniMaxH3DirectorHUD, MiniMaxH3SpatialDirector
from .depth_beat_mask import MiniMaxH3DepthBeatMask
from .sprite_trajectory import MiniMaxH3AppendSpriteTrajectory
from .step_x0_observer import MiniMaxH3StepX0Observer
from .subject_boxes import MiniMaxH3SubjectBoxes
from .subject_track import MiniMaxH3SubjectTrack
from .vae_precision import MiniMaxH3VAEPrecision
from .video_mask import MiniMaxH3MaskedSource
from steering.live_policy import MiniMaxH3SelfMaskRouter
from steering.query_router import MiniMaxH3QueryRouter
from steering.spatial_query_router import MiniMaxH3SpatialQueryRouter


def _channel_balance_class():
    """Defer to sibling ComfyUI-H3-Quant pack when installed so one object registers."""
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


class H3ExplorationsExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            # --- Encoders, Loaders, LoRAs, Attention & Precision ---
            MiniMaxH3EncoderLoader,  # Guarded Qwen3-VL loader (marker check + key audit)
            MiniMaxH3LoRABranch,  # Unmerged int8-safe compute-dtype LoRA branch
            # MiniMaxH3OverlayLoader retired 2026-11-25 to archive/overlay_loader.py.
            MiniMaxH3VAEPrecision,  # Separate fp32 VAE encode / fp16-bf16 decode precision
            MiniMaxH3SageAttention,  # Direct NHD SageAttention + rotated QK + int64 indexing
            MiniMaxH3Sol,  # comfy-kitchen Sol sparse attention with conditioning sinks
            MiniMaxH3ChannelBalance,  # Opt-in SmoothQuant Q/K channel fold (shared w/ H3-Quant)
            MiniMaxH3ExactBlocks,  # Per-block exact SDPA/flash attention bypass

            # --- Video Conditioning, References, Guides & Sizing ---
            MiniMaxH3Resolution,  # Trained-family canvas picker & same-as-picture snap
            MiniMaxH3Preflight,  # Pre-sample PackedLayout token & int32 stride counter
            MiniMaxH3Conditioning,  # Joint AV conditioning (crop/prompt fixes, t=1 support)
            MiniMaxH3AppendRefImage,  # Ordered <Picture N> reference builder
            MiniMaxH3AppendRefVideo,  # Ordered <Video N> reference builder (17n+5, 2 fps Qwen)
            MiniMaxH3AppendRefAudio,  # Ordered hop-aligned <Audio> reference builder
            MiniMaxH3ReferenceConditioning,  # Compiler for ordered typed references
            MiniMaxH3EncodeReferences,  # Cache-once Qwen3-VL prefix & VAE video ref encoder
            MiniMaxH3PromptOnReferences,  # Prompt compiler reusing cached ref prefix
            MiniMaxH3ReferenceNoise,  # Clean-level override for visual reference rows
            MiniMaxH3GuideLatent,  # Keyframe & spatial/temporal mask guide conditioning

            # --- Single-Frame Image (`MiniMax-H3-Image` & `P-H3-Image`) ---
            MiniMaxH3ImageSize,  # 4 MP+ generation sizing for H3-Image / P-H3-Image
            MiniMaxH3ImageDownscale,  # Aspect-preserving post-decode Lanczos shrink
            MiniMaxH3ImagePromptExpander,  # Local heylook SSE prompt/aspect expander (image)
            MiniMaxH3EncodeImageReferences,  # 16-still Qwen/VAE prefix cache (H3-Image edits)
            MiniMaxH3PromptOnImageReferences,  # Prompt compiler on cached H3-Image references
            MiniMaxH3PH3ImageConditioning,  # P-H3-Image parity conditioning (4-8 MP, seed=42)
            MiniMaxH3PH3ImagePromptUpsampler,  # Lazy Qwen3.5-9B upsampler for P-H3-Image

            # --- Prompt Expansion, Edit Guidance, Steering, Query Routing & Spatial Director ---
            MiniMaxH3PromptExpander,  # 1x local heylook video prompt expander (+ ref_end_percent)
            MiniMaxH3AutoEditPrompt,  # 2x surgical heylook prompt rewriter -> EditPlan
            MiniMaxH3PromptEdit,  # Safe substring prompt-edit compiler
            MiniMaxH3EditGuider,  # Vector-free orig + s*(edit-orig) guider w/ [F,H,W] masks
            MiniMaxH3SoundCues,  # Procedural 3-frame-snapped cond_audio islands
            MiniMaxH3AppendSpriteTrajectory,  # 2D kinematic D-pad sprite <Video N> trajectory
            MiniMaxH3SpatialDirector,  # Interactive portal/target/mask 2D+time world director
            MiniMaxH3DepthBeatMask,  # Composable MoGe-2 depth + WAV beat [F,H,W] MASK producer
            MiniMaxH3DirectorHUD,  # Director / Split A/B diagnostic HUD overlay exporter
            MiniMaxH3QueryRouter,  # Single-pass 1x cross-modal Q_vid / Q_aud depth router
            MiniMaxH3SpatialQueryRouter,  # Single-pass 1x spatial box/mask visual world router
            MiniMaxH3SelfMaskRouter,  # Zero-coordinate Step-0 self-masked router & H3LivePolicy bus
            MiniMaxH3Steer,  # Post-refiner subspace erasure / 4-bus steering dial
            MiniMaxH3WordVector,  # J-Lens lexical contrastive vector synthesizer
            MiniMaxH3MakeMediaControlVector,  # Contrastive steering vector from media pairs
            MiniMaxH3ReleaseReferences,  # Independent ref/video/audio/guide end_percent release
            MiniMaxH3LoadControlVector,  # Saved .safetensors vector file loader
            MiniMaxH3MakeControlVector,  # Legacy text-pair steering vector builder
            MiniMaxH3LayerNormMatch,  # Legacy post-refiner conditioning row norm matcher
            MiniMaxH3ApplyControlVectors,  # [Deprecated] Multi-vector steering monolith
            MiniMaxH3PromptList,  # Deterministic __wildcard__ file list loader
            MiniMaxH3FillPromptLists,  # Multi-window wildcard prompt filler

            # --- Audio Freeze, Chained Song Loops & Audio Refinement ---
            MiniMaxH3FreezeAudio,  # 40 Hz audio freeze on NestedTensor(video=1, audio=0)
            MiniMaxH3FreezeAudioWindow,  # Windowed audio-freeze loop segment conditioning
            MiniMaxH3EncodeTrack,  # One-shot stereo PCM -> 40 Hz latent track encoder
            MiniMaxH3AudioAttentionGain,  # Per-block audio-token attention gain multiplier
            MiniMaxH3AudioRefRouter,  # Single-pass 1x Q_aud block-depth melody/timbre router
            MiniMaxH3AudioFreezeSong,  # Timeline-driven chained song-loop AV driver
            MiniMaxH3AudioRefineMask,  # Reopen audio latents for pass-2 refinement
            MiniMaxH3FrozenVideoCache,  # Cached frozen-video K/V projections across refine

            # --- Masked Video-to-Video & Perception (`SAM 3.1`, `Sapiens2`, `SAM 3D Body`) ---
            MiniMaxH3SAM31Corrections,  # Non-destructive SAM 3.1 trunk/GELU/multiplex fixes
            MiniMaxH3SubjectTrack,  # Cut-aware single-subject SAM 3.1 mask tracker
            MiniMaxH3SaveShotTable,  # Shot-table & gap summary exporter
            MiniMaxH3SubjectBoxes,  # Per-frame bounding boxes extracted from subject masks
            MiniMaxH3Sapiens2Loader,  # Sapiens2 body-part segmentation model loader
            MiniMaxH3SubjectParts,  # Isolated per-frame part masks & edge mattes
            MiniMaxH3BodyModelLoader,  # SAM 3D Body model loader (Meta OpenCV warp parity)
            MiniMaxH3BodyPose,  # Per-frame body & hand pose estimator on subject mask
            MiniMaxH3BodyMeshVideo,  # 4-color marked body/hand mesh video renderer
            MiniMaxH3SAM3DBodyViTHLoader,  # ViT-H checkpoint loader behind SAM 3D Body nodes
            MiniMaxH3MaskedSource,  # Causal-pooled AV mask & keep-out compiler
            MiniMaxH3RestorePlate,  # Kept-plate latent restoration between two samplers
            MiniMaxH3MaskedPrompt,  # Structured <Subject N> masked-hole prompt builder

            # --- Provenance & Observability ---
            MiniMaxH3ProvenanceStamp,  # Resolve/stamp sigmas, Sol closure, HEAD & grid telemetry
            # MiniMaxH3DenoiseMaskProbe retired 2026-11-25 to archive/denoise_mask_probe.py.
            MiniMaxH3StepX0Observer,  # Save per-step x0 latent predictions
            MiniMaxH3CoreSparseCapture,  # Capture core BlockSparseAttention Q/K/V tensors
            MiniMaxH3EchoEdit,  # Inversion-free EchoEdit+ + AV-FlowEdit sampler
        ]


async def comfy_entrypoint() -> H3ExplorationsExtension:
    # before any schema is read: the Prompt List's file dropdown lists this folder
    register_wildcards_folder()
    return H3ExplorationsExtension()
