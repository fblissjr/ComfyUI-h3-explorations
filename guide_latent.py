"""Guide-latent conditioning for MiniMax H3 video and single-frame editing.

Attaches a full-clip or single-frame guide tensor (and optional spatial/temporal
mask or audio track) to `minimax_keyframes` on `positive` conditioning, where
`PackedLayout` (`comfy/ldm/minimax/model.py:375-397`) injects it into the DiT's
non-denoised `cond` segment at `VISUAL_COND_TIMESTEP = 0.999` (`img_update =
False`) with rotary position alignment at `cond_t = cursor + FRAME_RESCALE *
resolved_frame_index`.

What this node closes over ComfyUI core's `MiniMaxH3AddGuide`:

1. **Single-frame (`t=1`) guide-latent editing (`single_frame=True`).**
   `MiniMaxH3ReferenceConditioning` and `MiniMaxH3ReferenceToVideo` enforce a
   minimum `length` of 5 frames (`temporal_shape(5) -> latent_t=2`). To run a
   guide-latent control LoRA on a single still image with `P-H3-Image`, graphs
   previously had to chain three conditioning nodes (`PH3ImageConditioning` for
   a `t=1` latent, `MiniMaxH3ReferenceToVideo(length=5)` for `minimax_refs`
   without `"Edit <Picture 1>."` prefixing, and `MiniMaxH3AddGuide` on top) and
   discard the 5-frame latent. With `single_frame=True`, `MiniMaxH3GuideLatent`
   replaces the incoming AV latent with a 1-frame AV latent (`video: [B, 24, 1,
   H/16, W/16], audio: [B, 32, 2, 2]`) at the same spatial canvas and anchors
   `guide[:1]` at `resolved_frame_index = 0`.

2. **First-class spatial/temporal `mask` guide keyframe.**
   Paired guide+mask control adapters (such as `minimax_h3_lms_v1.0.safetensors`)
   consume a second keyframe entry in `minimax_keyframes` at the same
   `resolved_frame_index` holding the VAE-encoded 3-channel mask `[T, H, W, 3]`,
   without requiring external mask-to-RGB conversion and chained `MiniMaxH3AddGuide`
   nodes.
"""

from __future__ import annotations

import math

import torch
import comfy.model_management
import comfy.nested_tensor
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import _av_latent, _encode_ref_audio, _resize


def resolve_guide_frames(
    batch_frames: int,
    frame_count: int,
    frame_idx: int,
    *,
    single_frame: bool = False,
) -> tuple[int, int]:
    """Return `(resolved_frame_index, guide_frames)` or raise `ValueError`.

    - When `single_frame=True`, the target latent is collapsed to 1 frame, so
      `frame_count` is 1, `guide_frames` is 1, and only `frame_idx` resolving
      to frame 0 (`0` or `-1`) is legal.
    - When `single_frame=False`, `batch_frames < 5` anchors 1 frame and
      `batch_frames >= 5` snaps down to the model's `17k + 5` temporal grid.
    """
    if batch_frames < 1:
        raise ValueError("MiniMaxH3GuideLatent: guide image batch is empty")
    eff_frame_count = 1 if single_frame else frame_count
    if eff_frame_count < 1:
        raise ValueError(
            f"MiniMaxH3GuideLatent: target latent has invalid frame_count={eff_frame_count}"
        )
    if single_frame or batch_frames < 5:
        guide_frames = 1
    else:
        guide_frames = 5 + ((batch_frames - 5) // 17) * 17

    resolved = frame_idx if frame_idx >= 0 else eff_frame_count + frame_idx
    if resolved < 0 or resolved + guide_frames > eff_frame_count:
        if single_frame:
            raise ValueError(
                f"MiniMaxH3GuideLatent: single_frame=True has 1 frame "
                f"(valid frame_idx is 0 or -1), got frame_idx={frame_idx}"
            )
        if guide_frames == 1:
            raise ValueError(
                f"MiniMaxH3GuideLatent: frame_idx {frame_idx} is outside the "
                f"video's {eff_frame_count} frames"
            )
        raise ValueError(
            f"MiniMaxH3GuideLatent: a {guide_frames}-frame guide clip at "
            f"frame_idx {frame_idx} does not fit in the video's {eff_frame_count} frames"
        )
    return resolved, guide_frames


def mask_to_rgb(mask: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Convert a ComfyUI `MASK` (`[H, W]`, `[T, H, W]`, or `[B, 1, H, W]`) to a
    3-channel RGB image batch `[target_frames, H, W, 3]` in `[0, 1]`."""
    if not isinstance(mask, torch.Tensor):
        raise ValueError("MiniMaxH3GuideLatent: mask must be a torch.Tensor")
    m = mask
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4 and m.shape[1] == 1:
        m = m.squeeze(1)
    elif m.ndim != 3:
        raise ValueError(
            f"MiniMaxH3GuideLatent: expected MASK with shape [H, W] or [T, H, W], "
            f"got {tuple(mask.shape)}"
        )
    if m.shape[0] < 1:
        raise ValueError("MiniMaxH3GuideLatent: mask batch is empty")
    if m.shape[0] == 1 and target_frames > 1:
        m = m.expand(target_frames, -1, -1)
    elif m.shape[0] < target_frames:
        raise ValueError(
            f"MiniMaxH3GuideLatent: mask has {m.shape[0]} frames, fewer than "
            f"the {target_frames} guide frames being anchored"
        )
    else:
        m = m[:target_frames]
    return m.clamp(0.0, 1.0).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def _validate_av_latent(latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("MiniMaxH3GuideLatent expects a MiniMax H3 AV latent dict")
    samples = latent["samples"]
    if (
        not getattr(samples, "is_nested", False)
        or len(samples.tensors) != 2
        or samples.tensors[0].ndim != 5
        or samples.tensors[0].shape[1] != 24
    ):
        raise ValueError("MiniMaxH3GuideLatent expects a MiniMax H3 AV latent")
    return samples.tensors[0], samples.tensors[1]


class MiniMaxH3GuideLatent(io.ComfyNode):
    """Attach a guide video/image (and optional mask or audio) to `minimax_keyframes`."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3GuideLatent",
            display_name="MiniMax H3 Guide Latent",
            category="MiniMaxH3",
            description=(
                "Anchor a guide video clip or still image (plus optional spatial/temporal "
                "mask or audio) into `minimax_keyframes` on `positive` conditioning. "
                "Set `single_frame=True` to collapse the AV latent to 1 frame (`t=1`) "
                "for P-H3-Image still-image guide-latent editing without multi-node "
                "conditioning workarounds."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Video or single-frame VAE, required when `guide` or `mask` is connected.",
                ),
                io.Latent.Input("latent"),
                io.Image.Input(
                    "guide",
                    optional=True,
                    tooltip=(
                        "Guide image or video frames to anchor in the DiT `cond` segment. "
                        "Multi-frame video batches snap down to the model's 17k+5 grid "
                        "(5, 22, 39, ..., 124, ...); batches shorter than 5 frames or "
                        "`single_frame=True` anchor the first frame."
                    ),
                ),
                io.Int.Input(
                    "frame_idx",
                    default=0,
                    min=-3600,
                    max=3600,
                    tooltip=(
                        "Pixel frame index to anchor the guide's first frame at. "
                        "Negative values count from the end of the target latent."
                    ),
                ),
                io.Boolean.Input(
                    "single_frame",
                    default=False,
                    tooltip=(
                        "When True, replaces `latent` with a 1-frame AV latent "
                        "(video [B, 24, 1, H/16, W/16], audio [B, 32, 2, 2]) at the same "
                        "canvas size and anchors `guide[:1]` at frame 0 for single-frame "
                        "P-H3-Image guide-latent editing."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional spatial/temporal mask to VAE-encode and append as a "
                        "second keyframe at the same `resolved_frame_index` for paired "
                        "guide+mask control LoRAs."
                    ),
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Optional soundtrack to anchor starting at `frame_idx`, cropped "
                        "to the target video's remaining duration."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="Audio VAE, required when `audio` is connected.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        latent,
        frame_idx: int = 0,
        single_frame: bool = False,
        vae=None,
        guide=None,
        mask=None,
        audio=None,
        audio_vae=None,
    ) -> io.NodeOutput:
        video, audio_tensor = _validate_av_latent(latent)
        if mask is not None and guide is None:
            raise ValueError(
                "MiniMaxH3GuideLatent: `mask` requires a `guide` image or video batch"
            )
        if guide is None and audio is None:
            raise ValueError(
                "MiniMaxH3GuideLatent needs a `guide` image/video or `audio` to anchor"
            )
        if single_frame and audio is not None:
            raise ValueError(
                "MiniMaxH3GuideLatent: `audio` cannot be anchored when `single_frame=True`"
            )

        batch_size = int(video.shape[0])
        height = int(video.shape[3]) * 16
        width = int(video.shape[4]) * 16
        if single_frame:
            out_latent = _av_latent(width, height, 1, 2, batch_size=batch_size)
            frame_count = 1
        else:
            out_latent = latent
            frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(int(video.shape[2])))

        if guide is not None:
            if vae is None:
                raise ValueError(
                    "MiniMaxH3GuideLatent: anchoring `guide` frames needs the `vae` input"
                )
            resolved_frame_index, guide_frames = resolve_guide_frames(
                int(guide.shape[0]),
                frame_count,
                frame_idx,
                single_frame=single_frame,
            )
        else:
            resolved_frame_index, guide_frames = resolve_guide_frames(
                1,
                frame_count,
                frame_idx,
                single_frame=single_frame,
            )

        keyframe: dict = {"resolved_frame_index": resolved_frame_index}
        if guide is not None:
            frames = _resize(guide[:guide_frames], width, height, "center")
            keyframe["latent"] = vae.encode(frames)

        if audio is not None:
            if audio_vae is None:
                raise ValueError(
                    "MiniMaxH3GuideLatent: anchoring `audio` needs the `audio_vae` input"
                )
            audio_latent, audio_rt = _encode_ref_audio(audio_vae, audio)
            max_rt = math.floor(
                int(audio_tensor.shape[-1]) - FRAME_RESCALE * resolved_frame_index
            )
            if max_rt < 1:
                raise ValueError(
                    f"MiniMaxH3GuideLatent: frame_idx {frame_idx} is past the end of "
                    f"the video's audio track"
                )
            if audio_rt > max_rt:
                audio_latent = audio_latent[..., :max_rt].clone()
            keyframe["audio_latent"] = audio_latent

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)

        if mask is not None:
            mask_rgb = mask_to_rgb(mask, guide_frames)
            mask_resized = _resize(mask_rgb, width, height, "center")
            keyframes.append(
                {
                    "resolved_frame_index": resolved_frame_index,
                    "latent": vae.encode(mask_resized),
                }
            )

        out_positive = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
        return io.NodeOutput(out_positive, out_latent)
