"""What this render will actually cost, before you queue it.

Every other node in the graph knows one piece. `MiniMaxH3Resolution` knows the
video and nothing about references. `MiniMaxH3ReferenceFit` knows one
reference and nothing about the video. The total only exists once conditioning
is assembled, which is where this sits.

It reads the same `PackedLayout` the model builds, so the sequence length here
is the one attention will run at rather than an estimate. Four things it
reports that nothing else can:

**Whether you are inside the trained family.** Core's conditioning nodes take
width and height as plain ints and never call `adapt_canvas`, so the 768 short
edge and the 768x1344 area cap constrain nothing you type. 1024x1024 is legal,
32-divisible, renders, costs more per frame than 16:9, and is outside the
family the checkpoint was trained on. Nothing else says so.

**What the segments cost relative to each other.** Reference tokens ride every
sampling step exactly as video tokens do, so "my references are 36% of the
sequence" is the number that decides whether to resize them.

**What a different aspect ratio would cost.** A tradeoff you cannot act on is
not a tradeoff. The alternatives are computed at the same length and
conditioning, so the comparison is honest.

**Where the int32 thresholds sit for the layout H3 actually uses.** q, k and v
are views of one fused buffer with `stride_seq = 3*heads*head_dim = 21504`, so
the Triton quant kernels' int32 offset crosses near 99,864 tokens, not the
299,593 a contiguous layout would give. KJNodes' Token Counter computes the
contiguous figure and so stays silent through the range that matters. That
crossing is fixed in every sage build able to run this repo's attention node,
because `build_kernel` refuses any sageattention without `sageattn_consume`
and the int64 fix precedes it. The next ceiling is the `csrc/fused` uint32
wrap near 199,728, which no legal H3 length reaches on its own -- about 660 frames
against a 345 maximum. Both numbers are stated so an absent warning is not
mistaken for clearance.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

try:
    from .h3_rules import describe_length
except ImportError:  # pragma: no cover
    from h3_rules import describe_length  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# stride_seq for a fused qkv view at H3's 56 heads x 128 head_dim, and the
# contiguous figure for contrast. See the module docstring.
_FUSED_STRIDE = 3 * 56 * 128
_CONTIGUOUS_STRIDE = 56 * 128
_INT32_FUSED = 2**31 // _FUSED_STRIDE
_CSRC_FUSED = 2**32 // _FUSED_STRIDE

_ALTERNATIVES = ("1:1", "4:3", "3:2", "16:9", "9:16")


def _bar(fraction, width=20):
    filled = max(0, min(width, round(fraction * width)))
    return "#" * filled + "." * (width - filled)


class MiniMaxH3Preflight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Preflight",
            display_name="MiniMax H3 Preflight",
            category="model/conditioning/minimax",
            description=(
                "Reports what this render will cost before you queue it: "
                "sequence length broken down by segment, whether the "
                "resolution is inside the trained family, what other aspect "
                "ratios would cost at the same length, and where the int32 "
                "limits sit for H3's fused layout. Pass-through; it changes "
                "nothing. Wire it between conditioning and the sampler."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("samples"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="samples"),
                io.Int.Output(display_name="sequence_length"),
                io.String.Output(display_name="report"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, conditioning, samples) -> io.NodeOutput:
        from comfy.ldm.minimax.model import PackedLayout
        from comfy_extras.nodes_minimax_h3 import adapt_canvas

        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            video, audio = latent.unbind()[:2]
            audio_t = audio.shape[-1]
        else:
            video, audio_t = latent, 0
        if video.ndim != 5:
            raise RuntimeError(
                f"Expected an H3 video latent of shape [B, C, T, H, W]; got "
                f"{tuple(video.shape)}. Wire this to the latent the H3 "
                f"conditioning node produced.")

        latent_t = video.shape[2]
        # h/w round up to the DiT's 2x2 patch, matching model_base's extra_conds
        lat_h = (video.shape[3] + 1) // 2 * 2
        lat_w = (video.shape[4] + 1) // 2 * 2
        width, height = lat_w * 16, lat_h * 16

        # Scheduled conditioning can differ in text length; report the largest,
        # because the peak is what has to fit.
        layout = max(
            (PackedLayout(cond.shape[1], latent_t, lat_h, lat_w, audio_t,
                          keyframes=cd.get("minimax_keyframes"),
                          refs=cd.get("minimax_refs"),
                          frame_count=cd.get("minimax_frame_count"))
             for cond, cd in conditioning),
            key=lambda l: l.seq_len)

        by_kind: dict[str, int] = {}
        for a, b, kind in layout.segments:
            by_kind[kind] = by_kind.get(kind, 0) + (b - a)
        total = layout.seq_len

        tokens_per_frame = (width // 32) * (height // 32)
        in_family = adapt_canvas(width, height) == (width, height)
        # `minimax_frame_count` is set ONLY on the keyframe path -- core
        # writes it inside `if keyframes:` and MiniMaxH3ReferenceToVideo never
        # writes it at all. Sourcing the duration line from it alone meant the
        # line vanished on 7 of the 8 shipped graphs, including every ref
        # graph, which is exactly where the 345-frame ceiling matters most.
        # `latent_t` is already in hand, so derive it when the key is absent
        # rather than printing nothing and letting absence read as "fine".
        frames = None
        for _cond, cd in conditioning:
            if cd.get("minimax_frame_count"):
                frames = cd["minimax_frame_count"]
                break
        derived = frames is None
        if derived and latent_t:
            # inverse of video_latent_t: latent_t = ((n - 5) // 17) * 5 + 2
            frames = ((latent_t - 2) // 5) * 17 + 5 if latent_t > 2 else 5

        label = {"text": "text", "cond": "keyframes", "ref_img": "references",
                 "ref_audio": "audio refs", "audio": "audio", "video": "video"}
        lines = [
            f"{width}x{height}  "
            f"{'trained family' if in_family else 'OUTSIDE trained family'}"
            f"  {tokens_per_frame} video tokens/frame",
        ]
        if frames:
            lines.append(f"{describe_length(frames)}  {latent_t} latent frames"
                         + ("  (derived from the latent)" if derived else ""))
        lines.append(f"sequence length {total:,}")
        for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {label.get(kind, kind):<12}{n:>8,}  "
                         f"{_bar(n / total)}  {100 * n / total:5.1f}%")

        # Alternatives at the same length and conditioning: only the video
        # segment moves, so the comparison is exact rather than modelled.
        video_tokens = by_kind.get("video", 0)
        rest = total - video_tokens
        lines.append("if the aspect ratio changed, same length:")
        for name in _ALTERNATIVES:
            aw, ah = (int(x) for x in name.split(":"))
            cw, ch = adapt_canvas(aw * 1000, ah * 1000)
            alt_video = (cw // 32) * (ch // 32) * latent_t
            alt = rest + alt_video
            mark = "  <- current" if (cw, ch) == (width, height) else ""
            lines.append(f"  {name:<6}{cw}x{ch:<6}{alt:>9,}  "
                         f"{(alt - total) / total:+6.0%}{mark}")

        if total >= _CSRC_FUSED:
            lines.append(f"int32: {total:,} is past the csrc/fused uint32 wrap "
                         f"at {_CSRC_FUSED:,}. This one is NOT fixed.")
        elif total >= _INT32_FUSED:
            lines.append(
                f"int32: past the fused crossing at {_INT32_FUSED:,}, which "
                f"every sage build that can run this node has fixed. Next "
                f"ceiling {_CSRC_FUSED:,}, unreachable at legal lengths.")
        else:
            lines.append(f"int32: under the fused crossing at {_INT32_FUSED:,} "
                         f"(a contiguous layout would say {_CONTIGUOUS_STRIDE and 2**31 // _CONTIGUOUS_STRIDE:,}).")

        report = "\n".join(lines)
        logger.info("[h3] preflight %s", report.replace("\n", " | "))

        unique_id = getattr(cls.hidden, "unique_id", None)
        if unique_id:
            try:
                from server import PromptServer
                PromptServer.instance.send_progress_text(report, unique_id)
            except Exception:
                pass

        return io.NodeOutput(conditioning, samples, total, report)
