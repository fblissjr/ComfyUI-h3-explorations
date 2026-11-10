"""What an ordered H3 reference list costs, before anything is encoded.

Every reference is read twice. The video VAE encodes one copy into reference
latent rows that the DiT attends on every sampling step; Qwen3-VL reads a copy
as vision tokens that sit in the text segment ahead of the prompt, and the
hidden states of those positions ride the DiT's text segment on every step
too. `MiniMaxH3AppendRefImage.size_policy` sizes the first copy and its
`qwen_view` the second.

`price_references` takes the same references, prompt and canvas the
conditioner takes, prices both copies of every reference with the SAME sizing
functions the conditioner calls (`reference_geometry.fit_reference_image`,
`reference_conditioning.qwen_view_size`, `reference_geometry.qwen_image_size`),
builds the packed sequence the model will build (`comfy.ldm.minimax.model
.PackedLayout`, from shapes alone) and `format_report` turns it into text.
`MiniMaxH3ReferenceConditioning` calls both for its preview. No VAE, no encoder
forward: the only model it touches is the tokenizer, for the prompt's token
count.

This module held a node, `MiniMaxH3ReferenceReport`, and a report picture until
0.173.0 (owner, 2026-09-29): no graph used it and the conditioner's preview
carries the same text. The pricing functions stay.

**What is exact and what is not.** Every pixel size, every DiT row count and
every vision-token count is the geometry the conditioner will produce, from
the same functions. The prompt and label token counts come from the installed
tokenizer. Reference audio rows are an estimate: the aligned encode pads to
the audio VAE's hop, which this module does not load, so a soundtrack's rows
can be one hop off. The report says so on the line.

**Two ceilings apply to the encoder's copy, in order.** The selected
`image_policy` pre-applies its own bounds (none for `comfy`, the release's
declaration for `release`), and then the loaded encoder's own processor
applies core's `process_qwen2vl_images` defaults inside `preprocess_embed`.
The report shows the copy after both, because that is what lands in the
sequence, and says on the line when the second stage moved it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from comfy_extras.nodes_minimax_h3 import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    adapt_canvas,
    temporal_shape,
    video_latent_t,
)

from .h3_rules import normalize_prompt
from .reference_geometry import (
    IMAGE_POLICIES,
    fit_reference_image,
    latent_rows,
    qwen_image_size,
)

logger = logging.getLogger(__name__)

#: Merged patch: `patch_size * merge_size` of the installed Qwen3-VL processor.
#: Inherited from `comfy/text_encoders/qwen3vl.py`'s call site (16) and
#: `process_video_block`'s defaults (merge 2); one merged token per 32x32.
MERGED_PATCH = 32


@dataclass
class StillPricing:
    index: int
    label: str
    source: tuple[int, int]
    vae: tuple[int, int]
    dit_rows: int
    qwen: tuple[int, int]
    qwen_tokens: int
    separate: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class VideoPricing:
    index: int
    label: str
    source: tuple[int, int]
    prepared_frames: int
    vae: tuple[int, int]
    latent_t: int
    dit_rows: int
    qwen: tuple[int, int]
    sampled: int
    qwen_tokens: int
    has_soundtrack: bool
    audio_rows: int
    notes: list[str] = field(default_factory=list)


@dataclass
class AudioPricing:
    index: int
    label: str
    audio_rows: int
    notes: list[str] = field(default_factory=list)


@dataclass
class SequencePricing:
    canvas: tuple[int, int]
    frames: int
    latent_t: int
    image_policy: str
    video_policy: str
    encoder_bounds: tuple[int, int]
    items: list
    prompt_tokens: int | None
    label_tokens: int
    vision_tokens: int
    text_len: int
    segments: dict
    total: int

    @property
    def dit_reference_rows(self) -> int:
        return sum(getattr(i, "dit_rows", 0) for i in self.items)

    @property
    def prompt_share(self) -> float | None:
        if self.prompt_tokens is None or self.text_len == 0:
            return None
        return self.prompt_tokens / self.text_len


def _native_encoder_bounds(clip) -> tuple[int, int]:
    """The pixel bounds the loaded encoder's own processor applies to a still.

    `MiniMaxH3EncoderLoader` records them on the transformer as
    `_h3_image_bounds`; a CLIP from core's `CLIPLoader` carries nothing and
    runs the same code path, so the defaults are read out of core. Never
    typed here.
    """
    model = clip
    for attribute in ("cond_stage_model", "qwen3vl_32b", "transformer"):
        model = getattr(model, attribute, None)
        if model is None:
            break
    bounds = getattr(model, "_h3_image_bounds", None) if model is not None else None
    if bounds is not None:
        return int(bounds[0]), int(bounds[1])
    import inspect
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    params = inspect.signature(process_qwen2vl_images).parameters
    return int(params["min_pixels"].default), int(params["max_pixels"].default)


def _smart_resize(width: int, height: int, bounds: tuple[int, int]) -> tuple[int, int]:
    """The encoder's own resize, imported rather than restated."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    h, w = smart_resize(height=height, width=width, factor=MERGED_PATCH,
                        min_pixels=bounds[0], max_pixels=bounds[1])
    return int(w), int(h)


def merged_tokens(width: int, height: int) -> int:
    return (width // MERGED_PATCH) * (height // MERGED_PATCH)


def _pair_grid(width: int, height: int, bounds: tuple[int, int]) -> tuple[int, int]:
    """Core's per-pair video block sizing (`comfy/text_encoders/minimax.py`
    `process_video_block`), on shapes alone."""
    factor = MERGED_PATCH
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > bounds[1]:
        beta = math.sqrt((height * width) / bounds[1])
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < bounds[0]:
        beta = math.sqrt(bounds[0] / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar, h_bar


def _count_text_tokens(clip, text: str) -> int:
    tokens = clip.tokenize(text)
    batches = tokens["qwen3vl_32b"]
    return sum(len(batch) for batch in batches)


def price_references(records, width: int, height: int, length: int,
                     image_policy: str = "comfy", video_policy: str = "comfy",
                     clip=None, prompt: str = "") -> SequencePricing:
    """Price every record the way `_compile_reference_records` will size it."""
    from . import reference_conditioning as rc
    from .reference_order import assign_labels

    if image_policy not in IMAGE_POLICIES:
        raise ValueError(f"unknown image_policy {image_policy!r}")
    if video_policy not in rc.VIDEO_POLICIES:
        raise ValueError(f"unknown video_policy {video_policy!r}")

    frame_count, latent_t, audio_t = temporal_shape(length)
    duration = frame_count / FPS
    bounds = _native_encoder_bounds(clip) if clip is not None else (3136, 12845056)
    labels = assign_labels(rc._order_records(records))
    # One (label, soundtrack label) per record. `assign_labels` returns one
    # label per presentation item, and a video with a soundtrack is two items
    # (its `<Audio j>`, then its `<Video k>`), so indexing it by record
    # mislabelled every record after the first sounded video until 2026-10-03.
    record_labels, pointer = [], 0
    for record in records:
        sound = None
        if isinstance(record, rc.RuntimeVideoReference) and record.soundtrack is not None:
            sound, pointer = labels[pointer], pointer + 1
        record_labels.append((labels[pointer], sound))
        pointer += 1
    items = []
    blocks = []
    label_texts = []
    vision_tokens = 0

    for index, record in enumerate(records):
        label, sound_label = record_labels[index]
        if isinstance(record, rc.RuntimeImageReference):
            _, sh, sw = rc._image_shape(record.image, f"reference image {index + 1}")
            role_w, role_h = fit_reference_image(
                sw, sh, size_policy=record.size_policy,
                short_edge=record.short_edge, allow_upscale=record.allow_upscale,
                canvas_w=width, canvas_h=height)
            notes = []
            separate = bool(record.qwen_short_edge)
            if not separate:
                vae_w, vae_h = role_w, role_h
                if image_policy != "comfy":
                    vae_w, vae_h = qwen_image_size(role_w, role_h, image_policy)
                    if (vae_w, vae_h) != (role_w, role_h):
                        notes.append(f"{image_policy} bounds moved both copies "
                                     f"from {role_w}x{role_h}")
                qwen_w, qwen_h = vae_w, vae_h
            else:
                vae_w, vae_h = role_w, role_h
                qwen_w, qwen_h = rc.qwen_view_size(sw, sh, record.qwen_short_edge)
                if image_policy != "comfy":
                    pre = qwen_image_size(qwen_w, qwen_h, image_policy)
                    if pre != (qwen_w, qwen_h):
                        notes.append(f"{image_policy} bounds moved the encoder "
                                     f"copy from {qwen_w}x{qwen_h}")
                    qwen_w, qwen_h = pre
            # The encoder's own processor, after the policy.
            enc_w, enc_h = _smart_resize(qwen_w, qwen_h, bounds)
            if (enc_w, enc_h) != (qwen_w, qwen_h):
                notes.append(f"encoder's own bounds moved its copy from "
                             f"{qwen_w}x{qwen_h}")
                qwen_w, qwen_h = enc_w, enc_h
            if record.size_policy == "max":
                scale = record.short_edge / min(sw, sh)
                if scale > 1.0 and not record.allow_upscale:
                    notes.append(f"dit_short_edge {record.short_edge} inert: "
                                 f"source short edge {min(sw, sh)} is under it "
                                 f"and allow_upscale is off")
                elif scale > 1.0:
                    notes.append(f"upscaled x{scale:.2f} on each side "
                                 f"(x{scale * scale:.1f} rows)")
                elif scale < 1.0:
                    notes.append(f"shrunk to the {record.short_edge} short edge")
            elif record.size_policy == "tiered":
                tier_edge = 1024 if min(width, height) < 768 else 2048
                notes.append(f"tiered: {tier_edge} short edge from {width}x{height} canvas")
            else:
                notes.append("match: capped at the canvas area, never enlarged")
            if not separate:
                notes.append("shared: the encoder reads the video model's copy")
            rows = latent_rows(vae_w, vae_h)
            tokens = merged_tokens(qwen_w, qwen_h)
            vision_tokens += tokens
            if not getattr(record, "use_vae", True):
                # The append node turned this still's VAE copy off: the
                # compiler builds no block for it, so it costs no rows.
                rows = 0
                notes.append("use_vae off: text encoder only, no reference rows")
            items.append(StillPricing(index, label, (sw, sh), (vae_w, vae_h), rows,
                                      (qwen_w, qwen_h), tokens, separate, notes))
            if rows:
                blocks.append({"kind": "image", "latent_h": vae_h // 16,
                               "latent_w": vae_w // 16})
            label_texts.append(f"{label}: ")
            continue

        if isinstance(record, rc.RuntimeVideoReference):
            frames = rc._prepare_reference_video(record.frames, record.loaded_fps,
                                                 frame_count)
            n = int(frames.shape[0])
            _, sh, sw = rc._image_shape(frames, f"reference video {index + 1}")
            cw, ch = adapt_canvas(sw, sh)
            notes = []
            if video_policy == "comfy" and sw * sh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(sw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(sh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                notes.append("comfy: a video under the canvas is not enlarged")
            elif (cw, ch) != (sw, sh):
                notes.append(f"{video_policy}: fitted to the release canvas rule")
            lt = video_latent_t(n)
            rows = lt * latent_rows(cw, ch)
            sampled = len(range(0, n, FPS // 2))
            pairs = (sampled + 1) // 2
            if video_policy == "release":
                qw, qh = rc._configured_qwen_video_size(sampled, cw, ch, "release")
                tokens = pairs * merged_tokens(qw, qh)
                notes.append("release: the encoder's clip-wide budget sized the "
                             "2 fps samples")
            else:
                qw, qh = _pair_grid(cw, ch, bounds)
                tokens = pairs * merged_tokens(qw, qh)
            vision_tokens += tokens
            audio_rows = 0
            soundtrack = record.soundtrack
            has_sound = soundtrack is not None
            if soundtrack is not None:
                wave = soundtrack["waveform"]
                rate = int(soundtrack["sample_rate"])
                seconds = min(float(wave.shape[-1]) / rate, duration)
                audio_rows = int(round(seconds * AUDIO_LATENT_FPS)) * 2
                notes.append("soundtrack rows estimated to the target duration; "
                             "the aligned encode can add one VAE hop")
            if not getattr(record, "use_vae", True):
                # The append node turned this video's VAE copy off: the
                # compiler builds no block for it, and core builds the audio
                # latent only behind the video one, so no audio rows either.
                rows = 0
                audio_rows = 0
                notes.append("use_vae off: text encoder only, no reference rows"
                             + (", soundtrack not heard" if has_sound else ""))
            items.append(VideoPricing(index, label, (sw, sh), n, (cw, ch), lt, rows,
                                      (qw, qh), sampled, tokens, has_sound,
                                      audio_rows, notes))
            if rows:
                blocks.append({"kind": "video_audio" if audio_rows else "video",
                               "latent_t": lt, "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": audio_rows // 2})
            if has_sound:
                label_texts.append(f"{sound_label}: ")
            label_texts.append(f"{label}: ")
            label_texts.extend(f"<{(i + 0.5):.1f} seconds>" for i in range(pairs))
            continue

        if isinstance(record, rc.RuntimeAudioReference):
            wave = record.audio["waveform"]
            rate = int(record.audio["sample_rate"])
            seconds = min(float(wave.shape[-1]) / rate, duration)
            audio_rows = int(round(seconds * AUDIO_LATENT_FPS)) * 2
            items.append(AudioPricing(index, label, audio_rows, [
                "rows estimated to the target duration; the aligned encode can "
                "add one VAE hop",
                "the encoder sees only the label; audio needs the audio VAE to "
                "reach the DiT"]))
            blocks.append({"kind": "audio", "ref_audio_t": audio_rows // 2})
            label_texts.append(f"{label}: ")
            continue
        raise TypeError(f"not a runtime reference record: {record!r}")

    prompt_tokens = None
    label_tokens = 0
    if clip is not None:
        prompt_tokens = _count_text_tokens(clip, normalize_prompt(prompt)) if prompt else 0
        label_tokens = sum(_count_text_tokens(clip, t) for t in label_texts)
    # <|vision_start|> and <|vision_end|> around every vision block.
    vision_blocks = sum(1 for t in label_texts if t.startswith("<Picture")) + sum(
        1 for t in label_texts if t.endswith("seconds>"))
    text_len = (prompt_tokens or 0) + label_tokens + vision_tokens + 2 * vision_blocks

    from comfy.ldm.minimax.model import PackedLayout
    lat_h = (height // 16 + 1) // 2 * 2
    lat_w = (width // 16 + 1) // 2 * 2
    layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t, refs=blocks)
    segments: dict[str, int] = {}
    for a, b, kind in layout.segments:
        segments[kind] = segments.get(kind, 0) + (b - a)

    return SequencePricing(
        canvas=(width, height), frames=frame_count, latent_t=latent_t,
        image_policy=image_policy, video_policy=video_policy,
        encoder_bounds=bounds, items=items, prompt_tokens=prompt_tokens,
        label_tokens=label_tokens, vision_tokens=vision_tokens,
        text_len=text_len, segments=segments, total=layout.seq_len)


def format_report(p: SequencePricing) -> str:
    """The text form. Monospace; one block per reference, then the sequence."""
    out = [f"MiniMax H3 reference report  --  {p.canvas[0]}x{p.canvas[1]}, "
           f"{p.frames} frames ({p.latent_t} latent), image_policy={p.image_policy}, "
           f"video_policy={p.video_policy}",
           f"encoder still bounds {p.encoder_bounds[0]:,}..{p.encoder_bounds[1]:,} px",
           ""]
    for it in p.items:
        if isinstance(it, StillPricing):
            out.append(f"{it.label}  still {it.source[0]}x{it.source[1]}")
            out.append(f"    video model sees  {it.vae[0]}x{it.vae[1]:<6} "
                       f"{it.dit_rows:>7,} DiT rows, attended every step")
            out.append(f"    text encoder sees {it.qwen[0]}x{it.qwen[1]:<6} "
                       f"{it.qwen_tokens:>7,} vision tokens, ahead of the prompt")
        elif isinstance(it, VideoPricing):
            out.append(f"{it.label}  video {it.source[0]}x{it.source[1]}, "
                       f"{it.prepared_frames} frames after 24 fps prep")
            out.append(f"    video model sees  {it.vae[0]}x{it.vae[1]} x {it.latent_t} "
                       f"latent frames  {it.dit_rows:>7,} DiT rows")
            out.append(f"    text encoder sees {it.qwen[0]}x{it.qwen[1]} x {it.sampled} "
                       f"samples at 2 fps  {it.qwen_tokens:>7,} vision tokens")
            if it.has_soundtrack:
                out.append(f"    soundtrack        {it.audio_rows:>7,} audio rows (estimate)")
        else:
            out.append(f"{it.label}  audio  {it.audio_rows:,} rows (estimate)")
        for note in it.notes:
            out.append(f"      - {note}")
        out.append("")
    out.append("packed sequence the DiT attends on every step:")
    order = [("video", "target video"), ("audio", "target audio"),
             ("text", "text segment"), ("ref_img", "reference stills (latent rows)"),
             ("ref_video", "reference video (latent rows)"),
             ("ref_audio", "reference audio")]
    seen = set()
    for kind, name in order:
        if kind in p.segments:
            seen.add(kind)
            out.append(f"    {name:<34} {p.segments[kind]:>9,}")
    for kind, n in p.segments.items():
        if kind not in seen:
            out.append(f"    {kind:<34} {n:>9,}")
    out.append(f"    {'TOTAL':<34} {p.total:>9,}")
    out.append("")
    if p.prompt_tokens is None:
        out.append("text segment: prompt not counted (no CLIP wired); "
                   f"{p.vision_tokens:,} vision tokens + {p.label_tokens:,} label tokens")
    else:
        out.append(f"text segment: {p.prompt_tokens:,} prompt tokens + "
                   f"{p.label_tokens:,} label tokens + {p.vision_tokens:,} vision "
                   f"tokens (+ delimiters) = {p.text_len:,}")
        share = p.prompt_share or 0.0
        out.append(f"the prompt is {share:.0%} of what the text encoder reads; the "
                   f"rest is references placed ahead of it")
    return "\n".join(out)
