"""The ordered-reference model, and the one place labels are assigned.

**Pure stdlib on purpose.** Both a runtime node and two static checks have to
agree about what a reference list means, and the checks run with no ComfyUI on
the path. Nothing here imports torch, comfy or this pack's node modules, so
`bench/` can import it directly rather than reconstructing the rule -- which is
how two label authorities drift apart.

## Two models, and why both exist for now

The shipped node takes references through fixed sockets: `ref_image_0..2`,
`ref_video_0..2`, `ref_video_audio_0..2`, `ref_audio_0..2`. Core walks images,
then videos, then standalone audio, and pairs a soundtrack to a video **by
socket-name suffix**. Order is a property of which socket you plugged into, and
"standalone audio comes last" is not a choice anyone made -- it falls out of
the iteration order.

The ordered model replaces both of those deliberately, and
`docs/research/conditioning_nodes.md` records which contracts are PRESERVED and
which are intentionally REPLACED. Preserved: one shared `<Audio j>` counter, a
soundtrack's label emitted immediately before its video's, a sounded video
producing two presentation items but one DiT block. Replaced: pairing becomes
**ownership** -- a video record carries its own soundtrack, so a mis-numbered
socket cannot silently attach the wrong track -- and position becomes
**list order**, so a standalone audio reference can precede a video.

**A test asserting AGREE on all seven contracts would reject a correct
replacement for doing its job.** `legacy_order` exists so equivalence can be
demonstrated where it is claimed: feed it a socket-shaped input and this module
reproduces the legacy labels exactly, which is the AGREE half. The DIFFER half
is any ordering the socket model could not express.

## Retirement

`bench/check_ref_prompt_labels.wired_labels` stays the authority for graphs
that still wire sockets, and is retired when the last one is repointed --
never maintained alongside this as a second permanent authority.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRef:
    """One still, labelled `<Picture N>`. Keyframes count as images here."""
    name: str = ""


@dataclass(frozen=True)
class VideoRef:
    """One clip, labelled `<Video N>`, which OWNS its optional soundtrack.

    `has_soundtrack` is the whole difference from the socket model. There, a
    track belonged to a video because their socket numbers matched; here it
    belongs because the record says so, and there is no numbering to mismatch.
    """
    name: str = ""
    has_soundtrack: bool = False


@dataclass(frozen=True)
class AudioRef:
    """A standalone audio reference, labelled `<Audio N>` from the shared counter."""
    name: str = ""


def assign_labels(records) -> list[str]:
    """The labels the tokenizer will emit, in emission order.

    Three rules, and all three are preserved from core rather than invented:

    - `<Picture N>` and `<Video N>` count independently, from 1.
    - `<Audio N>` is **one shared counter** across soundtracks and standalone
      audio. A prompt saying `<Audio 1>` means something different depending
      on whether a video with sound precedes it, which is why the counter
      cannot be split.
    - A soundtrack's `<Audio j>` is emitted **immediately before** its own
      `<Video k>`, never after.

    What is NOT preserved is that images lead and standalone audio trails.
    Records are emitted in list order, so the caller's order is the answer.
    """
    labels: list[str] = []
    n_img = n_vid = n_aud = 0
    for rec in records:
        if isinstance(rec, ImageRef):
            n_img += 1
            labels.append(f"<Picture {n_img}>")
        elif isinstance(rec, VideoRef):
            if rec.has_soundtrack:
                n_aud += 1
                labels.append(f"<Audio {n_aud}>")
            n_vid += 1
            labels.append(f"<Video {n_vid}>")
        elif isinstance(rec, AudioRef):
            n_aud += 1
            labels.append(f"<Audio {n_aud}>")
        else:
            raise TypeError(f"not a reference record: {rec!r}")
    return labels


def legacy_plan(inputs) -> list:
    """Core's grouped sockets -> the ordered plan they actually mean.

    **The authority here is core's observed behaviour, not
    `check_ref_prompt_labels.wired_labels`.** That distinction is not
    pedantic: `wired_labels` reduces soundtracks to a COUNT and pairs
    `k < n_vaud`, so it cannot represent a sparse socket set, and it gets one
    wrong. Wiring `ref_video_0`, `ref_video_1` and only `ref_video_audio_1`,
    core emits `video, audio, video` -- the track belongs to the SECOND clip,
    by suffix. `wired_labels` reports the audio against the first. Measured
    2026-08-22 by driving core; the ordered model is built against core.

    So this adapter pairs by **exact suffix**, the way
    `comfy_extras/nodes_minimax_h3.py:313-314` does -- `ref_video_audio_N`
    belongs to `ref_video_N` through a string join, and nothing else. It then
    preserves core's group order: images, then videos each carrying whatever
    track its own suffix names, then standalone audio.

    That is the whole legacy model expressed as a plan, which is what makes
    equivalence checkable against core rather than against another
    reimplementation of core.
    """
    def _wired(prefix):
        return sorted(
            (k for k, v in inputs.items()
             if k.startswith(prefix) and v is not None),
            key=lambda k: int(k.rsplit("_", 1)[-1]))

    records: list = []
    # A keyframe is a <Picture N> as much as a reference is, and core emits
    # them from the plain images= list before any reference.
    for key in ("first_frame", "last_frame"):
        if inputs.get(key) is not None:
            records.append(ImageRef(name=key))
    records += [ImageRef(name=k) for k in _wired("ref_images.ref_image_")]

    tracks = {k.rsplit("_", 1)[-1] for k in
              _wired("ref_video_audios.ref_video_audio_")}
    for k in _wired("ref_videos.ref_video_"):
        suffix = k.rsplit("_", 1)[-1]
        records.append(VideoRef(name=k, has_soundtrack=suffix in tracks))

    records += [AudioRef(name=k) for k in _wired("ref_audios.ref_audio_")]
    return records


def plan_kinds(records) -> list[str]:
    """The record sequence as core's own `ref_items` type strings.

    Core appends TWO presentation entries for a sounded video -- the
    soundtrack's, then the video's -- and one DiT block. Comparing this
    against core's `ref_items` is a stronger equivalence than comparing
    labels, because it checks the sequence rather than a rendering of it.
    """
    kinds: list[str] = []
    for rec in records:
        if isinstance(rec, ImageRef):
            kinds.append("image")
        elif isinstance(rec, VideoRef):
            if rec.has_soundtrack:
                kinds.append("audio")
            kinds.append("video")
        elif isinstance(rec, AudioRef):
            kinds.append("audio")
        else:
            raise TypeError(f"not a reference record: {rec!r}")
    return kinds
