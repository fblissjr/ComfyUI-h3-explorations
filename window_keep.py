"""Keep a song window's source latent and its conditioning between runs of one stretch.

`MiniMaxH3AudioFreezeSong` (`audio_freeze_song.py`) encodes and samples inside
itself, so ComfyUI's node cache cannot hand a second run the parts of the
first that did not change. Two of them do not depend on the seed, the sampler
or the schedule, and a test run that changes only those computes them again:
each window's source latent (the plate through the video VAE) and each
window's conditioning. What they cost against a whole run is in
`bench/results/2026-10-06_masked_render_time_breakdown.md`. This module keeps
both in the server's memory, for the session.

## The key is identity, never a hash of gigabytes

A kept value is found by two things together:

  plain values   what the encode was asked for: the window's first frame and
                 length, the canvas, the text, the settings that change what
                 is encoded.
  live objects   what it was computed from: the loader's frame tensor, the
                 VAE, the encoder, each reference record. An entry matches
                 only while each is the SAME OBJECT (`is`), held here by weak
                 reference.

On a second queue in one server session core's cache hands back the same
objects for every node upstream that did not change, which is the case this
is for. A loader that decoded again, a re-encoded reference or a restarted
server gives new objects: a miss and a fresh encode, never a wrong hit. Equal
content in a new object is a miss on purpose; telling it apart from changed
content would mean reading every frame. Because the references are weak, the
store never keeps the loaded frames alive, and an entry goes when anything it
was computed from goes.

The conditioning keep is reached by every song graph, not only the masked
ones: its key takes a source of None, and then holds the text, the length,
the canvas, the encoder and the references alone.

The price of identity, stated so nobody is surprised: a Masked Source that
executes again (a changed blend or threshold) hands on a new mask tensor, so
a conditioning that used the mask (a motion reference) is encoded again on
that run. The source latent still hits, unless `paint_out` or a late start
put the mask into what is encoded. The mask is not keyed by its content
because `mask_store.py` keeps float values, so its bits are not provably the
whole of it, and hashing the floats reads about a gigabyte per window on
every run. A later step that would close the miss: the Masked Source handing
on the key its mask is kept under.

## What is kept is on the host and is checked for writes

`_to_host` is the line that guarantees no kept tensor holds card memory:
every tensor in a value is moved to the CPU at `put`, whatever device the
caller had it on. (In the song node both are already there: `vae.encode`
returns on core's intermediate device, and so does the text encoder.)

A kept conditioning is handed to the guider of every later run that hits it.
Nothing in core writes into a conditioning tensor today. A tensor's version
counter would be the cheap way to hold that, and it is not there to ask:
nodes run under `torch.inference_mode()`, and an inference tensor raises
"Inference tensors do not track version counter" on `_version` (tried on
2026-10-06). So `put` takes a digest of the value's bytes and `get` compares it: a value
that was written to since it was kept is dropped, logged as a warning,
counted in `written_to`, and encoded again. `get` returns fresh lists and
dicts around the kept tensors, so a caller that appends to what it was given
changes nothing here.

The tensors themselves are shared, not copied, and for the source latent that
reaches further than it looks: the song node's `z.to(device=..., dtype=...)`
returns the kept tensor itself when both already match, so the kept object
travels into that run's latent. What makes that safe is
`audio_freeze.py::MiniMaxH3FreezeAudioWindow`, which clones the video before
it writes the previous window's context in (`video = video.clone()`), and the
late start's multiply, which makes a new tensor. If either ever writes in
place, the digest is the backstop: the next `get` drops the entry and says so.

## Budgets

Each keep has its own byte budget, least recently used out first, and keeps
nothing while the host is short of available memory (the floor is
`reference_encode.STORE_HOST_FLOOR_BYTES`, one copy). Nothing is written to
disk.
"""

from __future__ import annotations

import hashlib
import logging
import weakref
from collections import OrderedDict
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

#: Host memory the source latents may hold, in bytes. Reasoned: a window's video latent is a
#: few tens of MiB by its shape (24 channels at a sixteenth of the canvas, about a hundred
#: latent frames), so this holds a song of a few dozen windows.
LATENT_BYTES = 1 * 1024 ** 3
#: Host memory the conditionings may hold, in bytes. Reasoned: about a hundred MiB per window
#: at the motion graph's sequence length (a few thousand rows of the encoder's width), so a
#: three-window stretch several times over; a whole song turns over, oldest first.
COND_BYTES = 2 * 1024 ** 3


def _host_short() -> bool:
    """Whether the host has less available memory than the reference store's floor."""
    from .reference_encode import STORE_HOST_FLOOR_BYTES, _host_available
    return _host_available() < STORE_HOST_FLOOR_BYTES


def _walk(value, fn: Callable[[torch.Tensor], Any]):
    """`value` with `fn` applied to every tensor in it, in fresh lists, tuples and dicts.
    Anything else is passed through as it is."""
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: _walk(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_walk(v, fn) for v in value)
    return value


def _tensors(value) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    _walk(value, lambda t: out.append(t) or t)
    return out


def _to_host(value, move: Callable[[torch.Tensor], torch.Tensor] = lambda t: t.to("cpu")):
    """`value` with every tensor on the CPU. This is what keeps card memory out of the store:
    a tensor already there is kept as it is, any other goes through `move`."""
    return _walk(value, lambda t: t if t.device.type == "cpu" else move(t))


def digest(value) -> str:
    """A hash of every tensor's shape, dtype and bytes, in the order `_walk` meets them."""
    h = hashlib.blake2b(digest_size=16)
    for t in _tensors(value):
        h.update(f"|{tuple(t.shape)}|{t.dtype}|".encode())
        data = t.detach().contiguous()
        if data.dtype == torch.bfloat16:
            data = data.view(torch.int16)       # numpy has no bfloat16; the bytes are what is hashed
        if data.numel():                        # a view with a zero in its shape cannot be cast
            h.update(memoryview(data.numpy()).cast("B"))
    return h.hexdigest()


class _Entry:
    __slots__ = ("value", "refs", "nbytes", "digest")

    def __init__(self, value, refs, nbytes, value_digest):
        self.value, self.refs, self.nbytes, self.digest = value, refs, nbytes, value_digest


class Keep:
    """Values by (plain values, identity of live objects), least recently used out first."""

    def __init__(self, name: str, budget: int, short: Callable[[], bool] = _host_short):
        self.name, self.budget, self.short = name, int(budget), short
        self.entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self.written_to = 0

    @staticmethod
    def _key(static: tuple, alive: tuple) -> tuple:
        return (static, tuple(id(o) for o in alive))

    def _drop(self, key: tuple) -> None:
        self.entries.pop(key, None)

    def get(self, static: tuple, alive: tuple):
        """The kept value, or None. A hit needs every object in `alive` to be the one the value
        was kept for, and the value's bytes to be what they were then."""
        key = self._key(static, alive)
        entry = self.entries.get(key)
        if entry is None:
            return None
        if len(entry.refs) != len(alive) or any(ref() is not obj for ref, obj in zip(entry.refs, alive)):
            # an id reused by a new object after the old one was collected
            self._drop(key)
            return None
        if digest(entry.value) != entry.digest:
            self.written_to += 1
            self._drop(key)
            logger.warning("[h3-keep] a kept %s was written to after it was kept; dropped, and made again. "
                           "Something downstream changes a tensor it was handed.", self.name)
            return None
        self.entries.move_to_end(key)
        return _walk(entry.value, lambda t: t)

    def put(self, static: tuple, alive: tuple, value) -> bool:
        """Keep `value` on the host under the key. False when it was not kept: the host is
        short of memory, or the value alone is over the budget."""
        value = _to_host(value)
        nbytes = sum(t.numel() * t.element_size() for t in _tensors(value))
        key = self._key(static, alive)
        self._drop(key)
        if nbytes > self.budget or self.short():
            return False
        # A dead object's entry can never hit again; the callback frees it at once.
        refs = tuple(weakref.ref(o, lambda _, key=key: self._drop(key)) for o in alive)
        self.entries[key] = _Entry(value, refs, nbytes, digest(value))
        while self.nbytes() > self.budget:
            self.entries.popitem(last=False)
        return True

    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries.values())

    def clear(self) -> None:
        self.entries.clear()


LATENTS = Keep("source latent", LATENT_BYTES)
CONDS = Keep("conditioning", COND_BYTES)

#: `video_mask.START_NOISE`, `MOTION_NONE`, `ZOOMED`, `WIRED`, `GROW_FIXED` and `GROW_SUBJECT`, restated so this
#: module imports nothing from the pack at load; `bench/check_window_keep.py` holds them to the originals.
START_NOISE = "noise"
MOTION_NONE = "none"
MOTION_ZOOM = "subject only, zoomed in"
MOTION_WIRED = "a video I wire"
MOTION_WIRED_ZOOM = "a video I wire, zoomed in"
ZOOMED = (MOTION_ZOOM, MOTION_WIRED_ZOOM)
WIRED = (MOTION_WIRED, MOTION_WIRED_ZOOM)
BOX_REPLACED = "what is replaced"
GROW_FIXED = "a fixed margin"
GROW_SUBJECT = "the subject's size"


def _vae_dtypes(vae) -> tuple:
    """The dtype the VAE casts pixels to and the dtype its encoder holds. The precision node
    (`vae_precision.py`) changes both on the loaded module, in place."""
    encoder = getattr(getattr(vae, "first_stage_model", None), "encoder", None)
    held = next(iter(encoder.parameters()), None) if encoder is not None else None
    return (str(getattr(vae, "vae_dtype", None)), None if held is None else str(held.dtype))


def latent_key(source: dict, vae, first_frame: int, frames: int, width: int, height: int) -> tuple[tuple, tuple]:
    """The key for one window's source latent, as (plain values, live objects).

    What `video_mask.window` hands the VAE is the loader's frames cut and fitted, and only with
    `paint_out` or a late start does the mask change it; so the mask and its settings are in
    the key only then, and a new mask on an unchanged plate is a hit.
    """
    late = source.get("start_from", START_NOISE) != START_NOISE
    masked = bool(source.get("paint_out")) or late
    # `held_tail` changes the mask on the frames past the source's end, which reaches the encode only where
    # the mask does (`masked`); it is in the key there and nowhere else
    static = ("latent", int(first_frame), int(frames), int(width), int(height), _vae_dtypes(vae),
              source.get("held_tail") if masked else None,
              bool(source.get("paint_out")), source.get("start_from", START_NOISE) if late else None,
              int(source["start_blur"]) if late else None,
              # what the hole is widened by: the cap, the rule, and under the subject's size the feather,
              # which is that margin's floor (`video_mask.margins`); the mask it reads is among `alive`
              (int(source["grow_pixels"]), source.get("grow_by"),
               int(source.get("feather_pixels", 0)) if source.get("grow_by") == GROW_SUBJECT else None)
              if masked else None)
    alive = (source["frames"], vae) + ((source["mask"],) if masked else ())
    return static, alive


def cond_key(clip, text: str, frames: int, width: int, height: int, references, vae, audio_vae,
             source: dict | None, first_frame: int) -> tuple[tuple, tuple]:
    """The key for one window's conditioning, as (plain values, live objects).

    The encoder is in it by identity and by what is patched onto it. Each reference record
    is in it by identity (a tuple cannot be weakly referenced; its frozen records can). With
    a motion reference the window's own frames and mask go in too, since
    `video_mask.motion_reference` is built from them. Zoomed in, so does what frames it: the
    subject's box on each of the window's frames and the shot table's text, which between them
    decide `video_mask.window_boxes`.
    """
    records = tuple(references or ())
    moving, with_source = None, ()
    if source is not None and source.get("motion_reference", MOTION_NONE) != MOTION_NONE:
        framed = None
        # what the reference looks like beyond its choice: the blur, the grey, and which box they and a zoom read
        look = (float(source.get("motion_blur_share", 0.0)), int(source.get("motion_blur_pixels", 0)),
                bool(source.get("motion_grey", False)), source.get("motion_box"), source.get("held_tail"))
        if source["motion_reference"] in ZOOMED or look[0] > 0.0:
            rows = source.get("region_boxes") if look[3] == BOX_REPLACED else source.get("subject_boxes")
            framed = (None if rows is None
                      else tuple(rows[int(first_frame):int(first_frame) + int(frames)].flatten().tolist()),
                      str(source.get("shot_table") or ""))
        # the widening is half of `grow_pixels` under either `grow_by` (`video_mask.motion_widening`), so the
        # choice is not in this key; it is in the latent's above, where the region's margin is
        moving = (source["motion_reference"], int(source["motion_short_edge"]), int(source["grow_pixels"]) // 2,
                  bool(source.get("motion_vae", False)), int(first_frame), framed, look)
        with_source = (source["frames"], source["mask"])
        if source["motion_reference"] in WIRED and source.get("motion_frames") is not None:
            # the wired video is what the reference is cut from: another video on the same source is another
            # conditioning (until 2026-10-10 it was not in the key, and a kept one would have been handed back)
            with_source += (source["motion_frames"],)
    static = ("cond", str(text), int(frames), int(width), int(height),
              str(clip.patcher.patches_uuid), clip.layer_idx, len(records),
              None if vae is None else _vae_dtypes(vae), audio_vae is not None, moving)
    alive = (clip,) + records + tuple(o for o in (vae, audio_vae) if o is not None) + with_source
    return static, alive
