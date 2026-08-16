"""The reference pipeline's input rules, in one place.

MiniMax H3 has hard limits that ComfyUI does not enforce. Each one is a
render that succeeds and is wrong -- the failure mode this repo exists to
convert into something visible. They live here rather than inline so the
constants cannot drift between the nodes, the bench and the workflow
generator, which is the same reason `workflows/h3_config.py` exists.

Sources, all in `coderef/diffusers` (the reference implementation):

  aspect    modular_pipelines/minimax_h3/modular_pipeline.py defines
            MINIMAX_H3_MIN/MAX_ASPECT_RATIO; resolve_canvas_size raises
            outside them.
  duration  the FLOOR only, modular_pipelines/minimax_h3/before_denoise.py.
            The ceiling here is the model's, not the reference's -- see below.
  grid      align_num_frames, modular_pipeline.py -- frames must be
            17n + 5 for the video VAE to encode them.

ComfyUI enforces only the grid, and only by snapping. Its node accepts
`length` up to 3600 (`comfy_extras/nodes_minimax_h3.py`), i.e. 150 seconds,
with no ceiling at all, and `adapt_canvas` resolves a canvas for any aspect.

**The ceiling is 362 frames, and it is the model's, not the reference
pipeline's. Owner decision, 2026-08-16.** Every "345 is the largest legal
count" claim this repo used to carry is withdrawn -- 345 was the largest
count *diffusers* will emit, which is a fact about diffusers.

`MAX_LENGTH = 362` is the longest length H3 was trained on. Know what that
rests on before quoting it: **one upstream statement recorded on 2026-08-14
(`6e85e48`) with no artifact attached**, plus one third-party config that
ships it (LightX2V's `minimax_h3_t2av.json`, 362 frames). MiniMax's own
README gives a rounded "4-15 seconds", which neither confirms it nor refutes
it -- a product spec is not a training bound -- and the official checkpoint
configs are silent: no max frame count, no `max_position_embeddings`, RoPE is
theta-based. The owner weighed that and chose 362. It is a decision made on
thin evidence, recorded as such, rather than a measurement.

The reference pipeline refuses 362 (its `max_duration` is a hard-coded 15.0
and 362 is 15.083s; verified at the source, twice). That is now a
portability note and nothing more: ask `reference_would_emit()` if you care
whether a graph also runs in diffusers, and do not call the answer legality.
"""

from __future__ import annotations

import math

# The aspect range the released checkpoint was trained over.
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4

# H3 generates at a fixed 24 fps; everything it conditions on is resampled
# onto that, so duration and frame count are the same statement.
FPS = 24
MIN_DURATION = 5.0

# The ceiling, in frames rather than seconds, because the grid is what the
# model actually constrains and seconds are the derived quantity. Deriving
# MAX_DURATION from it is what keeps `duration_in_range(362)` true: no
# on-grid count lands on a round 15.0s, so a seconds-first ceiling always
# excludes its own maximum by 0.083s.
MAX_LENGTH = 362
MAX_DURATION = MAX_LENGTH / FPS  # 15.083s

# diffusers' hard-coded `max_duration`. Kept, and kept SEPARATE, for the one
# question it answers: would a graph exported from here also run in the
# reference pipeline. That is portability, not legality -- see the docstring.
REFERENCE_MAX_DURATION = 15.0

# The video VAE encodes 17 pixel frames per chunk and keeps 5 latents.
FRAME_FACTOR = 17
FRAME_REMAINDER = 5

CANVAS_MULTIPLE = 32

# One frame is the single length that is not a video, and it is the only
# exception to the 17n+5 grid. It exists because H3 is a capable single-image
# edit model at exactly one frame; see `single_frame.py`, which lifts core's
# floor to reach it, and note that ComfyUI's own VAE already carries a `t == 1`
# branch (`comfy/ldm/minimax/vae.py`) -- this is a mode the stack anticipated,
# not one bolted on here.
#
# **Every duration rule below is about VIDEO and none of them applies to it.**
# A single frame is 0.042s, so a naive reading of the 5-15s window calls it
# illegal, which is how a correct render ends up with a warning printed over
# it. Ask `is_single_frame()` before asking anything about duration.
SINGLE_FRAME = 1


def is_single_frame(length):
    """True for the one length that is an image rather than a clip."""
    return int(length) <= SINGLE_FRAME


def aspect_in_range(width, height):
    return MIN_ASPECT_RATIO <= (width / height) <= MAX_ASPECT_RATIO


def describe_aspect_range():
    return f"1:{1 / MIN_ASPECT_RATIO:g} to {MAX_ASPECT_RATIO:g}:1"


def snap_length(length):
    """Round a frame count up to the next `17n + 5` the video VAE can encode.

    Written as a loop rather than a closed form on purpose. DiffSynth shipped
    the closed form `(n + f - 1) // f * f + r`, which overshoots by a whole
    17-frame chunk whenever `n % 17` is in 1..4 -- it returns 39 for a request
    of 20 where 22 is correct -- and had to fix it in `0e772dd`. ComfyUI uses
    the loop and is right. The arithmetic saved is not worth re-deriving a bug
    someone else already paid for.
    """
    length = int(length)
    # Matches core exactly, which is the whole contract of this function:
    # `temporal_shape` clamps with `max(1, length)` and `align_frame_count`
    # returns 1 unchanged, so anything at or below 1 is one frame and not a
    # 5-frame clip. Before the single-frame path existed core clamped at 5 and
    # so did this; the two moved together, deliberately.
    if length <= SINGLE_FRAME:
        return SINGLE_FRAME
    if length < FRAME_REMAINDER:
        return FRAME_REMAINDER
    while length % FRAME_FACTOR != FRAME_REMAINDER:
        length += 1
    return length


def duration_of(length):
    return length / FPS


def duration_in_range(length):
    """True if the SNAPPED frame count is inside the trained duration window.

    Takes the raw count and snaps first, because that is the order the
    reference checks in and the order that catches the 346 -> 362 case. A
    caller who checks the request rather than the result gets a pass on a
    length the model will not run.

    **False at length=1 means "not a video", not "illegal".** One frame is
    0.042s and falls outside the window trivially. Callers that refuse or warn
    on a False must ask `is_single_frame()` first, or they report a correct
    single-image render as a problem -- which is worse than not checking.
    """
    return MIN_DURATION <= duration_of(snap_length(length)) <= MAX_DURATION


def max_legal_length():
    """Longest length H3 was trained on: 362 frames, 15.083s.

    Returns `MAX_LENGTH` rather than deriving it, because deriving a frame
    count from a seconds ceiling is what produced the 345 answer this repo
    shipped for a week.
    """
    return MAX_LENGTH


def reference_would_emit(length):
    """True if diffusers' pipeline would also emit this length.

    The portability question, and the ONLY thing the reference's 15.0s
    ceiling is evidence about. A False here means a graph exported from this
    repo will not run unmodified in diffusers -- it does not mean the model
    cannot generate it. 362 is the case that separates the two.
    """
    return MIN_DURATION <= duration_of(snap_length(length)) <= REFERENCE_MAX_DURATION


def min_legal_length():
    return snap_length(math.ceil(MIN_DURATION * FPS))


def describe_length(length):
    """One line for a log or an error, showing the snap when there is one."""
    snapped = snap_length(length)
    if is_single_frame(snapped):
        # Never "1 frames (0.042s at 24fps)". That reads as a broken video and
        # sends the reader looking for the bug, when it is the mode they asked
        # for. Say what it is.
        return "1 frame (single image, not a clip)"
    if snapped == length:
        return f"{length} frames ({duration_of(snapped):.3f}s at {FPS}fps)"
    return (f"{length} -> {snapped} frames "
            f"({duration_of(snapped):.3f}s at {FPS}fps)")
