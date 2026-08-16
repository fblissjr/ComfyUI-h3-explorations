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
  duration  modular_pipelines/minimax_h3/before_denoise.py. The
            ceiling is checked AFTER the frame count snaps, with a comment
            naming the trap: 346 frames passes a pre-snap check and then
            rounds to 362, i.e. 15.083s.
  grid      align_num_frames, modular_pipeline.py -- frames must be
            17n + 5 for the video VAE to encode them.

ComfyUI enforces only the grid, and only by snapping. Its node accepts
`length` up to 3600 (`comfy_extras/nodes_minimax_h3.py`), i.e. 150 seconds,
with no ceiling at all, and `adapt_canvas` resolves a canvas for any aspect.

**362 is TRAINED and this file's ceiling does not say otherwise. Corrected
2026-08-14 and the correction matters more than the rule.** `MAX_DURATION`
below is the reference *pipeline's* `max_duration`, a hard-coded 15.0 in
`modular_pipeline.py`. 362 frames is 15.083s, so the reference pipeline does
refuse it -- that transcription is accurate and was checked at the source.

What was wrong was the inference drawn from it. This docstring said the
refusal made 362 "illegal" and the repo's long preset out of distribution.
**362 is the longest length H3 was trained on** (upstream, 2026-08-14). The
15.0 is a round number in a spec that lands one grid step below the real
training maximum, not a statement about the model. There is no on-grid count
at exactly 15.0s, which is how the gap appears.

So `duration_in_range` answers "would the reference pipeline emit this",
which is worth knowing and is NOT "is this in distribution". Do not treat a
False as a reason to avoid a length, and do not let a checker built on it
report a valid render as a problem -- a check that goes red while the state
is correct is worse than no check.
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
MAX_DURATION = 15.0

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
    """Largest on-grid frame count inside the duration ceiling."""
    n = math.floor((MAX_DURATION * FPS - FRAME_REMAINDER) / FRAME_FACTOR)
    return n * FRAME_FACTOR + FRAME_REMAINDER


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
