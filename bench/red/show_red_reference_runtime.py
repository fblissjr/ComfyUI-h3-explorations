"""Show the typed-reference runtime acceptance check red.

Mutations are temporary in-memory replacements; no installed ComfyUI or repo
file is edited. Each case removes one implementation property, and the
near-misses prove the restored subject remains green.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from harness import Harness, MUTATION, NEAR_MISS, main  # noqa: E402
import check_reference_runtime as C  # noqa: E402

R = C.R


@contextmanager
def _patched(name, replacement):
    original = getattr(R, name)
    setattr(R, name, replacement(original))
    try:
        yield
    finally:
        setattr(R, name, original)


def _under(name, replacement, verdict):
    def run():
        with _patched(name, replacement):
            try:
                verdict()
            except AssertionError:
                return False
            return True
    return run


def build():
    h = Harness(
        subject="bench/check_reference_runtime.py",
        verdict=lambda holds: not holds,
    )
    h.baseline(C.all_hold)

    h.case(
        "M1 loaded_fps is ignored and 30 fps frames pass through",
        MUTATION,
        _under(
            "_resample_video_to_24fps",
            lambda _original: lambda frames, _fps: frames,
            C.video_is_normalized_to_24fps,
        ),
    )
    h.case(
        "M2 mono upmix and duration trim are both skipped",
        MUTATION,
        _under(
            "_prepare_audio",
            lambda _original: lambda audio, _duration, _field: audio,
            C.audio_is_stereo_and_target_bounded,
        ),
    )
    h.case(
        "M3 VHS ownership accepts metadata for another width",
        MUTATION,
        _under(
            "_loaded_fps",
            lambda _original: (
                lambda video_info, _count, _height, _width:
                float(video_info["loaded_fps"])
            ),
            C.video_metadata_is_owned,
        ),
    )
    h.case(
        "M4 compiler reverses the user's record order",
        MUTATION,
        _under(
            "_compile_reference_records",
            lambda original: (
                lambda records, *args, **kwargs:
                original(tuple(reversed(records)), *args, **kwargs)
            ),
            C.compiler_preserves_one_order_for_both_lists,
        ),
    )
    h.case(
        "M5 release Qwen policy reuses the VAE frames",
        MUTATION,
        _under(
            "_release_qwen_video_frames",
            lambda _original: lambda frames: frames,
            C.release_video_policy_is_opt_in_and_two_stage,
        ),
    )
    h.case(
        "M6 encoder policy reads release config instead of encoder config",
        MUTATION,
        _under(
            "_qwen_video_settings",
            lambda original: (
                lambda _policy: original("release")
            ),
            C.encoder_policy_reads_encoder_config,
        ),
    )

    h.case(
        "G1 restored: 24 fps normalization",
        NEAR_MISS,
        lambda: _holds(C.video_is_normalized_to_24fps),
    )
    h.case(
        "G2 restored: audio normalization",
        NEAR_MISS,
        lambda: _holds(C.audio_is_stereo_and_target_bounded),
    )
    h.case(
        "G3 restored: metadata ownership",
        NEAR_MISS,
        lambda: _holds(C.video_metadata_is_owned),
    )
    h.case(
        "G4 restored: shared compiler order",
        NEAR_MISS,
        lambda: _holds(C.compiler_preserves_one_order_for_both_lists),
    )
    h.case(
        "G5 restored: distinct release Qwen view",
        NEAR_MISS,
        lambda: _holds(C.release_video_policy_is_opt_in_and_two_stage),
    )
    h.case(
        "G6 restored: encoder config owns encoder policy",
        NEAR_MISS,
        lambda: _holds(C.encoder_policy_reads_encoder_config),
    )
    return h


def _holds(verdict):
    try:
        verdict()
    except AssertionError:
        return False
    return True


if __name__ == "__main__":
    main(build)
