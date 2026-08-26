#!/usr/bin/env python3
"""A PDD render selects the fused head its sampling step was fused for.

## The escaped defect that earns this check

On 2026-08-26 four PDD arms rendered at 1344x768 with every gate green --
`check_distill_settings` graded the shift and step count, `check_lora_alpha`
graded the scale, `check_attention_defaults` graded the chain, the node logged
208 backbone modules patched and 50 adaln re-injected, and its own
boundary-residual warning stayed silent through all four. The clips were
wrong anyway.

`_StepTracker` recovers the sampling time by a nearest-row lookup against the
1025-row `adaln_t_table`, which quantises `t` to about 1e-3. A `t` sitting
exactly ON a block boundary comes back a fraction BELOW it, and the interval
membership that followed then returned the previous block. Measured against
the real table and the real 8-step shift-12 schedule, the heads came out

    video  [0, 0, 2, 3, 4, 5, 6, 6]   instead of   [0, 1, 2, 3, 4, 5, 6, 7]
    audio  [0, 1, 1, 3, 4, 5, 6, 7]

Wrong at step 1 and at step 7 -- step 7 being the largest jump in this
schedule and the place the fused heads differ most from each other.

**Nothing could have caught it.** The recovered `t` was correct to 4e-5, so
the residual warning is silent by construction; the render completes; the
output is merely wrong. It is the shape `docs/checks.md` calls a
silent-success, and the only thing that found it was driving the tracker with
real inputs instead of reading the code.

## What this asserts, i.e. what breaks if a case is deleted

  real schedule        the tracker, the REAL `adaln_t_table` read off a
                       shipped checkpoint, and the real 8-step boundaries
                       select heads 0..nfe-1 in order, for video AND audio.
                       Audio matters separately: it runs its own shift, so a
                       fix that only lines the video stream up would pass a
                       video-only case
  the old bug is red   the same drive with snapping disabled reproduces the
                       escaped selection and FAILS. Without this the check
                       could go green on a tracker that had quietly lost the
                       snap and gone back to pure membership, which is exactly
                       the regression it exists to stop
  off schedule warns   a run at a step count the file was not fused for sets
                       `warned`. This is the OTHER half: the snap tolerance
                       and the residual tolerance must agree about what "on
                       schedule" means, or one regime silently borrows the
                       other's behaviour
  membership survives  `step_for_t` with no snap still does plain interval
                       membership. The fix added a branch; this says it did
                       not replace the fallback the off-schedule path needs

Needs a checkpoint on disk and ComfyUI importable for `safetensors`. No CUDA,
no server, no model load -- it reads one small buffer out of a header.

Exit codes: 0 all cases passed, 1 a case failed, 2 passed but a control was
skipped (no checkpoint on disk to read a real table from).

    python bench/check_pdd_head_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # ComfyUI root
sys.path.insert(0, str(HERE.parent))              # this repo

import pdd_math as M                              # noqa: E402
import pdd_lora as P                              # noqa: E402

#: Any pruned checkpoint carries the curve table; they are what a PDD arm runs
#: against. Named rather than globbed so a missing file SKIPS loudly instead of
#: quietly grading nothing.
CHECKPOINTS = ("minimax_h3_ref2va_pruned_int8_convrot.safetensors",
               "minimax_h3_fl2va_pruned_int8_convrot.safetensors")

failures: list[str] = []
skipped: list[str] = []


def check(name, fn):
    try:
        detail = fn()
    except AssertionError as exc:
        failures.append(name)
        print(f"  FAIL  {name}: {exc}")
    else:
        print(f"  ok    {name}" + (f"   {detail}" if detail else ""))


def real_table():
    """`adaln_t_table` from a checkpoint on disk, or None."""
    from safetensors import safe_open
    root = HERE.parents[2] / "models" / "diffusion_models"
    for name in CHECKPOINTS:
        for candidate in (root / name, root / "diffusion_models" / name):
            if not candidate.exists():
                continue
            with safe_open(candidate, framework="pt") as f:
                if "adaln_t_table" in set(f.keys()):
                    return f.get_tensor("adaln_t_table").float(), name
    return None, None


def time_shift_sigma(sigma, from_shift, to_shift):
    """`comfy/ldm/minimax/model.py::time_shift_sigma`, restated.

    The audio stream's time is the video sigma carried to the audio shift, so
    the audio boundaries only line up if this matches upstream. Restated rather
    than imported because importing the model module pulls in CUDA-adjacent
    machinery this check deliberately does without.
    """
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def drive(table, bounds_v, bounds_a, nfe, snap_on=True):
    """Run the tracker over one full sampling schedule. Returns (video, audio)."""
    tracker = P._StepTracker(table, bounds_v, bounds_a, nfe, "check")
    if not snap_on:
        # Reproduce the pre-fix selector without editing the module: pure
        # interval membership, which is what `snap=0.0` still means.
        def update(t_emb, video_seg, audio_seg, _t=tracker):
            tv = _t._t(t_emb, video_seg[2])
            ta = _t._t(t_emb, audio_seg[2])
            _t.video = M.step_for_t(tv, _t.bounds_v, _t.nfe)
            _t.audio = M.step_for_t(ta, _t.bounds_a, _t.nfe)
        tracker.update = update

    rows = table.shape[0]

    def emb(t):
        pos = min(max(t, 0.0), 1.0) * (rows - 1)
        i0 = min(int(pos), rows - 2)
        return torch.lerp(table[i0], table[i0 + 1], pos - i0)

    video, audio = [], []
    for boundary in bounds_v[:-1]:
        sigma_v = 1.0 - float(boundary)
        t_v = 1.0 - sigma_v
        t_a = 1.0 - time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A)
        tracker.update(torch.stack([emb(t_v), emb(t_a)]), (0, 1, 0), (1, 2, 1))
        video.append(tracker.video)
        audio.append(tracker.audio)
    return video, audio, tracker


SHIFT_V, SHIFT_A = 12.0, 3.0
NUM_STEPS, BLOCK = 32, 4
NFE = NUM_STEPS // BLOCK

print("PDD fused-head selection over a real schedule")

table, source = real_table()
if table is None:
    skipped.append("no checkpoint with an adaln_t_table on disk")
    print("  SKIP  no pruned H3 checkpoint found; the real table is the whole "
          "point of this check and a synthetic one would grade itself")
else:
    bounds_v = M.block_bounds(SHIFT_V, NUM_STEPS, BLOCK)
    bounds_a = M.block_bounds(SHIFT_A, NUM_STEPS, BLOCK)
    want = list(range(NFE))

    def real_schedule():
        video, audio, tracker = drive(table, bounds_v, bounds_a, NFE)
        assert video == want, (
            f"video heads {video}, want {want}. A step that decodes another "
            f"block's interval is silent: the residual stays ~0 and the render "
            f"completes.")
        assert audio == want, (
            f"audio heads {audio}, want {want}. Audio runs shift {SHIFT_A} and "
            f"is selected independently, so a video-only fix passes without it.")
        assert not tracker.warned, "on-schedule run raised the off-schedule warning"
        return f"video and audio both {want[0]}..{want[-1]}, from {source}"

    def old_bug_is_red():
        video, audio, _ = drive(table, bounds_v, bounds_a, NFE, snap_on=False)
        assert video != want or audio != want, (
            "pure interval membership selected the right heads, so this check "
            "would pass a tracker that had lost its boundary snap. Either the "
            "table stopped quantising or the drive is not reaching the "
            "selector -- both make the case above meaningless.")
        return f"membership alone gives video {video}"

    def off_schedule_warns():
        wrong = M.block_bounds(SHIFT_V, NUM_STEPS * 2, BLOCK)
        tracker = P._StepTracker(table, bounds_v, bounds_a, NFE, "off-schedule")
        rows = table.shape[0]
        for boundary in wrong[1:5]:
            pos = min(max(float(boundary), 0.0), 1.0) * (rows - 1)
            i0 = min(int(pos), rows - 2)
            e = torch.lerp(table[i0], table[i0 + 1], pos - i0)
            tracker.update(torch.stack([e, e]), (0, 1, 0), (1, 2, 1))
        assert tracker.warned, (
            "a run off the fused schedule did not warn. The snap tolerance and "
            "the residual tolerance must agree on what 'on schedule' means, or "
            "snapping quietly absorbs the case the warning exists for.")
        return "a 16-step grid against 8-step heads is reported"

    check("real schedule selects every head in order", real_schedule)
    check("the escaped selection is still red without snapping", old_bug_is_red)
    check("off the fused schedule warns", off_schedule_warns)


def membership_survives():
    bounds = M.block_bounds(SHIFT_V, NUM_STEPS, BLOCK)
    got = [M.step_for_t(t, bounds, NFE) for t in (0.0, 0.005, 0.2, 0.37, 1.0)]
    assert got == [0, 0, 6, 7, 7], (
        f"plain membership returned {got}; the off-schedule fallback moved")
    return "unsnapped membership unchanged"


check("interval membership survives the fix", membership_survives)

print()
if failures:
    print(f"{len(failures)} failure(s): {', '.join(failures)}")
    sys.exit(1)
if skipped:
    print(f"passed, but SKIPPED: {'; '.join(skipped)}")
    sys.exit(2)
print("all ok -- every sampling step decodes the interval it was fused for")
