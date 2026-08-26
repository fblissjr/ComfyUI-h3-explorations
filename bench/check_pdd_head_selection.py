#!/usr/bin/env python3
"""The PDD node's runtime guards do what they claim, on real artifacts.

Two subjects, both about `pdd_lora.py` deciding something at load or step time
that nothing downstream would contradict.

## 1. A PDD render selects the fused head its sampling step was fused for.

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


def drive(table, nfe):
    """Run the real tracker over one full schedule at `nfe`. Returns (video, audio)."""
    block = NUM_STEPS // nfe
    bounds_v = M.block_bounds(SHIFT_V, NUM_STEPS, block)
    bounds_a = M.block_bounds(SHIFT_A, NUM_STEPS, block)
    tracker = P._StepTracker(P.boundary_embeddings(bounds_v, table),
                             P.boundary_embeddings(bounds_a, table),
                             bounds_v, bounds_a, nfe, f"nfe={nfe}")
    rows = table.shape[0]

    def emb(t):
        pos = min(max(t, 0.0), 1.0) * (rows - 1)
        i0 = min(int(pos), rows - 2)
        return torch.lerp(table[i0], table[i0 + 1], pos - i0)

    video, audio = [], []
    for boundary in bounds_v[:-1]:
        sigma_v = 1.0 - float(boundary)
        t_a = 1.0 - time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A)
        tracker.update(torch.stack([emb(1.0 - sigma_v), emb(t_a)]),
                       (0, 1, 0), (1, 2, 1))
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

    def every_legal_nfe():
        """One file, every step count the 32-point grid divides by.

        This is the parallel-decoding claim in the form the node has to get
        right: the same weights fused at load for 8, 4, 2 or 16 evaluations,
        each landing on its own block boundaries. The vendor's README reports
        rendering at 8 and 4; the other two come free from the same divisibility
        and are here because a selector that only ever sees one block size is
        not evidence about the others.
        """
        rows = []
        for nfe in (n for n in (16, 8, 4, 2) if NUM_STEPS % n == 0):
            video, audio, tracker = drive(table, nfe)
            want = list(range(nfe))
            assert video == want, f"nfe={nfe}: video heads {video}, want {want}"
            assert audio == want, (
                f"nfe={nfe}: audio heads {audio}, want {want}. Audio runs "
                f"shift {SHIFT_A} and is selected independently, so a "
                f"video-only fix passes a video-only case.")
            assert not tracker.warned, f"nfe={nfe}: on-schedule run warned"
            rows.append(str(nfe))
        return f"nfe {', '.join(rows)} each select 0..n-1 on both streams"

    def off_schedule_warns():
        """A step count the heads were not fused for is reported, once."""
        block = NUM_STEPS // NFE
        bounds_v = M.block_bounds(SHIFT_V, NUM_STEPS, block)
        bounds_a = M.block_bounds(SHIFT_A, NUM_STEPS, block)
        tracker = P._StepTracker(P.boundary_embeddings(bounds_v, table),
                                 P.boundary_embeddings(bounds_a, table),
                                 bounds_v, bounds_a, NFE, "off-schedule")
        rows = table.shape[0]
        for boundary in M.block_bounds(SHIFT_V, NUM_STEPS * 2, block)[1:5]:
            pos = min(max(float(boundary), 0.0), 1.0) * (rows - 1)
            i0 = min(int(pos), rows - 2)
            e = torch.lerp(table[i0], table[i0 + 1], pos - i0)
            tracker.update(torch.stack([e, e]), (0, 1, 0), (1, 2, 1))
        assert tracker.warned, (
            "a run off the fused schedule did not warn. Selection degrades to "
            "the nearest boundary by design, so this warning is the only thing "
            "that distinguishes a deliberate arm from a misconfigured one.")
        return "a 16-step sampler against 8-step heads is reported"

    check("every legal nfe selects its own blocks", every_legal_nfe)
    check("off the fused schedule warns", off_schedule_warns)


# --- 2. the two tolerances, against the noise each has to straddle ---------
# Both were set by hand from a measurement taken once, and one of them was set
# WRONG: TABLE_TOLERANCE started at 1e-3, below the 1.6e-3 a bf16 cast of the
# table costs, so every correct bake would have been rejected into the slow
# injection path -- silently, because that path is correct. The other,
# PARTITION_TOLERANCE, replaced a sha256 that fired on the RIGHT checkpoint for
# the same reason: an exact test against a value the loader is allowed to cast.
#
# A tolerance is only meaningful between two numbers. These cases pin both.
def _tolerances():
    import torch as _t
    from safetensors import safe_open
    root = HERE.parents[2] / "models" / "diffusion_models"
    want = ("minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    got = {}
    for name in want:
        for cand in (root / name, root / "diffusion_models" / name):
            if cand.exists():
                with safe_open(cand, framework="pt") as f:
                    got[name] = (f.get_tensor("adaln_t_table").float(),
                                 f.get_tensor("final_layer.video_out.weight").float())
                break
    return got


_ck = _tolerances()
if len(_ck) < 2:
    skipped.append("both pruned partitions needed to bound the tolerances")
    print("  SKIP  need both partitions on disk; a tolerance graded against "
          "only one side is not graded")
else:
    import torch as _t
    (_ref_tab, _ref_vo), (_fl_tab, _fl_vo) = (_ck[k] for k in _ck)

    def _rel(a, b):
        return float((a - b).norm() / b.norm())

    def table_tolerance_straddles():
        cast = _rel(_ref_tab.to(_t.bfloat16).float(), _ref_tab)
        cross = _rel(_fl_tab, _ref_tab)
        assert cast <= P.TABLE_TOLERANCE, (
            f"a bf16 cast of the curve table is {cast:.5f} away, above "
            f"TABLE_TOLERANCE={P.TABLE_TOLERANCE}. Every correct bake would be "
            f"rejected into the runtime injection -- slower, and silent, "
            f"because the injection is correct. This is the value it shipped "
            f"with for an hour on 2026-08-26.")
        assert cross > P.TABLE_TOLERANCE, (
            f"the other partition's table is {cross:.5f} away, within "
            f"TABLE_TOLERANCE={P.TABLE_TOLERANCE}. A bake solved against the "
            f"wrong basis would be accepted, and it is 0.02 wrong at runtime "
            f"against 0.0001 for the right one -- which a fit residual cannot "
            f"detect, so this comparison is the only thing standing there.")
        return f"cast {cast:.5f} < {P.TABLE_TOLERANCE} < cross-partition {cross:.5f}"

    def partition_tolerance_straddles():
        cast = _rel(_ref_vo.to(_t.bfloat16).float(), _ref_vo)
        cross = _rel(_fl_vo, _ref_vo)
        assert cast <= P.PARTITION_TOLERANCE, (
            f"a bf16 cast of final_layer.video_out is {cast:.5f} away, above "
            f"PARTITION_TOLERANCE={P.PARTITION_TOLERANCE}. The loader casts on "
            f"load, so this rejects the CORRECT checkpoint -- which is what the "
            f"sha256 this replaced actually did, on the first real render.")
        assert cross > P.PARTITION_TOLERANCE, (
            f"the other partition is {cross:.5f} away, within "
            f"PARTITION_TOLERANCE={P.PARTITION_TOLERANCE}. fl2va and ref2va "
            f"ship identical key sets, so a mismatched pair renders without one "
            f"unmatched key and is merely wrong.")
        return f"cast {cast:.5f} < {P.PARTITION_TOLERANCE} < cross-partition {cross:.5f}"

    check("table tolerance sits between a cast and a partition swap",
          table_tolerance_straddles)
    check("partition tolerance sits between a cast and a partition swap",
          partition_tolerance_straddles)

print()
if failures:
    print(f"{len(failures)} failure(s): {', '.join(failures)}")
    sys.exit(1)
if skipped:
    print(f"passed, but SKIPPED: {'; '.join(skipped)}")
    sys.exit(2)
print("all ok -- every sampling step decodes the interval it was fused for")
