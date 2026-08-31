# AWQ v2 calibration candidate pool

**Status:** Accepted candidate-pool evidence; not a calibration manifest or run authorization
**Observation date:** 2026-08-24
**Source revision:** `StellarVoyager/H3-IR` at
`460db3256f19dc70d0def2068a22e6e0dca87e8e`

## Decision boundary

The current rights-clean AWQ v2 source is H3-IR only. Its 1,110 rows partition
into a 1,028-row vision-bearing candidate pool and 82 text-only exclusions.
The excluded rows remain available as a T2VA regression reservoir; they cannot
join the same sequential `oneshot` trace as the vision-bearing population.

This file accepts the source inventory, role partition, overlay measurements,
and exact-media component constraint. It does **not** choose the absolute
calibration population. That count follows from the quantization lane's measured
RTX 4090 feasibility pilot.

## Mutually exclusive row roles

**MEASURED.** The accepted pool builder assigns picture roles per picture before
summarizing each row. This is necessary because 40 requests contain both a
keyframe and ordinary reference pictures, which require different upstream
geometry.

| Primary row role | Rows | Pool share |
|---|---:|---:|
| multi-image 2--3 | 520 | 50.6% |
| multi-image 4--9 | 278 | 27.0% |
| keyframe-only | 91 | 8.9% |
| single-image | 79 | 7.7% |
| keyframe-plus-reference | 40 | 3.9% |
| video-reference | 20 | 1.9% |

The per-picture inventory contains 141 keyframe pictures and 2,960 ordinary
reference pictures. Keyframes use target-canvas geometry; ordinary reference
stills retain reference geometry. A row-wide geometry field is invalid for the
mixed role.

## Overlay properties

Overlays are coverage properties, not additional buckets:

| Overlay | Rows | Pool share |
|---|---:|---:|
| dialogue markers | 505 | 49.1% |
| wide/tall input | 341 | 33.2% |
| audio label | 120 | 11.7% |
| small-source input | 7 | 0.7% |

The seven small-source rows occupy six independent exact-media components.
At least two of those components must be reserved for holdout.

## Split constraint

**MEASURED.** Connected components over individual declared media SHA-256s
reduce the 1,028 rows to 410 indivisible components. Of these, 154 contain more
than one row, covering 772 rows; the largest component contains 76 rows.

Calibration and holdout must therefore be assigned by whole component. A
splitter must place the largest components first, treat video coverage as a
constrained subproblem, and report achieved role and overlay shares rather than
asserting exact targets. Near-duplicate media review and full local hash
verification remain acceptance checks for the eventual split.

## Rights and excluded sources

- **Accepted:** H3-IR, declared `cc0-1.0` at dataset and row level with
  `redistribution_allowed: true`.
- **Excluded pending rights:** `marcuskwan/canvas-preview30` has no declared
  license or README.
- **Fixture only pending terms:**
  `consciousengines/h3-video-edit-showcase` declares `license: other`.
- **Not calibration inputs:** avatar_500 is a prompt-template reference;
  Malcolmrey files are generated outputs; the inspected JAV set is T2VA and
  declares Sora-2 provenance.

Candidate media without H3 IR may be evaluated later under a separately
approved authoring plan. It is not part of this accepted pool.

## Provenance

Producer:
[`bench/build_h3_calibration_pool.py`](../../../../bench/build_h3_calibration_pool.py),
SHA-256 `2b609c96114791837c075eea2252be252eb01c22c7245d68eb2ee8450fbbae1a`.

Outputs:

- [`2026-08-24_h3_calibration_pool.jsonl`](../../../../bench/results/archive/v2_encoder/2026-08-24_h3_calibration_pool.jsonl),
  SHA-256 `00febc71610f53a2336020bc5157c6377b46b752922a573a6653458892701887`;
- [`2026-08-24_h3_calibration_pool_excluded.jsonl`](../../../../bench/results/archive/v2_encoder/2026-08-24_h3_calibration_pool_excluded.jsonl),
  SHA-256 `c7473fa6bf5113210a2732910bf1f78b5c40444c69ae10d390bb8383101bc86a`;
- [`2026-08-24_h3_calibration_pool_summary.json`](../../../../bench/results/archive/v2_encoder/2026-08-24_h3_calibration_pool_summary.json),
  SHA-256 `a661cba1fdf8f349be923adb290a1100edaa4286f073a62eb1664aeadc3a39b0`.

**MEASURED, repaired after an escaped video defect.** The builder now opens and
SHA-256-verifies every declared image and video against the pinned snapshot
before a row can enter the pool. Each output row carries its per-file
`media_verification` record. The rebuilt pool contains 3,121 verified media
references across 1,651 distinct files; no media-kind or primary-role exemption
remains. [`check_pool_media_integrity.py`](../../../../bench/check_pool_media_integrity.py)
re-verifies the emitted artifacts and demonstrates red controls for a deleted
file, a corrupted file, a missing declared hash, and a mismatched video file.

The eventual component split and capture preflight must recompute the same
3,121 declarations from the pinned snapshot. A declared digest copied from a
source row is not sufficient evidence by itself.
