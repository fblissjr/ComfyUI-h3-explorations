# How every violation arm under `bench/` is graded, 2026-08-25

**Status:** Dated scoping record. Classification only — no code was changed and
no arm was reworded.
**Question:** three controls failed silently on 2026-08-25 by reporting a null
or unreached result as a pass. Do the repo's older violation arms share that
defect?
**Answer:** **no, and the premise for reworking them does not hold.** Details
below, including the one residual weakness that is real and shared.

**Evidence class.** Every classification below is **source-read**, not
executed, with one exception: `bench/red/spine_control.py` was **executed** and
passed, because nine harnesses inherit their grading from the spine it controls
and reading that file is not the same as watching it work.

## What counts as a violation arm here

Code that deliberately breaks something and asserts the subject notices. That
is the population the question is about. Excluded: assertions over the live
repo state, parity comparisons, and benchmarks, none of which inject a defect.

## The classes

- **(a) exit code of the whole check** — the mutation is graded by whether some
  process or check returned non-zero. Free-passing when the subject already
  carries a finding, because then it returns non-zero regardless.
- **(b) baseline-relative** — the mutated verdict is compared against the
  unmutated one. Splits into **(b1) whole-check baseline** and **(b2) per-arm
  baseline**.
- **(c) specificity-asserted** — the mutation must produce a *named* effect: a
  named field moving, a rejection reason naming the offending file, a key
  landing in a named result bucket, a measured value moving. Strictly stronger
  than (b), because an unrelated finding cannot satisfy it.

## Counts

| class | arms | cases |
|---|---:|---:|
| (a) exit code, free-pass condition **open** | 0 | 0 |
| (a) exit code, free-pass condition **closed by bidirectional cases** | 1 | 2 |
| (b1) whole-check baseline | 9 | 98 |
| (b2) per-arm baseline | 2 | 21 |
| (c) specificity-asserted | 8 | — |

## (a) — one arm, and it is not free-passing

`bench/red/spine_control.py` is the only place a control is graded on a process
exit code, and it is the right place: its subject *is* a harness, whose exit
code is its verdict. The free-pass condition is closed by construction, because
it asserts **both** directions — an inert mutation must produce exit 1, a
healthy harness must produce exit 0. A subject broken for any unrelated reason
fails the healthy case. Executed 2026-08-25: both cases pass.

## (b1) — the nine spine harnesses

Every `bench/red/show_red_*.py` runs on `bench/red/harness.py`, which already
implements what today's failures cost us the hard way. A case carries a *kind*,
not an authored verdict: a `MUTATION` must move the verdict away from the
unmutated baseline, a `NEAR_MISS` must leave it. An exception is `ERRORED`,
never counted as a difference — so a typo in a mutation cannot read as proof
the check works.

Two properties are worth naming because they are better than the standard
today's rule asks for:

- **It fails closed on a red baseline.** If the subject is already failing, a
  mutation that adds another finding leaves the verdict RED, which equals the
  baseline, which the harness reports as WRONG. Exit-code grading would have
  called that a pass. This is the exact inversion of the trap.
- **The needle check is free.** "A mutation that never reached the subject
  leaves the verdict unchanged" is already what `MUTATION` asserts, so an
  inert mutation is caught without a separate assertion.

## (b2) — the two per-arm harnesses

`review_v2_calibration_bundle.py` (12 mutations) and `check_marker_corpus.py`
(9) grade each mutation on **the arm it targets** gaining a problem the
unmutated baseline did not have. Both were written today, the first after its
exit-code version passed every mutation while noticing none of them.

## (c) — specificity-asserted, the strongest group

These do not ask "did something change". They name what must change:

| arm | what the mutation must produce |
|---|---|
| `check_native_h3_presentation.py` | a **named record field** per mutation (`MUTATION_PLAN`), plus invariant fields byte-identical on rows the mutation cannot reach |
| `prove_calibration_seam.py` | detection **and** non-detection: mutations in `SEAM_BLIND` must leave the chain intact, because their defect lives outside the batch |
| `check_pool_media_integrity.py` | a rejection reason **naming the offending file**, plus a clean control that must stay green |
| `check_calibration_model_mapping.py` | the mutated key in the **correct result bucket** — missing, unexpected, shape, dtype |
| `check_calibration_precision_policy.py` | a **measured delta**: the corrupt tap must move the tower output |
| `check_h3_awq_encoder.py` | a failed resolution **and** proof the widening actually widened something |
| `check_lora_alpha.py` | a synthetic unsafe file rejected **and** a baked near-miss accepted |
| `check_model_files.py` | five named controls with per-control expected verdicts, run every invocation |

`check_distill_grid.py` belongs here indirectly: `grade_arm` is a pure
collector that `show_red_distill_grid.py` drives with synthetic arms, so the
mutation reaches the same code a real graph does rather than a reporter.

## Why the rework this was scoping is not earned

The repo's rule is that no new check lands until a drift instance appears that
the existing gates provably could not have caught. That cuts both ways, and
here it cuts against the rework. All three of today's instances were in code
that the older arms' grading had never governed:

1. `review_v2_calibration_bundle.py`'s first violation arm — **written today**,
   fixed before it was committed.
2. Its `timestamp-shift` mutation, written against decoded text where the
   record holds `U+0120` — **written today**.
3. `pilot_sequential_feasibility.py`'s modifier-entered control — existing code,
   but the defect was **a branch that had only ever met the CPU tier**, not its
   grading. It grades on a value change, which is class (c).

**A correction to how that third instance has been described**, including by
me: it did not report a null result as a pass. Its guard is
`weight_before is not None and weight_before != weight_after`, so a null read
yields "unchanged", which **refuses** to emit the candidate. It failed closed.
The shape it shares with the other two is real — a reading whose precondition
was not met produced a verdict anyway — but the direction is the opposite, and
a rework programme justified by "these controls pass when they should fail"
would be built on a mis-stated instance.

## The residual weakness that is real

**(b1) grades on the whole-check verdict, so a mutation aimed at arm X is
satisfied by arm Y firing.** That is not a free pass — the verdict genuinely
moved, and a defect genuinely exists — but it is weaker than (c), which the
same repo already demonstrates is achievable. A mutation could be mis-attributed
without anything saying so.

Nothing observed today is an instance of this, so by the repo's own bar it does
not yet earn a rework. It is worth recording as the known edge of the (b1)
guarantee, and worth reaching for (c) in **new** arms, which costs nothing at
authoring time and is what the strongest eight already do.
