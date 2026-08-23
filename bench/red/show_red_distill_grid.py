"""Show `bench/check_distill_grid.py` red, and green where firing would be wrong.

Drives the two COLLECTORS -- `grade_arm` and `grade_published` -- rather than
the reporter. A harness feeding only the reporter would pass a grader that
returned its own baseline, which is the hole
`show_red_check_graph_discovery.py` records.

The mutations come in two families, because the check has two independent
sources and losing either one differently:

  * arm mutations move what WE run (the scheduler, the shift, the step count)
    off a grid the vendor pins;
  * publication mutations move what the VENDOR says, which must be caught even
    though our graphs did not change -- otherwise a README edit silently
    redefines the target and every graph case stays green against it.

The NEAR_MISS cases are the ones worth arguing about. `simple` is EXACT at the
step counts every distilled graph here runs (4 and 8, both dividing 1,000) but
quantizes at 16, and 16 is the base checkpoint's step count. A check that fired
on the 16-step base graphs would be red while the state is correct, which
`CLAUDE.md` rates worse than no check -- so G3 pins that those graphs are out of
scope by construction rather than by luck. G1 and G2 pin the other direction:
the shift and the scheduler this repo actually ships must NOT trip it.
"""
import sys
from pathlib import Path

from harness import Harness, MUTATION, NEAR_MISS, main, subject

C = subject("bench/check_distill_grid.py")

VENDOR_README = (Path(__file__).resolve().parents[2] / "coderef"
                 / "Minimax-H3-Turbo" / "README.md")


def published():
    """The vendor's parsed grid, or None. The fixture guard below owns absence."""
    if not VENDOR_README.is_file():
        return None
    return C.parse_vendor_grid(VENDOR_README.read_text())


def repub(nfe=None, q=None, vid=None, aud=None, sv=None, sa=None):
    """The published tuple with one field replaced, for publication mutations.

    Editing `coderef/` is not an option: it is a gitignored checkout of somebody
    else's repository, and a harness that writes into it would leave the machine
    dirty in a way no `git status` here reports.
    """
    base = published()
    assert base is not None, "fixture guard should have exited before this"
    n0, sv0, sa0, q0, v0, a0 = base
    return (nfe or n0, sv if sv is not None else sv0,
            sa if sa is not None else sa0, q or q0, vid or v0, aud or a0)


# (label, kind, callable) -- each returns a problem list; empty is GREEN.
def build():
    h = Harness(subject="bench/check_distill_grid.py")
    h.fixture(VENDOR_README,
              "the vendor grid is one of the two independent sources; without "
              "it the publication mutations have nothing to mutate")

    # Baseline: the arm every 768p turbo graph ships, on grid.
    h.baseline(lambda: C.grade_arm(6.0, 3.0, "simple", 4))

    # --- arm mutations: what WE run moves off the grid --------------------
    h.case("M1 scheduler beta at 4 steps (the owner-recipe deviation)",
           MUTATION, lambda: C.grade_arm(6.0, 3.0, "beta", 4))
    h.case("M2 scheduler normal", MUTATION,
           lambda: C.grade_arm(6.0, 3.0, "normal", 4))
    h.case("M3 sgm_uniform -- the NEAREST miss, off by only 0.0074", MUTATION,
           lambda: C.grade_arm(6.0, 3.0, "sgm_uniform", 4))
    h.case("M4 ddim_uniform returns a different LENGTH, not a close grid",
           MUTATION, lambda: C.grade_arm(6.0, 3.0, "ddim_uniform", 4))
    # --- the seam with check_distill_settings.py --------------------------
    # These three were written as MUTATION and came back GREEN, which is the
    # harness working: `grade_arm` takes the shift and the step count as the
    # DEFINITION of the target grid, so it cannot detect a wrong one. Whether
    # this LoRA belongs at shift 6 and 4 steps is `check_distill_settings.py`'s
    # subject; asserting it here would be a second copy of that judgement, and
    # the two would drift. Kept as NEAR_MISS so the boundary is pinned rather
    # than implied -- if this file ever DOES start grading the shift, these go
    # red and someone has to decide which check owns it.
    h.case("G5 shift 12 on a 768p LoRA -- check_distill_settings' subject, "
           "not this one", NEAR_MISS,
           lambda: C.grade_arm(12.0, 3.0, "simple", 4))
    h.case("G6 audio shift 12 -- likewise", NEAR_MISS,
           lambda: C.grade_arm(6.0, 12.0, "simple", 4))
    h.case("G7 8 steps on a 4-step LoRA -- likewise", NEAR_MISS,
           lambda: C.grade_arm(6.0, 3.0, "simple", 8))

    # --- publication mutations: the VENDOR's numbers move -----------------
    h.case("M8 published video sigma edited (README drifts from its own rule)",
           MUTATION,
           lambda: C.grade_published(repub(vid=[1.0, 0.98, 0.9231, 0.8])))
    h.case("M9 published audio sigma edited", MUTATION,
           lambda: C.grade_published(repub(aud=[1.0, 0.91, 0.75, 0.5])))
    h.case("M10 published q is no longer (N-i)/N", MUTATION,
           lambda: C.grade_published(repub(q=[1.0, 0.8, 0.6, 0.4])))
    h.case("M11 published video shift changed, sigmas left alone", MUTATION,
           lambda: C.grade_published(repub(sv=6.0)))

    # --- near-misses: firing on these would make the check useless --------
    h.case("G1 the shift and scheduler every 768p turbo graph ships",
           NEAR_MISS, lambda: C.grade_arm(6.0, 3.0, "simple", 4))
    h.case("G2 the 544p turbo arm: shift 12/3, simple, 8 steps", NEAR_MISS,
           lambda: C.grade_arm(12.0, 3.0, "simple", 8))
    h.case("G3 base graphs at 16 steps are OUT of scope, not off-grid: "
           "simple quantizes there (1000 % 16 != 0) and the base checkpoint "
           "was never distilled to a step grid", NEAR_MISS,
           lambda: C.grade_arm(12.0, 3.0, "simple", 20))
    h.case("G4 the README exactly as the vendor publishes it", NEAR_MISS,
           lambda: C.grade_published(published()))
    return h


if __name__ == "__main__":
    sys.exit(main(build))
