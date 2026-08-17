"""A deliberately broken harness. The spine MUST report this one failing.

Its single MUTATION does not mutate: the case returns the same verdict as the
baseline, which is what a mutation that never reached its subject looks like.
That is the exact defect this directory exists to remove, so the spine is only
trustworthy if it goes red here.

Not a check, not a subject, and it takes no arguments. `spine_control.py` runs
it and requires exit 1.
"""
from harness import Harness, MUTATION, main

STATE = {"broken": False}


def probe():
    """Stands in for a subject: returns an error list when STATE says broken."""
    return ["would be an error"] if STATE["broken"] else []


def build():
    h = Harness(subject="(none -- fixture)")
    h.baseline(probe)
    # The bug, on purpose: STATE is never flipped, so the verdict cannot move.
    h.case("inert mutation, must be caught", MUTATION, probe)
    return h


if __name__ == "__main__":
    main(build)
