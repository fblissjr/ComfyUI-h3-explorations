"""A correct harness. The spine MUST report this one passing.

The inert fixture alone proves only that the spine can fail. A control that can
only go one way cannot distinguish "the spine judges correctly" from "the spine
fails on everything", which is the emptiest-check problem `CLAUDE.md` names.
This is the other side.

One MUTATION that genuinely moves the verdict, one NEAR_MISS that genuinely
does not. `spine_control.py` runs it and requires exit 0.
"""
from harness import Harness, MUTATION, NEAR_MISS, main

STATE = {"broken": False}


def probe():
    return ["would be an error"] if STATE["broken"] else []


def real_mutation():
    STATE["broken"] = True
    try:
        return probe()
    finally:
        STATE["broken"] = False


def near_miss():
    """Changes something the subject is required to ignore."""
    STATE["irrelevant"] = True
    return probe()


def build():
    h = Harness(subject="(none -- fixture)")
    h.baseline(probe)
    h.case("real mutation moves the verdict", MUTATION, real_mutation)
    h.case("irrelevant change does not", NEAR_MISS, near_miss)
    return h


if __name__ == "__main__":
    main(build)
