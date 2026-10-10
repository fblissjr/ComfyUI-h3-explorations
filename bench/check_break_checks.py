#!/usr/bin/env python3
"""`bench/break_checks.py` tells a break a check sees from one it does not, and from a run that was red for no reason.

Why it exists: on 2026-10-10 a session reported fourteen deliberate breaks as seen. The copies they were made in
could not import ComfyUI, so every run was red whatever was broken, and four of the breaks had never been seen by
any case. The tool that came out of it is only worth having if its own three claims hold, on a real check of this
pack that needs no model (`bench/check_subject_tracks.py`):

  a_control_is_run_first_and_is_green  the unbroken copy passes, in a copy shaped so the check imports ComfyUI
  a_real_break_is_seen                 the line `one_each` joins on, turned off: the check goes red on its own case
  a_change_that_does_nothing_is_not_seen   a comment edited: the check stays green and the tool says NOT SEEN
  text_not_found_once_is_not_applied   and nothing is run for it
  an_import_error_is_the_wrong_reason  a module that does not exist, imported: red, and called the wrong reason
  the_tree_is_not_touched              the file broken in the copy is unchanged in the tree

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_break_checks.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from _lib import case, finish  # noqa: E402

import break_checks as B  # noqa: E402

CHECK, FILE = "check_subject_tracks", "subject_tracks.py"
#: Each `old` is a line of `subject_tracks.py` this check relies on finding exactly once; if the file moves on, the
#: case says NOT APPLIED and names it, and the line here is updated.
JOIN = "        held = [(shared, -area[k], -k) for shared, k in inside if shared >= same * area[i]]"
COMMENT = "    thing: dict[int, int] = {}                       # detection: the detection that started its thing"
_state: dict = {}


def results() -> tuple[dict, dict]:
    """One run of the tool for every case: the controls, and each break's result by a name."""
    if not _state:
        before = hashlib.sha256((REPO / FILE).read_bytes()).hexdigest()
        breaks = [
            {"name": "real", "file": FILE, "check": CHECK, "old": JOIN, "new": "        held = []"},
            {"name": "nothing", "file": FILE, "check": CHECK, "old": COMMENT, "new": COMMENT.replace("# detection:", "# a detection:")},
            {"name": "absent", "file": FILE, "check": CHECK, "old": "this text is not in the file at all", "new": "x"},
            {"name": "import", "file": FILE, "check": CHECK, "old": "import torch\n", "new": "import torch\nimport a_module_that_does_not_exist\n"},
        ]
        controls, got = B.break_checks(breaks, sys.executable)
        _state.update(controls=controls, by_name={r["name"]: r for r in got},
                      untouched=hashlib.sha256((REPO / FILE).read_bytes()).hexdigest() == before)
    return _state["controls"], _state["by_name"]


def a_control_is_run_first_and_is_green():
    controls, by_name = results()
    assert controls == {CHECK: True}, f"the unbroken copy must pass, in a copy the check can import ComfyUI from: {controls}"
    assert len(by_name) == 4, "with a green control every break is tried"


def a_real_break_is_seen():
    got = results()[1]["real"]
    assert got["outcome"] == B.SEEN, f"`one_each` joining nothing must turn its case red: {got['outcome']}, {got['said'][:2]}"
    assert any("nested_detections_are_one_thing" in line for line in got["said"]), f"and the failing line names the case: {got['said'][:3]}"


def a_change_that_does_nothing_is_not_seen():
    got = results()[1]["nothing"]
    assert got["outcome"] == B.NOT_SEEN, f"a comment edited changes nothing, and the tool must say the check did not see it: {got['outcome']}"


def text_not_found_once_is_not_applied():
    got = results()[1]["absent"]
    assert got["outcome"] == B.NOT_APPLIED and "0 time(s)" in got["said"][0], got


def an_import_error_is_the_wrong_reason():
    got = results()[1]["import"]
    assert got["outcome"] == B.WRONG_REASON, (f"a red caused by an import error is not a break being seen; this is the fault of "
                                              f"2026-10-10: {got['outcome']}, {got['said'][:2]}")


def the_tree_is_not_touched():
    results()
    assert _state["untouched"], f"{FILE} in the tree changed while breaks were made in the copy"


def main() -> int:
    print("bench/check_break_checks.py\n")
    for fn in (a_control_is_run_first_and_is_green, a_real_break_is_seen, a_change_that_does_nothing_is_not_seen,
               text_not_found_once_is_not_applied, an_import_error_is_the_wrong_reason, the_tree_is_not_touched):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
