"""ComfyUI-h3-explorations: tinkering and research hub for the MiniMax H3 ecosystem.

Ships the MiniMax H3 SageAttention node plus supporting keyframe, provenance,
and chain-assert nodes. See README.md.
"""

from .nodes import comfy_entrypoint
from . import single_frame

# ---------------------------------------------------------------------------
# TEMPORARY, AND OFF UNLESS ASKED FOR. Patches ComfyUI core in memory so
# `length=1` reaches the single-image edit path. Does nothing at all unless
# `H3_EXPLORATIONS_SINGLE_FRAME=1` is set, and says nothing when it does
# nothing. DELETE THIS CALL AND single_frame.py once ComfyUI ships the change
# itself (Comfy-Org/ComfyUI#15644) -- the shim detects that, does nothing, and
# says so at startup, so the console tells you when.
#
# Opt-in since 2026-08-27, by owner decision: the patch is process-global, so
# leaving it on by default charged every install for a path most never use.
#
# Called here rather than from inside a node because a node cannot do it. The
# floor is enforced by `execution.py::validate_inputs`, which raises
# `value_smaller_than_min` before any node executes -- a node placed in the
# graph would be rejected along with the graph it exists to enable. The
# environment is read at import, which is before registration builds the
# schema. `apply()` never raises, and refuses to install anything that would
# move an answer at `length >= 2`. See single_frame.py for scope and
# retirement, and bench/check_single_frame.py for the control that shows the
# refusal working.
# ---------------------------------------------------------------------------
single_frame.apply()

__all__ = ["comfy_entrypoint", "single_frame"]
