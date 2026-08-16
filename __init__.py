"""ComfyUI-h3-explorations: tinkering and research hub for the MiniMax H3 ecosystem.

Ships the MiniMax H3 SageAttention node plus supporting keyframe, provenance,
and chain-assert nodes. See README.md.
"""

from .nodes import comfy_entrypoint
from . import single_frame

# ---------------------------------------------------------------------------
# TEMPORARY: patches ComfyUI core in memory so `length=1` reaches the
# single-image edit path. DELETE THIS CALL AND single_frame.py once ComfyUI
# ships the change itself (Comfy-Org/ComfyUI#15644) -- the shim detects that,
# does nothing, and says so at startup, so the console tells you when.
#
# Here rather than inside a node because what has to change is a schema ComfyUI
# reads at registration and at prompt validation, both of which happen without
# any node executing. `apply()` never raises, and refuses to install anything
# that would move an answer at `length >= 2`. See single_frame.py for scope and
# retirement, and bench/check_single_frame.py for the control that shows the
# refusal working.
# ---------------------------------------------------------------------------
single_frame.apply()

__all__ = ["comfy_entrypoint", "single_frame"]
