"""ComfyUI-h3-explorations: tinkering and research hub for the MiniMax H3 ecosystem.

Ships the MiniMax H3 SageAttention node plus supporting keyframe, provenance,
and chain-assert nodes. See README.md.
"""

from .nodes import comfy_entrypoint

# The single-frame shim that used to be applied here is archived, and with it
# the last thing in this pack that modified ComfyUI core. Nothing here patches
# core now, at import or otherwise; model changes go through ModelPatcher's own
# `add_object_patch`. See `archive/single_frame.py` for what it was and
# `docs/h3_image_editing.md` for why the path it served is parked.

__all__ = ["comfy_entrypoint"]
