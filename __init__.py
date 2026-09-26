"""ComfyUI-h3-explorations: tinkering and research hub for the MiniMax H3 ecosystem.

Ships the MiniMax H3 SageAttention node plus supporting keyframe, provenance,
and chain-assert nodes. See README.md.
"""

from .nodes import comfy_entrypoint
from . import pipeline_telemetry as _telemetry

# Inert unless the server starts with H3_TELEMETRY set (docs/pipeline_telemetry.md).
# It observes through core's cache-provider API and a logging handler; it
# patches nothing.
_telemetry.install_from_env()

# The single-frame shim that used to be applied here is archived, and with it
# the last thing in this pack that modified ComfyUI core. Nothing here patches
# core now, at import or otherwise; model changes go through ModelPatcher's own
# `add_object_patch`. See `archive/single_frame.py` for what it was and
# `docs/h3_image_editing.md` for why the path it served is parked.

__all__ = ["comfy_entrypoint"]
