"""ComfyUI-h3-explorations: tinkering and research hub for the MiniMax H3 ecosystem.

Ships the MiniMax H3 SageAttention node plus supporting keyframe, provenance,
and chain-assert nodes. See README.md.
"""

from .nodes import comfy_entrypoint
from . import pipeline_telemetry as _telemetry
from .control_vector_nodes import register_control_vectors_folder as _register_control_vectors_folder

_register_control_vectors_folder()

# Inert unless the server starts with H3_TELEMETRY set (docs/pipeline_telemetry.md).
# It observes through core's cache-provider API and a logging handler; it
# patches nothing.
_telemetry.install_from_env()


def _register_studio_routes() -> None:
    """Add the app cleanup route when its folder is here.
    """
    import importlib.util
    import logging
    from pathlib import Path

    path = Path(__file__).resolve().parent / "app" / "routes.py"
    if not path.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("h3_image_studio_routes", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.register()
    except Exception:
        logging.getLogger(__name__).warning(
            "[h3] app's cleanup route was not loaded from app/routes.py", exc_info=True)


_register_studio_routes()

# The single-frame shim that used to be applied here is archived, and with it
# the last thing in this pack that modified ComfyUI core. Nothing here patches
# core now, at import or otherwise; model changes go through ModelPatcher's own
# `add_object_patch`. See `archive/single_frame.py` for what it was and
# `docs/h3_image_editing.md` for why the path it served is parked.

__all__ = ["comfy_entrypoint"]
