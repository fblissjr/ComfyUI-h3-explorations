"""Thin backward-compatibility shim aliasing `steering.nodes` (`Option B` Zero Manifest Change)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _ensure_steering_pkg() -> None:
    if "steering" in sys.modules:
        return
    pkg_dir = Path(__file__).resolve().parent / "steering"
    spec = importlib.util.spec_from_file_location(
        "steering",
        pkg_dir / "__init__.py",
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load steering package from {pkg_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["steering"] = mod
    spec.loader.exec_module(mod)


_ensure_steering_pkg()

from steering import legacy_nodes as _legacy_nodes  # noqa: E402, F401
from steering import nodes as _nodes  # noqa: E402

sys.modules[__name__] = _nodes
