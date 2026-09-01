"""Import the Sol-Attn node that ACTUALLY RUNS, for scripts that need its code.

## Why this exists

Until 2026-08-30 the running node was `vendor/sol_attn_minimax.py`, so a script
wanting "what the node does" loaded that file. The node is now
`sol_attn_h3.py`, a first-class module in this pack, and the vendored file was
restored to a pristine upstream drop and its pack renamed `.disabled` --
ComfyUI does not load it and it cannot run on the installed kernel at all,
because it passes `centroid_tail` unconditionally and comfy-kitchen#117 removed
that argument.

**Six scripts kept loading the vendored file anyway**, which is this repo's own
2026-08-17 rule repeating: "Building the replacement is not the change.
Retiring the original and repointing everything that cites it is the change."
They were grading, plotting and measuring against a file nothing executes.
This module is the repoint, in one place, so the next move of the node is one
edit rather than six.

**`vendor/sol_attn_minimax.py` is still the source of truth for what UPSTREAM
said** -- that is why it is kept pristine. It is a diff target, not an import
target. `bench/check_sol_kernel.py` is the one caller that should still reach
for it, because asserting it is unmodified is its whole job.

## Why the module cannot just be spec-loaded

`sol_attn_h3.py` carries `from .block_spec import parse_blocks` at module
level, so loading it under a bare name raises "attempted relative import with
no known parent package". It is loaded here as a member of a synthetic package
whose `__path__` is the repo, which resolves the relative import without
executing `__init__.py` -- that would pull in every node in the pack and with
it all of ComfyUI, which an offline analysis script does not need.

    from _live_sol import live_sol
    perm, _ = live_sol().morton_perm(grid, "cpu", "hilbert")
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]

#: Named so a traceback says which package the module was loaded under, and
#: distinct from the pack's real name so it cannot collide with a running
#: ComfyUI that has already imported it for real.
_PACKAGE = "_h3_live_sol"

_CACHE: dict[str, types.ModuleType] = {}


def _package() -> types.ModuleType:
    """A namespace package rooted at the repo, without running __init__.py."""
    existing = sys.modules.get(_PACKAGE)
    if existing is not None:
        return existing
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(REPO)]
    sys.modules[_PACKAGE] = package
    return package


def _member(name: str) -> types.ModuleType:
    hit = _CACHE.get(name)
    if hit is not None:
        return hit
    path = REPO / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path.relative_to(REPO)} is missing")
    # ComfyUI is a checkout beside this pack, not an installed package, and
    # `sol_attn_h3` imports `comfy_api.latest`. Appended rather than inserted:
    # a caller that has already arranged its own sys.path wins.
    if str(COMFY) not in sys.path:
        sys.path.append(str(COMFY))
    _package()
    full = f"{_PACKAGE}.{name}"
    # A member the node already imported relatively (`from . import
    # sol_observe`) is the SAME module object a caller must get back. Loading
    # it again here made two copies under one name: the check armed one and
    # the node read the other, so every armed case ran unarmed and failed
    # "no file written" -- found on check_sol_observe.py's first run.
    existing = sys.modules.get(full)
    if existing is not None:
        _CACHE[name] = existing
        return existing
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    _CACHE[name] = module
    return module


def live_sol() -> types.ModuleType:
    """`sol_attn_h3`, the node every shipped graph wires."""
    return _member("sol_attn_h3")


def block_spec() -> types.ModuleType:
    """`block_spec`, which owns `parse_blocks` since it left the Sol node."""
    return _member("block_spec")


def sol_observe() -> types.ModuleType:
    """`sol_observe`, the env-gated route recorder the Sol node calls."""
    return _member("sol_observe")
