"""Where ComfyUI's media and this box's captures live, resolved rather than typed.

These directories are per-machine. Several scripts here carried them as string
literals, which is wrong twice over: the path names somebody's specific storage
layout, and the script only runs on that one machine.

Resolution order, most authoritative first:

1. The environment variable, which is the escape hatch for any layout.
2. ComfyUI's own `folder_paths`, when importable -- it knows the directories
   ComfyUI was actually launched with, including ones redirected by a launcher
   to a network share, which is exactly the case a repo-relative guess gets
   wrong.
3. `<comfy>/input` or `<comfy>/output` beside this checkout, which is right for
   a stock install and is derived from this file's own location rather than
   assumed.

Returns `None` when nothing resolves, so callers fail with their own message
instead of silently writing somewhere unexpected. Nothing here creates a
directory.

    H3_COMFY_INPUT     reference images and other render inputs
    H3_COMFY_OUTPUT    rendered video and stills
    H3_CAPTURE_ROOT    the collection of activation-capture directories
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _comfy_root():
    """The ComfyUI checkout, by marker rather than by depth."""
    if len(_REPO.parents) < 2:
        return None
    root = _REPO.parents[1]
    markers = ("comfyui_version.py", "comfy", "main.py")
    return root if any((root / m).exists() for m in markers) else None


def _from_folder_paths(getter: str):
    try:
        import folder_paths  # only importable inside a running ComfyUI
    except Exception:
        return None
    try:
        value = getattr(folder_paths, getter)()
    except Exception:
        return None
    return Path(value) if value else None


def _resolve(env_var: str, getter: str, fallback: str):
    raw = os.environ.get(env_var)
    if raw:
        return Path(os.path.expanduser(raw))
    via_comfy = _from_folder_paths(getter)
    if via_comfy is not None:
        return via_comfy
    root = _comfy_root()
    return (root / fallback) if root else None


def comfy_input():
    """ComfyUI's input directory, or None."""
    return _resolve("H3_COMFY_INPUT", "get_input_directory", "input")


def comfy_output():
    """ComfyUI's output directory, or None."""
    return _resolve("H3_COMFY_OUTPUT", "get_output_directory", "output")


def capture_root():
    """The activation-capture collection, or None.

    Captures live outside the repo by design -- they are large and unversioned
    -- so there is no repo-relative fallback worth guessing. `H3_CAPTURE_ROOT`
    or nothing.
    """
    raw = os.environ.get("H3_CAPTURE_ROOT")
    return Path(os.path.expanduser(raw)) if raw else None


def describe(name: str, env_var: str) -> str:
    """The message a caller prints when a directory could not be resolved."""
    return (f"could not locate the {name} directory. Set {env_var}, or run "
            f"where ComfyUI's `folder_paths` is importable.")
