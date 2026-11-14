"""This machine's locations, read from the gitignored `h3.env` at the repo root.

The repo is public, so a folder that is this machine's is never written into a tracked file
(`bench/check_no_owner_paths.py`). The values live in `h3.env`, which git ignores; `h3.env.example`
(tracked) lists every key with what it means and no value, and `bench/write_h3_env.py` writes
`h3.env` by discovering the values (from `<comfy>/start.sh` and the model links), so nobody types a path.

`load()` copies each key in `h3.env` into `os.environ` unless the environment already has it, so an
exported variable always wins and a missing file changes nothing. Only `H3_*` keys are taken. The file
holds locations only: switches such as `H3_CAPTURE` (which arms a capture) never come from it.

Called by `workflows/h3_config.py::output_dir`, `bench/_paths.py` and `bench/_lib`. A shell takes the
same file with `set -a; . ./h3.env; set +a`.

`H3_ENV_FILE` names another file to read instead; a check that must see the environment as unset points
it at an empty file (`os.devnull`), so this machine's `h3.env` cannot answer for it.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / "h3.env"


def parse(text: str) -> dict[str, str]:
    """`KEY=value` lines; blank lines, `#` comments and a leading `export ` are allowed, and one pair of
    matching quotes around the value is removed. Keys that do not start with `H3_` are ignored."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.startswith("H3_") or not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def load(path: Path | None = None) -> dict[str, str]:
    """Copy `path`'s keys (default: `$H3_ENV_FILE`, else `ENV_FILE`) into `os.environ` where unset or
    empty; return what was set."""
    if path is None:
        path = Path(os.environ.get("H3_ENV_FILE") or ENV_FILE)
    try:
        values = parse(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    applied = {}
    for key, value in values.items():
        if value and not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied

