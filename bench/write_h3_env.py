#!/usr/bin/env python3
"""Write this machine's locations into the gitignored `h3.env`, discovered rather than typed.

Discovers the six keys of `h3.env.example` that this machine can answer by itself:

  H3_OUTPUT_DIR, H3_COMFY_OUTPUT   `--output-directory` in `<comfy>/start.sh`
  H3_INPUT_DIR, H3_COMFY_INPUT     `--input-directory` in `<comfy>/start.sh`
  H3_MODEL_STORE                   the folder `<comfy>/models/sapiens2/*` links into
  H3_CAPTURE_ROOT                  `$H3_MODEL_STORE/h3_captures`

and writes them into `h3.env` at the repo root (`workflows/h3_env.py::ENV_FILE`). Keys already in the
file that this script does not discover (the optional ones, filled by hand) are kept as they are. A key
it cannot discover is left out and named, never guessed. `--dry-run` prints what it would write.

    <comfy>/.venv/bin/python bench/write_h3_env.py [--dry-run]

Refuses to run when git does not ignore `h3.env`, so the file can never be the one that goes public.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("h3_env", REPO / "workflows" / "h3_env.py")
assert _spec is not None and _spec.loader is not None
h3_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h3_env)

DISCOVERED = ("H3_OUTPUT_DIR", "H3_INPUT_DIR", "H3_COMFY_OUTPUT", "H3_COMFY_INPUT", "H3_MODEL_STORE",
              "H3_CAPTURE_ROOT")


def flag(start_sh: str, name: str) -> str | None:
    """The value of `--name` in a launcher script, from the first line that STARTS with the flag
    (`--name path` or `--name=path`, after indentation), as `docs/wiki/ops/comfy_notes.md`'s awk recipe
    reads it. A flag named inside a comment or mid-line is not the launcher's argument."""
    m = re.search(rf"^[ \t]*--{re.escape(name)}(?:[ \t]+|=)(['\"]?)([^\s'\"\\]+)\1", start_sh, re.MULTILINE)
    return m.group(2) if m else None


def discover(comfy: Path) -> tuple[dict[str, str], list[str]]:
    found: dict[str, str] = {}
    missing: list[str] = []
    start = comfy / "start.sh"
    text = start.read_text(encoding="utf-8") if start.is_file() else ""
    for key_pair, name in ((("H3_OUTPUT_DIR", "H3_COMFY_OUTPUT"), "output-directory"),
                           (("H3_INPUT_DIR", "H3_COMFY_INPUT"), "input-directory")):
        value = flag(text, name)
        for key in key_pair:
            if value:
                found[key] = value
            else:
                missing.append(f"{key} (no --{name} in {start.name if text else 'a missing start.sh'})")
    links = sorted((comfy / "models" / "sapiens2").glob("*")) if (comfy / "models" / "sapiens2").is_dir() else []
    targets = [Path(os.path.realpath(p)).parent for p in links if p.is_symlink()]
    if targets:
        found["H3_MODEL_STORE"] = str(targets[0])
        found["H3_CAPTURE_ROOT"] = str(targets[0] / "h3_captures")
    else:
        missing += ["H3_MODEL_STORE (no links in <comfy>/models/sapiens2/)", "H3_CAPTURE_ROOT (follows H3_MODEL_STORE)"]
    return found, missing


def render(existing: str, found: dict[str, str]) -> str:
    """The new file: discovered keys first, then every other line of the old file unchanged."""
    keep = [line for line in existing.splitlines()
            if h3_env.parse(line).keys().isdisjoint(found)]
    head = ["# This machine's locations. Gitignored; never commit. Written by bench/write_h3_env.py,",
            "# which rewrites the discovered keys below and keeps every other line. Keys: h3.env.example."]
    body = [f"{k}={found[k]}" for k in DISCOVERED if k in found]
    rest = [line for line in keep if not line.startswith("# This machine's locations")
            and not line.startswith("# which rewrites the discovered keys")]
    while rest and not rest[0].strip():
        rest.pop(0)
    return "\n".join(head + body + ([""] + rest if rest else [])) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Write this machine's locations into the gitignored h3.env.")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
    args = ap.parse_args()
    ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "h3.env"]).returncode == 0
    if not ignored:
        print("refuse: git does not ignore h3.env here, so writing machine paths into it could publish them. "
              "Add `/h3.env` to .gitignore first.", file=sys.stderr)
        return 1
    comfy = REPO.parents[1]
    found, missing = discover(comfy)
    path = h3_env.ENV_FILE
    new = render(path.read_text(encoding="utf-8") if path.is_file() else "", found)
    if args.dry_run:
        print(new, end="")
    else:
        path.write_text(new, encoding="utf-8")
        print(f"wrote {path.name}: {', '.join(k for k in DISCOVERED if k in found)}")
    for m in missing:
        print(f"not discovered: {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
