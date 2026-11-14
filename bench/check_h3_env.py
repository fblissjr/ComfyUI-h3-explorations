#!/usr/bin/env python3
"""That this machine's locations stay out of git: `h3.env`, `h3.env.example`, `workflows/h3_env.py` and
`bench/write_h3_env.py`.

What each case defends, i.e. what breaks if it is deleted:

- `h3.env is ignored by git` -- the real file holds absolute paths and the repo is public. Control: the
  tracked template beside it is NOT ignored, so the rule is not one that hides everything.
- `the template holds keys and no values` -- every key line in `h3.env.example` is empty and the file names
  no machine folder, so copying it can publish nothing.
- `no switch is a key in the template` -- `H3_CAPTURE` arms a capture and must never come from a file, nor
  the per-run keys (`H3_SESSION`, `H3_TRACE`, `H3_DEVICE`, `H3_SOL_*`, `H3_LLM_*`), set or commented.
- `the loader fills only what is unset` -- `h3_env.load` on a fixture: `export`, quotes, comments, an inline
  comment and a key without the `H3_` prefix parse as intended; an exported value is kept; a missing file
  changes nothing; `H3_ENV_FILE` pointed at an empty file stands in for this machine's `h3.env`.
- `the writer reads the launcher's flag, not a comment` -- `write_h3_env.flag` on a launcher fixture whose
  comment names the flag before the real, indented, backslash-continued line; both spellings; absent is None.
  Control: the first version, which took the first mention anywhere, reads the comment.
- `the writer's keys are the template's` -- each key `write_h3_env.py` discovers is a key of the template.

No card, no model, no server; git for the first case.

    <comfy venv python> bench/check_h3_env.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h3_env = _module("h3_env", REPO / "workflows" / "h3_env.py")
writer = _module("write_h3_env", REPO / "bench" / "write_h3_env.py")
EXAMPLE = (REPO / "h3.env.example").read_text(encoding="utf-8")
SWITCHES = ("H3_CAPTURE", "H3_SESSION", "H3_TRACE", "H3_DEVICE", "H3_SINGLE_FRAME_DEBUG", "H3_ENV_FILE")
SWITCH_PREFIXES = ("H3_SOL_", "H3_LLM_")
# top-level folders a machine path starts with, and a drive letter, built from parts so the pattern names
# no folder itself
MACHINE = re.compile("(" + "|".join("/" + top + "/" for top in ("home", "mnt", "Users", "media", "srv"))
                     + r"|[A-Za-z]:\\)")


def ignored(name: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", name]).returncode == 0


def is_ignored():
    assert ignored("h3.env"), "git does not ignore h3.env"
    assert not ignored("h3.env.example"), "control failed: the template is ignored too"
    return "control (the template) is tracked"


def template_empty():
    keys = h3_env.parse(EXAMPLE)
    assert keys, "the template parses to no keys"
    full = {k: v for k, v in keys.items() if v}
    assert not full, f"keys with values in the template: {sorted(full)}"
    machine = MACHINE.findall(EXAMPLE)
    assert not machine, f"the template names a machine folder: {machine}"
    return f"{len(keys)} keys, all empty"


def no_switches():
    named = set(re.findall(r"^#?\s*(H3_[A-Z0-9_]+)=", EXAMPLE, re.MULTILINE))
    bad = sorted(k for k in named if k in SWITCHES or k.startswith(SWITCH_PREFIXES))
    assert not bad, f"switches as keys in the template: {bad}"


def loader():
    fixture = (
        "# comment\n\nexport H3_A=/x/a\nH3_B=\"/x/b c\"\nH3_C='/x/c'\nH3_D=/x/d # note\n"
        "OTHER=/x/o\nH3_E=\nH3_PRESET=/from/file\n"
    )
    parsed = h3_env.parse(fixture)
    want = {"H3_A": "/x/a", "H3_B": "/x/b c", "H3_C": "/x/c", "H3_D": "/x/d", "H3_E": "", "H3_PRESET": "/from/file"}
    assert parsed == want, parsed
    saved = dict(os.environ)
    try:
        for k in list(want) + ["H3_ENV_FILE"]:
            os.environ.pop(k, None)
        os.environ["H3_PRESET"] = "/exported"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "h3.env"
            f.write_text(fixture)
            applied = h3_env.load(f)
            assert os.environ["H3_PRESET"] == "/exported", "the file overrode an exported value"
            assert "H3_E" not in os.environ, "an empty value was set"
            assert set(applied) == {"H3_A", "H3_B", "H3_C", "H3_D"}, applied
            assert h3_env.load(Path(d) / "absent.env") == {}, "a missing file changed something"
            for k in ("H3_A", "H3_B", "H3_C", "H3_D"):
                os.environ.pop(k)
            os.environ["H3_ENV_FILE"] = os.devnull
            assert h3_env.load() == {}, "H3_ENV_FILE pointed at an empty file still loaded something"
            os.environ["H3_ENV_FILE"] = str(f)
            assert set(h3_env.load()) == {"H3_A", "H3_B", "H3_C", "H3_D"}, "H3_ENV_FILE was not followed"
    finally:
        os.environ.clear()
        os.environ.update(saved)


def writer_flag():
    launcher = (
        "#!/bin/bash\n#   --output-directory   where finished renders are written\n"
        "exec python main.py \\\n    --output-directory /srv/out \\\n    --input-directory=/srv/in \\\n    --port 8188\n"
    )
    assert writer.flag(launcher, "output-directory") == "/srv/out", writer.flag(launcher, "output-directory")
    assert writer.flag(launcher, "input-directory") == "/srv/in"
    assert writer.flag(launcher, "temp-directory") is None
    first_mention = re.search(r"(?<![\w-])--output-directory(?:[ \t]+|=)(\S+)", launcher)
    assert first_mention and first_mention.group(1) == "where", "control did not read the comment"
    return "control (first mention anywhere) reads 'where', as the first version did"


def writer_keys():
    keys = set(h3_env.parse(EXAMPLE))
    missing = [k for k in writer.DISCOVERED if k not in keys]
    assert not missing, f"discovered keys missing from the template: {missing}"


case("h3.env is ignored by git", is_ignored)
case("the template holds keys and no values", template_empty)
case("no switch is a key in the template", no_switches)
case("the loader fills only what is unset", loader)
case("the writer reads the launcher's flag, not a comment", writer_flag)
case("the writer's keys are the template's", writer_keys)
sys.exit(finish())
