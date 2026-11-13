#!/usr/bin/env python3
"""Run the standalone H3-Image Studio Tier 1 offline suite (`app/test_app.py`) under `--cpu`.

The H3-Image Studio lives in a gitignored standalone directory (`app/server.py`,
`app/routes.py`, `app/h3_image_tools.py`, `app/studio.sh`, `app/static/` with
`index.html`, `app.js`, and `style.css`, plus `app/test_app.py`), so its
24 Tier 1 offline checks stay invisible to `bench/run_checks.py` unless a
tracked wrapper invokes them. Because `app/` is optional in fresh clones (and
omitted in `bench/break_checks.py` copies), this wrapper calls
`bench/_lib::needs` and returns `exit 2` ("nothing graded") when
`app/test_app.py` is absent on disk.

Claims, i.e. what breaks if a case is deleted (delegated to `app/test_app.py`):

- `UI_DEFAULTS` & builder parity -- Studio defaults, T2I/Edit workflow builders,
  `MiniMax-H3-Image` `1..16` picture wiring, and one-frame recipes match
  `workflows/build_workflows.py` and pass ComfyUI's `execution.validate_prompt`.
- Live schema binding -- Mask, mode, LoRA, sampler, and attention dropdowns,
  plus control-vector and edit-guidance wiring, read from live node schemas and
  emit distinct `PreviewImage` nodes with zero `SaveImage` nodes.
- Privacy, isolation, & asset safety -- EXIF/metadata stripping on download,
  upload/temp path scoping, listed-Host checks, queue-aware session cleanup,
  stop-request scoping, and `docs/wiki/system_prompts/` loading stay green.

What it does not run: Tier 2 headless Chrome CDP (`app/test_app.py --browser <url>`),
which requires a live Studio HTTP process and ComfyUI `:8188` server.

No card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_image_studio.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import EXIT_FAIL, EXIT_OK, REPO, SWEEP_ENV, needs  # noqa: E402

STUDIO_TEST = REPO / "app" / "test_app.py"
STUDIO_FILES = (
    "app/test_app.py",
    "app/server.py",
    "app/routes.py",
    "app/h3_image_tools.py",
    "app/studio.sh",
    "app/static/index.html",
    "app/static/app.js",
    "app/static/style.css",
)


def main() -> int:
    needs("app/test_app.py (standalone Studio checkout)", STUDIO_TEST.is_file())
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env[SWEEP_ENV] = "1"
    done = subprocess.run(
        [sys.executable, str(STUDIO_TEST)],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        stdin=subprocess.DEVNULL,
    )
    if done.stdout:
        print(done.stdout, end="" if done.stdout.endswith("\n") else "\n")
    return EXIT_OK if done.returncode == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
