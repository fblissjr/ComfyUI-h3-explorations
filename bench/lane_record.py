#!/usr/bin/env python3
"""The masked lane's record in one SQLite file: build it from a project's record files, hold it to them, give files back.

    <comfy venv python> bench/lane_record.py --help

The tool is `bench/lane_db/` (its `cli.py` says each command); this file only starts it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_db.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
