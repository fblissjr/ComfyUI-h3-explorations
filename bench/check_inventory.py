#!/usr/bin/env python3
"""Check `bench/inventory.py` on synthetic records (`done`, `partial`, `nothing laid`, `not seen`) and CLI refusals."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inventory as inv_mod  # noqa: E402
from _lib import case, finish  # noqa: E402


def states_and_markdown_table() -> str:
    """`done`, `partial`, `nothing laid`, and `not seen` states, owed spans, and laid piece descriptions are derived from the four records."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vdir = root / "data" / "clip"
        tdir = root / "tables"
        vdir.mkdir(parents=True)
        tdir.mkdir(parents=True)
        stamp = "2026-10-25T1200Z"

        (vdir / "plan.json").write_text(json.dumps({
            "subjects": {
                "lead": {"parts": ["Face_Neck"]},
                "second": {"parts": ["Face_Neck", "Hair"]},
                "third": {"parts": ["Face_Neck"]},
            }
        }))
        (vdir / "shots.json").write_text(json.dumps({
            "shots": [
                {
                    "first": 0,
                    "last": 9,
                    "who": {
                        "lead": {"present": "yes"},
                        "second": {
                            "present": "yes",
                            "seen_looks": [{"first": 0, "last": 9, "part": "head", "is": "not seen", "by": "owner", "why": "turned away"}],
                        },
                    },
                },
                {
                    "first": 10,
                    "last": 19,
                    "who": {
                        "lead": {"present": "yes"},
                        "third": {"present": "yes"},
                    },
                },
            ]
        }))
        (vdir / "laid.jsonl").write_text(
            json.dumps({"first": 0, "last": 9, "subject": "lead", "load": "lead_fp_0000", "owner": "kept"}) + "\n"
            + json.dumps({"first": 10, "last": 19, "subject": "lead", "load": "lead_fp_0010", "frames": [[10, 16]], "owner": "kept"}) + "\n"
        )
        import hashlib
        laid_sha = hashlib.sha256((vdir / "laid.jsonl").read_bytes()).hexdigest()
        (tdir / f"t_clip_0000_0019_{stamp}.json").write_text(json.dumps({
            "made": f"Made {stamp} from data/clip/laid.jsonl (sha256 {laid_sha[:16]}, 2 lines)",
            "reading": [
                {"first": 0, "last": 9, "subject": "lead", "stands": [{"render": "lead_fp_0000", "frames": [[0, 9]]}]},
                {"first": 0, "last": 9, "subject": "second", "stands": []},
                {"first": 10, "last": 19, "subject": "lead", "stands": [{"render": "lead_fp_0010", "frames": [[10, 16]]}]},
                {"first": 10, "last": 19, "subject": "third", "stands": []},
            ],
        }))
        (tdir / f"ledger_clip_{stamp}.json").write_text(json.dumps({
            "renders": {
                "1": {"name": "lead_fp_0000", "written": "2026-10-23T18:00:00Z", "graph": {"render_region": "zoomed in on what is replaced"}},
                "2": {"name": "lead_fp_0010", "written": "2026-10-14T18:00:00Z", "graph": {"render_region": "the whole frame"}},
            }
        }))
        (tdir / f"frame_coverage_clip_0000_0019_{stamp}.json").write_text(json.dumps({
            "shots": [
                {
                    "shot": [0, 9],
                    "subject": "lead",
                    "frames": list(range(0, 10)),
                    "step": [4] * 10,
                    "state": ["carried"] * 10,
                },
                {
                    "shot": [0, 9],
                    "subject": "second",
                    "frames": list(range(0, 10)),
                    "step": [1] * 10,
                    "state": ["left out by a look: owner: turned away"] * 10,
                },
                {
                    "shot": [10, 19],
                    "subject": "lead",
                    "frames": list(range(10, 20)),
                    "step": [4] * 7 + [2] * 3,
                    "state": ["carried"] * 5 + ["emptied inside a piece: floor"] * 2 + ["captured, not rendered"] * 3,
                },
                {
                    "shot": [10, 19],
                    "subject": "third",
                    "frames": list(range(10, 20)),
                    "step": [1] * 10,
                    "state": ["no capture"] * 10,
                },
            ]
        }))

        out_md = root / "inventory.md"
        out_json = root / "inventory.json"
        r = subprocess.run(
            [
                sys.executable, str(HERE / "inventory.py"),
                "--video", "clip", "--data", str(root / "data"),
                "--tables", str(tdir), "--stamp", stamp,
                "--out", str(out_md), "--json", str(out_json),
            ],
            capture_output=True, text=True, check=False,
        )
        assert r.returncode == 0, r.stderr
        doc = json.loads(out_json.read_text())
        assert doc["totals"] == {"done": 1, "not seen": 1, "partial": 1, "nothing laid": 1}, doc["totals"]
        assert doc["shot_subjects"] == 4 and doc["owed_frames"] == 15, doc
        by_key = {(row["first"], row["subject"]): row for row in doc["rows"]}
        assert by_key[(0, "lead")]["state"] == "done" and by_key[(0, "lead")]["owed"] == 0
        assert "lead_fp_0000 (10-23, zoom, 0-9)" in by_key[(0, "lead")]["pieces_laid"]
        assert by_key[(0, "second")]["state"] == "not seen" and by_key[(0, "second")]["by_a_look"] == 10
        assert by_key[(10, "lead")]["state"] == "partial" and by_key[(10, "lead")]["owed"] == 5
        assert by_key[(10, "lead")]["owed_spans"] == "15-19"
        assert "lead_fp_0010 (10-14, whole, 10-16)" in by_key[(10, "lead")]["pieces_laid"]
        assert by_key[(10, "third")]["state"] == "nothing laid" and by_key[(10, "third")]["owed"] == 10

        md = out_md.read_text()
        assert "| 0-9 | 10 | lead | done | 0 |  | 0 | lead_fp_0000 (10-23, zoom, 0-9) |" in md, md
        assert "| 10-19 | 10 | lead | partial | 5 | 15-19 | 0 | lead_fp_0010 (10-14, whole, 10-16) |" in md, md

        # Refusal when laid.jsonl sha256 disagrees with delivery table JSON
        (tdir / f"t_clip_0000_0019_{stamp}.json").write_text(json.dumps({
            "made": f"Made {stamp} from data/clip/laid.jsonl (sha256 {'0' * 16}, 2 lines)",
            "reading": [],
        }))
        r_sha = subprocess.run(
            [
                sys.executable, str(HERE / "inventory.py"),
                "--video", "clip", "--data", str(root / "data"),
                "--tables", str(tdir), "--stamp", stamp,
                "--out", str(out_md),
            ],
            capture_output=True, text=True, check=False,
        )
        assert r_sha.returncode == 1 and "does not match" in r_sha.stderr and "Traceback" not in r_sha.stderr, r_sha.stderr

        # Refusal when reading[].stands disagrees with laid.jsonl
        (tdir / f"t_clip_0000_0019_{stamp}.json").write_text(json.dumps({
            "made": f"Made {stamp} from data/clip/laid.jsonl (sha256 {laid_sha[:16]}, 2 lines)",
            "reading": [
                {"first": 0, "last": 9, "subject": "lead", "stands": [{"render": "wrong_load_name", "frames": [[0, 9]]}]},
            ],
        }))
        r_stands = subprocess.run(
            [
                sys.executable, str(HERE / "inventory.py"),
                "--video", "clip", "--data", str(root / "data"),
                "--tables", str(tdir), "--stamp", stamp,
                "--out", str(out_md),
            ],
            capture_output=True, text=True, check=False,
        )
        assert r_stands.returncode == 1 and "reading[].stands" in r_stands.stderr and "Traceback" not in r_stands.stderr, r_stands.stderr

    return "done, not seen, partial (including emptied_inside + unlaid tail), nothing laid, and delivery table sha256/stands checks match"


def refusals_have_no_traceback() -> str:
    """Missing files or folders exit 1 with REFUSED and no traceback."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r = subprocess.run(
            [
                sys.executable, str(HERE / "inventory.py"),
                "--video", "missing", "--data", str(root),
                "--tables", str(root / "no_tables"), "--stamp", "2026-10-25T1200Z",
                "--out", str(root / "out.md"),
            ],
            capture_output=True, text=True, check=False,
        )
        assert r.returncode == 1 and "REFUSED:" in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr)
    return "missing inputs refuse with exit 1, REFUSED message, and no traceback"


case("done, partial, nothing laid, and not seen states and Markdown/JSON outputs", states_and_markdown_table)
case("CLI refusals exit 1 with REFUSED and no traceback", refusals_have_no_traceback)

if __name__ == "__main__":
    sys.exit(finish())
