"""Check ``bench/timeline_page.py``: whole-video NLE timeline data file and HTML builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bench"))

import timeline_page  # noqa: E402

SCRIPT = REPO_ROOT / "bench" / "timeline_page.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_fixture(tmp: Path) -> dict[str, Path]:
    video = "clip"
    stamp = "2026-10-24T2341Z"
    data_dir = tmp / "data"
    vdir = data_dir / video
    tables_dir = tmp / "tables"

    shots_path = vdir / "shots.json"
    laid_path = vdir / "laid.jsonl"
    plan_path = vdir / "plan.json"
    answers_path = vdir / "owner_answers.jsonl"
    verdicts_path = vdir / "verdicts" / "verdicts.jsonl"
    table_path = tables_dir / f"t_{video}_rebuild_{stamp}.json"
    ledger_path = tables_dir / f"ledger_{video}_{stamp}.json"
    cov_path = tables_dir / f"frame_coverage_{video}_rebuild_{stamp}.json"

    _write_json(
        plan_path,
        {
            "subjects": {
                "lead": {"parts": ["head"], "region": "zoom"},
                "second": {"parts": ["head"], "region": "whole"},
            }
        },
    )

    _write_json(
        shots_path,
        {
            "shots": [
                {
                    "first": 0,
                    "last": 9,
                    "order": 1,
                    "scene": "opening",
                    "who": {
                        "lead": {"present": "yes"},
                        "second": {
                            "present": "yes",
                            "seen_looks": [
                                {
                                    "first": 0,
                                    "last": 9,
                                    "part": "head",
                                    "is": "not seen",
                                    "by": "owner",
                                    "why": "off screen",
                                }
                            ],
                        },
                    },
                },
                {
                    "first": 10,
                    "last": 19,
                    "order": 2,
                    "scene": "middle",
                    "who": {
                        "lead": {
                            "present": "yes",
                            "seen_looks": [
                                {
                                    "first": 10,
                                    "last": 14,
                                    "part": "head",
                                    "is": "not seen",
                                    "by": "owner",
                                    "why": "turned away",
                                }
                            ],
                        },
                        "second": {"present": "yes"},
                    },
                },
            ]
        },
    )

    laid_rows = [
        {
            "name": "clip_0000_0009_lead_v1",
            "first": 0,
            "last": 9,
            "subject": "lead",
            "status": "laid",
        },
        {
            "name": "clip_0015_0017_lead_v2",
            "first": 10,
            "last": 19,
            "subject": "lead",
            "status": "laid",
        },
    ]
    _write_jsonl(laid_path, laid_rows)
    laid_sha = hashlib.sha256(laid_path.read_bytes()).hexdigest()

    _write_json(
        ledger_path,
        {
            "renders": {
                "clip_0000_0009_lead_v1": {
                    "name": "clip_0000_0009_lead_v1",
                    "written": "2026-10-19T12:00:00Z",
                    "file": f"/mnt/share/output/Video/{video}/clip_0000_0009_lead_v1.mp4",
                    "graph": {"render_region": "zoom_bbox"},
                },
                "clip_0015_0017_lead_v2": {
                    "name": "clip_0015_0017_lead_v2",
                    "written": "2026-10-21T12:00:00Z",
                    "file": f"/mnt/share/output/Video/{video}/clip_0015_0017_lead_v2.mp4",
                    "graph": {"render_region": "whole_frame"},
                },
            }
        },
    )

    fake_home = "/" + "home" + "/example"
    _write_json(
        table_path,
        {
            "made": f"built from data/{video}/laid.jsonl (sha256 {laid_sha[:16]})",
            "assembly": {
                "command": (
                    f"python {fake_home}/bench/assemble.py "
                    f"--source /mnt/share/output/Video/{video}/raw.mp4 "
                    f"--out /mnt/share/output/Video/{video}/assembled.mp4"
                ),
            },
            "rows": [
                {
                    "name": "clip_0000_0009_lead_v1",
                    "file": f"/mnt/share/output/Video/{video}/clip_0000_0009_lead_v1.mp4",
                },
                {
                    "name": "clip_0015_0017_lead_v2",
                    "file": f"/mnt/share/output/Video/{video}/clip_0015_0017_lead_v2.mp4",
                },
            ],
        },
    )
    table_sha = hashlib.sha256(table_path.read_bytes()).hexdigest()

    _write_json(
        cov_path,
        {
            "table_sha256": table_sha,
            "shots": [
                {
                    "shot": [0, 9],
                    "subject": "lead",
                    "frames": list(range(0, 10)),
                    "step": [4] * 10,
                    "step_name": ["carried"] * 10,
                    "state": ["carried"] * 10,
                    "piece": ["clip_0000_0009_lead_v1"] * 10,
                    "track_px": [500] * 10,
                    "carried_px": [480] * 10,
                    "changed_share": [0.25] * 10,
                },
                {
                    "shot": [0, 9],
                    "subject": "second",
                    "frames": list(range(0, 10)),
                    "step": [1] * 10,
                    "step_name": ["left_out_by_a_look"] * 10,
                    "state": ["left out by a look (not seen)"] * 10,
                    "piece": [None] * 10,
                    "track_px": [0] * 10,
                    "carried_px": [0] * 10,
                    "changed_share": [None] * 10,
                },
                {
                    "shot": [10, 19],
                    "subject": "lead",
                    "frames": list(range(10, 20)),
                    "step": [1] * 5 + [4] * 3 + [2] * 2,
                    "step_name": ["left_out_by_a_look"] * 5 + ["carried"] * 3 + ["captured_not_rendered"] * 2,
                    "state": ["left out by a look (not seen)"] * 5 + ["carried"] * 3 + ["captured, not rendered"] * 2,
                    "piece": [None] * 5 + ["clip_0015_0017_lead_v2"] * 3 + [None] * 2,
                    "track_px": [0] * 5 + [400] * 3 + [350] * 2,
                    "carried_px": [0] * 5 + [390] * 3 + [0] * 2,
                    "changed_share": [None] * 5 + [0.3] * 3 + [None] * 2,
                },
                {
                    "shot": [10, 19],
                    "subject": "second",
                    "frames": list(range(10, 20)),
                    "step": [1] * 10,
                    "step_name": ["no_capture"] * 10,
                    "state": ["no capture"] * 10,
                    "piece": [None] * 10,
                    "track_px": [0] * 10,
                    "carried_px": [0] * 10,
                    "changed_share": [None] * 10,
                },
            ],
            "seams": [
                {
                    "shot": [10, 19],
                    "subject": "lead",
                    "at": 15,
                    "before": {
                        "piece": "clip_0010_0014_lead_v1",
                        "day": "10-19",
                        "written": "2026-10-19T10:00:00Z",
                        "file": f"/mnt/share/output/Video/{video}/clip_0010_0014_lead_v1.mp4",
                    },
                    "after": {
                        "piece": "clip_0015_0017_lead_v2",
                        "day": "10-21",
                        "written": "2026-10-21T12:00:00Z",
                        "file": f"/mnt/share/output/Video/{video}/clip_0015_0017_lead_v2.mp4",
                    },
                }
            ],
        },
    )

    _write_jsonl(
        answers_path,
        [
            {
                "at": "2026-10-16T04:19:57Z",
                "row": "clip_0000_0019_joined",
                "frames": [0, 19],
                "frame": 16,
                "source_frame": 16,
                "words": "Pinned whole-cut note at frame 16.",
            },
            {
                "at": "2026-10-24T12:00:00Z",
                "row": "clip_0015_0017_lead_v2",
                "frames": [15, 17],
                "subject": "lead",
                "piece": "clip_0015_0017_lead_v2",
                "tag": "seam",
                "words": f"Check frames 18-19 tail in {fake_home}/data/{video}/clip.mp4.",
            },
        ],
    )
    _write_jsonl(
        verdicts_path,
        [
            {
                "at": "2026-10-25T02:00:00Z",
                "shot": [10, 19],
                "person": "second",
                "verdict": "note",
                "words": "Need second laid across 10-19.",
                "source": "the lead's terminal, 2026-10-25",
            }
        ],
    )

    return {
        "video": video,
        "stamp": stamp,
        "data_dir": data_dir,
        "tables_dir": tables_dir,
        "laid": laid_path,
        "cov": cov_path,
    }


def check_build_and_html() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        fx = _make_fixture(tmp)
        out_html = tmp / "out" / "TIMELINE.html"
        out_json = tmp / "out" / "timeline.json"

        model = timeline_page.build_timeline_model(
            video=fx["video"],
            data_dir=fx["data_dir"],
            tables_dir=fx["tables_dir"],
            stamp=fx["stamp"],
            out_html_dir=out_html.parent,
        )

        assert model["schema"] == "h3_timeline/1"
        meta = model["meta"]
        assert meta["laid_matches_table"] is True
        assert meta["stale_banner"] == ""
        totals = meta["totals"]
        assert totals["shots"] == 2
        assert totals["subjects"] == 2
        assert totals["laid_pieces"] == 2
        assert totals["owed_frame_subjects"] == 12
        assert totals["owed_spans"] == 2
        assert totals["left_out_frame_subjects"] == 15
        assert totals["seams"] == 1
        assert totals["notes"] == 3
        assert totals["laid_frames_by_age_tier"] == {"pre-10/20": 10, "10/20-10/22": 3}

        # Verify CLI writes HTML and JSON and scrubs machine paths
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--video",
                fx["video"],
                "--data",
                str(fx["data_dir"]),
                "--tables",
                str(fx["tables_dir"]),
                "--stamp",
                fx["stamp"],
                "--out-json",
                str(out_json),
                "--out-html",
                str(out_html),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}: {proc.stderr}"
        html_text = out_html.read_text(encoding="utf-8")
        json_text = out_json.read_text(encoding="utf-8")

        for token in (
            ':root[data-theme="dark"]',
            "@media (prefers-color-scheme: dark)",
            "id=\"mainPlayer\"",
            "id=\"sourceSwitcher\"",
            "id=\"ageFilterBar\"",
            "data-age-filter=\"24h\"",
            "data-age-filter=\"48h\"",
            "data-age-filter=\"72h\"",
            "data-age-filter=\"older\"",
            "id=\"nleCanvas\"",
            "id=\"pinNoteForm\"",
            "id=\"calloutStrip\"",
        ):
            assert token in html_text, f"missing token in HTML: {token}"

        for forbidden in ("/" + "home/", "/" + "mnt/", "http://", "https://", ":8765"):
            assert forbidden not in html_text, f"forbidden token {forbidden!r} in HTML"
            assert forbidden not in json_text, f"forbidden token {forbidden!r} in JSON"


def check_stale_laid_and_refusals() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        fx = _make_fixture(tmp)

        # Append a new row to laid.jsonl so laid_sha256 diverges from table_json
        with fx["laid"].open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "name": "clip_0010_0019_second_v1",
                        "first": 10,
                        "last": 19,
                        "subject": "second",
                        "status": "laid",
                    }
                )
                + "\n"
            )

        out_html = tmp / "out_stale" / "TIMELINE.html"
        out_json = tmp / "out_stale" / "timeline.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--video",
                fx["video"],
                "--data",
                str(fx["data_dir"]),
                "--tables",
                str(fx["tables_dir"]),
                "--stamp",
                fx["stamp"],
                "--out-json",
                str(out_json),
                "--out-html",
                str(out_html),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        html_text = out_html.read_text(encoding="utf-8")
        assert "stale-banner" in html_text and "laid.jsonl has moved since rebuild" in html_text

        # Corrupt coverage_json table_sha256 -> must refuse cleanly with REFUSED: and no traceback
        bad_cov = json.loads(fx["cov"].read_text(encoding="utf-8"))
        bad_cov["table_sha256"] = "0" * 64
        _write_json(fx["cov"], bad_cov)

        proc_bad = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--video",
                fx["video"],
                "--data",
                str(fx["data_dir"]),
                "--tables",
                str(fx["tables_dir"]),
                "--stamp",
                fx["stamp"],
                "--out-json",
                str(out_json),
                "--out-html",
                str(out_html),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_bad.returncode != 0, "expected non-zero exit on mismatched table_sha256"
        assert "REFUSED:" in proc_bad.stderr, f"expected REFUSED: in stderr, got: {proc_bad.stderr}"
        assert "Traceback" not in proc_bad.stderr, f"unexpected traceback: {proc_bad.stderr}"


def main() -> int:
    cases = [
        ("build_and_html", check_build_and_html),
        ("stale_laid_and_refusals", check_stale_laid_and_refusals),
    ]
    for name, fn in cases:
        fn()
        print(f"ok {name}")
    print(f"all {len(cases)} case(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
