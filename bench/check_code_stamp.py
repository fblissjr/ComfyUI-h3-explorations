#!/usr/bin/env python3
"""Check code-version stamps (`writer_stamp` / `npz_writer_fields`) and upstream-mtime checks across derived lane writers.

Claims, i.e. what breaks if a case is deleted:
- `writer_stamp, npz_writer_fields, and verify_writer_stamp`: `writer_stamp` or `npz_writer_fields` could omit `script`, `git_commit`, or `sha256`, or `verify_writer_stamp` could accept a wrong script name or mismatched file hash.
- `check_upstream_mtime tolerance and stale detection`: same-step mtime jitter within `MTIME_TOLERANCE_S = 1.0` s could false-refuse, or an upstream input modified `> 1.0` s after a downstream output could go undetected.
- `pose_along_shot, crop_reference_to_head, draw_face_points, and head_reference stamps`: any of the four reference-chain writers could stop stamping its `.json` or `.npz` output, `head_reference.write_filled` could raise on duplicate `NPZ_STAMP_KEYS`, or `draw_face_points.verify` / `head_reference.run` could miss a stale upstream `--mesh` or `--points` file.
- `head_tube_parts and measure_moge2_depth stamps and upstream mtime check`: `head_tube_parts.py` or `measure_moge2_depth.py` could omit `writer_stamp` / `npz_writer_fields`, or `measure_moge2_depth.run_measurement` could fail to report a newer `--source` video when refusing an existing output.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    MTIME_TOLERANCE_S,
    NPZ_STAMP_KEYS,
    REPO,
    STAMP_KEYS,
    case,
    check_upstream_mtime,
    finish,
    npz_writer_fields,
    refused,
    verify_writer_stamp,
    writer_stamp,
)
from _lib import face_lines_fixture as fl  # noqa: E402

import crop_reference_to_head as crh  # noqa: E402
import draw_face_points as dfp  # noqa: E402
import draw_lid_marks as marks  # noqa: E402
import head_reference as hr  # noqa: E402
import head_tube_parts as htp  # noqa: E402
import measure_moge2_depth as mmd  # noqa: E402
import pose_along_shot as pas  # noqa: E402


def _assert_npz_stamped(npz_path: Path, expected_script: str) -> None:
    with np.load(npz_path) as z:
        for k in NPZ_STAMP_KEYS:
            assert k in z.files, f"{npz_path.name} missing {k!r}"
        got_script = str(z["writer_script"].item())
        got_commit = str(z["writer_commit"].item())
        got_sha = str(z["writer_sha256"].item())
    assert got_script == expected_script, f"{npz_path.name}: writer_script={got_script!r}, expected {expected_script!r}"
    assert len(got_commit) >= 7, f"{npz_path.name}: writer_commit={got_commit!r}"
    assert len(got_sha) == 64, f"{npz_path.name}: writer_sha256={got_sha!r}"


def _assert_json_stamped(
    json_path: Path,
    expected_script: str,
    *,
    expected_extras: tuple[str, ...] = (),
) -> dict:
    rec = json.loads(json_path.read_text(encoding="utf-8"))
    assert "writer_stamp" in rec, f"{json_path.name} missing 'writer_stamp'"
    problems = verify_writer_stamp(rec["writer_stamp"], expected_script=expected_script)
    assert not problems, f"{json_path.name} invalid writer_stamp: {problems}"
    if expected_extras:
        extras = rec["writer_stamp"].get("extra_scripts")
        assert isinstance(extras, dict), f"{json_path.name} writer_stamp missing extra_scripts"
        for e in expected_extras:
            assert e in extras, f"{json_path.name} extra_scripts missing {e!r}"
            assert len(extras[e].get("sha256", "")) == 64, f"{json_path.name} extra_scripts[{e!r}] bad sha256"
    return rec


def _case_helpers() -> str:
    s = writer_stamp(
        REPO / "bench" / "pose_along_shot.py",
        extra_scripts=("bench/draw_face_points.py",),
    )
    problems = verify_writer_stamp(s, expected_script="bench/pose_along_shot.py")
    assert not problems, f"unexpected stamp problems: {problems}"
    assert set(STAMP_KEYS).issubset(s.keys())
    assert "bench/draw_face_points.py" in s.get("extra_scripts", {})

    bad = verify_writer_stamp({"script": "x"}, expected_script="bench/pose_along_shot.py")
    assert bad, "verify_writer_stamp accepted incomplete dict"
    wrong_script = verify_writer_stamp({**s, "script": "bench/other.py"}, expected_script="bench/pose_along_shot.py")
    assert wrong_script, "verify_writer_stamp accepted mismatched script"

    fields = npz_writer_fields(REPO / "bench" / "pose_along_shot.py")
    assert set(fields.keys()) == set(NPZ_STAMP_KEYS)
    assert str(fields["writer_script"].item()) == "bench/pose_along_shot.py"
    return f"commit={s['git_commit']} sha256={s['sha256'][:12]}..."


def _case_upstream_mtime() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        up = d / "upstream.npz"
        down = d / "downstream.mp4"
        up.write_bytes(b"up")
        down.write_bytes(b"down")
        base_t = 1_700_000_000.0
        os.utime(up, (base_t, base_t))
        os.utime(down, (base_t, base_t))
        assert check_upstream_mtime(down, {"--points": up}) == []

        os.utime(up, (base_t + 0.5, base_t + 0.5))
        assert check_upstream_mtime(down, {"--points": up}) == []

        os.utime(up, (base_t + MTIME_TOLERANCE_S + 2.5, base_t + MTIME_TOLERANCE_S + 2.5))
        stale = check_upstream_mtime(down, {"--points": up})
        assert len(stale) == 1 and "--points" in stale[0] and "upstream.npz" in stale[0], f"got {stale}"
    return f"tolerance={MTIME_TOLERANCE_S}s"


def _case_pose_crop_draw_and_head_ref() -> str:
    names = list(fl.NAMES)
    row_p = fl.face([(128.0, 128.0)])[0].astype(np.float32)
    row_s = np.full(len(names), 0.9, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        vid = d / "mesh.mp4"
        src = d / "src.mp4"
        frames = np.full((3, 256, 256, 3), 90, dtype=np.uint8)
        frames[:, 60:200, 60:200] = 140
        marks.write_video(vid, frames)
        marks.write_video(src, frames)

        # 1. pose_along_shot.run
        pts_out = d / "pose_pts.npz"
        pas.run(
            str(vid),
            0,
            3,
            (256, 256),
            pts_out,
            "check_code_stamp",
            predict=lambda _pic, _box: (row_p, row_s),
            names=names,
            model="synthetic",
            say=lambda *_: None,
        )
        _assert_npz_stamped(pts_out, "bench/pose_along_shot.py")
        _assert_json_stamped(pts_out.with_suffix(".json"), "bench/pose_along_shot.py")

        # 2. crop_reference_to_head.run
        crop_out = d / "cropped.mp4"
        crh.run(
            str(vid),
            str(pts_out),
            0,
            3,
            crop_out,
            "check_code_stamp",
            no_head_track=True,
            say=lambda *_: None,
        )
        _assert_json_stamped(crop_out.with_suffix(".json"), "bench/crop_reference_to_head.py")
        _assert_npz_stamped(d / "cropped_points.npz", "bench/crop_reference_to_head.py")

        # 3. draw_face_points.run + verify stale upstream warning
        drawn_out = d / "drawn.mp4"
        dfp.run(
            str(crop_out),
            str(d / "cropped_points.npz"),
            0,
            3,
            drawn_out,
            "check_code_stamp",
            say=lambda *_: None,
        )
        _assert_json_stamped(drawn_out.with_suffix(".json"), "bench/draw_face_points.py")

        future_t = drawn_out.stat().st_mtime + 10.0
        os.utime(d / "cropped_points.npz", (future_t, future_t))
        verify_lines: list[str] = []
        dfp.verify(
            str(drawn_out),
            str(crop_out),
            str(d / "cropped_points.npz"),
            0,
            3,
            say=verify_lines.append,
        )
        assert any("WARNING:" in ln and "cropped_points.npz" in ln for ln in verify_lines), (
            f"expected stale upstream warning from verify, got {verify_lines}"
        )

        # 4. head_reference.write_filled + head_reference.run + stale refusal
        filled_out = d / "filled_pts.npz"
        hr.write_filled(str(pts_out), {1: (0, 2)}, filled_out)
        _assert_npz_stamped(filled_out, "bench/head_reference.py")
        _assert_json_stamped(
            filled_out.with_suffix(".json"),
            "bench/head_reference.py",
            expected_extras=hr.TOOLS,
        )

        ref_out = d / "ref.mp4"
        hr.run(
            str(vid),
            str(pts_out),
            str(src),
            0,
            3,
            ref_out,
            "check_code_stamp",
            no_head_track=True,
            say=lambda *_: None,
        )
        _assert_json_stamped(
            ref_out.with_suffix(".json"),
            "bench/head_reference.py",
            expected_extras=hr.TOOLS,
        )

        ref_future = ref_out.stat().st_mtime + 10.0
        os.utime(pts_out, (ref_future, ref_future))
        msg = refused(
            lambda: hr.run(
                str(vid),
                str(pts_out),
                str(src),
                0,
                3,
                ref_out,
                "check_code_stamp",
                no_head_track=True,
                say=lambda *_: None,
            ),
            "takes a new name",
            "pose_pts.npz",
        )
        assert "newer than" in msg
    return "pose_along_shot, crop_reference_to_head, draw_face_points, head_reference ok"


def _case_head_tube_parts_and_moge2() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # 5. head_tube_parts.run_head_tube_parts
        logits = np.full((2, 29, 32, 32), -4.0, dtype=np.float32)
        logits[:, 0] = 2.0
        logits[:, 2, 8:24, 8:24] = 5.0  # Face_Neck
        tube = np.zeros((2, 32, 32), dtype=bool)
        tube[:, 6:26, 6:26] = True
        prov_out = d / "htp_prov.json"
        masks_out = d / "htp_masks.npz"
        htp.run_head_tube_parts(
            logits,
            tube,
            provenance_out=prov_out,
            masks_out=masks_out,
        )
        _assert_json_stamped(prov_out, "bench/head_tube_parts.py")
        _assert_npz_stamped(masks_out, "bench/head_tube_parts.py")

        # 6. measure_moge2_depth.run_measurement
        src_vid = d / "moge_src.mp4"
        frames = np.full((2, 64, 64, 3), 120, dtype=np.uint8)
        marks.write_video(src_vid, frames)
        moge_dir = d / "moge_out"
        mmd.run_measurement(
            source=src_vid,
            frames=[0, 1],
            out_dir=moge_dir,
            size=(64, 64),
            checkpoint="tiny",
            device="cpu",
            cell_width=64,
        )
        _assert_npz_stamped(moge_dir / "depth.npz", "bench/measure_moge2_depth.py")
        _assert_json_stamped(
            moge_dir / "record.json",
            "bench/measure_moge2_depth.py",
            expected_extras=("vendor/meta_moge2/hyden_moge2.py",),
        )

        future_t = (moge_dir / "depth.npz").stat().st_mtime + 10.0
        os.utime(src_vid, (future_t, future_t))
        msg = refused(
            lambda: mmd.run_measurement(
                source=src_vid,
                frames=[0, 1],
                out_dir=moge_dir,
                size=(64, 64),
                checkpoint="tiny",
                device="cpu",
                cell_width=64,
            ),
            "REFUSED",
            "already exists",
            "moge_src.mp4",
        )
        assert "newer than" in msg
    return "head_tube_parts and measure_moge2_depth ok"


def main() -> int:
    case("writer_stamp, npz_writer_fields, and verify_writer_stamp", _case_helpers)
    case("check_upstream_mtime tolerance and stale detection", _case_upstream_mtime)
    case("pose_along_shot, crop_reference_to_head, draw_face_points, and head_reference stamps", _case_pose_crop_draw_and_head_ref)
    case("head_tube_parts and measure_moge2_depth stamps and upstream mtime check", _case_head_tube_parts_and_moge2)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
