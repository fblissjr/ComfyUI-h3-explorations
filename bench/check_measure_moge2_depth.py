#!/usr/bin/env python3
"""Check `vendor/meta_moge2/hyden_moge2.py` (`vendor/meta_moge2/LICENSE`) and `bench/measure_moge2_depth.py` on CPU.

Claims, i.e. what breaks if a case is deleted:
1. `raw_rgb_input_guard`: deleting `validate_raw_rgb` or its dtype/range checks lets `uint8`
   `[0, 255]` tensors or pre-normalized ImageNet `[-2.1, 2.6]` tensors enter `HyDenMoGe`, corrupting
   both the DINOv2 branch (double-normalized) and the `CNN_Large` branch (which takes raw `[0, 1]`).
2. `focal_shift_and_intrinsics_recovery_geometry`: breaking `recover_focal_shift`,
   `recover_intrinsics`, or `fov_deg_from_intrinsics` fails to recover known `(fx, fy, shift)` and
   metric `depth = points[..., 2] + shift` from a synthetic 3D point cloud within `1e-3`.
3. `end_to_end_tool_run_npz_record_and_sheet`: breaking `measure_moge2_depth.py` breaks `depth.npz`,
   `record.json` (model, frames, box, region depth stats, intrinsics variation stats), or the
   side-by-side `sheet.jpg` RGB + depth panels.
4. `state_dict_key_completeness`: dropping a head, buffer, or `mask_token` from `HyDenMoGe` changes
   the 953-key checkpoint schema (738 parameters + 215 buffers) so `load_state_dict(..., strict=True)`
   fails, or modifying `vendor/meta_moge2/LICENSE` breaks the upstream license hash / byte-for-byte match.
5. `refusals_leave_no_partial_files`: removing CLI/input guards on existing outputs, missing source
   video, bad `--box`, or out-of-range frame numbers crashes with a traceback or writes partial files.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_measure_moge2_depth.py
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish, refused  # noqa: E402
from _lib.frames import write_mask_video  # noqa: E402

sys.path.insert(0, str(REPO))
from vendor.meta_moge2.hyden_moge2 import (  # noqa: E402
    CHECKPOINT_NUM_KEYS,
    DEFAULT_CHECKPOINT,
    HyDenMoGe,
    fov_deg_from_intrinsics,
    normalized_view_plane_uv,
    recover_focal_shift,
    recover_intrinsics,
    validate_raw_rgb,
)
import measure_moge2_depth  # noqa: E402

#: sha256 of `coderef/metadepth/LICENSE` at commit 810b77c3e56712813a0de42a130cfbe6f5b19b90 (inherited: read off upstream LICENSE on 2026-10-17).
LICENCE_SHA256 = "363d326fed0d13fd2418039c6f4bb56a809b0e316aaae3d5222d395581060288"


def _write_synthetic_rgb_video(path: Path, width: int = 128, height: int = 96, num_frames: int = 6) -> None:
    cmd = [
        "nice", "-n", "19", "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24",
        "-i", "-", "-c:v", "ffv1", "-level", "3", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    yy = np.linspace(30, 220, height, dtype=np.uint8)[:, None, None]
    xx = np.linspace(20, 200, width, dtype=np.uint8)[None, :, None]
    for f in range(num_frames):
        r = np.broadcast_to(yy, (height, width, 1))
        g = np.broadcast_to(xx, (height, width, 1))
        b = np.full((height, width, 1), 40 + f * 25, dtype=np.uint8)
        frame = np.concatenate([r, g, b], axis=-1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def raw_rgb_input_guard() -> str:
    tiny = HyDenMoGe.tiny_for_tests()
    u8_img = torch.randint(0, 256, (1, 3, 56, 56), dtype=torch.uint8)
    refused(lambda: validate_raw_rgb(u8_img), "raw-RGB guard", "uint8")
    refused(lambda: tiny.forward(u8_img), "raw-RGB guard", "uint8")

    imagenet_img = (torch.rand(1, 3, 56, 56, dtype=torch.float32) - 0.485) / 0.229
    refused(lambda: validate_raw_rgb(imagenet_img), "raw-RGB guard", "ImageNet")
    refused(lambda: tiny.infer(imagenet_img), "raw-RGB guard", "ImageNet")

    unscaled_f32 = torch.rand(1, 3, 56, 56, dtype=torch.float32) * 255.0
    refused(lambda: validate_raw_rgb(unscaled_f32), "raw-RGB guard", "[0, 1]")

    bad_shape = torch.rand(1, 4, 56, 56, dtype=torch.float32)
    refused(lambda: validate_raw_rgb(bad_shape), "raw-RGB guard", "shape")

    valid_rgb = torch.linspace(0.0, 1.0, 3 * 56 * 56, dtype=torch.float32).reshape(1, 3, 56, 56)
    validate_raw_rgb(valid_rgb)
    out = tiny.infer(valid_rgb)
    assert out["depth"].shape == (1, 56, 56), f"unexpected depth shape {out['depth'].shape}"
    return "uint8 [0, 255], ImageNet [-2.1, 2.6], unscaled float32, and bad shapes refused; [0, 1] float32 passes"


def focal_shift_and_intrinsics_recovery_geometry() -> str:
    h, w = 96, 128
    aspect_ratio = w / h
    fx_true = 1.25
    fy_true = fx_true * aspect_ratio
    shift_true = 0.45

    uv = normalized_view_plane_uv(w, h, aspect_ratio=aspect_ratio, dtype=torch.float32)
    diag = math.sqrt(1.0 + aspect_ratio ** 2)
    focal_true = fx_true * 2.0 * aspect_ratio / diag

    u = uv[..., 0]
    v = uv[..., 1]
    z_true = 1.8 + 0.5 * u + 0.35 * v + 0.4 * (u ** 2 + v ** 2)
    x_cam = (u / focal_true) * z_true
    y_cam = (v / focal_true) * z_true
    pts_unshifted = torch.stack([x_cam, y_cam, z_true - shift_true], dim=-1).unsqueeze(0)
    mask = torch.ones((1, h, w), dtype=torch.bool)

    focal_rec, shift_rec = recover_focal_shift(pts_unshifted, mask)
    assert abs(float(focal_rec.item()) - focal_true) < 1e-3, f"focal {focal_rec.item()} != {focal_true}"
    assert abs(float(shift_rec.item()) - shift_true) < 1e-3, f"shift {shift_rec.item()} != {shift_true}"

    pts_rec, depth_rec, intr_rec, _ = recover_intrinsics(
        pts_unshifted, mask, aspect_ratio=aspect_ratio
    )
    fx_rec = float(intr_rec[0, 0, 0].item())
    fy_rec = float(intr_rec[0, 1, 1].item())
    cx_rec = float(intr_rec[0, 0, 2].item())
    cy_rec = float(intr_rec[0, 1, 2].item())
    max_depth_err = float((depth_rec[0] - z_true).abs().max().item())
    assert abs(fx_rec - fx_true) < 1e-3, f"fx {fx_rec} != {fx_true}"
    assert abs(fy_rec - fy_true) < 1e-3, f"fy {fy_rec} != {fy_true}"
    assert cx_rec == 0.5 and cy_rec == 0.5, f"principal point {(cx_rec, cy_rec)} != (0.5, 0.5)"
    assert max_depth_err < 1e-3, f"max depth error {max_depth_err} >= 1e-3"
    assert torch.allclose(pts_rec[..., 2], depth_rec, atol=1e-5)

    fov_x_deg, fov_y_deg = fov_deg_from_intrinsics(intr_rec)
    expected_fov_x = math.degrees(2.0 * math.atan(0.5 / fx_true))
    assert abs(float(fov_x_deg.item()) - expected_fov_x) < 1e-2
    return (
        f"fx={fx_rec:.4f} (true {fx_true:.4f}), fy={fy_rec:.4f} (true {fy_true:.4f}), "
        f"shift={float(shift_rec.item()):.4f}m (true {shift_true:.4f}m), max_depth_err={max_depth_err:.2e}m"
    )


def end_to_end_tool_run_npz_record_and_sheet() -> str:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        vid = tmp / "src.mkv"
        msk = tmp / "fg.mkv"
        out_dir = tmp / "moge_out"
        _write_synthetic_rgb_video(vid, width=128, height=96, num_frames=6)
        fg_mask = np.zeros((6, 96, 128), dtype=bool)
        fg_mask[:, 24:72, 32:96] = True
        write_mask_video(msk, fg_mask)

        rc = measure_moge2_depth.main([
            "--source", str(vid),
            "--frames", "0,2,4",
            "--out-dir", str(out_dir),
            "--size", "128x96",
            "--box", "16,12,112,84",
            "--region", f"fg={msk}@0:cyan",
            "--checkpoint", "tiny",
            "--device", "cpu",
            "--cell", "160",
        ])
        assert rc == 0, f"expected exit 0, got {rc}"

        npz_path = out_dir / "depth.npz"
        json_path = out_dir / "record.json"
        sheet_path = out_dir / "sheet.jpg"
        assert npz_path.is_file() and json_path.is_file() and sheet_path.is_file()

        npz = np.load(npz_path)
        assert list(npz["frames"]) == [0, 2, 4], f"unexpected frames {list(npz['frames'])}"
        assert npz["depth"].shape == (3, 96, 128) and npz["depth"].dtype == np.float32
        assert bool(np.isfinite(npz["depth"]).all()) and float(npz["depth"].min()) > 0.0
        assert npz["mask"].shape == (3, 96, 128) and npz["mask"].dtype == bool
        assert npz["intrinsics"].shape == (3, 3, 3)
        assert list(npz["box"]) == [16, 12, 112, 84], f"unexpected box {list(npz['box'])}"

        rec = json.loads(json_path.read_text())
        assert rec["model"]["input_range"] == [0.0, 1.0], f"unexpected input_range {rec['model']['input_range']}"
        assert rec["frames"] == [0, 2, 4], f"unexpected record frames {rec['frames']}"
        assert rec["box"] == [16, 12, 112, 84], f"unexpected record box {rec['box']}"
        assert len(rec["per_frame"]) == 3
        for row in rec["per_frame"]:
            assert row["depth_min_m"] is not None and row["depth_median_m"] is not None
            assert "fg" in row["regions"] and row["regions"]["fg"]["pixels"] == (72 - 24) * (96 - 32)
        s = rec["summary"]
        assert len(s["fov_x_deg_least_median_most"]) == 3
        assert s["fov_x_deg_spread"] >= 0.0 and s["fov_x_deg_std"] >= 0.0

        sheet = np.asarray(Image.open(sheet_path))
        expected_h = 3 * (measure_moge2_depth.HEADER_H + 120)
        assert sheet.shape == (expected_h, 320, 3), f"unexpected sheet shape {sheet.shape}"
        left_panel = sheet[:, :160]
        right_panel = sheet[:, 160:]
        assert float(left_panel.std()) > 10.0 and float(right_panel.std()) > 10.0, "empty RGB or depth panel in sheet.jpg"
    return "depth.npz (3x96x128), record.json (intrinsics + region stats), and sheet.jpg (320x444) verified"


def state_dict_key_completeness() -> str:
    lic_path = REPO / "vendor" / "meta_moge2" / "LICENSE"
    assert lic_path.is_file(), "vendor/meta_moge2/LICENSE is missing"
    lic_bytes = lic_path.read_bytes()
    lic_sha = hashlib.sha256(lic_bytes).hexdigest()
    assert lic_sha == LICENCE_SHA256, f"vendor/meta_moge2/LICENSE hash {lic_sha} != {LICENCE_SHA256}"
    up_lic = REPO / "coderef" / "metadepth" / "LICENSE"
    if up_lic.is_file():
        assert lic_bytes == up_lic.read_bytes(), "vendor/meta_moge2/LICENSE does not match coderef/metadepth/LICENSE byte for byte"

    with torch.device("meta"):
        model = HyDenMoGe()
    sd = model.state_dict()
    assert len(sd) == CHECKPOINT_NUM_KEYS, f"expected {CHECKPOINT_NUM_KEYS} state_dict keys, got {len(sd)}"
    required_prefixes = {
        "cnn_encoder",
        "encoder",
        "image_feats",
        "mask_head",
        "neck",
        "points_head",
        "scale_feats",
        "scale_head",
    }
    actual_prefixes = {k.split(".")[0] for k in sd}
    assert actual_prefixes == required_prefixes, f"prefix mismatch: {actual_prefixes} vs {required_prefixes}"
    assert "encoder.backbone.mask_token" in sd and tuple(sd["encoder.backbone.mask_token"].shape) == (1, 1024)
    assert tuple(sd["image_feats.0.weight"].shape) == (1600, 1600, 3, 3)
    assert tuple(sd["scale_feats.0.weight"].shape) == (1312, 1312)

    if DEFAULT_CHECKPOINT.is_file():
        ckpt_sd = torch.load(str(DEFAULT_CHECKPOINT), map_location="meta", weights_only=True)
        assert set(sd.keys()) == set(ckpt_sd.keys()), "state_dict keys do not match checkpoint"
        for k, v in sd.items():
            assert tuple(v.shape) == tuple(ckpt_sd[k].shape), f"shape mismatch on {k}: {v.shape} vs {ckpt_sd[k].shape}"
        return f"all {len(sd)} state_dict keys and shapes match checkpoint on disk; vendor/meta_moge2/LICENSE hash verified"
    return f"all {len(sd)} state_dict keys and shapes verified on meta device; vendor/meta_moge2/LICENSE hash verified"


def refusals_leave_no_partial_files() -> str:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        vid = tmp / "src.mkv"
        _write_synthetic_rgb_video(vid, width=64, height=48, num_frames=4)

        # 1. Missing source video
        out_missing = tmp / "out_missing"
        refused(
            lambda: measure_moge2_depth.main([
                "--source", str(tmp / "no_such.mkv"),
                "--frames", "0",
                "--out-dir", str(out_missing),
                "--checkpoint", "tiny",
            ]),
            "REFUSED:",
            "source video not found",
        )
        assert not out_missing.exists()

        # 2. Bad --box
        out_box = tmp / "out_box"
        refused(
            lambda: measure_moge2_depth.main([
                "--source", str(vid),
                "--frames", "0",
                "--out-dir", str(out_box),
                "--size", "64x48",
                "--box", "0,0,80,48",
                "--checkpoint", "tiny",
            ]),
            "REFUSED:",
            "--box",
        )
        assert not out_box.exists()

        # 3. Frame out of range
        out_range = tmp / "out_range"
        refused(
            lambda: measure_moge2_depth.main([
                "--source", str(vid),
                "--frames", "0,9",
                "--out-dir", str(out_range),
                "--size", "64x48",
                "--checkpoint", "tiny",
            ]),
            "REFUSED:",
            "out of range",
        )
        assert not out_range.exists()

        # 4. Existing output file
        out_exists = tmp / "out_exists"
        out_exists.mkdir()
        (out_exists / "record.json").write_text("{}\n")
        refused(
            lambda: measure_moge2_depth.main([
                "--source", str(vid),
                "--frames", "0",
                "--out-dir", str(out_exists),
                "--size", "64x48",
                "--checkpoint", "tiny",
            ]),
            "REFUSED:",
            "already exists",
        )
        assert not (out_exists / "depth.npz").exists()
    return "missing source, bad --box, out-of-range frame, and existing output refused with no partial files"


if __name__ == "__main__":
    case("raw_rgb_input_guard", raw_rgb_input_guard)
    case("focal_shift_and_intrinsics_recovery_geometry", focal_shift_and_intrinsics_recovery_geometry)
    case("end_to_end_tool_run_npz_record_and_sheet", end_to_end_tool_run_npz_record_and_sheet)
    case("state_dict_key_completeness", state_dict_key_completeness)
    case("refusals_leave_no_partial_files", refusals_leave_no_partial_files)
    sys.exit(finish())
