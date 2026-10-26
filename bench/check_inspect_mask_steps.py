#!/usr/bin/env python3
"""Deterministic check for `bench/inspect_mask_steps.py` across all eight forensic categories `[0]`–`[7]`,
multi-window `(1, 4, 4, 4, 4)` cadence, face-only vs. hair+face subjects, sub-span offsets, and `--strips` rendering.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import inspect_mask_steps as ims


def _write_mkv(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w = masks.shape
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    vw = cv2.VideoWriter(str(path), fourcc, 24.0, (w, h), True)
    for i in range(n):
        u8 = (masks[i].astype(np.uint8) * 255)
        bgr = np.stack([u8, u8, u8], axis=-1)
        vw.write(bgr)
    vw.release()


def _write_color_mp4(path: Path, n: int, h: int, w: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 24.0, (w, h), True)
    for i in range(n):
        frame = np.full((h, w, 3), 60 + (i % 40), dtype=np.uint8)
        vw.write(frame)
    vw.release()


def main() -> int:
    # 1. Verify (1, 4, 4, 4, 4) cyclic cadence
    assert ims.FRAME_PER_TOKEN == (1, 4, 4, 4, 4), f"unexpected FRAME_PER_TOKEN: {ims.FRAME_PER_TOKEN}"
    assert ims.run_lengths(12) == [1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4]
    assert sum(ims.run_lengths(12)) == 39

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        in_dir = tmp / "input"
        out_dir = tmp / "output"
        data_dir = tmp / "data"
        vid_out = out_dir / "Video" / "fun"
        in_dir.mkdir(parents=True)
        vid_out.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        h, w = 128, 128
        first = 100
        n = 9  # 2 latent steps of (1, 4, 4) plus 2 pad_after frames -> 1 + 4 + 4 + 2 = 11 window frames (steps: 1, 4, 4, 2)
        _write_color_mp4(in_dir / "fun_joined_from_laid.mp4", 120, h, w)

        # Write laid.jsonl with one prior pass (added before build) and one later pass (added way in the future)
        laid_dir = data_dir / "fun"
        laid_dir.mkdir(parents=True)
        laid_rows = [
            {
                "load": "third_fp_0100a",
                "subject": "third",
                "frames": [[100, 108]],
                "added": "2026-09-25T10:00:00Z",
                "file": "day/third_fp_0100a.mp4",
            },
            {
                "load": "lead_fp_0100z",
                "subject": "lead",
                "frames": [[100, 108]],
                "added": "2099-01-01T00:00:00Z",
                "file": "day/lead_fp_0100z.mp4",
            },
        ]
        (laid_dir / "laid.jsonl").write_text("\n".join(json.dumps(r) for r in laid_rows) + "\n")

        # Prior capture `2026-10-20_third_fp_0100a` to test prior_pass_drift
        p_cap = data_dir / "2026-10-20_third_fp_0100a"
        (p_cap / "runs" / "third_fp_0100a").mkdir(parents=True)
        (p_cap / "subjects" / "second").mkdir(parents=True)
        p_reg = np.zeros((n, h // 16, w // 16), dtype=bool)
        p_reg[:, 1:5, 1:5] = True
        np.savez_compressed(p_cap / "runs" / "third_fp_0100a" / "region.npz", region=p_reg)
        p_tr_sec = np.zeros((n, h, w), dtype=bool)
        p_tr_sec[:, 20:60, 20:60] = True  # 1600 px on raw
        np.savez_compressed(
            p_cap / "subjects" / "second" / "masks__shot_c_0100a.npz",
            track=np.packbits(p_tr_sec.astype(np.uint8), axis=-1),
        )
        (p_cap / "manifest.json").write_text(
            json.dumps({
                "first_frame": first,
                "frames": n,
                "size": [w, h],
                "subjects": [{"label": "second", "sightings": [{"by": "shot_c_0100a"}]}],
                "runs": [{"name": "third_fp_0100a", "subject": "third"}],
            })
        )

        # Current capture `2026-10-25_second_hfp_0100b`
        cap = data_dir / "2026-10-25_second_hfp_0100b"
        run_dir = cap / "runs" / "second_hfp_0100b"
        subj_dir = cap / "subjects" / "second"
        third_dir = cap / "subjects" / "third"
        str_dir = cap / "subjects" / "strangers"
        for d in (run_dir, subj_dir, third_dir, str_dir):
            d.mkdir(parents=True)

        # Build masks for `second` (hair+face):
        # f=0..3 (100..103): clean 1600 px
        # f=4 (104): edge-shrunk 900 px (< 0.75 * 1600)
        # f=5 (105): 0 px carried inside active step 2 (blue-only + swallowed by strangers + collapsed from 1600 px in raw)
        # f=6 (106): 20 px speck inside active step 2 (speck-in-blue)
        # f=7,8 (107,108): 0 px carried, while PAD_9,10 have 1600 px in step 3 (pad-bleed + blue-only)
        track = np.zeros((n, h, w), dtype=bool)
        carried = np.zeros((n, h, w), dtype=bool)
        head_track = np.zeros((n, h, w), dtype=bool)
        for i in range(4):
            track[i, 20:60, 20:60] = True
            head_track[i, 20:60, 20:60] = True
            carried[i, 20:60, 20:60] = True
        # f=4: shrunk to 30x30 = 900 px
        track[4, 20:50, 20:50] = True
        head_track[4, 20:50, 20:50] = True
        carried[4, 20:50, 20:50] = True
        # f=5: collapsed to 0 px on Pass 2
        # f=6: speck 4x5 = 20 px
        track[6, 20:40, 20:40] = True
        head_track[6, 20:24, 20:25] = True
        carried[6, 20:24, 20:25] = True

        np.savez_compressed(
            subj_dir / "masks__shot_c_0100b.npz",
            track=np.packbits(track.astype(np.uint8), axis=-1),
            parts=np.packbits(carried.astype(np.uint8), axis=-1),
        )
        _write_mkv(subj_dir / "head_track__shot_c_0100b.mkv", head_track)
        _write_mkv(subj_dir / "parts_held__shot_c_0100b.mkv", carried)
        _write_mkv(run_dir / "carried.mkv", carried)

        # Third subject track (so [6] excludes `third` and only flags unnamed stranger)
        third_tr = np.zeros((n, h, w), dtype=bool)
        third_tr[:, 70:100, 70:100] = True
        np.savez_compressed(
            third_dir / "masks__shot_c_0100b.npz",
            track=np.packbits(third_tr.astype(np.uint8), axis=-1),
        )

        # Strangers keep-out on share (with sub-span offset: starts at frame 95, 5 frames before capture first=100)
        str_first = 95
        str_n = 14
        raw_str = np.zeros((str_n, h, w), dtype=bool)
        # On sf=105 (index 10 in raw_str), raw_str swallows second's 20:60, 20:60 head box
        raw_str[10, 20:60, 20:60] = True
        ko_dir = vid_out / "keep_outs" / "shot_c_0100b"
        ko_dir.mkdir(parents=True)
        _write_mkv(ko_dir / "shot_c_0100b_strangers.mkv", raw_str)

        # Whole-frame classes on share: class 15 (Upper_Lip, non-face_ids/non-Apparel) on 20:60, 20:60 so class filter recovers it
        wc = np.zeros((str_n, h, w), dtype=np.uint8)
        wc[10, 20:60, 20:60] = 15
        np.savez_compressed(ko_dir / "shot_c_0100b_strangers_classes.npz", classes=wc, first=np.int64(str_first))

        # SAM 3.1 people.npz: on sf=106 (frame index 11 from shots_first=95), an unnamed stranger at 32:64, 32:64 inside blue region (16:64, 16:64)
        p_mask = np.zeros((2, h, w), dtype=bool)
        p_mask[0, 32:64, 32:64] = True  # unnamed stranger inside region on sf=106, disjoint from track[6, 20:40, 20:40] and raw_str[11]=0
        p_mask[1, 70:100, 70:100] = True  # `third` (tracked subject, must be excluded by [6]!)
        np.savez_compressed(
            ko_dir / "shot_c_0100b_people.npz",
            masks=np.packbits(p_mask.astype(np.uint8), axis=-1),
            frame=np.array([11, 11], dtype=np.int32),
            score=np.array([0.92, 0.95], dtype=np.float32),
            person=np.array([4, 2], dtype=np.int32),
        )
        (ko_dir / "shot_c_0100b_strangers.json").write_text(
            json.dumps({
                "first_source_frame": str_first,
                "source": "fun_0000_6394_24p_sound_in_step.mov",  # intentional mismatch vs fun_joined_from_laid.mp4
                "whole_frame_classes": "shot_c_0100b_strangers_classes.npz",
                "people_file": "shot_c_0100b_people.npz",
            })
        )
        np.savez_compressed(
            str_dir / "masks__shot_c_0100b.npz",
            track=np.packbits(np.zeros((n, h, w), dtype=np.uint8), axis=-1),
        )

        # Region cells `[n, h//16, w//16]`
        reg_cells_arr = np.zeros((n, h // 16, w // 16), dtype=bool)
        reg_cells_arr[:, 1:4, 1:4] = True
        np.savez_compressed(
            run_dir / "region.npz",
            region=reg_cells_arr,
            carried=np.packbits(carried.astype(np.uint8), axis=-1),
        )

        # Planned window region file (11 window frames = 9 real + 2 pad_after; 4 latent steps: 1, 4, 4, 2)
        pred_dir = cap / "predicted"
        pred_dir.mkdir(parents=True)
        w_tok = np.zeros((4, h // 16, w // 16), dtype=np.uint8)
        w_tok[:, 1:4, 1:4] = 1
        w_mask = np.zeros((11, h, w), dtype=bool)
        w_mask[:9] = carried
        # Step 2 holds wf 5,6,7,8 (sf 105,106,107,108). Make wf 7,8 = 0 px, and make Step 3 hold wf 7,8 + PAD?
        # Wait: with (1, 4, 4, 4):
        # Step 0 = wf 0 (sf 100)
        # Step 1 = wf 1..4 (sf 101..104)
        # Step 2 = wf 5..8 (sf 105..108) -> wf 5=0px (blue-only), wf 6=20px (speck), wf 7,8=0px (blue-only)
        # To test pad_bleed on Step 2, let n = 7 (sf 100..106) + 2 pad_after (wf 7, 8 = PAD_7, PAD_8)!
        # Let's check: if n = 7 and pad_after = 2 (total_wf = 9, 3 latent steps: 1, 4, 4):
        # Step 0 = wf 0 (sf 100)
        # Step 1 = wf 1..4 (sf 101..104)
        # Step 2 = wf 5, 6 (sf 105, 106, both 0 px) + wf 7, 8 (PAD_7, PAD_8, both 900 px)!
        # Then Step 2 is BOTH a pad_bleed_step AND makes sf 105, 106 blue_only_frames!
        # Let's put speck (20 px) on wf 3 (sf 103) inside Step 1!
        carried[3] = False
        carried[3, 20:24, 20:25] = True  # 20 px speck on sf=103 inside Step 1 (max=1600 px)
        carried[6] = False  # 0 px on sf=106 so Step 2 real frames (sf 105, 106) are both 0 px!
        n_test = 7
        np.savez_compressed(
            subj_dir / "masks__shot_c_0100b.npz",
            track=np.packbits(track[:n_test].astype(np.uint8), axis=-1),
            parts=np.packbits(carried[:n_test].astype(np.uint8), axis=-1),
        )
        _write_mkv(subj_dir / "head_track__shot_c_0100b.mkv", head_track[:n_test])
        _write_mkv(subj_dir / "parts_held__shot_c_0100b.mkv", carried[:n_test])
        _write_mkv(run_dir / "carried.mkv", carried[:n_test])
        np.savez_compressed(
            third_dir / "masks__shot_c_0100b.npz",
            track=np.packbits(third_tr[:n_test].astype(np.uint8), axis=-1),
        )
        np.savez_compressed(
            str_dir / "masks__shot_c_0100b.npz",
            track=np.packbits(np.zeros((n_test, h, w), dtype=np.uint8), axis=-1),
        )
        np.savez_compressed(
            run_dir / "region.npz",
            region=reg_cells_arr[:n_test],
            carried=np.packbits(carried[:n_test].astype(np.uint8), axis=-1),
        )

        w_tok3 = np.zeros((3, h // 16, w // 16), dtype=np.uint8)
        w_tok3[:, 1:4, 1:4] = 1
        w_mask9 = np.zeros((9, h, w), dtype=bool)
        w_mask9[:n_test] = carried[:n_test]
        w_mask9[7, 20:50, 20:50] = True  # PAD_7 has 900 px
        w_mask9[8, 20:50, 20:50] = True  # PAD_8 has 900 px
        np.savez_compressed(
            pred_dir / "second_hfp_0100b_window_1_planned_region.npz",
            tokens=w_tok3,
            mask=np.packbits(w_mask9.astype(np.uint8), axis=-1),
            meta=json.dumps({
                "first_frame": 0,
                "frames": 9,
                "width": w,
                "height": h,
                "pad_before": 0,
                "pad_after": 2,
            }),
        )
        (pred_dir / "second_hfp_0100b_plan.json").write_text(
            json.dumps({
                "windows": [{
                    "number": 1,
                    "first_frame": 0,
                    "frames": 9,
                    "trim": 0,
                    "first_written_frame": 0,
                    "last_written_frame": 6,
                    "pad_before": 0,
                    "pad_after": 2,
                }]
            })
        )

        (cap / "manifest.json").write_text(
            json.dumps({
                "name": "2026-10-25_second_hfp_0100b",
                "clip": "fun_joined_from_laid.mp4",
                "first_frame": first,
                "frames": n_test,
                "size": [w, h],
                "owners": {
                    "file": "owners.npz",
                    "labels": ["second", "third"],
                    "claimed_by_more_than_one_px": 420,
                    "contested_px": 420,
                },
                "subjects": [
                    {
                        "label": "second",
                        "sightings": [{
                            "by": "shot_c_0100b",
                            "shots_first_source_frame": str_first,
                            "part_is_made_of": ["Face_Neck", "Hair"],
                            "part_sizes": {"floor_px": 256},
                            "inputs": {
                                "track": {"file": "shot_c_0100b_second_track.mkv", "written": "2026-10-01T00:00:00"},
                                "parts": {"file": "shot_c_0100b_second_part.mkv", "written": "2026-10-25T15:00:00"},
                            },
                        }],
                    },
                    {
                        "label": "third",
                        "sightings": [{"by": "shot_c_0100b"}],
                    },
                    {
                        "label": "strangers",
                        "sightings": [{
                            "by": "shot_c_0100b",
                            "shots_first_source_frame": str_first,
                            "track": "shot_c_0100b_strangers.mkv",
                        }],
                    },
                ],
                "runs": [{
                    "name": "second_hfp_0100b",
                    "subject": "second",
                    "plan": "second_hfp_0100b_plan.json",
                    "files": ["second_hfp_0100b_window_1_planned_region.npz"],
                }],
            })
        )

        env = {**os.environ, "H3_INPUT_DIR": str(in_dir), "H3_OUTPUT_DIR": str(out_dir)}
        strip_jpg = tmp / "strip.jpg"
        rep_json = tmp / "report.json"
        res = subprocess.run(
            [
                sys.executable,
                str(HERE / "inspect_mask_steps.py"),
                "--capture",
                str(cap),
                "--strips",
                str(strip_jpg),
                "--json",
                str(rep_json),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        rep = json.loads(rep_json.read_text())

        # Verify [0] Provenance
        prov = rep["provenance"]
        assert prov["pass_number"] == 2, f"expected Pass #2, got {prov['pass_number']}"
        assert len(prov["prior_laid_passes"]) == 1 and prov["prior_laid_passes"][0]["load"] == "third_fp_0100a"
        assert len(prov["later_laid_passes"]) == 1 and prov["later_laid_passes"][0]["load"] == "lead_fp_0100z"
        assert any("strangers source=" in m for m in prov["provenance_mismatches"])
        assert len(prov["stale_at_build"]) == 1 and prov["stale_at_build"][0]["input"] == "track"
        assert prov["depth_status"]["depth_used"] is False
        assert prov["depth_status"]["contested_px"] == 420
        assert len(prov["prior_pass_drift"]) == 1
        assert any(d["source_frame"] == 105 for d in prov["prior_pass_drift"][0]["drifted_frames"])

        # Verify [1] Pad bleed
        assert len(rep["pad_bleed_steps"]) == 1 and rep["pad_bleed_steps"][0]["step"] == 2

        # Verify [2] Blue-only frames (sf 105, 106)
        assert [r["source_frame"] for r in rep["blue_only_frames"]] == [105, 106]

        # Verify [3] Speck-in-blue frames (sf 103)
        assert [r["source_frame"] for r in rep["speck_in_region_frames"]] == [103]

        # Verify [4] Edge-shrunk whole-head frames (sf 104)
        assert [r["source_frame"] for r in rep["edge_shrunk_whole_frames"]] == [104]

        # Verify [5] Strangers self-swallow (sf 105, recoverable=1600 px)
        sw = [r for r in rep["strangers_self_swallow_frames"] if r["source_frame"] == 105]
        assert len(sw) == 1 and sw[0]["recoverable_with_class_filter_px"] == 1600

        # Verify [6] Missed SAM strangers in blue region (person#4 on sf=106, while person#2 `third` is excluded!)
        assert len(rep["missed_sam_strangers_in_region"]) == 1
        assert rep["missed_sam_strangers_in_region"][0]["source_frame"] == 106
        assert rep["missed_sam_strangers_in_region"][0]["person_id"] == 4

        # Verify [7] Strip image written
        assert strip_jpg.is_file() and strip_jpg.stat().st_size > 1000

        # Verify clean refusal (no traceback) on missing manifest
        bad = subprocess.run(
            [sys.executable, str(HERE / "inspect_mask_steps.py"), "--capture", str(tmp / "nonexistent")],
            env=env,
            capture_output=True,
            text=True,
        )
        assert bad.returncode != 0
        assert "Traceback" not in bad.stderr
        assert "error: missing" in bad.stderr

    print("ok: check_inspect_mask_steps passed all [0]-[7] forensic and refusal checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
