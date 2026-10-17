#!/usr/bin/env python3
"""`bench/rebuild_delivery.py` on a synthetic fixture and refusal mutations: each case is a way the provenance, delivery,
gap-check, stack, review-pages, and answers-snapshot chain could skip a step, continue past a failure, hardcode a
session path or date, or leave a writable snapshot.

1. **`full_chain_with_clip_and_without_clip`**: on an 8-frame synthetic fixture (source, capture with track/parts/region,
   reference chain, piece render with painted change and mask picture, build + queue records, `laid.jsonl`, verdicts,
   answers, review items), running `rebuild_delivery.py --clip` executes `laid_verify`, `ledger`, `delivery_table`,
   `assemble`, `gap_check`, `stack`, `pages`, and `snapshot` in order, writing every `.out` log, the ledger, the table,
   the assembled clip + `.check.json`, the per-frame gap check, the stacked clip + `.1fps.` + `.stack.json`, `covered.json`,
   `TO_REVIEW.html` and `PROVENANCE.html` in `Video/<video>/`, and a `0444` read-only snapshot of `owner_answers.jsonl`.
   A second run without `--clip` at a later stamp discovers the existing clip, the previous/watched tables, the day
   folder, the extra captures, and the spots files, and runs `laid_verify`, `ledger`, `delivery_table`, `gap_check`,
   `pages`, and `snapshot` without re-encoding video.
2. **`discovery_rules_are_deterministic`**: `discover_table_dirs` includes immediate subfolders holding `t_*.json` and
   skips others; `discover_also_captures` keeps the newest capture per span group; `discover_previous_table` and
   `discover_watched_table` sort whole-span tables by stamp and skip `_joined_` per-shot tables and the current stamp;
   `discover_day_folder` picks the newest `day_*` holding `[0-9][0-9][0-9][0-9]_*` piece folders; `discover_existing_clip`
   picks the newest joined clip that has a `.mp4.check.json` and skips `_STACKED.mp4` and `.1fps.` files.
3. **`stops_at_first_refusal_and_names_the_step`**: a failure at `init`, `laid_verify`, `ledger`, `delivery_table`,
   `assemble`, `gap_check`, `stack`, `pages`, or `snapshot` stops the chain immediately, names the step that failed,
   leaves every later step unrun, and exits with `REFUSED:` and no Python traceback.
4. **`nothing_of_a_video_in_code`**: `bench/rebuild_delivery.py` contains no machine mount, home path, server port,
   Tailscale IP, video stem, or hardcoded session day.

Needs ffmpeg, ffprobe, and Pillow. No model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_rebuild_delivery.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import REPO, case, finish, needs, refused  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

import assemble_delivery as asm  # noqa: E402
import clip_folder  # noqa: E402
import laid as laid_mod  # noqa: E402
import rebuild_delivery as rd  # noqa: E402

TOOL = Path(rd.__file__).resolve()
W, H, FRAMES = 256, 192, 8
RATE = "24"
PICTURE = "nb_colors=5:c0=0x905030:c1=0x2a5fa0:c2=0xc8b450:c3=0x3c6432:c4=0xa04682:x0=40:y0=30:x1=220:y1=160:seed=7"
RECT = (40, 40, 104, 120)
QUEUED_AT, WRITTEN = "2026-10-16T10:00:00+00:00", 1792146600.0
TEXT = (
    "subject_definitions:\n"
    "<Subject 1> is the woman of <Picture 1>, who follows the figure in <Video 1>.\n"
    "<Subject 2> is the movements of the figure in <Video 1>: a closer view of a plain grey body model.\n\n"
    "summary:\n<Subject 1> performs <Subject 2>.\n"
)


def _run_ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _make_source(path: Path) -> None:
    _run_ffmpeg(
        "-f", "lavfi", "-i", f"gradients=size={W}x{H}:rate={RATE}:speed=0.03:{PICTURE}",
        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
        "-frames:v", str(FRAMES), "-shortest",
        "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
        "-c:v", "libx264", "-crf", "12", "-g", "250", "-pix_fmt", "yuv420p", *asm.BT709_TAGS,
        "-c:a", "aac", "-ac", "2", "-b:a", "96k", str(path),
    )


def _make_piece_from_source(source: Path, piece: Path, rect: tuple[int, int, int, int]) -> None:
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-an", "-i", str(source),
            "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-f", "rawvideo", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(FRAMES, H, W, 3).copy()
    x0, y0, x1, y1 = rect
    arr[:, y0:y1, x0:x1] = (arr[:, y0:y1, x0:x1].astype(np.int32) + 128) % 256
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", RATE, "-i", "-",
            "-vf", asm.TO_BT709, "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", *asm.BT709_TAGS,
            str(piece),
        ],
        input=arr.tobytes(),
        check=True,
    )


def _make_flat_video(path: Path, w: int, h: int, frames: int, expr: str = "60+15*N") -> None:
    _run_ffmpeg(
        "-f", "lavfi", "-i", f"nullsrc=size={w}x{h}:rate={RATE},format=gray,geq=lum='{expr}'",
        "-frames:v", str(frames), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
    )


def _write_png_graph(path: Path, graph: dict) -> None:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("prompt", json.dumps(graph))
    Image.new("RGB", (4, 4)).save(path, pnginfo=info)


def _build_fixture(root: Path, video: str = "clip") -> dict[str, Path]:
    share = root / "share"
    inputs = root / "inputs"
    data = root / "data"
    vdir = data / video
    for d in (share, inputs, vdir, vdir / "verdicts", vdir / "review"):
        d.mkdir(parents=True, exist_ok=True)

    src_video = inputs / f"{video}.mp4"
    _make_source(src_video)
    (inputs / "still_a.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    about = {
        "name": f"{video}.mp4",
        "bytes": src_video.stat().st_size,
        "frames": FRAMES,
        "rate": 24.0,
        "size": [W, H],
        "head_sha256": rd.sha256_file(src_video),
    }
    clip_folder.write_manifest(vdir, {"source": about, "derived": about})

    shots = {
        "schema": "h3_shots/1",
        "video": f"{video}.mp4",
        "subjects": ["a"],
        "span": [0, FRAMES - 1],
        "shots": [{"first": 0, "last": FRAMES - 1, "who": {"a": {"present": "yes"}}}],
    }
    plan = {
        "schema": "h3_plan/1",
        "subjects": {"a": {"region": "face and hair", "parts": ["hair", "face_and_neck"]}},
        "aliases": {"singer": "a"},
        "load_prefixes": {"a": ["pa_"]},
    }
    (vdir / "shots.json").write_text(json.dumps(shots, indent=2) + "\n")
    (vdir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    # Capture folder with track, parts, and run region
    cap = data / f"2026-10-16_{video}_0_{FRAMES - 1}_capture"
    (cap / "subjects" / "a").mkdir(parents=True, exist_ok=True)
    (cap / "runs" / "pa_0000").mkdir(parents=True, exist_ok=True)
    mask_bool = np.zeros((FRAMES, H, W), dtype=bool)
    x0, y0, x1, y1 = RECT
    mask_bool[:, y0:y1, x0:x1] = True
    packed = np.packbits(mask_bool, axis=-1)
    np.savez_compressed(
        cap / "subjects" / "a" / "masks__p.npz",
        track=packed,
        covered=np.ones(FRAMES, dtype=bool),
        parts=packed,
    )
    (cap / "subjects" / "a" / "part_grades.json").write_text(json.dumps({"frames": []}) + "\n")
    for mkv in ("track.mkv", "parts_held.mkv"):
        _make_flat_video(cap / "subjects" / "a" / mkv, W, H, FRAMES)
    np.savez_compressed(
        cap / "runs" / "pa_0000" / "region.npz",
        carried=packed,
        read=np.ones(FRAMES, dtype=bool),
        region=np.zeros((FRAMES, 1, 1), dtype=bool),
    )
    cap_manifest = {
        "name": cap.name,
        "clip": f"{video}.mp4",
        "first_frame": 0,
        "frames": FRAMES,
        "size": [W, H],
        "subjects": [{"label": "a", "sightings": [{"by": "p", "part_is_made_of": ["Face_Neck", "Hair", "Upper_Lip"]}]}],
        "runs": [{"name": "pa_0000", "subject": "a", "left_as_the_source_with_no_mask": [], "left_as_the_source_across_a_cut": []}],
    }
    (cap / "manifest.json").write_text(json.dumps(cap_manifest, indent=2) + "\n")

    # Reference video chain in the frozen form (body model -> cut to head -> face points drawn)
    ref_dir = share / "Video" / video / "day_2026-10-16" / "0000_pa_0000_inputs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    mesh_mp4 = ref_dir / "mesh_00001.mp4"
    cut_mp4 = ref_dir / "cut.mp4"
    ref_mp4 = ref_dir / "ref.mp4"
    for p in (mesh_mp4, cut_mp4, ref_mp4):
        _make_flat_video(p, W, H, FRAMES, "30+20*N")
    _write_png_graph(ref_dir / "mesh_00001.png", {"1": {"class_type": "SaveVideo", "inputs": {}}})
    (ref_dir / "cut.json").write_text(
        json.dumps({"box": [10, 20, 110, 95], "scale": [2.0, 2.0], "made_from": {"reference_video": str(mesh_mp4)}}) + "\n"
    )
    (ref_dir / "ref.json").write_text(
        json.dumps({
            "line_width_px": 1.5,
            "parts_left_out": ["mouth"],
            "a_dot_at_every_point": False,
            "made_by": "s1",
            "made_from": {"body_model_video": str(cut_mp4)},
        }) + "\n"
    )

    # Piece render + _with_mask.mp4 + .png saved graph
    piece_rel = f"Video/{video}/day_2026-10-16/0000_pa_0000/pa_0000_00001.mp4"
    piece_path = share / piece_rel
    piece_path.parent.mkdir(parents=True, exist_ok=True)
    _make_piece_from_source(src_video, piece_path, RECT)
    mask_pic = piece_path.with_name("pa_0000_00001_with_mask.mp4")
    _make_flat_video(mask_pic, W, H * 2, FRAMES, "40+20*N")

    track_path = str(cap / "subjects" / "a" / "track.mkv")
    parts_path = str(cap / "subjects" / "a" / "parts_held.mkv")
    load_node = lambda v, at=0.0, count=FRAMES: {  # noqa: E731
        "class_type": "VHS_LoadVideoFFmpegPath",
        "inputs": {"video": v, "start_time": at, "frame_load_cap": count},
    }
    graph = {
        "1": load_node(str(src_video), 0.0),
        "2": load_node(track_path),
        "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0]}},
        "4": load_node(parts_path),
        "5": {"class_type": "ImageToMask", "inputs": {"image": ["4", 0]}},
        "6": load_node(str(ref_mp4)),
        "9": {
            "class_type": "MiniMaxH3MaskedSource",
            "inputs": {
                "frames": ["1", 0],
                "mask": ["3", 0],
                "parts": ["5", 0],
                "replace": "the wired parts",
                "grow_pixels": 16,
                "motion_reference": "a video I wire",
                "motion_video": ["6", 0],
            },
        },
        "10": {
            "class_type": "MiniMaxH3AudioFreezeSong",
            "inputs": {
                "source": ["9", 0],
                "window_frames": 141,
                "context_frames": 90,
                "seed": 7,
                "context_noise": 0.8,
                "prompt": TEXT,
                "continue_from": "",
            },
        },
        "11": {"class_type": "LoadImage", "inputs": {"image": "still_a.png"}},
    }
    _write_png_graph(piece_path.with_suffix(".png"), graph)
    os.utime(piece_path, (WRITTEN, WRITTEN))
    os.utime(mask_pic, (WRITTEN, WRITTEN))

    # Build and queue records
    job_dir = data / f"2026-10-16_{video}_pa_0000_job"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "pa_0000.txt").write_text(TEXT)
    job_file = job_dir / "pa_0000.job.json"
    job_file.write_text(json.dumps({"job": {"name": "pa_0000", "video": video, "subjects": [{"label": "a"}], "left_out": []}}) + "\n")
    built = {
        "load": {
            "name": "pa_0000",
            "subject": "a",
            "carried": "held",
            "first": 0,
            "frames": FRAMES,
            "motion": "mesh",
            "motion_file": str(ref_mp4),
            "still": "still_a.png",
            "seed": 7,
        },
        "commit": "abc1234",
        "capture": str(cap),
        "files": {"track": [track_path, 0], "parts": [parts_path, 0], "others": []},
        "settings": {"window": {"value": 141}, "context": {"value": 90}, "context_noise": {"value": 0.8}},
        "output_folder": f"Video/{video}/day_2026-10-16/0000_pa_0000",
        "uncommitted_node_code": [],
        "text": {"file": str(job_dir / "pa_0000.txt")},
        "job_file": str(job_file),
    }
    (job_dir / "pa_0000.render.built.json").write_text(json.dumps(built, indent=2) + "\n")
    queued = {
        "commit": "abc1234",
        "queued_at": QUEUED_AT,
        "gate": {"verdict": "clear", "overridden": []},
        "look": {"by": "a reader"},
        "status": "success",
        "files": [piece_rel],
    }
    (job_dir / "pa_0000.render.queued.json").write_text(json.dumps(queued, indent=2) + "\n")

    # Write laid.jsonl and verify via laid.read
    laid_path = vdir / "laid.jsonl"
    laid_entry = {
        "schema": laid_mod.SCHEMA,
        "first": 0,
        "last": FRAMES - 1,
        "subject": "a",
        "load": "pa_0000",
        "file": piece_rel,
        "sha256": laid_mod.sha256(piece_path),
        "owner": "unconfirmed",
        "by": "synthetic fixture render covering shot 0-7",
        "frames": [[0, FRAMES - 1]],
        "added": "2026-10-16T12:00:00Z",
        "added_by": "test_reader",
    }
    laid_path.write_text(json.dumps(laid_entry) + "\n")
    assert len(laid_mod.read(str(laid_path), str(share), check_bytes=True)) == 1, "synthetic laid.jsonl must contain 1 valid entry"

    # Verdicts, owner_answers.jsonl, and review/items.json
    answers_path = vdir / "owner_answers.jsonl"
    answers_path.write_text(
        json.dumps({"at": "2026-10-16T12:00:00Z", "who": "owner", "row": "pa_0000", "what": "words", "words": "looks clean"}) + "\n"
    )
    verdicts_path = vdir / "verdicts" / "verdicts.jsonl"
    verdicts_path.write_text(
        json.dumps({
            "at": "2026-10-16T12:00:00Z",
            "row": "pa_0000",
            "what": "words",
            "words": "looks clean",
            "source": "answers.jsonl:1",
            "render": piece_rel,
        }) + "\n"
    )
    items_path = vdir / "review" / "items.json"
    items_path.write_text(
        json.dumps({
            "items": [
                {"id": "pa_0000", "name": "pa_0000", "first": "0", "frames": str(FRAMES), "state": "open", "look": "her face on 0-7"},
            ]
        }, indent=2) + "\n"
    )

    return {
        "root": root,
        "share": share,
        "inputs": inputs,
        "data": data,
        "vdir": vdir,
        "source": src_video,
        "piece": piece_path,
        "laid": laid_path,
        "answers": answers_path,
        "verdicts": verdicts_path,
        "items": items_path,
    }


def full_chain_with_clip_and_without_clip() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_fixture(Path(tmp))
        video = "clip"
        day = "2026-10-17"
        stamp1 = "2026-10-17T1800Z"

        # 1. Run with --clip: all 8 steps run in order
        ns1 = rd.build_parser().parse_args([
            "--video", video,
            "--clip",
            "--share", str(fx["share"]),
            "--inputs", str(fx["inputs"]),
            "--data", str(fx["data"]),
            "--day", day,
            "--stamp", stamp1,
            "--stack-work", str(fx["root"] / "stack_work_1"),
            "--today", day,
        ])
        out1 = rd.rebuild(ns1, repo=REPO)
        assert out1["steps"] == list(rd.STEPS_WITH_CLIP), f"--clip step order mismatch: got {out1['steps']}"

        out_dir = fx["data"] / f"{day}_{video}_delivery_tables"
        for name in (
            f"laid_verify_{video}_{stamp1}.out",
            f"ledger_{video}_{stamp1}.out",
            f"ledger_{video}_{stamp1}.json",
            f"ledger_{video}_{stamp1}.md",
            f"ledger_{video}_{stamp1}_summary.md",
            f"table_{video}_0000_0007_{stamp1}.out",
            f"t_{video}_0000_0007_{stamp1}.txt",
            f"t_{video}_0000_0007_{stamp1}.json",
            f"reading_{video}_0000_0007_{stamp1}.md",
            f"assemble_{video}_0000_0007_{stamp1}.out",
            f"laid_after_{video}_{stamp1}.out",
            f"coverage_{video}_0000_0007_{stamp1}.out",
            f"frame_coverage_{video}_0000_0007_{stamp1}.json",
            f"frame_coverage_{video}_0000_0007_{stamp1}.md",
            f"stack_{video}_0000_0007_{stamp1}.out",
            f"stack_check_{video}_0000_0007_{stamp1}.out",
            f"covered_{video}_{stamp1}.out",
            f"page_to_review_{video}_{stamp1}.out",
            f"page_provenance_{video}_{stamp1}.out",
            f"snapshot_{video}_{stamp1}.out",
            f"rebuild_{video}_{stamp1}.json",
        ):
            assert (out_dir / name).is_file(), f"expected step artifact or log {name} in {out_dir}"

        joined_mp4 = fx["share"] / "Video" / video / f"day_{day}" / f"{video}_0000_0007_joined_{stamp1}_from_laid.mp4"
        stacked_mp4 = joined_mp4.with_name(joined_mp4.stem + "_STACKED.mp4")
        assert joined_mp4.is_file() and Path(str(joined_mp4) + ".check.json").is_file(), "joined clip or .check.json missing"
        assert stacked_mp4.is_file() and stacked_mp4.with_name(stacked_mp4.stem + ".stack.json").is_file(), "stacked clip or .stack.json missing"
        assert (fx["vdir"] / "review" / "covered.json").is_file(), "covered.json missing"
        assert (fx["share"] / "Video" / video / "TO_REVIEW.html").is_file(), "TO_REVIEW.html missing"
        assert (fx["share"] / "Video" / video / "PROVENANCE.html").is_file(), "PROVENANCE.html missing"

        snap1 = fx["vdir"] / "owner_answers_snapshots" / f"owner_answers_{stamp1}.jsonl"
        assert snap1.is_file(), "owner_answers snapshot missing"
        assert stat.S_IMODE(os.stat(snap1).st_mode) == 0o444, f"snapshot mode must be 0o444 (read-only), got {oct(stat.S_IMODE(os.stat(snap1).st_mode))}"
        assert rd.sha256_file(snap1) == rd.sha256_file(fx["answers"]), "snapshot sha256 does not match owner_answers.jsonl"

        # 2. Run without --clip at a later stamp: reuses the existing joined clip + .check.json and discovers stamp1's table
        stamp2 = "2026-10-17T1805Z"
        ns2 = rd.build_parser().parse_args([
            "--video", video,
            "--share", str(fx["share"]),
            "--inputs", str(fx["inputs"]),
            "--data", str(fx["data"]),
            "--day", day,
            "--stamp", stamp2,
            "--today", day,
        ])
        out2 = rd.rebuild(ns2, repo=REPO)
        assert out2["steps"] == list(rd.STEPS_WITHOUT_CLIP), f"no-clip step order mismatch: got {out2['steps']}"
        assert not out2["clip_built"], f"expected clip_built=False without --clip, got {out2['clip_built']}"
        assert Path(out2["discovered"]["previous_table"] or "").name == f"t_{video}_0000_0007_{stamp1}.json", (
            f"expected previous_table to resolve to stamp1's table, got {out2['discovered']['previous_table']}"
        )
        assert Path(out2["discovered"]["watched_table"] or "").name == f"t_{video}_0000_0007_{stamp1}.json", (
            f"expected watched_table to resolve to stamp1's table, got {out2['discovered']['watched_table']}"
        )
        snap2 = fx["vdir"] / "owner_answers_snapshots" / f"owner_answers_{stamp2}.jsonl"
        assert snap2.is_file() and stat.S_IMODE(os.stat(snap2).st_mode) == 0o444, "second snapshot missing or writable"

    return (
        "--clip runs laid_verify -> ledger -> delivery_table -> assemble -> gap_check -> stack -> pages -> snapshot; "
        "without --clip reuses the existing clip's .check.json, links the prior table, updates both pages, and writes a 0444 snapshot"
    )


def discovery_rules_are_deterministic() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data"
        share = root / "share"
        inputs = root / "inputs"
        video = "clip"
        for p in (data / video, inputs):
            p.mkdir(parents=True)

        # source_info prefers source.name over derived.name while reading frames from derived
        (inputs / f"{video}.mp4").write_bytes(b"mp4")
        (inputs / f"{video}_working.mov").write_bytes(b"mov")
        (data / video / "shots.json").write_text(json.dumps({"shots": [{"first": 0, "last": 5}]}))
        (data / video / "manifest.json").write_text(json.dumps({
            "source": {"name": f"{video}.mp4"},
            "derived": {"name": f"{video}_working.mov", "frames": 8},
        }))
        src_p, s0, s1 = rd.source_info(data, video, inputs)
        assert (src_p.name, s0, s1) == (f"{video}.mp4", 0, 7), (
            f"expected source_info to prefer source.name {video}.mp4 with span 0-7 from derived.frames, got {(src_p.name, s0, s1)}"
        )

        # Table dirs + trial subfolder
        d0 = data / f"2026-10-13_{video}_delivery_tables"
        d1 = data / f"2026-10-15_{video}_delivery_tables"
        d2 = data / f"2026-10-16_{video}_delivery_tables"
        trial = d2 / "2026-10-16_trial"
        empty_sub = d2 / "no_tables_here"
        for p in (d0, d1, d2, trial, empty_sub):
            p.mkdir(parents=True)
        (d0 / f"t_{video}_0000_0007_THROWAWAY_whole_mended_2026-10-13T2121Z.json").write_text("{}")
        (d1 / f"t_{video}_0000_0007_2026-10-15T1200Z.json").write_text("{}")
        (d2 / f"t_{video}_0000_0007_2026-10-16T1200Z.json").write_text("{}")
        (d2 / f"t_{video}_0000_0007_joined_shot0.json").write_text("{}")
        (trial / f"t_{video}_0000_0007_trial.json").write_text("{}")

        dirs = rd.discover_table_dirs(data, video)
        assert [p.name for p in dirs] == [d0.name, d1.name, d2.name, trial.name], (
            f"expected table dirs and trial subfolder with t_*.json, got {[p.name for p in dirs]}"
        )

        # Whole-span tables exclude _joined_, THROWAWAY_*, and exclude_stamp
        prev = rd.discover_previous_table(data, video, 0, 7, exclude_stamp="2026-10-16T1200Z")
        watched = rd.discover_watched_table(data, video, 0, 7, exclude_stamp="2026-10-17T0000Z")
        assert prev is not None and prev.name == f"t_{video}_0000_0007_2026-10-15T1200Z.json", (
            f"expected newest prior whole-span table before 2026-10-16T1200Z (ignoring THROWAWAY and _joined_), got {prev}"
        )
        assert watched is not None and watched.name == f"t_{video}_0000_0007_2026-10-15T1200Z.json", (
            f"expected earliest whole-span watched table (ignoring THROWAWAY and _joined_), got {watched}"
        )
        latest = rd.discover_previous_table(data, video, 0, 7, exclude_stamp="2026-10-17T0000Z")
        assert latest is not None and latest.name == f"t_{video}_0000_0007_2026-10-16T1200Z.json", (
            f"expected newest whole-span table 2026-10-16T1200Z (ignoring THROWAWAY and _joined_), got {latest}"
        )

        # Extra captures grouped by (stem_without_trailing_digits, length_capture), newest date/rev kept per group
        for name in (
            f"2026-10-10_{video}_0_100_capture",
            f"2026-10-12_{video}_25_100_capture",
            f"2026-10-10_{video}_rest_a_1039_capture",
            f"2026-10-10_{video}_rest_a3_1039_capture",
            f"2026-10-10_{video}_rest_a4_1039_capture",
            f"2026-10-10_{video}_rest_b_1466_capture",
            f"2026-10-10_{video}_rest_b2_1466_capture",
            f"2026-10-14_{video}_200_300_capture",
        ):
            c = data / name
            c.mkdir(parents=True)
            (c / "manifest.json").write_text("{}")
        caps = rd.discover_also_captures(data, video)
        assert [c.name for c in caps] == [
            f"2026-10-12_{video}_25_100_capture",
            f"2026-10-14_{video}_200_300_capture",
            f"2026-10-10_{video}_rest_a4_1039_capture",
            f"2026-10-10_{video}_rest_b2_1466_capture",
        ], f"expected newest capture per span group (rest_a4 over rest_a, rest_b2 over rest_b), got {[c.name for c in caps]}"

        # Spots discovery
        for sname in ("joined_clip_spots_2026-10-13.json", "joined_clip_spots_2026-10-14.json"):
            (data / video / sname).write_text("{}")
        spots = [p.name for p in rd.discover_spots(data, video)]
        assert spots == [
            "joined_clip_spots_2026-10-13.json",
            "joined_clip_spots_2026-10-14.json",
        ], f"expected sorted joined_clip_spots files, got {spots}"

        # Day folder discovery prefers newest day_* with [0-9][0-9][0-9][0-9]_* piece folders
        (share / "Video" / video / "day_2026-10-15" / "0000_pa").mkdir(parents=True)
        (share / "Video" / video / "day_2026-10-16" / "0100_pb").mkdir(parents=True)
        (share / "Video" / video / "day_2026-10-17").mkdir(parents=True)  # no piece folder inside
        day_dir = rd.discover_day_folder(share, video, "2026-10-17")
        assert day_dir.name == "day_2026-10-16", (
            f"expected newest day_* folder containing 4-digit piece folders (day_2026-10-16), got {day_dir.name}"
        )

        # Existing clip discovery ignores _STACKED.mp4, .1fps., and clips without .check.json
        day16 = share / "Video" / video / "day_2026-10-16"
        c1 = day16 / f"{video}_0000_0007_joined_2026-10-16T1000Z.mp4"
        c2 = day16 / f"{video}_0000_0007_joined_2026-10-16T1200Z.mp4"
        c_no_check = day16 / f"{video}_0000_0007_joined_2026-10-16T1400Z.mp4"
        c_stacked = day16 / f"{video}_0000_0007_joined_2026-10-16T1500Z_STACKED.mp4"
        for p in (c1, c2, c_no_check, c_stacked):
            p.write_bytes(b"mp4")
        for p in (c1, c2, c_stacked):
            Path(str(p) + ".check.json").write_text("{}")
        found_clip = rd.discover_existing_clip(share, video, 0, 7)
        assert found_clip == c2, f"expected newest unstacked clip with .check.json ({c2.name}), got {found_clip}"

    return "table folders, extra captures, spots, prior/watched tables, day piece folders, and existing clips are discovered deterministically"


def stops_at_first_refusal_and_names_the_step() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_fixture(Path(tmp))
        video = "clip"
        day = "2026-10-17"
        stamp = "2026-10-17T1900Z"

        # CLI refusals on bad --day, bad --video, and missing laid file: no traceback, exits with REFUSED: step init failed
        out = refused([sys.executable, str(TOOL), "--video", video, "--share", str(fx["share"]), "--inputs", str(fx["inputs"]),
                       "--data", str(fx["data"]), "--day", "10-17-2026"])
        assert "REFUSED: step init failed" in out and "--day" in out, f"expected init refusal naming --day, got {out!r}"

        out = refused([sys.executable, str(TOOL), "--video", "bad/stem", "--share", str(fx["share"]), "--inputs", str(fx["inputs"]),
                       "--data", str(fx["data"])])
        assert "REFUSED: step init failed" in out and "--video" in out, f"expected init refusal naming --video, got {out!r}"

        out = refused([sys.executable, str(TOOL), "--video", video, "--share", str(fx["share"]), "--inputs", str(fx["inputs"]),
                       "--data", str(fx["data"]), "--laid", str(fx["root"] / "missing_laid.jsonl")])
        assert "REFUSED: step init failed" in out and "laid file is not there" in out, f"expected init refusal naming missing laid file, got {out!r}"

        # Step-by-step refusal injection: failing at step i stops immediately and never invokes step i+1
        def make_dummy_outputs(step_name: str, cmd: list[str]) -> None:
            if step_name == "ledger":
                for flag in ("--json", "--md", "--summary"):
                    Path(cmd[cmd.index(flag) + 1]).write_text("{}" if flag == "--json" else "# ledger\n")
            elif step_name == "delivery_table":
                t_txt = Path(cmd[cmd.index("--table") + 1])
                t_json = Path(cmd[cmd.index("--json") + 1])
                r_md = Path(cmd[cmd.index("--reading") + 1])
                t_txt.write_text("# table\n")
                r_md.write_text("# reading\n")
                cov_p = t_json.parent / f"frame_coverage_{video}_0000_0007_{stamp}.json"
                clip_p = fx["share"] / "Video" / video / f"day_{day}" / f"{video}_0000_0007_joined_{stamp}_from_laid.mp4"
                t_json.write_text(json.dumps({
                    "assembly": {
                        "whole": True,
                        "command": f"python bench/assemble_delivery.py --out {clip_p}",
                        "stack": {
                            "whole": True,
                            "coverage_json": str(cov_p),
                            "build": f"python bench/stack_delivery.py build --out {clip_p.with_name(clip_p.stem + '_STACKED.mp4')}",
                            "check": f"python bench/stack_delivery.py check --out {clip_p.with_name(clip_p.stem + '_STACKED.mp4')}",
                        },
                    }
                }))
            elif step_name == "assemble" and "assemble_delivery.py" in " ".join(cmd):
                clip_p = Path(cmd[cmd.index("--out") + 1])
                clip_p.parent.mkdir(parents=True, exist_ok=True)
                clip_p.write_bytes(b"mp4")
                Path(str(clip_p) + ".check.json").write_text("{}")
            elif step_name == "gap_check":
                Path(cmd[cmd.index("--json") + 1]).write_text("{}")
                Path(cmd[cmd.index("--md") + 1]).write_text("# cov\n")
            elif step_name == "pages":
                out_p = Path(cmd[cmd.index("--out") + 1])
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text("<html></html>" if out_p.suffix == ".html" else "{}")

        subproc_steps = ("laid_verify", "ledger", "delivery_table", "assemble", "gap_check", "stack", "pages")
        for idx, fail_step in enumerate(subproc_steps):
            called: list[str] = []
            step_out_dir = fx["root"] / f"out_fail_{fail_step}"

            def runner(step: str, cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str], _fail=fail_step, _called=called) -> tuple[int, str]:
                if not _called or _called[-1] != step:
                    _called.append(step)
                if step == _fail:
                    return 2, f"synthetic refusal in {_fail}\n"
                make_dummy_outputs(step, cmd)
                return 0, f"ok {step}\n"

            ns = rd.build_parser().parse_args([
                "--video", video,
                "--clip",
                "--share", str(fx["share"]),
                "--inputs", str(fx["inputs"]),
                "--data", str(fx["data"]),
                "--out-dir", str(step_out_dir),
                "--day", day,
                "--stamp", stamp,
            ])
            try:
                rd.rebuild(ns, repo=REPO, runner=runner)
            except rd.Refused as e:
                assert e.step == fail_step, f"expected refusal step {fail_step!r}, got {e.step!r} ({e})"
                assert f"synthetic refusal in {fail_step}" in str(e), f"expected refusal message from {fail_step}, got {e}"
            else:
                raise AssertionError(f"expected Refused at {fail_step}, but rebuild continued past failure")
            assert called == list(subproc_steps[: idx + 1]), (
                f"expected chain to stop immediately at {fail_step}, but ran {called}"
            )

        # Snapshot refusal when an existing snapshot for the same stamp has different bytes
        snap_dir = fx["vdir"] / "owner_answers_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        bad_snap = snap_dir / f"owner_answers_{stamp}.jsonl"
        bad_snap.write_text("tampered\n")
        try:
            rd.snapshot_owner_answers(fx["data"], video, stamp)
        except rd.Refused as e:
            assert e.step == "snapshot" and "differs from" in str(e), f"expected snapshot sha256 mismatch refusal, got {e}"
        else:
            raise AssertionError("expected snapshot refusal on mismatched existing snapshot")

    return "every step refusal stops the chain immediately, names the step, leaves later steps unrun, and prints no traceback"


def nothing_of_a_video_in_code() -> str:
    src = TOOL.read_text()
    forbidden = ("/mnt/", "/home/", "http://", "https://", ":8765", "100.87.", "fun_", '"fun"', "'fun'", "day_2026")
    found = [tok for tok in forbidden if tok in src]
    assert not found, f"forbidden tokens in {TOOL.name}: {found}"
    return "bench/rebuild_delivery.py names no mount path, home folder, URL, port, IP, video stem, or session day"


if __name__ == "__main__":
    case("full_chain_with_clip_and_without_clip", full_chain_with_clip_and_without_clip)
    case("discovery_rules_are_deterministic", discovery_rules_are_deterministic)
    case("stops_at_first_refusal_and_names_the_step", stops_at_first_refusal_and_names_the_step)
    case("nothing_of_a_video_in_code", nothing_of_a_video_in_code)
    sys.exit(finish())
