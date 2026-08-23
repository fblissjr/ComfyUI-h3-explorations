#!/usr/bin/env python3
"""Stage two generated clips and build the requested side-by-side API graph.

The graph is the executable form of the standalone comparison workflow:
two ``VHS_LoadVideo`` upload nodes feed KJNodes ``ImageConcatMulti`` (right,
match size), which feeds ``VHS_VideoCombine`` at 24 fps. Audio is intentionally
not connected so one clip cannot silently become the comparison clock.

Run through uv. ``--left`` and ``--right`` are copied into ComfyUI's input
folder under opaque names; the manifest retains the source mapping for later
unblinding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(filename: str) -> dict:
    return {
        "class_type": "VHS_LoadVideo",
        "inputs": {
            "video": filename,
            "force_rate": 0.0,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "AnimateDiff",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--pair", required=True,
                        help="opaque pair label, for example i2va_pair_01")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    for source in (args.left, args.right):
        if not source.is_file():
            raise SystemExit(f"missing comparison clip: {source}")
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    for side, source in (("a", args.left), ("b", args.right)):
        destination = args.input_dir / f"h3_encoder_ab_{args.pair}_{side}.mp4"
        shutil.copy2(source, destination)
        staged.append((side, source, destination))

    graph = {
        "1": _loader(staged[0][2].name),
        "2": _loader(staged[1][2].name),
        "3": {
            "class_type": "ImageConcatMulti",
            "inputs": {
                "inputcount": 2,
                "image_1": ["1", 0],
                "direction": "right",
                "match_image_size": True,
                "image_2": ["2", 0],
            },
        },
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "frame_rate": 24.0,
                "loop_count": 0,
                "filename_prefix": f"Video/h3_encoder_ab_{args.pair}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 13,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    graph_path = args.out_dir / f"{args.pair}_api.json"
    graph_path.write_text(json.dumps(graph, indent=2) + "\n")
    manifest = {
        "pair": args.pair,
        "left": {
            "source": str(staged[0][1]),
            "staged": staged[0][2].name,
            "sha256": _sha256(staged[0][1]),
        },
        "right": {
            "source": str(staged[1][1]),
            "staged": staged[1][2].name,
            "sha256": _sha256(staged[1][1]),
        },
        "graph": graph_path.name,
        "graph_sha256": _sha256(graph_path),
        "audio": "intentionally disconnected",
    }
    manifest_path = args.out_dir / f"{args.pair}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(graph_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
