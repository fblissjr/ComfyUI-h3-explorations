#!/usr/bin/env python3
"""Side-by-side and top-to-bottom video stacking utility for paired evaluation.

Automatically selects the optimal stacking layout based on canvas aspect ratio:
  - Widescreen / Landscape (16:9, 4:3, 21:9) -> Stacked VERTICALLY (Top / Bottom)
    (Avoids creating unwieldy 2700px+ ultrawide windows on standard displays)
  - Portrait (9:16, 3:4) -> Stacked HORIZONTALLY (Left / Right)
    (Combines two tall portrait clips into a balanced landscape frame)
  - Square (1:1) -> Stacked HORIZONTALLY (Left / Right)

Includes customizable text overlays, blind randomization, and format normalization.

Usage:
    # Auto-layout comparison:
    python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 -o comparison.mp4

    # With custom labels:
    python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --label1 "Arm A (LoRA)" --label2 "Arm B (Checkpoint)"

    # Force specific layout:
    python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --layout horizontal
    python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --layout vertical

    # Create a blind evaluation comparison video + keyfile:
    python bench/stack_eval_clips.py arm_a.mp4 arm_b.mp4 --blind --keyfile blind_key.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402


def probe_video(path: str | Path) -> dict:
    """Probe video dimensions, duration, and frame rate via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-of", "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    r_fps = stream.get("r_frame_rate", "24/1")
    if "/" in r_fps:
        num, den = r_fps.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 24.0
    else:
        fps = float(r_fps)
    return {"width": width, "height": height, "fps": fps, "aspect": width / height}


def build_stacked_video(
    clip1: str | Path,
    clip2: str | Path,
    output: str | Path,
    layout: str = "auto",
    label1: str | None = "Clip 1",
    label2: str | None = "Clip 2",
    font_size: int = 28,
    crf: int = 18,
) -> Path:
    """Combine two clips into a stacked side-by-side or top-bottom video with overlays."""
    clip1_path = Path(clip1).resolve()
    clip2_path = Path(clip2).resolve()
    out_path = Path(output).resolve()

    if not clip1_path.is_file():
        raise FileNotFoundError(f"Clip 1 not found: {clip1_path}")
    if not clip2_path.is_file():
        raise FileNotFoundError(f"Clip 2 not found: {clip2_path}")

    info1 = probe_video(clip1_path)
    info2 = probe_video(clip2_path)

    w1, h1, aspect1 = info1["width"], info1["height"], info1["aspect"]
    w2, h2 = info2["width"], info2["height"]

    # Determine layout
    if layout in ("auto", None):
        # Landscape / wide -> vertical stacking (top / bottom)
        if aspect1 >= 1.2:
            resolved_layout = "vertical"
        # Portrait -> horizontal stacking (left / right)
        elif aspect1 <= 0.9:
            resolved_layout = "horizontal"
        # Square -> horizontal
        else:
            resolved_layout = "horizontal"
    elif layout in ("horizontal", "hstack", "side-by-side", "h"):
        resolved_layout = "horizontal"
    elif layout in ("vertical", "vstack", "top-bottom", "v"):
        resolved_layout = "vertical"
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Build FFmpeg filtergraph
    # 1. Scale clip 2 to match clip 1's geometry if different
    # 2. Add text labels if requested
    # 3. Stack horizontally or vertically

    filters = []

    # Format normalization & scaling
    if resolved_layout == "horizontal":
        # Match heights
        filters.append(f"[0:v]scale=-2:{h1}[v0]")
        filters.append(f"[1:v]scale=-2:{h1}[v1]")
    else:
        # Match widths
        filters.append(f"[0:v]scale={w1}:-2[v0]")
        filters.append(f"[1:v]scale={w1}:-2[v1]")

    # Drawtext overlays
    def make_drawtext(label: str, pos_label: str) -> str:
        escaped = label.replace(":", "\\:").replace("'", "\\'")
        if pos_label == "top-left":
            pos = "x=20:y=20"
        elif pos_label == "bottom-left":
            pos = "x=20:y=h-th-20"
        else:
            pos = "x=20:y=20"
        return (
            f"drawtext=text='{escaped}':{pos}:fontsize={font_size}:fontcolor=white:"
            f"box=1:boxcolor=black@0.65:boxborderw=10"
        )

    if label1:
        dt1 = make_drawtext(label1, "top-left")
        filters.append(f"[v0]{dt1}[v0_labeled]")
        v0_ref = "[v0_labeled]"
    else:
        v0_ref = "[v0]"

    if label2:
        dt2 = make_drawtext(label2, "top-left")
        filters.append(f"[v1]{dt2}[v1_labeled]")
        v1_ref = "[v1_labeled]"
    else:
        v1_ref = "[v1]"

    # Stacking
    if resolved_layout == "horizontal":
        filters.append(f"{v0_ref}{v1_ref}hstack=inputs=2[vout]")
    else:
        filters.append(f"{v0_ref}{v1_ref}vstack=inputs=2[vout]")

    filter_str = ";".join(filters)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(clip1_path),
        "-i", str(clip2_path),
        "-filter_complex", filter_str,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    print(f"[stack_eval] Building comparison video:")
    print(f"  Clip 1: {clip1_path.name} ({w1}x{h1}, aspect {aspect1:.2f})")
    print(f"  Clip 2: {clip2_path.name} ({w2}x{h2})")
    print(f"  Layout: {resolved_layout.upper()} ({'Top/Bottom' if resolved_layout == 'vertical' else 'Side-by-Side'})")
    print(f"  Labels: {label1!r} vs {label2!r}")
    print(f"  Output: {out_path}")

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"[stack_eval] Generated comparison video: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("clip1", help="First video file (or Arm A)")
    parser.add_argument("clip2", help="Second video file (or Arm B)")
    parser.add_argument("-o", "--output", default=None, help="Output comparison video path")
    parser.add_argument(
        "--layout",
        choices=["auto", "horizontal", "vertical", "hstack", "vstack"],
        default="auto",
        help="Stacking layout (default: auto -> vertical for landscape, horizontal for portrait)",
    )
    parser.add_argument("--label1", default="Clip 1", help="Label for clip 1 (default: 'Clip 1')")
    parser.add_argument("--label2", default="Clip 2", help="Label for clip 2 (default: 'Clip 2')")
    parser.add_argument("--no-labels", action="store_true", help="Omit text overlay labels")
    parser.add_argument("--blind", action="store_true", help="Randomize order of Clip 1 and Clip 2 for blind evaluation")
    parser.add_argument("--keyfile", default=None, help="Path to write sealed keyfile when using --blind")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality (default: 18)")
    parser.add_argument("--font-size", type=int, default=28, help="Overlay font size (default: 28)")

    args = parser.parse_args()

    c1 = args.clip1
    c2 = args.clip2

    l1 = None if args.no_labels else args.label1
    l2 = None if args.no_labels else args.label2

    if args.blind:
        arms = [(c1, "arm_a"), (c2, "arm_b")]
        random.shuffle(arms)
        c1, name1 = arms[0]
        c2, name2 = arms[1]
        l1 = "Clip 1" if not args.no_labels else None
        l2 = "Clip 2" if not args.no_labels else None

        key_data = {
            "Clip 1": {"source": str(c1), "arm": name1},
            "Clip 2": {"source": str(c2), "arm": name2},
        }
        if args.keyfile:
            key_path = Path(args.keyfile)
            key_path.write_text(json.dumps(key_data, indent=2) + "\n")
            print(f"[stack_eval] Sealed blind keyfile written: {key_path}")

    if not args.output:
        p1 = Path(c1).stem
        p2 = Path(c2).stem
        out_name = f"comparison_{p1}_vs_{p2}.mp4"
        out_dir = _paths.comfy_output()
        if out_dir is None:
            raise SystemExit(_paths.describe("ComfyUI output", "H3_COMFY_OUTPUT"))
        out_path = out_dir / "Video" / out_name
    else:
        out_path = Path(args.output)

    build_stacked_video(
        c1,
        c2,
        out_path,
        layout=args.layout,
        label1=l1,
        label2=l2,
        font_size=args.font_size,
        crf=args.crf,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
