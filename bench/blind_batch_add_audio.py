#!/usr/bin/env python3
"""Replace a blind batch's silent singles with their muxed `-audio.mp4` sources, in place.

An instrument, not a check. `bench/blind_batch.py` copied the silent
`X.mp4` for every single until 2026-09-04, and the singles are the only place
a judge hears the audio (stacks map video only), so the first ladder batch
was scored deaf. This repairs such a batch without opening its key.

## How a clip finds its source

The key is sealed and stays sealed: the source is found by content. For each
`clip_NN.mp4` in the batch's MANIFEST, the row's graph names the
`filename_prefix` directory on the output share; every `*_NNNNN.mp4` in it
that is not itself a `-audio.mp4` and has the clip's byte size is compared
byte for byte, and exactly one must match. That match is the control: a clip
whose source cannot be found, or matches two files, or whose source has no
`-audio.mp4` sibling, refuses the whole batch before any file is touched.
The muxed sibling is then copied over the clip, mtime preserved, and the
result is probed for an audio stream.

Arm labels pass through this process in memory (a source filename carries
one) and are never printed: the only output is a count.

Usage:

    H3_COMFY_OUTPUT=<share> python bench/blind_batch_add_audio.py \\
        --batch <share>/Video/blind/ladder_2026-09-03
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _paths import comfy_output  # noqa: E402
from blind_batch import has_audio_stream as has_audio, prefix_of, single_source  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--batch", required=True, help="the blind batch directory holding MANIFEST.json")
    ap.add_argument("--output-root", default=None,
                    help="ComfyUI output directory; default H3_COMFY_OUTPUT or folder_paths")
    args = ap.parse_args()
    root = Path(args.output_root) if args.output_root else comfy_output()
    if root is None or not root.is_dir():
        sys.exit("refuse: no output root; set H3_COMFY_OUTPUT or pass --output-root")
    batch = Path(args.batch)
    manifest = json.loads((batch / "MANIFEST.json").read_text())
    jsonl = REPO / manifest["jsonl"]
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]

    plan: list[tuple[Path, Path]] = []
    already = 0
    for item in manifest["clips"]:
        clip = batch / item["clip"]
        if has_audio(clip):
            already += 1
            continue
        r = rows[item["row"]]
        pfx = prefix_of(REPO / r["graph"], r["label"])
        if pfx is None:
            sys.exit(f"refuse: {r['graph']} has no filename_prefix to search by")
        folder = root / Path(pfx).parent
        size = clip.stat().st_size
        cands = [c for c in folder.glob("*_[0-9][0-9][0-9][0-9][0-9].mp4")
                 if c.stat().st_size == size and filecmp.cmp(clip, c, shallow=False)]
        if len(cands) != 1:
            sys.exit(f"refuse: {clip.name}: {len(cands)} byte-identical sources under "
                     f"{folder.relative_to(root)}; expected one")
        src = single_source(cands[0])
        if src == cands[0]:
            sys.exit(f"refuse: {clip.name}: its source has no -audio.mp4 sibling")
        plan.append((src, clip))
    for src, clip in plan:
        shutil.copy2(src, clip)
        if not has_audio(clip):
            sys.exit(f"refuse: {clip.name} carries no audio stream after the copy; batch is half-repaired")
    print(f"{len(plan)} singles replaced with their muxed sources, {already} already carried audio -> {batch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
