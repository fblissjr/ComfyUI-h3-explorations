#!/usr/bin/env python3
"""That `vendor_config/` still is what the release ships, and still parses.

Two failure modes, and they need different evidence:

1. **The files drifted.** Somebody edited a vendored config by hand, or a copy
   went wrong. Caught by hashing against `vendor_config/sha256.json`, which is
   what they hashed to when copied on 2026-08-21. Runs everywhere, needs no
   weights.
2. **The release changed under us.** Caught only by comparing against the
   published repo, which is 200 GB and lives outside this tree. When it is on
   disk this check diffs against it; when it is not, it says so in as many
   words. **A skipped comparison must not look like a passed one** -- that is
   the whole reason this check prints what it examined rather than only what
   failed.

Point it at the release with `--release <path>` or `H3_RELEASE_DIR`. Neither is
written into the repo: it is a path on somebody's disk, not a property of the
project.

The readers are exercised too. A vendored file that hashes correctly and then
fails to yield a bound is still a broken vendored file, and `vendor_config.py`
raises rather than returning a default precisely so this can catch it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VENDOR = REPO / "vendor_config"
# Where each vendored file came from inside the release.
ORIGIN = {
    "tokenizer_config.json": "tokenizer/tokenizer_config.json",
    "preprocessor_config.json": "processor/preprocessor_config.json",
    "video_preprocessor_config.json": "processor/video_preprocessor_config.json",
    "fl2va_model_index.json": "FL2VA/model_index.json",
    "ref2va_model_index.json": "Ref2VA/model_index.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=os.environ.get("H3_RELEASE_DIR"),
                    help="the published MiniMax H3 repo, if it is on this box")
    args = ap.parse_args()

    failures = []
    recorded = json.loads((VENDOR / "sha256.json").read_text())

    print(f"vendored files: {len(ORIGIN)} expected")
    for name in sorted(ORIGIN):
        path = VENDOR / name
        if not path.exists():
            failures.append(f"{name} is missing from vendor_config/")
            print(f"  FAIL  {name}: missing")
            continue
        want = recorded.get(name)
        got = sha(path)
        if want is None:
            failures.append(f"{name} has no recorded hash")
            print(f"  FAIL  {name}: no recorded hash")
        elif want != got:
            failures.append(f"{name} changed since it was vendored")
            print(f"  FAIL  {name}: {got[:16]} != recorded {want[:16]}")
        else:
            print(f"  ok    {name}  {got[:16]}")

    print("\nreaders:")
    import vendor_config as vc
    try:
        toks = vc.additional_special_tokens()
        lo_i, hi_i = vc.image_pixel_bounds()
        lo_v, hi_v = vc.video_pixel_bounds()
        video_geometry = vc.video_patch_geometry()
        parts = vc.partition_tasks()
        print(f"  ok    {len(toks)} special token(s), image {lo_i}..{hi_i}, "
              f"video {lo_v}..{hi_v}")
        print(f"  ok    video patch geometry {video_geometry}")
        print(f"  ok    partitions {parts}")
    except Exception as exc:  # a vendored file that parses but says nothing
        failures.append(f"a reader failed: {exc}")
        print(f"  FAIL  reader: {exc}")

    print("\nagainst the release:")
    if not args.release:
        print("  SKIPPED -- no --release and no H3_RELEASE_DIR. The files were "
              "verified against each other, NOT against the release. This is "
              "not a pass on question 2.")
    else:
        rel = Path(args.release).expanduser()
        if not rel.exists():
            failures.append(f"--release {rel} does not exist")
            print(f"  FAIL  {rel} does not exist")
        else:
            for name, origin in sorted(ORIGIN.items()):
                src = rel / origin
                if not src.exists():
                    failures.append(f"{origin} missing from the release")
                    print(f"  FAIL  {origin}: not in the release")
                elif sha(src) != sha(VENDOR / name):
                    failures.append(f"{name} differs from the release")
                    print(f"  FAIL  {name}: differs from {origin}")
                else:
                    print(f"  ok    {name} == {origin}")

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
