#!/usr/bin/env python3
"""Hold the candidate pool's media verification to a standard it can fail.

`build_h3_calibration_pool.py::media_status` is the gate that decides whether a
row's declared media actually exists at the pinned revision and hashes to what
the row declares. Its predecessor did not exist: video rows were exempt from
every media check, so twenty `video-reference` rows entered the accepted pool
naming nineteen files of which sixteen were absent from the local snapshot. A
declared hash nobody recomputes is a claim.

Two arms, and the second is the one that matters:

1. **Live arm.** Every row of the emitted pool and its exclusion complement is
   re-verified against the pinned snapshot. Every declared file must be present
   and hash-correct, and no pooled row may carry a media rejection.
2. **Violation arm.** The same function is handed a scratch tree carrying four
   deliberate defects -- a deleted file, a corrupted file, a row declaring no
   hash, and a mismatched *video* file. Each must return a rejection reason
   naming that file. A gate that cannot be shown failing is not evidence that
   it passed, and the fourth case is the escaped defect itself.

The live arm hashes about 1.2 GiB and takes tens of seconds. CPU only, no model.
Run it with the ComfyUI venv python (`docs/comfy_notes.md`).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_h3_calibration_pool import (  # noqa: E402
    EXCLUDED,
    POOL,
    media_status,
    pinned_snapshot,
)


def live_arm(root: Path) -> tuple[int, list[str]]:
    """Re-verify every declared file of every emitted row."""
    if not POOL.is_file() or not EXCLUDED.is_file():
        return 0, ["pool outputs absent; run build_h3_calibration_pool.py first"]

    failures: list[str] = []
    checked = 0
    for path, pooled in ((POOL, True), (EXCLUDED, False)):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "media_verification" not in row:
                failures.append(
                    f"row {row['id']} predates media verification; rebuild the pool"
                )
                continue
            reason, records = media_status(root, row)
            checked += len(records)
            bad = [r for r in records if r["status"] != "verified"]
            if pooled and (reason or bad):
                failures.append(
                    f"pooled row {row['id']} carries unverified media: {reason or bad}"
                )
            elif not pooled and bad and "declared media did not verify" not in (
                row.get("exclusion_reason") or ""
            ):
                failures.append(
                    f"excluded row {row['id']} has unverified media but was "
                    f"excluded for a different reason: {row.get('exclusion_reason')!r}"
                )
    return checked, failures


def violation_arm() -> list[str]:
    """Four deliberate defects. Each must produce a rejection reason."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "media").mkdir()
        good = root / "media" / "present.bin"
        good.write_bytes(b"real media bytes")
        good_sha = hashlib.sha256(good.read_bytes()).hexdigest()

        clean = {
            "id": "control-clean",
            "images": ["media/present.bin"],
            "videos": [],
            "media_sha256": {"media/present.bin": good_sha},
        }
        reason, records = media_status(root, clean)
        if reason is not None:
            failures.append(f"clean control was rejected: {reason}")
        if [r["status"] for r in records] != ["verified"]:
            failures.append(f"clean control did not verify: {records}")

        corrupt = root / "media" / "corrupt.bin"
        corrupt.write_bytes(b"real media bytes")
        corrupt_sha = hashlib.sha256(corrupt.read_bytes()).hexdigest()
        corrupt.write_bytes(b"tampered media bytes")

        video = root / "media" / "asvideo.mp4"
        video.write_bytes(b"different bytes")

        cases = [
            ("deleted file", "media/deleted.mp4", good_sha, "videos"),
            ("corrupted file", "media/corrupt.bin", corrupt_sha, "videos"),
            ("undeclared hash", "media/present.bin", None, "images"),
            ("mismatched video", "media/asvideo.mp4", good_sha, "videos"),
        ]
        for label, rel, declared, field in cases:
            row = {
                "id": f"control-{label.replace(' ', '-')}",
                "images": [],
                "videos": [],
                "media_sha256": {} if declared is None else {rel: declared},
            }
            row[field] = [rel]
            reason, records = media_status(root, row)
            if reason is None:
                failures.append(f"{label}: media_status returned green")
                continue
            if rel not in reason:
                failures.append(f"{label}: reason does not name {rel}: {reason}")
            if any(r["status"] == "verified" for r in records):
                failures.append(f"{label}: a record still says verified: {records}")
            print(f"  violation '{label}' -> {reason}")
    return failures


def main() -> int:
    root, revision = pinned_snapshot()
    print(f"pinned snapshot revision {revision}")

    print("violation arm:")
    failures = violation_arm()

    print("live arm: recomputing every declared media hash")
    checked, live_failures = live_arm(root)
    failures.extend(live_failures)
    print(f"  {checked} declared media files recomputed")

    if failures:
        for message in failures:
            print(f"RED: {message}")
        return 1
    print("GREEN: every declared media file verified, and the gate fails on a "
          "deleted, corrupted, undeclared, or mismatched-video file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
