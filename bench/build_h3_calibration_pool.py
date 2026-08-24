#!/usr/bin/env python3
"""Build the deterministic candidate pool for AWQ v2 calibration and holdout.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no model,
no GPU. Reads the H3-IR dataset from the local Hugging Face cache.

This emits a *pool*, not a manifest. The pool is every row eligible for
calibration or holdout, each carrying its primary role, its overlay properties,
its hashes and its rights evidence. Selecting N rows from the pool is a later
step that needs a feasibility-measured N; nothing here picks a population size,
because the first preflight's error was choosing bucket counts before measuring.

Three things this does differently from the rejected preflight:

- **Primary roles are mutually exclusive.** An earlier draft of the plan gave
  percentages for overlapping categories -- a wide/tall row is often also
  multi-image, dialogue-marked and audio-labelled -- which cannot be a
  partition. Roles are assigned by a declared priority order; everything else is
  an overlay counted across the partition, never a bucket.
- **Role comes from the dataset, not from prose.** `channel`, `videos` and the
  IR's own "is the first/last frame" declaration decide the role. The rejected
  preflight inferred task type from prose patterns.
- **Every excluded row is emitted with a reason**, so the pool and its
  complement together account for every source row.

Determinism: the snapshot revision is pinned from the cache's own `refs/main`
rather than taken from whichever snapshot a glob returns first; rows are keyed by
the dataset's own `id` and sorted by it; no output depends on dict iteration
order. There is no sampling here, so there is no seed.

Four corrections from review, each of which had made the pool provisional:

1. **Geometry is read from the real image files**, not imported from the
   rejected preflight's `source_inventory.jsonl`. Depending on a quarantined
   artifact meant deleting it would have silently removed this pool's evidence.
2. **The revision is pinned**, not globbed.
3. **Media grouping is connected components over individual media hashes**, not
   a hash of the whole media set. Two rows sharing one image must land on the
   same side of a split even when their other media differ; set-hashing misses
   exactly that case and undercounts the constraint.
4. **Picture role is per picture, not per row.** H3-IR mixes them: 40 of the 132
   rows carrying a first/last-frame declaration also carry ordinary reference
   pictures in the same request. Since a keyframe arrives at canvas geometry and
   a reference at its own, a row-wide role would assign one geometry to pictures
   that need two.
5. **Every declared media file is opened and hashed, with no exemption**, added
   2026-08-24 for the escaped defect `media_status` documents. The earlier
   version verified nothing for `video-reference` rows and read only image
   headers for the rest, so a row could enter the pool naming a file that was
   not present. `bench/check_pool_media_integrity.py` holds this red/green.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "bench" / "results"
POOL = OUT / "2026-08-24_h3_calibration_pool.jsonl"
EXCLUDED = OUT / "2026-08-24_h3_calibration_pool_excluded.jsonl"
SUMMARY = OUT / "2026-08-24_h3_calibration_pool_summary.json"

KEYFRAME_DECL = re.compile(r"<Picture (\d+)> is the (first|last) frame")
MARKERS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>",
           "<|caption_start|>", "<|caption_end|>")

# Rights, read from each source's own declaration rather than assumed.
RIGHTS = {
    "StellarVoyager/H3-IR": {
        "license": "cc0-1.0",
        "declared_in": "dataset tag, README front matter, and per-row "
                       "`license` + `redistribution_allowed` fields",
        "eligible": True,
    },
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def picture_roles(row: dict) -> dict[int, str]:
    """Role per one-based picture ordinal.

    A picture the IR declares as a first or last frame is a **keyframe**: the DiT
    places it on the target timeline sharing the target spatial grid, so it must
    arrive at canvas geometry. Every other picture is an ordinary **reference**
    with its own grid. H3-IR mixes the two inside single requests, so this cannot
    be a row-wide property.
    """
    declared = {int(n) for n, _ in KEYFRAME_DECL.findall(row.get("target_ir") or "")}
    count = len(row.get("images") or [])
    return {i: ("keyframe" if i in declared else "reference")
            for i in range(1, count + 1)}


def primary_role(row: dict, roles: dict[int, str]) -> str:
    """Mutually exclusive row role, by declared priority.

    Video first: it changes the Qwen presentation to `<Video k>` plus timestamp
    markers and is governed by a separate sizing policy. Then the picture-role
    composition, because a request carrying both a keyframe and ordinary
    references needs two geometries and is its own case. Then image count, which
    is a presentation-length distinction only.
    """
    if row.get("videos"):
        return "video-reference"
    kinds = set(roles.values())
    if kinds == {"keyframe"}:
        return "keyframe-only"
    if "keyframe" in kinds:
        return "keyframe-plus-reference"
    n = len(row.get("images") or [])
    if n >= 4:
        return "multi-image-4-9"
    if n >= 2:
        return "multi-image-2-3"
    if n == 1:
        return "single-image"
    return "no-vision"


def overlays(row: dict, dims: list[tuple[int, int]]) -> dict:
    """Properties counted across the partition. Never a bucket."""
    ir = row.get("target_ir") or ""
    return {
        "wide_or_tall": any(w / h >= 1.9 or w / h < 0.6 for w, h in dims),
        "small_source": bool(dims) and max(w * h for w, h in dims) < 500_000,
        "markers": {m: ir.count(m) for m in MARKERS if ir.count(m)},
        "audio_label": bool(row.get("has_independent_audio")),
        "video_audio_track": bool(row.get("has_video_audio_track")),
    }


CACHE = Path.home() / ".cache/huggingface/hub/datasets--StellarVoyager--H3-IR"


def pinned_snapshot() -> tuple[Path, str]:
    """The revision the cache itself declares, not whichever glob returns first."""
    ref = CACHE / "refs" / "main"
    if not ref.exists():
        raise FileNotFoundError(f"{ref} is missing; cannot pin a revision")
    revision = ref.read_text().strip()
    root = CACHE / "snapshots" / revision
    if not root.exists():
        raise FileNotFoundError(f"declared revision {revision} is not in the cache")
    return root, revision


def image_dimensions(root: Path, relative_paths: list[str]) -> list[tuple[int, int]]:
    """Read geometry from the real files. PIL reads the header only."""
    from PIL import Image

    out = []
    for rel in relative_paths:
        path = root / rel
        if not path.exists():
            continue
        with Image.open(path) as im:
            out.append(im.size)
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_status(root: Path, row: dict) -> tuple[str | None, list[dict]]:
    """Verify every declared media file of one row against its declared hash.

    Returns ``(rejection_reason_or_None, per-file records)``.

    **This is the repair for an escaped defect.** Until 2026-08-24 the only
    media check here was a count of images whose geometry PIL could read, and it
    was skipped entirely for `video-reference` rows. Videos were therefore never
    opened, never hashed, and never required to exist: all 20 video rows entered
    the accepted pool on the strength of a declared filename. Sixteen of the
    nineteen distinct files they name were absent from the pinned snapshot, and
    nothing said so. A declared hash that is never recomputed is a claim, not
    evidence, so every declared file is now opened and hashed with no exemption
    by role or media kind. Held red/green by
    `bench/check_pool_media_integrity.py`, which feeds this function a missing
    file, a corrupted file, and a wrong declared hash.
    """
    declared = row.get("media_sha256") or {}
    records = []
    problems = []
    for rel in list(row.get("images") or []) + list(row.get("videos") or []):
        want = declared.get(rel)
        path = root / rel
        record = {"path": rel, "declared_sha256": want}
        if want is None:
            record["status"] = "undeclared"
            problems.append(f"{rel}: no declared sha256")
        elif not path.is_file():
            record["status"] = "missing"
            problems.append(f"{rel}: not in the pinned snapshot")
        else:
            actual = file_sha256(path)
            record["actual_sha256"] = actual
            record["size"] = path.stat().st_size
            if actual == want:
                record["status"] = "verified"
            else:
                record["status"] = "mismatch"
                problems.append(f"{rel}: sha256 {actual[:12]} != declared {want[:12]}")
        records.append(record)
    if not problems:
        return None, records
    return ("declared media did not verify against the pinned snapshot: "
            + "; ".join(problems)), records


class Components:
    """Union-find over individual media hashes.

    A media *set* hash groups only rows whose media match exactly. Disjointness
    needs the transitive closure: if row A and row B share one image, and B and C
    share a different one, all three must stay together.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main() -> int:
    root, revision = pinned_snapshot()
    train = root / "data" / "train.jsonl"
    if not train.exists():
        print(f"{train} is missing from the pinned snapshot; nothing to build.")
        return 2
    rows = [json.loads(l) for l in train.read_text().splitlines()]

    pool, excluded = [], []
    for index, row in enumerate(rows):
        dims = image_dimensions(root, row.get("images") or [])
        media_reason, media_records = media_status(root, row)
        roles = picture_roles(row)
        role = primary_role(row, roles)
        ir = row.get("target_ir") or ""
        record = {
            "id": row["id"],
            "source": "StellarVoyager/H3-IR",
            "source_revision": revision,
            "source_index": index,
            "channel": row.get("channel"),
            "primary_role": role,
            "picture_roles": {str(k): v for k, v in sorted(roles.items())},
            "overlays": overlays(row, dims),
            "image_count": len(row.get("images") or []),
            "video_count": len(row.get("videos") or []),
            "images": list(row.get("images") or []),
            "videos": list(row.get("videos") or []),
            "media_sha256": row.get("media_sha256") or {},
            "media_verification": media_records,
            "image_dimensions": [list(d) for d in dims],
            "prompt_sha256": sha(ir),
            "normalized_prompt_sha256": sha(normalize(ir)),
            "license": row.get("license"),
            "redistribution_allowed": row.get("redistribution_allowed"),
        }

        reason = None
        if role == "no-vision":
            reason = ("text-only: the sequential calibration trace admits one "
                      "modality envelope, so a row with no vision block cannot "
                      "join a vision-traced run. Belongs to the held-out T2VA "
                      "regression arm, not to calibration.")
        elif not row.get("redistribution_allowed"):
            reason = "source row does not declare redistribution_allowed"
        elif row.get("license") != "cc0-1.0":
            reason = f"unexpected per-row license {row.get('license')!r}"
        elif media_reason:
            reason = media_reason
        elif len(dims) != len(row.get("images") or []):
            reason = (f"read geometry for {len(dims)} of "
                      f"{len(row.get('images') or [])} images; the rest are not "
                      "in the pinned snapshot")

        if reason:
            excluded.append({**record, "exclusion_reason": reason})
        else:
            pool.append(record)

    # Deterministic: keyed and sorted by the dataset's own id.
    pool.sort(key=lambda r: r["id"])
    excluded.sort(key=lambda r: r["id"])

    seen = Counter(r["normalized_prompt_sha256"] for r in pool)

    # Connected components over INDIVIDUAL media hashes. Rows sharing any one
    # media file must land on the same side of a split, transitively.
    comp = Components()
    owner_of_hash = {}
    for i, r in enumerate(pool):
        comp.find(i)
        for digest in sorted((r["media_sha256"] or {}).values()):
            if digest in owner_of_hash:
                comp.union(i, owner_of_hash[digest])
            else:
                owner_of_hash[digest] = i
    members = defaultdict(list)
    for i in range(len(pool)):
        members[comp.find(i)].append(i)
    for root_index, idxs in members.items():
        for i in idxs:
            pool[i]["media_component"] = pool[root_index]["id"]
            pool[i]["media_component_size"] = len(idxs)
    multi = {k: v for k, v in members.items() if len(v) > 1}
    component_sizes = Counter(len(v) for v in members.values())

    by_role = Counter(r["primary_role"] for r in pool)
    ov = defaultdict(Counter)
    for r in pool:
        o = r["overlays"]
        if o["wide_or_tall"]:
            ov[r["primary_role"]]["wide_or_tall"] += 1
        if o["small_source"]:
            ov[r["primary_role"]]["small_source"] += 1
        if any(k in o["markers"] for k in ("<d>", "</d>")):
            ov[r["primary_role"]]["dialogue"] += 1
        if o["audio_label"]:
            ov[r["primary_role"]]["audio_label"] += 1

    verified_files = Counter()
    for record in pool + excluded:
        for item in record["media_verification"]:
            verified_files[item["status"]] += 1

    summary = {
        "source": "StellarVoyager/H3-IR",
        "source_revision": revision,
        "rows_read": len(rows),
        "media_files_by_status": dict(sorted(verified_files.items())),
        "pool": len(pool),
        "excluded": len(excluded),
        "exclusion_reasons": dict(Counter(r["exclusion_reason"].split(":")[0]
                                          for r in excluded)),
        "primary_roles": dict(by_role),
        "overlays_by_role": {k: dict(v) for k, v in ov.items()},
        "duplicate_normalized_prompts": sum(1 for v in seen.values() if v > 1),
        "media_components": len(members),
        "media_components_multi_row": len(multi),
        "rows_in_multi_row_components": sum(len(v) for v in multi.values()),
        "largest_media_component": max(component_sizes),
        "media_component_size_histogram": dict(sorted(component_sizes.items())),
        "picture_roles": dict(Counter(
            v for r in pool for v in r["picture_roles"].values())),
        "rights": RIGHTS,
        "notes": [
            "Primary roles partition the pool; overlays are counted across it "
            "and are not additive with the roles.",
            "No population size is chosen here. Selection needs a "
            "feasibility-measured N, which is owned by the encoder/quant lane.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    POOL.write_text("".join(json.dumps(r) + "\n" for r in pool))
    EXCLUDED.write_text("".join(json.dumps(r) + "\n" for r in excluded))
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"H3-IR @ {revision}: {len(rows)} rows -> pool {len(pool)}, "
          f"excluded {len(excluded)}")
    print(f"declared media files, recomputed: {dict(sorted(verified_files.items()))}")
    print(f"\n{'primary role':<20} {'rows':>5} | {'wide/tall':>9} {'small':>5} "
          f"{'dialogue':>8} {'audio':>5}")
    for role, n in by_role.most_common():
        o = ov[role]
        print(f"{role:<20} {n:>5} | {o['wide_or_tall']:>9} {o['small_source']:>5} "
              f"{o['dialogue']:>8} {o['audio_label']:>5}")
    print(f"\npicture roles: {summary['picture_roles']}")
    print(f"duplicate normalized prompts: {summary['duplicate_normalized_prompts']}")
    print(f"media components: {summary['media_components']} "
          f"({summary['media_components_multi_row']} multi-row, covering "
          f"{summary['rows_in_multi_row_components']} rows; largest "
          f"{summary['largest_media_component']})")
    print(f"\nwrote {POOL.name}, {EXCLUDED.name}, {SUMMARY.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
