#!/usr/bin/env python3
"""Is the released H3 text encoder the stock Qwen3-VL-32B-Instruct, byte for byte?

The Hub stores a SHA-256 for every LFS file, so the question is answered
without downloading either checkpoint: the 14 `text_encoder/` shards of the
MiniMax-H3 release are compared, by name, size and LFS digest, against the 14
shards of `Qwen/Qwen3-VL-32B-Instruct`. Identical digests on identically laid
out shards is identity of every tensor in them, including the embedding table
and the seven H3 token rows.

What it settles: whether any post-training of the released encoder shipped.
What it does not: what MiniMax runs behind their API, which is not observable
from the release.

Optionally hashes the local shards too (`--local DIR`, slow: the full
checkpoint is read) so the local copy is tied to the same digests.

Needs network access to the Hub; run with the ComfyUI venv python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from h3_producer_provenance import producer_provenance  # noqa: E402

RELEASE = "MiniMaxAI/MiniMax-H3"
RELEASE_PREFIX = "text_encoder/"
STOCK = "Qwen/Qwen3-VL-32B-Instruct"


def hub_shards(api, repo: str, prefix: str) -> tuple[dict, str]:
    info = api.model_info(repo, files_metadata=True)
    shards = {}
    for sibling in info.siblings:
        name = sibling.rfilename
        if name.startswith(prefix) and name.endswith(".safetensors"):
            shards[name[len(prefix):]] = {
                "sha256": sibling.lfs.sha256 if sibling.lfs else None,
                "size": sibling.size,
            }
    return shards, info.sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default=None,
                        help="local text-encoder directory to hash against the Hub digests")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    release, release_rev = hub_shards(api, RELEASE, RELEASE_PREFIX)
    stock, stock_rev = hub_shards(api, STOCK, "")
    names = sorted(set(release) | set(stock))
    rows = []
    for name in names:
        a, b = release.get(name), stock.get(name)
        rows.append({
            "shard": name,
            "release": a, "stock": b,
            "identical": bool(a and b and a["sha256"] and a["sha256"] == b["sha256"]
                              and a["size"] == b["size"]),
        })
    all_identical = bool(rows) and all(r["identical"] for r in rows)

    local = None
    if args.local:
        directory = Path(args.local).expanduser().resolve()
        local = {}
        for name in names:
            path = directory / name
            if not path.exists():
                local[name] = {"present": False}
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(chunk)
            local[name] = {"present": True, "sha256": digest.hexdigest(),
                           "matches_release": digest.hexdigest() == release.get(name, {}).get("sha256")}

    report = {
        "question": "is the released H3 text encoder byte-identical to stock "
                    "Qwen3-VL-32B-Instruct",
        "method": "Hub LFS SHA-256 and size per shard, compared by shard name; no download",
        "release": {"repo": RELEASE, "prefix": RELEASE_PREFIX, "revision": release_rev,
                    "shards": len(release)},
        "stock": {"repo": STOCK, "revision": stock_rev, "shards": len(stock)},
        "shards": rows,
        "all_shards_identical": all_identical,
        "consequence_if_identical": "every tensor of the released encoder, the "
                                    "embedding table and the seven H3 token rows "
                                    "included, is the stock release; no post-training "
                                    "shipped",
        "not_established": "what runs behind the MiniMax API",
        "local": local,
        "producer": producer_provenance(__file__),
    }
    out = Path(args.out).expanduser().resolve()
    out.write_text(json.dumps(report, indent=2) + "\n")
    identical = sum(1 for r in rows if r["identical"])
    print(f"{identical}/{len(rows)} shards identical; all identical: {all_identical}")
    if local:
        print(f"local: {sum(1 for v in local.values() if v.get('matches_release'))}/{len(local)} "
              "shards match the release digests")
    print(f"wrote {out.name}")
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
