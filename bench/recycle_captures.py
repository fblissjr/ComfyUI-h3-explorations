#!/usr/bin/env python3
"""List the activation captures under `H3_CAPTURE_ROOT` with what each still
owes, and delete the tensors of one that owes nothing.

Captures are transient by the owner's policy (2026-09-03): the disk is not
unlimited, a capture's value ends when the analysis it was taken for is
recorded, and what must outlive it is small -- the manifest and the inventory
record under `bench/results/`. This tool is the recycling half of that: it
never deletes a capture the repo cannot account for.

    python bench/recycle_captures.py                  # the table
    python bench/recycle_captures.py --budget-gb 200  # what to delete, oldest first, to fit
    python bench/recycle_captures.py --delete NAME    # delete NAME's tensors, if it owes nothing

A capture is RECYCLABLE when all of:
  - its directory carries `manifest.json` and `retention.json`;
  - `bench/results/` holds a manifest copy with the same `graph_sha256` and
    the same tensor sha256 set (the copy is what outlives the tensors);
  - an inventory record under `bench/results/` names the capture;
  - `retention.json`'s `keep_until` has passed, or `--delete` is given with
    `--reason`, which is written into the marker.

Deletion removes `qkv_*.pt` and `final_*.pt` only. `manifest.json`,
`retention.json` and a new `DELETED.json` (when, why, bytes freed) stay, so
the directory keeps saying what it was. Nothing here is automatic: the
owner runs `--delete`, one capture at a time.

`retention.json`, written by whoever takes the capture:
    {"purpose": ..., "keep_until": "YYYY-MM-DD", "delete_when": ...,
     "records_in_repo": [...], "set_by": ...}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
sys.path.insert(0, str(REPO / "bench"))
import _paths  # noqa: E402


def _tensors(d: Path) -> list[Path]:
    """The files a delete would remove: regular files only, never a symlink,
    never a directory wearing the name (audit finding 3, 2026-09-03)."""
    out = []
    for p in sorted(d.glob("qkv_*.pt")) + sorted(d.glob("final_*.pt")):
        if p.is_symlink() or not p.is_file():
            continue
        out.append(p)
    return out


def _size(d: Path) -> int:
    total = 0
    for p in _tensors(d):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _repo_records(name: str, manifest: dict | None):
    """The manifest copy and inventory record that account for `name`."""
    copy = None
    if isinstance(manifest, dict):
        want = ((manifest.get("workload") or {}).get("graph_sha256"),
                sorted(str(t.get("sha256")) for t in manifest.get("captured_tensors") or [] if isinstance(t, dict)))
        for p in RESULTS.glob("*capture_manifest*.json"):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(m, dict):
                continue
            got = ((m.get("workload") or {}).get("graph_sha256"),
                   sorted(str(t.get("sha256")) for t in m.get("captured_tensors") or [] if isinstance(t, dict)))
            if got == want:
                copy = p.name
                break
    inventory = None
    for p in RESULTS.glob("*capture_inventory*.json"):
        # the record's own capture names, not a substring of the file (a
        # directory called `0` matched every inventory; audit finding 10)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names = {str(c.get("name") or c.get("capture") or c.get("dir") or "").split("/")[-1]
                 for c in (doc.get("captures") or []) if isinstance(c, dict)}
        if name in names:
            inventory = p.name
            break
    return copy, inventory


def survey(root: Path) -> list[dict]:
    rows = []
    root = root.resolve()
    # Direct, real subdirectories only: a symlinked capture directory would
    # redirect both the delete and the marker write to wherever it points
    # (audit finding 1, 2026-09-03), so it is not a capture to this tool.
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        if d.resolve().parent != root:
            continue
        tensors = _tensors(d)
        marker = d / "DELETED.json"
        deleted = marker.is_file() and not marker.is_symlink()
        if not tensors and not deleted:
            continue
        manifest = None
        if (d / "manifest.json").is_file():
            try:
                manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = None
            if not isinstance(manifest, dict):
                manifest = None
        retention = None
        if (d / "retention.json").is_file():
            try:
                retention = json.loads((d / "retention.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                retention = None
            if not isinstance(retention, dict):
                retention = None
        copy, inventory = _repo_records(d.name, manifest)
        keep_until = None
        if retention and retention.get("keep_until"):
            try:
                keep_until = dt.date.fromisoformat(retention["keep_until"])
            except ValueError:
                keep_until = None
        expired = keep_until is not None and keep_until < dt.date.today()
        owes = []
        if manifest is None:
            owes.append("manifest.json")
        if retention is None:
            owes.append("retention.json")
        if copy is None:
            owes.append("a manifest copy under bench/results/ (same graph and tensor hashes)")
        if inventory is None:
            owes.append("an inventory record under bench/results/ naming it")
        rows.append({"name": d.name, "dir": d, "bytes": _size(d), "tensors": len(tensors),
                     "deleted": deleted, "manifest": manifest is not None,
                     "retention": retention, "keep_until": keep_until, "expired": expired,
                     "copy": copy, "inventory": inventory, "owes": owes,
                     "recyclable": not owes})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget-gb", type=float, help="print which recyclable captures to delete, oldest first, to fit")
    ap.add_argument("--delete", metavar="NAME", help="delete NAME's tensors if it owes nothing")
    ap.add_argument("--reason", help="required with --delete before keep_until has passed")
    args = ap.parse_args()

    root = _paths.capture_root()
    if root is None or not root.is_dir():
        print("  skip  H3_CAPTURE_ROOT is not set or not a directory; start.sh exports it for the server, "
              "the shell needs it too")
        return 2
    rows = survey(root)
    if not rows:
        print(f"  no captures under the collection root")
        return 0
    total = sum(r["bytes"] for r in rows)
    print(f"  {len(rows)} capture(s), {total / 2**30:.1f} GiB of tensors\n")
    for r in rows:
        state = ("DELETED" if r["deleted"] else
                 "recyclable" + (" (keep_until passed)" if r["expired"] else
                                 f" (keep until {r['keep_until']})" if r["keep_until"] else " (no keep_until)")
                 if r["recyclable"] else "owes " + "; ".join(r["owes"]))
        purpose = (r["retention"] or {}).get("purpose", "")
        print(f"  {r['name']:40} {r['bytes'] / 2**30:7.1f} GiB  {r['tensors']:3} tensors  {state}")
        if purpose:
            print(f"  {'':40} purpose: {purpose[:120]}")
    if args.budget_gb is not None:
        over = total - args.budget_gb * 2**30
        print(f"\n  budget {args.budget_gb:.0f} GB: " + ("within it" if over <= 0 else f"over by {over / 2**30:.1f} GiB"))
        if over > 0:
            freed = 0
            # oldest first by the directory's own modification time, not by
            # name (names are date-prefixed by convention only)
            for r in sorted((r for r in rows if r["recyclable"] and not r["deleted"]),
                            key=lambda r: r["dir"].stat().st_mtime):
                print(f"    delete {r['name']}  (frees {r['bytes'] / 2**30:.1f} GiB)")
                freed += r["bytes"]
                if freed >= over:
                    break
            if freed < over:
                print("    ...and the rest is owed by captures this tool will not touch; give them records or an owner")
    if args.delete:
        r = next((r for r in rows if r["name"] == args.delete), None)
        if r is None:
            print(f"\n  no capture named {args.delete!r}"); return 1
        if r["deleted"]:
            print(f"\n  {args.delete} is already deleted"); return 0
        if not r["recyclable"]:
            print(f"\n  refusing: {args.delete} owes " + "; ".join(r["owes"])); return 1
        if not r["expired"] and not args.reason:
            print(f"\n  refusing: keep_until {r['keep_until']} has not passed; pass --reason to delete early"); return 1
        marker = r["dir"] / "DELETED.json"
        if marker.is_symlink() or marker.exists():
            print(f"\n  refusing: {marker.name} already exists or is a symlink in {args.delete}"); return 1
        # The marker is created first, exclusively and without following a
        # link (audit finding 2), so a crash mid-delete still leaves a record
        # saying a delete began; it is completed with the byte count after.
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o644)
        os.close(fd)
        freed, removed = 0, 0
        for p in _tensors(r["dir"]):
            try:
                freed += p.stat().st_size
                p.unlink()
                removed += 1
            except OSError as exc:
                print(f"  could not remove {p.name}: {exc}")
        marker.write_text(json.dumps({
            "deleted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "bytes_freed": freed, "tensors_removed": removed, "tensors_listed": r["tensors"],
            "reason": args.reason or f"keep_until {r['keep_until']} passed",
            "records_in_repo": [r["copy"], r["inventory"]],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\n  deleted {removed} of {r['tensors']} tensor file(s) of {args.delete}, {freed / 2**30:.1f} GiB freed; "
              f"manifest.json, retention.json and DELETED.json remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
