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

The delete is a recoverable transition (Codex, 2026-09-03: the first
version wrote the marker before the unlinks, so a crash or one failed
unlink left a capture that read as deleted and could not be retried).
`DELETING.json` is written first with the planned files; each unlink is
attempted; on any failure the file stays, the exit is nonzero, and the
survey shows the capture as an INCOMPLETE delete that `--delete` resumes;
only when no planned tensor remains is it renamed atomically to
`DELETED.json`. `--self-test` proves that path on a temp root by making
one unlink fail, then resuming.

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
        pending = d / "DELETING.json"
        incomplete = pending.is_file() and not pending.is_symlink()
        if not tensors and not deleted and not incomplete:
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
                     "deleted": deleted, "incomplete": incomplete, "manifest": manifest is not None,
                     "retention": retention, "keep_until": keep_until, "expired": expired,
                     "copy": copy, "inventory": inventory, "owes": owes,
                     "recyclable": not owes})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget-gb", type=float, help="print which recyclable captures to delete, oldest first, to fit")
    ap.add_argument("--delete", metavar="NAME", help="delete NAME's tensors if it owes nothing")
    ap.add_argument("--reason", help="required with --delete before keep_until has passed")
    ap.add_argument("--self-test", action="store_true", help="prove the delete transition on a temp root; touches nothing else")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

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
                 "INCOMPLETE delete, tensors remain; rerun --delete to resume" if r["incomplete"] else
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
        if not r["incomplete"]:
            if not r["recyclable"]:
                print(f"\n  refusing: {args.delete} owes " + "; ".join(r["owes"])); return 1
            if not r["expired"] and not args.reason:
                print(f"\n  refusing: keep_until {r['keep_until']} has not passed; pass --reason to delete early"); return 1
        return delete_capture(r, args.reason)
    return 0


def delete_capture(r: dict, reason: str | None) -> int:
    """The recoverable transition: DELETING.json first (planned files), each
    unlink attempted, the marker renamed to DELETED.json only when nothing
    planned remains. A failure leaves DELETING.json, prints what remains and
    exits nonzero; running --delete again resumes."""
    d = r["dir"]
    pending, done = d / "DELETING.json", d / "DELETED.json"
    for m in (pending, done):
        if m.is_symlink():
            print(f"\n  refusing: {m.name} is a symlink in {r['name']}"); return 1
    if done.exists():
        print(f"\n  refusing: {done.name} already exists in {r['name']}"); return 1
    if pending.exists():
        plan = json.loads(pending.read_text(encoding="utf-8"))
        print(f"\n  resuming an incomplete delete of {r['name']} begun {plan.get('begun_at')}")
    else:
        plan = {"begun_at": dt.datetime.now().isoformat(timespec="seconds"),
                "planned": [p.name for p in _tensors(d)],
                "reason": reason or f"keep_until {r['keep_until']} passed",
                "records_in_repo": [r["copy"], r["inventory"]], "bytes_freed": 0, "removed": []}
        fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(plan, indent=2) + "\n")
    failed = []
    for name in plan["planned"]:
        p = d / name
        if name in plan["removed"] or not p.exists():
            if name not in plan["removed"]:
                plan["removed"].append(name)
            continue
        if p.is_symlink() or not p.is_file():
            failed.append((name, "not a regular file")); continue
        try:
            size = p.stat().st_size
            p.unlink()
            plan["bytes_freed"] += size
            plan["removed"].append(name)
        except OSError as exc:
            failed.append((name, str(exc)))
    pending.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if failed:
        for name, why in failed:
            print(f"  could not remove {name}: {why}")
        print(f"\n  INCOMPLETE: {len(failed)} of {len(plan['planned'])} planned tensor(s) remain in {r['name']}; "
              f"{pending.name} records the state; rerun --delete to resume")
        return 1
    plan["deleted_at"] = dt.datetime.now().isoformat(timespec="seconds")
    pending.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, done)   # atomic: the capture is deleted only once every planned file is gone
    print(f"\n  deleted {len(plan['removed'])} tensor file(s) of {r['name']}, {plan['bytes_freed'] / 2**30:.1f} GiB freed; "
          f"manifest.json, retention.json and DELETED.json remain")
    return 0


def self_test() -> int:
    """Prove the transition on a temp root: a delete with one unlink forced
    to fail leaves DELETING.json and tensors and exits nonzero; a resume
    completes and renames the marker; the survey reads each state right."""
    import hashlib
    import tempfile
    global RESULTS
    with tempfile.TemporaryDirectory(prefix="h3_recycle_selftest_") as tmp:
        root = Path(tmp) / "root"; res = Path(tmp) / "results"
        root.mkdir(); res.mkdir()
        RESULTS = res
        cap = root / "2026-01-01_selftest"; cap.mkdir()
        names = [f"qkv_L8_S8_b{i}_s1.pt" for i in range(3)]
        shas = []
        for n in names:
            (cap / n).write_bytes(os.urandom(64)); shas.append(hashlib.sha256((cap / n).read_bytes()).hexdigest())
        manifest = {"workload": {"graph_sha256": "g" * 64},
                    "captured_tensors": [{"filename": n, "sha256": h} for n, h in zip(names, shas)]}
        (cap / "manifest.json").write_text(json.dumps(manifest)); (res / "x_capture_manifest_y.json").write_text(json.dumps(manifest))
        (cap / "retention.json").write_text(json.dumps({"purpose": "self-test", "keep_until": "2026-01-02"}))
        (res / "x_capture_inventory_y.json").write_text(json.dumps({"captures": [{"capture": cap.name}]}))
        rows = survey(root); r = rows[0]
        assert r["recyclable"] and r["expired"], r["owes"]
        # force one PLANNED unlink to fail, the way a permission or I/O error
        # would mid-delete: the file is planned as a regular file and the
        # unlink itself raises
        real_unlink = os.unlink
        def flaky(path, *a, **k):
            if os.fspath(path).endswith(names[1]):
                raise OSError("simulated unlink failure")
            return real_unlink(path, *a, **k)
        os.unlink = flaky
        try:
            rc = delete_capture(r, None)
        finally:
            os.unlink = real_unlink
        assert rc == 1 and (cap / "DELETING.json").is_file() and not (cap / "DELETED.json").exists(), "failure must leave DELETING.json"
        assert (cap / names[1]).exists() and not (cap / names[0]).exists() and not (cap / names[2]).exists(), "the others were removed, the victim remains"
        r2 = survey(root)[0]
        assert r2["incomplete"] and not r2["deleted"], "survey must read the incomplete state"
        rc = delete_capture(r2, None)
        assert rc == 0 and (cap / "DELETED.json").is_file() and not (cap / "DELETING.json").exists(), "resume must complete and rename"
        assert not any((cap / n).exists() for n in names) and (cap / "manifest.json").is_file()
        r3 = survey(root)[0]
        assert r3["deleted"] and not r3["incomplete"]
        assert delete_capture(r3, None) == 1, "a deleted capture must refuse a second delete"
    print("  ok    self-test: a failed unlink leaves DELETING.json and exits 1; a resume completes and renames to DELETED.json; the survey reads both states")
    return 0


if __name__ == "__main__":
    sys.exit(main())
