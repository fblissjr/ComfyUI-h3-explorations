#!/usr/bin/env python
"""List what each watched upstream landed since its last read, and its open PRs and issues.

The mechanical half of an upstream sweep (`docs/wiki/upstream/index.md`,
"Running a sweep"). For every entry in `docs/wiki/upstream/watchlist.json` it
prints:

- where the local clone stands against its remote branch (it never moves the
  checkout; `--fetch` only updates remote refs);
- the commits from the entry's `last_read.rev` to the remote tip that touch the
  entry's `paths`, oldest first, with date and subject;
- with `--gh`, the open PRs and issues matching the entry's `keywords` updated
  since `last_read.date`, and the PRs merged since then, through `gh`.

It prints data, never a verdict: which of these matter is the sweep's reading,
recorded in `docs/wiki/upstream/sweeps/<date>.md`. A keyword match is a lead,
not a finding (`H3` also matches HTTP/3 work in some repos).

After a sweep's record is written, `--mark NAME --record PATH` moves that
entry's `last_read` to the remote tip it was read at, so the next run starts
there.

    python bench/upstream_watch.py                  # every entry, git only
    python bench/upstream_watch.py --only sglang --gh
    python bench/upstream_watch.py --gh --json > /tmp/x.json
    python bench/upstream_watch.py --mark sglang --record docs/wiki/upstream/sweeps/2026-11-15.md

Standard library only; run from the repo root with ComfyUI's venv Python.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST = ROOT / "docs" / "wiki" / "upstream" / "watchlist.json"


def _git(clone: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(clone), *args], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"git {' '.join(args)} failed")
    return out.stdout


def _gh(*args: str) -> list[dict]:
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        return [{"error": out.stderr.strip()}]
    return json.loads(out.stdout or "[]")


def load() -> dict:
    return json.loads(WATCHLIST.read_text())


def clone_state(entry: dict, fetch: bool) -> dict:
    """Commits since last read on the watched paths, read from the remote branch."""
    if not entry.get("clone"):
        return {"clone": None}
    clone = (ROOT / entry["clone"]).resolve()
    if not (clone / ".git").exists():
        return {"clone": entry["clone"], "error": "no git checkout at this path"}
    remote_ref = f"origin/{entry.get('branch', 'main')}"
    if fetch:
        _git(clone, "fetch", "--quiet", "origin", entry.get("branch", "main"))
    tip = _git(clone, "rev-parse", "--short=10", remote_ref).strip()
    head = _git(clone, "rev-parse", "--short=10", "HEAD").strip()
    tip_date = _git(clone, "log", "-1", "--format=%cs", remote_ref).strip()
    last = entry["last_read"]["rev"]
    is_ancestor = subprocess.run(
        ["git", "-C", str(clone), "merge-base", "--is-ancestor", last, remote_ref],
        capture_output=True,
    ).returncode == 0
    rng = f"{last}..{remote_ref}"
    total = int(_git(clone, "rev-list", "--count", rng).strip() or 0)
    pathspecs = entry.get("paths") or []
    log = _git(
        clone, "log", "--reverse", "--format=%h%x09%cs%x09%s", rng, "--", *pathspecs
    ) if pathspecs else ""
    commits = [
        dict(zip(("rev", "date", "subject"), line.split("\t", 2)))
        for line in log.splitlines()
        if line.strip()
    ]
    return {
        "clone": entry["clone"],
        "head": head,
        "remote_tip": tip,
        "remote_tip_date": tip_date,
        "head_is_tip": _git(clone, "rev-parse", "HEAD").strip()
        == _git(clone, "rev-parse", remote_ref).strip(),
        "range": rng.replace(remote_ref, tip),
        "commits_in_range": total,
        "watched_commits": commits,
        "last_read_is_ancestor": is_ancestor,
        # commits dated before the last read arrived by a merge of older history
        "dated_before_last_read": sum(
            1 for c in commits if c["date"] < entry["last_read"]["date"]
        ),
    }


def gh_state(entry: dict) -> dict:
    """Open PRs and issues by keyword, updated since the last read; PRs merged since."""
    repo = entry.get("github")
    if not repo:
        return {}
    since = entry["last_read"]["date"]
    fields = "number,title,updatedAt,url,author"
    seen: dict[str, dict] = {}
    for kind, state, qual in (
        ("pr", "open", f"updated:>={since}"),
        ("pr", "merged", f"merged:>={since}"),
        ("issue", "open", f"updated:>={since}"),
    ):
        for kw in entry.get("keywords") or []:
            rows = _gh(
                kind, "list", "-R", repo, "--state", state, "--limit", "100",
                "--search", f"{kw} {qual}", "--json", fields,
            )
            for row in rows:
                if "error" in row:
                    seen[f"error:{kind}:{state}"] = row
                    continue
                key = f"{kind}:{state}:{row['number']}"
                hit = seen.setdefault(key, {
                    "kind": kind, "state": state, "number": row["number"],
                    "title": row["title"], "updated": row["updatedAt"][:10],
                    "url": row["url"], "author": (row.get("author") or {}).get("login"),
                    "matched": [],
                })
                hit["matched"].append(kw)
                words = [w for w in kw.lower().split() if w]
                hit["in_title"] = hit.get("in_title", False) or not words or all(
                    w in row["title"].lower() for w in words
                )
    return {"since": since, "items": sorted(
        seen.values(), key=lambda r: (r.get("kind", ""), r.get("state", ""), -r.get("number", 0))
    )}


def report(entry: dict, args) -> dict:
    out = {"name": entry["name"], "github": entry.get("github"),
           "last_read": entry["last_read"]}
    try:
        out["git"] = clone_state(entry, args.fetch)
    except RuntimeError as exc:
        out["git"] = {"error": str(exc)}
    if args.gh:
        out["gh"] = gh_state(entry)
    return out


def show(r: dict) -> None:
    lr = r["last_read"]
    print(f"## {r['name']}  ({r.get('github') or 'no GitHub repo'})")
    print(f"last read {lr['rev']} on {lr['date']}  record: {lr.get('record', '-')}")
    g = r["git"]
    if g.get("error"):
        print(f"git: {g['error']}")
    elif g.get("clone"):
        lag = "" if g["head_is_tip"] else f"  (checkout at {g['head']}, behind or apart)"
        print(f"clone {g['clone']}: remote tip {g['remote_tip']} ({g['remote_tip_date']}){lag}")
        print(f"range {g['range']}: {g['commits_in_range']} commits, "
              f"{len(g['watched_commits'])} on watched paths")
        if not g["last_read_is_ancestor"]:
            print(f"  note: {lr['rev']} is not an ancestor of the tip (history rewritten?)")
        if g["dated_before_last_read"]:
            print(f"  note: {g['dated_before_last_read']} watched commits are dated before "
                  f"{lr['date']} (older history merged in)")
        for c in g["watched_commits"]:
            print(f"  {c['rev']} {c['date']} {c['subject']}")
    gh = r.get("gh")
    if gh:
        items = [it for it in gh["items"] if "error" not in it]
        strong = sum(1 for it in items if it["in_title"])
        print(f"gh since {gh['since']}: {len(items)} items, {strong} with a keyword "
              f"in the title (*); the rest matched only in the body")
        for it in sorted(gh["items"], key=lambda it: not it.get("in_title", True)):
            if "error" in it:
                print(f"  error: {it['error']}")
                continue
            star = "*" if it["in_title"] else " "
            print(f" {star}{it['kind']} {it['state']:6} #{it['number']} {it['updated']} "
                  f"{it['title']}  [{', '.join(k or 'all' for k in it['matched'])}] {it['url']}")
    print()


def mark(data: dict, name: str, record: str) -> None:
    entry = next((e for e in data["entries"] if e["name"] == name), None)
    if entry is None:
        sys.exit(f"no entry named {name!r}")
    if not (ROOT / record).exists():
        sys.exit(f"record {record} does not exist; write the sweep record first")
    if entry.get("clone"):
        clone = (ROOT / entry["clone"]).resolve()
        rev = _git(clone, "rev-parse", "--short=10",
                   f"origin/{entry.get('branch', 'main')}").strip()
    else:
        rev = "-"
    entry["last_read"] = {"rev": rev, "date": _dt.date.today().isoformat(), "record": record}
    WATCHLIST.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{name}: last_read -> {rev} on {entry['last_read']['date']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--only", action="append", help="entry name (repeatable)")
    ap.add_argument("--gh", action="store_true", help="also query open PRs/issues via gh")
    ap.add_argument("--fetch", action="store_true", help="git fetch the remote branch first")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--mark", metavar="NAME", help="move NAME's last_read to its remote tip")
    ap.add_argument("--record", help="sweep record path, with --mark")
    args = ap.parse_args()

    data = load()
    if args.mark:
        if not args.record:
            sys.exit("--mark needs --record")
        mark(data, args.mark, args.record)
        return 0
    if args.gh and not shutil.which("gh"):
        sys.exit("--gh needs the gh CLI on PATH")
    entries = [e for e in data["entries"] if not args.only or e["name"] in args.only]
    reports = [report(e, args) for e in entries]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            show(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
