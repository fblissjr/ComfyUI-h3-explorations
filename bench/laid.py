#!/usr/bin/env python3
"""The list of what is laid: one JSON line a piece, the ONLY thing the delivery table reads for what stands.

    <python> bench/laid.py seed --table T.json --share SHARE --out data/<video>/laid.jsonl --by NAME
    <python> bench/laid.py verify --laid data/<video>/laid.jsonl --share SHARE
    <python> bench/laid.py refs --laid data/<video>/laid.jsonl --ledger LEDGER.json --share SHARE
    (read by) <python> bench/delivery_table.py ... --laid data/<video>/laid.jsonl

**Why** (the owner's order of 2026-10-16, `docs/wiki/decisions.md`): what the cut laid was decided in four places
(the owner's keeps, a session's `prefers.txt`, lines typed into a rebuild script, the newest table by file time),
and what the owner reviewed drifted from what was laid. Here one line says which render is laid on a shot for a
subject, in what order among that shot's lines, the sha256 of its file as laid, and who chose it: `owner` is `kept` where the owner's last choice on that render is a keep (`delivery_table.KEEP`)
given on the render's own row, `kept in a joined clip` where that keep was given on a joined clip that laid it (the
ledger counts it as a keep on each piece, the ruling of 2026-10-14; the owner watched the piece inside the joined
clip, not alone), else `unconfirmed` (mrgan's cold read, 2026-10-16).

**A line** (`schema` `h3_laid/1`): `first`, `last` (the shot, as the shots file has it), `subject`, `load`, `file`
(relative to the share, as the ledger has it), `sha256`, `owner`, `by` (the chooser's words: the owner's keep with
its time and row, or a session's reason), `frames` (what the table that seeded it laid, for reading only: the
table works the frames out again), `added`, `added_by`. On one shot and subject the lines are laid in file order,
each on frames no line before it took.

**Hard failures** (`read`, and so the table): a line that is not JSON or lacks a field, two lines naming one load
on one shot and subject, a file that is missing, a file whose bytes are not the line's sha256. Changing a laid
piece is a new line written by hand with its reason, never an edit that keeps the old sha256.

**`refs`** says, for every line, whether the reference video its render was given is the file at that path now. The
render's build record names it (`files.motion`); where the queue record kept a digest (records queued from
631a9206 on) the bytes are compared (`same bytes as queued` / `CHANGED since queued`); where it did not, only the
file's time is read against the render's (`older than the render`, consistent and not proof; `REWRITTEN after the
render`, the render read something else). Data, not a verdict: a rewritten reference says the record points at the
wrong file, not that the picture is wrong; the original may be under `references/*_SUPERSEDED_*`.

What it cannot see: whether the owner watched these bytes when they kept the render (answers before 2026-10-16 do
not record the clip's sha256), and whether a render read the reference its record names (`bench/laid.py` does not
open the graph).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCHEMA = "h3_laid/1"
FIELDS = ("first", "last", "subject", "load", "file", "sha256", "owner", "by")
OWNER = ("kept", "kept in a joined clip", "unconfirmed")


def zoomed(f: Path) -> Path:
    """The companion the assembler reads at `--size source` (`assemble_delivery.py`, the zoomed piece)."""
    return f.with_name(f.name.replace("_00001.mp4", "_00001_zoomed.mp4"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(path: str) -> str:
    return sha256(Path(path))


def read(path: str, share: str, check_bytes: bool = True) -> list[dict]:
    """Every line, with its 1-based `line` number; stops on any hard failure named in the docstring."""
    out, seen = [], set()
    for n, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            x = json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}, line {n}: not JSON ({e})")
        missing = [k for k in FIELDS if k not in x]
        if x.get("schema") != SCHEMA or missing:
            raise SystemExit(f"{path}, line {n}: schema {x.get('schema')!r} (wanted {SCHEMA!r}); missing {missing}")
        if x["owner"] not in OWNER:
            raise SystemExit(f"{path}, line {n}: owner {x['owner']!r} is not one of {OWNER}")
        key = (x["first"], x["last"], x["subject"], x["load"])
        if key in seen:
            raise SystemExit(f"{path}, line {n}: {x['load']} on {x['first']}-{x['last']}, {x['subject']}, is named twice")
        seen.add(key)
        if check_bytes:
            f = Path(share) / x["file"]
            if not f.is_file():
                raise SystemExit(f"{path}, line {n}: {f} is missing")
            now = sha256(f)
            if now != x["sha256"]:
                raise SystemExit(f"{path}, line {n}: {f} is sha256 {now[:16]} now, and the line says {x['sha256'][:16]}: the laid piece changed")
            if x.get("zoomed_sha256") and sha256(zoomed(f)) != x["zoomed_sha256"]:
                raise SystemExit(f"{path}, line {n}: {zoomed(f)} is not the bytes the line says: the laid piece's zoomed companion changed")
        out.append({**x, "line": n})
    return out


def seed(table: str, share: str, by: str) -> list[dict]:
    """The lines that reproduce what a table laid, from its JSON's reading (`stands`), in its order."""
    import delivery_table as dt
    t = json.loads(Path(table).read_text())
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for e in t["reading"]:
        for st in e["stands"]:
            c = st.get("the_owners_last_choice")
            kept = bool(c and c["said"] in dt.KEEP)
            in_joined = kept and "_joined_" in str(c.get("row") or "")
            if st.get("step") == 0:
                chose = f"a session's --prefer: {st.get('the_leads_prefer')}"
            elif st.get("step") == 1:
                chose = f"the owner's keep of {str(c['at'])[:16]}Z (row {c.get('row')})"
            else:
                chose = "nobody chose it: the old rule's step 2, the newest render in the frozen form" + (
                    f"; the owner's last choice on it: \"{c['said']}\" ({str(c['at'])[:16]}Z)" if c else "; the owner has no choice on it")
            if st.get("step") == 0 and kept:
                chose += f"; and the owner's keep of {str(c['at'])[:16]}Z (row {c.get('row')})"
            lines.append({"schema": SCHEMA, "first": e["first"], "last": e["last"], "subject": e["subject"], "load": st["render"],
                          "file": st["file"], "sha256": sha256(Path(share) / st["file"]), "owner": ("kept in a joined clip" if in_joined else "kept") if kept else "unconfirmed",
                          "by": chose, "frames": st["frames"], "added": now, "added_by": by,
                          "seeded_from": Path(table).name})
    return lines


def refs(lines: list[dict], ledger: dict, repo: Path) -> list[dict]:
    """One row a laid line and reference: what the render's records say it read, against the file now."""
    out = []
    for x in lines:
        r = ledger["renders"].get(x["file"]) or {}                      # the ledger is keyed by the render's file
        rec = next((c for c in reversed(r.get("records") or []) if c.get("built_file")), None)   # the newest build record
        if rec is None:
            out.append({**x, "reference": None, "state": "no build record names a reference"})
            continue
        built = json.loads((repo / rec["built_file"]).read_text())
        queued = json.loads((repo / rec["queued_file"]).read_text()) if rec.get("queued_file") and (repo / rec["queued_file"]).is_file() else {}
        m = (built.get("files") or {}).get("motion") or (built.get("load") or {}).get("motion_file")
        motion = [m] if isinstance(m, str) else [m[0]] if m and isinstance(m[0], str) else [v[0] for v in m or [] if isinstance(v, list)]  # [path, first frame]
        if not motion:
            out.append({**x, "reference": None, "state": "the build record names no reference video" + (f" (its region: {r['form']['replaces'][:60]})" if (r.get("form") or {}).get("replaces") else "")})
        written = datetime.datetime.fromisoformat(str(r.get("written")).replace("Z", "+00:00")).timestamp() if r.get("written") else None
        for ref in motion:
            f = Path(ref)
            kept = (queued.get("digests") or {}).get(ref)
            if not f.is_file():
                state = "MISSING at that path"
            elif kept:
                state = "same bytes as queued" if sha256(f) == kept else "CHANGED since queued"
            elif written is None:
                state = "no digest kept and no render time"
            else:
                state = "no digest kept; older than the render" if f.stat().st_mtime <= written else "no digest kept; REWRITTEN after the render"
            found = None
            if kept and state == "CHANGED since queued":
                # the bytes it read, where a remake set the original aside (references/<name>_SUPERSEDED_*/ the same file name)
                found = next((str(g) for g in sorted(f.parent.parent.glob(f"{f.parent.name}_SUPERSEDED_*/{f.name}")) if sha256(g) == kept), None)
                state += f"; the bytes it read are at {found}" if found else "; the bytes it read are not found beside it"
            out.append({**x, "reference": ref, "state": state, "read_at": found})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="write the list that reproduces what a table laid")
    s.add_argument("--table", required=True, help="a delivery table's JSON (bench/delivery_table.py --json)")
    s.add_argument("--share", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--by", required=True, help="the session that seeds it")
    v = sub.add_parser("verify", help="print every line with its file's state now")
    v.add_argument("--laid", required=True)
    v.add_argument("--share", required=True)
    rf = sub.add_parser("refs", help="every line's reference as its render read it, against the file now")
    rf.add_argument("--laid", required=True)
    rf.add_argument("--ledger", required=True)
    rf.add_argument("--share", required=True)
    rf.add_argument("--json", help="the rows as JSON here")
    a = p.parse_args()
    if a.cmd == "refs":
        rows = refs(read(a.laid, a.share, check_bytes=False), json.loads(Path(a.ledger).read_text()), HERE.parent)
        for x in rows:
            print(f"{x['line']:3d} {x['first']:>4}-{x['last']:<4} {x['subject']:7} {x['load']:22} {x['state']:42} {x['reference'] or ''}")
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")
        flagged = [x for x in rows if not x["state"].startswith(("same bytes", "no digest kept; older"))]
        print(f"{len(rows)} reference rows for {len(set(x['line'] for x in rows))} lines; {len(flagged)} not shown to be the bytes the render read")
        return 0
    if a.cmd == "seed":
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists: the list is changed by a new line written by hand, never seeded over")
        lines = seed(a.table, a.share, a.by)
        Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in lines))
        print(f"wrote {a.out}: {len(lines)} lines from {a.table}; " + "; ".join(f"{sum(x['owner'] == o for x in lines)} {o}" for o in OWNER))
        return 0
    lines = read(a.laid, a.share, check_bytes=False)
    bad = 0
    for x in lines:
        f = Path(a.share) / x["file"]
        state = "MISSING" if not f.is_file() else ("bytes as laid" if sha256(f) == x["sha256"] else "CHANGED since laid")
        if state == "bytes as laid" and x.get("zoomed_sha256") and sha256(zoomed(f)) != x["zoomed_sha256"]:
            state = "ZOOMED COMPANION CHANGED"
        bad += state != "bytes as laid"
        print(f"{x['line']:3d} {x['first']:>4}-{x['last']:<4} {x['subject']:7} {x['load']:22} {x['owner']:21} {state}")
    counts = "; ".join(f"{sum(x['owner'] == o for x in lines)} {o}" for o in OWNER)
    print(f"{len(lines)} lines in {a.laid} (sha256 {digest(a.laid)[:16]}): {counts}; {bad} missing or changed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
