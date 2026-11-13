"""The lane's record in one SQLite file: build it from a project's record files, hold it to them, give files back.

    <comfy venv python> bench/lane_record.py asks
    <comfy venv python> bench/lane_record.py prove --project fun --work SCRATCH_FOLDER
    <comfy venv python> bench/lane_record.py build --db data/lane.sqlite
    <comfy venv python> bench/lane_record.py compare --project fun2 --db data/lane.sqlite
    <comfy venv python> bench/lane_record.py export --project fun2 --db data/lane.sqlite --out FOLDER [--subjects second,third]
    <comfy venv python> bench/lane_record.py dump --db SCRATCH.sqlite

**What it buys.** The lane's state is thousands of JSON files that every tool joins its own way. This puts what a
person said and decided, the laid list and a project's closing cut into typed tables (bench/lane_db/migrations/), with
a project level above the video: footage facts are shared by the projects on a video, and a replacement's facts are
one project's. A second project on the same footage then starts from every footage fact, and lays the first project's
standing pieces of the subjects it keeps (`keeps` in data/projects.json) beside its own.

`asks`    the lines of bench/lane_db/mapping.tsv the owner has not ruled on, with the action a scratch build takes.
`prove`   builds twice into SCRATCH_FOLDER and prints three comparisons (bench/lane_db/prove.py): the two builds, each
          record file against what the database gives back, and the delivery table made from the database's files
          against the table on record for the project's final cut. Exits 1 on any difference. Needs H3_OUTPUT_DIR
          and H3_INPUT_DIR (the table tool checks the laid renders' bytes on the share).
`build`   one fresh database of every project of data/projects.json. The lane's own file, data/lane.sqlite, is refused
          while any mapping line is `ask`; it is written whole to a new file and moved into place, never built in place.
`compare` each record file in a project's folder against what the database gives back, as figures: which files are
          equal, and where one is not, what differs. For a project that keeps another's pieces, a laid list that
          still holds the other's remade pieces differs here. Exits 1 on any difference.
`export`  shots.json, plan.json, cast/cast.jsonl, laid.jsonl and laid.removed.jsonl from the database. A project's laid
          list is the pieces it keeps from an earlier project, then its own lines. With `--subjects`, only laid.jsonl
          is written, holding those subjects' pieces.
`dump`    every table as text, a row a line.

A build is a pure function of the files: nothing is updated in place, and a changed file means a new build. Tools
that write the database themselves come one at a time, later (docs/wiki/masked_v2v/2026-11-10_simplification_plan.md).
It reads no picture and hashes no render: what a record says a file's sha256 was is kept on the row that says it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import cover, db, export, importer, prove


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("asks")
    p = sub.add_parser("prove")
    p.add_argument("--project", required=True)
    p.add_argument("--work", required=True, type=Path)
    p = sub.add_parser("build")
    p.add_argument("--db", required=True, type=Path)
    p = sub.add_parser("compare")
    p.add_argument("--project", required=True)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--data", type=Path, default=db.REPO / "data", help="the data folder the record files are in (the repo's)")
    p = sub.add_parser("export")
    p.add_argument("--project", required=True)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--subjects", help="comma-separated labels: the laid list holds only their standing pieces")
    p = sub.add_parser("dump")
    p.add_argument("--db", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "asks":
            asks = cover.Mapping().asks()
            for r in asks:
                print(f"{r['source']:13s} {r['prefix']:18s} recommended: {r['target']:36s} {r['note']}")
            print(f"{len(asks)} line(s) of bench/lane_db/mapping.tsv are `ask`")
            return 0
        if a.cmd == "prove":
            return prove.prove(a.project, a.work.resolve())
        if a.cmd == "build":
            out = a.db.resolve()
            asks = cover.Mapping().asks()
            if out == db.THE_RECORD and asks:
                raise db.Refused(f"{len(asks)} line(s) of bench/lane_db/mapping.tsv are `ask`: the lane's own file is built once the owner has ruled "
                                 "(bench/lane_record.py asks); build to a scratch path until then")
            if out.is_relative_to(db.REPO) and out != db.THE_RECORD:
                raise db.Refused(f"{a.db} is inside the repo and is not data/lane.sqlite: build to a scratch path")
            new = out.with_name(out.name + ".new")
            con, notes = prove.build(new, db.REPO / "data")
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
            os.replace(new, out)
            for note in notes:
                print(f"note: {note}")
            print(f"built {out}")
            return 0
        con = db.connect(a.db)
        if a.cmd == "dump":
            sys.stdout.write(db.dump(con))
            return 0
        if a.cmd == "compare":
            found = export.differences(con, a.project, a.data)
            for rel, lines in found.items():
                print(f"{rel}: {'equal' if not lines else f'{len(lines)} difference(s)'}")
                for line in lines:
                    print(f"    {line[:400]}")
            return 1 if any(found.values()) else 0
        if a.cmd == "export":
            ex = export.Export(con, a.project)
            if a.subjects:
                keep = a.subjects.split(",")
                unknown = [s for s in keep if s not in ex.labels]
                if unknown:
                    raise db.Refused(f"{unknown} are not subjects of the video ({', '.join(ex.labels)})")
                a.out.mkdir(parents=True, exist_ok=True)
                rows = ex.laid(keep)
                (a.out / "laid.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                print(f"wrote {a.out / 'laid.jsonl'}: {len(rows)} standing piece(s) of {', '.join(keep)}")
            else:
                for path in ex.write(a.out):
                    print(f"wrote {path}")
            return 0
    except db.Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    return 0
