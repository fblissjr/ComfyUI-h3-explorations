"""The lane's record in one SQLite file: build it from a project's record files, hold it to them, give files back.

    <comfy venv python> bench/lane_record.py asks
    <comfy venv python> bench/lane_record.py prove --project fun --work SCRATCH_FOLDER
    <comfy venv python> bench/lane_record.py build --project fun --db SCRATCH.sqlite
    <comfy venv python> bench/lane_record.py export --project fun --db SCRATCH.sqlite --out FOLDER [--subjects second,third]
    <comfy venv python> bench/lane_record.py dump --db SCRATCH.sqlite

**What it buys.** The lane's state is thousands of JSON files that every tool joins its own way. This puts what a
person said and decided, the laid list and a project's closing cut into typed tables (bench/lane_db/migrations/), with
a project level above the video: footage facts are shared by the projects on a video, and a replacement's facts are
one project's. A second project on the same footage then starts from every footage fact, and from the first project's
laid pieces of the subjects it does not remake (`export --subjects`).

`asks`    the lines of bench/lane_db/mapping.tsv the owner has not ruled on, with the action a scratch build takes.
`prove`   builds twice into SCRATCH_FOLDER and prints three comparisons (bench/lane_db/prove.py): the two builds, each
          record file against what the database gives back, and the delivery table made from the database's files
          against the table on record for the project's final cut. Exits 1 on any difference. Needs H3_OUTPUT_DIR
          and H3_INPUT_DIR (the table tool checks the laid renders' bytes on the share).
`build`   one fresh database from the project's files. The lane's own file, data/lane.sqlite, is refused while any
          mapping line is `ask`; it is written whole to a new file and moved into place, never built in place.
`export`  shots.json, plan.json, cast/cast.jsonl, laid.jsonl and laid.removed.jsonl from the database. With
          `--subjects`, the laid list holds only those subjects' standing pieces and no removed list is written.
`dump`    every table as text, a row a line.

A build is a pure function of the files: nothing is updated in place, and a changed file means a new build. Tools
that write the database themselves come one at a time, later (docs/masked_v2v/2026-11-10_simplification_plan.md).
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
    p.add_argument("--project", required=True)
    p.add_argument("--db", required=True, type=Path)
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
            con, notes = prove.build(new, db.REPO / "data", a.project)
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
