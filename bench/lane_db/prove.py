"""Hold a build of a closed project to what today's tools made from its files: the delivery table of its final cut.

Three things are compared, and each is printed as figures:
  1. two builds of the same files, row for row (the build is a function of the files);
  2. every record file against what the database gives back (export.differences);
  3. the delivery table bench/delivery_table.py writes from the files the database gives back, against the table on
     record for the project's final cut: every section but when and by what command it was made.

What it cannot see: the picture, the renders' bytes beyond what the table tool itself checks, and anything the first
tier does not hold. Two things the table tool reads are second-tier and are read from their files: the ledger on
record, and each load's job file (linked beside the database's files, for which cut a piece was made on).

The table on record names each capture as `data/<day>_<load>`. Where that folder is now `data/<project>/<day>_<load>`
(fun's were moved on 2026-11-09), the ledger is read with the capture at its new place, and the comparison takes the
table's `run=` with the project's folder left out. Without that, today's tool on today's files ties no row to its run.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

from . import cover, db, export, importer


def build(path: Path, data: Path) -> tuple:
    """A fresh database of every project of data/projects.json, and the importer's notes."""
    for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        p.unlink(missing_ok=True)
    con = db.connect(path, create=True)
    notes = importer.Import(con, data).run()
    return con, notes


def moved(path: str, folder: str) -> str:
    """A recorded repo path, at its place today: `data/<x>` read as `data/<folder>/<x>` where only that is there."""
    rel = Path(path).relative_to(db.REPO) if Path(path).is_absolute() and Path(path).is_relative_to(db.REPO) else Path(path)
    if (db.REPO / rel).exists() or rel.parts[:1] != ("data",):
        return str(rel)
    there = Path("data", folder, *rel.parts[1:])
    return str(there) if (db.REPO / there).exists() else str(rel)


def table_from(con, project: str, work: Path, python: str) -> tuple[dict, list[str], dict, list[str], list[str]]:
    """The delivery table from the database's files, the table on record, and what was read from a moved place."""
    ex = export.Export(con, project)
    folder = ex.p["folder"]
    cut = con.execute("SELECT c.*, o.relpath FROM cut c JOIN location o ON o.id = c.location_id WHERE c.project_id = ? AND c.is_final = 1", (ex.p["id"],)).fetchone()
    if cut is None:
        raise db.Refused(f"the database holds no final cut of {project}")
    record = next(db.REPO.glob(f"data/{folder}/*_delivery_tables/rebuild_{folder}_{cut['stamp']}.json"))
    rec = json.loads(record.read_text())
    on_record = json.loads((record.parent / Path(rec["artifacts"]["table_json"]).name).read_text())
    on_record_txt = (record.parent / Path(rec["artifacts"]["table_txt"]).name).read_text().splitlines()
    files = work / folder                                  # named as the project's folder: the laid tool reads the video's name from it
    ex.write(files)
    jobs = 0
    for job in sorted((db.REPO / "data" / folder).glob("*_job")):      # the second tier's files, read at their place
        link = files / job.name
        if not link.exists():
            link.symlink_to(job, target_is_directory=True)
        jobs += 1
    ledger = json.loads((record.parent / Path(rec["artifacts"]["ledger_json"]).name).read_text())
    notes, n = [f"{jobs} job folder(s) of data/{folder} linked beside the database's files: the laid tool reads which cut a piece was made on "
                 "from its job file, which the first tier does not hold"], 0
    for render in ledger["renders"].values():
        for r in render.get("records") or []:
            if r.get("capture") and moved(r["capture"], folder) != r["capture"]:
                r["capture"], n = moved(r["capture"], folder), n + 1
    if n:
        notes.append(f"{n} capture path(s) of the ledger on record read at data/{folder}/ (the folders moved)")
    (work / "ledger.json").write_text(json.dumps(ledger))
    was = shlex.split(on_record["assembly"]["own_command"])
    args, i = [python, "bench/delivery_table.py"], 2
    mine = {"--ledger": work / "ledger.json", "--shots": files / "shots.json", "--plan": files / "plan.json", "--laid": files / "laid.jsonl",
            "--table": work / "t.txt", "--reading": work / "reading.md", "--json": work / "t.json", "--stack-work": work / "stack"}
    while i < len(was):
        flag, value = was[i], was[i + 1]
        args += [flag, str(mine[flag]) if flag in mine else moved(value, folder) if flag == "--also-capture" else value]
        i += 2
    done = subprocess.run(args, cwd=db.REPO, capture_output=True, text=True)
    if done.returncode:
        raise db.Refused(f"bench/delivery_table.py exited {done.returncode} on the database's files: {(done.stderr or done.stdout)[-1500:]}")
    made = (work / "t.json").read_text().replace(f"data/{folder}/2", "data/2")
    made_txt = (work / "t.txt").read_text().replace(f"data/{folder}/2", "data/2").splitlines()
    return json.loads(made), made_txt, on_record, on_record_txt, notes


def prove(project: str, work: Path, data: Path | None = None, python: str = sys.executable) -> int:
    data = data or db.REPO / "data"
    work.mkdir(parents=True, exist_ok=True)
    bad = 0
    con, notes = build(work / "lane_a.sqlite", data)
    con_b, _ = build(work / "lane_b.sqlite", data)
    for note in notes:
        print(f"  note: {note}")
    a, b = db.dump(con), db.dump(con_b)
    rows = sum(1 for line in a.splitlines() if not line.startswith("## "))
    print(f"1. two builds of every project of data/projects.json: {rows} rows each over {len(db.tables(con)) - 1} tables; {'equal' if a == b else 'DIFFER'}")
    bad += a != b
    print("2. each record file against what the database gives back (raw-only fields left out of the file):")
    for rel, found in export.differences(con, project, data).items():
        print(f"     {rel}: {'equal' if not found else f'{len(found)} DIFFERENCE(S)'}")
        for f in found[:10]:
            print(f"        {f[:300]}")
        bad += bool(found)
    made, made_txt, was, was_txt, moved_notes = table_from(con, project, work, python)
    print("3. the delivery table from the database's files against the final cut's table on record (count on record, count made):")
    for note in moved_notes:
        print(f"     note: {note}")
    for key in sorted((set(was) | set(made)) - {"made", "assembly"}):
        same = was.get(key) == made.get(key)
        print(f"     {key:40s} {len(was.get(key, [])):4d} {len(made.get(key, [])):4d}  {'equal' if same else 'DIFFERS'}")
        bad += not same
    body = lambda lines: [x for x in lines if not x.startswith("#")]
    notes_ = lambda lines: [x for x in lines if x.startswith("#")][1:]
    for what, f in (("rows of the .txt", body), ("comment lines of the .txt after its first", notes_)):
        same = f(was_txt) == f(made_txt)
        print(f"     {what:40s} {len(f(was_txt)):4d} {len(f(made_txt)):4d}  {'equal' if same else 'DIFFER'}")
        bad += not same
    pieces = con.execute("SELECT count(*) FROM cut_piece p JOIN cut c ON c.id = p.cut_id JOIN project j ON j.id = c.project_id WHERE j.name = ?", (project,)).fetchone()[0]
    print(f"   the cut's {pieces} piece(s) in the database are the table's rows on record, each tied to one standing laid line (the importer refuses otherwise)")
    asks = cover.Mapping().asks()
    if asks:
        print(f"{len(asks)} line(s) of mapping.tsv are `ask`: this build took the recommended action on each")
    print(f"the builds, the files and the table are under {work}")
    return 1 if bad else 0
