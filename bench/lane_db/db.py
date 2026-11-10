"""Open the lane's database: foreign keys on, the schema brought up to date, the roots named.

The file is never in git and never on the media share (SQLite's locking is not safe on a network filesystem).
A place is a root's name and a relative path; the roots are resolved here, at run time, and no absolute path is stored.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MIGRATIONS = HERE / "migrations"
THE_RECORD = REPO / "data" / "lane.sqlite"

# name -> (what it is, how it is found at run time)
ROOTS = {
    "data": ("the repo's data/ folder, where the lane's records are", lambda: REPO / "data"),
    "share_output": ("the media share's output folder (H3_OUTPUT_DIR)", lambda: _env("H3_OUTPUT_DIR")),
    "share_input": ("the media share's input folder (H3_INPUT_DIR)", lambda: _env("H3_INPUT_DIR")),
}


class Refused(Exception):
    """The database or the importer will not go on, and says why."""


def _env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def root_path(name: str) -> Path:
    found = ROOTS[name][1]()
    if found is None:
        raise Refused(f"the root {name!r} is not set: {ROOTS[name][0]}")
    return found


def migrations() -> list[tuple[int, Path]]:
    found = sorted((int(p.name.split("_")[0]), p) for p in MIGRATIONS.glob("[0-9]*.sql"))
    if [n for n, _ in found] != list(range(1, len(found) + 1)):
        raise Refused(f"the migrations are not numbered 1..N with none missing: {[p.name for _, p in found]}")
    return found


def connect(path: Path | str, create: bool = False) -> sqlite3.Connection:
    """A connection with foreign keys on and the schema at the newest migration. A new file is made only with `create`."""
    path = Path(path)
    if not create and not path.is_file():
        raise Refused(f"no database at {path}")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode = WAL")
    have = con.execute("PRAGMA user_version").fetchone()[0]
    todo = [(n, p) for n, p in migrations() if n > have]
    if have > len(migrations()):
        raise Refused(f"{path} is at schema {have}, newer than this code's {len(migrations())}")
    con.execute("PRAGMA foreign_keys = OFF")              # a migration may make a table again; the links are checked after
    for n, p in todo:
        con.executescript("BEGIN;\n" + p.read_text() + f"\nPRAGMA user_version = {n};\nCOMMIT;")
    con.execute("PRAGMA foreign_keys = ON")
    if not con.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise Refused("this SQLite does not enforce foreign keys")
    if todo:
        broken = con.execute("PRAGMA foreign_key_check").fetchall()
        if broken:
            raise Refused(f"after migrating {path} to schema {todo[-1][0]}, {len(broken)} row(s) point at a row that is not there")
        for name, (what, _) in ROOTS.items():
            con.execute("INSERT OR IGNORE INTO root (name, what) VALUES (?, ?)", (name, what))
        con.commit()
    return con


def tables(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def dump(con: sqlite3.Connection, leave_out: tuple[str, ...] = ("import_run",)) -> str:
    """Every table as text, a row a line, in primary-key order: what two builds are compared by, and a diff a person can read.
    `import_run` holds the build's own time and is left out."""
    out = []
    for t in tables(con):
        if t in leave_out:
            continue
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        out.append(f"## {t} ({', '.join(cols)})")
        for row in con.execute(f"SELECT * FROM {t} ORDER BY {', '.join(str(i + 1) for i in range(len(cols)))}"):
            out.append("\t".join("" if v is None else repr(v) for v in row))
    return "\n".join(out) + "\n"
