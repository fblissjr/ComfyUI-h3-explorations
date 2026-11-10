"""Hold a record's field paths to mapping.tsv: every path is covered by its longest prefix, or the import is refused.

A key that is data is collapsed before it is matched: a subject's label to <subject>, a number to <n>, a frame span to
<span>. The mapping is the list of every field the lane's record files have grown, with where each goes.
"""
from __future__ import annotations

import re
from pathlib import Path

from .db import Refused

MAPPING = Path(__file__).resolve().parent / "mapping.tsv"
ACTIONS = {"col", "child", "said_by", "history", "artifact", "view", "meta", "raw", "ask"}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
SPAN = re.compile(r"^\d+-\d+$")


class Mapping:
    def __init__(self, path: Path = MAPPING):
        self.rules: dict[str, list[dict]] = {}
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            src, prefix, action, target, note = (line.split("\t") + ["", "", ""])[:5]
            if action not in ACTIONS:
                raise Refused(f"{path.name}:{n}: {action!r} is not an action ({', '.join(sorted(ACTIONS))})")
            self.rules.setdefault(src, []).append({"prefix": prefix, "action": action, "target": target, "note": note, "line": n, "hits": 0})

    def rule(self, source: str, path: str) -> dict | None:
        best = None
        for r in self.rules.get(source, ()):
            p = r["prefix"]
            if (path == p or path.startswith(p + ".") or path.startswith(p + "[")) and (best is None or len(p) > len(best["prefix"])):
                best = r
        return best

    def acts(self, source: str, path: str) -> str:
        """What happens to a path: its action, with an `ask` read as its recommended action's first word."""
        r = self.rule(source, path)
        if r is None:
            raise Refused(f"{source}: the field {path!r} is in no line of {MAPPING.name}")
        return r["target"].split()[0] if r["action"] == "ask" else r["action"]

    def asks(self) -> list[dict]:
        return [dict(r, source=s) for s, rs in self.rules.items() for r in rs if r["action"] == "ask"]


def paths(record, labels: set[str], path: str = "") -> set[str]:
    """Every field path of a record, data keys collapsed."""
    out: set[str] = set()
    if isinstance(record, dict):
        for k, v in record.items():
            key = "<subject>" if k in labels else "<n>" if NUM.match(k) else "<span>" if SPAN.match(k) else k
            p = f"{path}.{key}" if path else key
            out.add(p)
            out |= paths(v, labels, p)
    elif isinstance(record, list):
        for v in record:
            out |= paths(v, labels, f"{path}[]")
    return out


def hold(mapping: Mapping, source: str, records, labels: set[str]) -> None:
    """Refuse when any record of a source has a field no line covers."""
    missing: dict[str, int] = {}
    for rec in records:
        for p in paths(rec, labels):
            r = mapping.rule(source, p)
            if r is None:
                missing[p] = missing.get(p, 0) + 1
            else:
                r["hits"] += 1
    if missing:
        listed = "; ".join(f"{p} ({n} record(s))" for p, n in sorted(missing.items()))
        raise Refused(f"{source}: {len(missing)} field path(s) are in no line of {MAPPING.name}, so nothing is imported: {listed}")


def without_raw(mapping: Mapping, source: str, record, labels: set[str], path: str = ""):
    """A record with every field the mapping keeps raw only taken out: what the database must give back."""
    if isinstance(record, dict):
        out = {}
        for k, v in record.items():
            key = "<subject>" if k in labels else "<n>" if NUM.match(k) else "<span>" if SPAN.match(k) else k
            p = f"{path}.{key}" if path else key
            if mapping.acts(source, p) == "raw":
                continue
            out[k] = without_raw(mapping, source, v, labels, p)
        return out
    if isinstance(record, list):
        return [without_raw(mapping, source, v, labels, f"{path}[]") for v in record]
    return record
