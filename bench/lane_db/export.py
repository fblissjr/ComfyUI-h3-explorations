"""Give a project's record files back from the database, in the shapes today's tools read: shots.json, plan.json,
cast/cast.jsonl, laid.jsonl and laid.removed.jsonl.

This is the bridge while the tools move onto the database one at a time, and the importer's proof: a file given back,
read as data, equals the file that was imported with the fields the mapping keeps raw taken out (`differences`; a
key whose value is null counts as a key that is not there).
Nothing here reads import_source.raw. Key order is this module's own; the comparison is of the data.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import cover
from .db import Refused


def _rows(con, sql, *args):
    return con.execute(sql, args).fetchall()


def _put(d: dict, **kv) -> dict:
    """Add the keys that have a value."""
    d.update({k: v for k, v in kv.items() if v is not None})
    return d


class Export:
    def __init__(self, con: sqlite3.Connection, project: str):
        self.con = con
        row = con.execute("SELECT p.*, v.name video, v.source_name, v.frames, v.film_words FROM project p JOIN video v ON v.id = p.video_id WHERE p.name = ?",
                          (project,)).fetchone()
        if row is None:
            raise Refused(f"the database holds no project {project!r}")
        self.p = row
        self.labels = [r["label"] for r in _rows(con, "SELECT label FROM subject WHERE video_id = ? ORDER BY ord", row["video_id"])]

    # ---- shots.json ------------------------------------------------------------------------------------------------
    def _who(self, shot_id: int, over_ord: int | None) -> dict:
        out = {}
        for a in _rows(self.con, """SELECT a.*, s.label FROM appearance a JOIN subject s ON s.id = a.subject_id
                                    WHERE a.shot_id = ? AND a.answered_over_ord IS ? ORDER BY a.id""", shot_id, over_ord):
            w = {"present": a["present"], "how": a["how"]}
            for kind in ("on", "boxed"):
                w[f"frames_{kind}"] = [r["frame"] for r in _rows(self.con, "SELECT frame FROM appearance_frame WHERE appearance_id = ? AND kind = ? ORDER BY ord", a["id"], kind)]
            _put(w, by=a["said_by"], frames_from=a["frames_from"])
            looks = [_put({"first": s["first"], "last": s["last"], "part": s["part"], "is": s["is_seen"]}, why=s["why"], by=s["said_by"],
                          entered_by=s["entered_by"], at=s["at"])
                     for s in _rows(self.con, "SELECT * FROM seen_look WHERE appearance_id = ? ORDER BY ord", a["id"])]
            if looks:
                w["seen_looks"] = looks
            out[a["label"]] = w
        return out

    def _shot(self, s) -> dict:
        return _put({"first": s["first"], "last": s["last"], "who": self._who(s["id"], None)}, scene_score=s["scene_score"], note=s["note"])

    def shots(self) -> dict:
        shots = []
        for s in _rows(self.con, "SELECT * FROM shot WHERE video_id = ? AND ord IS NOT NULL ORDER BY ord", self.p["video_id"]):
            d = self._shot(s)
            overs = _rows(self.con, "SELECT DISTINCT answered_over_ord o, answered_over_at at, answered_over_by by FROM appearance WHERE shot_id = ? AND answered_over_ord IS NOT NULL ORDER BY o", s["id"])
            if overs:
                d["answered_over"] = [{"at": o["at"], "by": o["by"], "who": self._who(s["id"], o["o"])} for o in overs]
            for how, key in (("merged", "merged_from"), ("split", "split_from"), ("moved", "moved_from")):
                olds = [self._shot(o) for o in _rows(self.con, """SELECT s.* FROM shot_from f JOIN shot s ON s.id = f.from_shot_id
                                                                        WHERE f.shot_id = ? AND f.how = ? ORDER BY f.ord""", s["id"], how)]
                if olds:
                    d[key] = olds[0] if how == "split" else olds
            order = _rows(self.con, """SELECT j.label, o.said_by FROM shot_order o JOIN subject j ON j.id = o.subject_id
                                       WHERE o.project_id = ? AND o.shot_id = ? ORDER BY o.ord""", self.p["id"], s["id"])
            if order:
                d["order"], d["order_by"] = [o["label"] for o in order], order[0]["said_by"]
            shots.append(d)
        cut_on = self.con.execute("SELECT c.name FROM video v JOIN copy c ON c.id = v.shots_copy_id WHERE v.id = ?", (self.p["video_id"],)).fetchone()
        return {"schema": "h3_shots/1", "video": self.p["source_name"], "source_frames": self.p["frames"], "copy": cut_on["name"],
                "span": [0, self.p["frames"] - 1], "subjects": self.labels, "shots": shots}

    # ---- cast/cast.jsonl -------------------------------------------------------------------------------------------
    def cast(self) -> list[dict]:
        out = []
        for r in _rows(self.con, """SELECT d.*, s.label FROM seed d JOIN subject s ON s.id = d.subject_id WHERE s.video_id = ? ORDER BY d.line""", self.p["video_id"]):
            d = {"answer": r["answer"], "at": r["at"], "box": None if r["x"] is None else [r["x"], r["y"], r["w"], r["h"]], "by": r["said_by"]}
            _put(d, drawn=r["drawn"], encloses=r["encloses"])
            d.update(frame=r["frame"], schema="h3_cast/1", size=None if r["picture_w"] is None else [r["picture_w"], r["picture_h"]],
                     subject=r["label"], table=r["from_table"])
            out.append(d)
        return out

    # ---- plan.json -------------------------------------------------------------------------------------------------
    def plan(self) -> dict:
        con, subjects = self.con, {}
        for sp in _rows(con, """SELECT sp.*, s.label, s.id sid, o.relpath still FROM subject_plan sp JOIN subject s ON s.id = sp.subject_id
                                JOIN location o ON o.id = sp.still_location_id WHERE sp.project_id = ? ORDER BY s.ord""", self.p["id"]):
            pid, sid = sp["id"], sp["sid"]
            removed = {r["what"]: r["note"] for r in _rows(con, "SELECT * FROM plan_removal WHERE subject_plan_id = ?", pid)}
            d = {"region": sp["region"], "by": sp["said_by"], "still": sp["still"],
                 "parts": [r["part"] for r in _rows(con, "SELECT part FROM plan_part WHERE subject_plan_id = ? ORDER BY ord", pid)]}
            _put(d, inputs_by=sp["inputs_by"])
            keep_outs = [r["name"] for r in _rows(con, "SELECT name FROM plan_keep_out WHERE subject_plan_id = ? ORDER BY ord", pid)]
            if keep_outs:
                d["keep_outs"] = keep_outs
            _put(d, keep_outs_by=sp["keep_outs_by"], keep_outs_removed_by=removed.get("keep_outs"), hair_band_removed_by=removed.get("hair_band"))
            restore = []
            for r in _rows(con, "SELECT * FROM restore_rule WHERE subject_plan_id = ? ORDER BY ord", pid):
                rule = {"classes": [c["part_class"] for c in _rows(con, "SELECT part_class FROM restore_class WHERE restore_rule_id = ? ORDER BY ord", r["id"])],
                        "first": r["first"], "last": r["last"]}
                _put(rule, touching=r["touching"], exception=r["exception"], exception_by=r["exception_by"])
                loads = [c["load_name"] for c in _rows(con, "SELECT load_name FROM restore_load WHERE restore_rule_id = ? ORDER BY ord", r["id"])]
                if loads:
                    rule["loads"] = loads
                restore.append(rule)
            if restore:
                d["restore"] = restore
            _put(d, restore_by=sp["restore_by"], restore_removed_by=removed.get("restore"))
            t = con.execute("SELECT * FROM subject_text WHERE subject_plan_id = ?", (pid,)).fetchone()
            text = _put({}, who=t["who"], looks=t["looks"], her=t["her"], she=t["she"], hair=t["hair"], looks_by=t["looks_by"],
                        hair_words=t["hair_words"], hair_words_by=t["hair_words_by"], picture_hair=t["picture_hair"], picture_hair_by=t["picture_hair_by"],
                        hair_change_removed_by=removed.get("text.hair_change"))
            if t["sings"] is not None:
                text["sings"] = bool(t["sings"])
            d["text"] = text
            look = {}
            change = [{"room": r["room"], "by": r["said_by"]} for r in _rows(con, "SELECT * FROM look_change WHERE subject_plan_id = ? ORDER BY ord", pid)]
            if change:
                look["change"] = change
            _put(look, stay_removed_by=removed.get("look.stay"))
            if look:
                d["look"] = look
            _put(d, look_by=sp["look_by"])
            voice = [{"first": r["first"], "last": r["last"], "voice": r["voice"], "by": r["said_by"]}
                     for r in _rows(con, "SELECT * FROM voice_span WHERE subject_id = ? ORDER BY ord", sid)]
            if voice:
                d["voice"] = voice
            _put(d, pieces_by_removed=removed.get("pieces"))
            clothing = [{"first": r["first"], "last": r["last"], "words": r["words"], "by": r["said_by"]}
                        for r in _rows(con, "SELECT * FROM clothing_words WHERE subject_plan_id = ? ORDER BY ord", pid)]
            if clothing:
                d["clothing"] = clothing
            d["depth"] = {"on": bool(sp["depth_on"]), "why": sp["depth_why"]}
            d["recipe"] = {"margin": sp["margin_px"], "zoom_up_to": sp["zoom_up_to"], "crop": sp["crop"], "open_mouth": sp["open_mouth"],
                           "closed_mouth": sp["closed_mouth"], "by": sp["recipe_by"]}
            marks = [{"first": r["first"], "last": r["last"], "mark": r["mark"], "by": r["said_by"]}
                     for r in _rows(con, "SELECT * FROM head_mark WHERE subject_id = ? ORDER BY ord", sid)]
            if marks:
                d["head_marks"] = marks
            hair = con.execute("SELECT * FROM facial_hair WHERE subject_id = ?", (sid,)).fetchone()
            if hair:
                d["source_facial_hair"], d["source_facial_hair_by"] = bool(hair["has"]), hair["said_by"]
            overrides = [_put({"first": r["first"], "last": r["last"]}, margin=r["margin_px"], zoom_up_to=r["zoom_up_to"], open_mouth=r["open_mouth"],
                              closed_mouth=r["closed_mouth"], why=r["why"], by=r["said_by"])
                         for r in _rows(con, "SELECT * FROM shot_override WHERE subject_plan_id = ? ORDER BY ord", pid)]
            if overrides:
                d["shot_overrides"] = overrides
            subjects[sp["label"]] = d
        aliases = {r["alias"]: r["label"] for r in _rows(con, """SELECT a.alias, s.label FROM alias a JOIN subject s ON s.id = a.subject_id
                                                                 WHERE s.video_id = ? ORDER BY a.ord""", self.p["video_id"])}
        by = con.execute("SELECT a.said_by FROM alias a JOIN subject s ON s.id = a.subject_id WHERE s.video_id = ? LIMIT 1", (self.p["video_id"],)).fetchone()
        if by and by["said_by"]:
            aliases["_by"] = by["said_by"]
        prefixes: dict = {}
        for r in _rows(con, "SELECT s.label, l.prefix FROM load_prefix l JOIN subject s ON s.id = l.subject_id WHERE l.project_id = ? ORDER BY s.ord, l.ord", self.p["id"]):
            prefixes.setdefault(r["label"], []).append(r["prefix"])
        _put(prefixes, _by=self.p["load_prefixes_by"])
        departures = [{"first": r["first"], "last": r["last"], "subject": r["label"], "note": r["note"], "by": r["said_by"]}
                      for r in _rows(con, "SELECT d.*, s.label FROM departure d JOIN subject s ON s.id = d.subject_id WHERE d.project_id = ? ORDER BY d.ord", self.p["id"])]
        not_people = [_put({"what": r["what"], "shots": [[x["first"], x["last"]] for x in _rows(con, "SELECT * FROM not_person_span WHERE not_person_id = ? ORDER BY ord", r["id"])],
                            "by": r["said_by"]}, counts_as_no_one=r["counts_as_no_one"])
                      for r in _rows(con, "SELECT * FROM not_person WHERE video_id = ? ORDER BY ord", self.p["video_id"])]
        out = {"schema": "h3_plan/1", "video": self.p["source_name"], "subjects": subjects, "aliases": aliases}
        if prefixes:
            out["load_prefixes"] = prefixes
        out.update(departures=departures, text={"film": self.p["film_words"]}, not_people=not_people)
        return out

    # ---- laid.jsonl and laid.removed.jsonl -------------------------------------------------------------------------
    def _laid(self, l) -> dict:
        con, lid = self.con, l["id"]
        states = _rows(con, "SELECT * FROM laid_state WHERE laid_id = ? ORDER BY ord", lid)
        word = lambda s: None if s["word_file"] is None else f"{s['word_file']}:{s['word_line']}"
        last = states[-1]
        d = {"schema": "h3_laid/1", "first": l["first"], "last": l["last"], "subject": l["label"], "load": l["load"], "file": l["file"],
             "sha256": l["sha256"], "owner": last["owner_status"], "by": last["why"],
             "frames": [[f["first"], f["last"]] for f in _rows(con, "SELECT * FROM laid_frames WHERE laid_id = ? ORDER BY ord", lid)],
             "added": l["added_at"], "added_by": l["added_by"]}
        _put(d, owner_word=word(last), zoomed_sha256=l["zoomed_sha256"])
        replaces = _rows(con, "SELECT * FROM laid_replaces WHERE laid_id = ? ORDER BY ord", lid)
        if replaces:
            def said(r) -> str:
                if r["sha256_start"] is None:
                    return r["load_name"]
                on = [[f["first"], f["last"]] for f in _rows(con, "SELECT * FROM laid_replaces_frames WHERE laid_id = ? AND replaces_ord = ? ORDER BY ord", lid, r["ord"])]
                return f"{r['load_name']} (sha256 {r['sha256_start']}, laid on {json.dumps(on)})"
            parts = [said(r) for r in replaces]
            if any(r["sha256_start"] for r in replaces):      # the long form is a sentence; names alone are a name or a list
                d["replaces"] = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
            else:
                d["replaces"] = parts[0] if len(parts) == 1 else parts
        _put(d, no_rebuild=l["no_rebuild_why"], partial_lay_why=l["partial_lay_why"], seeded_from=l["seeded_from"],
             owner_was=l["owner_before_correction"], owner_corrected=l["owner_corrected_why"])
        looks = [r["words"] for r in _rows(con, "SELECT words FROM laid_look WHERE laid_id = ? ORDER BY ord", lid)]
        if looks:
            d["owner_looks"] = looks
        if len(states) > 1:
            d["history"] = [{"at": s["at"], "by_session": s["by_session"], "owner_was": before["owner_status"], "by_was": before["why"],
                             "owner": s["owner_status"], "by": s["why"], "owner_word": word(s), "no_rebuild": s["no_rebuild_why"]}
                            for before, s in zip(states, states[1:])]
        carves = [{"load": c["load_name"],
                   "frames": [[f["first"], f["last"]] for f in _rows(con, "SELECT * FROM laid_carve_frames WHERE laid_id = ? AND carve_ord = ? ORDER BY ord", lid, c["ord"])],
                   "at": c["at"], "reason": c["reason"]} for c in _rows(con, "SELECT * FROM laid_carve WHERE laid_id = ? ORDER BY ord", lid)]
        if carves:
            d["carved_by"] = carves
        return d

    LAID = """SELECT l.*, s.label, d.name load, o.relpath file, a.sha256, z.sha256 zoomed_sha256
              FROM laid l JOIN subject s ON s.id = l.subject_id JOIN load d ON d.id = l.load_id JOIN location o ON o.id = l.render_location_id
              JOIN artifact a ON a.id = l.render_artifact_id LEFT JOIN artifact z ON z.id = l.zoomed_artifact_id """

    def laid(self, subjects: list[str] | None = None) -> list[dict]:
        """What the project lays, as a laid list: the pieces it keeps from an earlier project on the footage as they
        stand, in that project's order, then its own standing lines in theirs. `subjects` keeps only those subjects'."""
        ids = _rows(self.con, "SELECT id FROM project_laid WHERE for_project_id = ? ORDER BY kept_from_another DESC, file_id, list_line", self.p["id"])
        rows = [self.con.execute(self.LAID + "WHERE l.id = ?", (r["id"],)).fetchone() for r in ids]
        return [self._laid(l) for l in rows if subjects is None or l["label"] in subjects]

    def laid_removed(self) -> list[dict]:
        out = []
        for l in _rows(self.con, self.LAID + "WHERE l.project_id = ? AND l.removed_line IS NOT NULL ORDER BY l.file_id, l.removed_line", self.p["id"]):
            r = self.con.execute("SELECT * FROM laid_removal WHERE laid_id = ?", (l["id"],)).fetchone()
            out.append(_put({"schema": "h3_laid_removed/1", "at": r["at"], "by": r["why"], "by_session": r["by_session"], "removed": self._laid(l)},
                            no_rebuild=r["no_rebuild_why"], carved_whole_by=r["carved_whole_by"]))
        return out

    # ---- to files, and against the files ---------------------------------------------------------------------------
    def files(self) -> dict[str, tuple[str, object, bool]]:
        """relative path -> (the mapping's source, the data, whether it is a line a record)."""
        return {"shots.json": ("shots", self.shots(), False), "plan.json": ("plan", self.plan(), False), "cast/cast.jsonl": ("cast", self.cast(), True),
                "laid.jsonl": ("laid", self.laid(), True), "laid.removed.jsonl": ("laid-removed", self.laid_removed(), True)}

    def write(self, folder: Path) -> list[Path]:
        made = []
        for rel, (_, data, lines) in self.files().items():
            path = Path(folder) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data) if lines else json.dumps(data, ensure_ascii=False, indent=1) + "\n")
            made.append(path)
        return made


def _differ(a, b, path: str, out: list[str]) -> None:
    if len(out) >= 40:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        a, b = ({k: v for k, v in d.items() if v is not None} for d in (a, b))     # a key that is null is a key that is not there
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in the {'file' if k in a else 'database'} ({json.dumps(a.get(k, b.get(k)))[:90]})")
            else:
                _differ(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} in the file, {len(b)} from the database")
        for i, (x, y) in enumerate(zip(a, b)):
            _differ(x, y, f"{path}[{i}]", out)
    elif a != b or type(a) is not type(b):
        out.append(f"{path}: file {json.dumps(a)[:90]} database {json.dumps(b)[:90]}")


def differences(con: sqlite3.Connection, project: str, data: Path, mapping: cover.Mapping | None = None) -> dict[str, list[str]]:
    """Each record file in the project's folder against what the database gives back, as data, with the raw-only fields
    taken out of the file. For a project that keeps another's pieces, the laid list given back is those pieces and its
    own lines: a file that still holds the other project's remade pieces differs, and the difference says so."""
    mapping = mapping or cover.Mapping()
    ex = Export(con, project)
    labels = set(ex.labels)
    out, read_any = {}, False
    for rel, (source, mine, lines) in ex.files().items():
        path = Path(data) / ex.p["folder"] / rel
        if not path.is_file():                           # a later project on a video has no cast file of its own, and may have no removed list
            continue
        read_any = True
        text = path.read_text()
        theirs = [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)
        found: list[str] = []
        if source in ("shots", "plan"):
            header_key = "shots" if source == "shots" else "subjects"
            header = cover.without_raw(mapping, f"{source}-header", {k: v for k, v in theirs.items() if k != header_key}, labels)
            _differ(header, {k: v for k, v in mine.items() if k != header_key}, rel, found)
            a, b = theirs[header_key], mine[header_key]
            if source == "plan":
                if list(a) != list(b):
                    found.append(f"{rel}: subjects {list(a)} in the file, {list(b)} from the database")
                a, b = list(a.values()), list(b.values())
            _differ([cover.without_raw(mapping, source, r, labels) for r in a], b, f"{rel}:{header_key}", found)
        elif source == "laid":
            _laid_differ([cover.without_raw(mapping, source, r, labels) for r in theirs], mine, rel, found)
        else:
            _differ([cover.without_raw(mapping, source, r, labels) for r in theirs], mine, rel, found)
        out[rel] = found
    if not read_any:
        raise Refused(f"no record file of {project} is under {Path(data) / ex.p['folder']}: nothing was compared")
    return out


def _laid_differ(theirs: list[dict], mine: list[dict], rel: str, out: list[str]) -> None:
    """A laid list against the one given back, piece by piece: a piece is its load, its render's sha256 and its subject."""
    key = lambda r: (r["load"], r["sha256"], r["subject"])
    a, b = {key(r): r for r in theirs}, {key(r): r for r in mine}
    if len(a) != len(theirs) or len(b) != len(mine):
        return _differ(theirs, mine, rel, out)              # a piece twice in a list: line by line
    for what, only in (("in the file and not given back", [k for k in a if k not in b]), ("given back and not in the file", [k for k in b if k not in a])):
        if only:
            by_subject = ", ".join(f"{label} {sum(1 for k in only if k[2] == label)}" for label in sorted({k[2] for k in only}))
            out.append(f"{rel}: {len(only)} piece(s) {what} ({by_subject}): {', '.join(k[0] for k in only)}")
    both = [k for k in a if k in b]
    for k in both:
        _differ(a[k], b[k], f"{rel}:{k[0]}", out)
    if not out and [key(r) for r in theirs] != [key(r) for r in mine]:
        out.append(f"{rel}: the same pieces, in another order")
