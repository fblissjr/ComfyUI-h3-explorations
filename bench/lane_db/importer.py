"""Build the lane's database from the projects' record files, the first tier: what a person said and decided, the laid
lists, and the cut that closed a project.

Every project of data/projects.json is imported, a video at a time. The first project on a video brings the footage's
facts (copies, shots, who is on them, seeds, answers, and the footage facts its plan file carries). A later project on
the same video brings only its own: its plan per subject, its laid lines, the subjects it keeps from the earlier
project. Its shots file must be the footage's. The footage facts its plan file carries are kept as that plan's word,
and the latest plan's word stands (the `*_standing` views); where it differs from the plan before it, a note names
the fields.

Every record is held to mapping.tsv before anything is written (cover.hold): a field no line covers stops the import.
Each record goes into import_source verbatim, tied to the row made from it. A build is a pure function of the files:
it is made fresh each time, and two builds of the same files give the same rows (db.dump leaves out only the build's time).

What this tier cannot see: the picture, and anything the records do not say. It does not hash the files a record
names; a place gets its bytes from a hashing pass (cli.py verify), and what a record says was there is kept on the row.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

from . import cover
from .db import REPO, Refused, ROOTS, root_path

LAID_SENTENCE = re.compile(r"^(?P<name>\S+)(?: \(sha256 (?P<sha>[0-9a-f]+), laid on (?P<on>\[\[.*?\]\])\))?$")
MADE = re.compile(r"at commit (?P<commit>[0-9a-f]+) from the ledger .*\(sha256 (?P<sha>[0-9a-f]+), (?P<lines>\d+) lines")
WORD = re.compile(r"^(?P<file>.+):(?P<line>\d+)$")
REMOVALS = {"keep_outs_removed_by": "keep_outs", "hair_band_removed_by": "hair_band", "pieces_by_removed": "pieces",
            "restore_removed_by": "restore"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spans(value) -> list[tuple[int, int]]:
    """A `frames` value as spans: [first, last], or a list of those."""
    if value and isinstance(value[0], int):
        if len(value) != 2:
            raise Refused(f"a span of {len(value)} numbers: {value}")
        return [(value[0], value[1])]
    return [(a, b) for a, b in value]


class Import:
    def __init__(self, con: sqlite3.Connection, data: Path, mapping: cover.Mapping | None = None):
        self.con, self.data = con, Path(data)
        self.map = mapping or cover.Mapping()
        self.notes: list[str] = []
        self.labels: set[str] = set()
        self.subjects: dict[str, int] = {}
        self.files: dict[str, int] = {}
        self.answer_ids: dict[tuple[str, int], int] = {}

    # ---- small writers -------------------------------------------------------------------------------------------
    def ins(self, table: str, **cols) -> int:
        cols = {k: (int(v) if isinstance(v, bool) else v) for k, v in cols.items()}
        cur = self.con.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", list(cols.values()))
        return cur.lastrowid

    def artifact(self, sha: str, size: int | None = None) -> int:
        row = self.con.execute("SELECT id, bytes FROM artifact WHERE sha256 = ?", (sha,)).fetchone()
        if row:
            if size is not None and row["bytes"] is None:
                self.con.execute("UPDATE artifact SET bytes = ? WHERE id = ?", (size, row["id"]))
            return row["id"]
        return self.ins("artifact", sha256=sha, bytes=size)

    def location(self, root: str, relpath: str) -> int:
        row = self.con.execute("SELECT id FROM location WHERE root = ? AND relpath = ?", (root, relpath)).fetchone()
        return row["id"] if row else self.ins("location", root=root, relpath=relpath)

    def place(self, text: str, default: str) -> int:
        """A path as a record wrote it, as a place: an absolute path under a root, `Video/...` under the output root,
        `data/...` under the data root, anything else under `default`."""
        if text.startswith("/"):
            for name in ROOTS:
                base = ROOTS[name][1]()
                if base is not None and Path(text).is_relative_to(base):
                    return self.location(name, str(Path(text).relative_to(base)))
            raise Refused(f"the absolute path {text!r} is under no root that is set ({', '.join(ROOTS)}): set H3_OUTPUT_DIR and H3_INPUT_DIR")
        if text.startswith("data/"):
            return self.location("data", text[len("data/"):])
        if text.startswith("Video/"):
            return self.location("share_output", text)
        return self.location(default, text)

    def read(self, rel: str, source: str) -> tuple[str, int]:
        """A record file's text, and its import_file row: the place and the bytes this build read."""
        path = self.data / rel
        if not path.is_file():
            raise Refused(f"{source}: no file at data/{rel}")
        raw = path.read_bytes()
        self.files[rel] = self.ins("import_file", location_id=self.location("data", rel),
                                   artifact_id=self.artifact(hashlib.sha256(raw).hexdigest(), len(raw)), source=source)
        return raw.decode("utf-8"), self.files[rel]

    def src(self, file_id: int, line: int | None, record, table: str | None = None, row: int | None = None) -> None:
        self.ins("import_source", file_id=file_id, line=line, raw=json.dumps(record, ensure_ascii=False), into_table=table, into_id=row)

    def subject(self, label: str) -> int:
        if label not in self.subjects:
            raise Refused(f"{label!r} is not a subject of the video ({', '.join(self.subjects)})")
        return self.subjects[label]

    def project_at(self, at: str) -> int:
        """Whose a dated record is: the project of this video whose time it falls in."""
        t = at.replace("Z", "")
        for p in self.projects:
            if (p["starts_at"] is None or p["starts_at"].replace("Z", "") <= t) and (p["ends_before"] is None or t < p["ends_before"].replace("Z", "")):
                return p["id"]
        raise Refused(f"no project of the video holds the time {at}")

    # ---- the projects file ---------------------------------------------------------------------------------------
    def projects_file(self) -> list[list[dict]]:
        """Every project of the file as a row, and the projects grouped by video, in the file's order."""
        text, fid = self.read("projects.json", "projects")
        doc = json.loads(text)
        if doc.get("schema") != "h3_projects/1":
            raise Refused(f"data/projects.json is {doc.get('schema')!r}, not h3_projects/1")
        cover.hold(self.map, "projects", [{k: v for k, v in doc.items() if k != "projects"}] + doc["projects"], set())
        videos: dict[str, list[dict]] = {}
        for p in doc["projects"]:
            videos.setdefault(p["video"], []).append(p)
        for name, found in videos.items():
            footage = found[0]["footage_records"]
            if found[0]["folder"] != footage or any(p["footage_records"] != footage for p in found):
                raise Refused(f"data/projects.json: the projects on {name!r} do not all read the footage records of the first of them ({footage!r})")
            shots = json.loads((self.data / footage / "shots.json").read_text())
            vid = self.ins("video", name=name, source_name=shots["video"], frames=shots["source_frames"])
            for p in found:
                p["video_id"] = vid
                p["id"] = self.ins("project", name=p["name"], video_id=vid, folder=p["folder"], starts_at=p["starts_at"], ends_before=p["ends_before"],
                                   said_by=p["by"])
                self.src(fid, None, {k: v for k, v in p.items() if k not in ("id", "video_id")}, "project", p["id"])
        return list(videos.values())

    def enter(self, p: dict) -> None:
        self.entry, self.name, self.folder, self.project_id = p, p["name"], p["folder"], p["id"]

    # ---- footage: copies, shots, who is on them, seeds -----------------------------------------------------------
    def manifest(self) -> None:
        text, fid = self.read(f"{self.folder}/manifest.json", "manifest")
        doc = json.loads(text)
        cover.hold(self.map, "manifest", [doc], self.labels)
        self.copies: dict[str, int] = {}
        entries = [("source", doc["source"])] + [("working", c) for c in doc.get("copies_before", [])] + [("working", doc["derived"])]
        for ord_, (kind, c) in enumerate(entries):
            self.copies[c["name"]] = cid = self.ins(
                "copy", video_id=self.video_id, kind=kind, name=c["name"], rate=c["rate"], frames=c["frames"], width=c["size"][0],
                height=c["size"][1], bytes=c["bytes"], head_sha256=c["head_sha256"], first_source_frame=c.get("first_source_frame"),
                checked=c.get("checked"), made_by=c.get("made_by"), entered_at=c.get("entered_at"), entered_by=c.get("entered_by"),
                replaced_at=c.get("replaced_at"), replaced_by=c.get("replaced_by"), why=c.get("why"), ord=ord_)
            self.src(fid, None, c, "copy", cid)
        self.working_copy = doc["derived"]["name"]
        self.src(fid, None, {k: v for k, v in doc.items() if k not in ("source", "derived", "copies_before")})

    def appearances(self, shot_id: int, who: dict, over: dict | None = None, over_ord: int | None = None) -> None:
        for label, w in who.items():
            aid = self.ins("appearance", shot_id=shot_id, subject_id=self.subject(label), present=w["present"], how=w.get("how"),
                           said_by=w["by"], frames_from=w.get("frames_from"),
                           answered_over_at=over and over["at"], answered_over_by=over and over["by"], answered_over_ord=over_ord)
            for kind in ("on", "boxed"):
                for i, f in enumerate(w.get(f"frames_{kind}") or []):
                    self.ins("appearance_frame", appearance_id=aid, kind=kind, ord=i, frame=f)
            for i, s in enumerate(w.get("seen_looks") or []):
                self.ins("seen_look", appearance_id=aid, ord=i, first=s["first"], last=s["last"], part=s["part"], is_seen=s["is"],
                         why=s.get("why"), said_by=s["by"], entered_by=s.get("entered_by"), at=s["at"])

    def shots(self) -> None:
        text, fid = self.read(f"{self.folder}/shots.json", "shots")
        doc = json.loads(text)
        header = {k: v for k, v in doc.items() if k != "shots"}
        self.labels = set(doc["subjects"])
        cover.hold(self.map, "shots-header", [header], self.labels)
        cover.hold(self.map, "shots", doc["shots"], self.labels)
        if doc["schema"] != "h3_shots/1" or doc["span"] != [0, doc["source_frames"] - 1]:
            raise Refused(f"shots.json: schema {doc['schema']!r}, span {doc['span']} of {doc['source_frames']} frames")
        if doc["copy"] not in self.copies:
            raise Refused(f"shots.json was cut on {doc['copy']!r}, which the video's manifest does not name as a copy")
        self.con.execute("UPDATE video SET shots_copy_id = ? WHERE id = ?", (self.copies[doc["copy"]], self.video_id))
        for i, label in enumerate(doc["subjects"]):
            self.subjects[label] = self.ins("subject", video_id=self.video_id, label=label, ord=i)
        self.src(fid, None, header, "video", self.video_id)
        earlier: dict[str, int] = {}                      # an earlier shot, by its whole record: one row however many shots came from it
        self.shot_ids: dict[tuple[int, int], int] = {}

        def shot_row(rec: dict, ord_: int | None) -> int:
            sid = self.ins("shot", video_id=self.video_id, first=rec["first"], last=rec["last"], scene_score=rec.get("scene_score"),
                           note=rec.get("note"), ord=ord_)
            self.appearances(sid, rec["who"])
            return sid

        for i, rec in enumerate(doc["shots"]):
            sid = self.shot_ids[(rec["first"], rec["last"])] = shot_row(rec, i)
            self.src(fid, i, rec, "shot", sid)
            for n, over in enumerate(rec.get("answered_over") or []):
                self.appearances(sid, over["who"], over, n)
            for how, key in (("merged", "merged_from"), ("split", "split_from"), ("moved", "moved_from")):
                olds = rec.get(key)
                for n, old in enumerate([olds] if isinstance(olds, dict) else olds or []):
                    ident = json.dumps(old, sort_keys=True)
                    if ident not in earlier:
                        earlier[ident] = shot_row(old, None)
                    self.ins("shot_from", shot_id=sid, from_shot_id=earlier[ident], how=how, ord=n)
        self.footage_shots = doc

    def shot_orders(self) -> None:
        """Which subject is made first on a shot is the project's. A later project's shots file must be the footage's."""
        doc = self.footage_shots
        if self.folder != self.projects[0]["folder"]:
            path = self.data / self.folder / "shots.json"
            if not path.is_file():
                self.notes.append(f"{self.name}: no shots.json in data/{self.folder}; it has no shot orders of its own")
                return
            text, fid = self.read(f"{self.folder}/shots.json", "shots")
            doc = json.loads(text)
            cover.hold(self.map, "shots-header", [{k: v for k, v in doc.items() if k != "shots"}], self.labels)
            cover.hold(self.map, "shots", doc["shots"], self.labels)
            bare = lambda d: [{k: v for k, v in rec.items() if k not in ("order", "order_by")} for rec in d["shots"]]
            if bare(doc) != bare(self.footage_shots) or {k: v for k, v in doc.items() if k != "shots"} != {k: v for k, v in self.footage_shots.items() if k != "shots"}:
                raise Refused(f"data/{self.folder}/shots.json is not the footage's shots (data/{self.projects[0]['folder']}/shots.json): a project on the "
                              "same video reads the footage's shots, and only the order of subjects on a shot is its own")
            self.src(fid, None, {k: v for k, v in doc.items() if k != "shots"}, "project", self.project_id)
        for rec in doc["shots"]:
            for n, label in enumerate(rec.get("order") or []):
                self.ins("shot_order", project_id=self.project_id, shot_id=self.shot_ids[(rec["first"], rec["last"])], subject_id=self.subject(label),
                         ord=n, said_by=rec["order_by"])

    def cast(self) -> None:
        text, fid = self.read(f"{self.folder}/cast/cast.jsonl", "cast")
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "cast", [r for _, r in recs], self.labels)
        for n, r in recs:
            if r["schema"] != "h3_cast/1":
                raise Refused(f"cast.jsonl:{n}: schema {r['schema']!r}")
            box, size = r.get("box") or [None] * 4, r.get("size") or [None] * 2
            sid = self.ins("seed", subject_id=self.subject(r["subject"]), file_id=fid, line=n, frame=r["frame"], x=box[0], y=box[1], w=box[2], h=box[3],
                           picture_w=size[0], picture_h=size[1], answer=r["answer"], encloses=r.get("encloses"), drawn=r.get("drawn"),
                           from_table=r["table"], said_by=r["by"], at=r["at"])
            self.src(fid, n, r, "seed", sid)

    # ---- the plan --------------------------------------------------------------------------------------------------
    @staticmethod
    def footage_facts(doc: dict) -> dict:
        """The footage facts a plan file carries, by where they sit in it."""
        out = {"text.film": doc["text"]["film"], "aliases": doc.get("aliases"), "not_people": doc.get("not_people")}
        for label, p in doc["subjects"].items():
            for key in ("head_marks", "voice", "source_facial_hair", "source_facial_hair_by"):
                out[f"subjects.{label}.{key}"] = p.get(key)
        return out

    def plan(self) -> None:
        first = self.folder == self.projects[0]["folder"]
        if not first and not (self.data / self.folder / "plan.json").is_file():
            self.notes.append(f"{self.name}: no plan.json in data/{self.folder}; it has no plan in the record")
            return
        text, fid = self.read(f"{self.folder}/plan.json", "plan")
        doc = json.loads(text)
        header = {k: v for k, v in doc.items() if k != "subjects"}
        cover.hold(self.map, "plan-header", [header], self.labels)
        cover.hold(self.map, "plan", list(doc["subjects"].values()), self.labels)
        video = self.con.execute("SELECT source_name FROM video WHERE id = ?", (self.video_id,)).fetchone()[0]
        if doc["schema"] != "h3_plan/1" or doc["video"] != video:
            raise Refused(f"plan.json: schema {doc['schema']!r}, video {doc['video']!r} (the shots file says {video!r})")
        if set(doc["text"]) != {"film"}:
            raise Refused(f"plan.json: text holds {sorted(doc['text'])}, and only `film` has a column")
        self.src(fid, None, header, "project", self.project_id)
        self.con.execute("UPDATE project SET film_words = ? WHERE id = ?", (doc["text"]["film"], self.project_id))
        mine = self.footage_facts(doc)
        if not first:
            before, was = self.footage_plan
            other = sorted(k for k in set(mine) | set(was) if mine.get(k) != was.get(k))
            if other:
                self.notes.append(f"{self.name}: data/{self.folder}/plan.json says {len(other)} footage fact(s) differently from "
                                  f"data/{before}/plan.json; the later word stands, and the earlier is kept as {before}'s: {', '.join(other)}")
        self.footage_plan = (self.folder, mine)
        aliases = dict(doc.get("aliases") or {})
        by = aliases.pop("_by", None)
        for n, (alias, label) in enumerate(aliases.items()):
            self.ins("alias", project_id=self.project_id, subject_id=self.subject(label), alias=alias, ord=n, said_by=by)
        prefixes = dict(doc.get("load_prefixes") or {})
        self.con.execute("UPDATE project SET load_prefixes_by = ? WHERE id = ?", (prefixes.pop("_by", None), self.project_id))
        for label, found in prefixes.items():
            for n, prefix in enumerate(found):
                self.ins("load_prefix", project_id=self.project_id, subject_id=self.subject(label), ord=n, prefix=prefix)
        for n, d in enumerate(doc.get("departures") or []):
            self.ins("departure", project_id=self.project_id, subject_id=self.subject(d["subject"]), ord=n, first=d["first"], last=d["last"],
                     note=d["note"], said_by=d["by"])
        for n, p in enumerate(doc.get("not_people") or []):
            nid = self.ins("not_person", project_id=self.project_id, video_id=self.video_id, ord=n, what=p["what"], counts_as_no_one=p.get("counts_as_no_one"), said_by=p["by"])
            for i, (a, b) in enumerate(p["shots"]):
                self.ins("not_person_span", not_person_id=nid, ord=i, first=a, last=b)
        if list(doc["subjects"]) != sorted(self.subjects, key=lambda s: self.subjects[s]):
            raise Refused(f"plan.json's subjects {list(doc['subjects'])} are not the shots file's, in its order")
        for label, p in doc["subjects"].items():
            sub = self.subject(label)
            recipe, ref = p["recipe"], p.get("reference") or {}
            said_twice = {k: (ref[k], recipe.get(k)) for k in ("crop", "open_mouth", "closed_mouth") if k in ref and ref[k] != recipe.get(k)}
            more = sorted(set(ref) - {"crop", "open_mouth", "closed_mouth", "why"})
            if said_twice or more:
                raise Refused(f"plan.json, {label}: `reference` says what `recipe` does not ({said_twice or more}); the mapping keeps `reference` raw "
                              "only while `recipe` says every form it does")
            if set(recipe) != {"margin", "zoom_up_to", "crop", "open_mouth", "closed_mouth", "by"}:
                raise Refused(f"plan.json, {label}: recipe holds {sorted(recipe)}")
            if set(p["depth"]) != {"on", "why"}:
                raise Refused(f"plan.json, {label}: depth holds {sorted(p['depth'])}")
            pid = self.ins("subject_plan", project_id=self.project_id, subject_id=sub, region=p["region"], said_by=p["by"], inputs_by=p.get("inputs_by"),
                           still_location_id=self.place(p["still"], "share_input"), margin_px=recipe["margin"], zoom_up_to=float(recipe["zoom_up_to"]),
                           crop=recipe["crop"], open_mouth=recipe["open_mouth"], closed_mouth=recipe["closed_mouth"], recipe_by=recipe["by"],
                           depth_on=p["depth"]["on"], depth_why=p["depth"]["why"], look_by=p.get("look_by"), restore_by=p.get("restore_by"),
                           keep_outs_by=p.get("keep_outs_by"))
            self.src(fid, None, p, "subject_plan", pid)
            for n, part in enumerate(p["parts"]):
                self.ins("plan_part", subject_plan_id=pid, ord=n, part=part)
            for n, name in enumerate(p.get("keep_outs") or []):
                self.ins("plan_keep_out", subject_plan_id=pid, ord=n, name=name)
            t = p["text"]
            known = {"who", "she", "her", "looks", "looks_by", "hair", "hair_words", "hair_words_by", "picture_hair", "picture_hair_by", "sings",
                     "not_used", "hair_change_removed_by"}
            if set(t) - known:
                raise Refused(f"plan.json, {label}: text holds {sorted(set(t) - known)}, with no column")
            self.ins("subject_text", subject_plan_id=pid, **{k: t.get(k) for k in sorted(known - {"not_used", "hair_change_removed_by"})})
            removed = {what: p[key] for key, what in REMOVALS.items() if key in p}
            if "hair_change_removed_by" in t:
                removed["text.hair_change"] = t["hair_change_removed_by"]
            look = p.get("look") or {}
            if set(look) - {"change", "stay_removed_by"}:
                raise Refused(f"plan.json, {label}: look holds {sorted(look)}")
            if "stay_removed_by" in look:
                removed["look.stay"] = look["stay_removed_by"]
            for what, note in removed.items():
                self.ins("plan_removal", subject_plan_id=pid, what=what, note=note)
            for n, c in enumerate(look.get("change") or []):
                self.ins("look_change", subject_plan_id=pid, ord=n, room=c["room"], said_by=c["by"])
            for n, o in enumerate(p.get("shot_overrides") or []):
                self.ins("shot_override", subject_plan_id=pid, ord=n, first=o["first"], last=o["last"], margin_px=o.get("margin"),
                         zoom_up_to=o.get("zoom_up_to"), open_mouth=o.get("open_mouth"), closed_mouth=o.get("closed_mouth"), why=o["why"], said_by=o["by"])
            for n, c in enumerate(p.get("clothing") or []):
                self.ins("clothing_words", subject_plan_id=pid, ord=n, first=c["first"], last=c["last"], words=c["words"], said_by=c["by"])
            for n, r in enumerate(p.get("restore") or []):
                if set(r) - {"classes", "first", "last", "touching", "exception", "exception_by", "loads"}:
                    raise Refused(f"plan.json, {label}: a restore holds {sorted(r)}")
                rid = self.ins("restore_rule", subject_plan_id=pid, ord=n, first=r["first"], last=r["last"], touching=r.get("touching"),
                               exception=r.get("exception"), exception_by=r.get("exception_by"))
                for i, c in enumerate(r["classes"]):
                    self.ins("restore_class", restore_rule_id=rid, ord=i, part_class=c)
                for i, name in enumerate(r.get("loads") or []):
                    self.ins("restore_load", restore_rule_id=rid, ord=i, load_name=name)
            # footage facts, as this project's plan says them
            for n, m in enumerate(p.get("head_marks") or []):
                self.ins("head_mark", project_id=self.project_id, subject_id=sub, ord=n, first=m["first"], last=m["last"], mark=m["mark"], said_by=m["by"])
            for n, v in enumerate(p.get("voice") or []):
                self.ins("voice_span", project_id=self.project_id, subject_id=sub, ord=n, first=v["first"], last=v["last"], voice=v["voice"], said_by=v["by"])
            if "source_facial_hair" in p:
                self.ins("facial_hair", project_id=self.project_id, subject_id=sub, has=p["source_facial_hair"], said_by=p["source_facial_hair_by"])

    # ---- what the owner said -------------------------------------------------------------------------------------
    def about(self, r: dict) -> str:
        """What an answer is about, by rules a reader can check; `not classed` where none settles it."""
        if r["who"] != "owner" or r["what"] == "removed":
            return "not an answer"
        if r["what"] == "parts":
            return "plan"
        if "table" in r or "sheet" in r or str(r.get("row", "")).startswith("gap_"):
            return "footage"
        if "clip" in r or "render" in r or "clip_not_known" in r:
            return "render"
        return "not classed"                              # answers_on_loads classes more of these once the laid lists are in

    def answers(self) -> None:
        rel = f"{self.folder}/owner_answers.jsonl"
        if self.folder != self.projects[0]["folder"] and not (self.data / rel).is_file():
            return
        text, fid = self.read(rel, "answers")
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "answers", [r for _, r in recs], self.labels)
        not_spans: list[int] = []
        names = {self.folder, *self.con.execute("SELECT name, source_name FROM video WHERE id = ?", (self.video_id,)).fetchone()}
        for n, r in recs:
            if r.get("video", self.folder) not in names:
                raise Refused(f"owner_answers.jsonl:{n}: video {r['video']!r}, not one of {sorted(names)}")
            about = self.about(r)
            person, frame = r.get("person"), r.get("source_frame")
            aid = self.answer_ids[(f"data/{rel}", n)] = self.ins(
                "answer", file_id=fid, video_id=self.video_id, project_id=self.project_at(r["at"]) if about in ("render", "plan") else None, about=about,
                line=n, at=r["at"], who=r["who"], what=r["what"], page_row=r.get("row"), tile=r.get("n"), shot_said=r.get("shot"),
                subject_id=self.subject(r["subject"]) if r.get("subject") else None, frame=r.get("frame"),
                source_frame=frame if isinstance(frame, int) else None, source_frame_default=frame == "default",
                chosen=r.get("chosen"), words=r.get("words"),
                person_tile=person if isinstance(person, int) else None, person_label=person if isinstance(person, str) else None,
                table_name=r.get("table"), table_frame=r.get("table_frame"),
                clip_location_id=self.place(r["clip"], "share_output") if "clip" in r else None,
                clip_artifact_id=self.artifact(r["clip_sha256"]) if r.get("clip_sha256") else None, clip_not_known=r.get("clip_not_known"),
                render_location_id=self.place(r["render"], "share_output") if "render" in r else None,
                render_artifact_id=self.artifact(r["render_sha256"]) if r.get("render_sha256") else None,
                said_in=r.get("said_in"), taken_down_by=r.get("taken_down_by"), relayed_by=r.get("relayed_by"), relayed_from=r.get("relayed_from"),
                corrects=r.get("corrects"))
            if isinstance(frame, str) and frame != "default":
                raise Refused(f"owner_answers.jsonl:{n}: source_frame {frame!r}")
            for role, key in (("about", "frames"), ("bad", "bad_frames")):
                if r.get(key) and any(isinstance(v, str) for v in r[key]):
                    not_spans.append(n)                   # the page wrote a text and a joined number, e.g. ["415", 41522]
                    continue
                for i, (a, b) in enumerate(spans(r[key]) if r.get(key) else []):
                    self.ins("answer_frame", answer_id=aid, role=role, ord=i, first=a, last=b)
            for role in ("replace", "keep_out", "leave"):
                for i, c in enumerate(r.get(role) or []):
                    self.ins("answer_part", answer_id=aid, role=role, ord=i, part_class=c)
            self.src(fid, n, r, "answer", aid)
        if not_spans:
            self.notes.append(f"data/{rel}: {len(not_spans)} answer line(s) hold a `frames` that is not a span (a text and a joined number, as the page wrote "
                              f"it): kept raw only, on lines {not_spans[0]}-{not_spans[-1]}")

    def answers_on_loads(self) -> None:
        """An owner's answer no other rule classed, whose row is a load's name, is about that load's render."""
        rows = self.con.execute("""SELECT a.id, a.at FROM answer a WHERE a.video_id = ? AND a.about = 'not classed'
                                   AND a.page_row IN (SELECT d.name FROM load d JOIN project p ON p.id = d.project_id WHERE p.video_id = ?)""",
                                (self.video_id, self.video_id)).fetchall()
        for row in rows:
            self.con.execute("UPDATE answer SET about = 'render', project_id = ? WHERE id = ?", (self.project_at(row["at"]), row["id"]))
        for file, in self.con.execute("""SELECT DISTINCT o.relpath FROM answer a JOIN import_file i ON i.id = a.file_id JOIN location o ON o.id = i.location_id
                                         WHERE a.video_id = ? ORDER BY 1""", (self.video_id,)).fetchall():
            counts = self.con.execute("""SELECT a.about, count(*) n FROM answer a JOIN import_file i ON i.id = a.file_id JOIN location o ON o.id = i.location_id
                                         WHERE o.relpath = ? GROUP BY 1 ORDER BY 1""", (file,)).fetchall()
            self.notes.append(f"data/{file}: answers by what they are about: " + ", ".join(f"{c['about']} {c['n']}" for c in counts))

    # ---- the laid list ---------------------------------------------------------------------------------------------
    def load(self, name: str, at: str) -> int:
        pid = self.project_at(at)
        row = self.con.execute("SELECT id FROM load WHERE project_id = ? AND name = ?", (pid, name)).fetchone()
        return row["id"] if row else self.ins("load", project_id=pid, name=name)

    def word(self, cited: str | None) -> tuple[str | None, int | None, int | None]:
        """A cited line (file:line) as its file, its line, and the answer row where the file is the answers file."""
        if cited is None:
            return None, None, None
        m = WORD.match(cited)
        if not m:
            raise Refused(f"a cited line {cited!r} is not file:line")
        file, line = m["file"], int(m["line"])
        answers = file.endswith("/owner_answers.jsonl")
        if answers and (file, line) not in self.answer_ids:
            raise Refused(f"{cited} is cited, and no answers file in the record has that line")
        return file, line, self.answer_ids[(file, line)] if answers else None

    def laid_row(self, r: dict, where: str, fid: int, list_line: int | None, removed_line: int | None) -> int:
        if r["schema"] != "h3_laid/1":
            raise Refused(f"{where}: schema {r['schema']!r}")
        lid = self.ins(
            "laid", project_id=self.project_at(r["added"]), subject_id=self.subject(r["subject"]), load_id=self.load(r["load"], r["added"]),
            first=r["first"], last=r["last"], render_location_id=self.place(r["file"], "share_output"), render_artifact_id=self.artifact(r["sha256"]),
            zoomed_artifact_id=self.artifact(r["zoomed_sha256"]) if r.get("zoomed_sha256") else None, added_at=r["added"], added_by=r["added_by"],
            no_rebuild_why=r.get("no_rebuild"), partial_lay_why=r.get("partial_lay_why"), seeded_from=r.get("seeded_from"),
            owner_before_correction=r.get("owner_was"), owner_corrected_why=r.get("owner_corrected"), file_id=fid, list_line=list_line,
            removed_line=removed_line)
        history = r.get("history") or []
        states = []
        if history:
            states.append({"owner": history[0]["owner_was"], "why": history[0]["by_was"], "word": None, "at": r["added"], "by": None, "no": None})
            for i, h in enumerate(history):
                if set(h) != {"at", "by_session", "owner_was", "by_was", "owner", "by", "owner_word", "no_rebuild"}:
                    raise Refused(f"{where}: a history entry holds {sorted(h)}")
                if (h["owner_was"], h["by_was"]) != (states[-1]["owner"], states[-1]["why"]):
                    raise Refused(f"{where}: history entry {i} starts from a state the one before it did not leave")
                states.append({"owner": h["owner"], "why": h["by"], "word": h["owner_word"], "at": h["at"], "by": h["by_session"], "no": h["no_rebuild"]})
            if (states[-1]["owner"], states[-1]["why"], states[-1]["word"]) != (r["owner"], r["by"], r.get("owner_word")):
                raise Refused(f"{where}: the line's owner, by and owner_word are not what its last history entry left")
        else:
            states.append({"owner": r["owner"], "why": r["by"], "word": r.get("owner_word"), "at": r["added"], "by": None, "no": None})
        for i, s in enumerate(states):
            file, line, answer = self.word(s["word"])
            self.ins("laid_state", laid_id=lid, ord=i, owner_status=s["owner"], why=s["why"], word_file=file, word_line=line, answer_id=answer,
                     at=s["at"], by_session=s["by"], no_rebuild_why=s["no"])
        for i, (a, b) in enumerate(spans(r["frames"])):
            self.ins("laid_frames", laid_id=lid, ord=i, first=a, last=b)
        replaces = r.get("replaces")
        if isinstance(replaces, str):
            replaces = [p.removeprefix("and ") for p in re.split(r"(?<=\)), ", replaces)]
        for i, part in enumerate(replaces or []):
            m = LAID_SENTENCE.match(part)
            if not m:
                raise Refused(f"{where}: `replaces` holds {part!r}, which is not a load's name or a name with (sha256 ..., laid on ...)")
            self.ins("laid_replaces", laid_id=lid, ord=i, load_name=m["name"], sha256_start=m["sha"])
            for n, (a, b) in enumerate(json.loads(m["on"]) if m["on"] else []):
                self.ins("laid_replaces_frames", laid_id=lid, replaces_ord=i, ord=n, first=a, last=b)
        for i, words in enumerate(r.get("owner_looks") or []):
            self.ins("laid_look", laid_id=lid, ord=i, words=words)
        for i, c in enumerate(r.get("carved_by") or []):
            self.ins("laid_carve", laid_id=lid, ord=i, load_name=c["load"], reason=c["reason"], at=c["at"])
            for n, (a, b) in enumerate(c["frames"]):
                self.ins("laid_carve_frames", laid_id=lid, carve_ord=i, ord=n, first=a, last=b)
        return lid

    def keeps(self) -> None:
        keeps = self.entry.get("keeps")
        if not keeps:
            return
        earlier = next((p for p in self.projects if p["name"] == keeps["from"] and p["id"] < self.project_id), None)
        if earlier is None:
            raise Refused(f"data/projects.json: {self.name} keeps pieces from {keeps['from']!r}, which is not an earlier project on its video")
        for label in keeps["subjects"]:
            self.ins("project_keeps", project_id=self.project_id, from_project_id=earlier["id"], subject_id=self.subject(label), said_by=keeps["by"])

    def not_its_own(self, r: dict) -> str | None:
        """A line of this project's laid list dated before the project began: `kept` where it is a standing piece the
        project keeps from the earlier one, as it stands there; else `not kept`. None for a line of its own time."""
        if self.project_at(r["added"]) >= self.project_id:
            return None
        same = self.con.execute(
            """SELECT l.id FROM project_laid l JOIN load d ON d.id = l.load_id JOIN artifact a ON a.id = l.render_artifact_id
               JOIN subject s ON s.id = l.subject_id
               WHERE l.for_project_id = ? AND l.kept_from_another = 1 AND d.name = ? AND a.sha256 = ? AND s.label = ?""",
            (self.project_id, r["load"], r["sha256"], r["subject"])).fetchall()
        frames = lambda lid: [[f["first"], f["last"]] for f in self.con.execute("SELECT first, last FROM laid_frames WHERE laid_id = ? ORDER BY ord", (lid,))]
        return "kept" if any(frames(row["id"]) == [list(x) for x in spans(r["frames"])] for row in same) else "not kept"

    def laid(self) -> None:
        own = self.folder != self.projects[0]["folder"]
        rel = f"{self.folder}/laid.jsonl"
        if own and not (self.data / rel).is_file():
            self.notes.append(f"{self.name}: no laid.jsonl in data/{self.folder}")
            return
        text, fid = self.read(rel, "laid")
        self.laid_sha, self.laid_lines = hashlib.sha256(text.encode()).hexdigest(), len(text.splitlines())
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "laid", [r for _, r in recs], self.labels)
        seen = {"kept": [], "not kept": []}
        for n, r in recs:
            other = self.not_its_own(r) if own else None
            if other:
                seen[other].append(r)
                self.src(fid, n, r)                      # verbatim, and no row: the piece is the earlier project's
                continue
            self.src(fid, n, r, "laid", self.laid_row(r, f"{rel}:{n}", fid, n, None))
        if own and (seen["kept"] or seen["not kept"]):
            kept = self.con.execute("SELECT count(*) FROM project_laid WHERE for_project_id = ? AND kept_from_another = 1", (self.project_id,)).fetchone()[0]
            by_subject = lambda rows: ", ".join(f"{label} {sum(1 for r in rows if r['subject'] == label)}" for label in sorted({r["subject"] for r in rows}))
            self.notes.append(f"{self.name}: data/{rel} holds {len(seen['kept'])} line(s) that are pieces it keeps as they stand (of the {kept} it keeps), "
                              f"and {len(seen['not kept'])} dated before it began that are not ({by_subject(seen['not kept']) or 'none'}): "
                              "none of these is imported as its own")
        rel = f"{self.folder}/laid.removed.jsonl"
        if own and not (self.data / rel).is_file():
            return
        text, fid = self.read(rel, "laid-removed")
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "laid-removed", [r for _, r in recs], self.labels)
        for n, r in recs:
            if r["schema"] != "h3_laid_removed/1":
                raise Refused(f"{rel}:{n}: schema {r['schema']!r}")
            lid = self.laid_row(r["removed"], f"{rel}:{n}", fid, None, n)
            self.ins("laid_removal", laid_id=lid, at=r["at"], by_session=r["by_session"], why=r["by"], no_rebuild_why=r.get("no_rebuild"),
                     carved_whole_by=r.get("carved_whole_by"))
            self.src(fid, n, r, "laid", lid)

    # ---- the cut that closed the project ---------------------------------------------------------------------------
    def final_cut(self) -> None:
        final = self.entry.get("final_cut")
        if not final:
            return
        stamp = final["stamp"]
        found = sorted((self.data / self.folder).glob(f"*_delivery_tables/rebuild_{self.folder}_{stamp}.json"))
        if len(found) != 1:
            raise Refused(f"{len(found)} rebuild record(s) of {self.folder} are stamped {stamp}, where one is meant")
        rel = str(found[0].relative_to(self.data))
        text, fid = self.read(rel, "cut-record")
        rec = json.loads(text)
        cover.hold(self.map, "cut-record", [rec], self.labels)
        if rec["video"] != self.folder or rec["stamp"] != stamp:
            raise Refused(f"data/{rel}: video {rec['video']!r}, stamp {rec['stamp']!r}")
        table_rel = f"{found[0].parent.relative_to(self.data)}/{Path(rec['artifacts']['table_json']).name}"
        text, tfid = self.read(table_rel, "cut-table")
        table = json.loads(text)
        cover.hold(self.map, "cut-table", [table], self.labels)
        made = MADE.search(table["made"])
        if not made:
            raise Refused(f"data/{table_rel}: its `made` sentence names no commit and laid list")
        if not self.laid_sha.startswith(made["sha"]) or int(made["lines"]) != self.laid_lines:
            raise Refused(f"the cut {stamp} was built from laid list {made['sha']} ({made['lines']} lines), and data/{self.folder}/laid.jsonl is "
                          f"{self.laid_sha[:16]} ({self.laid_lines} lines): its pieces cannot be tied to the list as it stands")
        clip = self.place(rec["artifacts"]["clip"], "share_output")
        sidecar_path = root_path("share_output") / (rec["artifacts"]["clip"] + ".json")
        if not sidecar_path.is_file():
            raise Refused(f"the cut's sidecar is not at {rec['artifacts']['clip']}.json under the output root")
        raw = sidecar_path.read_bytes()
        side = json.loads(raw)
        cover.hold(self.map, "cut-sidecar", [side], self.labels)
        sfid = self.ins("import_file", location_id=self.location("share_output", rec["artifacts"]["clip"] + ".json"),
                        artifact_id=self.artifact(hashlib.sha256(raw).hexdigest(), len(raw)), source="cut-sidecar")
        if self.place(side["video"], "share_output") != clip:
            raise Refused(f"the sidecar is of {side['video']}, not of the cut")
        args = shlex.split(side["command"])
        source = Path(args[args.index("--source") + 1]).name
        if source not in self.copies:
            raise Refused(f"the cut was assembled from {source!r}, which the video's manifest does not name as a copy")
        cut = self.ins("cut", project_id=self.project_id, stamp=stamp, location_id=clip, artifact_id=self.artifact(side["video_sha256"], side["video_size"]),
                       first=rec["span"][0], last=rec["span"][1], made_at=side["made_at"], laid_list_sha256_start=made["sha"],
                       laid_list_lines=int(made["lines"]), table_made_at_commit=made["commit"], source_copy_id=self.copies[source],
                       is_final=1, final_said_by=final["by"])
        self.src(fid, None, rec, "cut", cut)
        self.src(sfid, None, side, "cut", cut)
        self.src(tfid, None, {k: v for k, v in table.items() if k != "rows"}, "cut", cut)
        for n, row in enumerate(table["rows"]):
            if set(row) - {"first", "last", "file", "name", "piece_first", "subject", "region", "restore", "run"}:
                raise Refused(f"data/{table_rel}: a row holds {sorted(row)}")
            laid = self.con.execute(
                """SELECT l.id FROM project_laid l JOIN load d ON d.id = l.load_id JOIN location o ON o.id = l.render_location_id
                   JOIN subject s ON s.id = l.subject_id JOIN laid_frames f ON f.laid_id = l.id
                   WHERE l.for_project_id = ? AND d.name = ? AND o.relpath = ? AND s.label = ? AND f.first <= ? AND f.last >= ?""",
                (self.project_id, row["name"], row["file"], row["subject"], row["first"], row["last"])).fetchall()
            if len(laid) != 1:
                raise Refused(f"data/{table_rel}: row {n} ({row['name']} on {row['first']}-{row['last']}) is {len(laid)} standing laid piece(s), where one is meant")
            capture = run = None
            if row.get("run"):
                capture, run = row["run"].rsplit(":", 1)
                if not capture.startswith("data/"):
                    raise Refused(f"data/{table_rel}: row {n} is tied to {capture!r}, which is not under data/")
                capture = capture[len("data/"):]
            pid = self.ins("cut_piece", cut_id=cut, ord=n, laid_id=laid[0]["id"], first=row["first"], last=row["last"], piece_first=row["piece_first"],
                           region=row["region"], capture_folder=capture, run_name=run)
            for i, token in enumerate(row["restore"]):
                self.ins("cut_piece_restore", cut_piece_id=pid, ord=i, token=token)
            self.src(tfid, n, row, "cut_piece", pid)

    # ---- the build -----------------------------------------------------------------------------------------------
    def run(self) -> list[str]:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip() or None
        with self.con:
            for projects in self.projects_file():
                self.projects, self.video_id = projects, projects[0]["video_id"]
                self.labels, self.subjects = set(), {}
                self.enter(projects[0])                   # the footage's records are the first project's
                self.manifest()
                self.shots()
                self.cast()
                for p in projects:
                    self.enter(p)
                    self.answers()
                for p in projects:
                    self.enter(p)
                    self.shot_orders()
                    self.plan()
                    self.keeps()
                    self.laid()
                    self.final_cut()
                self.answers_on_loads()
                for row in self.con.execute("""SELECT f.name folder, p.name, count(*) n FROM laid l JOIN project p ON p.id = l.project_id
                                               JOIN import_file i ON i.id = l.file_id JOIN location o ON o.id = i.location_id
                                               JOIN project f ON o.relpath LIKE f.folder || '/%' WHERE f.id <> p.id AND p.video_id = ? GROUP BY 1, 2""",
                                            (self.video_id,)).fetchall():
                    self.notes.append(f"{row['n']} laid line(s) in {row['folder']}'s lists are dated in {row['name']}'s time and are {row['name']}'s")
            self.ins("import_run", at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), tool_commit=commit, project="every project")
        bad = self.con.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise Refused(f"{len(bad)} row(s) point at a row that is not there: {[tuple(b) for b in bad[:5]]}")
        return self.notes
