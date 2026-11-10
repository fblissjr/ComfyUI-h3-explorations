"""Build the lane's database from a project's record files, the first tier: what a person said and decided, the laid
list, and the cut that closed the project.

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
    def __init__(self, con: sqlite3.Connection, data: Path, project: str, mapping: cover.Mapping | None = None):
        self.con, self.data, self.name = con, Path(data), project
        self.map = mapping or cover.Mapping()
        self.notes: list[str] = []
        self.labels: set[str] = set()
        self.subjects: dict[str, int] = {}
        self.files: dict[str, int] = {}

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

    def project_at(self, at: str | None) -> int:
        """Whose a dated record is: the project of this video whose time it falls in. An undated one is this project's."""
        if at is None:
            return self.project_id
        t = at.replace("Z", "")
        for p in self.projects:
            if (p["starts_at"] is None or p["starts_at"].replace("Z", "") <= t) and (p["ends_before"] is None or t < p["ends_before"].replace("Z", "")):
                return p["id"]
        raise Refused(f"no project of the video holds the time {at}")

    # ---- the projects file ---------------------------------------------------------------------------------------
    def projects_file(self) -> None:
        text, fid = self.read("projects.json", "projects")
        doc = json.loads(text)
        if doc.get("schema") != "h3_projects/1":
            raise Refused(f"data/projects.json is {doc.get('schema')!r}, not h3_projects/1")
        cover.hold(self.map, "projects", [{k: v for k, v in doc.items() if k != "projects"}] + doc["projects"], set())
        mine = next((p for p in doc["projects"] if p["name"] == self.name), None)
        if mine is None:
            raise Refused(f"data/projects.json names no project {self.name!r}")
        self.entry, self.folder = mine, mine["folder"]
        shots = json.loads((self.data / mine["footage_records"] / "shots.json").read_text())
        self.video_id = self.ins("video", name=mine["video"], source_name=shots["video"], frames=shots["source_frames"])
        self.projects = []
        for p in doc["projects"]:
            if p["video"] != mine["video"]:
                continue
            pid = self.ins("project", name=p["name"], video_id=self.video_id, folder=p["folder"], starts_at=p["starts_at"],
                           ends_before=p["ends_before"], said_by=p["by"])
            self.projects.append(dict(p, id=pid))
            self.src(fid, None, p, "project", pid)
            if p["name"] == self.name:
                self.project_id = pid

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

        orders = []
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
            if "order" in rec:
                orders.append((sid, rec))
        self.orders = orders

    def shot_orders(self) -> None:
        for sid, rec in self.orders:
            for n, label in enumerate(rec["order"]):
                self.ins("shot_order", project_id=self.project_id, shot_id=sid, subject_id=self.subject(label), ord=n, said_by=rec["order_by"])

    def cast(self) -> None:
        text, fid = self.read(f"{self.folder}/cast/cast.jsonl", "cast")
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "cast", [r for _, r in recs], self.labels)
        for n, r in recs:
            if r["schema"] != "h3_cast/1":
                raise Refused(f"cast.jsonl:{n}: schema {r['schema']!r}")
            box, size = r.get("box") or [None] * 4, r.get("size") or [None] * 2
            sid = self.ins("seed", subject_id=self.subject(r["subject"]), line=n, frame=r["frame"], x=box[0], y=box[1], w=box[2], h=box[3],
                           picture_w=size[0], picture_h=size[1], answer=r["answer"], encloses=r.get("encloses"), drawn=r.get("drawn"),
                           from_table=r["table"], said_by=r["by"], at=r["at"])
            self.src(fid, n, r, "seed", sid)

    # ---- the plan --------------------------------------------------------------------------------------------------
    def plan(self) -> None:
        text, fid = self.read(f"{self.folder}/plan.json", "plan")
        doc = json.loads(text)
        header = {k: v for k, v in doc.items() if k != "subjects"}
        cover.hold(self.map, "plan-header", [header], self.labels)
        cover.hold(self.map, "plan", list(doc["subjects"].values()), self.labels)
        video = self.con.execute("SELECT source_name FROM video WHERE id = ?", (self.video_id,)).fetchone()[0]
        if doc["schema"] != "h3_plan/1" or doc["video"] != video:
            raise Refused(f"plan.json: schema {doc['schema']!r}, video {doc['video']!r} (the shots file says {video!r})")
        self.con.execute("UPDATE video SET film_words = ? WHERE id = ?", (doc["text"]["film"], self.video_id))
        if set(doc["text"]) != {"film"}:
            raise Refused(f"plan.json: text holds {sorted(doc['text'])}, and only `film` has a column")
        self.src(fid, None, header, "project", self.project_id)
        aliases = dict(doc.get("aliases") or {})
        by = aliases.pop("_by", None)
        for n, (alias, label) in enumerate(aliases.items()):
            self.ins("alias", subject_id=self.subject(label), alias=alias, ord=n, said_by=by)
        prefixes = dict(doc.get("load_prefixes") or {})
        self.con.execute("UPDATE project SET load_prefixes_by = ? WHERE id = ?", (prefixes.pop("_by", None), self.project_id))
        for label, found in prefixes.items():
            for n, prefix in enumerate(found):
                self.ins("load_prefix", project_id=self.project_id, subject_id=self.subject(label), ord=n, prefix=prefix)
        for n, d in enumerate(doc.get("departures") or []):
            self.ins("departure", project_id=self.project_id, subject_id=self.subject(d["subject"]), ord=n, first=d["first"], last=d["last"],
                     note=d["note"], said_by=d["by"])
        for n, p in enumerate(doc.get("not_people") or []):
            nid = self.ins("not_person", video_id=self.video_id, ord=n, what=p["what"], counts_as_no_one=p.get("counts_as_no_one"), said_by=p["by"])
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
            # footage facts the plan file carries
            for n, m in enumerate(p.get("head_marks") or []):
                self.ins("head_mark", subject_id=sub, ord=n, first=m["first"], last=m["last"], mark=m["mark"], said_by=m["by"])
            for n, v in enumerate(p.get("voice") or []):
                self.ins("voice_span", subject_id=sub, ord=n, first=v["first"], last=v["last"], voice=v["voice"], said_by=v["by"])
            if "source_facial_hair" in p:
                self.ins("facial_hair", subject_id=sub, has=p["source_facial_hair"], said_by=p["source_facial_hair_by"])

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
        return "not classed"

    def answers(self) -> None:
        text, fid = self.read(f"{self.folder}/owner_answers.jsonl", "answers")
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "answers", [r for _, r in recs], self.labels)
        self.answer_ids: dict[int, int] = {}
        classed: dict[str, int] = {}
        not_spans: list[int] = []
        names = {self.folder, self.con.execute("SELECT source_name FROM video WHERE id = ?", (self.video_id,)).fetchone()[0]}
        for n, r in recs:
            if r.get("video", self.folder) not in names:
                raise Refused(f"owner_answers.jsonl:{n}: video {r['video']!r}, not one of {sorted(names)}")
            about = self.about(r)
            classed[about] = classed.get(about, 0) + 1
            person, frame = r.get("person"), r.get("source_frame")
            aid = self.answer_ids[n] = self.ins(
                "answer", video_id=self.video_id, project_id=self.project_at(r["at"]) if about in ("render", "plan") else None, about=about,
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
            self.notes.append(f"{len(not_spans)} answer line(s) hold a `frames` that is not a span (a text and a joined number, as the page wrote "
                              f"it): kept raw only, on lines {not_spans[0]}-{not_spans[-1]}")
        self.notes.append("answers by what they are about: " + ", ".join(f"{k} {v}" for k, v in sorted(classed.items())))

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
        answers = file == f"data/{self.folder}/owner_answers.jsonl"
        if answers and line not in self.answer_ids:
            raise Refused(f"{cited} is cited, and the answers file has no such line")
        return file, line, self.answer_ids[line] if answers else None

    def laid_row(self, r: dict, where: str, list_line: int | None, removed_line: int | None) -> int:
        if r["schema"] != "h3_laid/1":
            raise Refused(f"{where}: schema {r['schema']!r}")
        lid = self.ins(
            "laid", project_id=self.project_at(r["added"]), subject_id=self.subject(r["subject"]), load_id=self.load(r["load"], r["added"]),
            first=r["first"], last=r["last"], render_location_id=self.place(r["file"], "share_output"), render_artifact_id=self.artifact(r["sha256"]),
            zoomed_artifact_id=self.artifact(r["zoomed_sha256"]) if r.get("zoomed_sha256") else None, added_at=r["added"], added_by=r["added_by"],
            no_rebuild_why=r.get("no_rebuild"), partial_lay_why=r.get("partial_lay_why"), seeded_from=r.get("seeded_from"),
            owner_before_correction=r.get("owner_was"), owner_corrected_why=r.get("owner_corrected"), list_line=list_line, removed_line=removed_line)
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

    def laid(self) -> None:
        text, fid = self.read(f"{self.folder}/laid.jsonl", "laid")
        self.laid_sha, self.laid_lines = hashlib.sha256(text.encode()).hexdigest(), len(text.splitlines())
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "laid", [r for _, r in recs], self.labels)
        for n, r in recs:
            self.src(fid, n, r, "laid", self.laid_row(r, f"laid.jsonl:{n}", n, None))
        text, fid = self.read(f"{self.folder}/laid.removed.jsonl", "laid-removed")
        recs = [(n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        cover.hold(self.map, "laid-removed", [r for _, r in recs], self.labels)
        for n, r in recs:
            if r["schema"] != "h3_laid_removed/1":
                raise Refused(f"laid.removed.jsonl:{n}: schema {r['schema']!r}")
            lid = self.laid_row(r["removed"], f"laid.removed.jsonl:{n}", None, n)
            self.ins("laid_removal", laid_id=lid, at=r["at"], by_session=r["by_session"], why=r["by"], no_rebuild_why=r.get("no_rebuild"),
                     carved_whole_by=r.get("carved_whole_by"))
            self.src(fid, n, r, "laid", lid)
        twice = self.con.execute("""SELECT o.relpath, count(DISTINCT l.render_artifact_id) n FROM laid l JOIN location o ON o.id = l.render_location_id
                                    GROUP BY o.id HAVING n > 1""").fetchall()
        if twice:
            self.notes.append(f"{len(twice)} render place(s) are named by laid lines with more than one sha256 (written over in place): "
                              + ", ".join(t["relpath"] for t in twice[:5]))

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
                """SELECT l.id FROM laid_standing l JOIN load d ON d.id = l.load_id JOIN location o ON o.id = l.render_location_id
                   JOIN subject s ON s.id = l.subject_id JOIN laid_frames f ON f.laid_id = l.id
                   WHERE d.name = ? AND o.relpath = ? AND s.label = ? AND f.first <= ? AND f.last >= ?""",
                (row["name"], row["file"], row["subject"], row["first"], row["last"])).fetchall()
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
            self.projects_file()
            self.manifest()
            self.shots()
            self.shot_orders()
            self.cast()
            self.plan()
            self.answers()
            self.laid()
            self.final_cut()
            self.ins("import_run", at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), tool_commit=commit, project=self.name)
        bad = self.con.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise Refused(f"{len(bad)} row(s) point at a row that is not there: {[tuple(b) for b in bad[:5]]}")
        late = self.con.execute("""SELECT p.name, count(*) n FROM laid l JOIN project p ON p.id = l.project_id WHERE p.name <> ? GROUP BY p.name""", (self.name,)).fetchall()
        for row in late:
            self.notes.append(f"{row['n']} laid line(s) in {self.folder}'s list are dated in {row['name']}'s time and are {row['name']}'s")
        return self.notes
