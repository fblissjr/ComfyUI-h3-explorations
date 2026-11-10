#!/usr/bin/env python3
"""`bench/lane_db/` (the lane's record in SQLite) on a small made-up project: the schema's rules, the importer's refusals,
and that what goes in comes back.

  the_schema_keeps_its_rules      every table is STRICT; a connection has foreign keys on; the migrations are numbered
                                  with none missing; a row that points at no row is refused
  json_only_where_it_is_said      after a build, no text of any table but `import_source.raw` reads as a JSON object or
                                  list; no file of bench/lane_db/ calls a JSON function of SQLite
  every_mapping_line_is_used      the made-up project has a record for every line of mapping.tsv it could reach (so the
                                  cases below drive the mapping, not a corner of it); the cut's lines are named as not
                                  driven here
  an_unmapped_field_stops_it      a record with a field no mapping line covers is refused by name, and the database
                                  holds no row of that build
  a_build_is_a_function           two builds of the same files hold the same rows (db.dump, the build's time left out)
  files_come_back                 each record file equals what the database gives back, raw-only fields left out; one
                                  value changed in the database is named by the comparison
  a_place_is_written_once         a location given other bytes after its first is refused by the trigger
  history_is_states               a laid line's `history` becomes states in order, and a history that does not start
                                  from the state before it is refused
  a_dated_record_is_its_projects  a laid line dated in the second project's time lands in the second project, and is in
                                  neither the first's standing list nor its file given back
  a_project_keeps_anothers_pieces the second project lays the first's standing pieces of the subject it keeps, then its
                                  own lines; a line of its file dated before it began is imported as nobody's, and the
                                  note counts the kept and the not kept; its plan is its own and its footage facts are
                                  the video's; `compare` names the piece its file holds and the record does not, and
                                  exits 1
  a_footage_fact_said_twice       each plan's footage facts are kept as that plan's word; where a second project's plan
                                  says one differently a note names the field, the later word stands (`*_standing`),
                                  the earlier stays as the first project's, and each plan file still comes back as it
                                  is; a shots file that is not the footage's is refused
  an_old_file_is_brought_up       a database at the first schema is brought to the newest with every link whole, its
                                  footage facts under the video's first project
  subjects_start_a_project        `export --subjects` writes only those subjects' standing pieces
  the_lanes_file_waits            `build` refuses data/lane.sqlite while a mapping line is `ask`, and any other path
                                  inside the repo

    <comfy venv python> bench/check_lane_db.py

No card, no share, no file of the owner's: a temporary data folder. It does not run `prove` (that reads a real project
and the share), and so does not check the delivery table or the final cut's import.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, refused  # noqa: E402
from lane_db import cover, db, export, importer  # noqa: E402
from lane_db.cli import main  # noqa: E402

SHA = lambda n: f"{n:064x}"


def project() -> dict[str, object]:
    """A made-up project `walk` on a made-up video, and a second project `walk2` on the same footage. relative path -> data."""
    who = lambda present, by="a person, on the who page": {"present": present, "how": "frame 3: here", "frames_on": [3], "frames_boxed": [], "by": by}
    old = {"first": 0, "last": 19, "who": {"a": who("yes"), "b": who("no")}, "scene_score": 0.5}
    shots = {"schema": "h3_shots/1", "video": "walk.mp4", "source_frames": 40, "copy": "walk_24p.mov", "span": [0, 39], "subjects": ["a", "b"],
             "written_by": "a session", "cuts_from": {"0-39": "a shots file"},
             "shots": [
                 {"first": 0, "last": 9, "scene_score": 0.4, "note": "a note", "split_from": old,
                  "who": {"a": dict(who("yes"), frames_from="a tool", frames_boxed=[4],
                                    seen_looks=[{"first": 2, "last": 3, "part": "face_and_neck", "is": "not seen", "why": "turned away", "by": "a look",
                                                 "entered_by": "a session", "at": "2026-01-02T00:00:00Z"}]), "b": who("no")},
                  "answered_over": [{"at": "2026-01-01T00:00:00Z", "by": "a session", "who": {"a": who("unsure"), "b": who("unsure")}}]},
                 {"first": 10, "last": 19, "split_from": old, "who": {"a": who("yes"), "b": who("yes")}, "order": ["b", "a"], "order_by": "a session"},
                 {"first": 20, "last": 39, "who": {"a": who("no"), "b": who("yes")},
                  "merged_from": [{"first": 20, "last": 29, "who": {"a": who("no"), "b": who("yes")}, "note": "merged"},
                                  {"first": 30, "last": 39, "who": {"a": who("no"), "b": who("yes")}, "scene_score": 0.7}],
                  "moved_from": [{"first": 20, "last": 38, "who": {"a": who("no"), "b": who("yes")}, "scene_score": 0.6}]}]}
    seed = lambda n, label, **more: dict({"answer": "here", "at": f"2026-01-01T00:00:0{n}Z", "box": [1.0, 2.0, 3.0, 4.0], "by": "a person", "frame": n,
                                          "schema": "h3_cast/1", "size": [64, 48], "subject": label, "table": "a table"}, **more)
    cast = [seed(1, "a"), seed(2, "b", encloses="head", drawn="approximate"), dict(seed(3, "a"), answer="withdrawn", box=None, size=None)]
    subject = lambda still, **more: dict({
        "region": "face", "by": "a person", "still": still, "parts": ["face_and_neck"], "inputs_by": "a session",
        "text": {"who": "the walker", "looks": "plain", "her": "her", "she": "she", "hair_words": "dark hair", "hair_words_by": "a person",
                 "picture_hair": "dark hair", "picture_hair_by": "a person"},
        "reference": {"crop": "cranium", "open_mouth": "outlined", "closed_mouth": "sculpted", "why": "a person"}, "reference_by": "a session",
        "depth": {"on": True, "why": "a person"},
        "recipe": {"margin": 16, "zoom_up_to": 4.0, "crop": "cranium", "open_mouth": "outlined", "closed_mouth": "sculpted", "by": "a person"}}, **more)
    plan = {"schema": "h3_plan/1", "video": "walk.mp4", "written_by": "a session",
            "subjects": {
                "a": subject("stills/a.png", keep_outs=["strangers"], keep_outs_by="a session", restore_by="a session", pieces_by_removed="a session",
                             restore=[{"classes": ["Eyeglass"], "first": 2, "last": 5, "exception": "a person", "exception_by": "a session",
                                       "loads": ["a_fp_0000"], "touching": "head"}],
                             voice=[{"first": 0, "last": 9, "voice": "silent", "by": "a person"}],
                             shot_overrides=[{"first": 0, "last": 9, "margin": 32, "zoom_up_to": 2.0, "open_mouth": "inner outline",
                                              "closed_mouth": "sculpted", "why": "small", "by": "a person"}]),
                "b": subject("b.jpg", keep_outs_removed_by="a session", hair_band_removed_by="a session", restore_removed_by="a session",
                             look={"change": [{"room": "hairline", "by": "a session"}], "stay_removed_by": "a person"}, look_by="a session",
                             clothing=[{"first": 10, "last": 19, "words": "a coat", "by": "a session"}],
                             head_marks=[{"first": 12, "last": 13, "mark": "hidden", "by": "a person"}],
                             source_facial_hair=True, source_facial_hair_by="a person")},
            "aliases": {"walker": "a", "_by": "a session"}, "load_prefixes": {"a": ["a_"], "b": ["b_"], "_by": "a session"},
            "departures": [{"first": 0, "last": 9, "subject": "a", "note": "a note", "by": "a person"}],
            "text": {"film": "A made-up film."},
            "not_people": [{"what": "a poster", "shots": [[0, 9]], "by": "a person", "counts_as_no_one": "in these shots"}]}
    plan["subjects"]["a"]["text"].update(sings=True, looks_by="a person", hair="her dark hair", not_used="the coat", hair_change_removed_by="a person")
    answer = lambda sec, **more: dict({"at": f"2026-01-03T00:00:{sec:02d}Z", "who": "owner", "row": "a_fp_0000", "what": "choice", "chosen": "keep this one"}, **more)
    answers = [
        answer(1, video="walk", frames=[0, 9], clip="Video/walk/clips/a.mp4", clip_sha256=SHA(11), render="Video/walk/s/a_fp_0000_00001.mp4", render_sha256=SHA(1)),
        answer(2, what="words", words="good", chosen=None, frame=3, source_frame=3, subject="a", relayed_by="a session", relayed_from="a terminal",
               said_in="the terminal", taken_down_by="a session", corrects="line 1", clip="data/walk/review/a.mp4", clip_not_known="no item"),
        answer(3, row="b20", n="7", shot=1, source_frame=12, table_frame=4, table="a table", person=2, video="walk.mp4"),
        answer(4, row="segments", what="parts", chosen="parts", person="b", source_frame="default", preview="a preview", replace=["Face_Neck"],
               keep_out=["Left_Hand"], leave=["Torso"], not_said=["Apparel"]),
        answer(5, bad_frames=[[2, 3]], what_about="a mark", marked_by="a session", sheet="data/walk/sheet.webp", note="a note", source="a terminal",
               answers=[{"inline": "an answer"}]),
        dict(answer(6), who="a session (test of the page)")]
    for a in answers:
        if a.get("chosen") is None:
            a.pop("chosen", None)
    laid = lambda n, label, load, first, last, **more: dict({
        "schema": "h3_laid/1", "first": first, "last": last, "subject": label, "load": load, "file": f"Video/walk/s/{load}_00001.mp4", "sha256": SHA(n),
        "owner": "kept", "by": "the owner's keep", "frames": [[first, last]], "added": "2026-01-04T00:00:00Z", "added_by": "a session"}, **more)
    kept = {"at": "2026-01-05T00:00:00Z", "by_session": "a session", "owner_was": "unconfirmed", "by_was": "a go to render", "owner": "kept",
            "by": "the owner's keep", "owner_word": "data/walk/owner_answers.jsonl:1", "no_rebuild": "a batch"}
    lines = [
        laid(1, "a", "a_fp_0000", 0, 9, owner_word="data/walk/owner_answers.jsonl:1", zoomed_sha256=SHA(21), history=[kept], no_rebuild="a batch",
             replaces="a_fp_0000x (sha256 00aa, laid on [[0, 4]]), and a_fp_0005x (sha256 00bb, laid on [[5, 9]])",
             owner_looks=["the owner's look"], recipe="data/walk/recipes/a/"),
        laid(2, "b", "b_fp_0010", 10, 19, frames=[[10, 15]], partial_lay_why="hidden after 15", replaces=["b_fp_0010x", "b_fp_0010y"],
             seeded_from="a table", owner="kept in a joined clip", owner_was="kept", owner_corrected="given on a joined clip"),
        laid(3, "b", "b_fp_0020", 20, 39, replaces="b_fp_0020x"),
        laid(4, "a", "a_fp_0010", 10, 19, added="2026-02-02T00:00:00Z")]            # dated in the second project's time
    removed = [{"schema": "h3_laid_removed/1", "at": "2026-01-06T00:00:00Z", "by": "the owner: wrong", "by_session": "a session", "no_rebuild": "a batch",
                "carved_whole_by": "b_fp_0020", "laid_sha256_before": SHA(31), "laid_sha256_after": SHA(32),
                "removed": laid(5, "b", "b_fp_0020x", 20, 39, owner="unconfirmed", by="a go to render",
                                carved_by=[{"load": "b_fp_0020", "frames": [[20, 39]], "at": "2026-01-06T00:00:00Z", "reason": "carved by a newer keep"}])}]
    copy_of = lambda name, **more: dict({"name": name, "rate": 24.0, "frames": 40, "size": [64, 48], "bytes": 100, "head_sha256": SHA(41)}, **more)
    manifest = {"schema": "h3_clip_analysis", "version": 1, "source": copy_of("walk.mp4", rate=23.976), "fingerprint": SHA(42), "masks": [],
                "derived": copy_of("walk_24p_sound.mov", checked="equal", made_by="a session"),
                "copies_before": [copy_of("walk_24p.mov", first_source_frame=0, checked="equal", made_by="a session", entered_at="2026-01-01T00:00:00Z",
                                          entered_by="a session", replaced_at="2026-01-01T01:00:00Z", replaced_by="walk_24p_sound.mov", why="sound")],
                "voice": {"frames": [0, 39], "frames_are": "its own", "named_at": "2026-01-01T00:00:00Z", "named_by": "a session",
                          "per_frame": "data/walk/voice.csv", "spans": "data/walk/voice.json"}}
    entry = lambda name, folder, starts, ends, **more: dict({"name": name, "folder": folder, "video": "walk", "footage_records": "walk",
                                                             "starts_at": starts, "ends_before": ends, "by": "a person"}, **more)
    projects = {"schema": "h3_projects/1", "written_by": "this check",
                "projects": [entry("walk", "walk", None, "2026-02-01T00:00:00Z"),
                             entry("walk2", "walk_again", "2026-02-01T00:00:00Z", None, keeps={"from": "walk", "subjects": ["b"], "by": "a person"})]}
    # the second project: the same footage, another still for `a`; its laid file holds the first's pieces and one line of its own
    again = copy.deepcopy(plan)
    again["subjects"]["a"]["still"] = "stills/a2.png"
    mine = laid(6, "a", "a_fp_0000b", 0, 9, added="2026-02-03T00:00:00Z", owner="unconfirmed", by="a go to render")
    return {"projects.json": projects, "walk/manifest.json": manifest, "walk/shots.json": shots, "walk/cast/cast.jsonl": cast, "walk/plan.json": plan,
            "walk/owner_answers.jsonl": answers, "walk/laid.jsonl": lines, "walk/laid.removed.jsonl": removed,
            "walk_again/shots.json": copy.deepcopy(shots), "walk_again/plan.json": again,
            "walk_again/laid.jsonl": [copy.deepcopy(lines[0]), copy.deepcopy(lines[1]), copy.deepcopy(lines[2]), mine]}


def write(folder: Path, files: dict[str, object]) -> Path:
    for rel, data in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in data) if rel.endswith(".jsonl") else json.dumps(data, indent=1))
    return folder


def built(tmp: Path, files: dict[str, object] | None = None, name: str = "a"):
    data = write(tmp / f"data_{name}", files or project())
    con = db.connect(tmp / f"{name}.sqlite", create=True)
    return con, data, importer.Import(con, data)


def main_check() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        con, data, imp = built(tmp)
        imp.run()

        def the_schema_keeps_its_rules():
            loose = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql NOT LIKE '%STRICT%'")]
            assert not loose, f"not STRICT: {loose}"
            assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert con.execute("PRAGMA user_version").fetchone()[0] == len(db.migrations())
            try:
                con.execute("INSERT INTO laid_frames (laid_id, ord, first, last) VALUES (9999, 0, 1, 2)")
            except Exception as e:
                assert "FOREIGN KEY" in str(e), e
            else:
                raise AssertionError("a laid_frames row of no laid row was taken")
            return f"{len(db.tables(con))} tables at schema {len(db.migrations())}"

        def json_only_where_it_is_said():
            found = []
            for table in db.tables(con):
                for col in [r[1] for r in con.execute(f"PRAGMA table_info({table})") if r[2] == "TEXT"]:
                    if (table, col) == ("import_source", "raw"):
                        continue
                    for (v,) in con.execute(f"SELECT {col} FROM {table} WHERE {col} LIKE '{{%' OR {col} LIKE '[%'"):
                        try:
                            json.loads(v)
                        except ValueError:
                            continue
                        found.append(f"{table}.{col}: {v[:40]}")
            assert not found, found
            calls = [f"{p.name}: {m}" for p in sorted((db.HERE).rglob("*")) if p.suffix in (".py", ".sql")
                     for m in re.findall(r"\bjson_(?:extract|each|tree|object|array|group_array)\b|->>", p.read_text())]
            assert not calls, f"a JSON function of SQLite in the tool: {calls}"
            return "import_source.raw is the one JSON column"

        def every_mapping_line_is_used():
            unused = [f"{s} {r['prefix']}" for s, rs in imp.map.rules.items() for r in rs if r["hits"] == 0]
            cut = [u for u in unused if u.startswith("cut-") or u == "projects final_cut"]
            rest = sorted(set(unused) - set(cut))
            assert not rest, f"no record of the made-up project reaches: {rest}"
            return f"{len(cut)} line(s) of the final cut's sources are not driven here"

        def an_unmapped_field_stops_it():
            files = project()
            files["walk/laid.jsonl"][0]["a_new_field"] = 1
            con2, _, imp2 = built(tmp, files, "unmapped")
            said = refused(imp2.run, "a_new_field", "mapping.tsv")
            assert con2.execute("SELECT count(*) FROM laid").fetchone()[0] == 0 and con2.execute("SELECT count(*) FROM shot").fetchone()[0] == 0, \
                "rows of the refused build were kept"
            return said[:70]

        def a_build_is_a_function():
            con2, _, imp2 = built(tmp, name="again")
            imp2.run()
            a, b = db.dump(con), db.dump(con2)
            assert a == b, "two builds of the same files differ"
            return f"{len(a.splitlines())} lines of dump"

        def files_come_back():
            diff = export.differences(con, "walk", data)
            assert set(diff) == {"shots.json", "plan.json", "cast/cast.jsonl", "laid.jsonl", "laid.removed.jsonl"}, sorted(diff)
            # the fourth laid line is the second project's by its date, and is not in the first's file given back
            assert diff.pop("laid.jsonl") == ["laid.jsonl: 1 piece(s) in the file and not given back (a 1): a_fp_0010"], "the laid list did not come back as its first three lines"
            assert not any(diff.values()), diff
            con.execute("SAVEPOINT broken")
            con.execute("UPDATE shot_override SET margin_px = 33")
            con.execute("UPDATE seen_look SET part = 'hair'")
            named = export.differences(con, "walk", data)
            con.execute("ROLLBACK TO broken")
            assert any("margin" in f for f in named["plan.json"]) and any("seen_looks" in f for f in named["shots.json"]), named
            return "five files; two changed values named"

        def a_place_is_written_once():
            loc = con.execute("SELECT id FROM location LIMIT 1").fetchone()[0]
            one, two = (con.execute("SELECT id FROM artifact ORDER BY id LIMIT 1 OFFSET ?", (n,)).fetchone()[0] for n in (0, 1))
            con.execute("SAVEPOINT once")
            con.execute("UPDATE location SET artifact_id = ? WHERE id = ?", (one, loc))
            con.execute("UPDATE location SET artifact_id = ?, verified_at = 'again' WHERE id = ?", (one, loc))     # the same bytes: taken
            try:
                con.execute("UPDATE location SET artifact_id = ? WHERE id = ?", (two, loc))
            except Exception as e:
                assert "written once" in str(e), e
            else:
                raise AssertionError("other bytes were taken at a place that had its bytes")
            finally:
                con.execute("ROLLBACK TO once")

        def history_is_states():
            states = [tuple(r) for r in con.execute("SELECT s.ord, s.owner_status, s.why, s.word_line, s.answer_id IS NOT NULL FROM laid_state s JOIN laid l ON l.id = s.laid_id WHERE l.list_line = 1 ORDER BY s.ord")]
            assert states == [(0, "unconfirmed", "a go to render", None, 0), (1, "kept", "the owner's keep", 1, 1)], states
            files = project()
            files["walk/laid.jsonl"][0]["history"][0]["owner_was"] = "kept"
            files["walk/laid.jsonl"][0]["history"].insert(0, copy.deepcopy(files["walk/laid.jsonl"][0]["history"][0]))
            _, _, imp2 = built(tmp, files, "history")
            return refused(imp2.run, "history entry", "did not leave")[:60]

        def a_dated_record_is_its_projects():
            got = [tuple(r) for r in con.execute("""SELECT p.name, d.name FROM laid l JOIN project p ON p.id = l.project_id JOIN load d ON d.id = l.load_id
                                                    JOIN import_file i ON i.id = l.file_id JOIN location o ON o.id = i.location_id
                                                    WHERE l.list_line = 4 AND o.relpath = 'walk/laid.jsonl'""")]
            assert got == [("walk2", "a_fp_0010")], got
            assert any("walk2" in n for n in imp.notes), imp.notes
            assert [r["load"] for r in export.Export(con, "walk").laid()] == ["a_fp_0000", "b_fp_0010", "b_fp_0020"]
            assert [r["load"] for r in export.Export(con, "walk2").laid(["a"])] == ["a_fp_0010", "a_fp_0000b"]

        def a_project_keeps_anothers_pieces():
            ex = export.Export(con, "walk2")
            assert [(r["subject"], r["load"]) for r in ex.laid()] == [("b", "b_fp_0010"), ("b", "b_fp_0020"), ("a", "a_fp_0010"), ("a", "a_fp_0000b")], ex.laid()
            assert con.execute("SELECT count(*) FROM laid").fetchone()[0] == 6, "a line of the second project's file dated before it began became a row"
            note = next(n for n in imp.notes if "pieces it keeps" in n)
            assert "holds 2 line(s) that are pieces it keeps" in note and "1 dated before it began that are not (a 1)" in note, note
            stills = [r[0] for r in con.execute("SELECT o.relpath FROM subject_plan sp JOIN project p ON p.id = sp.project_id JOIN location o ON o.id = sp.still_location_id WHERE p.name = 'walk2' ORDER BY sp.id")]
            assert stills == ["stills/a2.png", "b.jpg"], stills
            counts = [con.execute(q).fetchone()[0] for q in ("SELECT count(*) FROM head_mark", "SELECT count(*) FROM head_mark_standing",
                                                             "SELECT count(*) FROM shot WHERE ord IS NOT NULL")]
            assert counts == [2, 1, 3], f"each plan's head mark, one standing, and the shots once: {counts}"
            diff = export.differences(con, "walk2", data)
            assert diff["shots.json"] == [] and diff["plan.json"] == [], diff
            assert diff["laid.jsonl"] == ["laid.jsonl: 1 piece(s) in the file and not given back (a 1): a_fp_0000",
                                          "laid.jsonl: 1 piece(s) given back and not in the file (a 1): a_fp_0010"], diff["laid.jsonl"]
            con.commit()
            assert main(["compare", "--project", "walk2", "--db", str(tmp / "a.sqlite"), "--data", str(data)]) == 1, "compare did not exit 1 on a difference"
            assert main(["compare", "--project", "walk2", "--db", str(tmp / "a.sqlite"), "--data", str(tmp / "nowhere")]) == 1, "compare passed with no file to read"

        def a_footage_fact_said_twice():
            files = project()
            files["walk_again/plan.json"]["subjects"]["b"]["head_marks"][0]["last"] = 14
            con2, _, imp2 = built(tmp, files, "twice")
            imp2.run()
            note = next((n for n in imp2.notes if "footage fact" in n), "")
            assert "subjects.b.head_marks" in note and "data/walk/plan.json" in note and "the later word stands" in note, imp2.notes
            assert [tuple(r) for r in con2.execute("SELECT p.name, h.first, h.last FROM head_mark h JOIN project p ON p.id = h.project_id ORDER BY h.id")] \
                == [("walk", 12, 13), ("walk2", 12, 14)]
            # THE CASE THAT GOES RED when the first plan's word stands over a later one
            assert [tuple(r) for r in con2.execute("SELECT first, last FROM head_mark_standing")] == [(12, 14)]
            data2 = tmp / "data_twice"
            assert export.differences(con2, "walk", data2)["plan.json"] == [] and export.differences(con2, "walk2", data2)["plan.json"] == [], \
                "a plan file did not come back as it is"
            assert not any("footage fact" in n for n in imp.notes), "a note of a difference where the two plans agree"
            files = project()
            files["walk_again/shots.json"]["shots"][0]["last"] = 8
            _, _, imp3 = built(tmp, files, "shots")
            return refused(imp3.run, "is not the footage's shots")[:60]

        def an_old_file_is_brought_up():
            import sqlite3
            old = sqlite3.connect(tmp / "old.sqlite")
            old.executescript((db.MIGRATIONS / "0001_first_tier.sql").read_text() + "\nPRAGMA user_version = 1;")
            old.executescript("""INSERT INTO root VALUES ('data', 'x'); INSERT INTO artifact (id, sha256) VALUES (1, '%s');
                INSERT INTO location (id, root, relpath) VALUES (1, 'data', 'v/cast.jsonl'), (2, 'data', 'v/owner_answers.jsonl'), (3, 'data', 'v/laid.jsonl');
                INSERT INTO import_file (id, location_id, artifact_id, source) VALUES (1, 1, 1, 'cast'), (2, 2, 1, 'answers'), (3, 3, 1, 'laid');
                INSERT INTO video (id, name, source_name, frames, film_words) VALUES (1, 'v', 'v.mp4', 9, 'a film');
                INSERT INTO project (id, name, video_id, folder, said_by) VALUES (1, 'v', 1, 'v', 'x'), (2, 'v2', 1, 'v_again', 'x');
                INSERT INTO subject (id, video_id, label, ord) VALUES (1, 1, 'a', 0);
                INSERT INTO seed (subject_id, line, frame, answer, from_table, said_by, at) VALUES (1, 1, 2, 'here', 't', 'x', 'x');
                INSERT INTO head_mark (subject_id, ord, first, last, mark, said_by) VALUES (1, 0, 2, 3, 'hidden', 'x');
                INSERT INTO voice_span (subject_id, ord, first, last, voice, said_by) VALUES (1, 0, 2, 3, 'silent', 'x');
                INSERT INTO facial_hair (subject_id, has, said_by) VALUES (1, 1, 'x');
                INSERT INTO alias (subject_id, alias, ord, said_by) VALUES (1, 'old', 0, 'x');
                INSERT INTO not_person (id, video_id, ord, what, said_by) VALUES (1, 1, 0, 'a poster', 'x');
                INSERT INTO not_person_span (not_person_id, ord, first, last) VALUES (1, 0, 0, 9);
                INSERT INTO subject_plan (project_id, subject_id, region, said_by, still_location_id, margin_px, zoom_up_to, crop, open_mouth, closed_mouth,
                                          recipe_by, depth_on) VALUES (2, 1, 'face', 'x', 1, 16, 4.0, 'head', 'outlined', 'sculpted', 'x', 1);
                INSERT INTO answer (video_id, about, line, at, who, what) VALUES (1, 'footage', 1, 'x', 'owner', 'words');
                INSERT INTO load (id, project_id, name) VALUES (1, 1, 'l');
                INSERT INTO laid (id, project_id, subject_id, load_id, first, last, render_location_id, render_artifact_id, added_at, added_by, list_line)
                  VALUES (1, 1, 1, 1, 0, 9, 3, 1, 'x', 'x', 1);
                INSERT INTO laid_state (laid_id, ord, owner_status, why, at) VALUES (1, 0, 'kept', 'x', 'x');
                INSERT INTO laid_frames (laid_id, ord, first, last) VALUES (1, 0, 0, 9);""" % SHA(1))
            old.commit()
            old.close()
            up = db.connect(tmp / "old.sqlite")
            assert up.execute("PRAGMA user_version").fetchone()[0] == len(db.migrations())
            got = [tuple(r) for r in up.execute("SELECT l.file_id, s.file_id, a.file_id, (SELECT count(*) FROM laid_standing), (SELECT count(*) FROM project_laid) FROM laid l, seed s, answer a")]
            assert got == [(3, 1, 2, 1, 1)], got
            assert not up.execute("PRAGMA foreign_key_check").fetchall()
            # the old file's one word a video stood for every project: the later project with a plan is given it, so what stood still stands
            facts = [[r[0] for r in up.execute(f"SELECT project_id FROM {t} ORDER BY 1")] for t in ("head_mark", "voice_span", "facial_hair", "alias", "not_person")]
            assert facts == [[1, 2]] * 5 and [r[0] for r in up.execute("SELECT film_words FROM project ORDER BY id")] == ["a film", "a film"], facts
            assert up.execute("SELECT count(*) FROM not_person_span s JOIN not_person n ON n.id = s.not_person_id").fetchone()[0] == 2
            standing = [up.execute(f"SELECT count(*) FROM {t}_standing").fetchone()[0] for t in ("head_mark", "voice_span", "facial_hair", "alias", "not_person")]
            assert standing == [1] * 5 and up.execute("SELECT film_words FROM film_words_standing").fetchone()[0] == "a film", standing

        def subjects_start_a_project():
            con.commit()
            out = tmp / "seed"
            assert main(["export", "--project", "walk", "--db", str(tmp / "a.sqlite"), "--out", str(out), "--subjects", "b"]) == 0
            rows = [json.loads(line) for line in (out / "laid.jsonl").read_text().splitlines()]
            assert [(r["subject"], r["load"]) for r in rows] == [("b", "b_fp_0010"), ("b", "b_fp_0020")], rows
            assert main(["export", "--project", "walk2", "--db", str(tmp / "a.sqlite"), "--out", str(tmp / "seed2"), "--subjects", "b"]) == 0
            assert (tmp / "seed2" / "laid.jsonl").read_text() == (out / "laid.jsonl").read_text(), "the kept pieces are not the first project's, as they stand"
            assert sorted(p.name for p in out.iterdir()) == ["laid.jsonl"]
            assert main(["export", "--project", "walk", "--db", str(tmp / "a.sqlite"), "--out", str(out), "--subjects", "nobody"]) == 1

        def the_lanes_file_waits():
            asks = cover.Mapping().asks()
            inside = main(["build", "--db", str(db.REPO / "bench" / "x.sqlite")])
            assert inside == 1 and not (db.REPO / "bench" / "x.sqlite").exists()
            if not asks:
                return "no mapping line is `ask`: only the path inside the repo was tried"
            before = db.THE_RECORD.exists()
            assert main(["build", "--db", str(db.THE_RECORD)]) == 1
            assert db.THE_RECORD.exists() == before, "data/lane.sqlite was touched"
            return f"{len(asks)} line(s) are `ask`"

        for fn in (the_schema_keeps_its_rules, json_only_where_it_is_said, every_mapping_line_is_used, an_unmapped_field_stops_it, a_build_is_a_function,
                   files_come_back, a_place_is_written_once, history_is_states, a_dated_record_is_its_projects, a_project_keeps_anothers_pieces,
                   a_footage_fact_said_twice, an_old_file_is_brought_up, subjects_start_a_project,
                   the_lanes_file_waits):
            case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main_check())
