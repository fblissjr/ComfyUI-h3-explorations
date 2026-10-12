#!/usr/bin/env python3
"""`bench/sweep_loads.py` against job files made here: a copy differs from its source in the named change and the names only.

A sweep is read as the effect of one input. A copy that carries a second difference, or a first one in the wrong
place, makes that reading false, and the renders look like any others. So this writes small job files, runs the
tool, and holds each copy to its source by UNDOING what the sweep says it did and asking for the source back,
which uses none of the tool's own comparison.

1. **A still.** The named subjects' stills change in the job block and in a load that names its own; another
   subject's do not; undoing the stills, the tag on the names and the note gives the source back, key for key.
   The copy sits beside its source; the record says the change, each load's old and new name, which loads the
   change touches and every place the files differ; the commands are for the touched loads only, in order.
2. **A text.** The label's file and who wrote it change, the frames do not; the loads touched are those that
   take that label, a default included; a text that is not beside the job file is refused, and so is a text with
   nobody named.
3. **A motion video.** Only the named load's, with its first frame left as it was; a load that had none gets
   the key and the record says it was absent; a file that does not exist is refused.
4. **A setting.** Every load gets it, over the file's defaults; a load already at that value is not touched; a
   key that is not a setting is refused.
5. **Nothing is half written.** Two kinds of change, no change, a change that touches nothing in one of the job
   files, a name no job file has, a tag used before and a tag that is not a name are each refused, and no copy
   and no record exists afterwards.
6. **A difference outside the change is found.** `differences` names a changed value, an added key, a removed
   key, a list of another length and a number that became a string; a copy edited after the sweep is said by
   `landed`, with where.
7. **A reference and the text that describes it are one change.** A motion video and a text given together
   make one copy with both, of the kind `reference`; undoing both gives the source back; a still with a text is
   still two changes and refused. `LOAD=FILE@N` sets the load's `motion_first` with the file.
8. **The capture a copy is built on.** With `--capture` for its job the copy names that folder, the record says
   it is its own and the difference is accounted for; without it the copy keeps its source's and the tool says
   so, in what it prints and beside the commands; a job that was not given is refused with nothing written.
9. **Where each render landed.** From the records the job builder leaves: not built, built and not queued, and
   queued with what the server said and the files, each beside its source load's name.

No model, no card, no server, no ffmpeg.

    <comfy venv python> bench/check_sweep_loads.py
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import sweep_loads as sw  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="check_sweep_loads_"))
quiet = lambda *_: None  # noqa: E731
DATE = "2026-01-01"

KITCHEN = {"what": "a job of two subjects",
           "job": {"name": "kitchen", "note": "what the render is meant to decide", "source": "/clips/x.mov", "first": 600, "frames": 400,
                   "canvas": [1024, 768], "output": "Video/x",
                   "subjects": [{"label": "lead", "pick": "largest", "still": "lead_a.jpg"}, {"label": "second", "pick": "largest", "still": "mom_a.jpg"}],
                   "left_out": [], "capture": "data/2026-01-01_kitchen",
                   "texts": {"t1": {"file": "t1.txt", "frames": [600, 700], "by": "a session"}, "t2": {"file": "t2.txt", "frames": [600, 700], "by": "a session"}}},
           "defaults": {"seed": 5, "text": "t1"},
           "loads": [{"name": "lead_0600", "first": 600, "frames": 100, "subject": "lead", "motion": "mesh", "motion_file": "/share/lead_ref.mp4", "motion_first": 600},
                     {"name": "second_0600", "first": 600, "frames": 100, "subject": "second", "text": "t2", "still": "mom_own.jpg", "seed": 9,
                      "continue_from": "/out/window.safetensors"}]}
STREET = {"job": {"name": "street", "source": "/clips/x.mov", "first": 0, "frames": 300, "canvas": [1024, 768], "output": "Video/x",
                  "subjects": [{"label": "lead", "pick": "largest", "still": "lead_a.jpg"}], "capture": "data/2026-01-01_street",
                  "texts": {"s1": {"file": "s1.txt", "frames": [0, 99], "by": "a session"}}},
          "defaults": {},
          "loads": [{"name": "lead_0000", "first": 0, "frames": 100, "subject": "lead", "text": "s1", "seed": 5}]}


def room(name: str) -> tuple[Path, Path, Path]:
    """A fresh folder with the two job files and their texts: (the folder, kitchen.json, street.json)."""
    folder = WORK / name
    (folder / "jobs").mkdir(parents=True)
    for stem, job in (("kitchen", KITCHEN), ("street", STREET)):
        (folder / "jobs" / f"{stem}.json").write_text(json.dumps(job, indent=1) + "\n")
    for text in ("t1.txt", "t2.txt", "s1.txt", "t1_new.txt"):
        (folder / "jobs" / text).write_text("words\n")
    return folder, folder / "jobs" / "kitchen.json", folder / "jobs" / "street.json"


def run(folder: Path, files, tag: str, **change) -> dict:
    spec = sw.change_of(change.get("still", []), change.get("text", []), change.get("text_by", ""), change.get("motion_file", []), change.get("setting", ""))
    return json.loads(sw.make(list(files), spec, tag, "the check", captures=change.get("captures"), data=folder / "data", date=DATE,
                              say=change.get("say", quiet)).read_text())


def undone(copied: dict, source: dict, tag: str) -> dict:
    """A copy with the tag taken off its names and the source's note put back: what is left differs from the source by the change alone."""
    back = copy.deepcopy(copied)
    assert back["job"]["name"].endswith(f"_{tag}"), back["job"]["name"]
    back["job"]["name"] = back["job"]["name"][:-len(tag) - 1]
    assert back["job"]["note"].startswith(f"SWEEP {tag}: ") and "the check" in back["job"]["note"], back["job"]["note"]
    if "note" in source["job"]:
        assert back["job"]["note"].endswith(source["job"]["note"]), "the source's own note is not kept under the sweep's"
        back["job"]["note"] = source["job"]["note"]
    else:
        del back["job"]["note"]
    for one in back["loads"]:
        assert one["name"].endswith(f"_{tag}"), one["name"]
        one["name"] = one["name"][:-len(tag) - 1]
    return back


def refused(fn, *words: str) -> None:
    try:
        fn()
    except sw.Refused as why:
        assert all(w in str(why) for w in words), f"refused without saying {words}: {why}"
        return
    raise AssertionError(f"not refused; expected a refusal saying {words}")


def a_still() -> str:
    folder, kitchen, street = room("still")
    rec = run(folder, (kitchen, street), "b", still=["lead=lead_b.jpg", "second=mom_b.jpg"])
    k, s = (json.loads((folder / "jobs" / f"{stem}__b.json").read_text()) for stem in ("kitchen", "street"))
    assert [sub["still"] for sub in k["job"]["subjects"]] == ["lead_b.jpg", "mom_b.jpg"] and k["loads"][1]["still"] == "mom_b.jpg"
    assert "still" not in k["loads"][0], "a load that named no still of its own was given one"
    back = undone(k, KITCHEN, "b")
    back["job"]["subjects"][0]["still"], back["job"]["subjects"][1]["still"], back["loads"][1]["still"] = "lead_a.jpg", "mom_a.jpg", "mom_own.jpg"
    assert back == KITCHEN and list(back) == list(KITCHEN) and list(back["job"]) == list(KITCHEN["job"]), "undoing the stills and the names does not give the source back"
    back = undone(s, STREET, "b")
    back["job"]["subjects"][0]["still"] = "lead_a.jpg"
    assert back == STREET, "the second job file's copy differs from it beyond the still"
    assert rec["change"] == {"kind": "still", "values": {"lead": "lead_b.jpg", "second": "mom_b.jpg"}} and rec["made_by"] == "the check"
    first = rec["copies"][0]
    assert first["job"] == {"from": "kitchen", "to": "kitchen_b"} and first["capture"] == KITCHEN["job"]["capture"]
    assert [(x["from"], x["to"], x["touched"]) for x in first["loads"]] == [("lead_0600", "lead_0600_b", True), ("second_0600", "second_0600_b", True)]
    assert sorted(first["changed_at"]) == ["job.subjects[0].still", "job.subjects[1].still", "loads[1].still"], first["changed_at"]
    assert {d["where"] for d in first["differs_at"]} == set(first["changed_at"]) | {"job.name", "job.note", "loads[0].name", "loads[1].name"}
    assert any("second_0600_b" in line and "SAME stored window" in line for line in first["continues_another_run"]), first["continues_another_run"]
    only = run(folder, (kitchen,), "c", still=["second=mom_c.jpg"])["copies"][0]                 # one subject of the two
    assert only["touched"] == ["second_0600_c"] and [x["touched"] for x in only["loads"]] == [False, True]
    one = json.loads((folder / "jobs" / "kitchen__c.json").read_text())
    assert one["job"]["subjects"][0]["still"] == "lead_a.jpg" and one["loads"][0]["name"] == "lead_0600_c", "the subject not named changed, or its load kept its name"
    cmds = run(folder, (kitchen, street), "d", still=["lead=lead_d.jpg"])["commands"]
    assert not any("second_0600_d" in c for c in cmds), "a load the change does not touch is in the commands"
    order = [c.split("bench/masked_job.py ")[1].split()[:2] for c in cmds if "masked_job.py" in c]
    assert order == [["build", "plan"], ["queue", "data/<the"], ["build", "render"], ["queue", "data/<the"]] * 2, order
    assert sum("between plan and render" in c for c in cmds) == 2 and all("--load lead_0" in c for c in cmds if "masked_job.py" in c)
    assert sorted(p.name for p in (folder / "data").iterdir()) == [f"{DATE}_sweep_{t}" for t in "bcd"]
    return "two subjects' stills over two job files, one subject of two, and the commands for the touched loads only"


def a_text() -> str:
    folder, kitchen, street = room("text")
    rec = run(folder, (kitchen,), "words", text=["t1=t1_new.txt"], text_by="the check, from frames 600 to 700")
    k = json.loads((folder / "jobs" / "kitchen__words.json").read_text())
    assert k["job"]["texts"]["t1"] == {"file": "t1_new.txt", "frames": [600, 700], "by": "the check, from frames 600 to 700"}, k["job"]["texts"]["t1"]
    back = undone(k, KITCHEN, "words")
    back["job"]["texts"]["t1"].update(file="t1.txt", by="a session")
    assert back == KITCHEN, "the copy differs from its source beyond the text's file and who wrote it"
    assert rec["copies"][0]["touched"] == ["lead_0600_words"], "the load that takes the label by the file's default is not the one touched"
    refused(lambda: run(folder, (kitchen,), "gone", text=["t1=not_there.txt"], text_by="x"), "not beside the job file")
    refused(lambda: run(folder, (kitchen,), "nobody", text=["t1=t1_new.txt"]), "--text-by")
    refused(lambda: run(folder, (kitchen, street), "half", text=["t1=t1_new.txt"], text_by="x"), "touches nothing", "street.json")
    return "a label's file and its writer change and its frames do not; the load that takes it by default is the one touched"


def a_motion_video() -> str:
    folder, kitchen, street = room("motion")
    new = folder / "reference.mp4"
    new.write_bytes(b"")
    rec = run(folder, (kitchen, street), "crop", motion_file=[f"lead_0600={new}", f"lead_0000={new}"])
    k, s = (json.loads((folder / "jobs" / f"{stem}__crop.json").read_text()) for stem in ("kitchen", "street"))
    assert k["loads"][0]["motion_file"] == str(new) and k["loads"][0]["motion_first"] == 600 and "motion_file" not in k["loads"][1]
    back = undone(k, KITCHEN, "crop")
    back["loads"][0]["motion_file"] = "/share/lead_ref.mp4"
    assert back == KITCHEN
    back = undone(s, STREET, "crop")
    del back["loads"][0]["motion_file"]
    assert back == STREET and rec["copies"][1]["changed_at"] == ["loads[0].motion_file (absent on one side)"], rec["copies"][1]["changed_at"]
    assert rec["copies"][0]["touched"] == ["lead_0600_crop"]
    refused(lambda: run(folder, (kitchen,), "nofile", motion_file=["lead_0600=/nowhere/x.mp4"]), "is not a file")
    refused(lambda: run(folder, (kitchen,), "noload", motion_file=[f"lead_9999={new}"]), "touches nothing")
    return "only the named loads', with the first frame left as it was; a load that had none is recorded as absent on one side"


def a_setting() -> str:
    folder, kitchen, street = room("set")
    rec = run(folder, (kitchen, street), "seed9", setting="seed=9")
    k, s = (json.loads((folder / "jobs" / f"{stem}__seed9.json").read_text()) for stem in ("kitchen", "street"))
    assert k["loads"][0]["seed"] == 9 and k["loads"][1]["seed"] == 9 and k["defaults"] == KITCHEN["defaults"] and s["loads"][0]["seed"] == 9
    assert rec["copies"][0]["touched"] == ["lead_0600_seed9"], "a load already at that seed is touched, or the one that took the default is not"
    assert rec["copies"][0]["changed_at"] == ["loads[0].seed (absent on one side)"] and rec["copies"][1]["changed_at"] == ["loads[0].seed"]
    back = undone(k, KITCHEN, "seed9")
    del back["loads"][0]["seed"]
    assert back == KITCHEN
    assert rec["change"]["values"] == {"seed": 9}, "a number given on the command line became a string"
    refused(lambda: run(folder, (kitchen,), "bad", setting="colour=red"), "not a setting")
    refused(lambda: run(folder, (street,), "same", setting="seed=5"), "touches nothing")
    return "every load gets the setting over the defaults; a load already at the value is not touched; a number stays a number"


def a_reference_and_its_text() -> str:
    folder, kitchen, street = room("reference")
    new = folder / "reference.mp4"
    new.write_bytes(b"")
    rec = run(folder, (kitchen,), "fp", motion_file=[f"lead_0600={new}@604"], text=["t1=t1_new.txt"], text_by="the check")
    k = json.loads((folder / "jobs" / "kitchen__fp.json").read_text())
    assert rec["change"]["kind"] == "reference" and "and with it" in rec["change_in_words"] and "from source frame 604" in rec["change_in_words"]
    assert k["loads"][0]["motion_file"] == str(new) and k["loads"][0]["motion_first"] == 604 and k["job"]["texts"]["t1"]["file"] == "t1_new.txt"
    assert sorted(rec["copies"][0]["changed_at"]) == ["job.texts.t1.by", "job.texts.t1.file", "loads[0].motion_file", "loads[0].motion_first"], rec["copies"][0]["changed_at"]
    assert rec["copies"][0]["touched"] == ["lead_0600_fp"], rec["copies"][0]["touched"]
    back = undone(k, KITCHEN, "fp")
    back["loads"][0].update(motion_file="/share/lead_ref.mp4", motion_first=600)
    back["job"]["texts"]["t1"].update(file="t1.txt", by="a session")
    assert back == KITCHEN, "undoing the reference and its text does not give the source back"
    refused(lambda: run(folder, (kitchen,), "two", still=["lead=lead_b.jpg"], text=["t1=t1_new.txt"], text_by="x"), "one thing")
    refused(lambda: run(folder, (kitchen,), "three", motion_file=[f"lead_0600={new}"], setting="seed=1"), "one thing")
    refused(lambda: run(folder, (kitchen, street), "half", motion_file=[f"lead_0600={new}", f"nobody_0000={new}"], text=["t1=t1_new.txt"], text_by="x"), "touches nothing")
    plain = run(folder, (street,), "at", motion_file=[f"lead_0000={new}@12"])["copies"][0]["changed_at"]
    assert plain == ["loads[0].motion_file (absent on one side)", "loads[0].motion_first (absent on one side)"], plain
    return "a motion video and its text make one copy with both; the frame the new video begins on goes with the file; a still with a text is still refused"


def the_capture_a_copy_is_built_on() -> str:
    folder, kitchen, street = room("capture")
    said = []
    rec = run(folder, (kitchen, street), "own", still=["lead=lead_b.jpg"], captures={"kitchen": "data/2026-01-01_kitchen_own"}, say=said.append)
    k, s = (json.loads((folder / "jobs" / f"{stem}__own.json").read_text()) for stem in ("kitchen", "street"))
    assert k["job"]["capture"] == "data/2026-01-01_kitchen_own" and s["job"]["capture"] == STREET["job"]["capture"]
    first, second = rec["copies"]
    assert first["capture"] == "data/2026-01-01_kitchen_own" and first["capture_is_its_own"] is True and second["capture_is_its_own"] is False
    assert "job.capture" in {d["where"] for d in first["differs_at"]} and "job.capture" not in first["changed_at"], "the capture is counted as the change"
    back = undone(k, KITCHEN, "own")
    back["job"]["subjects"][0]["still"], back["job"]["capture"] = "lead_a.jpg", KITCHEN["job"]["capture"]
    assert back == KITCHEN
    notes = [c for c in rec["commands"] if "between plan and render" in c]
    assert "KEEPS ITS SOURCE'S CAPTURE" not in notes[0] and "data/2026-01-01_kitchen_own" in notes[0], notes[0]
    assert "KEEPS ITS SOURCE'S CAPTURE" in notes[1] and "--capture street=PATH" in notes[1], notes[1]
    assert any("ITS SOURCE'S" in line and STREET["job"]["capture"] in line for line in said) and any("its own, by --capture" in line for line in said), said
    assert sw.landed(folder / "data" / f"{DATE}_sweep_own" / "sweep.json", data=folder / "data", say=quiet)["landed"]["copies_edited_since"] == [], \
        "a copy with a capture of its own reads as edited"
    refused(lambda: run(folder, (kitchen,), "nojob", still=["lead=lead_c.jpg"], captures={"hall": "data/x"}), "--capture", "hall")
    assert not (folder / "jobs" / "kitchen__nojob.json").exists() and not (folder / "data" / f"{DATE}_sweep_nojob").exists()
    return "a copy names the capture given for its job and it is accounted for; one that keeps its source's is told so twice; an unknown job is refused"


def nothing_is_half_written() -> str:
    folder, kitchen, street = room("half")
    for what, call, words in (
            ("two kinds", lambda: run(folder, (kitchen,), "x", still=["lead=a.jpg"], setting="seed=1"), ("one thing",)),
            ("no change", lambda: run(folder, (kitchen,), "x"), ("one thing",)),
            ("a change that touches nothing in one job file", lambda: run(folder, (kitchen, street), "x", still=["second=mom_b.jpg"]), ("touches nothing", "street.json")),
            ("a name no job file has", lambda: run(folder, (kitchen, street), "x", still=["lead=lead_b.jpg", "nobody=n.jpg"]), ("nobody",)),
            ("a tag that is not a name", lambda: run(folder, (kitchen,), "Two Words", still=["lead=lead_b.jpg"]), ("--tag",)),
            ("nobody named", lambda: sw.make([kitchen], sw.change_of(["lead=b.jpg"], [], "", [], ""), "x", " ", data=folder / "data", date=DATE, say=quiet), ("--by",)),
            ("the same file twice", lambda: run(folder, (kitchen, kitchen), "x", still=["lead=lead_b.jpg"]), ("twice",))):
        refused(call, *words)
        left = sorted(p.name for p in (folder / "jobs").glob("*__*.json"))
        assert not left and not (folder / "data").exists(), f"{what}: refused, and {left or 'a record'} was written"
    run(folder, (kitchen,), "x", still=["lead=lead_b.jpg"])
    refused(lambda: run(folder, (kitchen,), "x", still=["lead=lead_c.jpg"]), "never reuses a tag")
    (folder / "data" / f"{DATE}_sweep_x" / "sweep.json").unlink()
    refused(lambda: run(folder, (kitchen,), "x", still=["lead=lead_c.jpg"]), "kitchen__x.json", "never reuses a tag")
    return "each refusal leaves no copy and no record; a tag used before is refused by its record and by its copy"


def a_difference_outside_the_change() -> str:
    a = {"job": {"name": "n", "canvas": [1024, 768]}, "loads": [{"name": "l", "seed": 5}]}
    for what, edit, where in (
            ("a changed value", lambda b: b["job"]["canvas"].__setitem__(1, 720), "job.canvas[1]"),
            ("an added key", lambda b: b["loads"][0].__setitem__("margin", 48), "loads[0].margin (absent on one side)"),
            ("a removed key", lambda b: b["loads"][0].pop("seed"), "loads[0].seed (absent on one side)"),
            ("a list of another length", lambda b: b["loads"].append({"name": "m"}), "loads (lists of different lengths)"),
            ("a number that became a string", lambda b: b["loads"][0].__setitem__("seed", "5"), "loads[0].seed")):
        b = copy.deepcopy(a)
        edit(b)
        found = [d[0] for d in sw.differences(a, b)]
        assert found == [where], f"{what}: {found}"
    assert sw.differences(a, copy.deepcopy(a)) == []
    folder, kitchen, _ = room("edited")
    rec_path = sw.make([kitchen], sw.change_of(["lead=lead_b.jpg"], [], "", [], ""), "e", "the check", data=folder / "data", date=DATE, say=quiet)
    assert sw.landed(rec_path, data=folder / "data", say=quiet)["landed"]["copies_edited_since"] == []
    target = folder / "jobs" / "kitchen__e.json"
    job = json.loads(target.read_text())
    job["loads"][0]["margin"] = 16                                                              # a second difference, by hand, after the sweep
    target.write_text(json.dumps(job, indent=1) + "\n")
    said = []
    after = sw.landed(rec_path, data=folder / "data", say=said.append)["landed"]["copies_edited_since"]
    assert len(after) == 1 and "loads[0].margin" in after[0] and any("EDITED SINCE THE SWEEP" in line for line in said), after
    return "every kind of difference is named with its place; a copy edited after the sweep is said, with where"


def where_each_render_landed() -> str:
    folder, kitchen, street = room("landed")
    rec_path = sw.make([kitchen, street], sw.change_of(["lead=lead_b.jpg"], [], "", [], ""), "f", "the check", data=folder / "data", date=DATE, say=quiet)
    rows = {r["load"]: r for r in sw.landed(rec_path, data=folder / "data", say=quiet)["landed"]["loads"]}
    assert set(rows) == {"lead_0600_f", "lead_0000_f"} and all(r["render"] == "not built" for r in rows.values()), rows
    built = folder / "data" / "2026-01-02_kitchen_f_job"
    built.mkdir()
    (built / "lead_0600_f.render.built.json").write_text(json.dumps({"built_at": "then", "commit": "abc", "output_folder": "/out/kitchen_f/lead_0600_f"}))
    queued = folder / "data" / "2026-01-03_street_f_job"
    queued.mkdir()
    (queued / "lead_0000_f.render.built.json").write_text(json.dumps({"built_at": "later", "commit": "abc", "output_folder": "/out/street_f/lead_0000_f"}))
    (queued / "lead_0000_f.render.queued.json").write_text(json.dumps({"status": "success", "files": ["Video/x/a.mp4", "Video/x/a_with_mask.mp4"]}))
    (folder / "data" / "2026-01-03_street_job").mkdir()                                           # the SOURCE's folder: not the copy's
    said = []
    saved = sw.landed(rec_path, data=folder / "data", say=said.append)
    rows = {r["load"]: r for r in saved["landed"]["loads"]}
    assert rows["lead_0600_f"]["render"] == "built, not queued" and rows["lead_0600_f"]["output_folder"] == "/out/kitchen_f/lead_0600_f"
    assert rows["lead_0600_f"]["source_load"] == "lead_0600" and rows["lead_0600_f"]["commit"] == "abc"
    assert rows["lead_0000_f"]["render"] == "queued: success" and rows["lead_0000_f"]["files"] == ["Video/x/a.mp4", "Video/x/a_with_mask.mp4"]
    assert json.loads(rec_path.read_text())["landed"]["loads"] == saved["landed"]["loads"], "what was read is not in the record"
    assert any("the source's load: lead_0000" in line and "queued: success" in line for line in said) and any("Video/x/a.mp4" in line for line in said)
    return "not built, built and not queued, and queued with the server's word and the files, each beside its source load"


case("a still: the copy is its source with the stills and the names changed, and nothing else", a_still)
case("a text: the file and who wrote it", a_text)
case("a motion video: the named load's only", a_motion_video)
case("a setting of every load", a_setting)
case("a reference and the text that describes it are one change", a_reference_and_its_text)
case("the capture a copy is built on", the_capture_a_copy_is_built_on)
case("nothing is half written", nothing_is_half_written)
case("a difference outside the change is found", a_difference_outside_the_change)
case("where each render landed", where_each_render_landed)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
