#!/usr/bin/env python3
"""The same loads again with one thing changed: copies of masked job files, a record of the sweep, the commands in order.

    <python> bench/sweep_loads.py make JOB.json [JOB.json ...] --tag stills_b --by <session> --still lead=lead_b.jpg [--still second=mom_b.jpg]
    <python> bench/sweep_loads.py make JOB.json ... --tag text_c --by <session> --text t1=words_c.txt --text-by "<who wrote it, from which frames>"
    <python> bench/sweep_loads.py make JOB.json ... --tag crop --by <session> --motion-file second_hf_1039_e=/share/.../reference.mp4
    <python> bench/sweep_loads.py make JOB.json ... --tag seed2 --by <session> --set seed=730451893
    <python> bench/sweep_loads.py landed data/<date>_sweep_<tag>/sweep.json

**What it buys.** Once a shot's windows, masks and reference are settled, the next question is the same loads
with one input changed: other stills, another text, another reference video, another seed. Every load is
already a job file that names its still, capture, motion video, seed and settings by path
(`bench/masked_job.py`), so the repeat is a copy of that file with one key changed and new names. Done by hand,
a copy picks up a second difference nobody meant, and a render is then read as the effect of the first. This
writes the copies, proves each differs from its source in the named change and the names only, records the
sweep, and prints what to run.

**It queues nothing and builds nothing.** The queue is the card holder's, and a look sits between a build and a
queue. `make` reads job files and writes job files; `landed` reads the records `bench/masked_job.py` leaves.
No server, no card.

**One change a sweep**, of one kind:

- `--still SUBJECT=FILE` (repeatable): that subject's still in the job block, and the still of any load of that
  subject that names its own. Several subjects' stills in one sweep are one change: the stills.
- `--text LABEL=FILE` (repeatable) with `--text-by WHO`: the file a text label points at, and who wrote it. The
  file is beside the job file, as a job's texts are, and must exist. The frames the text is tied to do not
  change: a text for other frames is another job.
- `--motion-file LOAD=FILE` (repeatable): that load's motion video. The file must exist and must begin on the
  source frame the one it replaces begins on (`motion_first` is not changed).
- `--set KEY=VALUE`: one setting of every load (`workflows/masked_job_graph.py::SETTINGS`: the seed, the margin,
  the window, a short edge), written into each load, where it stands over the file's `defaults`.

Every job file given must be changed by it, and every subject, label or load the change names must be in one of
them; otherwise nothing is written. Nothing is derived: a copy carries its capture, its masks, its reference
video and its texts as the paths its source carries, word for word.

**What else differs, and only this:** the job's `name` and every load's `name` take the tag on the end, because
a run never reuses a name (`bench/masked_job.py`: the job folder, the output folder and the capture's runs are
all named by them), and the job's `note` says the file is a sweep's copy, of what, by whom, and keeps the
source's note under it. `differences` lists every place two job files differ, and a copy with a difference
outside those is not written.

**The copies sit beside their sources**, as `<source stem>__<tag>.json`, so a text file named beside the job
file is still beside it. The record goes under `data/<date>_sweep_<tag>/sweep.json` (untracked): the change,
each source and its copy, each load's old and new name and whether the change touches it, every place the two
files differ with both values, and the commands.

**The commands** are `bench/masked_job.py`'s own, for the loads the change touches, in the order a load goes:
build the plans, queue each plan, then, between plan and render, what every load needs and this tool does not
print because it is the capture's own and differs by job (the capture reads the plans, `verify`, the gate, the
look), then build the renders and queue each. A load the change does not touch is renamed like the rest and is
not in the commands: rendering it again would be a repeat, which is another test.

**`landed`** reads the record and, for each copy, the job folders `bench/masked_job.py` made for it
(`data/*_<job name>_job/`): for each touched load, whether its render was built and queued, what the server
said, and where the files are. It writes that into the record and prints it, with the source load's name
beside each so a pair can be laid side by side. It also reads each copy again against its source: a copy edited
since the sweep is said, with what now differs.

WHAT IT CANNOT KNOW, and says:
- Whether a still is in the server's input folder. `bench/masked_job.py build render` tests that.
- Whether a new motion video shows the same frames as the one it replaces. It is a path; open it.
- Whether a load that continues another run (`continue_from`) should continue the source's window or the copy's.
  The copy keeps the path it was given, and the tool says so for each such load.
- Anything about the pictures. Two renders that differ in one input are still one seed and one shot each.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import masked_job_graph as graphs  # noqa: E402

KINDS = ("still", "text", "motion_file", "set")
#: A tag is part of file, folder and run names. Reasoned: what those take without quoting.
TAG = re.compile(r"[a-z0-9][a-z0-9_]*\Z")


#: The exit status of a refusal. Inherited: `bench/masked_job.py::REFUSED`, so the two tools refuse alike.
REFUSED = 5


class Refused(Exception):
    """Nothing was written, and this says why."""


def differences(a, b, at: str = "") -> list[tuple[str, object, object]]:
    """Every place two job files differ, as (where, the first's value, the second's). A list is compared place by
    place; a place one side lacks is given as None with the word `absent` in the path."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in list(a) + [k for k in b if k not in a]:
            where = f"{at}.{key}" if at else str(key)
            if key not in a or key not in b:
                out.append((where + " (absent on one side)", a.get(key), b.get(key)))
            else:
                out += differences(a[key], b[key], where)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(at + " (lists of different lengths)", len(a), len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{at}[{i}]")]
    return [] if a == b and type(a) is type(b) else [(at, a, b)]


def _pairs(given: list[str], what: str) -> dict[str, str]:
    out = {}
    for text in given:
        name, sep, value = text.partition("=")
        if not sep or not name or not value:
            raise Refused(f"{what} takes NAME=VALUE; got {text!r}")
        if name in out:
            raise Refused(f"{what} names {name!r} twice")
        out[name] = value
    return out


def change_of(still: list[str], text: list[str], text_by: str, motion_file: list[str], setting: str) -> dict:
    """The one change of a sweep, from the command line: {"kind", "values", and for a text "by"}."""
    given = [k for k, v in zip(KINDS, (still, text, motion_file, setting)) if v]
    if len(given) != 1:
        raise Refused("a sweep changes one thing: give one of --still, --text, --motion-file, --set"
                      + (f" (given: {', '.join('--' + g.replace('_', '-') for g in given)})" if given else ""))
    kind = given[0]
    if kind == "set":
        values = _pairs([setting], "--set")
        key = next(iter(values))
        if key not in graphs.SETTINGS:
            raise Refused(f"--set {key}: not a setting of a load; one of {', '.join(graphs.SETTINGS)}")
        try:
            values[key] = json.loads(values[key])                   # a number stays a number, as a job file has it
        except ValueError:
            pass
        return {"kind": kind, "values": values}
    values = _pairs({"still": still, "text": text, "motion_file": motion_file}[kind], "--" + kind.replace("_", "-"))
    if kind == "text":
        if not text_by.strip():
            raise Refused("--text needs --text-by: who wrote the new text, and from which frames")
        return {"kind": kind, "values": values, "by": text_by.strip()}
    if text_by:
        raise Refused("--text-by goes with --text")
    return {"kind": kind, "values": values}


def in_words(change: dict) -> str:
    values = change["values"]
    return {"still": "the still of " + ", ".join(f"{k} is {v}" for k, v in values.items()),
            "text": "the text " + ", ".join(f"{k} is {v}" for k, v in values.items()) + f", written by {change.get('by', '')}",
            "motion_file": "the motion video of " + ", ".join(f"{k} is {v}" for k, v in values.items()),
            "set": "every load's " + ", ".join(f"{k} is {v!r}" for k, v in values.items())}[change["kind"]]


def copy_of(source: dict, change: dict, tag: str, folder: Path, note: str) -> tuple[dict, list[str], list[str], set[str]]:
    """(the copy, the places the change was made, the names of the loads it touches, the names the change used).

    `folder` is where the source job file is: a text file is named beside it."""
    out = copy.deepcopy(source)
    job, loads, values = out["job"], out.get("loads", []), change["values"]
    made, touched, used = [], [], set()
    if change["kind"] == "still":
        for i, subject in enumerate(job.get("subjects", [])):
            if subject.get("label") in values and subject.get("still") != values[subject["label"]]:
                subject["still"] = values[subject["label"]]
                made.append(f"job.subjects[{i}].still")
                used.add(subject["label"])
        for i, one in enumerate(loads):
            if one.get("subject") in values:
                if "still" in one and one["still"] != values[one["subject"]]:
                    one["still"] = values[one["subject"]]
                    made.append(f"loads[{i}].still")
                    used.add(one["subject"])
                if one["subject"] in used:
                    touched.append(one["name"])
    elif change["kind"] == "text":
        for label, name in values.items():
            entry = (job.get("texts") or {}).get(label)
            if entry is None:
                continue
            if not (folder / name).is_file():
                raise Refused(f"the text {name} is not beside the job file, in {folder}: a job's texts are files beside it")
            if entry.get("file") != name:
                entry["file"] = name
                made.append(f"job.texts.{label}.file")
                if entry.get("by") != change["by"]:
                    entry["by"] = change["by"]
                    made.append(f"job.texts.{label}.by")
                used.add(label)
        defaults = source.get("defaults", {})
        touched = [one["name"] for one in loads if one.get("text", defaults.get("text")) in used]
    elif change["kind"] == "motion_file":
        for i, one in enumerate(loads):
            if one.get("name") in values:
                if not Path(values[one["name"]]).is_file():
                    raise Refused(f"the motion video {values[one['name']]} for load {one['name']!r} is not a file")
                if one.get("motion_file") != values[one["name"]]:
                    one["motion_file"] = values[one["name"]]
                    made.append(f"loads[{i}].motion_file" + ("" if "motion_file" in source["loads"][i] else " (absent on one side)"))
                    used.add(one["name"])
                    touched.append(one["name"])
    else:
        (key, value), = values.items()
        defaults = source.get("defaults", {})
        for i, one in enumerate(loads):
            if one.get(key, defaults.get(key)) != value or type(one.get(key, defaults.get(key))) is not type(value):
                made.append(f"loads[{i}].{key}" + ("" if key in one else " (absent on one side)"))
                one[key] = value
                touched.append(one["name"])
        if made:
            used.add(key)
    for one in loads:
        one["name"] = f"{one['name']}_{tag}"
    job["name"] = f"{job['name']}_{tag}"
    job["note"] = note + (f" The source's own note follows and is of the source: {source['job']['note']}" if source["job"].get("note") else "")
    return out, made, [f"{name}_{tag}" for name in touched], used


def allowed(source: dict, made: list[str]) -> set[str]:
    """Where a copy may differ from its source: the names, the note, and the places the change was made."""
    names = {"job.name", "job.note" if "note" in source["job"] else "job.note (absent on one side)"}
    return names | {f"loads[{i}].name" for i in range(len(source.get("loads", [])))} | set(made)


def beyond(source: dict, made: list[str], copied: dict) -> list[tuple[str, object, object]]:
    """The places a copy differs from its source that the sweep does not account for. Empty for a copy as written."""
    may = allowed(source, made)
    return [d for d in differences(source, copied) if d[0] not in may]


def commands(copies: list[dict]) -> list[str]:
    """`bench/masked_job.py`'s commands for the touched loads of every copy, in the order a load goes."""
    out = []
    for c in copies:
        if not c["touched"]:
            continue
        loads = " ".join(f"--load {name}" for name in c["touched"])
        folder = f"data/<the date it is built>_{c['job']['to']}_job"
        out.append(f"<python> bench/masked_job.py build plan {c['copy']} {loads}")
        out += [f"<python> bench/masked_job.py queue {folder} plan --load {name} --wait" for name in c["touched"]]
        out.append(f"# between plan and render, for {c['job']['to']}: the capture {c['capture']} reads the plans; verify; the gate; the look "
                   "(the capture's own commands, as for any load)")
        out.append(f"<python> bench/masked_job.py build render {c['copy']} {loads}")
        out += [f"<python> bench/masked_job.py queue {folder} render --load {name} --wait" for name in c["touched"]]
    return out


def make(job_files: list[Path], change: dict, tag: str, by: str, *, data: Path = REPO / "data", date: str | None = None, say=print) -> Path:
    """Write the copies and the sweep's record; returns the record's path. Nothing is written unless all of it can be."""
    if not TAG.match(tag):
        raise Refused(f"--tag {tag!r}: lower-case letters, digits and _ only; it becomes part of file, folder and run names")
    if not by.strip():
        raise Refused("--by: say who is making the sweep")
    if len({p.resolve() for p in job_files}) != len(job_files):
        raise Refused("a job file is given twice")
    now = datetime.datetime.now(datetime.timezone.utc)
    record_path = data / f"{date or now.date().isoformat()}_sweep_{tag}" / "sweep.json"
    if record_path.exists():
        raise Refused(f"{record_path} exists: a sweep never reuses a tag")
    planned, used = [], set()
    for path in job_files:
        if not path.is_file():
            raise Refused(f"{path} is not a file")
        source = json.loads(path.read_text())
        if "job" not in source or "name" not in source["job"] or not source.get("loads"):
            raise Refused(f"{path} is not a masked job file: it needs a `job` block with a `name`, and `loads`")
        target = path.with_name(f"{path.stem}__{tag}.json")
        if target.exists():
            raise Refused(f"{target} exists: a sweep never reuses a tag")
        note = (f"SWEEP {tag}: a copy of the job {source['job']['name']} ({path.name}) with one change, {in_words(change)}; its name and every "
                f"load's name take _{tag}, and nothing else differs. Made by {by.strip()} at {now.strftime('%Y-%m-%dT%H:%M:%SZ')} with "
                f"bench/sweep_loads.py; the sweep's record is {record_path}.")
        copied, made, touched, names = copy_of(source, change, tag, path.parent, note)
        if not made:
            raise Refused(f"the change touches nothing in {path}: {in_words(change)}. A copy with no change is a repeat, which is another test")
        extra = beyond(source, made, copied)
        if extra:                                                    # the tool's own proof, before a file exists
            raise Refused(f"the copy of {path} would differ from it outside the change: {[d[0] for d in extra]}")
        used |= names
        continues = [one["name"] for one in copied["loads"] if one.get("continue_from")]
        planned.append((path, target, source, copied, made, touched, continues))
    asked = set(change["values"])
    if asked - used:
        raise Refused(f"the change names {', '.join(sorted(asked - used))}, which no job file given has (or already has at that value)")
    copies = []
    for path, target, source, copied, made, touched, continues in planned:
        target.write_text(json.dumps(copied, indent=1) + "\n")
        differs = [{"where": where, "source": a, "copy": b} for where, a, b in differences(source, copied)]
        copies.append({"source": str(path), "copy": str(target), "job": {"from": source["job"]["name"], "to": copied["job"]["name"]},
                       "capture": copied["job"].get("capture"),
                       "loads": [{"from": old["name"], "to": new["name"], "subject": new.get("subject"), "touched": new["name"] in touched}
                                 for old, new in zip(source["loads"], copied["loads"])],
                       "touched": touched, "changed_at": made, "differs_at": differs,
                       "continues_another_run": [f"{name} keeps `continue_from` as its source has it: it continues the SAME stored window"
                                                 for name in continues]})
    record = {"sweep": tag, "made_by": by.strip(), "made_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "change": change, "change_in_words": in_words(change),
              "what_else_differs": "the job's name and note, and every load's name; nothing more (`differs_at` lists every place, with both values)",
              "copies": copies, "commands": commands(copies),
              "not_known": ["whether a still is in the server's input folder: `bench/masked_job.py build render` tests it",
                            "whether a new motion video or text is right for the frames: they are paths here",
                            "anything about the pictures: each render is still one seed on one shot"],
              "landed": "not read yet: run `bench/sweep_loads.py landed` on this record once renders are queued"}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    say(f"sweep {tag}: {in_words(change)}")
    for c in copies:
        say(f"  {c['job']['from']} -> {c['job']['to']}: {c['copy']}")
        say(f"    changed at {', '.join(c['changed_at'])}; loads the change touches: {', '.join(c['touched']) or 'none'}"
            + (f"; renamed and not touched: {', '.join(x['to'] for x in c['loads'] if not x['touched'])}" if len(c["touched"]) < len(c["loads"]) else ""))
        for line in c["continues_another_run"]:
            say(f"    {line}")
    say(f"the record: {record_path}")
    say("to run, in order (nothing here is queued or built by this tool):")
    for line in record["commands"]:
        say("  " + line)
    return record_path


def landed(record_path: Path, *, data: Path = REPO / "data", say=print) -> dict:
    """Read where each touched load's render landed, from the records `bench/masked_job.py` left, into the sweep's record."""
    record = json.loads(record_path.read_text())
    rows, edited = [], []
    for c in record["copies"]:
        source, target = Path(c["source"]), Path(c["copy"])
        if not source.is_file() or not target.is_file():
            edited.append(f"{target if source.is_file() else source} is gone: the copy cannot be held to its source")
        else:
            extra = beyond(json.loads(source.read_text()), c["changed_at"], json.loads(target.read_text()))
            if extra:
                edited.append(f"{target} no longer differs from {source.name} in the change alone: also at " + ", ".join(d[0] for d in extra))
        folders = sorted(data.glob(f"*_{c['job']['to']}_job"))
        for one in c["loads"]:
            if not one["touched"]:
                continue
            row = {"load": one["to"], "source_load": one["from"], "job": c["job"]["to"], "render": "not built"}
            for folder in folders:
                built, queued = folder / f"{one['to']}.render.built.json", folder / f"{one['to']}.render.queued.json"
                if built.is_file():
                    was = json.loads(built.read_text())
                    row.update(render="built, not queued", job_folder=str(folder), built_at=was.get("built_at"), commit=was.get("commit"),
                               output_folder=was.get("output_folder"))
                if queued.is_file():
                    was = json.loads(queued.read_text())
                    row.update(render=f"queued: {was.get('status', 'no status yet')}", files=was.get("files", []))
            rows.append(row)
    record["landed"] = {"read_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "loads": rows,
                        "copies_edited_since": edited}
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    say(f"sweep {record['sweep']}: {record['change_in_words']}")
    for row in rows:
        say(f"  {row['load']} (the source's load: {row['source_load']}): {row['render']}"
            + (f"; {row['output_folder']}" if row.get("output_folder") else "")
            + (f"; {len(row['files'])} file(s)" if "files" in row else ""))
        for name in row.get("files", []):
            say(f"      {name}")
    for line in edited:
        say("  EDITED SINCE THE SWEEP: " + line)
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make", help="write the copies, the record and the commands; builds and queues nothing")
    m.add_argument("job_files", type=Path, nargs="+")
    m.add_argument("--tag", required=True, help="the sweep's name: goes on the end of every job and load name")
    m.add_argument("--by", required=True, help="who is making it")
    m.add_argument("--still", action="append", default=[], metavar="SUBJECT=FILE")
    m.add_argument("--text", action="append", default=[], metavar="LABEL=FILE")
    m.add_argument("--text-by", default="", help="with --text: who wrote the new text, and from which frames")
    m.add_argument("--motion-file", action="append", default=[], metavar="LOAD=FILE")
    m.add_argument("--set", default="", metavar="KEY=VALUE", dest="setting", help="one setting of every load")
    d = sub.add_parser("landed", help="read where each render landed into the sweep's record")
    d.add_argument("record", type=Path)
    a = p.parse_args()
    try:
        if a.command == "make":
            make(a.job_files, change_of(a.still, a.text, a.text_by, a.motion_file, a.setting), a.tag, a.by)
        else:
            landed(a.record)
    except Refused as why:
        print(f"REFUSED: {why}")
        return REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
