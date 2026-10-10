#!/usr/bin/env python3
"""Build the graphs of a masked job from its job file, and queue them.

    <python> bench/masked_job.py build preview JOB.json          # writes data/<date>_<name>_job/, touches no server
    <python> bench/masked_job.py queue data/<date>_<name>_job preview [--wait]

**What it buys.** Every masked session until 2026-10-10 wrote its own script to turn "this clip, these people,
these shots" into graphs, each one patching a graph lifted out of an earlier render, and a fix made in one script
never reached the next. This is that script once, tracked, with a check (`bench/check_masked_job.py`). The graphs themselves are `workflows/masked_job_graph.py`'s; this file reads the
job, refuses what must be refused, writes what a run was made from, and queues it. The order a job goes in is the
masking board's `guide-order-of-operations-masked-job`.

**The job file** is the loads file `bench/capture_masked_run.py --loads` takes, with a `job` block beside
`defaults` and `loads` that the capture does not read (`masked_job_graph`'s docstring has its keys).

**build** makes a folder, `data/<date>_<name>_job/` (untracked, beside the capture `data/<date>_<name>` the
previews are read into), and writes there: `job.json`, the job file as given; the graph, `<stage>.json`; and
`<stage>.built.json`, which says which commit built it (and which of the builder's own files were not as that
commit has them), the tracker's mask version, and where the graph will save each file. It reads no server and uses no card. Only the `preview` stage exists so far: the first graph
of every job, who is where, nothing sampled (step 1 of the order). It ends by printing the capture command the
preview's files go into.

**queue** is the only step that uses the card, so it runs on the lead's word. It validates the graph against
what the running server serves (`build_workflows.validate_api`), posts it, and writes `<stage>.queued.json`;
with `--wait` it stays until the server is done and writes what was saved.

**What it refuses**, each with exit status `REFUSED`:

- a job folder that exists, or an output folder that exists: a new name for every run (the order's step 1 has
  what a reused preview folder nearly did on 2026-10-10);
- a `corrections` line whose `corrections_at` is not the tracker's mask version today. A correction names a
  person by the number on a tile, and the numbers are that version's;
- at `queue`: a graph that was queued before; an input the server does not serve; and node code in the tree
  that is not committed, unless `--override "WHO: WHY"` is given, which is written into the record. The server
  serves a commit, and a graph built beside uncommitted node code can describe a server nobody is running.

Not here yet, and coming in this order: the song node's plan per load, the render loads behind the capture's
`verify` and `preflight --gate`, and the one-window patch.

Run it from ComfyUI's environment, with `H3_OUTPUT_DIR` set as every bench tool that reads renders needs it.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402
import masked_job_graph as graphs  # noqa: E402

#: The exit status of a refusal, apart from the capture's own (`GATE_BLOCKED`, `VERIFY_FAILED`) and from a crash.
REFUSED = 5
STAGES = ("preview",)
#: Where node code is not: a change under one of these is not loaded by the server. `workflows/h3_config.py` is
#: the exception: node files load it by path at run time (`sol_observe.py`, `pdd_lora.py`).
NOT_NODE_CODE = ("bench/", "docs/", "internal/", "changelog.d/", "data/", "workflows/")
NODE_CODE_ALL_THE_SAME = ("workflows/h3_config.py",)
#: The files a graph is made by. A record that names a commit says so only when these are in it.
BUILDER = ("workflows/masked_job_graph.py", "workflows/h3_config.py", "bench/masked_job.py")
HOST = "127.0.0.1:8188"


class Refused(Exception):
    """Something the tool will not do, said in words the caller can act on."""


def tracker_mask_version(repo: Path = REPO) -> int:
    """`MiniMaxH3SubjectTrack.MASK_VERSION`, read from the source and not imported: the node file needs ComfyUI."""
    for node in ast.walk(ast.parse((repo / "subject_track.py").read_text())):
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3SubjectTrack":
            for line in node.body:
                if isinstance(line, ast.Assign) and any(getattr(t, "id", "") == "MASK_VERSION" for t in line.targets):
                    return int(ast.literal_eval(line.value))
    raise Refused("subject_track.py no longer sets MASK_VERSION on MiniMaxH3SubjectTrack; the corrections rule "
                  "has nothing to read")


def check_corrections(job: dict, version: int) -> None:
    """A correction typed by number is refused unless it says which tracker version its numbers were read at."""
    for subject in job["subjects"]:
        if not str(subject.get("corrections", "")).strip():
            continue
        at = subject.get("corrections_at")
        if at != version:
            raise Refused(
                f"subject {subject['label']!r} has corrections "
                + ("and no `corrections_at`" if at is None else f"read at the tracker's mask version {at}")
                + f"; the tracker is at {version}. A correction names a person by the number on a tile: read the "
                  f"numbers off a tile this version made, then set corrections_at to {version}.")


def job_folder(job: dict, date: str, data: Path) -> Path:
    return data / f"{date}_{job['name']}_job"


def output_folder(job: dict, output: Path) -> Path:
    return output / job["output"] / job["name"]


def head(repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def uncommitted_node_code(repo: Path = REPO) -> list[str]:
    """Tracked Python the server loads that differs from HEAD."""
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain",
                          "--untracked-files=no", "--", "*.py"],
                         capture_output=True, text=True, check=True).stdout
    paths = [line[3:].strip() for line in out.splitlines() if line.strip()]
    return [p for p in paths if p in NODE_CODE_ALL_THE_SAME or not p.startswith(NOT_NODE_CODE)]


def builder_not_committed(repo: Path = REPO) -> list[str]:
    """The builder's own files that are not as HEAD has them, untracked ones included."""
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--", *BUILDER],
                         capture_output=True, text=True, check=True).stdout
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def build(job_file: Path, stage: str, *, date: str, data: Path, output: Path, repo: Path = REPO) -> Path:
    """Write a stage's graph and its record into a new job folder; returns the folder."""
    given = json.loads(job_file.read_text())
    if "job" not in given:
        raise Refused(f"{job_file} has no `job` block")
    job = given["job"]
    try:
        graphs.check_job(job)
        graph = {"preview": graphs.preview}[stage](job)
    except ValueError as e:
        raise Refused(f"{job_file}: {e}") from e
    version = tracker_mask_version(repo)
    check_corrections(job, version)
    folder, saves_to = job_folder(job, date, data), output_folder(job, output)
    for path, what in ((folder, "job folder"), (saves_to, "output folder")):
        if path.exists():
            raise Refused(f"the {what} {path} exists. A run never reuses a name: give the job a new `name`.")
    folder.mkdir(parents=True)
    (folder / "job.json").write_bytes(job_file.read_bytes())
    (folder / f"{stage}.json").write_text(json.dumps(graph, indent=1) + "\n")
    record = {"stage": stage, "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "commit": head(repo), "builder_not_committed": builder_not_committed(repo),
              "uncommitted_node_code": uncommitted_node_code(repo),
              "tracker_mask_version": version, "job_file": str(job_file), "output_folder": str(saves_to),
              "saves": {s["label"]: graphs.prefixes(job, s["label"]) for s in job["subjects"]}}
    (folder / f"{stage}.built.json").write_text(json.dumps(record, indent=1) + "\n")
    return folder


def capture_command(job: dict, date: str) -> str:
    """The capture the preview's files go into, with a name left for each file the server numbers.

    The paths are under the server's output folder. What a subject holds (`held_things`) is not in it: the
    capture takes that mask as a subject of its own when a job needs it, never as a part."""
    lines = [f"bench/capture_masked_run.py files --name {job['name']} --date {date} --source {job['source']} "
             f"--first {job['first']} --frames {job['frames']}"]
    for subject in job["subjects"]:
        p = graphs.prefixes(job, subject["label"])
        pose = f"pose=<{p['pose']}_pose_table.json>," if subject.get("pose", True) else ""
        lines.append(f"    --mask {subject['label']}:track=<{p['track']}_*.mkv>,parts=<{p['parts']}_*.mkv>,"
                     f"classes=<{p['classes']}_*.mkv>,shots=<{p['shots']}*.json>,{pose}by={job['name']},"
                     f"at={job['first']}")
    return " \\\n".join(lines)


def call(host: str, path: str, data: dict | None = None) -> dict:
    request = urllib.request.Request(f"http://{host}{path}", data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise Refused(f"the server answered {e.code} to {path}: {e.read().decode()[:4000]}") from e
    except urllib.error.URLError as e:
        raise Refused(f"no server at {host}: {e.reason}") from e


def before_queue(folder: Path, stage: str, served: dict, *, override: str = "", repo: Path = REPO) -> tuple[dict, dict]:
    """Every refusal of `queue`, with nothing posted; returns the graph and the start of its record."""
    built_file, graph_file = folder / f"{stage}.built.json", folder / f"{stage}.json"
    if not built_file.exists() or not graph_file.exists():
        raise Refused(f"{folder} holds no built `{stage}`; run `build {stage}` first")
    if (folder / f"{stage}.queued.json").exists():
        raise Refused(f"`{stage}` of {folder.name} was queued before. A run never reuses a name: build the job "
                      f"again under a new `name`.")
    built, graph = json.loads(built_file.read_text()), json.loads(graph_file.read_text())
    if Path(built["output_folder"]).exists():
        raise Refused(f"the output folder {built['output_folder']} exists; something has written there since the build")
    dirty = uncommitted_node_code(repo)
    if dirty and ":" not in override:
        raise Refused("node code in the tree is not committed: " + ", ".join(dirty) + ". The server serves a "
                      'commit. Commit it, or say who queues beside it and why: --override "WHO: WHY".')
    import build_workflows
    errors = build_workflows.validate_api(graph, served, f"{folder.name}/{stage}")
    if errors:
        raise Refused("the running server does not accept this graph:\n  " + "\n  ".join(errors))
    return graph, {"stage": stage, "commit": head(repo), "uncommitted_node_code": dirty, "override": override}


def queue(folder: Path, stage: str, *, host: str, override: str = "", wait: bool = False) -> int:
    graph, record = before_queue(folder, stage, call(host, "/object_info"), override=override)
    answer = call(host, "/prompt", {"prompt": graph, "client_id": "masked_job"})
    record |= {"prompt_id": answer.get("prompt_id"), "host": host,
               "queued_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    queued = folder / f"{stage}.queued.json"
    queued.write_text(json.dumps(record, indent=1) + "\n")
    print(f"queued {record['prompt_id']}  ({queued})")
    if not wait or not record["prompt_id"]:
        return 0
    started = time.time()
    while True:
        history = call(host, f"/history/{record['prompt_id']}")
        if record["prompt_id"] in history:
            break
        time.sleep(3)
    entry = history[record["prompt_id"]]
    status = entry.get("status", {})
    record |= {"seconds": round(time.time() - started), "status": status.get("status_str"), "files": [], "said": {}}
    for kind, detail in status.get("messages", []):
        if kind in ("execution_error", "execution_interrupted"):
            record["error"] = {k: str(detail.get(k))[:3000] for k in ("node_type", "exception_type", "exception_message")}
    for nid, out in entry.get("outputs", {}).items():
        for value in out.values():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict) and "filename" in item:
                    record["files"].append(f"{item.get('subfolder', '')}/{item['filename']}")
                elif isinstance(item, str):
                    record["said"].setdefault(nid, []).append(item)
    queued.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['status']} in {record['seconds']} s; {len(record['files'])} files; the record is {queued}")
    if "error" in record:
        print(json.dumps(record["error"], indent=1))
    return 0 if record["status"] == "success" else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write a stage's graph into a new job folder; no server, no card")
    b.add_argument("stage", choices=STAGES)
    b.add_argument("job_file", type=Path)
    b.add_argument("--date", default=datetime.date.today().isoformat())
    q = sub.add_parser("queue", help="post a built stage to the server: the one step that uses the card")
    q.add_argument("folder", type=Path)
    q.add_argument("stage", choices=STAGES)
    q.add_argument("--host", default=HOST)
    q.add_argument("--wait", action="store_true", help="stay until the server is done and record what it saved")
    q.add_argument("--override", default="", metavar='"WHO: WHY"',
                   help="queue beside uncommitted node code; written into the record")
    a = p.parse_args()
    try:
        if a.command == "build":
            folder = build(a.job_file, a.stage, date=a.date, data=REPO / "data", output=h3_config.output_dir())
            built = json.loads((folder / f"{a.stage}.built.json").read_text())
            print(f"{folder / (a.stage + '.json')}\n  built at {built['commit']}, tracker mask version "
                  f"{built['tracker_mask_version']}; it will save under {built['output_folder']}")
            if built["builder_not_committed"]:
                print("  the builder itself is not as that commit has it: " + ", ".join(built["builder_not_committed"]))
            if built["uncommitted_node_code"]:
                print("  node code not committed (queue will ask for an override): "
                      + ", ".join(built["uncommitted_node_code"]))
            print("when it has run, the capture:\n" + capture_command(json.loads(a.job_file.read_text())["job"], a.date))
            return 0
        return queue(a.folder, a.stage, host=a.host, override=a.override, wait=a.wait)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    sys.exit(main())
