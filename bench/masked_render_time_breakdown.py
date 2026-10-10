#!/usr/bin/env python3
"""Where a masked song render's seconds go: by node, and by stage inside the song node.

Three sources, joined on the prompt id of one render:

  the runner's row      `bench/run_graph_arms.py`'s JSONL: seconds per node from the
                        websocket feed, the total, the patches, the substrate.
  the song node's report  `MiniMaxH3AudioFreezeSong` times its own stages
                        (`audio_freeze_song.py`, `mark`): per window "[N] seconds: ..."
                        and at the end "seconds by stage, T in all: ...". Read from a
                        saved `/history` outputs file, one per label.
  the telemetry record  `pipeline_telemetry.py`'s JSONL, optional: which models core
                        loaded while the song node ran, and when.
  the server's log      optional: the sampler's progress bar, for the seconds each step
                        took, and Sol-Attn's first sparse line, for how many steps ran
                        dense before it and how many rows the sequence has.

    <python> bench/masked_render_time_breakdown.py \
        --rows RUN.jsonl --history-dir DIR [--telemetry-dir DIR] [--server-log LOG] \
        [--label A --label B] [--state A="first prompt after a server start"] \
        --out data/results/<date>_<name>.json

Prints the tables as Markdown and writes the same numbers to `--out`. Pure
reading: no server, no GPU, no ComfyUI import.

What it does not measure. `REPEAT_KEEPS` and `SONG_ONLY_ON_REPEAT` say what a
second run of the same stretch would pay again; they are read from the code
(the song node's inputs, core's node cache), not timed, and the output labels
every figure built on them `by rule`. A load line is placed in a stage by
laying the stages end to end from the node's start, so it is off by at most
the node's unaccounted seconds, which are printed beside it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import telemetry_report

REPO = Path(__file__).resolve().parent.parent
SONG_CLASS = "MiniMaxH3AudioFreezeSong"

#: The song node's stages in the order one run meets them (`audio_freeze_song.py::execute`).
BEFORE_WINDOWS = ("track encode", "motion reference", "conditioning")
PER_WINDOW = ("source encode", "window setup", "sampling", "decode", "composite", "write")
AFTER_WINDOWS = ("join and mux",)

#: Stages whose result depends on neither the seed nor the schedule, with what it does depend
#: on. Reasoned from the node's code, not measured: a second run of the same stretch with only
#: the seed or the step count changed computes these again today and would get the same answer.
REPEAT_KEEPS = {
    "track encode": "the track and the audio level",
    "motion reference": "the window's source frames and mask, and the Masked Source's motion settings",
    "conditioning": "each window's text, the still and, with a motion reference, the window's source frames",
    "source encode": "the window's source frames at the canvas, and the mask only under paint_out or a late start",
}
#: On a second queue in one server session with only the song node's inputs changed, core
#: serves every other node from its cache. Reasoned from core's cache key (a node's inputs and
#: its ancestors'); the telemetry's `not_observed` list is the observable when such a run exists.
SONG_ONLY_ON_REPEAT = "by rule: the song node and the preview of its report run again; every node upstream is served from core's cache"

#: Patches whose values the output carries: the ones a render's seconds depend on. Every other
#: patch is listed by its field alone, so a record of where the time went holds no text that
#: says what a clip or a still shows (the subject's words, a correction, a still's file name).
PATCH_VALUES_KEPT = ("VHS_LoadVideoFFmpeg", "MiniMaxH3Resolution", "BasicScheduler", SONG_CLASS)

LAP = re.compile(r"^\[(\d+)\] seconds: (.+)$")
#: a render after the node's stage timing gained a per-window conditioning line
COND_LAP = re.compile(r"^\[(\d+)\] conditioning seconds: (.+)$")
TOTAL = re.compile(r"^seconds by stage, (\d+) in all: (.+)$")
LIVE = re.compile(r"^\[(\d+)\] source kept outside the mask: ([\d.]+)% of the window's video tokens regenerate")
UNSAMPLED = re.compile(r"^\[(\d+)\] nothing masked in this window")
WINDOW = re.compile(r"^\[(\d+)\] \S+-\S+, (\d+) frames.*, (reused|renders)$")


ANSI = re.compile(r"\x1b\[[0-9;]*m")
BAR = re.compile(r"^\s*\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<")
SPARSE = re.compile(r"\[h3-sol\] sparse \((\d+), (\d+), (\d+), (\d+)\)")


def _clock(text: str) -> int:
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def sampling_steps(log_text: str, prefix: str, steps: int) -> dict | None:
    """Seconds per sampling step for the render whose song node wrote `prefix`, from the server's log.

    The progress bar prints the elapsed time at each step to the second, so one step is right
    to a second and a mean over several is finer. Sol-Attn logs its first sparse call once per
    sequence shape in a server's life: where that line falls in a bar says how many steps ran
    dense before it, and it is absent from a render whose shape an earlier render already logged.
    """
    lines = ANSI.sub("", log_text).replace("\r", "\n").split("\n")
    end = next((i for i in range(len(lines) - 1, -1, -1)
                if f"{SONG_CLASS}:" in lines[i] and prefix in lines[i]), None)
    if end is None:
        return None
    start = next((i for i in range(end, -1, -1) if "got prompt" in lines[i]), 0)
    bars, bar, dense, rows = [], None, None, None
    for line in lines[start:end]:
        if m := BAR.match(line):
            k, n, at = int(m[1]), int(m[2]), _clock(m[3])
            if n != steps:
                continue
            if k == 0:
                if bar is None or bar:
                    bar = {}
                    bars.append(bar)
            elif bar is not None:
                bar.setdefault(k, at)       # the first print of a step; later ones are refreshes
        elif (m := SPARSE.search(line)) and bar is not None and dense is None:
            dense, rows = max(bar, default=0), int(m[2])
    windows = []
    for bar in (b for b in bars if b):
        at = [0] + [bar[k] for k in sorted(bar)]
        took = [b - a for a, b in zip(at, at[1:])]
        entry = {"steps_s": took, "sampler_s": at[-1]}
        if dense is not None and len(took) > dense:
            entry["dense_step_mean_s"] = sum(took[:dense]) / dense if dense else None
            entry["sol_step_mean_s"] = sum(took[dense:]) / (len(took) - dense)
        windows.append(entry)
    return {"steps": steps, "dense_steps_before_sol": dense, "sequence_rows": rows,
            "resolution": "elapsed seconds as the progress bar prints them, whole seconds",
            "windows": windows}


def _pairs(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(", "):
        name, _, value = part.rpartition(" ")
        out[name] = float(value)
    return out


def parse_report(text: str) -> dict:
    """The song node's own timing and what it says of each window, from its report text."""
    laps, cond_laps, live, frames, reused, unsampled, total = {}, {}, {}, {}, [], [], None
    for line in text.splitlines():
        line = line.strip()
        if m := LAP.match(line):
            laps[int(m[1])] = _pairs(m[2])
        elif m := COND_LAP.match(line):
            cond_laps[int(m[1])] = _pairs(m[2])
        elif m := TOTAL.match(line):
            total = _pairs(m[2])
        elif m := LIVE.match(line):
            live[int(m[1])] = float(m[2])
        elif m := UNSAMPLED.match(line):
            unsampled.append(int(m[1]))
        elif m := WINDOW.match(line):
            frames[int(m[1])] = int(m[2])
            if m[3] == "reused":
                reused.append(int(m[1]))
    if total is None:
        raise ValueError("no output text carries a `seconds by stage` line")
    return {"stages": total, "windows": laps, "conditioning_windows": cond_laps,
            "regenerated_percent": live, "window_frames": frames,
            "reused_windows": reused, "unsampled_windows": unsampled}


def find_report(outputs: dict) -> str:
    """The report among a saved `/history` outputs map: the one text that carries the stage line."""
    for node_out in (outputs or {}).values():
        for value in (node_out or {}).get("text") or []:
            if isinstance(value, str) and "seconds by stage" in value:
                return value
    raise ValueError("no output text carries a `seconds by stage` line")


def find_telemetry(directory: Path, prompt_id: str) -> Path | None:
    """`<date>_<time>_<prompt id prefix>.jsonl`: the file whose last name part starts the prompt id."""
    for path in sorted(directory.glob("*.jsonl")):
        prefix = path.stem.rsplit("_", 1)[-1]
        if prefix and prompt_id.replace("-", "").startswith(prefix.replace("-", "")):
            return path
    return None


def timeline(parsed: dict) -> list[tuple[str, int | None, float, float]]:
    """(stage, window, start, end) laid end to end from the node's start."""
    out, t = [], 0.0
    by_window = parsed.get("conditioning_windows") or {}
    for stage in BEFORE_WINDOWS:
        if stage not in parsed["stages"] or (by_window and stage != BEFORE_WINDOWS[0]):
            continue
        out.append((stage, None, t, t + parsed["stages"][stage]))
        t = out[-1][3]
    # with per-window lines the conditioning loop is laid window by window, as it ran
    for number in sorted(by_window):
        for stage, took in by_window[number].items():
            out.append((stage, number, t, t + took))
            t = out[-1][3]
    for number in sorted(parsed["windows"]):
        for stage, took in parsed["windows"][number].items():
            out.append((stage, number, t, t + took))
            t = out[-1][3]
    for stage in AFTER_WINDOWS:
        if stage in parsed["stages"]:
            out.append((stage, None, t, t + parsed["stages"][stage]))
            t = out[-1][3]
    return out


def loads_in_song(rows: list[dict], song_id: str, parsed: dict) -> list[dict]:
    """Core's load lines while the song node ran, each with the stage the end-to-end timeline puts it in."""
    start = None
    for r in rows:
        if r["type"] == "node_start" and r["node"] == song_id:
            start = r["t"]          # the last start before the end is the run (`docs/pipeline_telemetry.md`)
    if start is None:
        return []
    line = timeline(parsed)
    out = []
    for r in rows:
        if r["type"] != "log" or r.get("node") != song_id:
            continue
        at = r["t"] - start
        where = next(((s, w) for s, w, a, b in line if a <= at < b), (None, None))
        out.append({"at_s": round(at, 1), "kind": r.get("kind"), "what": (r.get("groups") or [None])[0],
                    "stage": where[0], "window": where[1]})
    return out


def timing_patches(patches) -> list[dict]:
    out = []
    for patch in patches or []:
        kept = str(patch.get("field", "")).split(".", 1)[0] in PATCH_VALUES_KEPT
        out.append(patch if kept else {"nodes": patch.get("nodes"), "field": patch.get("field"),
                                       "value": "<not carried: does not bear on time>"})
    return out


def patched(graph: dict, patches) -> dict:
    """The graph with the row's patches applied, for reading a value a patch may have set."""
    for patch in patches or []:
        names = str(patch.get("field", "")).split(".", 1)
        for nid in patch.get("nodes") or []:
            if len(names) == 2 and nid in graph:
                graph[nid]["inputs"][names[1]] = patch.get("value")
    return graph


def stage_load(rows: list[dict], song_id: str, parsed: dict) -> list[dict]:
    """What the card and the host's processors did during each stage, from the telemetry's samples.

    Mean GPU utilisation over the samples inside the stage, and the server process's CPU seconds
    across it, so a stage that waits on the processor reads apart from one that waits on the card.
    None where a stage holds fewer than two samples.
    """
    start = None
    for r in rows:
        if r["type"] == "node_start" and r["node"] == song_id:
            start = r["t"]
    if start is None:
        return []
    samples = [r for r in rows if r["type"] == "sample"]
    out = []
    for stage, window, a, b in timeline(parsed):
        inside = [x for x in samples if a <= x["t"] - start < b]
        util = [x["gpu"].get("util_gpu") for x in inside if (x.get("gpu") or {}).get("util_gpu") is not None]
        cpu = [x["host"]["proc_cpu_user_s"] + x["host"]["proc_cpu_sys_s"] for x in inside
               if (x.get("host") or {}).get("proc_cpu_user_s") is not None
               and x["host"].get("proc_cpu_sys_s") is not None]
        two = len(inside) >= 2
        out.append({"stage": stage, "window": window, "seconds": round(b - a, 1), "samples": len(inside),
                    "gpu_util_mean": round(sum(util) / len(util), 1) if two and util else None,
                    "cpu_s": round(cpu[-1] - cpu[0], 1) if two and len(cpu) >= 2 else None})
    return out


def breakdown(row: dict, outputs: dict, telemetry: Path | None, state: str | None,
              server_logs: tuple[str, ...] = ()) -> dict:
    graph = patched(json.loads((REPO / row["graph"]).read_text()), row.get("patches"))
    classes = {nid: node["class_type"] for nid, node in graph.items()}
    total = float(row["total_s"])
    per_node = {str(k): float(v) for k, v in (row.get("per_node_s") or {}).items()}
    nodes = [{"node": nid, "class_type": classes.get(nid, "?"), "seconds": s, "share": s / total}
             for nid, s in sorted(per_node.items(), key=lambda kv: -kv[1])]
    head = {"label": row.get("label") or row.get("arm"), "prompt_id": row.get("prompt_id"),
            "graph": row["graph"], "graph_sha256": row.get("graph_sha256"),
            "patches": timing_patches(row.get("patches")), "substrate": row.get("substrate"),
            "cache_state": state, "total_s": total, "nodes": nodes,
            "outside_any_node_s": total - sum(per_node.values())}
    song_id = next(nid for nid, c in classes.items() if c == SONG_CLASS)
    steps = (graph.get(str((graph[song_id]["inputs"].get("sigmas") or [None])[0]), {}).get("inputs") or {}).get("steps")
    # a preview shares its render's filename prefix and samples nothing: its row has no bars
    for log_text in (() if graph[song_id]["inputs"].get("preview") is True else server_logs):
        found = sampling_steps(log_text, str(graph[song_id]["inputs"]["filename_prefix"]), steps) \
            if isinstance(steps, int) else None
        if found and found["windows"]:
            head["sampling_steps"] = found
            break
    try:
        parsed = parse_report(find_report(outputs))
    except ValueError as exc:
        # a preview samples nothing and a render before bbe80949 did not time its stages:
        # the seconds by node still stand
        head["song_node"] = None
        head["no_stages"] = str(exc)
        if telemetry is not None:
            summary = telemetry_report.phases(telemetry_report.load(telemetry))
            head["telemetry"] = {"record": telemetry.name, "total_s": summary["total_s"],
                                 "not_observed": [f"{nid} {classes.get(nid, '?')}" for nid in summary["not_observed"]]}
        return head
    song_s = per_node[song_id]
    staged = sum(parsed["stages"].values())
    stages = []
    for name, took in sorted(parsed["stages"].items(), key=lambda kv: -kv[1]):
        entry = {"stage": name, "seconds": took, "share_of_run": took / total, "share_of_song_node": took / song_s,
                 "kept_on_a_seed_or_step_change": name in REPEAT_KEEPS}
        if name in PER_WINDOW:
            # by window, never a mean over them: a run's last window is usually shorter
            entry["by_window_s"] = {str(n): lap.get(name) for n, lap in sorted(parsed["windows"].items())}
        stages.append(entry)

    keepable = sum(s for name, s in parsed["stages"].items() if name in REPEAT_KEEPS)
    preview = sum(s for nid, s in per_node.items()
                  if classes.get(nid) == "PreviewAny" and graph[nid]["inputs"].get("source", [None])[0] == song_id)
    repeat = song_s + preview
    out = {
        **head,
        "song_node": {
            "node": song_id, "seconds": song_s, "share_of_run": song_s / total,
            "staged_s": staged, "unaccounted_s": song_s - staged,
            "unaccounted_is": "the node's work outside its own clock: the track's hash and the window keys "
                              "before it, the metadata PNG and the shot table after it",
            "stages": stages, "windows": parsed["windows"],
            "conditioning_by_window": parsed["conditioning_windows"],
            "window_frames": parsed["window_frames"],
            "regenerated_percent_of_video_tokens": parsed["regenerated_percent"],
            "reused_windows": parsed["reused_windows"], "unsampled_windows": parsed["unsampled_windows"],
        },
        "repeat_run": {
            "basis": SONG_ONLY_ON_REPEAT,
            "pays_again_s": repeat, "share_of_this_run": repeat / total,
            "of_which_does_not_depend_on_seed_or_steps_s": keepable,
            "that_as_a_share_of_the_repeat": keepable / repeat,
            "keeps": {name: REPEAT_KEEPS[name] for name in parsed["stages"] if name in REPEAT_KEEPS},
        },
    }
    if telemetry is not None:
        rows = telemetry_report.load(telemetry)
        summary = telemetry_report.phases(rows)
        seen = {n["node"]: n for n in summary["nodes"]}
        out["telemetry"] = {
            "record": telemetry.name, "total_s": summary["total_s"],
            "song_node_s": (seen.get(song_id) or {}).get("seconds"),
            "not_observed": [f"{nid} {classes.get(nid, '?')}" for nid in summary["not_observed"]],
            "models_at_song_start": (seen.get(song_id) or {}).get("models_at_start"),
            "models_at_song_end": (seen.get(song_id) or {}).get("models_at_end"),
            "loads_in_song_node": loads_in_song(rows, song_id, parsed),
            "stage_load": stage_load(rows, song_id, parsed),
            "load_stage_is_within_s": round(song_s - staged, 1),
        }
    return out


def _f(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def _pct(x):
    return "-" if x is None else f"{100.0 * x:.1f}%"


def _steps_markdown(run: dict) -> list[str]:
    steps = run.get("sampling_steps")
    lines: list[str] = []
    if steps and steps["windows"]:
        lines += ["", f"Sampling, by step ({steps['resolution']}); "
                  + (f"{steps['dense_steps_before_sol']} step(s) ran dense before Sol-Attn's first sparse call, on a "
                     f"sequence of {steps['sequence_rows']} rows:" if steps["dense_steps_before_sol"] is not None
                     else "Sol-Attn's first sparse line is not in this render's log (the shape was logged earlier):"),
                  "", "| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |", "|---|---|---|---|---|"]
        for i, w in enumerate(steps["windows"], 1):
            lines.append(f"| {i} | {' '.join(str(x) for x in w['steps_s'])} | {_f(w.get('dense_step_mean_s'))} | "
                         f"{_f(w.get('sol_step_mean_s'), 2)} | {w['sampler_s']} |")
    return lines


def markdown(run: dict) -> str:
    song = run["song_node"]
    numbers = sorted(int(n) for n in (song or {}).get("windows", {}))
    frames = (song or {}).get("window_frames", {})
    lines = [f"### {run['label']}", "",
             f"{_f(run['total_s'])} s in all. Cache state: {run['cache_state'] or 'not stated'}.", "",
             "| node | class | seconds | share of the run |", "|---|---|---|---|"]
    for n in run["nodes"]:
        if n["seconds"] >= 0.05:
            lines.append(f"| {n['node']} | `{n['class_type']}` | {_f(n['seconds'])} | {_pct(n['share'])} |")
    lines.append(f"| | outside any node | {_f(run['outside_any_node_s'])} | {_pct(run['outside_any_node_s'] / run['total_s'])} |")
    if song is None:
        lines += ["", f"No stages: {run['no_stages']}."]
        lines += _steps_markdown(run)
        tel = run.get("telemetry")
        if tel and tel["not_observed"]:
            lines += ["", "Nodes with no boundary in the telemetry (served from cache, unused, or uncacheable): "
                      + ", ".join(tel["not_observed"]) + "."]
        return "\n".join(lines)
    lines += ["", "Inside the song node:", "",
              "| stage | seconds | share of the run | "
              + " | ".join(f"window {n}" + (f" ({frames[n]} frames)" if n in frames else "") for n in numbers)
              + " | same on a seed or step change |",
              "|---|---|---|" + "---|" * len(numbers) + "---|"]
    for s in song["stages"]:
        cells = [_f((s.get("by_window_s") or {}).get(str(n))) if "by_window_s" in s else "" for n in numbers]
        lines.append(f"| {s['stage']} | {_f(s['seconds'])} | {_pct(s['share_of_run'])} | " + " | ".join(cells)
                     + f" | {'yes' if s['kept_on_a_seed_or_step_change'] else 'no'} |")
    lines.append(f"| not on the node's clock | {_f(song['unaccounted_s'])} | {_pct(song['unaccounted_s'] / run['total_s'])} | "
                 + " | ".join("" for _ in numbers) + " | |")
    if song.get("conditioning_by_window"):
        lines += ["", "Conditioning, by window: " + "; ".join(
            f"[{n}] " + ", ".join(f"{name} {took:.1f}" for name, took in lap.items())
            for n, lap in sorted(song["conditioning_by_window"].items(), key=lambda kv: int(kv[0]))) + "."]
    if song["regenerated_percent_of_video_tokens"]:
        lines += ["", "Video tokens regenerated, by window: "
                  + ", ".join(f"[{n}] {v:.1f}%" for n, v in sorted(song["regenerated_percent_of_video_tokens"].items()))
                  + (f"; not sampled: {song['unsampled_windows']}" if song["unsampled_windows"] else "") + "."]
    lines += _steps_markdown(run)
    rep = run["repeat_run"]
    lines += ["", f"A repeat of this run ({rep['basis']}) pays {_f(rep['pays_again_s'])} s again, "
              f"{_pct(rep['share_of_this_run'])} of this run; {_f(rep['of_which_does_not_depend_on_seed_or_steps_s'])} s of "
              f"that ({_pct(rep['that_as_a_share_of_the_repeat'])}) is in stages that do not depend on the seed or the step count."]
    tel = run.get("telemetry")
    if tel:
        lines += ["", f"Model loads while the song node ran (`{tel['record']}`; the stage is right to within "
                  f"{tel['load_stage_is_within_s']} s):", "", "| at, s | line | what | stage | window |", "|---|---|---|---|---|"]
        for e in tel["loads_in_song_node"]:
            lines.append(f"| {e['at_s']} | {e['kind']} | {e['what'] or ''} | {e['stage'] or 'outside the stages'} | {e['window'] or ''} |")
        busy = [e for e in tel.get("stage_load") or [] if e["seconds"] >= 1.0]
        if busy:
            lines += ["", "What the card and the processors did in each stage of a second or more (the server "
                      "process's CPU seconds; other processes are not in it):", "",
                      "| stage | window | seconds | GPU utilisation, mean | CPU seconds | CPU seconds per second |",
                      "|---|---|---|---|---|---|"]
            for e in busy:
                per = None if e["cpu_s"] is None else e["cpu_s"] / e["seconds"]
                lines.append(f"| {e['stage']} | {e['window'] or ''} | {_f(e['seconds'])} | {_f(e['gpu_util_mean'])} | "
                             f"{_f(e['cpu_s'])} | {_f(per)} |")
        if tel["not_observed"]:
            lines += ["", "Nodes with no boundary in the telemetry (served from cache, unused, or uncacheable): "
                      + ", ".join(tel["not_observed"]) + "."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, action="append",
                    help="bench/run_graph_arms.py's JSONL; may be given more than once")
    ap.add_argument("--history-dir", type=Path, required=True,
                    help="one <label>.json per render, holding the /history outputs under `outputs`")
    ap.add_argument("--telemetry-dir", type=Path, help="pipeline telemetry records; matched by prompt id")
    ap.add_argument("--server-log", type=Path, action="append", default=[],
                    help="a server's log, for seconds per sampling step; may be given more than once")
    ap.add_argument("--label", action="append", help="a row to read; every row with no error when omitted")
    ap.add_argument("--state", action="append", default=[], metavar="LABEL=TEXT",
                    help="the cache state a render ran in, in words")
    ap.add_argument("--out", type=Path, help="write the numbers here")
    args = ap.parse_args()

    states = dict(s.split("=", 1) for s in args.state)
    rows = [json.loads(line) for path in args.rows for line in path.read_text().splitlines() if line.strip()]
    runs = []
    server_logs = tuple(path.read_text(errors="replace") for path in args.server_log)
    for row in rows:
        label = row.get("label") or row.get("arm")
        if (args.label and label not in args.label) or row.get("error") or row.get("warmup"):
            continue
        history = args.history_dir / f"{label}.json"
        saved = json.loads(history.read_text()) if history.exists() else {}
        telemetry = find_telemetry(args.telemetry_dir, row["prompt_id"]) if args.telemetry_dir else None
        if args.telemetry_dir and telemetry is None:
            print(f"no telemetry record for {label} ({row['prompt_id']}) in {args.telemetry_dir}", file=sys.stderr)
        runs.append(breakdown(row, saved.get("outputs") or saved, telemetry, states.get(label), server_logs))
    if not runs:
        print("no row to read", file=sys.stderr)
        return 2
    for run in runs:
        print(markdown(run), end="\n\n")
    if args.out:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or None
        args.out.write_text(json.dumps({
            "what": "seconds by node and by song-node stage for masked song renders, with what a repeat run pays again",
            "tool": "bench/masked_render_time_breakdown.py", "tool_commit": commit,
            "rows": [path.name for path in args.rows], "seconds_are": "wall clock; per node from the websocket feed, per stage from the "
            "song node's own clock at stage ends", "repeat_run_is": "by rule, not timed (the tool's REPEAT_KEEPS)",
            "runs": runs}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
