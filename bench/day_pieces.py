#!/usr/bin/env python3
"""A folder for a day's end: every piece that stands in a delivery table, in frame order, each with the graph it was queued with, its text, what that graph names and whether each is there now, and the line that queues it again.

    <python> bench/day_pieces.py --table T.json --ledger LEDGER.json --items data/<stem>/review/items.json \\
        --inputs <the server's input folder> --models <the server's models folder> --out <share>/Video/<stem>/day_<date> \\
        [--clip PLACED.mp4 --stacked PLACED_STACKED.mp4 --clip-table PLACED.txt] [--host 127.0.0.1:8188]

**What it buys.** The owner asked (2026-10-13) for one place at the end of a day: the joined clip, and each piece
with a workflow by which that piece can be run again. Everything needed exists and is in four places: the delivery
table says which pieces stand and on which frames; the render ledger ties each to its build and queue records; the
job folder holds the graph AS IT WAS QUEUED (`<load>.render.json`, the API form, which opens in the ComfyUI page as
it is) and its text; the review list names its stacked clip. Nobody could open one folder and run a piece again.

**What it writes, into `--out`.** `index.html`: one row a piece in frame order, with its frames, subject and
region, the day its render was written, the owner's last choice on it where they made one, and links BY RELATIVE
PATH to its render and its stacked clip (they are on the same share; the share takes no links, so nothing here is
one). With `--clip`, the first rows link to the joined clip, its stacked version and its table. Then a subfolder a
piece, holding small files only:

- `graph_as_queued.json`: the queued graph, byte for byte;
- `queue_this.json`: the same graph as the body the server's `/prompt` takes;
- `text.txt`: the load's text as queued;
- `inputs.txt`: every file the graph names, one a line, THERE or MISSING; for one that is there its size, when it
  was written, and WRITTEN AFTER THE GRAPH WAS QUEUED where that is so (it may have been made again since). A
  model is a name the server resolves: found or not found under `--models`, said as that. A missing input is
  printed and nothing is refused: knowing what is missing is what the owner asked for;
- `how_to_queue.txt`: the one line, and that the graph writes under the same output name as before, so a run of it
  adds a next-numbered file beside the kept one.

**No video is copied.** A piece whose record holds no queued graph is still a row and a folder, and says so.

**What it cannot see.** Whether a graph run again makes the same picture: the node code and the models may have
moved since (the build record's commit is on the row). Whether an input that is there is the file that was there.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

MEDIA = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".mkv", ".webm", ".wav", ".flac", ".mp3", ".npz", ".json")
MODELS = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin")
#: Inputs that name where a graph WRITES, not what it reads.
WRITES = ("filename_prefix",)


def named(graph: dict) -> list[tuple[str, str, str]]:
    """(node class, input, value) for every string a graph's nodes name that is a file, a model or an output name."""
    out = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        for key, value in (node.get("inputs") or {}).items():
            # a file is a string with no space in it that ends as a media or model file does, or an absolute path; a
            # text with a slash in it is a text
            if isinstance(value, str) and (key in WRITES or (not any(c.isspace() for c in value) and (value.startswith("/") or value.lower().endswith(MEDIA + MODELS)))):
                out.append((str(node.get("class_type")), key, value))
    return sorted(set(out))


def utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inputs_lines(graph: dict, queued_at: str, inputs: Path, share: Path, models: Path | None) -> list[str]:
    """One line a thing the graph names: THERE or MISSING, and what can be said of one that is there."""
    try:
        queued = datetime.fromisoformat(queued_at).timestamp() if queued_at else None
    except ValueError:
        queued = None
    lines = []
    for cls, key, value in named(graph):
        where = f"{cls}.{key} = {value}"
        if key in WRITES:
            lines.append(f"WRITES   an output name, not an input: {where}")
            continue
        if value.lower().endswith(MODELS) and "/" not in value.strip("/"):
            found = sorted(models.rglob(value)) if models and models.is_dir() else []
            lines.append((f"THERE    a model, found under the server's model folders as {found[0]}: " if found else
                          "MISSING  a model, NOT FOUND under the server's model folders" + ("" if models else " (no --models was given)") + ": ") + where)
            continue
        tried = [Path(value)] if os.path.isabs(value) else [inputs / value, share / value]
        hit = next((p for p in tried if p.is_file()), None)
        if hit is None:
            lines.append(f"MISSING  looked for as {' and as '.join(str(p) for p in tried)}: {where}")
            continue
        stat = hit.stat()
        after = "  WRITTEN AFTER THE GRAPH WAS QUEUED" if queued is not None and stat.st_mtime > queued else ""
        lines.append(f"THERE    {stat.st_size} bytes, written {utc(stat.st_mtime)}{after}: {where}" + ("" if os.path.isabs(value) else f" (as {hit})"))
    return lines


def pieces(table: dict, ledger: dict, items: dict) -> list[dict]:
    """Every piece row of the table, in frame order, with its records, its stacked clip and the owner's last choice."""
    stacked = {i["id"]: i for i in items.get("items", [])}
    out = []
    for row in sorted(table["rows"], key=lambda r: (r["first"], r["last"], r["name"])):
        render = (ledger.get("renders") or {}).get(row["file"]) or {}
        record = (render.get("records") or [{}])[0]
        built = record.get("built_file")
        choices = [o for o in render.get("owner") or [] if o.get("what") == "choice"]
        item = stacked.get(row["name"]) or {}
        out.append({"first": row["first"], "last": row["last"], "name": row["name"], "subject": row.get("subject"), "region": row.get("region"),
                    "render": row["file"], "written": render.get("written"), "commit": record.get("commit"), "queued_at": record.get("queued_at"),
                    "graph": built.replace(".built.json", ".json") if built else None, "text": built.replace(".built.json", ".txt") if built else None,
                    "stacked": os.path.realpath(item["path"]) if item.get("path") else None,
                    "owner_last_choice": (f"{choices[-1]['said']} ({choices[-1]['at']})" if choices else None)})
    return out


def link(target, out: Path, words: str) -> str:
    if not target or not Path(target).is_file():
        return f'<span class="no">{html.escape(words)}: not there</span>'
    return f'<a href="{html.escape(os.path.relpath(target, out))}">{html.escape(words)}</a>'


def write(found: list[dict], out: Path, repo: Path, share: Path, inputs: Path, models: Path | None, host: str, made: str, joined: dict) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    said, rows = [], []
    for p in found:
        folder = out / f"{p['first']:04d}_{p['name']}"
        folder.mkdir(exist_ok=True)
        graph_file = repo / p["graph"] if p["graph"] else None
        if graph_file is not None and graph_file.is_file():
            raw = graph_file.read_bytes()
            (folder / "graph_as_queued.json").write_bytes(raw)
            graph = json.loads(raw)
            (folder / "queue_this.json").write_text(json.dumps({"prompt": graph.get("prompt", graph)}))
            lines = inputs_lines(graph.get("prompt", graph), p["queued_at"], inputs, share, models)
            (folder / "inputs.txt").write_text(f"What {p['name']}'s queued graph names, read {made}. THERE or MISSING is of this moment; a missing input is not a fault of "
                                               "this folder.\n\n" + "\n".join(lines) + "\n")
            (folder / "how_to_queue.txt").write_text(
                f"curl -s -X POST -H 'Content-Type: application/json' --data-binary @queue_this.json http://{host}/prompt\n\n"
                "Run from this folder, with the server up. The graph writes under the same output name as before (its filename_prefix), so a run of it "
                "adds a next-numbered file beside the kept one. graph_as_queued.json opens in the ComfyUI page as it is.\n")
            missing = sum(1 for line in lines if line.startswith("MISSING"))
            state = f"graph as queued; {len(lines)} thing(s) named, {missing} MISSING"
        else:
            (folder / "NO_QUEUED_GRAPH.txt").write_text(f"{p['name']}: no queued graph is beside its build record ({p['graph'] or 'the ledger holds no build record for it'}).\n")
            state = "NO QUEUED GRAPH beside its record"
        text_file = repo / p["text"] if p["text"] else None
        if text_file is not None and text_file.is_file():
            shutil.copyfile(text_file, folder / "text.txt")
        said.append(f"{p['first']}-{p['last']} {p['name']}: {state}")
        rows.append(f'<tr><td class="f">{p["first"]}-{p["last"]}</td><td><b>{html.escape(p["name"])}</b><br>{html.escape(str(p["subject"]))}, {html.escape(str(p["region"]))}</td>'
                    f'<td>{html.escape(str(p["written"] or "not known")[:10])}</td><td>{html.escape(p["owner_last_choice"] or "no choice of the owner\'s on it")}</td>'
                    f'<td>{link(share / p["render"], out, "the render")}<br>{link(p["stacked"], out, "stacked")}</td>'
                    f'<td><a href="{html.escape(folder.name)}/">{html.escape(folder.name)}/</a><br>{html.escape(state)}'
                    + (f'<br>built at commit {html.escape(str(p["commit"]))}' if p["commit"] else "") + '</td></tr>')
    top = "".join(f"<li>{link(joined[k], out, words)}</li>" for k, words in (("clip", "THE JOINED CLIP"), ("stacked", "the joined clip, stacked against the source"),
                                                                            ("clip_table", "the table it was built from")) if joined.get(k))
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>the day\'s pieces</title><style>body{font:16px/1.45 system-ui,sans-serif;margin:16px;background:#14161a;color:#e6e6e6}'
        'table{border-collapse:collapse}td,th{border-top:1px solid #333;padding:8px;vertical-align:top;text-align:left}a{color:#8ab4f8}.f{font-family:ui-monospace,monospace}'
        '.no{color:#8a8f99}</style>'
        f'<h1>The pieces that stand, in frame order</h1><p>Made {html.escape(made)} by bench/day_pieces.py. {len(found)} piece(s). Links are relative paths on this share; '
        'no video is copied here. Each piece\'s folder holds the graph it was queued with, its text, what that graph names and whether each is there now, and the line that '
        f'queues it again.</p>{"<ul>" + top + "</ul>" if top else ""}'
        '<table><tr><th>frames</th><th>piece</th><th>render written</th><th>the owner\'s last choice</th><th>the files</th><th>its folder</th></tr>' + "".join(rows) + '</table>\n')
    return said


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--table", required=True, help="the delivery table's .json (bench/delivery_table.py --json): which pieces stand")
    ap.add_argument("--ledger", required=True, help="the render ledger's .json: each piece's records and the owner's lines")
    ap.add_argument("--items", required=True, help="the review list (data/<stem>/review/items.json): each piece's stacked clip")
    ap.add_argument("--inputs", required=True, help="the server's input folder: where a graph's relative file names are looked for")
    ap.add_argument("--models", help="the server's models folder: where a graph's model names are looked for")
    ap.add_argument("--out", required=True, help="the folder to write, on the share")
    ap.add_argument("--clip", help="the placed joined clip, linked from the page's first rows")
    ap.add_argument("--stacked", help="its stacked version")
    ap.add_argument("--clip-table", help="the table beside it")
    ap.add_argument("--host", default="127.0.0.1:8188", help="the server in the line that queues a graph")
    a = ap.parse_args()
    ledger = json.load(open(a.ledger))
    found = pieces(json.load(open(a.table)), ledger, json.load(open(a.items)))
    made = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    said = write(found, Path(a.out), Path(__file__).resolve().parent.parent, Path(ledger["share"]), Path(a.inputs), Path(a.models) if a.models else None, a.host, made,
                 {"clip": a.clip, "stacked": a.stacked, "clip_table": a.clip_table})
    print("\n".join(said))
    print(f"\n{len(found)} piece(s) written to {a.out}; index.html there; no video copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
