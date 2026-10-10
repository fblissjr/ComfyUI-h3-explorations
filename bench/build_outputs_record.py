#!/usr/bin/env python3
"""Join a run JSONL to the clips the server wrote for it, before the server goes down.

The shape is `data/results/2026-09-03_ladder_outputs.json`, which was built
by hand on 2026-09-03 and is what `bench/measure_clip_loudness.py` reads: one
entry per judged arm with its label, prompt id, seed, graph, timings, and the
muxed clip basename. `/history` is the only place the arm-to-clip join is
recorded, and it dies with the server process, so run this while the server
that rendered the rows is still up.

    python bench/build_outputs_record.py --jsonl data/results/<date>_<run>.jsonl \\
        --host 127.0.0.1:8188 --out data/results/<date>_<run>_outputs.json --what "..."

**When the history is already gone** (2026-09-10: a restart came before this
ran), `--counter-fallback` joins through `bench/blind_batch.py::locate_clips`
instead: each label's clips on the share in filename-counter order,
cross-checked against every row's render window by mtime, refusing rather than
guessing exactly as a blind batch does. The record says per arm which route
found its clip (`clip_source`), so a reader can tell the exact join from the
reconstructed one.

Refusals: a judged row whose prompt id the server does not know (rendered by
another process, or the history was lost) unless the fallback finds it; a row
with more or fewer than one muxed clip; an absolute path in the record. Warmup
rows are listed with `warmup: true` and their clips, since they took counter
slots. Basenames only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def history_entry(host: str, prompt_id: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://{host}/history/{prompt_id}", timeout=10) as r:
            doc = json.load(r)
    except Exception:
        return None
    return doc.get(prompt_id) or None


def muxed_outputs(entry: dict) -> list[str]:
    """The `-audio.mp4` names the combine node recorded, basenames only."""
    out = []
    for node_out in entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fn = item.get("filename", "")
                if fn.endswith(".mp4"):
                    out.append(Path(fn).name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--out", required=True)
    ap.add_argument("--what", required=True, help="one sentence on what this run is")
    ap.add_argument("--counter-fallback", action="store_true",
                    help="join rows the server's history no longer knows through "
                         "blind_batch.locate_clips (filename counter + mtime window)")
    ap.add_argument("--output-root", default=None,
                    help="the output share, for --counter-fallback; default as blind_batch "
                         "resolves it (H3_COMFY_OUTPUT, else the live server's --output-directory)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit(f"refuse: {args.jsonl} has no rows")

    fallback: dict[int, Path] | None = None
    if args.counter_fallback:
        sys.path.insert(0, str(HERE))
        import blind_batch as _bb  # noqa: E402 -- the one implementation of the join
        root = Path(args.output_root) if args.output_root else _bb.comfy_output()
        fallback = {i: _bb.single_source(p) for i, p in _bb.locate_clips(rows, args.host, root).items()}

    arms, problems, sources = [], [], set()
    for i, r in enumerate(rows):
        pid = r.get("prompt_id")
        entry = history_entry(args.host, pid) if pid else None
        if entry is not None:
            clips = muxed_outputs(entry)
            muxed = [c for c in clips if c.endswith("-audio.mp4")]
            source = "history"
        elif fallback is not None and i in fallback:
            clips = [fallback[i].name]
            muxed = [c for c in clips if c.endswith("-audio.mp4")]
            source = "counter+mtime"
        else:
            problems.append(f"row {i} ({r.get('label')}): prompt id {pid!r} unknown to {args.host}")
            continue
        if len(muxed) != 1:
            problems.append(f"row {i} ({r.get('label')}): {len(muxed)} muxed clip(s) via {source}, expected one: {clips}")
            continue
        sources.add(source)
        arms.append({
            "label": r.get("label"), "prompt_id": pid, "seed": r.get("seed"),
            "graph": r.get("graph"), "graph_sha256": r.get("graph_sha256"),
            "rendered": r.get("rendered"), "warmup": bool(r.get("warmup")),
            "total_s": r.get("total_s"), "sampler_s": r.get("sampler_s"), "decode_s": r.get("decode_s"),
            "suspect_cache_hit": r.get("suspect_cache_hit"),
            "outputs": muxed, "clip_source": source,
            "note": r.get("judge_note") or r.get("note"),
        })
    if problems:
        sys.exit("refuse: the run and the clips do not join:\n  " + "\n  ".join(problems))
    route = ("the rendering server's /history" if sources == {"history"} else
             "the filename counter and mtime window (blind_batch.locate_clips), the history being gone"
             if sources == {"counter+mtime"} else "a mix of /history and the counter fallback, per arm")
    record = {
        "what": args.what,
        "measured": _dt.date.today().isoformat(),
        "produced_by": f"bench/build_outputs_record.py joining {Path(args.jsonl).name} through {route}; basenames only, the clips live in the output folder the launcher names",
        "source_jsonl": str(Path(args.jsonl)) if not Path(args.jsonl).is_absolute() else Path(args.jsonl).name,
        "arms": arms,
    }
    def scrub(node, where="record"):
        if isinstance(node, dict):
            for k, v in node.items(): scrub(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node): scrub(v, f"{where}[{i}]")
        elif isinstance(node, str) and (node.startswith("/") or node.startswith("~") or "/home/" in node):
            sys.exit(f"refuse: {where} carries an absolute path: {node!r}")
    scrub(record)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    n_judged = sum(1 for a in arms if not a["warmup"])
    print(f"{len(arms)} arm(s) joined ({n_judged} judged, {len(arms) - n_judged} warmup) via "
          f"{route}; wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
