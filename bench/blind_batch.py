#!/usr/bin/env python3
"""Turn a run_graph_arms JSONL into a blinded batch of clips with a sealed key.

An instrument, not a check: it grades nothing about the repo. It exists because
CLAUDE.md's standard for a perceptual claim about a knob is a distribution --
many seeds per arm, judged blind in aggregate -- and the only blinding tool
here, `bench/stack_eval_clips.py`, takes two clips and one key. A 4-step render
is about three minutes, so 8 seeds x 4 arms is an hour of card and ~30 neutral
clips; this is the layer that hides which arm each one came from.

## What it does

Reads the JSONL `bench/run_graph_arms.py` appends to. For every non-warmup row
it locates the render's video file, copies it under a neutral name
(`clip_NN.mp4`, NN from a seeded shuffle) into `<output>/Video/blind/<session>/`,
writes a MANIFEST beside the clips that maps each clip to its JSONL row index
and nothing else, and writes the KEY -- clip -> (label, seed, source file) --
to `internal/blind_keys/<session>.json`, which is gitignored. The judge opens
the batch directory; nobody opens the key until the scores are written.

## How a row finds its clip

`run_graph_arms` appends `_<label>` to every `filename_prefix`, so the arm is
in every output filename; this tool must never expose those names. Two ways
to find the file, in order:

1. **`prompt_id`**, recorded per row since 2026-08-20: `GET /history/<id>` on
   the server that rendered it names the exact output files. Exact, and the
   only handle that survives a counter restart.
2. **Fallback, when a row has no `prompt_id` or the server has no history for
   it:** `<prefix>_NNNNN.mp4` files for the row's label on the output share,
   taken in counter order among the rows of that label, cross-checked against
   the row's `ts` .. `ts + wall_s` window by mtime. Refuses if any same-prefix
   clip predates the JSONL's first row (the counter continues from earlier
   renders and the order would be wrong) or if the mtime check fails.

Only `<prefix>_NNNNN.mp4` is taken; the share also holds `-audio.mp4` and
`.png` siblings per render, and a wrong sibling would blind the wrong file.

## Refusals, each a row that must not be judged

`suspect_cache_hit` (the server returned a stored result; not a render),
`error`, or a clip that cannot be found. A batch with any of those is refused
whole: a missing clip shifts every neutral number after it.

## Pairs, for two-arm sessions

`--pairs A,B` additionally stacks the i-th judged clip of arm A with the i-th
of arm B (matched by run index, which `run_graph_arms` seeds identically
across arms unless the seed bases were set apart on purpose) into
`pair_NN.mp4`, top/bottom or side-by-side per `bench/stack_eval_clips.py`'s
layout rule, labelled "Clip 1" / "Clip 2" with the order drawn per pair from
the same seeded RNG. The key records which arm sat where. Stacks carry no
audio -- `build_stacked_video` maps video only, and one track for two arms
would be a bias anyway -- so the singles stay in the batch for the audio
half of the brief. A pair is a presentation of two different samples, never a
matched comparison (CLAUDE.md, the different-sample rule); the judgement is
still the aggregate over pairs.

The output root comes from `H3_COMFY_OUTPUT` (or `bench/_paths.comfy_output()`
when run where `folder_paths` resolves); the share's path is typed in the
shell, never here, and the manifest stores paths relative to the batch
directory.

    H3_COMFY_OUTPUT=<share> python bench/blind_batch.py \\
        --jsonl bench/results/2026-08-20_session1_lora_file.jsonl \\
        --session session1 --shuffle-seed 7
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _paths import comfy_output  # noqa: E402
from stack_eval_clips import build_stacked_video  # noqa: E402

KEY_DIR = REPO / "internal" / "blind_keys"


def history_outputs(host: str, prompt_id: str):
    """Video files the server recorded for this prompt, or None."""
    try:
        with urllib.request.urlopen(f"http://{host}/history/{prompt_id}", timeout=10) as r:
            doc = json.load(r)
    except Exception:
        return None
    entry = doc.get(prompt_id)
    if not entry:
        return None
    out = []
    for node_out in entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fn = item.get("filename", "")
                if fn.endswith(".mp4") and not fn.endswith("-audio.mp4"):
                    out.append(Path(item.get("subfolder", "")) / fn)
    return out or None


def prefix_of(graph_path: Path, label: str) -> str | None:
    doc = json.loads(graph_path.read_text())
    for node in doc.values():
        pfx = node.get("inputs", {}).get("filename_prefix")
        if isinstance(pfx, str):
            return f"{pfx}_{label}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--session", required=True, help="batch name; also the key file's name")
    ap.add_argument("--shuffle-seed", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--output-root", default=None,
                    help="ComfyUI output directory; default H3_COMFY_OUTPUT or folder_paths")
    ap.add_argument("--pairs", default=None, metavar="A,B",
                    help="also emit blinded stacked pairs of these two arm labels, matched by run index")
    args = ap.parse_args()
    pair_labels = None
    if args.pairs:
        pair_labels = [x.strip() for x in args.pairs.split(",")]
        if len(pair_labels) != 2 or pair_labels[0] == pair_labels[1]:
            sys.exit("refuse: --pairs needs two distinct arm labels")

    root = Path(args.output_root) if args.output_root else comfy_output()
    if root is None or not root.is_dir():
        sys.exit("refuse: no output root; set H3_COMFY_OUTPUT or pass --output-root")

    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    rows_idx = [(i, r) for i, r in enumerate(rows) if not r.get("warmup")]
    if not rows_idx:
        sys.exit("refuse: no non-warmup rows")
    bad = [(i, "suspect_cache_hit") for i, r in rows_idx if r.get("suspect_cache_hit")]
    bad += [(i, f"error: {r['error']}") for i, r in rows_idx if r.get("error")]
    if bad:
        sys.exit(f"refuse: rows that must not be judged: {bad}")

    # Resolve every row's clip before copying anything. A row's `ts` is
    # written when the render FINISHES, so its clip's mtime precedes `ts` by up
    # to `wall_s`; every window below is [ts - wall_s - slack, ts + slack].
    def window(r):
        t_end = time.mktime(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S"))
        return t_end - float(r.get("wall_s") or 0) - 120, t_end + 5
    first_start = min(window(r)[0] for _, r in rows_idx)
    # Counter alignment walks EVERY row of a label, warmup included: the
    # warmup's clip took a counter slot too, and aligning only the judged rows
    # shifted every clip by one on the first self-test (2026-08-20).
    per_label: dict[str, list] = {}
    for i, r in enumerate(rows):
        per_label.setdefault(r["label"], []).append((i, r))
    judged = {i for i, _ in rows_idx}
    located: dict[int, Path] = {}
    for label, lrows in per_label.items():
        pending = []
        for i, r in lrows:
            files = history_outputs(args.host, r.get("prompt_id")) if r.get("prompt_id") else None
            if files:
                if len(files) != 1:
                    sys.exit(f"refuse: row {i} ({label}) has {len(files)} video outputs in history; expected one")
                located[i] = root / files[0]
            else:
                pending.append((i, r))
        if not pending:
            continue
        if any(i in located for i, _ in lrows):
            sys.exit(f"refuse: {label}: some rows resolve by prompt_id and some do not; the counter "
                     "order cannot be aligned across a partial history")
        graph = REPO / lrows[0][1]["graph"]
        pfx = prefix_of(graph, label)
        if pfx is None:
            sys.exit(f"refuse: {graph} has no filename_prefix to search by")
        cands = sorted((root / Path(pfx).parent).glob(Path(pfx).name + "_[0-9][0-9][0-9][0-9][0-9].mp4"))
        if any(c.stat().st_mtime < first_start for c in cands):
            sys.exit(f"refuse: {label}: clips with this prefix predate the JSONL's first row; "
                     "the counter continues from earlier renders and order cannot be trusted")
        if len(cands) < len(pending):
            sys.exit(f"refuse: {label}: {len(cands)} clips on the share for {len(pending)} rows")
        for (i, r), c in zip(pending, cands):
            lo, hi = window(r)
            if not (lo <= c.stat().st_mtime <= hi):
                sys.exit(f"refuse: {label} row {i}: {c.name} mtime is outside the row's render window")
            if i in judged:
                located[i] = c
    located = {i: p for i, p in located.items() if i in judged}
    missing = [i for i, _ in rows_idx if i not in located or not located[i].is_file()]
    if missing:
        sys.exit(f"refuse: clips not found for rows {missing}")

    batch = root / "Video" / "blind" / args.session
    if batch.exists():
        sys.exit(f"refuse: {batch} exists")
    batch.mkdir(parents=True)
    rng = random.Random(args.shuffle_seed)
    order = [i for i, _ in rows_idx]
    rng.shuffle(order)
    manifest, key = [], {}
    for n, i in enumerate(order, 1):
        name = f"clip_{n:02d}.mp4"
        shutil.copy2(located[i], batch / name)
        manifest.append({"clip": name, "row": i})
        r = rows[i]
        key[name] = {"row": i, "label": r["label"], "seed": r["seed"],
                     "source": str(located[i].relative_to(root)), "graph": r["graph"]}
    pairs_manifest = []
    if pair_labels:
        a, b = pair_labels
        ra = [i for i, r in rows_idx if r["label"] == a]
        rb = [i for i, r in rows_idx if r["label"] == b]
        if not ra or not rb:
            sys.exit(f"refuse: --pairs {a},{b}: an arm has no judged rows")
        if len(ra) != len(rb):
            sys.exit(f"refuse: --pairs {a},{b}: {len(ra)} against {len(rb)} judged rows; "
                     "pairs match by run index and the arms are uneven")
        for n, (ia, ib) in enumerate(zip(ra, rb), 1):
            first, second = ((ia, a), (ib, b)) if rng.random() < 0.5 else ((ib, b), (ia, a))
            name = f"pair_{n:02d}.mp4"
            # The stacker prints its input filenames, which carry the arm
            # labels; a blind batch must not put those on the judge's terminal.
            with contextlib.redirect_stdout(io.StringIO()):
                build_stacked_video(located[first[0]], located[second[0]], batch / name,
                                    label1="Clip 1", label2="Clip 2")
            pairs_manifest.append({"pair": name, "rows": [first[0], second[0]]})
            key[name] = {"clip_1": {"row": first[0], "label": first[1], "seed": rows[first[0]]["seed"]},
                         "clip_2": {"row": second[0], "label": second[1], "seed": rows[second[0]]["seed"]}}
    (batch / "MANIFEST.json").write_text(json.dumps(
        {"session": args.session, "jsonl": args.jsonl, "clips": manifest, "pairs": pairs_manifest,
         "note": "row indices only; the key is sealed elsewhere"}, indent=1))
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    key_path = KEY_DIR / f"{args.session}.json"
    if key_path.exists():
        sys.exit(f"refuse: {key_path} exists")
    key_path.write_text(json.dumps({"session": args.session, "shuffle_seed": args.shuffle_seed,
                                    "jsonl": args.jsonl, "key": key}, indent=1))
    print(f"{len(order)} clips" + (f" and {len(pairs_manifest)} stacked pairs" if pairs_manifest else "") + f" -> {batch}")
    print(f"key sealed at internal/blind_keys/{args.session}.json -- DO NOT OPEN BEFORE SCORING")
    return 0


if __name__ == "__main__":
    sys.exit(main())
