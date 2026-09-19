#!/usr/bin/env python3
"""Build a blinded panel of stacked pairs from clips that already exist, with a sealed key.

    H3_COMFY_OUTPUT=<share> python bench/blind_panel.py --spec <spec.json>

`bench/blind_batch.py` blinds ONE `run_graph_arms` JSONL: arms by seed, pairs
matched by run index. A panel across SCENES is a different shape: each contest
is two named clips, often rendered on different days (a reference clip from an
older batch against a new arm), and the batch tool cannot express it. This is
the same pipeline for that shape: neutral `clip_NN.mp4` singles with sound,
`pair_NN.mp4` stacks in a seeded order with a seeded slot draw, a MANIFEST
with row indices only, the key sealed under `internal/blind_keys/`, and the
same `score.html`.

## The spec

    {"session": "...", "shuffle_seed": 7, "brief": "optional text for the page",
     "contests": [
       {"name": "diner", "kind": "real",
        "a": {"label": "default", "clip": "Video/.../x_00001.mp4"},
        "b": {"label": "dense",   "clip": "Video/.../y_00001.mp4"},
        "differ_only": ["MiniMaxH3SolAttn[0].", "ModelAttentionBackend[0]."]}]}

Clip paths are relative to the ComfyUI output root and name the SILENT file;
each single is copied from its `-audio.mp4` sibling (`blind_batch.single_source`).
The spec carries the arm labels, so it lives under `internal/`, never in a
record.

## Contest kinds, and what each refuses

- `real`, `two_seed_decoy`: two different clips. The graphs embedded in them
  are diffed (`bench/diff_clip_graphs.py`) and every differing input must
  start with one of the contest's `differ_only` prefixes, written as
  `Class[n].field`; anything else refuses the panel. A pair that differs in
  more than it claims is how two verdicts of 2026-09-18 came to be about
  something else. The differing inputs are stored in the key.
- `identical_decoy`: the same clip in both halves. An attention check: a
  judge who picks a winner here voids the session. Refused unless `a` and
  `b` name the same file, so it cannot happen by accident, and no other kind
  accepts the same file twice.
- `low_anchor`: `b` is `a` with a known mild defect added by post-processing,
  so it carries no graph of its own and is not diffed; the spec's `note` says
  what was done to it. A judge who misses it voids the session.

The kind is in the key and nowhere the judge can see. No GPU, no server.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import shutil
import sys
from pathlib import Path

import orjson

_SUMMARY = (__doc__ or "").split("\n")[0]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _paths import comfy_output  # noqa: E402
from blind_batch import KEY_DIR, has_audio_stream, single_source  # noqa: E402
from blind_score_app import load_rubric, write_score_app  # noqa: E402
from diff_clip_graphs import flat, graph_of  # noqa: E402
from stack_eval_clips import build_stacked_video  # noqa: E402

KINDS = ("real", "two_seed_decoy", "identical_decoy", "low_anchor")
DIFFED = ("real", "two_seed_decoy")


def graph_differences(a: Path, b: Path) -> dict[str, list]:
    """{`Class[n].field`: [value in a, value in b]} for every input that differs."""
    fa, fb = flat(graph_of(str(a))), flat(graph_of(str(b)))
    out = {}
    for key in sorted(set(fa) | set(fb), key=str):
        va, vb = fa.get(key, "<absent>"), fb.get(key, "<absent>")
        if va != vb:
            cls, n, field = key
            out[f"{cls}[{n}].{field}"] = [va, vb]
    return out


def check_contest(c: dict, root: Path) -> tuple[Path, Path, dict]:
    name = c.get("name") or "<unnamed>"
    if c.get("kind") not in KINDS:
        sys.exit(f"refuse: contest {name}: kind must be one of {KINDS}")
    paths = []
    for side in ("a", "b"):
        entry = c.get(side) or {}
        if not entry.get("label") or not entry.get("clip"):
            sys.exit(f"refuse: contest {name}: side {side} needs a label and a clip")
        p = root / entry["clip"]
        if not p.is_file():
            sys.exit(f"refuse: contest {name}: side {side} clip not found under the output root: {entry['clip']}")
        paths.append(p)
    a, b = paths
    same = a.resolve() == b.resolve()
    if same != (c["kind"] == "identical_decoy"):
        sys.exit(f"refuse: contest {name}: kind {c['kind']} with "
                 f"{'the same file' if same else 'two different files'} in both halves")
    differences = {}
    if c["kind"] in DIFFED:
        allowed = c.get("differ_only")
        if not allowed:
            sys.exit(f"refuse: contest {name}: kind {c['kind']} needs `differ_only` prefixes")
        differences = graph_differences(a, b)
        if not differences:
            sys.exit(f"refuse: contest {name}: the two embedded graphs are identical")
        stray = [k for k in differences if not any(k.startswith(p) for p in allowed)]
        if stray:
            sys.exit(f"refuse: contest {name}: the graphs differ outside `differ_only`: {stray}")
    elif c["kind"] == "low_anchor" and not c.get("note"):
        sys.exit(f"refuse: contest {name}: a low anchor needs a `note` saying what was done to side b")
    return a, b, differences


def main() -> int:
    ap = argparse.ArgumentParser(description=_SUMMARY)
    ap.add_argument("--spec", required=True, help="panel spec JSON; carries arm labels, so keep it under internal/")
    ap.add_argument("--output-root", default=None,
                    help="ComfyUI output directory; default H3_COMFY_OUTPUT, else the live server's --output-directory")
    ap.add_argument("--silent-ok", action="store_true", help="accept singles with no audio stream")
    ap.add_argument("--check-only", action="store_true", help="run every refusal and write nothing")
    args = ap.parse_args()

    spec = orjson.loads(Path(args.spec).read_bytes())
    session, contests = spec.get("session"), spec.get("contests") or []
    if not session or not contests or "shuffle_seed" not in spec:
        sys.exit("refuse: the spec needs `session`, `shuffle_seed` and at least one contest")
    names = [c.get("name") for c in contests]
    if len(set(names)) != len(names):
        sys.exit(f"refuse: contest names repeat: {names}")
    root = Path(args.output_root) if args.output_root else comfy_output()
    if not root.is_dir():
        sys.exit(f"refuse: output root is not a directory: {root}; set H3_COMFY_OUTPUT or pass --output-root")

    checked = [check_contest(c, root) for c in contests]
    deaf = [c["name"] for c, (a, b, _) in zip(contests, checked)
            if not (has_audio_stream(single_source(a)) and has_audio_stream(single_source(b)))]
    if deaf and not args.silent_ok:
        sys.exit(f"refuse: a single would carry no audio stream in contests {deaf}; pass --silent-ok if that is meant")
    batch = root / "Video" / "blind" / session
    key_path = KEY_DIR / f"{session}.json"
    if batch.exists() or key_path.exists():
        sys.exit(f"refuse: session {session} already exists (batch directory or key)")
    if args.check_only:
        print(f"{len(contests)} contests pass every check; nothing written")
        return 0

    batch.mkdir(parents=True)
    rng = random.Random(spec["shuffle_seed"])
    order = list(range(len(contests)))
    rng.shuffle(order)
    # Singles are numbered in their own shuffle so clip_NN does not give away
    # which pair a single belongs to, or which half of it.
    singles = [(ci, side) for ci in order for side in ("a", "b")]
    rng.shuffle(singles)
    row_of = {cs: row for row, cs in enumerate(singles)}
    manifest_clips, key = [], {}
    for row, (ci, side) in enumerate(singles):
        c, src = contests[ci], single_source(checked[ci][0 if side == "a" else 1])
        clip_name = f"clip_{row + 1:02d}.mp4"
        shutil.copy2(src, batch / clip_name)
        manifest_clips.append({"clip": clip_name, "row": row})
        key[clip_name] = {"row": row, "contest": c["name"], "label": c[side]["label"],
                          "source": str(src.relative_to(root))}
    manifest_pairs = []
    for n, ci in enumerate(order, 1):
        c, (a, b, differences) = contests[ci], checked[ci]
        slots = [("a", a), ("b", b)] if rng.random() < 0.5 else [("b", b), ("a", a)]
        pair_name = f"pair_{n:02d}.mp4"
        # The stacker prints its input filenames, which carry the arm labels.
        with contextlib.redirect_stdout(io.StringIO()):
            build_stacked_video(slots[0][1], slots[1][1], batch / pair_name, label1="Clip 1", label2="Clip 2")
        manifest_pairs.append({"pair": pair_name, "rows": [row_of[(ci, s)] for s, _ in slots]})
        key[pair_name] = {"contest": c["name"], "kind": c["kind"], "note": c.get("note", ""),
                          "clip_1": c[slots[0][0]]["label"], "clip_2": c[slots[1][0]]["label"],
                          "a_label": c["a"]["label"], "b_label": c["b"]["label"],
                          "graph_differences_a_to_b": differences}
    (batch / "MANIFEST.json").write_bytes(orjson.dumps(
        {"session": session, "clips": manifest_clips, "pairs": manifest_pairs,
         "note": "row indices only; the key is sealed elsewhere"}, option=orjson.OPT_INDENT_2))
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(orjson.dumps({"session": session, "shuffle_seed": spec["shuffle_seed"], "key": key},
                                      option=orjson.OPT_INDENT_2))
    print(f"{len(manifest_pairs)} stacked pairs and {len(manifest_clips)} singles -> {batch}")
    print(f"key sealed at internal/blind_keys/{session}.json -- DO NOT OPEN BEFORE SCORING")
    try:
        app = write_score_app(batch, load_rubric(), spec.get("brief") or None)
        print(f"open {app.name} in that directory to score")
    except Exception as exc:
        print(f"warning: the batch is complete but score.html was not written: {exc}")
        print("re-run bench/blind_score_app.py --batch <batch> to write it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
