#!/usr/bin/env python3
"""Make `vendor/meta_sapiens2/` from `coderef/sapiens2`: Meta's names for its models' points and classes, and its licence.

`vendor/meta_sapiens2/` holds the text of Meta's that this pack needs to read the Sapiens2 models and that the
checkpoints themselves do not carry: the names of the pose model's points, and the names and preview colours of
the segmentation model's classes. Both checkpoints call theirs `LABEL_0` on. The text is Meta's, under the
Sapiens2 License, so it is kept in a folder of its own with that licence beside it
(`vendor/meta_sapiens2/README.md`).

    python bench/build_meta_sapiens2.py            # write the two lists and the licence from coderef/sapiens2
    python bench/build_meta_sapiens2.py --check    # say whether the folder is what this would write

It reads Meta's files as text and runs none of them.

  the pose points    `sapiens/pose/configs/_base_/keypoints308.py`: `dataset_info["keypoint_info"]`, each entry's
                     `name` in the order of its `id`, with the entries whose name starts `teeth_` left out and the
                     ones after them moved down, which is what that file does to itself when it is loaded
                     (`remove_teeth`). Each entry's `swap` is kept too: the name of its left-right partner.
  the classes        `sapiens/dense/src/datasets/seg/seg_utils.py`: the table `DOME_CLASSES_29`, each row's `name`
                     and `color` in the order of its index.
  the licence        `LICENSE.md`, byte for byte.

Each list records the upstream commit it was read at. `bench/check_meta_sapiens2.py` holds the folder to all of
this and to the checkpoints' own configs. Without `coderef/sapiens2` this returns 2.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "vendor" / "meta_sapiens2"
UPSTREAM = REPO / "coderef" / "sapiens2"
UPSTREAM_NAME = "facebookresearch/sapiens2"

POSE_SOURCE = "sapiens/pose/configs/_base_/keypoints308.py"
SEG_SOURCE = "sapiens/dense/src/datasets/seg/seg_utils.py"
SEG_TABLE = "DOME_CLASSES_29"
POSE_FILE, SEG_FILE, LICENCE = "pose_keypoints308.json", "seg_classes29.json", "LICENSE.md"
TEETH = "teeth_"


def _assigned(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise SystemExit(f"upstream no longer assigns {name} at the top of its file")


def pose_points(upstream: Path = UPSTREAM) -> tuple[list[str], dict[str, str]]:
    """Meta's names for the pose model's points in the checkpoint's order, and each name's left-right partner."""
    info = _assigned(ast.parse((upstream / POSE_SOURCE).read_text()), "dataset_info")
    given = {kw.arg: kw.value for kw in info.keywords}
    if ast.literal_eval(given["remove_teeth"]) is not True:
        raise SystemExit("upstream no longer removes the teeth points: the rule this list is made by has changed")
    rows = []
    for key, value in zip(given["keypoint_info"].keys, given["keypoint_info"].values):
        row = {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords if kw.arg in ("name", "id", "swap")}
        if ast.literal_eval(key) != row["id"] or row["id"] != len(rows):
            raise SystemExit(f"upstream's point {row} is not at its own id")
        rows.append(row)
    teeth = [r["id"] for r in rows if r["name"].startswith(TEETH)]
    if teeth != list(range(teeth[0], teeth[-1] + 1)):
        raise SystemExit("upstream's teeth points are no longer one run of ids")
    kept = [r for r in rows if not r["name"].startswith(TEETH)]
    return [r["name"] for r in kept], {r["name"]: r["swap"] for r in kept if r["swap"]}


def seg_classes(upstream: Path = UPSTREAM) -> tuple[list[str], list[list[int]]]:
    """Meta's names and preview colours for the segmentation model's classes, by index."""
    table = _assigned(ast.parse((upstream / SEG_SOURCE).read_text()), SEG_TABLE)
    rows = ast.literal_eval(table.args[0])
    if [i for i, _ in rows] != list(range(len(rows))):
        raise SystemExit(f"upstream's {SEG_TABLE} is no longer in the order of its indexes")
    return [r["name"] for _, r in rows], [[int(v) for v in r["color"]] for _, r in rows]


def commit(upstream: Path = UPSTREAM) -> str:
    done = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def _rows(items: list, indent: str = "  ") -> str:
    return "[\n" + ",\n".join(indent + json.dumps(i) for i in items) + "\n ]"


def pose_text(names: list[str], swap: dict[str, str], at: str) -> str:
    """The pose list's file: one name a line, so a change reads as a line."""
    pairs = "{\n" + ",\n".join(f"  {json.dumps(a)}: {json.dumps(b)}" for a, b in swap.items()) + "\n }"
    return ("{\n"
            ' "what": "Meta\'s names for the Sapiens2 pose model\'s points, in the checkpoint\'s order, with each name\'s left-right partner",\n'
            f' "upstream": {json.dumps(UPSTREAM_NAME)},\n "commit": {json.dumps(at)},\n "read_from": {json.dumps(POSE_SOURCE)},\n'
            ' "how": "dataset_info[\\"keypoint_info\\"]: each entry\'s name in the order of its id, the entries named teeth_ left out and the '
            'ones after them moved down, as that file does to itself (remove_teeth); swap is each entry\'s own left-right partner",\n'
            f' "names": {_rows(names)},\n "swap": {pairs}\n' "}\n")


def seg_text(names: list[str], colors: list[list[int]], at: str) -> str:
    return ("{\n"
            ' "what": "Meta\'s names and preview colours for the Sapiens2 segmentation model\'s classes, by index",\n'
            f' "upstream": {json.dumps(UPSTREAM_NAME)},\n "commit": {json.dumps(at)},\n "read_from": {json.dumps(SEG_SOURCE)},\n'
            f' "table": {json.dumps(SEG_TABLE)},\n "names": {_rows(names)},\n "colors": {_rows(colors)}\n' "}\n")


def built(at: str | None = None) -> dict[str, bytes]:
    """Every file of Meta's in the folder, as this would write it. `at` is the commit to record; else the checkout's."""
    at = commit() if at is None else at
    return {POSE_FILE: pose_text(*pose_points(), at).encode(), SEG_FILE: seg_text(*seg_classes(), at).encode(),
            LICENCE: (UPSTREAM / LICENCE).read_bytes()}


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--check", action="store_true", help="write nothing; say whether the folder is what this would write")
    v = a.parse_args()
    if not (UPSTREAM / POSE_SOURCE).exists():
        print(f"{UPSTREAM.relative_to(REPO)} is not checked out: nothing to read")
        return 2
    files = built()
    if v.check:
        differ = [name for name, data in files.items() if not (FOLDER / name).is_file() or (FOLDER / name).read_bytes() != data]
        print("the folder is what this would write" if not differ else "these differ from what this would write: " + ", ".join(differ))
        return 1 if differ else 0
    FOLDER.mkdir(exist_ok=True)
    for name, data in files.items():
        (FOLDER / name).write_bytes(data)
        print(f"wrote {(FOLDER / name).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
