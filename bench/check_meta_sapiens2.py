#!/usr/bin/env python3
"""`vendor/meta_sapiens2/` is Meta's text as Meta wrote it: its licence, and its names for the Sapiens2 models' points and classes.

The Sapiens2 checkpoints call their points and classes `LABEL_0` on; the names are in Meta's code. This pack
keeps the two lists it reads in `vendor/meta_sapiens2/`, with Meta's licence beside them
(`bench/build_meta_sapiens2.py` says how the lists are made). A list that is off by one
reads as a face that is slightly wrong and nothing says so, so each copy is held to its source.

1. **The licence is upstream's file.** `vendor/meta_sapiens2/LICENSE.md` has the hash recorded here, and is byte for
   byte the reference checkout's file where that is present.
2. **The pose list is what upstream's file gives.** Rebuilt from the reference checkout by the builder's rule
   (the entries of `keypoint_info` in the order of their ids, the teeth left out as that file leaves them out),
   the names, their left-right partners and the file's own text are the tracked ones. Not graded without the
   checkout.
3. **The pose list holds together.** Every name once; every left-right partner is in the list and points
   back; no teeth; and every name the shared figures (`bench/_lib/face_points.py`, through its fixture) look
   for is there. Needs nothing.
4. **The pose list's pairs are the checkpoint's.** The model's own config has no names, but it has
   `flip_pairs`, which point is the mirror of which: the list's partners, as positions, are that set, and the
   list is as long as the config's labels. This holds the order to the checkpoint and is not a proof of it:
   it sees a list moved along and two points of different pairs changing places, and it does not see the left
   and the right of one pair changing places, nor two unpaired points. Case 2 sees those. Not graded where the
   model is not found: it is looked for in the folder `H3_POSE_MODEL_FOLDER` names, then under ComfyUI's
   `models/sapiens2/`, then beside wherever the Sapiens2 models linked from there are kept.
5. **The class table is upstream's.** Names and colours are the reference checkout's `DOME_CLASSES_29`, row
   for row, and the file's text is what the builder writes. Not graded without the checkout.
6. **The class table is the node's.** `CLASS_NAMES` and `PALETTE` in `sapiens2_parts.py` are read as text:
   where they are written out there, they are this table, index for index; where they are not, that file names
   `seg_classes29.json`, which is where it takes them from.

No model is loaded, no card, no server.

    <comfy venv python> bench/check_meta_sapiens2.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import COMFY, REPO, case, finish, skip  # noqa: E402
from _lib import face_points_fixture as fx  # noqa: E402

import build_meta_sapiens2 as build  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

FOLDER = build.FOLDER
#: sha256 of `LICENSE.md` as upstream ships it at the commit the lists record. Inherited: read off the reference
#: checkout's file on 2026-10-11. Meta may change the agreement; a new file is a new hash here.
LICENCE_SHA256 = "1c77a629ee16972ddb9844378038b20321143a6faf0cac26c2c04dc9b61c5288"
#: The environment variable that names the pose model's folder, for a store that is not under ComfyUI's models.
MODEL_ENV = "H3_POSE_MODEL_FOLDER"


def upstream() -> Path:
    if not (build.UPSTREAM / build.POSE_SOURCE).exists():
        skip("coderef/sapiens2 is not checked out: the folder was not compared with upstream's files")
    return build.UPSTREAM


def table(name: str) -> dict:
    return json.loads((FOLDER / name).read_text())


def the_licence() -> str:
    data = (FOLDER / build.LICENCE).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    assert got == LICENCE_SHA256, f"vendor/meta_sapiens2/LICENSE.md is not the file recorded here: its hash is {got}"
    there = build.UPSTREAM / build.LICENCE
    if not there.exists():
        return "it has the recorded hash; the reference checkout is not present, so it was not compared with upstream's file"
    assert data == there.read_bytes(), "vendor/meta_sapiens2/LICENSE.md is no longer the reference checkout's file, byte for byte"
    return "it has the recorded hash and is the reference checkout's file byte for byte"


def the_pose_list_is_upstreams() -> str:
    up = upstream()
    names, swap = build.pose_points(up)
    mine = table(build.POSE_FILE)
    assert mine["names"] == names, f"the tracked names are not upstream's: first difference at position {next((i for i, (a, b) in enumerate(zip(mine['names'], names)) if a != b), min(len(names), len(mine['names'])))}"
    assert mine["swap"] == swap, "the tracked left-right partners are not upstream's"
    assert (FOLDER / build.POSE_FILE).read_text() == build.pose_text(names, swap, mine["commit"]), "the file is not what the builder writes: it was edited by hand"
    assert mine["read_from"] == build.POSE_SOURCE and mine["upstream"] == build.UPSTREAM_NAME
    now = build.commit(up)
    return (f"{len(names)} names and {len(swap)} partners are what upstream's file gives, and the file is the builder's text"
            + ("" if now == mine["commit"] else f"; the checkout now stands at {now[:8] or 'a commit git could not read'} and the list records {mine['commit'][:8]}"))


def the_pose_list_holds_together() -> str:
    mine = table(build.POSE_FILE)
    names, swap = mine["names"], mine["swap"]
    assert len(set(names)) == len(names), "a name is in the list twice"
    assert not [n for n in names if n.startswith(build.TEETH)], "a teeth point is in the list"
    assert all(a in names and b in names and swap.get(b) == a and a != b for a, b in swap.items()), "a left-right partner is not in the list or does not point back"
    missing = [n for n in fx.NAMES if n not in names]
    assert not missing, f"the shared figures look for points the list does not name: {missing}"
    assert len(mine["commit"]) == 40, mine["commit"]
    return f"{len(names)} names, each once; {len(swap) // 2} left-right pairs that point back at each other; all {len(fx.NAMES)} names the shared figures look for are there"


def model_folder() -> tuple[Path, str]:
    """The pose model's folder and which of the three places it was found in, for the case's line."""
    given = os.environ.get(MODEL_ENV)
    shelf = COMFY / "models" / "sapiens2"
    beside = sorted({p.resolve().parent / h3_config.POSE_MODEL for p in shelf.iterdir()}) if shelf.is_dir() else []     # where its linked models are kept
    places = ([(Path(given), f"in the folder {MODEL_ENV} names")] if given else []) + [(shelf / h3_config.POSE_MODEL, "under ComfyUI's models/sapiens2")]
    places += [(folder, "beside the Sapiens2 models linked from ComfyUI's models/sapiens2") for folder in beside]
    for folder, where in places:
        if (folder / "config.json").exists():
            return folder, where
    skip(f"the pose model was not found: {MODEL_ENV} names no folder with a config.json, and there is no {h3_config.POSE_MODEL} under ComfyUI's "
         "models/sapiens2 or beside the models linked from there")


def the_pairs_are_the_checkpoints() -> str:
    folder, where = model_folder()
    config = json.loads((folder / "config.json").read_text())
    mine = table(build.POSE_FILE)
    at = {n: i for i, n in enumerate(mine["names"])}
    listed = {tuple(sorted((at[a], at[b]))) for a, b in mine["swap"].items()}
    own = {tuple(sorted(int(i) for i in pair)) for pair in config.get("flip_pairs") or []}
    assert own, "the model's config has no flip_pairs: the one thing the list's order was held to is gone"
    assert len(config["id2label"]) == len(mine["names"]), f"the model has {len(config['id2label'])} points and the list {len(mine['names'])}"
    assert listed == own, f"the list's left-right pairs are not the checkpoint's: {len(listed - own)} of the list's are not its, {len(own - listed)} of its are not the list's"
    return f"the list's {len(listed)} left-right pairs, as positions, are the checkpoint's own flip_pairs, and both have {len(at)} points; the model was found {where}"


def the_class_table_is_upstreams() -> str:
    up = upstream()
    names, colors = build.seg_classes(up)
    mine = table(build.SEG_FILE)
    assert mine["names"] == names, "the tracked class names are not upstream's " + build.SEG_TABLE
    assert mine["colors"] == colors, "the tracked class colours are not upstream's " + build.SEG_TABLE
    assert (FOLDER / build.SEG_FILE).read_text() == build.seg_text(names, colors, mine["commit"]), "the file is not what the builder writes: it was edited by hand"
    assert mine["read_from"] == build.SEG_SOURCE and mine["table"] == build.SEG_TABLE and mine["commit"] == table(build.POSE_FILE)["commit"]
    return f"{len(names)} names and colours are upstream's {build.SEG_TABLE} row for row, and the file is the builder's text"


def the_class_table_is_the_nodes() -> str:
    mine = table(build.SEG_FILE)
    assert len(mine["names"]) == len(mine["colors"]) == len(set(mine["names"])) and all(len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c) for c in mine["colors"])
    text = (REPO / "sapiens2_parts.py").read_text()
    said = []
    for name, want in (("CLASS_NAMES", tuple(mine["names"])), ("PALETTE", tuple(tuple(c) for c in mine["colors"]))):
        value = next((n.value for n in ast.parse(text).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)), None)
        assert value is not None, f"sapiens2_parts.py no longer assigns {name}"
        try:
            written = ast.literal_eval(value)
        except ValueError:
            assert build.SEG_FILE in text and FOLDER.name in text, f"sapiens2_parts.py neither writes {name} out nor names vendor/meta_sapiens2/{build.SEG_FILE}"
            said.append(f"{name} is not written out in the node, which names the file")
            continue
        assert tuple(written) == want, f"sapiens2_parts.py's {name} is not the table in vendor/meta_sapiens2/{build.SEG_FILE}, index for index"
        said.append(f"{name} as written out in the node is the file's, {len(want)} for {len(want)}")
    return "; ".join(said)


case("the licence is upstream's file", the_licence)
case("the pose list is what upstream's file gives", the_pose_list_is_upstreams)
case("the pose list holds together", the_pose_list_holds_together)
case("the pose list's pairs are the checkpoint's", the_pairs_are_the_checkpoints)
case("the class table is upstream's", the_class_table_is_upstreams)
case("the class table is the node's", the_class_table_is_the_nodes)
sys.exit(finish())
