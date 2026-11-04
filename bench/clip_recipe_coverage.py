#!/usr/bin/env python3
"""Which rendered recipes have a graph in this repo, and which have none.

    H3_OUTPUT_DIR=<share> python bench/clip_recipe_coverage.py --since 2026-09-24 --task r2v i2v
    python bench/clip_recipe_coverage.py --root <dir> --skip private --recursive

An experiment is only kept if its graph is. This reads the graph each clip was
rendered from (the first-frame PNG beside it, via `diff_clip_graphs.graph_of`),
reduces it to a recipe, groups the clips by recipe, and says which graphs under
`h3_config.graph_paths(include_bench=True)` carry the same recipe. A group with
no match prints its nearest graphs, the ones that share the checkpoint and the
LoRAs, so the difference is one line to read.

**The recipe is what decides the output, not how the graph was built.** It is
the task, the checkpoints, each LoRA with its strength, block selection and any
other input moved off its inert value (`LORA_KNOBS`), the sigma schedules
(including `pdd:N`, the schedule a PDD node emits at `steps=N` when its SIGMAS
output is wired), the sigma shift, and VSA's keep percent. It leaves out the
seed, the prompt, the images, the canvas and every filename, which are inputs
and not recipes, and it leaves out Sol-Attn and the attention chain. The last is
deliberate: Sol became the default in every video graph after the first
distill renders, so a clip from before that has a recipe that a graph today
matches except for attention, and the tool prints `sol`/`nosol` beside each
group so that difference is visible and does not hide a match.

**Limits.** It compares what the graph says, not what the code did with it (see
`diff_clip_graphs.py`). A schedule is compared as written, so `pdd:8` and
PDD8's own sigmas pinned in a `ManualSigmas` node read as two recipes though
they sample the same points. A group that matches on recipe but whose graph was built
for another scene (the market-scene reference graphs, for one) is a match here;
the prompt and images are not part of the recipe. A clip with no readable
sidecar is counted and skipped, never guessed at. Clips rendered by the
h3-mutant-distill release's `H3ExactLoRA` node match through the pack graph they
were parity checked against, by recipe, not by node class (that package and its
parity check left this tree on 2026-11-04; the release itself is untouched).

Reads PNG sidecars only. No GPU, no server.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

import orjson

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))

import h3_config as C  # noqa: E402
from diff_clip_graphs import graph_of  # noqa: E402

#: Node classes that apply a LoRA to the DiT, each with the widget that holds
#: its strength. Provenance: read from the node schemas, inherited (core's
#: `LoraLoaderModelOnly` calls it `strength_model`). `H3ExactLoRA` is the
#: mutant repo's node and is listed so its graphs, if one is passed in, reduce
#: to the same recipe as the pack node it was parity checked against.
LORA_CLASSES = {"MiniMaxH3PDDLoRA": "strength", "MiniMaxH3LoRABranch": "strength",
                "H3ExactLoRA": "strength", "LoraLoaderModelOnly": "strength_model"}

#: Inputs that change the output, beyond file, strength and blocks, with the
#: value that leaves them inert. A LoRA whose input differs carries `name=value`
#: in its recipe. Provenance: the defaults in `pdd_lora.py` and `lora_branch.py`
#: `define_schema`, inherited. `H3ExactLoRA` takes none of them and computes
#: what `MiniMaxH3PDDLoRA` computes at these defaults (the parity check that
#: showed it, `check_mutant_parity.py`, is in git), so the two reduce alike.
#: The PDD node's `steps` is not here: it decides the
#: schedule only through the node's SIGMAS output, so it lands in `sigmas`.
LORA_KNOBS = {
    "MiniMaxH3PDDLoRA": {"patch_heads": True, "nfe": 0, "head_strength": -1.0, "unmerged_blocks": "",
                         "unmerged_strength": -1.0, "unmerged_window": "", "backbone_apply": "exact branch"},
    "MiniMaxH3LoRABranch": {"modules": "all", "start_percent": 0.0, "end_percent": 1.0},
}

REF_CONDITIONING = ("MiniMaxH3ReferenceConditioning", "MiniMaxH3ReferenceToVideo")
FRAME_CONDITIONING = ("MiniMaxH3Conditioning", "MiniMaxH3ImageToVideo")


def _sigmas(text: str) -> tuple:
    try:
        return tuple(round(float(x), 6) for x in str(text).split(",") if x.strip())
    except ValueError:
        return (str(text),)


def _value(graph: dict, v):
    """A widget value, read through a link to a primitive node (the shipped PDD graphs
    wire `steps` from a PrimitiveInt). A link to anything else is kept as `link`."""
    if isinstance(v, list) and len(v) == 2:
        src = graph.get(str(v[0]), {}).get("inputs", {})
        return src["value"] if "value" in src and not isinstance(src["value"], list) else "link"
    return v


def _lora(graph: dict, cls: str, i: dict) -> str:
    strength = _value(graph, i.get(LORA_CLASSES[cls], 1.0))
    strength = f"{strength:g}" if isinstance(strength, (int, float)) else str(strength)
    knobs = [f"{k}={_value(graph, i[k])}" for k, inert in LORA_KNOBS.get(cls, {}).items()
             if k in i and _value(graph, i[k]) != inert]
    extra = f"{{{','.join(knobs)}}}" if knobs else ""
    return f"{Path(str(i.get('lora_name'))).name}@{strength}[{i.get('blocks') or 'all'}]{extra}"


def recipe(graph: dict) -> dict:
    """The recipe of an API graph: a dict of hashable fields, plus `sol`, which is shown and not matched."""
    by_class: dict[str, list] = defaultdict(list)
    for node in graph.values():
        by_class[node["class_type"]].append(node.get("inputs", {}))
    if any(c in by_class for c in REF_CONDITIONING):
        task = "r2v"
    elif any(c in by_class for c in FRAME_CONDITIONING) and "LoadImage" in by_class:
        task = "i2v"
    else:
        task = "t2v"
    loras = sorted(_lora(graph, cls, i) for cls in LORA_CLASSES for i in by_class.get(cls, []))
    # A PDD node whose SIGMAS output (index 1) feeds a node emits the schedule
    # from its `steps`; it is named, not computed, since computing it needs the
    # file's metadata. `pdd:8` and PDD8's sigmas pinned in a ManualSigmas node
    # are the same schedule and still read as different recipes.
    consumed = {str(v[0]) for n in graph.values() for v in n.get("inputs", {}).values()
                if isinstance(v, list) and len(v) == 2 and v[1] == 1}
    pdd_sigmas = [(f"pdd:{_value(graph, n.get('inputs', {}).get('steps', 8))}",)
                  for nid, n in graph.items()
                  if n["class_type"] == "MiniMaxH3PDDLoRA" and nid in consumed]
    return {
        "task": task,
        "unets": tuple(sorted(i.get("unet_name", "") for i in by_class.get("UNETLoader", []))),
        "loras": tuple(loras),
        "sigmas": tuple(sorted([_sigmas(i.get("sigmas", "")) for i in by_class.get("ManualSigmas", [])]
                               + pdd_sigmas, key=str)),
        "shift": tuple(sorted(f"{i.get('shift_video')}/{i.get('shift_audio')}"
                              for i in by_class.get("MiniMaxH3SigmaShift", []))),
        "vsa_keep": tuple(sorted(str(i.get("selection.keep_percent"))
                                 for i in by_class.get("BlockSparseAttention", []))),
        "sol": "sol" if "MiniMaxH3Sol" in by_class else "nosol",
    }


MATCHED = ("task", "unets", "loras", "sigmas", "shift", "vsa_keep")
NEAR = ("task", "unets", "loras")


def _pick(r: dict, fields) -> tuple:
    return tuple(r[f] for f in fields)


def graph_recipes() -> list[tuple[str, dict]]:
    out = []
    for path in C.graph_paths(REPO / "workflows", include_bench=True):
        try:
            graph = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            continue
        if isinstance(graph, dict) and graph and all(
                isinstance(v, dict) and "class_type" in v for v in graph.values()):
            out.append((str(path.relative_to(REPO / "workflows")), recipe(graph)))
    return out


def _collapse(names: list[str]) -> list[str]:
    """Drop `_savelat` and `_x0` twins when a base graph of the same recipe is listed."""
    base = [n for n in names if not n.removesuffix("_api.json").endswith(("_savelat", "_x0"))]
    return base or names


def _listing(names: list[str], cap: int = 6) -> str:
    """At most `cap` names, then a count: a generic reference recipe is carried by dozens of graphs."""
    return ", ".join(names[:cap]) + (f", and {len(names) - cap} more" if len(names) > cap else "")


def _short(text: str) -> str:
    return (text.replace("minimax_h3_", "").replace("_pruned_int8_convrot.safetensors", "")
            .replace("_pruned_rank64_comfy.safetensors", "").replace("_comfy.safetensors", "")
            .replace("h3/", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None,
                    help="directory of renders; default is Video/ under h3_config.output_dir()")
    ap.add_argument("--since", default="1970-01-01", help="only clips written on or after this date (YYYY-MM-DD)")
    ap.add_argument("--task", nargs="*", choices=("t2v", "i2v", "r2v"), default=None,
                    help="only these tasks (r2v is any reference graph, i2v a first or last frame)")
    ap.add_argument("--recursive", action="store_true", help="also read sub-folders of the root")
    ap.add_argument("--skip", nargs="*", default=[], metavar="PART",
                    help="skip any clip whose path under the root has a component starting with PART")
    args = ap.parse_args()

    root = args.root or (C.output_dir() / "Video")
    since = datetime.datetime.fromisoformat(args.since).timestamp()
    pngs = root.rglob("*.png") if args.recursive else root.glob("*.png")
    groups: dict[tuple, list] = defaultdict(list)
    unreadable = 0
    for png in pngs:
        rel = png.relative_to(root).parts
        if any(part.startswith(skip) for part in rel for skip in args.skip) or png.stat().st_mtime < since:
            continue
        try:
            r = recipe(graph_of(str(png)))
        except (ValueError, KeyError, orjson.JSONDecodeError):
            unreadable += 1
            continue
        if args.task and r["task"] not in args.task:
            continue
        groups[_pick(r, MATCHED) + (r["sol"],)].append((png.stat().st_mtime, png.name, r))

    graphs = graph_recipes()
    covered = uncovered = 0
    for key in sorted(groups, key=lambda k: (k[0], min(m for m, _, _ in groups[k]))):
        members = sorted(groups[key])
        r = members[0][2]
        day = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d")  # noqa: E731
        exact = _collapse([n for n, g in graphs if _pick(g, MATCHED) == _pick(r, MATCHED)])
        print(f"{r['task']}  {len(members)} clip(s)  {day(members[0][0])}..{day(members[-1][0])}  {r['sol']}  "
              f"e.g. {members[0][1]}")
        print(f"    unet {[_short(u) for u in r['unets']]}  lora {[_short(x) for x in r['loras']]}")
        print(f"    sigmas {[','.join(f'{v:g}' if isinstance(v, float) else str(v) for v in s) for s in r['sigmas']]}"
              f"  shift {list(r['shift'])}  "
              f"vsa keep {list(r['vsa_keep'])}")
        if exact:
            covered += 1
            print(f"    GRAPH: {_listing(exact)}")
        else:
            uncovered += 1
            near = _collapse([n for n, g in graphs if _pick(g, NEAR) == _pick(r, NEAR)])
            print(f"    NO GRAPH with this recipe. Same checkpoint and LoRAs: {_listing(near) or 'none'}")
    print(f"\n{covered} recipe(s) with a graph, {uncovered} without; {unreadable} clip(s) with no readable sidecar; "
          f"read {sum(len(v) for v in groups.values())} clip(s) under {root.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
