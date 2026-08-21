#!/usr/bin/env python3
"""No graph and no constant may name a model file ComfyUI cannot offer.

## The escape this exists for

On 2026-08-21 the owner deleted `minimax_h3_video_vae_int8_convrot.safetensors`
from disk. `workflows/h3_config.py` still named it, so every video graph the
generator wrote pointed at a file that was gone -- and
`build_workflows.py --validate` reported **113 graphs validated against
object_info: ok** minutes later. Its validation checks node classes, input
names and link shapes; it does not check a widget's VALUE against the combo
options `/object_info` lists for that input. Nothing else looked either.
`check_lora_alpha.py` resolves `*_LORA` constants on disk and nothing else,
by design.

So the failure was: a rename or a deletion anywhere in the model directories
leaves every graph broken and every check green, until a render fails at the
loader.

## What it asserts

`/object_info` is the authority, not the filesystem. A file present on disk
that ComfyUI does not index is unusable, and a name ComfyUI offers is usable
whatever the disk layout underneath -- symlinks into other volumes are the
normal case in this install.

  resolves    every `*.safetensors` name in a shipped graph is one the live
              server offers for THAT node class. A VAE name under `UNETLoader`
              is a failure even though both files exist.
  constants   every model name in `h3_config.MODELS`, plus `IMAGE_VAE`, is
              offered by the class that loads it.

## Scope, stated because it is smaller than it looks

Discovery is `h3_config.graph_paths()`, so this covers `workflows/` and
`workflows/image/` -- the shipped set. **`workflows/bench/` is NOT covered**,
because `GRAPH_DIRS` deliberately excludes it, and the stamped bench graphs
carried the deleted VAE too. If a bench graph names a missing file, this check
stays green. Fixing that means changing what `GRAPH_DIRS` means, which is a
larger decision than this check.

## Controls

Four deliberate checks run on every invocation, because a corpus in which
every name is already valid cannot tell a working check from an inert one:

  control:missing   a synthetic graph naming a file no server offers -> red
  control:wrongclass a synthetic graph naming a REAL vae under UNETLoader -> red
  control:bothforms both combo declaration forms are read, on a fixture
  control:empty     an empty corpus must not report success

Needs a live ComfyUI. With none it SKIPS loudly and exits 2, because the only
authority it has is the server's.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "workflows"))

from h3_config import MODELS, IMAGE_VAE, graph_paths  # noqa: E402

WORKFLOWS = _HERE.parents[0] / "workflows"
DEFAULT_URL = "http://127.0.0.1:8188"

# Which loader class owns each h3_config model constant.
CONSTANT_CLASS = {
    "unet": "UNETLoader", "unet_fl2va": "UNETLoader", "unet_ref2va": "UNETLoader",
    "unet_hybrid_b30": "UNETLoader", "unet_hybrid_adaln_all": "UNETLoader",
    "clip": "CLIPLoader", "video_vae": "VAELoader", "audio_vae": "VAELoader",
}


def object_info(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/object_info", timeout=30) as fh:
        return json.load(fh)


def offered(oi: dict, cls: str) -> set[str]:
    """Every combo value the server offers for a class, across its inputs.

    **Two declaration forms coexist and both have to be read.** The legacy one
    puts the options in slot 0 (`[["a.safetensors", ...], {...}]`); the v3 one
    puts the string `"COMBO"` there and the options under `options` in the spec
    dict. On this install 196 inputs use the second form, `tiny_vae` on
    `ModelPreviewOverrideKJ` among them -- and the first version of this check
    read only the first form, so it reported 55 shipped graphs naming a
    `taeh3.safetensors` the server "does not offer" while the server was
    offering it. A check that cries wolf over correct state is worse than no
    check, so this reads both and the fixture below pins it.
    """
    spec = oi.get(cls)
    if spec is None:
        return set()
    out: set[str] = set()
    for section in ("required", "optional"):
        for _name, decl in (spec.get("input", {}).get(section) or {}).items():
            if not decl:
                continue
            if isinstance(decl[0], list):
                out |= {v for v in decl[0] if isinstance(v, str)}
            elif decl[0] == "COMBO" and len(decl) > 1 and isinstance(decl[1], dict):
                out |= {v for v in (decl[1].get("options") or [])
                        if isinstance(v, str)}
    return out


def names_in_graph(g: dict) -> list[tuple[str, str, str]]:
    """(node_id, class, filename) for every model-looking widget value."""
    found = []
    nodes = g.get("nodes") if isinstance(g.get("nodes"), list) else None
    if nodes is not None:                                   # UI format
        for n in nodes:
            vals = n.get("widgets_values") or []
            vals = vals.values() if isinstance(vals, dict) else vals
            for v in vals:
                if isinstance(v, str) and v.endswith(".safetensors"):
                    found.append((str(n.get("id")), n.get("type", "?"), v))
        return found
    for nid, node in g.items():                             # API format
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, str) and v.endswith(".safetensors"):
                found.append((nid, node["class_type"], v))
    return found


def grade(items, oi: dict) -> list[str]:
    """Every (where, class, name) that the server does not offer for its class."""
    bad = []
    for where, cls, name in items:
        if name not in offered(oi, cls):
            bad.append(f"{where}: {cls} names {name!r}, which the server does "
                       f"not offer for that class")
    return bad


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    try:
        oi = object_info(base)
    except (urllib.error.URLError, OSError) as exc:
        print(f"SKIP  no ComfyUI at {base} ({exc.__class__.__name__}). The "
              "server's own offerings are the only authority this check has, "
              "so it asserts nothing rather than falling back to the disk.")
        return 2

    items = []
    for p in graph_paths(WORKFLOWS):
        for nid, cls, name in names_in_graph(json.loads(p.read_text())):
            items.append((f"{p.relative_to(WORKFLOWS.parent)}#{nid}", cls, name))
    for key, cls in CONSTANT_CLASS.items():
        if key in MODELS:
            items.append((f"h3_config.MODELS[{key!r}]", cls, MODELS[key]))
    items.append(("h3_config.IMAGE_VAE", "VAELoader", IMAGE_VAE))

    failures = grade(items, oi)
    print(f"{len(items)} model reference(s) from "
          f"{len(graph_paths(WORKFLOWS))} shipped graph(s) and h3_config")

    both_forms = {"X": {"input": {
        "required": {"legacy": [["legacy_only.safetensors"], {}]},
        "optional": {"v3": ["COMBO", {"options": ["v3_only.safetensors"]}]}}}}
    seen = offered(both_forms, "X")

    controls = []
    fake = [("control", "VAELoader", "no_such_model_zzz.safetensors")]
    controls.append(("control:missing", bool(grade(fake, oi))))
    wrong = [("control", "UNETLoader", MODELS["video_vae"])]
    controls.append(("control:wrongclass", bool(grade(wrong, oi))))
    controls.append(("control:empty", not items))
    controls.insert(2, ("control:bothforms",
                        seen == {"legacy_only.safetensors", "v3_only.safetensors"}))

    ok = True
    verdict = {"control:missing": ("went red as it must", "did NOT fire"),
               "control:wrongclass": ("went red as it must", "did NOT fire"),
               "control:bothforms": ("reads both combo forms",
                                     "misses a combo form, so a green means "
                                     "nothing")}
    for name, fired in controls[:3]:
        print(f"  {'ok  ' if fired else 'FAIL'} {name} {verdict[name][0 if fired else 1]}")
        ok &= fired
    if controls[3][1]:
        print("  FAIL control:empty  nothing was collected, so a pass here "
              "would mean nothing")
        ok = False
    else:
        print("  ok   control:empty  the corpus is non-empty")

    for f in failures:
        print(f"  FAIL {f}")
    if failures:
        print(f"\n{len(failures)} reference(s) name a file the server cannot "
              "load. A graph that names a missing model fails at the loader, "
              "after the queue and after the model load it did manage.")
    elif ok:
        print("  ok   resolves  every reference is offered by its own class")
    return 0 if ok and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
