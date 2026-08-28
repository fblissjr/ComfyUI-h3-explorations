#!/usr/bin/env python3
"""No graph or constant may name a model file its real loader cannot use.

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

`/object_info` is the authority for discovery, not the filesystem. A file
present on disk that ComfyUI does not index is unusable. Discovery is not
format acceptance, though: on 2026-08-23 core `CLIPLoader` offered the owner's
compressed-tensors AWQ file, then misdetected its full Hugging Face namespace
as ordinary Qwen3-VL and could not load it. That escaped this check's original
"offered means usable" claim. The W4A16 file therefore has one explicit owner,
this repo's `MiniMaxH3AWQEncoderLoader`; naming it under core is red even while
core's menu contains it. Symlinks into other volumes remain normal.

  resolves    every `*.safetensors` name in a shipped graph is one the live
              server offers for THAT node class. A VAE name under `UNETLoader`
              is a failure even though both files exist.
  constants   every model name in `h3_config.MODELS`, plus `IMAGE_VAE`, is
              offered by the class that loads it.
  format owner the custom compressed-tensors W4A16 artifact is loaded only by
              the repo adapter that recognizes and repacks that format.

## Scope

Discovery is `h3_config.graph_paths(include_bench=True)`: every directory in
`GRAPH_DIRS` AND `workflows/bench/`. Naming them here would be a second copy;
`GRAPH_DIRS` is `("",)` since the single-frame lane was parked on 2026-08-27,
and it was `("", "image")` before that. The bench directory sits outside
`GRAPH_DIRS` because it is exempt from *schema* grading -- its stamped graphs
read another pack's closure internals and are expected to break against the
live schema. That exemption does not extend here. A bench graph naming a file
that no longer exists is not schema drift, it is the same broken graph a
shipped one would be, and on 2026-08-21 the three stamped graphs carried the
deleted video VAE exactly as the shipped ones did.

`workflows/archive/` stays out: it is history, and history is allowed to name
files that are gone.

## Controls

Deliberate controls run on every invocation, because a corpus in which
every name is already valid cannot tell a working check from an inert one:

  control:missing   a synthetic graph naming a file no server offers -> red
  control:wrongclass a synthetic graph naming a REAL vae under UNETLoader -> red
  control:bothforms both combo declaration forms are read, on a fixture
  control:format-owner core offering the AWQ filename does not make core a
                       compatible loader
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

from h3_config import CORE_LOADED_ENCODERS, ENCODER_V1, MODELS, IMAGE_VAE, graph_paths  # noqa: E402

WORKFLOWS = _HERE.parents[0] / "workflows"
DEFAULT_URL = "http://127.0.0.1:8188"

# Which loader class owns each h3_config model constant.
CONSTANT_CLASS = {
    "unet": "UNETLoader", "unet_fl2va": "UNETLoader", "unet_ref2va": "UNETLoader",
    "unet_hybrid_b30": "UNETLoader", "unet_hybrid_adaln_all": "UNETLoader",
    # Resolved per file below, not pinned: the shipped encoder became a
    # ComfyUI-native build on 2026-08-27 and this map is what said
    # otherwise.
    "clip": None,
    "video_vae": "VAELoader", "audio_vae": "VAELoader",
}
AWQ_LOADER = "MiniMaxH3AWQEncoderLoader"


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


def grade_format_owner(items) -> list[str]:
    """The escaped case: a filesystem menu entry is not loader support.

    **Branches on the FILE, not on which constant happens to hold it.** This
    read `MODELS["clip"]` and demanded the adapter for whatever that named,
    which was correct only while the shipped encoder was always a W4A16
    artifact. When it became the ComfyUI-native INT8 build on 2026-08-27 the
    check inverted: it declared a native file "compressed-tensors AWQ" and
    demanded the adapter that cannot open it. Same defect as the generator's
    loader choice the same evening, and the same rule -- branch on the
    observable, which here is membership of `CORE_LOADED_ENCODERS`.
    """
    bad = []
    # Encoder loaders only. A VAELoader naming a VAE is not this rule's
    # business, and scoping by CORE_LOADED_ENCODERS membership alone made every
    # non-encoder file look like an adapter artifact.
    encoder_loaders = {"CLIPLoader", AWQ_LOADER}
    for where, cls, name in items:
        if cls not in encoder_loaders:
            continue
        native = name in CORE_LOADED_ENCODERS
        if native and cls == AWQ_LOADER:
            bad.append(
                f"{where}: {name!r} is a ComfyUI-native encoder and must use "
                f"CLIPLoader, not {AWQ_LOADER}, which opens only "
                "compressed-tensors W4A16 artifacts."
            )
        elif not native and cls != AWQ_LOADER:
            bad.append(
                f"{where}: {name!r} is compressed-tensors AWQ and must use "
                f"{AWQ_LOADER}, not {cls}. Core menu discovery is not format "
                "support."
            )
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
    for p in graph_paths(WORKFLOWS, include_bench=True):
        for nid, cls, name in names_in_graph(json.loads(p.read_text())):
            items.append((f"{p.relative_to(WORKFLOWS.parent)}#{nid}", cls, name))
    for key, cls in CONSTANT_CLASS.items():
        if key in MODELS:
            resolved = cls or ("CLIPLoader" if MODELS[key] in CORE_LOADED_ENCODERS
                               else AWQ_LOADER)
            items.append((f"h3_config.MODELS[{key!r}]", resolved, MODELS[key]))
    items.append(("h3_config.IMAGE_VAE", "VAELoader", IMAGE_VAE))

    failures = grade(items, oi) + grade_format_owner(items)
    print(f"{len(items)} model reference(s) from "
          f"{len(graph_paths(WORKFLOWS, include_bench=True))} graph(s) "
          f"(shipped and bench) and h3_config")

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
    # Must name an artifact the adapter really owns. Using MODELS["clip"]
    # stopped firing the moment the shipped encoder became native --
    # the control was asserting a pairing that is now correct.
    core_awq = [("control", "CLIPLoader", ENCODER_V1)]
    controls.insert(3, ("control:format-owner", bool(grade_format_owner(core_awq))))

    ok = True
    verdict = {"control:missing": ("went red as it must", "did NOT fire"),
               "control:wrongclass": ("went red as it must", "did NOT fire"),
               "control:bothforms": ("reads both combo forms",
                                     "misses a combo form, so a green means "
                                     "nothing"),
               "control:format-owner": ("rejects core's false-positive menu entry",
                                         "accepted discovery as format support")}
    for name, fired in controls[:4]:
        print(f"  {'ok  ' if fired else 'FAIL'} {name} {verdict[name][0 if fired else 1]}")
        ok &= fired
    if controls[4][1]:
        print("  FAIL control:empty  nothing was collected, so a pass here "
              "would mean nothing")
        ok = False
    else:
        print("  ok   control:empty  the corpus is non-empty")

    for f in failures:
        print(f"  FAIL {f}")
    if failures:
        print(f"\n{len(failures)} reference(s) name a missing file or use a "
              "loader that does not own its format. Either fails only after "
              "the graph reaches that loader.")
    elif ok:
        print("  ok   resolves  every reference is offered by its own class")
    return 0 if ok and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
