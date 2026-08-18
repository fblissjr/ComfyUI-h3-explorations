#!/usr/bin/env python3
"""Emit the Phase B workload grid as API graphs, plus a manifest of arms.

The 2026-08-18 workload grid varies, one axis at a time around a shared
centre point (1024x768, length 311, 3 image references at native size,
Sol at SOL_RECOMMENDED_CUDA, ref2va, 16 steps, er_sde):

  length   243 / 311 / 362          (the `3d`-aligned values + the shipped max)
  canvas   768x768 / 1024x768 / 1152x768 / 1344x768   (owner's pick, 2026-08-18)
  refs     0 (t2v) / 3 / 6          (count ladder, size and aspect held fixed)
  sizing   native vs 2048 short-edge upscale          (the 5x row-cost fork)

Graphs are written to the output directory as ad-hoc arms rather than into
`workflows/` -- they are one experiment's inputs, not shipped graphs, and
`build_api` is the same builder the shipped set uses, so they cannot drift
from it. The manifest JSON beside them records every arm's builder inputs.
Static token pricing is NOT computed here: run `bench/preflight_graph.py`
on each emitted graph (the 2026-08-18 run saved those as *_preflight.txt
sidecars next to the graphs) and join timing rows to arms by label.

Reference assets come from `internal/reference_library.md`'s count-ladder
set: the six 1024x1024 assets (three faces, three styles), which move count
while holding size and aspect fixed. The 3-ref arms use the first three.

Usage:
  bench/gen_phaseb_grid.py --out <dir>
  bench/run_graph_arms.py --arm <label>=<dir>/<label>.json ... --out results.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cfg = _load("h3_config", REPO / "workflows" / "h3_config.py")
bw = _load("build_workflows", REPO / "workflows" / "build_workflows.py")

# The count ladder from internal/reference_library.md: same size, same
# aspect, so a count arm varies count and nothing else.
LADDER = (
    "h3_refs/face_elderly_man_suit_1024x1024.png",
    "h3_refs/face_freckled_woman_redhair_1024x1024.png",
    "h3_refs/face_young_man_glasses_1024x1024.png",
    "h3_refs/style_impasto_floral_1024x1024.png",
    "h3_refs/style_impasto_lighthouse_storm_1024x1024.png",
    "h3_refs/style_pencil_cottage_1024x1024.png",
)

CENTRE = dict(width=1024, height=768, length=311, refs=3, upscale=False)


def arms():
    """Yield (label, overrides) with the centre point emitted once."""
    seen = set()

    def emit(label, **kv):
        key = tuple(sorted({**CENTRE, **kv}.items()))
        if key in seen:
            return None
        seen.add(key)
        return (label, {**CENTRE, **kv})

    for L in (243, 311, 362):
        a = emit(f"len{L}", length=L)
        if a:
            yield a
    for w, h in ((768, 768), (1152, 768), (1344, 768)):
        a = emit(f"cv{w}x{h}", width=w, height=h)
        if a:
            yield a
    for n in (0, 6):
        a = emit(f"refs{n}", refs=n)
        if a:
            yield a
    a = emit("up2048", upscale=True)
    if a:
        yield a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory for graphs + manifest")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for label, a in arms():
        task = "t2v" if a["refs"] == 0 else "r2v"
        kw: dict = dict(
            sage=True, sol=cfg.SOL_RECOMMENDED_CUDA,
            length=a["length"], width=a["width"], height=a["height"],
            ref_upscale=a["upscale"],
            out_prefix=f"Video/phaseb/{label}",
        )
        if task == "r2v":
            # The prompt must name exactly the sockets the graph wires --
            # labels derive from socket positions, so an unnamed reference
            # rides along as unconditioned cost. Faces are `character`;
            # the style paintings ride as `subject`, which is mechanically
            # labelled even if semantically loose -- these arms are timed,
            # not judged.
            roles = (("character",) * min(a["refs"], 3)
                     + ("subject",) * max(0, a["refs"] - 3))
            kw.update(ref_image_count=a["refs"],
                      ref_images=LADDER[: a["refs"]],
                      prompt=bw._ref_prompt(images=roles))
        g = bw.build_api(task, **kw)
        p = out / f"{label}.json"
        p.write_text(json.dumps(g, indent=1) + "\n")
        manifest.append({"label": label, "task": task, **a, "graph": str(p)})
        print(f"wrote {p.name}: {task} {a['width']}x{a['height']} "
              f"len {a['length']} refs {a['refs']}"
              f"{' upscale2048' if a['upscale'] else ' native'}")

    (out / "grid_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} arms; manifest at {out / 'grid_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
