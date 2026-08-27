#!/usr/bin/env python3
"""What `qwen_short_edge` actually buys, under each encoder snapshot's bounds.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no model weights, no media.

The question this settles. `MiniMaxH3AppendRefImage.qwen_short_edge` sizes the
encoder-only view of a reference; the loaded artifact's snapshot then bounds
that view. Those are two stages and the second can erase the first. On
2026-08-27 `h3_config` asserted that under the v1 snapshot the knob "could not
do anything" -- true, and stated without the numbers that make it checkable or
tell you how far it generalizes. This produces them.

It restates no resize rule. The view comes from
`reference_conditioning.qwen_view_size` and the bounds application from
`reference_geometry.qwen_image_size`, which imports `smart_resize` from the
installed processor. The bounds themselves come from
`h3_awq_encoder.snapshot_contract` against the real `ARTIFACT_SNAPSHOTS`
registry, so a snapshot that moves on disk moves here.

Where the bounds are applied differs from where they are *declared*, and it
does not change this arithmetic. Every shipped graph wires `image_policy`
`comfy`, so the node pre-applies nothing and the encoder's own
`preprocess_embed` -- installed by the AWQ adapter from the snapshot -- applies
them instead. Under `encoder` or `release` the node pre-applies them and the
encoder finds nothing left to do. One `smart_resize` with one set of bounds
either way, so the merged-token count is the same and the stage that owns it
is not.

Merged tokens are `(w // 32) * (h // 32)`: patch 16 with merge 2. That is the
count that lands in the TEXT segment ahead of the prompt, which is why it is
the number reported rather than pixels.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = Path.home() / "ComfyUI"

# ComfyUI's root ahead of the repo's PARENT, and the repo itself never on the
# path: this repo has its own `nodes.py`, so a bare `import nodes` inside
# `comfy_extras` finds ours and dies on a relative import. The trap is in
# `docs/comfy_notes.md` and it bit this file on its first run. The repo's
# modules use relative imports, so they are reached as a package through
# `importlib`, the same way `bench/check_reference_runtime.py` reaches them.
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(COMFY))

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

awq = importlib.import_module(f"{REPO.name}.h3_awq_encoder")
qwen_view_size = importlib.import_module(
    f"{REPO.name}.reference_conditioning").qwen_view_size
qwen_image_size = importlib.import_module(
    f"{REPO.name}.reference_geometry").qwen_image_size

sys.path.insert(0, str(REPO / "bench"))
from h3_producer_provenance import producer_provenance  # noqa: E402

#: Aspect ratios a reference plausibly arrives at. Square is kept because it is
#: the only one v1's band does not flatten, which is the exception that shows
#: the mechanism is the band width and not the knob.
ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "4:3": (4, 3), "1:1": (1, 1)}

#: What the graphs can ask for. 2048 is `REF_IMAGE_SHORT_EDGE`, the stage-one
#: size an upscaling reference reaches and therefore what `qwen_short_edge` 0
#: hands the encoder; 512 is `REF_QWEN_SHORT_EDGE` as shipped 2026-08-27.
SHORT_EDGES = (512, 1024, 2048)

#: Source big enough that no sweep value enlarges it, so the sweep measures the
#: knob rather than the source's ceiling.
SOURCE_SHORT_EDGE = 4096


def _snapshots() -> dict[str, dict]:
    """Both live snapshots, by the registry rather than by literal bounds."""
    out = {"v1": awq.snapshot_contract(None)}
    v2_dir = awq.ARTIFACT_SNAPSHOTS[
        "qwen3vl_32b_minimax_h3_w4a16_awq_v2-comfy.safetensors"]
    if v2_dir is not None and Path(v2_dir).is_dir():
        out["v2"] = awq.snapshot_contract(v2_dir)
    return out


def _merged_tokens(width: int, height: int, contract: dict) -> tuple[int, int, int]:
    """`(view_w, view_h, merged_tokens)` after the snapshot's bounds."""
    bounded_w, bounded_h = qwen_image_size(width, height, "encoder", contract)
    geometry = contract["image_geometry"]
    factor = int(geometry["patch_size"]) * int(geometry["merge_size"])
    return bounded_w, bounded_h, (bounded_w // factor) * (bounded_h // factor)


def main() -> int:
    snapshots = _snapshots()
    missing = [name for name in ("v1", "v2") if name not in snapshots]

    rows = []
    for aspect, (aw, ah) in ASPECTS.items():
        if aw >= ah:
            src_h, src_w = SOURCE_SHORT_EDGE, round(SOURCE_SHORT_EDGE * aw / ah)
        else:
            src_w, src_h = SOURCE_SHORT_EDGE, round(SOURCE_SHORT_EDGE * ah / aw)
        for short_edge in SHORT_EDGES:
            view_w, view_h = qwen_view_size(src_w, src_h, short_edge)
            row = {"aspect": aspect, "source": [src_w, src_h],
                   "qwen_short_edge": short_edge, "view": [view_w, view_h]}
            for name, contract in snapshots.items():
                bw, bh, tokens = _merged_tokens(view_w, view_h, contract)
                row[name] = {"bounded_view": [bw, bh], "merged_tokens": tokens}
            rows.append(row)

    spread = {}
    for name in snapshots:
        per_aspect = {}
        for aspect in ASPECTS:
            counts = sorted({r[name]["merged_tokens"]
                             for r in rows if r["aspect"] == aspect})
            per_aspect[aspect] = {
                "distinct_token_counts": counts,
                "knob_is_inert": len(counts) == 1,
            }
        per_aspect["inert_for_every_aspect"] = all(
            v["knob_is_inert"] for k, v in per_aspect.items() if k in ASPECTS)
        spread[name] = per_aspect

    record = {
        "question": "does qwen_short_edge change the encoder's view, under each "
                    "artifact snapshot's declared still-image bounds",
        "method": "repo functions only: reference_conditioning.qwen_view_size for "
                  "the view, reference_geometry.qwen_image_size for the bounds "
                  "(smart_resize from the installed processor), "
                  "h3_awq_encoder.snapshot_contract for the bounds themselves. "
                  "No media: geometry depends only on width and height.",
        "path_policy": "logical identifiers only",
        "producer": producer_provenance(__file__),
        "source_short_edge": SOURCE_SHORT_EDGE,
        "bounds": {name: list(c["image_bounds"]) for name, c in snapshots.items()},
        "snapshot_sources": {name: c["source"] for name, c in snapshots.items()},
        "snapshots_not_installed": missing,
        "rows": rows,
        "spread": spread,
    }

    out = REPO / f"bench/results/{date.today().isoformat()}_qwen_view_under_snapshot.json"
    out.write_text(json.dumps(record, indent=1) + "\n")

    for name, contract in snapshots.items():
        lo, hi = contract["image_bounds"]
        print(f"\n{name}  bounds {lo}..{hi}  ({contract['source']})")
        print(f"  {'aspect':>7} " + " ".join(f"{se:>7}" for se in SHORT_EDGES)
              + "   inert")
        for aspect in ASPECTS:
            counts = [r[name]["merged_tokens"]
                      for r in rows if r["aspect"] == aspect]
            inert = spread[name][aspect]["knob_is_inert"]
            print(f"  {aspect:>7} " + " ".join(f"{c:>7}" for c in counts)
                  + f"   {'yes' if inert else 'no'}")
    if missing:
        print(f"\nnot installed, so not measured: {', '.join(missing)}")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
