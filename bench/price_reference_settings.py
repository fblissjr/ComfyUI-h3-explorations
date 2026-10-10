#!/usr/bin/env python3
"""Price a list of reference stills under a list of append-node settings.

    python bench/price_reference_settings.py --canvas 1344x768 \
        --sources 1208x1350,3840x1280,1920x1080 \
        [--out data/results/<date>_reference_settings.json]

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no model weights, no media: geometry depends only on width and height.

The question this answers is the one the owner asked on 2026-09-13: "what
happens to these three images under these settings?" Every number here comes
from the functions the conditioner calls (`reference_geometry.fit_reference_image`,
`reference_conditioning.qwen_view_size`, `reference_geometry.qwen_image_size`)
and from core's own still-image bounds read out of
`h3_encoder_loader.native_encoder_contract`, so a row cannot disagree with a
render. `MiniMaxH3ReferenceConditioning`'s preview does the same inside a graph, on the real
stills; this is the version you can run on sizes alone.

Two columns per still. The video model's copy (`size_policy`, `dit_short_edge`,
`allow_upscale`) becomes DiT reference rows attended on every sampling step.
The text encoder's copy (`qwen_view`, `qwen_short_edge`) becomes vision tokens
in the text segment ahead of the prompt. Under `shared` they are one copy.

The settings swept are the arms of `bench/refview2_arms.json`, plus `match`,
so the record and the ablation read against each other.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = Path.home() / "ComfyUI"

# ComfyUI's root ahead of the repo's PARENT, and the repo itself never on the
# path: this repo has its own `nodes.py`. `docs/comfy_notes.md` has the trap.
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(COMFY))

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

geo = importlib.import_module(f"{REPO.name}.reference_geometry")
rc = importlib.import_module(f"{REPO.name}.reference_conditioning")
report = importlib.import_module(f"{REPO.name}.reference_report")
loader = importlib.import_module(f"{REPO.name}.h3_encoder_loader")

sys.path.insert(0, str(REPO / "bench"))
from h3_producer_provenance import producer_provenance  # noqa: E402

#: The arms of `bench/refview2_arms.json`, plus `match`, in one table. The
#: node defaults (vendor parity) are the first row.
SETTINGS = {
    "parity":      dict(size_policy="max", short_edge=2048, allow_upscale=True,  qwen_short_edge=0),
    "up_q512":     dict(size_policy="max", short_edge=2048, allow_upscale=True,  qwen_short_edge=512),
    "noup_shared": dict(size_policy="max", short_edge=2048, allow_upscale=False, qwen_short_edge=0),
    "noup_q512":   dict(size_policy="max", short_edge=2048, allow_upscale=False, qwen_short_edge=512),
    "noup_q1024":  dict(size_policy="max", short_edge=2048, allow_upscale=False, qwen_short_edge=1024),
    "noup_q2048":  dict(size_policy="max", short_edge=2048, allow_upscale=False, qwen_short_edge=2048),
    "match_shared": dict(size_policy="match", short_edge=2048, allow_upscale=False, qwen_short_edge=0),
}


def _wh(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def price(sources, canvas, bounds) -> dict:
    cw, ch = canvas
    out = {}
    for name, s in SETTINGS.items():
        rows = []
        for (sw, sh) in sources:
            vw, vh = geo.fit_reference_image(
                sw, sh, size_policy=s["size_policy"], short_edge=s["short_edge"],
                allow_upscale=s["allow_upscale"], canvas_w=cw, canvas_h=ch)
            if s["qwen_short_edge"]:
                qw, qh = rc.qwen_view_size(sw, sh, s["qwen_short_edge"])
            else:
                qw, qh = vw, vh
            ew, eh = report._smart_resize(qw, qh, bounds)
            rows.append({
                "source": [sw, sh], "vae_view": [vw, vh],
                "dit_rows": geo.latent_rows(vw, vh),
                "qwen_view": [qw, qh], "encoder_applied": [ew, eh],
                "vision_tokens": report.merged_tokens(ew, eh),
                "encoder_bounds_moved_it": (ew, eh) != (qw, qh),
            })
        out[name] = {
            "settings": s, "stills": rows,
            "total_dit_rows": sum(r["dit_rows"] for r in rows),
            "total_vision_tokens": sum(r["vision_tokens"] for r in rows),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--canvas", default="1344x768")
    ap.add_argument("--sources", required=True,
                    help="comma-separated WxH still sizes, in append order")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the record here (JSON); prints a table either way")
    args = ap.parse_args()
    canvas = _wh(args.canvas)
    sources = [_wh(t) for t in args.sources.split(",") if t.strip()]
    contract = loader.native_encoder_contract()
    bounds = tuple(contract["image_bounds"])
    table = price(sources, canvas, bounds)

    print(f"canvas {canvas[0]}x{canvas[1]}; encoder still bounds {bounds[0]:,}..{bounds[1]:,} px "
          f"({contract['source']})")
    print(f"{'setting':<13} {'still':<11} {'video model sees':<18} {'DiT rows':>9}   "
          f"{'text encoder sees':<18} {'tokens':>8}")
    for name, arm in table.items():
        for r in arm["stills"]:
            flag = " *" if r["encoder_bounds_moved_it"] else ""
            print(f"{name:<13} {r['source'][0]}x{r['source'][1]:<7} "
                  f"{r['vae_view'][0]}x{r['vae_view'][1]:<13} {r['dit_rows']:>9,}   "
                  f"{r['encoder_applied'][0]}x{r['encoder_applied'][1]:<13} "
                  f"{r['vision_tokens']:>8,}{flag}")
        print(f"{'':<13} {'TOTAL':<11} {'':<18} {arm['total_dit_rows']:>9,}   {'':<18} "
              f"{arm['total_vision_tokens']:>8,}")
    print("* the encoder's own bounds moved that copy after the policy")

    if args.out:
        record = {
            "date": date.today().isoformat(),
            "question": "what each append-node setting does to these stills, both copies",
            "method": ("reference_geometry.fit_reference_image for the video model's copy, "
                       "reference_conditioning.qwen_view_size then reference_report._smart_resize "
                       "under core's own still bounds for the text encoder's copy; "
                       "merged tokens are (w//32)*(h//32)"),
            "canvas": list(canvas), "sources": [list(s) for s in sources],
            "encoder_bounds": list(bounds), "encoder_bounds_source": contract["source"],
            "producer": producer_provenance(Path(__file__)),
            "settings": table,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
