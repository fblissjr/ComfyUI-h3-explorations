#!/usr/bin/env python3
"""Independent read of #36's gate dial (`2026-09-29_fasth3_gate_dial.md`).

Three things, each from the saved final latents and the measure records, none
from the write-up:

- the write-up's hf, moved share, detail and chroma table against the
  `2026-09-29_dial_<scene>_{resolution,tone}.json` rows;
- `torch.equal` on the saved video and audio latents: scale 0 against the
  no-gates arm (the write-up's D3), alpha 1 through the loader against the FastH3
  checkpoint arm, and two controls that must be unequal (scale 0 against the
  FastH3 checkpoint arm, scale 0.75 against alpha 1);
- how far each scale's latent path sits from alpha 1's
  (`2026-09-29_dial_look_anchor_divergence.json`, mean of `rel_l2`).

CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_gate_dial_reanalysis.py \\
        --latents <output>/latents --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file

RESULTS = Path(__file__).resolve().parent / "results"
SCENES = ["look_anchor", "slapstick_moving_piano", "radio_drama"]
LATENT = "h3_probe_t2v_fasth3_8step_contract_savelat_{kind}_look_anchor__{arm}_00001_.latent"


def arm_of(clip: str) -> str:
    m = re.search(r"__(overlay_all|g\d+)_", clip)
    return m.group(1) if m else clip


def table() -> dict:
    out = {}
    for scene in SCENES:
        res = json.loads((RESULTS / f"2026-09-29_dial_{scene}_resolution.json").read_text())["rows"]
        tone = json.loads((RESULTS / f"2026-09-29_dial_{scene}_tone.json").read_text())["rows"]
        by = {}
        for r in res:
            by.setdefault(arm_of(r["clip"]), {}).update(hf=r["hf"], moved_share=r["moved_share"])
        for r in tone:
            by.setdefault(arm_of(r["clip"]), {}).update(detail=r["detail"], chroma=r["chroma"])
        out[scene] = by
    return out


def equalities(latents: Path) -> dict:
    res = {}
    for kind in ("video", "audio"):
        t = {arm: load_file(str(latents / LATENT.format(kind=kind, arm=arm)))["latent_tensor"]
             for arm in ("g000", "fasth3_nogates", "fasth3_rerun", "overlay_all", "g075")}
        res[kind] = {
            "shape": list(t["g000"].shape),
            "g000_equals_nogates": torch.equal(t["g000"], t["fasth3_nogates"]),
            "overlay_all_equals_fasth3_rerun": torch.equal(t["overlay_all"], t["fasth3_rerun"]),
            "control_g000_equals_fasth3_rerun": torch.equal(t["g000"], t["fasth3_rerun"]),
            "control_g075_equals_overlay_all": torch.equal(t["g075"], t["overlay_all"]),
        }
    return res


def divergence() -> dict:
    d = json.loads((RESULTS / "2026-09-29_dial_look_anchor_divergence.json").read_text())
    return {arm_of(k): statistics.fmean(v["rel_l2"]) for k, v in d["latents"].items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args()
    rec = {"table": table(), "equalities": equalities(args.latents), "mean_rel_l2_to_alpha1_look_anchor": divergence()}
    print(json.dumps(rec, indent=1))
    if args.record:
        args.record.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
