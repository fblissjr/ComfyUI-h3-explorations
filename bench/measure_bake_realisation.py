"""Does an offline bake DELIVER the LoRA, or does it win on distance by
discarding it?

**Why this exists as its own script.** `bench/measure_merge_rounding_regimes.py`
finds that quantising `W_release + d` once lands at the base checkpoint's own
error -- `e_baked ~= e_shipped` -- which is the whole argument for baking. That
number alone is not enough, and this repo has already been caught by exactly
this shape: on 2026-08-31 a stored-weight distance ranked round-to-nearest above
stochastic rounding, and the ranking reversed once somebody asked whether the
winning arm had applied the update at all (`CHANGELOG.md` 0.99.7, `b653466`).
`e_baked ~= e_shipped` is consistent with two very different worlds:

    the bake rounds to the target and the delta survives   -> the lever is real
    the bake rounds back onto the base and the delta dies  -> the lever is a
                                                              distance artifact

`realised_along_d` separates them, and it is the statistic that did the
reversing:

    realised_along_d(after, before) = <after - before, d> / <d, d>

with `before` being the arm's own no-LoRA baseline -- `Q(W_q)` for a merge,
`Q_rtn(W_release)` for a bake -- so each arm is asked how far IT moved along
`d`, rather than all of them being compared against one arm's baseline.

**The mechanism the numbers should show, stated first so a surprise is
visible.** A merge starts from `W_q`, which is already ON the int8 grid, so a
delta below one quantisation step rounds straight back to the same codes and is
discarded. A bake starts from `W_release`, which is OFF the grid, so the delta
shifts where the rounding lands and survives in the codes. If that is right the
bake realises ~1.0 everywhere, including on the modules where RTN merging
realises almost nothing.

Three arms, all 200 int8 modules, at strength 1.0:

    merge_rtn     Q_rtn(W_q + d)      against Q_rtn baseline W_q
    merge_stoch   Q_seed(W_q + d)     against W_q, shipped per-key seed
    bake_rtn      Q_rtn(W_ref + d)    against Q_rtn(W_ref)

    python bench/measure_bake_realisation.py --lora <lora> --base <base> \
        --reference <release>/FL2VA/transformer \
        --out bench/results/2026-08-31_bake_realisation.json

**Stored weights only.** Nothing here was rendered, and `int8_convrot` is W8A8
so the activation rounding is untouched (`docs/open_experiments.md` #23).
"""

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_checkpoint_delta import header  # noqa: E402
from analyze_quant_delta import (  # noqa: E402
    Reference, hf_to_comfy, head_dim, marker, weight_in_compute_space)

sys.path.insert(0, str(_HERE.parents[2]))
import comfy.utils  # noqa: E402
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_convrot_weight, quantize_int8_convrot_weight)

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def quant(w: np.ndarray, gs: int, seed: int | None) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    q, s = quantize_int8_convrot_weight(t, gs, stochastic_rounding=seed)
    return dequantize_int8_convrot_weight(q, s, gs).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="all")
    args = ap.parse_args()

    hdr, off = header(args.base)
    ref = Reference(args.reference)
    hd = head_dim(hdr)
    n_blocks = 1 + max(int(k.split(".")[1]) for k in hdr
                       if k.startswith("blocks.") and k.endswith(".comfy_quant"))
    blocks = (list(range(n_blocks)) if args.blocks == "all"
              else [int(x) for x in args.blocks.split(",")])

    rows = []
    with safe_open(args.lora, "pt") as f:
        for blk in blocks:
            for kind in KINDS:
                mod = f"blocks.{blk}.{kind}"
                gs = int((marker(args.base, hdr, off, mod) or {})
                         ["convrot_groupsize"])
                w_q = weight_in_compute_space(args.base, hdr, off, mod)
                w_ref = hf_to_comfy(mod + ".weight",
                                   ref.get(mod + ".weight"), hd)
                a = f.get_tensor(f"diffusion_model.{mod}.lora_A.weight").float()
                b = f.get_tensor(f"diffusion_model.{mod}.lora_B.weight").float()
                al = float(f.get_tensor(f"diffusion_model.{mod}.alpha").item())
                d = ((al / a.shape[0]) * (b @ a)).numpy()
                seed = int(comfy.utils.string_to_seed(
                    f"diffusion_model.{mod}.weight"))

                d64 = d.astype(np.float64)
                dd = float((d64 * d64).sum())

                def along(after, before):
                    return float(((after - before).astype(np.float64) * d64).sum() / dd)

                rows.append({
                    "block": blk, "kind": kind, "groupsize": gs, "seed": seed,
                    "pdd_rel": float(np.linalg.norm(d64)
                                     / np.linalg.norm(w_ref.astype(np.float64))),
                    "merge_rtn": along(quant(w_q + d, gs, None), w_q),
                    "merge_stoch": along(quant(w_q + d, gs, seed), w_q),
                    "bake_rtn": along(quant(w_ref + d, gs, None),
                                      quant(w_ref, gs, None)),
                })
                r = rows[-1]
                print(f"  {mod:28s} pdd_rel {r['pdd_rel']:.5f}  "
                      f"merge_rtn {r['merge_rtn']:7.4f}  "
                      f"merge_stoch {r['merge_stoch']:7.4f}  "
                      f"bake {r['bake_rtn']:7.4f}", flush=True)
                del w_q, w_ref, d, d64

    # The producer asserts its own shape, per CLAUDE.md: an output whose shape
    # is the shape of what survived is indistinguishable from a complete one,
    # and this lane has already shipped a capture that dropped a whole kind.
    expected = len(blocks) * len(KINDS)
    kinds_seen = sorted({r["kind"] for r in rows})
    shape_ok = (len(rows) == expected and kinds_seen == sorted(KINDS)
                and len({r["block"] for r in rows}) == len(blocks))

    def stat(arm):
        v = np.asarray([r[arm] for r in rows])
        return {"mean": float(v.mean()), "min": float(v.min()),
                "max": float(v.max()), "median": float(np.median(v)),
                "below_0_9": int((v < 0.9).sum())}

    summary = {arm: stat(arm) for arm in ("merge_rtn", "merge_stoch", "bake_rtn")}

    out = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/measure_bake_realisation.py",
        "question": ("whether an offline bake delivers the LoRA update or only "
                     "appears to, given that its stored-weight distance is "
                     "indistinguishable from the base checkpoint's own"),
        "statistic": "<Q(W + d) - Q(W), d> / <d, d>, each arm against its own "
                     "no-LoRA baseline",
        "checkpoint": Path(args.base).name,
        "lora": Path(args.lora).name,
        "reference": "/".join(args.reference.parts[-3:]),
        "shape_check": {"expected": expected, "got": len(rows),
                        "kinds": kinds_seen, "ok": shape_ok},
        "is_not": ("a runtime, activation or output measurement. Stored "
                   "weights only. Nothing here was rendered."),
        "summary": summary,
        "modules": rows,
        "reproduce": ("python bench/measure_bake_realisation.py --lora <lora> "
                      "--base <base> --reference <release>/FL2VA/transformer "
                      "--out <out>"),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    if not shape_ok:
        raise SystemExit(f"shape check FAILED: {out['shape_check']}")
    print(f"\n{len(rows)} modules -> {args.out}")
    for arm, s in summary.items():
        print(f"  {arm:12s} mean {s['mean']:.4f}  min {s['min']:.4f}  "
              f"max {s['max']:.4f}  below 0.9: {s['below_0_9']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
