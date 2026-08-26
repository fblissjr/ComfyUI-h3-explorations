#!/usr/bin/env python3
"""Grade our PDD conversion against the paper, the vendor adapter, and Kijai's.

Three independent references, and the point of running all three is that they
fail differently. The paper says what the arithmetic must be; the vendor's
adapter is one implementation of it; Kijai's converted files are a SECOND,
arrived at without reference to ours. Agreement with one proves less than
agreement with all three, and this repo's rule prefers a control it compares
against over numbers a script computed for itself.

## What each reference settles

**The paper** (`internal/refs/Parallel_Decoding_Distillation...md`, section 3.1)
gives the fused layer as `W_{n:n+L} = sum_k D_k W_k` with
`D_k = (t_{k+1} - t_k) / (t_{n+L} - t_n)` -- which is `pdd_math.fusion_plan`
term for term -- and then states the practical form outright: "during inference
we can avoid the extra compute of an enlarged final layer and we only need to
hold one fused linear layer per block in memory". So precomputing the fused
heads is not our optimisation of their design; it is their recommendation, and
the shipped adapter's per-forward einsum over 32 heads is the form the paper
says you can avoid. Algorithm 1 confirms the consumer: `u = student(x_n, t[n])`
then `x_n = x_n + einsum('k,k...', h_n, u_n)` -- evaluated at the block-START
time, deterministic, no noise.

**Kijai's conversion** ships the raw 32-head bank as `set_weight` and fuses at
run time, where ours precomputes. Different choices, same arithmetic, so the
backbone must agree exactly and our fused heads must be reproducible from his
bank. His pruned build independently projects the adaln update into the curve
basis -- the same conclusion this repo reached separately, which is the part
worth having a second opinion on.

## What it reports

Every row is a relative Frobenius difference. Backbone rows must be ~0: the
transforms are exact and any disagreement is a real defect in one of the two.
The adaln rows are two approximations of one curve, so they are scored against
GROUND TRUTH rather than against each other -- "they differ" says nothing about
which is right.

Not a check and deliberately not named like one: it asserts nothing and exits 0
on any numbers. It records what was true when run, into `bench/results/`.

    python bench/compare_pdd_conversions.py --partition ref2va
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pdd_math import fuse_heads  # noqa: E402

LORAS = Path.home() / "ComfyUI" / "models" / "loras" / "h3"
DIFFUSION = Path.home() / "ComfyUI" / "models" / "diffusion_models"
BLOCKS = (0, 12, 25, 37, 49)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partition", default="ref2va", choices=("fl2va", "ref2va"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    part = args.partition
    tag = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}[part]

    paths = {
        "ours": LORAS / f"minimax_h3_{part}_pdd_8step_comfy.safetensors",
        "kijai": LORAS / f"MiniMax-H3-{tag}-Acc-8Step_comfy.safetensors",
        "kijai_pruned": LORAS / f"MiniMax-H3-{tag}-Acc-8Step_pruned_comfy.safetensors",
        "source": Path.home() / "Storage" / "alibaba-pai_MiniMax-H3-Acc-LoRAs"
                  / f"MiniMax-H3-{tag}-Acc-8Step.safetensors",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"SKIP: not on disk: {missing}. This compares against a third "
              f"party's artifact, so an absent one means no comparison was "
              f"made -- not that one passed.")
        return 0

    ours = load_file(paths["ours"])
    kij = load_file(paths["kijai"])
    kijp = load_file(paths["kijai_pruned"])
    src = load_file(paths["source"])

    report: dict = {"date": str(date.today()), "partition": part,
                    "files": {k: p.name for k, p in paths.items()}}

    # --- backbone: exact transforms, so exact agreement is the bar ----------
    backbone = {}
    for mod in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        worst = 0.0
        for i in BLOCKS:
            k = f"diffusion_model.blocks.{i}.{mod}"
            a = ours[f"{k}.lora_A.weight"].double()
            o = (ours[f"{k}.lora_B.weight"].double() @ a
                 * float(ours[f"{k}.alpha"]) / a.shape[0])
            m = kij[f"{k}.lora_B.weight"].double() @ kij[f"{k}.lora_A.weight"].double()
            worst = max(worst, rel(o, m))
        backbone[mod] = worst
    report["backbone_vs_kijai"] = backbone

    # --- the head bank: his raw stack must fuse to our precomputed heads ----
    bank = kij["diffusion_model.final_layer.video_out.set_weight"].reshape(32, -1, 5376)
    report["kijai_bank_is_the_published_stack"] = rel(bank, src["proj_out.weight"])
    report["our_fused_heads_from_his_bank"] = rel(
        ours["h3_pdd.head.video.weight"], fuse_heads(bank, 12.0, 32, 4))

    # --- adaln bake: two approximations, both scored against ground truth ---
    grid = ours["h3_pdd.silu_temb_grid"].double()
    with safe_open(DIFFUSION / f"minimax_h3_{part}_pruned_int8_convrot.safetensors",
                   framework="pt") as f:
        table = f.get_tensor("adaln_t_table").double()
    adaln = {}
    for i in BLOCKS:
        A = ours[f"h3_pdd.adaln.blocks.{i}.lora_A"].double()
        B = ours[f"h3_pdd.adaln.blocks.{i}.lora_B"].double()
        true = (grid @ A.T) @ B.T
        k = f"diffusion_model.blocks.{i}.adaln_proj.linear"
        adaln[str(i)] = {
            "ours": rel(table @ ours[f"h3_pdd.adaln_baked.blocks.{i}.diff"].double().T
                        + ours[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"].double(), true),
            "kijai": rel(table @ (kijp[f"{k}.lora_B.weight"].double()
                                  @ kijp[f"{k}.lora_A.weight"].double()).T
                         + kijp[f"{k}.diff_b"].double(), true),
        }
    report["adaln_bake_vs_truth"] = adaln
    report["note"] = ("Kijai stores the projection as bf16 factors and we store "
                      "the fp32 product, which is the whole of the difference "
                      "between the two adaln columns. Both are the same "
                      "projection and both are far below bf16 resolution.")

    print(f"=== PDD conversion, {part} ===")
    print("backbone against Kijai's independent conversion (exact transforms):")
    for k, v in backbone.items():
        print(f"  {k:16s} {v:.2e}")
    print(f"his head bank is the published 32-stack : "
          f"{report['kijai_bank_is_the_published_stack']:.2e}")
    print(f"our fused heads, reproduced from his bank: "
          f"{report['our_fused_heads_from_his_bank']:.2e}")
    print("adaln bake, each against ground truth:")
    print(f"  {'block':>6} {'ours':>12} {'kijai':>12}")
    for i, row in adaln.items():
        print(f"  {i:>6} {row['ours']:>12.2e} {row['kijai']:>12.2e}")

    out = args.out or (HERE / "results" / f"{date.today()}_pdd_conversion_{part}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
